// The DSpark drafter of Inkling 2 Mini for the libllama engine (llama_spec.h,
// --llama-dflash-gguf with a drafter whose GGUF says dflash.has_markov_context),
// on libllama alone. The reference is the vLLM-XPU Inkling port
// (inkling_xpu/dspark.py and vllm_model.py at 6b71b909, `code`); the drafter
// graph is llama.cpp's port of it (#32218 as contrib/llama.cpp 0214-0216, with
// 0218's chain length), and the cycle around it is arcint's speculative loop,
// the one MTP and DFlash2 run (backend_llama.cpp generate_spec, verify_walk.h).
// The design and every deviation from the reference: contrib/llama.cpp/README.md,
// section "The DSpark driver".
//
// The drafter: three layers of their own (two of them windowed to 1,024
// positions, the middle one global), a K/V cache of context rows made from four
// taps of the target (0214: the inputs of three layer visits and the final
// norm's output), and a recurrent Markov head over the block:
// - after every target decode (prompt chunks and verifies alike) the taps are
//   concatenated a token and decoded into the draft context as an embedding
//   batch at the tokens' positions: each drafter layer mixes them into its own
//   context K/V, and the graph returns f_rnn, the Markov head's context row of
//   each position (0215, "KV injection"); the rows of the last block + 1
//   positions go into the lane's ring (exec/dspark_ring.h);
// - a draft at pos0 is one drafter decode of the block [id_last, mask x 7] at
//   pos0..pos0+7 (the trained block, as the reference's _block), the anchor row
//   carrying the ring's f_rnn of pos0 - 1 as its state embedding; the graph runs
//   the Markov chain over the first n_draft rows (0218) and returns the
//   chained, softcapped logits; the drafts are their argmax (dspark_pick);
// - the block's K/V then leave the draft cache, so the verify's features land
//   there; after the verify walk both contexts and the ring are cut at the
//   accepted end.
// The target's recurrent rollback (n_rs_seq = --llama-dflash, 0217 for Inkling's
// conv windows) and arcint's verify walk are the MTP path's.
#include "exec/llama_spec.h"

#ifdef ARCINT_LLAMA

#include <llama-ext.h>   // staging API of the pinned llama.cpp: layer inputs, nextn rows, DFlash metadata, 0218
#include <llama.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "exec/dspark_ring.h"
#include "exec/spec_state.h"
#include "util/log.h"

namespace lgc {

bool llama_model_is_dspark_markov(const llama_model* model) {
    char v[16] = {};
    return model != nullptr && llama_model_meta_val_str(model, "dflash.has_markov_context", v, sizeof(v)) >= 0 &&
           std::strcmp(v, "true") == 0;
}

namespace {

ggml_type kv_ggml(const std::string& t) {
    return t == "q8_0" ? GGML_TYPE_Q8_0 : t == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
}

class LlamaDspark final : public LlamaSpec {
public:
    LlamaDspark(llama_model* model, llama_context* ctx_tgt, int n_draft, int n_seq, int n_ubatch, int threads,
                double min_p, bool kv_unified, const std::string& kv_k, const std::string& kv_v, int shortlist,
                const std::string& probe_path, std::string& err)
        : ctx_tgt_(ctx_tgt), shortlist_(shortlist) {
        char meta[64] = {};
        llama_model_meta_val_str(model, "general.architecture", meta, sizeof(meta));
        if (std::string(meta) != "dflash") {
            err = log::format("the drafter is a '%s' model, not a DFlash-family drafter ('dflash')", meta);
            return;
        }
        if (!llama_model_is_dspark_markov(model)) {
            err = "the drafter has no Markov context (dflash.has_markov_context): not an Inkling 2 DSpark drafter";
            return;
        }
        // the Inkling head drafts from the mask rows: its anchor row carries f_rnn
        // (0215 asserts it), and it has no confidence head to stop on
        if (llama_model_meta_val_str(model, "dflash.sample_from_anchor", meta, sizeof(meta)) < 0 ||
            std::strcmp(meta, "false") != 0) {
            err = "the DSpark drafter does not say dflash.sample_from_anchor = false (the Inkling layout)";
            return;
        }
        if (min_p > 0.0) {
            // llama.cpp refuses it too: "DSpark draft has no confidence head: please set --spec-draft-p-min 0"
            err = "--llama-dflash-min-p: the DSpark drafter has no confidence head to stop on; leave it at 0";
            return;
        }
        const llama_model* tgt = llama_get_model(ctx_tgt);
        char               tarch[64] = {};
        llama_model_meta_val_str(tgt, "general.architecture", tarch, sizeof(tarch));
        const std::string refusal = dspark_target_refusal(tarch);
        if (!refusal.empty()) {
            err = refusal;
            return;
        }
        const int32_t*     ids = llama_model_target_layer_ids(model);
        const uint32_t     n_ids = llama_model_target_layer_ids_n(model);
        if (ids == nullptr || n_ids == 0) {
            err = "the drafter names no target taps (dflash.target_layers)";
            return;
        }
        n_embd_tgt_ = llama_model_n_embd(tgt);
        n_embd_dec_ = llama_model_n_embd(model);
        // the Markov context row is n_embd wide in both (0216's check)
        if (n_embd_dec_ != n_embd_tgt_) {
            err = log::format("the drafter's width %d does not match the model's %d", n_embd_dec_, n_embd_tgt_);
            return;
        }
        n_vocab_ = llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (n_vocab_ != llama_vocab_n_tokens(llama_model_get_vocab(tgt))) {
            err = log::format("the drafter's vocabulary (%d) is not the model's (%d)", n_vocab_,
                              llama_vocab_n_tokens(llama_model_get_vocab(tgt)));
            return;
        }
        // the taps are layer VISITS of the target (0214: Inkling's loop visits its
        // core blocks three times), the last one the final norm's output (visit n_layer)
        const int n_visits = llama_model_n_layer(tgt);
        for (uint32_t k = 0; k < n_ids; ++k) {
            const int32_t id = ids[k];
            if (id < 0 || id > n_visits) {
                err = log::format("the drafter's tap %d is not a layer visit of the model (0-%d)", id, n_visits);
                return;
            }
            taps_.push_back(static_cast<uint32_t>(id));
        }
        n_embd_enc_ = static_cast<int>(n_ids) * n_embd_tgt_;

        block_ = 0;
        if (llama_model_meta_val_str(model, "dflash.block_size", meta, sizeof(meta)) >= 0) block_ = std::atoi(meta);
        if (block_ < 2) {
            err = log::format("the drafter's block size %d drafts nothing", block_);
            return;
        }
        // the reference's width is 1..7 with a block of 8 (dspark.py: "between one and
        // seven proposals"; the mask rows of the block)
        n_draft_ = std::min(n_draft, block_ - 1);
        if (n_draft_ < n_draft)
            log::warn("dspark", "--llama-dflash %d exceeds the drafter's %d draft rows (block %d): %d drafts", n_draft,
                      block_ - 1, block_, n_draft_);
        n_block_ = dspark_block_rows(block_, n_draft_);
        bool causal = false;
        if (llama_model_meta_val_str(model, "dflash.attention.causal", meta, sizeof(meta)) >= 0)
            causal = std::strcmp(meta, "true") == 0;
        // the block's mask rows are mask_embd (0215): any token id stands in their place
        mask_ = llama_vocab_mask(llama_model_get_vocab(model));
        if (mask_ == LLAMA_TOKEN_NULL) mask_ = 0;
        n_swa_ = std::max(0, llama_model_n_swa(model));

        // the draft context, as llama.cpp makes it for draft-dspark
        // (common/speculative.cpp, common_speculative_init_result and
        // common_speculative_context_params_dft): the target's n_ctx and stream
        // layout (the global drafter layer attends the whole context, as the
        // reference sizes it: max_context), no recurrent snapshots, ctx_other =
        // the target (embeddings and LM head), f16 K/V (the reference's serving
        // type, dspark.py with INKLING2_DSPARK_BLOCK), the windowed layers at
        // window size (swa_full false), a block's outputs a lane
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx                 = llama_n_ctx(ctx_tgt);
        cp.kv_unified            = kv_unified;
        cp.swa_full              = false;
        const int block_rows     = n_seq * n_block_;
        cp.n_ubatch              = static_cast<uint32_t>(std::max(n_ubatch, block_rows));
        cp.n_batch               = cp.n_ubatch;
        cp.n_seq_max             = static_cast<uint32_t>(n_seq);
        cp.n_rs_seq              = 0;
        cp.n_outputs_max         = static_cast<uint32_t>(block_rows);
        cp.n_outputs_max_per_seq = static_cast<uint32_t>(n_block_);
        cp.ctx_other             = ctx_tgt;
        // --llama-dflash-kv (f16 as trained by default); a quantized cache asks for flash attention, so a
        // context that cannot take the type refuses here instead of serving another
        const DsparkKvPlan kvp   = dspark_kv_plan(kv_k, kv_v);
        cp.flash_attn_type       = kvp.flash_attn_on ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.type_k                = kv_ggml(kvp.type_k);
        cp.type_v                = kv_ggml(kvp.type_v);
        cp.n_threads             = threads;
        cp.n_threads_batch       = threads;
        ctx_dft_                 = llama_init_from_model(model, cp);
        if (ctx_dft_ == nullptr) {
            err = log::format("llama.cpp could not create the DSpark draft context (K/V %s:%s)", kvp.type_k.c_str(),
                              kvp.type_v.c_str());
            return;
        }
        n_ubatch_dft_ = static_cast<int>(llama_n_ubatch(ctx_dft_));
        llama_set_causal_attn(ctx_dft_, causal);
        // f_rnn comes back as the nextn row of every injected position (0216)
        llama_set_embeddings_nextn(ctx_dft_, true, /*masked=*/false);
        // 0218: the Markov chain over the first n_draft rows of the trained block
#if ARCINT_DSPARK_RED == 4
        llama_set_dspark_chain(ctx_dft_, 0);   // red case: the n_max + 1-row block, its chain over every row
#else
        llama_set_dspark_chain(ctx_dft_, static_cast<uint32_t>(n_draft_));
#endif
        // 0241: the chain over each row's base top-K, the drafted ids out (0: the full chain, the logits out)
        llama_set_dspark_shortlist(ctx_dft_, static_cast<uint32_t>(shortlist_));
        // 0243: --llama-dflash-shortlist-probe: a line a verified draft; with the shortlist, the graph's probe too
        if (!probe_path.empty()) {
            probe_ = std::fopen(probe_path.c_str(), "w");
            if (probe_ == nullptr) {
                err = log::format("--llama-dflash-shortlist-probe %s: cannot be written", probe_path.c_str());
                return;
            }
            if (shortlist_ > 0) llama_set_dspark_probe(ctx_dft_, true);
            log::warn("dspark", "--llama-dflash-shortlist-probe %s: the probe is on (%s); a measurement run, not a timing "
                                "run", probe_path.c_str(), shortlist_ > 0 ? "the full chain's pick at every shortlist step: "
                                "its GEMV and ARGMAX over the vocabulary are back in every propose" : "no shortlist: the drafts "
                                "and kept counts only");
        }
        for (uint32_t id : taps_) llama_set_embeddings_layer_inp(ctx_tgt, id, true);
        bt_ = llama_batch_ext_init(ctx_tgt);
        bd_ = llama_batch_ext_init(ctx_dft_);
        if (bt_ == nullptr || bd_ == nullptr) {
            err = "the batches for the target's taps or the drafter could not be allocated";
            return;
        }
        lanes_.resize(static_cast<size_t>(n_seq));
        probe_pending_.resize(static_cast<size_t>(n_seq));
        for (Lane& ln : lanes_) ln.ring.init(static_cast<size_t>(block_) + 1, static_cast<size_t>(n_embd_dec_));
        state_.assign(static_cast<size_t>(block_rows) * static_cast<size_t>(n_embd_dec_), 0.0f);

        std::string ls;
        for (uint32_t id : taps_) ls += (ls.empty() ? "" : ",") + std::to_string(id);
        log::info("dspark", "DSpark drafter (Inkling 2, Markov context): %d drafts a block of %d (the Markov chain over "
                            "%d rows), taps at layer visits %s, window %d, %s attention, %s:%s K/V; draft context of n_ctx %u, "
                            "ubatch %d; the verify is %d rows; %s",
                  n_draft_, n_block_, n_draft_, ls.c_str(), n_swa_, causal ? "causal" : "non-causal", kvp.type_k.c_str(),
                  kvp.type_v.c_str(), llama_n_ctx(ctx_dft_), n_ubatch_dft_, n_draft_ + 1,
                  shortlist_ > 0 ? log::format("shortlist %d (the chain over each row's base top-%d, the ids out)", shortlist_, shortlist_).c_str()
                                 : "the full chain (no shortlist)");
#if ARCINT_DSPARK_RED != 0
        log::warn("dspark", "THIS BUILD IS THE DSPARK RED CASE %d (ARCINT_DSPARK_RED): not for serving", ARCINT_DSPARK_RED);
#endif
    }

    ~LlamaDspark() override {
        if (ctx_tgt_ != nullptr)
            for (uint32_t id : taps_) llama_set_embeddings_layer_inp(ctx_tgt_, id, false);
        if (bt_ != nullptr) llama_batch_ext_free(bt_);
        if (bd_ != nullptr) llama_batch_ext_free(bd_);
        if (ctx_dft_ != nullptr) llama_free(ctx_dft_);
        if (probe_ != nullptr) {
            std::fclose(probe_);
            log::info("dspark", "--llama-dflash-shortlist-probe: %llu lines; %llu drafts without a probe of the graph, "
                                "%llu drafts not verified (no line), %llu lines not written",
                      static_cast<unsigned long long>(probe_lines_), static_cast<unsigned long long>(probe_missing_),
                      static_cast<unsigned long long>(probe_unclosed_), static_cast<unsigned long long>(probe_write_fail_));
        }
    }

    bool ok() const { return bd_ != nullptr; }
    int  n_draft() const { return n_draft_; }

    llama_context* draft_context() const override { return ctx_dft_; }
    size_t draft_head_device_bytes() const override { return 0; }   // the target's LM head, through ctx_other

    int decode(const int* toks, size_t n, size_t pos, int seq, bool all_logits) override {
        const SpecPart p{ seq, toks, n, pos, all_logits };
        return decode_multi(&p, 1);
    }

    // one target decode of every part, then each part's rows into the drafter
    // (the reference's store() after every target forward)
    int decode_multi(const SpecPart* parts, size_t n_parts) override {
        llama_batch_ext_clear(bt_);
        for (size_t k = 0; k < n_parts; ++k)
            for (size_t i = 0; i < parts[k].n; ++i) {
                const int32_t idx = llama_batch_ext_add_token(bt_, parts[k].seq, parts[k].toks[i]);
                if (idx < 0) return -1;
                const llama_pos p = static_cast<llama_pos>(parts[k].pos + i);
                llama_batch_ext_set_pos(bt_, idx, &p);
                if (parts[k].all_logits) llama_batch_ext_set_output_logits(bt_, idx, true);
            }
        const int r = llama_process(ctx_tgt_, LLAMA_PROCESS_TYPE_DECODE, bt_);
        if (r != 0) return r;
        size_t off = 0;
        for (size_t k = 0; k < n_parts; ++k) {
            const int ri = inject(parts[k], off);
            if (ri != 0) return ri;
            off += parts[k].n;
        }
        return 0;
    }

    std::vector<int> draft(int seq, int id_last, size_t pos0, int n_max) override {
        std::vector<int> out;
        DraftPart        p{ seq, id_last, pos0, n_max, &out };
        draft_multi(&p, 1);
        return out;
    }

    // one drafter decode of every lane's block [id_last, mask x (block - 1)] at
    // pos0..pos0+block-1, the anchor's state embedding the ring's f_rnn of
    // pos0 - 1 (a lane whose ring lacks it drafts nothing this step, as
    // llama.cpp's port); the drafts the argmax of the first rows, truncated to
    // the lane's budget; then the blocks' K/V out of the draft cache
    void draft_multi(DraftPart* parts, size_t n_parts) override {
        llama_batch_ext_clear(bd_);
        std::vector<int32_t> beg(n_parts, -1);
        const size_t         w = static_cast<size_t>(n_embd_dec_);
        size_t               used = 0;   // rows of state_ handed to the batch
        for (size_t k = 0; k < n_parts; ++k) {
            DraftPart& p = parts[k];
            p.out->clear();
            if (dspark_step_drafts(n_draft_, p.n_max, block_) <= 0) continue;
            Lane&        ln  = lanes_[static_cast<size_t>(p.seq)];
            const float* ctx = ln.ring.get(dspark_anchor_pos(static_cast<int64_t>(p.pos0)));
            if (ctx == nullptr) {
                if (ln.no_context++ == 0)
                    log::verbose("dspark", "lane %d: no Markov context row for position %zu: no draft this step "
                                           "(counted; logged once a lane)", p.seq, p.pos0 > 0 ? p.pos0 - 1 : 0);
                continue;
            }
            check_window(p.seq, p.pos0);
            for (int i = 0; i < n_block_; ++i) {
                const int32_t idx = llama_batch_ext_add_token(bd_, p.seq, i == 0 ? p.id_last : mask_);
                if (idx < 0) return;
                if (i == 0) beg[k] = idx;
                const llama_pos pos = static_cast<llama_pos>(p.pos0 + static_cast<size_t>(i));
                llama_batch_ext_set_pos(bd_, idx, &pos);
                // the shortlist's graph returns ids, not logits: its rows ask for none (no 6.4 MB read)
                llama_batch_ext_set_output_logits(bd_, idx, shortlist_ == 0);
                // the state embedding: f_rnn on the anchor row, zero on the mask rows (0216)
                float* st = state_.data() + used * w;
                if (i == 0) std::memcpy(st, ctx, w * sizeof(float));
                else std::fill(st, st + w, 0.0f);
                ++used;
                if (!llama_batch_ext_set_embd_state(bd_, idx, llama_embd{ st, 1, w })) {
                    log::warn("dspark", "the draft context takes no state embedding: no drafts this step");
                    return;
                }
            }
        }
        bool any = false;
        for (int32_t b : beg) any |= b >= 0;
        if (!any) return;
        const int rc = llama_process(ctx_dft_, LLAMA_PROCESS_TYPE_DECODE, bd_);
        if (rc != 0) log::warn("dspark", "the drafter's decode failed (%d): no drafts this step", rc);
        const size_t vocab = static_cast<size_t>(n_vocab_);
        for (size_t k = 0; k < n_parts; ++k) {
            if (beg[k] < 0) continue;
            DraftPart& p = parts[k];
            if (rc == 0 && shortlist_ > 0) {
                ids_.resize(static_cast<size_t>(n_block_));
                const int n = llama_get_dspark_ids(ctx_dft_, p.seq, ids_.data(), n_block_);
                if (n < 0 && no_ids_++ == 0)
                    log::warn("dspark", "lane %d: --llama-dflash-shortlist %d: the drafter's graph returned no ids (the "
                                        "shortlist path declined this drafter or block): no drafts (counted; logged once)",
                              p.seq, shortlist_);
                *p.out = dspark_ids_take(ids_.data(), n, dspark_step_drafts(n_draft_, p.n_max, block_), n_vocab_);
            } else if (rc == 0) {
                logits_.resize(static_cast<size_t>(n_block_) * vocab);
                bool ok = true;
                for (int i = 0; i < n_block_ && ok; ++i) {
                    const float* l = llama_get_logits_ith(ctx_dft_, beg[k] + i);
                    ok             = l != nullptr;
                    if (ok) std::copy(l, l + vocab, logits_.begin() + static_cast<long>(static_cast<size_t>(i) * vocab));
                }
                if (ok)
                    *p.out = dspark_pick(logits_.data(), static_cast<size_t>(n_block_), vocab,
                                         dspark_step_drafts(n_draft_, p.n_max, block_));
            }
            if (probe_ != nullptr) probe_hold(p);
            // the block leaves; the verify's features land at pos0.. next
            llama_memory_seq_rm(llama_get_memory(ctx_dft_), p.seq, static_cast<llama_pos>(p.pos0), -1);
        }
    }

    void reset(int /*seq*/) override {}   // the ring is keyed by position: nothing carried across a cut

    // the cut is seq_rm's; the probe's line is written here, with the verify's kept count
    void accept(int seq, int n_accepted) override {
        if (probe_ == nullptr || seq < 0 || static_cast<size_t>(seq) >= probe_pending_.size()) return;
        DsparkProbePending& pp = probe_pending_[static_cast<size_t>(seq)];
        if (!pp.set) return;
        if (dspark_probe_write(probe_, dspark_probe_line(seq, pp, n_accepted))) {
            ++probe_lines_;
        } else if (probe_write_fail_++ == 0) {
            log::warn("dspark", "lane %d: --llama-dflash-shortlist-probe: the probe log could not be written (a short "
                                "write or flush): the log is short (counted; logged once)", seq);
        }
        pp.set = false;
    }

    bool seq_rm(int seq, size_t from) override {
        llama_memory_t mt = llama_get_memory(ctx_tgt_);
        llama_memory_t md = llama_get_memory(ctx_dft_);
        Lane&          ln = lanes_[static_cast<size_t>(seq)];
        if (!llama_memory_seq_rm(mt, seq, static_cast<llama_pos>(from), -1)) {
            llama_memory_seq_rm(mt, seq, -1, -1);
            llama_memory_seq_rm(md, seq, -1, -1);
            ln.ring.clear();
            return false;
        }
        llama_memory_seq_rm(md, seq, static_cast<llama_pos>(from), -1);
        ln.ring.trim(static_cast<int64_t>(from));
        return true;
    }

    // the drafter's state at n tokens: its windowed layers' K/V (PARTIAL_ONLY:
    // the sliding-window caches; the global layer's cells below n stay in the
    // cache, as the target's do) and the ring's rows before n
    Bytes carried_state(int seq, size_t n) override {
        if (n == 0) return {};
        const size_t                 size = llama_state_seq_get_size_ext(ctx_dft_, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const std::vector<uint8_t>   ring = lanes_[static_cast<size_t>(seq)].ring.save(static_cast<int64_t>(n));
        const uint64_t               sz   = size;
        Bytes s = spec_state_alloc(SpecStateKind::DsparkWindowRing, n, 8 + size + ring.size());
        uint8_t* at = s.data() + kSpecStateHeader;
        std::memcpy(at, &sz, 8);
        if (size > 0 &&
            llama_state_seq_get_data_ext(ctx_dft_, at + 8, size, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size)
            return {};
        std::memcpy(at + 8 + size, ring.data(), ring.size());
        return s;
    }

    // after a checkpoint's restore: the window and the ring back, cut to n. A checkpoint at 0 tokens carries
    // nothing (the lane starts empty). Any other failure clears the drafter's sequence: its global layer then has
    // no cell below n for the rest of the lane's life (acceptance falls, the output stays the target's), so it is
    // warned and counted (the decode line's "dflash lane failures", cumulative), never silent
    void set_carried_state(int seq, size_t n, const Bytes& state) override {
        llama_memory_t md = llama_get_memory(ctx_dft_);
        Lane&          ln = lanes_[static_cast<size_t>(seq)];
        if (n == 0) {
            llama_memory_seq_rm(md, seq, -1, -1);
            ln.ring.clear();
            return;
        }
        SpecStateView    v;
        DsparkStateParts parts;
        bool ok = spec_state_view(state, SpecStateKind::DsparkWindowRing, n, v) && dspark_state_split(v.data, v.size, parts);
        if (ok && parts.win_size > 0)
            ok = llama_state_seq_set_data_ext(ctx_dft_, v.data + parts.win_off, parts.win_size, seq,
                                              LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == parts.win_size;
        if (ok) ok = ln.ring.load(static_cast<int64_t>(n), v.data + parts.ring_off, parts.ring_size);
        if (ln.health.on_restore(ok)) {
            llama_memory_seq_rm(md, seq, -1, -1);
            ln.ring.clear();
            log::warn("dspark", "lane %d: the drafter's state did not come back with the checkpoint at %zu tokens "
                                "(%s): its global layer has no cell below %zu for this lane until it is prefilled again; "
                                "drafts stay verified, acceptance falls (restore failures on this lane: %llu)",
                      seq, n, state.empty() ? "none saved" : "refused", n,
                      static_cast<unsigned long long>(ln.health.restore_failures));
            return;
        }
        llama_memory_seq_rm(md, seq, static_cast<llama_pos>(n), -1);
    }

    int state_failures() const override {
        uint64_t f = 0;
        for (const Lane& ln : lanes_) f += ln.health.restore_failures;
        return static_cast<int>(f);
    }

    // every prompt row goes into the drafter (its global layer attends all of
    // them, as the reference's store() takes every position); the plan only says
    // which chunks' f_rnn rows must come back to the host (the ring's)
    void prefill_plan(int seq, size_t n_end, const std::vector<size_t>& ckpts) override {
        Lane& ln = lanes_[static_cast<size_t>(seq)];
        ln.n_end = static_cast<int64_t>(n_end);
        ln.ckpts.assign(ckpts.begin(), ckpts.end());
    }

    bool set_kv_windows(const uint32_t* n_cells, int n) override {
        return llama_memory_seq_windows(llama_get_memory(ctx_dft_), n_cells, n);
    }

private:
    struct Lane {
        DsparkRing           ring;          // f_rnn of the last block + 1 positions
        int64_t              n_end = -1;    // the running prefill's end (prefill_plan); -1: none planned
        std::vector<int64_t> ckpts;         // the checkpoints that prefill saves
        uint64_t             mismatches = 0;
        uint64_t             no_context = 0;
        DsparkLaneHealth     health;        // failed checkpoint restores (review C3)
    };

    // the part's rows (from batch offset `off` of the target's decode) into the
    // drafter at their positions: the taps side by side a token, in chunks of the
    // draft ubatch (common/speculative.cpp's process()); each chunk's f_rnn rows
    // into the ring when the ring may read them
    int inject(const SpecPart& p, size_t off) {
        Lane&                     ln = lanes_[static_cast<size_t>(p.seq)];
        std::vector<const float*> src(taps_.size());
        for (size_t k = 0; k < taps_.size(); ++k) {
            src[k] = llama_get_embeddings_layer_inp(ctx_tgt_, taps_[k]);
            if (src[k] == nullptr) return -1;
        }
        const size_t ch   = static_cast<size_t>(n_ubatch_dft_);
        const size_t we   = static_cast<size_t>(n_embd_enc_);
        const size_t wt   = static_cast<size_t>(n_embd_tgt_);
        const size_t wd   = static_cast<size_t>(n_embd_dec_);
        const int64_t rows = static_cast<int64_t>(ln.ring.rows());
        for (size_t at = 0; at < p.n; at += ch) {
            const size_t  n_chunk = std::min(ch, p.n - at);
            const int64_t lo      = static_cast<int64_t>(p.pos + at);
            const int64_t hi      = lo + static_cast<int64_t>(n_chunk);
            const bool    want    = dspark_chunk_needs_rows(lo, hi, p.all_logits, ln.n_end, ln.ckpts, rows);
            feat_.resize(n_chunk * we);
            for (size_t k = 0; k < taps_.size(); ++k)
                for (size_t i = 0; i < n_chunk; ++i)
                    std::memcpy(feat_.data() + i * we + k * wt, src[k] + (off + at + i) * wt, wt * sizeof(float));
            llama_batch_ext_clear(bd_);
            for (size_t i = 0; i < n_chunk; ++i) {
                const int32_t idx = llama_batch_ext_add_embd(bd_, p.seq, llama_embd{ feat_.data() + i * we, 1, we });
                if (idx < 0) return -1;
                const llama_pos q      = static_cast<llama_pos>(lo) + static_cast<llama_pos>(i);
                const llama_pos pos[4] = { q, q, q, 0 };
                llama_batch_ext_set_pos(bd_, idx, pos);
            }
            llama_set_embeddings_nextn_skip(ctx_dft_, !want);
            const int rc = llama_process(ctx_dft_, LLAMA_PROCESS_TYPE_DECODE, bd_);
            llama_set_embeddings_nextn_skip(ctx_dft_, false);
            if (rc != 0) {
                log::error("dspark", "lane %d: injecting %zu rows at %lld into the drafter failed (%d)", p.seq, n_chunk,
                           static_cast<long long>(lo), rc);
                return rc;
            }
            if (!want) continue;
            const float* f = llama_get_embeddings_nextn(ctx_dft_);
            if (f == nullptr) {
                log::error("dspark", "lane %d: the drafter returned no Markov context rows", p.seq);
                return -1;
            }
            // the last block + 1 positions of the chunk (0216's pos_min); the earlier
            // ones a later chunk of the same part overwrites or the ring never reads
            const size_t keep0 = n_chunk > ln.ring.rows() ? n_chunk - ln.ring.rows() : 0;
            for (size_t i = keep0; i < n_chunk; ++i) ln.ring.put(lo + static_cast<int64_t>(i), f + i * wd);
        }
        if (!p.all_logits && ln.n_end >= 0 && static_cast<int64_t>(p.pos + p.n) >= ln.n_end) {
            ln.n_end = -1;   // the planned prefill is done: what follows are verifies
            ln.ckpts.clear();
        }
        return 0;
    }

    // before a draft at pos0 the drafter holds the target's rows through pos0 - 1
    // (common/speculative.cpp's begin() checks it): a drafter that ends elsewhere
    // drafts from a stale or shifted context. Counted and logged once a lane;
    // the drafts stay verified either way
    void check_window(int seq, size_t pos0) {
        const llama_pos pd = llama_memory_seq_pos_max(llama_get_memory(ctx_dft_), seq);
        if (pd == static_cast<llama_pos>(pos0) - 1) return;
        Lane& ln = lanes_[static_cast<size_t>(seq)];
        if (ln.mismatches++ == 0)
            log::warn("dspark", "lane %d: the drafter holds positions up to %d before a draft at %zu (the target's %d): "
                                "its context and the target disagree", seq, static_cast<int>(pd), pos0,
                      static_cast<int>(pos0) - 1);
    }

    llama_context*        ctx_tgt_      = nullptr;
    llama_context*        ctx_dft_      = nullptr;
    llama_batch_ext*      bt_           = nullptr;
    llama_batch_ext*      bd_           = nullptr;
    std::vector<uint32_t> taps_;     // the target's layer visits the drafter reads
    std::vector<float>    feat_;     // one injection chunk: [n, n_taps x n_embd]
    std::vector<float>    state_;    // the block rows' state embeddings, n_seq x block rows
    std::vector<float>    logits_;   // one block's logits rows
    std::vector<int32_t>  ids_;      // 0241: one block's drafted ids (the shortlist path)
    int                   shortlist_ = 0;
    uint64_t              no_ids_    = 0;   // 0241: blocks the shortlist graph returned no ids for
    // 0243: --llama-dflash-shortlist-probe
    std::FILE*                      probe_ = nullptr;
    std::vector<DsparkProbePending> probe_pending_;   // a lane's draft until its verify
    std::vector<int32_t>            probe_full_;
    std::vector<int8_t>             probe_in_;
    uint64_t                        probe_lines_ = 0, probe_missing_ = 0, probe_unclosed_ = 0, probe_write_fail_ = 0;

    // a lane's offered drafts and, with the shortlist, the graph's probe of them, held for accept()
    void probe_hold(const DraftPart& p) {
        DsparkProbePending& pp = probe_pending_[static_cast<size_t>(p.seq)];
        if (pp.set) ++probe_unclosed_;   // the last draft of this lane was never verified
        pp.set = false;
        if (p.out->empty()) return;
        pp.pos0    = static_cast<int64_t>(p.pos0);
        pp.id_last = p.id_last;
        pp.sl      = shortlist_;
        pp.d       = *p.out;
        pp.f.clear();
        pp.in.clear();
        if (shortlist_ > 0) {
            probe_full_.assign(static_cast<size_t>(n_block_), -1);
            probe_in_.assign(static_cast<size_t>(n_block_), 0);
            const int n = llama_get_dspark_probe(ctx_dft_, p.seq, probe_full_.data(), probe_in_.data(), n_block_);
            if (n < static_cast<int>(pp.d.size())) {
                if (probe_missing_++ == 0)
                    log::warn("dspark", "lane %d: --llama-dflash-shortlist-probe: the drafter's graph returned no probe for "
                                        "this draft (%d of %zu positions): no line (counted; logged once)",
                              p.seq, n, pp.d.size());
                return;
            }
            for (size_t i = 0; i < pp.d.size(); ++i) {
                pp.f.push_back(probe_full_[i]);
                pp.in.push_back(probe_in_[i]);
            }
        }
        pp.set = true;
    }
    std::vector<Lane>     lanes_;
    int                   n_draft_      = 0;
    int                   block_        = 0;
    int                   n_block_      = 0;
    int                   n_vocab_      = 0;
    int                   n_embd_tgt_   = 0;
    int                   n_embd_dec_   = 0;
    int                   n_embd_enc_   = 0;
    int                   n_ubatch_dft_ = 64;
    int                   n_swa_        = 0;
    llama_token           mask_         = 0;
};

}  // namespace

std::unique_ptr<LlamaSpec> make_llama_dspark(llama_model* model_dft, llama_context* ctx_tgt, int& n_draft, int n_seq,
                                             int n_ubatch, int threads, double min_p, bool kv_unified,
                                             const std::string& kv_k, const std::string& kv_v, int shortlist,
                                             const std::string& probe_path, std::string& err) {
    auto s = std::make_unique<LlamaDspark>(model_dft, ctx_tgt, n_draft, n_seq, n_ubatch, threads, min_p, kv_unified,
                                           kv_k, kv_v, shortlist, probe_path, err);
    if (!s->ok() || !err.empty()) return nullptr;
    n_draft = s->n_draft();
    return s;
}

}  // namespace lgc

#endif  // ARCINT_LLAMA
