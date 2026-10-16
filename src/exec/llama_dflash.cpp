// The DFlash2 drafter for the libllama engine (llama_spec.h,
// --llama-dflash-gguf), on libllama alone: llama.cpp's draft-dflash at the
// pin (common/speculative.cpp, common_speculative_impl_draft_dflash,
// :981-1429) with the server's cycle around it
// (tools/server/server-context.cpp). The common library is not linked (see
// llama_spec.cpp). docs/campaigns/dflash2-agent.md is the design.
//
// The drafter is a five-layer model of its own architecture (`dflash`,
// src/models/dflash.cpp) that keeps a K/V cache of the target's hidden
// states, a 2,048-position sliding window:
// - after every target decode (prompt chunks and verifies alike) the target's
//   inputs of the drafter's target layers (llama_get_embeddings_layer_inp,
//   five rows of n_embd a token) are concatenated per token and decoded into
//   the draft context as an embedding batch at the tokens' positions: the
//   drafter writes them into its cache as K/V ("KV injection",
//   dflash.cpp:844-916);
// - a draft is one drafter decode of [id_last, mask x N] at pos0..pos0+N,
//   non-causal over the injected context and the block (dflash.cpp:953-1045)
//   with the target's LM head; the DFlash2 selector packs each position's
//   top-k candidates and their transition scores into the nextn rows
//   (dflash.cpp:550-639), and the walk (exec/dflash_walk.h) picks the drafts;
// - the block's K/V then leave the draft cache, so the verify's features land
//   there (server-context.cpp:3444); after the verify walk both contexts are
//   cut at the accepted end (server-context.cpp:4529).
// The target's recurrent rollback (n_rs_seq) and arcint's verify walk are the
// MTP path's: nothing in them is MTP-specific.
// A context checkpoint carries the drafter's window, as the server's does
// (carried_state).
// Beyond the reference: a prefill injects only the prompt rows the drafter's
// window and the prefill's checkpoints need (kDflashPrefillRows,
// prefill_plan).
#include "exec/llama_spec.h"

#ifdef ARCINT_LLAMA

#include <llama-ext.h>   // staging API of the pinned llama.cpp: layer inputs, nextn rows, DFlash metadata
#include <llama.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "exec/dflash_walk.h"
#include "exec/spec_state.h"
#include "util/log.h"

#ifndef ARCINT_DFLASH_RED
#define ARCINT_DFLASH_RED 0
#endif

namespace lgc {
namespace {

// Prompt rows a prefill injects into the drafter: its 2,048-position window
// plus a 512-row ubatch. Beyond the reference (llama.cpp injects every
// prompt row): the drafter attends 2,048 positions, so rows further back
// reach no draft; the operator's decision, 2026-10-16
// (docs/campaigns/dflash2-agent.md, open question 5). The window before
// each context checkpoint the same prefill takes is injected too
// (prefill_plan), so a checkpoint carries the drafter state the reference
// would have saved with it.
constexpr size_t kDflashPrefillRows = 2560;

class LlamaDflash final : public LlamaSpec {
public:
    LlamaDflash(llama_model* model, llama_context* ctx_tgt, int n_draft, int n_seq, int n_ubatch, int threads,
                double min_p, bool kv_unified, std::string& err)
        : ctx_tgt_(ctx_tgt), min_p_(static_cast<float>(min_p)) {
        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        if (std::string(arch) != "dflash") {
            err = log::format("the drafter is a '%s' model, not a DFlash drafter ('dflash')", arch);
            return;
        }
        const llama_model* tgt = llama_get_model(ctx_tgt);
        top_k_ = llama_model_dflash_selector_top_k(model);
        char meta[32] = {};
        if (top_k_ <= 0 || llama_model_meta_val_str(model, "lilicorr.candidate_topk", meta, sizeof(meta)) >= 0) {
            err = "the drafter has no DFlash2 selector (a DFlash1, LiliCorr or DSpark drafter); "
                  "--llama-dflash-gguf drives DFlash2";
            return;
        }
        const int32_t* ids = llama_model_target_layer_ids(model);
        const uint32_t n_ids = llama_model_target_layer_ids_n(model);
        if (ids == nullptr || n_ids == 0) {
            err = "the drafter names no target layers (dflash.target_layers)";
            return;
        }
        n_embd_tgt_ = llama_model_n_embd(tgt);
        n_embd_dec_ = llama_model_n_embd(model);
        // the drafter's input rows are n_ids x its own n_embd
        // (llama_model_dflash::load_arch_hparams), the target's features
        // n_ids x the target's: one width
        if (n_embd_dec_ != n_embd_tgt_ || llama_model_n_embd_out(model) != n_embd_dec_) {
            err = log::format("the drafter's width %d does not match the model's %d", n_embd_dec_, n_embd_tgt_);
            return;
        }
        const int n_vocab_dft = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const int n_vocab_tgt = llama_vocab_n_tokens(llama_model_get_vocab(tgt));
        if (n_vocab_dft != n_vocab_tgt) {
            err = log::format("the drafter's vocabulary (%d) is not the model's (%d)", n_vocab_dft, n_vocab_tgt);
            return;
        }
        if (static_cast<size_t>(top_k_) + static_cast<size_t>(top_k_) * static_cast<size_t>(top_k_) >
            static_cast<size_t>(n_embd_dec_)) {
            err = log::format("the selector's %d candidates do not fit a row of %d", top_k_, n_embd_dec_);
            return;
        }
        const int n_layer_tgt = llama_model_n_layer(tgt);
        for (uint32_t k = 0; k < n_ids; ++k) {
            // the converter writes the checkpoint's ids + 1: layer input i is
            // the output of layer i - 1 (conversion/qwen.py:825)
            int32_t id = ids[k];
#if ARCINT_DFLASH_RED == 1
            id -= 1;   // red case: the checkpoint's ids without the converter's + 1 (the wrong layers)
#endif
            if (id < 0 || id > n_layer_tgt) {
                err = log::format("the drafter's target layer %d is not a layer of the model (0-%d)", id, n_layer_tgt);
                return;
            }
            layers_.push_back(static_cast<uint32_t>(id));
        }
        n_embd_enc_ = static_cast<int>(n_ids) * n_embd_tgt_;

        // the block: llama.cpp's dflash.block_size (default 16), drafts up
        // to block_size - 1 (common/speculative.cpp:1045-1097)
        int block = 16;
        if (llama_model_meta_val_str(model, "dflash.block_size", meta, sizeof(meta)) >= 0) block = std::atoi(meta);
        if (block < 2) {
            err = log::format("the drafter's block size %d drafts nothing", block);
            return;
        }
        n_draft_ = std::min(n_draft, block - 1);
        if (n_draft_ < n_draft)
            log::warn("dflash", "--llama-dflash %d exceeds the drafter's trained block of %d: %d drafts", n_draft, block,
                      n_draft_);
        bool causal = false;
        if (llama_model_meta_val_str(model, "dflash.attention.causal", meta, sizeof(meta)) >= 0)
            causal = std::strcmp(meta, "true") == 0;
        mask_ = llama_vocab_mask(llama_model_get_vocab(model));
        if (mask_ == LLAMA_TOKEN_NULL) {
            err = "the drafter's vocabulary has no mask token";
            return;
        }
        n_swa_ = std::max(0, llama_model_n_swa(model));

        // the draft context, as llama.cpp makes it (common/speculative.cpp:
        // 2572-2700): the target's n_ctx and stream layout, no recurrent
        // snapshots, ctx_other = the target (embeddings and LM head), the
        // draft's default f16 cache, outputs a block a lane. Not llama.cpp's
        // library default swa_full = true: that sizes the sliding-window
        // cache to the whole context (common's default is false)
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = llama_n_ctx(ctx_tgt);
        cp.kv_unified      = kv_unified;
        cp.swa_full        = false;
        // every lane's block fits one ubatch (the selector reads whole
        // blocks of one ubatch, dflash.cpp:560-567); injections are cut into
        // chunks of it; the batch is the ubatch. llama.cpp reserves the
        // compute buffer for a token ubatch of this size, and the dflash
        // graph computes the LM head for every row: n_ubatch x the
        // vocabulary in f32 (the target's 512: 485 MiB at 248,320 tokens).
        // llama.cpp's draft context takes --spec-draft-n-ubatch for it
        // (common/arg.cpp:4178, speculative.cpp:2609-2611)
        const int block_rows = n_seq * (n_draft_ + 1);
        cp.n_ubatch        = static_cast<uint32_t>(std::max(n_ubatch, block_rows));
        cp.n_batch         = cp.n_ubatch;
        cp.n_seq_max       = static_cast<uint32_t>(n_seq);
        cp.n_rs_seq        = 0;
        cp.n_outputs_max         = static_cast<uint32_t>(block_rows);
        cp.n_outputs_max_per_seq = 1;
        cp.ctx_other       = ctx_tgt;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.type_k          = GGML_TYPE_F16;
        cp.type_v          = GGML_TYPE_F16;
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        ctx_dft_           = llama_init_from_model(model, cp);
        if (ctx_dft_ == nullptr) {
            err = "llama.cpp could not create the DFlash draft context";
            return;
        }
        n_ubatch_dft_ = static_cast<int>(llama_n_ubatch(ctx_dft_));
        llama_set_causal_attn(ctx_dft_, causal);
        // DFlash2 reads its lattice from the nextn rows, one a token, and no
        // logits (common/speculative.cpp:1137)
        llama_set_embeddings_nextn(ctx_dft_, true, /*masked=*/false);
        for (uint32_t id : layers_) llama_set_embeddings_layer_inp(ctx_tgt, id, true);
        bt_ = llama_batch_ext_init(ctx_tgt);
        bd_ = llama_batch_ext_init(ctx_dft_);
        if (bt_ == nullptr || bd_ == nullptr) {
            err = "the batches for the target's layer inputs or the drafter could not be allocated";
            return;
        }
        lanes_.resize(static_cast<size_t>(n_seq));

        std::string ls;
        for (uint32_t id : layers_) ls += (ls.empty() ? "" : ",") + std::to_string(id);
        log::info("dflash", "DFlash2 drafter: %d drafts a block of %d, top-%d selector, target layer inputs %s, "
                            "window %d, %s attention, p_min %.2f; draft context of n_ctx %u (its cache holds the window), ubatch %d",
                  n_draft_, block, top_k_, ls.c_str(), n_swa_, causal ? "causal" : "non-causal",
                  static_cast<double>(min_p_), llama_n_ctx(ctx_dft_), n_ubatch_dft_);
#if ARCINT_DFLASH_RED != 0
        log::warn("dflash", "THIS BUILD IS THE DFLASH RED CASE %d (ARCINT_DFLASH_RED): not for serving", ARCINT_DFLASH_RED);
#endif
    }

    ~LlamaDflash() override {
        if (ctx_tgt_ != nullptr)
            for (uint32_t id : layers_) llama_set_embeddings_layer_inp(ctx_tgt_, id, false);
        if (bt_ != nullptr) llama_batch_ext_free(bt_);
        if (bd_ != nullptr) llama_batch_ext_free(bd_);
        if (ctx_dft_ != nullptr) llama_free(ctx_dft_);
    }

    bool ok() const { return bd_ != nullptr; }
    int  n_draft() const { return n_draft_; }

    int decode(const int* toks, size_t n, size_t pos, int seq, bool all_logits) override {
        const SpecPart p{ seq, toks, n, pos, all_logits };
        return decode_multi(&p, 1);
    }

    // One target decode of every part, then each part's rows into the
    // drafter (common_speculative_process after every target decode)
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

    // One drafter decode of every lane's block [id_last, mask x N] at
    // pos0..pos0+N (always N + 1 positions: one graph shape, llama.cpp's
    // n_max + 1), the walk truncated to each lane's budget, then the blocks'
    // K/V out of the draft cache
    void draft_multi(DraftPart* parts, size_t n_parts) override {
        llama_batch_ext_clear(bd_);
        std::vector<int32_t> beg(n_parts, -1);
        const int            n_block = n_draft_ + 1;
        for (size_t k = 0; k < n_parts; ++k) {
            DraftPart& p = parts[k];
            p.out->clear();
#if ARCINT_DFLASH_RED == 9
            continue;   // the diagnostic: no drafts, the layer inputs and the injection as served
#endif
            if (p.n_max <= 0) continue;
            check_window(p.seq, p.pos0);
            for (int i = 0; i < n_block; ++i) {
                const int32_t idx = llama_batch_ext_add_token(bd_, p.seq, i == 0 ? p.id_last : mask_);
                if (idx < 0) return;
                if (i == 0) beg[k] = idx;
#if ARCINT_DFLASH_RED == 3
                const llama_pos pos = static_cast<llama_pos>(p.pos0 + static_cast<size_t>(i) + 1);   // red case: the block off by one
#else
                const llama_pos pos = static_cast<llama_pos>(p.pos0 + static_cast<size_t>(i));
#endif
                llama_batch_ext_set_pos(bd_, idx, &pos);
            }
        }
        bool any = false;
        for (int32_t b : beg) any |= b >= 0;
        if (!any) return;
        const int rc = llama_process(ctx_dft_, LLAMA_PROCESS_TYPE_DECODE, bd_);
        const float* lattice = rc == 0 ? llama_get_embeddings_nextn(ctx_dft_) : nullptr;
        if (lattice == nullptr) log::warn("dflash", "the drafter's decode failed (%d): no drafts this step", rc);
        const size_t stride = static_cast<size_t>(n_embd_dec_);
        for (size_t k = 0; k < n_parts; ++k) {
            if (beg[k] < 0) continue;
            DraftPart& p = parts[k];
            if (lattice != nullptr)
                *p.out = dflash_walk(lattice + static_cast<size_t>(beg[k]) * stride, static_cast<size_t>(n_block), stride,
                                     top_k_, min_p_, std::min(p.n_max, n_draft_));
#if ARCINT_DFLASH_RED != 4
            // the block leaves; the verify's features land at pos0.. next
            llama_memory_seq_rm(llama_get_memory(ctx_dft_), p.seq, static_cast<llama_pos>(p.pos0), -1);
#endif
        }
    }

    void reset(int /*seq*/) override {}   // the drafter carries no row

    void accept(int /*seq*/, int /*n_accepted*/) override {}   // llama.cpp's is a no-op

    bool seq_rm(int seq, size_t from) override {
        llama_memory_t mt = llama_get_memory(ctx_tgt_);
        llama_memory_t md = llama_get_memory(ctx_dft_);
        if (!llama_memory_seq_rm(mt, seq, static_cast<llama_pos>(from), -1)) {
            llama_memory_seq_rm(mt, seq, -1, -1);
            llama_memory_seq_rm(md, seq, -1, -1);
            return false;
        }
#if ARCINT_DFLASH_RED == 5
        if (from == 0)   // red case: the rejected rows stay in the drafter (a sequence is still cleared)
#endif
            llama_memory_seq_rm(md, seq, static_cast<llama_pos>(from), -1);
        return true;
    }

    // the drafter's window K/V (its sliding-window cache: PARTIAL_ONLY
    // writes the SWA caches only, llama-kv-cache.cpp:2280), as the server's
    // checkpoint.update_dft (server-context.cpp:2597)
    std::vector<uint8_t> carried_state(int seq, size_t n) override {
#if ARCINT_DFLASH_RED == 6
        (void) seq;
        (void) n;
        return {};   // red case: checkpoints without the drafter's state
#else
        if (n == 0) return {};
        const size_t size = llama_state_seq_get_size_ext(ctx_dft_, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (size == 0) return {};
        std::vector<uint8_t> s = spec_state_alloc(SpecStateKind::DflashWindow, n, size);
        if (llama_state_seq_get_data_ext(ctx_dft_, s.data() + kSpecStateHeader, size, seq,
                                         LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != size)
            return {};
        return s;
#endif
    }

    // after a checkpoint's restore: its window back (the server's load_dft,
    // server-context.cpp:3770-3771), cut to n; without one the drafter's
    // sequence starts empty and the next injections refill it
    void set_carried_state(int seq, size_t n, const std::vector<uint8_t>& state) override {
        llama_memory_t md = llama_get_memory(ctx_dft_);
        SpecStateView  v;
        if (n == 0 || !spec_state_view(state, SpecStateKind::DflashWindow, n, v)) {
            llama_memory_seq_rm(md, seq, -1, -1);
            log::verbose("dflash", "lane %d: no drafter state with the checkpoint at %zu tokens: the window refills", seq, n);
            return;
        }
        if (llama_state_seq_set_data_ext(ctx_dft_, v.data, v.size, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != v.size) {
            llama_memory_seq_rm(md, seq, -1, -1);
            log::warn("dflash", "lane %d: loading the drafter's state at %zu tokens failed: the window refills", seq, n);
            return;
        }
        llama_memory_seq_rm(md, seq, static_cast<llama_pos>(n), -1);
    }

    // the prompt rows the drafter takes: the last kDflashPrefillRows before
    // n_end (its window at the first draft; at least the window) and the
    // window before each checkpoint (its saved state); the others stay out.
    // A drafter without a sliding window attends every row: it takes them all
    void prefill_plan(int seq, size_t n_end, const std::vector<size_t>& ckpts) override {
        Lane& ln = lanes_[static_cast<size_t>(seq)];
        ln.keep.clear();
#if ARCINT_DFLASH_RED == 8
        (void) n_end;
        (void) ckpts;
        return;   // the equivalence arm: every prompt row, as the reference injects them
#endif
        if (n_swa_ <= 0) return;
        const size_t w    = static_cast<size_t>(n_swa_);
        const size_t rows = std::max(kDflashPrefillRows, w);
        ln.keep.emplace_back(n_end > rows ? n_end - rows : 0, n_end);
        for (size_t c : ckpts) ln.keep.emplace_back(c > w ? c - w : 0, c);
    }

    bool set_kv_windows(const uint32_t* n_cells, int n) override {
        return llama_memory_seq_windows(llama_get_memory(ctx_dft_), n_cells, n);
    }

private:
    struct Lane {
        // a prefill's prompt rows the drafter takes, [from, to) ranges
        // (prefill_plan); none planned: every row
        std::vector<std::pair<size_t, size_t>> keep;
        uint64_t                               mismatches = 0;   // drafts whose drafter did not end where the target does
    };

    // The part's rows (from batch offset `off` of the target's decode) into
    // the drafter at their positions: the target layers' inputs side by side
    // a token, in chunks of the draft ubatch (common/speculative.cpp:1209-1271).
    // A prompt part injects its rows from the first one in a planned range
    // on (the ranges end at batch boundaries: a checkpoint starts a batch,
    // n_end ends the last, so a part's rows in them are its tail); a
    // verify's are all injected, as the reference injects every row
    int inject(const SpecPart& p, size_t off) {
        Lane&  ln = lanes_[static_cast<size_t>(p.seq)];
        size_t i0 = 0;
        if (!p.all_logits && !ln.keep.empty()) {
            i0 = p.n;
            for (const auto& [a, b] : ln.keep)
                if (b > p.pos && a < p.pos + p.n) i0 = std::min(i0, a > p.pos ? a - p.pos : 0);
        }
        if (i0 >= p.n) return 0;
        llama_memory_t  md    = llama_get_memory(ctx_dft_);
        const llama_pos start = static_cast<llama_pos>(p.pos + i0);
        // past rows that stayed out: the drafter's cells start a new run (the
        // rows before reach no draft; a checkpoint that needed them is saved)
        if (!p.all_logits) {
            const llama_pos pm = llama_memory_seq_pos_max(md, p.seq);
            if (pm >= 0 && start > pm + 1) llama_memory_seq_rm(md, p.seq, -1, -1);
        }
        std::vector<const float*> src(layers_.size());
        for (size_t k = 0; k < layers_.size(); ++k) {
            src[k] = llama_get_embeddings_layer_inp(ctx_tgt_, layers_[k]);
            if (src[k] == nullptr) return -1;
        }
        const size_t n_rows = p.n - i0;
        const size_t ch     = static_cast<size_t>(n_ubatch_dft_);
        const size_t we     = static_cast<size_t>(n_embd_enc_);
        const size_t wt     = static_cast<size_t>(n_embd_tgt_);
        for (size_t at = 0; at < n_rows; at += ch) {
            const size_t n_chunk = std::min(ch, n_rows - at);
            feat_.resize(n_chunk * we);
            for (size_t k = 0; k < layers_.size(); ++k)
                for (size_t i = 0; i < n_chunk; ++i) {
#if ARCINT_DFLASH_RED == 2
                    const size_t row = off + i0 + at + (n_chunk - 1 - i);   // red case: the rows shuffled (reversed)
#else
                    const size_t row = off + i0 + at + i;
#endif
                    std::memcpy(feat_.data() + i * we + k * wt, src[k] + row * wt, wt * sizeof(float));
                }
            llama_batch_ext_clear(bd_);
            for (size_t i = 0; i < n_chunk; ++i) {
                const int32_t idx = llama_batch_ext_add_embd(bd_, p.seq, llama_embd{ feat_.data() + i * we, 1, we });
                if (idx < 0) return -1;
                const llama_pos q      = start + static_cast<llama_pos>(at + i);
                // an M-RoPE draft reads four positions an embedding row
                // (llama_batch_ext::set_token_pos), the others the first
                const llama_pos pos[4] = { q, q, q, 0 };
                llama_batch_ext_set_pos(bd_, idx, pos);
            }
            const int rc = llama_process(ctx_dft_, LLAMA_PROCESS_TYPE_DECODE, bd_);
            if (rc != 0) {
                log::error("dflash", "lane %d: injecting %zu rows at %d into the drafter failed (%d)", p.seq, n_chunk,
                           static_cast<int>(start + static_cast<llama_pos>(at)), rc);
                return rc;
            }
        }
        return 0;
    }

    // Before a draft at pos0 the drafter holds the target's rows through
    // pos0 - 1, as the server's begin() checks (common/speculative.cpp:
    // 1155-1171): a drafter that ends elsewhere drafts from a stale or
    // shifted window. Counted and logged once a lane; the drafts stay
    // verified either way
    void check_window(int seq, size_t pos0) {
        const llama_pos pd = llama_memory_seq_pos_max(llama_get_memory(ctx_dft_), seq);
        if (pd == static_cast<llama_pos>(pos0) - 1) return;
        Lane& ln = lanes_[static_cast<size_t>(seq)];
        if (ln.mismatches++ == 0)
            log::warn("dflash", "lane %d: the drafter holds positions up to %d before a draft at %zu (the target's %d): "
                                "its window and the target disagree", seq, static_cast<int>(pd), pos0,
                      static_cast<int>(pos0) - 1);
    }

    llama_context*        ctx_tgt_     = nullptr;
    llama_context*        ctx_dft_     = nullptr;
    llama_batch_ext*      bt_          = nullptr;
    llama_batch_ext*      bd_          = nullptr;
    std::vector<uint32_t> layers_;   // the target layers whose inputs the drafter takes
    std::vector<float>    feat_;     // one injection chunk: [n, n_layers x n_embd]
    std::vector<Lane>     lanes_;
    float                 min_p_       = 0.0f;
    int                   n_draft_     = 0;
    int                   top_k_       = 0;
    int                   n_embd_tgt_  = 0;
    int                   n_embd_dec_  = 0;
    int                   n_embd_enc_  = 0;
    int                   n_ubatch_dft_ = 512;
    int                   n_swa_       = 0;
    llama_token           mask_        = LLAMA_TOKEN_NULL;
};

}  // namespace

std::unique_ptr<LlamaSpec> make_llama_dflash(llama_model* model_dft, llama_context* ctx_tgt, int& n_draft, int n_seq,
                                             int n_ubatch, int threads, double min_p, bool kv_unified,
                                             std::string& err) {
    auto s = std::make_unique<LlamaDflash>(model_dft, ctx_tgt, n_draft, n_seq, n_ubatch, threads, min_p, kv_unified, err);
    if (!s->ok() || !err.empty()) return nullptr;
    n_draft = s->n_draft();
    return s;
}

}  // namespace lgc

#endif  // ARCINT_LLAMA
