// The MTP drafter for the libllama engine (llama_spec.h), on libllama alone:
// the single-head case of llama.cpp's draft-mtp (common/speculative.cpp,
// common_speculative_impl_draft_mtp, at the pinned commit). llama.cpp's
// common library is not linked: it compiles another cpp-httplib (0.58)
// next to arcint's (0.18) into one binary.
//
// The target context outputs the hidden row the MTP head takes (its "nextn"
// embedding) for every token. The MTP context holds, at position q, token q
// paired with the target's row of position q - 1, so its KV follows the
// target's. A target batch leaves its rows pending until the tokens that
// follow them are known: a prompt's are, a verify's are once the walk has
// accepted a drafts (accept). draft() then decodes rows 0..a, each with the
// token that followed it -- the accepted drafts, then the token the walk
// drew -- and the last row's output is draft 0, as Strata pairs the window
// with the target's picks (src/core/mtp.cpp:771-800): one MTP forward a
// cycle fewer than pairing each verify token with the previous row and
// drafting from there (llama.cpp's draft-mtp). Further drafts feed each
// drafted token with the MTP head's own row; they leave the MTP context
// before the verify. A target decode that finds rows still pending (no draft
// was asked for) writes them first, with its own first token.
//
// With a draft vocabulary (--llama-mtp-vocab) the MTP context returns its
// head input rows (unmasked nextn rows) and no logits, and DraftHead scores
// those token rows of the output head only, on the model's device: Strata
// (mtp.hpp:156, 106,299 rows), NInfer (131,072) and HyperQwen (40,960) draft
// from such a subset, where llama.cpp reads all 248,320 rows a draft step.
#include "exec/llama_spec.h"

#ifdef ARCINT_LLAMA

#include <fcntl.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>
#include <llama-ext.h>   // staging API of the pinned llama.cpp: nextn embeddings
#include <llama.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "exec/token_ids.h"
#include "util/log.h"

namespace lgc {
namespace {

// The output head's rows of a token subset, on the device the model runs on
// (the OpenCL backend exposes the one card GGML_OPENCL_PLATFORM selected):
// read from the GGUF in its own quantization, multiplied with a draft step's
// head input, the subset's argmax mapped back to a token id.
class DraftHead {
public:
    ~DraftHead() {
        if (buf_ != nullptr) ggml_backend_buffer_free(buf_);
        if (ctx_ != nullptr) ggml_free(ctx_);
        if (be_ != nullptr) ggml_backend_free(be_);
    }

    bool init(const std::string& gguf, const std::string& ids_path, int n_embd, int n_vocab, std::string& err) {
        std::ifstream f(ids_path, std::ios::binary);
        if (!f) {
            err = "cannot read " + ids_path;
            return false;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        const bool json = ids_path.size() >= 5 && ids_path.compare(ids_path.size() - 5, 5, ".json") == 0;
        if (!parse_token_ids(ss.str(), json, n_vocab, ids_, err)) {
            err = ids_path + ": " + err;
            return false;
        }

        // the head's rows: output.weight, n_vocab rows of n_embd, in whichever
        // shard of a split GGUF holds it (Flash-Next: the second of three)
        std::string     file;
        gguf_context*   g  = nullptr;
        int64_t         ti = -1;
        if (!find_tensor(gguf, "output.weight", file, g, ti, err)) return false;
        const ggml_type type      = gguf_get_tensor_type(g, ti);
        const size_t    row_bytes = ggml_row_size(type, n_embd);
        const size_t    off       = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti);
        const size_t    size      = gguf_get_tensor_size(g, ti);
        gguf_free(g);
        if (size != row_bytes * static_cast<size_t>(n_vocab)) {
            err = "output.weight is not " + std::to_string(n_vocab) + " rows of " + std::to_string(n_embd);
            return false;
        }
        std::vector<uint8_t> rows(row_bytes * ids_.size());
        const int fd = ::open(file.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "cannot open " + file;
            return false;
        }
        bool ok = true;
        for (size_t i = 0; i < ids_.size() && ok; ++i) {
            const off_t at = static_cast<off_t>(off + row_bytes * static_cast<size_t>(ids_[i]));
            ok = ::pread(fd, rows.data() + i * row_bytes, row_bytes, at) == static_cast<ssize_t>(row_bytes);
        }
        // device-bound bytes: not kept twice in the page cache
        ::posix_fadvise(fd, static_cast<off_t>(off), static_cast<off_t>(size), POSIX_FADV_DONTNEED);
        ::close(fd);
        if (!ok) {
            err = "short read of output.weight from " + file;
            return false;
        }

        // the model's card: the build's one GPU backend is OpenCL, and the
        // engine pins its platform and device before the model loads
        // (backend_llama.cpp); a second GPU backend would need the model's
        // device named here
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev == nullptr || (be_ = ggml_backend_dev_init(dev, nullptr)) == nullptr) {
            err = "no GPU backend for the draft head";
            return false;
        }
        ggml_init_params ip{ ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, /*no_alloc=*/true };
        ctx_ = ggml_init(ip);
        w_   = ggml_new_tensor_2d(ctx_, type, n_embd, static_cast<int64_t>(ids_.size()));
        h_   = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, n_embd);
        out_ = ggml_mul_mat(ctx_, w_, h_);
        gf_  = ggml_new_graph(ctx_);
        ggml_build_forward_expand(gf_, out_);
        buf_ = ggml_backend_alloc_ctx_tensors(ctx_, be_);
        if (buf_ == nullptr) {
            err = "cannot allocate the draft head on the device";
            return false;
        }
        ggml_backend_tensor_set(w_, rows.data(), 0, rows.size());
        logits_.resize(ids_.size());
        n_embd_ = n_embd;
        return true;
    }

    size_t rows() const { return ids_.size(); }

    // `name` in `gguf` or, for a split GGUF (split.count > 1, shards named
    // <prefix>-0000k-of-0000N.gguf), in the shard that holds it; `g` stays
    // open for the caller to free
    static bool find_tensor(const std::string& gguf, const char* name, std::string& file, gguf_context*& g,
                            int64_t& ti, std::string& err) {
        gguf_init_params gp{ /*no_alloc=*/true, /*ctx=*/nullptr };
        g = gguf_init_from_file(gguf.c_str(), gp);
        if (g == nullptr) {
            err = "cannot open " + gguf;
            return false;
        }
        file = gguf;
        ti   = gguf_find_tensor(g, name);
        if (ti >= 0) return true;
        const int64_t kc = gguf_find_key(g, "split.count");
        const int     n  = kc >= 0 && gguf_get_kv_type(g, kc) == GGUF_TYPE_UINT16 ? static_cast<int>(gguf_get_val_u16(g, kc)) : 1;
        gguf_free(g);
        g = nullptr;
        const std::string tag = "-00001-of-";
        const size_t      at  = gguf.rfind(tag);
        if (n <= 1 || at == std::string::npos) {
            err = std::string("no ") + name + " in " + gguf + " (a model with tied embeddings has none)";
            return false;
        }
        for (int k = 2; k <= n; ++k) {
            char no[8];
            std::snprintf(no, sizeof no, "%05d", k);
            const std::string shard = gguf.substr(0, at + 1) + no + gguf.substr(at + 6);
            g = gguf_init_from_file(shard.c_str(), gp);
            if (g == nullptr) continue;
            ti = gguf_find_tensor(g, name);
            if (ti >= 0) {
                file = shard;
                return true;
            }
            gguf_free(g);
            g = nullptr;
        }
        err = std::string("no ") + name + " in the " + std::to_string(n) + " shards of " + gguf;
        return false;
    }

    // the token the head input `h` drafts, -1 when the device refuses
    int pick(const float* h) {
        ggml_backend_tensor_set(h_, h, 0, static_cast<size_t>(n_embd_) * sizeof(float));
        if (ggml_backend_graph_compute(be_, gf_) != GGML_STATUS_SUCCESS) return -1;
        ggml_backend_tensor_get(out_, logits_.data(), 0, logits_.size() * sizeof(float));
        return ids_[static_cast<size_t>(std::max_element(logits_.begin(), logits_.end()) - logits_.begin())];
    }

private:
    std::vector<int32_t>  ids_;
    std::vector<float>    logits_;
    ggml_backend_t        be_   = nullptr;
    ggml_context*         ctx_  = nullptr;
    ggml_backend_buffer_t buf_  = nullptr;
    ggml_tensor*          w_    = nullptr;
    ggml_tensor*          h_    = nullptr;
    ggml_tensor*          out_  = nullptr;
    ggml_cgraph*          gf_   = nullptr;
    int                   n_embd_ = 0;
};

class LlamaMtp final : public LlamaSpec {
public:
    LlamaMtp(llama_model* model, llama_context* ctx_tgt, int n_draft, int n_seq, int n_batch, int n_ubatch,
             int threads, const std::string& gguf, const std::string& vocab, ggml_type type_k, ggml_type type_v,
             std::string& err)
        : ctx_tgt_(ctx_tgt), n_draft_(n_draft) {
        const int n_heads = llama_model_n_layer_nextn(model);
        if (n_heads <= 0) {
            err = "the GGUF has no MTP layer";
            return;
        }
        if (n_heads != 1) {
            err = log_heads(n_heads);
            return;
        }
        n_embd_  = llama_model_n_embd_out(model);
        n_vocab_ = llama_vocab_n_tokens(llama_model_get_vocab(model));
        llama_context_params cp = llama_context_default_params();
        cp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx           = llama_n_ctx(ctx_tgt);
        // a prompt batch, the rows a verify left pending before it (up to
        // n_draft + 1: a follow-up turn continues where the last verify
        // stopped), and a sequence's first (zero-row) entry
        cp.n_batch         = static_cast<uint32_t>(n_batch + n_draft + 2);
        cp.n_ubatch        = static_cast<uint32_t>(n_ubatch);
        cp.n_seq_max       = static_cast<uint32_t>(n_seq);
        cp.n_rs_seq        = 0;
        cp.n_outputs_max_per_seq = 1;   // one drafted token a step
        cp.n_outputs_max         = static_cast<uint32_t>(n_seq);
        cp.ctx_other       = ctx_tgt;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.type_k          = type_k;   // the target's cache types (--llama-kv)
        cp.type_v          = type_v;
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        ctx_dft_           = llama_init_from_model(model, cp);
        if (ctx_dft_ == nullptr) {
            err = "llama.cpp could not create the MTP draft context";
            return;
        }
        if (!vocab.empty()) {
            head_ = std::make_unique<DraftHead>();
            if (!head_->init(gguf, vocab, n_embd_, n_vocab_, err)) return;
        }
        llama_set_embeddings_nextn(ctx_tgt, true, /*masked=*/false);   // a row for every token
        // rows for the output tokens; with the draft head, every token's (the
        // draft steps then request no logits)
        llama_set_embeddings_nextn(ctx_dft_, true, /*masked=*/head_ == nullptr);
        bt_ = llama_batch_ext_init(ctx_tgt);
        bd_ = llama_batch_ext_init(ctx_dft_);
        lanes_.resize(static_cast<size_t>(n_seq));
        zero_.assign(static_cast<size_t>(n_embd_), 0.0f);
        h_step_.resize(static_cast<size_t>(n_embd_));
    }

    ~LlamaMtp() override {
        if (bt_ != nullptr) llama_batch_ext_free(bt_);
        if (bd_ != nullptr) llama_batch_ext_free(bd_);
        if (ctx_dft_ != nullptr) llama_free(ctx_dft_);
    }

    bool ok() const { return bd_ != nullptr; }

    size_t head_rows() const { return head_ ? head_->rows() : 0; }

    int decode(const int* toks, size_t n, size_t pos, int seq, bool all_logits) override {
        Lane& ln = lanes_[static_cast<size_t>(seq)];
        llama_batch_ext_clear(bt_);
        for (size_t i = 0; i < n; ++i)
            if (!add(bt_, toks[i], pos + i, seq, all_logits, nullptr)) return -1;
        const int r = llama_process(ctx_tgt_, LLAMA_PROCESS_TYPE_DECODE, bt_);
        if (r != 0) return r;
        const float* h_tgt = llama_get_embeddings_nextn(ctx_tgt_);
        if (h_tgt == nullptr) return -1;

        // the MTP context up to this batch's first token: rows a previous batch
        // left pending, or, at a sequence's start, the reference's zero row
        clear_d();
        if (ln.pending > 0 && ln.base + static_cast<size_t>(ln.pending) == pos) {
            if (!add_pending(ln, toks[0], seq, false)) return -1;
        } else {
            ln.pending = 0;
            if (ln.mtp_end == pos && !add(bd_, toks[0], pos, seq, false, zero_.data())) return -1;
        }
        const size_t row_bytes = static_cast<size_t>(n_embd_) * sizeof(float);
        if (all_logits) {
            // a verify: which rows stand is known after the walk
            ln.rows.assign(h_tgt, h_tgt + n * static_cast<size_t>(n_embd_));
            ln.next.assign(toks + 1, toks + n);
            ln.base    = pos;
            ln.pending = static_cast<int>(n);
        } else {
            // a prompt: every row but the last is followed by a known token
            for (size_t i = 0; i + 1 < n; ++i)
                if (!add(bd_, toks[i + 1], pos + i + 1, seq, false, h_tgt + i * static_cast<size_t>(n_embd_))) return -1;
            ln.rows.resize(static_cast<size_t>(n_embd_));
            std::memcpy(ln.rows.data(), h_tgt + (n - 1) * static_cast<size_t>(n_embd_), row_bytes);
            ln.next.clear();
            ln.base    = pos + n - 1;
            ln.pending = 1;
        }
        if (nd_ > 0) {
            const int rd = llama_process(ctx_dft_, LLAMA_PROCESS_TYPE_DECODE, bd_);
            if (rd != 0) return rd;
        }
        ln.mtp_end = std::max(ln.mtp_end, pos + (all_logits ? 1 : n));
        return 0;
    }

    std::vector<int> draft(int seq, int id_last, size_t pos0, int n_max) override {
        Lane&            ln = lanes_[static_cast<size_t>(seq)];
        std::vector<int> out;
        const int        n  = std::min(n_max, n_draft_);
        if (n <= 0 || ln.pending <= 0 || ln.base + static_cast<size_t>(ln.pending) != pos0) return out;
        // the pending rows with what followed them; the last one's output is draft 0
        // (kept if the MTP context refuses it: the verify's decode writes them)
        const int pending = ln.pending;
        clear_d();
        if (!add_pending(ln, id_last, seq, head_ == nullptr)) return out;
        const float* row = nullptr;
        for (int i = 0; i < n; ++i) {
            if (i > 0) {
                clear_d();
                if (!add(bd_, out.back(), pos0 + static_cast<size_t>(i), seq, head_ == nullptr, row)) break;
            }
            if (llama_process(ctx_dft_, LLAMA_PROCESS_TYPE_DECODE, bd_) != 0) {
                if (i == 0) ln.pending = pending;
                break;
            }
            if (i == 0) ln.mtp_end = pos0 + 1;
            int tok = -1;
            const float* h = nullptr;
            if (head_) {
                h   = llama_get_embeddings_nextn_ith(ctx_dft_, nd_ - 1);   // unmasked: the batch index
                tok = h != nullptr ? head_->pick(h) : -1;
            } else {
                const float* l = llama_get_logits_ith(ctx_dft_, -1);
                h   = llama_get_embeddings_nextn_ith(ctx_dft_, -1);
                tok = l != nullptr ? static_cast<int>(std::max_element(l, l + n_vocab_) - l) : -1;
            }
            if (h == nullptr || tok < 0) break;
            out.push_back(tok);
            std::memcpy(h_step_.data(), h, static_cast<size_t>(n_embd_) * sizeof(float));
            row = h_step_.data();
        }
        // the drafts leave; the entry of pos0 (id_last, a target row) stays
        llama_memory_seq_rm(llama_get_memory(ctx_dft_), seq, static_cast<llama_pos>(pos0 + 1), -1);
        return out;
    }

    // as the reference starts a sequence: the next entry takes a zero row
    void reset(int seq) override {
        Lane& ln   = lanes_[static_cast<size_t>(seq)];
        ln.pending = 0;
    }

    // a verify of n rows kept a + 1 of them: rows 0..a stand, row a followed
    // by the token the walk drew
    void accept(int seq, int n_accepted) override {
        Lane& ln = lanes_[static_cast<size_t>(seq)];
        if (ln.pending <= 0) return;
        ln.pending = std::min(std::max(n_accepted, 0) + 1, ln.pending);
    }

    bool seq_rm(int seq, size_t from) override {
        Lane&          ln = lanes_[static_cast<size_t>(seq)];
        llama_memory_t mt = llama_get_memory(ctx_tgt_);
        llama_memory_t md = llama_get_memory(ctx_dft_);
        if (!llama_memory_seq_rm(mt, seq, static_cast<llama_pos>(from), -1)) {
            llama_memory_seq_rm(mt, seq, -1, -1);
            llama_memory_seq_rm(md, seq, -1, -1);
            ln.mtp_end = 0;
            reset(seq);
            return false;
        }
        llama_memory_seq_rm(md, seq, static_cast<llama_pos>(from), -1);
        ln.mtp_end = std::min(ln.mtp_end, from);
        // pending rows up to the cut stay: the next decode writes them, the
        // last with its own first token; a cut at or before their first row
        // leaves that position to the zero row, as a sequence start
        if (ln.pending > 0 && ln.base + static_cast<size_t>(ln.pending) > from) {
            if (from <= ln.base) reset(seq);
            else ln.pending = static_cast<int>(from - ln.base);
        }
        return true;
    }

private:
    // per sequence: target rows waiting for the tokens that follow them --
    // row i of position base + i, followed by next[i] (the last by the token
    // the caller supplies); mtp_end: the first position the MTP context lacks
    struct Lane {
        std::vector<float> rows;
        std::vector<int>   next;
        size_t             base    = 0;
        int                pending = 0;
        size_t             mtp_end = 0;
    };

    static std::string log_heads(int n) {
        return "the GGUF has " + std::to_string(n) + " MTP layers; --llama-mtp drives a single-head MTP (qwen35, qwen35moe)";
    }

    // the pending rows into bd_: row i at base + i + 1 with next[i], the last
    // with `tok` (callers check that base + pending is its position); `out`:
    // logits for the last
    bool add_pending(Lane& ln, int tok, int seq, bool out) {
        for (int i = 0; i < ln.pending; ++i) {
            const bool last = i + 1 == ln.pending;
            if (!add(bd_, last ? tok : ln.next[static_cast<size_t>(i)], ln.base + static_cast<size_t>(i) + 1, seq,
                     out && last, ln.rows.data() + static_cast<size_t>(i) * static_cast<size_t>(n_embd_)))
                return false;
        }
        ln.pending = 0;
        return true;
    }

    void clear_d() {
        llama_batch_ext_clear(bd_);
        nd_ = 0;
    }

    // a token entry, as common_batch renders one (common/common.cpp)
    bool add(llama_batch_ext* b, int tok, size_t pos, int seq, bool out, const float* row) {
        const int32_t idx = llama_batch_ext_add_token(b, seq, tok);
        if (idx < 0) return false;
        if (b == bd_) ++nd_;
        const llama_pos p = static_cast<llama_pos>(pos);
        llama_batch_ext_set_pos(b, idx, &p);
        if (row != nullptr && !llama_batch_ext_set_embd_token(b, idx, llama_embd{ row, 1, static_cast<size_t>(n_embd_) }))
            return false;
        if (out) llama_batch_ext_set_output_logits(b, idx, true);
        return true;
    }

    llama_context*     ctx_tgt_ = nullptr;
    llama_context*     ctx_dft_ = nullptr;
    llama_batch_ext*   bt_      = nullptr;
    llama_batch_ext*   bd_      = nullptr;
    int                n_draft_ = 0;
    int                n_embd_  = 0;
    int                n_vocab_ = 0;
    int                nd_      = 0;   // entries in bd_
    std::unique_ptr<DraftHead> head_;
    std::vector<Lane>  lanes_;
    std::vector<float> zero_;
    std::vector<float> h_step_;
};

}  // namespace

std::unique_ptr<LlamaSpec> make_llama_mtp(llama_model* model, llama_context* ctx_tgt, int n_draft, int n_seq,
                                          int n_batch, int n_ubatch, int threads, const std::string& gguf,
                                          const std::string& vocab, ggml_type type_k, ggml_type type_v,
                                          std::string& err) {
    auto s = std::make_unique<LlamaMtp>(model, ctx_tgt, n_draft, n_seq, n_batch, n_ubatch, threads, gguf, vocab,
                                        type_k, type_v, err);
    if (!s->ok() || !err.empty()) return nullptr;
    if (s->head_rows() > 0) log::info("mtp", "draft head: %zu token rows", s->head_rows());
    return s;
}

}  // namespace lgc

#endif  // ARCINT_LLAMA
