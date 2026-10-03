// The MTP drafter for the libllama engine (llama_spec.h), on libllama alone:
// the single-head case of llama.cpp's draft-mtp (common/speculative.cpp,
// common_speculative_impl_draft_mtp, at the pinned commit). llama.cpp's
// common library is not linked: it compiles another cpp-httplib (0.58)
// next to arcint's (0.18) into one binary.
//
// The target context outputs the hidden row the MTP head takes (its "nextn"
// embedding) for every token. After each target decode the MTP context
// decodes the same tokens, token k paired with the target's row of token
// k - 1 (the first with the row carried over from the previous batch), so
// its KV follows the target's. A draft decodes the last token with the
// carried row, takes the argmax, and feeds each drafted token with the MTP
// head's own row; the draft region is dropped from the MTP context before
// the verify rewrites it. accept(n) carries the verify's row n.
#include "exec/llama_spec.h"

#ifdef ARCINT_LLAMA

#include <llama-ext.h>   // staging API of the pinned llama.cpp: nextn embeddings
#include <llama.h>

#include <algorithm>
#include <cstring>

namespace lgc {
namespace {

class LlamaMtp final : public LlamaSpec {
public:
    LlamaMtp(llama_model* model, llama_context* ctx_tgt, int n_draft, int n_seq, int n_batch, int n_ubatch,
             int threads, std::string& err)
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
        cp.n_batch         = static_cast<uint32_t>(n_batch);
        cp.n_ubatch        = static_cast<uint32_t>(n_ubatch);
        cp.n_seq_max       = static_cast<uint32_t>(n_seq);
        cp.n_rs_seq        = 0;
        cp.n_outputs_max_per_seq = 1;   // one drafted token a step
        cp.n_outputs_max         = static_cast<uint32_t>(n_seq);
        cp.ctx_other       = ctx_tgt;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
        ctx_dft_           = llama_init_from_model(model, cp);
        if (ctx_dft_ == nullptr) {
            err = "llama.cpp could not create the MTP draft context";
            return;
        }
        llama_set_embeddings_nextn(ctx_tgt, true, /*masked=*/false);   // a row for every token
        llama_set_embeddings_nextn(ctx_dft_, true, /*masked=*/true);   // rows for the output tokens
        bt_ = llama_batch_ext_init(ctx_tgt);
        bd_ = llama_batch_ext_init(ctx_dft_);
        pending_h_.assign(static_cast<size_t>(n_seq), std::vector<float>(static_cast<size_t>(n_embd_), 0.0f));
        verify_h_.assign(static_cast<size_t>(n_seq), {});
        verify_rows_.assign(static_cast<size_t>(n_seq), 0);
        h_step_.resize(static_cast<size_t>(n_embd_));
    }

    ~LlamaMtp() override {
        if (bt_ != nullptr) llama_batch_ext_free(bt_);
        if (bd_ != nullptr) llama_batch_ext_free(bd_);
        if (ctx_dft_ != nullptr) llama_free(ctx_dft_);
    }

    bool ok() const { return bd_ != nullptr; }

    int decode(const int* toks, size_t n, size_t pos, int seq, bool all_logits) override {
        llama_batch_ext_clear(bt_);
        for (size_t i = 0; i < n; ++i)
            if (!add(bt_, toks[i], pos + i, seq, all_logits, nullptr)) return -1;
        const int r = llama_process(ctx_tgt_, LLAMA_PROCESS_TYPE_DECODE, bt_);
        if (r != 0) return r;

        // the MTP context over the same tokens, each with the previous one's row
        std::vector<float>& carried = pending_h_[static_cast<size_t>(seq)];
        const float*        h_tgt   = llama_get_embeddings_nextn(ctx_tgt_);
        if (h_tgt == nullptr) return -1;
        llama_batch_ext_clear(bd_);
        for (size_t i = 0; i < n; ++i) {
            const float* row = i == 0 ? carried.data() : h_tgt + (i - 1) * static_cast<size_t>(n_embd_);
            if (!add(bd_, toks[i], pos + i, seq, false, row)) return -1;
        }
        const int rd = llama_process(ctx_dft_, LLAMA_PROCESS_TYPE_DECODE, bd_);
        if (rd != 0) return rd;

        const size_t row_bytes = static_cast<size_t>(n_embd_) * sizeof(float);
        std::memcpy(carried.data(), h_tgt + (n - 1) * static_cast<size_t>(n_embd_), row_bytes);
        // a verify's rows, for accept(); a prefill batch needs none
        std::vector<float>& vh = verify_h_[static_cast<size_t>(seq)];
        if (all_logits) {
            vh.assign(h_tgt, h_tgt + n * static_cast<size_t>(n_embd_));
            verify_rows_[static_cast<size_t>(seq)] = static_cast<int>(n);
        } else {
            verify_rows_[static_cast<size_t>(seq)] = 0;
        }
        return 0;
    }

    std::vector<int> draft(int seq, int id_last, size_t pos0, int n_max) override {
        std::vector<int> out;
        const float*     row = pending_h_[static_cast<size_t>(seq)].data();
        int              tok = id_last;
        for (int i = 0; i < std::min(n_max, n_draft_); ++i) {
            llama_batch_ext_clear(bd_);
            if (!add(bd_, tok, pos0 + static_cast<size_t>(i), seq, true, row)) break;
            if (llama_process(ctx_dft_, LLAMA_PROCESS_TYPE_DECODE, bd_) != 0) break;
            const float* l = llama_get_logits_ith(ctx_dft_, 0);
            const float* h = llama_get_embeddings_nextn_ith(ctx_dft_, 0);
            if (l == nullptr || h == nullptr) break;
            tok = static_cast<int>(std::max_element(l, l + n_vocab_) - l);
            out.push_back(tok);
            std::memcpy(h_step_.data(), h, static_cast<size_t>(n_embd_) * sizeof(float));
            row = h_step_.data();
        }
        // the verify's own pass rewrites the draft region with the target's rows
        llama_memory_seq_rm(llama_get_memory(ctx_dft_), seq, static_cast<llama_pos>(pos0), -1);
        return out;
    }

    // as the reference starts a sequence: a zero row (common/speculative.cpp)
    void reset(int seq) override {
        std::fill(pending_h_[static_cast<size_t>(seq)].begin(), pending_h_[static_cast<size_t>(seq)].end(), 0.0f);
        verify_rows_[static_cast<size_t>(seq)] = 0;
    }

    void accept(int seq, int n_accepted) override {
        const int rows = verify_rows_[static_cast<size_t>(seq)];
        if (rows <= 0) return;
        const int i = std::min(std::max(n_accepted, 0), rows - 1);
        std::memcpy(pending_h_[static_cast<size_t>(seq)].data(),
                    verify_h_[static_cast<size_t>(seq)].data() + static_cast<size_t>(i) * static_cast<size_t>(n_embd_),
                    static_cast<size_t>(n_embd_) * sizeof(float));
    }

    bool seq_rm(int seq, size_t from) override {
        llama_memory_t mt = llama_get_memory(ctx_tgt_);
        llama_memory_t md = llama_get_memory(ctx_dft_);
        if (!llama_memory_seq_rm(mt, seq, static_cast<llama_pos>(from), -1)) {
            llama_memory_seq_rm(mt, seq, -1, -1);
            llama_memory_seq_rm(md, seq, -1, -1);
            reset(seq);
            return false;
        }
        llama_memory_seq_rm(md, seq, static_cast<llama_pos>(from), -1);
        return true;
    }

private:
    static std::string log_heads(int n) {
        return "the GGUF has " + std::to_string(n) + " MTP layers; --llama-mtp drives a single-head MTP (qwen35, qwen35moe)";
    }

    // a token entry, as common_batch renders one (common/common.cpp)
    bool add(llama_batch_ext* b, int tok, size_t pos, int seq, bool out, const float* row) {
        const int32_t idx = llama_batch_ext_add_token(b, seq, tok);
        if (idx < 0) return false;
        const llama_pos p = static_cast<llama_pos>(pos);
        llama_batch_ext_set_pos(b, idx, &p);
        if (row != nullptr && !llama_batch_ext_set_embd_token(b, idx, llama_embd{ row, 1, static_cast<size_t>(n_embd_) }))
            return false;
        if (out) llama_batch_ext_set_output_logits(b, idx, true);
        return true;
    }

    llama_context*                  ctx_tgt_ = nullptr;
    llama_context*                  ctx_dft_ = nullptr;
    llama_batch_ext*                bt_      = nullptr;
    llama_batch_ext*                bd_      = nullptr;
    int                             n_draft_ = 0;
    int                             n_embd_  = 0;
    int                             n_vocab_ = 0;
    std::vector<std::vector<float>> pending_h_;   // per sequence: the row the next MTP token takes
    std::vector<std::vector<float>> verify_h_;    // per sequence: the last verify's target rows
    std::vector<int>                verify_rows_;
    std::vector<float>              h_step_;
};

}  // namespace

std::unique_ptr<LlamaSpec> make_llama_mtp(llama_model* model, llama_context* ctx_tgt, int n_draft, int n_seq,
                                          int n_batch, int n_ubatch, int threads, std::string& err) {
    auto s = std::make_unique<LlamaMtp>(model, ctx_tgt, n_draft, n_seq, n_batch, n_ubatch, threads, err);
    if (!s->ok()) return nullptr;
    return s;
}

}  // namespace lgc

#endif  // ARCINT_LLAMA
