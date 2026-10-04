// The libllama engine's speculative decoding: the GGUF's MTP head drafting
// (llama.cpp's draft-mtp, rebuilt on libllama in llama_spec.cpp), the target
// verifying, behind a narrow interface.
#pragma once

#ifdef ARCINT_LLAMA

#include <cstddef>
#include <ggml.h>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

namespace lgc {

class LlamaSpec {
public:
    virtual ~LlamaSpec() = default;

    // One target decode of tokens [pos, pos + n) of `seq`, with logits for
    // every token when `all_logits` (a verify) and for none otherwise, the
    // batch then fed to the drafter (it pairs each token with the target's
    // hidden row). The logits of token i are llama_get_logits_ith(ctx, i).
    // llama.cpp's return code.
    virtual int decode(const int* toks, size_t n, size_t pos, int seq, bool all_logits) = 0;
    // Up to n_max tokens following `id_last` at position pos0.
    virtual std::vector<int> draft(int seq, int id_last, size_t pos0, int n_max) = 0;
    // The sequence's context no longer ends where the drafter's carried
    // hidden row belongs (a trimmed prefix): start that row afresh.
    virtual void reset(int seq) = 0;
    // How many of the last draft's tokens the verify kept.
    virtual void accept(int seq, int n_accepted) = 0;
    // Drop positions >= from in both contexts; false when the target's
    // memory refuses (beyond its recurrent snapshots), and then both are
    // cleared for the sequence.
    virtual bool seq_rm(int seq, size_t from) = 0;
};

// model: loaded with load_mtp from `gguf`; ctx_tgt: n_rs_seq >= n_draft.
// vocab: the file of token ids the drafts are drawn from (empty: all).
// Null with `err` set when the model has no MTP layer, a context cannot be
// made or the draft vocabulary cannot be built.
std::unique_ptr<LlamaSpec> make_llama_mtp(llama_model* model, llama_context* ctx_tgt, int n_draft, int n_seq,
                                          int n_batch, int n_ubatch, int threads, const std::string& gguf,
                                          const std::string& vocab, ggml_type type_k, ggml_type type_v,
                                          std::string& err);

}  // namespace lgc

#endif  // ARCINT_LLAMA
