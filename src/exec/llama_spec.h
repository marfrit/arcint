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

// One lane's tokens in a target decode shared by several lanes: the target's
// logits rows of a part start at the sum of the earlier parts' n
struct SpecPart {
    int        seq        = 0;
    const int* toks       = nullptr;
    size_t     n          = 0;
    size_t     pos        = 0;
    bool       all_logits = false;
};

// One lane's drafting in a draft step shared by several lanes: up to n_max
// drafts after id_last at pos0, into *out
struct DraftPart {
    int               seq     = 0;
    int               id_last = 0;
    size_t            pos0    = 0;
    int               n_max   = 0;
    std::vector<int>* out     = nullptr;
};

class LlamaSpec {
public:
    virtual ~LlamaSpec() = default;

    // One target decode of tokens [pos, pos + n) of `seq`, with logits for
    // every token when `all_logits` (a verify) and for none otherwise, the
    // batch then fed to the drafter (it pairs each token with the target's
    // hidden row). The logits of token i are llama_get_logits_ith(ctx, i).
    // llama.cpp's return code.
    virtual int decode(const int* toks, size_t n, size_t pos, int seq, bool all_logits) = 0;
    // Several lanes' tokens in one target decode (each lane's verify rows, as
    // llama.cpp's server batches its slots), then one MTP-context decode with
    // every lane's entries. The parts' sequences are distinct.
    virtual int decode_multi(const SpecPart* parts, size_t n_parts) = 0;
    // Several lanes' drafts, one MTP-context decode a draft step for all the
    // lanes still drafting (llama.cpp's common_speculative_draft drafts for
    // every slot at once). The parts' sequences are distinct.
    virtual void draft_multi(DraftPart* parts, size_t n_parts) = 0;
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
    // The carried hidden row of a sequence that holds exactly n tokens (the
    // target's row of position n - 1), empty when the drafter carries none
    // there; stored with a context checkpoint and set back after its restore
    // and the cut to n, as llama.cpp's server stashes the draft state with
    // each checkpoint (common_speculative_get_state).
    virtual std::vector<float> carried_row(int seq, size_t n) const = 0;
    virtual void               set_carried_row(int seq, size_t n, const std::vector<float>& row) = 0;
    // Named lanes: the draft context's cell windows, as the target's
    // (llama_memory_seq_windows, contrib/llama.cpp 0026); false when its
    // memory cannot take them.
    virtual bool set_kv_windows(const uint32_t* n_cells, int n) = 0;
};

// model: loaded with load_mtp from `gguf`; ctx_tgt: n_rs_seq >= n_draft.
// vocab: the file of token ids the drafts are drawn from (empty: all).
// min_p: drafting stops at a token the head gives less than min_p, which is
// left out (0: always n_draft).
// kv_unified: the target's (named lanes, --lane-ctx): the draft context then
// holds one pool of the target's n_ctx that every sequence can reach, as the
// target does; else n_ctx / n_seq per sequence, the target's stream.
// Null with `err` set when the model has no MTP layer, a context cannot be
// made or the draft vocabulary cannot be built.
std::unique_ptr<LlamaSpec> make_llama_mtp(llama_model* model, llama_context* ctx_tgt, int n_draft, int n_seq,
                                          int n_batch, int n_ubatch, int threads, const std::string& gguf,
                                          const std::string& vocab, double min_p, ggml_type type_k, ggml_type type_v,
                                          bool kv_unified, std::string& err);

}  // namespace lgc

#endif  // ARCINT_LLAMA
