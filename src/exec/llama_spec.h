// The libllama engine's speculative decoding behind a narrow interface: the
// GGUF's MTP head drafting (llama.cpp's draft-mtp, rebuilt on libllama in
// llama_spec.cpp) or a DFlash2 drafter (llama.cpp's draft-dflash, rebuilt in
// llama_dflash.cpp), the target verifying.
#pragma once

#ifdef ARCINT_LLAMA

#include <cstddef>
#include <cstdint>
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
    // ARCINT_MTP_TRACE (exec/mtp_trace.h): the head's top probability of every
    // draft step the chain made, in order, then the stopping draft's when
    // --llama-mtp-min-p cut the chain. Null: no probability computed beyond
    // what min-p itself needs, the off switch's zero cost.
    std::vector<float>* probs = nullptr;
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
    // The drafter's state of a sequence that holds exactly n tokens, as
    // opaque bytes (exec/spec_state.h): the MTP drafter's carried hidden row
    // (the target's row of position n - 1; empty when it carries none
    // there), the DFlash2 drafter's sliding-window K/V. Stored with a context
    // checkpoint and set back after its restore and the cut to n, as
    // llama.cpp's server stashes the draft side with each checkpoint
    // (common_speculative_get_state and the draft context's partial state).
    // A state that is not this drafter's at n is not loaded.
    virtual std::vector<uint8_t> carried_state(int seq, size_t n) = 0;
    virtual void                 set_carried_state(int seq, size_t n, const std::vector<uint8_t>& state) = 0;
    // A prefill of the sequence is about to run up to n_end tokens and save
    // context checkpoints at the token counts `ckpts`: a drafter may leave
    // out prompt rows neither the drafts after n_end nor those checkpoints'
    // states need. The MTP drafter takes every row.
    virtual void prefill_plan(int seq, size_t n_end, const std::vector<size_t>& ckpts) {
        (void) seq;
        (void) n_end;
        (void) ckpts;
    }
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
                                          bool kv_unified, uint32_t n_kv_reserve, std::string& err);

// model_dft: a DFlash2 drafter (general.architecture "dflash" with a
// selector), loaded on the target's card; ctx_tgt: n_rs_seq >= n_draft,
// n_outputs_max_per_seq >= 1 + n_draft. The draft context is llama.cpp's
// (common/speculative.cpp:2572-2700): the target's n_ctx and kv_unified,
// ctx_other = ctx_tgt (the target's embeddings and LM head), f16 K/V, a
// sliding-window cache. min_p: the selector's stop (0: n_draft always).
// n_ubatch: the draft context's (--llama-dflash-ubatch), raised to a block a
// lane. n_draft: requested, and set to what the drafter's
// trained block allows (block_size - 1). Null with `err` set when the GGUF
// is not a DFlash2 drafter for this target or the context cannot be made.
std::unique_ptr<LlamaSpec> make_llama_dflash(llama_model* model_dft, llama_context* ctx_tgt, int& n_draft, int n_seq,
                                             int n_ubatch, int threads, double min_p, bool kv_unified,
                                             std::string& err);

}  // namespace lgc

#endif  // ARCINT_LLAMA
