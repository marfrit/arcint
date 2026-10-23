#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/sampling.h"

// DESIGN.md §3.6: greedy, temperature, top-k, top-p, repetition penalty.
// Nothing else in v1, and nothing silent — a request that asks for temperature
// gets temperature, or an error, never argmax wearing a temperature label.
//
// Kept out of the backend so it can be tested on a machine with no GPU: given
// the same logits and seed it must produce the same token, every time.
//
// The token history is owned here and maintained incrementally. Rebuilding it
// per step would make decode O(prompt_length) per token, on models whose
// advertised context is 262144.
namespace lgc {

// clean-turns step 2: a constraint on which token may come next (the
// tool-call grammar, src/exec/llama_toolgrammar.h). An interface so the
// sampler stays device-free and testable on synthetic rows; the llama backend
// supplies the grammar. The sampler drives it as llama.cpp's
// common_sampler_sample does (common/sampling.cpp at the pin): draw first,
// test the drawn token alone, and only when it is rejected mask the row and
// draw again; every generated token is accepted into it.
class TokenConstraint {
public:
    virtual ~TokenConstraint() = default;
    // May `token` come next? Cheap: one token.
    virtual bool allows(int token) = 0;
    // Set every logit the constraint rejects to -inf (the whole row).
    virtual void mask(float* logits, size_t vocab) = 0;
    // `token` was generated; advance.
    virtual void accept(int token) = 0;

    // clean-turns review B4: how many times this request's constraint lost
    // sync with the generated text -- the moment it refuses to trust its own
    // engine and the rest of the request runs unconstrained (fail open, so a
    // broken grammar cannot strand the turn at the length cap). The backend
    // copies the count into the per-request stats line, so the operator sees
    // it in the request's own numbers without hunting through the logs.
    int desync() const { return desyncs_; }

    // One desync per request, counted once: the helper latches, not the
    // callers. A second loss of sync in the same request still re-logs (the
    // error line is the incident record) but does not re-count, so the stats
    // line stays honest no matter how many paths record through it. Public,
    // not protected: the sampler's mask guard records a thrown mask through
    // it too -- the constraint's own guard (like allows') handles its engine,
    // and the sampler's backstop handles whatever constraint throws anyway.
    void note_desync(int token, const std::string& why);

protected:
    int desyncs_ = 0;
    bool note_desynced_ = false;
};

class Sampler {
public:
    Sampler(const SamplerParams& params, uint64_t seed);

    // The prompt counts toward `repetition_penalty` but NOT toward
    // `presence_penalty` or `frequency_penalty`: OpenAI and vLLM apply those
    // two over generated tokens only. Counting an 8k prompt into them would
    // dock every common word before the model has said anything.
    void set_prompt(const std::vector<int>& prompt);

    // Records a token the model actually generated.
    void observe(int token);

    // clean-turns: the think-block mask. `start_in_think_block` says whether
    // the template opened the block before the first token (the ChatML form
    // writes the opener as the lead of the assistant turn, so the block is open
    // before the model has produced anything). `observe_think_text` feeds the
    // decoded pieces in generation order so the mask clears when the block
    // closes, and engages when the model opens the block itself. Both are
    // inert unless params.think_eog_mask or params.tool_grammar is set (the
    // grammar needs the block's state; only think_eog_mask masks anything).
    void start_in_think_block(bool open);
    void observe_think_text(std::string_view piece);
    bool think_block_open() const { return think_state_ == ThinkState::thinking; }

    // Replace which tokens and tags spell the block. The params carry the
    // switch; only the backend knows the vocabulary, so it hands the ids in
    // here once per request.
    void set_think_mask_ids(ThinkBlockIds ids);

    // clean-turns step 2: the tool-call grammar, or nullptr. Not owned. It is
    // dormant while a think block is open (the reference's lazy grammar is not
    // applied while the reasoning budget sampler counts, common/sampling.cpp
    // grammar_should_apply), so it needs the block's state: the backend feeds
    // observe_think_text whenever params.tool_grammar is set, mask or not.
    void set_constraint(TokenConstraint* c) { constraint_ = c; }

    // Picks the next token from one row of logits. The row is scratch space and
    // is modified in place.
    int sample(float* logits, size_t vocab);

    uint64_t seed() const { return seed_; }

    // Exposed for tests: the pure argmax the greedy path must agree with.
    static int argmax(const float* logits, size_t vocab);

private:
    int    draw(float* logits, size_t vocab);
    bool   tracks_think() const { return params_.think_eog_mask || params_.tool_grammar; }
    void   apply_penalties(float* logits, size_t vocab) const;
    void   mask_illegal_while_thinking(float* logits, size_t vocab) const;
    void   measure_think_tags();
    size_t collect_candidates(const float* logits, size_t vocab, bool full);

    // Where the block is in the generation. undecided: the opening is not
    // settled yet and the lead is still examined; thinking: the block is open
    // and the tail is watched for the closer; content: settled (the block
    // closed, or never opened) and the lead is never examined again for the
    // rest of the turn. The mask and the grammar's dormancy both read
    // "thinking"; nothing else can be undone.
    enum class ThinkState { undecided, thinking, content };

    SamplerParams   params_;
    uint64_t        seed_;
    std::mt19937_64 rng_;

    std::unordered_map<int, int> seen_;       // prompt + generated
    std::unordered_map<int, int> generated_;  // generated only

    std::vector<std::pair<float, int>> scratch_;

    // The mask's state (see ThinkState), plus the two text windows the states
    // examine. Both are bounded by the longest tag, so they cost nothing per
    // step: the lead holds the generation's first non-blank characters until
    // the opening is settled, the tail the newest characters to see a closer
    // that straddles two pieces.
    ThinkState  think_state_ = ThinkState::undecided;
    std::string think_tail_;   // the newest characters, to see a closer land
    std::string think_lead_;   // the first non-blank characters, until the
                               // block's opening is settled
    size_t      think_max_tag_ = 0;

    // The closer guard's once-latch (review B3): the closer id was found in
    // the masked set and left in the row so the block can close; the warn is
    // printed once per request, not once per step.
    mutable bool closer_stripped_ = false;

    TokenConstraint* constraint_ = nullptr;
};

}  // namespace lgc
