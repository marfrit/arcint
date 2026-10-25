#pragma once

#include <cstdint>
#include <deque>
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

    // Records a token the model actually generated, with the text that token
    // decoded to. The id and the text are one observation: the think-block
    // tracker decides on the id when the vocabulary spells the tag as one
    // special token (a spelled-out closer is ordinary tokens and does not
    // close the block), and falls back to the text otherwise. `piece` may be
    // empty (a caller that only tracks the id, e.g. the TTS codec loop, which
    // never runs the think-block tracker).
    void observe(int token, std::string_view piece = {});

    // clean-turns: the think-block mask. `start_in_think_block` says whether
    // the template opened the block before the first token (the ChatML form
    // writes the opener as the lead of the assistant turn, so the block is open
    // before the model has produced anything). `observe` feeds the (id, text)
    // pairs in generation order so the mask clears when the block closes, and
    // engages when the model opens the block itself. Both are
    // inert unless params.think_eog_mask or params.tool_grammar is set (the
    // grammar needs the block's state; only think_eog_mask masks anything).
    void start_in_think_block(bool open);
    // Text-only feed, for a caller that has no token id (the tests; a
    // backend always calls observe(id, piece)). It is the text fallback: a
    // known id still governs its role, so ordinary text cannot close or open
    // a block whose tag is one special token.
    void observe_think_text(std::string_view piece);
    // The tracker alone, without the history counts: a decode loop whose
    // history is maintained elsewhere (the speculative lane's verify walk)
    // still needs the emitted token's id and text here.
    void observe_think(int token, std::string_view piece);
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

    // loop-guard: the fixed sentence's ids, tokenised by the backend (which
    // owns the vocabulary). The sampler forces them one a draw, at a token
    // boundary, so the caller's emit/observe/decode path carries them exactly
    // as a drawn token. Empty keeps the ladder's nudge rung inert.
    void set_loop_guard_nudge(std::vector<int> ids) {
        params_.loop_guard_nudge_ids = std::move(ids);
    }
    // loop-guard door: the multi-newline token ids, computed by the backend
    // from its vocabulary's pieces (two or more newline characters and nothing
    // else). Masked while the door rung is open and the block is open.
    void set_loop_guard_door(std::vector<int> ids) {
        params_.loop_guard_door_ids = std::move(ids);
    }
    // The per-request counters, for the stats line and the response object.
    struct LoopGuardCounts {
        int detections = 0;
        int nudges     = 0;
        int slaps      = 0;
        int aborts     = 0;
        // door: the block closed by the model's own draw while the door was
        // open (not by a slap), and the draws at which the door mask was active
        // and a masked token sat in the row's top-k candidates.
        int self_closes = 0;
        int door_masked = 0;
        // gap 2: draws at which an end-of-generation id was a finite candidate
        // the door removed inside the open block (think_eog_mask off).
        int door_eog_masked = 0;
    };
    LoopGuardCounts loop_guard_counts() const {
        return {loop_detections_, loop_nudges_, loop_slaps_, loop_aborts_,
                loop_self_closes_, loop_door_masked_, loop_door_eog_masked_};
    }
    // The answer phase kept looping after a slap: the caller ends the turn on
    // FinishReason::Loop instead of emitting another token. Latching: once set
    // the turn is over.
    bool loop_abort() const { return loop_abort_; }

    // Picks the next token from one row of logits. The row is scratch space and
    // is modified in place.
    int sample(float* logits, size_t vocab);

    // llava: the sampler values this draw would use right now -- the thinking
    // row, or the answer row when the block closed in a thinking turn. The
    // test hook for the switch's cases; it reads no state that sample() does
    // not read, so it is the row the next draw gets.
    struct EffectiveRow {
        float temperature;
        float top_p;
        int   top_k;
        float min_p;
        // llava: the row also decides which presence/frequency values apply
        // and which history they read; a draw after the switch uses the
        // answer row's values over the answer's own history.
        float presence_penalty;
        float frequency_penalty;
        bool  answer_row;
    };
    EffectiveRow effective_row() const;

    uint64_t seed() const { return seed_; }

    // Exposed for tests: the pure argmax the greedy path must agree with.
    static int argmax(const float* logits, size_t vocab);

private:
    int    draw(float* logits, size_t vocab);
    bool   tracks_think() const
        { return params_.think_eog_mask || params_.tool_grammar || params_.answer_configured()
                 || params_.loop_guard != LoopGuardMode::Off; }
    // llava: the answer row applies only in a thinking turn, after a block
    // actually opened and closed. A non-thinking turn, or one whose block
    // never opened, keeps the thinking row.
    bool   use_answer_row() const {
        return params_.answer_configured() && params_.thinking_turn && think_block_ever_open_
               && think_state_ == ThinkState::content;
    }
    // loop-guard: the ladder's state and its per-request counters.
    void   loop_observe_(std::string_view piece);
    void   loop_line_ended_(const std::string& line);
    // near match: the line's hashed character-3-grams, sorted and deduped;
    // compared by Jaccard against the remembered distinct counted lines.
    using LoopGrams = std::vector<uint32_t>;
    LoopGrams loop_grams_(const std::string& line) const;
    bool      loop_near_match_(const LoopGrams& grams) const;
    void   loop_on_detect_(bool content_phase);
    // gap 1: the answer gets a clean detector window at the block's close (by
    // the model's own draw after the door opened, or by the slap's forced
    // closer), so no reasoning line counts against the answer.
    void   loop_start_answer_();
    void   loop_do_nudge_();
    void   loop_do_slap_();
    void   loop_do_abort_();
    void   apply_penalties(float* logits, size_t vocab) const;
    void   mask_illegal_while_thinking(float* logits, size_t vocab) const;
    // loop-guard door: mask the multi-newline ids while the door is open and
    // the block is open. It keeps only the masked ids' logits, for the counter
    // the draw reads once its candidate set exists.
    void   mask_door_while_thinking(float* logits, size_t vocab);
    // loop-guard door counter: was a masked id among the candidates this draw
    // considers? Runs on the candidate set the draw collects, measured against
    // the logits the ids had at mask time. Consumes the stored pair list, so a
    // grammar redraw in the same sample() does not count twice.
    void   count_door_masked_(float floor, bool inclusive);
    // gap 2's counter: the same test for the end-of-generation ids the door
    // removed inside the open block, over the draw's own candidate set.
    void   count_door_eog_masked_(float floor, bool inclusive);
    void   measure_think_tags();
    void   build_think_id_table();
    void   observe_think_(std::string_view piece, int token);
    size_t collect_candidates(const float* logits, size_t vocab, bool full, int top_k);

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
    // llava: the answer row's own history. Empty at the switch; the closer
    // token itself is not counted, every token observed after it is. Read only
    // while the answer row is in effect; before the switch and in a turn that
    // never switches, the two penalty counts read `generated_` as before.
    std::unordered_map<int, int> answer_generated_;
    bool answer_closer_now_ = false;   // the token that closed the block, once

    std::vector<std::pair<float, int>> scratch_;

    // The mask's state (see ThinkState), plus the two text windows the states
    // examine. Both are bounded by the longest tag, so they cost nothing per
    // step: the lead holds the generation's first non-blank characters until
    // the opening is settled, the tail the newest characters to see a closer
    // that straddles two pieces.
    ThinkState  think_state_ = ThinkState::undecided;
    // llava: whether the block ever opened in this turn. The answer row
    // switches only after a block that actually opened; a turn that settled
    // undecided -> content without one keeps the thinking row.
    bool        think_block_ever_open_ = false;
    std::string think_tail_;   // the newest characters, to see a closer land
    std::string think_lead_;   // the first non-blank characters, until the
                               // block's opening is settled
    size_t      think_max_tag_ = 0;   // the longest tag: the closer tail's bound
    size_t      think_max_open_ = 0;  // the longest opener: the lead's bound
                                      // (N1: the look-ahead is bounded by it)

    // The id rule: when the vocabulary spells a tag as one special token the
    // block opens and closes on that id, not on the decoded text (a model that
    // QUOTES the closer spells it from ordinary tokens and produces the same
    // text; the text path read that as the real closer). close_id_ is the id
    // that closes the block that is open: the first close id when the template
    // opened it, the close id paired with the opener that opened it otherwise.
    // -1 means "no id; the text fallback is in force for that role".
    int          think_close_id_ = -1;
    bool         think_has_open_ids_ = false;   // at least one opener is one id
    std::vector<std::pair<int, int>> think_open_to_close_;   // open id -> its close id

    // The closer guard's once-latch (review B3): the closer id was found in
    // the masked set and left in the row so the block can close; the warn is
    // printed once per request, not once per step.
    mutable bool closer_stripped_ = false;

    TokenConstraint* constraint_ = nullptr;

    // loop-guard: the incremental line share. Only lines longer than 20
    // characters count; a line ends at a newline in the decoded text. The
    // window holds the newest `loop_guard_window` such lines, each flagged
    // whether it repeats an earlier line of this turn; the share is the flag's
    // count over the window's size. No whole-text re-scan per token.
    struct LoopLine {
        std::string text;
        bool        repeated = false;
    };
    std::string                       loop_line_;   // the open line's text
    std::deque<LoopLine>              loop_window_;
    std::unordered_map<std::string, int> loop_seen_;   // line -> prior count
    // The near rule's memory: the last kLoopGuardNearMemory distinct counted
    // lines' gram sets, per request. Exact repeats are not stored twice.
    std::deque<LoopGrams>             loop_gram_memory_;
    // Hysteresis: true while the share is below (threshold - 0.1). A
    // detection counts only on a low -> over crossing, so a dip just under the
    // threshold (a nudge's own line) does not count a second loop.
    bool loop_low_       = true;
    bool loop_nudged_    = false;
    bool loop_slapped_   = false;
    bool loop_abort_     = false;
    bool loop_slap_pending_ = false;
    // door: open from the detection moment through nudge and slap until the
    // block closes; the mask reads this and the block's state. `forced_closer`
    // marks the token sample() returned for a slap, so observe_think_ does not
    // count that close as the model's own.
    bool loop_door_open_     = false;
    bool loop_forced_closer_ = false;
    int  loop_since_detect_  = 0;   // counted lines since detection (door grace)
    int  loop_self_closes_   = 0;
    int  loop_door_masked_   = 0;
    int  loop_door_eog_masked_ = 0;
    // The masked ids' logits at mask time (id, logit), not the whole row: the
    // count reads them against the candidate floor the draw's collection gives.
    std::vector<std::pair<int, float>> door_masked_logits_;
    std::vector<std::pair<int, float>> door_eog_masked_logits_;
    int  loop_nudge_cursor_ = -1;   // next nudge id, -1 when none pending
    // Set when sample() returns the last nudge id: observe_think_ clears the
    // partial reasoning line after that piece is fed, so the sentence's tail
    // does not prefix the model's next reasoning line.
    bool loop_nudge_flush_ = false;
    int  loop_since_nudge_  = 0;    // counted lines since the nudge
    int  loop_detections_   = 0;
    int  loop_nudges_       = 0;
    int  loop_slaps_        = 0;
    int  loop_aborts_       = 0;
    mutable bool loop_closer_warned_ = false;
};

}  // namespace lgc
