#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// DESIGN.md §3.6. Greedy, temperature, top-k, top-p, min-p, repetition
// penalty (min-p since 0.5.9, for role play: operator, 2026-10-04). Defaults come from the served GGUF and the operator flags;
// explicit request fields always win.
namespace lgc {

// loop-guard: detect a reasoning loop and escalate (nudge, slap, abort). The
// rungs are named so a request can pick one; Off is today's behaviour exactly.
// Detect counts and logs only. Nudge injects the fixed sentence once. Slap
// forces the think-block's closer at the detection moment (the rung the
// operator's evidence points at). Ladder runs nudge, then a slap after the
// grace, then an abort if the answer phase itself loops.
enum class LoopGuardMode { Off, Detect, Nudge, Slap, Ladder };

const char* loop_guard_name(LoopGuardMode m);
std::optional<LoopGuardMode> loop_guard_parse(std::string_view name);

// The fixed sentence the nudge rung injects, as if the model had generated it.
// The backend tokenises it once per request; the sampler only forces the ids.
inline constexpr const char* kLoopGuardNudge =
    "I am repeating myself. I will decide now and write the final answer.";

// The sampler regime a request starts from, before its own fields apply.
struct SamplerDefaults {
    float temperature        = 0.7f;
    float top_p              = 0.8f;
    int   top_k              = 20;
    float min_p              = 0.0f;
    float repetition_penalty = 1.05f;
    float presence_penalty   = 0.0f;

    // The non-thinking row, set by the operator's --nothink-* flags. It
    // applies only to a request that resolves to thinking off, and only when
    // at least one such flag is set (nothink_row). An unset field falls back
    // to the normal row above; a request's own fields still win over both.
    // No --nothink-* flag means the normal row for every request, exactly as
    // before.
    std::optional<float> nothink_temperature;
    std::optional<float> nothink_top_p;
    std::optional<int>   nothink_top_k;
    std::optional<float> nothink_min_p;
    std::optional<float> nothink_presence_penalty;
    bool nothink_row = false;   // at least one --nothink-* flag set it

    // clean-turns: the operator's switch for the think-block mask. Off keeps
    // today's behaviour; the artifact cannot decide it, so it is not read from
    // the GGUF.
    bool think_eog_mask = false;

    // clean-turns step 2: the operator's switch for the tool-call grammar.
    // Off keeps today's behaviour; on, a request that declared tools is
    // constrained to a well-formed call. A no-op when no tools are declared.
    bool tool_grammar = false;

    // loop-guard: the ladder's switch and its thresholds. Off is today's
    // behaviour, byte-identical: no detector, no forced tokens, no new
    // finish reason. window counts the recent reasoning lines (> 20 chars),
    // threshold is the repeat share that trips the detector, grace is how
    // many more lines after a nudge the ladder waits before the slap.
    LoopGuardMode loop_guard = LoopGuardMode::Off;
    int   loop_guard_window    = 200;
    float loop_guard_threshold = 0.8f;
    int   loop_guard_grace     = 40;

    // llava: the answer row -- the sampler values used for tokens drawn after
    // the think block has closed in a thinking turn. Each field is only the
    // ones the operator set; an unset field keeps the thinking row's value.
    // No field set at all means the switch is off, byte-identical to before.
    std::optional<float> answer_temperature;
    std::optional<float> answer_top_p;
    std::optional<int>   answer_top_k;
    std::optional<float> answer_min_p;
    std::optional<float> answer_presence_penalty;
    std::optional<float> answer_frequency_penalty;

    // "provisional" -- the Qwen3-family card values, carried because no
    //                  GGUF key is read yet; /props reports this so nobody
    //                  mistakes an inherited default for a measured one.
    // "operator"    -- at least one --temp/--top-p/... flag set it.
    std::string provenance = "provisional";
};

// What spells the think block, for the mask below. The ids come from the
// vocabulary; the tag strings come from think_tag_pairs() in reasoning.h,
// which is the only place they are written out anywhere in this tree.
struct ThinkBlockIds {
    // One entry per tag pair, aligned with open_tags/close_tags below:
    // {open_id, close_id}, each -1 when that tag is not one single special
    // token in the vocabulary. The state machine needs the PAIRING (a block
    // opened by one pair's opener is closed by that pair's closer); the
    // flattened open_ids/close_ids the mask reads carry the same ids as sets.
    std::vector<std::pair<int, int>> open_close_ids;
    std::vector<int> eog;        // the vocabulary's end-of-generation ids: the
                                 // block is not a place to end a turn. Together
                                 // with the request's own stop ids (SamplerParams::
                                 // stop_token_ids, same field the decode loop's
                                 // stop check reads) this is the set the mask
                                 // holds back while the block is open
    std::vector<int> open_ids;   // the block's opening marker id: a second one
                                 // while the block is open is the same defect
                                 // one step later
    std::vector<int> close_ids;  // the block's closer id, when the vocabulary
                                 // spells it as one added token: the mask must
                                 // never kill it (emitting it is what lets the
                                 // block close), so the mask's guard leaves it
                                 // out of the masked set in every case
    std::vector<std::string> close_tags;  // the closers, matched as TEXT: a
                                          // model can spell them as ordinary
                                          // text (measured on the coder
                                          // artifact), so they cannot be
                                          // assumed to be single ids
    std::vector<std::string> open_tags;   // likewise the openers, for a model
                                          // that opens the block itself
};

struct SamplerParams {
    float    temperature        = 0.7f;
    float    top_p              = 0.8f;
    int      top_k              = 20;
    // llama.cpp's min-p: keep the tokens with p >= min_p * p_max, on the
    // logits before temperature (0 disables)
    float    min_p              = 0.0f;
    float    repetition_penalty = 1.05f;
    float    presence_penalty   = 0.0f;
    float    frequency_penalty  = 0.0f;
    uint64_t seed               = 0;
    bool     seeded             = false;
    int      max_tokens         = -1;  // -1: until EOS or the context limit
    bool     ignore_eos         = false;

    std::vector<std::string> stop;
    std::vector<int>         stop_token_ids;

    // clean-turns: while a think block is open, the end-of-turn token is not a
    // legal next token. Off keeps today's behaviour exactly. See ThinkBlockIds.
    bool think_eog_mask = false;
    ThinkBlockIds think_ids;

    // clean-turns step 2: constrain the tool-call form to the declared schemas
    // with a GBNF grammar, so a call the model starts is one it finishes. Off
    // keeps today's behaviour exactly; on, the backend builds the grammar from
    // the request's ToolSchemas and drives it around the sampler. The grammar
    // is a no-op when the request declared no tools.
    bool tool_grammar = false;

    // loop-guard: the switch, its thresholds and the tokenised nudge sentence.
    // Off keeps today's behaviour exactly. The sampler forces a token by id at
    // a draw boundary: for the nudge, each id of the sentence in turn; for the
    // slap, the think-block's closer id once. A forced token flows through the
    // caller's emit/observe/decode path like any drawn token, so it lands in
    // the history, the KV, reasoning_content and the stream.
    LoopGuardMode loop_guard = LoopGuardMode::Off;
    int   loop_guard_window    = 200;
    float loop_guard_threshold = 0.8f;
    int   loop_guard_grace     = 40;
    std::vector<int> loop_guard_nudge_ids;   // kLoopGuardNudge, tokenised

    // top_k == 1 collapses the distribution just as temperature 0 does; both
    // must take the argmax path, or "greedy" means two different things.
    bool greedy() const { return temperature <= 0.0f || top_k == 1; }

    // llava: the answer row after the think block, per field. Unset fields
    // keep the thinking row's value; `answer_configured()` is what arms the
    // switch. `thinking_turn` is the request's enable_thinking after the
    // operator default (the API sets it); a turn that did not declare thinking
    // never switches, whatever the model does with tags.
    std::optional<float> answer_temperature;
    std::optional<float> answer_top_p;
    std::optional<int>   answer_top_k;
    std::optional<float> answer_min_p;
    // llava: the card's non-thinking row also carries a presence penalty, so
    // the answer row does too. Unset falls back to the thinking row's value;
    // the penalty's history is the answer's own (see Sampler::observe).
    std::optional<float> answer_presence_penalty;
    std::optional<float> answer_frequency_penalty;
    bool                 thinking_turn = false;

    bool answer_configured() const {
        return answer_temperature || answer_top_p || answer_top_k || answer_min_p ||
               answer_presence_penalty || answer_frequency_penalty;
    }
};

struct SamplerOverrides {
    // The operator layer's transport for the non-thinking row (--nothink-*);
    // a request never sets these, and sampler_apply ignores them.
    std::optional<float>                    nothink_temperature;
    std::optional<float>                    nothink_top_p;
    std::optional<int>                      nothink_top_k;
    std::optional<float>                    nothink_min_p;
    std::optional<float>                    nothink_presence_penalty;

    std::optional<float>                    temperature;
    std::optional<float>                    top_p;
    std::optional<int>                      top_k;
    std::optional<float>                    min_p;
    std::optional<float>                    repetition_penalty;
    std::optional<float>                    presence_penalty;
    std::optional<float>                    frequency_penalty;
    std::optional<uint64_t>                 seed;
    std::optional<int>                      max_tokens;
    std::optional<bool>                     ignore_eos;
    std::optional<std::vector<std::string>> stop;
    std::optional<std::vector<int>>         stop_token_ids;
    std::optional<bool>                     think_eog_mask;
    std::optional<bool>                     tool_grammar;
    // loop-guard: the request's own rung for this request only. Absent keeps
    // the operator's switch; the window/threshold/grace stay operator-side.
    std::optional<LoopGuardMode>            loop_guard;
    // The operator layer carries these; a request parser never fills them.
    std::optional<int>                      loop_guard_window;
    std::optional<float>                    loop_guard_threshold;
    std::optional<int>                      loop_guard_grace;
    // llava: the request's answer row fields, for an A/B in one process.
    // Absent keeps the operator's setting; a named field wins for that request.
    std::optional<float>                    answer_temperature;
    std::optional<float>                    answer_top_p;
    std::optional<int>                      answer_top_k;
    std::optional<float>                    answer_min_p;
    std::optional<float>                    answer_presence_penalty;
    std::optional<float>                    answer_frequency_penalty;

    bool any() const;
};

// The row a request starts from: the non-thinking row when `nothink` is true
// and the operator configured one (SamplerDefaults::nothink_row), else the
// normal row. Unset nothink fields fall back to the normal row.
SamplerParams sampler_from_defaults(const SamplerDefaults& d, bool nothink = false);

// The operator layer (config flags): overrides only the fields it sets and
// marks the defaults' provenance "operator". Same validation as a request,
// so an out-of-range flag fails at boot with the same message a client would
// get at 400.
std::optional<std::string> sampler_defaults_apply(SamplerDefaults& d, const SamplerOverrides& o);

// Applies the overrides in place. Returns an error message when a field is out
// of range, in which case `p` is left untouched.
std::optional<std::string> sampler_apply(SamplerParams& p, const SamplerOverrides& o);

}  // namespace lgc
