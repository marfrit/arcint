#include "core/sampling.h"

#include "util/log.h"

namespace lgc {

const char* loop_guard_name(LoopGuardMode m) {
    switch (m) {
        case LoopGuardMode::Off:    return "off";
        case LoopGuardMode::Detect: return "detect";
        case LoopGuardMode::Nudge:  return "nudge";
        case LoopGuardMode::Slap:   return "slap";
        case LoopGuardMode::Door:   return "door";
        case LoopGuardMode::Ladder: return "ladder";
        case LoopGuardMode::LadderNoDoor: return "ladder-nodoor";
    }
    return "off";
}

std::optional<LoopGuardMode> loop_guard_parse(std::string_view name) {
    if (name == "off")    return LoopGuardMode::Off;
    if (name == "detect") return LoopGuardMode::Detect;
    if (name == "nudge")  return LoopGuardMode::Nudge;
    if (name == "slap")   return LoopGuardMode::Slap;
    if (name == "door")   return LoopGuardMode::Door;
    if (name == "ladder") return LoopGuardMode::Ladder;
    if (name == "ladder-nodoor") return LoopGuardMode::LadderNoDoor;
    return std::nullopt;
}

const char* loop_guard_match_name(LoopGuardMatch m) {
    switch (m) {
        case LoopGuardMatch::Exact: return "exact";
        case LoopGuardMatch::Near:  return "near";
    }
    return "near";
}

std::optional<LoopGuardMatch> loop_guard_match_parse(std::string_view name) {
    if (name == "exact") return LoopGuardMatch::Exact;
    if (name == "near")  return LoopGuardMatch::Near;
    return std::nullopt;
}

bool SamplerOverrides::any() const {
    return temperature || top_p || top_k || min_p || repetition_penalty || presence_penalty ||
           frequency_penalty || seed || max_tokens || ignore_eos || stop || stop_token_ids ||
           think_eog_mask || tool_grammar || think_ctx_reserve || loop_guard || loop_guard_match ||
           loop_guard_window || loop_guard_threshold || loop_guard_grace || loop_guard_door_grace ||
           answer_temperature || answer_top_p || answer_top_k ||
           answer_min_p || answer_presence_penalty || answer_frequency_penalty ||
           nothink_temperature || nothink_top_p || nothink_top_k || nothink_min_p ||
           nothink_presence_penalty;
}

SamplerParams sampler_from_defaults(const SamplerDefaults& d, bool nothink) {
    SamplerParams p;
    p.temperature        = d.temperature;
    p.top_p              = d.top_p;
    p.top_k              = d.top_k;
    p.min_p              = d.min_p;
    p.repetition_penalty = d.repetition_penalty;
    p.presence_penalty   = d.presence_penalty;
    if (nothink && d.nothink_row) {
        if (d.nothink_temperature)        p.temperature        = *d.nothink_temperature;
        if (d.nothink_top_p)              p.top_p              = *d.nothink_top_p;
        if (d.nothink_top_k)              p.top_k              = *d.nothink_top_k;
        if (d.nothink_min_p)              p.min_p              = *d.nothink_min_p;
        if (d.nothink_presence_penalty)   p.presence_penalty   = *d.nothink_presence_penalty;
    }
    p.think_eog_mask     = d.think_eog_mask;
    p.tool_grammar       = d.tool_grammar;
    p.think_ctx_reserve  = d.think_ctx_reserve;
    p.loop_guard         = d.loop_guard;
    p.loop_guard_match   = d.loop_guard_match;
    p.loop_guard_window    = d.loop_guard_window;
    p.loop_guard_threshold = d.loop_guard_threshold;
    p.loop_guard_grace      = d.loop_guard_grace;
    p.loop_guard_door_grace = d.loop_guard_door_grace;
    p.answer_temperature = d.answer_temperature;
    p.answer_top_p       = d.answer_top_p;
    p.answer_top_k       = d.answer_top_k;
    p.answer_min_p       = d.answer_min_p;
    p.answer_presence_penalty  = d.answer_presence_penalty;
    p.answer_frequency_penalty = d.answer_frequency_penalty;
    return p;
}

namespace {

// Shared between the request path and the operator-flag path: one source of
// range truth, one wording for the refusal.
std::optional<std::string> sampler_validate(const SamplerOverrides& o) {
    if (o.temperature && (*o.temperature < 0.0f || *o.temperature > 2.0f)) {
        return log::format("temperature must be in [0, 2], got %g", *o.temperature);
    }
    if (o.top_p && (*o.top_p <= 0.0f || *o.top_p > 1.0f)) {
        return log::format("top_p must be in (0, 1], got %g", *o.top_p);
    }
    if (o.top_k && *o.top_k < 0) {
        return log::format("top_k must be >= 0 (0 disables), got %d", *o.top_k);
    }
    if (o.min_p && (*o.min_p < 0.0f || *o.min_p > 1.0f)) {
        return log::format("min_p must be in [0, 1] (0 disables), got %g", *o.min_p);
    }
    if (o.repetition_penalty && *o.repetition_penalty <= 0.0f) {
        return log::format("repetition_penalty must be > 0, got %g", *o.repetition_penalty);
    }
    if (o.presence_penalty && (*o.presence_penalty < -2.0f || *o.presence_penalty > 2.0f)) {
        return log::format("presence_penalty must be in [-2, 2], got %g", *o.presence_penalty);
    }
    // The non-thinking row is validated by the same rules as its normal-row
    // twin, so a bad --nothink-* flag fails at boot with the same wording.
    if (o.nothink_temperature && (*o.nothink_temperature < 0.0f || *o.nothink_temperature > 2.0f)) {
        return log::format("nothink temperature must be in [0, 2], got %g", *o.nothink_temperature);
    }
    if (o.nothink_top_p && (*o.nothink_top_p <= 0.0f || *o.nothink_top_p > 1.0f)) {
        return log::format("nothink top_p must be in (0, 1], got %g", *o.nothink_top_p);
    }
    if (o.nothink_top_k && *o.nothink_top_k < 0) {
        return log::format("nothink top_k must be >= 0 (0 disables), got %d", *o.nothink_top_k);
    }
    if (o.nothink_min_p && (*o.nothink_min_p < 0.0f || *o.nothink_min_p > 1.0f)) {
        return log::format("nothink min_p must be in [0, 1] (0 disables), got %g", *o.nothink_min_p);
    }
    if (o.nothink_presence_penalty &&
        (*o.nothink_presence_penalty < -2.0f || *o.nothink_presence_penalty > 2.0f)) {
        return log::format("nothink presence_penalty must be in [-2, 2], got %g",
                           *o.nothink_presence_penalty);
    }
    if (o.frequency_penalty && (*o.frequency_penalty < -2.0f || *o.frequency_penalty > 2.0f)) {
        return log::format("frequency_penalty must be in [-2, 2], got %g", *o.frequency_penalty);
    }
    if (o.max_tokens && *o.max_tokens < 1) {
        return log::format("max_tokens must be >= 1, got %d", *o.max_tokens);
    }
    if (o.think_ctx_reserve && *o.think_ctx_reserve < 0) {
        return log::format("think_ctx_reserve must be >= 0 (0 disables), got %d", *o.think_ctx_reserve);
    }
    // llava: the answer row is validated like the thinking row, so a flag or a
    // request field out of range is the same refusal.
    if (o.answer_temperature && (*o.answer_temperature < 0.0f || *o.answer_temperature > 2.0f)) {
        return log::format("answer_temperature must be in [0, 2], got %g", *o.answer_temperature);
    }
    if (o.answer_top_p && (*o.answer_top_p <= 0.0f || *o.answer_top_p > 1.0f)) {
        return log::format("answer_top_p must be in (0, 1], got %g", *o.answer_top_p);
    }
    if (o.answer_top_k && *o.answer_top_k < 0) {
        return log::format("answer_top_k must be >= 0 (0 disables), got %d", *o.answer_top_k);
    }
    if (o.answer_min_p && (*o.answer_min_p < 0.0f || *o.answer_min_p > 1.0f)) {
        return log::format("answer_min_p must be in [0, 1] (0 disables), got %g", *o.answer_min_p);
    }
    if (o.answer_presence_penalty &&
        (*o.answer_presence_penalty < -2.0f || *o.answer_presence_penalty > 2.0f)) {
        return log::format("answer_presence_penalty must be in [-2, 2], got %g", *o.answer_presence_penalty);
    }
    if (o.answer_frequency_penalty &&
        (*o.answer_frequency_penalty < -2.0f || *o.answer_frequency_penalty > 2.0f)) {
        return log::format("answer_frequency_penalty must be in [-2, 2], got %g", *o.answer_frequency_penalty);
    }
    // loop-guard's thresholds: a window of 0 or a threshold outside [0, 1]
    // cannot describe a share, so the refusal is the same for a flag and a
    // request.
    if (o.loop_guard_window && *o.loop_guard_window < 1) {
        return log::format("loop_guard_window must be >= 1, got %d", *o.loop_guard_window);
    }
    if (o.loop_guard_threshold && (*o.loop_guard_threshold < 0.0f || *o.loop_guard_threshold > 1.0f)) {
        return log::format("loop_guard_threshold must be in [0, 1], got %g", *o.loop_guard_threshold);
    }
    if (o.loop_guard_grace && *o.loop_guard_grace < 0) {
        return log::format("loop_guard_grace must be >= 0, got %d", *o.loop_guard_grace);
    }
    if (o.loop_guard_door_grace && *o.loop_guard_door_grace < 0) {
        return log::format("loop_guard_door_grace must be >= 0, got %d", *o.loop_guard_door_grace);
    }
    return std::nullopt;
}

}  // namespace

std::optional<std::string> sampler_apply(SamplerParams& p, const SamplerOverrides& o) {
    // Validate everything before mutating, so a rejected request cannot leave a
    // half-applied sampler behind.
    if (auto err = sampler_validate(o)) return err;

    if (o.temperature)        p.temperature        = *o.temperature;
    if (o.top_p)              p.top_p              = *o.top_p;
    if (o.top_k)              p.top_k              = *o.top_k;
    if (o.min_p)              p.min_p              = *o.min_p;
    if (o.repetition_penalty) p.repetition_penalty = *o.repetition_penalty;
    if (o.presence_penalty)   p.presence_penalty   = *o.presence_penalty;
    if (o.frequency_penalty)  p.frequency_penalty  = *o.frequency_penalty;
    if (o.max_tokens)         p.max_tokens         = *o.max_tokens;
    if (o.ignore_eos)         p.ignore_eos         = *o.ignore_eos;
    if (o.stop)               p.stop               = *o.stop;
    if (o.stop_token_ids)     p.stop_token_ids     = *o.stop_token_ids;
    // Symmetric, like every other field: a request that names it gets what it
    // names. A client that sends nothing gets the operator's setting, and that
    // is where the fix reaches the real agent sessions — they send nothing.
    // Making the operator's on un-vetoable would be a silent override (§3.6).
    if (o.think_eog_mask) p.think_eog_mask = *o.think_eog_mask;
    if (o.tool_grammar)   p.tool_grammar   = *o.tool_grammar;
    if (o.think_ctx_reserve) p.think_ctx_reserve = *o.think_ctx_reserve;
    if (o.loop_guard)     p.loop_guard     = *o.loop_guard;
    if (o.loop_guard_match) p.loop_guard_match = *o.loop_guard_match;
    // llava: the request's answer row fields override the operator's, for that
    // request only (a client A/B without a restart). Absent keeps the default.
    if (o.answer_temperature) p.answer_temperature = *o.answer_temperature;
    if (o.answer_top_p)       p.answer_top_p       = *o.answer_top_p;
    if (o.answer_top_k)       p.answer_top_k       = *o.answer_top_k;
    if (o.answer_min_p)       p.answer_min_p       = *o.answer_min_p;
    if (o.answer_presence_penalty)  p.answer_presence_penalty  = *o.answer_presence_penalty;
    if (o.answer_frequency_penalty) p.answer_frequency_penalty = *o.answer_frequency_penalty;
    if (o.seed) {
        p.seed   = *o.seed;
        p.seeded = true;
    }

    return std::nullopt;
}

std::optional<std::string> sampler_defaults_apply(SamplerDefaults& d, const SamplerOverrides& o) {
    if (auto err = sampler_validate(o)) return err;

    if (o.temperature)        d.temperature        = *o.temperature;
    if (o.top_p)              d.top_p              = *o.top_p;
    if (o.top_k)              d.top_k              = *o.top_k;
    if (o.min_p)              d.min_p              = *o.min_p;
    if (o.repetition_penalty) d.repetition_penalty = *o.repetition_penalty;
    if (o.presence_penalty)   d.presence_penalty   = *o.presence_penalty;
    if (o.nothink_temperature)        d.nothink_temperature        = *o.nothink_temperature;
    if (o.nothink_top_p)              d.nothink_top_p              = *o.nothink_top_p;
    if (o.nothink_top_k)              d.nothink_top_k              = *o.nothink_top_k;
    if (o.nothink_min_p)              d.nothink_min_p              = *o.nothink_min_p;
    if (o.nothink_presence_penalty)   d.nothink_presence_penalty   = *o.nothink_presence_penalty;
    const bool nothink_set = o.nothink_temperature || o.nothink_top_p || o.nothink_top_k ||
                             o.nothink_min_p || o.nothink_presence_penalty;
    if (nothink_set) d.nothink_row = true;
    if (o.think_eog_mask)     d.think_eog_mask     = *o.think_eog_mask;
    if (o.tool_grammar)       d.tool_grammar       = *o.tool_grammar;
    if (o.think_ctx_reserve)  d.think_ctx_reserve  = *o.think_ctx_reserve;
    if (o.loop_guard)         d.loop_guard         = *o.loop_guard;
    if (o.loop_guard_match)   d.loop_guard_match   = *o.loop_guard_match;
    if (o.loop_guard_window)    d.loop_guard_window    = *o.loop_guard_window;
    if (o.loop_guard_threshold) d.loop_guard_threshold = *o.loop_guard_threshold;
    if (o.loop_guard_grace)     d.loop_guard_grace     = *o.loop_guard_grace;
    if (o.loop_guard_door_grace) d.loop_guard_door_grace = *o.loop_guard_door_grace;
    if (o.answer_temperature) d.answer_temperature = *o.answer_temperature;
    if (o.answer_top_p)       d.answer_top_p       = *o.answer_top_p;
    if (o.answer_top_k)       d.answer_top_k       = *o.answer_top_k;
    if (o.answer_min_p)       d.answer_min_p       = *o.answer_min_p;
    if (o.answer_presence_penalty)  d.answer_presence_penalty  = *o.answer_presence_penalty;
    if (o.answer_frequency_penalty) d.answer_frequency_penalty = *o.answer_frequency_penalty;
    if (o.temperature || o.top_p || o.top_k || o.min_p || o.repetition_penalty || o.presence_penalty ||
        o.think_eog_mask || o.tool_grammar || o.think_ctx_reserve || o.loop_guard || o.loop_guard_match ||
        o.loop_guard_window || o.loop_guard_threshold ||
        o.loop_guard_grace || o.loop_guard_door_grace || o.answer_temperature || o.answer_top_p || o.answer_top_k ||
        o.answer_min_p || o.answer_presence_penalty || o.answer_frequency_penalty || nothink_set) {
        d.provenance = "operator";
    }
    return std::nullopt;
}

}  // namespace lgc
