#include "core/sampling.h"

#include "util/log.h"

namespace lgc {

bool SamplerOverrides::any() const {
    return temperature || top_p || top_k || min_p || repetition_penalty || presence_penalty ||
           frequency_penalty || seed || max_tokens || ignore_eos || stop || stop_token_ids ||
           think_eog_mask || tool_grammar ||
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
    if (o.temperature || o.top_p || o.top_k || o.min_p || o.repetition_penalty || o.presence_penalty ||
        o.think_eog_mask || o.tool_grammar || nothink_set) {
        d.provenance = "operator";
    }
    return std::nullopt;
}

}  // namespace lgc
