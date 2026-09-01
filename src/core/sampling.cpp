#include "core/sampling.h"

#include "util/log.h"

namespace lgc {

bool SamplerOverrides::any() const {
    return temperature || top_p || top_k || repetition_penalty || presence_penalty ||
           frequency_penalty || seed || max_tokens || ignore_eos || stop || stop_token_ids;
}

SamplerParams sampler_from_defaults(const SamplerDefaults& d) {
    SamplerParams p;
    p.temperature        = d.temperature;
    p.top_p              = d.top_p;
    p.top_k              = d.top_k;
    p.repetition_penalty = d.repetition_penalty;
    p.presence_penalty   = d.presence_penalty;
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
    if (o.repetition_penalty && *o.repetition_penalty <= 0.0f) {
        return log::format("repetition_penalty must be > 0, got %g", *o.repetition_penalty);
    }
    if (o.presence_penalty && (*o.presence_penalty < -2.0f || *o.presence_penalty > 2.0f)) {
        return log::format("presence_penalty must be in [-2, 2], got %g", *o.presence_penalty);
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
    if (o.repetition_penalty) p.repetition_penalty = *o.repetition_penalty;
    if (o.presence_penalty)   p.presence_penalty   = *o.presence_penalty;
    if (o.frequency_penalty)  p.frequency_penalty  = *o.frequency_penalty;
    if (o.max_tokens)         p.max_tokens         = *o.max_tokens;
    if (o.ignore_eos)         p.ignore_eos         = *o.ignore_eos;
    if (o.stop)               p.stop               = *o.stop;
    if (o.stop_token_ids)     p.stop_token_ids     = *o.stop_token_ids;
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
    if (o.repetition_penalty) d.repetition_penalty = *o.repetition_penalty;
    if (o.presence_penalty)   d.presence_penalty   = *o.presence_penalty;
    if (o.temperature || o.top_p || o.top_k || o.repetition_penalty || o.presence_penalty) {
        d.provenance = "operator";
    }
    return std::nullopt;
}

}  // namespace lgc
