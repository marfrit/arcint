#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/model_registry.h"

// DESIGN.md §3.6. Greedy, temperature, top-k, top-p, repetition penalty —
// nothing else in v1. Model-aware defaults come from the allowlist entry;
// explicit request fields always win.
namespace lgc {

struct SamplerParams {
    float    temperature        = 0.7f;
    float    top_p              = 0.8f;
    int      top_k              = 20;
    float    repetition_penalty = 1.05f;
    float    presence_penalty   = 0.0f;
    float    frequency_penalty  = 0.0f;
    uint64_t seed               = 0;
    bool     seeded             = false;
    int      max_tokens         = -1;  // -1: until EOS or the context limit
    bool     ignore_eos         = false;

    std::vector<std::string> stop;
    std::vector<int>         stop_token_ids;

    // top_k == 1 collapses the distribution just as temperature 0 does; both
    // must take the argmax path, or "greedy" means two different things.
    bool greedy() const { return temperature <= 0.0f || top_k == 1; }
};

struct SamplerOverrides {
    std::optional<float>                    temperature;
    std::optional<float>                    top_p;
    std::optional<int>                      top_k;
    std::optional<float>                    repetition_penalty;
    std::optional<float>                    presence_penalty;
    std::optional<float>                    frequency_penalty;
    std::optional<uint64_t>                 seed;
    std::optional<int>                      max_tokens;
    std::optional<bool>                     ignore_eos;
    std::optional<std::vector<std::string>> stop;
    std::optional<std::vector<int>>         stop_token_ids;

    bool any() const;
};

SamplerParams sampler_from_defaults(const SamplerDefaults& d);

// Applies the overrides in place. Returns an error message when a field is out
// of range, in which case `p` is left untouched.
std::optional<std::string> sampler_apply(SamplerParams& p, const SamplerOverrides& o);

}  // namespace lgc
