#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// DESIGN.md §3.6. Greedy, temperature, top-k, top-p, min-p, repetition
// penalty (min-p since 0.5.9, for role play: operator, 2026-10-04). Defaults come from the served GGUF and the operator flags;
// explicit request fields always win.
namespace lgc {

// The sampler regime a request starts from, before its own fields apply.
struct SamplerDefaults {
    float temperature        = 0.7f;
    float top_p              = 0.8f;
    int   top_k              = 20;
    float min_p              = 0.0f;
    float repetition_penalty = 1.05f;
    float presence_penalty   = 0.0f;

    // "provisional" -- the Qwen3-family card values, carried because no
    //                  GGUF key is read yet; /props reports this so nobody
    //                  mistakes an inherited default for a measured one.
    // "operator"    -- at least one --temp/--top-p/... flag set it.
    std::string provenance = "provisional";
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

    // top_k == 1 collapses the distribution just as temperature 0 does; both
    // must take the argmax path, or "greedy" means two different things.
    bool greedy() const { return temperature <= 0.0f || top_k == 1; }
};

struct SamplerOverrides {
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

    bool any() const;
};

SamplerParams sampler_from_defaults(const SamplerDefaults& d);

// The operator layer (config flags): overrides only the fields it sets and
// marks the defaults' provenance "operator". Same validation as a request,
// so an out-of-range flag fails at boot with the same message a client would
// get at 400.
std::optional<std::string> sampler_defaults_apply(SamplerDefaults& d, const SamplerOverrides& o);

// Applies the overrides in place. Returns an error message when a field is out
// of range, in which case `p` is left untouched.
std::optional<std::string> sampler_apply(SamplerParams& p, const SamplerOverrides& o);

}  // namespace lgc
