#pragma once

#include <cstdint>
#include <random>
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

    // Picks the next token from one row of logits. The row is scratch space and
    // is modified in place.
    int sample(float* logits, size_t vocab);

    uint64_t seed() const { return seed_; }

    // Exposed for tests: the pure argmax the greedy path must agree with.
    static int argmax(const float* logits, size_t vocab);

private:
    void   apply_penalties(float* logits, size_t vocab) const;
    size_t collect_candidates(const float* logits, size_t vocab);

    SamplerParams   params_;
    uint64_t        seed_;
    std::mt19937_64 rng_;

    std::unordered_map<int, int> seen_;       // prompt + generated
    std::unordered_map<int, int> generated_;  // generated only

    std::vector<std::pair<float, int>> scratch_;
};

}  // namespace lgc
