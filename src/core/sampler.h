#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "core/sampling.h"

// DESIGN.md §3.6: greedy, temperature, top-k, top-p, repetition penalty.
// Nothing else in v1, and nothing silent — a request that asks for temperature
// gets temperature, or an error, never argmax wearing a temperature label.
//
// Kept out of the backend so it can be tested on a machine with no GPU: given
// the same logits and seed it must produce the same token, every time.
namespace lgc {

class Sampler {
public:
    Sampler(const SamplerParams& params, uint64_t seed);

    // Picks the next token from one row of logits. `history` is the tokens
    // generated so far plus the prompt, for the penalties. The row is scratch
    // space and is modified in place.
    int sample(float* logits, size_t vocab, const std::vector<int>& history);

    uint64_t seed() const { return seed_; }

    // Exposed for tests: the pure argmax the greedy path must agree with.
    static int argmax(const float* logits, size_t vocab);

private:
    void apply_penalties(float* logits, size_t vocab, const std::vector<int>& history) const;

    SamplerParams params_;
    uint64_t      seed_;
    std::mt19937_64 rng_;
    std::vector<std::pair<float, int>> scratch_;
};

}  // namespace lgc
