// Golden token sequences for the sampler, at a fixed seed and a fixed row.
//
// The expected arrays were recorded on the base commit (0e216c84, before the
// no-full-sort change) with ARCINT_SAMPLER_GOLDEN_PRINT=1, which prints the
// sequences instead of comparing them. Only the Sampler API that exists at
// the base commit is used, so the file compiles on the base as well.
//
// Scope on purpose: only the cases that must stay BIT-FOR-BIT unchanged by
// the no-full-sort change (top_k 20, top_p < 1.0, min_p, greedy). A
// no-truncation case would legitimately drift (the tail's relative order may
// differ from the old full sort), so it is not pinned here.
#include "core/sampler.h"
#include "harness.h"

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

using namespace lgc;

namespace {

std::vector<float> golden_row() {
    // Fixed row: 64 tokens. Even tokens are spaced linearly from 0 down to -8
    // (so top_p 0.9, top_k 20 with top_p 0.95 and min_p 0.05 each cut inside
    // the row), odd tokens sit at -30 (never selected).
    std::vector<float> l(64);
    for (size_t i = 0; i < 64; ++i) {
        l[i] = (i % 2 == 0)
            ? -8.0f * static_cast<float>(i / 2) / 31.0f
            : -30.0f;
    }
    return l;
}

std::vector<int> golden_sequence(const SamplerParams& params, uint64_t seed, int n) {
    auto row = golden_row();
    Sampler s(params, seed);
    std::vector<int> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back(s.sample(row.data(), row.size()));
    return out;
}

std::string print(const std::vector<int>& seq) {
    std::string out = "{";
    for (size_t i = 0; i < seq.size(); ++i) {
        if (i != 0) out += ", ";
        out += std::to_string(seq[i]);
    }
    out += "}";
    return out;
}

bool print_requested() {
    const char* e = std::getenv("ARCINT_SAMPLER_GOLDEN_PRINT");
    return e != nullptr && e[0] == '1';
}

void check_sequence(const char* name, const std::vector<int>& expected,
                    const std::vector<int>& actual) {
    if (print_requested()) {
        std::printf("%s: %s\n", name, print(actual).c_str());
        return;
    }
    // An empty expected array is a defect, not a skip: it would pin nothing.
    CHECK(!expected.empty());
    CHECK_EQ(actual, expected);
}

SamplerParams top_k20_params() {
    SamplerParams p;
    p.temperature        = 1.0f;
    p.top_k              = 20;
    p.top_p              = 0.95f;
    p.repetition_penalty = 1.0f;
    return p;
}

SamplerParams top_p09_params() {
    SamplerParams p;
    p.temperature        = 0.8f;
    p.top_k              = 0;
    p.top_p              = 0.9f;
    p.repetition_penalty = 1.0f;
    return p;
}

SamplerParams min_p_params() {
    SamplerParams p;
    p.temperature        = 1.0f;
    p.top_k              = 0;
    p.top_p              = 1.0f;
    p.min_p              = 0.05f;
    p.repetition_penalty = 1.0f;
    return p;
}

SamplerParams greedy_params() {
    SamplerParams p;
    p.temperature        = 0.0f;
    p.top_k              = 0;
    p.top_p              = 1.0f;
    p.repetition_penalty = 1.0f;
    return p;
}

}  // namespace

TEST(golden_top_k20_top_p095) {
    static const std::vector<int> expected = {0, 6, 10, 0, 4, 10, 6, 16, 0, 8, 16, 0, 10, 0, 2, 2, 2, 2, 2, 16, 4, 2, 6, 2, 2, 0, 6, 0, 8, 6, 0, 12, 4, 2, 2, 2, 0, 0, 4, 0, 12, 4, 4, 16, 0, 8, 8, 2, 2, 2, 6, 0, 2, 14, 6, 2, 8, 2, 2, 2, 6, 18, 22, 4, 12, 8, 16, 8, 2, 2, 0, 0, 16, 0, 8, 2, 14, 4, 12, 16, 0, 2, 2, 0, 6, 0, 14, 8, 0, 0, 12, 14, 4, 0, 0, 14, 12, 4, 0, 6};
    check_sequence("golden_top_k20_top_p095", expected,
                   golden_sequence(top_k20_params(), 777, 100));
}

TEST(golden_top_k0_top_p09) {
    static const std::vector<int> expected = {0, 8, 4, 6, 2, 2, 0, 8, 0, 10, 0, 0, 0, 4, 0, 12, 10, 4, 4, 10, 10, 4, 4, 4, 2, 10, 10, 0, 2, 4, 8, 6, 8, 0, 14, 4, 12, 0, 4, 2, 0, 4, 0, 14, 0, 6, 0, 4, 8, 10, 2, 10, 12, 14, 8, 2, 8, 6, 0, 4, 0, 0, 2, 4, 8, 0, 0, 2, 0, 0, 0, 2, 0, 6, 10, 0, 0, 2, 2, 14, 4, 4, 4, 2, 0, 0, 6, 0, 4, 10, 0, 0, 10, 6, 0, 0, 14, 0, 8, 0};
    check_sequence("golden_top_k0_top_p09", expected,
                   golden_sequence(top_p09_params(), 888, 100));
}

TEST(golden_min_p_0_05) {
    static const std::vector<int> expected = {2, 8, 2, 0, 2, 0, 14, 6, 4, 6, 10, 2, 0, 2, 0, 2, 2, 0, 2, 2, 0, 2, 6, 10, 0, 18, 0, 16, 0, 6, 12, 4, 6, 10, 6, 4, 4, 6, 4, 2, 0, 10, 2, 2, 2, 6, 0, 4, 6, 6, 4, 0, 6, 20, 2, 18, 14, 4, 14, 4, 4, 12, 2, 2, 2, 4, 0, 2, 10, 0, 16, 12, 6, 2, 6, 0, 0, 0, 0, 4, 8, 22, 0, 0, 2, 2, 2, 2, 8, 2, 2, 14, 2, 12, 14, 4, 4, 18, 0, 12};
    check_sequence("golden_min_p_0_05", expected,
                   golden_sequence(min_p_params(), 999, 100));
}

TEST(golden_greedy) {
    static const std::vector<int> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    check_sequence("golden_greedy", expected,
                   golden_sequence(greedy_params(), 1234, 100));
}
