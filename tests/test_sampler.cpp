#include "core/sampler.h"
#include "harness.h"

#include <map>

using namespace lgc;

namespace {

SamplerParams greedy_params() {
    SamplerParams p;
    p.temperature = 0.0f;
    p.top_k       = 0;
    p.top_p       = 1.0f;
    p.repetition_penalty = 1.0f;
    return p;
}

SamplerParams sampling_params(float temp, int top_k, float top_p) {
    SamplerParams p;
    p.temperature        = temp;
    p.top_k              = top_k;
    p.top_p              = top_p;
    p.repetition_penalty = 1.0f;
    return p;
}

std::vector<float> logits(std::vector<float> v) { return v; }

}  // namespace

TEST(sampler_greedy_is_argmax) {
    auto    l = logits({0.1f, 0.9f, 0.4f, -2.0f});
    Sampler s(greedy_params(), 1);
    CHECK_EQ(s.sample(l.data(), l.size(), {}), 1);
    CHECK_EQ(Sampler::argmax(l.data(), l.size()), 1);
}

TEST(sampler_top_k_one_is_greedy_too) {
    // top_k 1 collapses the distribution exactly as temperature 0 does; both
    // must take the same path or "greedy" means two things.
    auto l = logits({0.1f, 0.9f, 0.4f});
    SamplerParams p = sampling_params(1.0f, 1, 1.0f);
    Sampler s(p, 12345);
    CHECK(p.greedy());
    CHECK_EQ(s.sample(l.data(), l.size(), {}), 1);
}

TEST(sampler_is_reproducible_for_a_seed) {
    auto draw = [](uint64_t seed) {
        Sampler          s(sampling_params(1.0f, 0, 1.0f), seed);
        std::vector<int> out;
        for (int i = 0; i < 24; ++i) {
            auto l = logits({1.0f, 1.05f, 0.95f, 1.02f, 0.9f});
            out.push_back(s.sample(l.data(), l.size(), {}));
        }
        return out;
    };
    CHECK_EQ(draw(42), draw(42));
    CHECK(draw(42) != draw(43));
}

TEST(sampler_respects_top_k) {
    // Only ids 0 and 1 may ever come out, however long we draw.
    Sampler s(sampling_params(2.0f, 2, 1.0f), 7);
    for (int i = 0; i < 200; ++i) {
        auto      l  = logits({5.0f, 4.9f, 1.0f, 0.5f, 0.1f});
        const int id = s.sample(l.data(), l.size(), {});
        CHECK(id == 0 || id == 1);
    }
}

TEST(sampler_respects_top_p) {
    // One token holds essentially all the mass; a modest top_p must keep only it.
    Sampler s(sampling_params(1.0f, 0, 0.5f), 3);
    for (int i = 0; i < 100; ++i) {
        auto l = logits({12.0f, 0.0f, -1.0f, -2.0f});
        CHECK_EQ(s.sample(l.data(), l.size(), {}), 0);
    }
}

TEST(sampler_never_returns_nothing_for_a_tiny_top_p) {
    // top_p 0 must degenerate to greedy, not to an empty candidate set.
    Sampler s(sampling_params(1.0f, 0, 0.0001f), 5);
    auto    l = logits({0.5f, 2.0f, 0.1f});
    CHECK_EQ(s.sample(l.data(), l.size(), {}), 1);
}

TEST(sampler_temperature_widens_the_distribution) {
    auto spread = [](float temp) {
        Sampler       s(sampling_params(temp, 0, 1.0f), 99);
        std::map<int, int> seen;
        for (int i = 0; i < 400; ++i) {
            auto l = logits({2.0f, 1.0f, 0.0f, -1.0f});
            ++seen[s.sample(l.data(), l.size(), {})];
        }
        return seen.size();
    };
    CHECK(spread(2.0f) >= spread(0.2f));
}

TEST(sampler_repetition_penalty_pushes_seen_tokens_down) {
    SamplerParams p = greedy_params();
    p.repetition_penalty = 4.0f;

    // Without history, id 1 wins. With id 1 in history it must lose.
    auto    fresh = logits({1.0f, 2.0f, 1.5f});
    Sampler a(p, 1);
    CHECK_EQ(a.sample(fresh.data(), fresh.size(), {}), 1);

    auto    penalised = logits({1.0f, 2.0f, 1.5f});
    Sampler b(p, 1);
    CHECK_EQ(b.sample(penalised.data(), penalised.size(), {1, 1, 1}), 2);
}

TEST(sampler_penalties_apply_before_greedy_is_decided) {
    // A greedy request with a penalty must still feel the penalty; deciding
    // greedy first would skip it entirely.
    SamplerParams p    = greedy_params();
    p.presence_penalty = 5.0f;

    auto    l = logits({1.0f, 2.0f});
    Sampler s(p, 1);
    CHECK_EQ(s.sample(l.data(), l.size(), {1}), 0);
}

TEST(sampler_frequency_penalty_scales_with_count) {
    SamplerParams p     = greedy_params();
    p.frequency_penalty = 0.4f;

    auto    once = logits({1.0f, 1.5f});
    Sampler a(p, 1);
    CHECK_EQ(a.sample(once.data(), once.size(), {1}), 1);  // 1.5 - 0.4 = 1.1 still wins

    auto    twice = logits({1.0f, 1.5f});
    Sampler b(p, 1);
    CHECK_EQ(b.sample(twice.data(), twice.size(), {1, 1}), 0);  // 1.5 - 0.8 = 0.7 loses
}

TEST(sampler_ignores_out_of_range_history_ids) {
    SamplerParams p = greedy_params();
    p.repetition_penalty = 2.0f;

    auto    l = logits({1.0f, 2.0f});
    Sampler s(p, 1);
    CHECK_EQ(s.sample(l.data(), l.size(), {-1, 999999}), 1);
}
