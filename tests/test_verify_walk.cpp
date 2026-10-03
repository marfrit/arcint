// The verify walk of speculative decoding (src/exec/verify_walk.h) on
// synthetic logits: what it keeps, emits and ends on, and that a sampled
// walk draws exactly what the plain loop draws from the same rows.
#include "exec/verify_walk.h"
#include "harness.h"

#include <vector>

using namespace lgc;

namespace {

constexpr size_t kVocab = 16;

// rows whose argmax is the given token, row by row (a ramp, so a sampled
// draw has every token to choose from)
std::vector<float> rows_with_argmax(const std::vector<int>& argmax) {
    std::vector<float> r(argmax.size() * kVocab);
    for (size_t i = 0; i < argmax.size(); ++i)
        for (size_t v = 0; v < kVocab; ++v)
            r[i * kVocab + v] = static_cast<int>(v) == argmax[i] ? 4.0f : 0.1f * static_cast<float>(v % 5);
    return r;
}

SamplerParams greedy() {
    SamplerParams p;
    p.temperature        = 0.0f;
    p.repetition_penalty = 1.0f;
    return p;
}

struct Run {
    VerifyWalk       w;
    std::vector<int> out;
};

Run walk(std::vector<float> rows, const std::vector<int>& draft, int budget, int stop_tok = -1,
         int cancel_at = -1, const SamplerParams& sp = greedy(), uint64_t seed = 1) {
    Sampler sampler(sp, seed);
    Run     r;
    r.w = walk_verify(rows.data(), rows.size() / kVocab, kVocab, draft, sampler, budget,
                      [&](int t) { return t == stop_tok; },
                      [&](int t) {
                          r.out.push_back(t);
                          return static_cast<int>(r.out.size()) - 1 == cancel_at ? Control::Cancel : Control::Continue;
                      });
    return r;
}

}  // namespace

TEST(verify_walk_full_acceptance_emits_every_row) {
    const Run r = walk(rows_with_argmax({5, 6, 7}), {5, 6}, 100);
    CHECK_EQ(r.w.accepted, 2);
    CHECK_EQ(r.w.emitted, 3);
    CHECK_EQ(r.w.last, 7);
    CHECK(!r.w.done);
    CHECK(r.out == std::vector<int>({5, 6, 7}));
}

TEST(verify_walk_mismatch_keeps_the_prefix) {
    const Run r = walk(rows_with_argmax({5, 9, 7}), {5, 6}, 100);
    CHECK_EQ(r.w.accepted, 1);
    CHECK_EQ(r.w.emitted, 2);   // the matching draft, then the target's own token
    CHECK_EQ(r.w.last, 9);
    CHECK(!r.w.done);
}

TEST(verify_walk_stop_token_is_not_emitted) {
    const Run r = walk(rows_with_argmax({5, 9, 7}), {5, 6}, 100, /*stop_tok=*/9);
    CHECK_EQ(r.w.accepted, 1);
    CHECK_EQ(r.w.emitted, 1);
    CHECK(r.w.done);
    CHECK(r.w.reason == FinishReason::Stop);
}

TEST(verify_walk_budget_ends_on_length) {
    // a full acceptance with room for two tokens: the third row is not emitted
    const Run r = walk(rows_with_argmax({5, 6, 7}), {5, 6}, 2);
    CHECK_EQ(r.w.emitted, 2);
    CHECK_EQ(r.w.accepted, 2);
    CHECK(r.w.done);
    CHECK(r.w.reason == FinishReason::Length);
}

TEST(verify_walk_cancel_aborts_after_the_token) {
    const Run r = walk(rows_with_argmax({5, 6, 7}), {5, 6}, 100, -1, /*cancel_at=*/0);
    CHECK_EQ(r.w.emitted, 1);
    CHECK_EQ(r.w.accepted, 0);
    CHECK(r.w.done);
    CHECK(r.w.reason == FinishReason::Abort);
}

TEST(verify_walk_sampled_draws_what_the_plain_loop_draws) {
    SamplerParams sp;
    sp.temperature        = 1.0f;
    sp.top_p              = 1.0f;
    sp.top_k              = 0;
    sp.repetition_penalty = 1.3f;   // history matters: observe() order must match
    const std::vector<float> base = rows_with_argmax({3, 3, 3, 3, 3, 3});
    for (uint64_t seed = 1; seed <= 40; ++seed) {
        // the plain loop: sample a row, observe it, next row
        Sampler          plain(sp, seed);
        std::vector<float> rows = base;
        std::vector<int> want;
        for (size_t i = 0; i < 6; ++i) {
            const int t = plain.sample(rows.data() + i * kVocab, kVocab);
            want.push_back(t);
            plain.observe(t);
        }
        // a draft equal to the plain draws is kept whole
        const Run all = walk(base, std::vector<int>(want.begin(), want.end() - 1), 100, -1, -1, sp, seed);
        CHECK(all.out == want);
        CHECK_EQ(all.w.accepted, 5);
        // a draft wrong at position 2 stops there, with the plain draws so far
        std::vector<int> bad(want.begin(), want.end() - 1);
        bad[2] = (bad[2] + 1) % static_cast<int>(kVocab);
        const Run part = walk(base, bad, 100, -1, -1, sp, seed);
        CHECK_EQ(part.w.accepted, 2);
        CHECK(part.out == std::vector<int>(want.begin(), want.begin() + 3));
    }
}
