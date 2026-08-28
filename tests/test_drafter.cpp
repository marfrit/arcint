#include "core/drafter.h"
#include "harness.h"

using namespace lgc;

TEST(drafter_proposes_nothing_without_a_match) {
    NgramDrafter d(3, 4);
    CHECK(d.draft({1, 2, 3, 4, 5}, 4).empty());
}

TEST(drafter_proposes_nothing_for_a_short_sequence) {
    NgramDrafter d(3, 4);
    CHECK(d.draft({1, 2}, 4).empty());
    CHECK(d.draft({}, 4).empty());
}

TEST(drafter_continues_a_repeat) {
    // "10 11 12" recurs; whatever followed it last time is the guess, filling
    // the whole budget rather than stopping at the end of the original run.
    NgramDrafter d(3, 4);
    CHECK_EQ(d.draft({10, 11, 12, 13, 14, 99, 10, 11, 12}, 4),
             (std::vector<int>{13, 14, 99, 10}));
}

TEST(drafter_respects_the_token_budget) {
    NgramDrafter d(2, 8);
    const auto got = d.draft({1, 2, 3, 4, 5, 6, 7, 1, 2}, 3);
    CHECK_EQ(got.size(), 3u);
    CHECK_EQ(got, (std::vector<int>{3, 4, 5}));
}

TEST(drafter_respects_its_own_maximum) {
    NgramDrafter d(2, 2);
    CHECK_EQ(d.draft({1, 2, 3, 4, 5, 6, 7, 1, 2}, 8).size(), 2u);
}

TEST(drafter_prefers_the_most_recent_occurrence) {
    // "1 2" appears twice; the later one is followed by 9, the earlier by 3.
    NgramDrafter d(2, 2);
    CHECK_EQ(d.draft({1, 2, 3, 0, 1, 2, 9, 0, 1, 2}, 2), (std::vector<int>{9, 0}));
}

TEST(drafter_extends_a_pure_repeat_with_itself) {
    // {5,7} repeating: the earlier occurrence is followed by {5,7} again, so
    // proposing the pattern's own continuation is right, not a bug. The budget
    // is clamped by what is actually available to copy.
    NgramDrafter d(2, 4);
    CHECK_EQ(d.draft({5, 7, 5, 7}, 4), (std::vector<int>{5, 7}));
}

TEST(drafter_never_reads_past_the_end) {
    // Every match start is at most size-ngram-1, so a proposal always has at
    // least one token to copy and the take is clamped to what remains.
    NgramDrafter d(1, 64);
    for (size_t n = 2; n < 12; ++n) {
        std::vector<int> seq(n, 4);        // maximally self-matching
        const auto got = d.draft(seq, 64);
        CHECK(got.size() <= n - 1);
    }
}

TEST(drafter_zero_budget_proposes_nothing) {
    NgramDrafter d(2, 4);
    CHECK(d.draft({1, 2, 3, 1, 2}, 0).empty());
}
