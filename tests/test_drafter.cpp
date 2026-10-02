#include "core/drafter.h"
#include "core/draft_policy.h"
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

TEST(drafter_reports_how_long_its_match_was) {
    // the last 5 tokens (7 8 9 1 2) agree with an earlier run; ngram 2.
    NgramDrafter d(2, 4);
    CHECK_EQ(d.draft({7, 8, 9, 1, 2, 3, 4, 7, 8, 9, 1, 2}, 4), (std::vector<int>{3, 4, 7, 8}));
    CHECK_EQ(d.last_match(), 5u);
    CHECK(d.draft({1, 2, 3}, 4).empty());
    CHECK_EQ(d.last_match(), 0u);
}

// The cost policy (core/draft_policy.h, Strata's): it takes a confident
// lookup window over a plain step before it has data ...
TEST(draft_policy_takes_a_confident_lookup_before_any_data) {
    DraftPolicy p;
    const auto pk = p.choose(1, 3, 30);  // a 30-token match: the 0.96 prior
    CHECK(pk.lookup);
    CHECK_EQ(pk.t, 4);
}

// ... and declines lookups that keep failing once a plain step's cost is known,
// verifying fewer drafts (or none) instead.
TEST(draft_policy_declines_lookups_that_keep_failing) {
    DraftPolicy p;
    for (int i = 0; i < 40; ++i) {
        p.observe(false, 1, 0, 0, 70.0);        // a plain step: 70 ms
        p.observe(true, 4, 0, 4, 100.0);        // a 4-token window, nothing accepted: 100 ms
        p.observe(true, 2, 0, 4, 85.0);
    }
    const auto pk = p.choose(1, 3, 4);          // a short match: that bucket keeps failing
    CHECK(!pk.lookup);
    CHECK_EQ(pk.t, 1);
}

// The MTP side (choose_chain): a chain is verified only as far as its
// expected committed tokens per measured ms beat a plain step.
TEST(draft_policy_cuts_an_mtp_chain_its_windows_cannot_pay_for) {
    DraftPolicy p;
    const float probs[3] = {0.6f, 0.6f, 0.6f};
    for (int i = 0; i < 40; ++i) {
        p.observe(false, 1, 0, 0, 55.0);         // a plain step: 55 ms
        p.observe_chain(2, probs, 1, 0, 110.0);  // windows cost ~linearly in tokens
        p.observe_chain(3, probs, 2, 1, 160.0);
        p.observe_chain(4, probs, 3, 1, 200.0);
    }
    CHECK_EQ(p.choose_chain(probs, 3), 0);
}

TEST(draft_policy_verifies_a_confident_chain_when_windows_are_cheap) {
    DraftPolicy p;
    const float probs[3] = {0.97f, 0.97f, 0.97f};
    for (int i = 0; i < 40; ++i) {
        p.observe(false, 1, 0, 0, 55.0);
        p.observe_chain(2, probs, 1, 1, 70.0);
        p.observe_chain(3, probs, 2, 2, 85.0);
        p.observe_chain(4, probs, 3, 3, 100.0);
    }
    CHECK_EQ(p.choose_chain(probs, 3), 3);
}

TEST(draft_policy_calibrates_the_heads_probability_by_outcome) {
    DraftPolicy p;
    const float probs[1] = {0.97f};
    const double before = p.chain_rate(0.97f);
    for (int i = 0; i < 40; ++i) p.observe_chain(2, probs, 1, 0, 70.0);  // confident, always wrong
    CHECK(p.chain_rate(0.97f) < 0.3);
    CHECK(before > 0.9);
}
