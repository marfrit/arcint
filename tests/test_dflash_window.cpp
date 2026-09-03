// The pending-window feed cap, pulled out of backend_ov.cpp's
// dflash_append/dflash_draft so it can be checked without OpenVINO or a
// card. src/core/dflash_window.h has the measured record: feeds of
// kDflashWindow (2,048) rows or more in one infer were observed to fail on
// GPU plugins without patch 0014, and the cap below was measured to draft
// successfully on the unpatched plugin at several prompt sizes. It also
// retracts an earlier, narrower claim (a single feed of *exactly* `window`
// rows as THE trigger) that a later instrumented failure at 1,615 rows
// falsified -- see the header for the full record. This file does not
// re-narrate that; it only pins the arithmetic.
#include "core/dflash_window.h"
#include "harness.h"

using namespace lgc;

namespace {

constexpr size_t kWindow = 2048;   // mirrors backend_ov.cpp's kDflashWindow

}  // namespace

// A single feed must never reach the window: this is the whole fix, and the
// case that reproduces the measured defect -- a prompt long enough that the
// window trim lands the pending buffer at exactly 2,048 rows before anything
// has been fed (a 2,300-row prompt gets there well before its end).
TEST(feed_never_reaches_window) {
    const auto plan = dflash_window_plan(0, 2300, kWindow);
    CHECK(plan.feed <= kWindow - 1);
}

// The exact boundary pinned: pending sits at exactly `window` rows (append's
// own trim keeps it there, never above) with nothing newly arriving. The
// single-shot feed for this cycle must be exactly `window - 1`, not merely
// "under the window" -- the cap is a hard `window - 1`, never less.
TEST(feed_at_exact_boundary_is_window_minus_one) {
    const auto plan = dflash_window_plan(kWindow, 0, kWindow);
    CHECK_EQ(plan.keep, kWindow);
    CHECK_EQ(plan.feed, kWindow - 1);
}

// Requirement 3 (byte-identical behaviour under the window): a prompt whose
// pending buffer never reaches window - 1 must get the exact same single
// feed the pre-fix code always computed -- the whole buffer, in one shot --
// so the accepted-per-cycle counts of any record taken under the window
// cannot move.
TEST(feed_matches_pending_below_cap) {
    const auto plan = dflash_window_plan(500, 0, kWindow);
    CHECK_EQ(plan.feed, static_cast<size_t>(500));
    CHECK_EQ(plan.keep, static_cast<size_t>(500));
}

TEST(feed_matches_pending_just_under_cap) {
    // window - 1 pending rows: still no capping, the single feed is exactly
    // what today's code already sends in one infer.
    const auto plan = dflash_window_plan(kWindow - 1, 0, kWindow);
    CHECK_EQ(plan.feed, kWindow - 1);
    CHECK_EQ(plan.keep, kWindow - 1);
}

// At and beyond the window, the existing "keep only the most recent `window`
// rows" trim must be unchanged -- only the feed gets capped, not the
// buffer's retained size.
TEST(keep_still_trims_to_window) {
    CHECK_EQ(dflash_window_plan(kWindow, 0, kWindow).keep, kWindow);
    CHECK_EQ(dflash_window_plan(kWindow - 10, 50, kWindow).keep, kWindow);
    CHECK_EQ(dflash_window_plan(100, 50, kWindow).keep, static_cast<size_t>(150));
}

// Whatever a capped feed leaves behind must be fed in a following step, and
// that remainder is always small: append's trim never lets `keep` exceed
// `window`, so at most one row is ever deferred past a capped feed.
TEST(remainder_after_cap_is_small_and_feedable_next) {
    const auto first = dflash_window_plan(kWindow, 0, kWindow);
    const size_t remainder = first.keep - first.feed;
    CHECK(remainder >= 1);
    CHECK(remainder < kWindow);

    // Feeding the remainder alone next cycle must not need a second cap --
    // one extra step always finishes the job.
    const auto second = dflash_window_plan(remainder, 0, kWindow);
    CHECK_EQ(second.feed, remainder);
}

// feed_steps is what dflash_draft actually iterates (not dflash_window_plan
// -- see the header), so these pin the full step sequence, not just its
// single-shot approximation.
TEST(feed_steps_at_exact_window) {
    CHECK_EQ(feed_steps(2048, kWindow), (std::vector<size_t>{2047, 1}));
}

TEST(feed_steps_just_under_window_is_one_step) {
    CHECK_EQ(feed_steps(2047, kWindow), (std::vector<size_t>{2047}));
}

TEST(feed_steps_two_windows) {
    CHECK_EQ(feed_steps(4096, kWindow), (std::vector<size_t>{2047, 2047, 2}));
}

TEST(feed_steps_two_windows_minus_one) {
    CHECK_EQ(feed_steps(4095, kWindow), (std::vector<size_t>{2047, 2047, 1}));
}

TEST(feed_steps_below_cap_is_one_step) {
    CHECK_EQ(feed_steps(500, kWindow), (std::vector<size_t>{500}));
}

TEST(feed_steps_of_zero_pending_is_empty) {
    CHECK_EQ(feed_steps(0, kWindow), std::vector<size_t>{});
}

// Every step feed_steps produces must itself respect the cap -- the
// invariant the whole fix exists for, restated over the full sequence
// rather than one plan.
TEST(feed_steps_never_exceed_cap) {
    for (size_t pending : {1u, 2046u, 2047u, 2048u, 2049u, 4096u, 10000u}) {
        for (size_t feed : feed_steps(pending, kWindow)) {
            CHECK(feed <= kWindow - 1);
            CHECK(feed >= 1);
        }
    }
}
