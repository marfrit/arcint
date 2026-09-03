#pragma once

#include <cstddef>
#include <vector>

// DFlash2's pending-context window arithmetic, pulled out of
// backend_ov.cpp's dflash_append/dflash_draft so the feed-cap rule can be
// checked without OpenVINO or a card.
//
// Measured facts (24 GB card, dense 27B, DFlash2 head unless noted):
//
//   - 2026-09-03: a 2,155-token prompt's first draft, on a GPU plugin
//     without patch 0014, failed with a layout mismatch ("variable layout
//     ...x0x128 assign output layout ...x2048x128"); the backend logged
//     "dflash: draft failed, disabling the drafter". The feed size in that
//     run was not instrumented: the assign output was 2,048 rows on a 0-row
//     variable (the error text), and that the feed itself was 2,048 rows
//     is the window trim's arithmetic, not a logged value. Patch 0014
//     (patches/0014-gpu-assign-adopts-output-layout.patch -- the plugin-side
//     fix: an Assign that adopts the output layout) clears it. Which code
//     path inside the plugin skips the layout update is not known.
//
//   - Retracted: this file previously generalized from that one failure to
//     "the trigger is a single feed of *exactly* `window` rows". A later
//     instrumented failure, on the re-exported f16 head, fed only 1,615 rows
//     -- well under the window -- and still failed, which falsifies that
//     claim. What is measured is narrower: the one uninstrumented failure
//     above plus the instrumented 1,615-row one, both on plugins without
//     0014; what makes a feed unsafe is not established, and this file does
//     not claim to know.
//
//   - Measured since delivery: with the packaged (unpatched) plugin and the
//     feed cap below (window - 1, with a priming pass for the remainder --
//     see feed_steps), prompts of 2,481 / 3,251 / 17,126 tokens drafted at
//     1.94 / 2.06 / 1.39 accepted tokens per cycle (64-token probes), and a
//     3,000-token decode sustained 2.84 accepted per cycle. The cap is kept on the strength of
//     this measurement, not because the underlying plugin defect is
//     understood.
//
// The rule this helper encodes: a single feed into the drafter's state
// should never carry `window` rows or more. It is capped at `window - 1`,
// and whatever does not fit waits for a following feed. Below that cap
// nothing changes -- the feed is the same single-shot value the pre-fix
// code always used, so every prompt under the window keeps byte-identical
// accepted-per-cycle counts.
namespace lgc {

struct DflashWindowPlan {
    // Total rows retained in the pending buffer once the existing
    // "keep only the most recent `window` rows" trim is applied.
    size_t keep = 0;
    // The single-shot feed size the pre-fix code would have used for `keep`
    // rows, capped at `window - 1`. dflash_append uses only `keep` (the
    // trim); dflash_draft does not use this field at all -- it drives its
    // multi-step feed loop from feed_steps() below, so the loop under test
    // is the loop that runs. This field stays as the single-shot view of the
    // same cap, pinned by its own test.
    size_t feed = 0;
};

// `pending`: rows already buffered, not yet fed to the drafter.
// `new_rows`: rows being appended now.
// `window`: the drafter's context window (backend_ov.cpp's kDflashWindow).
inline DflashWindowPlan dflash_window_plan(size_t pending, size_t new_rows, size_t window) {
    const size_t total = pending + new_rows;
    if (window == 0) return {total, total};
    const size_t keep = total > window ? window : total;
    const size_t cap  = window - 1;
    const size_t feed = keep > cap ? cap : keep;
    return {keep, feed};
}

// The full sequence of feeds needed to move `pending` rows into the
// drafter's state without any single infer carrying `window` rows or more:
// chunks of `window - 1`, with whatever remains (always < window - 1) as
// the last element. dflash_draft iterates this directly -- every step
// before the last is a priming feed whose output is discarded (only
// new_feats ever becomes persistent state); the last step also carries the
// draft block and is the one whose output gets scored.
inline std::vector<size_t> feed_steps(size_t pending, size_t window) {
    std::vector<size_t> steps;
    const size_t         cap = window > 0 ? window - 1 : pending;
    size_t               remain = pending;
    while (remain > 0) {
        const size_t feed = remain > cap ? cap : remain;
        steps.push_back(feed);
        remain -= feed;
    }
    return steps;
}

}  // namespace lgc
