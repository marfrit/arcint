// The MTP draft-depth cutoff (--llama-mtp-depth-max), as the serving loop's
// round decides it: a slot drafts while the context it would hold is at or
// under the cutoff, and decodes one token a round past it. The reference is
// the fork's --spec-draft-depth-max (`code`,
// arc-b580/tools/server/server-context.cpp:447,3654-3657): drafting stops when
// `prompt.n_tokens() > N` and the draft-model hook is skipped for a batch
// beyond the cutoff; N 0 drafts always.
//
// The depth is the round's, not the request's: `pos0 + 1` -- the lane's slot
// tokens at the round's start plus the id_last that stands in the context
// next. Recomputing it every round is what lets a later, shorter request on
// the same slot draft again.
//
// The cutoff changes how many tokens a verify carries, never which token a row
// draws: the verify walk samples the target's own row and keeps a draft only
// when it equals the draw (src/exec/verify_walk.h), so with the same rows the
// greedy stream is the drafting arm's. Under sampling the texts may differ by
// design (operator, 2026-11-02: the answer's facts stay the bar).
#pragma once

#include <algorithm>
#include <cstddef>

namespace lgc {

// True while the slot may draft: depth at or under the cutoff, or cutoff 0
// (off, today's behaviour).
inline bool spec_drafting_at(size_t depth, int depth_max) {
    return depth_max <= 0 || depth <= static_cast<size_t>(depth_max);
}

// The round's draft width: the fixed ceiling clipped by the tokens that may
// still be emitted (the loop's plain-loop budget), then 0 past the cutoff. The
// cutoff off is byte-identical to the width computed before the flag existed.
inline int spec_width(int n_draft, int budget, int depth_max, size_t depth) {
    const int n_max = std::max(0, std::min(n_draft, budget - 1));
    return spec_drafting_at(depth, depth_max) ? n_max : 0;
}

}  // namespace lgc
