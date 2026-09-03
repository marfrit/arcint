#!/usr/bin/env python3
"""Offline oracle for the DFlash2 selector's dump (M11, the M11 design note (not in the repository)
§1.2 O; DESIGN.md §7.0.2r).

ARCINT_DFLASH_DUMP=<path> makes the server append one JSON line per verify
cycle that actually drafted through DFlash: the lattice (per row, the top-K
candidate token ids, their unary scores, and the transition score from every
reachable predecessor), the drafts proposed, how many were accepted as
served, the target's realized token for every draft row PLUS one extra row
(index == len(drafts_proposed)) that is always the genuine next-anchor token
regardless of whether anything was rejected, and (since round 2,
) `lane`, `request_id`, `past` and the arm
identity (`mode`, `lambda`, `topk`, `block`) -- backend_ov.cpp's
dflash_dump_write / dflash_dump_build_lattice.

WHAT "ON-PATH" MEANS, AND WHY IT BOUNDS EVERYTHING BELOW (round-2 fix, ):
a verify cycle's candidate sets (tokens[i]/unary[i]/transition[i]) come from
ONE masked forward through the small drafter -- all rows at once, independent
of which candidate is picked at any other row (the M11 design note (not in the repository): "the
draft head produces all q-1 rows before selection"). The candidate sets are
therefore always valid to inspect, at any row. The GROUND TRUTH is not: for
row i, `realized[i]` is the TARGET's prediction conditioned on the served
draft sequence up to that row -- valid as truth only while everything it is
conditioned on is itself true. For i < accepted, drafts_proposed[i] IS the
true token (that is what "accepted" means: it equals what the target
independently predicted there). For i == accepted (if accepted <
len(lattice_rows)), realized[accepted] is STILL genuine truth -- its
conditioning is exactly the true, accepted prefix -- and this is exactly the
token the served stream commits next (`next = pick_row(..., accepted)` in
backend_ov.cpp), so it doubles as the chained anchor for the following cycle.
For i > accepted, realized[i] was computed with a REJECTED (wrong) token
already in its context and is not usable as truth from this cycle alone.

Consequently the on-path truth for one cycle never exceeds `accepted + 1`
tokens, and the per-cycle oracle bound below is entirely LOCAL: it never
needs cycle c+1's data to be computed (it only needs c+1's `past` to CONFIRM
the chain is unbroken, which is a diagnostic, not an input to the oracle
number itself). "Chaining cycles of one request" (grouping by lane +
request_id, sorted by `cycle`, asserting past_{c+1} == past_c + 1 +
accepted_c) is therefore about two things this script does, neither of which
changes a single cycle's own oracle math: (1) catching gaps -- a missing or
reordered dump line, or a decode step that did not go through DFlash at all
-- which would otherwise silently make an unrelated pair of cycles look
adjacent; (2) aggregating served/oracle numbers per request/lane rather than
blindly pooling cycles from different requests or arms in one file.

WHAT THIS SCRIPT PRINTS, per file:
  - cycles / requests / lanes seen, and any past-invariant gaps found
  - mean accepted (as served): what the actual selector achieved
  - mean chain oracle: the on-path bound above (unbounded rank, i.e. "was the
    true token ANYWHERE in the row's candidate set") -- at most accepted+1
    per cycle, by construction
  - realizable top-b, by unary and by transition-from-the-true-predecessor,
    for b in {2, 3, 4}: the same on-path bound, but additionally requiring
    the true token to RANK within the top b of its row under that scoring,
    not merely be present in the top-K. This is what the M11 design note (not in the repository)'s own
    §1.2 (O) split is actually asking for.

ON THE "SINGLE-PATH" / "OVER-THE-TREE" TERMINOLOGY -- CORRECTED (round 2,
): the M11 design note (not in the repository) (O) uses both terms and treats them as bounds that
COULD differ ("low single-path but high over-the-tree oracle => the tree,
not the re-ranker") -- the design note does not claim they coincide, and
does not contain the phrase "over-the-tree oracle if cheap" as a description
of this coincidence; that phrasing was this task's own delegation prompt,
paraphrasing the note, and got misattributed to the note itself in round 1.
Separately, and this part IS this script's own reasoning, not the design
note's: for the UNBOUNDED-RANK on-path oracle above (chain oracle), the
"single-path" reading (one selector, one committed choice per row) and the
"over-the-tree" reading (the accepted prefix may follow any candidate at
each row independently) are the same number here, because with a
deterministic greedy target there is exactly one true continuation and any
omniscient selector -- chain or tree -- finds it or does not; nothing about
tree-shaped verification changes what "the true token was offered" means. A
genuine tree-verify oracle would need per-branch ground truth (a separate
target forward per candidate path), which this dump does not carry.
That omniscient coincidence makes the design's own split VACUOUS at the
unbounded-rank chain oracle -- which is exactly why this script also reports
the BOUNDED (top-b) realizable numbers: a re-ranker is a selector restricted
to reordering by SCORE within the K candidates already offered, so "would a
top-b-by-score selector have gotten this row right" is the number that
actually carries information about whether re-ranking (chain track) or a
wider search (tree track) is where the gap lives -- distinct from, and a
better proxy for the design's split than, the vacuous omniscient coincidence.

SANITY GATE : mean chain oracle >= mean accepted must hold per cycle by
construction (oracle_c = accepted_c + a non-negative bonus, using only
drafts_proposed[i] for i < accepted -- which is a candidate offered at row i
by definition -- and, for the +1 bonus, the request's own realized[accepted]
field). A violation means the dump's own fields are inconsistent with each
other (a corrupt/truncated line, or a schema this script does not
understand), not a real result, so the file is refused rather than
silently reporting a negative gap. The caveat: under a repetition penalty
!= 1.0, realized[] is computed BEFORE this cycle's own commits
(backend_ov.cpp) while the served stream's actual next anchor is chosen
AFTER them, so realized[accepted] can, in principle, disagree with what
actually got served next -- this can break the CHAIN (the gap check below
would catch that as an unexpected `past`), but it does not defeat the
`oracle >= served` inequality itself, since that inequality never depends on
chaining to a next cycle.

Self-checked without a card: --self-test builds a small synthetic dump in a
temp file, replays it through the same code path as a real dump, asserts the
printed numbers, and unlinks the temp file when done .
"""
import argparse
import json
import os
import statistics
import sys
import tempfile

TOP_B_VALUES = (2, 3, 4)


def true_tokens_for_cycle(rec):
    """The on-path truth for one cycle, per the module docstring: the
    accepted drafts (trusted directly -- each was itself an offered
    candidate, by definition of "accepted"), plus realized[accepted] if that
    row exists (index == accepted, within [0, len(lattice_rows))) -- the one
    row beyond the served prefix a selector could ALSO be checked against,
    since its ground truth is still genuine. Nothing beyond that: rows past
    accepted+1 have no verified truth in this cycle's own data.
    """
    drafts   = rec.get("drafts_proposed", [])
    accepted = int(rec.get("accepted", 0))
    rows     = rec.get("lattice_rows", [])
    realized = rec.get("realized", [])

    truth = list(drafts[:accepted])
    if accepted < len(rows) and accepted < len(realized):
        truth.append(realized[accepted])
    return truth


def oracle_prefix(rec):
    """The chain oracle: longest on-path prefix (unbounded rank -- "was the
    true token anywhere in the row's candidate set"). Never more than
    accepted+1 by construction of true_tokens_for_cycle.
    """
    truth = true_tokens_for_cycle(rec)
    rows  = rec.get("lattice_rows", [])
    m = 0
    for row, want in zip(rows, truth):
        if want in row.get("tokens", ()):
            m += 1
        else:
            break
    return m


def _rank(tokens, scores, want):
    """0-based rank of `want` among `tokens`, sorted by `scores` descending
    (ties broken by the earliest index, matching dflash_select_path's own
    tie rule -- immaterial to a b-threshold cut except exactly on a tie at
    the boundary). None if `want` is not among `tokens` or scores is too
    short to rank it.
    """
    if want not in tokens:
        return None
    idx = tokens.index(want)
    if idx >= len(scores):
        return None
    order = sorted(range(len(tokens)), key=lambda j: (-scores[j] if j < len(scores) else float("inf"), j))
    return order.index(idx)


def topb_prefix(rec, b, by):
    """The longest on-path prefix (same truth as oracle_prefix) whose true
    token additionally RANKS within the top `b` of its row, scored either by
    `by="unary"` (the row's own unary field) or `by="transition"` (the score
    from the TRUE predecessor: row 0's flat transition array is already "from
    the anchor" -- always the true predecessor for row 0 -- and row i>0's
    transition[p] is the row for predecessor candidate index p, so p must be
    the true predecessor's OWN index within row i-1's token list, tracked as
    this walk proceeds).
    """
    truth = true_tokens_for_cycle(rec)
    rows  = rec.get("lattice_rows", [])
    m = 0
    prev_idx = None  # index of the true token within the PREVIOUS row's list; None => row 0's anchor case
    for i, want in enumerate(truth):
        if i >= len(rows):
            break
        row    = rows[i]
        tokens = row.get("tokens", [])
        trans  = row.get("transition", [])
        if by == "unary":
            scores = row.get("unary", [])
        elif i == 0:
            scores = trans   # flat K-length array, from the anchor
        elif prev_idx is not None and prev_idx < len(trans):
            scores = trans[prev_idx]   # this row's K-length row of the K_prev x K matrix
        else:
            break   # cannot identify the true predecessor's transition row
        rank = _rank(tokens, scores, want)
        if rank is None or rank >= b:
            break
        m += 1
        prev_idx = tokens.index(want)
    return m


def load_cycles(path):
    cycles = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cycles.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"{path}:{lineno}: skipping malformed line: {e}", file=sys.stderr)
    return cycles


class RefusedDump(Exception):
    """Raised when a dump's own fields are internally inconsistent -- see
    the module docstring's SANITY GATE section."""


def _validate_cycle(rec, where):
    drafts   = rec.get("drafts_proposed", [])
    accepted = int(rec.get("accepted", 0))
    rows     = rec.get("lattice_rows", [])
    if accepted > len(drafts):
        raise RefusedDump(f"{where}: accepted ({accepted}) exceeds drafts_proposed "
                          f"({len(drafts)}) -- corrupt or truncated record")
    if accepted > len(rows):
        raise RefusedDump(f"{where}: accepted ({accepted}) exceeds lattice_rows "
                          f"({len(rows)}) -- corrupt or truncated record")
    served = accepted
    oracle = oracle_prefix(rec)
    if oracle < served:
        raise RefusedDump(
            f"{where}: chain oracle ({oracle}) < served ({served}) -- the dump's own "
            "fields disagree with each other (see the module docstring's caveat on "
            "repetition_penalty != 1.0 breaking the chain, though this inequality does "
            "not depend on chaining) rather than a real measured result")


def chain_requests(cycles):
    """Groups cycles by (lane, request_id), sorted by `cycle` within each
    group, and checks the past-invariant between consecutive cycles of the
    same request. Returns (groups, gaps) -- groups is {(lane, request_id):
    [cycle, ...]}, gaps is a list of human-readable strings. Cycles missing
    `lane`/`request_id` (an older dump, or MTP/ngram cycles that should not
    reach this dump at all) are grouped under key None and never checked for
    gaps -- there is nothing to chain them against.
    """
    groups = {}
    for rec in cycles:
        if "lane" not in rec or "request_id" not in rec:
            key = None
        else:
            key = (rec["lane"], rec["request_id"])
        groups.setdefault(key, []).append(rec)

    gaps = []
    for key, group in groups.items():
        if key is None:
            continue
        group.sort(key=lambda r: r.get("cycle", 0))
        for a, b in zip(group, group[1:]):
            if "past" not in a or "past" not in b:
                continue
            expected = a["past"] + 1 + int(a.get("accepted", 0))
            if b["past"] != expected:
                gaps.append(
                    f"lane {key[0]} request {key[1]}: cycle {a.get('cycle')} -> "
                    f"{b.get('cycle')} expected past {expected}, got {b['past']} "
                    f"({b['past'] - expected:+d})")
    return groups, gaps


def analyze(path, cycles=None):
    """Prints the per-file summary and returns it as a dict (used by
    --self-test to check the numbers directly rather than by eye). Returns
    {"refused": True, "reason": ...} without raising if the sanity gate
    fires, so a caller iterating multiple files does not lose the rest.
    """
    if cycles is None:
        cycles = load_cycles(path)

    if not cycles:
        print(f"{path}: no cycles")
        return {"cycles": 0}

    try:
        for i, rec in enumerate(cycles):
            _validate_cycle(rec, f"{path} line {i + 1} (cycle {rec.get('cycle')})")
    except RefusedDump as e:
        print(f"{path}: REFUSED -- {e}", file=sys.stderr)
        return {"refused": True, "reason": str(e)}

    groups, gaps = chain_requests(cycles)
    lanes    = {r.get("lane") for r in cycles if "lane" in r}
    requests = {(r.get("lane"), r.get("request_id")) for r in cycles
                if "lane" in r and "request_id" in r}

    served = [int(c.get("accepted", 0)) for c in cycles]
    oracle = [oracle_prefix(c) for c in cycles]
    topb   = {
        (by, b): [topb_prefix(c, b, by) for c in cycles]
        for by in ("unary", "transition")
        for b in TOP_B_VALUES
    }

    n           = len(cycles)
    mean_served = statistics.fmean(served)
    mean_oracle = statistics.fmean(oracle)
    mean_topb   = {k: statistics.fmean(v) for k, v in topb.items()}

    print(f"{path}: {n} cycle(s), {len(requests)} request(s) on {len(lanes)} lane(s)")
    if gaps:
        print(f"  {len(gaps)} past-invariant gap(s):")
        for g in gaps:
            print(f"    {g}")
    print(f"  mean accepted (as served): {mean_served:.3f}")
    print(f"  mean chain oracle:         {mean_oracle:.3f}  "
          f"(gap {mean_oracle - mean_served:+.3f} tokens/cycle)")
    print("  realizable top-b, by unary:      " +
          "  ".join(f"b={b} {mean_topb[('unary', b)]:.3f}" for b in TOP_B_VALUES))
    print("  realizable top-b, by transition: " +
          "  ".join(f"b={b} {mean_topb[('transition', b)]:.3f}" for b in TOP_B_VALUES))

    return {
        "cycles": n,
        "requests": len(requests),
        "lanes": len(lanes),
        "gaps": gaps,
        "served": served,
        "oracle": oracle,
        "topb": topb,
        "mean_served": mean_served,
        "mean_oracle": mean_oracle,
        "mean_topb": mean_topb,
        "refused": False,
    }


def _self_test_dump():
    """Builds the synthetic cycles used by --self-test:

    Chain of 2 cycles, lane 0, request 0, r=1 (transition arrays are single
    scores per candidate -- enough to demonstrate a by-unary/by-transition
    ranking divergence without needing real codebook math):

    cycle 0 (past=100, 3 rows, K=3):
      row0: tokens=[1,2,3] unary=[5,9,7] transition=[10,1,5] (flat, from the
            anchor). drafts_proposed[0]=2 (the served pick: highest UNARY,
            matching greedy at lambda=1 when the bilinear term does not
            override it) -- true, since accepted>=1.
            By-unary rank of 2: 0 (top-1: unary 9 is the max) -- realizable
            at b=2.
            By-transition rank of 2: score 1, the LOWEST of {10,1,5} -- rank
            2 (0-based) -- NOT realizable at b=2, IS at b=3. This is the
            divergence the by-unary/by-transition split exists to show: the
            served selector's own choice can rank low by unary alone once
            the bilinear term dominates, which top-b-by-unary-only would
            have missed.
      row1: tokens=[10,11,12] unary=[1,1,9] transition=[[1,1,9],[1,1,9],[1,1,9]]
            (K_prev x K; every predecessor row identical, since the point
            here is unary ranking, not transition). drafts_proposed[1]=12
            (top-1 by both unary and transition) -- accepted=2 so far.
      row2: tokens=[20,21,22] unary=[3,3,3] transition=[[3,3,3]]*3.
            drafts_proposed[2]=20 (index 0), realized[2] (=index accepted=2)
            = 21 -- IS offered (tokens[2] contains 21) so the chain oracle
            gets a +1 bonus here (oracle=3, served=2); by unary rank of 21
            is 1 (tied scores, 0-based index order) -- realizable at b=2.
      accepted=2, realized=[2,12,21,?] (index 3 unused since accepted<3).
      => served=2, oracle=3 (the +1 bonus row).

    cycle 1 (past=103=100+1+2, matching the chain invariant; 2 rows, K=2):
      row0: tokens=[30,31] unary=[4,1] transition=[9,1] (flat, from THIS
            cycle's anchor, which is cycle 0's realized[2]=21).
            drafts_proposed[0]=30 (top by both) -- accepted so far=1.
      row1: tokens=[40,41] unary=[2,2] transition=[[2,2],[2,2]].
            drafts_proposed[1]=41, realized[1]=40 (the row IS offered but
            the served selector picked WRONG: 41 != realized[1]=40, so
            accepted stops at 1, not 2) -- wait: accepted is given directly
            as 1 in the record (matching a genuine rejection at row 1);
            realized[accepted]=realized[1]=40, which tokens[1]=[40,41] DOES
            contain -- chain oracle bonus applies: oracle=2.
      accepted=1, realized=[30,40,?] (index 1 = the bonus row, index 2
      unused).
      => served=1, oracle=2.

    A third, DISCONNECTED cycle on the same lane/request (past=999, wildly
    inconsistent with cycle 1's past+1+accepted=105) exercises gap
    detection: it must be reported as a gap, not silently chained.

    Combined: mean served = (2+1+served_of_cycle2)/3, mean oracle =
    (3+2+oracle_of_cycle2)/3 -- cycle 2 is a trivial 1-row, fully-rejected
    cycle (accepted=0, its own row's realized[0] not offered) contributing
    served=0, oracle=0, chosen simple on purpose so it does not perturb the
    hand-checked numbers above beyond a known constant.
    """
    common_arm = {"mode": "greedy", "lambda": 1.0, "topk": 3, "block": 4}
    cycle0 = {
        "cycle": 0, "lane": 0, "request_id": 0, "past": 100, "arm": common_arm,
        "anchor": 999,
        "lattice_rows": [
            {"tokens": [1, 2, 3], "unary": [5, 9, 7], "transition": [10, 1, 5]},
            {"tokens": [10, 11, 12], "unary": [1, 1, 9],
             "transition": [[1, 1, 9], [1, 1, 9], [1, 1, 9]]},
            {"tokens": [20, 21, 22], "unary": [3, 3, 3],
             "transition": [[3, 3, 3], [3, 3, 3], [3, 3, 3]]},
        ],
        "drafts_proposed": [2, 12, 20],
        "accepted": 2,
        "realized": [2, 12, 21, 21],
    }
    cycle1 = {
        "cycle": 1, "lane": 0, "request_id": 0, "past": 103, "arm": common_arm,
        "anchor": 21,
        "lattice_rows": [
            {"tokens": [30, 31], "unary": [4, 1], "transition": [9, 1]},
            {"tokens": [40, 41], "unary": [2, 2], "transition": [[2, 2], [2, 2]]},
        ],
        "drafts_proposed": [30, 41],
        "accepted": 1,
        "realized": [30, 40, 40],
    }
    cycle2 = {
        "cycle": 2, "lane": 0, "request_id": 0, "past": 999, "arm": common_arm,
        "anchor": 5,
        "lattice_rows": [{"tokens": [50, 51], "unary": [1, 1], "transition": [1, 1]}],
        "drafts_proposed": [52],
        "accepted": 0,
        "realized": [52, 60],
    }
    return [cycle0, cycle1, cycle2]


def self_test():
    cycles = _self_test_dump()

    ok = True

    def check(name, got, want):
        nonlocal ok
        if got != want:
            print(f"SELF-TEST FAIL: {name}: got {got!r}, want {want!r}", file=sys.stderr)
            ok = False

    # red case: an off-path "lucky match" beyond accepted+1 must NOT
    # extend the oracle. realized[2]=8 sits in tokens[2]=[7,8,9], but row 2's
    # context was never verified true (accepted=1, so only rows 0-1 are
    # on-path) -- a walk that used realized[] unconditionally (round 1's
    # bug) would wrongly count it and return 3 instead of 2.
    off_path_bait = {
        "lattice_rows": [{"tokens": [1, 2, 3]}, {"tokens": [4, 5, 6]}, {"tokens": [7, 8, 9]}],
        "drafts_proposed": [2, 99],
        "accepted": 1,
        "realized": [2, 5, 8],
    }
    check("oracle_prefix ignores off-path lucky match", oracle_prefix(off_path_bait), 2)

    # Direct checks of the pure functions, independent of file I/O.
    check("oracle_prefix cycle0", oracle_prefix(cycles[0]), 3)
    check("oracle_prefix cycle1", oracle_prefix(cycles[1]), 2)
    # By unary, all 3 rows rank within top-2 (row0: token2 has the max
    # unary, rank 0; row1: token12 has the max, rank 0; row2: token21 ties
    # at rank 1) -- no divergence from the chain oracle here.
    check("topb unary b=2 cycle0", topb_prefix(cycles[0], 2, "unary"), 3)
    # By transition-from-the-true-predecessor, row0's true token (2) has the
    # LOWEST transition score of the three (1, vs 10 and 5) -- rank 2, so it
    # misses top-2 but clears top-3, which is the divergence this synthetic
    # cycle exists to demonstrate (see _self_test_dump's docstring).
    check("topb transition b=2 cycle0", topb_prefix(cycles[0], 2, "transition"), 0)
    check("topb transition b=3 cycle0", topb_prefix(cycles[0], 3, "transition"), 3)

    groups, gaps = chain_requests(cycles)
    check("groups", len(groups), 1)
    check("gap count", len(gaps), 1)

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            for c in cycles:
                f.write(json.dumps(c) + "\n")

        result = analyze(path)
        check("refused", result.get("refused"), False)
        check("cycles", result["cycles"], 3)
        check("requests", result["requests"], 1)
        check("lanes", result["lanes"], 1)
        check("gaps found", len(result["gaps"]), 1)
        check("served", result["served"], [2, 1, 0])
        check("oracle", result["oracle"], [3, 2, 0])
        check("mean_served", result["mean_served"], 1.0)
        check("mean_oracle", result["mean_oracle"], 5 / 3)

        # The sanity gate: a corrupted record (accepted exceeds what
        # drafts_proposed/lattice_rows can support) must refuse the file,
        # not silently report a bogus/negative gap.
        bad_path = path + ".bad"
        with open(bad_path, "w") as f:
            bad = dict(cycles[0])
            bad["accepted"] = 99
            f.write(json.dumps(bad) + "\n")
        try:
            bad_result = analyze(bad_path)
            check("refused on out-of-range accepted", bad_result.get("refused"), True)
        finally:
            os.unlink(bad_path)

        # A second, DIFFERENT corruption: accepted is in range (passes the
        # bounds checks above) but the served drafts themselves are not
        # among their own row's candidates -- a hand-corrupted record could
        # get here even though a real dump never can (dflash_select_path
        # only ever returns candidates it was given). This is the one case
        # that exercises the `oracle < served` gate specifically, since the
        # bounds checks alone do not catch it.
        bad2_path = path + ".bad2"
        with open(bad2_path, "w") as f:
            bad2 = {
                "cycle": 0, "lane": 0, "request_id": 0, "past": 0, "arm": {},
                "lattice_rows": [{"tokens": [1, 2, 3]}, {"tokens": [4, 5, 6]}],
                "drafts_proposed": [999, 998],   # not offered by either row
                "accepted": 2,
                "realized": [999, 998, 1],
            }
            f.write(json.dumps(bad2) + "\n")
        try:
            bad2_result = analyze(bad2_path)
            check("refused on served draft not in its own candidate set",
                  bad2_result.get("refused"), True)
        finally:
            os.unlink(bad2_path)
    finally:
        os.unlink(path)   #: --self-test must not leak a file per run

    if ok:
        print("SELF-TEST PASS")
    else:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="*", help="ARCINT_DFLASH_DUMP JSONL file(s)")
    ap.add_argument("--self-test", action="store_true",
                     help="check this script against a synthetic dump and exit; "
                          "no card, no real dump needed")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.dumps:
        ap.error("give at least one dump file, or --self-test")

    refused = False
    for path in args.dumps:
        result = analyze(path)
        refused = refused or result.get("refused", False)
    if refused:
        sys.exit(1)


if __name__ == "__main__":
    main()
