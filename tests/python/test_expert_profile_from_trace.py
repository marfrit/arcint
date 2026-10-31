"""tools/expert_profile_from_trace.py, device-free: the STRP file's round trip
and the ranking on a synthetic trace whose order is known.

1. The ranking, red first. A two-layer, four-expert trace with two requests
   whose per-request distributions fix the order: pair (0, 1) is the hottest
   (1/9 of request 0's pairs and 4/9 of request 1's), then (0, 0) and (1, 0) at
   the same score with (layer, expert) as the tie-break, and so on. The test
   asserts the exact pair list. Red mutation (seen failing): the sort key
   becomes (score ascending) instead of (-score, layer, expert) -- every pair
   is in the reverse order and the assertion fails.

2. The weighting is Strata's mix, red first. One long request routes pair
   (0, 0) a thousand times, one short request routes (0, 3) five times. The
   mix normalizes each request (Strata's make_profile_mix normalizes each
   TRACE, its unit being one prompt's run) and weights 1, so the short one's
   single pair ranks first; `--raw` keeps make_profile's additive counts and
   the long request ranks first. Red mutation (seen failing): drop the
   `c / total` normalization -- the raw order becomes the mix order and the
   assertion fails.

3. The file's round trip, red first. The writer's bytes are read back by the
   replay's reader (the STRP reader arcint's read_profile mirrors) and by the
   tool's own, the ranking survives, and a second run on the same input writes
   the same bytes (determinism). The trailing per-layer rank table is checked
   against the ranking (Strata's writer writes it; arcint's reader stops after
   the pairs). Red mutation (seen failing): swap the pair's u16 fields in the
   writer -- the reader returns (expert, layer) and the round trip fails.

The trace format's own round trip and its reader are tests/python/
test_expert_route_trace.py's; only the profile layer is priced here.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import expert_cache_replay as ecr  # noqa: E402
import expert_profile_from_trace as ept  # noqa: E402


def synthetic(path, nl=2, ne=4, requests=None):
    """Two requests: a prompt, a decode, then the next prompt. Request 0's
    pairs: (0,0)x3 (0,1)x1 (1,2)x2 (1,3)x1 (0,3)x1 (1,1)x1; request 1's:
    (0,1)x4 (1,0)x3 (0,2)x1 (1,3)x1."""
    caches = [[dict(il=0, residents=[0]), dict(il=1, residents=[0])]]
    events = [
        dict(kind=ecr.K_PROMPT, first_pos=0, n_tokens=2, tokens=[1, 2],
             ids={0: [0, 0, 0, 1], 1: [2, 2, 3]}),
        dict(kind=ecr.K_DECODE, first_pos=2, n_tokens=1, tokens=[3], ids={0: [3], 1: [1]}),
        dict(kind=ecr.K_PROMPT, first_pos=0, n_tokens=2, tokens=[4, 5],
             ids={0: [1, 1, 1, 1], 1: [0, 0, 0]}),
        dict(kind=ecr.K_DECODE, first_pos=2, n_tokens=1, tokens=[6], ids={0: [2], 1: [3]}),
    ]
    ecr.write_route(str(path), nl, ne, 4, caches, events)


def build(tmp_path, trace, *extra):
    out = tmp_path / "profile.bin"
    argv = [str(trace), "--out", str(out), "--n-layer", "2", "--n-expert", "4", *extra]
    import subprocess
    r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "expert_profile_from_trace.py")] + argv,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def test_ranking_on_a_synthetic_trace(tmp_path):
    trace = tmp_path / "synthetic.ert1"
    synthetic(trace)
    ranked, _slots = ept.read_profile(build(tmp_path, trace), 2, 4)
    # scores: (0,1) 5/9, (0,0) 3/9, (1,0) 3/9, (1,2) 2/9, (1,3) 2/9, then
    # (0,2) (0,3) (1,1) at 1/9; ties by (layer, expert)
    assert ranked == [(0, 1), (0, 0), (1, 0), (1, 2), (1, 3), (0, 2), (0, 3), (1, 1)]


def test_the_mix_does_not_outvote(tmp_path):
    """One request routes (0, 0) 900 times and (0, 1) 100; a second routes
    (0, 3) 5 times. The mix gives the short request's single pair 1.0 and the
    long one's 0.9, so (0, 3) ranks first; the raw counts put (0, 0) first."""
    a = tmp_path / "a.ert1"
    b = tmp_path / "b.ert1"
    caches = [[dict(il=0, residents=[0]), dict(il=1, residents=[0])]]
    ecr.write_route(str(a), 2, 4, 4, caches,
                    [dict(kind=ecr.K_PROMPT, first_pos=0, n_tokens=1, tokens=[1],
                          ids={0: [0] * 900 + [1] * 100, 1: []}),
                     dict(kind=ecr.K_VERIFY, first_pos=1, n_tokens=2, tokens=[2, 3], ids={0: [0], 1: []})])
    ecr.write_route(str(b), 2, 4, 4, caches,
                    [dict(kind=ecr.K_PROMPT, first_pos=0, n_tokens=1, tokens=[1], ids={0: [3] * 5, 1: []})])
    ranked, _ = ept.read_profile(build(tmp_path, a, "--raw"), 2, 4)
    assert ranked[0] == (0, 0)
    out = tmp_path / "mix.bin"
    import subprocess
    r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "expert_profile_from_trace.py"),
                        str(a), str(b), "--out", str(out), "--n-layer", "2", "--n-expert", "4"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ranked, _ = ept.read_profile(out, 2, 4)
    # a, b are two files as ONE history: a's prompt and decode are request 0,
    # b's prompt starts request 1. Scores: (0,3) 1.0, (0,0) 900/901, (0,1)
    # 100/901, then the fill
    assert ranked[:3] == [(0, 3), (0, 0), (0, 1)], "the mix: a 5-pair request's pair beats 900"


def test_the_file_round_trips_and_is_deterministic(tmp_path):
    trace = tmp_path / "synthetic.ert1"
    synthetic(trace)
    second = tmp_path / "second"
    second.mkdir()
    p1 = build(tmp_path, trace)
    p2 = build(second, trace)
    assert p1.read_bytes() == p2.read_bytes(), "the same input must give the same bytes"
    mine, _ = ept.read_profile(p1, 2, 4)
    theirs, _ = ecr.read_profile(p1, 2, 4)   # the reader arcint's read_profile mirrors
    assert mine == theirs == [(0, 1), (0, 0), (1, 0), (1, 2), (1, 3), (0, 2), (0, 3), (1, 1)]
    blob = p1.read_bytes()
    assert len(blob) == 24 + 4 * 8 + 4 * 2 * 4, "header, pairs, then the per-layer table"
    table = struct.unpack_from("<8i", blob, 24 + 4 * 8)
    want = [-1] * 8
    for rank, (layer, e) in enumerate(mine):
        want[layer * 4 + e] = rank
    assert list(table) == want


def test_the_reader_refuses_a_wrong_version_and_geometry(tmp_path):
    trace = tmp_path / "synthetic.ert1"
    synthetic(trace)
    p = build(tmp_path, trace)
    blob = bytearray(p.read_bytes())
    struct.pack_into("<I", blob, 4, 2)   # version
    bad = tmp_path / "v2.bin"
    bad.write_bytes(bytes(blob))
    with pytest.raises(SystemExit) as e:
        ept.read_profile(bad, 2, 4)
    assert "version 2" in str(e.value)
    with pytest.raises(SystemExit):
        ept.read_profile(p, 3, 4)   # another geometry
    assert struct.unpack_from("<5I", p.read_bytes(), 4)[3] == 8   # slots hint = n_ranked


def test_the_replay_reader_refuses_a_short_header(tmp_path):
    """The replay's read_profile length guard: a file shorter than the 24-byte
    header is a SystemExit naming the length, not a struct.error. Red mutation
    (seen failing): drop the `len(b) < 24` guard -- the unpack raises."""
    p = tmp_path / "short.bin"
    p.write_bytes(b"STR")
    with pytest.raises(SystemExit) as e:
        ecr.read_profile(p, 2, 4)
    assert "shorter than a 24-byte" in str(e.value)


def test_profile_slots_apply_the_walks_distribution(tmp_path):
    """profile_slots gives each layer the slots the global walk actually spent
    on it, so budgets that do not fit the trace's per-layer counts land where
    the engine's walk lands them. Layers with trace budgets 2 and 3 (5 frames);
    the ranking offers 3 pairs of layer 0 and 4 of layer 1, so layer 0 takes 3
    slots (its trace count + 1) and layer 1 the remaining 2. Red mutation (seen
    failing): truncate every layer back to the trace's n_slots -- layer 0 gets
    2 and one frame goes unspent; the total is 4, not 5."""
    layers = [dict(il=0, n_expert=8, n_slots=2), dict(il=1, n_expert=8, n_slots=3)]
    ranked = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)]
    seeded = ecr.profile_slots(layers, ranked)
    assert [L["n_slots"] for L in seeded] == [3, 2], "the walk's distribution, not the trace's"
    assert [L["slots"].tolist() for L in seeded] == [[0, 1, 2], [0, 1]]
    assert sum(L["n_slots"] for L in seeded) == 5, "the whole budget is spent"


def test_replay_seeds_the_slots_and_holds_them_with_adapt_off(tmp_path):
    """The replay's --profile seeding and its adapt-off mode (--every 0, the
    engine's LLAMA_EXPERT_CACHE_ADAPT_EVERY=0): profile_slots models the
    engine's global walk -- one pass over the whole ranking, each pair into its
    layer's next free slot until the trace's total slot budget is spent -- so
    with adaptation off no swap runs and every step's hit count is the same
    fixed placement. Red mutation (seen failing): profile_slots ranks by expert
    index instead of the profile's order -- the seeded slots differ; and
    adapt_every=0 adapting every step (the old `since < 0`) -- swaps appear and
    the hit counts vary."""
    trace = tmp_path / "synthetic.ert1"
    synthetic(trace)
    ranked, _ = ept.read_profile(build(tmp_path, trace), 2, 4)
    layers = [dict(il=0, n_expert=4, n_slots=2), dict(il=1, n_expert=4, n_slots=2)]
    seeded = ecr.profile_slots(layers, ranked)
    # the profile's order per layer: layer 0 (0,1) then (0,0); layer 1 (1,0)
    # then (1,2)
    assert seeded[0]["slots"].tolist() == [1, 0]
    assert seeded[1]["slots"].tolist() == [0, 2]
    steps = [(1, [np.array([1], np.int32), np.array([0], np.int32)]),
             (1, [np.array([0], np.int32), np.array([2], np.int32)])]
    h, per_step, occ, sw = ecr.simulate(seeded, steps, "strata", adapt_every=0, max_swaps=96, delay=2)
    assert sw == 0, "adapt off must swap nothing"
    assert h == 1.0 and occ.tolist() == [2, 2], "placement only: the seeded pairs all hit"
    # the trace's steps in the profile's own order: (0,3) and (1,3) miss
    steps_miss = [(1, [np.array([3], np.int32), np.array([3], np.int32)])]
    h2, _per, _occ, sw2 = ecr.simulate(seeded, steps_miss, "strata", adapt_every=0, max_swaps=96, delay=2)
    assert sw2 == 0 and h2 == 0.0
    # a stream long enough for the engine's own thresholds (min_usage 2, gain
    # 1.5) to be met: the residents stay at zero use while experts 2 and 3 are
    # used every step, so with adaptation ON (every 1) a swap runs; with it OFF
    # nothing does. A mutation that lets adapt_every=0 still adapt (the old
    # `since < 0`) swaps here and fails the sw == 0 line.
    steps_long = [(1, [np.array([2], np.int32), np.array([3], np.int32)])] * 6
    h_off, _p, _o, sw_off = ecr.simulate(seeded, steps_long, "strata", adapt_every=0,
                                         max_swaps=96, delay=2)
    assert sw_off == 0 and h_off == 0.0, "adapt off over a long stream swaps nothing"
    h_on, _p, _o, sw_on = ecr.simulate(seeded, steps_long, "strata", adapt_every=1,
                                       max_swaps=96, delay=2)
    assert sw_on > 0, "the long stream does meet the swap thresholds, so adapt-off is observable"
    assert h_on > h_off, "adaptation moves the hit rate off the placement-only rate"
