"""tools/expert_cache_replay.py and the ERT1 routing trace, device-free.

Three cells, red first:

1. The round trip, in a DEFAULT pytest. The fixture
   tests/python/fixtures/expert-route-trace-v2.ert1 is the byte output of
   tools/expert_route_trace_selftest.cpp compiled against patch 0069's
   serializer; the test reads it with the replay's reader, rebuilds the same
   bytes with the replay's writer and compares them. EXPERT_ROUTE_TRACE_SRC
   (the llama tree with patch 0069) is optional: set, the test also compiles
   the selftest and requires its output to equal the fixture, so the fixture
   cannot drift from the patch. Red mutation (seen failing): the serializer's
   LAYERS record writes `n_slots + 1` -- the reader misreads the residents and
   the records that follow.

2. A prompt's 4-token checkpoint tail is a PROMPT, not a verify. The fixture
   has one at position 10, beside a verify at 200 and a decode at 203; the
   replay's steps are the two decode graphs only, and the accept sidecar is
   aligned by POSITION, not by order. The selftest also checks the cache's
   graph-kind decision directly (prompt tail -> PROMPT, verify -> VERIFY,
   decode -> DECODE, an embedding batch -> PROMPT). Red mutation (seen
   failing): drop `!all_outputs` from llama_expert_route_graph_kind -- the tail
   classifies VERIFY and the selftest exits 3.

3. The foresight predictor on a synthetic trace where the drafts are known
   BEFORE their verify. Two requests: the even one trains a token->expert
   table (its verify's token[1] row is the expert pair the NEXT step routes),
   the odd one is scored. The odd verify's draft token SAME ids carry GARBAGE
   routing rows, so a predictor that read the verify's routing (the old `mtp`)
   cannot reach the oracle while one that reads only the draft token ids does
   (horizon 1, delay 1: the oracle's first landing row). Red mutation (the
   test asserts it): the old row-based predictor misses; and a provider of
   zeros must not reach the oracle.

The synthetic windows are small on purpose: the cells price the classification,
the alignment and the plan source, not the verify width.

Round 6 adds the continuous-log cells: a second ERT1 header mid-file replays,
a `.000` argument finds its `.001` rotations, a later file's LAYERS updates
`residents`, an unclean restart's one missing last graph a file is tolerated,
the MTP expert matrix is fitted on training requests only and is not identity,
`--accept DIR` reads its sidecars, `--single DIR` is refused, a repeated open
does not append, a missing DIR is created, and a rotation puts its accept line
and graph in the same file (red: line before rotation).

Round 7 adds two cells: a graph record that is still pending must not rotate
away from its accept line when the MTP DRAFT records cross the cap between
`after_graph` and the writing `advance` (red: rotate while pending, the old
file keeps the line with no graph), and a write failure stops tracing with one
WARN and no further write (red: keep `opened` set after the failed write).

Round 8 adds three cells: the pending guard is process-wide, so under a layer
split cache B's record never lands in `.001` while cache A's accept line stays
in `.000` (red: let `rotate_now` test this cache's flag instead of the
process-wide count); a rotation's LAYERS record carries every cache's
residents, so each file replays alone under a split; and `advance` must never
call `trace_prepare` (a source-order scan with the function landmarks named,
red: put the call back into `advance`).
"""
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import expert_cache_replay as ecr  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE = FIXTURES / "expert-route-trace-v2.ert1"
ACCEPT = FIXTURES / "expert-route-trace-v2.accept"


def _rebuild(tr):
    """The replay's writer output for a loaded trace, to compare bytes."""
    from tempfile import NamedTemporaryFile
    events = []
    for e in tr["events"]:
        if e["kind"] == ecr.K_TABLES:
            events.append(dict(kind=ecr.K_TABLES, changes=e["changes"]))
        elif e["kind"] in (ecr.K_PROMPT, ecr.K_DECODE, ecr.K_VERIFY, ecr.K_DRAFT):
            events.append(dict(kind=e["kind"], first_pos=e["first_pos"], n_tokens=e["n_tokens"],
                               tokens=e["tokens"].tolist(), ids={il: ids.tolist() for il, ids in e["ids"].items()}))
    caches = [[dict(il=c["il"], residents=c["residents"].tolist()) for c in cache] for cache in tr["caches"]]
    with NamedTemporaryFile(suffix=".ert1", delete=False) as f:
        path = f.name
    try:
        ecr.write_route(path, tr["n_layer"], tr["n_expert"], tr["n_used"], caches, events)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def test_trace_round_trip(tmp_path):
    tr = ecr.load_route(FIXTURE)
    assert (tr["version"], tr["n_layer"], tr["n_expert"], tr["n_used"]) == (2, 6, 32, 4)
    assert not tr["torn"]
    (cache,) = tr["caches"]
    assert [(c["il"], c["n_slots"]) for c in cache] == [(2, 5), (5, 5)]
    assert cache[0]["residents"].tolist() == [0, 3, 7, 11, -1]
    assert cache[1]["residents"].tolist() == [1, 2, 3, 4, -1]

    graphs = [e for e in tr["events"] if e["kind"] in (1, 2, 3, 4)]
    assert [(g["kind"], g["first_pos"], g["n_tokens"]) for g in graphs] == [
        (1, 0, 3), (1, 10, 4), (3, 200, 3), (2, 203, 1), (4, 204, 2)]
    assert graphs[0]["tokens"].tolist() == [100, 101, 102]
    assert graphs[0]["ids"][2].tolist() == [0, 3, 7, 11, 2, 9]      # rank order kept
    assert graphs[2]["tokens"].tolist() == [2000, 2001, 2002]
    assert graphs[2]["ids"][5].tolist() == [11, 12, 13, 14, 15, 16]
    assert ecr.read_accept(ACCEPT, [dict(kind=g["kind"], first_pos=g["first_pos"], n_tokens=g["n_tokens"])
                                   for g in graphs]) == [-1, -1, 1, 0, -1]
    # the MTP block's own decode: INPUT tokens (accepted target, previous
    # draft), not the drafted tokens, and its one block's ids
    assert graphs[4]["tokens"].tolist() == [4000, 4001]
    assert graphs[4]["ids"][6].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [(e["kind"], e["changes"]) for e in tr["events"] if e["kind"] == 5] == [
        (5, [(2, 5, 1), (2, 0, 0), (5, 11, 1)])]

    # the writer's bytes, byte for byte
    assert _rebuild(tr) == FIXTURE.read_bytes()

    # optional: the fixture is the patched C++ writer's output, not a stale copy
    src = os.environ.get("EXPERT_ROUTE_TRACE_SRC")
    if src:
        src = Path(src)
        ser = src / "src" / "llama-expert-route-trace.cpp"
        if ser.exists():
            exe = tmp_path / "selftest"
            r = subprocess.run(["g++", "-std=c++17", "-O1", "-I", str(src / "src"),
                                str(REPO_ROOT / "tools" / "expert_route_trace_selftest.cpp"), str(ser),
                                "-o", str(exe)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            out = tmp_path / "synthetic.ert1"
            assert subprocess.run([str(exe), str(out)]).returncode == 0, "the graph-kind red cell failed"
            files = ecr.expand_traces(str(out))
            assert len(files) == 1, files
            assert Path(files[0]).read_bytes() == FIXTURE.read_bytes(), "fixture differs from the patched writer"


def test_prompt_tail_and_position_alignment():
    tr = ecr.load_route(FIXTURE)
    layers, graphs, steps, acc = ecr.route_to_steps(tr, ACCEPT)
    assert [l["il"] for l in layers] == [2, 5]
    # the 4-token tail at position 10 is a prompt graph, not a decode step
    assert [g["kind"] for g in graphs] == [ecr.K_PROMPT, ecr.K_PROMPT, ecr.K_VERIFY, ecr.K_DECODE]
    assert [(s["kind"], s["first_pos"]) for s in steps] == [(ecr.K_VERIFY, 200), (ecr.K_DECODE, 203)]
    # aligned by position: one entry a graph, -1 for the prompt graphs
    assert acc == [1, 0]
    res = ecr.trace_residency(layers, tr)
    assert res[0][5] and not res[0][0] and res[1][11]


def _pair(t, l):
    return [(2 * t) % 16, (2 * t + 1) % 16] if l == 0 else [(2 * t + 8) % 16, (2 * t + 9) % 16]


def _synthetic_foresight(tmp_path, n_steps=6, k=2):
    """Two requests, identical token ids. The even request's verify token[1]
    routes the NEXT step's first pair (the training table); the odd request's
    verify token[1] carries garbage rows, so only a token-id predictor reaches
    the oracle. Every step is two tokens: [a_i, d_i]; the actual first row of
    step i+1 is pair(i+1), which is what horizon 1 from delay 1 sees."""
    layers = [dict(il=0, n_expert=16, n_slots=4, residents=np.array([0, 1, 14, 15], np.int32)),
              dict(il=1, n_expert=16, n_slots=4, residents=np.array([8, 9, 14, 15], np.int32))]
    events = []
    garbage = [15, 15]
    for req in (0, 1):
        # a prompt whose routing recurs in the answer, so the baseline can admit it
        pids = {0: [0, 1, 0, 1, 0, 1, 0, 1], 1: [8, 9, 8, 9, 8, 9, 8, 9]}
        events.append(dict(kind=ecr.K_PROMPT, first_pos=0 if req == 0 else 1000, n_tokens=4,
                           tokens=[10 + req, 11 + req, 12 + req, 13 + req], ids=pids))
        for i in range(n_steps):
            draft = 500 + i
            # token a_i routes pair(i), the draft token routes the NEXT step's
            # pair in the even request; the odd request's draft rows are garbage
            ids = {0: _pair(i, 0) + (_pair(i + 1, 0) if req == 0 else garbage),
                   1: _pair(i, 1) + (_pair(i + 1, 1) if req == 0 else garbage)}
            events.append(dict(kind=ecr.K_VERIFY, first_pos=(200 if req == 0 else 1200) + 2 * i,
                               n_tokens=2, tokens=[1000 + i, draft], ids=ids))
    trace = tmp_path / "foresight.ert1"
    ecr.write_route(trace, 2, 16, k, [layers], events)
    lines = []
    for i in range(2 * n_steps):
        pos = (200 if i < n_steps else 1200) + 2 * (i % n_steps)
        lines.append("verify %d pos %d rows 2 accepted 1\n" % (i + 1, pos))
    Path(str(trace) + ".accept").write_text("".join(lines))
    return trace, layers


def test_foresight_reaches_the_oracle_without_the_verify_rows(tmp_path):
    trace, layers = _synthetic_foresight(tmp_path)
    tr = ecr.load_route(trace)
    layers, graphs, steps, acc = ecr.route_to_steps(tr, str(trace) + ".accept")
    assert len(steps) == 12 and all(a == 1 for a in acc if a >= 0)
    prov, _, _, foresight, rid = ecr.providers(layers, graphs, steps, acc, horizon=1, delay=1)
    # the odd request's verify rows are garbage; the token-id foresight still
    # carries the next step's pair
    assert foresight[6][0].nonzero()[0].tolist() == _pair(1, 0)
    assert foresight[0][0].nonzero()[0].tolist() == _pair(1, 0)

    stream = ecr.step_stream(layers, steps)
    sim = lambda pol, pf: ecr.simulate(layers, stream, pol, adapt_every=1, max_swaps=96,
                                       delay=1, provider=pf)[0]
    fores = sim("plan", prov["foresight"])
    oracle = sim("plan", prov["oracle"])
    base = sim("strata", None)
    assert fores == pytest.approx(oracle, abs=1e-9), "the token-id predictor %r against the oracle %r" % (fores, oracle)
    assert base < oracle, "the past-based baseline reached the oracle: the cell is trivial"

    # RED 1: a predictor of zeros must not reach the oracle
    zeros = lambda si: np.zeros((len(layers), 16))
    red = sim("plan", zeros)
    assert red < oracle, "ignoring the predictor's plan still reached the oracle %r: the check is blind" % red
    assert red == pytest.approx(base), "a plan of zeros does not replay the baseline (%r against %r)" % (red, base)

    # RED 2: the old source -- the verify's own accepted rows -- reads the
    # garbage under the odd request, so it must miss the oracle
    nl = len(layers)
    row_based = []
    for g in steps:
        k = len(next(iter(g["ids"].values()))) // g["n_tokens"]
        ids = [x[t * k:(t + 1) * k] for t in range(1, 2) for x in g["ids"].values()]
        row_based.append(ecr._future_from_ids(layers, ids))
    old = sim("plan", lambda si: ecr.normalize(row_based[si]))
    assert old < oracle, "the verify rows were enough (%r against %r): the synthetic is not honest" % (old, oracle)


def test_learned_map_stays_in_its_request():
    """The map works in step space and both its features and its target stop at
    the request's ends, so a scored request cannot appear in a training row.

    Red mutation (seen failing): drop `and rid[j] == rid[si]` from
    LearnedMap.target -- the target runs into the next request.
    """
    nl, ne = 1, 4
    use = np.zeros((6, nl, ne), np.int64)
    rid = np.array([0, 0, 0, 1, 1, 1])
    for si in range(6):
        use[si, 0, si % ne] = 1
    use[3] *= 100                                       # the held-out request
    prior = np.zeros((6, nl, ne))
    foresight = np.zeros((6, nl, ne))
    lm = ecr.LearnedMap(horizon=3, delay=1)
    # the target starts at si + delay and stops before the next request
    y = lm.target(use, rid, 0)                          # rows 1 and 2, not row 3
    assert y[0].nonzero()[0].tolist() == [1, 2]
    assert not y[0][3]
    # features of the first step of request 1 never look back into request 0
    X = lm.feats(use, prior, foresight, rid, 3)
    assert all(not f[0][0] for f in X[:4]), "a feature reached into the previous request"
    # and the fit for an even request never sees the odd request's rows
    lm.fit(use, prior, foresight, rid, [0, 1])
    train_rows = {tuple(lm.target(use, rid, si)[0].nonzero()[0].tolist()) for si in (0, 1)}
    assert train_rows == {(1, 2), (2,)}, "a training target crossed the request boundary: %r" % train_rows


# ---------------------------------------------------------------- round 3 ----

def _trace(tmp_path, name, layers, events):
    path = tmp_path / name
    ecr.write_route(path, len(layers), layers[0]["n_expert"], 2, [layers], events)
    return path


def _two_layers():
    return [dict(il=0, n_expert=16, n_slots=4, residents=np.array([0, 1, 14, 15], np.int32)),
            dict(il=1, n_expert=16, n_slots=4, residents=np.array([8, 9, 14, 15], np.int32))]


def _prompt(pos, tokens):
    return dict(kind=ecr.K_PROMPT, first_pos=pos, n_tokens=len(tokens), tokens=tokens,
                ids={0: [0, 1] * len(tokens), 1: [8, 9] * len(tokens)})


def _decode(pos, token):
    return dict(kind=ecr.K_DECODE, first_pos=pos, n_tokens=1, tokens=[token],
                ids={0: [2, 3], 1: [10, 11]})


def _verify(pos, tokens):
    return dict(kind=ecr.K_VERIFY, first_pos=pos, n_tokens=len(tokens), tokens=tokens,
                ids={0: [4, 5] * len(tokens), 1: [12, 13] * len(tokens)})


def test_kind_decision_mirrors_the_serializer():
    """The serializer's graph-kind decision, mirrored in Python (finding 3).

    The C++ cell only runs under EXPERT_ROUTE_TRACE_SRC; this one always runs.
    Red before the mirror existed. Mutation: drop `!all_outputs` from the
    mirror -- the 4-token prompt tail classifies VERIFY.
    """
    kind = ecr.route_graph_kind
    assert kind(4, 4, False, True) == ecr.K_PROMPT      # a prompt's 4-token tail
    assert kind(4, 4, True, True) == ecr.K_VERIFY
    assert kind(1, 4, True, True) == ecr.K_DECODE
    assert kind(3, 4, False, True) == ecr.K_PROMPT      # a prompt batch
    assert kind(5, 4, True, True) == ecr.K_PROMPT       # longer than decode_max
    assert kind(4, 4, True, False) == ecr.K_PROMPT      # an embedding batch, no ids
    assert kind(2, 4, True, True) == ecr.K_VERIFY


def test_prompt_last_token_decode_is_relabeled(tmp_path):
    """arcint runs a prompt's last token alone with logits in the non-spec path
    (backend_llama.cpp: all but the last prefilled, the last decoded for its
    logits), so a trace records it as a 1-token DECODE at the prompt's end and
    the sampler never writes an accept line for it. The replay relabels it the
    prompt's last token, keeps it out of the scored steps and reports the count.

    Red before the relabel: read_accept exits "no line for the decode graph at
    position 3".
    """
    lay = _two_layers()
    events = [_prompt(0, [100, 101, 102]), _decode(3, 102), _verify(4, [102, 200])]
    trace = _trace(tmp_path, "lasttoken.ert1", lay, events)
    Path(str(trace) + ".accept").write_text("verify 1 pos 4 rows 2 accepted 1\n")
    tr = ecr.load_route(trace)
    layers, graphs, steps, acc = ecr.route_to_steps(tr, str(trace) + ".accept")
    assert [(g["kind"], g["first_pos"]) for g in graphs] == [
        (ecr.K_PROMPT, 0), (ecr.K_PROMPT, 3), (ecr.K_VERIFY, 4)]
    assert tr["relabeled"] == 1
    assert steps and all(s["kind"] != ecr.K_DECODE for s in steps)
    assert acc == [1]

    # no sidecar and no verify graph: the non-spec loop's trace, same relabel
    ns = _trace(tmp_path, "lasttoken-nonspec.ert1", lay,
                [_prompt(0, [100, 101, 102]), _decode(3, 102), _decode(4, 103)])
    tr2 = ecr.load_route(ns)
    _, graphs2, steps2, acc2 = ecr.route_to_steps(tr2, None)
    assert tr2["relabeled"] == 1 and acc2 is None
    assert [g["kind"] for g in graphs2] == [ecr.K_PROMPT, ecr.K_PROMPT, ecr.K_DECODE]
    # a sidecar-less speculative trace has one-row verifies dressed as DECODE;
    # with verify graphs present the adjacency alone cannot tell them apart
    tr3 = ecr.load_route(trace)
    _, graphs3, steps3, _ = ecr.route_to_steps(tr3, None)
    assert tr3["relabeled"] == 0 and [g["kind"] for g in graphs3] == [
        ecr.K_PROMPT, ecr.K_DECODE, ecr.K_VERIFY]


def test_accept_aligns_per_request_with_repeated_positions(tmp_path):
    """Every request restarts at position 0, so the same position appears in
    several requests. The sidecar is aligned in request order, checked by
    position and row count -- never a global position set. Red before: "two
    lines for position 3". A decode at a prompt's end WITH a line stays a
    decode (the prompt-last relabel is only for a missing line).
    """
    layers = _two_layers()
    events = [_prompt(0, [100, 101, 102]), _decode(3, 102),
              _prompt(0, [110, 111, 112]), _decode(3, 112)]
    trace = _trace(tmp_path, "tworeq.ert1", layers, events)
    Path(str(trace) + ".accept").write_text(
        "verify 1 pos 3 rows 1 accepted 0\nverify 2 pos 3 rows 1 accepted 0\n")
    tr = ecr.load_route(trace)
    layers, graphs, steps, acc = ecr.route_to_steps(tr, str(trace) + ".accept")
    assert acc == [0, 0]
    assert tr["relabeled"] == 0
    assert [g["kind"] for g in steps] == [ecr.K_DECODE, ecr.K_DECODE]
    assert [r for r in ecr.request_of(graphs, steps)[1]] == [0, 1]


def test_trailing_accept_line_is_the_unwritten_last_graph(tmp_path):
    """The cache writes a graph when the NEXT graph's advance runs, so a
    process's last graph is never written (finding 4). The sidecar still has
    its line; the replay keeps it apart instead of refusing the trace.
    """
    layers = _two_layers()
    events = [_prompt(0, [100, 101, 102]), _decode(3, 102), _verify(4, [102, 200])]
    trace = _trace(tmp_path, "short.ert1", layers, events)
    Path(str(trace) + ".accept").write_text(
        "verify 1 pos 3 rows 1 accepted 0\nverify 2 pos 4 rows 2 accepted 1\n"
        "verify 3 pos 6 rows 4 accepted 2\n")
    tr = ecr.load_route(trace)
    layers, graphs, steps, acc = ecr.route_to_steps(tr, str(trace) + ".accept")
    assert acc == [0, 1]
    assert tr["unwritten_last_graph"] == 1


def test_default_report_is_held_out(tmp_path, capsys):
    """With odd requests the default report scores only the odd requests; the
    token table is fitted on the even ones (finding 2). Red before: an
    "all decode steps" section scored the fitted table in-sample.
    """
    layers = _two_layers()
    events = [_prompt(0, [100, 101, 102]), _decode(3, 102),
              _prompt(0, [110, 111, 112]), _decode(3, 112)]
    trace = _trace(tmp_path, "report.ert1", layers, events)
    Path(str(trace) + ".accept").write_text(
        "verify 1 pos 3 rows 1 accepted 0\nverify 2 pos 3 rows 1 accepted 0\n")
    import argparse
    a = argparse.Namespace(trace=str(trace), accept=str(trace) + ".accept", delay=2, cap=96,
                           every=6, horizon=1, learned=False, sweep=False, bounds=False, budget=4)
    ecr.main_route(a)
    out = capsys.readouterr().out
    assert "held out" in out
    assert "all decode steps" not in out


def test_sweep_marks_one_request_in_sample(tmp_path, capsys):
    """--sweep on a trace with one request fits the table on the scored
    request; the fitted rows must say so (finding 2), not pass as foresight.
    """
    layers = _two_layers()
    events = [_prompt(0, [100, 101, 102]), _decode(3, 102), _decode(4, 103), _decode(5, 104)]
    trace = _trace(tmp_path, "one.ert1", layers, events)
    Path(str(trace) + ".accept").write_text(
        "verify 1 pos 3 rows 1 accepted 0\nverify 2 pos 4 rows 1 accepted 0\n"
        "verify 3 pos 5 rows 1 accepted 0\n")
    import argparse
    a = argparse.Namespace(trace=str(trace), accept=str(trace) + ".accept", delay=2, cap=96,
                           every=6, horizon=6, learned=False, sweep=True, bounds=False, budget=4)
    ecr.main_route(a)
    out = capsys.readouterr().out
    assert "no held-out request" in out
    assert "(in-sample)" in out


# ---------------------------------------------------------------- round 4 ----
# The blend: the engine's usage ranking is the BASE, a predictor only adds a
# bonus for experts it predicts with confidence above a threshold, and a swap
# happens only when the blended gain beats the resident's by a margin, so a
# weak guess cannot evict a hot resident.

def test_blend_without_bonus_replays_the_baseline(tmp_path):
    """Weight 0 (or a threshold no prediction clears) must replay `strata`
    exactly: the usage ranking is the base, not a replacement. Red before the
    blend policy existed: simulate raised on policy 'blend'.
    """
    layers = _two_layers()
    events = [_prompt(0, [100, 101, 102]), _decode(3, 102), _decode(4, 103),
              _decode(5, 104), _decode(6, 105), _decode(7, 106)]
    trace = _trace(tmp_path, "blendbase.ert1", layers, events)
    Path(str(trace) + ".accept").write_text(
        "".join("verify %d pos %d rows 1 accepted 0\n" % (i + 1, 3 + i) for i in range(5)))
    tr = ecr.load_route(trace)
    layers, graphs, steps, acc = ecr.route_to_steps(tr, str(trace) + ".accept")
    stream = ecr.step_stream(layers, steps)
    prov, _, _, _, _ = ecr.providers(layers, graphs, steps, acc, horizon=3, delay=1)
    hot = lambda si: ecr.normalize(prov["foresight"](si) * 0)   # a predictor of zeros
    base = ecr.simulate(layers, stream, "strata", adapt_every=1, delay=1, max_swaps=96)
    off = ecr.simulate(layers, stream, "blend", adapt_every=1, delay=1, max_swaps=96,
                       provider=hot, blend_weight=0.0, blend_threshold=0.1, blend_margin=1.5)
    assert off[0] == pytest.approx(base[0], abs=1e-12)
    assert off[3] == base[3]


def test_blend_rule_keeps_hot_residents_on_a_weak_guess():
    """A prediction below the threshold contributes no bonus, so the resident's
    usage stays above the candidate's and no swap happens. Red before: a plan
    fed a weak prediction swapped anyway.
    """
    nl, ne = 1, 8
    res = np.zeros((nl, ne), bool)
    res[0, 0] = True                 # the hot resident
    usage = np.zeros((nl, ne))
    usage[0, 0] = 40.0
    usage[0, 7] = 1.0                # the cold candidate the guess names
    p = np.zeros((nl, ne))
    p[0, 0] = 0.95
    p[0, 7] = 0.05                   # below the threshold
    plan = ecr.blend_plan(res, usage, p, weight=4.0, threshold=0.10, margin=0.0,
                          min_usage=2.0, decay=1.0)
    assert plan == [], "a weak guess below the threshold evicted a hot resident: %r" % plan


def test_blend_rule_swaps_on_a_confident_gain_above_the_margin():
    """Above the threshold the bonus lifts the candidate, but only a gain that
    beats the resident's by the margin may swap. Red before: the margin was
    ignored and every positive gain swapped.
    """
    nl, ne = 1, 8
    res = np.zeros((nl, ne), bool); res[0, 0] = True; res[0, 5] = True
    usage = np.zeros((nl, ne)); usage[0, 0] = 100.0; usage[0, 5] = 1.0; usage[0, 7] = 1.0
    p = np.zeros((nl, ne)); p[0, 5] = 0.60; p[0, 7] = 0.40
    small = ecr.blend_plan(res, usage, p, weight=0.05, threshold=0.10, margin=1.5,
                           min_usage=2.0, decay=1.0)
    assert small == [], "a gain under the margin swapped: %r" % small
    big = ecr.blend_plan(res, usage, p, weight=3.0, threshold=0.10, margin=1.5,
                         min_usage=2.0, decay=1.0)
    assert [(li, ei, lo, eo) for _, li, ei, lo, eo in big] == [(0, 7, 0, 0)], big


def test_oracle_blended_stays_inside_the_rule():
    """The oracle is blended the same way, so its upper bound includes the
    threshold and the margin: with a threshold no prediction clears it is the
    baseline, and with the bonus on it is no worse than its usage base."""
    nl, ne = 1, 8
    res = np.zeros((nl, ne), bool); res[0, 0] = True; res[0, 5] = True
    usage = np.zeros((nl, ne)); usage[0, 0] = 1.0; usage[0, 5] = 1.0; usage[0, 7] = 1.0
    oracle = np.zeros((nl, ne)); oracle[0, 5] = 0.6; oracle[0, 7] = 0.4
    opaque = ecr.blend_plan(res, usage, oracle, weight=1.0, threshold=1.1, margin=0.0,
                            min_usage=2.0, decay=1.0)
    assert opaque == []
    seen = ecr.blend_plan(res, usage, oracle, weight=1.0, threshold=0.1, margin=0.0,
                          min_usage=2.0, decay=1.0)
    assert [(li, ei, lo, eo) for _, li, ei, lo, eo in seen] == [(0, 7, 0, 0)]


def test_blend_sweep_report_is_held_out(tmp_path, capsys):
    """--blend-sweep reports the odd requests only and names the best blend per
    predictor beside the baseline and the oracle, with its swaps. Red before:
    the flag did not exist.
    """
    layers = _two_layers()
    events = [_prompt(0, [100, 101, 102]), _decode(3, 102),
              _prompt(0, [110, 111, 112]), _decode(3, 112)]
    trace = _trace(tmp_path, "blendsweep.ert1", layers, events)
    Path(str(trace) + ".accept").write_text(
        "verify 1 pos 3 rows 1 accepted 0\nverify 2 pos 3 rows 1 accepted 0\n")
    import argparse
    a = argparse.Namespace(trace=str(trace), accept=str(trace) + ".accept", delay=2, cap=96,
                           every=6, horizon=24, learned=False, sweep=False, bounds=False,
                           budget=4, blend_sweep=True, thresholds=(0.0, 0.1),
                           weights=(0.5, 1.0), horizons=(24, 48), margins=(0.0,))
    ecr.main_route(a)
    out = capsys.readouterr().out
    assert "held out" in out
    assert "blend, best per predictor" in out
    assert "best blend" in out


def _compiled_selftest(tmp_path):
    """tools/expert_route_trace_selftest.cpp against the pinned tree's writer;
    None when EXPERT_ROUTE_TRACE_SRC is unset (the device-free cells below then
    skip, as the byte cell does)."""
    src = os.environ.get("EXPERT_ROUTE_TRACE_SRC")
    if not src:
        return None
    src = Path(src)
    ser = src / "src" / "llama-expert-route-trace.cpp"
    if not ser.exists():
        return None
    exe = tmp_path / "selftest"
    r = subprocess.run(["g++", "-std=c++17", "-O1", "-I", str(src / "src"),
                        str(REPO_ROOT / "tools" / "expert_route_trace_selftest.cpp"), str(ser),
                        "-o", str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def test_rotation_each_file_replays_alone_and_concatenation_equals(tmp_path):
    exe = _compiled_selftest(tmp_path)
    if exe is None:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    plain = tmp_path / "plain.ert1"
    rot = tmp_path / "rot.ert1"
    out = subprocess.run([str(exe), str(plain)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    # a cap of 314 bytes splits the same event stream into two files
    env = dict(os.environ, LLAMA_EXPERT_ROUTE_TRACE_MAX_MB="0.0003")
    out = subprocess.run([str(exe), str(rot)], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    files = ecr.expand_traces(str(rot))
    assert len(files) == 2, files

    # each file carries its header and a LAYERS snapshot, so it replays alone
    for f in files:
        tr = ecr.load_route(f)
        assert not tr["torn"], f
        assert tr["events"][0]["kind"] == ecr.K_LAYERS, f
        layers, graphs, steps, _acc = ecr.route_to_steps(tr, None)
        assert steps, f          # a file with no decode step (the DRAFT one) has one

    # concatenated, the rotated files are the unrotated trace's records
    a = ecr.load_traces(ecr.expand_traces(str(plain)))
    b = ecr.load_traces(files)
    ae = [e for e in a["events"] if e["kind"] != ecr.K_LAYERS]
    be = [e for e in b["events"] if e["kind"] != ecr.K_LAYERS]
    assert [e["kind"] for e in ae] == [e["kind"] for e in be]
    for x, y in zip(ae, be):
        assert x.get("first_pos") == y.get("first_pos")
        if x["kind"] in (ecr.K_PROMPT, ecr.K_DECODE, ecr.K_VERIFY, ecr.K_DRAFT):
            assert x["tokens"].tolist() == y["tokens"].tolist()
            assert {il: ids.tolist() for il, ids in x["ids"].items()} == \
                   {il: ids.tolist() for il, ids in y["ids"].items()}
    # the second file's header snapshot is the residency at the rotation: the
    # selftest mutates slot 0 to expert 5 before the TABLES record, so the
    # snapshot already holds it (the TABLES record repeats the change)
    assert b["caches"][0][0]["residents"].tolist() == [5, 3, 7, 11, -1]
    own = ecr.trace_residency([dict(il=c["il"], n_expert=tr["n_expert"], n_slots=c["n_slots"],
                                   slots=c["residents"]) for c in b["caches"][0]], b)
    assert np.where(own[0])[0].tolist() == [3, 5, 7, 11]


def test_last_graph_written_at_request_end(tmp_path):
    exe = _compiled_selftest(tmp_path)
    if exe is None:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    complete = tmp_path / "complete.ert1"
    assert subprocess.run([str(exe), str(complete)]).returncode == 0
    accept = FIXTURES / "expert-route-trace-v2.accept"
    tr = ecr.load_traces(ecr.expand_traces(str(complete)))
    ecr.route_to_steps(tr, [str(accept)])
    assert tr["unwritten_last_graph"] == 0     # the decode at 203 is in the trace

    # the red case: the last counted graph is not written (as without the flush)
    short = tmp_path / "short.ert1"
    env = dict(os.environ, EXPERT_ROUTE_SELFTEST_SKIP_LAST="1")
    assert subprocess.run([str(exe), str(short)], capture_output=True, text=True, env=env).returncode == 0
    tr = ecr.load_traces(ecr.expand_traces(str(short)))
    ecr.route_to_steps(tr, [str(accept)])
    assert tr["unwritten_last_graph"] == 1


def test_train_curve_more_data_never_lowers_a_perfect_predictor(tmp_path, capsys):
    """Synthetic: five requests share one token pool, every token's routing is
    fixed, so the token table from the FIRST training request already predicts
    every held-out draft exactly; more training requests add identical rows.
    The curve must therefore not lower the held-out hits, and its training
    sizes must grow. Red mutation (seen failing): fit a size on the last
    `len - n` training requests instead of the first `n` -- the steps column
    becomes 24, 12, 12, not increasing."""
    import argparse
    nl, ne, k, slots = 2, 16, 2, 4
    layers = [dict(il=0, residents=np.array([0, 1, 2, 3])),
              dict(il=1, residents=np.array([4, 5, 6, 7]))]
    events = []
    for r in range(6):
        events.append(dict(kind=ecr.K_PROMPT, first_pos=0, n_tokens=1, tokens=[r],
                           ids={0: [r % ne, (r + 1) % ne], 1: [r % ne, (r + 1) % ne]}))
        for j in range(12):
            tok = 3000 + j
            e = tok % ne
            events.append(dict(kind=ecr.K_VERIFY, first_pos=10 + j, n_tokens=2, tokens=[tok, tok],
                               ids={0: [e, (e + 1) % ne, e, (e + 1) % ne],
                                    1: [e, (e + 1) % ne, e, (e + 1) % ne]}))
    trace = tmp_path / "curve.ert1"
    ecr.write_route(trace, nl, ne, k, [layers], events)
    tr = ecr.load_traces([str(trace)])
    layers, graphs, steps, _acc = ecr.route_to_steps(tr, None)
    _gid, rid = ecr.request_of(graphs, steps)
    a = argparse.Namespace(delay=1, cap=96, every=6, thresholds=(0.0,), weights=(10.0,),
                           margins=(0.0,), horizon=24)
    ecr.train_curve(a, layers, graphs, steps, tr, rid)
    rows = [l for l in capsys.readouterr().out.splitlines() if l.strip().startswith(tuple("0123456789"))]
    parsed = []
    for l in rows:
        parts = l.split()
        parsed.append((int(parts[0]), int(parts[1])))            # training requests, steps
    assert len(parsed) >= 2
    assert all(b[1] > a_[1] for a_, b in zip(parsed, parsed[1:])), parsed
    hits = [float(l.split("%")[0].split()[-1]) for l in rows]   # the draft-id table's column
    assert all(h >= hits[0] - 1e-9 for h in hits), hits


# ---------------------------------------------------------------- round 6 ----
# The continuous log's month: a restarted unit must not append a second header,
# a directory spanning a restart must replay, a rotation must keep a graph and
# its accept line together, and the MTP predictor must fit on training requests
# with a learned expert map, not by identity.

def test_reader_resynchronises_on_a_second_header(tmp_path):
    """An old defective file has a second header at its old tail (the writer
    appended into a reused pid's file). The reader replays past it.

    Red mutation (seen failing): drop the `b.find(b"ERT1", o0)` resync -- the
    second header reads as "unknown record kind 69" and load_route raises.
    """
    tail = tmp_path / "tail.ert1"
    ecr.write_route(tail, 6, 32, 4, [],
                    [dict(kind=ecr.K_DECODE, first_pos=999, n_tokens=1, tokens=[7],
                          ids={2: [5, 6], 5: [11, 12]})])
    body = tail.read_bytes()[20:]
    header = b"ERT1" + struct.pack("<IIII", 2, 6, 32, 4)
    bad = tmp_path / "bad.ert1"
    bad.write_bytes(FIXTURE.read_bytes() + header + body)
    tr = ecr.load_route(bad)
    assert tr["resynced"] == 1 and tr["torn"] is False, tr
    assert [(e["kind"], e.get("first_pos")) for e in tr["events"]] == [
        (ecr.K_LAYERS, None), (1, 0), (1, 10), (3, 200), (5, None), (2, 203),
        (4, 204), (2, 999)]


def test_expand_traces_from_dot_000_finds_its_siblings(tmp_path):
    """A file given as `.000` (the production first file) must find `.001`.
    Red mutation (seen failing): append `.001` to the whole argument -- it looks
    for `proc.000.ert1.001`, not `proc.001.ert1`."""
    first = tmp_path / "proc.000.ert1"
    second = tmp_path / "proc.001.ert1"
    ecr.write_route(first, 2, 16, 2, [], [])
    ecr.write_route(second, 2, 16, 2, [], [])
    assert ecr.expand_traces(str(first)) == [str(first), str(second)]


def test_load_traces_reads_a_later_layers_snapshot_as_residents(tmp_path):
    """After a rotation the second file's LAYERS is the current residency. The
    loader wrote it into `slots`, which route_to_steps never reads.
    Red mutation (seen failing): store `slots` instead of `residents`."""
    f0 = tmp_path / "a.ert1"
    f1 = tmp_path / "b.ert1"
    ecr.write_route(f0, 2, 16, 2, [[dict(il=0, residents=np.array([0, 1, 2, 3], np.int32))]], [])
    ecr.write_route(f1, 2, 16, 2, [[dict(il=0, residents=np.array([4, 5, 6, 7], np.int32))]], [])
    tr = ecr.load_traces([str(f0), str(f1)])
    assert tr["caches"][0][0]["residents"].tolist() == [4, 5, 6, 7]
    layers, _, _, _ = ecr.route_to_steps(tr, None)
    assert layers[0]["slots"].tolist() == [4, 5, 6, 7]


def test_unclean_restart_tolerates_one_missing_last_graph_a_file(tmp_path):
    """A killed process leaves its last accept line without a graph at the end
    of ITS sidecar. A directory history spans several processes: one such line a
    file is tolerated, not only at the whole history's end.
    Red mutation (seen failing): allow only one trailing line at the end -- the
    second file's line is off by one and read_accept exits."""
    def one(name, token):
        f = tmp_path / name
        ecr.write_route(f, 2, 16, 2,
                        [[dict(il=0, residents=np.array([0, 1, 2, 3], np.int32)),
                          dict(il=1, residents=np.array([4, 5, 6, 7], np.int32))]],
                        [dict(kind=ecr.K_DECODE, first_pos=0, n_tokens=1, tokens=[token],
                              ids={0: [2, 3], 1: [10, 11]})])
        return f
    a = one("proc-a.ert1", 100)
    b = one("proc-b.ert1", 101)
    aa = tmp_path / "proc-a.accept"
    bb = tmp_path / "proc-b.accept"
    aa.write_text("verify 1 pos 0 rows 1 accepted 0\n"
                  "verify 2 pos 1 rows 1 accepted 0\n")   # killed after its last line
    bb.write_text("verify 1 pos 0 rows 1 accepted 0\n")
    tr = ecr.load_traces([str(a), str(b)])
    _, graphs, steps, acc = ecr.route_to_steps(tr, [str(aa), str(bb)])
    assert len(steps) == 2
    assert acc == [0, 0]
    assert tr["unwritten_last_graph"] == 1


def test_mtp_matrix_fits_training_requests_and_is_not_identity(tmp_path):
    """The MTP predictor maps MTP experts to target experts through a per-layer
    matrix fitted on the even requests. The DRAFT input token at t predicts the
    draft at verify row t+1; an identity map or a fit that reads the odd request
    would give the wrong expert.
    Red mutation (seen failing): use the old identity token table -- the odd
    input expert 0 predicts 0, not the trained target 2; or fit all requests --
    matrix[0, 0, 3] gains the odd pair."""
    layers = [dict(il=0, n_expert=4, n_slots=2, residents=np.array([0, 1], np.int32))]
    events = [
        dict(kind=ecr.K_PROMPT, first_pos=0, n_tokens=1, tokens=[10],
             ids={0: [0]}),
        dict(kind=ecr.K_VERIFY, first_pos=10, n_tokens=2, tokens=[100, 200],
             ids={0: [0, 2]}),
        dict(kind=ecr.K_DRAFT, first_pos=9, n_tokens=1, tokens=[100],
             ids={6: [0]}),
        dict(kind=ecr.K_PROMPT, first_pos=0, n_tokens=1, tokens=[11],
             ids={0: [1]}),
        dict(kind=ecr.K_VERIFY, first_pos=20, n_tokens=2, tokens=[101, 201],
             ids={0: [1, 3]}),
        dict(kind=ecr.K_DRAFT, first_pos=19, n_tokens=1, tokens=[101],
             ids={6: [0]}),
    ]
    trace = tmp_path / "mtp.ert1"
    ecr.write_route(trace, 1, 4, 2, [layers], events)
    tr = ecr.load_route(trace)
    layers, graphs, steps, acc = ecr.route_to_steps(tr, None)
    drafts = tr["drafts"]
    assert [d["req"] for d in drafts] == [0, 1]
    gid, _ = ecr.request_of(graphs, steps)
    mat = ecr.mtp_expert_matrix(layers, drafts, graphs, gid, {0})
    assert mat[0, 0, 2] == pytest.approx(1.0)      # the even pair 0 -> 2
    assert mat[0, 0, 3] == 0.0                     # the odd pair never trained
    assert mat[0, 1, 3] == 0.0
    by_graph = {id(graphs[d["graph"]]): d for d in drafts}
    f = ecr.step_mtp_foresight(mat, layers, steps, by_graph)
    assert f[0][0].nonzero()[0].tolist() == [2]
    assert f[1][0].nonzero()[0].tolist() == [2]    # trained, not identity 0
    assert f[1][0][0] == 0.0


def test_rotation_keeps_the_accept_line_and_graph_in_one_file(tmp_path):
    """At a rotation the cache rotates before arcint writes the verify's accept
    line, so line and graph land in the new file (force_rotate makes it
    independent of the byte cap). Red: write the line before the rotation.
    Red mutation (seen failing): set LINE_BEFORE_ROTATE=1 -- the new file has a
    verify graph with no line and route_to_steps exits."""
    exe = _compiled_selftest(tmp_path)
    if exe is None:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    def run(name, before):
        out = tmp_path / name
        env = dict(os.environ, EXPERT_ROUTE_SELFTEST_ROTATE_AT_LINE="1")
        if before:
            env["EXPERT_ROUTE_SELFTEST_LINE_BEFORE_ROTATE"] = "1"
        assert subprocess.run([str(exe), str(out)], capture_output=True, text=True, env=env).returncode == 0
        return ecr.expand_traces(str(out))

    fixed = run("fixed", False)
    assert len(fixed) >= 2
    saw_line_and_graph = False
    for f in fixed:
        tr = ecr.load_route(f)
        assert not tr["torn"] and tr["events"][0]["kind"] == ecr.K_LAYERS
        ap = f + ".accept" if os.path.exists(f + ".accept") else None
        _, graphs, steps, acc = ecr.route_to_steps(tr, ap)   # no out-of-step line
        if ap is not None and steps:
            assert len(acc) == len(steps)
        if any(g["kind"] == ecr.K_VERIFY for g in graphs) and ap is not None:
            assert len(open(ap).read().splitlines()) == 1
            saw_line_and_graph = True
    assert saw_line_and_graph, fixed

    red = run("red", True)
    failed = []
    for f in red:
        ap = f + ".accept" if os.path.exists(f + ".accept") else None
        if ap is None:
            continue
        try:
            ecr.route_to_steps(ecr.load_route(f), ap)
        except SystemExit as e:
            failed.append((f, str(e)))
    assert failed, "line-before-rotation passed: the rotation cell is blind"


def test_pending_record_never_rotates_away_from_its_accept_line(tmp_path):
    """The cache's real call order for a verify: after_graph prepares (the cap
    is not reached), arcint writes the accept line, the MTP DRAFT records push
    the cap over, and advance writes the pending record -- which must not
    rotate, or the record lands in a new file while its line stays in the old
    one. The rotation happens at the next after_graph.

    Red mutation (seen failing): run the red arm, which rotates at advance
    while the count is non-zero (`EXPERT_ROUTE_SELFTEST_ROTATE_ON_PENDING=1`) --
    the old file keeps the line with no graph and the replay counts an
    unwritten last graph."""
    exe = _compiled_selftest(tmp_path)
    if exe is None:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    env = dict(os.environ, LLAMA_EXPERT_ROUTE_TRACE_MAX_MB="0.00014",
               EXPERT_ROUTE_SELFTEST_PENDING_ORDER="1")

    out = tmp_path / "pending.ert1"
    r = subprocess.run([str(exe), str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    files = ecr.expand_traces(str(out))
    assert len(files) == 2, files
    first, second = files
    tr = ecr.load_route(first)
    _, graphs, steps, _acc = ecr.route_to_steps(tr, first + ".accept")
    assert tr["unwritten_last_graph"] == 0, "the record rotated away from its accept line"
    assert [g["first_pos"] for g in graphs if g["kind"] == ecr.K_VERIFY] == [200]
    assert [d["first_pos"] for d in tr["drafts"]] == [300, 301, 302, 303]
    assert [g["first_pos"] for g in steps] == [200]
    # the rotation waits for the next after_graph, which writes its snapshot
    assert ecr.load_route(second)["events"][0]["kind"] == ecr.K_LAYERS

    # the red arm: rotate at the advance that writes the pending record
    env["EXPERT_ROUTE_SELFTEST_ROTATE_ON_PENDING"] = "1"
    red = tmp_path / "pending-red.ert1"
    r = subprocess.run([str(exe), str(red)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    rfiles = ecr.expand_traces(str(red))
    assert len(rfiles) >= 2, rfiles
    tr = ecr.load_route(rfiles[0])
    assert tr["events"][0]["kind"] == ecr.K_LAYERS
    assert not [e for e in tr["events"] if e["kind"] == ecr.K_VERIFY], "the red arm did not split"
    # the old file keeps the line with no graph: reading it alone fails loudly
    with pytest.raises(SystemExit, match="decode graphs"):
        ecr.route_to_steps(tr, rfiles[0] + ".accept")


def test_layer_split_pending_is_process_wide_and_snapshot_has_every_card(tmp_path):
    """A layer split: cache A and cache B each hold a contiguous layer range and
    each writes its graph record for the same verify. A's record is pending and
    its accept line is already in `.000`; B's `after_graph` prepare and B's
    `advance` prepare must both leave A's file alone. After BOTH records are
    written the next prepare rotates, and the new file's single LAYERS record
    carries all four layers (two a cache), so `.001` replays alone.

    Red mutation (seen failing): `rotate_now` returns on this cache's flag
    instead of `pending_graphs` -- the first prepare rotates while A's record
    waits, the selftest exits 8."""
    exe = _compiled_selftest(tmp_path)
    if exe is None:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    env = dict(os.environ, LLAMA_EXPERT_ROUTE_TRACE_MAX_MB="0.0002",
               EXPERT_ROUTE_SELFTEST_CROSS_CACHE="1")
    out = tmp_path / "split.ert1"
    r = subprocess.run([str(exe), str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    files = ecr.expand_traces(str(out))
    assert len(files) == 2, files
    first, second = files
    # A's line and A's record stay in the first file; B's record too
    tr = ecr.load_route(first)
    # both caches' records for the same verify are in the first file (a record
    # in `.001` would be a rotation away while A's line was in `.000`), and the
    # first file's own LAYERS already holds both cards
    assert sorted(e["first_pos"] for e in tr["events"] if e["kind"] == ecr.K_VERIFY) == [200, 200]
    # the rotation's LAYERS record: every cache's layers, one record
    second_tr = ecr.load_route(second)
    assert second_tr["events"][0]["kind"] == ecr.K_LAYERS
    assert [L["il"] for L in second_tr["caches"][0]] == [2, 5, 7, 9]
    # each file replays alone: both files carry all four layers
    assert [L["il"] for L in tr["caches"][0]] == [2, 5, 7, 9]


def test_advance_never_prepares_and_after_graph_does(tmp_path):
    """Round 7 moved the rotation to `after_graph`; `advance` and `flush_trace`
    only write the pending record. The round-7 cells mirrored that order inside
    the selftest, so restoring `trace_prepare()` into `advance` still passed all
    30 cells. This cell reads the cache's source and names the landmarks: the
    `advance` body must have no `trace_prepare` call, `after_graph` must call
    it once, and the rotation decision must test the process-wide count.

    Red mutation (seen failing): insert `trace_prepare();` before `progress();`
    in `llama_expert_cache::advance` -- the advance body check fails."""
    src = os.environ.get("EXPERT_ROUTE_TRACE_SRC")
    if not src:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    text = (Path(src) / "src" / "llama-expert-cache.cpp").read_text()

    def body(src_text, signature):
        start = src_text.index(signature)
        open_brace = text.index("{", start)
        depth, i = 0, open_brace
        while i < len(text):
            if src_text[i] == "{":
                depth += 1
            elif src_text[i] == "}":
                depth -= 1
                if depth == 0:
                    return src_text[open_brace:i + 1]
            i += 1
        raise AssertionError("unterminated body: " + signature)

    def calls(code):
        # comments name the rule; only a call is the defect
        return "\n".join(ln.split("//", 1)[0] for ln in code.splitlines())

    advance = calls(body(text, "bool llama_expert_cache::advance("))
    after = calls(body(text, "void llama_expert_cache::after_graph("))
    assert "trace_prepare(" not in advance, "advance prepares: a rotation can split a pending record"
    assert after.count("trace_prepare(") == 1, "after_graph does not take the rotation"
    trace_h = (Path(src) / "src" / "llama-expert-route-trace.h").read_text()
    trace_cpp = (Path(src) / "src" / "llama-expert-route-trace.cpp").read_text()
    assert "pending_graphs" in trace_h
    decision = body(trace_cpp, "bool llama_expert_route_trace_rotate_now(")
    assert "pending_graphs" in decision, "the rotation decision is not process-wide"
    assert "event_layers_all" in trace_cpp and "set_residents" in trace_cpp


def test_write_failure_stops_tracing_with_one_warn(tmp_path):
    """A closed fd (the selftest's cheap form of a full disk): the write fails,
    tracing stops -- one WARN, `opened` cleared, no sidecar, and the next record
    is a no-op rather than a second failure or a run-ending throw.

    Red mutation (seen failing): drop `opened = false;` in the writer's error
    path -- the program prints `write-fail opened 1` and the test fails."""
    exe = _compiled_selftest(tmp_path)
    if exe is None:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    out = tmp_path / "write-fail.ert1"
    env = dict(os.environ, EXPERT_ROUTE_SELFTEST_WRITE_FAIL="1")
    r = subprocess.run([str(exe), str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "write-fail opened 0 sidecar 0" in r.stdout, r.stdout
    warns = [ln for ln in r.stderr.splitlines() if "WARN: expert route trace" in ln]
    assert len(warns) == 1, warns
    assert "cannot write" in warns[0], warns[0]


def test_open_never_appends_and_creates_a_missing_directory(tmp_path):
    """A missing DIR is created; an existing file base is not appended into.
    Files are O_EXCL with a fresh sequence. Red mutation (seen failing): open
    with O_APPEND|O_CREAT -- the file base gains a header (and no .001), and a
    missing DIR becomes a base file, not a directory."""
    exe = _compiled_selftest(tmp_path)
    if exe is None:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    missing = tmp_path / "new-dir"
    assert subprocess.run([str(exe), str(missing)]).returncode == 0
    assert missing.is_dir()
    assert len(ecr.expand_traces(str(missing))) == 1
    # a second process into the same directory is a new family, not an append
    assert subprocess.run([str(exe), str(missing)]).returncode == 0
    files = ecr.expand_traces(str(missing))
    assert len(files) == 2, files
    for f in files:
        tr = ecr.load_route(f)
        assert not tr["torn"] and tr["resynced"] == 0

    base = tmp_path / "base.ert1"
    base.write_bytes(b"old")
    assert subprocess.run([str(exe), str(base)]).returncode == 0
    assert base.read_bytes() == b"old", "the writer appended into an existing file"
    nxt = Path(str(base) + ".001")
    assert nxt.exists(), "an existing file base did not advance to .001"
    assert not ecr.load_route(nxt)["torn"]


def test_non_numeric_max_mb_falls_back_to_the_default(tmp_path):
    """A non-numeric LLAMA_EXPERT_ROUTE_TRACE_MAX_MB used to parse as 0 and
    rotate every record. It now falls back to 256 MB with one WARN.
    Red mutation (seen failing): atoi/strtod without the validity check -- the
    run writes several files, not one."""
    exe = _compiled_selftest(tmp_path)
    if exe is None:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    out = tmp_path / "bad-mb"
    env = dict(os.environ, LLAMA_EXPERT_ROUTE_TRACE_MAX_MB="many")
    r = subprocess.run([str(exe), str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "WARN: expert route trace" in r.stderr
    assert len(ecr.expand_traces(str(out))) == 1


def test_accept_directory_and_single_directory(tmp_path):
    """`--accept DIR` returns its *.accept files (it used to return None); a
    directory with --single is refused with a clear message.
    Red mutation (seen failing): filter a directory's *.ert1 for .accept."""
    d = tmp_path / "acc"
    d.mkdir()
    (d / "b.accept").write_text("verify 1 pos 0 rows 1 accepted 0\n")
    (d / "a.accept").write_text("verify 1 pos 0 rows 1 accepted 0\n")
    fs = ecr.expand_accept(str(d))
    assert sorted(os.path.basename(f) for f in fs) == ["a.accept", "b.accept"]

    import argparse
    trace = tmp_path / "single.ert1"
    ecr.write_route(trace, 2, 16, 2, [_two_layers()], [])
    a = argparse.Namespace(trace=str(tmp_path), accept=None, single=True, delay=2, cap=96,
                           every=6, horizon=24, learned=False, sweep=False, bounds=False,
                           budget=4, blend_sweep=False, train_curve=False)
    with pytest.raises(SystemExit, match="--single needs one trace file"):
        ecr.main_route(a)


def test_accept_directory_is_used_by_main(tmp_path, capsys):
    """The trace has a sidecar, passed as a directory: no "no accept sidecar".
    Red mutation (seen failing): expand_traces then filter .accept -> None."""
    layers = _two_layers()
    trace = _trace(tmp_path, "accmain.ert1", layers,
                   [_prompt(0, [100, 101, 102]), _decode(3, 102)])
    Path(str(trace) + ".accept").write_text("verify 1 pos 3 rows 1 accepted 0\n")
    d = tmp_path / "sidecars"
    d.mkdir()
    Path(str(d / (trace.name + ".accept"))).write_text(Path(str(trace) + ".accept").read_text())
    import argparse
    a = argparse.Namespace(trace=str(trace), accept=str(d), single=False, delay=2, cap=96,
                           every=6, horizon=1, learned=False, sweep=False, bounds=False,
                           budget=4, blend_sweep=False, train_curve=False)
    ecr.main_route(a)
    out = capsys.readouterr().out
    assert "no accept sidecar" not in out


def test_mtp_route_ids_are_expanded_into_the_graph():
    """The MTP draft block's routed ids are a dead end of the graph: marked
    output, consumed by nothing. The scheduler allocates only the nodes the
    forward walk reaches, so without an explicit expand the tensor's buffer
    stays null, decode's read-back skips it, and no DRAFT record is written
    although the graph ran -- round 8's card run wrote 0 DRAFT records with MTP
    accepting ~65 %.

    Device-free repro (round 9, code): a CPU qwen4exp MTP context, one draft
    decode a token, the writer opened without a cache -- 0 DRAFT records on the
    mutation, 2 with the expand.

    Red mutation (seen failing): delete `ggml_build_forward_expand(gf, ids_out);`
    from the trace's MTP block -- this cell names the landmark and fails.

    The block must also sit AFTER the FFN's `ggml_build_forward_expand(gf,
    weights)` (its `call early so that topk-moe can be used`). A graph's node
    order is the expand order, and 0064's `ggml_opencl_can_fuse_topk_moe` wants
    ten consecutive nodes: SOFT_MAX, RESHAPE(src n[0]), ARGSORT(src n[0]),
    VIEW(src n[2]), GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE. With the
    block before that expand, `expand(ids_out)` appends SOFT_MAX, ARGSORT, VIEW,
    CONT first and the weights expand adds the rest after them, so the fusion
    scan sees ARGSORT at n[1] and the MTP draft's router runs unfused -- and
    `fusion-coder-night.md` records that the fused kernel's softmax sum order
    changes greedy texts and draft acceptance, so that drafter is not the served
    drafter. The CONT consumer placed after the ten nodes keeps the fusion:
    `ggml_can_fuse_subgraph` exempts nodes listed in outs, and the VIEW is
    outs[0] (the cache's own ids_out is placed exactly that way).

    Red mutation (seen failing): move the trace block back before
    `ggml_build_forward_expand(gf, weights)` -- this cell's order assertion
    fails."""
    src = os.environ.get("EXPERT_ROUTE_TRACE_SRC")
    if not src:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    text = (Path(src) / "src" / "llama-graph.cpp").read_text()
    start = text.index("// the trace's MTP block")
    stop  = text.index("res->set_route_ids(il, ids_out);", start)
    block = text[start:stop]
    assert "ggml_set_output(ids_out)" in block, "the block no longer marks the ids an output"
    out_at = block.index("ggml_set_output(ids_out)")
    exp_at = block.find("ggml_build_forward_expand(gf, ids_out)")
    assert exp_at != -1, "the MTP ids are not expanded into the graph: their buffer stays null and no DRAFT record is written"
    assert exp_at > out_at, "the expand must come after the output mark"
    assert text.count("res->set_route_ids(il, ids_out);") == 1, "a second MTP ids block appeared"
    weights_at = text.index("ggml_build_forward_expand(gf, weights);")
    assert weights_at < start, (
        "the trace's MTP block sits before the weights expand: its CONT is appended "
        "before the router's ten nodes, so ggml_opencl_can_fuse_topk_moe no longer "
        "matches and the MTP draft's router runs unfused -- a different drafter from "
        "the served one")


def test_fused_router_name_survives_the_graph_callback():
    """The card check's fused line is keyed on a tensor name, and two things
    eat it (round 11 review, `code`):

    - `llama_context::graph_get_cb` (llama-context.cpp) renames every layer's
      tensor with `ggml_format_name(cur, "%s-%d", name, il)`, so a
      `ggml_set_name(ids_out, "ffn_moe_ids_trace")` placed BEFORE
      `cb(ids_out, "ffn_moe_ids_trace", il)` is overwritten to
      `ffn_moe_ids_trace-<il>`;
    - the backend's scan compared `sizeof("ffn_moe_ids_trace")` bytes, which
      includes the NUL, so even the exact name compared 19 bytes against an
      18-byte name: the match could never succeed.

    Both are asserted at source: the name is set after cb, and the scan is the
    18-byte prefix (sizeof minus one), never the NUL-terminated form.

    Red mutation (seen failing): move `ggml_set_name(ids_out, ...)` above
    `cb(ids_out, ...)` -- the order assertion fails; or compare
    `sizeof("ffn_moe_ids_trace")` without `- 1` -- the prefix assertion fails,
    the byte compare is 19 bytes against an 18-byte name."""
    src = os.environ.get("EXPERT_ROUTE_TRACE_SRC")
    if not src:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    graph = (Path(src) / "src" / "llama-graph.cpp").read_text()
    start = graph.index("// the trace's MTP block")
    stop = graph.index("res->set_route_ids(il, ids_out);", start)
    block = graph[start:stop]
    cb_at = block.find('cb(ids_out, "ffn_moe_ids_trace", il);')
    name_at = block.find('ggml_set_name(ids_out, "ffn_moe_ids_trace");')
    assert cb_at != -1, "the MTP ids CONT is no longer handed to the graph callback"
    assert name_at != -1, "the MTP ids CONT no longer carries the fused-router name"
    assert name_at > cb_at, (
        "the fused-router name is set before the graph callback; cb renames the "
        "tensor to \"ffn_moe_ids_trace-<il>\" and the backend compare misses it")

    opencl = (Path(src) / "ggml" / "src" / "ggml-opencl" / "ggml-opencl.cpp").read_text()
    scan_at = opencl.index('strncmp(cgraph->nodes[j]->name, "ffn_moe_ids_trace"')
    scan_end = opencl.index(";", scan_at)
    scan = opencl[scan_at:scan_end]
    assert "sizeof(\"ffn_moe_ids_trace\") - 1" in scan, (
        "the fused-router scan compares sizeof(\"ffn_moe_ids_trace\") bytes, which "
        "includes the NUL: the callback's \"ffn_moe_ids_trace-<il>\" and even the "
        "exact name never match, so the card check's line cannot print")


def test_fused_router_line_is_warned_once_per_process():
    """The card check reads the fused-router line out of the unit's log, so it
    must be emitted where arcint keeps it (`code`, round 11 review):

    - at WARN once per process: an INFO line is routed to verbose and dropped
      at the units' default log level, and printed per fused dispatch of every
      MTP decode it floods a month-long trace (~100 lines a second);
    - with the graph scan decided once, not per fused dispatch.

    Red mutation (seen failing): replace `GGML_LOG_WARN(` with `GGML_LOG_INFO(`
    in the fused-router block -- the WARN assertion fails; or delete the
    `mtp_trace_logged` once-guard -- the once assertion fails."""
    src = os.environ.get("EXPERT_ROUTE_TRACE_SRC")
    if not src:
        pytest.skip("EXPERT_ROUTE_TRACE_SRC unset")
    opencl = (Path(src) / "ggml" / "src" / "ggml-opencl" / "ggml-opencl.cpp").read_text()
    start = opencl.index("// a card check's evidence that the trace-on drafter")
    stop = opencl.index("float cmin = 0.0f", start)
    block = opencl[start:stop]
    assert 'GGML_LOG_WARN("opencl: topk-moe fused in the MTP trace graph' in block, (
        "the fused-router line is not a WARN: arcint routes ggml INFO to verbose and "
        "drops it at the units' default level, so the card check never reads it")
    assert "GGML_LOG_INFO(" not in block, "an INFO form of the fused-router line returned"
    assert "mtp_trace_logged" in block and "if (!mtp_trace_logged)" in block, (
        "the fused-router line is not held to once per process: it prints per fused "
        "dispatch of every MTP decode, a flood on a month-long trace")
    assert "sizeof(\"ffn_moe_ids_trace\") - 1" in block, "the scan is not a prefix compare"
    assert "cgraph != mtp_trace_scanned_graph" in block, (
        "the graph scan is not decided once per graph: with the trace on every fused "
        "dispatch scans cgraph->n_nodes, including the target graph's routers that "
        "never hold the name, so the trace-on A/B prices the scan as the trace")
