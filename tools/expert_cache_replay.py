#!/usr/bin/env python3
"""Offline replay of the libllama engine's expert cache over a routing trace.

Two trace formats are read:

* ECT1, the 2026-10-15 measurement switch (`LLAMA_EXPERT_CACHE_TRACE` on the
  llama tree's branch p0057-trace, not in contrib/llama.cpp/patches): header
  ("ECT1", the layer count, then per layer il, n_expert, n_slots, one expert's
  bytes as u64 and the slotted experts at load) and per counted decode step its
  tokens and each layer's routed ids (u32 count, i32 ids).
* ERT1, campaign expert-foresight's production trace (patch 0069, a host
  read-back of the ids the cache already reads; arcint's
  `--llama-expert-route-trace DIR`, off by default). One file a process,
  rotated at a size cap with a header and a LAYERS snapshot in every file; the
  reader takes a directory or a list as ONE ordered history. Header "ERT1",
  version, n_layer, n_expert, top-k,
  then records (src/llama-expert-route-trace.h): LAYERS (a cache's
  layers with their resident slots at load), PROMPT / DECODE / VERIFY / DRAFT
  graphs (tokens and each layer's routed ids, token-major, top-k in rank
  order) and TABLES (every residency change, so the run's own set can be
  rebuilt). Version 2 adds every graph's first position. A verify's accepted
  drafts are NOT in the trace (the cache cannot see the sampler); arcint writes
  them beside it, `<path>.accept`, one line a decode graph: "verify <ord> pos
  <p> rows <n> accepted <a>". Every request restarts at position 0, so the same
  position recurs across requests: the lines are walked in request order and
  checked against each graph's position AND row count, never a global position
  set. A prompt's checkpoint tail is a PROMPT record with no line, so it cannot
  shift the walk, and an out-of-step line fails loudly. arcint writes the
  process's last graph at a request's end (the cache otherwise writes a graph
  only at the next graph's advance), so one trailing line is kept apart as an
  unwritten last graph when the flush did not run. An MTP DRAFT record is the
  draft block's own decode with its INPUT tokens (step 0's accepted target
  token, then the previous draft) and the MTP block's routed ids -- NOT the
  drafted tokens; it is the `mtp` predictor's source, never a cache step, and
  its experts do not map to target experts by identity. A 1-token DECODE at
  a PROMPT's end with no line is the non-spec loop's prompt-last call and is
  relabeled PROMPT, outside the scored steps. A torn last record is ignored.

Policies (strata as src/llama-expert-cache.cpp runs it, patches 0021 and 0042;
lfru as patch 0055 ran it, dropped from the series in 0.8.1):
  strata  per-layer usage counts decayed by 0.7 an adapt, a swap while the bank
          expert's usage beats the slot's by 1.5 (Strata's adapt);
  lfru    Infernix's score uses / (now - last + 1) per layer, the round's
          experts protected, up to --budget admissions a layer;
  glfru   the same score over one pool of frames for all layers, so a swap may
          move a frame from one layer to another (Infernix's global LFRU).
  plan    a planned future: per plan, a provider's predicted per-layer expert
          counts rank banks against slots and swap while the gain holds -- the
          same cadence, 96-swap cap and landing as the engine.
"exact" uses Infernix's clock (a round ages an expert by the layer count) and
only the step's misses as candidates. Every 6 decode tokens a plan, capped at
96 swaps; its swaps land --delay counted steps later and no plan runs while
some are in flight (the engine's two-phase non-blocking swaps). Hits count
against the residency at the step, as the engine counts them. Frames are
counted, not bytes: the layers' experts are not one size.

On an ERT1 trace the predictor plans (campaign expert-foresight stage 1) are
fed into that same mechanism and scored against the baseline and the oracle:
  foresight the verify's draft token ids known BEFORE the verify, mapped
          through a per-layer token->expert table fitted on the even requests --
          the routing of a verify (accepted or not) is not used as a feature;
  prompt  the routing of the request's own prompt, as a prior at decode start;
  both    the two, normalized per layer and summed;
  mtp     the DRAFT record's MTP expert counts at each input position, through
          a per-layer MTP-expert -> target-expert matrix fitted only on the even
          requests (the input token at t pairs with the verify's target row
          t+1, the drafted token); no DRAFT record of a training request in the
          trace means no serving result tests this mechanism yet;
  learned per layer, a linear map of recent routing (+1, +2, +6 and +12 steps),
          the prompt prior and the foresight counts to the next --horizon tokens'
          occurrence from the landing, fitted on even requests and scored on odd
          ones (stage 1b), windows cut at request ends.

Round 4 adds the blended plan (`simulate(..., "blend")`, `blend_plan`): the
engine's usage ranking stays the base, a predictor adds a bonus only where its
per-layer share clears --blend-threshold, and a swap runs only while the
blended gain beats the resident's by --blend-margin. --blend-sweep selects the
threshold, the weight, the margin and H on the even requests (--blend-tune
scored selects on the scored steps) and scores the odd ones, best per
predictor, beside the baseline and the blended oracle.

Reporting. The token table is fitted on the even requests. The default report
therefore scores the odd requests (held out) whenever they exist; without one
it scores all steps and marks every fitted row (foresight, both, learned) as
in-sample, so a fitted table is never read as foresight. --sweep does the same,
and --learned adds the linear map to the held-out report.

--bounds adds what no online policy can beat on the trace: the best fixed set per
layer in hindsight, a planner inside the same mechanism that knows the next H
steps' routing, and Belady's MIN per layer. On ERT1 the oracle is counted in
tokens (--horizon, default 24) instead of steps, as the campaign states it.

Usage: expert_cache_replay.py <trace> [--budget N] [--delay N] [--cap N] [--every N]
       [--bounds] [--accept PATH] [--learned] [--horizon N] [--sweep]
       [--blend-sweep] [--thresholds ...] [--weights ...] [--margins ...] [--horizons ...]
       (needs numpy)
The older replay tools for the retired engine's traces and policies are at the
tag openvino-history (docs/history.md).
"""
import os, struct, argparse
import numpy as np


def load(path):
    b = open(path, "rb").read()
    assert b[:4] == b"ECT1"
    o = 4
    (nl,) = struct.unpack_from("<I", b, o); o += 4
    layers = []
    for _ in range(nl):
        il, ne, ns = struct.unpack_from("<iii", b, o); o += 12
        (eb,) = struct.unpack_from("<Q", b, o); o += 8
        ex = np.frombuffer(b, "<i4", ns, o).copy(); o += 4 * ns
        layers.append(dict(il=il, n_expert=ne, n_slots=ns, eb=eb, slots=ex))
    steps = []
    while o + 4 <= len(b):
        (nt,) = struct.unpack_from("<I", b, o); o += 4
        ids = []
        for _ in range(nl):
            (c,) = struct.unpack_from("<I", b, o); o += 4
            ids.append(np.frombuffer(b, "<i4", c, o).copy()); o += 4 * c
        steps.append((nt, ids))
    return layers, steps


# ------------------------------------------------------- Strata profile ----
# The STRP file arcint's --llama-expert-profile reads (read_profile in
# src/llama-expert-cache.cpp of the llama tree) and Strata's tools/
# make_profile.py writes, as patch 0070's writer does: "STRP", u32 version 1,
# n_layer, n_expert, slots, n_ranked, then n_ranked (u16 layer, u16 expert),
# then the per-layer rank table (i32 expert -> rank); the reader stops after
# the pairs and does not require the table.
def read_profile(path, n_layer=48, n_expert=512):
    b = open(path, "rb").read()
    if len(b) < 24:
        raise SystemExit("%s: %d bytes, shorter than a 24-byte STRP header" % (path, len(b)))
    if b[:4] != b"STRP":
        raise SystemExit("%s: not a Strata profile (magic %r)" % (path, b[:4]))
    ver, nl, ne, slots, n = struct.unpack_from("<5I", b, 4)
    if ver != 1:
        raise SystemExit("%s: profile version %d, this reader reads version 1" % (path, ver))
    if (nl, ne) != (n_layer, n_expert):
        raise SystemExit("%s: %dx%d, not %dx%d" % (path, nl, ne, n_layer, n_expert))
    if n > n_layer * n_expert:
        raise SystemExit("%s: %d ranked pairs, more than the geometry's %d" % (path, n, n_layer * n_expert))
    if len(b) < 24 + 4 * n:
        raise SystemExit("%s: %d ranked pairs, %d bytes (truncated)" % (path, n, len(b)))
    return [struct.unpack_from("<HH", b, 24 + 4 * i) for i in range(n)], slots


def profile_slots(layers, ranked):
    """The layers with their slots seeded from a profile's ranking, a NEW list,
    modelling the ENGINE's global walk (src/llama-expert-cache.cpp init): one
    pass over the whole ranking, each pair takes the next free slot of its own
    layer until the run's budget is spent, then a layer with no slot gets its
    zero expert. The replay's budget is the trace's total slot count (the trace
    carries no expert bytes), so this counts frames, not bytes: a per-layer
    expert's byte size differs (a Q8_0 down against gate/up) and the engine's
    walk spends bytes. The arms therefore share one frame budget and differ
    only in which experts fill it; the slot count a layer ends with is the
    walk's distribution, not the trace's 93-155: the walk gives a layer more
    slots than the trace's count when the ranking spends the budget there, and
    the layers after it fewer. Every layer's total slots is the count the walk
    actually gave it (n_slots too), so the placement is the engine's.

    A profile with fewer pairs of a layer leaves the walk's remaining budget to
    later layers; the walk skips a pair when its layer would lose its last bank
    entry (`chosen + 1 >= n_expert`, as init does).
    """
    n_layers = len(layers)
    per_layer = [0] * n_layers
    for L in layers:
        per_layer[L["il"]] += int(L["n_slots"])
    budget = sum(per_layer)   # frames, see above
    by_il = {L["il"]: i for i, L in enumerate(layers)}
    chosen = [[] for _ in layers]
    # duplicate guard, as init's slot_of[e] == -2 marker
    taken = [set() for _ in layers]
    used = 0
    for il, e in ranked:
        i = by_il.get(il)
        if i is None or e < 0 or e >= layers[i]["n_expert"]:
            continue
        if used + 1 > budget:
            break
        if e in taken[i] or len(chosen[i]) + 1 >= layers[i]["n_expert"]:
            continue
        taken[i].add(e)
        chosen[i].append(e)
        used += 1
    # every layer keeps at least one slot: its bank then fits 512 matrices with
    # its zero entry (init's `chosen[i].empty()` branch)
    for i, L in enumerate(layers):
        if not chosen[i] and L["n_expert"] > 0:
            chosen[i].append(0)
            used += 1
    # apply the walk's distribution, not the trace's per-layer counts: a layer
    # keeps exactly the slots the walk gave it (see the docstring)
    out = []
    for i, L in enumerate(layers):
        ex = np.asarray(chosen[i], np.int32)
        out.append(dict(L, n_slots=int(ex.size), slots=ex))
    return out


def compare_profiles(a, layers, steps, rid, paths, every_on, n_layer):
    """The held-out hit rate of several profiles at one slot budget, adaptation
    on (the served mode, `every_on` tokens a plan) and off (placement only).
    The trace's own resident set is the first row: it is the run the trace was
    captured from, so it is what the served unit's profile does on this
    workload."""
    odd = [i for i, r in enumerate(rid) if r % 2 == 1]
    idx = odd if odd else list(range(len(steps)))
    tag = ("the odd requests (held out)" if odd
           else "all %d decode steps (no odd request: not held out)" % len(steps))
    stream = step_stream(layers, [steps[i] for i in idx])
    print("profile comparison, %d decode steps of %s, delay %d, %d swaps a plan every %d tokens, "
          "adapt off means the profile's placement never adapts:" % (len(idx), tag, a.delay, a.cap, every_on))
    print("  %-46s %-12s %8s %8s %8s" % ("placement", "adapt", "hits", "swaps", "slots"))
    arms = [("the trace's own resident set (the captured run)", layers)]
    for p in paths:
        ranked, _slots = read_profile(p, n_layer, layers[0]["n_expert"])
        arms.append((os.path.basename(p), profile_slots(layers, ranked)))
    for name, L in arms:
        for mode, every in (("on", every_on), ("off", 0)):
            h, _, occ, sw = simulate(L, stream, "strata", delay=a.delay, max_swaps=a.cap, adapt_every=every)
            print("  %-46s %-12s %7.2f %% %8d %8d" % (name, mode, 100 * h, sw, int(sum(occ))))
    if not odd:
        print("  (no held-out request: these hit rates are not held out)")


K_LAYERS, K_PROMPT, K_DECODE, K_VERIFY, K_DRAFT, K_TABLES = 0, 1, 2, 3, 4, 5


def route_graph_kind(n_tokens, decode_max, all_outputs, has_tokens):
    """The serializer's kind decision (`llama_expert_route_graph_kind`), mirrored
    here so it is tested without a C++ compiler: a prompt call does not output
    every row (its 4-token checkpoint tail included), an embedding batch has no
    token ids, a batch longer than decode_max is a prompt; otherwise one token
    is a DECODE and 2..decode_max a VERIFY."""
    prompt = (not all_outputs) or (not has_tokens) or n_tokens > decode_max
    if prompt:
        return K_PROMPT
    return K_DECODE if n_tokens == 1 else K_VERIFY


def prompt_last_candidates(graphs):
    """Graph indices a 1-token DECODE that sits at the end of the preceding
    PROMPT graph: arcint's non-spec path runs a prompt's last token alone for
    its logits (`backend_llama.cpp`, all but the last prefilled), so the trace
    calls it a DECODE while the sampler never writes an accept line for it."""
    out = set()
    for i in range(1, len(graphs)):
        p, g = graphs[i - 1], graphs[i]
        if (p["kind"] == K_PROMPT and p["first_pos"] >= 0
                and g["kind"] == K_DECODE and g["n_tokens"] == 1
                and g["first_pos"] == p["first_pos"] + p["n_tokens"]):
            out.add(i)
    return out


def _need(b, o, n):
    if o + n > len(b):
        raise struct.error("torn record")


def load_route(path):
    """Read an ERT1 trace. Returns a dict; a torn trailing record is dropped.

    A pre-round-6 file can carry a second header at its old tail: the old
    writer appended into a reused pid's file. On a record parse failure the
    reader looks for an "ERT1" header with the same geometry from the failed
    record's start on, and replays the records that follow it; `resynced`
    reports that. A mere magic inside an id record does not qualify: the whole
    header, version and geometry must match."""
    b = open(path, "rb").read()
    assert b[:4] == b"ERT1", "not an ERT1 trace: %r" % b[:4]
    o = 4
    ver, nl, ne, nu = struct.unpack_from("<IIII", b, o); o += 16
    caches, events, torn = [], [], False
    resynced = 0
    while o < len(b):
        o0 = o
        kind = b[o]; o += 1
        try:
            if kind == K_LAYERS:
                _need(b, o, 4); (n,) = struct.unpack_from("<I", b, o); o += 4
                L = []
                for _ in range(n):
                    _need(b, o, 8); il, ns = struct.unpack_from("<ii", b, o); o += 8
                    _need(b, o, 4 * ns)
                    r = np.frombuffer(b, "<i4", ns, o).copy(); o += 4 * ns
                    L.append(dict(il=il, n_slots=ns, residents=r))
                # several LAYERS records in one file (a layer split writes one
                # a cache at init, and a rotation writes one over them all):
                # one cache list, so each file replays alone
                if not caches:
                    caches.append(L)
                else:
                    cur = {c["il"]: c for c in caches[0]}
                    for c in L:
                        if c["il"] in cur:
                            cur[c["il"]]["residents"] = c["residents"].copy()
                            cur[c["il"]]["slots"] = c["residents"].copy()
                            cur[c["il"]]["n_slots"] = c["n_slots"]
                        else:
                            caches[0].append(dict(c))
                events.append(dict(kind=kind, layers=L))
            elif kind in (K_PROMPT, K_DECODE, K_VERIFY, K_DRAFT):
                pos = -1
                if ver >= 2:
                    _need(b, o, 4); (pos,) = struct.unpack_from("<i", b, o); o += 4
                _need(b, o, 4); (nt,) = struct.unpack_from("<I", b, o); o += 4
                _need(b, o, 4 * nt)
                tok = np.frombuffer(b, "<i4", nt, o).copy(); o += 4 * nt
                _need(b, o, 4); (nL,) = struct.unpack_from("<I", b, o); o += 4
                ids = {}
                for _ in range(nL):
                    _need(b, o, 8); il, c = struct.unpack_from("<ii", b, o); o += 8
                    _need(b, o, 4 * c)
                    ids[il] = np.frombuffer(b, "<i4", c, o).copy(); o += 4 * c
                events.append(dict(kind=kind, first_pos=pos, n_tokens=nt, tokens=tok, ids=ids))
            elif kind == K_TABLES:
                _need(b, o, 4); (n,) = struct.unpack_from("<I", b, o); o += 4
                ch = []
                for _ in range(n):
                    _need(b, o, 9)
                    il, e = struct.unpack_from("<ii", b, o); o += 8
                    r = b[o]; o += 1
                    ch.append((il, e, r))
                events.append(dict(kind=kind, changes=ch))
            else:
                raise struct.error("unknown record kind %d" % kind)
        except struct.error:
            # the failed record's first byte may itself start the second header
            idx = b.find(b"ERT1", o0)
            while idx >= 0:
                if idx + 20 <= len(b):
                    v2, nl2, ne2, nu2 = struct.unpack_from("<IIII", b, idx + 4)
                    if v2 in (1, 2) and (nl2, ne2, nu2) == (nl, ne, nu):
                        resynced += 1
                        o = idx + 20
                        break
                idx = b.find(b"ERT1", idx + 1)
            if idx < 0:
                torn = True
                break
    return dict(version=ver, n_layer=nl, n_expert=ne, n_used=nu, caches=caches,
                events=events, torn=torn, resynced=resynced)

def _rotation_parts(arg):
    """(root, extension, the sequence in arg or -1) for `f`, `f.001`,
    `f.000.ert1`: the siblings of any sequence are found from the root."""
    import re
    m = re.match(r"^(.*?)(?:\.(\d{3}))?(\.ert1)?$", arg)
    return m.group(1), m.group(3) or "", int(m.group(2)) if m.group(2) else -1


def expand_traces(arg, siblings=True):
    """A trace argument as its files, ordered: a file is itself; a directory
    gives its *.ert1 files by mtime then name (a process's rotation writes .000,
    .001, ... in order; several processes' files interleave chronologically).
    A file given as `.000` (or `.000.ert1`) also finds its `.001` siblings; it
    does not read backwards, and it is always first."""
    if os.path.isdir(arg):
        fs = [os.path.join(arg, f) for f in os.listdir(arg) if f.endswith(".ert1")]
        fs.sort(key=lambda f: (os.path.getmtime(f), f))
        return fs
    if not os.path.exists(arg):
        raise SystemExit("no trace at %s" % arg)
    # a first file and its rotations, `.001`, `.002`, ... (the directory form is
    # the production one; this keeps a test's explicit first file readable)
    root, ext, seq = _rotation_parts(arg)
    fs, i = [arg], max(seq + 1, 1)
    while siblings and os.path.exists("%s.%03d%s" % (root, i, ext)):
        fs.append("%s.%03d%s" % (root, i, ext)); i += 1
    return fs


def expand_accept(arg):
    """The accept sidecars for --accept: a directory gives its *.accept files
    (a directory of traces no longer yields None); a trace file gives the
    sidecars of it and its rotation siblings; a sidecar is itself."""
    if os.path.isdir(arg):
        fs = [os.path.join(arg, f) for f in os.listdir(arg) if f.endswith(".accept")]
        fs.sort(key=lambda f: (os.path.getmtime(f), f))
        if not fs:
            raise SystemExit("no *.accept sidecar in %s" % arg)
        return fs
    if arg.endswith(".accept"):
        if not os.path.exists(arg):
            raise SystemExit("no accept sidecar at %s" % arg)
        return [arg]
    fs = [p + ".accept" for p in expand_traces(arg) if os.path.exists(p + ".accept")]
    if not fs:
        raise SystemExit("no accept sidecar beside %s" % arg)
    return fs


def load_traces(paths):
    """Several ERT1 files as ONE ordered history. Each file keeps its own
    header; a later file's LAYERS record is the residency at that moment, so it
    updates the first file's cache layers instead of adding a second cache.
    Events are concatenated in file order: a request that spans a rotation
    stays one request."""
    if not paths:
        raise SystemExit("no trace files")
    tr = load_route(paths[0])
    for ev in tr["events"]:
        ev["file"] = paths[0]
    for p in paths[1:]:
        nxt = load_route(p)
        if nxt["n_layer"] != tr["n_layer"] or nxt["n_expert"] != tr["n_expert"] \
           or nxt["n_used"] != tr["n_used"]:
            raise SystemExit("%s: geometry %d/%d/%d differs from %s's %d/%d/%d" % (
                p, nxt["n_layer"], nxt["n_expert"], nxt["n_used"], paths[0],
                tr["n_layer"], tr["n_expert"], tr["n_used"]))
        tr["torn"] = tr["torn"] or nxt["torn"]
        tr["resynced"] = tr.get("resynced", 0) + nxt.get("resynced", 0)
        for ev in nxt["events"]:
            if ev["kind"] == K_LAYERS:
                cur = {L["il"]: L for L in tr["caches"][0]} if tr["caches"] else {}
                for L in ev["layers"]:
                    if L["il"] in cur:
                        # route_to_steps reads "residents"; a later file's
                        # snapshot used to land in an unread "slots"
                        cur[L["il"]]["residents"] = L["residents"].copy()
                        cur[L["il"]]["slots"] = L["residents"].copy()
                        cur[L["il"]]["n_slots"] = L["n_slots"]
                    else:
                        tr["caches"][0].append(dict(L))   # a cache the first file did not hold
                if not tr["caches"]:
                    tr["caches"].append(list(ev["layers"]))
            else:
                ev["file"] = p
                tr["events"].append(ev)
    return tr


def read_accept(path, graphs, relabel_candidates=None, report=None):
    """The accept sidecar, aligned to the trace's target decode graphs.

    arcint writes one line a decode graph ("verify <ord> pos <p> rows <n>
    accepted <a>"), p the graph's first position. Every request restarts at
    position 0, so the same position recurs in several requests: the lines are
    walked in request order and checked against each graph's position AND row
    count -- never a global position set, never the ordinal alone. A prompt's
    checkpoint tail is a PROMPT record and has no line, so it cannot shift the
    walk; a line out of step fails loudly.

    A 1-token DECODE at the end of a PROMPT with no line is the prompt's last
    token (arcint runs it alone for its logits); it is relabeled PROMPT and
    left out of the scored steps. A process's last graph is never written (the
    cache writes a graph when the next graph's advance runs), so one trailing
    line is kept apart as the unwritten last graph instead of refusing the
    trace.

    Returns a list, one entry a graph: -1 where the graph is not a decode graph,
    else the accepted count. Raises on a doubled or out-of-step line, on a
    missing line, and on more than one unwritten trailing line.
    """
    if report is None:
        report = {}
    sidecars = list(path) if isinstance(path, (list, tuple)) else [path]
    texts = [open(q).read() for q in sidecars]
    ends, parsed, n = [], [], 0
    for text in texts:
        lines = [l.split() for l in text.splitlines() if l.strip()]
        parsed.extend(dict(zip(ln[0::2], ln[1::2])) for ln in lines)
        n += len(lines)
        ends.append(n)
    trace_of = [os.fspath(q)[:-7] if os.fspath(q).endswith(".accept") else os.fspath(q) for q in sidecars]
    # a graph's file lets a line at the end of ITS sidecar be told apart from a
    # line that belongs to a later graph in the same file (a relabeled prompt-last
    # decode). Without file annotations (a bare load_route) the old final-line
    # rule stands.
    per_file = any(g.get("file") is not None for g in graphs)
    def file_index(li):
        k = 0
        while k < len(ends) and li >= ends[k]:
            k += 1
        return k if k < len(ends) else -1
    def is_file_end(li):
        return li == ends[file_index(li)] - 1 if file_index(li) >= 0 else False
    def line_matches(d, g):
        return int(d["pos"]) == g["first_pos"] and int(d["rows"]) == g["n_tokens"]
    def graph_in_line_file(gi, li, d):
        k = file_index(li)
        if k < 0 or k >= len(trace_of):
            return False
        for j in range(gi, len(graphs)):
            if graphs[j].get("file") == trace_of[k] and line_matches(d, graphs[j]):
                return True
        return False
    acc = [-1] * len(graphs)
    report["relabeled"] = 0
    report["unwritten_last_graph"] = 0
    if any(g["first_pos"] >= 0 for g in graphs):
        if any("pos" not in d for d in parsed):
            raise SystemExit("accept sidecar: a line without pos beside a version-2 trace")
        cand = set(relabel_candidates if relabel_candidates is not None else prompt_last_candidates(graphs))
        li = 0
        for gi, g in enumerate(graphs):
            if g["kind"] not in (K_DECODE, K_VERIFY):
                continue
            d = parsed[li] if li < len(parsed) else None
            # one killed process's last line at the end of its own file: skip it,
            # count it, and retry the same graph with the next file's line
            while d is not None and per_file and is_file_end(li) and not line_matches(d, g) \
                  and not graph_in_line_file(gi + 1, li, d):
                report["unwritten_last_graph"] += 1
                li += 1
                d = parsed[li] if li < len(parsed) else None
            if d is not None and int(d["pos"]) == g["first_pos"] and int(d["rows"]) == g["n_tokens"]:
                acc[gi] = int(d["accepted"])
                li += 1
                continue
            if gi in cand:
                g["kind"] = K_PROMPT          # the prompt's last token, not a decode step
                g["relabeled"] = True
                report["relabeled"] += 1
                continue
            if d is None:
                raise SystemExit("accept sidecar: no line for the %s graph at position %d (rows %d)"
                                 % ("decode" if g["kind"] == K_DECODE else "verify",
                                    g["first_pos"], g["n_tokens"]))
            raise SystemExit("accept sidecar line %d: pos %s rows %s against the trace's position %d rows %d -- "
                             "lines and graphs are out of step (a multi-lane run?)"
                             % (li + 1, d["pos"], d["rows"], g["first_pos"], g["n_tokens"]))
        for li2 in range(li, len(parsed)):
            if is_file_end(li2) and int(parsed[li2]["rows"]) > 0 \
               and (not per_file or not graph_in_line_file(0, li2, parsed[li2])):
                report["unwritten_last_graph"] += 1   # this process's last graph, not in its trace
            else:
                raise SystemExit("accept sidecar: line %d has no decode graph and is not the last line of its "
                                 "file -- only one unwritten last graph a process is allowed" % (li2 + 1))
        return acc
    k = 0                 # a version-1 trace: by order, as the old format was written
    for gi, g in enumerate(graphs):
        if g["kind"] in (K_DECODE, K_VERIFY):
            if k >= len(lines):
                raise SystemExit("accept sidecar: %d lines for %d decode graphs" % (len(lines), k + 1))
            d = parsed[k]
            if int(d["rows"]) != g["n_tokens"]:
                raise SystemExit("accept sidecar line %d: rows %s against the trace's %d tokens -- "
                                 "lines and graphs are out of step (a multi-lane run?)" % (k + 1, d["rows"], g["n_tokens"]))
            acc[gi] = int(d["accepted"])
            k += 1
    if k != len(lines):
        raise SystemExit("accept sidecar: %d lines, %d decode graphs" % (len(lines), k))
    return acc


def write_route(path, n_layer, n_expert, n_used, caches, events, header=b"ERT1"):
    """Write a synthetic ERT1 trace (the tests' and a probe's writer).

    caches: one list a cache ({il, residents} a layer of it, as the engine
    writes one LAYERS record a card); events: [{kind, n_tokens, tokens, ids}|
    {kind: 5, changes}]. The record layout is the one src/
    llama-expert-route-trace.cpp writes; a round-trip test compiles that writer
    and compares both.
    """
    b = bytearray(header + struct.pack("<IIII", 2, n_layer, n_expert, n_used))
    for cache in caches:
        b.append(K_LAYERS); b += struct.pack("<I", len(cache))
        for c in cache:
            b += struct.pack("<ii", c["il"], len(c["residents"]))
            b += struct.pack("<%di" % len(c["residents"]), *c["residents"])
    for e in events:
        b.append(e["kind"])
        if e["kind"] == K_TABLES:
            b += struct.pack("<I", len(e["changes"]))
            for il, ex, r in e["changes"]:
                b += struct.pack("<iiB", il, ex, r)
        else:
            toks = e["tokens"]
            b += struct.pack("<i", int(e.get("first_pos", -1)))
            b += struct.pack("<I", len(toks))
            b += struct.pack("<%di" % len(toks), *toks)
            b += struct.pack("<I", len(e["ids"]))
            for il, ids in sorted(e["ids"].items()):
                b += struct.pack("<ii", il, len(ids))
                b += struct.pack("<%di" % len(ids), *ids)
    open(path, "wb").write(bytes(b))


def route_to_steps(tr, accept=None):
    """A trace's graphs as (layers, steps, extras) in the ECT1-shaped form.

    Graph records that cover disjoint layers (one a card under a layer split)
    and arrive back to back are merged into one ubatch. A step the layers do
    not all cover is skipped with a count (it cannot be scored). Prompt graphs
    are kept with their routing for the prompt prior; the steps the cache
    counts (decode and verify graphs) carry their accepted-draft counts.
    """
    if len(tr["caches"]) != 1:
        raise SystemExit("the trace has %d caches' layer records; a layer split's graphs are merged, "
                         "its initial residency is not -- refusing" % len(tr["caches"]))
    L = tr["caches"][0]
    layers = [dict(il=c["il"], n_expert=tr["n_expert"], n_slots=c["n_slots"],
                   slots=c["residents"]) for c in L]
    lay = {l["il"]: i for i, l in enumerate(layers)}
    graphs, merged = [], []
    drafts = []
    for ev in tr["events"]:
        if ev["kind"] == K_DRAFT:
            # the MTP block routes its own experts for the drafted tokens; it
            # is not one of the cache's layers, so it never enters a step
            drafts.append(ev)
            continue
        if ev["kind"] in (K_PROMPT, K_DECODE, K_VERIFY):
            last = merged[-1] if merged else None
            if last is not None and last["kind"] == ev["kind"] and np.array_equal(last["tokens"], ev["tokens"]) \
               and last["first_pos"] == ev["first_pos"] \
               and not (set(last["ids"]) & set(ev["ids"])):
                last["ids"].update(ev["ids"]); last["events"] += [ev]; last["file"] = ev.get("file")
            else:
                merged.append(dict(kind=ev["kind"], first_pos=ev["first_pos"], n_tokens=ev["n_tokens"],
                                   tokens=ev["tokens"], ids=dict(ev["ids"]), events=[ev], file=ev.get("file")))
    for g in merged:
        if set(g["ids"]) != set(lay):
            raise SystemExit("a graph covers layers %s, the cache's layers are %s" % (sorted(g["ids"]), sorted(lay)))
        g["ids"] = {lay[il]: ids for il, ids in g["ids"].items()}
        graphs.append(g)
    report = {}
    if accept:
        acc = read_accept(accept, graphs, report=report)
    else:
        acc = None
        # No sidecar: a run without a verify graph is the non-spec loop, whose
        # prompt-last decode has no accept line and is relabeled. A trace WITH
        # verify graphs was speculative: its 1-token decodes are one-row
        # verifies, and without the sidecar there is no line to tell them apart
        # from the prompt-last call, so nothing is relabeled.
        if not any(g["kind"] == K_VERIFY for g in graphs):
            for gi in sorted(prompt_last_candidates(graphs)):
                graphs[gi]["kind"] = K_PROMPT
                graphs[gi]["relabeled"] = True
                report["relabeled"] = report.get("relabeled", 0) + 1
        report.setdefault("unwritten_last_graph", 0)
    tr["relabeled"] = report.get("relabeled", 0)
    tr["unwritten_last_graph"] = report.get("unwritten_last_graph", 0)
    steps = [g for g in graphs if g["kind"] in (K_DECODE, K_VERIFY)]
    step_acc = [acc[i] for i, g in enumerate(graphs) if g["kind"] in (K_DECODE, K_VERIFY)] if acc else None
    # a DRAFT record pairs with the verify whose rows it fed: its first_pos is
    # the verify's first_pos minus 1 (the MTP context starts at the accepted
    # target token), its n_tokens is the verify's rows minus one or equal. The
    # request id puts its fit on the training requests only; a DRAFT with no
    # match stays unpaired and is never a feature.
    gid, _rid = request_of(graphs, steps)
    used = set()
    for d in drafts:
        gi = None
        for j, g in enumerate(graphs):
            if j in used or g["kind"] != K_VERIFY or not g["ids"]:
                continue
            if g["first_pos"] == d["first_pos"] + 1 and g["n_tokens"] >= d["n_tokens"]:
                gi = j
                break
        d["graph"] = gi
        d["req"] = gid[gi] if gi is not None else None
        if gi is not None:
            used.add(gi)
    tr["drafts"] = drafts
    return layers, graphs, steps, step_acc


def step_stream(layers, steps):
    """ERT1 graphs in the (tokens, ids a layer) form simulate() walks."""
    return [(g["n_tokens"], [g["ids"][i] for i in range(len(layers))]) for g in steps]


def _use_matrix(layers, graphs, n_used):
    """(S, nl, ne) routed counts a graph; the top-k's rank order is lost here
    (the cache counts usage, not rank)."""
    nl, ne = len(layers), layers[0]["n_expert"]
    use = np.zeros((len(graphs), nl, ne), np.int64)
    for si, g in enumerate(graphs):
        for l, x in enumerate(g["ids"].values()):
            np.add.at(use[si, l], x[(x >= 0) & (x < ne)], 1)
    return use


def _future_from_ids(layers, ids, weight=1.0):
    nl, ne = len(layers), layers[0]["n_expert"]
    out = np.zeros((nl, ne))
    for l, x in enumerate(ids):
        np.add.at(out[l], x[(x >= 0) & (x < ne)], weight)
    return out


def normalize(fut):
    """Per layer to sum 1 (no layer's span of a plan outweighs another's)."""
    s = fut.sum(1, keepdims=True)
    return fut / np.where(s > 0, s, 1.0)


def blend_plan(res, usage, pred, weight=0.0, threshold=0.0, margin=1.5,
               min_usage=2.0, decay=1.0, max_swaps=None):
    """The blended plan rule: the engine's usage ranking is the BASE, a
    predictor only adds a bonus for experts whose predicted per-layer share
    clears `threshold`, and a swap happens only while the blended gain beats
    the resident's by `margin` (the engine's 1.5 gain). A weak guess clears no
    threshold, so it adds no bonus and cannot evict a hot resident; weight 0
    replays the usage baseline's plan exactly.

    The bonus is `weight * share * max(usage)` a layer: a confident predicted
    expert is worth at most that weight times the layer's hottest resident, in
    the same units as usage, so the base ranking keeps its scale. Returns
    [(gain, layer, candidate, layer, resident)] sorted by gain and capped."""
    nl, ne = res.shape
    share = normalize(pred)
    scale = usage.max(1) if usage.size else np.zeros(nl)
    scale = np.where(scale > 0, scale, 1.0)
    bonus = weight * share * scale[:, None] * (share >= threshold)
    score = usage + bonus
    plan = []
    for l in range(nl):
        cand = np.where(~res[l] & ((usage[l] >= min_usage) | (bonus[l] > 0)))[0]
        cold = np.where(res[l])[0]
        cand = cand[np.argsort(-score[l, cand], kind="stable")]
        cold = cold[np.argsort(score[l, cold], kind="stable")]
        for j in range(min(len(cand), len(cold))):
            gain = score[l, cand[j]] - score[l, cold[j]]
            if gain <= margin: break
            plan.append((gain, l, cand[j], l, cold[j]))
    plan.sort(key=lambda t: -t[0])
    if max_swaps is not None:
        plan = plan[:max_swaps]
    return plan


def _token_rows(layers, g):
    """A graph's (token id, per-layer routed ids) rows, token-major."""
    nl = len(layers)
    k = len(next(iter(g["ids"].values()))) // g["n_tokens"] if g["n_tokens"] else 0
    out = []
    for t in range(g["n_tokens"]):
        tok = int(g["tokens"][t]) if t < len(g["tokens"]) else -1
        out.append((tok, [g["ids"][l][t * k:(t + 1) * k] for l in range(nl)]))
    return out


def token_expert_table(layers, graphs, train):
    """Per layer, a token -> routed-expert count table, fitted on the training
    graphs only. This is the foresight source the engine has BEFORE a verify:
    the draft token ids are in the verify graph's `tokens[1..]`, known to the
    cache from the draft pass, and the table says which experts such a token
    routed to in the requests the predictor trained on. Nothing here reads a
    scored request's rows."""
    nl, ne = len(layers), layers[0]["n_expert"]
    table = {}
    keep = {id(g) for g in train}
    for g in graphs:
        if id(g) not in keep:
            continue
        for tok, ids in _token_rows(layers, g):
            if tok < 0:
                continue
            if tok not in table:
                table[tok] = np.zeros((nl, ne), np.int64)
            for l, x in enumerate(ids):
                np.add.at(table[tok][l], x[(x >= 0) & (x < ne)], 1)
    return table


def request_of(graphs, steps):
    """(the request id of every graph, of every step). A prompt that follows a
    decode starts the next request (a conversation's next turn); a prompt's
    checkpoint tail stays in its request."""
    req, seen = 0, False
    gid = []
    step_id = {id(g) for g in steps}
    rid = []
    for g in graphs:
        if g["kind"] == K_PROMPT:
            if seen:
                req += 1
            seen = False
            gid.append(req)
        else:
            gid.append(req)
            if id(g) in step_id:
                rid.append(req)
                seen = True
    return gid, rid


def step_prompt_prior(layers, graphs, gid, rid):
    """Every step's request prompt routing: the prompt graphs of THAT request,
    summed; no other request's prompt leaks into it."""
    nl, ne = len(layers), layers[0]["n_expert"]
    per = {}
    for gi, g in enumerate(graphs):
        if g["kind"] != K_PROMPT:
            continue
        if gid[gi] not in per:
            per[gid[gi]] = np.zeros((nl, ne))
        per[gid[gi]] += _future_from_ids(layers, [g["ids"][l] for l in range(nl)])
    zero = np.zeros((nl, ne))
    return [per.get(r, zero).copy() for r in rid]


def step_foresight(table, layers, steps):
    """Every step's foresight: the draft token ids of its verify graph mapped
    through the training table. A verify's `tokens[1..]` are the drafts, known
    before the verify; `tokens[0]` is the token before them and is not used."""
    nl, ne = len(layers), layers[0]["n_expert"]
    out = []
    for g in steps:
        f = np.zeros((nl, ne))
        if g["kind"] == K_VERIFY:
            for t in range(1, g["n_tokens"]):
                tok = int(g["tokens"][t]) if t < len(g["tokens"]) else -1
                if tok in table:
                    f += table[tok]
        out.append(f)
    return out


def mtp_expert_matrix(layers, drafts, graphs, gid, train_reqs):
    """Per target layer, a matrix MTP expert -> target expert, fitted on the
    TRAINING requests only. The MTP block's own routing of the input token at a
    draft position predicts the NEXT target row's routing (the drafted token),
    so a row pairs `draft input t` with the verify's target ids at row t+1.
    Normalized per target layer. DRAFT tokens are input tokens, not drafted
    tokens (the record writes step 0's accepted target token, then the previous
    draft), and their ids do not map by identity; that was the round-6 defect.
    Empty when no DRAFT record of a training request is in the trace: no
    serving result then tests the mechanism yet."""
    nl, ne = len(layers), layers[0]["n_expert"]
    mat = np.zeros((nl, ne, ne), np.int64)
    for d in drafts:
        if d.get("req") not in train_reqs:
            continue
        gi = d.get("graph")
        if gi is None:
            continue
        g = graphs[gi]
        if g["kind"] != K_VERIFY or not g["ids"]:
            continue
        kg = len(next(iter(g["ids"].values()))) // g["n_tokens"]
        if kg == 0 or not d["ids"]:
            continue
        km = len(next(iter(d["ids"].values()))) // d["n_tokens"] if d["n_tokens"] else 0
        if km == 0:
            continue
        n = min(d["n_tokens"], g["n_tokens"] - 1)
        for t in range(n):
            m = np.concatenate([x[t * km:(t + 1) * km] for x in d["ids"].values()])
            for l in range(nl):
                tgt = g["ids"][l][(t + 1) * kg:(t + 2) * kg]
                for em in m[(m >= 0) & (m < ne)]:
                    for te in tgt[(tgt >= 0) & (tgt < ne)]:
                        mat[l, em, te] += 1
    norm = np.zeros((nl, ne, ne))
    for l in range(nl):
        row = mat[l].sum(axis=1, keepdims=True)
        norm[l] = mat[l] / np.where(row > 0, row, 1)
    return norm


def step_mtp_foresight(matrix, layers, steps, by_graph):
    """Every step's MTP foresight: the matrix applied to the MTP block's routed
    ids at each draft input position, summed over positions and layers. Zero
    when a verify has no matching DRAFT record."""
    nl, ne = len(layers), layers[0]["n_expert"]
    out = []
    for g in steps:
        f = np.zeros((nl, ne))
        d = by_graph.get(id(g))
        if g["kind"] == K_VERIFY and d is not None and d["ids"]:
            km = len(next(iter(d["ids"].values()))) // d["n_tokens"] if d["n_tokens"] else 0
            n = min(d["n_tokens"], g["n_tokens"] - 1)
            for t in range(n):
                m = np.concatenate([x[t * km:(t + 1) * km] for x in d["ids"].values()])
                v = np.zeros(ne)
                np.add.at(v, m[(m >= 0) & (m < ne)], 1)
                for l in range(nl):
                    f[l] += v @ matrix[l]
        out.append(f)
    return out


def providers(layers, graphs, steps, acc, horizon=24, delay=2, drafts=None):
    """The predictor plans, each a function of the plan's step index, in step
    space (never a graph index).

    `foresight` maps a verify's draft token ids (known before the verify) through
    a token->expert table fitted on the even requests; `prompt` is the request's
    own prompt routing; `both` the two normalized and summed; `oracle` the next
    `horizon` tokens' actual routing, from the landing on.
    """
    nl, ne = len(layers), layers[0]["n_expert"]
    use_step = _use_matrix(layers, steps, None)
    gid, rid = request_of(graphs, steps)
    train = [g for gi, g in enumerate(graphs) if gid[gi] % 2 == 0]
    train_reqs = {gid[gi] for gi, g in enumerate(graphs) if gid[gi] % 2 == 0}
    table = token_expert_table(layers, graphs, train)
    prior_step = step_prompt_prior(layers, graphs, gid, rid)
    foresight_step = step_foresight(table, layers, steps)
    mtp = None
    if drafts:
        by_graph = {id(graphs[d["graph"]]): d for d in drafts if d.get("graph") is not None}
        mtp = step_mtp_foresight(mtp_expert_matrix(layers, drafts, graphs, gid, train_reqs),
                                 layers, steps, by_graph)

    zero = np.zeros((nl, ne))

    def crosses(si):
        # a plan whose landing is in another request is not foresight: the next
        # request has not been launched and is not this request's future
        return si + delay >= len(steps) or rid[si + delay] != rid[si]

    def oracle(si):
        # the next `horizon` tokens from the landing on, token by token, within
        # the request (a new request's routing is not known to a decode plan)
        left = horizon
        out = np.zeros((nl, ne))
        j = si + delay
        while left > 0 and j < len(steps) and rid[j] == rid[si]:
            g = steps[j]
            k = len(next(iter(g["ids"].values()))) // g["n_tokens"]
            take = min(g["n_tokens"], left)
            ids = [np.concatenate([x[t * k:(t + 1) * k] for t in range(take)])
                   for x in (g["ids"][l] for l in range(nl))]
            out += _future_from_ids(layers, ids)
            left -= take
            j += 1
        return out

    def foresight(si):
        return zero if crosses(si) else normalize(foresight_step[si])

    def prompt(si):
        return zero if crosses(si) else normalize(prior_step[si])

    def both(si):
        return zero if crosses(si) else normalize(foresight_step[si]) + normalize(prior_step[si])

    out = dict(oracle=oracle, foresight=foresight, prompt=prompt, both=both)
    if mtp is not None:
        out["mtp"] = lambda si: zero if crosses(si) else normalize(mtp[si])
        out["mtp_both"] = lambda si: zero if crosses(si) else normalize(mtp[si]) + normalize(foresight_step[si])
    return out, use_step, prior_step, foresight_step, rid


class LearnedMap:
    """Stage 1b: per layer, a linear map from routing to the next window's
    expert occurrence, fitted on some requests and scored on others.

    Features, per layer and PLAN STEP (never a graph index): the routed counts
    of the last 1, 2, 6 and 12 steps of the SAME request (each normalized to sum
    1 a layer), the request's prompt prior (normalized) and the step's foresight
    (the draft ids through the training table, normalized). Target: the expert
    counts of the next `horizon` tokens from the landing (`si + delay`, as the
    oracle), cut at the request's end and normalized. One 6-coefficient map a
    layer, least squares over all (step, expert) pairs of the training steps.
    Even requests train, odd requests score; the feature arrays are built per
    request, so nothing from a scored request enters a training row."""

    F = 6

    def __init__(self, horizon, delay=2):
        self.horizon = horizon
        self.delay = delay
        self.w = None

    @staticmethod
    def _start(rid, si):
        j = si
        while j > 0 and rid[j - 1] == rid[si]:
            j -= 1
        return j

    def feats(self, use, prior, foresight, rid, si):
        start = self._start(rid, si)
        f = []
        for span in (1, 2, 6, 12):
            lo = max(start, si - span + 1)
            f.append(normalize(use[lo:si + 1].sum(0).astype(float)))
        f.append(normalize(prior[si]))
        f.append(normalize(foresight[si]))
        return np.stack(f, 0)          # (6, nl, ne)

    def target(self, use, rid, si):
        left = self.horizon
        out = np.zeros(use.shape[1:], float)
        j = si + self.delay
        while left > 0 and j < use.shape[0] and rid[j] == rid[si]:
            nt = max(int(use[j].sum()), 1)
            out += use[j] * min(use[j].sum(), left) / nt
            left -= nt
            j += 1
        return normalize(out)

    def fit(self, use, prior, foresight, rid, train_steps):
        nl = use.shape[1]
        rows = {l: ([], []) for l in range(nl)}
        for si in train_steps:
            X = self.feats(use, prior, foresight, rid, si)
            y = self.target(use, rid, si)
            for l in range(nl):
                rows[l][0].append(X[:, l, :].T)      # (ne, F)
                rows[l][1].append(y[l])              # (ne,)
        w = np.zeros((nl, self.F))
        for l in range(nl):
            A = np.concatenate(rows[l][0])
            b = np.concatenate(rows[l][1])
            if len(b):
                w[l], *_ = np.linalg.lstsq(A, b, rcond=None)
        self.w = w
        self.arrays = (use, prior, foresight, rid)
        return True

    def predict(self, si):
        use, prior, foresight, rid = self.arrays
        X = self.feats(use, prior, foresight, rid, si)        # (6, nl, ne)
        return normalize(np.einsum("lf,fle->le", self.w, X))


def simulate(layers, steps, policy, adapt_every=6, max_swaps=96, budget=4, delay=2,
             min_usage=2.0, min_gain=1.5, decay=0.7, exact=False, provider=None,
             blend_weight=0.0, blend_threshold=0.0, blend_margin=1.5):
    nl = len(layers); ne = layers[0]["n_expert"]
    res = np.zeros((nl, ne), bool)
    for i, L in enumerate(layers):
        res[i, L["slots"][L["slots"] >= 0]] = True
    usage = np.zeros((nl, ne)); cnt = np.zeros((nl, ne)); last = np.zeros((nl, ne))
    now = 0; since = 0; pending = []          # (land_step, [(layer, expert, resident)])
    hits = misses = 0; per_step = []; n_sw = 0
    age = float(nl) if exact else 1.0
    for si, (nt, ids) in enumerate(steps):
        keep = []
        for land, flips in pending:
            if land <= si:
                for l, e, r in flips: res[l, e] = r
            else:
                keep.append((land, flips))
        pending = keep
        now += 1
        group = []
        h_step = m_step = 0
        for l, x in enumerate(ids):
            x = x[(x >= 0) & (x < ne)]
            h = int(res[l, x].sum()); h_step += h; m_step += len(x) - h
            np.add.at(usage[l], x, 1.0)
            g = np.unique(x); group.append(g); cnt[l, g] += 1; last[l, g] = now
        hits += h_step; misses += m_step; per_step.append((h_step, m_step))
        since += nt
        if adapt_every <= 0 or since < adapt_every:   # 0: never adapt (LLAMA_EXPERT_CACHE_ADAPT_EVERY=0)
            continue
        since = 0
        if pending:                            # no plan while swaps are in flight; the engine still decays
            if policy == "strata": usage *= decay
            continue
        plan = []                              # (gain, (layer_in, e_in), (layer_out, e_out))
        if policy == "plan":
            fut = provider(si)                     # the planned window: (nl, ne) counts
            for l in range(nl):
                c = np.where(~res[l] & (fut[l] > 0))[0]
                v = np.where(res[l])[0]
                c = c[np.argsort(-fut[l, c], kind="stable")]; v = v[np.argsort(fut[l, v], kind="stable")]
                for j in range(min(len(c), len(v))):
                    if fut[l, c[j]] <= fut[l, v[j]]: break
                    plan.append((fut[l, c[j]] - fut[l, v[j]], (l, c[j]), (l, v[j])))
        elif policy == "blend":
            plan = blend_plan(res, usage, provider(si), weight=blend_weight,
                              threshold=blend_threshold, margin=blend_margin,
                              min_usage=min_usage, decay=decay, max_swaps=max_swaps)
            usage *= decay
        elif policy == "strata":
            for l in range(nl):
                c = np.where(~res[l] & (usage[l] >= min_usage))[0]
                v = np.where(res[l])[0]
                c = c[np.argsort(-usage[l, c], kind="stable")]; v = v[np.argsort(usage[l, v], kind="stable")]
                for j in range(min(len(c), len(v))):
                    if usage[l, c[j]] < usage[l, v[j]] + min_gain: break
                    plan.append((usage[l, c[j]] - usage[l, v[j]], (l, c[j]), (l, v[j])))
            usage *= decay
        else:
            score = cnt / (age * (now - last) + 1.0)
            prot = np.zeros((nl, ne), bool)
            for l in range(nl): prot[l, group[l]] = True
            if exact:
                cand = np.zeros((nl, ne), bool)
                for l in range(nl): cand[l, group[l]] = True
                cand &= ~res
            else:
                cand = ~res & (cnt > 0)
            if policy == "lfru":
                for l in range(nl):
                    c = np.where(cand[l])[0]; v = np.where(res[l] & ~prot[l])[0]
                    nb = min(budget, len(c), len(v))
                    c = c[np.argsort(-score[l, c], kind="stable")][:nb]; v = v[np.argsort(score[l, v], kind="stable")][:nb]
                    for j in range(nb):
                        if not score[l, c[j]] > score[l, v[j]]: break
                        plan.append((score[l, c[j]] - score[l, v[j]], (l, c[j]), (l, v[j])))
            else:  # glfru
                cl, ce = np.where(cand); vl, ve = np.where(res & ~prot)
                nb = min(budget * nl, len(cl), len(vl))
                ci = np.argsort(-score[cl, ce], kind="stable")[:nb]; vi = np.argsort(score[vl, ve], kind="stable")[:nb]
                for j in range(nb):
                    a = (cl[ci[j]], ce[ci[j]]); b = (vl[vi[j]], ve[vi[j]])
                    if not score[a] > score[b]: break
                    plan.append((score[a] - score[b], a, b))
        if policy == "blend":
            plan = [(g, (li, ei), (lo, eo)) for g, li, ei, lo, eo in plan]
        plan.sort(key=lambda t: -t[0]); plan = plan[:max_swaps]
        n_sw += len(plan)
        if plan:
            flips = []
            for _, (li, ei), (lo, eo) in plan:
                flips.append((lo, eo, False)); flips.append((li, ei, True))
            pending.append((si + delay, flips))
    return hits / max(1, hits + misses), per_step, res.sum(1), n_sw


def bounds(layers, steps, delay=2, every=6, cap=96):
    nl = len(layers); ne = layers[0]["n_expert"]; S = len(steps)
    use = np.zeros((S, nl, ne), np.int64)
    for si, (_, ids) in enumerate(steps):
        for l, x in enumerate(ids):
            np.add.at(use[si, l], x[(x >= 0) & (x < ne)], 1)
    tot = use.sum()
    init = np.zeros((nl, ne), bool)
    for i, L in enumerate(layers): init[i, L["slots"][L["slots"] >= 0]] = True
    ns = init.sum(1)
    out = {"fill-only (the load-time profile)": (use * init).sum() / tot}
    best = np.zeros((nl, ne), bool); tu = use.sum(0)
    for l in range(nl): best[l, np.argsort(-tu[l], kind="stable")[:ns[l]]] = True
    out["best fixed set per layer, in hindsight"] = (use * best).sum() / tot
    cum = np.concatenate([np.zeros((1, nl, ne), np.int64), np.cumsum(use, 0)])
    for H in (6, 24, 96):                      # the mechanism's cadence, cap and landing; the next H steps known
        res = init.copy(); pending = []; since = 0; h = 0
        for si, (nt, _) in enumerate(steps):
            keep = []
            for land, flips in pending:
                if land <= si:
                    for l, e, r in flips: res[l, e] = r
                else: keep.append((land, flips))
            pending = keep
            h += (use[si] * res).sum(); since += nt
            if since < every: continue
            since = 0
            if pending: continue
            fut = cum[min(S, si + delay + H)] - cum[min(S, si + delay)]
            plan = []
            for l in range(nl):
                c = np.where(~res[l])[0]; v = np.where(res[l])[0]
                c = c[np.argsort(-fut[l, c], kind="stable")]; v = v[np.argsort(fut[l, v], kind="stable")]
                for j in range(min(len(c), len(v))):
                    if fut[l, c[j]] <= fut[l, v[j]]: break
                    plan.append((fut[l, c[j]] - fut[l, v[j]], l, c[j], v[j]))
            plan.sort(key=lambda t: -t[0])
            if plan[:cap]:
                flips = []
                for _, l, ei, eo in plan[:cap]: flips += [(l, eo, False), (l, ei, True)]
                pending.append((si + delay, flips))
        out["foresight planner in the mechanism, next %d steps" % H] = h / tot
    h = 0                                       # Belady's MIN per layer
    for l in range(nl):
        res = set(np.where(init[l])[0].tolist()); seq = [np.where(use[si, l] > 0)[0].tolist() for si in range(S)]
        nxt = {}; nexts = [None] * S
        for si in range(S - 1, -1, -1):
            nexts[si] = dict(nxt)
            for e in seq[si]: nxt[e] = si
        for si in range(S):
            row = seq[si]
            h += sum(int(use[si, l, e]) for e in row if e in res)
            for e in row:
                if e in res: continue
                if len(res) >= ns[l]:
                    v = max((r for r in res if r not in row), key=lambda r: nexts[si].get(r, S + 1), default=None)
                    if v is None: continue
                    res.discard(v)
                res.add(e)
    out["Belady MIN per layer"] = h / tot
    return out




def trace_residency(layers, tr):
    """The trace's own resident slot sets after its TABLES records, by layer
    index: enough to rebuild the run's residency instead of guessing it."""
    res = [np.zeros(L["n_expert"], bool) for L in layers]
    for i, c in enumerate(tr["caches"][0]):
        if i < len(res):
            res[i][c["residents"][c["residents"] >= 0]] = True
    lay = {l["il"]: i for i, l in enumerate(layers)}
    for ev in tr["events"]:
        if ev["kind"] == K_TABLES:
            for il, e, r in ev["changes"]:
                if il in lay and 0 <= e < layers[lay[il]]["n_expert"]:
                    res[lay[il]][e] = bool(r)
    return res


def fit_learned(a, use, prior, foresight, rid, horizon):
    train = [i for i, r in enumerate(rid) if r % 2 == 0]
    lm = LearnedMap(horizon, delay=a.delay)
    if not lm.fit(use, prior, foresight, rid, train):
        return None, train
    return lm, train


def evaluate(a, layers, steps, idx, prov, learned, horizon):
    """One predictor set over `idx` steps of the trace, in the engine's own
    mechanism. Returns (name, hits, swaps, slots a layer at the end) rows."""
    stream = step_stream(layers, [steps[i] for i in idx])
    wrap = lambda f: (lambda si: f(idx[si]))
    rows = [("strata (the baseline)", "strata", None),
            ("foresight: draft ids -> experts", "plan", wrap(prov["foresight"])),
            ("the prompt's routing as a prior", "plan", wrap(prov["prompt"])),
            ("both", "plan", wrap(prov["both"]))]
    if prov.get("mtp") is not None:
        rows.append(("foresight: MTP draft routing", "plan", wrap(prov["mtp"])))
        if prov.get("mtp_both") is not None:
            rows.append(("MTP routing + draft-id table", "plan", wrap(prov["mtp_both"])))
    if learned is not None:
        rows.append(("learned linear map", "plan", wrap(prov["learned"])))
    rows.append(("oracle, next %d tokens" % horizon, "plan", wrap(prov["oracle"])))
    out = []
    for name, pol, pf in rows:
        h, _, occ, sw = simulate(layers, stream, pol, delay=a.delay, max_swaps=a.cap,
                                 adapt_every=a.every, provider=pf)
        out.append((name, h, sw, occ))
    return out


def train_curve(a, layers, graphs, steps, tr, rid):
    """`--train-curve`: a FIXED held-out set (the odd requests; without one, the
    last half of the requests) and increasing training-set sizes from the
    remaining requests (1, 2, 4, ... requests, chronological). Per size, the
    token->expert table and the learned linear map are fitted on THAT subset
    only; each predictor is scored on the held-out steps in the round-4 blend
    rule, its (threshold, weight, margin) chosen on the training steps; the
    strata baseline is the same at every size. More training data must not lose
    held-out hits for a predictor that is exact."""
    nl = len(layers)
    reqs = sorted(set(rid))
    odd = [i for i, r in enumerate(rid) if r % 2 == 1]
    held_tag = "the odd requests"
    if not odd:
        last = reqs[len(reqs) // 2:] if len(reqs) > 1 else reqs
        held_tag = "the last %d requests" % len(last)
        held_set = set(last)
        odd = [i for i, r in enumerate(rid) if r in held_set]
    if not odd:
        print("train-curve: no held-out step (one request only); skipped")
        return
    train_reqs = [r for r in reqs if r not in {rid[i] for i in odd}]
    if not train_reqs:
        print("train-curve: no training request after the held-out set; skipped")
        return
    sizes, k = [], 1
    while k < len(train_reqs):
        sizes.append(k); k *= 2
    sizes.append(len(train_reqs))
    gid, _ = request_of(graphs, steps)
    use = _use_matrix(layers, steps, None)
    prior = step_prompt_prior(layers, graphs, gid, rid)
    held_stream = step_stream(layers, [steps[i] for i in odd])
    base_h = simulate(layers, held_stream, "strata", delay=a.delay, max_swaps=a.cap,
                      adapt_every=a.every)[0]
    thresholds = a.thresholds; weights = a.weights; margins = a.margins
    print("train-curve: %d held-out decode steps of %s, %d held-out requests, %d training requests; "
          "delay %d, %d swaps a plan every %d tokens, thresholds %s, weights %s, margins %s" % (
              len(odd), held_tag, len({rid[i] for i in odd}), len(train_reqs), a.delay, a.cap, a.every,
              list(thresholds), list(weights), list(margins)))
    print("  %-22s %8s %8s %22s %22s" % ("training requests", "steps", "baseline",
                                          "draft-id table, blended", "learned map, blended"))

    def tune(pred, train_idx):
        stream = step_stream(layers, [steps[i] for i in train_idx])
        wrap = lambda si: pred[train_idx[si]]
        best = None
        for thr in thresholds:
            for w in weights:
                for m in margins:
                    h = simulate(layers, stream, "blend", delay=a.delay, max_swaps=a.cap,
                                 adapt_every=a.every, provider=wrap,
                                 blend_weight=w, blend_threshold=thr, blend_margin=m)[0]
                    if best is None or h > best[0]:
                        best = (h, thr, w, m)
        return best[1:]

    for n in sizes:
        subset = set(train_reqs[:n])
        train_graphs = [g for gi, g in enumerate(graphs) if gid[gi] in subset]
        train_idx = [i for i, r in enumerate(rid) if r in subset]
        table = token_expert_table(layers, graphs, train_graphs)
        fs = step_foresight(table, layers, steps)
        thr_w = tune(fs, train_idx)
        h_f = simulate(layers, held_stream, "blend", delay=a.delay, max_swaps=a.cap,
                       adapt_every=a.every, provider=lambda si: fs[odd[si]],
                       blend_weight=thr_w[1], blend_threshold=thr_w[0], blend_margin=thr_w[2])[0]
        lm = LearnedMap(a.horizon, a.delay)
        lm.fit(use, prior, fs, rid, train_idx)
        thr_l = tune([lm.predict(i) for i in range(len(steps))], train_idx)
        h_l = simulate(layers, held_stream, "blend", delay=a.delay, max_swaps=a.cap,
                       adapt_every=a.every, provider=lambda si: lm.predict(odd[si]),
                       blend_weight=thr_l[1], blend_threshold=thr_l[0], blend_margin=thr_l[2])[0]
        print("  %-22d %8d %7.2f %% %12.2f %% (%.3f/%.2f/%.1f) %12.2f %% (%.3f/%.2f/%.1f)" % (
            n, len(train_idx), 100 * base_h, 100 * h_f, thr_w[0], thr_w[1], thr_w[2],
            100 * h_l, thr_l[0], thr_l[1], thr_l[2]))


def fitted_row(name):
    """Rows that read the token/expert tables fitted on the even requests: they
    are in-sample when they are scored on those same requests."""
    return name.startswith("foresight") or name == "both" or name.startswith("learned") \
        or name.startswith("MTP")


def report_label(name, held_out):
    return name if held_out else name + " (in-sample)"


def main_route(a):
    if getattr(a, "single", False) and os.path.isdir(a.trace):
        raise SystemExit("--single needs one trace file, not a directory: %s (use it on a .000.ert1 file)" % a.trace)
    paths = [a.trace] if getattr(a, "single", False) else expand_traces(a.trace)
    tr = load_traces(paths)
    acc_paths = None
    if a.accept is None:
        acc_paths = [p + ".accept" for p in paths if os.path.exists(p + ".accept")]
        if not acc_paths:
            acc_paths = None
    else:
        acc_paths = expand_accept(a.accept)
    layers, graphs, steps, acc = route_to_steps(tr, acc_paths)
    if getattr(a, "train_curve", False):
        _gid, rid = request_of(graphs, steps)
        train_curve(a, layers, graphs, steps, tr, rid)
        return
    tot = sum(L["n_slots"] for L in layers)
    print("ERT1 v%d: %d layers, %d experts, top-%d; %d cached layers, %d slots (%d-%d a layer); "
          "graphs %d (%d prompt, %d decode, %d verify), %d MTP draft records, %d decode steps, %d routed pairs%s" % (
              tr["version"], tr["n_layer"], tr["n_expert"], tr["n_used"], len(layers), tot,
              min(L["n_slots"] for L in layers), max(L["n_slots"] for L in layers),
              len(graphs), sum(g["kind"] == K_PROMPT for g in graphs),
              sum(g["kind"] == K_DECODE for g in graphs), sum(g["kind"] == K_VERIFY for g in graphs),
              len(tr.get("drafts", [])), len(steps),
              sum(sum(len(x) for x in g["ids"].values()) for g in steps),
              "; TORN tail" if tr["torn"] else ""))
    if tr.get("relabeled"):
        print("relabeled: %d prompt-last 1-token DECODE graph%s as PROMPT (no accept line; "
              "outside the scored steps)" % (tr["relabeled"], "" if tr["relabeled"] == 1 else "s"))
    if tr.get("unwritten_last_graph"):
        print("the trace ends one step short: the accept sidecar's last line is the process's "
              "unwritten last graph (kept apart, not scored)")
    if acc is None:
        print("no accept sidecar: the decode graphs are not aligned to their walks "
              "(the predictors do not need it; the oracle does not either)")
    gid_all, rid_all = request_of(graphs, steps)
    if getattr(a, "profiles", None):
        compare_profiles(a, layers, steps, rid_all, a.profiles, a.every, tr["n_layer"])
        return
    prov, use, prior, foresight, rid = providers(layers, graphs, steps, acc, horizon=a.horizon,
                                                 delay=a.delay, drafts=tr.get("drafts"))
    learned, train = fit_learned(a, use, prior, foresight, rid, a.horizon) if a.learned else (None, [])
    if a.learned:
        if learned is None:
            print("learned: too few training steps; skipped")
        else:
            print("learned: %d coefficients a layer, trained on %d of %d decode steps (the even requests), "
                  "scored on the odd ones" % (LearnedMap.F, len(train), len(steps)))
        if learned is not None:
            prov["learned"] = lambda si: learned.predict(si)

    # the run's own residency against the baseline's, a fidelity check of the
    # trace's TABLES records (a mismatch is a diagnostic run's mismatch, not a policy verdict)
    own = trace_residency(layers, tr)
    base = simulate(layers, step_stream(layers, steps), "strata", delay=a.delay, max_swaps=a.cap, adapt_every=a.every)[2]
    inter = sum(int(np.logical_and(own[i], base[i] > 0).sum()) for i in range(len(layers)))
    print("resident set: the trace's own tables %d slots, the baseline replay's %d, %d shared (%.1f %% of the trace's)" % (
        sum(int(r.sum()) for r in own), int(sum(base)), inter, 100.0 * inter / max(1, sum(int(r.sum()) for r in own))))

    # after the fidelity check (which compares the trace's own set with the
    # trace's own set): seed the slots from a profile, for the report below
    if getattr(a, "profile", None):
        ranked, _slots = read_profile(a.profile, tr["n_layer"], layers[0]["n_expert"])
        budget = sum(L["n_slots"] for L in layers)
        seeded = profile_slots(layers, ranked)
        layers = seeded
        print("seeded from %s: %d slots over %d layers (%d-%d a layer), the walk's distribution of the "
              "trace's %d-slot budget" % (
                  os.path.basename(a.profile), sum(L["n_slots"] for L in layers), len(layers),
                  min(L["n_slots"] for L in layers), max(L["n_slots"] for L in layers), budget))

    if getattr(a, "blend_sweep", False):
        odd = [i for i, r in enumerate(rid) if r % 2 == 1]
        idx = odd if odd else list(range(len(steps)))
        held = ("the odd requests (held out)" if odd
                else "all steps (no held-out request: fitted rows are in-sample)")
        thresholds = getattr(a, "thresholds", (0.005, 0.01, 0.02, 0.05))
        weights = getattr(a, "weights", (0.1, 0.3, 1.0))
        margins = getattr(a, "margins", (1.5,))
        horizons = getattr(a, "horizons", (24, 48, 96))
        tune = ("the even requests" if (odd and getattr(a, "blend_tune", "even") == "even")
                else "the scored steps")
        print("blend, best per predictor tuned on %s, %s, delay %d, %d swaps a plan every %d tokens: "
              "thresholds %s, weights %s, margin %s, H %s" % (
                  tune, held, a.delay, a.cap, a.every, list(thresholds), list(weights), list(margins), list(horizons)))
        base = evaluate(a, layers, steps, idx, prov, None, a.horizon)
        base_name, base_h, base_sw, _ = base[0]
        print("  baseline: %s %5.2f %% hits  %5d swaps" % (base_name, 100 * base_h, base_sw))
        rows = {}
        oracle_plain = {}
        stream = step_stream(layers, [steps[i] for i in idx])
        wrap = lambda f: (lambda si: f(idx[si]))
        # The configuration is chosen on the even requests and scored on the odd
        # ones (--blend-tune scored chooses on the scored steps instead, a
        # selection on held-out data); without odd requests both are all steps.
        train_idx = [i for i, r in enumerate(rid) if r % 2 == 0 and odd]
        tune_idx = train_idx if (odd and getattr(a, "blend_tune", "even") == "even") else idx
        tune_stream = step_stream(layers, [steps[i] for i in tune_idx])
        tune_wrap = lambda f: (lambda si: f(tune_idx[si]))

        cands = {}

        def sweep(name, pf, H):
            # choose the cell across H on the tuning steps only; the odd steps
            # are scored once, after every H's cells are known
            for thr in thresholds:
                for w in weights:
                    for m in margins:
                        h, _, _, sw = simulate(layers, tune_stream, "blend", delay=a.delay, max_swaps=a.cap,
                                               adapt_every=a.every, provider=tune_wrap(pf),
                                               blend_weight=w, blend_threshold=thr, blend_margin=m)
                        if name not in cands or h > cands[name][0]:
                            cands[name] = (h, sw, H, thr, w, m)

        def score(name, pf):
            _, _, H, thr, w, m = cands[name]
            h, _, _, sw = simulate(layers, stream, "blend", delay=a.delay, max_swaps=a.cap,
                                   adapt_every=a.every, provider=wrap(pf),
                                   blend_weight=w, blend_threshold=thr, blend_margin=m)
            rows[name] = (h, sw, H, thr, w, m)

        # foresight, prompt and both are fixed functions of the step (their
        # window is the draft ids or the request's prompt), so H does not move
        # them: one sweep, labelled with the first H.
        prov0, _, _, _, _ = providers(layers, graphs, steps, acc, horizon=horizons[0], delay=a.delay, drafts=tr.get("drafts"))
        tuned = {}
        names0 = ["foresight", "prompt", "both"] + (["mtp", "mtp_both"] if prov0.get("mtp") is not None else [])
        for name in names0:
            if prov0.get(name) is not None:
                sweep(name, prov0[name], horizons[0])
                tuned[name] = prov0[name]
        for H in horizons:
            prov_h, use_h, prior_h, foresight_h, rid_h = providers(layers, graphs, steps, acc, horizon=H,
                                                                  delay=a.delay, drafts=tr.get("drafts"))
            lm_h, _ = fit_learned(a, use_h, prior_h, foresight_h, rid_h, H) if a.learned else (None, [])
            if lm_h is not None:
                prov_h["learned"] = lambda si, lm=lm_h: lm.predict(si)
            oracle_plain[H] = evaluate(a, layers, steps, idx, prov_h, None, H)[-1]
            for name in ("learned", "oracle"):
                pf = prov_h.get(name)
                if pf is not None:
                    sweep(name, pf, H)
                    if name not in tuned or H == cands[name][2]:
                        tuned[name] = pf
        for name in cands:
            score(name, tuned[name])
        best_oracle_unbl = max(oracle_plain.items(), key=lambda kv: kv[1][1])
        _oname, oh, osw, _occ = best_oracle_unbl[1]
        print("  oracle (the rule off), best H=%d: %5.2f %% hits  %5d swaps  %+6.2f points against the baseline"
              % (best_oracle_unbl[0], 100 * oh, osw, 100 * (oh - base_h)))
        for H in horizons:
            _n, hh, sww, _o = oracle_plain[H]
            print("    rule off at H=%d: %5.2f %% hits  %5d swaps  %+6.2f against the baseline"
                  % (H, 100 * hh, sww, 100 * (hh - base_h)))
        # the upper bound inside the blend rule is the blended oracle, the same
        # threshold, bonus and margin as every predictor; compare against it.
        ub = rows.get("oracle", (base_h, base_sw, horizons[0], 0.0, 0.0, 0.0))[0]
        print("  blended oracle (the upper bound inside the rule): %5.2f %% hits" % (100 * ub))
        print("  %-12s %-34s %8s %8s %14s %18s" % ("predictor", "best blend (H, thr, weight, margin)",
                                                    "hits", "swaps", "vs baseline", "vs blended oracle"))
        order = ("foresight", "prompt", "both", "mtp", "mtp_both", "learned", "oracle")
        for name in order:
            if name not in rows:
                continue
            h, sw, H, thr, w, m = rows[name]
            shown = report_label(name, bool(odd)) if fitted_row(name) else name
            label = "%s H=%d thr=%.3f w=%.2f m=%.1f" % (
                "blend" if name != "oracle" else "blend oracle", H, thr, w, m)
            print("  %-22s %-34s %7.2f %% %8d %+13.2f %+18.2f" % (
                shown, label, 100 * h, sw, 100 * (h - base_h), 100 * (h - ub)))
        return

    if a.sweep:
        odd = [i for i, r in enumerate(rid) if r % 2 == 1]
        idx = odd if odd else list(range(len(steps)))
        tag = ("the odd requests (held out)" if odd
               else "all steps (no held-out request: fitted rows are in-sample)")
        print("horizon sweep, %s, delay %d, %d swaps a plan every %d tokens:" % (tag, a.delay, a.cap, a.every))
        names = None
        table = {}
        for H in (6, 12, 24, 48):
            prov_h, use_h, prior_h, foresight_h, rid_h = providers(layers, graphs, steps, acc, horizon=H, delay=a.delay, drafts=tr.get("drafts"))
            lm_h, _ = fit_learned(a, use_h, prior_h, foresight_h, rid_h, H) if a.learned else (None, [])
            if lm_h is not None:
                prov_h["learned"] = lambda si, lm=lm_h: lm.predict(si)
            rows = evaluate(a, layers, steps, idx, prov_h, lm_h, H)
            names = [r[0] for r in rows]
            table[H] = [r[1] for r in rows]
        names = ["oracle, next H tokens" if n.startswith("oracle") else n for n in names]
        print("  %-38s %s" % ("predictor / oracle", "  ".join("%6d" % H for H in table)))
        for k, name in enumerate(names):
            shown = report_label(name, bool(odd))
            print("  %-38s %s" % (shown, "  ".join("%5.1f%%" % (100 * table[H][k]) for H in table)))
        return

    odd = [i for i, r in enumerate(rid) if r % 2 == 1]
    # The token table is fitted on the even requests. With an odd request it is
    # scored out of sample, and that is the report; without one every fitted row
    # is in-sample and is marked as such rather than read as foresight.
    sections = ([(odd, "the odd requests (held out)", True)]
                if odd else
                [(list(range(len(steps))), "all %d decode steps (no odd request)" % len(steps), False)])
    for idx, tag, held_out in sections:
        print("-- %d decode steps of %s" % (len(idx), tag))
        use_learned = learned
        for name, h, sw, occ in evaluate(a, layers, steps, idx, prov, use_learned, a.horizon):
            shown = report_label(name, held_out) if fitted_row(name) else name
            print("  %-38s %5.2f %% hits  %5d swaps  slots a layer at the end %d-%d"
                  % (shown, 100 * h, sw, occ.min(), occ.max()))
    if a.bounds and acc is not None:
        for k, v in bounds(layers, step_stream(layers, steps), delay=a.delay, every=a.every, cap=a.cap).items():
            print("  %-50s %.2f %%" % (k, 100 * v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--cap", type=int, default=96, help="swaps a plan (the engine's LLAMA_EXPERT_CACHE_MAX_SWAPS)")
    ap.add_argument("--every", type=int, default=6, help="decode tokens a plan (LLAMA_EXPERT_CACHE_ADAPT_EVERY)")
    ap.add_argument("--bounds", action="store_true")
    ap.add_argument("--accept", default=None, help="the trace's accept sidecar (default <trace>.accept when it exists)")
    ap.add_argument("--train-curve", action="store_true", dest="train_curve",
                    help="a fixed held-out set and increasing training-set sizes: the token table and the "
                         "learned map fitted per size, blended (round 4), scored held out, against the baseline")
    ap.add_argument("--single", action="store_true",
                    help="replay exactly this one trace file (no sibling rotations), so a rotated file can be "
                         "shown to replay alone")
    ap.add_argument("--profile", default=None,
                    help="seed the cache's slots from this STRP profile instead of the trace's own residents "
                         "(the budget stays the trace's per-layer slot count)")
    ap.add_argument("--profiles", nargs="+", default=None,
                    help="compare these STRP profiles' held-out hit rates at one slot budget, adapt on and off")
    ap.add_argument("--learned", action="store_true", help="stage 1b: fit and score the linear map (even/odd requests)")
    ap.add_argument("--horizon", type=int, default=24, help="the oracle's and the learned map's window, in tokens")
    ap.add_argument("--blend-sweep", action="store_true", dest="blend_sweep",
                    help="the round-4 blend: sweep the confidence threshold, the bonus weight, the gain "
                         "margin and H, best per predictor, held out")
    ap.add_argument("--blend-tune", choices=("even", "scored"), default="even",
                    help="choose the blend's configuration on the even requests (default) or on the "
                         "scored steps (a selection on held-out data)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=(0.005, 0.01, 0.02, 0.05))
    ap.add_argument("--weights", type=float, nargs="+", default=(0.1, 0.3, 1.0))
    ap.add_argument("--margins", type=float, nargs="+", default=(1.5,))
    ap.add_argument("--horizons", type=int, nargs="+", default=(24, 48, 96))
    ap.add_argument("--sweep", action="store_true",
                    help="the horizon sweep: every predictor and the oracle at 6, 12, 24 and 48 tokens")
    a = ap.parse_args()
    if os.path.isdir(a.trace) or open(a.trace, "rb").read(4) == b"ERT1":
        main_route(a)
        raise SystemExit(0)
    layers, steps = load(a.trace)
    tot = sum(L["n_slots"] for L in layers)
    print("layers %d, slots %d (%d-%d a layer), expert bytes %s, steps %d (decode %d), routed %d" % (
        len(layers), tot, min(L["n_slots"] for L in layers), max(L["n_slots"] for L in layers),
        sorted(set(L["eb"] for L in layers)), len(steps), sum(1 for s in steps if s[0] <= 8),
        sum(sum(len(x) for x in s[1]) for s in steps)))
    for pol, ex in (("strata", False), ("lfru", False), ("lfru", True), ("glfru", False), ("glfru", True)):
        h, per, occ, sw = simulate(layers, steps, pol, delay=a.delay, budget=a.budget, exact=ex,
                                   max_swaps=a.cap, adapt_every=a.every)
        print("%-6s%-6s %.2f %%  slots a layer at the end %d-%d" % (pol, " exact" if ex else "", 100 * h, occ.min(), occ.max()))
    if a.bounds:
        for k, v in bounds(layers, steps, delay=a.delay, every=a.every, cap=a.cap).items():
            print("%-50s %.2f %%" % (k, 100 * v))
