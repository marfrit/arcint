#!/usr/bin/env python3
"""Build a per-card expert profile (STRP) from our own ERT1 routing traces.

The format is the one arcint's `--llama-expert-profile` reads (`read_profile`,
src/llama-expert-cache.cpp of the llama tree) and Strata's writer emits
(tools/make_profile.py): magic "STRP", u32 version 1, n_layer, n_expert, the
slots a card holds (a hint, the cache ignores it), n_ranked, then n_ranked
pairs of (u16 layer, u16 expert). The ranking rule is Strata's: the pairs a
trace routed, most frequent first, ties by (layer, expert); the pairs still
missing, interleaved across the layers (`expert` outer, `layer` inner). A base
profile, if given, is appended in its own order, never reordered (Strata's
make_profile.rank_profile; `--reorder` puts the traces first, `--no-base`
drops the base).

What a trace contributes, exactly: every routed pair of every PROMPT, DECODE
and VERIFY graph, once per position, prompt and decode at the SAME weight --
Strata's make_profile.read_trace counts every record's k ids alike and does not
distinguish a prompt position from a decode position, so neither do we. An MTP
DRAFT record is never counted: it is a different layer's (the MTP block's) own
expert ids, not the target model's. A request restarts at position 0; request
ids are the replay's (`request_of`: a graph's first position going backwards
starts a request).

How several requests are mixed: Strata's make_profile_mix turns every trace
into a distribution (its counts / its total) and multiplies it by the trace's
weight, so a long trace cannot outvote a short one; the ranking is the
weighted sum. Its unit is a trace file, because a Strata trace file is one
prompt's one-shot run. An ERT1 file is one process's many requests, so here the
unit is a REQUEST (each request's pairs normalized to sum 1) by default --
`--raw` keeps make_profile's additive counts instead, and `--weights
FILE:W[,FILE:W]` sets a per-file weight (1 by default; Strata's B70 profile
used 1 for each prompt trace and 2 for the summary trace, weight 1:2).

Deterministic: the same inputs give the same bytes (no timestamps, a stable
sort on (-score, layer, expert), and the fill order fixed).

    python3 tools/expert_profile_from_trace.py TRACE... --out PATH
        [--split all|even|odd] [--requests 0,2,4] [--kinds prompt,decode,verify]
        [--raw | --per-file] [--weights FILE:W[,FILE:W]]
        [--base FILE [--reorder | --no-base]] [--n-expert 512] [--n-layer 48]

TRACE is a file, a `.000` file (its `.001` rotations follow), or a directory of
*.ert1 files; several arguments are one ordered history, as the replay reads it.
"""
import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expert_cache_replay as ecr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAGIC, VERSION = b"STRP", 1
DEFAULT_N_LAYER, DEFAULT_N_EXPERT = 48, 512

K_PROMPT, K_DECODE, K_VERIFY = ecr.K_PROMPT, ecr.K_DECODE, ecr.K_VERIFY


# --- the file format: the reader arcint's read_profile implements -----------
def read_profile(path, n_layer=DEFAULT_N_LAYER, n_expert=DEFAULT_N_EXPERT):
    """(ranked pairs, the header's slots hint). Refuses a wrong magic, a
    version not 1, or a geometry other than the model's -- as arcint does."""
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit("%s: not a Strata profile (magic %r)" % (path, blob[:4]))
    if len(blob) < 24:
        raise SystemExit("%s: %d bytes, shorter than a header" % (path, len(blob)))
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if ver != 1:
        raise SystemExit("%s: profile version %d, this reader reads version 1" % (path, ver))
    if (nl, ne) != (n_layer, n_expert):
        raise SystemExit("%s: %dx%d, not %dx%d" % (path, nl, ne, n_layer, n_expert))
    if len(blob) < 24 + 4 * n:
        raise SystemExit("%s: %d ranked pairs, %d bytes (truncated)" % (path, n, len(blob)))
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)], slots


def write_profile(path, ranked, n_layer=DEFAULT_N_LAYER, n_expert=DEFAULT_N_EXPERT, slots=None):
    """Strata's write_profile: the header, the ranked pairs, then the per-layer
    table (expert -> rank, -1). arcint's read_profile stops after the pairs and
    ignores the table; it is written so the file is byte for byte Strata's
    writer's output and Comparable across tools."""
    if slots is None:
        slots = len(ranked)
    out = bytearray(MAGIC + struct.pack("<5I", VERSION, n_layer, n_expert, slots, len(ranked)))
    for layer, e in ranked:
        out += struct.pack("<HH", layer, e)
    table = [[-1] * n_expert for _ in range(n_layer)]
    for rank, (layer, e) in enumerate(ranked):
        table[layer][e] = rank
    for layer in range(n_layer):
        out += struct.pack("<%di" % n_expert, *table[layer])
    Path(path).write_bytes(bytes(out))


# --- counting ---------------------------------------------------------------
def count_graphs(graphs, kinds, requests, req_of_graph):
    """{request: {(layer, expert): count}} over the chosen graph kinds and
    request ids. Prompt and decode positions count alike."""
    per_req = defaultdict(lambda: defaultdict(int))
    for gi, g in enumerate(graphs):
        if g["kind"] not in kinds:
            continue
        r = req_of_graph[gi]
        if requests is not None and r not in requests:
            continue
        for l, ids in g["ids"].items():
            xs = [int(e) for e in ids]
            for e in xs:
                per_req[r][(l, e)] += 1
    return per_req


def score_counts(per_req, unit, weights):
    """The weighted score per pair. `unit` is "request" (Strata's mix with a
    request as the trace unit: each normalized by its total, then scaled),
    "file" (per trace file) or "raw" (make_profile's additive counts)."""
    score = defaultdict(float)
    for r, counts in per_req.items():
        total = float(sum(counts.values()))
        if total <= 0:
            continue
        w = weights[r]
        if unit == "raw":
            for p, c in counts.items():
                score[p] += w * c
        else:
            for p, c in counts.items():
                score[p] += w * c / total
    return score


def rank_profile(base_pairs, score, n_layer, n_expert, no_base=False, reorder=False):
    """Strata's rank_profile over a weighted score instead of a raw frequency:
    the score's pairs most frequent first (ties by (layer, expert)), the base's
    order (unless --no-base) then the fill. `--reorder` puts the score's pairs
    before the base, which is what a base that already ranks every pair needs."""
    ranked, seen, counts = [], set(), defaultdict(int)

    def take(pairs, key):
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)
                counts[key] += 1

    trace_pairs = [p for p, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
    if no_base or reorder:
        take(trace_pairs, "trace")
    if not no_base:
        take(base_pairs, "base")
    if not (no_base or reorder):
        take(trace_pairs, "trace")
    take(((layer, e) for e in range(n_expert) for layer in range(n_layer)), "fill")
    return ranked, counts


# --- command ----------------------------------------------------------------
def parse_weights(specs):
    """{path: weight} from FILE:W,FILE:W or repeated --weights."""
    out = {}
    for spec in specs or []:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            head, sep, tail = part.rpartition(":")
            try:
                path, w = head, float(tail) if sep else 1.0
            except ValueError:
                path, w = part, 1.0
            if not sep:
                raise SystemExit("--weights needs FILE:W, got %r" % part)
            out[str(Path(path))] = w
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="ERT1 trace files, a .000 file (rotations follow) or a directory")
    ap.add_argument("--out", required=True, help="the STRP file to write")
    ap.add_argument("--split", choices=("all", "even", "odd"), default="all",
                    help="the training requests: all (default), the even requests (the replay's fitted set) "
                         "or the odd ones (its held-out set)")
    ap.add_argument("--requests", default=None, help="an explicit request list, e.g. 0,2,4 (the replay's request ids)")
    ap.add_argument("--kinds", default="prompt,decode,verify",
                    help="the graph kinds counted (prompt, decode, verify; an MTP draft record never counts)")
    ap.add_argument("--raw", action="store_true", help="make_profile's additive counts instead of the mix")
    ap.add_argument("--per-file", action="store_true", dest="per_file",
                    help="normalize per trace FILE (Strata's trace unit) instead of per request")
    ap.add_argument("--weights", action="append", default=None, help="FILE:W[,FILE:W]: a per-file weight")
    ap.add_argument("--base", default=None, help="a profile whose order is kept first (never reordered)")
    ap.add_argument("--no-base", action="store_true", dest="no_base")
    ap.add_argument("--reorder", action="store_true", help="rank the trace's pairs before the base's")
    ap.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
    ap.add_argument("--n-expert", type=int, default=DEFAULT_N_EXPERT)
    a = ap.parse_args()

    if a.no_base and a.base:
        raise SystemExit("--base and --no-base are exclusive")
    kinds = {name.strip().lower() for name in a.kinds.split(",") if name.strip()}
    known = {"prompt": K_PROMPT, "decode": K_DECODE, "verify": K_VERIFY}
    unknown = kinds - set(known)
    if unknown:
        raise SystemExit("unknown graph kind(s) %s; known: %s (a draft record is not a target layer's routing)" % (
            sorted(unknown), sorted(known)))
    paths = ecr.expand_traces(a.traces[0]) + [p for t in a.traces[1:] for p in ecr.expand_traces(t)]
    tr = ecr.load_traces(paths)
    if (tr["n_layer"], tr["n_expert"]) != (a.n_layer, a.n_expert):
        raise SystemExit("the trace is %dx%d, the model asked for is %dx%d" % (
            tr["n_layer"], tr["n_expert"], a.n_layer, a.n_expert))
    layers, graphs, steps, _acc = ecr.route_to_steps(tr, None)
    gid, rid = ecr.request_of(graphs, steps)
    req_of_graph = gid

    requests = None
    if a.requests is not None:
        requests = {int(x) for x in a.requests.replace(" ", "").split(",") if x}
    elif a.split != "all":
        requests = {r for r in rid if r % 2 == (0 if a.split == "even" else 1)}
    wspec = parse_weights(a.weights)

    def weight_of(unit_key):
        return wspec.get(str(Path(unit_key)), wspec.get(unit_key, 1.0)) if wspec else 1.0

    weights = {r: weight_of(g.get("file", "")) for gi, g in enumerate(graphs) for r in [gid[gi]]}
    per_req = count_graphs(graphs, {known[k] for k in kinds}, requests, req_of_graph)
    if a.per_file:
        # regroup the requests by their file, so a file is the trace unit
        by_file = defaultdict(lambda: defaultdict(int))
        for gi, g in enumerate(graphs):
            if g["kind"] not in {known[k] for k in kinds}:
                continue
            r = gid[gi]
            if requests is not None and r not in requests:
                continue
            f = g.get("file")
            for l, ids in g["ids"].items():
                xs = [int(e) for e in ids]
                for e in xs:
                    by_file[f][(l, e)] += 1
        per_req = by_file
        weights = {f: weight_of(f) for f in by_file}
        unit = "request"
    else:
        unit = "raw" if a.raw else "request"
    score = score_counts(per_req, unit, weights)
    base_pairs = read_profile(a.base, a.n_layer, a.n_expert)[0] if a.base else []
    ranked, counts = rank_profile(base_pairs, score, a.n_layer, a.n_expert,
                                  no_base=a.no_base, reorder=a.reorder)
    write_profile(a.out, ranked, a.n_layer, a.n_expert)
    back, slots = read_profile(a.out, a.n_layer, a.n_expert)
    assert back == ranked, "the profile did not survive the round trip"
    n_req, n_pairs = len(per_req), sum(sum(c.values()) for c in per_req.values())
    print("wrote %s: %d ranked pairs (%d from the traces, %d from the base, %d filled in); "
          "%d routed pairs over %d %s, %d distinct pairs; unit: %s%s" % (
              a.out, len(ranked), counts["trace"], counts["base"], counts["fill"],
              n_pairs, n_req, "request" if n_req == 1 else "requests", len(score),
              "raw counts" if a.raw else ("a file normalized then weighted" if a.per_file
                                          else "a request normalized then weighted"),
              ", %d of %d requests left out" % (len(set(rid) - requests), len(set(rid))) if requests else ""))


if __name__ == "__main__":
    main()
