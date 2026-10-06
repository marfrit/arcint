#!/usr/bin/env python3
"""The expert-access census instrument for 0.5.2 VENICE.

Campaign: docs/campaigns/expert-hot-set-lru.md.
Design:   docs/design-expert-hot-set-lru.md (format v1, the canonical summary,
          hot-set selection, rounds-to-plateau, the patch-0013 join).

This tool is DEVICE-FREE and stdlib-only. It reads a per-token routed-expert
trace in format v1, derives the census summary, ranks experts per layer,
selects the hot set for a given per-layer slot budget, and reports
rounds-to-plateau. It is the instrument the served-path trace feeds and the
input the static-partition seed is built from.

Format v1 (what `tools/expert_lru_replay.py::load_trace` already parses):

    # arcint routing trace v1
    # artifact_sha256=<...> card=<PCI id|none> device=<ov device|cpu> depth=<n>
    # kv=<precision> f16=<on|off> chunk=<n> offload_ratio=<pct> tier=<...>
    # run=<run id> utc=<ISO-8601> tool=<producer>
    0 0 3 17 88 210 411 455 477 501
    0 1 5 9 44 121 200 300 388 490
    ...

`token_idx` and `layer_idx` are non-negative decimals; the rest of the line is
the routed expert ids (top_k of them). `#`-lines and blank lines are skipped.
A data row with fewer than three fields is MALFORMED and refused (the replay
loader silently drops such rows; this instrument must not).

The plugin's own dump (patch 0013) is a DIFFERENT schema and is parsed by
`read_plugin_csv`: `layer,weight_offset,expert,count`, ordered by
`weight_offset`, `layer` a 0-based rank of the weight-file offset, `#`-lines
skipped, trailer `# total,<T>`. It is a cross-check, joinable to a trace on
`weight_offset` given an exported decoder-index -> weight-offset map; it is
NOT a pure function of the trace.

Evidence classes: this file measures nothing. Its output is a deterministic
function of its input trace (format, counting, ranking) -- the trace carries
the measurement, this instrument carries the arithmetic.
"""
import argparse
import json
import sys
from collections import defaultdict, OrderedDict

# The trace's layer index is the decoder-layer position (0-based), matching
# `tools/expert_lru_replay.py`'s model. No geometry constants are assumed here
# beyond what a trace line carries, so a truncated or partial trace parses.


def read_provenance(path):
    """Return the `#`-header key=value pairs as a dict of strings.

    Lines are `# key=value` (one pair) or several whitespace-separated pairs
    on one line; every `key=value` token is captured. The v1 magic line
    (`# arcint routing trace v1`) has no `=`, so it is ignored. Free-text
    tokens without `=` are ignored. Later duplicate keys overwrite earlier
    ones.
    """
    prov = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if not ln.startswith("#"):
                break
            body = ln[1:].strip()
            if body.startswith("arcint routing trace"):
                continue
            # one or more `key=value` pairs, one per line or several per line
            # (the design header uses both spellings); a token without `=` is
            # free text and is ignored.
            for tok in body.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    prov[k.strip()] = v.strip()
    return prov


def parse_trace(path, strict=True):
    """(token, layer, [expert_id, ...]) rows in file order.

    `strict=True` refuses a data row with fewer than three fields (a malformed
    row), which the replay loader would silently drop.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, ln in enumerate(f, 1):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            if len(p) < 3:
                if strict:
                    raise ValueError(
                        f"{path}:{lineno}: malformed trace row {ln!r} "
                        f"(need at least 'token layer expert ...')")
                continue
            try:
                tok = int(p[0])
                lay = int(p[1])
                experts = [int(x) for x in p[2:]]
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: non-integer field in {ln!r}") from e
            if tok < 0 or lay < 0 or any(e < 0 for e in experts):
                raise ValueError(f"{path}:{lineno}: negative index in {ln!r}")
            rows.append((tok, lay, experts))
    return rows


def write_router_trace(path, by_layer, n_tokens, provenance=None):
    """Write a format-v1 routed-expert trace.

    `by_layer` maps decoder-layer index -> a sequence (list/array) of per-token
    selected-expert rows, each row a sequence of `top_k` ids. Rows are emitted
    token-major, ids sorted ascending per row, matching format v1's
    deterministic order. Stdlib-only, so the device-free reference-router
    writer (`ref_forward_stream.py --router-trace`) shares this exact code.
    """
    lines = ["# arcint routing trace v1"]
    for k, v in (provenance or {}).items():
        lines.append(f"# {k}={v}")
    layers = sorted(by_layer)
    for tok in range(n_tokens):
        for lay in layers:
            ids = by_layer[lay]
            if tok >= len(ids):
                continue
            row = sorted(int(x) for x in ids[tok])
            lines.append(" ".join([str(tok), str(lay)] + [str(x) for x in row]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def trace_shape(rows):
    toks = {r[0] for r in rows}
    lays = {r[1] for r in rows}
    acc = sum(len(r[2]) for r in rows)
    return {
        "rows": len(rows),
        "tokens": len(toks),
        "layers": len(lays),
        "accesses": acc,
        "token_min": min(toks) if toks else None,
        "token_max": max(toks) if toks else None,
    }


def canonical_summary(rows):
    """[(layer, expert, count)] sorted by (layer, expert).

    A pure function of `rows`: the census summary in the trace's own
    layer-index space (not patch 0013's weight-offset-rank space).
    """
    hist = defaultdict(int)
    for _tok, lay, experts in rows:
        for e in experts:
            hist[(lay, e)] += 1
    return [(lay, e, hist[(lay, e)]) for (lay, e) in sorted(hist)]


def write_summary(rows, out_path, provenance=None):
    lines = []
    if provenance:
        lines.append("# census summary (derived from a format-v1 routing trace)")
        for k, v in provenance.items():
            lines.append(f"# {k}={v}")
    lines.append("layer,expert,count")
    total = 0
    for lay, e, c in canonical_summary(rows):
        lines.append(f"{lay},{e},{c}")
        total += c
    lines.append(f"# total,{total}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return "\n".join(lines) + "\n"


def frequency_rank(rows, layer=None):
    """[(expert, count)] ranked by count DESC, ties broken by expert id ASC.

    With `layer` given, only that layer's rank. Deterministic by construction.
    """
    hist = defaultdict(int)
    for _tok, lay, experts in rows:
        if layer is not None and lay != layer:
            continue
        for e in experts:
            hist[e] += 1
    return sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))


def select_hot_set(rows, slots_per_layer):
    """{layer: [expert, ...]} -- the top-`slots_per_layer` per layer by rank.

    Layers with fewer distinct routed experts than the budget get all of them.
    The returned list is in rank order (count desc, id asc), so it is a stable
    seed order as well as a membership set.
    """
    if slots_per_layer < 0:
        raise ValueError("slots_per_layer must be >= 0")
    layers = sorted({lay for _t, lay, _e in rows})
    out = {}
    for lay in layers:
        rank = frequency_rank(rows, lay)
        out[lay] = [e for e, _c in rank[:slots_per_layer]]
    return out


def hot_coverage(rows, hot):
    """Fraction of routed accesses whose expert is in its layer's hot set."""
    hits = acc = 0
    for _tok, lay, experts in rows:
        hs = set(hot.get(lay, ()))
        for e in experts:
            acc += 1
            if e in hs:
                hits += 1
    return (hits / acc) if acc else 0.0


def _token_prefixes(tokens, max_prefixes=None):
    """Distinct token values in ascending order, then powers-of-two lengths
    over that list, always ending at the full length. Deterministic."""
    toks = sorted(set(tokens))
    n = len(toks)
    if n == 0:
        return []
    lengths = []
    k = 1
    while k < n:
        lengths.append(k)
        k *= 2
    lengths.append(n)
    if max_prefixes is not None:
        # keep the leading powers and always the final full length
        keep = sorted(set(lengths[: max_prefixes] + [n]))
        lengths = [x for x in keep if 0 < x <= n]
    return [toks[:L] for L in lengths]


def rounds_to_plateau(rows, slots_per_layer, stable_window=2, hit_eps=1e-3,
                      max_prefixes=None):
    """Report the first prefix length at which the hot set stops changing.

    Plateau is the first prefix length whose selected hot set per layer is
    unchanged for `stable_window` consecutive larger prefixes AND whose
    coverage change against the previous prefix is below `hit_eps`. Returns
    a dict with `rounds`, `plateau` (bool), `prefix_tokens`, and `series`
    (per-prefix: tokens, hot_size, coverage, changed).
    """
    prefixes = _token_prefixes([r[0] for r in rows], max_prefixes=max_prefixes)
    # `stable_window` counts stable PREFIX PAIRS: a value below 2 would let the
    # first prefix declare a plateau against a previous prefix that does not
    # exist, so it is refused rather than silently clamped.
    if stable_window < 2:
        raise ValueError(f"stable_window must be >= 2, got {stable_window}")
    if hit_eps < 0:
        raise ValueError(f"hit_eps must be >= 0, got {hit_eps}")
    series = []
    prev_hot = None
    prev_cov = None
    stable = 0
    plateau_at = None
    for toks in prefixes:
        tset = set(toks)
        sub = [r for r in rows if r[0] in tset]
        hot = select_hot_set(sub, slots_per_layer)
        cov = hot_coverage(sub, hot)
        changed = (prev_hot is not None and hot != prev_hot)
        if prev_hot is None:
            changed = False
        series.append({
            "tokens": len(toks),
            "hot_size": sum(len(v) for v in hot.values()),
            "coverage": cov,
            "changed": changed,
        })
        if prev_hot is not None and not changed and prev_cov is not None and abs(cov - prev_cov) < hit_eps:
            stable += 1
        else:
            stable = 0
        if stable + 1 >= stable_window and plateau_at is None and len(series) >= 2:
            # the run starts at the earlier of the stable pair
            plateau_at = series[-2]["tokens"]
        prev_hot = hot
        prev_cov = cov
    return {
        "rounds": plateau_at,
        "plateau": plateau_at is not None,
        "prefix_tokens": [len(t) for t in prefixes],
        "series": series,
    }


def read_plugin_csv(path):
    """[(layer, weight_offset, expert, count)] from patch 0013's dump.

    Contract: header `layer,weight_offset,expert,count`, rows ordered by
    `weight_offset`, `#`-lines skipped, trailer `# total,<T>`. Refuses a row
    whose field count is not four (a schema that drifted is not silently
    accepted). The four columns are returned in file order.
    """
    out = []
    total = None
    with open(path, "r", encoding="utf-8") as f:
        for lineno, ln in enumerate(f, 1):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                if s.startswith("# total,"):
                    try:
                        total = int(s.split(",", 1)[1])
                    except ValueError:
                        total = None
                continue
            p = s.split(",")
            if p[0] == "layer":          # header line
                continue
            if len(p) != 4:
                raise ValueError(
                    f"{path}:{lineno}: patch-0013 CSV row has {len(p)} fields, "
                    f"expected 4 (layer,weight_offset,expert,count): {s!r}")
            out.append((int(p[0]), int(p[1]), int(p[2]), int(p[3])))
    if total is not None and total != sum(r[3] for r in out):
        raise ValueError(
            f"{path}: '# total,{total}' disagrees with the summed counts "
            f"({sum(r[3] for r in out)})")
    return out


def join_plugin_to_trace(rows, plugin_rows, layer_key_by_index):
    """Cross-check the plugin CSV against the canonical summary on
    `(weight_offset, expert)`.

    `layer_key_by_index` maps decoder-layer index -> the layer's weight-file
    offset (patch 0018's `layer_key`, the same key patch 0013 writes as
    `weight_offset`). Returns `(plugin_counts, trace_counts, mismatches)`,
    each keyed by `(weight_offset, expert)`; a mismatch is a key whose two
    counts differ (or is absent on one side).
    """
    plugin = defaultdict(int)
    for _lay, off, e, c in plugin_rows:
        plugin[(off, e)] += c
    trace = defaultdict(int)
    for _tok, lay, experts in rows:
        off = layer_key_by_index.get(lay)
        if off is None:
            continue
        for e in experts:
            trace[(off, e)] += 1
    keys = sorted(set(plugin) | set(trace))
    mismatches = [(k, plugin.get(k, 0), trace.get(k, 0))
                  for k in keys if plugin.get(k, 0) != trace.get(k, 0)]
    return plugin, trace, mismatches


def parse_call_trace(path, strict=True):
    """`(call_seq, layer_key, top_k, [expert_id, ...])` rows from patch 0044's
    per-call trace.

    Line grammar: `<call_seq> <layer_key> <top_k> <expert id...>`. `#`-lines
    and blanks are skipped. A row with fewer than four fields (a call with no
    ids) is malformed and refused.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, ln in enumerate(f, 1):
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if len(p) < 4:
                if strict:
                    raise ValueError(
                        f"{path}:{lineno}: malformed call-trace row {s!r} "
                        f"(need 'call_seq layer_key top_k expert...')")
                continue
            try:
                seq = int(p[0])
                lk = int(p[1])
                tk = int(p[2])
                ids = [int(x) for x in p[3:]]
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: non-integer field in {s!r}") from e
            if seq < 0 or lk < 0 or tk < 0 or any(x < 0 for x in ids):
                raise ValueError(f"{path}:{lineno}: negative field in {s!r}")
            rows.append((seq, lk, tk, ids))
    return rows


def split_topk_chunks(ids, top_k):
    """Split a call's flattened routed ids into per-token chunks of `top_k`.

    Raises on a mis-sized call rather than truncating: a wrong chunk is silent
    downstream, so it must fail here. `top_k <= 0` and a non-multiple length
    are both refused.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if len(ids) == 0 or len(ids) % top_k != 0:
        raise ValueError(
            f"ids length {len(ids)} is not a positive multiple of top_k "
            f"{top_k}; refusing rather than truncating")
    return [ids[i:i + top_k] for i in range(0, len(ids), top_k)]


def layer_key_index_map(layer_keys, explicit=None):
    """Map each `layer_key` to its 0-based decoder-layer index.

    Default (no `explicit`): ascending weight-file offset (export) order, the
    identity patch 0018 already assumes. `explicit` is a decoder-index ->
    layer_key map (the shape `seed_text` emits); it must be a bijection and
    must cover every key the trace carries, else the map is refused loudly
    (a re-ordered export must not silently be read as decoder order).
    """
    keys = sorted(set(layer_keys))
    if explicit is None:
        return {k: i for i, k in enumerate(keys)}
    idx_to_key = {int(k): int(v) for k, v in explicit.items()}
    vals = list(idx_to_key.values())
    if len(set(vals)) != len(vals):
        raise ValueError(f"decoder indices are not unique in the map: {vals}")
    inv = {v: k for k, v in idx_to_key.items()}
    missing = [k for k in keys if k not in inv]
    if missing:
        raise ValueError(
            f"trace layer_key(s) absent from the exported map: {missing}; "
            f"refusing (the map and the artifact disagree)")
    return inv


def call_trace_to_v1(call_rows, layer_key_map=None, skip_batched=False,
                     from_seq=0):
    """Convert a patch-0044 per-call trace to format v1.

    Decode (T=1) rows: every call carries exactly one `top_k` chunk. Tokens are
    reconstructed by counting a repeated `layer_key` as the next token (one
    call per layer per decode step), which is exact for the autoregressive
    decode loop; a call carrying more than one token's ids is a batched/prefill
    call, and for prefill the per-token `token_idx` is not defined by this
    trace, so it is not guessed.

    A served trace opens with that batched prefill call, so `skip_batched=True`
    skips it instead of refusing it -- but a skipped call is never dropped
    silently: every skip is counted and its token count reported, and a trace
    in which *every* call was batched is REFUSED rather than converted to an
    empty census. An empty call list stays an empty conversion.

    `from_seq` drops every call with `call_seq < from_seq` before conversion.
    It is the corpus floor: a served process records the load-time plateau /
    activation probes too (they route experts through the same provider), and
    those calls are not the corpus. The caller passes the call count recorded
    at the instant the corpus request was posted.

    A skipped prefill that ended mid-sequence is undetectable from the trace
    alone, so token labels after it are a reconstruction: aggregate counts do
    not depend on them, the LRU/plateau do.
    """
    selected = [r for r in call_rows if r[0] >= from_seq]
    lk_index = layer_key_index_map([r[1] for r in selected], layer_key_map)
    rows = []
    tok = 0
    seen = set()
    skipped_calls = 0
    skipped_tokens = 0
    for _seq, lk, top_k, ids in sorted(selected, key=lambda r: r[0]):
        chunks = split_topk_chunks(ids, top_k)
        if len(chunks) != 1:
            if not skip_batched:
                raise ValueError(
                    f"call seq {_seq} carries {len(chunks)} tokens' ids; the "
                    f"decode converter refuses batched/prefill calls (use "
                    f"--skip-batched to skip and report them)")
            skipped_calls += 1
            skipped_tokens += len(chunks)
            continue
        if lk in seen:
            tok += 1
            seen = set()
        seen.add(lk)
        rows.append((tok, lk_index[lk], sorted(chunks[0])))
    if selected and not rows:
        raise ValueError(
            f"no decode rows in the call trace: all {skipped_calls} call(s) "
            f"carried more than one token's ids; refusing to emit an empty "
            f"census")
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, {"tokens": (tok + 1) if rows else 0, "layer_keys": len(lk_index),
                  "calls": len(call_rows),
                  "calls_selected": len(selected),
                  "from_call_seq": from_seq,
                  "batched_calls_skipped": skipped_calls,
                  "batched_tokens_skipped": skipped_tokens}


def _check_range(from_seq, to_seq):
    """Refuse an inverted call-seq range. A regime is selected by a half-open
    range, so a range that cannot select anything must fail loudly rather than
    emit an empty census (or a silent all-zero join)."""
    if to_seq is not None and to_seq < from_seq:
        raise ValueError(
            f"call_seq range [{from_seq}, {to_seq}) is inverted; the ceiling is "
            f"exclusive and must be >= the floor")


def census_from_call_trace(call_rows, layer_key_map=None, from_seq=0, to_seq=None):
    """[(layer, expert, count)] over EVERY routed id of every selected call.

    This is the aggregate census, and it is derived from the call trace
    itself, NOT from the decode-only v1 rows: a batched prefill call carries
    many tokens' ids and the aggregate needs no token label, so every id of
    every call in the half-open range `[from_seq, to_seq)` is one routed
    access. `call_trace_to_v1` remains the decode-only converter for the
    row-level consumers (LRU replay, rounds-to-plateau), whose token labels are
    a reconstruction.

    **The range is how a REGIME is selected**, and that matters because
    coverage is a property of the (seed x regime) pair, not of the seed alone:
    on the served window 004 trace the same 5-slot seed covered 4.11% of the
    PREFILL and 13.86% of the DECODE, and a seed calibrated on the prefill
    doubled the prefill coverage (8.11%) at the same budget. A served trace
    starts with the load-time probe, then the batched prefill calls, then the
    decode calls, so prefill-only is `[call_seq_start, decode_start)`.

    Returns `(summary, meta)`; `summary` is sorted by `(layer, expert)` in the
    trace's decoder-layer space (ascending export order by default).
    """
    calls = len(call_rows)
    _check_range(from_seq, to_seq)
    selected_rows = [r for r in call_rows
                     if r[0] >= from_seq and (to_seq is None or r[0] < to_seq)]
    if not selected_rows:
        where = (f"call_seq >= {from_seq}" if to_seq is None
                 else f"call_seq in [{from_seq}, {to_seq})")
        what = ("the call trace is empty" if calls == 0
                else f"{where} selects none of the {calls} call(s)")
        raise ValueError(
            f"{what}; refusing to emit an empty census (not a census)")
    lk_index = layer_key_index_map([r[1] for r in selected_rows], layer_key_map)
    hist = defaultdict(int)
    selected = batched = decode = tokens = accesses = 0
    for _seq, lk, top_k, ids in selected_rows:
        selected += 1
        chunks = split_topk_chunks(ids, top_k)
        if len(chunks) > 1:
            batched += 1
        else:
            decode += 1
        tokens += len(chunks)
        li = lk_index[lk]
        for e in ids:
            hist[(li, e)] += 1
            accesses += 1
    summary = [(lay, e, hist[(lay, e)]) for (lay, e) in sorted(hist)]
    meta = {"calls": calls, "calls_selected": selected,
            "batched_calls": batched, "decode_calls": decode,
            "tokens_observed": tokens, "accesses": accesses,
            "from_call_seq": from_seq, "to_call_seq": to_seq,
            "layers": len(lk_index)}
    return summary, meta


def join_plugin_to_call_trace(call_rows, plugin_rows, from_seq=0, to_seq=None):
    """Cross-check patch 0013's CSV against the call-trace census on
    `(weight_offset, expert)`.

    The call trace already carries the raw `layer_key` (patch 0018's
    weight-file offset, the same value patch 0013 writes as `weight_offset`),
    so this join needs no export-order assumption and no decoder-index map.
    The same half-open `[from_seq, to_seq)` range as the census applies, so a
    join can be taken over one REGIME.
    Returns `(plugin_counts, trace_counts, mismatches)` keyed by
    `(layer_key, expert)`; a mismatch is a key whose counts differ or which is
    absent on one side.
    """
    _check_range(from_seq, to_seq)
    plugin = defaultdict(int)
    for _lay, off, e, c in plugin_rows:
        plugin[(off, e)] += c
    trace = defaultdict(int)
    for seq, lk, _top_k, ids in call_rows:
        if seq < from_seq or (to_seq is not None and seq >= to_seq):
            continue
        for e in ids:
            trace[(lk, e)] += 1
    keys = sorted(set(plugin) | set(trace))
    mismatches = [(k, plugin.get(k, 0), trace.get(k, 0))
                  for k in keys if plugin.get(k, 0) != trace.get(k, 0)]
    return plugin, trace, mismatches


def require_attribution(prov):
    """Refuse a provenance dict missing a non-empty artifact and card."""
    artifact = prov.get("artifact_sha256") or prov.get("artifact")
    if not artifact or not prov.get("card"):
        raise ValueError(
            "provenance must carry a non-empty artifact_sha256= (or artifact=) "
            "and card= -- the plugin cannot know either, so the harness "
            "injects them; refusing to write an unattributable census")
    return artifact


def read_provenance_file(path):
    """`key=value` pairs from a harness-written provenance file.

    The plugin cannot know the artifact, card, KV dtype or digests, so the
    harness writes them here; bare `key=value` lines and `#`-prefixed v1
    header spellings are both accepted, and free text is ignored.
    """
    prov = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            body = ln.strip().lstrip("#").strip()
            if body.startswith("arcint routing trace"):
                continue
            for tok in body.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    prov[k.strip()] = v.strip()
    return prov


def seed_text(hot, layer_key_by_index=None):
    """The static-partition seed as text, ONE line per layer
    `<key> <expert> <expert> ...`.

    The line key is patch 0018's structural `layer_key` (the layer's first OTD
    weight-file offset) when `layer_key_by_index` maps decoder-layer index ->
    `layer_key`; otherwise the decoder-layer index itself. The space is
    stated in the header (`# space=layer_key` / `# space=layer`) so a
    consumer can never silently read one key space as the other -- the
    plugin's own parser REFUSES a file that does not declare
    `space=layer_key`. The experts are emitted in frequency-rank order (count
    desc, id asc); the consumer re-sorts ascending for slot assignment.

    This is format v2. The v1 form this replaces was one `layer expert` pair
    per line and carried no space declaration, so a decoder-index seed and a
    `layer_key` seed were indistinguishable.
    """
    if layer_key_by_index:
        keys = list(layer_key_by_index.values())
        if len(set(keys)) != len(keys):
            raise ValueError(
                "seed_text: the layer_key map maps two decoder layers to the "
                "same layer_key; a duplicate key is refused rather than "
                "written (the plugin would refuse it at load)")
    space = "layer_key" if layer_key_by_index else "layer"
    lines = ["# hot-set seed v2 (frequency rank, id tie-break)",
             f"# space={space}"]
    if layer_key_by_index:
        lines.append("# layer_key_by_index=" + json.dumps(
            {str(k): v for k, v in sorted(layer_key_by_index.items())}))
    for lay in sorted(hot):
        key = layer_key_by_index.get(lay) if layer_key_by_index else lay
        if key is None:
            raise ValueError(
                f"seed_text: decoder layer {lay} has no layer_key in the "
                f"supplied map")
        lines.append(" ".join([str(key)] + [str(e) for e in hot[lay]]))
    return "\n".join(lines) + "\n"


def read_census_summary(path):
    """[(layer, expert, count)] from a canonical census summary CSV.

    Contract: header `layer,expert,count`, `#`-lines skipped, optional
    trailer `# total,<T>`. Refuses a row whose field count is not three or
    whose fields are not non-negative integers. This is the CORPUS census
    (every routed id of every selected call, batched prefill included), which
    is what a hot set must be seeded from -- not the decode-only v1 rows.
    """
    out = []
    total = None
    with open(path, "r", encoding="utf-8") as f:
        for lineno, ln in enumerate(f, 1):
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                if s.startswith("# total,"):
                    try:
                        total = int(s.split(",", 1)[1])
                    except ValueError:
                        total = None
                continue
            p = s.split(",")
            if p[0] == "layer":          # header line
                continue
            if len(p) != 3:
                raise ValueError(
                    f"{path}:{lineno}: census summary row has {len(p)} "
                    f"fields, expected 3 (layer,expert,count): {s!r}")
            try:
                lay, e, c = int(p[0]), int(p[1]), int(p[2])
            except ValueError:
                raise ValueError(
                    f"{path}:{lineno}: census summary row is not three "
                    f"non-negative integers: {s!r}")
            if lay < 0 or e < 0 or c < 0:
                raise ValueError(
                    f"{path}:{lineno}: census summary row is negative: {s!r}")
            out.append((lay, e, c))
    if total is not None and total != sum(r[2] for r in out):
        raise ValueError(
            f"{path}: '# total,{total}' disagrees with the summed counts "
            f"({sum(r[2] for r in out)})")
    return out


def select_hot_set_from_counts(counts, slots_per_layer):
    """{layer: [expert, ...]} -- top-`slots_per_layer` per layer from
    a canonical census summary `[(layer, expert, count)]`.

    Same rank rule as `select_hot_set`: count DESC, ties broken by expert id
    ASC. This is the corpus-census path (every selected call's ids counted),
    so a seed built here is the regime the served path actually sees.
    """
    if slots_per_layer < 0:
        raise ValueError("slots_per_layer must be >= 0")
    by_layer = defaultdict(list)
    for lay, e, c in counts:
        by_layer[lay].append((e, c))
    out = {}
    for lay in sorted(by_layer):
        rank = sorted(by_layer[lay], key=lambda kv: (-kv[1], kv[0]))
        out[lay] = [e for e, _c in rank[:slots_per_layer]]
    return out


def _cmd_shape(args):
    rows = parse_trace(args.trace)
    prov = read_provenance(args.trace)
    shp = trace_shape(rows)
    for k, v in prov.items():
        print(f"# {k}={v}")
    print(json.dumps(shp, sort_keys=True))
    return 0


def _cmd_summary(args):
    rows = parse_trace(args.trace)
    text = write_summary(rows, args.out, read_provenance(args.trace))
    if not args.out:
        sys.stdout.write(text)
    else:
        print(f"wrote {args.out}")
    return 0


def _cmd_select(args):
    if bool(args.census) == bool(args.trace):
        raise ValueError(
            "select takes exactly one of --census / --trace")
    layer_keys = None
    if args.layer_keys:
        with open(args.layer_keys, encoding="utf-8") as f:
            raw = json.load(f)
        layer_keys = {int(k): int(v) for k, v in raw.items()}
    if args.census:
        hot = select_hot_set_from_counts(read_census_summary(args.census),
                                         args.slots_per_layer)
    else:
        rows = parse_trace(args.trace)
        hot = select_hot_set(rows, args.slots_per_layer)
    text = seed_text(hot, layer_keys)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_plateau(args):
    rows = parse_trace(args.trace)
    rep = rounds_to_plateau(rows, args.slots_per_layer,
                            stable_window=args.stable_window,
                            hit_eps=args.hit_eps,
                            max_prefixes=args.max_prefixes)
    print(f"layer,expert,count")
    for lay, e, c in canonical_summary(rows):
        print(f"{lay},{e},{c}")
    print(f"# total,{sum(c for _l, _e, c in canonical_summary(rows))}")
    print(f"# rounds_to_plateau={rep['rounds']} plateau={rep['plateau']}")
    for s in rep["series"]:
        print(f"# prefix_tokens={s['tokens']} hot_size={s['hot_size']} "
              f"coverage={s['coverage']:.6f} changed={int(s['changed'])}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


def _cmd_census_from_call_trace(args):
    call_rows = parse_call_trace(args.call_trace)
    layer_keys = None
    if args.layer_keys:
        with open(args.layer_keys, encoding="utf-8") as f:
            raw = json.load(f)
        layer_keys = {int(k): int(v) for k, v in raw.items()}
    prov = read_provenance_file(args.provenance)
    require_attribution(prov)
    summary, meta = census_from_call_trace(call_rows, layer_keys, args.from_call_seq,
                                           args.to_call_seq)
    lines = ["# census summary (patch 0044 call trace; batched calls counted)"]
    for k, v in prov.items():
        lines.append(f"# {k}={v}")
    for k, v in meta.items():
        lines.append(f"# {k}={v}")
    lines.append("layer,expert,count")
    total = 0
    for lay, e, c in summary:
        lines.append(f"{lay},{e},{c}")
        total += c
    lines.append(f"# total,{total}")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}: {len(summary)} cells, total {total}, "
              f"{meta['calls_selected']} call(s) selected of {meta['calls']} "
              f"({meta['batched_calls']} batched, {meta['decode_calls']} decode)")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_join_plugin_call_trace(args):
    call_rows = parse_call_trace(args.call_trace)
    plugin_rows = read_plugin_csv(args.csv)
    plugin, trace, mismatches = join_plugin_to_call_trace(
        call_rows, plugin_rows, args.from_call_seq, args.to_call_seq)
    print(f"# join on (weight_offset=layer_key, expert); from_call_seq={args.from_call_seq} "
          f"to_call_seq={args.to_call_seq}")
    print("# mismatches are keys where patch 0013's ALL-call CSV and the floored "
          "call-trace census disagree (the pre-floor/probe share when "
          "--from-call-seq > 0), not an instrument fault")
    print(f"plugin_keys={len(plugin)} trace_keys={len(trace)} "
          f"mismatches={len(mismatches)}")
    for (lk, e), pc, tc in mismatches[:40]:
        print(f"{lk},{e},{pc},{tc}")
    if len(mismatches) > 40:
        print(f"... ({len(mismatches) - 40} more)")
    return 0


def _cmd_plugin_csv(args):
    rows = read_plugin_csv(args.csv)
    print(f"rows={len(rows)} total={sum(r[3] for r in rows)}")
    for lay, off, e, c in rows[:10]:
        print(f"{lay},{off},{e},{c}")
    if len(rows) > 10:
        print(f"... ({len(rows) - 10} more)")
    return 0


def _cmd_from_call_trace(args):
    call_rows = parse_call_trace(args.call_trace)
    layer_keys = None
    if args.layer_keys:
        with open(args.layer_keys, encoding="utf-8") as f:
            raw = json.load(f)
        layer_keys = {int(k): int(v) for k, v in raw.items()}
    prov = read_provenance_file(args.provenance)
    require_attribution(prov)
    rows, rep = call_trace_to_v1(call_rows, layer_keys,
                                 skip_batched=args.skip_batched,
                                 from_seq=args.from_call_seq)
    prov = OrderedDict(list(prov.items()) + [
        ("source", "plugin-0044-call-trace"),
        ("calls", rep["calls"]),
        ("calls_selected", rep["calls_selected"]),
        ("from_call_seq", rep["from_call_seq"]),
        ("batched_calls_skipped", rep["batched_calls_skipped"]),
        ("batched_tokens_skipped", rep["batched_tokens_skipped"]),
        ("tokens", rep["tokens"]),
        ("layers", rep["layer_keys"]),
        # token labels are rebuilt from the layer-key wrap, not carried by the
        # trace; say so in the header rather than leaving it implicit.
        ("token_labels", "reconstructed"),
    ])
    lines = ["# arcint routing trace v1"]
    for k, v in prov.items():
        lines.append(f"# {k}={v}")
    for tok, lay, ids in rows:
        lines.append(" ".join([str(tok), str(lay)] + [str(x) for x in ids]))
    text = "\n".join(lines) + "\n"
    note = (f"{rep['batched_calls_skipped']} batched prefill call(s), "
            f"{rep['batched_tokens_skipped']} token(s), skipped")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}: {rep['tokens']} tokens x {rep['layer_keys']} layers "
              f"from {rep['calls']} calls ({note})")
    else:
        sys.stdout.write(text)
        if rep["batched_calls_skipped"]:
            sys.stderr.write(f"# {note}\n")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("shape", help="trace shape and provenance")
    p.add_argument("--trace", required=True)
    p.set_defaults(func=_cmd_shape)

    p = sub.add_parser("summary", help="canonical layer,expert,count summary")
    p.add_argument("--trace", required=True)
    p.add_argument("--out", default="")
    p.set_defaults(func=_cmd_summary)

    p = sub.add_parser("select", help="frequency-ranked hot-set seed")
    p.add_argument("--trace", default="",
                   help="format-v1 trace; mutually exclusive with --census")
    p.add_argument("--census", default="",
                   help="canonical census summary CSV (layer,expert,count); the "
                        "CORPUS census a hot set is seeded from; mutually "
                        "exclusive with --trace")
    p.add_argument("--slots-per-layer", type=int, required=True)
    p.add_argument("--out", default="")
    p.add_argument("--layer-keys", default="",
                   help="JSON map decoder-layer index -> weight-file offset "
                        "(patch 0018's layer_key); emits the seed keyed by "
                        "layer_key with a `# space=layer_key` header")
    p.set_defaults(func=_cmd_select)

    p = sub.add_parser("plateau", help="summary + rounds-to-plateau")
    p.add_argument("--trace", required=True)
    p.add_argument("--slots-per-layer", type=int, required=True)
    p.add_argument("--stable-window", type=int, default=2)
    p.add_argument("--hit-eps", type=float, default=1e-3)
    p.add_argument("--max-prefixes", type=int, default=None)
    p.add_argument("--out", default="")
    p.set_defaults(func=_cmd_plateau)

    p = sub.add_parser("plugin-csv", help="parse/validate patch 0013's CSV")
    p.add_argument("--csv", required=True)
    p.set_defaults(func=_cmd_plugin_csv)

    p = sub.add_parser("from-call-trace",
                       help="convert patch 0044's per-call trace to format v1 (decode)")
    p.add_argument("--call-trace", required=True)
    p.add_argument("--provenance", required=True,
                   help="harness-written key=value file; must carry a NON-EMPTY "
                        "artifact_sha256= (or artifact=) and card= -- the plugin "
                        "cannot know them; the rest of §2's header is not checkable here")
    p.add_argument("--skip-batched", action="store_true",
                   help="skip (and report) a batched prefill call instead of refusing; "
                        "an all-batched trace is still refused")
    p.add_argument("--from-call-seq", type=int, default=0,
                   help="drop calls with call_seq below this floor (the corpus start; "
                        "excludes the load-time probe calls)")
    p.add_argument("--out", default="")
    p.add_argument("--layer-keys", default="",
                   help="JSON decoder-index -> layer_key map; default is ascending export order")
    p.set_defaults(func=_cmd_from_call_trace)

    p = sub.add_parser("census-from-call-trace",
                       help="the aggregate census (every call's ids, batched included) "
                            "from patch 0044's per-call trace")
    p.add_argument("--call-trace", required=True)
    p.add_argument("--provenance", required=True,
                   help="harness-written key=value file; must carry a NON-EMPTY "
                        "artifact_sha256= (or artifact=) and card=")
    p.add_argument("--from-call-seq", type=int, default=0,
                   help="count only calls with call_seq >= this floor")
    p.add_argument("--to-call-seq", type=int, default=None,
                   help="EXCLUSIVE ceiling; floor+ceiling select one REGIME "
                        "(e.g. prefill-only = [call_seq_start, decode_start))")
    p.add_argument("--out", default="")
    p.add_argument("--layer-keys", default="")
    p.set_defaults(func=_cmd_census_from_call_trace)

    p = sub.add_parser("join-plugin-call-trace",
                       help="join patch 0013's CSV to the call-trace census on "
                            "(weight_offset=layer_key, expert)")
    p.add_argument("--call-trace", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--from-call-seq", type=int, default=0)
    p.add_argument("--to-call-seq", type=int, default=None,
                   help="EXCLUSIVE ceiling; the same regime range as the census")
    p.set_defaults(func=_cmd_join_plugin_call_trace)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
