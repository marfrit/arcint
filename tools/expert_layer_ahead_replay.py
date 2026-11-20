#!/usr/bin/env python3
"""Replay of a layer-ahead expert predictor over an LAC1 router-input capture.

Campaign expert-foresight, "Queued (operator, 2026-11-19): a layer-ahead
predictor", step 2 (`docs/campaigns/expert-foresight.md`): "replay the next
layer's router on this layer's input, with and without a learned correction,
against the current cache baseline at 1 / 2 / 4 layers ahead". The capture is
tools/expert_router_capture_read.py's LAC1 file (tools/
expert_router_capture_format.h): per token and MoE layer the router input (the
hidden state after the layer's ffn norm) and the routed ids in rank order.

  expert_layer_ahead_replay.py CAPTURE GGUF [--depths 1 2 4] [--ranks 4 8 16]
       [--slots 5915] [--profile FILE] [--discount-shard K] [--block N]
       [--lr X] [--json FILE] [--limit N] [--gguf-py DIR]

Arms, per depth d (1, 2, 4 layers ahead) and target layer lt = l + d:

  A reuse      layer lt's router weight applied to layer l's captured input,
               no training. `reuse = W_lt . x_l`.
  B corrected  `W_lt . x_l + U_lt . (V_lt . x_l)`, per (target layer, d) a
               rank-r pair, trained ONLINE in token order: predict, score,
               then update (test-then-train; a token's own update never
               enters its score). Loss, the reference's shape (Strata-HotRod's
               predictor, `paper`, ideas only, reimplemented): centred MSE on
               the target layer's real top-32 logits plus a hinge keeping every
               other expert below the 32nd logit minus 0.1. The real logits are
               layer lt's router on its OWN captured input (`W_lt . x_lt`).
  C oracle     the real ids (the ceiling). For K below n_used the ceiling's
               all-used recall is capped at K / n_used by construction; the
               rank-0 recall is 100 %.

Metrics, per (arm, depth, K) with K in {4, 8, 16, 32}, prompt and decode
separately:
  recall_top1  the real rank-0 expert inside the predicted top-K, mean a
               (token, target layer);
  recall_used  the real n_used ids inside the predicted top-K, ids-weighted
               (|topK & real ids| / n_used), mean a token.
And the cache metric the campaign needs: only the experts NOT resident in the
card's cache. The residency is modelled by tools/expert_cache_replay.py's own
mechanism -- ecr.simulate(layers, steps, "strata", ...), the served cache's
policy, 96 swaps a plan every 6 tokens landing --delay steps later, over the
capture's tokens in file order (a prompt's tokens are real routing and really
move the cache) -- and these are counted only over the real experts that are
NOT resident at that token:
  recall_top1_nonres  of the tokens whose real rank-0 expert is non-resident,
                      the share whose prediction still lands inside top-K;
  recall_nonres_used  ids-weighted recall over the non-resident real ids.

Slots, per layer, as the served unit runs them: `--slots N` over the capture's
layers, split flat (5,915 gives 123 a layer and +1 on the first 11; the served
B70 unit now runs 6,639 at 13.05 GiB -- pass `--slots 6639 --profile
expert-profile-strata.bin` for the production placement, and the JSON says
`slot_mode: flat | profile`, so a flat run is never read as the unit's).
`--profile STRP` uses
tools/expert_cache_replay.py's read_profile/profile_slots walk instead (the
production distribution, 93-155 a layer), the engine's own distribution rule.
`--discount-shard K` divides the share by K, for the split-expert-file layout
(see docs/campaigns/expert-foresight.md on the sharded series).

Report: one JSON with every number (--json FILE, or stdout) and a text table a
standard width. The depths' first layers have no (l - d) source and are
reported in `layers_without_source`, never faked: for d, target layers 0..d-1
are excluded from every arm.

The depths' router matmuls run on threads: numpy releases the GIL in matmul and a
reference-BLAS build is single-threaded, so three depths overlap (measured on the
synthetic 4,096-token file: the number is in the campaign's replay record, this
tool's own run time is printed nowhere and reported in the handover). Nothing is
faked across depths: each depth's corrections and counters are its own.

Device-free. Router weights come from the GGUF's `blk.{lt}.ffn_gate_inp.weight`
(the pinned llama tree's gguf-py; set ARCINT_GGUF_PY or --gguf-py to its
gguf-py directory, or have the module importable). The tensor is F32 in the
conversion (llama-quant.cpp keeps `ffn_gate_inp` unquantized); F16, BF16 and
quantized types are dequantized when found. Weights are read as gguf-py hands
them over; a file whose dims are transposed is recognized by the geometry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import expert_cache_replay as ecr          # noqa: E402
import expert_router_capture_read as capture  # noqa: E402

KS = (4, 8, 16, 32)
TOP32 = 32
HINGE = 0.1


# ------------------------------------------------------------ GGUF routers ----
def resolve_gguf(gguf_py=None):
    """The pinned gguf-py: an explicit --gguf-py, then either GGUF_PY or
    ARCINT_GGUF_PY, then ARCINT_LLAMA_DIR/gguf-py, then the module itself."""
    candidates = [gguf_py,
                  os.environ.get("GGUF_PY"),
                  os.environ.get("ARCINT_GGUF_PY")]
    llama = os.environ.get("ARCINT_LLAMA_DIR")
    if llama:
        candidates.append(os.path.join(llama, "gguf-py"))
    for c in candidates:
        if c and os.path.isdir(c) and str(c) not in sys.path:
            sys.path.insert(0, str(c))
    try:
        import gguf
    except Exception as e:  # noqa: BLE001
        raise SystemExit("gguf-py is not importable (%s): pass --gguf-py DIR or set ARCINT_GGUF_PY" % e)
    return gguf


def _dequant(gguf, data, ttype):
    if ttype == 0:                     # F32
        return np.asarray(data, np.float32)
    if ttype == 1:                     # F16
        return np.asarray(data, np.float32)
    if ttype == 30:                    # BF16: gguf-py hands it over as uint8 pairs
        # join each pair into the bf16 bit pattern, then place it in the high half of an f32
        u16 = np.ascontiguousarray(data).view(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape)
    from gguf import quants
    return np.asarray(quants.dequantize(data, ttype), np.float32)


def _orient(w, n_embd, n_expert):
    """A router as (n_embd, n_expert): logits = w.T @ x. gguf-py's data is the
    file's row order, dims[0] fastest; the engine's shape is {n_embd, n_expert},
    so data is (n_embd, n_expert) and a transposed file (n_expert, n_embd) is
    recognized by the geometry."""
    if w.ndim != 2:
        raise SystemExit("router weight is %dd, not 2d" % w.ndim)
    if w.shape == (n_embd, n_expert):
        return w
    if w.shape == (n_expert, n_embd):
        return w.T
    raise SystemExit("router weight %s, neither (%d, %d) nor (%d, %d)"
                     % (w.shape, n_embd, n_expert, n_expert, n_embd))


def load_routers(path, n_layer, n_embd, n_expert, gguf_py=None):
    """(n_layer, n_embd, n_expert) float32; logits = einsum('le,lek->lk', x, W)."""
    gguf = resolve_gguf(gguf_py)
    r = gguf.GGUFReader(path)
    by_name = {t.name: t for t in r.tensors}
    W = np.zeros((n_layer, n_embd, n_expert), np.float32)
    missing = []
    for il in range(n_layer):
        name = "blk.%d.ffn_gate_inp.weight" % il
        t = by_name.get(name)
        if t is None:
            missing.append(name)
            continue
        w = _dequant(gguf, t.data, t.tensor_type)
        if w.size != n_embd * n_expert:
            raise SystemExit("%s: %d values, not %dx%d" % (name, w.size, n_embd, n_expert))
        # gguf-py's data is the file's dims reversed (`t.shape` is the file's
        # dims reversed too, so data's numpy shape is reversed(t.shape)); a
        # quantized tensor's flat bytes come back in that same row order.
        W[il] = _orient(w.reshape(tuple(reversed(t.shape))), n_embd, n_expert)
    if missing:
        raise SystemExit("%s: no %d of %d router tensors (first missing %s)"
                         % (path, len(missing), n_layer, missing[0]))
    return W


# --------------------------------------------------------------- capture ----
class Capture:
    """positions, phases and ids of every token (small); the inputs are streamed
    block by block, never held whole (a Flash-Next token is 246,729 bytes)."""

    def __init__(self, path, limit=None):
        h = capture.read_header(path)
        self.header = h
        self.path = path
        self.positions = []
        self.phases = []
        self.token_ids = []
        ids = []
        for tok in capture.iter_raw(path, limit=limit):
            self.positions.append(tok.position)
            self.phases.append(tok.phase)
            self.token_ids.append(tok.token_id)
            row = np.empty((h.n_layer, h.n_used), np.int16)
            for il, raw in enumerate(tok.layers):
                row[il] = np.frombuffer(raw.ids_bytes, "<i2", h.n_used)
            ids.append(row)
        self.n_tokens = len(ids)
        self.ids = np.stack(ids) if ids else np.zeros((0, h.n_layer, h.n_used), np.int16)
        self.positions = np.asarray(self.positions, np.int32)
        self.phases = np.asarray(self.phases, np.int8)
        self.token_ids = np.asarray(self.token_ids, np.int32)

    def blocks(self, block):
        """(start, X, ids, phases) blocks in file order. X is (b, n_layer,
        n_embd) float32; ids int32 rank order."""
        h = self.header
        in_bytes = 2 * h.n_embd
        ids_bytes = 2 * h.n_used
        layer_bytes = in_bytes + ids_bytes
        rec = h.record_bytes
        n = 0
        with open(self.path, "rb") as f:
            f.seek(capture.HEADER.size)
            while n < self.n_tokens:
                want = min(block, self.n_tokens - n)
                chunk = f.read(rec * want)
                if len(chunk) < rec * want:
                    raise SystemExit("%s: token %d is torn (%d of %d bytes)"
                                     % (self.path, n, len(chunk), rec * want))
                X = np.empty((want, h.n_layer, h.n_embd), np.float32)
                ids = np.empty((want, h.n_layer, h.n_used), np.int32)
                for j in range(want):
                    base = j * rec + capture.TOKEN.size
                    for il in range(h.n_layer):
                        off = base + il * layer_bytes
                        X[j, il] = np.frombuffer(chunk, "<f2", h.n_embd, off)
                        ids[j, il] = np.frombuffer(chunk, "<i2", h.n_used, off + in_bytes)
                yield n, X, ids, self.phases[n:n + want]
                n += want


# ---------------------------------------------------------------- cache ----
def slot_mode(profile):
    """What the JSON says the slot placement is: a profile's engine walk, or a
    flat split that is an assumption, never the served unit's."""
    return "profile" if profile else "flat"


def flat_slots(n_layer, n_expert, slots=5915, discount_shard=1):
    """The served unit's 5,915 slots over 48 layers, flat: 123 a layer and +1 on
    the first 11. `discount_shard` divides the share (the split-file layout).
    The shape ecr.simulate expects: il, n_expert, n_slots, slots."""
    share = int(slots // discount_shard)
    base, extra = divmod(share, n_layer)
    out = []
    for il in range(n_layer):
        ns = base + (1 if il < extra else 0)
        slots = np.arange(ns, dtype=np.int32)
        out.append(dict(il=il, n_expert=n_expert, n_slots=ns, slots=slots))
    return out


def profile_layers(path, n_layer, n_expert, slots=5915):
    """The STRP profile's ranking through ecr.profile_slots: the engine's global
    walk, which gives each layer the slots the ranking spends there."""
    ranked, _ = ecr.read_profile(path, n_layer, n_expert)
    base = flat_slots(n_layer, n_expert, slots)
    return ecr.profile_slots(base, ranked)


def cache_residency(cap, layers, delay=2, every=6, cap_swaps=96, policy="strata"):
    """(n_tokens, n_layer, n_expert) bool: the modelled residency at each token,
    from ecr.simulate's own policy walk (its `resid_log` addition: a copy of
    the residency BEFORE the step's hits count, as the engine counts them)."""
    steps = [(1, cap.ids[t].astype(np.int32)) for t in range(cap.n_tokens)]
    log = []
    hit, _, occ, swaps = ecr.simulate(layers, steps, policy, delay=delay,
                                        adapt_every=every, max_swaps=cap_swaps,
                                        resid_log=log)
    resid = np.stack(log) if log else np.zeros((0,))
    return hit, occ, swaps, resid
    return hit, occ, swaps, resid


# --------------------------------------------------------------- metrics ----
def router_logits(X, W):
    """(b, n_targets, n_expert) = W_lt . x_l a target layer and token: the batched
    matmul over layers, each layer's own weight against every token's row."""
    return np.swapaxes(np.matmul(np.swapaxes(X, 0, 1), W), 0, 1)


def top_k(logits, K):
    """The predicted ids by logit, descending; stable ties (np.argsort default
    is stable for the -logits key's own ordering)."""
    K = min(K, logits.shape[-1])
    return np.argsort(-logits, axis=-1, kind="stable")[..., :K]


def top32(logits):
    """The ranked top-32 ids, once, from which every K <= 32 is a slice: a
    partition for the 32 candidates and a stable sort inside them, instead of a
    full 512-wide sort a K. With n_expert <= 32 it is the full order."""
    ne = logits.shape[-1]
    n = min(TOP32, ne)
    if n == ne:
        return np.argsort(-logits, axis=-1, kind="stable")
    idx = np.argpartition(-logits, n - 1, axis=-1)[..., :n]
    val = np.take_along_axis(logits, idx, axis=-1)
    order = np.argsort(-val, axis=-1, kind="stable")
    return np.take_along_axis(idx, order, axis=-1)


def count_rows(pk, ids_sel, res_sel):
    """The Stats terms over a batch of (token, target layer) rows, as sums: `pk`
    (b, n_targets, K) predicted ids, `ids_sel` (b, n_targets, n_used) real ids
    in rank order, `res_sel` (b, n_targets, n_expert) residency at the token."""
    top1 = ids_sel[..., 0]
    eq = pk[..., :, None] == ids_sel[..., None, :]        # (b, nt, K, n_used)
    hit = eq.any(axis=-1)                                 # per predicted id
    top1_hit = (pk == top1[..., None]).any(axis=-1)
    res_ids = np.take_along_axis(res_sel, ids_sel, axis=-1)
    nres = ~res_ids
    res_top1 = np.take_along_axis(res_sel, top1[..., None], axis=-1)[..., 0]
    return dict(
        top1_hit=int(top1_hit.sum()), top1_n=int(top1_hit.size),
        used_hit=int(hit.sum()), used_n=int(ids_sel.size),
        nres_top1_hit=int((top1_hit & ~res_top1).sum()),
        nres_top1_n=int((~res_top1).sum()),
        nres_hit=int((eq & nres[..., None, :]).any(axis=-1).sum()),
        nres_n=int(nres.sum()),
        nres_tokens=int((nres.any(axis=-1)).sum()))


def row_stats(pred_ids, real_ids, resid_row=None):
    """One (token, target layer) as a dict of 1-counted terms. `real_ids` n_used
    ids in rank order; `resid_row` the residency at that token, or None."""
    real = [int(e) for e in real_ids]
    pred = set(int(e) for e in pred_ids)
    top1 = real[0]
    out = dict(top1_hit=int(top1 in pred), top1_n=1,
               used_hit=sum(1 for e in real if e in pred), used_n=len(real))
    out.update(nres_top1_n=0, nres_top1_hit=0, nres_hit=0, nres_n=0,
               nres_tokens=0)
    if resid_row is not None:
        nres = [e for e in real if not resid_row[e]]
        out["nres_n"] = len(nres)
        out["nres_hit"] = sum(1 for e in nres if e in pred)
        out["nres_tokens"] = int(bool(nres))
        if not resid_row[top1]:
            out["nres_top1_n"] = 1
            out["nres_top1_hit"] = int(top1 in pred)
    return out


class Stats:
    """Counters keyed (arm, depth, K, phase); phase is 'prompt', 'decode' or
    'all'. Every term is a count; the report's ratios divide them."""

    TERMS = ("top1_hit", "top1_n", "used_hit", "used_n", "nres_top1_hit",
             "nres_top1_n", "nres_hit", "nres_n", "nres_tokens")

    def __init__(self):
        self.c = {}

    def add(self, key, row):
        c = self.c.setdefault(key, dict.fromkeys(self.TERMS, 0))
        for t in self.TERMS:
            c[t] += row.get(t, 0)

    def add_sums(self, key, sums):
        """One aggregation a block: `sums` carries every TERM as a total."""
        c = self.c.setdefault(key, dict.fromkeys(self.TERMS, 0))
        for t in self.TERMS:
            c[t] += sums.get(t, 0)

    def ratio(self, key, a, b):
        c = self.c.get(key)
        if not c or not c[b]:
            return None
        return c[a] / c[b]

    def block(self, arm, depth, K):
        out = {}
        for phase in ("prompt", "decode", "all"):
            key = (arm, depth, K, phase)
            out[phase] = dict(
                tokens=self.c.get(key, {}).get("top1_n", 0),
                recall_top1=self.ratio(key, "top1_hit", "top1_n"),
                recall_used=self.ratio(key, "used_hit", "used_n"),
                recall_top1_nonres=self.ratio(key, "nres_top1_hit", "nres_top1_n"),
                recall_nonres_used=self.ratio(key, "nres_hit", "nres_n"),
                nres_tokens=self.c.get(key, {}).get("nres_tokens", 0))
        return out


# ------------------------------------------------- learned correction ----
class Correction:
    """Per target layer a rank-r pair U (n_expert, r), V (r, n_embd), applied to
    the SOURCE layer's captured input: delta = U (V x). Trained online in token
    order, predict-then-score-then-update; a token's own label only enters the
    update after its score.

    `events` records ('predict'|'update', token) per call, so a test can pin
    that scoring precedes updating; `losses` keeps the per-token loss."""

    def __init__(self, n_layer, n_embd, n_expert, depth, rank, lr=0.05, seed=0):
        self.depth = depth
        self.rank = rank
        self.lr = lr
        self.targets = np.arange(depth, n_layer, dtype=np.int32)
        self.sources = self.targets - depth
        nt = self.targets.size
        rng = np.random.default_rng(seed)
        self.U = np.zeros((nt, n_expert, rank), np.float32)
        self.V = rng.standard_normal((nt, rank, n_embd)).astype(np.float32) / n_embd ** 0.5
        self.events = []
        self.losses = []

    def predict(self, x, token=None):
        """Delta logits (n_targets, n_expert) from the source layers' inputs."""
        xs = x[self.sources]                                  # (nt, n_embd)
        z = np.einsum("trd,td->tr", self.V, xs)                # (nt, rank)
        delta = np.einsum("tnr,tr->tn", self.U, z)             # (nt, n_expert)
        self.events.append(("predict", token))
        self.last = (xs, z, delta)
        return delta

    def update(self, target_logits, token=None, orders=None, clip=10.0):
        """One SGD step on the pre-update parameters against this token's real
        target logits; returns the loss the score was judged at. A global-norm
        clip keeps the bilinear pair's early gradients from overshooting (a
        random V and a zero U grow fast); it is the update's own numerical
        guard, not a schedule."""
        xs, z, delta = self.last
        g = np.empty_like(target_logits)
        loss = 0.0
        for ti in range(delta.shape[0]):
            g[ti], li = loss_grad(delta[ti], target_logits[ti],
                                  None if orders is None else orders[ti])
            loss += li / delta.shape[0]
        # delta's gradient in raw logit space: the centring is subtracted back
        dU = np.einsum("tn,tr->tnr", g, z)
        dV = np.einsum("tnr,tn->tr", self.U, g)
        dV = np.einsum("tr,td->trd", dV, xs)
        norm = float(np.sqrt(np.sum(dU * dU) + np.sum(dV * dV)))
        scale = (clip / norm) if norm > clip else 1.0
        self.U -= self.lr * scale * dU
        self.V -= self.lr * scale * dV
        self.events.append(("update", token))
        self.losses.append(float(loss))
        return loss


def loss_grad(pred, target, order=None):
    """The reference-shaped loss on ONE target row: centred MSE over the real
    top-32 logits and a hinge keeping the others below the 32nd minus 0.1.
    Returns (dL/dpred raw, loss). Centring subtracts the mean over all logits;
    a gradient through it is one mean subtraction. `order` is the row's ranked
    top-32 when the caller already has it (the run's real logits do)."""
    ne = target.size
    n_top = min(TOP32, ne)
    top = (np.argsort(-target, kind="stable") if order is None else order)[:n_top]
    mask = np.zeros(ne, bool)
    mask[top] = True
    pc = pred - pred.mean()
    tc = target - target.mean()
    n = float(n_top)
    g = np.zeros(ne, np.float64)
    g[mask] = 2.0 * (pc[mask] - tc[mask]) / n
    loss = float(((pc[mask] - tc[mask]) ** 2).mean())
    others = ~mask
    if others.any():
        thr = tc[top[-1]] - HINGE
        viol = pc[others] > thr
        k = float(others.sum())
        if viol.any():
            loss += float(np.maximum(pc[others] - thr, 0.0).mean())
        g[others] += viol / k
    return g - g.mean(), loss


# ---------------------------------------------------------------- report ----
def run(cap, W, slots, depths, ranks, lr, block, limit=None, discount_shard=1,
        delay=2, every=6, cap_swaps=96, policy="strata", seed=0):
    nl, n_embd, ne = W.shape
    hit, occ, swaps, resid = cache_residency(cap, slots, delay, every, cap_swaps, policy)
    stats = Stats()
    corr = {(d, r): Correction(nl, n_embd, ne, d, r, lr=lr, seed=seed)
            for d in depths for r in ranks}
    first = {}
    first_score = {}
    KS_ = KS

    def score(pk32, ids_sel, res_sel, phase_mask, arm, d):
        """Every K is a slice of the ranked top-32; prompt, decode and all are
        the two phase masks' sums."""
        for K in KS_:
            k = min(K, pk32.shape[-1])
            for ph, m in (("prompt", phase_mask == capture.PHASE_PROMPT),
                          ("decode", phase_mask == capture.PHASE_DECODE)):
                if m.any():
                    stats.add_sums((arm, d, K, ph),
                                   count_rows(pk32[m][..., :k], ids_sel[m], res_sel[m]))
                else:
                    stats.add_sums((arm, d, K, ph), dict.fromkeys(Stats.TERMS, 0))
            # 'all': the same rows without the phase split
            stats.add_sums((arm, d, K, "all"),
                           count_rows(pk32[..., :k], ids_sel, res_sel))

    def depth_run(d, X, ids, phases, own, start):
        targets = np.arange(d, nl, dtype=np.int32)
        if targets.size == 0:
            first[d] = [int(i) for i in range(min(d, nl))]
            return
        first[d] = [int(i) for i in range(d)]
        src = targets - d
        base = router_logits(X[:, src], W[src])        # (b, nt, ne) reuse logits
        real = own[:, targets]                         # the target layer's own input
        ids_sel = ids[:, targets]                      # (b, nt, n_used)
        res_sel = resid[start:start + X.shape[0]][:, targets]
        base32 = top32(base)
        real32 = top32(real)
        score(base32, ids_sel, res_sel, phases, "A", d)
        score(ids_sel, ids_sel, res_sel, phases, "C", d)
        # B: online correction, one token at a time, in file order
        for r in ranks:
            c = corr[(d, r)]
            arm = "B%d" % r
            for j in range(base.shape[0]):
                t = start + j
                delta = c.predict(X[j], token=t)
                pred = base[j] + delta
                pk32 = top32(pred)
                ph = "prompt" if phases[j] == capture.PHASE_PROMPT else "decode"
                one_ids = ids_sel[j:j + 1]
                one_res = res_sel[j:j + 1]
                for K in KS_:
                    k = min(K, pk32.shape[-1])
                    row = count_rows(pk32[None, :, :k], one_ids, one_res)
                    stats.add_sums((arm, d, K, "all"), row)
                    stats.add_sums((arm, d, K, ph), row)
                if t == 0 and (d, r) not in first_score:
                    # the very first scored token, before any update
                    first_score[(d, r)] = int((pk32[..., :1] ==
                                               ids_sel[j][..., 0][:, None]).any(axis=-1).sum())
                # the score above used the pre-update parameters; only now does
                # this token's label enter the parameters
                c.update(real[j], token=t, orders=real32[j])

    # The depths' logit matmuls run on threads: numpy's matmul releases the GIL,
    # and a reference-BLAS build is single-threaded, so three depths overlap
    # (measured here ~3.4x against the sequential walk). The per-depth state
    # (its corrections, its counters) is separate; the shared arrays are read.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, len(depths))) as ex:
        for start, X, ids, phases in cap.blocks(block):
            own = router_logits(X, W)                     # every layer on its own input
            futs = [ex.submit(depth_run, d, X, ids, phases, own, start) for d in depths]
            for f in futs:
                f.result()
            if limit is not None and start + X.shape[0] >= limit:
                break
    return stats, first, dict(hits=hit, slots_per_layer=occ, swaps=swaps), corr, first_score


def depth_report(stats, first, depths, ranks):
    out = {}
    for d in depths:
        arms = ["A"] + ["B%d" % r for r in ranks] + ["C"]
        out[str(d)] = dict(layers_without_source=first[d],
                           arms={a: {str(K): stats.block(a, d, K) for K in KS} for a in arms})
    return out


def print_table(rep, depths, ranks):
    arms = ["A"] + ["B%d" % r for r in ranks] + ["C"]
    print("depth  arm   K  phase   recall@1   recall_used   @1 non-res   used non-res")
    for d in depths:
        for a in arms:
            for K in KS:
                for phase in ("prompt", "decode"):
                    t = rep[str(d)]["arms"][a][str(K)][phase]
                    fmt = lambda v: "  --  " if v is None else "%6.3f" % v
                    print("%5d  %-4s %2d  %-6s  %s      %s        %s       %s"
                          % (d, a, K, phase, fmt(t["recall_top1"]), fmt(t["recall_used"]),
                             fmt(t["recall_top1_nonres"]), fmt(t["recall_nonres_used"])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture")
    ap.add_argument("gguf")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--slots", type=int, default=5915,
                    help="the cache's slot count over the capture's layers (default 5915)")
    ap.add_argument("--discount-shard", type=int, default=1, help="divide the slot share by K")
    ap.add_argument("--profile", default=None, help="a STRP profile: the engine's slot walk")
    ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--cap", type=int, default=96, help="swaps a plan")
    ap.add_argument("--policy", default="strata", choices=("strata", "lfru", "glfru"))
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gguf-py", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    cap = Capture(a.capture, limit=a.limit)
    h = cap.header
    W = load_routers(a.gguf, h.n_layer, h.n_embd, h.n_expert, a.gguf_py)
    if a.profile:
        slots = profile_layers(a.profile, h.n_layer, h.n_expert, a.slots)
    else:
        slots = flat_slots(h.n_layer, h.n_expert, a.slots, a.discount_shard)
    stats, first, cache, _corr, _fs = run(cap, W, slots, a.depths, a.ranks, a.lr, a.block,
                              limit=a.limit, discount_shard=a.discount_shard,
                              delay=a.delay, every=a.every, cap_swaps=a.cap,
                              policy=a.policy, seed=a.seed)
    rep = dict(
        capture=dict(path=os.path.basename(a.capture), n_layer=h.n_layer,
                     n_embd=h.n_embd, n_expert=h.n_expert, n_used=h.n_used,
                     n_tokens=cap.n_tokens,
                     prompt_tokens=int((cap.phases == capture.PHASE_PROMPT).sum()),
                     decode_tokens=int((cap.phases == capture.PHASE_DECODE).sum())),
        gguf=dict(path=os.path.basename(a.gguf), routers="blk.<il>.ffn_gate_inp.weight",
                  dtype_of_routers="F32 (the conversion keeps ffn_gate_inp) or dequantized"),
        config=dict(depths=a.depths, ranks=a.ranks, Ks=list(KS), slots=a.slots,
                    discount_shard=a.discount_shard, profile=a.profile,
                    policy=a.policy, delay=a.delay, every=a.every, swaps_a_plan=a.cap,
                    lr=a.lr, block=a.block, limit=a.limit, seed=a.seed,
                    loss="centred MSE on the real top-32 logits + hinge below the 32nd - 0.1",
                    order="test-then-train: predict, score, then update"),
        cache=dict(slot_mode=slot_mode(a.profile), slots=a.slots,
                   hit_rate=cache["hits"], slots_per_layer=[int(x) for x in cache["slots_per_layer"]],
                   swaps=cache["swaps"]),
        depths=depth_report(stats, first, a.depths, a.ranks),
        online=dict(order="predict, score, then update (Correction.events pins it)",
                    lr=a.lr, ranks=a.ranks),
    )
    text = json.dumps(rep, indent=1, sort_keys=True)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    print_table(rep["depths"], a.depths, a.ranks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
