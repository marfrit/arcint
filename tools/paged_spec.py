#!/usr/bin/env python3
"""Speculative decoding on the paged path, dense Qwen3.8, MTP head drafting.

The la.* speculative convention, established empirically (stage-2 probes):

    la.cache_interval    = [1]           checkpoint after every token
    la.block_indices     = [c, s0..sk-1] committed row + one scratch per token
    promotion            = use checkpoint row s_i as next step's committed row
    rejection            = reuse row c; the committed row is never written
    attention KV         = rolls back by past_lens arithmetic alone
    fresh rows           = MUST be zeroed: the kernels read the committed row
                           even at past_lens 0 (GenAI zeroes rows for this)

Mechanism invariants, measured bitwise before this was built: a spec pass
computes identical logits to a plain pass over the same tokens; the last
checkpoint equals the in-place state; the committed row is untouched (m=0 is a
strict no-op). NOT deliverable: bitwise equality against a no-spec baseline --
a k-token pass computes bitwise-different state than k 1-token passes (DESIGN
3.2), on this path as on the stateful one. Gates: mechanism invariance,
bitwise determinism across runs, warm-restore equality, non-zero acceptance.
"""
import sys, time
import numpy as np, openvino as ov
from openvino import opset13 as op
from openvino._offline_transformations import paged_attention_transformation as pat

MODEL = "/models/ov/qwen38-b7c1-ov"
DEV = sys.argv[1] if len(sys.argv) > 1 else "GPU.0"
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 512
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 120
KVB, CHUNK, ROWS = 16, 512, 4
VOCAB, HID = 248320, 5120

core = ov.Core(); ctxd = core.get_default_context(DEV)

# On the A770 every gigabyte is contested: the base model must compile FIRST
# (its compile-time peak on top of a resident embeddings model OOMs where the
# reverse order fits), the embeddings go to the CPU (a gather does not need the
# card), and the prefill chunk drops to keep the activation peak inside the
# reservation. The MTP head does not fit at all there -- handled below.
TIGHT = DEV != "GPU.0"
# The embeddings result crosses host memory either way, so the gather might as
# well run where there is room -- next to the head on the other card. Measured:
# 0.38 s per 120 tokens on the CPU against 0.02 on a GPU.
EMB_DEV = "GPU.0" if TIGHT else DEV
# The head does not fit beside the dense model on the A770 (13.59 + 0.97 +
# 1.66 = 15.25 of 15.11 GiB) -- but the box has a second card, and the head is
# a separate graph glued through host memory: per step only a 5120-float hidden
# row and a token embedding cross, ~20 KB each. Weights stay put, activations
# travel. On the B60 the head fits beside the coder production (13.3 + 1.66 +
# 0.97 of 22.7 GiB), so both cards stay productive.
HEAD_DEV = "GPU.0" if TIGHT else DEV
if TIGHT: CHUNK = 256

# ---- paged base model, hidden state exposed for the head -------------------
m = core.read_model(MODEL + "/openvino_language_model.xml")
pat(m)
node = m.get_results()[0].input_value(0).get_node()
for _ in range(8):
    if node.get_type_name() == "MatMul": break
    node = node.input_value(0).get_node()
res = op.result(node.input_value(0)); res.get_output_tensor(0).set_names({"hidden_states"})
m.add_results([res]); m.validate_nodes_and_infer_types()
pg = core.compile_model(m, DEV, {"KV_CACHE_PRECISION": ov.Type.f16})
r = pg.create_infer_request()

emb = core.compile_model(MODEL + "/openvino_text_embeddings_model.xml", EMB_DEV).create_infer_request()
def embed(ids):
    return emb.infer({0: np.array(ids, dtype=np.int64).reshape(1, -1)})[0][0]

BLOCKS = (DEPTH + NEW * 2) // KVB + 8
gdn_t, conv_t = [], []
for i in pg.inputs:
    k, et = i.get_any_name(), i.get_element_type()
    if   k.startswith("conv_state_table."):
        t = ctxd.create_tensor(et, ov.Shape([ROWS, 10240, 4]), {}); r.set_tensor(k, t); conv_t.append(t)
    elif k.startswith("gated_delta_state_table."):
        t = ctxd.create_tensor(et, ov.Shape([ROWS, 48, 128, 128]), {}); r.set_tensor(k, t); gdn_t.append(t)
    elif k.startswith("key_cache."):
        r.set_tensor(k, ctxd.create_tensor(et, ov.Shape([BLOCKS, 4, 256, KVB]), {}))
    elif k.startswith("value_cache."):
        r.set_tensor(k, ctxd.create_tensor(et, ov.Shape([BLOCKS, 4, KVB, 256]), {}))

def fwd(ids_or_emb, past, la_rows, interval, n=None):
    if n is None:
        e = embed(ids_or_emb); n = len(ids_or_emb)
    else:
        e = ids_or_emb
    tot = past + n
    nblk = (tot + KVB - 1) // KVB
    i32 = lambda a: ov.Tensor(np.array(a, dtype=np.int32))
    for k, v in {"inputs_embeds": ov.Tensor(e),
                 "position_ids": ov.Tensor(np.tile(np.arange(past, tot, dtype=np.int64), (4, 1))),
                 "past_lens": i32([past]), "subsequence_begins": i32([0, n]),
                 "block_indices": i32(list(range(nblk))), "block_indices_begins": i32([0, nblk]),
                 "max_context_len": ov.Tensor(np.array(tot, dtype=np.int32)),
                 "la.block_indices": i32(la_rows), "la.block_indices_begins": i32([0, len(la_rows)]),
                 "la.past_lens": i32([past]), "la.cache_interval": i32([interval])}.items():
        r.set_tensor(k, v)
    r.infer()
    lg = np.asarray(r.get_tensor("logits").data).reshape(-1, VOCAB)
    h = np.asarray(r.get_tensor("hidden_states").data).reshape(-1, HID)
    return lg, h

# ---- the MTP head (stateful, one position behind, committed tokens only) ---
class Head:
    def __init__(self):
        self.layer = core.compile_model(MODEL + "/openvino_mtp_layer.xml", HEAD_DEV).create_infer_request()
        self.lm = core.compile_model(MODEL + "/openvino_mtp_lm_head.xml", HEAD_DEV).create_infer_request()
        self.len = 0; self.pos = 0

    def feed(self, hidden, tok_emb, want_draft):
        return self._run(hidden.reshape(1, 1, HID), tok_emb.reshape(1, 1, HID), want_draft)

    def prime(self, hiddens, tok_embs):
        n = hiddens.shape[0]
        if n: self._run(hiddens.reshape(1, n, HID), tok_embs.reshape(1, n, HID), False)

    def _run(self, h, e, want_draft):
        n = h.shape[1]
        mask = np.zeros((1, 1, n, self.len + n), dtype=np.float32)
        for i in range(n):
            mask[0, 0, i, self.len + i + 1:] = -np.inf
        self.layer.infer({"hidden_states": h.astype(np.float32),
                          "input_embeds": e.astype(np.float32),
                          "position_ids": np.arange(self.pos, self.pos + n,
                                                    dtype=np.float32).reshape(1, n),
                          "attention_mask": mask, "beam_idx": np.zeros(1, dtype=np.int32)})
        self.len += n; self.pos += n
        if not want_draft: return -1
        self.lm.set_input_tensor(self.layer.get_output_tensor(0))
        self.lm.infer()
        lg = np.asarray(self.lm.get_output_tensor(0).data).reshape(-1, VOCAB)
        return int(lg[-1].argmax())

def read_rows(row):
    out = []
    for t in gdn_t + conv_t:
        host = ov.Tensor(ov.Type.f16, t.get_shape()); t.copy_to(host)
        out.append(np.asarray(host.data)[row].copy().view(np.uint16))
    return out

def write_row(row, saved):
    for t, data in zip(gdn_t + conv_t, saved):
        host = ov.Tensor(ov.Type.f16, t.get_shape())
        a = np.asarray(host.data); a[:] = 0
        a[row] = data.view(np.float16).reshape(a[row].shape)
        host.copy_to(t)

def zero_tables():
    for t in gdn_t + conv_t:
        z = ov.Tensor(ov.Type.f16, t.get_shape())
        np.asarray(z.data)[:] = 0
        z.copy_to(t)

def generate(prompt, new, use_mtp, committed_row=0, warm=None, keep_snapshot=False):
    """warm = snapshot from an earlier run: skip prefill, restore the committed
    row, re-prime the head from stored hiddens. The prompt KV blocks are
    untouched by decode (it writes only at positions >= past), so they are
    reused as-is -- the paged prefix-cache primitive."""
    c, s0, s1 = committed_row, (committed_row + 1) % ROWS, (committed_row + 2) % ROWS
    head = Head()
    snapshot = None
    if warm is None:
        zero_tables()
        pos = 0; lg = h = None; h_chunks = []
        while pos < len(prompt):                   # absolute-grid chunked prefill
            take = min(CHUNK - (pos % CHUNK) if pos % CHUNK else CHUNK, len(prompt) - pos)
            lg, h = fwd(prompt[pos:pos + take], pos, [c, c], 0)
            if use_mtp: h_chunks.append(h.copy())
            pos += take
        nxt = int(lg[-1].argmax()); h_pend = h[-1].copy()
        h_all = None
        if use_mtp:
            h_all = np.concatenate(h_chunks, axis=0)
            head.prime(h_all[:-1], embed(prompt)[1:])   # (h_t, x_{t+1}), t < P-1
        if keep_snapshot:
            snapshot = {"row": read_rows(c), "nxt": nxt, "h_pend": h_pend.copy(),
                        "h_all": None if h_all is None else h_all.copy()}
    else:
        write_row(c, warm["row"])
        nxt, h_pend = warm["nxt"], warm["h_pend"].copy()
        if use_mtp:
            head.prime(warm["h_all"][:-1], embed(prompt)[1:])
    past = len(prompt)
    out = [nxt]
    proposed = accepted = passes = 0
    t_head = t_base = t_emb = 0.0
    t0 = time.time()
    while len(out) < new:
        if use_mtp:
            t1 = time.time(); e_n = embed([nxt]); t_emb += time.time() - t1
            t1 = time.time()
            d = head.feed(h_pend, e_n, True)            # head consumes nxt, drafts d
            t_head += time.time() - t1
            proposed += 1
            t1 = time.time(); e_d = embed([d]); t_emb += time.time() - t1
            t1 = time.time()
            lg, h = fwd(np.concatenate([e_n, e_d], axis=0), past, [c, s0, s1], 1, n=2)
            t_base += time.time() - t1
            passes += 1
            want = int(lg[0].argmax())
            if want == d:                               # accept: promote s1
                accepted += 1
                t1 = time.time()
                head.feed(h[0], e_d, False)             # head consumes d
                t_head += time.time() - t1
                out.append(d)
                past += 2
                nxt = int(lg[1].argmax()); h_pend = h[1].copy()
                c, s0, s1 = s1, c, s0
            else:                                       # reject: promote s0
                past += 1
                nxt = want; h_pend = h[0].copy()
                c, s0, s1 = s0, c, s1
            out.append(nxt)
        else:
            lg, h = fwd([nxt], past, [c, c], 0)
            passes += 1
            past += 1
            nxt = int(lg[-1].argmax())
            out.append(nxt)
    dt = time.time() - t0
    return out[:new], {"passes": passes, "proposed": proposed, "accepted": accepted,
                       "seconds": dt, "tps": len(out[:new]) / dt, "final_row": c,
                       "snapshot": snapshot, "t_head": t_head, "t_base": t_base,
                       "t_emb": t_emb}

rng = np.random.default_rng(11)
prompt = [int(x) for x in rng.integers(1000, 40000, size=DEPTH)]

off, st_off = generate(prompt, NEW, use_mtp=False)
on1, st_on1 = generate(prompt, NEW, use_mtp=True, keep_snapshot=True)
s_a = read_rows(st_on1["final_row"])       # before the next run scribbles on it
on2, st_on2 = generate(prompt, NEW, use_mtp=True, committed_row=1)
s_b = read_rows(st_on2["final_row"])
on3, st_on3 = generate(prompt, NEW, use_mtp=True, warm=st_on1["snapshot"])
s_c = read_rows(st_on3["final_row"])

acc = 100.0 * st_on1["accepted"] / max(1, st_on1["proposed"])
print(f"== {DEV} (head on {HEAD_DEV}), depth {DEPTH}, {NEW} new tokens ==")
print(f"  mtp OFF : {st_off['tps']:5.1f} t/s  ({st_off['passes']} passes)")
print(f"  mtp ON  : {st_on1['tps']:5.1f} t/s  ({st_on1['passes']} passes, "
      f"accept {acc:.1f}% = {st_on1['accepted']}/{st_on1['proposed']})")
o = st_on1
print(f"            base {o['t_base']:.2f} s, head {o['t_head']:.2f} s, "
      f"embed {o['t_emb']:.2f} s, other {o['seconds']-o['t_base']-o['t_head']-o['t_emb']:.2f} s")
print(f"  gate: deterministic tokens across runs         : {on1 == on2}")
print(f"  gate: deterministic state, bitwise             : "
      f"{all(np.array_equal(a, b) for a, b in zip(s_a, s_b))}")
print(f"  gate: warm restore == cold, tokens             : {on3 == on1}")
print(f"  gate: warm restore == cold, state bitwise      : "
      f"{all(np.array_equal(a, b) for a, b in zip(s_a, s_c))}")
print(f"  gate: acceptance is non-zero                   : {st_on1['accepted'] > 0}")
print(f"  MTP on == MTP off (reported, DESIGN 3.2)       : {on1 == off}")
