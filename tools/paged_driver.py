"""The paged-path prototype, with an honest admission check. The oracle for the
C++ port (DESIGN §3.5/§7.0.3): it proves the la.* convention, device-resident
state tensors, the absolute-grid prefill, and the startup reservation.

The reservation, all measured rather than assumed:

    weights + graph        GPU_MEMORY_STATISTICS (usm_device + cl_mem) after
                           compile -- NOT the .bin size, the graph adds to it
    activation peak        run one prefill chunk and read the delta; it scales
                           linearly with the chunk (measured 2.28 / 1.14 / 0.57 /
                           0.28 GiB at 2048 / 1024 / 512 / 256 on the A770), so
                           the chunk size is the knob that buys context
    GDN state slab         one row per sequence across all state tables
    KV                     ctx x 20 KiB/token (10 layers x 2 x 2 heads x 256 x f16)
    margin                 256 MiB allocator slack, stated, not hidden

Admission is refused with the numbers -- the same shape as the context-overflow
400 -- instead of meeting the same 15.1 GiB at runtime as CL_OUT_OF_RESOURCES
mid-request. Clearing a budget check by not doing the arithmetic is not a
capability; this driver admits configurations GenAI refuses because its peak is
genuinely smaller and it can prove it.

    python3 tools/paged_driver.py GPU.1 49152 512    # admit and serve
    python3 tools/paged_driver.py GPU.1 49152 2048   # refuse, with the numbers

Beware: GPU_MEMORY_STATISTICS lumps usm_host in with device memory if summed
blindly; the plugin also spills some prefill buffers host-side by choice (~0.45
GiB at chunk 512), which must not be counted against the card.
"""
import sys
import numpy as np, openvino as ov
from openvino._offline_transformations import paged_attention_transformation as pat

MODEL = "/models/ov/qwen36-coder-b5-ov"
DEV = sys.argv[1] if len(sys.argv) > 1 else "GPU.1"
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
KV_BLOCK = 16
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
KV_BYTES_PER_TOKEN = 10 * 2 * 2 * 256 * 2      # layers x k/v x kv_heads x head x f16

core = ov.Core()
ctx_dev = core.get_default_context(DEV)
total = core.get_property(DEV, "GPU_DEVICE_TOTAL_MEM_SIZE")

def stats():
    return dict(core.get_property(DEV, "GPU_MEMORY_STATISTICS"))

def resident(st=None):
    # Only what actually occupies the card. usm_host lives in system RAM and
    # spilled allocations must not be counted against device memory.
    st = st if st is not None else stats()
    return sum(v for k, v in st.items() if k in ("usm_device", "cl_mem"))

emb = core.compile_model(MODEL + "/openvino_text_embeddings_model.xml", DEV).create_infer_request()
def embed(ids): return emb.infer({0: np.array(ids, dtype=np.int64).reshape(1, -1)})[0][0]

m = core.read_model(MODEL + "/openvino_language_model.xml"); pat(m)
pg = core.compile_model(m, DEV, {"KV_CACHE_PRECISION": ov.Type.f16})
r = pg.create_infer_request()
st0 = stats()
after_compile = resident(st0)
print(f"  device total                     {total/2**30:7.2f} GiB")
for k, v in sorted(st0.items()):
    if v: print(f"     after compile {k:14s} {v/2**30:7.2f} GiB")
print(f"  DEVICE-resident after compile    {after_compile/2**30:7.2f} GiB   (usm_device + cl_mem)")

# GDN slab: one row per sequence across every state table.
gdn_slab = 0
def dt(shape):
    global gdn_slab
    t = ctx_dev.create_tensor(ov.Type.f16, ov.Shape(shape), {})
    gdn_slab += int(np.prod(shape)) * 2
    return t
for k in (i.get_any_name() for i in pg.inputs):
    if   k.startswith("conv_state_table."):        r.set_tensor(k, dt([1, 8192, 4]))
    elif k.startswith("gated_delta_state_table."): r.set_tensor(k, dt([1, 32, 128, 128]))
print(f"  GDN state slab                   {gdn_slab/2**20:7.1f} MiB   (allocated)")

# Activation peak: run the largest forward this configuration will ever run --
# one prefill chunk -- against a KV pool sized for exactly that, and measure.
probe_blocks = (CHUNK + KV_BLOCK - 1) // KV_BLOCK
kv_names = [i.get_any_name() for i in pg.inputs
            if i.get_any_name().startswith(("key_cache.", "value_cache."))]
for k in kv_names:
    shape = [probe_blocks, 2, 256, KV_BLOCK] if k.startswith("key") else [probe_blocks, 2, KV_BLOCK, 256]
    r.set_tensor(k, ctx_dev.create_tensor(ov.Type.f16, ov.Shape(shape), {}))

def fwd(ids, past):
    n, tot_ = len(ids), past + len(ids)
    nblk = (tot_ + KV_BLOCK - 1) // KV_BLOCK
    i32 = lambda a: ov.Tensor(np.array(a, dtype=np.int32))
    for k, v in {"inputs_embeds": ov.Tensor(embed(ids)),
                 "position_ids": ov.Tensor(np.tile(np.arange(past, tot_, dtype=np.int64), (4, 1))),
                 "past_lens": i32([past]), "subsequence_begins": i32([0, n]),
                 "block_indices": i32(list(range(nblk))), "block_indices_begins": i32([0, nblk]),
                 "max_context_len": ov.Tensor(np.array(tot_, dtype=np.int32)),
                 "la.block_indices": i32([0, 0]), "la.block_indices_begins": i32([0, 2]),
                 "la.past_lens": i32([past]), "la.cache_interval": i32([0])}.items():
        r.set_tensor(k, v)
    r.infer()
    lg = r.get_tensor("logits").data
    return int(lg.reshape(-1, lg.shape[-1])[-1].argmax())

fwd(list(range(1000, 1000 + CHUNK)), 0)
st1 = stats()
peak = resident(st1)
for k in sorted(set(st0) | set(st1)):
    d = st1.get(k, 0) - st0.get(k, 0)
    if d: print(f"     prefill delta {k:14s} {d/2**30:+7.2f} GiB")
probe_kv = probe_blocks * KV_BLOCK * KV_BYTES_PER_TOKEN
activation = peak - after_compile - gdn_slab - probe_kv
print(f"  DEVICE activation peak, chunk {CHUNK} {activation/2**30:6.2f} GiB   (measured)")

# ---- the admission check ---------------------------------------------------
margin = 256 * 2**20                     # allocator slack, deliberately stated
kv_wanted = CTX * KV_BYTES_PER_TOKEN
budget = total - after_compile - activation - gdn_slab - margin
max_ctx = int(budget // KV_BYTES_PER_TOKEN) // KV_BLOCK * KV_BLOCK
print(f"  KV for requested ctx {CTX:<6}      {kv_wanted/2**30:7.2f} GiB")
print(f"  admissible budget                {budget/2**30:7.2f} GiB  -> max ctx {max_ctx}")
if kv_wanted > budget:
    print(f"  REFUSED: ctx {CTX} needs {kv_wanted/2**30:.2f} GiB of KV but "
          f"{budget/2**30:.2f} GiB remain after weights {after_compile/2**30:.2f} + "
          f"activations {activation/2**30:.2f} + state {gdn_slab/2**20:.0f} MiB "
          f"+ margin 0.25; max ctx here is {max_ctx}")
    sys.exit(0)

# admitted: allocate the real pool and prove it serves at depth, prefilling on
# the absolute chunk grid exactly as the engine does
nblk_full = CTX // KV_BLOCK
for k in kv_names:
    shape = [nblk_full, 2, 256, KV_BLOCK] if k.startswith("key") else [nblk_full, 2, KV_BLOCK, 256]
    r.set_tensor(k, ctx_dev.create_tensor(ov.Type.f16, ov.Shape(shape), {}))
import time
DEPTH = min(4096, CTX - 64)
toks = list(np.random.default_rng(7).integers(1000, 40000, size=DEPTH))
pos = 0
t_pf = time.time()
while pos < DEPTH:
    take = min(CHUNK - (pos % CHUNK) if pos % CHUNK else CHUNK, DEPTH - pos)
    t = fwd([int(x) for x in toks[pos:pos + take]], pos)
    pos += take
pf = time.time() - t_pf
t0 = time.time(); n = 40
for i in range(n): t = fwd([t], DEPTH + i)
dec = n / (time.time() - t0)
st2 = stats()
print(f"  ADMITTED: ctx {CTX} at chunk {CHUNK}. prefill {DEPTH} tok in {pf:.1f} s "
      f"({DEPTH/pf:.0f} t/s), decode at depth {DEPTH}: {dec:.1f} t/s")
print(f"  device now {resident(st2)/2**30:.2f} GiB of {total/2**30:.2f}, "
      f"host spill {st2.get('usm_host', 0)/2**30:.2f} GiB")
