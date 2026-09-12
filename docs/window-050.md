# window-050 — the 0.5.0 prediction window, operating manifest

Recorded 2026-09-12. This file is the manifest a window operator executes: the
holds, the stop order, where the logs go, which suites run on which device with
which compile config, and the PREDICTION TEMPLATE that must be filled in BEFORE
the serving measurement is taken.

**Every command carries a state marker, and the markers are the point:**

| marker | meaning |
|---|---|
| `RUN@<sha>` | executed on the dev host, on the tree named by `<sha>`, and its output is recorded here or in RECONCILE |
| `RUN@wt+<sha>` | executed on a WORKING TREE at `<sha>` carrying uncommitted deltas — the code is not provably `<sha>`; RECONCILE names the delta |
| `RUN@unrecorded` | executed, but the tree it ran on was never recorded and cannot now be established |
| `DRY` | the command is correct and was exercised in a no-op / device-free form, but its real effect was not produced |
| `UNTESTED` | written down from the design and never executed — treat every claim about it as a guess |

A command with no marker is a defect in this file. No command here is marked
`RUN` unless this repository holds the output.

### CF-MANIFESTSHA — the law added 2026-09-12 (REVIEW e78812d, F7)

**A `RUN` marker must name the commit its output came from.** The review's
words: the manifest recorded *"CPU-only control, for the same tree (`RUN`,
2026-09-12): 101 passed with shards, 33 passed / 59 skipped device-free"*, while
at the tip those figures were 112 and 42/70. *"Both figures are true of their
own trees and a reader cannot tell them apart — which is the b929924 gap one
level up, in the tracked operating document."* This file already applied that
rule to the reserved coherence row ("in the same commit as the measurement") and
not to its own 38 `RUN` markers.

Two conventions, so the law is applicable rather than aspirational:

* For a measurement **of the tree** (a suite result, a node count, a parity
  figure), the id is the tree the measurement ran on.
* For an observation that does **not** depend on the tree (host state, service
  state, card enumeration, a plugin property), the id is the **session tip** at
  which it was recorded — it dates the observation without claiming the code
  caused it.

`tests/python/test_window_manifest.py` enforces it: every `RUN` in this file
must carry an id, and the number of `RUN@unrecorded` markers is a **ratchet**
that may only ever go down.

---

## 0. What this window is, and what it is not

The window measures **arcint serving Flash-Next on the reserved A770**. It is
not the export: the export is device-free and is gated by the Python suites and
the C++ ladder, both of which run without a card. What needs a card is the
serving prediction and the GPU parity columns.

Two things the window does **not** unblock, stated here so nobody schedules it
expecting them (the refusal in `tools/export_qwen4_exp.py` enumerates all three
blockers and names the residency assumption):

1. **No full-size config exists.** `build_backbone_ir` has only
   `_tiny_config()`; nothing translates `geometry` into a `Qwen4ExpTextConfig`.
   Code-level, device-free.
2. **Residency at f32.** Measured, not asserted — see §5. A weight strategy is
   required and a bigger card is not one.

---

## 1. Holds — the GPU host must not go to sleep mid-window

The GPU host (and with it the dev container and both cards) is shut down
nightly by a cron on the fleet's power-control host unless a lock says
otherwise. The lock is self-expiring and never overwrites a running foreign
lock.

```
# RUN@e78812d  — place a hold for the window's length plus slack
ssh <power-host> 'sudo <hold-script> <hours> "arcint dev"'
# RUN@e78812d  — check remaining time
ssh <power-host> 'sudo <hold-script> status'
# UNTESTED — release early (not exercised; the lock was left to expire)
ssh <power-host> 'sudo <hold-script> release'
```

Observed output shape (`RUN@e78812d`, 2026-09-12):

```
gesperrt bis 12.09. 06:00 (noch 5 h 0 min)  pid=4015039 owner=hold-data grund=arcint dev
```

If the GPU host is off, wake it via its smart plug (`UNTESTED` this session —
the host was already up):

```
# UNTESTED
ssh <power-host> 'sudo <plug-switch-script> <plug-id> on'
```

**If the GPU host freezes or dies silently** — GPU experiments can do that — its
kernel log survives elsewhere; the local journal dies with the box. Check this
FIRST after a freeze:

```
# UNTESTED this session (no freeze occurred)
ssh <power-host> 'sudo tail -100 <netconsole-capture>'
```

---

## 2. Service stop order — the cards are FULL while the services run

Both resident units hold their model VRAM permanently. Any process that wants a
card must stop the resident service first: loading next to it fails with
allocation errors at best, host OOM at worst.

**Verify the mapping yourself before trusting it** — it has drifted once without
anyone noticing until a benchmark caught it by accident:

```
# RUN@e78812d
ssh <dev-host> 'systemctl --user list-units | grep -i arcint'
```

`RUN@e78812d` output, 2026-09-12:

```
arcint-agent.service   active running   Qwen3.8-27B dicht + MTP, GPU.0 (Arc Pro B60), :8087
arcint.service         active running   Qwen3.6-27B-A3B-Coder B5, GPU.1 (Arc A770), :8080
```

So: **GPU.0 = B60 = `arcint-agent` = :8087** and **GPU.1 = A770 = `arcint` =
:8080**. `llama-agent.service` does not exist — do not reference it in a
stop/restore command. `openarc-coder.service` is the retired, disabled rollback
path and stays stopped (reservation mneme 361).

### Stop order, with timestamps recorded

```
# RUN@e78812d
date -Is
systemctl --user stop arcint-agent        # frees GPU.0 (B60)
date -Is
systemctl --user stop arcint              # frees GPU.1 (A770)
date -Is
systemctl --user is-active arcint-agent arcint
```

`RUN@e78812d`, 2026-09-12: agent stopped `23:54:40Z`, coder stopped `23:54:41Z`, both
report `inactive`, host RAM in use fell 18 GiB → 0 GiB.

### Take a coherence baseline BEFORE stopping

Needed because the restore probe is only meaningful against a before-figure, and
because the agent is a **reasoning** model: a 12-token budget is consumed
entirely by `reasoning_content` and returns an EMPTY `content`, which looks like
a broken endpoint and is not one.

```
# RUN@e78812d  — max_tokens must be >= ~64; 200 used here
curl -s http://127.0.0.1:8087/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-agent","messages":[{"role":"user",
       "content":"What is the capital of France? Answer in one word."}],
       "max_tokens":200,"temperature":0}'
```

`RUN@e78812d` baseline, 2026-09-12 23:54Z: `content='Paris'`, 27 completion tokens, MTP
`accepted_prediction_tokens=13 rejected_prediction_tokens=1`.

### Card enumeration once the cards are free

```
# RUN@e78812d
ssh <dev-host> '<venv>/bin/python3 -c "
import openvino as ov; c=ov.Core()
print(c.available_devices)
for d in c.available_devices:
    if d.startswith(\"GPU\"): print(d, c.get_property(d,\"GPU_DEVICE_TOTAL_MEM_SIZE\"))"'
```

`RUN@e78812d` output, 2026-09-12:

```
OV 2026.4.0-22849-71640275d29   devices ['CPU', 'GPU.0', 'GPU.1']
GPU.0 Intel(R) Arc(TM) Pro B60 Graphics (dGPU)  24385683456  = 22.71 GiB
GPU.1 Intel(R) Arc(TM) A770 Graphics (dGPU)     16225243136  = 15.11 GiB
OPTIMIZATION_CAPABILITIES (both): FP32, BIN, FP16, INT8, GPU_HW_MATMUL,
                                 GPU_USM_MEMORY, EXPORT_IMPORT
INFERENCE_PRECISION_HINT default (both): float16      <-- see §3
```

---

## 3. `INFERENCE_PRECISION_HINT` — wire it or the parity columns are fiction

**The Intel GPU plugin defaults `INFERENCE_PRECISION_HINT` to `float16`.** CPU
defaults to `float32`. Measured (`RUN@829a213`, OV 2026.4.0, both Arc cards). So a
`compile_model(model, "GPU.0")` with no config executes the graph in f16 while
the suite's assertions compare it against an f64 reference at a floor of ~1e-7.
f16 carries about three decimal digits. That alone fails every GPU parity leg,
independently of any plugin defect, and it must be eliminated before a transform
or a kernel is blamed.

Until 2026-09-12 no q4e suite passed any compile config — six files, eleven call
sites, all bare. The earlier GPU.1 window's "ALL 22 legs fail" result predates
the wiring and its root-cause list should be re-read with this in hand.

`tests/python/q4e_device.py` now owns the decision:

```python
GPU_INFERENCE_PRECISION = "f32"      # "float32" is REJECTED by the plugin
def compile_for(core, model, device):
    cfg = {"INFERENCE_PRECISION_HINT": "f32"} if str(device).upper().startswith("GPU") else {}
    return core.compile_model(model, device, cfg)
```

Accepted value forms, measured (`RUN@829a213`):

```
accepted 'f32'             -> <Type: 'float32'>
REJECTED 'float32'         -> Wrong value float32 for property key
                              INFERENCE_PRECISION_HINT.
                              Supported values: bf16, f16, f32, undefined
accepted ov.Type.f32       -> <Type: 'float32'>
```

Every GPU leg PRINTS the precision the compiled model reports, and the attention
piece ASSERTS it contains f32 before reading any floor — a leg cannot claim f32
while the plugin ran f16.

---

## 4. Suites, per device — TEE'd, one process per file

**Fresh process per test file, always.** A GPU fault at one cell poisons every
later cell in the same process; a past "over 2,048" claim was retracted for
exactly that. `gpu_window.sh` enforces it and sweeps for SIGTERM-ignoring
leftovers between files (GPU test processes have survived SIGTERM before, and two
arcint-class processes on one card wedge under load).

```
# RUN@wt+2e99661  — the driver, from a byte-exact staged tree
ssh <dev-host> 'cd <staged-tree> && TREE=<staged-tree> LOG=<staged-tree>/logs ./gpu_window.sh'
```

Per-file invocation it issues:

```
# RUN@wt+2e99661
Q4E_GPU=GPU.0,GPU.1 Q4E_GGUF_SHARDS=<shards> \
  <venv>/bin/python3 -m pytest <file> -q -s --tb=short
```

CPU-only control (`RUN@e78812d`): `Q4E_GPU=` empty → **112 passed** with
shards, **42 passed / 70 skipped, 0 errors** device-free.

> The figures here were **101** and **33/59** until 2026-09-12. Both were
> true — of the tree they ran on, which was not the tip. That is the
> defect CF-MANIFESTSHA exists to prevent, recorded rather than edited
> away; §8's own numbers moved for the same reason.

### GPU RESULTS — `RUN@be57428` 2026-09-12, both cards, f32 pinned

Suite, one process per file, `Q4E_GPU=GPU.0,GPU.1`, after the MOE-GPU-FUSION
fix (§4.3):

| file | `RUN@wt+2e99661` | `RUN@be57428` | note |
|---|---|---|---|
| `test_hc_block.py` | 13 passed | **13 passed** | green on both cards |
| `test_hc_combine_block.py` | 13 passed | **13 passed** | green on both cards |
| `test_ple_block.py` | 16 passed | **16 passed** | green on both cards |
| `test_attention_piece.py` | 15 passed | **19 passed** | REAL width; +CF-BOUNDS, +CF-ROPEAB |
| `test_moe_chunk_partition.py` | — | **16 passed** | new (CF-CHUNKCOV), 3 geometries |
| `test_moe_block.py` | 10 failed, 6 passed | **2 failed, 14 passed** | see below |
| `test_backbone.py` | 8 failed, 6 passed | **2 failed, 12 passed** | remaining 2 are the GDN T=96 root |
| `test_gdn_block.py` | 2 failed, 5 passed | 4 failed, 9 passed | §4.2 — more shapes, same defect |

Five whole files green on both cards, including the real-width dense-causal
attention piece and the real-weights MoE chunk partition.

The two remaining `test_moe_block.py` GPU failures are
`test_moe_row_locality` on both cards. That cell **could not fail before**: the
model did not compile, so the row-locality property was never evaluated on a
card at all. It is a NEW finding surfaced by the fix, carried forward, not
fixed here — and it is the honest cost of the fix being real.

`test_gdn_block.py` goes from 2 failed to 4 failed because §4.2's doctrine
added the T=65/T=66 boundary pair; the number of DEFECTS is unchanged and the
number of shapes that see it went up.

#### 4.1 The attention piece on GPU — the real-width piece that runs

| dev | T=64 \|ov−pin64\| | ratio | T=96 \|ov−pin64\| | ratio | compile | prec |
|---|---|---|---|---|---|---|
| CPU | 1.855e-07 | 2.6× | 1.535e-07 | 1.0× | 0.10 s | float32 |
| GPU.0 B60 | 1.631e-07 | 2.3× | 1.777e-07 | 1.1× | 0.11 s | float32 |
| GPU.1 A770 | 1.620e-07 | 2.3× | 2.475e-07 | 1.6× | 0.25 s | float32 |

Real width (H=2560, heads 24, kv 2, head_dim 256, rotary 64), real fed GGUF
tensors, 150 nodes, 0.186 GiB of constants, against the pin WITH its real QSA
indexer. Gate is 20× the pin's own f32-vs-f64 rounding; the worst card leg is
2.3×. Every leg printed and asserted `float32`.

#### 4.2 GDN on GPU — localised to row 65, and the acceptance doctrine decided

`RUN@be57428`, 2026-09-12. The failure survives; what changed is that it is now
localised to a row and a shape, and that the GPU acceptance question is settled
by measurement instead of left open.

**THE DOCTRINE: distance-to-truth, gated at 20x the reference's own f32
rounding.** BOTH sides were measured against an f64 recomputation from the same
weights, on CPU and both cards:

```
device  T     |ov-r64|    |r32-r64|    |ov-r32|   ov/floor
CPU     64   5.7251e-07  4.4179e-07  5.6624e-07       1.30
CPU     96   8.3250e-07  9.4986e-07  9.3132e-07       0.88
CPU    128   7.5987e-07  8.0206e-07  9.9838e-07       0.95
CPU    256   1.6980e-06  9.1336e-07  1.4603e-06       1.86
GPU.0   64   7.1151e-07  4.4179e-07  6.8545e-07       1.61
GPU.0   96   7.4504e-02  9.4986e-07  7.4504e-02   78437.26
GPU.0  128   7.5063e-02  8.0206e-07  7.5063e-02   93588.29
GPU.0  256   1.2177e-01  9.1336e-07  1.2177e-01  133325.58
GPU.1   96   7.4505e-02  9.4986e-07  7.4505e-02   78437.33
```

Three measured reasons, not a preference:

1. **The f32 reference is sound** — 4.42e-07 to 9.50e-07 from f64 at every T,
   on every device (it is torch on CPU, so device-independent). That is what
   makes a relative gate possible: the denominator is trustworthy.
2. **Where the defect lives the two criteria are indistinguishable** —
   `|ov-r32|` and `|ov-r64|` agree to four significant figures at T ≥ 96,
   because the error dwarfs both floors. The defect cannot decide the question.
3. **Where they differ, the absolute gate is the unanchored one.** The floor
   itself MOVES with T (4.42e-07 → 9.50e-07), so the old flat `< 1e-5` calls a
   result 20× the floor and one 10× the floor the same thing.

Wired into `test_gdn_block.py`: `|ov − ref_f64| ≤ 20 × |ref_f32 − ref_f64|`,
plus a second assert that fails outright if the f32 reference leaves the floor.
CPU 0.94×, GPU.0 T=64 0.53×, GPU.1 T=64 0.39× — all pass.

**THE LOCALISATION.** Per-row on GPU.0 at T=96 and T=128: row 63 → 2.6e-07,
**row 64 → 1.3e-07 — both at the float floor** — first bad row **65**, and
`n_bad = T − 65` exactly. The T-sweep pins the threshold:

```
T    pad   max-abs      first bad row   n bad     (GPU.0; CPU clean at every T)
64     0   6.8545e-07        -1            0
65    63   1.1735e-06        -1            0
66    62   3.6024e-02        65            1
67    61   2.5580e-02        65            2
68    60   4.6185e-02        65            3
80    48   4.5380e-02        65           15
96    32   7.4504e-02        65           31
```

Reproduced on GPU.1 through the wired gate: T=65 ratio 0.51× (clean), T=66
18717× with exactly one bad row at index 65.

So the second chunk's **first** row is correct on both cards, and the corruption
begins at the first row that mixes the carried state with the in-chunk
accumulation. **That refines rather than confirms the earlier reading**: the
2026-09-12 record named "the first inter-chunk state carry" as the divergence
point; row 64 at 1.3e-07 says the carry ARRIVES correct. The op responsible is
still **NOT identified** and this item stays **OPEN**. What is established: the
row (65), the shape threshold (a non-first chunk holding a second live row),
determinism across both cards, and CPU at the float floor at every shape.

T=65 and T=66 are now parametrised into the suite, so the boundary pair is a
standing gate rather than a note in a document.

#### 4.3 MOE-GPU-FUSION — **FIXED**, wired, with a before/after

`RUN@be57428`, 2026-09-12. Neither of the two candidate shapes this manifest
recorded on 2026-09-12 (`RUN@wt+2e99661`: split the router into two models, or
keep the router on CPU) turned out to be needed.

The fusion is not triggered by "a router". It is triggered by **this** router.
The q4e emitter built the dense `[T, E]` gate with the pin's one-hot construct
(`one_hot → multiply → reduce_sum`); the production 35B-A3B export builds the
same gate with `ScatterElementsUpdate` (`tools/export_mtp.py:472-494`). The
plugin's matcher fires on the former and then fails to register its own
primitive:

```
program_builder.cpp:268  Input moerouterfused:MoERouterFused_1152.out1
                         hasn't been found in primitive_ids map
```

The 22-node reproducer, re-run first, splits the two cleanly:

| variant | nodes | CPU | GPU.0 | GPU.1 |
|---|---|---|---|---|
| `q4e_router` (one_hot) | 22 | OK sum=64.0000 | **FAIL** MoERouterFused | **FAIL** |
| `tiled_router` (scatter) | 21 | OK sum=64.0000 | **OK** 0.63 s | **OK** 0.53 s |
| softmax only | 7 | OK | OK | OK |
| topk only | 8 | OK | OK | OK |

**The swap costs nothing, measured rather than argued:**

```
EQUIVALENCE on CPU  |one_hot-router - scatter-router| = 0.000000e+00
```

FULL MoE BLOCK, one variable (`q4e/moe.py::_router_gate`), same weights, same
input:

| device | BEFORE one_hot (422 nodes) | AFTER scatter (421 nodes) |
|---|---|---|
| CPU | OK | OK, \|dev−CPU\| 0.000000e+00 |
| GPU.0 | **FAIL** MoERouterFused | **OK**, \|dev−CPU\| 4.023314e-07 |
| GPU.1 | **FAIL** MoERouterFused | **OK**, \|dev−CPU\| 4.451722e-07 |

Pin fidelity is untouched: both forms scatter `torch.topk`'s OWN indices, so
the selection reproduces the pin exactly, tie-break included. What changed is
the ops that carry the scatter — and the 0.0 is the proof, not the claim.

**A SECOND, DIFFERENT MoE BLOCKER, found behind the first** (`RUN@be57428`):
the full TILED block with u4 compressed expert bodies compiles on CPU (85
nodes) and fails on both cards at a different site:

```
compile_graph.cpp:54  [GPU] Failed to select implementation for
    name:fullyconnectedcompressed:MatMul_82  type: fully_connected
    original_type: FullyConnectedCompressed
    could not create a primitive descriptor for the matmul primitive
```

So the router was the first gate and the compressed-weight matmul is the next
one. `UNTESTED`: whether a different group size, a u8 declaration, or a
scale/zero-point layout change clears it. That is the next MoE item, and it is
the one that stands between the serving-shape IR and a card.

### MoE legs and MOE-GPU-FUSION — status, superseded text kept

Until `RUN@be57428` this section read "**carried forward, not fixed**" and
listed three `UNTESTED` items: whether pinning f32 alone changes the outcome
(it does not — the fusion failure is structural), whether a plugin key disables
the transform (no such key exists; the full `SUPPORTED_PROPERTIES` was
enumerated), and whether a graph-shape variant sidesteps the matcher. **The
third one was the answer**, and it is now wired and measured above rather than
hypothesised.

A documented workaround that needs a test change is a cell, not a defeat. A
workaround asserted without a measured before/after is neither. This one has
its before/after, on both cards, with a 0.0 equivalence leg behind it.

---

## 4.4 THE BOOT — the serving-shape IR on the reserved card

`RUN@198b736` (the IR) / `RUN@be57428` (the cards), 2026-09-12. No prediction
was written for this and none is claimed: the job was that the path either
lights up or its first failure is named exactly.

**IT LIT UP.** The serving-shape IR — real geometry (H=2560, E=512, I=640,
vocab 248,320), expert bodies slot-referenced as rank-4 u4 constants over
sparse pages, PLE fed by `ngram_row_ids [1,T,16] i64`, dense-causal attention,
the tiled MoE lowering — compiles AND runs a forward on both cards:

```
layers  device  nodes  declared GiB  arena KiB  build s  compile s  infer s  peak host GiB
     1  CPU      2290          7.49          0     3.14       5.17    0.619          28.50
     1  GPU.1    2290          7.49          0     2.85      14.02    0.400           7.63
     1  GPU.0    2290          7.49          0     3.57      10.48    0.629           7.57
     2  CPU      4670         58.03          0     2.94       9.22    1.160          39.84
     2  GPU.1    4670         58.03          0     ~3.5       FAIL      ---           5.49
     2  GPU.0    4670         58.03          0     3.51       FAIL      ---           5.57
```

Every leg that compiled returned `logits (1, 8, 248320)`, `finite=True`, f32
pinned.

### The 2-layer failure, named exactly — and it is a NEW serving constraint

```
engine.cpp:319  [GPU] Exceeded max size of memory object allocation:
                requested 25600122880 bytes, but max alloc size supported
                by device is 4294959104 bytes   (GPU.1, A770)
                                    24385683456 bytes   (GPU.0, B60)
                Please try to reduce batch size, use lower precision, or set
                ov::intel_gpu::hint::enable_large_allocations config property
                to true.
```

25,600,122,880 B is the PLE n-gram table exactly: 320,001,536 × 160 nibbles.
Layer index 1 is the PLE layer, so it enters at the 2-layer step and nowhere
earlier — which is why 1 layer boots on both cards and 2 does not.

**THE A770 CAPS A SINGLE MEMORY OBJECT AT 4,294,959,104 B ≈ 4.00 GiB**, not at
its 15.11 GiB of VRAM. The B60's cap is its whole 24,385,683,456 B. That is a
per-OBJECT limit, distinct from the per-card capacity §5 measures, and it has
not appeared in this repository before. It does not bite the CARD tier's
largest single tensor today (`embed_tokens` / `lm_head` at 248,320 × 2,560 f32
= 2.37 GiB each, comfortably under 4 GiB), but any strategy that consolidates
weights into one large object on GPU.1 has a 4 GiB ceiling, and a quantised
n-gram table must be chunked or host-resident on either card regardless.

`UNTESTED`: whether `ov::intel_gpu::hint::enable_large_allocations` lifts the
A770's 4 GiB cap, and at what cost. The plugin names the lever; nothing here
has pulled it.

Two things to read off this and NOT more than these:

* **`absmax = 0.0000e+00` on every leg, and that is correct.** Every weight is
  an unwritten page. This is a STRUCTURE artifact: it proves the graph
  compiles, allocates, schedules and produces a finite tensor of the right
  shape on the reserved card. It proves nothing whatsoever about numerics, and
  no parity claim may be built on it. The numeric gates live in the per-piece
  suites, on real fed tensors.
* **The GPU path costs 7.6 GiB of host memory where the CPU path costs 28.5.**
  The CPU plugin expands the u4 constants host-side; the GPU plugin does not.
  That is a 3.7x difference in host residency for the same graph, measured, and
  it matters for the offload budget — but it is one shape at one layer count
  and is not yet a serving figure.

The 2-layer step is where the PLE n-gram table enters (declared 7.49 -> 58.03
GiB, because layer index 1 is the PLE layer), and the CPU leg pays 39.84 GiB of
peak host for it. That is the first quantitative sign of what §5's HOST-MMAP
tier costs when it is carried as a graph constant instead of an mmap — which is
exactly why serving reads it through `src/exec/ngram_table.h` instead.

### Boot sequence as executed

```
# RUN@be57428
date -Is                                       # 06:03:10Z
systemctl --user stop arcint-agent             # frees GPU.0 (B60)
systemctl --user stop arcint                   # frees GPU.1 (A770)
systemctl --user is-active arcint-agent arcint # inactive inactive
<venv>/bin/python -c "<card enumeration>"      # GPU.0 22.71 GiB, GPU.1 15.11 GiB
# one process per leg, never two on a card at once
Q4E_GPU= <venv>/bin/python boot.py <layers> <device> <T>
```

### What the boot did NOT do, stated plainly

* **The slot pool was not pointed at the real shards.** The expert bodies are
  declared u4 and unfilled; nothing yet maps the shipped Q3_K_XL expert rows
  into that layout with scales and zero-points. So there is no offload-policy
  log, no miss-tier rate and no slot-churn figure from this window — those need
  weights, not a card. `UNTESTED`.
* **No generation request was issued.** A forward on zero weights produces zero
  logits; a decode loop over them would be a ceremony, not a measurement.
* **The full 48-layer stack was not compiled on a card.** At 48 layers the
  declared constants are 183.07 GiB and the 2-layer CPU leg already peaks at
  39.84 GiB of host. The structure builds (§4.4 above, 84,372 nodes, 6.43 s);
  compiling it needs the quantised fill and the host-mmap tier, not a bigger
  card. `UNTESTED`.
* **The paged port contract is not emitted** — 13 ports, each named with its
  feed site, carried as a strict xfail in
  `tests/python/test_serving_shape.py`.

## 4.5 THE FRONTIER GPU PASS — what the next window runs, in order

`RUN@5663a44` for the CPU preparation below; every GPU row is `UNTESTED` and
belongs to the frontier. This section exists because the 0.5.0 window is now
ONE pass: the emitter-side work is done, and each item here is a prepared
experiment with a reproducer, an expected outcome and a stated first step —
not a topic. An item with no reproducer does not belong on this list.

Run them in this order. Each earlier item is a control for the ones after it.

| # | item | reproducer | expected | if it fails |
|---|---|---|---|---|
| 1 | **1-layer boot, then depth / 4 GiB chunking** | §4.4's boot sequence at `--layers 1`, then 2, then deeper | the A770 refuses a 25,600,122,880 B object (cap 4,294,959,104 B); the B60 accepts it. Start at ONE layer: it is the cheapest shape that can carry the object-cap refusal, and a depth that boots is a control for the depth that does not | chunk the PLE table, or pull `ov::intel_gpu::hint::enable_large_allocations` and record the cost — the lever is named in §4.4 and has never been pulled |
| 2 | **u4 compressed selection (the fill)** | `tools/repro_fc_compressed_selection.py` | `production_2d` selects a compressed primitive on both cards; `as_emitted` is the open question. **Detect on `MOECompressed` / `GatherMatmulCompressed` / `moe_3gemm_fused_compressed`, NOT `FullyConnectedCompressed`, and record the three gate terms** — see §4.5.1's RE-AIMED block | §4.5.1 below — the candidates are built to isolate it |
| 3 | **scatter router legs** | `tests/python/test_moe_block.py` on GPU.0/GPU.1 | scatter passes on both cards, `\|dev−CPU\| ≈ 4e−07` (`RUN@be57428` §4.3) | a regression in the swap, not a new question |
| 4 | **GDN row 65 — FIXED IN THE EMITTER** | `tests/python/test_gdn_block.py` on both cards; `Q4E_GDN_UT_MODE=batched` reproduces the defect on demand | ~~first bad row 65 at every T ≥ 66, both cards (`RUN@be57428` §4.2)~~ → **green at every T with the default `perchunk` emission** (`RUN@61bd61a`, GPU.1: T=96/128/192/256 at 1.0–1.3× the per-run f32 floor, 0 bad rows; backbone T=96 1.038e-04 → 5.960e-08). Item 4's job is now to confirm it on **GPU.0** as well, which this seat did not run | if GPU.0 disagrees with GPU.1, the fix is card-specific and the emitter default must go back behind a device check |
| 5 | **KLD gate vs the llama-fork reference, BOTH context regimes** | `tools/kld_harness.py` on the booted artifact, at a T **below** and a T **above** the 2051 boundary | mean per-token KL(P_ref‖P_cand) ≤ 0.0599 nats at both. The boundary is not decorative: the QSA→dense price is exactly 0.0 for T ≤ 2051 and non-zero above it (`RUN@692c0a6`, §8), so a gate run only below it has not exercised the dense rows at all | a KLD that passes below 2051 and fails above localises to the QSA→dense seam, which is the one place the price is known to change |
| 6 | **MoE compile at short T, under the real plugin** | `tools/repro_moe_compile_short_T.py <T> GPU.N` | the CPU plugin dies on SIGSEGV at T=6 and T=8 and nowhere else; whether the CARD's plugin shares the cliff is unknown | if the card refuses the same two shapes, a short prefill is a serving constraint, not a curiosity |

Item 6 is run LAST on each card because it may take the process down.

### 4.5.2 THE GDN ROW-65 DEFECT IS FIXED IN THE EMITTER (`RUN@61bd61a`, 2026-09-12)

The multi-chunk GDN corruption that item 4 was written to re-confirm is **gone
on GPU.1**, fixed in `tools/q4e/gdn.py` with no OpenVINO change. What changed is
the SHAPE the ops see, never the arithmetic: the chunk axis is no longer a
tensor axis at all but a Python loop, so no emitted op carries a live chunk
axis. `ut_mode` / `Q4E_GDN_UT_MODE` selects among four emissions of the same
algebra, which agree **bit-identically on CPU** (0.00e+00).

| T | C | batched (RED) | perchunk (GREEN) | floor | perchunk ratio |
|---|---|---|---|---|---|
| 96 | 2 | 9.7893e-02 | **8.9964e-07** | 7.065e-07 | 1.3× |
| 128 | 2 | 9.7893e-02 | **1.1901e-06** | 1.183e-06 | 1.0× |
| 192 | 3 | 9.7893e-02 | **1.1901e-06** | 1.183e-06 | 1.0× |
| 256 | 4 | 9.9173e-02 | **1.1901e-06** | 1.183e-06 | 1.0× |

**THE ATTRIBUTION IN `~/win-050/FINDINGS` IS WRONG, and the upstream draft must
not be filed as written.** The spike's own hypothesis was wrong in the same
direction, which is how it was caught. De-batching *only* the unrolled
triangular solve — `debatched`, which nearly doubles the node count and so
demonstrably changes the graph — reproduces the corruption **bit-identically**
(9.7893e-02, 31 rows, first row 65). And the unroll at rank 5 with the chunk
axis live is **clean in isolation on the card** (5.520e-08, both slices). No
isolated op reproduces the fault: not the rank-5 matmul, not `cumsum`, not the
pairwise decay. It needs the whole rank-5 region in one graph, so it is a
fusion/graph-context effect rather than a per-op miscompile.

The drafted one-liner — *"slice index 0 is always correct and every other slice
is deterministically wrong; batch=1 is always correct"* — is also falsified by
the evidence already in hand: at T=64, `HV=4`, so the solve is **already batched
four wide** and every slice is correct. Batch is not 1 in the clean case. The
axis that matters is the chunk axis specifically.

**What the fix costs**, and why it is not the serving answer: one ~1,900-op
unroll per chunk, linear in C where the batched form was nearly flat.

| T | C | batched | perchunk | ×36 GDN blocks (48-layer stack) |
|---|---|---|---|---|
| 256 | 4 | 2,224 | 7,750 | 80,064 → **279,000** |
| 2048 | 32 | 3,680 | 61,062 | 132,480 → **2,198,232** |

So multi-chunk **static** prefill is now correct at the shapes this window
boots, and is still not the route at serving prefill lengths — that remains the
chunked stateful-prefill increment. Mitigation ladder unchanged in destination,
changed in starting point: the `T ≤ 64` restriction is lifted.

Not run by this seat, and therefore open: **GPU.0** (B60) confirmation, and any
shape with C > 4. Item 5
needs item 1 to have produced a booted artifact; if item 1 does not boot, item
5 does not run and says so rather than being run at a reduced geometry whose
number would not transfer (§7.2).

On item 6's sweep width: the reviewer seat widened it from 15 values to 41
(contiguous 1..34 plus 36/40/48/56/64/96/128, fresh process per T) on the dev
host's CPU, 2026-09-12, and found T=6 and T=8 and **no third hit length** —
RE-REVIEW §Priority 5, a dated record, not a live claim of this document. The
card's own sweep is the window's to take.

### 4.5.1 The u4 compressed-selection experiment, prepared

**THE QUESTION.** `q4e.expert_fill` now puts real Q3_K_XL rows into the
serving-shape IR's u4 expert bodies. The residency story rests on those weights
STAYING u4 once the card compiles the graph: if the plugin decompresses them to
f16 at compile time, the declared 4-bit slice is a 16-bit one and every row of
§5's ledger is out by 4×. Two sub-questions, and the second is the one this
repository has already been bitten by:

* **Q1 SELECTION** — does a compressed primitive appear in the runtime graph at
  all (`FullyConnectedCompressed`, `MOECompressed`, the 3GEMM MoE fusion), or
  does it arrive as plain MatMuls over decompressed weights?

  **RE-AIMED 2026-09-12, from the plugin's own pipeline.** Q1 is the right
  question — it is worth 4× on every row of §5's ledger — but it was pointed at
  the wrong primitive, and a window that greps for `FullyConnectedCompressed`
  on the expert bodies will record a false negative. Dated read of the pinned
  plugin tree (dev host, `~/ovsrc-pkg`, build 2026.4.0-22849;
  `src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp`, a FOREIGN
  tree this suite cannot regenerate — recorded in the `FLEET_IRS_*` manner):

      :645  // MOE: TiledMoeBlock -> GatherMatmuls(compressed)
            //      -> MoeOp(compressed) -> MoeOpWithRouting(compressed).
            // Gated on supports_immad (systolic-only) and oneDNN
            // (required for expert GEMM dispatch).
      :647  if (device_info.supports_immad && config.get_use_onednn()
            && !config.get_moe_disable_fusion()) {
      :648      const std::vector<ov::element::Type>
                supported_compressed_weights_types{u4, i4, i8, u8};
      :654      ConvertTiledMoeBlockToGatherMatmuls(...)
      :658      ConvertGatherMatmulToGatherMatmulCompressed({f32, f16}, ...)
      :661      FuseMoERouter / MoeOpFusion

  Two consequences the window must carry. **(a)** The expert bodies never reach
  `FullyConnectedCompressed` — that is the projection head's primitive. They
  reach `GatherMatmulCompressed` and then the MoE op, which is what the served
  35B's A770 graph actually carries (`MOECompressed`,
  `moe_3gemm_fused_compressed`, `moe_router_fused`, forty of each —
  DESIGN §7.0.2u). Detect on those three names. **(b)** `u4` is already in the
  pass's admitted type list at `:648`, so "is u4 selectable at all" is answered
  on the record and is not what the window is testing. What it is testing is
  the **gate**: `supports_immad` AND `use_onednn` AND NOT `moe_disable_fusion`,
  plus whether our emitted shape matches the matcher. **Record all three gate
  terms beside the verdict** — a run with oneDNN off gets no fusion at all and
  a `u4` that falls back, which looks identical to "u4 was not selected" and
  means something entirely different.
* **Q2 IMPLEMENTATION** — if selected, is it the jit kernel or the `ocl:ref`
  fallback? `docs/prefill-baseline.md` §M2 measured 40 of 371
  `FullyConnectedCompressed` nodes falling from `jit:gemm:any__i8` onto
  `ocl:ref:any__i8` at a 2048-token prefill, consuming a THIRD of the chunk
  (68.3 ms against 38.9 ms for the other 331) — and all 371 are on the jit
  kernel at M=1 and M=2. The compressed path is real, shape-sensitive, and has
  a reference-kernel cliff. That evidence is all at **i8**; the expert bodies
  are **u4**, grouped, rank-4 before the collapsing Reshape.

**THE PRIOR, and it is not nothing.** §4.4 measured the same graph costing 7.6
GiB of host on the GPU path against 28.5 GiB on the CPU path — "the CPU plugin
expands the u4 constants host-side; the GPU plugin does not". That is evidence
about LOAD-TIME host residency, not about which kernel the compiled graph runs,
so it constrains Q1 without answering it and says nothing about Q2.

**THE CANDIDATES**, each one variable from `as_emitted` except the control:

| candidate | differs by | why it is in the list |
|---|---|---|
| `as_emitted` | — | what `serving_shape._compressed_expert` emits today |
| `f16_scale` | scale f16 not f32 | `gguf_graph.cpp:229` builds an f16 scale |
| `u8_scalar_zp` | zero-point scalar u8 not per-group u4 | `gguf_graph.cpp:225-228`, `RepackZeroPoint::U8Scalar` |
| `production_2d` | rank-3 + rank-2 MatMul | **POSITIVE CONTROL**: `gguf_graph.cpp:216-235`, the shape arcint already serves on these cards |
| `no_reshape` | trailing Reshape removed | **NEGATIVE CONTROL**: `verify_moe_lowering.py:33-42` records a real GPU compile crashing without it |

**WHAT THE WINDOW EXECUTES, IN ORDER.** The reproducer's own header carries this
and the numbered steps; in short: characterise on CPU first and check the IR
hashes still match, then `production_2d` on the card (if THAT is not
compressed, the detector is wrong and no other verdict counts), then
`as_emitted` — which is the row this manifest is waiting for — then
`f16_scale` / `u8_scalar_zp` only if `as_emitted` came back uncompressed, then
`no_reshape` last.

**IR HASHES, so the window can prove it ran what was characterised.** Written
by the frontier's CPU step 1, not filled in here: a hash recorded in this
document from an engineer session would be a claim about a tree the window has
not run. The reproducer prints `sha256` per candidate and the window records
them in the same commit as the GPU result, per CF-MANIFESTSHA.

`RUN@5663a44`, dev host, CPU, OV 2026.4.0-22849, E=8 M=64 I=640 H=2560
group=128 — every candidate builds, compiles, and hashes. **The Q1/Q2 columns
are the window's to fill**, in the same commit as the GPU result.

| candidate | ops | IR sha256 (CPU, `RUN@5663a44`) | Q1 selection | Q2 kernel |
|---|---|---|---|---|
| `as_emitted` | 12 | `de8a7a8a3a60daa60f299ad85f8677005154837256cb0462f13e3a8133334d57` | | |
| `f16_scale` | 13 | `d2e06b2e44f447c9c5adbc322409ce9518b8b21307852f50a9f8bf695367b89b` | | |
| `u8_scalar_zp` | 12 | `4fd60e75f2396c67760dec5242097b548d600e0a6840fb79523a03f5382d8cf4` | | |
| `production_2d` | 13 | `4324e6003e4699d7cad1a5ad211ac98e903f030fb480679020f8d9be1f3d3fc2` | | |
| `no_reshape` | 10 | `5f548715ed8b397d12fafb8b6bde23986d444096bb1fc9c81049afa7910afe78` | | |

**WHAT CPU ALREADY SAYS, and it is less than it looks.** `FullyConnectedCompressed`
and `MOECompressed` are GPU-plugin primitives; the CPU plugin neither names nor
runs those passes, so a CPU histogram cannot answer Q1 either way and the
reproducer refuses to print a verdict on CPU. What it does show is one real
signal: four of the five candidates collapse to a single `FullyConnected`
(`brgemm_avx2_f32`) with the u4 `Const` surviving into it, while `no_reshape`
stays a plain `MatMul`. Even the CPU plugin's primitive choice responds to the
trailing Reshape — consistent with `verify_moe_lowering.py:33-42`, on a
different plugin, and not a substitute for it.

**ONE CORRECTION ALREADY MADE HERE**, because the negative control was wrong
the first time: deleting the trailing Reshape from the rank-4 chain does not
build at all (`Incompatible MatMul matrix dimension ... 2560 ... 128`) — that
is a shape error, not the defect. What `verify_moe_lowering.py:33-42` records
is a FLAT RANK-3 weight with no groups dimension, which is shape-valid and
compiles on CPU. `no_reshape` is now that.

**THE OTHER QUESTION THE FILL RAISES, and it is not this one — but it now has
a number.** The shipped checkpoint is a mixed k-quant (gate/up `IQ3_XXS`, down
`IQ4_NL`, measured `RUN@5663a44`). `q4e.gguf_feed` hands back DEQUANTISED f32
and `q4e.expert_fill` re-quantises to u4 grouped-affine — a SECOND
quantisation. Measured end to end on one real-weights expert piece
(`RUN@5663a44`, 8 real experts at real width, M=16, CPU):

| leg | value |
|---|---|
| \|executed − dequantised-u4 reference\| | 2.423189e-09 |
| f32 floor for the contraction | 2.171543e-08 |
| \|executed − raw f32 reference\| | **5.657107e-04** |
| output magnitude | 3.220204e-03 |
| **quantisation cost, relative** | **17.6%** |

The plumbing sits BELOW the f32 floor, so that 17.6% is quantisation and not a
packing bug. It is large. ~~Carrying the file's own blocks through
`src/core/gguf_repack.cpp` instead — which is what `gguf_apply_to_template`
does for the dense models arcint serves today — is the alternative, and which
one the 0.5.0 artifact ships is a decision, not a measurement. It is **not
made here**. Deciding it needs the KLD gate (§7) run on both at full geometry,
which needed the fill to exist first. It does now.~~

**RETRACTED 2026-09-12 — there is no such alternative on this artifact, and
the sentence above stays visible as retracted rather than being edited away
(DESIGN §7.0.1).** The alternative was named and never measured. Measured now,
and gated by `tests/python/test_repack_route.py`, four coordinates:

| # | coordinate | cited | verdict, generated by the cell |
|---|---|---|---|
| C1 | type | `gguf_repack.cpp:260` | `repack_supported` admits ggml 8/12/13/14 (Q8_0, Q4_K, Q5_K, Q6_K). The shipped bodies are IQ3_XXS 94, IQ4_NL 43, IQ4_XS 2, Q8_0 5 — **139 of 144 refused on type alone** |
| C2 | rank | `gguf_repack.cpp:266` | `repack_tensor` takes 2-D; every expert body is rank 3 `[in, out, E]`. Closes the Q8_0 tail too — **0 of 144 can enter `repack_tensor` at all** |
| C3 | loss | `gguf_repack.h` §"What is exact and what is not", `gguf_repack.cpp:486` | the path is a TRANSCODE: a K-quant group scale "rounds to f16" and `repack_bound_steps` bounds a non-zero deviation. "Zero added loss" is a property of Q8_0's own stored form, not of the path |
| C4 | representation | IQ3_XXS + IQ4_XS: `design-gguf-native.md:52`. IQ4_NL: measured, `test_repack_route.py`'s IQ4_NL cell | the matched IR chain is uniform affine over a u4/i4/u8/i8 `Constant` — `(q − zp) · s`, sixteen equally spaced levels — and a codebook is not that. **Sits above `gguf_repack.cpp` and survives its repair.** Attribution corrected under H3: `design-gguf-native.md:52` names IQ3_XXS and IQ4_XS (96 bodies, 28.94 GiB) and does NOT name IQ4_NL; the 43 IQ4_NL bodies (18.90 GiB) rest on the measured levels — spacings 11…24, 2.18×, best affine fit off by 0.847 of a step — and are in any case closed independently by C1+C2 |

So the KLD gate cannot be run "on both at full geometry": the second path does
not exist to be run. **The 0.5.0 artifact ships the u4 fill, as PLUMBING AND
FIXTURE**, which is where the ruling put it — not because the decision was
weighed and went that way, but because the alternative was measured and is not
reachable from here. The prize was real and is worth recording for whoever
opens this next: the shipped expert bodies are **51.99 GiB** at the file's own
mixed quantisation against the 56.25 GiB uniform-int4 figure in the WP6 fit, so
a block-carrying route would have been lossless *and* 4.26 GiB smaller.

What is NOT claimed: that no block-carrying route exists. C4 closes the affine
chain only. A lossless IQ4_NL carry is representable in principle (u4 codes as
indices into a 16-entry codebook `Gather`, times the per-32 f16 scale) and is
NOT attempted, because a `Gather` decode is not the pattern
`ConvertTiledMoeBlockToGatherMatmuls` matches and would take the expert path
off the fused OTD route entirely — a design decision with a measurable cost,
not a plumbing fix. IQ3_XXS, 94 of the 144 bodies, does not have even that
option in reach. Full derivation, every hop cited both sides: the DESIGN NOTE
of 2026-09-12 in `RECONCILE-0.5.0.local.md`.

## 5. Residency — the SIZE LEDGER, and the number that decides the window

`RUN@wt+2e99661`, 2026-09-12, real checkpoint geometry, T=64, CPU, every row either built
and measured (`graph`) or computed from the shipped tensor list (`file`):

```
piece              tier       xN   nodes  const/inst GiB   total GiB  compile s  source
gdn_block          CARD        36    2054         0.2194       7.897       0.48  graph
gatedresidual      CARD        96      44         0.0247       2.369       0.05  graph
embed              CARD         1       5         2.3682       2.368       0.09  graph
lm_head            CARD         1       4         2.3682       2.368       0.76  graph
attention_block    CARD        12     150         0.1856       2.227       0.11  graph
moe_shexp          CARD        48      17         0.0183       0.879       0.02  graph
moe_router         CARD        48      22         0.0049       0.234       0.02  graph
ple_block          CARD         1     286         0.1235       0.124       0.09  graph
hc_combine         CARD         1      35         0.0245       0.025       0.04  graph
moe_experts        OFFLOAD     48     n/a         9.3750     450.000        n/a  file
ple_ngram_table    HOST-MMAP    1     n/a       190.7358     190.736        n/a  file
TOTAL CARD  18.492   TOTAL OFFLOAD  450.000   TOTAL HOST-MMAP  190.736
GRAND TOTAL 659.228
```

**Assumption named: every weight becomes an f32 ov Constant** — the emitter's
actual behaviour.

```
CARD tier 18.492 GiB vs B60  22.71 GiB -> FITS         (+4.218 GiB)
CARD tier 18.492 GiB vs A770 15.0  GiB -> DOES NOT FIT (-3.492 GiB)
f32 constants cost 7.72x the shipped checkpoint (659.2 vs 85.38 GiB, WP6b)
```

Cross-checked against the refusal's independently recomputed residency
(659.1 GiB over the whole mapped tensor list, a different route entirely):
**0.02% apart**.

**Consequence for this window: the card-resident-first set at f32 does not fit
the reserved A770.** It clears the B60 with 4.2 GiB spare. A window that intends
to serve on GPU.1 needs a weight strategy first — quantised constants, and a
gather for the n-gram table — or it needs to be a GPU.0 window. This is not a
scheduling preference; it is 3.5 GiB.

---

## 6. The 86k-node compile — **ANSWERED**

The question this section carried was: *"whether the GPU plugin's per-shape
kernel JIT copes with 86k nodes is STILL UNANSWERED"*. It was unanswerable
while §4.3's fusion defect stopped program building in seconds at every layer
count including 4. With §4.3 wired, the experiment ran.

CPU behaviour, unchanged reference (`RUN@57b1952`, REVIEW 57b1952 §7): node
count is linear in GDN layer count at ~1,879 nodes/block, and an 86k-node graph
compiles on CPU in 22 s.

```
layers    nodes  nonconst   xml MB  build s  compile s   (RUN@57b1952, CPU)
     4     9671      3727     4.25     0.10       1.92
    12    28663     10975    12.71     0.38       6.09
    36    85639     32719    38.05     0.95      21.95
```

`RUN@be57428`, 2026-09-12, 20-minute hard budget per attempt, each attempt in
its own child process:

```
layers  device   nodes   const B   build s  compile s  peak host GiB  result
     4   GPU.0    9727    849,892     0.10       5.41           1.34  OK
     4   GPU.1    9727    849,892     0.11       5.34           1.33  OK
    12   GPU.0   28831  2,432,804     0.30      15.88           1.22  OK
    12   GPU.1   28831  2,432,804     0.31      16.43           1.23  OK
    36   GPU.0   86143  7,181,540     0.96     204.90           2.00  OK
    36   GPU.1   86143  7,181,540     0.92     203.79           2.01  OK
```

**YES — the GPU plugin's per-shape JIT copes with 86 thousand nodes.** 86,143
nodes compile on **both** cards, in 204.90 s (B60) and 203.79 s (A770) — within
0.5% of each other — well inside the 20-minute budget, at 2.0 GiB of peak host
memory. The compile cost is not a card property. Every previous attempt, at every layer count,
died in seconds with a `RuntimeError` that had nothing to do with node count.

**The cost is superlinear in nodes and that is the finding worth carrying**:

```
nodes    compile s   ms per node
 9,727        5.41       0.56
28,831       15.88       0.55
86,143      204.90       2.38
```

Flat at ~0.55 ms/node to 29k nodes, then **4.3× worse per node** at 86k. The
GPU compile is 9.3× the CPU's 21.95 s at the same scale, against 2.8× at 4
layers. Whatever the plugin does that is not linear starts between 29k and 86k
nodes — that is where a node-count reduction (the F3 hoist) would pay, and it
is now a measurable payoff rather than a guess.

The F3 hoist (~65% node cut) is still **`UNTESTED`**, but its premise is no
longer hypothetical: it was "only worth measuring against a GPU compile that
can start", and the compile now starts.

A timeout is a result and gets written down as one. So is a budget that was
never approached — twice, in opposite directions.

## 7. The KLD gate

`tools/kld_harness.py` exists and is red-probed: mean per-token
KL(P_ref‖P_cand) ≤ **0.0599 nats**. It now pins f32 on GPU devices for the same
reason §3 gives — a KLD read off an f16 forward is not a measurement of the
export.

What still stands between the tip and the gate (`UNTESTED` as a whole — the gate
has not been run on an assembled Flash-Next artifact):

1. **The standing law is only partly served.** The real-weights sweep exercises
   gdn + hc + moe + ple jointly through the assembled backbone at four sequence
   lengths, and the dense-causal attention piece now has its own real-weights leg
   at two shapes. But a compensating pair of errors across two blocks inside the
   joint path would still be invisible, and nothing would localise it. REVIEW
   58e3e09 finding 3 asked for one real-weights cell PER emitted block; attention
   is the first one to exist.
2. **KLD needs a reference trustable at FULL geometry.** Everything measured so
   far at real WIDTH is per piece; the assembled legs are tiny geometry (2 of 512
   experts, ff 32 of 640, hidden 16 of 2560, vocab 257 of 248320, 4 layers of
   48). Real bytes, not real shape — no number there transfers to the full model
   by itself.
3. **Full-size emission is blocked by the enumerated blockers**, in order, and
   none of them is a device.

---

## 8. PREDICTION TEMPLATE

**Fill this in BEFORE the serving measurement is taken.** A prediction written
after the number is not a prediction. Constants measured this session are filled
in; everything else is an explicit blank, and a blank left blank is a better
record than a blank guessed.

### Constants that now exist (measured, with provenance)

| term | value | provenance |
|---|---|---|
| card under test | A770, 15.11 GiB reported (16,225,243,136 B) | `RUN@e78812d` GPU.1 enumeration |
| alternate card | B60, 22.71 GiB reported (24,385,683,456 B) | `RUN@e78812d` GPU.0 enumeration |
| CARD-tier weights at f32 | 18.492 GiB | `RUN@wt+2e99661` size ledger |
| → fits A770 reserve | **no**, −3.492 GiB | `RUN@wt+2e99661` size ledger |
| → fits B60 | yes, +4.218 GiB | `RUN@wt+2e99661` size ledger |
| offload tier | 450.000 GiB f32 (48 × 9.375) | `RUN@wt+2e99661` size ledger, file-sourced |
| host-mmap tier | 190.736 GiB f32 n-gram table | `RUN@wt+2e99661` size ledger, file-sourced |
| f32 : quantized ratio | 7.72× | `RUN@wt+2e99661` size ledger vs WP6b 85.38 GiB |
| GPU inference precision | f32, pinned explicitly | `RUN@829a213` §3 |
| GPU.1 max single allocation | **4,294,959,104 B (4.00 GiB)** | `RUN@be57428` §4.4 |
| GPU.0 max single allocation | 24,385,683,456 B (whole VRAM) | `RUN@be57428` §4.4 |
| serving-shape boot, 1 layer | compiles + infers on BOTH cards | `RUN@be57428` §4.4 |
| → GPU.1 compile / infer | 14.02 s / 0.400 s, 7.63 GiB host | `RUN@be57428` §4.4 |
| → GPU.0 compile / infer | 10.48 s / 0.629 s, 7.57 GiB host | `RUN@be57428` §4.4 |
| → CPU host cost, same graph | 28.50 GiB (3.7× the GPU path) | `RUN@be57428` §4.4 |
| QSA→dense price, T ≤ **2051** | **0.0, exact** (boundary derived, not the budget 2048) | `RUN@692c0a6` attention piece |
| QSA→dense price, T = 2052 | 2.307817e-06 over **1**/2052 rows | `RUN@692c0a6` attention piece |
| QSA→dense price, T = 2080 | 2.385560e-02 over **29**/2080 rows = T−2051 | `RUN@692c0a6` attention piece |
| attention piece floor vs pin | 1.855e-07 (T=64), 1.535e-07 (T=96) | `RUN@e78812d` attention piece |
| KLD gate threshold | ≤ 0.0599 nats mean per-token | harness, red-probed |
| serving-shape IR, 48 layers | 84,372 nodes, 36 GDN + 12 dense-causal | `RUN@198b736` `--serving-shape` |
| → declared constants | 183.07 GiB | `RUN@198b736` |
| → materialised on disk | **0 KiB** — but see the qualifier below; this figure does not discriminate | `RUN@198b736`, qualified `RUN@5663a44` |
| → build cost | 6.43 s, 4.6 GiB RSS | `RUN@198b736` |
| expert body declared type | u4, rank-4 [E, out, groups, 128] | `RUN@198b736` contract test |
| per-expert int4 slice | 2,457,600 B (gate+up+down) | `flash_next_offload.h:45`, re-derived |
| → as the IR walk reads it | 4,915,200 B = **exactly 2×** | `RUN@198b736` (u4 ceiled to 1 B) |
| `slot_pool_from_ir` on any arcint IR | **nullopt** — 0 of 172 IRs carry a moe-typed op (52 of them over 100k; the wider census re-run 2026-09-12) | `RUN@198b736` |
| MoE router on GPU | scatter shape OK both cards; one_hot FAILs | `RUN@be57428` §4.3 |
| → cost of the swap | 0.000000e+00 on CPU | `RUN@be57428` |
| GDN GPU first bad row | 65, at every T ≥ 66, both cards | `RUN@be57428` §4.2 |
| GPU acceptance doctrine | \|ov−r64\| ≤ 20 × \|r32−r64\| | `RUN@be57428` §4.2 |
| 86k-node GPU compile | **204.90 s** on B60, 86,143 nodes | `RUN@be57428` §6 |
| → compile cost scaling | 0.55 ms/node to 29k, 2.38 ms/node at 86k | `RUN@be57428` §6 |
| shipped expert bodies, census | IQ3_XXS 94, IQ4_NL 43, IQ4_XS 2, Q8_0 5 = 144 over 48 layers | `RUN@bd5f53c` `test_repack_route.py`, regenerated off `/flash-model` |
| → shipped expert bytes on disk | **51.99 GiB** (55,823,564,800 B) | same cell |
| → the same experts as emitted u4 | **56.25 GiB** (60,397,977,600 B = 512 × 48 × 2,457,600) | `flash_next_offload.h:45`, re-derived |
| block-carrying repack route | **CLOSED**: 0 of 144 bodies enter `repack_tensor` (139 refused on type, the Q8_0 tail on rank) | `RUN@bd5f53c` `test_repack_route.py` |
| MoE fusion gate, plugin side | `supports_immad && use_onednn && !moe_disable_fusion`; admitted weight types `{u4, i4, i8, u8}` | dated foreign-tree read, `transformations_pipeline.cpp:647-648`, `~/ovsrc-pkg` @ 2026.4.0-22849 |

The two expert-byte rows are both correct and they are not the same number.
**56.25 GiB is the artifact's**, and it is the one every residency row here is
computed against, because the IR declares uniform u4. 51.99 GiB is what the
GGUF holds at its own mixed quantisation — reachable only by a block-carrying
route, which §4.5.1 records as closed. Do not "correct" the fit model to
51.99: that would be sizing a pool for weights this artifact does not contain.

**QUALIFIER on "materialised on disk 0 KiB" (`RUN@5663a44`, 2026-09-12).**
That row is `SparseArena.disk_kib()`, and on the dev host's filesystem it
CANNOT distinguish an unwritten arena from a written one. Measured, ZFS
(recordsize 131072), one 4 GiB sparse file per row:

| written into the file | `st_blocks × 512` after msync | after `sync` + 12 s |
|---|---|---|
| nothing | 512 B | 512 B |
| 512 MiB of zeros | 512 B | 512 B |
| 512 MiB of **random** | 512 B | **439,174,656 B** |

ZFS allocates on transaction-group commit rather than on msync, and stores an
all-zero record as a hole. So 0 KiB is true of the unfilled build, and would be
equally true of a build that had materialised every constant as zeros, and of
one that had just written the real weights. The row is kept because it is one
more sign and because it does discriminate on an eagerly-accounting
filesystem; it is **not** the evidence for the keystone. That is the peak-RSS
assertion in
`tests/python/test_serving_shape.py::test_the_full_48_layer_stack_emits_at_real_geometry`
(CF-RESIDENT, ceiling 5.31 GiB derived from a 4.52-GiB authored peak and a
6.23-GiB cheapest defect) and, for a FILLED build, the non-zero fraction on
read-back, which no filesystem can fake.

This qualifier was found while accepting the fill, not by re-reading the row.

### Terms still to be predicted — `UNTESTED`, fill before measuring

A term is filled here ONLY where a measured constant already determines it, and
the arithmetic is shown so the prediction can be attacked before the window
rather than explained after it. Everything else is left blank on purpose.

| term | prediction | then: measured | provenance of the prediction |
|---|---|---|---|
| resident GiB on the card | **7.8** GiB of expert pool on a 15.11 GiB A770 (backbone 2.3 + KV 3.0 + activation 2.0 leaving 7.8) | | `flash_next_fit.py --vram 15.11 --dram 44`, generated |
| → expert pool total, VRAM+DRAM | **24.0 GiB of 56.25 (43%)** = VRAM 7.8 + DRAM 16.2 | | same run |
| KV precision pinned | **u8** — a configuration choice, not a measurement; it is the `--kv 3.0` GiB row's premise | | handoff E5; pin it explicitly or this row is fiction |
| prefill t/s at T=2048 | | | needs the window; no bandwidth model predicts prefill |
| decode t/s, single lane | **10.0 t/s** at the WP6b hit rate, bandwidth-bound, MTP amort 1.0 | | `flash_next_fit.py --hit-rates 0.881`, generated. A PROJECTION and labelled as one in `flash_next_offload.h`'s header — every input measured, the t/s analytic |
| expert hit rate (cache) | **88.1%**, PER-LAYER LRU at a 16 GiB budget | | WP6b routing-trace replay, `flash_next_offload.h`'s cache-model paragraph. Do NOT substitute the global-LRU 93.8%: same trace, more optimistic model, and adopting it overstates the t/s above |
| expert miss cost (per miss) | **1.362 ms** from NVMe (2,457,600 B ÷ 1.68 GiB/s); 0.0515 ms if the slice is already DRAM-resident | | `flash_next_offload.h:45` slice bytes ÷ the WP6b in-container `dd iflag=direct` figure |
| per-token full-miss traffic | **1.0986 GiB** (10 active × 48 layers × slice) | | `flash_next_fit.py`, generated |
| MTP acceptance rate | | | needs the window |
| MTP overhead term | | | needs the window |
| amortised t/s incl. MTP | | | needs MTP acceptance; the `--amort` dial is wired and defaulted to 1.0 above, i.e. NO amortisation assumed |
| 86k-node GPU compile, s | | **204.90** (B60, 86,143 nodes) | `RUN@be57428` §6 |

**What would falsify the decode row.** 10.0 t/s assumes every miss is paid at
NVMe bandwidth and nothing else is the bottleneck. A measured decode materially
BELOW it means the bottleneck is not expert bandwidth — compute, the router, or
the host excursion — and the streaming plan's whole premise needs re-reading. A
measured decode materially ABOVE it means the hit rate beat the per-layer
replay, and the replay is the thing to re-run. Either way the number to compare
is single-lane, MTP off; with MTP on, compare against `--amort` set to the
measured acceptance and not against this row.

### The coherence line — RESERVED, LEAVE EMPTY

Only the frontier's window writes this row, **in the same commit as the
measurement**. An engineer session must not fill it, and must not fill it with a
baseline probe from a different model either.

| probe | served model | answer | tokens | MTP acc/rej |
|---|---|---|---|---|
| "What is the capital of France? Answer in one word." | | | | |

(For contrast and NOT as a substitute: the pre-window baseline of the *resident
agent* — a different model, `qwen3.8-agent` on :8087 — answered `Paris` in 27
completion tokens, MTP 13 accepted / 1 rejected, at 2026-09-12 23:54Z. That row
belongs to §2's stop procedure, not to this table.)

---

## 9. Morning restore — non-negotiable, before any close-out

```
# RUN@e78812d
date -Is
systemctl --user start arcint arcint-agent
systemctl --user is-active arcint arcint-agent
sleep 20 && curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8087/props
# coherence probe, >= 64 max_tokens (the model reasons before it answers)
curl -s http://127.0.0.1:8087/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-agent","messages":[{"role":"user",
       "content":"What is the capital of France? Answer in one word."}],
       "max_tokens":200,"temperature":0}'
# zombie sweep: a SIGTERM-ignoring test process on a card wedges the service
pgrep -af pytest
```

Restore checklist — all of it, with evidence, or the session reports the GPU
state as NOT restored:

- [ ] `arcint-agent` active, `/props` → 200, probe answers coherently
- [ ] `arcint` (coder, GPU.1) active — it was running before the window and
      leaving it down is an undocumented change
- [ ] `openarc-coder` still stopped and disabled (reservation mneme 361)
- [ ] no leftover pytest / arcint processes holding either card
- [ ] device memory back to the services
- [ ] scratch trees under `~` removed
- [ ] timestamps of every stop and start recorded in RECONCILE

---

## 10. What this manifest deliberately does not contain

No host names beyond the ones already tracked in this repository's public
documents, no addresses, no credentials, no access paths. Operator-local detail
belongs in `CLAUDE.local.md`. If a window needs infrastructure detail that is not
there, ask rather than writing it here.
