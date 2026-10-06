# window-050 — the 0.5.0 prediction window, operating manifest

> Kept on `main` because a test reads this manifest by path. It is a dated
> record of the 0.5.0 window, not current state (DESIGN.md is). Other
> `docs/` documents it links to live on the development branch, `qfndev`.

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

**HOW TO READ ANY SUITE COUNT IN THIS FILE (K2, `RUN@wt+fe68342`).** A
passed/skipped split is only meaningful next to the switches that were set,
because every skip in this suite is gated. Two honest readings of the same
commit differed by two passes and the reconciliation is that there are exactly
five coordinates, now enumerated and held by
`tests/python/test_suite_guards.py`:

| coordinate | effect when open |
|---|---|
| `Q4E_GPU` | empty → CPU only; a device list adds the per-device legs |
| `Q4E_GGUF_SHARDS` | unset → every real-weight cell skips **by name** |
| `Q4E_SERVING_FULL` | `1` → runs the 48-layer keystone build (off by default on purpose) |
| `Q4E_GDN_UT_MODE` | read by `tools/q4e/gdn.py`, so it is a switch the suite obeys through an imported module rather than through a test file; its effect on the split is a row of the close-out matrix like any other |
| `Q4E_GDN_CORE`, `Q4E_GDN_CHUNK` | *[added 2026-09-27]* read by `tools/q4e/serving_shape.py` (the LYON chunked GDN core and its chunk); unset, the sequential core, as served |
| **a git work tree** | *not* an env var: a `git archive` extract has no `.git`, so the two cells gated on `git ls-files` (`test_citations` LEG 2 and `test_window_manifest`'s sha resolution) skip |

So a clone and a tarball of the same commit differ by exactly **two** passes,
and both are correct. `test_the_suite_declares_no_count_gate_outside_the_
recorded_set` goes red if a new `Q4E_*` switch appears, and
`test_the_checkout_shaped_gates_are_exactly_the_recorded_cells` goes red if a
third checkout-gated cell does — in a decorator or in the body of a cell, since
`pytest.skip("no .git")` written inline is the same gate — the point being that
a future disagreement is always attributable to a named coordinate, and
**nobody reconciles two counts by opening a gate.** The measured matrix for the
current tip is in the close-out; the historical figures above keep the commit
ids their own markers carry (the `@<sha>` suffix the header table defines) and
are not retro-fitted.

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

  **CORRECTED 2026-09-13, twice, and the bullet above is kept as it was
  written because it is what the window recorded.** (a) "not emitted" named
  the wrong side: no exporter emits those ports on any artifact this fleet
  serves. `load_paged` runs `ov::pass::SDPAToPagedAttention` over the IR it
  has just read (`backend_ov.cpp:2582`) and the ports are that pass's output,
  produced from three constructs of the *stateful* graph. (b) The count is no
  longer a fixed 13 to recite: the suite's `_PAGED_PORT_TABLE` carries one row
  per port with its status, both cells read that table, and the number of rows
  is written nowhere. As of the same day the full-attention layers are
  stateful and the pass produces seven of them; the GDN layers' two state
  tables and four `la.*` ports are what the xfail still covers. The design
  note carries the reading.

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
| 5 | **KLD gate vs the llama-fork reference, BOTH context regimes** | `tools/kld_harness.py` on the booted artifact, at a T **below** and a T **above** the 2051 boundary | mean per-token KL(P_ref‖P_cand) ≤ 0.0599 nats at both (PROVISIONAL bar — inherited: 1.5 × 0.0399 nats, Qwen3.6-35B-A3B UD-Q3_K_XL vs BF16, 2026-08-11; provenance REVIEW ba2d5de F2). The boundary is not decorative: the QSA→dense price is exactly 0.0 for T ≤ 2051 and non-zero above it (`RUN@692c0a6`, §8), so a gate run only below it has not exercised the dense rows at all | a KLD that passes below 2051 and fails above localises to the QSA→dense seam, which is the one place the price is known to change |
| 6 | **MoE compile at short T, under the real plugin** | `tools/repro_moe_compile_short_T.py <T> GPU.N` | the CPU plugin dies on SIGSEGV at T=6 and T=8 and nowhere else; whether the CARD's plugin shares the cliff is unknown | if the card refuses the same two shapes, a short prefill is a serving constraint, not a curiosity |

Item 6 is run LAST on each card because it may take the process down.

Item 5 needs item 1 to have produced a booted artifact; if item 1 does not
boot, item 5 does not run and says so rather than being run at a reduced
geometry whose number would not transfer (§7.2).

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
shape with C > 4.

## 4.6 ITEM 4 — THE BOOT AT DEPTH, PREDICTED BEFORE IT RAN (2026-09-13)

Written with **no boot output in existence** for the tree it predicts (the last
boot output this repository holds is `RUN@be57428`, §4.4, taken on the
pre-stateful graph). Every row was `UNTESTED` when written — the prediction
commit is `8a84598` — and the "Measured" table below is the run that followed
it, `RUN@8a84598`, pasted verbatim next to what was predicted. Every term is a
B60 / A770 measurement this repository already holds, or a device-free reading
of the pinned OpenVINO source; no external number appears.

> **AMENDMENT (2026-09-13, operator CR #2 — a mark, never an erasure).** The
> Paris line that this section's acceptance commit (`8a84598`) defined for
> 0.5.0 — the served model's own token for the capital of France — **moved to
> 0.5.1 by operator CR #2 (2026-09-13)**. 0.5.0 ships as the **mechanism
> release**: the served path live on both cards through the runtime's own
> binding site, the thirteen paged ports fed, the dynamic-T serving class,
> the n-gram table as USM-host ports, the reviewers' riders and the PLE
> oracle cell, the KLD instrument with its own floor (red at depth 4, as
> predicted), the ladder green. The France row in §8 carries the dated probe
> token (`ramework`, id 5613) **as mechanism evidence, not as an answer**. The
> clause text of this section stays legible as written at `8a84598`; its
> falsifiable Paris clause travels unchanged to the 0.5.1 acceptance
> (`docs/window-051.md`), where the row that fills it is the model's own token
> at depth 48 or a named refusal.

**The reproducer is `tools/boot_serving_shape.py`**, committed with this
section so the experiment is defined before it is run. It transcribes
`load_paged`'s own sequence stage by stage (build → state prototypes → the
pass → compile → request → the forward's `set_tensor` order) and captures the
first refusal by name. It fills nothing: every weight is an unwritten arena
page, as in `RUN@be57428`. "Depth" here is the LAYER count, the sense §4.5
item 1 uses ("1-layer boot, then depth / 4 GiB chunking").

### The constraints, both already on the record

| constraint | value | provenance |
|---|---|---|
| A770 per-object cap | 4,294,959,104 B | `RUN@be57428` §4.4 |
| B60 per-object cap | 24,385,683,456 B (its whole VRAM) | `RUN@be57428` §4.4 |
| the PLE n-gram table, ONE object | 25,600,122,880 B = 320,001,536 × 160 nibbles, enters at layer index 1 | `RUN@be57428` §4.4 |
| first full-attention layer | index 3 (`kind = "attn" if i % 4 == 3`), so the first SDPA exists at **depth 4** | `serving_shape.build_serving_shape_ir` |
| what the pass demands | a stateful graph AND at least one `v13::ScaledDotProductAttention`; each absence is an `OPENVINO_ASSERT` by name | pinned OV source `71640275`, `src/core/src/pass/sdpa_to_paged_attention.cpp`, `run_on_model` |
| what the served forward feeds first | `inputs_embeds`, unconditionally (`backend_ov.cpp:6153`); the IR declares `input_ids` | contract cell `test_the_converted_surface_against_every_tensor_the_forward_feeds` |
| 1-layer boot, pre-stateful graph | compiles + infers on both cards, `absmax = 0.0000e+00` | `RUN@be57428` §4.4 |

### The prediction — read the two constraints together

**The depths the pass accepts (≥ 4) and the depths the A770's object cap
admits (≤ 1, and the B60's too, since 25.6 GB exceeds its 24.39 GB) are
disjoint. There is no depth at which the served path boots this IR on either
card.** That is the headline, and it is falsified by ANY served-path forward
returning logits at any depth on any card.

| # | leg | prediction (written at `8a84598`, before the run) | dies if |
|---|---|---|---|
| P1 | pass, device-free, depth 1 (and 2, 3) | **REFUSED**: `No ScaledDotProductAttention operation observed in the graph, cannot perform the SDPAToPagedAttention transformation.` | the pass converts a graph with no SDPA |
| P2 | pass, device-free, depth 4 | OK; `PagedAttentionExtension` 1, `PagedCausalConv1D` 3, `PagedGatedDeltaNet` 3, one `key_cache.`/`value_cache.` pair, all nine index ports | the pass refuses depth 4, or the op census differs |
| P3 | compile, depth 4, served props (`KV_CACHE_PRECISION` u8, no precision hint), **A770** | FAIL at `engine.cpp:319`: `requested 25600122880 bytes, but max alloc size supported by device is 4294959104 bytes` | depth 4 compiles on the A770 |
| P3b | the same on the **B60** | FAIL at `engine.cpp:319`, the same request against `24385683456 bytes` | depth 4 compiles on the B60 |
| P4 | France question through the served path | **no forward is reached on either card**; the §8 coherence line records that, not a token | any served-path `infer()` returns |
| P5 | CONTROL: depth 1, `--no-pass`, f32 pinned, both cards, prompt ids `760,6511,314,9338,369` | compile OK, infer OK, `finite=True`, `absmax=0.0000e+00`, argmax `0` at all five positions, RAW greedy last-position token id **0** | the now-stateful depth-1 graph (v5::Loop core, stateful conv, Variables + Assign sinks — their first card compile outside the fusion) fails to compile; or any logit is non-zero; or any argmax is not 0 |
| P6 | CONTROL: depth 2, `--no-pass`, both cards | FAIL at `engine.cpp:319` with P3/P3b's numbers — the object survived the keystone reshape unchanged | depth 2 compiles, or the requested byte count moved |
| P7 | decode behaviour | on the served path: none (P4). On the control: no decode step is possible against the static query block — a 1-token block against the compiled `[1, 5]` port is **refused at `set_tensor`** with a shape-incompatibility text naming both shapes | the runtime accepts a `[1, 1]` block on a `[1, 5]` port |

**What "answers the France question" means after this table:** it cannot be
answered in this window, and the reason is not a missing fix in the C++'s
feed order. Before any of the handoff's three suspects (`input_ids` /
`ngram_row_ids` / `conv_mask` declared-never-fed; the static query block; the
rope span) can be MEASURED on a card, the served path has to reach a forward,
and P1+P3 say it cannot: the n-gram table has to leave the graph (host-mmap
gather, which is how serving reads it — `src/exec/ngram_table.h`) or be
chunked under 4 GiB, and the minimum served depth is 4. Those two are the
first two lines of increment 5's spec, and they are read off the device in the
measurement commit rather than argued here.

**What is NOT predicted**, on purpose: any timing (compile seconds on the
stateful graph, infer seconds) — the driver prints them and the measurement
commit records them as first readings, not as confirmations. The
`enable_large_allocations` lever is not pulled: "max depth within the cap"
means the cap as measured.

### The commands, in order (one process per leg, `timeout 1200` each)

```
# RUN@8a84598 — device-free, the P1/P2 legs
<venv>/bin/python tools/boot_serving_shape.py --layers 1 --stage pass
<venv>/bin/python tools/boot_serving_shape.py --layers 2 --stage pass
<venv>/bin/python tools/boot_serving_shape.py --layers 3 --stage pass
<venv>/bin/python tools/boot_serving_shape.py --layers 4 --stage pass
# RUN@8a84598 — A770 first (reserved card), then B60; P5, P6, P3
<venv>/bin/python tools/boot_serving_shape.py --layers 1 --device GPU.1 --no-pass --ids 760,6511,314,9338,369
<venv>/bin/python tools/boot_serving_shape.py --layers 2 --device GPU.1 --no-pass --ids 760,6511,314,9338,369
<venv>/bin/python tools/boot_serving_shape.py --layers 4 --device GPU.1 --ids 760,6511,314,9338,369
<venv>/bin/python tools/boot_serving_shape.py --layers 1 --device GPU.0 --no-pass --ids 760,6511,314,9338,369
<venv>/bin/python tools/boot_serving_shape.py --layers 2 --device GPU.0 --no-pass --ids 760,6511,314,9338,369
<venv>/bin/python tools/boot_serving_shape.py --layers 4 --device GPU.0 --ids 760,6511,314,9338,369
```

The prompt ids are `"The capital of France is"` under the shipped GGUF's own
tokenizer (`llama-tokenize` from the pinned reference build, §ITEM 3):
`760 'The'  6511 ' capital'  314 ' of'  9338 ' France'  369 ' is'`. The chat
form of the probe (`"What is the capital of France? Answer in one word."`) is
12 tokens under the same tokenizer and is not used here: the driver has no
chat template and the served path would not reach it either way.

### Measured — `RUN@8a84598`, 2026-09-13 07:36:52Z–07:38:46Z, both cards

The ladder above, run verbatim on a `git archive` extract of `8a84598` whose
`git write-tree` id (`ebadf8df…`) was computed on the export host and
independently on the dev host over the extracted tree. Plugin: the venv's
stock `2026.4.0-22849-71640275d29` (the same commit the served `+p15` builds
from), one process per leg, `timeout -s KILL 1200`, both serving units
inactive throughout (they were down before the session, by operator word, and
were not started). Raw logs: one file per leg beside the extract.

| # | card | measured | matches? |
|---|---|---|---|
| P1 | — | depth 1, 2, 3 each: `RuntimeError: Check 'ov::op::util::has_op_with_type<ov::op::v13::ScaledDotProductAttention>(model)' failed at src/core/src/pass/sdpa_to_paged_attention.cpp:81: No ScaledDotProductAttention operation observed in the graph, cannot perform the SDPAToPagedAttention transformation.` | **yes**, all three |
| P2 | — | depth 4: pass OK; `{'PagedCausalConv1D': 3, 'PagedGatedDeltaNet': 3, 'PagedAttentionExtension': 1}`; ports after: `input_ids[-1], position_ids[-1], ngram_row_ids[1,T,16], conv_mask[1,T], max_context_len[], past_lens, subsequence_begins, block_indices_begins, block_indices, key_cache.0, value_cache.0, la.block_indices, la.block_indices_begins, la.past_lens, la.cache_interval, conv_state_table.0-2[-1,10240,4], gated_delta_state_table.0-2[-1,48,128,128]` | **yes** |
| P3 | A770 | depth 4, served props: `FAIL after 4.89s peak_host_GiB=8.55 … Check '!exceed_allocatable_mem_size' failed at src/plugins/intel_gpu/src/runtime/engine.cpp:319: [GPU] Exceeded max size of memory object allocation: requested 25600122880 bytes, but max alloc size supported by device is 4294959104 bytes.` | **yes** |
| P3b | B60 | depth 4, served props: `FAIL after 4.14s peak_host_GiB=8.55 … requested 25600122880 bytes, but max alloc size supported by device is 24385683456 bytes.` | **yes** |
| P4 | both | no served-path `infer()` was reached on either card at any depth | **yes** |
| P5 | A770 | depth 1 control: `COMPILE FAIL after 17.65s peak_host_GiB=7.20 … Error has occured for: convolution:GroupConvolution_118 \| Weights feature maps number(=1) is not equal to: input feature maps number(=10240) \| Weights/ifm mismatch` | **NO — falsified** |
| P5 | B60 | depth 1 control: `COMPILE FAIL after 6.55s peak_host_GiB=7.45`, the same `GroupConvolution_118` text | **NO — falsified** |
| P6 | A770 | depth 2 control: `FAIL after 4.03s peak_host_GiB=5.63`, `engine.cpp:319`, `requested 25600122880 … 4294959104` | **yes** |
| P6 | B60 | depth 2 control: `FAIL after 3.70s peak_host_GiB=5.63`, `engine.cpp:319`, `requested 25600122880 … 24385683456` | **yes** |
| P7 | A770 / B60 | not reached: no forward compiled, so no `set_tensor` was measured against a static block | **not measured** |

First readings, not predictions: build 3.1–5.8 s at every depth; the stateful
depth-1 graph is **382 nodes** (`RUN@be57428`'s pre-stateful one was 2,290 —
the v5::Loop core collapsed the unrolled delta rule); depth 4 declares 63.43
GiB; the pass adds nothing to peak host (0.96 GiB at depth 4 before compile).

**The §8 France line stays a refusal, as predicted (P4).** Two of the three
suspects in the handoff (`input_ids` / `ngram_row_ids` / `conv_mask`
declared-never-fed; the static query block) were never reached; the third
(the rope span) was never in play at position 0. What the device wrote instead
is below.

### P5 — the control is dead on both cards, and the root is LOCALISED

The `RUN@be57428` witness ("the structure lights up on the card") no longer
exists for the stateful graph: the depth-1 control refuses to compile on both
cards at `GroupConvolution_118`. That op is `stateful_short_conv`'s depthwise
GroupConvolution, the construct read out of `PagedCausalConv1DFusion`'s source
in ITEM 2 (rank-4 `[conv_dim, 1, 1, K]` weights, groups = `conv_dim`, over the
`Concat(axis=-1)` of the K-column state and the new block). The default
emitter `q4e.gdn._causal_conv_silu` is a K-term slice/multiply unroll and
carries no GroupConvolution at all — which is why the earlier boot compiled and
this one does not.

Localised with `tools/repro_stateful_conv_compile.py` (`RUN@wt+8a84598` —
the reproducer was the working tree's only delta over `8a84598`; it lands in
the same commit as this section), one op each at the real geometry
(`conv_dim` 10240, K 4, T 5), one process per leg:

| variant | CPU | GPU.1 (A770) | GPU.0 (B60) |
|---|---|---|---|
| `default` — `_causal_conv_silu`, 37 ops | OK, `out(1, 10240, 5)` finite | OK 0.23 s | OK 0.11 s |
| `groupconv_static` — the SAME GroupConvolution over a static `[1, 10240, 9]`, 4 ops | OK, `out(1, 10240, 6)` finite | **OK** 0.11 s | **OK** 0.11 s |
| `stateful` — as emitted: ReadValue `[?,10240,4]` → Concat → GroupConvolution → Slice, 32 ops | compiles; **infer throws** `Node Assign_27 of type Assign … Check 'input.getDesc().isDefined() && output.getDesc().isDefined()' failed at src/plugins/intel_cpu/src/nodes/reorder.cpp:550: Can't reorder data with dynamic shapes` | **COMPILE FAIL** 0.11 s, `GroupConvolution_30 … Weights/ifm mismatch` | **COMPILE FAIL** 0.11 s, the same |

So: the op and its weight layout compile and run on both cards; what neither
plugin takes UNFUSED is the construct's dynamic-length input (the Concat of a
`[?, conv_dim, K]` ReadValue with the block) — the GPU plugin derives the
weights' feature-map count as 1 against the input's 10240 and refuses at
compile, and the CPU plugin compiles it and then cannot reorder the
dynamic-shaped state at the Assign. On the SERVED path this construct is what
`PagedCausalConv1DFusion` consumes (P2: `PagedCausalConv1D` ×3 after the
pass), so the raw op never reaches a plugin there. The construct is therefore
correct FOR the pass and unrunnable WITHOUT it, and the control form of the
boot — compile the stateful graph directly — is gone with it.

### What the device wrote as increment 5's spec, in the order it will be met

1. **The n-gram table must leave the graph** (host-mmap gather, as serving
   reads it through `src/exec/ngram_table.h`) or be chunked under
   4,294,959,104 B. Until then no depth ≥ 2 compiles on either card, cap or
   no cap (P3, P3b, P6: the B60's whole-VRAM cap is below the object too).
2. **The minimum served depth is 4.** The pass refuses anything without an
   SDPA (P1), and the emitter's first full-attention layer is index 3. A
   depth ladder for the served path starts at 4, never at 1.
3. **There is no unfused control any more.** The stateful conv construct is
   pass-only (P5, localised above). A "does the structure light up" witness
   for a card has to be the FUSED graph — which needs (1) first — or a
   static-length variant of the construct that the pass still matches, which
   is a design question and not decided here.
4. Only after (1)–(3) reach a forward do the handoff's three suspects become
   measurable: `inputs_embeds` is fed and not declared (the contract cell's
   `NOT DECLARED`), `input_ids`/`ngram_row_ids`/`conv_mask` are declared and
   never fed, and the query block is static in T. None of them was reached
   in this window and none is fixed by it.

Not done, on purpose: no lever pulled (`enable_large_allocations`), no fill,
no emitter change, no C++ change, no probe past a refusal. The
`ZOMBIE after p1-d1-pass` line in the ladder log is the sweep's `pgrep -f`
matching the launcher shell whose command line carried the pattern (the
self-match class already on record), not a leftover process; every later leg
swept clean and both cards were free at the end.


## 4.7 INCREMENT 5 — MAKE A DEPTH COMPILE, PREDICTED BEFORE IT RAN (2026-09-13)

Written with **no card output in existence** for the tree it predicts. The
device's spec from §4.6 was three lines: (a) the n-gram table leaves the graph
as a constant or is chunked under the A770's cap; (b) the minimum served depth
is 4; (c) there is no unfused control. What this increment did about each,
device-free, before the window — and what the window is asked to falsify.

### (a) The table — chunked CONSTANTS falsified first, by arithmetic

The route sketch named "chunk the table into sub-cap constant objects, row ids
indexing the slice" as the smallest change. It is dead before any card is
touched, on three numbers this manifest already holds:

| term | value | provenance |
|---|---|---|
| the table, any way it is cut | 25,600,122,880 B | `RUN@be57428` §4.4, `RUN@8a84598` §4.6 |
| A770 VRAM, total | 16,225,243,136 B | `RUN@e78812d` §8 enumeration |
| B60 VRAM, total | 24,385,683,456 B | `RUN@e78812d` §8 enumeration |

A GPU-plugin constant is device-resident, whole. Six objects of 4.27 GB are
still 25.6 GB of device memory, and neither card has it: the table exceeds the
A770's VRAM by 9.4 GB and the B60's by 1.2 GB **whether it is one object or
six**. The only way "chunked constants" could compile is if the kernel driver
evicted the chunks to system memory under pressure — which is not a route
anyone designed, and is the mechanism behind this host's recorded freeze class
(the xe CAT-error item: a direct-submission BO evicted under VRAM pressure).
That is why it was falsified on paper and NOT probed on the card: the probe
IS the hazard. `UNTESTED`, deliberately: whether the plugin or driver spills a
chunked-constant graph to host memory, and at what cost.

**What landed instead: the table is PARAMETER PORTS.** `ngram_table.K`, u8
`[rows_K, 80]`, one per chunk under the A770 cap, six of them at real
geometry — five of 53,686,272 rows (4,294,901,760 B, 57,344 B under the cap)
and a last of 51,570,176 — bound once per request from host memory. The
gather stays in the graph (`q4e.serving_shape.ngram_chunked_gather`: chunk id
and local row derived from the fed id in i32, one Gather per port at a
clamped index, one Select per chunk, nibbles unpacked low-first after the
gather). This is the host-mmap tier the served runtime already reads through
`src/exec/ngram_table.h`, handed to the compiled graph instead of gathered
beside it. Three facts from the pinned plugin source decided the form, each
one a thing the window checks rather than trusts:

- the per-object cap is checked in `engine::check_allocatable` BEFORE the
  allocation type is looked at, so a USM-host object is capped the same as a
  device one → chunk under the cap even though it lives on the host;
- `SyncInferRequest::allocate_inputs` defers a STATIC input's allocation
  ("reserve a null slot; materialized lazily or replaced by set_tensor()") and
  allocates a dynamic one eagerly → the ports are static, so creating the
  request allocates none of the 25.6 GB on the device;
- `prepare_input` shares a USM-host tensor from the device's own context with
  the graph without a device copy only when `!convert_needed`, and the
  precision pipeline rewrites a u4 Parameter to u8 anyway → the ports are u8
  bytes of a row, not the declared u4.

Device-free gates, all green on this tree and all RED on `37d9b33` (run on the
dev host, CPU, one extract per tree):
`test_no_constant_of_the_graph_exceeds_the_per_object_cap` (red: exactly
`ple/ngram_table_u4 [320001536,160] u4 = 25,600,122,880 B`),
`test_the_ngram_table_travels_as_ports_that_partition_the_vocabulary`,
`test_the_chunked_gather_is_the_whole_table_gather` (numeric, CPU, every chunk
edge; mutations hi-nibble-first 306/320 wrong, chunk-from-local 184/320
wrong), and the two contract cells whose recorded sets moved.

### (b) Depth 4 is the ladder's first rung — nothing predicted for 1–3

Unchanged from §4.6 P1: the pass refuses without an SDPA. The window runs no
leg below 4.

### (c) The witness is the FUSED graph — decided, with the alternative measured

The static-length alternative was checked device-free rather than left as "a
design question": a short-conv Variable of `[1, conv_dim, K]` with the beam
Gather dropped **still matches** `PagedCausalConv1DFusion` (pass output at
depth 4: `PagedCausalConv1D 3, PagedGatedDeltaNet 3, PagedAttentionExtension
1`, every port), and the unfused depth-1 graph in that form **compiles and
infers on CPU** (absmax 0.0) where §4.6 P5's form died at the Assign. So a
control is one line away. It is NOT adopted, for three reasons: the served
path never runs the unfused graph, so a control witnesses nothing the served
path needs; adopting it changes a construct the ports table was read against
(the served artifact's own `[?, ...]` state plus beam Gather) for the benefit
of a leg the served path does not have; and if the fused graph compiles, the
control has nothing left to say — while if it does not, the static form is the
localisation tool, one line away, and gets used then. The witness for this
increment is the fused graph at depth 4 on the A770, which is also the
acceptance.

### The prediction — probe first, then the boot

The probe (`tools/probe_ngram_table_ports.py`) runs BEFORE the boot because
the boot cannot tell a wrong row from a right one: every weight there is an
unwritten page. The probe writes sentinel rows, a function of the GLOBAL row
index, at both edges of every chunk, and gathers them through the same
`ngram_chunked_gather` the IR emits.

| # | leg | prediction (written before the run) | dies if |
|---|---|---|---|
| Q1 | probe, A770, `--rows 4096 --chunks 3` | compile OK; 3 USM-host tensors; verdict **EXACT** (0 rows wrong) — the Select picks the right port across both boundaries on the card | any row wrong |
| Q2 | probe, A770, `--rows 53686272 --chunks 1` = 4,294,901,760 B, one object | allocation **accepted** (57,344 B under the cap); `usm_host` grows by 4.00 GiB and `usm_device` does NOT; verdict **EXACT**, including the row at byte offset 4,294,901,680 — `gather_ref.cl` indexes in `uint` and that offset is under 2³² | refused at alloc (the cap counts USM-host differently); or `usm_device` grows by ~4 GiB (a device copy — the shared path was not taken); or the top rows mismatch (index arithmetic overflow) |
| Q3 | probe, A770, `--over-cap` | **REFUSED** at `engine.cpp:319`, `requested 25600122880 bytes, but max alloc size supported by device is 4294959104 bytes` — the same text the constant drew | the whole table is accepted as one USM-host object |
| B1 | boot, pass, depth 4 | OK; census as §4.6 P2; the six `ngram_table.K` ports survive the pass untouched | the pass touches or drops them |
| B2 | boot, **compile, depth 4, A770**, served props (`KV_CACHE_PRECISION` u8) | **OK** — the acceptance of this increment. No allocation of the table happens here (static inputs are lazy) | any refusal, named |
| B3 | table binding | 6 USM-host tensors, 23.84 GiB, bound; `usm_host` +23.84 GiB, `usm_device` unchanged by them | `usm_device` grows by the table, or a binding is refused |
| B4 | served feed order | **first refusal at `set_tensor(inputs_embeds)`**: the IR declares `input_ids`, the forward feeds `inputs_embeds` (§4.6, the contract cell's `NOT DECLARED`). This is the France line's served-path signature | the served feed reaches `infer()` |
| B5 | `--probe` (input_ids in place of inputs_embeds, LABELLED) | infer OK, `finite=True`, `absmax=0.0000e+00`, argmax `0` at all five positions, RAW greedy token **0**; second infer OK | any refusal; any non-zero logit |
| B6 | decode probe | a `[1, 1]` block against the `[1, 5]` port is refused at `set_tensor` with the shape text | accepted |
| B7 | the same ladder on the **B60** | as B2–B6 | any difference between the cards, named |

**Not predicted**, on purpose: compile seconds, infer seconds, the host cost
of 23.84 GiB of USM-host allocation (whether the driver commits the pages on
allocation), peak host RSS. First readings.

**The §8 France line after this window**, if the table holds: the served-path
signature (B4) and the labelled probe's token (B5), both dated, in place of
§4.6's refusal. A token id `0` from zero weights is not an answer and is not
written as one; it is the first forward the served path's own compiled graph
has returned on this IR. Paris is still not dated by this increment — the
weights are unwritten pages — and the frontier order's three suspects (fed
inputs, the static query block, the rope span) become measurable the moment
B5 holds, which is what the order asked for.

### The commands, in order (one process per leg, `timeout -s KILL 1200` each)

```
# probe, A770 (reserved card) — Q1, Q2, Q3
<venv>/bin/python tools/probe_ngram_table_ports.py --device GPU.1 --rows 4096 --chunks 3
<venv>/bin/python tools/probe_ngram_table_ports.py --device GPU.1 --rows 53686272 --chunks 1
<venv>/bin/python tools/probe_ngram_table_ports.py --device GPU.1 --over-cap
# boot, depth 4 — B1..B6 on the A770, then B7 on the B60
<venv>/bin/python tools/boot_serving_shape.py --layers 4 --stage pass
<venv>/bin/python tools/boot_serving_shape.py --layers 4 --device GPU.1 --ids 760,6511,314,9338,369
<venv>/bin/python tools/boot_serving_shape.py --layers 4 --device GPU.1 --ids 760,6511,314,9338,369 --probe
<venv>/bin/python tools/boot_serving_shape.py --layers 4 --device GPU.0 --ids 760,6511,314,9338,369 --probe
```

### Measured — `RUN@6e3004e`, 2026-09-13 09:09:50Z–09:11:50Z, then the localisations

The ladder above, run verbatim on a `git archive` extract of `6e3004e` (tree
`d0b30232…`, hashed on both hosts), one process per leg, `timeout -s KILL
1200`, both serving units inactive throughout, plugin the venv's stock
`2026.4.0-22849`. Raw logs: `boot-6e3004e/` beside the extract, then
`localise/` for what followed. **Two predictions died, and both deaths were
localised the same morning to a mechanism each, with the mechanism measured
rather than narrated.**

| # | card | measured | matches? |
|---|---|---|---|
| Q1 | A770 | 3 chunks × 4096 rows: compile 0.3 s, `usm_host` +0.001 GiB, verdict **EXACT**, 0 rows wrong of 80 | **yes** |
| Q2 | A770 | 1 chunk × 53,686,272 rows = 4,294,901,760 B: allocation accepted in 0.26 s; `usm_host` 4.000 GiB, `usm_device` **0.000** after set_tensor AND after infer (no device copy — the shared path was taken); infer 0.002 s; verdict **MISMATCH, 32 rows wrong of 80** | **NO — falsified**, on the rows, not on the transport |
| Q3 | A770 | `create_host_tensor(u8, [320001536, 80])`: **REFUSED** in 0.05 s, `engine.cpp:319 … requested 25600122880 bytes, but max alloc size supported by device is 4294959104 bytes` | **yes** |
| B1 | — | pass OK at depth 4, `PagedCausalConv1D 3, PagedGatedDeltaNet 3, PagedAttentionExtension 1`, the six `ngram_table.K` ports present and untouched after the pass | **yes** |
| B2 | A770 | depth 4, served props: **`COMPILE FAIL after 18.23s peak_host_GiB=13.56 … plugin.cpp:54: \| map::at`** | **NO — falsified** |
| B2 | B60 | the same, `13.61s peak_host_GiB=10.87 … map::at` | **NO — falsified** |
| B3–B7 | both | not reached | — |

**Q2, the rows.** Deterministic: seed 11 gives the same 32 rows wrong whether
the sentinels are written before the bind or after one infer
(`--order bind-then-write`), and whether every byte of the chunk was touched
first (`--touch-all`: the wrong rows then read `0xEE` — the NEIGHBOUR row's
fill — instead of zeros, so the kernel reads a wrong row, not an unmapped
page). Seed 12: 35 of 80. Four 1 GiB chunks instead of one 4 GiB: 31 of 80.
The wrong set is not a threshold: row 26,804,367 is wrong and the edge row
26,843,136 just above it is right. The predicate that fits is **"the row id
is not exactly representable in f32"** — 240 of 240 probed rows across three
runs agree with it, none disagrees. The graph's in-graph decomposition
(`Convert i64→i32 → Divide → Multiply → Subtract`) is executed by the GPU
plugin in f32, which is exact only below 2²⁴ = 16,777,216, and the table has
320,001,536 rows. Measured the other way the same hour: with the chunk id and
the local id FED by the host and the graph doing only Equal / Select / Gather
(`--index select`), **0 rows wrong of 80** at one 4 GiB chunk and at four; the
per-chunk-tensor form (`--index direct`, Gather only) likewise 0 of 80. So the
index path carries no arithmetic any more: `ngram_row_ids` became
`ngram_chunk_ids` (i32) + `ngram_local_ids` (i64), split on the host where
the hash already lives. This is the CPU's i64 finding of q4e.ple's header,
met again on the card at a lower threshold.

**B2, the node.** Plugin verbose output is not compiled into this build, so
the graph was cut after named nodes (`--cut`) and each prefix compiled on the
B60: `ple/gathered` OK 0.8 s; `layer0/mixer_out` (paged conv + GDN) OK 4.1 s;
`layer0/out` (+ MoE) OK 6.7 s; `ple/out` OK; `layer1/out` OK 12.4 s;
`layer2/out` OK 6.6 s; `attn3/q_rope` OK; `attn3/k_rope` OK;
**`attn3/att_out` FAIL `map::at`** (u8 KV and f16 KV alike);
`layer3/mixer_out` FAIL. The PagedAttentionExtension node itself. Its inputs,
dumped beside the served artifact's after the same pass: 28 inputs on both,
same Parameters, same Constants, same rt_info — and **q/k/v `[1, 30720]`,
`[1, 2560]`, `[1, 2560]` on ours against `[?, ?]` on the served graph.** The
pass flattens the SDPA's operands with `Reshape [0, -1]`
(`state_management_pattern.cpp`, `q_reshape`): dimension 0 is kept as the
token axis, the rest folded. On the served graph that yields
`[tokens, heads·d]` because the pass also forces `input_ids` to `[-1]` +
`Unsqueeze(1)` and the graph is dynamic in both batch and sequence, so at
runtime the batch axis IS the token axis. On this emitter's static
`[1, heads, T, d]` it yields one token of 30,720 features. The two
linear-attention fusions have no such asymmetry — `PagedGatedDeltaNetFusion`
(`flatten_batch_length`) and `PagedCausalConv1DFusion` (`Reshape [-1,
hidden]`) flatten B·L into tokens themselves, which is why every prefix
through `layer2/out` compiled. **This is the "query block static in T"
suspect of the frontier order, in its true form: the SDPA's operands must be
token-major, and nothing else in the graph has to change for the pass.**
Fixed in the emitter at the one construct: q, k, v presented to the SDPA as
`[T, heads, 1, d]` (a transpose; the same bytes), the mask `[T, 1, 1, total]`,
the kv broadcast reading its batch off the input, the PA output transposed
back. After the fix the node's operands read `[5, 6144]`, `[5, 512]`,
`[5, 512]` — the served layout with T written in. The contract suite is
green on that tree (28 passed 1 skipped) and it goes back to the cards
below.

### The third death, and the layout rule it wrote

Token-major but STATIC operands (`[5, 6144]`) still drew `map::at` at the
same cut; token-major AND DYNAMIC (`[?, 6144]`, the token count read off
`position_ids`, which the pass rewrites to `[-1]`, and the operands gathered
along it — an identity permutation) compiled: cut `attn3/att_out` OK on the
A770 in 11.6 s and on the B60 in 8.9 s, then the whole depth-4 graph. Every
served PagedAttentionExtension runs with a dynamic token axis; this plugin
has no static path for it, and says so with `map::at` (the throw site sits
in a stripped `.so`; gdb caught 770 `out_of_range` throws in the process, the
last two inside the plugin under `compile_model`, and could name none).

Then the first forward died at the PLE's additive join: hidden `[5, 5,
10240]`. After the pass the embedding comes out `[tokens, 1, hidden]` and
every static tensor of this graph is `[1, T, hidden]` — the same bytes — and
a binary op between the two BROADCASTS rather than refuses. Pinned with one
reshape where the token axis enters the static block. That is the whole
layout rule for a static-T graph under this pass: token-major and dynamic at
the SDPA operands, `[1, T, …]` everywhere else, one reshape at the seam.

### Measured after the fixes — `RUN@806b76f`, 2026-09-13 09:38–09:46Z, both cards

Tree `6c0106b6…` (`git write-tree` on both hosts), the tree the cards ran and
the tree commit `806b76f` carries. One process per leg, both serving units
inactive throughout. Logs: `boot-wt-final2/` beside the extract.

| # | card | measured | matches the prediction? |
|---|---|---|---|
| Q1/Q2 | A770 | emitter's own gather (`--index select`): **EXACT**, 0 rows wrong of 80 at 1 × 4 GiB and at 4 × 1 GiB | **yes**, after the index fix |
| B1 | — | pass OK; the ports after: `input_ids[-1]`, `position_ids[-1]`, `ngram_chunk_ids[1,5,16]`, `ngram_local_ids[1,5,16]`, `conv_mask[1,5]`, six `ngram_table.K`, the nine index ports, `key_cache.0`/`value_cache.0`, three `conv_state_table.N`, three `gated_delta_state_table.N` | **yes** |
| **B2** | **A770** | **`COMPILE OK 19.30s` (15.3 s / 16.7 s on two earlier runs of the same code), `prec=float16`, `peak_host_GiB=10.82`, `device_resident_GiB=7.86`** | **yes — the acceptance** |
| B3 | A770 | six USM-host tensors, 25,600,122,880 B, bound in 1.91 s; `usm_device` 7.86 → **7.86** GiB (no device copy), `usm_host` 0 → 23.85 GiB; peak host RSS unchanged by the binding (the driver does not commit the pages on allocation) | **yes** |
| B4 | A770 | `SERVED PATH VERDICT: first refusal at set_tensor(inputs_embeds): … Port for tensor name inputs_embeds was not found.` | **yes** |
| B5 | A770 | `--probe` (input_ids for inputs_embeds, labelled): **`INFER OK 3.161s out(1, 5, 248320) finite=True absmax=0.0000e+00`**, argmax `[0, 0, 0, 0, 0]`, RAW greedy token **0**; second infer **0.031 s**; `usm_device` still 7.86 GiB after infer | **yes** |
| B6 | A770 | a 1-token block is **ACCEPTED at `set_tensor`** (the port is `[?]` after the pass) and refused at **`infer`** by the first baked reshape: `Reshape_18 … input {[1,1,10240]} … Requested output shape [1,5,10240] is incompatible` | **no — the refusal moved one stage later**; the query block is static in the GRAPH, not at the port |
| B7 | B60 | compile 11.82 s, resident 7.87 GiB; table bound 1.94 s; the same served refusal; probe `INFER OK 0.044s`, absmax 0, token 0; second infer 0.017 s; the same decode refusal | **yes** |
| — | both | no leftover process after any leg (the `ZOMBIE` lines in two ladder logs are the launcher shell matching its own command line, the self-match class already on record; a `pgrep -a python` count after the last leg reads 0) | — |

First readings, not predictions: build 3.1–31.8 s at depth 4 (the 24.8 s and
31.8 s readings came while the CPU suite ran beside them); the A770's first
forward pays 3.16 s of kernel jit against the B60's 0.044 s, and the second
forward is 0.031 s / 0.017 s; `declared_GiB` at depth 4 is 15.75 without the
table (the u4-ceiled figure) against 63.43 with it.

**What the forward is and is not.** Every weight is an unwritten page, so
`absmax = 0` and token `0` are the structure lighting up on the served path's
own compiled graph — the first time that has happened for this IR — and not
an answer. The gathered table rows are whatever the USM-host allocation held
and are multiplied into zero projections. Paris is not dated by this window.

**The never-fed set after this window** (the contract cell asserts it):
`input_ids` (the forward feeds `inputs_embeds`), `conv_mask`,
`ngram_chunk_ids`, `ngram_local_ids`, and the `ngram_table.` family (the
runtime has no site that binds the mapping to a compiled model's ports).
Feeding them is the next increment, and every one of them is now reachable
by a forward on a card.

## 4.8 FEED-THE-PORTS — THE FIRST REAL-WEIGHT FORWARD, PREDICTED BEFORE IT RAN (2026-09-13)

Written with **no real-weight output in existence** for this tree. What the
increment changed before any card was touched, each with its device-free gate:

- **dynamic in T across the class.** No port, reshape or slice of the
  serving-shape graph carries the block length: the parity emitters take
  `seq_len=None` and build with `-1`, the PLE conv's taps end in negative
  stops, the GDN Loop's trip count and buffer come off `ShapeOf`, the SDPA
  operands are token-major and dynamic (§4.7), the conv construct slices its
  state with negative indices and NO `ShapeOf` (the pass's
  `TotalSequenceLengthPattern` matched the old `ShapeOf → Gather` over the
  conv concat and threw once the length went dynamic). Every parity suite is
  green under the same emitters — the change is structure-only there.
- **`inputs_embeds` is the port** the served forward feeds
  (backend_ov.cpp:6153), embedded on the host; the embedding weight left the
  graph. The served feed order is accepted end to end now.
- **one rope table pair for the full context**, `rope/cos` and `rope/sin`,
  `[262144, rotary]`, shared by every full-attention layer; the cell that
  pinned the T-only table went red and was replaced by the cell that gates the
  shared one.
- **the table rows are the GGUF's own IQ4_NL bytes** (90 a row: five blocks
  of an f16 scale and 16 nibble bytes), decoded AFTER the gather with exact
  ops (integers below 2²⁴, a 32-entry power-of-two table, the codebook by
  Gather). `test_the_in_graph_iq4nl_decode_is_gguf_pys_bit_for_bit`: equal
  to gguf-py's `dequantize(raw, IQ4_NL)` on 128 rows including subnormal and
  negative scales, no tolerance. The chunk partition follows: 47,718,400 rows
  (4,294,656,000 B) a chunk, **seven** ports, 26.82 GiB in all.
- **the dense fill.** `build_serving_shape_ir(feed=GgufFeed)` writes every
  dense weight of the built layers, the PLE, the final mixer and the head
  from the shards through `feed.fitted` (the same keys the parity suites feed
  by), the expert bodies through the existing `ExpertFiller`; the driver
  embeds the prompt from `token_embd` and binds the table from the shard's
  memmap, one copy into the USM-host chunks.
- **the driver feeds what the runtime cannot yet**: the two id ports from
  `q4e.ngram_ids` (the generator moved out of the PLE parity cell; validated
  there against the committed Link-3 vectors) split at the port partition,
  and `conv_mask` = ones. Labelled as the driver's on every line.

### The prediction — the France prompt at depth 4, real weights

| # | leg | prediction (written before the run) | dies if |
|---|---|---|---|
| R1 | build with `--shards`, depth 4 | dense fill writes **95** tensors (3 GDN layers × 22 + 1 attention layer × 19 + PLE 6 + final mixer 3 + head 1); expert fill 4 layers × 3 kinds; build completes | a key the feed cannot map (raises by name), or a shape the buffer refuses |
| R2 | compile, A770, served props | **OK** — the same graph as the zero-weight boot; weights do not change the structure | a refusal |
| R3 | table bind | seven USM-host chunks, 28,800,138,240 B, copied from the shard memmap; `usm_device` unchanged | an allocation refused, or the host OOM-kills the process (the budget: ~27 GiB pinned + the compile peak + the arena's dirty pages against 48 GiB) |
| R4 | the France prompt, served feed order + the driver's id/mask feeds | `INFER OK`, logits `(1, 5, 248320)`, **finite, not all zero, absmax > 0** | any non-finite, or all-zero logits (a fill that did not land) |
| R5 | the greedy token | **NOT `Paris`** and not predicted: four of forty-eight layers produce a real distribution, not the model's. The id and its string are recorded raw | — (nothing here is falsifiable except finiteness) |
| R6 | a 1-token decode block on the same request | `INFER OK`, logits `(1, 1, 248320)`, finite | a shape refusal anywhere: the graph would still be static somewhere |
| R7 | one long block on a fresh request: **`--long 1600` on the A770, `--long 2100` on the B60** | `INFER OK`, finite; positions up to 1,599 / 2,099 gather inside the shared rope table. (R7 was first written as 2,100 on both cards and was FALSIFIED by the zero-weight run before this section was committed: the A770 refuses a 2,100-token block at `engine.cpp:319`, `requested 5505024000 bytes` = the tiled MoE's `[512, 2100, 2560]` f16 intermediate, over the 4,294,959,104 B cap. That makes the A770's prefill-block ceiling under this lowering **1,638 tokens**; 1,024 and 1,600 ran, 2,100 did not. The B60 ran 2,100. Rewritten here rather than edited away — the increment's first measurement was the prediction's own falsification, on the record.) | a refusal, or a non-finite logit |
| R8 | the B60 | the same as R2–R6 | a difference between the cards |

**What R4/R5 are and are not.** The first real-weight logits of this IR on
the served path's own compiled graph. Not the model's answer: forty-four
layers are missing, so the argmax is a number with a face and no meaning.
The KLD gate does not read them (served path only).

**Not predicted**: every timing, the host peak, the fill's wall time.

### Measured, run 1 — `RUN@b88f275` on the A770 and B60, 2026-09-13 11:05Z — RETRACTED on the PLE rows

The ladder ran on the prediction commit's tree with `--shards /flash-model`.
**It hashed the n-gram rows with the wrong constants**: the driver passed the
PLE ordinal the parity cells use on both of their sides (1), and the
reference derives a PLE layer's constants from its ordinal among the PLE
layers (`modeling_qwen4_exp.py:1268`, `ple_layer_ids.index(layer_idx + 1)`),
which for the real model's one PLE layer is **0** — as the served loader
already derives it. So run 1's PLE gathered real rows of a wrong hash. Its
logits are real-weight logits of a graph with one wrong input, kept here as
the first reading and superseded by run 2 below; everything that does not
depend on the PLE rows stands.

| # | card | measured (run 1) | prediction |
|---|---|---|---|
| R1 | A770 | dense fill **95 tensors, 3.64 GiB written**; expert fill 12 bodies, 4 layers, 5,387,059,200 B; arena written 5.02 GiB; build **410 s** | **yes** (95 exactly) |
| R2 | A770 | compile OK 12.95 s, `prec=float16`, resident 6.79 GiB; peak host 27.67 GiB (the fill's dequant temporaries, before the table) | **yes** |
| R3 | A770 | seven chunks copied from the shard's memmap in 3.6 / 3.6 / 3.5 / 4.3 / 3.7 / 3.5 / 2.6 s, 28.33 s in all; `usm_host` 26.83 GiB, `usm_device` unchanged; host during the forward: 21 GiB used, 26 available — no OOM | **yes** |
| R4 | A770 | `INFER OK 7.268s out(1, 5, 248320) finite=True absmax=1.0346e+01`; second forward 0.035 s | **yes** (finite, non-zero) — with the wrong PLE rows |
| R5 | A770 | argmax per position `[2042, 65606, 2972, 18230, 220]`; last position top 5: `220='Ġ':5.614, 98390='å´ĩ':5.357, 115899='ä¸ĢåŃĹ':5.188, 55072='_subplot':5.062, 59126='ĠFeet':5.060` — a space, not Paris | as written: not Paris, not predicted |
| R6 | A770 | 1-token block `INFER OK 0.127s out(1, 1, 248320) finite=True` | **yes** |
| R7 | A770 | `--long 1600`: `INFER OK 17.829s out(1, 1600, 248320) finite=True absmax=1.8620e+01` | **yes** |
| R8 | B60 | build 433.78 s (the same 95 tensors, 12 bodies); compile 9.58 s, resident 6.79 GiB; table bound in 49.28 s (chunks 5 and 6 at 15.2 / 7.9 s: the shard pages were being evicted under the 28.78 GiB host peak); `INFER OK 8.701s`, the SAME argmax `[2042, 65606, 2972, 18230, 220]`, last-position top 5 `220='Ġ':5.588, 98390:5.314, 115899:5.163, 59126:5.071, 55072:5.052` (the A770's within f16 noise, two of the five swapped in order at the third decimal); 1-token block OK 0.049 s; `--long 2100` `INFER OK 24.127s out(1, 2100, 248320) finite=True absmax=1.8514e+01` past the 2051 boundary | **yes** — the cards agree |

### Measured, run 2 — `RUN@2413cab` (tree `3e50948c…`), 2026-09-13 11:24–11:42Z, both cards — THE MEASUREMENT

The same ladder with the hash ordinal derived from the config
(`q4e.ngram_ids.ple_ordinal` → 0). *(Amended 2026-09-13, REVIEW
3b5df79..6743ffb F1: "derived from the config" did not happen in this run —
`real_config()` carried no `ple_layer_ids` and `ple_ordinal` returned 0 from
a fallback branch, for every layer index. The 0 is right by an independent
witness: the GGUF stores its own PLE hash constants and the ordinal-0
derivation matches all of them bit for bit, ordinal 1 matches none; the
fallback is removed and the witness is a shards cell in the F1 commit. No
number in this table moves.)* The row ids moved (min 7,226,134, max
316,425,755 against run 1's 4,023,550 / 317,350,792), the logits moved with
them, and both cards agree at the last position to the third decimal.

| # | card | measured (run 2) | prediction |
|---|---|---|---|
| R1 | both | dense fill 95 tensors, 3.64 GiB; expert fill 12 bodies; build 431.35 s (A770 leg) / 418.26 s (B60 leg) | **yes** |
| R2 | A770 / B60 | compile OK 10.15 s / 6.57 s, resident 6.79 GiB | **yes** |
| R3 | A770 / B60 | table bound in 33.03 s / 29.49 s, 26.82 GiB USM host, `usm_device` unchanged; peak host 27.67 GiB; no OOM | **yes** |
| R4 | A770 | **`INFER OK 10.383s out(1, 5, 248320) finite=True absmax=9.1495e+00`**; second forward 0.035 s | **yes** |
| R4 | B60 | `INFER OK 6.750s … finite=True absmax=9.1205e+00`; second 0.027 s | **yes** |
| R5 | A770 | argmax per position `[3181, 14160, 6387, 18230, 5613]`; last position top 5: **`5613='ramework':3.586`**, `67416='SetName':3.548`, `215412='antul':3.541`, `37124='ĠsetIs':3.522`, `55541='ambre':3.512` | not Paris, as written; a token with a face and no meaning |
| R5 | B60 | argmax `[53108, 14160, 6387, 18230, 5613]` — position 0 differs (near-tied logits under f16), positions 1–4 identical; last position `5613='ramework':3.573, 67416:3.557, 215412:3.537, 37124:3.529, 55541:3.506` | the cards agree where the margin exists |
| R6 | A770 / B60 | 1-token block `INFER OK` 0.103 s / 0.054 s, `(1, 1, 248320)`, finite | **yes** |
| R7 | A770 | `--long 1600`: `INFER OK 17.194s out(1, 1600, 248320) finite=True absmax=1.8889e+01` | **yes** |
| R7 | B60 | `--long 2100`: `INFER OK 28.597s out(1, 2100, 248320) finite=True absmax=1.8888e+01`, past the 2051 boundary, rope positions to 2,099 | **yes** |
| R8 | B60 | as above | **yes** |

**What this is.** The first real-weight logits of the serving-shape IR on
the served path's own compiled graph, on both cards, with every port fed:
the served feed order end to end, the driver's id and mask feeds labelled as
its own, the real table bound from the shard's bytes. Every row of the
prediction held except the one the prediction could not make (R5), and the
long-block row it had already had to rewrite.

**What it is not.** Forty-four layers are missing. `ramework` is the argmax
of a four-layer prefix of a forty-eight-layer model and says nothing about
Paris. The KLD gate does not read these logits (served path only; the
served path for THIS IR is the runtime's binding site, which compiles and
passes its cells and has not run on a card — no artifact directory for the
IR exists yet).

**Host cost, first readings.** The fill's build is 7 minutes (gguf-py
dequantisation of 95 dense tensors and the u4 packing of 12 expert bodies);
the peak host is 27.67 GiB before the table and the table adds 26.82 GiB of
pinned USM host; the container's 48 GiB held with the shard pages evicting
under it (run 1's B60 leg paid 15.2 s and 7.9 s for two chunk copies for
exactly that reason).

## 4.9 THE SERVED PATH — the binary on an artifact directory, depth 4 (2026-09-13)

Until this section the serving-shape IR had been fed on a card by one thing:
the labelled probe (`tools/boot_serving_shape.py`), which transcribes the
served load sequence stage by stage. The runtime's own binding site
(`bind_ngram_ports` / `feed_ngram_ports`, §4.8) compiled and passed its cells
and had never run, because nothing `load_artifact` would admit existed. The
France and KLD gates are served-path gates (§7, §8); this section makes them
reachable.

**The artifact directory** — `tools/export_serving_artifact.py --layers 4`,
`RUN@wt+6743ffb` (this increment's working tree over 6743ffb; the commit that
carries this section is the tree), written 2026-09-13 12:16–12:27Z on the dev
host, `serving-shape.json` beside it:

| file | what | size |
|---|---|---|
| `openvino_language_model.xml/.bin` | `build_serving_shape_ir(feed=GgufFeed, n_layers=4)`, real weights, `compress_to_fp16=False`; 1,723 nodes; dense fill 95 tensors 3,906,155,136 B, expert fill 12 bodies 5,387,059,200 B; build 556.8 s, save 82.1 s, peak host 21.60 GiB | 9,427,885,993 B (8.78 GiB); xml sha `2910a860bf9dc6bb` |
| `openvino_text_embeddings_model.xml/.bin` | `token_embd.weight` dequantised f32 [248320, 2560], Gather by `input_ids` [1, T], T dynamic | 2.37 GiB |
| `openvino_tokenizer.*`, `openvino_detokenizer.*`, `tokenizer.json`, `tokenizer_config.json` | passthrough from the qwen3.8 artifact, admitted after the tool compared all 248,077 ids tokenizer.json defines against the GGUF's `tokenizer.ggml.tokens`: 0 disagree (the 243 GGUF ids past the last defined one are the padded tail); the France prompt tokenises to `760,6511,314,9338,369` on both | sha `87a7830d63fcf43b` |
| `chat_template.jinja` | the GGUF's own `tokenizer.chat_template` | 9,993 chars, sha `12827f24b742ea4e` |
| `config.json` | the real geometry at `num_hidden_layers 4`, `layer_types` 3 × linear_attention + qwen_sparse_attention, the n-gram five, `ple_layer_ids [2]`, `eos_token_id 248044` = the GGUF's `qwen4exp.ple.eos_token_id` (llama.cpp hashes with that key; `tokenizer.ggml.eos_token_id` is 248046 = `<|im_end|>`, a different token) | |
| `generation_config.json` | `eos_token_id [248044, 248046]` — the loader takes `eos_ids.front()` as the hash boundary, so the PLE eos is first; both stop generation | |

**What the served path needed that the probe did not** (each a refusal or a
throw the code would have produced, read off the code before the run; the
first two were measured on the probe before the branch existed):

1. `gguf_geometry` refuses every architecture but `qwen35` by name, so `--gguf`
   cannot open the shard for the table. `--ngram-gguf FILE` opens the shard
   for `per_layer_token_embd.weight` alone: no geometry check, no template
   rewrite; refused with `--gguf`, on the stateful path, and when the IR
   declares no `ngram_table.K` port (config cells, red-first on the
   `--gguf` + `--ngram-gguf` pair).
2. `position_ids` is fed as `[sections, n]` and `paged_sections_` is read off
   `ps[0].get_length()`; this IR's post-pass port is the flat `[-1]`. Fed at
   the port's rank now (`paged_pos_flat_`).
3. The loader counted only the literal `full_attention`; the pin's
   `qwen_sparse_attention` reported 0 attention layers of 4 and the registry
   split could never sum. `artifact_counts_qwen_sparse_attention_as_full_attention`
   read `n_gdn_layer 4` before the loader learnt the name (red), 3 after.
4. The allowlist: `qwen3.8-flash-next-d4` / `qwen38-flash-next-d4-ov`, hashes
   read off the directory above; the registry cells went from 6 to 7 entries.

Ladder on the exact tree, the dev host, `ARCINT_GGUF_REAL` = shard 2: **532 cases
run, 0 failed, 0 skipped** (`RUN@wt+6743ffb`, ladder-1.log).

### The command, one card per process

```
# RUN@wt+6743ffb  (~/inc7/served_d4.sh GPU.1 a770-d4, then GPU.0 b60-d4)
arcint --model /models/ov/qwen38-flash-next-d4-ov --device GPU.1 \
  --ngram-gguf <shard 2> --prefill-chunk 1536 --n-ctx 8192 --parallel 1 \
  --no-logits-slice --port 8091 -v
# /props until 200 (the only gate), then
POST /v1/completions {"prompt":"The capital of France is","max_tokens":8,"temperature":0}
POST /v1/chat/completions {"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":32,"temperature":0}
```

`--prefill-chunk 1536`: under the A770's 1,638-token block ceiling (§4.8
R7) and a multiple of the 16-token KV page. `--no-logits-slice`: every
prefill row comes back, which the KLD replay (§4.11) needs; the France
question does not care.

### Measured — the served-path signature, `RUN@wt+6743ffb`, 2026-09-13 12:29–12:48Z, A770

The binary loaded the artifact, compiled, bound the table through its own
site — and the first forward never returned. Three legs, in order:

| leg | what | measured |
|---|---|---|
| A770, defaults (`served-a770-d4`) | embeddings graph compiled on GPU.1 beside the paged model | `load_artifact` admits; `--ngram-gguf` opens shard 2 (402 tensors); paged model compiled in **21.8 s, 6.79 GiB resident**; embeddings on GPU.1, drafters 1.18 GiB; KV ports i8 for u8; **table bound: 7 ports, 26.82 GiB USM host in 76.7 s, id ports declared, conv_mask declared, hash ordinal 0** (the runtime's binding site, on a card, first time). Then the activation probe's first forward (128 tokens of id 0, chunk floor): **no return in 9 min 20 s**. Main thread in `xe_wait_user_fence_ioctl → __flush_workqueue` (`/proc/<pid>/stack`), 28 threads in futex wait, both GPUs at `act_freq 0` and rising idle residency, no GT reset in dmesg, `fdinfo` not read on this leg. `SIGKILL` took it (12:40:33Z). **The served-path signature, as the device wrote it: a user-fence wait on an idle card.** |
| A770, `--emb-device CPU` (`served-a770-d4-embcpu`) | the one change: the embeddings gather runs on the host | boots: table bound in 27.2 s (page cache warm); reservation `weights+graph 6.79 + activations 3.62 (chunk 512)` — the climb settled the served chunk at **512**, not the configured 1536; `/props → 200` at 12:43:29Z, 2 min 21 s after launch; `fdinfo`: `drm-resident-gtt 27.35 GiB` (the table, all of it resident), `drm-resident-vram0 10.49 GiB`. |
| → France, raw completion, greedy, 8 tokens | `"The capital of France is"` (5 ids) | **`rameworkenessooter5ussionxigy引`** — first token `ramework` = id 5613, **the probe's token (§4.8 R5), reached through the served path**: the C++ hash (ordinal 0, eos 248044), the C++ split, the runtime-bound table, the artifact's embedding. Cold: prefill 5 tok 8.15 s (`restore 7.28 s` — the first request's kernel JIT lands in that column), decode 8 tok 0.63 s (12.7 t/s). Warm, same prompt: identical text, prefill 0.05 s, **decode 38.2 t/s**. |
| → France, chat form, 32 tokens | the template renders 64 prompt tokens | `reasoning_content`: `像在\n inté rendschedulers 2ingham aalii/sogos …` — 32 tokens of the same nature; prefill 64 tok 0.48 s (134 t/s), decode 24.6 t/s. |
| B60, defaults (`served-b60-d4`) | embeddings on GPU.0 beside the paged model — the A770's hanging configuration | **boots**: table bound in 44.1 s; activation fit `0.290 GiB fixed + 6849.3 KiB per chunk token`, served chunk **1024** measured 6.98 GiB; reservation 6.79 + 1.18 + 6.98 + 0.25 of 22.71 GiB; `/props → 200` at 12:50:35Z, 2 min 16 s after launch. France raw, greedy 8: **`rameworkenessooter5rawd.githubusercontent /riet`** — the first four tokens the A770's (`ramework`, `eness`, `ooter`, `5`), the fifth diverges (`rawd` against `ussion`; §4.8 saw the same class of near-tie under f16 at position 0). Cold prefill 5 tok 3.08 s, decode 17.2 t/s; warm: identical text, prefill 0.03 s, **decode 47.1 t/s**. Chat form: `reasoning_content` `像在\n mere (把头/head/head/head#__otto …`, 46.6 t/s. Teardown clean, leftover 0. |

> **RETRACTED the same hour, and re-measured.** The operator's word at
> 12:57Z: *the reviewer was working on the host in parallel during the
> first leg.* The hang was therefore never attributable to the embeddings'
> placement — one difference was named while a second, unnamed one was
> live. Re-run, quiet host, `served-a770-d4-rerun`, identical command to the
> first leg (embeddings on GPU.1), 12:57:49–12:59:57Z: **boots** in 1 min
> 58 s (paged model 46.9 s, table 45.0 s); activation fit `−1.101 GiB fixed
> + 10975.6 KiB per chunk token`, served chunk 512 measured 4.26 GiB; France
> raw **`rameworkenessooter5ussionxigy引`**, byte for byte the emb-CPU
> leg's; cold prefill 0.41 s, warm decode **37.9 t/s**; chat form 34.4 t/s;
> teardown clean. So: the served path boots and answers on the A770 in the
> default configuration; the first leg's fence wait on an idle card is
> recorded as what it was — a run taken while another session held the
> host — and the sentence "the hang is the A770's with this artifact" is
> withdrawn. `--emb-device CPU` is not a required configuration; it remains
> a measured, working one (the KLD leg of §4.11 ran under it).

The driver script of the emb-CPU leg died between `/props` and its first
POST (its `run.log` ends at the props line; no process, no trace); the
France requests above were posted by hand against the still-running server
and appended to the same log, marked as such. The script now traces itself
(`set -x`) instead of aborting.

**What this is.** The runtime's binding site ran on a card, every port fed
by the runtime and not by a driver, and the served path produced the token
the probe had produced. **Paris is NOT dated** — `ramework` is a
four-of-forty-eight-layer prefix's argmax, as §4.8 said, now on the served
path. **What it is not**: an explanation of the first leg's hang — see the
retraction under the B60 row: that leg ran while another session held the
host, and the identical command re-run quiet boots and answers.

## 4.10 FULL DEPTH — THE ARITHMETIC BEFORE THE FILL, PREDICTED BEFORE ANY 48-LAYER CARD OUTPUT (2026-09-13)

Operator CR of 2026-09-13 folds Berlin into 0.5.0: 48 layers, "experts
host-bound, slow is fine", the France question and the KLD gate at full
depth. This section is written with the depth-4 served run's numbers in hand
and **no 48-layer artifact built and no 48-layer compile attempted**. Every
row below is falsifiable by the fill or the card.

### What the depth-4 artifact measures about the full one (`RUN@wt+6743ffb`, serving-shape.json)

| term | per 4 layers, measured | 48 layers, arithmetic |
|---|---|---|
| dense f32 tensors | 95 = 3 × 22 (GDN) + 19 (attn) + 6 (PLE) + 3 (final) + 1 (head) | 36 × 22 + 12 × 19 + 10 = **1,030** |
| dense f32 bytes | 3,906,155,136, of which the head is 2,542,796,800 and the four layers + PLE + final 1,363,358,336 | 12 × 1,363,358,336 + 2,542,796,800 = **18,903,096,832 B (17.6 GiB)** f32; the plugin carries them as f16 on the card: **8.8 GiB** |
| expert bodies, u4 | 12 = 4 × 3, 5,387,059,200 B (1,346,764,800 a layer) | 144, **64,644,710,400 B (60.2 GiB)** — u4 is the card's own type, no halving left |
| rope tables | 2 × [262144, 64] f32, 134,217,728 B, shared | the same 134 MB |
| `.bin` | 9,427,885,993 B | ≈ 9,427,885,993 + 11 × (1,363,358,336 + 5,387,059,200) = **83,682,478,889 B (77.9 GiB)**, written to /models (1.7 TB free) |
| build wall | 556.8 s (7.3 s index + fill + 82.1 s save) | ≈ 12 × the per-layer share ≈ **95–110 min**; peak host ≈ the same 21.6 GiB (the arena is a file) |
| device-resident after compile | 6.79 GiB (A770 and B60 alike, §4.8) | dense f16 8.8 + experts 60.2 + rope ≈ **69 GiB** before one KV page or one activation |

Against the cards: A770 16,225,243,136 B (15.11 GiB), B60 24,385,683,456 B
(22.71 GiB). Against the host: 48 GiB total, and the table already takes
26.82 GiB of it as pinned USM host (§4.8 R3).

### The predictions

| # | leg | prediction (written before the run) | dies if |
|---|---|---|---|
| F1 | `export_serving_artifact.py --layers 48` | completes: 1,030 dense tensors, 144 bodies, a `.bin` within 2 % of 83.7 GB, build under 2 h, no host OOM (the arena is a file; the d4 peak was 21.6 GiB); config.json `num_hidden_layers 48`, 12 × `qwen_sparse_attention` | a key the feed cannot map, the host OOM-killer, or a `.bin` far off the arithmetic (a per-layer term this table does not know) |
| F2 | compile on the A770, served props | **REFUSED**: 69 GiB of constants do not fit 15.11 GiB. The device's words are either the per-allocation refusal at `engine.cpp:319` (no single object is over the 4.00 GiB cap — the largest is one gate_up body, 838,860,800 B — so this one is NOT expected) or an out-of-device-memory error from the allocator when the sum crosses the card; which of the two, and at which layer's constant, is what the run measures. UNTESTED alternative on the record (§4.7): the driver spills device allocations to system memory instead of refusing — then the compile proceeds until the host's 48 GiB is consumed. The run is fenced with a cgroup `MemoryMax` (systemd-run --user) so that outcome is a kill of the process with the fence's words, not a host freeze | the compile COMPLETES and a forward runs: then constants are not device-resident the way this arithmetic assumes, and full depth is reachable as built |
| F3 | compile on the B60 | **REFUSED** the same way at 22.71 GiB; 69 > 22.71 by a factor of three, so no chunk size or KV precision changes it | as F2 |
| F4 | the France question at full depth | **NOT REACHED through this artifact on these cards** — reached only if F2 or F3 die. The France line for full depth stays EMPTY in this section; the row that fills it is the model's own tokens, raw, in the measurement commit of whichever route runs 48 layers | — |
| F5 | "experts host-bound" as the CR words it | **does not fit either**, by arithmetic and before any code: 60.2 GiB of expert bodies as USM-host ports next to the 26.82 GiB table is 87 GiB against 48 GiB of host RAM; the table as a host-side gather (0 pinned) leaves 60.2 against ~46 usable. So a single compiled 48-layer graph with every expert port bound at once is refused by the host, not the card — the outcome F2's fence exists for | a host with more RAM, or a route that binds fewer than 48 layers' experts at once (a segmented forward streaming expert bodies per layer group, which is Lisbon's mechanism) |

**What this section does not predict**: timings of a refusal, which
constant the allocator names, whether the xe driver spills.

### Measured — `RUN@092df69` (the fill) and `RUN@wt+894af62` (the compile leg), 2026-09-13 13:03–15:13Z

| # | measured | prediction |
|---|---|---|
| F1 | `export_serving_artifact.py --layers 48 --tree 092df69`: build **4,914.5 s** (81.9 min; the fill was SIGSTOPped 14:26–14:50Z for the reviewer's card legs and that pause is not in the figure — the process clock stops with it), 17,354 nodes, **1,030 dense tensors 17,169,341,952 B (15.99 GiB)**, **144 bodies 64,644,710,400 B** (the arithmetic's number to the byte), save 2,141 s, `.bin` **81,948,724,009 B (76.32 GiB)** — 2.07 % under the 83.7 GB predicted (the per-layer dense term was over-estimated from four layers that included the PLE and the final mixer), so the "within 2 %" clause **dies by 0.07 points** and the count clauses hold; sha256 pass 277.9 s; peak host **32.81 GiB** (against the depth-4 run's 21.6 — the row "peak host ≈ the same" was wrong by 11 GiB; the arena's dirty pages and the 76 GiB save both charge the process); no host OOM. xml sha `3f574b776a8dd2c6`, allowlisted as `qwen3.8-flash-next` / `qwen38-flash-next-ov`. | **yes** on the counts, the time and no-OOM; **dies** on the 2 % `.bin` clause and the peak-host guess |
| F2 | A770, served binary, `--prefill-chunk 1536 --n-ctx 8192`, fenced `systemd-run --user --scope -p MemoryMax=44G` (`served-a770-d48`): `load_artifact` admits, `--ngram-gguf` opens shard 2, `read_model` + `SDPAToPagedAttention` pass, then **6 min 8 s in `compile_model`** during which `fdinfo` showed the constants going to **host memory, not the card**: `drm-resident-gtt` 5.3 → 0.4 → 1.1 → 6.4 → **11.8 GiB** (sawtooth: allocate, convert, release, re-allocate), `drm-resident-vram0` flat at **7.6 MiB** throughout. At 15:13:26Z the host kernel logged `arcint: page allocation failure: order:0, mode:0x104dc2(GFP_HIGHUSER\|__GFP_ZERO\|__GFP_RETRY_MAYFAIL)` from `xe_gem_create_ioctl` with **Mem-Info: shmem 9,767,873 pages (37.3 GiB), free 676,952 pages (2.6 GiB)** — the physical host, not the container — and the plugin threw **`[CL ext] Can not allocate 419430400 bytes for USM Host. ptr: 0, error: -6`** (`CL_OUT_OF_HOST_MEMORY`; 419,430,400 B is one down-projection body `[512, 2560, 640]` u4). arcint exited on the exception; no OOM killer fired; the cgroup fence **never engaged** (driver memory is not charged to it — the caveat measured in §4.9, now with consequence). Teardown clean, leftover 0. | **REFUSED, as predicted — but not by the mechanism predicted.** The prediction named the card's allocator; the device's words name the HOST: this plugin generation stages every constant of the compile in USM host memory before anything reaches VRAM, and 69 GiB of constants meet the host's ~46 GiB first. The "UNTESTED spill" alternative is therefore not an alternative: spilling is what the compile does, and the host is the ceiling it hits. The "dies if" (compile completes) did not occur. |
| F3 | B60 — **not executed, engineer's call, sustained by the frontier in the ba2d5de review, still the operator's to overrule**: the refusal is host-side and card-independent (the allocation that failed is USM Host; VRAM was untouched at 7.6 MiB on a card with 8 GiB free), the physical host was within 2.6 GiB of its OOM killer, and a second leg buys the same words at the same risk. The order said "then B60"; this row is where that step was not taken and why. (The word this row avoids is the manifest's marker word.) | — |
| F4 | **The France question at full depth was not reached.** The line stays EMPTY at full depth; §8's coherence row keeps the depth-4 served token. | **yes** |
| F5 | Confirmed from an unexpected side: the compile's own staging already needs the whole constant set in host memory, so "experts host-bound" is not a residency choice this plugin can be asked for on this host — it is the failure mode. A route that compiles fewer than 48 layers' experts at once is the only one left (see the handoff's CR). | **yes** |

**Numbers the fill produced that the next route needs** (the CYCLE ACCOUNT's constants): 144 bodies × 448,921,600 B = 64,644,710,400 B of u4 expert weights; 1,030 dense tensors 17,169,341,952 B f32; a 48-layer `.bin` of 81,948,724,009 B; 17,354 nodes; a fill of 82 min at 32.8 GiB peak host; and the plugin's compile-time staging of every constant in USM host memory, measured on the A770.

## 4.11 THE KLD GATE'S SERVED HALF — the instrument, red-probed at depth 4 (2026-09-13)

`tools/kld_served.py` (device-free cells in `tests/python/test_kld_served.py`):
`--replay` posts the pinned capture's own token windows to a running arcint as
token-id prompts (`/v1/completions` `prompt: [ids]`, one greedy token — the
form the API now accepts; batches stay refused), while the server appends
every paged forward's logits to `ARCINT_LOGITS_DUMP` (`"ARCLGT01"` + lane,
past, n, rows, vocab + rows × vocab f32; `--no-logits-slice` so rows == n);
`--compare` reconstructs the reference log-probs row by row (llama.cpp's
uint16 rows: f32 scale, f32 min log-prob, `q = round((logit − min) / scale)`,
transcribed from `perplexity.cpp` and pinned to the quantisation step by a
cell), log-softmaxes the served rows, and reports mean per-token
KL(P_ref‖P_served) over the recorded rows `n_ctx/2 .. n_ctx−2`, split at row
index 2051. REPORT ONLY; the 0.0599-nat bar (PROVISIONAL — inherited, §7) is
printed beside the means with its provenance sentence.

Reference: `/flash-model/kld-ref/qwen4exp-c2735-chunks2.dat`, sha
`af7993b7…`, llama.cpp master `56b9eb28`, n_ctx 2735, 2 windows, 1,367 rows a
window (684 below the boundary, 683 at or above), 91,266-byte corpus.

**Red probe, predicted**: through the depth-4 artifact the instrument must
read a mean KL FAR above 0.0599 (PROVISIONAL bar) in both regimes (forty-four layers are
missing; `ramework` is not a distribution over English) and an argmax
agreement near zero. A reading near the bar at depth 4 would falsify the
instrument, not pass the model. This prediction was written in the tree
before the run and **not committed before it** — the discipline of §4.6–4.8
was kept in letter for the card runs of §4.9/§4.10 and only in substance
here; said rather than hidden.

**Measured — `RUN@wt+6743ffb`, A770, `--emb-device CPU`, 2026-09-13
12:51–12:57Z (`kld-a770-d4`).** Server up in 4 min 31 s (table bind 138.8 s
this time — the shard pages had been evicted by the B60 leg's own bind);
served chunk 512. Replay: window 0, 2,735 ids, HTTP OK in 27.4 s (prefill
26.35 s = 103.8 t/s: graph 13.48 s, embed 12.87 s on the host), greedy
token ` ev`; window 1 in 22.1 s (21.92 s, 124.8 t/s), `xyz`. Dump
6,325,207,856 B, 17 records in 5 windows — the server's three load-time
probe windows and the two replays, which the compare selects by their
2,735-row length. Compare, 1,367 recorded rows a window (684 below the
boundary, 683 at or above):

| window | mean KL below 2051 | mean KL at/above 2051 | max | argmax agreement |
|---|---|---|---|---|
| 0 | 1.168541e+01 | 1.124390e+01 | 2.36e+01 | 0.0007 |
| 1 | 1.161744e+01 | 1.171603e+01 | 1.93e+01 | 0.0000 |
| **all** | **1.165143e+01** | **1.147997e+01** | | |

**The instrument reads red at depth 4, as predicted**: 11.65 / 11.48 nats
against the 0.0599 bar (PROVISIONAL), two hundred times over; argmax agreement 1 row in
1,367. Nothing here is a statement about the model. The same command against
a full-depth artifact is the gate's measurement — REPORT ONLY.

**The instrument's own floor — `RUN@wt+ba2d5de`, A770, defaults (embeddings
on the card, served chunk 512), 2026-09-13 15:54–15:59Z
(`kld-floor-a770-d4`).** REVIEW ba2d5de F1: the reviewer's B60 leg found two
identical replays of the same windows in the same process differing from row
14 / 261 onward (chunk 1024; KL(A‖B) 1e-3..5e-3 mean, max 0.24; argmax
differing in 207 and 115 of 2,735 rows). So the served logits are not
run-to-run deterministic at this length and a reading needs its floor
beside it. `--replay --repeat 2` (four 2,735-token prefills, 13.4–15.1 s
each = 182–204 t/s at chunk 512) and `--compare` pairing the last two
replays, same recorded rows:

| window | KL(A‖B) mean | KL(A‖B) max | argmax agreement | max \|logit diff\| |
|---|---|---|---|---|
| 0 | 4.289103e-04 | 9.6520e-02 | 0.9700 | 2.142 |
| 1 | 2.170477e-19 | 1.8088e-15 | 1.0000 | 0.000 |
| **all** | **2.144551e-04** | **9.6520e-02** | | |

Window 1's two replays are **bit-identical**; window 0's differ in 41 of
1,367 argmaxes with a max KL of 0.097 in one row. The floor is therefore
not a constant of the backend but a per-window event — which is the
reviewer's "the mechanism is an increment of its own" — and at 2.1e-4 mean
it sits 280× below the inherited bar while its max in one row is 1.6× the
bar. The reference means re-read against replay B: 11.65 / 11.48 (11.6835 /
11.2465 and 11.6180 / 11.7171 per window), unchanged to the third decimal
from the single-replay run. Every gate reading from now on carries this
row beside it; the mechanism is not narrated here.

### 4.11.1 THE FLOOR'S MECHANISM — measured, not narrated (WP0.1 rider, 2026-09-13)

**Instrument.** `tools/boot_serving_shape.py` on the depth-4 artifact with
the real table bound (`--artifact --shards`), the KLD capture's own window-0
ids as the prompt (`--capture … --take 1024`), the state rows zeroed before
every forward exactly as `zero_paged_rows` does (`copy_from` of a zero row —
the first chain read the *carried state* instead: 887 of 1,024 argmaxes
moved between two identical forwards, run 1, withdrawn), `--repeat N`
comparing every forward with the first and with its predecessor, a sha256
digest per output, `--cut` for the per-node localiser, `--plugin-prop` for
the plugin's own switches. A770 (GPU.1), f16 default, KV u8, 22 legs
`RUN@c836354` … `RUN@e53d75b`, 16:27–17:35Z, one process per leg, cards
free between legs.

**What the device wrote.** The output of an identical forward changes
**once, permanently, per event**, never back — a staircase of discrete
output classes (digests), each step starting at an earlier row than the
last; the step count and timing depend on the process, not the input:

| leg | kernel cache | forwards | classes (by digest) | steps at forward | first differing row | max \|logit diff\| | argmaxes moved / 1,024 | per-forward s before → after |
|---|---|---|---|---|---|---|---|---|
| L1 same request ×6 | warm-ish (first of the day) | 6 | 2 | #3 | 0 | 1.89 | 52 | 7.0 → 5.2 |
| L2 fresh request ×6 | warm-ish | 6 | 2 | #5 | 0 | 1.24e-4 | 0 | 9.5 → 5.4 |
| L9 same ×12 | warm (after L1–L8) | 12 | **1** | — | — | 0 | 0 | 6.7 → 5.2 |
| L10 T=512 ×12 | warm | 12 | **1** | — | — | 0 | 0 | 3.7 → 2.6 |
| L18 cold (`NEO_CACHE_PERSISTENT=0`) ×12 | cold | 12 | 2 | #11 | 0 | 1.25e-4 | 0 | 7.3 → 7.1 (no drop) |
| L20 cold again ×12 | cold | 12 | **4** | #3, #6, #12 | 84 → 16 → 0 | 1.42 → 2.15 → 1.83 | 18 → 57 → 67 | 7.0 → 5.2 at #4 |
| L22 f16 after the cold legs ×12 | partly warm | 12 | 2 | #4 | 0 | 1.24e-4 | 0 | 6.3 → 5.2 |
| L8 / L16 / L17 cuts at `layer3/mixer_out`, `layer2/out` (warm) | warm | 6–8 | 1 | — | — | 0 | — | flat |

Two magnitudes of step recur across legs: a **small** one (max \|diff\|
1.24–1.25e-4, no argmax moves; L2, L18, L22) and a **large** one (1.4–2.2 in
one logit, 18–67 argmaxes of 1,024 at depth 4; L1, L20). The first forwards
of two cold processes (L18 `44f5d3c3beba`, L20 `5689b99aa3e2`) already
differ from each other and from a warm process's (L9 `1f7b36ad7284`), so
"the first forward" is not one class either. With every kernel present at
compile (L9, L10, after the same shapes had run in earlier processes on the
same card) **no step occurs in 12 forwards** — the output is bit-identical
throughout.

**The plugin's own name for it.** The pinned source
(`src/plugins/intel_gpu/include/intel_gpu/runtime/options.inl:138`) carries
`GPU_DISABLE_ASYNC_COMPILATION` — *"Disable feature that allows to
asynchronously prepare static-shaped implementations for the primitives
with shape-agnostic kernels selected during compilation"*. That is the
behaviour measured above: the graph starts on shape-agnostic kernels, each
static-shape kernel that finishes compiling replaces its primitive's
implementation between two forwards (a different kernel, a different
summation order, a different f16 rounding — a permanent step in the output),
and a warm kernel cache hands every static kernel over at compile time so
nothing arrives later. The switch itself is a debug option **compiled out
of the release plugin**: `Option not found: GPU_DISABLE_ASYNC_COMPILATION`
(L11, L12, L19, 1.2 s each), so the mechanism was pinned by the cold/warm
contrast, not by turning it off. An f32 leg (L21) was refused before its
first forward: the MoE tile `[512, 1024, 2560]` in f32 is 5,368,709,120 B
against the A770's 4,294,959,104 B cap (the 1,638-token block cap of §4.7 is
the f16 figure); f32 at T ≤ 819 was not run.

> **REVIEW AMENDMENT (2026-09-13, review 5afec70..87d0ae5, HIGH-1): the warm/cold rule above did not reproduce. Re-executed legs: the A770 staircase stayed one-time discrete-class (3 of 3, digests shared across processes), never per-forward noise - but two WARM legs with zero kernel-cache writes stepped once (forward 4, forward 9), one cold leg never stepped in 12, and "warm 12 bit-identical" held 1 of 3 warm legs. Restated honestly: the A770 steps once, permanently, at an UNPREDICTABLE forward, warm or cold; a floor is SETTLED by the observed FLOOR pair, not by cache state. The async-compilation attribution is a never-toggled candidate (the switch is compiled out of release builds - verified this review) and was, under §7.0.1, narrated until this mark: it is now an untested hypothesis, not a mechanism. No measured number above is wrong; the generalisation around them was.**

**What this changes.** The served floor of §4.11 (2.1e-4 mean, one window
with 41 argmaxes moved) is the *small* step caught between two replays of
the same process; the large step (up to 67 argmaxes of 1,024 at depth 4) is
what a reading straddles if a kernel arrives mid-run. Two replays after the
kernel set has settled are bit-identical, so the floor is a property of the
process's compile timeline, not of the input. `tools/kld_served.py --replay
--warmup N` posts every window N extra uncounted times first; `--compare`
pairs every replayed window against the last, so a step inside the warm-ups
shows in the earlier pairs. A gate reading names its warm-up count and its
floor beside the means; a run whose floor pairs are not bit-identical
carries a kernel step and says so. The per-node cuts (`RUN@e53d75b`,
17:22–17:36Z, 8 forwards each, warm cache: `layer0/out`, `ple/out`,
`layer1/out`, `layer2/out`, `layer3/mixer_out` — 354 / 600 / 918 / 1,236 /
1,473 ops) were **all bit-identical over 8 forwards** (digests
`7b02e562e407`, `4dbc5a100364`, `5dde8404d7d4`, `f3b4171fc19f`,
`e4d1bb672152`): every kernel of every cut was cached by then, so the cuts
localised nothing, as the rule above says they cannot without a cold JIT. The
localisation of WHICH primitive steps first is therefore not on the record;
the instrument for it exists (`--cut` under `NEO_CACHE_PERSISTENT=0`,
repeated until the step falls inside the run) and the row stays open. What
is on the record is the mechanism at the plugin level and the three facts a
reading needs: one-time, permanent, cache-dependent.

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
KL(P_ref‖P_cand) ≤ **0.0599 nats — PROVISIONAL**. It now pins f32 on GPU
devices for the same reason §3 gives — a KLD read off an f16 forward is not a
measurement of the export.

**Where 0.0599 comes from (REVIEW ba2d5de F2, 2026-09-13; the derivation
cell).** Not from Flash-Next. The 2026-08-11 expert-quantisation campaign on
Qwen3.6-35B-A3B measured its R0 — UD-Q3_K_XL against BF16, wikitext-2,
`-c 512 --chunks 64` — at mean KLD **0.0399** and set its bar at 50 % over
R0: **bar = 1.5 × 0.0399 = 0.05985 → 0.0599** (dated 2026-08-11, attributed to
that campaign, caveated: another model's number; the note later moved it to
0.0581 against a re-uploaded R0). The harness used to describe that R0 as a
class that had passed against this bar; it was the bar's own input, and the
sentence is gone from the harness (`kld_harness.py` now defines
`INHERITED_R0_NATS × INHERITED_BAR_MULTIPLIER`, pinned by a cell, and every
verdict and every `kld_served.py` line carries `BAR_PROVENANCE`).

**Frontier bar decision (2026-09-13), the three conditions under which the
number stays in use:** (1) every place the bar appears, its provenance
sentence sits beside it; (2) nothing is called "passing against" it; (3) the
0.5.1 acceptance commit re-derives the bar from THIS model's own reference
round-trip — since BF16 of this model may fit nowhere this repository can
run, that commit DECIDES AND STATES the round-trip pair (for instance the
served quantised path against the llama reference on the same shards) and
its own measured floor, doctrine: a stated multiple of the reference's own
rounding error, never an imported multiplier. Until that commit lands every
reading prints the inherited bar, tagged PROVISIONAL, and decides nothing.
A second caveat, measured in §4.11: the served logits have a noise floor of
their own, and a bar is readable only against it — no KL mean is quotable
without its measured run-to-run floor beside it.

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
| KLD gate threshold | ≤ 0.0599 nats mean per-token — **PROVISIONAL**: 1.5 × the Qwen3.6-35B-A3B UD-Q3_K_XL R0 of 0.0399 (2026-08-11), another model's number; re-derived in the 0.5.1 acceptance commit (§7) | harness, red-probed |
| serving-shape IR, 48 layers | 84,372 nodes, 36 GDN + 12 dense-causal | `RUN@198b736` `--serving-shape` |
| → declared constants | 183.07 GiB | `RUN@198b736` |
| → materialised on disk | **0 KiB** — but see the qualifier below; this figure does not discriminate | `RUN@198b736`, qualified `RUN@5663a44` |
| → build cost | 6.43 s, 4.6 GiB RSS | `RUN@198b736` |
| expert body declared type | u4, rank-4 [E, out, groups, 128] | `RUN@198b736` contract test |
| per-expert int4 slice | 2,457,600 B (gate+up+down) | `flash_next_offload.h:45`, re-derived |
| → as the IR walk reads it | 4,915,200 B = **exactly 2×** | `RUN@198b736` (u4 ceiled to 1 B) |
| `slot_pool_from_ir` on any arcint IR | **nullopt** — 0 of 172 IRs carry a moe-typed op (52 of them over 100k; the wider census re-run 2026-09-12). 2026-09-13: the two serving-shape artifacts make it 174; the depth-4 one re-checked by the reviewer — no moe-typed op, `slot_pool` None | `RUN@198b736` |
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
| "What is the capital of France? Answer in one word." | serving-shape IR, `RUN@8a84598`, 2026-09-13 | **no forward reached on either card** — depths 1–3: the pass refuses (`No ScaledDotProductAttention operation observed in the graph`); depth 4 (and 2): `engine.cpp:319`, `requested 25600122880 bytes` against `4294959104` (A770) / `24385683456` (B60). §4.6 | 0 | — |
| "The capital of France is" (ids `760,6511,314,9338,369`) | serving-shape IR at depth 4, `RUN@806b76f`, 2026-09-13 09:38–09:46Z, A770 then B60 | **served path: refused at `set_tensor(inputs_embeds)` — `Port for tensor name inputs_embeds was not found`** (the IR declares `input_ids`). **Labelled probe** (input_ids fed in its place, nothing else): `INFER OK`, logits `(1, 5, 248320)`, finite, `absmax 0.0000e+00`, greedy token id **`0`** on both cards — zero weights, a structure witness and not an answer. §4.7 | 1 (probe) | — |

| "The capital of France is" (ids `760,6511,314,9338,369`) | serving-shape IR at depth 4, **REAL weights** (Q3_K_XL shards; dense fill 95 tensors, 12 expert bodies, the IQ4_NL table bound as 7 ports), `RUN@2413cab`, 2026-09-13 11:24–11:42Z, A770 then B60 | served feed order accepted end to end; logits `(1, 5, 248320)`, finite, absmax 9.1495 (A770) / 9.1205 (B60); **greedy token id `5613` = `ramework`** on both cards (3.586 / 3.573) — the argmax of a four-of-forty-eight-layer prefix, not an answer. §4.8 | 1 | — |

| "The capital of France is" (5 ids) | **THE SERVED PATH**: `arcint --model qwen38-flash-next-d4-ov --ngram-gguf <shard 2>`, A770, `RUN@wt+6743ffb`, 2026-09-13 12:47Z (`--emb-device CPU`) and 12:59Z (defaults, quiet host) | greedy, 8 tokens, both runs byte for byte: **`rameworkenessooter5ussionxigy引`** — first token `ramework` (5613), the probe's; warm decode 38.2 / 37.9 t/s. Chat form ("What is the capital of France? Answer in one word."): `reasoning_content` `像在\n inté rendschedulers 2ingham aalii/sogos …`. Not an answer: depth 4 of 48. **B60**, same binary, embeddings on the card, 12:50Z: `rameworkenessooter5rawd.githubusercontent /riet` — the first four tokens the A770's, warm decode 47.1 t/s. §4.9 | 8 / 32 | — |

§4.6 predicted the first row before its window ran (P4) and the window wrote
it as predicted. §4.7 predicted the second row's served-path signature (B4)
and its probe token (B5) before its window ran, and the window wrote both.
§4.8 predicted the third row's shape — finite, non-zero, not Paris, not
predicted — before its window ran, and the window wrote it (the first run
of that window, with the parity cells' hash ordinal, is retracted in §4.8
and its token `Ġ` is not this line's). **Paris is not dated.** The row
stays the coherence line for the first window that runs the FULL model;
that window replaces `ramework` with the model's own token, dated, in the
same commit as its measurement.

> **AMENDMENT (2026-09-13, operator CR #2 — a mark, never an erasure).** The
> Paris line moved to 0.5.1 by operator CR #2 (2026-09-13); 0.5.0 ships as the
> mechanism release. The France rows above are the served path's dated probe
> token as **mechanism evidence, not an answer**, and they stay exactly as
> measured. The full-model window this paragraph reserves the row for is
> 0.5.1's (`docs/window-051.md`, row (e)); the falsifiable clause carries
> over: the 48-layer served answer to "The capital of France is" is the
> token whose surface form is `Paris` (its id recorded when it is measured),
> or a named refusal that says which layer / which knowledge refuses.

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
