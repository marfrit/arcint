# window-050 — the 0.5.0 prediction window, operating manifest

Recorded 2026-09-12. This file is the manifest a window operator executes: the
holds, the stop order, where the logs go, which suites run on which device with
which compile config, and the PREDICTION TEMPLATE that must be filled in BEFORE
the serving measurement is taken.

**Every command carries a state marker, and the markers are the point:**

| marker | meaning |
|---|---|
| `RUN` | executed on the dev host and its output is recorded here or in RECONCILE |
| `DRY` | the command is correct and was exercised in a no-op / device-free form, but its real effect was not produced |
| `UNTESTED` | written down from the design and never executed — treat every claim about it as a guess |

A command with no marker is a defect in this file. No command here is marked
`RUN` unless this repository holds the output.

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

## 1. Holds — `data` must not go to sleep mid-window

`data` (and with it dirac and both cards) is shut down nightly at 03:00 by a
hertz cron unless a lock says otherwise. The lock is self-expiring and never
overwrites a running foreign lock.

```
# RUN  — place a hold for the window's length plus slack
ssh hertz 'sudo /opt/herding/bin/hold-data.sh <hours> "arcint dev"'
# RUN  — check remaining time
ssh hertz 'sudo /opt/herding/bin/hold-data.sh status'
# UNTESTED — release early (not exercised; the lock was left to expire)
ssh hertz 'sudo /opt/herding/bin/hold-data.sh frei'
```

Observed output shape (`RUN`, 2026-09-12):

```
gesperrt bis 12.09. 06:00 (noch 5 h 0 min)  pid=4015039 owner=hold-data grund=arcint dev
```

If `data` is off, wake it via the FRITZ!DECT plug (`UNTESTED` this session — the
host was already up):

```
# UNTESTED
ssh hertz 'sudo /opt/herding/power/plug-switch <AIN> on'
```

**If `data` freezes or dies silently** — GPU experiments can do that — its kernel
log survives elsewhere; the local journal dies with the box. Check this FIRST
after a freeze:

```
# UNTESTED this session (no freeze occurred)
ssh hertz 'sudo tail -100 /var/log/boltz-netcon.log'
```

---

## 2. Service stop order — the cards are FULL while the services run

Both resident units hold their model VRAM permanently. Any process that wants a
card must stop the resident service first: loading next to it fails with
allocation errors at best, host OOM at worst.

**Verify the mapping yourself before trusting it** — it has drifted once without
anyone noticing until a benchmark caught it by accident:

```
# RUN
ssh dirac 'systemctl --user list-units | grep -i arcint'
```

`RUN` output, 2026-09-12:

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
# RUN
date -Is
systemctl --user stop arcint-agent        # frees GPU.0 (B60)
date -Is
systemctl --user stop arcint              # frees GPU.1 (A770)
date -Is
systemctl --user is-active arcint-agent arcint
```

`RUN`, 2026-09-12: agent stopped `23:54:40Z`, coder stopped `23:54:41Z`, both
report `inactive`, host RAM in use fell 18 GiB → 0 GiB.

### Take a coherence baseline BEFORE stopping

Needed because the restore probe is only meaningful against a before-figure, and
because the agent is a **reasoning** model: a 12-token budget is consumed
entirely by `reasoning_content` and returns an EMPTY `content`, which looks like
a broken endpoint and is not one.

```
# RUN  — max_tokens must be >= ~64; 200 used here
curl -s http://127.0.0.1:8087/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-agent","messages":[{"role":"user",
       "content":"What is the capital of France? Answer in one word."}],
       "max_tokens":200,"temperature":0}'
```

`RUN` baseline, 2026-09-12 23:54Z: `content='Paris'`, 27 completion tokens, MTP
`accepted_prediction_tokens=13 rejected_prediction_tokens=1`.

### Card enumeration once the cards are free

```
# RUN
ssh dirac '~/openarc-venv/bin/python3 -c "
import openvino as ov; c=ov.Core()
print(c.available_devices)
for d in c.available_devices:
    if d.startswith(\"GPU\"): print(d, c.get_property(d,\"GPU_DEVICE_TOTAL_MEM_SIZE\"))"'
```

`RUN` output, 2026-09-12:

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
defaults to `float32`. Measured (`RUN`, OV 2026.4.0, both Arc cards). So a
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

Accepted value forms, measured (`RUN`):

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
# RUN  — the driver, from a byte-exact staged tree
ssh dirac 'cd ~/gpunight && TREE=$HOME/gpunight LOG=$HOME/gpunight/logs ./gpu_window.sh'
```

Per-file invocation it issues:

```
# RUN
Q4E_GPU=GPU.0,GPU.1 Q4E_GGUF_SHARDS=/flash-model \
  ~/openarc-venv/bin/python3 -m pytest <file> -q -s --tb=short
```

CPU-only control, for the same tree (`RUN`, 2026-09-12): `Q4E_GPU=` empty →
**101 passed** with shards, **33 passed / 59 skipped** device-free.

### GPU RESULTS — `RUN` 2026-09-12, both cards, f32 pinned

Suite, one process per file, `Q4E_GPU=GPU.0,GPU.1`:

| file | result | note |
|---|---|---|
| `test_hc_block.py` | **13 passed** | green on both cards |
| `test_hc_combine_block.py` | **13 passed** | green on both cards |
| `test_ple_block.py` | **16 passed** | green on both cards |
| `test_attention_piece.py` | **15 passed** | green on both cards, REAL width |
| `test_gdn_block.py` | 2 failed, 5 passed | GPU T=96 only; see §4.2 |
| `test_moe_block.py` | 10 failed, 6 passed | every GPU leg, MoERouterFused |
| `test_backbone.py` | 8 failed, 6 passed | every GPU leg, same root |

**This is the first time any GPU parity leg in this suite has been green.** The
previous window recorded "ALL 22 GPU.1 legs fail"; it ran before §3 was wired.
Three whole files and the real-width attention piece now pass on both cards.

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

#### 4.2 GDN on GPU — fails at ≥2 chunks, and the magnitude is NOT explained

```
[ov-parity] device=CPU    T= 64  max-abs=1.401e-06  KLD=2.418e-12
[ov-parity] device=CPU    T= 96  max-abs=1.401e-06  KLD=2.162e-12
[ov-parity] device=GPU.0  T= 64  max-abs=9.313e-07  KLD=1.774e-12
[ov-parity] device=GPU.0  T= 96  max-abs=7.981e-02  KLD=4.997e-03   <-- FAIL
[ov-parity] device=GPU.1  T= 64  max-abs=9.388e-07  KLD=1.742e-12
[ov-parity] device=GPU.1  T= 96  max-abs=7.981e-02  KLD=4.997e-03   <-- FAIL
```

T=64 is one chunk; T=96 is two. Identical to four figures on both cards, so it is
deterministic, not a race. An independent tiny-width sweep against an f64
reference locates the divergence point exactly (`RUN`):

```
    T chunks            CPU          GPU.0          GPU.1
   32      1      2.041e-12      2.951e-12      2.837e-12
   64      1      3.457e-12      2.673e-12      2.645e-12
   96      2      3.428e-12      3.796e-08      3.796e-08
  128      2      4.783e-12      3.100e-08      3.100e-08
  192      3      8.155e-12      5.957e-08      5.957e-08
  256      4      8.201e-12      5.071e-08      5.071e-08
```

CPU stays at 1e-12 at every chunk count. GPU steps up by four orders of
magnitude at the FIRST inter-chunk state carry and then stays flat — it does not
grow with chunk count. So the carry (`S' = S·chunk_decay + kᵀ·v_new`) is
confirmed as **a** divergence point between the two plugins, at 3–6e-08, which
is the f32 floor.

**That does not explain 7.981e-02.** Six orders of magnitude separate the two
measurements. Naming the carry as the cause of the test failure would be
narrating a mechanism rather than measuring one. **Status: the GDN GPU failure at
T≥96 is OPEN with its root cause unidentified.** What is established is the
failure itself, its determinism across cards, and that a carry-related
CPU/GPU difference exists at the f32 floor.

#### 4.3 MOE-GPU-FUSION — localised to 22 nodes, workaround found

Both the `test_moe_block.py` and `test_backbone.py` GPU failures have one root,
in two manifestations:

```
add_required_reorders.cpp:342  No layout format available for
    moerouterfused:MoERouterFused_301441, impl_type: any
    (format: bfyx, data_type: f32)  shape=[64,2]
program_builder.cpp:268        Input moerouterfused:MoERouterFused_24271.out1
    hasn't been found in primitive_ids map
```

`[64,2]` is `[T, top_k]`. The **plugin's own** MoE router fusion builds a
primitive it then cannot lay out or register. It is not our graph.

Minimal reproducer and the piece-level sidestep (`RUN`, both cards):

| piece | nodes | CPU | GPU.0 | GPU.1 |
|---|---|---|---|---|
| `moe_router` (gate only) | 22 | OK | **FAIL** MoERouterFused | **FAIL** |
| `moe_shared_expert` | 17 | OK | OK, \|dev−CPU\| 6.112e-10 | OK, 4.948e-10 |
| `moe_experts_chunk_8` | 196 | OK | OK, \|dev−CPU\| 2.766e-04 | OK, 3.052e-04 |

So **the expert bodies and the shared expert run on GPU**; only the router's
22-node softmax→topk subgraph dies. Levers tried, each a measured result:

| lever | GPU.0 | GPU.1 |
|---|---|---|
| baseline, f32 pinned | FAIL | FAIL |
| `EXECUTION_MODE_HINT=ACCURACY` | FAIL | FAIL |
| `OV_GPU_DISABLE_TRANSFORMATIONS=1` | FAIL | FAIL |
| **softmax only, no topk** | **OK** sum=64.0000 | **OK** |
| **topk only, no softmax** | **OK** sum=1566.3918 | **OK** |

No MoE-specific key exists in the GPU plugin's `SUPPORTED_PROPERTIES` (the full
list was enumerated; the only transform-adjacent keys are
`GPU_ENABLE_SDPA_OPTIMIZATION`, `GPU_ENABLE_LOOP_UNROLLING`,
`GPU_DISABLE_WINOGRAD_CONVOLUTION`, `EXECUTION_MODE_HINT`). Neither that hint nor
the documented `OV_GPU_DISABLE_TRANSFORMATIONS` env var suppresses the fusion.

**THE WORKAROUND, measured: the matcher needs the softmax→topk PAIR. Either half
alone compiles and runs on both cards.** Two ways to use that:

1. Emit the router as two models — softmax side and topk side — so neither
   presents the full pattern. `UNTESTED` as a wired change; the two halves were
   each proven to run.
2. Keep the router on CPU and the expert bodies on GPU. The router is 22 nodes
   and 5 MB; the cost is a host round-trip of `[T, E]`. `UNTESTED` as a wired
   change.

Both need a test change, and a documented workaround with a measured
before/after is a cell, not a defeat. Neither is wired tonight; what is recorded
is the characterisation and the two candidate shapes.

A 22-node reproducer is also exactly the right size for an upstream report.

### MoE legs and MOE-GPU-FUSION — stated honestly

The MoE GPU legs have historically hit a plugin fusion path
(`MOE-GPU-FUSION` in RECONCILE). Status as of this manifest: **carried forward,
not fixed.** What is and is not known must stay separated:

- `UNTESTED` — whether pinning f32 alone changes the MoE GPU outcome. It is the
  first thing to read off the run, and it is a cell, not a prediction.
- `UNTESTED` — a plugin key that disables the router-fusion transform. No such
  key has been verified to exist in OV 2026.4; until one is quoted from the
  plugin's own `SUPPORTED_PROPERTIES` or source, "disable the transform" is a
  hypothesis.
- `UNTESTED` — graph-shape variants that sidestep the matcher (for example the
  already-landed router / expert-chunk / shared-expert split, which is a
  different graph shape than the fused single-layer MoE).

A documented workaround that needs a test change is a cell, not a defeat. A
workaround asserted without a measured before/after is neither.

---

## 5. Residency — the SIZE LEDGER, and the number that decides the window

`RUN`, 2026-09-12, real checkpoint geometry, T=64, CPU, every row either built
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

## 6. The 86k-node compile — an observation with a budget, not a milestone

CPU behaviour is known (`RUN`, REVIEW 57b1952 §7): node count is linear in GDN
layer count at ~1,879 nodes/block, and an 86k-node graph (the tiny width at the
checkpoint's 36 GDN layers) compiles on CPU in 22 s without falling over.

```
layers    nodes  nonconst   xml MB  build s  compile s      (RUN, CPU)
     4     9671      3727     4.25     0.10       1.92
    12    28663     10975    12.71     0.38       6.09
    36    85639     32719    38.05     0.95      21.95
```

Whether the GPU plugin's per-shape kernel JIT behaves the same on 86k nodes was
**not** answered there and is not answerable from CPU numbers.

`RUN` 2026-09-12, 20-minute hard budget per attempt, each attempt in its own
child process so a wedged compile cannot take the run with it:

```
layers  device   nodes      const B  build s  compile s  peak host GiB  result
     4     CPU    9671      756,660     3.15       1.59           0.83  OK
     4   GPU.0      --           --       --         --             --  RuntimeError
     4   GPU.1      --           --       --         --             --  RuntimeError
    12     CPU   28663    2,153,108     3.29       5.77           1.17  OK
    12   GPU.0      --           --       --         --             --  RuntimeError
    12   GPU.1      --           --       --         --             --  RuntimeError
    36     CPU   85639    6,342,452     3.92      18.95           2.17  OK
    36   GPU.0      --           --       --         --             --  RuntimeError
    36   GPU.1      --           --       --         --             --  RuntimeError
```

**THE EXPERIMENT IS BLOCKED UPSTREAM OF ITS OWN QUESTION, and no budget was
spent.** Every GPU attempt fails in seconds, at every layer count including 4,
because the backbone contains an MoE layer and the plugin's `MoERouterFused`
defect (§4.3) stops program building before any node-count behaviour is reached.
The 20-minute budget was never approached. So: **whether the GPU plugin's
per-shape kernel JIT copes with 86k nodes is STILL UNANSWERED**, and it cannot be
answered with this graph until §4.3 has a wired workaround. Saying "the 86k
compile failed on GPU" would be true and useless — it failed at 9,671 nodes too,
for a reason that has nothing to do with node count.

The CPU column reproduces REVIEW 57b1952 §7 (there: 9671 / 28663 / 85639 nodes,
1.92 / 6.09 / 21.95 s; here 1.59 / 5.77 / 18.95 s on a quieter host). Peak host
memory is modest and linear-ish: 0.83 → 1.17 → 2.17 GiB.

The F3 hoist candidate (~65% node cut) was **not attempted** this session — it is
only worth measuring against a GPU compile that can start, which §4.3 currently
prevents. `UNTESTED`.

A timeout is a result and gets written down as one. So is a blocker that makes
the timeout unreachable.

---

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
| card under test | A770, 15.11 GiB reported (16,225,243,136 B) | `RUN` GPU.1 enumeration |
| alternate card | B60, 22.71 GiB reported (24,385,683,456 B) | `RUN` GPU.0 enumeration |
| CARD-tier weights at f32 | 18.492 GiB | `RUN` size ledger |
| → fits A770 reserve | **no**, −3.492 GiB | `RUN` size ledger |
| → fits B60 | yes, +4.218 GiB | `RUN` size ledger |
| offload tier | 450.000 GiB f32 (48 × 9.375) | `RUN` size ledger, file-sourced |
| host-mmap tier | 190.736 GiB f32 n-gram table | `RUN` size ledger, file-sourced |
| f32 : quantized ratio | 7.72× | `RUN` size ledger vs WP6b 85.38 GiB |
| GPU inference precision | f32, pinned explicitly | `RUN` §3 |
| QSA→dense price, T ≤ 2048 | **0.0, exact** | `RUN` attention piece |
| QSA→dense price, T = 2080 | 2.386e-02 over 29/2080 rows | `RUN` attention piece |
| attention piece floor vs pin | 1.855e-07 (T=64), 1.535e-07 (T=96) | `RUN` attention piece |
| KLD gate threshold | ≤ 0.0599 nats mean per-token | harness, red-probed |

### Terms still to be predicted — `UNTESTED`, fill before measuring

| term | prediction | then: measured | provenance of the prediction |
|---|---|---|---|
| resident GiB on the card | | | |
| KV precision pinned | | | |
| prefill t/s at T=2048 | | | |
| decode t/s, single lane | | | |
| expert hit rate (cache) | | | |
| expert miss cost (per miss) | | | |
| MTP acceptance rate | | | |
| MTP overhead term | | | |
| amortised t/s incl. MTP | | | |
| 86k-node GPU compile, s | | | |

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
# RUN
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
