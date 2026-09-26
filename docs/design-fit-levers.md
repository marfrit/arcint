# The two fit levers — the host-compile wall and the dense term (2026-09-26)

Campaign: `docs/campaigns/sub4bit-vram-kernel.md` (the full-depth all-resident
arm of Qwen3.6-35B-A3B on the 16 GiB card). Predecessor: design note
`docs/design-qwen35moe-serving-shape.md` §12–§14, whose §14.3 left the
"≈2.9× compile materialisation" as its own defect and whose closing analysis
leg named two levers, needed together: a dense u8/i4 form and a reduction of
that materialisation. Evidence classes: `code`, `measured-here`; every card
number names the card (A770, `GPU.1`, PCI `8086:56a0`, 15.11 GiB), the depth,
the KV precision and the flags.

## 1. The baseline this note started from

The handover's arithmetic, all `measured-here` on 2026-09-25/26:

| term | GiB |
|---|---|
| packed expert bodies | 12.031 |
| dense + lm_head + norms (lm `.bin` − experts) | 2.519 |
| activations @ chunk 1024 / 512 / 256 | 1.700 / 0.850 / 0.425 |
| drafters (the embeddings model on the card) | 0.950 |
| margin | 0.250 |
| A770 | 15.111 |

and the wall in front of it: the packed full-depth compile was SIGKILLed by the
host watchdog at 44,292,600 kB RSS (a 14.55 GiB artifact; "≈2.9×"), the
re-laid f16 one OOMed at 50.1 GiB. Two corrections to this table are below
(§3.3, §4.1): the dense term was not 2.519 GiB, and the 2.9× was not a
property of compiling.

## 2. The instrument

`tools/bigalloc.c` — an `LD_PRELOAD` shim that records every host allocation
at or above a threshold (malloc family, aligned allocators, anonymous mmap)
with its call stack, keeps per-stack live/peak/total bytes and snapshots the
per-stack live bytes when the process-wide total sets a new peak (quantised:
a new snapshot only after the peak has grown by `BIGALLOC_STEP`, default
256 MiB, so "at the peak" means within one step of it); dumps periodically so
a watchdog kill still leaves its last state.
`tools/bigalloc_report.py` symbolises the dump with `nm` against unstripped
copies of the same objects: the pinned OpenVINO's Release link passes `-s`
(`cmake/developer_package/compile_flags/sdl.cmake`), so the plugin and core
were **relinked without `-s` from the unchanged objects** — `.text`
byte-identical to the stripped builds (sha256 of the section compared), 73,542
text symbols. Red-first cells: `tests/python/test_bigalloc.py` (5; mutants:
free tracking removed → 2 red, peak snapshot removed → 1 red, the
`BIGALLOC_ONLY` filter removed → 1 red).

Two device-free probes sit beside it:
`tools/native_moe_match_probe.cpp` runs ONLY the native MoE matcher pass
(`ConvertTiledMoeBlockNativeToMoeCompressed`, in `libopenvino.so`) on an IR
and prints what it produced; `tools/native_moe_block_ab.cpp` runs one block IR
on the CPU plugin (the exact f32 oracle) and on a GPU with given properties
and prints the diff against the band `tests/python/test_native_lowering_gpu.py`
uses (2 % of the element + 1 % of its row's RMS). The Python binding cannot
drive a patched runtime (it asserts its build number equals the runtime's),
so the lowering cell calls the C++ runner when `ARCINT_NATIVE_BLOCK_AB` is set.

## 3. Lever 2 — the "2.9×" was unfused decode chains, constant-folded

### 3.1 What held the memory (`measured-here`)

The tracer on the packed depth-4 compile (plugin `ov-0052` + 0053, A770):
peak **15.06 GiB** of large allocations for a 3.63 GiB artifact, all of it
`ov::Node::constant_fold` — `ConstantFolding` inside `ConvertPrecision`
(10.0 GiB live at the peak) and `MultiplyMultiplyFusion` inside
`CommonOptimizations`, in 512 MiB and 1 GiB tensors: one fully dequantised
`[256, 512, 2048]` f32 expert tensor each; **233.5 GiB allocated in total**.
Both passes run AFTER the native matcher (`transformations_pipeline.cpp`
L671 against L777/L783), so the chain was only there to fold because the
matcher had left it in place.

### 3.2 Why the matcher left it (four defects)

1. **The packed block's scale anchor** (`code`, fixed by patch **0054**): 0052
   registered the `scale` Multiply as the anchor the fused op takes as its
   scale; the callback's Constant guard then refused every packed block. The
   record's "the matcher fires (`iq2sp RESOLVED` ×8)" rested on a print
   inside `resolve()`, which runs before that guard. The probe reads the
   pass's output instead: packed depth-4 IR → `MOE_COMPRESSED 0` on the 0052
   core, **4** with 0054 (the scale slot is now the f16 `d` Constant
   `[E, out, K/256, 1]` the kernels read).
2. **`--dense-fp16` compresses the chain's value Constants** (`measured-here`,
   fixed by patch **0057**): `save_model(compress_to_fp16=True)` turns every
   f32 Constant — grids, tables, ±1, divisors — into f16 + `Convert`; the
   blocks' bare `wrap_type<Constant>` then matched no layer. Probe, full-depth
   IRs: the f16 re-laid artifact fused **0 of 40 even on the 0052 core** (so its
   50.1 GiB compile was this too), the f32 re-laid one 40 of 40. 0057 accepts a
   value Constant bare or behind that `Convert` (all such values are exact in
   f16; the kernels carry their own tables and read only the anchors); both
   fused 40/40 afterwards.
3. **A stale full-depth artifact** (`measured-here`): the packed d40 predated the
   rank-5 signs chain (only d4 was rebuilt after it); re-exported as
   `qwen36-35b-a3b-d40packed2-ov`.
4. The chains a fused op does not replace are the only ones the pipeline can
   fold, so (1)–(3) are the whole mechanism: with them fixed the full-depth
   packed compile peaks at **0.47 GB anonymous** (`g2`, `ov-0057`, 60.1 s,
   against 44.3 GB and a kill). The host ceiling of "37.4 layers" is gone.

### 3.3 Correction to the record

The "≈2.9× materialisation" (design note §14.3, the campaign's 2026-09-26
analysis entry, the handover) is **retracted as a compile cost**: it was the
constant folding of unfused native chains, measured by the tracer, and the
compile of the same artifact with the chains fused holds 0.47 GB. The
candidates §14.3 listed (pack the downs; "packed Constant + expanded copy +
reorder buffers") were not the mechanism.

## 4. Lever 1 — the dense projections in u8

### 4.1 The dense term, measured

A constant census of the packed d40 IR: experts **10.875 GiB** (u8 9.719 +
f16 0.781 + u4 0.375), non-expert f16 **3.677 GiB** (lm_head 0.947, the rest
per-layer projections, norms, rope tables). The handover's 2.519 was the file
size minus the fill census, and the fill census over-counts the experts by
exactly the difference (`code` + arithmetic): it sums the split parts, which
carry the packed `d` and the IQ3_XXS / IQ4_NL scales as f32 where the artifact
stores f16, and the IQ4_NL codes unpacked at a byte a value where the artifact
packs u4 — 0.156 + 0.578 + 0.047 + 0.375 = 1.156 GiB = 12.031 − 10.875. The
GGUF stores every dense projection as Q6_K (252 tensors).

### 4.2 The form

`tools/q4e/dense_u8.py`, exporter flag `--dense-u8`. Q6_K values are exactly
`(d·sc)·q`, `q ∈ [-32, 31]` per 16-value group. The pass recovers `(q, s)` from
the dequantised values (so no second read of the file and no assumption about
the feed's head re-orders), in its canonical coarsest form (gcd of the group's
q divided out — a finer valid scale is exact too but rounds worse in f16:
max relative deviation 1.7e-2 without it, 9.47e-4 with it, on the real shard),
and writes the plugin's own compressed chain — u8 `[N, K/16, 16]` →
`Convert(f16)` → `Subtract(u8 32)` → `Multiply(f16 scale)` → `Reshape` →
`Convert(f32)` → `MatMul` — the form arcint's C++ Q6_K repack already serves
(`src/core/gguf_repack.cpp`, type 14; DESIGN 7.0.2ba). A tensor with any group
not of that form is left as it was and named in the report.

On the real shard (`measured-here`): all 252 Q6_K dense tensors recover every
group (the lm_head's 31,784,960 groups in 13.8 s); the RMS weight error is
**1.32×** the f16 artifact's own rounding (4.33e-6 against 3.29e-6 — two f16
roundings, scale then product, against one).

### 4.3 What it may not touch (both measured, both on the served path)

- **The shared expert.** `FuseMOESharedExpert` hands its four weights to the
  MoE op as they are (`any_input()`); the kernel reads plain weights. Kept, by
  name (`emit_shared_expert` now names them).
- **Attention k/v while q is compressed.** Depth-4 served logits A/B against
  the f32-dense artifact (A770, `ov-0057`, u8 KV, chunk 1024, ratio 0 +
  dispatch, `--dyn-quant off`, one 1,000-token prompt), bisected by projection
  class: lm_head only KL 1.2e-7; the 11 per-layer projections (q converted,
  k/v not) KL 2.0e-4; k/v only KL 1.2e-6; everything but the lm_head
  **KL 2.73, argmax 7/1000** — q, k and v all compressed is the failing
  condition. The plugin fuses those three horizontally (`FullyConnected` + 3
  `Crop` in the runtime graph); the fused kernel is exact in isolation at
  rank 3 and rank 2 (block runner, T 1/300), so the defect needs the served
  attention context. Kept k/v plain (2 × 512 × 2048 a layer). **The mechanism
  inside the plugin is OPEN**; the reproducer is the depth-4 artifact with
  `ARCINT_DENSE_U8_SHAPES=8192x2048,4096x2048,2048x4096,512x2048`.
- The plugin's int8 **dynamic activation quantization** for compressed FCs is
  on by default in arcint for non-GGUF artifacts; the block runner reads it at
  9.8× the band at T = 300 and within it with the quantization off. The f16
  artifact never went through that path, so the u8 artifact is served with
  `--dyn-quant off` to keep the comparison to it about the weights.

With both kept: depth-4 served A/B against f32 dense — **KL 1.95e-4, argmax
983/1000**, the same generated text. Full depth: 111 projections converted,
140 kept (120 shared-expert, 20 k/v); 3.291 GiB of f16 → 1.851 GiB; lm `.bin`
15,623,664,527 → **14,077,670,352 B** (the 1.440 GiB the pass reports);
admitted as `qwen3.6-35b-a3b-native-d40packed-u8`.

**Correction, same day** (found by review, `code` + `measured-here`): the
first full-depth u8 export read 14,490,689,806 B, 0.385 GiB more than the
pass's own saving. `ov::save_model(compress_to_fp16=True)` calls
`compress_model_to_f16`, which skips the WHOLE model once
`DetectCompressedWeights` finds any compressed-weight chain
(`compress_float_constants.cpp`, `is_model_optimized`), so every constant the
pass kept stayed f32 (0.772 GiB against 0.386 as f16, constant census). The
exporter now plans from the exact f32 values, compresses the rest
(`compress_model_transformation`), then splices the u8 chains
(`plan()` / `commit()`); a cell asserts no f32 constant survives, and the old
order leaves one at unit scale. The plan holds the u8 arrays across the
compression (≈1.85 GiB at full depth): the export's host peak read 40.96 GiB
of the 44 GiB container, against 36.51 before — the forecast for a full-depth
`--dense-u8` export. Device residency did not move (the plugin held
those constants as f16 either way: 13.11 GiB, same digests, g9 below). A
projection whose f16-scale rounding exceeds 2⁻⁹ relative is now kept, not
converted (a scale in f16's subnormal range); the real shard's worst is
9.47e-4, so nothing changed there. `--dense-u8` refuses a non-qwen35moe
family (the k/v exclusion is by the names that emitter gives them).

## 5. Correctness defects the block cells found on the way

The lowering cell was extended with IQ2_S, IQ2_S-packed and the served
all-resident dispatch route, and run through the C++ runner at the 35B's own
routing (E = 256, top-8; T 1 and 6):

- **Per-expert dispatch read IQ4_NL and Q8_0 wrong** (67–263× the band),
  every other route right. The matcher sets `zp = scale` for those formats and
  the constant cache gives both inputs one buffer; the slot fill wrote the
  scale device-transposed and then the zp's raw copy over it. Patch **0056**
  skips a native zp that aliases its scale (the device zp of a native format is
  never read by a kernel), and adds the missing Q8_0 down slot offset and the
  IQ2_S down strides and decode branch. After: every pair either checkpoint
  uses passes on every route (≤ 0.40 of the band). Consequence for the record:
  the 35B's three IQ4_NL down layers and every Flash-Next IQ4_NL layer were
  computed wrong on every dispatch-route measurement before 0056 (VENICE rate
  legs, DESIGN 7.0.2ce/7.0.2cf's native-dispatch divergence) — those readings
  are suspect until re-measured.
- **The IQ2_S-packed decoders read 32 of every 256 values** (CPU tier and
  OpenCL both indexed qs/signs/qh/scales and the input by the 256-block index):
  patch **0055**; 86× → ≤ 0.39 of the band.
- **The emitter's re-laid IQ2_S chain interleaved its two sub-block scales**
  (`code`: broadcast `[.,1,2] → [.,16,2]`); the CPU-plugin decode was wrong by
  up to 27 on a 30 weight, the fused GPU path (which reads the compact
  Constant in ggml's order) was not. `test_native_expert_chain` had no IQ2_S
  case; it has one now, red before the fix.
- Still open: IQ2_S as a *down* projection and Q8_0 as *gate/up* under
  dispatch throw `Unable to cast reference from base to derived type` at
  compile; neither checkpoint uses them.

Served level after the fixes: packed against re-laid, depth 4, A770 — KL
**4.18e-5**, argmax 993/1000 (was 0.634 and 252/1000).

## 6. The resident pool, filled at bind (patch 0058)

At ratio 0 the pool holds every expert and the resident set is sorted by
expert id, so slot i is expert i — the order the compile's constant upload
already wrote the weights in. The pool nevertheless re-read each expert from
the `.bin` on its first routing (`measured-here`, full depth, `ov-0057`, the
u8 artifact, embeddings on the CPU, depth 1 only — run g6: 7,267 misses in
the load probe plus a 32-token decode, 50,354 tensor reads at 5.0 ms average,
decode 1.0 t/s; g7 below, the same configuration with a warmer page cache,
read 1.4 ms a tensor and 3.0 t/s). 0058
uploads only the scale/zp tensors at `bind()` (the kernels read the scale
device-transposed) and marks every slot filled; weight bytes do not move.

## 7. The gate — PASS: the full-depth artifact loads and serves fully resident

One card, three processes (`measured-here`). A770 `GPU.1` (`8086:56a0`,
15.11 GiB); artifact `qwen36-35b-a3b-d40packed-u8-ov` (40 layers, packed IQ2_S
gate/up, IQ3_XXS/IQ4_NL downs, dense u8 group-16); `--offload-ratio 0
--moe-per-expert-dispatch` (the tier auto-enabled, no expert on it), u8 KV,
`--prefill-chunk 1024` (the load settles chunk 512), `--emb-device CPU`,
`--dyn-quant off`, `--n-ctx 8192`, `MOE_OTD_PERF_LOG=1` (g7–g9); arm driver with `unshare -rm` for the
container's sparse CPU list; physical-host sampler with the 4 GiB watchdog (0
trips). g7 on plugin prefix `ov-0057` (0003–0057) and g8 on `ov-0058`
(0003–0058), both on the first u8 export (kept constants stored f32, §4.3's
correction); g9 on `ov-0058` and the re-export that is committed
(lm xml `8a778f9dca2cb283`, 14,077,670,352 B), binary `4d004864…`.

| | g7 (lazy fill) | g8 (filled at bind, 0058) | g9 (0058, committed artifact) |
|---|---|---|---|
| compile | 56.8 s | 40.9 s | 59.2 s |
| device-resident | 13.11 GiB | 13.11 GiB | 13.11 GiB |
| reservation → max ctx per lane | 84,704 | 84,704 | 84,704 |
| T_boot (launch → `/props`) | 245.2 s | **154.6 s** | 173.1 s |
| expert misses / hit rate | 10,146 / 90.6 % | **0 / 100 %** | 0 / 100 % |
| decode @ depth 1 (32 tok) | 3.0 t/s | **7.3 t/s** | 7.7 t/s |
| prefill @ 4096 | 12.0 t/s | 12.5 t/s | 12.5 t/s |
| decode after the 4096 prefill | 7.9 t/s | 7.9 t/s | 7.9 t/s |
| digest depth 1 / 4096 | `5f4625c0bf7c` / `b1a16fbc9d4c` | **the same** | **the same** |
| `wait4` max RSS | 0.93 GiB | 1.08 GiB | 1.02 GiB |

The reservation, verbatim from g8 (g9 reads the same): `weights+graph 13.11 GiB + drafters 0.00 +
expert slots 0.25 (probe-static) + activations 0.49 (all 1 lane, chunk 512) +
margin 0.25 + 1 x (GDN rows 95.6 MiB + KV 11.3 KiB/token) of 15.11 GiB -> max
ctx 84704 per lane`. Depth-1 text, both runs: `":\nYou are a helpful
assistant.\n\nuser:\nWhat is the difference between a \"sightseeing\" and a
\"tourist\"?\n\nWhat is"`.

What the table does and does not say:

- **Fit.** Full depth is all-resident on the 16 GiB card with 84,704 tokens of
  u8 KV per lane. **262144 is not reachable** at u8 KV (2.83 GiB of KV against
  ~0.93 GiB free after the rest). With the embeddings on the card (0.95 GiB)
  the reservation read 27,808, and the process then held 14.85 GiB of VRAM plus
  0.43 GiB of GTT and decoded at 0.9 t/s: the analytic ledger undercounts by
  ≈0.4 GiB at the edge, and the spill runs over the card's PCIe 3.0 x4 link.
  Embeddings on the CPU is the configuration that fits with room.
- **Equivalence.** 0058 moves no weight byte: g7 and g8 serve the same digests
  at both depths, two processes apart (the A770 is the bit-readable card); g9
  on the re-export serves them too, so storing the kept constants f16 instead
  of f32 changed nothing on the device.
- **Rate.** Decode 7.3–7.9 t/s against the recorded 40-layer int4 comparand's
  5.4 / 5.3 (tier + dispatch, 2026-09-25) and the depth-4 native arm's 57–71
  scaled by depth. Prefill 12.5 t/s @4096 against the int4 comparand's 16.0: it
  did not move with 0 misses, so it is the per-expert dispatch kernels at
  prefill, not the fill — LYON's lever, not this gate's.
- **Quality.** The depth-4 A/B (§4.3) bounds the dense form at KL 1.95e-4
  against f32 dense; no full-depth KLD against the model's own reference was
  run here.

## 8. OWED

- The plugin mechanism of the q/k/v horizontal-fusion defect under compressed
  weights (§4.3); the reproducer is named there.
- IQ2_S as a down projection and Q8_0 as gate/up under dispatch (refused at
  compile).
- The dispatch-route measurements taken before patch 0056 (the VENICE rate
  legs, DESIGN 7.0.2ce/cf) read IQ4_NL layers wrongly and are to be re-measured.
- The fit ledger's ≈0.4 GiB undercount at the edge (§7).
- A full-depth quality reading (KLD against the model's own reference) of the
  u8 artifact.
- The prefill rate under dispatch (§7), the LYON campaign's.
