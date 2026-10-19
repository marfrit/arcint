# qwen3tts-card-real-time — the Qwen3-TTS card arm is slower than real time, and not faster than the CPU: the gen graphs fall back to the CPU

**Open 2026-10-19.** The fixed model speaks German and stops at `codec_eos`
(`qwen3tts-german-genpath` closed), but the card arm through arcint's own
`/v1/audio/speech` runs at about **0.26× real time** on the B60 — slower than
real time, and no faster than the same loop on the CPU. The milestone
criterion B3 and the unit's serving shape both need real time on the card. A second window (2026-10-19) then measured
placement and pipelining, one variable each: the decoder's flush cost is
fixed per call, the flush window (72, the decoder's `wav_tfm_swa`)
cannot be lowered seamlessly, and the overlap hypothesis is refuted with
numbers; the hybrid stays the best arm. See the second measured section.

## Charter

`docs/milestone-1.2.0.md` §6 B3: real-time factor (audio seconds per process
second) at the fixed 8-sentence set, the first request separate; the card
beats real time by a stated margin at the chosen quant. The measurement below
was taken in the card arm that closed C4; it fails B3.

Distinct from the closed `qwen3tts-german-genpath` (the Q8_0 mmproj's code
predictor made German unintelligible and never stop): that defect is fixed
and its gate passed. This campaign is the **speed** lever on the same path.

## Reference to follow

The mechanism to measure against is the upstream loop itself: the pinned
llama.cpp `tools/tts/tts.cpp` and `tools/mtmd/mtmd-helper-gen.cpp`
(`step_gen` per frame, the code predictor and the code2wav decoder). On the
card the graphs are built by
`libmtmd`/`clip` on the chosen backend; the lever is the **OpenCL backend's
`supports_op` coverage** (`ggml/src/ggml-opencl/ggml-opencl.cpp`) and the
per-node profile. Evidence class: `code`.

## Gate

B3, `measured-here`: the eight German sentences of the fixed set, one binary
and model set, the round trip already green at the answer. The card's
real-time factor at the fixed set beats real time (proposal, printed with
every arm); the CPU arm of the same loop is the cross-check, and B1's
`GGML_OPENCL_PROFILING` per-node profile comes before any kernel work.

## Current state (measured-here, 2026-10-19, two windows)

Card: B60 (PCI `8086:e211`), dev binary sha256 `0179adb5…`, Q8_0 backbone
(`ac7931ae…`) + **bf16 mmproj** (`b9503e95…`), GPU.0, 4,096 ctx, the enrolled
voice `gate-fleurs` (FLEURS `de_de`, CC-BY-4.0), sampling t 0.8 / top_p 0.95 /
top_k 40 / repetition 1.0, seed 42; the full clone path through
`POST /v1/audio/speech`. Eight sentences, no frame cap reached:

| item | measured |
|---|---|
| frames / audio per sentence | 46–69 frames (3.68–5.52 s at 12 Hz) |
| process per sentence | 14.86–20.89 s |
| corpus total | **36.48 s audio / 141.38 s process = real-time factor 0.258** |
| CPU cross-check (same loop, bf16 pair, 2026-10-19) | 44 frames / 3.52 s in 15.1 s = **0.233** |
| the gen graph | OpenCL compute buffer 805.23 MiB, CPU compute buffer 278.40 MiB, **graph splits = 947, nodes = 6492** |
| unsupported on OpenCL (warmup list, both the speaker-encoder and the gen graphs) | `ARANGE` ×354, `GET_ROWS` ×46, `SIN` ×29, `LOG` ×24, `PAD_REFLECT_1D` ×22, `SUM` ×15, `COL2IM_1D` ×6 (f32) |

**B1 profile done (2026-10-19): the graph splits build, not the graph compute.**
The instrumented per-phase arms below replaced the disposition above with a
measured one. Two graphs exist: the **code predictor** (`CLIP_GEN_PROCESS_GEN_CODE`,
1 clip call per frame, 6792 nodes, 1007 splits) and the **code2wav**
(`GEN_WAV`, 1 call per sentence, 6492 nodes, 947 splits); the speaker encoder
(`OTHER`, 559 nodes, 44 splits) is a fixed ~14-99 ms. The earlier "947 / 6492"
row is the code2wav graph, not the per-frame one.

Arms on the B60 (PCI `8086:e211`), dev binary sha named, same fixed model pair
and `gate-fleurs`/seed 42/reference sampling, the 8-sentence set:

| arm | binary sha256 | RTF | per frame | predictor | backbone | code2wav | splits |
|---|---|---|---|---|---|---|---|
| wall (TTSPROF, no cl profiling) | `cecebd27…` | 0.267 | 280.3 ms | 272.4 ms (97.2 %) | 7.90 ms | 14.5 s / 8 | GEN_CODE 1007, GEN_WAV 947 |
| ocl (+ `GGML_OPENCL_PROFILING`) | `561aceb5…` | 0.252 | 298.8 ms | 288.6 ms | 10.2 ms | 14.5 s | same |
| `0061` OpenCL ARANGE | `9bdb2e91…` | 0.280 | 265.3 ms | 257.4 ms | 7.90 ms | 14.5 s | GEN_CODE **299**, GEN_WAV **239** |
| hybrid (mmproj graphs on CPU, backbone on device) | `47838c99…` | **0.828** | 72.2 ms | 61.9 ms | 10.3 ms | 13.5 s / 8 | 1 |

Per-frame split of the wall arm (456 frames, 38.0 s audio, 142.4 s process):
code predictor **124.2 s (87 %)**, code2wav **14.5 s (10 %)** (1,813 ms/flush),
backbone decode 3.6 s (2.8 %), prompt 0.43 s, speaker encoder 0.11 s.
`build_us` 0.65 ms/call and `alloc_us` 1.16 ms/call are **0.2 %** of the
predictor call — graph build/alloc is not the lever.

The `ocl` arm: OpenCL kernel *execution* totals **18,581 ms** over the run
(code2wav `im2col_f16` 5,685 ms, predictor `mul_mat_f16_f32_1row` 3,931 ms,
backbone `mul_mv_q8_0_f32_flat` 2,186 ms, `mul_mat_f16_f16` 2,139 ms, concat
1,098 ms, `flash_attn_q1` 671 ms, rms_norm 442 ms). Host enqueue (queued→submit
in the trace) is **1,949 ms** (host/device 0.11). Against the wall's 145.1 s
of clip compute, **~126 s (87 %) is outside device-kernel execution**.

**The split count is a suspect, now measured not to be the cost.**`0061` cut
GEN_CODE splits 1007→299 and GEN_WAV 947→239 but moved the arm only 0.267→0.280
(−4.8 %) — ~708 splits were worth ~15 ms, ~0.02 ms each. C4 stays green
(WER 0.0455, `codec_eos` 46-69 frames).

**The CPU arm corrects §9.4.** The earlier "CPU cross-check 0.233" ran the
OpenCL-enabled `llama-tts` (`-ngl 0`), whose clip graph still carried its
splits. The same loop built with `GGML_OPENCL=OFF` (single CPU backend,
splits=1) runs the whole 6792-node predictor in **59.9 ms/frame** and the
backbone in 41.2 ms/frame: the predictor is **4.5× slower on the card
(272.4 ms) than on one CPU backend (59.9 ms)**, while device execution is only
~12 % of wall. The overhead is **per-node: ~6,600 tiny device kernels per
frame**, not copies (0.02 ms/split) nor launch enqueue (1.9 s/run). The code
predictor histogram (CPU arm, nodes 6792): `RESHAPE` 1242, `CPY` 774,
`MUL_MAT` 686, `CONCAT` 543, `MUL` 445, `VIEW` 357, `ARANGE` 354,
`RMS_NORM` 351, `ADD` 341, `PERMUTE` 288, `GET_ROWS` 282, `SCALE` 262,
`ROPE` 176, `FLASH_ATTN_EXT` 87; unsupported-on-OpenCL: `GET_ROWS`/`ARANGE`
(some variants), `SIN` 29, `LOG` 39, `SUM` 30, `COL2IM_1D` 6.

**Best arm: the hybrid at 0.828** — predictor and code2wav on CPU, backbone
on the device. Its 48.3 s process is predictor **29.7 s (61 %)** + code2wav
**13.5 s (28 %)** (1,692 ms/flush, no faster than the card's 1,813 ms) +
prompt 0.57 + speaker 0.80. It is **17 % short of real time** and is *not* a
card-only arm.

**Placement and overlap, second window (measured-here, 2026-10-19):** the same
B60 (PCI `8086:e211`), a second dev binary `arcint-pipe` `fd0890de…` (the B1
scratch tree + a second GEN_WAV clip context + an optional code2wav worker),
the fixed model pair, `gate-fleurs`, seed 42, the 8-sentence gate, **C4
unchanged and PASS (WER 0.0455) on every arm**:

| arm | placement | window_frames | process / 8 | RTF | frames / sentence |
|---|---|---|---|---|---|
| hybrid (B1, the reference) | predictor+codec CPU, backbone card | 72 | 48.3 s | 0.828 | 62,61,56,66,51,64,69,51 |
| seq | predictor CPU, **codec card** | 72 | 53.3 s | 0.751 | same 62,61,56,66,51,64,69,51 |
| seq16 | same, window 16 | 16 | 95.2 s | 0.420 | same counts; samples differ (below) |
| async | same + code2wav on a worker thread | 16 | 74.6 s | 0.536 | **66,50,45,72,48,57,60,54** |

- **Placement: the codec is cheapest on the CPU.** Its flush is 1,692 ms on
  the CPU (hybrid) against 1,780-2,000 ms on the card (seq; 14.8 s over the
  eight). The seq arm (predictor CPU, codec card, sequential) runs 0.751, **9 %
  slower** than the hybrid's 0.828: putting the codec on the card where the
  predictor does not need its CPU cores does not pay. `measured-here`.
- **The decoder flush cost is fixed per call.** seq16 decodes 256 codes
  (16 frames) in **1.78-1.80 s**, essentially the same as 992 codes (62
  frames) in **1.95 s**; the cost is the 6,492-node code2wav graph and its
  carried state, not the frame count. Splitting a sentence into 16-frame
  flushes multiplies decoder work (~28 flushes ≈ 50 s) and lands at 0.420.
  `measured-here`.
- **Chunking below the window is not seamless.** At the same window the card
  codec against the CPU codec differs by at most **483/32768** (0.0147; mean
  ~3.8); at window 16 against 72 the same sentence differs by up to
  **25,749/32768** (0.79 of full scale) at the same sample count, WER still
  0.0455. The
  `clip.cpp:1945` `hparams.wav_tfm_swa = 72` (the same value the helper's
  `window_frames` "must match") is the reason. `measured-here`.
- **Pipelining is refuted.** With `window_frames=72` a 51-69-frame sentence
  has **one** flush, at its end, so no intra-request flush exists to overlap;
  the gate's requests are sequential. Overlap would need a lower window, which
  multiplies the fixed per-call cost; the async worker (window 16, its own
  clip context and scheduler) lands at 0.536, worse than both the hybrid
  (0.828) and the sequential card arm (0.751). It does move CPU and card work
  together (74.6 s against seq16's 95.2 s), but the card's decoder queue plus
  the backbone's own card decode keep the wall a sum, not a max.
- **The async arm also changes the output.** Same seed and gate, its sampled
  frame counts drift to 66,50,45,72,48,57,60,54, where the deterministic arms
  give 62,61,56,66,51,64,69,51 (seq and seq16 reproduce one another exactly):
  the decoder on the card runs concurrently with the backbone decode on the
  card. The mechanism is not adoptable on correctness grounds.
  `measured-here`.
- **Time to first audio.** At the default window the first flush is the only
  flush, so a listener hears nothing until the whole sentence decodes:
  **5.3-6.8 s** (hybrid), **5.8-7.4 s** (seq). `measured-here`.

- **CPU thread split (both graphs on the CPU) not run.** It is the refuted
  async mechanism with the two clip contexts sharing the CPU's cores instead
  of the card's; the fixed per-call decoder cost (~1.69 s) would still
  multiply under a lowered window, and the two threads share cores. It is
  worth measuring only if lever (a) first cuts the decoder's per-call cost.
  `not measured`.

**Disposition (updated): the best measured arm stays the B1 hybrid at 0.828**
(mmproj graphs on the CPU, the backbone on the card); card-only real time
remains out of reach, and the overlap hypothesis is **refuted with numbers**
— the decoder's per-call cost is fixed, the flush window must stay 72 (its
seam), so there is nothing to overlap and lowering it costs both wall and
waveform. **Accepted by the operator 2026-10-19:** the production TTS unit
takes the hybrid placement (code2wav on the CPU, backbone on the card); its price is the
0.828-and-17 %-short row against the far worse card-only (0.751, 0.420) and
async (0.536) arms. Real time still needs the predictor's per-node count reduced
(lever b); the decoder's per-call cost is reduced by the default patch 0062
(lever a), not a placement or a thread.

**Open levers, in order by measured share:** (a) the code2wav fixed per-call
cost was **explained and its landing measured** (above: the always-72-frame
padded window; the valid-frames computation scales it with the flushed count,
C4 card PASS, and it is now the engine default patch
`0062-mtmd-qwen3tts-gen-valid-frames`); (b) the code predictor's node/launch
count (`cache_set`'s concat+arange+get_rows and the per-step sampling, ~1,600
of 6,792 nodes) — reduce nodes or fuse. Placement is settled by the disposition
above (the hybrid; accepted by the operator 2026-10-19); pipelining is closed
by the measured refutation.

`0061` (OpenCL `ARANGE`) is **measured, reviewed and dropped from the numbered
series.** Op suite on the B60 (the CPU reference): **2/2 PASS** green,
**0/2 FAIL** with `GGML_OPENCL_ARANGE_RED=1` (ERR 4.0, both cases), `support`
reports ARANGE SUPPORTED; the diff is a correct f32 kernel plus its
`supports_op`/dispatch, the red switch negating the step. `measured-here`. Its
gain is weak (RTF 0.267→0.280, −4.8 % wall, 1,007→299 GEN_CODE splits) and its
**only beneficiary is the card-only predictor path**, which the placement
decision does **not** adopt (the hybrid's mmproj graphs have splits=1 and never
run the card's ARANGE); a public PR-shaped patch whose path is out of reach
pays no maintenance. **Proposal, operator may veto.** The artifacts stay on
record (the three raw diffs, the op log) and it is regenerable as
`0061-*.patch` if a card-only path is ever adopted. The instrumentation is a
scratch build; the arms above name their binaries.

**The hybrid placement landed as a real option (2026-10-19, `code`).** The
scratch switch refuted by the placement window is now the documented
`--tts-gen-device cpu|gpu` flag (`gpu` default, the card-only line; `cpu` the
hybrid), config-parsed and unit-tested like the other `--tts` options; a
`cpu` request needs `--tts` and the device-free stub refuses it
(`src/config.{h,cpp}`, `tests/test_config.cpp`). `contrib/systemd/
arcint-tts.service` carries `--tts-gen-device cpu` (accepted by the
operator 2026-10-19; `docs/speech-service.md`). The flag is applied in `backend_llama.cpp`,
which sets `mtmd_context_params.use_gpu = false` for a `cpu` request. **The
placement is checked, not assumed:** the same code captures libmtmd's
`clip_init` line (`clip.cpp:192`, "CLIP using %s backend") while the `mmproj`
loads and refuses the boot if a `cpu` request built any clip context on the
card; `ARCINT_TTS_GEN_FORCE_GPU=1` is the red switch and
`tools/tts_placement_check.sh` runs red-then-green on the card host. C4 is
unchanged (the placement is the same arm the campaign measured). This landing
is `code`; the 0.828/0.751/1,692 ms numbers it rests on stay `measured-here`.

**Lever (a) characterized (measured-here, CPU, 2026-10-19).** The fixed ~1.8 s
code2wav flush is the **padded 72-frame `wav_tfm_swa` window**, not any of the
other suspects. The GEN_WAV graph has a constant shape:
`qwen3tts-gen.cpp build()` sets `n_frames = hparams.wav_tfm_swa` (72) and the
caller pads the codes to `n_frames_w * n_codes` (`clip.cpp`), so the decoder
computes 72 frames whether the caller flushes 16 or 72. A pure-CPU `llama-tts`
(`GGML_OPENCL=OFF`) measures, per flush: 16 frames 1.766 s, 44 frames 1.760 s,
62 frames 1.78 s, 72 frames 1.78 s; `compute_us` is 99.6 % of the flush
(1.75-1.78 s) while build 0.6 ms, alloc 0.8 ms, input 1.0 ms and output 2.0 ms
are 0.3 %: graph rebuild/realloc, per-call weight re-conversion and the input
copy are not the cost. A thread A/B (t = 16, 12, 8, 4, 2, 1) holds the flush at
1.763-1.766 s (t=1 the fastest): thread underuse is not it. `perf` on an
isolated flush (`-n 1`, window 1, ~96 % GEN_WAV): dense bf16/f16 `tinyBLAS`
gemms ~28 %, `im2col` ~11 %, `add`/`mul`/`concat`/`sin`/`sqr` the rest — im2col
is real but not dominant, the padded **matmul** work is.

- **One-variable experiment: compute only the valid frames -- adopted and
  landed as the engine default (`measured-here`, card + CPU, 2026-10-19).**
  The GEN_WAV graph takes the caller's valid frame count instead of the fixed
  72 (`models.h`/`qwen3tts-gen.cpp`/`clip.cpp`; patch
  `0062-mtmd-qwen3tts-gen-valid-frames`, `code`; no env switch, no arcint
  flag). On the CPU the flush becomes frame-proportional (44 frames 1.11 s,
  55 1.37, 60 1.49, 64 1.59 against the fixed 1.77-1.79) and the 8-sentence
  flush total falls 14.25 -> 10.60 s. On the B60 the one-variable card arm
  (the 6e68d58 source built without and with the patch, the full
  `/v1/audio/speech` route, `--tts-gen-device cpu`, window 72, the fixed
  8-sentence gate, seed 42) measures the padded production arm at real-time
  factor **0.729** (55.66 s process / 40.56 s audio, worst time-to-first-audio
  11.0 s) against the valid-frames arm's **0.802** (50.58 s; 9.52 s) -- +10 %,
  still short of real time. **C4 PASS on the card: both arms round-trip at
  corpus WER 0.0455** through `whisper-server -ng`, and `codec_eos` is sampled
  on both (40.56 s of audio, no cap). The same-binary sample diff is at most
  **191/32768** (0.0058 of full scale), inside the window-72 card-vs-CPU bar
  483/32768 already on record; the patch binary's own arm is bit-identical on
  all 8 sentences to the switch arm measured first and differs from the padded
  production arm exactly as that arm does -- the default engages the measured
  mechanism. The red case `MTMD_TTS_WAV_PAD=1` forces the padded window back:
  its 8 sentences are identical to the production padded arm and it runs 0.7285
  (the default 0.8006), so the patch's own
  claim (the flush scales with the frames) can fail by the repo's rule. The `-1`
  default keeps the padded window for a no-codes caller, and the generic
  `out_audio` block's null guard is kept (the upstream PocketTTS `GEN_WAV` path
  passes features instead of codes and never dereferences a null codes pointer).
  This arm's denominator and first-audio figures (11.0 s padded / 9.52 s
  valid-frames) differ from the earlier placement window's 0.828 and
  5.3-6.8 s on the same route and card: a different source window and its
  sampled lengths on the same manifest (run-to-run and source-window
  variation). The lever's **ratio** (+10 %) is what it claims. It is a
  deviation from the reference (upstream's `clip.cpp`/`tts.cpp` pad every flush
  to the attention window), named with its reason: the measured fixed per-call
  cost; the `wav_tfm_swa` constant stays for the padded and no-codes fallback,
  no invariant is at stake, and the answer-level bar permits it (`CLAUDE.md`
  "References first"). `0062` applies cleanly to the pin + series (checked
  against the 0001-0060 tree and the 0001-0058 tree). The lever is real, the
  fixed cost is explained, and the engine's default now computes the caller's
  frames.

## Where it lives

- The loop: `src/exec/backend_llama.cpp` (`generate_speech`); upstream
  `~/src/llama-port/tools/{tts,mtmd}`.
- The backend coverage: `ggml/src/ggml-opencl/ggml-opencl.cpp`'s
  `supports_op`; the open classes above; `COL2IM_1D`/transposed conv is the
  milestone §5 prediction now measured on the card.
- The valid-frames default: `contrib/llama.cpp/patches/
  0062-mtmd-qwen3tts-gen-valid-frames.patch` (`tools/mtmd/{clip.cpp,
  models/models.h, models/qwen3tts-gen.cpp}`).
- The gate: `tools/tts_gate.py`; the model set `docs/milestone-1.2.0.md` §2.
- The card arm's record: milestone §9.3, `docs/speech-service.md`.
