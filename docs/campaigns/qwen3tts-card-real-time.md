# qwen3tts-card-real-time — the Qwen3-TTS card arm is slower than real time, and not faster than the CPU: the gen graphs fall back to the CPU

**Open 2026-10-19.** The fixed model speaks German and stops at `codec_eos`
(`qwen3tts-german-genpath` closed), but the card arm through arcint's own
`/v1/audio/speech` runs at about **0.26× real time** on the B60 — slower than
real time, and no faster than the same loop on the CPU. The milestone
criterion B3 and the unit's serving shape both need real time on the card.

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

## Current state (measured-here, 2026-10-19)

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

**Disposition: on this (upstream) graph path, card-only real time is out of
reach with numbers, without a node-count change to the code predictor.** The
predictor is launch-bound on OpenCL (272 ms/frame vs 60 ms/frame single
backend), and the code2wav is ~1.8 s/flush on card and CPU alike. The split
reduction (`0061`) and per-op kernels for the six remaining unsupported classes
do **not** address the dominant per-node overhead.

**Open levers, in order by measured share:** (a) the code2wav node count
(6492 nodes, 947→239 splits, 1.8 s/flush *both* engines) — a CPU-side or
node-reducing restructure; (b) the code predictor's node/launch count
(`cache_set`'s concat+arange+get_rows and the per-step sampling, ~1,600 of
6,792 nodes) — reduce nodes or fuse; (c) the placement decision: the hybrid's
0.828 is an operator call (a card-only B3 vs a CPU-codec arm), with the price
of each side on this record.

`0061` (OpenCL `ARANGE`) is measured on the record (weak but non-regressing);
its `test-backend-ops` red/green and review are **owed** before it is a numbered
patch of the series. The instrumentation is a scratch build; the arms above name
their binaries.

## Where it lives

- The loop: `src/exec/backend_llama.cpp` (`generate_speech`); upstream
  `~/src/llama-port/tools/{tts,mtmd}`.
- The backend coverage: `ggml/src/ggml-opencl/ggml-opencl.cpp`'s
  `supports_op`; the open classes above; `COL2IM_1D`/transposed conv is the
  milestone §5 prediction now measured on the card.
- The gate: `tools/tts_gate.py`; the model set `docs/milestone-1.2.0.md` §2.
- The card arm's record: milestone §9.3, `docs/speech-service.md`.
