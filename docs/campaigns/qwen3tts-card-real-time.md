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

Disposition: the card arm is **not** compute-bound on the card — it is a
947-island graph whose unsupported f32 ops (the code2wav transposed-conv
`COL2IM_1D` among them, milestone §5) execute on the CPU between OpenCL
segments. The CPU arm of the same loop being no slower confirms it:
per-frame tiny-batch work and CPU islands dominate the wall clock.

Open: B1's per-node profile (`GGML_OPENCL_PROFILING`), then which of the
seven op classes and which split boundary carries the share; only then a
kernel lever (each a numbered patch with a red case, as the series requires).

## Where it lives

- The loop: `src/exec/backend_llama.cpp` (`generate_speech`); upstream
  `~/src/llama-port/tools/{tts,mtmd}`.
- The backend coverage: `ggml/src/ggml-opencl/ggml-opencl.cpp`'s
  `supports_op`; the open classes above; `COL2IM_1D`/transposed conv is the
  milestone §5 prediction now measured on the card.
- The gate: `tools/tts_gate.py`; the model set `docs/milestone-1.2.0.md` §2.
- The card arm's record: milestone §9.3, `docs/speech-service.md`.
