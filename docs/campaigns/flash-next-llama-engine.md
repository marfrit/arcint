# flash-next-llama-engine — Qwen3.8-Flash-Next served by the libllama engine on one Arc card + RAM

**Open.** Stage 1 (static placement, IQ kernels) built and served; KL leg
passed; the baseline arm's decode rate on the B60 owed.

## Charter

Serve Flash-Next (`qwen4exp`, 48 layers, 512 experts of which 10 a token,
the UD-Q3_K_XL GGUF: 52 GiB of experts in IQ3_XXS / IQ4_NL / IQ4_XS / Q8_0,
a 26.8 GiB per-layer embedding table, ~4 GiB of Q8_0 rest) through
`--engine llama` on the B60, the experts that do not fit VRAM in RAM and on
NVMe, with the mechanisms of the roadmap's P1-P6 rebuilt in this engine.

Stage 1: whole layers' experts on the card as far as they fit, the rest
computed by llama.cpp's CPU backend from RAM; the IQ types' matrix kernels
on Intel (they do not exist in ggml-opencl at the pin). Stage 2: Strata's
per-expert cache inside the OpenCL backend's `MUL_MAT_ID` (slots sized per
layer, decayed-usage swaps, non-blocking admission, the miss split over the
link, the doorbell), replacing the per-layer split.

## Reference to follow

- Strata (`~/src/Strata-ref`, `code`; source map and effects in
  `expert-hot-set-lru.md` and `tier-handoff-doorbell.md`, which built the
  same mechanisms on the OpenVINO path): the expert cache
  (`src/core/expert_cache.cpp`, `src/program/generate.cpp:4413-4482`), the
  miss split (`src/core/expert_source.cpp:1614-1696`), the doorbell
  (`src/core/layer.cpp:995`, `src/kernels/cuda/elementwise.cu:186-311`).
- llama.cpp at the pin (`code`): `--n-cpu-moe` / `-ot` place a layer's
  experts per tensor (`src/llama-model-loader.cpp:1235-1260`); the CPU
  backend's `MUL_MAT_ID` touches only the selected experts
  (`ggml/src/ggml-cpu/ggml-cpu.c:1663-1667`); the OpenCL backend declares no
  mmap support, which turns mmap off for the whole model under
  `--load-mode auto` (`src/llama-model.cpp:1512-1519`), and no `offload_op`.
  The IQ dequantizers to follow: `ggml/src/ggml-quants.c`
  (`dequantize_row_iq3_xxs`, `_iq4_nl`, `_iq4_xs`) and the CUDA MMVQ dots
  (`ggml/src/ggml-cuda/vecdotq.cuh`).

## Gate

On the B60, the Flash-Next GGUF from NVMe, against the stage's baseline arm
(stage 1: every expert on the CPU, `-ot exps=CPU --load-mode mmap`) in the
same window:
- the answer-level bar (`CLAUDE.md`): the capital check and the 20,085-token
  needle answered; mean KL against the baseline arm's reference within 0.03
  nats of the baseline, top-1 agreement down by at most 1 point;
- per-phase speed: decode faster at 4k and 20k, prefill within the spread or
  faster;
- reported beside the rate: VRAM and RAM in use, the experts' split, and
  for stage 2 the GPU hit share and the share of misses over the link.

For scale (not a gate): the OpenVINO path's Flash-Next on the B60 (`d48q8`,
patches 0076/0077, `measured-here` 2026-10-02) decodes 12.7 t/s after the
20,085-token needle and 16.0 t/s on the 500-token long answer.

## Current state

- Stage 1 built (`contrib/llama.cpp/patches/0002`: IQ3_XXS / IQ4_XS / IQ4_NL
  matvec and GEMM on Intel; arcint `--llama-cpu-moe N --llama-threads N`,
  `qwen4exp` admitted). `measured-here` 2026-10-03, B60, the GGUF from NVMe:
  - llama-bench (512-token prefill / 64 decode tokens, page cache warm):
    every expert on the CPU 0.2 t/s decode (A770, from ZFS); 14 expert
    layers on the card 44.4 / 9.0 t/s; 17 layers, CPU repacking on, 8 CPU
    threads 54.0 / 12.6 t/s (more layers overcommit the B60's 22.7 GB, and
    xe then pages VRAM over the link: 0.05 t/s on the A770 at 15.9 GB);
  - served through arcint (16 expert layers on the card, 32,768 context):
    the capital check "Paris"; the 20,045-token needle answered
    ("ORANGE-FALCON-77") at 49.2 t/s prefill; the 500-token long answer at
    10.9 t/s decode. The OpenVINO path's arms after 0076/0077: needle decode
    12.7 t/s, long answer 16.0 (`expert-hot-set-lru.md`,
    `tier-handoff-doorbell.md`).
- KL leg (`measured-here` 2026-10-03, B60, `llama-perplexity` 8 x 512
  tokens, reference: the whole model on the CPU backend): baseline arm
  (every expert layer on the CPU, `-ncmoe 48`) mean KL 0.0289 +- 0.0018,
  top-1 95.54 %; stage 1 (16 expert layers on the card, `-ncmoe 32`) 0.0294
  +- 0.0019, 95.59 %: +0.0006 nats, top-1 +0.05 points. Passes.
- Owed for the stage gate: the baseline arm's decode at 4k and 20k on the B60
  from NVMe (the 0.2 t/s above is the A770 from ZFS).
- Under `--load-mode auto` the load reads the whole model into anonymous
  memory and is OOM-killed; arcint sets mmap whenever experts stay on the CPU.

## Where it lives

`contrib/llama.cpp/patches/` (ggml-opencl kernels, later the expert tier);
`src/exec/backend_llama.cpp` (placement flags).
