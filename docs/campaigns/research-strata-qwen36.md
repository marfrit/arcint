# research-strata-qwen36 — Strata's Qwen3.6-35B-A3B engine against our coder

**Opened 2026-10-25.** A public report put Qwen3.6-35B-A3B (UD-Q4_K_XL) at "up
to 300 tok/s" decode and ~7,000 tok/s prefill on a 16 GB RTX 5080, on a fork
of Strata. Our coder is the same family (`qwen35moe`, an A3B MoE with an MTP
head), fully resident on the Arc A770 16 GB, at 44-51 t/s decode on the served
check's prose answers and 73-79 t/s on code (MTP 4). This note reads the
fork's source to say where the difference comes from and what transfers.

Sources: the fork `github.com/xyehya/Strata`, branch `q36-r2` (HEAD
`379f0950`, 57 commits on upstream Strata `5c8dd322`), its model card
`huggingface.co/ykarout/Strata-Qwen3.6-35B-A3B` (`paper`), upstream Strata's
Qwen3.6 engine path (`src/qwen35/`). Paths below are in the fork.

## What the numbers are (`paper`, the model card)

RTX 5080 16 GB, 64 GB DDR5, the server, fresh code prompts, 384 tokens at
temperature 0.7, MTP 2: decode 228 t/s at 4K down to 159 at 240K; an agent
run (opencode) 240-288 t/s; prefill 6,500-7,000 t/s at 4-16K. The model does
not fit the card: 80 experts per layer in VRAM (54 with the default Q8
packing), the rest computed by a CPU pool. The "up to 300" is the agent run's
peak. MTP acceptance 73-86 % (prose) and 88-95 % (agent) counts only the
drafts that passed the `--spec-min-p 0.5` gate (`q36_engine.cu:5419`), at
depth 2 and sampled: it is not comparable to our per-draft figures at depth 4
without the same denominator.

## How the engine decodes (`code`)

- **One batched verify forward** over the window (`q36_engine.cu:4082`): the
  matmuls take up to 8 columns a launch (`:2528-2537`), each expert's weights
  are read once for the window, one int8-KV attention pass serves every row
  (`q36_attn.cuh:340`, `:626`), the gated delta-net rows are computed without
  writing state and `commit()` replays the kept rows (`q36_engine.cu:2954-2963`).
- **Few, fused launches**: ~18-20 a layer at one row, ~750-800 a forward over
  40 layers (counted in `layer()`, `:2541-2865`). Fused: add + RMSNorm + q8_1
  quantize (`k_add_rms_q`, `q36_kernels.cuh:216`); MoE combine + shared-expert
  gate + residual + the next norm + the next quantize (`k_moe_out_s8`, `:327`);
  router top-k + grouping (`k_route_group`, `:417`); the router and the
  shared-expert gate as one 257-row product (`q36_engine.cu:2795`, `:2809`);
  the shared expert's gate + up + SwiGLU + quantize (`k_sh_gu`,
  `q36_kernels.cuh:169`). CUDA graphs are used only when every expert is in
  VRAM (`q36_engine.cu:4396-4397`); the 5080 run is hybrid, so it runs
  without graphs and with a host wait on the routing each layer (`:3072`).
- **MTP** (`spec_step`, `:4033-4110`): the head reads the main model's
  final-normed hidden (`:2867`, `:2884`); after a window accepts `a` drafts
  the accepted positions are run through the MTP block again, seeded with the
  main model's verified hiddens (`:4039-4050`); later drafts chain on the
  block's own output and stop at the first below P (`--spec-min-p`,
  `:4053-4071`). `--mtp-sampled q` is speculative sampling: the draft sampled
  from q, kept with min(1, p/q), a rejection resampled from norm(max(0, p - q))
  (`q36_sampling.hpp:187-190`, `:300-307`).
- **Expert placement** (hybrid only): profile-seeded slots, a rebalance every
  32 tokens, a CPU pool for misses beside the GPU's resident experts; the GPU
  does not read expert weights from host memory (`--hybrid-zero-copy` maps
  only ids and activations, `:490-503`).

## Against our coder

| mechanism | Strata (`code`) | arcint, the coder (A770) | disposition |
|---|---|---|---|
| verify window | one batched forward, weights once a window | one batched verify graph (libllama, `llama_spec.cpp`) | equivalent (`code`) |
| MTP pairing | accepted positions re-run with verified hiddens | the same: rows 0..a with the target's rows, chained drafts leave before the verify (`llama_spec.cpp:7-21`) | equivalent (`code`) |
| draft count, gate | 2 drafts, stop below p 0.5 | 4 drafts with the 40,960-row draft vocabulary, no gate on the coder unit; 2 drafts measured 86-90 % accepted, 4 drafts 77.6 % and faster (73.6-79.3 t/s, `mtp-cycle-wall.md`) | measured here for our card; Strata's gate untried on the coder |
| sampled acceptance | speculative sampling (`q`) | sample-and-compare; the coder serves at temperature 0.6 | **not built** (`mtp-cycle-wall.md` names it): hardware-independent |
| launches a forward | ~750-800, fused | ~2,034 a token on the 0002-era build (`llama-engine-kernel-gap.md`; patches 0049 and 0051 removed some since; not re-counted) at a 2.2-2.6 us floor a launch on the A770 | **the largest gap**: the kernel-gap campaign's levers B-E, G-I are this list |
| fused ops | add+norm+quantize, combine+gate+residual+norm+quantize, router+top-k, shared gate+up+SwiGLU+quantize | ggml's graph, with ggml-cuda-style fusions where ported (topk_moe, 0049's three, 0051's GDN gating) | partly; the rest open |
| int8 KV decode attention on tensor cores | `mma.sync` (sm_80+) | q8_0:q4_0 KV, DPAS kernels (0015-0020) | CUDA-specific; ours exists in Intel form |
| CPU expert pool, prefetch, lend-staging | the 5080 run's main structure | not needed: the coder is resident | not applicable |
| GPU shortlist sampling | `--sample-gpu`, +3 % (`paper`) | sampling on the CPU, 0.03 s over a 300-token answer (`measured-here`) | not worth it here |

Bandwidth frame (an estimate, not a measurement): the 5080 has ~960 GB/s,
the A770 ~560 GB/s; the same efficiency would put our coder near 0.58 x the
5080's figure. Our weights' share of a decode step was 6.3 of 22.9 ms of
device time (`llama-engine-kernel-gap.md`, `measured-here`, the 0002-era
trace): the step is bound by the small ops and the launches around the
matmuls, not by the weights. That is where Strata's fused kernels go.

## Transferable, ranked for the coder

1. **Fewer launches a verify** (portable kernel structure): the fusions above,
   in ggml-opencl, one at a time with the bit-identity check of 0049
   (`feedback`: fusions must keep the rounding, or MTP acceptance moves). First
   step owed: re-count the coder's launches a verify on the current series
   and the device time outside matmuls and attention (`GGML_OPENCL_PROFILING`).
2. **Speculative-sampling acceptance** for sampled requests (`code` reference
   `q36_sampling.hpp:180-307`): the coder unit samples at 0.6; the gain is the
   acceptance difference between sample-and-compare and min(1, p/q), measured
   on the coder at its thinking row.
3. **A draft gate on the coder** (`--llama-mtp-min-p`, built, used by
   Flash-Next): an A/B on the coder's unit flags, cheap.

Not transferable: the tensor-core attention, `cp.async`, stream priorities
(CUDA), and the hybrid expert machinery (the coder is resident).
