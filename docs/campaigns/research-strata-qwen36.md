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

## Re-count (`measured-here`, 2026-10-25)

The 1.2.6 tree built with `GGML_OPENCL_PROFILING` (one kernel launch a row of
`cl_profiling.csv`), the coder's unit flags, the A770, one prose prompt at
temperature 0, two runs of 100 and 300 decode tokens each, so the difference
is 200 tokens of decode; the profiling build itself decodes slower (event
profiling), so its device times count and its wall rates do not.

| | launches | device time | matmuls | attention | other |
|---|---|---|---|---|---|
| plain, a token | 1,635 (0002 era: 2,034) | 19.25 ms | 9.86 ms (451) | 2.86 ms (20) | 6.54 ms (1,164) |
| MTP 4, a verify step (2.63 tokens) | 1,962 | 39.14 ms (14.87 a token) | 26.23 ms (610) | 4.10 ms (28) | 8.81 ms (1,323) |

Against the normal build's ~21 ms a plain token, the step is device-bound:
the launches' host cost is not the wall, the small ops' device time is. The
largest "other" kernels a plain token: the MoE router
`kernel_topk_moe_softmax_norm` 1.37 ms (40 launches, ~34 us each, for a
softmax and top-k over 184 experts); `kernel_gemv_f32_f32_mc` 0.74 (100);
`kernel_rms_norm_mul` 0.65 (131); `kernel_get_rows_f32` 0.49 (62);
`kernel_quantize_q8_1_sg` 0.48 (161); the gated delta-net 0.35 (30);
`kernel_mul` 0.35 (70); `kernel_add_row` 0.33 (120); the f32 copies 0.39
(60); `kernel_swiglu` 0.21 (80); `kernel_rms_norm` 0.19; `kernel_moe_combine_f32`
0.18; `kernel_concat_b4_pack` 0.15. The matmuls' 9.9 ms a token is not
amortized by MTP (9.97 a token at MTP 4: the window's rows route to
different experts).

## Transferable, ranked for the coder

1. **Re-count** — done (above): 1,635 launches and 19.25 ms of device time
   a plain token, of which 6.54 ms in 1,164 small-op launches; device-bound.
2. **Speculative-sampling acceptance** for sampled requests (`code` reference
   `q36_sampling.hpp:180-307`): the coder unit samples at 0.6; the gain is the
   acceptance difference between sample-and-compare and min(1, p/q), measured
   on the coder at its thinking row.
3. **A draft gate on the coder** (`--llama-mtp-min-p`, built, used by
   Flash-Next): an A/B on the coder's unit flags, cheap.
4. **The fusion campaign** (portable kernel structure), sized by item 1:
   Strata's fused kernels as the reference, mapped onto the kernel-gap
   campaign's open levers (B activation quantize, C routing chain, D the
   shared-expert tail into the combine, E SwiGLU in the matvec epilogue, F the
   recurrent state in place, G-I residual + norm and the q/k fusions),
   estimated at 9-10 ms of a 26.5 ms coder step on the 0002-era trace
   (`llama-engine-kernel-gap.md`, `measured-here`); the re-count puts the
   small ops at 6.54 ms of 19.25 (34 %) and the first target at the MoE
   router (1.37 ms a token, Strata's `k_route_group` the reference), then the
   norm/quantize chain (rms_norm_mul + quantize + add_row + mul, ~1.8 ms
   together, Strata's `k_add_rms_q` / `k_moe_out_s8`), the F32 matvec
   (0.74 ms) and the copies/concat (0.54 ms). The matmuls' own efficiency
   (9.9 ms a token for the active weights) is the other half and a separate
   lever. One fusion at a time in
   ggml-opencl, each with 0049's bit-identity check (a regrouped rounding
   kept KL inside the bar but moved MTP acceptance) and the answer-level
   bar; the implementation goes to a non-Qwen implementer (the house rule).

Not transferable: the tensor-core attention, `cp.async`, stream priorities
(CUDA), and the hybrid expert machinery (the coder is resident).
