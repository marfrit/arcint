# Pass 4 — the kept candidates against arcint's own code (2026-11-05)

Scope: candidates 1-5 and 7, plus 10 as a parity check. #6 and the fp8/QKV-fp8 variants are
**listed, not audited** (they change the answer by design). Nothing was built or run. `arcint` =
the workspace at `d4b6169` (patches 0001-0098); `llama` = llama.cpp at the pin `3e4dfa8` with the
served series applied, extracted at `~/research/ipex-llm/llama-served`. Line numbers are that
patched tree's. `code` = read; `code, symbol only` = a name; no `measured-here` in this file.

Quality filter (operator, 2026-10-29): exact and rounding-only in scope, judged by `CLAUDE.md`'s
answer-level bar (KL ≤ 0.03 nats worse than the baseline arm, argmax −1 point, the facts/needle
right); answer-changing levers are listed only. A rounding-only A/B needs the noise-twin floor
(concept-mining #20) before its result counts.

| case | arcint has | class | units | price (`paper`) | cost |
|---|---|---|---|---|---|
| 1 prefill GEMM, materialized-f16 + vendor tiling | **partly**: q4_0 → f16 tile → XMX GEMM (0096) | rounding-only | agent, coder | B60 pp512 +15.3 %, A770 +23.6 % | medium |
| 2 QKV projection + norm + RoPE as one op, combined wqkv | **partly**: norm+RoPE fused (0074, opt-in); projection separate, no combined weight | rounding-only (BF16/f16) | all three | none stated | medium |
| 3 MLP fused gate/up/SiLU/down, q4_0 | **partly**: q4_K GEMV gate+up+GLU fuse (default); no q4_0, no down, no prefill | rounding-only | coder, agent (decode) | none stated | medium |
| 4 weight repack at load + card-shaped matvec | **lacks**: raw q4_0 blocks, legacy matvec shapes | exact | all three (decode) | none stated | low |
| 5 q4_0 GEMM for 16-64 rows | **partly**: 0009 few-column XMX up to 8 columns | rounding-only | all three (verify/MTP) | none stated | low-med |
| 7 MoE route/top-k | **has**: 0064 fused router; grouped routing not our model | exact | Flash-Next | — | done |
| 10 prefill attention | **has**: DPAS at head 128/256 (0008/0041/0045) | — | agent, coder | no gap read | — |

---

## 1. The prefill GEMM: a materialized f16 operand and a vendor-grade tile

- **What arcint has** (`code`): Q4_0 weights from 16 columns take `kernel_mul_mm_q4_0_f16`
  (`llama/ggml/src/ggml-opencl/kernels/mul_mm_kq_f16.cl:2240`), built at
  `ggml-opencl.cpp:2860`, dispatched at `:29180`; 1-4 columns take `kernel_mul_mv_q4_0_q8_1`
  (`:29158`), 5-15 columns stay upstream's float path. The GEMM dequantizes each K tile into local
  memory and runs the K-quants' XMX tile; `GGML_OPENCL_Q40_XMX=0` keeps the float kernels
  (`:29136`). Measured in 0096's record: pp512 B60 2332.94 ± 3.28, A770 1956.15 ± 2.93
  (`measured-here`), and the tiled GEMM is ~95 % of a 512-token prefill.
- **How its build differs from IPEX's** (`code`): IPEX (`binary.md` §5) dequantizes the **whole**
  src0 to f16 in a pooled scratch buffer and then calls a oneDNN matmul over that buffer; arcint
  dequantizes the same values **a tile at a time inside the GEMM**. Same value class; different
  granularity of the dequantize and different tiling/scheduling.
- **Quality class**: rounding-only (q4_0's decoded values are byte-identical; the f16 accumulation
  order differs). KL bar; the twin floor applies.
- **Units**: the agent's 512-token prefill and the coder's 1,024 are the unit shapes; the Llama 2
  512 row is the table and this candidate serves it first.
- **Measurement that prices it**: (a) IPEX's own arm, `GGML_SYCL_DISABLE_DNN=0` vs `=1` on the same
  card, isolates the vendor-GEMM share; (b) arcint's q4_0 GEMM's achieved GFLOPs from a
  `GGML_OPENCL_PROFILING` run at 512 rows, against the card's XMX peak — if we are below ~70 % of
  peak, a tiling arm has room; (c) an A/B of the materialized-f16 path below.
- **Implementation sketch**: reuse the existing f16 XMX GEMM (`kernel_mul_mm_f16w_f16`, built
  `ggml-opencl.cpp:1165`, `:2854`, dispatch `:29416-29421`) and add a per-ubatch dequantize-to-f16 into a scratch
  buffer sized to the largest weight (a new `kernel_q4_0_to_f16` in `cvt.cl`-style, one row a
  work-group), then dispatch `mul_mm` on it when `ne1 >= 16` and the scratch fits. Red-first
  tests: `test-backend-ops -o MUL_MAT -p type_a=q4_0` 161/161 both cards with the new path and
  with it off; the 0096 mutant (`(char4)(7)`) still fails its 43/44 cases and none from 16; a
  token comparison against the CPU reference, KL inside 1e-4; the fallback when the scratch does
  not fit keeps the tile path.

## 2. QKV projection + norm + RoPE in one op, and a combined weight

- **What arcint has** (`code`): the norm+RoPE fusion exists and is opt-in — patch 0074,
  `GGML_OPENCL_FUSE_QK_NORM_ROPE=1` (`ggml-opencl.cpp:8640`), the `can_fuse` 3-node gate
  (`:10964`; the 3-node pattern at `:9906`), kernel `kernel_rms_norm_mul_rope_multi_f32`
  (`kernels/rms_norm.cl`), the 2-node norm+mul prefix is checked after it (`:10956`). It fuses
  the *post-projection* chain only, three kernels a side of the layer before it.
- **How its build differs** (`code`): IPEX's graph emits **one** op for projection+norm+RoPE
  (`build_bigdl_qkv_fusion`), with either three weights or a single `wqkv` weight; the Python path
  merges the weights at load (`models/common.py:379-412`, `torch.cat` of the three quantized
  weights into one `FP4Params`). The kernel geometry is small-row ESIMD (binary §2), so this is
  a decode-first lever; the prefill behaviour of the fused op is unread.
- **Quality class**: rounding-only (the op output is BF16; the norm and rope expressions are
  arcint's own 0074 expression if the port uses it). KL bar.
- **Units**: all three at decode/verify; the coder's 4-6-row verify is where a 2×16 tile lands.
- **Measurement**: launch count per layer under `GGML_OPENCL_PROFILING`, tg128 with
  `GGML_OPENCL_FUSE_QK_NORM_ROPE` on/off; a q4_0 qkv-GEMM A/B (three calls vs one merged-weight
  call) at pp512 and at 6 rows.
- **Implementation sketch**: (a) enable 0074 in the served defaults after its card A/B; (b) merge
  the q/k/v quantized weights at load along `out` (a `ggml_cpy` run or a load-time
  `torch.cat`-equivalent in `llama_model_loader`/`src/exec`), so one GEMM writes a [3·n_embd] row;
  (c) keep 0074 as the epilogue. Red-first: 0074's geometry cases in `test-backend-ops` (RMS_NORM
  + ROPE) with the fused switch on and off; merged-weight tokens identical to the three-weight
  path; the q4_0 `MUL_MAT` suite with the merged shape.

## 3. The MLP fused across gate/up/SiLU/down

- **What arcint has** (`code`): `fuse_mm_glu` default-on (`ggml-opencl.cpp:760`), the
  MUL_MAT+MUL_MAT+GLU sub-graph gate (`:9836-9878`), kernel
  `kernel_gemv_noshuffle_q4_k_f32_glu` (`:1563`) — a **GEMV** fuse for a q4_K FFN. The check
  requires two sibling MUL_MATs and a GLU, no SWIGLU_OAI (`:9878`); it fires at decode shapes.
  The MoE version is the MUL_MAT_ID/ADD_ID/GLU fuse (`:9475-9537`).
- **How its build differs** (`code`): IPEX's `mlp_forward_*` kernels also fuse the pair, and its
  Python gate is **one row** (`models/utils.py:291`), so it is the same decode class; the q4_0
  form and the `upgate` single-weight form are what arcint lacks. The down projection is separate
  on both sides (the kernel returns the GLU result, `models/common.py:150-160`). So the readable
  gap is q4_0 coverage, not a prefill fusion.
- **Quality class**: rounding-only; KL bar.
- **Units**: coder and agent at decode; no prefill prize read.
- **Measurement**: tg128 with `GGML_OPENCL_FUSE_MM_GLU=0/1` on a q4_0 model; the fused kernel's
  time under profiling; `[FUSE_MM_GLU]` debug lines count the fires (`:10166`).
- **Implementation sketch**: instantiate the existing `_glu` GEMV for q4_0's planes (the
  `kernel_mul_mv_q4_0_q8_1` decode already exists, `:29158`) and add the q4_0 branch to the gate;
  red-first: the gate's debug fire count for q4_0, tokens identical to unfused, 0096's
  `MUL_MAT` suite untouched.

## 4. Weight repack at load

- **What arcint has** (`code`): none for q4_0 (the 0096 record reads Q4_0 as IQ4_NL's flat planes;
  no load-time repack appears in the series — a search of the patches found no q4_0 reorder).
- **How its build differs** (`code`): IPEX's `reorder_qw_q4_0` writes the weights in the matvec's
  read order at load (`binary.md` §2/§4), and its matvec has per-card geometries
  `<32,16,2|8>`/`<64,32,8>`.
- **Quality class**: exact — a permutation, no value change.
- **Units**: all three at decode (the matvec is bandwidth-bound; a coalesced read order is the
  lever).
- **Measurement**: tg128 on Llama 2 Q4_0 and the coder with repack on/off; the matvec's achieved
  GB/s at 400 tokens.
- **Implementation sketch**: a load-time kernel (or a host pass) that permutes each 32-weight
  block into the order `kernel_mul_mv_q4_0_q8_1` reads contiguously; the dequant path stays
  byte-equal. Red-first: `test-backend-ops -o MUL_MAT -p type_a=q4_0` 161/161 both, the same
  tokens on and off, and a block-permutation inverse test.

## 5. A q4_0 GEMM for 16-64 rows

- **What arcint has** (`code`): 0009's few-column XMX GEMM covers up to 8 columns; 0054
  (`opencl-lowbit-gemm-decode`) serves the decode GEMM shapes. From 16 columns the full q4_0 XMX
  GEMM (0096) is the only path.
- **How its build differs** (`code`): IPEX's batch kernel gate is 2-64 rows with
  `x.shape[1] % 128 == 0` (`low_bit_linear.py:292-333`), and on BMG it takes >1 rows for
  SYM_INT4/WOQ_INT4/FP8E5; the kernel is `vec_q4_0_batch_kernel<8,2,16|32,…>` with f32
  activations. arcint's 8-column cap leaves 16-64 rows on the big tile, which is a different
  trade (padding and K-loop) than a small-row tile.
- **Quality class**: rounding-only.
- **Units**: the coder's 4-6-row verify and MTP, the agent's 6-row verify; prefill chunks of
  16-64 rows on Flash-Next.
- **Measurement**: per-shape `MUL_MAT` time at 16/32/48/64 columns on both cards (test-backend-ops
  `-b`), and a served verify step's ms.
- **Implementation sketch**: a tile of 16-64 rows over the existing q4_0 decode (q8_1 x8 dot) or
  a small-M XMX tile; red-first: `test-backend-ops MUL_MAT` at those batches, and a token
  comparison on the verify path.

## 7. MoE route (already ours) and 10. prefill attention (parity)

- **7** (`code`): arcint's fused router is `kernel_topk_moe_softmax_norm` (patch 0064's record:
  softmax, top-k, clamp, normalise in one call; 34.1 us a call before, 40 calls of the coder's
  19.25 ms token on the A770). IPEX's `moe_softmax_topk` is the same idea; the *grouped* top-k
  (DeepSeek routing) is not our units' shape. **No gap for Flash-Next.**
- **10** (`code`): arcint's DPAS prompt attention at head 128/256 (`flash_attn_dpas.cl:25` `DK
  256`, built for `dk` 256 and 128 `ggml-opencl.cpp:4719-4725`, dispatch `:23175-23209`); IPEX's
  causal XMX SDP at 64/128 blocks is the same class. The only readable difference is the fp8
  variant (quality-changing, listed). **No port.**

## What this audit does not cover

- The closed binary's fused-op shape threshold and its oneDNN primitive dtype: candidates 1/2/3
  are priced only by the card measurements above, and a negative never tried is not verified.
- prefill at 20k on Flash-Next: the candidate 1 mechanism is dense-geometry; the expert stream
  stays the wall there (`research-concept-mining.md`'s measured stage), so no claim is made.
