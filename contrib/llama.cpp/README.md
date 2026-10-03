# Patches carried against the pinned llama.cpp

The `--engine llama` executor (`src/exec/backend_llama.cpp`,
`docs/llama-engine.md`) builds llama.cpp as a subproject from
`ARCINT_LLAMA_DIR`. The pin is ggml-org/llama.cpp commit `bed0a85`
(2026-10-02); the tree is that commit with the patches below applied in
order:

    git -C <llama.cpp> checkout bed0a85
    git -C <llama.cpp> apply <arcint>/contrib/llama.cpp/patches/*.patch

A patch that does not apply cleanly to the pin is a bug in this directory, not
a reason to move the pin. Each patch stays PR-shaped, so it can be offered
upstream.

## 0001-opencl-intel-kquant-integer-dot-and-xmx.patch

ggml's OpenCL kernels are tuned for Adreno; on Intel Arc the K-quant matrix
products were the whole cost of the two served models, and K-quant
`MUL_MAT_ID` was not supported at all, so llama.cpp kept a K-quant MoE's
experts in host RAM and ran them on the CPU. The patch adds, for Intel GPUs
with `cl_khr_integer_dot_product` (both Arc cards here) and only there:

- **Matrix-vector, 1 to 4 columns** (`kernels/mul_mv_kq_q8_1.cl`): the
  activation quantized per 32 values to two int8 terms (codes, and the codes
  of what they leave), the weights multiplied with 4x8-bit integer dot
  products, the shape of llama.cpp's CUDA MMVQ. Q4_K, Q5_K, Q6_K, on ggml's
  flat planes; 1 to 4 columns per pass at the cost of one (a speculative
  verify, a few lanes); `MUL_MAT_ID` one (token, slot) pair per work-group,
  the expert read on the device.
- **Matrix-matrix** (`kernels/mul_mm_kq_f16.cl`, needs
  `cl_intel_subgroup_matrix_multiply_accumulate`): weights dequantized to f16
  per work-group tile in local memory, laid out so one sub-group block read is
  one DPAS operand; activations as f16 columns (a power-of-two scale per
  column keeps them under f16's range); fp16 DPAS on the XMX units with f32
  accumulation. `MUL_MAT_ID` sorts the (token, slot) pairs by expert on the
  device and runs one tile of one expert per work-group.
- `MUL_MAT_ID` with Q4_K/Q5_K/Q6_K reported as supported on these devices,
  so the experts load into VRAM; and K-quant `MUL_MAT` of any column count
  exempt from the stock Adreno rule that declines 512 or more columns
  without the Adreno GEMM. llama.cpp asks `supports_op` with a 512-token op
  when it places weights, so that rule had put every small K-quant tensor
  on the CPU -- for the coder, the 60 small tensors of its gated delta-net
  layers, one GPU-CPU-GPU round trip per layer per token.
- The activation of matmuls that share an input (q/k/v, gate/up) converted
  once per graph.

The sub-group size of the DPAS kernels follows the device (8 where it is
supported, i.e. Xe-HPG; 16 on Xe2); a program a device refuses to build is
skipped and ggml's kernels take that case.

Switches: `GGML_OPENCL_KQ_Q8_1=0` turns all of it off (ggml's kernels, a
K-quant MoE's experts on the CPU); `GGML_OPENCL_KQ_MV=0` turns off the
`MUL_MAT` matvec (the `MUL_MAT_ID` matvec stays); `GGML_OPENCL_KQ_MM=0` turns
off the GEMMs; `GGML_OPENCL_KQ_MM_ID_MIN_TOK` (default 16) is the token count
from which `MUL_MAT_ID` takes the GEMM instead of the matvec;
`GGML_OPENCL_KQ_DEDUP=0` converts every activation; tile overrides
`GGML_OPENCL_KQ_MM_TILE` / `GGML_OPENCL_KQ_MM_ID_TILE` ("RG,TG,NSGM,NSGN",
checked against the device's local memory and work-group size) and
`GGML_OPENCL_KQ_Q8_1_NDST`.

Design references (`code`): llama.cpp's CUDA `vec_dot_q4_K_q8_1` and its
siblings (the q8_1 activation and the per-sub-block scaling); the DPAS operand
layouts from the emulation in bashbaug/SimpleOpenCLSamples
`20_matrixexperiments-{i8,bf16}`. An int8-activation DPAS GEMM (the CUDA MMQ
shape) was built first and dropped: it reached 20-30 TOPS but moved the dense
27B's prefill KL by 0.0028 nats and top-1 agreement by 0.9 points; the f16
GEMM is faster and leaves the KL where ggml's float kernels have it.

Measured here (`measured-here`, 2026-10-03; A770 GT clock pinned at
2000 MHz; llama-bench `-p 512 -n 128`, default batch 2048 / ubatch 512, one
sequence):

| | stock `bed0a85` | patched |
|---|---|---|
| Qwen3.8-27B Q4_K_M, B60: prefill 512 | 71.1 t/s | 417.9 t/s |
| Qwen3.8-27B Q4_K_M, B60: decode | 10.3 t/s | 18.0 t/s |
| Qwen3.6-35B-A3B coder Q4_K_M, A770: prefill 512 | 111.2 t/s (experts on the CPU) | 605.0 t/s |
| Qwen3.6-35B-A3B coder Q4_K_M, A770: decode | 7.7 t/s (experts on the CPU) | 37.7 t/s |

Of the coder's gain, 527 -> 605 t/s prefill and 20.9 -> 37.7 t/s decode is
the placement exemption alone, measured with the kernels unchanged.

Kernels alone (Q4_K, 17408 x 5120, one column; a standalone harness, device
time by events): matvec 84 -> 351 GB/s on the A770 and 163 -> ~400 GB/s on
the B60 (ggml's flat kernel -> this one; the B60 alternates between about
310 and 400 GB/s run to run); 2 to 4 columns cost about one. GEMM at 512
columns 34 TOPS (A770) and 41-43 TOPS (B60), against ~4 TFLOPS for ggml's.

Answers: `test-backend-ops` MUL_MAT and MUL_MAT_ID, every Q4_K/Q5_K/Q6_K case
on both cards (mutants of each path fail 37 to 81 cases). KL against the same
GGUF on the CPU backend, 16 chunks of 512 tokens of English and C++ text,
baseline arm ggml's float kernels: prefill (batched) dense 0.003560 ->
0.003560 nats, same top token 97.84 -> 97.82 %; coder 0.00728 -> 0.00696,
95.93 -> 96.10 %. Decode (one token per ubatch): see `docs/llama-engine.md`.

## 0002-opencl-intel-iq3xxs-iq4xs-iq4nl.patch

Flash-Next's GGUF (UD-Q3_K_XL) keeps its experts in IQ3_XXS (gate, up) and
IQ4_NL / IQ4_XS / Q8_0 (down); ggml-opencl has no IQ3_XXS or IQ4_XS kernel
and no IQ4_NL `MUL_MAT_ID`, so every expert ran on the CPU. 0002 adds the
three types to 0001's machinery on Intel:

- matvec, 1 to 4 columns (`kernels/mul_mv_iq_q8_1.cl`): weights expanded to
  signed int8 per 32-weight sub-block (IQ3_XXS through its grid and sign
  tables, held in local memory; the IQ4 types through `kvalues_iq4nl`) and
  multiplied with integer dot products against 0001's two-term activation.
  IQ3_XXS and IQ4_XS are read as stored (98- and 136-byte blocks); IQ4_NL
  from ggml-opencl's flat planes, with K a multiple of 32 (Flash-Next's down
  projections have K = 640);
- the fp16-DPAS GEMM dequantizes the three types into its local tile, and
  takes a partial last 256-weight super-block (IQ4_NL);
- one type index for the six types in the dispatch; `supports_op` claims a
  K-quant or IQ op only when its kernels were built (it builds them on first
  ask), and declines IQ shapes no kernel takes, so their weights stay on the
  CPU instead of reaching a missing kernel.

Measured here (`measured-here`, A770, standalone harness, device time):
IQ4_XS / IQ4_NL matvec ~200 GB/s on 5120 x 2560, IQ3_XXS 82 GB/s (its
98-byte blocks are read with 16-bit loads); an expert's 640 x 2560 matrix in
13-20 us. `test-backend-ops` MUL_MAT and MUL_MAT_ID pass for all three types
on both cards; a mutant of the IQ4 expansion fails the matvec cases and one
of the IQ3_XXS signs the GEMM cases. Reviewed: no blockers; the should-fix
(a declined dispatch falling into ggml's `default: abort` for the IQ types)
is the `supports_op` rule above.

## 0003-opencl-intel-gated-delta-net.patch

`GGML_OP_GATED_DELTA_NET` (Qwen3.5/3.6/3.8's linear-attention layers) on
Intel, head size 128, scalar gate (`kernels/gated_delta_net_intel.cl`), in the
shape of OpenVINO's `paged_gated_delta_net_opt.cl` (`code`): lanes along the
key dimension, each sub-group holding R state rows in registers, reductions
by sub-group shuffles. ggml's `gated_delta_net.cl` keeps 96 floats a SIMD32
lane (the state spills every token) and reduces through local memory.

- A prompt: a work-group of NSG sub-groups per (head, sequence, NSG x R
  rows); q/k/v/g/beta come in chunks of TC tokens staged in local memory,
  the next chunk loaded into registers while the current one is computed
  (Strata's `gdn_rec_cols_pipe_kernel`, `src/prefill/kernels.cu:380`,
  `code`), so q/k are read once a work-group.
- One token: one sub-group per R rows, the next token's inputs prefetched.
- Shapes (`measured-here`, a harness against a CPU port of ggml-cpu's op,
  512 tokens): A770 (32 heads) R 2 x 64 sub-groups, TC 16, 256 GRF: 451 us
  against 12,042 for `gated_delta_net.cl`; B60 (48 heads) R 4 x 32, TC 8,
  128 GRF: 875 against 5,692; one token 82 -> 12.7 us (A770), 51 -> 5.5 us
  (B60). OpenVINO's own shape (one sub-group a work-group, R 4) ran 1,230 us
  on the A770 at 256 GRF and spilled at 128; a butterfly reduction of the R
  rows measured slower on both cards.
- `test-backend-ops`: six cases at head size 128 added (the 3.5/3.6 head
  ratios, 512 tokens, two sequences, strided inputs, snapshots); 42/42 pass
  on both cards; a 0.1 % change of beta in either kernel fails them.

Switches: `GGML_OPENCL_GDN_INTEL=0` keeps `gated_delta_net.cl`;
`GGML_OPENCL_GDN_INTEL=Rtg,Rpp,NSG,TC` overrides the shape (checked against
the device's local memory and work-group size).

Measured here (`measured-here`, 2026-10-03, llama-bench pp512 / tg128):
coder (A770) 600.3 -> 1,005.6 / 37.4 -> 38.8 t/s; dense 27B (B60) 409.2 ->
493.8 / 17.97 -> 18.20; with the switch off the build reproduces the
baseline (601.3, 409.9). KL against the CPU backend (16 x 512 tokens),
before -> after: dense 0.003560 -> 0.003562 (batched), 0.003561 -> 0.003561
(one token per ubatch), top-1 97.82 -> 97.84 %; coder 0.00696 -> 0.00700,
0.00714 -> 0.00700, top-1 96.10 -> 95.88 % and 96.05 -> 96.03 %.

## 0004-opencl-intel-decode-small-ops.patch

Three launches that a decode step makes per layer, sized for Adreno:
- `kernel_quantize_q8_1` (0001's activation quantization) ran one work-item
  per 32-value block: 64 work-items for a 2048-wide row, 7.4 us on the B60
  and 10 on the A770 (`measured-here`, a trace of the decode step). Now a
  sub-group per block, values read coalesced, the block's maxima and sum by
  sub-group reductions (`kernel_quantize_q8_1_sg`; the sum is added in
  another order). `GGML_OPENCL_KQ_QUANT_SG=0` keeps the old kernel.
- The GLU kernels (`glu.cl`) ran one work-group per row: a 17,408-wide
  decode row in one work-group, 15 us. On Intel each row's elements are now
  split over the grid's second dimension (up to 64 work-groups a row); the
  kernels take the split from the grid, so other devices, dispatched as
  before, run them as before.
- `kernel_mul` for a row that is not a broadcast row ran 64 work-items a row
  (the coder's gated shared expert at decode, 2048 wide: 14.5 us); on Intel
  now up to 256.

## 0005-opencl-intel-f32-few-row-matmul.patch

F32 weights of at most 256 rows on Intel: the gated delta-net's `ssm_alpha`
/ `ssm_beta` where a GGUF keeps them F32 (48 x 5120 on Qwen3.8-27B), and a
MoE router (256 x 2048 on the Qwen3.6 coder). ggml gave a prompt one 64 x 64
output tile per 64 columns over the whole K (8 work-groups for 48 rows and
512 tokens: 92 ms of the dense model's 512-token prefill, `measured-here`)
and a single column a matvec at 58 GB/s. Now up to 8 columns take ggml's
`kernel_gemv_f32_f32_mc` (K split over a work-group; it was opt-in for 2 to
8), and more take the tiled GEMM with K cut into slices run as its batch
dimension, then `kernel_sum_slices_f32`. `GGML_OPENCL_F32_SKINNY=0` keeps
ggml's paths. `test-backend-ops` cases at these shapes added. The coder's
router in a 512-token prefill: 24.3 -> 9.0 ms (A770, `measured-here`).

## 0006-opencl-intel-moe-tokens-as-dpas-a.patch

`MUL_MAT_ID` prefill (16 or more tokens) in the shape of arcint's OpenVINO
patch 0064 (`kernel_mul_mm_id2_*`): a work-group takes a tile of up to XT
(token, slot) pairs of one expert and XSG sub-groups of weight rows, one row
per lane; per 256-weight super-block the pairs' activations are staged once
in local memory in the DPAS a layout, and each lane builds its row's b
operand in registers, so weights never pass through local memory and are
read once a tile. For Q4_K / Q5_K / Q6_K the b operand is the codes
themselves as halves 1024 + q, built with integer operations on whole words
(the activations staged with the middle two of every four halves swapped to
match the byte pairs the words give), one DPAS chain per sub-block (32
weights; 16 for Q6_K), applied in f32 with the sub-block's activation sum:
D sc (chain - 1024 S) - M m S. The IQ types dequantize to f16 per k16 step.
The route (pairs sorted by expert, cut into tiles) is computed once for a
layer's gate, up and down, which share their ids.

Shapes: XT 32 x XSG 16 on SG 8, 32 x 8 on SG 16 (128 rows either way);
`GGML_OPENCL_KQ_MM_ID2_SHAPE=XT,XSG` overrides, `GGML_OPENCL_KQ_MM_ID2=0`
keeps 0001's MUL_MAT_ID GEMM, `GGML_OPENCL_KQ_ROUTE_CACHE=0` routes every
call.

The way there (`measured-here`, the coder's Q4_K gate/up on the A770, 100
calls in a 512-token prefill): 0001's kernel 193 ms; this layout with an f16
dequantization per lane 175 ms (16 x 8) and 143 ms (32 x 8); with the
dequantization replaced by a constant 77 ms. A form with `half2` arithmetic
and the scales hoisted into an array measured 175 ms: IGC's ISA dump shows
`half2` operations not packed in SIMD8 and the array in private memory. The
integer-built codes: 117 ms (32 x 16). Q6_K's codes path (one chain per
16 weights) measured the same as its f16 dequantization (41 ms for the
coder's down projections). The route cache removes 40 launches a prefill.

### Measured and gated together: 0004, 0005, 0006

`measured-here`, 2026-10-03, llama-bench, A770 GT clock pinned at 2000 MHz;
before = 0001-0003, after = 0001-0006:

| | prefill 512 | prefill 2048 | decode |
|---|---|---|---|
| coder, A770 | 1,005.6 -> 1,292.2 t/s | 747 (0004/0005 only) -> 874.7 | 38.8 -> 43.4 |
| dense 27B, B60 | 491.6 -> 521.1 | | 18.21 -> 19.16 |

KL against the CPU backend (16 x 512 tokens), batched / one token per
ubatch: coder 0.00700 / 0.00700 -> 0.00706 / 0.00703 (top-1 95.88 / 96.03
-> 95.86 / 96.10 %); dense (0004-0005; 0006 has no MoE to act on) 0.003562 /
0.003561 -> 0.003559 / 0.003561 (top-1 97.84 / 97.84 -> 97.79 / 97.84 %).
The acceptance task served through arcint on 0001-0006: coder 10/10 at
temperature 0 and 10/10 in 6 of 6 sampled runs (0.7); dense 10/10 at
temperature 0 and 10/10 in 15 of 20 sampled runs (mean 8.6), against 11 of
20 (mean 7.0) for the 0001-0002 build on record.

## 0007-opencl-intel-decode-attention-split.patch

Decode attention (one query row, f16 K/V, head size a multiple of 128) on
Intel. ggml sends Intel to its basic q1 kernel, one sub-group per head
walking every key; its flash-decoding split kernel keeps all DV
accumulators in each lane and is limited to head size 128. Both served
models have head size 256, so their decode slowed with context
(`measured-here`, llama-bench `-d`): coder (A770) 43.4 / 13.6 / 4.2 t/s at
0 / 4,096 / 16,384 tokens, dense 27B (B60) 19.2 / 12.8 / 6.4.

`flash_attn_f32_f16_q1_split_intel` does the split kernel's job with the
lanes along the head dimension: one sub-group of 16 per (head, split of 128
keys), the query (scaled) in registers, K and V rows by sub-group block
reads, one sub-group reduction a key, the online softmax over four keys a
step, and ggml's partial records, merged by `flash_attn_f32_merge`. On
this path the partial buffer is kept across calls. The logit softcap is
applied before the mask, as ggml-cpu does (ggml's own split kernel applies
it after), and scores are clamped at the kernel's -3e38 so that no `-inf`
reaches `exp()` under `-cl-finite-math-only`.

Measured (`measured-here`; test-backend-ops perf, head size 256, 24 query
heads on 2 KV heads): 4,096 keys 4,159 -> 96 us (A770), 1,560 -> 57 us
(B60); 16,384 keys 21,315 -> 396 us, 6,514 -> 270 us. Gathers of single
halves instead of block reads ran 424 us at 4,096 keys on the A770. A
form that served a KV head's query heads from one work-group with 16-key
K/V tiles staged in local memory measured slower (A770 395 us, B60 79 us)
and is not in the patch. llama-bench decode at 0 / 4,096 / 16,384 tokens:
coder 47.8 / 47.3 / 43.5 t/s, dense 19.75 / 19.39 / 17.95.
`test-backend-ops -o FLASH_ATTN_EXT`: 2,642 of 2,644 pass on the A770; the
two failures (four and 75 query rows) fail the same without the patch, and
the patch fixes a third that did (one query row, head size 128, 4,096 keys).

Switches: `GGML_OPENCL_FA_INTEL_SPLIT=0` keeps the basic q1 kernel;
`GGML_OPENCL_FA_INTEL_KV_PER_SPLIT` sets the keys per split (128);
`GGML_OPENCL_FA_INTEL_OPTS` appends build options to Intel's FA programs
(`-D FDI_KB=`, `-cl-intel-256-GRF-per-thread`, for tuning).

Gate (`measured-here`, 0001-0006 -> 0001-0007): KL against the CPU backend
with one token per ubatch (the decode path), coder 0.007034 -> 0.006986
(top-1 96.10 -> 96.18 %), dense 0.003561 -> 0.003560 (97.84 -> 97.84 %);
the acceptance task served through arcint: coder 10/10 at temperature 0 and
10/10 in 6 of 6 sampled runs; dense 10/10 at temperature 0, and 10/10 in 10 of 20
sampled runs (mean 7.0; 0001-0006 gave 15 of 20, mean 8.6, and 0001-0002 11
of 20, mean 7.0; the largest difference, 15 against 10, has a two-sided
Fisher p of 0.19, and the means differ by 1.6 points at a standard error of
1.1 -- the arms do not separate, and the KL is the same to four places).

### Review of 0003-0007

A review (2026-10-03) of the five patches found no blocker. Its should-fix
items are in the patches: the decode-attention softcap order and clamp, the
partial buffer and the GLU / `kernel_mul` dispatch changes restricted to
Intel, the spill log for the new FA kernel, `n_expert` / `n_used` / `n_tok`
in the route-cache key, the gated delta-net kernels dropped (to ggml's) when
the built kernel's work-group limit is below the shape, and two checks
(the MoE shape override against local memory, 16-byte offsets for the F32
float4 kernels). After them (`measured-here`, A770): `test-backend-ops`
GATED_DELTA_NET 42/42, MUL_MAT_ID 338/338, MUL_MAT 1,045/1,045,
FLASH_ATTN_EXT 2,642/2,644 (the two that fail without the patches),
SWIGLU 24/24, GEGLU 24/24, MUL 93/93; the coder 1,285.6 t/s prefill,
47.8 / 47.4 t/s decode at 0 / 4,096 tokens.
