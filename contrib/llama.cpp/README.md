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

Decode attention (up to 8 query rows -- a decode step, a speculative
verify -- f16 K/V, head size a multiple of 128) on Intel. ggml sends Intel to its basic q1 kernel, one sub-group per head
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

Speculative verification: a 2-row MTP verify went to ggml's prefill
kernel at 3.5 ms a call, 61 % of the device time of a speculative run on
the coder (A770, `measured-here`); with up to 8 rows taking the split
kernel, llama.cpp's `draft-mtp` loop (temperature 0, the acceptance prompt,
400 tokens) gives the coder 56.9 / 68.1 / 67.6 t/s with 1 / 2 / 3 drafts
(97.5 / 92.6 / 83.2 % accepted; 24.1 / 32.0 / 35.4 before) and the dense
model 25.2 / 31.5 / 31.9 (91.0 / 84.0 / 76.3 %; 15.1 / 19.7 / 21.2 before),
against 47.8 and 19.75 plain; the greedy text is the same for 1, 2 and 3
drafts on both. Four drafts (a 5-row verify) fall to 46.0 and 10.7: the
K-quant matvec takes at most 4 columns. The test cases with 2-8 rows pass
(FLASH_ATTN_EXT 2,642/2,644 as before).

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

## 0008-opencl-intel-prompt-attention-dpas.patch

Prompt attention (more than 8 query rows, f32 Q, f16 K/V, head size 256, an
f16 mask, no sinks, ALiBi or softcap) on the XMX units
(`kernels/flash_attn_dpas.cl`). ggml's `flash_attn_f32_f16.cl` computes it
in f32 scalar code: 344 GFLOPS on the B60, 309 on the A770 (`measured-here`,
test-backend-ops, 512 rows x 4,096 keys, 24 heads on 4), and at 4,096
tokens of context it was 66 % of the coder's prefill.

- Sub-group 16 (Xe2, B60): a work-group of 16 sub-groups takes 128 query
  rows; per tile of 32 keys, staged in local memory for all of them (V
  transposed), S = Q K^T by DPAS with Q in local memory in the a layout, the
  mask added, the online softmax in f32 (row max and sum by sub-group
  reductions), and O += P V by DPAS -- S's result layout (lane = key,
  component = row) is P's a layout at sub-group 16. 256 GRF.
- Sub-group 8 (Xe-HPG, A770): a pair of sub-groups shares 8 query rows,
  each owning half of the 256 output columns (16 float8 in registers) and
  computing S itself; P is re-laid from S by shuffles (two keys a lane).
  The shuffles run in every lane: under a branch they read lanes the branch
  left inactive (NaN, `measured-here`).
- The work-group stops after the last key any of its rows sees, each row's
  masked tail scanned backwards (llama.cpp's causal mask). Taking the last
  row's alone failed test-backend-ops' masks with -inf blocks.
- The row sums add the f16-rounded probabilities that P V multiplies, each
  lane its share, reduced across the sub-group once at the end.
- V is staged transposed with the key fastest, so neighbouring work-items
  store neighbouring halves (key-major they all hit one bank).
- Known cost: the masked-tail scan reads one half a step per row. For
  llama.cpp's causal mask of one sequence the tail is at most the ubatch;
  several sequences in one batch, or a sliding window, lengthen it.

Review (a Fable pass over the first version), applied: -inf recognised by
its f16 bit pattern rather than a threshold; no correction factor before a
row has seen a key; the row sums deferred; the key-fastest V staging; the
loader checks the kernel's local memory against the device's next to its
work-group size; the dispatch also requires unit strides along dimension 0
of Q, K and V, and a mask at least n_kv x n_q. Together, against the
version before them at test-backend-ops (`measured-here`, the case below):
B60 8.66 -> 8.22 ms, A770 24.1 -> 22.4 ms. With V staged key-major
again (`GGML_OPENCL_FA_DPAS_OPTS`, a since removed define): B60 8.60, A770
24.1-24.3 ms.

Measured (`measured-here`, test-backend-ops, 512 rows x 4,096 keys, 24
heads on 4, f16 mask, the patched tree): B60 149.9 -> 8.2 ms (6.3 TFLOPS),
A770 167.0 -> 22.6 ms (2.3 TFLOPS), `flash_attn_f32_f16.cl` -> this
kernel. (An earlier 6.7 ms on the B60 came from a standalone harness with
no mask buffer and is not the figure to compare.) llama-bench prefill,
before -> after, on the version before the review (the reviewed one is
faster at the kernel): dense 27B (B60) 525 -> 633 t/s at 512 tokens, 145
-> 540 at 512 tokens after 4,096, 590 at 4,096 tokens, 363 at 512 after
16,384; coder (A770) 1,285 -> 1,608, 350 -> 1,032, 602 -> 1,322 at 4,096
tokens (the OpenVINO path's coder: 1,379 at 4k), 262 -> 889 at 4,096 after
4,096. `test-backend-ops -o FLASH_ATTN_EXT`: 2,643/2,644 on the B60 and
2,642/2,644 on the A770, the failures those of the unpatched tree (a
softcap case this kernel does not take; a head-64 case on the A770). KL,
batched (the prompt path), against the same reference: coder 0.007059 ->
0.007043, dense 0.003559 -> 0.003559, top-1 within 0.1 point. The coder
10/10 on the acceptance task and the dense model 10/10 greedy, 13/20
sampled (the version before the review).

Switches: `GGML_OPENCL_FA_DPAS=0` keeps `flash_attn_f32_f16.cl`;
`GGML_OPENCL_FA_DPAS_OPTS` adds build options to the kernel.

## 0009-opencl-intel-few-column-kquant-xmx.patch

A K-quant weight times 2 to 16 f32 columns -- a speculative verify of 1 to
15 drafts, a few lanes -- on the XMX units, each weight read once for all
of them (`kernels/mul_mm_kq_few.cl`). Before it, 0001's q8_1 matvec took up
to 4 columns and ggml's GEMM tile the rest. Neither held up at this shape
(`measured-here`, test-backend-ops, Q4_K 4,096 x 14,336, B60). The matvec
cost 76 / 123 / 169 us at 1 / 2 / 4 columns: every column pays two integer
dot products per weight, and the kernel is compute-bound from 2 columns,
not "at the cost of one", as 0001's section and loader comment have it
(A770: 95 / 92 / 183 us). From 5 columns the GEMM took 837 us. On the dense
27B a step of 4 tokens cost 1.6x a step of 1, a step of 8 cost 6x.

- A lane is a weight row. Its code bytes (two 64-byte lines a super-block)
  become halves 1024 + q by integer operations, as 0006 builds them: the
  fp16 DPAS b operand. The tokens are the a operand, 8 a DPAS, read from a
  tiled f16 copy of the columns, so one a load serves the row's 16 k.
- `kernel_mul_mm_few_prep` writes that copy once per input (matmuls that
  share it, q/k/v and gate/up, reuse it within the graph, like the other
  K-quant paths): the tile, each 16 values' sum S, and each 32 values'
  power-of-two scale (1 unless a block reaches 2^15).
- A sub-block's DPAS chain starts at -1024 S (-1056 S for Q6_K's offset
  codes), so it ends at sum q x. The block scale and D sc - M m S apply
  once per sub-block.
- S must be the sum of exactly the halves the DPAS multiplies, since the
  chain's 1024 offset turns any difference into a thousandfold one. Under
  ggml's `-cl-unsafe-math-optimizations -cl-finite-math-only` the compiler
  summed the unrounded floats for `(float) h`, which put results 4-9 % off
  in a harness at real shapes (`measured-here`). The prep therefore decodes
  each half from its bits with integer operations. Subnormal halves go to
  zero first, so S and the DPAS agree whether or not the XMX units flush
  them.
- NK sub-groups of a work-group take every NK-th super-block and add their
  sums in local memory: 4 on Xe2, 8 on Xe-HPG. One build takes up to 8
  columns, another up to 16.
- The path starts at 2 columns on Xe2 and at 4 on Xe-HPG. There the q8_1
  matvec stays ahead at 2 and 3 columns: the coder's step at 3 tokens ran
  104.3 t/s on it against 96.3 through this path, at 4 tokens 123.9
  against 124.4 (llama-bench, A770, `measured-here`).

Measured (`measured-here`):

- test-backend-ops, 4,096 x 14,336, B60:
  - Q4_K: 111 us at 2, 4 and 8 columns, 158 at 16.
  - Q5_K: 156 at 8.
  - Q6_K: 191 at 8, against 902 for the GEMM.
- The same shape on the A770 (q8_1 matvec for comparison): Q4_K 126 us at
  8 columns.
- llama-bench, dense 27B (B60), the step at n tokens over the 1-token step
  (50.2 ms):

  | tokens | before | 0009 |
  |---|---|---|
  | 2 | 1.32x | 1.15x |
  | 4 | 1.63x | 1.17x |
  | 8 | 6.0x | 1.21x |
  | 16 | 6.2x | 1.51x |

- `test-backend-ops -o MUL_MAT`: 1,096/1,096 on both cards. That count
  includes cases this patch adds (2-16 columns, rows not a multiple of the
  sub-group, 5 super-blocks, src1 broadcast over dimension 2), and 51 of
  them failed before the sum fix.
- KL, every matmul through this path (ubatch 4 and 8, against the same
  reference, the baseline arm the tree without 0009):
  - dense: 0.003560 -> 0.003558 at ubatch 4, 0.003561 -> 0.003558 at
    ubatch 8;
  - coder (A770, ubatch 4): 0.006986 -> 0.007035, top-1 96.18 -> 96.01 %.

Served with `--llama-mtp` (acceptance task at temperature 0, 10/10 in every
configuration):

| model | drafts | before | 0009 |
|---|---|---|---|
| dense, B60 | 3 | 31-35 t/s | 43.5 t/s |
| dense, B60 | 4 | | 46.2 t/s |
| dense, B60 | 5 | | 48.0 t/s |
| dense, B60 | 6 | | 48.0 t/s |
| coder, A770 | 2 | 67-69 t/s | 65.2 t/s |
| coder, A770 | 3 | | 69.2 t/s |
| coder, A770 | 4 | | 71.7 t/s |

Strata and NInfer verify with the same kind of kernel
(`docs/campaigns/mtp-cycle-wall.md`).

Switches: `GGML_OPENCL_KQ_FEW=0` keeps the q8_1 matvec and the GEMM;
`GGML_OPENCL_KQ_FEW_MIN` sets the least column count.

## 0010-mtp-masked-nextn-rows-of-the-outputs.patch

The Qwen3.5/3.6 MTP graphs (`src/models/qwen35.cpp`, `qwen35moe.cpp`)
captured the nextn row (`t_h_nextn`) before selecting the output rows.
A context with masked nextn embeddings then copies the first `n_outputs`
rows of all tokens (`llama-context.cpp`, the masked extraction), so a batch
of several tokens with one output returned token 0's row in place of the
output token's. The trunk graph selects first (`qwen35.cpp:178-213`). The
patch does the same in both MTP graphs, and only for masked contexts.

llama.cpp's own draft-mtp never hits it: its catch-up batch has no outputs
and its drafts are single tokens. arcint's drafter does, once draft 0 comes
from the catch-up batch, as Strata's does
(`docs/campaigns/mtp-cycle-wall.md`). Without the patch, drafts after the
first chained from the wrong row: served draft acceptance fell from 78 to
57 % on the dense 27B at 3 drafts, and from 91 to 75 % on the coder at 2
(`measured-here`).

## 0011-opencl-intel-kquant-gemm-2d-block-loads.patch

The K-quant GEMM on Xe2 with both operands read by 2D block loads
(`kernels/mul_mm_kq_2d.cl`, `cl_intel_subgroup_2d_block_io`, B60 only). It
takes the shape of arcint's OpenVINO kernel (marfrit-openvino patch 0029),
moved onto ggml-opencl's flat planes.

- **Weights.** A lane is a weight row. One transposed 32-bit block read
  brings 32 bytes of each of the sub-group's sixteen rows. They are decoded
  in registers to the fp16 DPAS b operand.
- **Activations.** The tokens are the a operand, 32 rows x 16 halves a
  read, straight from the f16 copy of the activations. A work-group's
  sub-groups share them through L1: no local memory, no barrier.
- **Reuse.** The weights are decoded once per 64 tokens. A work-group has
  16 sub-groups, so 256 weight rows.

The kernel serves where every 2D surface is at least 64 bytes a row: Q5_K
from K = 512, Q6_K from K = 1,024. It is built only where the device's base
alignment is a multiple of 64 bytes, since the planes are sub-buffers at that
alignment. It serves only within the extension's 2^24 limits.

Why (`measured-here`, B60, docs/campaigns/llama-engine-kernel-gap.md lever
7): `mul_mm_kq_f16.cl` was 73 % of a 4,096-token dense prefill. unitrace
showed its threads waiting on loads 52 % of the time, with every thread
slot full.

Four alternatives measured slower or equal before this one:
- the activations tiled for block reads;
- taller tiles;
- the activation loads pipelined in registers;
- k-tiles of 64 through local memory, double-buffered.

Measured (`measured-here`, B60):

- test-backend-ops, 4,096 x 512 x 14,336, the activation conversion
  included:

  | type | before | after |
  |---|---|---|
  | Q4_K | 1,722 us | 1,262 us (47.6 TFLOPS) |
  | Q5_K | 1,726 us | 1,511 us |
  | Q6_K | 1,748 us | 1,557 us |

- `test-backend-ops -o MUL_MAT`: 1,123/1,123 on both cards, with cases
  this patch adds (Q5_K at K = 512, a src1 broadcast over dimension 2).
  Three injected decode faults (Q4_K's min sign, Q6_K's
  scale halves, Q5_K's fifth bit) each failed 8-19 of them.
- llama-bench, dense 27B: prefill 632 -> 758 t/s at 512 tokens, 594 -> 702
  at 4,096.
- KL batched: 0.003559 -> 0.003559, top-1 97.84 %.

Switches: `GGML_OPENCL_KQ_2D=0` keeps `mul_mm_kq_f16.cl`;
`GGML_OPENCL_KQ_2D_SHAPE = "TM,WG,AT"` sets the token tile, the sub-groups
and the activation read height (8 or 32).

## 0012-opencl-intel-kquant-gemm-2d-order.patch

Two changes to 0011's kernel. Both came from reading oneDNN's JIT GEMM,
which the OpenVINO path runs on the same B60, beside ours. The oneDNN
kernels were dumped with `ONEDNN_JIT_DUMP`, disassembled with IGA and
compared line by line with IGC's dump of 0011.

- **The b operand outer.** A 32-value sub-block's first b operand is
  multiplied with every token group, then its second. A group's two
  dependent DPAS are 8 apart instead of adjacent. Adjacent ones made IGC
  drain the matrix unit between them. In the ISA, single and paired DPAS
  chains became chains of 4 and 8, and the DPAS-completion waits a
  super-block fell from 53 to 45.
- **Token tiles the fast grid dimension.** A 256-row weight panel is read
  by its token tiles in turn while it is in L2, not once per tile from
  memory.

Measured (`measured-here`, B60):

| test | 0011 | 0012 |
|---|---|---|
| test-backend-ops 4,096 x 512 x 14,336, Q4_K | 1,264 us | 1,229 us |
| the same, Q5_K | 1,509 us | 1,398 us |
| the same, Q6_K | 1,557 us | 1,484 us |
| llama-bench dense 27B prefill, 512 tokens | 758 t/s | 891 t/s |
| the same, 4,096 tokens | 703 t/s | 816 t/s |

- KL batched: 0.003559 -> 0.003559.
- MUL_MAT 1,123/1,123 on both cards.

Tried in the same pass and left out (`measured-here`):
- the next super-block's weights read a super-block ahead into a second
  register set: Q4_K 1,229 -> 1,770 us;
- 96-token tiles;
- L1 prefetch of the next activation tile;
- two-column activation reads.

## 0013-opencl-intel-kquant-gemm-packed-decode-int8.patch

Four changes to the B60's 2D-block GEMM (`kernels/mul_mm_kq_2d.cl`), each
measured on its own. The diagnosis behind them is in
`docs/campaigns/llama-engine-kernel-gap.md` (lever 7).

- **Packed fp16 decode (Q4_K, Q5_K, Q6_K).** The weights were decoded one
  value per 32-bit lane operation: shift, mask, convert, multiply-add,
  convert to half. Five instructions a weight. Now one `bfn` masks two codes
  into a dword as halves 1024 + q, and two 32-wide half instructions
  subtract 1024 (exact) and scale. Those two are inline vISA (`hscale`):
  IGC keeps a per-lane `half2` as two planar halves and moved them apart
  and back around every operation.
  - The pairs are codes (k, k + 2). So the f16 activations are written
    with k permuted inside every group of four (`kernel_cvt_f32_f16_p4`,
    P4).
  - Q4_K's main loop: 1,640 -> 970 instructions for 128 DPAS. Q5_K:
    2,145 -> 1,217. Q6_K: 2,386 -> 1,195.
- **The activation conversion, eight values a work-item.** 16-byte stores
  in place of 2-byte ones: 182 -> 139 us at 512 x 14,336.
- **A work-group barrier every 16 blocks (`KSYNC`).** The sub-groups share
  the activation tile through L1. Without a barrier they drift apart over
  a long k loop. At K = 17,408 the load-store cache hit 86.5 % of reads
  against 92 % at 5,120 (unitrace ComputeBasic), and the vector engines'
  shared-function hold was 32.5 % against 15 %. Down projection, 512
  tokens: Q4_K 1,496 -> 1,342 us, Q5_K 1,660 -> 1,405, Q6_K 1,960 ->
  1,431. The K = 5,120 shapes are within 1.4 %.
- **Q4_K on int8 DPAS (`kernel_mul_mm_kq2d_q4_K_i8`).** One i8 x i8 DPAS a
  32-value sub-block, at twice fp16's multiply rate.
  - The activations go in as int8 per (token, 256) (`kernel_quant_i8`),
    with their integer sums per 32.
  - The codes go in as s8 (code - 8). A sub-block's product is then at
    most 32 x 127 x 8 = 32,512 in magnitude, a 16-bit value.
  - Its 6-bit scale is applied by one integer `mad` on word sources
    (inline vISA `mad16`) into an int32 accumulator that spans the
    super-block. The constant part, 8 D sc - DM m, is one fp16 DPAS a
    super-block against the sums.
  - The activation tile and the weights are read a sub-block ahead.
  - On by default; `GGML_OPENCL_KQ_2D_I8=0` keeps the fp16 kernel.
  - **A deviation from the references, with its price.** llama.cpp's MMQ
    quantizes activations per 32 (q8_1), and oneDNN, on the OpenVINO path,
    per 64. This kernel does it per 256 so that the integer accumulation
    spans a super-block. The per-32 forms rescale in float after every
    sub-block, and they measured slower here (1,599-2,140 us against
    1,229 for fp16, the first attempts in lever 7). Price on the dense
    27B: see the KL and sampled rows below. Decode is unaffected; its
    matvec keeps q8_1 per 32.

The inline vISA shares a program with the fp16 kernels. A driver that
rejects it loses the whole 2D path, with a warning, and prompts fall back
to `mul_mm_kq_f16.cl`.

The 2D path and the older kernels now keep the f16 activations in
different orders, in the same buffer. Where a MoE layer's routed experts
(natural order) and its shared expert (P4) read one input on the B60, the
input is converted twice. None of the served models do that on this
path.

Measured (`measured-here`, B60, `GGML_OPENCL_PLATFORM=0`):

Device time (unitrace), the dense model's shapes, 512 tokens, KSYNC 16 in
both arms; fp16 plus its conversion against int8 plus its quantization:

| m x k (dense projection) | fp16 | int8 |
|---|---|---|
| 17,408 x 5,120 (gate, up) | 1,191 + 44 us | 1,080 + 43 us |
| 5,120 x 17,408 (down) | 1,344 + 184 us | 1,139 + 132 us |
| 12,288 x 5,120 (q) | 902 + 50 us | 775 + 44 us |
| 5,120 x 6,144 (out) | 440 + 57 us | 390 + 46 us |

llama-bench, dense 27B Q4_K_M, `-fa 1`, two repeats, the arms interleaved:

| build | prefill 512 | prefill 4,096 |
|---|---|---|
| 0012 | 892 t/s | 816 t/s |
| + packed decode, Q4_K | 919 | 839 |
| + Q5_K, Q6_K | 935 | 852 |
| + the conversion | 941 | 857 |
| + KSYNC 16 | 968 | 880 |
| + int8 Q4_K (0013) | 1,029 | 930 |
| 0013 as committed (reviewed tree) | 1,030 | 931 |
| the same, `GGML_OPENCL_KQ_2D_I8=0` | 973 | 884 |

- KL batched, against the same GGUF on the CPU: 0012 0.003559 (top-1
  97.843 %); packed decode 0.003562 (97.843 %); 0013 with int8 0.004034
  (97.721 %), with `GGML_OPENCL_KQ_2D_I8=0` 0.003562 (97.843 %). The int8 activations alone (fp16 kernel fed int8-per-256
  values, an emulation): 0.004048, 97.50 %.
- The acceptance task, served (`--engine llama`, MTP 5, the 40,960-id
  draft head): 10/10 at temperature 0 on the dense model (B60) and the
  coder (A770, MTP 4). Twenty sampled runs (temperature 0.7) per arm on
  the dense model, the runs at 10/10 and the mean: 0013 14 of 20 (8.7);
  0013 with int8 off 12 of 20 (7.7); 0012 15 of 20 (8.15). The arms do
  not separate. The record's earlier plain arms were 10, 11 and 15 of 20.
- The coder's prefill on the A770 (llama-bench, 512 tokens) is unchanged
  at 1,610 t/s: none of this patch runs there.
- `test-backend-ops -o MUL_MAT` 1,123/1,123 on both cards, int8 on and
  off. Two injected faults failed it: the conversion without P4 (30
  cases), the int8 kernel without its constant term (9).
- MUL_MAT_ID on the B60: the same 74 MXFP4 failures as 0012. The pristine
  pin `bed0a85` fails the same 74 on the B60 and passes them on the A770.
  They are upstream's, not this series'.

Measured and left out (`measured-here`, B60):
- two 16-row weight tiles a sub-group, so each activation read feeds two
  DPAS. Slower at every shape, fp16 and int8. Dense prefill 742-780 t/s
  against 839 (fp16).
- weights and scales read a step ahead in the fp16 kernel: Q4_K 1,158 ->
  1,204 us. The activation tile read a half-sub-block ahead: 1,066 ->
  1,273 us.
- cooperative L1 prefetch of the activation tile, one or two blocks ahead:
  no gain in either kernel.
- 8-row activation reads (fp16): 1,066 -> 1,439 us.
- a banded raster (bands of 4 token tiles): down -4 %, gate/up +6 %.
- Q4_K int8 with the 6-bit scale split as 8 s_hi + s_lo: two i8 x u8 DPAS
  a sub-block and no per-sub-block rescale. Correct, but slower than the
  fp16 kernel at K = 5,120 (1,251 against 1,180 us).
- the int8 kernel at 16 tokens a sub-group in 128-GRF mode (8 threads an
  XVE instead of 4): spills, 37 instructions a DPAS.
- the conversion with each column read once and held in registers: 139 ->
  136 us, not kept.

## 0014-opencl-intel-searched-tiles.patch

Defaults found by a genetic search, and the hooks to repeat it
(`docs/campaigns/kernel-autotune-ga.md`, `tools/kq_tune/`).

- **A770 (SG 8):**
  - the tokens-as-a MUL_MAT_ID kernel (ID2) off by default;
  - the tile kernel's tile 2,4,1,1 -> 2,4,2,2;
  - the plain GEMM's tile 4,4,1,16 -> 4,4,2,8 (as on the B60).

  ID2 had replaced the tile kernel at the tile kernel's old tile. With its
  tile searched, the tile kernel is faster, and faster than ID2's best shape
  too (1,702 t/s at 24 x 32). `GGML_OPENCL_KQ_MM_ID2=1` turns ID2 back on;
  the B60 keeps it on.
- **B60:** the int8 Q4_K kernel's work-group barrier every 32 blocks
  (`KSYNC_I8`), the fp16 kernels' every 16.
- **Hooks, inert when unset:**
  - `GGML_OPENCL_KQ_2D_T<i>` = "TM,WG,AT,KSYNC" builds the 2D GEMM of type
    i (0 Q4_K with its int8 kernel, 1 Q5_K, 2 Q6_K) from a program of its
    own. A shape that does not build or fit is ignored, with a warning;
  - `GGML_OPENCL_KQ_2D_OPTS` passes defines to the 2D program;
  - test-backend-ops perf gains a dense 27B's prompt GEMM shapes (Q4_K,
    Q5_K, Q6_K at 512 tokens).

Measured (`measured-here`, llama-bench, `-fa 1`, two interleaved repeats
each):

| model, card | prefill 512: 0013 -> 0014 | prefill 4,096: 0013 -> 0014 |
|---|---|---|
| coder, A770 | 1,613 -> 1,739-1,742 t/s (+7.9 %) | 1,345-1,346 -> 1,431-1,432 (+6.4 %) |
| dense 27B, B60 | 1,028 -> 1,033-1,034 (+0.5 %) | 930 -> 934.5-934.8 (+0.5 %) |

- The coder at 2,048 tokens with ubatch 1,024, a GEMM shape the search
  never saw (on the search harness, the tiles set through the environment):
  1,659 -> 1,882 t/s (+13.4 %).
- The acceptance task, served: the coder 10/10 at temperature 0 and 3 of 3
  sampled; the dense model 10/10 at temperature 0.
- `GGML_OPENCL_KQ_2D_OPTS` also reaches the per-type programs. It must not
  set `TMM`: the host dispatches the int8 kernel for 32 tokens a sub-group.
- The coder's KL: 0.007043 -> 0.007040, top-1 95.88 -> 96.08 %. The B60
  change is a barrier interval: no arithmetic changes (`code`: the 2D
  kernels use no local memory).
- MUL_MAT 1,123/1,123 on both cards; MUL_MAT_ID 338/338 on the A770. On the
  B60 MUL_MAT_ID shows the pin's own 74 MXFP4 failures (0013).

## 0015-opencl-intel-quantized-kv-attention.patch

Attention over a quantized KV cache on the Intel kernels: K and V as q8_0
or q4_0, the pairs 8:8, 8:4 and 4:4. arcint offers 8:8 and 8:4 with
`--llama-kv` (`q8_0`, `q8_0:q4_0`): 4:4 misses the answer-level bar on the
dense 27B (below). q8_0 takes the cache to 53 % of f16, 8:4 to 41 %.

Upstream (`code`):
- a symmetric q8_0 or q4_0 cache takes the pin's q8_0 / q4_0 attention
  kernels, and on an Intel card a decode row takes the basic per-row one;
- an asymmetric pair (8:4) has no kernel: every call dequantizes all of K
  and V to f32 on the GPU (`ggml_cl_flash_attn_dequant_kv_gpu`), then runs
  the f32 kernels.

On the dense 27B (B60), q8_0 with the stock kernels (`measured-here`):
- prefill 201 t/s at 4,096 tokens and 27 t/s at 16k depth;
- decode 4.65 t/s at 16k.

- **Prompt (`flash_attn_dpas.cl`):** the same kernel, built three more
  times with `-DKT -DVT` (0 f16, 1 q8_0, 2 q4_0). Only the staging load
  changes: the tiles go to local memory as f16 either way, so the DPAS
  part is the f16 kernel's. Both sub-group sizes (B60 16, A770 8).
- **Decode and verify rows, up to 8 (`flash_attn_f32_f16.cl`):** the three
  quantized kernels are 0007's K-split decode with a loader in place of the
  f16 row read. The loader reads lane l's values of a row from the blocks:
  one byte a lane for q8_0, a nibble for q4_0, the block scale broadcast.
  The f16 kernel keeps its own body: built from the shared one, it decoded
  4 % slower on the A770 (below). It is built for sub-group 16 and a head size that is a
  multiple of 32 (both cards serve it at 16).
- **Host:** both routes come after the SoA-to-AoS reconstruction, so a
  quantized tensor uploaded as SoA (test-backend-ops) is read as blocks
  too. The f16 prompt route moved with them. The dequantization block
  (GPU, or host for a strided view) is skipped when the decode kernel
  takes the call; the review checked that the kernel is then always the one
  dispatched (`code`).
- **Switches:**
  - `GGML_OPENCL_FA_DPAS_Q=0` (read at load) leaves a quantized prompt to
    the upstream kernels;
  - `GGML_OPENCL_FA_INTEL_SPLIT=0` leaves decode to them, the f16 split
    decode (0007) included.
- **Tests:** test-backend-ops gains 90 FLASH_ATTN_EXT cases at the hybrid
  Qwens' geometry:
  - head size 256, 24 heads on 4 and 16 on 2;
  - batch 1, 3, 8, 64 and 512;
  - KV 512, 1,013 (a partial key tile and split) and 4,096;
  - for each pair.

  test-backend-ops uploads K/V with `set_tensor`, which stores a q8_0 /
  q4_0 tensor as SoA, so every case goes through the reconstruction.
  Served, the cache is filled by SET_ROWS and stays AoS. That path is
  covered by the KL runs and the acceptance task below, not by these
  cases.
- **Test results:**
  - FLASH_ATTN_EXT: B60 2,733 of 2,734, A770 2,732 of 2,734. All three
    failures are the pin's own f16 K/V cases (head size 256 with logit
    softcap; on the A770 also one at head size 64), and they fail before
    0015 too.
  - **Red:** two faults injected, a q8_0 decode lane reading the block's
    upper half and q4_0 prompt staging with offset 7. Then 2,653 of 2,704
    pass on the B60 (before the tail cases were added). Of the 51
    failures, 50 come from the faults: the q8 cases at batch 1, 3 and 8,
    and the q4 cases at batch 32, 64 and 512. The 51st is the softcap
    case.
- **f16 decode, 0014 against 0015**, llama-bench tg128, same card,
  interleaved:
  - with the f16 kernel built from the shared body: dense B60 19.70 ->
    19.71 t/s at depth 0 and 17.94 -> 17.94 at 16k, but coder A770
    47.83-47.84 -> 45.81-46.03 at depth 0 and 43.49-43.62 -> 42.90-43.07
    at 16k;
  - with its own body back (as shipped): coder A770 47.84-47.85 ->
    47.85-47.86 at depth 0, 43.66-43.68 -> 43.64-43.65 at 16k.

Measured (`measured-here`, llama-bench `-fa 1 -r 2`; t/s):

| model, card, KV | prefill 4,096 | decode | prefill at 16k | decode at 16k |
|---|---|---|---|---|
| dense 27B, B60, f16 | 934 | 19.7 | 474 | 17.95 |
| dense 27B, B60, 8:8 | 896 | 18.9 | 435 | 17.1 |
| dense 27B, B60, 8:4 | 893 | 18.9 | 432 | 17.1 |
| coder, A770, f16 | 1,435 | 47.85 | 493 | 43.65 |
| coder, A770, 8:4 | 1,361 | 42.0 | 446 | 38.0 |

(llama-bench decode, no MTP: the served decode with MTP is higher.)

- **KL against the CPU reference** (perplexity window, f16 KV as the
  baseline arm):

  | model | KV | KL | top-1 |
  |---|---|---|---|
  | dense 27B | f16 | 0.004034 | 97.72 % |
  | dense 27B | 8:8 | 0.003966 | 97.65 % |
  | dense 27B | 8:4 | 0.005756 | 97.33 % |
  | dense 27B | 4:4 | 0.007800 | 96.52 % |
  | coder | f16 | 0.007040 | 96.08 % |
  | coder | 8:8 | 0.006996 | 95.98 % |
  | coder | 8:4 | 0.009245 | 95.54 % |
  | coder | 4:4 | 0.011329 | 95.03 % |

  8:8 and 8:4 are within the answer-level bar. 4:4 is not: top-1 drops by
  1.2 points on the dense model and 1.05 on the coder.
- **Context:** the dense 27B served on the B60 with MTP (5 drafts, the
  40,960-id draft head) at 131,072 tokens of context and an 8:8 cache:
  - peak VRAM 23.06 GB;
  - a 128,133-token prompt prefilled at 166 t/s, then decoded at 7.5 t/s
    at that depth;
  - the acceptance task 10/10 at temperature 0 (decode 47.1 t/s).

  With an f16 cache and MTP, 131,072 overcommits the card: the copy engine
  was reset.

## 0016-opencl-intel-gqa-decode-attention.patch

Decode and MTP-verify attention on Xe2 (the B60) in one pass over K and V
per KV head (`docs/campaigns/gqa-small-t-decode.md`), after NInfer's small-T
kernel (`src/ops/attention/causal_softmax/small_t_bf16.cuh`, `code`).

0007/0015's split kernel gives every (query head, row, split) its own
sub-group. Every verify row re-reads the whole KV, so 6 rows cost 5.4x one
row at 131k keys.

`flash_attn_gqa_dpas.cl` takes a KV head's query heads and rows together:
- One work-group per (KV head, split). Row m of the tile is (head m mod G,
  token m div G); 8 rows a DPAS M, two sub-groups each, one per half of the
  output columns.
- QKᵀ and PV on the XMX units.
- V staged in local memory as f16 per 32 keys, transposed. K read straight
  from the cache when it is f16, staged too when it is quantized.
- The partial records of `flash_attn_f32_merge`, which is reused.
- f16, q8_0 / q8_0, q8_0 / q4_0, q4_0 / q4_0. Head size 256, no ALiBi or
  softcap.
- Built only where the device has sub-group 16 and not 8 (Xe2). The A770
  keeps 0015's kernels.
- Routed from 4 rows (`GGML_OPENCL_FA_GQA_MIN_ROWS`); 1-3 rows stay on the
  split kernel. `GGML_OPENCL_FA_GQA=0` turns it off.
- `GQA_K_STAGED` / `GQA_K_DIRECT` and `BK` are compile-time switches
  (`GGML_OPENCL_FA_GQA_OPTS`). The sides (1, 2 or 4) come from
  `GGML_OPENCL_FA_GQA_SIDES`, used for the build and the dispatch alike.
  These are the genes of a structural search
  (`docs/campaigns/kernel-autotune-ga.md`).

Measured (`measured-here`, B60, dense 27B geometry: 24 query heads on 4,
head size 256):

- **One attention layer, f16, test-backend-ops perf:**

  | keys | rows | 0015 | 0016 |
  |---|---|---|---|
  | 8k | 6 | 0.67 ms | 0.48 ms |
  | 32k | 6 | 2.75 ms | 1.79 ms |
  | 131k | 6 | 10.3 ms | 4.83 ms |

  One row stays on 0015 (1.91 ms at 131k).
- **The forms tried** (131k keys, 6 rows, f16): no local memory 8.86 ms;
  K and V staged 5.27; K direct, V staged 4.83. One side per tile 10.6, four
  sides 6.96; BK 16: 4.91, BK 64: 9.67.
- **q8_0 KV, llama-bench at 32k depth:**
  - a 6-row forward 47.7 -> 52.6 t/s with K staged; direct K 30.7 (every
    sub-group converting the same blocks);
  - a 3-row forward 29.2 -> 27.7, hence the 4-row threshold.
- **Served, the agent's flags** (131,072 tokens, q8_0, MTP 5 drafts), a
  62,597-token prompt:
  - decode at that depth 11.0 -> 15.6 t/s (+42 %);
  - verify time 9.78 -> 6.65 s;
  - identical draft statistics (72 of 235 accepted);
  - prefill unchanged (286.9 t/s).
- **Answers:**
  - greedy acceptance task 10/10 with 0.5.6 and with 0016;
  - its decode 47.5 -> 47.6 t/s;
  - sampled 10 runs mean 7.6 (0.5.6: 7.2-7.4);
  - KL through this kernel: llama-perplexity with q8_0 KV and a 6-token
    ubatch, so every attention call is a verify-shaped call. 0015 0.003593
    (top-1 97.82 %), 0016 0.003589 (97.79 %).
- **Tests:** FLASH_ATTN_EXT 2,757 of 2,758 at the shipped threshold, the
  failure being the pin's own f16 softcap case. 0016 adds 24 cases at the
  dense geometry with 4-7 rows (partial row tiles), KV 1,013 and 4,096,
  f16, q8_0 and q8_0 / q4_0.
- **Red:** `GGML_OPENCL_FA_GQA_OPTS=-DBK=24` (a third of each tile's keys
  skipped) fails every case at 4-8 rows (42).
- **Open:** a 6-row call at 131k is still 2.5x a 1-row call (the gate's
  bar is 2x). Staging and the products do not overlap, and both sides
  compute S.

## 0017-opencl-intel-head-size-128-xmx-attention.patch

The XMX attention kernels at head size 128 as well as 256, for Mistral Small
3.2 24B and its finetunes (Cydonia 24B): 40 layers, all full attention, 32
query heads on 8 KV heads.
- 0008's prompt kernel and 0016's verify kernel are built a second time with
  `-DDK=128`, for f16 and the quantized pairs. Routing is by head size; the
  gates are as before.
- Decode at head size 128 already ran on 0007/0015's split kernel.
- test-backend-ops gains 48 cases at that geometry: batch 1, 3, 5, 6, 7, 8,
  64 and 512, KV 1,013 and 4,096, f16, q8_0, q8_0 / q4_0. Also two prompt perf
  cases at depth.

Measured (`measured-here`, B60, Cydonia 24B v4.3 Q4_K_M, llama-bench against
the 0016 build, where head size 128 took the upstream kernels):

| KV | prefill 512 | prefill 4,096 | prefill 512 at 16k depth | decode |
|---|---|---|---|---|
| f16, 0016 | 1,099 t/s | 472 | | 25.2 |
| f16, 0017 | 1,407 | 1,225 | | 25.2 |
| q8_0, 0016 | | | 168 | 18.2 at 16k |
| q8_0, 0017 | | | 451 | 18.2 at 16k |

- FLASH_ATTN_EXT 2,805 of 2,806 (the pin's own f16 softcap case at head
  size 256).
- Red: BK=24 in both kernels (`GGML_OPENCL_FA_DPAS_OPTS`,
  `GGML_OPENCL_FA_GQA_OPTS`) fails 41 of the 53 cases at that geometry. That
  is the 36 added before the 5- and 7-row ones, plus the pin's 17. The 12
  still passing are batch 1 and 3, which stay on the split kernel.
- Served at `--n-ctx 98304` with q8_0 KV: an 89,265-token prompt at 221 t/s,
  then decode at 8.4 t/s at that depth.
- KL against the model's own Q8_0 on the CPU: 0.019742 (0016) -> 0.019850
  (0017), top-1 94.76 -> 94.83 %; with q8_0 KV 0.020316, 94.85 %.
- The quant (0.5.9): bartowski's imatrix Q4_K_M of the same model (also
  14.33 GB), same corpus, window and reference, 0017. It gives KL 0.016792
  (top-1 94.73 %) with f16 KV and 0.016918 (94.90 %) with q8_0 KV, against
  the static quant's 0.019850 / 0.020316. Prefill 1,223 t/s at 4,096 tokens
  and decode 25.0 t/s, as the static quant. The writer serves the imatrix
  quant since 0.5.9. The corpus is general text, not role play.
- A770 (its sub-group-8 build of the prompt kernel): FLASH_ATTN_EXT 2,804
  of 2,806, the same two f16 failures as before 0017.

## 0018-opencl-intel-gqa-decode-searched.patch

The verify kernel of 0016/0017 as a structural search found it
(`tools/kq_tune/fagqa.py`, `docs/campaigns/kernel-autotune-ga.md`, "code-level
search"). Two changes of default and two new switches:
- **`GQA_S_SPLIT` (default with two or more sides):** the sides of an 8-row
  tile split QKᵀ by 16-key group and exchange S through local memory (one
  barrier more per tile). Before, both sides computed all of S.
  `GQA_NO_S_SPLIT` turns it off.
- **The kernel's own split size, 512 keys** (`GGML_OPENCL_FA_GQA_KV_PER_SPLIT`;
  the per-head split kernel keeps 128). A work-group covers a whole KV head,
  so longer splits mean fewer partials. It changes nothing from 131,072 keys
  on: the 256-split cap already made the splits 512 keys long there, so the
  gain at 131k is S-split's (`code`).
- **`GQA_PF` (off):** the next V tile prefetched into registers. 8 alone
  gains 7 %, but with `GQA_S_SPLIT`, or at 4, IGC falls back to a rolled
  form (3,181-3,383 assembly lines against 10,176-12,453) at 2-2.5x the time.
  Removed again by 0019, whose 64-value V staging it no longer matched.
- The test-backend-ops perf cases gain 4 and 8 rows.

The search (`measured-here`, B60, one attention layer at the dense 27B's
geometry, f16, the geometric mean over 4, 6 and 8 rows at 32k and 131k keys,
against 0.5.9's kernel re-measured every 8 evaluations, drift under 0.4 %):
- exhaustive over sides, K mode, tile and split size (90 genomes): best 0.863
  (split 512), median 1.356;
- GA adding S-split (59 of 216 evaluated): best 0.783 (S-split, split 512);
- GA adding prefetch (60 evaluated): best 0.798 (prefetch 8, no S-split);
- the local enumeration of S-split × prefetch × split (18): 0.784 for
  S-split and split 512; both genes together 2.44.

End to end (`measured-here`, B60, q8_0 KV, the same binaries with the
switches set by environment):

| model | rows at 32k depth | 0.5.9 | 0018 |
|---|---|---|---|
| dense 27B | 4 / 6 / 8 | 36.7 / 52.6 / 62.1 t/s | 39.4 / 64.2 / 78.5 |
| Cydonia 24B | 4 / 6 / 8 | 29.2 / 42.3 / 58.6 | 36.1 / 58.8 / 76.8 |

- Served, the agent's flags, a 62,597-token prompt: decode at that depth
  15.6 -> 16.4 t/s, identical draft statistics, the acceptance task 10/10
  in both.
- KL through the kernel (dense 27B, q8_0, 6-token ubatches): 0.003589 ->
  0.003587, top-1 97.79 -> 97.87 %.
- Gate: FLASH_ATTN_EXT at head sizes 256 and 128 with the kernel taking
  every row count, 860 of 861 (the failure is the pin's softcap case) for:
  - the exhaustive best;
  - the three best S-split genomes;
  - the two best of the local enumeration (S-split; prefetch 8).
  Prefetch has no failing-first test of its own; it is off.
- 0018's defaults: FLASH_ATTN_EXT 2,805 of 2,806 on the B60. A fault in the S
  exchange (a side reads the other group's S) fails 68 cases, all at 4-8
  rows. The A770 (2,804 of 2,806) does not build this kernel; its run shows
  only that nothing else changed.

## 0019-opencl-intel-attention-agent-searched.patch

The prompt kernel (0008's `flash_attn_dpas.cl`, the sub-group-16 build only)
and the decode / verify kernel (0016's `flash_attn_gqa_dpas.cl`) as two LLM
mutation agents rewrote them overnight, one kernel each, every candidate
compiled, tested and timed on the B60 before the next
(`docs/campaigns/kernel-autotune-ga.md`, "LLM mutation search"). Kernel
source only; the host is unchanged.

Prompt kernel:
- **Q in registers, not local memory.** The DPAS a-operand of every 16-column
  chunk is built once from global memory (lane = column, component = row) and
  kept for the whole key loop; the Q staging loop and its 64 KB buffer go.
- **K and V share one local buffer** (`BK * DK` halves, 16 KB at head size
  256): K is staged, S computed, the buffer rewritten as transposed V, P V
  computed, then the next K staged. 96 KB of local memory a work-group
  becomes 16 KB (`code`), which leaves room for more work-groups an Xe core
  (the occupancy itself is not measured); the extra barriers are paid for
  on purpose.
- Loads issued early into registers: the first K tile while the Q
  registers are built, each tile's V before the loop-top barrier.
- Q K^T as two DPAS chains a key group over the halves of the head
  dimension (four independent chains at BK 32).
- One row maximum across the key groups before the sub-group reduction, and
  one sub-group vote that skips the eight row reductions and the rescale of
  O and l when no row's running maximum can rise (the common case away from
  the diagonal; exact, the skipped correction factor is 1).
- Compile-time guards: four staging units a work-item, an even chunk count.

Decode / verify kernel:
- **Wider staging units:** quantized K and all V staged 64 values a
  work-item (V: four 16-value loads before the stores; K: four load-store
  pairs) instead of 8.
  This is the q8_0 path the agent serves (`GQA_K_STAGED`); f16 K stays direct.
- Q K^T before the V staging, so one barrier covers the V tile and the
  S exchange; the S exchange laid out lane-major ([tile][group][lane][row]),
  one `vstore8` / `vload8` a group.
- Masking as `fmax(s + mask, M_INIT)`, one sub-group maximum per row across
  the groups, P and l as float8 vectors. l is now the sum of the f32 P,
  where P V multiplies P rounded to half (before: the sum of the rounded P);
  the KL below includes it.
- `GQA_PF` removed (see 0018). Guard: head sizes a multiple of 64.

Measured (`measured-here`, B60, the dense 27B Q4_K_M, q8_0 KV, 32k keys deep,
llama-bench with both arms running the same tests in the same order; the
shipped arm is 0018):

| | 0018 | 0019 |
|---|---|---|
| pp512 (prompt) | 297.5 t/s | 428.7 (+44.1 %) |
| pp6 (MTP-5 verify) | 64.2 | 71.2 (+10.9 %) |
| pp8 | 78.5 | 87.3 (+11.2 %) |

- KL against the f16 reference (16 chunks of 512, q8_0): prompt path
  (ubatch 512) 0.003966 -> 0.004099, same top-1 97.647 -> 97.745 %; verify
  path (ubatch 6) 0.003587 -> 0.003588, 97.868 -> 97.843 %.
- FLASH_ATTN_EXT at head sizes 256 and 128: 860 of 861 on the B60 for both
  kernels (the failure is the pin's softcap case). On the A770 the same run
  passes 860 of 861 but exercises neither change: the A770 builds the
  sub-group-8 prompt kernel, which 0019 does not touch, and not the GQA
  kernel at all. Its four sub-group-8 builds compile to byte-identical
  binaries before and after (ocloc, dg2).
- Served (arcint on libllama, the agent's flags, two arcint builds of one
  source with the series to 0018 and to 0019, two rounds each): a
  62,597-token prompt prefills at 287.2 / 286.8 -> 408.1 / 407.7 t/s
  (+42 %), and the answer after it decodes at 18.7 / 18.7 -> 21.6 / 21.6 t/s
  (+15.5 %, MTP 5). Both answers coherent and on the prompt's task, each
  build's text the same in both rounds. The acceptance task at T=0 with
  thinking off (the agent's protocol): 8/10 for both. The 0018 section's
  16.4 t/s and 10/10 came from another arcint working tree with the switches
  set by environment; across the two windows the absolute numbers do not
  compare, within each the arms do. A greedy score is one trajectory and
  flips between 8 and 10 with the configuration (`docs/llama-engine.md`).
- One review fix was dropped again: a guard around the rescale's
  `native_exp` put 896 bytes of spill into the q8_0 prompt build and took
  15 % of pp512 (`measured-here`); the kernel comment says why it is not
  there.

## 0020-opencl-intel-a770-attention-gemm-agent-searched.patch

The Arc A770's (sub-group 8) prompt-attention kernel and K-quant GEMM tiles,
as two LLM mutation agents rewrote them in one day
(`docs/campaigns/kernel-autotune-ga.md`, "A770"). Kernel source only, every
change under `#if SG == 8`: the Arc Pro B60's sub-group-16 builds compile
to byte-identical binaries (ocloc, both files, `measured-here`).

Prompt attention (`flash_attn_dpas.cl`, the SG 8 kernel):
- the sub-group vote of 0019 (skip the row reductions and the rescale when
  no row maximum rises);
- Q staged as vectors; K's leading dimension and V's rows padded by 16 halves
  against local-memory bank conflicts (local memory a work-group at head
  size 256: 49,152 -> 58,368 B, within the 64 KB the host checks);
- S in four explicit accumulators (two key groups), the mask branch hoisted,
  the mask load clamped and unconditional, per-row mask offsets;
- the reciprocal of l precomputed for the final scale. This step moved
  IGC's register allocation as a whole (12,093 -> 9,744 assembly lines,
  pp512 at 16k +14 % in one step): an allocation effect, which a compiler
  change can move again;
- K and V staging in one merged loop, their global loads interleaved
  (spill 2,336 -> 0 bytes);
- in the shared `load8` helper, under `#if SG == 8`: q8_0 scaled in half
  (bit-identical to the float path unless f16 denormals are flushed,
  `code`).

K-quant GEMM tiles (`mul_mm_kq_f16.cl`, SG 8; MUL_MAT_ID for the MoE
experts and MUL_MAT):
- the weight tile as a double buffer of two 128-weight chunks in the same
  local memory as before (`MT * 256` halves, the host's check unchanged):
  the DPAS of one chunk is issued before the next chunk is dequantized, one
  barrier a chunk;
- q4_K / q5_K weights read 32 bytes at a time; a * q - b as one f32 fma on
  coefficients rounded to half first;
- `kernel_cvt_f32_f16` as float4 where aligned; a lane owning a tile row in
  the q4_K/q5_K dequant when the tile is at most 32 rows.

Measured (`measured-here`, A770, the coder GGUF Q4_K_M, q8_0 KV):

| | 0019 | 0020 |
|---|---|---|
| pp512 after 16,384 (llama-bench, paired, 2 runs) | 496.2 t/s | 692.3 (+39.5 %) |
| pp512 after 4,096 | 1,116.2 | 1,346.8 (+20.7 %) |
| pp512 after 0 | 1,691.9 | 1,744.6 (+3.1 %) |

Those are the attention kernel alone; the GEMM alone: pp512 after 4,096
1,116 -> 1,149 (+2.9 %), after 0 1,694 -> 1,769 (+4.4 %).

- Served (arcint, MTP 4, q8_0 KV, 16k context, two arcint builds of one
  source, two rounds): a 15,004-token prompt prefills at 761.8 / 762.4 ->
  1,008.6 / 1,008.1 t/s (+32 %). OpenVINO's path served the same prompt at
  1,025 t/s that morning. Decode 41.5 -> 39.5 t/s in both rounds: the same
  cost a draft-verify cycle (verify 54 -> 55 ms, propose 23 ms), a lower
  draft acceptance on this one greedy answer (55.6 -> 51.3 %); 0020 changes
  no decode kernel on the A770.
- The coder's acceptance task (T=0, thinking off): 10/10 in both arms.
- KL against the coder's reference (16 x 512, ubatch 512): attention
  0.006996 -> 0.007062, same top-1 95.980 % in both; the GEMM bit-equal
  (0.006996).
- FLASH_ATTN_EXT at head sizes 256 and 128: 860 of 861 on the A770 (the
  pin's softcap case); MUL_MAT_ID 338/338, MUL_MAT 1,055/1,055. Not in
  test-backend-ops: q4_0/q4_0 attention at head size 128, plain MUL_MAT
  IQ4_NL with K not a multiple of 256 at more than 4 columns.

## 0021-expert-cache-slots-and-usm-bank.patch

A per-expert cache for MoE layers whose experts stay in host memory
(`src/llama-expert-cache.{h,cpp}`, `docs/campaigns/flash-next-llama-engine.md`
"Stage 2b"), after Strata's expert cache and adapt step and FreeToken's
gather of missing experts (`code`, both). Off unless
`llama_model_params.expert_cache_bytes` (or `LLAMA_EXPERT_CACHE_MIB`) is set
and the experts are placed on the CPU (`--n-cpu-moe` / `-cmoe`).

libllama:
- `llama_model_params.expert_cache_bytes`, `.expert_cache_profile` (Strata's
  STRP decode profile) and `llama_model_expert_cache_info`; the loader keeps
  each GGUF shard's path;
- per layer: slot tensors on the card (the hot experts, sized in bytes), a
  bank of the other experts in USM host memory, residency tables;
  `build_moe_ffn` looks the router's ids up in the tables and sums the slot
  and bank branches; decode's ids are read back once a step for the usage
  counts; every 12 decode tokens hot bank experts exchange with cold slots
  on the card's queue.

ggml-opencl:
- USM host buffers (`clHostMemAllocINTEL`) behind `get_proc_address`
  ("ggml_backend_opencl_set_alloc_host"), their tensors passed to the kq
  kernels as pointers; a `cl_mem` over host memory is migrated to the card by
  the driver (`measured-here`, B60);
- `get_rows` for I32; slice writes, reads and exchanges of weight tensors in
  their card layout (Q8_0 and IQ4_NL as scales, then quants); batched reads;
- the kq MoE kernels skip a pair whose id is -1 (the matvec writes its zeros,
  the GEMM's tile router drops it after a zero fill of a cache tensor's
  output); a gather kernel copies the routed experts from a USM bank into a
  card-side mirror before the kernels run.

ggml-cpu: `MUL_MAT_ID` leaves a row zero for a negative id (both grouping
loops); unused by 0021's final design, kept for a CPU branch. The K-quant
matvec returns on a negative id (defensive: the cache takes IQ3_XXS, IQ4_XS
and IQ4_NL experts, Q8_0 down projections, nothing else).

Measured on the Arc Pro B60 with Qwen3.8-Flash-Next UD-Q3_K_XL from NVMe
(`measured-here`, 2026-10-06; 12,800 MiB of slots, 36.5 GiB of bank):
- the 20,045-token needle prefills at 149.6 t/s against stage 1's 81.9 (16
  expert layers on the card), answered; 235.0 t/s with 2,048-token ubatches
  (arcint's default with the cache);
- decode 12.2-12.3 t/s against stage 1's 10.5-12.4; 75 % of decode's routed
  experts on the card after adaptation;
- KL against the CPU reference 0.029634 / 95.784 % same top-1, stage 1
  0.029206 / 95.931 %.
