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
  included (about 250 us of it):

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
