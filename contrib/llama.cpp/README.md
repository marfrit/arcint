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

A patch's KL check runs two arms: one sequence a batch, and four sequences a
batch (`llama-perplexity -b 4*c`, ubatches that mix the sequences). arcint
decodes one sequence per call, but llama.cpp's own batching does not. 0022
shipped a defect only the second arm sees (fixed by 0027).

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
  card-side mirror before the kernels run;
- the IQ matvec's trailing arguments (a mirror and a split): for a slot
  tensor paired with its bank, ids at or above the split read the mirror, so
  a decode step runs one branch (INT_MAX: off).

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

## 0022-opencl-intel-low-bit-iq-and-f16-small-n.patch

The ISTA-DASLab GSQ-RCO quantisations of Qwen3.8-Flash-Next (the files
Strata serves) store their experts in types ggml-opencl had no kernel for:
gate/up IQ2_S, IQ2_XXS or IQ1_M, down Q2_0; their dense projections add
IQ3_S. llama.cpp then left those layers to the CPU. The patch adds them, and
the decode-path work Flash-Next and its MTP verify needed on Intel:

- **Low-bit kq types** (kq type index 6-10, blocks as stored): matrix-vector
  for 1 to 4 columns and `MUL_MAT_ID` (`mul_mv_iq_q8_1.cl`, one 32-weight
  sub-block a lane expanded to int8 and two scales; IQ1_M's values are
  8 (g + delta), exact in int8), the f16-tile GEMM (`mul_mm_kq_f16.cl`), the
  expert cache (0021) taking them; Q2_0's K need only be a multiple of 64.
  The expanders run in 32-bit only (64-bit integer arithmetic is emulated:
  IQ2_S 168 -> 146 us, IQ3_S 284 -> 255 at 4096 x 14336, B60); IQ2_XXS and
  IQ1_M read their grids from local memory, IQ2_S and IQ3_S from constant
  memory (a local copy per work-group cost more there).
- **IQ2_XXS, IQ2_S, Q2_0 and IQ3_S in planes on the card**, per weight slice
  (an expert): each field of every block back to back, so a lane reads
  aligned 32-bit words where the 18-, 66-, 82- and 110-byte GGUF blocks left
  fields at 2-byte offsets (single-byte loads). A slice keeps its size and
  place: the expert cache's slot / bank copies and gathers stay byte copies;
  `set_tensor`, `get_tensor`, the slice procs and the USM bank convert GGUF
  blocks <-> planes (`ggml_cl_lb_convert`); an odd block count a slice stays
  in blocks and its ops go to the CPU. 4096 x 14336, one column (B60): Q2_0
  184 -> 99 us, IQ2_XXS 211 -> 111, IQ3_S 255 -> 156, IQ2_S 146 -> 142.
- **F16 / BF16 (stored as F16) times 2 to 8 columns** in one read of each
  weight row (`kernel_mul_mv_f16_f32_nc`): the tiled GEMM took 2.80 ms for
  4096 x 14336 at 2, 4 or 8 columns against 0.29 ms at one; now 0.70-0.89.
  Few rows and a long K (a hyper-connection inject, a router at 2+ columns)
  take a work-group a row (`kernel_mul_mv_f16_f32_ksplit`).
- **ADD** of two contiguous same-shape F32 tensors on the flat kernel (the
  row kernel gave a [2560 x 4] add one 64-wide work-group a row: ~23 us).
- **GGML_OP_DSV4_HC_PRE / _POST** (identity comb), as ggml-sycl computes
  them: llama.cpp's fusion probe then builds Qwen3.8-Flash-Next's
  hyper-connection mixers fused.
- **The MoE router fused** (SOFT_MAX .. ARGSORT .. GET_ROWS .. SUM_ROWS,
  CLAMP, DIV -> one kernel, as ggml-cuda's topk_moe; `GGML_OPENCL_FUSE_TOPK_MOE=0`
  turns it off).
- **The expert cache's gather for 2+ tokens** copies each routed bank expert
  once (the per-pair grid copied an expert two tokens share twice and spent
  ~650 us a projection on a 4-token step, profiled).
- **IQ4 codebook per card**: register selects where the device has 8-wide
  sub-groups (A770: MV_ID 34 -> 28 us), constant gathers elsewhere (B60:
  130 vs 146 us dense). Superseded by patch 0035 (a local-memory codebook
  on both cards).
- **SCALE -> SILU and SCALE -> SIGMOID [-> SCALE] fused** into one
  elementwise kernel (`kernel_scale_act_f32`, `GGML_OPENCL_FUSE_SCALE_ACT=0`
  turns it off): fewer launches a step, the served text identical; no
  decode change measurable above the run-to-run spread.
- The gather's shape for 2+ tokens is switchable for measurement:
  `GGML_OPENCL_GATHER_PAIRS_MULTI=1` takes the per-pair grid back,
  `GGML_OPENCL_GATHER_CHUNKS` sets the per-expert split. Served, the shape
  did not move the gathers' time (their bytes over the link do).
- A matvec grouped by expert for 2-15 tokens (`kernel_mul_mv_idg_*`), opt-in
  only (`GGML_OPENCL_KQ_IDG=1`): slower than a pair a work-group where the
  matvec is not bandwidth-bound (the coder's verify 7.85 -> 9.18 s, A770).
- **The expert cache's decode matvec reads a missed expert where it lies**
  (USM host memory, by pointer argument; Strata's expert kernels read its
  pinned host mirror the same way): no gather into the card-side mirror
  first (`GGML_OPENCL_BANK_DIRECT=0` gathers). Flash-Next MTP 2: 26.0 ->
  28.2 t/s.
- **The order of a verify's expert pairs**: a permutation computed once per
  layer (`kernel_moe_pair_perm`, the three projections share the ids) puts
  the pairs read over the link first (`GGML_OPENCL_PAIR_ORDER`, 1 by
  default): 29.8 -> 30.7 t/s; bank pairs last 29.3.
- **Q2_0 rows of 20 sub-blocks** (Flash-Next's 2560 x 640 down projections)
  take 4 lanes a row instead of 16: 135 -> 89 us at 3 tokens, 47 -> 32 at
  one.
- **A 2D weight times columns lying back to back** (an MTP step's [K, 4,
  tokens]) runs as one few-column product; the F32 GEMV takes up to 1,024
  rows (an MTP layer's F32 router).
- **F16 / BF16 weights times 16+ columns on the XMX tiles** of the K-quant
  GEMM (`kernel_mul_mm_f16w_f16`; `GGML_OPENCL_F16_XMX=0` keeps the tiled
  GEMM): Flash-Next's prompt-side hyper-connection products 4-5x faster
  (320 x 10240 at 512 tokens 1,907 -> 468 us); the needle's prefill 342.8 ->
  383.4 t/s.
- **The QSA indexer's ops on the card**: TOP_K (a radix select a row),
  CPY F32 <-> I32, FILL / REPEAT / SET_ROWS for F16. Without them the
  scheduler ran four CPU islands per attention layer (98 splits a Flash-Next
  verify graph), each draining the queue: decode 32.0 -> 33.0-34.2 t/s,
  the needle 381 -> 422-425 t/s, the long text identical.
- Opt-in, measured no faster served: the hyper-connection read fused into
  three kernels (`GGML_OPENCL_FUSE_HC_MIX=1`, after Strata's `fused_gr` v3
  split by stream and K half; 30.8 t/s off, 30.4 on), a projection's matvec copying the next
  projection's misses into a card-side mirror (`GGML_OPENCL_MOE_PREFETCH=1`,
  31.4 -> 31.6), the K-split F16 kernel with 4 rows a work-group
  (`GGML_OPENCL_F16_KSPLIT4=1`); measured slower and kept opt-in: a verify's
  pairs grouped by expert and decoded once (`GGML_OPENCL_KQ_GROUPED=1`,
  30.5 -> 29.5); the activation as one int8 term in the K-quant and IQ
  matvecs (`GGML_OPENCL_KQ_ONE_TERM=1`, llama.cpp's and Strata's single
  q8_1 instead of codes plus residual codes): IQ4_XS 49.6 -> 43.9 us at 3
  columns, Flash-Next 31.0 -> 31.6-31.8 t/s, but the coder 43.3 -> 40.7 (its
  MTP drafts accepted 42.7 -> 38.2 %). The expert cache's adaptation constants take
  `LLAMA_EXPERT_CACHE_{ADAPT_EVERY,MAX_SWAPS,MIN_GAIN,MIN_USAGE,DECAY}`
  (the hit rate stayed at 89.9-90.7 % across them), and it counts a
  verify's distinct experts (logged every 200 steps). `GGML_OPENCL_IQ_OPTS` adds defines to the IQ
  matvec builds. Measured slower and removed: the IQ4 codebook by byte
  pairs and per lane in local memory, a per-work-group scan for the pair
  order. (Patch 0035's local codebook is a different build: one table per
  work-group, with each value already in its byte.)

`test-backend-ops` against the CPU (`measured-here`, B60 and A770): MUL_MAT
557/557 and MUL_MAT_ID 181/181 for the IQ, low-bit, F16 and BF16 types,
DSV4_HC_PRE 18/18, DSV4_HC_POST 4/4, ADD 103/103, TOPK_MOE 416/416, SOFT_MAX
214/214. An unfiltered MUL_MAT_ID run fails the 74 MXFP4 cases on the B60 only, and
so does the 0021 tree's own build (0/74, `measured-here`): a defect of the
MXFP4 MoE path on Xe2 that predates this patch, not a type it touches. The
final tree (B60): MUL_MAT f16/bf16 434/434 (the XMX path's red case, K cut
by 16, fails 7), MUL_MAT_ID IQ/low-bit 193/193 (with Q2_0 at K 640, 384 and
320; the 4-lane path's red case fails 47 of 87), HC_MIX 10/10 fused (its red
case, the mix 5 % off, fails 9 of 10), MUL_MAT IQ4_XS/Q4_K 94/94. Run
`test-backend-ops` from its own `bin/` when the kernels are not embedded,
and with `-b GPUOpenCL`: a wrong backend name skips everything and still
prints OK.

Served, B60, Qwen3.8-Flash-Next IQ2_XS from the expert cache, the gate's
requests (`measured-here`, 2026-10-06): decode 21.3 t/s plain (14,500 MiB of
slots), 27.0 t/s with MTP 2 from an MTP-only file (13,500 MiB), prefill
371.1 t/s at 20k; answers right; the cache's file check 0 of 6 copies
differ. Before the patch's decode work: 16.0 and
13.8. The resident models on the same build: the dense 27B agent (B60, MTP 5)
35.9 -> 38.7 t/s, the coder (A770, MTP 4) 40.1 -> 43.3 t/s, the long answer
greedy. With the rest of the patch (2026-10-07): MTP 2 30.7-31.6 t/s
(13,500-14,500 MiB), MTP 3 with arcint's `--llama-mtp-min-p 0.5` 31.3, the
needle's prefill 383 t/s; the residents unchanged (38.2 / 43.5).

**Known defect (found 2026-10-08, fixed by 0027):** with 0022 applied, a batch
whose ubatches mix sequences (llama.cpp's own batching: `llama-server -np
N`, `llama-perplexity` with several sequences a batch) gives wrong logits
on the B60. The dense Qwen3.8-27B scores KL 0.2438 (top-1 81.96 %) against
0.0041 for one sequence; pin + 0001-0021 is right. arcint decodes one
sequence per call and is not affected. The bisection is in
`docs/campaigns/lanes-agent-subagent.md` (on the `qfndev` branch).

## 0023-opencl-one-context-per-platform-multi-device.patch

One process on two Intel Arc cards, for llama.cpp's layer split. Intel's
OpenCL runtime (26.27) lists two different GPUs as two platforms, one device
each, and ggml-opencl probed one platform and made one context for all its
devices. With the patch:

- **Selection:** `GGML_OPENCL_DEVICES="P:D[,P:D...]"` selects devices across
  platforms, in that order; the first is the main device. Each platform gets
  one context holding its selected devices. When the variable is unset,
  `GGML_OPENCL_PLATFORM`/`GGML_OPENCL_DEVICE` and the default behave as
  before.
- **Builds:** programs are built for the one device that uses them
  (`clBuildProgram` with that device). The program cache saves that device's
  binary; its key already carries the device name and driver.
- **Buffers:** `supports_buft` accepts only the device's own buffer type.
  Activations crossing devices go through the scheduler's host copy (the
  backend has no events and no `cpy_tensor`).
- **Names:** devices `GPUOpenCL`, `GPUOpenCL1`, ...; buffer types and
  backends `OpenCL`, `OpenCL1`, ...
- **Per-device state** that was process-static:
  - the maximum allocation size;
  - the 512-wide flash-attention failure flags;
  - the transpose, `get_tensor` and cumsum scratch buffers;
  - 0021's USM entry points (resolved per platform);
  - its exchange scratch and its keep ring (finished on the owning queue);
  - 0022's prefetch state.
  `get_tensors(_async)` and `exchange_slices` assert that their tensors
  share one device.
- **Synchronisation:** `sync_with_other_backends` waits only on backends in
  its own context. An event cannot cross contexts.

Measured on the B60 + A770 (`measured-here`, 2026-10-07, both cards on their
own platforms):

- **Device lists:**
  - `GGML_OPENCL_DEVICES=0:0,1:0` lists both cards;
  - unset, the B60 alone, as before;
  - the 0001-0022 build with the variable set lists one OpenCL device (the
    red case).
- **`test-backend-ops` with both devices in the process:**
  - MUL_MAT 1201/1201 on each card;
  - GET_ROWS 29/29 on each;
  - MUL_MAT_ID 520/520 on the A770;
  - the same counts as the A770 alone under `GGML_OPENCL_PLATFORM=1`.
  - On the B60, MUL_MAT_ID 446/520: the 74 MXFP4 cases noted under 0022,
    the same set as the 0001-0022 build's.
- **The coder Q4_K_M** (14.94 GiB), llama-bench `-fa 1`:

  | | pp512 | pp4096 | tg64 |
  |---|---|---|---|
  | B60 alone | 2,618 | 2,470 | 77.2 |
  | split `-ts 3/2` | 2,210 | 2,945 | 57.5 |
  | split `-ts 1/1` | 2,148 | 2,976 | 55.0 |

  This model fits on the B60 alone, so a split costs decode: the A770 runs
  its layers slower, and every token crosses once.
- **Answer:** the capital question across both cards is right ("Paris").

## 0024-expert-cache-per-device.patch

The expert cache of 0021, one per GPU under llama.cpp's layer split (0023):
each card keeps slots, tables and a USM bank for the layers it runs, in its
own context, with its own budget and counters.

- **API:**
  - `llama_model_params.expert_cache_bytes_dev` /
    `n_expert_cache_bytes_dev` give a budget per device, in device order;
    without them `expert_cache_bytes` goes to the first card with cached
    layers, as before. `LLAMA_EXPERT_CACHE_MIB` takes `A,B`.
  - `llama_model_expert_cache_count`, `_info_dev` and `_device` report each
    cache. `llama_model_expert_cache_info` sums them; `adapts` counts the
    shared plan once.
- **Swap planning:** one plan over every cache, with one cap. The largest
  gains win wherever they are, and each swap runs on the queue of the card
  that owns its layer, as Strata's adapt does across its stages
  (`code`: generate.cpp:4430-4471).
- **The profile:** each cache ranks only its own layers' pairs.
- **Single card unchanged:** the same layers, budget, plan order and logs
  (the log lines now name the device and the layer range).

Measured on the B60 + A770 (`measured-here`, 2026-10-07; Flash-Next IQ2_XS,
MTP 2, 131,072 tokens of context):

- **Copy checks:** 0 differences on both cards.
- **Answers:** right in every arm.
- **Eviction:** none (each process's GTT stayed at its bank).
- **Split at layer 40:** the A770 holds every expert of its 8 layers, 90.4 %
  of the routed experts are on a card, prefill 378 t/s, decode 18.6 t/s.
  The B60 alone: 84.4 %, 408 and 19.8. The campaign gate (decode above the
  one-card arm) is not met yet:
  `docs/campaigns/layer-split-two-cards.md`.

## 0025-opencl-strided-rows-f32-copy.patch

A row-parallel f32 copy (`kernel_cpy_f32_f32_rows`) for same-shape copies
with contiguous rows of at least 256 floats whose destination is not
contiguous. Each work item moves one float4 of a row; one extra item copies a
row's last up to 3 floats. Before it, ggml-opencl ran such copies through its
generic `kernel_cpy_f32_f32`, one 64-item work-group a row.

- **Which copy:** the case that mattered is the MTP snapshot write of the
  hybrid models (`src/models/delta-net-base.cpp`, the recurrent state's
  `1 + n_rs_seq` planes into a view whose planes are `mem_size` rows apart).
  It is contiguous only when every configured sequence is in the ubatch.
- **When it hit:** with two sequences configured (arcint's named lanes, or
  `--parallel 2`) and one busy, each GDN layer's write took ~4.4 ms. A verify
  of the dense Qwen3.8-27B took ~260 ms instead of ~55.
- **Switch:** `GGML_OPENCL_CPY_FLAT=0` turns off this path and the flat one
  (the old behaviour). The kernel is loaded as optional; without it the old
  path runs.

Measured on the B60 (`measured-here`, 2026-10-08):

- **`test-backend-ops -o CPY`:** 146/146, with three new strided-destination
  cases.
  - The red case: the destination row stride taken from the source, the
    mutation checked in the built kernel file. It fails 2 of 146; with the
    old path forced the same build passes 146/146.
  - A first mutant missed its target and passed 146/146; it ran the
    unchanged kernel, so it is not counted.
- **Perf, every CPY case:** the snapshot-sized case (786,432 floats x 6
  planes into a strided view) 4,595.6 -> 95.2 us; two cases faster by more
  than 10 %; none slower by more than 10 %.
- **The dense Qwen3.8-27B** (Q4_K_M, q8_0 KV, MTP 5), the same 300-token
  answer:
  - one lane 41.0 t/s;
  - two lanes with one busy 10.9 t/s before, 40.9 t/s after;
  - the text byte-identical across the three, on two questions;
  - both lanes decoding at once: 18.3 + 20.1 t/s; the subagent lane's text
    identical to its run alone, the agent lane's diverging at char 372 into
    an equivalent sentence.

## 0026-kv-cache-cell-windows.patch

Cell windows in a unified KV cache (`kv_unified`): sequence i finds its
cells in its own range of the pool, and its attention runs over that range
only. Without it, a sequence's attention spans the pool up to the highest
used cell, so it grows with another sequence's cells, masked
(`get_n_kv`, the stream's `used_max_p1`). That was arcint's named lanes'
cost: a subagent lane slowed with the agent lane's fill.

- **API:** `llama_memory_seq_windows(mem, n_cells, n_seq)`. Sequence i gets
  the cells from the sum of the sizes before it, `n_cells[i]` long. Each size
  is a multiple of 256 and the sum at most the cache. Call it on an empty
  memory, before the first decode. It returns false and changes nothing for
  a memory whose attention cache cannot take windows: several streams, a
  shared source cache, the indexed hybrid (qwen4exp's QSA k-pool selects
  cells by absolute index), or a memory other than `llama_kv_cache` and the
  attention half of `llama_memory_hybrid`.
- **The cache:**
  - `find_slot` searches the window of the ubatch's sequence. It starts at
    the window's search head, or lower if the window's last used cell is
    lower: a sequence cut back by `seq_rm` refills from its end, not above
    its old high-water mark. `clear()` resets the heads.
  - `slot_info` carries the window's first cell and size. `get_n_kv` counts
    to the window's last used cell, and `get_k`/`get_v` view from its first
    cell. The KQ mask fills column j from cell j plus that offset.
  - `prepare()` restores the window heads with the cells. A sequence's
    state restore goes through `find_slot`, so it lands in its window; the
    tokens' own sequence ids choose the window. A whole-cache restore is
    refused while windows are set, and so is `seq_cp` between two
    windowed sequences (the destination never attends over the source's
    window). The relative-position buckets read cells from the offset too.
- **A ubatch of several sequences** (a hybrid memory's equal split): each
  token takes a cell in its own window, and the views span the whole stream
  as before. It is correct, without the speed-up.
- **Graph reuse:** the mask input records the window's first cell, and a
  graph is reused only for the same one: a graph's K/V views have the
  offset built in.
- **`llama-perplexity` test switches:** `LLAMA_PPL_SEQ_WINDOWS=1` gives each
  sequence a window of `n_ctx`, and `LLAMA_PPL_PER_SEQ=1` decodes each
  sequence of a KL batch in its own `llama_decode` (arcint's lanes).

Upstream has no per-sequence cap or range in a unified cache: the server's
`--kv-unified-per-slot` is one cap for all slots, and non-unified streams
are one size each (`docs/campaigns/research-agent-lanes.md` on the `qfndev`
branch).

Measured on the B60 (`measured-here`, 2026-10-08; the dense Qwen3.8-27B
Q4_K_M, q8_0 KV, MTP 5; arcint's named lanes, 131,072 + 32,768):

- **The red case for the search head** (found in review): an agent request
  of 29k, then a fresh 4k request in the same lane. Without the reset, the
  4k request decodes 34.1 t/s and finishes in 22.7 s (its cells above the
  old 29k). With it, 37.8 t/s and 15.6 s.
- **The cause, before the patch:** llama.cpp's KV debug shows `n = 33,741`
  while the subagent lane, holding 4,508 cells, decodes next to an agent lane
  of 29k. That cell decodes 30.3 t/s against 38.4 with the pool empty.
- **After:**
  - subagent 4k with the agent at 29k: 37.5 t/s (38.0 with the pool empty);
  - agent 29k: 32.2 t/s, against 32.3 in a one-lane server;
  - the subagent's 300-token answer with the agent lane at about 20k, 64k
    and 118k: 32.1 / 32.0 / 32.0 t/s, the same text each time and the same
    as the agent lane's alone (32.1 t/s). The first build, before the
    search-head reset: 28.7 / 28.5 / 28.3. Before 0025 and this patch:
    7.7 / 7.3 / 6.5. With 0025 alone, a 4k subagent cell next to a 130k
    agent: 14.2;
  - both lanes at once: 15.9 + 15.8 t/s, the sum of one lane's rate.
- **Answers:** the capital and the 20k needle are right on both lanes. The
  acceptance task on the agent lane scores 8/10 greedy, as the one-lane
  unit does, and 10, 8, 10 sampled.
- **KL** against the Q8 reference (`llama-perplexity`, 16 chunks of 512):
  - one sequence: 0.004099, top-1 97.745 %;
  - four sequences decoded one at a time in a shared pool: the same values
    without windows and with them.
- **Not this patch:** four sequences in ubatches that mix them score
  0.2438 / 81.96 %, with windows or without, unified or in separate streams.
  arcint decodes one sequence a call and never takes that path; see
  `docs/campaigns/lanes-agent-subagent.md` (on the `qfndev` branch).

## 0027-opencl-kq-cache-stack-temporary.patch

Fixes the known defect noted under 0022: a llama.cpp batch whose ubatches mix
sequences gave wrong logits on the Intel path.

- **The cache:** the Intel K-quant products convert their f32 activation once
  per graph and reuse it for every product that shares the input (q/k/v,
  gate/up). The four conversion slots (`kq_q8`, `kq_f16`, `kq_few`, `kq_i8`)
  are keyed by the tensor, its buffer, offset and size, and the graph count.
- **The temporary:** 0022's flat-columns path multiplies a 3D / 4D `src1` as
  one 2D product through a stack copy (`ggml_tensor s1 = *src1`). That copy
  has the same address in every call. ggml-alloc puts each layer's input at
  the same offset, so layer N+1's product matched layer N's conversion and
  multiplied stale data.
- **When it hit:** a mixed ubatch's recurrent output projection
  (`final_output`, [6144, 128, 4] for four sequences of 128 tokens) took the
  flat path in every GDN layer. One sequence per ubatch keeps `src1` 2D and
  never takes it.
- **The fix:** after the flat product, a slot keyed by the temporary is
  cleared (`ggml_cl_kq_forget`), also after the Adreno broadcast loop, which
  builds the same kind of temporary (that build compiles; it is not arcint's).

Found and measured on the B60 (`measured-here`, 2026-10-08; the dense
Qwen3.8-27B Q4_K_M, q8_0 KV, `llama-perplexity` KL against the Q8 reference,
16 chunks of 512, four sequences a batch):

- **Before:** mixed ubatches score KL 0.2438, top-1 81.96 %; one sequence
  scores 0.004099, top-1 97.745 %. Pin + 0001-0021 is right.
- **Localised:**
  - `GGML_OPENCL_KQ_DEDUP=0` alone fixes it;
  - a per-node dump hides it (each node is its own graph);
  - a log of every slot lookup and write shows `linear_attn_out-9` hitting
    a slot that `linear_attn_out-8` had filled, whose key tensor then read
    `final_output-9`.
- **After:** mixed ubatches, unified or in separate streams, score 0.004099
  and 97.745 %, the same as one sequence and as arcint's per-sequence path.
- **Flash-Next** (its hyper-connection streams take the flat path as 3D
  products): the capital, the 20k needle and a 300-token answer are
  byte-identical with and without the patch (23.4 / 23.3 t/s, the served
  flags at 32,768 tokens). No stale hit happened there.
- **The dense agent** (one lane, q8_0 KV, MTP 5, 32k; its draft context
  could take the flat path too): with and without the patch, the same text
  on two prompts, the same draft acceptance (172/634 and 1159/2005), and the
  same acceptance-task code (10/10).
- arcint decodes one sequence per call, so its served path never took the
  stale hit. The patch matters for llama.cpp's own batching with these
  patches (`llama-server -np N`) and for batching lanes into one decode.

## 0028-opencl-fa-row-blocks.patch

Intel: an attention call with 9 to 64 query rows runs in blocks of up to 8.
Each block is a recursive call on views: q's rows, the mask's rows and the
output's rows, offset by their strides. The small-row kernels (the q1 split
kernels and 0016's GQA kernel) take at most 8 rows. Without blocks, a verify
of two lanes (12 rows) fell to the prefill kernel, which pays a K/V
conversion every call. Rows of attention are independent, so a block attends
as the whole call would. Calls without a mask (a kernel may derive causality
from the row index) are not split. `GGML_OPENCL_FA_ROW_BLOCKS=0` keeps one
call.

Measured on the B60 (`measured-here`, 2026-10-09):

- **`test-backend-ops -o FLASH_ATTN_EXT`:** 2805/2806 with and without
  blocks; the one failure is the softcap case noted under 0022.
  - The red case: the block's mask offset dropped fails 426 cases
    (2380/2806); the same build with blocks off passes 2805.
- **Two lanes of the dense 27B in shared verifies** (arcint's combiner, MTP
  5): the prefill kernel had taken 2,034 calls at 4.3 ms. With 131,072 +
  32,768 lanes, one such step outran xe's job timeout (an engine reset).
  With blocks, a shared step takes 282 ms instead of 1,030.
- **Since 0029** a shared step's attention runs one op per lane (about 6
  rows each), so blocks no longer fire on the lanes' path: both lanes busy
  decode 22.3 + 20.9 t/s with blocks off, as with them on. They fire on a
  prompt tail of 9-64 tokens over a deep context (release review,
  2026-10-09): a 17-25-token follow-up over the dense agent's 122,383 tokens
  prefills in 0.65-0.88 s with blocks and 1.35-1.59 s without (three
  follow-ups each, the same answers).

## 0029-kv-windows-per-sequence-attention.patch

0026's windows, inside a ubatch of several sequences. 0026 fell back to the
whole stream's view there, so each sequence attended over every window's
cells, masked. With this patch, when windows are set, flash attention is on
and the ubatch is an equal split of windowed sequences, the attention
builder runs one op per sequence. Each op takes the sequence's query rows,
its window's K/V (`get_k_part`, `get_v_part`) and its slice of the mask. The
slice is made contiguous once a graph and shared by all layers. The outputs
are concatenated along the tokens.

- `llama_kv_cache::get_win_parts` takes a ubatch whose tokens come in one
  contiguous run per sequence, each sequence windowed: a hybrid memory's
  equal split, or a batch that adds each sequence's tokens together (an MTP
  draft context's split is simple, not equal). Otherwise the whole-stream
  view stays, as before.
- The mask spans every part's cells (`n_kv` up to the furthest window's
  end).
- The slot search of a mixed ubatch starts each window's head at its last
  used cell at the latest, as 0026's single-sequence path does. Found in
  the release review: without it a lane's verify cells went above the
  previous step's high-water mark every step, and the rollback's freed
  cells stayed holes. Measured (two lanes, 1,500 tokens each at once): a
  lane's span of cells 3,070 for 1,525 tokens without the clamp, 1,524 for
  1,518 with it; 53.4 s against 43.6 s for both answers.
- Graph reuse compares the parts, also in the hybrid memory's input. That
  one holds the attention input without registering it, so the attention
  input's own check never ran there. Found in review: a graph built for
  other parts (the lanes in another order, a window grown by a padding
  step while the furthest end stayed) was reused with stale views.
  - Measured (two lanes busy, a diagnostic on the new check): it refused 60
    such reuses in two 300-token runs. Before the fix, acceptance with both
    lanes busy was 23-26 %; after it, 28.0-28.2 %, as alone (28.2 %).
  - `llama-perplexity`'s `LLAMA_PPL_SEQ_FLIP=1` (every other batch's
    sequences in reverse order) did not reach it: another check refused the
    reuse there, so it is no red case.

Measured on the B60 (`measured-here`, 2026-10-09; the dense Qwen3.8-27B
Q4_K_M, q8_0 KV):

- **KL** against the Q8 reference, four sequences a batch in mixed ubatches,
  windows on: 0.004099 / 97.745 %, equal to one sequence and to the same
  batch without windows; the same with the slot-search clamp, per-sequence
  decoding and the flipped order.
  - The red case: every part's window offset forced to 0 gives 1.3291 /
    61.79 % with windows; without windows the same build stays at 0.004099.
- **Two lanes in shared steps** (131,072 + 32,768, MTP 5, a 300-token
  answer on each at once, with the reuse fix): 21.6-21.7 t/s a lane, 43.3
  together, against 15.9 + 15.8 taking turns. One lane's text equals its
  run alone.

## 0030-opencl-iq2-xs.patch

Intel: IQ2_XS joins 0022's low-bit types as kernel type 11, so the expert
cache takes it. The GSQ-RCO IQ3_XXS quant of Flash-Next carries IQ2_XS
gate/up experts in 10 of its 48 layers (6 with Q2_0 down, 4 with IQ4_NL
down). Without kernels for them, the cache refused those layers and their
experts ran on the CPU. The 300-token answer decoded at 3.7 t/s.

- **The type:** a 74-byte block is `d`, 32 codes of 16 bits and 8 scale
  bytes. A code is a 9-bit index into the 512-entry `iq2xs_grid` and a
  7-bit index into the sign table that IQ2_XXS already uses; a scale byte's
  nibbles scale the sub-block's two halves, as with IQ2_S
  (`dequantize_row_iq2_xs`).
- **On the card** in planes, like the other low-bit types: `d | qs[64] |
  scales[8]`.
- **Tables:** the matvec reads the grid and the signs from local memory, as
  IQ2_XXS does (640 entries). The GEMM reads them from constant memory.
- **Kernels:** the matvec (1-4 columns), MUL_MAT_ID, the grouped paths
  (`idg`, `idgg`) and the XMX GEMM (`mul_mm`, `mul_mm_id`).
- **The expert cache's type filter** takes IQ2_XS.

Measured (`measured-here`, 2026-10-09):

- **`test-backend-ops`:**
  - B60: MUL_MAT 1221/1221; MUL_MAT_ID 468/542, where the 74 failures are
    the pin's own MXFP4 cases;
  - A770: 542/542 and 1221/1221;
  - the IQ2_XS cases run on the card (22 MUL_MAT_ID, before: not
    supported), including Flash-Next's shape (2560 -> 640, 1-64 tokens);
  - the red case: the sign index off by one fails MUL_MAT and MUL_MAT_ID
    (ERR 1.0-1.4 against 5e-4);
  - the grouped kernels run only when switched on. With
    `GGML_OPENCL_KQ_GROUPED=1` the IQ2_XS cases pass 22/22 on both cards,
    and the mutant fails 10 of them. `GGML_OPENCL_KQ_IDG=1` aborted for
    every low-bit type; 0031 fixes that.
- **Flash-Next IQ3_XXS served on the B60:** 13,000 MiB of slots, 32,768
  context, MTP 2.
  - all 48 layers cached (38 before);
  - 7,814 slots on the card and a 27.43 GiB bank in host memory;
  - the capital, the 20k needle and the long answer right;
  - speed: 4.2-5.0 t/s on 300-token answers to three different prompts,
    against 3.7 t/s without the patch. The decode is bound by storage
    reads: 4,000-4,700 major faults and ~1 GB per answer from the ZFS pool
    where this file sits. IQ2_XS on the NVMe decodes ~33 t/s.
  - **Retracted:** "27.6-28.0 t/s from the second answer on". Those answers
    repeated one prompt, whose pages were already in the page cache; the
    "first answer slower because of the hit rate" reading goes with it.
  - xe evicts nothing: the process's GTT stays at the bank's 28.1 GB
    (fdinfo every 0.5 s).
- **KL as served** (16 x 512 against unsloth's Q8_0): 0.2180 nats, top-1
  89.90 %. The CPU arm gives 0.2211 / 89.73 %, within the error bars.

## 0031-opencl-idg-args.patch

Intel: the opt-in grouped MUL_MAT_ID matvec (`GGML_OPENCL_KQ_IDG=1`, 2-15
tokens, one read of an expert's weights for its tokens) aborted with
`CL_INVALID_ARG_INDEX` for every low-bit type. The per-pair kernel's last
arguments (the pair order, the next projection's prefetch) were also set on
the grouped kernel, which takes the route's tiles in their place. Found in
0030's review. The grouped branch now skips them and, like
`GGML_OPENCL_KQ_GROUPED`, drops an armed prefetch (it copies nothing for the
next projection). The default path is unchanged.

Its review found a second defect on that path: with an expert cache, a
9-15-token batch (a two-branch graph, ids -1 for the pairs the other branch
serves) left those pairs' rows unwritten in the slot projection. The router
drops a -1 pair; the per-pair kernel writes zeros for it, the grouped kernel
does not. The zero fill there was keyed on the call (no single branch)
instead of the tensor. It now covers every slot or bank tensor on the
grouped path, as it does for the GEMM. test-backend-ops cannot reach this:
it has no cache and no -1 ids.

Measured (`measured-here`, 2026-10-09):

- **KL as served**, Flash-Next IQ3_XXS on the B60 with 13,000 MiB of
  slots, 2 x 512 tokens in ubatches of 12 against unsloth's Q8_0:

  | arm | mean KL | top-1 |
  |---|---|---|
  | the default path, before | 0.218212 | 89.877 % |
  | `GGML_OPENCL_KQ_IDG=1`, before the fix | 6.378638 | 28.333 % |
  | `GGML_OPENCL_KQ_IDG=1`, with it | 0.217460 | 89.706 % |
  | the default path, after | 0.218212 | 89.877 % |

  The default path's values are the same before and after the fix.

`test-backend-ops -o MUL_MAT_ID`:

- **The red case:** before the patch, `GGML_OPENCL_KQ_IDG=1` aborts at the
  first IQ2_XXS case.
- **B60:** 468/542 with the switch, as without it; the 74 failures are the
  pin's own MXFP4 cases.
- **A770:** 542/542 both ways.

## 0032-opencl-bank-stream.patch

Intel, an expert cache's prefill, opt-in (`GGML_OPENCL_BANK_STREAM=1`).
A prefill ubatch reads each layer's bank experts, those not in VRAM slots,
from USM host memory. Until now a gather kernel copied the routed ones into
a card-side mirror on the compute queue, ahead of each projection's GEMM:
~0.83 s of a 2,048-token chunk's ~4 s of kernels on Flash-Next IQ2_XS, with
the copy engine idle (2 % busy). This patch copies each bank tensor whole on
a second in-order queue into a ring of VRAM slots, ahead of the GEMM that
reads it. It follows Strata's prefill ring (`src/prefill/prefill.cpp:71-104`)
and Infernix's expert stream: the copy engine works while the compute
engine runs the earlier projections.

- **The order:** the first prefill graph learns the order of its bank
  tensors and gathers as before. From the next graph on, each bank tensor's
  copy is queued `ring - 1` tensors ahead
  (`GGML_OPENCL_BANK_STREAM_RING`, default 4).
- **Events:** a GEMM waits for its copy; a copy waits for the GEMM that
  last read its slot.
- **Lookahead stays inside a graph.** Decode's swaps change the banks
  between graphs.
- **Not streamed:** IQ4_NL banks (scales and quants split) still gather.
- **The slots' layout is the bank's,** as the gather's mirror's was, so the
  GEMM reads them unchanged.
- **Red switch:** `GGML_OPENCL_BANK_STREAM_RED=1` fills each slot from the
  next bank tensor.

Measured (`measured-here`, 2026-10-10, B60, Flash-Next IQ2_XS,
13,000 MiB of slots):

- **Overlap first.** A standalone probe: a 2 GiB USM-host-to-card copy on a
  second queue (150 ms, 14.3 GB/s) ran entirely under a compute-bound or
  memory-bound kernel on the first queue. The B70 recipe's trap (no overlap
  within one process on Level Zero) does not apply on this OpenCL path.
- **Served needle prefill, 20,045 tokens** (MTP 2, 32,768 ctx, one build):

  | | prefill | compute engine | copy engine | answer |
  |---|---|---|---|---|
  | off | 416.1 t/s | 96 % | 2 % | right |
  | on | 497.6 t/s | 95 % | 88 % | right |

- **KL as served** (16 x 512 against unsloth's Q8_0, 141 bank tensors
  streamed, the copy engine up to 94 % busy): 0.294050 / 87.083 %, the same
  as off.
- **The red case** aborts the run at the next synchronisation; it never
  passes.
- **VRAM:** a ring of 4 takes 4 x 187.7 MiB = 751 MiB. That is why it is
  opt-in until the served configurations are checked for eviction.
- **Review** (an outside model, after the first push; fixed before the
  measurements below):
  - A slot now records the order index its last copy holds, and an
    "issued ahead" tensor is read from its slot only if the slot still
    holds it. Otherwise it gathers. The earlier check accepted any index
    issued in the graph, which would have read another bank after a
    deviation from the learnt order.
  - The per-graph state resets on every graph (the backend's graph
    counter), not only when the order's first tensor comes round.
  - Each graph's copies wait for a marker of the main queue's work before
    the graph. Decode's swaps rewrite bank entries on that queue, and the
    copies' ordering against them had rested on host waits made for other
    reasons.
  - A tensor larger than the ring's slots gathers instead of being cut.
  - A USM base that changed (a model reloaded in-process) relearns the
    order.
  - The main queue is flushed after the copies are issued.
  - Not changed: the barrier still precedes the route and convert kernels,
    not only the GEMM.
- **After the fixes** (`measured-here`, 2026-10-10):
  - KL with the stream on: 0.294050 / 87.083 %, unchanged; the red case
    aborts.
  - The deployed unit's configuration (9,500 MiB of slots, 131,072 ctx,
    MTP 2), served needle: off 406.6 t/s, ring 4 **494.9 t/s (+21.7 %)**,
    ring 2 392.9 t/s (one tensor of lookahead is not enough). Answers
    right in every arm.
  - Nothing evicted: GTT stays at the bank's 24,482 MiB. With a ring of 4,
    VRAM peaks at 24,243 MiB against 23,448 off.
  - These served arms ran inside the host's weekly backup window (loads of
    ~15 min). Their decode figures (25.2 off, 22.7 / 23.0 on) are void
    until re-measured outside it.
- **A second review** (of the fixes): sound, with three more changes.
  - A graph entering the learnt order at index i issues from i, not from 0.
  - Freeing a USM bank in the order drains the copy queue and drops the
    learnt state (the in-process reload case).
  - `synchronize` drains the copy queue too: a copy a GEMM never consumed
    still reads host memory.
  - KL afterwards: 0.294050 / 87.083 %, unchanged.
- **With a larger expert-GEMM tile** (`GGML_OPENCL_KQ_MM_ID_TILE`;
  `measured-here`, 2026-10-11, served needle, stream on, 13,000 MiB,
  32,768 ctx). The B60's default `2,2,1,1` dequantises a 16-row block for
  every 32 tokens; a larger tile shares the dequant across more tokens and
  rows:

  | tile (rows x tokens, sub-groups) | prefill | decode |
  |---|---|---|
  | 2,2,1,1 (16 x 32, 1), the default | 494.7 | 30.7 |
  | 4,4,2,2 (64 x 128, 4) | 600.5 | 31.3 |
  | 4,4,2,4 (64 x 256, 8) | **634.1** | 30.4 |
  | 4,4,4,2 (128 x 128, 8) | 589.5 | 29.0 |
  | 4,8,2,2 (64 x 256, 4) | 475.7 | 31.3 |

  - Answers right in every arm.
  - KL with the stream and 4,4,2,4: 0.294050 / 87.083 %, unchanged.
  - 0.7.2 prefills the needle at ~416-422 t/s; the stream and this tile at
    634.1, above Strata's own engine on this card (620.5).

## 0033-opencl-b60-mm-id-tile.patch

Intel, sub-group-16 devices (the B60): MUL_MAT_ID's GEMM tile defaults to
`4,4,2,4` (RG, TG, NSGM, NSGN: 64 rows x 256 tokens, 8 sub-groups) instead
of `2,2,1,1` (16 rows x 32 tokens, one sub-group). The tile kernel
dequantises its weight block once per token tile. With the old shape an
expert of Flash-Next, at ~40 tokens per 2,048-token ubatch, paid one full
and one nearly empty tile; a larger tile shares each dequantised block
across more tokens and rows. The SG 8 tile (A770) is unchanged.
`GGML_OPENCL_KQ_MM_ID_TILE` still overrides it.

Measured (`measured-here`, 2026-10-11, B60):

- **The served needle** (Flash-Next IQ2_XS, 13,000 MiB, 32,768 ctx,
  0032's stream on), by tile; the table is under 0032.
  - With this default and no override: 647.7 t/s, answers right.
  - KL with 0032's stream and this tile: 0.294050 / 87.083 %, unchanged.
- **The deployed configuration** (9,500 MiB, 131,072 ctx, MTP 2), two
  processes each:
  - prefill 408.3 / 408.2 (0.7.2's behaviour) -> 639.8 / 644.5 with
    0032's stream and this tile;
  - decode equal: 26.6-29.1 against 26.7-28.6, one baseline outlier 16.0;
  - nothing evicted; VRAM peaks at 24.2-24.3 GB (0032's ring) against 23.4.
- **The coder on the B60** (its IQ3_XXS / IQ4_XS experts take the
  tokens-as-a kernel, which does not read this tile): pp4096 1,905.5 ->
  1,906.0.
- **test-backend-ops:** MUL_MAT_ID 468/542 (the pin's 74 MXFP4 cases),
  MUL_MAT 1221/1221.
- **Review:** sound. Local memory 32 KiB (the plain tile's), 128 work-items,
  the route tiles' buffer bounded for any tile, rows clamped; the
  tokens-as-a path does not read the tile.

## 0034-opencl-iq-swar.patch

The low-bit expert kernels (`mul_mv_iq_q8_1.cl`, and the expand helpers
shared with `mul_mm_kq_f16.cl`) do their per-byte weight arithmetic on
32-bit words instead of on `char4` vectors:

- **The sign step** (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S): four grid
  bytes negated under a sign mask, `(g ^ m) + (m & 0x01010101)` in place of
  `char4(g ^ m) - char4(m)`.
  - Exact because no grid byte is 0 (IQ2 grids: 8, 25, 43; IQ3_XXS: 4-62;
    IQ3_S: odd 1-15), so `~g + 1` never carries out of a byte.
- **IQ1_M's ±delta** and **Q2_0's -1 offset**: the same trick with a
  0x80808080 bias, so a byte's borrow cannot cross into the next byte.

Why: in the B60's ISA (`ocloc compile -device bmg`, disassembled), IGC
lowers `char4` arithmetic to one byte at a time. It moves each byte into
its own register, widens it, computes, and packs the result back. Those
moves were more than half of the matvecs' instructions.
`-DLB_CHAR4_ARITH` (through `GGML_OPENCL_IQ_OPTS`) keeps the `char4` form
for measurement.

Measured (`measured-here`, 2026-10-11):

- **ISA, B60** (total instructions / movs per kernel, before -> after):

  | kernel | total | movs |
  |---|---|---|
  | IQ2_XXS | 1,405 -> 1,052 | 526 -> 206 |
  | IQ2_S | 1,585 -> 1,230 | 541 -> 221 |
  | IQ1_M | 1,355 -> 1,138 | 408 -> 222 |
  | Q2_0 | 1,291 -> 1,069 | 384 -> 194 |

  The dp4a count is unchanged (32 in each).
- **B60 kernels** (`test-backend-ops perf`, MUL_MAT_ID at Flash-Next's
  shapes, one token):

  | kernel | before | after |
  |---|---|---|
  | IQ2_XXS | 40.7 us | 25.0 us |
  | IQ2_S | 50.6 us | 31.1 us |
  | Q2_0 gate/up | 33.4 us | 21.7 us |
  | Q2_0 down | 32.1 us | 24.9 us |

  Three tokens, Q2_0 down: 89.9 -> 63.5 us.
- **A770 kernels:** within ±7 %. IQ2_S and Q2_0 are 4-11 % faster; IQ2_XXS
  at three tokens is 6 % slower (73.6 against 69.2 us).
- **Served Flash-Next** (IQ2_XS, B60, the unit's configuration, two
  processes each way): decode 31.7-33.5 against 30.7-31.9 t/s; prefill
  equal (637-650 against 639-651). The answers are right (capital, needle).
- **The coder** (IQ3_XXS / IQ4_XS experts, A770): served decode 42.1
  against 42.1 t/s, two processes each way. IQ3_XXS kernels are equal
  (237.7 against 239.0 us at one token); IQ4_XS is 1-4 % faster.
- **KL:** 0.294050 / 87.083 %, unchanged.
- **test-backend-ops:**
  - B60: MUL_MAT_ID 468/542 (the pin's 74 MXFP4 cases), MUL_MAT 1221/1221;
  - A770: 542/542 and 1221/1221.
- **Review:** the word forms are bit-exact against the `char4` forms over
  every byte value each one is used on, checked exhaustively.

## 0035-opencl-iq4-local-codebook.patch

The IQ4 codebook lookup (`kv4`, used by the IQ4_XS and IQ4_NL matvecs:
dense, MUL_MAT_ID and the grouped kernel) reads four tables from local
memory and ORs the results:

- **The tables:** the 16 codebook values, once per byte position, each
  already shifted into its byte (64 words). `LOCAL_TABLES` copies them per
  work-group, as it does IQ3_XXS's grid.
- **The lookup:** a word of four weights is `t[n0] | t[16 + n1] | t[32 + n2]
  | t[48 + n3]`.
- **On both cards.** It replaces the constant-memory byte lookups
  assembled as a `char4` (the B60) and the register selects (the A770,
  `IQ4_KV_SELECT`).
- **Switches** (`GGML_OPENCL_IQ_OPTS`, for measurement): `-DIQ4_KV_CHAR4`
  (the old form), `-DIQ4_KV_SHIFTED` (the same tables in constant memory),
  `-DIQ4_KV_SELECT`, and `-DIQ4_LT_RED` (the test's red case: two entries
  swapped).

Why (`measured-here`, B60):

- **ALU-bound, not memory-bound.** The dense IQ4_XS matvec was bound by
  the integer pipe, like the low-bit kernels (stage 6):

  | kernel | XVE active | ALU1 utilisation | memory (SBID) stall |
  |---|---|---|---|
  | IQ4_XS | 68 % | 62 % | 35 % |
  | Q4_K | 52 % | 41 % | 53 % |

  At 4096×14336 (larger than L2), the same bytes per weight within 6 %,
  IQ4_XS streamed 170-237 GB/s and Q4_K 415-426.
- **The ISA** (`ocloc`, `bmg`) showed two costs:
  - each constant lookup widened its index to a 64-bit address;
  - IGC assembled the `char4` one byte at a time. A shift/OR form compiled
    to the same code.
- **Instruction counts:**
  - pre-shifted tables in constant memory: dense kernel 761 -> 693
    instructions; slower, because the 64-bit gathers remain;
  - in local memory: 761 -> 630, movs 267 -> 86.
- **Stage 3c's local-memory codebook was a different build.** It measured
  slower (59.5 against 44.2 us), with a 16-byte copy per lane.

Measured (`measured-here`, 2026-10-11):

- **B60 kernels** (test-backend-ops perf, `char4` -> local):

  | case | 1 column | 3 columns |
  |---|---|---|
  | dense IQ4_XS, 4096×14336 | 131.1 -> 92.3 us | 199.6 -> 155.9 us |
  | dense IQ4_XS, 6144×2560 | 43.4 -> 24.9 us | 49.7 -> 35.7 us |
  | dense IQ4_XS, 2560×6144 | 35.5 -> 21.8 us | 41.5 -> 30.6 us |
  | IQ4_NL, 6144×2560 | 37.7 -> 21.6 us | |

  MUL_MAT_ID IQ4_XS: 36.7 -> 22.1 us.
- **A770 kernels** (selects -> local):
  - dense 4096×14336: 115.2 -> 91.2 us;
  - dense 6144×2560: 29.8 -> 21.9 us;
  - MUL_MAT_ID: 26.4 -> 20.9 us;
  - three columns: mixed (34.9 -> 36.6, 43.5 -> 36.7 us).
- **Served Flash-Next, device time** (B60, the unit's configuration, a
  profiling build, the same session, local against `char4`):
  - the dense IQ4_XS calls are 12-20 % shorter (the 10240×2560 QKV
    projection 78.2 -> 62.9 us, the output head 1,772 -> 1,401 us);
  - per token: dense matvecs 11.13 -> 10.35 ms, all kernels 26.89 -> 26.09
    ms;
  - the same answers and draft acceptance.
- **Served Flash-Next, wall** (B60): inside the run-to-run spread.
  - Plain build, 13,000 MiB, two processes each way: 32.3-33.4 against
    33.0-34.4 t/s.
  - Profiling build: 28.6 / 29.0 against 28.1 / 28.7 t/s.
- **The coder on the A770** (IQ4_XS experts, MTP 4), two processes each
  way: select 41.0-42.1 t/s, local 41.9-42.9; with the host's selects
  removed (the patch as adopted) 42.0-43.0. Same answers.
- **test-backend-ops:**
  - B60: MUL_MAT 1221/1221, MUL_MAT_ID 468/542 (the pin's 74 MXFP4
    cases);
  - A770: 1221/1221 and 542/542;
  - the red case (`-DIQ4_LT_RED`, no other switch) fails 6-7 of 14 IQ4
    MUL_MAT cases and 1-2 of 3-4 MUL_MAT_ID cases on both cards.
- **Exactness:** the same int8 values reach the same dot products, so the
  outputs are bit-identical to the old forms. KL is unaffected; the
  perplexity runs' 512-token batches take the GEMM, which this patch does
  not touch.
- **Review:** no defects. One nit fixed: `-DIQ4_KV_CHAR4` now cancels the
  local table.

## 0036-opencl-get-rows-q8-0.patch

OpenCL GET_ROWS takes q8_0 sources. Before it, GET_ROWS took f32, f16, i32
and plain Q4_0 only (`ggml_opencl_supports_op`), so every gather from a
q8_0 tensor ran on the CPU.

- **What it hit:** Flash-Next with `--llama-kv q8_0`. The KV type also types
  the QSA indexer's key cache, and the indexer gathers rows of it
  (`gather_key_gate`, `gather_pooled`, `src/models/qwen4exp.cpp`). The
  scheduler copied each layer's 8 MB indexer cache to the CPU, gathered
  there and copied back, in the target graph and in every MTP draft.
- **Two layouts:**
  - A quantized KV or indexer cache is written by kernels (SET_ROWS), so
    it keeps ggml's plain 34-byte blocks: `kernel_get_rows_q8_0`. arcint
    never writes it from the host: its only state restore, a lane's
    checkpoint, passes `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, which restores
    the f32 recurrent state and skips the attention and indexer caches
    (`code`: `src/exec/backend_llama.cpp`, `llama_memory_hybrid_idx::state_read`).
    A full-state restore (llama-server's slot load) would go through
    `set_tensor`, which flattens a whole contiguous q8_0 tensor and has no
    guard for a partial write; that path is pre-existing and not used here.
  - A q8_0 tensor uploaded through `set_tensor` is flattened into a quant
    and a scale buffer: `kernel_get_rows_q8_0_flat`.
  - The host tells them apart by the tensor's extra: the buffer context
    lists the flattened extras. A view keeps the extra it was created with,
    so the parent's (`view_src`) decides.
  - One work-item per output element.
- **The test** `GET_ROWS(type=q8_0|f16, n, m, r)` (`test_get_rows_cache`):
  SET_ROWS fills a never-uploaded cache with zeros, then writes rows into
  it, and GET_ROWS reads them back. That covers the plain layout, which
  `set_tensor`-initialized tests cannot reach. `-DGET_ROWS_Q8_RED` (the
  next block's scale) is the red case of both kernels.

Measured (`measured-here`, 2026-10-13):

- **test-backend-ops GET_ROWS:** 41/41 on the B60 and on the A770. The red
  case fails both cache cases (ERR 0.070 and 0.068 against limits of 9.5e-7 and 2.7e-7)
  and the four flattened q8_0 cases (31/41).
- **The scheduler** (`GGML_SCHED_DEBUG=2`, Flash-Next IQ2_XS, 32K): with
  q8_0 KV, an 80-token answer had 222 of 444 splits on the CPU, against 52
  of 104 with f16 KV. With the patch the same pair of served arms had 481
  and 493 CPU splits of 962 and 986 over four requests: q8_0 no longer adds
  any. (In the f16 dump the CPU splits are the token-embedding gathers, whose
  tables live in host memory.)
- **Served Flash-Next** (B60, 131,072 ctx, MTP 2, the bank stream; each
  arm's freed VRAM given to the expert slots):

  | KV | expert budget | needle prefill | decode | hit rate |
  |---|---|---|---|---|
  | f16 | 9,500 MiB | 649.8 | 30.0 / 31.3 | 89.3 % |
  | q8_0, before the patch | 11,300 MiB | 521.5 | 13.3 / 13.4 | 91.5 % |
  | q8_0, with it | 11,300 MiB | 617.2 | 30.9 / 31.0 | 91.4 % |

  The capital, the 20k needle and both long answers right in every arm.
- **KL** (16 x 512 against unsloth's Q8_0, 13,000 MiB): f16 KV 0.294050 /
  87.08 %, q8_0 KV 0.294037 / 87.35 %, inside the answer-level bar. These
  512-token chunks do not reach the indexer gather: the red kernel gives
  the same numbers. The served needle did not catch the red kernel either
  (the indexer's top-k selection survives a perturbed scale), hence the
  test above.

## 0037-opencl-fa-dpas-f16kv.patch

A quantized prompt's K/V go to f16 in blocks of keys, and the f16 DPAS
flash-attention kernel runs on each block (Intel Xe2, sub-group 16).

- **What it fixes:** the quantized builds of `flash_attn_dpas.cl` (0008)
  dequantize K/V in every work-group, so each value is converted once per
  query head and query block, 24 times in a 512-row chunk of the dense 27B.
  The byte and word unpacking adds 527 instructions to a tile's main loop
  (1,893 against the f16 build's 1,366; offline ISA, `ocloc -device bmg`).
  Served at 58k tokens with q8_0:q4_0 KV, attention took 75.2 s against
  f16's 56.3 s.
- **The reference:** llama.cpp's CUDA backend converts a quantized K/V to
  f16 once a call (`launch_fattn`, `need_f16_K/V`) and runs the f16 kernel.
  **Deviation:** CUDA converts the whole view. At the dense agent's 245,760
  tokens that is 960 MiB, more than the unit's margin, so here the
  conversion runs a block of keys at a time (16,384 keys: two 32 MiB
  scratch buffers, plus 14 MB of carried state at 512 rows).
- **The carry:** the f16 kernel built with `FA_CARRY` takes `kv0`, offsets
  the mask by it, and keeps its online-softmax state between blocks. The
  state is O, m and l per lane in f32, stored in the registers' own layout,
  so l keeps its per-lane partial sums as in one pass. Loading O row by row
  spilled 1,408 bytes, and IGC then recompiled the kernel with rolled loops,
  4x slower; vector loads straight into the accumulators do not spill.
- **The conversion:** `flash_attn_kv_to_f16`, a work-item per 8 values,
  using the kernel's own `load8`, so the f16 values equal what the
  quantized build stages.
- **Switches:**
  - `GGML_OPENCL_FA_DPAS_F16KV=0` keeps the quantized kernels;
  - `GGML_OPENCL_FA_DPAS_F16KV_BLOCK` sets the keys a block;
  - `-DFA_CARRY_RED` (through `GGML_OPENCL_FA_DPAS_OPTS`) drops the carried
    sums: the red case.
  - The path sits under the quantized DPAS kernels' gate, so
    `GGML_OPENCL_FA_DPAS_Q=0` turns it off too.
- **Also in the patch:** a test-backend-ops perf case, prompt attention at
  depth for the dense 27B's geometry, f16 and q8_0/q4_0.

Measured (`measured-here`, 2026-10-13):

- **test-backend-ops FLASH_ATTN_EXT, head size 256:**
  - 260/261 on the B60 and the A770, the same three failures as without the
    patch;
  - 260/261 with 64-key blocks, so the carry runs;
  - the red case fails 18 quantized prompt cases.
- **Perf** (B60, 512 rows, 24 query heads on 4): q8_0:q4_0 at 8,192 keys
  14.6 -> 12.3 ms, at 65,536 keys 111.9 -> 92.5 ms (f16: 8.3 and 63.0). The
  test's quantized arms include a per-call unflattening of the uploaded K/V
  (~2.8 ms at 8k) that a served cache does not have.
- **Served** (the dense agent's model and flags, q8_0:q4_0, n_ctx 131,072, a
  58,269-token prompt, the profiling build): prefill 425.9 -> 479.1 t/s;
  attention 75.2 s -> 57.7 s, plus 2.2 s of conversion. The f16-KV arm runs
  498.8 t/s.
- **Gate** (the agent's flags at 245,760): the dev arcint with 0001-0037 on the B60, the dense
  agent's flags (q8_0:q4_0, n_ctx 245,760, MTP 5):
  - **Answers:** the capital right; the needle at 40 % of a 231,786-token
    prompt right (prefill 192.0 t/s, peak 24,275 MiB, GTT 28 MiB); the
    acceptance task 10/10 greedy.
  - **A/B at 132,874 tokens**, the same window, the needle right in both:
    prefill 249.0 t/s without the path, 292.3 t/s with it (+17 %); peak
    24,144 / 24,224 MiB.
  - **KL** (16 x 512 against `dense-ref.kld`): 0.005950 / 97.304 % without
    the path, with it, and with 64-key blocks. With 64-key blocks and the
    carried sums dropped (the red case), KL is 4.4957 / 36.7 %. So the path
    and the carry run, and with them KL is identical to six digits.

## 0038-opencl-fa-dpas2.patch

A second prompt flash-attention kernel for Intel Xe2, `flash_attn_dpas2.cl`,
after vLLM's head-256 policy and sycl-tla's mainloop. It runs f16 K/V in
place and 0037's f16 blocks, and adds a mask pre-pass.

- **The reference:**
  - vllm-xpu-kernels `csrc/xpu/attn/fmha_utils.hpp`, `chunk_policy_head256`
    (`code`): 8 query rows a sub-group with all output columns, 32
    sub-groups a work-group, 32-key tiles;
  - sycl-tla `xe_fmha_fwd_mainloop.hpp` (`code`): K and V straight from
    global memory into registers by 2D block reads, the next tile
    prefetched, no local memory, Q re-read each tile.
- **What changes against 0008's kernel:**
  - K arrives as the b operand of Q K^T by a transposed 2D read, V as the
    b operand of P V by a VNNI-transforming one. The local-memory staging,
    the scalar 16-bit transposing stores of V and the barriers are gone.
  - The main loop runs 972 instructions per 64 DPAS instead of 1,366
    (offline ISA, `ocloc -device bmg`).
  - Q goes to f16 once a call (`flash_attn_q_to_f16`, scaled) and is read
    by 2D reads; holding it in registers, as 0008 does, left no room at head
    size 256 (O alone is 128 GRF).
- **IGC's value-returning builtins** (`__builtin_IB_subgroup_block_read_flat_*`)
  do the reads. The extension's functions write a private array, and in this
  kernel that put the accumulators in scratch memory without a spill being
  reported.
- **The mask by blocks** (after llama.cpp's Vulkan
  `flash_attn_mask_opt.comp`): `flash_attn_mask_cls` writes a byte per 8
  rows x 32 keys, mixed / all zero / nothing visible, once a call. All-zero
  blocks skip the mask reads; blocks with nothing visible are skipped by
  that sub-group, which v2 can do because its loop has no barrier.
- **Switches:**
  - `GGML_OPENCL_FA_DPAS2=0` keeps 0008's kernel;
  - `GGML_OPENCL_FA_DPAS2_MCLS=0` turns the mask classes off;
  - `GGML_OPENCL_FA_DPAS2_NSG` sets the sub-groups a work-group (32 or 16);
  - the red cases are `-DFA2_RED` (keys one row off), `-DFA2_MCLS_RED`
    (mixed blocks taken as all zero) and `-DFA_CARRY_RED`.
  - The f16 path needs 64-byte aligned heads and a row pitch of a multiple
    of 16 bytes; otherwise the old kernel runs.
- **Also in the patch:**
  - `test_flash_attn_ext_causal`: a causal mask as llama.cpp builds it, so
    all three block classes occur; correctness cases at 512 and 300 rows,
    and perf cases at depth;
  - comment fixes from 0037's review.

Measured (`measured-here`, 2026-10-13, B60):

- **test-backend-ops FLASH_ATTN_EXT, head sizes 256 and 128:**
  - 864/865 with the classes, without them, and with 64-key blocks, the
    same three failures as the old kernel; the A770 is unchanged (sub-group
    8 keeps 0008's kernel);
  - red cases: keys one row off fails 90 prompt cases; mixed blocks as all
    zero fails 62, 4 of them causal.
- **Perf** (512 rows, 24 query heads on 4, a causal mask):

  | shape | 0008 | v2 without classes | v2 |
  |---|---|---|---|
  | f16, 8,192 keys | 8.21 ms | 5.43 ms | 4.02 ms |
  | f16, 65,536 keys | 63.0 ms (13.1 TFLOPS) | 43.1 ms | 30.6 ms (26.9 TFLOPS) |
  | q8_0:q4_0, 65,536 keys (0037) | 92.2 ms | 68.4 ms | 56.5 ms |

- **Served without the classes** (the dense agent's model and flags,
  q8_0:q4_0):
  - prefill at 58,269 tokens 479.1 -> 568.3 t/s (n_ctx 131,072);
  - at 132,874 tokens 292.3 -> 375.0 t/s (n_ctx 245,760);
  - at 231,786 tokens 192.0 -> 257.4 t/s;
  - the needle right in every arm, the acceptance task 10/10;
  - KL 0.005950 / 97.304 % as before (the keys-off red case 0.6815).
- **Served with the classes** (the same window, n_ctx 245,760 unless stated):
  - prefill at 58,269 tokens 641.2 t/s (n_ctx 131,072; attention 26.2 s,
    30 % of device time, level with the Q4_K GEMM);
  - at 132,874 tokens 383.4 -> 451.6 t/s with the classes on;
  - at 231,786 tokens 324.2 t/s;
  - the capital, both needles and the acceptance task (10/10) right;
    decode unchanged (16.2 t/s at 128k); peak 24,283 MiB, GTT flat;
  - KL 0.005950 / 97.304 % with the classes off, on, and with 64-key blocks;
    the red case (mixed blocks taken as all zero) 0.4797 / 77.7 %.
  - Against arcint 0.7.3 (0001-0035) on the same prompts: 425.9 -> 641.2 t/s
    at 58k, 249.0 -> 451.6 at 133k, 157.2 (the 0.7.3 record at 236,938) ->
    324.2 at 232k.
  - These served arms ran the tree before the review. The review added the
    pitch >= width term to the surface check (no real view fails it) and the
    all-zero-mask test, which passes on both cards (866/867, same three
    failures; the keys-off red case fails it).
- **Reviewed** before the push by a second model: no defects. It checked the
  2D-read argument conventions against sycl-tla's builtin wrappers, the
  class-map indexing in the carry case, the host's argument order for both
  kernels, and that nothing reaches the A770 or masks with ALiBi, softcap or
  sinks.

## 0039-opencl-fa-sg8-q-global.patch

The sub-group-8 prompt flash attention (Intel Xe-HPG, the A770) keeps Q out
of local memory, so a work-group can take 16 sub-groups instead of 8.

- **What it fixes:** 0008's sub-group-8 build staged Q in local memory next
  to the K and V tiles: 58 KB of the core's 64 (the build's `slm_size`,
  offline ISA). A core therefore ran one 8-thread work-group, half a thread
  per EU, with nothing to hide latency. Served on the coder (44k tokens),
  attention was 63 % of prefill device time at ~2.9 TFLOPS.
- **Why not 0038's design here:** the A770 has no 2D block I/O, and with
  head size 256 a sub-group cannot keep enough rows in registers to reuse
  K/V reads (O alone is 128 GRF). So the local-memory K/V sharing stays and
  more sub-groups share it.
- **`FA_QG`:** Q goes to f16 once a call (`flash_attn_q_to_f16_blk`), in
  the a-operand blocks the kernel used to write into local memory, and is
  read by `intel_sub_group_block_read8`. Local memory falls to 45 KB (one
  work-group a core), now with 16 sub-groups: 16 threads.
- **Switches:** default on for sub-group 8.
  - `GGML_OPENCL_FA_SG8_QG=0` restores the old build;
  - `GGML_OPENCL_FA_SG8_NSG` sets 8, 16 or 32;
  - `-DFA_QG_RED` (through `GGML_OPENCL_FA_DPAS_OPTS`) reads Q one chunk
    off: the red case.
  - Sub-group 16 (Xe2) is unchanged.

Measured (`measured-here`, 2026-10-13, the A770 at its pinned 2000 MHz):

- **test-backend-ops FLASH_ATTN_EXT, head sizes 256 and 128:** 866/867 at
  8, 16 and 32 sub-groups, the same three failures as before; the red case
  fails 92 prompt cases.
- **Perf** (512 rows, causal mask, 24 query heads on 4):

  | shape | before | Q global, 8 | 16 | 32 |
  |---|---|---|---|---|
  | q8_0:q4_0, 32,768 keys | 142.5 ms | 144.2 | 70.8 | 73.2 |
  | f16, 32,768 keys | 108.5 ms | 108.3 | 83.0 | 66.7 |

  Q out of local memory alone changes nothing; the work-group size does,
  so occupancy was the limit. 16 is the best quantized arm, the coder's
  case.
- **Served:** the coder's model and unit flags (q8_0:q4_0, n_ctx 98,304,
  MTP 4).
  - Speed: prefill at 44,329 tokens 476.6 -> 685.5 t/s (attention 55.6 ->
    27.1 s); at 56,800 tokens 410.1 -> 616.2 t/s in one window; decode
    unchanged (36 t/s).
  - Answers: the needle right at 56,800 in both arms and at 88,853, the
    capital right.
  - KL (16 x 512 against `coder-ref-q8.kld`): 0.026908 / 92.402 % in both
    arms; the red case 1.1127 / 56.3 %.
  - The acceptance task: 9/10 greedy in both arms, with the identical
    answer. The deployed 0.7.3 package scores the same 9/10 with the same
    answer today, so neither this patch nor 0036-0038 moved it.
- **Reviewed** before the push by a second model: no defects in this build.
  It checked the layout, padding and work-group geometry, that 0037's
  carry path cannot be reached on sub-group 8, and that the Xe2 path is
  unchanged. Two robustness points go into the next patch to these files:
  the kernel's Q base ignores `q_offset` (the host passes 0), and a failure
  to create the conversion kernel would leave the FA_QG kernel in use with
  f32 Q.

## 0040-opencl-fa-sg8-load8-words.patch

The sub-group-8 prompt flash attention (the A770) turns q8_0 / q4_0 codes
into halves with word operations, and takes 0039's review points.

- **`load8` for sub-group 8:**
  - the codes become halves 1024 + u by integer operations on whole words
    (0006's form): four 2-byte loads (the 34- and 18-byte blocks are 2-byte
    aligned), `^ 0x80808080` for q8_0, the nibble mask for q4_0, 0x6400 as
    the exponent;
  - the value is (x - 1152) or (x - 1032), an exact integer, times the
    scale, rounded once as before: identical by construction (`code`; the
    tests judge within a tolerance, so they do not show bit identity);
  - the main loop's byte-typed operations go 282 -> 10 and its movs 797 ->
    607 (offline ISA);
  - `-DFA_LOAD8_BYTES` keeps the byte conversion.
- **From 0039's review:** the global-Q base adds `q_offset`, and a failure to
  create the conversion kernel drops the global-Q build instead of running
  it on f32 Q.

Measured (`measured-here`, 2026-10-13, the A770):

- **test-backend-ops FLASH_ATTN_EXT, head sizes 256 and 128:** 866/867 on
  the A770, the same with the byte conversion forced, and on the B60.
- **Perf** (q8_0:q4_0, 512 rows, causal): 8,192 keys 16.29 ms against the
  byte form's 16.34-16.37; 32,768 keys 69.9-70.2 against 70.7-70.9. That is
  ~1 %: at 16 sub-groups the staging is load-bound, not conversion-bound
  (`dense-prefill-attention.md`, stage 3).
- **Reviewed** before the push by a second model: no defects. It checked the
  bit manipulation against ggml's block layouts, the bit-identity argument
  (no denormal flushing in the build options, no contraction of
  `(x - c) * d`) and the host's release paths. Its one optional note: the
  conversion kernel stays allocated when the main kernel fails its size
  check (harmless, never dispatched).

## 0041-opencl-fa-dpas2-q-pairs.patch

`flash_attn_dpas2.cl` (0038) reads Q two 16-dim chunks at a time, each of
S's two DPAS chains keeping its order.

- **Why:** an ablation of the kernel (B60, f16, causal, 65,536 keys, 512
  rows; each run removes one part) put the 32 small 2D reads of Q a tile at
  ~9 of 31 ms, more than K's and V's reads together:

  | removed | time |
  |---|---|
  | nothing | 30.8 ms |
  | K's reads | 30.2 |
  | V's reads | 28.7 |
  | Q's reads | 21.9 |
  | K's and V's reads | 19.3 |
  | the prefetch | 49.4 (it carries the kernel) |

- **What:** one `u16 m8k16v2` read brings two adjacent chunks (8 rows x 32
  halves, as two 16-wide blocks): 8 reads a tile at head size 256 instead
  of 32. The first chain still sums chunks 0..7 in order and the second
  8..15, so the results are bit-identical. `-DFA2_Q1` keeps the one-chunk
  reads.
- **What lost on the way:**
  - *Interleaving the chains* (even chunks into one, odd into the other)
    kept KL (0.005748 against 0.005950). But it flipped the dense agent's
    greedy acceptance answer to a different, wrong program: 2/10 three
    times out of three, against 10/10. A sum reordered inside the bar can
    still move a greedy answer; the task gate caught it.
  - *Holding part of Q in registers across tiles* spilled from four chunk
    pairs on (offline ISA).

Measured (`measured-here`, 2026-10-13, B60):

- **test-backend-ops FLASH_ATTN_EXT, head sizes 256 and 128:** 866/867, the
  same three failures; the keys-off red case fails 94; the A770 is
  unchanged (sub-group 8 does not use this kernel).
- **Identity** (dense 27B, q8_0:q4_0): KL 0.005950 / 97.304 % in both
  forms; the acceptance answer is the same file in both (md5 `c76c468a`,
  10/10).
- **Perf** (512 rows, causal):

  | shape | one-chunk reads | 0041 |
  |---|---|---|
  | f16, 8,192 keys | 4.02 ms | 3.72 |
  | f16, 65,536 keys | 30.46 ms | 28.41 |
  | q8_0:q4_0, 8,192 keys | 7.55 ms | 7.26 |
  | q8_0:q4_0, 65,536 keys | 56.55 ms | 53.78 |

- **Served** (the agent's flags at 245,760): the 132,874-token needle
  prefill went 452.0 -> 464.2 t/s with the interleaved form. The ordered
  form runs the same reads, with the chains in the old order.
- **Reviewed** before the push by a second model, both forms. Layout: the
  two-block read gives chunks 2j and 2j+1 in the one-chunk operand layout,
  checked against sycl-tla's builtin wrappers. Order: each chain's
  accumulation sequence equals the one-chunk form's for head sizes 256 and
  128, and the coordinates stay inside the surface. No defects.

## 0042-expert-cache-nb-admission.patch

The expert cache's swaps (0021) leave the compute queue: Strata's adapt
stream (`code`, Strata `src/program/generate.cpp:122-158`, `:4396-4480`).

- **Why:** an adapt exchanged its experts between slots and bank entries on
  the card's in-order queue, ahead of the next graph. The copies (a swap
  moves an expert's ~1.4 MB each way over the link; 1,800-3,100 swaps a
  300-token answer) held up the next verify: its input writes waited 2.5-6
  ms against 0.93-1.0 ms with no swaps (`measured-here`, Flash-Next, B60;
  `LLAMA_EXPERT_CACHE_MAX_SWAPS=0` as the discriminator).
- **What:**
  - Each layer's bank tensors get spare entries after the zero entry
    (`LLAMA_EXPERT_CACHE_SPARES`, 4 by default, ~263 MB more pinned host
    memory for Flash-Next). Strata reserves 96 exchange buffers over all
    layers (`expert_source.cpp:1332-1357`); here they sit in each layer's
    bank, so a swap beyond its layer's spares waits for a later adapt, as
    Strata's beyond its buffers' room do.
  - A swap runs in two steps on a second in-order queue, polled between
    ubatches without a wait: the victim's bytes are copied out of its slot
    into a spare entry; once that has landed the victim is served from the
    bank and the newcomer is copied into the freed slot; once that has
    landed the newcomer is served from the slot and its old bank entry
    becomes a spare. No adapt is planned while a swap is in flight
    (Strata, `generate.cpp:4431`).
  - The tables change only between graphs; a fence (a marker on the compute
    queue, a barrier on the adapt queue) placed after each table upload
    keeps a copy from writing a slice an earlier graph still reads.
  - The copies are enqueued right after the owner's graph (`after_graph`),
    not before the next one: the copy engine also writes a graph's inputs,
    and the first build, which enqueued them in `step()`, left the verify's
    input wait where it was (2.4-5.9 ms) and lost ~1.5 % (below).
  - The first landed swap is checked against the file; the red case
    (`LLAMA_EXPERT_CACHE_MUTATE=1`, one landed slot corrupted) reports 1 of
    6. `LLAMA_EXPERT_CACHE_NB=0` restores the exchange. A layer whose bank
    would pass the GEMM's 512 matrices gets fewer spares; swaps put off for
    want of a spare are logged at teardown.
- **Overlap, the mechanism checked** (`measured-here`, B60, a probe): a 48
  MiB copy from USM host memory on a second queue finishes in 3.5 ms while
  a 37.5 ms kernel runs on the first, the pair in 37.5 ms.

Measured (`measured-here`, 2026-10-13, B60, Flash-Next with the unit's
flags: q8_0 KV, 11,300 MiB of slots, MTP 2, the bank stream; one binary,
arcint with the lazy n-gram table, 0001-0042; A B A B, one process an arm;
the capital, the 20k needle, three 300-token answers):

| | the exchange (`NB=0`) | 0042, the copies behind the graph |
|---|---|---|
| decode: hash table / compiler / TCP | 32.3, 32.3 / 36.2, 36.1 / 34.5, 34.4 t/s | 33.1, 33.1 / 37.0, 36.9 / 35.2, 35.4 |
| the verify's input wait | 2.6-6.0 ms | 0.94-0.95 |
| decode experts on the card | 89.9 / 93.8 / 92.2 % | 88.5 / 93.3 / 91.3 |
| 20k needle prefill | 673.3, 673.9 t/s | 673.1, 673.2 |
| swap check | 0 of 6 | 0 of 6 (red case 1 of 6) |

Decode +2.4 % on the mean. The newcomers land two verifies later than the
exchange's, which costs ~1 point of hits; the waits saved outweigh it. The
first build (the copies enqueued before the next graph) decoded 31.5, 31.6
/ 35.8, 35.9 / 33.9, 33.9 against the exchange's 32.2, 32.3 / 36.2, 36.1 /
34.5, 34.4 in its own A/B. Answers right in every arm, the draft acceptance
counts identical (170/259, 169/262, 163/252): a hit and a bank read run the
same kernel on the same bytes.

- **Reviewed** before the push by a second model: no race found (a slice
  is written on the adapt queue only while no table maps it, behind a fence
  covering the compute work before the last table change); the review's
  points went in: the spares clamped so a bank stays within the GEMM's 512
  matrices, the ticket list emptied whenever no ticket is outstanding, the
  deferred swaps logged. The build with them (`measured-here`, B60, the
  same flags): decode 33.1 / 36.9 / 35.3 t/s, swap check 0 of 6 (red case 1
  of 6), no layer clamped, prefill 672.8 t/s.

## 0043-opencl-cache-mm-id-min-tok.patch

An expert cache's tensors (its slots and its bank, named `.slots` / `.bank`)
take `MUL_MAT_ID`'s GEMM from 64 tokens instead of 16
(`GGML_OPENCL_KQ_MM_ID_CACHE_MIN_TOK` overrides the 64; every other MoE tensor
keeps `GGML_OPENCL_KQ_MM_ID_MIN_TOK`'s 16).

- **Why:** a prefill ubatch on the GEMM path with the bank stream on (0032)
  copies each bank tensor whole into the VRAM ring, the whole bank over the
  link (22.37 GiB at Flash-Next's sizes, the load log's bank line) whatever
  the token count. A short follow-up prompt
  paid ~1.9 s for 24 tokens (`measured-here`, B60, the 0.7.5 package, the
  unit's flags). Priced first with the switches alone (one process an arm,
  seconds to prefill):

  | prompt | the bank stream (0.7.5) | the gather (stream off) | GEMM from 64 | GEMM from 512 |
  |---|---|---|---|---|
  | 24 tokens (a second request) | 1.87 | 0.78 | 0.64 | 0.65 |
  | 166 tokens | 1.86 | 1.51 | 1.51 | 1.54 |
  | 694 tokens | 1.94 | 2.64 | 1.93 | 2.63 |
  | 20,045 tokens | 29.69 | 37.79 | 29.69 | 29.72 |

  The bank stream learns its order on a process's first GEMM graph and
  gathers there (`ggml_cl_bank_stream`). With a threshold of 64 that graph
  was the 166-token prompt's, with 512 the 694-token one's: those cells are
  the gather's cost (1.51, 2.63), not the threshold's.

- **Measured** (`measured-here`, 2026-10-13, B60, Flash-Next with the unit's
  flags, one binary: the 0.7.5 tree + 0043; 16 restored by the switch; A B A
  B, one process an arm):

  | prompt | 0043 | GEMM from 16 |
  |---|---|---|
  | 24 tokens (the first request) | 0.74-0.77 s | 0.87-0.88 (the stream's learning graph: gathered) |
  | 24 tokens | 0.64-0.65 s | 1.86-1.87 |
  | 28 tokens | 0.57 s | 1.82-1.83 |
  | 166 tokens | 1.51-1.52 s (the learning graph: gathered) | 1.85-1.86 |
  | 694 tokens | 1.93 s | 1.93 |
  | 20,045 tokens | 29.67-29.68 s | 29.69-29.70 |

  - The prompt's tokens now take the matvec's arithmetic (q8_1 activations,
    as decode always has) instead of the GEMM's: the texts differ. The
    hash-table answer after a 28-token prompt accepted 166 of 267 drafts
    (33.0 / 32.9 t/s) against 170 of 259 (34.6 / 34.6), the same in both
    runs of each arm; the answers right in every arm.
  - KL (`llama-perplexity`, 16 x 512 against the CPU reference
    `fn-ref-q8.kld`, q8_0 KV, the expert cache; 32-token ubatches so that
    the two thresholds take different paths): 0.294000 / 87.181 % against
    the GEMM's 0.293562 / 87.279 % (+0.0004 nats, -0.10 points: inside the
    bar); 8.53 against 25.48 s a pass. The 512-token arm (the GEMM either
    way) reads 0.294037 / 87.353 %, stage 7's value.
- The coder (its experts resident, no `.slots` / `.bank` tensors) and the
  dense models do not take this branch (`code`).
- **Open, priced:** a prompt of 64 tokens or more still streams the whole
  bank once the order is learnt, ~1.84 s a graph up to ~1,000 tokens. The
  crossover against the gather (`measured-here`, B60, the 0043 binary, the
  order learnt on a 2,665-token prompt first, two runs each, identical to
  the hundredth): 173 tokens 1.84 s streamed against 1.56 gathered; 317:
  1.84 / 1.96; 461: 1.85 / 2.26; 702: 1.90 / 2.64; 1,470: 2.36 / 3.51. A
  stream threshold near 250 tokens would save at most ~0.3 s on prompts of
  64-250 tokens; built as 0044.

## 0044-opencl-bank-stream-min-tok.patch

The bank stream (0032) runs from 256 tokens a ubatch
(`GGML_OPENCL_BANK_STREAM_MIN_TOK` overrides; 0 streams every GEMM ubatch as
before); below that a MoE GEMM gathers its routed bank experts into the
card-side mirror instead of copying every bank tensor whole.

- **Why:** 0043's open item. The streamed GEMM costs ~1.84 s a graph at
  Flash-Next's sizes up to ~1,000 tokens whatever its tokens; gathered, 173
  tokens take 1.56 s and 317 take 1.96 (the crossover near 250).
- **Measured** (`measured-here`, 2026-10-14, B60, Flash-Next with the unit's
  flags, one binary: the 0.7.6 tree + 0044, the old behaviour by the switch
  set to 0; A B A B, one process an arm; the stream's order learnt first on
  a 2,665-token prompt):

  | prompt | 0044 | the stream for every GEMM ubatch |
  |---|---|---|
  | 77 tokens | 1.11 s | 1.84 |
  | 173 tokens | 1.54 s | 1.83 |
  | 269 tokens | 1.83 s | 1.83-1.84 |
  | 317-702 tokens | 1.83-1.90 s | 1.83-1.90 |
  | 20,045 tokens | 29.67-29.68 s | 29.66-29.67 |
  | 28 tokens (the matvec, 0043) | 0.58 s | 0.58 |

  The 300-token answer after them decoded at 32.3-32.4 t/s in every arm,
  the draft acceptance the same (166 of 267): the gather fills the mirror
  in the bank's layout, as the ring does, and the GEMM reads the same bytes
  (`code`); the needle right in every arm.
- **Reviewed** before the push by a second model: a graph streams all its
  bank tensors or gathers all of them (the gate reads the ubatch's tokens),
  a gathered graph touches no stream state, the learnt order and the
  per-graph ring state survive alternation, the load-time mirror holds a
  whole bank tensor; the series 0001-0044 applies onto the pin. No defect.
  The ring is now allocated at the second prompt of 256 tokens or more
  instead of the second of 64 or more (the same VRAM, later).

## 0045-opencl-fa-dpas2-split-kv.patch

`flash_attn_dpas2.cl` (0038) splits a call's keys across work-groups when its
row tiles would leave a wave part empty, after llama.cpp CUDA's
`parallel_blocks` (`code`, `ggml-cuda/fattn-common.cuh` `launch_fattn` and
`flash_attn_combine_results`, at bed0a85).

- **Why** (`measured-here`, B60, f16, causal, 65,536 keys, 32 sub-groups):
  a work-group takes 256 query rows of one head and a Xe core holds one (32
  sub-groups at 256 GRF), so the B60 runs 20 at a time. The time is the
  number of waves times 9.4 ms:

  | rows | work-groups | waves | time |
  |---|---|---|---|
  | 256 | 24 | 2 | 18.9 ms |
  | 512 | 48 | 3 | 28.2 |
  | 768 | 72 | 4 | 37.6 |
  | 1,280 | 120 | 6 | 56.9 |
  | 2,048 | 192 | 10 | 93.9 |

  The agent's 512-token ubatch gives 48 work-groups, and the third wave is
  40 % full. vLLM's prefill uses the same 256-row tile without a split
  (`code`, `XeFHMAIndividualTileScheduler`), but its prompt chunks are
  thousands of tokens long, so its last wave is a small share of the work.
- **What:**
  - **The split count:** the reference's rule. Take the fewest splits whose
    waves are at least 95 % full, else the fullest, at most the call's 32-key
    tiles and 16 (`GGML_OPENCL_FA_DPAS2_SPLIT` lowers the cap; 1 turns the
    split off).
  - **The wave size:** `CL_DEVICE_MAX_COMPUTE_UNITS` (the vector engines,
    160 on the B60) / 8 x 32 / sub-groups a work-group, which is 20 on the
    B60; `GGML_OPENCL_FA_DPAS2_WAVE` overrides it. At 512 rows the B60 gets
    two splits.
  - **The state:** each split's work-groups take their key range with their
    own online-softmax state, in the carry layout (0037). Split 0's state is
    the carry, so it can take in the earlier key blocks' state; the others
    use slots of a scratch buffer.
  - **The merge:** `flash_attn_split_combine` merges the states into the
    carry (more key blocks follow) or into O.
  - **One split:** the old paths run unchanged.
- **Measured** (`measured-here`, 2026-10-14, B60):
  - **test-backend-ops FLASH_ATTN_EXT, head sizes 256 and 128:**
    - 866/867 with and without splits; the one failure is the same
      logit-softcap case in both;
    - the red case (`-DFA_SPLIT_RED`, the states added without their maxima)
      fails 92 (first written 94: the count included two summary lines);
    - the A770 is unchanged (sub-group 8 does not use this kernel).
  - **Offline ISA:** no spill, no private memory and no recompile retry, in
    the plain and carry kernels and in the merge kernel.
  - **Perf** (causal, the same binary, two runs each):

    | shape | no split | 0045 |
    |---|---|---|
    | f16, 65,536 keys, 512 rows | 28.17 ms | 23.89 (34.5 TFLOPS) |
    | f16, 65,536 keys, 256 rows | 18.9 ms | 12.34 |
    | f16, 8,192 keys, 512 rows | 3.72 ms | 3.39 |
    | q8_0:q4_0, 65,536 keys, 512 rows (with the test's ~22 ms unflattening) | 53.95 ms | 50.37 |
    | 768 rows and more | | unchanged |

  - **Served** (the dense agent's flags at 245,760, q8_0:q4_0; one dev
    binary, 0.7.6 tree + 0045, splits off by the switch; server-side
    prefill):

    | prompt | no split | 0045 |
    |---|---|---|
    | 54,895 tokens | 668.0 t/s | 690.6 (+3.4 %) |
    | 138,028 tokens, the needle | 461.6 | 490.1 (+6.2 %) |
    | 228,120 tokens, the needle | 342.9 | 370.1 (+7.9 %) |

  - **Answers:** the capital and both needles right in both arms.
  - **Acceptance task:** 10/10 three times in each arm. With the splits it
    is a different program (md5 `9b6b7a99` against `c76c468a`): the merge
    reorders the softmax sums, and the answer moved but stayed right.
  - **VRAM:** peak 24,271 -> 24,291 MiB (the scratch slots); GTT flat at
    28 MiB.
  - **KL** (16 x 512, `-b 512`, so a call has 512 rows and splits in two):
    0.005652 / 97.475 % against 0.005950 / 97.304 % without splits. The red
    case gives 0.216763 / 83.1 %.
  - **A blind run first:** with llama-perplexity's default batch (four
    sequences, 2,048 rows a call, 192 work-groups) the rule chooses one
    split. KL was then identical in all three arms, the red case included,
    so that run measured nothing.
- **Reviewed** before the push by a second model. It checked:
  - the carry and slot addressing against the merge's;
  - the work-group arithmetic;
  - argument order (the v1 carry kernel gets none of the new arguments);
  - that a split boundary keeps the mask and mask-class indexing;
  - empty and all-masked splits (`M_INIT` weighs 0, or 1 when no split saw
    a key);
  - mixed split and unsplit key blocks on the shared carry buffer;
  - that one split is bit-identical to the old path.

  No defect. Its notes:
  - The 95 % rule scores waves, not tiles. A short key block (160 keys,
    24 row tiles) can get 4 splits that cost as much as one: the reference
    has the same rounding, and it costs only on short blocks.
  - Only the comment naming the cap of 16 changed after the measured build.

## 0046-expert-cache-tables-one-write.patch

An expert cache's residency tables are rows of one tensor per card, written
in one call per adapt step, after Strata's one residency table and its one
upload (`code`, `src/program/generate.cpp:3589-3606`, `res_upload` at
`4406-4413`).

- **Why** (`measured-here`, 0042's record): each table was its own tensor and
  its own non-blocking write. Flash-Next has 3 a layer and an adapt touches up
  to 48 layers in each of its two phases (the victims out, the newcomers in),
  so up to ~290 writes of 2 KiB. A probe timed 240 of them at 1.64 ms on the
  B60, against 54 us as one write.
- **What:**
  - `tab_all`, I32 [1, n_expert, rows]: each layer's tables (slot, bank, the
    single branch's slot-or-bank, then the Q8_0 down pair) are views of its
    rows; the graph's `get_rows` reads them with the view offset (`code`).
  - One host copy; `upload_tables` fills a layer's rows, and `flush_tables`
    writes the changed range in one non-blocking `set_slices`. The host copy
    changes again only at a later adapt step, after a graph has passed, as the
    per-table copies did.
  - **A table check:** the tables read back (a blocking read) against
    `slot_of`/`bank_of`, at load and at the first adapt; the red case
    (`LLAMA_EXPERT_CACHE_MUTATE=2`, the write leaving out its last row) and
    the slot red case (`=1`) are separate now.
- **Measured** (`measured-here`, 2026-10-14, B60, Flash-Next with the unit's
  flags: 131,072 ctx, q8_0 KV, 11,300 MiB of slots, MTP 2, the bank stream;
  the 0.7.7 package against the 0.7.7 tree + 0046, A B A B, one process an
  arm; the capital, the 20k needle, three 300-token answers):

  | answer | 0.7.7 | 0046 |
  |---|---|---|
  | hash table | 31.4 / 31.5 t/s | 31.6 / 31.5 |
  | compiler | 36.6 / 36.7 | 36.7 / 36.8 |
  | TCP | 35.1 / 35.1 | 35.3 / 35.3 |
  | time writing swaps and tables, an answer | 0.09 / 0.06 / 0.07 s | 0.05 / 0.03 / 0.04 |

  - The same text in all four arms (md5 of each answer), the same hit rates
    (87.6-93.2 %), the capital and the needle right; the swap check 0 of 6.
  - The table check: 0 of 144 rows at load and at the first adapt; the red
    case 1 of 144.
  - Prefill: adapt steps are planned on ubatches of up to 8 tokens only
    (`code`, `advance()`), so a prompt's ubatches write no tables beyond
    the landing of swaps already in flight.
- **Reviewed** before the push by a second model:
  - the views' allocation (the parent first in the context);
  - every reader honours the view offset (only the six `get_rows`);
  - the views as VIEW nodes in the scheduler;
  - the host copy's lifetime against the in-order queue and the adapt fence;
  - the range arithmetic and both red cases.

  No defect. It found one pre-existing issue: the cache's context was sized
  at 8 tensors a layer, but a layer with a Q8_0 down projection needs 9.
  The context is now sized exactly. That fix and a reworded load log line
  came after the measured build. The final build, checked on Flash-Next:
  the capital, the needle, the hash-table answer's same text (md5
  `d2069d0a`) at 31.6 t/s, and the table checks at 0 of 144.

## 0047-input-copies.patch

A graph's inputs (positions, masks, ids) reach the card without a blocking
write each and without a wait each.

- **Why** (`measured-here`, stage 8 of `docs/campaigns/flash-next-llama-engine.md`):
  - With one GPU the scheduler creates no events (`n_copies == 1`,
    `ggml-backend.cpp`, `code`). So for each user input of a split it
    synchronizes the backend and then copies synchronously. CUDA on one GPU
    runs the same sequence; there, the calls are cheaper.
  - On the B60 a sync costs ~10.6 us and a blocking 1 KiB write ~10.4 us.
  - Flash-Next's verify has 34 inputs (0.93-0.95 ms a graph) and a draft
    graph 17 (0.40 ms). The GPU idles through them.
- **What:**
  - **The scheduler** waits once before a split's first user input instead
    of before each. Each copy is synchronous and nothing in the input loop
    computes, so one wait covers them all (`code`; reviewed).
    `GGML_SCHED_INPUT_SYNC_EACH=1` restores the wait per input.
  - **The OpenCL backend:**
    - a write of at most 4 MiB into a buffer that holds no weights is
      copied into a 32 MiB USM host ring and enqueued without a wait, so the
      caller may reuse its bytes at once;
    - the ring is reused after a `clFinish` when it wraps;
    - weights, USM buffers and converted layouts keep their own paths;
    - `GGML_OPENCL_STAGE_INPUTS=0` writes blocking again.
- **Measured** (`measured-here`, 2026-10-14, B60):
  - **Flash-Next, the unit's flags** (131,072 ctx, q8_0 KV, 11,300 MiB of
    slots, MTP 2, the bank stream). One binary (the 0.7.7 tree + 0046 +
    0047); the old behaviour by both switches; A B A B; `GGML_SCHED_TIMING`:

    | | old | 0047 |
    |---|---|---|
    | a verify's input copies | 926-949 us | 582-593 us |
    | a draft's input copies | 395-404 us | 134-141 us |
    | hash table, 300 tokens | 31.6 / 31.6 t/s | 32.0 / 32.0 |
    | compiler | 36.8 / 36.8 | 37.3 / 37.3 |
    | TCP | 35.3 / 35.3 | 35.8 / 35.7 |
    | the 20k needle's prefill | 650.4 / 650.3 t/s | 650.9 / 650.5 |

  - **Answers:** the same text in every arm (md5 of each answer), and the
    capital and the needle right.
  - **The red case** (`GGML_OPENCL_STAGE_RED=1`: the ring's stale bytes
    reach the card): the capital answered "!!!!!!!!", the needle empty, the
    answer all "!".
  - **test-backend-ops** (the full suite, the B60): 11,143 of 11,294 with
    staging and without, the same 151 failing cases (the error values differ
    run to run: the inputs are random). The red case fails 1,066.
  - test-backend-ops alone cannot catch a staging fault: its CPU reference
    reads its inputs back from the GPU buffer, so both sides see the same
    bytes. The served red case is the one that bites.
  - **The dense agent (B60) and the coder (A770)**, each with its unit's
    flags, the same binary and switches, A B A B, three 300-token answers
    each (`measured-here`):

    | | old | 0047 |
    |---|---|---|
    | agent, input copies a graph (draft / verify) | 203 / 245-249 us | 65-67 / 114-118 |
    | agent decode | 36.8 / 39.4 / 37.1 t/s | 37.1-37.2 / 39.7-39.8 / 37.5-37.6 |
    | coder (A770), input copies a graph | 1,930-2,020 us | 133-255 |
    | coder decode | 42.9-43.0 / 39.9 / 36.9-37.0 t/s | 49.3-49.4 / 45.9-46.0 / 42.6 |

    The same text in every arm (md5s), the capital right. The coder gains
    14-15 %: the A770 sits behind the chipset's PCIe 3.0 x4 link, where a
    sync and a blocking write cost ~60 us an input.
- **Reviewed** before the push by a second model:
  - one wait a split is enough (non-user inputs between user inputs, several
    splits on one backend, the CPU backend, events present);
  - the ring's ordering against the backend's other queues (the bank stream
    and the expert cache's adapt queue read no buffer this path writes);
  - which buffers stage;
  - the threading (one thread writes a context's tensors).

  It found two defects, fixed before the final build:
  - After the context's last free the ring was freed but still marked on, so
    a write after a reload in the same process would have used it. The ring
    is now decided again.
  - `GGML_OPENCL_STAGE_RED` was first offered as a test-backend-ops red case,
    which cannot fail (above); the served red case replaced it.

  Its notes, taken: the ring is allocated only for a staging candidate, and
  the red switch is read as a number. The final build: the capital, the
  needle, the hash-table answer's same text at 32.0 t/s, inputs 591 / 138 us.

## 0048-opencl-fa-sg8-v-pairs.patch

The A770's prompt attention (`flash_attn_dpas.cl`, sub-group 8) with f16 V
stages V a key pair a work-item: two keys' 8 values packed into 8 uints (the
P V b operand's pairs) and stored as 8 dwords, one a dim, instead of 16
scalar halves. The pair is keys j and j + 8 of a 16-key group, so P's a
operand comes straight from S's two 8-key halves without shuffles, and a
sub-group's V loads read 8 consecutive rows. `GGML_OPENCL_FA_SG8_VBLK=0`
keeps the old staging; `-DFA_VBLK_RED` (through `GGML_OPENCL_FA_DPAS_OPTS`)
swaps the pair: the red case. Quantized V keeps the old staging (below).

- **Why:** the A770's ablations after 0039 put V's transposing stores at ~12
  ms of 70 (`docs/campaigns/dense-prefill-attention.md`, A770 stage).
- **The forms tried** (`measured-here`, 2026-10-14, the A770 at its pinned
  2000 MHz, test-backend-ops perf, causal, 24 heads on 4, 512 rows, 65,536
  keys; offline ISA from ocloc acm-g10):

  | form | f16 | q8_0:q4_0 | ISA |
  |---|---|---|---|
  | old (transposed tile, 8 scalar stores) | 166.1 ms | 139.8 | f16 spills 2,304 B |
  | V in the b operand's blocked layout, block reads | 155.6 | 162.8 | spills 1,760 B: the block reads' addresses |
  | key pairs (2i, 2i+1), 8 dword stores, a separate pass | 137.8 | 155.4 | no spill |
  | the same fused into K's loop / V before K | 137.1 / 136.6 | 158.9 / 156.7 | no spill |
  | pairs (j, j + 8), no shuffles (this patch) | 128.6-129.0 | 150-152 | no spill, 2,732 instructions (f16) |
  | the same with the pair decoded packed / loads batched | | 150.0 / 148.6 | |

  The quantized loss is the decode: with V read as plain halves instead
  (timing only), the old form takes 168.6 ms and the pair form 133.7. The
  old loop hides V's dequantization under its loads; the pair form exposes
  it, and neither decoding into packed pairs nor batching the loads won it
  back. So the define reaches the f16 program only (the kernel refuses a
  quantized V at compile time).
- **Measured** (`measured-here`, the final build, two runs each):
  - **Kernel** (f16): 32,768 keys 82.7 -> 64.0 ms, 65,536 keys at 256 /
    512 / 1,024 rows 103.3 -> 74.4, 166.1 -> 128.6, 352.3 -> 280.1 (-20 to
    -28 %); q8_0:q4_0 unchanged (140.1 / 140.2).
  - **test-backend-ops FLASH_ATTN_EXT** (the A770): 2,808 of 2,812 with
    and without, the same 4 failing cases; the red case fails 70.
  - **Served** (the coder's model with `--llama-kv f16` at 49,152, MTP 4,
    one binary, the switch off in the other arm, A B A B): the 40,841-token
    needle 661.6-665.8 -> 734.4-741.2 t/s (+11 %), 18,413 tokens 903-912
    -> 933-973; the needles and the capital right; the acceptance task
    10/10 in all four arms, the same program; peak VRAM 15,818-15,834 MiB.
  - **KL** (16 x 512 at 512 rows a call, against `coder-ref-q8.kld`, f16
    KV): 0.024717 / 93.04 % against 0.024837 / 93.14 %; the red case
    1.3828 / 51.8 %.
  - The deployed coder runs q8_0:q4_0 at 98,304 and does not take this
    path. f16 KV at 98,304 would not fit the card (49,152 peaks at 15.8
    GB). At 49,152 its 41k prefill (734-741 t/s) is no better than the
    deployed q8_0:q4_0 today: 725.8 t/s on the same 40,841-token needle at
    98,304 (0049's gate below, with 0047 in the tree). An earlier note set
    it against 0039's 685 t/s, a stale number; retracted.
- **Reviewed** before the push by a second model:
  - the pairing against the DPAS operand layout (the same permutation of
    the 16 keys in P and V, so the contraction is unchanged);
  - the staging indices, bounds and tail clamps;
  - the barriers, and the old path being untouched without the define;
  - that only the f16 program gets the define.

  No defect. Two comments on the shuffle path were updated afterwards
  (comments only).

**Tried after 0048, not taken: quantized K/V to f16 once a call on the A770**
(llama.cpp CUDA's need_f16_K/V, `code`, `fattn-common.cuh` `launch_fattn`).
For a quantized prompt call, sub-group 8 converted both views to f16 scratch
(the quantized kernel's own `load8`, so the same halves) and ran 0048's f16
kernel on them (`GGML_OPENCL_FA_SG8_KV2F16`; llama tree `p0049-a770-kv2f16`,
unmerged). Measured (`measured-here`, 2026-10-14, A770):
- **Correct:** test-backend-ops the same 4 failing cases with and without,
  the red case failing 95; KL (q8_0:q4_0, 512 rows a call) 0.026847 /
  92.75 % against 0.026908 / 92.40 %, the red case 1.38.
- **Served** (the coder's unit flags, q8_0:q4_0 at 98,304, A B A B, the
  pairs identical): the 40,841-token needle 725.8 -> 732.4 t/s (+0.9 %),
  81,974 tokens 502.9 -> 509.6 (+1.3 %); the needles right, the task 10/10
  against 9/10.
- **The price:** peak VRAM 15,977 -> 16,139 MiB of the card's 16,288: the
  scratch grows with the depth (~201 MB at 98,304 for the coder's two KV
  heads) and leaves 149 MiB. A thin margin is the known way into xe
  eviction (memory `project-xe-eviction-thrash-margin`); a 1 % gain does not
  pay for it.
- **The premise was overstated:** test-backend-ops' quantized arms include
  a per-call unflattening of the test's set_tensor-uploaded K/V that a served
  cache does not have, so its "f16 8 % faster than quantized" at one shape
  was mostly that; served, the quantized kernel is within ~1 % of the f16
  one. Kernel-level quantized-vs-f16 comparisons on this test need that cost
  subtracted.

**Failure counts corrected (2026-10-14):** the counts of failing
test-backend-ops cases in 0047, 0048 and its addendum were `grep -c FAIL`,
which also counted the two summary lines ("Backend OpenCL: FAIL" and the
closing "FAIL"); each is two lower than first written (153 -> 151, 6 -> 4,
the red cases 1,068 -> 1,066, 72 -> 70, 97 -> 95). 0045's red case was
counted the same way: 92, not 94. The passed/total pairs came from the
summary line and stand.

## 0049-decode-launches.patch

Fewer kernels in a decode graph, the same bits. A Flash-Next verify graph
launched ~4,150 kernels (`measured-here`, the 0.7.5 profile: 58.4 ms of
kernel time); at ~3 us an enqueue that is ~12 ms of host time a graph.
Four changes, three of them llama.cpp CUDA's fusions (`code`, `ggml-cuda.cu`
`ggml_cuda_can_fuse`):

- **The rollback snapshots without a `cont`** (`src/models/qwen4exp.cpp`
  `build_conv_state_at`): each of the K = n_rs_seq + 1 conv-state snapshots
  was `ggml_cpy(ggml_cont(tail), dst)`, two kernels; now `ggml_cpy(tail,
  dst)` as `delta-net-base.cpp` `build_conv_state` does. A pure move: the
  same bytes.
- **rms_norm + scale** (bias 0; the GDN q/k l2-norm, `build_gdn_l2_norm`):
  `kernel_rms_norm` takes the scale. Under `-cl-fast-relaxed-math` IGC
  regrouped `(x * scale) * post` to `x * (scale * post)`, one rounding fewer:
  KL stayed inside the bar but the texts changed and MTP's draft acceptance
  fell (the coder 40.7 -> 36.4 % on one answer). The intermediate now passes
  through an XOR with a kernel argument that is always 0, which keeps the two
  roundings: the same bits as the two ops.
- **ssm_conv + silu**: the conv kernels apply `kernel_silu`'s expression.
- **sigmoid + mul** of the same shape (the GDN z gate; CUDA's
  `ggml_cuda_op_unary_mul` conditions): one kernel.
- **Switches:** `GGML_OPENCL_FUSE_0049=<mask>` (1 rms_norm + scale, 2
  ssm_conv + silu, 4 sigmoid + mul; 0 off; `GGML_OPENCL_FUSE_RMS_SCALE=0` is
  the same as 0). `GGML_OPENCL_FUSE_0049_RED=1` drops the folded scale: the
  red case.
- **Measured** (`measured-here`, 2026-10-14, the units' flags, A B A B,
  three 300-token answers):
  - **Bit-identical:** every answer's md5 and every draft-acceptance figure
    equal to the unfused arm (Flash-Next against the 0.7.8 package, the agent
    and the coder against the mask 0), the capital and the needle right.
  - Flash-Next (B60): the verify graph 12,346-12,407 -> 11,666-11,678 us of
    scheduler time; decode 32.0-32.1 / 37.3-37.4 / 35.7-35.8 -> 32.2-32.3 /
    37.5-37.6 / 36.0-36.1 t/s.
  - The dense agent (B60): the verify graph 8,640-8,662 -> 8,139-8,143 us;
    decode 37.2 / 39.8 / 37.5-37.6 -> 37.3 / 39.9 / 37.7.
  - The coder (A770): the verify graph 6,695-6,699 -> 6,380-6,422 us; decode
    49.4 / 45.9 / 42.5 -> 49.8 / 46.4 / 43.0.
  - Per fusion on the coder before the fix: masks 2 and 4 already gave the
    unfused bits; mask 1 alone changed them.
  - KL (the first build, the regrouped norm; 16 x 512 against the
    references): Flash-Next 0.293942 / 87.18 % against 0.292021 / 87.40 % at
    512 rows a call, 0.294093 against 0.294000 at 32; the dense 27B 0.005650
    / 97.33 % against 0.005652 / 97.48 %; the coder 0.026801 / 92.45 %
    against 0.026908 / 92.40 %. The final build is bit-identical to the
    unfused path, so these bound it from above.
  - test-backend-ops (the full suite, the B60): 11,143 of 11,294 with the
    fusions on and off, the same 151 failing cases.
  - The red case crashed the server on its first request (the GDN state
    blows up without the q/k scale): the fused path runs.
- **Reviewed** before the push by a second model (the first form): the cpy
  without the cont against the OpenCL copy kernels and the CPU, the fusion
  checks (single use, shapes, contiguity), argument indices, the order
  against the existing fusions; no defect. It flagged the fast-math
  regrouping as compiler-contingent, which the per-fusion md5 test then
  confirmed and the opaque zero fixed.

## 0050-gdn-snapshots-in-cache.patch

The gated delta net's K rollback snapshots go straight into the recurrent
cache. In the rollback path (`delta-net-base.cpp` `build_recurrent_attn`, K =
n_rs_seq + 1 > 1) the op wrote the attention output and K state snapshots
into its own output, and a `ggml_cpy` then moved the snapshots into
`ssm_states_all`: one more kernel and a copy of K x 128 x 128 x heads floats
per GDN layer every verify. vllm-xpu-kernels' delta stage updates the state
in place (`code`, `csrc/xpu/gdn_attn/gated_delta_rule.hpp`); here the Intel
GDN kernels (`gated_delta_net_intel.cl`, `_tg` and `_pp`) take the
snapshots' destination (buffer, offset, slot stride). When the GDN node's
next real node is that `CPY`, with the shapes and strides the graph builder
gives it, they write into the cache rows and the `CPY` is skipped.
`GGML_OPENCL_FUSE_0050=0` keeps the copy; `GGML_OPENCL_FUSE_0050_RED=1`
writes every slot over slot 0 (the red case).

- **Measured** (`measured-here`, 2026-10-14, the units' flags, one binary,
  the switch off in the other arm, A B A B, three 300-token answers):

  | service | off | on |
  |---|---|---|
  | dense agent (B60, MTP 5: K = 6) | 37.3 / 39.9 / 37.7 t/s | 39.2-39.3 / 42.0 / 39.6 (+5.1-5.3 %) |
  | coder (A770, MTP 4: K = 5) | 49.7 / 46.2-46.3 / 42.9 | 50.7 / 47.1 / 43.6-43.7 (+1.8-2.0 %) |
  | Flash-Next (B60, MTP 2: K = 3) | 32.2 / 37.5-37.6 / 35.9-36.0 | 32.6-32.7 / 38.2 / 36.6-36.7 (+1.2-1.9 %) |

  - Every answer's md5 and every draft-acceptance figure equal in both
    arms; the capital and the needle right. The red case changed all three
    Flash-Next texts: the rollback reads the slots the kernel writes.
  - test-backend-ops GATED_DELTA_NET 42/42 on both cards, on and off (single
    ops: the fused pattern is not in them, so the suite checks the unfused
    path with the new arguments; the served md5s check the fused one).
  - The scheduler's host time a graph does not move: the saving is the copy
    on the card.
- **Reviewed** before the push by a second model: the unfused addresses
  identical to the old ones, K = 1 untouched, the generic kernel not given the
  new arguments, the fusion's shape and stride checks against
  `delta-net-base.cpp`'s views, the slot count, that nothing else reads the
  snapshot region, the node order and the scheduler's splits. No defect. Its
  notes, applied after the measured build: the unused `snap_size` dropped, and
  the `CPY`'s compute flag required, as the graph loop does.

## 0051-gdn-gating.patch

The gated delta net's gating computed inside the Intel GDN kernels.
`build_recurrent_attn`'s callers form the gate as
`softplus(alpha + dt_bias) * A` and beta as `sigmoid(b)`: four small ops a
layer ahead of the GDN. vllm-xpu-kernels' delta stage takes a and b raw with
A_log and dt_bias and gates inside (`code`,
`csrc/xpu/gdn_attn/gated_delta_rule.hpp`). Here the OpenCL backend, before
each graph, finds the chain in a window before each GDN node (every
intermediate used once), skips the ADD, SOFTPLUS, MUL and SIGMOID, and gives
`gated_delta_net_intel.cl` the raw projections with dt and A a head. The
kernel repeats the separate kernels' expressions with an always-0 XOR
between the steps, so the roundings are the separate kernels'.

The kernel reads alpha and b when the GDN runs, later than the allocator
keeps them: it frees alpha after the ADD. So `delta-net-base.cpp`
(`gdn_keep_gating_inputs`) attaches them as extra sources of the GDN node
(`src[6]` alpha, `src[7]` b), which keeps them alive; the op's meaning does
not change and the CPU ignores them. The backend folds only when those
sources are the chain's raw inputs, and refuses a fold if any computed node
between the chain and the GDN writes over their bytes.

`GGML_OPENCL_FUSE_0050` bit 2 is the switch (default 3: 0050 and 0051);
`GGML_OPENCL_FUSE_0051_PART` folds the gate (1), beta (2) or both (3);
`GGML_OPENCL_FUSE_0051_RED=1` passes A in place of dt (the red case).

- **Measured** (`measured-here`, 2026-10-14, the units' flags, one binary,
  the bit off in the other arm, A B A B, three 300-token answers):

  | service | off | on |
  |---|---|---|
  | dense agent (B60, 48 GDN layers) | 39.2 / 42.0 / 39.6-39.7 t/s | 39.4 / 42.1 / 39.7-39.8 (+0.3 %) |
  | coder (A770, 30 GDN layers) | 50.7 / 47.0-47.1 / 43.6-43.7 | 51.2 / 47.6 / 44.1-44.2 (+0.9-1.2 %) |
  | Flash-Next (B60, 36 GDN layers) | 32.7-32.8 / 38.2-38.3 / 36.6-36.7 | 32.7-32.8 / 38.3 / 36.6-36.7 (neutral) |

  - Every answer's md5 equal in both arms on all three; every GDN node
    folded (36/36, 48/48, 30/30); the red case changed all three Flash-Next
    texts and lost the needle.
  - Flash-Next's verify graph spends 0.3-0.4 ms less enqueueing (11.5-11.6
    -> 11.1-11.4 ms of host time), but its ~63 ms round is not host-bound.
  - test-backend-ops GATED_DELTA_NET 42/42 on both cards (single ops: the
    unfused path with the new arguments). The four kernel shapes compile
    offline with no spill and no recompile; the gating adds 6-20 % to the
    kernels' instruction count.
- **Not bit-identical at first, and why.** The first build's helpers sat
  inside `#ifdef GDN_PP`: the generation program did not compile and the
  backend fell back to `gated_delta_net.cl` silently (its "42/42" and md5s
  measured the generic kernel). Fixed, the fold changed every md5 while the
  answers stayed right. A switch per part showed beta's fold bit-identical
  and the gate's not; the ISA of the fused gate matched the separate kernels
  instruction for instruction. An overlap check then refused 36 of 36 folds:
  alpha's memory was reused between the ADD and the GDN. The extra sources
  fixed it. A map over the whole graph each compute, in the first pre-pass,
  cost about what the launches saved; it is a window search now.
- **Reviewed** twice by a second model. The first review read the version
  before the liveness fix: its one finding, the reshaped gate's own use
  count, is checked now. The second read the final diff, including every
  place in ggml, the allocator, the scheduler and the other backends that
  iterates a node's sources. It found no defect, and made two notes. A
  model whose beta is `reshape(sigmoid(b))` (qwen3next) folds nothing, as
  the match requires the sigmoid first; none of the served models builds it
  that way. And upstream's `graph_optimize` hook with `add_alloc_dep`, which
  CUDA uses, would keep alpha and b alive without touching the graph
  builder: the cleaner route at a rebase.

## 0052-reserve-cells.patch

The compute buffers sized for the prompts that need them, not for a full
cache. The scheduler reserves for a prompt ubatch at a completely full
cache: 2,048 tokens (the expert cache's whole chunk) against 131,072 cells
for Flash-Next, 3,048 MiB on the B60 plus 770 for the MTP draft context,
for the whole session. A listing of the reservation's largest tensors
(`measured-here`) named the QSA indexer's head scores, pools x 4 heads x
the ubatch's tokens in f32, 1,026 MiB, with a ReLU copy of the same size,
and the selection mask, cells x tokens in f16.

Two changes, the values unchanged:
- The indexer's ReLU in place (`qwen4exp.cpp`): the product's only reader.
  On its own, 3,048 -> 2,278 MiB and 770 -> 577.
- `llama_context_params::n_kv_reserve` (arcint `--llama-reserve-cells N`):
  the reservation sees min(cache, N) cells and their pools, and
  `llama_context::decode` cuts a batch's ubatches so that tokens x cells
  stays within n_ubatch x N (the batch's last position stands for the
  cells). The MTP draft context gets the same budget over its own ubatch.
  At 20,480 cells: 915 and 380 MiB. A prompt keeps whole 2,048-token
  chunks to ~20k tokens; deeper chunks are cut (1,861 tokens at 22.5k
  cells, 580 at 72k).

- **Measured** (`measured-here`, 2026-10-14, B60, Flash-Next at 131,072
  tokens, 3 drafts with the stop at 0.5, one binary; the capital, the 20k
  needle, Strata's long request twice, a 72k-token needle):

  | arm | slots | decode, first / second | 20k prefill | 72k prefill | peak VRAM |
  |---|---|---|---|---|---|
  | before | 8,191 (11,300 MiB) | 34.8 / 38.0 t/s | 657 t/s | 560 t/s (128.7 s) | 23.82 GiB |
  | ReLU in place | 8,191 | 34.8 / 38.0 | 658 | 560 | 22.85 |
  | ReLU, 11,850 MiB | 8,595 | 35.6 / 39.3 | 658 | 560 | 23.37 |
  | ReLU, `--llama-reserve-cells 20480`, 13,400 MiB | 9,724 | 38.4 / 43.0 | 662 | 416 (173.0 s) | 23.35 |

  Every arm: Paris, both needles, the same text (md5). The deep prompt's
  cost is the bank streamed once per (smaller) ubatch.
- **Not taken on the way:** a first build capped the cells but not the
  indexer's pool count (reserved from the cache size): 3,048 -> 2,613 MiB
  only, and an arm that spent the expected saving on slots ran 1.8 GiB
  over the card (26.2 GB resident, a 24-token prompt in 7 s); stopped by
  pid, the card recovered without a reboot.
- **Reviewed** before the push by a second model: no defect that changes an
  answer. Applied after the measured build: the cut ubatch never falls below
  a split's rollback tail (n_seq_max x (n_rs_seq + 1) + 1 tokens), so a small
  `--llama-reserve-cells` cannot abort `split_equal`. Its limits, recorded:
  the budget bounds the tensors that grow with tokens x cells; those that
  grow with cells alone (the re-pool after an early edit of a deep lane)
  can grow the buffer once by ~100-150 MiB at 131K, and named lanes without
  KV windows count cells by position, which undercounts them (growth back
  toward the old reservation, not corruption). The Flash-Next unit runs
  one lane.
