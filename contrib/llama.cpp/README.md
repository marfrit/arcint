# Patches carried against the pinned llama.cpp

The `--engine llama` executor (`src/exec/backend_llama.cpp`,
`docs/llama-engine.md`) builds llama.cpp as a subproject from
`ARCINT_LLAMA_DIR`. The pin is ggml-org/llama.cpp commit `3e4dfa8`
(2026-10-15; until then `bed0a85` of 2026-10-02, see "Moving the pin"); the
tree is that commit with the patches below applied in order:

    git -C <llama.cpp> checkout 3e4dfa8
    git -C <llama.cpp> apply <arcint>/contrib/llama.cpp/patches/*.patch

A patch that does not apply cleanly to the pin is a bug in this directory, not
a reason to move the pin. Each patch stays PR-shaped, so it can be offered
upstream.

A patch's KL check runs two arms: one sequence a batch, and four sequences a
batch (`llama-perplexity -b 4*c`, ubatches that mix the sequences). arcint
decodes one sequence per call, but llama.cpp's own batching does not. 0022
shipped a defect only the second arm sees (fixed by 0027).

Where a section below compares with "the OpenVINO path" or cites a
`marfrit-openvino` plugin patch, it means arcint's retired OpenVINO engine and
its runtime; both are at the tag `openvino-history` (`docs/history.md`).

## Moving the pin

The pin moves only as a release decision of its own (charter C3,
`docs/milestone-1.0.0.md`), never to make a patch apply.

What a move costs (`measured-here`, 2026-10-15, a clone of upstream master
`e467e5b` against `bed0a85`, 274 commits apart; counted at 0055, 0056 adds one
to each count): 43 of the 55 patches touch
`ggml/src/ggml-opencl/ggml-opencl.cpp`, where upstream changed 445 lines in
8 commits, among them `f2cc728`, which split `load_cl_kernels` (the
function most of the series extends; patch 0001 already fails there). 13
patches touch llama's `src/` (0010, 0021, 0022, 0024, 0026, 0029, 0030,
0042, 0046, 0049, 0051, 0052, 0055), where upstream made 53 commits, most of
the churn in `llama-context.cpp` (278 lines) and `llama-graph.cpp` (454).
`tests/test-backend-ops.cpp` (650 lines) is the third shared file. The
kernel files are almost all the series' own (`flash_attn_f32_f16.cl` 170
and `cpy.cl` 2 upstream lines are the exceptions). A move is a port of the
host side, not a re-apply.

The procedure:

1. A branch at the new commit; the GitHub tarball's sha256 in
   `contrib/packaging/arcint/build-deb.sh` (`LLAMA_COMMIT`,
   `LLAMA_TARBALL_SHA256`), the same two ARGs in `docker/Dockerfile.arcint`
   (its build checks them against the recipe), and the pin named here.
2. The patches ported in order, each committed and then regenerated as a
   patch against its predecessor, so each stays PR-shaped and applies with
   `git apply` on the new pin.
3. Each patch's own record re-run on the port: its test-backend-ops suites on
   both cards, its red case failing and its old-form switch passing, and
   the KL arms where the patch carries one (one and four sequences a batch).
4. The three services against the last release on the same cards: the
   answers (facts, the needle, the acceptance task) and the answer-level
   bar, and decode / prefill against the pin's build; a slowdown from
   upstream's own changes is recorded with its cause, not absorbed.
5. Released like any other version: the pre-tag served check, the second
   review, the fresh container.

**Moved to `3e4dfa8`, 2026-10-15** (upstream master that day, 288 commits on;
the survey of what upstream changed and what is open there is
`docs/campaigns/research-upstream-llama-2026-10.md`):

- **The port** (`code`): the 56 patches carried one commit at a time; 46
  applied unchanged, 10 resolved by hand. 0001 and 0002 above upstream's
  split-out compile options; 0005 and 0022 test cases beside upstream's;
  0007's Intel partials buffer beside upstream's pooled one; 0021's
  `build_moe_ffn` taken whole with upstream's MoE-cache `slots` passed
  where the experts are the model's own tensors (upstream's cache is off
  unless `moe_cache_size` is set; a layer our cache holds gets no upstream
  slots at all, so the two never cache the same layer, a defect of the first
  port found in its review); 0022's
  scheduler timing beside upstream's `ggml_backend_sched_copy_input`; 0024's
  accessors beside upstream's Hadamard loading; 0047's single wait for a
  split's user inputs moved into `ggml_backend_sched_copy_input`; 0052
  beside upstream's `sched_new` and `embd_all`, and qwen4exp's indexer
  (below); 0056's test target beside upstream's consolidated state tests.
- **Reviewed** by a second model (`code`): the regenerated patches rebuild
  the port tree exactly and the tarball checksum is right; one defect (both
  MoE caches could hold one layer when a tool enabled both), fixed in 0021.
- **Two semantic changes found by running it** (`measured-here`):
  - upstream's qwen4exp builds `ggml_lightning_indexer`, which OpenCL has no
    kernel for, and its probe is off (`auto_flid = false`), so the op ran on
    the CPU; 0052 now turns the probe on and keeps the unfused graph (the
    product, the ReLU in place, the heads summed) where the op is missing:
    "Lightning Indexer not supported, set to disabled" on both cards;
  - the batch API checks a token's embedding against the model's input
    width; an MTP context's hidden state is now a state embedding. arcint's
    draft batch (`src/exec/llama_spec.cpp`) uses
    `llama_batch_ext_set_embd_state`; with the token setter the first draft
    failed (`embedding size mismatch, got 1 rows x 10240`).
- **Suites** (`measured-here`, test-backend-ops, both cards, each op family
  the series touches: MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET,
  SSM_CONV, GET_ROWS, CPY, SET_ROWS, TOP_K, FILL, RMS_NORM, SOFT_MAX): every
  case passes on both bases; the new base carries upstream's added cases
  (MUL_MAT 1,221 -> 2,170). The pin's own failures on `3e4dfa8`: 74 MXFP4
  MUL_MAT_ID cases on the B60, three head-64 FLASH_ATTN_EXT cases on the
  A770.
- **Step 3, the red cases** (`measured-here`, 2026-10-16, the 54-patch
  series after the switch removal, both cards): every patch's red case
  re-run on the new base, each with the arm that must pass (the patch's off
  switch with the fault kept, or the build without it).
  - 95 test-backend-ops reds (a define, an environment switch or an edit of
    the kernel file the test build loads), 24 of them designed for this
    move for the 15 patches whose record had none or only prose (0001,
    0002, 0004-0009, 0012, 0014, 0019, 0020, 0033, 0034, 0040): every one
    fails its patch's cases and passes its control. Most counts equal the record (0016 42,
    0038 62, 0041 and 0045 92, 0053 49 of 87, 0054 87 of 87, 0039 92 on
    the A770); 0033's red reaches 1 case of 11 (it proves the suite runs
    the new tile shape, not more). Two needed work: 0025's recorded red
    (the destination's row stride from the source) changed nothing in its
    own cases, whose source and destination differ in the plane stride:
    re-designed on `nb2`, 2 of 6 fail as recorded; and 0036's cases tested
    nothing, a defect of the port (see 0036, fixed).
  - Host-code and served reds (a scratch tree rebuilt per red): 0006's
    route cache keys dropped, coder KL 2.31 nats against 0.0254 (the cache
    off: 0.0254); 0010's MTP rows in the old order, the coder's texts the
    same bytes and decode 43.5 / 39.6 / 38.6 against 50.7 / 47.1 / 43.6
    t/s; 0023's pre-patch tree lists one GPU of two; 0024's layer filter
    dropped, both cards' caches claim layers 7-29 (green: 7-13 and 21-29);
    0026's search head not reset, a 4k request after a 29k one in the same
    lane 9.0 s against 6.3 (prefill 670 against 885 t/s); 0027's pre-patch
    tree, four sequences a batch, 0.2438 / 81.96 % as recorded; 0029's
    window offsets forced to 0, four sequences in windows (one unified KV
    stream, `-kvu`), 1.3291 / 61.77 % as recorded (windows off on the same
    build: 0.0041); 0028's mask
    offset dropped, 423 of 2,820 attention cases fail (row blocks off:
    2,820 pass); 0043's and 0044's thresholds removed under the bank
    stream's red, NaN KL (the same red with the thresholds: 0.2875 and
    0.2881); 0052's indexer score negated, Flash-Next keeps the capital and
    loses the 20k needle and the long answer; 0052's reservation cap
    ignored, the compute buffer for 20,480 cells 2,112 MiB against 945
    (the draft context's 428 against 380).
  - Served reds by switch (Flash-Next, the unit's flags): the cache's
    checks report 1 of 6 copies (0021, 0042) and 1 of 144 table rows
    (0046); 0032's stream red aborts; 0047's staging red answers garbage
    and loses the needle (the copies unstaged: the same bytes as default);
    0049's red crashes at the first request, 0050's changes the long
    answer, 0051's loses the needle (each off: the same bytes); 0056's CPU
    rows zeroed change the long answer, its IQ4_NL split red faults
    (ggml-cpu's dot products: the same bytes); 0056's test fails 19 rows
    in its red mode.
  - Left: 0031 and 0055 dropped; 0052's ubatch cut (memory only, a 72k
    prompt) not run.
- **Step 3, the KL arms** (`measured-here`, one sequence a batch against
  four): the dense agent 0.004376 / 97.819 % against 0.004099 / 97.745 %
  (16 x 512, q8_0 KV, B60; four in one unified stream 0.004090 / 97.475 %,
  with a window each 0.004118 / 97.917 %); the coder 0.025383 / 92.770 % both (A770);
  Flash-Next IQ3_XXS 0.290954 / 86.618 % against 0.288792 / 86.716 % (8 x
  512, ubatch 4, the CPU misses on, B60). Each pair inside the answer-level
  bar.
- **KL as served** (`measured-here`, the old base against the new on the
  same card): Flash-Next (8 x 512, ubatch 4, the CPU misses on) 0.370044 /
  84.41 % -> 0.369325 / 84.07 %; the dense agent (16 x 512, q8_0 KV)
  0.004099 / 97.745 % on both; the coder (A770) 0.025383 / 92.770 % on both.
- **Served** (`measured-here`, dev builds of the old and the new base, the
  units' flags; Flash-Next as its unit runs since 0.7.11 with the Q4_K draft
  head, two processes each, alternated): Flash-Next 44.8-44.9 / 43.0-43.1 /
  42.1 / 45.1-45.2 / 37.8-37.9 / 36.1-36.2 -> 44.9-46.0 / 43.1-43.3 / 42.1 /
  46.5-46.8 / 37.1-37.3 / 37.6-37.8 t/s (+1.3 % on the mean), Paris and the
  needle, peak VRAM 23,741-23,781 -> 23,771-23,775 MiB; the dense agent's
  prefill 697.7 / 493.4 -> 695.5 / 491.9 t/s at 55k / 138k, the needle, the
  acceptance task 10/10 (the same program); the coder 51.2 / 47.5 / 44.1 ->
  51.1 / 47.6 / 44.1 t/s, the same texts (md5), the needle, the task 9/10.

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
- `GGML_OPENCL_GATHER_CHUNKS` sets the gather's per-expert split for
  measurement. Served, the gather's shape did not move its time (its bytes
  over the link do).
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
- Built, measured no faster or slower served, and removed in 0.8.1 with
  their switches (operator, 2026-10-16; the code is in 0.8.0's patch): the
  hyper-connection read fused into three kernels (after Strata's
  `fused_gr` v3 split by stream and K half; 30.8 t/s off, 30.4 on), a
  projection's matvec copying the next projection's misses into a
  card-side mirror (31.4 -> 31.6), a K-split F16 kernel with 4 rows a
  work-group, the per-pair gather grid for a few tokens, a matvec grouped
  by expert for 2-15 tokens (the coder's verify 7.85 -> 9.18 s, A770), a
  verify's pairs grouped by expert and decoded once (30.5 -> 29.5), and
  the activation as one int8 term in the K-quant and IQ matvecs (llama.cpp's
  and Strata's single q8_1 instead of codes plus residual codes: IQ4_XS
  49.6 -> 43.9 us at 3 columns, Flash-Next 31.0 -> 31.6-31.8 t/s, but the
  coder 43.3 -> 40.7, its MTP drafts accepted 42.7 -> 38.2 %). The expert cache's adaptation constants take
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
case, the mix 5 % off, fails 9 of 10; since 0.8.1 the case checks the
unfused chain), MUL_MAT IQ4_XS/Q4_K 94/94. Run
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
- **Kernels:** the matvec (1-4 columns), MUL_MAT_ID and the XMX GEMM
  (`mul_mm`, `mul_mm_id`); until 0.8.1 also the grouped paths (`idg`,
  `idgg`), removed with their switches.
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
  - the grouped kernels (removed in 0.8.1) ran only when switched on. With
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

## 0031-opencl-idg-args (dropped in 0.8.1)

**Dropped from the series in 0.8.1** with the grouped matvec it fixed,
which was removed with its switches (0022, operator 2026-10-16); the rest
of the series keeps its numbers. The record follows.

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

- **Fixed in 0.8.1** (`measured-here`, 2026-10-16, the red-case run of the
  pin move): on `3e4dfa8` the kernels load at the first buffer allocation,
  and the support check read the q8_0 kernel before that. A support query
  that came first (a test-backend-ops run filtered to these cases) answered
  "not supported", 0/0 on both cards; the unfiltered suite and the served
  graphs (weights allocated before the scheduler asks) were not affected.
  The check now builds the kernels first, as the others in the series do.

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
  ssm_conv + silu, 4 sigmoid + mul; 0 off; until 0.8.1 also the alias
  `GGML_OPENCL_FUSE_RMS_SCALE`). `GGML_OPENCL_FUSE_0049_RED=1` drops the folded scale: the
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

## 0053-opencl-lowbit-decode-spread.patch

Fewer integer operations in the low-bit expert matvecs' weight decode,
the same values. Flash-Next's expert kernels are bound by integer issue,
not by memory (stage 3d of `docs/campaigns/flash-next-llama-engine.md`).
Two decodes did per-byte work with shifts and masks:
- **IQ2_S and IQ3_S signs.** Eight sign bits per eight weights became a
  byte mask through `lb_signmask4` (seven operations per four weights).
  Now the four bits spread to the bytes' low bits by one multiply,
  `((b & 0xF) * 0x00204081) & 0x01010101` (copies at shifts 0, 7, 14, 21 do
  not overlap), and the signed value is `(g ^ (s * 0xFF)) + s`.
- **Q2_0's fields.** Four 2-bit fields of a byte went to four bytes with
  three shifts, four masks and three ors. Now fields 0, 2 and 1, 3 spread by
  one multiply each (`* 0x1001` and `* 0x40040` after masking `0x33` and
  `0xCC`).

Both identities hold for every input byte (checked exhaustively), so the
results are bit-identical. `-DLB_SIGN_MASK4` and `-DQ2_SPREAD_SHIFT`
through `GGML_OPENCL_IQ_OPTS` keep the old forms (measurement);
`-DLB_SIGN_RED` and `-DQ2_SPREAD_RED` are the red cases.

- **Kernels** (`measured-here`, 2026-10-15, B60, test-backend-ops perf,
  MUL_MAT_ID with 512 experts and top 10, two repeats each):

  | kernel | before | 0053 | instructions (offline ISA, bmg) |
  |---|---|---|---|
  | IQ2_S 640 x 2560, 1 / 3 tokens | 31.1 / 70.9 us | 29.1 / 63.6 | 487 -> 425 |
  | Q2_0 640 x 2560, 1 / 3 tokens | 21.8 / 52.0 | 20.9 / 49.7 | 372 -> 340 |
  | Q2_0 2560 x 640, 1 / 3 tokens | 24.8 / 63.5 | 24.6 / 62.3 | |
  | IQ3_S | | | 553 -> 489 |

- **Suites:** MUL_MAT and MUL_MAT_ID for IQ2_S, IQ3_S, Q2_0, IQ2_XXS,
  IQ2_XS and IQ1_M pass on the B60 and the A770. The red cases fail on the
  B60: signs 6/13 + 2/4 (IQ2_S), 6/13 + 8/12 (IQ3_S); Q2_0's fields 10/20
  + 49/87.
- **Served** (`measured-here`, 2026-10-15, B60, Flash-Next with the unit's
  0.7.9 flags, one binary, the old forms through `GGML_OPENCL_IQ_OPTS`,
  two processes each; Strata's request twice and two 300-token answers):

  | arm | decode t/s | verify s |
  |---|---|---|
  | before | 38.6 / 43.0-43.1 / 38.3-38.4 / 41.4 | 11.43-11.44 / 10.08-10.09 / 6.81-6.82 / 6.24 |
  | 0053 | 38.7-39.2 / 43.3-43.4 / 38.6-38.8 / 41.8-42.0 | 11.22-11.37 / 10.00-10.01 / 6.71-6.76 / 6.13-6.17 |

  +0.9 % on the mean; Paris, the 20k needle, the same texts (md5), prefill
  664 t/s in both. The coder and the dense agent have no tensors of these
  types.
- **Not taken on the way** (`measured-here`, the same harness):
  - The low-bit MUL_MAT_ID at 8 rows a sub-group where 16 lanes divide a
    row (gate/up only; down keeps 4 rows and its 4-lane rows, `MV_LB4`,
    which compiles only at 4): faster in isolation (IQ2_S 71.3 -> 67.5 us
    at three tokens) but ~1 % slower served (verify 11.44 -> 11.57-11.70
    s), with or without the sign change. Half the work-groups cover the
    missed experts' link reads less well; not measured further.
  - IQ2_S's grid as 16-bit codes (its bytes take three values): slower in
    every form tried; the campaign document has the table.
- **Reviewed** before the push by a second model: no value change and no
  build break; it rechecked both identities exhaustively (every IQ2_S and
  IQ3_S grid word against every sign byte) and found every IQ2_S, IQ3_S and
  Q2_0 kernel family (dense, MUL_MAT_ID, grouped, 4-lane rows) through
  `lb_expand`. Applied: a red case stacked on a switch that keeps the old
  form (`LB_SIGN_RED` with `LB_SIGN_MASK4` or `LB_CHAR4_ARITH`,
  `Q2_SPREAD_RED` with `Q2_SPREAD_SHIFT`) is a compile error instead of a
  green run that tests nothing (preprocessor lines only; the measured
  build's default source is otherwise the same). Recorded: the prompt GEMM's
  own copy of the IQ2_S / IQ3_S decode (`mul_mm_kq_f16.cl`) still uses
  `lb_signmask4`; 0053 is decode only.

## 0054-opencl-lowbit-gemm-decode.patch

0053's two decodes in the prompt GEMM's own copy of the low-bit decode
(`mul_mm_kq_f16.cl`, the expert GEMM from 64 tokens): the IQ2_S / IQ3_S sign
bits and Q2_0's fields spread by multiplies, the same text and the same
switches and red cases (the 0053 review had named this copy). The GEMM
build now also takes `GGML_OPENCL_IQ_OPTS`, so one binary can A/B it.

- **Suites** (`measured-here`, 2026-10-15): MUL_MAT and MUL_MAT_ID for
  IQ2_S, IQ3_S and Q2_0 pass on the B60 and the A770. With 0054 the red
  cases fail every case of these types on the B60 (IQ2_S 13/13 + 4/4, IQ3_S
  13/13 + 12/12, Q2_0 20/20 + 87/87), where 0053's alone left the GEMM
  cases passing.
- **Served** (`measured-here`, B60, Flash-Next with the unit's 0.7.9 flags,
  one binary, two processes an arm, the old forms through
  `GGML_OPENCL_IQ_OPTS`): the 20k needle's prefill 664.0 / 664.2 -> 665.8 /
  665.3 t/s (+0.2 %), short prompts inside the spread; the needle right,
  the same texts (md5).
- **Reviewed** by a second model: no defect. Noted: `-DLB_CHAR4_ARITH`
  through `GGML_OPENCL_IQ_OPTS` now selects the GEMM's char4 forms as well
  as the matvec's; label such prefill cells accordingly.

## 0055-expert-cache-lfru (dropped in 0.8.1)

**Dropped from the series in 0.8.1** (operator, 2026-10-16): not a general
gain (+1.6 % on the gate set, -2.1 % on six other prompts, below), and the
unit never ran it. Strata's adapt rule is the policy. The patch is in
0.8.0's series; the record of what it measured follows.

Infernix's expert-cache policy as an opt-in alternative to Strata's adapt
rule (`LLAMA_EXPERT_CACHE_POLICY=lfru`). Infernix (`code`,
`src/models/qwen4_exp/program/expert_cache/expert_cache.cpp`, `LfruPolicy`)
scores an expert uses / (now - last + 1), its uses never reset, protects
the round's routed experts from eviction and admits a miss only over a
lower-scoring victim, within a budget. Here, per layer inside the existing
slots and swap queue (0042): at each adapt (every 6 decode tokens) up to
`LLAMA_EXPERT_CACHE_LFRU_BUDGET` (4) admissions a layer, the round's experts
protected, the plan's global cap (96 swaps) unchanged.

Two forms:
- the default (the port): one clock tick a counted decode step, every
  bank expert used before a candidate;
- `LLAMA_EXPERT_CACHE_LFRU_EXACT=1`, Infernix's form: its clock ticks once a
  routed layer call (a round ages an expert by the layer count), and only
  the step's misses are candidates. Admission still runs at the adapt, not
  every layer call.

- **Measured** (`measured-here`, 2026-10-15, B60, Flash-Next with the 0.7.9
  unit's flags and 0053/0054; Strata's request, two 300-token answers, the
  20k needle; one binary an arm pair):

  | arm | decode t/s | on the card |
  |---|---|---|
  | Strata's adapt (5 processes) | 38.7-39.4 / 38.6-39.0 / 41.7-42.0 | 89.0 / 93.3 / 95.4 % |
  | LFRU port, budget 4 (4) | 39.0-39.5 / 39.9-40.2 / 42.1-42.4 | 89.6 / 95.1 / 96.2 % |
  | LFRU exact, budget 4 (2) | 39.0 / 39.8 / 42.2-42.4 | 89.3 / 94.6 / 96.0 % |
  | LFRU port, budget 1 (2) / 8 (1) | 38.2-38.9 / 39.6 / 42.1; 39.1 / 40.0 / 42.0 | |

  The port +1.6 % on the means (40.4 against 39.8 t/s over the four and
  five processes; per answer +0.6 / +3.2 / +1.1 %), the exact form +1.4 %.
  First recorded as +1.3-1.8 %, an arithmetic slip on the same runs
  (corrected 2026-10-15 in the 0.7.10 release review). Every arm Paris, the needle in
  30.8-31.2 s, the same texts (md5): the policy moves which experts are
  read over the link, not the values. Not taken: no credit for the
  columns of rejected drafts (Infernix's `live`; the next ubatch's position
  minus the last): 38.8 / 39.7-39.8 / 41.9-42.0, no gain. Not built:
  Infernix's single pool over all layers (the slots are per-layer
  tensors) and its warm start from saved state.
- **Reviewed** by a second model: no defect in the tables, the counts or the
  default path (Strata's rule unchanged; the plan never double-books a slot,
  as for Strata's rule, since it runs only when no swap is in flight). Its
  notes became the exact form above. Recorded: a budget that does not parse
  (`LLAMA_EXPERT_CACHE_LFRU_BUDGET=abc`) is 0, no admissions.
- **Outside the gate set** (`measured-here`, 2026-10-15, after the 0.7.10
  tag; B60, the 0.7.10 package, the unit's flags, fresh processes
  alternated, two an arm; six 300-token answers: a Python function, a
  German explanation, a history summary, a Bash script, a story, public-key
  cryptography; the same texts in every arm):

  | arm | decode t/s | on the card |
  |---|---|---|
  | Strata's adapt | 38.6-39.1 / 29.4 / 32.5-32.6 / 30.6-31.2 / 30.6-30.7 / 37.1-37.4 | 85.7 % |
  | LFRU, budget 4 | 37.9 / 29.5-29.6 / 31.5 / 30.2 / 28.6-28.7 / 37.7 | 85.2 % |

  LFRU -2.1 % on the means (32.6 against 33.3), per prompt -6.5 to +1.2 %;
  it swaps 1.5-2x as often (3,840-5,376 swaps a request against
  2,265-3,359). The hit counts repeat exactly across processes of an arm.
  The +1.6 % above is the gate set's, not general.
- **Operator** (2026-10-16): dropped, see above.

## 0056-expert-cache-cpu-misses.patch

Strata's doorbell tier for the expert cache's decode misses, with Strata's
CPU kernels. Opt-in with `LLAMA_EXPERT_CACHE_CPU_MISSES=1`. At a decode
layer's first expert projection a kernel hands the first
`GGML_OPENCL_CPU_MISS_MAX` (16) missed (token, expert) pairs to the CPU: -1
in the combined ids, so the card's three projections skip them, and the pairs
and the activation published to USM host memory behind a release store. The
card reads the other misses over the link as before. A coordinator thread and
a spinning pool (`LLAMA_EXPERT_CACHE_CPU_MISS_THREADS`, 8) compute the pairs'
gate, up, SwiGLU and down from the bank in host RAM while the card computes
the hits. After the down projection a kernel polls the done flag with
L1/L3-uncached loads (0077's form) and copies the rows in.

The CPU's rows are Strata's multi-token AVX2 kernels (`code`, Strata
`src/kernels/cpu/iq_avx2.cpp`, `q2_avx2.cpp`, MIT; `ggml-opencl-miss-cpu.cpp`):
a 32-value chunk is decoded once and every token of the window applies it.
The pairs are grouped by expert, so an expert missed by several tokens of a
verify is decoded once. Two departures: the weights are read from the bank's
planes (0022) instead of GGUF blocks, and the activations' quantizers are
written here (ggml's q8_K rule for IQ2_S / IQ2_XXS, Strata's 32-value chunks
for Q2_0). A layer of another type (Flash-Next's three IQ1_M layers) uses
ggml-cpu's dot products, as does `GGML_OPENCL_CPU_MISS_MT=0`.

A prototype with ggml-cpu's dot products (stage 3c of
`docs/campaigns/flash-next-llama-engine.md`) did not pay: ggml-cpu's Q2_0 dot
product has no SIMD form.

- **The kernels** (`measured-here`, 2026-10-15, the host's Ryzen 7 5700X, one
  thread, seeded data):

  | kernel | ggml-cpu, 1 / 4 tokens | Strata's rows, 1 / 4 tokens |
  |---|---|---|
  | IQ2_S 2560 x 640 (gate, up) | 98 / 394 us | 127 / 168 us |
  | Q2_0 640 x 2560 (down) | 661 / 2,637 us (0.7 GB/s) | 72 / 153 us |

- **Test** (`tests/test-opencl-miss-cpu.cpp`, ctest): the rows against
  ggml-cpu's `vec_dot` for IQ2_S, IQ2_XXS and Q2_0 in planes, 1-6 tokens a
  pass: the i-quants exact, Q2_0 within 2.7e-4 (Strata's activation chunks
  against ggml's q8_0). Its red mode (the planes read as GGUF blocks) fails
  18 of 18.
- **Served** (`measured-here`, 2026-10-15, B60, Flash-Next with the 0.7.10
  unit's flags and Strata's adapt rule, one binary, fresh processes; Strata's
  request, the two gate answers, a Python function, a German explanation and
  a story, 500 / 300 tokens; decode t/s):

  | CPU takes a layer | decode | mean |
  |---|---|---|
  | none (6 processes) | 38.7-39.0 / 38.6-38.8 / 41.7-41.8 / 34.7-34.9 / 31.8-31.9 / 31.5-31.6 | 36.3 |
  | up to 2 | 41.5 / 39.1 / 40.7 / 37.6 / 35.3 / 32.5 | 37.8 (+4.2 %) |
  | up to 4 (2) | 44.0 / 41.8-41.9 / 41.4-41.5 / 39.9 / 36.8 / 32.2 | 39.4 (+8.5 %) |
  | up to 8 (2) | 42.8-42.9 / 40.9 / 41.8 / 43.2-43.3 / 37.3 / 36.0-36.1 | 40.4 (+11.3 %) |
  | up to 16, 6 threads | 42.6 / 42.8 / 43.5 / 44.8 / 38.5 / 34.6 | 41.1 (+13.4 %) |
  | up to 16, 8 threads (the default; 4, two of them after the review's fixes) | 43.8-44.0 / 43.1-43.3 / 43.7-43.9 / 45.8-46.1 / 38.9 / 35.0-35.1 | 41.8 (+15.2 %) |
  | up to 16, 10 / 12 threads | 43.9 / 43.0 / 43.6-43.7 / 46.1-46.4 / 38.9 / 34.9 | as 8 |
  | up to 24 / 32 | 42.6 / 41.1 / 43.5 / 45.6 / 38.3 / 35.5; 42.2 / 44.0 / 43.2 / 44.4 / 37.1 / 33.3 | below 16 |

  Every arm Paris and the 20k needle (30.6-30.9 s). The texts differ from
  the card-only arm (the CPU's sums are not the card's) and repeat exactly
  across processes of an arm. The gain is largest where the cache misses
  most: the code answer (78 % of its experts on the card) +32 %.
- **KL as served** (`measured-here`, `llama-perplexity` against the
  Flash-Next reference capture, 8 x 512 tokens in ubatches of 4, so every
  ubatch takes the decode path and its misses go to the CPU; the cache
  through `LLAMA_EXPERT_CACHE_MIB` / `_PROFILE`, `--no-repack -lm mmap` as
  arcint loads):

  | arm | mean KL | top-1 |
  |---|---|---|
  | card only (2 runs) | 0.367287 | 84.363 % |
  | CPU up to 8 | 0.369126 | 84.412 % |
  | CPU up to 16 | 0.370044 | 84.412 % |
  | red case (`GGML_OPENCL_CPU_MISS_RED=1`: the CPU's rows left zero) | 0.436437 | 81.912 % |

  +0.0028 nats and +0.05 points against the bar's +0.03 and -1; the red case
  fails it.
- **The coder at the same time** (`measured-here`, the A770's unit answering
  300-token questions in a loop while Flash-Next decodes on the B60, two
  processes an arm): the coder 42.0 t/s end to end with Flash-Next's CPU
  misses against 42.4 without (-1 %); Flash-Next keeps its gain under that
  load (+14 %).
- **Reviewed** by a second model before the commit (`code`): the plane
  addressing, each decoder against ggml's references and Strata's, the
  quantizers, the SwiGLU clamp, the bounds, the memory ordering and the swap
  copies against the CPU's reads were found correct. Four defects, fixed
  and re-measured (the same texts, KL and decode as above):
  - the pool's claim could race with a new run's reset (a worker woken
    between the reset and the epoch bump), hanging the coordinator: the
    claim is now one CAS on a word holding the run's epoch, task count and
    next index;
  - the sign table's static initializer sat under the AVX2 target pragma,
    so loading the library could fault on a CPU without AVX2: it is built
    before the pragma (its initializer disassembles to SSE2 only), and
    clang gets the attribute form of the pragma;
  - the publish is flushed to the card, and the CPU's wait for it is a 30 s
    hang guard that aborts with a message instead of a 2 s timeout whose
    NaN rows reached the sampler;
  - a publish now runs below the kernel checks, on the per-pair matvec only,
    and waits out a job never merged before it rewrites the shared buffer.
- **On the unit** (operator, 2026-10-15): the Flash-Next unit sets
  `LLAMA_EXPERT_CACHE_CPU_MISSES=1` from 0.7.11 on (an `Environment=` line;
  it spins 8 threads at 100 % while Flash-Next decodes).
- **Fixed after 0.7.12: IQ4_NL banks on the slow path** (`measured-here`,
  B60, ISTA-DASLab's IQ3_XXS file, 2026-10-15). A layer whose types the
  multi-token rows do not take goes through ggml-cpu's dot products, which
  read each weight row as GGUF blocks. An IQ4_NL tensor in USM host memory
  is not stored that way: its first write (`ggml_cl_usm_set_tensor`) keeps
  its scales and its quants as two arrays. The slow path read the split as
  blocks: layer 0 of the IQ3_XXS file (IQ2_XS gate/up, IQ4_NL down; 18 of
  its 48 layers have an IQ4_NL down, the IQ2_XS file none) returned 13,230
  non-finite values in 40,960, layer 1's whole input was non-finite, its
  fused top-k chose no expert (INT_MAX ids), and its first CPU pair carried
  bank row 1,911,949,923 of 329: a fault in `row_iq<IQ2_XXS>` at the first
  request. Found by instrumenting the publish and the CPU path (the ids,
  the table and the routed ids at that node; non-finite counts a layer).
  The row reader rebuilds GGUF blocks from the two arrays
  (`ggml_cl_miss_cpu_join_iq4nl`), and the arrays are looked up at a layer's
  first publish: a first fix looked them up at registration, which comes
  before the bank's fill, and was measured inert. With the fix: no
  non-finite row; IQ3_XXS at 12,000 MiB of slots decodes 33.9 / 37.5 / 37.3
  / 33.5 / 30.6 / 30.9 t/s on six answers with the misses against 20.7 /
  26.2 / 28.9 / 18.7 / 21.2 / 18.5 without (+52 % on the mean), KL 0.290954
  / 86.618 % against 0.286562 / 86.765 % (8 x 512, ubatch 4), Paris and the
  needle right; the IQ2_XS unit's six texts unchanged (md5).
  `test-opencl-miss-cpu` checks split rows against the GGUF bytes; its red
  mode, and `GGML_OPENCL_CPU_MISS_SPLIT_RED=1` served (a fault at the first
  request), read the split as blocks.

## 0057-opencl-intel-rope-apply-fused.patch

The fork's `GGML_OP_ROPE_APPLY` (`code`, `leejet/ggml` `e05d18a1`,
`ggml.h:1877`, `ggml.c:4381`): a precomputed `pe` (`[2,2,d/2,tokens]`,
`[[cos,-sin],[sin,cos]]`) rotated against `a` (`F32 [d,heads,tokens,batch]`)
in one pass, returning contiguous `F32 [d,tokens,heads*batch]`. Ported: the
op definition (`ggml.h`, `ggml.c`, identical to the fork's), the CPU forward
(identical to the fork's `ops.cpp`), an original OpenCL kernel
(`kernels/rope_apply_f32.cl`) with the fork's CPU/CUDA pair arithmetic, its
`supports_op` (F32, even `d`, contiguous `dst`, at most `INT_MAX` pairs: the
kernel indexes in `int`) and dispatch, and `ggml-rpc.h`'s `GGML_OP_COUNT`
assert with the protocol patch version (an RPC build failed to compile
otherwise; the fork did the same in `087d92e7`). Nothing in libllama calls
the op (`code`). sd.cpp reaches it through
`contrib/sdcpp/upstream-rope-apply.patch`; upstream mode otherwise takes
its generic two repeats and four muls a layer.

**Served** (`measured-here`, B60, Z-Image Turbo, 8 steps, seed 42, the same
window): a 1024² step **2.14 s against 4.77 s** with sd.cpp's generic RoPE
(2.2x); at 512² the generic RoPE was 37 % of the f16 run's device time
(`repeat_f32` 3,054 + `mul` 2,188 ms of 14,053) and the fused op is 70 ms
a run. The program is built with `-cl-mad-enable -cl-fast-relaxed-math`, so
`#pragma OPENCL FP_CONTRACT OFF` does not reproduce the fork's separately
rounded CUDA arithmetic (`code`); the answer-level figure is the gate's.

**The port's defect** (`measured-here`, 2026-10-17, found by the image gate):
the kernel added `row*ne0` and `i0`/`i1` to a `global char * dst` -- byte
offsets on a char pointer -- so most cells were never written and the image
was white. Fixed with `*sizeof(float)`; every offset in the kernel is now in
bytes (`code`).

**Op suites**: `test_rope_apply` in `test-backend-ops`: Z-Image's geometry
(head size 128, 30 heads, 64 / 1056 / 4128 tokens), both pair layouts, three
forms of `a` (contiguous, the non-contiguous k view of a fused qkv as sd.cpp
builds it without qk_norm, padded rows), batch 2, and an `xcheck` form whose
output concatenates the fused op and sd.cpp's own generic decomposition
(`rope.hpp:1085-1122`) so the CPU forward is checked against independent
code. B60: 26/26 pass (`measured-here`); the CPU red (sign and swap) fails
all 16 `xcheck` cases.

**Red case**: a swapped cos/sin turns the image gate red (512² cos worst
0.890391, PSNR 9.70 dB, against the CPU reference, both arms prompted,
`measured-here`), and the op suite: with the kernel file mutated the same way
0/26 `ROPE_APPLY` cases pass on the B60, 26/26 with the file restored
(`measured-here`). The A770 passes the 26 too.

## 0058-opencl-intel-q8_0-xmx-gemm.patch

Upstream's generic q8_0 GEMM (`kernel_mul_mm_q8_0_f32_l4_lm`) was 68-81 % of
the image step (`measured-here`, `docs/campaigns/imggen-q8_0-xmx-gemm.md`).
The patch adds `deq_q8_0` (the int8 code and the f16 per-32 scale of the
tree's SoA split to a f16 local tile) and `kernel_mul_mm_q8_0_f16` on the
f16-weights GEMM's tile and `MM_BODY` (0022), the K-quant pattern
(0001/0009/0011-0014): dequantize a 256-wide super-block to f16 in local
memory, fp16 DPAS over k16, f32 accumulation. Dispatch: Q8_0 x F32 -> F32,
16 or more src1 columns (after `ggml_cl_flat_cols`), `ne00 % 32 == 0`, 2D,
contiguous, `view_offs == 0`, and only a flattened (SoA) Q8_0 tensor, read
through a view's `view_src` (a plain extra cast to the SoA struct would hand
the kernel a size as a buffer). The kernel is created only with sub-group 16:
the Xe2 B60, not the A770 (`code`).

Chunks past `ne00` are never written and never read (`kmax`, `code`).
`GGML_OPENCL_Q8_XMX=0` falls back to upstream's kernels;
`GGML_OPENCL_Q8_XMX_RED=1` zeroes the last 16 weights of every row (a kernel
argument), the red case.

**The language services.** The branch takes every 2D Q8_0 weight times a
batch of 16 or more columns on the B60, which before ran upstream's f32
kernels (16-31 columns the flat matvec, 32 or more the `l4_lm` GEMM), now
with f16 weights and activations and f32 accumulation (`code`). In the
served GGUFs the only Q8_0 weights on the B60 are in the MTP draft heads
(the agent's `nextn.eh_proj`; 16 in Flash-Next's head, among them K = 320
and 640); the agent's, Flash-Next's and the writer's main weights have none,
and the coder runs on the A770 (`measured-here`, a GGUF census). Drafts are
verified by the main model, so the answers cannot change; the speed is the
release A/B's (A3-0058).

**Served** (`measured-here`, B60, Z-Image Turbo Q8_0 + Qwen3-4B Q8_0, the
service's flags, prompted, 8 steps, seed 42, Euler, one window):

| arm | resolution | step | gate vs the CPU reference |
|---|---|---|---|
| 0058 | 1024² | **2.14 s** | PASS, cos worst 0.999926, PSNR 42.59 dB, SSIM 0.999257 |
| 0058 off (`Q8_XMX=0`), the baseline arm | 1024² | 10.95 s | PASS, 0.999922, 42.40 dB |
| 0058 red (`Q8_XMX_RED=1`) | 1024² | 2.14 s | FAIL, 0.987219, 19.56 dB |
| 0058 | 512² | **0.476 s** | PASS, 0.999987, 46.54 dB |
| 0058 off | 512² | 3.03 s | PASS, 0.999986, 45.84 dB |
| 0058 red | 512² | 0.478 s | FAIL, 0.995502, 23.39 dB |

5.1x on the 1024² step and 6.4x at 512²; a profile at 512² counts 1,605
launches of `kernel_mul_mm_q8_0_f16` and none of the generic kernel. At
1024² the kernel is 69 % of the step at ≈ 33.5 TFLOP/s, 34 % of the B60's
XMX f16 peak. Faster than the f16 weights on 0022 (2.21 s) in 5.5 GiB less
memory.

**Op suites**: Q8_0 `MUL_MAT` cases for the branch (K off the 256
super-block: 288, 2080, 96; odd m; n = 16, 17, 31, 32, 65, 512; the DiT's
shapes; a view of the first rows of a larger weight; 3D src1 flattened to 16
and 32 columns). B60: 151/151 pass with the branch and without it
(`GGML_OPENCL_Q8_XMX=0`); with the red, 99/151 (`measured-here`). The A770,
where the branch is not built, passes the 151.

**The first build was wrong** (`measured-here`, 2026-10-18): the deq wrote a
global K coordinate into the per-super-block tile; the gate caught it (cos
worst 0.83). Fixed before the figures above.

`GGML_OPENCL_F16_XMX_RED` (0022) shortens `ne00` the way 0058's first red
did, so it misreads every row rather than leaving 16 of K out (`code`); it is
a gross red, recorded, not changed here.

## 0059-opencl-intel-conv-3d.patch

`CONV_3D` and `IM2COL_3D` were the pin's two OpenCL gaps for image work
(`docs/milestone-1.1.0.md` §0.1); sd.cpp's `ggml_ext_conv_3d`
(`src/core/ggml_extend.cpp:451`) probes `IM2COL_3D` and falls to
`ggml_conv_3d_direct` (`GGML_OP_CONV_3D`) on a backend that lacks it, and a
backend with neither runs the Wan VAE's 3D convolutions on the CPU. The
patch adds `GGML_OP_CONV_3D` on the **implicit GEMM**, the form CUDA's
`im2col_3d` and Vulkan's `conv3d_mm` use, and the one arcint's
`conv2d_f16_f32.cl` already has: no im2col matrix in global memory. The
kernel folds the depth into the reduction axis (`K = OC`,
`CRS = IC*KD*KH*KW`, `NPQ = N*OD*OH*OW`) and reuses the conv2d tile
(`BS_K 64`, `BS_CRS 16`, `BS_NPQ 64`, `TS_K 4`, `TS_NPQ 8`, f32
accumulation via `mad`) over `[KW, KH, KD, IC*OC]` weights,
`[IW, IH, ID, IC*N]` input and `[OW, OH, OD, OC*N]` output. Two files:
`conv3d.cl` (f32 weights) and `conv3d_f16_f32.cl` (f16 weights, the VAE's),
the input always F32 and the output always F32 (`ggml_conv_3d_direct`'s own
layout, `code`). Admission: `supports_op` takes (f16 weights x f32 input ->
f32) and (f32 x f32 -> f32), at most 2^32 elements a tensor (the kernel
indexes in 32 bits); `GGML_OPENCL_CONV_3D=0` returns false, which is the op
suite's "unsupported" arm, **not** a served baseline: Qwen-Image's graph then
crashes (`GGML_ASSERT(dst->extra)`). The likelier mechanism (`code`, the
scheduler's passes read, not measured): the bias `ggml_add_inplace` after
the conv is pinned to OpenCL by its weight while its destination is a view
into the CPU buffer the moved conv wrote. `GGML_OPENCL_CONV3D_RED=1` ignores
the weights, so every output is 0 -- a gross tripwire, like 0022's f16 red.

**Op suite** (`measured-here`, 2026-10-18, B60 `8086:e211`, OpenCL 0:0,
`test-backend-ops -o CONV_3D`, the CPU reference): **276/276 pass** with the
branch; the red FAILs 270 of the 276 (the 6 that pass are the `CRS = 0`
degenerate shapes, 0 in both arms); `GGML_OPENCL_CONV_3D=0` reports every
case unsupported. The 276 are the tree's own `test_conv_3d` cases -- strides
2/3, padding 0-4, dilation 1/2, `KD/KH/KW` 1 and 3 (a 1-frame 3D plane, a
5x1x3 kernel), `kernel_offset` 1, and the zero-size `KW`/`KH`/`KD` shapes
(`code`).

The 1.1.1 release adds the Wan VAE's own shapes to the suite: one frame
with the causal time padding done beforehand (`ID = KD = 3`, `p = 0`,
`OD = 1`), the 96- and 384-channel blocks, the 16 -> 384 conv1 and the
96 -> 3 head, for both weight types: **284/284 pass** on the B60, the 8 new
cases included; the red passes 6/284 (`measured-here`, the release window). Served: Qwen-Image's
VAE decodes on the card (2.66 s at 512²); the image gate, with the 1.1.1
per-model floors, is in `docs/milestone-1.1.1.md` §9. The campaign is
`docs/campaigns/imggen-wan-vae-conv3d.md`. A lever (`arithmetic`, not
measured): for one frame only the kernel's last time tap meets data, so two
thirds of the VAE's conv FLOPs multiply padding.

**No `IM2COL_3D`**: with a supported `CONV_3D`, sd.cpp's fallback needs
only the direct op; the im2col form would turn the same GEMM into a
`mul_mat` and hand the XMX tiles a flattened matrix, which is the shape to
measure when the VAE's own time is known (`arithmetic`, not measured).

## 0060-opencl-intel-leaky-relu.patch

sd.cpp's ESRGAN upscaler (`src/model/upscaler/esrgan.hpp`, RRDBNet's
`ggml_leaky_relu`) was the one unary the pin's OpenCL lacked, so a mixed
CPU/OpenCL upscaler graph sent `GGML_OP_LEAKY_RELU` to the CPU and
**segfaulted** (`measured-here`, 2026-10-18,
the ESRGAN/RRDBNet tile path's `process_tiles_2d` backtrace); the model itself is fine (the
pure-CPU `--mode upscale` run completed). The likelier mechanism (`code`,
not measured): RRDBNet calls `ggml_leaky_relu(..., inplace = true)`, so the
node moved to the CPU writes through a view of an OpenCL buffer; the CPU
forward itself is not at fault, and any other unsupported in-place op in an
sd.cpp graph would crash the same way. Note the op is **top-level**, not
a `GGML_UNARY_OP` enum: `ggml.c:2787` sets `GGML_OP_LEAKY_RELU`
(`code`). The patch adds it as six variants of the `unary_ext` shape
(`leaky_relu.cl`: f32, f32_4, f16, f16_4, and the strided `_nc` pair), the
negative slope the op's own parameter (`dst->op_params`, float) as the
trailing argument. Admission: f32 or f16, contiguous or strided.
`GGML_OPENCL_LEAKY_RELU_RED=1` negates the slope, so the negative branch
moves -- the red case.

**Op suite** (`measured-here`, B60, `test-backend-ops -o LEAKY_RELU -j 1`,
the CPU reference): **7/7 pass**; with the red, 7/7 FAIL. The 1.1.1 release
adds a row-padded strided case, so the `_nc` kernels run in the suite.
In the release window the strided cases measured wrong (ERR 1.4-1.7 on a
row-padded view, `measured-here`; the kernel and the CPU reference read
right, not attributed), so `supports_op` admits contiguous input only and a
strided input stays on the CPU: 7/7 pass, the two strided cases reported
unsupported on OpenCL, the red 0/7. Serial `-j 1` only:
the driver aborts the process at `-j 4` (a Neo `reference_tracked_object`
abort before any case reports, `measured-here`), and `-o ADD -j 4` is green
in the same window, so it is the LEAKY_RELU cases, not the card, not the
suite -- recorded, not explained.

**Served** (`measured-here`, the same B60 window): `--mode upscale` ESRGAN
4x (the CLI), `RealESRGAN_x4plus.safetensors`, a 512² PNG to 2048², **27 s**
on OpenCL; before the patch the same command segfaulted in the CPU fallback.
Through sd-server's `POST /sdcpp/v1/upscale` (the units, the upscaler
found in `--hires-upscalers-dir`): 512² to 2048² in **27.9 s**
(`measured-here`, the release window).
The upscale is the print route; the campaign is
`docs/campaigns/imggen-leaky-relu-esrgan.md`.

## 0062-mtmd-qwen3tts-gen-valid-frames.patch

Qwen3-TTS's `GEN_WAV` code2wav decoder built a **fixed 72-frame** graph: the
graph took `hparams.wav_tfm_swa` (the decoder's attention window) as its frame
count and the caller padded the codes to that window, so the decoder computed
72 frames for a 16- or a 72-frame flush alike (`clip.cpp`). The per-call cost
was therefore fixed (1.766-1.79 s per flush on a pure CPU backend, 99.6 %
compute) and a chunked flush multiplied it (`measured-here`, 2026-10-19; the
campaign's lever (a), `docs/campaigns/qwen3tts-card-real-time.md`).

The patch makes the graph take the caller's valid frames. `clip_get_graph_builder`
passes the codes-derived frame count to `clip_graph_qwen3tts_gen`'s new
`gen_n_frames` field (default `-1` keeps the window for the no-codes path),
`clip_encode`'s input no longer rear-pads to 72, and the output tail drop
keys on the caller's count under the default (so it no longer fires) and on
`wav_tfm_swa` under the red. **`MTMD_TTS_WAV_PAD=1` is the red case**: it forces
the padded window back, so the pre-patch behaviour and the fixed per-call cost
return and the patch's own test can fail. The default (variable unset) is the
valid frames; there is no arcint flag.

The generic `out_audio` block (`clip.cpp`, shared with other gen-audio models)
keeps its null guard: `n_frames_w` falls back to `hparams.wav_tfm_swa` when
`params->codes` is null, and `n_frames` was already guarded. A `GEN_WAV` caller
with features instead of codes -- the upstream PocketTTS path -- therefore never
dereferences a null codes pointer and computes its own frame count (`wav_tfm_swa`
is 0 for PocketTTS); for Qwen3-TTS a no-codes `GEN_WAV` call aborts at the
pre-existing `GGML_ASSERT(params->codes != nullptr)`.

**CPU** (the same window): the flush becomes frame-proportional, 44 frames
1.11 s, 55 1.37, 60 1.49, 64 1.59 against the fixed 1.77-1.79; the 8-sentence
flush total falls 14.25 -> 10.60 s. **Card** (B60 `8086:e211`, pinned Q8_0
backbone + bf16 mmproj, `--tts-gen-device cpu`, the fixed 8-sentence gate, seed
42, window 72, the full `/v1/audio/speech` route): the padded production arm
runs real-time factor 0.729 (55.66 s process / 40.56 s audio, worst
first-audio 11.0 s) against the valid-frames arm 0.802 (50.58 s; 9.52 s); both
round-trip at corpus WER 0.0455 through `whisper-server -ng` (C4 PASS) and
`codec_eos` is sampled on both. These 0.729/0.802 figures are the first run
(the switch-source build); the final patch binary's red/green run on the same
route measured the padded arm 0.7285 and the default 0.8006 (two builds,
run-to-run variation). The same-binary sample diff is at most
191/32768 (0.0058 of full scale), inside the window-72 card-vs-CPU bar
483/32768 already on record; the patch binary's own arm is bit-identical to the
scratch-switch arm measured first. **Red case (`measured-here`, 2026-10-19, one
binary, the same B60 window):** `MTMD_TTS_WAV_PAD=1` returns the padded arm's
waveform -- identical on all 8 sentences to the production padded arm (real-time
factor 0.7285) -- while the default differs from it at <=191/32768 and runs
0.8006; C4 is 0.0455 on the default. The red breaks the lever's own claim (the
flush scales with the frames), so the test can fail by the repo's rule; the
classification is in `docs/switches.md` (red case).
Sub-real-time still: it is lever (a), not B3. `0062` applies cleanly on the
0001-0060 and the 0001-0058 trees.

Evidence class: `measured-here`. The campaign is
`docs/campaigns/qwen3tts-card-real-time.md`; the milestone record is
`docs/milestone-1.2.0.md` §9.5.

## 0063-opencl-intel-cpu-miss-rows.patch

0056's CPU half of the expert cache's miss path had Strata's multi-token
AVX2 rows for IQ2_S, IQ2_XXS and Q2_0 only, and a layer takes that fast path
only when all three of its projections do (`code`): every layer of the
served Flash-Next GGUFs with an IQ2_XS, IQ3_XXS, IQ3_S or IQ4_NL projection
(the IQ3_XXS file's IQ3_S, IQ3_XXS, IQ2_XS gate/up and IQ4_NL down layers;
all of UD-Q3_K_XL's) ran ggml-cpu's single-token dot products instead. The
patch ports Strata's rows for those four (`src/kernels/cpu/iq_avx2.cpp`,
MIT): IQ2_XS (its own loop, two 4-bit sub-scales) and IQ3_S from the bank's
planes, IQ3_XXS from GGUF blocks (the bank keeps it so), IQ4_NL from its split
bank (scales and quants as two arrays over the whole tensor) through a new
entry point with Q8_0 activations quantized as ggml-cpu's AVX2
`quantize_row_q8_0` does; a per-type block constant (0.125 for the IQ2 types,
0.25 IQ3_XXS, 1.0 IQ3_S); single-token loops shaped like ggml's AVX2 ones
for IQ3_XXS and IQ3_S, where the multi-token form lost at one token. The
caller sends an IQ4_NL split bank to the new entry and aborts on an unsplit
one at the first miss (every USM IQ4_NL tensor is split when written, so it
would be a bank not yet filled), and the generic entry aborts on a type it
has no field table for. `GGML_OPENCL_CPU_MISS_MAX` defaults to 64 (16 before):
with 10 experts a token and at most 4 tokens a verify window that is every
miss on the CPU, Strata's design.

Kernels (`measured-here`, Zen 3, -O3, one 2560 x 640 matrix, one thread):
each equal to ggml-cpu's `vec_dot` on `ggml_quantize_chunk` weights for nt
1..8, both layouts and an odd-block slice (max error 0 observed; the tests'
gate is a 1e-4 tolerance), activations byte-identical to ggml's
`from_float`; deliberate mutations (signs, a scale, the qh bit, nibble
order) fail; at four tokens 2.48x (IQ2_XS), 2.86x (IQ3_XXS), 2.71x (IQ3_S),
1.77x (IQ4_NL) ggml's time, at one token 1.00-1.04x. Reviewed by a second
model (kernels, then the caller: one fix, the unsplit-bank guard now aborts
instead of falling back to a path that would read the split arrays as blocks).

Served (`measured-here`, the B60, Flash-Next with the unit's flags, arcint
v1.2.0 built against the tree without and with the patch, the six answers'
mean decode): IQ3_XXS 33.85 / 33.87 -> 36.13 / 36.08 t/s at the old cap 16,
37.73 at 32, 37.93 at 64; the old path forced on the patch build
(`GGML_OPENCL_CPU_MISS_MT=0`, the red case: the new rows off) 32.83; KL as
served at ubatch 4 0.2178 / 89.66 % against 0.2191 / 89.56 %; prefill 621
t/s in every arm; Paris and the needle in every run. UD-Q3_K_XL 12.48 ->
12.55 at cap 16, 16.68 at 64 and at 128. A byte tripwire: three of
IQ3_XXS's eight answers change wording with the patch, and the forced-old-
path control differs from the base build in four, so the difference is not
only the new rows; not explained, reported (the answers and KL are right).


## 0064-opencl-intel-topk-moe-subgroup.patch

The fused top-k MoE router (`kernel_topk_moe_softmax_norm`: softmax, the k
largest, their sum clamped, the k weights divided by it) ran one work-group
of 256 a token with k rounds of local-memory scans and barriers: 34.1 us a
call, 40 calls = 1.364 ms of the coder's 19.25 ms token on the A770, its
largest small op (`measured-here`, OpenCL profiling). The patch gives each
token one 16-lane sub-group with its probabilities in registers, four tokens a
work-group, and does the k rounds with sub-group reductions only (ties to the
lower expert index, as before). The private array is sized by compile-time
buckets of the expert count (n4 .. n64 floats a lane: 184 experts take n16,
512 n32), because one 64-float array for every count spilled on both cards
(ocloc: A770 ~239, B60 ~87 registers); n16 and the B60's n32 compile without
spills. `GGML_OPENCL_TOPK_MOE_LEGACY=1` keeps the old kernel in the same
binary.

`measured-here` (A770 coder unless noted): the router 34.1 -> 13.5 us a call
(1.364 -> 0.542 ms a token); test-backend-ops TOPK_MOE 424/424 on both cards
with either kernel, three own one-line mutations of the new kernel failing
8/8 of its new cases on both cards; KL against the coder reference
0.023656 -> 0.023594 nats, same top 95.42 -> 95.37 % (one sequence a batch;
four a batch 0.023594 / 95.368 % with 0065 on top); the acceptance task
10/10 four times. The sum over the probabilities runs in another order, so
the greedy texts change and the draft acceptance with them: the coder's
ab.py A/B failed two of its three decode cells on that, while the device time
per verify step fell 1.4-2.5 %. Flash-Next (B60, 512 experts): ab.py PASS,
prefill +0.7 %, decode +0.4 %. Written by DeepSeek (two rounds), verified on
the cards by Claude.

## 0065-opencl-intel-cpy-run.patch

A speculative verify writes the conv-state rollback snapshots as K = n_draft +
1 separate copies a gated delta-net layer (`src/models/delta-net-base.cpp`
`build_conv_state`): each moves three floats a row and is launch-bound, 150
launches and 0.90 ms of a 39.1 ms verify step on the A770 coder
(`measured-here`). The patch matches a run of up to 8 consecutive f32 CPY
nodes (VIEW-type no-ops between them skipped) whose sources and destinations
are views of one tensor each with equal shapes, narrow rows (ne0 < 32) and
disjoint destinations, and runs it as one launch (`kernel_cpy_f32_f32_run`,
a work-item a row writes all K windows). The copy is exact.
`GGML_OPENCL_FUSE_CPY_RUN=0` turns it off.

`measured-here`: the snapshots 150 launches / 0.900 ms -> 30 / 0.175 ms a
step; CPY_RUN_FUSION 5/5 fused and off and all 146 CPY cases on both cards,
three own mutations failing 4/5, 4/5 and 1/5 (the stride one only bites at
two sequences); the coder's ab.py PASS, decode +1.8 to +1.9 %, the same
answers. Written by DeepSeek, verified by Claude.

## 0066-opencl-intel-gdn-tg-verify.patch

The Intel gated delta-net dispatch took the prompt kernel
(`kernel_gated_delta_net_pp`: work-groups that stage chunks of TC = 16 tokens
in local memory) for every batch over one token, so a 5-token verify paid the
staging for one partial chunk: 35 us a launch against 11.7 us for the
one-token generation kernel (`measured-here`, the coder's re-count). The
generation kernel already loops over the tokens with the state in registers
and writes the K snapshots a token, with the same arguments; the patch sends
batches up to 8 tokens to it. `GGML_OPENCL_GDN_TG_MAX=N` moves the limit (1 =
before). Four test cases at the verify shapes.

`measured-here`: GATED_DELTA_NET 53/53 on both cards at limits 1, 8 and
1,000,000 (the generation kernel for every case, 2,048-token prompts
included); a mutation of its snapshot slot passes at 1 and fails 5 cases at 8.
The coder's GDN time a verify step 1.044 -> 0.674 ms, ab.py PASS, decode +1.1
to +1.2 %, the same answers; with 0065 on the agent (dense 27B, B60, MTP 5)
decode +2.3 to +2.4 %, the same answers, prefill equal; with 0064 and 0065 on
Flash-Next ab.py PASS, prefill +0.7 %, decode +1.2 %.

## 0067-llama-embeddings-nextn-skip.patch

With a draft vocabulary (`--llama-mtp-vocab`) arcint's MTP context runs with
unmasked nextn embeddings (a row for every token, no logits), so its prompt
catch-up copied a row a prompt token to the host that nothing reads: 41k rows
x 2,048 floats on the coder, over the A770's PCIe 3.0 x4 link, -0.8 % prefill
(`measured-here`); Flash-Next's rows (4 hyper-connection streams) cost it
-0.9 %. `llama_set_embeddings_nextn_skip(ctx, true)` leaves the rows on the
device for the following decodes without changing the graph (no re-reserve,
which toggling the mask would force). arcint sets it around the draft
context's catch-up decodes (`src/exec/llama_spec.cpp`).

`measured-here`: the coder with the vocabulary, ab.py with vs without the
patch: prefill-41k 723.25 -> 729.20 t/s, decode equal; Flash-Next with the
vocabulary: prefill equal to no vocabulary (622.65 / 622.88).

## 0068-llama-prefill-slot-loan.patch

The prefill-slot-loan (`docs/campaigns/prefill-slot-loan.md`): a prompt borrows
the expert cache's tail layers' slots, the compute buffers are reserved at the
prompt's ubatch, and the return re-reserves at the resident ubatch, takes the
tail slot buffer back, refills every lent slot from the lend bank and verifies
it against the file. The design is `docs/campaigns/prefill-slot-loan-design.md`
(evidence class on every disposition). The mechanism, `code`:

- the tail slot tensors live in their own buffer at load
  (`ggml_backend_buft_alloc_buffer_n`, the tensor list -- no byte range, so
  `n_slots` varying per profile rank and a Q8_0 bank in the same context do not
  matter); a loan frees it and `sched_reserve()` at the prompt's chunk allocates
  the compute buffers where it was. The tensors' `buffer`/`data`/`extra` are
  cleared before the free because the OpenCL `free_buffer` does not clear them,
  and the re-allocation runs `init_tensor`, which rebuilds the device extras.
- the lent layers' ex-slot experts stay reachable in a **lend bank** in host
  memory beside the bank, and the graph builds a third MUL_MAT_ID branch over it
  (`llama-graph.cpp`), so every tensor a loan graph sees stays under the grouped
  kernel's `MAX_EXPERT 512` local table: no compile define is raised and the
  GGUF fallback Strata measured at -26 % prompt with 18 % uncovered is not paid.
- `n_ubatch_reserve` keeps the resident reserve at the unit's 2,048 while the
  loan's reserve is the chunk; the effective ubatch is capped by it, so a graph
  larger than the reserve cannot reallocate through the buffer type and escape
  the loan.
- `sched_reserve()` replaces the scheduler (`llama-context.cpp`), and the expert
  cache binds to the first one it decodes (`llama-expert-cache.cpp`): a loan
  rebinds (synchronize, zero `ids_pending`, adopt), so adapt runs after the
  second request too. A swap in flight is finished before the lend, not polled.
- the arcint side is an RAII scope around `prefill_locked` only, so the last
  prompt token's decode runs after the refill's wait, and every exit (a throw,
  wave2a's cancellation) gives the slots back; a request start refuses while a
  loan is out.

Second review round (`code`, `docs/campaigns/prefill-slot-loan-design.md` §10):

- the lent branch is tested **before** the `n_tokens <= 8` decode branch: while
  lent the slot tensors have no buffer and no data, and a 1-token tg graph (every
  `sched_reserve`) or a prompt's 4-token last batch would otherwise allocate them
  into the compute buffer and read uninitialised bytes.
- the lend bank is the slots' mirror: both swap sites (the exchange, the adapt
  queue's phase 2) write the incoming expert's rows into `lend[k]`, host to host;
  a swap in flight is settled (copies issued and waited, both table halves
  applied), not only polled.
- the refill reads back through `get_slices` (which restores a planar type's
  layout) instead of a raw USM pointer, one staging buffer and a wait a layer:
  `set_tensor_slices` converts GGUF blocks again, so the planes would verify
  wrong on every IQ3_S loan.
- the return's re-reserves cannot throw (`slot_loan_reserve`): they run from a
  destructor, where a throw is `std::terminate`; a failed return leaves the loan
  out and the request-start check refuses.
- `decode`'s cell budget is `ub_cap x n_kv_reserve`, not `n_ubatch x
  n_kv_reserve`: under a loan the two differ, and a budget off `n_ubatch` takes a
  graph several times the reserve and reallocates through the buffer type.
- a lend that cannot be published puts its tables back (the patch's own
  `src/llama-loan-book.h`, shared with the device-free test), and the tail buffer
  has its own owner: it is taken out of, and never added to, the list the
  destructor walks.
- `slot_loan_bytes` has a "no parameter set" sentinel (`LLAMA_SLOT_LOAN_BYTES_ENV`),
  so a tool's explicit 0 is off and `LLAMA_SLOT_LOAN_MIB` cannot turn the loan on;
  `.lend.bank` tensors stay out of the bank stream's learned order.

Third review round (`code`, `docs/campaigns/prefill-slot-loan-design.md` §11):

- an adapt's pending phase is never cleared: a planned, unissued phase (an
  aborted or failed ubatch returns before `after_graph`) is issued behind a
  fence and settled, so the phase-1 spares come back and a phase-2 slot is not
  orphaned (the refill would copy it back and verify it against -1, zeros).
  The phase state and its transitions are this patch's `src/llama-loan-book.h`,
  shared with arcint's device-free test.
- the settle fences before the phase it issues and between the phase-1 tables
  and the phase-2 copies, as `progress` does: the copies are on the adapt queue,
  unordered against the compute queue.
- the lend bank's mirror is once a swap (all three k in one call), outside the
  k loop: 0.14 GiB a 96-swap adapt instead of 0.42, on the decode thread.
- a failed resident re-reserve after `sched_new`, or at the loan's return,
  poisons the context (`slot_loan_poison`): `llama_context_slot_loan_out` reports
  it, so the request-start check refuses rather than prefilling on a scheduler
  with no reserve for its shape. A lend whose undo leaves the tables differing
  marks the cache unusable the same way (`loan_broken`).
- the lend, the refill and the drop skip a cache with no lend set
  (`lend_bytes() == 0`, the other card under a layer split) instead of turning a
  good loan or a good return into a refusal.
- the refill restores the zero slot (row `n_slots`), which init wrote and the
  slot branch reads, and the verify covers `s <= n_slots`; `lend_undo` checks
  the tables again after its undo (its read is the wait) and marks the cache
  unusable if they still differ.
- the flag rule lives in the book header and `init` (`resolve_lend_bytes`): only
  `llama_model_default_params`' sentinel reads `LLAMA_SLOT_LOAN_MIB`; an explicit
  0 is off. The lend's comment no longer calls the table writes blocking.

Fourth round (the first card run, `code` and `measured-here`;
`docs/campaigns/prefill-slot-loan-design.md` §13):

- IQ4_NL is lendable. The served Flash-Next IQ3_S has every layer's
  `ffn_down_exps` in IQ4_NL; the old `no_split()` excluded it, so no layer was
  lendable and the selection left non-chosen eligible layers marked while no
  tail context existed -- `ggml_new_tensor_3d` in a null context at load (the
  segfault, `measured-here`). The swap mirror now reads the bank slice through
  `get_tensor_slices` and writes the lend slice through `set_tensor_slices`, the
  converting pair the refill already uses, so the split (scales, quants) layout
  is never copied raw. A Q8_0 layer stays non-lendable (its bank lives in the
  slot buffer).
- the lend set is computed from scratch: every layer `lendable = false`, the
  scan starts at the tail, skips a non-lendable layer or one that does not fit
  the remaining budget and continues earlier (the tail buffer is a tensor list,
  so no contiguity), and only the chosen entries are set. Every loop walks that
  set. Nothing chosen means the loan is off -- no tail/lend context, every slot
  in the ordinary buffer, one line `the slot loan lends nothing: <reason>`.
- the poison policy is the operator's `poison-exit` (2026-10-29):
  `llama_loan_book::poison_exit` writes one ERROR line naming the step and the
  counts/layers and exits non-zero, so the unit restarts clean. The context's
  failed resident re-reserves and the cache's unverified/unallocatable refill
  and differing undo call it; staying up instead is that one function's body.
- `slot_loan_begin` initializes the memory module before it replaces the
  scheduler, so a refusal there leaves the live scheduler intact; the failed
  loan-reserve path reserves the resident chunk again after the refill, as the
  return does.
- `llama_context_slot_loan_state()` ("off", "armed", "out", "poisoned",
  "broken") is what the request-start refusal names and what `/props` publishes
  under `slot_loan.state`.
- `llama_loan_book::ubatch_cap()`/`cell_budget()` keep the loan-off decode path
  identical: with `reserve_ubatch == n_ubatch` the cap is `n_ubatch` and the
  budget is `n_ubatch x n_kv_reserve`.

Fifth review round (`code`, `docs/campaigns/prefill-slot-loan-design.md` §14):

- the refill writes a slot tensor whose card layout `set_tensor` builds (IQ4_NL
  and Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/MXFP4/Q4_K/Q5_K/Q6_K) **whole**, not by slices:
  the tail buffer's re-allocation installs the plain `ggml_tensor_extra_cl`,
  while the split layout (IQ4_NL's d/q sub-buffers) exists only after a
  whole-tensor `set_tensor` write -- as init does. A slice write, the verify's
  `get_slices`, an adapt copy's regions or a kq weight arg would cast the plain
  extra to the split one (`q = data_device`, `d = actual_size`: a CL_CHECK
  abort or silent corruption), on every lent layer of the served model, whose
  down is IQ4_NL. The refill assembles the `(n_slots+1) x nb2` image (the lend
  bank's rows plus the zero slot) and writes it with a blocking
  `ggml_backend_tensor_set`; planar types (IQ3_S, IQ3_XXS, IQ4_XS, IQ2_*, Q2_0)
  keep the plain extra and the slice path.
- `poison_exit` is `std::_Exit(70)`, not `std::exit`: static destructors must
  not run from the generation thread under arcint's mutex (the OpenCL device
  contexts and the CPU-miss coordinator can hang there, and the unit only
  restarts on an exit). The line goes through the llama log callback first
  (`log_line`) and then on stderr with a flush.

Round 7 (`code`, `docs/campaigns/prefill-slot-loan-design.md` §15):

- the refill copies **regions from the lend bank**, not staged host bytes. The
  bank holds the CARD'S layout (it is filled through `set_tensor`/`set_slices` at
  load, and every adapt's mirror copies bank bytes already in that layout), so a
  refill is one non-blocking region copy a tensor from USM host memory into the
  new slots -- `ggml_backend_opencl_lend_write` (three new proc addresses:
  `lend_write`, `lend_zero`, `lend_read`) -- with no `(n_slots+1) x nb2` host
  image, no whole write, no temporary upload buffer and conversion kernel, no
  plane/GGUF conversion, and one wait at the end instead of one a layer. The
  measured 2.66-2.86 s for 6.20 GiB (~2.3 GB/s) against the link's 14.3 GB/s was
  that staging and conversion; the trace line now prints its phases
  (`stage, write, convert, tables, wait`) so a card run shows where the time goes.
- the zero slot (row `n_slots`) is filled from a device zero page by
  `lend_zero` (a copy a region), so no host bytes are held past an enqueue.
- the split extra is rebuilt **without bytes**: the re-allocated tail buffer's
  `init_tensor` installs the plain `ggml_tensor_extra_cl`, and `lend_write` calls
  `ggml_cl_iq4_nl_extra` before it copies (the `iq4_nl_split_tensors` set records
  which tensors have it). That was the fifth review's whole write; the write path
  is gone, so no type needs to be named in a rule -- BF16's f16-on-card layout is
  in the bank's bytes and the copy is exact without it being added anywhere.
- the refill's check: by default every row of every lent tensor (the zero slot
  included) is compared **with the lend bank, byte for byte, in the card's
  layout** (`lend_read` on both, no conversion, no GGUF read) -- the bank's
  provenance to the file is the load's new `lend bank check` over every lendable
  row against a read of the file, once, plus every mirror. `LLAMA_SLOT_LOAN_VERIFY=file`
  is the gate's direct form (GGUF bytes against the file, batched: one read-back
  a tensor instead of a blocking read a row), `=0`/`off` is the timing arm; a word
  that is neither is the default, never silently off.
- Round 8 (`code`, §16): `lend_read` emits a tensor's regions one after the
  other (`[d x n][q x n]`), so the per-row compare of the round-7 default read
  the slot's q region against the bank's d region once it passed the scales --
  ~8/9 rows "differed" on a correct copy and exited 70. Round 9, item 3: both
  sides are read with the same `n` and compared **row by row inside each region**
  with the region's own row stride (`lend_compare_rows`); `bad` is a row count
  like `n` (round 8 added one per tensor while the line said rows) and the first
  mismatch names its layer, region and row. The zero slot rides in the same pass
  as the slot image's extra row.
- `LLAMA_SLOT_LOAN_MUTATE=1` corrupts one DATA byte of a lent row after the
  copies (`lend_perturb`), which either reference catches and the poison exit
  fires; the round-7 form skipped a copy, left the slot's split extra unbuilt and
  the file check's `get_slices` cast a plain extra (an Intel driver segfault, no
  ERROR line). `=3` skips the lent graph's measurement; after the reserve the
  actual buffer sum is compared with the sizing's expectation
  (`reserve_matches`), so the mismatch is deterministic instead of depending on
  the card's reserve failing. Round 9, item 2: the comparison is `actual <=
  expect`, not equality -- a buffer shared between bufts counts differently in
  `sched_reserve_size` and `ggml_backend_sched_get_buffer_size`, so a healthy
  reserve can legitimately sum below; only actual above the expectation is the
  defect. Round 10: round 8's `=3` skipped only the lent flag, premised on one
  more MUL_MAT_ID dst; the code shows the lent branch REPLACES the slot branch
  and drops both down tables (`llama-graph.cpp:2510-2521`), so the skip RAISES
  `need` and the check passed on the card (the loan lent and returned normally,
  `measured-here`). `=3` now measures the graph at the resident chunk while the
  reserve allocates the prompt's chunk (`measure_tokens`), a real under-count
  that `actual <= expect` rejects.
- lendable is only what the region copy is exact for (`region_copyable`):
  plain-no-extra types, the planar set and IQ4_NL. The bank stores the types
  `set_tensor` gives a specialised extra (Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, MXFP4,
  Q8_0, Q4_K, Q5_K, Q6_K) raw, and BF16 raw while the card converts it to f16; a
  copy there is the old crash or wrong weights with a passing bank check.
- the plan needs `need + 384 MiB <= lend` (`size_fits`, round 9 item 1 back to
  round 7's rule): the freed lend bytes are real and the reserve is what stays
  free beside the plan. Round 8's `headroom_fits` read `ggml_backend_dev_memory`,
  whose OpenCL form is `free = global_mem_size - 1 GiB` -- a constant, so the
  check could never fail; it is gone from the loan path.
- the poison exit drains the compute, adapt and bank-stream copy queues first
  (`ggml_backend_opencl_drain`), bounded by a 5 s watchdog, so `_Exit` does not
  tear the address space down under in-flight work (the `xe` page fault and GuC
  engine reset on both exit-70 arms). Round 9 item 4: the watchdog's
  `std::thread` construction is guarded (`watchdog_spawn`), because a throw there
  under the policy would reach `std::terminate` and the exit would not be 70.
  Round 10: the drain alone was not enough on `MUTATE=4` -- the tail buffer was
  freed at the lend while the adapt/bank-stream queues could still hold copies
  over it, and the exit's drain then executed them against released memory (the
  B60's `xe -ENOENT` and GuC reset, `measured-here`). Every tail-buffer free now
  drains first (`drain_now`, `drop_tail_buffer`) and the return's poison path
  drains before it poisons, so no queued command outlives the memory it targets.
- Round 11: `sched_reserve`'s own `synchronize()` leaves the adapt queue out
  (`ggml-opencl.cpp:9062`), while `sched_new` replaces the live scheduler and the
  galloc frees its buffers at once. Every scheduler replace on the loan path now
  follows a full drain: `slot_loan_reserve` drains before `sched_reserve` and
  again on a throw, `slot_loan_begin` drains before its own `sched_new`, and a
  failed/skipped return synchronizes the scheduler and drains before the poison.
  The cache destructor and `drop_tail_buffer` drain whenever the loan is armed
  (`lend_bytes_ != 0`, before freeing anything; unchanged with the loan off). A
  throw inside `slot_loan_begin`'s sizing takes the resident re-reserve and the
  rebind instead of reaching `Scope::begin` on an unreserved scheduler with a
  dangling `owner_sched`; a failed tail allocation clears its dangling tensors
  before the poison. `ggml_opencl: drain: N device(s), compute, adapt and
  bank-stream queues finished` is the card's proof line (design §19, R36-R39).
- Round 12: the drains stay, but the surviving `MUTATE=4` fault was not queued
  work (`code`: no adapt-queue command targets a galloc buffer, and OpenCL
  retains a `cl_mem` until its commands finish). The card's discriminator was
  idle time (`measured-here`: `MUTATE=1` is idle hundreds of ms while the refill
  verifies, no fault; `MUTATE=4` exits microseconds after the last GPU work and
  faults 3 of 3; direct submission off: no fault, 0 of 3). `std::_Exit` runs no
  destructor, so the runtime's direct-submission rings are live when `xe` closes
  the fd; `poison_exit` now waits `LLAMA_SLOT_LOAN_EXIT_IDLE_MS` (default 2,000
  ms, clamped to 4,000 under the 5 s watchdog) after its drain. The drain line
  is `GGML_LOG_WARN`: arcint shows WARN and above, and the round-11 card window
  never saw the INFO line. Card: the default does not fault 3 of 3, idle 0
  faults 1 of 1, idle 100 does not 1 of 1 (design §20).
- Round 13: the wait line is written before the sleep, so the log stated an asked
  value rather than what happened; `poison_exit` now writes a second WARN line
  after the sleep with the measured elapsed, and the watchdog's thread writes
  ONE constant line with `write(2)` before `_Exit` (async-signal-safe; the drain
  may be hung and the log route sits on the lock it holds). `off`/`OFF`/`false`/
  `no` all read as 0 (round 12's `strcmp("off")` silently took the default for
  an `OFF` arm). Design §21, R44-R46.
- Round 14: `LLAMA_SLOT_LOAN_MUTATE=5` is the red case for the branch the KL
  gate tests -- `mutate_loan_bank_row` moves one lent layer's lend-bank row
  index by one in the host table before its non-blocking write, so the card reads
  the neighbour's rows during the prefill; no verify compares a table row (the
  refill's check compares slot bytes with the bank's), so the run refills,
  returns and scores, and the WARN line names layer, expert and both rows. Every
  earlier red arm exited before scoring, and `MUTATE=1` + `VERIFY=off` was a
  false red (its corruption lands after the prefill). Design §22, R49.
- Round 17: the fixed-margin rule (operator's decision 2026-10-30):
  `size_fits` keeps the 384 MiB peak reserve and adds a fixed 1 GiB under the
  card's total (`kCardMargin`, `kLendMargin` = 1.375 GiB), and
  `extend_lend_layers` grows the ask-bounded lend one candidate tail layer at a
  time until `lend - need >= kLendMargin`, stopping at the layer the margin
  needs (`code`, design §24b). In the cache (`extend_lend`), a grown layer's
  lend bank is allocated in host memory and filled from the rows its slots hold
  now, then its resident slot buffer is freed; a candidate outside the ask's
  pick keeps its slot tensors in its own small buffer (the ask's pick stays in
  the one tail buffer) until growth frees it, and the refill allocates it back.
  The tail bounds the growth; when it runs out `slot_loan_begin` refuses into
  the resident path with one WARN naming `need`, the lend and the fixed margin.
  The `loan out` line now prints the margin (`lend - need`) beside the fixed
  one. Device-free: `tests/test_loan_book.cpp` R58-R60.
- Round 24: the card's fdinfo discovery. The OpenCL runtime opens TWO xe clients
  per card in one process (a probe client and the real one, `measured-here`,
  round 22's window), so round 20's unique-pdev match never engaged and the
  kernel query, the fdinfo figure and the eviction guard were all absent. A
  card's figure is now the SUM of `drm-resident-vram0` and `drm-resident-gtt`
  over the distinct `drm-client-id`s whose `drm-pdev` is the device's (dedupe by
  client id only; the scan's job), and `DRM_XE_DEVICE_QUERY_MEM_REGIONS` is
  taken on any render fd of that pdev -- `used`/`total` are device-wide.
  `match_fdinfo` is gone; `sum_fdinfo` returns `FdinfoAgg`, `read_resident`
  keeps the ggml-sum fallback and tripwire. The load probe stays in
  `llama_expert_cache::init` after the cache's own buffer allocations and the
  weights, so the backend's fds exist; its line prints the summed
  `fdinfo vram0 ... over N client(s)`. Device-free: R99-R102.

`measured-here`: the card's numbers are the campaign's gate (stage 3); this
patch is stage 1, whose pass bar is the trace lines and the red runs in
`docs/campaigns/prefill-slot-loan-design.md` §7, §10, §11, §13, §19, §20, §21,
§22 and §24.
## 0069-expert-route-trace.patch

A production option, off by default: it records the expert cache's routing
into binary trace files so a replay can price predictors offline before any
serving change (`docs/campaigns/expert-foresight.md`). Arcint's
`--llama-expert-route-trace DIR` sets `LLAMA_EXPERT_ROUTE_TRACE` before the
model's load; unset, the cost is one `getenv` at the cache's init, one cached
bool the graph builder and the accept writer ask, and nothing else — no
allocation, no read-back, no write (`code`, `llama-expert-route-trace.cpp`).
A trace with token ids is the prompt and the answer in another form: whoever
enables it stores every request's content on the serving host.

A file a process, rotated at a size cap (`LLAMA_EXPERT_ROUTE_TRACE_MAX_MB`,
default 256). `DIR` gives `expert-route-<pid>-<unix-time>.000.ert1`,
`.001.ert1`, ...; a path that is an existing non-directory is a *base*: the
file already sitting there is skipped (`O_EXCL`) and the sequence starts at
`.001`, then `.002`. Every file is opened `O_EXCL` and a fresh name
is chosen, never appended: a restarted unit with a reused pid gets a new base,
and a file that already exists advances the sequence, so a second header never
lands in an old file (`code`). A `DIR` that does not exist is created; if that
fails the trace is refused with one `WARN`, not treated as a file path. A
non-numeric or non-positive `LLAMA_EXPERT_ROUTE_TRACE_MAX_MB` falls back to
256 MB with a `WARN` (it used to parse as 0, rotating every record). Every
file repeats the header `ERT1`, version, the model's `n_layer`/`n_expert`/
top-k, and after a rotation a `LAYERS` record of the cache's residents at that
moment — so each file replays alone; concatenating the files gives the
unrotated trace's records. Nothing is ever deleted by arcint. A pre-round-6
file with a second header at its old tail still replays: the reader
resynchronises on a later `ERT1` header with the same geometry (`code`).

Records (`src/llama-expert-route-trace.h`): a cache's layers with their
resident slots at load; each graph's kind (prompt, decode, verify, MTP
draft), its first position, its token ids and every layer's routed ids in the
router's rank order, token-major; and every residency change the cache
applies, so a replay rebuilds the run's own resident set. Version 2 adds the
graph's first position; the kind is the call's shape, not the token count — a
prompt call does not output every row, a decode or verify call does — so a
prompt's 4-token checkpoint tail is a `PROMPT` record, not a `VERIFY`; the
decision is the pure function `llama_expert_route_graph_kind` the device-free
selftest checks. A `DRAFT` record is an MTP draft context's own decode: the
draft head is a separate model without an expert cache, so while the trace is
on its one MoE block's `selected_experts` is a graph output. The record
carries the MTP block's INPUT tokens (step 0 the accepted target token, then
the previous draft) — not the drafted tokens — and that block's routed ids;
the copy goes through the OpenCL backend's batch hook
(`ggml_backend_opencl_get_tensors_async`, the entry point the cache already
uses; `ggml_backend_tensor_get_async` is a blocking fallback on OpenCL because
`iface.get_tensor_async` is NULL there), is issued behind the graph with no
wait and written at that context's `synchronize`, with one wait at a decode
that starts before a synchronize.
The last graph of each request is written at the request's end
(`llama_expert_route_trace_flush`, from arcint's generate and its destructor,
under `mu_`), because the cache otherwise writes a graph only at the next
graph's advance.

A verify's accepted drafts are not in the trace — the cache cannot see the
sampler — so arcint writes an accept sidecar beside the *current* trace file:
one line a decode graph, `verify <ord> pos <p> rows <n> accepted <a>`,
aligned by the graph's first position and checked against its token count.
The sidecar follows the rotation, so each trace file has its own: the cache
rotates at `after_graph`, BEFORE arcint writes the verify's accept line, so the
line and the graph record land in the same file. A graph record's rotation is
taken only there: while a record is pending (an MTP DRAFT record can push the
cap over between `after_graph` and the writing `advance`) no rotation happens,
so the line never parts from its record; the rotation waits for the next
`after_graph` (`llama_expert_route_trace_rotate_now`, `code`). With
`LLAMA_EXPERT_CACHE_ASYNC_IDS=0` the switch is respected: the ids are read
with the wait it asks, even while the trace is on.

The cost, `code`: on a decode step the ids are the ones the cache already
reads back for its usage counts (`ids_pending`, the read issued behind the
graph by `after_graph`), so no new device wait is added; the trace copies
those host bytes into one record (a record is a few KB: top-k × tokens ×
cached layers × 4 bytes) and writes it once. A prompt graph has no such
read-back, so a traced prompt gets one host read and one wait at the next
ubatch boundary. An MTP `DRAFT` record is a read behind the draft graph (the
batch hook, no wait), written after that context's synchronize; a decode that
starts before a synchronize waits once, never a graph. A failed write logs one `WARN` and
clears `trace_on`, so no read-back continues.
The flush at a request's end is one sync, only when a graph is pending and
the trace is on. Off: none of these paths run.

The MTP block sits **after** the FFN's `call early so that topk-moe can be
used` weights expand, and never before it (`code`): a graph's node order is
the expand order, and 0064's `ggml_opencl_can_fuse_topk_moe` wants the
router's ten nodes (SOFT_MAX, RESHAPE, ARGSORT, VIEW, GET_ROWS, RESHAPE,
SUM_ROWS, CLAMP, DIV, RESHAPE) consecutive from SOFT_MAX, which that expand
is what appends them. Placed before it (round 9's build), the trace's own
expand appended SOFT_MAX, ARGSORT, VIEW, CONT first, the weights expand
followed with the rest, the scan met ARGSORT where it wants RESHAPE, and the
MTP draft's router ran unfused -- a drafter whose softmax sum order differs
from the served one (`docs/campaigns/fusion-coder-night.md`). After the ten
nodes the CONT keeps the fusion: `ggml_can_fuse_subgraph` exempts the nodes
in `outs`, and the VIEW it consumes is `outs[0]`, the same placement the
cache's own `ids_out` below has.

Evidence on the card (`code`, the line's print is unmeasured): the ids CONT
is named `ffn_moe_ids_trace` -- set **after** the graph callback `cb(ids_out,
"ffn_moe_ids_trace", il)`, because the callback renames a layer's tensor
`"%s-%d"` (llama-context.cpp's `ggml_format_name`) and a name set first would
be `ffn_moe_ids_trace-<il>` (`code`, round 11). `ggml_cl_topk_moe_fused`
compares the 18-byte **prefix** (never the NUL-terminated form, which is 19
bytes against an 18-byte name) and prints, once per process at WARN,
`opencl: topk-moe fused in the MTP trace graph (n_expert N, n_tok T)`.
WARN because arcint routes ggml INFO to verbose and drops it at the units'
default level; once because a per-dispatch line floods a month-long trace.
The line is the check that the trace-on drafter's router is the served one;
a broken node order makes `can_fuse` fail and the line is absent (and its
absence also covers the trace being off). Only an MTP trace graph is
reported, and its scan is decided once per graph, not once per fused
dispatch, so a traced target graph does not flood the log (`code`, round 11).

Size: a prompt record is `n_tokens × top-k × n_layer × 4 B`, about 1.9 KB a
prompt token (100k prompt ~190 MB), so DIR belongs on a dataset with room, not
the root filesystem.

Device-free: `tests/python/test_expert_route_trace.py` reads the committed
fixture `tests/python/fixtures/expert-route-trace-v2.ert1`, which is
`tools/expert_route_trace_selftest.cpp`'s output compiled against this patch,
and rebuilds the same bytes with the replay's writer; with
`EXPERT_ROUTE_TRACE_SRC=<llama tree>` set it also compiles the selftest,
requires the writer's output to equal the fixture (so the fixture cannot
drift), and runs the rotation red case: a small cap gives two files, each
replays alone, and their concatenation equals the unrotated trace (`code`).
Round 6's cells add: a second header resync, a `.000` argument finding its
`.001` siblings, a later file's `LAYERS` updating `residents`, one missing
last graph a killed process, a rotation putting the accept line and graph in
one file (red: line before rotation), a repeated open not appending, a missing
DIR created, and a non-numeric max falling back (`code`; the red mutations
were seen failing). Round 7 adds: a pending graph record never rotates away
from its accept line when MTP `DRAFT` records cross the cap between
`after_graph` and the writing `advance` (red: rotate while pending), and a
write failure stops tracing with one `WARN`, `opened` cleared and no sidecar
(red: keep `opened` set after the failed write) -- both drive the selftest's
`EXPERT_ROUTE_SELFTEST_PENDING_ORDER` / `_WRITE_FAIL` arms (`code`).
Round 9 adds the MTP ids' expand cell: the block's expand sits between the
output mark and `set_route_ids` (red: delete it, the buffer stays null and no
DRAFT record is written), and round 10 extends that cell with the placement
assertion -- the weights expand appears before the trace block (red on
round 9's placement: seen failing, then green after the move). Round 11 adds
two cells: the fused-router name is set after `cb` and the backend scan is the
18-byte prefix (red: name before `cb`, `sizeof` without `- 1` -- both seen
failing), and the line is a WARN held by a static once-per-process flag with
its graph scan decided once per graph (red: `GGML_LOG_INFO`, or a truthy
once-guard -- both seen failing). These source cells read `src/llama-graph.cpp`
and `ggml/src/ggml-opencl/ggml-opencl.cpp` and need `EXPERT_ROUTE_TRACE_SRC`;
the recipe's ctest does not run this python file at all.
The reader is `tools/expert_cache_replay.py`, which still reads the
2026-10-15 `ECT1` format of the dropped `LLAMA_EXPERT_CACHE_TRACE` knob.

## 0070-expert-cache-save.patch

A production option, off by default. Writes the expert cache's learned routing
as Strata's STRP profile at teardown, so a warm start can load it with
`--llama-expert-profile`. Arcint's `--llama-expert-cache-save FILE` sets
`LLAMA_EXPERT_CACHE_SAVE` before the model's load; unset, one `getenv` at the
cache's init and no allocation, no write. The file is Strata's whole STRP file: header, ranked pairs, then the
trailing layer x expert rank table (`expert -> rank`, -1), which the reader
stops before (`code`; the C++ writer's bytes equal the scope tool's Python
writer byte for byte).

Two ranking rules, both Strata's (`code`, commit 3d6f1718). The default `xe`
is its SYCL/xe save (`save_profile`, `sycl/src/program/generate.cpp`): the
raw routing counts mixed with the start's ranking as pseudo-counts,
lambda 2, `STRATA_PROFILE_PRIOR_ENTRIES` 800,000 routed entries by default, a
prior pair at rank r worth `prior_entries * 2 * exp(-2 * r / n) / (1 -
exp(-2)) / n`; a short conversation does not wipe the prior. `--llama-expert-
cache-save-rule cu` selects its `rank_learned_profile` (`src/core/
expert_cache.cpp`): resident pairs by heat, counted non-residents by heat,
then the rest, ties by the prior's rank then the index. The save ranks from a
per-pair `heat` in Strata's xe unit, raw routing counts:
`llama_expert_profile_add_heat` in the cache's `count()`, one a routed pair,
beside `usage += 1`, in double (a month at ~33 t/s passes float's 2^24 on the
hottest pairs). It is allocated only when the save is on. Counting in
`count()`, not in `adapt()`, keeps it accumulating with
`LLAMA_EXPERT_CACHE_ADAPT_EVERY=0`; a decayed sum would be 1/(1-decay) larger
(3.33 at decay 0.7) and weigh like ~240k prior entries instead of the 800k the
prior is measured with. The save's success and failure lines are `WARN` (the
units' default level drops `INFO`), and the xe line prints the routed entries
counted next to the prior, as Strata's stderr line does.

The destructor saves after `adapt_finish` has landed the pending copies but
without `progress()`, so the host tables stay at the last phase: every pending
incoming expert is marked resident (the copy in flight counts as done) and the
outgoing one stays resident only in phase 1, cleared in phase 2 when the host
table has freed its slot. A layer split has a cache a card and one path cannot
name both; nothing is written there with one `WARN`. arcint refuses at config
a save path equal to `--llama-expert-profile` (the save would overwrite the
file the run reads) and an unknown rule name.

Device-free cells: `tools/expert_profile_save_selftest.cpp` (round trip and
same placement, truncation, version before geometry, prior-rank tie-break,
pending incoming/outgoing in both phases, xe prior mass, heat unit and heat
width) and `tests/python/test_expert_profile_save.py` (the C++/Python byte
equality, and that the heat call sits in `count()` not `adapt()`), wired into
CTest as `expert-profile-save`. The Python cell is stdlib `unittest` and CTest
runs it with `python3` alone: pytest is not a build dependency. Red mutations
and the recipe line are in `docs/campaigns/research-concept-mining/audit.md`
§14a.

Patch order: this patch and 0069 both touch `llama-expert-cache.{h,cpp}` (the
destructor, the layer struct) that the prefill slot loan (0068) already
changes. Both are therefore regenerated on top of 0068 and exported in series
order; the whole 0001-0068, 0069, 0070 series applies on the pin and the
recipe passes (`measured-here`, the recipe line in audit 14a).

## 0071-opencl-intel-moe-int8-dpas.patch

`MUL_MAT_ID` on int8 DPAS for the prompt's routed experts on a sub-group 16
device (the B60's shapes first): the IQ3_S grid and IQ4_NL values decoded
in-kernel to s8 pieces, q8_1-style int8 activations per 32, one s8
8x16x32 DPAS a 32-weight block per 8-pair group, scaled in float by
(weight `d * (1 + 2 sc)`) x (activation scale). It is Strata's fused form
(`code`: `sycl/src/prefill/esimd_moe.dp.cpp` `gate_up`/`down`, the ESIMD
decode of 16 rows a lane and the q8_1 `quantize_rows`; `docs/INTEL.md`'s
+21 % at 1,024 tokens, +12 % at 4,095 on a B70, KL mean 2e-4 nats, top-1
99.7 %, all `paper`) in OpenCL. The existing grouped-by-expert route is
reused unchanged; the int8 path routes with `tile = 32` so one route cache
entry serves a layer's gate/up/down as before.

The OpenCL port differs from ESIMD in three places (`code`):
- the DPAS operands are built with `intel_sub_group_block_read_us8` from a
  local 256-weight super-block staged as [block][pair][32 bytes] (a 8-pair
  group of a block is 256 contiguous bytes; `mul_mm_kq_i8.cl`), where Strata's
  ESIMD passes raw register blocks;
- the weight planes are the tile kernel's plane form (patch 0022: IQ3_S
  `d | qs[64] | qh[8] | signs[32] | scales[4]`, IQ4_NL code and scale
  planes), IQ3_S's qs and signs read as aligned global `uint` words (the
  plane form keeps them 4-aligned), where Strata reads the GGUF blocks as
  stored at 110 / 18 bytes;
- an expert cache's bank mirrors and the bank stream pass through
  `ggml_cl_kq_weight_args` as they did for the f16 tiles; the activation has
  its own preallocated buffers (`prealloc_kq_id8_{q,d}`), so the q8_1 matvec's
  `prealloc_kq_{xq,xd}` and its `kq_q8` dedup key are untouched. They hold
  [column][K] codes and [column][K / 32] scales, the fp16 path's activation
  size (`n_cols = ne11 * ne12`): 21 MB of codes and 2.6 MB of scales at an 8k
  chunk and K 2560, against 210 MB + 26 MB for [pair][K] (`n_used` 10x, real
  VRAM at Flash-Next's margin; `measured-here`, the test's byte arithmetic).
  The quantizer reads columns, not pair indices, so the route buffer needs no
  -1 fill and a recomputed route does not clear the activation (the columns are
  unchanged; the dedup key is the tensor, its buffer, offset and size in this
  graph); the route cache key still carries the path (`rc.i8`) with the tile.

Blocks and scales: the activation block is 32 values, `scale = amax / 127`,
codes round half away from zero (llama.cpp's q8_1, Strata's `quantize_rows`).
The integer sum q8_1 also stores is not used here: each 32-block's int32 DPAS
product is exact and converted to float, then scaled, so no integer `mad` on
a super-block accumulator is needed (the Q4_K 2D kernel's form is not this
one). IQ3_S: `d` per 256-block, `sc` per 32 -> `d * (1 + 2 sc)`; IQ4_NL: `d`
per 32-block. K ascending, one DPAS chain a 32-block; the decode is done once
per block and applied to every 8-pair group of the tile (as Strata applies a
decode to every row tile). The quantizer runs one work-item a (column,
32-block) in 64-wide groups, the column the fp16 kernel's `kq_i8_col`
`(i % ne11) + t * ne11`, and its dedup entry lets a MoE layer's gate and up
share one quantization of src1. The activation is staged a 256-weight
super-block with one barrier pair (not a 32-block), each staged row the pair's
activation column (as the fp16 kernel stages `xh`), the last group taking the
remaining blocks when K is not a multiple of 256 (IQ4_NL down is 640). IQ3_S's
qs and signs planes are read as aligned global `uint` words (the plane form's
4-alignment: nblk is a multiple of 16 at `ne01 % 16 == 0`), as `lb_expand`
reads them; qh and the scale byte stay single-byte fields.

Handled formats (`code`): IQ3_S gate/up (K a multiple of 256, 16-row output
multiples) and IQ4_NL down (K a multiple of 32); any other type, and any
device at sub-group 8, keeps today's fp16 tile path (`GGML_OPENCL_KQ_MM_ID2`
or the tile kernel). The int8 kernels are their own program
(`mul_mm_kq_i8.cl`, built with `build_program_from_source_optional` after the
XMX program) and only when `GGML_OPENCL_KQ_MM_ID_I8` is set non-empty and not
0 and the device is not sub-group 8 (`kq_i8_program_wanted` and `sg8`
directly: the gate runs before `kq_mm_sg` is assigned, so it does not read
that field's default): the default path pays
no second program compile a unit start and does not read the int8 sources; a
build failure there costs the int8 arm only; the dispatch falls back when the
kernels are null. Switches: `GGML_OPENCL_KQ_MM_ID_I8`
unset = `0`, fp16 (this patch keeps int8 opt-in; a later commit flips the
default to auto after recording the B60's KL/task/speed), `0` = fp16,
`1` = int8 wherever the format is handled and the device is sub-group 16
with its kernels present, `auto` = int8 while the group's
mean routed pairs per expert are at or under
`GGML_OPENCL_KQ_MM_ID_I8_CROSS` (default 128, Strata's measured 130-250 rows
per expert on a B70, `paper`; the B60's value is owed by the card window),
and `GGML_OPENCL_KQ_MM_ID_I8_FMT` (a comma or whitespace separated list,
`\n`/`\r` too, of `iq3_s` and `iq4_nl`, case-insensitive; unset, empty,
whitespace-only or a value with no token = both, parsed once at the dispatch;
a format left out takes the fp16 tile; one INFO line at the first int8
decision names the parsed formats). The mean uses `min(n_expert, pairs)` as the experts touched, never
understating it.

Card round 3 (`measured-here`, a B60, one announced window, the pin with the
branch's patches, `GGML_OPENCL_DEVICES=0:0`, `test-backend-ops` from that
tree): Intel's offline compiler (`ocloc 26.27 -device bmg`, the backend's
options, `-DSG=16`) compiles the assembled int8 program with 0 errors (with a
`half` variable restored: 11 errors). Forced int8 on the pre-fix commit
passes 2 of 12 `MUL_MAT_ID` cases (the matvec ones, n=1 and n=5); every GEMM
case fails (IQ3_S NaN, IQ4_NL NMSE 1.90), the fp16 arm passes 12/12. The
defect, named (`code`, then `measured-here`): the kernel's block loop counts
`kb` in elements of K (`k0 += 256`, `kb += 32`) while the IQ3_S and IQ4_NL
kernels read it as a 32-block index (`sb = kb / 8`, `p = kb & 7`, `blk = ...
* n32 + kb`), so the weight reads land beyond the block and, on the last
experts, out of bounds (a read that can fault the card's engines). The fix is
the fp16 tile kernel's own arithmetic (`j32 = kb / 32`, `LB_BI` / `LB_P`):
`sb = kb / 256`, `p = (kb / 32) & 7`, `blk = row * n32 + kb / 32`, in
`kq_i8_ops.inc` helpers the kernels call. After it forced int8 passes 12/12;
the red (both 32-block weight scales `ws = 1.f` on top of the fix) fails
2/12, IQ3_S NMSE 0.86, IQ4_NL NMSE 1.00, no engine reset. Perf (`us/run`,
fp16 arm -> int8 arm): IQ3_S m=640 k=2560 n=16 2,259.39 -> 1,142.74; IQ4_NL
m=2560 k=640 n=16 1,199.10 -> 929.77; IQ3_S n=512 8,482.40 -> 4,154.80;
IQ4_NL n=512 5,035.29 -> 3,825.53. Also fixed here: the program gate now
reads the device's sub-group before `kq_mm_sg` is assigned, and the dispatch
conjunction includes the kernels and the sub-group 16 condition, so a
sub-group 8 device never builds or dispatches the SG16 kernel with an 8x8
local size.

Card round 4 (`measured-here`, a B60, one announced window, the commit
`60cf5a6`): `test-backend-ops` on the committed text, forced int8 12/12 at the
served shapes, the red (both 32-block weight scales `ws = 1.f`) 2/12; int8
perf (`us/run`): IQ3_S m=640 k=2560 n=16 1,125.75, IQ4_NL m=2560 k=640 n=16
912.03, IQ3_S n=512 4,156.28, IQ4_NL n=512 3,826.86 (the fp16 floor is round
3's column). Served Flash-Next IQ3_S (`tools/fn_sweep.py`, the unit's flags,
switch 0 vs 1, two rounds with the second reversed), prompt t/s fp16 -> int8:
with the slot loan 2k 229.6/250.6 -> 252.7/245.4 (inside the spread), 8k
470.4/476.6 -> 474.0/484.5 (+1.2 %), 19k 509.4/514.5 -> 524.1/521.0 (+2.1 %);
without the loan 8k 442.6/442.3 -> 456.7/456.2 (+3.2 %); decode 32.60/34.18
vs 33.35/34.08 (equal); the capital right in every arm; 5 of 6 answer md5s
differ between the arms (a byte tripwire, not the gate). KL (`measured-here`,
one 8,192-token window, `-ub 2048`, `--n-cpu-moe 48` with the expert cache,
the device pinned): the red arm PPL 435,809, mean KL +11.5; int8 twice, mean
KL +0.0059 / +0.0054 nats over the fp16 twins' floor 0.0064 (the 0.03 bar
passes); argmax 0.9585 both against the twins' 0.9692, -1.07 points, the
1-point bar fails by 0.07. Strata's activation quantization is the same form
(`code`, `sycl/src/prefill/moe_fused_iq.dp.cpp`: per 32 values, `amax/127`,
SwiGLU output to int8 for the down); its own top-1 99.7 % is `paper`.
Round 5 adds the per-format switch so the next window prices IQ3_S-only and
IQ4_NL-only against the same reference and fp16 twins, plus a served
IQ3_S-only arm; a layer whose gate/up are int8 and whose down is fp16 reroutes
through the cache key's `rc.i8` (`code`), so the route is per call; the
reroute costs one extra `moe_route_tiles` a layer although both tiles have the
same geometry, not measured.

Round 6 review (`measured-here`, two compilers): the format parser's string
literals sat in the `.inc` prepended to the OpenCL program, where a literal is
`__constant char *`; Intel's offline compiler for the B60 on the exact assembly
(`.inc` + `mul_mm_kq_f16.cl` + `mul_mm_kq_i8.cl`, `-DSG=16 -cl-std=CL3.0`)
reported an address-space change and failed with -11, and clang-18 failed the
same at CL1.2/2.0/3.0, so the test with a clang printed the assembled-program
FAIL at SG16 and SG8 (exit 1); the backend fell back silently and an int8 arm
measured the fp16 tile. Fixed (`code`): the `.inc`'s host-only block (dispatch,
format list, launch sizing) is behind `#ifdef __cplusplus`, which the device
never sees and the test's `opencl_part` strips; the device-free text check
rejects a string literal in the `.inc`'s OpenCL part (red: a literal added
there -> `FAIL opencl: a string literal in the .inc's OpenCL part`). The fix
is `code`; its clang and card compile are `measured-here` in round 8 (0
errors).

Round 7 review (`measured-here`, clang-18 `-target spir64 -x cl -cl-std=CL3.0
-DSG=16` on the backend's assembly): 4 errors "use of undeclared identifier
'kq_i8_col'". `kq_i8_col` is device code (called by `I8_STAGE`) but round 6
left it in the host-only block; `-DSG=8` compiled only because the int8
kernels are under `#if SG == 16`. Fixed (`code`; the compile is
`measured-here` in round 8, clang-18 SG16/SG8 and the B60 ocloc 0 errors): the device
helpers (`kq_i8_col`, decode, quantize, indices) sit above the guard, and the
device-free text check requires every `kq_i8_*` the `.cl` kernels call to be
defined in the `.inc`'s OpenCL part (red: `kq_i8_col` back in the host block
-> `FAIL opencl: the kernel calls kq_i8_col, not defined in the .inc's
OpenCL part`). Also (`code`): `opencl_part` counts nesting; the parser accepts
`\n`/`\r`, compares case-insensitively, and a no-token value is both; the
duplicate `const` on `qq` in the shared `mul_mm_kq_f16.cl` text is dropped.

Card round 8 (`measured-here`, a B60, windows 27 and 28, `cecdea0`'s build; no
kernel change): clang-18 on the assembled program at SG16 and SG8, 0 errors,
the test with `CLANG` all OK; Intel's offline compiler for the B60 (`-device
bmg`, the backend's options, `-DSG=16`) 0 errors. `test-backend-ops`
`MUL_MAT_ID` at the served shapes: the INFO line names the formats (`int8
DPAS formats: iq3_s,iq4_nl` / `iq3_s`); forced int8 12/12, the red (both
32-block weight scales `1.f`) 2/12, int8 IQ3_S only 12/12; the eval list
widened to n = 256, 512, 2048 (a diagnostic edit of the test tree, not in the
patch), fp16 12/12 and int8 12/12, no multi-tile defect inside the NMSE bar
5e-4. KL (`measured-here`, one 8,192-token window, `-ub 2048`, `--n-cpu-moe
48` with the expert cache, the device pinned; the twins were bit-identical to
the reference: KL 0, argmax 1.000): red PPL 429,684, KL +11.49; IQ3_S gate/up
only +0.0059 / +0.0039 nats, argmax -2.95 / -2.47 points; IQ4_NL down only
+0.0117, -4.18 points; both +0.0098, -4.27 points. The KL bar (0.03) passes
every arm, the argmax bar (1 point) fails every arm; window 24's -1.07 points
came from twins at 0.9692, a noisy floor that day. Served Flash-Next
(`measured-here`, fp16 vs int8 IQ3_S only, two rounds reversed): 8k with the
loan 463.8/478.7 vs 475.6/469.7 t/s, 19k 513.0/483.6 vs 511.2/512.1, 8k
without 444.1/444.8 vs 444.1/437.2, decode 34.13/34.43 vs 34.03/34.17 — no
gain beyond the spread, the capital right in every arm. Status (`measured-here`):
opt-in, off by default, not adopted on any unit; owed: the expert GEMM's share
of the prefill (the profiling step, never run). Strata's own figure (`paper`,
`docs/INTEL_INT8_EXPERTS.md:83-88`): KL 0.0002, top-1 99.7 % against its FP16
path on 6,002 teacher-forced positions; ours is one 8K corpus window through
the expert cache, a different method, the same-method comparison owed.

Round 8's test fixes (`code`): `opencl_part`'s inverted `#else` ternary
(`st.empty()` must read `skip`, not an empty stack) and a unit case for its
nesting (red: the ternary inverted -> segfault; restored all OK).

Pricing cases (`code`): the patch adds Flash-Next MUL_MAT_ID cases to
`tests/test-backend-ops.cpp` in both lists. The eval (correctness) list gets
IQ3_S 2560 -> 640 gate/up and IQ4_NL 640 -> 2560 down, 512 experts, 10 used,
16 and 64 tokens, with src1 as a broadcast row (`b=true`, the served gate/up)
and per-slot (`b=false`); the correctness mode runs that list, not the perf
list. The perf list gets the same shapes at batch 16 and 512, so `perf`
prices the fp16 and the int8 arm.

Device-free (`measured-here`): `tests/test-opencl-kq-i8.cpp`, built by the
llama tests, red-first with one-line mutations each seen failing:
- the IQ3_S and IQ4_NL s8 decode against ggml's own `dequantize_row_iq3_s`
  and `dequantize_row_iq4_nl`, value for value, 400 random blocks each
  (red: the qh 9th index bit dropped -> `FAIL IQ3_S k 4 decode -15 want ...
  got ...`; the IQ4_NL low nibbles read as the high ones -> `FAIL IQ4_NL k 0`);
- the 32-value quantizer, its scale and the round-half-away cases at scale 1
  (red: `rint` -> `FAIL quant: .5 away from zero at scale 1`);
- the dispatch rule: env 0/1/auto, unset = off in this patch, the
  sub-group-16 and kernels-present conjunction (`kq_i8_mode` reads both), the
  crossover boundary (red: `<` for `<=` -> `FAIL at the crossover: int8`; the
  sub-group/kernels guard removed -> `FAIL dispatch: switch on, SG 8 ->
  fp16`; auto on SG 8 -> `FAIL dispatch: auto on SG 8 -> fp16`; the pre-patch
  auto default -> `FAIL unset -> off (this patch)`);
- the per-format list `kq_i8_fmt_allowed`: unset, empty, whitespace-only
  (space, tab, `\n`, `\r`) and a no-token value (`,`, ` , , `) are both,
  names are case-insensitive, `iq3_s` only makes IQ4_NL fp16 and `iq4_nl`
  only makes IQ3_S fp16, both named, commas and whitespace as separators, an
  unlisted format and a prefix that is not a token (reds: the final `return
  1` -> `FAIL fmt: iq3_s only -> IQ4_NL fp16`; the name fixed to `iq3_s` ->
  `FAIL fmt: iq4_nl only -> IQ3_S fp16`; the whitespace-only line dropped ->
  `FAIL fmt: whitespace-only -> both`; whitespace dropped from the token
  delimiter -> `FAIL fmt: whitespace is a separator too`; `\n`/`\r` dropped
  -> `FAIL fmt: a newline-only value -> both`; the case fold dropped -> `FAIL
  fmt: names are case-insensitive`; no-token no longer both -> `FAIL fmt: a
  separators-only value -> both`);
- the block indices of an element offset `kb` against the fp16 tile kernel's
  own (`j32 = kb / 32`, `LB_BI` / `LB_P`): IQ3_S's super-block and sub-block
  and IQ4_NL's 32-block for every `kb` of K 640 and 2560 (reds: `kb / 8` ->
  `FAIL index: IQ3_S super-block of kb`; `(kb / 32) & 7` -> `FAIL index:
  IQ3_S sub-block of kb`; `kb` -> `FAIL index: IQ4_NL 32-block of kb`);
- the program gate: unset, empty, `0` and an unparsable word are not built,
  `1` and `auto` are (red: `return 1` -> `FAIL program: unset -> not built`);
- the activation's column: `kq_i8_col`, the fp16 kernel's
  `(i % ne11) + t * ne11`, a broadcast b's `n_used` pairs sharing one column,
  and the served 8k sizes (21 / 210 / 42 MB) (red: return the pair ->
  `FAIL col: slot 9 of token 0 shares the column`);
- the ASSEMBLED int8 program, as `load_cl_kernels` builds it (`kq_i8_ops.inc`,
  then `mul_mm_kq_f16.cl`, then `mul_mm_kq_i8.cl`, at `-DSG=16`): the
  C++-only branch of the `.inc` is behind `#ifdef __cplusplus`, the
  OpenCL-visible text must carry no C++-only token, the `.inc`'s OpenCL part
  no string literal (it is `__constant char *`; red: a literal added there ->
  `FAIL opencl: a string literal in the .inc's OpenCL part`) and every
  `kq_i8_*` the `.cl` kernels call defined there (red: `kq_i8_col` back in the
  host block -> `FAIL opencl: the kernel calls kq_i8_col, not defined`), no reserved
  OpenCL type name (half, float, int, ...) may be used as an identifier, and, when
  `clang`, `clang-18`, `clang-19` or `$CLANG` is present, the assembled text
  with a prelude for the Intel subgroup builtins the clang header lacks
  compiles as OpenCL C at CL3.0, `-target spir64`, `-DSG=16` and `-DSG=8`;
  each overloaded prelude declaration carries
  `__attribute__((overloadable))` (without it clang-18 rejects the two
  same-name f16 DPAS declarations as "conflicting types" and the test exits 1;
  red: the attribute dropped from one overload -> `FAIL opencl: overloaded
  prelude declaration ...`). When no clang is found the compile half is skipped
  and the text checks are the pre-run; the card compile gate runs Intel ocloc
  on the same text. (reds: `half` as a variable in `I8_STAGE` -> `FAIL opencl: reserved type
  name half used as an identifier` (the round-3 blocker that failed the
  program build); `nullptr` -> `FAIL opencl: C++-only token nullptr in the
  OpenCL part`, the blocker that failed every device's XMX program when the
  `.inc` was prepended there).
The kernel helpers and the dispatch rule are one text,
`ggml/src/ggml-opencl/kernels/kq_i8_ops.inc`; the int8 kernels
(`kernels/mul_mm_kq_i8.cl`) get it prepended in their own program by a CMake
embed step (so the embedded build has no runtime include), and a
non-embedded build copies the `.inc` beside the kernel sources, so the
runtime never reads it empty.

Owed (no class until run): the expert GEMM's share of the prefill wall and
the B60 crossover; the same-method quality comparison against Strata's
teacher-forced form. Commands in
`docs/campaigns/int8-dpas-prompt-experts.md` §"Card window".
