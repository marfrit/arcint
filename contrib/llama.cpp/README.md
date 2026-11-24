# Patches carried against the pinned llama.cpp

The `--engine llama` executor (`src/exec/backend_llama.cpp`,
`docs/llama-engine.md`) builds llama.cpp as a subproject from
`ARCINT_LLAMA_DIR`. The pin is ggml-org/llama.cpp commit `3e4dfa8`
(2026-10-15; until then `bed0a85` of 2026-10-02, see "Moving the pin"); the
tree is that commit with the patches below applied one by one in byte order (the
order `LC_ALL=C` sorts the names: `0129-` before `0129a-`, `0130b-` before
`0130b1-`). `apply-series.sh` does that whatever the shell's locale; a plain
`patches/*.patch` glob under en_US.UTF-8 puts `0129-` after `0129k-` and stops
at `0129b` (GitHub issue #6):

    git -C <llama.cpp> checkout 3e4dfa8
    <arcint>/contrib/llama.cpp/apply-series.sh <llama.cpp>

A patch that does not apply cleanly to the pin is a bug in this directory, not
a reason to move the pin. Since the fold of 0001-0096 (2026-11) the directory
holds twelve topic patches, `0000-01-*` to `0000-12-*`, followed by 0097 and
the later patches one by one. The patches are no longer PR-shaped: the
repository is where the concepts are read (the sections below, the code), and
a topic patch is the diff between two intermediate trees of the fold, which
need not build. The sum is byte-identical to the unfolded series;
`patches/FOLDED.md` maps every old number to its folded patch and names the
commit that holds the unfolded files. To offer something upstream, extract
it from the code and its section below.

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
   patch against its predecessor, so each applies with `git apply` on the new
   pin (the folded topic patches are ported as the unfolded ones were, and
   re-folded).
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

The numbers below 0097 are history labels: the files they name were folded into
the `0000-NN-*` patches, and `patches/FOLDED.md` resolves each number to its
folded patch. 0097 and later are files of their own.

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
`GGML_OPENCL_FA_INTEL_KV_PER_SPLIT` sets the keys per split (128, by depth since 0097);
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
- beyond this patch's kernels: 2026-11-12, the B60 (`measured-here`): the 2D,
  tile and ID2 kernels at their shipped tiles rebuilt at 128 GRF: faster in
  none of the 19 timed op-suite cases at 512 tokens; the 128 arms' outputs
  checked only in part (`docs/campaigns/research-inferzero.md`, on the `qfndev` branch).

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
(`gdn_keep_gating_inputs`, now `src/models/gdn-gate-inputs.h`) attaches them as
extra sources of the GDN node (`src[6]` alpha, `src[7]` b **as originally
written; patch 0081 moved them to `src[8]`/`src[9]`** — the family's slots, see
0081 below), which keeps them alive; the op's meaning does
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

## 0072-opencl-uniform-work-groups.patch

The backend's launches without non-uniform work groups (`code`: the patch's
own text, `ggml-opencl.cpp` and the kernel sources). OpenCL 2.0+ accepts a
global size that is not a multiple of the local size as a non-uniform work
group; the individual driver does, LEO (OpenCL over Level Zero,
`EnableLEO=1`, `docs/campaigns/research-vram-arena.md` §8) does not and returns
`CL_INVALID_WORK_GROUP_SIZE` (-54) at the first ragged launch, which failed
the stage 2a gate at the set_tensor converts (`measured-here`, the gate's
first window, `docs/campaigns/vram-user-space.md`).

One rule, one place: `ggml-opencl-uniform.h` rounds every dimension of the
global size up to a whole local size, and `enqueue_ndrange_kernel` applies it
to every launch through the wrapper; a direct `clEnqueueNDRangeKernel` call in
the backend rounds before enqueueing (67 launch sites with a local size,
`code`: the device-free test counts them from the source). Rounding is
idempotent, so a grid that is already a whole multiple is unchanged. The
work-items of the rounded tail exit in the kernel. This replaces the pin's
two-way fallback (`local_work_size_ptr = NULL` when `n % 64 != 0` and the
device has no non-uniform support, at 19 sites): that path changed the local
size, this one keeps it and rounds the count. The pin's
`CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT` query and its field are gone with
that fallback (0072 reads neither: the individual driver's answer does not
match what LEO accepts, so every launch is made uniform instead of trusting
the capability).

Two classes, told apart by the build, not by a runtime test (`code`): 170 of
the 368 launch sites live only in `GGML_OPENCL_USE_ADRENO_KERNELS` regions, and
in that build both helpers return their input unchanged — the Adreno launches
keep the pin's ragged sizes and the Adreno kernels need no bound of their own
(several decompose their axes by modulo, `kernel_transpose_32` on
`get_global_id(2)`, `mul_mm_f16_f32_kq_kqv.cl:204`). LEO is not an Adreno
driver: `GGML_OPENCL_USE_ADRENO_KERNELS` builds do not define the Intel
launches and an Adreno device never runs through Level Zero. The Intel class
(198 sites) rounds; the test's Adreno build asserts the no-op.

The inventory of the Intel class (`code`: every launch read in the patch's
tree), by class:

- **Whole multiple by construction, unchanged by the rule.** Tile and
  block-derived grids: `ne01*nth` / `ne10*nth` / `ne11*nth` with local `nth`,
  every `CEIL_DIV(..., L)*L` (the matmul, im2col, conv, flash-attention,
  MoE-sort and -reorder, GEMM, GEMV-split, repack and set/get-rows grids), the
  one-group reductions (softmax, argsort, sum_rows, norm, cpy, diag) whose
  local size is the tile, and every local size 1 (the converts' restore legs,
  the dequant copies, the reconstruct kernels).
- **Ragged, the kernel already bounds the work-item.** The elementwise grids
  over `ggml_nelements` and the convert grids over n_blocks/ne01. The bound
  was already there (`neg`, `pad`, `upscale`, `tri`, `fill`, `q5_0`, `q5_1`,
  `q6_K`, `iq4_nl`, the q4_K/q5_K/q6_K trans4 converts, the add_id/glu-style
  n-argument kernels); the test below keeps them.
- **Ragged, the kernel got its bound in this patch.** Read each to find the
  index it must bound and what it did not have (`code`):
  - the convert kernels a separate arrays form: `kernel_convert_block_q1_0`,
    `_q4_0`, `_q4_0_noshuffle`, `_q4_1`, `_q4_1_noshuffle`,
    `_q5_0_noshuffle`, `_q5_1_noshuffle`, `_mxfp4`, `_q8_0`, `_q4_K`,
    `_q4_K_noshuffle`, `_q5_K`, `_q5_K_noshuffle`, `_q6_K`,
    `_q6_K_noshuffle`, `_iq4_nl`, `_iq4_nl_noshuffle`, the q6_K and IQ4_NL
    restore legs — an appended `ulong n_blk` and
    `if (get_global_id(0) >= n_blk) return;` (the q8_0 convert at `n_blocks`
    against local 64 is the gate's abort site);
  - the elementwise kernels over the flat count: `gelu`, `gelu_erf`,
    `gelu_quick`, `silu`, `relu`, `sigmoid_f32`/`_f16`, `clamp`,
    `tanh_f32`/`_f32_4`/`_f16`/`_f16_4`, `expm1_*`, `abs_*`, the
    `unary_ext` and `leaky_relu` macros, `softplus_*`, `scale_f32`,
    `sqr_cont_*`, `sqrt_cont_*` — an appended `int n` and the same exit (the
    `_4` variants' n is the element count after the /4 the host already does);
  - the row broadcasts `add_row`, `add_row_f16`, `mul_row`, `mul_row_f16`,
    `div_row`/`_f16`, `sub_row`/`_f16` — the row length `ne` is the modulo,
    not the count, so an appended `int n_total` carries the launch's true
    count and the exit is on it (red: a guard on `ne` drops a whole broadcast);
  - `ssm_conv` (an appended `ne01`, the dim-0 count), `solve_tri` (the
    existing `k` bounds dim 0; appended `ne2`/`ne3` bound dims 1/2),
    `diag_mask_inf` (the existing `ne00` bounds dim 0) and `_8` (an appended
    `int n_f4`, float4 units — the item writes two).
- **Sub-group or barrier collectives** (`code`): the gemv 32b-trans kernels
  (`gemv_noshuffle_q4_k/q5_k/q6_k_f32_32b_trans`, launched `ne01 x 8` against
  local `64 x 8`) must not lose a lane before their `sub_group_broadcast` and
  `barrier`. They clamp the row index for the loads and gate only the final
  `dst[i01] = sum` on `i01 < ne01`, after the collectives. The conv, flash
  attention, argsort and MoE-sort kernels are whole-multiple grids (class 1),
  so their groups stay full.

Round 2 (`code`, the review of the first commit; the round's own measurements
are `measured-here` where the reviewer ran them):
- `ssm_conv.cl` did not compile: the appended `int ne01` followed a comma that
  sat inside the preceding parameter comment, so both kernels read
  `int act /* ... */ int ne01` (`expected ')'`, then an undeclared `ne01` at
  the bound; 4 errors with `clang-18 -target spir64 -x cl -cl-std=CL3.0
  -Xclang -finclude-default-header -fsyntax-only`). ssm_conv is built at init
  through the fatal `build_program_from_source`, so every start died before
  the first op. The separator is now outside the comment (`int act, // ...`,
  then `int ne01`), and the test carries a structural check for the class: a
  parameter list whose comment line ends with a comma (red: the bug restored ->
  `FAIL: ssm_conv.cl:16 a comma inside a parameter comment ends the line`).
- `transpose.cl`: the `padded_rows` exit had gone into `kernel_transpose_16`,
  which has no such parameter; the kernel that has it is
  `kernel_transpose_32_16` (Adreno-only, image buffers). With the Adreno class
  unrounded, `transpose.cl` is back at the pin unchanged: the first commit's
  edits there are reverted, and the README's claim that the transpose kernels
  bound their launch was wrong.
- the test's compile half (round 2's form, superseded in round 4 by the explicit
  list below): when a clang exists (`$CLANG`, `clang`, `clang-18`,
  `clang-19`), every `kernels/*.cl` is syntax-checked as OpenCL C at
  `-target spir64 -x cl -cl-std=CL3.0 -Xclang -finclude-default-header`. A
  file the backend assembles with an `#include`, or builds with its own `-D`
  defines (the names read from `ggml-opencl.cpp`; e.g. `-DSG`, `-DWG`,
  `-DBLOCK_STRIDE_A`), is listed and skipped with that reason. Round 3: the
  full sweep also hit `kernels/sdpa_xmem_f32_f16_os8.cl`, which calls Qualcomm
  builtins (`qcom_get_physical_sub_group_id`,
  `qcom_sub_group_constant_load8`, `qcom_sub_group_sync`) that the generic
  front end cannot know, and which fails identically on the pre-0072 text; any
  source whose code calls a `qcom_*` builtin (or uses such an attribute) is
  now listed and skipped with that reason (`qcom_reqd_sub_group_size`, an
  attribute behind the driver's own pragma guard, still compiles and is not
  skipped). 153 compiled, 43 skipped (16 with a `-D`, 27 vendor). No clang is
  installed on this host, so the half prints its note and Claude runs it on a
  host that has one (red: a stub clang returning 1 for `ssm_conv.cl` ->
  `FAIL: ssm_conv.cl does not compile as OpenCL C`, exit 1; the ssm_conv comma
  restored still fails that half as well as the comment check).
- hygiene: the dead capability field and its query are gone (above); the two
  lines the review flagged as mis-indented are corrected.

Device-free tests (`tests/test-opencl-workgroups.cpp`, the CTest
`test-opencl-workgroups` and, for the Adreno build, the same source built with
`-DGGML_OPENCL_USE_ADRENO_KERNELS` as `test-opencl-workgroups-adreno`; no GPU):
the rounding helper (per dimension, idempotence, local 0/1/NULL, a tail; in
the Adreno build that it returns its input), a text check over
`ggml-opencl.cpp` that every `clEnqueueNDRangeKernel` call with a local size
has the rounding call between it and the previous launch and that no
`local_work_size_ptr` fallback remains, the parameter-comment check, and the
compile half above. Reds, each seen failing: one helper line removed ->
`FAIL: launch at byte ... passes a local size without rounding`, exit 1; the
helper returning its input -> `FAIL: round 10/64`, `FAIL: round 65/64`,
`FAIL: round three dimensions`; the Adreno no-op removed from the
one-dimension helper -> `FAIL: adreno: round leaves the count` in the Adreno
build; the ssm_conv comma restored -> the parameter-comment FAIL above; a stub
clang failing `ssm_conv.cl` -> the compile FAIL above.

Round 4 test fix (`code`, from the review of `00b7532`, `measured-here` for the
measurement that motivated it): round 3's auto-discovery still failed on any
host that has a clang - with `clang-18` it reported errors on 39 sources
(`flash_attn_*.cl`, `gated_delta_net_intel.cl`, `mul_mm_kq_{2d,f16,few,i8}.cl`,
`mul_mv_*.cl`, ...), every one of them failing identically on the pre-0072
text (Intel sub-group builtins and `-D` macros a generic front end does not
know), so the half caught nothing and the suite could not pass. It is now an
explicit list: `tests/opencl_clang_clean.txt` names the 135 kernel sources
that compile with 0 errors under a generic clang, measured with
`clang-18 -target spir64 -x cl -cl-std=CL3.0 -Xclang
-finclude-default-header -fsyntax-only`, both before the series' 0072 change
and after. The half compiles exactly those and fails on any error; a source
not in the list is counted and reported but not compiled (61 of the 196
sources, the vendor-builtin and `-D` ones); a listed name that is missing
fails. The structural no-clang checks stay (the parameter-comment comma). Red
(the one-line mutation): the comma after `int act` in `ssm_conv.cl` restored ->
`FAIL: ssm_conv.cl does not compile as OpenCL C` plus the comment check, 2
failures, exit 1 (a stub clang standing in for that compiler failure here;
Claude runs the half with clang-18). Refresh: compile `kernels/*.cl` with the
command above and keep the names that report no error in both texts.

Card window 33 (`measured-here`: Claude, 2026-10-31, `2ba3656`'s build (that round's backend text is byte-identical to this commit's; only the test half changed since),
compute-runtime 26.27.39122.11, the same binaries under both drivers):
- `clang-18 -target spir64 -x cl -cl-std=CL3.0 -Xclang -finclude-default-header
  -fsyntax-only` on every kernel file the patch touches, the pre-0072 text
  against the patched: no file gains an error (round 2's ssm_conv and transpose
  defects are closed).
- `test-backend-ops`, the full eval (12,881 cases): **B60** default 12,724
  passed, LEO 12,724 passed, the failure sets identical by name (76 MXFP4
  cases - 74 `MUL_MAT_ID`, the pin's (README 0012), and 2 `MUL_MAT_ID_W4A8`/`_W4A4`; 70 `MUL_MAT_VEC_FUSION`, 8
  `SWIGLU_OAI`, 3 `ADD_{SILU,SIGMOID,SOFTPLUS}_MUL`, all failing under the
  default driver too), so on the B60 LEO passes exactly what the default driver
  passes. **A770** default 12,867 passed (3 `ADD_*_MUL`, 3 `FLASH_ATTN_EXT`
  hsk=64 f16, 8 `SWIGLU_OAI`); LEO meets the same failures, then the runtime
  aborts after 4,615 passing cases in the next test (`MUL_MAT` type_a=mxfp4
  m=2880 n=32 k=2880, which the default driver passes): `Abort was called at
  1898 line in file: .../xe_hpg_core/hw_cmds_generated_xe_hpg_core.inl` - an
  assertion in the runtime's DG2 command generation, not a launch error (in
  this release LEO is supported only on the later products, `code`:
  `isLEOSupported()`, `docs/campaigns/research-vram-arena.md` §8).
- release A/B (`tests/acceptance/ab.py`, two rounds, default against LEO
  through a wrapper exporting `EnableLEO=1 NEOReadDebugKeys=1`): **agent**
  (B60) PASS, prefill 55k 716.45 -> 714.45 (-0.28 %), 138k 503.65 -> 503.25
  (-0.08 %); **coder** (A770) decode +5.9 % (51.55 -> 54.60, 52.80 -> 55.85,
  45.30 -> 48.00 t/s), prefill 41k -5.3 % (730.15 -> 691.15), FAIL on prefill;
  texts byte-identical in every run of both arms on both units.

Disposition (`measured-here`): 0072 closes LEO's -54 on both cards - it is no
longer a launch defect but a runtime limit. Route A is usable on the B60 at the
default driver's numbers (same pass set, same answers, prefill inside the
floor). On the A770 route A is blocked by the runtime's abort in its DG2
command generation, owed the newest compute-runtime in the fresh-host
container, and its LEO prefill loss (-5.3 % at 41k) is unexplained, owed a
profile under both drivers.

Trajectory: at stage 2a's first window the eval stopped at the first ragged
launch under LEO (the set_tensor converts, -54) before any op; window 33 above
is the full eval, and the earlier 75xx-case run that produced
A770 7522/7525 and B60 7451/7525 is superseded by the 12,881-case suite.

To re-run, build the series from the pin, then

    GGML_OPENCL_DEVICES=0:0 ./bin/test-backend-ops -b GPUOpenCL          # B60, default driver
    GGML_OPENCL_DEVICES=0:0 EnableLEO=1 NEOReadDebugKeys=1 ./bin/test-backend-ops -b GPUOpenCL
    GGML_OPENCL_DEVICES=1:0 ./bin/test-backend-ops -b GPUOpenCL          # A770, default driver
    GGML_OPENCL_DEVICES=1:0 EnableLEO=1 NEOReadDebugKeys=1 ./bin/test-backend-ops -b GPUOpenCL

and `tests/acceptance/ab.py` for the A/B, the LEO arm's binary through the
wrapper. The coder's prefill loss is owed a cause: rebuild with
`-DGGML_OPENCL_PROFILING=ON` and compare `cl_trace.json` under both drivers for
the same prefill chunk (kernel names, device durations, copies).

## 0076 — retired, not carried

The Intel >4 GiB JIT option (`-cl-intel-greater-than-4GB-buffer-required` /
`-ze-opt-greater-than-4GB-buffer-required`) is not a patch. **Evidence class:
`measured-here` (the card host, offline; compute-runtime/ocloc 26.27.39122.11).**
On dg2 the default program is stateful (`send.ugm` with ExBSO and surface state
`0x220E8504`); either option makes it stateless (`0x020E8584`, no surface state).
On bmg all three builds are identical and already stateless. `clinfo`: the
A770's `CL_DEVICE_MAX_MEM_ALLOC_SIZE` is 4,294,959,104 B and the B60's is
24,385,683,456 B. So on the A770 no `cl_mem` exceeds 4 GiB and the option moves
every kernel to stateless addressing for nothing; on the B60 it changes nothing.
The old "the slots are a >4 GiB buffer" premise is contradicted by the OpenCL
type's split at `CL_DEVICE_MAX_MEM_ALLOC_SIZE` (`code`,
`docs/campaigns/research-vram-arena.md` §2.4). Two programs build their own
option strings (argsort, div) and are not on the JIT builder's path (`code`).

## 0077-opencl-alias-pages.patch

A **sampled tripwire** for VRAM page-table aliasing at the expert cache's init
and each learned bank-stream order's ring (re)allocation (a relearn after a
reset scans again), a named deviation from Strata's full scan. **Evidence class: `code` (Strata's
`sycl_queue.hpp:168-206, 259-295`, `research-reference-audit.md`) and
`measured-here` for the device-free cells; the card arm is owed.**

The round-1 build intersected host virtual ranges of distinct USM host
allocations. Two distinct allocations cannot overlap, so it could not fire on
any real class; it is removed (`code`: no `ggml_cl_ranges_overlap` remains).
The reference's class lives in VRAM page tables: two virtual 64 KiB blocks
inside one allocation sharing one physical page, where the later write is
visible when the earlier block reads back.

**The deviation, said.** Strata tags every 64 KiB block of an allocation of
32 MiB or more with GPU kernels, frees and re-allocates behind a growing spacer
up to 8 times when a tag is lost, then warns and proceeds (`code`, the lines
above). This patch samples at most 16 blocks an allocation and one region a
slot tensor, through blocking host<->device copies; for Strata's
one-aliased-page class that is a sample, not a full scan. The review put the
one-64 KiB-block sample's hit chance at order 2 % (estimate, not measured
here); one whole slice is ~0.6 MiB on this artifact, a larger sample, so its
hit chance is higher and is not quantified. The full-scan
alternative is a write kernel plus a read-back kernel over every block: one
64 KiB write and one read a block, i.e. **2 × allocation bytes of traffic** in
one scan pass at a learned order; at the unit's 12,000 MiB slot budget that is ~24 GiB of
traffic (formula, `code`; the seconds are the card arm's), with the same
warn-and-proceed verdict. It is not built here; the operator can choose it
against this sampled cost.

The patch is that sampled write/read-back, at the cache's init and each learned
bank-stream order's ring allocation (not model load; a relearn after a reset
scans again):

- `ggml/src/ggml-opencl/ggml-opencl-alias.h` (new): the bounded plan (at most
  16 blocks an allocation, evenly spread, 64 KiB aligned) and the pattern, so
  the device-free cell tests exactly what the backend runs;
- `ggml-opencl.cpp`: when a learned order completes and the ring is (re)allocated
  (K=4), the blocks are written, then all read back; `ggml_cl_bank_stream_reset`
  clears the learned state, so the next learned order scans again. The USM **host** staging ring is not
  scanned (host RAM, not a GPU page table, `code`). A finding names the block
  and a scan ends with one summary line:
  `alias check, bank stream ring: d of N sampled blocks differ`. A finding
  **warns and proceeds**, the ring included; Strata's spacer retry is not built.
  Cost: ~4 MiB written plus ~4 MiB read for the K=4 ring (`code`; the seconds
  owed);
- `src/llama-expert-cache-alias.h` (new) and `llama-expert-cache.cpp`: the same
  tripwire over every slot tensor in its slot buffer (`bslots`, the ask's
  `b_lend`, the candidates' per-layer spares) at the init, after the usage marks
  and before the fill. The region is **one whole slice**: offset a multiple of
  the tensor's `nb[2]`, size exactly `nb[2]`. The OpenCL set path's planar
  conversion asserts `offset % nb[2] == 0 && size % nb[2] == 0`, so the old
  arbitrary 64 KiB offset died at load on IQ3_S (the use configuration) and the
  SOA path read past a short caller buffer (`code`). The sampled switch covers
  the planar types (IQ2_XXS, IQ2_XS, IQ2_S, Q2_0, IQ3_S) and IQ3_XXS, whose
  OpenCL set path is a plain write and accepts a whole slice; the whole-tensor
  readers (Q4_0, Q1_0, Q8_0, IQ4_NL, Q4_K) are not sampled because their set
  path reads the whole tensor. On the IQ3_S Flash-Next artifact the measured
  count is 94 sampled expert tensors, ~53.8 MiB of whole slices (`code`, the
  GGUF dims and the cache's slot tensors). The per-type slice sizes are in
  `docs/campaigns/prefill-slot-loan-design.md:498-503` (IQ3_S 704,000 B,
  IQ3_XXS 422,400 B, IQ4_XS 870,400 B, IQ4_NL 921,600 B, Q8_0 1,740,800 B);
  the upper bound if all 144 slot tensors (48 layers x 3) were IQ3_S is
  144 x 704,000 B = 101,376,000 B, ~97 MiB. The artifact's measured ~54 MiB
  written plus ~54 MiB read (~108 MiB traffic), one-time (`code`; the seconds
  are the card arm's); the summary line reports the sampled bytes. The plan
  holds spans and ids, not expected bytes: the pattern is regenerated at each
  read-back, so the peak host transient is one region (~0.6 MiB here), not the
  sampled total, and the load's RAM forecast has room for it. The sampled total
  is not held on the host. `LLAMA_EXPERT_CACHE_ALIAS=0` turns it off for the
  load-cost arm. A finding **warns and the load proceeds**, not a refused cache
  (`code`). The patterns are overwritten by the expert load; nothing the model
  computes changes.

Cells (`code`, device-free): `tests/test-opencl-alias.cpp` (plan, pattern, a
two-page model that aliases on purpose reports the earlier block, the bank-ring
call, no host-ring scan, findings and the summary); `tests/test-expert-cache-alias.cpp`
(the region arithmetic over every sampled type's `nb[2]`, pattern, one differing
region is one finding, writes before reads, warn-and-proceed, no held expected
bytes and the summary's sampled bytes). Reds (`measured-here`, each seen
failing): the arbitrary 64 KiB offset restored (`FAIL: the region starts on a
whole slice` and `FAIL: the region lies inside the tensor`); the doubled
region restored (`FAIL: the region is one whole slice`); IQ3_XXS removed from
the sampled switch (`FAIL: the planar types are the ones sampled`); a
whole-tensor reader added to the sampled switch (`FAIL: the whole-tensor SOA
readers are not sampled`); a held expected region restored (`FAIL: the plan
holds no expected bytes and the summary reports them`); the summary's byte
count removed (`FAIL: the finding is logged with its count and the sampled
bytes`); the host staging-ring scan restored (`FAIL: the USM host staging ring
is not scanned`); the slot call refusing the load (`FAIL: a finding warns and
the load proceeds`).

Card arms owed: a fresh Flash-Next load at the unit's slot budget, the
`alias check` summary lines, the load seconds with `LLAMA_EXPERT_CACHE_ALIAS=0`
and `=1`, and a served check. A device-side alias cannot be constructed on
purpose (the runtime chooses the pages), so the red is the check's own mutation
and a healthy load's zero findings.

Implemented by meseeks (a private DeepSeek-V4.1-Flash quant, run locally) over 1
review round; briefed and reviewed by Claude.

The three cells are labelled `unit` for a llama tree configured with
`LLAMA_BUILD_TESTS=ON` (`ctest -L unit`); the package recipe builds the engine
with tests off, so they do not run there.

## 0078-opencl-staging-bound (dropped in 1.4.1)

**Dropped from the series in 1.4.1** (operator, 2026-11-01): it cost
Flash-Next 2.2 % of its decode. The 1.4.0 release gate measured 32.97 against
1.3.0's 33.70 t/s, pooled over nine runs of each build in three passes on the
B60 (window 48). The bisect in window 50 (`measured-here`, ab.py flashnext, four
rounds each against the 1.4.0 package) named this patch:

| 1.4.0 minus | decode, variant against 1.4.0 (t/s) | |
|---|---|---|
| 0078 | 33.75 against 32.85 (+2.7 %) | back at 1.3.0's 33.70 |
| 0077 and 0078 | 33.42 against 32.95 (+1.4 %) | nothing beyond 0078 alone |
| 0075 | 33.40 against 33.85 | no gain |

The mechanism is not measured. The likely path is the CPU-island copies, which
0078 sends through the same scheduler function as the graph inputs: Flash-Next
is the only unit with CPU islands (`--llama-cpu-moe 48`), and the coder's and
the agent's numbers did not move. The coder's checkpoint-restore gain (time to
first token 2.07 / 2.07 s against 2.17 / 2.37 s, window 47) goes with the patch.
A narrowed 0078 (graph inputs only, CPU-island copies on the old path) is owed
and must clear Flash-Next's decode before it returns. The record below is the
1.4.0 patch as it was.

The graph-input staging bound. **Evidence class: `code` and `measured-here` for
the device-free cells; the card arms are owed.**

0047 staged a graph input into a USM ring and removed the blocking write for
inputs at or below 4 MiB; above it the write blocked. The round-1 0078 removed
that cap by splitting a large input into 4 MiB chunks, which the review found
wrong twice (`measured-here`): an input larger than the ring wraps inside its
own loop (the f16 mask at 200K depth, ubatch 512 is ~200 MiB: six `clFinish`es
and a 200 MiB host memcpy where one blocking DMA was), and removing the refusal
rerouted every non-weight `set_tensor`, including the checkpoint restore and
`llama_state_seq_set_data` (DESIGN §3.4's 38.75 → 0.61 s resume).

The bound, both terms:

- `ggml/src/ggml-opencl/ggml-opencl-staging.h` (new): `ggml_cl_stage_fits` (an
  exact fit at the head is allowed; a zero-size ring fits nothing);
- `ggml-backend.h` / `ggml-backend.cpp`: a thread-local marker,
  `ggml_backend_sched_input_copy_begin/end/active`, held by a guard at
  `ggml_backend_sched_copy_input`; another thread's restore is never staged
  (`code`). It **gets the old blocking write**: 1.3.0's 0047 staged every
  non-WEIGHTS `set_tensor` at or below 4 MiB, including
  `llama_state_seq_set_data`'s KV and recurrent chunks, so the restore timing
  before/after is a card arm owed before 0078 is called equivalent (DESIGN
  §3.4's 38.75 → 0.61 s resume). A CPU island's host->device copy goes through
  the same scheduler copy function, so a non-INPUT copy up to the ring's free
  space is staged too (`code`).
- `ggml_cl_stage_write`: stages only when the marker is active and the whole
  copy fits the ring's free space. A copy larger than the ring is tested before
  the wrap and returns false without a `clFinish`, taking the blocking write; a
  wrap before a smaller write is the 0047 reuse (a `clFinish`, then the head
  resets). One `clEnqueueWriteBuffer` of the whole copy; no chunk loop. `GGML_OPENCL_STAGE_MB` sets the ring (8-1024 MiB, default 32);
  `GGML_OPENCL_STAGE_INPUTS=0` turns staging off; `STAGE_RED=1` is the
  stale-ring red.

Cells (`code`, device-free): `tests/test-opencl-staging.cpp` (the fit table, one
ring-bound ask after the explicit larger-than-the-ring return, the marker's
thread-local guard, no `size > (4u << 20)`, no chunk loop). Reds
(`measured-here`, each seen failing): the marker check removed
(`FAIL: staging asks the scheduler's graph-input marker`); the larger-than-ring
return moved after the wrap (`FAIL: a copy larger than the ring does not finish
the ring`).


**Window 47 (`measured-here`, Claude, 2026-11-01, the A770, the coder with its unit's flags, the 1.3.0 package
against this branch's binary f3ec32b, two rounds alternated):** a 40,365-token prompt sent twice with different
final questions; the second request hits 97.5 % of the prefix (39,337 tokens) and so takes the checkpoint
restore. Time to the first streamed byte: request 1 (cold) package 48.04 / 48.22 s, branch 47.92 / 47.96 s;
request 2 (restore) package 2.17 / 2.37 s, branch 2.07 / 2.07 s. The restore path does not regress under the
bound. The coder's 0077 check is a no-op (Q4_K slots are not sampled); 0077's load cost is measured on
Flash-Next in the release window.

Card arms owed (each on the unit's own `--model`, `--mmproj`, `--device` and
`--port` flags, collected as `UNIT_FLAGS`; the baseline is the `9f1ea8e` deb,
the branch is this patch series' deb):

```
GGML_SCHED_TIMING=1 GGML_OPENCL_STAGE_INPUTS=1 arcint $UNIT_FLAGS --port 8090
GGML_SCHED_TIMING=1 GGML_OPENCL_STAGE_INPUTS=0 arcint $UNIT_FLAGS --port 8090
for MB in 32 64 128 256; do GGML_SCHED_TIMING=1 GGML_OPENCL_STAGE_MB=$MB arcint $UNIT_FLAGS --port 8090; done
GGML_SCHED_TIMING=1 GGML_OPENCL_STAGE_RED=1 arcint $UNIT_FLAGS --port 8090
```

- a 100K agent prefill and decode, `STAGE_INPUTS=1` against `=0`, alternated
  twice: `GGML_SCHED_TIMING=1` inputs us/graph and decode;
- the `GGML_OPENCL_STAGE_MB` sweep above, at least one value covering the depth's
  f16 mask (Flash-Next's 200K mask is ~200 MiB; the unit's slot budget decides
  the ring);
- checkpoint restore: 24.3k-token follow-up TTFT on the branch and the baseline
  deb, same flags, before/after 0078 is called equivalent;
- `GGML_OPENCL_STAGE_RED=1`: stale ring bytes, the path's own red.

Adoption follows the repository's rule: any measured gain with no other target
degrading is adoptable. The comparison is the per-graph input time with
`STAGE_INPUTS=1` against `=0`, and the restore's TTFT on the branch against the
baseline deb.

Implemented by meseeks (a private DeepSeek-V4.1-Flash quant, run locally) over 1
review round; briefed and reviewed by Claude.

## 0079-opencl-dg2-leo-image-limit.patch

The A770's LEO abort, worked around at the creation site. **Evidence class of
the defect: `measured-here` + `code`.** On a DG2 device under the runtime's LEO
mode (`OpenCL on Level Zero`, `EnableLEO=1 NEOReadDebugKeys=1`), an
`IMAGE1D_BUFFER` (`CL_R`, `UINT32`) above **16,384 texels** aborts inside
`RENDER_SURFACE_STATE::setWidth` at the 14-bit field check
(`UNRECOVERABLE_IF((value - 1) > 0x3fff)`, compute-runtime 26.35 source); 16,384
fits, 16,385 does not. The default driver passes every size; the B60 under LEO
passes every size. The abort is in the runtime's DG2 command generation, so the
allocation succeeds and there is no `clCreateImage` error to catch. The
reproducer is `tools/leo_image1d_buffer_repro.c` (`measured-here`, runtime 26.27
and 26.35).

The patch is narrow by design: on **DG2 under LEO only** an image above 16,384
texels is refused; every image that fits is still created, and the driver's own
`CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` is still honoured. On the default driver and
on the B60 the rule's DG2 or LEO term is false, so the images, kernels and
outputs are the same as before (the default-driver arm's pass set and texts are
the gate). The rule is one device-free header function and one predicate pair:

- `ggml/src/ggml-opencl/ggml-opencl-image-limit.h` (new): the rule, the
  `16,384` bound, the LEO version predicate (`CL_DEVICE_VERSION` contains
  `LEO`, the device string reads `OpenCL 3.0 LEO`; `measured-here`) and the DG2
  name predicate (Intel Arc A-series / Pro A-series; `code`);
- `ggml-opencl.cpp`: at init, `leo_mode` from the device version and `dg2` from
  `CL_DEVICE_NAME` / `CL_DEVICE_ID_INTEL` when
  `cl_intel_device_attribute_query` is exposed (the A770 is `0x56a0`,
  `measured-here`; the id window covers the desktop A-series ids only, the name
  predicate runs first and covers the Arc A and Pro A names; a Flex part would be
  missed, `code`). One call site
  `ggml_cl_image_texels_ok()` builds the mode from the context.

Every `IMAGE1D_BUFFER` creation in the patched `ggml-opencl.cpp` (`code`,
inventory over the 0001-0072 text; counts before this patch):

- **one** is reachable in the Intel build outside an Adreno guard: the MXFP4
  weight image in `ggml_backend_opencl_buffer_set_tensor`
  (`CL_MEM_OBJECT_IMAGE1D_BUFFER`, `CL_RG`/`CL_UNSIGNED_INT32`,
  `nelements/32*2` texels; the 2880x2880 case is 518,400). **Guarded here**; the
  refused branch leaves `extra->q_img = nullptr`, and every Intel MXFP4 kernel
  this backend selects reads the `q` buffer, so no kernel choice changes. (The
  MoE weight-image variants are `GGML_OPENCL_USE_ADRENO_KERNELS`-only.)
- **one** image pool creation (`ggml_cl_img_pool_get_or_create`) with six
  callers in the source, two of them compiled in the Intel build: flash
  attention's prefill K view (DK=512, `GGML_OPENCL_FA_PREFILL_K_IMG`) and its
  decode K view (`GGML_OPENCL_FA_K_IMG`), both opt-in; the four f16 KQ/KQV image
  mul_mat callers sit inside the Adreno `#ifdef` (`code`). **Guarded here**,
  before the sub-buffer is created (a refusal after it would leak one sub-buffer
  handle per call, reachable on the A770 only with one of the two switches set;
  the cell checks the order);
  `nullptr` is the pool's existing "no image" answer, and every caller null-checks
  it and takes its buffer kernel. The coder's prefill flash attention
  (`flash_attn_f32_f16_dpas`) is buffer-only and never a pool client.
- **all others** are under `#ifdef GGML_OPENCL_USE_ADRENO_KERNELS` (not
  compiled in the Intel build: `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`), or live
  in functions whose body compiles on Intel but whose only call sites sit inside
  that `#ifdef` (`ggml_cl_mul_mat_q1_0_f32_adreno`, the `A_image1d`/`D_image1d`
  images of `ggml_cl_mul_mat_kq_kqv_adreno`): the `#ifdef` is the guard, not the
  `use_adreno_kernels()` predicate. They are not reachable on the A770 and are
  untouched.
- The two extra fields `q_img`/`d_img` set in the SoA conversion are
  `GGML_OPENCL_USE_ADRENO_KERNELS`-only as well.

Kernel selection sites that assume an image (`code`; each already refuses a
missing image, so no site needed a new branch):

- q4_K MoE decode weight-image: `&& extra0_q4_K->q_img != nullptr` (Adreno);
- MXFP4 MoE decode weight-image: `&& extra0_mxfp4->q_img != nullptr` (Adreno);
  both strings sit inside Adreno `#ifdef`s, dead text in the Intel build; on the
  A770 the proof is that the `gpu_family == INTEL` branches of the MXFP4 mul_mv and
  mul_mv_id read `extra0_mxfp4->q`, never `q_img` (`code`);
- flash attention: `if (prefill_k_img == nullptr)` resets the kernel to
  `fa.f32_f16_split.at(dk_dv)`; the other FA K-image path tests
  `if (k_img == nullptr)` and sets `use_fa_k_img = false`;
- q8_0 and q4_K dense dp4a wimg variants: their `q_img` locals are
  null-checked where created and released.

Tests (`tests/test-opencl-image-limit.cpp`, device-free; `code`): the rule
table including **16,384 allowed / 16,385 refused** under DG2+LEO, default
driver and B60 permissive at the same sizes, a driver max below the bound; the
LEO version and DG2 name predicates; and text checks that the MXFP4 site and
the image pool ask the bound, the MXFP4/q4_K selection sites require a live
image, and FA falls back. CMake adds `test-opencl-image-limit` under
`GGML_OPENCL`.

Red-first (local, `measured-here`): four mutations each fail the test and exit
1, then the restored text passes:
- DG2 term dropped -> `FAIL: B60 LEO: 16385 allowed` (and 1036800);
- LEO term dropped -> `FAIL: A770 default driver: 16385 allowed` (and 1036800);
- bound `>` -> `>=` -> `FAIL: A770 LEO: 16384 fits`;
- MXFP4 selection loses `&& extra0_mxfp4->q_img != nullptr` ->
  `FAIL: mxfp4 wimg selection needs a live image`.
The series build (`-DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`) and
`ctest -R test-opencl-image-limit` pass.

Card gate (owed; Claude runs it in a card window, **not yet measured**):
1. **A770, fresh-host container, runtime 26.35** and **the card host, runtime
   26.27**; the reproducer still aborts under LEO:
   `cc -O2 -o /tmp/leo_repro tools/leo_image1d_buffer_repro.c -lOpenCL`
   `EnableLEO=1 NEOReadDebugKeys=1 /tmp/leo_repro 16385 1` -> abort at
   `setWidth`; `EnableLEO=1 NEOReadDebugKeys=1 /tmp/leo_repro 16384 1` -> read
   back, 0 mismatches; `/tmp/leo_repro 16385 1` (default driver) -> 0
   mismatches. This proves the bound and that the patch did not hide it.
2. **A770 full op suite** from the series build:
   `GGML_OPENCL_DEVICES=1:0 ./bin/test-backend-ops -b GPUOpenCL` (default;
   record the pass count and the failure names), then
   `GGML_OPENCL_DEVICES=1:0 EnableLEO=1 NEOReadDebugKeys=1 ./bin/test-backend-ops -b GPUOpenCL`.
   Expected (`measured-here` on 26.27/26.35 before the patch): the default set
   is 12,867 / 12,881 with 3 `ADD_*_MUL`, 3 `FLASH_ATTN_EXT` hsk=64 f16 and 8
   `SWIGLU_OAI` failing; under LEO the same failures and the same 12,867, with
   no abort at MXFP4 m=2880 n=32 k=2880. A different pass count or failure name
   is a finding, not a pass.
3. **Coder served under LEO against the default driver**, two rounds,
   `--device GPU.1` (A770). The LEO arm is the same binary through a wrapper:
   `printf '#!/bin/sh\nexport EnableLEO=1 NEOReadDebugKeys=1\nexec %s "$@"\n'
   <arcint-binary> > /tmp/arcint-leo && chmod +x /tmp/arcint-leo`
   `python3 tests/acceptance/ab.py coder --old <arcint-binary> --new
   /tmp/arcint-leo --gguf <coder.gguf> --device GPU.1 --assets <assets>
   --rounds 2 --out /tmp/ab-79`. Record decode/prefill per arm, the answers'
   md5 majority, the acceptance line and the `FAIL` lines. The prior LEO arm
   (`measured-here`, 26.27): decode +5.6-5.8 % (51.60->54.60, 52.80->55.75,
   45.35->47.90 t/s), prefill 41k -5.3 % (730.15->691.15), FAIL on prefill;
   the patch does not claim that loss fixed.
4. **The owed codegen check** on `flash_attn_f32_f16_dpas` under both drivers:
   in two empty directories,
   `CLI_DumpProgramBinaries=1 GGML_OPENCL_DEVICES=1:0 ./bin/test-backend-ops -o
   'FLASH_ATTN_EXT' -b GPUOpenCL` (default) and the same with
   `EnableLEO=1 NEOReadDebugKeys=1`. Disassemble every dumped `.bin` with
   `ocloc disasm -d igc -device dg2 -file <bin>` (drop `-device dg2` if the
   local `ocloc` rejects it), find the kernel named
   `flash_attn_f32_f16_dpas`, and record **register count and spill bytes per
   mode** (`Number of registers used`, `Spill Memory Size`). The LEO prefill
   loss sits in that kernel (+14.8 %, 21,643 -> 24,844 ms, 858 calls); a
   register/spill difference between the modes is the cause to confirm or
   refute, not a claim yet.

For an A770 LEO run the log line `ggml_opencl: LEO on DG2: weight images above
16384 texels are skipped` (once at init) and, at the first large MXFP4 tensor,
`ggml_opencl: mxfp4 weight image skipped (518400 texels, DG2 + LEO)`, are the
patch's own evidence that the branch was taken; the absence of both means the
detection did not match and the run is not a test of the patch.
## 0074-opencl-intel-qk-norm-rope.patch

The q/k RMS norm and its rotation sit adjacent on every attention layer of both
served models (`code`: `src/models/qwen35.cpp:309-321`, `qwen4exp.cpp:994-1008`:
`rms_norm -> mul(1-D [head_dim] weight) -> rope_multi`, mode 40), three kernels
a side, six launches a layer. The patch takes the three nodes as one dispatch:
`kernel_rms_norm_mul_rope_multi_f32` (`kernels/rms_norm.cl`) computes the norm's
float4 sum and scale, applies the weight on the rounded normed value and rotates
with `rope_yarn`'s cos/sin and the interleaved M-RoPE's position layout — the
three expressions `kernel_rms_norm_mul` and `kernel_rope_multi_f32` use, with
the normed row no longer stored and reloaded. The gate is the 3-node pattern in
`ggml_opencl_can_fuse`, checked before the 2-node norm+mul prefix (which would
otherwise swallow its first two nodes), and only for mode
`GGML_ROPE_TYPE_IMROPE`, a 1-D weight, a 4-aligned contiguous input row and a
contiguous f32 output; `rope->src[0] == mul` is required explicitly
(`ggml_can_fuse` also accepts the mul as rope's `src[1]`, which is not this
kernel's shape), and the sections' sum must be positive — the per-op path
asserts it and the fused kernel's `ic % sect_dims` would divide by zero on the
GPU without a word (`code`).

Switch: `GGML_OPENCL_FUSE_QK_NORM_ROPE=1`, **off by default** (`code`). It
**failed on the cards (window 42, `measured-here`)**: with the switch on,
`test-backend-ops -o RMS_NORM_MUL_ROPE` was 45 of 227 failing on the A770 and a
device fault on the B60, because the kernel stepped the positions' offset in
elements instead of bytes (round 4, `code`). **Fixed and re-run (window 45,
`measured-here`):** fused-on 227/227 on both cards; served, off vs on, coder
55.10/55.75 vs 53.92/55.08 t/s, agent 41.97/42.47 vs 42.05/42.40, Flash-Next
inconclusive; no decode gain, so it stays **off**.

`code`: the patch adds the served shapes to `test-backend-ops`'s
`RMS_NORM_MUL_ROPE` — `[128,16,1]` and `[128,16,5]` with the strided Q layout
(`nb[1] = 2*ne0*4`, the joint Q|gate projection's query half) and the 1-D
weight, `[128,8,6]` contiguous — which is where the red case is run: one term
dropped from the kernel (the epsilon, the weight, the rotation) fails those
cases on the card. Device-free, `tests/python/test_fuse_qk_norm_rope.py` reads
the patched tree: the epsilon in the scale, the weight at both rotated
positions, the norm-then-weight rounding order, the rotation's formula and the
M-RoPE position layout, the switch's default and the 3-before-2 dispatch order.
Mutations run red before restore (`measured-here` on the patched tree): the
epsilon dropped, the weight read at `w[ic]`, the weight folded before the
rounding, the default flipped on, the dispatch moved below the prefix, the
sections guard dropped, the operand-order guard dropped — each made its cell
fail; restored, the cell passes. The cells are **source-text assertions** (they
read the patched tree), not numerics.

**Rounding, stated plainly** (`code`): the fused kernel rounds the normed value
in two steps — `(x * scale)` then `* w`, kept apart by the 0049 `opaque` guard.
`kernel_rms_norm_mul` has no such guard (`rms_norm.cl`, the plain `(x * scale) *
f[i00]`), so it may fold `x * (scale * w)` itself and the two paths' bits are
not promised to agree either way. The gate is the answer-level bar
(`CLAUDE.md`) with an md5/argmax tripwire per fusion; `test-backend-ops`' NMSE
cannot see the rounding-order mutation, which is why the tripwire exists.

**Window 42, `measured-here` (the first card run).** Switch off 227/227 on both
cards; on, the A770 fails 45 of 227 and the B60 exits 134 after 6 failures
(`clWaitForEvents` error -14, `CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`).
Every failure is `mode=40` (IMROPE) with `[ROPE] ERR 1.1-2.4`; the A770's values
are the unrotated normed row (a host mirror of the reference against an
unrotated row gives the same NMSE band). Served, two rounds each, the capital
right everywhere, the default path equal to the package binary: coder (A770)
53.27/52.90 vs 56.03/55.70 t/s (-5 %), MTP acceptance 35.8 -> 32.6 %; agent
(B60) 39.37/39.35 vs 42.40/42.32 (-7 %), acceptance 39.0 -> 34.9 %; Flash-Next
33.35/34.58 vs 33.95/33.93, expert hits 81.2 -> 83.5/82.9 %.

**Window 43, `measured-here`: the defect, found and fixed (`code`).** The
diagnostic print of round 3 read **pos=0/0/0/0 for all four position streams of
every dispatch, 67 of 67** (`~/w42/diag74-a770.log`: `theta=0, cos=1, sin=0`,
e.g. `n_dims=8192 ... sect=8190 secs=2730/2730/2730/0 pos=0/0/0/0`), while the
test fills the positions with `rand() % 512` (`code`: the case's
`initialize_tensors`). That is theta 0 for every row — the unrotated output the
NMSE band predicted. The cause is the kernel's own first line: `src1` is a
`global int *` and `src1 = src1 + offset1` stepped the operand offset in
**elements** (x4 bytes) instead of bytes, reading a wrong, zeroed region; the B60
then read past the buffer and faulted. `kernel_rope_multi_f32` casts every
offset through `global char *`; so does this kernel now, for src1
(`src1 = (global int *)((global char *) src1 + offset1)`). The fused-on suite and
the served A/B are owed to the next window; no speed is claimed.

**Why it was slower — from the code (`inferred`; the A/B has not been re-run).**
The fused dispatch ran one work-group a row with lws = `ne00` (128), while the
rope phase gives one lane a rotated pair: half the lanes idle in the rotation,
and the group waits behind the norm's two barriers before the rotation starts.
The two kernels it replaces have the same norm geometry (one group a row) and an
all-active 64-lane rope group with no barrier. The fusion's only saving is 2
launches a side, ~0.4 % at the wall by the audit's pricing, below its +0.5 %
bar. The patch now sets lws to one lane a pair (rounded to a subgroup, capped so
the subgroup count fits the local `sum`), which removes the idle lanes but not
the barrier serialization. Whether that clears the bar is the next A/B's answer,
not this text's.

To prove the fix in the next window (print the offset the host passed and the
four ints the kernel reads, then restore):

    sed -i '/float2 cos_sin_theta = rope_yarn_fused/a\            if (i01 == 0 && i02 == 0 && get_local_id(0) == 0) { printf("FUSED offset1=%lu n_dims=%d n_offs=%d sect=%d secs=%d/%d/%d/%d pos=%d/%d/%d/%d theta=%f cos=%f sin=%f\\n", offset1, n_dims, n_offs, sect_dims, sections.s0, sections.s1, sections.s2, sections.s3, pos[0], pos[ne02], pos[2*ne02], pos[3*ne02], theta_base * pow(freq_base, inv_ndims*iw), cos_sin_theta.s0, cos_sin_theta.s1); }' ggml/src/ggml-opencl/kernels/rms_norm.cl
    GGML_OPENCL_DEVICES=1:0 GGML_OPENCL_FUSE_QK_NORM_ROPE=1 ./bin/test-backend-ops -b GPUOpenCL -o RMS_NORM_MUL_ROPE
    git checkout ggml/src/ggml-opencl/kernels/rms_norm.cl

`pos=` must read the case's position values (not four zeros) with `offset1` the
tensor's byte offset; the same print on the B60 (`GGML_OPENCL_DEVICES=0:0`) must
not abort.

## 0075-opencl-intel-cpu-miss-prefetch.patch

Two pieces from the audit's item 6, behind `GGML_OPENCL_CPU_MISS_PREFETCH`
(**0 off by default**; `1` the rows' hints, `2` also the page-touch pass,
`code`): one block-ahead `__builtin_prefetch` in each of the six row functions,
and a page touch over a missed expert's three bank slices at publish.

**The premise is not Strata's — a build difference, named** (`code`). Strata's
cold reads are file-backed rows (78 % of its CPU lane, `paper`, St 9e3fe046).
Here a bank is USM host memory (`clHostMemAllocINTEL`, the `usm_host_alloc`
entry points in `ggml-opencl.cpp`, the buffer context's comment) and **every
byte is written at load** by `ggml_cl_usm_set_tensor` — the planar loop and the
IQ4_NL split copy — so its pages are resident before the first miss, and
`drop_caches` evicts page cache, not this memory. Prediction: major faults near
0 and the level-2 pass buying nothing, by construction. The window's fault count
decides; no number is claimed here (`measured-here` owed).

**The hints are narrow, and said so** (`code`): each names only the next block's
field 1 (the payload plane), two blocks ahead — 64-128 bytes, inside the
hardware prefetcher's reach; fields 0 (the scale) and 2 (the signs/qh plane) are
unhinted. A hint's gain, if any, is expected in the run's spread, not a step.

**The touch is a synchronous barrier on the lane's critical path** (`code`):
`ggml_cl_miss_pool::run` claims tasks and spins until `done == n`, so the
calling (coordinator) thread waits; "in parallel" is across experts and the
pool's workers only, not overlapped with the card's round. The slow path calls
it once a pair, so a multiply-routed expert's slice is touched once a pair —
redundant, kept for the one pass's simplicity.

**Deviation, named** (`code`): the sketch's hint is "ahead of the next expert's
first field"; the build hints the next block within the same expert — a rows'
task sees one expert only. The cold-page half is built as sketched.

`measured-here`, device-free: `tests/test-opencl-miss-cpu.cpp` gains the level-1
parity arm (the rows' bits identical at level 0 and level 1 — what a hint must
be), the ahead-distance arm and the residency arm (a 16-page mapping dropped to
absent, one byte a page touches it all resident; a touch striding 8192 leaves 8
of 16 absent and fails — the pass's mechanism, not its premise).
`tests/python/test_cpu_miss_prefetch.py` reads the patched tree for the
default-off switch, one hint in every row function, the volatile touch and the
`>= 2` guard in both compute paths; its six red mutations (a hint dropped from
`row_iq3s_1`, the distance 0, the stride, the non-volatile pointer, the default
on, the guard removed) were each seen failing, and the C cells' two (stride,
distance) likewise. Both cells are **source-text assertions**, not numerics.

**Window 42, `measured-here` (the card run; the record).** Flash-Next,
`GGML_OPENCL_CPU_MISS_MAX=64`: level 0 34.12/34.40 t/s with **0 and 10 major
faults** over the leg, level 1 34.28 t/s (in the run's spread), level 2
30.12 and 33.73 with one leg at **+479,730 major faults**. The premise holds
(faults ~0 at level 0), level 1 buys nothing, level 2 costs, so **the switch
stays off**.

**The level-2 fault spike, from the code** (`code`): the touch addressed a
missed expert's slices at `bptr[k] + b * bank->nb[2]`. For an IQ4_NL bank that is
the SPLIT form — `ggml_cl_usm_set_tensor` lays the scales and the quants as two
arrays over every expert — so the tensor's own `nb[2]` stride is not expert `b`'s
offset and the walk can run past the split's region, into whatever mapping
follows it (the GF mmap is file-backed: a major fault a page). The patch now
addresses the split arrays by the block index `row_dot_iq4nl` reads
(`split_d + b*nblk*2`, `split_q + b*nblk*qb`) and keeps `nb[2]` only for the
block/planar banks. It is a fix for the spike, not for a default: level 2 still
buys nothing by the premise.

Owed on the cards, unchanged: the cold/warm pair at level 0 and the same A/B on
the other units; the audit's bar — major faults above ~5 % of the CPU lane —
decides a default. With faults ~0 already measured, it is ~0.

## 0080-llama-mtp-kv-only.patch

The MTP head's prompt pass computed only its K/V. A draft context's catch-up
decode (the head's prompt fill) has no output rows: it exists so the later
draft steps can read this layer's K/V. Ours built the whole MTP block for every
prompt row anyway — `Qcur_full` (query + gate), attention, the gated output
projection, the residual, FFN and the shared head — and cut only the output
rows (patches 0010 and 0067 cut the *outputs* rows, not the graph; `code`:
`src/models/qwen35.cpp`'s `graph_mtp` constructor, `qwen35moe.cpp`'s the same).
**Evidence class of the mechanism: `code`** (fork 595d554c90, `LLAMA_MTP_KV_ONLY`
in `arc-b580/src/models/qwen35.cpp:756-793`): the fork gates the graph after
the cache store when `n_outputs == 0` and masked extraction is on, keeping the
K/V ops identical and in the same order on the same shapes. The quality class
is `exact`: the retained K/V rows are the full graph's (`code`; the fork's byte
equality claim, not measured here).

The patch, behind `LLAMA_MTP_KV_ONLY` (**off by default**; `1` enables it,
`0` keeps the full graph, `code`): after the cache store the graph returns —
K projection, reshape, K norm, RoPE, the optional cache rotations, then V, then
`expand V`, `expand K`, `cpy_k`, `cpy_v`, the store half of `build_attn` in its
own order. The Q projection is kept for `n_tokens <= 8`, as the fork does (its
same-input mat-mul run's tile depends on the largest member; this OpenCL
backend has no such run today — `code`, the only MUL_MAT fusion is 0019's
gate+up+GLU — kept for parity with the fork's rule). The branch is taken only
when `n_outputs == 0`, `cparams.embeddings_nextn_masked`, and the layer has no
fused `wqkv` and no K/V bias — a fused projection would not be reproduced by
the branch, so those layers keep the full graph (`code`). With the switch off
the graph is byte-identical to before.

**Scope (`code`)**: `qwen35` (the dense agent) and `qwen35moe` (the coder),
both share the block shape. `qwen4exp` (Qwen3.8-Flash-Next)'s MTP graph is a
hyper-connection graph whose K/V store sits inside `build_layer_attn`; it does
not share the shape, so it is not in this patch. `qwen3next` shares the shape
but is not a served unit; it is left as it is.

Device-free (`code`): `tests/python/test_mtp_kv_only.py` reads the patched tree
and asserts the branch's store half — the K mat-mul, reshape, K norm and RoPE
the full graph's `build_qkv(reshape=false)` builds, V the same, V expanded
before K, K stored before V, the rotations before the expands — that nothing
follows the store (no attention, output projection, FFN or head), that the
branch is gated on a masked no-output decode on a layer without a fused wqkv or
K/V bias, and that the switch defaults off. Red mutations run before restore
(`measured-here` on the patched tree): cpy_v dropped, the default flipped
(`e == nullptr || ...`), the K norm dropped from the branch, the expand order
swapped, the early `return;` dropped — each made its cell fail; restored, the
cell passes. These are **source-text assertions** (they read the patched
tree), not numerics.

Card arms (owed; Claude runs them in a card window, **not yet measured**):
1. **B60, agent (dense 27B), MTP 5**, the 18K/55K corpus prompt from
   `tests/acceptance/cells/served_check.py`, two arms in one window, switch off
   the old binary's arm and on the new one's:
   `printf '#!/bin/sh\nexport LLAMA_MTP_KV_ONLY=1\nexec %s "$@"\n' <arcint-binary> > /tmp/arcint-kvonly && chmod +x /tmp/arcint-kvonly`
   `python3 tests/acceptance/ab.py agent --old <arcint-binary> --new /tmp/arcint-kvonly --gguf <agent.gguf> --device GPU.0 --assets <assets> --rounds 2 --out /tmp/ab-80-agent`.
   Record `prefill-55k`/`prefill-138k`, the decode rate, `draft accept` and
   `verify` of both arms, and the texts md5; the bar is prefill up with the
   same texts and the same accept counters (a K/V that differed would change
   the draft's proposals and its acceptance; the audit's price is ~5 % on long
   prompts, `paper`). The same with the coder on the A770 (`--device GPU.1`,
   `ab.py coder`, MTP 4) — its prefill-41k is the prompt-side arm.
2. **Draft K/V equality, stated as an observable**: three prompts served under
   both arms in one process each; the `draft accept %` and the `verifies` of
   the runs must be identical (the runs are the same workload, so any K/V row
   difference moves them). A direct per-row dump is not available in either
   arm; if the accept counters differ, the switch stays off and the finding is
   recorded, not absorbed.

Window 58 (`measured-here`, 2 rounds an arm, the bundle `LLAMA_MTP_KV_ONLY=1 LLAMA_SYNC_TRIM=2` against off):
the coder (A770) PASS, texts md5 identical, decode +1.8 / +2.0 / +1.8 %, prefill-41k 729.75 -> 777.80 (+6.6 %);
the agent (B60) PASS, md5 identical, prefill equal (55k 715.85 -> 714.95, 138k 503.25 -> 502.70); Flash-Next (B60)
PASS, prefill equal, decode 32.40 -> 32.90 inside a 4 % run spread. Window 57 (one run each) put the coder's
prefill gain on 0080 and its decode gain on 0083 bit 2.


## 0081-llama-gdn-state-rows.patch

The GDN recurrent state read straight from the cache. Ours gathers each layer's
live state rows into a temp before the gated delta-net runs
(`src/llama-graph.cpp`'s `build_rs`: `ggml_get_rows(states, state_copy)`), and the
kernel then reads the temp; patch 0050 folds only the snapshot *writes* into the
cache, so the gather and its ~3 MB a layer are paid every decode and verify
(`code`, the drawn state; the fork's price is +3.8 % TG and a dirty temp slowing
the next GEMV ~15 %, `paper`). **Evidence class of the mechanism: `code`** (fork
9b4be76a1c; `arc-b580/src/models/qwen35.cpp`'s `build_rs_cache_view`, the CPU
reference's `rows` mode over `src[6]` in `ggml/src/ggml-cpu/ops.cpp:10815-10845`,
the SYCL kernels' `s_ids` and `s_row_stride` reads). Quality class `exact`: the
bytes are the gathered row's (`code`; the fork measured 400 greedy tokens
identical).

The patch adds the rows form of the op — `ggml_gated_delta_net_rows(ctx, q, k,
v, g, beta, states, rows, K)` sets `src[6]` to an I32 array of cache row indices
over the 2D cache view (`states->ne[0] == D`, `rows->ne[0] == n_seqs`; output
layout identical to the gathered form's) — and the build side:
`build_rs_cache_view` returns the cache view after the same zero-state hygiene,
gathers and relocates only the *extra* rows (`inp->s_copy_tail`, `n_rs - n_seqs`,
the same destination `build_rs` writes), and the GDN op reads this sequence's
live row `inp->s_copy_main[0]` inside the kernel. The three served archs
(`qwen35`, `qwen35moe`, `qwen4exp`) pass the row index.

**The GDN family's source slots, one signal each** (`measured-here`, the card
host, 2026-11-02; defect): src[6] was taken three times — 0051's alpha (on by
default), this patch's row index, 0086's records. Served with 0051's fold on,
src[6] held the F32 alpha: the OpenCL `supports_op` declined the node, it fell
to the CPU, and the CPU kernel asserted `GGML_ASSERT(src_rows->type ==
GGML_TYPE_I32)` — **every unit crashed with every new switch off**. The fix,
which is this patch: `op_params[1]` is the family's mode word
(`GGML_GDN_MODE_GATHERED 0`, `_ROWS 1` in `ggml.h`, `_REPLAY 2` from 0086) and
both GATED_DELTA_NET builders set it; the CPU op, the OpenCL `supports_op` and
the OpenCL dispatch select their state path from that word and never from
whether src[6] is set; the rows index stays `src[6]`; the raw gating projections
of 0051 ride `src[8]`/`src[9]` (the helper moved to
`src/models/gdn-gate-inputs.h`, which the builders and the device-free cell
share) and 0051's fold matches `gd->src[8] == al` / `gd->src[9] == br`. The
replay op (0086/0087) keeps `src[6]` records / `src[7]` control and carries the
REPLAY mode word, which its CPU and OpenCL consumers assert. Evidence class of
the slots: `code` (the consumers' reads); of the crash: `measured-here`.

**Switch `LLAMA_GDN_STATE_ROWS`, off by default** (`code`): `1` builds the rows
form, `0` keeps the gather. With it off the graph and every op's arguments are
as before, byte-identical (the slot move is liveness only: 0051's fold finds the
same two tensors, and off-graph bytes are unchanged).
The build gate (`llama_gdn_state_rows_enabled`)
declines a **multi-sequence** ubatch — `s_copy` may map a live row into the
extra relocation's destination — and any batch the fused GDN op would not take
(the autoregressive path needs a 4D state). The gate's single-sequence condition
is the audit's, not a kernel limit (the CPU and OpenCL row reads take any
`n_seqs`; the multi-sequence C arm proves it).

**Ordering, stated plainly** (`code`): the extra relocation's cpy is expanded
before the GDN, so a live row that fell inside its destination would be written
before it is read. That is the fork's own documented hazard
(`arc-b580/src/llama-graph.cpp:3684-3694`); with `n_rs_seq > 0` the live row maps
into `rs_idx * size + src0` (rollback plane) while a reorder-less extra row maps
into plane 0, so the served path's rows do not overlap — the multi-sequence
decline keeps the general case out. Confirm it in the served arm: the answers
and the draft acceptance counters equal (below).

Device-free (`measured-here`): `tests/test_gdn_gate_slots.cpp` builds the served
graph's gate chain — `beta = sigmoid(b)`, `g = reshape(mul(softplus(add(alpha,
dt_bias)), A))` — and attaches the raw projections through the shared header
`gdn-gate-inputs.h`: the plain node with 0051's fold active (all new switches
off), the rows form and the replay node all compute the plain path's bytes
(replay: its attention region, and its whole output is unchanged by the extra
sources). Red (`measured-here`): `gdn-gate-inputs.h` writing `src[6]`/`src[7]`
again (the served pre-fix form) makes the plain+extras arm abort in the CPU op's
I32 assert; the plain builder announcing ROWS aborts the same way.
`tests/python/test_gdn_gate_slots.py` asserts the mode word's definition and use
in both builders, the CPU op, the OpenCL `supports_op` and the dispatch, the
`src[8]`/`src[9]` slots and 0051's match, and the replay's own slots; red
mutations run: the helper on the old slots, the plain builder's mode flipped, the
CPU op reading `dst->src[6] != nullptr` again, 0051's prepass on the old slots.

`tests/test_gdn_state_rows.cpp` runs both forms on
the CPU backend at four shapes (one sequence with `ids[0] = 2`, two sequences
with `ids {1, 3}` or `{2, 3}`, K = 1 and K = 3, three tokens) and requires the
packed outputs — attention scores and snapshots — byte-identical; the cache is
wider than the rows used, so a stride read from `nb[3]` cannot add up.
`tests/python/test_gdn_state_rows.py` reads the patched tree for the switch's
default, the multi-sequence decline, the fused-op term, the extra-only gather,
the three archs' row index, `src[6]`, the CPU's `nb[1]` stride and
`state_rows[iv3]`, both kernels' `rows[seq]` read and the dispatch. Red
mutations run before restore (`measured-here` on the patched tree): the switch
flipped on, the `n_seqs != 1` decline dropped, the fused-op term dropped, the
main rows gathered again, the CPU's row index replaced by the ordinal, the
Intel kernel's rows term dropped, the dispatch's `use_rows` forced to 0 — each
made its cell fail; restored, it passes. The Python cell is a source-text
assertion; the C cell is numeric.

Card arms (owed; Claude runs them in a card window, **not yet measured**):
1. **`test-backend-ops -o GATED_DELTA_NET`** from the series build on the B60
   and the A770: `GGML_OPENCL_DEVICES=0:0 ./bin/test-backend-ops -b GPUOpenCL -o
   GATED_DELTA_NET` (and `1:0` on the A770). The patch adds four rows cases (TG,
   TG with 2 sequences, verify with K = 3 and 2 sequences, PP-64) whose reference
   is the CPU rows form, so the kernel's row read is compared directly. Record
   the pass count with and without the rows cases; expected: every existing case
   unchanged and the four rows cases pass.
   Red on the card: take the Intel kernel's `state_base` rows term out
   (`use_rows ? ... : ...` -> `((ulong)seq * H_v + head) * state_size` in
   `gated_delta_net_intel.cl`), rebuild, and the four rows cases fail; restore.
2. **B60 agent (dense 27B), MTP 5, 138K needle from
   `tests/acceptance/cells/served_check.py`**, two arms in one window:
   `printf '#!/bin/sh\nexport LLAMA_GDN_STATE_ROWS=1\nexec %s "$@"\n' <arcint-binary> > /tmp/arcint-gdnrows && chmod +x /tmp/arcint-gdnrows`
   `python3 tests/acceptance/ab.py agent --old <arcint-binary> --new /tmp/arcint-gdnrows --gguf <agent.gguf> --device GPU.0 --assets <assets> --rounds 2 --out /tmp/ab-81-agent`.
   Record decode, prefill-55k/138k, `draft accept` and the texts md5. Bar: decode
   >= +2.5 % with the same texts and acceptance (`paper` price +3.8 % TG); if the
   texts differ or the acceptance counters move, the switch stays off and the
   finding is recorded.
3. The same with the coder on the A770 (`--device GPU.1`, `ab.py coder`, MTP 4)
   and with Flash-Next (B60, MTP 3, IQ3_XXS): answers right, decode recorded.
4. **The defect's card check (first, before any A/B)**: the served binary with
   **0051's fold on, the default** (`GGML_OPENCL_FUSE_0051_PART=3` is the
   default part; unset `GGML_OPENCL_FUSE_0050` = 3) and **every new switch off**
   (`LLAMA_GDN_STATE_ROWS` unset, `LLAMA_GDN_REPLAY` unset) must start and
   answer at a short prompt on each unit. Before this fix it crashed at the first
   GDN node (the CPU op's I32 assert). The ground source is the three units'
   one-shot arm:

      cd <arcint build>
      ./bin/arcint --llama-gguf <agent.gguf> --device GPU.0 -p "<a 20-token prompt>" -n 8
      # the coder on GPU.1, Flash-Next on GPU.0, the same line
      # expect: exit 0, "GDN gating folded into the kernel for N of N GDN nodes"
      # in the log, and 8 tokens out

   Red on the card (the pre-fix form, not to be kept): `gdn-gate-inputs.h`'s two
   assignments back to `src[6]`/`src[7]`; rebuild; the same line crashes with
   `GGML_ASSERT(src_rows->type == GGML_TYPE_I32)`. Restored, it answers.
   `test-backend-ops -o GATED_DELTA_NET` stays the kernel's own arm (1).

## 0082-llama-gdn-qk-l2-norm.patch

One l2_norm for the hybrid layers' adjacent q/k head groups, and the bf16 gate
dispatch read. **Evidence class of the mechanism: `code`** (fork's form:
`arc-b580/src/models/qwen35.cpp:548-554`, one `ggml_l2_norm` over the joint
`[head_k_dim, 2*num_k_heads, n_seq_tokens, n_seqs]` view of the conv output,
then q and k as views of its result). Ours normed q and k separately — two
`build_gdn_l2_norm` pairs (`rms_norm` + `scale`) over strided views, four nodes
a layer (`code`: `qwen35.cpp:441-442`, `qwen35moe.cpp`, `qwen4exp.cpp`); the
fork's Metal bundle measured +7.7 % 2B / +2.6 % 27B for the hybrid bundle,
`paper`, not separable to this node pair.

The patch adds one device-free helper, `src/models/gdn-qk-l2-norm.h` (ggml.h
only), holding `build_gdn_l2_norm` and the joint form: a `view_4d` of the conv
output keeping its group stride (`nb1_qkv`), one `rms_norm(x, eps/n)` and
`scale(1/sqrt(n))` over it, then q and k as views of the dense result at offsets
`0` and `num_k_heads * nb[1]`. The same normalisation runs per row, so the bytes
are the two separate norms' (`exact`; the fork claims byte-identical outputs for
its fusions, `code`). The three served archs call it; `clef` and `qwen3next`
share the two-norm form but are not served units and are left as they are.

**Switch `LLAMA_FUSE_QK_L2_NORM`, off by default** (`code`): `1` builds the
joint form, `0` the two separate norms — byte-identical to before.

**Item 8's second half, the bf16 48-row gate matvec: no missing piece, and the
dispatch read says so** (`code`). The audit's "not verified, `inferred` from the
dispatch tables" is now read: bf16 is stored as f16 on device
(`ggml-opencl.cpp:29086`, `src0t = BF16 ? F16 : type`), and the F16 matvec
branch (`:29202-29227`) takes `src1->ne[1] <= 8` columns with
`(src0->ne[1] <= 64 || (src0->ne[1] <= 1024 && src1->ne[1] >= 2))` — a
`[5120, 48]` gate weight at 1-6 columns takes `kernel_mul_mv_f16_f32_ksplit`
(one column) or `..._nc` (2-6), a work-group a row, **not** the tiled GEMM.
The fork's `mmvf.cu` fix (bf16 mat-vec under 64 rows at 2-8 columns,
10.5 -> 4.4-7.3 us) therefore has no counterpart to port here: the dispatcher
already routes it there. No kernel work in this patch; the cell asserts the
conditions, so a future narrowing of them fails it.

Device-free (`measured-here`): `tests/test_gdn_qk_l2_norm.cpp` builds a strided
conv mix on the CPU backend and compares the joint form's q and k bytes against
the two separate norms at five shapes (head 4 / 2 k heads / 1 and 3 tokens / 1
and 2 sequences; head 8 / 3 k heads / 2 tokens / 2 sequences; a compact mix
without the v tail). The joint form's q and k are views of its dense result, so
their token stride carries both head groups; the cell walks the views' logical
elements with their own strides instead of copying linearly.
`tests/python/test_gdn_qk_l2_norm.py` reads the patched tree: the switch default
off, the two-norm arm kept in all three archs, the joint view's group stride,
the k view's head-group offset, and the bf16 matvec conditions. Red mutations run
before restore (`measured-here` on the patched tree): the switch flipped on, a
model's two-norm arm dropped, the k view's `num_k_heads *` dropped, the joint
view's group stride set contiguous, the bf16 row term narrowed to 48 — each made
its cell fail; restored, it passes. The Python cell is source text; the C cell is
numeric, and its numeric red was run through the recipe itself: with the joint
view's group stride taken as the q|k row width (`nb1_qkv` ->
`ggml_row_size(type, head_k_dim)`), the cell failed every shape whose token or
sequence stride is exercised (3 of 5: three tokens, two tokens x two sequences,
the compact two tokens x two sequences), and the one-token shapes passed; the
fix (restored `nb1_qkv`) makes all five pass. The bug was in this patch's own
first draft, found by the cell.

Card arms (owed; Claude runs them in a card window, **not yet measured**):
1. **B60 agent and A770 coder, served, switch off vs on**, two rounds:
   `printf '#!/bin/sh\nexport LLAMA_FUSE_QK_L2_NORM=1\nexec %s "$@"\n' <arcint-binary> > /tmp/arcint-qk2 && chmod +x /tmp/arcint-qk2`
   `python3 tests/acceptance/ab.py agent --old <arcint-binary> --new /tmp/arcint-qk2 --gguf <agent.gguf> --device GPU.0 --assets <assets> --rounds 2 --out /tmp/ab-82-agent`
   and `ab.py coder ... --device GPU.1`. Record decode, prefill, `draft accept`
   and the texts md5; the bar is >= +2 % bundled with identical answers (`paper`),
   and the md5/argmax tripwire is the answer-level gate. If no gain, the switch
   stays off.
2. **Launch count** before/after: one GDN layer's encoded nodes a token,
   `GGML_OPENCL_...`? there is no op-count dump; use the served `LLAMA_LOG`
   graph lines or a counters arm. Owed, stated so.
3. B60 Flash-Next (IQ3_XXS, MTP 3): answers right, decode recorded — the same
   A/B form.

## 0083-round-trip-trim.patch

Round-trip discipline on the decode/MTP step, the three cuts that apply to
this stack, behind one bit mask. `LLAMA_SYNC_TRIM` defaults to **0 here** (all
bits off, byte-identical to before); the fork's commits below default to 3.
The mechanisms are `code` at the cited lines; the queue and per-read costs are
the fork's `measured-here` on its SYCL stack, **not measured on the B60 here**.
No ggml op or kernel changes, so `test-backend-ops` has no new case; the card
arm is a served A/B.

**The count, before → after** (code reading, this tree; an MTP verify of 5
rows, the drafter's per-step read, one attention graph):

| round trip | before | after (bits 1/2/4) | where |
|---|---|---|---|
| context synchronizes a verify (5 logit rows + the nextn row) | 6 | 1 | `src/llama-context.cpp` C getters (each `synchronize()`); verify reads at `src/exec/backend_llama.cpp:2611` and `src/exec/llama_spec.cpp:495` |
| context synchronizes a draft step, head on | 1 | 1 | one nextn getter a step, `src/exec/llama_spec.cpp:632` |
| context synchronizes a draft step, head off, one lane | 2 | 1 | `src/exec/llama_spec.cpp:635,636` (logits and nextn, same step) |
| KV rotation inputs uploaded a graph (target verify and draft alike) | 2 split inputs, 2 blocking copies, ~272 KB | 0 | `src/llama-kv-cache.cpp:1692,1708` (builders), `:2066,2078` (setters) |
| small device→host reads for a batch's sampled ids, n rows | n | 1 | `src/llama-context.cpp:3242` (one copy); per-row read stays the fallback |

Bit 1 (`code`, fork 60ef575f3b): `synchronize()` returns before the device wait
when `!sync_pending` and not under pipeline parallelism; `sync_pending` is set
at encode, decode, graph compute and the epoch iterator, cleared at the wait
(`llama-context.cpp:1778`, `:2573`, `:2872`, `:3846`, `:4943`). A getter after
the first on a clean context finds its results already complete; it cannot wait
for another context's work (the one in-order queue is shared).

Bit 2 (`code`, fork 4ec75c3416): the cache allocates `attn_rot_dev` once in its
own buffer type when there is one non-host buffer (`llama-kv-cache.cpp:389`),
fills it once from `attn_rot_hadamard`, and `build_input_k_rot/v_rot` return
those tensors; `set_input_*_rot` skip a non-input tensor. Same Hadamard values,
copied once instead of uploaded a graph. `attn_inp_k_rot` is 256×256 f32,
`attn_inp_v_rot` 64×64.

Bit 4 is the form of "one id read where there are several" that applies here
(`code`): with backend samplers, decode copied each sampled id with its own
small async read (`copy_tensor_async_rows`). The graph now concatenates the
batch's sampled ids into one I32 tensor after the sampler chains
(`llama-graph.cpp:4352`, n_rows−1 tiny CONCAT kernels) and decode reads it once
(`llama-context.cpp:3242`); the same integers in the same row order. This bit
is **not on arcint's served path today**: arcint samples in its own `Sampler`
and does not use backend sampling. The fork's draft-chain form (all draft ids
of a one-graph chain, fork b0376aec39, ~0.2 ms a small read there) does not
apply: arcint's drafter runs one graph a draft step, each step's token being
the next step's input, and the expert-route trace's ids are already one batch
(patch 0069).

Switch-off equivalence: bits 0 leave each call as it was — the same returns,
the same blocking copies, the same per-row reads. Bits on change only the
number of waits and copies, never a value.

Red cases, each run and seen failing before it was restored (device-free cell
`tests/python/test_round_trip_trim.py`, `ctest -L unit`, `round-trip-trim`):
(a) `LLAMA_SYNC_TRIM` default 0 → 3: the default-off cell fails; (b) the
predicate's `!sync_pending` dropped: the predicate cell fails; (c) the
device-rotation return dropped from `build_input_v_rot`: the rotation cell
fails; (d) the sampled-id concat's bit 4 → 8: the one-read cell fails.

Recipe: `~/bin/arcint-recipe.sh ~/src/ligence-gaps-b gaps-b580-b` — green,
18/18 unit tests (2026-11-02), the patch applies on the pin.

Card arm (Claude, one announced window, no GPU run here): the same-binary
served A/B with the switch off against bits 1+2+4 on, two runs each, alternated;
the answers and the answer-level bar (facts, needle, task), decode and prefill
per phase. Exact commands, with the binary named by the package under test:

    cp bin/arcint /tmp/arcint-trim
    printf '#!/bin/sh\nexec env LLAMA_SYNC_TRIM=0 /tmp/arcint-trim "$@"\n' > /tmp/arm-off; chmod +x /tmp/arm-off
    printf '#!/bin/sh\nexec env LLAMA_SYNC_TRIM=7 /tmp/arcint-trim "$@"\n' > /tmp/arm-on;  chmod +x /tmp/arm-on
    python3 tests/acceptance/ab.py flashnext --old /tmp/arm-off --new /tmp/arm-on \
        --gguf <model.gguf> --device GPU.<N> --assets <dir> --rounds 2
    python3 tests/acceptance/ab.py agent     --old /tmp/arm-off --new /tmp/arm-on \
        --gguf <model.gguf> --device GPU.<N> --assets <dir> --rounds 2

Bar: decode (the target phase with MTP on) improves; prefill stays inside the
run's spread. The fork's own numbers (2 syncs + 2 blocking copies a graph,
~0.2 ms a small read) are its stack's; the B60 price is this window's to say.

Window 58 (`measured-here`, 2 rounds an arm, the bundle `LLAMA_MTP_KV_ONLY=1 LLAMA_SYNC_TRIM=2` against off):
the coder (A770) PASS, texts md5 identical, decode +1.8 / +2.0 / +1.8 %, prefill-41k 729.75 -> 777.80 (+6.6 %);
the agent (B60) PASS, md5 identical, prefill equal (55k 715.85 -> 714.95, 138k 503.25 -> 502.70); Flash-Next (B60)
PASS, prefill equal, decode 32.40 -> 32.90 inside a 4 % run spread. Window 57 (one run each) put the coder's
prefill gain on 0080 and its decode gain on 0083 bit 2.


## 0084-llama-fast-topk.patch

The long vocabulary's top-k, selected straight from the logits row
(`code`, fork a3586a10d8). The top-k sampler partial-sorts a candidate array
built over the whole vocabulary; over 248320 entries that is a full copy plus
`std::partial_sort` per sample. `llama_top_k_from_logits` (new, declared in
`include/llama.h`, defined in `src/llama-sampler.cpp`) starts from the k first
entries, replaces the heap top by every strictly larger logit in index order
and sorts the heap — the steps of libstdc++ `std::partial_sort`, so entries
and the order of equal logits are identical, ties, NaN and inf included. No
rounding, no reordering.

**Count, before → after**: a sample's candidate work goes from 248320 entries
built and partial-sorted to k entries' heap plus one scan. The entry count is
`code`; the microseconds the fork measured (200K entries, k=1 241→9.6 µs,
k=10 647→71 µs; +1.9 % greedy and +1.85 % Hermes decode with MTP) are
`measured-here` on its stack, not on the B60 here.

The patch is a library entry point; the switch lives at its caller, arcint's
`Sampler::collect_candidates` (`src/core/sampler.cpp:1107`): with
`LLAMA_SAMPLER_FAST_TOPK=1` (default **0**) and `top_k > 0` and the top-k
smaller than the vocabulary, the row's top-k comes from the helper; otherwise
the old full-array path runs unchanged, byte-identical. The probe path
(`top_k <= 0`, 2048 candidates) and the full-sort fallback keep
`std::partial_sort`. A future llama common-sampler caller can use the same
entry point; the common library is not built by arcint (`LLAMA_BUILD_COMMON
OFF`), so nothing in the product changes with the patch alone.

The exactness cell is device-free: `tests/test_fast_topk.cpp` checks the
helper against `std::partial_sort` over random arrays at 64/1024/4096/248320
entries, k 1/2/10/128/2048, 40 arrays a shape, plus ties, all-equal, NaN and
inf (the fork's own test shape). It runs in the `unit` ctest. Red mutation, run
and seen failing before it was restored: the tie guard `>` made `>=` in both
scan loops (`if (logits[i] >= heap[0].logit)`) — the tie and all-equal cells
fail (`FAIL fast_topk_ties_keep_the_partial_sort_order`,
`FAIL fast_topk_all_equal`). Rows with the mutation: 17 of 18 ctests passed,
`unit` failed at `test_fast_topk.cpp:55/56`.

Recipe: `~/bin/arcint-recipe.sh ~/src/ligence-gaps-b gaps-b580-b` — green,
18/18 (2026-11-02). No `test-backend-ops` case: no ggml op changes.

Card arm (Claude, one announced window): the same-binary served A/B with
`LLAMA_SAMPLER_FAST_TOPK=0` against `1`, two runs each, alternated, on the
agent (Hermes-style top_k 20) and greedy/MTP; answers the same bytes, decode
improves. Exact commands:

    cp bin/arcint /tmp/arcint-topk
    printf '#!/bin/sh\nexec env LLAMA_SAMPLER_FAST_TOPK=0 /tmp/arcint-topk "$@"\n' > /tmp/arm-old; chmod +x /tmp/arm-old
    printf '#!/bin/sh\nexec env LLAMA_SAMPLER_FAST_TOPK=1 /tmp/arcint-topk "$@"\n' > /tmp/arm-new; chmod +x /tmp/arm-new
    python3 tests/acceptance/ab.py agent     --old /tmp/arm-old --new /tmp/arm-new \
        --gguf <model.gguf> --device GPU.<N> --assets <dir> --rounds 2
    python3 tests/acceptance/ab.py flashnext --old /tmp/arm-old --new /tmp/arm-new \
        --gguf <model.gguf> --device GPU.<N> --assets <dir> --rounds 2

Bar: the answer-level gate (the same answers, mean KL no worse than the
baseline arm's by 0.03 nats, argmax agreement down by at most 1 point);
decode improves, prefill stays inside the run's spread.

## 0086-ggml-gdn-replay-op.patch

The gated delta-net's rollback by replay, the op and its CPU reference. Ours
rolls a recurrent sequence back by keeping whole state snapshots (patch 0050):
`n_rs_seq + 1` S planes per layer, and a verify's rollback reads the plane. The
fork keeps one base state and, for each token after it, a record of the step's
own small vectors — the value delta (`S_v * H` floats), the key (`S_v * H`) and
the gate (`H`) — and re-applies them onto the base on rollback, so a rollback
costs `R = H * (2 * S_v + 1)` floats per token a layer instead of a whole
`S_v * S_v * H` plane. **Evidence class of the mechanism: `code`** (fork
`arc-b580`: `ggml/include/ggml.h:599`, the CPU reference
`ggml-cpu/ops.cpp:10998-11134`, "a replayed record repeats exactly its state
update (scale by g, then mad k * delta)"); quality class `exact` (`paper`: the
fork's replay == snapshot chain test; our device-free chain is below).

The op `ggml_gated_delta_net_replay(ctx, q, k, v, g, beta, state, rec, ctrl, K)`
appends `GGML_OP_GATED_DELTA_NET_REPLAY` to the enum (last, so no existing op
number moves; `GGML_OP_COUNT` 103). `rec` is F32 `[R * (K - 1), 2, n_seqs]`
(two banks), `ctrl` I32 `[2, n_seqs]` = `{n_rec, bank}`: `n_rec` records of
that bank are replayed onto the input state before the new tokens. The packed
output is `[attention `S_v*H*n_tokens` | base state `S_v*S_v*H` | records
`(K-1)*R`]`; the base is the state after token `n_tokens - 1 - m`,
`m = min(n_tokens - 1, K - 1)`, and one record per later token. `n_tokens == 0`
replays the stored records only and returns the base. The CPU reference is
bit-identical to the snapshot path's per-token step: scale the state by
`exp(g)`, then mad each value row with `k * delta`; the record stores the delta
per value row, the key per key row and the gate per head.

**The node's own slots** (`code`, patch 0081's contract): the replay builder sets
`op_params[1] = GGML_GDN_MODE_REPLAY`; `src[6]` is this op's records and `src[7]`
its control, and both its CPU reference and its OpenCL consumer assert the mode
word, so a node of another GDN form (or a raw gating projection of 0051, which
rides `src[8]`/`src[9]` since 0081) can never reach this body. src[6] was taken
three times before 0081 (0051's alpha, the rows index, these records); the
replay op is a different op number, so with the gating slots moved and the mode
words in place nothing crosses.

**Device-free (`measured-here`)**: `tests/test_gdn_replay.cpp` runs a three-step
chain (5 tokens, 3 with a rollback of 2, then 1; S 16, H 4, K 5) through both
forms on the CPU backend and requires every step's attention output and the
final base state byte-identical. `tests/python/test_gdn_replay.py` reads the
patched tree for the enum's position, the packing, the prologue's operation
order, the record write, the dispatch and the chain case. Red mutations, each
run and failing before restore (`measured-here`): the prologue's mad dropped,
the record's gate written as 0 — each fails the C chain; the Python cell's own
faults are listed under 0087. The Python cell is source text; the C cell is
numeric. The cards' arm is the replay-chain case in `test-backend-ops`
(`exact=true` compares the replay and snapshot chains, max `|diff|` must be 0).

## 0087-gdn-replay-memory-opencl.patch

The memory, graph, models and the Intel OpenCL kernel for the replay. The
decision is taken in `llama_memory_recurrent`'s constructor, before the tensors
are created: `LLAMA_GDN_REPLAY` non-zero, `n_rs_seq > 0`, one sequence
(`n_seq_max == 1`, `mem_size == 1`), offload, a qwen35/qwen35moe/qwen4exp model,
F32 S, `S == 128`, `ssm_d_inner == S*H` and `n_embd_s() == S*S*H`, both banks of
records fitting one S row, and every recurrent layer on the OpenCL backend
(`code`). Any other model silently keeps the snapshot path. With replay on, S
allocates `mem_size * 2` rows (plane 0 base, plane 1 two record banks); the
snapshot planes `2..n_rs_seq` are not allocated. The conv state keeps all its
planes in both modes.

Rollback (`seq_rm`) reduces the sequence's pending record count instead of
moving the snapshot index; a rollback beyond the kept records warns ("cannot
roll back %d tokens, only %u kept") and declines. A whole-sequence removal
resets the pending records with the cells, so no stale record is replayed onto
a cleared state. Each launch takes the base
from plane 0 at its row and toggles the record bank; a state save with pending
records writes header `s_trans = 2` and the records after the S rows, and the
reader takes `s_trans == 2` only in replay mode — a snapshot reader rejects it
as incompatible rather than load a stale state (`code`).

Replay and the rows form of patch **0081 combine by exclusion**: replay's
`inp->s_rpl_ctrl` set makes the models take the replay branch and
`llama_gdn_state_rows_enabled` is not consulted, so a replay run reads its base
through its own row indices and does not use `s_rows`. With replay off and
either switch as before, the rows form and the snapshot path are unchanged and
byte-identical. With replay on and rows off, the replay is the one in use.

OpenCL: `gated_delta_net_intel.cl` gets a `GDN_RPL` build of the generation
kernel with the replay prologue (re-applying `n_rec` records in the step's own
order: `S = fma(k, delta, g*S)` row by row, the same sub-group reduction) and
the epilogue (base once, one record per later token: delta per value row, key
per key row, gate per head). The program is built only when the switch is on
(`code`: the fork's SYCL kernel has the same prologue and epilogue); the decode
price is not measured here — the card arm below measures it. If the device
refuses the build the kernel stays null and the op runs on the CPU.
`supports_op` takes F32 state, a scalar gate, F32 records and I32 control,
`S_v == 128` and the node's REPLAY mode word (patch 0081's contract), and the
dispatch asserts the same before it reads `src[6]`/`src[7]`; the dispatch passes 30 arguments with `{H_v*sg, n_seqs,
S_v/rows}` work-groups. With the switch off, the backend's programs, kernels,
arguments and bytes are as before.

**VRAM saved, from the shapes (`code`; the three served units)**: one S plane is
`S_v * S_v * H * 4` bytes a layer, and replay frees `n_rs_seq - 1` of them a
layer (conv state untouched). Agent (Qwen3.8-27B, 65 blocks, interval 4 -> 49
recurrent layers, `S_v 128`, `H = 6144/128 = 48`, 3 MiB a plane, MTP 5 ->
`n_rs_seq 5`): **588 MiB**. Flash-Next (`qwen4exp`, 48 blocks, interval 4 -> 36
recurrent, `H = 6144/128 = 48`, 3 MiB, MTP 3 -> `n_rs_seq 3`): **216 MiB**.
Coder (`qwen35moe`, 41 blocks, interval 4 -> 31 recurrent, `H = 4096/128 = 32`,
2 MiB, MTP 4 -> `n_rs_seq 4`): **186 MiB**. The interval is the hparams default
4 used where the GGUF does not carry `full_attention_interval`; a model that
carries a different interval scales linearly with its recurrent-layer count.
The audit's fork figure is 432 MiB (48 recurrent x 3 freed planes x 3 MiB at
`n_rs_seq 4`, `paper`).

**Device-free (`measured-here`)**: the C chain above plus
`tests/python/test_gdn_replay.py` — the switch's default off and the
`atoi(env) != 0` term, `n_rows_s = rpl ? mem_size * 2 : n_rows`, the rollback's
pending-record bound and warp, `s_trans = 2` and the records' save/read, the
rows exclusion in both qwen35 and qwen4exp, the replay branch's op and the
record cpy/set-rows, the `GDN_RPL` kernel's prologue `fma`, the epilogue's
record writes and the host's build gate. Red mutations, each run and failing
before restore (`measured-here` on the patched tree): the CPU prologue's mad
dropped and the record's gate written as 0 — each fails the C chain (first
differing element 640, `FAILED: 2`) and the Python cell; `n_rows_s` left at
`n_rows`; the kernel's prologue `fma` rewritten as `a*b + c*d`; the switch's
comparison flipped to `== 0`; the whole-sequence pending reset dropped. Restored,
`ctest -L unit` is 26/26. `GGML_OPENCL_DEVICES` and the recipe
builds stay the same; nothing is built for the switch when it is off.

**Card arms (owed; Claude runs them in a card window, not yet measured)**:
1. Sequential arms on the B60 (`GGML_OPENCL_DEVICES=0:0`) and the A770
   (`GGML_OPENCL_DEVICES=1:0`), one fresh process an arm:
   `LLAMA_GDN_REPLAY=1 ./bin/test-backend-ops -b GPUOpenCL -o GATED_DELTA_NET_REPLAY`
   and the same without the env for `GATED_DELTA_NET`; the replay-chain cases
   (chains A/B, 4 shapes) must pass at max `|diff| 0`, and the existing
   snapshot cases must be unchanged. Red on the card: rewrite the kernel's
   prologue `fma` as `a*b + c*d`, rebuild, and the chain cases fail; restore.
2. Served arm, agent (dense 27B), MTP 5, 138K needle:
   `printf '#!/bin/sh\nexport LLAMA_GDN_REPLAY=1\nexec %s "$@"\n' <arcint-binary> > /tmp/arcint-gdnrpl && chmod +x /tmp/arcint-gdnrpl`
   `python3 tests/acceptance/ab.py agent --old <arcint-binary> --new /tmp/arcint-gdnrpl --gguf <agent.gguf> --device GPU.0 --assets <assets> --rounds 2 --out /tmp/ab-87-agent`.
   Record decode, prefill-55k/138k, `draft accept`, the texts md5, and peak
   VRAM (fdinfo) — expected ~588 MiB lower at MTP 5 from the shapes above.
   Bar: answers and texts equal, `draft accept` unchanged, peak VRAM lower by
   the stated amount; decode recorded. If a text differs or the acceptance
   moves, the switch stays off and the finding is recorded.
3. The same with Flash-Next on the B60 (MTP 3, IQ3_XXS; expected ~216 MiB) and
   the coder on the A770 (MTP 4, `--device GPU.1`; expected ~186 MiB). Record
   each; a rollback-heavy verify (a long draft, repeated cuts) is the one to
   watch.

**0086/0087 fix (2026-11-14): qwen35moe had no replay branch** (`measured-here`, the
coder on the A770 with `LLAMA_GDN_REPLAY=1`, the separate MTP head file; then
device-free on the CPU backend). The gate in `llama_memory_recurrent` admits
qwen35, qwen35moe and qwen4exp, and `build_recurrent_attn` takes the replay
when `inp->s_rpl_ctrl` is set, but 0086/0087 added the state branch to
`src/models/qwen35.cpp` and `qwen4exp.cpp` only. `qwen35moe.cpp`'s
`build_layer_attn_linear` stayed on the snapshot gather `build_rs(inp, ...)`,
which reads `s_copy`, so `rs_s_rpl_rows` was an input object no node consumed.
The graph allocator gives an unread input no buffer, and the first decode's
`rs_set_input_rpl` (`src/llama-graph.cpp:402`) asked
`ggml_backend_buffer_is_host(s_rpl_rows->buffer)`: `GGML_ASSERT(buffer)` in
`ggml_backend_buffer_get_type` (`ggml/src/ggml-backend.cpp:346` at the pin plus
the series; 346 is get_type, get_usage's assert is at 341). The same command
without the variable served, because the branch is never taken then. The fix
is the hunk in `0000-07-gdn-and-fusions.patch` that qwen35moe's state builder
already carried for the rows form: it gains the `gdn_rpl` branch (state
`build_rs(ssm_states_all, s_rpl_rows, s_rpl_main, ...)`, rows form off under
replay), the text qwen35 and qwen4exp have. With it, `rs_s_rpl_rows` is read by
the gather (13 input tensors, 12 before), the decode passes, and the
checkpoint's partial state save (base + pending records) writes. The same CPU
run with replay off and on, a rollback of 4 after 22 tokens then 4 new tokens,
gives the same logits (`after 22` and the 64-logit sum equal, `measured-here`).
Device-free cell: `tests/python/test_gdn_replay_model_branches.py` (text, red
on the fault with the qwen35moe branch dropped, green with it). Card arms
unchanged; the coder's served replay arm is recorded under 0127 (window 270).

## 0088-opencl-gqa-int8-qk.patch

The GQA decode kernel's Q Kᵀ on the XMX units straight from the quantized
cache: Q rows quantized to int8 once a call (one RNE scale per row per 32-dim
block), the q8_0 key block's raw 32 code bytes as the DPAS B operand, the
integer result scaled by `dq[row][block] * dk[key][block]`; P V stays f16 DPAS.
Opt-in, `GGML_OPENCL_FA_GQA_I8=1`; off by default, where the source compiles to
the f16 form it was and no helper text is prepended (behaviour unchanged).

The mechanism is the fork's DPAS decode attention (`code`:
`arc-b580/ggml/src/ggml-sycl/fattn-dec.cpp:688-747`, its inline
`xe2-dpas.hpp:15-42`): A is Q as s8, B the raw keys, one key a lane, the int
accumulator scaled a block in float. The s8 x s8 operand packing is 0071's
(`code`: `mul_mm_kq_i8.cl:109`, the strided `intel_sub_group_block_read_us8`
over a row-major [row][32 codes] tile — lane l gets dims (2l, 2l+1) of each of
8 rows), which is why Q's codes are stored in dim order; the fork's s8 x s4
packs its nibbles as dims i and i+16 a byte, so its a short is
`(code[l], code[l+16])` (`code`: `fattn-dec.cpp:721-731`). The local form of that read is
`cl_intel_subgroup_local_block_io`, enabled in the kernel. The
fork's B is s8 x s4 over q4_0 bytes; ours is the public
`intel_sub_group_i8_i8_matrix_mad_k32` over q8_0 bytes, which is what arcint's
agent and coder caches hold. Its premise is arcint's measured 0018 fact
(`measured-here`: direct quantized K made every sub-group convert the same
blocks, 30.7 t/s against 52.6 staged at 32k), which the raw-byte operand
removes.

- **Q once a call** into local memory: [tile][block][lane][row] shorts and
  [tile][block][row] scales, by the work-group's own threads (not a
  per-sub-group pass). The quantizer, the q8_0 block offsets and the staged
  row stride are one text (`fa_gqa_i8_ops.inc`, the 0071 pattern: the backend
  prepends it to the kernel program and the device-free test includes it).
  RNE (`rint`), scale `amax / 127`, code `round(x / scale)`, clamp-free
  because the scale is the block's amax; a zero row keeps dq 0 and codes 0.
  The codes are written ascending dims into a row-major [row][32] tile; the
  DPAS a operand reads that tile with the strided block read, not per lane.
  Evidence: the fork's form (`code`) and ggml's q8_0 block layout (`code`,
  `ggml-common.h`).
- **B straight from the cache**: block `b` at byte `34 b`, codes at `+2`, a
  lane's 32 bytes packed into 8 ints; the block's half `d` is `dk`. Earned
  from the fork's same bytes-vs-conversion split (`code`).
- **Scale order**: DPAS is integer, so the 32-block product is exact and is
  converted to float, multiplied by the row's `dq` vector and the key's `dk`,
  and added to the f32 S. The fork accumulates the same way (`code`).
- **P V** is unchanged: P is S shortened to half, V is staged f16, the f16
  DPAS chain (`code`).
- **Which pairs**: only q8_0 K (KT 1), the 8:8 and 8:4 caches, at head sizes
  256 and 128. A q4_0 K (the refused 4:4 pair) keeps the f16 form: the public
  OpenCL extension has no s8 x s4 k32, and the fork's inline vISA is Xe2
  SIMD16 only. Not taken (`code`).
- **Sub-groups**: the kernel is built for sub-group 16 only (0016: 16 lanes are
  a key tile's 16 keys, the 8x16x32 form), and the int8 builtin is that
  sub-group's. The A770 (DG2, sub-group 8) cannot take this form from the
  code: 0071's record has the s8 DPAS kernel SG16-only and the A770's path
  f16 (`measured-here`, 0071 round 3), and the fork's own s4 vISA is
  `dpas.s4.s8.8.8 (M1,16)`. The A770 keeps 0015's split kernels; no SG8 int8
  form is built.
- **Row cap**: the dispatch's `(n_head / n_head_kv) * n_q <= 64` stands. The
  agent's 24-on-4 (G 6) takes 6 rows = 36; the coder's 16-on-2 (G 8, head 256)
  takes 6 rows = 48. Both are inside the cap (`code`).
- **Quality**: rounding-only (the Q int8 adds one rounding), as the audit
  classes it (`code`); judged at the answer-level bar, not bit equality.

**The Q staging's extent — the card's defect** (`measured-here`, the card host,
2026-11-02): `GGML_OPENCL_FA_GQA_I8=1 GGML_OPENCL_FA_GQA_MIN_ROWS=1`, head size
256: **135 of 135 FLASH_ATTN_EXT cases failed, ERR ~0.10 NMSE** against the
5e-4 bound (e.g. `FLASH_ATTN_EXT(hsk=256,hsv=256,nh=8,nr23=[4,1],kv=1024,
nb=32,mask=1,...,type_K=q8_0,type_V=q8_0)`), while the f16 arm passed the same
geometry. Cause (`code`): the staging loop enumerated `(row, block)` pairs up
to `min((n_rows + 7) / 8, 8) * QB` — the row-**tile** count times the blocks a
row — instead of the rows a call. It wrote only the first `tiles` rows of a
tile (`rt_max` pairs per block, `r = e / QB`), so every other row of the a
operand was whatever the local allocation held and its S was that garbage; a
uniform-ish garbage S averages the V rows, which is the ~0.10. n_rows is
`G * n_q` (the agent 6 a row, 36 at 6 rows; the coder 8 and 48), so at least
5 of 6, and at least 7 of 8 rows, were unstaged in every routed case — the
same ERR for all of them. 0.10 is a layout error, not int8 rounding: the
device-free S selftest quantized every row itself and modelled no staging, so
it stayed green. Fix: the extent is one text, `fa_gqa_i8_stage_rows(n_rows)` in
`fa_gqa_i8_ops.inc` (the kernel and the cell read the same helper) — every row
of every row tile, capped at the 8 tiles the a operand holds, rows past
`n_rows` zeroed with `dq` 0 like the f16 path's zero Q row.

Expected after the fix (`measured-here`, the f16 arm's own bound): the same
cases pass at **NMSE <= 5e-4**, the f16 path's order plus the one int8 Q
rounding; the S-level bound the cell states is max abs (int8 - f16) <= 1e-2 *
max|S| (0.6 % measured on the model), and the case's NMSE should land at the
f16 arm's own ~1e-4 to low-1e-4, not above 5e-4.

Device-free (`measured-here`, the recipe host):

- `tests/test_fa_gqa_i8.cpp` (built in the `unit` ctest): the q8_0 block
  offsets, the RNE quantizer against an independent `nearbyint` reference with
  the .5 corners, and the (b) S selftest **at the operands the kernel builds**:
  the staged a-operand tile (the helper's `(row, block)` pairs, a poison
  pattern wherever the staging does not reach), the strided block read that
  gives lane l dims (2l, 2l+1) of each row, the b operand as the lane's key
  (8 ints, ascending dims), and the systolic dot4 pairing per stage d (the
  shorts of lanes 2d, 2d+1 against the b dword d), scaled by
  `dq[row][block] * dk[key][block]` — against the f16 form, at the agent's and
  the coder's row counts (6, 8, 36, 48), stated bound max abs (int8 − f16) ≤
  1e-2 × max|S| (prints the measured value), plus the staging-extent cell
  (`fa_gqa_i8_stage_rows` covers every row of every tile the batch needs).
- `tests/python/test_fa_gqa_i8.py` (the `unit` ctest, `LLAMA_SRC`): the switch
  defaults off, `-DGQA_I8=1` and the helper prepend only on a KT 1 arm, the
  staged form a second build only with the switch, the 1-3 row routing under
  `GGML_OPENCL_FA_GQA_MIN_ROWS`, and the kernel's `#if GQA_I8 && KT == 1`,
  raw-code B operand and f16 P V.
- Red mutations, each committed on a throwaway branch, built and run through
  the recipe before restored (`measured-here`, 25 of 26 ctests, the two cells
  red; the last full run is the green 26/26):
  (a) the switch's default flipped on (`atoi` dropped) →
  `FAIL test_switch_is_default_off`;
  (b) the KT 1 guard dropped (`i8_arm = fi8_on`) → the same default-off cell
  at its `gkv[i][0] == 1` assertion;
  (c) the staged dispatch dropped (`n_q < 4` → `false`, 0089) →
  `FAIL test_routing_takes_the_staged_form_for_1_to_3_rows`;
  (d) `fa_gqa_i8_qs` reads `34 b + 18` instead of `+2` →
  `FAIL offset: codes start` and `FAIL s-tile: max abs error 5.176399 above
  the stated bound 1e-2 * max|S| = 0.208869`;
  (e) `FA_GQA_I8_RINT` truncated to `(int) x` →
  `FAIL quant: 1.5 -> 1, want 2 (RNE)` (the S selftest stays inside the bound
  at 0.6172 % — truncation alone is small, which is why the .5 corners exist);
  (f) `fa_gqa_i8_stage_rows` returning the tile count (the pre-fix extent, no
  `* 8`) → the S cells fail on every row the staging did not reach (the model's
  poison), the staging-extent cell fails, and `tests/python/test_fa_gqa_i8.py`'s
  extent cell fails — the card's 135 of 135 at ERR ~0.10.
  The card red is (d) too: every new FLASH_ATTN_EXT case at 1-6 rows fails
  with it. (f)'s card red is the measured defect above.

Card arms (owed; Claude's announced window), the exact commands:
- correctness and (a)/(c): the new cases are
  `test-backend-ops test -o FLASH_ATTN_EXT` at the agent's geometry
  (256, 256, 4, {6, 1}) and the coder's (256, 256, 2, {8, 1}), 1/2/3/4/6 rows,
  KV 1,013, 4,096 and 131,072, q8_0/q8_0 and q8_0/q4_0, added at the eval
  list. Run both arms of the switch on the same binary:

      cd <llama tree with 0001-0089>
      cmake -B build -DGGML_OPENCL=ON -DLLAMA_CURL=OFF
      cmake --build build -j --target test-backend-ops
      GGML_OPENCL_DEVICES=0:0 ./build/bin/test-backend-ops test -o FLASH_ATTN_EXT
      GGML_OPENCL_DEVICES=0:0 GGML_OPENCL_FA_GQA_I8=1 GGML_OPENCL_FA_GQA_MIN_ROWS=1 \
          ./build/bin/test-backend-ops test -o FLASH_ATTN_EXT
      # perf (us/run), both arms:
      GGML_OPENCL_DEVICES=0:0 ./build/bin/test-backend-ops perf -o FLASH_ATTN_EXT
      GGML_OPENCL_DEVICES=0:0 GGML_OPENCL_FA_GQA_I8=1 GGML_OPENCL_FA_GQA_MIN_ROWS=1 \
          ./build/bin/test-backend-ops perf -o FLASH_ATTN_EXT
      # red (a) on the switch-on arm: fa_gqa_i8_qs -> 34 * b + 18, rebuild, the
      # 1-6 row cases fail; restore.
      # red (f) on the switch-on arm: fa_gqa_i8_stage_rows returns `tiles`
      # (the pre-fix extent), rebuild, every routed case fails at ERR ~0.10;
      # restore, and the same cases pass at NMSE <= 5e-4 (the f16 arm's bound).
      # A770: the same two runs; the GQA kernel is SG16-only, so both arms
      # must pass identically (0015's kernels serve).
- **(f)'s card check is the defect's, run first**: the switch-on arm at head
  size 256, every case of it:

      GGML_OPENCL_DEVICES=0:0 GGML_OPENCL_FA_GQA_I8=1 GGML_OPENCL_FA_GQA_MIN_ROWS=1 \
          ./build/bin/test-backend-ops test -o FLASH_ATTN_EXT -p hsk=256
      # expect: every routed case PASS at NMSE <= 5e-4; the f16 arm
      # (GGML_OPENCL_FA_GQA_I8 unset) is the same list's bound.
      # before the fix: 135 of 135 FAIL, ERR ~0.10.
- (b) on the card: the eval list's q8_0 cases already compare each arm against
  the CPU reference at the same 16-key tile geometry; the S-level comparison
  itself is the device-free cell above (the DPAS s8 product is the exact
  integer contraction, so the only difference is Q's rounding, which that cell
  bounds). A separate on-card DPAS-vs-f16-DPAS probe was not built.
- served A/B: agent and coder, `GGML_OPENCL_FA_GQA_I8=0` against `1` with
  `GGML_OPENCL_FA_GQA_MIN_ROWS=1`, two rounds each, alternated:

      cp bin/arcint /tmp/arcint-gqai8
      printf '#!/bin/sh\nexec env GGML_OPENCL_FA_GQA_I8=0 /tmp/arcint-gqai8 "$@"\n' > /tmp/arm-off; chmod +x /tmp/arm-off
      printf '#!/bin/sh\nexec env GGML_OPENCL_FA_GQA_I8=1 GGML_OPENCL_FA_GQA_MIN_ROWS=1 /tmp/arcint-gqai8 "$@"\n' > /tmp/arm-on; chmod +x /tmp/arm-on
      python3 tests/acceptance/ab.py agent --old /tmp/arm-off --new /tmp/arm-on \
          --gguf <agent.gguf> --device GPU.<N> --assets <assets> --rounds 2 --out /tmp/ab-88-agent
      python3 tests/acceptance/ab.py coder --old /tmp/arm-off --new /tmp/arm-on \
          --gguf <coder.gguf> --device GPU.<N> --assets <assets> --rounds 2 --out /tmp/ab-88-coder
- KL (agent, q8_0:q4_0, MTP 5's 6-row ubatches so every attention call is the
  verify path; coder, q8_0:q4_0, MTP 4's 5-row ubatches), 16 x 512 against the
  arm's own Q8_0 reference capture, mean KL and argmax agreement against the
  baseline arm's, bar +0.03 nats / −1 point:

      GGML_OPENCL_DEVICES=0:0 GGML_OPENCL_FA_GQA_I8=0 ./build/bin/llama-perplexity \
          -m <agent.gguf> --no-repack -lm mmap -ctk q8_0 -ctv q4_0 -fa 1 -ub 6 -b 6 \
          -c 512 --kl-divergence-base <agent-ref.kld> --kl-divergence /tmp/kl-off.log
      GGML_OPENCL_DEVICES=0:0 GGML_OPENCL_FA_GQA_I8=1 GGML_OPENCL_FA_GQA_MIN_ROWS=1 \
          ./build/bin/llama-perplexity -m <agent.gguf> --no-repack -lm mmap -ctk q8_0 -ctv q4_0 \
          -fa 1 -ub 6 -b 6 -c 512 --kl-divergence-base <agent-ref.kld> --kl-divergence /tmp/kl-on.log
      # the same with -ub 5 for the coder
- Task battery: the acceptance task at T=0 with thinking off, the agent's and
  the coder's protocols, 10 cases, answers equal or better (the task is the
  answer-level gate; the KL arm above is the numeric one).

Status (`code`, `measured-here`): built, device-free green, off by default, no
card number yet; the defaults stay off until the operator's A/B. The gate's
per-layer bar is the audit's: the 6-row call below 0016's 4.83 ms at 131k and
the 1-row call below 0015's 1.91 ms, answers right, KL inside 0.03 nats.

Served price (windows 58 and 60, the agent on the B60, a 115,395-token prompt, one run an arm, the same answer
in every arm, `measured-here`): decode off 21.03 t/s, `GGML_OPENCL_FA_GQA_I8=1` alone 15.83 (-25 %),
`GGML_OPENCL_FA_GQA_MIN_ROWS=1` (0089) alone 19.05 (-9 %), both 13.27 (-37 %). Not a verdict on the fork's
technique: its build is SYCL s8 x s4 over q4_0 keys, ours OpenCL s8 x s8 over q8_0 keys with Q quantized a call
into local memory. test-backend-ops perf cannot price the q8_0 arms (its quantized cases rebuild the SoA every
call); its f16 single-row case shows MIN_ROWS=1's cost alone (1.91 -> 2.29 ms at 131,072 keys).


## 0089-opencl-gqa-staged-1-3-rows.patch

The 1-3 row form on the same DPAS kernel: a key tile's raw q8_0 rows are
copied into local memory once a work-group (coalesced dword loads, the row
68 dwords at head size 256, the staged stride 69 so lane = key reads spread
over banks) and every lane reads its own block of 32 codes from there instead
of the direct form's per-lane global gather. It is the fork's ST mode (`code`:
`fattn-dec.cpp:703-731`, launched for nq == 1; its measured 48K 1-token
433 -> 244 us with the one-wave split count, `paper`). The same source, a
second program with `-DGQA_ST=1`, built only when `GGML_OPENCL_FA_GQA_I8=1`
and only for the q8_0 K pairs (KT 1); the program already carries
`-cl-intel-256-GRF-per-thread`, the fork's 256-register launch.

- **Routing**: rows below `GGML_OPENCL_FA_GQA_MIN_ROWS` take the staged kernel
  when it was built (so `GGML_OPENCL_FA_GQA_I8=1 GGML_OPENCL_FA_GQA_MIN_ROWS=1`
  routes 1-3 rows); 4+ rows and any pair without a staged form take the direct
  kernel; a failed staged build falls back to the direct kernel and logs a
  WARN. With the switch off the staged array is empty and `MIN_ROWS` keeps its
  default 4, so nothing changes (`code`).
- **What is staged**: K as raw bytes, V as f16 (0019's transposed staging,
  unchanged); the Q int8 quantization of 0088 and P V's f16 DPAS are unchanged
  (`code`).
- **Quality**: rounding-only, the same one Q rounding as 0088; no new one.
- **The Q staging's extent is 0088's fix, shared**: both forms read the same
  staged a operand (`fa_gqa_i8_stage_rows`, which this patch's second build
  compiles in unchanged); the card's red hit the direct and the staged form
  alike, and the staged form's own two defects would have shown as mismatched
  K rows, not as 0088's uniform ~0.10. The 1-3 row cases are part of (f)'s
  card check above.

Device-free (`measured-here`): the offset cell asserts the staged row/stride
arithmetic (68/69 at 256, 34/35 at 128, an odd stride), and the routing cell
asserts the second build, the KT 1 restriction and the 1-3 row dispatch. Red
(c): the staged dispatch dropped (`n_q < 4` → false) fails the routing cell;
(a) the block-offset mutation also fails every new case at 1-3 rows on the
card.

Card arms: the same commands as 0088's above, with the `MIN_ROWS=1` switch-on
arm pricing the staged rows against the switch-off arm; the new 1/2/3-row
cases are the correctness arm, and the perf list's 1/3-row cases at 131,072
keys price them. Bar: the 1-row call below 0015's 1.91 ms at 131k on the B60,
answers right, KL inside 0.03 nats. Owed to Claude's window; no card number
here yet.

Status: built, device-free green, off by default; the 0088/0089 pair is the
opt-in until the card A/B.


## 0090-opencl-gdn-rows-write-layout.patch

Fixes 0081's OpenCL rows form (`LLAMA_GDN_STATE_ROWS`). The rows form READS a sequence's state from its row of
the 2D recurrent cache, but both Intel kernels (`kernel_gated_delta_net_pp` and `_tg`) also used that cache-row
base for the WRITES: the verify snapshots (`snap_buf + slot * snap_stride + state_base`) and the final state in
`dst` (`data_dst + s_off + state_base`). Those buffers have the per-sequence layout the CPU reference writes
(`ggml-cpu/ops.cpp`: `s_out = state_out_base + (seq * H + head) * S_v * S_v`, `code`), so on the card the state
landed at the wrong offset. Measured (window 57, the A770 coder, one run, `measured-here`): with the switch on an
answer gate failed, the output md5 differed and decode fell from 51 to 21 t/s. The fix splits the base:
`state_base_rd` (the cache row, rows form only) for the read, `state_base` (per sequence) for every write. The generic kernel (`gated_delta_net.cl`) had the same defect in its K == 1 final write only (its snapshots already used `state_out_base`); it now writes to `state_out_base` too. Why the card cases missed it: 0081 registered its rows cases only in `make_test_cases_perf`, which times and never compares; 0090 adds them (plus a small 2-sequence case) to the eval list.
Device-free cell: `tests/python/test_gdn_rows_write_layout.py` (text). Card arms (window 58, `measured-here`, with 0092 and 0093): test-backend-ops GATED_DELTA_NET eval 60 of 60
on the A770 and on the B60 (rows 5 of 5); the pre-0090 kernels fail 7 cases on the A770 (rows 0 of 5) and on
the B60 the first rows case faults the compute engine (CAT error: the writes leave the buffer), the fault that
preceded window 56's Flash-Next abort (`the CPU misses' publish never arrived`). Served: the coder with the
switch on, texts md5 identical to off, decode 51.5 -> 51.2 t/s (one run: no gain); Flash-Next with the exact
bundle completes, answers right, prefill-20k 536.9 and decode 32.8 t/s (window 56 off arm 534.2 / 33.1).


## 0091-replay-hybrid-input.patch

Fixes 0086's replay (`LLAMA_GDN_REPLAY`) on hybrid models, which are all of ours (qwen35 / qwen4exp: attention
plus gated delta-net). The replay controls (`s_rpl_rows`, `s_rpl_ctrl`, `s_rpl_wrow`) and the record count
(`rpl_take`, which advances `rpl_n`) were filled only by `llm_graph_input_rs::set_input`. The three hybrid inputs
(`llm_graph_input_mem_hybrid`, `_k`, `_iswa`) carry their own copy of the recurrent part and copied only `s_copy`,
so on a hybrid model the controls stayed unset and no records were ever counted. The first draft rollback was then
refused (`seq_rm: cannot roll back 4 tokens, only 0 kept`, then `llama.cpp refused the draft rollback` and HTTP 500;
window 56, the agent on the B60, `measured-here`). One helper (`rs_set_input_rpl`, and `rs_can_reuse_rpl` for graph
reuse) now serves the recurrent input and all three hybrid inputs. The op-level replay cases (test-backend-ops
`GATED_DELTA_NET_REPLAY` chains) could not see this: they build the op, not a model's graph input. Device-free cell:
`tests/python/test_gdn_replay_hybrid_input.py` (text). Served (window 58, the agent on the B60, one run, `measured-here`): capital and the 138k needle right, texts
md5 identical to off, no refused rollback; prefill 715.4 -> 690.5 t/s at 55k (-3.5 %) and 501.1 -> 494.1 at
138k (-1.4 %). The agent A/B carries no decode metric, so there is no case for a default yet.
Window 59 (the agent on the B60 at a 115,395-token prompt, one run an arm, `measured-here`): resident VRAM
22,814 against 23,390 MiB after load and 23,725 against 24,301 after the request (-576 MiB), decode 20.96 against
21.03 t/s, prefill 214.9 against 211.5 s; the answer opens the same, its bytes differ (KL not measured).
Window 61 (the agent on the B60, `measured-here`): the acceptance task greedy without thinking 10/10 with
replay and 10/10 without (a different program); at `--n-ctx 262144` (the model's trained length) with replay the
unit loads at 23,256 MiB resident (245,760 without replay: 23,390), a needle at 40 % of a 254,970-token prompt is
right (731.5 s, 348.6 t/s), the peak is 24,199 of 24,480 MiB and GTT stays flat (64 MiB). A verify costs the same
124.4 ms at 245,760 off, 245,760 with replay and 262,144 with replay at a 115k prompt; the decode rates differ by
the answers' draft acceptance only. KL by perplexity cannot see replay (it never rolls back).


## 0092-tbo-replay-chain-pool.patch

test-backend-ops only. 0086's `test_gdn_replay_chain` builds a whole chain of launches in one graph and ran out
of the case context's 128-tensor pool (`ggml_new_object: not enough space in the context's memory pool (needed
130080, available 129712)`, `GGML_ASSERT(obj_new)`), which aborted the eval run at the first chain case. So no
chain case and none of the cases after it (0090's rows cases among them) had ever completed on a card (window 58,
the A770, `measured-here`). `test_case` gets a `max_tensors()` hook (default 128, the upstream budget) used by
its four context sites; the chain case returns 4096.


## 0093-tbo-gdn-rows-ids-set.patch

test-backend-ops only. 0081's rows case wrote its row ids through `(int32_t *) t->data`, which is host memory only
on the CPU backend; on a GPU backend it is a device-buffer address, so the first rows case on a card killed the
test (SIGSEGV, exit 139; the B60 logged a copy-engine CAT error as the context died; window 58, `measured-here`).
The ids now go through `ggml_backend_tensor_set`.


## 0094-defaults-kv-only-sync-trim.patch

Turns on by default the two switches that measured a gain with nothing degrading: 0080's MTP KV-only pass
(`LLAMA_MTP_KV_ONLY`, now on unless `=0`) and 0083's bit 2, the KV-cache rotation matrices kept on the device
(`LLAMA_SYNC_TRIM` now 2 unless set; `0` restores upstream). Window 58 (`measured-here`, 2 rounds an arm against
off): the coder (A770) decode +1.8 / +2.0 / +1.8 %, prefill-41k +6.6 %, texts md5 identical; the agent and
Flash-Next equal (Flash-Next's qwen4exp graph has no KV-only pass). Every other switch of the series stays off:
0081 rows, 0082, 0083 bits 1 and 4 and 0084 gave nothing; 0088/0089 cost decode; replay (0086/0087/0091) is a
per-unit choice where its VRAM buys context (the agent's sample unit).

## 0095-opencl-fwht-hadamard.patch

The KV cache rotations (`llama_mul_mat_hadamard`, the `attn_rot_dev_*` matrices that 0094 keeps on the device) are
MUL_MAT nodes with `GGML_HINT_SRC0_IS_HADAMARD`: a product with the orthonormal Sylvester Hadamard matrix, which the
CPU (`ggml_compute_forward_fwht`) and CUDA (`ggml-cuda/fwht.cu`) backends compute as a fast Walsh-Hadamard transform
of each src1 row. The OpenCL backend ignored the hint and ran them as dense F32 products: the F32 skinny split-K path
(GEMM + slice sum), 24,615 pairs and 750 ms of a 22.3 s B60 decode window, ~2.35 ms a verify step
(`measured-here`, Qwen3.8-27B, q8_0:q4_0 KV, MTP 5; identified by a src0-name trace of every F32 MUL_MAT). The patch
adds `kernel_fwht_f32_{64,128,256}` (`fwht_f32.cl`: one row per 16-wide subgroup, the stages below 16 by subgroup
shuffles, the rest in registers, as fwht.cu does per warp) and takes the product ahead of every other MUL_MAT path
under the conditions of the reference (a MUL_MAT; src0 / src1 / dst F32, src0 2-D, src1 and dst contiguous and the
same shape). Intel only, and only where `cl_intel_required_subgroup_size` lets the program require the 16-wide
subgroup its indexing assumes (the build is optional: without it every hinted product keeps the dense path). Widths
64-256 cover every model here (`n_rot_k` doubles from 64 with the head dim, `n_rot_v` is pinned at 64); 512 and up
keep the dense product (the 512 kernel spilled at SIMD16 on the A770, offline ocloc). F16 src1 (which the CUDA
reference also takes) declines: the graph's rotations are F32. `GGML_OPENCL_FWHT=0` keeps the dense product.
test-backend-ops `MUL_MAT_HADAMARD` (the upstream cases; 5 of the 14 reach the kernel, the F16-src1, 512+ and batched-src0
cases decline to the dense path): 14/14 on the B60 and the A770 with and without the switch; a mutant (one butterfly
sign flipped) fails exactly the 5 with the switch on and passes with it off. Reviewed by Qwen3.8-Flash-Next (the op check,
the 2-D src0 and the subgroup-size gate came from its review).

Served (`measured-here`, the same binary with `GGML_OPENCL_FWHT=0` against on, each card in one window): the dense
Qwen3.8-27B (B60, q8_0:q4_0 KV, 262,144 tokens, replay, MTP 5) a verify step 59.5 -> 57.5 ms (-3.4 %), decode 47.7 ->
50.0 t/s over 3 prompts x 2 rounds at temp 0 (draft acceptance 46.5 -> 47.2 %), prefill 714.8 -> 726.0 t/s at 55k and
502.5 -> 507.4 at 138k, texts identical in the A/B tool's cells; the coder (A770, q8_0:q4_0 KV, MTP 4) a verify step
35.0 -> 33.1 ms (-5.5 %), prefill 775.9 -> 781.1 at 41k, every answer right, draft acceptance lower on all 3 prompts
(37.9 -> 36.2, 39.2 -> 33.7, 30.0 -> 28.8 %) so decode t/s stayed mixed (+1.1, -4.5, +3.2 %) with the texts differing;
Flash-Next (B60, one round) prefill 534.5 -> 538.5, decode 33.1 -> 31.8 t/s at acceptance 69.3 -> 62.7 %, inconclusive; three
rounds of the release A/B (window 75) gave 32.63 -> 32.67 t/s, acceptance 66-70 % in both arms, PASS.

Against draft acceptance and at the answer (`measured-here`, window 76, the main-merge RC 31e021e4 with `GGML_OPENCL_FWHT=0` against
on, the deployed unit flags, 12 greedy answers of up to 300 tokens per run, two rounds an arm). The coder (A770) 72.26 and 71.83 t/s off, 74.47
and 75.33 on (+4.0 %), acceptance 72.0 and 72.6 % off, 72.1 and 72.6 % on, a verify step 27.4 -> 26.3 ms. The agent (B60) 46.23 and 46.38 off,
47.59 and 47.51 on (+2.7 %), acceptance 47.0 and 47.2 % off, 46.9 and 46.5 % on, a verify step 59.1 -> 57.2 ms. KL of the agent against
window 61's CPU reference (16 x 512, q8_0:q4_0 KV): 0.004946 -> 0.004095 nats at `-ub 512`, 0.006860 -> 0.006872 at `-ub 1`, top-1 96.96 ->
96.94 % and 96.74 -> 96.89 %. Inside the answer-level bar on all of them.

The release A/B against 1.5.0 (window 75) went red on the coder's `decode-b`: 53.75 -> 51.50 t/s, the only metric of the leg below the old arm,
with decode-a +2.3 %, decode-c +2.6 %, prefill-41k +0.8 %. That answer's text moved with the rounding and its draft acceptance fell 39.2 -> 33.7 %
while its verify step went 34.7 -> 32.5 ms. Three answers each carried one md5 per arm, so `ab.py` gated every run instead of a majority answer.
The wider set above shows no acceptance cost on the coder, so this is one answer's text, not the kernel's speed. The operator accepted the
red cell on 2026-11-05.

## 0096-opencl-intel-dense-q4_0.patch

Dense Q4_0 weights on the Intel fast paths: the file of the community GPU tables (Llama 2 7B Q4_0) took upstream's float
kernels, the tiled GEMM ~95 % of a 512-token prefill and the 8x matvec 2/3 to 4/5 of a decode step (`measured-here`,
unitrace). Q4_0 has IQ4_NL's flat planes (16 code bytes and one f16 scale per 32 weights, low nibbles weights 0..15, high
nibbles 16..31) with the value n - 8, so the path is IQ4_NL's: up to 4 columns `kernel_mul_mv_q4_0_q8_1` (the q8_1
integer-dot matvec), from 16 columns `kernel_mul_mm_q4_0_f16` (the fp16 XMX GEMM, n - 8 taken in integers); 5 to 15 columns
stay on upstream's path, unmeasured. A KV cache's Q4_0 is AoS and declines. `GGML_OPENCL_Q40_XMX=0` keeps the float kernels.

Measured (`measured-here`, window 73/74, both cards, A770 GT pinned at 2000 MHz; arcint at qfndev cb8151bd, served through
`tools/llama_scoreboard.py`, n_ctx 4096, f16 KV, 5 repetitions):

| Chip | pp512 before -> after | tg128 before -> after |
|---|---|---|
| Arc A770 16 GB | 362.73 -> 1956.15 ± 2.93 (5.4x) | 39.48 -> 63.56 ± 0.06 (+61 %) |
| Arc Pro B60 24 GB | 425.57 -> 2332.94 ± 3.28 (5.5x) | 76.24 -> 75.77 ± 0.02 (-0.6 %) |

The answer-level bar on both cards, each arm against a CPU reference over 16 x 512 tokens: mean KL 0.000431-0.000432 nats
with the switch on and off, at `-b 512` (the GEMM) and `-b 4` (the matvec); argmax agreement 99.07 % on, 99.09 % off (A770,
`-b 512`, 0.02 points); all 8 fact questions right in all four served arms, with the same text on and off and across the cards.
`test-backend-ops -o MUL_MAT -p type_a=q4_0`: 161/161 on both cards, switch on and off. Mutants with the switch on: `(char4)(7)`
fails 43 (A770) and 44 (B60) cases, all at 1-4 columns and none from 16; `(char16)(7)` fails 32 on both, 28 of them from 16
columns and none at 1-4. With the switch off the matvec mutant passes 161/161 on both cards.

On the B60 the matvec takes 10.4 ms a token over ~3.7 GB of weights (code and scale bytes), about 360 GB/s of the card's specified ~456 GB/s (`measured-here` for the kernel time, under unitrace over 400
tokens; the bytes are computed from the shapes), and the old 8x kernel took 10.7 ms, so 0096 moved its decode little. CORRECTED: this section first called that decode bound by memory; 0097 then moved it 6.5 % by attention alone.
The A770's gain is the same kernel at 10.7 ms against the old kernel's 20.9 ms. Models without Q4_0 weights do not enter these paths (`code`).

## 0097-opencl-fa-intel-split-by-depth.patch

The per-head Intel decode attention kernel (0007, quantized 0015) took 128 keys a split at any depth. One sub-group walks a split's keys in order, and the switch had never been searched. A grid of 16, 32, 64, 128, 256 and 512 keys a
split at 0, 256, 1,024, 4,096 and 16,384 keys of depth, llama-bench decode, flash attention on (`measured-here`, both cards, Llama 2 at f16 and q8_0:q4_0, the coder on the A770 and the dense 27B on the B60, both q8_0:q4_0):
- short KVs want short splits: at depth 0, 16 keys beat 128 by 11 and 7 % (Llama 2 f16, A770 and B60), 21 and 14 % (q8_0:q4_0), 14 % on the coder and 2 % on the dense 27B;
- deep f16 K/V wants long ones: 512 keys beat 128 by 8 and 10 % at 16,384;
- quantized K/V stays near 128 from 4,096 on, where 256 and 512 lose up to 25 %. The coder's best at 4,096 is 32 keys (+6 %), which this rule leaves; the Llama 2 files want 64-128 there, so it depends on the model's geometry.

Unset, the switch now takes n_kv/16 in 16-512 for f16 K/V and n_kv/32 in 16-128 for quantized K/V; a value fixes it, and 128 is the old default. The 256-split cap is unchanged. The GQA kernel (from 4 rows on Xe2) keeps its own 512.

Gate (`measured-here`, against the deployed 31e021e4, one window). Served Llama 2 7B: A770 f16 63.57 -> 70.22, q8_0:q4_0 54.89 -> 66.06, B60 f16 75.77 -> 80.70, q8_0:q4_0 68.63 -> 78.87 t/s, prefill level. The coder on its deployed flags, 12 answers a run, two rounds: 74.65 -> 82.78 t/s,
a verify step 26.2 -> 24.1 ms, draft acceptance 72.95 -> 71.9 %; the A/B's decode 53.70 / 51.50 / 47.40 -> 58.70 / 59.65 / 55.90; the task 10/10 x3. The agent equal, 47.71 -> 47.76 t/s, verify 57.2 ms, the needle right at 254,970 tokens, peak 24,190 MiB. Writer and vision equal, same md5. Flash-Next red on the
variance rule only, accepted by the operator: 32.97 -> 33.33 t/s, the spread 10.8 against 5.5 %, verify 67-72 ms in both arms. KL against the CPU reference, Llama 2, one token a ubatch: 0.000431 in all four arms, top-1 99.09 -> 99.07 %. `test-backend-ops -o FLASH_ATTN_EXT`: B60 2880/2880; A770 2877/2880. The A770's three (head size 64, 8 heads,
4 or 8 KV rows, ERR about 0.84) fail the same on the old binary and with 128 forced: an older defect in another path, found and fixed in 0098. A mutant that floors the split count fails 183 (A770) and 166 (B60) cases. Reviewed read-only (no code findings).

## 0098-opencl-fa-split-rules-and-q1-split-sg.patch

Three changes on decode attention, all from one window of measurements.

**The split by card.** Quantized K/V left 0097's rule, n_kv/32 in 16-128. The grid (`measured-here`, llama-bench, decode, 0 to 32,768 keys, q8_0:q4_0) says the cards differ: the B60 wants n_kv/32 up to 512 (Llama 2 at 16,384 keys 36.52 -> 41.26 t/s, at 8,192 50.52 -> 53.92; the 27B within 1 %), and the A770 wants shorter splits (the coder at 4,096 keys 49.64 -> 52.20, at 8,192 48.19 -> 49.88), so `dg2` takes n_kv/64 in 16-64, and 128 from 16,384 keys. f16 K/V keeps n_kv/16 in 16-512.

**The GQA kernel's split at short context.** 64 keys below 2,048 keys, 512 from there. The verify step on the 27B behind ~300 tokens 57.2 -> 56.3 ms in both rounds; equal at 3,300 keys; 512 stays best at 12,400 (64.0 against 70.2 at 64). Decode over 12 answers did not show it: 48.56 -> 48.04 t/s, inside the run-to-run spread, the draft acceptance 47.4 -> 46.1 %. A small lever, adopted on the verify step's own time.

**The plain K-split kernel's work-group on Intel.** `flash_attn_f32_f16_q1_split` and its q8_0 and q4_0 forms step `Q1_WG_SIZE` = FA_SG keys a work-item and size local arrays by it, and ran in a work-group of 64 at FA_SG 32 (`code`): keys past the first 32 were visited twice, local indices ran past the arrays. On Intel a work-group is now 32 items, and the f16 kernel, which reduces over its sub-group, pins a sub-group of FA_SG. Heads the Intel split kernels do not build reach these kernels from 2,048 keys: f16 K/V at 40-112, quantized at 64 and 96. No served model has such a head (`code`, the GGUF metadata; the TTS model was not read).

Red, `measured-here`: of the new cases (head 64 and 96 x f16, q8_0, q4_0, 2,048 keys, one row), on 0097's kernels the A770 fails f16 at both heads (ERR 0.80, 0.83), and the older 3 cases with 4 or 8 KV rows fail with them (ERR about 0.84). The output was a near-uniform average of V. Passing on the old kernels: f16 on the B60, and the quantized forms on both cards; the wrong answer is measured only for f16 on the A770, the overrun in the others is read from the code. Green: 2886/2886 FLASH_ATTN_EXT on both cards.

Gate (`measured-here`, against the 1.5.1 package, one window with the host quiet where it counts). KL against the CPU reference, Llama 2, q8_0:q4_0, one token a ubatch: 0.001890 on both cards. Answers: the task 10/10 x3, the capital and the needle right on Flash-Next. Served, deployed flags, two rounds, means:

| Unit, keys | 1.5.1 t/s | 0098 t/s | Verify ms |
|---|---|---|---|
| coder, ~300 | 82.71 | 82.72 | 24.06 -> 24.09 |
| coder, ~3,300 | 75.91 | 78.13 | 26.93 -> 25.80 |
| coder, ~7,000 | 71.28 | 74.52 | 28.26 -> 27.60 |
| agent, ~300 | 48.56 | 48.04 | 57.22 -> 56.27 |
| agent, ~15,800 | 41.29 | 41.57 | 67.67 -> 67.61 |
| agent, ~29,700 | 38.30 | 37.96 | 73.35 -> 73.38 |

The A/Bs: coder and agent PASS, decode and prefill equal, on prompts deep enough that the rules do not differ. Flash-Next, 4 rounds on a quiet host (the first 2 rounds ran during a copy and an unpack on the same machine and are discarded): prefill 538.77 -> 538.25, decode 33.25 -> 33.45, the one text both arms produced 34.8 -> 35.2 t/s; the A/B reports FAIL on its variance rule only, 9.57 % against an allowed 9.02 %, sampled answers differing in every run. Accepted by the operator (2026-11-05). Reviewed read-only: no code defect; the review's corrections are in the comment here. The review also read the double count as a wrong answer in the quantized kernels, which the red run does not show.

`GGML_OPENCL_FA_INTEL_KV_PER_SPLIT` and `GGML_OPENCL_FA_GQA_KV_PER_SPLIT` fix the sizes as before.

## 0099-opencl-q4_0-q8_0-int8-gemm.patch

Dense Q4_0 and Q8_0 weights on Intel without local memory. Priced from IPEX-LLM's lead on Llama 2 (`docs/campaigns/research-concept-mining/ipex-llm/`): its oneDNN f16 GEMM took about 95 ms of a 192 ms 512-token prefill on the B60, its dequantization about 40 ms, and the tile GEMM of 0096 about 160 ms for the same products (`measured-here`, the wheel's oneDNN verbose log and unitrace).

- **Xe2 (`mul_mm_kq_2d.cl`):** 0096 put Q4_0 on the local-memory tile while the K-quants had the 2D-block GEMM. `kernel_mul_mm_kq2d_q4_0` is that shape on Q4_0's planes: `(q - 8) d` in halves with the rounding of a dequantization to f16; `kernel_mul_mm_kq2d_q4_0_i8` is one int8 DPAS per 32 weights.
- **Alchemist (`mul_mm_q40_i8_sg8.cl`):** no 2D loads. The activations are quantized once a graph into `[k / 32][token][32 bytes]`, so an `a` operand of 8 tokens is one 256-byte block read; a lane is a weight row. Q4_0 and Q8_0. Before this, Q8_0 on sub-group 8 had no fast path at all: its XMX tile sits inside `#if SG != 8`, so the coder's shared experts ran ggml's float kernel.

`GGML_OPENCL_Q40_2D` = `0` the tile, `1` fp16 (Xe2 only), `2` int8 (default). `GGML_OPENCL_Q8_I8=0` keeps the kernels below it. `GGML_OPENCL_Q40_SG8_SHAPE` = `TMM,WG` (default `32,8`). K must be a multiple of 256; otherwise the old paths take the product.

`measured-here`, llama-bench, Llama 2 7B Q4_0, flash attention on, 5 repetitions, two rounds (A770 GT at 2000 MHz):

| Card, arm | pp512 | tg128 | mean KL, whole chunks | mean KL, one token | same top |
|---|---|---|---|---|---|
| B60, 0096's tile | 2482 | 81.7 | 0.00043 | 0.00043 | 99.09 % |
| B60, 2D fp16 | 3575 | 81.7 | 0.00043 | 0.00043 | 99.09 % |
| B60, 2D int8 | 4389 | 81.7 | 0.00105 | 0.00043 | 98.82 % |
| A770, 0096's tile | 2073 | 70.4 | 0.00043 | 0.00043 | 99.07 % |
| A770, int8 (shape 32,16) | 2794 | 70.6 | 0.00104 | 0.00043 | 98.65 % |
| A770, int8 (shape 32,8) | 2813 | | | | |

The same-host IPEX-LLM wheel for comparison: B60 2826 and A770 2604 with a whole-chunk KL of 0.016. A one-token ubatch has no 16 columns, so it takes the matvec: its KL does not see these kernels. The B60's shape did not move past the spread (work-group 4 to 32 sub-groups, `KSYNC_I8` 8 to 9999).

The coder's own file on the A770, llama-bench, 4,096 tokens, ubatch 1,024, f16 KV, 3 repetitions, two rounds: Q8_0's kernel off the profile's top 12 (810 ms of 6.7 s before), prefill 1,695 -> 1,866 t/s (+10.1 %), decode 58.4 both. KL against the file's CPU reference, 8 chunks: ubatch 512 0.00554 -> 0.00572, same top 97.01 -> 96.91 %; ubatch 32 0.00558 -> 0.00588, same top 96.72 -> 96.77 %.

Suites: `test-backend-ops -o MUL_MAT` q4_0 161/161 on both cards, q8_0 151/151 on the A770. Red, each seen failing with the mutant compiled in: `-DQ40_RED` through `GGML_OPENCL_KQ_2D_OPTS` or `GGML_OPENCL_Q40_SG8_OPTS` 23 cases of 161 on either card and arm (25 error lines), `-DQ8_RED` 26 of 151.

`GGML_OPENCL_Q8_XMX=0` no longer reaches ggml's Q8_0 kernels on Alchemist by itself; `GGML_OPENCL_Q8_I8=0` comes first there. `kq_i8` carries no layout tag: the Xe2 and sub-group 8 quantizers are built for different cards and never coexist.

## 0100-opencl-downgrade-lines-and-counts.patch

A path other than the one a weight's type promises now says so. The coder's Q8_0 ran ggml's float kernel at prefill for months (0099), seen only in a profile. One line, `DOWNGRADE src=opencl <key>`, at `GGML_LOG_WARN` the first time a key occurs in the process, and a count per key, read with `ggml_backend_opencl_downgrades`; arcint serves them on `/props` as `downgrades`.

| Where | Key |
|---|---|
| an optional program not built, or not fitting a work-group (17 sites) | `path=fallback reason=not-built\|no-fit what=<kernels>` |
| `supports_op` sending a K-quant or IQ product to the CPU | `op=MUL_MAT\|MUL_MAT_ID type= k= path=cpu reason=shape\|kernels-missing` (its count is the scheduler's probes, not executions) |
| ggml's generic tiled GEMM (Intel) | `op=MUL_MAT type= k= cols=1-4\|5-15\|16+ path=generic reason=no-fast-path` |
| the per-column matvec at 16 or more columns (Intel): below 32 columns the generic switch breaks out to it | `op=MUL_MAT type= k= cols=16+ path=matvec reason=no-fast-path` |

A matvec below 16 columns is a decode step's normal path and says nothing; neither does the MUL_MAT_ID matvec, whose columns are experts x tokens. The list is not complete: expert GEMM selection and flash attention's shape limits are not covered yet.

Served, deployed flags, 12 short answers and prompts of 396, 621, 1,821, 6,021 and 19,521 tokens (`measured-here`): the coder none, the agent none, Flash-Next 8 keys, BF16 at 5-15 columns (K 10240, 2560, 320) and F32 (K 128, 2560). Their cost is not priced.

Red, each seen reported, on the coder: `GGML_OPENCL_Q8_I8=0` 4 keys, Q8_0 at K 2048 and 512 on the generic GEMM and on the matvec; a program that does not compile, `not-built what=q40_i8_sg8`. The first version of this patch missed the matvec, and short prompts never reached the generic path, so that red run reported nothing: a card check of this kind needs long prompts.

## 0101-opencl-downgrade-expert-and-fa.patch

0100's list, widened to what it left out: an expert product on ggml's generic `MUL_MAT_ID` (`op=MUL_MAT_ID type= k= path=generic reason=no-fast-path`), and a prompt's flash attention (more than 8 rows) off the XMX kernel (`op=FLASH_ATTN head= kv= n_q=9+ path=f32_f16 reason=no-dpas-kernel|shape`). Sub-group 8 has no GQA decode kernel by design, and an expert product below 16 tokens takes the matvec by design: neither reports.

Served on all three units with prompts up to 19,521 tokens, nothing new (`measured-here`). Red, each seen reported: `GGML_OPENCL_FA_DPAS=0` on Llama 2 through arcint, `op=FLASH_ATTN head=128 kv=f16 n_q=9+ path=f32_f16 reason=no-dpas-kernel`, prefill 2814 -> 1461 t/s; `test-backend-ops -o MUL_MAT_ID -p type_a=q8_0`, the generic expert key at K 256 and 96. `llama-bench` installs a null log callback (`llama-bench.cpp:2286`), so it never shows these lines: check through arcint's `/props` or a tool that keeps the log.

## 0102-opencl-q4_K-int8-sg8.patch

Dense Q4_K on Alchemist takes 0099's shape: int8 DPAS, the activations packed once a graph, no local memory. From 16 columns, K a multiple of 256; the quantizer also writes the integer sums per 32 for Q4_K's constant term, a k16 fp16 DPAS a block. `GGML_OPENCL_KQ_SG8_I8=0` keeps the tile.

`measured-here`, the coder's file on the A770, 4,096 tokens, ubatch 1,024, f16 KV, 3 repetitions, two rounds: prefill 1,865 -> 1,919 t/s (+2.9 %), decode 58.6 both, the kernel 838 -> 632 ms over the profile's run. KL against the file's CPU reference, 8 chunks: ubatch 512 0.00572 -> 0.00585, same top 96.91 -> 97.40 %; ubatch 32 0.00588 -> 0.00600, same top 96.77 -> 96.62 %. `MUL_MAT` q4_K 129/129, with `GGML_OPENCL_KQ_SG8_I8=0` too; red (`-DQK_RED`) 117/129. Served A/B and the acceptance task are owed.

## 0103-opencl-id-int8-sg8.patch

The prompt's expert products on Alchemist take int8 DPAS: IQ4_XS, IQ4_NL and Q4_K, K a multiple of 256, rows a multiple of 8. `kernels/mul_mm_id_i8_sg8.cl`, its own program, so a compile failure costs this arm only. `kernel_id_pack_i8_sg8` copies the quantized columns once per (activation, route) into the route's tile order, `[tile][k/32][TMM][32]`, so gate and up share it; no local memory and no barrier. The IQ4 weight codes are `kvalues_iq4nl`, -127..113, never rounded: `id_kv4` is a mux over the code's bit planes, with no memory read; `-DID_KV_TABLE` is Strata's byte-pair table (they tie once 0104 is in). The expert cache's slots and the bank decline to the tile, and a decline at dispatch is reported (0104). `GGML_OPENCL_KQ_ID_SG8_I8=0` keeps the tile and does not build the program.

Alone it is a loss, recorded because the reason is what 0104 acts on (`measured-here`, the coder's file on the A770, 4,096 tokens, ubatch 1,024, f16 KV): prefill 1,918 -> 1,636 t/s, 1,680 with the table. Tile sizes do not rescue it: TMM 16, 32, 64, 128 give 1,490, 1,636, 1,578, 1,082. Taking the kernel apart with timing-only switches (IQ4_XS, 2,797 ms over the profile, the fp16 tile 2,042): activation reads 50 %, weight reads 3 %, the decode 20 %, stores 1 %, DPAS and the rest 20 %.

Differences from Strata's form (`code`, `sycl/src/prefill/esimd_moe.dp.cpp`, `esimd_moe.hpp`): Strata runs int8 only on Battlemage by default and keeps Alchemist on FP16, so an Alchemist int8 path has no reference result; its thread owns 16 rows (sub-group 16), here 8; it picks 2, 4 or 8 row tiles per decode from the experts' mean rows, here fixed; and it scales the activations per 32 of K, here per 256 (`kernel_quant_i8_sg8`, shared with 0099), which is 8x coarser and the likely part of the KL below.

## 0104-opencl-id-int8-row-sets.patch

A sub-group owns `NR` sets of 8 weight rows and reads each activation tile once, into registers, for all of them, which divides the reads that were half of 0103's kernel. Alchemist's DPAS takes 8 rows, so Strata's 16-row thread is here two DPAS on one A operand. A float8 at sub-group 8 is 8 GRF, so the register budget binds: `acc` and `af` are 16 GRF a set and pair group, and NR 3 needs about 296 of 256; IGC spills and retries with rolled loops. `ID_NOAF`, the default, drops `af`, folding the activation's scale into each 32-block's fma: 8 GRF, one more multiply a DPAS, the same value to one more f32 rounding. Q4_K's constant term moves out of the K loop. `GGML_OPENCL_KQ_ID_NR` picks NR, 2 by default; `-DID_KEEP_AF` keeps `af`.

`measured-here`, the same file and card, 4,096 tokens, ubatch 1,024, f16 KV, 3 repetitions, same window: off 1,916 / 1,918; NR 1 2,043; NR 2 with `af` 1,938; **NR 2 without `af` 2,151 (+12.2 %)**, table 2,149; NR 3 without `af` 2,042, table 2,092; NR 4 without `af` 1,430 (spilled). Decode 58.6 in every arm. On the exact tree, two rounds: 1,915 -> 2,150 t/s. Served on the coder's deployed flags, q8_0:q4_0 KV, against 1.5.4: prefill at 41k 843.3 -> 958.5 t/s (+13.7 %); at a 90k prompt 166.8 -> 159.2 s.

Answers (`measured-here`): the acceptance task 10/10 x3, deployed 10/10; KL against the file's CPU reference, 8 chunks, off -> on: ubatch 512 .00585 -> .00803, same top 97.40 -> 96.67 %; ubatch 32 .00600 -> .00909, 96.62 -> 96.81 %. Inside the bar.

The A/B's decode cells went red: answer b 58.8 -> 54.5 t/s, c 52.25 -> 51.4, a 56.35 -> 56.55, with different texts. A verify costs the same in both arms (30.6-31.1 ms), and the new texts took 389 verifies against 376. Whether quantized prefill costs draft acceptance was measured apart: 24 generated prompts, each with its own 2k-token prefix so that its prefill takes this path, deployed flags, two rounds: tokens a verify 2.45, 2.40 -> 2.43, 2.47, accepted 75, 76 % -> 75, 76 %. No loss; the red is text variation over three answers.

Memory: 43 MiB more, peak 16,208 -> 16,251 MiB on these prompts and 16,220 -> 16,263 at the 90k prompt, of the A770's 16,288. GTT stayed at 32 MiB, so nothing is evicted, but the margin is 25 MiB.

Suites, `-o MUL_MAT_ID` on the A770: iq4_xs 22/22, q4_K 105/105, iq4_nl 25/25, off the same; red (`-DID_RED`, a negated weight scale) 3/22, 40/105, 7/25. The first form of this patch made Q4_K spill at every NR, seen in the IGC dump, not in the suite. On the A770 only: both programs build only at sub-group 8, so the B60 does not change.

## 0105-mtp-head-without-output.patch

`qwen35moe.cpp` only. A head-only MTP GGUF (the trunk's probe tensor `blk.0.attn_norm.weight` absent) may carry no `output.weight`. Before, the loader made a `TENSOR_DUPLICATED` copy of `token_embd` as `output` for it, 398 MiB on the coder's card, which the draft never reads when `--llama-mtp-vocab` is set: the MTP context then returns the nextn rows and arcint scores them against rows it reads from a file (`DraftHead`). Now an MTP-only file without `output.weight` gets no fallback, and `graph_mtp` leaves `res->t_logits` null where it asserted on a missing head. A file with `output.weight` or `nextn.shared_head_head`, a trunk-only file and a full model load and build as before. A context that asks for logits from such a head has none to read: arcint refuses that combination at start (a head file without `output.weight` needs `--llama-mtp-vocab`).

`code`, not yet built or run here: the head file's saving is `output` (398 MiB on the coder), the card proof is owed (`docs/campaigns/mtp-head-fit.md`).

## 0106-opencl-fa-head-512.patch

Flash attention at head size 512 on Intel, for Gemma 4's full-attention layers (E4B: 42 layers, 7 of them full attention at head size 512, 35 sliding-window at 256, 8 query heads on 2 KV heads, 18 layers reading earlier layers' K/V).

What it was (`code`, `measured-here` where a count is given). ggml's `supports_op` declines `FLASH_ATTN_EXT` at head size 512 on every Intel device (`if (dk == 512) { if (gpu_family == INTEL) return false; ...`, upstream at the pin and at `bed0a85`). The two cache types then fail differently:

- A quantized V cache forces flash attention on (`llama_init_from_model`), so the scheduler gives the 7 head-512 attentions of each graph to the CPU, with their Q, K, V and mask copied there. No DOWNGRADE line said so: the decline happened in `supports_op`. B60, arcint 1.6.0, q8_0:q4_0, a 6,544-token prompt at 251 t/s, decode about 69 t/s.
- An f16 cache leaves flash attention on auto, and the probe (`resolve_fused_ops`) found layer 5's attention on the CPU and turned flash attention off for the whole model (its log line). Every layer's attention is then `K Q` (`MUL_MAT`, src0 the f16 K cache as a permuted 3-D view, 2 KV heads broadcast to 8, k = 256 or 512) and `KQ V` (src0 the transposed f16 V cache, a strided view, k = n_kv). The F16 XMX path takes 2-D contiguous weights only (`src0->ne[2] == 1`, contiguous), so both land on ggml's generic tiled GEMM after a copy of each source to a contiguous buffer: the window's keys `type=f16 k=256` 763 (35 sliding-window K Q per prompt graph and K Q V at n_kv 256 for the short requests), `k=512` 119 (7 a graph x 17 graphs) and one `k=<n_kv>` key a ubatch, 42 each (n_kv 768 to 6,656). Prefill 1,482 t/s, decode about 79.
- Not attention, and not changed here: `type=f32 k=256` (697 = 41 x 17 prompt graphs) is `blk.N.proj.weight`, the per-layer-input projection, stored as F32 [256 x 2,560] in the GGUF, on the generic F32 GEMM; `type=f32 k=512` (187 = 11 x 17) is the K-cache Hadamard rotation (`llama_mul_mat_hadamard`) at `n_rot_k` 512 for the full-attention layers (7 Q and 4 K rotations a graph), which 0095's FWHT covers up to 256 only.

What 0106 does. The prompt kernels get a V column split: `DVH`, the output columns a work-group computes, DV unless set. At head size 512 they are built with `DVH` 256, two work-groups a head (`n_head x 2` in dimension 1), each computing S and the softmax over all 512 dims itself and P V for its half of V. A sub-group then holds 8 rows of 256 f32 output columns in 128 of its 256 GRF, as at head size 256; all 512 columns would be all 256. The halves compute the same S, m and l in the same order. Xe2: `flash_attn_dpas2.cl` (plain, `FA_CARRY` with its combine, the Q conversion), its S loop unrolled four chunk pairs at a time (`FA2_SU` 4; fully unrolled the plain build spilled 448 bytes and the carry build 1,408 with an IGC retry, offline). `flash_attn_dpas.cl` at sub-group 16 gets no head-512 build (its Q registers alone are another 128 GRF), so on Xe2 quantized pairs and f16 K/V the 2D reads cannot take go through the carry path's f16 blocks (`flash_attn_kv_to_f16` copies f16 rows too now). Xe-HPG: `flash_attn_dpas.cl` at sub-group 8, f16 (`FA_QG`, `FA_VBLK`) and the three quantized pairs, local memory 62,464 of the A770's 65,536 bytes (45 KB at head size 256), one work-group a core as at 256. They are built on first use, from `supports_op`, with the head-256 builds' options; `supports_op` loads the device's kernels first (they otherwise load at its first buffer allocation, after a filtered `test-backend-ops` run has already asked), and the attempt is recorded only once they are loaded. Ggml's head-512 prepass program is not compiled on Intel, only its tile sizes are set. Decode and verify rows (up to 8; 9-64 as row blocks of 8) take the existing split kernel `flash_attn_*_q1_split_intel` and its merge from the decode-only program at head size 512; on Xe-HPG that program is built at 256 GRF (at 128 the f16 kernel spilled 9,024 bytes and the quantized 1,504-1,728; at 256 none; Xe2 at 128 none). `supports_op` takes head size 512 on Intel where those kernels exist and the call is theirs: f32 Q, f16 K/V or q8_0:q8_0, q8_0:q4_0, q4_0:q4_0, an f16 mask or none, no sinks, ALiBi or softcap, no broadcast over dimension 3. Otherwise it declines as before and says why: `op=FLASH_ATTN head=512 kv=<k>:<v> path=declined reason=switched-off|kv-type|sinks-alibi-softcap|mask-type|broadcast|no-decode-kernel|no-dpas-kernel` (a `supports_op` count, the scheduler's probes). With an f16 cache flash attention then stays on for the whole model, so the head-256 layers return to 0008/0038's kernels too. `GGML_OPENCL_FA_DK512=0` keeps ggml's decline.

Deviation from the reference (`code`). llama.cpp's CUDA backend computes all 512 columns in one block (`fattn-mma-f16.cuh`, the DKQ 512 configs: Q not kept in registers, `Q_in_reg` false, 32-64 keys a softmax step, threads and occupancy traded per column count); here Q is not in registers either (re-read by 2D reads or staged), but S is computed once per half of V, so a head-512 prompt does 1.5x the minimal DPAS work (Q K^T twice). Sharing S between the halves through local memory (0018's S exchange) is the next form to try; it is not built.

Offline (`measured-here`, ocloc 26.27 / IGC 2.38.2, `-device bmg` and `-device acm-g10`): every head-512 build compiles without spill; the head-256 and head-128 builds' ISA is identical to the series' apart from IGC's shader hash. On the cards (`measured-here`, 2026-11-07, the review-fixed tree 20e67a73): the head-512-only filtered `test-backend-ops` run executes 82 cases and all pass on both cards. `GGML_OPENCL_FA_DK512=0` turns 81 of them into "not supported". The red build (`-DFA_DVH_RED`: every half reads V's first 256 columns) fails 36 head-512 prompt cases on each card, and the head-256 cases and those of up to 64 rows pass; the green build passes 145/145. The full `FLASH_ATTN_EXT` suite passes 3,078 cases with 0 fail on both cards.

Served A/B against 1.6.0 (`measured-here`, B60, arcint served, Gemma 4 E4B, ctx 16,384, a 6.5k-token prompt, decode over 400 tokens including the prefill):

| KV | prefill 1.6.0 -> 0106 | decode 1.6.0 -> 0106 |
|---|---|---|
| q8_0:q4_0 | 176-246 -> 1,905-2,053 t/s | about 65 -> 75.5 t/s |
| f16 | 1,276-1,294 -> 2,113-2,381 t/s | 79 -> 81 t/s |

With f16 KV the thinking answers were broken in 2 of 3 runs before and are right in 3 of 3 after. KL against the CPU reference (`measured-here`, c 2,048): f16 KV, ubatch 512, B60 0.0079 (A770 0.0095); q8_0:q4_0 0.019 on both cards (the 1.6.0-era baseline 0.0205); q8_0:q4_0 at ubatch 1 0.011. Granite 4.2 8B is unchanged: 0 downgrades, prefill 1,591-1,704 t/s (B60).

Tests (`test-backend-ops`): Gemma 4's geometry, `FLASH_ATTN_EXT` at head sizes 512 and 256, 8 query heads on 2, f16, q8_0:q8_0 and q8_0:q4_0, 1, 3 and 8 rows (decode, verify), 12 and 64 (8-row blocks), 128 and 512 (the prompt kernels), 513, 1,013 and 4,096 keys. At head size 512 with 128 and 512 rows also: no mask, an all-zero mask, a permuted Q (f16 and q8_0:q4_0), q4_0:q4_0 (and at 1 and 8 rows), a causal mask at 4,096 keys, and a causal 128-row prompt behind 20,000 keys (f16 and q8_0:q4_0): past the carry path's 16,384-key block, so the quantized one crosses a block. And the `MUL_MAT` products the same attention is without flash attention (K Q permuted and broadcast, K Q V over a strided transposed V at n_kv 1,280 / 1,792; 1, 3, 64 and 512 columns), which still take the generic GEMM and the f16 matvec; on Intel the head-256 one-column K Q passes only with 0107.

## 0107-opencl-f16-matvec-x8-q-staging.patch

A correctness fix, upstream's defect (`code`: the same lines at the pin and at `bed0a85`). `kernel_mul_mat_f16_f32_l4_x8` (`mul_mv_f16_f32_l4.cl`), the f16 matvec for one column a batch at K 128-256 with 64 or more rows a multiple of 8, staged the column in local memory one float4 a sub-group lane, `if (sgs_lid < ne00 / 4) q_loc[sgs_lid] = y4[sgs_lid];`, written for Adreno's 64-wide sub-groups. On Intel the work-group is 32 work-items and IGC builds the kernel at SIMD32 on both cards (offline, IGC 2.38.2), so at K 256 the entries from 32 on were never written and every product read stale local memory for its second half. Now every work-item of the work-group stages its share (`_x8_pair`'s staging likewise).

Two opt-in relatives assume 64 lanes beyond their staging and are now refused off Adreno, with one log line when asked for: `_x8_pair` (`GGML_OPENCL_MM_KQ_PAIR=1`: `sgs_lid >> 5` picks the row of a pair, always 0 on 16 or 32 lanes, so rows 1, 3, 5 and 7 of every 8 were never written) and `_x8_gqa4` (`GGML_OPENCL_MM_KQ_GQA=1`, K 128 and 8 query heads a KV head: `sgs_lid >> 3` picks the query head, so heads 4-7 were never written on 32 lanes). Neither is on by default (`code`; found in review).

Where it bites: batched products of one column, which in practice is attention without flash attention at decode. The host picks the kernel when `ne11 * ne12 >= 4`, so weights (one batch) never reach it. Gemma 4 E4B with an f16 cache ran without flash attention before 0106, and at one token a ubatch each of its 35 sliding-window layers' K Q (K 256) and every K Q V at n_kv 256 went through it; the head-512 K Q and longer K Q V take `_y8`, which reads the column from global memory and is right. `measured-here` by the operator's KL arms on the B60 (llama-perplexity, Gemma 4 E4B Q4_K_M, c 2,048, CPU reference, series through 0099): f16 KV with flash attention off at ubatch 512 KL 0.0077, top-1 97.5 %; at ubatch 1 KL 3.37, top-1 35.5 %, PPL 129 against 14.1. Gemma's own graph ops replayed by `test-backend-ops` (`test-export-graph-ops`, then `--test-file`) fail at `kq-0` (f16 [256,1024,2] permuted x f32 [256,1,8]) with ERR 1.04 on the B60 and 0.997 on the A770. Served, a thought that started right and fell into a loop. After 0107, f16 KV with flash attention off at ubatch 1: KL 0.0044 on both cards, top-1 98.5 %. With 0106 Gemma keeps flash attention on and no longer reaches this kernel; the fix stands for every other model run without it.

Tests: `MUL_MAT` f16 x f32 with one column a batch, 2 batches broadcast to 8: K 256 at 256, 1,016, 1,280 and 4,096 rows on the permuted K view, K 128 and 256 on contiguous sources, K Q V at n_kv 256 over a strided view. Red (`measured-here`, 2026-11-07): the tree without 0107 fails the K 256 one-column cases, 3 FAIL on each card with ERR about 1.0, and passes K 128, the green control, whose 32 float4 the 32 lanes did fill (K 64 never reaches the kernel, the host takes it from K 128). With 0107: 41/41 green on both cards. Asking for an opt-in relative off Adreno logs one warning (64 lanes assumed) and the cases stay green.

## 0108-dory-looped-ngpt.patch

The `dory` architecture, NVIDIA's looped nGPT (Dory1.0-4B). **Provenance:** upstream llama.cpp pull request 30306 ("dory support for llama.cpp", author `brianchmielNV`, branch `dory-support`), open and unmerged on 2026-11-07, head `1c3cdd23` (seven commits; the model card pins the second, `98a74c97`, which lacks the later placement fixes). Ported from the C++ hunks of that diff, 317 added lines in `src/models/dory.cpp` (new), `src/models/models.h`, `src/llama-arch.{h,cpp}`, `src/llama-model.{h,cpp}`, `src/llama-model-loader.cpp`, `src/llama-context.cpp`, `src/llama-kv-cache.cpp`, `src/llama-quant.cpp`, `src/llama-ext.h` and `common/fit.cpp`. The graph was read against vLLM's merged implementation (`vllm/model_executor/models/dory.py`, `code`) and agrees on the nGPT normalisation (`x / sqrt(sum x^2 + 1e-6)`, which is `build_gdn_l2_norm`), the slerp residual with `|alpha| * alpha_init / init_std`, the q scale `(sqk * sqrt(hd) / init_std)^2`, the output gate, the SwiGLU scales `suv * sqrt(n_embd)`, the logit scale `1 / init_std`, the loop seed `norm(state + input_h)` and the two RoPE bases by profile. It is built (the recipe host's package build, with the unit ladder passing) and, since 2026-11-07, run on both cards (results at the end of this section).

What it is. The file holds 72 physical blocks (16 input, 40 recurrent, 16 output); the recurrent ones run `dory.num_loops` times (4 in the file) over the same weights, so `hparams.n_layer_all` becomes the executed count (112 at 2 loops, 192 at 4). The weights, the offload placement (`n_layer_physical`, `layer_physical(il)`), the offload fit in `common/fit.cpp` and the quantizer's FFN counts follow the physical blocks; the graph, the KV cache and `llama_model_n_layer` follow the executed layers. `dory.recurrent_kv_cache_mode` is `per_loop` (a KV a loop and layer) or `last_loop` (one a layer, each loop overwriting the previous loop's rows). The 217 BF16 scale tensors are cast to F32 in the graph, which is why `weight_buft_supported` gets a `GGML_OP_CPY` probe: a backend that cannot cast BF16 puts the scale in a CPU buffer (0109 for OpenCL). `llama_model_n_layer_physical` is new in `llama-ext.h`; arcint's admission reads the physical count through it. The context shift is refused for the architecture.

What was dropped, and why: the Python conversion (`conversion/dory.py`, `gguf-py` constants, writer and tensor mapping; the GGUF is converted upstream of this repository); the PR's `llama_arch_supports_save` hunk (the pin names that list `llama_model_saver_supports_arch`, in `src/llama-model-saver.cpp`; Dory is added to it there, so the saver tests skip it); the PR's `tests/test-llama-archs.cpp` fixture and its `tests/CMakeLists.txt` line (that file differs from the pin's at 11 of 16 hunks; adapting it blind would be a test that has never failed). Instead the pin's generic arch tests skip Dory (`arch_supported` and the backend loop, as they skip other architectures that need a fixture), because the generic fixture carries none of its recurrence metadata and `LLM_ARCH_DORY` is first in `llm_arch_all()`. Dory's coverage is arcint's own unit tests and the card arm. One loader line (`get_arr<uint32_t>` for a string key) is placed after the pin's neighbouring instantiation, where the context drifted. The rest applies as it was written. Added to the port: a Dory file that carries per-tensor `.scale` tensors is refused at load, because the generic scale pass after `load_arch_tensors` names them by executed layer index and a looped copy would read another block's scale (the published file has none).

Findings on the pull request (`code`, read; none run). (1) `last_loop` makes a token's hidden state depend on where the prefill's chunk boundaries fall: loop i of a chunk attends to loop i's KV inside the chunk and to the last loop's KV before it. The PR's own test says so ("shared final-loop KV depends on chunk boundaries") and compares such a model only against the same batch. For arcint this means chunked prefill, prefix reuse and a different `--prefill-chunk` give different, equally valid results in that mode, so equivalence there is judged at the answer, and `per_loop` (the file's default) has no such dependence. (2) `swa_full` stays true in arcint, so the sliding-window layers keep a full-length cache and a prefix trim is exact; the KV is then 96 attention layers at 4 loops under `per_loop` (36 attention blocks of the 72, 8 + 20 x 4 + 8 executed), each a full-length cache, which sizes `--n-ctx` on a 16 GB card. (3) `graph_max_nodes` is `max(1024, 128 * n_layer)`, a bound chosen by hand for the executed layer count; a node-count overflow would abort at the graph build, not give wrong output. (4) The model card's pinned revision `98a74c97` is not the head this patch follows: the later commits move the BF16 placement and the offload counts, which is what this patch needs.

On the cards (`measured-here`, 2026-11-07, build 6fa32c1e, Dory1.0-4B Q5_K_M, served; decode over 400 tokens including the prefill, prefill over a 5.9k-token prompt):

| card | loops / KV | decode | prefill | VRAM resident after the requests (fdinfo) |
|---|---|---|---|---|
| A770 | 2, q8_0:q4_0 | 23.7 t/s | 512-665 t/s | 5.2 GiB |
| A770 | 2, f16 | 24.9 t/s | 727 t/s | 7.1 GiB |
| A770 | 4, q8_0:q4_0 | 14.3 t/s | 396 t/s | 6.2 GiB |
| A770 | 4, `last_loop`, q8_0:q4_0 | 14.3 t/s | 392 t/s | 4.7 GiB |
| B60 | 2, q8_0:q4_0 | 36.4 t/s | 1,287 t/s | 4.6 GiB |
| B60 | 2, f16 | 37.9 t/s | 1,592 t/s | 6.6 GiB |
| B60 | 4, q8_0:q4_0 | 22.0 t/s | 742 t/s | 5.6 GiB |

The KV cache layers counted at start are 56 (2 loops), 96 (4 loops, `per_loop`) and 36 (`last_loop`), as the formula says. The answers are right in every cell (a capital, a sum with thinking, a tool call). KL against the CPU reference on the A770: 2 loops 0.0091 (ubatch 1 0.0091), 4 loops 0.0107, q8_0:q4_0 at 2 loops 0.016. 0 DOWNGRADE lines. The `last_loop` row saves 1.5 GiB of peak VRAM at equal speed; the equivalence across chunk sizes in that mode is judged at the answer, as the findings above say.

## 0109-opencl-cpy-bf16.patch

The OpenCL `CPY`, `DUP` and `CONT` take a BF16 source. `supports_op` accepted only F32, F16 and I32 sources, so the cast Dory's graph makes of each BF16 scale vector to F32 was refused by the device: `weight_buft_supported` then put all 217 scale tensors in CPU buffers and the scheduler split the graph around each one (`inferred`: hundreds of splits a forward; CUDA had the same failure before the PR's own fix, a 3.1x slowdown in the scoping study's reading of the PR discussion, `paper`). A BF16 tensor is already stored as F16 on the device (`set_tensor` converts it with `kernel_convert_bf16_to_f16`, `tensor_get` converts back; `code`, `ggml-opencl.cpp`), so `ggml_cl_cpy` takes BF16 as F16 and `supports_op` lists BF16 beside F16 with the same destinations (F32, F16). No new kernel.

Test: `test-backend-ops -o CPY` gets BF16 to F32 and BF16 to F16 at Dory's shapes, vectors of 2560, 4096 and 7168 (the model width, the attention width, the FFN width) and the activation-shaped 2560 x 256, contiguous and with the first two axes swapped. The existing 256 x 4 x 4 x 4 BF16 sources run now instead of being skipped as unsupported. Limits of that test: `test-backend-ops` reads its CPU reference inputs back from the device, so the upload's BF16 to F16 rounding (values above 65504 or below 6e-8 lose bits) is invisible to it; whether the rounding costs anything is the served KL's to decide, and the scale vectors are small numbers (init std 0.0177, `alpha_init` 0.05).

Red case: build without this patch and `-o CPY` reports the BF16 cases as not supported, and the served model shows the splits.

Served red/green (`measured-here`, 2026-11-07, A770, build 6fa32c1e, Dory 2 loops, q8_0:q4_0, same window): without 0109 decode 14.7 t/s and prefill 540 t/s; with it 23.7 and 665. The red arm printed no DOWNGRADE line: a weight-fed op that `supports_op` declines lands on the CPU invisibly, because the decline happens before the scheduler and the path-key logging of 0100 never sees it. That is an open gap in the downgrade accounting (0106 closed it for head-512 attention only, with its `path=declined` keys). 0111 closes it for every weight the loader places.

## 0110-opencl-f32-weights-f16-copy.patch

Gemma 4 E4B keeps its per-layer-input projections F32 in the GGUF: `blk.N.proj.weight` [256 x 2,560] and `blk.N.inp_gate.weight` [2,560 x 256], 42 of each, 220 MB, read from the file's header (`code`). The prompt's `proj` products took ggml's F32 tiled GEMM (`kernel_mul_mm_f32_f32_l4_lm`, the key `type=f32 k=256 cols=16+ path=generic reason=no-fast-path`, 41 a prompt graph: the last layer's rows are cropped to the output token), and `inp_gate` (256 rows) the F32 skinny path of 0005, which reports nothing. Both read F32 at decode. Every value of the four tensors read (layers 0, 20, 41) is a bf16 (the low 16 bits zero), at most 0.78 in magnitude; 0.02-0.56 % of them are below 2^-17, where f16 keeps fewer bits (`measured-here`, a read of the file).

Now an F32 2-D weight on Intel gets an f16 copy when it is written (`ggml_cl_f16_copy_build`, from `set_tensor`, a whole write into a buffer marked as weights), and `MUL_MAT` multiplies with the copy through the F16 paths (`ggml_cl_mul_mat_f16_copy`): from 16 columns the XMX GEMM `kernel_mul_mm_f16w_f16` (sub-group 16; at sub-group 8 it is not built, so the A770 keeps F32 there), 1 column the f16 matvec, 2-8 the few-column `kernel_mul_mv_f16_f32_nc`. 9-15 columns and the Hadamard rotations keep the F32 paths. A 3-D src1 whose columns lie back to back is flattened (`ggml_cl_flat_cols`) and the flattened product takes the copy; one that does not flatten keeps F32. A bf16 value converts to f16 exactly from 2^-17 up; the XMX path also rounds the activations to f16 (`kernel_cvt_f32_f16`, scaled by a power of two per column above 16,384). Kept F32: weights under 256 rows or with K under 256, views, USM-host buffers, a weight with a value outside f16, and by name the expert routers (`ffn_gate_inp`), the Hadamard rotation matrices (the KV cache's `attn_rot_dev_N` and the model's `prism.hadamard.N`: their buffers are marked as weights, but their products go to 0095's transform or carry the Hadamard hint, never to a copy), and the vocabulary-size products `output.weight` and `token_embd` (an F32 output head would add a copy of hundreds of MB to GB). Any later write to the weight drops its copy; a buffer `clear` or `reset` drops them all.

Memory: the copies are device memory outside the weight buffers' size, so no fit or reserve computation sees them; 110 MB for Gemma 4 E4B. The device's total (count and MiB) is logged at the first graph after loading and again when it changes. Of the other served files (headers read, `measured-here`) none has an F32 weight the rule takes: the 27B's `ssm_alpha` / `ssm_beta` have 48 rows, the routers are named `ffn_gate_inp`, `ssm_conv1d` is not a MUL_MAT. A multimodal projector file is loaded into a buffer marked as weights too, so its F32 2-D matrices (Gemma 3's `input_projection`, which the converter forces to F32) do get a copy, at half their F32 size, counted in that total; `GGML_OPENCL_F32W_F16=0` keeps them F32. The projector files here (Qwen2.5-VL Q8_0, Qwen3-TTS Q8_0 and bf16) have none: their F32 tensors are 3-D convolution weights.

The key: a weight the rule would copy but that has no copy (switched off, a failed allocation, a value outside f16, a partial write) reports `op=MUL_MAT type=f32 k=<K> cols=1|2-8|9-15|16+ path=f32 reason=no-f16-copy` at every product, whatever F32 path then runs (the skinny path and the few-column matvec report nothing else), and the generic-GEMM and matvec keys are not added for it. So with the switch off both of Gemma's projections report, `k=256` and `k=2560`, at every column count; with the copy the old `type=f32 k=256 cols=16+ path=generic reason=no-fast-path` key vanishes. The rotations' `type=f32 k=512` key stays as it was, and so do the routers' keys.

Reference (`code`). Upstream's Vulkan backend multiplies F32 weights at f16: on an fp16 device `matmul_f32_f32` is built with `FLOAT_TYPE` float16_t, so both operands are rounded to f16 in shared memory before the cooperative-matrix product (`vulkan-shaders-gen.cpp`, `matmul_shaders`). CUDA keeps F32: cuBLAS SGEMM above 16 columns, TF32 mma (`mmf.cu`) up to 16 on Ampere and later. ggml-opencl itself already stores BF16 weights as F16 (`kernel_convert_bf16_to_f16` in `set_tensor`). Deviations: the copy is made once at load, not per product; the routers stay F32, which Vulkan does not do.

Estimate before building (`inferred` from shapes and two `measured-here` rates on the B60: 0022's 320 x 10,240 f16 product at 512 tokens, 1,907 us on ggml's tiled GEMM against 468 us on the XMX tile, about 1.8 and 7 TFLOPS). Per prompt token `proj` and `inp_gate` are 1.31 MFLOP a layer each, 107 MFLOP over 41 layers, 0.7 TFLOP for a 6,656-token prompt: on the F32 kernels about 200 ms (`proj`) and 120-180 ms (`inp_gate`), on the XMX tile about 30 and 50 ms, so about 250-300 ms off a 2.9-3.5 s prefill (+7-10 %), at f16 and q8_0:q4_0 alike. Decode reads 220 MB of F32 a token of about 3 GB; f16 halves that, worth 0.4-1.1 ms of a 12.3 ms step if the matvecs run at 100-300 GB/s (unmeasured). The other generic product, the K-cache rotation at 512 (`type=f32 k=512`, q8_0:q4_0 only), is 17 GFLOP an ubatch, about 125 ms a prompt on the tiled GEMM; extending 0095's transform to 512 would save most of it and is not done here.

`GGML_OPENCL_F32W_F16`: `0` no copies, `1` every column count, `2` 16 or more columns only (the default). Why 2 (`measured-here`, B60, Gemma 4 E4B Q4_K_M, llama-bench pp4096 / tg128, ubatch 512, FA on, 5 repetitions, each arm twice interleaved): off 4,244 / 83.70 t/s at f16 KV and 4,026 / 77.46 at q8_0:q4_0; prompts only 4,408 / 83.77 and 4,173 / 77.40 (+3.9 % and +3.7 % prefill, decode equal); every column count 4,411 / 83.07 and 4,176 / 76.92 (the same prefill, 0.7 % slower decode: at one column the f16 matvec loses to the F32 skinny path). The expected red counts below were taken under `1`; under the default `2` a bare `GGML_OPENCL_F32W_F16_RED=1` behaves like `RED=1 F32W_F16=2`. `GGML_OPENCL_F32W_F16_RED=1` negates the last 16 of K in every row of each copy and says so in the log (the red case).

Tests (`test-backend-ops -o MUL_MAT -p 'weights='`, 29 cases): Gemma's two shapes as weights at 1, 2, 3, 7, 8, 12, 16, 17, 64, 100 and 512 columns, and with a 3-D src1 (2 x 3 and 2 x 8 columns, flattened to 6 and 16); controls that keep F32: both shapes at 512 columns from the graph's own buffer, and a weight named `blk.0.ffn_gate_inp.weight` (256 x 2,048, 64 columns). The tolerance, NMSE 5e-4, would not catch a single wrong row of 2,560 (one row zeroed is about 4e-4); the red case changes every row. Expected with the red switch: on the B60 the 24 cases the copy takes (every weight case but 12 columns) fail and the 5 others pass; with `GGML_OPENCL_F32W_F16=2` the 12 from 16 columns (the 3-D 16-column case included); on the A770 the 12 at 1-8 columns (the 3-D 6-column case included), and none with `=2`. With `GGML_OPENCL_F32W_F16=0` all 29 pass and both projections report `reason=no-f16-copy`. On the cards (`measured-here`, 2026-11-07, build b16f7bb5, Gemma 4 E4B): the op suite is green on both cards; the red switch fails the cases it should (B60 14, A770 8); `GGML_OPENCL_F32W_F16=0` shows `no-f16-copy` keys; the full `MUL_MAT` suite passes 2,299 cases with 0 fail on both cards. KL against the CPU reference, 0110 off -> on: f16 KV ubatch 512 0.0079 -> 0.0095, top-1 97.80 -> 97.56 %; f16 KV ubatch 1 0.0044 -> 0.0044; q8_0:q4_0 0.0192 -> 0.0190. Served on the B60 with 0110 off, on and for prompts only: prefill within the run-to-run spread (f16 KV 2,108-2,429 t/s across the arms), decode equal; the estimate above (+7-10 %) is therefore not confirmed, and a repeated benchmark is pending. The key `type=f32 k=256` vanishes with 0110 on; the one Gemma still reports is `type=f32 k=512`, the K Hadamard rotation at 512 (q8_0:q4_0 only). The coder (A770, the unit's flags) peaks at 16,276 MiB with and without the patch. Unmeasured: at one column `inp_gate` now runs `kernel_mul_mat_f16_f32_1row` in place of the F32 skinny GEMV.

## 0111-llama-placement-downgrades.patch

A weight that the device declines at load, and that therefore lands in host memory, now says so. The loader probes every weight's op on the buffer types of its layer's list (`weight_buft_supported` in `src/llama-model-loader.cpp`) and takes the first one the device's `supports_op` accepts; when the device declines, the weight goes to a CPU buffer and the scheduler splits the graph around it. No backend path is ever taken for such a weight, so 0100's keys never see it: Dory on a build without 0109 decoded at 14.7 t/s against 23.7 and printed no DOWNGRADE line (`measured-here`, 0109's section).

The report sits at the loader's decision, because only there are the reason (the declined probe op) and the intent (the buffer type the layer asked for) both known; a walk over the model's buffers after the load would see the placement but not why. `src/llama-placement-report.h` (device-free, arcint's unit tests include it) holds the rule and the key; `llama_placement_downgrades` in `llama-ext.h` returns the counts in `ggml_backend_opencl_downgrades`' format, and arcint puts them into `/props` `downgrades` next to the OpenCL keys.

| Where | Key |
|---|---|
| a weight whose layer asked for a device buffer, the device declined the probe op, and a host buffer took it | `src=llama op=<probe op> type=<weight type> weight=<name, every numeric part as N> buft=<buffer type it got> reason=device-declines` |

One `DOWNGRADE src=llama <key>` line at warn level the first time a key occurs in the process; the count is the number of weights under the key, summed over the models the process loads (the MTP head and a drafter add to the target). The buffer type is the one after the loader's mmap swap, so `CPU`, not the host buffer type the swap replaces. Not reported, because somebody asked for them: a `tensor_buft_overrides` match (arcint's `--llama-cpu-moe`, which also carries the expert cache and its CPU misses), a lazily read tensor (Flash-Next's n-gram table), the input layer (every `LLM_TENSOR_LAYER_INPUT` tensor, `token_embd` and `per_layer_token_embd` among them: its list is the CPU list by design, and the loader skips the layer), and every tensor of a list that does not start in device memory (layers or an output not offloaded). Device memory is decided by the buffer type's owner, not by `is_host`: a buffer type that is not a host buffer and belongs to a GPU, an integrated GPU or the meta device of a split. `is_host` alone cannot tell, because the CPU's extra buffer types (`CPU_REPACK`, AMX) hold repacked data and leave it unset; with extra buffer types on, `CPU_REPACK` heads the CPU list, and the first form of this patch, which ruled on `is_host`, took the input layer's list for a device list and reported every input embedding (`measured-here`, 2026-11-07, card arms of e01ed64a: Dory `op=GET_ROWS type=bf16 weight=token_embd.weight`, Gemma 4 E4B `type=q4_K weight=token_embd.weight` and `type=q5_K weight=per_layer_token_embd.weight`). The same rule also missed a decline that lands in `CPU_REPACK` (it took that for device memory too); it now reports, under `buft=CPU_REPACK`. A fall from one device buffer type to another (a row-split buffer to the device's own) is no host placement and says nothing. A memory-fit pass (`no_alloc`) places nothing and does not count. A K-quant or IQ weight the OpenCL device declines also keeps 0100's `op=MUL_MAT|MUL_MAT_ID ... path=cpu` key, which its `supports_op` writes during the same probe. Under `--llama-layer-split` every layer and the output have a device list, so a decline on either card reports. Not covered: the multimodal projector, which libmtmd loads without this loader.

Expected keys, from the model files' headers and the series' `supports_op` (`code`): Dory with 0109 none, without it 6 keys of `op=CPY type=bf16` (`blk.N.residual_alpha.weight` 72, `blk.N.attn_gate_scale.weight`, `blk.N.attn_qk_scale.weight`, `blk.N.ffn_gate_scale.weight`, `blk.N.ffn_up_scale.weight` 36 each, `output_scale.weight` 1: 217 weights; `measured-here` on the card, exactly these six with these counts, plus the input-embedding key the fixed rule no longer gives); Gemma 4 E4B, Granite 4.2 8B, the coder with its MTP head and the 27B none. Flash-Next one, `op=SSM_CONV type=f16 weight=blk.N.ple_conv1d.weight buft=CPU`: the f16 n-gram conv kernel of its one PLE layer is probed as `SSM_CONV`, whose OpenCL support needs an F32 weight, while the graph only takes column views of it and casts them to F32 (`CONT` and `CPY` of F16, which the card runs). So the weight is in a CPU buffer today, and its four casts a forward run on the CPU. 0112 probes the tensor with the op the graph applies to it, and the key goes away.

Tests: arcint's `tests/test_placement.cpp` runs the header's rule (device memory by owner, on the boolean form and on the CPU backend's real buffer types, the extra ones included, which must be present in the build), key and counts, the merge of both sources into `/props`, and checks the call site in the loader as text (set only by the probe's selection, after the mmap swap, guarded against the fit pass). The card case: Dory served from a build without 0109 must print the 6 keys, the full tree none.

## 0112-qwen4exp-ple-conv-probe-cpy.patch

Flash-Next's PLE conv kernel (`blk.N.ple_conv1d.weight`, f16 [4 x 10,240] in both served files, one PLE layer, `blk.1`) is probed with the op the graph applies to it. `LLM_TENSOR_PLE_CONV1D` carried `GGML_OP_SSM_CONV`; OpenCL's `supports_op` takes `SSM_CONV` for an F32 weight only, so the loader put the f16 kernel in a CPU buffer, which 0111 reports as `op=SSM_CONV type=f16 weight=blk.N.ple_conv1d.weight buft=CPU reason=device-declines`. The graph never runs `SSM_CONV` on it (`code`, `src/models/qwen4exp.cpp`, the PLE block's depthwise conv): for each of the 4 taps it takes a column view (`ggml_view_2d`, one f16 a channel, 10,240 channels, a row stride of 8 bytes), makes it contiguous (`CONT`, f16 to f16) and casts it to F32 (`CPY`, f16 to f32) before a `MUL` with the shifted activations. OpenCL takes both: `CONT` and `CPY` from F16 to F16 or F32, any strides (`kernel_cpy_f16_f16`, `kernel_cpy_f16_f32` address by byte strides, so the view's 2-byte offsets are fine). The probe is now `CPY`, the cast, so the kernel lands in the device buffer and the four views and casts run on the card instead of the CPU, with the transfers around them. `LLM_TENSOR_PLE_CONV1D` is used by `qwen4exp` only.

Memory: 4 x 10,240 x 2 = 81,920 bytes, one layer: 80 KiB of weights more on the card, and the casts' outputs (10,240 f16 and f32 values a tap) in the compute buffer. Against the unit's peak of 23,478 MiB of 24,480 (its file's comment) that is 0.0003 % of the card: not a switch.

Tests: arcint's `tests/test_placement.cpp` checks that the probe op is `CPY` and that every read of the kernel in the graph is the column view that is then cast. The card A/B (served, B60, the unit's flags, two different long prompts and a 400-token greedy decode each, the 0111-only build against 0111 + 0112, two rounds interleaved) on the card (`measured-here`, 2026-11-07): prefill 503.4 -> 514.1 t/s (+2.1 %) on one prompt and 518.8 -> 523.4 (+0.9 %) on the other; decode moved between 18 and 27 t/s in both arms across rounds, so its change is not readable at that spread; peak VRAM 24,035 / 24,034 MiB. The 0111 key `op=SSM_CONV type=f16 weight=blk.N.ple_conv1d.weight` is printed by the 0111-only build and gone with 0112.

## 0113-dory-scales-folded-at-load.patch

Dory's per-channel scale vectors are computed once at load instead of on every forward, and two norms the graph repeats on an already normalised vector are dropped. **This deviates from the PR's graph** (upstream pull request 30306, ported in 0108); the operator allowed that for Dory on 2026-11-07, judged at the answer-level bar.

Why. The PR's graph builds each scale from its BF16 weight in the forward: a cast, a scale and an `abs` or a square for every vector a sublayer reads (`|alpha| * alpha_init / init_std`, `(sqk * sqrt(hd) / init_std)^2`, `(s_gate * sqrt(gate_init) / init_std)^2`, `suv * sqrt(n_embd)`, and `logit_scale / init_std` once), and the forward is long because the recurrent blocks run once a loop. Counted on the recipe host from the graph builder (`measured-here`, 2026-11-07: a small program calls `llama_graph_reserve` on the CPU backend for one token, ctx 8,192, ubatch 512, flash attention on, the Q5_K_M file, and counts the nodes; "launching" leaves out `NONE`, `VIEW`, `RESHAPE`, `PERMUTE` and `TRANSPOSE`):

| loops, KV | PR graph: nodes / launching | `LLAMA_DORY_FOLD=1` | `LLAMA_DORY_FOLD=2` (default) |
|---|---|---|---|
| 2, q8_0:q4_0 | 4,947 / 3,771 | 3,881 / 2,761 | 3,655 / 2,535 |
| 2, f16 | 4,275 / 3,547 | 3,209 / 2,537 | 2,983 / 2,311 |
| 4, q8_0:q4_0 | 8,477 / 6,461 | 6,651 / 4,731 | 6,261 / 4,341 |
| 4, f16 | 7,325 / 6,077 | 5,499 / 4,347 | 5,109 / 3,957 |

The PR graph at 2 loops, q8_0:q4_0, by op: 901 `SCALE`, 673 `MUL_MAT` (449 weight products and the KV cache's 224 Hadamard rotations), 564 `RMS_NORM`, 449 `MUL`, 337 `CPY` (the BF16 casts), 112 each of `ABS`, `SQR`, `SUB` and `ROPE`. Counted the same way with the OpenCL backend's adjacency fusions applied by op (0049's rms_norm + scale and sigmoid + mul, rms_norm + mul, scale + silu; `inferred`, because the fusions' own type, shape and alias checks do not run on the CPU), the card launches about 3,151 kernels a forward at 2 loops q8_0:q4_0, 2,028 with this patch and 1,580 with 0114 too; at 4 loops 5,397 and 2,704. At the B60's host gap of about 2.6 us a launch (memory note, `measured-here` on other graphs) the PR graph's launches alone are about 8 ms of a 27 ms decode step at 2 loops; that decode is launch-bound is `inferred`, the card A/B decides.

What changes, as the graph is built (`src/models/dory.cpp`):
- `load_tensors_post`, a new virtual of `llama_model_base` called after the weights and the Hadamard tensors are loaded (`src/llama-model.cpp`), computes the final vectors in F32: `alpha` = `fabs(x * (alpha_init / init_std))` [n_embd], `sq` = `(x * (sqrt(hd) / init_std))^2 * (1 / sqrt(hd))` as a 2-D [hd, n_head] tensor, `sg` = `(x * (sqrt(gate_init) / init_std))^2` [n_q], `cg`, `cu` = `x * sqrt(n_embd)` [n_ff], and the logit scale `x * (1 / init_std)` [n_vocab]. The factors are the graph's own float expressions and the operations run in the graph's order, so the values equal what the CPU backend computed in the forward, bit for bit. A `LLAMA_DORY_FOLD` value other than 0, 1 or 2 logs a warning and 2 serves. A looped copy of a block shares its physical block's vectors.
- Placement (revised after review). Folded, the graph never reads the 217 BF16 sources, so they are created with a new loader flag, `TENSOR_HOST_ONLY` (`src/llama-model-loader.{h,cpp}`): the CPU's own buffer type, mapped from the file under mmap, no probe and so no 0111 key for a weight nothing reads. The values are read from that host copy, which is the file's BF16, not from a device copy: OpenCL holds a BF16 tensor as F16 (0109), which the PR graph's on-device cast read, so on the card the folded values are the exact BF16 ones where the PR graph read F16-rounded ones (they differ only outside f16's normal range, below 6.1e-5 or above 65,504). Each fold is placed by its own layer's device: `select_buft_f32_mul` (`src/llama-model.cpp`) probes an F32 `MUL` of 512 rows by the vector over the layer's buffer-type list (the output layer's for the logit scale), as the loader probes a weight, and a CPU choice is the CPU's own buffer type. A device that declines that `MUL` puts the fold in host memory, and the fold is then reported as 0111 reports a weight: `DOWNGRADE src=llama op=MUL type=f32 weight=blk.N.residual_alpha.weight.fold buft=CPU reason=device-declines` and the like. The first form placed each fold by its source's buffer type, which a BF16 `CPY` probe had chosen: on a device without 0109 the folds went to host memory, the graph split at every sublayer, 0114 never matched, and 0111 kept reporting the sources the graph no longer read (found in review). `own_buffer` hands each context and buffer to the model, so `memory_breakdown` counts them. The memory-fit pass (`no_alloc`) now runs `load_tensors_post` too, with the loader's zero-size dummy buffers: the fit counts the folds and plans the folded graph. Size: 217 vectors, 1,126,400 values, 4.3 MiB F32 (the load log line on the recipe host's CPU: `Dory: 217 scale vectors folded at load, 4.30 MiB F32 in CPU`, which names the buffer types); the BF16 sources (2.1 MiB) are host memory, mapped from the file under mmap.
- q: `rms_norm(q, 1e-6/hd) * sq` instead of `(rms_norm(q, 1e-6/hd) * (1/sqrt(hd))) * reshape(sq)`. At hd 256 the factor is 2^-4 and both orders give the same bits. The move is made only when sqrt(hd) is a power of two (checked at load, `sq_has_qnorm`); otherwise sq keeps the PR's values and the q norm keeps its scale, `norm(q) * sq`; the pair is now one `RMS_NORM` + `MUL`, which OpenCL fuses (`kernel_rms_norm_mul`). The 2-D `sq` leaves no `RESHAPE` node between the two.
- The output gate `sigmoid(gate * sg)` and the SwiGLU `swiglu_split(g * cg, u * cu)` (one `GLU` node instead of `SILU` and `MUL`; on the CPU the same values).
- At `LLAMA_DORY_FOLD=2` (the default) also: the residual of every sublayer but the first is the previous sublayer's output, which is already `norm(...)`, and is no longer normalised again (`a = norm(residual)` stays for the embedding); and the `norm` after each loop's last block, applied to that block's output, which is already normalised, is dropped. A second norm of `v = norm(t)` divides by `sqrt(S/(S + eps) + eps)`, S the sum of squares of t and eps 1e-6: a factor of about 1 + (eps/2)(1/S - 1), under 1e-6 away from 1 unless S is far below 1 (1 - 5e-7 for a large S). So they only move rounding; dropped, a slerp output with a very short t (residual and update nearly opposite) is not rescaled again, a case the measurements below did not show (found in review).

What it does to the numbers on the CPU (`measured-here`, 2026-11-07, recipe host, `llama-perplexity`, wikitext-2 test, 16 chunks of 512, ubatch 512, the CPU's default f16 KV, `--override-kv dory.num_loops=int:2 --override-kv dory.recurrent_kv_cache_mode=str:per_loop`, against the logits of the build without this patch; PPL 12.1421):

| arm | mean KL | top-1 |
|---|---|---|
| `LLAMA_DORY_FOLD=0` (the PR graph, this build) | 0.000000 (max 6e-5, the logit file's own rounding) | 100.000 % |
| `LLAMA_DORY_FOLD=1` (folded vectors, GLU, q's scale moved) | 0.000000 (max 6e-5) | 100.000 % |
| `LLAMA_DORY_FOLD=2` (and the two norms dropped) | 0.007985 +- 0.000377 (max 0.64) | 95.858 % |
| the residual norm dropped alone / the loop-end norm dropped alone (ablation build) | 0.007262 / 0.005433 | 95.466 / 96.495 % |
| control: the PR graph with every layer's output multiplied by 1 + 2^-23 | 0.008968 +- 0.000766 | 95.637 % |

So the folded vectors and the GLU change nothing on the CPU, and dropping the two norms costs what any one-ulp change costs this model: the control, which moves every layer's output by at most one ulp and nothing else, gives slightly more KL and a lower top-1 than the patch. Dory amplifies a rounding difference through its 112 (192) layers to about 0.008-0.009 nats against a fixed reference; the A770's KL against the CPU reference was 0.0091 at 2 loops before this patch (0108's section), which is the same floor. A CPU-against-itself comparison has a floor of 0, which is no bar (CLAUDE.md, the answer-level bar), so the card arms decide: base deb and this deb against the same CPU reference, on the same card and in the same window, at ubatch 512 and 1. At 4 loops (the same text and settings, PPL 12.0318): `LLAMA_DORY_FOLD=0` 0.000000 (top-1 99.975 %: two near-ties that the logit file's rounding flips, the PR graph against its own logits gives the same), `1` 0.000000 (99.975 %), `2` 0.008032 +- 0.000483 (95.515 %), and the one-ulp control 0.008028 +- 0.000376 (95.686 %).

On the cards (`measured-here`, 2026-11-07, arcint served, a 5.9k-token prompt and a 400-token greedy decode, decode from the slot's own timing; base = 0111 + 0112, two rounds interleaved): B60 2 loops q8_0:q4_0 decode 28.6 -> 39.1 t/s (+37 %), prefill 2,522 -> 2,731 (+8 %); 4 loops q8_0:q4_0 17.2 -> 23.6 (+37 %), 1,474 -> 1,609; 2 loops f16 30.5 -> 42.8 (+40 %, the same answer's md5), 2,637 -> 2,874. A770 2 loops q8_0:q4_0 19.1 -> 28.5 (+49 %), 1,054 -> 1,121; 4 loops 11.5 -> 17.2 (+50 %), 624 -> 663. Priced apart at 2 loops q8_0:q4_0: `LLAMA_DORY_FOLD=1` 38.4 t/s (B60) and 27.7 (A770), `GGML_OPENCL_FUSE_NGPT=0` 38.1 and 27.3, so the fold carries most of the gain and the fused update 2-3 % more. KL against the CPU running the PR graph (`LLAMA_DORY_FOLD=0`), 16 chunks of 512, base -> new: B60 2 loops f16 0.00787 -> 0.00787 (top-1 96.47 -> 96.42 %), ubatch 1 0.00792 -> 0.00792, 4 loops 0.01128 -> 0.01153; A770 0.00796 -> 0.00792, ubatch 1 0.00792 -> 0.00791, 4 loops 0.01126 -> 0.01125; q8_0:q4_0 new 0.0126 (B60) and 0.0123 (A770). The review fixes (placement by an F32 `MUL` probe, the sources host-only) served the same text at the same speed (B60 39.2, A770 28.5 t/s) and the same KL.

The PR graph stays one switch away: `LLAMA_DORY_FOLD=0` (read at load) builds exactly 0108's graph (KL 0 above), `1` the folded vectors only. With 0 the OpenCL fusion of 0114 finds nothing to fuse (its pattern has the alpha chain in the middle). The memory-fit pass (`no_alloc`) builds the PR graph, which needs slightly more compute buffer, so the fit stays conservative.

Not done here, and why. Folding the SwiGLU scales into the FFN weight rows (a per-row scale of a Q5_K matrix rescales its super-block scales, which are f16: not exact, and the series has no requantizing load). Dropping the norm before `ffn_down` and `wo` (the residual's `norm` of their output is scale-invariant, so only `rms_norm` would be needed, not its 1/sqrt(n)): on OpenCL that pair is already one launch (0049), so it would save CPU nodes only.

Tests: none at the op level (no new op; the CPU KL above is the equivalence check). The node counts come from the counting program kept with the handover, not in the repository.

## 0114-opencl-ngpt-update-fused.patch

Dory's nGPT residual update in one OpenCL kernel. Each sublayer ends with `h = norm(a + alpha * (norm(x) - a))`, `norm(v) = rms_norm(v, 1e-6/n) * (1/sqrt(n))` over the width 2,560, where `a` is the residual, `x` the sublayer's output and `alpha` a per-channel vector: seven nodes after 0113 (`RMS_NORM SCALE SUB MUL ADD RMS_NORM SCALE`, in this order in the graph), five kernels with 0049's rms_norm + scale. `kernel_ngpt_slerp_f32` (`kernels/rms_norm.cl`) computes it a row a work-group: the sum of squares of x, then `t = a + (x*r1*s1 - a) * alpha` and its sum of squares, then `t*r2*s2`, recomputing t in the last pass instead of keeping it. vLLM computes the same update (`slerp_alpha` in its `models/dory.py`, `code`) inside one `torch.compile`d region; this is the same idea as a pattern in the backend's graph loop (`ggml_opencl_can_fuse_ngpt`), not a new ggml op, so the CPU backend and every other backend run the seven nodes as before.

The match (`ggml_opencl_can_fuse_ngpt`, checked before the shorter `RMS_NORM` patterns): the seven ops in sequence with `ggml_can_fuse` (each intermediate used once, not an output), `SUB` as `b - a`, `ADD` reading the same `a`, alpha on either side of the `MUL`, everything F32, x, a and the result contiguous and of one shape, the width a multiple of 4, alpha one contiguous row of that width, both scales without a bias. Liveness (the memory note on fused kernels that read an input after its consumer would have): the kernel reads x, a and alpha up to its last pass, and the allocator may have placed the result over x or a, which are dead after `SUB` and `ADD` in the unfused graph. The result may lie exactly over x or over a (the same offset: each element is read and written by one work-item, after both reductions, so a row is read before it is written); any other overlap with x or a, and any overlap with alpha, refuses the fusion and the seven nodes run. Each distinct count a graph shows is logged once (at most 32 lines): `the nGPT update fused N times in a graph of G nodes (K of them in place), M refused`, K counting the fusions whose result lies exactly over x or a (the branch the op tests cannot reach, below); so the prompt's and the decode's graphs, and a later graph that refuses where an earlier one fused, each say so (the first form logged the first graph only; found in review); on Dory, N is 112 at 2 loops and 192 at 4 by op adjacency on the CPU graph (`measured-here`, the counting program of 0113), and M is the card's to show. Launches a forward (`inferred`, 0113's estimate): 2,028 -> 1,580 at 2 loops q8_0:q4_0, 1,356 at f16, 2,704 at 4 loops q8_0:q4_0, 2,320 at f16.

Switches: `GGML_OPENCL_FUSE_NGPT=0` runs the seven nodes (off switch); `GGML_OPENCL_FUSE_NGPT_RED=1` makes the fused kernel read alpha as 1, so the update returns `norm(norm(x))` (the red case, logged at start). `GGML_OPENCL_DISABLE_FUSION` turns it off with the others.

Tests (`test-backend-ops -o NGPT_SLERP`, 9 cases, `test_ngpt_slerp`): width 2,560 at 1, 7 and 512 rows (decode, a few rows, a prompt ubatch), 256 x 64, 2,560 x 4 x 3 (3-D rows), alpha as the `MUL`'s first operand (one row), and three controls the match refuses: the in-place ops (`add`, `rms_norm` and `scale` in place over a are views, which `ggml_can_fuse` refuses), a bias of 0.5 on the last `SCALE`, and width 2,558. The harness allocates every tensor apart, so the allocator's in-place placement, where the result lies exactly over a dead x or a, does not occur in it; that branch runs only in the model's graph, where the log line's in-place count shows it and the KL arms judge it (found in review). alpha is drawn from [0.01, 0.5] (the model's are positive and small), the vectors from [-1, 1]. On both cards (`measured-here`, 2026-11-07): the cases pass by default and with `GGML_OPENCL_FUSE_NGPT=0`; with `GGML_OPENCL_FUSE_NGPT_RED=1` exactly the 6 fused cases fail (ERR 1.93-2.04) and the in-place and bias controls pass, which shows that the fusion ran in each of the 6. The width-2,558 control is reported "not supported" by the OpenCL backend (its `RMS_NORM` declines that width), so it is skipped there, not run as a control. The match is limited to Intel and Adreno, the two sub-group sizes the host assumes (as the backend's rms_norm kernels do). The local memory is sized for the smallest sub-group the kernel could be built at (8 a sub-group), since the kernel counts its sub-groups from `get_max_sub_group_size()`, which follows the required 32 only where the device advertises `cl_intel_required_subgroup_size`. A kernel that does not build logs a warning and the key `DOWNGRADE src=opencl path=fallback reason=not-built what=ngpt_slerp` (0100's scheme), and the seven nodes run as before (both found in review). Compiled on the recipe host with the OpenCL backend (no device there); an offline `ocloc` build of `rms_norm.cl` for `bmg` and `acm-g10` exits 0, but the dump did not name a file after this kernel, so its spill size is unchecked.

## 0115-opencl-few-column-int8-q4_0-q8_0-iq4.patch

Q4_0, Q8_0, IQ4_XS and IQ4_NL weights times 2 to 16 columns (a speculative verify, a few lanes) on int8 DPAS, each weight read once for all columns; and DOWNGRADE keys for the paths below 16 columns that a few-column kernel would beat.

**The defect** (`measured-here`, B60, `test-backend-ops perf -o MUL_MAT`, 4096 x 14336, us a call at 1 / 2 / 4 / 5 / 8 columns): Q4_K 77 / 89 / 89 / 89 / 92, Q5_K 95 / 111 / 111 / 111 / 111, Q6_K 125 / 171 / 171 / 171 / 172; Q4_0 79 / 126 / 139 / 211 / 415; Q8_0 153 / 271 / 495 / 594 / 972; IQ4_XS 92 / 138 / 173 / 979 / 974. An AREX-2 27B mix (15 IQ4_XS, 7 Q4_0 and 4 Q8_0 tensors) verified its MTP 5 drafts (6 columns) in 184 ms against 56 ms for an all-K-quant Q4_K_M of its size, at equal draft acceptance: 16 against 49 t/s decode. No DOWNGRADE line said so: 0100's keys stop at 16 columns.

**Causes, from the dispatch** (`code`):
- The K-quants have two few-column paths: `ggml_cl_mul_mv_kq_q8_1`, the integer-dot matvec, built for 1 to 4 columns only (`NCOLS` 1..4, `kernel_mul_mv_kq_q8_1[4][12]`), and `ggml_cl_mul_mm_kq_few` (0009, fp16 DPAS, lanes = weight rows), which takes type indices 0..2 only: Q4_K, Q5_K, Q6_K.
- **IQ4_XS** (and IQ4_NL) take the matvec at 1-4 columns; from 5 the few-column kernel declines (`ti > 2`) and `ggml_cl_mul_mm_kq_f16` takes the product: the dequantizing tile GEMM, a 64-row x 512-column tile on the B60 (`kq_mm_plain` 4,4,2,8 at sub-group 16; 256 columns on the A770), 64 work-groups for 4,096 rows, its cost the same at 5 columns as at 512. Hence the 5.7x step at 5 and the flat 979 / 974.
- **Q4_0** takes `kernel_mul_mv_q4_0_q8_1` at 1-4 columns (0096) and the 2D / int8 GEMMs from 16 (0099); between them `ggml_cl_mul_mat_q4_0_xmx` returns false and ggml's `kernel_mul_mat_q4_0_f32_1d_16x_flat` runs with the columns in its grid's second dimension: one pass over the weight a column (211 at 5, 415 at 8, ~50 us a column). 0096 had said these 5-15 columns stay on upstream's path, unmeasured.
- **Q8_0** is not in `ggml_cl_kq_type_index`, so neither few-column path sees it; its GEMMs start at 16 columns (`GGML_OPENCL_Q8_XMX`, `GGML_OPENCL_Q8_I8`) and ggml's at 32 (`ne11 < 32` breaks out); everything from 2 columns takes `kernel_mul_mv_q8_0_f32_flat` with the columns in the grid, ~120 us a column.
- `ggml_cl_downgrade_mv` reported the matvec only at 16 or more columns, and the tile GEMM below 16 reported nothing.

**The change.** `kernels/mul_mm_few_i8.cl`, its own program (a compile failure costs this arm only, `DOWNGRADE ... reason=not-built what=fewq`). It is 0009's shape on int8 DPAS: a lane is a weight row, its 32 codes of a block the b operand as exact int8 values (Q8_0's codes as they are, Q4_0's `n - 8` bytewise, the IQ4 types' `kvalues_iq4nl` through 0103's byte-pair table in local memory), the tokens the a operand, 8 a DPAS, two token groups for 9-16 columns; the NK sub-groups of a work-group (4 on Xe2, 8 on Xe-HPG, as 0009) take every NK-th chunk of 256 weights and add their sums in local memory; a Q4_0 / Q8_0 / IQ4_NL K that is not a multiple of 256 ends block by block. The activation is the 1-4-column matvec's (`kernel_quantize_q8_1`): per 32 values two int8 terms, the codes and the codes of what they leave, each with its f32 scale, so a block is two DPAS, `acc += ws (xd0 dot(a0, b) + xd1 dot(a1, b))`; it is quantized once a graph per src1 into `[G][K / 32][2][8 tokens][32 bytes]` (one 256-byte sub-group block read an operand) and shared by the products that read it (`kq_fewq`). A block's dot is at most 32 x 127 x 128: exact in int32 and in f32. The same code builds at sub-group 8 (Xe-HPG) and 16 (Xe2).

Dispatch (`ggml_cl_mul_mm_fewq`, ahead of the Q8_0 GEMMs and the Q4_0 / IQ paths): a 2-D contiguous weight times 2 to 16 contiguous f32 columns (3-D src1 that are columns back to back are flattened first, as before). Q8_0 from 2 columns on both cards (it has no multi-column matvec); Q4_0, IQ4_XS and IQ4_NL from 2 on Xe2 and from 4 on Xe-HPG, where the K-quants' measurements kept the q8_1 matvec at 2 and 3 (0009's rule, unmeasured for these types). Q4_0 and Q8_0 need their SoA planes in VRAM (a KV cache's AoS Q4_0 / Q8_0 and an expert bank decline); IQ4_XS and IQ4_NL take the weight arguments of the other IQ kernels (slot, bank USM or staged copy). K a multiple of 32 (IQ4_XS 256).

Switches: `GGML_OPENCL_FEWQ=0` does not build the program (the old paths, and their new DOWNGRADE keys); `GGML_OPENCL_FEWQ_MIN` (both minima), `GGML_OPENCL_FEWQ_MAX` (16: 15 gives 16 columns back to the GEMMs); `GGML_OPENCL_FEWQ_R` row sets a sub-group (1), `GGML_OPENCL_FEWQ_NK`; `GGML_OPENCL_FEWQ_OPTS` adds defines: `-DFEWQ_RED` leaves every eighth block of K out (the red case), `-DFEWQ_ONE_TERM` drops the residual term (one DPAS a block, Strata's single q8_1 term), `-DFEWQ_IQ4_MUX` decodes IQ4 with 0103's select tree instead of the table. With `GGML_OPENCL_FEWQ_OPTS` set, or `GGML_OPENCL_FEWQ_LOG=1`, the build logs `few-column int8 product (G=g) built with GGML_OPENCL_FEWQ_OPTS=...` and the first product of each type and token group `few-column int8 product: <type> k=<K> x <n> columns (G=, R=, NK=)`: the proof that a run took the path.

**The reference's form, as an arm.** Strata serves these types at 1-8 columns with an integer-dot MMVQ templated on the column count (`code`, `sycl/src/kernels/cuda/native_mmvq.dp.cpp`: `native_q8_0_mmvq`, `native_iq4_xs_mmvq` -> `small_mmvq` / `native_mmvq_multi_kernel<F, NCOLS, ...>`, NCOLS 1..8, a single q8_1 term with an f16 scale; its Q8_0 also pairs two matrices of one input, `native_mmvq_pair`). 0115's default differs from it in the instruction (int8 DPAS, lanes = rows, against dp4a with a row a sub-group), in the activation (two int8 terms against one; `-DFEWQ_ONE_TERM` is the single term), and in the width (16 columns against 8). The reason is this card's K-quant evidence, not an invariant: 0009's DPAS form held 111 us at 2, 4 and 8 columns where the q8_1 matvec went 123 / 169 us at 2 / 4 (Q4_K, B60, `measured-here`), and the matvec needs 256 GRF to keep 2-4 columns in registers (0001, A770). `GGML_OPENCL_KQ_MV_NCOLS=5..8` builds arcint's own q8_1 matvec (`mul_mv_iq_q8_1.cl`) up to that many columns for the IQ types and Q4_0, adds a Q8_0 one (`kernel_mul_mv_q8_0_q8_1`, `-DMV_Q8_0`), and routes these four types at 2 columns up to it there instead: Strata's mechanism on this backend, for the card A/B. Off by default, and then those builds and their text are as before.

**DOWNGRADE keys** (0100's scheme; a decode step's one-column matvec still says nothing):

| Where | Key |
|---|---|
| the per-column matvec at 2-15 columns (`ggml_cl_downgrade_mv`; now also Q4_0's two ggml matvecs) | `op=MUL_MAT type= k= cols=2-4\|5-15 path=matvec-per-col reason=no-few-col` |
| the dequantizing tile or the 2D-block GEMM below 16 columns (`ggml_cl_mul_mm_kq_f16`) | `op=MUL_MAT type= k= cols=5-15 path=gemm reason=no-few-col` |

The columns are a weight's: a 3-D weight's batches are not counted as its columns. Which types can still report (`code`): any type without a few-column kernel on the matvec (Q4_1, Q5_0, Q5_1, Q1_0, MXFP4, Q2_K, Q3_K); the IQ types other than IQ4_XS / IQ4_NL (IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ1_M, Q2_0) at 5-15 columns on the tile, which is the same cliff IQ4_XS had; a K-quant whose few-column kernel declines (an src1 offset that is not 16-byte aligned); and these four types where the new kernel declines.

Deployed units, the coder measured in the 1.6.2 gate (`measured-here`, A770, the unit's flags, RC 9c0c730e against 1.6.1): the path trace shows its Q8_0 shared experts (`ffn_{gate,up}_shexp` at K 2048, `ffn_down_shexp` at K 512, all 40 layers) at 2, 3, 4 and 5 columns "declined by choice (window, size, K)" (480 lines, 40 a weight and width), `/props` reports no downgrade, and the A/B is equal (decode 60.15 / 60.20, 56.65 / 56.60, 53.95 / 53.90 t/s, prefill at 41k 962.35 / 962.70, the texts identical in every run). The agent reports no downgrade and its A/B is equal. What follows was the expectation before that run (`inferred` from `code` and the files' recorded mixes, `docs/llama-engine.md`): the coder's shared experts are Q8_0 (`shq8`, `ffn_{gate,up,down}_shexp`, K 2048 and 512) and its MTP 4 verify is 5 columns, so without 0115's kernel it would report `op=MUL_MAT type=q8_0 k=2048 cols=5-15 path=matvec-per-col reason=no-few-col` and `k=512` (it paid five weight passes for each); with it, those products take the new kernel and nothing is expected (refuted by the run above in its first half: the size rule keeps them off the new kernel, on the old path by choice, and nothing is reported); its other dense tensors are K-quants (0009 from 4 columns on the A770). The agent (Q4_K_M, MTP 5) is expected to report nothing. Flash-Next's dense IQ3_XXS / IQ2_* weights would report the `path=gemm` key if a product reaches 5-15 columns (MTP 3 verifies 4; a prompt's last ubatch or a short continuation can be 5-15): that would be a new key, explained by the IQ cliff above, and is not in `tests/acceptance/expected-downgrades.json`, which this patch leaves as it was.

Tests (`test-backend-ops -o MUL_MAT`): the four types at 2, 3, 5, 6, 8, 12 and 16 columns, M x K 256 x 2560 (10 chunks over 4 sub-groups: uneven), 1001 x 2560, 256 x 4096, 77 x 4096, 512 x 14336, 1001 x 14336; Q4_0, Q8_0 and IQ4_NL also at K 2080 (one tail block) and 96 (no full chunk), M 129; a src1 broadcast over dimension 2 at 2, 4 and 8 columns (flattened to 4, 8, 16); 222 cases, plus the older few-row cases that now reach the kernel (48 x 2560, Q4_0's 40 x 32 ... 17408 x 512). Perf: the four types and Q4_K at 6, 7, 9, 12, 15 and 16 columns at 4096 x 14336 (the shape's loop has 1-5 and 8), and the coder's shared-expert shapes 512 x 2048 and 2048 x 512 in Q8_0 at 1 and 5. Compiled on the recipe host with the OpenCL backend and embedded kernels (no device there); the kernel builds, the red case, the curve and the served check are the card's.

**After the first card window** (`measured-here`, window 167/168, 6222b3f9). Ops: 626/626 on both cards with 8 few-column log lines; `-DFEWQ_RED` fails 412 (B60) and 328 (A770) with the built-with line; `GGML_OPENCL_FEWQ=0` green with 40 `no-few-col` keys; the K-quants 315/315. Perf, 4096 x 14336, us off -> on:

| card | type | 2 | 3 | 4 | 5 | 8 | 9 | 16 |
|---|---|---|---|---|---|---|---|---|
| B60 | Q4_0 | 126 -> 80 | 132 -> 80 | 139 -> 80 | 211 -> 80 | 415 -> 83 | 417 -> 132 | 110 -> 134 |
| B60 | Q8_0 | 276 -> 151 | 379 -> 150 | 496 -> 151 | 596 -> 151 | 963 -> 153 | 1082 -> 212 | 934 -> 214 |
| B60 | IQ4_XS | 139 -> 153 | 156 -> 154 | 173 -> 154 | 990 -> 154 | 974 -> 153 | 975 -> 170 | 986 -> 170 |
| B60 | IQ4_NL | 122 -> 140 | 156 -> 141 | 174 -> 140 | 1012 -> 141 | 1008 -> 142 | 1001 -> 162 | 1011 -> 162 |
| A770 | Q4_0 | 89 -> 89 | 93 -> 93 | 102 -> 115 | 499 -> 116 | 537 -> 117 | 538 -> 154 | 493 -> 156 |
| A770 | Q8_0 | 173 -> 195 | 195 -> 195 | 379 -> 196 | 578 -> 195 | 963 -> 197 | 1089 -> 203 | 505 -> 206 |
| A770 | IQ4_XS | 101 -> 101 | 115 -> 115 | 139 -> 194 | 669 -> 194 | 674 -> 195 | 674 -> 245 | 678 -> 247 |
| A770 | IQ4_NL | 98 -> 98 | 112 -> 112 | 128 -> 185 | 689 -> 185 | 697 -> 186 | 699 -> 232 | 706 -> 235 |

The coder served (A770, deployed flags, MTP 4, 12 greedy answers, 2 rounds): 1.6.1 87.2 / 86.7 t/s, 24.6 / 24.4 ms a verify; 6222b3f9 83.6 / 83.8 t/s, 25.6 / 25.6 ms, acceptance equal (75-76 %), no downgrade key in either. -4 %.

The changes on that record:
- **Column windows per card and type** (`fewq_min_t` / `fewq_max_t`), so that no width is slower than before: B60 Q4_0 2-15 (16 keeps its int8 GEMM), Q8_0 2-16, IQ4_XS and IQ4_NL 3-16; A770 Q4_0, IQ4_XS and IQ4_NL 5-16 (their q8_1 matvec keeps 2-4), Q8_0 3-16 (ggml's matvec keeps 2). `GGML_OPENCL_FEWQ_MIN` / `_MAX` override all four.
- **The coder's loss** (`code`, then `inferred`): its only Q8_0 tensors are the 120 shared-expert products (gate / up 512 x 2048, down 2048 x 512, 1 MiB each; the GGUF's header), and its MTP head has none. `test-backend-ops perf` repeats the op inside one graph (`ggml_graph_add_node(gf, out)`, n_runs times), so the activation prep, cached per graph, ran once there: the per-product times (gate / up 9.45 -> 6.71 us, down 6.81 -> 8.58, A770, 5 columns, `measured-here`) leave it out. Served, a layer pays two prep launches (gate and up share one); the verify lost 1.0 ms over 40 layers, about 14 us a prep. For such a weight ggml's per-column matvec reads from L2. So Q8_0, and Q4_0 from 5 columns, the two types whose other path is that matvec, decline when the weight's bytes times the graph's products that share its prep are below 6 MiB (`GGML_OPENCL_FEWQ_MIN_KB`, 6144), counted by a pre-pass over each graph (`ggml_cl_fewq_prepass`, keyed by the activation's buffer and offset); and when K has fewer chunks of 256 than the NK sub-groups that split it (2048 x 512: 2 chunks for 8). The coder's three products then take 1.6.1's path. A product declined by this choice reports no key (`ggml_cl_fewq_chosen_off`, read by `ggml_cl_downgrade_mv`); `GGML_OPENCL_FEWQ_MIN_KB=0` turns both rules off (the op suite's setting, so every test shape reaches the kernel). The IQ4 types and Q4_0 at 2-4 columns are not declined by size: their other paths pay a prep of their own (the q8_1 matvec) or are the tile GEMM.
- **9 to 16 columns** (`code` and `inferred` from the table): there is no 16-row int8 DPAS (8 rows of a an instruction), so 16 tokens are two DPAS on one b, one pass over the weight. Two passes of 8 would cost about twice the 2-8 figure, which is flat (B60 160 / 302 / 308 us for Q4_0 / Q8_0 / IQ4_XS against 132 / 212 / 170 for one pass; A770 232 / 390 / 388 against 154 / 203 / 245): one pass is cheaper on every type and both cards. The step from 8 to 9 columns is the activation: a work-group reads two int8 terms of 16 tokens for every K (32 bytes a K) against 16 rows' codes (16 bytes a K for Q8_0, 8 for Q4_0), from L1 / L2. Two levers, as arms: `GGML_OPENCL_FEWQ_R2=2`, two row sets a sub-group for the 16-token build only (each activation read serves twice the rows), and `-DFEWQ_ONE_TERM`, half the activation bytes.
- **Review:** the preps step src1's columns by `ne00 * 4`, not `nb[1]` (a permuted `[K, 1, n]` src1 is contiguous with nb1 = K n 4; the K-quants' 0009 prep had the same stride; new cases `m=256, n=1, k=2048, nr=[4|8|12,1], per=[0,2,1,3]` for the four types and Q4_K, which fail on 6222b3f9); under `GGML_OPENCL_KQ_MV_NCOLS` a Q8_0 product the arm's matvec cannot take goes on to the few-column kernel instead of declining; the build's fit check reads the kernels' local memory against the device's (NK 16 x R 4 at sub-group 16 is 64 KiB); `ggml_cl_downgrade_few_gemm` lost a label its callers could not reach. Not covered by the op suite: one activation shared by several products in a graph (test-backend-ops runs single ops), for which the served A/B has a `GGML_OPENCL_KQ_DEDUP=0` arm.
- **Flash-Next** (`code`, the GGUF headers): the served IQ3_S file's 2-D dense weights are IQ4_XS (attention, shared gate / up, the embedding), IQ4_NL (shared down, K 640), Q4_K-Q6_K, one Q8_0 (a shared down, K 640), BF16 (the hyper-connection products, the router), F16 / F32; the MTP head adds Q8_0 attention, shared expert, `eh_proj` and hyper-connection products (K 320 to 10,240). None of IQ3_XXS, IQ2_*, Q2_K, Q3_K, Q4_1, Q5_0 / Q5_1, Q1_0 or MXFP4 is a dense weight there, so the keys the review expected have no tensor to come from, and `tests/acceptance/expected-downgrades.json` stays as it was. The rollback file (IQ3_XXS) has dense IQ3_S (attention, shared gate / up, `ssm_out`) and Q2_0 (shared down), which would report `path=gemm reason=no-few-col` at 5-15 columns.
- `GGML_OPENCL_FEWQ_LOG=2` logs each shape and the path it took once (`few-column int8 product: q8_0 k=2048 m=512 x 5 columns: declined by choice (window, size, K)`), to see what a served model routes where.

**The tuned build on the cards** (`measured-here`, windows 169/170, 20179f79): ops 638/638 on both cards with the size rule on and off, the red case failing (built), `GGML_OPENCL_FEWQ=0` 42 keys, the K-quants 318, the stride cases passing; no width slower than before on either card. The coder (A770, deployed flags) against 807d9a11's code: 86.72 / 86.13 / 86.95 -> 84.96 / 85.33 / 85.33 t/s, 24.73 / 24.22 / 24.74 -> 24.82 / 24.84 / 24.88 ms a verify, the task 10/10 in both, no keys; -1.6 % decode, and its greedy texts moved, so a product still took the kernel. `GGML_OPENCL_FEWQ_LOG=2` printed nothing in arcint's log.

- **Why the coder still took it** (`code`): the size rule counted the products sharing a prep by the activation's buffer and offset. The graph allocator gives each layer's normed activation the same offset, so the 40 layers' gate / up counted as about 80 products sharing one prep, 80 x 1 MiB, above the 6 MiB bar: gate and up took the kernel at 3-5 columns (2 is outside the A770's window), down stayed off on the K rule. The count is now keyed by the activation tensor (`ggml_cl_fewq_key`); a flattened 3-D src1, a stack copy, maps back to the graph's tensor. With it the coder's three Q8_0 products (`code`, the GGUF headers: `ffn_gate_shexp` and `ffn_up_shexp` 512 x 2048, `ffn_down_shexp` 2048 x 512, 40 each; no Q4_0, IQ4_XS or IQ4_NL among its 2-D weights, its experts are 3-D and never reach this path; the MTP head's 2-D weights are Q4_K, Q6_K and F32) all keep the old path at every verify width (2-5 columns under the cost policy), so its greedy texts should be the base's.
- **The A770's windows** already leave Q8_0 at 2 columns and Q4_0 / IQ4 up to 4 on the old paths, where they are as fast.
- **The trace:** arcint routes ggml's info lines to its verbose log (`route_llama_log`), so `GGML_OPENCL_FEWQ_LOG=2` now logs at warn level, and `GGML_OPENCL_FEWQ_TRACE=<file>` has the backend append the same lines to a file at any log level: once a shape, with the weight's name and the path (`q8_0 k=2048 m=512 x 5 columns (blk.0.ffn_gate_shexp.weight): declined by choice (window, size, K)`).

**The dense 27B agent** (`measured-here`, 2026-11-08, the B60, the agent unit's flags with MTP 5, 12 answers an arm, sampled at the unit's temperature 1.0). 0115 takes one product there, the MTP head's `blk.64.nextn.eh_proj.weight` (Q8_0, K 10240) at 2-15 columns; the time per verify is unchanged (56.2-56.6 ms in every arm). A first A/B (two rounds, then a screening round) read the draft acceptance about one point lower than 1.6.1 (46.0-46.6 % against 47.0-47.7 %, decode -1 to -1.5 %) and `GGML_OPENCL_FEWQ=0` back at 47.7 %. The explanation drawn from it, int8 activations on `eh_proj` costing the drafts, was `inferred`, and a build that declines `eh_proj` by name was measured against it: three rounds of each arm, interleaved, gave the decline 46.6 / 45.9 / 47.8 % (48.5 / 47.9 / 49.6 t/s), 0115 as it is 46.9 / 46.8 / 48.1 % (49.1 / 49.0 / 49.7 t/s), and 1.6.1 47.8 / 47.5 / 46.6 % (49.8 / 49.1 / 48.2 t/s), the trace confirming which path each arm took. The arms overlap within a spread of about 1.5 points and 1.6 t/s, so the earlier gap was the sampling's spread: there is no measured effect of 0115 on the agent, the `eh_proj` explanation is retracted, and the decline was not adopted.

## 0116-opencl-q3_K.patch

Dense Q3_K weights on Intel at every column count. Before (`measured-here`, B60, `test-backend-ops perf -o MUL_MAT`, 4096 x 14336), Q3_K was "not supported" at 1 to 512 columns, so a Q3_K tensor ran on the CPU in a served graph; upstream's OpenCL backend has no Q3_K kernel either (`code`). The campaign is `docs/campaigns/q3k-opencl.md` (the reference, the gate, the AREX-2 roles).

**Storage.** Q3_K is kq type index 12 and is kept in planes like 0022's low-bit types (`ggml_cl_lb_fields`): per weight slice `d[nblk] | hmask[32 nblk] | qs[64 nblk] | scales[12 nblk]`, so every field a lane reads is an aligned 32-bit word (the 110-byte GGUF block leaves every other block 2-byte aligned). Reads and writes convert at `set_tensor` / `get_tensor` and the slice copies, as for those types. An odd block count a slice stays in GGUF blocks and is declined by `supports_op`, with a key (below); so is a view of fewer elements than its parent (its kernels would read the parent's planes at the view's block count).

**Paths** (`ggml_cl_mul_mm_kq_few`, `ggml_cl_mul_mv_kq_q8_1`, `ggml_cl_mul_mm_kq_f16`, in that order, as for the K-quants):
- 1 to 4 columns (1 to 3 on Xe-HPG): `kernel_mul_mv_q3_K_q8_1` (`mul_mv_iq_q8_1.cl`, `mv_q3_K`), the integer-dot matvec on the two-term int8 activation of 0001, with Q6_K's lane map: 8 lanes a block, a lane 8 qs bytes and 8 hmask bytes for 32 weights at four scales, `q2 | 0xFC` per byte where the hmask bit is clear (no carry crosses a byte), signed x signed `dp4a`; built for 1 to 4 columns (to 8 under 0115's arm).
- 2 to 16 columns (4 to 16 on Xe-HPG): `kernel_mul_mm_few_q3_K_f16` (`mul_mm_kq_few.cl`), 0009's few-column fp16 DPAS product in Q6_K's form: a lane is a weight row, its 16 codes of a k16 step the b operand as halves `1024 + u` (`u = q2 + 4 h`, the weight `u - 4`), the chain started at `-1028 S`, the 6-bit scale per k16 step; two token groups from 9 columns.
- more than 16 columns, and 5 to 15 where the few-column kernel declines: on Xe2 `kernel_mul_mm_kq2d_q3_K`, the 2D-block GEMM (below, the second step); elsewhere, and where its alignment rule fails, `kernel_mul_mm_q3_K_f16`, the dequantizing tile GEMM (`deq_lb`, `lb_expand` type 12): the weights to f16 in local memory a super-block at a time, fp16 DPAS (the first version, about 2x Q4_K on the B60).
- `GET_ROWS` (a Q3_K token embedding): `kernel_get_rows_q3_K`, a work-group a row, a work-item a 32-weight sub-block, `d (sc - 32)` times the code in f32 as ggml's `dequantize_row_q3_K`.
- `MUL_MAT_ID` stays on the CPU (no kernel; the key below). A follow-up.

**Switches.** `GGML_OPENCL_Q3K_OPTS` adds defines to the three builds that hold Q3_K kernels (the IQ matvec program, the few-column program, the tile GEMM), each logging `Q3_K kernels (<build>) built with GGML_OPENCL_Q3K_OPTS=...`: `-DQ3K_RED` drops the top two bits of every scale (the red case), `-DQ3K_ONE_TERM` runs the matvec on one int8 term (Strata's single q8_1 term, an arm). `GGML_OPENCL_KQ_MV_NCOLS=5..8` (0115's arm) also routes Q3_K at 2 to that many columns to its matvec instead of the few-column product: Strata's `NCOLS` MMVQ on this backend, for the card A/B. `GGML_OPENCL_KQ_FEW=0` and `GGML_OPENCL_KQ_FEW_MIN` act on Q3_K as on the K-quants.

**Difference from the reference** (`code`, Strata `4b9e05a2`): Strata's `native_q3_k_mmvq` (`sycl/src/kernels/cuda/native_mmvq.dp.cpp:436-534, 3570-3628`) is llama.cpp CUDA's `vec_dot_q3_K_q8_1`: GGUF blocks read as 16-bit pairs, 16 threads a block, one q8_1 term with an f16 scale, a per-byte saturating subtract of the hmask's 4, `NCOLS` 1 to 8; wide products dequantize to bf16 (`dequant_bf16.dp.cpp:109-131`) for a GEMM. Here: planes (aligned words), two int8 activation terms (a single-term q8_1 matvec moved the dense 27B's KL by 0.0026-0.0028 nats, `docs/campaigns/llama-engine-kquant-kernels.md`), fp16 DPAS at 2 to 16 columns (0009's K-quant evidence on this card, not an invariant), and a dequantizing tile in place of a separate dequant pass. Both of Strata's choices are arms (`-DQ3K_ONE_TERM`, `GGML_OPENCL_KQ_MV_NCOLS`).

**DOWNGRADE keys** (0100's scheme):

| Where | Key |
|---|---|
| an odd block count a slice (Q3_K), or a view that is not a whole slice of its planar parent (every planar type; `supports_op`, a probe's count) | `op=MUL_MAT type=<t> k= path=cpu reason=odd-blocks\|view` |
| the kernels not built, or a shape they do not take (0100's IQ branch) | `op=MUL_MAT type=q3_K k= path=cpu reason=shape\|kernels-missing` |
| 5 to 15 columns on the tile GEMM because the few-column kernel declined (0115's key) | `op=MUL_MAT type=q3_K k= cols=5-15 path=gemm reason=no-few-col` |
| `GET_ROWS` declined | `op=GET_ROWS type=q3_K k= path=cpu reason=shape\|kernels-missing` |
| Xe2, the tile GEMM because the 2D GEMM's alignment rule fails | `op=MUL_MAT type=q3_K k= cols=16+\|5-15 path=tile reason=2d-shape` |
| `MUL_MAT_ID` | `op=MUL_MAT_ID type=q3_K k= path=cpu reason=kernels-missing` |

A Q3_K weight the loader puts on the CPU after a decline also reports 0111's `src=llama ... reason=device-declines`.

**Tests** (`test-backend-ops`): `MUL_MAT` Q3_K at 1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17 and 512 columns on M x K 256 x 2560 and 1001 x 2560 (10 super-blocks: the K split uneven), 77 x 4096, 129 x 512 (rows ragged), 8 x 1024 (below a sub-group), 512 x 14336, and 129 x 256 (an odd block count: declined); a src1 broadcast over dimension 2 (flattened) and a permuted `[K, 1, n]` src1; the AREX-2 roles (17408 x 5120, 5120 x 17408, 10240 x 5120, 6144 x 5120, 5120 x 6144, 12288 x 5120, 1024 x 5120, the head 248320 x 5120) at 1, 2, 6, 8, 16 and 512 columns. `GET_ROWS` Q3_K at 5120 x 4096 (64 rows), 512 x 5, a 512 x 6 x 3 x 2 batch, a viewed src1, and 256 x 5 (odd: declined). Perf: Q3_K at 6, 7, 9, 12, 15 and 16 columns at 4096 x 14336 (1-5, 8 and 512 come from the shape's loop), and Q3_K and Q4_K at the AREX-2 roles' shapes at 1, 2, 6, 8, 16 and 512.

The index math was checked without a device (`tools/q3k_layout_emul.py`, `measured-here`, an emulation of the kernels' arithmetic in Python): the planes round-trip; the tile GEMM's and `GET_ROWS`' decode equals `dequantize_row_q3_K` on every weight of 24 random blocks; the matvec's lane map and the few-column product's codes against swapped activations give the reference dot product to rounding (relative 1e-14, 3e-13); `--red` (the `-DQ3K_RED` mutation) breaks all three. **The first card run, and the fixes** (window 186, the b6d0e532 build, `measured-here`). The kernels built on both cards. `-o MUL_MAT -p type_a=q3_K` gave 86 OK and 12 not supported, then died with SIGSEGV (rc 139) on both cards, in the case after `m=256, n=8, k=1024, nr=[2,1]`: the permuted `[K, 1, n]` case (`m=256, n=1, k=2048, nr=[4,1], per=[0,2,1,3]`). The kernel log showed a write fault in libc; the full `-o MUL_MAT` on the B60 hit the same case, once with a copy-engine page fault and engine reset. The cause (`measured-here`, the b6d0e532 conversion copied verbatim into a host program under AddressSanitizer, the case's own tensors): test-backend-ops sets every tensor of a case, views too, and the permuted view `[2048, 256]` of the `[2048, 1, 256]` weight was converted with its own ne0 ne1 and nb[2], 256 "slices" of the whole matrix: 57,671,680 bytes touched for a 225,280-byte write, `heap-buffer-overflow` in memcpy. 0022's code had the defect for every planar type; no earlier case wrote a planar view. The copy-engine fault is `inferred` to be the same corruption (not reproduced apart).

- The layout moved to a device-free header, `ggml/src/ggml-opencl/ggml-opencl-planes.h` (`ggml_cl_lb_fields`, `ggml_cl_lb_base`, `ggml_cl_lb_planar`, `ggml_cl_lb_convert`, `ggml_cl_lb_io`). A tensor's bytes are its base's (a view's at its offset), and `set_tensor` / `get_tensor` (device and USM) go through `ggml_cl_lb_io`: the base's slices the range touches are converted whole, and a slice covered in part is read, patched and written back, so any range is exact. arcint's `tests/test_planes.cpp` runs the header against a bounds-checked host buffer: the permuted view, a partial range, a reshape of a 3-D base and an offset view of its rows; its red cases (the base taken as the tensor itself; the read of a partly covered slice skipped) fail it (`measured-here`, the recipe host).
- The view rule in `supports_op`: a Q3_K view must be a whole slice of a planar parent (the same ne0 ne1, offset 0). A `[K, N, E]` parent reshaped to `[K, N E]` had the elements and passed the old count check. New case `test_mul_mat_reshape3d` (declined).
- The few-column build's work-group fit is per kernel: one that misses NK x SG is released alone, with `path=fallback reason=no-fit what=kq_few:<kernel>` (before, all four went, under one key).
- The three programs that hold Q3_K kernels (the IQ matvec, the few-column product, the tile GEMM) are built again without the Q3_K text (`-DGGML_OPENCL_NO_Q3K`) when they fail with it, so a Q3_K compile failure costs Q3_K alone, keyed `path=fallback reason=not-built what=iq_matvec:kernel_mul_mv_q3_K_q8_1,kernel_get_rows_q3_K`, `what=kq_few:kernel_mul_mm_few_q3_K_f16`, `what=kq_xmx:kernel_mul_mm_q3_K_f16`. `GGML_OPENCL_Q3K_OPTS=-DQ3K_BREAK_BUILD` makes the Q3_K text fail to compile (the red case of the fallback: the other types' cases must stay green, Q3_K's become not supported).
- Kernel text: only preprocessor lines changed (the `GGML_OPENCL_NO_Q3K` guards, the `Q3K_BREAK_BUILD` error, an explicit `LB_BYTES` arm for Q3_K, 110 as before); without those defines the compiled kernels are those of b6d0e532.
- Tests: K 768 (three super-blocks a row: the matvec's two-block step ends on a half) at 256 and 1000 rows, 1 to 512 columns.


**The second review** (on aa0720c3). (1) The view rule covers every planar type: 0022's IQ2_XXS, IQ2_XS, IQ2_S, Q2_0 and IQ3_S accepted a contiguous offset-0 view of fewer rows, or a `[K, N, E]` reshape to `[K, N E]`, while their kernels take a slice's block count from the view (`mul_mv_iq_q8_1.cl` `mv_lb`, `nblk = ne01 nbr`) against planes laid per slice of the base; now they are declined with `op=MUL_MAT type=<t> k= path=cpu reason=view`, and `GGML_OPENCL_LB_VIEW_RED=1` accepts them again (the red case). `test_mul_mat_reshape3d` runs for all six planar types, as a reshape and as the first 32 of 64 rows. (2) A range of whole slices converts straight between the caller's bytes and the planes (on USM in place, else through one uninitialized staging buffer and one transfer: the cost before 0116); only a partial range reads, patches and writes back. (3) `tests/test_planes.cpp`: a partial range across slice boundaries, IQ2_XS through the same code, the expected planes built from `ggml-common.h`'s block structs rather than the field table, the direct path calling no transfer; red cases R2 (the partial read skipped: 3 cases fail), R3 (the base taken as the tensor: fails, then aborts) and R4 (Q3_K's d and hmask fields swapped: the struct case fails), `measured-here` on the recipe host.

**The review's lows** (on ca3111ac / a05b036e). A view that is one whole slice of a planar base at a slice-aligned offset (a 2-D slice of a 3-D weight: the base's ne0 ne1, offset a multiple of the slice) is accepted again (`ggml_cl_lb_whole_slice`), since the kernels take their planes at the view's offset and its block count from it, which is that slice's; `test_mul_mat_reshape3d` has the case (slice 2 of 3). The Q3_K 2D branch has the K-quant branch's `ne00 < 2^23` guard. `tests/test_planes.cpp` compares the direct path's planes against the block struct's layout and checks the slice rule. Kernel text unchanged: the card numbers of a05b036e (window 190) apply.

Card record (`measured-here`, the operator's windows): window 189 on aa0720c3 green on both cards, Q3_K `MUL_MAT` 178 OK / 0 FAIL, the full `MUL_MAT` 2,725 OK / 0 FAIL to completion, `GET_ROWS` green; `-DQ3K_RED` 179 FAIL; `-DQ3K_BREAK_BUILD` drops Q3_K alone, keyed per program; `GGML_OPENCL_KQ_FEW=0` and `GGML_OPENCL_KQ_MV_NCOLS=8` green, their reds red; `MUL_MAT_ID` only the B60's 76 MXFP4 failures of the tree without 0116; no engine reset. Window 190 on a05b036e: the dry runs clean (the 2D case without a fallback key, the crash case declined `reason=view`).

**The first card numbers** (window 186, b6d0e532, `measured-here`, the operator's runs). Correctness: the kernels build on both cards, `-DQ3K_RED` fails 86 cases, `GET_ROWS` green; `MUL_MAT` crashed (above). The B60's 76 MXFP4 `MUL_MAT_ID` failures occur on the tree without 0116 too. B60, us a product, Q3_K / Q4_K, 4096 x 14336: 78 / 78 at 1 column, 78-79 / 85-87 at 2-8, 97-99 / 136-140 at 9-16; at 6 columns ffn_down 129 / 158, ffn_gate / up 112 / 122. Prompts on the tile GEMM about 2x Q4_K: 4096 x 14336 at 512 columns 1,816 / 826, ffn_down 2,442 / 1,191. Arms: `GGML_OPENCL_KQ_MV_NCOLS=8` 145-475 us at 2-8 columns (far worse: the few-column kernel stays the default), `-DQ3K_ONE_TERM` about equal (75 us at 1 column; the two-term default stays).

**The prompt path on Xe2** (after those numbers). `kernel_mul_mm_kq2d_q3_K` (`mul_mm_kq_2d.cl`) is Q6_K's 2D-block GEMM on Q3_K's planes: a lane a weight row, its hmask and qs bytes by transposed 32-bit block reads (surfaces of ne01 rows of nb x 32 and nb x 64 bytes), d and the 12 scale bytes per lane; a sub-block's two bits and hmask bit as halves `1024 + u`, `(p - 1028) sc` in halves (`hscale3`), the activations the a operand in the P4 order; Q4_K's tile (64 tokens, 16 sub-groups). It takes Q3_K from 5 columns where the few-column kernel does not, when the hmask and qs planes start 64-byte aligned (a slice's block count a multiple of 32: every AREX-2 role) and the weight is in device memory; otherwise the tile GEMM, keyed `op=MUL_MAT type=q3_K k= cols=16+|5-15 path=tile reason=2d-shape`. In the 2D program, built with the Q3_K-only fallback (`what=kq_2d:kernel_mul_mm_kq2d_q3_K`), its work-group fit its own. Not int8: Q3_K's scales are per 16 weights and the int8 DPAS sums 32 (Q4_K's int8 kernels have one scale per 32); Xe-HPG (the A770, no 2D block reads) keeps the tile, as Q5_K and Q6_K do there. The target is Q4_K's 512-column numbers (826 us at 4096 x 14336 on the B60); the curve is owed. `tools/q3k_layout_emul.py` checks its operand packing (every weight back at its k).

On the recipe host (no device), tree `d3444fc8`: the package recipe applies the series, builds and passes arcint's unit tests (`RECIPE d3444fc: 100% tests passed, 0 tests failed out of 36 EXIT 0`), and `test-backend-ops` with the new cases compiles against the pin plus the series. The OpenCL C kernels are compiled only by a device's driver: their builds, the op suites and the red case ran on both cards in window 189 (above); the 1.6.2 gate's candidate, RC 9c0c730e, gives 186 of 186 on both cards and the red build 0 of 186 (the B60 in the gate window, the A770 the evening before; `measured-here`). The 2D GEMM's curve against Q4_K is not measured and no Q3_K model was run through the server.

## 0117-opencl-q2_K.patch

Dense Q2_K on Intel at every column count, in 0116's structure (the campaign: `docs/campaigns/q2k-opencl.md`). Before, `supports_op` declined Q2_K `MUL_MAT` (`code`) and the tensor ran on the CPU. Strata has no Q2_K kernel (`code`), so the forms are arcint's own: Q4_K's min term, Q3_K's per-16 scales.

- **Storage:** kq type index 13, planes per slice: `dm[nblk] | scales[16 nblk] | qs[64 nblk]` (`ggml_cl_lb_fields`; d and dmin one half pair).
- **1-4 columns:** `kernel_mul_mv_q2_K_q8_1` (`mv_q2_K`): Q3_K's lane map, unsigned x signed `dp4a` on the 2-bit codes, `d s q` against the two int8 activation terms, and the min term `dmin m` times the sum of the lane's activation codes (both terms), the form of ggml's `vec_dot_q2_K_q8_K` with its q8_K block sums.
- **2-16 columns:** `kernel_mul_mm_few_q2_K_f16`: 0009's fp16 DPAS product, codes as halves `1024 + q`, the chain from `-1024 S`, then `xsc (D s t - M m S)` per k16 step (Q4_K's form at Q2_K's per-16 scales).
- **Prompts:** on Xe2 `kernel_mul_mm_kq2d_q2_K`, Q4_K's `(p - 1024) dl + nml` decode (`hscale`) on block reads of the qs plane (a surface of ne01 rows of nb x 64 bytes, its start 64-byte aligned: a slice's block count a multiple of 16); dm and the 16 scale bytes per lane. Otherwise the tile GEMM `kernel_mul_mm_q2_K_f16` (`q2k_v32` in `lb_v32`: `d s q - dmin m` in f32), keyed `op=MUL_MAT type=q2_K k= cols= path=tile reason=2d-shape`.
- **GET_ROWS:** `kernel_get_rows_q2_K`, ggml's order in f32.
- **Keys and rules:** the view rule and the odd-block key as for Q3_K; `MUL_MAT_ID` on the CPU, keyed `kernels-missing`.
- **Switches:** `GGML_OPENCL_Q2K_OPTS` (`-DQ2K_RED`: the scale byte's nibbles swapped, the red case; `-DQ2K_BREAK_BUILD`: the Q2_K text does not compile). 0116's build fallback now drops Q2_K's text, then Q3_K's, then both (`ggml_cl_build_program_q3k_q2k`), keyed `path=fallback reason=not-built what=<program>:<kernels>` per type, in the IQ matvec, few-column, tile and 2D programs.

Tests (`test-backend-ops`): Q2_K at 1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17 and 512 columns on 0116's shapes (129 x 256 odd: declined), K 768, the 2D-aligned 1000 x 2048 and 1008 x 1024 at 17, 100 and 512, the broadcast and permuted src1, the AREX-2 roles (17408 x 5120, 5120 x 17408, 10240 x 5120, 6144 x 5120, 5120 x 6144, 12288 x 5120, 1024 x 5120, 248320 x 5120) at 1, 2, 6, 8, 16 and 512; `test_mul_mat_reshape3d` for Q2_K; `GET_ROWS` as for Q3_K; perf: the curve at 4096 x 14336 and the AREX-2 shapes next to Q3_K and Q4_K. Device-free (`measured-here`, the recipe host and Python): `tools/q2k_layout_emul.py` PASS, `--red` FAIL; `tests/test_planes.cpp` with the Q2_K planes against `block_q2_K`, its red R5 failing. On the cards (`measured-here`, RC 9c0c730e; the 1.6.2 gate window on both cards, the A770's red and break runs the evening before): `-o MUL_MAT -p type_a=q2_K` 193 of 193, `-DQ2K_RED` 0 of 193 (195 FAIL lines on each card), `-DQ2K_BREAK_BUILD` together with `-DQ3K_BREAK_BUILD` over both types' cases 0 cases on the card, every one declined (437 of 437 not supported on the B60), with both types' `not-built` keys (8 on the B60, 6 on the A770) and no failure; a Q2_K break alone, leaving Q3_K built, was not run on a card; Llama 2 7B Q2_K, mean KL to the gate's reference logits, 0.18746-0.18749 on both cards at ubatch 512, 8 and 1 against the CPU's 0.18828 on the same file; llama-bench pp512 / tg128 1952.29 / 77.95 (A770), 3491.61 / 98.54 t/s (B60). The perf curve next to Q3_K and Q4_K is not measured and no Q2_K model was run through the server.


## 0118-dory-loops-per-context.patch

Dory's loop count per context at run time. Before, the count was the GGUF's (`--llama-dory-loops` wrote a kv override at load), so one unit served one count. With this patch the per-layer tables are built for a maximum, and a context chooses any count from 1 to that maximum between decodes (evidence class `code` for the mechanism, `measured-here` for the figures below).

**API** (`include/llama.h`): `llama_set_dory_loops(ctx, n)` (1 to the maximum; 0 is the model's own count; false when the model cannot run that count, the cache is not touched), `llama_get_dory_loops(ctx)`, `llama_model_dory_loops_max(model)` (0 for any other architecture).

**Maximum.** The environment variable `LLAMA_DORY_MAX_LOOPS=N` is read at model load. It builds the executed-layer tables (`physical`, the cache layer map, the scales) for N loops, N from 1 to 4. A value above the file's count needs the `last_loop` cache: with `per_loop` the load warns and the file's count serves, because `per_loop` keeps a cache layer per executed copy and a turn at another count would read layers that earlier turns never wrote. Without the variable the maximum is the file's count and nothing changes.

**Graph.** A turn at fewer loops than the maximum skips the unused recurrent copies. The output blocks keep their index in the table of the maximum, so they keep one cache layer at every count and a cache written at one count is laid out for all of them. The count is part of the graph-reuse key (`llm_graph_params::allow_reuse`), and `graph_reserve` reserves at the maximum, so the compute buffer does not grow when a request asks for more loops than the first one did.

**Switches** (tests only): `LLAMA_DORY_RED_REUSE=1` leaves the count out of the graph-reuse key (the red case: a graph built for one count is reused at another); `LLAMA_DORY_RESERVE_DEFAULT=1` reserves at the file's count.

**Measured, CPU** (`measured-here`, 2026-11-12, the CPU backend, Dory1.0-4B Q5_K_M file, f16 KV cache, context 4,096, micro-batch 512, 8 threads, six made-up fixture conversations, tables for 4 loops, `last_loop` cache; probe: `tools/dory_turn_probe.cpp`, aggregator `tools/dory_turn_probe_agg.py`, fixtures `tools/dory_turn_cases.txt`; the probe is built by hand against a llama.cpp tree with this series applied, not by the arcint build): the 9 load probes bit-equal to loads at a fixed count; 18 of 18 greedy cases gave the same tokens and 2,252 logit rows bit-equal to a model loaded at that count (1, 2 and 4); 4,504 rows bit-equal to a state-transfer arm; weight, cache and compute buffer sizes equal to the unpatched library at 1, 2 and 4 loops (the unpatched library's lines and texts come from an earlier run of a probe without the new API calls, which is not in the repository: the committed probe links only against a patched tree); with `LLAMA_DORY_RED_REUSE=1` 6 of 9 probes fail. This red case is the one that fails; it is the CPU probe's.

**Continuing a cache written at another count** (`measured-here`, same setup, the probe's PAIR output). "Floor" is the same count, batched teacher-forcing against step-by-step decode, which differs by 0.002-0.004 nats and 98.0-98.6 % argmax agreement by itself. "Mixed" is the history cached at count X and the final turn run at Y in the same context, against a model loaded at Y. dKL and dAgree are mixed minus floor; "worst case" is the agreement change of the worst single conversation, in points.

| X->Y | rows | mixed KL | agree % | floor KL | agree % | dKL | dAgree | worst case |
|---|---|---|---|---|---|---|---|---|
| 1->2 | 724 | 0.0074 | 98.3 | 0.0036 | 98.6 | +0.0038 | -0.3 | -1.2 |
| 1->4 | 763 | 0.0083 | 97.1 | 0.0030 | 98.4 | +0.0054 | -1.3 | -5.0 |
| 2->1 | 765 | 0.0074 | 97.0 | 0.0021 | 98.0 | +0.0052 | -1.0 | -2.5 |
| 2->4 | 763 | 0.0044 | 97.8 | 0.0030 | 98.4 | +0.0014 | -0.7 | -3.1 |
| 4->1 | 765 | 0.0077 | 96.5 | 0.0021 | 98.0 | +0.0056 | -1.6 | -2.5 |
| 4->2 | 724 | 0.0045 | 98.6 | 0.0036 | 98.6 | +0.0009 | +0.0 | -0.6 |

Over the floor, across six count transitions of 724-765 rows each on six conversations, the mean KL rises by +0.001 to +0.006 nats and argmax agreement falls by 0 to 1.6 points; the worst single conversation lost 5.0 points (1->4). This is under the repository's answer-level bar in mean KL (0.03 nats) but not in agreement (1 point) for three of the six transitions (1->4 -1.31, 2->1 -1.05, 4->1 -1.57 points unrounded: 741 against 751 of 763 rows, 742 against 750 of 765, 738 against 750 of 765), which is why a caller that changes the count re-prefills (arcint does, `--llama-dory-max-loops`). The other documents point at this table.

**Measured, card** (`measured-here`, 2026-11-12, the 16 GB card (Arc A770), OpenCL, Dory1.0-4B Q5_K_M, the served unit's own arguments: 196,608 cells, q8_0:q4_0 cache, `last_loop`, 2 loops, micro-batch 512; greedy; a dev build of this feature before its last review round, which moved the effort decision ahead of the template render in the request handler, made an empty typed effort read as unset and changed no decode path; five fresh processes on a free card; made-up prompts: a one-word capital question, a bicycle question, a 2,000-token text with a needle, a primes list, an arithmetic question with thinking; served spot check `tools/dory_loops_spot.py`):

1. Three processes were loaded at a fixed count of 1, 2 and 4 loops without the flag (the reference texts) and one at 2 loops with `--llama-dory-max-loops 4`. On the flagged process 20 of 20 greedy answers (5 prompts x effort unset, medium, low, high) equal the text (answer and thinking text) of the process loaded at that count, with equal prompt token counts; Paris and the needle were right in every group. The reference answers differ between any two counts in 2 of the 5 prompts, so 3 of the 5 prompts could not have shown a wrong count.
2. Every response named its count: chat (`dory` object), chat stream (one chunk), Responses (`arcint.dory`), token mode (the final event); `/props` carried `max_loops: 4` and the effort map (`unset: 2`).
3. A conversation whose effort changed turn by turn (medium, medium, high, high, unset, xhigh): each change re-prefilled (`reprefill: true`) and each repeat did not; the server's lane lines show 9 re-prefills in 29 requests, each naming the tokens held and the count they were written at.
4. Card memory of the process (the kernel driver's per-client count): 14,665.9 MiB after the load in all five processes and 15,358.2 to 15,359.2 MiB after the requests in all five; the tables for 4 loops cost 0.0 MiB against a load at 1, 2 or 4. llama.cpp's own lines are equal in all five: weights on the card 2,853.58 MiB, cache 2,808.01 + 8,424.03 MiB, compute buffer 446.02 MiB.
5. Time, single requests, one run each (walls, not rates): the 2,000-token prompt plus its 7 answer tokens took 1.06 / 1.61 / 2.71 s at 1 / 2 / 4 loops; the bicycle answer 88 tokens in 1.54 s, 80 in 2.06 s, 103 in 4.30 s. The flagged process matched the fixed-count processes to 0.01 s in every cell.
6. The library's red switch did not fail on the served route: with `LLAMA_DORY_RED_REUSE=1` (the count left out of the graph-reuse key) 5 of 5 answers of the high group still equalled the 4-loop reference. Reason, evidence class `code`, not measured: `allow_reuse` in `llama-graph.h` compares the micro-batch shape (`n_tokens`, `n_seqs` and the rest) and returns false on a difference before it reaches the count check this patch adds. A count change makes the server re-prefill, the first batch after it has another shape than the last decode batch, so the graph is rebuilt whatever the key says. The red case that fails stays the CPU probe (6 of 9 probes with that switch). The served spot check has no red run of its own; its ability to fail rests on point 1's "2 of 5".

**Not measured:** a deep prompt (the re-prefill cost at depth is the prefill cost; the unit's prefill at 98k-177k tokens was 51 t/s at 2 loops in an earlier run the same day on the same card with the same unit arguments, `measured-here`, so a level change deep in a session costs a full prefill), the B60, KL on the card, any sampled run. No served unit carries the flag. The patch does not touch the kernels.

## 0119-ggml-fa-paged-kv.patch

Paged K/V for `GGML_OP_FLASH_ATTN_EXT`, its CPU reference, and the backend
refusal that keeps an unported backend from reading a pool as if it were the
whole layer (design: `docs/campaigns/kv-streaming-design.md` (on the `qfndev` branch), increment I1;
reference: the fork's `1f6e1fccc`, `ggml/include/ggml.h:2528-2560`). Evidence
class `code` for the mechanism (the fork's checkout at `b540a4b9b`), `code`
for arcint; the cell below is the only number and it is not a rate.

- **API** (`ggml/include/ggml.h`): `ggml_flash_attn_ext_set_pages(a, pages,
  table, page_size, ring_k, ring_v)`: `k`/`v` hold a pool of `page_size`-row
  pages (`[n_embd, n_pool_rows, n_head_kv, 1]`, the cache's row layout), the
  attended sequence is `pages->ne[0]*page_size` rows, `pages` is `I32
  [n_pages, n_tables]` with row `table` used, entry `>= 0` is a pool page and
  entry `< 0` is page `-1 - entry` of the ring (same row layout). It asserts
  the mask's `ne[0]` is at most the attended length -- a narrow mask covers the
  last `mask->ne[0]` keys, the keys before it count as visible (the fork's
  `1ca44251c` form, `ggml_flash_attn_ext_mask_col0`) -- that the pool's and the
  ring's rows are whole pages, and that the ring's types and head counts equal
  the pool's.
  `ggml_flash_attn_ext_set_sync(a, tag, ready, consumed)` carries the
  cross-queue flags (`I32 [1]` tag in device memory, host-visible `ready` and
  `consumed`); `ggml_flash_attn_ext_set_page_runs(a, runs)` carries the table
  as runs of consecutive pool or ring pages, which a backend may split on. The
  last two are plumbing for increments I2/I4: nothing reads them here.
- **Op state** (`ggml/src/ggml-impl.h`): the op params `GGML_FA_PARAM_*`
  (page size 5, ready 6-7, consumed 8-9, page table row 10, page runs 11-12,
  per-row log-sum-exp 13-14, the backend-internal one) and the src slots
  `GGML_FA_SRC_PAGES` 5, `RING_K` 6, `RING_V` 7, `TAG` 8, appended after the
  mask src so the existing kernels' indices do not move; the getters
  `ggml_flash_attn_ext_n_kv` / `mask_col0` / `get_pages` / `get_ready` /
  `get_consumed` / `get_page_runs`.
- **CPU reference** (`ggml/src/ggml-cpu/ops.cpp`, `..._flash_attn_ext_paged`):
  wait for `ready >= tag[0]` if a tag is set (one thread, then a barrier), copy
  each logical row from `table[i/page_size]`'s pool page or ring page into
  contiguous K/V and widen a narrower mask into the attended length (its first
  `col0` columns zero, its mask columns copied at `col0`), run the existing
  contiguous path on those copies, then store `consumed = tag[0]`. A node with a
  narrow mask but no page table takes the same entry point (`mask_col0 > 0`),
  which is how the compact-mask case without a ring exercises the widening. The
  copies are plain memory moves, so the output is byte-equal to the unpaged
  path's on the same bytes. `ggml/src/ggml-cpu/ggml-cpu.c` reserves the copies
  and the widened mask in the graph plan's work buffer.
- **Refusal** (`ggml/src/ggml-opencl/ggml-opencl.cpp`, `supports_op`): a
  `FLASH_ATTN_EXT` with a page table or a sync tag is declined. Its kernels do
  not take pages yet (that is the design's I3), and a backend returning true
  would read the pool as the whole layer. The scheduler then places the node
  on the CPU reference; no backend reads a table it does not understand.
- **Test** (`tests/test_fa_paged.cpp`, arcint's `arcint-fa-paged`, ctest
  `fa-paged`, LABELS unit): four cases over the same bytes, unpaged against
  paged with shuffled pool slots, unused slots filled `0x5a`, a ring for every
  second/third/all page in reverse order, a two-row table with a garbage row 0,
  a causal mask, 1 thread: f16 at 1,024 keys / 6 rows with 4 KV heads x 6
  queries; q8_0:q4_0 at 2,048 keys / 6 rows with a ring every 3 pages; q8_0:q4_0
  at 1,024 keys / 2 rows without a ring; f16 at 512 keys / 1 row, one KV head,
  a ring every page; q8_0:q4_0 at 1,024 keys with a compact mask covering the
  last 768 keys (`col0=256`) and a ring every 2 pages; f16 at 1,024 keys with a
  compact mask covering the last 512 keys (`col0=512`) and no ring. The f32 outputs are compared with `memcmp` -- the gather is
  a copy, so equality is expected, not an NMSE -- and the reference must be
  finite, so a read of the fill shows.
- **Red cases** (each run and seen failing before the patch was restored): the
  gather's `e >= 0 ? pool : ring` becomes `pool`, so a ring page reads a pool
  row -- 3 of 4 cases fail then (the ring arms; the no-ring arm cannot fail on
  it); and the widening's `memcpy(out + col0, ...)` becomes `memcpy(out, ...)`,
  so the visible columns sit at the wrong keys -- the two compact-mask cases
  fail and the four full-mask cases stay green, which is what shows the
  widening runs. Restored, all 6 cases pass.
- **Measured, CPU** (`measured-here`, 2026-11-14, the recipe host, the patched
  tree's own `arcint-fa-paged`, 1 thread): 6 of 6 cases byte-equal (full and
  compact masks), the ring red build 3 of 4 failing, the widening red build 2
  of the 2 compact-mask cases failing. The recipe at the widening's red tree
  (`kv-stream-red-widen`, `b4a8cd98`): 38 with 1 failure, `fa-paged` failing
  exactly those two cases, EXIT 8. The recipe at the green tree: 37 of 37 unit tests
  pass, EXIT 0, the .deb built; at the red tree (`kv-stream-red`, `fcbbfe90`):
  36 of 37, `fa-paged` the one failure, EXIT 8, no .deb -- it compiles and
  fails. Not a rate and not on a card.
- **Not measured:** any card; any speed of the CPU gather; the cross-queue
  flags' paths (`set_sync`, `set_page_runs`) and the OpenCL kernels, which this
  patch deliberately does not touch (I3/I4).

## 0120-llama-kv-stream-map.patch

The adaptive KV stream's page map, host store, eviction/write-back and page
table (design: `docs/campaigns/kv-streaming-design.md` (on the `qfndev` branch), increment I2; reference:
the fork's `src/llama-kv-stream.{h,cpp}` at `b540a4b9b`). Evidence class `code`
for the mechanism; the I1 paged op (`0119`) is what reads the pages. Not wired
into the KV cache yet -- that is I3's prerequisite, named below.

- **What** (`src/llama-kv-stream.{h,cpp}`, added to `src/CMakeLists.txt`): one
  sequence's attention layers keep their K/V in pages of 256 cells. The pool
  tensors are the cache's (`[n_embd_gqa, n_slots*page_size]`); a host store
  buffer of the pool's buffer type holds every page that left the pool, per
  layer K pages then V pages in page order. `prepare(cells, n_kv)`:
  - makes the pages a ubatch writes resident, evicting the oldest non-written
    page when no slot is free and asserting the pool holds at least a ubatch;
  - gives free slots to the newest attended pages;
  - marks the written pages dirty;
  - packs the attended pages that are not resident into the layer's ring slot
    (two slots, alternating by layer parity) and records the ring index, so
    `synced()` reports a ubatch that streams;
  - `assign` reads a returning page back from the host store, `evict` writes a
    dirty page back exactly; `set_input_pages` fills the `I32
    [n_kv/page_size, n_layer]` table (slot >= 0, or `-1 - ring index`), `runs`
    gives the resident attended pages as runs for a gather, `row` maps a cell,
    `layout()` changes whenever a page moves (a graph built around the layout is
    not reused), `get_rows`/`set_rows` read and write wherever the rows live
    (the checkpoint path and the cell's integrity probe), and
    `set_debug_rotate(true)` moves every resident page through the host store
    before every ubatch (the fork's `LLAMA_KV_STREAM_DEBUG=rotate`, which needs
    the pool to hold the context).
- **Deviations from the fork, named**: no copier or publisher thread, no copy
  engine, no `ready`/`consumed` flags, no prefetch -- page moves are synchronous
  `ggml_backend_tensor_set/get` on the owning backend; I3/I5 put them on the
  second in-order OpenCL queue with events and add the kernel-side wait
  (0032/0044's form, 0056's doorbell). Slots are sized by the caller (`n_ring`
  pages), not by a probe of the backend's paged attention or by the streamed
  pages a phase needs -- I3 adds the probe and that sizing. No phase budgets and
  no pool resize (I5). Nothing is wired into `llama_kv_cache` yet: the pool
  views, the graph inputs and the prepare call are I3's, because the CPU arm is
  a cell, not a served model.
- **Test** (`tests/test_kv_stream_map.cpp`, arcint's `arcint-kv-stream-map`,
  ctest `kv-stream-map`, LABELS unit, CPU backend, 1 thread):
  - the map: written pages resident, the newest attended pages take free slots,
    a full pool evicts the oldest non-written page, an attended page reads the
    bytes that were written, rotation writes every page back and reads the same
    bytes, a two-page pool over four pages streams pages 1 and 2 packed at ring
    rows 0 and 256, a written page outside the attended range is resident while
    an unattended one is not, table rows agree;
  - the pages through attention: four ubatches of 256 cells over a 1,024-cell
    f16 cache with a two-page pool and a four-page ring; each ubatch prepares,
    writes its cells through `set_rows`, and runs a 6-row causal paged
    attention over the pool and the ring; its f32 output is `memcmp`-equal to
    the same attention over a contiguous cache holding the same bytes. At the
    fourth ubatch two of four pages stream.
- **The graph's write** (review of 0120): a cell written straight into the pool
  tensor (as `cpy_k`/`cpy_v` write it on the device) is marked dirty only by
  `prepare`; the cell evicts after a second ubatch takes its slot, and the host
  store must return the bytes it wrote. The no-op mutation of
  `for (uint32_t c : cells) dirty[c/page_size] = 1;` fails that cell (4 checks)
  and nothing else -- the section is the only place the graph's write is made. The branch `kv-stream-red4`
  (`487e82b1`) is that no-op: recipe EXIT 8, 1 of 40 failing, `kv-stream-map`
  the one, its four graph-write checks red.
- **Red case** (branch `kv-stream-red2`): `evict` drops the write-back
  (`dirty[page]` -> `false`). It compiles and fails 6 checks -- the rotation
  cell (a dirty page evicted without its write-back reads different bytes) and
  the third and fourth ubatch's attention (an evicted page streams what the host
  store last held). Restored, every check passes.
- **Measured, CPU** (`measured-here`, 2026-11-14, the recipe host, the patch's
  own cell, 1 thread): green all checks, the red build 6 failing. The recipe at
  the green tree (`99d94fe0`): 38 of 38 unit tests, EXIT 0, the .deb built,
  `kv-stream-map` among them and `kv-stream-refuse` reported Not Run (Disabled);
  at the red tree (`kv-stream-red2`, `5a01de69`): 38 with 1 failure,
  `kv-stream-map` failing, EXIT 8, no .deb.
- **Not measured:** any card; the copier/publisher threads, the second queue,
  the sync flags and the pool sizing (I3/I5); a served model (the CPU arm is
  the cell; arcint's llama engine requires an OpenCL card at start).

## 0122-opencl-fa-paged-decode.patch

The B60's GQA decode kernel learns 0119's page table and ring, and the backend
accepts a paged `FLASH_ATTN_EXT` exactly for the shapes it now handles (design:
`docs/campaigns/kv-streaming-design.md` (on the `qfndev` branch), increment I3; reference: the fork's
`1f6e1fccc`/`2a5beae21` page addressing at `b540a4b9b`, `code`).

- **Kernel** (`ggml/src/ggml-opencl/kernels/flash_attn_gqa_dpas.cl`): one
  helper, `fa_gqa_page_row(table, ps, key, pool, pool_nb1, ring, ring_nb1)`:
  `table[key/ps] >= 0` reads row `e*ps + key%ps` of the pool, `e < 0` reads
  `(-1-e)*ps + key%ps` of the ring; a null table reads `key` rows as before. The
  kernel gains 12 trailing args (table, page size, and the two ring slots'
  pointer, offset and nb1/nb2/nb3). Its five K/V row reads go through the
  helper: staging (`GQA_K_STAGED`), the int8 staged rows (0089), the int8 Q K^T's
  direct and staged b operands (0088), and the V staging. **Deviation from the
  fork, named:** the fork makes paging a compile-time `PAGED` switch; here one
  build carries the helper with a `table == 0` runtime check, so an unpaged call
  pays one uniform compare per key row (the rows are 544 B and up for K, so the
  read dominates; the cost is to be measured, not inferred). A separate build
  per switch is not taken because arcint's kernels are built once per type pair.
- **Dispatch** (`ggml/src/ggml-opencl/ggml-opencl.cpp`): `n_kv` of a paged op
  is `pages->ne[0]*page_size` (the attended length), not the pool's rows; a
  paged op routes to the GQA kernel from **1** row (the 0089 staged form for
  1-3 instead of the split kernel, which would attend the pool as the whole
  layer); the 12 args are set for that launch only; a paged op that would reach
  another kernel aborts with a message (`GGML_ABORT`, no silent path).
- **`supports_op`**: true for a paged `FLASH_ATTN_EXT` only when the page table
  is `I32`, contiguous, `[n_pages, n_tables]` with a valid row and page size
  256; K/V is q8_0/q4_0, q8_0/q8_0, q4_0/q4_0 or f16/f16; head size 256, GQA
  `(q heads/kv heads)*rows <= 64`, `q` f32; a full-width mask (the compact mask
  is I5's); the ring rows are whole pages; and the GQA kernel of that pair is
  built. A **sync tag** (`ready`/`consumed`) is still declined (I5's queue);
  every other paged shape is declined. No CPU fallback is relied on: a served
  start must refuse before this point (design §4.3, I5's marker cell).
- **Test** (`0123`): paged cases in `test-backend-ops`, the agent's geometry.
- **Not compiled here**: the recipe host builds the OpenCL C++ (EXIT 0) but has
  no `ocloc`/driver, so the kernel text is compiled only in Claude's card
  window. **Not measured:** any card, any speed, KL. The card commands are in
  the handover and the design's I3.

## 0123-tests-fa-paged-backend.patch

Paged `FLASH_ATTN_EXT` cases in `test-backend-ops`, so the card window runs
0122's kernel over the agent's shapes (design: increment I3).

- **Case** `test_flash_attn_ext_paged`: head 256, `nh` KV heads, `nr` query
  heads a KV head, `kv` keys, `nb` rows, K/V types, `ring_every`, `n_slots`.
  The pool is the KV cache's layout (`[hs*nh, rows]`, all heads of a token in a
  row) with `n_slots` resident pages in shuffled slots and unused pages between
  them; every `ring_every`-th page from page 0 is in the ring, in reverse order
  behind an unused page. The table is row 1 of two; row 0 is garbage. `n_slots`
  smaller than `kv/256` is what makes pages stream.
- **Registered**: at the agent's geometry (24 query heads on 4 KV heads, G 6),
  6-row verify at kv 1,024 / 8,192 / 65,536 / 262,144 with the pool half the
  context (ring every 2), and 1-row decode at the same kv with every page in
  the ring and a 4-page pool; f16 K/V at 8,192; and the all-resident arm
  (`ring_every=0`) at 8,192. 10 cases.
- **Run**: `test-backend-ops test -o FLASH_ATTN_EXT -p 'n_slots=' -b <card>`,
  and `perf` with the same filter (the case's `op_flops` prices a paged run).
- **Measured, CPU** (`measured-here`, 2026-11-14, the recipe host, the patched
  tree's `test-backend-ops`, `-p 'n_slots='`): 10 of 10 OK, including the
  262,144-key arms, against the CPU paged reference. That is the case plumbing
  and the CPU path; it is **not** a card result and says nothing about 0122's
  kernel.
- **Not measured:** the card (Claude's window; the commands are in the
  handover), paged against unpaged time, KL.

## 0121-llama-kv-stream-cache.patch

The KV stream reaches `llama_kv_cache` and the graph, off by default (design:
`docs/campaigns/kv-streaming-design.md` (on the `qfndev` branch), increment I3's prerequisite; reference:
the fork's `143b62090`/`c68455740` cache and graph wiring at `b540a4b9b`,
`code`). Not wired to the copy queue or the phase budgets (I5), and the graph
reuse check is conservative (below).

- **Parameter** (`include/llama.h`, `src/llama-cparams.h`): `kv_resident_size`,
  device bytes of the resident KV pages, default **0 = off**, copied into
  cparams (`llama-context.cpp`). arcint sets it from `--llama-kv-resident`
  (`src/exec/backend_llama.cpp`, I5): a 32-bit byte count, so at most 4095 MiB.
- **Cache** (`llama-kv-cache.cpp`): with the flag on, the cache validates the
  paged kernel's shapes -- one stream (or unified), no V transpose, no shared
  source cache, no SWA, no windows, q8_0 K / q4_0 V, head 256 -- refusing
  otherwise at construction, never falling back. It sizes `n_slots` from the
  bytes (`slots_for`, at least 8 pages), shrinks the K/V tensors to
  `n_slots*256` rows, and constructs `llama_kv_stream` with the pool tensors, a
  ring of `n_pages-n_slots` pages and the pool's buffer type. `get_k`/`get_v`
  return the whole pool's view under streaming (the attention reads it through
  the table). `apply_ubatch` calls `prepare(idxs[0], get_n_kv(sinfo))` before
  the graph, so the written pages are resident before the graph's store and the
  streamed pages are in the ring slots before its attention.
- **Graph** (`llama-graph.cpp/.h`): the KV input carries the page table
  (`I32 [n_kv/256, n_layer]`, a host input filled in `set_input`) and every
  attention layer's ring views; `build_attn` sets them on the fused
  `FLASH_ATTN` node with the layer's table row and expands the inputs.
  **Deviation, named:** a stream-on graph is never reused
  (`llm_graph_input_attn_kv::can_reuse` returns false), because a page move
  changes what the views mean; the fork compares its `layout_id` instead. The
  cost is one graph build per ubatch, to be measured with the stream on.
- **Not in this patch**: the copy queue and the ready/consumed flags (I5), the
  phase budgets and the pool resize, the table's device upload for OpenCL (the
  scheduler places the host input; the OpenCL kernel argument takes the tensor's
  extra, which I5's queue path owns), and any served run.
- **Test** (`tests/test_kv_stream_cache.cpp`, `arcint-kv-stream-cache`, ctest
  `kv-stream-cache`): reads the patched tree's source and checks the wiring's
  shape (default 0, the validation strings, the shrunk rows, the stream's
  construction, `prepare` in `apply_ubatch`, the table and ring views on the FA
  node, the fill in `set_input`, the reuse refusal, the pool views in `get_k`).
  Red: the default set to 1 fails the first check; a normal start would then
  ask for a stream nobody requested.
- **Measured, CPU** (`measured-here`, 2026-11-14, the recipe host): the cell
  passes at the green tree (`5886672e`: 39 of 39 unit tests, EXIT 0, deb built;
  `kv-stream-cache` among them); the patched library builds.
  **Not measured:** a stream-on model run -- no GGUF and no card on the recipe
  host, and arcint's llama engine needs an OpenCL card at start. The device-free
  behavioural evidence stays `kv-stream-map` (green) with its reds
  `kv-stream-red2` (eviction) and `kv-stream-red3` (a one-page table shift).

## 0123z-opencl-fa-paged-prefill.patch

I4: a paged prompt's pages are gathered into f16 blocks and the f16 prompt
kernel runs on them; the decode path's page table is uploaded to the device
(I5's table upload). Design: increment I4; reference: the fork's `9e9e46f86`
chunked paged prefill at `b540a4b9b` (`code`) -- **deviation, named**: the fork
attends page runs as chunks merged by log-sum-exp; here the rows are gathered
by 0037's conversion kernel into its two 32 MiB blocks and the existing f16
kernel runs with its carry, which is the fork's own gather form. The chunk size
is 0037's block (`GGML_OPENCL_FA_DPAS_F16KV_BLOCK`), not the fork's
`GGML_CUDA_PREFILL_CHUNK`; the page-runs merge is left to I5's card round.

- **Conversion kernel** (`flash_attn_dpas.cl`): `flash_attn_kv_to_f16` gains
  eight trailing args (the table, page size, the logical key base `kv0`, and the
  ring's pointer, offset and nb1/nb2/nb3). With a table, logical key `kv0+key`
  reads pool page `e` (entry >= 0) or ring page `-1-e` with the ring's own
  strides; a null table reads the contiguous view as before.
- **Dispatch** (`ggml-opencl.cpp`): a paged op with more than 64 GQA rows (a
  prompt) forces the 0037 gather path; the table row is uploaded once to a
  device buffer on the compute queue and released after the queued conversions
  (a queued command holds it). The 0038 f16 kernel then reads the blocks. The
  decode path's table is uploaded the same way when the page tensor is not
  already a device buffer. Cost: 4 KiB per ubatch at 262,144 cells.
- **`supports_op`**: a paged prefill is accepted when the conversion kernel and
  the f16 prompt kernel are built; the GQA row cap now applies only to the
  decode shape, so a 512-row prompt is not declined (and cannot slip to the CPU
  reference).
- **Cases** (`test-backend-ops`): paged prefill at the agent's geometry, 512
  rows, kv 8,192 and 4,096, q8_0:q4_0 and f16, pools half the context and the
  whole one (3 cases, next to 0123's).
- **Measured, CPU** (`measured-here`, 2026-11-14, the recipe host): the five
  paged cases with `n_slots=16` (the three prefill arms among them) 5 of 5 OK on
  the CPU. **Not compiled or measured on a card**: the OpenCL C++ builds
  (recipe), the kernel text's first build is Claude's window, and the gather
  path's speed is unmeasured.

## 0123zz-opencl-fa-paged-predicate.patch

The paged `FLASH_ATTN_EXT` predicate as tested logic, and the kernel fix that
window 226 exposed: every paged case reported "not supported [OpenCL]" on the
B60, 0 of 10 run, because the GQA program failed its optional build and the
kernel stayed null. Evidence `code` (the source order) + `measured-here`
(window 226: unpaged agent shapes supported and passing on the same tree).

- **The kernel** (`flash_attn_gqa_dpas.cl`): 0122 put `fa_gqa_page_row` between
  `__attribute__((intel_reqd_sub_group_size(16)))` and `kernel void
  flash_attn_gqa_dpas`. An intervening declaration takes the attribute for
  itself, so the kernel compiled without a required sub-group size, the DPAS
  builtins failed, `build_program_from_source_optional` returned null, and
  `supports_op`'s `kernels_decode` was false for every pair. The helper now
  sits before the attribute; nothing is between the attribute and the kernel.
- **The predicate** (`ggml-opencl-fa-paged.h`, new): the supports_op logic as
  plain data -- `ggml_cl_fa_paged_shape`, `_kv_ok`, `_shape_ok`, `_ok`. True
  exactly where the kernels handle a paged case: I32 contiguous table with a
  valid row and page size 256, the four KV pairs, head 256, one stream, GQA
  divisibility; a full-width mask; a whole-page ring (or no ring); and the
  kernel of the shape (GQA for `(q heads/kv heads)*rows <= 64`, the conversion
  + f16 prompt kernel for a prompt). `ggml_opencl_supports_op` fills it from the
  tensors and calls it.
- **Diagnostics**: `GGML_OPENCL_FA_PAGED_DEBUG=1` logs the clauses (table, kv,
  shape, mask, ring, the two kernel flags) and the verdict as one INFO line, so
  a declined paged case names its clause on the card.
  `GGML_OPENCL_FA_PAGED_DUMP=1` logs what a paged launch is handed: `n_kv`,
  page size, the table row and its byte offset, the pool's and the ring's row
  counts and nb1/nb2 for K and V, and the table row's entries read back.
- **Window 232's bindings are right and the failure is inside one type path.**
  The dump shows, for a failing q8_0:q4_0 ring case and the passing f16 ring case
  at the same shape, the same table, the same pool/ring row counts and the
  strides the tensors have (q8_0 K 1088/272, q4_0 V 576/144; f16 2048/512); the
  table row maps every ring page to a small `-1-e` and every resident page to a
  slot within the pool. The row formula therefore indexes inside the pool and
  the ring, and no stride is wrong. What is left is a *type path*: the staged K
  loader for q8_0 (`GQA_K_STAGED`), the q4_0 V staging, or the 0088 int8 b
  operand -- f16 reads the ring through `kb16`/`load16(VT=0)` and passes.
  Isolation runs, one per arm, same line (no kernel change between them):
  - `GGML_OPENCL_FA_GQA_OPTS=-DGQA_K_DIRECT` -- q8_0 K read per lane instead of
    staged: a pass names the staged loader, a NaN keeps it in V/int8.
  - `GGML_OPENCL_FA_GQA_I8=0` explicitly -- excludes 0088's int8 b operand.
  - `GGML_OPENCL_FA_DPAS_F16KV_BLOCK=4096` on the prefill cases -- moves the
    conversion's block boundary; a NaN that moves with the block names the
    conversion, one that stays at the same output index is the row read.
  **The fix is owed after those three lines**; this patch carries no kernel
  change for window 232.
- **Tests**: arcint's `tests/test_fa_paged_predicate.cpp` (`fa-paged-predicate`,
  ctest LABELS unit) drives `ggml_cl_fa_paged_ok` over the agent's decode shape
  (true) and flips every clause (each must refuse), plus prompts and 1-row
  decode; `tests/python/test_fa_gqa_i8.py` asserts the attribute is immediately
  followed by the kernel (nothing between).
- **Window 228's two bugs, fixed:** the KV clause compared enum *literals*
  (`Q4_0` is 2 in ggml.h, not 4), so the agent's q8_0:q4_0 pair was declined
  while f16 passed; the header now names `GGML_TYPE_*`. And the table row: the
  backend passed a device tensor's base and ignored the used row, so the kernel
  read row 0 -- `INT32_MAX` in the paged case -- and a pool row of 2^31 rows went
  out of bounds (the CL_EXEC_STATUS_ERROR and the assertion in window 228). The
  used row is now passed with its byte offset (`ulong table_off`, a device
  tensor's row), or a host row is uploaded; the conversion kernel takes the same
  offset. The debug line reports both: `kv` and the kernel-clause flags.
- **Measured, CPU** (`measured-here`, 2026-11-14, the recipe host): the
  predicate cell passes, the source-text gate passes, the recipe is green
  (`b3d290fe`: 41 of 41, EXIT 0); the red branch `kv-stream-red6` (`7683bf7b`,
  the `kernels_ok` clause dropped) fails 2 checks -- a prompt without its
  kernels and no-kernel-at-all -- recipe EXIT 8. A second, safe red (no kernel
  executes): the KV clause's `GGML_TYPE_Q4_0` written as the literal `4` -- the
  predicate logs `kv 0` and declines the agent's pair. **Not compiled on a card**: the
  kernel's next build is Claude's window; the diagnostic line is the clause
  evidence to collect there.

## 0123zzz-tests-fa-paged-pair-isolation.patch

Window 232's pair isolation, three paged cases next to 0123's (design: increment
I3; no kernel change). The bindings were cleared by the dump: a failing
q8_0:q4_0 ring case and the passing f16 ring case at the same shape get the same
table, pool/ring row counts and the strides those tensors have. So the failure
is inside one type path, and two pair arms name it in one run:

- `q8_0 / q8_0`, 8,192 keys, 6 rows, ring every 2, a 16-page pool;
- `q4_0 / q4_0`, the same shape;
- `q8_0 / q8_0`, 512 rows (prefill), the same shape.

If `q8_0 / q8_0` passes while `q8_0 / q4_0` fails, the defect is the q4_0 V
staging (decode) or V's conversion (prefill); if both quantized pairs fail, it
is the K side; if `q4_0 / q4_0` alone fails, it is the q4_0 K staged loader.

**Measured, CPU** (`measured-here`, 2026-11-14, the recipe host): the three
cases with a ring pass on the CPU (3 of 3), so the case data and table are
well-formed. **Not on a card:** the next window's run.

## 0123zzzz-opencl-fa-paged-ring-aos.patch

The paged kernels read the ring slots as q8_0 / q4_0 blocks, but a ring slot
written through `set_tensor` as a whole tensor (the op suite's ring inputs) is
split into SoA planes by the OpenCL upload. The FA dispatcher now rebuilds a
flattened ring K and V to blocks before the paged decode and the paged prefill
gather read them (`ggml_cl_flash_attn_reconstruct_aos` on
`GGML_FA_SRC_RING_K/V`), as it already did for K and V. The prefill
conversion released its page-table row unconditionally, and a device-resident
table row is not its to release: only an uploaded row is released now (the
op suite's heap corruption, "corrupted double-linked list"). Measured on the
B60 (`measured-here`, 2026-11-13): 15 of 16 paged cases passed with it, every
q8_0 / q4_0 case among them; the 16th, f16 prefill with a ring, is
0123zzzzz's.

## 0123zzzzz-opencl-fa-paged-f16-in-place.patch

A paged f16 prompt took the f16 in-place branch of the prompt kernel
(`flash_attn_dpas2.cl` reading K/V in place). That branch reads the K/V view
row after row over the call's `n_kv` keys, so with paged K/V it attended the
pool's rows as the sequence: it missed the page table and the ring, and when
the table names more pages than the pool holds it read past the pool (the open
case `kv=8192, nb=512, f16, ring_every=2, n_slots=16`: 8,192 keys over a
4,096-row pool). The quantized pairs already took the gather (each key's row
from the pool or the ring into f16 blocks, then the f16 kernel); paged f16 now
takes it too. One condition, `kv_pages == nullptr`, on the in-place branch
(`code`).

Measured on the B60 (`measured-here`, 2026-11-14, window 250): all 18 paged
cases pass on the series through this patch (pin + 8658e830); the tree without
it (03fcccd9) fails exactly the f16 ring prefill (17 of 18). The full
`FLASH_ATTN_EXT` suite on the fixed tree: 3,115 of 3,115.

## 0123zzzzz1-llama-kv-stream-hybrid-layers.patch

The KV stream's ring views were built for every layer with `has_kv(il)`, over
`n_layer_all`. On a hybrid model (the dense 27B, `LLM_ARCH_QWEN35`: gated
delta-net and attention layers) `has_kv` is true for the recurrent layers too,
while the attention cache holds only `il < n_layer() && !is_recr(il)`; the
cache's `map_layer_ids.at()` then threw at context creation
(`llama_init_from_model: failed to initialize the context: unordered_map::at`,
`measured-here`, B60, windows 252 and 253: the agent's command line with
UD-Q5_K_M and `--llama-kv-resident 3584`, with and without MTP). The ring
views are now built only for the layers the cache holds
(`llama_kv_cache::kv_has_layer`); the page table is set on those layers' FA
nodes as before, and the recurrent layers stay in their own memory. Until now
the stream ran only in op tests and unit cases, none of them on a hybrid model.

## 0123zzzzz2-kv-stream-partial-writes-refusal.patch

The first served run of the KV stream beyond its pool (window 255, the dense 27B
as UD-Q5_K_M at 262,144 with `--llama-kv-resident 3584`, B60) served to 131,072
tokens and then ended in a page fault: xe reset the compute and copy engines
(`measured-here`, no wedge). Two defects, both in the stream's page moves
(`code`):

- **The fault.** The OpenCL `set_tensor` splits every contiguous q8_0 / q4_0
  tensor into SoA planes, writing `ggml_nbytes(tensor)` from `data` at `offset`
  whatever `size` is. The stream moves single pages into its pool and ring
  slots with `ggml_backend_tensor_set(slot, page, offset, bytes)`, so its first
  move read past the host page and wrote planes past the slot. A partial write
  of a q8_0 / q4_0 tensor that is not a weight now takes the plain copy (the
  attention reads these as blocks), when the tensor is not a view and was
  never split; every other write keeps the old paths. The cache itself never
  met this: `SET_ROWS` writes it. (A first form asserted on a partial write
  into a split tensor; `test-backend-ops` initializes such tensors in parts
  and stopped on it, window 266, so the old paths stay for them.)
- **The page moves are not per layer.** The stream has two ring slots, a layer
  using `slots[ikv % 2]`, and `prepare()` fills them for every layer before the
  graph runs, so each layer but the last two would read another layer's pages.
  The design copies layer i's pages while layer i-1 computes (I5's copy queue),
  which is not built. The cache now refuses a budget that does not hold the
  context ("the resident budget holds N of M pages; streaming the rest needs the
  per-layer copy (I5)"), so a start that would stream is refused instead of
  serving wrong attention; `LLAMA_KV_STREAM_UNSAFE=1` lets the stream run for
  its development.

Measured on the B60 (`measured-here`, 2026-11-14, window 257, UD-Q5_K_M at
262,144 with a 3,584 MiB budget): the start is refused ("the resident budget
holds 551 of 1024 pages; streaming the rest needs the per-layer copy (I5)").
With `LLAMA_KV_STREAM_UNSAFE=1` the server answers the capital and then ends
inside a 150k prompt with another page fault, this time the copy engine alone
(`engine_class=bcs`): the planar split was one fault source, not the only one.
Streaming beyond the pool stays refused until I5.

## 0125-kv-stream-review-fixes.patch

Applies after 0124, not before: its dispatcher hunks sit beside the wide-verify
block 0124 adds, so the glob order of `patches/*.patch` puts it last of the
series. (0123zzzzz3 and 0123zzzzz5 still apply before 0124.)

Fixes the four defects an outside review of the KV stream found on qfndev, each
with a device-free arm (`code`):

- C, the store indices: `set_input_k_idxs` and `set_input_v_idxs` write
  `llama_kv_stream::row(cell)` once the stream is on, so `SET_ROWS` writes a
  cell's pool row instead of its cell index; beyond the pool that index would
  land past the pool (`k->ne[1]` is the resident rows). The state save and
  restore go through `get_rows`/`set_rows` for the same reason, `clear()`
  calls `stream->clear()`, and a KV window, several streams in an ubatch or a
  transposed V are refused at `apply_ubatch` / the cache's construction rather
  than mapped wrong. The V-transposed branch is unreachable with the stream on
  because the cache refuses `v_trans` at construction.
- D, one paged predicate: `ggml-opencl-fa-paged.h` now carries the shape's
  rows, mask heads, op params (ALiBi, softcap, sinks) and the caps' built
  kernels, and `ggml_cl_fa_paged_form` names the form the dispatcher takes (1
  the GQA decode kernel, 2 the f16-block prefill conversion, 0 none). The
  dispatcher calls the same function before choosing and aborts on a declined
  shape; supports_op calls it too, so the two cannot disagree. The old
  `kernel_fa_dpas_q_dk[0] != nullptr` (always true, an array's address) is
  gone, and the prefill half asks for the conversion and carry kernels it
  actually uses.
- E, the CPU plan's widened mask: `ggml-cpu.c` reserves it whenever
  `ggml_flash_attn_ext_mask_col0(node) > 0`, with or without a page table, so
  a non-paged call with an offset mask no longer runs past the work buffer.
- G, the refusal: a stream-on graph whose fused node is not
  `LLM_FUSED_OP_FLASH_ATTN` throws instead of asserting, and arcint's refusal
  rule's red switch (`ARCINT_KV_STREAM_REFUSE_OFF=1`) is compiled only into
  the unit target (`ARCINT_KV_STREAM_REFUSE_RED`), never into the served
  binary.

Red arms, run on this host (`measured-here`, device-free): the cache cell's
mapping removed (`data[...] = c`) fails its page-map check; the predicate's
`ggml_cl_fa_paged_params_ok` clause dropped accepts ALiBi, softcap and sinks
(three checks fail); the CPU plan's `n_kv` taken from the page table alone
fails the mask-width check. The per-layer red (the copy guarded off) is the
kv-stream-layers cell from 0123zzzzz3.

## 0126-ggml-max-src-paged-attention.patch

`GGML_MAX_SRC` 10 -> 12. `ggml_tensor::src[]` had 10 entries while 0123zzzzz3's
KV stream puts three sources at 9/10/11 (`GGML_FA_SRC_HOST_K/HOST_V/PLAN`,
ggml-impl.h), so `ggml_flash_attn_ext_set_host` wrote past the array into the
tensor's following fields and every FA node read garbage as its plan. Measured
on the B60, window 266 (`measured-here`): the first `FLASH_ATTENTION_EXT` op case
asserted `host_k && host_v && ring_k && ring_v` at ggml-opencl.cpp:23577 before
any OK; the CPU arm passed because its nodes did not reach the OpenCL check.
`ggml-impl.h` now carries a `static_assert(GGML_FA_SRC_PLAN < GGML_MAX_SRC)` for
C and C++. The constant's users (`ggml.c`'s graph loops, `ggml-backend.cpp`'s
scheduling, the RPC structs, the Vulkan/SYCL loops and Vulkan's debug `srci_name`
table) are all sized by the macro and grow with it; `srci_name` lists ten names
and zero-fills the rest, and no reader indexes past the ten in debug output.
Device-free arms: the predicate cell checks every index, and the layers cell
reads the node's sources back (`src[HOST_K/V] == host store`, `src[PLAN] == plan`)
after `set_host` -- at 10 that read is outside the array.

Red arm, run here (`measured-here`, device-free): a translation unit with
`GGML_MAX_SRC` forced to 10 before ggml-impl.h fails to compile
("static assertion failed: GGML_MAX_SRC must hold every paged FA source"); the
layers cell's source read would then read outside `src[]`.

## 0129-kv-stream-usm-clear-and-test-host-store.patch

Two defects the round-3 build hit on the B60, window 269 (`measured-here`).

1. The served start never came up: the stream's USM host buffer is a placeholder
   `cl_mem` of 64 bytes beside host bytes, and `buffer_clear` filled
   `buffer->size` (the pool's MiB) into that placeholder. The device logged
   `Fault response: Unsuccessful -ENOENT` three times, coredumped, and
   `clFinish(queue)` returned -5. `buffer_clear` now `memset`s the USM bytes and
   returns; the placeholder is never filled. The earlier arm passed because
   below the pool the stream's host store is small or untouched by graph fill.

2. The op suite's two-layer paged case aborted on "a paged attention with a host
   store that is not USM host memory": test-backend-ops allocates the host store
   in the backend's ordinary buffer, while the serving stream uses USM. The
   dispatcher now copies a regular host store on the device (`clEnqueueCopyBuffer`
   from the host store's one AOS region, the plan through `clEnqueueReadBuffer`,
   at most 1024 pages) and keeps the USM write path for serving. Window 273 then
   showed the same case aborting on "neither USM nor one AOS region": the op test
   uploads its host store whole, and a whole q8_0/q4_0 write splits it into SoA
   planes (`measured-here`, 2026-11-14, window 273, facet 2). The dispatcher now
   rebuilds a SoA store with the restore kernel into a temporary AOS buffer (the
   same form the ring's reconstruct uses) and copies pages from it; a store whose
   K and V layouts differ aborts with its own message.

The copy geometry is now asserted where the offsets are built (`measured-here`,
reasoned from window 269's fault): the plan's page count equals both host
stores' page counts, a host page's bytes equal the ring's page row span, and the
ring's row stride equals a row of all its heads. A mismatch aborts with the
message instead of faulting the device; the layers cell checks the same three
relations device-free at the reserve shapes.

Red arm, run device-free here (`measured-here`): the layers cell with a plan
row shifted by two pages fails `FAIL layers copy geometry: a page copy runs
outside its ring view or the page bytes differ`; at its real rows it prints
`OK: every attention layer reads its own streamed pages`.

Red arm, device-free here: the `memset` arm and the copy arm are both
reachable only through the OpenCL path, so the red evidence is the card's log
above (the placeholder fill and the dispatcher abort); the guard is that a
non-USM store with a split region (SoA) aborts rather than byte-copying.

## 0124-opencl-fa-wide-verify.patch

## 0123zzzzz3-kv-stream-per-layer-pages.patch

Fixes the second defect of 0123zzzzz2: the stream's page moves are now per
attention layer, so a KV window beyond its pool serves correctly (`code`). The
refusal 0123zzzzz2 added is removed with it; a budget below the context streams
and serves.

- **The stream.** `llama_kv_stream::prepare()` no longer copies any page into
  the ring; it fills the plan `I32 [n_pages]` (the ring row of each streamed
  page, `-1` for a resident one). The host store is now one USM host-memory K
  and V tensor per attention layer, `[n_embd_gqa, page_size, n_pages]`, made
  through `ggml_backend_opencl_set_alloc_host(true)` and accessed by the slice
  entry points (`host_set_page/host_get_page`); the whole class still runs on a
  CPU buffer type for the device-free arm. A device without the USM entry point
  refuses at construction instead of putting the host store on the card.
- **The API** (`0123zzzzz3`, ggml): `ggml_flash_attn_ext_set_host(a, host_k,
  host_v, plan)` adds the stream's tensors as FA srcs 9/10/11. They are leaves
  in the stream's own buffer, expanded in the graph so the scheduler places
  the FA node on their backend.
- **The copies.** Each FA dispatch reads the plan and enqueues, before its own
  kernels, one `clEnqueueWriteBuffer` per streamed page from the USM host store
  into the K/V ring slot at `row*page_size*row_size`, on the in-order compute
  queue. Layer i+2 reuses slot `ikv % 2` only after layer i's commands were
  enqueued, so in-order execution gives every layer its own pages. The CPU
  paged op does the same copy (host memory), so the CPU reference is
  self-consistent and the graph can be tested device-free. This is step one
  of I5: correct, not hidden. The second queue, per-slot events,
  ready/consumed flags and the after-graph prefetch are the overlap, still
  deferred (design §1.5, §4.3).
- **Not built**: the backend probe that sizes the slots (I3's item); bytes over
  the link equal the copies' (each page is read once per attention call), but
  they are not hidden behind compute.

Test: `tests/test_kv_stream_layers.cpp` in arcint's suite
(`arcint-kv-stream-layers`, device-free): three layers share one ring slot, one
pool page and two streamed pages; each layer's output is byte-equal to a
contiguous attention over its own cells and the slot ends with the last
layer's pages. The CPU arm is green; the red arm guards the copy off
(`if (false && plan)`) and fails all three layers' outputs and the slot check.
Not compiled on a card: the OpenCL dispatch's first build is Claude's window.

## 0123zzzzz5-test-backend-ops-paged-layers.patch

`test-backend-ops` gains `test_flash_attn_ext_paged_layers`: two attention
layers in one graph, sharing one ring and one plan (the stream's
`slots[ikv % 2]` are shared across layers), each with its own pool, host store
and table row; the two outputs are joined by an ADD, so one output covers both
layers. Decode (1 row) and a 6-row verify at kv 8,192 with a pool of 16 pages
and every second page in the ring. The CPU reference performs the same
per-layer copy, so a backend that copies every layer up front reads the last
layer's pages and the summed output differs; a correct backend passes.
The previous paged cases used ONE attention layer, which is how the defect
above was missed. Added but not yet run on a card; the CPU-side shapes are
compiled (`code`), the card run is Claude's window.



A verify of 9 to 64 query rows (more than 8 drafts, several lanes' verifies in
one ubatch, a short prompt tail) runs as one call of the GQA decode kernel
(`flash_attn_gqa_dpas.cl`) when its (head, token) rows fit the kernel's 8 row
tiles (64 rows: 10 tokens at the dense 27B's 6 query heads a KV head), instead
of 0028's blocks of 8 rows. K/V is then staged once per work-group for all
rows, where the blocks stage it once per block. The kernel is unchanged.

- **Routing:** the decision is made after the op parameters are read (it needs
  the K/V pair, ALiBi and softcap) and only when the GQA kernel for the pair is
  built. A wide call skips the prompt kernel (`flash_attn_dpas.cl`) and the K/V
  dequantization, and runs with the Intel decode settings of the 4-8-row GQA
  calls (split size, partial buffer). Calls that do not fit keep the blocks;
  4-8-row calls route as before. Paged K/V takes the same call (the kernel reads
  the page table). The A770 builds no GQA kernel (it has 8-wide sub-groups), so
  nothing changes there (`code`).
- **Switches:** `GGML_OPENCL_FA_WIDE_VERIFY=0` keeps the blocks;
  `GGML_OPENCL_FA_DEBUG=1` logs each call's form, rows and tiles (test-backend-ops
  shows it; the server's log does not).
- **Tests:** `test-backend-ops` cases at the agent's geometry, 9, 10, 17, 21 and
  32 tokens, f16 (4,096 and 8,192 keys) and q8_0 / q4_0 (8,192 and 65,536), and
  two paged verifies (10 and 17 rows).

Measured on the B60 (`measured-here`, 2026-11-14):

- **Op suite** (window 242, trees of 38f96336): wide calls take `form=gqa`; every
  q8_0 / q4_0 and paged case passes, wide and switched off. The red copy (a wide
  call launches one tile fewer) fails exactly the wide cases (9 and 10 rows,
  the paged 10; ERR 0.07-0.23); the block cases stay green. f16 at 32,768 keys
  misses the 5e-4 bound (0.0007) on every route, the switch-off blocks included,
  hence the 4,096 / 8,192 arms. Full `FLASH_ATTN_EXT` suite: 3,109 of 3,115, the
  failures those f16 cases and the f16 paged prefill with a ring (0123's open
  case).
- **Served** (window 243, dev binary of f165bf9c at 8 tiles, the agent's
  command line, a 122,880-token prefix, 16 follow-up tails each, arms in
  A B C C B A order): a 10-token tail took 561 / 560 ms against 653 / 652 with
  the switch off (-14 %); a 20-token tail (120 rows, the blocks) 851 ms on both;
  prefill 232.4-232.7 s on every arm. Decode's 4-8-row verify keeps its kernel and
  route (`code`); at 8 tiles it took 144.5-147.5 ms a verify step (window 245).
- **Answers** (window 244, `llama-perplexity`, the agent's Q4_K_M with q8_0 /
  q4_0 KV and FA against the Q8_0 CPU reference, 32 chunks of 512 at ubatch 10,
  so 60-row calls): mean KLD 0.009329 against 0.009322 switched off, same top
  96.752 % against 96.777 % (bar: 0.03 nats, 1 point). The red build gives
  0.888 / 78.63 %, so the arm runs the path.

**16 tiles (128 rows): left out, open.** A 16-tile build passes the op suite
(wide calls to 21 tokens, the red fails them), served a 20-token tail in 747 ms
against 851 (-12 %, window 243), and at ubatch 20 gives the switched-off KL
bit for bit (window 244). Its 4-8-row verify step, which runs the same kernel
built with larger local arrays, took 147.7-148.7 ms against 144.5-147.5 at 8
tiles (window 245, on greedy texts that diverged), so the trade between those
tails and every decode step is not measured yet. It needs the tile switch and
the tile-sized kernel arrays back (wide-verify3 f165bf9c has them).

*Retracted* (2026-11-14, windows 245-249). A first reading called the 16-tile
build defective: one greedy 256-token answer after a 22-token question over
122,880 tokens came back empty in all five 16-tile arms. Window 248 then gave
the same empty answer at 8 tiles, and window 249 measured the position: the
package 1.6.2 itself answers it empty, and 30 samples at temperature 1 end at
once 16 times on the package, 12 on this patch and 11 with it switched off.
The model puts about 40-50 % on ending there, so the greedy pick flips with the
B60's rounding on any build; it is no defect of the kernel or the route.

History. The first form of this patch never ran its kernel. Wide calls reached
the prompt kernel first, and its two red cases mutated dead code. Its perf
figures (32 rows at 64k keys: 44.5 ms against 139) were the prompt kernel in one
call. Served, that one call is slower than the blocks: a 20-23-token tail over
122,880 tokens took 1,178 ms against 845 ms (window 240, package 1.6.2 with
`GGML_OPENCL_FA_ROW_BLOCKS=0`), which confirms 0028's own measurement.
`test-backend-ops` perf is no basis for that comparison with quantized K/V: its
K/V tensors are flattened, and the dispatcher rebuilds them to blocks once per
call, so the blocks pay it once per block. Its 24-tile builds measured the
`flash_attn_f32_f16` fallback (the kernel does not build at 256 GRF), and its
first launches on wide rows crashed on a base that read the page table of an
unpaged call (fixed in kv-ringaos).

## 0127-opencl-replay-subgroup.patch

Fixes 0086/0087's OpenCL replay kernel (`GDN_RPL`, `kernel_gated_delta_net_rpl`): it carried no
`__attribute__((intel_reqd_sub_group_size(SG)))`, while the generation (`tg`) and prompt (`pp`)
kernels do. The replay body is the generation kernel's per-token step with a replay prologue and
a record epilogue, and its `intel_sub_group_block_read/write` and `sub_group_reduce_add` work over
one sub-group of SG work-items with every index `m*SG` (SG 8 on the A770, 16 on the B60). Without
the declaration the compiler's sub-group size is not pinned to SG, and the kernel's indexing holds
only at SG; which size the compiler picked was not read back (`inferred`). The card result below
is the evidence: the replay state is wrong from the first launch without the declaration and
right with it.

**Measured (`measured-here`, 2026-11-14, window 267, the replay-coder tree before this patch,
`test-backend-ops -b GPUOpenCL -o 'GATED_DELTA_NET.*'`, `LLAMA_GDN_REPLAY=1`, one process an
arm)**: 58/66 on the A770 **and** 58/66 on the B60, the same 8 `GATED_DELTA_NET_REPLAY` chain
cases failing on both (4 shapes × exact/precision). The exact arms, `[replay != snapshot: max
|diff| 1.54019 / 1.54458 / 1.47103 / 1.31309]` (A770) and 1.381 / 1.537 / 1.645 / 1.393 (B60);
the precision arms `[CONCAT] ERR 0.10-0.24`. Every non-replay `GATED_DELTA_NET` case passed. So
the op is wrong on the cards and not only on the A770's SG 8; the dense agent's served replay
arms passed the answer-level bar without this (their KL/needle did not see the state), which is
why the defect survived: the chain cases had never run on a card — the plain `-o
GATED_DELTA_NET` matches that name exactly and excludes every `GATED_DELTA_NET_REPLAY` case, and
the earlier card arms used it.

**Red before, green after** (`measured-here`, 2026-11-14, window 270, the tree with 0127 and
0128): 68/68 on the A770 and 68/68 on the B60; the same tree without 0127 (the red) 58/68 on the
A770, the 8 chain cases failing again (max `|diff|` 1.50). The no-replay cases stay green. The
device-free cell is `tests/python/test_gdn_replay_model_branches.py`'s 0127 checks: the replay
kernel declares the sub-group size, as the other two kernels do (red on the pre-patch tree).
0128 adds the coder's chain geometry (16 key heads, 2 v repeats = 32 value heads, K 5) beside
the existing shapes, so the served unit's head counts are in the op suite. Served (window 270):
the coder with `LLAMA_GDN_REPLAY=1` on the A770 answers the capital, prefills 32k at 1031.5 t/s
(1065.9 without replay), decodes at 65k at 34.6-39.7 t/s and scores the acceptance task 10/10;
the dense agent with replay on the B60 answers the capital, no assert.

## 0129h-kv-stream-paged-read-bounds-and-request-end-log.patch

Two follow-ups of window 283 (`measured-here`). The paged FA's copy bounds
(0129f) cover what the copies write, not what the kernel reads: under
`GGML_OPENCL_FA_PAGED_DEBUG=1` the dispatch now reads this layer's table row
(`GGML_FA_PARAM_PAGES_ROW`), prints its largest ring index and slot against
the ring pages in the layer's view and the pool's pages, and aborts on either
out of range. The stream's second line: `clear()` prints the previous
request's last ubatch counts (`KV streaming (request end)`) before it resets,
so a served request that never prints a second prepare line still shows what
it streamed.

## 0129j-kv-stream-paged-layers-table-both-rows.patch

Window 284 (`measured-here`) read `max slot 2147483647 of pool pages 19, max
ring -1` for the 2-layer op case: its `initialize_tensors` filled only one
table row (`table[n_pages() + i]`), so row 0 held the `INT32_MAX` fill and
layer 0's kernel read slot `INT32_MAX` / no ring entry, and the queue failed
at the next synchronize (`-5`). Both rows are now filled (each with ascending
ring entries, 0129f). Red first: the pre-patch case printed that line; the
throwaway arm that only raises the pool pages passes, the real fix removes
the sentinel.

## 0129k-kv-stream-paged-read-sentinel-abort.patch

`INT32_MAX` (an unwritten table row's fill) now aborts under
`GGML_OPENCL_FA_PAGED_DEBUG=1` beside the range checks: window 284's arm
printed a violation and continued, and a check that prints and continues is
not a check. It cannot fire on a correct table; the sentinel is named in the
abort message.

## 0129l-kv-stream-paged-overlap-copy-queue.patch

I5's step two, Intel, opt-in (`GGML_OPENCL_FA_PAGED_OVERLAP=1`): the paged
FA's page moves leave the compute queue for their own in-order queue, so
layer L's copies run while layer L-1's kernels do. Two parities (one per
slot), as FreeToken's prefill double buffer orders its staging copies
(`python/freetoken/moe/offload_cache.py`, `prefetch_prefill_layer` and
`release_prefill_layer`) and Strata's chunk ring its halves
(`src/dsv41/ds4_model.cu`, `copy_ahead` and the flow's waits): a copy into a
slot waits for the FA that last read it (`consumed`, a marker on the compute
queue after the layer's kernels), a copy also waits for a marker of the
compute queue taken when it is enqueued (0032's `gmark` pattern), and the FA
waits for its copy (`ready`). It deviates from the fork: no copier threads
and no per-layer atomics; the host enqueues one layer ahead and the two
queues order the rest. Default off, unset leaves every copy on the compute
queue as 0129zzzzz3 had it. `GGML_OPENCL_FA_PAGED_OVERLAP_NOWAIT=1` is the
development red arm: the copies stay on the copy queue, the FA does not wait
for its ready event, so a race can show as wrong bytes or a failing check; a
red arm that reads right proves nothing. `synchronize` drains the copy queue.

## 0129m-kv-stream-request-end-on-seq-rm.patch

Window 284's served log: seven requests, two prepare lines, no request-end
line. The served path ends a request by removing its sequence from the cache
(`llama_kv_cache::seq_rm`), not by `clear()`. The stream's `request_end()` is
now called there when the removal covers the whole sequence (`p0 < 0 &&
p1 < 0`); `clear()` calls it too, so a cleared cache logs its last counts
once and not twice.

## 0129n-kv-stream-paged-overlap-one-layer-lead.patch

Window 286 (`code` + `measured-here`) shows 0129l's form is serial by
construction: its copies sit inside their own layer's FA dispatch and wait a
marker of the compute queue taken there, so a copy starts only when the
compute queue reaches that layer, and the FA waits for it -- plus a marker, a
barrier and two flushes a layer, +8 ms/step at 261k. `0129n` gives the copies
real lead: `graph_compute` walks the graph's paged FA nodes once (dispatch
order, `op == FLASH_ATTN_EXT` with `src[PLAN]`); at the first paged FA, node 0's
copies wait a marker of the compute queue (after the input uploads, per 0042's
measured placement) and node 1's follow immediately; every other node's pages
are enqueued at the END of the previous node's dispatch, right after its
kernels. A copy waits only its parity's `consumed` event -- the FA two
attention layers back. Nothing waits the compute queue besides node 0's
marker: the streamed pages' source (the USM host store) and the plan are
written by the host before the graph and nothing on the device writes them
during it (this ubatch's new K/V land in the pool through `set_rows`; the ring
receives bytes only from these copies). The same switch, default off; NOWAIT
still drops only the FA's ready wait, so a slot is read mid-copy.
`0129n` also coalesces: consecutive plan pages with consecutive ring rows go
as ONE command a run (C7); under DEBUG the dispatch prints commands before and
after coalescing and the bytes. The plan-text byte count is `2 x pages` before.

## 0129p-kv-stream-request-end-api.patch

Window 286 (`measured-here`): zero request-end lines in four served arms. A
served request is not ended by `seq_rm` -- the cache keeps the prefix for reuse
and the removal comes at the NEXT request on the divergent tail. The line now
sits where a completion ends: `llama_memory_request_end(mem)` (llama.h;
`llama_kv_cache::request_end` calls the stream's), called from arcint's
`generate` and `generate_spec_lane` before their return, so one line per
completion with the stream's counts since its last line.

## 0129q-kv-stream-paged-overlap-slot-and-no-fallback.patch

Five corrections to `0129n`, all from reading it (`code`):
- No silent fallback (CLAUDE.md). `ggml_cl_fa_paged_overlap_issue` returns -1 when the ring is
  not one region or the host store is not USM; each such node now WARNs once a process
  ("node %d of %d fell back to the compute queue"), and `overlap_after` prints once a process,
  at the last node of the first graph with streamed pages, "%d of %d nodes issued ahead,
  %d fell back". `issue` prints its own DEBUG line tagged "(lead)". The dispatch's DEBUG
  "bytes" field prints bytes now, not the command count.
- `issue`'s guard read `o.issued[ni]` out of range: `before()` asks for node 1 on a one-node
  chain. It now returns false for `ni >= size` before touching the vector.
- The parity is keyed by the RING SLOT, not by the node's place: `overlap_slot` reads the
  node's K ring's buffer and `view_offs` (`ggml_cl_regions == 1`) and matches it against two
  remembered slot keys. The stream's slots are `slots[ikv % 2]` (llama-kv-stream.h, 0120);
  the node's index in the chain happens to match today, but the MTP head's graph or a graph
  that skips an attention layer need not keep that order. A third distinct slot aborts under
  DEBUG with both keys; without DEBUG that node falls back.
- Node 0's copies still wait a compute-queue marker taken at the first paged FA, so they wait
  the first GDN layers as well as the input uploads. Kept: the uploads are the only compute-
  queue write the copies must follow (0042), and moving the marker to `graph_compute` would
  not shorten it. Price ~1/16 of the copy time (`inferred`, about 17 ms a step at 261k).
- The op suite does not reach this path: its host store is a plain buffer (not USM) and its
  2-layer ring is SoA (`nrk = 2`), so coalescing and the lead both take the fallback. An op
  case that reaches `ggml_cl_fa_paged_enqueue_pages` needs USM allocations and a single-region
  q8_0/q4_0 ring; not built yet, so the served NOWAIT arm is the only red evidence.

## 0129r-kv-stream-request-end-counts.patch

Window 288 (`code` + `measured-here`): zero request-end lines with 0129p in every served
arm, though `generate_spec_lane` calls it (backend_llama.cpp:2605, the MTP-5 path). From the
code: `llama_kv_stream::request_end()` returned when `logged_first` was false, but
`logged_first` stays true after the first line a stream ever printed, and `prepare()` prints
only when `!logged_first || (n_streamed > 0 && !logged_stream)` -- a later request that
streams after one already logged prints nothing and never clears the flag. Warmup and a
short probe log; the 65k/261k requests do not. The counts were also only assigned inside
that print gate. Now `prepare()` keeps `att_last`/`streamed_last` every ubatch, and
`request_end()` prints unconditionally (one line per completion) and resets both flags.

## 0129s-kv-stream-paged-verify-and-overlap-summary.patch

- `GGML_OPENCL_FA_PAGED_VERIFY=1`: exact integrity of what each paged FA reads, on the card.
  `ggml_cl_fa_paged_verify` builds a small FNV kernel (`fa_paged_verify`) and enqueues it on
  the compute queue at the FA's position: after the lead's ready barrier (so after its
  copies) and, in the unset/sync arms, after the copies just enqueued on that same queue;
  the FA's own kernels are enqueued after this function returns. It hashes EXACTLY the ring
  rows the plan names as streamed for this node, at the slot's own base offset and row
  stride, in 4096-byte chunks, K then V. The host hashes the same bytes of the plan's pages
  in the USM host store, which `prepare()` wrote before the graph. The ring row stride
  equals the host page stride (K 1088*256 = 278528 = `kb`, V 576*256 = 147456 = `vb`), so
  the compared bytes are the attended bytes; the check reads the slot after the copies in
  the same queue position, so it observes the state FA j sees. What it does not prove is the
  kernel's own addressing (it validates the copies, not the paged kernel's row math).
  Per process the first mismatch WARNs with node, host page, ring row and chunk; totals are
  in the summary line. A mismatch is a WARN, not an abort, and `NOWAIT` is the red arm: a
  slot read mid-copy must show chunk hashes differ. If it stays clean, the copies are far
  enough ahead (or the sync drain covers them) and the check is not reaching the race --
  say so rather than claim it; the arm's log then does.
- The overlap summary line: 0129q printed it from the LAST chain node's dispatch tail, which
  needs that dispatch to take the tail and its `ov` to hold; window 288's zero count says no
  dispatch satisfied both, while the DEBUG lead lines prove the 16 issues. It is printed at
  the next graph's start now -- `ggml_backend_opencl_graph_compute` always runs -- with the
  previous graph's "issued ahead / fell back" and the VERIFY totals.

## 0129t-kv-stream-paged-verify-plan-index-and-selftest.patch

Read-back of 0129s (`code`): `ggml_cl_fa_kv_stream_pages`' copy loop takes ring row `pl[p]`
from `hk + p*kb`, `p` the PLAN index; the verify compacted the streamed rows alone and then
read `host + i*nb` with `i` the compacted index, so the host side compared page i of the
store against the i-th streamed row -- a false red whenever the streamed pages are not the
front of the plan. From the code, the served layout does put them at the front while the
pool is short of the attended pages: `prepare()` assigns the newest attended pages first
("free slots take the newest attended pages", llama-kv-stream.cpp:224-228) and the streamed
pages are that oldest prefix, so 0129s would have passed on a fresh 261k decode -- and
failed as soon as evictions (the victim is the oldest non-written page) break the prefix.
The plan index now travels with the ring row (`pages` beside `rows`), the host hash reads
`host + pages[i]*nb`, and the first-mismatch WARN prints that plan index as "host page".
`GGML_OPENCL_FA_PAGED_VERIFY=2` is a green self-test: on the first checked node one byte of
the host-side expected value (the comparison's copy, never the store) is flipped, so exactly
one chunk must mismatch -- proof the comparison can fail without a race. `=1` unchanged.

## 0129u-kv-stream-paged-next-graph-head-lead.patch

Window 289b (`measured-here`) + `0129n`'s code: node 0's copies are the exposed 28.0 ms gap
before the first paged FA of every graph. From the code, node 0's copy waits a compute-queue
marker taken at that dispatch, so it starts only after every kernel before FA 0 (the graph's
stems and the first GDN layers), and node 1's copies follow it on one in-order copy queue;
FA 0 waits only `ready[slot 0]`, but 28 ms > one layer's ~11 ms is the marker's start delay
plus the copy, not node 1's wait. `0129u` removes it from the graph: at the LAST paged FA of
a graph, `ggml_cl_fa_paged_prefetch_head` copies the NEXT graph's node 0 pages into their
slot, waiting the `consumed` event of the FA that last read that slot (two attention layers
back), so they run beside this graph's last FA, its last layer, the output head and the
host's sampling. At the next graph's `before(0)`, `ggml_cl_fa_paged_prefetch_same` compares
this graph's node 0 plan rows with the rows those copies filled; equal rows reuse the copy
(FA 0 waits its ready event and copies nothing), a difference re-copies (counted). Valid
because streamed pages are immutable in the host store: `prepare()`
(`src/llama-kv-stream.cpp:172-259`) writes the store only through `zero_host`/`host_set_page`
for pages it saves, and this ubatch's new K/V go to resident cache cells
(`src/llama-kv-cache.cpp:1730,1765`), never to streamed pages. First graph: no pre-copy, the
old path. After a `seq_rm` or a pool change: the next `prepare()` reassigns ring rows and the
comparison re-copies. No invariant conflicts: the pool partition and slots are unchanged,
nothing is evicted for the prefetch. DEBUG prints per graph "node 0 reused N pages" or
"rows differ, re-copied", with the running issued/reused/re-copied counts.

## 0129v-kv-stream-paged-prefetch-pairs-and-chain-key.patch

Read-back of 0129u (`code`): `prefetch_same` compared the sequence of ring rows only.
`prepare()` numbers ring rows in plan order (`src/llama-kv-stream.cpp:225-230`: streamed pages
get `n_streamed++` as the plan walks), so two plans with the same number of streamed pages but
a different set both give `[0..n-1]`, and node 0 would reuse rows holding the previous plan's
pages. The prefetch now records `(plan page, ring row)` pairs and the issuing graph's chain
length and first-node slot; reuse needs all three identical. From the code, a same-count set
change is reachable in a decode: an eviction swaps the oldest non-written page for a slot
(`llama-kv-stream.cpp:207-216`), a request's prefix reuse moves which pages are resident, and
the MTP head's graph is its own chain whose tail can leave its own node 0 in the slot. The
DEBUG line now carries both chain lengths ("chain %d, prefetched by chain %d").

## 0129w-kv-stream-request-summary-and-hybrid-request-end.patch

Window 290 (`measured-here` + `code`): the VERIFY=2 self-test fired correctly (1 of 312 chunk
hashes differed, the flipped byte), but three lines stayed silent.
- The self-test line tested `o.v_nodes == 0` after `o.v_nodes++`, so it could never print;
  the flag is taken before the increment.
- The once-per-process summary printed at the first graph with the counters still zero ("0 of
  16 nodes issued ahead"). The graph's counters now roll into a process total, and
  `ggml_backend_opencl_fa_paged_summary` (weak-called from `llama_kv_stream::request_end`)
  prints the REQUEST's totals at the request's end -- issued ahead, fell back, verify nodes /
  bytes / mismatches, next-graph reused / re-copied -- and resets them. The fallback WARN
  stays once per process.
- The request-end line never reached the stream: 0129p overrides `request_end()` only on
  `llama_kv_cache`, and the served hybrid model's memory is `llama_memory_hybrid`
  (`src/llama-model.cpp:3317-3324`, the QWEN3NEXT/QWEN35/QWEN35MOE branch without SWA or an
  indexer), which kept the `llama_memory_i` no-op. `llama_memory_hybrid` now forwards to
  `mem_attn`; `llama_memory_hybrid_iswa` does the same, and `llama_kv_cache_iswa` (which had
  no override) forwards to its base and SWA caches. `llama_memory_hybrid_idx` inherits the
  hybrid override.

## 0130a-fa-decode-depth-op-cases.patch

The red half of 0130b (`code`): the 4-7 row decode kernel's depths with quantized K/V had no
op case. 0016's depth arms stop at 131,072 and are f16, and the 0088 arms reach 131,072 at 1-3
rows only, so the served decode (q8_0 K / q4_0 V, head 256, 24 query heads on 4, 4-7 rows) was
tested at 4,096 and 1,013 but never at 65k or 240k. Added: those types at 4, 5, 6 and 7 rows,
4,096 / 65,536 / 245,760 keys, eval and perf. Window 291's unitrace says a call at 240k reads
~938 keys a split and takes 8.34 ms in this form, so this is the case the switch is red against.

## 0130b-fa-decode-v2-double-buffered-v-stage.patch

`GGML_OPENCL_FA_DECODE_V2=1` (default off), 4-8 row form of `flash_attn_gqa_dpas.cl`
(`code`): the kernel staged each tile's V between its Q K^T and its P V, so the global loads
sat on the critical path and the DPAS units were idle while they ran -- window 291 measures
~48 GB/s at 240k, and the 2026-10-04 ablations showed staging and products adding up rather
than overlapping. Now two V stages (16 KB each at head 256, BK 32 -> `local` 42 KB becomes
58 KB a work-group, still one work-group a core per 0045, 64 KB device limit), the tile t+1
stage runs at the loop's top before t's Q K^T, P V reads the buffer of this tile, and the
consumed buffer toggles at the tile's end. Barrier count is unchanged (one WAR fence a tile
at its top, one before P V). `GGML_OPENCL_FA_DECODE_V2_BROKEN=1` is the red variant: the stage
targets the buffer the products read, so every arm must fail. The base kernels stay built and
are replaced only when a V2 programme builds and fits local memory; the log line carries the
built count and the local bytes. The 1-3 row q8q4 split kernel is untouched.

## 0130c-kv-stream-paged-lead-reachability-and-request-totals.patch

Window 292 (`measured-here`) + `code`. This patch carries 0129x, 0129y and 0129z folded
(numbers kept in its comments):
- **0129x, the dead lead's cause.** `ggml_cl_fa_paged_overlap_after` -- which issues the next
  layer's copies one layer ahead AND, at a graph's last paged FA, calls
  `ggml_cl_fa_paged_prefetch_head` (0129u's cross-step lead) -- sat only in the tail of
  `ggml_cl_flash_attn` (`ggml-opencl.cpp:26102`). The split-decode branch, which the served
  decode takes (the GQA kernel runs through `fd_k_split` with `f32_merge`), `return`s at
  :25979 (`code`), before that tail. So on a decode neither the one-layer lead nor the tail
  prefetch ever ran: window 292's DEBUG arm printed 2,240 "commands (lead)" lines and not one
  "next-graph lead" line, every node's copies were issued by its own dispatch (`before`'s
  fallback), and every request-end summary read "0 reused, 0 re-copied". The branch now calls
  the tail too, after both kernels are enqueued.
- **0129y, no silent fallback.** Each early return of `prefetch_head` (empty chain, ring slot
  not cached, plan not USM, `enqueue_pages` -1, and a plan with no streamed page) names itself
  once a process, and the request-end summary now reads "next-graph lead: N issued, N reused,
  N re-copied", so "the tail never prefetched" is distinguishable from "it prefetched and the
  next graph rejected it".
- **0129z, the verify-only arm's totals.** The counters rolled up only under `OVERLAP`, so
  window 292's A-uverify arm (VERIFY=1, OVERLAP unset) printed no totals and could not show it
  had checked anything. The roll-up and the zeroing now run when either switch is on; the
  chain (0129n's lead) stays overlap-only. 0129s/0129t's verification itself is unchanged.

## 0130d-kv-stream-paged-verify-post-fa.patch

`code` for the mechanism; the card number is owed (no GPU in this session). The gate is
device-free: `tests/python/test_fa_paged_lead.py`'s `FaPagedPostFA`, three mutations run and
seen failing.
- **VERIFY=3, the post-FA hash.** 0129s/0129t's `fa_paged_verify` hashes the streamed rows
  the FA reads, but at the FA's start and with a blocking read: it proves what FA j reads
  when it starts, not what the slot holds while j runs. `=3` enqueues the same hash at that
  start (pre), then the SAME rows again right after FA j's kernels (post), both on the
  compute queue, both read back `CL_FALSE` with an event. On the split-decode branch the
  post sits after `k_merge` and before `ggml_cl_fa_paged_overlap_after`'s consumed marker;
  on the prompt tail after the tail kernels and before the same call. The pairs are awaited
  at the next graph's start and at the request's end; a chunk whose two hashes differ means
  a copy landed while the FA read the slot. WARN once a process:
  `paged FA post-FA changed: node %d page %d chunk %d`, and the request summary gains
  `post-FA changes N` (`code`).
- **The red arm.** `GGML_OPENCL_FA_PAGED_NOCONSUME=1`, development only, drops exactly the
  consumed wait in `ggml_cl_fa_paged_overlap_issue` -- the 0129n-0129w decode, where
  `consumed[]` stayed nullptr, so a copy into a slot could start while the FA two layers
  back read it. The prefetch's own consumed wait is untouched (`code`).
- **Why the post is before the marker.** The consumed marker waits the queue's prior work,
  the post hash included; a copy that waits the marker cannot land inside the FA it guards.
  With the wait dropped the only ordering left is the copy queue's own, which is the race
  `=3` is built to see (`code`).

## 0130e-fa-decode-perf-depth-cases.patch

`code` for the mechanism; the timed number is owed (no GPU in this session). 0130a added the
4-7 row decode kernel's depth cases to the TEST list only; `make_test_cases_perf()` builds
its own list, so window 293b's perf arms matched nothing (`test-backend-ops perf -o
FLASH_ATTN_EXT -p '<depth regex>'` ran 0 cases). This patch puts the same 3 K/V pairs x
{4,5,6,7} rows x {4,096, 65,536, 245,760} keys into the perf list, so a red-first run of the
filter prints `us/run` for the kernel's own served shapes.

## 0130f-fa-decode-v-register-prefetch.patch

`code` for the mechanism; the card number is owed (no GPU in this session). 0130b's V2
double buffer stores the loaded halves to local memory immediately after the loads, so the
thread stalls at the store before tile t's products run; the second buffer hides nothing
within a sub-group. 0130f loads tile t+1's V halves into the thread's private registers at
the loop's top, runs tile t's QK^T, softmax and P V, and stores the registers to the other
stage only after P V, so the load's value is consumed after the products. Two stages and
the same local footprint as V2 (58,224 B at head 256 q8_0 K/q4_0 V); the build option set is
identical, one switch apart: `GGML_OPENCL_FA_DECODE_PRE=1`, default off, replacing a pair's
kernel only after it built and fits local memory, with a WARN and the base kernel when it
does not. `GGML_OPENCL_FA_DECODE_PRE_BROKEN=1` is a separate build whose store goes to the
buffer P V reads, before the products: the red op cases. The gate is device-free,
`tests/python/test_fa_decode_v2.py`'s `FaDecodePre`, three mutations run and seen failing.

## 0130g-fa-paged-post-selftest.patch

`code` for the mechanism; the card number is owed (no GPU in this session). Window 294's
NOCONSUME arms printed `post-FA changes 0` on every request-end line at both depths, so
VERIFY=3's collect and compare path had never been seen to fire. `GGML_OPENCL_FA_PAGED_VERIFY_SELFTEST=3`
writes ONE byte (pattern 1 byte, offset = the first streamed row's chunk 0) into the K slot on
the compute queue inside `verify3_post`, after the FA's kernels and before the post hash, on
one node a request (the first streamed node of the first decode graph); the post hash must
differ from the pre hash in exactly one chunk, so exactly one
`paged FA post-FA changed: node %d page %d chunk %d` WARN and `post-FA changes 1` in the
summary. One K byte of one streamed key is corrupted in the ring slot until the next copy
refills it: the answer on that node's decode chain may change, so this arm is not for serving.
Gate: `tests/python/test_fa_paged_lead.py`'s `FaPagedPostSelftest`, three mutations run and
seen failing.

## 0130h-fa-paged-forced-race-delay.patch

`code` for the mechanism; the card number is owed (no GPU in this session).
`GGML_OPENCL_FA_PAGED_DELAY=<n>`, development only: a busy-wait of n iterations in its own
one-work-item kernel on the compute queue, enqueued BETWEEN this FA's split kernels and its
`k_merge`, so it lengthens FA j's hold of its slot and adds nothing before FA j starts. With
`NOCONSUME=1` the copy into that slot two layers later no longer waits the marker and lands
inside FA j (nonzero `post-FA changes`); with the consumed wait on, the same delay reads
`post-FA changes 0` -- the measured proof of 0130c. The cross-step prefetch's consumed wait is
not part of NOCONSUME (0130d's exactly-one-wait arm): it stays unconditional, which is why
window 294's red arm could not land. Gate: `FaPagedForcedRace`, three mutations run and seen
failing.

## 0130k-fa-gqa-no-private-memory.patch

`code` for the mechanism; the ISA is the card host's (ocl 26.27, `-device bmg`, `-DDK=256
-DGQA_SIDES=2 -DKT=1 -DVT=2`): base/V2/PRE all reported `//.private memory size 5888`, 61
private loads and 8 private stores inside the dpas span. `GGML_OPENCL_FA_GQA_NOPRIV=1`, default
off, compiles the same source with `FA_UNROLL` (`_Pragma("unroll")`) on every loop that indexes
a private array (8, NDH, KC, NKB, PRE's 2: all compile-time) and the V stage's `vv[64]`
replaced by 64 component stores straight to local memory, in both stage macros; the reduction
order is base's, no reassociation. BROKEN is a separate build that writes value 15 of every
staged V row as zero, the red op cases. Expected: `//.private memory size 0`. Gate:
`tests/python/test_fa_decode_v2.py`'s `FaDecodeNoPriv`, three mutations run and seen failing.

## 0130l-fa-gqa-no-private-file-level-directives.patch

`code` and `measured-here` (recipe host, cpp 14.2.0): 0130k broke compilation of the
production GQA decode kernel, NOPRIV off. It put `#ifdef GQA_NOPRIV` / `#else` / `#endif`
inside the backslash-continued bodies of `FA_GQA_STAGE_V` and `FA_GQA_STORE_V`; a `#` inside
a macro body is the stringize operator, so `cpp -x c -P -DDK=256 -DGQA_SIDES=2 -DKT=1
-DVT=2 flash_attn_gqa_dpas.cl` on b69f5bee's file prints `276:32: error: '#' is not followed
by a macro parameter`, rc 1, and ocloc 26.27.39122.11 `-device bmg` gave rc 245 with no ISA
for every form (base, NOPRIV, NOPRIV_BROKEN, NOPRIV+PRE, PRE). The alternatives are now
whole macro definitions under file-level `#ifdef`, the broken zero is a value macro
(`#define FA_GQA_V15(x) 0.0h` vs `(x)`), and `FA_UNROLL` is wrapped in
`#if defined(GQA_NOPRIV)` so its empty branch leaves no blank line: `cpp -P` of the fixed
base is identical to the pre-0130k file (25f3b7a5), diff 0 lines. Everything else in 0130k is
unchanged: the unrolled private-array loops, the 64 component stores, base's reduction order.
Gate: `tests/python/test_kernel_preproc.py` runs `cpp -x c -P` over the kernel under every
define set the host builds (base, I8, ST, V2, PRE, PRE+BROKEN, NOPRIV, NOPRIV+BROKEN,
NOPRIV+PRE) at DK=256, GQA_SIDES=2, K/V q8_0/q4_0 (KT=1, VT=2) and f16 (KT=0, VT=0); red on
b69f5bee, green here, and one mutation (a directive inside the continued body) run and seen
failing. `build-deb.sh` runs the same matrix after the patch loop, so the recipe fails on a
kernel that does not preprocess.

## 0130n-fa-gqa-no-private-host-route.patch

`code`, no host change: the host builds the GQA kernel through
`build_program_from_source_optional`, so a refused programme is never silent. Base failing
(`pg == nullptr`) leaves `kernel_fa_gqa_dk[di][i]` null; the V2/PRE/NOPRIV arms skip that pair
and each warns; the tail prints `ggml_opencl: flash_attn_gqa_dpas.cl not built;
flash_attn_f32_f16.cl takes decode` with the downgrade record `path=fallback
reason=not-built what=fa_gqa_dpas`, and decode takes `flash_attn_f32_f16.cl`. This entry
records that route so 0130k's build failure cannot be mistaken for a silent fallback.

## 0130o-fa-gqa-nopriv-macro-continuations.patch

`measured-here` (ocl 26.27, `-device bmg`, q8_0 K / q4_0 V, head 256, 2 sides): 0130l's
component-store lines in the NOPRIV stage/store bodies carried no trailing backslash, so
each macro ended after its four loads and the stores landed at file scope. `cpp` accepts
that; ocloc did not (rc 245, `use of undeclared identifier 'dst'`) for every NOPRIV form,
while the base form was unaffected. 0130o adds the continuations, and the cpp gate now
fails when a macro parameter (`dst`, `k0i`) survives preprocessing: red on 0130l in 6 of 18
sets, green with 0130o. ISA (`measured-here`, same window): base 5,888 B private, 101
private loads / 54 stores, instCount 6,448; NOPRIV 0 B, 0 private messages, 3,840 B spill
(6,921 inst); NOPRIV+PRE 6,144 B private (10,176 B spill). Lesson: a cpp-clean file is not a
compiling file; check that the body ends where intended.

## 0130p-fa-paged-timeline.patch

`code`: `GGML_OPENCL_FA_PAGED_TIMELINE=1`, development only. Markers on both queues carry
the waits of the command after them, so their END is that command's start: per slot copy
start/end, FA split start/end, merge end, the consumed marker and the post hash start of the
request's first split-decode graph (a prompt graph is dropped and the next one recorded),
each in device ns relative to the graph's first paged command; then the derived margin
`copy(k+2).start - FA(k).split end` with its minimum and negative count per slot, and the
cross-step prefetch's copy start against that slot's first FA split end and against the last
FA's split end. The copy queue is created with `CL_QUEUE_PROFILING_ENABLE` only when the
switch is on; the compute queue's stamps need `GGML_OPENCL_PROFILING` in the build, and a
build without it warns once instead of printing zeros. No host wait is added.

## 0130q-fa-paged-nofix-dev-switch-lines.patch

`code`: `GGML_OPENCL_FA_PAGED_NOFIX=1`, development only, drops exactly the cross-step
prefetch's consumed wait -- 0130c's own site -- and `NOCONSUME` keeps its meaning (the
intra-graph wait in `issue` only). Every development switch prints one line with its value
when first read (NOCONSUME, NOFIX, DELAY n, VERIFY mode, VERIFY_SELFTEST, TIMELINE), and
the request-end summary gains `NOCONSUME consumed waits dropped N; NOFIX cross-step waits
dropped N; delay kernels N; self-test writes N`, so window 296's defect -- no line saying
the red arm had run red -- cannot recur.

## 0130r-fa-paged-timeline-nofix-gate.patch

`code`: `tests/python/test_fa_paged_lead.py`'s `FaPagedTimelineAndNoFix`, source-text
cells for the timeline's marks, the margin arithmetic and the negative count, the two
profiling sources, NOFIX called exactly once in the prefetch and NOCONSUME once in `issue`,
every switch's startup line and the request-end counts. Mutations run and seen failing,
then restored: the split start stamp moved after the merge, NOFIX added to `issue`'s wait,
the summary's dev-switch line renamed.

## 0130w-fa-paged-timeline-slot-and-host-time.patch

`code`: window 298b's timeline paired once per ring slot and could not say why a copy did
not land inside an FA, so this patch carries the three mechanisms as one file -- they are
one timeline build's measurements, and none changes a runtime default.

- **The event carries node and slot separately** (0130w's slot fix): the FA stamps passed
  the dispatch's *node index* in the slot field, so only nodes 0/1 matched a ring slot and
  the margin compared one pair per slot. `struct fa_paged_tl_ev` gains `node` beside
  `slot`; the FA's slot is `ggml_cl_fa_paged_overlap_slot` of the chain node (the copy
  stamps already used the issue's slot), the lines print "slot %d node %d", and the margin
  pairs copy k+2 against FA k per slot.
- **The host submit time** (0130x): each stamp keeps the host time it was submitted at
  (`CLOCK_MONOTONIC`), the base takes `clGetDeviceAndHostTimer` so both clocks are in one
  frame (an ICD without it warns once and keeps the device stamps relative to the base
  event), the FA's dispatch entry and exit are stamped host-only, and each copy prints
  `start-submit` -- 'the host submitted late' and 'the device held the copy' separate.
- **`GGML_OPENCL_FA_PAGED_DELAY_AT=splits`** (0130y): moves 0130h's spin before the split
  kernels, the only placement that widens the ring's read window; the default `merge`
  keeps its meaning. For the cross-step prefetch to land inside a reader, the FA to
  lengthen is the slot's last reader two attention layers before the graph's last FA --
  the FA whose `overlap_after` armed `consumed[p]` (`code`).

Gate: `FaPagedTimelineSlotHostDelay` in `tests/python/test_fa_paged_lead.py`; three
mutations run and seen failing, restored: an FA stamp back to the node in the slot field,
the AT=splits spin removed, the submit time dropped from the kept stamps.

## 0131c-fa-paged-timeline-no-marker.patch

`code`: the timeline's stamps are the commands' own events; it enqueues no command of its
own and waits no host in a graph. This file also carries 0131d (one file, one build; a
review that wants them apart splits the hunks).

- **0131c, no marker.** 0130p stamped with `clEnqueueMarkerWithWaitList`, four with output
  events on the compute queue per split-decode FA, a copy-start marker carrying the copy's
  same wait list and an unflushed copy-end marker on the copy queue. Window 299 measured
  the overlap's whole gain gone with the switch unset (451.9 ms/step against 261.3/262.1
  before 0130p) and 0131b's early return only hid it while the switch was off. Now:
  `enqueue_ndrange_kernel` hands out its command's event on request, `k_split`'s event is
  both the split start (its start) and the split end (its end), `k_merge`'s end is the merge
  end, `enqueue_pages` hands out its first and last write command's events (the copy's
  start and end), the consumed marker is stamped as it is (role 5, the marker the next copy
  waits anyway), the post hash kernel's start is the post-start stamp when VERIFY=3/4 runs.
  The base is no longer a command: `tl_finish` takes the earliest device start among the
  stamps. The overlap's remaining markers are the ones the schedule needs (`o.ready`,
  `o.pre_ready`, the after-inputs order, `o.consumed`) -- none of them the timeline's.
- **Which call serialized, from the code (`code`; the cost mechanism `inferred`):** the
  copy-start marker was enqueued whenever `nw` was non-zero, i.e. also with the switch off
  (`if (nw || ggml_cl_fa_paged_timeline())`), so every issue put an extra command with a
  cross-queue wait list and an output event nobody consumed on the queue that carries the
  copies. The compute-side markers ran only under the switch. A marker is a command: ICC
  schedules it as its own node, and its event exists only to be released. Whether that
  costs 190 ms/step as a submission boundary or as an event-scheduling stall cannot be
  decided on this host; window 299's numbers are the measurement that matters.
- **0131d, VERIFY=4: a post-read pair that waits nothing in the graph.** The same pre-read
  hash 0130s/0130d use, plus a hash of the slot's rows enqueued after the reader's split
  kernels and merge (`verify3_post`'s second hash), plus a re-hash when the slot's next
  copy has been issued: a kernel on the compute queue whose only wait list is that copy's
  `o.ready` event, no readback in the graph and no host wait. The pair is collected only at
  the request's end: pre-read against the reader's hash is `read` (a copy landing inside
  the read), the reader's hash against the re-hash is `late` (a copy landing after it), and
  a missing re-hash is counted, not silently skipped. `VERIFY=3`'s collect at a graph's
  start ordered the next graph's dispatch behind the previous graph's last FA (the host
  waited the readback events); VERIFY=4 does not collect there at all. Line:
  `paged FA post-read (request end): pre-read vs read N mismatches, read vs re-hash M,
  re-hash missing on K pairs, stage-1 self-test writes S`; a difference WARNs
  `ggml_opencl: paged FA post-read changed: node %d page %d chunk %d`, once a process.
  `GGML_OPENCL_FA_PAGED_VERIFY_SELFTEST=4` is its stage-1 red arm: one byte written into
  the slot between the pre-read hash and the reader's split kernel, on one node a request.
- **Why TIMELINE on and off gave different greedy texts (`code` for the dependencies,
  `inferred` for the cause):** the markers added no wait to the compute queue and the copy
  queue's copy-start marker carried exactly the wait list its writes carried, so no
  buffer's reader/producer order changed -- the math kernels see the same dependencies.
  What changed is when the copies run relative to the readers: 190 ms/step of extra
  schedule. A copy landing inside a reader's window changes bytes that reader sees, and
  that is a data race, not rounding; the alternative is the B60's per-forward
  nondeterminism (CLAUDE.md's 0.136-nat floor). VERIFY=4, with its self-test proving the
  detector fires, decides which -- on the card.
- **Gate:** `FaPagedTimelineNoMarkerAndPostRead` in `tests/python/test_fa_paged_lead.py`.
  Four mutations run and seen failing, then restored: a marker back in the recording start,
  the copy stamps taken from no command event, the re-hash enqueued without the copy's
  ready event, VERIFY=4's pairs collected at a graph's start.

## 0131j-fa-paged-overlap-graph-boundary-order.patch

Three orderings of the KV window's copy-queue overlap (`GGML_OPENCL_FA_PAGED_OVERLAP=1`) that
the 1.7.0 release review found open (evidence class `code`; none was seen failing on a card,
and the serial stream without the overlap is ordered by the in-order compute queue):

- **The slots' last readers across a graph.** `ggml_cl_fa_paged_overlap_graph` released both
  `consumed[]` events at every new graph. In the reused-prefetch branch (0129u) node 1 is
  issued without the input marker, so its copy then had no wait at all; with an even
  attention chain (16 layers) slot 1's last reader is the previous graph's last FA. A decode
  step drains the host through the logits read, so the window measured so far could not
  meet it; a graph that follows without a drain (consecutive no-logit ubatches whose plan
  repeats) could overwrite slot 1 while that FA still reads it. `consumed[]` is now kept
  across the boundary: the first copies into a slot wait its last reader, and a kept event
  that has completed is a no-op wait.
- **The waited marker is flushed.** The consumed marker is enqueued on the compute queue and
  waited by the copy queue; OpenCL requires the queue holding a waited event's command to be
  flushed. `clFlush(ctx->queue)` follows the marker.
- **The reuse key names the layer.** `ggml_cl_fa_paged_prefetch_same` compared the plan's
  (page, row) pairs, the chain length and the slot. The plan is shared by every layer and
  `graph_compute` runs once per scheduler split, so two splits of equal paged-FA count share
  all three at their node 0. The key now also holds node 0's host store (`pre_host_k`).

Gate: `tests/python/test_fa_paged_overlap_order.py` (`fa-paged-overlap-order`), red on the
series up to 0131i (all three checks fail), green with 0131j. The card arm is the 1.7.0
gate's KV-window smoke (window 170).

## 0131k-fa-paged-predicate-gqa-kernel-and-split-slot.patch

The paged FLASH_ATTN_EXT predicate (`ggml-opencl-fa-paged.h`, 0123zzzzz6) now follows the
dispatcher's own order, and a paged call can no longer reach a kernel that does not read the page
table without an abort. The 1.7.0 gate's op suite found the first defect, on the A770 with the new
tools (window 170, evidence class `measured-here`): the first paged case aborted with "a paged
decode FLASH_ATTN_EXT reached a kernel without a page table". The review of the first fix found
the rest (`code`). arcint's serving cannot reach any of them: it refuses the KV window off the
B60, and on the B60 with the agent's shapes (6 query heads per KV head, at most 6 tokens a decode
step) and no switch set, both predicates accept the same shapes and the dispatch is unchanged
(the 11-64-token calls run as row blocks either way; the old predicate called them the conversion).
0131m then took paged calls out of the row blocks altogether (below).

- **1-8 rows: the GQA decode, or nothing.** A paged decode of 1-8 rows runs on the GQA kernel.
  The split decode (`intel_q_decode`) only opens that path; the dispatcher then needs the direct
  GQA kernel and `(heads / KV heads) * rows <= tiles * 8`. The old clause checked the split alone.
  The A770 builds the split but no GQA kernel (that program needs sub-group 16), so `supports_op`
  accepted the case and the dispatcher aborted. Any other 1-8-row shape counted as the f16
  conversion, but the conversion takes only 9+ rows (`fa_dpas_ok`). Such a call fell to the split
  kernel (with `GGML_OPENCL_FA_INTEL_SPLIT=0`, the generic decode kernels) and read the pool
  rows as the attended sequence, silently, because the abort checked form 1 only. Reachable on
  the B60 with `GGML_OPENCL_FA_GQA=0`, `GGML_OPENCL_FA_INTEL_SPLIT=0`, or a model with more
  query heads per KV head (16 heads at 5 rows), not with the agent's defaults. Those shapes are now declined: the scheduler gives them to the CPU, whose
  FLASH_ATTN_EXT reads the page table, the plan and the host stores (`ops.cpp`).
- **9-64 rows: the wide call, else 0028's row blocks, else the conversion** (superseded by 0131m:
  paged calls no longer take the row blocks). The row blocks
  (`GGML_OPENCL_FA_ROW_BLOCKS`, on by default) split a 9-64-row call into 8-row calls before the
  conversion could take it, so such a shape needs the 8-row GQA decode. The predicate counted it
  as the conversion. The caps now carry the row-block switch.
- **The conversion needs the pair's prompt kernel.** The dispatcher's prompt branch requires
  `kernel_fa_dpas_q_dk[0][pair]` (or `kernel_fa_dpas_dk[0]` for f16) before it reaches the
  conversion; `kernels_prefill` now includes it.
- **The split decode is read by its own slot, after its build.** `q_q1_split_intel` holds three
  kernels in the dispatcher's `kvq` order (q8q8, q8q4, q4q4). Both callers indexed it with the
  GQA arrays' index, which starts at f16: the agent's q8_0/q4_0 pair read q4q4's split, q4_0/q4_0
  read past the array, and f16 read q8q8's instead of `f32_f16_q1_split_intel`.
  `ggml_cl_fa_paged_split_slot` gives the slot. These kernels are built on the dispatcher's first
  head-256 call, so `supports_op` saw none of them before it; it now builds the same program first,
  as its head-512 branch already did.
- **The wide call needs the merge kernel.** Its launch reads `f32_merge` unchecked; the caps
  now carry it, so a failed f32_f16 build declines the shape instead of throwing.
- **Loud where it was silent.** The dispatcher aborts on any paged call that reaches the decode
  kernels off the GQA kernel (it checked form 1 only); a prompt the conversion does not take (a
  mask narrower than the keys, which the predicate does not model) ends there. A second abort
  before the in-place prompt kernels is defensive: no accepted shape reaches it.

Gate: `tests/test_fa_paged_predicate.cpp` (`fa-paged-predicate`). Its new and changed cases fail
by assertion on the 0131j header: a split without a GQA kernel, 6 tokens over 4 tiles, 1-8 rows
without the split, 11 tokens past the cap (row blocks, not the conversion), the wide verify off.
The file itself does not compile there (the slot map and the `row_blocks_on` and `merge` caps
are new); with those removed, the listed cases fail. The card arm is the A770 and B60
FLASH_ATTN_EXT suites and the KV-window smoke on the release tools (window 170k).

## 0131m-fa-paged-no-row-blocks.patch

A paged FLASH_ATTN_EXT never takes 0028's row blocks. The blocks split a 9-64-row call that does
not fit the wide GQA call into 8-row calls, re-entering the dispatcher with a stack copy of the
node. Under `GGML_OPENCL_FA_PAGED_OVERLAP=1` the overlap recognises its nodes by the graph's own
node (`chain[cur] == dst`), so those 8-row calls copied their pages on the compute queue, outside
the overlap's chain, and the outer node never took its turn: the previous graph's cross-step
prefetch into ring slot 0 was neither waited nor invalidated, and the next graph found the same
(host page, ring row) pairs and reused a slot the blocks had since filled with another layer's
pages (`code`).

Found by the 1.7.0 gate (`measured-here`, the B60, Qwen3.8-27B UD-Q5_K_M, q8_0:q4_0, MTP 5): with
the overlap the served answer moved with timing and differed from the serial stream's (window
170n at 120k; 171a at 30k with a 256 MiB window: answer md5 1759821941af against 5f932b72915b).
Switching the prefetch off, or never reusing it, gave the serial answer bit for bit (171a);
`VERIFY=1` found node 0's reused ring row 0 different from host page 0 in all 8,216 chunks, once
a run (171b): the question's prefill (about 20 tokens, 120 rows) is the one row-block graph of
each request, and the question's K/V cached from it changed the next request's answer. A store
generation guarding the host store (an earlier 0131m) refused nothing and cured nothing (171c);
`GGML_OPENCL_FA_ROW_BLOCKS=0` on the unchanged release candidate made the timing-slowed overlap equal
to the serial stream for the first time (171e: 30k and 120k, md5, accepted drafts and verifies),
while at full speed the answer still moved: a second defect, 0131n.

The wide call's dispatch condition (`gqa_wide`) also requires the split's merge kernel, which its
launch reads: the predicate's wide clause did, so without the merge kernel a wide-shaped call was
accepted as the conversion and dispatched to the split branch, which throws (review, `code`; the
merge kernel is built on both cards, so no served path changes).

Such a call now takes the conversion, whose copies the overlap orders, in the serial stream too,
so both streams run the same kernels. The cost is the conversion on the rare short-tail ubatch
(0028 measured 4.3 ms an attention call there at 4k on the B60). The serial stream was never
wrong: with the overlap off there is no copy queue, and the blocks' copies and kernels are in
order on the one compute queue.

Gate: `tests/test_fa_paged_predicate.cpp` (9-64 rows that are not wide take the conversion: 11
tokens, a 20-token prefill, without the split, without the direct kernel, the wide verify off)
and `tests/python/test_fa_paged_overlap_order.py` (the row-block condition excludes paged calls).
Card arm: with 0131n, window 171f (below).

## 0131n-fa-paged-overlap-after-every-exit.patch

The overlap's after-step (`ggml_cl_fa_paged_overlap_after`: the slot's consumed marker, the next
layer's copies issued one layer ahead) runs on every exit of the flash-attention dispatch, through
a guard set right after the paged copy. It ran only at the split decode's return and at the
function's end (`code`). The prompt conversion returns earlier, after its kernels: on every
512-row ubatch, and since 0131m on the 9-64-row calls. On such a node the copy queue had waited
the slot's consumed event and released it, nothing set it again, and the copies of the layer two
on, issued by their own dispatch, waited nothing while the conversion could still read the slot.
The prompt's K/V went into the cache from attention that may have read another layer's pages
(reasoning: that is how a wrong read in the prompt changes the next request's answer; the read
itself was not observed, VERIFY's host waits close the window).

Found by the 1.7.0 gate's window 171g (`measured-here`, the B60, Q5, 256 MiB window, 30k, row
blocks off in every arm): the overlap without the cross-step reuse (`NOREUSE` x2, `NOLEAD` x2) still
gave three answers other than the serial one (md5 5f932b72915b x3, 4477046395ac against
98d05d0226bc), while the arm that issues no copy ahead of its own attention (`NOAHEAD`, reuse on)
equalled the serial stream in both runs, as did `VERIFY=4` (0 pre-read mismatches). The cross-step
reuse was not the cause; NOLEAD's equality in 171a was timing.

Gate: `tests/python/test_fa_paged_overlap_order.py` (the after-step is the guard, and no exit
calls it by hand; red on the 0131m tree). Card arms: window 171f (0131m + 0131n, the B60, Q5): the
overlap at full speed equal to the serial stream bit for bit (md5, accepted and proposed drafts,
verifies) in 3 runs at 30k (256 MiB window) and 3 at 120k (1,024 MiB, both requests), with the
host's timing varied, `VERIFY=1` (1,232 nodes, 0 mismatches) and `VERIFY=4` (0); the release
candidate without either fix with `VERIFY=1`: node 0's ring row 0 wrong in all 8,216 chunks. Window
171h (0131m without 0131n): the overlap at full speed differs from the serial stream in both runs
(md5 5f932b72915b, c1da3368f9af against 98d05d0226bc), and equals it under `VERIFY=1`, whose host
waits close the race. Decode at 30k 92.5 ms a step with the overlap against 127.0; at 120k 170.6
against 292.6.

## 0132-opencl-g31-kq2d-tiles.patch

The Arc Pro B70 (BMG-G31: 32 Xe2 cores, 608 GB/s) runs the same kernels as the
B60 (BMG-G21: 20 cores, 456 GB/s). On the dense 27B its prompt is +51 % at 4,096
tokens, not the cores' +60 %. Its device profile (unitrace, `measured-here`,
2026-11-19) puts the shortfall in the 2D K-quant GEMMs: Q4_K int8 1.52x, Q6_K
1.39x, Q5_K 1.30x of the B60's time. The fp16-path GEMMs scale like the
bandwidth (1.33x), not the cores.

`tools/kq_tune/kqga_e2e.py` searched their tiles on the B70: two seeds of about
100 evaluations, llama-bench pp512 against the default re-measured every 8. Both
seeds found the default within the drift (best 0.9926 and 0.9956), and both bests
share Q4_K's int8 kernel on 8 sub-groups (WG 8, against 16). Interleaved against
the default, 4 pairs: pp512 +0.14 % (3 ahead, 1 equal), pp4096 with q8_0:q4_0 KV
+0.31 % (4 of 4). The change moves no arithmetic: MUL_MAT 2941/2941 both ways, and
the mean KL of the 27B against its CPU reference at ubatch 512 is 0.011867 in both
arms and both repeats (top-1 96.044 %).

The patch reads the Intel device id (`CL_DEVICE_ID_INTEL`, as 0079 does) and,
on 0xe223 only, builds each type's program with those tiles when no
`GGML_OPENCL_KQ_2D_T<i>` is set, logging the tiles it took.
`GGML_OPENCL_KQ_2D_DEVICE=0` keeps the defaults. On the built patch (B70, the release tree's tools with 0132, 4 interleaved
pairs against `GGML_OPENCL_KQ_2D_DEVICE=0`): pp512 +0.36 % (3 of 4 ahead), pp4096 q8_0:q4_0 +0.66 % (4 of 4); the tile line
logged for all three types, none with the switch at 0 (the red case); MUL_MAT 2941/2941; the 27B's KL 0.011867 twice. The B60
was not run: its id takes the unchanged branch (`code`). The real gap (the fp16 Q5_K and
Q6_K GEMMs at bandwidth) needs a structural change, an int8 path for those types,
not a tile.

## 0133-inkling2-arch-and-hparams.patch

Ports the arch, kv and tensor names and the hparams of mainline #32217 (5c008e0), text half
only: the `inkling2` arch, the relative-position and log-scaling keys, the short-conv and
block-residual hparams, `vocab_size_unpadded`, the recurrent conv state width, and the
`inkling2` tokenizer pre. Vision/mmproj, conversion and CUDA stay out.

Gate: builds with `test-llama-archs`.

## 0134-inkling2-relative-position-input-and-model.patch

Ports the text model of #32217 and the mask fixes of #32429 (8af68d4) on the pin's shape:
the pin has dense KQ input masks and no `kv_rows`, so the relative-position input holds the
per-cell distances and the bias is folded into the mask (cast to F32, added, F16 for flash
attention) with a zero sink logit; the pin's iswa `build_attn` takes the mask as a parameter.
Also ports `build_conv_state_store` (rollback slots included) and `set_input_relative_pos`
with the pin's cell-window offset, and adds the model loader and graph: per-head relative
bias, q/k L2 clip, input and post convolutions, block attention residuals, MoE with sigmoid
gating and situ, final logit softcap, unpadded vocabulary.

Gate: on the IQ4_NL GGUF, "What is the capital of France?" answers Paris. KL against mainline
690b87d8's CPU base logits (2048 tokens of English and code, -c 2048, 3 chunks): mean
0.000000 nats, maximum 0.000055, same top p 99.967 %.

## 0135-model-load-ffn_gate_up_exps-scales.patch

Ports #32657: the generic `.scale` pass loads `ffn_gate_up_exps.scale` when the merged
gate_up tensor exists, and rejects a file that carries both the split and the merged scale.

## 0136-ggml-scale-row-contiguous-input.patch

Mainline's ggml change from the 953-commit gap between the pin and the model: SCALE takes a
row-contiguous input, not only a contiguous one, and its f32 and bf16 loops are mainline's
(rows stepped by i1/i2/i3 over src0 and dst, the bias branch reading src0 with src0's own
strides). The merged gate_up expert path passes one half of the matmul result (a view) to
situ()'s linear-beta branch, so both matter.

Red first: `test-backend-ops -b CPU -o '^SCALE'` on the pin's loop fails the four new view
cases (half of a wider and taller base, bias 0 and 1.5, f32 and bf16: NMSE 0.65 to 1.32)
against the use_ref CPU reference, and passes after the port. The use_ref branch is a
per-element reference: without it the CPU test compares the same kernel with itself. The
CPU KL gate on IQ4_NL after the change: mean 0.000000 nats, max 0.000055, same top p
99.967 %, unchanged.

## 0137-inkling2-loop-cache-slots.patch

Ports the Mini's loop of #32694: the `inkling2_loop` arch, the loop kv and tensor names, the
cache-slot expansion (`n_layer_all = n_phys + (passes - 1) * core`), the per-slot layer
arrays, the weight placement through the pin's `n_layer_physical`/`layer_physical`, the
message controllers, and the graph that runs the core `loop.count` times with the previous
pass's block bank.

Gate: the same Paris answer and the same KL numbers above.

## 0138-inkling2-loop-slots-after-scale.patch

Ports #32704: `post_arch_tensors()` runs after the generic scale pass, and the loop's cache
slots alias the physical layers there. Each cache slot would otherwise create its own copy
of the per-tensor and per-expert scale tensors, and the aliasing happens after the pass, so
the slots share the physical layer's tensors (including the expert weights, so the experts
live once per physical layer and the expert cache keys on those pointers).

## 0139-inkling2-arch-tests-and-save.patch

Covers `inkling2` and `inkling2_loop` in `test-llama-archs`, and keeps the loop arch out of
the saver (the file holds 24 layers and the load expands them to 48 cache slots).

Gate: `test-llama-archs` all pass. Red case on the same base logits: 2 loop passes instead
of 3 gives mean KL 0.498889 nats, maximum 11.487644, same top p 76.279 %.

## 0140-fa-indexed-bias-op-slots.patch

`ggml_flash_attn_ext_add_indexed_bias(fa, table, indices)` takes the contract of
upstream PR 32740 and lays it in arcint's slots, because that PR's `src[6]`/`src[7]`
and `op_params` 5 are the paged attention's `ring_k`/`ring_v` and page size here. The
table is `src[12]`, the indices `src[13]`, the flag `op_params` i32 15 (the last word
of the 64-byte block), `GGML_MAX_SRC` 14, and the `static_assert`s beside the paged
ones keep the constants in step. The semantics are the PR's: `table[head,
indices[key, query], query]` is added to the mask value before its F16 rounding, and
queries of all streams are flattened in stream order. The setter holds the shapes
(`table` F32 contiguous `[n_head, n_bins, n_q*streams]`, `indices` I32 contiguous
`[n_kv, n_q*streams]`, a mask covering every attended column). Every backend that
accepts FA and does not implement the mechanism declines the op in `supports_op`
(cuda, cann, et, hexagon, metal, openvino, rpc, sycl, virtgpu, vulkan, webgpu, zdnn),
so the scheduler never runs a kernel that would ignore the flag. Declines do not
carry to ARM or the SVE path: CPU implements it.

## 0141-fa-indexed-bias-cpu-reference.patch

The CPU reference is what test-backend-ops compares against, so it must own the same
arithmetic as the kernels. This tree has two score loops (the one-chunk loop, and the
tiled path `0129`-era work added): both read `bin = indices[query*n_kv + key]`, add
`table[(query*n_bins + bin)*n_head + head]` to the mask in F32, round the sum to F16
once and multiply the ALiBi slope after it. A bin past `n_bins` reads no bias (the
PR's CUDA guard), and a `-INFINITY` mask entry stays `-INFINITY`, tested on the mask,
not on the sum.

## 0142-fa-indexed-bias-tests.patch

`test-backend-ops`' `test_flash_attn_ext` gains `bias`, `n_bins` and `sinks_zeros`,
and the indices a sparse out-of-range value (every 997th), so the `bin < n_bins`
guard is a compared case rather than a comment. Cases at the model's shape: head
128, 24 query heads over 8 K/V heads (`nh=8`, `nr23={3,1}`), `n_bins` 1025, kv
1024/1025/1088 (the window edge and the pad bin), nb 1/4/8 (decode and the verify
rows) and 512 (prefill), the four K/V type pairs f16, q8_0/q8_0, q8_0/q4_0,
q4_0/q4_0, sinks on including the all-zero one the model passes and nr23 `{1,8}` for
the flattened 8-stream queries. `tests/test-flash-attn-indexed-bias.cpp` is the PR's
standalone expectation test without the row list, plus two red cases: a constant bin
against a dense mask holding that bin's value (bit-equal on CPU, and a backend that
reads the mask alone cannot reproduce it), and a hand-computed F16 rounding case at
magnitude 100 where a sum kept in F32 misses by 0.015 of the answer. CPU is both the
reference and the backend under test on this machine, so the F16 contract is proved
by the standalone test only: with the CPU bias switched off, that assert fails and
test-backend-ops still passes both sides.

## 0143-fa-indexed-bias-opencl-decode.patch

The decode and verify kernels read the bias where they fold the mask:
`flash_attn_gqa_dpas.cl` (a row's mask pair, `rh[r]`/`rt[r]` as head and query)
and the `flash_attn_*_q1_split_intel` family, the f16 body and the shared
`fdi_split_body` the quantized pairs call. Both are build-time variants,
`-DFA_BIAS_TABLE=<n_bins>`, off by default, so the plain kernels compile and run
as before. The host compiles a variant on first use, keyed by kind, head size,
kernel slot and `n_bins`, binds the table, the bins and `n_bins` behind the plain
arguments, and `supports_op` declines a bias call only when no variant covers its
shape, naming the reason on a `DOWNGRADE` line (`switched-off`, `device`,
`kv-type`, `shape`, `paged`, `bias-tensors`, `bias-table`, `bias-indices`,
`no-merge`, `no-bias-kernel`, `row-blocks`).
The split's merge kernel belongs to the `f32_f16` variant, which is built on
first use, so `supports_op` and the dispatch ask for it before they read the
merge map: without that the variants never existed at support time and every
decode bias shape declined. The bias tensors are checked in the layout the
setter asserts - the table one row a (stream, query) in dim 2, dim 3 one, the
indices `[kv, rows]` - so a verify of 4-8 rows and a call over several streams
are accepted as the kernels handle them; a table read as dim 3 accepted only
the single-row shapes. `GGML_OPENCL_FA_INTEL_SPLIT=0` keeps ggml's kernels. Sinks need no kernel change:
`flash_attn_f32_merge` joins them across splits (the merge launch already passes
`offset_sinks`).

## 0144-fa-indexed-bias-sinks-opencl-prefill.patch

The XMX prompt kernels add the bias inside their mask epilogues -
`flash_attn_dpas.cl` sub-group 16 and 8, `flash_attn_dpas2.cl`'s classed branch
and its bounds branch - with the same F16 single rounding. Sinks enter `m` and
`l` once after the key loop; `flash_attn_split_combine` joins a split's state
first (`m_final = max(max(m, sink))`, `l = Σ l·e^(m-m_final) + e^(sink-m_final)`),
and the single-call epilogues do the same for their own rows. The carry store
path keeps state only, so the final block's division adds the sink once.

These kernels take the sink and bias arguments unconditionally, NULL and 0 when
a call has none: the prompt mask epilogue already branches per row, so one
pointer test per row is cheaper than a second variant map. `fa_dpas_ok` no
longer declines sinks or a bias, it checks the tensors' shapes (`fa_sinks_ok`,
`fa_bias_ok`) and skips the mask class map for a bias call - a bias tile is
never all-zero, and classing it would skip the tiles the mask alone lets out.
`o` and `l` follow `m` when the sink raises it, and each sink term is added to
the sub-group-reduced `l` rather than to every lane's partial: with the term in
each lane, a 16-lane subgroup counted a call's sink 16 times, and the failing
cases were exactly the sink prompts above 8 rows.
`supports_op` hands a bias call of more than 8 rows to those kernels: it asks for
a prompt kernel covering the head size, no ALiBi or softcap, the mask and the
tensor layouts, and declines the 9-64-row single-batch shapes whose row blocks
re-enter the dispatcher with sliced rows but whole-batch indices.

## 0145-opencl-scale-view-log-arange.patch

The looped model's graph on the OpenCL backend, at decode and at a 512-token prefill: 4,062 view
nodes skipped, 11,826 nodes, 150 unique op shapes on the prefill and 144 on the decode, 27 op
types. Every type is accepted by the pin's `ggml_opencl_supports_op` except three: SCALE on a
row-contiguous view (the merged gate_up half that situ() scales, per MoE layer visit), LOG (the
log-scaled positions, one node a graph) and ARANGE (the short-conv store's rollback slot order,
only with recurrent rollback). This patch adds them: a stride-addressed `kernel_scale_f32_nc`
(supports_op now takes `ggml_is_contiguous_rows`, the kernels picked by `ggml_is_contiguous`),
`kernel_log_f32`/`_4` and `kernel_arange_f32`, each with its own creation and dispatch.

The rest stays as the pin had it. Named slow paths: FLASH_ATTN_EXT with sinks (the model passes
a zero sink logit for the depth bias's +exp(0) denominator) skips Intel's DPAS prefill kernels,
whose condition refuses sinks, and takes the f32/f16 prompt kernels that support per-head f16
masks and sinks - correct, slower. The per-head mask's mask_ne2 is read by the kernels. IQ4_NL
and Q8_0 MUL_MAT_ID go through the K-quant q8_1 id path (IQ4_NL k-unit 32, weights contiguous);
a device that refuses those id kernels declines every expert product to the CPU, so the card
scan must show no `op=MUL_MAT_ID ... kernels-missing` DOWNGRADE. SOFT_MAX over the mix's 2-5
keys is a width below the Intel kernel's required sub-group; three test cases now cover it.

Gate on the card: `test-export-graph-ops` writes the graph's 294 unique op shapes, and
`test-backend-ops -b OpenCL --test-file` replays them against the CPU reference; the SCALE view,
LOG, ARANGE and small SOFT_MAX cases are in the suite directly. No FA kernel is touched.

## 0146-inkling2-rel-bias-indexed-flash-attention.patch

Wires the Inkling 2 Mini rel attention to the indexed bias op of 0140-0142. With flash attention
on, `build_attn_mha` builds the FLASH_ATTN_EXT node, attaches the compact table (F32
`[n_head, n_bins, n_tokens]`) and the distance indices (I32 `[n_kv, n_tokens]`), and asks the
layer's backend through `ggml_backend_supports_op` on that node. A backend that supports it reads
the bias in its kernel; the CPU does. A backend that declines it - OpenCL until 0143-0144 land,
CUDA, and the rest that 0140 makes decline - gets the dense fallback: the bias is gathered from
the table by the indices and added to the mask in F32 before the one F16 rounding, exactly what
the model used to build itself. With flash attention off the model's existing dense fold is
untouched. The sink logit already rides inside the table (minus the per-head sink), so the FA
sink argument stays the zero vector.

Gate, CPU, IQ4_NL, `-c 2048`, three chunks, against the FA-on mainline base logits: the indexed
arm at `-fa on` has mean KL 0.000000 and same top 99.967 %, the dense arm with the indexed op
forced declined has mean KL 0.000000 and the same top. Red case: dropping the table read in the
CPU reference gives mean KL 4.011524 and same top 27.208 % on the indexed arm, while the dense
arm is unchanged (0.001894 against that base, because a FA-on base cannot be matched with flash
attention off; the mainline build's own `-fa off` is 0.001954 against the same base).

The card run is owed: the OpenCL kernels are 0143-0144 of the other branch; until they are in
the series, an Inkling run on OpenCL declines the indexed bias and takes the dense fallback.

## 0147-opencl-small-concat-repeat.patch

The OpenCL CONCAT and REPEAT kernels gave one work item a whole output row and walked
it in a strided loop, so a few kilobytes cost a full serial chain: `kernel_concat_b4`
at 2304x2 to 2304x5 and `kernel_repeat_f32` at 2304x1, 2304x3 and 2304x4 took 16 and
20 us per launch. Both now have a small variant that copies one element per work item,
rows flattened over dims 0 and 1, local size 256, and it runs only when the output holds
at most 16384 elements (64 KiB of F32): large copies keep the old geometry. Only 4-byte
elements take the small variant, and the copy stays byte-identical.

`test-backend-ops` gains the decode shapes: CONCAT along dim 1 of [2304,1] pieces
(2304x2 to 2304x5), the dim 0 2304+2304 split (4608x1), narrow 4x2304 and 4x1024 rows,
and REPEAT of [2304,1] to itself, to 3 and to 4 rows. The same entries are in the perf
list, so `perf -o CONCAT` and `perf -o REPEAT` time exactly these shapes.

Gate, CPU build: `test-backend-ops -b CPU -o CONCAT` 205/205, `-o REPEAT` 21/21.

Gate on the card, owed, both builds:

    test-backend-ops -b GPUOpenCL -o CONCAT
    test-backend-ops -b GPUOpenCL -o REPEAT
    test-backend-ops perf -b GPUOpenCL -o CONCAT
    test-backend-ops perf -b GPUOpenCL -o REPEAT

## 0148-inkling2-depth-mix-stack.patch

The depth mix rebuilt `states` with one CONCAT per earlier block at every layer that starts a
block, and broadcast the query with REPEAT at every visit: 260 CONCAT and 120 REPEAT launches
for a decode token (16-20 us each before 0147). Now a block start writes its residual into a row
of one F32 stack, a visit writes cur into the last row and takes one view over the rows it needs,
and a decode passes the query unrepeated because `mul_mat` takes src1 ne2=1. A second stack holds
the previous pass's bank. The values and their order are unchanged: the rows hold the blocks
newest first, as the old concat chain built them, and the copy is byte-identical.

The stack is read only through views. The scheduler leaves such a tensor without a backend, and
ggml-alloc aborts on `buffer_id >= 0`, so each stack gets one direct consumer: a SUM_ROWS whose
result is unused. A decode token runs 3 of them (one res stack, two bank stacks), one per stack
and not per visit, each reading the stack once (18 KiB a token). A cheaper anchor would need a
backend handle for the stack; the scheduler exposes none that a model graph can reach.

Graph counts for a decode step (test-export-graph-ops, CPU, -c 2048): nodes 11490 -> 11243,
CONCAT 548 -> 288, REPEAT 120 -> 0, SUM_ROWS 192 -> 195; the prompt graph goes 11490 -> 11363.

Gate, CPU build, the pinned IQ4_NL weights, base logits from the same weights:

    llama-perplexity -m <pinned IQ4_NL>.gguf -f /tmp/ink/gate.txt -c 2048 --chunks 3 -fa on \
        --kl-divergence --kl-divergence-base /tmp/ink/base.logits

holds: mean KL divergence 0.000000, same top p 99.967 %. Red case, the visit writing the
previous slot instead of the last row: mean KL 10.334091, same top 1.369 %.

On the card (Arc Pro B60 at a 165 W cap, OpenCL, flash attention on, `measured-here`), 0147 vs
0147+0148: mean KL against the CPU base logits identical (0.001986, same top 98.534 %); llama-bench
tg128 29.09 -> 29.50 t/s at depth 0 and 27.28 -> 27.56 at 4,096; pp512 +0.6 %.

    llama-perplexity -m <pinned IQ4_NL>.gguf -f /tmp/ink/gate.txt -c 2048 --chunks 3 -fa on \
        --kl-divergence --kl-divergence-base /tmp/ink/base.logits

## 0149-opencl-sigmoid-strided-f32.patch

Inkling 2 Mini's expert activation (SITU) takes `sigmoid` of the gate half of the fused gate_up
result, a strided view. The OpenCL backend accepted SIGMOID only on contiguous input, so the
scheduler ran that node on the CPU: per decode token 48 graph splits, each a blocking read of the
gate to the host and about three queue synchronizations (`measured-here`: gdb backtraces of
`clEnqueueReadBuffer` and `clWaitForEvents`, all from `ggml_backend_sched_graph_compute_async`
and `sched_copy_input`; `GGML_SCHED_DEBUG=2`: 49 CPU splits per graph, the token embedding and 48
SIGMOID nodes). The host could not run ahead of the card, so decode was bound by kernel enqueue
(about 6,700 launches a token at about 3.6 us host each, unitrace).

`kernel_sigmoid_f32_nc` takes the input by its strides, one work group per row, the layout of
`kernel_tanh_f32_nc`, and the formula of `kernel_sigmoid_f32` (the same rounding);
`ggml_cl_sigmoid` uses it for a non-contiguous f32 input, and `supports_op` admits that case. The
contiguous paths and the f16 and bf16 paths are unchanged.

Card (Arc Pro B60 at a 165 W cap, Inkling 2 Mini IQ4_NL, OpenCL, flash attention on,
`measured-here`), 0148 vs 0148+0149:

| | 0148 | 0149 |
|---|---|---|
| CPU splits per graph | 49 | 1 |
| pp512 depth 0 / 4,096 (t/s) | 185.0 / 178.4 | 417.9 / 387.8 |
| tg128 depth 0 / 4,096 (t/s) | 29.50 / 27.62 | 34.86 / 32.25 |
| mean KL vs the CPU base, same top | 0.001986, 98.534 % | 0.001931, 98.501 % |

test-backend-ops `-o SIGMOID`: the two strided f32 cases (`v=1`, [128,2,2,2] and [5,7,11,13]),
"not supported" before, pass; the red case (`exp(+x)` in the new kernel) fails exactly those two
(ERR 2.05 and 1.94). The contiguous bf16 [128,2,2,2] case is borderline at the test's 1e-7 NMSE
threshold with random inputs in both builds: 6 of 50 runs fail on 0148, 2 of 50 on 0149.

## 0150-ggml-glu-situ-fused.patch

Inkling 2 Mini's expert activation (SITU) ran as a chain of nine small ops per MoE visit on the
gate and up halves of the merged gate_up result, two strided views: scale(gate, 1/beta), tanh,
scale(beta), sigmoid(gate), mul, then the up branch scale(up, 1/lb), tanh, scale(lb) and the
final mul. 48 visits a decode token.

The patch adds a GLU variant: `GGML_GLU_OP_SITU` with op params beta and linear_beta (the way
`GGML_GLU_OP_SWIGLU_OAI` carries alpha and limit), `ggml_situ` and `ggml_situ_split`, and
`build_moe_ffn` runs `LLM_FFN_SITU` as that one op on the gate and up views.

The CPU implementation is the reference and keeps the chain's formula and order:
`beta*tanh(gate*(1/beta)) * (1/(1+exp(-gate))) * (linear_beta > 0 ? linear_beta*tanh(up*(1/linear_beta)) : up)`.
`kernel_situ_f32` in `glu.cl` takes both halves by their strides, so the merged gate_up tensor
and the split two-tensor form share it; f32 only, and `supports_op` admits exactly that case.

`measured-here`, CPU tree (`test-export-graph-ops` at `-c 2048`) and the CPU KL gate on the
IQ4_NL file, base 547c468 vs this patch:

| | base | patch |
|---|---|---|
| decode graph nodes (`tg`) | 11,243 | 10,859 |
| prefill graph nodes (`pp`) | 11,363 | 10,979 |
| mean KL vs the base logits, same top | - | 0.000000, 99.967 % |

The unmeasured card expectation is 8 kernel launches fewer per MoE visit, 384 a decode token,
against about 6,778 before.

test-backend-ops `-o SITU` (correctness, merged and split, [2304,8,1,1], [2304,512,1,1], an odd
small shape, plus the split [1152,...] forms): six cases, pass on the CPU. On a card the same
six compare the OpenCL kernel against the CPU reference. The KL red case, dropping the sigmoid
in the CPU op, gives mean KL 1.787650 and same top 56.044 %.

On the card (Arc Pro B60 at a 165 W cap, Inkling 2 Mini IQ4_NL, OpenCL, flash attention on, `measured-here`),
0149 vs 0149+0150: tg128 34.88 -> 36.60 t/s at depth 0 (+4.9 %), 32.23 -> 33.74 at 4,096, 28.78 -> 29.93 at
32,768; pp512 +2.7 % (417.9 -> 429.1). Mean KL against the CPU base 0.001931 -> 0.001947, same top 98.501 ->
98.175 %. test-backend-ops `-o SITU` on the card: 6 / 6 (the op's name is the GLU variant's; `-o GLU` selects
nothing); the red case (sigmoid dropped in `kernel_situ_f32`) fails all 6 (ERR ~0.5) and moves the model's KL
to 1.78; SWIGLU unchanged, 24 / 24.

## 0152-fa-bias-carry-block-bins.patch

A quantized K/V pair's prompt path dequantizes K and V to f16 in blocks of 16,384 keys
(`GGML_OPENCL_FA_DPAS_F16KV_BLOCK`) and runs the f16 prompt kernel on each block with the softmax
state carried between them (`FA_CARRY`, `kv0` the block's first key). The kernels advance the mask
by `kv0`, but 0144 read the indexed bias's bins at `[key]`, the key within the block: every block
after the first took the first block's relative-position bins. f16 K/V reads in place (`kv0 = 0`)
and a mixed pair takes the f32_f16 kernel, so only q8_0/q8_0, q8_0/q4_0 and q4_0/q4_0 past 16,384
keys were affected. The bins are now read at `kv0 + key` in `flash_attn_dpas.cl` (both kernels) and
`flash_attn_dpas2.cl` (both lookups).

`measured-here` (Arc Pro B70 at a 220 W cap, Inkling 2 Mini IQ4_NL, KL at 32,768 tokens against the
CPU's f16 logits): q8_0/q8_0 0.037040, same top 94.586 % before; 0.005965 / 98.065 % after (f16
0.005306, q8_0 K only 0.005412, q8_0 V only 0.005688, the CPU's own q8_0/q8_0 0.005908). Located by
switch: `GGML_OPENCL_FA_DPAS_Q=0` (the quantized prompt kernels off) gave 0.005554, `FA_DPAS2=0` and
`FA_GQA=0` left 0.037; at 2,048 tokens (one block) q8_0/q8_0 was 0.002351. With q8_0/q8_0 KV the
needle at 64,000 and 120,000 tokens had been missed; with f16 KV it was found at 64,000.

Two test-backend-ops cases run the quantized pair's prompt path with the bias, Inkling's shape
(head 128, 24 query heads over 8, 1,025 bins), in one block (1,025 keys, 512 rows) and in two
(16,448 keys, 128 rows). No case reached that path before: 0142's quantized bias cases have 4 rows
(decode) and its prompt case is f16, so the whole FLASH_ATTN_EXT suite passed on 0150 even with
64-key blocks forced (3,171 / 3,171). With the two cases: 3,173 / 3,173 on 0152 at both block sizes.

## 0153-attn-seq-pos-min.patch

`llama_memory_seq_pos_min` for a hybrid memory is the **max** over its attention and
recurrent parts, and the recurrent cell sits at the newest position (one cell a sequence),
so on a hybrid it reports that position and hides the sliding-window cache's oldest cell.
A window-sized SWA cache (`swa_full = false`) cannot be trimmed to an arbitrary prefix:
`llama_kv_cache_iswa::seq_rm` forwards to the SWA cache, whose `seq_rm` always returns true,
so a cut below the retained window drops exactly the keys the next query would attend to.
arcint had no way to ask the attention part.

0153 adds `llama_memory_attn_seq_pos_min(llama_memory_t, llama_seq_id)` to `llama-ext.h`:
the SWA cache's `seq_pos_min` for a hybrid-iswa or iswa memory, the memory's own value
otherwise. `src/llama-context.cpp` implements it by the memory's concrete type (RTTI); it
reads a cache, writes nothing, and is valid before and after a decode.

**arcint's use** (`code`, `--llama-swa-window`): the flag sets `llama_context_params.swa_full
= false`, so the SWA layers hold `n_swa + n_ubatch` cells padded to 256 instead of the whole
context. `src/core/swa_window.h` decides a prefix cut: the retained history (this call)
must start at or below the next query's oldest key (`common - n_swa + 1`, clamped at 0);
otherwise the lane resumes from the newest context checkpoint at or below `common` and
prefills from it, and clears if there is none. The checkpoint's
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` state already carries the SWA cells at the pin
(`llama_kv_cache_iswa::state_write` calls `kv_swa->state_write`; the base cache writes
nothing under the flag), so no state-format change was needed — only the position query.

**CPU measurement** (`measured-here`, the recipe host, the pinned tree with the series,
`-DGGML_OPENCL=OFF`, Inkling 2 Mini IQ4_NL, the 8-core CPU build, `tools/swa_ckpt_probe.cpp`):
the partial state at 2,048 tokens is 163.7 MiB and at 512 tokens 83.7 MiB at f16 (delta 80.0
MiB = 512 cells x 40 SWA layers x 4,096 bytes: the window is in the blob). The probe's arms,
on the same model: a control arm prefills the same edited prompt in another ubatch split
(mean KL 0, the CPU's batch-shape floor here), an append-only follow-up against a fresh full
prefill has the same top token and mean KL 0, and an edited prompt below the ring resumes
from its checkpoint with the same top token and mean KL 0.001496 (the restored SWA cells are
re-found in the ring's slot order, so the FA sums them in another order; the model's
batch-shape floor is 0). The pre-change trim-only arm is skipped on a hybrid (its recurrent
refusal blocks the trim, `n_rs_seq` 0); on the attention-only SWA models below it runs and
is the red case. The arcint unit case (`tests/test_swa_window.cpp`) is red 4 of 7 with the
old always-true rule.

**Other SWA models** (`measured-here`, same host, the same probe): Dory1.0-4B Q5_K_M,
`per_loop` (24 global + 72 SWA cache layers, window 1,024): floor 0, append-only 0, the
checkpoint resume same top and mean KL 0.011592; the trim-only arm keeps a hole — mean KL
5.887903, top token differs. Gemma 4 E4B Q4_K_M (4 global + 20 SWA, window 512): floor 0,
append-only 0, the checkpoint resume same top and mean KL 0.022537; trim-only mean KL
22.814130, top differs. Both resumes are inside the 0.03 answer-level bar with the same top
token; the residual is cell re-placement, not the window.

**The depth cost of the full-size cache on the CPU** (`measured-here`, llama-bench on the
same 8-thread CPU build, f16 KV, one repetition each; the `swa_full=false` arm is the stock
tool, the `true` arm is the same tool with `cparams.swa_full` forced true, a measurement
build only): Gemma 4 E4B at 16,384 depth pp16384 48.33 -> 61.56 t/s and tg64 8.67 -> 12.56
t/s with the window (prefill +27 %, decode +45 %: its 20 SWA layers over 16k masked keys).
Inkling 2 Mini at the same depth pp16384 18.79 -> 19.87 and tg16 7.34 -> 7.94 t/s: the
single 16-token decode reads no separable effect there, its CPU token is weight-bound
(7-8 t/s) and the two runs sat in different host-load windows; the card is where the
effect was measured (30.4/31.9 t/s in llama-bench at depth 16,384 against 20.5 q8_0 /
24.3 f16 served with the full-size cache). The window cache never regressed the CPU decode.

**Dory's `last_loop` is refused with the flag** (`code`, arcint's load): that mode overwrites
each loop pass's KV, so a prompt's prefill depends on where its chunk boundaries fall
(upstream compares such a model only against the same batch; DESIGN's Dory paragraph). The
probe measures it: the floor arm — the *same* prompt in another ubatch split — is mean KL
0.026827 nats; the append-only follow-up 0.002890 same top, the checkpoint resume 0.008718
same top, trim-only 6.011353 top differs. A checkpoint resume puts the continuation on other
boundaries than the full prefill, so the window cache cannot be held to the probe's
equivalence there; `--llama-dory-kv per_loop` serves with the flag, `last_loop` starts
without it.

## 0156-opencl-q8_0-mul_mat_id-int8-dot.patch

Inkling 2 Mini's four coda layers keep their experts in Q8_0: gate_up K=2304 and down K=1152,
128 experts, top-8. Their `MUL_MAT_ID` had no OpenCL fast path, so a decode token took the generic
flat float kernel (`kernel_mul_mv_id_q8_0_f32_flat`, one float dot per value) with the downgrade
line `op=MUL_MAT_ID type=q8_0 k=1152|2304 path=generic reason=no-fast-path`: 8 launches at about
84 us each, 0.67 ms of the token, against the IQ4_NL experts' `kernel_mul_mv_id_iq4_nl_q8_1`
(88 launches, 4.8 ms, about 330 GB/s on the same card at 1.8.0). The IQ types reach their fast
path through `ggml_cl_kq_type_index`, which returns -1 for Q8_0, so the Q8_0 product fell through
to the generic arm.

The new arm, `ggml_cl_mul_mat_id_q8_0`, takes Q8_0's decode shapes the same way the IQ types do:
`kernel_kq_quantize_q8_1` quantizes the activation to q8_1, and `kernel_mul_mv_id_q8_0_q8_1` in
`mul_mv_iq_q8_1.cl` runs the int8 dot of the existing `mv_q8_0` helper on the flat Q8_0 planes,
one work-group per (expert pair, token) on the third dimension. The pair is the group id: slot i =
k % n_used, token t = k / n_used, the expert from `ids`, the activation column (i % ne11) + t *
ne11, and the weight blocks at the expert's rows (e * ne01 + row), as the matvec does. Guarded to
`n_tok < 16` (a decode batch; prefill keeps the flat kernel), Q8_0 / f32 / i32 / f32, ne00 divisible
by 32, contiguous, the flat SoA planes (`ggml_cl_is_q8_0_soa`), not an expert cache's `.slots` /
`.bank` (their -1 ids are not placed here), and src0's ne3 = 1. The kernel is built with `-DMV_Q8_0`
in the one-column program, which is now always done; the q8_0 matvec above stays behind the 0115 arm.

Four test-backend-ops cases take exactly the coda shapes (Q8_0, 128 experts, top-8, K 2304, 2304
rows, 1 and 4 tokens; K 1152, 2304 rows, 1 and 4 tokens) in the test list and in the perf list, so
perf mode times them. The CPU build's `test-backend-ops -o MUL_MAT_ID` passes, the OpenCL build
compiles and the .cl passes the C preprocessor. On the card (`measured-here`, B60): the q8_0
`MUL_MAT_ID` cases pass, and a red build (the kernel's dot broken) fails 29 of them.

## 0157-opencl-tiny-mv-splitk.patch

Developed as patch 0151 on a working branch and renumbered here (its comments still say 0151): it was written against an older base, so it
is applied here after 0156 with offsets resolved and the logic unchanged. A one-column decode
matvec of few output rows leaves most of the card idle: the f32 GEMV puts one work-group per
output row (out=48 on 160 cores) and the flat q8_0 kernel one work-group every 8 rows (out=24 is
3 work-groups). Both get a split-K variant that reduces a K slice of one (row) per work-group into
partials, and `kernel_sum_slices_f32` sums the slices in a fixed order, so a shape sees a fixed
slice count and a fixed order. `kernel_gemv_f32_f32_splitk` (in `mul_mm_f32_f32_l4_lm.cl`) covers
f32 weights at one activation row, out < 512 and K >= 1024, the slices picked so out * slices
reaches about 256 while K divides by 4 * slices; `kernel_mul_mv_q8_0_f32_splitk` (in
`mul_mv_q8_0_f32_flat.cl`) covers the flat q8_0 planes at one column, 2D weight, out < 512 and K
>= 1024, blocks divisible by the slice count. A multi-column step (an MTP or spec verify, ne11 > 1)
keeps the existing kernels. `GGML_OPENCL_F32_SKINNY=0` still turns the f32 arm off.

test-backend-ops carries the Inkling 2 Mini decode shapes (one activation row, K=2304, f32 and q8_0
weights, out 2, 3, 4, 5, 24, 32, 64, 128, 384) in the test list and the perf list, and the production
shapes the split-K reaches: the agent 27B's `ssm_alpha` / `ssm_beta` (F32 48 x 5120, and a q8_0 twin),
the coder's routers (F32 184 x 2048 and 256 x 2048, which stay unsplit) and Flash-Next's MTP head injects
(Q8_0 4 x 10240, split 64 ways; and 320 x 10240). Two defects were found on the card only and fixed:
the q8_0 host set one kernel argument too many (CL_INVALID_ARG_INDEX), and the f32 kernel started each
slice in floats on float4 pointers, so every slice past the first read the wrong rows (the 27B's KL
went to 0.54). On the card (`measured-here`, B60): `MUL_MAT` 2965 of 2965; the red build (the slice
bug) fails 14 f32 split-K cases.

The split-K takes every model whose decode has a one-column matvec with out < 512 and K >= 1024
of f32 or flat q8_0 weights. In the agent's Qwen3.8-27B Q4_K_M all 96 such tensors are the
48 gated delta-net layers' `ssm_alpha.weight` and `ssm_beta.weight` (5120 x 48, five slices), whose
F32 weights the f32 arm covers. Qwen3.8-Flash-Next's main model has none; its MTP head's two
Q8_0 injects (4 x 10240) take the q8_0 arm. The coder's 1-D shared-expert gate (F32 [2048], a
1 x 2048 product) takes the f32 arm, 32 slices. On the card (`measured-here`, B60/B70): KL to the
CPU reference at micro-batch 1 the agent 0.010389 -> 0.010350, the coder 0.005796 -> 0.005664,
Dory and Flash-Next bit-identical; Inkling decode +1.2 to +1.3 % from 0156 + 0157.

## 0158-ggml-norm-clip-fused.patch

Inkling 2 Mini clips three rows per layer visit: the q rows at head width 128 over 24 heads, the
k rows at 128 over 8, and the relative rows at width 16 over 24 heads. Each clip ran as a chain
of small ops on the row: the optional per-channel scale, sqr, sum_rows, sqrt, scale_bias(1, eps),
fill(threshold), div, clamp(0, 1) and mul, 9 nodes with a scale and 8 without, 26 a visit, 48
visits a decode token. The thresholds are the metadata's `attention.qk_clip` 22.627417 for q and
k and `attention.relative_clip` 8.0 for the relative rows; the eps is the graph's 1e-6.

The patch adds `GGML_OP_NORM_CLIP`: src0 the rows, src1 the per-channel scale or null, op params
threshold and eps; `ggml_norm_clip` builds it and the Inkling graph's `clip()` emits it, so no fill
constant is created. The CPU reference keeps the chain's order: the per-channel mul, the sum of
squares in row order with the wide accumulator the existing `sum_rows` uses, sqrt, then the
scale_bias order `norm*1 + eps`, then `threshold/norm`, then MIN(factor, 1) before MAX(factor, 0),
then the mul. It is bit-identical to the unfused chain at all eight test shapes (max abs 0).

`kernel_norm_clip_f32` in `norm_clip.cl` runs one 32-lane work-group a row, the row sum in a fixed
lane order (a strided per-lane partial, then a tree with barriers), f32 throughout, and re-reads
the scaled row for the multiply. `supports_op` admits f32 rows that are contiguous, with no scale
or an f32 scale of the row's own channel and head count. The OpenCL build compiles and the .cl goes
through the C preprocessor.

`measured-here`, CPU tree (`test-export-graph-ops` at `-c 2048`, 0157 vs this patch) and the CPU KL
gate on the IQ4_NL file, base logits the same window:

| | base | patch |
|---|---|---|
| decode graph nodes (`tg`) | 10,859 | 9,755 |
| prefill graph nodes (`pp`) | 10,979 | 9,875 |
| mean KL vs the base logits, same top | - | 0.000000, 99.967 % |

Both graphs lose 1,104 nodes: the clips' 1,248 launch nodes a token become 144 NORM_CLIP nodes.
On the card (`measured-here`, B60): `NORM_CLIP` 8 of 8, its red build 0 of 8; Inkling decode
34.13 -> 35.69 t/s at 4k (+4.6 %), 37.06 -> 38.93 at depth 0, KL to the CPU reference inside the bar.

test-backend-ops `-o NORM_CLIP`: eight cases, the q/k/rel shapes of decode and of a 512-token
prompt and an odd width (8/8 on the CPU build; on a card the same eight compare the kernel with
the CPU reference). Red case, dropping MIN(factor, 1) in the CPU op (and in the kernel for the
card tree): mean KL 3.417977, same top 38.938 %, and the unfused-chain equivalence check mismatches
all eight shapes (max abs 0.61 to 14.5).

## 0159-opencl-topk-moe-sigmoid-router.patch

Patch 0064's top-k MoE fusion matched only a SOFT_MAX router. Inkling 2 Mini gates with
`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` and a per-expert selection bias, so its router ran
unfused: sigmoid, reshape, bias add, argsort, view, get_rows, reshape, sum_rows, scale(sum, 1,
1e-20), div, reshape — 11 nodes a layer, 48 layer visits a decode token.

The matcher (`ggml_opencl_can_fuse_topk_moe_sigmoid`) scans the same tail after the sigmoid:
ARGSORT, VIEW, GET_ROWS, RESHAPE, SUM_ROWS, then either `SCALE(sum, 1, eps)` + DIV or CLAMP + DIV,
and the closing RESHAPE; the bias ADD is optional and is used only for the selection, the weights
stay the unbiased probabilities. The fused kernel (`kernel_topk_moe_sigmoid_norm_n*`) takes one
token a sub-group, reduces the k maxima over the lanes with descending ties to the lower expert,
writes ids at the VIEW and weights at DIV (the closing RESHAPE is a view of DIV at the same byte
offset, but its `nb[1]` is 4 because `ne[0] == 1`; DIV holds the contiguous k-float rows,
`nb[1] == k*4`, which is the row stride the write needs), accumulates the selected p f32 serially
by lane 0 in the j order SUM_ROWS uses, then adds eps or clamps and divides. Softmax arms keep
the existing kernel and the same 9-node dispatch unchanged.

`measured-here`, CPU tree (`test-export-graph-ops` at `-c 2048`, 0158 vs this patch) and the CPU KL
gate on the IQ4_NL file, base logits the same window:

| | before | after |
|---|---|---|
| decode graph nodes (`tg`) | 9,755 | 9,755 |
| mean KL vs the base logits, default batch | - | 0.000000 |
| mean KL vs a `-ub 1` base logits at `-ub 1` | - | -0.000000 |

The CPU graph is unchanged by construction: the fusion is OpenCL-only, so the CPU gates cannot
see it; the card numbers below are the evidence for the fused path. The batch base logits
compared against the `-ub 1` decode path read 0.001989, a pre-existing path difference.

`measured-here`, card (B60): KL against the CPU reference 0.001950 at the default
batch and 0.001988 at `-ub 1` (the 0158 build reads 0.001999 / 0.002078, the path difference); KL
against 0158's own `-ub 1` logits 0.000543, same top 99.251 %; decode d4096 35.52 -> 35.94 t/s,
d0 38.89 -> 39.15. The 0.000543 is the sigmoid's own codegen: the fused kernel evaluates
`1/(1+exp(-x))` inside the IQ matvec program, the unfused chain in `sigmoid.cl`'s work-item
kernel, so p differs in the last bits and near-ties can flip an id; bit-identity of p would
require dispatching the standalone sigmoid kernel, which deletes the fusion. Stated, not hidden.

An earlier version regressed on the card: its `wnode` (10/9) wrote the closing RESHAPE,
whose `nb[1] == 4` is not the row stride — 44 op cases failed and prefill fell to 412 t/s; its
`k % 4` float4 sum path could not be separated from that bug and is dropped. The weight node is
the DIV, card-proven.

test-backend-ops: the builder gains the `norm_eps` arm and Inkling's router shapes, 128 experts
top 8, n_tokens 1 and 512, with the bias pinned at +-3 (so a fused arm that writes the biased
value as the weight fails clearly). The graph is built in the emitted node order; `validate_graph`
asserts it on every backend, and a fused-launch counter
(`ggml_backend_opencl_topk_moe_sigmoid_fused_count` through `get_proc_address`) fails the case
when it did not reach the SIGMOID kernel; the card green run printed "fused arm reached" for the
cases and `GGML_OPENCL_FUSE_TOPK_MOE=0` passes them on the unfused chain. CPU build 426/426.

Architectures that set `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` (`code`): afmoe, cohere2moe,
glm-dsa, glm4-moe, glm5-next, hy-v3, k2-horizon, laguna, mimo2, nemotron-h-moe, nemotron-h,
step35 and inkling2 match when metadata has `expert_weights_norm` and there is no group
selection; deepseek2 routes through `n_expert_groups`, llama4 selects on the logits and has
`norm_w` false, so they do not match. Production: Inkling matches; Qwen3.8 27B and Dory are
dense; coder qwen35moe and Flash-Next qwen4exp gate by SOFT_MAX and keep the existing arm.

The red case: a build whose kernel writes the biased selection value as the weight (in
`mul_mv_iq_q8_1.cl`). The +-3 Inkling cases and the card KL then fail while the ids stay right.

## 0160-card-memory-and-graph-stats.patch

The spill guard's budget is the driver's figure, not OpenCL's. On the B60 the two disagree
(`measured-here`): `DRM_XE_DEVICE_QUERY_MEM_REGIONS` says 24,480 MiB, `CL_DEVICE_GLOBAL_MEM_SIZE`
23,256 MiB, and the agent 27B peaked at 24,196 MiB without spilling, so the allocations come out
of the region total. The patch exposes `llama_device_card_memory(dev, out)` to `llama-ext.h`: the
loan's existing probe (this process's xe clients by drm-pdev, then the region query on one of
their render fds), returned as the driver's total, the device-wide used when it is a figure,
this process's fdinfo vram0, and a reason. The bool is false when no region total is available
(no `drm/xe_drm.h`, no client, a failed ioctl); a total with the perfmon-less used of 0 is still
a driver figure and returns true with `used_known` false. The guard then bounds by the total
with the other clients' share at 0 and says why. It is not part of the loan's rule, which keeps
its own probe.

It also exposes `llama_context_graph_stats(ctx, out)`: the reserved prompt and token graphs'
node and split counts, for the "runs" tier's start warning. The values are what the reserve
logged before, kept as members.

arcint's rule is `src/core/spill_guard.h` (pure, message included) and the load path in
`src/exec/backend_llama.cpp`. Device demand is the target context's breakdown by kind plus the
draft context's own breakdown (the in-model MTP context: its KV 416 + reserve/compute 316 + head
rows 172 = 904 MiB at the agent's 262,144; a separate `--llama-mtp-gguf`/`--llama-dflash`
drafter adds its own model), the expert cache's own on-card bytes (slots, Q8_0 banks, gather
mirror, loan tail), and this process's fdinfo bytes above the known demand (a projector, the KV
stream, lazily grown scratch). Deliberate host bytes — a pinned expert bank, host buffers — are
a separate line and never a refusal term. Other clients (`used - fdinfo`) count.

The margin is `kSpillGrowthMarginBytes` = 422 MiB = the worst measured growth beyond the full
prediction (agent 294 at a 255k needle, Inkling 231 with the switch, Flash-Next 184 and 172 on
two measured first-request legs, Inkling 65,536 full SWA 226) + 128 MiB headroom. A 512 MiB headroom would refuse
the agent, which is measured to run; 128 keeps it. The card's lazy allocations are inside the
draft breakdown and the expert-cache bytes, not in this margin.

The Inkling 65,536 full-SWA start fits: breakdown 23,460, predicted 23,459 before the margin,
23,585 after the first request, a 60,000-token needle answered with VRAM peak 23,686 and GTT at
most 308 MiB — no spill. Inkling 131,072 full SWA is above the card by its breakdown alone
(30,372 MiB) and is the red case; Flash-Next `--llama-expert-cache 40000` is the other red case.
The cross-check is after the first request: `|M2 - G| <= 422 MiB` and `M2 <= card`.

## 0161-qwen4exp-dense-layer-rule.patch

Qwen3.8-Flash-Next's own model card calls it "an experimental preview of the architecture that
will underpin Qwen4" (`model_type qwen4_exp`), so `qwen4exp` is the code a dense Qwen4 27B will
serve through. It was MoE-only: every layer built a router and expert tensors, and the graph's
`GGML_ASSERT(ffn_gate_inp != nullptr)` rejected any layer without one. A dense GGUF could not load.

The per-layer rule is vLLM's reading of the config, now from metadata: a layer is **MoE** iff it is
not in `mlp_only_layers`, `num_experts > 0`, and `(il+1) % decoder_sparse_step == 0`; otherwise it
is a dense MLP(`feed_forward_length`). The patch adds `{arch}.mlp_only_layers` as a per-layer bool
vector (true = mlp-only, the converter translates HF's index list and pads the MTP block: false,
or true for a checkpoint without expert tensors, whose MTP block is then dense too), and reuses `{arch}.moe_every_n_layers` for `decoder_sparse_step` with 1 as the absent
default. Dense layers load `ffn_gate`/`ffn_up`/`ffn_down` only; MoE layers take the unchanged
router/expert/shared-expert code, and the dense branch in `build_layer_ffn` is gated on
`ffn_gate_inp == nullptr`, so an existing MoE GGUF builds today's graph.

Two assumptions left the loader. The GDN/QSA schedule is read from
`attention.recurrent_layers`; if that key is absent, `full_attention_interval` is now required
instead of defaulting to 4 (the 3:1 schedule is not known for Qwen4). The indexer is optional by
presence: a full-attention layer with none of the four indexer tensors runs dense attention, the
memory filter allocates the indexer cache only for layers that carry it, and the load log says
`layer N has no QSA indexer tensors, running dense attention`; a partial set (1-3 of 4) is an
error, and a file with no indexer key group at all allocates no indexer cache. The load log also names each dense MLP layer and the MoE/dense counts. HC low rank and the
HC stream count stay metadata-only (a 320 rank is nowhere hard-coded).

`measured-here`, CPU: the dense fixture (0163) fails on the pre-patch build (integer divide by
`expert_used = 0` in the expert width) and loads and runs after it (PPL 292.9699,
`llama-perplexity -c 128 -b 16 -t 2`); a synthetic qwen4exp MoE model saved with the new build keeps the old
build's logits at mean KLD 0.000000 and same top p 100.000 (0165). The real Flash-Next GGUF's
CPU KL is owed on a host that holds it.

## 0162-qwen4exp-converter-dense-metadata.patch

The converter now accepts a dense qwen4_exp config (`num_experts` 0 or absent) and keeps the arch
name **qwen4exp**: the runtime decides MoE or dense per layer, so a second dense arch or a
different GGUF name would split one model family across two code paths for no gain. The same
converter serves the current Flash-Next file; the new keys are defaults that reproduce today's
all-MoE graph (step 1, mlp_only false).

`set_gguf_parameters` writes `mlp_only_layers` as the per-layer bool vector (HF's indices, padded
over `block_count - num_hidden_layers` with false, or with true for a checkpoint without expert
tensors) and `moe_every_n_layers` from
`decoder_sparse_step` (default 1, 0 refused). The indexer key group is written only when the
config carries `indexer_n_heads`, so a checkpoint without QSA carries no indexer keys. `FFN_GATE`,
`FFN_UP` and `FFN_DOWN` join the arch's tensor list, so `mlp.gate_proj`/`up_proj`/`down_proj`
map to the dense tensors 0161 asks for; an MoE layer's `mlp.experts.*` and shared-expert names are
unchanged. Which of the two published dense widths (14336, 17408) is real stays the config's
`intermediate_size`; the loader reads it, nothing is baked in.

A checkpoint with no MTP block now converts. `_QwenMtpMixin.__init__` used to assert
`opt_num_mtp_layers != 0`; the count comes from `mtp_num_hidden_layers` or the indexed
`mtp.layers.N` tensors, and zero now logs `no MTP layers in the checkpoint, converting the
trunk only` and writes no `nextn_predict_layers` key. One MTP block (`--mtp`, below) still takes
the shared remapper: `mtp.layers.N.*` to `model.layers.<num_hidden_layers>.*`,
`mtp.pre_fc_norm_*` to `enorm`/`hnorm`, and qwen4exp's `_MTP_EXTRA` sends
`fc_embedding`/`fc_hidden` through the `eh_proj` concatenation and `hyper_connection_mixer` to
`nextn_hc_head`.

The GDN geometry had come from the wrong config layer. `AutoConfig` resolves
`model_type: qwen4_exp` to the multimodal outer `Qwen4ExpConfig`, which carries only the image
ids; the text fields live under `text_config`. A checkpoint that spells the GDN counts at the
root therefore gets `Qwen4ExpTextConfig`'s defaults - 16 key heads, 32 value heads - and
`_reorder_v_heads` reshapes a 6-element `A_log` into `[16, 2, 1, 1]`. qwen4exp now registers a
raw `config.json` hparams loader, bypassing `AutoConfig` for this model type; `TextModel`
merges `text_config` afterwards, so the nested layout (the real one) and a flat text-only
layout both keep their own `linear_num_key_heads` / `linear_num_value_heads`. The indexed-layer
spelling is accepted as `full_attention`, `indexed_attention` (transformers 5.8's name) or
`qwen_sparse_attention` (5.17's canonical name after the text config's `__post_init__`
rewrite), so compress ratios are written for QSA layers in the real layout too. Evidence: `code` and the transformers source
(`transformers/models/qwen4_exp/configuration_qwen4_exp.py`, fetched 2026-11-23: defaults 16/32,
`model_type = "qwen4_exp"`, `sub_configs_defaults.text_config = Qwen4ExpTextConfig`);
Flash-Next's `text_config` at `de4b8e4d43` has 16/48/128, Qwen3.5-9B's 16/32/128.

The converter keeps the raw-config path for the text model and should: the outer
`Qwen4ExpConfig` is vision+text and its text class enforces `num_experts > 0` and drops keys
it does not declare (`intermediate_size`, `mlp_only_layers`, `decoder_sparse_step`,
`mtp_num_hidden_layers`), which a dense text-only model needs. The fixture's config is still
authored so transformers accepts it: integer `image_token_id`, `video_token_id`,
`vision_start_token_id`, `vision_end_token_id` inside the fixture vocabulary, a minimal
`vision_config`, and a strict-valid text config - `num_experts` 512, `num_experts_per_tok` 1,
positive `moe_intermediate_size`/`shared_expert_intermediate_size`, `indexer_kv_heads` 1,
`indexer_budget` divisible by `indexer_compress_ratio`, and `full_attention` entries. Every
trunk layer is in `mlp_only_layers`, so the runtime still selects the dense path for all of
them; a checkpoint with no `.mlp.experts` tensors makes the converter pad the MTP block dense
too (Flash-Next has expert tensors, so its MTP stays MoE). This round-trips through
`AutoConfig` under transformers 5.17 as well as through the raw path.

The fixture now writes that nested layout (the real checkpoint's `full_attention` spelling),
and its GDN ratio is 3 (2 key / 6 value heads at 32), the Qwen4 27B's reported 16/48 ratio. `tools/qwen4-v-reorder-check.py` asserts the
permutation against an explicitly built reference index at ratio 3 (fixture and Flash-Next
16/48) and at the pre-existing ratio 2 (Qwen3.5-9B 16/32); with torch it runs the converter's
own `_reorder_v_heads` at all three and fails unless the ratio-2 output equals the old
permutation. Without torch it checks the numpy reference and says the converter half is
skipped.

The torch run found the fixture's own writer missing the converter's 1-D reorder: `ssm_a` (the
converter's `A_log`) and `ssm_dt.bias` are per-V-head tensors, and the converter reorders them
as `(num_v, head_dim=1)` like `in_proj_a`/`in_proj_b`. The llama.cpp GDN graph consumes the
tiled V order that the converter emits (the same order `in_proj_z` and `out_proj` carry), so the
fixture writer now applies the same reorder to `A_log` and `dt_bias`; the earlier 1.4-1.7 max
diffs were permutations, not rounding. The compare tool now prints a `FAIL values ...` line
with the element count and only prints the one-ulp note when the tensor is within tolerance.
`qwen4exp.vocab_size` is removed from the fixture GGUF: the generic loader reads `n_vocab` from
`tokenizer.ggml.tokens` when the key is absent, and the converter does not write it for a GPT-2
vocab. `CONVERTED_GGUF=<converted.gguf> tools/qwen4-dense-fixture-check.sh` now runs the
loader-level gate: the fixture's logits saved by `llama-perplexity`, then the converter's GGUF
with `--kl-divergence` against them, requiring mean KLD 0 and same top p 100.

The fixture also ships a tokenizer, so `Qwen2Model.set_vocab` runs without an override: a
minimal byte-level BPE (`tokenizer.json`, `tokenizer_config.json`) with the 256 GPT-2 byte
tokens and no prefix space. It carries one merge, joining byte 0 and byte 1, so llama.cpp finds
the merges key a GPT-2 vocab requires; no text contains that pair, so encoding stays one token
per byte and `get_vocab_base_pre` sees the same sequence; the patch adds its `chkhsh` branch
mapping to the `gpt-2` pre-tokenizer (the only structural match in llama.cpp). The merged
string is a normal vocab entry at the last id, so it stays out of the emitted token ids. This
branch is for the synthetic fixture alone - a real Qwen4 tokenizer keeps its own pre hash.
The fixture GGUF now writes the same GPT-2 vocab and merges, so the converter's GGUF and the
fixture's GGUF tokenize `probe.txt` identically and can be compared at the logits; tokenizer
keys themselves stay outside the comparison.

**Evidence class**: `code` for the converter change, `measured-here` for the torch run (0163).
That run first found the MTP assert before any file was written; this patch removes it, and the
fixture gained its tokenizer for the run. The 0163
fixture writes the same checkpoint as safetensors + `config.json` + tokenizer for that run, and
its GGUF is written through gguf-py's qwen4exp tensor mapping with the same transforms as
`modify_tensors`, so the loader test exercises the names and shapes the converter produces. The
two commands are in 0163; a torch host runs them on the no-MTP and MTP variants.

## 0163-qwen4exp-dense-fixture.patch

No dense Qwen4 weights exist, so the plumbing is built against a synthetic fixture:
`tools/qwen4-dense-fixture.py` writes random weights with numpy (no transformers model classes,
no `trust_remote_code`, no Python model loader) at scaled-down published ratios - hidden 512,
Q 12 : KV 2 heads at head 32 (the 24:4 ratio), HC 4 streams with low rank 64 (640/5120), GDN
key value 2:6 heads at 32 (16:48), QSA on layers 2 and 4, layer 4 deliberately without indexer
tensors, a dense SwiGLU MLP of 1408 on all six layers, no PLE keys - as safetensors +
`config.json`, a minimal byte-level BPE tokenizer, a `probe.txt` for perplexity, and a GGUF.
A fixed seed makes it reproducible.

`config.json` is the HF layout: outer `model_type` `qwen4_exp`, the text fields under
`text_config`, as the real checkpoint has them. `--mtp` writes the MTP variant: `mtp_num_hidden_layers` 1 in the text config and one nextn block in
the qwen35 / Flash-Next layout - a full-attention QSA layer at `mtp.layers.0`, `fc_embedding` +
`fc_hidden` (concatenated to `eh_proj` as the converter does), `pre_fc_norm_embedding` /
`pre_fc_norm_hidden`, and a `hyper_connection_mixer` with the down/up pair. The GGUF gets
`block_count` 7, `nextn_predict_layers` 1 and one extra indexer compress ratio. It also
carries `mtp.norm`, the qwen35-style `shared_head_norm`: 0161 adds that tensor to the arch's
list and loads it optional (`TENSOR_NOT_REQUIRED`) because a Qwen4 checkpoint may ship it, but
`graph_mtp` does not consume it - the head norm the graph uses is `nextn_hc_head_norm`. The
check asserts both are loaded.

`tools/qwen4-dense-fixture-check.sh` runs the V-head reorder test, generates the fixture, runs
`llama-perplexity` on the CPU and asserts the indexer-fallback log line, the dense-MLP classification and a finite PPL. It
runs three arms at `measured-here`, CPU, `-c 128 -b 16 -t 2`: the mixed one above (PPL
292.9699) and `--no-qsa`, a checkpoint with no indexer keys and no indexer tensors at all
(dense attention everywhere, PPL 297.9606), and `--mtp` (trunk PPL 292.9699; the same trunk
weights). RED on the pre-0161 build (integer divide by zero before the loader reaches the
assert); GREEN after it. The mixed arm runs one dense-attention layer without an indexer
beside a QSA layer, and no PLE, in the same forward; the no-QSA arm exercises the absent key
group and the absent cache. The MTP arm loads the nextn tensors (`n_layer_all = 7`) and
`llama-cli --spec-type draft-mtp` builds and runs the in-model MTP graph (`mtp_tok_embd-6`,
`mtp_h_input` nodes), which perplexity does not exercise; `nextn.shared_head_norm` loads
optional when the checkpoint carries it.

**Converter comparison** (a torch host, CPU only, after `convert_hf_to_gguf.py` runs end to end -
no `set_vocab` override). The reorder check first, then both variants:

```
python3 tools/qwen4-v-reorder-check.py

python3 tools/qwen4-dense-fixture.py --out <dir> && \
python3 convert_hf_to_gguf.py <dir>/hf --outtype f32 --outfile <dir>/converted.gguf && \
python3 tools/qwen4-dense-fixture-compare.py <dir>/qwen4-dense-fixture.gguf <dir>/converted.gguf
CONVERTED_GGUF=<dir>/converted.gguf bash tools/qwen4-dense-fixture-check.sh

python3 tools/qwen4-dense-fixture.py --out <dir-mtp> --mtp && \
python3 convert_hf_to_gguf.py <dir-mtp>/hf --outtype f32 --outfile <dir-mtp>/converted.gguf && \
python3 tools/qwen4-dense-fixture-compare.py <dir-mtp>/qwen4-dense-fixture-mtp.gguf <dir-mtp>/converted.gguf
```

`qwen4-dense-fixture-compare.py` checks the tensor name set, every shape, every value within
1e-5 (bit-exactness is reported but not required: torch and numpy libm may differ by one ulp),
and every `qwen4exp.*` key; tokenizer keys are excluded by design.

`measured-here`, a torch host (CPU only, transformers 5.17): both variants convert end to end and
match the fixture's GGUF: the plain one 123 tensors and 31 `qwen4exp.*` keys, the MTP one 151
tensors and 32 keys, every name and shape equal and every value within 1e-5. The V-head reorder
check passes at ratio 3 (the fixture and Flash-Next's 16/48) and keeps the earlier permutation at
ratio 2 (Qwen3.5-9B's 16/32).

## 0164-test-backend-ops-dense-qwen4-shapes.patch

Four real-shape groups are added to the eval (and where the GDN cases sit, the perf) list:
- HC `4 x 5120` with low rank 640, 1 and 4 tokens;
- QSA attention, 24 query heads on 4 KV heads at head size 256, over the selected 2048 cells,
  1 decode row and 6 verify rows;
- GDN 16 key/value heads at 128 with 48 V heads (v_repeat 3), 1 and 4 tokens;
- a dense SwiGLU FFN at 5120 hidden with intermediate 14336 and the disputed 17408, gate/up and
  down, a 4-token activation.

`measured-here`, CPU: `HC_MIX` 2/2, `GATED_DELTA_NET -p head_count=16` 21/21 (the group's other
cases included), `FLASH_ATTN_EXT -p kv=2048` 26/26, and the four FFN `MUL_MAT` cases 4/4 pass. `measured-here`,
card (B60, OpenCL): `HC_MIX` 2/2, `GATED_DELTA_NET` 21/21, `FLASH_ATTN_EXT` 26/26, the FFN
`MUL_MAT` groups 21/21 and 2/2, nothing unsupported. The commands:

    LLAMA_BIN=<opencl build> ./bin/test-backend-ops -b <card> -o HC_MIX -p 'n_embd=5120'
    LLAMA_BIN=<opencl build> ./bin/test-backend-ops -b <card> -o GATED_DELTA_NET -p 'head_count=16'
    LLAMA_BIN=<opencl build> ./bin/test-backend-ops -b <card> -o FLASH_ATTN_EXT -p 'kv=2048'
    LLAMA_BIN=<opencl build> ./bin/test-backend-ops -b <card> -o MUL_MAT -p 'm=(14336|17408)'
    LLAMA_BIN=<opencl build> ./bin/test-backend-ops -b <card> -o MUL_MAT -p 'm=5120,n=4,k=(14336|17408)'

## 0165-qwen4exp-moe-frozen-check.patch

The serving qwen4exp (Flash-Next) path is frozen: its MoE behaviour and its logits must stay
bit-identical. `tools/qwen4exp-moe-frozen-check.sh` takes `BASE_BIN_DIR` (the pre-change build)
and `NEW_BIN_DIR`, saves a synthetic qwen4exp MoE model at a fixed seed with the new build,
writes the base build's logits and compares the new build's logits with `--kl-divergence`. The
gate is mean KLD 0 and same top p 100 %.

`measured-here`, CPU, pre-0161 against this tree: mean KLD 0.000000, same top p 100.000, PPL
126.5492 both (evidence: the per-layer rule leaves every MoE layer's `ffn_gate_inp` path and its
indexer cache in place). The real Flash-Next GGUF's CPU arm (and its KL) is still owed.

`measured-here`, card (B60, OpenCL) against the CPU, the fixture's logits at `-c 128`: the mixed
fixture mean KLD 0.000137 (same top p 97.18 %) at the default batch and 0.000133 at `-ub 1`;
`--no-qsa` 0.000001 and 0.000000; the MTP variant's trunk equals the mixed fixture's. The fixture's
weights are F32 at hidden 512, so its few fallback lines (small-K F32/F16 products) are the
fixture's shapes, not the 27B's. The commands, at the published dense shapes and the context the
unit serves:

    # fixture CPU vs OpenCL logits (KL), the same file on both arms
    tools/qwen4-dense-fixture.py --out <dir>
    <cpu build>/bin/llama-perplexity -m <dir>/qwen4-dense-fixture.gguf \
        -f <dir>/probe.txt -c 128 -b 16 --save-all-logits <dir>/cpu.logits
    <opencl build>/bin/llama-perplexity -m <dir>/qwen4-dense-fixture.gguf \
        -f <dir>/probe.txt -c 128 -b 16 --kl-divergence \
        --kl-divergence-base <dir>/cpu.logits
    # the shapes on the card: the 0164 commands above, and `perf` for the same -o/-p sets
    # the memory line for the HC model at the served 262,144 context: the load's KV cache,
    # compute buffer and model MiB, plus the guard's figures (the card's used/total, the
    # process's fdinfo vram0 and GTT) at `--n-ctx 262144`; the real GGUF is owed


## 0167-opencl-q8_0-mul_mat_id-prompt-kernel.patch

Inkling 2 Mini's four coda layers (20-23) keep their experts in Q8_0 while layers 0-19 hold
IQ4_NL: gate_up K=2304 -> 2 x 2304 rows and down K=1152 -> 2304 rows, 128 experts, top-8.
0156 serves their **decode** batch (`n_tok < 16`) on the int8 dot; a **prompt** batch had no
GEMM and fell to `kernel_mul_mv_id_q8_0_f32_flat`, the flat float matvec — one float dot per
value, at 512 tokens about 800 ms a layer for gate_up against 87 ms for an IQ4_NL layer on the
same card (`measured-here`, B60, 23.2 % of a pp4096 ub512 profile).

0167 adds `kernel_mul_mm_id_q8_0_i8` to `mul_mm_kq_i8.cl`, beside the IQ3_S and IQ4_NL kernels
of 0071: one work-group takes I8XT = 32 route pairs of one expert and I8XSG = 8 sub-group
groups of 16 output rows, stages the pairs' int8 activations a 256-weight super-block in local
memory, and runs the s8 8x16x32 DPAS a 32-block per 8-pair group, K ascending, scaled by the
weight block's `d` and the activation's per-32 scale — the IQ4_NL kernel's body and macros
(`I8_COLS`, `I8_STAGE`, `I8_DPAS`, `I8_STORE`) unchanged. The difference is the weight read:
a Q8_0 block holds its 32 codes as s8 already, so the b operand is the flat plane's 32 bytes
read as eight words (no decode, no grid table, no K super-block), and `ws` is the block's half
scale. The block's index is `(expert * ne01 + row) * (ne00 / 32) + kb / 32`, the flat plane's
layout 0156's decode kernel uses.

`kq_i8_ops.inc` gains `kq_i8_q80_wanted(GGML_OPENCL_Q8_MM_ID)`: unset is **on** (a Q8_0 prompt
MUL_MAT_ID has no other GEMM), `0` and `off` (case-insensitive) turn it off. The Q8_0 decode arm
is not gated by it. The program gate is the union — the int8 program is built when either the
K-quant arm's `GGML_OPENCL_KQ_MM_ID_I8` or this arm wants it, so the default path pays the
program compile a unit start. One limitation stays: the kernel is the SG16 s8
DPAS, so it runs on Intel only (Adreno's SG8 arm is 0156's and the K-quant tile's).

## 0168-opencl-q8_0-mul_mat_id-prompt-dispatch.patch

`ggml_cl_mul_mat_id_q8_0_prompt` takes a prompt batch the way the IQ experts take theirs:
`kernel_moe_route_tiles` sorts the (token, slot) pairs by expert into 32-pair tiles (the IQ
arm's route, and its per-ids cache), `kernel_kq_mm_id_quant_i8` quantizes src1's columns to the
q8_1-style int8 form the tile kernel stages (one activation a column — the frozen gate/up
broadcast serves one column a token, not one a pair), and `kernel_mul_mm_id_q8_0_i8` runs the
tile. Guards: `n_tok >= 16`, ne12 == n_tok, src0's ne3 == 1, ne00 a multiple of 32, ne01 a multiple of 16,
at most 512 experts (the route kernel's local arrays), contiguous, the flat Q8_0 SoA planes, not an expert cache's `.slots` / `.bank`, and the int8 kernels built.
Anything else keeps the generic arm, so the card's fallback is exactly today's.

The dispatch sits after 0156's decode arm in `ggml_cl_mul_mat_id`; 0156's function and its
guards are byte-identical.

## 0169-opencl-q8_0-mul_mat_id-prompt-tests.patch

Six test-backend-ops `MUL_MAT_ID` cases in the coda shapes: 128 experts, top-8, 2304 rows, the
gate/up K 2304 with a broadcast b and the down K 1152 with a column a pair (ne11 = 8), at 16
tokens (the arm's floor), 512 and 2048 (the prefill chunks). They are in the eval list; the perf
list keeps 0156's decode cases, which stay as they were.

CPU: `test-backend-ops -b CPU -o MUL_MAT_ID` 1038/1038, the Q8_0 arms 84/84 and the six new
cases OK; `test-opencl-kq-i8` is all OK device-free.

`measured-here`, card (B60 at its 165 W cap, Inkling 2 Mini IQ4_NL with the Q8_0 coda):
- `MUL_MAT_ID` 542/616, the 74 failures all `mxfp4` (known, before this patch); the red build
  (the kernel's `ws` scale dropped) fails 47 Q8_0 cases.
- `llama-bench -fa 1 -p 4096 -b 4096`, two rounds: `-ub 512` 410.6 -> 522.4 t/s (+27.2 %). The
  prompt profile's Q8_0 MUL_MAT_ID share falls from 23.2 % on the flat matvec to 2.1 % on the tile.
- KL against the CPU reference: 0.001952 (same top p 98.50 %) against the baseline's 0.001950
  (98.34 %) at the default batch; 0.001988 at `-ub 1`, identical to the baseline (the decode
  batch does not take this arm).

## 0170-opencl-f32-wide-n-gemm.patch

Two of Inkling 2 Mini's three generic-matvec shapes go to the tiled F32 GEMM
(`kernel_mul_mm_f32_f32_l4_lm`) that `ggml_cl_mul_mat_f32_skinny` already runs: the relative
bias table (K 16, 1025 rows, a 2-D weight over a 3-D src1 whose ne[2] is the tokens) and the
controller pairs (K 64, 2304 rows, 512-2048 columns). The GEMV arm's row bound (256 above 8
columns) is a one-column-shape bound, so the GEMM branch takes any row count; K need only be a
multiple of the kernel's BK 16 (a single block at K 16, four at K 64, ks stays 1, no slice
sum). A batched src1 keeps one K slice and advances a batch per 64-column block: the kernel's
`i13 / r3`, `i12 / r2` split gets r2 = b's batches and r3 = 1, and `batch_stride_b` becomes a
batch's columns (`ne00 * ne11`) where it was a slice's K. The f16-copy GEMM (0110) declines
these: it requires a 2-D src1, and it would multiply through an f16 copy of the weight, where
the l4 path is exact F32. Shapes the arm already accepted are unchanged: for them `kslice` and
`ne00 * ne11` coincide only when ks == 1 and the batch is flat, and the 2-D case keeps
`kslice`.

## 0171-opencl-f32-tiny-k-batched.patch

The depth mix's second product — src0 an activation [K <= 8, 2304, tokens], src1 the scores
[K, 1-3, tokens] — fits none of the existing paths: the tiled GEMM needs K a multiple of BK 16,
the GEMV arm K % 64, and the f16 copy a 2-D src1. `kernel_mul_mat_f32_f32_smallk` takes it
batched: a lane a row (a 64-row block), the K loop (1-8) in registers, one work-item a column
of a batch, the batch on the second dimension. `GGML_OPENCL_F32_SMALLK=0` turns it off and
leaves the generic matvec.

## 0172-opencl-f32-wide-n-tests.patch

test-backend-ops at the graph's shapes: `MUL_MAT` f32/f32 1025 x 512 and 2048 columns at K 16,
2304 x the same columns at K 64, and a new `test_mul_mat_f32_wide` (a 2-D weight, a 3-D b) at
2304 rows, K 1 / 6, one or three columns a token, 512 and 2048 tokens. CPU: 8/8 OK, the CPU
MUL_MAT suite unchanged; the OpenCL build compiles and the matvec kernel file passes `cpp -P`.

## 0173-expert-cache-projected-bytes.patch

The spill guard's cache suggestion needs on-card bytes for a flag it has not started. The
cache's own accounting is here: `llama_expert_cache::project_device_bytes(budget)` replays the
init's slot plan over the stored ranking, recomputes the lend set at that budget, and sizes the
slot buffer, Q8_0 banks, tables, lend tail, the candidate layers' spare buffers and the gather
mirror with `ggml_backend_alloc_ctx_tensors_from_buft_size` on the cache's own buffer type, the
same functions load uses. No allocation is made. `llama_model_expert_cache_projected_bytes`
exposes it for one cache. arcint walks down from the refused flag (coarse steps, then refined upward) to the largest flag
whose projected bytes fit the free bytes beside the other terms; the projection is not
monotonic at low budgets, where a Q8_0 bank holds every expert. The suggestion is a flag value,
so following it starts. `ARCINT_GUARD_TRACE=1` logs every probe (flag, projected bytes, free
bytes, verdict).

`measured-here`, B70, Flash-Next on its unit's flags with `--llama-expert-cache 40000`: refused,
the projection at 40,000 is 40,409 MiB (the load's own figure), the suggestion is 19,107; the
restart with 19,107 starts (cache 19,437 MiB on the card) and measures 32,402 of 32,656 MiB after
the first request, against 32,192 predicted.

## 0175-opencl-f32-tiny-k-batched-guard.patch / 0176-opencl-f32-tiny-k-batched-tests.patch

A card profile put [2304, 1|3|4, 512] of the depth mix's second product on the generic F32 matvec:
1,408 + 384 + 128 calls at 1.078 / 1.667 / 1.498 ms, 4.7 MB of output a call, 22-75 GB/s against
the card's 456. The arm declined every one of them: it compared src0's rows (ne1 = 2304) with
dst's *columns* (ne1 = 1..4) instead of dst's ne0. Corrected, with the tensors' own strides
(src0's K and row steps, src1's, dst's) so a transposed src0 view is in reach, and one lane a
row, the K loop in registers, a column a block of 64 lanes. At decode the shape [2304, 1, 1] is
one launch a mix and launch-bound, so a faster kernel does not help there — the launch count
does, and the transposed copy is the launch to fold: `ggml_cont` of a `ggml_transpose` feeding
this product. A CPU reference refuses a transposed mul_mat src0
(`GGML_ASSERT(!ggml_is_transposed(a))`, seen locally), so the fold has to be the backend's own
read and the graph keeps the copy for now; the fusion of CONT(TRANSPOSE(x)) into this arm is the
way to take it without a copy.

0171 shipped the tiny-K arm on when unset and 0170's wide-N GEMM unconditional; 0175 put them
behind `GGML_OPENCL_F32_SMALLK=1` (the mix shapes) and `GGML_OPENCL_F32_WIDE_N=1` (the K 16 / 64
wide-N ones), and 0180 then made the tiny-K and small-M arms default on. `GGML_OPENCL_F32_WIDE_N` stays default **off** — a card run showed no
measurable gain from it, so the default tree is the 0175/0176/0178 build on, not 0170's. The
wide-N GEMM no longer takes shapes the old selection already served (its row bound, or K a
multiple of the GEMV's 64), and its slice sum runs only above one K slice: unconditional, it
copied 12,595,200 elements a call, 218 ms of a prompt window, and gained nothing.

The permuted `kernel_cpy_f32_f32_pack` of [24, 1025, 512] (1.33 ms a call, ~75 GB/s each way) is
producer `ggml_cont(ggml_permute(table, 1, 0, 2, 3))` after the relative-bias table product and
consumer `ggml_get_rows(table, rel_pos)`: the permute exists because the gather walks the
distance axis, which get_rows only does on ne1, so folding it needs a gather that walks ne0, not
a GEMM read.


## 0177-opencl-f32-tiny-k-batched-batch-stride.patch

A card run of 0175/0176 with `GGML_OPENCL_F32_SMALLK=1` gave KLD 10.38 at the default batch (top-p
agreement 0.07 %) and op `MUL_MAT(m=2304,k=6,n=1..4,nb=512,n_slot=8)` ERR ~2.0 — the four shapes
whose src0 is a 3-D view over the depth-mix stack; the plain 2-D ones and the decode shape
(nb = 1) passed. Cause: a 3-D src0 is a slab a token at its own nb[2] (the view keeps the stack's
stride), and the kernel's address had only the K and row steps, so every column read slab 0. The
batch step is an argument now (0 for a shared 2-D weight), and the arm requires a contiguous 3-D
src0 so a slab's step is the tensor's own. The CPU cannot show it: the arm is Intel OpenCL only
and the CPU reference computes the batched product with its own kernel, so only a device run (or
the op test's CPU comparison on the device) sees the layout.

The card then put the n=3 and n=4 cases on the fault: `MUL_MAT(m=2304,k=6,n=3,nb=512,n_slot=8)`
read out of bounds ("Engine memory CAT error" and a timed-out job, `clWaitForEvents -14`), which
is 0179's subject; n=1 was correct from 0177 on.

## 0178-opencl-f32-split-k-gemv-batched.patch / 0179-opencl-f32-tiny-k-batched-column-stride.patch

Two independent F32 arms, both behind their own knob and both off by default.

0178 is the depth mix's **first** product: `scores = mul_mat(states, queries)` in
`src/models/inkling2.cpp:216`, src0 the `rms_norm` of a `view_3d` of the bank stack (K 2,304 on
ne0, M the block count 2 on ne1, the 512 tokens on ne2), src1 `msg_query` [2,304, 4] repeated over
the tokens, dst [2, 4, 512]. The profile put `kernel_mul_mm_f32_f32_l4_lm` at 128x1x512 for exactly
this dst: the tiled GEMM's column tile is 64, so a call ran 512 batches of a 64x64 tile holding
8 outputs, each work-group the whole 144-block K loop — 2.0 ms a call, 384 calls, 4.9 % of a
prompt window (measured-here, B60, pp4096 ub 512). `kernel_gemv_f32_f32_splitk_batch` is its own
layout: one work-group a (token, column, row), the K split into 16-aligned slices and summed. The
split is **36, a slice 64** floats: 2,304 = 36 * 64, and a 64-way split would leave 36-long slices,
off the 16 alignment, so the largest aligned slice is 64. Target: 9.44 MB of src0, 18.87 MB of
src1 (4.72 MB unique; the repeat is materialized), 16 KB of dst — 28.3 MB, **~71 us at 400 GB/s**
against 2.0 ms. `GGML_OPENCL_F32_SMALL_M=1` turns it on; unset keeps 0175-0177's selection.

0179 fixes the tiny-K arm's OOB: a batch's columns were flattened as (token, column) and b and
dst were stepped by the flattened index, so a token's slab step was multiplied by the column
count (`dst[c*rd]`, `src1[c*rb]`, and src0's slab by `c*ba`). n=1 has one column a token and was
correct; n=3 and n=4 read src0 3x and 4x past its buffer (src1 and dst are contiguous, so their
flattened index stayed in range). The column is split into its extents now
(`nc1 = src1->ne[1]` binds it, b and dst advance a token by their own nb[2], src0's slab by the
token index), so every read stays inside.

The red series cannot reach the K=1 case by construction: its mutation (0175/0177's K sum cut to
its first term) is the *whole* sum when K=1, so `MUL_MAT(m=2304,k=1,n=1,nb=512,n_slot=8)` passes
there and only the K>1 cases fail. What failed the K=1 case on the card was 0177's stride defect
— the slab-0 read — at ERR 1.999, which is why 0177's own red is the right one for it.

Both arms also compute, before the enqueue, the furthest byte each kernel can reach from the
global size and the strides handed to it (a view's offset included) and compare it with the
allocation of the backend buffer the tensor lives in (`CL_MEM_SIZE` of its `data_device`):
`GGML_OPENCL_BOUNDS=1` aborts a violation, and the check is what an op run on the card exercises
first. It catches a read or write past the allocation, not one into a neighbouring tensor of the
same buffer; in the op cases below each tensor is its buffer, so there the max byte equals the
buffer exactly. 0170's wide-N arm (off by default) has no such check:

0179's cases (the mix layout, src0 a contiguous copy of the transposed slice, k a block count,
n a query count, nb 512 tokens, n_slot 8):
- k=6, n=1: src0 6 * 2304 * 512 * 4 = 28.31 MB, src1 1 * 6 * 512 * 4 = 49.2 KB, dst
  2304 * 1 * 512 * 4 = 4.72 MB — max bytes 28,311,552 / 49,152 / 4,718,592, each equal to its
  buffer;
- k=6, n=3: src1 3 * 6 * 512 * 4 = 147.5 KB (max 147,456), dst 2304 * 3 * 512 * 4 = 14.16 MB
  (max 14,155,776);
- k=6, n=4: src1 4 * 6 * 512 * 4 = 196.6 KB (max 196,608), dst 2304 * 4 * 512 * 4 = 18.87 MB
  (max 18,874,368);
- k=1, n=1: src0 1 * 2304 * 512 * 4 = 4.72 MB (max 4,718,592), dst 4.72 MB.
0178's shape (k 2304, m 2, n 4, nb 512): src0 2 * 2304 * 512 * 4 = 9,437,184, src1
4 * 2304 * 512 * 4 = 18,874,368, dst 2 * 4 * 512 * 4 = 16,384, partials 36 * 4096 * 4 = 589,824 —
each is its buffer's size, read and written to its last float.


**Card, measured-here (B60, Inkling 2 Mini IQ4_NL, the series through 0179):** op
`MUL_MAT` green 2,980/2,980 with `GGML_OPENCL_BOUNDS=1`; the red series fails 6 of the 7 targeted
cases, and the kernel log shows no fault either way. pp4096 at ub 512: 0167-0169 alone 522.5 t/s,
`SMALL_M` 538.1 (+3.0 %), `SMALLK` 606.9 (+16.2 %), both 628.2 (+20.2 %), both with the int8
expert arm 676.6 (+29.5 %); ub 1024 with both and the int8 arm 712.2. Decode at d4096 36.0 ->
36.4 t/s. KL against the CPU reference with both arms: default batch 0.001979 (0167-0169's baseline
0.001952), `-ub 1` 0.001949. 0178's kernel reaches ~1.6 ms a call — its gain is mostly the
`l4_lm` node it removes, not the new arm's own throughput.

**0178's 1.6 ms a call:** the tensors are 28.3 MB (71 us at 400 GB/s), the traffic 75.5 MB (189 us).
The launch is 36 slices x 4,096 outputs, 147,456 work-groups of which 16 lanes work at a 64-float
slice. Each work-group reads 256 B of an src0 row and 256 B of an src1
column and writes one float, so the traffic is 75.5 MB — src0 four times (one per query column,
9.44 MB each time) and src1 twice (one per slot row, 18.87 MB each time) — and a call is 921
work-groups a core in 1.6 ms, 1.7 us a group. The time is the barrier tree, not the bytes: four
steps of the 16-lane reduction and a store per work-group, 36 work-groups a output. The next
step, not written: one work-group a token, its eight outputs (2 rows x 4 columns) as four
subgroups of 16 lanes with the K loop inside the subgroup and no split-K or slice-sum launch —
a token's src0 2 x 9 KB read four times and its src1 4 x 9 KB twice, the same 75.5 MB at a
barrier-free reduction. Beyond that the repeat is the observation: src1's four query columns are
the same rows for all 512 tokens (18.9 MB of repeated data, 4.7 MB unique), and reading one slab
would need the graph to hand the arm the 2-D parent, which the dispatch cannot see.

## 0180-opencl-f32-arms-on-by-default.patch

`GGML_OPENCL_F32_SMALLK` and `GGML_OPENCL_F32_SMALL_M` are on unless set to `0` (`off` also works);
`GGML_OPENCL_F32_WIDE_N` stays off, its arm having shown no measured gain (0170/0172 stay in the
series because 0175, 0176 and 0178 are built on their hunks). Unset, the tree takes 0175-0179's
arms and 0167-0169's dispatch otherwise.

Evidence that no other production model reaches the arms (`measured-here`, the arms on against off
on the same card, `GGML_OPENCL_BOUNDS=1`): Qwen3.8 27B on the B60, mean KLD 0.000000 between the
two runs, same top p 100 %; Flash-Next IQ3_S on the B70 (experts on the CPU), 0.000000 and 100 %;
no bounds abort. That matches the code: both arms take only F32 x F32 products, and no node of
those models has K <= 8 (SMALLK) or a 3-D activation on both sides with at most 8 rows and columns
(SMALL_M); their F32 tensors are 2-D weights with K of 128 and more, or 1-D scales.

## 0190-opencl-q8_iq4-mul_mat_id-wide-tile.patch / 0191-...-wide-dispatch.patch / 0192-...-wide-tests.patch

The int8 expert tile (0071, and 0167-0169's Q8_0 prompt arm on it) runs one work-group a (route tile, 128-column band), and a launch is
`CEIL_DIV(ne01, 128)` bands deep: 18 for a 2,304-row expert, 36 for one of 4,608. Every one of those
work-groups stages the same tile's 32 pairs of int8 activations into local memory, so a tile's block
is filled 18 times over. The wide tile takes a 512-column band a work-group instead, 16 sub-groups
of 16 lanes with two columns a lane, 5 bands over 2,304 rows: the staged block is filled once for
512 columns, and the launch has 5 bands for the narrow tile's 18.

The model behind it, from the card's per-launch fit (`measured-here`, profiling build, Inkling, I8
on, pp4096, 44 IQ4_NL visits, 8 chunks at ub 512 = 352 launches a kernel): gate/up `0.71 ms + 6.1
us a token` and down `0.21 ms + 3.6 us a token`. The floor is launch and empty-tile work (~324 ms a
run at ub 512, ~80 at 2048); the slope is per-tile work, and a real 32-pair tile is ~1.35 us at
gate/up and ~0.8 us at down for 248 KB and 124 KB of traffic — weight blocks read a tile once, the
staged activations read once a band, 18 times a tile. Down's ratio to the reference is worse (1.72x
a visit against 1.23x) because its staged activation is a larger share of a work-group's bytes: the
fill's 16-byte loads step 1,152 bytes apart across 32 pairs, 4 sectors to a useful read, while the
reference's down reads its activation contiguous (trellis-serve, `xpu/csrc/exl3_moe.sycl:559`
`DownDpasKernel`, `ablk` `[b][r][MB][16]`, and `:432` `GateUpDpasKernel`, `acc = NT*MB*16` at `:466`,
`MB` 32 at `:795`, `T1` 72 at `:807`, @8f8e2fcc, MIT). The wide tile attacks both terms: bands
18 -> 5 per tile, so the staging slope falls and the empty/launch floor falls with the work-group
count. The predicted numbers it has to meet at ub 512 are gate/up 1,378 -> ~1,050 ms and down 742 ->
~510 ms a run; at ub 2048, gate/up 1,149 -> ~870 and down 671 -> ~470 (the slope coefficient is the
same a token at both, the floor differs).

Register budget: a lane holds two columns by 32 pair slots = 64 float accumulators, 64 GRF, plus the
two weight operands (8 words each), two scales and the staged-loop addresses, about 90 GRF of the
256 the program is built with (`-cl-intel-256-GRF-per-thread`, as the narrow program). The whole
2,304-row band would be 9 columns a lane, 288 floats, over the budget, and 4,608 rows 576; that is
why the band is 512 rather than the expert, and the split is the knob's geometry. Whether IGC spills was not
read from a build log. On the card the kernel times land above the prediction by 11 % (gate/up) and 17 % (down) at ub 512
and by 7 % and 3 % at ub 2048, not the several-fold a
rolled K loop gives (inference).

The tile is its own program (`mul_mm_kq_i8_wide.cl`, the helpers and the mm tables prepended as the
narrow one's), so with `GGML_OPENCL_KQ_MM_ID_I8_WIDE` unset nothing new is compiled and both int8
paths — the IQ4_NL arm and Q8_0's prompt arm — take the narrow tile and grid exactly as today. The
knob (`1` or `on`) reaches both. Both kernels keep every buffer the same and add no scratch: the
route's pairs, tiles and tile count, the int8 activation buffer and its scales, the weight planes and
dst. Under `GGML_OPENCL_BOUNDS=1` a wide launch checks the furthest byte of each of them
(`ggml_cl_bounds("int8 wide", ...)`), including `blkmax` over the last expert's last row and last K
block (on the planes or the stage the kernel reads; see below for a USM bank), `n_cols * ne00` on the activation, `max_tiles * sizeof(cl_int4)` on the tiles, and
`offsetd + (n_tok-1)*nb2 + (n_used-1)*nb1 + ne01*4` on dst.

Licence: the reference file is MIT; no line of it is copied here — the geometry is followed, the
code is this tree's OpenCL form of the narrow tile — so no NOTICE entry is owed. A later port that
copies lines would owe the MIT notice in the file header, a `THIRD_PARTY.md` row, and the repo and
commit (`trellis-serve` @8f8e2fcc) in the commit message.

Tests: `test-opencl-kq-i8` reads the knob's parse (unset/off/0 off, 1/on on) and assembles the wide
program as the backend does, text-checking it and compiling it when a clang exists. `test-backend-ops`
carries Inkling's real expert shapes, 128 experts top-8, at 512 and 2048 tokens. The correctness and perf lists carry
IQ4_NL gate/up as 2,304 rows at K 2,304, both projections in one tensor, 4,608 rows at K 2,304 (the 9-band launch), and
down as 2,304 rows at K 1,152 (4.5 super-blocks, so the staging loop's last group is partial); the 4,608-row and skew
cases are IQ4_NL only. 0192 adds 12 correctness cases: 10 IQ4_NL, 6 real-shape (the three shapes at two batch sizes)
and 4 skew (two shapes at two batch sizes), and two Q8_0 cases that take 0168's fallback (520 rows, not a multiple of 16;
520 experts, more than 512). The Q8_0 real-shape cases are 0169's six (two shapes at 16, 512 and 2048 tokens), so the
two patches carry 16 distinct real-shape cases and the two fallback cases. The perf list has six a batch size: the
three IQ4_NL shapes, two Q8_0 (2,304 rows K 2,304 and 2,304 rows K 1,152), timed next to the narrow tile, and one skew. It also carries
`test_mul_mat_id_skew`, every slot of every token on expert 0 (2,304 rows at K 2,304 and at K 1,152; the perf list has
the K 1,152 one), so one expert takes a whole chunk and
the rest are empty tiles. A tile is 32 pairs. Uniform top-8 over 128 experts gives ~32 pairs an expert at 512 tokens (one
tile) and ~128 at 2048 (four). The skew case gives 128 and 512 tiles to expert 0. The band loop, the last band's second
column past `ne01`, the partial super-block and the empty-tile tail are all reached. Red tree, the named mutation: in
`mul_mm_kq_i8_wide.cl`, `acc1[g] +=` becomes `acc0[g] +=` in `I8W_MAD`, so column 0 accumulates both products and column
1 stays zero.

Bounds under `GGML_OPENCL_BOUNDS=1` follow the weight arguments the kernel is given (`ggml_cl_kq_weight_args`): the SoA
planes with their offset, or the gathered stage an expert cache hands the kernel. The helper carries no branch for a
Q8_0 tensor stored as it came (not in SoA planes): unreachable (`code`), the wide int8 arm is IQ4_NL only (its
kernel exists only for type index 5), and the Q8_0 prompt arm returns before the call unless
`ggml_cl_is_q8_0_soa(src0)` holds, under `ggml_cl_mul_mat_id`'s `GGML_ASSERT(src0->extra)`. An expert bank in USM host memory
would be a pointer argument with no buffer size to compare against; the helper returns without bounding it and logs
nothing. For the wide tile that route does not occur (`code`): an IQ4_NL bank in USM is always gathered first, and the
Q8_0 arm refuses an expert cache's `.slots` and `.bank` tensors.

Card evidence (`measured-here`, the B60 at a 165 W cap unless the B70 (220 W) is named, Inkling 2 Mini IQ4_NL,
`GGML_OPENCL_KQ_MM_ID_I8=1` in every arm):

- Op cases with `GGML_OPENCL_BOUNDS=1`, on this revision built from the series itself (apply-series.sh on the pin; the
  applied tree's `git write-tree` is a5610fc1):
  - Inkling's 16 distinct real-shape MUL_MAT_ID cases (128 experts, top-8: 0192's 10 IQ4_NL, 0169's six Q8_0) pass 16/16
    with the knob on and with it off; without the knob, no wide program is built. 0192's two Q8_0 fallback cases (520
    rows; 520 experts) pass 2/2. The run logs the generic path once for their shared key (`DOWNGRADE src=opencl
    op=MUL_MAT_ID type=q8_0 k=256 path=generic reason=no-fast-path`, logged once a key), so a per-case path is not
    shown.
  - The MUL_MAT_ID suite gives 554 of 628 with the knob on; the 74 failures are the known mxfp4 set.
  - The red tree, knob on, fails all 16 (10 IQ4_NL, 6 Q8_0).
  - KL to the CPU reference (`-c 2048 --chunks 3`), knob off / on: ub 512 0.001932 / 0.001932 (same top p 98.208 %
    both), ub 1024 0.001917 / 0.001917 (98.403 % both).
  - No bounds abort in any of these.
  - The earlier rows (the 4,608-row single case 1/1, the suite 556 of 630 with four duplicate cases counted, and
    everything below) ran on a development worktree, not the applied series. It differed in two refusal conditions of
    0168 (`ne01 % 16`, more than 512 experts), which pass for every shape it ran, in patches 0161-0165 (the Qwen4
    converter and model files and their test cases), and in a comment.
- The kernel time, speed, served and logit rows below were measured on that worktree's earlier revision of these patches
  and were not re-measured on the applied series. Since
  then the bounds check follows the kernel's arguments and the test shapes were corrected. The kernel text changed only in its
  comments, and the dispatch only in the bounds helper (`code`).
- Kernel time, profiling build, pp4096, per visit (44 IQ4_NL visits a pass; the reference's figures are per visit over its
  48; the reference's are `measured-here` too: trellis-serve @8f8e2fcc on the same B60 at 165 W, a PTI profile over
  the same 4,096-token fixture), narrow -> wide:

  | ub | gate/up | down | the reference's gate/up / down |
  |---|---|---|---|
  | 512 | 31.32 -> 26.46 ms | 16.87 -> 13.56 ms | 25.4 / 9.8 ms |
  | 2048 | 26.12 -> 21.21 ms | 15.26 -> 11.03 ms | 20.3 / 7.5 ms |

  The prediction above was missed: gate/up landed at 1,164 / 933 ms a pass (predicted ~1,050 / ~870), down at 597 / 485
  (predicted ~510 / ~470). Gate/up is within 4-5 % of the reference a visit; down is 1.38x / 1.47x.
- llama-bench pp4096, knob off -> on: ub 512 676.5 -> 722.2 t/s, ub 1024 712.1 -> 765.7, ub 2048 672.4 -> 723.4.
- Served, the planned Inkling unit flags (262,144 cells, `--llama-swa-window`, q8_0 KV, `--prefill-ubatch 1024`), a fixed
  4,096-token prompt three times with `cache_prompt: false`: prefill 671.4 -> 718.4 t/s. Decode is unchanged: 35.17 t/s at a
  4,096-token depth, 38.94 at 0.
- Logits, knob on against off:
  - Inkling: mean KLD 0.000000 and the same top p at 512-token ubatches; against the CPU reference the same 0.001917 /
    98.403 % at ub 1024 and 0.001968 / 98.012 % at 2048.
  - The red reaches the tile through `llama-perplexity` at ub 512, 1024 and 2048: KL 3.18 against 0.0019.
  - Flash-Next IQ3_S on the B70 (no expert cache), with layers 28-47's experts on the card (`-ncmoe 28`, 512-token ubatches): 0.000000. Its
    down projections are IQ4_NL in 39 of 48 layers, and the red there gives 0.96, so the tile is reached and changes nothing.
  - Flash-Next on its unit's flags with the expert cache (13,000 MiB, `GGML_OPENCL_BANK_STREAM=1`, MTP 3), knob off
    against knob on with `GGML_OPENCL_BOUNDS=1`: the same greedy text (same md5) for three prompts of 12, 3,129 and
    8,391 tokens, and no bounds abort. A red build was not run on this route, so reach on it is not shown.
  - The agent's Qwen3.8 27B: 0.000000. It is dense; no MUL_MAT_ID node.
  - `-ub 1` takes the mat-vec, not the tiles.
- The "512-column band built" load line is printed by `test-backend-ops`. `llama-perplexity` does not print it even when
  the tile runs, so a red build, not the line, shows reach there.

## 0187-opencl-cpy-transpose.patch

The card's profile joins two slow `kernel_cpy_f32_f32_pack` copies: the bias table's
`ggml_cont(ggml_permute(table, 1, 0, 2, 3))` (`src/models/inkling2.cpp:390`, 1,353 us at
`[1025, 24, 512]`, 5,330 at 2048) and the depth mix's `ggml_cont(ggml_transpose(states))`
(`mix()` line 219, `[2304, k, T]` -> `[k, 2304, T]`, 8-20 B a destination row). Both swap dims 0
and 1: the source's fastest dimension is dim 1, the destination's is dim 0, so the pack kernel
gathers one side at a 4,100-B (or 96-B) stride and reaches 74-76 GB/s against a plain copy's
823-1,016 GB/s at the same bytes (`measured-here`, B60, constant data - the compression was in
that pair of numbers; on random data the plain rate is 360-390 GB/s).

This patch adds `kernel_cpy_f32_f32_t01`: a 16 x 16 tile of (dim 0, dim 1) through local memory,
`reqd_work_group_size`, reads coalesced over the source's fastest dimension, a tile barrier, and
writes coalesced over the destination's. Partial tiles on both edges are guarded per element (the
bias table's 24 heads and 1,025 bins are not multiples of 16). `ggml_cl_cpy` takes it only when
`GGML_OPENCL_CPY_TRANSPOSE=1`, only for F32 x F32 copies whose source and destination have the
same shape, the source's `nb[1] == 4` and `nb[0] == ne[1]*4`, the destination contiguous, and the
source's dim-2 stride covering the slab with dim 3 stepping by dim 2 (the exact terms are in the
next paragraph) - **unset, the pack kernel runs exactly as before**. Both
buffers' furthest byte is checked under `GGML_OPENCL_BOUNDS=1`. The red is a build option,
`GGML_OPENCL_CPY_TRANSPOSE_RED=1` (`-DCPY_T_RED`): the write guard becomes `bin_w < d1 - 1`, which
loses the last dim-1 edge - visible on 1,025 bins and on the `[33, 17, 3]` tile case.

**cp39's finding and the fix.** The first eligibility used `nb02 == nb10 * ne00` and
`nb03 == nb02 * ne01`, both false for every real source: the token slab's stride is
`ne00 * ne01 * sizeof(float)` (4 * 1025 * 24 = 98,400 for the bias table), not `4 * ne00`, and
dim 3 steps by the dim-2 count. The path was therefore never taken (80/80 CONT cases passing off
and on, the red passing, identical perf and pp) - a stride predicate that was wrong, invisible to
a CPU test. The predicate is now a **pure function** in `ggml-opencl-cpy-t01.h`
(`ggml_cl_cpy_t01_eligible`, no cl_ calls): F32 x F32, the same shape, a contiguous destination,
the source's dim-1 step one float, its **dim-0 step one full fast row (`ne[1]*4`)**, its token
stride covering the slab (`>= ne[0]*ne[1]*4`), dim-3 stepping by dim 2 - the form the kernel
actually indexes (`src + gz*s_nb2 + dim0*s_nb0 + dim1*s_nb1`, `dst + gz*d_nb2 + dim1*d_nb1 +
dim0*4`). `tests/test_opencl_cpy_t01.cpp` (arcint, `opencl-cpy-t01`, unit) feeds it the real
strides derived from the graph: the bias table's view `[24, 1025, T]` with
`nb = {4*1025, 4, 4*1025*24, ...}` from `inkling2.cpp:386/390`'s product and permute, and the
depth mix's transpose view `[k, 2304, T]` with `nb = {9,216, 4, 9,216*slots, ...}` from `mix()`
line 219 - and the negatives (a contiguous source, a slow step that is not one fast row, a
non-contiguous destination, an f16 pair). Run here: all pass; **with cp39's old terms put back,
both real cases FAIL** (`eligible=0, want 1`, exit 1) - that is the red, on the CPU, no card.

**The card can see the path ran** two ways, both in the script: the once-a-process stderr line
`0187 cpy-transpose taken ne=[24,1025,512,1]` (GGML_LOG_INFO), and the red arm failing - it can
only fail if the kernel ran. `GGML_OPENCL_CPY_TRANSPOSE_DEBUG=1` adds
`0187 cpy-transpose rejected: ... nb00==ne01*4 ...` naming which term failed.

Tests: `test_cont_t01` cases in the correctness list (`[1025, 24, 512]`, `[1025, 24, 2048]`,
`[2304, 4, 512]` with permute (1,0,2) and (0,2,1), the one-token `[1025, 24, 1]`, the partial
`[33, 17, 3]`, the one-head `[1, 1025, 64]` and `[24, 1025, 7]`) and the perf list (the three
measured shapes plus `[33, 17, 3]`). Random data, and the eval compares the destination against
the CPU; every line names `data=random-uniform,checked=out-vs-cpu`.

**Prediction, written before the card run** (`inferred`): against a plain copy's 360-390 GB/s on
random data - the 16 x 16 tile moves 64 B a read row (16 bins) and 96 B a write row (24 heads),
so the copy is tile-bound rather than stride-bound; `[1025, 24, 512]` should land at
**320-360 GB/s = 280-350 us** (3.9-4.8x the 1,353 us), `[1025, 24, 2048]` at **~1,300-1,600 us**
(3.3-4.1x the 5,330), and `[2304, 4, 512]` at only **~1.6-2x (200-330 us)**: its destination's
fastest dimension is 4 floats, so 12 of the tile's 16 lanes in dim 0 idle; a `CPY_T_TILE=8` build
or a narrow-dimension path is the follow-up if that one stays below 2x. Not measured when written;
the card numbers are below, and the prediction missed on all three shapes.

Gate on the card (`GGML_OPENCL_BOUNDS=1` where the op runs; run, the results are under "Card
evidence" below, except step (f) for Inkling at ub 512, which is still owed):

    # (a) the single case, knob on, bounds first
    GGML_OPENCL_BOUNDS=1 GGML_OPENCL_CPY_TRANSPOSE=1 test-backend-ops -b GPUOpenCL -o CONT \
        -p 'ne=\[1025,24,512,1\]'
    # (b) the CONT suite, knob off then on
    GGML_OPENCL_BOUNDS=1 test-backend-ops -b GPUOpenCL -o CONT
    GGML_OPENCL_BOUNDS=1 GGML_OPENCL_CPY_TRANSPOSE=1 test-backend-ops -b GPUOpenCL -o CONT
    # (c) the red: the 1,025-bin and 33x17 cases must FAIL
    GGML_OPENCL_BOUNDS=1 GGML_OPENCL_CPY_TRANSPOSE=1 GGML_OPENCL_CPY_TRANSPOSE_RED=1 \
        test-backend-ops -b GPUOpenCL -o CONT -p '1025|33,17'
    # (d) the perf shapes, knob off vs on
    test-backend-ops perf -b GPUOpenCL -o CONT -p '1025,24' ; GGML_OPENCL_CPY_TRANSPOSE=1 \
        test-backend-ops perf -b GPUOpenCL -o CONT -p '1025,24'
    test-backend-ops perf -b GPUOpenCL -o CONT -p '2304,4,512' ; GGML_OPENCL_CPY_TRANSPOSE=1 \
        test-backend-ops perf -b GPUOpenCL -o CONT -p '2304,4,512'
    # (e) the prompt, at the Inkling standard arm (`GGML_OPENCL_KQ_MM_ID_I8=1`
    # GGML_OPENCL_KQ_MM_ID_I8_WIDE=1`, `--llama-swa-window`, ub 1024 the main batch size, 512 and
    # 2048 as extra columns; decode is unaffected, the wide tile is a prefill path). Reference
    # (cp17, B60, measured-here): pp4096 with WIDE=1 = 722.2 / 765.7 / 723.4 t/s at ub
    # 512 / 1024 / 2048
    export GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1
    llama-bench -m <pinned IQ4_NL>.gguf -fa on -p 4096 -ub 1024 --llama-swa-window -r 3
    GGML_OPENCL_CPY_TRANSPOSE=1 llama-bench -m <pinned IQ4_NL>.gguf -fa on -p 4096 -ub 1024 --llama-swa-window -r 3
    # the extra columns, knob off vs on
    for U in 512 2048; do llama-bench -m <pinned IQ4_NL>.gguf -fa on -p 4096 -ub $U --llama-swa-window -r 3; \
        GGML_OPENCL_CPY_TRANSPOSE=1 llama-bench -m <pinned IQ4_NL>.gguf -fa on -p 4096 -ub $U --llama-swa-window -r 3; done
    # (f) KL at ub 512, knob off vs on
    GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1 \
    llama-perplexity -m <pinned IQ4_NL>.gguf -f <gate.txt> -c 2048 --chunks 3 -fa on -ub 512 \
        --kl-divergence --kl-divergence-base <base.logits>
    GGML_OPENCL_CPY_TRANSPOSE=1 llama-perplexity ...  (same line)

(`llama-bench` has no `--llama-swa-window`; its default `swa_full=false` is the same window. The card
runs below used plain `llama-bench`.)

Card evidence (`measured-here`, the B60 at a 165 W cap, the card-tested applied series with the kernel still
inside the guard, git write-tree `b7ea4eea617993d5cbeab0452ab636983fe068a6`, `GGML_OPENCL_KQ_MM_ID_I8=1
GGML_OPENCL_KQ_MM_ID_I8_WIDE=1`):
- The path runs. Step (a) printed `0187 cpy-transpose taken ne=[24,1025,512,1]`, and the red (c)
  failed on 5 cases. The CONT suite passed knob off and on.
- Op perf, knob off -> on. **The prediction missed: the kernel is slower than predicted on all
  three shapes.**
  - `[1025, 24, 512]`: 1,334 -> 415 us (70 -> 227 GB/s as `test-backend-ops` prints it, 2^30 bytes: 75 -> 243 in
    decimal GB/s; predicted 280-350 us).
  - `[1025, 24, 2048]`: 5,306 -> 1,660 us (predicted 1,300-1,600).
  - `[2304, 4, 512]`: 441 -> 356 us, 1.24x (predicted 200-330). The 4-float destination leaves 12
    of the tile's 16 dim-0 lanes idle.

  The plain copy reaches 360-390 GB/s on random data, so 227 GB/s leaves room. What limits the
  tile there is not yet known (open).
- pp4096 t/s, knob off -> on, three interleaved rounds on an idle host (the card host's other
  model server held, the recipe host quiet):

  | ub | off | on | gain |
  |---|---|---|---|
  | 512 | 720.65 / 720.53 / 720.63 | 769.67 / 769.83 / 769.60 | +6.8 % |
  | 1024 | 763.59 / 763.56 / 763.81 | 816.11 / 815.95 / 816.43 | +6.9 % |
  | 2048 | 718.39 / 718.66 / 718.49 | 764.40 / 764.21 / 764.33 | +6.4 % |

  At ub 512 that is 5,684 -> 5,322 ms for the 4,096 tokens. The bias-table copy alone accounts for about
  384 x 919 us = 353 ms of that (`inferred` from the op times); the depth-mix copies give the rest.
- Reach. The proof is the export: the predicate applied to CPU graph-op exports
  (`test-export-graph-ops`, every CPY / CONT / DUP line) finds 0 eligible copies in each model:
  - Qwen3.8-Flash-Next (IQ3_S, q8_0 KV) at its served ub 8192 and at 512: 0 of 27;
  - the 27B: 0 of 9;
  - Dory: 0 of 2.

  The Flash-Next MTP head is arcint's own GGUF, not a llama graph, and is not covered by the
  export. **No effect on them; Inkling only.** (An earlier `llama-bench` run with
  `GGML_OPENCL_CPY_TRANSPOSE_DEBUG=1` printed no taken and no rejected lines for the 27B and Dory. That
  is not evidence: `llama-bench` mutes every backend log line unless it runs with `-v`. A check that
  reads printed lines first asserts that the log is not muted, by counting a line known to print at
  that verbosity.) A CPU graph-op
  export of Inkling lists the eligible copies: the bias table `[24, 1025, T]` and the depth-mix
  stacks `[k, 2304, T]` (k 2..5), at T = 1 as well. So the path is also taken in decode
  (`inferred` from the CPU export; no decode run on the card printed the line, and the decode
  rate knob off vs on is not measured).
- Logits, knob off as the KL base, knob on (Mean KLD / Max KLD / top p):

  2 chunks each: 1,024 tokens at ub 512 / 1024, 512 at ub 1.

  | model | ub 512 | ub 1024 | ub 1 |
  |---|---|---|---|
  | Inkling 2 Mini | 0 / 6.2e-5 / 100 % | 0 / 5.6e-5 / 100 % | 0 / 3.7e-5 / 100 % |
  | Qwen3.8-27B | 0 / 4.4e-5 / 100 % | 0 / 5.0e-5 / 100 % | 0 / 4.1e-5 / 100 % |
  | Dory 4B | 0 / 5.2e-5 / 100 % | 0 / 5.8e-5 / 100 % | 0 / 4.8e-5 / 100 % |

  The 27B and Dory never take the path, yet their Max KLD is the same 4-6e-5: that is the card's
  run-to-run floor, not this patch. Inkling's KL against the CPU reference at ub 1024 was 0.001917
  / 98.403 % in both arms.
- The pushed patch differs from the card-tested tree in one place: `kernel_cpy_f32_f32_t01` moved below
  `#endif // GGML_CL_NO_CPY_PACK` in `cpy.cl`. Its creation is unconditional, so a device that defines the
  guard would have failed to load the program. Only Adreno A6X defines it (`ggml-opencl.cpp`, `no_cpy_pack`),
  so on Arc the same kernels compile from the same source. Applied series of this commit: write-tree
  `182258aeb0f232cdbfc5db9eec26f236347e3969`; offline gate 3/3 (default, `GGML_CL_NO_CPY_PACK`, the red;
  128 GRF, scratch 0).
- `llama-perplexity` filters ggml's INFO lines at its default verbosity, so the taken line shows only
  with `-lv 4`. With `GGML_OPENCL_CPY_TRANSPOSE_DEBUG=1` alone it printed 0 taken lines and 644 rejected
  ones (the model's other copies, e.g. `ne=[3,1024,1,1]` failing `same_shape` and `nb01==4`). With `-lv 4`
  it printed the taken line, in the Inkling ub 512 identity run too. A per-call counter would still
  give the count; it belongs to the follow-up on the narrow destination.

- At q8_0 KV (the Inkling units' cache type) the gain holds: Inkling pp4096 ub 1024 763.5 (`GGML_OPENCL_CPY_TRANSPOSE` off) ->
  816.3 t/s (on), +6.9 % (`measured-here`, llama-bench; not at the units' `--llama-swa-window` / 262,144 context).

## 0193-flash-attn-port-op-sources.patch / 0194-flash-attn-port-sink-positions.patch / 0198-opencl-flash-attn-port-kernel.patch / 0199-flash-attn-port-tests.patch

Inkling 2 Mini's prompt attention after trellis-serve's `fa::Kern<128>` (MIT; `THIRD_PARTY.md`), as an opt-in arm of the
OpenCL backend: `GGML_OPENCL_FA_PORT=1`. Unset, the dispatch and the model's graph are what they were. The full record -
the reference kernel, the plan, each card run, the reviews and the design of the next step - is
`contrib/refkernels/trellis-fa/README.md`.

- **0193** (`ggml.h`, `ggml-impl.h`, `ggml.c`): the op's new sources and their setters - `kpos` / `qpos` (I32), one class a
  16-key group and one flag a 8-row group (I8), the per-query sink logits (F32 `[n_head, rows]`, src 18) and the window
  (op param 16). `GGML_MAX_SRC` 14 -> 20 and `GGML_MAX_OP_PARAMS` 64 -> 80: **64 bytes more a tensor, with the knob or
  without**. The indexed-bias setter accepts the table as `[n_bins, n_head, rows]` too.
- **0194** (`ops.cpp`, `llama-graph.*`, `llama-kv-cache.*`, `inkling2.cpp`): for a prompt batch (flash attention, causal,
  the standard window, more than 8 rows, one stream) on a layer whose backend declares the port's sources, the sink is
  not folded into the table (the table is the bias only, bin-fastest as the MUL_MAT emits it) and rides as src 18; the
  kv-cache walk fills the positions and classes; the sources are on the op before the backend's `supports_op` is asked.
  The CPU's kernel reads both table layouts and the per-query sink. Any other graph is today's.
- **0198** (`fa_port_gate.h`, `ggml-opencl.cpp`, `kernels/flash_attn_port.cl`): the gate as pure functions (the strides the
  kernel indexes, the 2D block read's alignment, the work-group, the paged forms refused), one decision for `supports_op`
  and the launch, the port taken first in the dispatch, a bounds check of every buffer under `GGML_OPENCL_BOUNDS=1`, a
  DOWNGRADE line a refusal, one TAKEN line a process. The kernel: one sub-group a (head, 8-row) tile over 32-key blocks,
  K/V by 2D block reads, a class a key group and a skip a block, the bias run as one block read and a shuffle, the pad
  bin read once a tile, the online softmax with the sink as its initial state, Q held in registers at 256 GRF.
- **0199** (`test-backend-ops.cpp`): the port's cases in the model's layout (the graph export's prompt lines), their twins
  for the indexed-bias kernels of today on the same data, the pad case, the perf rows, and three device-free checks
  (`GGML_TEST_FA_PORT_TWINS`, `_DECISIONS`, `_GUARD`), each with a red.

Knobs: `GGML_OPENCL_FA_PORT` (the arm), `_GRF=128|256` (256), `_QHOIST`, `_QF32`, `_SKIP`, `_UNIFORM`, `_QTW`,
`_RED=1|2|4` (each must fail), `_PROBE=nobias` (a timing probe with wrong answers).

`measured-here`, the card host, B60 at 165 W. The op, us a call, window 1,024, 24 heads on 8, head 128, 1,025 bins:

| shape | the port, 256 GRF | the kernel it replaces (the twin) |
|---|---|---|
| q512 x kv1536 | 746 | 2,263 |
| q2048 x kv2048 | 2,092 | 9,577 |
| q2048 x kv3072 | 2,926 | 12,645 |

Inkling pp4096, t/s, the port off / on: 769.2 / 909.8 (ub 512), 816.0 / 973.9 (ub 1024), 763.9 / 980.8 (ub 2048), 749.5 /
975.7 (ub 4096). Mean KL at ub 1024 against the CPU base 0.001917 -> 0.001965, top p 98.40 -> 98.47 %. The op suite is
complete with the knob on (3,190 cases) and off (3,179); the reds fail every case they run; with the knob unset three
models match the development build. Not through a release gate.

**Limits, read before any unit uses the knob.** Every number above is at f16 KV (llama-bench's default). The port
REFUSES a q8_0 KV cache, and the Inkling units serve q8_0 KV: at q8_0 `GGML_OPENCL_FA_PORT` changes nothing
(`measured-here`, pp4096 ub 1024 with `GGML_OPENCL_CPY_TRANSPOSE=1`: 816.3 with the port off, 816.6 with it on and 7
refusals logged; 974 at f16 KV with the port taken). At the
units' own flags (262,144 context, `--llama-swa-window`) f16 KV does not fit the 24 GB card beside the model: arcint's
load-time spill guard refuses it (`measured-here`: context 8,516 MiB at f16 against 4,526 MiB at q8_0, which loads). So the port helps a served Inkling only once it reads
a q8_0 cache. Not verified: a partial offload (`-ngl` below all layers). At `-ngl 12` the run faulted (`measured-here`). The
cause, by code reading and the fault's arithmetic (not reproduced): Inkling's depth-mix stacks are graph leaves written
through views from layers on both devices, which faults or silently loses rows at a CPU / card boundary. The code
involved is byte-identical on the development build without the port (`code`). Serve Inkling fully offloaded.

Red / green without a device: `contrib/refkernels/trellis-fa/port_gate_test.cpp` on `fixtures/inkling-ub*.txt`
(`-DFA_PORT_GATE_RED`, `-DFA_PORT_GATE_RED_PAGED`), and the three `GGML_TEST_FA_PORT_*` checks above.

## 0208-test-backend-ops-inkling-expert-projection-perf.patch

The expert DOWN projection next to the fused GATE_UP as perf cases, so one card run compares them, plus
one correctness case each so the CPU reference runs both. The shapes, strides and ids layout are the
Inkling 2 Mini graph export's MUL_MAT_ID lines, named in the ub 512 / 1024 / 2048 exports:
`ffn_moe_down-0` (IQ4_NL weights `[1152, 2304, 128]`, nb 18 / 648 / 1,492,992; a token's F32
activations `[1152, 8, n]`, nb 4 / 4,608 / 36,864) and `ffn_moe_gate_up-0` (`[2304, 2304, 128]`,
nb 18 / 1,296 / 2,985,984; gate and up fused, so b is one broadcast column, `[2304, 1, n]`, nb
4 / 9,216 / 9,216). The ids are the constructor's own view: a token's 128-wide I32 row with the
8 used columns, nb1 512. 128 experts, top-8, F32 activations, at 512, 1,024 and 2,048 tokens; 2,304
rows both ways, K 1,152 for down and K 2,304 for gate/up. The routing is fixed (seed 20261124), each
token's eight slots eight distinct experts over the 128, not the base case's random re-draw: per-expert
pair counts 20..51 (mean 32) at 512, 46..91 (64) at 1,024, 100..166 (128) at 2,048, no expert empty.
`test_mul_mat_id_routed` keeps the routing between perf iterations, so the two arms time the same work.

Prediction before the card (inference): at 512 tokens the wide-tile per-visit figures of 0190 (26.46 ms
gate/up, 13.56 ms down over 8 ub chunks) are 3,310 and 1,700 us a call; the narrow tile gives 3,915 and
2,110. Down's narrow-to-wide difference is 0.41 ms a call, which is the 0.43 ms a call the 2026-11-24
pass reading gives over 384 calls (165 ms); the int8 floor is about 0.11 ms a call. At 1,024 and 2,048
the weights are re-read once a 512-column band, 2 and 4 times, so about 3.3 / 6.6 ms a call for down and
6.5 / 12.9 for gate/up, before the empty-tile floor.

Card commands, perf mode:
```
GGML_OPENCL_KQ_MM_ID_I8=1 test-backend-ops perf -b OpenCL -o MUL_MAT_ID -p 'routed=1'
GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1 test-backend-ops perf -b OpenCL -o MUL_MAT_ID -p 'routed=1'
```

CPU gate: `test-backend-ops -b CPU -o MUL_MAT_ID -p 'routed=1'` gives 2/2 OK (the two new correctness
cases). The whole `-o MUL_MAT_ID` family did not finish here within a 30-minute cap (163 OK, 0 fail before
the cap; the suite's large 512-expert cases dominate), so the full-family CPU row is open.
- Open: `llama-perplexity` does not print the once-a-process INFO line. The KL runs therefore do
  not themselves show the path was taken, although the `llama-bench` runs on the same binary and
  env do. Inkling's ub 512 logits arm (gate step (f) as written) was not run. Owed: a `GGML_OPENCL_CPY_TRANSPOSE_DEBUG=1` perplexity run, or a per-call counter.
## 0204-0207: Inkling 2 Mini's depth mix as one op (the plan, written before any card run)

A native DEPTH-MIX kernel for the prefill, ported from the reference's Triton `_mix_kernel`
(the trellis-serve project, `xpu/exl3xpu/inkling2/fused.py:41`, "a token a program, fp32"; the model's
`_mix` is `sgl_model.py:52` and the call sites `:612`, `:620`, `:644`, `:658`). The reference's
form is one launch a mix call: it reads the S states and cur once per chunk, normalizes them per
state, takes `q . v` as one sum over the embedding width, softmaxes over the states and writes
`sum_s p v_s`, optionally with an output RMSNorm and per-output weights folded in.

**The sites in the applied tree** (`src/models/inkling2.cpp`, lines as the series through 0192
has them, i.e. the chain the table below ran; patch 0206 moves them by its own added block):
`mix()` is `:191`; its chain is the cpy of cur into the store's last row `:213`, the states
`view_3d` of the store `:216`, the `rms_norm` over the k+1 states `:217`, the `repeat` of the
query over the tokens `:219-221`, `mul_mat(states, queries)` `:224`, `soft_max_ext` with
`1/sqrt(n_embd)` `:225`, `cont(transpose(states))` `:227` and `mul_mat(that, scores)` `:227`.
The call sites are the attn mix `:328` (an SWA layer without messages norms one column and the
arm folds it; full-attention layers norm three with three separate 1-D weights, kept as nodes)
and its per-column `ggml_cont` `:330`, the ffn mix `:421` (folded) and the output mix `:456`
(folded). `write_res_block` `:184` keeps its cpy: it fills the store's block rows, which later
mix calls read. The controllers (`control()` `:269`) are not part of the chain: they run
between a cross-pass message and the norm, so those visits keep the message merge, the three
norms and the ffn norm. The permute-conts that 0187 also takes are `:390` (the relative-bias
table) and `:394` (`kq_b`), both attention.

**The four probed parts, mapped** (measured-here, B60, one pp4096 pass at ub 512, `I8=1`,
`WIDE` off; the F32 arms are 0180's default-on):

| part | ms a pass | launches | site | kernel |
|---|---|---|---|---|
| stack copies | 341 | 961 | `:227` `ggml_cont(ggml_transpose(states))` | the pack kernel (a transposed view) |
| first product | 309 | ~960 | `:224` `mul_mat(states, queries)`, dst [k, nq, 512] | `kernel_gemv_f32_f32_splitk_batch` (0178) |
| weighted sum | 82 | 961 | `:227` `mul_mat(cont(transpose(states)), scores)`, dst [2304, nq, 512] | `kernel_mul_mat_f32_f32_smallk` (0171/0175) |
| query repeats | 33 | ~960 | `:219-221` `ggml_repeat(queries)` | the repeat kernel |
| total | 765 | | 12.7 % of the 6,042 ms pass | |

Not in the table and taken by the arm as well (`inference` from the node kinds): the states'
`rms_norm` `:217`, the cur cpy `:213` and the per-column cont `:357` - one read and one write of
4.72 MB each a call, plus 4.72 MB an output column.

**The form: a new ggml op with a CPU reference, not a backend fusion.** Both were priced:

- *A backend fusion of the existing nodes* (0049/0051 style; this backend's own `ggml_opencl_can_fuse`
  patterns for rms_norm+mul+add): the members are not adjacent - the store's block cpy
  (`write_res_block`) and the cur cpy sit between them, and the output's norm nodes sit after -
  so it needs `ggml_can_fuse_subgraph_ext` with an explicit skip mask and a whole-subgraph
  interceptor, and it cannot change a member's layout: the `cont(transpose)` node would have to
  be *skipped* as a member, and then the fusion must read the transposed view itself. Cost: a
  matcher that depends on the graph's node order and on every other patch's nodes, no CPU
  reference (so no op case against a second implementation), no eligibility unit test and no
  refusal reason; the gain is the same only if the skip mask is right.
- *A new op* (`GGML_OP_DEPTH_MIX`): an op id, a constructor, a CPU reference, an OpenCL kernel, a
  dispatch, a test case; `tests/python/test_gdn_replay.py`'s `GGML_OP_COUNT` gate moves with
  the table, 104 -> 105. Cost: +1 op in `ggml.c`'s tables, the CPU path (mechanical, 0158's
  shape). Gain: the model emits it behind the knob, so the matcher cannot drift from the graph;
  the eligibility rule is one pure function (`ggml_depth_mix_refuse`) shared by the model, the
  dispatch and the CPU op, unit-testable on the graph's own strides; the transpose copy is not
  a node at all; the dst layout can be chosen ([n_embd, n_tokens, nq], a column's token block
  contiguous) so the per-column cont goes.

**Kernel layout** (`depth_mix.cl`): a token a work-group of 256 lanes; `n_embd/256` floats of
each of the k+1 states a lane, read once from global into registers and never written back
(n_embd <= 2304 keeps that at 9 floats a state, 63 for 7 states). Sum of squares a state and
the query dots are partials in local memory, reduced in one round of 8 quantities (8
sub-groups of 32 lanes, a 5-step tree each); the scores, the softmax and the weighted sum run
on the register copy; the folded norm is a second reduction over the output columns. The
state/element loops are unrolled over their maxima with runtime guards, which keeps the
register array's indices compile-time - a dynamic index would spill to scratch, the 0190
lesson. `k` 1..6 and `nq` 1..4 bound the reduction's local buffer (35 KB at worst, sized at the
launch), `T > 1` (decode stays with today's chain; the FUSE session's decode block is the other
half of this chain). The offline gate (`ocloc` 26.27.1 + IGC 2.38.2, the B60 `bmg-g21`, the
backend's options plus `-cl-intel-256-GRF-per-thread`, base and the red variant): both variants
compile, `grf 256 simd 32 slm 1024 scratch 0 stack_calls -`, no ISA call, no retry dump - the
register array stays in GRF (no spill), and 256 GRF is the program's ceiling, so occupancy is
one work-group's worth; if the card's rate lands at the copies' rather than at the reference's,
the fallback is the two-pass form (states read twice from global, no register array) that the
same rule and kernel arguments allow (`inference`, not run).

**Bytes and the rate the reference implies.** One state row of a 512-token chunk is
2304 x 512 x 4 = 4.72 MB; at k=4 (5 states) a call reads 23.6 MB of states, 9.44 MB of query
rows (2 columns; 18.4 KB unique, so L2 after the first token) and writes 9.44 MB. The arm moves
~33 MB of DRAM a call. The reference costs 398.96 ms over ~960 calls, 415 us a call, and its
kernel reads the states three times (the dots pass, the NORM pass, the output pass): ~89.7 MB a
call, so 216 GB/s - 47 % of the card's 456 GB/s, `inference` from the operator's trace. The
arm's traffic is 0.37 of that; at the reference's own rate a call is ~153 us and a pass ~147 ms.

**Prediction** (`inference`, to be checked by the card runs below; nothing was run here, this
host has no OpenCL ICD):

| ubatch | today's chain, ms a pass | arm, ms a pass | us a call (k 2..5, nq 1..3) |
|---|---|---|---|
| 512 | 765 | 120-160 | 110-160 |
| 1024 | ~765 (bytes the same, launches halved) | 120-160 | 220-320 |
| 2048 | ~765 | 110-150 | 440-560 |

Against the reference level of 399 ms the arm's *traffic* is below it by construction (one state
read against three, no copy, no repeat materialization); going further below it needs the rate,
not less work: at 456 GB/s the 33 MB a call would be 72 us, and the reduction's barriers are
what stands between 153 and 72 us a call. Said plainly: the arm beats 399 ms if it reaches the
reference's bytes-per-second; it does not need to reach it to beat 765 ms.

**What 0187 does and what remains of it.** 0187 (another session) speeds the transposed copy
`kernel_cpy_f32_f32_pack`. Its estimate was 128-171 ms of the same 341 ms; the card measured
**1.24x** on this copy (441 -> 356 us at k=4, 66 ms of the 341, `measured-here`), so the depth
mix's share of it is ~66 ms and the rest of 0187's estimate is elsewhere (see the card row
below). The two gains must not be counted twice:

| arm | `GGML_OPENCL_CPY_TRANSPOSE` unset | `=1` |
|---|---|---|
| knob off | 765 ms (today) | ~699 ms (0187's measured 1.24x on this copy) |
| knob on | 120-160 ms | 120-160 ms - unchanged |

With the arm on, 0187's depth-mix target has nothing left: the [k, 2304, 512] cont of a
transposed view does not exist (the states are read by the kernel from the store's view and
cur, and no copy is materialized). What 0187 also takes and the arm does not touch: the
`[24, 1025, 512]` permute-cont after the relative-bias table product (`:390`) feeding the gather,
and the `[n_head, n_kv, tokens]` permute-cont of `kq_b` (`:394`) - attention, not depth mix, so
those shapes remain 0187's.

**The wide int8 tile does not move these numbers** (`code`): `GGML_OPENCL_KQ_MM_ID_I8_WIDE` is
read only by `ggml_cl_mul_mat_id`'s int8 route, so `MUL_MAT_ID`-only; no depth-mix node is one.
The card arms below still run with `GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1`,
the deploy setting.

**Rules the arm carries.** Opt-in (`GGML_OPENCL_DEPTH_MIX=1`; unset, `0`, `off` are off and the
model emits the chain node for node, decode untouched, other models untouched - they never emit
the op). One fixed stderr line when the path is taken: `depth_mix: TAKEN opencl f32
token-a-group`, once a process. A shape, type or stride the rule does not cover leaves
`DOWNGRADE src=llama op=DEPTH_MIX path=chain reason=<rule>` at the model and `DOWNGRADE
src=opencl op=DEPTH_MIX path=cpu reason=<rule>` at the dispatch, once a key each.
`GGML_OPENCL_DEPTH_MIX_STATS=1` prints one line a graph: `depth_mix: graph nodes=<n> calls=<c>
of <nodes>`. Every buffer is bounded before the launch under `GGML_OPENCL_BOUNDS=1` (blocks,
cur, queries, norm_w, dst; the furthest byte from the global size and the strides handed to the
kernel, a view's offset included).

**Correctness.** f32 accumulation as the reference. One summation order changes: the chain
normalizes a state first and accumulates normalized values in the dot; the kernel accumulates
raw values and scales the finished dot by `1/sqrt(mean(v^2) + eps)` - the same value, a
different rounding (`code`). The softmax, the weighted sum and the folded norm keep the chain's
order. Bit-identity is not claimed; the operator's bar is mean KL <= 0.03 nats worse than
baseline and top p down <= 1 point.

**Tests.** `test-depth-mix` (device-free, `code`): `ggml_depth_mix_refuse`/`ggml_depth_mix_switch`
on Inkling's own prefill nodes, built with the graph's constructors at its GGUF sizes (n_embd
2304, attn_res_block_size 4, store `[2304, 7, 512|2048]`, block view at the store's strides -
row step 2304 floats, slab step 7 x 2304 floats, k 2..5, nq 1 and 3), and eight red variants
that must each be refused by name. `test-backend-ops`: Inkling's shapes, k 2..5, 512 and 2048
tokens, nq 1 and 3, with and without the folded norm, random data; the perf list has k 2, 4, 5
at both sizes. Red option: `GGML_OPENCL_DEPTH_MIX_OPTS=-DDEPTH_MIX_RED` drops the softmax
denominator in the kernel only (the CPU op stays right, so the op cases must fail).

**Card result, 2026-11-24 (B60, the series cdd7db8d, `measured-here`): the path never ran.**
`test-backend-ops -o DEPTH_MIX` printed `not supported [OpenCL]` for every case (0 of 0 executed;
"2/2 backends passed" was two skips), the red therefore "passed" with 0 failures, and a
`llama-bench pp4096` at ub 512 with `GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_DEPTH_MIX_STATS=1`
printed no TAKEN line, no stats line and no DOWNGRADE line - the model's node went to the CPU
silently. Cause (`code`, the applied tree's `supports_op`, `case GGML_OP_DEPTH_MIX`): it also
required `op->src[i]->extra != nullptr` and `op->extra != nullptr`. `supports_op` is asked
before any tensor is allocated (test-backend-ops, and the scheduler when it assigns nodes), so
`extra` was always null: the op was refused everywhere. The rule was right; the dispatch
disagreed with it behind an allocation-dependent test.

The fix (0205, and 0206/0207 follow): the decision is `ggml_depth_mix_refuse` plus the dst
layout and the knob - types, shapes and strides only, as the other ops do; device storage is
the dispatch's invariant (an assert, not a condition). Every refusal logs a DOWNGRADE line
(`switch-off`, the rule's reason, `dst-layout`, `not-built`), and a graph that hands the arm
nodes and takes fewer of them logs `reason=not-taken nodes=N taken=T` at WARN once, so a
silent fallback cannot happen again. `GGML_OPENCL_DEPTH_MIX_STATS=1` prints the graph's node
and call counts.

Why the device-free test missed it and what now guards it: `test-depth-mix` tested the rule
and the switch, and the defect was in the backend's own additional condition - the two agreed
on nothing a device-free harness could see. The test now reads the `supports_op` case out of
`ggml-opencl.cpp` (`DEPTH_MIX_OPENCL_SRC`) and fails if that segment mentions `->extra`, if it
does not decide through `ggml_depth_mix_refuse(`, or if it has no `ggml_cl_downgrade(` - the
shape of the defect, checked as text.

**The real graph's strides** (`measured-here` by the graph export, `inkling-ub{512,
1024,2048}.txt`, `test-export-graph-ops` at c 4096, fa on; source tensors of the depth-mix
lines, op 25 `RMS_NORM` into op 29 `MUL_MAT` and op 34 `CPY`): the block store is
`[2304, 5, tokens]` - attn_res_block_size **6**, so `(24 + 5)/6 + 1 = 5` rows, not the 7 this
README first assumed - with `nb[1]` 9216 and `nb[2]` 5 * 9216 = **46080**; the states view
keeps them, 2..5 rows (k = 1..4 block rows, so the observed `k` range of the copies is the
states' 2..5), and at fewer rows the token slab step exceeds the states' extent; `cur` is
`[2304, tokens]` with `nb[1]` 9216; the query parent is `[2304, nq]` (nq 1 and 3) with
`nb[1]` 9216. The rule accepts these (a slab step >= the rows' extent, the rows back to
back), and `test-depth-mix` now builds exactly them - `[2304, 5, 512|2048]`, row step 9216,
slab step 46080, k 1..4, nq 1 and 3 - and asserts the literals; the op case set is k 1..4
(2..5 states) at 512 and 2048 tokens.

**Against 0187, now measured** (`measured-here`, the operator's card row): 0187's tiled
transpose takes the `[k, 2304, 512]` copies at **1.24x**, 441 -> 356 us at k=4, and pp4096
at ub 512 from **719.7 to 768.8 t/s** (`I8=1 WIDE=1`). The two gains are not additive; the
arm's prediction either way:

| baseline | depth-mix parts, ms a pass | arm on, ms a pass | pp4096 ub 512 |
|---|---|---|---|
| 0187 off | 765 | 120-160 | 805-815 t/s (from 719.7: 5.69 s a pass, -605..-645 ms saved) |
| 0187 on | ~699 (341 -> 275) | 120-160 | the same 805-815 - 0187's depth-mix share is subsumed |

With the arm on, 0187's depth-mix target has nothing left in either baseline: the copy is not a
node. What 0187 still takes and the arm does not touch stays as listed above (`:390`, `:394`).
A correction to the earlier table's "0187 off/on" row: the off/on pair is 765 against ~699 (not
594-637, which assumed all of 0187's 128-171 ms came from the depth mix; the measured 1.24x on
this copy is 66 ms of it).

**Card result 2, 2026-11-24 (mix-on-dev 475cebfb, applied series 4c8f94ba, B60, `measured-here`).**
The fix ran: the suite executed 19 cases, 16 OK; the red failed 19 (good). **The 3 failures are
exactly the folded-norm cases**, ERR ~2: `DEPTH_MIX(k=1,nt=512,nq=1,with_norm=1)` ERR 1.990,
`(k=4,nt=2048,nq=1,with_norm=1)` 1.931, `(k=3,nt=512,nq=3,with_norm=1)` 2.010. Every case
without the norm passes. The model run (`llama-bench pp4096 ub512`, knob on, `BOUNDS` unset)
then caused a **GPU page fault** (`clWaitForEvents -14`, 446 fault responses, an engine reset, a
devcoredump; the card recovered): all 446 are reads, one row each at the start of ~58 slabs
4,718,592 bytes apart (`inference` from the operator's kernel-log fit) - a [2304, 512] f32 slab
spacing, which is the compute buffer's per-token tensors (`dst`, `cur`) and equals the folded
norm calls' count a chunk (~57).

**Named defect #1 (code, fixed): the folded-norm path's local slots aliased the dot results.**
The first reduction left its results at `red[p*LS]` for `p = 0..ns*(1+nq)-1`; the norm's partial
sums were written at `red[q*LS + lane]`. For (k=1,nq=3) the norm partials [512..767] cover the
dot result read at 512, for (k=2,nq=4) [768..1023] covers 768 - writes from one sub-group
overwriting a result another sub-group may still read (a race; sub-groups do not run in
lockstep). Verified as an interval overlap (`aliascheck`: 4 overlaps) and fixed by moving the
norm's quantities past the first reduction's: `red[(np+q)*LS]` with `np = ns*(1+nq)`, the
reduce taking a base `off`, the dynamic local sized `(np+nq)*256*4`; the same check now finds 0
overlaps. This does **not** explain the three card failures (their (k,nq) do not overlap).

**Named defect #2 (code, withheld): a norm weight through a view is not covered.** The model's
norm weights are 1-D tensors, and the fold reached the op as a `reshape_2d` view, whose storage
resolves through the parent's extra (the backend's own comment flags that path for weight
tensors). Every folded-norm op case fails on the card while every no-norm case passes, and the
fault's slab spacing/count matches the folded calls; a host replay of the kernel's index math on
the graph's strides (`port3`: `blocks` rows + `cur` + `queries` + `dst`, ASan, every access
range-checked) is clean and agrees with the CPU op to fp32 rounding, so the strides are not the
fault's cause and the weight's storage is the one argument that replay cannot see. The rule now
refuses `norm_w->view_src != nullptr` (`nw-view`), the dispatch asserts the weight is the tensor
itself, and **the model no longer folds the norm**: the norms stay the chain's nodes, exactly as
today (no behaviour change), and the op suite carries no `with_norm` case until the path has its
own card evidence. The op's CPU reference and kernel keep the capability for a later patch.

**The arm's bounds are now always on** (not only under `GGML_OPENCL_BOUNDS=1`): the five
arguments (blocks, cur, queries, norm_w, dst) are checked against their buffers before the
launch and a violation aborts with the buffer's name and the byte it would reach - a fault
becomes a diagnosis. The op cases and the model run can therefore be run with `BOUNDS` unset and
still name any past-buffer launch.

**Card commands** (to run on the B60; `llama-bench` and the op cases need the arm's env):

```
# 1. one case first, the bounds check on
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 test-backend-ops -o DEPTH_MIX -p 'k=2 nt=512 nq=1'
# 2. the whole op set, then the TAKEN line on a model run
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 test-backend-ops -o DEPTH_MIX
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1 \
  llama-bench -m <Inkling IQ4_NL> -fa 1 -p 4096 -b 4096 -ub 512 -r 2 2>&1 | grep 'depth_mix: TAKEN'
# 3. the red: the op cases must fail
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 GGML_OPENCL_DEPTH_MIX_OPTS=-DDEPTH_MIX_RED \
  test-backend-ops -o DEPTH_MIX
# 4. op perf, one launch against the CPU op
GGML_OPENCL_DEPTH_MIX=1 test-backend-ops perf -o DEPTH_MIX
# 5. llama-bench pp4096, knob off then on, ub 512 / 1024 / 2048
for ub in 512 1024 2048; do
  for dm in 0 1; do
    GGML_OPENCL_DEPTH_MIX=$dm GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1 \
      llama-bench -m <Inkling IQ4_NL> -fa 1 -p 4096 -b 4096 -ub $ub -r 2
  done
done
```

**Card result 2, the causes named (2026-11-24; `code`, and `measured-here` on the recipe host
for the replays - no card run is behind this paragraph).** Both failures of card result 2 have
a named line, and neither is one of the two defects named above. The retractions first, on the
record:

- *"Named defect #2" above is retracted.* A norm weight seen through `reshape_2d` was never the
  problem: a view shares its parent's extra and the offset is `extra->offset + view_offs`, as
  for every other source of every op. The `nw-view` refusal, the dispatch's assert and the
  withheld fold are gone again.
- *The fault paragraph's inference is retracted* ("the compute buffer's per-token tensors,
  equals the folded norm calls' count"). The fold had no part in the fault, and withholding it
  would not have stopped it: the faulting read is on a path that never folds.
- "Named defect #1" (the local-slot alias) stands as a real race and its fix is kept; as its
  own text says, it explains neither card failure.

**The page fault: `src/models/inkling2.cpp`, the loop model's `message()`** (the lambda at
the top of `build_layer`; the line `ggml_cont(ggml_view_2d(messages, n_embd, n_tokens,
messages->nb[2], ctrl*messages->nb[1]))`). Inkling 2 Mini is the loop model
(`inkling2-loop.cpp`): from the second pass on, every core layer gets `messages = mix(bank,
bank_out, loop.msg_query, bank_stack)` - a mix over the previous pass's bank store (2 rows:
one block and cur) with four query columns, one a controller. That call is eligible (k 1,
nq 4), so with the knob on `mix()` returned the arm's dst, `[n_embd, n_tokens, 4]`, and
`message()` went on reading it with the chain's layout, `[n_embd, 4, n_tokens]`: a row step of
`messages->nb[2]`, which on the arm's dst is `n_embd * n_tokens * 4` = **4,718,592 bytes** at
ub 512 - one 2,304-float row (three pages) at the start of each `[2304, 512]` slab, 512 rows
far, in a tensor that ends after four. The copy kernel behind the `ggml_cont` read them: reads
only, one row a group, gaps of 0x480000, exactly the kernel log's fit. `ggml_view_2d` did not
object because it checks a view's contiguous size, not its strides. The attention call site
had been given the arm's layout (`took ? ... : ...`); `message()` had not, and the device-free
test and the op cases covered nq 1 and 3 on the 5-row store only - the export's nq 4 call on
the 2-row bank store (`node_3060`, `node_3064`) was in neither.

The fix (0204, 0206): every reader of a mix result's columns goes through one function,
`inkling2_mix_column`, which returns `ggml_depth_mix_column` (a contiguous view, column `c` at
`c * nb[2]`) for the arm's dst and the chain's strided view plus `ggml_cont` otherwise - the
expression the graph always had, so the knob unset is today's graph node for node. Both
branches assert `ggml_view_within_src` (new in ggml.h: a view's furthest byte, strides and
offset included, against its source), so a wrong layout aborts at graph build with no device
involved.

The replay (`tests/test_depth_mix.cpp`, `measured-here` on the recipe host): the loop's
message path at the export's ub 512 shapes on the CPU backend. `arcint-depth-mix
fault-before` computes the old read: `SEGV ... caused by a READ memory access` in
`ggml_compute_forward_dup_bytes` under ASan, signal 11 without it. `fault-after` computes the
fixed read: clean, the four columns equal to the op's. In the unit run the same shapes are
checked without computing: the old view's row step is asserted to be `2304 * T * 4` and the
view to leave its source (12 views over the three ubatch sizes), red first against a reach
check that passes everything (12 failures).

**The three folded-norm op failures: `depth_mix.cl` never added the norm weight's offset.**
The kernel adjusted `blk`, `cur`, `qry` and `dst` by their offsets and not `nw`: the dispatch
passed `offn` and the kernel read the weights from the start of their buffer - in
test-backend-ops, whatever tensor the buffer begins with, hence ERR ~2 on every `with_norm`
case and none elsewhere; in the model, the start of the weights' buffer. One line: `nw = nw +
offn`. The handed-over host port could not see it because it gave every argument its own
buffer at offset 0.

**What now tests it** (`tests/test_depth_mix.cpp` in this repository, a `unit` ctest of the
recipe; all `measured-here` on the recipe host, device-free):

1. *The rule on the export's lines.* `tests/depth_mix_export_lines.inc` holds 69 lines of the
   exported graph verbatim (23 a ubatch size: the states' `RMS_NORM` lines `node_91`,
   `node_1108`, `node_2118`, `node_8944` on the 5-row store and `node_3060` on the bank's 2-row
   store, at T and at 1; the `MUL_MAT` lines `node_95` .. `node_9639` that pair states with 1,
   3 and 4 query columns; the two `CPY` lines of cur into a store row; the two `CONT` lines of
   a chain column read). The 39 prefill calls they describe are built with the model's
   constructors, compared with the lines' ne and nb, and must be accepted: k 1..4 with nq 1
   and 3, the nq 1 calls again with a view-backed norm weight, and the bank's k 1, nq 4. The
   15 decode lines and ten out-of-scope variants must be refused, each by its reason. Red
   first, on every run: the handed-over rule (`nw-view`) fails exactly the 12 folded calls, a
   rule that accepts everything fails 25 checks.
2. *The launch plan.* `ggml_depth_mix_plan_make` (pure; the dispatch binds its strides, counts,
   local size and bounds from it) on every call: each argument's reach equals its tensor.
3. *The kernel's own text.* `depth_mix.cl` is compiled as C++ into the test (256 lanes as
   threads, `barrier()` a real barrier, the local memory shared) and run at the plan's
   arguments with every source at its own non-zero offset of one buffer, as on the backend,
   against the CPU op: nmse 2e-15 .. 2e-14 on the 13 calls of ub 512 and 4 of ub 2048. The red
   build of the same text (`-DDEPTH_MIX_RED_NW`, the offset dropped again) must not match and
   does not (nmse > 100 on all five folded calls).
4. *The fault's read*, as above.

`test-backend-ops` (0207) now has the exported shapes only, 45 cases: a ubatch size (512,
1024, 2048) has the 5-row store's k 1..4 at nq 1, nq 1 with the folded norm, and nq 3; the
2-row bank store's k 1 at nq 4; and two `cols` cases that add the columns' readers as the
model has them (a column view, an RMS norm, summed) on the bank's four columns and on a
full-attention layer's three. The case string is
`DEPTH_MIX(n_slot=..,k=..,nt=..,nq=..,with_norm=..,cols=..)`.

**The fold is back, with its own switch.** `GGML_OPENCL_DEPTH_MIX=1` folds the output norm
where the chain norms one mixed column with one weight and no controller sits between (an SWA
layer's attention in the prelude and the first pass, the ffn mix without messages, the output
mix). `GGML_OPENCL_DEPTH_MIX=nofold` takes the arm and leaves every norm a node: the card
bisect of the fold against the mix.

**Bounds.** The arm's five buffers are checked before every launch, always (the plan's reach
plus the tensor's offset against `CL_MEM_SIZE`); a violation aborts with the buffer's name.
`GGML_OPENCL_BOUNDS=1` adds the other arms' checks to the same run. Note what this check does
not see and never could have: the fault was a *different* node's launch (the copy of a bad
view), which is why the view is now refused where it is built.

**Offline gate** (`measured-here`, ocloc 26.27.1, bmg-g21, the backend's options plus
`-cl-intel-256-GRF-per-thread`): base, `DEPTH_MIX_RED` and `DEPTH_MIX_RED_NW` compile, each
`grf 256 simd 32 slm 1024 scratch 0 stack_calls -`, no ISA call, no retry dump: `ocloc 3/3
variants`.

**Prediction, revised before any card run of this revision (`inference`).** The earlier
120-160 ms a pass priced the bytes at the reference's rate and nothing else. The kernel also
passes 15 barriers a work-group at k 4, nq 1 (two reduction rounds of seven, one before), 23
with the fold, and a 256-lane group at SIMD32 is 8 hardware threads. Assumed, not measured:
~80 groups in flight at 256 GRF (640 threads), so a 512-token call is ~6.4 waves, and a group
at 25-45 us (the loads, ~200 flops a lane, the barriers at 1-2 us each):

| ubatch | us a call (k 4, nq 1) | ms a pass (~960 / 480 / 240 calls) | against |
|---|---|---|---|
| 512 | 150-300 | 145-290 | the chain 765 (699 with 0187), the reference 399 |
| 1024 | 300-600 | 145-290 | |
| 2048 | 600-1200 | 145-290 | |

pp4096 at ub 512, from the measured rows: 0187 on, 768.8 t/s is 5.33 s a pass, less 409-554 ms
= **833-858 t/s**; 0187 off, 719.7 t/s is 5.69 s, less 475-620 ms = 785-808 t/s. Not priced
and on top (`inference`): the states' `rms_norm`, the cur copy, the column copies and, with the
fold, one `rms_norm` and one `mul` a folded call. What would falsify it: `test-backend-ops
perf` above 300 us a call at `n_slot=5,k=4,nt=512,nq=1`; above ~415 us a call the arm is
behind the reference and the barriers are the first suspect (the two-pass form without the
register array, or the reduction in one round, is then the next step - and the place where an
inline-vISA reduction would be judged, from the ISA dump and that measured gap, not before).

**Card commands for this revision, in order** (the B60; `<dev>` the card's backend name as
test-backend-ops lists it; every step stops the sequence if it fails):

```
# 0. the arm's op cases are found at all: must list 45 cases, none "not supported"
GGML_OPENCL_DEPTH_MIX=1 test-backend-ops support -b <dev> -o DEPTH_MIX
# 1. ONE model-shaped case, bounds on: the loop's messages with their readers
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 test-backend-ops test -b <dev> \
  -o "DEPTH_MIX(n_slot=2,k=1,nt=512,nq=4,with_norm=0,cols=1)"
#    then one folded case
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 test-backend-ops test -b <dev> \
  -o "DEPTH_MIX(n_slot=5,k=4,nt=512,nq=1,with_norm=1,cols=0)"
# 2. the suite: 45 cases, 0 FAIL
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 test-backend-ops test -b <dev> -o DEPTH_MIX
# 3. the reds: every case must FAIL; then only the 12 with_norm=1 cases must FAIL
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 GGML_OPENCL_DEPTH_MIX_OPTS=-DDEPTH_MIX_RED \
  test-backend-ops test -b <dev> -o DEPTH_MIX
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 GGML_OPENCL_DEPTH_MIX_OPTS=-DDEPTH_MIX_RED_NW \
  test-backend-ops test -b <dev> -o DEPTH_MIX
# 4. the model, bounds on, one short pass first (-p 512 -r 1), then pp4096: one TAKEN line,
#    a stats line a graph with calls == nodes > 0, no "DOWNGRADE ... op=DEPTH_MIX", and the
#    kernel log clean (no fault, no CAT error) before anything else is run
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 GGML_OPENCL_DEPTH_MIX_STATS=1 \
GGML_OPENCL_CPY_TRANSPOSE=1 GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1 \
  llama-bench -m <Inkling IQ4_NL> -fa 1 -p 512 -n 0 -b 4096 -ub 512 -r 1 2>&1 | \
  grep -E "depth_mix:|DOWNGRADE|reaches"
#    (repeat with -p 4096)
# 5. the answer-level bar: llama-perplexity's KL against the knob-off arm's logits on the same
#    card and window (-b 512 so the prefill path runs), then the same with =nofold if it moves
# 6. op perf, the prediction's check
GGML_OPENCL_DEPTH_MIX=1 test-backend-ops perf -b <dev> -o DEPTH_MIX
# 7. pp4096, 0187 on, the arm off / on, ub 512 / 1024 / 2048
for ub in 512 1024 2048; do
  for dm in 0 1; do
    GGML_OPENCL_DEPTH_MIX=$dm GGML_OPENCL_CPY_TRANSPOSE=1 \
    GGML_OPENCL_KQ_MM_ID_I8=1 GGML_OPENCL_KQ_MM_ID_I8_WIDE=1 \
      llama-bench -m <Inkling IQ4_NL> -fa 1 -p 4096 -n 0 -b 4096 -ub $ub -r 3
  done
done
```

The command list further up (written for the previous revision) is superseded by this one: its
`-p 'k=2 nt=512 nq=1'` matches no case (the params string is comma-separated, and has
`n_slot` and `cols` now), and its model run read no bounds or stats line.

**The model's own graph, before and after** (`measured-here` on the recipe host:
`test-export-graph-ops` on the CPU, c 4096, fa on, ub 512, with `GGML_OPENCL_DEPTH_MIX=1`, so
the lines are what `src/models` emits, not a reconstruction). The handed-over tree's export has
the faulting node as a line: a `CONT` to `[2304, 512]` whose source is `[2304, 512]` with nb
`4, 4718592, 2415919104` - a row step of one slab, a 2.4 GB span over an 18.9 MB tensor. This
revision's export has no source with that row step at any of ub 512, 1024, 2048, and no copy of
a mix column at all. It has 12 distinct `DEPTH_MIX` lines a ubatch size, the shapes the op
cases and the unit test use: the 5-row store's k 1..4 at nq 1 (three sources, and four with
the norm weight) and nq 3, and the bank store's k 1 at nq 4 (`[2304, 1, T]` at slab step
18432, cur `[2304, T]`, queries `[2304, 4]`). `test-backend-ops --test-file <that export> -o
DEPTH_MIX` replays exactly those 12 lines (12/12 on the CPU backend here), which is step 2b of
the card list: the card's op cases from the graph's own lines.

**State of this revision** (`measured-here`, the recipe host): the series applies from the pin
in byte order, 150 patches, applied tree `92f3a0347c66d7001b160ae950493fd47a6037a8`; the
recipe (package build and the `unit` set) is green, 51 of 51, `depth-mix` among them;
test-backend-ops lists 45 `DEPTH_MIX` cases and passes them CPU against CPU (the graphs
build and the readers run; it says nothing about the card).

```
# 2b. after step 2: the graph's own DEPTH_MIX lines replayed (12 a file, 0 FAIL)
for ub in 512 1024 2048; do
  GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 test-backend-ops test -b <dev> \
    --test-file <knob-on export, ub $ub> -o DEPTH_MIX
done
```

**Review of the revision above, and a third defect it found (2026-11-24; `code`, and
`measured-here` on the recipe host; still no card run).** An independent code read confirmed
the fault's cause and the offset, and found what neither the op cases nor the replay could
see, because both compared the op with its own CPU twin:

**The op mixed RAW states; the chain and the reference mix NORMALIZED ones.** In the chain,
`states = ggml_rms_norm(states)` overwrites the variable, so both products read the normalized
states: the scores and the weighted sum. The reference's kernel says the same in one line,
`pw = p / sum(p) * inv`. `depth_mix.cl` and the CPU op used each state's `1/rms` for the score
and then accumulated `p[s] * v_s` on the raw rows. A norm after the mix hides a global scale,
not this: it is a per-state reweighting. It was in every revision the card ran, and the
card's 16 green op cases were green against the same arithmetic.

- Red first (`measured-here`): the unit test now builds **the model's chain** node for node
  (cpy, rms_norm, repeat, mul_mat, soft_max_ext, cont(transpose), mul_mat, and the column's
  rms_norm and mul where the call folds) on the exported shapes, with the store's rows at
  different scales (0.25 .. 4, cur at 3), and compares the op with it. Against the previous
  revision's library and kernel text: 16 of 16 replayed calls fail on both, nmse 0.0091 to
  0.64 (0.249 at k 1, nq 1; 0.0096 with the norm folded - the norm hides most of it at k 1, not
  all).
- The fix: the weights on the raw rows are `p / sum(p) * inv_s`, in the kernel (both branches)
  and in the CPU op. Green: the CPU op against the chain over all tokens nmse 2e-14 .. 1e-13,
  the kernel's text against the chain 7e-15 .. 2e-13, on the 13 calls of ub 512 and 4 of ub
  2048. The red build `-DDEPTH_MIX_RED_RAW` (the weights without `inv_s`) is kept in the test
  and must miss the chain: 0.0091 .. 0.68.
- The op cases (0207) now set the store's rows and cur at those different scales too, so the
  same red is visible on the card.

**The knob-on model against the knob-off model on the CPU** (`measured-here`, the recipe host,
no GPU: llama-perplexity, Inkling 2 Mini IQ4_NL, c 256, b 256, 3 chunks, 381 scored tokens,
the knob-off logits as the base; the CPU backend serves the op with its reference):

| arm | mean KL, nats | max KL | same top p | PPL (base 2.5907) |
|---|---|---|---|---|
| knob off again | 0.000000 | 0.000024 | 100.000 % | 2.5974* |
| `GGML_OPENCL_DEPTH_MIX=1` | 0.001537 ± 0.000238 | 0.0596 | 98.950 ± 0.523 % | 2.5915 |
| `=nofold` | 0.001549 ± 0.000219 | 0.0364 | 99.213 ± 0.453 % | 2.5877 |

(* the tool's final estimate line of that run; its KL columns are the comparison.) The mean KL
is a twentieth of the 0.03-nat bar. The top-p agreement of the folded arm is 1.05 points down
on 381 tokens (4 tokens), which is AT the 1-point bar and inside its own ±0.52: this run does
not decide that cell, a longer one on the card does (card step 5, both arms). The op agrees
with the chain to 1e-7 relative a call in the unit test; that a difference of that size
reaches 1.5e-3 nats at the logits through ~110 mix calls and the expert routing is
`inference`, not measured. This check was not run on the raw-state revision, so its
discrimination is not measured either; the chain comparison above is what failed that revision.

**The reds, and which cases each must fail** (45 op cases; the unit test has the last two
compiled in):

| build option (`GGML_OPENCL_DEPTH_MIX_OPTS`) | what it breaks | must FAIL |
|---|---|---|
| `-DDEPTH_MIX_RED` | the first state's weight doubled: the mix's direction, not its scale | all 45 |
| `-DDEPTH_MIX_RED_RAW` | the weights without `inv_s` (the third defect) | all 45 |
| `-DDEPTH_MIX_RED_NW` | the norm weight's offset dropped (the second defect) | the 12 `with_norm=1` cases only |

The previous `-DDEPTH_MIX_RED` (the softmax denominator dropped) is gone: it was a per-column
scale, and on the 12 folded cases the norm cancelled it, so it could not fail them. Offline
gate for the four builds: `ocloc 4/4 variants`, each `grf 256 simd 32 slm 1024 scratch 0`.

**The knob is the model's, on every backend** (`code`). `GGML_OPENCL_DEPTH_MIX` is read by
`src/models/inkling2.cpp`, not by the OpenCL backend alone: set, the graph carries
`GGML_OP_DEPTH_MIX` whatever the backend. The CPU backend serves it with the reference (the
table above is that), the OpenCL backend with the kernel, and any other GPU backend would send
the node to the CPU. It is not gated to OpenCL because the CPU arm is the device-free
model-level check.

**Card step 3 above gains one line** (after the two reds there):

```
GGML_OPENCL_DEPTH_MIX=1 GGML_OPENCL_BOUNDS=1 GGML_OPENCL_DEPTH_MIX_OPTS=-DDEPTH_MIX_RED_RAW \
  test-backend-ops test -b <dev> -o DEPTH_MIX      # all 45 must FAIL
```

**State of this revision** (`measured-here`, the recipe host): supersedes the state paragraph
above. The series applies from the pin in byte order, 150 patches, applied tree
`e564f5b5366b9f871bf74a0b5dfe7295c3476cbf`; the recipe result is in the commit that adds this
paragraph. The knob-on exports are unchanged by this revision (the graph is the same; the
op's arithmetic changed).
