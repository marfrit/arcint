# Research: inferzero (a Level Zero engine for Intel GPUs, at source) and 128 registers a thread on the B60

inferzero (`github.com/rizagp/inferzero`, Apache-2.0; its README: "Formerly arcinferzero (renamed 2026-11-10)")
is a C++ inference engine on Level Zero with its own OpenCL C kernels (`zero/kernels/*.cl`). It loads GGUF files
and runs on Windows. It was built for Intel laptop iGPUs; its switches for discrete B-series cards are in
`zero/xe2/engine_xe2.h` and its record for them is `docs/xe2-tuning.md`.

This is a narrow read of it, done 2026-11-12 at `39e8d1a7` together with its issue #4, against arcint's OpenCL
kernels (arcint's llama.cpp series, patches 0001-0117, release 1.6.2; kernels in
`ggml/src/ggml-opencl/kernels/`, dispatch in `ggml/src/ggml-opencl/ggml-opencl.cpp`). Nothing of inferzero was
built or run here. One item was measured here the same day: arcint's prompt GEMM kernels built at 128 registers a
thread instead of the shipped 256, on the B60 (section 1).

**Evidence classes.** `code`: read in the source named. `paper`: a document says so (inferzero's documents or
issue #4, or arcint's own records quoted as what they say they measured). `measured-here`: the offline
compiles and the B60 runs of 2026-11-12 in section 1. Where an inferzero document and its code differ, the code is stated and the difference
named.

**Who measured inferzero's numbers.** The owner's machine is a Core Ultra 5 125H iGPU (README: "Built and tuned
on a Core Ultra 5 125H with 32 GB of RAM: no discrete GPU, no XMX matrix engines"). Every discrete-card figure
in its documents is "Measured by a user on Arc Pro B60s (issue #4)": a user with four B60 on Windows, on one card
unless a GPU count is given. "The 140T user" below is issue #5 (an Arrow Lake H iGPU). All of these are `paper`
for arcint.

## Published one-B60 figures, side by side (not a ranking)

| model | inferzero (`paper`: README, issue #4's user) | arcint (`paper`: arcint's README, 1.5.3, its own record on this B60) |
|---|---|---|
| Qwen3.8 27B | IQ3_S; 4K prompt 1,024 tok/s with H8 on; generation 23.2, 49.3 with MTP | Q4_K_M; 512-token prompt 843.70 t/s served; 128 tokens 19.90, MTP off |
| Qwen3.6 35B-A3B | quant not named in that README section (the tuning document's 35B decode rows are UD-IQ4_XS); 4K prompt 3,232; generation 78.8, 113.7 with MTP | Q4_K_M; 512-token prompt 2331.85; 128 tokens 84.94, MTP off |

What differs between the columns: the files and their weight types; prompt depth 4,096 against 512; Windows
against Linux with the xe driver; a user's runs against this card host; inferzero's own tools against
arcint's served path read from its `/metrics` counters; H8 on for inferzero's 27B prompt, which rounds the
weights a second time (section 4); MTP off and an f16 KV cache in arcint's rows; the builds and dates (inferzero's
27B prompt figure is of 2026-11-08, its 35B prompt figure of 2026-11-02; arcint 1.5.3 of 2026-11-06). The rows do not rank the engines.

## 1. The prompt GEMMs at 128 registers a thread (`measured-here`)

### What inferzero does (`code`; figures `paper`)

- `xe2_gemm_grf()` (`zero/xe2/engine_xe2.h`) passes no register-file option for the dense prompt GEMM
  (`zero/kernels/gemm.cl`) on GPUs with 16-lane XMX, so the driver default of 128 applies. `ARC_GEMM_GRF=256`
  restores `-cl-intel-256-GRF-per-thread`. Default since 2026-11-02.
- The 2-8 token GEMV (`xe2_gemv_mt_grf()`) and the one-token GEMV (the default register file on every GPU, not a
  discrete-card choice; `ARC_GEMV_GRF=256` for the large file) also run at 128 there. The MoE expert GEMMs stay at 256 (`xe2_moe_grf()`; `ARC_MOE_GRF=128` is opt-in and "not yet
  measured on a B-series card", `docs/xe2-tuning.md`).
- The kernel that runs at 128: int8 DPAS (`intel_sub_group_i8_i8_matrix_mad_k32`), a sub-group tile of 16 tokens
  x 16 columns (`TM` 16, or 32), every DPAS from a zero accumulator and its 8 results scaled in float per 32
  values. Weights come from a separate just-in-time decode into an int8 copy (W8; re-quantized for Q2_K, Q4_K and
  Q5_K). Activations are int8 per 32, read with `intel_sub_group_2d_block_read_8b_8r32x2c` and a 2D prefetch of
  the next 64 bytes.
- Measured (`paper`, issue #4's user, Qwen3.8 27B IQ3_S, 4K prompt, `runz --bench-pp 4096`): 449.7-450 tok/s at
  256 against 608-617.5 at 128 (+37 %, 2026-11-02 and 11-06); in a sweep on package 4aae0c5 (2026-11-08) 770 at
  the default against 514 at 256. On the owner's 125H, 128 registers made the 35B's prompt 2.7-3.5x slower
  (519 -> 194 tok/s for the dense GEMMs), so the iGPUs keep 256. For the wide H8 tile (`gemm_xw.cl`, 32 x 64)
  the tuning document gives ocloc `bmg`: no spill at 256, 5.6 KB at 128. It gives no ocloc spill figure for
  `gemm.cl` at 128 on `bmg`.

### arcint before this run (`code`, `paper`)

- Every arcint prompt GEMM program is built with `-cl-intel-256-GRF-per-thread`: the 2D K-quant program, the
  tile and MUL_MAT_ID program, the few-column, sub-group-8 int8 and int8 MUL_MAT_ID programs (`code`,
  `ggml-opencl.cpp`). A prompt kernel built at the default register file on the B60 exists, and it is not a GEMM:
  the gated delta-net prompt kernel of patch 0003, recorded at 128 GRF with 875 us against 5,692 for the kernel it replaced
  (`paper`, `contrib/llama.cpp/README.md`).
- Patch 0013's record: "the int8 kernel at 16 tokens a sub-group in 128-GRF mode (8 threads an XVE instead of
  4): spills, 37 instructions a DPAS". `llama-engine-kernel-gap.md`: for the int8 kernel's ceiling "8 threads an
  XVE (128 GRF) does not fit"; a DPAS microbenchmark with operands in registers reached the rated peaks "at 128 or
  256 GRF". No 128-register trial of the fp16 2D kernels or of the tile kernels was recorded.
- This run adds the int8 kernels at their shipped 32 tokens a sub-group, the fp16 2D kernels, and the tile, ID and
  ID2 kernels, compiled offline and timed on the card.

### What was built

One tool build (`test-backend-ops`) of arcint's llama.cpp tree: the pin with patches 0001-0117 plus the
`IQ4_KV_MUX` measurement switch of the Strata #4185 read (`research-concept-mining.md`; it touches only the
decode matvec, is off unless defined, and is not part of the series). Kernel sources: `mul_mm_kq_2d.cl` md5
`1deceea2…`, `mul_mm_kq_f16.cl` `69655c6e…`, the same files in the offline compiles and in the build. One library
for every arm; the arms differ in environment only. The three switches are recorded in
`contrib/llama.cpp/README.md` (0013, 0014, 0054).

| arm | environment | effect |
|---|---|---|
| base | none | shipped: Q4_K on the int8 2D kernel, Q5_K / Q6_K on the fp16 2D kernels, IQ types on the tile / ID / ID2 kernels, all at 256 |
| f16 | `GGML_OPENCL_KQ_2D_I8=0` | Q4_K on the fp16 2D kernel, 256 |
| g128f | `GGML_OPENCL_KQ_2D_I8=0 GGML_OPENCL_KQ_2D_OPTS=-cl-intel-128-GRF-per-thread` | the 2D program at 128, Q4_K on fp16 |
| g128i | `GGML_OPENCL_KQ_2D_OPTS=-cl-intel-128-GRF-per-thread` | the 2D program at 128, Q4_K on int8 |
| iq128 | `GGML_OPENCL_IQ_OPTS=-cl-intel-128-GRF-per-thread` | the tile / ID / ID2 program at 128 (the option also reaches the IQ matvec programs, which these 512-column cases do not use) |

The option lands after the shipped `-cl-intel-256-GRF-per-thread` in the build string. Offline, with both options
passed, IGC compiled at 128: the later option wins.

### Offline compiles (`measured-here`, ocloc 26.27.39122.11, `-device bmg`, arcint's compile options and shipped defines)

Spill sizes in bytes, from the `.spill size` line of IGC's assembly dump. "Retry" is IGC's second compile, dumped
as a `_1_` file.

The 2D program (`mul_mm_kq_2d.cl`, TM 64, WG 16, AT 32; the int8 kernels TMM 32):

| kernel | 256: spill, instructions | 128: spill | 128: instructions |
|---|---|---|---|
| `kernel_mul_mm_kq2d_q4_K` (fp16) | 0, 2,581 | 0 | 2,582 |
| `_q5_K` | 0, 2,839 | 0 | 2,835 |
| `_q6_K` | 0, 2,821 | 320 | 2,879 |
| `_q3_K` | 0, 2,999 | 256 | 3,095 |
| `_q2_K` | 0, 2,676 | 0 | 2,691 |
| `_q4_0` | 0, 2,471 | 0 | 2,462 |
| `_q4_K_i8` | 0, 1,542 | 7,808; retry 7,744 | 2,141; retry 2,448 |
| `_q4_0_i8` | 0, 1,611 | 9,664; retry 4,032 | 2,060; retry 2,330 |

The DPAS counts do not change (128 a fp16 kernel, 36 and 32 the int8 ones). With smaller token tiles at 128 (TM 32
with TMM 16, TM 16 with TMM 8 and AT 8) no 2D kernel spills, but the int8 kernels' DPAS counts fall to 0-2: those are not
working builds of the int8 kernels, which the host dispatches for 32 tokens a sub-group (0014's record). The
smaller-tile builds were not timed.

The tile program (`mul_mm_kq_f16.cl`, SG 16, XT 32, XSG 8, RG,TG,NSGM,NSGN 4,4,2,8 plain and 4,4,2,4 for
MUL_MAT_ID, both sets compiled with the same results): at 256 none of its 35 GEMM kernels spills. At 128 every
plain and `_id_` tile kernel spills 4,288-7,232 bytes; IGC dumped a retry for 31 of the program's kernels (the 29
plain and `_id_` ones, and ID2 q4_K and q5_K). The ID2 kernels
spill less: iq4_xs and iq4_nl 0, q6_K 640, iq3_xxs 1,472, q4_K and q5_K 1,536-2,176.

### The option reached the card's compile (`measured-here`)

One perf leg for each 128-register build (g128f for the 2D program, whose build string g128i shares; iq128 for
the tile program), on the q5_K case of 4,096 rows at 512 tokens, with the driver's kernel cache off
(`NEO_CACHE_PERSISTENT=0`) and IGC's dump on:

- g128f: the 2D kernels at 128 with the offline spills (q4_K 0, q5_K 0, q6_K 320, q4_K_i8 7,808 and a 7,744
  retry); the tile kernels at 256 with 0.
- iq128: the dense iq4_xs and iq3_s tile kernels at 128 with 4,288 (retry), ID2 iq4_xs and iq4_nl at 128 with 0;
  the 2D kernels at 256 with 0.

These legs compile cold and are not timing samples.

### Timing (`measured-here`, 2026-11-12, the B60)

`test-backend-ops perf -b GPUOpenCL`, one process a leg, the arms interleaved in each round, the median of the
us/run figures over 3 rounds (iq128: the mean of 2 rounds, against base's median of 3). Every product has 512 columns (tokens). No model was loaded; the A770 was
not used. Percent is speed against base, + faster.

| op | type | rows x K (experts, used) | base | f16 | g128f | g128i | iq128 |
|---|---|---|---|---|---|---|---|
| MUL_MAT | q4_K | 17408 x 5120 | 1098.3 us | 1237.3 (-11.2 %) | 1426.5 (-23.0 %) | 4428.2 (-75.2 %) | |
| MUL_MAT | q4_K | 4096 x 14336 | 824.6 | 1148.0 (-28.2 %) | 1447.2 (-43.0 %) | 3371.2 (-75.5 %) | |
| MUL_MAT | q4_K | 5120 x 17408 | 1192.2 | 1438.5 (-17.1 %) | 1929.6 (-38.2 %) | 5147.0 (-76.8 %) | |
| MUL_MAT | q5_K | 17408 x 5120 | 1309.1 | 1313.6 (-0.3 %) | 1539.2 (-14.9 %) | 1540.5 (-15.0 %) | |
| MUL_MAT | q5_K | 4096 x 14336 | 1170.6 | 1171.2 (-0.0 %) | 1469.8 (-20.4 %) | 1469.3 (-20.3 %) | |
| MUL_MAT | q5_K | 5120 x 17408 | 1501.9 | 1505.0 (-0.2 %) | 1877.8 (-20.0 %) | 1881.0 (-20.2 %) | |
| MUL_MAT | q6_K | 17408 x 5120 | 1380.8 | 1382.9 (-0.2 %) | 1666.2 (-17.1 %) | 1651.0 (-16.4 %) | |
| MUL_MAT | q6_K | 4096 x 14336 | 1184.7 | 1183.8 (+0.1 %) | 1593.0 (-25.6 %) | 1596.7 (-25.8 %) | |
| MUL_MAT | q6_K | 5120 x 17408 | 1531.8 | 1529.6 (+0.1 %) | 2129.7 (-28.1 %) | 2163.7 (-29.2 %) | |
| MUL_MAT | iq2_xs | 4096 x 14336 | 1853.5 | | | | 3598.6 (-48.5 %) |
| MUL_MAT | iq3_s | 4096 x 14336 | 1829.0 | | | | 3559.2 (-48.6 %) |
| MUL_MAT | iq3_xxs | 4096 x 14336 | 2181.9 | | | | 3931.2 (-44.5 %) |
| MUL_MAT | iq4_xs | 4096 x 14336 | 1834.3 | | | | 3582.6 (-48.8 %) |
| MUL_MAT_ID | iq2_xs | 1792 x 2048 (32, 4) | 1388.5 | | | | 3234.7 (-57.1 %) |
| MUL_MAT_ID | iq2_xs | 768 x 2048 (128, 8) | 2151.6 | | | | 4122.3 (-47.8 %) |
| MUL_MAT_ID | iq3_s | 640 x 2560 (512, 10) | 8347.5 | | | | 14740.3 (-43.4 %) |
| MUL_MAT_ID | iq4_nl | 2560 x 640 (512, 10) | 5054.1 | | | | 5125.4 (-1.4 %) |
| MUL_MAT_ID | iq4_xs | 1792 x 2048 (32, 4) | 1685.7 | | | | 1891.1 (-10.9 %) |
| MUL_MAT_ID | iq4_xs | 768 x 2048 (128, 8) | 1768.4 | | | | 1998.1 (-11.5 %) |

What the table shows:

- No case got faster at 128.
- The fp16 2D kernel against itself (f16 against g128f, Q4_K): 13.3 %, 20.7 % and 25.5 % slower at 128. Q5_K
  (base and f16 against g128f / g128i): 15-20 %. Q6_K: 16-29 %. The Q4_K and Q5_K fp16 builds do not spill at
  128 and still lose 13-26 %. What costs them was not measured (no profile, no ISA comparison of the loops).
- The int8 2D kernel against itself (base against g128i, Q4_K), spilling 7.7-7.8 KB: 4.0-4.3x the time.
- The dense IQ tile kernels, spilling 4.3 KB with a retry: 1.8-1.95x the time.
- MUL_MAT_ID: iq3_s and iq2_xs take the tile kernel on sub-group 16 (`code`; it spills 5,248 bytes at 128): 1.8-2.3x
  the time.
  iq4_xs and iq4_nl take ID2 (`code`, the MUL_MAT_ID dispatch; no spill at 128): 1-12 % slower.
- The f16 arm restates 0013's int8 lead on Q4_K at 256: in speed the fp16 kernel is 11-28 % below the int8 one
  here (0013 recorded the fp16 kernel 9-15 % below the int8 one in device time at the model shapes).

### What the correctness evidence covers, and what did not run

- The shipped build: MUL_MAT with `type_a=q[2-6]_(K|0)`, 1033/1033 (the first pass); q5_K 104/104 (the second
  pass's dry run of the command, which also showed that the cases ran on the B60).
- The 128 arms: the build-proof and timing legs are perf mode, which runs each case without comparing it with the
  CPU backend; all exited 0. In the first pass, g128f's correctness leg (the same `q[2-6]_(K|0)` filter, the 2D
  program at 128) reported 526 cases OK and 0 FAIL, then was ended on purpose by the measuring session after 119 s
  (SIGKILL, exit 137) to start a shorter second pass inside the window; not a card fault and not a kernel failure. The second pass's four 128-register correctness legs (g128f q5_K, g128i q4_K,
  iq128 iq4_xs MUL_MAT and MUL_MAT_ID) and iq128's third timing round did not run: the window's 960-second budget
  was spent. So g128i's and iq128's outputs were never compared with the CPU reference. g128f's were for the first
  526 cases of its filter; how many of those dispatch to the 2D program was not counted (cases of 1 to 4 columns
  and small shapes take other programs, which the arm did not rebuild), and the int8 kernel was not in that arm.
- No served arm, no KL, no acceptance task: no kernel got faster.

### How arcint's build differs from inferzero's

- **Engine and choice of register file.** inferzero is a Level Zero engine with its own kernels and picks the
  register file per kernel family: 128 for the dense prompt GEMM and the GEMVs, 256 for the MoE GEMMs. Its 128 is
  "no option", the driver default. arcint's arms flipped one explicit option on whole programs (the 2D program,
  the tile program) and nothing else. Tiles, unrolling and read-ahead stayed as tuned for 256 (0013, and 0014's
  genetic search, both at 256). arcint's one-column decode matvecs are built without a register option, so they
  already run at the default 128; only the 2-4 column builds take 256 (`code`).
- **Tile.** inferzero's sub-group computes 16 tokens x 16 columns, its int8 DPAS results unpacked and scaled per
  DPAS. arcint's fp16 2D sub-group holds 64 tokens x 16 rows in 8 `float8` accumulators across K. Its int8 kernel
  holds 32 tokens x 16 rows and a register read-ahead of the next sub-block's codes and activation tile. arcint's
  tiles are larger per sub-group; the live register counts of the two kernels were not compared.
- **int8 against fp16.** inferzero's measured kernel is int8 for every weight type, from a pre-decoded int8 copy.
  arcint's int8 prompt kernels on the B60 cover Q4_K and Q4_0 and decode the GGUF codes inside the kernel; Q5_K,
  Q6_K and the IQ types ran fp16 DPAS.
- **Where it spilled.** arcint at 128: the int8 kernels (7.8-9.7 KB in the first compile, 4.0-7.7 KB in IGC's retry), the tile and ID
  kernels (4.3-7.2 KB, retries), Q6_K and Q3_K fp16 2D (256-320 bytes). Not the Q4_K, Q5_K, Q2_K and Q4_0 fp16
  2D kernels, not ID2 iq4_xs / iq4_nl.
- **What was measured.** inferzero: a 27B IQ3_S 4K-token prompt through its engine on Windows, by a user. arcint:
  the op suite's 512-token products on Linux (xe) on this card.

**Verdict, as narrow as the test:** arcint's prompt GEMM kernels as they are, rebuilt at 128 registers a thread,
on the B60, timed in the op suite at 512-token products, are faster in none of the 19 cases: from 1.4 % slower (ID2
iq4_nl, two rounds for the 128 arm, no per-round spread kept) and 11-12 % slower (ID2 iq4_xs, no spill) up to
4.3x the time (int8, spilled). This is not a verdict on inferzero's kernel, or on 128 registers for a kernel shaped
for them. The smaller-tile builds that compiled without a spill were not timed.

## 2. The decode "MoE prefetch" (`ARC_GEMV_PF`)

- **What it is** (`code`, `zero/kernels/decode.cl` `gemv`, the `#ifdef GEMV_PF` block). Without the switch a
  sub-group block-reads each row's two superblocks into a local-memory window, with a `sub_group_barrier` before
  and after the decode, row by row. With `-DGEMV_PF` the reads of all `GROWS` rows of the step are issued into a
  register array first. No prefetch builtin, nothing read ahead of the current step; the sums are unchanged.
- **What it reaches** (`code`). `xe2_gemv_pf()` turns it on for discrete 16-lane XMX cards. `zero/engine.h`
  appends it to `gemv` only, never to `gemv_v`. `gemv_v_ok` sends IQ4_XS, Q4_0, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S,
  IQ1_M, IQ1_S and Q6_K to `gemv_v`. The expert kernels (`moe_gateup_q`, `moe_down_q`) are built without it. So it
  reaches the dense one-token GEMVs of Q8_0, Q2_K-Q5_K, IQ3_S and IQ4_NL, not the experts. The tuning document's
  table calls it "on for MoE too".
- **Measured** (`paper`, issue #4's user, Windows, 256 greedy tokens, best of 2 where the record says so): 35B-A3B UD-IQ4_XS 79.93 ->
  88.91 tok/s (+11 %, package fffaf1a, 2026-11-12); on package 2c132a9 (2026-11-09) the opposite sign, 78.77 with
  the switch and 85.15 without; 27B IQ3_S +3.2 to +5.6 % in four pairs (2026-11-06 to 11-12); the 35B also 84.08 and 83.88 on
  2026-11-06 (no gain; the tuning document and the code comment differ on which arm is which); a further 35B run
  on 2026-11-12 (package 18acafc) lists 79.12 with `GEMV_PF=1` beside 80.35 without the small work-groups. The
  owner asked on issue #4 for a rerun because of the sign.
- **arcint.** The matvecs (`mul_mv_kq_q8_1.cl`, `mul_mv_iq_q8_1.cl`) have no local-memory weight window and no
  barrier between rows: each lane `vload`s its own weight words from global memory into registers inside an
  unrolled loop over `N_DST` 4 rows (Xe2). The window and barriers `GEMV_PF` works around are not there (`code`).
  Whether IGC issues the 4 rows' loads before the first row's dot products was not checked (no ISA read).
- **Disposition:** the window `GEMV_PF` removes does not exist in arcint's matvecs (`code`); the load order IGC
  emits was not read; nothing tried.

## 3. Other prompt-path mechanisms

| inferzero mechanism | arcint | evidence | how the builds differ | disposition |
|---|---|---|---|---|
| int8 DPAS prompt GEMM on int8 activations per 32, each DPAS from zero, float rescale per 32 per token (`gemm.cl`) | int8 DPAS for Q4_K and Q4_0 on Xe2 (0013, 0099), activations per 256, the per-32 weight scale by an integer `mad16` into one super-block accumulator; fp16 DPAS for the other K-quants and the IQ types. An int8-activation GEMM in the CUDA MMQ shape was built and dropped in 0001 (KL +0.0028 nats, top-1 -0.9 points), per-32 float rescaling in 0013 ran 1,599-2,140 us against 1,229 for fp16 | `code`; `paper` (arcint's records) | inferzero decodes the weights in a separate pass into an int8 copy (lossy for Q2_K, Q4_K, Q5_K); arcint decodes the GGUF codes in the kernel | read; arcint's own per-32 forms were measured slower in its kernels (`paper`); inferzero's pass not tried |
| 2D activation loads with a 2D prefetch of the next 64 bytes (`-DX2D`, `-DX2D_PF`) | 2D block loads for both operands on Xe2 (0011), 32-row activation reads; no prefetch builtin in the shipped GEMMs. L1 prefetch of the next activation tile recorded as no gain (0012, 0013) | `code`, `paper` | inferzero loads 8 rows a load and prefetches; arcint loads 32 rows (its 8-row reads were slower in 0013) | arcint's prefetch trials were in its own kernels; inferzero's form read, not tried |
| weight prefetch n chunks ahead (`ARC_GEMM_WPF`, off; +3.6 % without H8, slower with H8, which is now the 27B default, `paper`) | register read-ahead of the next sub-block in the int8 Q4_K kernel (default); in the fp16 kernels a read-ahead was slower (0012, 0013) | `code`, `paper` | prefetch to cache against a register double buffer | read, not tried |
| f16 XMX with whole-K accumulation (`gemm_x16.cl`, `ARC_X16=1`; `paper`, inferzero's CHANGELOG of 2026-11-08 and issue #4: 400-550 tok/s against that package's int8 default of 766 at 4K; an earlier run 454-555 against 615.5; out) | arcint's form for most types (`mul_mm_kq_2d.cl` fp16, `mul_mm_kq_f16.cl`) | `code`; `paper` for the figures | inferzero decodes from its int8 copy once per 16-column group; arcint from the GGUF codes; activation scale per row maximum against a power of two per column | present in another form |
| wide H8 tile, 32 x 64 a sub-group (`gemm_xw.cl`; slower on the B60: 585 / 945 / 717 against 674 / 993 / 842 tok/s at 512 / 4K / 16K, `paper`) | two 16-row weight tiles a sub-group measured slower at every shape in 0013 (fp16 and int8 Q4_K); 0104 shares activation reads over 2 row sets on the A770 (+12.2 %) | `code`, `paper` | inferzero's is H8-only (whole-K integer sums) | read, not tried (slower on the B60 by inferzero's record, `paper`; arcint's two-tile forms of 0013 are a different build) |
| MoE expert GEMM: several column groups (`ARC_MOE_CG`) or tiles (`ARC_MOE_WY`) a work-group; `ARC_MOE_WY=4` +1.5 % at 16K (`paper`); default 1 x 1 | ID2 runs one expert tile of 32 (token, slot) pairs x 8 sub-groups a work-group; no several-tiles-a-work-group form found (the ID2 header and its selection read, the tile kernel's dispatch not line by line) | `code` | inferzero gathers int8 activations through a row map; arcint stages f16 activations per super-block in local memory | read, not tried |
| MoE: decode every expert of the chunk once, then GEMM (from ~1,152 tokens); long chunks amortize it (35B 16K: chunk 8,192 3,172 tok/s, 4,096 2,966, 2,048 2,624, `paper`) | no separate decode pass at any chunk size: the ID2, tile and int8 kernels decode inside the tile | `code` | the gain rests on the just-in-time decode, which arcint does not have | not applicable as built (`code`: arcint has no separate decode pass); read, not tried |
| "two int8 passes": the rounding residual quantized to a second int8 row through the same weights (`-DQ2F`, `q2_init`): Gemma 4, Qwen3 dense and Qwen3-ASR, the vision and audio encoders, and by default the up projection of Flash Next's prompt-chunk mixers (KL 0.0647 -> 0.0483 against f32, prompts 5-7 % slower, `paper`); not the Qwen3.8 27B or 35B-A3B paths | arcint's matvec activations carry two int8 terms per 32 (0001) | `code` | the same idea on different paths; no discrete-card measurement of it found in inferzero's documents | present in another form |

## 4. H8: a Hadamard rotation for the prompt GEMMs (`zero/kernels/h8.cl`, `zero/engine_h8.h`)

- **What it computes** (`code`). Weights and activations are rotated by a normalized Walsh-Hadamard transform in
  blocks of up to 1,024 along K (`h8_block`), which leaves the exact product unchanged. In H8R, the default form,
  both are then rounded to int8: one scale per weight column (`h8r_colmax`), one per token over the whole K. The
  DPAS then accumulates the whole K in int32; the only float work is the epilogue. Activations are rotated from
  their f32 values for Qwen. The weights are rotated from the int8 copy, so they are rounded a second time.
- **When** (`code`). On by default for Qwen dense models on discrete 16-lane XMX cards (`h8_auto_wanted`: not MoE,
  Gemma, Flash Next or Qwen3), with 4 column groups. The header comment of `engine_h8.h` still says it is off
  unless `ARC_H8=1`; `h8_init` turns it on by default, so the comment is stale.
- **Speed** (`paper`, issue #4's user, 27B IQ3_S): package d980970 (2026-11-08) 615 / 766 / 706 -> 677 / 1,024 /
  937 tok/s at 512 / 4K / 16K; on package 4aae0c5, before the fused decode, 493-500 against 619 at 512. 0.34 GiB
  more GPU memory. On the 35B -0.8 % (the changelog rounds it to -1 %; the earlier sweep: no gain), so not on MoE.
- **Accuracy** (`paper`). Against inferzero's own default path on the B60: mean KL 0.00120-0.00154, max
  0.024-0.078, the same top token 97.25-98.04 %. Its limit (`docs/xe2-tuning.md`): mean KL 0.015 against f32. The
  "same top token" figures compare the H8 arm with the default arm directly, not each with a shared reference, so
  they are not the quantity arcint's argmax bar measures. The adopted default's B60 row and the 140T row carry the same
  three figures (0.00154 / 0.064 / perplexity 6.2847 -> 6.2851), and a laptop row the same KL pair; the B60's own
  sweep that day reads 0.00136. Which run produced the B60 row cannot be told from the documents. Two perplexity bases for the same model and text name
  (6.6371 and 6.2847-6.2865) are not explained there.
- **arcint** (`code`). A fast Walsh-Hadamard kernel exists (`kernels/fwht_f32.cl`, 0095), for the KV cache
  rotations only (widths 64-256). No weight or activation rotation for a matrix product (searched `hadamard`,
  `fwht`, `walsh` over the kernels; `hadamard`, `rotat`, `quarot`, `spinquant` over the records). arcint's int8
  GEMMs round only the activations, per 256, and keep the weight codes exact.
- **Disposition:** read, not tried.

## 5. Per-token overhead in decode (`docs/xe2-tuning.md` sections 5 and 10)

| inferzero mechanism | measured (`paper`, issue #4's user unless named) | arcint | evidence (arcint side) | disposition |
|---|---|---|---|---|
| launch counting (`ARC_LAUNCH_STATS=1`) | 35B 632 launches, 2.89 ms host a 12.0 ms token; 27B 837 kernels, 4.4 ms host, "GPU stays busy" | 0049 counted ~4,150 kernels a Flash-Next verify graph (~3 us an enqueue); `LLAMA_DECODE_TIMING`, `GGML_SCHED_TIMING` | `paper` (0049), `code` (the switches) | has the equivalent |
| replay of a recorded command list (probe only, not in the engine) | 700 dependent kernels 1.790 ms appended, 0.459 ms replayed | the installed compute runtime offers no `cl_khr_command_buffer`; OpenCL on Level Zero's cheaper enqueue (1.6-1.9 against 2.63 us a kernel) did not move decode (`flash-next-llama-engine.md`): a cheaper enqueue, not a replay or a batched list, so no verdict on either | `paper` | not available on arcint's stack in this form |
| batched command lists (`ARC_BATCH`) | 27B no gain (21.25-22.67 against 23.72); 35B 49-51 against 78.77 on package 2c132a9, 85.11 / 83.42 / 76.66 against 79.93 on fffaf1a, below the prefetch's 88.91 and marked "to measure with the prefetch on"; off | as above | `paper` | not available in this form |
| skip the per-layer host read of expert ids when every expert is resident (`ARC_XALL`) | 35B 78.77 -> 88.05 tok/s, host waits 43 -> 3 a token | arcint's resident MUL_MAT_ID matvecs read the expert ids on the device (`ID_PROLOGUE`, `MV_ID_KERNEL`); with the expert cache (0021) the ids are read back once a step | `code`, `paper` | not applicable to the resident path (`code`: the ids stay on the device); with the expert cache one read-back a step remains (0021), not compared |
| row kernels over many work-groups for 1-8 tokens (`ARC_WIDE_ROWS`, default on) | 27B 23.19 -> 23.72 (+2.3 %) | partly: 0004's sub-group q8_1 quantizer and GLU split; the fused `kernel_rms_norm_mul` runs one work-group a row; norm and quantization are two launches | `code`, `paper` | read, not tried |
| small work-groups for few-row GEMVs (`ARC_GEMV_SMALL_WG`) | 27B no gain, 35B -30 %; off again | arcint's matvec work-group is 64 work-items already; few-row F32 / F16 weights take split-K paths (0005, 0022) | `code` | read, not tried (no gain / -30 % by inferzero's record, `paper`) |
| gate and up GEMVs in one launch (`ARC_GEMV_GU`) | 27B 22.79 against 23.72; off | not built; arcint shares the quantized activation between gate and up (0001), not the launch | `code` | read, not tried |
| small-op fusions (SwiGLU in the up GEMM's epilogue, norm + quantize) | not measured separately on a discrete card | 0022, 0049, 0057, 0065, 0074, 0114 fuse other small ops (0064 reshapes 0022's fused router) | `paper` | read, not tried in inferzero's form |

## 6. Further mechanisms with a gain measured on a discrete card

| inferzero mechanism | measured (`paper`, issue #4's user) | arcint | evidence | how the builds differ | disposition |
|---|---|---|---|---|---|
| prompt attention on int8 DPAS: int8 q.k and u8 P x int8 V, bit-identical to its DP4a kernel (`attn_x.cl`, `ARC_ATTN_XMX`, default on discrete) | 64K prompts: 27B +2.1 % (605.0 -> 617.7), 35B +10.2 % (1,814.2 -> 1,999.7); 4K +0.2-1 % | prompt attention on fp16 DPAS (0008, 0038), quantized K/V converted to f16 first (0037); int8 Q.K DPAS only in the opt-in decode GQA kernel (0088); no int8 P x V | `code` | inferzero's KV cache is int8 with per-row scales and P is quantized to u8 per 16-key tile | read, not tried |
| the 2-8 token GEMV (MTP / DFlash2 verify) at 128 registers (`xe2_gemv_mt_grf()`) | 27B MTP 43.3 -> 47.1 tok/s (+8.8 %), acceptance 61.1 % in both | for the K-quants (0009, fp16) and Q4_0, Q8_0, IQ4_XS, IQ4_NL (0115, int8), 2-16 columns go to few-column DPAS kernels at 256; the other IQ types, IQ3_S (the type of inferzero's figure) among them, take the q8_1 matvec up to 4 columns and the tile GEMM from 5 (`code`); the 2-4 column q8_1 matvec is at 256 on purpose ("2 to 4 columns keep their per-column values in registers only with 256 GRF", A770 4 columns 106 -> 355 GB/s, a `ggml-opencl.cpp` comment) | `code`, `paper` | inferzero's verify kernel is a DP4a GEMV with tokens in registers; arcint's is a DPAS product with lanes as rows | read, not tried on the B60 |
| tensor parallel over several cards, sums through host memory, prompt sums in ~2 MB pieces (`zero/mgpu/`, `ARC_TP_PIPE`) | 27B generation 1 / 2 / 3 / 4 cards 21.9 / 35.7 / 43 / 53 tok/s; 35B 77.5 / 92.2 / 95.3 / 101.7, its prompt slower split (3,232 -> 2,040-2,122) | no row or tensor split; llama.cpp's layer split over two different cards (0023, 0024) and the open campaign `layer-split-two-cards.md` | `code` | four equal B60 without peer access against a B60 and an A770 on PCIe 3.0 x4 | read, not tried |

Measured on a discrete card and not taken, by inferzero's record (`paper`): `ARC_GEMV_GU`, `ARC_BATCH` (35B open),
`ARC_GVROWS`, `ARC_GEMV_GRF=256`, `ARC_MOE_GR_GU/_DN`, `--mtp-drafts 7`, `ARC_X16`, `ARC_GEMM_XW`,
`ARC_GEMV_SMALL_WG`.

## Open, not tried

- arcint's fp16 2D kernels at a smaller token tile (TM 32 or 16) with 128 registers: they compiled without a spill
  offline (section 1) and are reachable through `GGML_OPENCL_KQ_2D_SHAPE` with `GGML_OPENCL_KQ_2D_OPTS`; not
  timed.
- The correctness legs of the 128 arms of section 1 (g128i and iq128 never compared; g128f for 526 cases, the number
  reaching the 2D program not counted).
- arcint's few-column verify kernels (0009, 0115) at 128 registers on the B60; inferzero's 2-8 token GEMV gained
  there (`paper`), arcint's record for 2-4 columns at 256 is from the A770.
- inferzero's H8 rotation, its int8 prompt attention, the issue order of the matvec's row loads (an ISA read),
  several MoE tiles a work-group, and a fused norm + quantize row kernel: read, not tried.
