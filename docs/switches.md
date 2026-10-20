# Switches

Charter item A7 (`docs/milestone-1.0.0.md`, met 2026-10-16): every switch that
1.0 keeps is documented, and diagnostic and red-case switches are named as such.

The promise (charter C4, operator 2026-10-16): the **operational** switches
are stable within 1.x, like the HTTP surface and the command-line flags.
Tuning, off-switch, red-case and diagnostic switches are for measuring and
may change or go without notice.

A switch here is an environment variable read by arcint, by one of its tools,
or by the llama.cpp patch series in `contrib/llama.cpp/patches` (0001-0063,
without the dropped 0031, 0055 and 0061). Switches the pinned llama.cpp tree reads by itself
(`GGML_OPENCL_PLATFORM`, `GGML_OPENCL_FUSE_RMS_ADD`, `GGML_OPENCL_F16_MROW` and
the others in upstream's `ggml-opencl.cpp`) are upstream's and not listed;
a patch that only carries one in its context does not add it.
Each switch is in exactly one class:

- **operational**: changes how a service runs; a unit file may set it.
- **tuning**: a shape or threshold whose default was measured. Override it to
  measure.
- **opt-in alternative**: none left. The eight that measured slower or
  neutral were removed from the series in 0.8.1 (operator, 2026-10-16),
  with their red-case and debug switches: the hyper-connection read fused
  (`FUSE_HC_MIX`, `HC_MIX_RED`, `HC_MIX_DEBUG`), the K-split F16 matvec
  (`F16_KSPLIT4`), the per-pair gather grid for a few tokens
  (`GATHER_PAIRS_MULTI`), the two grouped MoE matvecs (`KQ_GROUPED`,
  `KQ_IDG`), one int8 activation term (`KQ_ONE_TERM`), the bank prefetch
  (`MOE_PREFETCH`) and Infernix's LFRU cache policy
  (`LLAMA_EXPERT_CACHE_POLICY`, `_LFRU_BUDGET`, `_LFRU_EXACT`; patch 0055
  dropped whole); also `FUSE_RMS_SCALE`, an alias of `FUSE_0049`. They are
  in 0.8.0's patches.
- **off switch**: turns a default-on mechanism off, to compare or to fall back.
- **red case**: breaks something on purpose, to prove a test can fail.
- **diagnostic**: logging, timing, profiling, debug dumps.

Columns: switch, default (what the code does when the variable is unset),
what it does, the patch that added it (or `arcint` with its source file), and
the evidence with its class (`measured-here` or `code`). "?" means not
determined. "—" means no number recorded. Numbers are as the patch README
states them. Unless said otherwise, the boolean switches read `atoi(value)`:
`0` is off, anything else is on.

Four names are not in a table:

- `LLAMA_PPL_SEQ_WINDOWS`, `LLAMA_PPL_PER_SEQ`, `LLAMA_PPL_SEQ_FLIP` (0026)
  are test hooks of `llama-perplexity`, not of the library: a KV window per
  sequence, each sequence decoded on its own, the sequences' order flipped.
  They shape the KL arms of 0026 and 0029 (`contrib/llama.cpp/README.md`).

- `GGML_GALLOC_TOP` is a debug commit on the llama.cpp checkout, not in the
  series. It lists the N largest tensors of a reservation (N up to 64). If it
  stays, it is a diagnostic; it needs a patch first.

## llama.cpp patch series

### Operational

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_OPENCL_BANK_STREAM` | off | Copies each expert-bank tensor whole into a VRAM ring on a second queue during prefill. | 0032 | Flash-Next needle prefill 416.1 -> 497.6 t/s, B60 (`measured-here`). The Flash-Next unit sets it. |
| `GGML_OPENCL_DEVICES` | unset: one platform, as before | `P:D[,P:D...]` picks OpenCL devices by platform and device index, across platforms. Overrides `GGML_OPENCL_PLATFORM` / `GGML_OPENCL_DEVICE`. | 0023 | `0:0,1:0` lists both cards; unset lists the B60 alone (`measured-here`). |
| `LLAMA_EXPERT_CACHE_MIB` | unset: no cache unless the API sets a budget | Expert-cache budget in MiB, `A,B` for one budget a device. Used only when the API gives none. | 0021, 0024 | — |
| `LLAMA_EXPERT_CACHE_PROFILE` | unset: expert-major round robin | Path of a Strata STRP decode profile that ranks the cached experts. Read together with `LLAMA_EXPERT_CACHE_MIB`. | 0021 | — |
| `LLAMA_EXPERT_CACHE_CPU_MISSES` | off | `1`: a share of each decode layer's missed experts computed on the CPU from the bank (Strata's doorbell tier and its multi-token AVX2 rows, README 0056). | 0056 | Flash-Next decode +15.2 % on six answers, KL +0.0028 nats, B60 (`measured-here`). |

### Tuning

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_OPENCL_BANK_STREAM_MIN_TOK` | 256 | Least tokens in a ubatch for the bank stream. Below it a MoE GEMM gathers its routed experts. | 0044 | 173 tokens 1.56 s gathered against 1.84 s streamed; 317 tokens 1.96 against 1.84 (`measured-here`). |
| `GGML_OPENCL_BANK_STREAM_RING` | 4 (values below 2 give 4) | Slots in the bank-stream ring. | 0032 | 4 slots take 751 MiB of VRAM (`measured-here`). |
| `GGML_OPENCL_FA_DPAS2_NSG` | 32 | Sub-groups a work-group in the head-256 prompt attention (32 or 16). | 0038 | — |
| `GGML_OPENCL_FA_DPAS2_SPLIT` | 16 | Cap on key splits of that kernel; 1 turns the split off. | 0045 | — |
| `GGML_OPENCL_FA_DPAS2_WAVE` | compute units / 8 x 32 / NSG (20 on the B60) | Work-groups a wave, used to choose the split count. | 0045 | — |
| `GGML_OPENCL_FA_DPAS_F16KV_BLOCK` | 16384 | Keys a block when quantized K/V is converted to f16 for prompt attention. | 0037 | — |
| `GGML_OPENCL_FA_DPAS_OPTS` | none | Extra build options for the DPAS attention programs. Carries the `-D*_RED` red cases (`FA_CARRY_RED`, `FA_QG_RED`, `FA_VBLK_RED`, `FA2_RED`, `FA2_MCLS_RED`). | 0008 | Red case: `-DBK=24` (README, 0017). |
| `GGML_OPENCL_FA_GQA_KV_PER_SPLIT` | 512 | Keys a split in the GQA decode kernel. | 0018 | Search best 512 against 128; no effect from 131,072 keys on (`code`). |
| `GGML_OPENCL_FA_GQA_MIN_ROWS` | 4 | Least query rows routed to the GQA decode kernel. | 0016 | 1-3 rows faster on the split kernel: 6 rows 10.3 ms against 4.83 ms the other way (`measured-here`). |
| `GGML_OPENCL_FA_GQA_OPTS` | none | Extra build options for the GQA attention programs. Carries `GQA_K_STAGED`, `GQA_K_DIRECT`, `BK`, and the red case `-DBK=24`. | 0016 | Red case fails 41 of 53 cases at one geometry (`measured-here`, README 0017). |
| `GGML_OPENCL_FA_GQA_SIDES` | 2 (1, 2 or 4) | Sub-groups per 8 rows in the GQA kernel, for build and dispatch. | 0016 | — |
| `GGML_OPENCL_FA_INTEL_KV_PER_SPLIT` | 128 | Keys a split in the per-head Intel decode kernel. | 0007 | — |
| `GGML_OPENCL_FA_INTEL_OPTS` | none | Extra build options for Intel's FA programs (for tuning). | 0007 | — |
| `GGML_OPENCL_FA_SG8_NSG` | 16 (8, 16 or 32; used with the global-Q build) | Sub-groups a work-group in the sub-group-8 prompt attention. | 0039 | — |
| `GGML_OPENCL_FUSE_0051_PART` | 3 | Which GDN gating inputs fold into the kernel: 1 gate, 2 beta, 3 both. | 0051 | Both: +0.3 % dense agent, +0.9-1.2 % coder, neutral Flash-Next (`measured-here`). |
| `GGML_OPENCL_GATHER_CHUNKS` | 16 | Work-groups an expert's bytes are cut into in the gather. | 0022 | Did not move the gathers' time (`measured-here`). |
| `GGML_OPENCL_IQ_OPTS` | none | Extra defines for the IQ matvec builds (measurement). | 0022 | Carries `-DLB_CHAR4_ARITH`, `-DIQ4_KV_CHAR4` (README 0035), `-DLB_SIGN_MASK4` and `-DQ2_SPREAD_SHIFT` (the decodes before 0053), and the red cases `-DLB_SIGN_RED`, `-DQ2_SPREAD_RED` (README 0053). Since 0054 it reaches the low-bit prompt GEMM's build too. |
| `GGML_OPENCL_KQ_2D_OPTS` | none | Extra defines for the 2D K-quant GEMM programs. | 0014 | — |
| `GGML_OPENCL_KQ_2D_SHAPE` | `64,16,32` | `TM,WG,AT`: token tile, sub-groups, activation read height (8 or 32). | 0011 | Shape as measured in patch 0029 of marfrit-openvino (`measured-here`). |
| `GGML_OPENCL_KQ_2D_T<i>` (i = 0, 1, 2) | unset: the `KQ_2D_SHAPE` values | `TM,WG,AT,KSYNC` for Q4_K, Q5_K, Q6_K. Read by `tools/kq_tune`. A shape that does not build is ignored with a warning. | 0014 | — |
| `GGML_OPENCL_KQ_FEW_MIN` | 2 (sub-group 16), 4 (sub-group 8) | Least column count for the few-column K-quant kernel. | 0009 | — |
| `GGML_OPENCL_KQ_MM_ID2_SHAPE` | 32 x 8 on sub-group 16, 32 x 16 on sub-group 8 | `XT,XSG` of the ID2 MoE GEMM. | 0006 | 32x16, 32x32, 16x32 within 3 %, 64x16 1.7x slower (A770, `measured-here`). |
| `GGML_OPENCL_KQ_MM_ID_CACHE_MIN_TOK` | 64 | Least tokens for the MUL_MAT_ID GEMM on an expert cache's `.slots` / `.bank` tensors. | 0043 | 24 tokens ~1.9 s on the GEMM against ~0.65 s on the matvec (`measured-here`). |
| `GGML_OPENCL_KQ_MM_ID_MIN_TOK` | 16 | Least tokens for the MUL_MAT_ID GEMM on other MoE tensors. | 0001 | — |
| `GGML_OPENCL_KQ_MM_ID_TILE` | sub-group 8: 2,4,2,2; sub-group 16: 4,4,2,4 | `RG,TG,NSGM,NSGN` of the MUL_MAT_ID tile kernel. Refused if it does not fit the device. | 0001 (default from 0033) | Flash-Next needle prefill 494.7 -> 634.1 t/s with the bank stream (`measured-here`). |
| `GGML_OPENCL_KQ_MM_TILE` | 4,4,2,8 | `RG,TG,NSGM,NSGN` of the plain GEMM. | 0001 | Searched on A770 and B60 (`measured-here`). |
| `GGML_OPENCL_KQ_Q8_1_NDST` | 4 (sub-group 16), 2 (sub-group 8) | Destination rows a work-item in the K-quant q8_1 matvec. | 0001 | — |
| `GGML_OPENCL_PAIR_ORDER` | 1 | Order of a verify's expert pairs: 1 bank pairs first, 2 last, 0 token order. | 0022 | 30.7 t/s first, 29.8 token order, 29.3 last (Flash-Next MTP 2, B60, `measured-here`). |
| `LLAMA_EXPERT_CACHE_ADAPT_EVERY` | 6 | Decode tokens between adapt rounds; 0 never adapts. | 0021 | Hit rate 89.9-90.7 % across the constants (`measured-here`). |
| `LLAMA_EXPERT_CACHE_DECAY` | 0.7 | Decay of the routing counts at each adapt. | 0022 | Same. |
| `LLAMA_EXPERT_CACHE_CPU_MISS_THREADS` | 8 | Threads of the CPU miss path (the coordinator and its workers). | 0056 | 6 lower, 10 and 12 as 8; with 0063 and the cap at 64 on UD-Q3_K_XL 12 as 8, 16 slower (13.4 against 16.7 t/s: the pool competes with the decode's 8 threads) (`measured-here`). |
| `GGML_OPENCL_CPU_MISS_MAX` | 64 (16 before 0063) | Missed pairs a decode layer hands to the CPU; the card reads the rest over the link. From 40 on every miss of a verify window (10 experts x <= 4 tokens). | 0056, 0063 | Under 0056's rows 2 / 4 / 8 / 16 / 24 / 32: +4.2 / +8.5 / +11.3 / +13.4 / below 16; with 0063's rows 16 / 32 / 64 on IQ3_XXS 36.1 / 37.7 / 37.9 t/s, on UD-Q3_K_XL 12.5 / - / 16.7, 128 = 64 (`measured-here`). |
| `LLAMA_EXPERT_CACHE_MAX_SWAPS` | 96 | Largest swaps a plan makes; 0 is the discriminator in the swap-queue measurement. | 0022 | Same. |
| `LLAMA_EXPERT_CACHE_MIN_GAIN` | 1.5 | Least gain for a swap. | 0022 | Same. |
| `LLAMA_EXPERT_CACHE_MIN_USAGE` | 2.0 | Least usage for a swap. | 0022 | Same. |
| `LLAMA_EXPERT_CACHE_SPARES` | 4 a layer (minimum 1 when set) | Spare bank entries a layer for swaps. | 0042 | ~263 MB more pinned host memory for Flash-Next (`measured-here`). |

### Off switch

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_OPENCL_BANK_DIRECT` | on | `0`: the decode matvec gathers a missed expert into the card-side mirror instead of reading it from host memory. | 0022 | Flash-Next MTP 2, B60: 26.0 -> 28.2 t/s with it on (`measured-here`). |
| `GGML_OPENCL_F16_XMX` | on | `0`: F16 / BF16 weights times 16+ columns use the tiled GEMM, not the XMX one. | 0022 | Needle prefill 342.8 -> 383.4 t/s with it on (`measured-here`). |
| `GGML_OPENCL_Q8_XMX` | on | `0`: q8_0 weights times 16+ columns use upstream's q8_0 kernels, not the XMX GEMM. On the B60 only (the kernel needs sub-group 16). | 0058 | 1024² image step 2.14 -> 10.95 s with it off, 512² 0.476 -> 3.03 s (`measured-here`). |
| `GGML_OPENCL_F32_SKINNY` | on (Intel) | `0`: skinny F32 products use ggml's paths. | 0005 | Coder router, 512-token prefill: 24.3 -> 9.0 ms, A770 (`measured-here`). |
| `GGML_OPENCL_FA_DPAS` | on (Intel with DPAS) | `0`: prompt attention keeps `flash_attn_f32_f16.cl`. | 0008 | 512 rows x 4,096 keys: B60 8.2 ms, A770 22.4 ms (`measured-here`). |
| `GGML_OPENCL_FA_DPAS2` | on (Xe2 only) | `0`: prompt attention keeps 0008's kernel. | 0038 | — |
| `GGML_OPENCL_FA_DPAS2_MCLS` | on | `0`: no mask classes (skipping all-zero and all -inf blocks). | 0038 | — |
| `GGML_OPENCL_FA_DPAS_F16KV` | on | `0`: quantized K/V stay in the quantized DPAS kernels, not staged to f16. | 0037 | Unpacking costs 38 % more instructions a tile (527 of 1,893, B60 ISA, `code`). |
| `GGML_OPENCL_FA_DPAS_Q` | on | `0`: quantized prompts use `flash_attn_f32_q8_0.cl` or host dequantization. Also turns off F16KV. Read at load. | 0015 | — |
| `GGML_OPENCL_FA_GQA` | on (Xe2 only) | `0`: decode and verify rows use `flash_attn_f32_f16_q1_split_intel`. | 0016 | — |
| `GGML_OPENCL_FA_INTEL_SPLIT` | on (Intel) | `0`: decode attention keeps the basic q1 kernel. | 0007 | Coder 47.8 / 47.3 / 43.5 t/s with it (`measured-here`). |
| `GGML_OPENCL_FA_ROW_BLOCKS` | on (Intel, 9-64 rows) | `0`: one attention call, not blocks of 8 rows. | 0028 | Two lanes' verifies at 4k spent 4.3 ms an attention call without it (`measured-here`). |
| `GGML_OPENCL_FA_SG8_QG` | on (sub-group 8) | `0`: Q stays in local memory, 8 sub-groups. | 0039 | — |
| `GGML_OPENCL_FA_SG8_VBLK` | on (sub-group 8, f16 V) | `0`: the old V staging. | 0048 | V's transposing stores ~12 ms of 70 on the A770 (`measured-here`). |
| `GGML_OPENCL_FUSE_0049` | 7 | Mask: 1 rms_norm + scale, 2 ssm_conv + silu, 4 sigmoid + mul; 0 off. | 0049 | Flash-Next verify graph 12,346-12,407 -> 11,666-11,678 us; decode 32.0 -> 32.2 t/s (`measured-here`). |
| `GGML_OPENCL_FUSE_0050` | 3 | Mask: bit 1 GDN snapshots written into the cache, bit 2 the gating fold (0051). `0` keeps the copy. | 0050 | Decode +5.1-5.3 % dense agent, +1.8-2.0 % coder, +1.2-1.9 % Flash-Next (`measured-here`). |
| `GGML_OPENCL_FUSE_SCALE_ACT` | on | `0`: SCALE + SILU / SIGMOID not fused. | 0022 | Fewer launches, text identical, no decode change above the spread (`measured-here`). |
| `GGML_OPENCL_FUSE_TOPK_MOE` | on | `0`: the MoE routing chain not fused into one kernel. | 0022 | TOPK_MOE 416/416 in test-backend-ops (`measured-here`). |
| `GGML_OPENCL_GDN_INTEL` | on (Intel) | `0`: keeps `gated_delta_net.cl`. `Rtg,Rpp,NSG,TC` overrides the shape. | 0003 | 512 tokens, harness: B60 875 against 5,692, A770 451 against 12,042 (`measured-here`). |
| `GGML_OPENCL_KQ_2D` | on (DPAS + 2D block, not sub-group 8) | `0`: keeps `mul_mm_kq_f16.cl`. | 0011 | — |
| `GGML_OPENCL_KQ_2D_I8` | on | `0`: Q4_K 2D GEMM keeps the fp16 kernel. | 0013 | 1,030 / 931 against 973 / 884 (README table, columns as there); KL +0.0005 nats (`measured-here`). |
| `GGML_OPENCL_KQ_DEDUP` | on | `0`: every activation is converted again. | 0001 | `=0` alone fixed a stale-cache defect that 0027 corrected (`measured-here`). |
| `GGML_OPENCL_KQ_FEW` | on (DPAS) | `0`: keeps the q8_1 matvec and the GEMM for few columns. | 0009 | Coder 3-token step 104 against 96 t/s (`measured-here`). |
| `GGML_OPENCL_KQ_MM` | on (DPAS) | `0`: keeps ggml's GEMM. | 0001 | — |
| `GGML_OPENCL_KQ_MM_ID2` | on for sub-group 16, off for sub-group 8 | `1` / `0` forces the ID2 MoE GEMM on / off. | 0006 | Sub-group 8: tile kernel 1,740 against ID2 1,702 t/s (`measured-here`). |
| `GGML_OPENCL_KQ_MV` | on | `0`: K-quant short batches use ggml's float kernels. | 0001 | — |
| `GGML_OPENCL_KQ_Q8_1` | on (Intel with integer dot) | `0`: all K-quant x q8_1 kernels off; the experts of a K-quant MoE leave the device. | 0001 | — |
| `GGML_OPENCL_KQ_QUANT_SG` | on | `0`: q8_1 quantization with one work-item a block. | 0004 | 7.4 us on the B60, 10 us on the A770 for the old kernel (`measured-here`). |
| `GGML_OPENCL_KQ_ROUTE_CACHE` | on | `0`: routes every MUL_MAT_ID call. | 0006 | — |
| `GGML_OPENCL_STAGE_INPUTS` | on (needs USM host alloc) | `0`: input writes up to 4 MiB are blocking again, not staged through a 32 MiB ring. | 0047 | Verify input copies 926-949 -> 582-593 us; decode 31.6 -> 32.0 t/s (`measured-here`). |
| `GGML_SCHED_INPUT_SYNC_EACH` | off | `1`: the scheduler waits for the backend before each input copy, as upstream does. | 0047 | Same measurement as `STAGE_INPUTS`. |
| `LLAMA_EXPERT_CACHE_ASYNC_IDS` | on (needs the async get) | `0`: `step()` reads the expert ids with a wait. | 0022 | — |
| `LLAMA_EXPERT_CACHE_NB` | on (needs the adapt entry points) | `0`: swaps go back to the exchange on the compute queue. | 0042 | Verify input wait 2.6-6.0 -> 0.94-0.95 ms; decode 32.3 -> 33.1 t/s (`measured-here`). |
| `GGML_OPENCL_CPU_MISS_MT` | on (AVX2, FMA, F16C) | `0`: the CPU miss path on ggml-cpu's single-token dot products, as its prototype. | 0056 | ggml-cpu's Q2_0 dot product 661 against 72 us a matrix (`measured-here`). |

### Red case

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_OPENCL_BANK_STREAM_RED` | off | Each ring slot is filled from the next bank tensor. | 0032 | The run aborts at the next synchronisation (`measured-here`). |
| `GGML_OPENCL_F16_XMX_RED` | off | The XMX F16 GEMM runs with K shortened by 16, which also shifts its row stride: every row misread, a gross red (`code`). | 0022 | — |
| `GGML_OPENCL_Q8_XMX_RED` | off | The XMX q8_0 GEMM zeroes the last 16 weights of every row (those K terms left out). | 0058 | 1024² image gate cos worst 0.999926 -> 0.987219, PSNR 42.59 -> 19.56 dB; 52 of 151 Q8_0 op cases fail (`measured-here`). |
| `GGML_OPENCL_FUSE_0049_RED` | off | The folded rms_norm scale is dropped. | 0049 | — |
| `GGML_OPENCL_FUSE_0050_RED` | off | Every GDN snapshot slot is written over slot 0. | 0050 | Changed the Flash-Next texts (`measured-here`). |
| `GGML_OPENCL_FUSE_0051_RED` | off | The gating fold reads A in place of dt. | 0051 | Changed all three Flash-Next texts and lost the needle (`measured-here`). |
| `GGML_OPENCL_STAGE_RED` | off | The staging ring's bytes are not copied, so stale bytes reach the card. | 0047 | The capital answered "!!!!!!!!", the needle empty (`measured-here`). |
| `LLAMA_EXPERT_CACHE_MUTATE` | 0 | `1` corrupts one landed slot; `2` leaves out the last changed row of a table write. | 0021 (`2` from 0046) | `1`: the swap check reports 1 of 6 (`measured-here`). |
| `GGML_OPENCL_CPU_MISS_RED` | off | The CPU's rows left zero. | 0056 | KL 0.3700 -> 0.4364, top-1 84.4 -> 81.9 % (`measured-here`). |
| `GGML_OPENCL_CPU_MISS_SPLIT_RED` | off | The slow path reads an IQ4_NL bank's split scales and quants as GGUF blocks (the defect fixed in 0.8.0). | 0056 | IQ3_XXS: a fault at the first request (`measured-here`). |
| `MTMD_TTS_WAV_PAD` | off | Forces Qwen3-TTS's `GEN_WAV` graph and its input back to the padded 72-frame `wav_tfm_swa` window, the pre-patch behaviour. | 0062 | On the patch binary all 8 sentences are identical to the production padded arm (0.7285), where the default differs at <=191/32768 (0.8006) (`measured-here`). |

### Diagnostic

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_SCHED_TIMING` | off | Logs the host's time per graph in input copies and split computes, every 50 graphs. | 0047 | Used for the 0047 numbers (`measured-here`). |
| `LLAMA_DECODE_TIMING` | off | Logs where a decode call spends its time. | 0022 | — |
| `LLAMA_KV_RESERVE_LOG` | off (set at all = on) | Logs a prompt cut into smaller ubatches by the KV reserve. | 0052 | — |

## arcint

arcint's flags are on the command line (`arcint --help`, `llm.txt`). Since
1.2.0 it reads two environment switches, both for the TTS unit; the 38 it
read before 0.8.0 belonged to the retired OpenVINO engine
(`src/exec/backend_ov.cpp`) and are listed in this file at the tag
`openvino-history` (`git show openvino-history:docs/switches.md`).

| switch | class | default | what it does | evidence |
|---|---|---|---|---|
| `ARCINT_TTS_PROFILES` | operational | unset | The enrolled voices' profile directory when `--tts-profiles` is not given (the flag wins); `tools/tts_enrol.py` reads it too. | `code`: `src/config.cpp` |
| `ARCINT_TTS_GEN_FORCE_GPU` | red case | unset | `1` (exactly) builds the audio-generation mmproj on the card even under `--tts-gen-device cpu`, so the boot's placement check must refuse; any other value is ignored. | `measured-here`: red refuses naming the card backend, green boots with `CLIP using CPU backend` only (milestone §9.5, `tools/tts_placement_check.sh`) |

The voice-similarity tool reads `ARCINT_SIM_MODEL`, the speaker model's
safetensors path when `--model` is not given (`tools/tts_voice_sim.py`; a
measuring tool, not a service switch).
