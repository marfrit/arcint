# Switches

Draft for charter item A7 (`docs/milestone-1.0.0.md`): every switch that 1.0
keeps is documented, and diagnostic and red-case switches are named as such.

This is a proposal. The operator decides which switches stay stable within
1.x (charter C4). Until then nothing here is a promise.

A switch here is an environment variable read by arcint, by one of its tools,
or by the llama.cpp patch series in `contrib/llama.cpp/patches` (0001-0052).
Each switch is in exactly one class:

- **operational**: changes how a service runs; a unit file may set it.
- **tuning**: a shape or threshold whose default was measured. Override it to
  measure.
- **opt-in alternative**: a mechanism left off because it measured slower or
  neutral.
- **off switch**: turns a default-on mechanism off, to compare or to fall back.
- **red case**: breaks something on purpose, to prove a test can fail.
- **diagnostic**: logging, timing, profiling, debug dumps.

Columns: switch, default (what the code does when the variable is unset),
what it does, the patch that added it (or `arcint` with its source file), and
the evidence with its class (`measured-here` or `code`). "?" means not
determined. "—" means no number recorded. Numbers are as the patch README
states them. Unless said otherwise, the boolean switches read `atoi(value)`:
`0` is off, anything else is on.

Two kinds of name are not in a table:

- `HOME` and `XDG_CACHE_HOME` are read by arcint only to place the GGUF
  verdict cache (`src/exec/backend_ov.cpp`). They are not switches.
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
| `GGML_OPENCL_IQ_OPTS` | none | Extra defines for the IQ matvec builds (measurement). | 0022 | Carries `-DLB_CHAR4_ARITH`, `-DIQ4_KV_CHAR4` (README 0035), `-DLB_SIGN_MASK4` and `-DQ2_SPREAD_SHIFT` (the decodes before 0053), and the red cases `-DLB_SIGN_RED`, `-DQ2_SPREAD_RED` (README 0053). |
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
| `LLAMA_EXPERT_CACHE_MAX_SWAPS` | 96 | Largest swaps a plan makes; 0 is the discriminator in the swap-queue measurement. | 0022 | Same. |
| `LLAMA_EXPERT_CACHE_MIN_GAIN` | 1.5 | Least gain for a swap. | 0022 | Same. |
| `LLAMA_EXPERT_CACHE_MIN_USAGE` | 2.0 | Least usage for a swap. | 0022 | Same. |
| `LLAMA_EXPERT_CACHE_SPARES` | 4 a layer (minimum 1 when set) | Spare bank entries a layer for swaps. | 0042 | ~263 MB more pinned host memory for Flash-Next (`measured-here`). |

### Opt-in alternative

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_OPENCL_F16_KSPLIT4` | off | K-split F16 matvec with 4 rows a work-group, for K >= 8192 and 64+ rows. | 0022 | Measured slower, kept opt-in; no number in the README (`measured-here`). |
| `GGML_OPENCL_FUSE_HC_MIX` | off | The hyper-connection read as three kernels instead of five. | 0022 | Flash-Next MTP 2, B60: 30.8 t/s off, 30.4 on (README); the code comment says 31.2 and 30.8 (`measured-here`). |
| `GGML_OPENCL_GATHER_PAIRS_MULTI` | off | The per-pair gather grid for a few tokens too. | 0022 | Served, the shape did not move the gathers' time (`measured-here`). |
| `GGML_OPENCL_KQ_GROUPED` | off | A verify's 2-4 token pairs grouped by expert and decoded once, low-bit types. | 0022 | 30.5 -> 29.5 t/s (`measured-here`). |
| `GGML_OPENCL_KQ_IDG` | off | MUL_MAT_ID matvec grouped by expert for 2-15 tokens. | 0022 (fixed in 0030, 0031) | Coder verify 7.85 -> 9.18 s on the A770; Flash-Next 125 -> 131 ms a verify (`measured-here`). |
| `GGML_OPENCL_KQ_ONE_TERM` | off | One int8 activation term in the K-quant and IQ dot products, as llama.cpp and Strata do. | 0022 | Flash-Next 31.0 -> 31.6-31.8 t/s; coder 43.3 -> 40.7, draft acceptance 42.7 -> 38.2 % (`measured-here`). |
| `GGML_OPENCL_MOE_PREFETCH` | off | A projection's matvec copies the next projection's misses into a card-side mirror. | 0022 | 31.4 -> 31.6 t/s (`measured-here`). |

### Off switch

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_OPENCL_BANK_DIRECT` | on | `0`: the decode matvec gathers a missed expert into the card-side mirror instead of reading it from host memory. | 0022 | Flash-Next MTP 2, B60: 26.0 -> 28.2 t/s with it on (`measured-here`). |
| `GGML_OPENCL_F16_XMX` | on | `0`: F16 / BF16 weights times 16+ columns use the tiled GEMM, not the XMX one. | 0022 | Needle prefill 342.8 -> 383.4 t/s with it on (`measured-here`). |
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
| `GGML_OPENCL_FUSE_RMS_SCALE` | unset | Alias of `FUSE_0049`: `0` is mask 0, other values mask 7. Ignored when `FUSE_0049` is set. | 0049 | — |
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

### Red case

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_OPENCL_BANK_STREAM_RED` | off | Each ring slot is filled from the next bank tensor. | 0032 | The run aborts at the next synchronisation (`measured-here`). |
| `GGML_OPENCL_F16_XMX_RED` | off | The last 16 of K left out of the XMX F16 GEMM. | 0022 | — |
| `GGML_OPENCL_FUSE_0049_RED` | off | The folded rms_norm scale is dropped. | 0049 | — |
| `GGML_OPENCL_FUSE_0050_RED` | off | Every GDN snapshot slot is written over slot 0. | 0050 | Changed the Flash-Next texts (`measured-here`). |
| `GGML_OPENCL_FUSE_0051_RED` | off | The gating fold reads A in place of dt. | 0051 | Changed all three Flash-Next texts and lost the needle (`measured-here`). |
| `GGML_OPENCL_HC_MIX_RED` | off | The fused hyper-connection mix is scaled by 1.05. | 0022 | — |
| `GGML_OPENCL_STAGE_RED` | off | The staging ring's bytes are not copied, so stale bytes reach the card. | 0047 | The capital answered "!!!!!!!!", the needle empty (`measured-here`). |
| `LLAMA_EXPERT_CACHE_MUTATE` | 0 | `1` corrupts one landed slot; `2` leaves out the last changed row of a table write. | 0021 (`2` from 0046) | `1`: the swap check reports 1 of 6 (`measured-here`). |

### Diagnostic

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `GGML_OPENCL_HC_MIX_DEBUG` | off | Logs why the hyper-connection read was not fused (first 4 times, 1-4 tokens). | 0022 | — |
| `GGML_SCHED_TIMING` | off | Logs the host's time per graph in input copies and split computes, every 50 graphs. | 0047 | Used for the 0047 numbers (`measured-here`). |
| `LLAMA_DECODE_TIMING` | off | Logs where a decode call spends its time. | 0022 | — |
| `LLAMA_KV_RESERVE_LOG` | off (set at all = on) | Logs a prompt cut into smaller ubatches by the KV reserve. | 0052 | — |

## arcint

The OpenVINO engine reads most of these (`src/exec/backend_ov.cpp`).

### Operational

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `ARCINT_MOE_DEVICE_POOL_BYTES` | unset: the slot pool sits in host memory | Bytes of expert slots held in VRAM. Parsed strictly: a typo is refused. | arcint (`src/exec/backend_ov.cpp`) | In `docs/serving-config-flash-next.md` (`code`). |
| `ARCINT_TOKENIZERS_SO` | the path compiled in (`ARCINT_TOKENIZERS_SO_DEFAULT`) | Path of the OpenVINO tokenizers library. | arcint (`src/exec/backend_ov.cpp`) | — |

### Tuning

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `ARCINT_DYN_QUANT_GROUP` | unset: the runtime's own (128 for a hybrid linear-attention model) | Dynamic-quantization group size along K, a number or `max`; needs `--dyn-quant on`. | arcint (`src/exec/backend_ov.cpp`) | Added 0.4.2 as a measurement switch (`code`). |
| `ARCINT_MTP_DRAFTS` | 3 (clamped 1-8) | Drafts a chained MTP window proposes at most. | arcint (`src/exec/backend_ov.cpp`) | ? |
| `ARCINT_MTP_MIN_P` | 0.5 | Probability under which the MTP chain stops. | arcint (`src/exec/backend_ov.cpp`) | ? |
| `ARCINT_PAGED_KV` | unset: the `--paged-kv` flag | Overrides `--paged-kv` at load, for A/B runs (`KEY[:VALUE]`, from f16, u8, i8, u4, i4). | arcint (`src/exec/backend_ov.cpp`) | `DESIGN.md` names it an A/B override (`code`). |

### Opt-in alternative

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `ARCINT_PA_HOST_INPUTS` | off (set at all = on) | Paged-attention index inputs live in USM host memory. | arcint (`src/exec/backend_ov.cpp`) | Added in 0.2.10 as an experiment; no result found (?). |

### Off switch

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `ARCINT_DRAFT_POLICY` | on | `0`: every draft and proposal is verified, with no cost policy. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_DRAFT_ROPE_F16` | RoPE of the drafter graphs kept in f32 | Set at all: RoPE runs in the plugin's default f16 again (the A/B lever). | arcint (`src/exec/backend_stub.cpp`) | The f16 overflow hypothesis was confirmed with `ARCINT_DRAFT_F32` (`measured-here`, CHANGELOG). |
| `ARCINT_NGRAM_READER` | on | `0`: staged n-gram rows use the synchronous `pread`. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_PREFILL_CHUNK_CAP` | on | `off`: removes the prefill chunk cap and the belt. | arcint (`src/exec/backend_ov.cpp`) | Used for a 118,454-token full-depth rung (`measured-here`, `src/exec/fit.h`). |

### Red case

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `ARCINT_FIT_SLOT_BYTES` | unset | Forces the expert slot pool to N bytes and skips the fit probe. `0` on an `--offload-ratio` run reproduces the pre-M7 arithmetic. | arcint (`src/exec/backend_ov.cpp`) | A green real figure and an OOM at 0 on one binary (`code`). |

### Diagnostic

| switch | default | what it does | patch | evidence |
|---|---|---|---|---|
| `ARCINT_BENCH_FORWARD` | off | Runs `bench_forward()` after warm-up. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_BLOCK_AB_HASH_ROWS` | off | `native_moe_block_ab` prints an FNV-1a hash of the first k output rows. | arcint (`tools/native_moe_block_ab.cpp`) | — |
| `ARCINT_BLOCK_AB_REPEAT` | off | `native_moe_block_ab` runs the GPU request N more times and prints mean, min, median. | arcint (`tools/native_moe_block_ab.cpp`) | — |
| `ARCINT_DFLASH_DUMP` | off | Dumps the DFlash lattice and per-cycle lines. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_DRAFT_F32` | off (set at all = on) | Compiles both drafter graphs in f32. A discriminator for the f16 overflow. | arcint (`src/exec/backend_stub.cpp`) | Confirmed the hypothesis (`measured-here`, CHANGELOG). |
| `ARCINT_FORWARD_SPLIT` | off | `1`: wall clock of the graph, logits and hidden-state readbacks over 64 forwards. `2`: also repeats the forward. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_LOGITS_DUMP` | off | Appends every paged forward's logits to a file (the served half of the KL gate). Reads logits back for every forward. | arcint (`src/exec/backend_ov.cpp`) | In `DESIGN.md` (`code`). |
| `ARCINT_PROFILE` | off (set at all = on) | Turns on plugin profiling and prints step profiles after warm-up. | arcint (`src/exec/backend_ov.cpp`) | In `DESIGN.md`, `docs/prefill-baseline.md` (`code`). |
| `ARCINT_PROFILE_CYCLE` | off (set at all = on) | One line per drafting cycle on the paged path, naming each segment. | arcint (`src/exec/backend_stub.cpp`) | In `DESIGN.md` (`code`). |
| `ARCINT_PROFILE_MWALL` | built-in token counts | Comma list of token counts for the M-wall profile. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_PROFILE_MWALL_LA` | off (set at all = on) | Passes the served verify's checkpoint rows in the M-wall profile. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_PROFILE_NODES` | off (set at all = on) | Lists every node of the decode step by name. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_PROFILE_PAST` | 0 | Profiles the chunk after prefilling to depth n. | arcint (`src/exec/backend_ov.cpp`) | In `docs/prefill-baseline.md` (`code`). |
| `ARCINT_PROFILE_RUNTIME_MODEL` | off | Path to dump the compiled graph (node names, op types, plugin layer names). | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_PROFILE_SWEEP` | off | Comma list that bisects the token count where kernel selection changes. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_PROFILE_TOKENS` | fixed ids | `random`: pseudo-random token ids for profiling. | arcint (`src/exec/backend_ov.cpp`) | — |
| `ARCINT_TRACE_TOKENS` | off (set at all = on) | Traces each committed token. | arcint (`src/exec/backend_ov.cpp`) | — |
| `BIGALLOC_MIN` | 8 MiB | Smallest allocation `bigalloc.so` records. | arcint (`tools/bigalloc.c`) | — |
| `BIGALLOC_ONLY` | all programs | Traces only the program with this short name. | arcint (`tools/bigalloc.c`) | — |
| `BIGALLOC_OUT` | `bigalloc.<pid>.txt` | Output file of the allocation trace. | arcint (`tools/bigalloc.c`) | — |
| `BIGALLOC_PERIOD` | 5 s | Seconds between rewrites of the trace. | arcint (`tools/bigalloc.c`) | — |
| `BIGALLOC_STEP` | 256 MiB | Logs each new peak of the live total, in steps of this size. | arcint (`tools/bigalloc.c`) | — |
| `CLDUMP_BINARIES` | off | `cldump.so` also saves program binaries. | arcint (`tools/cldump.c`) | — |
| `CLDUMP_DIR` | `/tmp/cldump` | Output directory of `cldump.so`. | arcint (`tools/cldump.c`) | — |
| `CLDUMP_SERIALIZE` | off | `cldump.so` calls `clFinish()` after each kernel enqueue. | arcint (`tools/cldump.c`) | Used 2026-09-20 to cut a B60 layer (`measured-here`, source comment). |
| `IGADIS_DUMP` | off | `igadis` writes the disassembly beside the input as `<file>.asm`. | arcint (`tools/igadis.cpp`) | — |
