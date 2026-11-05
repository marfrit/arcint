# Part 0 — the IPEX-LLM llama.cpp binaries, read (2026-11-05)

Material: `~/research/ipex-llm/x/` — the unpacked wheel `bigdl_core_cpp-2.7.0b20260110`
(PyPI, Apache-2.0 per its METADATA), `bigdl/cpp/libs/llama_cpp/` and a second copy under
`bigdl/cpp/libs/ollama/`. Everything here is data; nothing was executed. Tools: `nm -C -S`,
`objdump -d -M intel`, `readelf`, `strings`, a small Python SPIR-V reader.
Evidence class: `code` = read in a binary's symbols or disassembly; `code, symbol only` = only a
name supports it; `paper` = a report or a commit message (not used in this file except where a
build string is quoted). Nothing here is `measured-here`.

Target binaries: `libggml-sycl.so` (ELF x86-64, **not stripped**, `.text` 2.07 MB), `libllama.so`,
`libggml-base.so`, `llama-bench`. `llama-bench`'s build string names commit `3d3c767` (`code`:
the string is present verbatim; not orderable against our pin offline).

The second tree `x/bigdl/cpp/libs/ollama/` carries the same `libggml-sycl.so` family (an Ollama
build); it was not read further. `x/xe/` is the PyTorch path's kernels
(`xe_linear.cpython-310-x86_64-linux-gnu.so`, `xe_addons…so`), not the llama.cpp path.

---

## 1. What the file is made of

| part | size | content (`code`) |
|---|---|---|
| `.text` | 2.07 MB | host side: `ggml_sycl_*` ops, dispatch, oneDNN/oneMKL calls |
| `__CLANG_OFFLOAD_BUNDLE__sycl-spir64` | 4.05 MB at 0x308590 | **48 concatenated SPIR-V modules**, **524 `OpEntryPoint`s**, 4,364 `OpName` strings |
| `.esimd*` names | 7,392 occurrences in strings | ESIMD section names inside 11 of the 48 modules (0, 3, 4, 5, 8, 9, 12, 13, 14, 15, 42); 1,729 of them name an ESIMD section. There is no ELF `.esimd` section — the names live in the device images |

The SPIR-V reader: magic `0x07230203`, version 1.4, one module per aligned magic; `OpEntryPoint`
(15) and `OpName` (5) give the kernel names. Modules are grouped by translation unit, not one
kernel a module: module 14 is the 80-instance K-quant batch GEMM, module 15 the 30-instance q4_0
batch GEMM, module 24 the 20-instance MMQ, etc.

## 2. Kernels (`code`; the SPIR-V entry points and the host launch symbols agree)

Name, template parameters as mangled (they are the block/tile geometry), ESIMD yes/no, XMX yes/no.

| kernel | module | geometry read from the symbol | ESIMD | XMX | what its arguments say |
|---|---|---|---|---|---|
| `sdp_causal_xmx_kernel<128,64,8,16,16>` (+ 64/32, 64/64, 128/32) | 3 | Q block 64/128 × K block 32/64, sg 16, rows 8/16 | no | **yes** | 6 pointers (Q, K, V, mask, ?, ?), 19 strides, 2 ints, float `scale`; fp16 in, f32 out. A **causal prefill flash attention on DPAS** |
| `sdp_fp8_causal_xmx_kernel<…>` (+ the same four shapes) | 4 | same tiles | no | **yes** | K/V as `void*` (fp8 bytes), plus `unsigned long` mask stride and an int for KV type; causal, fp8 KV in |
| `reshape_fp8_{key,value}_cache_kernel<16,8|16,64|128>` | 4 | head dims 64/128, 16 keys a block | no | no | fp8 cache bytes → half out; **the fp8 cache is converted to f16 per attention call** |
| `sdp_fp16_kernel / sdp_fp16_casual / sdp_fp16_non_casual <32,64|80|96|128|192>` | 5 | 32-row query block, head dim 64/80/96/128/192 | no | no | decode-shaped SDP (1-32 rows), causal/non-causal variants |
| `sdp_fp8_kernel / sdp_fp8_casual <32,…>` | 5 | same | no | no | fp8 K/V decode SDP |
| `sdp_q80_kernel<32,128>` | 5 | q8_0 K/V decode SDP | no | no | `half*` Q, `signed char*` K/V, scales |
| `quantize_key_kernel<half|float, 64|80|96|128|192>` | 12 | one work-item a key | no | no | f16/f32 key → `unsigned char` (1 byte a value, **fp8**) |
| `quantize_key_value_kernel<half|float, 64|80|96|128>` | 12 | — | no | no | K **and** V → fp8 in one pass |
| `quantize_key_value_kernel_q8_0<half|float,32>` | 12 | 32-value blocks | no | no | the q8_0 alternative (`signed char` + scales) |
| `dequantize_key_kernel<64|80|96|128|192>` | 12 | — | no | no | fp8 cache → f16 |
| `qlinear_xpu_kernel_q4_0_2x16_qkv<64,32,2>` and `…_qkv_neox<32|64,16|32,2>` | 6, 7 | 2×16 / 4×16 tiles, 2 a block | **yes** | no | f32 activation, 4 weight pointers (q4_0), 3 scale/bias pointers, an int32 pointer (dims?), 3 half outputs Q/K/V, 8 strides, 3 floats (norm eps? rope theta, freq scale), 7 ints |
| `qlinear_xpu_kernel_q4_0_2x16_wqkv_neox<…>` | 6, 7 | same | **yes** | no | one f32 activation, one q4_0 weight, one bias, an int32 ptr, 3 half outputs — **a combined QKV weight** |
| `mlp_forward_q4_0_kernel<64,32,2>` / `<32,16,2|8>` | 10, 11 | — | no | no | f32 in, f32 out, q4_0 gate/up weights, 3 scale/bias, dims — gate+up+SiLU |
| `mlp_forward_q4_0_kernel_upgate<…>` | 10, 11 | — | no | no | one weight pointer at the front (gate and up interleaved/one tensor) — the "upgate" fused form |
| `mlp_forward_q4_k_kernel<32,16,2>` | 9 | — | no | no | the same for Q4_K |
| `linear_forward_kernel<float,2,4,16,…>` | 0 | rows × cols tiles, 6 instantiations | **yes** | no | q4_0 linear, f32 out (the ESIMD GEMM primitive the fusions use) |
| `vec_q4_0_batch_kernel<float,8,2,16|32,{1..8},64,…>` (30) and `vec_{q2,q3,q4,q5,q6}_K_batch_kernel` (54) | 14, 15 | 8 rows × 2 columns a tile | no | no | quantized weight × **float** activation, batch-shaped; IPEX's own quantized batch GEMM (`bigdl-core-xe/llama_backend/batch/kquants.cpp`, `int4.cpp`) |
| `ggml_mul_mat_qX_q8_1_sycl` (20 instantiations) | 24 | MMQ tiles | no | no | the upstream MMQ form |
| `mul_mat_vec_{qX,iqX}_q8_1_sycl` (22), `reorder_mul_mat_vec_*` | 23 | one row a work-item, `DMMV_X` columns | no | no | upstream MMVQ, plus reordered-weight variants |
| `dequantize_mul_mat_vec_q4_0_kernel<32,16,2|8>` / `<64,32,8>` | host `T` | 3 geometries | no | no | **card-shaped DMMV for q4_0** — the 64/32/8 one is a B60-shaped instance beside the 32/16 ones |
| `reorder_qw_q4_0 / q4_k / q6_k` | 28 | one row a work-item | no | no | a **weight repack at load** into the ESIMD/DMMV layout |
| `ggml_{q4_0,q4_K,q6_K,…}_format_convert_to_xpu(_ipex_llm_blksize)` | host `T` | — | no | no | the host-side entry of the same repack; `q4_0` has a blksize variant |
| `dg2_{arc,bmg}_ll256_allreduce` | 38, 39 | — | no | no | multi-GPU allreduce over a peer buffer |

ESIMD, therefore, is in exactly the qkv/linear/mlp fusion family and the q4_0 reorder path, not in
the attention or the batch GEMM (`code`).

## 3. Host ops, and which the pin does not have

`nm -DC` gives 1,946 symbols, of which the `ggml_sycl_op_*`/`ggml_sycl_*` host functions are the
dispatch. Read from the symbol list (`code`):

- upstream's set is present (`ggml_sycl_op_mul_mat`, `_q`, `_sycl`, `_vec_q`, `op_get_rows`…),
  and the pin's newer ops (`fused_glu`, `rms_norm_fused`, `arange`, `dsv4_*`, `lightning_indexer`)
  are **absent**, so this tree is not the pin's tree (`code`).
- IPEX-only host ops: `ggml_sycl_op_sdp_xmx_casual`, `_fp8_sdp_xmx_casual`, `_fp16_sdp`,
  `_fp8_sdp`, `_q80_sdp`, `_mlp_fusion`, `_mlp_fusion_upgate`, `_moe_fusion`, `_qkv_fusion`,
  `_wqkv_fusion`, `_norm_rope`, `_quantize_kv`, `_dequantize_kv`, `_quantize_q8_0_kv`,
  `_dequantize_q8_0_kv`, `_l2_norm`, `_gated_linear_attn`, `_add_full_params`, `_mul_full_params`,
  `_norm_full_params`, `_repeat_full_params`, `_rope_full_params`, `_silu_full_params`,
  `_get_rows_full_params`, plus `opt_for_reorder`, `is_UHD_or_IRIS`, `gpu_has_xmx`/
  `device_has_xmx`/`check_xmx`, `get_gpu_type`, `dg2_init`/`dg2_arc_init`/`dg2_bmg_init`, a
  **hybrid** buffer type and a **split** buffer type (`ggml_backend_sycl_hybrid_buffer_*`,
  `…split_buffer_*`).
- the pin's own `fattn-onednn.cpp` (`/home/…/llama-served/ggml/src/ggml-sycl/fattn-onednn.cpp:1`)
  is a oneDNN FA for F16 K/V with a 32-row minimum, and its `fusion.cpp` fuses RMS_NORM+MUL(+ADD),
  ADD+ADD, MUL_MAT+MUL_MAT(+GLU) and unary chains. IPEX's fusion family is a different one: a
  **custom `ggml_bigdl_*` op family in ggml core** (below), plus own kernels.

## 4. How the graph is rewritten (fusions)

`libggml-base.so` adds op constructors (`code`, exported `T`): `ggml_bigdl_qkv_fusion`,
`ggml_bigdl_wqkv_fusion`, `ggml_bigdl_mlp_fusion`, `ggml_bigdl_mlp_fusion_upgate`,
`ggml_bigdl_moe`, `ggml_bigdl_moe_mlp_fusion`, `ggml_bigdl_fp16_sdp`, `ggml_bigdl_quantize_kv`,
`ggml_bigdl_dequantize_kv`. Their constructors create one tensor with **op id 85** (0x55 in
`ggml_bigdl_mlp_fusion`'s constructor, `code`) and an **output type 30 = `GGML_TYPE_BF16`**
(`code`: the constructor sets `[tensor+0x4]` to 0x1e; compare the pin's enum,
`/home/…/llama-served/ggml/include/ggml.h:420`). The kernel writes `half`; the op's type is BF16,
so the value class is rounding-only at f16 width.

`libllama.so` adds the graph-side match (`code`): `llm_graph_context::build_bigdl_qkv_fusion
(llm_graph_input_attn_kv_unified*, ggml_cgraph*, 10 tensor pointers, int, 3 floats, 3 uints,
float, int)`; it takes the KV input, gets `k`, `v` and the head from the KV-cache state
(`call … llama_kv_cache_unified_state::get_k/get_v/get_head`, `code`), then constructs the fused
tensor. The 10 tensor pointers correspond to a q, k, v (or a combined wqkv) weight, their
scale/bias sets, the norm weights and the rope inputs; the three floats are the norm eps, the
rope parameters. So the matched pattern is: **RMS-norm(Q/K) + RoPE + the three QKV linear
projections on q4_0 weights, into one op** — with either three weights (`qkv_fusion`) or one
`wqkv` weight (`wqkv_fusion`), and a `neox`/`norm` rope variant. `mlp_fusion` is the
gate/up/SiLU/down chain, `mlp_fusion_upgate` the one-weight form, `moe_mlp_fusion` the MoE
version. `GGML_SYCL_DISABLE_OPT` gates the optimisation (`code`: read in `ggml_check_sycl`, default
0) and `GGML_SYCL_DISABLE_GRAPH` disables SYCL graph capture (`code`).

`ggml_bigdl_fp16_sdp` + `quantize_kv` + `dequantize_kv` are the attention-side graph ops. The
quantize/dequantize pair is read in the kernels: the cache is **fp8 (1 byte a value)**, and the
attention consumes it after a `reshape_fp8_*` to f16, or directly in `sdp_fp8_*`. A q8_0 form
exists beside it.

## 5. How prefill's matmul is dispatched (`code`)

Two dispatchers:

1. `ggml_sycl_mul_mat` (`0x9ed20`) is the choice: for a quantized src0 it either takes
   `ggml_sycl_op_dequantize_mul_mat_vec` (DMMV, the 1-row shape), `ggml_sycl_op_mul_mat_vec_q`
   (MMVQ, ≤ a few columns), or falls through to `ggml_sycl_op_mul_mat_sycl`; `opt_for_reorder`
   runs first for quantized types. The **general** path, i.e. the one a 512-token prompt takes,
   is `ggml_sycl_op_mul_mat_sycl` (`0xd5cb0`, 13 KB).
2. `ggml_sycl_op_mul_mat_sycl` (`code`, disassembly and its call list):
   - `ggml_is_quantized(src0)` → `ggml_get_to_fp16_sycl(src0->type)` (or `_fp32`), which returns a
     dequantizer; the whole src0 is **dequantized into a pooled scratch buffer**
     (`ggml_backend_sycl_context::pool`, `ggml_sycl_pool_alloc`), then a matmul is issued on the
     f16 operands;
   - the matmul is **oneDNN** by default: `dnnl::matmul::primitive_desc` ctor, `dnnl::matmul`,
     `dnnl::primitive::execute` with `dnnl::memory` wrappers, `dnnl_engine/stream/memory`
     (3 execute sites in the function); `GGML_SYCL_DISABLE_DNN` (`sscanf` int, default 0,
     `ggml_check_sycl` at `0x87848`) disables it;
   - `IPEX_LLM_SYCL_USE_MKL` (read at `0xd5e17`, exact `"1"`) swaps in
     `oneapi::mkl::blas::row_major::gemm` (f16) / `column_major::gemm` (f32) and
     `oneapi::math::blas::gemm_batch`; the MKL is loaded from
     `/opt/intel/oneapi/mkl/2025.0/lib` (`code`: the path string is in the binary);
   - a `GGML_TYPE_BF16` branch exists beside it (type 30 checked at `0xd5d03`) for a BF16 src0.

So on the measured Llama 2 7B Q4_0 prefill, **the matmul is: dequantize the q4_0 weights to f16
once per call, then a vendor f16 GEMM on XMX** — not the fork's own `vec_q4_0_batch_kernel`, and
not MMQ. The `batch_forward_*`/`vec_*_batch` symbols exist but no direct call to them was found
from either dispatcher (`code`; their trigger — likely `GGML_SYCL_DISABLE_DNN=1` or a type-specific
branch — was not read, so this stays a negative that was not fully tried).

The 4,000-odd lines of `ggml_sycl_op_mul_mat_q` (`0x15c300`) are the quantized MMQ dispatcher by
symbol and template instantiation; the default choice above does not route through it.

## 6. Device-specific paths (`code`, disassembly)

- `get_gpu_type(queue&)` (`0x27b7b0`) reads the SYCL device **name string**, builds a
  `std::string` from it, and runs a `std::regex_search` (the regex table is in `.rodata`); it is a
  name classifier, not a feature probe.
- `gpu_has_xmx(device&)` (`0x106310`) is one call: `device.has(aspect 0x3a)` — a SYCL aspect
  query; `device_has_xmx(queue&)` wraps it, `check_xmx` caches it.
- `dg2_init(queue&, int)` (`0x1386b0`): `device_id = device.get_info<ext::intel::info::device::
  device_id>()`; if `(device_id & 0xffffff00) == 0x5600` → `dg2_arc_init`; else
  `device_id − 0xe20b <= 6` (0xe20b…0xe211) → `dg2_bmg_init`. Arc and Battlemage take separate
  init; anything else only gets a `pattern_counter` increment.
- `dg2_arc_init` (`0x138b40`) and `dg2_bmg_init` (`0x139760`) allocate **one 32 MiB (0x2000000)
  buffer per device, 4 KiB-aligned**, device-side when `is_p2p` is set and host-side otherwise,
  and store it as `host_bufs[dev]` and `peer_bufs[dev]`: a peer/allreduce staging buffer for the
  `ll256_allreduce` family. They set no tile size, no subgroup width, no environment flag. The
  name "dg2_*_init" overstates what it does: it is P2P plumbing, not a per-architecture tuning
  table.
- `is_UHD_or_IRIS` is called in the matmul dispatcher to keep integrated GPUs off a path (`code,
  symbol only` for its internal rule).

## 7. Environment switches (`code`: string + the read site where disassembled)

| variable | read where | default / meaning |
|---|---|---|
| `GGML_SYCL_DISABLE_DNN` | `ggml_check_sycl` `0x87857`, `sscanf` int | **0 = oneDNN used**; non-zero disables the oneDNN matmul |
| `IPEX_LLM_SYCL_USE_MKL` | `op_mul_mat_sycl` `0xd5e17`, exact `"1"` | unset/0 = oneDNN; 1 = oneMKL gemm |
| `GGML_SYCL_DISABLE_OPT` | `ggml_check_sycl`, int | 0 = optimisations on (fusions, reorder) |
| `GGML_SYCL_DISABLE_GRAPH` | `ggml_check_sycl`, int | 0 = SYCL graph capture allowed; also disabled automatically for multiple devices / unsupported nodes (log strings) |
| `GGML_SYCL_PRIORITIZE_DMMV` | `ggml_sycl_mul_mat` `0x9ee8e` | 0 |
| `GGML_SYCL_FORCE_MMQ`, `GGML_SYCL_DMMV_X`, `GGML_SYCL_F16`, `GGML_SYCL_NO_PINNED`, `GGML_SYCL_DEBUG`, `GGML_SYCL_MAX_DEVICES` | strings present | upstream names; not all read sites followed |
| `IPEX_LLM_QUANTIZE_KV_CACHE` | string in `libllama.so` | turns the fp8/q8_0 KV cache on (the quantize/dequantize ops' gate; `code, symbol only` for the exact parse) |
| `IPEX_LLM_SCHED_MAX_COPIES` | string | scheduler copy cap; purpose not read |

Log/error strings that name a mechanism (`code`: strings): `"SYCL Optimization Feature:"`,
`"%s: disbale async uploads for ipex-llm sycl backend"`, `"%s: retrying batch size %d after cache
optimization"`, `"%s: disabling SYCL graphs due to multiple devices"`, `"%s: disabling SYCL graphs
due to unsupported node type %s"`, `"GGML_SYCL_DISABLE_DNN: %d"`, and the two `CHECK_TRY_ERROR`
strings that spell the gemm calls (`dpct::gemm_batch(...)`, `oneapi::mkl::blas::row_major::gemm
(...)`, `column_major::gemm(...)`). The "retrying batch size after cache optimization" is a
scheduler retry, not a kernel.

## 8. How KV is stored (`code`)

- fp8, one byte a value: `quantize_key_kernel` produces `unsigned char` (head dims 64/80/96/128/
  192, one work-item a key); `quantize_key_value_kernel` quantizes K and V together; the `sdp_fp8*`
  kernels read `unsigned char*` K/V; `reshape_fp8_{key,value}_cache_kernel` writes f16 tiles out of
  the fp8 cache for the f16 SDP path. This is **the KV held as fp8** the brief suspected, and it is
  opt-in (env).
- q8_0 beside it (`quantize_key_value_kernel_q8_0`, `sdp_q80_kernel`, `dequantize_kv_q8_0_kernel`).
- The attention is preceded by an explicit reshape/quantize op in the graph
  (`ggml_bigdl_quantize_kv`, `ggml_bigdl_dequantize_kv`, `ggml_bigdl_fp16_sdp`), not folded into the
  SDP kernel — except the fp8-XMX kernels, which read the bytes directly.

## 9. What a name alone supports, and what disassembly supports

- `code` (disassembly): the dequant-to-f16 + oneDNN/MKL prefill matmul; the BF16 op output; the
  48-module SPIR-V and the 524 entry points; the fp8 cache and its reshape kernels; the dg2 init
  allocations; `GGML_SYCL_DISABLE_DNN`/`IPEX_LLM_SYCL_USE_MKL` parse and defaults; the qkv fusion
  graph function's KV-state calls.
- `code, symbol only`: the exact template geometry of each kernel beyond the mangled
  parameters (tile sizes are read, their meaning is inferred); `batch_forward_*`/`vec_*_batch`
  whose call site is not found; `is_UHD_or_IRIS`' rule; `IPEX_LLM_QUANTIZE_KV_CACHE`'s parse;
  `IPEX_LLM_SCHED_MAX_COPIES`; `get_gpu_type`'s regex words (the table was not decoded).
- Not read at all: the internal bodies of the kernels (which would need the SPIR-V/ESIMD bodies,
  not names); the `ll256_allreduce` algorithm; the Ollama tree.

## 10. First conclusions (for pass 2)

1. **Prefill's matmul is a vendor f16 GEMM on dequantized weights.** For a dense Q4_0 model every
   weight matrix that a prompt chunk touches is dequantized to f16 into device scratch and fed to
   oneDNN (or oneMKL) — the XMX f16 path, with a library's tiling. The fork also carries its own
   quantized batch kernels, but they are not the default path. Our series' 0096 does the same
   *at kernel level* (q4_0 → f16 tiles → Intel XMX), so the gap here is the library's GEMM, not
   the idea.
2. **Graph-level QKV/MLP fusions in `libllama`. ** One op per `[q,k,v] + norm + rope` and per
   `gate/up/SiLU/down`, with ESIMD q4_0 bodies and a combined-wqkv form. This is the largest
   structural difference the binary shows; its value class is rounding-only (BF16/f16 op output).
3. **An fp8 KV cache** with reshape/dequant on the way into attention (answer-changing by
   design, listed not audited), and a q8_0 alternative (rounding-only).
4. **Weight repack at load** (`reorder_qw_*`, `format_convert_to_xpu`) into the layout the ESIMD
   and DMMV kernels read, with card-shaped DMMV instances (`<64,32,8>` beside `<32,16,2|8>`).
5. The attention side has an XMX causal SDP at 32/64/128-row blocks and a decode SDP family at
   head dims 64–192, but no evidence of a oneDNN FA (the pin has one; this fork's is its own).

What a card measurement must separate: the fused-op share (launch + intermediate traffic) from
the vendor-GEMM share (oneDNN's tiling) — a `GGML_SYCL_DISABLE_OPT=1` / `GGML_SYCL_DISABLE_DNN=1`
run of their `llama-bench` on the same card is what tells them apart. The commands are in the
handover (README).
