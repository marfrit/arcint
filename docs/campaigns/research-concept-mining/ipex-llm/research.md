# Pass 3 — the top 10's mechanism, read (2026-11-05)

Where: `binary.md`'s addresses/symbols for the llama.cpp path; the Python tree at
`~/research/ipex-llm/ipex-llm-ref` head `de6bce27` (`python/llm/src/ipex_llm/...`) for the
PyTorch path. Line numbers are that tree's. `code` = read; `code, symbol only` = a name;
`paper` = a commit message. Where a commit message and the code disagree, the code is written
below and the message named.

## 1. The prefill matmul (score 9)

- **llama.cpp path** (`code`): `ggml_sycl_mul_mat` (`0x9ed20`) chooses DMMV / MMVQ / the general
  `ggml_sycl_op_mul_mat_sycl` (`0xd5cb0`). That function calls `ggml_get_to_fp16_sycl(src0->type)`
  and dequantizes src0 into `ggml_backend_sycl_context::pool` scratch (two pool calls read), then
  constructs a oneDNN matmul: `dnnl::memory::desc` x3, `dnnl::matmul::primitive_desc`,
  `dnnl::primitive::execute` (3 sites), with `ggml_backend_sycl_context::engine_dnnl`/
  `stream_dnnl`. `IPEX_LLM_SYCL_USE_MKL` (`0xd5e17`) substitutes `oneapi::mkl::blas::row_major::
  gemm` (f16) and `column_major::gemm` (f32); `GGML_SYCL_DISABLE_DNN` (`0x87857`, default 0) turns
  oneDNN off. A BF16 src0 takes a separate branch (`0xd5d03`, type 30).
- **PyTorch path** (`code`): the prefill GEMM goes to `xe_linear.forward_new(x_2d, w, qtype,
  out_len)` after a float→half cast (`low_bit_linear.py:701-710`); the batch kernel is only for
  small row counts (below). So the two engines' prompt GEMMs differ: oneDNN in llama.cpp, the
  ESIMD qlinear kernels in PyTorch. The `llama-bench` rows are the llama.cpp path.
- **Unread**: the oneDNN primitive's data type (f16 vs bf16) and its blocking; `ONEDNN_VERBOSE`
  on the card would name it. The exact reason for the 15 % is not in the binary.

## 2. The QKV fusion (score 8)

- **llama.cpp path** (`code`): `build_bigdl_qkv_fusion` (`libllama.so` `0xaf380`) builds one
  tensor from 10 inputs; it calls `llama_kv_cache_unified_state::get_k/get_v/get_head` (`0xaf472`,
  `0xaf486`, `0xaf491`). The op's constructor sets op id 85 and type 0x1e = **BF16**
  (`libggml-base.so` `0x232c0`). Kernels: `qlinear_xpu_kernel_q4_0_2x16_qkv{,_neox}` and
  `…_wqkv_neox`, module 6/7 SPIR-V, plain SYCL (no genx intrinsics), no DPAS; the symbol's last ints are the norm eps and
  rope parameters. Geometry 32/64 × 16/32 × 2 — small-row.
- **PyTorch path** (`code`): no fused projection+norm+rope kernel at HEAD; the weight merge is
  `merge_quantized_qkv(q_proj, k_proj, v_proj, module)` (`models/common.py:379-412`): it moves the
  three quantized weights to CPU, `torch.cat`s them into one `FP4Params` (same `qtype`), a bias
  concat, and one `qkv_proj`. So the combined-weight idea is real, the fused kernel is llama.cpp's.
- **Disagreement resolved**: commit `a00efa0564` says "mlp & qkv fusion for FP16 Llama-7B"; the
  code's `mlp_fusion_check` requires one row (`utils.py:291`) — the commit is a decode fusion.

## 3. The MLP fusion (score 8)

- **llama.cpp**: `ggml_bigdl_mlp_fusion` (op id 85 family), `mlp_forward_q4_0_kernel` and
  `…_upgate` (module 10/11, plain SYCL) and `mlp_forward_q4_k_kernel` (module 9, genx
  intrinsics).
- **PyTorch** (`code`): `fuse_mlp_base(module, act, x)` (`models/common.py:147-160`) calls
  `xe_linear.mlp_forward_xpu(x_2d, gate, up, rows, cols, out_len, act, qtype)` after
  `mlp_fusion_check(x, qtype, training)`; that check returns false unless
  `x.numel() // x.size(-1) == 1` (`utils.py:289-302`), qtype in {SYM_INT4, FP8E5, FP4, IQ2_XXS,
  FP6, WOQ_INT4}, and a device rule for FP6. **One row only** — decode. The prefill path is the
  unfused gate/up/SiLU.

## 4. Weight repack (score 7)

- `reorder_qw_q4_0` (`code`, SPIR-V module 28 of `libggml-sycl.so`); host entry
  `ggml_q4_0_format_convert_to_xpu_ipex_llm_blksize` (symbol); `opt_for_reorder(...)`
  (`mul_mat_algo`) called before a quantized matmul at `ggml_sycl_mul_mat` `0x9f18b`. The
  per-card matvec geometries are three instantiations of
  `ggml_sycl_op_dequantize_mul_mat_vec_q4_0_kernel<32,16,2>`, `<32,16,8>`, `<64,32,8>` (`code`).
- No Python file:line — the repack is inside the wheels, not in this repository.

## 5. The q4_0 batch kernel (score 6)

- **PyTorch** (`code`): `use_batch_forward(x, qtype, output_len)` (`low_bit_linear.py:292-333`):
  `x.dtype in {float, half}`, `x.shape[1] % 128 == 0`, and for {SYM_INT4, ASYM_INT4, FP8E5, FP8E4,
  WOQ_INT4} `batch_size <= 48` or (`<= 64` and `x.shape[1] % 256 == 0` and `output_len % 64 ==
  0`); for {SYM_INT8, FP4, FP6, Q4_K, Q6_K, WOQ_INT4} `batch_size <= 48` on arc/pvc/mtl/arl; the
  return is true when `batch_size > 1`, or the named card/qtype pairs (bmg with SYM_INT4/WOQ_INT4/
  FP8E5). Dispatch: `xe_batch.batch_forward(x_2d, w, self.qtype)` at `:696-699` and `:824`.
  `q4_0`'s torch qtype is the 4-bit family (`SYM_INT4`/`WOQ_INT4`); the commit `12396` calls it
  "new q4_0 batch kernel" — the code agrees through the qtype, not through the name.
- The gate is a **small-batch** gate: nothing over 64 rows reaches it, so a 512-token prefill
  does not. The llama.cpp default dispatch also does not route to `batch_forward` (binary §5).
- The kernel family (`vec_q4_0_batch_kernel<float,8,2,…>`) uses f32 activations
  (`batch_forward_q4_0(float const*, unsigned char const*, float*, ...)`); commit `5e967205ac`
  removed the f16 cast, `code` agrees.

## 6. fp8 KV cache (score 5, listed)

- **PyTorch** (`code`): `use_quantize_kv_cache` (`models/utils.py:77-100`): env
  `IPEX_LLM_QUANTIZE_KV_CACHE` (and deprecated `BIGDL_QUANTIZE_KV_CACHE`, `IPEX_LLM_LOW_MEM`),
  else a quantized weight and `num_kv_heads >= 4` and (arc/bmg and `x.size(0) > 1`, or mtl/lnl/arl
  with GQA ratio ≤ 4). Cache ops: `init_fp8_kv_cache` `:102-140`, `append_fp8_kv_cache` `:117-140`,
  unbalanced and restore variants `:141-190`; the attention path at `:315-325` appends and hands
  the uint8 cache to the `sdp_fp8*` call (`models/common.py:271-286`).
- **llama.cpp** (`code`): `quantize_key_kernel`, `reshape_fp8_*`, `sdp_fp8_causal_xmx_kernel`
  (binary §8). Quality class: changes the answer; not audited.

## 7. MoE route and top-k (score 5)

- **PyTorch** (`code`): the attention/MLP dispatch files are per model; no grouped-routing file
  was read here beyond `models/deepseek.py` (commits `be1f073866`, `39e360fe9d`).
- **arcint** already fuses the router (patch 0064's record: softmax, top-k, clamp, normalise in
  one kernel); the *grouped* form is for DeepSeek-style routing, which our units do not have. So
  this is "partly, and the remaining part is not our model".

## 8. fp32 softmax (score 4)

- commit `11776` names "new fp32 softmax kernel"; the llama.cpp SPIR-V has
  `soft_max_f32_submitter<…>` instances at 32/64/128/256/512/1024/2048/4096/1024 blocks
  (binary §2). arcint's prompt attention keeps the online softmax in f32 (0008's record). No
  mechanism gap read.

## 9. Performance mode / batch retry (score 3)

- `libggml-sycl.so` strings: `"%s: retrying batch size %d after cache optimization"`, gated by
  `sched` code that was not disassembled; the env `IPEX_LLM_SCHED_MAX_COPIES` is a string only.
  `IPEX_LLM_PERFORMANCE_MODE` is in the Python tree. `code, symbol only`.

## 10. Prefill attention parity (score 2, no gap)

- IPEX: `sdp_causal_xmx_kernel<128,64,8,16,16>` and its three shapes, fp16 K/V (`code`).
  PyTorch gate: `use_sdp_causal` (`models/utils.py:271-280`: `q_len == kv_len`, head 64/80/96/128,
  xpu, f32/f16) and the dispatch `models/common.py:254-290` (head dims 64/80/96/128/192/256; fp8
  when the cache is uint8).
- arcint: `flash_attn_dpas.cl` DK 256 and 128 (`ggml-opencl.cpp:4719-4725`, dispatch
  `:23175-23209`), f32 Q, f16 K/V. Same mechanism at our head sizes; the fp8 variant is the only
  difference and it is quality-changing.

## What the closed binary leaves unread, and what a card measurement would tell apart

- The fused qkv/mlp op's shape threshold (does a 512-row prefill stay fused?) — a per-op profile
  (kernel names/launches in a SYCL trace, or a one-command `ONEDNN_VERBOSE` read showing whether
  the GEMM is oneDNN for that call) would tell.
- The oneDNN primitive's dtype and blocking — `ONEDNN_VERBOSE=1`.
- Whether `GGML_SYCL_DISABLE_DNN=1` makes the llama.cpp path take its own quantized kernels or
  the f32 batch GEMM — the same run's timing and a kernel-name trace would tell.
- Whether `GGML_SYCL_DISABLE_OPT` gates the bigdl graph fusions — no read path ties them; a
  compare of op counts would tell.
