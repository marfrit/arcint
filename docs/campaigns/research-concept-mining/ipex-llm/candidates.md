# Pass 2 — ranked concepts from IPEX-LLM, for arcint (2026-11-05)

Merged from the pass-1 records at 2-3 (131 records) and from `binary.md` (the llama.cpp path read
as a binary), ranked against the three served units. IPEX-LLM's numbers are `paper` for us; what
was read in the wheel is `code`; what the commits state is `paper`; nothing here is
`measured-here`. Scores 1-10 are this session's judgement; the pass-1 score is in parentheses.

Reference rows (the task's premise, `paper`): IPEX-LLM llama.cpp `llama-cpp-ipex-llm-2.3.0b20250724`,
Llama 2 7B Q4_0, `llama-bench` like-for-like. Arc Pro B60 (Ubuntu): pp512 **2861.84** / tg128
**82.24** with flash attention on, 2861.53 / 82.92 off. Arc A770 (Windows): up to **2564.25 / 72.54**
(fa on) and 2416.16 / 82.92 (fa off). arcint 1.5.1 on the same table (`paper` as published): B60
2482.36 / 80.58, A770 2074.21 / 69.71. Derived: prefill **+15.3 %** on the B60, **+23.6 %** on the
A770 (against the higher IPEX row), decode +2.1 to +2.9 %. Flash attention on vs off moves prefill
by **0.01 %** on the B60 — the attention is not what makes the lead.

A note on Llama 2's table versus the units: a dense Q4_0 gain is a benchmark row, not a unit. Every
candidate below names the unit it would serve; the top three also name what part of it is only the
table.

---

## Top 10

### 1. The prefill matmul as a dequantize-then-vendor-GEMM chain — score 9 (p1 2)
- **Evidence**: `binary.md` §5 (`code`): `ggml_sycl_mul_mat` falls through to
  `ggml_sycl_op_mul_mat_sycl` (`0xd5cb0`) for a quantized src0 with more than a few columns; that
  function dequantizes the whole src0 to f16 via `ggml_get_to_fp16_sycl` into a pooled scratch
  buffer and issues a **oneDNN** matmul (`dnnl::matmul::primitive_desc`, `dnnl::primitive::execute`,
  3 sites), or an **oneMKL** gemm with `IPEX_LLM_SYCL_USE_MKL=1` (`0xd5e17`). `GGML_SYCL_DISABLE_DNN`
  (int, default 0, `0x87857`) turns it off. A BF16 src0 branch exists (`0xd5d03`). Commits:
  `4da93709b1` ("use oneDNN GEMM for cpp", docs/setup), `f9ee7898c8` (oneDNN dependency fix). `paper`.
- **Measured**: the reference row above; no commit states a per-mechanism number. The isolating
  number does not exist yet — it is the experiment.
- **Why it ranks first**: it is the only mechanism that explains a **+15/+24 % prefill** on a dense
  Q4_0 model with attention on/off equal, and it is the one part of their prefill stack that our
  series does not have: arcint's 0096 dequantizes q4_0 into f16 **tiles** and runs its own XMX tile
  GEMM (`code`: `kernels/mul_mm_kq_f16.cl:2240`, dispatch `ggml-opencl.cpp:29180`), while IPEX
  feeds a whole-layer f16 matmul to a vendor library's tiling. Same idea, different tiling and
  scheduling; the 15-24 % is the difference between them.
- **What ports**: not oneDNN (it has no OpenCL path). What ports is the *shape* of the chain
  (materialize the layer's f16 operand once per ubatch, one large GEMM call, a library-grade tile
  and a stream that overlaps the dequantize with the previous GEMM) — and the measurement that says
  whether arcint's own tile leaves that much on the table.
- **Units**: the agent (dense 27B prefill) and the coder/A770 first; Flash-Next's prefill is
  expert-stream-bound, so it gains only where the dense part (attention projections, shared
  experts) is the wall. The Llama 2 row is the table; the unit prize is the agent's 512-token
  prefill and the coder's 1,024.
- **Cost**: medium — either improve the existing XMX tile's occupancy/tiling, or add a
  "materialize f16 operand + big GEMM" arm. A vendor library cannot be called from OpenCL; the
  alternative is to compare against `clBLAS`-class kernels only if one exists on the host (none is
  assumed), so the honest port is a tiling experiment on our own kernel.
- **Experiment**: on the B60 and A770, IPEX `llama-bench` on Llama 2 Q4_0 with
  `GGML_SYCL_DISABLE_DNN=0` (default) and `=1`; the delta is the vendor-GEMM share. Then arcint's
  own arm: a 512-token prefill profile of the q4_0 GEMM's achieved GFLOPs against the card's XMX
  peak; if our GEMM is below ~70 % of peak while IPEX's row is 15 % ahead, the tiling is the lever.
  Bar: prefill +10 % with answers and KL inside the answer-level bar (rounding-only class).

### 2. The QKV projection fused with its norm and RoPE, and a combined wqkv weight — score 8 (p1 3)
- **Evidence**: `binary.md` §4 (`code`): `llm_graph_context::build_bigdl_qkv_fusion` in
  `libllama.so` (10 tensor pointers, 3 floats, an int; it fetches K/V/head from the KV-cache
  state), op constructors `ggml_bigdl_qkv_fusion` / `ggml_bigdl_wqkv_fusion` in `libggml-base.so`
  (op id 85, output `GGML_TYPE_BF16`), and the small-row q4_0 bodies
  `qlinear_xpu_kernel_q4_0_2x16_qkv{,_neox}` / `…_wqkv_neox` (modules 6/7, plain SYCL, **no XMX**).
  Commits: `a00efa0564` (mlp & qkv fusion, FP16 Llama-7B), `53531ae4ee` (qkv fusion for fp8e5),
  `8ba57b41cd` (merge the quantized q/k/v weights into one), and the long Python chain of fused
  qkv/rope commits (records: `score 2`). `paper`/`code`.
- **Measured**: no commit number. The reference row cannot separate it. The kernel geometry
  (2×16 rows/cols, plain SYCL, no DPAS) reads as a **small-row** kernel, so the readable part is a
  decode/verify lever; whether its convergence path covers a 512-row prefill is **not readable
  from the binary** (`code, symbol only` for the shape claim).
- **arcint's state**: norm+RoPE fused exists and is opt-in (`code`: patch 0074,
  `GGML_OPENCL_FUSE_QK_NORM_ROPE=1`, kernel `kernel_rms_norm_mul_rope_multi_f32`); the three
  projection GEMMs and that kernel are still three or four launches, and no combined-weight form
  exists. So the *incremental* lever is folding the projection into the norm+RoPE kernel, not
  starting from scratch. At prefill the prize is the launch count and the q/k/v intermediate
  traffic; at decode it is the launch count.
- **Units**: all three (every layer has q/k/v). On the coder the A770's small-row shapes are where
  a small-tile form is aimed.
- **Cost**: medium — an epilogue in the XMX GEMM is not how OpenCL DPAS works here (the tile
  output is not row-resident per head); the practical form is a third kernel that reads the three
  GEMM outputs and does norm+RoPE, plus a combined-weight repack at load if the GEMMs are to be
  one call. Red-first test: 0074's geometry cases plus a fused-vs-unfused token comparison.
- **Experiment**: B60, Llama 2 Q4_0, `GGML_OPENCL_FUSE_QK_NORM_ROPE` on/off at pp512 and tg128,
  plus a launch count per layer under `GGML_OPENCL_PROFILING`; bar: +2-3 % decode, no regression
  at prefill, answers identical (the class is rounding-only).

### 3. The MLP fused across gate/up/SiLU/down on the batch path — score 8 (p1 3)
- **Evidence**: `binary.md` §2 (`code`): `mlp_forward_q4_0_kernel<64,32,2>` and
  `mlp_forward_q4_0_kernel_upgate` beside `mlp_forward_q4_k_kernel`; the q4_0 pair is plain
  SYCL (modules 10/11), the q4_k one carries genx intrinsics (module 9); graph op `ggml_bigdl_mlp_fusion` (BF16 output); commits `a00efa0564`, `11574` (silu_mul
  fusion for qwen2), `11678`/`12360` (fused MLP for gemma2/glm4), `11032` (fp6 MLP fusion),
  `019293e1b9` (MoE index fuse). `paper`/`code`.
- **Measured**: no commit number; the reference row does not separate it.
- **arcint's state**: `fuse_mm_glu` exists and is default-on, but it is a **GEMV** fuse of
  gate+up+GLU for a q4_K FFN (`code`: `ggml-opencl.cpp:9836-9878`, kernel
  `kernel_gemv_noshuffle_q4_k_f32_glu`); the prefill GEMM path and q4_0 are not fused, and the down
  projection is always separate. That is the gap this candidate names.
- **Units**: all three at decode/verify; at prefill the intermediate traffic (gate/up 512×11008×2,
  SiLU) is a few milliseconds a layer, so the prefill prize is small and the decode prize is the
  launch count. Name it as a decode lever first.
- **Cost**: medium — a q4_0 GEMM epilogue for SiLU over the gate/up pair is feasible in OpenCL if
  the two GEMMs share the row block; the down projection folded in is not (it consumes the full
  row). Red-first: 0049/0051's fused-vs-unfused cases and a token comparison.
- **Experiment**: coder (A770) and agent (B60), `GGML_OPENCL_FUSE_MM_GLU` on/off at tg128 and
  pp512, q4_0 and q4_K weights; bar: decode +2-4 %, prefill neutral, answers identical.

### 4. Weight repack at load, and card-shaped matvec instances — score 7 (p1 2)
- **Evidence**: `binary.md` §2 (`code`): `reorder_qw_q4_0`/`_q4_k`/`_q6_k` kernels (module 28),
  host `ggml_q4_0_format_convert_to_xpu(_ipex_llm_blksize)`, `opt_for_reorder` run before a
  quantized matmul (`ggml_sycl_mul_mat` `0x9f18b`), and **three DMMV geometries beside each other**
  `dequantize_mul_mat_vec_q4_0_kernel<32,16,2>`, `<32,16,8>`, `<64,32,8>` — a per-card shape.
  Commit `12516` (device-type getter extended to BMG) is the device-class part. `code`.
- **Measured**: no commit number; the decode part of the table is +2-3 %, the mechanism is not
  isolated.
- **arcint's state**: reads raw q4_0 blocks; whether any load-time repack exists is the audit item
  (the 0096 record says "Q4_0 has IQ4_NL's flat planes", no repack mentioned). A repack changes
  only where bytes sit, not the values — exact class.
- **Units**: all three at decode (matvec bandwidth); at prefill irrelevant.
- **Cost**: low-medium — a repack kernel at load plus one changed read path; exact by design.
- **Experiment**: B60 tg128 on Llama 2 Q4_0 and the coder, repack on/off; bar: +3-5 % decode,
  bytes equal.

### 5. The q4_0 batch GEMM with f32 activations, selected by device and shape — score 6 (p1 3)
- **Evidence**: `binary.md` §2 (`code`): `vec_q4_0_batch_kernel<float,8,2,16|32,{1..8},64>` (30
  instances) and `batch_forward_q4_0(float const*, unsigned char const*, float*, long,long,long,
  queue&)`; source path `bigdl-core-xe/llama_backend/batch/int4.cpp`. Commits `00fce5c940` (new q4_0
  batch kernel), `3d5fbf2069`/`145e8b480f` (batch-kernel condition), `2cce89691a` (enable
  `use_batch_forward` on Battlemage, and for fp8e5), `5e967205ac` (the kernel takes the f32 input,
  no f16 cast). `paper`/`code`.
- **Measured**: no commit number. This is IPEX's **own** quantized batch GEMM; in the llama.cpp
  binary the default dispatch does not route to it (`binary.md` §5: no direct call found; a
  negative that was not fully tried).
- **Why it is a candidate**: it is the one prefill GEMM form that could be **faster than a
  dequant-to-f16 chain** — f32 activation × q4_0 weights, no 2× weight materialization — and it is
  gated on Battlemage (B60's family) by the Python path. Its 8×2×16 tile is a different shape from
  the K-quant XMX tile we have.
- **Units**: the coder/A770 and the agent/B60 if the gating is by shape, not by card.
- **Cost**: medium — a q4_0 weight GEMM with deliberate f32 accumulation and a 16-column tile; the
  quality class is rounding-only (f32 activation, q4_0 weight) and needs the KL bar.
- **Experiment**: B60, Llama 2 Q4_0 pp512 with IPEX's `GGML_SYCL_DISABLE_DNN=1` (if that is what
  selects the batch kernel — stated as unverified) and on our side an arm of 0096 with f32
  activations instead of q8_1/x8; bar: +5 % prefill, KL inside the bar.

### 6. fp8 KV cache with a reshape into the attention — score 5 (p1 2, **listed, not audited**)
- **Evidence**: `binary.md` §8 (`code`): fp8 (one byte) `quantize_key_kernel`,
  `quantize_key_value_kernel`, `reshape_fp8_{key,value}_cache_kernel` to f16,
  `sdp_fp8_causal_xmx_kernel` reading the bytes directly; graph ops `ggml_bigdl_quantize_kv` /
  `dequantize_kv`; env `IPEX_LLM_QUANTIZE_KV_CACHE`. Commits `afaa871144` (fp8 KV cache in the
  Python path), `39d90839aa` (llama), `e29e2f1c78`/`11158` (e4m3), `10584` ("enable KV cache
  quantization by default for flex when 1 < batch ≤ 8"). `paper`/`code`.
- **Measured**: no commit number.
- **Quality class**: **changes the answer by design** (a lower-precision K cache); out of the
  audited set per the operator's filter. It is listed because it is their largest KV-memory lever
  and because our units already hold q8_0/q4_0 KV — an fp8 cache would trade quality for bytes.
- **Units**: agent (262k context), Flash-Next and coder — memory-bound contexts; not a speed lever
  for prefill.

### 7. The MoE route fused in one kernel and the grouped top-k — score 5 (p1 2)
- **Evidence**: commits `f6441b4e3d` (moe_softmax_topk), `be1f073866` (fuse the MoE block,
  moonlight), `39e360fe9d` (grouped top-k), `5faba06409` (MoE decode forward), `019293e1b9` (fuse
  MoE index computation); the PyTorch wheel `x/xe/` carries `moe_group_topk_kernel` (`code`:
  SPIR-V entry point in `xe_addons…so`). `paper`/`code, symbol only` for the llama.cpp side.
- **Measured**: no commit number.
- **arcint's state**: 0064 (top-k MoE by sub-group) and the router fusion (0021's record mentions
  SOFT_MAX..ARGSORT..GET_ROWS..SUM_ROWS fused); so partly. The grouped/top-k form is the missing
  piece for a 512-expert MoE.
- **Units**: Flash-Next only. **Quality class**: exact (a selection, same ids) if the tie order is
  preserved.
- **Cost**: low — a kernel over the router logits; red-first: ids equal to the unfused chain.
- **Experiment**: Flash-Next B60, route-kernel time under unitrace; bar: -1 ms a layer's route.

### 8. The fp32 softmax for long sequences — score 4 (p1 2)
- **Evidence**: commits `aa861df066` (route softmax to the new fp32 kernel), `11776` (new fp32
  softmax kernel); `rms_norm`/`soft_max` paths in the binary (`code, symbol only`). `paper`.
- **Measured**: no number.
- **arcint's state**: the softmax is f32 in the DPAS prompt kernels (`code`: 0008's "online
  softmax in f32"); the split decode kernels keep f32 accumulators. Likely no gap.
- **Units**: agent/Flash-Next at long KV. Rounding-only; low priority.

### 9. The batch-size retry and the performance mode — score 3 (p1 1)
- **Evidence**: `binary.md` §7 (`code`: `"%s: retrying batch size %d after cache optimization"`);
  commits `66fe2ee464` (`IPEX_LLM_PERFORMANCE_MODE`), `24c279e0ae` (input-length threshold),
  `a0bbd8e28d`. `code`/`paper`.
- **Measured**: no number.
- **What it is**: a scheduler retry that re-splits a batch after a cache/OOM event, and an
  input-length threshold that switches strategies. Low portability (our engine has its own ubatch
  policy); recorded as background, not a candidate for Wave 1.

### 10. XMX causal prefill attention parity — score 2 (p1 2, **no gap found**)
- **Evidence**: `binary.md` §2/§5 (`code`): `sdp_causal_xmx_kernel<128,64,8,16,16>` and its four
  shapes are a causal prefill flash attention on DPAS, fp16 K/V. arcint has the same mechanism at
  head sizes 128 and 256 (`code`: `kernels/flash_attn_dpas.cl:25` `DK 256` built for `dk` 256 and
  128, `ggml-opencl.cpp:4719-4725`, dispatch `:23175-23209`), with f32 Q and f16 K/V.
- **Measured**: the table's fa on/off equality says IPEX's attention is not the prefill lead;
  arcint's 0008 measured 344 GFLOPS on the B60 (512 rows × 4,096 keys).
- **Verdict**: not a gap for the two cards' head sizes. The fp8 and head-64/80/96/192 variants are
  either quality-changing or not our geometry. Keep as a reference point, not a candidate.

---

## Next 10 (recorded, not ranked)

11. **fp8 SDP at head sizes 64-192** (`sdp_fp8_*`, commits `fad1dbaf60`, `b268baafd6`,
    `dba7ddaab3`) — quality-changing, listed only.
12. **Groupwise prefill** (`70037ad55f`, `0763268e4c`) — Python-side grouping of a prefill by
    weight groups, no source and no number (`code, symbol only`); not rankable.
13. **Self-speculative decoding / lookahead / Eagle** (commits `9951`, `11104`, `10785`, `11021`)
    — our MTP is the served mechanism; this is background, not a port.
14. **oneDNN FA at the pin** (`fattn-onednn.cpp:1`, minimum 32 query rows, F16 K/V) — the pin's
    own, not IPEX's; a reference for what a library does, not a candidate.
15. **SYCL graph capture** (`GGML_SYCL_DISABLE_GRAPH`, `code`) — no OpenCL equivalent.
16. **Weight-format conversion at load for K-quants and q4_0** (`format_convert_to_xpu`) — folded
    into candidate 4.
17. **The BF16 fused-op output** (`code`: op type 30) — a value class note for candidates 2/3, not
    a candidate.
18. **fp6/fp4/nf4/fp8 weight types** (commits `11032`, `10555`, `a3322e2a6c`, `e29e2f1c78`) —
    answer-changing or new formats; listed by design.
19. **`is_UHD_or_IRIS` and `get_gpu_type` classifiers** (`code`) — host-side gating, no kernel
    lever; useful only if a shape gate must be card-aware (candidate 5).
20. **Multi-GPU `ll256_allreduce` and the 32 MiB peer buffers** (`dg2_*_init`, `code`) — not our
    serving shape (one card a unit).

## Lower ranks (from the records, score 0-1)

Perf-test configs and harness (hundreds of commits), model examples, docs, CI, NPU-only paths,
DeepSpeed/vLLM serving, and the pre-2023 BigDL Scala tree: they carry no mechanism for our three
units. The few with a mechanism are folded above (softmax, top-k, qkv/rope, oneDNN, fp8).

## Corrections the re-read forced

- The Python commits mostly state **no numbers** (a parse of all bodies found only two commits with
  a measurable figure among the source-file commits); the numbers in the reference table are the
  benchmark report's, `paper`, not the commits'. The records say "no numbers stated" wherever this
  is so.
- `GGML_SYCL_DISABLE_OPT` gates the in-backend optimisation in `ggml_check_sycl`, but the
  **graph-level** `ggml_bigdl_*` fusions are built in `libllama` and no read path ties them to that
  switch — so "disable OPT = fusions off" is **not verified**; the isolating arm for the fusions is
  not available by environment as read.
- The fused QKV/MLP kernels' geometries read as small-row tiles (plain SYCL for q4_0), not prefill GEMM tiles;
  the prefill claim for candidates 2/3 stays unproven until measured.
- `ggml_sycl_op_mul_mat_sycl`'s dequantize step is per call into a pooled scratch buffer, not a
  persistent f16 weight copy (read in the function's calls: `pool`, `ggml_get_to_fp16_sycl`).
