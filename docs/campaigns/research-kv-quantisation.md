# research-kv-quantisation — how other engines quantise KV and price prefill

Feeds `u8i4-prefill-price.md` (the +7/+25/+72% prefill cost of `--paged-kv
u8:i4` at ~9k/~38k/~72k tokens, opt-attention-path scratch read as the
mechanism, not yet measured). Literature/engine survey, not a measurement —
nothing here closes that gate. Access date throughout: 2026-09-05.

## llama.cpp / ik_llama.cpp

`-ctk`/`-ctv` set K/V dtype independently (f32/f16/bf16, q8_0, q5_0/q5_1,
q4_0/q4_1, iq4_nl, q5_k, q6_k, iq3_xs, iq2_xxs). Fused FA kernels require
**matching** K/V types; a mismatch silently falls back to the slower unfused
path (GitHub Discussion #22411) — the documented reason V-quant needs FA on,
not a free-standing K-only feature. No source on write-time vs deferred
V-quant, or scratch sizing. Measured: DGX Spark (GB10/Blackwell), Nemotron-3-
Nano-30B-A3B Q4_K_XL, build 8399 — prefill at parity to 32k (f16 328 vs q4_0
317 tok/s) then a cliff at 64k, 282.7 → 21.3 tok/s (-92%); decode degrades
smoothly (-35%) over the same range (NVIDIA dev-forum thread) — flat-then-
cliff-with-depth, the closest external analogue to arcint's own curve,
undiagnosed there too. A claimed "25-45x slowdown without `-DGGML_CUDA_
FA_ALL_QUANTS=ON`" traces to no primary source — **unverified**. License MIT;
no Intel Arc/SYCL/Vulkan KV-quant numbers found.

ik_llama.cpp (MIT fork) adds `q8_KV` beyond upstream's KV types; MLA modes
(`-mla 1/2/3`) are its bigger differentiator. A `q6_0` KV type could not be
confirmed — **unverified**. Absolute throughput exists (Threadripper PRO + RTX
A6000: prefill 69-114, decode 7-14 tok/s, #258) but isn't a quant-vs-f16 delta;
no prefill-specific cost found for it either.

## exllamav2 / exllamav3

MIT, CUDA-only, no Arc applicability. Q4/Q6/Q8 cache with a Hadamard rotation
before quantisation; "calibration-free" means no per-model calibration set.
Measured (author's `doc/qcache_eval.md`): Mistral-7B, Mixtral-8x7B, Llama2-7B
on The Pile plus ~9-12k-token summarisation — Q4/FP8 cache perplexity within
~0.1 of FP16. Speed (user-reported issue #499, not author-measured):
Llama-3-8B, RTX 4090, 20k context, FP16 36.24 → Q4 24.06 tok/s decode, a
memory-bound dequant kernel. No prefill-isolated number or write-time-vs-
deferred statement found.

## vLLM, SGLang, TensorRT-LLM

**vLLM** (Apache 2.0): FP8 KV cache, default uncalibrated per-tensor scale=1.0,
or LLM-Compressor-calibrated. Measured, vLLM's own blog (April 2026):
Gemma-4-E2B, head_dim=256 — FP8 prefill TTFT's quadratic coefficient runs ~1.6x
BF16's (6.93e-7 → 1.12e-6 ms/token²). Stated cause: FlashAttention-3 on Hopper
needs two-level FP32 accumulation past ~100k contraction-dim to avoid precision
loss (also seen against DeepSeek-V3) — **the closest published mechanism-level
analogue to arcint's "scratch scales with depth" reading**, and explicitly a
Hopper-kernel workaround: Blackwell/FlashInfer skips it, no regression shown
there. Mitigations shipped: tuned prefill tiling for head_dim 64/128, a
per-layer dtype-skip flag, torch.compile-fused query quantisation; no
tail-window discussion found. Not portable to Arc, but the mechanism
generalises: a fixed-precision-recovery buffer whose need is a function of
context length.

**SGLang** (Apache 2.0): `--kv-cache-dtype fp8_e4m3` shipped for MLA models
(DeepSeek V3/R1); MHA support in progress. Clearest first-party admission in
this survey (issue #10083): today's implementation quantises at store time
and dequantises at read time for attention — "suboptimal, wastes time for
quant/dequant" — proposed fix: fuse dequant into the attention kernel
(borrowing TensorRT-LLM's approach) so no standalone dequant step exists. No
tail-window or prefill/decode split published; not portable to Arc.

**TensorRT-LLM** (Apache 2.0, NVIDIA-only, no Arc applicability): INT8/FP8 KV
cache both dequantise before the attention math runs (NVIDIA docs) —
quantisation saves bandwidth/capacity, not attention compute. Third-party
measured (SqueezeBits' vLLM-vs-TensorRT-LLM series): up to 1.09x throughput
prefill-heavy, 1.45x decode-heavy, FP8 beating INT8 because dequant is
cheaper — a third source (with vLLM, SGLang) placing prefill cost at dequant.

## KIVI / KVQuant

KIVI (arXiv:2402.02750, ICML 2024, MIT code jy-yuan/KIVI): per-channel Key
quantisation, per-token Value quantisation, 2-bit, tuning-free. Keeps a
configurable `residual_length` — recent N tokens stay fp16 — the same
asymmetric-plus-residual shape arcint's u8:i4 uses, minus the residual window.
Prefill quantisation is progressive (applied as tokens are processed, not
deferred to a post-prefill pass), per the repo's own description. Measured:
2.6x peak memory reduction, 2.35-3.47x throughput (authors' numbers); no
prefill-only latency delta vs FP16 found — speed claims read as decode.

KVQuant (arXiv:2401.18079, NeurIPS 2024): per-channel, pre-RoPE Key
quantisation; per-token Value; sensitivity-weighted non-uniform datatype,
calibrated offline; sub-4-bit. <0.1 perplexity loss at 3-bit (paper's claim).
The 1M/10M-token context demo is the authors' own run — **mark as author's
claim**. ~1.7x speedup for matrix-vector multiplies (7B model, 80 GB card)
reads as decode-phase; no prefill-phase cost discussed.

## TurboQuant (2025), QAQ, QJL

TurboQuant (arXiv:2504.19874, ICLR 2026) is not KV-specific — a general online
vector-quantisation method (Shannon source-coding framing) with KV cache as one
application: Hadamard/JL rotation plus per-coordinate Lloyd-Max scalar
quantisation, calibration-free. Claimed 3.5 bits/channel "quality-neutral"
(paper only); no hardware/prefill numbers at abstract level. Adjacent 2025/2026
titles turned up in search but were **not** verified — noise.

QAQ (arXiv:2403.04643, Apache-2.0, ClubieDong/KVCacheQuantization): separate
non-uniform K/V quantisation by sensitivity, outlier handling, up to 10x
compression claimed; no residual window, no prefill/decode split found.

QJL (arXiv:2406.03482, AAAI 2025, Apache-2.0, amirzandieh/QJL):
Johnson-Lindenstrauss preconditioner plus 1-bit Key sign quantisation, "zero
overhead" (no stored scale/zero-point). Uses a layered scheme instead of a
residual window — higher precision for the first ~15 layers, coarser beyond, a
depth-in-layers tier rather than token-recency. >5x memory reduction at 3-bit
claimed, no accuracy loss; prefill vs decode not separated in docs.

Across all five academic methods, none reports prefill-phase time as a
first-class measured quantity — all optimise decode-time throughput; KIVI
alone names prefill and decode as distinct code paths.

## Intel Arc / OpenVINO / ipex-llm

OpenVINO GPU paged-attention properties (docs.openvino.ai 2025,
summary-of-summary — direct fetch returned only the nav shell, treat names as
reliable, not verbatim): `KV_CACHE_PRECISION` plus independent
`KEY_CACHE_PRECISION`/`VALUE_CACHE_PRECISION`, and `KEY_CACHE_QUANT_MODE`
choosing `BY_CHANNEL` (paged-attention backend, more accurate) vs `BY_TOKEN`
(SDPA backend). Precisions u8/u4/bf16/f16, group size via
`KEY_CACHE_GROUP_SIZE`/`VALUE_CACHE_GROUP_SIZE` — the same family arcint's
u8:i4 sits inside.

Most relevant hit: openvinotoolkit/openvino PR #29290 ("Enable KV-cache
compression in PagedAttention OCL kernel," 403'd on direct fetch, via a
fetch-tool summary — treat kernel names as reliable, not a verbatim diff)
enables INT8_ASYM KV compression and touches the **`pa_sdpa_opt` kernel**
specifically, adding a bandwidth `scale_factor`, and makes `kv_cache_update`
quantise on write, `kv_cache_rotate` dequantise→rotate→quantise. This confirms
upstream OpenVINO GPU routes mixed/compressed-precision KV through a distinct
`pa_sdpa_opt` path — the same "opt attention path" arcint's reading names — but
no OpenVINO issue or doc measures prefill-time cost there or claims
depth-scaling; a search for `exp_sums`/`max_logits`/`tmp_out` matched only a
different pair of PRs (#28279, #28013, second-token softmax fusion), not
confirmed as the same buffers. **Arcint's mechanism is plausible by
kernel-naming adjacency but unverified upstream too** — the same gap the parent
campaign names, now shown to exist in OpenVINO's own tracker as well.

A vendor blog (Medium, "INT4 KV Cache Compression... OpenVINO 2026.2," 403'd,
summary only — mark vendor/marketing) reports Llama-3.1-8B decode-latency-
per-token gains on an integrated Arc GPU, no prefill numbers. ipex-llm (Apache
2.0): Arc/Flex/Max support, low-bit **weight** quantisation (`sym_int4`, `fp8`,
`fp8_e4m3`) and vLLM-on-Intel-GPU integration, but no standalone KV-cache flag
with Arc-measured numbers found — a search gap.

## What transfers to arcint

Three mechanisms recur across the independently-measured sources (vLLM's FP8
TTFT post, SGLang's #10083, TensorRT-LLM's third-party comparison, OpenVINO's
own PR #29290) that would make u8:i4 prefill flat with depth:

1. **Fuse dequant into the attention kernel instead of a separate
   dequant-then-attend stage** (SGLang #10083, citing TensorRT-LLM's fused
   kernel as the model): the packing-class mismatch that makes micro-SDPA
   decline to the opt path (DESIGN §7.0.2ab) needs a kernel reading mixed u8/i4
   operands directly instead of materialising a scratch buffer — plugin-kernel
   work, not an engine-layer change.
2. **Bound the precision-recovery buffer to a fixed size instead of one scaling
   with context length** (vLLM's own FP8 TTFT regression: extra accumulation
   needed only past a hardware precision limit — the same shape as arcint's
   "scratch scales with depth" reading). If arcint's opt-path scratch
   (`exp_sums`/`max_logits`/`tmp_out`) shares that shape, size it per chunk —
   as the VRAM belt in `packed_values_prefill_scratch_bytes_ex` already does —
   rather than per depth; again plugin-kernel work, since the open point is
   whether *time* cost follows chunk or depth.
3. **A KIVI-style residual/tail window** (recent N tokens at full precision)
   sidesteps rather than answers the question: it caps how much of the sequence
   enters the mixed-precision path, at the cost of memory savings on that tail
   — engine-side (a paged-KV manager policy), not a plugin-kernel change, and
   it widens the byte-identity invariant's scope rather than changing it.

Only (2) is checkable without new kernel work: whether `fit.h`'s existing
per-chunk formula (`packed_values_prefill_scratch_bytes_per_token_ex`, sized by
chunk for VRAM) produces a *time* cost scaling with depth rather than chunk
size when profiled — the measurement this campaign's gate already calls for.
This survey doesn't change that gate; it shows the reading is plausible by
analogy to vLLM's and OpenVINO's own admitted mechanisms, not arcint-specific
speculation, and names the first check: depth-scaling of the opt path's own
scratch allocation against wall time.
