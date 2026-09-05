# Sub-4-bit weight execution on GPUs — a 2026-09-05 survey

Scope: sub-4-bit weight execution on GPUs, especially Intel Arc, feeding
`sub4bit-vram-kernel.md` (GPU kernel for VRAM-resident sub-4-bit MoE
experts) and `kquant-host-storage.md` (host CPU kernel for native
K-quant/IQ-class blocks). Each entry: format, kernel approach, measured
numbers (MARKETING = vendor claim, unverified), license, Arc status, URL.
Access date 2026-09-05 throughout.

## llama.cpp K-quants and I-quants (GGUF)

256-weight super-block (`QK_K`); i-quants add a codebook-index layer.
**Q2_K** 2.625 bpw: `scales[16]` (4-bit scale+min) + `qs[64]` (2-bit codes),
bit-unpack + scale multiply, no LUT. **Q3_K** 3.4375 bpw: `hmask[32]` +
`qs[64]` (low 2 bits) + `scales[12]` (6-bit), same approach. **IQ2_XXS/XS/S**
2.06–2.56 bpw: indices into a fixed 256-entry E8-lattice codebook, LUT
dequant. **IQ3_XXS/S** 3.06–3.44 bpw: same LUT, plus sign bits. **IQ4_XS**
~4.25 bpw (boundary): non-linear scale map, no codebook.

llama.cpp's **SYCL** backend lists dequant kernels for `IQ4_NL, IQ4_XS,
IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M` — coverage on paper,
but an open GitHub issue reports SYCL producing garbled output on Arc B580
with these types while **Vulkan** on the same model/card is correct — a
reliability gap, unresolved. Vulkan has dedicated per-type shaders
(`dequant_iq1_s.comp` etc.) and is the more stable Arc path today. No
independently-measured, format-isolated tokens/s or perplexity numbers for
IQ2/IQ3 vs Q4 on A770/B580/B60 were found with solid citation; general Arc
decode numbers in blog posts are MARKETING-grade for this purpose. License:
MIT. (github.com/ggml-org/llama.cpp — ggml-sycl, ggml-vulkan, B580 SYCL issue)

## ik_llama.cpp — IQ*_K, IQ*_KS, IQ*_KT (trellis)

Refines K/I-quant layouts (`IQ*_K`, `IQ*_KS`) and adds **trellis-coded**
types (`IQ1_KT`…`IQ4_KT`): dequant is a procedurally generated sequence,
not a stored codebook — better GPU throughput than LUT-based IQK at equal
bpw, ~5x slower to produce, slower CPU decode than IQK. No IQ_KT-vs-IQK
table found. **Disqualifying for arcint**: README states only **CPU
(AVX2+/NEON+) and CUDA (Turing+)** are "fully functional"; SYCL, Vulkan,
OpenCL, ROCm, Metal unmaintained. No Arc mention. License: MIT, but nothing
ports to OpenCL/oneDNN without a rewrite — format reference only.
(github.com/ikawrakow/ik_llama.cpp)

## exllamav3's EXL3 and QTIP (trellis families)

**EXL3**: trellis-coded, 2–8 bpw, non-uniform per-tensor allocation from
Hessian-weighted importance; procedural in-kernel codebook decodes a
tail-biting trellis, no stored LUT. turboderp's doc claims ~3.0 bpw ≈
GGUF-Q5 quality at half the size — self-reported, MARKETING until
reproduced. **CUDA only**: ROCm unimplemented, no Vulkan/SYCL/OpenCL, no
Arc mention. License: MIT. (github.com/turboderp-org/exllamav3)

**QTIP**, the method EXL3 derives from: Hadamard incoherence processing +
bitshift trellis + compute-based codes, CUDA-tuned. Paper (NeurIPS 2024,
arXiv:2406.11235) reports SOTA perplexity/size vs AQLM/QuIP# at 2–4 bpw on
Llama-2/3 — peer-reviewed, CUDA-only. **License: GPL-3.0** — disqualifying
for arcint's MIT/Apache/BSD rule regardless of CUDA-only status.

Both CUDA-only by construction, QTIP GPL-licensed — same verdict as
ik_llama.cpp's KT types: worth reading for the trellis idea, zero transfer
to Arc.

## AQLM and HQQ

**AQLM**: weight groups = sum of learned-codebook vectors (1×16: one
65,536-entry codebook; 2×8: two 256-entry codebooks), 2–3 bpw; dequant is a
codebook lookup. Measured: Llama-2-7B 1×16 (2-bit) WikiText2 PPL 6.29 vs
2×8 PPL 7.98 — real — but RTX 3090 matvec speedup only 1.31–1.57x vs FP16
despite the bit-count drop. A CPU variant splits 16-bit codebooks into
8-bit sub-codebooks for cache residency — relevant precedent for
`kquant-host-storage`'s host kernel (idea, not format). No Arc port.
License: MIT. (github.com/Vahe1994/AQLM, arxiv.org/pdf/2401.06118)

**HQQ**: calibration-free group quant, natively 8/4/3/2/1-bit via
scale/zero-point per group. VRAM claims (Llama-2-70B 2-bit ≈18.8 GB) are
vendor figures — MARKETING. Fast fused paths (torchao/Marlin) are **4-bit
only**; 3-bit/2-bit fall back to slower "Gemlite"/native backends, and
3-bit isn't in Gemlite's documented set — likely unfused Python dequant.
No Arc/XPU backend. License: Apache-2.0. (github.com/dropbox/hqq)

## Marlin, Machete, and 3-bit GPTQ/AWQ

Marlin/Machete (vLLM/Neural Magic): **4-bit/8-bit weight-only GEMM only** —
no genuine 3-bit variant. NVIDIA tensor-core only (Marlin: Ampere+/Hopper+;
Machete: Hopper) via CUDA — no Arc/SYCL port. 3-bit GPTQ exists as a Triton
kernel (GPTQModel changelog, Dec 2025) but is a minority path in a
4-bit-dominated ecosystem. Arc-relevant but off-target: **GPTQModel added
Intel XPU (Arc + Data Center Max) kernels in Nov 2025** (fused AWQ on Torch
2.8) — real and dated, 4-bit only. The single concrete "Arc-native
low-bit-adjacent kernel shipped in 2025–2026" found outside Intel's stack,
still int4-floored. (developers.redhat.com/Machete,
github.com/vllm-project/vllm/gptq_marlin.cu, github.com/ModelCloud/GPTQModel)

## ipex-llm — Intel's own low-bit formats on Arc

Documented `low_bit`: `sym_int4` (default), `asym_int4`, `sym_int5`,
`asym_int5`, `sym_int8`, plus `nf3`, `nf4`, `fp4`, `fp6`, `fp8`, and
GGUF-import modes `gguf_iq2_xxs`, `gguf_iq2_xs`, `gguf_iq1_s`,
`gguf_q4k_m/s`. No `sym_int3`. `nf3` (3-bit NormalFloat) and the imported
IQ2/IQ1 modes are the sub-4-bit entries that matter. **The one clear
precedent of sub-4-bit weights executing on Arc silicon today**: release
notes describe "initial INT2 support (based on llama.cpp's IQ2 mechanism)…
run Mixtral-8x7B on Intel GPU with 16GB VRAM" (Feb 2024, still current) — a
ported llama.cpp IQ2 kernel inside ipex-llm's own XPU/SYCL backend, not
OpenVINO/XeTLA. No perplexity/throughput numbers for `nf3`/`gguf_iq2_*` on
A770/B580/B60 found — capability real, measurement unpublished. License:
Apache-2.0. (github.com/intel/ipex-llm)

## OpenVINO / NNCF — INT3/INT2 status

Public `CompressWeightsMode` values (2026 docs): `INT8_SYM/ASYM`,
`INT4_SYM/ASYM`, `NF4`, `MXFP4`, `MXFP8_E4M3`. Nothing below 4-bit appears
in any public doc/changelog found — consistent with (not an independent
reconfirmation of) `sub4bit-vram-kernel.md`'s finding that
`INT3_SYM`/`INT2_SYM` exist in the enum but are undocumented; a direct
fetch of the enum source 404'd here, so this pass defers to that prior
finding. Execution: INT4_SYM/ASYM run on GPU today; even NF4/MXFP4/MXFP8
are flagged **experimental on GPU and NPU** in the 2026.2 release notes —
Intel's roadmap hasn't hardened even 4-bit-adjacent formats on GPU. No GPU
kernel below u4 found anywhere in NNCF/OpenVINO. License: Apache-2.0.
(github.com/openvinotoolkit/nncf, docs.openvino.ai weight-compression guide)

## XeTLA, cutlass-sycl / SYCL-TLA

**XeTLA**: int4 is labeled **experimental**, with a k-sliced dequant GEMM
policy (`dispatch_policy_int4_dequantize_kslicing`) doing in-kernel
int4→f16 dequant against group-wise asymmetric scale/zero-point
(GPTQ-style, group≈128). No int3/int2 kernel anywhere — int4 is the
floor, itself experimental. **SYCL-TLA** (formerly cutlass-sycl): mixed-
dtype GEMM (FP16+INT4), same int4 floor. Both Apache-2.0/BSD Intel OSS.
Intel Neural Compressor confirms INT4 weight-only quant on Arc B-Series,
no sub-4-bit claim. Net: Intel's own GEMM libraries don't reach below int4
even experimentally — strongest confirmation here that a sub-4-bit Arc
kernel has nothing Intel-native to adapt from; new code by construction.
(github.com/intel/xetla, github.com/intel/sycl-tla)

## Mixed per-expert precision in MoE

- **MoQE** — name collision. (a) Kim et al. (Microsoft, arXiv:2310.02410,
  2023): uniform ultra-low-bit (down to 2-bit) on *all* expert FFN weights;
  claims 79.6% size reduction, 1.24x speedup on A100 (single-source,
  MARKETING-adjacent) — not per-expert-selective, the opposite of arcint's
  idea. (b) Zhang et al. (arXiv:2508.09204, 2025): routes *inputs* to
  different full quantized model variants, unrelated.
- **MiLo** (arXiv:2504.02658, MLSys 2025): INT3 on all experts plus
  adaptive-rank low-rank compensators, calibration-free, Tensor-Core 3-bit
  kernels; tested on Mixtral-8x7B, DeepSeek-MoE. Code public, **MIT**.
  Uniform quant plus compensation, not frequency-based selection.
- **MxMoE** (Duanmu et al., ICML 2025, arXiv:2505.05799): per-linear-block
  bit-width via an ILP over quant-loss sensitivity and a hardware
  performance model — closer to arcint's goal but sensitivity-driven, not
  routing-frequency-driven. Reports 2.4 lower WikiText2 PPL than GPTQ at
  2.25-bit, up to 3.4x speedup, mainly RTX-4090, Qwen2/Qwen1.5-MoE; no Arc
  angle. License unconfirmed — don't treat as vendorable without checking.
- **MC-MoE** (arXiv:2410.06270, ICLR 2025) — closest match to arcint:
  training-free, per-expert bit allocation as a Linear Program over
  activation frequency, activation weight, and quantization loss.
- **MoPEQ** (arXiv:2509.02512, 2025) — a caution: routing frequency alone
  is a weak proxy for low-bit tolerance; pairs Hessian-trace sensitivity
  with frequency instead; ~1.5x size reduction within 5% accuracy loss,
  VLMs only. Code public (github.com/krishnateja95/MoE-Mixed-Prec), CC BY
  4.0. **MC-MoE and MoPEQ agree frequency alone is a weak importance
  proxy** — relevant to arcint's recon note that the routing histogram left
  most experts at 0–2 routings: picking the sub-4-bit set by frequency
  alone carries a documented risk to measure, not assume away.
- **llama.cpp's imatrix** already does per-tensor/per-expert *type*
  selection, not just scale: `llama-quantize --tensor-type` forces
  tensors/regexes to specific `ggml_type`s; imatrix recipes (GitHub
  discussion #15576) simulate quantize→dequantize error per candidate type,
  build a Pareto frontier, and greedily assign types under a bpw budget — a
  shipped precedent for heterogeneous per-tensor precision in one artifact.
- **Bpw map storage**: GGUF stores quant type as a per-tensor field in each
  tensor's metadata (name, shape, `ggml_type`, offset) — heterogeneous
  per-tensor precision is GGUF's normal case, no sidecar needed. No
  credible published sidecar-manifest precedent found elsewhere; arcint's
  unrolled `--moe-lowering` form would be inventing that pattern.

## What transfers to arcint

- **Resident VRAM format**: only llama.cpp's K-quant/I-quant family is a
  plausible source — simple, non-trellis, non-multi-codebook dequant
  (bit-unpack-and-scale for Q3_K/Q2_K, small-LUT for IQ3_XXS/IQ3_S),
  MIT-licensed, with an existing GPU dequant design (Vulkan shaders)
  readable for structure, though none of the code is OpenVINO/oneDNN-
  portable. Trellis families (EXL3, QTIP, ik_llama.cpp's KT types) are
  CUDA-only by construction, QTIP GPL-licensed — none is a candidate
  either way; their case depends on CUDA execution Arc/OpenCL lacks.
  AQLM/HQQ's codebook/LUT approaches showed weak GPU speedup even on
  well-supported CUDA kernels — a bad sign for a first Arc kernel that
  needs to prove the concept cheaply. **Q3_K (3.4375 bpw, 256-weight
  block, 6-bit scales) is the best first candidate**: closest to the
  campaign's 3.2–3.5 bpw target, simplest math, byte layout already
  specified in `sub4bit-vram-kernel.md`.
- **Host K-quant kernel**: same conclusion — Q3_K/IQ3_XXS decode is
  scalar/SIMD bit-unpack with no GPU-specific tricks, a bounded
  AVX2/scalar port, unlike porting a trellis generator's numerics to a
  host float pipeline. AQLM's codebook-splitting-for-cache trick (16-bit
  codebook split into 8-bit sub-codebooks) is the one transferable idea,
  not format, if a codebook-style grid ever needs cache residency.
- **Fusion-matcher/oneDNN-bypass shape**: every Intel-native library
  surveyed stops at int4 as a hard floor — none has a sub-4-bit kernel or
  fusion pattern to adapt. ipex-llm's ported IQ2 kernel lives in its own
  SYCL backend, outside OpenVINO's plugin architecture — it proves Arc's
  OpenCL/SYCL path *can* do in-kernel I-quant dequant, but transfers no
  code or pattern to `keep_moe_3gemm_const_precision.cpp`. The new fusion
  matcher and GEMV/GEMM have no upstream analog either way; the
  800–1,500-line estimate stays a HYPOTHESIS.
- **Per-expert selection axis**: MC-MoE and MoPEQ both caution that
  routing frequency alone is a weak importance proxy. Entry criterion (3)
  — the routing histogram — is necessary, not sufficient; a coarse
  sensitivity check on candidate experts would catch a rarely-routed but
  high-sensitivity expert before it's committed to the sub-4-bit tier.
- **Smallest first measurement**: the entry criterion already named — one
  expert layer, Q3_K vs the int4 baseline, same f16 source/calibration,
  decode correctness plus a fusion-impact profile — is the right first
  cut: it isolates the simplest, MIT-licensed format with the clearest
  byte-layout reference before any commitment to the larger build-out.
