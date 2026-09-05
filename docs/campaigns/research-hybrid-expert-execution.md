# research: hybrid GPU/CPU expert execution — prior art for the three open campaigns

Web research only, no code changes. Covers `static-partition-prefill` (a:
grouped prefill falls back to a per-expert loop), `partition-seeding` (b:
fixed seed vs. activation-informed placement), `kquant-host-storage` (c:
grouped int4 vs. sub-4-bit K-quant on the host tier). Access date for every
source below: 2026-09-05.

## "ninfer" and "freetoken" — what they actually are

Both names resolve to real, currently-shipping projects — not fabrications —
but neither is what "ahead of llama.cpp" suggests for arcint's problem.

**ninfer** (github.com/Neroued/ninfer, Apache-2.0) is a from-scratch C++/CUDA
engine for a *closed set* of dense Qwen checkpoints (Qwen3.6-27B, Qwen3.8-27B,
Qwen3.6-35B-A3B) on a single NVIDIA RTX 5090 (community forks retarget
4090/3090). No MoE, no CPU offload, no host compute tier anywhere in the
README — it is single-GPU, CUDA-only, and does not run anything that
doesn't fit the card. Irrelevant to all three campaigns.

**freetoken** (github.com/FlashML-org/FreeToken, arXiv:2608.16157) is the one
actually on-topic: an "edge-native MoE serving engine" doing "bandwidth-
adaptive CPU–GPU co-execution" for DeepSeek-V4-Flash, Qwen3.6-35B-A3B, and
GLM-5.2 on consumer/workstation NVIDIA cards. Reported numbers (RTX 5090,
Qwen3.6-35B decode 77–83 t/s; RTX 4060 8 GB laptop, same model, 39.3 t/s; RTX
PRO 6000, GLM-5.2 753B, 14.9 t/s) come from the paper's own agentic workloads
and vendor/community blog posts reproducing the authors' own claims — no
independent third-party remeasurement found. Detail below.

Neither name surfaced in any independent discussion of llama.cpp-class
engines as its actual *successor*; if the operator meant "the specialised
engines beyond llama.cpp for this," the closest real candidates are
KTransformers, ik_llama.cpp, and FreeToken itself — covered below with the
academic literature.

## KTransformers

kvcache-ai/ktransformers, Apache-2.0 (SOSP'25). Keeps attention/KV cache on
GPU, offloads routed-expert FFNs to CPU with AMX/AVX-512 INT4/INT8 kernels;
"hot" experts additionally pinned resident on GPU. **(a)** its headline is
specifically CPU-side *prefill*: AMX GEMM kernels replace the GEMV-only CPU
path other systems use only for decode, because AVX-only kernels are
"insufficient to enable meaningful compute offloading" at prefill batch
sizes — the same failure mode arcint's campaign names. **(b)** GPU pinning is
reported as informed by measured routing data in later releases, not purely
static. **(c)** no sub-4-bit K-quant, INT4/INT8 group quant only. Claimed
4.62–19.74× prefill / 1.25–4.09× decode speedup vs. earlier baselines
(dl.acm.org/doi/10.1145/3731569.3764843); "27.8× llama.cpp" is a vendor blog
figure (LMSYS/noze), not independently reproduced. x86 AMX/AVX-512 + CUDA
only; no Arc/OpenCL/SYCL/Vulkan anywhere in the material reviewed.

## HybriMoE

PKU-SEC-Lab/HybriMoE, built atop KTransformers (arXiv:2504.05897, DAC'26).
Dynamic intra-layer CPU/GPU load balancing plus an "impact-driven"
inter-layer prefetcher and a score-based cache — explicitly *not* a static
or seeded partition; residency follows a running activation score, the
opposite of the history-independence invariant `partition-seeding` must
hold. Reports 1.33× prefill / 1.70× decode vs. plain KTransformers on three
unnamed models, no per-model/hardware breakout at abstract level. Same
platform ceiling as KTransformers; no Arc/OpenCL/SYCL/Vulkan.

## PowerInfer / PowerInfer-2

SJTU-IPADS/PowerInfer, MIT (arXiv:2312.12456, 2406.06282). Predicts, per
input, which *neurons* activate (power-law "hot"/"cold" locality), splits GPU
(hot, resident) vs. CPU (cold, on demand) — a neuron-level predictor for
dense sparse-activation models, not an expert-level MoE router; transfers
only by analogy, since arcint's placement unit is a whole expert's weight
block. PowerInfer-2 retargets the idea at phone SoCs, no relevance to a
discrete-GPU host tier. No Arc/OpenCL/SYCL/Vulkan; no K-quant.

## Fiddler

efeslab/fiddler, Apache-2.0, ICLR'25 (arXiv:2402.07033). **(b)** the most
directly on-topic precedent for `partition-seeding`: place frequently-used
experts on GPU from *offline profiling* of expert popularity, and — the part
llama.cpp-style systems miss — let the CPU *compute* a non-resident expert's
output directly instead of paying a weight-transfer round trip. That
profiling is a one-time calibration, not per-request adaptation, so it is
compatible in shape with arcint's "pure function of (seed, histogram)"
invariant. **(a)** explicitly built to be good at both single-batch and
long-prefill scenarios by design, evidence a residency-aware batched prefill
is achievable without abandoning the whole layer to fallback. Claimed 8.2×
(Quadro RTX 6000) / 10.1× (L4) single-batch latency speedup vs. its own
baselines — paper numbers, not third-party reproduced. CUDA only; no Arc/
OpenCL/SYCL/Vulkan; standard int4/int8 group quant, no K-quant.

## MoE-Infinity

EfficientMoE/MoE-Infinity, Apache-2.0 (arXiv:2401.14361). **(b)** "activation-
aware" expert cache: traces per-request sparse activation, uses the trace to
drive prefetch/eviction — an online, request-history-sensitive scheme, same
category as HybriMoE/PowerInfer, the category arcint's seeding invariant must
not become. **(a)** not its focus. Reports 3.1–16.7× per-token latency vs.
vLLM/Ollama/DeepSpeed/BrainStorm across DeepSeek/Mixtral — a wide range
spanning multiple models/baselines in one number, likely a summary across
conditions rather than one configuration; treat the headline as marketing-
shaped despite peer review. CUDA/host/SSD tiers; no Arc/OpenCL/SYCL/Vulkan.

## Pre-gated MoE

Microsoft Research, ISCA'24 (arXiv:2308.12066). **(a)/(b)** an algorithm
change, not a pure system change: a modified gating function computed one
layer early, so layer L+1's expert selection is known while layer L still
executes, letting migration and compute overlap instead of serializing. A
genuinely different lever than arcint's three campaigns — it changes what
the model computes, not just where — and would need gate retraining, out of
scope for a serving-side plugin change. No public code found; paper-only
prior art. No Arc/OpenCL/SYCL/Vulkan; no K-quant.

## ExpertFlow

arXiv:2410.17954. A trained routing-path predictor (transformer, single
forward pass) plus a token scheduler grouping tokens by predicted expert
before dispatch — predictive and request-conditioned, not a fixed
calibration. Reports up to 93.72% GPU memory reduction and 10× throughput
vs. unnamed "strong offloading baselines." License not confirmed in this
pass — check the repo before any vendoring decision. No Arc/OpenCL/SYCL/
Vulkan.

## prima.cpp

Lizonghang/prima.cpp, MIT, ICLR'26 (arXiv:2504.08791). A different problem
shape: pipelined-ring parallelism across *multiple weak home devices*
(mixed CPU/GPU/RAM/VRAM/disk/Wi-Fi), with a "Halda" scheduler co-optimizing
per-device workload and device *selection* under RAM/VRAM constraints —
computed once per topology, not per request, a broad precedent for
calibration-driven placement, but arcint's problem is single-node,
single-model, GPU-plus-one-CPU-tier, not distributed pipeline placement.
Reports 5–17× lower time-per-output-token vs. llama.cpp/exo/dllama on
four-device clusters — not a comparable hardware shape. No Arc/OpenCL/SYCL/
Vulkan.

## llama.cpp: `--n-cpu-moe` / `--override-tensor` / `-ot`

ggml-org/llama.cpp, MIT. **(a)** `--n-cpu-moe N` moves the routed-expert FFN
weights of the first N layers to a CPU buffer type; `-ot`/`--override-tensor`
generalizes this to an arbitrary regex tensor-to-device assignment. Both are
**static, human-specified at load time** — a layer count or regex, fixed for
the process's lifetime, not derived from routing statistics or request
history (compatible in shape with arcint's invariant, if cruder than a
histogram). ggml's CPU `mul_mat_id` kernel already batches per assigned
expert across all tokens in the current ubatch — one batched matmul per
expert, not per-token dispatch — the same granularity arcint's own host
kernel (`moe_cpu_expert`) already uses. But llama.cpp never attempts a mixed
device/host split *within one layer's batch*: whichever tensors got assigned
to which device just run there, layer by layer — it doesn't solve arcint's
actual defect, it avoids it by assigning whole layers, not experts within a
layer, to one device or the other. **(c)** no sub-4-bit path beyond GGUF's
own Q3_K/IQ3_XXS/Q2_K quant types (the same block formats and byte counts
arcint's own kquant campaign already inventories), decoded by hand-written
AVX2/AVX-512 kernels; no Arc/OpenCL/SYCL/Vulkan.

## ik_llama.cpp

ikawrakow/ik_llama.cpp, MIT. **(a)** `-fmoe` fuses the up/gate/down FFN ops
for MoE layers into fewer kernel launches; no evidence of a residency-aware
device/host split within one prefill batch — same "whichever device, no
cross-device grouped batch" shape as upstream llama.cpp. **(c)** `-rtr`
(run-time-repack) repacks CPU-resident tensors — including K-quant/i-quant
families — into a row-interleaved layout at load for faster CPU matmul, but
the project's own docs flag the cost: "not all quantization types have a
CUDA implementation, this will result in matrix multiplications with these
tensors to be always done on the CPU" — repacking for CPU throughput
forecloses ever offloading that tensor to GPU again. Direct, concrete
warning for arcint's kquant campaign, which wants a *second*, K-quant-native
host path *beside* grouped-int4: whatever layout is chosen needs to stay
device-movable, or the two formats stop being interchangeable residents —
undecided by anything found here. Explicitly CPU (AVX2/NEON) + CUDA only;
maintainers' own words: "please do not enter issues related to ROCm, Vulkan,
Metal ... AVX CPUs" — no Arc, OpenCL, SYCL at all.

## ipex-llm / FlashMoE (Intel, runs on Arc today)

intel/ipex-llm. The one system here that actually runs a CPU/GPU-split MoE
model **on Arc hardware today**: FlashMoE (a CLI built atop llama.cpp) is
documented running DeepSeek V3/R1 (671B) and Qwen3MoE-235B on one or two Arc
cards (A770, B580 named explicitly). Being llama.cpp-based, its split almost
certainly inherits the same static per-tensor `--n-cpu-moe`/`-ot` assignment
above, not a novel grouped-GEMM residency split — inferred, not confirmed
against FlashMoE's own source in this pass. Backend is Intel's own SYCL/
oneAPI stack, not OpenCL/OpenVINO — an existence proof that CPU-tier MoE
offload works on Arc silicon, but on a different software stack than
arcint's OpenVINO plugin; none of its numbers transfer as OpenVINO/OpenCL
measurements without rerunning. No K-quant-native path found; low-bit
formats are INT4/FP4/INT8/FP8.

## CoX-MoE (closest single match for both (a) and (b))

arXiv:2605.17889, DAC'26. Targets both open questions at once. **(a)** argues
existing CPU-assist offload targets only decode-time GEMV ("insufficient…
to enable meaningful compute offloading") and leaves prefill's GEMM-heavy
shape "largely unexploited" — the same diagnosis as arcint's own
`grouped_fallbacks=40=num_layers` finding — and proposes an AMX-enabled
"coalescing-aware orchestration policy" batching prefill tokens across the
device/host split. **(b)** its "Expert-Aware Stratification" (EAS) is "a
lightweight data-driven pre-analysis framework that selects which experts
should be statically preloaded into VRAM before inference" — exactly
`partition-seeding`'s ask (a fixed calibration pass choosing the resident
set, not live routing). Caveat: PDF/abstract parsing in this pass could not
extract concrete hardware/model/prompt-length numbers or a public code link
— a mechanism match, not a numbers match, until the full text is read
directly. No Arc/OpenCL/SYCL/Vulkan; AMX generation unspecified.

## "Achieving Cloud-Grade SLOs..." (OSDI'26)

arXiv:2606.10493. **(a)** "stream-loading prefill" (SLP) overlaps loading of
not-yet-resident expert weights with compute of the current layer's resident
subset, similar in spirit to FreeToken's double-buffered layer prefetch —
claims 1,200 t/s prefill on dual-socket commodity CPUs + consumer GPU,
"distributed SLP" 1,800 t/s — the paper's own headline numbers (accepted,
not yet independently reproduced), and the hardware is dual-socket server
CPU, not a single desktop part, so not directly comparable to arcint's
reference cell. Useful mainly as a second independent confirmation that
streaming/overlapping expert load with device compute — rather than a hard
grouped-GEMM/host-loop split — is a live, competitive design point for the
same defect class as (a).

## What transfers to arcint

### static-partition-prefill (a)

The diagnosis is not novel: CoX-MoE and the KTransformers lineage
independently name the same failure mode arcint measured — CPU-assist
offload that only ever targeted decode-time GEMV chokes at prefill because
the batching unit is wrong, not because CPU-side compute is inherently too
slow. Concrete mechanism to imitate: batch all tokens routed to a given
expert across the whole prefill microbatch before dispatch (CoX-MoE's
coalescing-aware orchestration; ggml's `mul_mat_id` already does this at
expert granularity, just never combined with a GPU-resident subset in the
same layer). Arcint's own `moe_cpu_expert` kernel already batches per
expert — the missing piece is the *split*: route resident-expert tokens to
the existing device grouped-GEMM and the rest to the existing host kernel,
in the same layer, exactly what the campaign scopes. That is a plugin
change (`exec_prefill_onednn`'s refusal rule), not an engine change — none
of the surveyed systems replaced their serving engine, they added a
kernel-dispatch branch. What needs re-measuring, not assumed: every
multiplier above was measured on x86 with AMX or AVX-512; the Ryzen 5700X
host tier has AVX2 only, no AMX, so the CPU-side batched-GEMM ceiling on
this hardware is an open, unmeasured number — none of these claims can be
assumed to transfer even qualitatively without a fresh AVX2 measurement at
arcint's own reference cell.

### partition-seeding (b)

Fiddler and CoX-MoE's EAS are the direct precedents for "seed the resident
set from an offline calibration histogram, hold it fixed" — both compute
placement once, from a profiling corpus, not from live per-request routing,
structurally compatible with the "pure function of (seed, histogram table)"
invariant. What neither paper publishes: a corpus-size threshold at which
"hot" and "rarely-routed" separate from noise. Arcint's own M10 finding
(most of 7,360 experts sit at 0–2 routings on the one corpus tried) is not
answered by anything surveyed — a genuinely open measurement, not an
importable number. The larger group — PowerInfer, MoE-Infinity, HybriMoE,
ExpertFlow, FreeToken's LRU — are precedents for the lever this campaign
must explicitly *not* copy: all let residency track live per-request
activation, the category the history-independence invariant rules out for
arcint. Useful as citable negative controls for the design note. What needs
re-measuring: nothing here proves a frequency-informed static partition
beats arcint's current random `splitmix64` seed at decode — none of the
surveyed static-placement papers measured against a random-seed baseline
the way this campaign's gate requires; "does it move the number at all" is
still fully open.

### kquant-host-storage (c)

No surveyed system stores host-resident MoE experts as sub-4-bit K-quant
blocks decoded natively in place; the only sub-4-bit format found anywhere
is llama.cpp/ik_llama.cpp's own GGUF K-quant family (Q3_K/IQ3_XXS/Q2_K) —
the same byte counts arcint's own campaign doc already cites from that
lineage. No independent format or engine turned up to import instead.
ik_llama.cpp's `-rtr` is the one concrete, transferable warning: repacking
K-quant/i-quant blocks to a CPU-friendly interleaved layout at load speeds
CPU matmul but forecloses ever running that tensor on GPU again per the
project's own docs. Arcint's plan (K-quant native *beside* grouped-int4, not
a replacement) needs to either accept that same one-way trade for the
K-quant-resident set, or find a layout that stays device-movable —
undecided by anything surveyed, worth stating explicitly in the design
note. ipex-llm/FlashMoE and FreeToken both point the opposite direction from
arcint's decision: pre-merging into a native low-bit float bank format
(INT4/FP4/FTW) ahead of time, not K-quant block decode — useful context (the
road not taken, and why), not a mechanism to adopt. The core question this
campaign's own gate asks — does a native K-quant host kernel beat the
existing grouped-int4 host kernel at ~270/305 µs per expert — is not
answered, or even attempted, by anything found in this survey. That remains
arcint's own measurement to make, on its own hardware, with no borrowed
number to lean on.
