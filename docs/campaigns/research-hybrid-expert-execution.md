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

## Flash-Next serving comparables (added 2026-09-30)

Community reports for Qwen3.8-Flash-Next (125B-A6B plus the 51B n-gram
table). These are the reporters' own numbers, not measured here. Sources were
accessed 2026-09-30. The Reddit threads themselves were rate-limit-blocked, so
the rows come from the GitHub and Hugging Face write-ups of the same setups.

| setup | engine | expert placement | prefill | decode |
|---|---|---|---|---|
| 1× RTX 3090 24 GB (PCIe 4.0 x16), 77 GB DDR5-5600, 8c/16t, n-gram table on NVMe (syv-ai/HyperQwen #103) | exllamav3, EXL3 3.05 bpw, MTP ~83 % accept | 119 hot experts/layer on the card, 393/512 on 14 CPU threads with overlap | ~570 t/s (262k cold ≈ 7.5 min) | ~14–16 t/s with MTP |
| RTX 3060 12 GB, 48 GB DDR4-3200 (unsloth GGUF discussion #48) | llama.cpp fork, UD-Q2_K_XL, q4_0 KV, `-cmoe`, `--moe-cache-slots 42` | experts on CPU with a GPU expert cache | 40–60+ t/s | ~10+ t/s |
| 4× RTX 3090 24 GB, 31 GB RAM (tonyd2wild/Qwen38-Flash-Next-4x3090) | vLLM (W4A16 Marlin experts, FP8 n-gram table read from NVMe, 16 rows/token inside CUDA graphs); llama.cpp second lane | all experts on the cards | llama.cpp ~311 t/s | vLLM 55.8 t/s plain, 193 t/s with INT4 MTP draft (accept 0.66–0.92); llama.cpp 58.8 / 96–102 with MTP |
| **Ryzen 7 5700X (the dev host's CPU)**, 128 GB RAM, RX 9070 XT 16 GB, ROCm 7.2.4 (r/LocalLLM, "176B on 16 GB VRAM", 2026-09-02) | upstream llama.cpp, uncensored i1-Q4_K_S (104 GB), `--cpu-moe -ngl 99`, 8 decode threads pinned to physical cores, 16 batch threads, ubatch 512, f16 KV | **all experts on the CPU** | 192 t/s at a 26k prompt | **14.6 t/s** at 26k (15.8 empty, 12.2 at 132k unpinned); no MTP |
| Ryzen 9 5900X, 64 GB DDR4, RTX 3080 10 GB (a commenter on the same thread) | llama.cpp, 4.25 bpw, q4 KV, 99 % RAM used plus NVMe paging | experts on the CPU | 130 t/s at a 35k prompt | 13–15 t/s |
| Strix Halo, 128 GB unified, 70 W (r/LocalLLM benchmark thread) | Halogen 0.14 / gufo / CIRU | everything in unified memory | ~800–1,150 t/s at 32k–130k | 28–40 t/s with MTP (70–85 % accept), flat with depth |

Against arcint (`measured-here`, 2026-09-30, `docs/campaigns/qsa.md`): the
B60 24 GB with 52 GiB of host RAM, d48q8 native, ratio 75 + census128 +
30 GiB bank, 20,085-token prompt: **68.5 t/s prefill, 6.0 t/s decode, no
MTP**.

What transfers:
- **Same CPU, stock llama.cpp, every expert on the CPU: 14.6 t/s decode and
  192 t/s prefill, against arcint's 6.0 and 68.5 with a quarter of the experts
  on the card.** Our in-RAM tier bench was ~8.5 t/s, so the per-expert CPU
  path, not the hardware, is the decode gap. The thread's own explanation
  ("L3-resident experts") is wrong. Its MTP claim is narrated, not measured,
  but names a real risk for CPU-held experts: verify batches multiply tier
  work. A same-host llama.cpp run is the cheap bar to set before more tier
  machinery.
- The closest comparable (one 24 GB card plus host RAM) prefills about **8×
  faster**. Its host tier is the same shape (~77 % of experts on the CPU).
  With 83 % MTP acceptance its decode is ~2.5× ours; without MTP the gap is
  smaller but unmeasured here.
- Prefill is the largest gap: the per-expert CPU tier caps it. That is
  `prefill-expert-streaming`'s lever (card-side compute of host experts per
  prefill chunk).
- **MTP is worth 1.4–1.7× decode** in every report that has it; arcint
  exports no Flash-Next MTP head (ROMA R1).
- The n-gram table on NVMe through the page cache is the common practice;
  arcint's staged-rows route is equivalent.
- `--moe-cache-slots`-style GPU expert caches with CPU fallback are the
  llama.cpp-fork answer at 12 GB. Compare this with arcint's per-expert
  dispatch pool before building more tier machinery.

## Strata (added 2026-10-01) — paper read plus code read

Primary source: `~/src/Strata-ref` (checked out at commit `c499bd1`,
2026-10-01). The paper PDF `docs/paper/Strata-Paper.pdf` was read with
`pdftotext -layout`; the code was read directly at the paths below. Unlike the
web sections above, every row here is either **paper** or **code**, and the
two are separated where they disagree. Strata serves Qwen3.8-Flash-Next
(125B-A6B, 24,576 experts) from one 12 GB RTX 5070, 64 GB DDR5 and a six-core
Ryzen 5 7600 — the same model *shape* arcint serves, on different silicon.

### §3.2 One layer, three workers at once — the CUDA doorbell

- **paper.** The GPU runs the mixer and router and writes the chosen expert
  ids into a small block of pinned host memory (a "doorbell"). The CPU spins
  on that doorbell instead of waiting on the driver, splits the experts:
  resident→GPU, a share of misses→PCIe copy engine, the rest→CPU in place.
  The whole 48-layer pass is one captured CUDA graph per window size, so the
  host never makes a synchronizing driver call. One layer ≈ 0.6 ms at 4K
  (`paper` §3.2, Figure 2).
- **code.** `src/core/layer.cpp`: `doorbell_init` allocates `cudaHostAllocMapped`
  host buffers (`x_f`, `ids`, `weights`, `seq`, `flag`) and takes their device
  aliases with `cudaHostGetDevicePointer`; `moe_route` publishes through
  `doorbell_publish(...)` then `doorbell_ring(...)`; `doorbell_reset` zeroes
  `seq`/`flag`. The kernel side is `src/kernels/cuda/elementwise.cu`:
  `doorbell_publish_kernel` (volatile store + `__threadfence_system()`),
  `doorbell_ring_kernel` and `doorbell_wait_kernel`. `wait_flag_ge_kernel`
  (`src/kernels/cuda/verify_kernels.cu:426`) is the GPU-side spin:
  `while (*flag < value) strata_spin_pause(); __threadfence_system();`, called
  at `:520`. All of that is `code`.
- **evidence class: paper + code.**
- **what transfers.** The *discipline*, not the mechanism: a host-visible
  doorbell only works when the publishing store is fenced (`__threadfence_system`)
  — Strata's own history records that a memory-only spin never saw the datum
  until the fence was added (`elementwise.cu` comment block), which is the
  same class of bug arcint met in the served-prefill / direct-submission work.
  It does **not** grant lookahead: arcint's plugin exposes a layer's top-k ids
  only at that layer's own MoE hook (`code`: patches 0012/0017/0037/0044; the
  `nvme-direct-expert-tier` verdict, campaigns README — "the routing warning
  horizon is zero layers"), and Strata's doorbell publishes at the same
  instant. The doorbell removes the *driver-call* cost of the handoff, not the
  one-layer warning horizon.
- **what does not transfer.** CUDA graphs, mapped pinned memory, the copy
  engine, cuBLAS and the 0.6 ms/layer number are CUDA/RTX-specific. arcint
  runs OpenVINO on Intel cards behind the xe KMD and does not own the graph
  nodes; the plugin-side tier is the analogue. This is the same "plugin change,
  not an engine change" boundary the `static-partition-prefill` row already
  states.

### §3.3 Checking several tokens per pass — the MTP verify window

- **paper.** Qwen3.8-Flash-Next's own MTP head drafts up to three tokens; the
  last accepted token plus drafts go through all 48 layers as one *verify
  window*; drafts are kept while the MTP layer is ≥ 50 % confident; a wrong
  draft is rolled back. The output is exact — identical token for token to
  plain greedy decoding — tested with forced wrong drafts (`paper` §3.3, §6
  finding 3, §6.1).
- **evidence class: paper.**
- **what transfers.** The exactness claim is the same bar arcint's §3.4
  invariant and `agent-dense` MTP gate already hold, and it is a second
  independent engine reaching it. The structural warning transfers to
  `prefill-expert-streaming` and any tier work: a verify window routes *each*
  drafted token to its own experts, so window work multiplies host-tier
  misses (the paper's Finding 2 and Table 5 price this directly). arcint
  exports no Flash-Next MTP head yet (ROMA R1), so the speed half is not
  importable today.

### §3.4 An expert cache that follows the conversation

- **paper.** Routing is uneven. Startup fills VRAM from a profile recorded on
  *other* prompts; every four rounds up to 96 experts that the current
  conversation keeps asking for are swapped into the least-used slots, while
  the GPU is drafting. Hit rate: profile fill alone ≈ 0.50 at 4,500 slots on
  the 12 GB card, adaptive swapping raises it to ≈ 0.72 at 4K (`paper` §3.4,
  Figure 3; Finding 4).
- **code.** The swap policy lives in `src/program/generate.cpp`, not in
  `expert_cache.cpp`: `adapt_every = 4` (`:358`), `adapt_swaps = 96` (`:380`),
  a per-layer `usage` routing census, candidates with usage ≥ 2.0 vs victims
  ranked by least usage with a `+1.5` gain threshold, a global sort by gain
  truncated to `adapt_swaps`, and `resident_stage_swaps`/`commit_exchanges`
  (`:130`) staging the evicted blob back to RAM while the CPU computes `out`
  and admitting `in` only after the copy lands. `src/core/expert_cache.cpp`
  is the *storage*: `open` checks the allocation against free VRAM,
  `admit`/`slot_of` are the `(layer, expert) → slot or -1` table, `fill_slot*`
  and `verify_slot` (byte-compare of slot against host blob). The header
  states its own limits: eviction policy is deliberately absent there, and
  `set_per_layer_admission` exists because global arrival-order admission
  measured 2.97 % hit rate by filling inside the first position.
- **evidence class: paper + code.** `expert_cache.cpp` is `code`; the swap
  orchestration is `code` (`generate.cpp`); the numbers are `paper`.
- **what transfers.** Directly to the DESIGN §3.4 amendment of 2026-10-01 and
  to `expert-hot-set-lru` / `partition-seeding`: a deterministic-per-sequence
  adaptive tier is now the allowed mode, and Strata ships the mechanism —
  census, gain-ranked swap, deferred admission so nobody waits. The
  `verify_slot`/zeroed-slot discipline is also transferable: a wrong-residency
  table that answers a plausible token is the failure arcint has paid for most.
- **what does not transfer.** The hit-rate numbers (RTX 5070, 12 GB, Q2_0,
  24,576 experts) are not arcint's cards or artifact; the 96-swap/4-round
  cadence is a tuned constant; and the adaptive policy is exactly the
  history-dependent category arcint's pre-amendment invariant *excluded*.
  Note the code/paper split: an `expert_cache.cpp`-only reading would miss
  the adaptive tier entirely — it is in `generate.cpp`.

### §3.5 Reading the prompt: streaming experts to the GPU

- **paper.** Prompts are processed in chunks of 2,048 tokens; with that many
  tokens every expert is used many times, so the non-cached experts are copied
  from the pinned RAM arena over PCIe while the GPU computes the previous ones,
  dequantized to half precision and multiplied with cuBLAS; buffers are
  borrowed from the expert cache and refilled afterwards. Prompt speed flat to
  262K (`paper` §3.5).
- **evidence class: paper.**
- **what transfers.** The diagnosis transfers to `prefill-expert-streaming`:
  at prefill batch sizes the batching unit, not CPU FLOPs, is what a
  per-expert host loop gets wrong — batch the whole chunk's tokens per expert
  on the card. arcint measured that campaign as a loss (160 vs 90 s) with its
  pinned bank squeezed by the staging; Strata's design keeps *all* experts in
  one RAM arena and refills cache slots afterwards, which is the layout
  difference that makes its streaming pay. That comparison is worth a rerun
  before the campaign is closed for good.

### §6 findings 1, 4, 5, 7, 9, 10

| finding | disposition | evidence | transfers to arcint? |
|---|---|---|---|
| 1 — engine balanced; CPU waits ≈ 14 ms for GPU, GPU ≈ 13 ms for CPU; speeding one side cannot help by more than ~a third | closed | `paper` (§6, Table 5) | Yes as a bar. It is the reason `hybrid-expert-fetch` closed as a verdict: a deterministic PCIe split can hold §3.4 but a one-sided split is capped near a third, and arcint's own measured split lost. |
| 4 — adaptive cache beats static by a wide margin (0.50 → ~0.72) | closed | `paper` §3.4/§6; `code` `generate.cpp` | Yes — this is the measurement that justifies the 2026-10-01 §3.4 amendment and unblocks adaptive placement. The number itself does not transfer. |
| 5 — overlapping the two halves inside a layer did **not** pay: two token groups are exact but ~7 % slower (85.8 vs 91.8 t/s at 4K); dense weights read twice, fewer shared experts | closed | `paper` §6; `code` `verify.hpp` `set_split` (`--spec-split`, a non-default opt-in) and the `split_` comment "exact but slower" | Yes as a **negative control**. arcint's `hybrid-expert-fetch` split is the same shape; this is independent evidence that intra-layer token-group overlap is a loss unless it moves *different* work (e.g. copy engine), not a duplicate read. |
| 7 — i-quants limited by CPU arithmetic, not RAM: ~5 GB/s per core, 23–26 GB/s on six cores; decoding once for several tokens helps 2.0–2.4× but most experts serve one token | closed | `paper` §6 (Table 5's "CPU GB/s" and the bottleneck box) | Partly. The *diagnosis* transfers to `kquant-host-storage`: the host tier's per-expert CPU decode arithmetic is the cap, independent of RAM. The numbers are AVX-512 i-quant bytes/s on a Ryzen 5, not arcint's AVX2 host tier or its grouped-int4 kernel — not importable. |
| 9 — the overlap that paid: the copy engine. DMA the misses from the CPU thread when it plans the layer; GPU copy engine runs beside CPU and GPU work. 55 % of misses over PCIe for i-quants (arithmetic-bound), 20 % for Q2_0 (RAM-bound) | closed | `paper` §6; `code` `verify.hpp` `pcie_mode` / `fetch_dma`, the doorbell plan | Yes as the **discriminating variable**. arcint's `hybrid-expert-fetch` lost because its CPU tier was not left with free RAM bandwidth; Strata names exactly when the PCIe share pays (CPU compute-bound) and when it does not (CPU RAM-bound) — a testable condition for any future split. |
| 10 — refill without making anyone wait: evict the old expert at once, admit the new only when its copy lands; 91.7 → 94.4 t/s at 4K | closed | `paper` §6; `code` `generate.cpp` (`pending`, `apply_pending`, `host_res[out] = kNotResident` then `pending.emplace_back`, `adapt_live`) | Yes — the deferred-admission pattern is the concrete mechanism an arcint adaptive tier should copy, and it is compatible with the sequence-determinism law of the 2026-10-01 amendment. |

### §7 What could make it faster (projections, `paper` only)

The paper's own next steps: (a) keep conversation state between requests — the
largest single improvement for agents, because today every request re-reads its
whole prompt; (b) overlap **across** layers, not within one, by predicting the
next layer's experts; (c) cheaper i-quant CPU arithmetic (T-MAC-style lookup,
or converting hot experts to a CPU-friendly form at load); (d) a fused 8-bit
grouped prompt kernel; (e) temperature sampling via rejection in the verify
window; (f) free hardware settings. These are `paper` projections, not
measurements. **(a) maps directly to arcint's `kv-checkpoint-restore`
campaign** (backlog), which should treat Strata's "few hundred MB of recurrent
state + KV" figure as the shape of the prize, not a number. (b) is the
`hybrid-expert-fetch`/next-layer-prediction lever; (c) is
`kquant-host-storage`; (d) is the prompt-path half of
`prefill-expert-streaming`.

### What transfers to arcint's MoE tier, and what does not

- **Transfers:** (i) the adaptive tier as a *mode* — placement follows the
  conversation, bounded by sequence-determinism and the prefix-cache refusal
  (DESIGN §3.4 amendment, 2026-10-01); (ii) the swap mechanism — routing
  census, gain-ranked candidates/victims, deferred admission with no waiter
  (`generate.cpp`, `code`); (iii) the fence discipline for any host-visible
  handoff (`code`); (iv) `verify_slot`'s byte-compare and the zeroed-slot
  determinism; (v) Finding 9's condition for when a PCIe split pays, and
  Finding 5's negative control against intra-layer token-group overlap; (vi)
  Finding 1's one-third ceiling as a bar.
- **Does not transfer:** the RTX 5070 / Ryzen 5 numbers, the CUDA-graph and
  copy-engine mechanics, cuBLAS and ggml kernels, the 96/4 cadence, the
  24,576-expert hit-rate curve, and the `expert_cache.cpp`-only picture of
  the cache — the adaptive logic is in `generate.cpp`. Replacing arcint's
  serving engine is explicitly not the transfer; the surveyed systems added a
  kernel-dispatch branch (this file's own `static-partition-prefill` verdict,
  restated in `AGENTS.md`).
