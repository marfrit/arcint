# research: hybrid GPU/CPU expert execution — prior art

Web survey of 2026-09-05 (access date for every web source below), the
Flash-Next comparables of 2026-09-30, and the Strata source read of
2026-10-01. The web survey was written for (a) the grouped prefill with a
CPU tier, (b) the choice of resident experts, (c) sub-4-bit blocks on the
host tier; the campaigns that carry those levers now are named in each
section.

## "ninfer" and "freetoken" — what they actually are

Both names resolve to real, currently-shipping projects — not fabrications —
but neither is what "ahead of llama.cpp" suggests for arcint's problem.

**ninfer** (github.com/Neroued/ninfer, Apache-2.0) is a from-scratch C++/CUDA
engine for a *closed set* of dense Qwen checkpoints (Qwen3.6-27B, Qwen3.8-27B,
Qwen3.6-35B-A3B) on a single NVIDIA RTX 5090 (community forks retarget
4090/3090). No MoE, no CPU offload, no host compute tier anywhere in the
README — it is single-GPU, CUDA-only, and does not run anything that
doesn't fit the card. Not on topic for expert offload.

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
inter-layer prefetcher and a score-based cache: residency follows a running
activation score. Reports 1.33× prefill / 1.70× decode vs. plain KTransformers on three
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
profiling is a one-time calibration, not per-request adaptation. **(a)** explicitly built to be good at both single-batch and
long-prefill scenarios by design, evidence a residency-aware batched prefill
is achievable without abandoning the whole layer to fallback. Claimed 8.2×
(Quadro RTX 6000) / 10.1× (L4) single-batch latency speedup vs. its own
baselines — paper numbers, not third-party reproduced. CUDA only; no Arc/
OpenCL/SYCL/Vulkan; standard int4/int8 group quant, no K-quant.

## MoE-Infinity

EfficientMoE/MoE-Infinity, Apache-2.0 (arXiv:2401.14361). **(b)** "activation-
aware" expert cache: traces per-request sparse activation, uses the trace to
drive prefetch/eviction — an online scheme, same category as
HybriMoE/PowerInfer. **(a)** not its focus. Reports 3.1–16.7× per-token latency vs.
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
history. ggml's CPU `mul_mat_id` kernel already batches per assigned
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

### Prefill with a CPU tier (a)

CoX-MoE and the KTransformers lineage name the batching unit as what makes
CPU-assist offload choke at prefill: batch all tokens routed to an expert
across the prefill microbatch before dispatch (CoX-MoE's coalescing-aware
orchestration; ggml's `mul_mat_id`). Resident experts on the device and the
rest on the host in the same layer is patch 0037 (`static-partition-prefill`).
The prefill rate lever the expert engines use is computing every expert on
the GPU and streaming the non-resident ones (the OSDI'26 stream-loading
prefill above; FreeToken and Strata, below): `prefill-expert-streaming`. The
surveyed systems added a kernel-dispatch branch, not a new serving engine.
The CPU-side multipliers above were measured with AMX or AVX-512; the dev
host's CPU has AVX2.

### Choosing the resident experts (b)

Fiddler and CoX-MoE's EAS seed the resident set from an offline calibration
histogram and hold it fixed; neither publishes a corpus-size threshold at
which "hot" separates from noise. PowerInfer, MoE-Infinity, HybriMoE,
ExpertFlow and FreeToken's LRU let residency track live activation; that is
the mode `expert-hot-set-lru` builds, after Strata (section below), with DESIGN
§3.4 amended on 2026-10-01 to allow it.

### Sub-4-bit blocks on the host tier (c)

llama.cpp's CPU backend dots GGUF K-quant/i-quant blocks (Q3_K, IQ3_XXS,
Q2_K, IQ4_NL) in place, one token at a time. Strata's CPU tier (added
2026-10-01, below) decodes the same i-quant blocks natively on AVX2 several
tokens at once (`code`: `~/src/Strata-ref/src/kernels/cpu/iq_avx2.cpp`); that
is the reference for `kquant-host-storage`. ik_llama.cpp's `-rtr` repacks
K-quant/i-quant blocks into a CPU-interleaved layout at load, and its docs
note that a repacked tensor then always runs on the CPU. ipex-llm/FlashMoE and
FreeToken pre-merge experts into a native low-bit bank format (INT4/FP4/FTW)
ahead of time.

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
- **MTP is worth 1.4–1.7× decode** in every report that has it. The head is
  in the BF16 checkpoint (Strata's `tools/mtp_fetch.py` reads it); campaign
  `mtp-cycle-wall`.
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


### §3.2 One layer, three workers at once — the doorbell

- **paper.** The GPU runs the mixer and router and writes the chosen expert
  ids into a small block of pinned host memory (a "doorbell"). The CPU spins
  on that doorbell instead of waiting on the driver and splits the experts:
  resident → GPU, a share of misses → PCIe copy engine, the rest → CPU in
  place. The whole 48-layer pass is one captured CUDA graph per window size,
  so the host never makes a synchronising driver call. One layer ≈ 0.6 ms at
  4K (`paper` §3.2, Figure 2).
- **code.** `src/core/layer.cpp`: `doorbell_init` allocates
  `cudaHostAllocMapped` host buffers (`x_f`, `ids`, `weights`, `seq`, `flag`)
  and takes their device aliases; `moe_route` publishes through
  `doorbell_publish(...)` then `doorbell_ring(...)`. The kernels are in
  `src/kernels/cuda/elementwise.cu` (`doorbell_publish_kernel`: volatile
  stores + `__threadfence_system()`; `doorbell_ring_kernel` increments the
  sequence in memory; `doorbell_wait_kernel`); `wait_flag_ge_kernel`
  (`src/kernels/cuda/verify_kernels.cu:426`) is the GPU-side spin. The
  comment block in `elementwise.cu` records that a memory-only spin never saw
  the datum until the fence was added.
- **evidence class: paper + code.**
- **what transfers.** The mechanism: a fenced publish into mapped host
  memory, a host spin, a device wait on a host flag, and the pass recorded
  once. OpenCL candidates on arcint's stack: device-accessible USM host
  allocations and `cl_khr_command_buffer`. Campaign: `tier-handoff-doorbell`.

### §3.3 Checking several tokens per pass — the MTP verify window

- **paper.** Qwen3.8-Flash-Next's own MTP head drafts up to three tokens; the
  last accepted token plus drafts go through all 48 layers as one *verify
  window*; drafts are kept while the MTP layer is ≥ 50 % confident; a wrong
  draft is rolled back. The output is identical token for token to plain
  greedy decoding, tested with forced wrong drafts (`paper` §3.3, §6 finding
  3, §6.1).
- **code.** `src/core/mtp.cpp` (`MtpDrafter::draft`, the chain while the last
  draft's probability ≥ `min_p`); `tools/mtp_fetch.py` reads the 31 `mtp.*`
  tensors from the BF16 checkpoint by HTTP range.
- **evidence class: paper + code.**
- **what transfers.** The multi-draft chain and the head fetch; campaign
  `mtp-cycle-wall`. A verify window routes each drafted token to its own
  experts (`paper` Finding 2, Table 5); Strata's CPU kernels decode each block
  once for the whole window (`src/kernels/cpu/iq_avx2.cpp`).

### §3.4 An expert cache that follows the conversation

- **paper.** Routing is uneven. Startup fills VRAM from a profile recorded on
  *other* prompts; every four rounds up to 96 experts that the current
  conversation keeps asking for are swapped into the least-used slots, while
  the GPU is drafting. Hit rate: profile fill alone ≈ 0.50 at 4,500 slots on
  the 12 GB card, adaptive swapping raises it to ≈ 0.72 at 4K (`paper` §3.4,
  Figure 3; Finding 4).
- **code.** The swap policy lives in `src/program/generate.cpp`, not in
  `expert_cache.cpp`: `adapt_every = 4` (`:358`), `adapt_swaps = 96` (`:380`),
  a per-layer `usage` routing census decayed ×0.7 per round, candidates with
  usage ≥ 2.0 against victims ranked by least usage with a `+1.5` gain
  threshold, a global sort by gain truncated to `adapt_swaps`
  (`:4395-4478`), and `resident_stage_swaps`/`commit_exchanges` staging the
  evicted blob back to RAM while the CPU computes it and admitting the new
  one only after its copy lands. `src/core/expert_cache.cpp` is the
  *storage*: `open` checks the allocation against free VRAM, `admit`/`slot_of`
  are the `(layer, expert) → slot or -1` table, `fill_slot*` and
  `verify_slot` (byte-compare of slot against host blob).
  `set_per_layer_admission` exists because global arrival-order admission
  measured 2.97 % hit rate by filling inside the first position. The start
  profile ranks every pair (`tools/make_profile.py`).
- **evidence class: paper + code.**
- **what transfers.** The adaptive tier as a mode, the swap mechanism
  (census, gain-ranked candidates and victims, deferred admission with no
  waiter), `verify_slot`'s byte-compare. Campaign: `expert-hot-set-lru`; DESIGN
  §3.4 Amendments 1–2 (2026-10-01). The hit rates are the RTX 5070's with
  Q2_0; 96 swaps every 4 rounds is Strata's tuned default.

### §3.5 Reading the prompt: streaming experts to the GPU

- **paper.** Prompts are processed in chunks of 2,048 tokens; every expert is
  used many times, so the non-cached experts are copied from the pinned RAM
  arena over PCIe while the GPU computes the previous ones; buffers are
  borrowed from the expert cache and refilled afterwards. Prompt speed flat to
  262K (`paper` §3.5).
- **code** (wins where it differs from the paper). `src/prefill/prefill.cpp:71-104`:
  every non-resident expert streams through a ring from a chunk of 1,024 on
  (`stream_all_min()`), 384 slots when ≥ 90 % of experts are pinned; quantised
  MMQ expert kernels (`:458-486`); `--prefill auto` chunks up to 8,192
  (`bench/results/2026-09-28-prefill-speed/README.md`, 572 → 1,290 t/s at
  32K for Q2_0).
- **evidence class: paper + code.**
- **what transfers.** The whole design; campaign `prefill-expert-streaming`.

### §6 findings 1, 4, 5, 7, 9, 10

| finding | disposition | evidence | transfers to arcint? |
|---|---|---|---|
| 1 — engine balanced; CPU waits ≈ 14 ms for GPU, GPU ≈ 13 ms for CPU; speeding one side cannot help by more than ~a third | closed | `paper` (§6, Table 5) | Yes, as the bar for any one-sided speed-up. |
| 4 — adaptive cache beats static by a wide margin (0.50 → ~0.72) | closed | `paper` §3.4/§6; `code` `generate.cpp` | Yes: `expert-hot-set-lru`. |
| 5 — splitting a layer's tokens into two groups to overlap its halves is exact but ~7 % slower (85.8 vs 91.8 t/s at 4K): dense weights read twice, fewer shared experts | closed | `paper` §6; `code` `verify.hpp` `set_split` (`--spec-split`, non-default) | Yes, as Strata's own measurement of that split; its paying overlap is the copy engine (Finding 9). |
| 7 — i-quants limited by CPU arithmetic, not RAM: ~5 GB/s per core, 23–26 GB/s on six cores; decoding once for several tokens helps 2.0–2.4× | closed | `paper` §6 (Table 5); `code` `src/kernels/cpu/iq_avx2.cpp` | Yes: the AVX2 multi-token kernel is the reference for `kquant-host-storage`. |
| 9 — the overlap that paid: the copy engine. DMA the misses from the CPU thread when it plans the layer; the copy engine runs beside CPU and GPU work. 55 % of misses over PCIe for i-quants (arithmetic-bound), 20 % for Q2_0 (RAM-bound) | closed | `paper` §6; `code` `verify.hpp` `pcie_mode`, `expert_source.cpp:1614-1631` | Yes: `hybrid-expert-fetch`. The share follows whether the CPU tier is arithmetic- or RAM-bound. |
| 10 — refill without making anyone wait: evict the old expert at once, admit the new one when its copy lands; 91.7 → 94.4 t/s at 4K | closed | `paper` §6; `code` `generate.cpp` (`pending`, `apply_pending`, `host_res[out] = kNotResident`) | Yes: the admission pattern for `expert-hot-set-lru`. |

### §7 What could make it faster (projections, `paper` only)

The paper's own next steps: (a) keep conversation state between requests —
the largest single improvement for agents, because today every request
re-reads its whole prompt; (b) overlap **across** layers, not within one, by
predicting the next layer's experts; (c) cheaper i-quant CPU arithmetic
(T-MAC-style lookup, or converting hot experts to a CPU-friendly form at
load); (d) a fused 8-bit grouped prompt kernel; (e) temperature sampling via
rejection in the verify window; (f) free hardware settings. These are `paper`
projections, not measurements. (a) is `kv-checkpoint-restore`; (b) relates
to `hybrid-expert-fetch`; (c) is `kquant-host-storage`; (d) is the
prompt-path half of `prefill-expert-streaming`.

### What transfers to arcint's MoE tier

The adaptive tier and its swap mechanism (`expert-hot-set-lru`), the
doorbell and recorded pass (`tier-handoff-doorbell`), the PCIe share of the
misses (`hybrid-expert-fetch`), prefill streaming through a slot ring
(`prefill-expert-streaming`), multi-token CPU kernels (`kquant-host-storage`),
multi-draft MTP (`mtp-cycle-wall`), conversation state between requests
(`kv-checkpoint-restore`), and `verify_slot`'s byte-compare for any copied
expert. Numbers are the RTX 5070's and do not carry over; the mechanisms do.
None of these replaces arcint's serving engine: the surveyed systems added
kernel-dispatch branches and memory paths beside their engines.
