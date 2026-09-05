# research-speculative-cycle — how other engines keep MTP/spec-decode cycles cheap at depth

Feeds `mtp-cycle-wall.md`: on the dense 27B agent at 77,134 tokens, one MTP
cycle (propose = MTP-layer forward, verify = 2-token main-model forward,
plus bookkeeping) costs ≈390 ms against a ≈130 ms break-even, with the MTP
head capped at one predicted token/step (a 2-token ceiling) even at 90.8%
acceptance. Surveyed: vLLM, SGLang, TensorRT-LLM, llama.cpp, the academic
EAGLE/Medusa lineage, and Intel's stacks, all against the same
propose-cost-vs-verify-amortization problem. Access dates 2026-09-05.

## vLLM

Apache-2.0. MTP (DeepSeek-V3/R1, Qwen3.5, Gemma-4) is wired via
`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`, reusing
the checkpoint's own MTP layer(s). When `N` exceeds the native MTP depth,
vLLM **chains additional MTP forward passes** to draft more tokens before
one verify — raising tokens/cycle, not cutting cycle cost, since each
chained step pays its own forward. EAGLE-family verify batches `K+1` tokens
through the target in one forward under a size-matched CUDA graph; propose
and verify stay separate graphs, not fused. Two adaptive mechanisms are
real, merged code: "Dynamic Speculative Decoding" (length by QPS) and DSpark
adaptive verification (PR #47808) — a confidence head scores draft-token
survival, a per-step token budget from profiled cost tables applies on-GPU
with no host readback; needs FULL varlen-decode CUDA graphs, currently only
DeepSeek's SM100 sparse-attention backends qualify, not portable as-is. Best
benchmark (AMD MI300X/MI355X blog, methodology-backed) tops out at 32,768
context: native MTP 1.7–2.7x, DFlash-style block/parallel drafters 2.3–2.9x,
ahead of sequential MTP/EAGLE at depth; per-position acceptance decays hard
(85–95% at position 1, 33–50% at position 7) for sequential methods, which
block drafters avoid. No public number isolates
cycle cost at ~77k — an open gap. Intel Arc/XPU: n-gram/EAGLE/EAGLE3 listed
for Arc B-series (incl. B60, 24 GB), no Arc spec-decode numbers, no XPU
equivalent of the CUDA-graph verify-batching (looks CUDA/HIP-specific).
Sources: docs.vllm.ai/en/latest/features/speculative_decoding/{,/mtp/},
vllm.ai/blog/2026-08-14-dspark-adaptive-verification,
vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus,
vllm.ai/blog/2025-11-11-intel-arc-pro-b.

## SGLang

Apache-2.0. EAGLE-2/3 run as tree drafters: `speculative_num_steps` (depth)
× `speculative_eagle_topk` (branching) sizes the tree, verified in one pass
under a **prefix-tree attention mask**; `speculative_num_draft_tokens` caps
tree width. Tree masking historically forced eager-mode attention (blocking
fast kernels); a fused masked-verify kernel is implied, not confirmed
shipped. CUDA-graph handling precaptures one graph per supported verify
length, dispatching with no recapture — the mechanism making variable-length
verify cheap on NVIDIA hardware (LMSYS blog, 2026-08-17). "Adaptive
Speculative Decoding" tracks per-request accepted length, EMA-smooths it,
derives `target_steps ≈ round(ema_accept_len)+1` with hysteresis/warmup
guards, but only at `topk=1`. DSpark (PR #30261, 2026-07-06) sizes the
verify window per-request from the draft model's own confidence over a
ragged CUDA-graphed shape — the closest shipped mechanism to "cap by
predicted cost, not fixed depth." Best clean benchmark (Llama-3.1-8B,
MT-Bench, one H100, short context, marketing-adjacent): EAGLE-2 +54%,
EAGLE-3 +136%. Low-confidence third-party claims: EAGLE-3 acceptance
degrades past ~32k context, disabled above batch 32 in some deployments;
`LongSpec` (2502.17421, not cross-checked) targets verify-scales-with-context
directly, 1.5–2.2x to 128k on Llama-2/Qwen. Intel Arc: no mention in
SGLang's docs; several backends are CUDA-only, and the per-length CUDA-graph
mechanism is CUDA-specific by construction. Sources:
docs.sglang.io/advanced_features/{speculative_decoding,adaptive_speculative_decoding}.html,
lmsys.org/blog/2026-08-17-advanced-cuda-graph, arxiv.org/pdf/2502.17421.

## TensorRT-LLM

Apache-2.0 (per-file SPDX). MTP is DeepSeek-only: `num_nextn_predict_layers`
must equal `max_draft_len` — one chained MTP module per draft token, not one
head rerun N times; DeepSeek-R1's B200 config chains 3 MTP layers. "Relaxed
acceptance" (reasoning "thinking" phase only) swaps exact-token-match for a
candidate set: a token survives if `log P(top1) − log P(t) ≤ relaxed_delta`
among top `relaxed_topk`; NVIDIA's "limited"/"slight" quality-impact claim is
vendor-asserted, not quantified — marketing, as is a vendor figure of 80–90%
second-token MTP acceptance / "1.8x TPS" with no context or hardware named.
The whole drafting loop is captured as **a single CUDA graph** — closer to
fused propose+verify than any other engine here — but entirely
CUDA/Blackwell-specific; Intel Arc applicability is effectively none — the
module-chaining idea transfers as a concept, not the implementation. Sources:
nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html,
github.com/NVIDIA/TensorRT-LLM (blog2_DeepSeek_R1_MTP doc, LICENSE).

## llama.cpp and speculators

llama.cpp (MIT): classic draft-model decoding (`--model-draft`) — draft
proposes autoregressively, target verifies the batch in one forward.
KV-cache sharing isn't confirmed real: issue #25345 shows unified/paged KV
cache silently disables spec-decode, implying separate KV state. Parameters:
`--spec-draft-n-max`/`-n-min`, per-draft-model KV precision independent of
target, and `--spec-type` covering `draft-simple`, `draft-eagle3`,
`draft-mtp`, plus n-gram/prompt-lookup variants. An unverified community
fork claims an adaptive draft-length flag with large speedups on a
non-mainline backend — not upstream. No tokens/sec numbers found; the docs'
acceptance log lines (0.58, 0.70) carry no model/hardware/context, so are
illustrative only. Source: github.com/ggml-org/llama.cpp docs/speculative.md,
issue #25345. **speculators** (Neural Magic/vLLM ecosystem, vLLM-exclusive)
trains draft models, currently EAGLE-3-focused (v0.3.0, 2025-12-13) from
three verifier-layer hidden states plus token ids; production support for
Llama 3.1/3.2/3.3, Qwen3, GPT-OSS. No cycle-cost/long-context numbers
published. Source: vllm.ai/blog/2025-12-13-speculators-v030.

## Academic drafters: EAGLE-2/3, HASS, Medusa/Medusa-2, Hydra

All Apache-2.0/research code, none evaluated past short-context benchmarks
(MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DM) — **none report a long-context
(32k+) degradation study or a ms-level propose/verify split**, only
hardware-independent acceptance-length ("τ") and an aggregate speedup called
hardware-dependent without decomposing it. EAGLE-2 (2406.16858): a
context-aware dynamic draft tree from the draft head's own confidence,
3.05–4.26x, τ≈4.0–5.5. EAGLE-3 (2503.01840): drops feature-prediction
training for direct token prediction plus multi-layer feature fusion, up to
6.5x, ~1.4x over EAGLE-2. HASS (2408.15766): a training-time-only fix
(ranking-distillation loss, context-aligned training on self-generated
prefixes), **zero added inference cost**, same cycle shape as EAGLE-2,
2.81–4.05x from better acceptance alone. Medusa/Medusa-2 (2401.10774) is
structurally closest to "more than one token per MTP-style forward": K
lightweight FFN heads on the target's own final hidden state predict
t+1..t+K in the **same forward pass** as normal decoding — no separate
propose step, verified via one tree-masked forward, 2.2–3.6x. This
generalizes the MTP-head limit (1 head → 1 token) to K heads → K draft
positions per cycle without K sequential forward passes, but needs K
*trained* heads; re-running one head K times still pays K× its forward cost
per cycle, unlike Medusa. Hydra (2402.05109) fixes Medusa's
head-independence with sequentially-dependent heads, up to 1.3x over
Medusa, same verify mechanism and cost class. (arXiv IDs, fetched 2026-09-05.)

## Intel-specific

OpenVINO GenAI (Apache-2.0): classic two-model draft+verify via `LLMPipeline`
(separate target/draft models, independent device assignment), plus, as of
2026.0, an EAGLE-3 integration validated on Qwen3-8B for CPU and GPU — no
DeepSeek-style native MTP-head mechanism found. Vendor number on Core Ultra 7
268V + Arc 140V iGPU: ~1.3–1.4x (marketing-adjacent, no breakdown given).
The one directly relevant, independent data point: a community GitHub
discussion (#36484), GPU-target + NPU-draft on the same Lunar Lake iGPU,
frames its failure as **draft cost exceeding the verify window it was meant
to amortize** (drafting ~3 tokens costs ~150 ms, already more than the
~63 ms target verify) — the same shape as this campaign's own ≈390 ms vs
≈130 ms finding, on comparable Intel GPU hardware, independently measured.
ipex-llm (Apache-2.0) does self-speculative decoding: auto low-bit (INT4)
quantizes the same model as its own draft, verifies against the original
weights; vendor claim ~30% latency reduction on Intel GPU, no breakdown, and
a community issue (#13110) notes difficulty reproducing Intel's own
advertised Arc throughput even for plain inference — a caution flag on
ipex-llm numbers generally. Sources: github.com/openvinotoolkit/openvino.genai,
github.com/openvinotoolkit/openvino/discussions/36484,
github.com/intel/ipex-llm, github.com/intel/ipex-llm/issues/13110.

## What transfers to arcint

Three levers, in likely-first order:

1. **Multi-step MTP drafting per cycle, capped by measured cycle time** — the
   mechanism every surveyed engine implements in some form (vLLM's MTP
   chaining, TensorRT-LLM's chained modules, SGLang's/vLLM's adaptive
   draft-length work), and closest to `mtp-cycle-wall.md`'s cut-vs-verdict
   gate: re-run the MTP head to raise tokens/cycle above the 2-token
   ceiling, only while profiled per-step cost keeps the cycle under its own
   break-even (an `ARCINT_PROFILE_CYCLE`-driven cap, not a fixed depth).
   Plugin: nothing new, the MTP forward exists — a scheduling change around
   the reservation/profile sites in `backend_ov.cpp`. Engine: none.
2. **Fusing propose into the verify forward** — TensorRT-LLM's single-graph
   drafting loop and vLLM's size-matched verify graphs are the model; on
   OpenVINO, a fused MTP-forward-then-main-verify subgraph instead of two
   dispatches with bookkeeping between. The one lever needing new plugin
   work — the community NPU-draft finding suggests the payoff is real
   (cross-call overhead, not raw FLOPs, dominates at this scale), but nobody
   has shipped it for OpenVINO yet.
3. **Cap draft length by measured cycle time, not a static value** —
   SGLang's adaptive EMA scheme and DSpark's confidence-sized verify window
   are the cleanest reference designs: track accepted length and cycle
   latency, adjust the next cycle's draft depth, with hysteresis against
   oscillation. Engine: bookkeeping only (arcint already profiles via
   `ARCINT_PROFILE_CYCLE`); plugin: nothing.

First profile for this campaign: lever 1 — no plugin change, answers
`mtp-cycle-wall.md`'s "cycle cut" branch directly, with prior art at both
the vendor level (TensorRT-LLM, vLLM) and the independent-measurement level
(the OpenVINO NPU-draft discussion) confirming this failure mode isn't
stack-specific. Lever 2 is the higher-ceiling, higher-risk follow-up if
lever 1's cap still leaves the cycle above break-even at 77k.
