# ligence

A deliberately narrow LLM inference engine for Intel Arc GPUs.

ligence runs exactly three model families on exactly two cards, and tries to do
that better than the general-purpose engines do. The inspiration is
[NInfer](https://github.com/Neroued/ninfer) — a from-scratch engine that
supports two checkpoints on one GPU and beats every generalist on that pair —
translated to Intel: instead of hand-written CUDA, the kernel work is delegated
to OpenVINO's compiler stack, which already emits good Xe code. What ligence
owns is everything around the compute graph: the serving loop, the scheduler,
the KV and recurrent-state memory, prefix caching, and speculative decoding.

## Scope

**Models** (these, and nothing else):

| model | architecture | quant |
|---|---|---|
| Qwen3.6-35B-A3B | hybrid GDN + attention, MoE | q4, q8 |
| Qwen3.6-27B-A3B-Coder | hybrid GDN + attention, MoE | q4, q8 |
| Qwen3.8-27B | hybrid GDN + attention, dense | q4, q8 |

**Hardware** (these, and nothing else):

| card | VRAM | notes |
|---|---|---|
| Intel Arc A770 (DG2/Alchemist) | 16 GB | ~560 GB/s |
| Intel Arc Pro B60 (BMG G21) | 24 GB (~22.7 usable) | ~456 GB/s |

All three target models are hybrids: most layers use linear attention
(GatedDeltaNet), a minority use full attention. That single fact drives most of
the design — see [DESIGN.md](DESIGN.md).

## Features

- OpenAI-compatible **`/v1/chat/completions`** and **`/v1/completions`** over HTTP.
- **`/health`** (liveness, model-loaded, queue depth) and **`/props`**
  (model metadata, context length, quant, cache configuration, build info).
- **Prefix caching** for the full hybrid state — attention KV pages *and* GDN
  recurrent state — with a hard invariant: greedy output with a warm cache is
  byte-identical to greedy output with a cold one. Cache reuse that changes the
  answer is treated as a bug, not a documented quirk.
- **Paged KV cache** for the attention layers; block-aligned GDN state
  checkpoints for the linear-attention layers.
- **MTP speculative decoding** using the models' native multi-token-prediction
  heads where the checkpoint ships one (Qwen3.8), with a hook for external
  drafters.
- **Flash-attention-style fused SDPA** on the full-attention layers, where the
  OpenVINO kernel library provides it for the target Xe generation.
- **Tool-call parsing**: native Qwen tool-call output is returned as structured
  `tool_calls` (OpenAI-compatible). Parsed, never executed; requests without
  declared tools get raw text untouched.
- **Tokenizer and chat template come from the model artifact**, never from the
  server — the template hash is part of the model allowlist.
- **Model-aware sampling defaults** from the model card, per allowlist entry;
  explicit request fields always win.
- **Hard context-overflow rejection**: HTTP 400 with the numbers. No silent
  truncation, no context shift — on hybrid GDN models a shift is not even
  honestly implementable (the recurrent state cannot un-see past tokens).
  History management belongs to the client.
- **Console state output** in the llama.cpp tradition: per-request timing lines
  (prefill/decode token counts and rates), slot states, memory-map printouts at
  startup, cache hit statistics. stderr is the dashboard.

## Non-goals

- No web UI, no GUI, no gradio, no metrics dashboard. Console and HTTP JSON only.
- No model zoo. A checkpoint outside the table above is rejected at load time.
- No multi-GPU, no pipeline or tensor parallelism. One process, one card.
- No training, no LoRA, no quantization tooling (artifacts are produced offline).
- No Windows.

## Status

Design phase. See [DESIGN.md](DESIGN.md) for the architecture and milestones,
[llm.txt](llm.txt) for a machine-readable summary.

## Why not just use …

- **OpenVINO GenAI**: its continuous-batching path (required for prefix
  caching) produces measurably different greedy output than its stateful path,
  the equivalence test in its own CI has been skipped since 2025-02, and hybrid
  state is checkpointed at coarse intervals sized for memory rather than
  correctness. ligence keeps the OV *compiler* and replaces the pipeline layer.
- **vLLM**: no usable Intel path for these hybrids (the XPU build lags, and
  Arc is not a first-class target). The paged-KV and prefix-cache ideas are
  taken; the engine is not.
- **llama.cpp**: excellent console ergonomics and a sane server, but Vulkan on
  Battlemage is unoptimized (measured 7.9 tok/s dense where OpenVINO does 60)
  and SYCL is broken under the xe KMD. The ergonomics are taken; the backend
  situation is why ligence exists.
