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

**Models** (these, and nothing else). Geometry read off the IRs on
2026-08-28 and kept in [models/allowlist-raw.json](models/allowlist-raw.json);
all three are `full_attention_interval = 4`, 262144 context, and share one
tokenizer:

| model | architecture | layers (GDN + attn) | experts | weights | quant |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | hybrid GDN + attention, MoE | 40 (30 + 10) | 256 | 17.4 GiB | q4, q8 |
| Qwen3.6-27B-A3B-Coder | hybrid GDN + attention, MoE | 40 (30 + 10) | 184, pruned from 256 | 12.8 GiB | q4, q8 |
| Qwen3.8-27B | hybrid GDN + attention, dense | 64 (48 + 16) | dense | 13.4 GiB | q4, q8 |

**Hardware** (these, and nothing else):

| card | VRAM | notes |
|---|---|---|
| Intel Arc A770 (DG2/Alchemist) | 16 GB | ~560 GB/s |
| Intel Arc Pro B60 (BMG G21) | 24 GB (~22.7 usable) | ~456 GB/s |

All three target models are hybrids: most layers use linear attention
(GatedDeltaNet), a minority use full attention. That single fact drives most of
the design — see [DESIGN.md](DESIGN.md).

## Features

Marked **[M0]** where it works today against the stub backend, and **[planned]**
where the design exists but the code does not. Nothing marked [M0] has been
run against a model yet — there is no OpenVINO in the build until M1.

- **[M0]** OpenAI-compatible **`/v1/chat/completions`** and **`/v1/completions`** over HTTP.
- **[M0]** **`/health`** (liveness, model-loaded, queue depth) and **`/props`**
  (model metadata, context length, quant, cache configuration, build info).
- **[planned]** **Prefix caching** for the full hybrid state — attention KV pages *and* GDN
  recurrent state — with a hard invariant: greedy output with a warm cache is
  byte-identical to greedy output with a cold one. Cache reuse that changes the
  answer is treated as a bug, not a documented quirk.
- **[planned]** **Paged KV cache** for the attention layers; block-aligned GDN state
  checkpoints for the linear-attention layers.
- **[planned]** **MTP speculative decoding** using the models' native multi-token-prediction
  heads where the checkpoint ships one (Qwen3.8), with a hook for external
  drafters.
- **[planned]** **Flash-attention-style fused SDPA** on the full-attention layers, where the
  OpenVINO kernel library provides it for the target Xe generation.
- **[M0]** **Tool-call parsing**: native Qwen tool-call output is returned as structured
  `tool_calls` (OpenAI-compatible). Parsed, never executed; requests without
  declared tools get raw text untouched.
- **[planned]** **Tokenizer and chat template come from the model artifact**, never from the
  server — the template hash is part of the model allowlist. The hashes are
  pinned already; M0 stands in a reversible splitter and a ChatML renderer that
  M1 replaces with the artifact's own, and both are labelled as stubs in the
  code.
- **[M0]** **Model-aware sampling defaults** from the model card, per allowlist entry;
  explicit request fields always win.
- **[M0]** **Hard context-overflow rejection**: HTTP 400 with the numbers. No silent
  truncation, no context shift — on hybrid GDN models a shift is not even
  honestly implementable (the recurrent state cannot un-see past tokens).
  History management belongs to the client.
- **[M0]** **Streaming that does not corrupt anything**: a multi-byte code
  point is never split across two SSE chunks, a stop sequence never leaks out
  one fragment at a time, and tool-call syntax never reaches a content delta.
- **[M0]** **Cancellation**: a dropped client aborts the request at the next
  scheduler boundary and frees its slot.
- **[M0]** **Console state output** in the llama.cpp tradition: per-request timing lines
  (prefill/decode token counts and rates), slot states, memory-map printouts at
  startup, cache hit statistics. stderr is the dashboard.

## Non-goals

- No web UI, no GUI, no gradio, no metrics dashboard. Console and HTTP JSON only.
- No model zoo. A checkpoint outside the table above is rejected at load time.
- No multi-GPU, no pipeline or tensor parallelism. One process, one card.
- No training, no LoRA, no quantization tooling (artifacts are produced offline).
- No Windows.

## Status

Serving the real models on Intel Arc. Measured on a B60 with the 27B coder q4:
**51.4 t/s decode, up to 2247 t/s prefill**, and all three target models reach
**10/10** on the fleet's code-generation harness.

| milestone | state |
|---|---|
| M0 skeleton | done |
| M1 executor, greedy, 27B coder on B60 | done — 51.4 t/s, 10/10 |
| M2 chunked prefill, cache ledger, long context | done — **257,167 tokens loaded** |
| M3 prefix caching | done — warm/cold byte-identical, hit stats on console |
| M4 MTP | sampling done; MTP needs an export that keeps the head |
| M5 all three models | done — all three serve and reach 10/10; the 35B does not fit the A770 |

M2 is done and measured: slicing logits to the last token removed a wall at
~8k tokens, storing the attention KV as fp16 took 262144 context from 10.0 GiB
to 5.0 GiB, and with both in place ligence loaded **257,167 tokens** — 98% of
the artifact's maximum — in 8.1 minutes. What remains is OpenVINO's
paged-attention path, whose decode-side block-table convention is undocumented;
it is an optimisation now, not a blocker.

Verification: 133 unit cases, a 48-check curl round-trip, and an equivalence
suite that runs where the card is. Clean under ASan and UBSan on x86_64.

See [DESIGN.md](DESIGN.md) for the architecture and milestones,
[llm.txt](llm.txt) for a machine-readable summary.

## Building

C++20, CMake, no network at build time — `third_party/` holds the two vendored
single headers (cpp-httplib, nlohmann/json) and `models/allowlist-raw.json`
holds the IR metadata the allowlist is pinned against.

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure

    ./build/ligence --stub --port 8090 -v

`-DLIGENCE_OPENVINO=ON` is reserved for the executor backend and refuses to
configure until M1 lands it; meanwhile `--model` is refused at startup rather
than silently starting something that cannot run, and `--stub` is the only way
in. `-DLIGENCE_WERROR=ON` for the warning-clean build CI should use.

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
