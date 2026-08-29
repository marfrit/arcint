# arcint

A deliberately narrow LLM inference engine for Intel Arc GPUs.

arcint runs exactly three model families on exactly two cards, and tries to do
that better than the general-purpose engines do. The inspiration is
[NInfer](https://github.com/Neroued/ninfer) — a from-scratch engine that
supports two checkpoints on one GPU and beats every generalist on that pair —
translated to Intel: instead of hand-written CUDA, the kernel work is delegated
to OpenVINO's compiler stack, which already emits good Xe code. What arcint
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
| Intel Arc A770 (DG2/Alchemist) | 16 GB | ~560 GB/s; the 35B needs `--offload-ratio` here |
| Intel Arc Pro B60 (BMG G21) | 24 GB (~22.7 usable) | ~456 GB/s |

All three target models are hybrids: most layers use linear attention
(GatedDeltaNet), a minority use full attention. That single fact drives most of
the design — see [DESIGN.md](DESIGN.md).

## Features

Marked with the milestone that delivered them, and **[planned]** where the
design exists but the code does not. **[M0]** means it works against the stub
backend; **[M1]**–**[M5]** mean it runs against the real models on a real card.

- **[M0]** OpenAI-compatible **`/v1/chat/completions`** and **`/v1/completions`** over HTTP.
- **[M0]** **`/health`** (liveness, model-loaded, queue depth) and **`/props`**
  (model metadata, context length, quant, cache configuration, build info).
- **[M3]** **Prefix caching** for the full hybrid state — attention KV pages *and* GDN
  recurrent state — with a hard invariant: greedy output with a warm cache is
  byte-identical to greedy output with a cold one. Cache reuse that changes the
  answer is treated as a bug, not a documented quirk.
- **[M2]** **Block-aligned GDN state checkpoints** for the linear-attention
  layers, so a cache hit restores both halves of the hybrid state exactly.
  **[planned]** the paged KV cache for the attention layers — see the status
  note below, it is now the highest-value thing left.
- **[M4]** **Speculative decoding** with a hard greedy-invariance guarantee: a
  drafted token is accepted only when it equals what the sampler would have
  picked anyway, so the answer is byte-identical to non-speculative greedy.
  Two drafters: the **native MTP head** for Qwen3.8 (`--mtp on`, **93.3%
  acceptance**) and a prompt-lookup drafter (`--draft N`) through the
  external-drafter hook. optimum-intel drops the MTP head on export, so
  `tools/export_mtp.py` reconstructs it from the checkpoint's own weights.
  Both are **off by default and both are currently a net loss**, for one
  reason: rolling the graph state back costs 66% of decode time. Net of it,
  MTP is 1.35× faster. Speculative output is *not* guaranteed identical to
  plain greedy — verification is exact, but a multi-token verify pass and a
  single-token plain pass differ by up to 0.013 in the logits on this backend,
  which can flip a near-tie (measured: one token in 64). See DESIGN §3.2 and
  §3.5.1–3.5.2.
- **[M1]** **Fused SDPA** on the full-attention layers: OpenVINO selects
  `ocl::sdpa::opt` for them on both cards (confirmed in a decode profile).
- **[M0]** **Tool-call parsing**: native Qwen tool-call output is returned as structured
  `tool_calls` (OpenAI-compatible). Parsed, never executed; requests without
  declared tools get raw text untouched.
- **[M1]** **Tokenizer and chat template come from the model artifact**, never
  from the server — the template hash is part of the model allowlist.
- **[M0]** **Model-aware sampling defaults** from the model card, per allowlist entry;
  explicit request fields always win.
- **[M0]** **Hard context-overflow rejection**: HTTP 400 with the numbers. No silent
  truncation, no context shift — on hybrid GDN models a shift is not even
  honestly implementable (the recurrent state cannot un-see past tokens).
  History management belongs to the client.
- **[M0]** **Streaming that does not corrupt anything**: a multi-byte code
  point is never split across two SSE chunks, a stop sequence never leaks out
  one fragment at a time, and tool-call syntax never reaches a content delta.
- **[M6]** **Two client sessions per service**, in one process: `--parallel 2`
  gives each lane its own `InferRequest`s (language model, embeddings, both MTP
  requests), its own GDN checkpoint rows and its own KV block table out of a
  shared, refcounted page pool. Gated: two prompts interleaved are
  **byte-identical** to their solo runs in both start orders, and the Prüfstand
  scores **10/10 on each lane while both run it at once**. The measured cost of
  sharing: an agent session at 30k context and a subagent burst get 32.5 and
  31.2 t/s against 68.7 t/s alone, with the session's inter-token stall at
  **p95 17 ms**. Sequences are never batched into one graph call, which is what
  keeps the equivalence invariants intact.
- **[M6]** **Admission with the numbers**: a lane is a memory reservation, so a
  request beyond the reserved lanes is a **503 carrying the arithmetic** rather
  than an unbounded wait (`--queue-timeout S` restores queueing). A context that
  does not fit is refused at startup the same way — on the A770 two lanes of
  40960 serve, two of 65536 are refused with every term spelled out.
- **[M0]** **Cancellation**: a dropped client aborts the request at the next
  scheduler boundary and frees its slot — and with it the lane and its KV
  pages, without touching the other lane's stream.
- **[M0]** **Console state output** in the llama.cpp tradition: per-request timing lines
  (prefill/decode token counts and rates), slot states, memory-map printouts at
  startup, cache hit statistics. stderr is the dashboard.

## Non-goals

- No web UI, no GUI, no gradio, no metrics dashboard. Console and HTTP JSON only.
- No model zoo. A checkpoint outside the table above is rejected at load time.
- No multi-GPU, no pipeline or tensor parallelism. One process, one card
  (several *sequences* per process is M6; several *cards* is not on the list).
- No batching of two sequences into one graph call. Lanes interleave at
  execution granularity instead; batching is an optimisation with its own
  equivalence burden, and it has not been paid.
- No training, no LoRA, no quantization tooling (artifacts are produced offline).
- No Windows.

## Status

Serving the real models on Intel Arc **on the paged executor** — arcint-owned
block tables, measured-reservation admission, speculative decoding whose
rollback moves zero bytes. Measured on a B60 with the 27B coder q4: **68.6 t/s
decode (70.1 at 30k context — the depth collapse is gone), ~1970 t/s prefill,
10/10 on the fleet harness at both depths**, u8 KV by default at half the
memory. The dense Qwen3.8 serves with its MTP head at **36.2 t/s, 10/10
greedy**. The stateful executor stays behind `--no-paged` as the reference
implementation the equivalence suite compares against.

| milestone | state |
|---|---|
| M0 skeleton | done |
| M1 executor, greedy, 27B coder on B60 | done — 51.4 t/s, 10/10 |
| M2 chunked prefill, cache ledger, long context | done — **257,167 tokens loaded** |
| M3 prefix caching | done — warm/cold byte-identical, hit stats on console |
| M4 MTP | **done and served**: paged executor, rollback = checkpoint-row promotion (0.00 s on the console), 36.2 t/s at 93.2% acceptance, 10/10 greedy |
| M5 all three models | done — all three serve; the 35B now fits the A770 too, via `--offload-ratio` |
| M6 two lanes per service | done — byte-identical under interleaving on both cards, 10/10 on each lane concurrently, decode 67.6–69.9 vs 68.8 t/s single-stream |

M4 is done, and the measurement is the result. The Qwen3.8 MTP head — which no
public implementation can export, and which `tools/export_mtp.py` reconstructs
from the raw weights — reaches **93.3% acceptance**, and drafted tokens are
never taken on faith. It is *not* byte-identical to plain greedy, and the reason
turned out to govern chunked prefill too: on this backend, advancing the state
by two tokens in one call is not the same computation as advancing it twice by
one (§3.2). It is still
slower than not speculating, and by now the reason is precisely located:
rolling back 171 MiB of mostly-GDN recurrent state costs 66% of decode time,
inside OpenVINO's `VariableState` API. Net of that, MTP is 1.35× faster. So the
paged decode path is no longer an optimisation — it is the precondition, and it
is worth about 2× on Qwen3.8.

M2 is done and measured: slicing logits to the last token removed a wall at
~8k tokens, storing the attention KV as fp16 took 262144 context from 10.0 GiB
to 5.0 GiB, and with both in place arcint loaded **257,167 tokens** — 98% of
the artifact's maximum — in 8.1 minutes. What remains is OpenVINO's
paged-attention path, whose decode-side block-table convention is undocumented;
it is an optimisation now, not a blocker.

M6 is done, and it is one process rather than two. `--parallel 2` runs two
sequences with no shared mutable state and no lock on the decode loop; what
orders them is a FIFO turnstile around graph executions, which is what bounds
the stall (a decode step waits for at most one execution of the other lane) and
what makes each lane's output safe to read — the GPU plugin pools intermediate
buffers per compiled *model*, not per request, which is also why the second lane
costs **0.00 GiB** of activations. See DESIGN §4.1 and §7.2.

Verification: 155 unit cases, a 48-check curl round-trip, a lane-accounting
stress (200 requests, 24-way, 8 lanes, queueing and refusing), an equivalence
suite and a concurrency suite that run where the card is. Clean under ASan and
UBSan on x86_64.

See [DESIGN.md](DESIGN.md) for the architecture and milestones,
[llm.txt](llm.txt) for a machine-readable summary.

## Building

C++20, CMake, no network at build time — `third_party/` holds the two vendored
single headers (cpp-httplib, nlohmann/json) and `models/allowlist-raw.json`
holds the IR metadata the allowlist is pinned against.

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    ctest --test-dir build --output-on-failure

    ./build/arcint --stub --port 8090 -v

`-DARCINT_OPENVINO=ON` is reserved for the executor backend and refuses to
configure until M1 lands it; meanwhile `--model` is refused at startup rather
than silently starting something that cannot run, and `--stub` is the only way
in. `-DARCINT_WERROR=ON` for the warning-clean build CI should use.

## Deploying it

The installed binary is self-contained: it resolves the OpenVINO runtime and
the tokenizers extension through RPATH, both found at configure time, so a unit
file carries flags and not a library search path. Verified on dirac — with
`LD_LIBRARY_PATH` unset there are no unresolved objects, and it serves `/health`
under `env -i`.

    cmake -S . -B build-ov -DARCINT_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release \
          -DARCINT_GIT_SHA=$(git rev-parse --short=12 HEAD)
    cmake --build build-ov -j"$(nproc)"
    cmake --install build-ov --prefix ~/.local

Two lanes are a flag and a memory claim: `--parallel 2` reserves activations,
GDN checkpoint rows and KV for two concurrent sequences at the configured
`--n-ctx`, and refuses at startup with the arithmetic if the card cannot hold
them. If a deployment would rather queue than be refused at request time, set
`--queue-timeout S`.

`packaging/arcint.service` and `packaging/arcint.env` are a systemd **user**
unit and its configuration, matching how the fleet already runs its inference
services. Every arcint setting is a command line flag, so the env file holds
the flags and the unit itself never has to be edited. `-DARCINT_GIT_SHA` is
worth passing whenever the build tree has no `.git` — otherwise `--version` and
`/props` report `unknown`, and a running binary cannot say what it is.

On dirac specifically, unit changes go through the Roundhouse MCP rather than
hand-edited systemctl: a hand edit without a commit in `~/.config/systemd/user`
crashes roundhouse. The file is the content to hand it, not permission to
install it by hand.

Note the port: arcint and the fleet's existing coder endpoint both want the
same card, and the B60 holds a model's VRAM for the process lifetime. Running
arcint there means replacing that endpoint, not sitting beside it.

## Why not just use …

- **OpenVINO GenAI**: its continuous-batching path (required for prefix
  caching) produces measurably different greedy output than its stateful path,
  the equivalence test in its own CI has been skipped since 2025-02, and hybrid
  state is checkpointed at coarse intervals sized for memory rather than
  correctness. arcint keeps the OV *compiler* and replaces the pipeline layer.
- **vLLM**: no usable Intel path for these hybrids (the XPU build lags, and
  Arc is not a first-class target). The paged-KV and prefix-cache ideas are
  taken; the engine is not.
- **llama.cpp**: excellent console ergonomics and a sane server, but Vulkan on
  Battlemage is unoptimized (measured 7.9 tok/s dense where OpenVINO does 60)
  and SYCL is broken under the xe KMD. The ergonomics are taken; the backend
  situation is why arcint exists.
