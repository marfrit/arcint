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

## Where the artifacts come from

**arcint loads an OpenVINO IR directory, and nothing else.** Concretely:
`openvino_language_model.{xml,bin}`, `openvino_text_embeddings_model.{xml,bin}`,
the tokenizer and detokenizer IRs, `config.json` and the chat template. A GGUF
will not load. Neither will GPTQ or NVFP4 safetensors — those are llama.cpp and
vLLM formats, and OpenVINO does not read them.

**All three are downloadable, and every number in the *Measured* section was
taken on the published copy.** Two of them are Intel's own exports and need no
work at all; only the coder had to be built, because its checkpoint is a
community fine-tune with no official IR.

| model | where to get it | note |
|---|---|---|
| Qwen3.6-35B-A3B | [`OpenVINO/Qwen3.6-35B-A3B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov) | Intel's export, used as-is. Verified byte-identical to the copy these measurements ran on. |
| Qwen3.6-27B-A3B-Coder | [`marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov`](https://huggingface.co/marfrit/Qwen3.6-27B-A3B-Coder-int4-awq-se-ov) | Apache-2.0. No official IR exists for this checkpoint; see the calibration section below for what it cost. |
| Qwen3.8-27B | [`OpenVINO/Qwen3.8-27B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov) | Intel's export. **The MTP head is not in it** — reconstruct it with `tools/export_mtp.py`, or run without `--mtp`. |

All three are already in [models/allowlist-raw.json](models/allowlist-raw.json),
so a downloaded copy needs no registration. If arcint refuses one at load time,
the artifact differs from the one these numbers were measured on — that is the
assertion doing its job, not a bug to work around.

Two things learned the expensive way and worth stating, because they decide
whether you need the rest of this section at all:

- **Try the official IR before exporting anything.** The 35B above sat unused
  for weeks on the assumption that a stock export would not clear the quality
  bar. When it was finally measured it scored 10/10 on the acceptance task
  greedy, 3/3 on tool calls, clean German, at 62.7 t/s — better than the
  artifact it was being compared against. A custom export is what you do when
  there is no official one, not a default step.
- **The dense model's MTP head is the one piece nobody publishes.** optimum-intel
  drops it on export, so neither Intel's IR nor yours will have it;
  `tools/export_mtp.py` reconstructs it from the checkpoint's own weights. That
  is worth 1.35× on the dense model and is the only reason to touch the
  export pipeline for it at all.

If you only want to run arcint rather than build an artifact, take the table and
stop reading here. The rest of this section is for anyone who needs a different
calibration than the published ones provide.

**There is no shortcut from GGUF.** llama.cpp gained an OpenVINO backend in
early 2026, and it was built and tested against these models: the production
GDN hybrid aborts during scheduling because the recurrent DeltaNet states are
not mapped in the GGML frontend (`pre-allocated tensor (cache_r_l0) in buffer
(OPENVINO0) that cannot run CPY`), and a classic MoE falls to a CPU path and
asserts in `get_rows`. Its validation list is dense models only. So the path
starts at the Hugging Face checkpoint, not at a quantised file you already have.

    pip install "optimum[openvino]" nncf accelerate pillow "huggingface_hub[cli]"
    pip install "transformers==5.2.0" "openvino==2026.3.0" torchvision

    optimum-cli export openvino --model <checkpoint> \
      --task image-text-to-text --weight-format int4 --group-size 64 \
      --awq --scale-estimation --dataset <corpus> --num-samples 32 out/

### The calibration is a decision, not a default

This is the part that decides whether the artifact is usable, and it cannot be
inferred — it has to be measured against the task you actually serve. Same
model, same bit width, only the calibration changing, scored on the acceptance
task (greedy, and three sampled runs at the model card's own settings):

| export | greedy | sampled |
|---|---|---|
| naive int4, group 64, data-free | 0/10 | 8, 8, 8 |
| AWQ + scale estimation, image dataset | 7/10 | 10, 7, 0 |
| AWQ + scale estimation, code corpus | **10/10** | 10, 8, 8 |

Two findings from that series are worth more than the recipe:

- **Scale estimation is model-class dependent. Never set it blanket.** It is
  part of the 10/10 recipe for the MoE coder and it *destroys* the greedy path
  of the dense 27B — 0/10 with two entirely different corpora, degenerating into
  repetition loops, while AWQ-only on the same model is a healthy 7/10. Measure
  AWQ+SE against AWQ-only per model before believing either.
- **Calibration cuts both ways.** The code-and-English corpus that bought 10/10
  on code produced token salad in German prose (invented compounds, CJK
  characters mid-word) where the GGUF baseline was clean. What is not in the
  corpus is what you lose. Pick the corpus to match the distribution the
  endpoint will actually see, and say so on the artifact.

### Traps that cost a day each

- **`--task text-generation` does not export this architecture.** `qwen3_5_moe`
  exports only as `image-text-to-text`, the same shape Intel's own IRs use.
- **`--ratio` must stay 1.0.** Mixed precision makes group sizes non-uniform and
  the GPU's MoE fusion refuses to load the result. That is why every published
  Intel IR for these models is ratio 1.0.
- **Export on a stable OpenVINO, not a nightly.** An IR produced by a nightly
  segfaults on a stable runtime; a stable IR runs fine on a newer runtime.
- **Scale estimation with code samples is memory-hungry** in a way image
  datasets do not prepare you for: it OOM'd on a 121 GB machine, thrashed on
  247 GB (RSS 244 GB, 99.7% iowait), and completed on 494 GB with a ~152 GB
  peak. `TMPDIR` needs ~110 GB — the fp16 intermediate and the final save both
  land there, and running out produces `basic_ios::clear: iostream error` after
  the entire compression has already finished.
- **Do not use `ov::cache_dir` with these MoE IRs.** The first run writes the
  blob, the second imports it without expert weights and fails at inference
  ("expert weight provider not initialized"). Tracked upstream as
  openvinotoolkit/openvino#37607.

### Registering it

arcint refuses artifacts outside its allowlist at load time, so a new export
needs an entry in [models/allowlist-raw.json](models/allowlist-raw.json):
geometry, quantisation, and the tokenizer and chat-template hashes. That is the
mechanism behind "no model zoo" in the non-goals — the allowlist is the
assertion that the process is serving what it claims to serve.

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
  **byte-identical** to their solo runs in both start orders, and the acceptance
  task scores **10/10 on each lane while both run it at once**. The measured cost of
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

## Measured

One task, one model, four engines. The acceptance task is a Lua CSV parser to
RFC 4180: ten named cases (CRLF and bare LF, a missing final terminator, quoted
fields, an embedded comma, a doubled quote, an embedded newline, empty and
empty-quoted fields, no trimming). The candidate code is **executed**, not read;
one point per case. Greedy where the model tolerates it, otherwise the model
card's own sampling defaults.

Same artifact, same card, only the serving pipeline changes:

| engine | card | weights | decode | task |
|---|---|---|---|---|
| OpenVINO GenAI, stateful pipeline | B60 | int4 AWQ (expert-pruned 184/256) | ≈43 t/s | 10/10 |
| arcint, stateful executor (`--no-paged`) | B60 | same IR | 51.4 t/s | 10/10 |
| **arcint, paged executor, u8 KV** | B60 | same IR | **71.3 t/s** (68.6 at f16 KV) | 10/10 |
| arcint, paged, at ~30k context | B60 | same IR | 70.1 t/s | 10/10 |

For scale, the same model on the other card and the other engine — a different
quantisation and a different card, so read it as a rough bound, not as a row of
the series above:

| engine | card | weights | decode | task |
|---|---|---|---|---|
| llama.cpp, SYCL | A770 | GGUF Q4_K_M | 14.4 t/s | 10/10 |

Two facts the table does not show. The depth column is where the difference
actually lives: 70.1 t/s at ~30k against 71.3 at short context means the usual
collapse with depth is gone from the served path, which is a property of the
paged block tables rather than of raw throughput. And every arcint row above is
gated by byte-equality tests, not just by a score: warm cache against cold, one
lane against two, paged against stateful.

The dense Qwen3.8 serves with its reconstructed MTP head at **36.2 t/s** and
93.2% acceptance, 10/10 greedy. Prefill on the coder reaches ~1970 t/s.

## Status

Serving all three models on both cards, on the paged executor: arcint-owned
block tables, measured-reservation admission, speculative decoding whose
rollback moves zero bytes. u8 KV is the default, at half the memory and
bit-identical output. The stateful executor stays behind `--no-paged` as the
reference implementation the equivalence suite compares against.

| milestone | state |
|---|---|
| M0 skeleton | done |
| M1 executor, greedy, 27B coder on B60 | done — 51.4 t/s, 10/10 |
| M2 chunked prefill, cache ledger, long context | done — 257,167 tokens loaded |
| M3 prefix caching | done — warm/cold byte-identical, hit stats on console |
| M4 MTP | done and served — 36.2 t/s at 93.2% acceptance, 10/10 greedy |
| M5 all three models | done — the 35B fits the A770 too, via `--offload-ratio` |
| M6 two lanes per service | done — byte-identical under interleaving on both cards, 10/10 on each lane concurrently |

Verification: 155 unit cases, a 48-check curl round-trip, a lane-accounting
stress (200 requests, 24-way, 8 lanes, queueing and refusing), an equivalence
suite and a concurrency suite that run where the card is. Clean under ASan and
UBSan on x86_64.

Why these numbers look the way they do — including the ones that came out
negative, and the explanations that were retracted after a measurement
contradicted them — is in [DESIGN.md](DESIGN.md).
[llm.txt](llm.txt) is the machine-readable summary.

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
file carries flags and not a library search path. With `LD_LIBRARY_PATH` unset
there are no unresolved objects, and it serves `/health` under `env -i`.

    cmake -S . -B build-ov -DARCINT_OPENVINO=ON -DCMAKE_BUILD_TYPE=Release \
          -DARCINT_GIT_SHA=$(git rev-parse --short=12 HEAD)
    cmake --build build-ov -j"$(nproc)"
    cmake --install build-ov --prefix ~/.local

Two lanes are a flag and a memory claim: `--parallel 2` reserves activations,
GDN checkpoint rows and KV for two concurrent sequences at the configured
`--n-ctx`, and refuses at startup with the arithmetic if the card cannot hold
them. If a deployment would rather queue than be refused at request time, set
`--queue-timeout S`.

`packaging/arcint.service` is a systemd **user** unit. Its flags are literal
rather than `${ARCINT_PORT}` out of an environment file, and that is not
laziness: the fleet's unit manager reads the port, the served name and the
context out of `ExecStart`, so an env-file indirection hides all three from the
port board and from an automated edit. `ExecStart` points at `/usr/bin/arcint`,
which is where the package puts the binary — building into `~/.local` instead
means changing that one line.
`-DARCINT_GIT_SHA` is worth passing whenever the build tree has no `.git` —
otherwise `--version` and `/props` report `unknown`, and a running binary cannot
say what it is.

The name the endpoint answers to is `--served-model-name`, and it is separate
from `--model-id` on purpose: the first is what `/v1/models` reports and what a
discovering proxy therefore pins its roster to, the second is the allowlist
assertion about which artifact this process will accept. Renaming an endpoint is
a one-flag edit and never weakens the assertion.

One process holds its model's VRAM for its whole lifetime. Whatever else was
serving that card has to stop first: arcint replaces an endpoint, it does not
sit beside one.

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
