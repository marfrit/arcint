# arcint — Design

Status: **0.5.6**, runtime floor `marfrit-openvino +p25` (patches 0003–0074
on the pinned OpenVINO nightly, §1.1; `+p27` for the Flash-Next tier's opt-in
switches of patches 0076–0077). Since 0.5.5 a second executor serves GGUFs
through libllama (§7.11); from 0.5.6 it serves the agent. This document states what arcint *is*
today: the architecture, the invariants, the gates, and the standing measured
value of every subsystem. It is not a diary.

**Where the history lives.** Every number below was measured, and most were
measured more than once, corrected or retracted on the way. That dated record —
the full §7.0.x measurement entries, the campaign documents
(`docs/campaigns/`), the design notes (`docs/design-*.md`), the milestone
charters, the acceptance windows and the reviews — lives on the development
branch, **`qfndev`**, and not on `main`. Section numbers in this document are
the record's numbers: a citation such as "DESIGN §7.0.2ae" in a source comment
resolves here to a one-line standing result (§7.0.2, the record index) and on
`qfndev` to the full entry. A `docs/…` path named in a source comment or a patch
header that is absent here is on `qfndev` as well.

**Evidence classes.** Every standing number carries one: `measured-here` (this
repository measured it on its own hardware, card and configuration named),
`code` (read from source — ours, the plugin's, or a named external project's),
`paper` (a vendor specification or a publication). A label is not evidence; a
narrated mechanism that was not measured is retracted, not edited (§7.0.1).

**Hardware named in this document.** "The 16 GiB card" is an Intel Arc A770
(Xe-HPG, ACM-G10, PCI `8086:56a0`, 15.11 GiB usable to OpenVINO), "the 24 GB
card" an Intel Arc Pro B60 (Xe2, BMG-G21, PCI `8086:e211`, 22.71 GiB usable).
OpenVINO's device order and the DRM card numbering need not agree, so card
identity is established by PCI id.

## 1. The one idea

General-purpose engines pay for generality twice on this hardware: once in
kernel quality (Vulkan/SYCL paths that were never tuned for Xe), and once in
pipeline complexity (abstractions for hundreds of architectures the target
machine will never load). NInfer showed on CUDA what specialization buys.
On Intel, the kernel half of that bet is largely won by someone else:
OpenVINO's graph compiler and kernel library emit good Xe code (on Battlemage
for the same checkpoint: 3.4× llama.cpp-SYCL, 7.6× llama.cpp-Vulkan, measured
in the 2026-08 campaigns that started this project). What is *not* won is the layer above — scheduling, cache
management, hybrid-state handling, sampling, serving — where OpenVINO GenAI
makes choices arcint cannot live with (§6).

So: **OpenVINO as compiler and kernel library, arcint as everything else.**

### 1.1 When a kernel has to change: smallest sufficient divergence

A fork of OpenVINO is not wanted: it erodes the boundary above, turns the
packaged runtime from a repack of an upstream wheel into our build, puts a
rebase on the calendar every time the nightly moves, and makes every
measurement one against our tree rather than against a version anyone can
obtain. And a kernel changes anyway when the work needs it — almost nobody
serves GDN hybrids on Arc, so nobody optimises these operators for it. The
rule is **smallest sufficient divergence**, in this order:

1. **Upstream PR**, written to upstream's conventions from the first line.
2. **A numbered patch series against the pinned version**, applied at build
   time by the packaging recipe (`contrib/packaging/marfrit-openvino/`), each
   patch PR-shaped so it can be re-offered. This is where arcint is today:
   patches 0003–0067 on upstream commit `71640275` (the 2026.4.0 nightly of
   2026-08-21), packaged as `marfrit-openvino +p20`. The per-patch record —
   what each changes and its standing measured effect — is
   `contrib/packaging/marfrit-openvino/patches/README.md`; `patches/` at the
   top of the tree mirrors the series byte for byte (a device-free test fails
   on drift) and also holds the two patches measured and deliberately not
   carried (0001, 0002).
3. **A maintained fork**, last resort, with a stated reason and an exit
   condition.

**Whatever is carried is published**: a number that depends on a patch nobody
else has is an anecdote. The plugin's build number carries the patch level
(`…-71640275d29-marfrit-pN`), and arcint reads it at load where a charge
depends on it (§7.5). The recipe resets the checkout hard before applying the
series, so a dirty measurement tree cannot leak into a package.

Consequence planned for, not discovered: a patched runtime is a real x86_64
compile (~35 minutes on eight cores) where the unpatched one was a wheel
repack in seconds.

## 2. Target constraints (facts, not choices)

- Every served model is a **hybrid GatedDeltaNet/attention** transformer: most
  layers carry a fixed-size recurrent state (conv state + delta-rule state), a
  minority are full-attention layers with a KV cache — one layer in four in
  every served family (`full_attention_interval = 4`, `code`: the IRs).
  Recurrent state per sequence is fixed and small; attention KV grows with
  context. At 262k context the KV of the few attention layers dominates; at 4k
  the weights dominate everything.
- One card, one process (§4.1 for lanes inside it). The Xe KMD only; SYCL
  runtimes older than the xe transition abort on memcpy, which is why
  OpenVINO's OpenCL/oneDNN path is the compute route on both cards.
- **The served families** (the allowlist, `src/core/model_registry.cpp`, is
  authoritative; every entry pins architecture, template and tokenizer hashes
  and the weight byte count):

  | family (`model_type`) | model | layers (GDN + attn) | experts | artifacts on the allowlist |
  |---|---|---|---|---|
  | `qwen3_5_moe` | Qwen3.6-27B-A3B-Coder | 40 (30 + 10) | 184 (pruned from 256) | the b5 int4 AWQ+SE export (production coder) |
  | `qwen3_5_moe` | Qwen3.6-35B-A3B | 40 (30 + 10) | 256 | Intel's int4 IR; the same with a reconstructed MTP head; the native-format serving-shape artifacts (full depth: `qwen3.6-35b-a3b-native-d40packed-u8`) |
  | `qwen3_5` (dense) | Qwen3.8-27B | 64 (48 + 16) | — | the b7c1 AWQ-only export (production agent, MTP head reconstructed); Intel's int4 IR as its own entry; any Q4_K_M-class GGUF of it through `--gguf` (§7.7) |
  | `qwen4_exp` | Qwen3.8 Flash-Next | 48 | 512, top-10 | the serving-shape artifacts built from the GGUF (full depth: `qwen3.8-flash-next-d48n`, the checkpoint's own expert formats) |
  | `qwen3_5` (dense) | Qwen3.5-2B | — | — | a provisional marker export |

  All families share one tokenizer (`87a7830d63fcf43b`). Entries marked
  "measurement artifact" in the registry (depth rungs, A/B rungs) are
  admitted so a measurement can load them; their answers are not the model's.
- Weights: int4/int8 IRs as exported, the GGUF K-quant and i-quant blocks as
  stored (§7.7, §7.8). KV: `f16`, `u8`, or asymmetric `u8:i4` on the paged path
  (§7.0.3).

## 3. Architecture

```
             ┌────────────────────────────────────────────────┐
             │ HTTP server (single thread pool, no framework) │
             │  /v1/chat/completions  /v1/completions         │
             │  /health  /props  /v1/models                   │
             └───────────────┬────────────────────────────────┘
                             │ request objects
             ┌───────────────▼───────────────┐
             │ Admission                     │  a lane is a memory
             │  lanes = measured reservation │  reservation (§4.3)
             └───────────────┬───────────────┘
                             │ one sequence per lane
     ┌───────────────────────▼─────────────────────────┐
     │ Executor (paged; stateful behind --no-paged)    │
     │  ┌───────────────┐   ┌──────────────────────┐   │
     │  │ Cache manager │   │ OV compiled graph(s) │   │
     │  │  paged KV     │◄─►│  language model      │   │
     │  │  GDN rows     │   │  embeddings gather   │   │
     │  │  prefix index │   │  MTP / DFlash head   │   │
     │  └───────────────┘   └──────────────────────┘   │
     │  turnstile (FIFO over graph executions, §4.1)    │
     │  sampling (greedy, temp/top-p/top-k, penalties)  │
     └──────────────────────────────────────────────────┘
```

Each lane owns its `InferRequest`s (language model, embeddings, drafter), its
GDN checkpoint rows, its KV block table and its logits buffer; the compiled
models and the refcounted KV page pool are shared. Sequences are never batched
into one graph call (§4.1).

### 3.1 Model artifacts

Input is an **OpenVINO IR directory** produced offline — the language model,
the text-embeddings model, the tokenizer and detokenizer IRs, `config.json`
and the chat template. arcint validates it against the compiled-in allowlist
(`src/core/model_registry.cpp`, transcribed from `models/allowlist-raw.json`,
which stays in the tree as the provenance record) and refuses anything else.
An entry pins the architecture hash, the chat-template hash, the tokenizer
hash, the layer geometry, the trained context and the **weight byte count**
(`weights_bytes` is a contract: a mismatch refuses the load, the same as a
hash). A field the raw metadata does not carry is left unpinned and reported
as `null` on `/props`, never invented. The allowlist keys on the artifact's
directory name; a download has to land in the directory the entry names.

Artifact provenance is part of the contract because calibration is: scale
estimation degenerates greedy decoding on the dense Qwen3.8 (0/10) while
AWQ-only stays healthy (`measured-here`, 2026-08), so that entry records
AWQ-only and must not be re-exported with SE.

A **GGUF opens on top of such a directory** (`--gguf FILE --model DIR`, §7.7),
and the **serving-shape artifacts** for `qwen4_exp` and the native
`qwen3_5_moe` rungs are emitted by `tools/export_serving_artifact.py` from the
GGUF shards (§7.8, §7.1). The vision IRs a VLM export carries are reported at
load and never loaded; `--vision` is reserved and refused.

### 3.2 Graph strategy

**The served path is the paged graph.** `ov::pass::SDPAToPagedAttention`
turns the exported stateful graph into one with explicit cache ports: paged
`key_cache`/`value_cache` per attention layer, `conv_state_table` and
`gated_delta_state_table` per GDN layer driven by a separate linear-attention
block table, and the `past_lens`/`subsequence_begins`/`block_indices` family.
arcint owns every byte of cache and every table; OpenVINO owns the math. The
original stateful executor stays behind `--no-paged` as the reference the
equivalence suite compares against. Serving-shape IRs (§7.1, §7.8) are paged
only.

**A different split of the same tokens is a different computation on this
backend.** Advancing the recurrent state by two tokens in one forward is not
bit-identical to advancing it twice by one (the last-row logits differ by up to
0.013 on the dense model, `measured-here`; the GDN scan and the matmul tiling
are chunk-sensitive). This single fact governs chunked prefill, speculative
decoding and the prefix cache, and the design answers it once:

- **Prefill chunks sit on an absolute grid** — multiples of the chunk size
  counted from position 0 — and prefix-cache checkpoints are restricted to the
  same grid (§3.4). A warm run starts on a boundary the cold run also stopped
  at, so the two see the same split by construction.
- The grid is **configuration, never a scheduling variable** (§4.1).
- Chunked-against-unchunked and one-chunk-size-against-another are
  **reported, not gated**; determinism at a fixed configuration is gated.

`--prefill-chunk` defaults to 2048: chunking is the mechanism that bounds
activation memory, and ordinary prompts land in one chunk. The fit may choose
a smaller served chunk when the card needs it (§7.5).

**The logits slice.** The graph emits logits for every prompt token; nothing
samples them. The load inserts a `Slice` before the LM head so prefill
computes the last `1 + draft` rows only, reads the token axis from the head's
declared shape (`[tokens, 1, hidden]` on the paged export, `[1, tokens,
hidden]` on a serving-shape IR), and **verifies the claim with a real probe
forward before serving** ("logits slice verified: 1 row(s) for a 128-token
forward") or refuses to start. `--no-logits-slice` turns it off for the
equivalence suite's proof that the two agree.

**The compiled-blob cache is opt-in and proven.** `--cache-dir` enables it;
whatever it returns is exercised by a real forward before the socket binds, and
a blob that fails is discarded and recompiled. The MoE import defect
(openvinotoolkit/openvino#37607: an imported blob without the expert weight
provider, 500 on the first infer) made this mandatory; the paged load compiles
its language model with the cache switched off.

### 3.3 Memory: paged KV + GDN rows

- **KV pages** are the plugin's 16-token pages; `--kv-block-size` (16/32) is
  the prefix cache's reuse granularity, a multiple of it. The pool is sized in
  **bytes** from the measured reservation (§7.5) and is **refcounted**
  (`src/core/block_pool.h`): a page can be live in one lane, held by a cache
  entry and mapped by the other lane at once. Eviction exists in one form only:
  **cached** prefixes are dropped when a live sequence needs pages; a live
  sequence's pages are never taken, and a pool that cannot be freed enough
  ends the request cleanly instead of failing on the card.
- **KV precision** is real quantisation with plugin-managed scales, chosen per
  side with `--paged-kv KEY[:VALUE]` (§7.0.3). A per-side bitwidth audit
  refuses a compiled model whose ports do not carry the requested widths
  (the plugin's u8-stored-as-i8 and 4-bit-in-8-bit-typed ports are measured
  aliases, not a change).
- **GDN rows**: each lane's recurrent state is one row of the state tables,
  plus checkpoint rows for the prefix cache. Fresh rows **must be zeroed**
  (the kernels read the committed row even at `past_lens = 0`); they are
  zeroed on the device from one resident zero row per state shape, and a
  checkpoint read or write goes through a device-side ROI copy.
- The stateful reference path stores its KV variables as f16 by graph surgery
  (`--kv-dtype fp16`, the default; `fp32` is what the artifact exports). A
  plain cast to int8 is refused: without scales it is not quantisation.

### 3.4 Prefix caching

- A hash chain over token blocks (content hash, keyed 128-bit, token identity
  verified on hit before reuse), one entry per (block hash, position).
- A hit restores **both halves of the hybrid state or neither**: the KV pages
  by reference and the GDN checkpoint row at the same block boundary. A hit
  lands on a block boundary by construction, so every page it maps is
  complete and never written again; the page a sequence writes into is always
  one it allocated itself. The backend asserts the alignment and falls back to
  a cold prefill otherwise.
- Snapshots land on the prefill-chunk grid by default (`--cache-grid 0`); a
  finer grid is exact and currently costs time (§7.3).
- An evicted entry can be demoted to a host tier instead of dropped (§4.4).
- **Invariant (tested, not aspirational): for any prompt and any cache state,
  greedy output is byte-identical to a cold run.** It holds per lane with the
  other lane active (§4.1), and it is **history-independent**: output depends
  on the request and the configuration, never on the process's own request
  history. The equivalence test is the *gate*: a change that breaks it does
  not merge, and a path that cannot meet it is configured out, not papered
  over. Standing consequences:
  - **`--moe-cpu-tier` with the prefix cache** is admitted only when the
    plugin reports a static residency partition
    (`MOE_CPU_TIER_STATIC_PARTITION`, patch 0018; read on the compiled model,
    fail closed). The earlier LRU tier chose device-f16 or host-f32 arithmetic
    per expert by residency, so a restored continuation forked from a cold run
    (`measured-here`, §7.0.2ae); under the static partition each expert's
    placement is a pure function of configuration, and the fork is gone
    (§7.0.2ai). A census-seeded resident set (patch 0046) keeps this: the seed
    is a pure function of a recorded census.
  - **A route whose kernel choice depends on the call size** must be one
    route for every call size: patch 0064's matrix-unit gate/up runs for
    every call, never beside a scalar kernel for small calls (§7.4).
  - **Native per-expert dispatch at partial residency** computes a resident
    expert on the card and a missed one on the host, and those are not
    bit-identical (§7.0.2cf, measured before patch 0056 and owed a
    re-measurement); the all-resident pool has no misses and no such mix.
- Tier-on output is identical to itself across processes and requests; it is
  not claimed identical to tier-off output (device f16 against host f32
  arithmetic, §7.0.2aj).

### 3.5 Speculative decoding

Speculation runs only under greedy, and a drafted token is accepted only when
it equals what the sampler would have picked from the logits row the verify
pass computed — penalties applied first, and through the same EOS,
`max_tokens` and `n_ctx` gates as a normally picked token. So a draft is never
taken on faith. **Greedy output is not claimed byte-identical to plain
decoding**: the verify pass is a multi-token forward (§3.2), and a near-tie can
land the other way. What is gated: determinism at a fixed configuration,
non-zero acceptance, and warm-equals-cold with the drafter on. Three drafters,
one per server, all off unless asked for: the native MTP head (`--mtp`), the
DFlash2 block-diffusion head (`--dflash`), and a prompt-lookup drafter
(`--draft N --draft-ngram K`). Standing numbers are in §7.6.

#### 3.5.1 The machinery

The verifier needs one logits row per drafted position, so the slice keeps
`1 + draft` rows and an out-of-range row is a guaranteed rejection, reported,
never clamped. On the stateful path a rejected draft needs a state rollback
that copies 70–171 MiB of mostly-GDN state per step (`measured-here`), which
is why speculation pays only on the paged path.

#### 3.5.2 The MTP head

optimum-intel drops the MTP layer on export. `tools/export_mtp.py` rebuilds it
from the checkpoint's own `mtp.*` tensors and extracts the base model's LM
head as a second IR. The forward pass was recovered by measurement with
acceptance as the oracle — sound only because a wrong head cannot change the
output, only depress acceptance: zero-centred norms applied as `(1 + w)`,
`q_proj` interleaving each head's query with its gate, a sigmoid output gate
(the config's `swish` scores 13% against the sigmoid's 66% offline,
`measured-here`). The same head pairs with Intel's public Qwen3.8 IR; Intel's
own exported MTP layer pairs with this repository's LM-head graph
(`--mtp-layer exported`). The 3.6 checkpoints' MoE head exports the same way
(`--moe-lowering tiled` emits the fusable tiled form).

#### 3.5.3 Speculation on the paged path

Draft tokens are checkpointed into scratch GDN rows (`la.cache_interval = 1`,
successive rows per token); the spec pass computes bitwise the logits of a
plain pass over the same tokens, the last checkpoint bitwise equals the
in-place state, and the committed row is never written. **Rollback is
promotion of a row index plus `past_lens` arithmetic: zero state bytes move**,
and the console prints `re-forward 0.00 s, rollback 0.00 s` because neither
exists. The drafter's own KV is exempt from rollback: it consumes committed
tokens only.

#### 3.5.4 The served port

The paged executor is the served path. Drafters can be parked on the other
card (`--mtp-device`, `--dflash-device`, `--emb-device`): per step only a
hidden row and an embedding cross, and the output is byte-identical to the
same-card run. The drafters' rotary subgraphs are kept at f32 (the plugin's
f16 default overflows at position 65,504 and acceptance collapsed there), and
the MTP layer's unpaged per-token state is charged against the reservation
(§7.5).

### 3.6 Sampling

Greedy, temperature, top-k, top-p, repetition, presence and frequency
penalties, host-side (`core/sampler.cpp`). Penalties apply *before* the greedy
decision. Seeded and reproducible: an unseeded request is given a seed and the
seed is logged. Defaults come in four layers — request fields over operator
flags (`--temp`, `--top-p`, `--top-k`, `--repetition-penalty`,
`--presence-penalty`, `--chat-template-kwarg enable_thinking=BOOL`) over the
artifact's `generation_config.json` over the model card — and `/props` reports
the resulting defaults with their provenance. There is no `--min-p`: the
sampler does not implement it, and a flag for an unimplemented knob would be a
lie. `usage.completion_tokens_details` reports accepted and rejected
prediction tokens per response.

### 3.7 Tokenizer, templates, tool calls

- **Tokenizer and chat template ship inside the artifact**; arcint never
  substitutes its own copy, and the template hash is pinned. minja renders it
  with polyfills off, so nothing rewrites what the template says; a tool
  call's arguments are handed to a template that declares it wants a mapping
  as the parsed object (input normalisation, not a polyfill).
- **Reasoning**: when the rendered prompt ends inside a `<think>` block, the
  server splits everything before the first `</think>` into
  `reasoning_content` (streaming sends it first); `enable_thinking` and
  `reasoning_effort` both switch it.
- **Incremental detokenization**: a UTF-8 code point is never split across SSE
  chunks, a stop sequence never leaks one fragment at a time.
- **Tool-call parsing**: both Qwen wire forms (JSON body, and the coder's
  `<function=>/<parameter=>` XML with schema-driven type coercion) are returned
  as OpenAI `tool_calls`; parsed, never executed. A request that declares no
  tools gets raw text untouched.
- **Cancellation**: a dropped client aborts the request at the next scheduler
  boundary and frees its lane and pages.

### 3.8 Context-overflow policy

A prompt (or a continuation) that exceeds the model context is **rejected with
HTTP 400** and a JSON body carrying the numbers (`prompt_tokens`, `n_ctx`,
`overflow`). No server-side truncation, no llama.cpp-style context shift, no
silent sliding window — for two reasons, one principled and one physical:

1. Any server-side history edit silently changes what the model saw, breaks
   prefix-cache identity, and violates the byte-equality invariant (§3.4).
2. On hybrid GDN models a context shift is not even implementable honestly:
   KV pages can be evicted, but the recurrent linear-attention state has
   already integrated every past token — **a GDN state cannot un-see**. Every
   shift would be an approximation, i.e. exactly the class of silent
   divergence this engine exists to refuse.

History management (compaction, summarization) is the client's job, as it is
with the OpenAI and Anthropic APIs. The 400 carries enough data for the client
to do it well.

## 4. Serving surface

| endpoint | contract |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible; `stream` (SSE) and non-stream; `chat_template_kwargs.enable_thinking` and `reasoning_effort` honoured |
| `POST /v1/completions` | raw completion, same sampler surface; token-id prompts accepted |
| `GET /health` | 200 + JSON: model, loaded, lanes free/total, queue depth, prefix-cache and host-tier counters |
| `GET /props` | model metadata, the reservation terms, the served cache block (`path`, `kv_dtype`, `kv_block_tokens`, `prefix_cache`, …), MTP/drafter state, build info, sampler defaults with provenance |
| `GET /v1/models` | the one served model, with the context it is **running with** (§4.2) |

Console output on stderr, llama.cpp tradition: one line per event, greppable,
no colours; `-v` adds a line per request. A request's lines carry its rates
and a split of where the time went, waiting on the other lane included:

```
lgc  slot 0: prefill 32768 tok in 42.07 s (778.9 t/s) | cache snapshot 0.04 s | graph 41.97 s, embed 0.06 s, pages 0.00 s, restore 0.00 s, wait 0.00 s, other 0.00 s
lgc  slot 0: decode     32 tok in  1.87 s ( 17.1 t/s) | graph 1.86 s, embed 0.00 s, sample 0.01 s, emit 0.00 s, wait 0.00 s, other 0.00 s | draft accept 0.0% (0/0), propose 0.00 s, verify 0.00 s, re-forward 0.00 s, rollback 0.00 s
```

### 4.1 Two lanes (M6)

`--parallel N` gives N lanes in one process; everything is gated at 2. **A
lane is one sequence's mutable state and nothing in it is shared**; the
compiled models and the page pool are shared, immutable or refcounted.
Weights are shared between `InferRequest`s of one `CompiledModel` and only
between those (two *compiles* of one graph cost one full copy each,
`measured-here`).

**Sequences are never mixed inside one graph execution.** Each lane steps its
own request over its own rows and pages — the only shape under which §3.4
survives, since a batched call changes the arithmetic of every sequence in it.

**The scheduler is a FIFO ticket lock (`Turnstile`, `src/core/turnstile.h`),
and it is a correctness mechanism first.** The GPU plugin pools intermediate
buffers per compiled model, not per request (a second `InferRequest` adds
0.00 GiB, `measured-here`), so two concurrent executions would overwrite each
other, and a request's output tensor is valid only until the next execution on
that model by anyone. Every shared compiled model (language model,
embeddings, drafter) is therefore taken in turn, and each lane copies its
outputs inside its turn, where they are consumed. The bound follows: a decode
step waits for at most one execution of the other lane, so the worst stall is
one prefill chunk, and `--prefill-chunk` is the operator's latency knob. The
chunk is never shrunk under contention (§3.2).

### 4.2 The name, and the context, are contracts with a proxy

A discovering proxy republishes whatever id `/v1/models` reports and takes the
context from the same object, so both are published there:

- `--served-model-name NAME` sets what `/v1/models` (`data[].id`) and `/props`
  (`model.id`) report and what a completion echoes back. Presentation only.
- `--model-id` is the artifact assertion — which checkpoint this process
  accepts — and keeps refusing anything outside the allowlist whatever the
  endpoint is called. `model.canonical_id` is always published beside
  `model.id`; `model.answers_to` lists both; `model.enforces_model_field` is
  `false` (one process serves one model).
- The model object carries `n_ctx` (what this process is **running with**) and
  `n_ctx_train` (the artifact's ceiling) as separate fields, plus `quant`,
  `lanes` and `canonical_id`.

### 4.3 Admission: a lane is a memory reservation

`--parallel N` is a claim about memory: the startup reservation (§7.5) reserves
activations, GDN rows and KV for N concurrent sequences at the requested
`n_ctx`. An N+1st sequence is **refused with those numbers** — HTTP 503
carrying the terms `/props` publishes — before a response byte is committed,
unless `--queue-timeout S` asks for waiting (the packaged unit passes 30:
the engine tells the truth, the deployment chooses the manners). When a lane
needs a page and the pool is dry, cached prefixes are dropped first; only then
does the request end, cleanly.

### 4.4 A host tier for evicted prefixes

`--cache-host-mib N` (0 = off) demotes an evicted prefix entry instead of
dropping it: its KV pages are copied to host buffers by page runs through
`RemoteTensor` ROI views and released, the GDN row (already host-resident)
stays, and a hit promotes the entry back — pages allocated, copied, and the
ordinary restore. Pages come back byte-exact, so §3.4 holds by construction;
the gate is the same warm-equals-cold check under a pool small enough to force
demotion and promotion (`--kv-pool-pages N` is the test knob). Measured on the
24 GB card: a demoted 4,096-token entry came back in 0.02 s and the warm
answer was byte-identical to cold (`measured-here`). The production agent runs
4,096 MiB: an offline replay of the operator's real sessions put 37% of the
agent's prefill in re-prefilling sessions evicted under pool pressure
(`measured-here`), and the pool total is fixed by the card.

## 5. Testing and acceptance

- **The acceptance task (the Prüfstand)**: a Lua CSV parser to RFC 4180, ten
  named cases, scored by **executing** the candidate code, one point per case.
  **10/10 on the served coder artifact is the bar**, and every artifact and
  configuration that claims production readiness is scored through the served
  endpoint. The score is a floor, not a fingerprint: a different program that
  also scores 10/10 is reported as different.
- **Equivalence suite** (`tests/equivalence/run.sh`, run where the card is):
  byte-equality gates — two greedy runs identical, warm prefix cache identical
  to cold, a continuation restored from the cache identical to a cold run,
  speculative decoding deterministic and exact on a copy-the-input prompt, MTP
  warm equal to cold, the logits slice leaving the answer unchanged — on one
  lane and, with `ARCINT_EXTRA_ARGS="--parallel 2"`, on a two-lane engine.
  Chunked against unchunked prefill and stateful against paged are **reported,
  not gated** (§3.2). A GGUF-opened or serving-shape model skips the stateful
  comparison (it serves paged only).
- **Concurrency suite** (`tests/concurrency/run.py`): no cross-lane bleed (two
  prompts interleaved are byte-identical to their solo runs **in both start
  orders**), both lanes used, cold/warm per lane with the other lane busy, the
  cache holding pages, cancellation leaving the other lane's bytes and giving
  back the lane and its pages, admission as a 503 with the numbers and no
  `CL_OUT_OF_RESOURCES` anywhere in the log, the stall reported. Verified red
  before green: a build in which both lanes index lane 0 fails exactly the
  bleed and cancellation checks.
- **Unit ladder, device-free, every commit**: `arcint-test` (hand-rolled
  harness; a skip is a failure unless named), a curl round trip against the
  stub, the stub-only lane-accounting stress, and the acceptance enumeration's
  own consistency checks — what bare `ctest` runs. Counts are printed by the
  run, not recited here. Warning-clean under `-Wall -Wextra -Wpedantic
  -Werror`; clean under ASan + UBSan with `-fno-sanitize-recover` on x86_64
  (ASan aborts at startup on aarch64). The plugin patches carry their own unit
  cells, run on a card whenever their patch changes.
- **Red before green.** A test must be able to fail: the red case is run
  first, and a check that was green before the change measures nothing.
- **Host memory is part of the measurement.** Driver and USM-host memory are
  charged to the physical host, not a container's cgroup, and on the dev host
  the ZFS ARC is large and not counted in `MemAvailable`. Deep-context and
  large-model legs run with the resident services stopped, a physical-host
  sampler running before the leg starts, and a watchdog.
- **Performance bars.** The coder on the 24 GB card decodes at ≥ 60 t/s warm
  (the OpenVINO GenAI baseline on the same card and artifact; gated, standing
  66.5 t/s, `measured-here`). The operator's prefill bar at depth, 460 t/s,
  is met on the 16 GiB card by the Qwen3.6-35B native artifact at 32k (778.9
  t/s, §7.1) and on the 24 GB card by the GGUF-opened dense 27B at 71.7k (464
  t/s, §7.7). Prefill is a first-class regression metric per card.

### 5.1 The test ladder: what runs when

Three classes, by cadence and card time. The acceptance row is generated from
`tests/acceptance/cells.json` by `tools/acceptance_manifest.py` (which also
writes `docs/release-checklist.md`); `--check` fails the device-free
`acceptance-enumeration` test when either has drifted.

| class | cadence | card time | members |
|---|---|---|---|
| Unit tests | every commit, every milestone | seconds to minutes | `arcint-test` via `ctest` in the stub build — among them config parsing and the refusal ladder (`tests/test_config.cpp`), the fit arithmetic (`tests/test_fit.cpp`), decode accounting and the cycle-profile line (`tests/test_decode_stats.cpp`, `tests/test_profile_cycle.cpp`), a 64-check curl round-trip (ctest's `roundtrip`) and the lane-accounting stress test (ctest's `stress`, `tests/concurrency/stress.sh`, stub-only). Plus the plugin unit tests each patch carries (`ov_gpu_unit_tests` filters: `patches_0015_paged_attention_*`, `regression_paged_attention_*` and 0016's own review suites, `moe_otd_perf_counters.*` from 0017) — these need a card but run in minutes, red-first wherever the test reports a found defect (construction locks say so), and run whenever their own patch changes |
| Milestone gates | once per milestone increment | one card window, minutes | `tests/equivalence/run.sh` and `tests/concurrency/run.py` on the configuration the milestone changes (M9: the two offload configurations; M11: drafter on/off at the depth in question), plus the milestone's own measurement cell (M14's reference cell, M11's step profile), one process per configuration, numbers into §7 |
| Acceptance | once per release, before the tag | hours | <!-- BEGIN GENERATED by tools/acceptance_manifest.py; verify with its check mode -->`coder-offload-1lane` gates byte-equality: cold vs warm cache, chunked vs unchunked prefill, one chunk size vs another (reports chunk sweep) (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-offload-2lane` gates the same byte-equality claims as coder-offload-1lane, held at --parallel 2 (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-offload-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `coder-served-large` gates byte-equality: two greedy runs, cold vs warm cache, a restored continuation vs cold, and (draft 4) a copy-the-input prompt (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-served-large-decode` gates byte-identity of the two requests' outputs; the warm second-request decode rate against its reference once filled (reports until then, §8.8) (reports 0.3.0 record (DESIGN §7.0.2ai): 53.4 t/s cold / 69.2 t/s warm decode) (references: decode-warm-2nd 66.5 t/s (gate lower-is-worse at 60.0 t/s); decode-cold-1st 66.6 t/s (report only, not gated); prefill-warm-2nd 2820.8 t/s (report only, not gated); prefill-cold-1st 2812.3 t/s (report only, not gated)); `coder-served-small` gates byte-equality (as coder-served-large), on the 16 GiB card (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-served-small-decode` gates byte-identity of the two requests' outputs; the warm second-request decode rate against its reference once filled (reports until then, §8.8) (reports 0.3.0 record (DESIGN §7.0.2ai): 48.0/49.5 t/s decode) (references: decode-warm-2nd 47.6 t/s (report only, not gated); decode-cold-1st 46.0 t/s (report only, not gated); prefill-warm-2nd 508.9 t/s (report only, not gated); prefill-cold-1st 510.5 t/s (report only, not gated)); `coder-served-small-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `agent-dense` gates MTP identity (equivalence's MTP section) and MTP acceptance > 10%; `agent-dense-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `tier-reference-cell` gates tier ON byte-identical to itself across processes and requests, tier OFF likewise, and E2 (reports ON vs OFF identity and the first divergence; every output's hash; the first process's rates beside the second's) (references: decode-warm-2nd-on 18.2 t/s (gate lower-is-worse at 14.8 t/s); decode-ratio-on-off 1.46 ratio (gate lower-is-worse at 1.17 ratio); decode-warm-2nd-off 12.5 t/s (report only, not gated); prefill-warm-2nd-on 27.9 t/s (report only, not gated); prefill-warm-2nd-off 87.2 t/s (report only, not gated); grouped-fallbacks-on 0 count (gate higher-is-worse at 0 count); decode-cold-1st-off 10.2 t/s (report only, not gated); prefill-cold-1st-off 80.5 t/s (report only, not gated); decode-cold-1st-on 13.0 t/s (report only, not gated); prefill-cold-1st-on 24.8 t/s (report only, not gated); decode-cold-warm-ratio-on 1.4 ratio (report only, not gated)); `ngram-determinism-repeat` gates six fresh processes produce identical output; `depth-ladder` gates the load completes at both KV precisions on both cards (reports t/s per card and precision) (references: prefill-large-u8 1025.5 t/s (report only, not gated); decode-large-u8 20.8 t/s (report only, not gated); prefill-large-u8i4 120.9 t/s (report only, not gated); decode-large-u8i4 45.9 t/s (report only, not gated); prefill-small-u8 621.0 t/s (report only, not gated); decode-small-u8 29.2 t/s (report only, not gated); prefill-small-u8i4 170.3 t/s (report only, not gated); decode-small-u8i4 26.1 t/s (report only, not gated)); `sanitizers` gates zero sanitizer reports (reports device-free build only (ARCINT_OPENVINO=OFF): the 4 rope-precision cases gated on ARCINT_OPENVINO are not instrumented by this cell); `package-build` gates the package build succeeds, is version-stamped, and its RPATH probe passes; a post-deploy smoke run follows the install, outside this cell; `pruefstand` gates 10/10 on the production coder artifact through the deployed package (references: score 10 points (gate lower-is-worse at 10 points)) <!-- END GENERATED by tools/acceptance_manifest.py --> |

The long prefills are acceptance work, not repeated per milestone unless the
milestone touches depth. Within one arm, repeated decode samples reuse the
prefix cache rather than re-prefill — a restored continuation is
byte-identical to a cold run by §3.4 — except where the configuration's
history-independence is itself the question, where each sample is cold.

## 6. Why the pipeline layer is rewritten (evidence)

- OV GenAI's continuous-batching path diverges from its stateful path under
  pure greedy — reproduced on this hardware, a measurable quality cost (−2/10
  on the acceptance task), reported upstream (openvino.genai#4367).
- Its own CB-vs-stateful equality test has been `@pytest.mark.skip` since
  2025-02-21 (internal ticket CVS-162891), even for OPT-125M.
- Hybrid-state prefix caching upstream checkpoints at memory-tuned intervals;
  resume between checkpoints is not equivalence-tested.
- The CB admission check refuses the coder on the 16 GiB card outright under
  its own worst-case assumptions, with no numbers; arcint's measured
  reservation serves it (§7.0.2a).
- The model-cache import bug (#37607) and its workaround were found here.

None of this is a reason to abandon OV's kernels — they are the fastest
correct compute on this hardware. It is the reason arcint keeps the compiler
and owns the state.

## 7. Milestones and measured state

Every milestone below met its exit criterion or states why its criterion was
the wrong one. Numbers name card, configuration and evidence class; the dated
measurement of each is the record entry named in §7.0.2.

| # | milestone | standing result |
|---|---|---|
| M0 | skeleton: HTTP, `/health`, `/props`, console | done |
| M1 | single-sequence inference | done — the coder at 51.4 t/s on the stateful executor, 24 GB card |
| M2 | paged KV + GDN rows, chunked prefill | done — the paged executor is the served path; 257,167 tokens loaded on the 24 GB card |
| M3 | prefix caching | done — warm/cold byte-identical, gated |
| M4 | MTP | done — acceptance measured, verification exact; greedy-invariance against plain decoding is not deliverable on this backend (§3.5) and is not claimed |
| M5 | all target models | done — every family serves; the 35B on the 16 GiB card now all-resident (§7.1) |
| M6 | two lanes | done — byte-identical under interleaving on both cards, 10/10 on each lane concurrently (§7.2) |
| M7 | fit pass | done — two-ledger measured reservation; an explicit `--n-ctx` is verify-only (§7.5) |
| M8 | asymmetric KV `u8:i4` | done — +28% context at u8's prefill rate from `+p6` (§7.0.3) |
| M9 | expert offload v2 | done — device-resident slot pool, async uploads; §3.4 closed under the host tier by the static partition (§7.4) |
| M10 | sub-4-bit experts | superseded — the checkpoint's own expert formats served natively (§7.4, §7.8) |
| M11 | drafting II | done — drafter fixes at depth landed; DFlash2 wins at 77k, MTP does not beat plain at depth on the dense agent (§7.6) |
| M12 | dispatch pin, tiled exporter | done — `--pin-dispatch` measured null on a quiet host (opt-in); `--moe-lowering tiled` |
| M13 | vision reserved | done — `--vision` refused, vision IRs reported and never loaded |
| M14 | host compute tier | done — `--moe-cpu-tier` (§7.4) |
| 0.4.x | GGUF opened in process | done — the dense 27B Q4_K_M serves 10/10; the mixed form meets the depth prefill bar (§7.7) |
| 0.5.x | Qwen Flash-Next; native experts; all-resident 35B | Flash-Next serves at full depth (§7.8); the 35B serves all-resident on the 16 GiB card (§7.1) |

### 7.0 What the paged path is worth

Same IR, same card, same compiler, only the pipeline varied: the paged graph's
decode step is 1.68× cheaper than the stateful graph's (11.3 against 19.01 ms
at depth 256 on the 24 GB card, `measured-here`) — the GDN head-major
transposes vanish, the paged GDN kernel is the optimised one, `PagedAttention`
replaces `IndirectSDPA` — and arcint's own serving loop is ~2% of a step. The
served coder decodes at 68.6 t/s (f16 KV) / 71.3 (u8) against the stateful
51.4, and at ~30k depth still 70.1 (no depth collapse), on the 24 GB card
(`measured-here`, 2026-08-29). At 32k context the stateful arcint had already
beaten GenAI's production pipeline 2.8× on prefill and 1.5× on decode on the
same card.

`PERF_COUNT` reports about half a kernel's device time, no transfers, and the
*last* inference; a profile of a past-0 chunk overstates every node share,
because chunk k attends to everything before it. Shares are read from device
timelines (the OpenCL intercept layer), and served rates from the server's own
lines.

### 7.0.1 The retraction rule

A claimed defect or explanation is accepted only with a measurement of its
root cause. A mechanism that was narrated and not measured is **retracted on
the record** rather than edited away, with what the measurement then showed.
The founding instance: the A770's decode and part of the MoE gap were once
explained by PCIe bandwidth. The decomposition falsified it — a resident
model's decode step is its kernels (node time equals the served step; host
work ~2%; the link carries ~1 MB of sliced logits per step, 0.3–1.7% of it,
`measured-here`). The bus is the explanation only where bytes genuinely cross
it: state snapshots on the stateful path, prefix-cache host demotion, expert
offload and the host tier.

### 7.0.2 The two cards, and the record index

**The cards do not run the same kernels.** The plugin selects per generation
(Xe2 has no SIMD8; 2D block loads exist only on Xe2; the matrix unit's
one-row cost differs), so a kernel-level conclusion drawn on one card does not
transfer to the other, and several patches select by architecture. Measured
random-read ceilings over incompressible bytes: 453 GB/s on the 24 GB card,
414–418 GB/s on the 16 GiB card (`measured-here`). Host links: x8 Gen4 at
14.25 GB/s measured DMA on the 24 GB card; x4 Gen3 on the 16 GiB card. **Run-to-run determinism is a per-card property**: the served
Flash-Next path is bit-identical across forwards on the A770 (eight repeats)
and not on the B60, where the GDN state output differs by one f16 ulp on a
fixed head set every forward (§7.0.2cb) — floors and byte-identity claims are
measured on the A770, and a B60 reading names the card.

**The record index.** Each entry of the dated record, with its standing
result. The full entry is on `qfndev`. Entries whose subject is now current
state point to the section that states it.

| § | subject | standing result |
|---|---|---|
| 7.0.2a | admission by measured reservation | every reservation term measured; the refusal carries the terms (§7.5) |
| 7.0.2b | does attention prefill use the matrix unit | yes: `sdpa_micro` prefill kernels carry `dpas` in both KV precisions (IGC shader dump, `measured-here`) |
| 7.0.2c | prefill, first baseline | served prefill is ~99% graph time; the shared-expert gate's M≥128 catalog miss named (see 7.0.2g) |
| 7.0.2d | the profiler's attribution gap | prefill is device-bound (95% busy); decode was host-bound on the stateful graph; `PERF_COUNT` sees ~55% of kernel time |
| 7.0.2e | the 142.74 ms copy | the paged logits slice had cut the wrong axis; fixed and verified at load (§3.2): prefill +27%, activations −2.09 GiB |
| 7.0.2f | decode's host half | the stateful decode was launch-bound in the plugin's per-node host path |
| 7.0.2g | shared-expert gate padding | `--gate-pad 16`: −13% prefill wall, −5% decode on the MoE; off by default |
| 7.0.2h | decode primitive histogram | a handful of op classes, not a long tail; walked primitives cost ~half a launched one |
| 7.0.2i | the padding's decode price | the extraction launch costs ~20 µs a layer however spelled; parked |
| 7.0.2j | the prefix cache in production | offline replay: hits on 97% of turns; pool capacity the lever (§4.4) |
| 7.0.2k | tool-call arguments contract | arguments handed as a mapping when the template asks (§3.7) |
| 7.0.2l | the finer snapshot grid | exact; costs ~0.45 s per continuation, mechanism open (§7.3) |
| 7.0.2m | the think block | the server closes and splits the reasoning (§3.7) |
| 7.0.2n | the MTP head on Intel's Qwen3.8 IR | pairs: 90.8–96.3% acceptance, 10/10 (§7.6) |
| 7.0.2o | the Qwen3.6 MTP head | correct first time (93.9% / 75.4%); loses on the MoE without patch 0003 |
| 7.0.2p | the MoE two-token verify | 20,480 subbuffer creations per verify forward; patch 0003 (§7.6) |
| 7.0.2q | serving defaults | the four-layer operator surface (§3.6) |
| 7.0.2r | DFlash2 | the public head serves as a drafter (§7.6) |
| 7.0.2s | expert offload, first measurement | superseded by 7.0.2v |
| 7.0.2t | the honest reservation | two ledgers: the device working set is charged, the host-mapped slot pool reported (§7.5) |
| 7.0.2u | dispatch pinning; the fusion contract | pinning null on a quiet host; the tiled MoE block is the fused pattern (§7.4) |
| 7.0.2v | the offload dial | the device-resident slot pool and async uploads (patches 0004–0007) make offload a working dial (§7.4) |
| 7.0.2w | asymmetric KV | `u8:i4` serves; two source bugs fixed in patches 0008–0010 (§7.0.3) |
| 7.0.2x | the host compute tier | computing capacity misses on the CPU beats uploading them (§7.4) |
| 7.0.2y | context by the flag | `u8:i4` +28% auto-fit on both served configurations; vision IRs never loaded |
| 7.0.2z | drafting II free levers | Viterbi, λ and longer blocks do not beat the greedy chain; the oracle floors re-rank headroom at +0.74 per cycle |
| 7.0.2aa | long context against the short-prompt numbers | depth tables per model (§7.6, §7.9); the u8:i4 prefill price there is superseded by 7.0.2as |
| 7.0.2ab | the u8:i4 VRAM fault | a depth-scaled scratch buffer the reservation did not charge; charged, then removed by 0020 on the micro-SDPA path (§7.3) |
| 7.0.2ac | patch 0015 | bounded attention partials and the argument-rebind fix; the deep-prompt crash was not this patch's (7.0.2ad) |
| 7.0.2ad | the deep-prompt crash | a driver/runtime interaction (the direct-submission semaphore buffer evicted under VRAM pressure), not a plugin kernel; the default reservation avoids it (§7.10) |
| 7.0.2ae | M9 equivalence; the tier's history dependence | the LRU tier violated §3.4; the static partition (patch 0018) closes it (§3.4) |
| 7.0.2af | the tier's readback | the per-layer "readback" is the host waiting for the GPU to reach the layer's router, not the transfer; `usm_host` destinations (0017) |
| 7.0.2ag | drafters at depth | f16 rotary overflow at 65,504 and the unpaged MTP state; both fixed (§3.5.4, §7.6) |
| 7.0.2ah | M10 re-scoped | the per-expert kernel bypassing the fusion became the native-dispatch route (§7.4) |
| 7.0.2ai | the 0.3.0 release gate | §3.4 holds under the static partition; tier decode holds; its prefill fallback was later split (7.0.2bx) |
| 7.0.2aj | the acceptance target's first run | enumeration corrected; tier ON is gated against itself, not against tier OFF (§3.4) |
| 7.0.2ak | the follow-up window | the runner's request index fixed; a two-process tier-ON divergence observed once, not reproduced (7.0.2al) |
| 7.0.2al | references filled | the acceptance references in §5.1 come from these runners' samples |
| 7.0.2am | turnstile tests | synchronise on tickets, not sleeps |
| 7.0.2an | the round-trip flake | derived ports collided with TIME-WAIT sockets; every test server probes its own port |
| 7.0.2ao | the Prüfstand cell | runs through the run manifest; its score is a gated metric |
| 7.0.2ap | patch 0019 | the prefill fallback's weight-side answer is three-way |
| 7.0.2aq | the tier-ON cold start | load-time probe forwards at tier speed own it; `--fit-ledger-dir` skips them (§7.5) |
| 7.0.2ar | the u8:i4 prefill price | located in the infer wall on the generic kernel; removed by 7.0.2as |
| 7.0.2as | patch 0020 | u8:i4 prefill on micro-SDPA at u8's rate, values still four-bit in VRAM (§7.0.3) |
| 7.0.2at | the scratch charge on micro-SDPA | none allocated; the fit stops charging it from `+p6` (§7.5) |
| 7.0.2au | the depth ladder on `+p6` | green on both cards at both precisions (§7.0.3) |
| 7.0.2av | the u8:i4 chunk ladder | every chunk to 2,048 serves 118k tokens without a fault; the micro-SDPA path's cap is 2,048 (§7.3) |
| 7.0.2aw | 0.3.1 deployment | the agent's pre-0.3.0 context was refused by honest accounting (7.0.2ax) |
| 7.0.2ax | the agent on `u8:i4` | 151,552 tokens with MTP on (§7.9) |
| 7.0.2ay | GGUF stage 1 | a GGUF opens on the served IR and serves 10/10 (§7.7) |
| 7.0.2az | K-quant decode on the matrix unit | a win on Xe-HPG, a loss on Xe2 (patch 0022) |
| 7.0.2ba | the repack at load | K-quant rows in the runtime's compressed form within a measured bound (§7.7) |
| 7.0.2bb | int8-dot K-quant decode | 8/10 on the Prüfstand; not carried; the native decode stays f32-exact |
| 7.0.2bc | the K-quant decode in llama.cpp's shape | patch 0023 |
| 7.0.2bd | root causes after 0023 | the early-execution timing regime named; the first-process stall needs a freshly compiled kernel binary and the driver's direct submission (the mechanism inside the driver unmeasured); every timing ritual warms first |
| 7.0.2be | the mixed form as default; the file's embedding | (§7.7) |
| 7.0.2bf | the same bytes through other stacks | the GGUF default form beats llama.cpp Vulkan on every cell and SYCL at prefill; SYCL level at decode |
| 7.0.2bg | the decode step on the device timeline | every FC kernel within 15% of bandwidth except K-quant Q6_K; the logits slice fixed at a K-quant head |
| 7.0.2bh | Q6_K decode read shape | patch 0024 (fewer messages, a 16 GiB-card gain) |
| 7.0.2bi | Q6_K without shuffles | patch 0025 |
| 7.0.2bj | Q6_K in 224-byte blocks | patch 0026, `--gguf-q6k aligned` |
| 7.0.2bk | the drained fused-op fallback | the runtime fusion check now accepts the K-quant kernel (patch 0027) |
| 7.0.2bl | mins packing | `--gguf-mins` exact / shared / nibble: exactness has a price and the price is a flag |
| 7.0.2bm | tiled A layout | patch 0028 |
| 7.0.2bn | tiled operands | 2D block loads, 64-row tile at 256 registers on Xe2 (patch 0029) |
| 7.0.2bo | the runtime's int4 gemm on the repacked set | at the card's f16 matrix rate; int8 activations dead at 32-wide groups; `--gguf-mins split` |
| 7.0.2bp | tall A reads | patch 0030: 32-row activation reads on Xe2 |
| 7.0.2bq | Q5_K decode row | three levers, three nulls; no patch |
| 7.0.2br | open items after 0.4.4 | the f16 gemm's split-K nondeterminism fixed (patch 0031); the load's parallel repack |
| 7.0.2bs | the short-prompt fault | micro-SDPA's next-K-tile prefetch ran past the buffer; patch 0032 (upstream's fix) |
| 7.0.2bt | kernel review, first pass | the recurrent-state rows zeroed on the device (§3.3) |
| 7.0.2bu | the u8:i4 alternation | the verify pass read four-bit value rows at the f16 row's alignment; patch 0033 |
| 7.0.2bv | small items | `/props` reports the served cache block; 0020's by-token decline lifted after a test-fill fix (0035) |
| 7.0.2bw | the hidden-state tap | MTP on a GGUF-opened model drafts (73.0% accepted) |
| 7.0.2bx | hybrid prefill split | patch 0037: resident experts through the grouped GEMM, the rest on the host tier (§7.4) |
| 7.0.2by | 0037's page fault on Xe2 | the gather ran past its tables; patch 0042 |
| 7.0.2bz | the Flash-Next fill | three provenance defects fixed; full depth answers correctly (§7.8) |
| 7.0.2ca | native expert formats served | the u4 repack residual removed; the KLD residual measured to its mechanism (§7.8) |
| 7.0.2cb | the served path's run-to-run floor | per-card: bit-identical on the A770, one-ulp nondeterministic GDN state output on the B60 (§7.0.2) |
| 7.0.2cc | census-seeded residency | patch 0046 (§7.4) |
| 7.0.2cd | the per-expert native serve | the fault was a one-expert slot pool (patch 0047) |
| 7.0.2ce | the native per-expert route's rate at partial residency | measured before patch 0056; owed a re-measurement (§7.4) |
| 7.0.2cf | card-vs-host divergence on the native route | measured before patch 0056; owed a re-measurement (§3.4) |
| 7.0.2cg | the n-gram table staged per forward | 26.82 GiB off the host ledger (§7.8) |
| 7.0.2ch | the NVMe expert tier | a load-time pinned fill, not a miss tier (§7.8) |
| 7.0.2ci | the full-depth 35B all-resident | serves on the 16 GiB card (§7.1) |
| 7.0.2cj | batched per-expert dispatch | patch 0059 (§7.1) |
| 7.0.2ck | per-expert dispatch grouped by expert | patch 0060 (§7.1) |
| 7.0.2cl | two host terms in the prefill | the logits slice on serving-shape IRs; the CPU embedding table read into memory (§7.1) |
| 7.0.2cm | several rows per load | patch 0061 (§7.1) |
| 7.0.2cn | no speculative readback | patch 0062 (§7.1) |
| 7.0.2co | IQ2_S-packed gate/up on the matrix unit | patch 0064, exact operands (§7.1) |
| 7.0.2cp | the CPU tier decodes a native expert once per call | patch 0065 (§7.8) |
| 7.0.2cq | the CPU tier's dots per AVX2 lane | patch 0066 (§7.8) |
| 7.0.2cr | the 35B decode step | per-layer routing waits were the largest idle term (§7.1) |
| 7.0.2cs | decode routes on the device | patch 0067 (§7.1) |
| 7.0.2ct | the 35B's dense decode GEMMs | below bandwidth in the served graph; open (§7.10) |

### 7.0.3 KV precision on the paged path

`--paged-kv` takes `f16`, `u8`, `i8`, `u4`, `i4` per side; the plugin keeps
the scales. Per-token costs (`measured-here`):

| precision | coder (Qwen3.6 MoE) | dense 27B agent |
|---|---|---|
| `f16` | 20.0 KiB/token | — |
| `u8` (default) | 11.3 KiB/token | 36.2 KiB/token |
| `u8:i4` (u8 keys by channel, i4 values) | 8.8 KiB/token | 28.2 KiB/token |
| `u4` | 6.3 KiB/token | — |

- **`u8` is the default because it is the setting that does not refuse**, not
  because it is fastest: at 262144 tokens on the coder it is 2.83 GiB against
  f16's 5.00, which is what makes two lanes at depth fit on the 24 GB card.
  f16 is faster at depth — u8's prefill rate is 16.5% below f16's at 57.8k
  and 22.1% below at 115.6k, and f16 decodes 7.8% faster at 53.5k; u8 leads
  decode at 32k by 2.5% (24 GB card, coder, `measured-here`). `--paged-kv f16` is for a one-lane deep-context endpoint
  on a card with room, and it spends prefix-cache reserve (§7.5).
- **`u8:i4` buys +28% auto-fit context** on both served configurations
  (coder on the 16 GiB card, dense agent on the 24 GB card), 10/10 on the
  acceptance task. From `+p6` its prefill runs on micro-SDPA with the values
  unpacked in registers (patch 0020) at u8's rate at a held chunk (459 against
  457 t/s at 37.7k, 401 against 398 at 71.7k, 16 GiB card, chunk 128), and the
  depth ladder at 98k is green on both cards at both precisions. Its greedy
  text is not claimed identical to u8's at depth (a four-bit value cache is
  lossier); byte-identity is gated within the precision.
- **`u4` is a capacity lever only**: at 32k its paged attention costs +63% on
  the decode step (−6% decode), `measured-here`.
- Among the asymmetric pairings only `u8:i4` is measured and served; four-bit
  keys with eight-bit values decline micro-SDPA, and a pairing the per-side
  bitwidth audit cannot confirm on the compiled ports is refused at load.

### 7.1 Qwen3.6-35B-A3B on the 16 GiB card

The stock int4 IR is 17.4 GiB and does not fit the 16 GiB card resident. It
serves there two ways:

- **The fused int4 route with expert offload** (`--offload-ratio`, the host
  tier, the static partition; §7.4): ratio 50 with an 8 GiB device pool and
  the tier decodes at 18.2 t/s warm against 12.5 without the tier
  (`measured-here`, the acceptance tier reference cell).
- **All-resident in the checkpoint's own expert formats** — the current best
  configuration. The serving-shape artifact
  `qwen3.6-35b-a3b-native-d40packed-u8` carries the GGUF's IQ2_S gate/up
  blocks verbatim (82 bytes per 256 values, patch 0052) over IQ3_XXS / IQ4_NL
  down, and its dense projections in the plugin's u8 group-16 form
  (`--dense-u8`; the shared expert and attention k/v stay plain, §7.10). It
  loads at 13.11 GiB device-resident and serves with
  `--offload-ratio 0 --moe-per-expert-dispatch --paged-kv u8 --emb-device CPU`
  (the tier is enabled internally and computes nothing, `cpu_tier_pairs=0`).

  | 16 GiB card, all-resident, u8 KV, chunk 1024, plugin 0003–0067 | value | evidence |
  |---|---|---|
  | max context per lane | 112,288 (262144 is not reachable at u8 KV) | `measured-here` |
  | prefill at 4,096 tokens | ~960 t/s | `measured-here` |
  | prefill at 32,768 tokens | 778.9 t/s at chunk 2048, 781.4 at 1024 (plugin 0003–0064) | `measured-here` |
  | decode after 4,096 tokens | 28.1 t/s | `measured-here` |
  | decode at depth 1 | 28.7–30.1 t/s | `measured-here` |
  | Prüfstand | 10/10 (29.4 t/s decode while answering) | `measured-here` |
  | equivalence suite at full depth | all gated checks pass (plugin 0003–0064) | `measured-here` |
  | host peak anonymous memory | ~1.6 GiB | `measured-here` |

  How it got there, each step same-digest: batched per-expert dispatch
  (0059), grouped by expert so each expert's weights are decoded once per tile
  (0060), several output rows per load (0061), no speculative hidden-state
  readback when no expert can miss (0062), IQ2_S-packed gate/up on the matrix
  unit with exact operands — `(2s + 1) · grid · sign` is an integer of
  magnitude ≤ 1,333, exact in f16 — for every call size (0064, Xe-HPG only;
  the B60 keeps the scalar gate/up), and decode routing on the device (0067:
  a kernel writes the pair table from the router's ids, so no layer waits on
  a host readback). The artifact-side levers: the logits slice on a
  serving-shape IR (activations 1366.7 → 359.5 KiB per chunk token) and the
  CPU embedding table read into host memory at load instead of faulted in
  from the mapped file per prompt (+0.92 GiB host).

The same artifact's decode step is now mostly device time, and half of that
is the dense GEMMs, which run below their isolated rate in the served graph
(§7.10).

### 7.2 Two lanes, measured (M6)

The second lane is free in weights and nearly free in activations (0.617 GiB
for lane 0 at a 128-token probe, 0.001–0.003 GiB for the second: the plugin
pools intermediates per compiled model). On the 24 GB card, coder, u8 KV:
single-stream decode 67.6–69.9 t/s at two lanes against 68.8 before M6 (the
bar was a regression under 5%); an agent session at 28.9k context decoding
beside bursts of a 309-token request gets 32.5 t/s while each burst gets
31.1–31.3 t/s at 0.42 s TTFT — 7% of aggregate throughput for sharing, the
session's inter-token stall at p95 17 ms, max one prefill chunk
(`measured-here`). The Prüfstand scores 10/10 on each lane with both running
it at once, all answers byte-identical. Two separate `arcint` processes on one
card under the xe KMD fault the card rather than share it; one process per
card.

### 7.3 Prefill and decode paths

- **Prefill** runs chunked on the absolute grid (§3.2), the chunk the fit's
  choice (§7.5). Attention prefill runs on `sdpa_micro` with `dpas`; the
  mixed stage (chunks after the first) runs on micro-SDPA for `u8` and, from
  `+p6`, for `u8:i4` (patch 0020). The generic paged-attention kernel's
  depth-scaled partial buffers are charged by the fit only on runtimes below
  `+p6` (with a measured chunk cap of 128 there); on micro-SDPA none are
  allocated and the path's measured chunk cap is 2,048.
- **Decode** is one paged forward per token (or one verify forward per draft
  cycle). On the served IR paths it is device-bound; per-step host work is the
  embedding lookup, the index build and sampling, each well under a
  millisecond.
- **The first request of a fresh process** compiles the runtime's kernels for
  its row count (~120–320 ms on the GGUF path) and pays first-execution costs;
  prefill figures here are warm unless marked. A cold artifact read from disk
  is a separate, host-side term (§7.5).
- **The prefix-cache snapshot grid** (`--cache-grid N`) is exact at every cut
  but costs ~0.45 s fixed per continuation plus ~new/1800 s against a
  chunk-aligned hit (24 GB card, coder, `measured-here`); the default stays
  the chunk grid, the mechanism is open.
- **Plugin fixes on these paths that every served number depends on**:
  micro-SDPA's next-K-tile prefetch bound (0032, upstream's fix), the u8:i4
  verify-pass value alignment (0033), the paged-attention argument rebind and
  current-call sizing (0015/0016), the compressed FC's deterministic gemm
  attribute (0031).

### 7.4 MoE routes

Four routes, selected by the artifact and the flags; §3.4 is the constraint
every one of them lives under.

1. **Fused int4, all resident.** The stock IR's tiled MoE block fuses into
   the plugin's `MOECompressed` (one fused op per layer; the positive control
   is the compiled graph's `moe_3gemm_fused_compressed` count). Patch 0003
   removed a per-infer subbuffer churn on multi-token forwards (the MoE's
   two-token verify forward 27.3 → 18.1 ms, byte-identical), which is what
   lets MoE speculation pay (§7.6).
2. **Fused int4 with expert offload** (`--offload-ratio N`: N% of experts on
   disk, LRU slots for the rest). Patches 0004–0007 give the slot pool a
   device-resident tier under a byte budget (`MOE_OTD_DEVICE_POOL_BYTES`, set
   by arcint from `ARCINT_MOE_DEVICE_POOL_BYTES`) and asynchronous batched
   uploads. **The host compute tier** (`--moe-cpu-tier`, patches 0011/0012)
   computes an expert that would evict a slot on the CPU (AVX2, the plugin's
   grouped-int4 layout, f32 accumulation) instead of uploading it. **The static
   partition** (0018) fixes the resident set per layer at bind as a pure
   function of configuration — or of a recorded routing census (0046,
   `MOE_CPU_TIER_SEED`) — which closes §3.4 with the tier on. **The hybrid
   prefill** (0037, fixed by 0042) runs the resident experts through the
   grouped GEMM and the rest on the host tier. Standing, 16 GiB card, 35B
   int4, ratio 50, 8 GiB pool, u8 KV (`measured-here`, the tier reference
   cell, 0.5.0.1 run): decode 18.3 t/s and prefill 27.7 t/s tier ON. Tier OFF
   depends on the host page cache holding the expert bytes: 12.5 t/s decode
   and 87.2 t/s prefill with them cached (misses read at 52 µs), 0.9 and
   12.6 t/s when misses read from disk (about 2.3 ms each, a 16 GiB host
   cache). The tier removes that dependence. With the tier, the host dispatch
   of non-resident experts serialises and owns the prefill gap (§7.10). On the 24 GB card at ratio 99 with the
   tier: 23.6 t/s decode.
3. **Native per-expert dispatch** (`--moe-per-expert-dispatch`, patches
   0038–0041, 0045, 0047, 0050–0067): the fused kernels are bypassed and only
   the routed experts are computed, each by a per-expert kernel that decodes
   the checkpoint's own block format in its inner loop (IQ3_XXS, IQ4_NL,
   IQ4_XS via the IQ4_NL layout, Q8_0, IQ2_S, IQ2_S-packed); an expert without
   a slot goes to the host tier. **All-resident** (`--offload-ratio 0`, 0051 +
   0058: every expert has a slot, slot i is expert i) this is route 4 below.
   **At partial residency** the card-vs-host arithmetic differs (§3.4); the
   rate and divergence readings on IQ4_NL layers taken before patch 0056
   (which fixed an aliased-scale defect) are owed a re-measurement.
4. **The all-resident native pool** — the Qwen3.6-35B route of §7.1, and the
   fastest MoE route on the record.

The CPU tier's native decoders (patch 0043; 0065 decodes each expert row once
per call, 0066 runs the dots one job per AVX2 lane with the multiply and add
kept separate, so a job's bytes do not change) are what route 2 and route 3
use for a miss, and what Flash-Next uses for a quarter of its experts (§7.8).

### 7.5 Fit and reservation

Admission is decided at load by a reservation in which **every term is
measured**, and a configuration that does not fit is refused at startup with
every term spelled out — never discovered at runtime as
`CL_OUT_OF_RESOURCES`:

```
load: reservation: weights+graph 13.11 GiB + drafters 0.00 + expert slots 0.13 (probe-static) + activations 0.31 (all 1 lane, chunk 1024) + margin 0.25 + 1 x (GDN rows 95.6 MiB + KV 11.3 KiB/token) of 15.11 GiB -> max ctx 112288 per lane
```

- **Weights and graph**: the device residency read after compile
  (`GPU_MEMORY_STATISTICS`, `usm_host` excluded), and read again after the
  drafters compile.
- **Activations**: probed by real forwards, climbing by doubling (the peak is
  affine in the chunk, and the plugin's intermediate pool never shrinks, so an
  over-large probe is a permanent tax); the chunk is the knob that buys
  context.
- **Expert slots** under offload: two ledgers — the device working set
  (probed) is charged, the host-mapped slot pool reported and not charged —
  and the driver's deferred commit is audited, never read as free memory.
- **Drafter state**: the MTP layer's unpaged KV (8 KiB per token at f32 on the
  dense model) is a per-token term.
- **GDN rows** per lane, **KV** per token at the served precision, and one
  policy term, `--fit-margin-mib` (default 256).
- **4-bit values below `+p6`** carry a depth-scaled scratch charge and a
  chunk belt; from `+p6` (read off the plugin's build number) neither.
- **`--n-ctx` omitted**: the fit adopts the maximum admissible depth and an
  allocate–audit–replay loop trims it by the measured overshoot; the prefix
  cache then gets no spare pages unless `--prefix-cache-reserve PCT` holds
  some back. **`--n-ctx` given**: verify-only, never lowered; an overshoot
  trims the prefix-cache reserve first and refuses only once it is at zero.
  The context ceiling is simply the depth at which the reserve reaches zero:
  the pool is sized in bytes, live pages are a count.
- **`--fit-ledger-dir PATH`** persists the probed terms per (artifact, device,
  flags, runtime) and skips the load-time probes on a matching key (on
  tier-ON loads the probes are ~170 s of forwards at tier speed); a forced
  debug term is never written to it.
- A **cold artifact on disk** is a host-side term the fit does not own: a file
  cache that several resident models overrun makes the first requests slow;
  it is the same with the tier on or off.

### 7.6 Speculative decoding: MTP, DFlash2, n-gram

Standing numbers, the 24 GB card, the dense Qwen3.8-27B int4, greedy
(`measured-here`):

| configuration | decode | acceptance |
|---|---|---|
| plain, short prompt (32k context, u8 KV, repetition penalty 1.0) | 24.0 t/s | — |
| MTP head (`--mtp on`), same | 33.0 t/s | 76.7% |
| DFlash2 int4 (`--dflash`), same | 44.8 t/s | 3.13 tokens per verify cycle |
| DFlash2 int4, drafter on the 16 GiB card | 39.8 t/s | same, output byte-identical |
| MTP head, own export, short prompt, paged | 36.2 t/s | 93.2%, 10/10 |
| Intel's IR + reconstructed head / + Intel's MTP layer | 36.9–37.3 / 37.7–38.1 t/s against 25.0 plain | 96.3% / 93.9% on code |
| plain at 77k tokens | 15.3 t/s | — |
| DFlash2 at 77k | **18.8 t/s** | 40.6%, ≈ 4.4 tokens per cycle |
| MTP at 77k | 4.9 t/s | 90.8% |

- **DFlash2** is the public block-diffusion head `incoai/Qwen3.8-27B-DFlash2`
  (`tools/export_dflash.py`), seven drafts per verify pass. It wins at depth;
  MTP, capped at two tokens per cycle, does not beat plain decoding at any
  measured long-context depth on this artifact (77k: 4.9 against 15.3 t/s) — serve deep contexts without MTP, or with
  DFlash2.
- **On a GGUF-opened model** the template's MTP head drafts at 73.0% (§7.7).
- **On the MoE** (Qwen3.6-35B stock int4) the head pairs (93.9% / 75.4%) and,
  with patch 0003 and an int4 head, speculation wins on a code prompt (72.9
  t/s against ~62 plain, 84.0% acceptance); not in production.
- **The n-gram drafter** accepts on copy-the-input prompts and nothing on free
  prose; it exists for drafter-less endpoints and the equivalence gate.

### 7.7 The GGUF path

`--gguf FILE --model DIR` takes the served IR of the same architecture as the
topology template and replaces its projections with the file's rows (Q4_K,
Q5_K, Q6_K, Q8_0). The file's geometry is checked against the template's; the
template's tokenizer, chat template, GDN state tensors and MTP layer are
served; its AWQ activation scales are set to one; the token embedding comes
from the file, dequantised on the host per token (`--gguf-embed file`, the
default). Dense models of the allowlisted families.

- **`--gguf-mode mixed`** (default): Q4_K repacked at load into the runtime's
  own compressed form — the mins as exact extra columns, the activation widened
  by its group sums — within a measured bound per weight (1/64 of a
  quantisation step for Q4_K, 1/32 for Q5_K/Q6_K, 1/16 for Q8_0; a tensor over
  its bound refuses the load); Q5_K and Q6_K as the file's rows through the
  K-quant kernel (patches 0021–0030; Q6_K relaid in 224-byte dword-aligned
  blocks by default, `--gguf-q6k aligned`). `repack` and `native` are the other
  forms. The repack's deviation verdict is cached per file between loads
  (`--gguf-check once`).
- **`--gguf-mins`**: `exact` (default), `split` (the min term as a separate
  small matmul; exact-class under a derived bound, a decode lever), `shared` and
  `nibble` (inexact; their deviation is reported, not refused).
- **Activations are f16** on a GGUF-opened model (`--dyn-quant off` there):
  f16 activations reproduce the native path's greedy output byte for byte;
  int8 activations scored 2/10 with the augmented columns.

Standing, the 24 GB card, dense Qwen3.8-27B Unsloth Q4_K_M, u8 KV, one lane,
MTP off (`measured-here`, plugin ≥ 0030/0031):

| form | resident | max ctx at u8 | prefill 856 tok (warm) | decode at 856 | prefill 71.7k | decode at 71.7k |
|---|---|---|---|---|---|---|
| mixed, exact (default) | 16.54 GiB | ~112k | 1,008 t/s | 54.1 ms/step (18.5 t/s) | 464 t/s | 73.0 ms/step (13.7 t/s) |
| mixed, split mins | 16.30 GiB | ~120k | 967 t/s | 53.5 ms/step | 458 t/s | 72.2 ms/step |
| native rows | 15.22 GiB | ~155k | 662 t/s | 51.3 ms/step | 395 t/s | 70.3 ms/step |
| Intel's int4 IR of the same model (reference) | 13.06 GiB | ~213k | 1,609 t/s | 23.1 t/s | 552 t/s | 16.5 t/s |

Prüfstand 10/10 through every form; the equivalence suite passes on the mixed
form. The same bytes through llama.cpp on the same card: SYCL 249 / 206 t/s
prefill at 1k / 10k and 14.2 / 12.4 t/s decode, Vulkan 126 / 108 and 7.8 / 7.0
(`measured-here`). The 16 GiB card cannot hold this model's GGUF.

### 7.8 Qwen3.8 Flash-Next (`qwen4_exp`)

A 48-layer hybrid with 512 routed experts (top-10), hyper-connections, a
per-layer n-gram embedding table (PLE) and sparse attention above a 2,051-token
boundary. arcint serves it from **serving-shape artifacts** emitted from the
GGUF shards by `tools/export_serving_artifact.py`
(`tools/q4e/serving_shape.py`): the GDN, attention, hyper-connection and MoE
blocks in the plugin's fusable patterns, the n-gram table as ports.

- **The fill reads the GGUF correctly**: the converter's folded norm gammas
  (`1 + w`) and `ssm_a = −exp(A_log)` are undone at the feed, the GDN output
  gate is a sigmoid, and value head h reads key head `h % 16` (tiled), all
  config-driven (`output_gate_type`, `gdn_key_head_map`). Layer 0 agrees with
  llama.cpp's own tensors at corr 0.9999.
- **The experts are carried in their own formats** (`--expert-format native`:
  IQ3_XXS / IQ4_XS / IQ4_NL / Q8_0 per layer as the checkpoint ships them),
  lowered by patch 0043 to `MOECompressed` with a format per projection. Against
  the model's own f32 forward at full depth (`tools/ref_forward_stream.py`) the
  native artifact is exact to 0.2% through 24 layers and its short-prompt
  logits sit at KL 0.017 nats (llama.cpp: 0.053); on a 2,735-token window it
  reads mean 0.369 / median 0.181 nats, argmax 0.827, where llama.cpp reads
  0.339 / 0.065 / 0.802 (`measured-here`). The remaining term is a
  long-context floor of the artifact's own (candidates: the f16 recurrent
  state, the prefill chunks, the f16 long-context attention).
- **The n-gram table is staged per forward** from the GGUF (`pread` of exactly
  the rows the forward names) instead of pinned: 2.884 MiB of staging against
  26.82 GiB of USM host, byte-identical answers (depth 4, `measured-here`).
- **The pinned expert set can arrive from NVMe at load** straight into VRAM
  through arcwell's batch DMA (patches 0048/0049, B60 only): cold TTFT 92.5
  against 99.7 s host-fed at depth 4. It is a load-time fill, not a miss tier —
  a layer's routing is visible only inside that layer's own MoE hook, so the
  warning horizon for a fetch is zero layers; the runtime miss tier stays the
  host hop.
- **Served** (`qwen3.8-flash-next-d48n`, 16 GiB card, `--offload-ratio 75
  --moe-cpu-tier --moe-per-expert-dispatch`, u8 KV, chunk 2048, plugin
  0003–0066, `measured-here`): 8.06 GiB device-resident; prefill 15.1 t/s over
  a 32,768-token prompt (14.8 t/s at 4,096); decode **0.5 t/s** — a quarter of
  the experts run on the host tier every token. The rate lever is the resident
  fraction.

### 7.9 Served configurations

The two configurations the project deploys (`contrib/systemd/` carries the
units with operator-local detail removed):

- **The coder** — Qwen3.6-27B-A3B-Coder b5 int4 on the 16 GiB card,
  `--paged-kv u8`, 2 GiB prefix cache, one lane. Prüfstand **10/10** through
  the deployed endpoint. Decode 46.0 t/s cold / 47.6 warm and prefill ~510 t/s
  at ~1k tokens; at 8.9k / 37.7k / 71.7k decode 45.1 / 44.5 / 40.3 t/s; at 98k
  prefill 621 t/s (`measured-here`). On the 24 GB card the same artifact
  decodes 66.5 t/s warm and prefills 2,821 t/s at ~1k.
- **The agent** — Qwen3.8-27B dense, the b7c1 AWQ export with the
  reconstructed MTP head, on the 24 GB card: `--paged-kv u8:i4 --mtp on
  --prefill-chunk 512 --n-ctx 151552`, 8 GiB prefix cache, 4 GiB host tier.
  151,552 tokens is what the honest reservation admits with MTP on at chunk
  512 (§7.5). An 850-token prompt prefills at 1,436 t/s and decodes at
  23.5 t/s (`measured-here`, `+p15`); at depth MTP loses to plain decoding
  (§7.6).

### 7.10 Known open items

- **The dense decode GEMMs of the 35B run below their isolated rate** in the
  served graph (up to 3.7× on the 2048×4096 output projection); the matrix-unit
  gate/up makes the next layer's input projection 1.8× slower, and an
  isolated neighbour kernel can slow the next FC 4.3× on the A770 (not on the
  B60). The mechanism is not measured; hardware counters were not obtained on
  the A770.
- **q, k and v all compressed are fused horizontally by the plugin and serve
  wrong** (KL 2.73 at depth 4); the fused kernel is exact alone. `--dense-u8`
  keeps attention k/v and the shared expert plain until the mechanism is
  found.
- **The tier's prefill on the fused offload route** serialises its host
  experts (27.9 against 87.2 t/s tier OFF).
- **The native dispatch route at partial residency**: quality and rate owed a
  re-measurement after patch 0056.
- **The B60's one-ulp GDN state nondeterminism** (§7.0.2cb): localised to the
  GDN state output, not the kernel choice, JIT, launch geometry or inter-kernel
  ordering; reported upstream as a sibling of openvinotoolkit/openvino#38099.
- **The xe direct-submission fault** (§7.0.2ad): a page-fault storm at the
  OpenCL runtime's own semaphore buffer when it is evicted under VRAM pressure
  with concurrent load. The default reservation keeps the headroom that avoids
  it; `NEOReadDebugKeys=1 EnableDirectSubmission=0` is the documented
  fallback. Upstream: drm/xe issue 8390, a comment on intel/compute-runtime
  issue 948.
- **The hybrid prefill's oneDNN grouped-primitive cache key** is
  routing-dependent (a new prompt can rebuild three primitives per layer);
  bucketing is owed.
- **A two-text alternation at 85 tokens on the GGUF path** (both forms, not the
  IR) is unattributed.
- **Flash-Next's rate** (§7.8) and its long-context KLD floor.

### 7.11 The libllama executor (`--engine llama`)

`src/exec/backend_llama.cpp` runs a GGUF through llama.cpp, pinned at
`bed0a85`, with ggml's OpenCL backend. arcint's Intel kernels are carried as
`contrib/llama.cpp/patches` 0001–0015 (`contrib/llama.cpp/README.md`, one
section per patch): K-quant and IQ matvecs and XMX GEMMs, the gated
delta-net, decode and prompt attention, the few-token verify, the B60's
2D-block GEMM with packed decode and int8 DPAS, searched tiles, and
attention over a quantized KV cache (`--llama-kv q8_0` or `q8_0:q4_0`; 4:4
misses the top-1 bar and is refused).

arcint keeps the HTTP surface, the chat template, the sampler, stop handling
and the lanes. llama.cpp keeps the weights, the tokenizer, the attention KV
and the recurrent state, one sequence per lane. MTP drafts with the GGUF's
own head (`src/exec/llama_spec.cpp`); the verify walk is
`src/exec/verify_walk.h`.

Standing values (`measured-here`, 2026-10-04):
- **Coder, A770:** decode 79.3 t/s with 4 drafts; prefill 1,431 t/s at
  4,096 tokens.
- **Dense 27B, B60:** decode 52.6 t/s with 5 drafts; prefill 935 t/s at
  4,096 tokens; 122,880 tokens of f16 KV without paging. The agent service
  runs it at 131,072 tokens with MTP and an 8:8 cache: peak VRAM 23.06 GB,
  a 128,133-token prompt at 166 t/s then decode at 7.5 t/s, task 10/10. KL
  0.003966 with the 8:8 cache.
- **Answers:** the acceptance task 10/10 at temperature 0 on both. Dense
  KL 0.004034 nats against the CPU reference, against 0.003559 for ggml's
  float kernels on the same card.

Open, and why the coder service stays on the OpenVINO executor:
- the coder's context: its Q4_K_M weights leave ~16k on the A770. Two
  IQ3_XXS mixes fit 98,304 tokens and pass the task, but miss the top-1 bar
  by 3.2 and 2.6 points (KL against Q8_0 logits);
- a shared prefix cache: the agent gave up OpenVINO's 8 GiB prefix cache and
  4 GiB host KV tier for the switch (operator, 2026-10-04).

`docs/llama-engine.md` has the details.

## 8. Open questions and deliberate deferrals

- **A static sequence dimension at decode is worth more than a hand-written
  kernel** (`kernels/README.md`): at dynamic shape the GDN transposes take the
  generic kernel, at static `S = 1` OpenVINO deletes them. The paged path
  collects this win (the transposed subgraph is replaced by the paged
  GDN/conv kernels). A decode-specialised second compiled model is dead:
  constant dedup is per compile, two compiles cost two full weight copies.
  `--custom-kernels` stays as an off-by-default measurement switch.
- **Drafting at depth as a runtime decision**: switch the drafter off per lane
  past the depth where measured acceptance stops paying for the multi-token
  step (per model and card, measured, not assumed).
- **KV codecs beyond u8/i4** (NInfer's int8 group-64 with a fused Hadamard
  pre-rotation): the known upgrade path; any custom-kernel proposal must come
  with a fusion-impact profile of the surrounding graph.
- **Plugin framework gap**: `primitive_inst::realloc_intermediates` can replace
  an intermediate buffer without rebinding kernel arguments; patch 0015 carries
  the local fix for paged attention, the framework-wide fix belongs upstream.
- **Multimodal**: the exports are `*ForConditionalGeneration`; v1 is text-only
  and the request parser rejects non-text content parts rather than dropping
  them.
- **An Anthropic Messages adapter and a local CLI**: small adapters over the
  same executor; not committed.
- **The hybrid-state transfer contract**: if the prefix cache's (paged KV, GDN
  row) pair ever crosses a process boundary, it carries a declared, checkable
  layout contract, not an assumed byte layout.
- **Multi-GPU, tensor parallelism, batching sequences into one graph call**:
  non-goals.
