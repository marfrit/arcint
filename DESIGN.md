# ligence — Design

Status: **M0 implemented** (serving skeleton on a stub backend, no OpenVINO
linked, no weights loaded); M1 next. Performance numbers cited below come from
the 2026-08 measurement campaigns on the actual target hardware (A770 on PCIe
x4 Gen3, B60 on x8 Gen4, both behind the xe KMD). Model-structure numbers come
from `models/allowlist-raw.json`, read off the IR directories on
`dirac:/models/ov/` on 2026-08-28.

## 1. The one idea

General-purpose engines pay for generality twice on this hardware: once in
kernel quality (Vulkan/SYCL paths that were never tuned for Xe), and once in
pipeline complexity (abstractions for hundreds of architectures the target
machine will never load). NInfer showed on CUDA what specialization buys.
On Intel, the kernel half of that bet is already won by someone else:
OpenVINO's graph compiler and kernel library emit good Xe code (measured:
3.4× llama.cpp-SYCL, 7.6× llama.cpp-Vulkan on Battlemage for the same
checkpoint). What is *not* won is the layer above — scheduling, cache
management, hybrid-state handling, sampling, serving — where OpenVINO GenAI
makes choices ligence cannot live with (see §6).

So: **OpenVINO as compiler and kernel library, ligence as everything else.**

## 2. Target constraints (facts, not choices)

- All three models are **hybrid GDN/attention** transformers. The majority of
  layers carry a fixed-size recurrent state (GatedDeltaNet: conv state +
  delta-rule state per layer); a minority are classic full-attention layers
  with a KV cache. Qwen3.8-27B is dense; the two 3.6 models are MoE with a
  shared expert. The IRs put concrete numbers on "majority" — every one of the
  three reports `full_attention_interval = 4`, i.e. one layer in four is full
  attention:

  | model | IR arch | layers | GDN | attn | n_embd | experts | ctx | weights |
  |---|---|---|---|---|---|---|---|---|
  | Qwen3.6-27B-A3B-Coder | `qwen3_5_moe` | 40 | 30 | 10 | 2048 | 184 (pruned from 256) | 262144 | 12.8 GiB |
  | Qwen3.6-35B-A3B | `qwen3_5_moe` | 40 | 30 | 10 | 2048 | 256 | 262144 | 17.4 GiB |
  | Qwen3.8-27B | `qwen3_5` | 64 | 48 | 16 | 5120 | dense | 262144 | 13.4 GiB |

  All three share one tokenizer (`87a7830d63fcf43b`), which is what makes a
  single tokenizer implementation viable across the whole target set. The two
  3.6 artifacts differ only in expert count: the coder is the 35B with 72 of
  its 256 experts pruned away.
- Recurrent state per sequence is **fixed-size and small** (tens of MB range);
  attention KV grows with context. This inverts the usual memory math: at 256k
  context the KV of the few attention layers dominates, at 4k the weights
  dominate everything.
- q4 (group-quantized int4) and q8 weight formats only. KV cache fp16 or q8.
- One card, one process. 16 GB (A770) is the binding constraint for the 35B
  MoE at q4; 22.7 GB (B60) fits any of the three with room for cache.
- Xe KMD only (i915 legacy paths are dead on the fleet). SYCL runtimes older
  than the xe transition abort on memcpy; this is why OpenVINO's OpenCL/oneDNN
  path is the only proven compute route on both cards simultaneously.

## 3. Architecture

```
             ┌────────────────────────────────────────────────┐
             │ HTTP server (single thread pool, no framework) │
             │  /v1/chat/completions  /v1/completions         │
             │  /health  /props                               │
             └───────────────┬────────────────────────────────┘
                             │ request objects
             ┌───────────────▼───────────────┐
             │ Scheduler                     │  slot model (llama.cpp-style),
             │  admission, batching,         │  continuous batching within a
             │  slot lifecycle               │  strict numerics budget (§6)
             └───────────────┬───────────────┘
                             │ micro-batches
     ┌───────────────────────▼─────────────────────────┐
     │ Executor                                        │
     │  ┌───────────────┐   ┌──────────────────────┐   │
     │  │ Cache manager │   │ OV compiled graph(s) │   │
     │  │  paged KV     │◄─►│  prefill graph       │   │
     │  │  GDN ledger   │   │  decode graph        │   │
     │  │  prefix index │   │  mtp head graph      │   │
     │  └───────────────┘   └──────────────────────┘   │
     │  sampling (greedy, temp/top-p/top-k, penalties) │
     └─────────────────────────────────────────────────┘
```

### 3.1 Model artifacts

Input format is **OpenVINO IR** produced offline (optimum-intel export, int4
AWQ or int8), one directory per model. ligence validates the IR against a
built-in allowlist (architecture hash + tensor inventory) and refuses anything
else. Calibration lessons are encoded in the allowlist: the campaign showed
that scale-estimation (SE) calibration degenerates greedy decoding on the
dense 3.8 (0/10) while AWQ-only stays healthy (7/10) — artifact provenance is
therefore part of the contract, not the user's problem.

The allowlist is compiled in (`src/core/model_registry.cpp`); the values are
transcribed from `models/allowlist-raw.json`, which is the raw IR read and
stays in the tree as the provenance record. Each entry pins the architecture
hash (`lm_xml_sha`), the chat-template hash and the tokenizer hash, plus the
layer geometry and the trained context. A field the raw metadata does not
carry is left unpinned: the artifact's observed value is then recorded and
warned about rather than enforced, and `/props` reports it as `null` instead
of a number. This is why the sampler defaults are still marked
`"provenance": "provisional"` — `allowlist-raw.json` carries no sampler
settings, so those values are inherited from the Qwen3 family card and are not
yet a measured property of any artifact.

### 3.2 Graph strategy

The transformer stack is compiled as **stateless graphs with explicit cache
I/O**: attention layers take KV page tensors in and out, GDN layers take their
recurrent state in and out. ligence owns every byte of cache; OV owns the
math. Three compiled entry points:

**What the artifact actually exports.** The IR is a *stateful* graph, not the
stateless one this section assumes: the language model carries 80 internal
variables — `cache_params.past.{key,value}.N` for the 10 full-attention layers
and `conv`/`ssm` pairs for the 30 GDN layers — and takes `inputs_embeds`,
`attention_mask`, `position_ids` and `beam_idx`, returning `logits`. The export
is a VLM (`Qwen3_5MoeForConditionalGeneration`), so token ids go through a
separate text-embeddings graph first; v1 never compiles the vision graphs.

M1 therefore runs the stateful graph as exported, one sequence per
`InferRequest`. That is not a retreat from ligence owning the cache: the state
is reachable through `ov::VariableState::get_state()`/`set_state()`, which is
exactly the handle the GDN ledger and its block-aligned checkpoints need, and
`ov::pass::SDPAToPagedAttention` is available for the attention side. The
three entry points below describe M2's shape, not M1's:

**Chunked prefill is not bit-exact on this backend.** Measured 2026-08-28 with
a pure-Python driver over the same graph, so this is the backend's property and
not ligence's: splitting a 224-token prefill changes the last-row logits by up
to ~2.8 in absolute value — kernel-path differences, not rounding noise — and
at chunk 7 that flips a generated token 25 steps in. Chunk sizes 1, 7 and 64
all differ from the unchunked run.

§3.4's rule applies to ligence's own configuration first: a path that cannot
meet the equality gate is configured out, not papered over. So `--prefill-chunk`
defaults to **0 (unchunked)**, which is the configuration that satisfies the
gate. The equivalence suite reports the chunking delta rather than gating on it,
because gating on something the backend cannot deliver would only produce a
permanently red test.

**The depth wall was never the KV, and it is now gone.** The graph emits
`logits` for *every* prompt token — `[tokens, 1, 248320]` — so an unchunked 8k
prefill materialised 8.1 GiB of logits on top of 12.8 GiB of weights and the
B60 answered `CL_OUT_OF_RESOURCES` from oneDNN. The attention KV at that depth
is about 335 MiB; it was nowhere near the problem.

Nothing samples those rows. `slice_logits_to_last_token` inserts a `Slice` on
the hidden state immediately before the LM head, so prefill computes one logit
row instead of one per token. It is on by default (`--no-logits-slice` turns it
off, and the equivalence suite uses that switch to *prove* the two agree rather
than assert it). Measured on the coder, same card:

| prefill | result |
|---|---|
| unchunked, no slice | HTTP 500, `CL_OUT_OF_RESOURCES` at ~8k tokens |
| chunked at 512, no slice | 9615 tok in 8.28 s (1161 t/s) — but not bit-exact |
| **unchunked + slice** | 9156 tok at **2247 t/s**; 18303 tok in 17.6 s; 36591 tok in 99.5 s |

That removes the *logits* term. It does not remove the others, and it is worth
being blunt about that, because the first version of this section was: an
unchunked prefill still holds activations for every prompt token at once, so
"deep context" bought that way scales with host RAM. Serving 262144 tokens by
owning enough memory to hold 262144 tokens of activations is not a design, it
is a bigger machine — and it is what took `data` down on 2026-08-28.

**The reference on this fleet already answers this.** `llama-agent.service`
serves 262144 context with the *35B* on a *16 GB* A770:

    -c 262144  -fa on  -ctk q8_0 -ctv q8_0  -ncmoe 30  --parallel 1

Four levers, none of them "more memory": flash attention (attention activations
independent of sequence length), **q8 KV** (a quarter of the fp32 this graph's
state variables use), MoE expert host-spill (§8's `-ncmoe` question, answered in
the affirmative), and llama.cpp's micro-batched prefill at n_ubatch 512, which
bounds activations by the batch rather than by the prompt.

So chunking is not the compromise, it is the mechanism, and `--prefill-chunk`
now defaults to **2048**: ordinary prompts still land in one chunk and stay
bit-identical to an unchunked run, while a long prompt is split rather than
allowed to grow without limit. The slice and the chunk are complementary — one
bounds the logits term, the other bounds the activation term — and what remains
for 262144 on the B60 is the KV term, which is where q8 comes in.

1. **prefill graph** — chunked, variable token count, writes KV pages and GDN
   states.
2. **decode graph** — fixed small shapes (1..k tokens per sequence for MTP
   verification), latency-tuned.
3. **mtp graph** — the native MTP head (Qwen3.8) as a separate small graph.

Compiled blobs are cached on disk. The campaign found the OV default cache
embeds weights in the blob and breaks fused-MoE import ("expert weight
provider not initialized", openvino#37607); the weightless mode with absolute
weight paths was expected to make imports safe (63 s warm vs 156 s cold on the
A770).

**Measured 2026-08-28 on the B60 with the b5 coder artifact: it does not.**
Weightless mode still produced a poisoned blob, and the failure is worse than a
slow start — the import succeeds, so the server comes up healthy and then
throws `expert weight provider not initialized` on the *first infer*, i.e. it
500s every request. Four cases, same artifact and device:

| cache | compile | result |
|---|---|---|
| none | 44.6 s | correct output |
| weightless blob written by an earlier process | 15.6 s | **poisoned** — 500 on first infer |
| weightless, fresh directory | 53.3 s | correct output, no reusable blob written |
| same fresh directory, second run | 50.7 s | correct output, still recompiled |

The only genuinely fast import observed was the broken one. So M1 takes the
position the rest of this document takes everywhere else: a cache that can
change the answer is not a cache. The blob cache is opt-in (`--cache-dir`,
off by default), and whatever it returns is **proven by a real forward pass at
load time** before the server binds its socket. A graph that fails that proof
is discarded and the IR is recompiled with caching switched off, so a poisoned
blob cannot reach a request. The cost of being wrong is ~45 s of startup; the
cost of trusting it is every answer.

### 3.3 Memory: paged KV + GDN ledger

- **KV storage type is now real, and it is what makes long context fit.** The
  artifact exports its key/value state as fp32, which at 262144 tokens is
  ~10.7 GiB against 12.8 GiB of weights — more than a 22.7 GiB card holds.
  OpenVINO's `KV_CACHE_PRECISION` property is *accepted* on the GPU but has no
  effect on this stateful graph (measured 2026-08-28: state stayed fp32 and
  greedy output was byte-identical for both `f16` and `u8`); it governs only the
  paged path. So `store_kv_state_as` changes the stored type directly —
  `Convert` on each key/value variable's initialiser, read and assign, then the
  Variable is relabelled. Compute stays fp32; only storage shrinks, and the GDN
  conv/ssm variables are untouched because they are fixed-size and are not what
  grows.

  Measured on the coder: KV **12.9 → 6.0 MiB** at 306 tokens, greedy output
  **byte-identical** to fp32, about 10% slower from the extra Converts.
  Extrapolated to 262144 tokens that is 10.7 → 5.4 GiB, putting the coder at
  ~19.2 GiB on a 22.7 GiB card. `--kv-dtype` defaults to `fp16`; `fp32` gives
  back what the artifact exports.
- **KV pages**: fixed block size (16 or 32 tokens, benchmark decides), fp16 or
  q8 per config, pool sized at startup from free VRAM after weights. Standard
  vLLM-style block tables, no eviction in v1 (a full pool rejects admission).
- **GDN ledger**: per-sequence fixed-size state slabs, plus **block-aligned
  checkpoints** for prefix caching. A checkpoint is written exactly at every
  KV-block boundary — not at a memory-tuned interval. This is the direct
  answer to OV GenAI's design (checkpoints every `block × multiplier` tokens,
  multiplier ≥ 8, sized for memory): coarse checkpoints force either state
  recomputation or approximate resume. Block-aligned checkpoints make prefix
  reuse *exact* by construction. The cost is bounded: GDN state is small, and
  the checkpoint budget is capped by config (`--gdn-checkpoint-budget`),
  degrading to sparser checkpoints *with mandatory recompute of the gap* —
  never to approximate resume.

**What the paged transformation actually offers.** OpenVINO 2026.4's
`paged_attention_transformation` turns this model's four inputs into 91, and it
does more than the attention side: alongside `key_cache.0..9` / `value_cache.0..9`
and the usual `past_lens` / `subsequence_begins` / `block_indices` family, it
exposes `conv_state_table.0..29` and `gated_delta_state_table.0..29` driven by a
*separate* linear-attention block table — `la.block_indices`,
`la.block_indices_begins`, `la.past_lens`, and `la.cache_interval`. That last
one is precisely the knob this document argues about: upstream sizes the GDN
checkpoint interval for memory, and §3.3's position is that it should equal the
KV block size.

Two facts about it, measured: the GPU plugin picks its own quantised KV layout
(`key_cache [?,2,256,12] u8`, `value_cache [?,2,16,132] u8` — its q8 default,
one of the two dtypes §3.3 allows), and prefill through this interface
reproduces the stateful path's first token exactly. The *decode*-side slot
convention for the `la.*` tables is undocumented and was not reverse-engineered
here: every convention tried produced correct prefill and degenerate decode. So
the paged path is understood and reachable but not yet adopted, and the M1/M2
executor keeps running the graph as exported.

### 3.4 Prefix caching

- Hash chain over token blocks (content hash, not pointer identity), one entry
  per (block hash, position). Collision handling is a first-class test case —
  OV GenAI shipped a collision bug (their #3489); ligence hashes with a keyed
  128-bit hash and verifies token identity on hit before reuse.
- A hit restores: KV pages by reference (copy-on-write) *and* the GDN
  checkpoint at the same block boundary. Both or neither — a prefix hit that
  cannot be satisfied for the GDN side falls back to recompute from the
  longest boundary where both exist.
- **As implemented on the stateful graph, "both or neither" is free.** A
  checkpoint is every one of the graph's 80 variables — the KV of the 10
  attention layers and the conv/ssm pairs of the 30 GDN layers — captured
  through `ov::VariableState`, so there is no way to restore one side without
  the other. Measured on the b5 artifact: 80 tensors, ~75 MiB for a 282-token
  prompt (the GDN half is fixed-size and dominates at short context), 0.08 s to
  snapshot and 0.07 s to restore. Restoring after deliberately poisoning the
  state with unrelated text reproduces the original continuation exactly.
- **Invariant (tested in CI, not aspirational):** for any prompt and any cache
  state, greedy output is byte-identical to a cold run. This is the
  anti-CVS-162891 stance: the equivalence test is the *gate*, and a change
  that breaks it does not merge. Failing kernels or fused paths that cannot
  meet it are configured out, not papered over.

### 3.5 Speculative decoding (MTP)

**Measured 2026-08-28, and it contradicts this section twice.** All three
checkpoints declare `mtp_num_hidden_layers: 1` — the 3.6 MoE pair is not
headless — and *none* of the three OpenVINO exports contains an MTP graph. Each
one has a single output, `logits`, and no `openvino_mtp_*.xml` beside it. The
optimum-intel export drops the head.

So MTP is not implementable against these artifacts, by anyone: there is no
draft head to call. What unblocks it is a re-export that keeps the MTP layer,
which is offline artifact work and outside the engine (README's non-goals put
artifact production offline on purpose). The allowlist now pins
`has_mtp_head = false` for all three — the flag gates serving, so it has to
follow the export — and records `mtp_in_checkpoint` separately so the reason is
visible. `--mtp on` is refused for every model rather than quietly doing
ordinary decoding under a speculative label.

**And even with a head, speculation would not pay on the stateful graph.**
Verifying a draft of k tokens advances the state by k whatever the outcome, so
a partial acceptance needs a rollback. The KV half could be truncated, but the
GDN conv/ssm states are overwritten in place every step and can only be undone
from a snapshot — the same 75 MiB snapshot §3.4 uses, at 0.08 s to take and
0.07 s to restore. Against 19.5 ms per decoded token at 51.4 t/s, that is
**7.7 accepted tokens per verify step just to break even**, well above the 2–4
that MTP heads typically deliver. Speculation would be a net loss.

This is the second thing the paged path buys, and the sharper one: with draft
tokens in scratch pages, rollback is block-table arithmetic instead of a
75 MiB copy. §3.5's own sentence — "draft tokens live in scratch pages and are
promoted only on acceptance" — turns out to be load-bearing rather than an
implementation detail. So M4 depends on M2's paged decode convention as well as
on a re-export.

The plan below stands unchanged for the day both arrive:

- Qwen3.8-27B: native MTP head, greedy draft of n tokens, verified in one
  decode-graph call (draft acceptance measured 45–75 % depending on language
  and domain in the vLLM campaigns; the same head, so similar rates expected).
- The hook accepts an external drafter graph (dflash-style) later.
- MTP interacts with the caches trivially by design: draft tokens live in
  scratch pages and are promoted only on acceptance.

### 3.6 Sampling

Greedy, temperature, top-k, top-p, repetition penalty, presence and frequency
penalties. Host-side in `core/sampler.cpp` (on-device top-k is an optimisation,
not a semantic change, and can come later). Penalties are applied *before* the
greedy decision, so a greedy request with a penalty still feels it. Seeded and
reproducible: an unseeded request is given a fresh seed and that seed is
logged, so any answer can be reproduced exactly. `/props` reports the sampler
defaults and their provenance. Nothing else in v1.

Implemented at M1 rather than M4, because the alternative was worse: the
executor took `argmax` unconditionally while the API happily accepted
`temperature`, which is precisely the silent divergence this engine exists to
refuse.

**Model-aware defaults**: each allowlist entry carries the model card's
recommended sampler settings (the fleet learned this the hard way: greedy
decoding sends some reasoning models into thinking spirals that look like
quant damage). Requests without explicit sampler fields get the card values;
explicit fields always win; `/props` shows both.

### 3.7 Tokenizer, templates, tool calls

- **Tokenizer and chat template ship inside the model artifact** (the OV IR
  directory already carries `openvino_tokenizers` and the jinja template).
  ligence never substitutes its own copy — template drift between exporter
  and server is a measured source of silent quality loss, so the artifact is
  the single source of truth and its template hash is part of the allowlist.
- **Incremental detokenization** for streaming: UTF-8 code points are never
  split across SSE chunks; multi-token characters are held back until
  complete.
- **Stop sequences** (string and token-id) and EOS handling per request;
  `usage` (prompt/completion token counts) in every response, streamed
  responses carry it in the final chunk.
- **Tool-call parsing**: the models' native tool-call format (Qwen XML-style)
  is parsed server-side and returned as structured `tool_calls` with
  `finish_reason: "tool_calls"`, OpenAI-compatible. ligence parses, never
  executes. Requests that declare no `tools` get raw text untouched — the
  fleet's proxy learned that a parser that eats tags on tool-less requests
  makes the content silently vanish.
- **Cancellation**: a dropped client connection aborts the request's GPU work
  at the next scheduler boundary and frees its pages. Agent clients time out
  and retry; a card that keeps computing for a dead socket is wasted joules
  and a blocked slot.

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
| `POST /v1/chat/completions` | OpenAI-compatible; `stream` (SSE) and non-stream; `chat_template_kwargs.enable_thinking` honored (the fleet standard switch) |
| `POST /v1/completions` | raw completion, same sampler surface |
| `GET /health` | 200 + JSON: model, loaded, slots free/total, queue depth |
| `GET /props` | model metadata: context length, quant, block size, KV dtype, GDN checkpoint config, MTP on/off, build info, sampler defaults |

Console output (stderr), llama.cpp tradition. This is what M0 actually prints
(copied from a run on a clean tree; a dirty tree appends `-dirty` to the sha):

```
lgc  boot: ligence 0.0.1 (b0ebc5c1ffcb) Release, GNU 14.2.0
lgc  boot: stub backend: no model, no OpenVINO, synthetic output. Nothing measured here is a model result.
lgc  load: qwen3.6-27b-a3b-coder q4 | 30 GDN + 10 attn layers | n_ctx 262144 (allowlist)
lgc  mem:  kv pool and GDN ledger are not allocated before M2
lgc  http: listening on 127.0.0.1:8090 | 2 slots
lgc  slot 0: prefill    31 tok in  0.00 s (5231184.6 t/s)
lgc  slot 0: decode     48 tok in  0.00 s (4755771.3 t/s)
```

The absurd stub rates are left as they are on purpose: a stub that emits
tokens in microseconds should look like one. Capping the number would make the
skeleton read like a measurement.

The same lines once the executor and the caches exist (M2/M3, **illustrative
formatting — the cache figures are not yet measured**; the model figures are
the real ones from the table in §2):

```
lgc  load: qwen3.6-27b-a3b-coder q4 | 30 GDN + 10 attn layers | weights 12.8 GiB
lgc  mem:  kv pool 4.2 GiB (2688 blocks à 32 tok) | gdn ledger 380 MiB | free 1.9 GiB
lgc  slot 0: prefill  3812 tok in  4.31 s (884.5 t/s) | cache hit 2816 tok (73.9%)
lgc  slot 0: decode    592 tok in  9.86 s ( 60.0 t/s) | mtp accept 61.2%
```

One line per event, greppable, no colors by default, `-v` raises verbosity
(`-v` adds one `http:` line per request, `-vv` is debug).

## 5. Testing and acceptance

- **Prüfstand gate**: the fleet's 10-point code-generation harness runs against
  every artifact/config combination that claims production readiness. The
  reference scores to hold: 10/10 for the coder (B5-class artifact), the 3.8
  artifact must match its GGUF reference before shipping q4.

  **Measured 2026-08-28, b5 coder on the B60.** The bar is written down as
  "10/10 greedy". Under greedy ligence scores **8/10**, deterministically. That
  is not a ligence defect, and the evidence says the bar was never greedy:

  | run | decoding | score |
  |---|---|---|
  | ligence | greedy (temperature 0) | 8/10, byte-identical across repeats |
  | ligence | artifact defaults, seeds 1/2/3 | 10/10, 10/10, 8/10 |
  | OpenArc, same day, same task | its defaults (no temperature field) | 10/10 |

  OpenArc cannot be asked for temperature 0 — `frage.py` records that a temp-0
  call throws in OV GenAI *and* makes OpenArc unload the model, so every
  reference run went through the artifact's sampling defaults (temperature 1.0,
  top_p 0.95, top_k 20, straight out of `generation_config.json`). The stored
  `antwort-b5-greedy.lua` carries no `<think>` block and its first ten lines are
  identical to ligence's greedy answer, which is consistent with the same model
  and the same prompt diverging only where sampling would.

  Three checks say the divergence is the decoder's regime and not ligence's
  arithmetic: the rendered prompt is **byte-identical to reference jinja2** in
  both thinking modes; greedy output is **byte-identical across repeated runs**;
  and greedy output is **byte-identical to an independent implementation** of
  the same graph over the same prompt. So ligence reproduces the reference
  quality under the reference's decoding regime, and the "10/10 greedy" wording
  should be read as "10/10 at the artifact's sampling defaults" until someone
  produces a genuinely greedy 10/10 on this artifact.
- **Equivalence suite**: `tests/equivalence/run.sh`, run where the card is.
  Green on the b5 coder as of 2026-08-28: two greedy runs byte-identical, warm
  prefix cache byte-identical to cold, cache hits reported on the console, and
  a continuation of a cached prompt hitting too. MTP on vs off joins it at M4.

  One line of the original list has been demoted from gate to measurement:
  chunked vs unchunked prefill, which this backend cannot deliver (see §3.2).
  It is reported with numbers on every run rather than asserted, and the
  shipped default is the unchunked configuration that does satisfy equality.
- **Determinism**: two identical greedy runs produce identical bytes (verified
  on the A770/Vulkan agent baseline as achievable on this hardware class).
- **In CI today (M0)**: 96 unit cases plus a 48-check curl round-trip, both
  under `ctest`. They cover the parts of the contract that need no GPU and are
  therefore already gateable — the overflow 400 and its numbers, tool-call
  parsing in both wire forms, the UTF-8 and stop-sequence hold-backs, the
  stream/non-stream equality of generated content, cancellation on client
  disconnect, slot accounting, and the allowlist's refusal of anything outside
  the table. The suite is verified red before green (breaking the UTF-8
  boundary rule fails five cases and exits non-zero), and builds warning-clean
  under `-Wall -Wextra -Wpedantic -Werror`.
- **Host memory is part of the measurement, not a footnote.** Deep-context runs
  are bounded by the *host*, not only by VRAM, and on this fleet the binding
  constraint is ZFS: `zfs_arc_max` is 40 GiB of the 62 GiB on `data`, ARC
  refills to ~24 GiB within minutes of boot because every compile reads a
  12.8 GiB weights file through it, and `MemAvailable` does not count ARC as
  reclaimable. ARC does shrink under pressure, but not fast enough to cover a
  large sudden allocation.

  Measured the hard way on 2026-08-28: a 64k unchunked prefill run alongside a
  resident `openarc-coder` (~26 GiB) and a full 21 GiB **zram** swap — which is
  compressed swap living in RAM, so filling it makes the squeeze worse rather
  than relieving it — produced three global OOMs, killed the engine, and left a
  zombie holding ~41 GiB of shmem that never came back. The same depth run with
  the resident services stopped and ARC at ~12 GiB never dropped below 30 GiB
  free.

  So: stop the resident services before a depth run (they are not just holding
  VRAM), watch host `MemAvailable` and `arcstats` rather than the container's
  view — lxcfs shows a container its cgroup cap, not the host's state — and put
  a watchdog on it. Nothing about this is visible from inside dirac.
- **Sanitizers**: clean under ASan + UBSan with `-fno-sanitize-recover`, unit
  suite and round-trip both, plus a 200-request 24-way concurrency stress
  across 8 slots with every slot exercised and released. ASan has to run on
  dirac: it aborts at startup on the aarch64 build container (allocator
  address-space check), so x86_64 is where that gate lives. It earns its
  keep — the M0 review's use-after-free in the argument parser reproduces as
  a clean ASan report on dirac and disappears with the fix.
- Perf regression tracking against the campaign numbers: B60 ≥ 60 t/s decode
  for the 27B coder q4 (the OpenVINO baseline it must beat to justify its
  existence), A770 ≥ 17 t/s for the dense 3.8.
- **Per-operation profile of a decode step** (`PERF_COUNT`, coder q4, B60,
  2026-08-28). Two findings, one solid and one alarming.

  *Which kernels run.* OpenVINO picks implementations by fixed priority with no
  benchmarking, so a reference kernel on a hot path means every faster candidate
  rejected our shapes. At ctx 512, by share of counted decode time:

  | op | share | implementation |
  |---|---|---|
  | FullyConnectedCompressed | 47% | `jit:gemm:any__i8` (oneDNN JIT) |
  | Transpose | 18% | **`permute_ref__f16` — reference** |
  | GatedDeltaNet | 8% | **`ocl::gated_delta_net::ref___` — reference** |
  | DynamicQuantize | 7% | `dynamic_quantize_gpu_opt` |
  | MOECompressed | 6% | `ocl::moe::moe_3gemm_swiglu_opt` |
  | StridedSlice / Concat / Gather / Range | ~7% | **`*_cpu_impl` — on the host** |

  The plugin's registry confirms this is not misconfiguration: it contains
  `PagedGatedDeltaNetOptImpl` and `PagedGatedDeltaNetRefImpl` but only
  `GatedDeltaNetRefImpl` — **there is no optimised GatedDeltaNet kernel for the
  non-paged path at all**, and that path is 30 of our 40 layers. Roughly a
  third of decode time is in reference or host-side implementations.

  *What scales.* Growing context 512 → 4096 (×8) should leave a one-token decode
  step almost unchanged, except attention, which reads a KV cache that grows
  linearly. Measured instead:

  | op | ×512→4096 | expected | note |
  |---|---|---|---|
  | IndirectSDPA | ×14.6 | ×8 | superlinear |
  | MOECompressed | ×5.8 | ×1 | **context-free by construction** |
  | Transpose | ×4.8 | ×1 | context-free |
  | DynamicQuantize | ×4.3 | ×1 | context-free |
  | GatedDeltaNet | ×3.8 | ×1 | state is fixed-size |
  | RMS | ×3.5 | ×1 | context-free |
  | FullyConnectedCompressed | ×1.7 | ×1 | context-free |

  A mixture-of-experts layer's cost has no dependence on sequence length — it
  sees one token. It cannot legitimately grow 5.8×. Whatever the mechanism,
  **the decode step is doing work proportional to the whole context**, which is
  what the fitted curve in §5 says from the outside: a linear term of 47.5 µs
  per context token and a quadratic term of 28.2 ns per token², crossing over at
  L ≈ 1688. The two independent observations agree.

  Not yet pinned to a node. The next step is `get_runtime_model()` at two depths
  and a diff of per-node `execTimeMcs`, looking for an output shaped `[…, L, L]`
  and for `_cpu_impl` on anything in the attention path.
- **Prefill, measured through ligence on the B60** (coder q4, chunked at 512):
  781–1723 t/s across 1.2k–18k tokens, e.g. 9615 tok in 8.28 s (1161 t/s) and
  18308 tok in 21.1 s (867 t/s). That is one to two orders above the 58 t/s
  llama.cpp-Vulkan figure below, which was the measured pain this project was
  started over. Decode at depth is the open problem instead: 51 t/s at short
  context, but 3.6 t/s at ~2.3k and 0.1 t/s at ~18k on the stateful graph.
  That collapse is what the paged path exists to fix, and it is the strongest
  argument for finishing §3.2's decode-side convention.
- **Prefill is its own bar, not a footnote**: the llama.cpp-Vulkan agent
  measured ~58 t/s prefill on the A770 (six minutes to first token at 21k
  context) while decode quality held (9/10 at 21k depth, failure was an
  ordinary logic slip, not degeneration). Deep-context agent turns die on
  prefill, not decode. ligence tracks prefill t/s per card as a first-class
  regression metric; the OV-compiled prefill graph baseline is measured at M1
  and becomes the floor.

## 6. Why the pipeline layer is rewritten (evidence)

- OV GenAI's continuous-batching path diverges from its stateful path under
  pure greedy — reproduced on fleet hardware, measurable quality cost (−2/10
  on the code eval), reported upstream (openvino.genai#4367, unanswered).
- Their own CB-vs-stateful equality test is `@pytest.mark.skip` since
  2025-02-21 (internal ticket CVS-162891) — even for OPT-125M.
- The in-code admission "not default … due to accuracy issues"
  (Qwen2VL/Gemma3, internal tickets 171180/189844) was deleted rather than
  resolved publicly ("Remove accuracy notes", 2026-08).
- Hybrid-state prefix caching upstream checkpoints at memory-tuned intervals;
  correctness of resume between checkpoints is not equivalence-tested.
- The model cache import bug (#37607) and its workaround were found here, not
  upstream.

None of this is a reason to abandon OV's kernels — they are the fastest
correct compute on this hardware. It is the reason ligence keeps the compiler
and owns the state.

## 7. Milestones

| # | milestone | exit criterion | state |
|---|---|---|---|
| M0 | skeleton: HTTP server, /health, /props, console format | curl round-trip | **done** (`e55e33b`) |
| M1 | single-sequence inference, greedy, 27B coder q4 on B60 | Prüfstand 10/10, ≥ 45 t/s | **done** — 51.4 t/s, 10/10 at artifact sampling defaults (§5) |
| M2 | paged KV + GDN ledger, chunked prefill | equivalence suite green, 256k context loads | partly — suite green, chunking in (not bit-exact, off by default); paged path mapped but not adopted; 256k unproven |
| M3 | prefix caching (block-aligned checkpoints) | warm/cold byte-equality, hit-rate stats on console | **done** — warm/cold byte-identical, hit stats on console |
| M4 | MTP for Qwen3.8, sampling beyond greedy | greedy-invariance with MTP on, measured acceptance | sampling **done** (M1); MTP **blocked twice** — no export contains the head, and rollback on the stateful graph costs 7.7 tokens per verify step (§3.5) |
| M5 | 35B MoE q4 on A770 (16 GB fit), q8 variants on B60 | all three models pass their gates | gates **done** — all three serve and reach 10/10; A770 fit **fails**, no q8 artifacts exist |

M0 went past its exit criterion on purpose: everything that does not need a
GPU was implemented properly rather than stubbed, because that is where the
invariants live and they are cheaper to get right before an executor is
underneath them. What is genuinely absent is OpenVINO, weights, the KV pool,
the GDN ledger, the prefix cache and MTP. Two placeholders are marked as such
in the code and must not survive M1: the stub tokenizer is a reversible
splitter and not a BPE, and `render_chatml_stub` is not the model's chat
template — §3.7 keeps that in the artifact, and M1 takes both from the IR.

**All three models, measured through ligence on the B60, 2026-08-28** (the
Prüfstand task, `enable_thinking` off):

| model | weights | greedy | at the artifact's sampling defaults |
|---|---|---|---|
| qwen3.6-27b-a3b-coder | 12.8 GiB | 8/10 | 10/10, 10/10, 8/10 (seeds 1–3) |
| qwen3.8-27b | 13.4 GiB | **10/10** | — |
| qwen3.6-35b-a3b | 17.4 GiB | **10/10** | 8/10 (seed 1) |

Two things fall out of that table. The "10/10 greedy" bar *is* reachable — for
two of the three models — which sharpens §5's finding: the coder is the one
artifact that needs sampling to get there, not the harness. And the 3.8 is
recorded in the allowlist as "provisional, 7/10", yet scores 10/10 greedy here;
that entry deserves re-measuring rather than being carried forward.

M5's other two clauses do not hold, for artifact reasons rather than engine
ones. **The 35B q4 does not fit the A770**: the artifact is 17.4 GiB against a
16 GB card, and the load fails in the OpenCL runtime. Fitting it needs either
the host-spill tier §8 leaves open or a smaller quant, not a change here.
**There are no q8 artifacts** on the fleet at all — every export under
`/models/ov` is int4/AWQ — so the q8 half of M5 is waiting on an export, not on
code.

Performance target that justifies the project, stated once: beat the OpenVINO
GenAI baseline on the same card and artifact (60 t/s, 27B q4, B60) while
holding the equivalence invariants that upstream skips.

## 8. Open questions and deliberate deferrals

- Exact KV block size (16 vs 32) and q8-KV numerics on Xe — benchmark at M2.
- Whether OV's fused SDPA is exposed cleanly enough for the decode graph, or
  whether the attention layers are better served by the PagedAttention op with
  ligence-owned page tables from day one.
- MoE expert placement on the 16 GB card: full-resident q4 vs a small
  host-spill tier for the 35B (the fleet's `-ncmoe`-style split, measured
  workable under llama.cpp, unproven under OV stateless graphs).
- External drafter (dflash-style) for the 3.6 pair: NInfer ships DFlash with
  draft windows up to 15 for exactly this model — evidence the payoff is
  real. Pulled forward to an M4/M5 decision rather than "someday".
- **Multimodal** (image/video): confirmed — all three IRs export as
  `*ForConditionalGeneration` (`Qwen3_5MoeForConditionalGeneration`,
  `Qwen3_5ForConditionalGeneration`), so the door is open. v1 is text-only to
  keep the core small, and the request parser rejects non-text content parts
  outright rather than dropping them silently. Revisit after M5.
- **Anthropic Messages API** and a **local CLI** (`--prompt`, `--messages
  FILE`): small adapters over the same executor, useful for Claude-family
  clients and offline Prüfstand runs. Candidates for M0.5, not committed.
- **Host-tier cache retention** (spill cold KV pages / GDN checkpoints to
  host RAM for longer-lived prefix reuse): NInfer does device/host state
  retention; on our PCIe x4 A770 the win is unproven. Measure before
  building.
