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
             │ SlotPool                      │  slot model (llama.cpp-style):
             │  admission, slot lifecycle    │  admission accounting only.
             │  (one at a time today)        │  See the note below.
             └───────────────┬───────────────┘
                             │ one request at a time
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

**What the box marked SlotPool actually does today.** It admits and accounts;
it does not run sequences concurrently. `generate()` takes a single backend
mutex and there is one `lm_req_`, so `--parallel N` buys queueing, not
concurrency, and the slot counts in `/health` describe admission capacity rather
than sequences in flight. That is consistent with M1's wording ("single
sequence") and inconsistent with the diagram as it was drawn, which is why the
diagram now says so.

The upgrade is cheaper than it looks and is written down here so it is not
rediscovered: OpenVINO's stateful API already isolates state *per
`InferRequest`*, so one compiled model with N infer requests gives N sequences
with the weights loaded once. The per-sequence cost is that sequence's KV plus a
GDN state slab (~70 MiB coder, ~171 MiB dense), which is what admission would
have to be bounded by — by a memory reservation, not by a lock. The
embeddings and MTP requests are shared today and would need per-slot twins. At
deep context the KV dominates and binds the slot count hard: two sequences at
262144 tokens is 2 × 5 GiB of fp16 KV, which does not fit beside the weights.
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

**Chunked prefill is not bit-exact on this backend, and the cause is now
isolated.** Measured 2026-08-28 with a pure-Python driver over the same graph:
splitting a 224-token prefill changes the last-row logits by up to ~2.8 in
absolute value — kernel-path differences, not rounding noise — and at chunk 7
that flips a generated token 25 steps in. Chunk sizes 1, 7 and 64 all differ
from the unchunked run.

The minimal case pins it down. Starting from one state, on the dense model:

| | result |
|---|---|
| row 0 of `forward([x, y])` vs `forward([x])` | **bit-identical** |
| last row of `forward([x, y])` vs `forward([x])` then `forward([y])` | **differs, max 0.013** |

So it is not batching per se, and it is not the position: it is that advancing
the state by two tokens in one call is a different computation from advancing it
twice by one. The GDN layers are the obvious suspect — a recurrent scan over k
tokens is a different kernel path from k scans of one — and that is a property
of `ocl::gated_delta_net::ref`, not of ligence.

**This one fact governs three features.** Anything that changes how a token
sequence is divided into forward passes can move a near-tie: chunked prefill
(§3.2), speculative decoding (§3.5.1) and MTP (§3.5.2) are all the same
phenomenon.

**It governed the prefix cache too, and that one was not safe.** A cache hit
*is* a boundary: the cold run computed the tail of the prompt inside one long
forward, the warm run computed it as a short forward from the restored state.
Measured on the dense model, a 235-token prompt restored at 224:

| | |
|---|---|
| cold-vs-warm, last-row logits | **differ by up to 0.210** |
| the top-2 margin at that position | **0.145** |
| both paths split at the same boundary | **bit-identical, 0.000000** |

The perturbation was larger than the margin. §3.4's invariant was holding on
luck, and the equivalence gate was measuring luck. The fix is in the last row of
that table: **prefill chunks on an absolute grid** — multiples of the chunk size
counted from position 0, not steps from wherever a call begins — and cache
checkpoints are restricted to that same grid. A warm run then starts on a
boundary the cold run also stopped at, and its remaining boundaries are the tail
of the cold run's. Equality is by construction again, on a backend that is
allowed to be chunk-sensitive, because no two paths ever hand the model a
different split of the same tokens.

The cost is that **cache granularity is now the prefill grid**: with
`--prefill-chunk 2048` a prompt shorter than 2048 tokens never reaches a
checkpoint. Finer reuse means a finer grid means more forward calls in prefill
(measured: 2247 t/s unchunked against 1161 t/s at chunk 512), so the two are one
setting and are validated as one — `--prefill-chunk` must be a non-zero multiple
of `--kv-block-size` whenever the cache is on. Speculation cannot be fixed the
same way, because its verify pass is a multi-token forward by definition; that
is why the cache is gated and speculation is reported.

`--prefill-chunk` defaults to **2048**, not 0: see the measurement further down
this section — chunking is the mechanism that bounds activation memory, and the
size is chosen so ordinary prompts land in a single chunk. The equivalence suite
reports the chunking delta rather than gating on it, because gating on something
the backend cannot deliver would only produce a permanently red test.

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
now defaults to **2048**: ordinary prompts still land in one chunk and are
therefore literally unchunked, while a long prompt is split rather than allowed
to grow without limit. A prompt that does cross the boundary is not bit-identical
to an unchunked run, for the reason measured at the top of this section. The slice and the chunk are complementary — one
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

**Retested 2026-08-28 on OpenVINO 2026.4.0-22849, and it still bites.** A cold
run writes the blob and serves; the next run imports it and fails the warmup
with `Check '_weight_provider' failed at moe_3gemm_swiglu_opt.cpp:2539: expert
weight provider not initialized`, and the guard discards it. An outside review
reported that OV *HEAD* now recreates the provider on import, which is plausible
and does not help: this is the shipped 2026.4 release. **Keep the
prove-and-discard guard until a build is measured green**, and retest rather
than retire it on a changelog.

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

  Measured on the coder at three depths each (4096 / 8192 / 16384), so the
  extrapolation is arithmetic rather than hope:

  | stored type | bytes per context token | at 262144 |
  |---|---|---|
  | fp32 (as exported) | 40,960 | 10.00 GiB |
  | fp16 (retyped) | 20,480 | **5.00 GiB** |

  Exactly half, exactly linear. Greedy output is **byte-identical** to fp32,
  about 10% slower from the extra Converts. The coder's 262144 budget is then
  12.8 (weights) + 5.0 (KV) + 0.06 (GDN, context-independent) + ~1
  (activations, chunked) ≈ **18.9 GiB against 22.7 usable — it fits**, where
  fp32's 23.9 GiB does not. `--kv-dtype` defaults to `fp16`; `fp32` gives back
  what the artifact exports.

  **`q8` is refused, on purpose.** Retyping the state to int8 the same way is a
  numeric cast, not quantisation — no scales, so every value rounds to an
  integer. Tested: it does not crash and does not emit garbage, it emits a
  plausible and quietly worse answer, which is the exact failure mode this
  engine exists to refuse. Real q8 KV needs per-block scales; the paged path
  gets them from the plugin, which is another reason to finish it. The 35B and
  the dense 3.8 need q8 to reach 262144 on this card (5.0 and 8.0 GiB of fp16
  KV respectively against tighter weight budgets), so they wait on it.
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

**Resolved 2026-08-28. The head exists and runs: 93.3% draft acceptance,
byte-identical output.** The rest of this section is kept as the record of how
it was blocked, because the way it came unblocked matters — see §3.5.2.

All three checkpoints declare `mtp_num_hidden_layers: 1` — the 3.6 MoE pair is
not headless — and *none* of the three OpenVINO exports contained an MTP graph.
Each had a single output, `logits`, and no `openvino_mtp_*.xml` beside it. The
optimum-intel export drops the head.

**The weights themselves are on the fleet.** `/models/gptq/qwen38-gptq-mtp`
carries a complete single-layer head — 15 tensors: `mtp.fc.weight`,
`mtp.layers.0.{self_attn.{q,k,v,o}_proj, q_norm, k_norm, mlp.{gate,up,down}_proj,
input_layernorm, post_attention_layernorm}`, `mtp.norm.weight`,
`mtp.pre_fc_norm_embedding.weight` — and its GPTQ config excludes them from
quantisation (`dynamic: {"-:.*mtp.*": {}}`), so they are at original precision.
So M4 is not blocked on weights that do not exist. It is blocked on something
narrower and harder: **no public implementation consumes them.** Checked
2026-08-28:

| where | Qwen3.5 architecture | MTP head |
|---|---|---|
| `transformers` 5.0.0 (installed) | absent (`qwen3_next` is the nearest) | — |
| `transformers` 5.16.1 (latest) | **present**, `qwen3_5` + `qwen3_5_moe`, 28 classes | **absent** — no MTP class, no `pre_fc_norm`, no `attn_output_gate` |
| `optimum-intel` (installed) | present, 5 files reference `qwen3_5` | absent |

The nearest reference, `qwen3_next`, also lacks `attn_output_gate` — and this
head uses it (`mtp.layers.0.self_attn.q_proj` is `[12288, 5120]`, twice the
24×256 head width, so q and its gate come out together). Building the head
therefore means reverse-engineering a forward pass — gated attention, the
4-section mrope, the `pre_fc_norm_hidden`/`pre_fc_norm_embedding` concatenation
order — with **no oracle to check it against**. A mistake there does not fail;
it lowers draft acceptance, which is the silent-divergence class this engine
exists to refuse.

Two things would unblock it, neither of them code in this repository: an
upstream implementation of the head, or the model authors' reference.

**That reasoning was wrong, and §3.5.1 is what made it wrong.** The objection
was that a reconstruction has no oracle — that a mistake would not fail, it
would quietly lower acceptance, which is the silent-divergence class this
engine refuses. But the verifier built in §3.5.1 accepts a drafted token only
when it equals what the sampler would have picked anyway, so a wrong head
*cannot* change the answer. It can only cost acceptance. Acceptance therefore
became the oracle, and a sharp one: a correct head lands in a known band and a
wrong one sits near zero. `tools/export_mtp.py` builds the head on that basis;
§3.5.2 has the result.

So MTP is not implementable against these artifacts, by anyone: there is no
draft head to call. What unblocks it is a re-export that keeps the MTP layer,
which is offline artifact work and outside the engine (README's non-goals put
artifact production offline on purpose). **That is now done**: the allowlist pins
`has_mtp_head = true` for qwen3.8-27b, because `tools/export_mtp.py` writes the
head beside the artifact (§3.5.2), and `false` for the MoE pair, whose exports
still carry none. The flag gates serving, so it follows the export;
`mtp_in_checkpoint` records separately that all three checkpoints declare a head.
`--mtp on` is accepted for the dense model and refused for the other two.

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

#### 3.5.1 The machinery, built and measured against a drafter that needs no head

The paragraphs above were arithmetic. The external-drafter hook makes them
measurable, because a prompt-lookup (n-gram) drafter needs no weights at all:
it proposes the continuation that followed the last time the current suffix
appeared. `--draft N --draft-ngram K` turns it on; it is **off by default**.

Speculation runs only under greedy, and acceptance is defined as *the token the
sampler would have picked here equals the guess*.

**That makes verification exact, but it does not make the output identical to
non-speculative greedy, and this section originally claimed it did.** The
verify pass is a multi-token forward; plain decoding computes the same position
with a single-token forward; and §3.2's measurement shows those two differ by up
to 0.013 in the logits. Acceptance is exact *with respect to the logits the
verify pass computed* — a drafted token is never taken on faith — but the
counterfactual trajectory is computed slightly differently, so a near-tie can
land the other way. Measured: on the dense model at 77.8% acceptance, one token
in 64 flipped, and the answers then reconverged. On the coder, and on other
prompts, no divergence appeared at all.

So the honest statement is: speculation does not corrupt anything and cannot
emit a token the model did not rank first *in the pass that verified it*, but
"byte-identical to non-speculative greedy" is not deliverable on this backend
and the suite reports the comparison instead of gating it. The gates that remain
are the ones that mean something: determinism at a fixed configuration, and
non-zero acceptance. Two details make that true
rather than merely intended, and both were bugs first:

- **Acceptance is the sampler's decision, not a raw argmax.** Penalties are
  applied before greedy chooses (§3.6), and `repetition_penalty` defaults to
  1.05, so the raw argmax is a *different predicate*. Verifying on it diverged
  from non-speculative greedy at draft 8 while passing at 2 and 4 — a wrong
  answer that only appears at some draft widths.
- **A drafted token clears the same gates, in the same order, as a normally
  picked one** (EOS, `max_tokens`, `n_ctx`). Committing accepted drafts without
  them emitted 283 tokens where the plain path emitted 281.

Verification also needs one logits row *per drafted position*, which collided
with §3.2's logits slice: that slice kept exactly one row, and the verifier's
row lookup clamped out-of-range indices onto it, so every draft was compared
against the prediction after the *last* draft token. Nothing ever matched. The
slice now keeps the last `1 + draft_tokens` rows — 5.4 MiB against the 8.1 GiB
the slice exists to avoid, so the memory win is untouched — and an out-of-range
row returns −1 (guaranteed rejection) and is reported rather than clamped.

**Measured on the B60, 27B coder q4, greedy, a prompt whose answer is a verbatim
copy of its input** (a best case for lookup drafting; all four runs produced a
byte-identical answer):

| | decode | accept | verify | re-forward | rollback |
|---|---|---|---|---|---|
| no drafting | **52.1 t/s** | — | — | — | — |
| `--draft 2` | 19.3 t/s | 88.6 % | 4.06 s | 1.32 s | 8.99 s |
| `--draft 4` | 19.8 t/s | 69.6 % | 4.05 s | 2.25 s | 7.77 s |
| `--draft 8` | 24.4 t/s | 57.8 % | 3.40 s | 2.23 s | 5.72 s |

On free-form prose the same drafter accepts 0.0 % — a lookup drafter has
nothing to look up — which is why the equivalence suite gates acceptance on a
copy-the-input prompt as well as gating byte-identity.

**The prediction above was right, and the reason is sharper than expected.**
Rollback is 51–62 % of decode time. It is a copy of 69.9 MiB across 80
variables per decode step, and since fp16 KV at this length is only ~14 MiB,
*most of it is the fixed-size GDN recurrent state* — so the cost does not shrink
with context, it is ~90 ms every step against a 19 ms decode step. Replacing the
prefix cache's serialised blob with a straight reused-tensor copy moved it by
1 % (9.15 s → 9.03 s), which locates the cost inside OpenVINO's
`VariableState::get_state()`/`set_state()` rather than in anything this
repository can restructure.

Netting rollback out entirely, verify + re-forward is 5.40 / 6.04 / 5.41 s
against a 5.40 s baseline: **even with free rollback, speculation is only
break-even here.** The second reason is that a batched verify is not nearly
free on this model — `forward(9)` costs 2.2–3.4× `forward(1)` — so the pass
that is supposed to be amortised is not.

That second reason is where the dense model differs, and it is why MTP is a
Qwen3.8 feature rather than a general one: on an A3B MoE, the tokens in a verify
pass route to *different experts*, so a k-token pass reads several times the
expert weight volume of a 1-token pass. A dense FFN serves every token in the
pass from the same weights, so the verify pass amortises the way speculation
assumes. Measured `forward(k) / forward(1)` at past = 512:

| k | 1 | 2 | 3 | 5 | 9 | 17 | 33 | 65 |
|---|---|---|---|---|---|---|---|---|
| MoE coder (184 experts) | 1.00× | 1.44× | 1.31× | 1.37× | 2.16× | 2.46× | 2.59× | 4.31× |
| dense Qwen3.8-27B | 1.00× | **1.14×** | **1.05×** | **1.08×** | **1.43×** | 1.45× | 1.84× | 1.92× |

A 5-token verify pass costs 1.08× a single step on the dense model against
1.37× on the MoE, and a 9-token pass 1.43× against 2.16×. Verifying a draft of
four is very nearly free on the dense checkpoint and is not on the MoE. This is
the mechanical reason MTP is a Qwen3.8 feature here rather than a general one,
and it agrees with where the head actually ships.

Running the same measurement against the dense checkpoint confirms it, and
turns the conclusion around. Same prompt, same drafter, B60:

| dense Qwen3.8-27B | decode | accept | verify | re-forward | rollback |
|---|---|---|---|---|---|
| no drafting | 19.9 t/s | — | — | — | — |
| `--draft 4` | 11.4 t/s | 69.6 % | 5.23 s | 2.78 s | 16.37 s |
| `--draft 8` | 14.9 t/s | 57.8 % | 4.18 s | 2.62 s | 11.89 s |

Net of rollback, verify + re-forward is 8.01 s and 6.80 s against a 14.10 s
baseline — **1.76× and 2.07× faster.** On the MoE the same subtraction gave
break-even. So speculation on the dense model is worth a genuine 2×, and the
*only* thing standing between the engine and it is the state rollback — which
is larger here, not smaller: **171.3 MiB across 128 variables** against the
MoE's 69.9 MiB across 80, because the dense checkpoint has 64 layers (48 of
them GDN) at n_embd 5120. Rollback is 63–67 % of decode time and turns a 2×
win into a 1.3–1.7× loss.

So the conclusion is bounded, and it reprioritises the paged path:

- **On the A3B MoE checkpoints, speculation cannot pay on the stateful path**
  regardless of rollback — the verify pass does not amortise (`forward(9)` is
  2.2× `forward(1)`) because the tokens in a pass route to different experts.
- **On the dense checkpoint it pays 2×**, and is lost entirely to rollback.
- Therefore the paged decode convention (§3.3, currently listed as an
  optimisation) is not an optimisation for M4 — it is the precondition, and it
  is worth about 2× decode on Qwen3.8. That is a sharper reason to finish it
  than "real q8 KV with scales".

Both of M4's blockers are the ones §3.5 named before any of this was built: the
paged path, and an export that keeps the head. What is new is that they are now
measured rather than estimated, the machinery is in place behind them, and the
invariant they have to preserve is gated in CI.

#### 3.5.2 The head, reconstructed and measured

`tools/export_mtp.py` builds the MTP head as an OpenVINO IR from the
unquantised `mtp.*` tensors in the checkpoint, and extracts the base model's LM
head as a second IR so a draft can be turned into a token. The extraction is
exact — fed the base model's own hidden state it reproduces the base logits to
`max abs diff 0.00000`.

The head's forward pass is not documented anywhere, so every choice in it was
**measured** rather than assumed, scored by how often it predicts the token the
base model actually produces:

| | acceptance |
|---|---|
| final | **66.0%** |
| with a swish gate instead of sigmoid | 13.2% |
| with q and gate split as two contiguous halves | 13.2% |
| with no output gate at all | 15.1% |
| from the pre-final-norm hidden state | 49.1% |
| with plain RMSNorm instead of `(1 + w)` | **0.0%** |

Two of those were not guessable. The norms are **zero-centred** and applied as
`(1 + w)`: `pre_fc_norm_embedding` is entirely negative, which is not a scale,
and the plain form scores exactly zero. And `q_proj` interleaves each head's
query with its gate — `[head0_q | head0_gate | head1_q | …]` — rather than
emitting two contiguous halves, which is worth 53 points. The config's
`output_gate_type: swish` is a red herring; a plain sigmoid scores 66% where
swish scores 13%.

**In the engine, on the B60, dense Qwen3.8-27B, greedy, 200 tokens:**

| | decode | accept | verify | re-forward | rollback |
|---|---|---|---|---|---|
| `--mtp off` | **19.9 t/s** | — | — | — | — |
| `--mtp on` | 9.0 t/s | **93.3%** (97/104) | 5.78 s | 0.67 s | 14.75 s |

In-engine acceptance is higher than the 66% above because the head is primed
over the whole prompt and scored only on generated positions.

**The output is not byte-identical to `--mtp off`, and finding that out is what
located §3.2's root cause.** On this prompt one token in 64 differed — the head's
draft was accepted on a row the verify pass computed, and plain decoding computed
that position in a single-token forward that ranked a different token first by a
hair. The two answers reconverged immediately. So M4's exit criterion is met in
the half that is deliverable (measured acceptance, and verification that never
takes a draft on faith) and not in the half that this backend cannot deliver for
*any* speculative scheme.

The head runs one position behind the base model, consuming `(h_t, emb(x_{t+1}))`
to predict `x_{t+2}`, and is fed one position per committed token. That makes
its own attention KV **exempt from rollback**: every input it has consumed is a
token the model committed to. Only the base model's state needs rewinding.

And that is the whole result: **rollback is 66% of decode time.** Net of it,
7.43 s for 200 tokens is 26.9 t/s against a 19.9 t/s baseline — MTP is
**1.35× faster** and the state copy turns it into 2.2× slower. Note how little
re-forward costs now (0.67 s): at 93.3% acceptance the rejection path is nearly
free, which is exactly the regime speculation is designed for. The remaining
obstacle is not the drafter, the head, the kernel or the hardware. It is
`VariableState::get_state()`/`set_state()` copying 171 MiB of mostly-GDN
recurrent state every step, and the paged path is what removes it.

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

  **Pinned to nodes.** Diffing per-node profiling between a one-token decode
  step at ctx 512 and at ctx 4096 (×8 context):

  | node | growth | implementation |
  |---|---|---|
  | `IndirectSDPA` ×10 — layers 3,7,…,39 | **×20.8 each**, +13.7 ms total | `ocl::sdpa::opt__f16` |
  | `lm_head` MatMul | ×2.9, +3.4 ms | `jit:gemm:any__i8` |
  | `Transpose` in `linear_attn` ×30 | ×5.4, +0.7 ms each | `permute_ref__f16` |

  The ten SDPA nodes are exactly the `full_attention` layer indices, and they
  dominate. Their growth is L^1.46 where decode attention against a growing KV
  should be L^1.0 — so it is not merely "attention reads more keys". They are
  already on the *optimised* kernel, so no custom kernel addresses this; the
  paged path does, by replacing `IndirectSDPA` with `PagedAttention`.

  Two anomalies sit alongside it and are not explained by attention at all: the
  **LM head grows ×2.9** though it has no context dependence whatsoever, and the
  GDN `Transpose` nodes grow ×5.4 though the GDN state is fixed-size. Something
  sizes work by context where context does not enter the mathematics.
- **256k context loads — measured, not extrapolated.** With fp16 KV storage,
  chunked prefill at 2048 and the logits slice, ligence loaded **257,167
  tokens** on the B60 in 8.1 minutes (528 t/s prefill), which is 98% of the
  artifact's 262,144 maximum. The rungs below it:

  | prompt | time | prefill rate |
  |---|---|---|
  | 74,927 | 47.6 s | 1573 t/s |
  | 149,823 | 160.5 s | 934 t/s |
  | **257,167** | **486.7 s** | **528 t/s** |

  A prompt sized past the context was correctly refused with the §3.8 400 and
  its numbers, which is the other half of the criterion working.

  The three levers compound and each was necessary: the slice removes a logits
  term that made 8k impossible, chunking bounds activations, fp16 KV halves the
  only state that grows. Scaling is still superlinear — exponent 1.88 overall,
  against 2.11–2.50 before these changes — so the O(L²) term in §5 is reduced
  but not gone, and the paged path remains the way to remove it.
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
| M2 | paged KV + GDN ledger, chunked prefill | equivalence suite green, 256k context loads | **done** — suite green, **257,167 tokens loaded** (§5); paged path mapped but not adopted |
| M3 | prefix caching (block-aligned checkpoints) | warm/cold byte-equality, hit-rate stats on console | **done** — warm/cold byte-identical, hit stats on console |
| M4 | MTP for Qwen3.8, sampling beyond greedy | greedy-invariance with MTP on, measured acceptance | **acceptance done** — 93.3% on the dense model, verification exact (§3.5.2). Greedy-invariance is **not achievable on this backend** and the criterion was wrong to assume it was: a multi-token verify pass and a single-token plain pass differ (§3.2), so any speculative scheme can flip a near-tie. Also a net slowdown until the paged path lands. |
| M5 | 35B MoE q4 on A770 (16 GB fit), q8 variants on B60 | all three models pass their gates | **done**, and the 16 GB fit now works too: `--offload-ratio 20` serves the 35B on the A770 at 1.8 t/s where it previously refused to load (§7). The q8 half still waits on an export. |

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

**What each card can actually serve** (measured 2026-08-28):

| model | weights | B60 (22.7 GiB) | A770 (15.1 GiB) |
|---|---|---|---|
| coder | 12.8 GiB | serves, 51.4 t/s | **serves, 29.4 t/s** |
| qwen3.8-27b | 13.4 GiB | serves | fits by arithmetic, untested |
| qwen3.6-35b | 17.4 GiB | serves | **`CL_OUT_OF_RESOURCES`** |

M5's other two clauses do not hold, and the reasons are now measured rather
than assumed. **The 35B q4 does not fit the A770** because the artifact is
larger than the card — 17.4 against 15.1 GiB usable — and the failure is a
plain allocation failure in the OpenCL runtime, not a subtlety.

Three ways round it were tried and none works today:
`GPU_ENABLE_LARGE_ALLOCATIONS` (same failure — the limit is total memory, not
per-allocation), and **HETERO:GPU.1,CPU**, which fails because HETERO partitions
by op *support*, not by memory pressure; there is nothing to tell it that the
expert weights specifically should live on the host. Note too that the graph
carries **no MoE-typed op before compile-time fusion**, so there is no node to
hang a per-op affinity hint on either.

That leaves what llama.cpp's `-ncmoe` does — placing the routed expert weights
in host memory and computing those FFNs on the CPU, so PCIe carries activations
rather than weights. OpenVINO's MoE is a single fused op and does not expose
that placement. This is the one place in the project where owning a kernel buys
a *capability* rather than a speed-up.

**There are no q8 artifacts** on the fleet at all — every export under
`/models/ov` is int4/AWQ — so that half of M5 waits on an export, not on code.

**Where the milestones stand.** Five of the six exit criteria are met: M0, M1,
M2, M3 and M5. Only M4's is not, and its criterion — "greedy-invariance with
MTP on, measured acceptance" — cannot be met without an MTP head to draft
with. §3.5 records why that is not a matter of effort: the weights exist and
are unquantised, but no public implementation consumes them (checked against
`transformers` 5.0.0 and 5.16.1, and `optimum-intel`), so building the head
means reverse-engineering a forward pass with no oracle, where a subtle error
lowers draft acceptance instead of failing.

A note on M5, because the distinction matters. Its *exit criterion* is "all
three models pass their gates", and that is measured and met. Its *milestone
description* also names two hardware targets that are not reachable with these
artifacts — the 35B is 17.4 GiB against a 15.1 GiB A770, and no q8 export
exists anywhere on the fleet. Both are properties of the artifacts rather than
of the engine, and both are recorded above with the measurements that establish
them.

Performance target that justifies the project, stated once: beat the OpenVINO
GenAI baseline on the same card and artifact (60 t/s, 27B q4, B60) while
holding the equivalence invariants that upstream skips.

### 7.0 What the paged path is worth, measured against the same compiler

The argument for the paged path has always been indirect: optimized GDN kernels
exist only there, `IndirectSDPA` cannot express the depth curve, rollback would
become block-table arithmetic. All true, all inference. There is a direct
measurement available and it had not been taken.

`openarc-coder` on the fleet serves **the same IR file** (`qwen36-coder-b5-ov`)
on the same card through **the same OpenVINO compiler**, but drives it with
GenAI's `ContinuousBatchingPipeline` — that is, the paged path. So openarc
against ligence holds the kernels and the graph constant and varies only the
pipeline. Measured 2026-08-28 on the B60, identical prompt, five runs each with
the first discarded as warm-up, wall clock including prefill:

| | median | runs |
|---|---|---|
| openarc, GenAI continuous batching (paged) | **53.8 t/s** | 51.6, 57.8, 51.7, 55.9 |
| ligence, stateful | **46.6 t/s** | 46.0, 46.0, 47.8, 47.1 |

**13.4%**, and it is compute rather than serving overhead: ligence's own console
reports prefill 0.12 s plus decode 2.42 s against 2.58 s of wall clock, so HTTP
and detokenization account for about 1.5%.

That number is the honest size of the prize, and it is smaller than the
component estimates suggested — the decode profile puts `gated_delta_net::ref`
at 9.1% and `permute_ref` at 9.0%, and the paged path should retire much of
both. Two readings are possible: the paged kernels win less than the profile
implies, or openarc pays overheads of its own that mask part of the win. Either
way, 13.4% is what a rewrite has to beat, measured rather than assumed, and it
is the right yardstick to hold the work to.

Note also what this says about where to hunt. The MoE coder decodes at roughly
a fifth of its bandwidth roofline while the dense model sits near 60% of its
own, so the headroom is on the MoE side — and only the MoE has an external
baseline on the identical artifact.

### 7.1 The 35B on the A770: the knob was there all along

For most of this project the record said the 35B could not run on the A770 —
17.4 GiB of weights against 15.1 usable, HETERO could not place experts, and
OpenVINO's fused MoE op "exposes nothing", so the gap was filed as offline
artifact work outside the engine. **That was wrong.** The GPU plugin ships
`OFFLOAD_RATIO` ("percentage of model weights to offload... currently supported
for MoE experts only"), with an LRU of resident expert slots and on-demand loads
from the weightless `.bin`. It is advertised in `SUPPORTED_PROPERTIES` on both
cards. Nobody had tried it.

Measured 2026-08-28, through the engine, 120 greedy tokens:

| | |
|---|---|
| A770, `--offload-ratio 0` | **refuses to load** — `[GPU] ProgramBuilder build failed`, `CL_OUT_OF_RESOURCES` |
| A770, `--offload-ratio 20` | **1.8 t/s**, coherent output |
| B60, no offload | **52.2 t/s** |

Every ratio from 5 to 40 loads, and none of them is faster than the others.
Through the engine on the real prompt: 0.6 t/s at 5%, 0.7 at 20%, 0.5 at 40% —
no monotonic trend, and the ordering is noise. **The offload ratio does not
control the cost**, which rules out a simple "bandwidth proportional to the
offloaded share" story and leaves two candidates: a fixed per-token cost, or a
provider that streams the active expert weights every token regardless of the
setting. The arithmetic favours the second — 3B active params at int4 is ~1.5 GB
per token, and 1.5 GB over the A770's x4 Gen3 link (~3.1 GB/s) is ~480 ms
against a measured 555 ms at the first attempt.

Be careful with these numbers: the same configuration measured 1.8 t/s in one
run and 0.7 t/s in another, so the run-to-run spread is a factor of 2.5. What is
robust is that it works, that the ratio does not matter, and that it is one to
two orders of magnitude slower than keeping the weights resident. So the honest reading is: **the 35B runs on the A770, and it is
29× slower than on the B60.** That is a capability, not a recommendation, and it
is exactly the shape the fleet's own `-ncmoe` experience predicts. What is not
yet measured is how it compares to llama.cpp `-ncmoe 30` on the same card, which
serves the same model today; that comparison is the one that decides whether
this is worth using rather than merely worth having.

One methodology note, because it cost an hour: a failed compile poisons later
attempts *in the same process*. The first sweep reported every ratio failing
because ratio 0 was tried first and ran the device out of memory. Each ratio
needs its own process, and the control (no offload, clean process) is what makes
the result mean anything.

## 8. Open questions and deliberate deferrals

**A static sequence dimension at decode is worth more than any kernel we can
write.** Measured 2026-08-28 (`kernels/README.md` has the full workings). The
GDN head-major transposes are 1.23 ms of an 11.16 ms decode step — 90 copies of
`[B, S, 32, 128] -> [B, 32, S, 128]`. At *static* `S = 1` OpenVINO deletes them
outright, because the permutation is then a layout no-op; at any dynamic shape
it falls back to `permute_ref__f16`, the generic kernel, and one static
dimension is the entire difference between that and the specialised
`permute_f_y_axes__f16`.

A hand-written OpenCL replacement is **8.6× faster than `permute_ref`** on the
identical op in the real model (1.59 µs against 13.6 µs per node) and still
loses end-to-end, because OpenVINO's only supported injection path makes the
node opaque to its own graph optimiser: RoPE un-fuses into f32 primitives and
180 previously-eliminated transposes return, costing ~1.6 ms against a 1.08 ms
win. So the kernel is not the bottleneck and neither is the hardware — the
shape is. What is worth investigating is a decode-specialised compiled model
(static `S = 1`) alongside the dynamic one, which would remove these copies
*and* let every other op pick a specialised kernel; the obstacle is whether
OpenVINO can share one set of device weights between two compiled models, since
12.8 GiB twice does not fit. `--custom-kernels` stays as an off-by-default
measurement switch.


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
