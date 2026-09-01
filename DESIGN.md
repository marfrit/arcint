# arcint — Design

Status: **M0 implemented** (serving skeleton on a stub backend, no OpenVINO
linked, no weights loaded); M1 next. Performance numbers cited below come from
the 2026-08 measurement campaigns on the actual target hardware (A770 on PCIe
x4 Gen3, B60 on x8 Gen4, both behind the xe KMD). Model-structure numbers come
from `models/allowlist-raw.json`, read off the IR directories on
`/models/ov/` on the dev host, 2026-08-28.

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
makes choices arcint cannot live with (see §6).

So: **OpenVINO as compiler and kernel library, arcint as everything else.**

### 1.1 When a kernel has to change: smallest sufficient divergence

The line above is load-bearing, and the first working kernel patch is what will
test it. Say the position plainly now rather than discover it then.

**A fork is not wanted.** It erodes exactly the boundary this section draws; it
turns the packaged runtime from a repack of an upstream wheel into *our* build;
it puts a rebase on the calendar every time the nightly moves; and it makes every
future measurement a measurement against our tree rather than against a version
anyone else can obtain. That last one matters most here, because the numbers in
this document are its argument, and a number nobody can reproduce is not one.

**And it happens anyway if that is what the work needs.** Almost nobody serves
GDN hybrids on Arc, so nobody optimises these operators for it — `PagedCausalConv1D`
has only a reference implementation, and the shared-expert gate falls to a
reference GEMM at prefill shapes (both §7.0.2c). Waiting for upstream to want
what only we need is not a plan. So the rule is not "no fork" but **smallest
sufficient divergence**, in this order:

1. **Upstream PR.** Written to their conventions from the first line — the impl
   class beside the ref one, the selector condition where selectors live, tests
   in their layout — so it can leave this tree without being rewritten. A merged
   kernel costs nothing to maintain and benefits every Xe user, and this project
   already has standing upstream (issues #4367 and #37607).
2. **A patch series against the pinned version.** If a PR is slow or refused,
   carry it as a numbered patch set applied at build time in the packaging
   repository (`debian/marfrit-openvino`), **not** as a divergent checkout. Each
   patch stays PR-shaped so it can be re-offered; the recipe gains a `patches/`
   directory and the changelog names the upstream discussion each belongs to.
   That is a fork only in the technical sense, and it rebases by re-applying
   rather than by merging.
3. **A maintained fork.** Last resort. It requires a stated reason here *and an
   exit condition* — what would have to become true for us to drop it.

The ordering is not politeness. An upstreamed patch is the cheapest form of the
same code and the only form that does not still need us next year.

**Two consequences to plan for rather than discover.**

- *The supply chain changes shape.* The packaged runtime currently repacks two
  upstream wheels against pinned checksums and builds in seconds. A patched build
  is a real compile, on x86_64, and **the CI has no capability for that target** —
  the same gap that already forces the engine's own package to be built by hand.
  A patched plugin turns that from temporary into structural.

  Stated precisely, because "no runner" would be wrong and would point at the
  wrong fix: an x86_64 runner exists and is registered, but with the single label
  `linux-amd64:host` — host execution on a musl userland, with no container
  runtime installed — so it cannot produce a glibc binary for the target
  distribution, and no job in the packaging workflow targets that label in any
  case (23 jobs on `arch-aarch64`, 5 on `debian-aarch64`, none on
  `linux-amd64`). The gap is therefore a **runner-configuration** one — a
  container runtime and a `debian-amd64` label, or a second container — not
  missing hardware. That is a much smaller thing to fix than it looked, and it
  unblocks both the engine's own package and any patched runtime.
- *Whatever we carry, we publish.* Patches live in the packaging repository where
  anyone can read them, and this document names them and says why. A performance
  number that depends on a patch nobody else has is not a result, it is an
  anecdote.

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
AWQ or int8), one directory per model. arcint validates the IR against a
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
recurrent state in and out. arcint owns every byte of cache; OV owns the
math. Three compiled entry points:

**What the artifact actually exports.** The IR is a *stateful* graph, not the
stateless one this section assumes: the language model carries 80 internal
variables — `cache_params.past.{key,value}.N` for the 10 full-attention layers
and `conv`/`ssm` pairs for the 30 GDN layers — and takes `inputs_embeds`,
`attention_mask`, `position_ids` and `beam_idx`, returning `logits`. The export
is a VLM (`Qwen3_5MoeForConditionalGeneration`), so token ids go through a
separate text-embeddings graph first; v1 never compiles the vision graphs.

M1 therefore runs the stateful graph as exported, one sequence per
`InferRequest`. That is not a retreat from arcint owning the cache: the state
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
of `ocl::gated_delta_net::ref`, not of arcint.

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
- **KV pages**: fixed block size, fp16 or q8 per config, pool sized at startup
  from free VRAM after weights. Standard vLLM-style block tables.

  Two things became concrete at M6. The page size is the **plugin's**, not
  `--kv-block-size`: the transformed graph's `key_cache`/`value_cache` ports are
  laid out in 16-token pages and all the byte arithmetic divides by that.
  `--kv-block-size` governs the *prefix cache's* reuse granularity, which is a
  multiple of it and therefore compatible. And the pool is **refcounted**
  (`src/core/block_pool.h`), because with two lanes a page can be live in one
  sequence, held by a cache entry, and mapped by the other lane at the same
  time; only a reference count can say when it is free. Eviction exists in
  exactly one form: **cached** prefixes are dropped when a live sequence needs
  pages. A live sequence's pages are never taken, and a pool that cannot be
  freed enough ends the request cleanly instead of failing on the card.
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
  OV GenAI shipped a collision bug (their #3489); arcint hashes with a keyed
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
- **On the paged path the two halves live in different places, each for a
  measured reason (M6).** The GDN checkpoint is a fixed-size host blob (~32 MiB
  per row) and travels inside the cache entry. The KV is large and already on
  the card, so it is *not* copied: the entry holds references to the pages
  themselves, and a hit maps them. That is what replaced the single-slot
  pool-epoch tag the C++ port shipped with — an epoch is only a way of noticing
  that the one sequence has overwritten the pool, and with two lanes there is no
  "the one sequence".

  Sharing pages needs no copy-on-write machinery, and the reason is structural
  rather than optimistic: a hit lands on a block boundary by construction, so
  every page it maps is **complete**, and a complete page is never written
  again — the page a sequence writes into is always one it allocated itself,
  with a refcount of one. The backend asserts the alignment rather than assuming
  it; a hit that is not block-aligned falls back to a cold prefill instead of
  writing into someone else's page.
- **Invariant (tested in CI, not aspirational):** for any prompt and any cache
  state, greedy output is byte-identical to a cold run. This is the
  anti-CVS-162891 stance: the equivalence test is the *gate*, and a change
  that breaks it does not merge. Failing kernels or fused paths that cannot
  meet it are configured out, not papered over. M6 adds the second half of the
  same claim: it holds **per lane, with the other lane active**.

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

#### 3.5.3 Speculative decoding on the paged path — built and measured (prototype)

`tools/paged_spec.py` implements the reconstructed GenAI speculative convention
on the dense model, MTP head drafting, and it closes the loop this section has
been circling since M4: **rollback costs zero state bytes.**

The convention, each clause established by a bitwise probe before building:

- `la.cache_interval = [1]`, `la.block_indices = [c, s0..sk-1]` — the pass
  checkpoints the state after every token into successive scratch rows.
- The spec pass computes **bitwise identical logits** to a plain pass over the
  same tokens; the last checkpoint **bitwise equals** the in-place state; the
  committed row is **never written** (m=0 is a strict no-op). Promotion is
  "use the checkpoint row's index next step" — no copy exists to get wrong.
- Attention KV rolls back by `past_lens` arithmetic alone.
- Fresh rows **must be zeroed**: the kernels read the committed row even at
  `past_lens = 0`. A dirty row corrupts the prefill itself — found as
  nondeterminism across runs, cause isolated to reused rows, and this is why
  GenAI zeroes fresh rows.
- What is *not* deliverable, here as on the stateful path: bitwise equality
  against a no-spec baseline. A k-token pass computes bitwise-different state
  than k single-token passes (all 48 GDN tables differ; §3.2). The achievable
  strong gates are mechanism invariance (above), bitwise determinism across
  runs, and warm-restore equality — and all of them hold.

**Measured** (Python driver — orchestration overhead included, so the C++ port
should only improve on this; 120 new tokens; acceptance is on a degenerate
continuation of a random-token prompt, so read the rates, not the 96.7%, which
sits above the 77.8–93.3% natural-prompt band):

| dense Qwen3.8, greedy | depth 512 | depth 4096 | gates |
|---|---|---|---|
| B60, paged, MTP off | 24.0 t/s | 23.6 t/s | all green |
| B60, paged, **MTP on** | **36.1 t/s** (96.7% acc) | **32.1 t/s** | all green |
| B60, stateful engine (baseline) | 19.9 t/s | — | |
| A770, paged, MTP off | 18.1 t/s | 17.7 t/s | all green |
| A770 single-card, MTP on | **refused by reservation**: base 13.59 + embeddings 0.97 + head 1.66 = 15.25 of 15.11 GiB | | |
| A770, **MTP on, head + embeddings on the B60** | **26.6 t/s** (96.7% acc) | **22.4 t/s** | all green |

**1.81× over the stateful baseline at depth 512, 1.61× at 4096** — the kickoff
expectation was 35–40 t/s on the B60 and the measurement landed at 36.1/32.1.
The 512→4096 slowdown is fully accounted, not narrated: +0.23 s in the base
verify passes (paged attention over more keys, matching MTP-off's proportional
drift) and +0.18 s in the head itself (its own attention over an 8× longer
primed KV). The warm-restore gate doubles as the paged prefix-cache primitive:
restore the committed row, reuse the untouched prompt KV blocks, re-prime the
head — tokens *and* final state bitwise-equal a cold run, with speculation on.

On the A770 alone the head does not fit beside the dense model with these
artifacts, and the reservation says so with numbers instead of
`CL_OUT_OF_RESOURCES` mid-request. But the box has two cards, and the head is a
separate graph glued through host memory: per step only a 5120-float hidden row
and a token embedding cross, ~20 KB each — weights stay put, activations
travel. With the head and the embeddings gather on the **B60** (1.66 + 0.97 GiB,
which fits beside the coder production's 13.3 of 22.7), the A770 serves dense
MTP at **26.6 t/s** — 1.47× its own paged baseline, 1.54× its stateful one —
and the head costs the same there as it does locally (0.56 s vs 0.52 s per 120
tokens; the cross-card hop is invisible next to the infer itself). The
embeddings placement matters more than expected: the CPU gather cost 0.38 s per
120 tokens against 0.02 on a GPU, an 8% swing on its own.

Two more A770 lessons for the port: compile the big model *first* (its
compile-time peak on top of a resident embeddings model OOMs where the reverse
order fits), and a single-card reservation refusal is not the end of the
answer when the box has a second card with room on it.

What the C++ port inherits from the prototype as requirements: zeroed rows,
device-resident tables set once, compile ordering, the reservation, the
three-row rotation, and the head fed committed tokens only.

#### 3.5.4 The C++ port: the measured path is the served path (2026-08-29)

The paged executor is the default serving path; `--no-paged` keeps the stateful
executor as the reference implementation the suite compares against. Everything
the prototype proved rides along as an invariant: zeroed rows, device-resident
tables set once, big-model-first compile ordering, `--emb-device`/`--mtp-device`
for parking the gather and the head on the other card, measured-reservation
admission with the refusal carrying the numbers, and rollback as checkpoint-row
promotion — the console prints `re-forward 0.00 s, rollback 0.00 s` because
neither exists here.

Two behaviours were added beyond the prototype, both forced by the A770:

- **The reservation probes small and extrapolates.** A probe at the configured
  chunk can itself OOM (observed: sometimes the driver spills, sometimes
  `CL_OUT_OF_RESOURCES` kills the process — the same borderline nondeterminism
  §7.1 met). The peak is linear in the chunk, so a 128-token probe fixes the
  slope, the largest chunk that admits the requested n_ctx is computed, and one
  guarded probe verifies it, stepping down on failure.
- **The chunk shrinks itself before the engine refuses.** The chunk is the knob
  that buys context; refusal is what remains at the floor.

The paged prefix-cache blob is one LA row plus the head's variables, cursor and
pending row — no KV copy. A pool-epoch tag makes that honest with one slot: a
cold prefill rewrites the pool and bumps the epoch, so an entry from an older
lineage is a miss instead of a wrong answer. Block-refcount multi-entry caching
is M6's business.

**Gates**: the full suite is green on the coder (B60 and A770) and the dense
model (B60), under the served default. Stateful-vs-paged is compared and
recorded per run (coder B60: byte-identical; dense: differs — near-tie class).
The dense-on-A770 suite is inadmissible at the suite's fixed `--n-ctx 8192`
(64 KiB/token of KV beside 13.59 GiB of weights admits ctx 6512), which is the
reservation stating a fact about the card, not a failure.

**Bars, measured against their stated values:**

| bar | stated | measured |
|---|---|---|
| B60 coder decode | ≥ 64.5 t/s (the Python driver) | **68.6 t/s** (71.3 under u8) — the founding 60 t/s bar falls with it |
| B60 coder at ~30k depth | — | **70.1 t/s**: the depth collapse is gone from the served path |
| A770 dense + MTP (head on B60) | ≥ 26.6 t/s | 24.9 t/s at 86.2% acceptance — **per-pass cost 74.2 ms vs the oracle's 74.0**; the delta is acceptance (the bar was set on a 96.7%-acceptance degenerate prompt), not machinery |
| prefill | ≥ 586–901 t/s band | 1969 t/s (B60, 30k prompt), 625 t/s (A770 dense chunk 512) |
| dense B60 + MTP | — | 36.2 t/s at 93.2% acceptance |

**Harness verdicts through the served endpoint** (the quality half this port
unblocks): coder **10/10** at base depth and **10/10 at the ~30k depth probe**;
u8 KV **10/10 at both depths** with the base answer *bitwise identical* to f16
and never slower — so **u8 is the paged default** per §7.0.3's protocol, halving
KV memory, with `ARCINT_PAGED_KV=f16` as the pin for A/B runs. The dense model
re-measured **10/10 greedy** through paged+MTP (36.2 t/s), superseding the
registry's stateful-era 8/10 — greedy is deterministic per configuration, so
both numbers are real; the paged path's near-tie landings score better here.

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
  arcint never substitutes its own copy — template drift between exporter
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
  `finish_reason: "tool_calls"`, OpenAI-compatible. arcint parses, never
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
| `GET /v1/models` | the one served model, with the context it is **running with** (§4.2) |

Console output (stderr), llama.cpp tradition. This is what M0 actually prints
(copied from a run on a clean tree; a dirty tree appends `-dirty` to the sha):

```
lgc  boot: arcint 0.0.1 (b0ebc5c1ffcb) Release, GNU 14.2.0
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

### 4.1 Two lanes (M6)

The use case that defines this: an agent session is mid-decode on a 30k
context, a subagent fires one request at the same model, and neither queues
behind the other or changes the other's bytes. `--parallel 2` is the product.
N stays configurable, everything below is gated at 2.

**A lane is one sequence's worth of mutable state, and nothing in it is
shared.** Per lane: an `InferRequest` for the language model, one for the
embeddings gather, two for the MTP head, its own GDN checkpoint rows, its own
KV block table, its own logits buffer. What is shared is either immutable or
refcounted: the compiled models, and the KV page pool. That split is what makes
a second lane affordable — **weights are shared between `InferRequest`s of one
`CompiledModel`, and only between those**: two *compiles* of one graph cost
0.791 → 1.582 GiB (§7.0.2), two *requests* cost the activations, and on this
plugin not even those (below).

**Sequences are never mixed inside one graph execution.** Each lane steps its
own request over its own rows and pages; there is no batching of two sequences
into one call. That is the rule all three reference engines agree on for hybrid
GDN models, and it is also the only shape under which the equivalence
invariants of §3.4 can survive, since a batched call changes the arithmetic of
every sequence in it.

**The scheduler is a ticket lock, and it is a correctness mechanism before it
is a fairness one.** `Turnstile` (src/core/turnstile.h) orders graph
executions: whoever asks first runs first. Three things follow.

1. *Bounded stall.* A decode step waits for at most one other execution, so the
   worst inter-token stall on the busy lane is one prefill chunk, not an
   unbounded run of them. Left to the plugin's own unordered lock no bound can
   be stated at all.
2. *Correctness.* The GPU plugin pools intermediate buffers **per compiled
   model, not per request** (measured: a second `InferRequest` adds 0.00 GiB,
   §7.2). Two concurrent executions would therefore write over each other's
   intermediates. The turnstile is what makes that impossible, and the same
   measurement is why each lane copies its logits and hidden state out of the
   request *inside* its turn — a request's own output tensor is only valid
   until the next execution on that model, by anyone. That applies to **every**
   shared compiled model, not just the big one: the embeddings gather and the
   MTP head are shared too, and their output is read the same way, so they take
   turns as well.
3. *Measurement.* The wait is timed where it happens, so the console reports
   what the other lane actually cost this one, as a p95 over decode steps
   rather than a mean: one long stall inside many short steps is exactly what a
   mean hides.

```
lgc  slot 1: prefill   309 tok in  0.42 s (734.5 t/s)
lgc  slot 0: decode    400 tok in 12.29 s ( 32.5 t/s) | graph 6.39 s, embed 0.05 s, sample 0.05 s, emit 0.09 s, wait 5.72 s, other 0.00 s | stall p95 17 ms max 516 ms (246 steps, 5.72 s total)
```

Waiting is its own term on that line, not folded into whichever phase happened
to block: a step that spent 10 ms queueing did not spend 10 ms gathering
embeddings, and a breakdown that says it did sends the next person profiling the
wrong kernel.

**The prefill grid is configuration, never a scheduling variable.** It would be
easy to shrink a prefilling lane's chunk under contention to cut the other
lane's stall. It is also forbidden: chunk boundaries are not bit-exact on this
backend (§3.2), so a grid that depended on what the other lane happened to be
doing would make a warm run diverge from a cold one for reasons no one could
reproduce. The stall bound is therefore a property of `--prefill-chunk`, which
is the honest place for the operator to trade it.

### 4.2 The name, and the context, are contracts with a proxy

Two facts about how this endpoint is consumed, both learned by breaking them.

**The name.** A discovering proxy reads model names from the backend's
`/v1/models` and republishes them, so whatever id appears there is what every
client downstream must send. Pinning that id to the allowlist's canonical name
therefore makes the allowlist a client-visible API: swapping one engine for
another on the same card and the same artifact renamed the endpoint, and every
caller sending the old name got nothing. So the presented name is its own knob:

- `--served-model-name NAME` sets what `/v1/models` (`data[].id`) and `/props`
  (`model.id`) report, and what a completion echoes back in its `model` field.
  Presentation only.
- `--model-id` is untouched by it. That flag is the *artifact* assertion — which
  checkpoint this process will accept — and it keeps refusing anything outside
  the allowlist whatever the endpoint is called. The two answer different
  questions and must not be one flag, which is what abusing `--model-id` as an
  alias source made them.
- Absent the flag, the canonical id is served exactly as before.
- Both names are recognised in a request's `model` field, and neither is
  enforced: one process serves one model, so there is nothing else a request
  could mean. `/props` says so rather than leaving it to be inferred —
  `model.answers_to` lists them and `model.enforces_model_field` is `false`.
- `model.canonical_id` is always published beside `model.id`, so a rename never
  costs artifact identity. The model registry stays keyed by the canonical id;
  a renamed model keeps its allowlist metadata.

**The context.** The same proxy takes the context length from the `/v1/models`
entry — trying `max_model_len`, `context_length`, `ctx`, `n_ctx`,
`max_context_length` — and asks `/props` only for template capabilities. A
context published on `/props` alone therefore reaches no client at all: they
fall back to their own defaults against a server configured for 262144. So the
model object carries it:

```json
{"id": "qwen3.6-coder", "object": "model", "owned_by": "arcint",
 "n_ctx": 262144, "n_ctx_train": 262144, "quant": "q4", "lanes": 1,
 "canonical_id": "qwen3.6-27b-a3b-coder"}
```

`n_ctx` is what this process is **running with**, which is the number a client
needs; `n_ctx_train` is the artifact's ceiling. They are different fields
because they are different facts — a server at `--n-ctx 40960` on a 262144
artifact must not report the ceiling — and a caller that wants to know the
headroom can see both. `quant` and `lanes` ride along because the proxy keeps
the whole object and they cost nothing.

### 4.3 Admission: a lane is a memory reservation

`--parallel N` is not a queue depth, it is a claim about memory: the startup
arithmetic of §7.0.2a reserves activations, GDN checkpoint rows and KV for N
concurrent sequences at the requested `n_ctx`. An N+1st sequence has nowhere to
live, so it is **refused with those numbers** — HTTP 503 carrying the same
terms `/props` publishes — rather than queued behind a session that may decode
for minutes, which a client cannot tell from a hang. `--queue-timeout S` (0 by
default) restores waiting for deployments that prefer it; `/health` reports the
queue depth either way.

The refusal happens before a single response byte is committed, which is the
only place a status code can still be chosen: once an SSE body has started, a
failure can only be a message inside a 200.

The default is a **behaviour change** and is worth stating as one: before M6 a
second concurrent request queued, and now it is refused unless a timeout says
otherwise. That is right for the engine — the number is a memory claim — and
wrong for a service endpoint whose callers are OpenAI-compatible clients, most
of which do not retry a 503. So `packaging/arcint.service` passes
`--queue-timeout 30`, and the two decisions stay separate: the engine tells the
truth, the deployment chooses the manners.

KV pages are the other half. The pool is refcounted (`src/core/block_pool.h`):
a page is handed out with one reference, gains one for every sequence or cache
entry that maps it, and returns to the free list when the last one goes. When a
lane needs a page and the pool is dry, **cached prefixes are dropped first** —
a cached page is reclaimable, a live sequence's is not — and only if that is
not enough does the request end, cleanly, rather than as an allocation failure
on the card.


### 4.4 A host tier for evicted prefixes (2026-08-30: designed, implemented, gated)

**Why.** The replay of real sessions (§7.0.2j) puts 37% of the agent's prefill
work into re-prefilling sessions the pool could not hold: 82 misses averaging
107k tokens, ~35 s each at 3000 t/s, because the operator interleaves long
sessions and the 377k-token pool holds two of them, not three. The pool total
is fixed by the card; the n_ctx/spare split does not change it. Host RAM is
48 GB on the dev host and the link moves 14.25 GB/s measured (§7.0.2e): a
150k-token prefix is 1.7 GB of u8 KV and comes back in ~0.12 s.

**What.** An evicted entry is *demoted*, not dropped: its KV pages are copied
to a host buffer, its page references released, and the entry stays in the LRU
list marked tiered, with its GDN row (already host-resident) untouched. A hit
on a tiered entry *promotes* it: pages are allocated (evicting — demoting —
further LRU entries if needed), the host buffer is copied back into them, and
the ordinary restore follows. If pages cannot be found even after that, the
hit degrades to a cold prefill, which is what happens today.

**Invariant.** Pages come back byte-exact, so a promoted entry is
indistinguishable from one that was never evicted, and §3.4's warm-equals-cold
gate holds by construction. The gate for the feature is the same gate under a
pool small enough to force demotion and promotion between the two requests.

**Budget.** `--cache-host-mib N`, 0 = off (the default until measured). The
byte budget counts the host KV buffers; LRU order is shared with the resident
entries, so a tiered entry ages out of the host tier the same way it aged out
of the pool.

**Copy path, read against the pinned runtime.** `ov::RemoteTensor` offers
whole-tensor `copy_to`/`copy_from` only; the ROI constructor exists on
`ov::Tensor`. A page is 16 tokens across 20 pool tensors (10 attention layers,
K and V) — ~9 KiB per tensor per page at u8, 181 KiB per page in all. Copying
page by page would be ~187k calls for a 150k-token prefix and the call cost,
not the bytes, would dominate. So the unit of copy is a **run** of pages
contiguous in the pool: the allocator hands out ascending free ids, so a
prefix filled in one prefill is mostly a few long runs; after churn it
fragments, and the design accepts that as a measured cost before reaching for
a gather kernel through the `--custom-kernels` seam (one launch per tensor
into a staging buffer, then one copy — the fallback if fragmentation makes run
copies slow).

**Interaction with copy-on-write.** Demotion releases only the entry's own
references; pages still shared with the live lane or with a resident ancestor
stay resident, and promotion first tries to re-share those rather than copy
them back. The first implementation copies everything and measures; sharing
on promotion is an optimisation with a number attached later.

**Implemented the same day, and gated.** `--cache-host-mib N` (0 = off).
Demotion copies an entry's pages to host buffers by page runs through
`RemoteTensor` ROI views and releases them; promotion allocates, copies back
and records the pages; one LRU order across tiers; `/health` carries
`cache.{entries,tiered_entries,host_mib,hits,demotions,promotions}`; the hit
line prints `from host tier in X s`. `--kv-pool-pages N` caps the pool so a
test can force eviction at a small context. The gate, on the B60 with the
coder artifact, three distinct ~5.8k-token prompts into a 514-page pool:

| | tier on (4096 MiB) | tier off |
|---|---|---|
| after A, B, C | 3 entries, 2 tiered, 90 MiB on host, 2 demotions | 1 entry each time |
| A again | **hit 4096 tok from host tier in 0.02 s**, 0.66 s total | cold, 1.79 s |
| A again vs A cold | **byte-identical** | byte-identical (both cold) |

The number to beat in production is the replay's 35 s per pool miss on the
agent; 45 MiB came back in 0.02 s here, and the 1.7 GB case is the next
measurement, on the agent endpoint with its real pool.

## 5. Testing and acceptance

- **Prüfstand gate**: the fleet's 10-point code-generation harness runs against
  every artifact/config combination that claims production readiness. The
  reference scores to hold: 10/10 for the coder (B5-class artifact), the 3.8
  artifact must match its GGUF reference before shipping q4.

  **Measured 2026-08-28, b5 coder on the B60.** The bar is written down as
  "10/10 greedy". Under greedy arcint scores **8/10**, deterministically. That
  is not a arcint defect, and the evidence says the bar was never greedy:

  | run | decoding | score |
  |---|---|---|
  | arcint | greedy (temperature 0) | 8/10, byte-identical across repeats |
  | arcint | artifact defaults, seeds 1/2/3 | 10/10, 10/10, 8/10 |
  | OpenArc, same day, same task | its defaults (no temperature field) | 10/10 |

  OpenArc cannot be asked for temperature 0 — `frage.py` records that a temp-0
  call throws in OV GenAI *and* makes OpenArc unload the model, so every
  reference run went through the artifact's sampling defaults (temperature 1.0,
  top_p 0.95, top_k 20, straight out of `generation_config.json`). The stored
  `antwort-b5-greedy.lua` carries no `<think>` block and its first ten lines are
  identical to arcint's greedy answer, which is consistent with the same model
  and the same prompt diverging only where sampling would.

  Three checks say the divergence is the decoder's regime and not arcint's
  arithmetic: the rendered prompt is **byte-identical to reference jinja2** in
  both thinking modes; greedy output is **byte-identical across repeated runs**;
  and greedy output is **byte-identical to an independent implementation** of
  the same graph over the same prompt. So arcint reproduces the reference
  quality under the reference's decoding regime, and the "10/10 greedy" wording
  should be read as "10/10 at the artifact's sampling defaults" until someone
  produces a genuinely greedy 10/10 on this artifact.
- **Equivalence suite**: `tests/equivalence/run.sh`, run where the card is.
  `ARCINT_EXTRA_ARGS="--parallel 2"` runs the whole of it on a two-lane engine,
  which is where M6 had to leave it green: every equality claim here is about
  one sequence, and they have to keep holding on an engine that can run two.
  Verified 2026-08-29, all checks passed.
  Green on the b5 coder as of 2026-08-28: two greedy runs byte-identical, warm
  prefix cache byte-identical to cold, cache hits reported on the console, and
  a continuation of a cached prompt hitting too. MTP on vs off joins it at M4.

  One line of the original list has been demoted from gate to measurement:
  chunked vs unchunked prefill, which this backend cannot deliver (see §3.2).
  It is reported with numbers on every run rather than asserted, and the
  shipped default is the unchunked configuration that does satisfy equality.
- **Concurrency suite** (M6): `tests/concurrency/run.py`, run where the card is,
  green on **both cards** 2026-08-29. What it gates:

  | check | why it is a gate and not a print |
  |---|---|
  | no cross-slot bleed | two prompts interleaved are byte-identical to their solo runs, **in both start orders**. A lane reading another lane's pages, GDN rows or logits buffer answers something plausible and different, which is the whole failure class this engine exists to refuse |
  | both lanes were used | equality proves nothing if everything ran on slot 0 — the console must show slot 1 working |
  | cold/warm per lane | §3.4's invariant, held while the other lane is busy, with the hit reported |
  | the cache holds pages | a hit that shared no KV page is not the thing being claimed |
  | cancellation | one client disappearing leaves the other's bytes alone and gives back both the lane and its pages |
  | admission | a third concurrent request is a 503 carrying the reservation numbers, and `CL_OUT_OF_RESOURCES` appears nowhere in the log |
  | the stall is reported | a bound nobody prints is a claim, not a measurement |

  **Verified red before green**, as §5 requires. A build in which both lanes
  index lane 0 — one line — fails exactly the four bleed checks and the
  cancellation check, and passes "both lanes were used" (the console still says
  slot 1), which is the point of having that check separate. The driver noticed
  too: `dmesg` recorded `Engine reset: engine_class=ccs/bcs` and
  `Fault response: Unsuccessful` on the B60 during the red run.
- **Determinism**: two identical greedy runs produce identical bytes (verified
  on the A770/Vulkan agent baseline as achievable on this hardware class).
- **In CI today**: 155 unit cases, a 48-check curl round-trip and the
  lane-accounting stress (`tests/concurrency/stress.sh`, stub-only), all three
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
  a watchdog on it. Nothing about this is visible from inside the guest.
- **Sanitizers**: clean under ASan + UBSan with `-fno-sanitize-recover`, unit
  suite and round-trip both, plus a 200-request 24-way concurrency stress
  across 8 slots with every slot exercised and released. ASan has to run on
  x86_64: it aborts at startup on an aarch64 build container (allocator
  address-space check), so x86_64 is where that gate lives. It earns its
  keep — the M0 review's use-after-free in the argument parser reproduces as
  a clean ASan report on x86_64 and disappears with the fix.
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
  chunked prefill at 2048 and the logits slice, arcint loaded **257,167
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
- **Prefill, measured through arcint on the B60** (coder q4, chunked at 512):
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
  prefill, not decode. arcint tracks prefill t/s per card as a first-class
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
correct compute on this hardware. It is the reason arcint keeps the compiler
and owns the state.

## 7. Milestones

| # | milestone | exit criterion | state |
|---|---|---|---|
| M0 | skeleton: HTTP server, /health, /props, console format | curl round-trip | **done** (`e55e33b`) |
| M1 | single-sequence inference, greedy, 27B coder q4 on B60 | Prüfstand 10/10, ≥ 45 t/s | **done** — 51.4 t/s, 10/10 at artifact sampling defaults (§5) |
| M2 | paged KV + GDN ledger, chunked prefill | equivalence suite green, 256k context loads | **done** — suite green, **257,167 tokens loaded** (§5); paged path mapped but not adopted |
| M3 | prefix caching (block-aligned checkpoints) | warm/cold byte-equality, hit-rate stats on console | **done** — warm/cold byte-identical, hit stats on console |
| M4 | MTP for Qwen3.8, sampling beyond greedy | greedy-invariance with MTP on, measured acceptance | **acceptance done** — 93.3% on the dense model, verification exact (§3.5.2). Greedy-invariance is **not achievable on this backend** and the criterion was wrong to assume it was: a multi-token verify pass and a single-token plain pass differ (§3.2), so any speculative scheme can flip a near-tie. Also a net slowdown until the paged path lands. |
| M6 | per-slot InferRequest scheduler | N slots = N InferRequests (embeddings and MTP twins included); admission bounded by the measured reservation curve (§7.0.2a terms, per slot); the 200-request 24-way concurrency stress passes; single-stream latency regresses < 5% with the suite green. Any regression beyond that gets a profile naming the contended resource before any tuning. | **done** (§7.2) — two lanes on both cards, cross-slot bleed gated byte-identical in both orders, Prüfstand 10/10 on each lane *concurrently*, stress green under ASan+UBSan, single-stream decode 67.6–69.9 vs 68.8 t/s |
| M5 | 35B MoE q4 on A770 (16 GB fit), q8 variants on B60 | all three models pass their gates | **done**, and the 16 GB fit now works too: `--offload-ratio 20` serves the 35B on the A770 at 1.8 t/s where it previously refused to load (§7). The q8 half still waits on an export. |

M0 went past its exit criterion on purpose: everything that does not need a
GPU was implemented properly rather than stubbed, because that is where the
invariants live and they are cheaper to get right before an executor is
underneath them. What is genuinely absent is OpenVINO, weights, the KV pool,
the GDN ledger, the prefix cache and MTP. Two placeholders are marked as such
in the code and must not survive M1: the stub tokenizer is a reversible
splitter and not a BPE, and `render_chatml_stub` is not the model's chat
template — §3.7 keeps that in the artifact, and M1 takes both from the IR.

**All three models, measured through arcint on the B60, 2026-08-28** (the
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
against arcint holds the kernels and the graph constant and varies only the
pipeline. Measured 2026-08-28 on the B60, identical prompt, five runs each with
the first discarded as warm-up, wall clock including prefill:

| | median | runs |
|---|---|---|
| openarc, GenAI continuous batching (paged) | **53.8 t/s** | 51.6, 57.8, 51.7, 55.9 |
| arcint, stateful | **46.6 t/s** | 46.0, 46.0, 47.8, 47.1 |

**13.4%** end to end. But that number measures openarc's *whole pipeline*, and
taking it as the size of the prize was wrong — the graph and the pipeline had to
be separated, and separating them changes the answer by a factor of four.

**The graph, profiled directly.** Same card, same artifact, same 256-token
depth, steady-state decode step (not the first after a prefill, which is slower,
and not depth 64, which hides the attention and transpose growth):

| | stateful | paged | delta |
|---|---|---|---|
| **decode step** | **19.01 ms** | **≈11.3 ms** | **−41%** |
| `FullyConnectedCompressed` ×371 | 9608 µs | 7217 µs | −2391 |
| `Transpose` → `permute_ref` ×90 | **3207 µs** | **eliminated** | **−3207** |
| `DynamicQuantize` ×160 | 1308 µs | 1296 µs | — |
| `MOECompressed` ×40 | 1180 µs | 569 µs | −611 |
| `GatedDeltaNet ref` → `PagedGatedDeltaNet opt` ×30 | 1105 µs | 468 µs | −637 |
| `StridedSlice` ×101 → ×40 | 495 µs | 140 µs | −355 |
| `Concat` ×50 → ×20 | 272 µs | 74 µs | −198 |
| `IndirectSDPA` → `PagedAttentionExtension` ×10 | 185 µs | ~0 | −185 |

Nearly half the saving is the transposes vanishing outright — the ones the
custom-kernel excursion could not beat from outside and §8 predicted would
dissolve here. The optimized paged GDN kernel is worth another 0.64 ms, and
`PagedAttention` costs essentially nothing where `IndirectSDPA` was already
visible at this shallow depth and grows as L^1.46.

**So the prize is 1.68× on the graph, not 13.4% end to end.** The difference
between those two numbers is GenAI's own pipeline: it reaches 51.0 t/s wall on a
graph that should allow far more, while arcint's serving loop measures at
**2%** of a decode step (graph 2.77 s of 2.82 s over 144 tokens; embed 0.4%,
sample 0.7%, emit 0.7%). Neither combination exists today. arcint keeping its
loop and running the paged graph projects to **≈87 t/s** on the coder against
51.1 measured — and that projection, not 13.4%, is what the work is worth.

A caution that cost an hour: `PERF_COUNT` inflates wall clock badly (53.6 ms
measured for a step whose node sum is 19.01 ms and whose real serving cost is
19.2 ms/token), and it reports the *last* inference. Profile a steady-state step
at a realistic depth or the numbers mean nothing — a depth-64 first-step profile
said the two paths were identical, which is how this was nearly missed.

Note also what this says about where to hunt. The MoE coder decodes at roughly
a fifth of its bandwidth roofline while the dense model sits near 60% of its
own, so the headroom is on the MoE side — and only the MoE has an external
baseline on the identical artifact.

#### 7.0.0a The same comparison at depth: 32k context inverts it

The short-prompt A/B above favours GenAI's pipeline by ~10%. At 31930 tokens of
prompt ("Write a CSV parser in Lua", filler ahead of it, greedy, 120-token
answers, prefill and decode measured separately — decode via t(240)−t(120) for
openarc because its prefix cache reuses the prefill and breaks a naive
subtraction):

| 32k context | prefill | decode |
|---|---|---|
| openarc production (B60, GenAI CB) | 867 t/s (36.8 s to first token) | 21.3 t/s |
| arcint (B60, stateful) | **2444 t/s (13.1 s)** | **32.3 t/s** |
| arcint (A770, chunk 512) | 901 t/s | 13.4 t/s |
| openarc / GenAI CB path (A770) | **refuses at startup** (§7.0.2a; GenAI's *stateful* path did serve here) | — |

At depth the stateful engine beats the production pipeline 2.8× on prefill and
1.5× on decode — the opposite of the short-prompt result. So "GenAI's pipeline
is ~10% ahead" is a shallow-context statement only; its scheduler gives back
far more than 10% at depth. Both engines feel the depth collapse (§5): arcint
52 → 32.3 t/s from 256 to 32k of context, openarc 51 → 21.3.

Small open lead from the same run: arcint's emit accounting rose to 0.43 s for
120 tokens at 32k (3.6 ms/token, ~12% of decode) where it is ~0.2 ms/token at
short context. The graph still dominates; worth a look, not a fire.

#### 7.0.0b The bar, so "done" has a definition

Decode rooflines by **parameter arithmetic** — active weight bytes per token
over memory bandwidth, explicitly *not* measured DRAM traffic, see the caveat:

| B60, q4, greedy | active bytes/token | roofline | measured | of roofline |
|---|---|---|---|---|
| coder MoE (~3B active) | ~1.5–1.9 GB | ~240–300 t/s | 52 t/s | **~20%** |
| dense Qwen3.8 (all 13.4 GiB) | ~13.4 GB | ~34 t/s | 19.9 t/s | **~59%** |

The caveat that keeps this table honest: the 19.01 ms measured decode step
would move **8.7 GB** at full bandwidth — 4.6–5.8× the active-weight
arithmetic. Two readings fit that gap and the per-kernel profile cannot
separate them: the kernels run far below bandwidth (the reference-kernel share
and the FC-dominated profile point this way), or real traffic exceeds the
arithmetic (router, shared experts, activations re-read). Separating them
needs DRAM counters (Level-Zero/PTI), not `PERF_COUNT`. Until then the
ceiling is quoted only as arithmetic, and the operational bar stands
regardless of which reading wins: **the MoE has ~4–5× of generic headroom on
this card, the dense model ~1.7×**, which is why generic substrate work aims
at the MoE and the model-specific lever (MTP) at the dense model.

### 7.0.1 The bus was not the explanation — retracted

An earlier version of this section explained the A770's behaviour, and part of
the MoE decode gap, by PCIe bandwidth. That was wrong, and the retraction is
recorded here rather than deleted because the error is instructive: the story
was plausible, arithmetically decorated, and never checked against the one
decomposition that would have falsified it.

The check is: sum the per-kernel times for one decode step, compare with the
step's wall clock, then multiply the bytes actually claimed to cross the link by
the measured link rate and see whether the product fills the difference.

| | |
|---|---|
| node-time sum, B60, depth 256 | **19.01 ms** |
| serving decode, same conditions | **19.2 ms/token** |
| arcint's host-side work (embed + sample + emit) | **2%** |
| bytes over the link per step: sliced logits 248320 × 4 B | **993 KB** |
| plus the embeddings hop | ~8 KB |
| that traffic at x8 Gen4 / x4 Gen3 | **0.06 / 0.32 ms** |

The kernels are the step. There is no unexplained residue for the bus to fill,
and the traffic that does cross is 0.3–1.7% of it. A fully resident model moves
tokens over the link, not weights. The cross-card evidence says the same: a 5×
link difference produces a 1.15× throughput difference on the dense model, which
no bandwidth-governed system does.

**Where the bus genuinely is the explanation**, and where it should have been
confined: `VariableState::get_state`/`set_state` snapshots — §3.5.1 measured
that at 51–67% of decode, moving 69.9–171.3 MiB per step — prefix-cache
serialisation, which is the same copy paid once per prompt, and expert-offload
streaming (§7.1). Those are host round-trips of tens to hundreds of MiB. Plain
decode of a resident model is not.

### 7.0.2a Admission by measured reservation — the paged driver on the card GenAI refuses

GenAI's `ContinuousBatchingPipeline` will not start the coder on the A770:
it refuses at startup even at `num_kv_blocks = 256` — 80 MiB of KV —
because its budget check prices in its own worst-case assumptions (batched
prefill activations, a minimum block pool, and a checkpoint pool of ~33 MiB
LA rows sized by its adaptive multiplier), and those exceed the 1.81 GiB left
beside 13.30 GiB of weights. Its error carries no numbers.

The prototype paged driver (`tools/paged_driver.py`) initially cleared that
wall by *not doing the arithmetic*, which is not a capability — a driver with
no admission check meets the same 15.1 GiB at runtime as
`CL_OUT_OF_RESOURCES` mid-request, strictly worse than refusing at startup.
The fix is a startup reservation in which every term is **measured**:

| term | source | A770, coder |
|---|---|---|
| weights + graph | `GPU_MEMORY_STATISTICS` after compile | **13.30 GiB** |
| activation peak | run one prefill chunk, read the delta | 2.28 / 1.14 / **0.57** / 0.28 GiB at chunk 2048 / 1024 / 512 / 256 |
| GDN state slab | one row across all state tables | 32 MiB |
| KV | ctx × 20 KiB/token | 0.94 GiB at ctx 49152 |
| margin | stated, not hidden | 0.25 GiB |

The activation peak is **linear in the chunk size** over this range, so the
chunk is the knob that buys context, and the admission table falls out of the
sweep (M6 refined the shape: it is affine rather than linear from the origin,
and the probe has to climb rather than jump — §7.2): chunk 2048
is inadmissible on this card outright (the earlier "it ran anyway" was the
plugin silently spilling 1.89 GiB to host), chunk 1024 admits ctx ≤ 20320,
chunk 512 admits ctx ≤ 50480, chunk 256 admits ctx ≤ 65344.

Scope, before the sentence: this is a statement about GenAI's
**continuous-batching path and its admission check**, not about GenAI or the
card. This exact model served production on this exact card for two days
through GenAI's *stateful* path (VLMPipeline, 43 t/s, 10/10 on the harness)
before the B60 arrived and the GPU.0 pin silently migrated to it. What the
A770 cannot get from GenAI is the paged path — the one with the optimized GDN
kernels, real q8 KV, and rollback-by-arithmetic — because the CB budget check
prices its worst-case assumptions into a residue they do not fit.

So the sentence this section exists for, with the numbers in it: **GenAI's CB
path refuses the coder on the A770 outright under its assumptions; the paged driver
serves ctx 49152 at chunk 512 because its measured peak is
13.30 + 0.57 + 0.03 + 0.94 + 0.25 = 15.09 of 15.11 GiB** — prefill 586 t/s,
decode 37.1 t/s at depth 4096, device at 14.33 GiB with the difference being
buffers the plugin keeps host-side by its own choice. The same physical residue
admits a working configuration because the arithmetic is tighter — sliced
logits, a single slot, multiplier-1 checkpoints, a measured peak — not because
the arithmetic was skipped. An inadmissible request is refused at startup with
those terms spelled out, the same shape as the context-overflow 400.

Two measurement traps, recorded so they are not rediscovered:
`GPU_MEMORY_STATISTICS` lumps `usm_host` in with device memory if summed
blindly (that made the first reservation conclude the model could not fit at
all), and the plugin spills some prefill buffers host-side by choice, which
must not be counted against the card.

### 7.0.2 The two cards do not run the same kernels

Same graph, same artifact, same depth 256, steady-state step:

| | A770 | B60 |
|---|---|---|
| decode step, node time | **31.19 ms** | **19.01 ms** |
| serving decode, coder | **37.7 t/s** | **51.1 t/s** |
| `FullyConnectedCompressed` | 14464 µs (46.4%) | 9608 µs (50.5%) |
| `Transpose` → `permute_ref` | 4812 µs (15.4%) | 3207 µs (16.9%) |
| `GatedDeltaNet ref` | 3197 µs (10.3%) | 1105 µs (5.8%) |
| `IndirectSDPA` | 1155 µs (3.7%) | 185 µs (1.0%) |
| `Reorder` bfyx→blocked | 489 µs | 120 µs |

The GDN reference kernel costs 2.9× more on the A770 and `IndirectSDPA` 6.2×,
against 1.5× on the GEMMs — so the gap between the cards is not uniform, and it
is largest exactly on the kernels the paged path replaces. The plugin selects
differently per generation (Xe2 has no SIMD8, and the plugin's own permute
kernel says so), so a kernel-level conclusion drawn on one card does not
transfer to the other. Host-side work is 2% on both.

#### 7.0.2c Prefill, measured at last (2026-08-29)

Full baseline in `prefill-baseline.md`; the three things that belong in the
architecture record:

1. **Served prefill is 99.2–99.8% graph time.** The per-phase breakdown
   (embeddings, page allocation, cache restore, device wait, remainder) accounts
   for under 1%. There is no host-side prefill overhead to find, so a share of
   the graph is a share of prefill.
2. **Forty `FullyConnectedCompressed` nodes run on a reference kernel during a
   2048-token chunk and take a third of it.** Named, not inferred:
   `layers.N.mlp.shared_expert_gate/ov_ext::linear/MatMul`, one per layer,
   layers 0–39. They cost 51.1 ms — 1.28 ms each — against 30.1 ms for the other
   331 FC nodes together. `shared_expert_gate` is `[1, 2048]`: one scalar per
   token, 4.2 MMAC at M=2048, arithmetically the cheapest FullyConnected in the
   layer. The reference kernel's cost is not proportional to its arithmetic.
   The trigger is **not** simply M>1: bisected in one load over eleven token
   counts, the fallback is absent through M=64 and present from M=128.

   **The scaling argument built on that sweep is withdrawn (2026-08-30).** It
   read the gate's 32.3 ms at M=128 against 41.4 ms at M=2048 — a 28% rise for
   16x the tokens — as "nearly flat", and concluded a fixed per-invocation cost
   such as a decompression redone per call. Re-reading the *same* sweep for the
   other ops kills that: `PagedCausalConv1D` and `PagedGatedDeltaNet`, which do
   genuine O(M) work, are **U-shaped** across it — 8417 → 4234 → 6390 us and
   20461 → 10628 → 15638 us, falling to a minimum at M=256 before rising. Time
   that *falls* while the work grows 128x is warm-up decay, not scaling: an
   ascending single-pass sweep superimposes a decaying first-touch bias on
   whatever it measures. Over M=128→2048 those two rise 45% and 44% against the
   gate's 28%, so at these sizes every op in the sweep is dominated by something
   other than its arithmetic, and "flatter than its neighbours" is all that can
   be said. Decompression is no longer indicated by this measurement; it remains
   one of two candidates, separable only by the single-variable arms already
   recorded. The profiler now runs each capture twice and dumps the second. `prefill-baseline.md` records the two single-variable arms and
   what each must show.

   **Sized before being fixed, and deliberately not fixed.** These nodes are
   depth-independent and nearly chunk-size-independent, so their cost over a
   prefill is `(N/C) x ~40 ms`: 4.6% of prefill wall at 14450 tokens, 2.7% at
   57792, 1.7% at 115564. The 33–39% figure is a share of a *past-0 chunk's node
   time*, and a past-0 chunk is the cheapest one in any real prefill. Both arms
   require the same pre-compile graph surgery that the fix does, so measuring
   costs what fixing costs. (The sentence that stood here named
   `PagedCausalConv1D` at "6.3% of both phases" as the larger prize; item 3
   below withdraws that figure for prefill, so it named the wrong candidate.
   The current sizing is in the two paragraphs that follow.)

   **The dispatch was read, not inferred (2026-08-30).** `ONEDNN_VERBOSE=dispatch`
   reports the decision from a normal run — no debug build was needed. For the
   gate's problem, `M x 2048 : 2048 x 1` with s8 activations, u4 group-quantised
   weights, an f16 destination and a **fused `eltwise_logistic` post-op**,
   `jit:gemm:any` is rejected with *"matching kernel not found in catalog"* at
   M = 128, 256, 416, 512, 1024 and 2048 — and **not at M = 1**, which is why
   decode never pays it. The M>=128 boundary bisected earlier is confirmed from
   the dispatcher's own log.

   The mechanism read out of the source — that the k-parallel candidates are
   filtered by `po_valid`, which our non-scale post-op and f16 destination both
   fail — is **falsified**. At `debuginfo=5` the same loop prints a skip reason
   per candidate; **no skip line appears for any of the six shapes, and zero
   `consider` records fall inside those six problems**. `select_kernel()`
   returned an empty list and the filter never ran. The null is load-bearing
   only because the instrument carries a positive control at the same verbosity
   gate: `info,gpu,gemm,consider` is guarded by the same `debuginfo >= 5`, and
   2198 of those records are in the same log. It is a **shape gap in the kernel
   catalog**, not a policy rejection.

   **Steerability, asked before committing to the long loop.** Not by an
   environment knob (`ONEDNN_GROUPED_GEMM_USED` is the only related string in the
   shipped plugin, and is unrelated); and **not by un-fusing the sigmoid — measured, not
   argued**. The debug build carries `GPU_DISABLE_POST_OPS_FUSIONS`, and the two
   arms are unambiguous: control reproduces production exactly (52
   `attr-post-ops:eltwise_logistic` descriptors, 6 catalog misses), and with
   fusion off the post-op count falls to **0** while the misses stay at **6**.
   The flag's own effect is the positive control, so the null is readable. The
   destination dtype is the same argument one step further out and is *not*
   tested — it feeds the filter that never executes, which is an inference.
   Possibly steerable
   by **changing N**, because the catalog matches on shape: padding the gate's
   output width to a catalogued N is a pre-compile graph rewrite we own outright
   — rung zero of §1.1, no OpenVINO change and no oneDNN change. So §1.1's first
   rung for this row is *not* a oneDNN PR after all; that stays the fallback if
   padding does not pay.

   **The share, refined by the corrected profiler.** With the profiler running
   each capture twice and dumping the second, the gate's forty nodes cost 54 ms
   per chunk, not ~40 ms, so the depth-independent arithmetic gives **6.0% of
   prefill wall at 14450 tokens** and ~2.2% at 115564 — superseding the 4.6% /
   1.7% above, which came from the single-pass capture. The 30.2% seen in a
   chunk inventory and these single-digit figures are both correct: a past-0
   chunk's node total is 179 ms while a mean chunk at 14450 tokens is 910 ms, a
   **5.1x** denominator difference. Nothing was truncated; totals were always
   summed over all pairs. Sixth instance of the past-0 error.

   **The corollary written here on 2026-08-30 is withdrawn the same night.**
   It read 910 ms of mean chunk against 179 ms at past 0 as putting 80% of a
   mean chunk in the depth-dependent term, making every kernel row a share of
   the other 20%. That divided a profiled node-time sum by a served wall time,
   and the two instruments do not agree well enough to be divided. The depth
   sweep meant to confirm it refutes it: a 2048-token chunk profiles at 229.1 ms
   (past 0), 273.1 ms (4096) and 353.7 ms (12288) — **linear in past** at
   ~0.0101 ms per past-token, growing 54% across that range rather than the 5x
   the corollary required.

   **What it uncovered instead is an attribution gap, and it is the larger
   finding.** Summing that sweep over a 13410-token prefill predicts **1.88 s**
   of node time; the same prefill, served from **the same process**, spends
   **6.20 s in the graph**. **Node times account for 30% of graph wall.** The
   profiler is not the explanation: with profiling on that prefill takes 6.20 s
   of graph and with it off 5.76 s, **+7.6%**, and the comparison above has it
   on for both sides. Dispatch overhead is not the explanation either — ~590 ms
   unattributed across ~1135 node executions would be ~520 us each. The
   disagreement is unexplained, and until it is closed **no node share can be
   converted into a share of served time**: every "X% of prefill wall" in this
   record, including the 6.0% and 2.2% written for the gate a few paragraphs
   above, depends on that conversion. Relative shares within one capture are
   unaffected. `docs/kernel-selectors.md` carries the numbers.

   It also retires an item that was marked closed: "at least 90% of served
   prefill wall attributed" was answered with the phase breakdown, but 99.4% in
   "graph" only says the time is *inside* the graph call. Opening the graph
   reaches about a third of it.

   **And the profiler's synthetic input is not neutral.** Captures fed a chunk
   of identical token ids, which in a mixture of experts routes every token to
   the same experts. `ARCINT_PROFILE_TOKENS=random` is the other arm: at past
   12288 the MoE row is **28.7% larger** under pseudo-random ids (25970 ->
   33415 us) while `MoERouterFused` is unchanged (1849 vs 1842 us), which is
   exactly where the effect should and should not appear. Small against the
   attribution gap (7.4 ms of ~590 ms), but it means MoE shares in every earlier
   capture are understated.

   A small chunk does not dodge it: chunk 64 avoids the fallback, but the chunk
   sweep measured what small chunks cost (1339 t/s at 512 against 1878 at 2048).
   The fallback is cheaper than the cure.
3. **`PagedCausalConv1D` is on a reference kernel in both phases** (6.3%, 30
   nodes, one per GDN layer). The paged transformation gave GDN an optimised
   kernel and left the causal conv behind.

   **Sized and re-framed 2026-08-30, before starting on it.** Two facts change
   what this item is:

   - **There is no optimised implementation to select.** The plugin registers
     `PagedCausalConv1DRefImpl` and a `PagedCausalConv1DRefGenerator`, and the
     only kernel string is `paged_causal_conv1d_ref`. No `Opt`. This is
     therefore *not* the shared-expert-gate situation, where a fast kernel
     existed and was not chosen; the lever here is writing a kernel behind the
     `--custom-kernels` seam, or filing upstream. Both are real work, and §1.1
     fixes the order to try them in and what a patch may not be allowed to
     become.
   - **It is a decode lever, not a prefill one.** "6.3% of both phases"
     overstates the prefill side by exactly the past-0 error that has now bitten
     three times: the conv is depth-independent, so over a real prefill its cost
     is `(N/C) x per-chunk`, which is **under 1% of prefill wall**. On decode,
     where every step pays it and nothing grows with depth, 6.3% of the step is
     6.3% of the rate — about 4 t/s of 68.

Two things that are **not** established and are recorded as such:

- The MTP prefill machinery — a per-chunk 16 MB hidden read-back and an O(L)
  mask rebuild — is real for **the dense Qwen3.8, which carries an MTP head**.
  The b5 coder ships none, so none of it runs on the artifact these numbers come
  from. It is a lead for the 3.8, not for the coder.
- "Attention prefill runs at f16 SIMD peak rather than the matrix engines" is
  **refuted 2026-08-30** (§7.0.2b: the prefill attention kernel is
  `sdpa_micro__prefill` and its assembly carries `dpas`, in both KV precisions).
  It had been inferred from the depth curve's quadratic coefficient
  (~12 TFLOPS effective against a ~90 TFLOPS XMX f16 peak). A coefficient is
  evidence, not a mechanism, and §7.0.2b already says `exec_type` cannot settle
  it. It stays labelled until instruction-level tracing does.

#### 7.0.2d The attribution gap, settled by an instrument that can see gaps (2026-08-30)

The profiler sums parts; a gap has no node to attribute it to. So the 70% of
prefill graph time no node accounted for was measured with an OpenCL device
timeline (kernel enqueue / start / end per command), which counts occupancy on
the device rather than node totals — a different counting principle from the
apparatus that found the gap. Full record, controls and pre-registration in
`docs/attribution-gap.md`.

**The pre-registered prediction was wrong, and the question was framed wrong.**
It asked one binary — card idle, or profiler blind — as though prefill and
decode had a shared answer. They do not:

- **Prefill is device-bound: 95.1% busy** (4.980 s of a 5.235 s span, reproduced
  to 0.05% across two clusters). The profiler was blind, not the card idle.
- **Decode is host-bound: ~43% busy**, so roughly half of a decode step is spent
  with the card idle. Corrected for tracer overhead, which is +0.9% on prefill
  and **+42% on decode** — a per-call interceptor lands on exactly the path that
  limits a host-bound phase. Uncorrected it would have read 77% idle.

The lesson for the next binary: **ask per phase.** One number for "the model"
hides the phase whose answer is the opposite.

**Why no node share ever converted.** Same run, same chunk, two ops with a clean
1:1 between nodes and launches: PERF_COUNT reported 14.97 ms where the device
spent 27.23 ms on the conv, and 36.78 against 67.19 ms on GDN — **1.82x and
1.83x**. It reports about 55% of a kernel's own device time, sees no transfer
at all, and does not enumerate sub-kernels (13342 device commands against
~7150 node executions). The profiler's tables now say so at the point of
printing, with the capture they are a share of, because six implicit
denominators produced three retracted headlines.

**What prefill is made of, as shares of its own wall time** (12916 tokens, f16
KV, chunk 2048): `clEnqueueMemcpyINTEL` 17.6% — six device-to-host copies of
142.74 ms, one per chunk, no node; `grouped_micro_gemm` 17.4%;
`sdpa_micro` 10.9%; **`ref_matmul` 10.1%** (the shared-expert gate — larger
than every node-share estimate of it); `gemm_kernel` 8.9%; GDN 8.0%; MoE
scatter 6.7%; `generic_eltwise_ref` 3.5%; conv **3.3%**; idle 4.9%.

**Item 3 above is retired.** `PagedCausalConv1D` was a decode lever or nothing,
and on the device it is 0.9 ms of a 190 ms untraced decode step: **0.47%**, not
the 6.3% its case rested on — that figure was a PERF_COUNT node share of a
past-0 capture, twice removed from wall time. A perfect conv kernel buys well
under 1 t/s. Nobody should re-derive the 6.3% from the older tables in this
section; they measure what the counter reports, not what the card does.

#### 7.0.2e The 142.74 ms copy was the logits, and the slice had missed (2026-08-30)

The largest single item the timeline priced was a device-to-host copy of
142.74 ms per prefill chunk — six per prefill, 17.6% of prefill wall, no node.
Three checks, none of them a kernel:

1. **Size.** Transfer tracking in the tracer: **2,034,237,440 bytes** per copy,
   which at 142.74 ms is **14.25 GB/s** — the link at full DMA rate, the same
   rate the 16 MiB embeddings copies run at. So a volume problem, not a path
   problem, and the pinned-versus-pageable question was dead before it was
   asked. Across the reservation probe the copies went 127 -> 254 -> 508 ->
   1017 -> 2034 MB for M = 128 ... 2048, and 444,989,440 bytes for the final
   448-token chunk: **993,280 bytes per token, exactly 248,320 x 4**. The
   artifact's vocabulary is 248,320. It was the full f32 logits, one row per
   prompt token.
2. **Why, when the log said "logits sliced to the last 1 row(s)".** The slice
   walks to the LM head and cuts axis `rank - 2` of its input. The dense export
   is `[1, tokens, hidden]`, where that is the token axis. The paged export is
   `[tokens, 1, hidden]`, where it is the singleton batch axis: "last 1 of 1",
   a no-op that returned true. Every chunk went on computing and copying
   `[M, 248320]` logits that only the last chunk's last row ever read. The
   graph declares both leading axes dynamic, so the layout cannot be read from
   the shape: an attempt to detect "the one axis that can exceed one row"
   found two and bailed, leaving the model unsliced — and produced an A/B
   whose before and after were identical to the millisecond, which is how that
   attempt was caught. The caller now states the token axis (0 on the paged
   path, `rank - 2` on the dense one) and **the reservation probe verifies the
   claim against a real forward**: `logits slice verified: 1 row(s) for a
   128-token forward`, or refuse to start.
3. **Needed at all?** No. Elided, not accelerated.

**Measured, same prompt, same card, f16 KV, chunk 2048, 12448 tokens:**

| | before | after |
|---|---|---|
| prefill wall | 5.02 s (2479 t/s) | **3.97 s (3137 t/s)** — **-21% / +27%** |
| device-to-host copies > 5 ms per prefill | 6 x 142.7 ms | **none** |
| transfers, share of prefill wall | 17.6% | 0.5% |
| device busy | 4.980 s | 3.935 s |
| activation reservation, chunk 2048 | 2.99 GiB (1283.9 KiB/token) | **0.90 GiB** (216.9 KiB/token) |
| greedy output, 96 tokens | — | **byte-identical** |

**Gate: Prüfstand 10/10** on the fixed build in the service's own configuration
(GPU.0, u8 KV default, 262144 context, prefix cache; `logits slice verified:
1 row(s) for a 128-token forward` in the boot log), 5222 tokens generated at
63.2 t/s, scored by executing the candidate. The bar is met; the package is not
yet cut.

The reservation line is the second win: 2.09 GiB of the activation budget was
logits nobody read, and it comes back as context or chunk. The LM-head GEMM
over the discarded rows went with the copy (`gemm_kernel` 468 -> 403 ms per
prefill). Decode is unchanged, as it should be: it always ran one row.

What remains of prefill on the device, as shares of wall (4.30 s): MoE expert
GEMM 20.9%, **`ref_matmul` 12.2%** (the gate; grown again with the
denominator), attention 12.0%, GDN 9.5%, `gemm_kernel` 9.4%, MoE scatter
7.9%, `generic_eltwise_ref` 4.2%, conv 3.9%, idle 8.4%.

#### 7.0.2f Decode's host-side half, named (2026-08-30)

The timeline left decode ~57% host-side and unattributed. Two instruments,
chosen so that one has no overhead worth speaking of:

1. **The GPU plugin's own `host_time_profiling`** (debug build; the option
   measures host time from the start of `infer()` until the plugin is ready to
   block on the final `clFinish`). It costs nothing measurable — 65.0 t/s under
   it against 63.2 on the gate run — but prints only an average over every
   infer of the process, so two runs of different length were solved for the
   decode step: 517 infers averaging 17.66 ms with 512 decode tokens, 133
   averaging 29.78 ms with 128. The five non-decode infers (reservation probe
   and prefill) cost 2238 ms in both; **a decode step costs 13.46 ms of host
   enqueue time, of a 14.9–15.3 ms step: 90%.**
2. **The tracer's call log**, which is heavy on this phase (28.9 t/s under it,
   +125%) and is therefore read for its shape only: on the inferring thread,
   **76% of the decode window is outside any OpenCL call**; the API itself is
   ~1130 `clEnqueueNDRangeKernel` per step at ~5 us each plus ~10,800
   `clSetKernelArg*` per step at ~0.1 us. The tracer inflates the inside-call
   bucket; "outside" is the clean one, and it is the large one.

**So decode is launch-bound in the plugin's per-node execution path.** Each
step walks ~1130 primitives, does per-node shape inference and argument update
on the host (the graph is dynamic-shape), and enqueues; the device, at ~6.5 ms
of work per step, runs behind the host and idles between launches. The wait
after the last enqueue is ~1.5 ms, which is exactly the tail of the device
finishing what was enqueued last. The mechanism is host throughput, ~12 us per
primitive all-in, not any kernel and not anything in arcint: arcint's own share
of a step (embed, sample, emit) is ~3%.

**What this prices.** No kernel row buys decode anything: the device is not the
limit. A decode lever is one of (a) fewer primitives per step — fusion in the
plugin — or (b) a cheaper per-primitive host path, which is what a static-shape
or shape-agnostic compile exists for, or (c) removing the host from the loop
between steps. All three are OpenVINO-plugin work, §1.1 applies, and none is
started here. The realistic ceiling if the host went to zero is the device's
own ~6.5 ms per step, i.e. roughly 2x decode; a fusion pass that halved the
primitive count would be worth ~1.4x.

#### 7.0.2g Padding N on the gate: measured, and it has a decode price (2026-08-30)

`pad_gate_matmuls` widens the shared-expert gate from `[2048, 1]` to
`[2048, N]` on the ov::Model before compile: the u4 weight, zero points and
scales get zero rows (a zero scale makes the dequantised row exactly zero
whatever the nibbles hold), the group-flattening Reshape learns the new width,
and a Slice takes column 0 back out before the sigmoid. Rung zero of §1.1. It
is behind `ARCINT_GATE_PAD=<n>` and **off by default**, for the reason below.

Same session, same card, f16 KV, chunk 2048, 12448 tokens, all against wall:

| N | oneDNN impl for the M=2048 gate | prefill wall | decode | greedy 96 |
|---|---|---|---|---|
| 1 (stock) | `ocl:ref` (catalog miss) | 3.97 s (3133 t/s) | 45.2–45.9 t/s | reference |
| 16 | `jit:gemm` | **3.45 s (3608 t/s)** | 43.0–43.6 t/s | identical |
| 32 | `jit:gemm` | 3.45 s | 43.0–43.5 | identical |
| 64 | `jit:gemm` | 3.45 s | 43.4–43.6 | identical |

**Prefill −13% wall (+15% throughput), N=16 is enough, output byte-identical.
Decode −5%, consistently** — six padded readings all below both stock ones.
The decode cost is the launch count, as §7.0.2f predicts: the node dump shows
+50 primitives walked per step (40 `StridedSlice`, 6 `Reshape`, 4 `Transpose`)
and +48 launched, at ~12 us each ≈ 0.6 ms of a 22 ms step. So the trade is
prefill-heavy workloads win and long answers lose, with break-even near a
500-token answer on a 12k prompt — not a default until the extra launch is
removed or decode's launch count has been cut (after which 48 launches cost
device time, ~0.1 ms, and the trade disappears). A version without the Slice
— replicated gate rows and a reshape on the consumer side — trades one launched
primitive for three walked ones and is untested.

**And a profiler note that belongs next to it.** PERF_COUNT's decode-step
capture reports the stock gate at 36.5 ms across its 40 nodes — 912 us each on
`jit:gemm:any` at M=1 — in a step whose wall is 22 ms and whose device time is
~6.5 ms; with padding it reports 241 us. Whatever that counter measures at M=1
for that node, it is not device time and it moves opposite to the wall. Decode
is read from the timeline and from wall, never from PERF_COUNT per node.

#### 7.0.2i The gate padding's decode price is the extraction, whatever the op (2026-08-30)

The forty `StridedSlice`s that cost `--gate-pad 16` its 5% of decode were
replaced by a `VariadicSplit` placed *after* the sigmoid — so the sigmoid stays
an FC post-op, and the split is the op the plugin lowers to a crop, which at
offset 0 is the case its in-place optimisation accepts. It took four attempts
to build, all of them the order in which a modified subgraph must be
re-inferred (a Reshape re-inferred before its input throws; a model-wide pass
before the consumer is rewired throws on the Multiply; the DFS reversed is the
bottom-up order). Then, same session, f16, 32768:

| | stock | pad16, StridedSlice | pad16, VariadicSplit after sigmoid |
|---|---|---|---|
| walked / launched | 2315 / 1171 | 2365 / 1219 | 2355 / **1212** |
| the 40 extraction nodes | — | `strided_slice_ref`, launched | `generic_eltwise_ref`, **launched** |
| decode, 512 tok | 71.6 t/s | (−5%) | **67.1 t/s (−6%)** |
| prefill 12448 | 3.97 s | 3.45 s | **3.45 s** |
| greedy 96 | ref | identical | **identical** |

The crop did not become a view; it launched as an eltwise copy, and the decode
price is the same. Two conclusions. The extraction costs ~20 us per layer in
host time however it is spelled, because a launched shape-changing primitive
on a dynamic graph is what costs, not the kernel name; and the padding's
decode price is therefore not removable from arcint's side — it goes away when
decode's launch count is cut (§7.0.2f), or if the plugin's crop-as-view
condition can be met, which is read next. The split variant is kept as the
implementation (fewer primitives than the slice, sigmoid still fused); the
flag stays off by default with the same break-even.

**Why the crop is not a view — read, not confirmed.** `prepare_buffer_fusing`
lets a dynamic crop through at build time on the simple-format check alone (no
padding exists yet) and defers the in-place decision to runtime, where a crop
along the feature axis gives its output a *dynamic* padding on that axis
(`update_in_place_crop_padding_along_feature` sets `_dynamic_dims_mask`). The
consumer here is a broadcasting `Multiply`, `[M,1]` against `[M,2048]`. None of
the explicit refusals in the pass match (gemm user, lstm/lora/mvn user,
non-constant split inputs, constant node), so the likely refusal is whether
that eltwise implementation accepts a dynamically padded broadcast operand.
That is a hypothesis; confirming it is a plugin-side read with a plugin-side
fix, and the payoff is ~0.85 ms of a decode step that vanishes anyway once
decode is no longer launch-bound. Parked at that price.

#### 7.0.2h The decode primitive histogram: a handful of classes, not a long tail

One decode step, every node the plugin walks, by name (`ARCINT_PROFILE_NODES`):
**2315 primitives walked, 1171 launched** (the timeline's ~1130 kernel launches
plus copies). The other 1144 are walked on the host with no kernel: 582
`Reshape`, 120 `Multiply`, 91 `Parameter`, 80 `Sigmoid`, 80 `Crop`, 46 `Add`,
40 `VariadicSplit`, 40 `Swish` — the eltwise ones are post-ops already fused
into an FC. What a walked-but-unlaunched primitive costs on the host against a
launched one is **not measured**; 13.46 ms over 2315 walked is 5.8 us each,
over 1171 launched 11.5 us each, and the truth is a mix. That split is the next
number this needs, because it decides whether the 582 Reshapes are a target.

Launched, by class: `FullyConnectedCompressed` **371** (32%), `DynamicQuantize`
**161** (14% — one per distinct FC input, the s8 activation quantisation), `RMS`
131, `Add` ~104, `Swish` ~60, `MoE` 40+40, `GDN` 30, `conv` 30, `Multiply` ~40,
`StridedSlice` 48. Six classes are ~75% of launches. Per layer: a GDN layer
walks 38.9 primitives (19 in `linear_attn`: 5 FC, 5 Reshape, 2 Multiply, 2
Swish, 2 Crop, RMS, conv, Add, Sigmoid, SoftPlus, VariadicSplit, Concat,
ShapeOf; 14 in `mlp`: 5 FC, MoE, 2 Multiply, 2 Reshape, Swish, Add, Sigmoid,
Concat), an attention layer 43.2. The 716 "outside layers" are per-layer nodes
that lost their names to transformations — 161 DynamicQuantize, 235 Reshape,
91 Parameters (the 91 state and cache inputs), 80 Add, the routers, the paged
GDN and attention nodes.

**The walked-versus-launched split, solved from two extra configurations
(2026-08-30).** Three decode configurations that move the two counts
differently, host enqueue time from the plugin's own profiler solved across
512/128-token pairs, and wall as the cross-check:

| config | walked | launched | host/step | step wall |
|---|---|---|---|---|
| stock | 2315 | 1171 | 14.0 ms | 16.8 ms |
| post-op fusion off (`GPU_DISABLE_POST_OPS_FUSIONS=1`) | 2315 | **1435** | 15.1 ms | 17.6 ms |
| gate padded to 16 | 2365 | 1219 | 15.7 ms | 17.8 ms |

Un-fusing the post-ops is the clean axis — +264 launches, not one more
primitive walked — and gives **3.3–4.1 us per launch** on top of the walk.
The budget then fixes the walk: 14.0 = 2315·a + 1171·b puts **a at 4.0–4.4 us
per walked primitive**. A primitive the plugin walks without launching costs
about half of one it launches, so the 2315 is the denominator and the 582
`Reshape`s are ~2.4 ms — 17% of the host budget, the size of the entire GDN
small-op target. They rank.

The padded configuration does not fit the two-cost model: 50 walked + 48
launched predicts ~0.4 ms and the measurement is 1.0–1.6 ms. The forty
`StridedSlice`s cost ~20 us each, five times a typical primitive — the
per-primitive cost depends on the op, and the padding's decode price is that
op specifically, which makes it fixable by a cheaper column extraction rather
than only by waiting for fusion.

**So fusion is three changes, not twenty**, and two of them may already exist
in the plugin: (1) the five `linear_attn` FCs of a GDN layer consume one input
and the plugin has a horizontal-FC fusion pass (`disable_horizontal_fc_fusion`
is a debug option) — why it does not fire here is a selector read, Phase-A
style, before any new code; (2) `DynamicQuantize` into its producer, −161; (3)
the GDN block's small ops around its FCs, ~10 launches per layer, −300. Together
that is the −50% the device bound needs (§7.0.2f: under ~545 launches or under
5.75 us each). Adding a fused primitive changes graph structure, so it needs its
own before/after on node inventory — the null-implementation control does not
cover it.

**The horizontal-FC fusion, tried (2026-08-30): it fires, it is wrong, and as
implemented it buys nothing.** The pass is general below its bound — every
later use is `fc_nodes.size()` — so the experiment was the one line
`max_num_fcs_to_fuse = 3 -> 8`, A = the pristine debug plugin, B = the patched
one, same arcint, u8, 32768. Node inventory before and after, as required for
a graph-structure change:

| | A (bound 3) | B (bound 8) |
|---|---|---|
| walked / launched | 2315 / 1173 | 2385 / **1198** |
| FC launched (in `linear_attn`) | 371 (150) | **161** (60) |
| nodes named `*_fused_*` | 10 (the attention q/k/v) | 80 |
| `Crop` launched | 0 | **110** |
| `Multiply` / `Add` launched | 40 / 104 | 80 / 134 |
| host enqueue per decode step | 12.7 ms | **14.4 ms** |
| decode, 512 tokens | 66.4 t/s | 66.1 t/s |
| prefill 12448 | 4.28 s | 4.03 s |
| greedy 96 | reference | **DIFFERENT — broken text** |

Three findings, each of which the inventory shows and the step time alone
would have hidden. (1) The fusion fired on 70 sets, not 30: the 40 MLP blocks
have four FCs on the post-attention norm too (shared-expert gate/up, the
scalar gate, one more), so FC launches fell by 210. (2) **Nothing was gained**:
the fused output's split materialised as 110 launched `Crop` kernels instead of
views, and the FCs' fused post-ops (40 `Multiply`, 30 `Add`) came back out as
kernels — net +25 launches, +70 walked, host time up 1.7 ms. The step is the
same to within noise. (3) **The output is wrong.** Byte-equality failed with
garbage text, so by the standing rule this is a different model and nothing
about it ships.

**Bisected the same morning: the GDN sets are correct, the MLP sets are the
bug.** With the raised bound restricted to non-MLP sets: 40 fused nodes, FC
launches 371 -> **281** (the −90 originally sized), **greedy 96 byte-identical
to A**. So the wrong output comes from fusing the MLP quartet, whose fourth
member is the width-1 `shared_expert_gate` — a shape the pass never meets at
bound 3 and evidently does not handle. And the correct GDN fusion is still
worth nothing: `Crop` 70 launched (two of the four pieces per layer — the
32-wide b and a projections at offsets 12288 and 12320, whose consumer is the
paged GDN primitive — materialise; the 8192- and 4096-wide pieces are views),
`Add` +30 (one per layer falls out of its FC post-op because its input is now
a crop rather than the FC), net launched 1173 -> 1198, decode 66.6 against
66.4 t/s. Patch recorded as `patches/0002-fc-horizontal-fusion-bound.patch`
and not carried; the plugin tree is back at the pinned commit.

**What that does to the fusion item.** Horizontal FC fusion on the GDN block is
worth at most ~−60 net launches even if the crop-as-view and post-op problems
were both solved, about 3% of a decode step — not the 10% it was sized at,
because the split and the lost post-op consume two thirds of the saving. The
decode launch count has to come from the classes that do not need a split:
`DynamicQuantize` into its producer (−160), the walked `Reshape`s (582, ~17% of
the host budget), and the GDN small ops. None of those is a one-line bound.

Why the crops are not views is readable: `prepare_buffer_fusing` optimises a
crop in place only when its offsets and the remaining padding are aligned for
every user (`is_optimizable_padding_for_crop`), never when a user is a `gemm`
that would then see padding, and in the dynamic case only through
`can_crop_be_optimized_simple_data_format`. The GDN split is 8192 / 4096 / 32 /
32 and the MLP one carries a width-1 column; the 32-wide and 1-wide pieces at
odd offsets are the likely failures, and it means a correct fusion still
needs a second change before it moves decode at all.

**Level-Zero, read before anyone plans on it.** The pinned plugin carries a
complete L0 runtime (`runtime/ze/`: engine, stream, kernel, memory, events;
`GPU_RT_TYPE=ZE` is an accepted build value; the package and the debug build
are `OCL`). But `ze_stream` creates **immediate** command lists only —
`zeCommandListCreateImmediate`, "submits commands immediately, no flush" — so
there is no recorded list to replay and every kernel is still appended by the
host per step. It would not touch the 76% of decode spent outside any runtime
call in any case. Not a lever in this plugin as written.

#### 7.0.2j The prefix cache in production: replayed, and two levers priced (2026-08-30)

No production traffic had ever reached an arcint unit (both journals hold
only benchmarks), so the operator's real pi session trees were replayed
offline against arcint's exact policy (`tools/cachesim.py`;
`docs/prefix-cache-production.md` has the tables and the pre-registration).
Hits on 97% of turns and 96% of prompt tokens from cache on the agent's
configuration; rewinds negligible under LRU; no rewritten-front misses in
the corpus. What remains is two things, both ours: **the snapshot grid** —
taken at the last prefill-chunk multiple, it re-prefills ~1900 tokens per
append turn on the agent where a 128-token grid would re-prefill ~970,
−28% of the agent's total prefill for one extra sub-chunk forward per
request — and **pool capacity**: 37% of the agent's prefill is re-prefilling
sessions evicted under pool pressure because the operator interleaves long
sessions, which points at a host tier for evicted pages (1.7 GB back over
the link in ~0.12 s against ~35 s of prefill), a design item. The
`--cache-reuse` question is closed on paper: no observed miss is a
near-front mismatch.

#### 7.0.2k Tool calls 500'd on the agent endpoint: the arguments contract (2026-08-30)

The first tool-using turn after the agents moved onto arcint failed with
*"Can only get item pairs from a mapping"* from the Qwen3.6 template's
`tool_call.arguments|items`. The wire format carries a tool call's arguments
as a JSON string (OpenAI's does, and every client following it); this
template iterates them as a mapping. minja detects that
(`requires_object_arguments`) and would convert — but arcint switches
`apply_polyfills` off under §3.7 so that minja never rewrites what the
template renders, and that switch also disabled the conversion. Handing the
template the type its contract asks for is input normalisation, not a
polyfill: `tool_call_arguments_for_template` now passes the parsed object
when the template's own capability flag says it wants one and the string
parses as an object, and leaves the string otherwise. Red on the live
endpoint, green on the source build with the real template, unit-tested,
shipped as 0.2.7.

#### 7.0.2l The finer snapshot grid: exact, and not yet cheap (2026-08-30)

`--cache-grid N` snapshots at the last multiple of N by cutting the chunk that
contains the snapshot point. The gate, B60, coder artifact, three prompt
lengths so the cut lands at three different offsets inside a 2048 chunk:
**warm equals cold byte for byte at every cut**, the chunk grid too, and the
u8-vs-f16 pair proved the comparison can fail (it differed at the longest
prompt). So the paged kernels are exact across a split, and §3.2's
absolute-grid rule — measured on the dense model — does not bind the paged
path. The hits land where the design says (6912, 11904; 92.6% and 96.1% of
the continuation against 82% on the chunk grid).

**And the arm lost on time.** Prefill after a fine-grid hit runs at 741–1409
t/s for ~1500 new tokens and 400–630 t/s for ~300, against 2059–2515 after a
chunk-aligned hit. A sweep over eleven hit offsets fits **~0.45 s fixed per
continuation plus ~new/1800**; the offset within the chunk does not matter.
The first hypothesis — the plugin creating oneDNN primitives for every
never-seen M — was tested with the wrong instrument: `ONEDNN_VERBOSE=1`
prints executions only, so its zero creations refute nothing. The same
continuation run again hits its own newer snapshot and cannot repeat the
forward. The mechanism is open: two candidates are the cut itself (two
forwards where one was) and a forward starting at a non-aligned past, and the
instrument that separates them is the plugin's host-time profile per request.
Until then the default stays the chunk; the flag is there for the
measurement. One more thing the sweep showed: the 21k-token prompt's snapshot
never hit on the next request, unexplained.

#### 7.0.2m The think block was arcint's to close, not the model's (2026-08-30)

An agent transcript showed the model's reasoning arriving inside `content`,
ending in a bare `</think>` — "qwen cannot not think". It can; the template
opens the block *in the prompt* (the generation prompt ends with
`<|im_start|>assistant\n<think>\n` unless `enable_thinking` is explicitly
false), so the answer begins inside it, and only the server knows that. vLLM
and llama.cpp split everything before the first `</think>` into
`reasoning_content`; arcint returned the lot as content, which the Prüfstand
harness had been quietly stripping for a week. `split_reasoning` and
`ReasoningStreamer` (core, unit-tested including a close tag straddling two
pieces) now do the split in both emit paths, keyed on whether the rendered
prompt ended inside a block; tool-call parsing sees the content part only;
streaming sends `reasoning_content` deltas first. Verified on the real
template: reasoning and content separated, `enable_thinking:false` gives plain
content and no reasoning field, no `</think>` leaks. Shipped as 0.2.8.

**Where the operator's thinking knob breaks, read end to end.** pi shows the
knob only for models flagged `reasoning: true`; its discovery package flags
that from a registry's `supported_parameters`, which a local endpoint has
none of, so the `[local]` entries come up without it and the knob never
appears. Once flagged, pi's `openai-completions` provider sends the level as
`reasoning_effort` by default, or — with `compat.thinkingFormat:
"qwen-chat-template"` — as `chat_template_kwargs.enable_thinking: !!level`,
which is the fleet-standard switch this server already honours. `reasoning_effort`
is now accepted too (an effort level is on, `none` is off; the precise switch
wins when both arrive). What no switch can express: Qwen3.6's template knows
on and off, so "medium" can only mean a **server-side reasoning budget** — a
cap on think-block tokens after which the server closes the block itself —
and that is a feature with a number to choose, not a flag.

#### 7.0.2n The reconstructed MTP head pairs with Intel's public Qwen3.8 IR (2026-08-30)

`OpenVINO/Qwen3.8-27B-int4-ov` — Intel's own export of the same checkpoint,
same geometry, same chat template and tokenizer (hashes identical to our
entry's), a different quantisation of the body — is allowlisted as its own
entry, `qwen3.8-27b-intel-int4`, never as an alias of `qwen38-b7c1-ov`: that
entry's status was measured on our AWQ export and an alias would assert it
for a file nobody measured. The head from `tools/export_mtp.py` was copied
beside Intel's IR (the loader keys `has_mtp_head` on the two files, nothing
pairs them), and the oracle is the one the exporter's docstring states: a
wrong head cannot change the answer, only depress acceptance.

**It pairs.** Intel's IR with the reconstructed head, `--mtp on`, B60, greedy,
thinking off: **draft accept 96.3% (157/163)** on a code prompt, **77.3%
(140/181)** on prose, 34.6–37.4 t/s decode — against ~0% for a head that does
not belong and 93.2% for the same head on our own export. The head consumes
the final hidden state and carries its own lm_head, which is exactly why a
re-quantised body underneath it should not matter, and did not.

**And it passes the bar.** The acceptance task on Intel's IR with the head,
`--mtp on`, greedy, thinking off: **10/10**, 622 tokens at 36.3 t/s, **draft
accept 90.8% (296/326)** — the same class as our own export's 93.2%. The entry's
status carries it.

**The net effect, measured the way the last such measurement was not.** The
operator's caveat: a previous speculation measurement ended unfavourable
because plain decoding beat drafting-plus-verifying. So three arms on Intel's
body, same two prompts, greedy, thinking off, 320 tokens, each arm run twice
(within-arm outputs identical):

| arm | code prompt | prose prompt |
|---|---|---|
| `--mtp off` | **25.0 t/s** | **24.9 t/s** |
| `--mtp on`, reconstructed layer | 36.9–37.3 t/s, accept 96.3% | 33.8–34.0 t/s, accept 77.3% |
| `--mtp on`, Intel's exported layer + our lm_head | **37.7–38.1 t/s**, accept 93.9% | **34.3–34.9 t/s**, accept 76.4% |

Speculation wins here: **+48% on code, +36% on prose** over plain decoding,
with the verify pass at ~7.1–7.9 s of the 8.4–9.5 s step total. Intel's own
layer, fed through the contract it declares (`inputs_embeds`, an i64 2-D
ones-mask, i64 positions — `--mtp-layer exported`), pairs with our lm_head
at two to three points lower acceptance and two to three percent higher
speed, its layer being int4 against our f16 reconstruction. So the lm_head
half is the whole of what the public IR lacks.

**Greedy outputs are not byte-identical between `--mtp off` and `--mtp on`**,
and it is the same divergence the paged+MTP path already carries in its status
line ("greedy is deterministic per configuration"): the answers agree for
1102 of 1717 characters on the prose prompt and then take a different but
equivalent phrasing at one near-tie token ("using a quadratic function to
determine the next offset" against "checking slots at increasing quadratic
distances"); on the code prompt the reconstructed layer's answer is identical
to plain decoding and Intel's layer's inserts one comment line. Both layers
diverge at the *same* point with the *same* alternative on the prose prompt,
which places the cause in the verify pass — the main model's M=2 forward
landing a near-tie differently from its M=1 step — and not in either head.
Acceptance stays the oracle for the head; the off/on comparison is a
different question and is recorded as such.

**And the premise of the handover has moved.** Intel's export directory
carries `openvino_mtp_model.{xml,bin}` of its own — 263 MB int4, one MTP
layer, the same input interface as our reconstructed layer (`hidden_states`,
`inputs_embeds`, `attention_mask`, `position_ids`, `beam_idx` → hidden), and
**no lm_head**. optimum-intel's development branch exports the head now; what
it does not export is the lm_head the draft needs to become a token, and
arcint's speculative path keys on `openvino_mtp_layer` + `openvino_mtp_lm_head`
rather than on Intel's file. So the standalone contribution is smaller and
more precise than "the head nobody publishes": it is the **lm_head half plus
the serving path**, and the open question is whether Intel's own MTP layer
paired with our lm_head graph reaches the same acceptance — a one-load
measurement, not done tonight.

#### 7.0.2o The Qwen3.6 MTP head: right first time, and speculation loses on the MoE (2026-08-30)

The 3.6 checkpoints carry the head — 19 `mtp.*` tensors — but its MLP is a
mixture of experts (256 experts, top-8, fused `gate_up_proj [E, 2I, H]` and
`down_proj [E, H, I]`, a sigmoid-gated shared expert), where the 3.8's is a
dense SwiGLU. `tools/export_mtp.py` now takes its geometry from the config
and builds the MoE block as every expert computed for every token with the
non-selected weights exactly zero; the lm_head half is the base IR's own,
cloned int8 (509 MB); the layer is 1.69 GB f16. Served from a directory of
symlinks plus the head, as `qwen3.6-35b-a3b-mtp`, so the agent's production
directory never acquires a head that `--mtp auto` would switch on unmeasured.

**The head is right.** B60, greedy, thinking off, 320 tokens, twice: draft
acceptance **93.9% (155/165)** on code and **75.4% (138/183)** on prose — the
same band as the 3.8's head — with no ablation needed; the zero-centred norms,
the per-head q/gate interleave and the top-k renormalisation all carried over
from the 3.8 reconstruction as they were.

**And it loses, by 30%.** `--mtp off` decodes at **71.5 t/s**; `--mtp on` at
**48–53 t/s**. The operator's earlier finding on this model — generating
tokens was faster than verifying predictions — reproduced with numbers:

| | per call | against a plain step |
|---|---|---|
| plain decode step (M=1) | 14.0 ms | 1.0× |
| verify forward (M=2) | 4.18 s / 165 = **25.3 ms** | **1.8×** |
| the head's draft | (6.03 − 4.18) s / 165 = **11.2 ms** | 0.8× |

Per accepted pair that is ~36.5 ms for ~1.9 tokens, 19 ms a token against 14
plain. The head reads 1.69 GB of f16 expert weights per draft, which an int4,
top-k-gathered head would cut to a few percent of that; fixing the head alone
brings the pair to ~27 ms, 14 ms a token: break-even. **The sentence that
stood here — that the M=2 forward runs the prefill path — was an inference
from the 1.8x and is retracted the same evening**: the pinned plugin already
routes token counts up to 32 through a batched-GEMV decode path written for
exactly this case, and the trace of an actual two-token verify shows it in
use: **the verify costs the device 10.0 ms against a plain step's 8.7 —
1.15x** — while its wall is 25.3 against 14.0. The 1.8x is host and
synchronisation in the speculation loop, 15.3 ms of it per verify against
5.3 per plain step, plus a draft that is 6.7 ms of device and 4.5 of host.
Per accepted pair 36.5 ms of wall for 16.7 ms of device; device-bound it
would be 8.8 ms a token, +59%. The levers are arcint's verify loop and the
head's host side first, the int4 head third (`docs/moe-m2-path.md`). The
agent unit stays as it is until the loop is attributed and cut.

Greedy answers with and without the head differ at one near-tie token late
in each answer (chars 621 and 1182 of ~1400 and ~1830; equivalent phrasing),
the verify-pass property already recorded for the 3.8.

#### 7.0.2p The M=2 loss found: 20,480 subbuffer creations per verify forward (2026-08-30)

The follow-up to 7.0.2o inverted twice under measurement. The "+59%" device
budget was measured against the wrong baseline (on device time the head is a
wash, 8.6 ms a token against 8.7 plain), and the "~10 ms unattributed in the
serving loop" was not in the loop: the plugin's own host-time profile at
level 2, differenced over two run lengths so probes and prefill cancel,
put the verify forward at 27.4 ms of *enqueue* and 1.1 ms of wait against a
plain step's 12.6 + 1.15 — everything arcint does outside `infer()` is under
a millisecond per token. Two mechanisms were then falsified by direct test
(105 blocking `clEnqueueMapBuffer` reads per verify: removing all of them
with USM-host index inputs bought 1.1 ms; the per-infer PagedAttention impl
rebuild: ~0.1 ms), and per-stage timers in the debug plugin found the real
one: at `token_num > 1` the MoE implementation rebuilds its per-expert mask
subbuffers on every infer — 512 `create_subbuffer` calls per layer, 20,480
per two-token forward, ~9 ms — for a prefill fallback that the batched-GEMV
path it takes never reads. At one token the block is skipped, which is why
no plain step ever showed it.

patches/0003 skips the masks below the GEMV threshold and creates them
lazily in the fallback. Byte-identical outputs both arms; the verify
forward drops 27.3 → 18.1 ms and `--mtp on` goes 44-46 → 60.5-61.2 t/s on
prose (80.7% acceptance) against 61.7-62.3 plain — a wash. With the int4
head (NNCF INT4_ASYM g64, 1.69 GB → 455 MB) the code prompt runs 72.9 t/s
at 84.0% acceptance, +17% over plain: the first configuration in which the
35B's speculation wins. Acceptance under int4 moves both ways by prompt
(71.4% on prose); Prüfstand before any card changes. None of this is in
production until the plugin fix ships. The head's remaining cost is host:
5.05 ms per draft, half of it shape inference over 123 primitives. Full
readings: docs/moe-m2-path.md.

#### 7.0.2r DFlash2: the external-drafter hook gets a real drafter (2026-09-01)

The public block-diffusion head for the 3.8 went from HF checkpoint to a
served drafter in one day because every stage had a gate: offline pairing
probe (3.39/3.76 per cycle against our int4 target, shuffled-features null
at ~1.1), OV export cycle-exact with the torch reimplementation, GPU-f16
cycle-exact with CPU after the residual-range fixes (the head peaks at ~128k
and f16 ends at 65504; exact rms identities — a 1/4 writer fold and per-norm
pre-scales with eps·pre² — fixed what input scaling and a naive 1/64 fold
measurably broke), and a serving grid: **44.8 t/s against 24.0 plain and
33.0 with the MTP head** on the B60 at 3.13 accepted per verify cycle,
byte-identical across draft placement (same card vs A770; the A770 arm
trades 5 t/s for 35k tokens of context headroom — 171,904 vs 136,640 with
the draft resident, 199,712 plain). `--dflash DIR`, one drafter per server,
greedy-only like the MTP head. Details and the open items:
docs/dflash-pairing-probe.md.

#### 7.0.2q Serving defaults: the operator layer (2026-09-01)

The MTP drafter engages only under greedy (accept-only-if-equal), so a third
of the throughput on a speculating endpoint depended on whether the caller
happened to send `temperature: 0` — measured 36.2 against 24.2 t/s on the
same server, same prompt. The chain had three layers (family card → artifact
→ request) and the missing one was the operator: the workaround in the field
was editing an allowlisted artifact's `generation_config.json`, which flips
the served provenance to "artifact" for what is an operator preference.

Resolution: flags, not a file — `--temp`, `--top-p`, `--top-k`,
`--repetition-penalty`, `--presence-penalty`, and
`--chat-template-kwarg enable_thinking=BOOL`. Precedence request > flags >
artifact > family card, each layer overriding only what it sets; any sampler
flag turns `/props`' `sampler_defaults.provenance` to `"operator"`;
validation is shared with the request path, so an out-of-range flag refuses
at boot with the wording a client would get at 400. No `--min-p`, because the
sampler does not implement `min_p` and a flag for an unimplemented knob would
be a lie; `--chat-template-kwarg` rejects every key but `enable_thinking` for
the same reason. `presence_penalty` is now also read from the artifact's
`generation_config.json` (it was implemented, request-settable, and silently
dropped on load — an oversight, mirrored in).

The regime is also visible per response now: `usage.completion_tokens_details`
carries `accepted_prediction_tokens` / `rejected_prediction_tokens` (OpenAI's
own field names), so a caller paying 24 t/s on a 36 t/s server can see the
zeros instead of inferring them from a log line it cannot read. The MTP
acceptance rule itself is untouched.

#### 7.0.2b The XMX question cannot be decided from the profile

The proposed five-minute check — grep a `ARCINT_PROFILE` for `dpas`/systolic
markers in the GEMM kernel names — was run on both cards and is **inconclusive
at this observability level**: the distinct GEMM `exec_type` strings are
identical on the A770 and the B60 (`jit:gemm:any__i8`, `gemm_tiled_opt__f32`,
`ocl::moe::moe_3gemm_swiglu_opt___f16`) and none carries any ISA marker.
Whether the B60 engages its matrix engines where the A770 could not is a real
question with a real consequence (per-card tuning of prefill and the lm_head),
but answering it needs instruction-level tracing (onetrace / Level-Zero PTI),
not `PERF_COUNT`. Recorded so the grep is not proposed again. What decode
numbers say regardless: nothing in the decode profiles is matrix-engine-shaped
— chase bytes, not TOPS.

**Re-asked 2026-08-30 for the f16 attention path, and the answer is that the
instrument cannot tell — with a reason that is more useful than the answer.**
The open question was narrower than "does XMX engage": the int8 GEMM path is
very probably matrix-fed already (331 JIT nodes at ~150 int8-TOPS is not
reachable otherwise), so what was open was the f16 attention kernel, where the
depth curve's quadratic coefficient suggested SIMD peak. That is an inference
from a rate, and two inferences pointing the same way are still not an
observation, so the check had to be a **marker**: the kernel identity, not a
throughput.

Three things were established without taking a card:

1. **No `exec_type` in any stored profile carries an ISA marker** — not for the
   GEMMs (already known) and not for attention. Every string names a kernel
   family and a dtype: `ocl::paged_attention::opt__f16`,
   `ocl::paged_gated_delta_net::opt___f16`, `jit:gemm:any__i8`. Nothing about
   `dpas`, `xmx` or `systolic`.

   **And for attention this is structural, not a gap in the profiler** (read off
   the selector, 2026-08-30). `paged_attention_opt.cpp` *is* the implementation;
   micro SDPA is chosen **inside** it —
   `rt_params->use_micro_sdpa = can_use_micro_sdpa_for(...)` — so `exec_type`
   reads `ocl::paged_attention::opt__f16` whichever branch runs. No amount of
   profile-name reading can ever separate the two. Written down so the grep is
   not re-run in the hope of a different day.
2. **The plugin cannot be asked.** This build has the GPU debug capabilities
   compiled out — zero occurrences of `OV_GPU_Verbose`, `DumpSources` or
   `ENABLE_DEBUG_CAPS` in the binary, against exactly two `OV_GPU_*` knobs
   present. There is no verbose kernel-selection log to turn on, so the
   implementation the plugin chose cannot be named at runtime.
3. **The kernel sources are not recoverable from the binary either** — no
   OpenCL source survives string extraction, so the `intel_sub_group_matrix_
   multiply_accumulate` intrinsic cannot be looked for in the attention kernel
   the way it can in a source tree.

What the binary *does* contain, stated as fact and not as an answer: DPAS
material exists, and it is oneDNN/CM microkernel infrastructure — an int8
builtin (`__builtin_IB_sub_group_idpas_s8_s8_8_1`), a bf16 vISA fragment, a
`CM_HAS_DPAS` assertion. That is consistent with the int8 GEMM path being
matrix-fed and says **nothing either way** about the f16 attention kernel, which
is a different implementation family (`ocl::` cldnn) from the one those
microkernels serve (`jit:gemm`, oneDNN). Four apparent "SDPA + dpas" hits are
false positives: `ov::pass::SDPAScaleFusion` contains the letters.

So: **outcome three from the plugin binary** — and then a cheaper instrument
answered it outright, without a plugin rebuild, `ENABLE_DEBUG_CAPS` or onetrace.
**The installed IGC carries its own debug keys** (`ShaderDumpEnable`,
`DumpToCustomDir`, `ShaderDumpEnableAll`, `ShaderDumpPidDisable`), so
`IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<path>` on the serving process dumps
every kernel it compiles, with generated assembly. `dpas` is then either in the
attention kernel's asm or it is not.

**Measured 2026-08-30, 24 GB card, coder artifact, one prefill request per
configuration. The answer is outcome one:**

```
sdpa_micro__prefill_10432584610579984572__sa      <- u8,  asm contains dpas
sdpa_micro__prefill_3448044640564163734__sa       <- u8,  asm contains dpas
sdpa_micro__prefill_10332065829643378056__sa      <- f16, asm contains dpas
sdpa_micro__prefill_7506610878414805460__sa       <- f16, asm contains dpas
```

Micro SDPA is compiled for prefill and its generated assembly carries `dpas`, in
**both** the KV precision the conjecture was formed on (u8) and the deployed one
(f16). The dump's control held: five `.asm` files contained `dpas`, so the marker
was detectable, and the two identified by entry name are the attention ones.

That settles the last unchecked gate — `query_microkernels_supported` returned
true — and with it the whole checklist. **Attention prefill runs on the matrix
path. The SIMD-peak conjecture is dead**, and §7.0.2c is corrected accordingly.

**What does not follow, and must not be quietly dropped.** The observation that
produced the conjecture stands: the depth curve's quadratic coefficient implies
~12 TFLOPS effective against a matrix-engine peak an order of magnitude higher.
The *mechanism* offered for it is now refuted; the *number* is unexplained.
Candidates, none measured: the FLOP arithmetic behind the coefficient is wrong;
the quadratic term contains more than attention; or attention prefill is
bandwidth-bound on KV reads rather than compute-bound, which a matrix unit does
not help. It is recorded as an open number rather than deleted along with the
theory it motivated.

**Two limits of this instrument, stated so it is not over-read.** The dump proves
*compilation and ISA*, not per-invocation dispatch — for that the debug-caps
trace or PTI remains the tool. And the compiled attention set differs between
precisions (the u8 build also carries `paged_attention_opt__multi_tokens`, the
f16 one does not), which is recorded as an observation and not interpreted.

**A near-miss of my own, recorded like the others.** The first probe listed
*filenames* for attention names and returned zero — IGC names dumps by hash, so
that measured nothing. Zero attention-named files maps exactly onto the
pre-registered outcome three, whose reading was "likely the micro path". A broken
probe would have produced the right conclusion for the wrong reason, and the only
thing that caught it was the control.

**Reading the gate narrows it to one unchecked condition.** The full micro-SDPA
gate is `paged_attention_opt.cpp:1396ff`: `supports_immad`, arch ≥ `xe_hpg`, not
`xe3p`, `k_head_size == v_head_size`, head size within the ceiling, no scores
output, no score aggregation, no alibi, and `valid_micro_stage` admitting
`PREFILL` and `MIXED`. **Every condition checkable from the model side passes.**
The installed plugin also carries the path — `micro_sdpa` appears six times in
the plugin binary and `sdpa_micro` once — so `ENABLE_ONEDNN_FOR_GPU` was on at
build time.

That inverts the expectation: **micro SDPA is probably already running on our
prefill, and the SIMD-peak conjecture is probably wrong.** Labelled as *a
reading, not a measurement* — it stacks three inferences (a source gate, string
presence, an architecture assumption), which is exactly the standard this
section refuses elsewhere.

What it earns is a target of one: the only gate not checked is the runtime
`query_microkernels_supported(engine, config)`. And the instrument prints the
answer directly — `can_use_micro_sdpa_for` carries
`GPU_DEBUG_TRACE_DETAIL << … << "can_use_micro_sdpa = " << can_use_micro_sdpa`,
which is one line from a build with debug capabilities on. **Expect 1. If it
prints 0, the cause is `query_microkernels_supported` and nothing else**, because
every other condition is checked and passes.

**Two near-misses on the way, recorded for the same reason `SDPAScaleFusion`
was.** Both were tidy and both were wrong:

- *"Chunked prefill is excluded from micro SDPA."* PR #29137's own text says it
  does not support "partial prefill calculation". At the pinned commit the gate
  actually reads `!desc->has_token_type_ids || stage == PagedAttentionStage::
  PREFILL`; `token_type_ids` is a Gemma input, this family has none, so `MIXED`
  was already admitted.
- *"The head-size ceiling excludes us."* It was raised 256 → 512 on 2026-08-27,
  and the test is `> 256`. This model's head size is **exactly 256**, so it
  passed even before the raise — **with zero margin**. A model one step wider
  would fall off this path silently, which is worth knowing before anyone
  chooses a fourth artifact.

**A door that is closed, so nobody chases the changelog.** The three micro-SDPA
PRs merged 26–27 August are the Gemma `token_type_ids` `MIXED` fix, the head-size
raise to 512, and an `xe3p` workaround. **None of them touches this
configuration.** Bumping the pinned runtime is not a lever here, only risk.

**A correction carried in from the deployment side, because the note is easy to
misapply.** The standing observation that "the A770 never uses XMX" is about
**llama.cpp/Vulkan**: ggml gates coopmat on `INTEL_XE2`, Alchemist fails the
check, and DP4A is the path. It says nothing about OpenVINO, and it does not
make the A770 a control group for this question. A second card is only a
contrast if it is measured on the same stack.

#### 7.0.3 KV precision on the paged path — u8 is the lever, u4 is a tax

The plugin accepts f16/u8/i8/u4/i4 for `KV_CACHE_PRECISION` on the paged path,
with plugin-managed scales — the real quantised KV that §3.3 refuses to fake
with a plain cast. Measured on the paged coder (Python driver, greedy, 100
tokens, one process per cell):

| decode t/s | depth 512 | 4096 | 32768 |
|---|---|---|---|
| B60 f16 | 64.5 | 63.3 | 55.4 |
| B60 **u8** | 64.9 | 63.6 | **56.8** |
| B60 u4 | 65.2 | 65.4 | **52.0** |
| A770 f16 / u8 / u4 (512) | 43.4 / 43.5 / 43.2 | 42.4 / 42.9 / — | |

**u8 is never slower over the depths in that table, is +2.5% at 32k, and halves
KV memory** — at 262k that is 5.00 GiB → **2.83 GiB** on the coder. (2.5 GiB
was the prediction from a clean halving; the plugin's u8 layout carries
per-block scales, so the measured cost is 11.3 KiB/token against f16's 20.0 —
0.565x. The saving is 43%, not 50%.)

**The qualifier is load-bearing and was missing (added 2026-08-29 from a
deployment-side measurement, B60, coder, one lane, 262144, warm prefix so the
number is decode and not a mixture).** At **53.5k** prompt tokens the ordering
reverses: u8 49.41 t/s against f16 53.25 t/s over three runs each, spreads 0.24%
and 1.3% — **f16 7.8% faster**, an order of magnitude outside either spread.

So the table above and that measurement are both true and describe different
points on a curve: **u8 leads at 32768 by 2.5%, f16 leads at 53.5k by 7.8%, and
the crossover lies between them.** It has not been located, and locating it is
three depths of a decode sweep. What must not survive is the sentence "u8 is
never slower", which reads as a property of the precision when it is a property
of a point. **u4 quarters the memory and costs 6% at
32k**, and the mechanism is named, not narrated: the profiled 32k step puts
`PagedAttentionExtension` at 58.6 ms under f16 and **95.4 ms under u4 (+63%)**
while every other kernel line is identical to the tenth of a millisecond — the
u4 dequant path costs far more than the 4× bandwidth it saves. So u4 is a
capacity lever only, priced; u8 is the default candidate.

**The prefill half of this decision was missing, and it is not small
(measured 2026-08-29, B60, b5 coder, one lane, chunk 2048, matched token
counts).** u8 was chosen above on decode evidence — never slower, +2.5% at 32k
— and nobody ran a prefill. Against f16 on the same graph:

| prompt tokens | paged u8 | paged f16 | u8 vs f16 |
|---|---|---|---|
| 14450 | 2245.6 t/s | 2178.0 t/s | **+3.1%** |
| 57792 | 1337.3 t/s | 1601.5 t/s | **−16.5%** |
| 115564 | 850.5 t/s | 1092.0 t/s | **−22.1%** |

u8 is neutral-to-better at shallow depth and costs up to 22% of prefill at
115k, growing with depth — the opposite shape to its decode behaviour.

**The default stays u8, chosen from the reservation arithmetic rather than from
the throughput number.** u8 is 11.3 KiB/token against f16's 20; at 262144 that
is 2.83 GiB against 5.00. What that buys is capability, not speed:

| configuration | u8 | f16 |
|---|---|---|
| B60, 1 lane, 262144 | 18.99 of 22.71 GiB | 21.16 of 22.71 GiB — fits |
| B60, 2 lanes, 262144 | 22.04 of 22.71 GiB — just fits | 26.4 GiB — **refused** |
| A770, deep context | max ctx 109056 | roughly half of that |

An engine's default has to be the setting that does not refuse. So: **u8 by
default; `--paged-kv f16` is for a one-lane deep-context endpoint on a card
with room**, and there it is worth up to 22% of prefill.

> **A default chosen on what refuses, rather than on what benchmarks, survives
> losing every performance leg.** u8 has now lost both — prefill at depth and,
> since the 53.5k decode measurement, decode at depth as well — and the default
> does not move, because none of that was ever the argument for it. A default
> picked because it was fastest today would have had to be revisited twice by
> now. This is the rule to apply to the next one. `ARCINT_PAGED_KV` remains as the A/B override so a
running deployment can be measured without a config edit.

**The second number a deployer needs, and the documentation only gave the
first.** The context ceiling per lane falls 606688 → 343632 under f16 and that
is stated above. What is not is what happens to the *prefix cache*. Measured on
the coder artifact, 24 GB card, one lane, n_ctx 262144:

| | pool pages | live per lane | spare for cached prefixes |
|---|---|---|---|
| u8 | 37918 | 16386 | **21532** — ~344k tokens of reserve |
| f16 | 21477 | 16386 | **5091** — ~81k tokens of reserve |

**The mechanism, because it also says how the term scales.** Three facts
compose:

1. the pool is sized in **bytes** — whatever VRAM is left after weights,
   activations, the GDN rows and the margin — so the page *count* it buys
   depends on what a page costs;
2. live pages are a fixed **count**, `n_ctx / kv_block_tokens + 2` per lane,
   independent of precision;
3. the reserve is simply the difference.

Checked against both configurations to the page: 37918 − 16386 = 21532 and
21477 − 16386 = 5091, with `262144 / 16 + 2 = 16386` in each. So **everything
that costs bytes comes out of the reserve, because the live side is a count and
cannot absorb it.** Raising `--n-ctx` does the same thing from the other
direction — live grows linearly while the affordable total does not move — and
that yields the useful corollary:

> **The context ceiling is not a separate limit. It is the depth at which the
> reserve reaches zero.** `606688 / 16 + 2 = 37920` against 37918 affordable
> pages, and `343632 / 16 + 2 = 21479` against 21477 — the reported ceilings, to
> two pages of block rounding. A deployer raising `--n-ctx` spends the prefix
> cache first and hits the ceiling only when there is none left.

**Two numbers corrected while checking this.** u8 does not halve KV: the
plugin's u8 layout carries per-block scales, so it is 11.3 KiB/token against
f16's 20.0 — **0.565x, not 0.5**. At 262144 that is 5.00 GiB → **2.83 GiB**, not
the 2.5 GiB predicted above before it was measured. For the same reason an f16
page does not cost twice a u8 page and the count does not halve: 37918 → 21477
is **1.77x**, and the reserve falls 4.2x only because the fixed live count is
subtracted from both.

For an agent or coder workload the reserve is the term that can eat the win —
every cold prefill gets 16–34% cheaper while fewer prefixes stay resident that
would have needed no prefill at all. Which way it nets out depends on reuse, so
this is not an argument against f16; it is the second half of the choice, and a
deployer was being shown only the first.

What is measured about quality so far: greedy token streams under u8/u4
diverge from f16 within the first tokens (first divergence at token 4–37
across cells) — a numerics change of the §3.2 near-tie class, not evidence of
degradation either way. The full 10-point harness at u8 vs f16 is the
outstanding half of this protocol and needs the C++ paged port (the prototype
has no HTTP endpoint for the Prüfstand to talk to). No default changes until
that verdict exists.

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

### 7.2 Two lanes, measured (M6, 2026-08-29)

The use case this milestone was scoped to: a long-running agent session
mid-decode, a subagent firing one request at the same model, neither queued
behind the other and neither changing the other's bytes. All of the below is
the b5 coder at u8 KV, `--parallel 2`.

**The second lane is free, and that is a measurement, not a hope.** Weights are
shared between `InferRequest`s of one `CompiledModel` (two *compiles* cost
0.791 → 1.582 GiB, §7.0.2). Activations turn out to be shared too: at a
128-token probe lane 0 costs **0.617 GiB** and the second lane **0.001–0.003
GiB**, because the GPU plugin pools intermediate buffers per compiled model
rather than per request. Pricing an imaginary second peak would have halved the
admissible prefill chunk on both cards; the reservation therefore probes each
lane and uses what it finds.

That same fact has a correctness edge, and it cost a debugging session to
notice: if intermediates are pooled per model, a request's **output** tensor is
only valid until the next execution on that model *by anyone*. Two lanes
reading their logits after the turn had passed would read each other's. Hence
the turnstile (§4.1) and the copy-out inside the turn.

**What the reservation admits, per card:**

| card | lanes | requested n_ctx | served chunk | weights+graph | activations, all lanes | GDN rows/lane | max ctx/lane |
|---|---|---|---|---|---|---|---|
| B60 22.71 GiB | 2 | 40960 | 2048 | 12.83 | 3.11 | 95.6 MiB | 293376 |
| A770 15.11 GiB | 2 | 8192 | 256 | 12.83 | 0.88 | 95.6 MiB | 44608 |
| A770 15.11 GiB | 2 | 40960 | 128 | 12.83 | 0.62 | 95.6 MiB | 57040 |
| A770 15.11 GiB | 2 | 65536 | — | — | — | — | **refused at startup** |

The refusal carries every term, as §7.0.2a requires:

```
could not bring up the OpenVINO executor: requested n_ctx 65536 on 2 lanes needs
1.42 GiB of KV but the reservation admits 57040 per lane (weights 12.83 +
activations 0.62 + margin 0.25 + 2 x state 0.093 of 15.11 GiB). Lower --n-ctx,
lower --parallel, or lower --prefill-chunk.
```

**The chunk has to be probed upward, and the reason is a property of the
plugin.** Its intermediate pool grows to the largest shape it has ever seen and
never shrinks, so an over-large probe is a permanent tax that no later, smaller
probe can undo. Predicting straight to chunk 1024 on the A770 left 1.87 GiB
resident and a budget of **zero** — a card that could serve nothing because of
a measurement. The peak is affine rather than linear-from-zero (0.62 GiB at 128
tokens, 0.77 at 256: most of it is fixed), so the engine now climbs by doubling,
takes a step only when the running fit says it fits with 25% headroom, and
re-fits from the two most recent points as it goes. Each card lands where its
own memory says: B60 2048, A770 256 at 8k and 128 at 40k.

**Single-stream regression, alternating A/B against the pre-M6 build** (28906-token
prompt, 200 tokens, steady state of three runs each):

| build | chunk | prefill | decode |
|---|---|---|---|
| pre-M6 (`fb99912`) | 1024 | 1644.5 t/s | **68.8 t/s** |
| M6, `--parallel 1` | 1024 | 1635.1 t/s | **68.6 t/s** |
| M6, `--parallel 2` | 1024 | 1631.8 t/s | **67.6 t/s** |
| M6, `--parallel 2` | 2048 (now admissible) | **1883.0 t/s** | 67.6–69.9 t/s |

Decode regresses **1.7%** at two lanes against a bar of 5%, and prefill is at
parity chunk-for-chunk. The last row is the shipped default and is faster than
the old one for a reason worth stating plainly: pre-M6 the from-zero slope
silently shrank a configured 2048 to 1024, which broke the engine's own rule
that only an *inadmissible* configuration is changed.

A sweep confirms the chunk is a prefill knob and nothing else — 512 / 1024 /
2048 give 1338.6 / 1639.0 / 1883.0 t/s prefill at **67.6 t/s decode
throughout**.

**One regression found and fixed en route, worth recording because the cause is
not where anyone would look.** Copying each prefill chunk's logits out of the
request cost **165 ms per chunk** — prefill 1644 → 1290 t/s at 30k depth, with
decode untouched. Reading that output back is expensive on this plugin, and
nothing sampled the intermediate chunks: only the last chunk's logits are ever
picked from. So the copy happens exactly where the value is consumed, which is
also the only place it needs protecting from the other lane.

**The agent+subagent scenario, end to end** (B60, one 28906-token session
decoding 400 tokens, five bursts of a 309-token / 48-token request beside it):

| | tokens/s | TTFT | prefill |
|---|---|---|---|
| the session, alone on the card | 68.7 | 15.34 s (1884.6 t/s) | — |
| the session, subagent active | **32.5** | — | — |
| each subagent burst | **31.1–31.3** | **0.42 s** | 734 t/s |
| both together | 63.7 | | |

Two lanes cost **7% of aggregate throughput** and split the card almost evenly.
The number the milestone asked for, measured rather than invented: the session's
**inter-token stall is p95 17 ms, max 516 ms**, over 246 stalled steps totalling
5.72 s of 12.29 s decoding. Its inter-token gap goes from 14.6 ms p50 with the
card to itself to 30.8 ms p50 / 34.8 ms p95 with a subagent on it.

The stall is a per-*token* figure, not a per-execution one: a decode step runs
two or three shared graphs (the embeddings gather, the model, and the head when
MTP is on), each takes its own turn, and what a reader feels is the sum. The
console breaks the waiting out rather than letting it hide inside whichever
phase happened to block —

```
slot 0: decode 400 tok in 12.29 s (32.5 t/s) | graph 6.39 s, embed 0.05 s, sample 0.05 s, emit 0.09 s, wait 5.72 s, other 0.00 s | stall p95 17 ms max 516 ms (246 steps, 5.72 s total)
```

**The bound behind those numbers is structural, not empirical.** A decode step
waits for at most *one* execution of the other lane, because the turnstile is
FIFO. So the p95 is one decode step of the other lane (~16 ms) and the max is
one prefill chunk (here a 309-token prompt in a single chunk, 0.42–0.51 s; at
the full 2048-token chunk it would be ~1.1 s at 1885 t/s). `--prefill-chunk` is
therefore the operator's latency knob, and §4.1 explains why it must not become
a scheduling variable instead.

**Quality, which is the part that would make all the above worthless if it
failed.** The Prüfstand harness against a two-lane server: **10/10 solo, and
10/10 on each lane when both run the task at the same time** — all three
answers byte-identical to each other. 479 tokens in 7.3 s alone, 13.9–14.2 s
each when sharing.

**One operating lesson, recorded because it cost an hour.** Two `arcint`
processes on one card under the `xe` KMD produce GPU faults, not just slow
sharing: `dmesg` showed `Check job timeout … not started`, an `Xe device
coredump`, and `reset done` naming both processes. That happened because a
measurement run was started while a previous server still held the card's
memory. The engine's own rule already covers it (§5: stop the resident services
before a depth run); it applies to arcint's own instances too.

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
shape is. The decode-specialised second compiled model that was floated here is
**dead, question answered**: constant dedup in the GPU plugin is per-compile
(`ops/constant.cpp:103-172` — read in the third outside review, and now
**measured**: compiling the 791 MB MTP layer twice takes device residency from
0.791 to 1.582 GiB, delta exactly one full copy), so two compiled models mean
two full weight allocations and 12.8 GiB
twice does not fit either card. The static-shape win goes where it is already
being collected instead: on the paged path, sequence identity is *input data*
(`past_lens`, block tables), the transposed subgraph is replaced by the paged
GDN/conv kernels, and no recompilation exists to want. `--custom-kernels`
stays as an off-by-default measurement switch.


- Exact KV block size (16 vs 32) and q8-KV numerics on Xe — benchmark at M2.
- Whether OV's fused SDPA is exposed cleanly enough for the decode graph, or
  whether the attention layers are better served by the PagedAttention op with
  arcint-owned page tables from day one.
- MoE expert placement on the 16 GB card: full-resident q4 vs a small
  host-spill tier for the 35B (the fleet's `-ncmoe`-style split, measured
  workable under llama.cpp, unproven under OV stateless graphs).
- External drafter (dflash-style) for the 3.6 pair: NInfer ships DFlash with
  draft windows up to 15 for exactly this model — evidence the payoff is
  real. Pulled forward to an M4/M5 decision rather than "someday".
  2026-09-01: for the **3.8**, the pairing is now measured, not hoped —
  the public DFlash2 head accepts 3.4–3.8 tokens per cycle against our int4
  artifact in a teacher-forced offline probe with a shuffled-features null
  control at ~1.1 (docs/dflash-pairing-probe.md). What remains is the build,
  not the question.
- KV codec beyond u8/i4: NInfer's int8 group-64 codec with a fused 256-wide
  Hadamard pre-rotation (encode fused into append, decode fused into
  attention), with published AIME/GPQA quality deltas. Recorded as the known
  upgrade path — and after the fusion-barrier measurement (§8 above), any
  custom-kernel proposal here must include a fusion-impact profile of the
  surrounding graph, not just the kernel micro-benchmark.
- MoE speculation revisit: blocked on the dense spec-paged numbers landing in
  the engine. The 1.37× verify-amortisation figure was measured on the
  *stateful* kernels and must be re-measured on the paged ones before any
  verdict is reused.
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
