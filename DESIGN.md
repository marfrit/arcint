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
is a bigger machine — and it is what took the dev host down on 2026-08-28.

**The reference on this fleet already answers this.** The serving unit
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
{"id": "coder-b5", "object": "model", "owned_by": "arcint",
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
- **In CI today**: 414 unit cases device-free (418 with the OpenVINO backend), a 64-check curl round-trip and the
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
  constraint is ZFS: `zfs_arc_max` is 40 GiB of the 62 GiB on the dev
  host, ARC
  refills to ~24 GiB within minutes of boot because every compile reads a
  12.8 GiB weights file through it, and `MemAvailable` does not count ARC as
  reclaimable. ARC does shrink under pressure, but not fast enough to cover a
  large sudden allocation.

  Measured the hard way on 2026-08-28: a 64k unchunked prefill run alongside a
  resident instance of the retired unit (~26 GiB) and a full 21 GiB **zram** swap — which is
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
  existence), A770 ≥ 17 t/s for the dense 3.8 (a bar from the GGUF/Vulkan
  agent baseline: the dense 27B's int4 IR does not fit the 16 GiB card at
  any depth, so it is not measurable there — §7.0.2ai; the dense agent is
  served on the 24 GB card).
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

### 5.1 The test ladder: what runs when

Measured practice of this campaign, not a new policy: three classes, by
cadence and card time.

| class | cadence | card time | members |
|---|---|---|---|
| Unit tests | every commit, every milestone | seconds to minutes | `arcint-test` via `ctest` in the stub build — among them config parsing and the refusal ladder (`tests/test_config.cpp`), the fit arithmetic (`tests/test_fit.cpp`), decode accounting and the cycle-profile line (`tests/test_decode_stats.cpp`, `tests/test_profile_cycle.cpp`), a 64-check curl round-trip (ctest's `roundtrip`) and the lane-accounting stress test (ctest's `stress`, `tests/concurrency/stress.sh`, stub-only). Plus the plugin unit tests each patch carries (`ov_gpu_unit_tests` filters: `patches_0015_paged_attention_*`, `regression_paged_attention_*` and 0016's own review suites, `moe_otd_perf_counters.*` from 0017) — these need a card but run in minutes, red-first wherever the test reports a found defect (construction locks say so), and run whenever their own patch changes |
| Milestone gates | once per milestone increment | one card window, minutes | `tests/equivalence/run.sh` and `tests/concurrency/run.py` on the configuration the milestone changes (M9: the two offload configurations; M11: drafter on/off at the depth in question), plus the milestone's own measurement cell (M14's reference cell, M11's step profile), one process per configuration, numbers into §7 |
| Acceptance | once per release, before the tag | hours | <!-- BEGIN GENERATED by tools/acceptance_manifest.py; verify with its check mode -->`coder-offload-1lane` gates byte-equality: cold vs warm cache, chunked vs unchunked prefill, one chunk size vs another (reports chunk sweep) (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-offload-2lane` gates the same byte-equality claims as coder-offload-1lane, held at --parallel 2 (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-offload-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `coder-served-large` gates byte-equality: two greedy runs, cold vs warm cache, a restored continuation vs cold, and (draft 4) a copy-the-input prompt (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-served-large-decode` gates byte-identity of the two requests' outputs; the warm second-request decode rate against its reference once filled (reports until then, §8.8) (reports 0.3.0 record (DESIGN §7.0.2ai): 53.4 t/s cold / 69.2 t/s warm decode) (references: decode-warm-2nd 66.5 t/s (gate lower-is-worse at 60.0 t/s); decode-cold-1st 66.6 t/s (report only, not gated); prefill-warm-2nd 2820.8 t/s (report only, not gated); prefill-cold-1st 2812.3 t/s (report only, not gated)); `coder-served-small` gates byte-equality (as coder-served-large), on the 16 GiB card (expected skip: mtp-section (coder artifact carries no MTP head)); `coder-served-small-decode` gates byte-identity of the two requests' outputs; the warm second-request decode rate against its reference once filled (reports until then, §8.8) (reports 0.3.0 record (DESIGN §7.0.2ai): 48.0/49.5 t/s decode) (references: decode-warm-2nd 47.6 t/s (report only, not gated); decode-cold-1st 46.0 t/s (report only, not gated); prefill-warm-2nd 508.9 t/s (report only, not gated); prefill-cold-1st 510.5 t/s (report only, not gated)); `coder-served-small-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `agent-dense` gates MTP identity (equivalence's MTP section) and MTP acceptance > 10%; `agent-dense-concurrency` gates no cross-slot bleed, cold/warm per lane, cancellation, admission (§4.1); `tier-reference-cell` gates tier ON byte-identical to itself across processes and requests, tier OFF likewise, and E2 (reports ON vs OFF identity and the first divergence; every output's hash; the first process's rates beside the second's) (references: decode-warm-2nd-on 16.4 t/s (gate lower-is-worse at 14.8 t/s); decode-ratio-on-off 1.31 ratio (gate lower-is-worse at 1.17 ratio); decode-warm-2nd-off 11.9 t/s (report only, not gated); prefill-warm-2nd-on 26.3 t/s (report only, not gated); prefill-warm-2nd-off 85.2 t/s (report only, not gated); grouped-fallbacks-on 400 count (report only, not gated)); `ngram-determinism-repeat` gates six fresh processes produce identical output; `depth-ladder` gates the load completes at both KV precisions on both cards (reports t/s per card and precision) (references: prefill-large-u8 1025.5 t/s (report only, not gated); decode-large-u8 20.8 t/s (report only, not gated); prefill-large-u8i4 120.9 t/s (report only, not gated); decode-large-u8i4 45.9 t/s (report only, not gated); prefill-small-u8 621.0 t/s (report only, not gated); decode-small-u8 29.2 t/s (report only, not gated); prefill-small-u8i4 170.3 t/s (report only, not gated); decode-small-u8i4 26.1 t/s (report only, not gated)); `sanitizers` gates zero sanitizer reports (reports device-free build only (ARCINT_OPENVINO=OFF): the 4 rope-precision cases gated on ARCINT_OPENVINO are not instrumented by this cell); `package-build` gates the package build succeeds, is version-stamped, and its RPATH probe passes; a post-deploy smoke run follows the install, outside this cell; `pruefstand` gates 10/10 on the production coder artifact through the deployed package (references: score 10 points (gate lower-is-worse at 10 points)) <!-- END GENERATED by tools/acceptance_manifest.py --> |

The long prefills are acceptance work: they are not repeated per
milestone unless the milestone itself touches depth (M8 was the
exception, by its own fault, not by rule). Within one arm, repeated
decode samples reuse the prefix cache rather than re-prefill — a
restored continuation is byte-identical to a cold run by §3.4, so
sampling decode repeatedly against one warm prefix measures decode,
not prefill, without re-paying it — except with `--moe-cpu-tier`,
where §7.0.2ae measured the restore forking; there each sample is
cold, or the tier's history is stated.

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

The 0.3.0 extension series (M7–M14: auto-fit, asymmetric KV, expert-offload
v2, sub-4-bit experts, tree drafting, exporter lowering, projector-off
loading, a vendored CPU compute tier) is planned in
`docs/milestone-0.3.0.md`; milestones join this table as they close. The
lines after it are recorded in `docs/milestone-0.4.0.md` (open and run GGUF
checkpoints; plugin patches in scope, not gated by hardware) and
`docs/milestone-0.5.0.md` (Qwen Flash Next; its recon pins the checkpoint
first) — charters with gates, nothing started as of 2026-09-05.

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

The retired unit, on the fleet, serves **the same IR file** (`qwen36-coder-b5-ov`)
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

#### 7.0.2s Expert offload measured: a fit lever, not a context dial (2026-09-01)

The economic idea is sound — rarely-routed experts do not earn their VRAM,
and at u8 KV every GiB freed is ~92k tokens of context — and the plugin's
mechanism for it exists (`OFFLOAD_RATIO`: % of experts on disk, GPU-resident
LRU slots for the rest, semantics confirmed in `ops/moe.cpp`). The sweep on
the coder (A770, ratios 0/10/25/50) says the implementation does not deliver
the trade:

- Any nonzero ratio reports ~1.20 GiB device-resident at load regardless of
  the ratio, because the OTD slot buffers commit physical memory lazily. The
  reservation then promises max ctx 1,172,016 — a number that would OOM as
  the slots fill. arcint now warns at load; set `--n-ctx` explicitly.
  *(Superseded 2026-09-02: the warning became a measured reservation, and the
  1.20 GiB was only the VRAM half of the story — §7.0.2t.)*
- Decode collapses from ~66 to 1.9–2.3 t/s at every ratio. The runtime's own
  counters (`MOE_OTD_PERF_LOG=1`, 16-token probe at ratio 10): gpu_hit_rate
  74% where ~90% capacity should give far more, and the wall time is not
  disk (27 µs avg, 1.2 s total) but **synchronous host-to-device slot
  uploads: 309 µs per tensor, 13.1 s of gpu_copy for 16 tokens**.

So `--offload-ratio` today is what M5 used it as — the way a model that does
not fit runs at all — and not a context-for-VRAM dial. The idea stays on the
table as plugin work: the upload path wants batching/async prefetch (the
same pattern Qwen3.8-Flash-Next uses for its host-resident n-gram table) and
an LRU that actually retains the hot set. Until then, context on a full card
comes from KV precision and lane budgets, not from expert eviction.
*(Superseded 2026-09-02, second half: with the slot pool placed device-side
and the upload path made asynchronous, the dial works — §7.0.2v.)*

#### 7.0.2t The honest reservation, and where the slot pool actually lives (2026-09-02)

M7 of the 0.3.0 series (`docs/milestone-0.3.0.md`), implemented and measured
on both cards. The load path now prices every term it used to imply: a
second residency read after the drafter compiles (0.47 GiB on the 35B/A770
configuration; 3.16 GiB on the B60 where the MTP head and DFlash drafter
share the card), an expert-slot term under offload, an activation term
clamped at zero, and an allocate–audit–replay loop that shrinks `n_ctx` by
the measured overshoot instead of guessing.

The activation term earned a correction of its own. The first measured runs
showed it *negative* (−0.09 GiB on the 16 GiB card, −0.27 on the 24 GiB
card), and the first explanation written down — LRU eviction noise in the
residency delta — was wrong: the pre-commit review found the probe's
baseline had moved past the GDN-slab allocation while an explicit
`− slab × lanes` survived in the probe's return, a double subtraction the
two readings corroborate to two decimals (0.093 and 0.296 GiB are the
slabs). The narrative is retracted per §7.0.1; the arithmetic is fixed; the
clamp stays as a guard that now logs an accounting error instead of
inventing a mechanism. With the term honest, the B60 activation reads
+0.03 GiB.

Three measurements set the shape of the final arithmetic:

- **The slot pool is host memory, not VRAM.** The plateau probe read
  0.11–0.14 GiB of device-side slot residency where the config-derived
  estimate said 12.01; the first implementation distrusted the probe and
  charged 12.01 GiB to the device budget (printed max ctx 99,696 — 35B q4,
  A770, u8 KV, one lane, ratio 20). The discriminator run — slot term forced
  to the probe figure, `--n-ctx 400000` — **served**, decoding at 2.0 t/s,
  with `drm-total-vram0` at 6.69 GiB against a promised 6.81 (0.8% of
  device total, inside the ≤2% acceptance bar) while `drm-total-gtt` held
  13.17 GiB. The probe was right; the reservation now keeps two ledgers
  (device = LRU working set, host = the pool, reported but not charged), and
  the §7.0.2s decode collapse reads differently: resident experts stream
  from host-mapped memory across PCIe on every token, so the M9 lever is
  pinning the hot set into VRAM, not only making uploads asynchronous.
- **Auto-adopt converges on the hand-tuned number.** With `--n-ctx` omitted
  on the B60 (b7c1, u8 KV, one lane), the fit clamps the artifact's 262,144
  to the admissible cap, allocates, audits, and corrects. Before the
  activation fix the estimate was 156,512 and the audit caught a real
  overshoot ("22.50 GiB resident against a 22.46 ceiling, pass 1/4"); after
  it, the estimate itself lands at 155,680 — within 0.02% of the 155,648
  the operator had found by hand — and one audit pass still trims to a
  served 155,488. Both runs end at the same number by measurement, which is
  the point: the audit corrects whatever the estimate got wrong. An
  explicit `--n-ctx` is never lowered; the fit then runs verify-only,
  trimming the reserve pages held for the prefix cache first, and refuses
  with the itemized terms only once that reserve is at zero — the last of
  four replay passes is forced to a live-only request (zero reserve)
  expressly to test that directly, rather than left to a trim sequence
  that a small overshoot need not drive to zero within the pass budget.
- **Deferred commit is now audited, not assumed.** Every offload load logs
  the committed-vs-requested comparison ("the driver reports 5.06 GiB of the
  5.34 requested — deferred commit; the reservation keeps the analytic
  figure"); under-commit is never read as free memory.

`--fit-margin-mib` (default 256, the old constant) is the only new flag;
`ARCINT_FIT_SLOT_BYTES` forces the slot term for experiments and skips the
probe. The pure arithmetic lives in `src/exec/fit.h` with the red case
pinned to the measured defect: zero-slot arithmetic must reproduce exactly
1,172,016, and a test proves that charging the host ledger to the device
budget would collapse an honest 1,159,024 to 710,992.

#### 7.0.2u Dispatch pinning is a null on a quiet host; the fusion contract, read off the card (2026-09-02)

Two answers from the enqueue-bound thread (§7.0.2f), both measured on the
B60 with b7c1 int4, u8 KV, one lane, n_ctx 131,072, host otherwise idle
(loadavg ≤ 0.8):

- **`--pin-dispatch` moves nothing here.** Per-infer stage-acc medians,
  base vs pinned to the same idle core: host-side sum 1,153 vs 1,066 µs
  (MTP-on, 98 infers) and 9,659 vs 9,109 µs (plain, 65 infers); wall 25.6
  vs 25.7 and 22.4 vs 22.7 t/s. The flag stays as the opt-in instrument for
  the contended-host arm, which remains untested by choice. A caution that
  cost two wrong readings elsewhere: `OV_GPU_HOST_TIME_PROFILING=3` itself
  triples plain decode (22.4 t/s instrumented vs 71.5 without), so
  instrumented and uninstrumented columns never mix.
- **The MoE fusion entry is the tiled block, and the card proves it.** The
  pass that production actually exercises is
  `ConvertTiledMoeBlockToGatherMatmuls` (gated on XMX/oneDNN; weight types
  u4/i4/i8/u8), whose source pattern is the optimum-intel export shape:
  `Reshape→Tile→Reshape`, three bias-free per-stack GEMMs over rank-4
  grouped-quant weights with an explicit rank-4→3 `Reshape` after the
  dequant chain, and a `ScatterElementsUpdate→Transpose→Reshape` router —
  no OneHot, no per-expert `NonZero` (the unrolled `FuseMOE` pattern is a
  different pass the GPU pipeline does not run). Positive control: the
  served 35B's compiled graph on the A770 carries `MOECompressed`,
  `moe_3gemm_fused_compressed` and `moe_router_fused`, forty of each — one
  per MoE layer. The exporter now emits this tiled form on request
  (`--moe-lowering tiled`, batched default unchanged), and an op-level diff
  against the fusing artifact fixed what a first replica got wrong — the
  router must run on the same rank-2 flattened tensor the Tile entry
  consumes (a rank-3 router crashed the pass's own rewrite on GPU), the
  quantization is genuinely grouped (group 64) behind the literal
  two-Convert f16 decompress chain, and the mixing step goes rank-4 before
  the reduce. The real 35B MTP head exports tiled in 5 s at 946 MB u8
  against the batched form's 1.69 GB f16 and compiles clean at production
  dimensions — but no fused MOE node appears when that head is compiled
  standalone through the nightly runtime. Fusion has so far been
  demonstrated only inside the engine's own compile of the base model
  (packaged plugin, paged pipeline context), so the head's fused serving —
  the fix for §7.0.2o's dense read — stays open behind one named
  discriminator: compile the head through the packaged plugin inside the
  engine's pipeline rather than the tooling runtime.

#### 7.0.2v The offload dial works after all: device-tier slots, measured (2026-09-02)

§7.0.2s closed with "a fit lever, not a context dial." That verdict is
overturned by four plugin patches (`patches/0004`–`0007`: perf counters, a
device-resident two-tier slot pool behind a per-compile byte budget, async
batched miss-uploads through a provider-owned staging ring, and the removal
of an unconditional per-MoE-layer `stream.finish()` on in-order queues) plus
the §7.0.2t root cause: the slot pool sat in host memory *by construction* —
the allocator's lockable-preferred path returns `usm_host`, with no OTD
special case, so every resident expert streamed across PCIe on every token.

The sweep (35B q4, the 16 GiB card, u8 KV, one lane, n_ctx 65,536, 16-token
probe, `MOE_OTD_DEVICE_POOL_BYTES` = 8 GiB where capped):

| configuration | decode | VRAM / GTT (fdinfo) |
|---|---|---|
| ratio 25, unpatched plugin | 0.4 t/s | 2.8 / 12.2 GiB |
| ratio 25, patched, pool 0 | 2.5 t/s | 2.8 / 12.3 GiB |
| ratio 10, 8 GiB pool | 4.2 t/s | 10.8 / 6.8 GiB |
| ratio 25, 8 GiB pool | 4.9 t/s | 10.8 / 4.3 GiB |
| ratio 50, 8 GiB pool | **9.1 t/s** | 10.8 / **0.36** GiB |

The async/finish patches alone are 6× on the host path; the device tier
takes the 35B on this card from 0.4 to 9.1 t/s — 23× — with the fdinfo
VRAM/GTT split flipping exactly as the pool budget dictates (the designed
red-first observable: an unpatched or misconfigured world cannot show the
flip). At ratio 50 the whole remaining pool fits the device tier and GTT
drops to 0.36 GiB. The fusion-impact obligation is discharged: the base
model still fuses ×40 under the patched plugin (graph dump). The property is
opt-in, default 0 — behavior without it is byte-for-byte the pre-patch path.

*(Flagged 2026-09-02, same day: a later four-cell grid found the then-live
plugin serving deterministic garbage under u8 KV, putting this table under
suspicion. Resolved the same day by a four-way plugin bisect: the build
carrying only 0004–0007 — the patches measured here — serves coherently;
the corruption enters with the later asymmetric-KV patches (0008/0009,
§7.0.2w). The numbers above were measured on a healthy path and stand. The
owed equivalence gates fired early and did their job either way.)*

The counters told the rest once their print path was fixed (it was gated on
the plugin's verbosity level, not on the collection flag — found in review,
routed through an always-on channel). One ratio-25 capped run: `gpu_hits
11,695 / misses 5,286` (68.9%, consistent with §7.0.2s's 74%),
`tensor_loads 47,574 = misses × 9` — nine constants per missed expert,
three GEMMs times weight/scale/zero-point — which resolves §7.0.2s's
reconciliation flag without a retraction: 42.4k was tensors, ~12k was
experts, the multiplier is 9. `avg_gpu_copy_us = 5` against the old 309
µs/tensor; `device_slot_buffers 241 / host 119` under the 8 GiB cap;
`alloc_fallbacks 0`; and `evictions = 0`, which by the design's own
decision rule (pin only if evictions exceed 5% of acquisitions) defers the
hot-set pin with a measurement rather than a hunch. Still owed: the
equivalence/Prüfstand gates for the patched paths before any production
rollout.

The same window also proved the M8 groundwork's fail-loud ladder on real
hardware: an unpatched plugin answers `--paged-kv u8:i4` with "Option not
found: VALUE_CACHE_PRECISION … refusing rather than silently falling back",
and the per-port audit caught a genuine surprise on the way — the plugin
compiles a u8 KV request as i8 storage ports (same width, its own
signedness), which the audit now accepts as an alias while still refusing
any actual bitwidth change. And a boundary for the head-fusion question
(§7.0.2u): the tooling runtime *does* fuse the base model when compiled from
Python with the offload properties (forty layers, positive control), while
the exported head — structurally identical to the fusing block under every
constraint a source-level walk can express — is still rejected by the
matcher. The walk has a blind spot the matcher does not; the named next
instrument is matcher logging on a debug build, and until it speaks, the
head serves unfused.

#### 7.0.2w Asymmetric KV serves: u8 keys, i4 values, and two bugs that were never a wall (2026-09-02)

M8's premise — u8 keys / i4 values to cut KV per token — is met: `--paged-kv
u8:i4` serves coherent output on the 35B at **8.8 KiB/token against u8's 11.3,
a 22% cut**, CPU-reference-verified. Getting there required retracting two
wrong calls this section made when it first parked the work, both worth the
record.

**"The decode kernels cannot do it" was wrong.** The claim rested on one
`is_i4_u4(kv_cache_precision)` boolean gating both operands — but that is a
shared JIT *symbol*, not shared tiling. The `.cl` already reads K via
`INPUT1_TYPE` and V via `INPUT2_TYPE`, the unpack blocks are already separate,
and the workgroup dispatch is V-side-driven, so there is no symmetric-head
assumption to break. The read-side split is ~8 macro renames (`IS_INT4_K` /
`IS_INT4_V`); for u8-K/i4-V the entire V path is byte-identical to i4/i4 and
only K flips to its existing unpacked arm. Writing plugin patches is in scope
for this series, so the kernel got written, not deferred.

**"A layout-sensitive miscompile" was wrong — it was two real source bugs.**
The u8-serves-garbage symptom that parked `0008` reproduced on a *cold*
from-scratch clone, killing the build-artifact theory. Root-caused to file
lines, both matching a CPU-reference test:

- *Bug A (symmetric u8 garbage).* The value cache is physically stored **signed
  i8** (the write does `convert_char_rte`), but the value *read* does
  `convert_half` with no forced signed cast — unlike the key read, which
  hardcodes `(char)` — so its signedness follows the declared port type.
  `0008` flipped the value port from i8 to u8 for a symmetric request, so
  `0xFB (−5)` read as `251`, dequant off by ~256·scale → salad. Two-part fix:
  the finalize value-mirror reads the *swapped* key member directly
  (`m_kv_cache_precision.value`, not the getter, which shadows the member
  through `m_user_properties` until finalize completes — the subtlety that
  had defeated rounds 3–5), **and** the value read gets an explicit `(char)`
  cast so it is correct regardless of the port's declared type.
- *Bug B (u8:i4 −nan).* The read kernel split per-side but the **write** kernel
  did not — `KVCacheUpdateGenerator` gated packing on the key precision alone,
  so V was written unpacked while the read expected packed u4 nibbles →
  misaligned scale/zp → NaN at the first element. Fix: split the write path
  (`KVCacheUpdateGenerator` + `pa_kv_cache_update_ref.cl`, all three write
  stages) per operand, mirroring the read.

Verified before promotion: 210 `paged_attention` unit tests pass — including
the pre-existing u4/u8 cases through the split kernels and new symmetric-u8,
symmetric-u4 and u8-K/i4-V CPU-reference cases — and both u8 and u8:i4 serve
coherent text at multi-chunk-prefill depth. An adversarial review before
promotion caught a matrix hole (`f16:i4` and other f16-mixed pairs now refuse
loudly at finalize, not slip through), a latent unsplit reorder kernel
(unreachable in this engine, now guarded), and two narrated-not-measured
header claims (retracted per §7.0.1); the patches (`0008`/`0009`/`0010`) then
promoted from `parked/` into the applied series.

The durable arcint-side surface stands as before: `--paged-kv KEY[:VALUE]`
over `{f16,u8,i8,u4,i4}`, a per-side bitwidth audit that treats the plugin's
u8-stored-as-i8 and 4-bit-in-8-bit-typed-port conventions as measured aliases,
and a KV cost model reading the packed width (a retracted over-broad first cut
is on the record in `fit.h`). Symmetric `u4` also serves and prices at 6.3
KiB/token. Still owed, named not chased: the u8:i4 **prefill** throughput
price (micro-SDPA declines for the packing-class mismatch, so prefill takes
the opt path, unpriced against §7.0.3's depth curve) and cold-vs-warm
prefix-cache byte-exactness at u8:i4. Production still serves `+p1`
(§`project-ovsrc-plugin-modified`) and does not yet carry these — deployment
is a separate decision.

#### 7.0.2x The host compute tier: the cold tail stops crossing the bus (2026-09-02)

M14 was scoped (docs/milestone-0.3.0.md) as a hybrid decode split inside the
plugin: the fused MoE GEMV keeps every expert that hits the device slot LRU,
and the misses that would have to *evict* a slot are computed on the host
instead of uploaded, at the seam that already knows the hit/miss partition
(`try_acquire_simultaneous`, patches `0011`/`0012`). Two decisions on the
record before any number: the CPU kernels are written for the plugin's actual
grouped-int4 layout, no repack; and llama.cpp on CPU is not the bar. The
gates were equivalence and an honestly reported decode number against M9's
device-tier upload path, win or lose.

The closing window (35B int4, the 16 GiB card, u8 KV, one lane, n_ctx
65,536, an 897-token prompt, 64 greedy tokens, two probes run1 / run2 with
the last-16 rate in parentheses):

| cell (device pool, LRU slots per layer) | tier OFF | tier ON |
|---|---|---|
| ratio 50, 8 GiB, 128 slots (P\*) | 10.4 / 10.6 (12.8 / 12.7) | **15.0 / 15.5** (15.7 / 15.8) |
| ratio 75, 5 GiB, 64 slots | 7.4 / 7.5 (7.6 / 7.5) | **14.1 / 14.8** (15.0 / 15.0) |

The window was pre-registered at the milestone row's ratio 20 and ran at
ratio 50: on this card ratio 20 spills resident experts into GTT (2.1 t/s
measured OFF), so the pre-registered P\* rule — the largest evicting pool at
the fastest OFF ratio — chose ratio 50. P\* is that reference: the largest
pool that still evicts (at
ratio 50 the pool budget only decides how many of the 360 slot buffers live
in VRAM rather than GTT; the slot *count* is fixed by the ratio, so
eviction rate is 26.7% of acquisitions regardless of pool, and 8 GiB is the
fastest OFF cell). The tier wins 45% there and doubles the ratio-75 cell,
where fewer slots make the cold tail bigger. Counters (per-layer averages;
the ratio-50 process ran one probe, the ratio-75 process two), tier ON: ratio 50 routes 3.21 experts per MoE layer to the host (LRU hit rate
47.9%, eviction count 7,132 against 10,147 per probe OFF), ratio 75 routes
5.51 (hit 29.0%). Per host expert on its worker thread 270 / 305 µs; per
layer dispatch-to-join 307 / 396 µs, of which the GPU hides ~30; the x and
routing-weight readback 283 / 286 µs; the y writeback 29 / 40; pool wake
5 / 8. The tier layer costs ≈ 620 µs at P\*, and the largest single term is
now the host readback of a 4 KiB `hidden_states`, ahead of the compute it
feeds. It is not the bytes; merging its two blocking copies into one wait
took only ~60 µs off it, and what the remaining ~280 µs is made of is
unmeasured. That is the next instrument, named and not chased: decompose
the readback, then test whether a host-visible residency for the decode
input removes it.

Equivalence held on every axis the endpoint exposes: the 64-token greedy
text is byte-identical OFF vs ON at both cells with the tier firing on
40–69% of the loads (the f32-accumulating host kernel against the f16 GPU
path never flipped an argmax); the acceptance task with the tier ON at P\*
scores 10/10 (468 tokens, 100.8 s); the fused primitive exists ×40 with the
tier on (its per-impl configuration line prints exactly forty times per
process); the host kernels match a double-precision oracle within bound at
the served shape (T1 ×5); and the kernel-skip red case was *run* on a card,
not argued — with the sentinel early-return removed from the OpenCL kernel
the T2 test fails, restored it passes, and the rebuilt plugin is
byte-identical to the one the table measured.

That held on every axis this window tested; it did not test a
continuation restored from the prefix cache across the process's own
request history, and that axis does not hold — see §7.0.2ae
(2026-09-04).

Three defects stood between the first tier-ON cell and that table, each
found by a counter and fixed with a measurement, per §7.0.1:

1. **Dormant by construction.** The first tier-ON run reproduced the OFF
   counters exactly (`cpu_tier_pairs 0`, `acquisitions 75,973` both ways).
   The tier branch double-counts a miss, so identical totals mean it never
   ran once: the impl's `clone()` goes through `make_deep_copy`, which
   default-constructs the copy and carries an explicit field list, and the
   clone is what the network executes. The M14 members were configured on
   the program's impl and defaulted on the running one; they are now in the
   list. Every tier-ON number before that fix is void and is recorded as
   such in the ledger.
2. **Compiled at -Os.** With the tier alive, the host kernel measured
   2,542 µs per expert in situ against 198 in an isolated RAM-bound bench.
   Two worker-side counters (wake lag 6 µs, task 2,542 µs) put the gap in
   the kernel itself, and the plugin's own flag line explained it: the
   graph library's Release options append `-Os` after the global `-O3`, so
   the one CPU hot path in the plugin was a size-optimised build. The same
   bench at the plugin's flag order gave 2,404 µs. A per-source `-O3`,
   emitted after the target's options, brought the in-situ task to 270 µs.
   The interim verdict — the tier ties or loses at P\* and wins only at
   ratio 75 — was measured on that -Os kernel and is superseded, not
   retracted: the measurement was right, the build was wrong.
3. **The envelope's bandwidth.** The design's 50 µs/expert assumed one
   expert gets the socket's full 50 GB/s; a single thread streams about
   8.5 GB/s, and the rewritten kernel (activations de-interleaved once per
   stage, a byte-spread via shuffle, four accumulators per column, the
   horizontal sum deferred, one column per pass so nothing spills) sits at
   198 µs from RAM against the original's 520. Bench caution for the record:
   a one-expert loop lives in L3 and reports the same 189 µs for either
   regime; only a rotating set larger than L3 measures the served case.

The property is opt-in (`MOE_CPU_TIER`, arcint `--moe-cpu-tier`, off by
default, refused without `--offload-ratio`); an unpatched plugin rejects it
loudly rather than serving the upload path under a "tier on" label.
Production still serves `+p1` and does not carry these patches; deployment
is a separate decision, and the Prüfstand on the served coder artifact is
the gate it waits on.

#### 7.0.2y Context by the flag, not the format; the vision IRs never loaded (2026-09-02)

One row closed in one window, one owed gate discharged, and a third row
re-measured before it is re-scoped.

**M13.** The served checkpoints are VLM exports; every artifact directory
carries three vision IRs beside the language model (merger 443–458 MB,
tower and positional encoder a few MB). The loader has never resolved
them — artifact resolution names only the language model and the text
embeddings — so "pay zero VRAM for the modality we don't serve" was true by
construction and unrecorded. What landed is the record and the guard:
`--vision` is reserved and refused (a mistyped invocation cannot look
multimodal), the artifact scan reports the present-but-unloaded IRs at load
time. A process-level check was tried and is not evidence: after compile
the process holds neither a mapping nor a descriptor on the language-model
file either, so a vision count of zero cannot discriminate; the claim rests
on the loader's code path and the inventory line, not on that check. The scan's first version under-counted (2 files, 1.7
MiB against a 457 MB merger on disk): it abbreviated two base names, and the
test fixture mirrored the constant, so the test passed. The runtime line
caught it on the first load; names are now spelled from a listing and the
test uses the literal export names (coder: 6 files, 428.3 MiB). Not run:
byte-identity against a separate text-only export — it needs a second
export of a served checkpoint, and the language-model IR *is* the text path
of the VLM export. Flagged as a scope call rather than dropped.

**The context claim, measured on the served configurations** (patched plugin
0001–0012, one lane, `--n-ctx` omitted so the fit pass adopts the maximum
admissible depth; the device-free red case in `tests/test_fit.cpp` failed
first with every precision fed the u8 cost and passed once the cost model's
own port widths were used):

| card, model, served extras | u8 | u8:i4 | gain |
|---|---|---|---|
| 24 GB, 27B dense agent, MTP on, prefix cache 8 GiB | 155,376 | 199,424 | +28.3% |
| 16 GiB, 27B-A3B coder, no offload, prefix cache 2 GiB | 133,456 | 171,312 | +28.4% |

M7's audit number for the first row was 155,488; the fit pass reproduces it
to within one overshoot correction. The second row's u8 figure is 35,152
tokens above the hand-set 98,304 the coder serves today. The gain matches
the cost model's 11.3 → 8.8 KiB/token to within 0.1 pp. Greedy text is
byte-identical u8 vs u8:i4 at 16 tokens on both cards, and the coder's
acceptance task at u8:i4 scores 10/10 without offload and 10/10 twice with
the offload path live (ratio 20, 8 GiB device pool; the two offloaded
answers identical, the non-offloaded one differing from them at line 10 —
the runs prefilled at different chunk sizes, 128 against 2,048, and chunk
boundaries are not bit-exact on this backend, §3.2). That is the acceptance
gate the patched series owed on the served coder artifact; the M8 row's
other owed items stay owed, and deployment stays a separate decision.

**M10, before any re-scope.** The row's gate is "≥ 15% more max context than
pure int4 on the same card" through sub-4-bit expert weights. The design
study (session note, not in the repository) found that the fused MoE op
stacks each layer's experts into one tensor with one element type and one
group size, that the decode kernel's packing is keyed to two elements per
byte, and that of NNCF 3.3.0's modes only INT3_SYM/INT2_SYM are sub-4-bit in
bytes (every 4-bit codebook mode is ≥ 4.25 bpw) — a 3-bit expert type would
be an 800–1,500-line kernel divergence (HYPOTHESIS) with a decode
regression of known sign (§7.0.3's u4-KV precedent). The table above is the
measured half of the argument: the same 1.676 GiB of KV on the 24 GB card,
and the coder's budget on the 16 GiB card, both buy 28% by a flag that
already serves. A recorded survey (session note) finds no published Intel or
OpenVINO IR below int4 for any Qwen3-MoE-class model and nothing below
W4A16 for MoE on vLLM's Intel roadmap, while ik_llama.cpp's recipes and
QuantMoE-Bench show that routed experts *can* go to 2–3 bits with measured
loss and that routing-frequency-aware allocation beats uniform at matched
budget (QuantMoE-Bench's controlled comparison is against random
allocation). One caveat on the record: the installed NNCF lists INT3_SYM
and INT2_SYM in its mode enum while its public documentation does not, so
they count as undocumented until a compression call with them succeeds.
The re-scope — keep the sub-4-bit question, reassign the context claim to
the levers that earned it — is proposed to the operator with these numbers,
not decided here. Built in the same window because every candidate route
and two older rows want it: a per-expert routing histogram in the plugin
(patches/0013), counted at the acquire seam before the hit/miss split, cost
below a decode probe's resolution, and validated on the coder at ratio 20:
40 layers, 60,740 routings per layer for the same token stream, 25 of
7,360 experts never routed by the acceptance prompt.

#### 7.0.2z Drafting II, measured: the free levers, the oracle, and what they close (2026-09-03)

M11's row asks for a chain re-ranker first (tokens per verify cycle above
the measured 3.13 at byte-exact greedy), a tree path after, and an ngram
drafter for drafter-less endpoints. The design step (session note) found
that the verify runs on the paged graph, which has no mask input, so a
tree needs a new paged-attention input and kernel; that the re-ranker's
usual input, the target's logits at draft positions, exists only after
the verify; and that the one unexploited signal is the lattice itself.
Four host-side levers cost nothing to try, so they went first, with an
offline oracle bound to price the rest.

**Baseline, reproduced.** 24 GB card, dense 27B int4, u8 KV, auto-fit
136,368, one lane, the DFlash2 int4 head on the same card, greedy,
repetition penalty 1.0. Prose 400 tokens: 3.28 / 3.12 tokens per cycle at
44.6 / 45.0 t/s (the record's 3.13 / 44.8). The acceptance task with the
drafter on scores 10/10 at 4.46 tokens per cycle; the thinking template
drafts worse, 2.02 per cycle; a warm prefix-cache hit keeps the drafter
(4.36 per cycle, text identical) — the design's prediction that the first
warm hit would disable it process-wide is refuted for a full-prompt hit.

**The levers, one window, every arm with the cycle dump on** (its cost is
in every number alike; against the pre-M11 run of the same probes the
prose wall time is unchanged, 10.4 against 10.5 s, and the code probe is
0.5 s longer over 234 cycles, about 2 ms per cycle). The t/s here are
client wall including the cold prefill, not the server's decode figure.
Tokens per cycle and t/s for prose / code / thinking (the thinking probe
is capped at 2,048 tokens, the 2.02 above ran to 10,240):

| arm | prose | code | thinking | text vs baseline |
|---|---|---|---|---|
| greedy, λ=1, k=16, block 8 (baseline) | 3.28 · 38.5 | 4.46 · 60.6 | 2.98 · 41.2 | — |
| the same, re-run | 3.28 · 38.6 | 4.46 · 60.8 | 2.98 · 41.4 | identical |
| Viterbi, λ=1 | 2.63 · 32.5 | 3.40 · 46.8 | 2.17 · 30.4 | identical |
| Viterbi, λ=0 | 3.05 · 37.2 | 4.35 · 59.2 | 2.68 · 37.3 | prose differs |
| Viterbi, λ=0.5 | 2.94 · 35.3 | 3.97 · 54.7 | 2.76 · 38.4 | code differs |
| Viterbi, λ=2 | 1.99 · 25.5 | 2.35 · 32.8 | 1.80 · 25.1 | prose, code differ |
| Viterbi, k=32 | 2.61 · 31.0 | 3.38 · 45.2 | 2.17 · 29.3 | identical |
| greedy, block 10 | 3.42 · 36.0 | 4.59 · 54.3 | 2.90 · 34.6 | prose differs |
| greedy, block 12 | 3.28 · 32.7 | 4.78 · 54.2 | 3.05 · 35.2 | identical |
| greedy, block 16 | 3.25 · 30.4 | 4.83 · 49.7 | 3.22 · 34.0 | identical |
| ngram drafter, no DFlash | 1.04 · 21.3 | 1.24 · 24.9 | — | acceptance answer identical |

Three negatives, each measured. Exact MAP over the lattice under the
selector's own score accepts fewer tokens than the greedy commit on every
probe, and at λ=2 by 40%: a score that ranks candidates well per row does
not select the most acceptable chain when maximised over paths. The
unary-only path (λ=0) sits 0.1 to 0.3 tokens per cycle below the baseline,
which is what the codebook term is worth. Longer blocks move tokens per
cycle by −3% to +8% on code and thinking and lose 6 to 21% of throughput
on every probe: the longer verify costs more than the extra accepted
tokens return. The ngram drafter's prose probe (21.3 t/s client wall)
falls below the documented plain decode (24.0 t/s, server figure) while
the acceptance task stays at 10/10 with the same answer as the DFlash
arm; block 16, the arm with the most tokens per cycle on code, also scores
10/10 with the same answer. Four arms changed the served text — λ=0 on
prose, λ=0.5 on code, λ=2 on both, block 10 on prose — at one position
each, the §3.2 near-tie class; the design's gate reports that count
rather than hiding it, and the byte-identical rows are the ones the row's
"same equivalence" admits.

**The oracle prices the rest, as a floor.** Replaying the baseline's
1,046 dumped cycles offline, on the accepted-drafts-per-cycle scale: 2.339
served, and the longest prefix any selector could have chosen from the
same top-16 candidate sets is at least 3.083 — the tool knows the true
token only up to each cycle's rejection row, so +0.744 accepted per cycle
is a lower bound on the re-rank headroom, not a ceiling. A realizable
tree of width 2, 3 or 4 by unary rank would reach at least 2.489, 2.663 or
2.772 (+0.15 to +0.43) before the mandatory re-forward a tree verify
carries; ranked by the transition score the true token is rarely inside
the top few at all (0.29 to 0.69). The design's pre-registered rule for
the adapter (headroom above 0.3 with the free levers taking less than
half of it) is therefore met, not refuted: the free levers took none of
it. The adapter track — a re-fit of the selector's sidecars against this
int4 target, five to eight days, CPU-only training — is a decision for
the operator with that floor in hand, not a closure by measurement; the
tree path is not pursued, since its realizable bound sits below the
re-forward it would carry and it needs a new paged-attention input.

The harness can fail: with the accept test's equality inverted (a deliberate
one-line red, rebuilt and run on the same card), the prose probe diverges
from the baseline at its second character and acceptance collapses to 1.36
tokens per cycle at 18.6 t/s; the red ran after the window rather than
before it, and the line was restored and the tree rebuilt before the
record was closed.

What lands: the block override, the selector options (greedy stays the
default and scores in the legacy association order; against the pre-M11
run it reproduces the accepted-per-cycle counts exactly on prose, 278 of
122 cycles, and on code, 809 of 234), the cycle dump and the oracle tool,
and the window above as the record. The row's first gate — more than 3.13
tokens per cycle at the same equivalence — is met by blocks 12 and 16 on
code and thinking, at a throughput loss, and by nothing else; the served
chain is near what this head and this selector give without new weights.

#### 7.0.2aa Long context against the naive benchmark: what depth does to every number (2026-09-03)

Every drafter and KV number on the record came from prompts under 900
tokens, and the 262k-token cells of the paged path were memory and gate
measurements, not throughput at depth. The operator asked how the naive
benchmark relates to long context. Two windows answered it, and each
defect the first one hit was disclosed the same day and fixed or bounded
before the second ran.

**Protocol.** A real document (this repository's design and docs, later a
non-repeating 230k-token corpus of the same) truncated to the target depth,
followed by a one-line summarisation request; 400 greedy tokens; the
prefix cache off so no shallower cell hits a deeper one's pages; one server
process per arm with the depths run shallowest first, and one process per
depth for the u8:i4 fault bracket — the first pass ran its depths
deepest-first inside one process, and a GPU fault at the deepest cell
made every later cell fail, which was published as "u8:i4 fails past
2,048 tokens" and retracted within hours. Rates are the server's own
prefill and decode lines; "tokens per cycle" is completion tokens over
verify cycles (the accepted drafts per cycle are one less).

**The dense 27B agent on the 24 GB card, u8 KV, auto-fit depth** (prefill
t/s · decode t/s · tokens per cycle):

| arm | 8.9k tokens | 37.7k | 76.4k | 143.1k |
|---|---|---|---|---|
| plain | 935 · 22.3 | 600 · 19.9 | 402 · 16.3 | 257 · 12.9 |
| MTP head | 876 · 25.7 · 1.89 | 541 · 12.6 · 1.79 | 349 · 1.0 · 1.00 | 117 · 0.3 · 1.00 |
| DFlash2 int4 | · 19.8 · 1.69 | · 11.0 · 1.43 | · 5.3 · 1.00 | refused (over the 136,368 auto-fit) |

Plain decode loses a quarter of its rate between 9k and 76k tokens
(22.3 to 16.3, then 12.9 at 143k) and prefill throughput falls from 935
to 257 tokens per second, under the auto-fit prefill chunk of 128. DFlash
is below plain at every depth here and the MTP head from 37.7k: at 8.9k
DFlash decodes at 19.8 against plain's 22.3 while accepting 0.69 drafts
per cycle (the MTP head is still ahead there, 25.7); at 37.7k
the MTP head accepts 79% of its drafts and still decodes slower than plain
(12.6 against 19.9); at 76k both accept nothing and the draft and verify
cost leaves them at 1.0 and 5.3 t/s. The MTP arm also prefills slower
than plain, by 6% at 8.9k and 13% at 76k and by half at 143k (117 against
257 t/s), and its verify takes 0.84 s per step at 76k and 3.1 s at 143k. By
code reading the reconstructed MTP layer keeps its own unpaged state and
is given a dense mask over the whole context each step; that reading does
not account for the per-step cost or the zero acceptance, and a profile
of one MTP step at depth is owed. Why DFlash accepts nothing at 76k is not
established either. Serve deep contexts without a drafter until both are.

**The coder on the 16 GiB card, no offload** (server prefill t/s · decode
t/s):

| KV | 8.9k | 37.7k | 71.7k |
|---|---|---|---|
| u8 | 525 · 45.1 | 462 · 44.5 | 403 · 40.3 |
| u8:i4 | 491 · 47.4 | 369 · 43.3 | 234 · 39.2 |

Decode is at parity; the u8:i4 prefill price — the M8 item owed since
§7.0.2w — is +7%, +25% and +72% of prefill time at those depths. By code
reading (§7.0.2w, the plugin's paged-attention selection), micro-SDPA
declines the i4-against-u8 packing, so the u8:i4 prefill takes the opt
kernel, whose per-chunk scratch is chunk × heads × head size × 4 bytes ×
⌈past ÷ 256⌉ and is reallocated on most chunks of a long prefill — about
930 MiB at 119k tokens; that this scratch is the price and the fault is
the best-supported reading, not a measurement. What is measured: the same
119,074-token prompt prefills at u8 and fails at u8:i4 with a driver
out-of-resources error, while 71,689 tokens pass at both (u8:i4 in a
fresh process, u8 inside its pass-2 arm); the edge for u8:i4 lies between
those two depths. A chunk-size
discriminator and the fix candidates (pre-sizing the partition buffers
once per request, or an unpack shim onto the micro-SDPA path) follow in
the record.

**What the pass fixed on the way.** The DFlash2 drafter disabled itself
for the whole process on any prompt over about 2,048 tokens: the head's
2,048-row state window, an Assign whose variable layout the plugin left at
zero rows exactly when the concatenation landed on the window, and a
permanent disable in the engine. Plugin patch 0014 lets the Assign adopt a
same-type, same-rank output layout (the served head then drafts at 17k
tokens and through 3,000-token decodes); the engine's disable is per lane
and re-armed on the next request, with a feed cap one row below the window
that carries a plugin without 0014; the export's trim takes a runtime slice
start and the tool reproduces the served int4 head byte for byte. Those
are in the changelog's unreleased section with their numbers.

#### 7.0.2ab The u8:i4 edge bracketed by chunk, the belt, and the profile that cannot see depth (2026-09-03)

The chunk-size discriminator owed by §7.0.2aa ran the same afternoon, one
process per cell, the coder on the 16 GiB card at `--paged-kv u8:i4` with
the same pool in every cell (n_ctx 131,072, 8,194 pages, 1.10 GiB of KV),
the prompt length being what varied:

| prefill chunk | prompt tokens | proxy (chunk × 16 KiB × ⌈past ÷ 256⌉) | result |
|---|---|---|---|
| 128 | 119,074 | 932 MiB | driver out-of-resources (171,312-token pool; passes with a 131,072-token pool, see below) |
| 64 | 119,074 | 466 MiB | prefills in 605 s (197 t/s), decodes |
| 128 | 71,689 | 562 MiB | passes |
| 256 | ≈72,000 | 1,128 MiB | driver out-of-resources |
| 512 | 35,227 | 1,104 MiB | prefills in 233 s (151 t/s), decodes |

Halving the chunk clears the 119k prompt, and doubling it breaks the 72k
one, so the fault does move with the chunk. But the chunk-512 cell passes
with a larger proxy than the chunk-128 cell that faults, so the
§7.0.2aa reading — that this one scratch buffer is the price and the
fault — is not a threshold and is retracted as a mechanism. What the
plugin source says (read, not measured): the mixed-stage paged attention
allocates its intermediate output as tokens × *query* heads × head size ×
4 bytes × partitions, 16 query heads on the coder against its 2 KV heads,
and the past-0 prefill stage allocates none of it, which is why the first
chunk never faults; micro-SDPA selection is per device and precision, not
per chunk, so it does not explain the chunk-512 cell. Why 1,104 MiB
passes where 932 MiB faults was open until the host kernel log and a
VRAM sampler were read the same afternoon.

**The fault, measured to its root.** The host kernel log carries, at the
time of every u8:i4 fault on the 16 GiB card, the pair `xe: VM worker
error: -12` followed by `exec queue reset detected` — no GPU page fault,
no coredump (a genuinely different failure, a single 76k-token forward on
the 24 GB card, produced 308 page faults and a coredump instead). The
`-12` is ENOMEM in the xe driver's rebind worker for compute-mode VMs
(`xe_vm.c`, `preempt_rebind_work_func`), whose page-table objects are
created VRAM-only and non-evictable on discrete cards (`xe_pt.c`,
`xe_pt_create`, `XE_BO_FLAG_VRAM_IF_DGFX | XE_BO_FLAG_NO_RESV_EVICT`), so a
full card fails a rebind on its own page tables even when the user
buffers were placed. The card's allocator, sampled every two seconds on
the host (`vram_mm` in the driver's debugfs) during the 119,074-token
prompt at chunk 128: with the 131,072-token pool free VRAM falls
monotonically from 1,319 MiB to a minimum of 492 MiB at the end of the
prefill and the prompt passes; with the 171,312-token auto-fit pool
(1.44 GiB of KV against 1.10, the belt switched off with
`ARCINT_PREFILL_CHUNK_CAP=off`) free VRAM reaches 0 MiB, swings 60 → 562
→ 56 MiB as the plugin re-sizes its ≈500 MiB buffer, the driver logs the
`-12`, and the runtime reports `CL_OUT_OF_RESOURCES`. So the fault is a
budget the engine did not charge: the mixed-stage attention buffers grow
with the past (they do not exist at past 0, where the activation probe
runs) to about 0.9 GiB at 119k tokens and chunk 128, and the reservation
had left no room for them. The fix is a reservation term — per token of
context, 1.5 × chunk × query heads × head size × 4 B ÷ 256 (the buffer
plus margin above the resize peak the sampler saw), i.e. 12 KiB charged
at chunk 128 against the 8.8 KiB the u8:i4 KV itself costs, 3 KiB at
chunk 32 — charged only under 4-bit values; the belt stays because it is what keeps the term small. Charged
honestly, the coder's u8:i4 auto-fit on the 16 GiB card lands at 101,984
tokens with the belt at 64 (599 MiB of scratch charged, 6.0 KiB per
token against 8.8 of KV), below the same card's u8 auto-fit of 133,456:
verified there with a 97,727-token prompt that prefills at 220 t/s with a
free-VRAM floor of 818 MiB where the uncharged pool reached zero (with
the acceptance ceiling narrowed by the same term the load trims once
more, to 101,824, and the floor is 814 MiB). The
4-bit values save 2.5 KiB per token and the prefill path spends 4
to 6 of it again, so on this card u8:i4 is a decode-side saving only
until the plugin stops sizing that buffer by depth. The candidates,
weighed on 2026-09-03: a bounded partition count (a fixed split of the
past instead of one partition per 256 tokens makes the buffer depth-
independent, no extra launches, the reduction runs over the fixed
slices — the patch to write); f16 partial outputs with f32 softmax
statistics (halves it, precision to be checked against the ladder);
pre-sizing once (removes the churn, not the size); an unpack shim onto
micro-SDPA (the 16-entry dequant table is free, the u8 staging copy of
the whole past per chunk is not); chunk 16 under 4-bit values (an
engine lever that buys a sliver, cost unmeasured); native 4-bit values
in micro-SDPA (upstream). Why
chunk 512 passed at 35k is now plain: 138 partitions of a 512-token
buffer at a shallow past against a pool with headroom is not the same
budget as 466 partitions at 119k; the proxy compared buffers, not
headroom. Two prior readings on this record — "the tmp_out threshold" in
§7.0.2aa and "not a threshold at all" above — were both narrated from
the buffer alone; the measurement is the free-VRAM floor.

Prior art found on the way (2026-09-03): an OpenVINO report of VRAM
growth to `CL_OUT_OF_RESOURCES` under xe that is stable under i915 on
the same kernel (issue 32665, 2025-11), a compute-runtime USM pool
regression that fails with free VRAM (issue 916, 2026-04; the host runs
26.27, after the fix window), and Intel-community reports of xe engine
resets under sustained LLM inference on the B-series that were closed
without a cause. Upstream xe is adding structured reporting of exactly
these rebind faults (a v6 series on the intel-xe list dated 2026-09-03);
on the host's kernel the bare warning is all there is.

**The belt.** The engine now caps the prefill chunk under 4-bit values
from the served pool depth, with the proxy above as the yardstick and a
512 MiB budget chosen so that the measured passing chunk is what the
budget yields: 64 at n_ctx 131,072, 32 at the 171,312-token auto-fit,
128 unchanged up to 65,536; it halves and rounds up to the KV block
granule, never raises, caps an explicit `--prefill-chunk` the same way
and logs the geometry it used. It runs after the pool depth is final, so
an explicit shallow `--n-ctx` is not capped for the auto-fit depth, and
the activation reservation stays the one probed at the pre-cap chunk
(over-charged, never unsafe). The first version fed the 2 KV heads and
would never have fired on the coder; the review caught it and the
geometry helper is now tested with the coder's own configuration. Its
price on short prompts, same card and pool, an 8,417-token prompt: 495 t/s
prefill at chunk 128, 437 at 64 and 440 at 32 — about 12%, and 32 costs
nothing over 64. Decode is unchanged. Verified on the card: at the
171,312-token auto-fit the belt lowers the chunk from 128 to 32 and the
119,074-token prompt that faulted at chunk 128 prefills in 705 s
(169 t/s) and decodes its 400 tokens.

**The MTP step at depth, timed.** The 27B agent on the 24 GB card at u8,
a 76,403-token prompt, 64 greedy tokens, one process per arm: with the MTP
head the 64 tokens take 58.7 s (1.1 t/s), of which the verify forward is
48.8 s — 763 ms per step against 97 ms for the plain step (6.3 s for the
same 64 tokens, 10.1 t/s) — and a further 130 ms per step is inside
"other" and unattributed. The verify timer wraps the main graph's forward
of the verify batch, not the MTP layer; the MTP layer and head run without
profiling and with a dense mask over the committed prefix. The per-node
profile at that depth does not measure this: with `ARCINT_PROFILE_PAST`
at 76,000 the node totals come to 3.4–3.7 s per capture for a step whose
wall is under 0.1 s, the engine's own caveat line fires, and the table is
flat across one, two and four tokens (paged attention 2.23, 2.37 and
2.41 s in counter terms). A wall clock of the forward per token count at
depth is the measurement that can see it. `ARCINT_PROFILE_MWALL` times
the paged forward per token count at a depth, ten repetitions each, the
past walked in served chunks (the hook's first run timed depth 1 by
reading the wrong variable, and its second forwarded the whole past in
one chunk and was refused by the card; both are in the record). The
dense 27B agent on the 24 GB card, u8, pseudo-random tokens, wall clock
per step:

| tokens per forward | at depth 1 | at 76,000 |
|---|---|---|
| 1 | 79.8 ms | 103.6 ms |
| 2 | 88.2 ms | 211.1 ms |
| 4 | 85.8 ms | 208.4 ms |
| 8 | 87.4 ms | 208.8 ms |

At depth a two-token forward costs twice a one-token forward, and four
or eight tokens cost the same as two: the plugin runs a different
attention stage once the query has more than one token, and at 76k that
stage costs about 208 ms whatever the token count, against 104 ms for
the single-token stage. Near depth 0 the same switch costs 10%. That is
the drafter tax at depth, measured: every verify forward pays two plain
steps, so a drafter breaks even only when it lands one accepted token
per cycle on average, and at 0% acceptance it runs at half plain — which
is where DFlash sits at 76k (5.3 t/s against a 209 ms cycle's 4.8, with
plain at 16.3 on the served path). The MTP arm is worse than that
accounts for. Its served verify forward of two tokens took 763 ms per
cycle in one run (64 tokens) and 451 ms in another (128 tokens), against
211 ms for the isolated two-token forward at the same depth; the
run-to-run difference is unexplained. What the served forward spends it
on was split with a measurement switch (`ARCINT_FORWARD_SPLIT`): the
graph itself runs 430 ms, the logits readback 0.2 ms, the hidden-state
readback nothing, and the MTP layer and head between forwards about
20 ms per cycle. The sweep was then run with the served path's inputs
one at a time — the MTP graphs loaded, the per-token checkpoint rows and
interval — and stayed at 105 / 210 / 208 ms for one, two and four
tokens each time, so neither explains the doubled graph time. The same
forward repeated inside the served process with identical inputs runs
340 ms (`ARCINT_FORWARD_SPLIT=2`), and running the served process alone
on the host changes nothing (445.2 ms with and without a window on the
other card), so about 100 ms of the served cost is state the first run
carries and the rest separates the served forward from the sweep's in a
way not yet identified; the driver's counters show the 24 GB card
taking about 17,000 recoverable GPU page faults per second during the
prefill and none during decode, so those are not it either. Open.
For the decision in §8 the served number is the one that counts: at 76k
a verify cycle costs four plain steps on this path, not two. What none
of this says is why either drafter accepts nothing at 76k.

RETRACTED (§7.0.1): the 445 ms vs 209 ms comparison above compared two
different compiled graphs (the served MTP arm's own graph against a
separate profiling harness), one of them under a PERF_COUNT-inflated
wall clock — not like-for-like, and the "open" gap it left is not a
question about this depth's own cost. See §7.0.2ag for the corrected,
clean-instrumentation measurement (the plain arm decodes at 15.3 t/s
at both 8.9k and 76k) and the mechanism this section could not yet
name.

#### 7.0.2ac Patch 0015: the attention scratch bounded, and the crash it uncovered (2026-09-03)

The operator chose the plugin-side fix over the engine's honest but
costly reservation: bound the number of partial results the mixed-stage
paged attention keeps per query token, and store them at their real
width. Reading the plugin for the design found the second half already
half true: the kernels declare that intermediate output as the model's
f16 output type, and the host had been sizing it with a hard-coded four
bytes per element — allocating twice what the kernels address. Patch
0015 therefore has three parts: the host sizes that buffer by the output
element size (byte-exact by construction); a read-write plugin key,
`PAGED_ATTENTION_MAX_PARTITIONS` (default 0, today's behaviour), that the
engine sets from `--paged-attention-max-partitions`; and, under the
bound, each work-group folds S consecutive 256-token partitions with an
online softmax merge into one partial, so the buffer is chunk × query
heads × head size × 2 B × the bound, whatever the depth. The merge folds
its rescale into the existing normalisation multiply in f32 with one
f16 store per fold — at fifteen folds about 0.7% worst-case and 0.1%
typical on a convex combination of dequantised values — so byte
exactness across the bounded arm is not claimed at depth; at the
unbounded setting a `p == 0` branch keeps the original divide and the
plugin is bit-identical to the unpatched one.

**What the card says.** With the patched plugin and the bound at 32 on
the 16 GiB card, the coder's u8:i4 auto-fit lands at 165,680 tokens at
chunk 128 with 48.5 MiB of scratch charged (against 101,824 at chunk 64
unbounded), and the belt stays silent. The plugin's 220 paged-attention
unit tests pass, including new cases with two subgroups per work-group
(the compressed-KV configuration this patch exists for), the dual-nibble
value kernel, the single-token path, a bound of one, and 40k- and
120k-token pasts. On the 24 GB card, same chunk 32, bound 0 against 32:
the 64-token greedy output and the acceptance answer are byte-identical
and the task scores 10/10 on both. At the unbounded setting the patched
plugin is byte-identical to the unpatched one on both cards.

**The crash.** The first served run at the bound died at 56k tokens and
beyond with a bus error: a jump to an unmapped address inside the
plugin's infer, on both cards, with nothing in the kernel log, while a
50k prompt passed and the plugin's own tests at 120k passed. Under gdb
with glibc's heap checking the same 119k run completed; with only the
free-fill perturbation it crashed again — a use-after-free whose victim
moves with the heap layout. The mechanism, read from the plugin and
stated as such: `primitive_inst::realloc_intermediates` replaces an
intermediate buffer's identity without raising the flags that rebind the
kernel arguments (those track outputs only), so after a genuine
reallocation the kernel keeps the freed handle. Nothing had exposed it
before because the buffer only ever grew; under the bound it plateaus
and takes the reuse path on every chunk of a long prefill. The patch
carries the local fix — the paged-attention implementation records the
identity of every intermediate its arguments were bound to and forces
the rebind when any changes — and the framework-wide fix is a backlog
item of its own (§8). With it the 56k cell that had crashed twice passed three times, but the
119k cell crashed again, two variants that kept the intermediate buffers
alive (a private never-freed set; a two-deep ring retired after a stream
finish) crashed the same way, and then the decisive cell: the UNPATCHED
plugin, at the unbounded setting, with an explicit 165,680-token pool
and the 119k prompt on the 24 GB card, crashed too. The deep-prompt
crash is therefore not this patch's: it needs a deep pool together with
a deep prompt, and the coder had never served such a pool at u8:i4
before because the VRAM fault (§7.0.2ab) came first. On the 24 GB card
every one of these crashes coincides with the kernel log's "Engine
memory CAT error, class=ccs" and an engine reset — a GPU-side
out-of-bounds access — while the 16 GiB card, which has no fault
reporting, logs nothing and the host process simply dies. The same
119k prompt passes with pools of 101,824 and 131,072 tokens at u8 and
u8:i4, and the dense agent serves 262,144-token pools with 143k prompts
at u8. The unpatched plugin at plain u8 with a 262,144-token pool and the
same 119k prompt passes on the 24 GB card, and the engine's per-forward
inputs — past lengths, subsequence bounds, ascending block ids, the
context length — are identical at any pool size, so the defect sits in
the asymmetric packed-value path this series added in patches 0008 to
0010, reached only when the pool is deep and the prompt reaches into it;
the patched plugin at the bound serves the same 119k prompt from a
131,072-token pool on the 16 GiB card, so the bounded path itself
holds; on the 24 GB card, whose GPU reports faults, even the unpatched
plugin crashes at a 131,072-token pool with that prompt, so the
consequence of the bad access depends on the card and on what lies
beyond the buffer, and the plugin's own paged-attention harness, which
runs one primitive once, passes at 170,000 tokens of past in every
configuration tried — the defect needs the served path's structure. On the 24 GB card with the
unpatched plugin, the same 131,072-token pool and chunk 128, a
97,727-token prompt (382 partitions of 256) prefills and a prompt of
about 100,000 tokens (391 partitions) crashes with an engine reset, as
does 119k: a tight bracket puts the threshold at 384 partitions (383 passes, 384
crashes), one short of where the finalization kernel would leave its
register path, so that switch is not it; a larger intermediate buffer at chunk 256 (203 partitions, 812 MiB)
passes, so the buffer's size is not it either — the count of 256-token
partitions is, at 384 exactly, one inside the register arm's capacity;
and then the patched plugin at its unbounded setting passed that very
cell, and the 119k one, where the unpatched one crashes. What the patch changes at
that setting is the f16 sizing of the intermediate output and the
argument rebind after a reallocation; the rebind explains every
observation at once — the unpatched kernels keep running on a handle
the framework has freed after each growth reallocation, harmless while
that memory stays mapped and fatal where it does not, invisible to a
single-execute harness, absent at u8 which allocates no such buffer,
and the fix that turned the 56k cell green — and a second reading has the plugin sizing those buffers from the
previous forward's partition count, so that the first chunk into a new
partition writes one partition past the buffer; a build carrying only
the fresh sizing passes 384 and 466 partitions on that card too. Two builds that only alter the finalization kernel — one forcing its
global-memory arm, one doubling its register capacity — pass the cell
as well, and so does a build carrying only the rebind, and so,
eventually, does an untouched control run back to back with a crashing
one. Every single-change perturbation tried clears the cell at least
once, host-side or kernel-side — a pattern that does not name a kernel
defect, and does not name the rebind as the writer either. What it
names is the GPU firmware's own catastrophic-error notification
tripping over something entirely outside these kernels: closed as a
diagnosis in §7.0.2ad, which supersedes this section's own read of the
crash (the rebind hypothesis, the finalization-kernel builds, and every
number this section quoted about which build passes how often).

Two operating notes survive that closure unchanged: the option inserted
into the plugin's configuration shifts the model-cache blob's
positional schema, so the GPU model cache must be cleared when
upgrading to a plugin level that carries 0015; and the engine prices
the bounded scratch at 2 bytes and the bound only when the plugin
accepts the key — a contract, since both ship in one patch.

#### 7.0.2ad The deep-prompt crash closed: a runtime semaphore, not a plugin kernel (2026-09-04)

RETRACTED (DESIGN §7.0.1, kept on the record rather than silently
corrected): every earlier note here and in CHANGELOG.md framed this
crash as depth-triggered — "past 98k tokens", "384 partitions" —
including §7.0.2ac's own tally. That framing was wrong. The prompt
that triggers it is one fixed 98,147-token prompt (331,500 characters;
an earlier note's "119k" for the same text was itself a miscount of a
separate, 400,000-character prompt). The crash depth within that one
prompt's prefill is random — traced at about 3,600, 12,000 and 40,000
tokens into the same 98k run on three separately crashing processes —
so no partition count or chunk size this record ever bracketed is a
threshold. The trigger is VRAM headroom together with concurrent host
or other-card load, not depth.

**The mechanism, in three legs.**
1. Every crash on the 24 GB card is a GPU page-fault storm at one
   fixed GPU virtual address, the same address across processes and
   builds: `0x0000d556aa670000` on the compute engine (483 fault lines
   over 9 crash events) and `0x0000d556aa740000` on the copy engine (8
   lines) — FaultType 0 (not present), AccessType 0 (read), FaultLevel
   4, "Fault response: Unsuccessful", hundreds of "Engine memory CAT
   error" lines and engine resets at the same instant as the faults,
   never a separate event; no "Timedout job" line accompanies any
   arcint crash.
2. The OpenCL runtime's own allocation log identifies that address: it
   is the exact start of a `SEMAPHORE_BUFFER` in local memory — a 64
   KiB semaphore buffer carved from a 17 × 64 KiB command-buffer
   region the runtime allocates at initialisation next to its 320 KiB
   ring-buffer allocations for the compute and copy engines — the
   runtime's own direct-submission ring and semaphore. No model
   buffer, KV page, or plugin intermediate sits at that address.
3. Read from the runtime source and the kernel driver's own
   documentation, not independently measured the way legs 1 and 2
   are: the runtime binds those buffers once, as ordinary user BOs,
   and never re-validates them; the kernel driver's own
   documentation states that user BOs are evictable. Under VRAM
   pressure with concurrent host or other-card activity, the
   driver's eviction traffic can reclaim the semaphore buffer;
   direct submission never issues the exec that would rebind it;
   the next semaphore wait reads a non-present page; the GuC
   reports the CAT error and the engine resets. A second candidate
   mechanism, not yet separated from this one, is read from the
   kernel driver's own tracker — see Upstream, below.

This is a diagnosed driver/runtime interaction, not a plugin defect,
and the plugin's own paged-attention kernels are exonerated on their
own evidence, not merely by elimination: a trace build logging every
intermediate buffer bound at every dispatch against every reallocation
recorded 0 stale bindings over 5,867 dispatches; the crash depth varies
within one fixed prompt, which a kernel-side threshold cannot explain;
and the untouched, unpatched plugin passes the identical configuration
on a quiet host (other card idle, no concurrent host load): 2 passes in
2 (845 s and 802 s), the pair's direct-submission-off partners also
passing (820 s and 840 s).

**Two knobs, measured.**

| knob | config | result |
|---|---|---|
| headroom — the fit's own scratch term active, `ARCINT_PREFILL_CHUNK_CAP` at its DEFAULT (not the diagnostic bypass) | 24 GB card, u8:i4, bound 32, `--n-ctx 131072` (belt settles at chunk 128, 48.5 MiB of scratch charged) | 2 pass in 2 (830.7 s) |
| direct submission off, `NEOReadDebugKeys=1 EnableDirectSubmission=0` | 24 GB card, u8:i4, the `ARCINT_PREFILL_CHUNK_CAP=off` bypass, the 98,147-token prompt | patched (0015) build: 3 pass in 3; untouched build: 2 pass in 2 (820 s, 840 s) |
| direct submission off, same switch | 16 GiB card, u8:i4, bound 32, 165,680-token pool, the same prompt | 5 pass in 5 (638 s, +1.4% wall against a 630 s control) |

At bound 32 with the 165,680-token pool the 16 GiB card's own earlier
tally, restored here: three crashes and four passes the same evening,
while the host was also building plugin variants (host load not recorded); the 630 s quiet-host
control above (5 pass in 5) is the later, separating result.

*KV format.* The fault was observed only with `--paged-kv u8:i4`; the
mechanism above is format-agnostic. A symmetric-u8 pool driven to the
same edge was not exercised and stays untested by decision.

State plainly what the first row means: every crash on the 24 GB card
on this record was produced with `ARCINT_PREFILL_CHUNK_CAP=off` — the
belt, the measured cap and the scratch-term reservation all disabled,
a diagnostic bypass this repository built specifically to reproduce
the plugin's own fault line (§7.0.2ab). arcint's default is the belt
on; with the reservation active at the bound (the first row above),
the identical 98k-token cell passed both times it was run. `ARCINT_PREFILL_CHUNK_CAP`
is the only default in that cell: the bound was set explicitly
(`--paged-attention-max-partitions 32`; the config default is 0), and
bound 0 with the term active is unmeasured here.

State plainly what the second and third rows do not yet prove: the
untouched-build control at this knob also passes on a quiet host, the
same condition that already makes the untouched plugin pass without
the knob at all (the exoneration above). `EnableDirectSubmission=0`
disables the exact mechanism leg 3 names, so its evidence is
mechanism-derived, not a rate result — the samples that exist (3/3 and
2/2 on the 24 GB card, 5/5 on the 16 GiB card) are consistent with it
working and are not, on their own, a discriminator against a quiet
host that already passes without it. Documented here as the fallback
for an operator who ever needs to run past the bound, not as the
primary defence — the primary defence is the default reservation in
the first row.

**Upstream.** The driver's own tracker carries the same signature:
`drm/xe` issue 8390 (open) reports "Timedout job" and engine resets
under sustained decode on four LLM stacks on this card's
generation, with one quoted page fault at `0x0000d556aa3b1000` on the
copy engine — the same runtime heap region this record's own fault
addresses sit in, "recurring across separate incidents and separate
process PIDs" in the reporters' own words. Two candidate mechanisms,
not yet separated (the fact-finding record's own wording, kept as
such): leg 3 above (the semaphore BO evicted, never re-validated), and
a second one read from the same tracker — a `drm/tip` commit, "drm/xe:
Order ring writes before ring tail updates" (Cc stable), fixing a
missing write-ordering barrier between a ring buffer and its tail
pointer for kernel-submitted jobs. Its own commit message states the
symptom as "command stream corruption, which typically manifests as a
hang or a spurious pagefault" — a peer candidate for the SAME fault
this record shows, not a distinct kernel-submitted-path story. That
commit closed two related upstream reports; one of them, an
eviction-test regression, is recorded as this card's generation, the
other report's card is not recorded. The fix is in the `linux-7.1.y`
stable branch and is not in `linux-7.0.y`; the dev host runs a 7.0.14
kernel with no 7.1 package offered by its distribution. Whether to move
the host to a kernel carrying that fix is an operator decision, not
made here. Cross-reported upstream: a comment on intel/compute-runtime
issue 948 (the same fault class on this card's generation; our comment
adds the allocation-log identification of the faulting buffer, the two
knobs, and the fact that the fault occurs on GuC 70.65.0), and `drm/xe`
tracker item 9141, linking 8390, 8651 and 7810.

**Non-determinism, a separate finding.** Repeated runs at a shallow
depth (16 GiB card, u8:i4, an 8,418-token prompt, chunk 128, unbounded
partitions) are not byte-identical run to run: untouched 13 of 14 runs
byte-equal to each other (one outlier), patch 0015 12 of 13 (one
outlier), 0015+0016 6 of 6 on the clean build of the shipped patch
stack (an earlier staging build that also carried an unrelated
experimental change was 5 of 6), and the bounded-32 arm 3 of 3
byte-equal to each other and reproducibly DIFFERENT from the unbounded
arms (the expected fold from the bound's own online-softmax merge,
above). Each outlier is consistent with a single-token divergence; the
rate — about one run in fourteen — is the same across every build
tried, so it is a property of the stack as already shipped, not
something 0015 or 0016 introduced.

**Patch status.** 0015 is committed; this round corrected its own prose
(buffer roles and indices; no code change; the index line below is
re-derived, not asserted). 0016 (the fresh intermediate sizing) was
reviewed: two real hunk findings were fixed (the sliding-window term
now mirrors the served formula; the bound is applied under the same
stale-count stage the served path uses), two red-first unit tests were
added, and the patch carries the deep-pool and growth regressions as
well. Gate on a clean build of the shipped stack (baseline + 0001-0016,
nothing else): all 19 tests in the combined filter pass on the 24 GB
card, and the 8,418-token ladder is 6 of 6 byte-equal (above). An
earlier staging build that also carried an unrelated experimental
change was 18 of 19, the one failure being the sliding-window test's
own fixture (an unsized cache), since repaired. 0016 remains a
correct hardening of a real stale-count defect in its own right — it
does not touch the crash this section closes. Buffer-index correction
for the record: on the served path (the MIXED stage, no scores
output), there are seven intermediates — `exp_sums` = 3, `max_logits` =
4, `tmp_out` = 5, 6 = the global-work-size-to-subsequence mapping; a
graph with a scores output shifts every one of these indices by two.

#### 7.0.2ae M9 equivalence gates on the offloaded path; the tier's residency-dependent arithmetic (2026-09-04)

The M9 equivalence gates this milestone's own row has owed since
§7.0.2y ran on the 24 GB card, `marfrit-openvino +p3`, source arcint at
HEAD. Two configurations: the coder at `--offload-ratio 20 --paged-kv
u8:i4` (8 GiB device pool) passes every check in `tests/equivalence/
run.sh` and every check in `tests/concurrency/run.py` (the drafter
never fires at this cell, and there is no MTP head to engage). The
35B (`qwen36-35b-a3b-int4`, u8 KV) at `--offload-ratio 50
--moe-cpu-tier` (8 GiB pool) passes concurrency (draft acceptance
~33%) and passes every equivalence check but one: two greedy runs are
identical, a warm-cache run is identical to cold, drafts are
identical — and the continuation check FAILS, deterministically, 3 of
3, with the same fork every time: a continuation restored from the
prefix cache diverges from the cold run at character 141 (390 against
388 bytes total), "compressed, rank-deficient summary" against
"compressed, global summary".

**Ruling out a split explanation.** The gate's kept work files show
PROMPT 235 tokens, CONT 245, cache hit 192 tokens — 192 = 3×64, the
prefix-cache snapshot grid equals the prefill chunk grid at this
cell's default (`--cache-grid 0`) — so both arms compute the identical
53-token tail chunk `[192, 245)` at decode entry; the restored arm's
warm boundaries are a suffix of the cold arm's own chunk boundaries,
not a different split. `batched_gemv_threshold` is 32, and both the
53-token tail chunk and the cold run's own 64-token chunks are above
it, so every prefill token in both arms takes the grouped, device-only
path — no host expert runs during prefill in either arm. Only the 64
decode tokens are tier-eligible at all: at `token_num` 1, each decode
step routes through the batched GEMV path, and per MoE layer (×40)
each of the 8 routed experts is resident-and-filled (device fused
GEMV) or pool-full (a host expert kernel) independently.

**Diagnosis** (the investigating design note's own reading, marked as
such and confirmed by E1/E2 below). The expert slot LRU pool is
process-global — it is not part of the prefix-cache entry and is never
reset per request. With the tier off, residency only ever decides
*where* an expert's bytes sit; with the tier on, residency **selects
the arithmetic**: a resident expert runs the device kernel (f16,
partial-accumulate GEMV), an evicted one runs the host kernel (f32),
and the two are documented as not bit-equal (patch 0011 §5). The cold
and restored arms enter decode with differently-shaped request
histories behind them (245 prefill tokens from an empty pool, against
235 prefill + 64 decode + a cache hit + 43 prefill + 64 decode + the
same 53-token tail) and so hold different resident sets; a few hundred
rows per decode token rounding differently is a sub-margin logit
perturbation that first flips an argmax at a near-tie — exactly a
fluent early agreement followed by a one-token fork, which is what
both logs show. Two mechanisms the note ruled out on measurement, not
assumption: the host kernel's own accumulation is per-job, independent
f32, with no cross-token or cross-expert term, and the reduction that
combines host and device output writes each row at a `flat_id` fixed
by `token × EXPERTS_PER_TOKEN + expert_slot` regardless of which side
produced it — **the reduction order is already residency- and
batch-shape-independent, so an order-only fix is a null.** A sentinel
defect was also ruled out (it would leave gross garbage from decode
token 1, not two fluent texts diverging at token ~27) and VRAM
pressure (moving slot buffers between VRAM and GTT is timing, not
bytes).

§3.4:469 is unconditional: byte-identical greedy output "for any prompt
and any cache state." This measurement shows output depending on a
variable that sentence does not admit — the process's own request
history, through LRU residency — which **is a violation of §3.4 as
written, not a property to file beside it.** The nearest weaker
candidate wording, "byte-exact only within one batch shape," does not
fit either: batch shape is identical in both arms (the same 53-token
tail chunk, `token_num` 1 at decode); the accurate qualifier is *within
one process history*.

**E1 measured:** the identical suite with the tier off
(`--offload-ratio 50` alone) passes every check, the continuation
included — the tier is the discriminator. **E2 measured:** with the
tier on and the prefix cache off, one process asked PROMPT, PROMPT,
CONT gives the same answer to PROMPT twice and a THIRD distinct
continuation for CONT — equal neither to the cold nor to the restored
text — so the tier is history-sensitive in general; the cache and the
restored state are exonerated, and the reading above stands as
measured.

**Ship decision.** F0 ships today, arcint-side, red-first (two cases in
`tests/test_config.cpp`): `--moe-cpu-tier` refuses `--prefix-cache-mib
> 0` at startup, in the same fail-loud ladder as the M8 asymmetric-KV
refusal (§7.0.2w) — a path that cannot meet the invariant is configured
out rather than papered over. The fix of record is **F1**: promote
patch 0011's `moe_cpu_expert_ref_gpu_emul` — already built as the
divergence *oracle* that predicts this exact fork — from oracle to the
*served* host kernel, F16C-vectorised, as plugin patch 0018 (0011's own
kernel files, not patch 0017's readback path, so a fresh patch number
is correct); the partition then becomes numerically invisible whatever
the history. Its cost is not yet measured — the deferred horizontal sum
changes shape — and the P\* decode cell is re-measured once it lands,
win or lose. Two alternatives were weighed and are not the plan of
record: a structural fix that fixes residency as a pure function of
expert id (cheaper, but gives up the measured 47.9% LRU hit rate at
P\*, so it owes its own decode number first) and snapshotting residency
into the cache entry (rejected — it couples an arcint cache entry to a
plugin-side pool and still leaves two differently-warmed *cold*
processes disagreeing with each other).

Recorded for the axis this closes: with `--moe-cpu-tier`, greedy output
is byte-exact for a repeated prompt *in the same process* (measured:
two greedy runs identical, a warm-cache run identical to cold at 235
tokens, drafts identical) and is **not** byte-exact against a
differently-warmed process. The OFF-vs-ON byte identity §7.0.2x
measured was one process history, not a claim about any history. F0
removes only the cross-restore axis; §3.4 stays violated with the
tier on until F1 (patch 0018).

**Update (2026-09-04): F1 retired on review; patch 0018 ships as F2
instead.** A code review found F1 as described above — promoting patch
0011's `moe_cpu_expert_ref_gpu_emul` oracle to a served, bit-equal host
kernel — not attainable at reasonable cost: the device GPU kernel
applies scale and zero-point per lane inside its own reduction loop,
and the cross-lane and cross-group reduction that follows runs in an
order the plugin does not document as fixed — a tree, not a serial
left-to-right accumulation — so a host reference cannot be made to
match it bit-for-bit without reproducing that same undocumented tree.
This is the review's finding, stated here as such and **not** an
independently re-measured fact in this repository; nothing in this
section's own E1/E2 measurements bears on it, and no build attempting
the bit-equal kernel has been benchmarked or rejected on cost grounds
in this tree. It supersedes only the "fix of record" framing above,
not the diagnosis (the LRU-residency mechanism and the E1/E2 exoneration
of the cache and the restore path stand).

Patch 0018 ships as **F2** instead: not a bit-equal kernel, but a
*static residency partition* — each expert's host-or-device placement
becomes a pure function of expert id, layer, and pool configuration,
fixed for the life of the process, so it never depends on request
history and the arithmetic split it feeds cannot vary within one
configuration. The detection contract follows the same shape as patch
0015's `PAGED_ATTENTION_MAX_PARTITIONS` key (§7.0.2ac): a read-only
GPU-plugin property, `MOE_CPU_TIER_STATIC_PARTITION`, that the plugin
reports `true` if and only if the served tier's host/device split is
that static partition, and reports `false` or omits the property
otherwise. arcint queries this property at load and lifts the
`--prefix-cache-mib` refusal for `--moe-cpu-tier` only when it reads
`true`; the property's absence, or a `false` reading, keeps the
refusal — fail closed, the same ladder F0 shipped in.

**Update (2026-09-04): written, and the allow branch is now verified —
after fixing three more load-time bugs the property alone did not
predict.** Patch 0018 exists (`patches/0018-moe-cpu-tier-static-partition.patch`,
13 files); the equivalence suite's own reach for it, plus a review pass
after the first passing run, exposed three further defects before the
mechanism could be trusted, none a numerics problem, all load-time
sequencing or reporting bugs distinct from the LRU-residency diagnosis
this section opened with:

- **The plateau probe crashes under a 100%-pinned pool.** arcint's
  device-pool sizing probe (used when no forced size is given) saturates
  the pool by observing eviction; the static partition pins every slot at
  bind time and evicts nothing, so the probe's first chunk always hit
  `LRUCache::evict_one`'s "no evictable (unpinned) slot" throw and the
  load died before this section's own detection logic ever ran. First
  fixed (2026-09-04) by skipping the probe when
  `MOE_CPU_TIER_STATIC_PARTITION` reads `true` and charging the pinned
  pool's analytic ceiling as the device figure — the same IR-walk
  (`slot_pool_from_ir`) or config-derived per-expert-bytes formula the
  host (GTT) estimate computes. The 0.3.0 release gate (§7.0.2ai) showed
  that ceiling is the wrong *admission* charge on the 16 GiB card: the
  driver keeps most of the pinned pool host-mapped there (§7.0.2t's
  two-ledger finding applies to the pinned pool exactly as to the LRU
  one), the probe had read 0.11 GiB of VRAM for the same 7.50 GiB pool,
  and charging 7.50 refused a configuration the card serves. So the
  probe now runs under the static partition as well — with the prefill
  fallback below fixed nothing evicts, and it saturates as the pinned
  set fills on first use — the ceiling is only the fallback when the
  probe throws, and it is logged beside the probe's reading as the
  cross-check this paragraph used to say nothing provided
  (`backend_ov.cpp`, `source: probe-static` for the measurement,
  `static` for the fallback). Nothing here calls the plugin's own
  `resident_slot_count()` — that count lives inside patch 0018's C++ and
  stays there; arcint's ceiling is an independent formula over the same
  model shape, and the probe is the measurement it is checked against.
- **The prefill fallback path has the identical crash, one call site
  over.** `OffloadExpertWeightProvider::acquire_one` — the per-expert path
  `exec_prefill_onednn` falls back to — had no static-partition awareness
  at all and unconditionally called the ordinary LRU insert-or-evict path,
  hitting the same all-pinned throw the plateau probe did. Fixed in the
  plugin: `acquire_one` (the abstract interface and both concrete
  providers) now returns `std::optional<size_t>`; a non-resident expert
  reaching this path returns `nullopt` and the caller falls back to
  `moe_cpu_expert()` inline per token instead of assuming a device slot is
  always obtainable.
- **The refusal gate itself queried the wrong object.** The paragraph
  above, as first written, read `MOE_CPU_TIER_STATIC_PARTITION` via
  `core_.get_property(device, ...)` — a device-level call made before any
  model is compiled. Patch 0018 registers that property only on
  `CompiledModel` (an `OV_CONFIG_RELEASE_OPTION`, not a Plugin-level
  entry — unlike `PAGED_ATTENTION_MAX_PARTITIONS`, which genuinely is
  dual-registered at both levels, the analogy this section drew on
  originally). The device-level call always throws for this key, folds to
  `false`, and the refusal fired unconditionally regardless of whether
  0018 was present — measured directly: the equivalence suite's own
  prefix-cache server refused to load with "the plugin does not report a
  static residency partition" in the same process class that, moments
  earlier without `--prefix-cache-mib`, correctly logged the property as
  `true` from the plateau-probe-skip's own (already-correct) query of
  `paged_model_` after compilation. Fixed by moving the refusal to that
  same post-compile call site, reusing the value it already computes
  correctly, rather than adding a second query.
- **The property itself never checked whether the tier was even on.**
  `MOE_CPU_TIER_STATIC_PARTITION`'s own default (`options.inl`) is a
  context-free lambda, evaluated once per `ExecutionConfig` construction
  with no way to see whether this exact compile turns `MOE_CPU_TIER` on,
  so it reported `true` (the F2 rule "in effect") for every load, tier on
  or off, contradicting its own doc comment and every "if and only if"
  reading of it in this document. Found in a review pass after the first
  passing gate run, before commit. Fixed in the plugin's
  `ExecutionConfig::finalize_impl` (`execution_config.cpp`), where
  `get_moe_cpu_tier()` already answers correctly for this compile:
  `if (!get_moe_cpu_tier()) { m_moe_cpu_tier_static_partition = false; }`.
  Red-first: a new device-free gtest confirmed red against the unfixed
  tree (`Expected: false` / `Actual: true`) and green after — see the
  patch header for the full write-up, including two things this same
  review pass looked at and found were NOT bugs (a routing-weight layout
  question in the prefill fallback, and a blocking-copy synchronization
  question), plus one it found but left unfixed as out of reach in any
  configuration arcint actually drives today.

With all four fixed, `tests/equivalence/run.sh` was run for real against
the 35B cpu-tier configuration with `--moe-cpu-tier --prefix-cache-mib
4096 --kv-block-size 32` on the 24 GB card, m18 plugin, in a clean,
uncontaminated run:

    ok   warm cache output is byte-identical to cold
    ok   the console reports a cache hit (cache hit 192 tok (81.7%))
    ok   a continuation of a cached prompt also hits
    ok   a continuation restored from cache matches a cold run

**Pass.** This is the check this section exists to fix; the allow branch
is verified on real hardware, not asserted. Everything else in that run
passed as well except one check unrelated to this mechanism —
`speculative decoding is deterministic across runs` (an n-gram lookup
drafter, `--draft 4 --draft-ngram 3`, no MoE routing involved) — which
passed in an earlier clean run of the same suite and failed in this one;
recorded as an open, separate item (§7.0.1: a flake between two
otherwise-identical runs, not touched by anything in this section, not
yet re-run enough times to characterise). Re-run at the 0.3.0 release
gate: six fresh processes on the tier configuration, six identical
outputs (§7.0.2ai) — one failure in about nine runs on the record, not
reproduced; the check stays in the suite and the flake stays open as a
rare event, not a mechanism. Production was confirmed restored and
health-checked after every load in this sequence.

**Update (2026-09-05): the oversized routing-weights declaration beside
the bug #2 fix, investigated — harmless, upstream, no change.** Patch
0018's header flagged, and the M14 handoff carried, a pre-existing line in
the resident branch of the same per-expert prefill loop,
`routing_weights_size = n_token * max_topk`, declaring a oneDNN memory
extent `max_topk` times what the gather kernel writes (one weight per
gathered row, `n_token` of them). Read against the plugin and the vendored
oneDNN rather than assumed: the object is the `down` matmul's per-row
binary-multiply post-op input, whose descriptor is fixed at primitive
creation as `{n_token, 1}` (`post_op_bin_mul(false)`,
`grouped_matmul_helper.hpp`) — exactly the `n_token` contiguous elements
gather wrote; `forward()` binds the caller's object verbatim, and oneDNN's
execute path (`cvt_primitive_args`) only classifies and counts arguments,
it never compares a bound memory's descriptor with the primitive's. The
inflated extent is neither read nor inspected, and it cannot overrun the
buffer: the scratch is allocated at `max_topk * token_num` and
`n_token ≤ token_num` (a token routes to a given expert at most once). At
the served shapes (`num_experts_per_tok` 10 for the coder, 8 for the 35B)
the declaration is 10×/8× the consumed count, all inside the allocation —
consistent with every gate on this path passing byte-identical. The line
is verbatim upstream at the pin; the tidy fix (declare `{n_token, 1}` to
mirror the creation-time descriptor) is an upstream cosmetic, not a patch
worth carrying (§1, smallest sufficient divergence). Closed.

#### 7.0.2af M14: the readback decomposed — the 283 µs was queue backlog, not transfer (2026-09-04)

The design note owed by §7.0.2x's own "next instrument, named and not
chased" landed as patch 0017 (counters only — the four new spans
`cpu_topk_id_ns`/`cpu_x_enq_ns`/`cpu_x_wait_ns`/`cpu_x_drain_ns`, a
warm/steady split so warm-up uploads stop biasing the steady-state
average, `usm_host` destinations for the x/routing-weight readback in
place of pageable `std::vector` buffers, and the topk_id read hoisted
into the same non-blocking wait as x and the routing weights, issued
before the provider's `try_acquire_simultaneous` call; an env-gated
`MOE_OTD_READBACK_PROBE` drain probe). First measured: 16 GiB card,
35B int4, ratio 50, 8 GiB device pool, 128 slots, u8 KV, one lane,
n_ctx 65,536, a 1,198-token prompt, 64 greedy tokens, two runs per
process, four processes, the `+p3` base with the staged plugin,
production units running.

RETRACTED (§7.0.1, kept on the record rather than silently corrected):
this window originally read the page cache as COLD, from `fincore`
showing 3.0 of the artifact's 18.3 GiB resident. That reading is
withdrawn — the container's root lives on ZFS, whose own file data
lives in the ARC, not in the page cache `fincore` reports (mmapped
pages only), so a low `fincore` figure does not establish a cold cache
one way or the other. What the window's own counters still carry,
unretracted: `avg_disk_io` read 52 µs over 265k tensor loads, 13.8 s
of disk time per 128 tokens.

RETRACTED, second and larger finding (§7.0.1, kept on the record
rather than silently corrected): every decode-RATE number in this
section — the four in the first window, the six in the clean rerun,
and the 2.6/5.0 t/s pair below — is void as a P\* number. arcint has
read the expert device-pool budget from `ARCINT_MOE_DEVICE_POOL_BYTES`
since the M7 auto-fit commit (parsed strictly and handed to the plugin
as the numeric property `MOE_OTD_DEVICE_POOL_BYTES`; the M8 review
removed the old behaviour of forwarding the raw env string, because
the plugin's own `stoull` silently mis-parsed typos). Every tier
window this section records exported the OLD, no-longer-forwarded
variable, so the plugin ran with its own default device pool for all
of it and the expert slots lived in host memory throughout: arcint's
own load line said so ("expert slots host-side: 7.50 GiB (GTT, source:
config)", "expert slot pool: device figure 0.12 GiB (probe)"), and the
runtime's own allocation log confirmed it (`BUFFER_HOST_MEMORY`
10.0 GiB in 533 allocations against `BUFFER` `LocalMemory` 3.2 GiB;
the kernel's own accounting: 2.05 GB VRAM used, 8.87 GB GTT). §7.0.2x's
own record predates this rename and ran with the real 8 GiB device
pool; this section's own cells are §7.0.2x's own GTT-spill regime
("ratio 20 spills into GTT, 2.1 t/s") under a different model and
ratio, not the reference cell, and every decode rate this section
quotes is retracted as a P\* number for that reason. The in-build
comparisons stand unretracted: the `usm_host` destination cutting
289 µs to 53 µs, the hoist being a null, the 3.5 ms per-layer queue
backlog, the counters' own internal identities, and the byte-identity
across all arms — every arm in this section shared the same wrong
pool, so the comparisons BETWEEN arms are unaffected by which pool
that was. A solo run with the correct variable, both units stopped,
files pre-read, reproduces the record within its own run-to-run
spread (see the reproduction paragraph below) and replaces the
rates above. Lesson for the record, one sentence: every tier
measurement window must export `ARCINT_MOE_DEVICE_POOL_BYTES`, not
the retired raw variable.

**Decode rates, and why the first window's numbers do not replace the
record on their own.** Tier OFF 2.4/2.6 t/s; pre-0017 tier ON 4.6/4.8;
0017 tier ON 5.0/5.0; 0017 with the drain probe armed 4.8/4.9 — all
about three times below the §7.0.2x record (10.4/10.6 OFF, 15.0/15.5
ON), which was taken with the expert files already resident in RAM.
These four numbers were stated plainly as NOT comparable to that
record, and their own relative order among each other confounded by
cache warming across the four processes run back to back that window,
not a clean measurement of the patch.

A clean rerun replaces the comparison, though not the gap to the
record: quiet host, both production units stopped, the same cell, six
processes in one fixed order — pre-0017 tier ON, 0017 tier ON,
pre-0017 tier ON again, 0017 tier ON with `MOE_OTD_READBACK_NOHOIST`,
0017 tier ON with the drain probe, 0017 tier OFF. Decode: tier OFF
2.6/2.6 t/s; pre-0017 ON 4.7/4.7 and 4.7/4.7 (its own two separate
samples agree with each other); 0017 ON 4.8/5.0; NOHOIST 4.9/5.0; the
drain probe 4.7/4.9. 0017 is about +4% over pre-0017; hoisted against
not-hoisted is a null, 5.0 against 5.0. All six arms are byte-identical
to each other over the 64 tokens, tier OFF included — this sample's
own residency history happened not to fork, unlike the M9 continuation
cell (§7.0.2ae); that is not evidence the mechanism there does not
apply here, only that it did not fire on this particular history.

**The counters, which are the actual point of this window.**
Pre-0017: `avg_cpu_x_us` 292 µs (reproducing the 283 µs on record),
`avg_cpu_compute_us` 740, `avg_cpu_join_wait_us` 701, 18,040 host
pairs over 5,076 tier layers (3.55 per layer — this cell's own
absolute values, not P\*'s). With 0017: `avg_cpu_x_us` 3,644 µs =
`avg_cpu_x_enq_us` 183 + `avg_cpu_x_wait_us` 3,460; `cpu_tier_calls`
5,120 (= 40 layers × 128 decode tokens, two 64-token runs per
process); every one of those 5,120 calls landed in the steady-state
bucket, none in warm (`cpu_x_warm_layers` 0). The merged wait does not
shrink the cost — it makes visible a queue backlog that patch 0012's
own blocking `topk_id` read used to absorb uncounted, one round trip
at a time. The decisive discriminator, the drain probe: a
`stream.finish()` timed immediately before `tx0` costs 3,578 µs on its
own, and the read that follows the drain then waits only 107 µs —
essentially all of the "readback" time is the drain itself, not the
4 KiB transfer after it.

A defect found in review of the first 0017 build is on the record
rather than silently fixed: `tx0` was stamped before the drain probe
instead of after it, which double-counts the drain into the probe
arm's own enqueue and topk_id spans; fixed in the rebuilt patch. The
probe arm's own `avg_cpu_x_enq_us` figure from that build, 3,685 µs,
is read as inflated by this defect and not used above — the
independently-timed `cpu_x_drain_ns` (3,578 µs) and the post-drain
wait (107 µs) are unaffected by the stamp's placement and are what the
conclusion below rests on.

**Conclusion, marked as measured, not as arithmetic.** The per-layer
"readback" cost is the host tier waiting for the GPU to reach this
layer's own router — a queue-backlog wait — not the 4 KiB transfer
that follows it. §7.0.2x's own candidate next lever, host-visible
residency for the decode input (design note §2, phase B), cannot
remove an event wait it was never going to touch, and is retired
before being built rather than measured into a loss. The tier's cost
is the serial dependence on the GPU's own layer work; if there is a
further lever it is overlap across that dependence, not memory
placement.

The in-build discriminator this section owed, `MOE_OTD_READBACK_NOHOIST`
(the topk_id hoist disabled, isolating it from the `usm_host`
destination change), confirms this directly rather than by inference:
run in the same six-process clean window above (5,120 tier-on calls
per build = 40 layers × 128 tokens, 5,074 of them with a host miss,
3.54 pairs per layer, hit rate 44.4%). Pre-0017's `avg_cpu_x_us`
reproduces again, 291 µs then 289 µs across its two samples. With the
hoist off, the blocking `topk_id` read is now counted on its own —
3,620 µs, the same GPU queue backlog the pre-0017 code paid but never
counted — and the x/routing-weight readback that follows it, now
landing in `usm_host` memory instead of a pageable `std::vector`,
costs only 53 µs (enq 13 + wait 40). Hoisted (0017 as shipped), the
merged wait is 3,651 µs (enq 184 + wait 3,467); the drain probe shows
drain 3,511 µs then the reads themselves 214 µs (enq 108 + wait 106;
the dump's own rounded avg reads 215). So the old 283 µs "readback"
was the x read landing in pageable memory *after* the topk read had
already drained the queue; the `usm_host` destination change alone
cuts that to 53 µs, which is the whole of
0017's own +4%; the hoist moves where the wait is counted, it does
not remove it — a null, as predicted. The 3.5–3.6 ms of per-layer
wait is the tier waiting for the GPU to reach that layer's own
router, inherent to the serial dependence: the conclusion above is no
longer an inference from one drain probe, it is measured in one
build. Host compute per expert across this window: pre-0017 683 µs
then 430 µs (its own two separate processes), 0017 421, NOHOIST 446,
the drain probe 557.

**The discrepancy, explained.** At this cell tier-OFF read 2.6 t/s
and tier-ON 5.0, against §7.0.2x's own 10.4/10.6 and 15.0/15.5,
measured 2026-09-02 — not a regression and not an open question: the
retraction above names the cause. Every process in this section ran
with the plugin's own default device pool, not the intended 8 GiB
one, because `ARCINT_MOE_DEVICE_POOL_BYTES` (the env var arcint has
used to set it since the M7 auto-fit commit) was never exported this
window; the expert slots therefore lived in host memory (GTT)
throughout, the same GTT-spill regime §7.0.2x's own record already
describes and prices ("ratio 20 spills into GTT, 2.1 t/s"). Host
load, CPU clocks and the PCIe link were checked against this cell and
are withdrawn from the record as candidates — not the cause, and no
longer needed now the cause is named. The prompt-length difference
between the two cells (897 tokens for §7.0.2x's own record, 1,198
here) stays named as a difference, not an explanation — it was never
ruled in or out, and the pool finding above does not depend on it
either way.

**Reproduction at the true pool, measured.** A solo run with
`ARCINT_MOE_DEVICE_POOL_BYTES` set to the intended 8 GiB budget, both
production units stopped, the 35B's own files read once before the
run (page cache warm by construction, not inferred from `fincore`):
16 GiB card, ratio 50, u8 KV, n_ctx 65,536, the 1,198-token prompt, 64
greedy tokens twice per process, the 0017 plugin. Loads in 25 s (not
the 5–9 minutes the earlier, wrong-pool windows took). Tier OFF
9.2/9.4 t/s; tier ON 11.0/15.3 t/s; the two arms byte-identical to
each other. OTD counters: hit rate 59.5% OFF against 44.4% ON, 17,972
host pairs, average host compute 687 µs; ZFS ARC misses flat for the
whole run. Against §7.0.2x's own record — 10.4/10.6 OFF, 15.0/15.5 ON
— this reproduces within the run-to-run spread: the discrepancy this
section opened is closed. The 0017 in-build comparisons above stand
exactly as recorded; their own absolute rates were taken at the wrong
pool and are labelled so throughout this section, not restated here.

**Equivalence.** Tier OFF, 0017 tier ON and 0017 with the drain probe
are byte-identical to each other over the 64 greedy tokens in the
first window. Pre-0017 tier ON differs from all three there (18,040
against 17,972 host pairs — a different residency history, the
§7.0.2ae mechanism, not a 0017 regression), so no tier build is held
to byte-identity against another build until F1 (§7.0.2ae) lands and
makes the partition numerically invisible. In the clean rerun above,
by contrast, all six arms — pre-0017 included, both its samples — are
byte-identical to each other: two real windows, two different
residency histories, one that forked and one that did not, both
consistent with §7.0.2ae's own reading (history-dependent, not
always-forking). Every tier-on log across both windows carries
exactly forty `cpu_tier on:` lines (fusion unchanged).

#### 7.0.2ag M11: the step profile at depth — the drafter's own state at the ceiling, and an f16 position overflow at 65,504 (2026-09-04)

The step profile owed since §7.0.2aa left the 76k verify cost
unexplained and both drafters at zero acceptance. New instrumentation,
`ARCINT_PROFILE_CYCLE` (one log line per decode cycle: propose and
verify terms, accepted count, cycle wall) plus the served decode
line's own propose/verify terms, committed `fe4d3df`. Measured: the
dense 27B agent artifact, 24 GB card, u8 KV, one lane, auto-fit depth,
prefix cache off, a chat request, 400 greedy tokens, one process per
arm and depth.

**8.9k tokens.** Plain 15.3 t/s (forward ≈ 62 ms). MTP 7.3 t/s, 115
cycles, `n` = 2, 0.85 accepted per cycle (85%), propose 27 ms (layer
18, head 7), verify infer 206 ms, cycle 253 ms. DFlash 6.0 t/s, 123
cycles, `n` = 8, 0.73 accepted per cycle, propose 68 ms, verify infer
215 ms, cycle 287 ms.

**76k (77,134 prompt tokens).** Plain 15.3 t/s (forward ≈ 65 ms) — the
same rate as 8.9k; depth costs the base graph nothing at this shape.
MTP 1.1 t/s, 372 cycles, 0 of 372 accepted, propose 140 ms (layer 98,
head 23), verify infer 761 ms and bimodal (min 441, median 552, p75
923, p95 1,622, max 1,754 — the spikes come in pairs every 10–12
tokens and move with the MTP layer's own infer time, correlation 0.84
at 76k against 0.29 at 8.9k), cycle 902 ms. DFlash 5.1 t/s, 0 of 2,604
accepted, propose 20 ms, verify infer 176 ms and FLAT (171–176 across
the run — cheaper than its own 8.9k figure), cycle 198 ms. Greedy
output is byte-identical across all three arms at both depths — the
M11 equivalence gate holds at depth, so what follows is a cost and
acceptance problem, not a correctness one.

**Mechanism: the MTP state at the ceiling** (the investigating design
note's own reading, marked as such until the verification window
below lands). The MTP layer's own KV state is stateful and unpaged —
4 heads × 256 × (K+V) × f32 = 8 KiB per token — primed over the whole
prompt: 0.59 GiB at 77,134 tokens, 1.19 GiB at the 155,488 tokens this
arm's own auto-fit admits. The reservation's drafter term is a
weights-only residency delta, blind to this per-token growth, so the
arm logs `resident 22.47 GiB against a 22.46 GiB ceiling` — over its
own stated budget by construction, not by drift. The periodic spikes
in the verify-infer distribution are read as the eviction cost of
carrying that overcommitted state past the ceiling. DFlash's own state
is a fixed `kDflashWindow` = 2,048-row window (~84 MiB) and does not
grow with depth, which is why its own verify is flat and, at 76k,
cheaper than at 8.9k. At 8.9k the MTP state is 71 MiB and fits inside
the margin — the same mechanism, just not yet over the edge, which is
consistent with the 8.9k arm being merely expensive (3.2× plain)
rather than broken.

**Mechanism: zero acceptance, an f16 position overflow.** Both
drafters compute their own rotary embedding in-graph from the absolute
position parameter, with `inv_freq[0] = 1.0`, and neither is compiled
with a precision hint — both run under the GPU plugin's default
`INFERENCE_PRECISION_HINT=f16`, which overflows at 65,504. The base
model's own rotary survives at the same depth because its position
chain is `ShapeOf`-derived and the plugin's own mixed-precision
marking keeps that subgraph at f32; the drafters, with no such
marking, do not get it. Bisected (MTP arm, one process per cell): at
60,333 tokens, 78% accepted (89 of 114), 3.5 t/s; at 70,403 tokens, 0
of 148 accepted, 1.5 t/s — a cliff between the two, not a decay,
consistent with a hard value boundary rather than a trained-range
falloff. Falsified on the way, each checked rather than assumed: rope
tables sized for an unextended max position (no such tables exist —
the angle is computed in-graph, not read from a materialised table);
a `rope_scaling` asymmetry between the base and either drafter
(neither declares one); a 16-bit index somewhere on the host-side
position or block-table path (none — the relevant fields are `size_t`
or `int32_t` throughout); an exporter-side `rt_info` marking meant to
carry the base's own f32 rotary hint into the drafter graphs (does not
survive `save_model` in this OpenVINO build, and the base itself
carries no such marking to copy in the first place).

**Correcting the record (§7.0.1): the "445 vs 209 ms" comparison in
§7.0.2aa was not like-for-like.** The 445 ms figure was the served
MTP arm's own compiled graph (which adds a `hidden_states` Result and
a wider logits slice the plain arm does not carry); the 209 ms figure
came from a separate profiling harness under `ARCINT_PROFILE`, which
also enables `ov::enable_profiling` — and PERF_COUNT inflates wall
clock, a caveat this file already states elsewhere and had not applied
here. Different graphs, one of them under an instrumentation tax: the
comparison is retracted, not the underlying cost. This window's own
clean measurement supersedes it: the plain arm decodes at 15.3 t/s at
both 8.9k and 76k — the served forward itself does not slow with
depth at this shape; every cost this section attributes to depth is
in the drafters, not the base graph.

**The gate, restated.** MTP with one draft per step can accept at most
2 tokens per cycle, so the 3.13-tokens-per-cycle criterion on the M11
row is DFlash's own break-even (198/63.3 ms) and must not be applied
to MTP unchanged; MTP's own break-even is a cycle wall under roughly
130 ms at 76k — a bar the state-term fix below has not yet been
measured against.

**Fixes implemented, arcint side, this repository.** The MTP-state
reservation term: 8 KiB per token folded into the per-token KV term at
the drafter's own reservation line, so the admitted context now
subtracts it before max-ctx is derived — computed by hand at this
section's own fixture constants, max_ctx falls from 1,172,016 to
686,192 with the term applied, the arm's true, bounded ceiling, not the
unbounded one the weights-only delta used to report. The unit test
(`tests/test_fit.cpp`,
`mtp_reservation_term_shrinks_admitted_ctx_by_at_least_the_derivative`)
does not assert this exact pair of numbers; it asserts the general
property they are one instance of — the reduction in max_ctx is at
least the per-token term's own derivative bound at the depth actually
admitted — and a companion test
(`a_fixed_weights_only_charge_fails_the_derivative_bound`) proves the
old, wrong shape (a fixed weights-only delta) fails that same bound.
`ARCINT_DRAFT_F32=1`: the
whole drafter compiled at f32, the coarse discriminator for the
overflow mechanism above. The rotary marking: the rotary subgraph of
each drafter graph (16 nodes in the MTP layer, 6 in DFlash — the angle
chain through `cos`/`sin`) marked with `ov::disable_conversion` after
`read_model`, on by default; `ARCINT_DRAFT_ROPE_F16=1` disables it for
comparison. None of these three change default behaviour beyond the
reservation term itself, which is a refusal (a smaller admitted
context), not a new code path.

**Verification window, measured.** Same cell as above; the build
carries all three fixes, rotary marking on by default.

At 70,403 tokens: MTP with the rotary fix disabled
(`ARCINT_DRAFT_ROPE_F16=1`) — 0 of 148 accepted, 5.1 t/s, verify
157 ms per cycle; the eviction spikes are gone (the MTP-state
reservation term alone took the verify cost from 581 to 157 ms) but
acceptance stays zero, isolating the two mechanisms from each other.
MTP default (rotary kept f32, 16 nodes marked) — 80.5% accepted (66
of 82), 7.9 t/s, output byte-identical to plain. MTP with
`ARCINT_DRAFT_F32=1` (the whole drafter compiled f32) — acceptance
returns (1 accepted per cycle over the 29 cycles run) at a
prohibitive 6.3 s per cycle (propose 520 ms, verify infer 5.8 s — the
f32 state doubles to 16 KiB per token and re-crosses the ceiling the
MTP-state term was built to catch), stopped after 29 cycles as
sufficient evidence rather than run to completion. DFlash default (6
nodes marked) — 27.5% accepted (98 of 357), 15.5 t/s against plain's
15.3, byte-identical.

At 77,134 tokens, with the MTP-state term now charged: `n_ctx` clamps
to the admissible 127,536 (was 155,488 unclamped; the reservation
line reads "MTP state 0.97 GiB (8.0 KiB/token)", no ceiling overshoot
this time). MTP default — 90.8% accepted (177 of 195), 4.9 t/s
(verify 58.2 s over 195 cycles ≈ 298 ms, propose ≈ 54 ms per cycle,
cycle ≈ 390 ms against MTP's own ~130 ms break-even) — MTP still
loses to plain's 15.3 t/s at this depth even with every draft
accepted, because its own two-token ceiling cannot outrun a 390 ms
cycle. DFlash default — 40.6% accepted (276 of 679), 18.8 t/s, above
plain, about 4.4 tokens per verify cycle: the M11 gate (> 3.13
tokens per verify cycle on the 24 GB card, byte-exact drafter on vs
off) PASSES at 76k with DFlash. Both arms are byte-identical to plain
(sha equal) at both depths.

**The decision input, stated plainly, as a recommendation and not a
change made.** On this artifact MTP never beats plain decoding at any
depth measured — 7.3 against 15.3 t/s at 8.9k, 3.5 t/s at 60k before
the fixes, 4.9 t/s at 76k with every fix landed — while DFlash loses
at 8.9k (6.0 t/s) and wins at 76k (18.8 t/s). The §8 "depth-gated
drafter" item can now be decided from data rather than deferred:
DFlash with a depth gate (or plain below the crossover), MTP off for
this artifact.

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

#### 7.0.2ah M10 re-scoped: the gate is priced in VRAM, and only a new kernel pays in that currency (2026-09-05)

**The decision (operator, 2026-09-05).** 0.3.0 ships with M10 re-scoped as
§7.0.2y proposed: the context claim the row was written to buy is
discharged by M8's u8:i4 flag (+28.3% on the 24 GB card, +28.4% on the
16 GiB card, §7.0.2y's table, measured); the VRAM-resident sub-4-bit
expert path is a new GPU-kernel milestone and is backlogged to 0.3.1
(`docs/milestone-0.3.0.md`, "Backlog for 0.3.1"); the host-side
sub-4-bit storage question stays attached to M14 as a candidate
extension, also 0.3.1. No M10 code was written for 0.3.0.

**The recon that informed it** — a bounded read of the sources, no code,
recorded here in its public-safe form (the working note is
operator-local):

- *NNCF 3.3.0.* `INT3_SYM`/`INT2_SYM` are implemented, not stubs; the
  OpenVINO output is a plain dequantize subgraph — a packed `u3`
  constant, the `2^(bits-1)` offset folded as an `i8` zero-point, `f16`
  group scales — and the pinned runtime defines `u2`/`u3` element types,
  so the artifact is representable. Mixed precision is per weight
  tensor, a binary primary/backup split by a sensitivity ratio; there is
  no per-node bit-width API. In the batched IR form the served artifacts
  use (one stacked constant per layer per tensor) per-expert granularity
  is inexpressible; in the unrolled form it is constructible with
  multi-pass compression under `ignored_scope`, unverified. NNCF 3.3.0
  is installed in the tooling venv, so the `compress_weights(INT3_SYM)`
  smoke test that would clear §7.0.2y's "undocumented" caveat is a
  ten-minute item — still unrun, carried to the backlog.
- *K-quant formats (llama.cpp).* Per 256-weight block: Q3_K 110 B
  (3.4375 bpw), IQ3_XXS 98 B (3.0625), Q2_K 84 B (2.625); the i-quants
  index a compile-time grid that an importer would vendor (MIT). So
  "dequant to int4 at load" is a re-quantization: the artifact on disk
  stays sub-4-bit, the resident representation is int4.
- *The pinned runtime.* The MoE fusion matcher
  (`keep_moe_3gemm_const_precision.cpp`) requires `u4` on all twelve
  weight and zero-point constants, so a `u3` expert artifact never
  reaches the fused MoE op; the kernel type table is `{u4, i4, u8, i8}`
  and oneDNN has no 3-bit type, so a `u3` GEMV/GEMM is a from-scratch
  kernel with in-kernel dequant — §7.0.2y's 800–1,500-line HYPOTHESIS
  stands as the order of magnitude. Per-expert uniformity is structural:
  one element type per layer per tensor in the batched form, one byte
  count per expert in the offload provider's per-expert bin ranges.
- *The gate, resolved.* Its currency is VRAM — "≥ 15% more max context"
  is KV headroom after weights. Route 2 as written (K-quant import,
  dequant to int4 at load) leaves int4 resident and is byte-identical to
  the baseline on the gated axis; its real wins (disk, host pool bytes, a
  host kernel computing the blocks natively) are M14's territory. Route 1
  (NNCF `u3`) needs the same new kernel. The routes converge; what
  remains is the resident format — `u3` group-quant or K-quant blocks —
  decidable by measurement on one expert layer once a kernel path exists.
- *Magnitude, bounded from the record (an estimate, not a measurement).*
  The coder's experts are ≈ 85% of its parameters (184 × 3 × 2048 × 512
  × 40 layers ≈ 23.2B of 27B), ≈ 10.9 GiB of the 12.8 GiB int4 weights;
  4 → 3.44 bpw on all of them frees ≈ 1.5 GiB against a KV budget of
  ≈ 1.4 GiB on the 16 GiB card (§7.0.2y, 171,312 tokens × 8.8 KiB); a
  rarely-routed-only variant at a third of the experts still clears 15%.
  The byte count is not the obstacle; the kernel is the milestone. The
  dense 27B has no experts, so M10 never touched the 24 GB card's agent.

**Carried to 0.3.1.** (1) The VRAM-resident sub-4-bit expert path: a
matcher for the new element type, an oneDNN bypass, GEMV/GEMM with
in-kernel dequant, judged by a fusion-impact profile (ground rule 2), with
a decode regression of known sign (§7.0.3's u4-KV precedent). (2) The two
pre-work measurements: the `INT3_SYM` smoke test, and the routing
histogram (patch 0013, `MOE_OTD_ROUTING_HIST`) over a longer corpus — the
acceptance prompt alone left most of 7,360 experts at 0–2 routings, no
distribution to threshold "rarely-routed" on. (3) The per-expert bpw map
as an artifact format (design). (4) Optional: K-quant storage for M14's
host tier. Closing M10 as re-scoped is not a "baseline reached" claim;
ground rule 3's survey obligation transfers to the 0.3.1 item.

#### 7.0.2ai The 0.3.0 release gate: the static partition holds §3.4, refuses the 16 GiB card, and is not yet fast (2026-09-05)

The acceptance set of §5/§5.1 run on the release-candidate bits — this
tree at the tag, and plugin patches 0003–0018 as the built
`marfrit-openvino +p4` package, its libraries extracted and loaded ahead
of the installed runtime in every process (each process's map checked:
the GPU plugin came from the extracted package and the runtime string
ended in `marfrit-p4`). One fresh process per arm; `ARCINT_MOE_DEVICE_
POOL_BYTES` at 8 GiB for every offload cell (§7.0.2af's lesson);
production stopped only on the card being borrowed and re-verified
fresh after every window.

**M14's reference cell under the static partition (16 GiB card, 35B
int4, ratio 50, 8 GiB pool, u8 KV, one lane, n_ctx 65,536, the
1,198-token prompt, 64 greedy tokens, two requests per process).** Tier
OFF, two processes: prefill 65.7/87.6 and 82.4/87.4 t/s, decode 9.5/11.4
and 11.0/11.3 — inside the 2026-09-04 reproduction's spread. Tier ON on
the bits as they stood at the start of the gate: **refused at load, both
processes.** The device term under the static partition was priced
analytically at the whole pinned pool, 7.50 GiB (`source: static`), and
9.17 GiB of weights + 0.47 of drafters + 7.50 left the 15.11 GiB card
zero KV. The LRU arm on the same card, same pool budget, minutes earlier,
had priced the same pool by the plateau probe at 0.11 GiB: the pool's
slots live in host-mapped memory on this card (§7.0.2t's two-ledger
finding), and only that 0.11 GiB class is VRAM. The analytic figure was
never wrong about the pool's size, it was wrong about where the driver
put it — the over-charge went unnoticed on the 24 GB card, where the M9
gate passed because 22.7 − 9.17 − 0.47 − 7.50 still leaves room. Fixed
in this release before the tag: the plateau probe runs under the static
partition too (with the prefill fallback's own fix in place nothing
evicts, so it saturates as the pinned set fills on first use), the
analytic figure is the fallback only when the probe throws and is
logged beside the probe's reading as a cross-check, and the zero-byte
refusal applies to the fallback alone — a probe that reads ~0 on this
card is a measurement. `slot_source` gained `probe-static`.
On the fixed binary, with no forced term, the probe ran under the
static partition and settled at 0.11 GiB (`source: probe-static`), the
cross-check line read "pinned-pool ceiling 7.50 GiB (source: config)
against 0.11 GiB measured resident (source: probe-static)", and the
reservation admitted max ctx 425,328 — the same admission the LRU arm
gets. Two tier-ON processes: 11.1/4.0 then 26.7/16.4 t/s (prefill/decode,
first then second request) and 25.8/15.4 then 26.6/16.4. All arms —
tier OFF twice, tier ON forced twice, tier ON fixed twice — produced the
same 64 tokens, and every process's second request matched its first.

**F2's runtime at the cell.** Warm, the tier under the static partition
holds M14's result: decode 16.4 t/s on the second request of both
processes (15.4 already on the first request of the warm-cache one), at
the LRU-era record (15.0/15.5, §7.0.2x) and above tier OFF's 11.3–11.4 on
the same request in the same window (first requests: 4.0 and 15.4 against
tier OFF's 9.5 and 11.0). Two costs stand beside it. Prefill runs at
26.6–26.7 t/s on the second request against tier OFF's 87.4–87.6 (first
requests 11.1 and 25.8 against 65.7 and 82.4): the plugin's counter
`grouped_fallbacks=40` says every layer's prefill took the per-expert
fallback (the grouped-GEMM prefill refuses any batch with a non-resident
expert, and under a static half every batch has one), with the host path
per token for the non-resident experts. And the first processes of a
fresh sequence pay a one-time warming: with the device term forced to the
probe's figure so the fit would admit them (a diagnostic, not the shipped
path), the first two tier-ON processes loaded in 340 and 275 s (tier OFF:
30–45), served their first request at 2.8/0.1 and 9.9/4.2 t/s and their
second at 9.2/4.7 and 26.7/16.6, while the fourth tier-ON process of the
night was fast from its first request; on the fixed binary tier-ON loads
ran 215–585 s against tier OFF's 30–45. `created_onednn_kernels=325` per
process is one candidate owner (first-use JIT of the per-expert kernels,
one per token count); a page cache warmed by the earlier processes'
first-use fills is another; the two were not separated. M14's own gate —
an honestly reported decode number against the device-tier path, win or
lose — reads, under the static partition: **decode holds, prefill loses
3×, and a cold sequence's first requests are minutes, not seconds.**
Carried to 0.3.1 as "the static partition's prefill and cold start"
(`docs/milestone-0.3.0.md`, backlog). One counter note for readers of
`OTD_PERF` lines: under the static partition the LRU hit counters read
`gpu_hits=0, gpu_misses=1116` — they count the LRU path, which the
pinned pool never takes; the line is not evidence of thrash.

**§3.4 under the static partition, the reason patch 0018 exists.** All
four served arms — tier OFF twice, tier ON twice — produced the same 64
tokens (one hash), and every process's second request matched its
first. E2, the history check §7.0.2ae introduced (tier on, no prefix
cache): PROMPT, PROMPT, CONT in one process, CONT in a fresh one. The
two PROMPT answers are identical, and **CONT after history is
byte-identical to CONT in a fresh process** — the exact fork that
condemned the LRU tier on 2026-09-04 is gone.

**The dense 27B on the 16 GiB card** refuses at any depth on this
artifact: 13.59 GiB of weights plus 1.18 of drafters (MTP off) against
15.11 usable leaves no KV at n_ctx 8,192. §5's "A770 ≥ 17 t/s for the
dense 3.8" bar dates from the GGUF/Vulkan agent baseline and is not
measurable on the int4 IR there; the dense agent is served on the 24 GB
card, where its numbers are below.

**The 24 GB card.** The coder at M9's cell (`--offload-ratio 20
--paged-kv u8:i4`, 8 GiB pool): the equivalence suite, all checks passed,
at one lane and again at two; the concurrency suite, all checks passed.
The n-gram flake of §7.0.2ae (the suite's "speculative decoding is
deterministic across runs", `--draft 4 --draft-ngram 3`, on the 35B tier
configuration): six fresh processes, one after another, six identical
outputs (4 of 12 drafts accepted in each) — one failure in about nine
runs on the record, not reproduced, kept as a rare event rather than a
mechanism. The coder served-style (no offload, u8 KV) on this card: 53.4
t/s on a cold first request, 69.2 on the second, byte-identical — §5's
≥ 60 t/s bar holds at steady state.
The dense 27B agent (`--paged-kv u8`) on the same card: the concurrency
suite, all checks passed; the equivalence suite, all checks passed —
including the MTP section, which is M11's shallow gate on the release
bits (MTP identical to plain greedy; MTP with the prefix cache, warm
equal to cold with a real hit; the head accepting 68.4% at this depth),
and the cold/warm cache section — with one server start skipped and then
isolated: `--prefill-chunk 1` refuses at load on any artifact with an
MTP head. The load-time check that verifies the logits slice runs a
probe forward of `--prefill-chunk` tokens and expected the two rows MTP's
verifier slices for, from a one-token forward; chunk 1 with `--mtp off`
loads, chunk 2 with MTP loads. Not a served configuration (the suite's
chunk sweep is a measurement, not a gate, §5), and a refusal rather than
a wrong answer — but it aborted the suite's sweep, and its message
blamed the token axis. Fixed before the tag: the expectation is
`min(slice rows, probe tokens)` (`logits_slice_rows_expected`, `fit.h`,
with the discriminating case red-first in `tests/test_fit.cpp`); on the
rebuilt binary chunk 1 with MTP loads and logs "logits slice verified: 1
row(s) for a 1-token forward (the slice keeps the last 2)", and the dense
agent's full equivalence suite with its chunk-1 sweep included passes
every gated check (chunks 1, 7 and 64 all differ from the unchunked
baseline, reported not gated, as §5 says they will).

**The coder exactly as served (16 GiB card, `--paged-kv u8`, n_ctx
98,304, prefix cache 2 GiB, the live unit's own flags on the release
binary).** Prüfstand — the acceptance task at the artifact's decoding
regime for this row, greedy, thinking off, scored by executing the
candidate — **10/10, twice, byte-identical answers**; 479-token greedy
answers at 48.0 and 49.5 t/s decode (prefill 251 and 676 t/s), one
reservation overshoot self-corrected at load (14.87 against 14.86 GiB,
pass 1 of 4). The equivalence suite on this configuration: all checks
passed; the concurrency suite: all checks passed.

**Device-free, on the final tree:** 409 unit cases (413 with the OpenVINO
backend on the dev host, one of them a CPU-affinity pin the container's
cgroup does not allow), the 64-check curl round-trip and the
lane-accounting stress, all green on this repository's aarch64 build host, where the affinity
case passes; the same three there under UBSan (`-fsanitize=undefined
-fno-sanitize-recover=all`) with zero runtime errors; and under
ASan+UBSan (no-recover, the stub build) on the x86_64 dev host: 409 cases
with the container-restricted affinity case failing, round-trip and stress
passed, zero sanitizer reports. Production on the dev host was restored and freshly verified
after every window (both units active on their cards, health 200).

#### 7.0.2aj The acceptance target's first real run: the enumeration corrected six times, and tier ON is not tier OFF (2026-09-05)

The 0.3.1 lead item (§5, `docs/design-0.3.1-test-ladder.md`) split the
ladder into `ctest -L unit` and `tests/acceptance/run.py --all`, and its
Increment 2 owed one card window: the committed runners on the committed
tree, both cards, every cell, no ad-hoc script. Run on the dev host on the
Increment-2 commit with the `+p4` runtime, both production units stopped
for the window and restored after (both active, health 200, fresh
verification). What follows is what the target did, cell by cell, and
what it corrected. The numbers here are the window's sanity envelope for
the fill (§8.4 of the design note); the references themselves come from
the follow-up window in §7.0.2ak, printed by the corrected runners.

**Passed on the 24 GB card** (`GPU.0`): the coder at M9's offload cell
(`--offload-ratio 20 --paged-kv u8:i4`, 8 GiB pool) — the equivalence
suite at one lane and at two, the concurrency suite; the coder served
(`--paged-kv u8`, no offload) — the equivalence suite; the dense 27B agent
(`--paged-kv u8`) — the equivalence suite including its MTP section. The
coder cells each carried the equivalence suite's "no MTP head, skipping
the MTP gates" path, promoted to a named skip `<cell>/mtp-section` —
three here and a fourth on the other card, none named on the command
line, so the run refused them (correction 1, below).

**Passed on the 16 GiB card** (`GPU.1`): the coder served (`--paged-kv
u8`) — the equivalence suite (its MTP skip likewise); the n-gram
determinism cell (`--draft 4 --draft-ngram 3` on the 35B tier
configuration, `--offload-ratio 50 --moe-cpu-tier`): six fresh processes,
six identical outputs (sha256 `434a635e…` on all six, 235 prompt tokens,
64 completion tokens, 33.3% of drafts accepted in each) — and the runner
would have failed a drafter that never fired, which is the check the
Increment-2 review added.

**The tier reference cell (16 GiB card, 35B int4, ratio 50, 8 GiB pool, u8
KV, one lane, n_ctx 65,536)** sized its prompt against the artifact's own
tokenizer to 1,167 tokens (target 1,198 ± 3%, two rounds) and ran the
four fresh processes, two requests each, 64 greedy tokens:

| process | request | prefill | decode |
|---|---|---|---|
| tier OFF, 1 | 1st | 79.8 t/s | 9.3 t/s |
| tier OFF, 1 | 2nd | 86.2 | 12.4 |
| tier OFF, 2 | 1st | 79.8 | 12.1 |
| tier OFF, 2 | 2nd | 86.1 | 12.5 |
| tier ON, 1 | 1st | 23.1 | 14.8 |
| tier ON, 1 | 2nd | 26.3 | 16.6 |
| tier ON, 2 | 1st | 25.4 | 15.6 |
| tier ON, 2 | 2nd | 26.3 | 16.4 |

Warm decode ON 16.6/16.4 against OFF 12.4/12.5 — ratio 1.34/1.31 on the
like-for-like second requests; §7.0.2ai's window had 16.4/16.4 against
11.3–11.4 (1.44). Tier OFF's warm decode drifted 9% between the two
windows on the same card and flags, which is why the OFF rate stays a
report and the ratio, taken inside one window, is the gate. Prefill
under the static partition: 26.3 warm against 86 — the 3× of §7.0.2ai
unchanged, 0.3.1's charter. E2 held: PROMPT == PROMPT in one tier-ON
process, and CONT from that warm process == CONT from a fresh one
(prefill 24.8–26.3, decode 15.3–16.6).

Then the finding. The runner compared all seven later outputs against
tier OFF's first: the three OFF outputs identical to it, all four ON
outputs different from it — `FAIL tier-reference-cell: runner exited 1`,
4 checks failed. That is not §3.4 failing. Device f16 and host f32
arithmetic are not bit-equal (§7.0.2ae, §7.0.2af), the 0.3.0 gate's
ON-equals-OFF agreement was a property of that one prompt, and §3.4
promises history-independence — tier ON identical to itself across
processes and requests, which the window's data shows (four ON outputs,
one text; four OFF outputs, one text) and E2. The runner was gating a
claim the engine never made; corrected on `main` the same day to gate
the two groups separately plus E2, and report ON-versus-OFF with the
first differing character (correction 4).

**The depth ladder (98,147-token prefill, n_ctx 102,243, one fresh process
per card and precision)** sized its prompt to 98,187 tokens (two rounds,
deviation 0.0%). On the 24 GB card at u8: served, no fault line, 1,018 and
1,026 t/s on the two full prefills, 27.7 t/s over the 32 decoded tokens
(the belt fires only under 4-bit values, so u8 kept the pre-cap chunk of
2,048). On the 24 GB card at u8:i4 the cell **failed**. The
chunk belt priced 4-bit values at this depth and cut the served chunk from
2,048 to 128 ("empirical belt, measured — limit: budget", scratch 606 MiB
at chunk 128, no partition bound set, f16 partials); the sizer's two
rounds then prefilled 78,867 tokens in 546 s (144 t/s) and 98,187 in
812 s (121 t/s) — an 8× prefill price against u8 at this depth and this
chunk, against the +72% §7.0.2aa measured at 71.7k on the 16 GiB card
with both arms at the same pre-belt auto-fit chunk; the two are not the
same measurement, since here the belt cut the chunk sixteenfold and u8
ran at 2,048, and the number is for the `u8i4-prefill-price` campaign to
take again with the chunk held constant — and the measured request, the
third deep prefill in that process, ended in `clEnqueueMapBuffer …
CL_OUT_OF_RESOURCES` and the process terminating on `clFinish`. The host
kernel log at that moment: a GuC job timeout on the 24 GB card ("Check
job timeout … not started"), a device coredump, a GT reset, and
`Timedout job` lines in two processes — the ladder's arcint and a python
process that was not the runner's (the runner's python never touches the
card). That signature is **not** §7.0.2ad's: that class shows a page-fault
storm at one address and CAT-error lines, and §7.0.2ad records that no
`Timedout job` line accompanies any of its crashes; here there were no
page-fault or CAT lines at all. The class is open. A second client on the
card is a candidate, not a finding — nothing in this window measured
VRAM headroom — and which client the python was is not on the record;
both processes were gone before it could be read. The record's passes of
this prompt on this card at u8:i4 (§7.0.2ad's knob table) ran with the
partition bound at 32 or the belt bypassed, one prefill per process, and
the host's state in those windows was not recorded, so they do not
bracket this one. Two consequences on `main`: the runner sizes its prompt
once per artifact and reuses it, so every cell after the sizing one asks
exactly one request (the shape the 0.3.0 gate measured; the sizing rounds
were also 546 s and 812 s of wall time for the u8:i4 cell at this depth),
and the cell's own u8:i4 result on the 24 GB card is owed by a rerun on a
quiet card, one prefill per process, recorded in §7.0.2ak. The
16 GiB card's two cells: at u8, served — 619 and 622 t/s on the two full
prefills, 31.7 t/s over 32 decoded tokens, no fault line; at u8:i4,
**refused at load**, and that refusal measured the runner, not the card:
the honest fit priced 4-bit values at chunk 128 (scratch 606 MiB,
unbounded partials) and admitted 101,232 tokens on one lane, the runner
asked for 102,243 (the prompt's target plus a flat 4,096) and missed by
1,011. The prompt itself (98,187 tokens plus 32 of completion) fits
inside what was admitted. The record's serves of this prompt on this card
at u8:i4 (§7.0.2ac, §7.0.2ad) ran with the partition bound at 32, which
the cell does not set — it measures the unbounded default a user gets
without the knob, and now asks for the prompt plus 2% plus the completion
plus block slack, nothing more; its 16 GiB u8:i4 result is owed by the
same rerun as the 24 GB one.

**Device-free cells, on the dev host:** sanitizers — a fresh ASan+UBSan
build (x86_64, no-recover, `ctest -L unit`), five unit tests passed, zero
sanitizer reports; package-build — the release recipe run on the
Increment-2 commit's own tarball (the recipe's default is the v0.3.0 tag),
unit gate five of five, `built: … arcint_0.3.0-1_amd64.deb` — labelled
0.3.0-1 by the recipe's version and not a release package, left in place.

**The six corrections, all on `main` before this record** (the runners
that will fill the references are the corrected ones):

1. *Expected skips.* The coder artifact carries no MTP head, so the
   equivalence suite's MTP section is a known skip of that artifact class,
   not an anomaly to name on every command line. `cells.json` now declares
   `expected_skips` per cell; run.py pre-names them, fails the cell if a
   declared skip does not occur, and rejects `--allow-skip` on one.
2. *Rate claims the suite never measured.* `coder-served-large` and
   `coder-served-small` said "≥ 60 t/s" and "48.0/49.5" in their gate text
   while running the equivalence suite, which measures no rate. A decode
   probe (`decode_probe.sh`: one fresh process, two requests on a
   1,050-token sized prompt, byte-identity gated, cold and warm rates
   emitted) is its own cell on each card.
3. *Concurrency on the small card and the dense agent* ran at the 0.3.0
   gate but were never enumerated. Two cells added. Sixteen cells now.
4. *The tier cell's gate*, above.
5. *The ladder's sizing shape.* Sizing the prompt through the server in
   every cell cost two full prefills per cell at this depth and made each
   process ask three deep requests where the 0.3.0 gate asked one. The
   runner sizes once per artifact and reuses the prompt.
6. *The ladder's margin.* A flat 4,096 tokens over the prompt is what the
   16 GiB card refused at u8:i4, not the prompt. The runner asks for the
   prompt plus its sizing tolerance, the completion and block slack.

And a seventh the fill exposed: the schema had no report-only reference,
so filling the tier cell would have gated its prefill (freezing the 3× as
a specification) or dropped the record. `gate_at: null` is now the
report-only form (design note §8.9), red-first in the self-test.

The target's own summary: `12 cell(s) run, 9 passed, 1 skipped, 4 named
skip(s) promoted, 2 failed`, exit 1, 8,512 s of wall time; the one skip
is the external cell, named. The two failures and the four unnamed skips
are the findings above, each resolved on `main` when the target reported
them — the tier cell's gate, the four declared skips, and the ladder's
margin and sizing shape — with the ladder's two u8:i4 results owed by a
rerun. Production was restored after the window and verified fresh: both
units active on their cards, health 200 on both endpoints.

#### 7.0.2ak The follow-up window: the runner's index was off by one, the first process of a window is cold, and two tier-ON processes disagreed (2026-09-05)

The small-card follow-up owed by §7.0.2aj — the corrected tier runner,
the small-card decode probe and the small-card concurrency cell, each
through `run.py --cell` on the tree with the report-only reference form,
`+p4` runtime, the 16 GiB card's unit stopped and the other card's unit
left running because the operator had that card. Rates were still
`REPORT` lines (references null). What it found, in the order it
matters:

**The decode probe (coder, `--paged-kv u8`, no offload, 16 GiB card, a
prompt sized to 1,050 ± 3 % tokens, 64 tokens, one fresh process, two
requests):**
byte-identical outputs; cold first request 46.0 t/s decode / 510.5 t/s
prefill, warm second 47.6 / 508.9 — beside §7.0.2ai's 48.0/49.5 on the
same card and precision. PASS.

**The tier reference cell (16 GiB card, 35B int4, ratio 50, 8 GiB pool,
u8 KV, one lane, n_ctx 65,536, 1,167 sized tokens, 64 greedy tokens,
four fresh processes, two requests each, then E2):**

| process | request | prefill | decode |
|---|---|---|---|
| tier OFF, 1 | 1st | 52.8 t/s | 0.3 t/s |
| tier OFF, 1 | 2nd | 5.3 | 1.7 |
| tier OFF, 2 | 1st | 48.8 | 11.7 |
| tier OFF, 2 | 2nd | 85.2 | 11.9 |
| tier ON, 1 | 1st | 8.6 | 4.9 |
| tier ON, 1 | 2nd | 26.3 | 16.5 |
| tier ON, 2 | 1st | 23.5 | 15.4 |
| tier ON, 2 | 2nd | 26.3 | 16.4 |
| E2, warm process | 1st, 2nd | 25.4, 26.3 | 15.5, 16.5 |

(The E2 process's CONT request and the fresh CONT process are not in
the table: the runner's report tails four lines of a six-line log, so
their rates were not extracted.) Loads, "paged model ready": off1 12.4 s,
off2 not extracted, on1 6.2 s, on2 15.6 s, the E2 process 5.3 s — every
load seconds, none of them §7.0.2ai's 215–585 s. The OFF processes priced
their expert slots from the plateau probe (`source: probe`), the ON
processes under the static partition (`source: probe-static`), each arm
as its flags say; within an arm the ledgers match line for line (the
same 0.11 GiB figure, the same 7.50 GiB host-side pool, the same driver
report).

Three readings, two of them corrections of the runner.

*The runner's index was off by one.* Its `metric_value` matched
"prefill|decode" and "t/s", and the server's load banner — the
artifact's recorded rates, "62.7 t/s decode at 53.5k, 1584 t/s prefill"
— matched both, so every fixed request index named the request before
it. The window emitted `decode-warm-2nd-off 0.3`, `prefill-warm-2nd-off
52.8`, `decode-warm-2nd-on 4.9`, `prefill-warm-2nd-on 8.6` and
`decode-ratio-on-off 16.33` — every one the first request's number
under the second request's name, and the ratio, §8.1's headline gate,
meaningless. Shown red against the fake server once it printed the
banner and distinct per-request rates (65.0 emitted for a second request
of 66.0), fixed in the tier runner to match the server's own `slot N:`
request lines, green (66.0); the ladder and the decode probe index from
the end, where the banner was always first, so their numbers stand and
only their matchers were tightened. The fake had hidden it by printing
no banner — a fixture that reproduces less of the real log than the grep
depends on proves the wrong thing — and a committed fixture log with the
real banner now guards the index. The corrected runner had, at the time
of writing, run only against the fake; its first card run is the
diagnostic rerun below. With the fix, the window's own second-process
figures give a ratio of 16.4/11.9 = 1.38 against §7.0.2aj's 1.31–1.34
and §7.0.2ai's 1.44. `grouped_fallbacks` was emitted by nothing: the
plugin's counters were not enabled, so §8.1's owed confirmation that 40
is per process regardless of request count did not happen here.

*The first process of a window is the cold one, on the OFF arm too.*
The order of events: the first window's driver restarted both
production units (both models loaded), the follow-up driver stopped the
16 GiB card's unit again within a minute, rebuilt the tree, and `off1`
was the next process on that card. It loaded in 12.4 s and then decoded
at 0.3 t/s — 216 s for 64 tokens, all of it in the graph — and its
second request prefilled at 5.3 t/s; `off2`, the same flags seconds
later, 48.8/11.7 then 85.2/11.9. Two things this separates that
§7.0.2ai could not: the load term was seconds here, not minutes, so the
first-request cliff exists without the slow load; and tier OFF creates
no host oneDNN kernels, so this instance of the cliff cannot be the
kernel-JIT candidate. What it was — page cache, first-use fills, or
something neither counter names — is a reading, one sample, no counter
read; the `static-partition-cold-start` campaign's ladder (steps 3 and
4) is where it gets measured, and its step 5 now compares outputs as
well as times. Consequence for the reference: the runner
takes each arm's metrics from the arm's *second* process (the first
process's lines are still reported), because a cold-sequence cost is
that campaign's number, not a decode regression. Warm ON decode
16.5/16.4 and prefill 26.3/26.3 reproduce §7.0.2aj's 16.6/16.4 and
26.3/26.3 exactly; warm OFF 11.9 sits between §7.0.2ai's 11.3–11.4 and
§7.0.2aj's 12.4–12.5.

*Two tier-ON processes disagreed.* The identity groups were gated as
corrected, each member against its group's baseline: all three later
OFF outputs identical to `off1`'s first; `on1`'s second output identical
to its first; **both of `on2`'s outputs different from `on1`'s first** —
`FAIL on2-run1 byte-identical to on1-run1`, `FAIL on2-run2 …`,
`tier-reference-cell: 2 check(s) failed`, exit 1. What was *not*
compared, because the runner compares against the baseline only:
`on2`'s two outputs with each other, and E2's texts with either —
the E2 process's PROMPT outputs were checked only against each other,
and the fresh E2 process answered CONT, a different prompt. So the
measured fact is exactly this: two tier-ON processes on the same flags
and the same ledger produced different greedy text for the same prompt,
where §7.0.2aj's four had produced one. §3.4's promise under the static
partition is tier ON identical to itself across processes; this is a
candidate violation. Its variable is not on the record: `on1` was the
process with the slow first request (8.6/4.9 against `on2`'s 23.5/15.4)
but the faster load (6.2 s against 15.6), so "cold" names the request,
not the process, and the resident set itself may have differed — the
plugin's per-process counters (`resident_checksum`, `static_partition`,
the seed) were not enabled, and the runner's work directory was gone
with the texts. The runner now prints every output's hash and, on any
identity failure, both texts and where they first part. Owed, on a quiet
card with the counters on and the outputs kept: the tier cell again,
and the two ON processes' texts and checksums against each other,
before any reference for this cell is written. Until then the tier
cell's references stay `null`.

**The small-card concurrency suite** (coder, `--paged-kv u8`, `--parallel
2`): all checks passed, 126 s. The window's wall time was 3,004 s.

The 16 GiB card's unit was restored after the window and verified
(active, health 200); the other card's unit was never touched.

#### 7.0.2al The ladder served on a quiet host, the divergence did not reproduce, and the references are filled (2026-09-05)

Three windows closed the day, each through `run.py --cell` on the
corrected runners, `+p4` runtime.

**The 24 GB card's two cells** (the card handed back by the operator for
this, its unit stopped for four minutes): the decode probe (coder,
`--paged-kv u8`, no offload, a prompt sized to 1,062 tokens, 64 tokens,
one fresh process, two requests) — byte-identical, cold 66.6 t/s decode
/ 2,812 t/s prefill, warm 66.5 / 2,821; §7.0.2ai's ad-hoc figure was
53.4 cold / 69.2 warm, and §5's ≥ 60 t/s bar holds on the runner's own
sample. The dense agent's concurrency suite (`--paged-kv u8`, two
lanes): all checks passed.

**The depth ladder again**, both units stopped, host load 0.75, the
runner sizing once (98,187 tokens) and every later cell asking exactly
one request:

| card | precision | prefill | decode (32 tok) | result |
|---|---|---|---|---|
| 24 GB | u8 | 1,025.5 t/s (95.8 s) | 20.8 t/s | served, no fault |
| 24 GB | u8:i4 | 120.9 t/s (812.4 s), chunk 128 | 45.9 t/s | served, no fault |
| 16 GiB | u8 | 621.0 t/s (158.1 s) | 29.2 t/s | served, no fault |
| 16 GiB | u8:i4 | 170.3 t/s (576.6 s), chunk 128 | 26.1 t/s | served, no fault |

So §7.0.2aj's two u8:i4 results are replaced, and its two readings
sharpen: the 24 GB card serves this prompt at u8:i4 with one prefill
per process on a quiet card, which leaves the morning's fault with the
two variables §7.0.2aj named (a third deep prefill in one process, a
second client on the card) and takes depth alone off the list; and the
16 GiB card serves it inside the honest fit once the runner asks for
what it needs (n_ctx 100,653 against 101,232 admitted). The u8:i4
prefill price at this depth, chunk 128 against u8's 2,048, is 8.5× on
the 24 GB card and 3.6× on the 16 GiB one — numbers for the
`u8i4-prefill-price` campaign, chunk held constant next time. The
short decode windows (32 tokens) are dominated by emit on the u8 cells
(0.83 of 1.54 s on the 24 GB card) and are reports only.

**The tier cell's diagnostic rerun** (16 GiB card, its unit stopped,
the other card's unit active and idle, host load 0.29, the plugin's
counters on, the runner printing every output's hash):

| process | request | prefill | decode |
|---|---|---|---|
| tier OFF, 1 | 1st / 2nd | 79.6 / 86.2 t/s | 9.1 / 12.4 t/s |
| tier OFF, 2 | 1st / 2nd | 79.8 / 86.4 | 12.0 / 12.5 |
| tier ON, 1 | 1st / 2nd | 15.7 / 26.2 | 14.6 / 16.4 |
| tier ON, 2 | 1st / 2nd | 25.5 / 26.3 | 13.0 / 16.4 |
| E2 warm process | 2nd / CONT | 26.3 / 23.9 | 16.5 / 13.9 |
| E2 fresh process | CONT | 25.5 | 16.0 |

All four tier-ON outputs one hash (`5d8e98890376923d`), all four tier-OFF
outputs one hash (`8b90f4b21b815a3c`), ON against OFF differing at
character 194 of 349 (permitted), E2 held; `tier-reference-cell` exit 0.
§7.0.2ak's divergence — two ON processes disagreeing — did **not**
reproduce, and neither did its first-process cliff (`off1` here 9.1 then
12.4, §7.0.2aj's 9.3 then 12.4). One observation, one failed
reproduction: it stays on the record as observed once, with its hashes
now printed on every run so the next occurrence can be read, and the
`static-partition-cold-start` campaign's ladder still compares outputs.
The emitted metrics, from each arm's second process: `decode-warm-2nd-
off 12.5`, `prefill-warm-2nd-off 86.4`, `decode-warm-2nd-on 16.4`,
`prefill-warm-2nd-on 26.3`, `decode-ratio-on-off 1.31`, and
`grouped-fallbacks-on 400` — not the 40 §8.1 took from §7.0.2ai as "one
per layer per process". Four hundred after two requests of 1,167 tokens
over 40 layers is five per layer per request, or ten per layer per
process, and the counter's unit is not on the record either way; the runner now prints
the whole `[OTD_PERF]` line per process (the work directory is deleted
on exit, and this run's lines went with it), and that counter is a
report until its unit is read off the plugin. The second process's warm ON
decode is 16.4 for the third window running (§7.0.2aj 16.6/16.4,
§7.0.2ak 16.5/16.4, here 16.4/16.4), prefill 26.3 in every one. §7.0.2ak
asked for the two ON processes' texts against each other before any
reference for this cell was written: four outputs, one hash, is that
comparison, made.

**The fill.** Per the design note's §8.3, from the committed runners'
own samples and nothing else, `gate_at` derived and written out:

- `tier-reference-cell/decode-warm-2nd-on`: three windows' second
  process, second request — 16.4, 16.4, 16.4 (spread 0) → value 16.4,
  gate 14.8. `decode-ratio-on-off`: 1.31, 1.38, 1.31 on the same pairs
  (spread 5.3 %) → value 1.31, gate 1.17. `decode-warm-2nd-off` (12.5,
  11.9, 12.5), both prefills (26.3; 86.4) and `grouped-fallbacks-on`
  (400): report-only, with the reasons above and in §8.1.
- `coder-served-large-decode/decode-warm-2nd`: 66.5, one sample, gated
  at §5's independent ≥ 60 t/s bar; its cold first request (66.6) and
  both prefills report-only.
- `coder-served-small-decode`: 47.6 warm, 46.0 cold, prefills ~510 —
  one sample each and no independent bar for this card: report-only
  until a second window.
- `depth-ladder`: the eight numbers above, one sample each: report-only.

Every reference names card, artifact, precision, lanes, n_ctx, the
device pool or its absence, the sized prompt, which process and which
request, the tree and the runtime. The §5.1 span and `docs/release-checklist.md` are regenerated
from them. From this commit on, a warm tier-ON decode below 14.8 t/s, a
ratio below 1.17 or a warm large-card decode below 60 t/s fails the
acceptance target unless named.

Production on the dev host was restored and verified after each window
(both units active, health 200); the 24 GB card's unit was stopped only
for its two cells.

#### 7.0.2am The turnstile tests synchronise instead of sleeping; the roundtrip flake did not reproduce under build load (2026-09-05)

The `turnstile-wall-time` campaign (`docs/campaigns/turnstile-wall-time.md`),
device-free, on the aarch64 build host: four cores, no card, the same host
class §5's Sanitizers paragraph runs UBSan on.

**The defect, shown red.** Two of `tests/test_turnstile.cpp`'s three cases
decided an assertion by a wall-clock sleep. The ordering case started the
third contender 30 ms after seeing the second's "started" flag; the wait
case released the held turn 50 ms after starting the other thread and
required a reported wait of at least 40 ms. Neither sleep observed
anything: both assumed the contender had reached `Turnstile::take()` —
where the ticket is issued under the mutex, and where the order is decided
— inside the window. With a 100 ms delay injected between the flag and
`take()` in the slowed thread, both cases failed three of three runs: the
order came out {3, 2}, and the waiter took an uncontended turn after the
release and reported zero.

**The rewrite.** `Turnstile` gains `issued()` beside `served()` — tickets
handed out, waiting ones included, `served() <= issued()` — test-only in
the same sense, nothing publishes it. The ordering case waits for
`issued()` to advance before it starts the next contender; the second
contender deliberately dawdles 20 ms before asking, which is the bad luck
the old form failed on. The wait case waits for the other's ticket, then
measures its own interval from that moment to just before the release and
requires the reported wait to be at least that: the waiter's clock starts
before its ticket and stops after the release, both orderings passing
through the turnstile's mutex, so the bound holds by construction under
any scheduling of either thread — given the one premise that
`steady_clock` is a single system-wide monotonic clock, which it is here
(`CLOCK_MONOTONIC`). Removing either
synchronising wait made its case fail ten of ten (mutation); fifty
consecutive runs green after. One draft of the wait case went red first
and is worth recording: the case takes a solo turn before the held one, so
the held turn is ticket 1 and `issued()` read 2 before the other thread had
asked — the wait anchors on the count observed after taking the held turn,
not on a literal.

**The gate.** `ctest -L unit --repeat until-fail:20` under a continuous
clean rebuild of this tree at `-j4` (308 and 289 s per build, load average
2.6–5.1 over 54 samples at 15 s, against 0.5 idle): twenty of twenty green
in 426.6 s. Then the `roundtrip` entry alone, forty more repeats under the
same load: forty of forty green in 356.6 s, about 9 s each against 5.5 s
idle. That is the reproduction attempt for the 0.3.1 window's one observed
roundtrip flake, whose load was a parallel build on this host and whose
log was not kept: sixty runs under matched load, not reproduced; it stays
on the record as observed once.

**The roundtrip sleep, the campaign's finding.** Of `tests/roundtrip.sh`'s
six sleeps, five are bounded polls for an observed event (four boot health
checks, the SIGTERM exit). One was not: the cancellation check slept a
fixed 0.5 s and then grepped the server log for the abort line — the
turnstile tests' class, a fixed window deciding an assertion. Measured
under the same build load, five runs, by the script's own clock
(`date +%s%N`) from curl's return after its 0.5 s cut to the first `grep`
that found the line: 11, 13, 13, 16 and 32 ms — the line was already there
at the poll's first look, and spawning that grep is most of the figure —
so the window had at least a fifteen-fold margin on this host and this is
*not* shown to be the flake's mechanism. It is
replaced anyway, by a bounded poll for the line itself (fifty steps of
0.1 s); red first with a pattern the server never logs — the check fails
after its bound instead of hanging — and green with the real one.

No served behaviour changed. `docs/design-0.3.1-test-ladder.md` §4's line
that kept the turnstile sleeps as "the one timing-dependent case" is
superseded by this section.

#### 7.0.2an The roundtrip flake reproduced on the dev host: two derived ports, never probed, collide with the previous runs' TIME-WAIT sockets (2026-09-05)

§7.0.2am closed the `turnstile-wall-time` campaign with the roundtrip
flake "not reproduced in sixty runs" on the aarch64 build host. The same
gate, run the same afternoon on the x86_64 dev container (eight cores, the
OpenVINO build tree, both production units running and untouched — the
unit set opens no device), reproduced it twice within the hour, and the
instrumented rerun measured the cause. That paragraph of §7.0.2am is
superseded here; the turnstile half of the campaign stands.

**The reproduction.** Load as in §7.0.2am: a continuous clean rebuild of a
stub-only tree at `-j8` (13–15 s per build; load average 0.4 rising to
7.6). `ctest -L unit --repeat until-fail:20`: the `unit` entry twenty of
twenty, `roundtrip` nineteen green and the twentieth red; `roundtrip`
alone, forty repeats asked: ten green, the eleventh red. Both reds the same
signature — the `--served-model-name` server ("alias") dies at start with
`could not bind 127.0.0.1:<port>`, its seven checks fail, everything else
passes.

**The cause, measured.** The script picked the main server's port by
asking the kernel (bind to port 0, read, close) and derived the other two
servers' ports from it: alias = main + 3, operator = main + 4 — never
probed. An instrumented copy dumped every socket on the neighbouring ports
before each derived server started, sixty runs back to back under the
same load: four failed (runs 34, 36, 53, 59), and in every one the alias
port was already held, before the alias server was even launched, by a
TIME-WAIT socket whose *local* port was the alias port — a curl source
port from an earlier run of the same loop. The chain is visible in the
dumps for two of the four: run 53's alias port (56394) is in run 34's list
of curl source ports to its main server, and run 59's (34618) is in run
53's. For runs 34 and 36 the holders' peers (44055, 41648) are earlier
runs' server ports whose logs were not kept, so the attribution there is
by the same pattern, not traced. A client socket's TIME-WAIT lasts 60 s.
The dumps show one more thing, measured: every one of the 342 TIME-WAIT
local ports in the five kept dumps is even, and every one of the ten
kernel picks is odd — on this kernel outgoing connections draw even
source ports and port-0 binds draw odd ones (the parity split is the
measurement; that it is the kernel's policy is a reading). So of the two
derived ports only main + 3 was ever exposed to curl's source ports;
main + 4 sits in the other class, which is why the operator server never
failed. That the port-0 pick itself never lands on a held port is a
reading as well; what was measured is that no pick collided in 240 runs
while the derived port did, four times in sixty. One further reading,
not measured here: a listener's address-reuse option does not override a
TIME-WAIT left by a client socket that set none, which is why the server's
bind fails rather than sharing.

So the trigger is run *density* inside the TIME-WAIT window, and the
build load was incidental. A reading with the section's own numbers: a
run issues about 35 curl connections (22 of them to the main server, the
ones the dumps list), the dev host's loop completed a run every ~2.5 s
(about 25 runs, ~850 held even ports inside any 60 s window), one derived
port sat in that parity class, and the class holds 14,116 ports — roughly
three to four expected failures in sixty runs; four were observed. The
aarch64 host, at ~9 s per run under its load, holds about a quarter of
that: about one expected in sixty, zero observed — consistent with the
rate, and not evidence against it, since a mean near one leaves a third
to a half chance of sixty clean runs. §7.0.2am's "not reproduced" was
therefore never evidence of absence. The 0.3.1 window's original flake —
one run, during the acceptance target, whose runners poll their servers
with curl — is consistent with the same mechanism; its log was not kept,
so that is a reading too.

**The fix.** Every server in `tests/roundtrip.sh` now picks its own port
with the same kernel probe (`free_port`); nothing is derived. Red first,
deterministic and device-free: a listener planted on main + 3 before the
alias server starts (a listener, not a TIME-WAIT — the same symptom from
a different holder) makes the old script fail as the dev host did
(`could not bind`, seven checks), and the fixed script passes with the
same listener in place. Green at the density that failed: the fixed
script ran 120 of 120 green on the dev host under the same load (load
average 1.7–8.5, a run every ~2.5 s, no `could not bind` in any log).

**What remains.** A probe-then-bind window still exists for every server,
the length of its start-up; a concurrent localhost connect landing on
exactly that port in that window would fail the same way. The container's
loopback traffic was sampled for 120 s at 0.5 s (every TCP socket, not
just the units' ports) to see who else connects there: nothing reached the
production units' ports at all — the unit manager that runs in the same
container reads systemd and the journal rather than polling the units'
HTTP endpoints — and the only loopback connections besides the
roundtrip's own were a handful to the unit manager's API, answering this
session's own queries. That is this container as configured today, not
a property of the script; and on this kernel such a connect would draw
an even port where the probe returns odd ones (a reading from the parity
split above), which narrows the window further. The belt, if it is ever
needed, is the server
binding port 0 itself and announcing the port, which removes the window;
not done, no served behaviour changed.

#### 7.0.2ao The Prüfstand cell runs from the dev host through the run manifest, and its score is a gated metric (2026-09-05)

The `pruefstand-cell-remote` campaign (`docs/campaigns/pruefstand-cell-remote.md`,
design note `docs/design-pruefstand-cell.md`), closed the same afternoon.

**What was actually there.** The acceptance target's one external cell
carried an environment variable, an invocation template naming a wrapper
that did not exist, a `score_parse` regex and a `reference_score` — and
`run.py`'s `run_external` read only the variable and the exit code. A
wrapper exiting 0 after printing `score: 3/10` passed the cell; the score
contract was inert. The harness itself is two steps on the session host
(a request script against an endpoint, a Lua scorer executing the answer
against ten RFC 4180 cases and printing `PUNKTE N/10`), and the variable
was the one card-requiring parameter still outside the run manifest that
`docs/design-0.3.1-test-ladder.md` §2 made the channel for everything else.

**The change.** The run manifest gains a `pruefstand` key from a new
configure-time cache variable (`ARCINT_ACCEPTANCE_PRUEFSTAND`, empty by
default; the value lives in `CMakeCache.txt` and the generated manifest,
never in a tracked file). The cell's `external` block names that key and
nothing else; the environment is not a channel any more, not even as a
fallback. The operator's wrapper prints one line, `ACCEPTANCE-METRIC score
<n> points`, and the cell's `references` gate `score` at 10 (`lower-is-
worse`), so §8.2's existing comparison does the work: 9 is REGRESSED, no
line is the missing-metric failure, a non-number is a hard fail.
`validate_cells` learned the external shape (a non-empty `manifest_key`, no
runner beside it), so a malformed external cell fails the device-free
`acceptance-enumeration` test. The cell also gained `timeout_seconds:
1800`. Red first, device-free: eight self-test assertions failed on the
old runner (the manifest key ignored, the environment honoured, no schema
check) and pass after; the self-test now runs 58 assertions.

**The real run.** From the dev host, `run.py --cell pruefstand` with the
manifest's key pointing at the operator's wrapper (outside the tree, a copy
of the harness's three files beside it), against the *deployed* coder
package on its own port — the installed `0.2.12-1+p3`, not this tree's
binary — with both production units running and no card touched: the
request took 9.9 s for 479 generated tokens (282 prompt tokens, greedy,
thinking off, 4,000-token cap), the scorer printed `PUNKTE 10/10`, the
runner compared `pruefstand/score: 10 points` against its reference and
exited 0. That is the first time the enumeration has seen the Prüfstand
result rather than a skip named on the command line. The reference's three
samples are that run and the two 0.3.0 release-candidate runs from the
session host (§7.0.2ai), which measured the release candidate rather than
the deployed package — the `binary` field says so.

**What the cell now measures, exactly.** The deployed package, whatever
version is installed, at the operator's endpoint: the cell's card class is
`deployed-package`, and the run says which version it hit. The
enumeration did not order it usefully — the cell sat before
`package-build`, and the install itself is outside every cell — so it now
comes last, after `package-build`; a release gate that wants the
*candidate* measured through it deploys the built package between the two,
or the cell's own first line names the older package, visibly.

#### 7.0.2ap The prefill fallback's weight-side answer is three-way; patch 0019, recipe at `+p5` (2026-09-05)

The `prefill-fallback-tristate` campaign
(`docs/campaigns/prefill-fallback-tristate.md`), closed on the item patch
0018's own header reported (§7.0.2ae).

**The defect, re-read on the tree.** In the plugin's per-expert prefill
loop, patch 0018 made the caller read `on_load_expert_weights()`'s
`false` as "not resident under the static partition: run this expert on
the host tier", and the host branch downcasts the weight provider to the
offload type. The same `false` was also the answer for "no offload tier at
all", where every expert's weights are on the device and the per-expert
handles are valid as initialised. Before 0018 the return was ignored and
the device path always ran; after it, a resident-only load reaching the
loop took the host branch for device weights through a downcast of the
wrong type. The loop is reached only through the dispatch's plain else —
both fast prefill paths off, two internal plugin properties arcint never
sets — so the branch was dormant, as 0018 said.

**The fix (patch 0019).** The answer is an enum with three values: no
offload tier, a device slot acquired or pinned, the host tier. The loop
takes the device path for the first two and the host branch for the
third, with an assertion that the provider is the offload one right
before the downcast.

**Red, then green.** A new plugin unit test drives a resident load of 40
tokens (above the batched-GEMV threshold) with both fast paths forced off
through the internal properties and checks the output against the
suite's own reference. On the 0018 tree it failed with "Can not open file
for mapping": the loop took the host branch and the misread provider
tried to map a weight file that does not exist — the misrouting, made
visible rather than silent. With 0019 it passes, and the four
static-partition cases and the sixteen smoke accuracy cases pass beside
it: 21 of 21 on the 24 GB card and 21 of 21 on the 16 GiB card, the
production units running beside the test binary throughout (a plugin unit
test's allocations are kilobytes), built and run in the staged 0018
source tree on the dev host, which now carries 0019. The provider-type
assertion sits above the host branch's blocking readbacks, so a wrong
provider fails before any device read. The new case's tolerance is the
suite's flat 0.15 plus 1 % of the reference — 40 tokens of u4 experts
reach |21|, where one f16 ULP is 0.016 and the first run missed the flat
bar by 0.006 on one element; a misrouted expert misses by tens. The
generated diff applies cleanly on the pristine pin with exactly 0003–0018
applied.

**What was not run, on the operator's word** ("a quick functional test,
not an hour of testing for thirty seconds of coding"): the three
acceptance cells the campaign named as the no-change proof. The branch is
on none of their paths. The `+p5` package is not built; the recipes are
bumped. Observed on the way and not investigated: the 0012-era sentinel
test (one token, batched-GEMV path, untouched by 0019) passes on the
24 GB card and fails on the 16 GiB card beside its resident service with
its first run's scratch read all zeros — pre-existing by path, not shown
so by a measurement on the unpatched tree.

#### 7.0.2aq The cold start's three owners, separated: the tier-ON load is the load-time probe forwards at tier speed, the first-request cliff is the disk, and kernel JIT is seven seconds (2026-09-05)

The `static-partition-cold-start` campaign's first window
(`docs/campaigns/static-partition-cold-start.md`, design note
`docs/design-static-partition-cold-start.md`), 16 GiB card, 35B int4,
ratio 50, 8 GiB device pool, u8 KV, one lane, n_ctx 65,536, prefix cache
off, chunk 2,048, the `+p4` runtime, the 1,167-token reference prompt, 64
greedy tokens, two requests per process, the plugin's counters on; the
dense agent's production unit serving on the other card throughout; the
host's ARC at its 40 GiB cap. Two earlier attempts of the same driver are
void — the first captured its server-starting function through a
command substitution and, read from the driver's logic rather than
measured, left each step's server running under the next; the second
failed its own prompt sizing — and both churned the host's file cache,
so this window's "as found" state is theirs, not the morning's.

**Step 1, answered without a card.** The compute runtime's on-disk
program cache is on by default in the container (5,019 files, 517 MB
against a 1 GiB cap) and grew when the plugin unit tests ran; the units
set no environment that changes it.

**The ladder.** Wall time from spawn to health 200; rates as the server
prints them; ARC misses from the host's kstat before and after each
process; the plugin's counters from the last `[OTD_PERF]` line.

| process | state | to health | request 1 prefill / decode t/s | request 2 | disk I/O in the plugin | ARC misses |
|---|---|---|---|---|---|---|
| P0 sizing, tier OFF | file cache cold after the churn (ARC 24 GB) | 264 s | 19.3 / — (957 tok), then 79.0 (1,167 tok) | — | 270 s cumulative, 278,775 tensor loads, 969 µs avg | +99,676 |
| P1 tier ON | file cached by P0 | 178 s | 23.2 / 12.4 | 26.2 / 16.6 | 5.0 s, 43,155 loads | +609 |
| R: `dd` of the 18.6 GB weight file | — | 16.2 s, 1.15 GB/s | — | — | — | +2,447 |
| P2 tier ON | file warm, runtime cache intact | 169 s | 25.5 / 15.6 | 26.3 / 16.3 | 2.3 s | +187 |
| P3 tier ON | file warm, runtime cache pointed at an empty directory | 176 s | 25.1 / 14.9 | 26.3 / 16.5 | 2.4 s | +297 |
| P4 tier ON | file warm, cache intact, a 222-token pre-warm request first (13.6 s) | 169 s | 26.0 / 16.5 | 26.3 / 16.4 | 2.4 s | +17 |

Every one of the eight outputs is the same 298 bytes
(`5d8e98890376923d`), the hash §7.0.2al's diagnostic rerun recorded for
all four of its tier-ON outputs (its "character 194 of 349" was the
ON-against-OFF comparison, 349 the longer of those two texts): the
static partition's history independence held across every cache state
here, and the §7.0.2ak divergence did not reappear.
`created_onednn_kernels` read 4,687 in each ladder process that ran the
full admission path with the tier on (4,734 with the tier off, 4,836 in
P4 with its pre-warm's extra shape, 4,346 with the plateau probe
bypassed) — not §7.0.2ai's 325, which was a different
counter read or a different tree; the figure is per process and
independent of the runtime cache, as oneDNN's in-process primitive
creation is by design. P3's empty runtime cache directory received 57
programs, 5.9 MB.

**What the deltas say.**

- *Kernel JIT (P3 − P2): 7 s of load, nothing on the requests.* The
  runtime cache is hit in every ordinary process and missing it costs
  seven seconds. Not the owner.
- *First-use fills (request 2 − request 1): 12.4 → 16.6 t/s decode in P1
  (file freshly cached), 15.6 → 16.3 in P2 (file warm), 16.5 → 16.4 in
  P4 after the pre-warm.* A residual of a few percent to a third,
  removed entirely by one 222-token request; not the recorded cliff.
- *The disk (P0): 270 s of cumulative read time inside the plugin, load
  264 s, the first prefill four times slower than the next.* The one term
  of the size the record described, and it belongs to the file cache: a
  40 GiB ARC serving two production models of 12 and 14 GB beside an
  18.6 GB test artifact cannot keep all three, and P0 read the artifact
  back from disk after the churn. Tier OFF pays it as much as tier ON;
  P1, one process later, read 5 s.
- *The tier-ON load is 169 s with every cache warm (P2, P4), 176 s with
  an empty runtime cache (P3) and 178 s with the file just cached (P1)*,
  and the compile inside it is 5.5 s (`paged model ready in 5.5 s`). What
  fills the rest is in the log's order: the expert-slot plateau probe (up
  to eight 128-token prefills of distinct tokens, until the device figure
  plateaus twice — it ran under the static partition too, `source:
  probe-static`, and settled at 0.11 GiB, which the next line calls
  "under 5 % of the host-side estimate") and the activation-fit ladder
  (128, 256, 512, 1,024, 2,048 tokens: 3,968 tokens; the logits-slice
  check inspects the ladder's own 128-token floor forward, not a further
  one). As an order-of-magnitude reading only: some 5,000 prefill tokens
  at the tier's served 25 t/s would be 200 s, more than the 163 s the
  warm load leaves after its compile, so the per-forward rates at 128 to
  2,048 tokens are higher than the served rate (D2's 24 s for the probe
  fits about five 128-token chunks, not eight). The direction is the
  point: the admission measurements are prefills, and tier-ON prefill
  runs at a third of tier OFF's (the `static-partition-prefill`
  campaign's own number), which is why the same path is the record's
  30–45 s with the tier off. The probe's share is measured by the discriminator below;
  the ladder's is the remainder after the compile, by the log's order.

**The discriminator**, two more tier-ON processes four minutes after
the ladder, the plateau probe bypassed in both by forcing its figure
(`ARCINT_FIT_SLOT_BYTES` at the probe's own 0.11 GiB; the log says
"Phase B probe and analytic walk skipped"). D1 reached health in 259 s
and served its first request at 10.2 / 8.0 t/s — slower than any ladder
process — and its counters say why: 180 s of cumulative disk reads
inside the plugin at 4.5 ms per load, where P2 had read 2.3 s at 54 µs.
The artifact had left the file cache again between the windows: the
ARC stood at 33.9 GB at the ladder's end and at 14.5 GB when read by
hand from the host's kstat after this pair (the discriminator's driver
sampled no ARC), the production unit serving on the other card in
between; the plugin's counter is the measurement, the shrinkage its
reading. So
D1 is not the probe bypass, it is a second sample of the disk term: a
load of 259 s and a first request at 0.4 of the warm rate, from reads
alone. D2, the same environment one process later with the artifact
re-cached by D1 (2.1 s of reads), reached health in 145 s and served
24.4 / 15.6: against P2's 169 s that is the plateau probe's share, 24 s.
`ARCINT_PREFILL_CHUNK_CAP` accepts only `off` (the belt switch), so D2's
"128" did nothing and the activation-fit ladder ran to 2,048 in both;
145 s less the 5.3 s compile is the ladder plus the logits check, about
140 s, by subtraction — not a per-forward timing, which the load path
does not log. Both outputs the same 298 bytes as every other.

**The owner, named.** Of the morning's 215–585 s tier-ON loads, ~170 s
is the load-time probe forwards running at tier prefill speed — a
constant, not a warming, and saveable: the figures those forwards
measure (the slot figure, the activation line) are functions of the
artifact, the device, the flags and the runtime, not of the process. The
rest is the disk term of a file cache that three models overrun, and it
slows the first request too — D1's 0.4 of the warm rate, from reads
alone; §7.0.2ai's 0.1–4.2 t/s decode did not reproduce in this window,
artifact cached or not. The disk term does not exist when the artifact
is cached and is the same with the tier off. The operator's
hypothesis — "save the heuristics and the cold start goes away" — is
right at the admission level: cache the fit ledger per (artifact, device,
flags, runtime) and the tier-ON load falls to the compile plus whatever
the ledger cannot vouch for. Kernel JIT is not worth a lever.

**Not closed here.** The lever is not built: a persisted fit ledger
changes the admission path (DESIGN §7.0.2a's terms, the refusal logic)
and needs its own red case (a ledger written by one process, read by the
next, refused when the artifact hash, device, flags or runtime differ).
The gate's cold metric in the tier cell is not emitted yet. Both are the
campaign's next step; the disk term is the host's memory and stays an
operational fact (or a pre-read of the artifact at service start, kept
out of the gate's timing).

#### 7.0.2ar The u8:i4 prefill price at a held chunk: +55 % and +90 %, in the infer wall and not in any counted kernel (2026-09-05)

The `u8i4-prefill-price` campaign's first window
(`docs/campaigns/u8i4-prefill-price.md`): 16 GiB card, coder int4, no
offload, one lane, prefix cache off, the `+p4` runtime, n_ctx sized to
the depth plus 2 % and 544 tokens, one fresh process per arm, one
32-token greedy request on a prompt sized by the runner (37,707 and
71,727 tokens), the coder unit stopped for the window and restored after
it. §7.0.2aa's +7/+25/+72 % were taken with each arm at its own auto-fit
chunk, before the belt; here the u8:i4 arm's belt picked chunk 128 at
both depths and the u8 arm was forced to the same 128, so the chunk is
held and the format is what differs.

| depth | u8:i4 prefill | u8 prefill, chunk 128 | u8:i4 time | infer wall ("graph") |
|---|---|---|---|---|
| 37,707 | 295.1 t/s, 127.8 s | 456.8 t/s, 82.6 s | +55 % | 127.2 s against 82.0 s |
| 71,727 | 209.3 t/s, 342.8 s | 397.9 t/s, 180.3 s | +90 % | 341.8 s against 179.3 s |

Neither arm logged a fault. The whole difference is inside the infer
wall; embeddings, page handling and restore are equal and small.

**Where the profiler puts it: nowhere it can see.** One process per arm
and depth with `ARCINT_PROFILE=128`, `ARCINT_PROFILE_SWEEP=128`,
`ARCINT_PROFILE_PAST=<depth>` — a synthetic prefill to the depth, then one
128-token chunk captured under `PERF_COUNT`:

| capture | u8:i4 node total | u8 node total | paged attention, u8:i4 | paged attention, u8 |
|---|---|---|---|---|
| chunk 128 at past 37,700 | 136.3 ms | 177.5 ms | 4.0 ms (10 nodes) | 46.7 ms |
| chunk 128 at past 71,700 | 139.1 ms | 218.9 ms | 7.2 ms | 88.2 ms |

Every other row is equal between the arms to within 1.5 ms (the 40
reference-kernel FullyConnected nodes at 65 ms, the 331 GEMM nodes at
35 ms, the GDN at 15.8 against 14.4 ms). The counter says the u8:i4 chunk is the
*cheaper* one, by 41 and 80 ms, while the served prefill says it is the
dearer one by 153 and 290 ms per chunk (the wall difference over the
295 and 561 chunks). So the price is not in any counted kernel's
execution, and §7.0.2ab's warning about this instrument holds a third
time: `PERF_COUNT` omits transfers and everything between kernels, and
that is where the price lives. The reading that the generic attention
kernel's *compute* is the cost (§7.0.2aa, "by code reading") is refuted
for every row the counter does attribute; the u8:i4 arm's attention row
itself is not attributed (below), so for attention that reading is
untested by this instrument, not refuted. What the counter cannot see —
the depth-scaling partial buffers the mixed stage allocates and merges
per chunk, the host work around them — is the candidate that remains,
and the discriminator below asks its growth directly. Both arms report the same
primitive implementation name for attention, so the profile does not
say which stage kernel ran inside it; the served rates do.

**Byte-identity does not extend to depth.** The two arms' 32-token
greedy continuations differ: at 37,707 tokens from character 126 of
182, at 71,727 from character 46. §7.0.2y's identity was measured at 16
tokens on short prompts; a four-bit value cache is lossier than an
eight-bit one. The reading is that at these depths the four-bit loss
reaches the argmax; not measured in this window (no quantisation-error
or margin figure, and no arm repeated at one depth). The campaign's
invariant is re-stated by it: a lever must leave u8:i4's output identical to u8:i4's
own before the lever, not to u8's. The u8:i4 arm produced the same 182
bytes at both depths; u8's two answers differ from each other.

**The values stay four-bit in VRAM** (the operator's constraint, 2026-09-
05, now in the campaign's invariants): the context gain is the format's
purpose, and a lever that materialises a u8 copy of the values for
prefill hands it back. Admissible levers dequantise in registers or in a
bounded, charged transient; the fast microkernel attention path taught
the packed value read is the one that fits. §7.0.2ab lists native
four-bit values in micro-SDPA as an upstream candidate, undecided; patch
0009's header put a kernel-source rewrite of the generic kernel over the
divergence bar for a memory lever.

**The discriminator: the partition bound.** The same 37,707-token
prompt, chunk 128, with `--paged-attention-max-partitions 32` (patch
0015's bounded partials: the mixed stage's partial buffers at a fixed
size, no per-chunk growth): u8:i4 prefilled at 276.4 t/s (136.4 s)
against 295.1 unbounded, u8 at 457.0 against 456.8, the bounded u8:i4
output byte-identical to the unbounded one. Bounding the scratch did not
move the price; it added the merge's own 7 %. So the partial buffers'
growth with depth — the second reading, from §7.0.2ab's era — does not
own it; their per-chunk reallocation, which the bound does not remove
(§7.0.2ab: pre-sizing removes the churn, not the size), is not tested by
it.

**What the profile cannot have seen, by arithmetic.** The artifact has
10 attention layers with 16 query heads of 256 (the load banner). Two
matmuls of 2 × 10 × 16 × 128 × 37,700 × 256 flops are 0.79 TFLOP, so
4.0 ms would be about 200 TFLOP/s — an order of magnitude above this
card — and the u8 rows calibrate the counter: 46.7 ms at past 37,700
and 88.2 ms at 71,700 (1.50 TFLOP) are both 17 TFLOP/s, the same rate
at both depths, as attention should scale. So the counter is not
reporting the u8:i4 arm's attention kernels, only some part of the
primitive (both arms name the same implementation, so which stage
kernels ran is not in the table). The instrument therefore does not attribute the u8:i4
price — the campaign's gate asks for a named kernel or stage at two
depths, and `PERF_COUNT` cannot name it for this arm, the class of
blindness §7.0.2ab recorded. Measured and standing: the price (+55 %,
+90 % at a held chunk, growing with depth), its location (the infer
wall, not embeddings or pages), and two refuted readings (counted kernel
compute; partial-buffer growth). Next instrument: a device-side timeline
of one chunk per arm (the OpenCL event trace §7.0.2ab used to calibrate
this counter), which sees every kernel launched whether or not the
profiler attributes it. Any lever remains microkernel work under the
four-bit invariant; the campaign is not closed by this window.

#### 7.0.2as The u8:i4 prefill price is removed: patch 0020 runs the mixed stage on micro-SDPA, parity with u8 at both depths, values still four-bit in VRAM (2026-09-05)

The `u8i4-prefill-price` campaign, closed by a lever the same evening
§7.0.2ar could not name the kernel for. The operator's call: "faster to
just write the kernel and check than to run hours of measurements to
find where the degradation lives" — and the check is the measurement.

**What the kernel is.** The value cache under u8:i4 was already read
in registers by the decode kernel (patch 0010); prefill's every chunk
after the first (the MIXED stage, past tokens from the cache) went to
the generic paged-attention kernel because patch 0009's selector
declined micro-SDPA for any key/value pair whose packing classes differ.
The microkernel path already knew four-bit values — upstream's symmetric
int4 KV cache (by-channel keys, packed values) runs on it — so the
change is the generator taking the *value* operand's type and layout
from the value precision (patch 0008's property) instead of the key's,
the kernel source gating each side's four-bit layout on its own macro
(five blocks), and the selector admitting eight-bit keys with four-bit
values. Nothing is copied or widened: the microkernel unpacks the
nibbles in registers, which is the constraint the campaign's invariants
state. One more thing the red case found: the kernel's value-pointer
advance across key chunks divided by an elements-per-byte constant
taken from the new-token input port, which is not four-bit under u8
keys while the cache is — every query whose causal context passed 128
keys came back NaN, and only those, until that constant was derived
from the value cache's own precision.

**Red, then green.** A new mixed-stage unit test (u8 keys by channel,
u4 values; 25 queries over a 34-token past, 25 over 128, 300 over 64;
the u4 micro regression's exactly-representable fill, 1e-2 against the
float reference; "sdpa_micro" asserted in the kernel dump) failed on
the 0019 tree with the dump naming the generic kernel, and passes with
the patch, three of three. Patch 0015's asymmetric prefill regressions
(128 new tokens over a 2,048-token past, random data) now run on
micro-SDPA and match; their assertion that micro-SDPA stays
declined is turned around, being the routing this patch changes. With
the u4 micro, asymmetric decode, bounded-partials, symmetric-u4 and
basic paged-attention suites: 276 of 277 pass, one pre-existing skip.
Four-bit values under BY_TOKEN keys are declined by the patch: the same
test with by-token keys gave NaN past 128 keys, not diagnosed, and not
what the plugin serves — its key mode defaults to by-channel and arcint
sets nothing.

**Served, on the recipe-built plugin** (the pin plus 0003–0020, built in
the package recipe's own tree and staged over the packaged runtime
layout), 16 GiB card, coder int4, no offload, one lane, chunk 128, the
price window's prompts, one fresh process per arm:

| depth | u8:i4, §7.0.2ar | u8:i4, patch 0020 | u8 |
|---|---|---|---|
| 37,707 tokens | 295.1 t/s, +55 % | 459.0 t/s (82.2 s) | 456.8 t/s (82.5 s) |
| 71,727 tokens | 209.3 t/s, +90 % | 401.0 t/s (178.9 s) | 397.5 t/s (180.4 s) |

Parity at both depths; the campaign's gate asked for ≤ +25 % at 71.7k.
The u8:i4 32-token greedy outputs are byte-identical to the generic
path's at both depths (§7.0.2ar's hashes); the Prüfstand through the
u8:i4 server at 37,707 tokens: 10/10, 479 tokens in 10.9 s. Decode, one
sample each: u8:i4 31.4 and 28.1 t/s at 32 tokens against u8's 33.3 and
30.8 — a report. No fault line in any arm. A first check on the debug
tree's build (which carries the two patches the recipe omits) had
already shown 462 against 462 t/s at 37,707 tokens; the recipe build is
the one on the record.

**One observation kept.** The u8 arm's 32-token answer at 37,707 tokens
on this build differs from the price window's u8 answer on the `+p4`
package (first difference at character 85 of 166), and the debug tree's
build gave a third answer on the same prompt (the 140 bytes the other
two builds give at 71,727 tokens), while at 71,727 all three agree byte
for byte. Two more fresh u8 processes on this build at 37,707 tokens
returned the same 32 tokens as this build's first (one hash, three
processes), so the u8 path is run-to-run deterministic here and the
difference is between builds — three builds, three answers, one prompt;
none of the three differs in a u8 kernel by reading. One sample per
build, not explained; a question for the release gate's own
equivalence cells, which compare within one build, and noted for the
next package's byte-identity check against `+p4` on this prompt.

**What this changes on the record.** §7.0.2ar's +55/+90 % at a held
chunk were the generic path's price (§7.0.2aa's +7/+25/+72 % carry the
chunk confound §7.0.2ar names); from `+p6` on the format's prefill is
u8's. `docs/model_requirements.md` §3's "u8:i4 is a
decode-only saving" verdict rested on the fit's charge for the generic
path's scratch (§7.0.2ab, §7.0.2ac) and on this price; the scratch term
is the fault campaign's and stays until it is re-measured on the
microkernel path, which allocates none of it. The campaign is closed;
the M8 row's owed item is closed with it.

#### 7.0.2at The scratch charge on the microkernel path: the generic kernel's partials are not allocated there, measured; the fit stops charging them from `+p6` (2026-09-05)

`u8i4-deep-prefill-fault`'s open question after §7.0.2as: the fit's
scratch term (§7.0.2ab, §7.0.2ac) prices three buffers the generic
paged-attention kernel allocates on the MIXED stage — `tmp_out`,
`exp_sums`, `max_logits`, sized by chunk × query heads × head size ×
element bytes × partitions — and patch 0020 moves the u8-key / 4-bit-
value pairing's MIXED stage off that kernel. Whether the buffers are
gone was a code reading (the plugin's `get_internal_buffer_descs`
allocates them only when micro-SDPA is not used, and the microkernel
path's own internal buffer is an index array of a few KiB), not a
measurement. This window measures it, and the `+p6` package is built.

**The measurement.** 16 GiB card, coder artifact, `--paged-kv u8:i4`,
chunk 128, prefix cache off, explicit n_ctx 73,678, the 71,727-token
price-window prompt, two requests per process (32 and 64 tokens), the
host's VRAM allocator sampled every 2 s through the kernel driver's
debugfs (`vram_mm`, the same counter §7.0.2ab used). Two runtimes, one
engine binary (the fit still charged 436.5 MiB on both loads): the
patch-0020 plugin staged over the `+p4` layout (the build §7.0.2as
served), then the `+p4` package's own plugin (generic kernel).

| plugin | idle after load | during the first prefill | during the second | consumed by the prefill |
|---|---|---|---|---|
| 0020 (micro-SDPA) | 1,821 MiB free | 1,812–1,821 MiB | 1,783–1,793 | **≤ 9 MiB** (first prefill) |
| `+p4` (generic) | 1,820 MiB free | 1,820 → **1,247 MiB** floor | 1,499–1,511 (held) | **573 MiB** |

The generic path consumes 573 MiB over the first prefill and holds it
(the plugin's intermediate pool grows to the largest shape seen and does
not shrink — §7.0.2ab's own observation); the microkernel path's free
VRAM does not move during the first prefill, and sits 28–38 MiB lower
during the second request (64 tokens against 32) — observed, not
explained, and two orders of magnitude under the generic path's
consumption. Outputs: both plugins produce the same 32-token
answer as the price window's u8:i4 output at this depth (one hash across
three builds of the kernel path), and the generic path's rate is the
§7.0.2ar price (209 t/s against the microkernel's 401 in §7.0.2as; this
window's own microkernel run was slowed by the package build sharing the
host, so its rate is not quoted). No fault line in either log.

A second reading, recorded and not explained: the generic path consumed
573 MiB where the fit charged 436.5 MiB (tmp_out at f16 288 MiB × 1.5
plus the two f32 partials, at 288 partitions). The 256 MiB margin covered
the difference at this depth. The proxy is a yardstick, not the plugin's
allocation arithmetic (§7.0.2ab said so); this is one more data point on
which side of it the yardstick errs at 71.7k, for the fault campaign.

**The fit.** From `+p6` on, a load with eight-bit keys and four-bit
values charges no scratch term and applies only the measured chunk cap
(128): `fit_context_packed_values` and `_at_depth` take a
`mixed_stage_on_micro` arm (fit.h), the belt's budget ladder does not run
there (it would halve the chunk for a buffer that no longer exists), and
the chunk ceiling and the Phase-E belt call site follow. The measurement
switch `ARCINT_PREFILL_CHUNK_CAP=off` still wins (no cap at all). The
detection is the GPU plugin's own build number, `marfrit-p<N>`, which the
recipe stamps and refuses to build without — 0020 adds no property
(0015's bound key was that patch's own contract); read through
`ov::Core::get_versions` on the plugin, not the core library, because a
staged runtime can pair one level's core with another's plugin (this
window's staged plugin reports `p5` and is priced as the generic path;
the `+p6` package's reports `p6`). The engine's gate is narrower than
the plugin's: 0020 admits any non-four-bit by-channel key with four-bit
values, so f16:i4 runs micro there too, but only u8:i4 is measured on
that path and f16:i4 keeps the charge for that reason; i4:i4 and by-token
keys are declined by 0020 and stay charged either way. Red first: the two arms' tests
failed on the arm-less signature (term still charged, chunk 32), 414 unit
cases green after.

**Functional check on the `+p6` package** (the operator's "quick
functional test, not an hour of testing"): the same card and artifact,
auto-fit, `--paged-kv u8:i4`, chunk 128, the package's libraries
extracted and put on the process's library path for the window (nothing
installed; the production units stopped and restored, as every window).
The fit admits **171,392** tokens (the reservation's ceiling 171,552,
trimmed by Phase E's page rounding) where the charged term admitted
101,824 (§7.0.2ab)
— the pool §7.0.2ab's fault was measured on, 171,312, with a prefill of
119,074 tokens at chunk 128 on the generic path. The
118,454-token prompt prefilled at chunk 128 in 346.9 s (341.5 t/s) and
decoded (21.3 t/s), no fault line in the log or the host's kernel log
grep; free VRAM sat at 971–978 MiB for the whole prefill — the idle level
after load — where §7.0.2ab's generic-path prefill of the same class on
the same pool size went to 0 MiB and faulted. One cell, one process; the
depth ladder on both cards at both precisions against the `+p6` package
is the fault campaign's own regression test and is still owed there.

**What this changes on the record.** `docs/model_requirements.md` §3's
"u8:i4 is a decode-only saving until the buffer stops scaling with depth"
is retired: on `+p6` the format's prefill is u8's (§7.0.2as) and its
auto-fit is the KV cost model's (+28 % over u8, §7.0.2y) with no scratch
term. The `+p6` package exists (built from the recipe, plugin stamped
`marfrit-p6`, not deployed; production stays at `+p3`). The fault
campaign keeps the belt and the charge for every plugin below `+p6` and
for the pairings 0020 does not admit, and its own gate (the fault
reproduced one variable at a time on the generic path) is unchanged.

#### 7.0.2au The depth ladder on `+p6`: green on both cards at both precisions; the measured chunk cap is now the whole of the u8:i4 prefill price (2026-09-05)

The debt §7.0.2at named: `depth-ladder` (tests/acceptance/cells.json, the
fault campaign's own regression test) against the `+p6` package's
runtime, both cards, both KV precisions, the engine at `d4dc137` (the
fit's microkernel arm, §7.0.2at) — so against the `+p4` references, which
were filled with the engine at `82cce71`, both the plugin level and the
engine moved. The package's libraries on the process's
library path for the window, nothing installed; both production units
stopped for it and restored after, health 200 on both. Coder int4, one
lane, no offload, prefix cache off, the cell's own sizing (98,187-token
prompt, n_ctx 100,653, one fresh process per cell, one request of 32
tokens; the first cell's process also carries the sizing rounds — its two
prefills read 1,024.3 then 1,025.6 t/s, 0.13 %, so the warm/cold
asymmetry against the other three cells is measured and negligible). The
chunk each cell ran is **by the code**, not by the log: the runner passes
no `--prefill-chunk`, so u8 runs the default 2,048 (`config.h`) and 4-bit
values the measured cap 128 (`kMaxMeasuredPackedValuesChunk`); the
load-time line that names the cap was written into each cell's server
log, which the runner deletes with its work directory — a runner defect,
fixed in the same commit (the cell now prints the load banner) and not
verified on a card until the next ladder. Lane count, offload and prefix
cache are the engine's defaults (one, none, off), by the same reading.
977 s wall for the four cells, from the window's driver, not the cell
log (840 s of it is the cells' own prefill and decode time).

| card | precision | prefill t/s | decode t/s | reference (`+p4`, §7.0.2al) |
|---|---|---|---|---|
| 24 GB | u8 | 1,025.6 | 45.7 | 1,025.5 / 20.8 |
| 24 GB | u8:i4 | **450.3** | 45.4 | 120.9 / 45.9 |
| 16 GiB | u8 | 620.9 | 29.3 | 621.0 / 29.2 |
| 16 GiB | u8:i4 | **365.3** | 26.4 | 170.3 / 26.1 |

Every cell served and no fault or out-of-resources line in any log; the
cell passes, its references are report-only and stay the `+p4` fill (the
release picks its level, and the references follow it then). u8 prefill
is unchanged within 0.1 t/s on both cards, as it must be: 0020 and the
fit arm touch only the u8:i4 path. u8 decode on the 24 GB card is not:
the `+p4` reference's 20.8 t/s against 45.7 here. The reference itself
recorded that sample as emit-dominated over its 32 tokens; this window's
decode line is graph-dominated (0.69 s graph, 0.00 s emit of 0.70 s), and
the u8:i4 cell on the same card read 45.9 then and 45.4 now. One sample
each, recorded, not explained.

**What the u8:i4 rows say.** 3.7× and 2.1× over the `+p4` references
at the same depth on the same cards — and still 2.3× and 1.7× under
u8. §7.0.2as measured parity *at a held chunk* (128 against 128); here
u8 runs the default 2,048 while u8:i4 is capped at 128 by
`kMaxMeasuredPackedValuesChunk`, the largest chunk any 4-bit-values
prefill was ever measured to pass, set when the generic path's buffers
were the fault. On the microkernel path those buffers are not allocated
(§7.0.2at), so the reading that fits is that what remains of the u8:i4
prefill price on `+p6` is the cap itself: sixteen chunks for every one u8
runs (768 against 48 at this depth), a chunk-count price rather than a
kernel price. The arithmetic on the record makes it plausible without
proving it: §7.0.2as measured u8 *at chunk 128* at 397.5 t/s against
u8:i4's 401.0 at 71.7k on this card, and this window's u8:i4 at 98k reads
365.3 — u8 held at 128 would land near it. That is consistent-with, not
measured; raising the cap is the measurement, not a decision: a
chunk ladder (256, 512, 1,024, 2,048) at u8:i4 on the microkernel path
with the VRAM sampler running, on the 16 GiB card first, at the fault
campaign's own depths. Until it is measured the cap stands; the fault
campaign carries the ladder as its next window.

**What this changes on the record.** The `+p6` runtime and the fit's
microkernel arm pass the release-gate cell that the fault campaign owns,
on both cards; the deployment decision for `+p6` has its acceptance
evidence. `docs/model_requirements.md` §3's owed depth ladder is paid.
§7.0.2as's "parity with u8" and §7.0.2at's "on `+p6` the format's prefill
is u8's" hold **at a held chunk of 128**, which is how they were measured;
in the default configuration the format still costs 2.3× and 1.7× of u8's
prefill time at 98k on the 24 GB and 16 GiB cards, and the cap is the
reading for why — narrowed here rather than left standing unqualified.

#### 7.0.2av The chunk ladder at u8:i4 on `+p6`: every rung to 2,048 serves 118k tokens without a fault on both cards; the microkernel path gets its own measured cap, and the package floor moves to `+p6` (2026-09-05)

§7.0.2au's next window, run the same evening: is the belt's measured
cap (128) — set when the generic kernel's buffers were the fault — still
needed on the microkernel path, or is it now the whole of the u8:i4
prefill price in the default configuration? The measurement that answers
it is a chunk ladder with the cap switched off.

**The ladder.** Coder int4, `--paged-kv u8:i4`, `+p6` runtime (plugin
stamped `p6`), engine at `d4dc137` + the ladder cell's banner change,
auto-fit, prefix cache off, `ARCINT_PREFILL_CHUNK_CAP=off` with an
explicit `--prefill-chunk`, one fresh process per rung, the 118,454-token
prompt of §7.0.2at (§7.0.2ab's fault depth class), one request of 16
tokens, the host's VRAM counter sampled every 2 s. 16 GiB card first,
every rung; then the 24 GB card at the two largest.

| card | chunk | served n_ctx (auto-fit) | prefill t/s | free VRAM during the prefill | fault |
|---|---|---|---|---|---|
| 16 GiB | 128 | 171,392 | 342.2 | 973 MiB, flat | none |
| 16 GiB | 256 | 167,760 | 432.4 | 968–973, flat | none |
| 16 GiB | 512 | 161,280 | 513.3 | 968–975, flat | none |
| 16 GiB | 1,024 | 148,176 | 562.7 | 968 floor, flat | none |
| 16 GiB | 2,048 | 124,896 | 592.4 | 962–970, flat | none |
| 24 GB | 2,048 | 262,144 | 780.0 | 10,094 → 7,832, then held | none |
| 24 GB | 1,024 | 262,144 | 723.3 | 8,089–8,103, flat | none |

Every rung served; no fault, out-of-resources, reset or timeout line in
any log. The served depth shrinks with the chunk on the 16 GiB card
because the activation term grows with it (0.02 GiB at 128, 0.40 at
2,048: the fit re-probed upward when its search asked for the bigger
chunk, and Phase E trimmed one pass on every 16 GiB rung — at 2,048
"overshoot 14.88 against a 14.86 GiB ceiling, correcting"); on the 24 GB
card every rung admits
the artifact's train maximum. Chunk 128's rate reproduces §7.0.2at's
(342.2 against 341.5). The 16-token outputs differ between chunks (128,
512 and the 256/1,024/2,048 group each hash differently) — chunk
boundaries move where the graph slices, the property the belt's own
comment states; equivalence is within a chunk, never across one.

Recorded, not explained: on every one of the seven rungs, on both
cards, the counter's free VRAM drops by 2.1–2.6 GiB within ±2 s of the
health mark (the 16 GiB rungs from ≈3,070 to ≈970 MiB, the 24 GB rungs
from ≈10,100–10,460 to ≈7,830–8,090) and is flat from then on through
the whole prefill; with a 2 s sampler the drop lands one sample before
or after the mark, which is why an earlier draft of this paragraph read
it as two behaviours. The engine's own residency audit reported
"deferred commit" for the 24 GB loads (the driver reporting 0.23–0.24 GiB
less than requested). The drop does not scale with the served pool
(1.07–1.47 GiB of KV on the 16 GiB rungs, 2.25 on the 24 GB ones), so it
is not read here as the pool's commit or as anything else; it precedes
the first chunk and is not what the belt exists for. The 24 GB card had
7.8 GiB free at its floor, the 16 GiB card 962 MiB.

**What changes.** The microkernel path gets its own cap,
`kMaxMeasuredPackedValuesChunkMicro = 2048` (fit.h): the largest chunk
measured on that path, the engine's default request, still a cap (nothing
above it is measured). The generic path keeps 128 — its cap was measured
against buffers that path still allocates. Every site the §7.0.2at arm
touched (both fit primitives, the ceiling, the seed, the Phase-E belt
site, the no-geometry branches, the load log) takes the path's cap. Red
first: the arm tests asked for 2,048 and 1,024 on the microkernel path
and failed on the 128 cap; 414 unit cases green after. The depth ladder
then ran again at the new default (`+p6`, both cards, both precisions,
the cell now printing each server's banner — its first run on a card):

| card | precision | served chunk | prefill t/s | decode t/s | at the 128 cap (§7.0.2au) |
|---|---|---|---|---|---|
| 24 GB | u8 | 2,048 | 1,025.0 | 21.3 | 1,025.6 / 45.7 |
| 24 GB | u8:i4 | 2,048 | **915.1** | 45.3 | 450.3 / 45.4 |
| 16 GiB | u8 | 1,024 | 620.9 | 16.5 | 620.9 / 29.3 |
| 16 GiB | u8:i4 | 2,048 | **665.0** | 26.4 | 365.3 / 26.4 |

Green, no fault line in any of the four logs, 825 s wall by the driver.
u8:i4 prefill at 98k is now 89 % of u8's on the 24 GB card and above it
on the 16 GiB card — where the banner shows u8 served at chunk 1,024 by
the fit's own activation ladder (its larger KV pool leaves less room at
this n_ctx) while u8:i4 got 2,048; not investigated further here. The
32-token decode figures move between runs on both cards at u8 (45.7 →
21.3 and 29.3 → 16.5 against unchanged u8:i4 figures), which puts
§7.0.2au's 20.8-against-45.7 remark in its place: that reference's
"emit-dominated" sample is one of two states a 32-token decode on a
fresh process lands in, and a decode reference at this cell's shape needs
more than one sample before it can gate anything. u8 prefill is unchanged
within 0.6 t/s across all three runs of the cell today.

**The package floor.** On the operator's decision the same evening — "move
the dependency to `+p6`; without it the mixed KV cache is possible but
pointless" — the arcint package's dependency floor moves from `+p4` to
`+p6` for 0.3.1 (`contrib/packaging/arcint/build-deb.sh`, the recipe's
changelog): below it u8:i4 prefills at +55 % to +90 % of u8's time
(§7.0.2ar) under a depth-scaled scratch charge (§7.0.2ab); at `+p6` it
prefills on micro-SDPA (§7.0.2as), the charge is gone (§7.0.2at), the
depth ladder is green on both cards (§7.0.2au) and the chunk cap is the
default's (this record). The `+p6` package is built and not yet deployed.

#### 7.0.2aw 0.3.1 tagged and deployed: the coder serves `+p6` at 10/10; the agent unit's pre-0.3.0 context is refused by the fit it never ran under (2026-09-05)

The tag `v0.3.1` (commit `0a16fa1`) closes the day: unit and acceptance
tests differentiated, patches 0019 and 0020, the fit's microkernel arm
and its 2,048 cap, the turnstile and round-trip fixes, the campaigns and
the two milestone records, and the runtime floor at `+p6` on the
operator's decision. Acceptance evidence for the tag, stated as what it
is: the depth ladder on both cards at both precisions against `+p6`
(§7.0.2au, §7.0.2av), the chunk ladder (§7.0.2av), the Prüfstand through
a u8:i4 server on the 0020 plugin (§7.0.2as) — and, on the operator's
word, no re-run of the byte-exactness, tier or concurrency cells for
this tag ("skip byte exactness for now").

**Deployment.** `marfrit-openvino +p6` installed over `+p3` on the dev
host, then `arcint 0.3.1-1` built from the tag's tarball by the recipe
(its unit gate five of five) over `0.2.12-1+p3`, both units restarted.
The restart fell into the host's scheduled backup, which held both
models' reads to about 12 MB/s for twenty minutes — the cold-start
record's disk term (§7.0.2aq) at its worst; no defect, noted because a
ten-minute silent load looks like a hang and is not one. The coder
(16 GiB card, u8, n_ctx 98,304, 2 GiB prefix cache) came up, reports
`0.3.1 (0a16fa1cdf6f)` on `/props`, and scores **10/10** on the
Prüfstand through the deployed endpoint.

The dense agent unit (24 GB card, `--n-ctx 155648 --mtp on --paged-kv
u8`, 8 GiB prefix cache) did not: its first process exited at the fit,
"requested n_ctx 155648 on 1 lane needs 5.38 GiB of KV but the
reservation admits 127536 per lane" — weights 13.59 GiB, drafters 3.16,
MTP state 0.97 (8.0 KiB/token), activations 0.03, margin 0.25, GDN rows
303 MiB, KV 36.2 KiB/token, of 22.71 GiB. That is the accounting 0.3.0
introduced and this unit never ran under: 0.2.12 served 155,648 here
without charging the MTP layer's KV state, which §7.0.2ag measured
overcommitting the card past 76k tokens. An explicit `--n-ctx` is
verify-only, never lowered (§7's M7 rule), so the refusal is the designed
answer, and systemd's on-failure restart then re-read the model every
ten minutes to refuse again; the unit is stopped. Three ways out, each
a served-behaviour change and so the operator's call, not this
record's: lower the context to what the fit admits (127,536 at these
flags); serve MTP off, the guidance §7.0.2ag already gives at depth on
this artifact, which frees the 0.97 GiB state and admits more; or serve
`--paged-kv u8:i4` on `+p6`, the format the floor exists for, which by
the KV cost model brings the pool under budget at 155,648 — unmeasured
on the dense artifact with MTP, so a window before it is served. The
coder's deployment stands either way.

#### 7.0.2ax The agent unit serves u8:i4 on `+p6`: 151,552 tokens with MTP on, and the MTP cycle wall at depth is now the unit's own number (2026-09-05)

The operator's answer to §7.0.2aw's three options was the third: serve
the dense agent with `--paged-kv u8:i4`, the format the `+p6` floor
exists for. What the fit then said, in order, each a fresh process:

| flags (24 GB card, dense 27B, MTP on, 8 GiB prefix cache) | KV KiB/token | activations | admits | requested |
|---|---|---|---|---|
| `u8`, default chunk (128 served) | 36.2 | 0.03 GiB | 127,536 | 155,648 — refused |
| `u8:i4`, default chunk (2,048 by §7.0.2av's cap) | 28.2 | 0.60 GiB | 139,104 | 155,648 — refused |
| `u8:i4 --prefill-chunk 512` | 28.2 | 0.15 GiB | 152,096 | 155,648 — refused |
| `u8:i4 --prefill-chunk 512 --n-ctx 151552` | 28.2 | 0.15 GiB | 152,096 | **served** |

Two things the table shows that the flags alone would not. The
microkernel path's 2,048 cap costs 0.57 GiB of activations on this
artifact (329 KiB per chunk token, probed), which is 20k tokens of u8:i4
KV — on a card this full, the chunk is a context lever, and 512 buys
13k tokens back. And the MTP state term (1.06–1.16 GiB at these depths)
plus the drafters' 3.16 GiB are what 0.2.12 never charged; §7.0.2ag's
measured overcommit is why they are charged now, and 155,648 with MTP on
does not fit the card under honest accounting at any chunk. The unit
serves 151,552 (the fit's 152,096 less the prefix cache's spare pages,
2,691 of them kept — Phase E's own correction), 2.6 % under the
configured 155,648 it ran at before. Both unit edits are committed in
the unit repository; the unit manager could not edit `--paged-kv` or
`--prefill-chunk` (not fields it knows), so this was a hand edit, per
the operator-local rule.

**Functional check through the deployed endpoint.** `/props` reports
`0.3.1 (0a16fa1cdf6f)`, MTP enabled. A 40-token request: 120 tokens at
30.5 t/s, draft acceptance 68 %. A 71,727-token request (the price
window's prompt): prefill 377.5 t/s at chunk 512 (167 s of graph in
190 s), then 64 tokens at **2.2 t/s** — verify 24.0 s of the 28.5 s,
acceptance 64 %, no fault or out-of-resources line. That decode is the
`mtp-cycle-wall` campaign's own defect on the served unit: §7.0.2ag
measured MTP at 4.9 t/s against plain's 15.3 at 77k on this artifact at
u8, and the guidance there is MTP off at depth; whether u8:i4 or the
chunk accounts for 2.2 against 4.9 is not measured here (one request,
two variables). The unit serves; its deep-context decode is the
campaign's number, not this deployment's, and turning MTP off would
also hand back the 1.16 GiB state and most of the 3.16 GiB of drafters
— the operator's next call, recorded with its numbers.

#### 7.0.2ay 0.4.0 stage 1: a GGUF opens in process on the served IR and serves at 10/10, its K-quant rows decoded in the kernel; the first kernel's rates are the price (2026-09-06)

The 0.4.0 charter's first serve (`docs/design-gguf-native.md`,
`docs/milestone-0.4.0.md`): the dense Qwen3.8-27B at Unsloth's Q4_K_M,
opened by `--gguf FILE --model <the dense IR directory>` on the 24 GB
card, plugin patch 0021 (`+p7`), the operator's rule in force — no
unpack at load, no reorder at compile, the only bytes on the card are
the file's.

**What the open does, measured on this file.** 866 tensors: Q4_K ×294,
Q6_K ×67, Q5_K ×48, Q8_0 ×1, F32 ×456. The pass replaced 497
projections in the template (Q4_K ×288, Q6_K ×65, Q5_K ×48, and the
dense file's F32 GDN alpha/beta ×96), 15.2 GB of rows, in 55 s of
compile; device-resident 14.94 GiB; health at 90 s. Kept from the
template: the embedding (i8 per row; the file's Q4_K needs the gather
kernel, a later patch), the norms, the GDN state tensors, the MTP layer.
The converter's value-head reorder was inverted on five tensors per GDN
layer (rows on four, the activation's columns on the output projection);
`tools/gguf_ir_compare.py` checks the inverse on the host — cosine
between the file's and the template's dequantized weights 0.04–0.43 as
stored, 0.993–0.996 un-reordered, on every one of them, layer 0 and 3.
The tokenizer is the template's, measured identical to the file's
(§milestone 2026-09-06); the chat template is the template's, the
file's differs (logged).

**The first serve scored 0/10 and the cause was the template's AWQ.**
The dense IR is AWQ-quantized, and the exporter left sixteen
activation-side multipliers in the graph, one per attention layer on
the gate path (`awq_mul/scale`, values 1 to 72). With the projections
replaced by the file's raw rows and those multipliers left standing,
the model stayed fluent and its Prüfstand answer looped (0/10, every
case a timeout or a wrong split). The compare tool had the signature
before the answer did: against the template's dequantized weights the
file's `o_proj` sits at cosine 0.95, `up_proj` at 0.50 and 0.81 (layers
0 and 3), `down_proj` at 0.95 and 0.97, while every other projection
reads 0.993–0.996 — consistent with AWQ's fold (the output projection's
input scaled through those multipliers, the down projection's input
through the up projection's rows), inferred from the cosines and not
measured further. The pass now sets every `awq_mul/scale` constant to
one (16 on this template) and compares every norm the file also carries
with the template's (209 constants, max |diff| 0 — the exporter did not
fold anything into the norms). With that: **Prüfstand 10/10 through
the GGUF-opened model**, all ten cases, the same harness and scorer as
every other 10/10 on the record.

**Rates: the kernel ladder, and the benchmark against Intel's own int4
export.** The first kernel served at 28.8 t/s prefill and 3.5 t/s
decode (24 GB card, `u8` KV, one lane, MTP off, prefix cache off, n_ctx
32,768, chunk 256, the plugin from the dev tree at patch 0021 staged
over the `+p6` layout) — one output column per work-item over eight
activation rows, no subgroup cooperation. Eight versions later the
same file serves at the rates in the table below. The ladder was
climbed on one shape, the dense model's gate projection (N 17,408, K
5,120, Q4_K, 50 MB of rows), with a timing test in the plugin's suite
(`DISABLED_gate_proj_shape_q4k_rows_1_to_2048`, one launch, ten
repetitions, the network's own stream waited on — the first timing
waited on the test stream and reported 1.1 TB/s on a 450 GB/s card,
retracted before it was written down). Each rung was a measurement,
not a guess:

| kernel (24 GB card, one launch) | M = 1 | M = 32 | M = 2048 |
|---|---|---|---|
| one subgroup per column, lanes over K, byte decode (the first) | 523 µs (16 GB card) | 9.4 ms | — |
| lane per column, uniform activation loads, word decode | 234 µs | 7.4 ms | 298 ms |
| + XMX for the tile, A operand read from global memory | 234 µs | 5.9 ms | 298 ms |
| + A tile staged in local memory per work-group (served) | 225 µs | 0.95 ms | 39.5 ms |

What each rung found. The compiled first kernel reported
`private_size 256`: a run-time loop bound over the per-row accumulators
sent them to scratch memory (the compiler's dump, `.zeinfo`; fixed by
compile-time bounds with clamped rows). The per-row cost then stayed at
about 230 µs through two rewrites of the inner product, because the
cost was never the arithmetic: with one subgroup per column, every
column re-read the activation row (178 MB per row per launch out of
cache). One lane per column with the activation address uniform over
the subgroup fixed decode (one broadcast read) but not the tile, and
the matrix-multiply instructions (`intel_sub_group_f16_f16_matrix_
mad_k16`, the decoded sub-block as the lane's column of B) did not
move the tile either — until an experiment that dropped the A-operand
loads alone brought the 2048-row launch from 298 ms to 11 ms. The
activation tile was being re-read once per 16 columns, 22.8 GB per
launch, at what the cache hierarchy gives for 32-byte reads scattered
over 32 rows. Staging each super-block's tile (32 rows × 256 values) in
local memory once per work-group of eight subgroups cut that by eight
and the launch to 39.5 ms. Decode sits at 222 GB/s of rows against the
card's ~450: the decoders' instruction count, not bandwidth (the
word-based decoders halved the first version's; the rest is a later
patch). Correctness on every rung: the eleven plugin cases on both
cards against the host reference (the tiled variant's tolerance widened
by 2^-11 of Σ|x·w| for its f16 weight copies), then the Prüfstand.

**The benchmark the operator asked for**, same card, same flags, one
fresh process per cell, the file through the K-quant kernel against
Intel's own int4 IR export of the same model (`qwen38-intel-int4-ov`,
allowlisted for the comparison), prefill plus a 64-token decode at two
depths; `u8` KV, one lane, MTP off, prefix cache off; the chunk is the
fit's choice per arm and is part of the result:

| arm | prompt | chunk | prefill | decode (64 tok) | resident |
|---|---|---|---|---|---|
| GGUF Q4_K_M, K-quant kernel | 856 | 256 | 213.0 t/s | 9.9 t/s | 14.94 GiB |
| GGUF Q4_K_M, K-quant kernel | 71,727 | 256 | 173.8 t/s | 8.5 t/s | 14.94 GiB |
| Intel int4 IR (`qwen38-intel-int4-ov`) | 856 | 2048 | 1,609.4 t/s | 23.1 t/s | 13.06 GiB |
| Intel int4 IR (`qwen38-intel-int4-ov`) | 71,727 | 2048 | 551.7 t/s | 16.5 t/s | 13.06 GiB |

Prüfstand through the GGUF-opened model on this kernel: 10/10 (the
gate before the benchmark; 182.7 t/s prefill and 12.1 t/s decode on
its 282-token / 1,079-token exchange). Read the table as it is: the
K-quant path is 3.2× to 7.6× slower at prefill and about 2× at decode, and
it is the same model at 1.9 GiB more resident (the template's i8
embedding and the file's Q6_K rows against Intel's int4 throughout). Two known contributors, both recorded and
neither priced apart: the GGUF arm prefills at chunk 256 because the
fit charges 2.65 GiB of activations for it (the kernel's f32 outputs
and the fused ops around them) where the IR's path charges a fraction,
and the decode kernel runs at half the card's bandwidth. No fault line
in any cell.

**What was built.** Stage 0: the reader (`src/core/gguf.*`), ggml's
dequantizers as the host reference (`src/core/gguf_dequant.*`), the
fixture with gguf-py's own decoding beside it, exact-equality tests.
Stage 1: the tensor map and the V-head inverse (`src/core/gguf_map.*`,
device-free tests), the op the engine builds (`src/exec/kquant_op.h`),
the template pass (`src/exec/gguf_graph.*`: memory-mapped template
read, decompression chains replaced by tagged u8 constants aliasing the
file's map and owning the file with them, the value-head inverse as a
gather on the projection's output — no row is copied — or on the
activation for the output projection, the AWQ neutralisation, the norm
comparison, a report),
`--gguf`, and plugin patch 0021: the primitive's three fields, shape
inference over `[N, K]` for a `[N, row_bytes]` memory with the refusal
on the main thread, the kernel and its decoders, seven correctness
cases on the 16 GiB card against a host reference, the
fully-connected suite otherwise unchanged (1,161 run, 1,070 passed, the
rest skipped, as before plus the eight). Two defects found and fixed on
the way: the impl re-derived the weights' shape as `[bytes / K, K]`
(27 rows of 512 for 48 of 288), and a missing kernel terminated the
process from a worker thread instead of refusing.

**Retracted from an earlier draft of this record** (§7.0.1): it said
the kernel's decoders were "one source, compiled as OpenCL C in the
plugin and as C++ in the host test, so the two cannot drift". No such
host compile exists; the kernel's reference in the plugin's tests is a
second, independent transcription of ggml's formulas (`kq_host`), and
the decoders are further checked end to end by the 10/10. The shared-
source host compile is owed. The first review of this change also
found the aliasing constants owned nothing — the file was kept alive
only by a runtime-info entry on a model the load discards after
compile — which held only because compile copies every constant to the
device; the constants now own the file through the constructor made
for that.

**Still open in stage 1.** The embedding from the file (gather kernel);
the MTP layer from the file; the plugin-below-+p7 refusal message
(untested: the op is simply unknown there); the rates' two named
contributors — the decode kernel at half the card's bandwidth (the
decoders' instruction count) and the prefill tile's activation reads
(2-D block loads, Xe2 only, are the next rung) — and the activation
reservation that holds the GGUF arm at chunk 256. The device-free test
of the pass on a toy template exists (`tests/test_gguf_graph.cpp`,
OpenVINO-gated: four projections, the AWQ constant, a norm; it checks
the replacement, the aliasing, the gathers, the neutralisation, the
refusal).

#### 7.0.2az 0.4.1 lever 1: the decode kernel on the matrix unit is a win on Xe-HPG and a loss on Xe2; the served rate on the 24 GB card did not move (2026-09-06)

The first lever of `docs/milestone-0.4.1.md`, taken with the plugin's
timing test (DESIGN §7.0.2ay's ladder harness, extended to the value and
down projections) and closed with the served benchmark. Plugin patch
0022 (`+p8` recipe, package not built) is what came out.

**The idea and what it did.** Decoding a K-quant value was a shift, a
mask, a convert and a multiply-add per value on the vector unit. The
matrix unit takes f16 operands, and every quantised integer q' below
1,024 *is* an f16 bit pattern: `0x6400 | q'` = f16(1024 + q'). So a lane
packs its 16 values with a shift, a mask and an or, runs the one-row
subgroup matrix multiply against the activations (and the same multiply
against f16 1.0 for Σx), and applies the block's scale and offset to the
two sums: Σ x·value = dl·S − mo·X. The 1,024× offset costs seven bits of
the f32 accumulator, measured within the test tolerance at K 1,280.
Correct on both cards (12/12 against the host reference). Rates, one
launch at the gate projection (N 17,408, K 5,120, Q4_K), one row:

| decode kernel, M = 1 | 24 GB card (Xe2) | 16 GiB card (Xe-HPG) |
|---|---|---|
| 0021 as served (fused multiply-add) | 225 µs | 509 µs |
| packed, one-row matrix multiply (v9) | 284 µs | 170 µs |
| 0022: matrix multiply on Xe-HPG, fused path on Xe2, split by width | 199–230 µs | 179 µs |

**Why Xe2 lost, measured by ablation** (each row removes one thing from
the v9 kernel; the differences overlap): the weights-only skeleton 148
µs (338 GB/s of rows; reading them lane-contiguous, 138 µs — the file's
layout is not the floor); the activation block reads add about 106 µs
(plain per-lane loads recover 35 of them, local memory 8); the four
one-row multiplies per sub-block add about 84 µs — 46 cycles each, on
Xe2 a one-row `dpas` occupies the systolic array like an eight-row one,
three times a lane's FMAs for the same 16 values; the packing adds
about 91 µs on top of those and 14 µs alone. On Xe-HPG the multiply is
cheap (removing it saved 26 of 181 µs) and the packing replaced 300 µs
of byte-wise decode. Hence the split by architecture. Readings refuted
by one timing each: prefetching the next super-block, a K split of
eight, a two-way unroll (all worse, or −6 %); a sub-block-granular K
split (1.6× slower on both cards: the unrolled eight-sub-block body is
what lets the compiler hoist a super-block's loads).

**The split by width.** With four subgroups per column fixed, the value
projection (N 1,024) launched 256 subgroups on a card with 1,280
thread slots; the decode work-group is now sized so a launch has about
4,096 subgroups on Xe2 (8,192 on Xe-HPG: eight per column of the widest
projection ran slower on Xe2 and faster on Xe-HPG, measured) within the
work-group limit and the row's super-block count. Down projection (N
5,120, K 17,408): 241 µs on the 24 GB card, 188 on the 16 GiB card;
value projection: 27 µs on both (a launch's fixed cost; 100 GB/s).

**Served, the same protocol as §7.0.2ay's benchmark** (24 GB card, `u8`
KV, one lane, MTP off, prefix cache off, chunk 256): decode 10.0 t/s at
856 prompt tokens against 9.9 on 0021, prefill unchanged at 212.6 t/s;
at 71,727 tokens 7.4 t/s against 8.5 — a difference this patch has no
mechanism for (the kernel does not see the depth), recorded and not
attributed. The profile of the decode step on the deployed runtime
(`ARCINT_PROFILE`, shares only) puts the K-quant kernel at 93.3 % of
node time; nothing else in the step is worth a lever. The window's
run-to-run spread on the 24 GB card is about 15 % for short launches
(the same binary timed the gate projection at 230 and 199 µs in two
consecutive windows); the 16 GiB card's spread is small.

**What this says about the milestone.** The 24 GB card's decode launch
sits at 200–230 µs against a 148 µs skeleton and a ~110 µs byte floor,
and no lever tried moved it: the arithmetic is not hidden behind the
stream and the matrix unit does not help one row. The gate (within 1.2×
of the IR's decode) is not in reach on this path with what is measured
here; the operator's question of a repack at load into the plugin's own
compressed layout — the layout Intel's IR path runs at 355 GB/s effective
— is the next decision, recorded in the milestone.

#### 7.0.2ba 0.4.1 lever 2: the repack at load — the K-quant rows in the plugin's own compressed form, the mins as columns, at the IR's prefill rate and 10/10 (2026-09-06)

The operator's decision after §7.0.2az ("repack it is then, but make
sure quality does not degrade — make an equivalent projection; if that
is impossible, we need to go back to the drawing board"). What was
built, what "equivalent" turned out to mean, what was measured, and the
one detour that was refuted on the way. `docs/design-gguf-native.md`
§3.6 is the design; `src/core/gguf_repack.*` the host pass;
`--gguf-native` keeps the 0.4.0 path.

**Equivalence, defined by arithmetic and then measured.** No form the
plugin computes reproduces ggml's f32 dequantizer bit for bit: a
K-quant scale d·sc is an f16 times a 6-bit integer (17 bits) and rounds
to f16, and the plugin's kernels compute q·scale in half, which rounds
every value at 2^-11 relative. So the gate is a bound on the deviation
of every weight from ggml's value, in units of the group's quantisation
step, the analytic worst case of that arithmetic per block type: 1/64
for Q4_K, 1/32 for Q5_K and Q6_K, 1/16 for Q8_0 (|q| up to 127). The
stored integers, Q8_0's scales and the mins are the block's own. The
fixture measures 0.013, 0.028, 0.029 and 0.051 steps at most and
0.003–0.011 RMS; the served file measures 0.0294 steps at most over its
25.6 G values, none over bound, at every load (a tensor over its bound
refuses the load). For scale: the quantisation itself moves a weight by
up to half a step, and the native path's own tiled kernel already
rounds its f16 copies at the same 2^-11 (§7.0.2ay). The served
computation on the host — the widened activation, the multiply-
accumulate — lands within 5 % of its f16 rounding budget of the f32
reference for every type (`tests/test_gguf_repack.cpp`).

**The mins ride as columns, because the fast path takes integer zero
points only.** The first repack carried Q4_K's and Q5_K's mins as an
f16 zero point per group (ml/dl). Legal IR, accepted by the plugin,
10/10 on the Prüfstand, and its 1k greedy output was byte-identical to
the native path's — but oneDNN declines an f16 zero point (its weight
zero points are u8/s8/u4/s4) and the OCL kernels took the 336
projections: 15.9 t/s decode, 98 t/s prefill, and the fit pinned at
chunk 128 because the probe measured 11 MB of activations per chunk
token (`activation fit … 11558.5 KiB per chunk token`, 1.41 GiB at
128, max context 42k). Two readings for that slope, each settled by one
probe: the 240 head-order gathers the pass adds (moved into the
repacked rows and column groups at load; the slope stayed at 11 MB),
and the plugin's dynamic activation quantization on that path (off:
279 KiB per token, chunk 2048 — but prefill 112 t/s on those kernels).
The zero point itself was the cause: dropped, the same projections went
to oneDNN at 298 KiB per token. So the min term is now exact columns of
the same tensor — for every group the integer mn, two u4 nibbles under
the super-block's own f16 dmin as the group scale — and the activation
is widened by its group sums (−16·Σx_g, −Σx_g) through one reduce, one
small matmul and one concat per distinct activation, shared by every
projection reading it; a column-reordered projection takes its order at
build and its augmented groups follow the heads. One plain fully-
connected with no zero point, oneDNN's path throughout.

**Measured, the augmented form** (24 GB card, `u8` KV, one lane, MTP
off, prefix cache off, one fresh process per cell, the deployed `+p7`
runtime; the plugin's dynamic activation quantization on, the IR's
setting):

| arm | prompt | chunk | prefill | decode (64 tok) | resident | max ctx |
|---|---|---|---|---|---|---|
| native rows, K-quant kernel (0.4.0) | 856 | 256 | 213.0 t/s | 9.9 t/s | 14.94 GiB | 100,176 |
| repacked, f16 zero point (the first form) | 856 | 128 | 98.3 t/s | 15.7 t/s | 18.18 GiB | 42,368 |
| **repacked, mins as columns (served)** | 856 | 2048 | **1,005.2 t/s** | **16.1 t/s** | 18.73 GiB | 46,368 |
| Intel int4 IR | 856 | 2048 | 1,609.4 t/s | 23.1 t/s | 13.06 GiB | 212,944 |
| repacked, mins as columns, f16 activations (**the default**) | 856 | 2048 | 939.5 t/s | 16.2 t/s | 18.73 GiB | 46,368 |
| repacked, mins as columns, **`u8:i4` KV** (the only KV that fits 71.7k beside 18.73 GiB) | 71,727 | 512 | 420.0 t/s | 13.4 t/s | 18.73 GiB | 80,016 |
| native rows (0.4.0), `u8` | 71,727 | 256 | 173.8 t/s | 8.5 t/s | 14.94 GiB | 100,176 |
| Intel int4 IR, `u8` | 71,727 | 2048 | 551.7 t/s | 16.5 t/s | 13.06 GiB | 212,944 |

Prüfstand through the repacked model: 10/10 under the served default
(f16 activations: 425.8 t/s prefill and 17.3 t/s decode on its 282-
token / 1,559-token exchange; 10/10 with int8 activations as well, at
646 t/s and 17.4 t/s). Load: 406–427 s, of
which the repack and its exhaustive deviation check are most (the
native open: 90–150 s). Resident: 18.73 GiB against 14.94 native — Q6_K
at u8 with an f16 scale per 16 is 9 bits per weight against 6.56, Q5_K
at u8 with its augmented columns 10 against 5.5, Q4_K 5.1 against 4.5 —
and with it the context at `u8` KV: 46k on this card, so the 71.7k cell
runs at `u8:i4` and says so.

**The greedy outputs.** The f16-zero-point form reproduced the native
path's 1k output byte for byte (sha 23e06c37e0d6, 346 characters). The
augmented form does not: its output agrees for the first 146
characters and then takes the other branch of a near-tie ("… the
mathematical nature of *the operations*" against "*how information*").
The same fork appears between two native runs at 71.7k with different
kernel versions (§7.0.2az: 5e3fb7a8d72c against 48c1b1b06c17), so it is
a property of the token, not of the projection. The one arithmetic
difference between the two repacked forms is oneDNN's per-token int8
activation quantization, the plugin's default for compressed weights
and the setting Intel's IR arm runs; with it off (`--dyn-quant off`,
f16 activations through the same oneDNN path) the augmented form
reproduces the native output byte for byte again (sha 23e06c37e0d6) at
939.5 t/s prefill and 16.2 t/s decode — the same rate. So f16
activations are the default for a GGUF-opened model: the projection is
equivalent to the bound above, and the served greedy output is the
native path's, byte for byte, at 856 tokens on this prompt. Int8
activations remain a flag (`--dyn-quant on`), the IR's own setting.

**What this closes and what it opens.** Lever 3 of the milestone (the
activation reservation at chunk 256) closed with the zero point: it was
never the K-quant kernel's f32 outputs. Prefill is at 62 % of the IR's
at 856 tokens; decode at 70 %, which is the resident bytes (18.7
against 13.1 GiB at about the same effective bandwidth). Open: the
resident size (Q6_K and Q5_K cost 37 % and 80 % more than their native
rows; a mixed open — those two types native, Q4_K repacked — is a flag
away and trades prefill for context), the load time (an exhaustive
check at every load), and the deferred stage-1 items.

#### 7.0.2bb 0.4.1, the native path's decode kernel on the integer dot: faster in isolation, 8/10 on the Prüfstand, and the same served rate; the timing instrument corrected (2026-09-06)

The operator's next window after §7.0.2ba ("start the work on 0021"):
the K-quant kernel's decode variant rebuilt the way llama.cpp's CUDA
path does it, and what the measurements said about that, about the
instrument, and about where the served decode step's time is.

**What was built.** Activations quantised per 32-block to int8 with an
f32 scale and the block's sum (q8_1's scheme), the weights unpacked
byte-parallel on dwords — Q4_K two operations per four values, Q5_K
six, Q6_K eleven with the −32 folded in and no min, Q8_0 two — and the
4×8-bit integer dot (`dot_acc_sat_4x8packed_*_int`, a native
instruction on Xe) accumulating in int32, the block's scale and min
applied once per sub-block. Four steps on the 24 GB card, the gate
projection at one row under the timing test as it then was:

| decode kernel, M = 1, 24 GB card | gate projection | down projection (K 17,408) |
|---|---|---|
| 0022 as shipped (f32 fused multiply-add) | 199–230 µs | 241 µs |
| int8 dot, activations quantised into local memory by the work-group | 230 µs | 291 µs |
| + activation dwords read once per super-block, broadcast from registers | 214 µs | 293 µs |
| + no prologue: each super-block quantised in registers inside the weight loop | 194 µs | 253 µs |
| + the decode work-group capped at eight subgroups | 196 µs | 199 µs |

Every step was correct on both cards (12/12 against the host reference
at a tolerance widened by 2^-7 of Σ|x·w| for the int8 activations). The
instruction count of the super-block body fell from about 1,900 (0021)
through 1,080 (§7.0.2az) to about 600 here, and the time barely moved
until the prologue went: an ablation that skipped the quantisation
prologue alone was worth 46 µs of 214, and neither a two-way unroll nor
an eight-way K split moved anything. The prologue's cost was its own
chain of gather loads and local-memory stores before any weight moved,
whatever it was split over.

**Served, and the quality gate.** The dense Q4_K_M through
`--gguf-native` on this kernel, 24 GB card, `u8` KV: decode 10.1 t/s at
856 prompt tokens (0.4.0's kernel: 9.9), 8.5 at 71.7k (8.5), the 1k
greedy output byte-identical to 0.4.0's — and **the Prüfstand at 8/10**,
the first score below ten on this model on any path: two cases wrong
("einfach CRLF" and "nur LF", both with a duplicated last field in the
parsed rows), the same generation that scored 10/10 on the f32 kernel
and on the repack. Int8 activations per 32 are llama.cpp's precision;
on this model and this task they are not this repository's. The int8
path is therefore not carried, and the native decode stays f32 exact.

**The instrument, corrected.** The timing test ran ten launches of one
50 MB weight back to back; 18 MB of it can stay in L2 between launches,
and every launch figure in §7.0.2ay, §7.0.2az and the table above is
that L2-assisted number. Rebuilt to stream — eight weight buffers in
rotation on one queue, so no launch finds its rows in cache and none
overlaps another (eight networks on their own streams overlapped, and
read 392 GB/s that no served step gets) — the 0022 kernel reads the gate
projection at 171 µs (293 GB/s) and the down projection at 219 µs
(229 GB/s) on the 24 GB card, the value projection at 97 µs for 3 MB;
the int8 kernel at 159 and 165 µs (316 and 304 GB/s); lane-contiguous
addresses gain 9 % there. So the kernels stream at about 300 GB/s from
DRAM, near what the runtime's own path gets, and the served decode step
— 99 ms for 15.2 GB, 154 GB/s effective, unchanged across every kernel
of this window — is not the inner loop. It is the 401 launches: their
fixed cost, the narrow projections (the value projection at 27–85 µs
for 3 MB, 35–110 GB/s), the head-order gathers. The served number never
moved because the kernel was never what it was waiting on.

**Refuted and retracted here (§7.0.1).** "The decode kernel is
instruction-bound" (§7.0.2az): three kernels with 1,900, 1,080 and 600
instructions per super-block served at the same rate. "The kernels read
rows at 222 GB/s" and every other launch-level bandwidth figure before
this section: L2-assisted, by up to a third of the rows. A work-group
cap of eight subgroups, taken for the down projection's occupancy: it
put the f32 kernel at 2.6 ms on two shapes (measured, reverted). The
16 GiB card's numbers today varied by up to 40 % between identical
runs and are not compared.

**What this closes.** The native path's decode is at its kernel's rate;
the remaining 40 % of the served step is per-launch and per-shape, the
repack path pays the same 401 launches and serves 16 t/s because the
runtime's fixed cost per launch is lower — the next lever for either
path is that fixed cost, measured per node with the profile, not the
inner loop. Plugin patch 0023 carries the corrected timing test only.

#### 7.0.2bc 0.4.1, the native decode kernel in llama.cpp's shape: block reads along K, 20 % more served decode at 1k, 10/10, byte-identical; a 16× reading retracted (2026-09-06)

The operator's window after §7.0.2bb ("kernel first … measurements after
the fact; if we end up losing again, I am willing to spend the utility
bill"), with the standing objection that 10–25 % of a card's bandwidth
is not a number to accept. What was built, what the instruments said,
and what the served path did.

**The shape.** llama.cpp's CUDA decode (read from `mmvq.cu` and
`vecdotq.cuh`, not its documentation): a thread block per output row,
its lanes along K, sixteen lanes per super-block. Transplanted to Xe as
patch 0023's decode variant: a work-group per group of output rows,
four subgroups with their lanes along K, a subgroup taking one
super-block per iteration, the subgroups' partials reduced through
local memory. Four steps, each correct on both cards (14/14 against the
host reference) before it was timed:

1. Lanes reading their eight quant bytes and two runs of eight
   activations with per-lane loads: 96 four-byte gathers per lane per
   row in the compiled kernel, 1.35–1.5× slower than 0022 on the wide
   shapes.
2. The 128 quant bytes of a Q4_K/Q5_K super-block as one sub-group
   block read (eight bytes per lane: positions l and l + 16 of every
   sub-block) and the 256 activations as another (sixteen halves per
   lane) — no gathers. The first cut took sixteen bytes per lane instead
   of eight, read 128 bytes past every last block, faulted the card
   (engine resets, ten-minute hangs per timing run) and failed the
   reference; the size fixed, 14/14. The block reads' lane mapping was
   probed rather than assumed: lane l holds element 16 i + l.
3. The activation block read once per super-block and shared by the
   work-group's rows (super-blocks outer, rows inner), the eight
   scale/min pairs decoded by lanes 0–7 and broadcast instead of by
   every lane: 195 µs on the gate projection.
4. The same loop with the super-block index clamped and the
   contribution masked, so a step of several super-blocks could issue
   its loads together: 156 µs at one super-block per step — faster than
   step 3 by 20 % for a change the compiler was meant to fold, and kept
   in that form because that is the form that was measured; two and
   four super-blocks per step were slower.

Q6_K needed its own form. Its 210-byte blocks are 2-aligned, and a
16-bit block read two bytes off a dword returns the neighbour's word on
every lane but the first (probed); so its 128 low-nibble bytes and 64
high-bit bytes are block-read as dwords from the dword at or below the
block and each lane takes its word from the lane that holds it with one
shuffle, the sixteen scales one uniform load with a per-lane select.
The first form of it, with per-lane scale gathers, was 814 µs on the
down projection; this one 508.

**In isolation** (the streamed timing test of §7.0.2bb, now over the
model's own tensor types at their shapes; every configuration warmed
once, then two repeats; 24 GB card unless named):

| M = 1, streamed | 0022 kernel | this kernel | 16 GiB card, this kernel |
|---|---|---|---|
| gate/up, Q4_K, N 17,408 × K 5,120 (44 MB) | 170 µs, 295 GB/s | **156 µs, 321 GB/s** | 171 µs |
| q/k/v, Q4_K, N 1,024 | 85–97 µs | 83–89 µs | 56–58 µs |
| Q4_K, N 5,120 × K 5,120 | 107 µs | 96–99 µs | 74–76 µs |
| down, Q4_K, N 5,120 × K 17,408 | 219 µs, 229 GB/s | **158–162 µs, 314 GB/s** | 258 µs |
| attention output, Q5_K, 5,120 × 5,120 | 137 µs | 105 µs | 79 µs |
| down, Q6_K, N 5,120 × K 17,408 (73 MB) | 646 µs, 113 GB/s | **508 µs, 144 GB/s** | 851 µs |
| Q6_K, N 1,024 | 107–126 µs | 84–85 µs | 71–73 µs |

The rows-per-group choice is by shape: four rows per work-group on the
K = 5,120 shapes (sixteen was 176–189 µs on the gate), sixteen on the
K = 17,408 down projection (four was 189 µs, sixteen 159: the shared
activation block is the lever when K is long). Four subgroups beat
eight (170–175 µs) and two (234 µs). The 16 GiB card's Q6_K figure is a
loss against the per-lane-gather form there (636 µs) and is left open:
that card does not serve a GGUF-opened model.

**The ceiling, measured rather than quoted.** A streaming read of
512 MiB of random bytes on the 24 GB card runs at 453 GB/s at every
work-group size tried — 99 % of the card's 456 GB/s specification.
(`clpeak` reports 1,139 GB/s on the same card: its test data compresses
on Xe2, and its figure is not a ceiling for weights.) The gate
projection at 321 GB/s is 71 % of what the memory system delivers to a
kernel; 0022's 295 was 65 %.

**Served** (dense Qwen3.8-27B Q4_K_M through `--gguf-native`, 24 GB
card, `u8` KV, chunk 256, one fresh process per cell; the same protocol
as §7.0.2ay's benchmark):

| prompt tokens | 0.4.0 native (0021 kernel) | this kernel |
|---|---|---|
| 856 | 213 t/s prefill, 9.9 t/s decode | 212 / **12.0** |
| 71,727 | 174 / 8.5 | 173 / 8.6 |

The Prüfstand at 10/10, its decode at 12.8–14.1 t/s over the run; the
greedy outputs byte-identical to 0.4.0's at both depths (the kernel is
exact: f32 accumulation over f16 activations). The step at 856 tokens
went from 101 to 83 ms — 18 ms, against about 10 ms of kernel time
saved by the table above (48 layers × the per-tensor differences), so
the launch sequence gained more than its kernels did. The step at
71.7k went from 118 to 116 ms, and that 2 ms is not explained by any
row above: at that depth the decode step is bound by something the
kernel does not touch, and the profile at depth (`ARCINT_PROFILE` at
71,700) is the next measurement, not a narrative.

**An intermediate result that would have ended the window wrongly.**
The first form of the kernel timed at 2.5–2.9 ms on the gate projection
on both cards, sixteen times 0022, and the reading written down was
"one work-group per row is dispatch-bound on Xe". It was not: the same
binary at the same shape timed 72 µs in one process and 2.8 ms in
another, and the pattern was the first process after every rebuild —
one of the three shapes stalls by about 2.5 ms in the process that
first compiles a kernel, and never again (the persistent kernel cache
is the candidate; not root-caused). Warmed, the form was 1.35–1.5×
slower than 0022, not 16×; work-group size costs tens of percent here
(sixteen-lane groups 275 µs against 229 for four subgroups), never an
order of magnitude. The timing ritual now warms every configuration
once and reports repeats, and the correctness cases gate the timing
loop (a kernel that faults the card hangs every timing run after it).

**Refuted and retracted here (§7.0.1).** "The served number never moved
because the kernel was never what it was waiting on" (§7.0.2bb): the
kernel was worth 18 ms of the 101 ms step at 1k, a fifth of it. What
survives of that section is the remainder — 83 ms of step for about
49 ms of kernel time — and the deep step, where the kernel moved
nothing. And the 16× dispatch reading above, which never reached this
document but did reach the kernel's comments for two hours.

**What this closes.** The native path's decode kernel is in llama.cpp's
shape with block reads along K, exact, 10/10, faster than 0022 on every
tensor type and shape of the served model on the 24 GB card, and 20 %
faster served at 1k. Plugin patch 0023 carries it (kernel, host,
utilities, the timing test over the model's shapes) in the `+p8`
recipe; the package is not built. Open: the deep step (profile at
71.7k), the 16 GiB card's Q6_K form, the first-process stall, and the
remaining gap between kernel time and step time at 1k.

#### 7.0.2bd 0.4.1, root causes after §7.0.2bc: the instrument's early-execution regime, the cold-cache stall, the deep step; and what other projects' Intel kernels do (2026-09-06)

The operator's directive after §7.0.2bc: "do the appropriate root cause",
and a survey, by research agents in English, Chinese, Japanese/Korean and
French/German/Russian sources, of optimised Intel GPU kernels for what
this repository has not considered. Three root causes were measured, one
survey item was measured and carried, and the rest are listed with their
evidence.

**Root cause 1 — the launch figures were an early-execution regime.**
Every launch figure in §7.0.2ay–§7.0.2bc came from the second and third
executions of a freshly built network (one warm-up execute, then two
timed passes over eight networks). Timing each launch alone shows the
second execution of a network costs about twice the third and later
ones (gate projection: 300 µs, then 155; the N = 1,024 shape: 330, then
150 — with a `finish` after each), and the served step is at the later
ones. The timing test now warms every network three times before the
clock. Steady state on the 24 GB card, this kernel:

| M = 1, steady, 24 GB card | early-execution figure (§7.0.2bc) | steady state |
|---|---|---|
| gate/up Q4_K 17,408 × 5,120 | 156 µs, 321 GB/s | **140–142 µs, 354–358 GB/s** |
| q/k/v Q4_K N 1,024 | 83–89 µs | **19–20 µs** |
| Q4_K 5,120² | 96–99 µs | 50–51 µs |
| down Q4_K 5,120 × 17,408 | 158–162 µs | 145 µs, 346 GB/s |
| attention output Q5_K 5,120² | 105 µs | 59–60 µs |
| down Q6_K 5,120 × 17,408 | 508 µs, 144 GB/s | 500 µs, 146 GB/s (397 with the prefetch below) |
| Q6_K N 1,024 | 84–85 µs | 30 µs |

The comparison with 0022 in §7.0.2bc was in the same regime on both
sides and stands; the absolute bandwidth claims were understated: the
gate projection streams at 79 % of the card's measured 453 GB/s
random-read ceiling, not 71 %, and the narrow shapes were dominated by
the per-execution cost, not by their bytes. What the second execution
pays is not root-caused (the plugin's first-executions path; not the
kernel).

**Root cause 2 — the first-process stall.** Reproduced at will: with the
driver's persistent kernel cache cold (`~/.cache/neo_compiler_cache`
removed), every timed launch of five of the seven shapes runs 2–6 ms
with wild jitter — all sixteen launches, not the first — while the two
long-K shapes (sixteen rows per work-group) never stall. With the cache
warm no launch stalls; with the cache on tmpfs the stall is unchanged
(disk I/O is not it); with the persistent cache disabled outright
(`NEO_CACHE_PERSISTENT=0`) there is no stall at all, only a mild
first-pass cost (480 → 280 µs). Timed alone on the shared stream the
stalled launches are quantised at 2,996 and 5,996 µs — one or two
3 ms ticks — and the stall picks different shapes in different runs. It
survives the plugin's asynchronous static-shape compilation being
disabled, the driver's completion wait being forced to spin, the
driver's direct-submission controller being switched off and its idle
timeout raised to 500 ms — and it vanishes when the driver's direct
submission itself is switched off (`NEO_EnableDirectSubmission=0`, cold
cache: every launch at 70–450 µs), as it does with the persistent cache
off. A thread dump during a stalled loop shows the compiler translating
in worker threads while one driver-internal thread sleeps in a loop.
So the stall needs two things: a kernel binary that this process
compiled and stored (not one it loaded from the cache), and the
user-mode ring; the mechanism inside the driver — a residency or
ring-restart round trip for a freshly compiled binary is the candidate
— is not measured further here. Not a kernel property, and not in a served step
(401 back-to-back launches keep the queue busy); it is why every
timing ritual warms every configuration once.

**Root cause 3 — the deep step.** The profiler's decode-step capture
prefilled the whole depth in ONE forward; at 71.7k tokens that is
770 GB of activations, the card ran out (`CL_OUT_OF_RESOURCES`), the
driver evicted to host memory (36 GB of shared memory) and the host
went out of memory — the dev host's user session died with it, both
production units with the session. The capture now walks to depth in
the served chunks (this commit). At 71.7k, this kernel's decode step:
the paged attention's 16 launches are 36 % of the step's counters
against 0.2 % at 1k, the 401 K-quant launches 58 % against 90 %. The
forward's wall clock, the same profiler process for each kernel (its
counters on, so above the served step):

| forward wall clock, M = 1 | at 1,000 tokens | at 71,700 tokens |
|---|---|---|
| 0021 (0.4.0's kernel) | 120.1 ms | 127.5 ms |
| 0022 | 119.2 ms | 131.3 ms |
| this kernel (0023) | 110.1 ms | 116.6 ms |

The forward gains 10–11 ms at both depths. The served step gained 18 ms
at 1k (101 → 83) and 1 ms at 71.7k (117.6 → 116.3): at depth the served
step is not the forward — about 15 ms of per-step work outside the graph
(the block table, sampling, the request path) sits beside it and hides
what the kernel saved. That work is the next measurement (a host-side
profile of the served loop at depth), not a kernel.

**The survey.** Four research agents read other projects' Intel GPU
kernels and the vendor's own material — llama.cpp's SYCL and OpenCL
backends, Intel's llm-scaler ESIMD kernels, OpenVINO's own int4 GEMV
kernel, oneDNN's generator, XeTLA and cutlass-sycl, Intel's Triton
backend, the oneAPI optimisation guide, Chips and Cheese's
micro-benchmarks, Intel's int2 GEMV paper — in English, Chinese
(mostly blocked to automated reading), Japanese and Korean (measured
llama.cpp backend comparisons on Arc, no kernel-level content) and
French, German and Russian (architecture coverage only). What was new
here, measured first:
- Software prefetch of the next super-block (one cache line per lane,
  cutlass-sycl's staged mainloop as the pattern): on the 24 GB card,
  steady, the Q4_K gate 142 → 207 µs (worse), N 1,024 20 → 25, Q5_K
  60 → 74 — and the Q6_K down projection 500 → **397 µs**, Q6_K
  N 1,024 30 → 33; on the 16 GiB card the Q6_K down projection 843 →
  950 (worse). Carried for Q6_K on Xe2 only (`KQ_PREFETCH` from the
  host by architecture); served, 12.1 t/s at 856 tokens and 10.1 at
  71.7k against 12.0 and 8.6 without it, 10/10, byte-identical.
- Large-register mode (Intel's guide; three agents): forced
  (`-cl-intel-256-GRF-per-thread`) the Q4_K gate 142 → 176 (worse), the
  Q6_K down projection 500 → 444; the compiler's automatic choice
  (`-cl-intel-enable-auto-large-GRF-mode`) leaves Q4_K alone and gives
  Q6_K the same 440 — the Q6_K kernel is register-heavy and
  latency-bound, the Q4_K one is not; on top of the prefetch it adds
  nothing (397 either way) and is not carried.
- llama.cpp SYCL's "reorder" (a load-time layout that separates quant
  bytes from scales; +31 % on the A770 for Q4_0, and the Q8_0 case on
  the B70 from 21 % to 66 % of bandwidth): this repository's repack
  (§7.0.2ba) is the same move taken further; an exact reorder of the
  native Q6_K into 4-byte-aligned planes is the open middle ground.
- OpenVINO's own int4 GEMV kernel (`fully_connected_gpu_gemv.cl`): K
  split over sixteen subgroups with the activation broadcast — the 0022
  shape; measured 29–35 % faster than its predecessor on long-K shapes
  in the upstream PR. Not new here.
- Not measured here, in the order the evidence suggests: LSC
  cache-control hints (L1 bypass for streamed weights; +11–54 % on Ponte
  Vecchio in Intel's Triton backend issue, untested on Battlemage); an
  exact reorder of native Q6_K into dword-aligned planes at load (the
  llm-scaler kernels repack Q6_K's high bits host-side for the same
  reason); the 2D block-load and prefetch intrinsics (Xe2 only); a
  persistent work-stealing kernel over row tiles (Intel's B60 MoE post,
  >80 % claimed); K-slicing across work-groups for the long-K shapes
  (XeTLA, llm-scaler); dequantisation by f16 denormal reinterpretation
  (oneDNN); routing the weights through the texture path (llama.cpp's
  Adreno kernels, unmeasured on Arc by anyone).
- Confirmed by the sources: sixteen-lane subgroups; dword alignment for
  block reads is a documented Xe-wide constraint; int8 activations are
  llama.cpp's own trade-off; ~65–85 % of peak DRAM bandwidth is where
  Intel's own decode kernels land on this card class (Intel's int2 GEMV
  paper: ~83 % on the B580).

**What this closes.** Patch 0023 carries the kernel with the Q6_K
prefetch, the host's architecture gate and the steady-state timing test;
the profiler's depth capture is fixed (fd5563d). Served on the 24 GB
card: 12.1 t/s decode at 856 prompt tokens against 0.4.0's 9.9, 10.1 at
71.7k against 8.5, Prüfstand 10/10, greedy outputs byte-identical at
both depths. Open, in order: the served loop's per-step work outside the
graph at depth (a host-side profile), which hides more of the kernel
than the kernel now costs; the exact dword-aligned reorder of native
Q6_K; the LSC hints; the second execution's cost in the plugin; the
16 GiB card's Q6_K form; `+p8` not built.

#### 7.0.2be 0.4.1, the table worked through: the served step split on the host, the mixed open as the default, the file's embedding, the verdict cache, `+p8` deployed, the gates on both cards (2026-09-06)

The operator's directive after §7.0.2bd: work through the seven open
items in order, adjust served units where a measurement says so, change
dependency sources where they pay, test the serving pipeline only at
the end of the turn and on both cards, and time every long run. Every
served number below is on the `+p8` runtime as installed on the dev
host (item 5), 24 GB card, `u8` KV, one fresh process per cell.

**1. The served step, split on the host.** `ARCINT_PROFILE_CYCLE` now
prints a line per plain decode step (the loop with no drafter had no
line; the cycle line was the drafting loop's): the embedding lookup,
paged_forward's own index build, graph wait, logits copy, the sampling,
the detokenise and callback halves of the commit, and the wall of the
whole iteration. The mean of the last 40 of 64 tokens, before the
changes below:

| step, ms | embedding | index | graph | logits | sampling | step | loop |
|---|---|---|---|---|---|---|---|
| native, 856 tokens | 6.7 | 0.06 | 75.7 | 0.09 | 0.15 | 82.7 | — |
| native, 71.7k | 3.0 | 0.07 | 93.0 | 0.10 | 0.16 | 96.4 | — |
| mixed, 856 | 1.0 | 0.05 | 70.1 | 0.09 | 0.15 | 71.5 | — |
| mixed, 71.7k | 3.1 | 0.06 | 88.6 | 0.08 | 0.14 | 91.9 | — |
| mixed, file embedding, 856 | 0.02 | 0.05 | 70.6 | 0.09 | 0.20 | 71.1 | 79.4 |
| mixed, file embedding, 71.7k | 0.00 | 0.06 | 90.0 | 0.09 | 0.13 | 90.3 | 91.1 |

What sits outside the kernels, in the order of size:

- The graph forward itself carries about 30 ms of launch sequence
  around the ~45 ms of K-quant kernels at 1k: some 1,300 launches per
  step, of which 209 RMS norms, 144 gathers, 96 Swish and 64 Multiply
  eltwise unfused around the gate/up projections, 96 small f16 matmuls,
  48 GDN steps, 48 convolutions, 16 paged attentions. Fusing the eltwise
  into the K-quant kernel as post-ops (the plugin's `FUSED_OPS` on the
  store) removes 160 launches and their intermediates (item 3) at once:
  the next plugin patch, not this one.
- The first decode step after a prefill costs 2.3× the steady step and
  the second 1.3× (216 and 100 ms against 93 at depth; 186 and 105
  against 75 at 1k): the plugin's first executions of the one-token
  shape, the same effect §7.0.2bd measured on the timing instrument.
- At depth the request's own accounting shows decode 7.22 s = graph
  6.09 s + emit 1.12 s over 64 tokens, and the first token's loop is
  1.23 s against its 0.22 s step: the first emitted piece after a 71.7k
  prefill cost about a second on the host in that run. With the commit's
  two halves on the step line (the detokeniser is an OpenVINO CPU model,
  the callback the server's), the rerun of the same cell did not
  reproduce it: emit 0.01 s over 64 tokens, the first step's detokenise
  0.56 ms, the callback 0.00, the decode 11.2 t/s with a steady step of
  89.5 ms. Observed once, not root-caused; the instrument stays.
- The template's embedding model cost 1–7 ms per token, 0.1 to 12 ms
  from step to step: a device round trip with a queue in front of it.
  Item 7 removes it.

**2. The mixed open, now the default** (`--gguf-mode mixed`: Q4_K
repacked into the runtime's compressed form, Q5_K and Q6_K the file's
rows in the K-quant kernel; `repack` and `native` stay flags):

| 24 GB card, u8 KV | resident | max ctx at u8 | 856: prefill / decode | 71.7k: prefill / decode | load |
|---|---|---|---|---|---|
| native (0023) | 14.94 GiB | 100k | 212 / 11.6 | 173 / 8.5 | 55–78 s |
| **mixed** | 16.26 GiB | 86k | 302 / 12.7 | 258 / 10.2 | 267–285 s, 88–168 with the verdicts kept |
| repack (§7.0.2ba) | 18.73 GiB | 46k | 1,005 / 16.1 | fits only at u8:i4 | 406 s |

The 71.7k cell is back inside `u8` KV, which the milestone's protocol
requires; the greedy outputs at both depths are byte-identical to the
native path's. Prefill at 302 t/s is the Q6_K down projection and the
Q5_K attention output through the K-quant kernel's tiled variant: the
milestone's lever 2, untouched. The decode figures over 64 tokens
scatter by ±2 t/s between runs (10.9, 12.7 and 13.6 for the same cell
today); the Prüfstand's 1,145-token decode is the steadier number:
14.5 t/s with the file's embedding against 12.1 with the template's.

**3. The activation reservation, its carrier measured by the mode
census.** The fit charges 10,986 KiB per chunk token in native mode,
3,471–3,906 in mixed, 0.03 GiB in all on the IR path. The difference
between native and mixed is the 288 Q4_K projections moving from the
K-quant op to the runtime's own fully-connected: about 26 KB per token
per native projection — an f16 output of the projection's width (17,408
for gate/up) that the plugin's memory pool keeps rather than reuses
across layers, plus the Swish and Multiply intermediates the runtime's
own path fuses away as post-ops. The plugin's allocation rule
(`primitive_inst::allocate_output`) goes through the pool only when the
node can share its buffer and the memory dependencies allow; which of
the two fails for the K-quant node needs the pool's dump, a debug-build
option. Not fixed here: the mixed open halves it (chunk 512–1024
instead of 256), and the fused post-op above removes the rest of the
intermediates.

**4. The load time.** Of a 285 s mixed load, the repack is 18 s and the
exhaustive deviation check 224 s; of the native load nothing is either.
The verdict per repacked projection is now kept between loads of the
same file (`--gguf-check once`, the default: keyed by the file's size
and mtime, the tensor's offset, type and dims, and the bound; only a
passed verdict is written; the configured cache directory or the
user's) — the next load of the served file was 88 s, of which the
repack 35 s and the check 0. `--gguf-check always` re-checks at every
load.

**5. `+p8` built and deployed.** The recipe at patches 0003–0023 built
in 13 minutes on the dev host (an incremental build; 1,058 objects),
`marfrit-openvino 2026.4.0~dev20260821+p8-1` installed there, both
production units restarted on it and answering. The served IR path is
untouched by 0022/0023 (the kernel exists only for tagged K-quant
constants): the coder's equivalence suite on the 16 GiB card under
`+p8` is 9 of 9 checks byte-identical (two greedy runs, sliced logits,
speculative determinism, the drafter's acceptance, warm against cold
cache, the cache hit, a cached continuation), the MTP section skipped
as declared for that artifact.

**6. The gates.** Prüfstand 10/10 through the GGUF-opened model in its
default form on the deployed runtime. The equivalence suite on the
GGUF-opened model passes its first check (two greedy runs
byte-identical) and then starts a stateful-path server for its
logits-slicing check, which a GGUF-opened model refuses by design
(`--gguf` serves on the paged path only), and stops: the cold/warm and
chunk checks need a variant of the suite that skips the stateful
sections — open, not a failure of the path. The 16 GiB card
cannot hold the GGUF-opened model: 16.26 GiB resident against a 16 GB
card, the load dies in the driver (`CL_OUT_OF_RESOURCES`); the gate's
"measured and recorded at both depths" is recorded as this, and the
native form at 14.94 GiB leaves no room for activations there either.
On the deployed runtime with the file's embedding, the native form
serves 212 / 12.8 t/s at 856 tokens and 173 / 10.3 at 71.7k (step 74.5
and 91.9 ms), the repack 16.2 t/s at 856 (a 57.6 ms step), the mixed
default 10.9–13.6 over 64 tokens and 14.5 over the Prüfstand's 1,145.

**7. The deferred items.** The embedding from the file:
`token_embd.weight` (Q4_K, 248,320 × 5,120, 682 MiB) is copied out of
the map into host memory at load and one row is dequantised per token
with ggml's own decoder (`--gguf-embed file`, the default); the
template's i8 embedding model no longer runs on the served path. The
first form read the rows from the map and page-faulted from disk at
37 ms per new token (measured, 8.8 t/s); copied, the lookup is 0.02 ms.
The greedy output at 856 tokens is byte-identical to the template
embedding's; at 71.7k it differs (086d5e71ad47 against 5e3fb7a8d72c):
the file's Q4_K rows are not the template's i8 rows, and at that depth
a near-tie flips. Prüfstand 10/10 either way. The MTP layer stays the
template's: `--mtp on` on the GGUF-opened model serves 13.6 t/s at 856
tokens with the same greedy output (the first measurement of that
combination).

**The 16 GiB card, the A/B 0023 had not had.** Patch 0023's header
measured the 24 GB card only; the operator asked for the fresh number.
On the A770, same instrument, same window, 14/14 both sides:

| A770, steady, µs | 0022 | 0023 as shipped | 0023, rule per architecture |
|---|---|---|---|
| gate Q4_K 17,408 × 5,120 | 159 | 154 | 154 |
| q/k/v Q4_K N 1,024 | 24–26 | 20–22 | 21 |
| Q4_K 5,120² | 53–56 | 54 | 54 |
| down Q4_K 5,120 × 17,408 | 171–174 | 224 | 204 |
| Q5_K 5,120² | 98 | 62 | 64 |
| down Q6_K 5,120 × 17,408 | 467 | 842 | 510 |
| Q6_K N 1,024 | 39 | 46 | 44 |

The long-K rule (sixteen rows per work-group) was the 24 GB card's; on
Xe-HPG the sweep puts the long-K projections at eight rows × eight
subgroups (the gate keeps 4 × 4), which the host now selects by
architecture, and the 24 GB card's figures are unchanged. Even so the
two down projections stay 19 % and 9 % behind 0022's lane-per-column
form on that card, while every other shape is ahead; per token of the
served model that is about level. What 0022's Xe-HPG decode had — the
sub-block packed for the one-row matrix multiply — is not in 0023's
tree; carrying it back as a second decode body selected by architecture
is a maintenance decision, put to the operator with these numbers. That
card cannot hold the GGUF-opened model of this size, so no served
number exists for it.

**What is left in the table.** The decode bar (within 1.2× of the IR's
23.1 and 16.5 t/s) is not met on any form: the mixed default serves
14.5 / 10.2, the repack 16.1 at 1k and does not fit at depth. The
levers that remain are named by measurement: the fused post-op on the
K-quant kernel (160 launches and the intermediates), the 2D block loads
for the tiled variant's A operand (the mixed form's prefill), and the
first-piece second at depth on the host. The `+p8` package is the
deployment; arcint itself is not yet packaged past 0.4.0.

#### 7.0.2bf 0.4.1, the same bytes on the same card through the other stacks: llama.cpp Vulkan and SYCL against the GGUF-opened forms and the Intel IR, at 1k and 10k (2026-09-07)

The operator's question after §7.0.2be — whether Vulkan's infrastructure
is a competitor — answered by measurement rather than argument, with the
depth held at 10k ("enough to get a trend"). One 24 GB card, the dense
Qwen3.8-27B Q4_K_M file for every GGUF arm, 64 generated tokens per cell,
one fresh process per cell. llama.cpp at upstream 7b13a84, built on the
dev host with its Vulkan backend (Mesa's driver) and its SYCL backend
(oneAPI 2026.1; its Level Zero path segfaulted in the container, whose
loader is absent, so SYCL ran on its OpenCL backend). `llama-bench -fa 1
-ngl 99`, prompt processing of 1,000 and 10,000 tokens from an empty
context, generation of 64 at depth 1,000 and 10,000. arcint's cells are
the served endpoint on the deployed `+p8` runtime, prompts of 856 and
10,010 tokens, `u8` KV.

| stack, B60 | prefill 1k | prefill 10k | decode at 1k | decode at 10k | resident |
|---|---|---|---|---|---|
| Intel int4 IR through arcint (the reference; different bytes) | 1,598 t/s | 1,434 | 23.4 | 23.4 | 13.06 GiB |
| arcint, repack form | 937 | 1,051 | 16.3 | 16.1 | 18.73 GiB |
| arcint, mixed form (the default) | 309 | 358 | 13.4 | 11.7 | 16.26 GiB |
| llama.cpp SYCL (OpenCL backend) | 249 | 206 | 14.2 | 12.4 | — |
| llama.cpp Vulkan | 126 | 108 | 7.8 | 7.0 | — |

Reading it. Against the two stacks that read the same bytes, arcint's
default form is ahead of Vulkan by 2.5× at prefill and 1.7× at decode,
and ahead of SYCL at prefill (1.2× at 1k, 1.7× at 10k) while level with
it at decode (13.4 against 14.2 at 1k, 11.7 against 12.4 at 10k). The
repack form is ahead of both on every cell. The trend from 1k to 10k is
the same shape on every stack: decode loses 10–13 %, prefill loses
10–17 % on llama.cpp's backends and gains on arcint's (the served chunk
grows into the prefill); the IR's decode does not move at all over that
range. Against the operator's goal (within 20 % of the IR: 1,147 t/s and
18.7 t/s at 10k), the mixed form is at 25 % and 63 % of the IR, the
repack at 73 % and 69 % — the repack is the closer form on rate and the
mixed form on size, and neither is inside the bar.

So Vulkan is not the infrastructure to move to for this card: it is the
slowest stack measured on it, as the fleet note already said. SYCL is
the stack to keep in view: its decode equals the mixed form's from the
same bytes with none of this repository's kernel work, which says the
K-quant decode here is at the level of Intel's own engineers' and not
beyond it, and that the next lever is not in the kernel.

Retracted here (§7.0.1): nothing; the earlier statements about Vulkan
on Battlemage (§7.0.2bd's survey, the fleet note) were narrated from
other people's measurements and are now this card's own.

#### 7.0.2bg 0.4.1, the decode step on the device timeline: every fully-connected kernel within 15 % of the card's bandwidth except the K-quant kernel on Q6_K; the launch-sequence and fused-post-op readings retracted; `--dyn-quant on` 2/10; the logits slice missing at the K-quant lm_head, fixed (2026-09-07)

§7.0.2be left the decode bar unmet with two levers named by
subtraction: "about 30 ms of launch sequence around ~45 ms of K-quant
kernels" and "96 Swish and 64 Multiply unfused around the gate/up
projections", to be settled by an ablation or by the device timeline
§7.0.2bb had proposed and nobody had built. Two things were already on
the record before any window: the K-quant kernel has carried the
plugin's `FUSED_OPS` on its store since patch 0021, and the decode-step
node dump of the native form (2026-09-06) lists the MLP's 64 Swish and
64 Multiply with no implementation and no time — fused — while the 96
Swish and 64 Multiply it does launch are the linear attention's gate and
norm and the attention's output gate. So the ablation had nothing to
remove, and the timeline was built instead: the OpenCL intercept layer
of §7.0.2d (already on the dev host), one served process per form,
`CLI_ChromePerformanceTiming` for the per-kernel device intervals, the
trace's tail segmented into decode steps on the one copy every step
ends with (the sampled row's logits, 248,320 × 4 bytes), steady state
the last 40 of 64 steps. `tools/cl_timeline_steps.py` is the parser.
24 GB card, 856-token prompt, `u8` KV, MTP off, the deployed `+p8`
runtime, the 0.4.0 tree with the step line. The greedy outputs of the
traced processes are byte-identical to the untraced runs' (the three
GGUF forms 23e06c37e0d6, the IR 50815ed40613). The tracer's cost on the
step is +4 % on the IR and +15–18 % on the GGUF forms (§7.0.2d's +42 %
was a launch-bound decode; these are not), and the device intervals do
not carry it: the untraced step of the same form and binary
(§7.0.2be's step lines, the IR's from its 23.1 t/s) is the denominator.

**The step, on the device.**

| form, 856 tokens | launches per step | device busy | untraced step | idle in the untraced step | traced step |
|---|---|---|---|---|---|
| Intel int4 IR (13.0 GiB of weights) | 1,265 | 37.9 ms | 43.3 ms | 5.4 ms (12 %) | 44.9 |
| repack (19,085 MiB) | 2,164 | 55.3 | 57.6 | 2.3 (4 %) | 66.4 |
| **mixed** (16,587 MiB) | 2,124 | 62.7 | 71.1 | 8.4 (12 %) | 83.8 |
| native (15,335 MiB) | 1,676 | 60.3 | 74.5 | 14.2 (19 %) | 87.2 |

No form is launch-bound. The mixed default keeps the card busy for
62.7 of its 71.1 ms; the "30 ms launch sequence" was the native form's
kernel sum subtracted from the mixed form's step. What the device time
is made of, per step, with the bytes each class reads and the rate that
implies against the 453 GB/s the random-read probe measured on this
card (§7.0.2bd):

| per step | runtime gemm on repacked or IR weights | K-quant, Q4_K | K-quant, Q5_K | K-quant, Q6_K (the lm_head within) | everything else |
|---|---|---|---|---|---|
| IR | 34.8 ms, 465 launches, 13.96 GB, **401 GB/s** | — | — | — | 3.1 |
| repack | 49.7 + 1.2 small, 385 launches, 20.0 GB, **393** | — | — | — | 4.4 |
| mixed | 30.2 + 0.9, 272 launches, 11.8 GB, **389** | — | 3.3, 48, 1.04 GB, 316 | 23.7, 65, 4.45 GB, **188** (5.35) | 4.6 |
| native | 0.5 | 28.2, 288, 10.5 GB, 372 (gate/up 354) | 3.3, 48, 318 | 23.7, 65, **188** (5.59) | 5.1 |

Every fully-connected kernel on this card runs at 86–89 % of the
measured ceiling — the runtime's own gemm on u4 IR weights, the same
gemm on the repacked rows, and the K-quant kernel on Q4_K rows alike —
except the K-quant kernel on Q6_K at 188 GB/s and, less so, on Q5_K at
316. "Everything else" (attention 0.8 ms, GDN 0.4, conv 0.2, the norms
0.5, the mixed form's activation widening about 1.4 across ~480 small
launches) is 3–5 ms on every form. The decode gap between the mixed
form and the IR is therefore three measured items and nothing else:
3.4 GB more bytes per token (17.4 against 14.0, of which 1.31 GB is the
repack's augmentation — each super-block's eight mins occupy a 32-wide
augmented group, half of it zeros, +12.5 % on the Q4_K set), the Q6_K
kernel at half the rate of the others (12 ms of the 23.7 would go at
the gemm's rate), and 6 ms more idle than the repack form. The idle
tracks the K-quant node count (0 → 2.3 ms, 113 → 8.4, 401 → 14.2: about
30–50 µs of host time per K-quant execution that the queue does not
hide), not the launch count (the repack form has the most launches and
the least idle); a per-execution cost on the custom node — §7.0.2be
item 3's unpooled output is the candidate — is the plugin-side thing to
read next, as a hypothesis.

The bar arithmetic, from these numbers. The mixed form must read
17.4 GB per token; at the gemm's 389 GB/s that is 44.7 ms, plus 4.6 of
everything else, plus the repack form's 2.3 of idle: 51.6 ms, 19.4 t/s,
against the operator's bar of 51.8 ms (19.3 t/s, the IR's 23.1 within
20 %). The bar sits exactly at the floor of the file's bytes on this
card. It is reachable only with every lever and no margin — the Q6_K
kernel at the others' rate (−12 ms), the idle at the repack form's
(−6), the augmentation's idle half (−1.7) — and the only margin the
file does not dictate is the augmentation's.

**The first decode step after a prefill** (§7.0.2be: 2.3×) is host
time: on the mixed form its 2,213 launches take 64.2 ms on the device
inside a 265 ms span, 201 ms with the card idle.

**What a served prefill is made of**, the same traces, the two chunks
of the 856-token prompt (chunk 512 on the mixed form, 2,048 on the
IR):

| prefill of 856 tokens | launches | device busy | span | idle | the largest items |
|---|---|---|---|---|---|
| IR | 3,236 | 541 ms | 736 | 195 (26 %) | gemm at M>1 321 ms (59 %), GDN 78, gemm at M=1 38, conv 19, dynamic quantisation 14 |
| mixed | 4,698 | 2,331 | 3,308 | 977 (30 %) | tiled K-quant Q6_K down 750 (32 %), f16 gemm on the repacked set 419 (18 %), **the lm_head's tiled kernel 332 (14 %)**, Q6_K tiled 311, Q5_K tiled 162, GDN 77, **a 850 MB device-to-host copy 60** |

Three readings. The tiled K-quant variant is 69 % of the mixed form's
prefill device time, as §7.0.2be said (lever 2), and the intercept
layer's kernel names carry a fact the unit tests never showed: every
tiled launch spills registers (`SPILL=1216–1472` bytes per thread; the
decode variant spills nothing), which the 2D block loads will have to
take into account. The runtime's f16 gemm on the repacked set is 18 %,
the share `--dyn-quant on` could halve. And the lm_head runs its tiled
kernel over every row of the chunk and copies 850,247,680 bytes —
856 × 993,280, the f32 logits of every prompt token — to the host once
per prefill: §7.0.2e's defect in a new place. The repack form's trace
copies one row per chunk; the mixed and native forms' copy every row,
and their load logs had said so all along ("logits NOT sliced: every
prefill chunk will compute and copy [M, vocab] logits"), one line among
a hundred. The slice walks from the first Result to the LM head and
accepted only a MatMul; on a GGUF-opened model whose `output.weight`
stays in the file's rows (mixed: Q6_K; native) the head is the K-quant
op. Fixed in the walk (the op's input 0 is the activation, input 1 the
u8 rows; `exec/graph_rewrites.h` exposes the rewrite), red first: a
K-quant head in the paged layout is not sliced by the old walk, and is
by the new one, with a MatMul head as the control (`tests/test_gguf_graph.cpp`).
The served effect on the mixed form is measured below; the reservation
fit had charged 993 KiB per chunk token for the logits alone.

**`--dyn-quant on` on the mixed form** (§7.0.2be's second prefill
lever: the runtime's per-token int8 activations on the repacked set),
same card and runtime, one fresh process per cell, the dyn-quant-off
control in the same window:

| mixed form, u8 KV | prefill | decode (64 tokens) | step | greedy output | Prüfstand |
|---|---|---|---|---|---|
| 856 tokens, dyn-quant on | 290.5 t/s | 13.7 t/s | 69.1 ms | 3cba6c3128b2 (differs) | **2/10** (9 of 10 cases return an empty table) |
| 856 tokens, off (control, same window) | 288.2 | 13.4 | 69.6 | 23e06c37e0d6 | 10/10 (§7.0.2be) |
| 71,727 tokens, on | 258.7 | 9.6 | 87.8 | 086d5e71ad47 (same as off) | — |
| 71,727 tokens, off (§7.0.2be) | 258 | 10.2–11.2 | 89.5–90.3 | 086d5e71ad47 | — |

Dead: the prefill does not move at either depth (the mixed form's
prefill is the tiled kernel's, above), the decode step gains 1–2 ms,
and the acceptance task fails. The mechanism is not measured; the
repack's augmented columns carry group sums of the activation and are
quantised with it, which is the first thing to test if the lever is
ever wanted, and the int8-activation form that scored 8/10 in §7.0.2bb
was a different scheme. The flag stays off for GGUF-opened models.

**The logits slice on the K-quant head, served** (the fixed tree on
the deployed runtime, mixed form, one fresh process per cell, the load
log now "logits sliced to the last 1 row(s)" and the probe's "slice
verified"):

| mixed form, u8 KV, 24 GB card | prefill | decode (64 tokens) | step | greedy output | activation fit | max ctx at u8 |
|---|---|---|---|---|---|---|
| 856 tokens, before (§7.0.2be, the same-window control above) | 288–302 t/s | 12.7–13.4 | 69.6–71.1 ms | 23e06c37e0d6 | 3,438 KiB per chunk token, chunk 1024 = 3.48 GiB | 86k |
| 856 tokens, sliced | **412.7** | 13.6 (Prüfstand 1,145 tokens: 14.5) | 68.9 | 23e06c37e0d6 | **2,129**, chunk 1024 = 2.06 GiB | — |
| 71,727 tokens, before | 258 | 10.2–11.2 | 89.5–90.3 | 086d5e71ad47 | — | 86,592 |
| 71,727 tokens, sliced | **291.2** | 9.7 | 87.6 | 086d5e71ad47 | 2,219 | **109,248** |

Prüfstand 10/10. Prefill +37 % at 1k and +13 % at depth, the outputs
byte-identical at both, and 1.3 GiB of activation reservation per
1,024-token chunk returned to the KV pool: the served ceiling at `u8`
moves from 86k to 109k tokens. Against the operator's prefill bar
(1,341 t/s at 1k, 460 at depth) the mixed form is now at 31 % and
63 %; the rest of the distance is the tiled kernel.

Retracted here (§7.0.1): §7.0.2be's "about 30 ms of launch sequence
around the ~45 ms of K-quant kernels" (the mixed form's device time is
62.7 of 71.1 ms and its K-quant kernels 27 ms of it; the 45 was the
native form's kernel sum); its "96 Swish and 64 Multiply eltwise
unfused around the gate/up projections", and the fused post-op named as
the next plugin patch on the strength of it (the MLP's eltwise has been
fused into the K-quant kernel since patch 0021; the launched ones are
the GDN's and the attention's gates, 0.2 ms of device time per step);
and the handoff's "decode is the launch sequence around the kernels,
not the kernels" (it is the bytes, the Q6_K kernel, and the host time
per K-quant node, in that order).

#### 7.0.2bh 0.4.1, the Q6_K decode rate on the 24 GB card, worked through in one window: fewer messages (patch 0024) move the 16 GiB card and not this one; alignment, layout, prefetch and dispatch measured null; the read shape the card wants measured by a probe, and it needs the bytes laid out for it (2026-09-07)

The operator's order after §7.0.2bg: the Q6_K decode rate next, in a
30-minute card window, the setup tailored to it. The record going in:
the K-quant kernel on Q6_K at 188 GB/s per served step against 372 for
the same kernel on Q4_K and 389–401 for the runtime's gemm, on a card
whose random-read ceiling is 453 (§7.0.2bg); the Q6_K row of patch 0023
block-reads the 128 ql and 64 qh bytes as dwords from the dword below
the 2-aligned block and gathers the tail word, the sixteen scales and
the two bytes of `d` per lane (§7.0.2bc). Counting messages per
super-block per row — Q4_K two, Q5_K three, Q6_K six — and setting them
against the three measured rates (72, 59 and 35 bytes per message at
372, 316 and 188 GB/s) gave a line straight enough to build on: the
first form (v26, now patch 0024) reads the block's last twenty bytes as
one 16-bit block read at the dword below them and takes every value by
a broadcast, three messages instead of six, the last block of the last
row guarded per lane against the 12–14 bytes the read runs past it.
Everything else was prepared off the card: the plugin build, a chain
that takes both cards for the unit tests and then only the 24 GB card
for the served cells, and the kernel's measurement knobs behind
compile-time defines switched at run time through the tree's
`ARCINT_CL_OPTS` (§7.0.2bc's knob).

**Patch 0024, measured** (the streamed timing test of 0023, every
process warmed once, one row; the served cells on the deployed `+p8`
base with the staged plugin, mixed form, `u8` KV, one fresh process
per cell):

| Q6_K decode, µs | 24 GB card, 0023 | 24 GB card, 0024 | 16 GiB card, 0023 | 16 GiB card, 0024 |
|---|---|---|---|---|
| down projection, N 5,120 × K 17,408 | 397 | **400** | 510 | **385** |
| N 1,024 × K 5,120 | 34 | 34 | 44 | 34 |
| every Q4_K and Q5_K shape | — | unchanged | — | unchanged |

Correctness 14/14 on both cards. Served on the 24 GB card: greedy
outputs byte-identical to §7.0.2bg's at 856 tokens (23e06c37e0d6) and
71,727 (086d5e71ad47), Prüfstand 10/10, prefill 433 / 291 t/s, the
decode step 70.6 and 88.2 ms against 68.9 and 87.6 — unchanged within
the run-to-run scatter. So the message count was the 16 GiB card's
limiter (−25 % on its down projection) and not the 24 GB card's, and
the straight line through three points is retracted as a mechanism
below. The patch stays: fewer messages, exact, a gain where it is one.

**What bounds the row on the 24 GB card, one part removed at a time**
(the same test, the down projection, the knobs off in every served
number above; wrong values by design, timing only):

| the v26 row with … | µs |
|---|---|
| everything (the baseline) | 400 |
| the value decode removed (the reads, the shuffles, one fold) | 291 |
| the shuffles and broadcasts removed (own-lane words) | 294 |
| the prefetch removed | 392 |
| only its three reads (folded, no decode, no shuffles) | 353 |
| the three reads from the cache line below (aligned), full decode | 402 |
| the same, reads only | 358 |
| the reads as per-row planes (the shape a load-time reorder gives), full decode | 402 |
| the same, reads only | 329 |

And the dispatch, rows × subgroups per work-group, Q6_K only (the Q4_K
rows in the same runs unchanged at 145–147): 16 × 4 (the rule) 399,
8 × 4 411, 4 × 4 442, 8 × 8 430, 16 × 8 422, 16 × 2 489, 4 × 8 468,
8 × 2 717; on the N 1,024 shape the rule's 4 × 4 (34) is the best of
the same eight. Four readings. The arithmetic and the variable-index
shuffles cost about 105 µs each on top of the reads, and the reads
alone, in this three-message shape, cost 330–358 — so the row is at a
floor set by its read shape and pays its ALU on top of it, where the
Q4_K row (one byte-wise block read per super-block, every lane's bytes
its own, no shuffles) pays neither. Forcing the same reads onto cache
lines or onto planes changes nothing: alignment is not the mechanism.
The Xe2 prefetch of §7.0.2bd is worth nothing on this form (392
without it). And the dispatch is already the best of eight.

**The read shape the card wants, measured by a probe** (an OpenCL
program over the down projection's 73 MB in random bytes, the decode
dispatch's 16 rows × 4 subgroups, five read shapes, each folded into a
register; best of five after three warm runs; the rate as the 210-byte
bytes each block carries, since the card fetches every line of the row
whatever the message shape):

| read shape, 24 GB card | µs | GB/s |
|---|---|---|
| 0023/0024's: block_read2 + block_read + us from the dword below, 210-byte stride | 253 | 289 |
| byte-wise uc8 + uc4 + us at the file's 210-byte stride (wrong values off a dword) | 255 | 287 |
| the same three byte-wise reads on 16-byte-aligned 224-byte blocks | 210 | 349 |
| one uc8, 128 bytes per block (Q4_K's one message) | 175 | 418 |
| two uc8, 256 bytes per block (over-read) | 183 | 399 |
| one block_read4 of 256 bytes from the dword below (one dword message) | 191 | 383 |

And the alignment rule for byte-wise block reads, probed the way
§7.0.2bc probed the 16-bit ones: `intel_sub_group_block_read_uc8` and
`_uc4` return the right bytes at +0, +4, +8 and +16 and the wrong ones
at +2 — a dword address is required and sufficient, so Q4_K's shape
cannot be read off the file's 2-aligned Q6_K blocks. Three messages of
mixed width from the dword below is what the file's layout allows, and
the card moves 289 GB/s that way against 383–418 for one or two wide
messages per block.

**What follows, on the record.** The Q6_K decode lever is a load-time
reorder of the native Q6_K rows into 224-byte blocks (ql 128, qh 64,
scales 16, d 2, pad 14; +6.7 % bytes on the Q6_K set, 4.24 → 4.53 GB
per token, about 0.3 GiB resident), read in the kernel as two byte-wise
block reads per super-block: the first the ql bytes in Q4_K's shape
(lane l holds positions l and l + 16 of every run, no shuffles), the
second qh in the same shape plus one scale byte and the bytes of `d`
per lane, broadcast by uniform index. The tiled prefill variant reads
the new stride with its existing vloads. Predicted from the probe:
183–210 µs for the down projection against 400, about 12 ms per served
step at 1k (§7.0.2bg's 23.7 → 12), which is the largest single decode
item left. The arcint side is a second layout under the K-quant op
(the file's rows stay the default for Q4_K/Q5_K), the plugin side one
decode row and one correctness case; both are hours, not a window, and
were not started. The augmentation packing and the host time per
K-quant node (§7.0.2bg) are unchanged by this section.

Retracted here (§7.0.1): §7.0.2bg's "bytes per message" as the Q6_K
mechanism on the 24 GB card (a line through three points; halving the
messages moved the 16 GiB card and not this one); the handoff's "an
exact dword-aligned reorder of the native Q6_K rows at load" as the
lever in that form (alignment alone is null with the current read
shape: 402 against 398; what the reorder buys is the shape, not the
alignment); the Xe2 Q6_K prefetch's 500 → 397 (§7.0.2bd) as a
standing gain (392 without it on this row). The ISA, counted after the window
through `libiga64` (the intercept layer's dumps are raw ISA;
`tools/igadis.cpp` disassembles them; the host had the library and no
binary — a tool installed as the need arose, the operator's standing
instruction of the same day): per row and super-block the decode
kernel runs about 296 instructions on Q6_K against 178 on Q4_K (4,729
against 2,845 for the sixteen-row bodies; 1,297 against ~800 for the
four-row ones), with 15 variable-index register moves per row and
super-block on Q6_K (the shuffles; Q4_K has none) and 858 shifts
against 245. The static count agrees with the ablation: the ALU the
Q6_K row pays over the Q4_K row is the shuffles and the wider decode,
and both go away only with a layout each lane can read its own bytes
from.

#### 7.0.2bi 0.4.1, the Q6_K decode row's ISA read instruction by instruction, and the row without variable-index shuffles (patch 0025) (2026-09-07)

The operator's order after §7.0.2bh: `+p9`, then a deep dive into the
indirect moves per row and super-block, and the instructions. `+p9`
(patches 0003–0024) built from the recipe in fourteen minutes and
was installed on the dev host, both units restarted on it. The dive:
the intercept layer's ISA dumps of the sixteen-row decode bodies of
§7.0.2bh, disassembled through `libiga64` (`tools/igadis.cpp`) and
counted by class, then read.

**The accounting, per row and super-block of the sixteen-row body**
(the down projection's kernel; the whole body divided by sixteen):

| per row and super-block | Q4_K | Q6_K (0024) | what it is in the Q6_K row |
|---|---|---|---|
| instructions | 178 | 296 | |
| indirect moves `mov r[a0]` + address-register setups | 0 + 0 | 15 + 21 | ten `intel_sub_group_shuffle` with a per-lane index and ten `sub_group_broadcast` with a run-time index (`4h + odd`); each is an `add a0` and an indirect move |
| shifts | 15 | 54 | the variable `>> hi` per word after the shuffle, then the byte extraction by shifts — the Q4_K row's compiler extracts bytes as register regions (`mov :uw <- :ub`, 256 of them) and shifts nothing |
| branches (goto, join, jmpi) and syncs | 2.5 + 0.8 | 12.5 + 2.2 | the per-row `last` guard of 0024 (a goto/join per row) and the lane-dependent selects |
| int-to-float converts | 16 | 24 | sixteen values, eight scales |
| mad / mul / bfn / sel / add | 24 / 19 / 2 / 4 / 12 | 16 / 25 / 16 / 12 / 47 | the same arithmetic split differently; the adds are the `- 32` per value and the address arithmetic |
| sends | 3.6 | 8.6 | the reads, and the tail window's per-lane path |

Everything the Q6_K row pays over the Q4_K row traces to one decision
of §7.0.2bc: the 2-aligned block is read as dwords from the dword
below it, and each lane then fetches its word from the lane that holds
it. That is 118 of the 296 instructions (the shuffles, their setups,
the variable shifts and the selects), and it is also why the compiler
cannot use byte regions for the extraction. §7.0.2bh's probe had
already shown that a 16-bit block read at the same dword-aligned base
is exact; what it did not say is that this read lands the right words
in the right lanes by itself: lane l holds words l, l + 16, l + 32,
l + 48 of the ql bytes and l, l + 16 of the qh bytes — for an even
block exactly the positions 2l, 2l + 1 of every run the decode wants,
for an odd block (the block starts one word later) the previous lane's,
which one fixed-delta `intel_sub_group_shuffle_down` per register puts
right, lane 15 taking the following register's lane 0, under a uniform
branch. The tail window's scales and `d` are then constant-index
broadcasts. That is the v27 row (patch 0025): no indirect move, no
variable shift, the same arithmetic, the same `last` guard.

**Measured** (the streamed timing test, warmed; correctness 14/14 on
both cards; the served cells on `+p9` with the staged v27 plugin):

| Q6_K decode, µs, streamed | 24 GB card, 0024 | 24 GB card, v27 | 16 GiB card, 0024 | 16 GiB card, v27 |
|---|---|---|---|---|
| down projection, N 5,120 × K 17,408 (73 MB) | 400 | **259** (282 GB/s) | 385 | **262** |
| N 1,024 × K 5,120 | 34 | **28** | 34 | **28** |
| every Q4_K and Q5_K shape | — | unchanged | — | unchanged |

The per-row `last` guard costs nothing the instrument can see (275 µs
with it removed for timing, inside the run-to-run band of the earlier
sweeps). Served on the 24 GB card, mixed form, `u8` KV, one fresh
process per cell, against the `+p9` runtime's own numbers of the same
morning (§7.0.2bh):

| mixed form | prefill | decode (64 tokens) | step | Prüfstand | greedy output |
|---|---|---|---|---|---|
| 856 tokens, 0024 | 433 t/s | 13.4 t/s | 70.6 ms | 10/10 at 14.4 t/s | 23e06c37e0d6 |
| 856 tokens, v27 | 432 | **15.3** | **61.0** | **10/10 at 16.7** | 23e06c37e0d6 |
| 71,727 tokens, 0024 | 291 | 9.6 | 88.2 | — | 086d5e71ad47 |
| 71,727 tokens, v27 | 291 | **11.9** | **79.9** | — | 086d5e71ad47 |

Eight milliseconds off the step at both depths, byte-identical, which
is the Q6_K set's 23.7 ms of §7.0.2bg going to about 15.4 at the new
rate (65 launches, the lm_head among them). Against the operator's
decode bar (19.3 t/s, a 51.8 ms step) the mixed form is now at 61.0 ms;
what remains, by the same accounting, is the 224-byte layout of
§7.0.2bh (the two-message read shape, 259 → 183–210 µs on the down
projection, about 5 ms per step), the host time per K-quant node
(about 6 ms), the augmentation's packing (1.7) and Q5_K (0.6). The
Q6_K row now reads its block in the same shape the 224-byte layout
would give it, so that layout is a stride and a type id away, not a
new kernel.

Retracted here (§7.0.1): nothing.

#### 7.0.2bj 0.4.1, the native Q6_K rows in 224-byte blocks (patch 0026, `--gguf-q6k`): the long-K decode at the probe's prediction, the prefill up a quarter, and the K = 5,120 shapes untouched (2026-09-07)

The layout §7.0.2bh designed and §7.0.2bi made a stride away: the
native Q6_K rows laid out at load in 224-byte super-blocks — the
file's 210 bytes, then 14 zero bytes — so every block is dword-aligned
and the decode row of patch 0025 takes its even path with no shuffle
at all. On the plugin side a second Q6_K type id (114) whose row is
224 bytes per super-block and whose decoders are the existing ones at
the wider stride; on arcint's side the copy at load
(`--gguf-q6k aligned`, the default; `file` keeps the file's rows and
type 14), 6.7 % more bytes on that set (4,243 → 4,525 MiB, the model
16.26 → 16.54 GiB resident), one memcpy per super-block, and the
correctness case on the fixture's own Q6_K tensor for both settings.
The plugin's correctness set grows by two cases (type 114 at one row
and at nineteen, 3-D).

**Measured** (the streamed timing test, warmed; 16/16 on both cards):

| Q6_K decode, µs | 24 GB card, file rows (0025) | 24 GB card, 224-byte blocks | 16 GiB card, file rows | 16 GiB card, 224-byte blocks |
|---|---|---|---|---|
| down projection, N 5,120 × K 17,408 | 262 | **204** (383 GB/s of the padded bytes, 359 of the file's) | 260 | **247** |
| N 1,024 × K 5,120 | 27 | 27 | 28 | 27 |

The probe's prediction for the long-K shape was 183–210 µs; 204. The
K = 5,120 shape does not move on either card. Served on the 24 GB
card, mixed form, `u8` KV, one fresh process per cell, the file-row
control in the same window:

| mixed form | prefill | decode (64 tokens) | step | Prüfstand | resident, max ctx at `u8` |
|---|---|---|---|---|---|
| 856 tokens, file rows (control) | 418 t/s | 15.5 t/s | 60.8 ms | — | 16.26 GiB, 79k |
| 856 tokens, 224-byte blocks | **531** | 15.5 | 59.8 | 10/10 at **17.0** | 16.54 GiB, 71k |
| 71,727 tokens, file rows (§7.0.2bi) | 291 | 11.9 | 79.9 | — | —, 109k |
| 71,727 tokens, 224-byte blocks | **335** | 10.3 (64 tokens; the step is the figure) | **77.7** | — | —, 101k |

Greedy outputs byte-identical to the file rows' at both depths
(23e06c37e0d6, 086d5e71ad47). Two readings. The decode step moved 1–2
ms where the long-K gain alone (32 down projections × 58 µs) is 1.9:
the layout does what it does for the down projections and nothing for
the other 33 Q6_K tensors of the set, which have K = 5,120 — the
attention-side projections and the lm_head (248,320 × 5,120, 1.04 GB
per step) — and run on the short-K dispatch (4 rows × 4 subgroups),
where the N 1,024 timing above is near the launch floor and says
nothing about the lm_head's rate; that is the next thing to time, on
the served step's device timeline. And the prefill gained a quarter
at 1k and 15 % at depth without a kernel change: the tiled variant's
vector loads of the block were paying for the 2-byte alignment as
well, which no prefill measurement had isolated. Against the
operator's bars the mixed form stands at 40 % of the prefill bar at
1k (531 of 1,341) and 73 % at depth (335 of 460), and at 59.8 ms
against the decode bar's 51.8.

**Correction, the same afternoon, from the served step's device
timeline** (the instrument of §7.0.2bg on the v28 process, 856
tokens, steady state): the K = 5,120 Q6_K shapes did move — the
twenty-four mid-size projections run at 123 µs per launch against 229
under 0024 and the lm_head at 2.56 ms against 5.35 — and the timing
test's N 1,024 shape, at 27 µs, sits at the launch floor and cannot
show it; "did not move" above is retracted. The split of that gain
between 0025 (the shuffles) and 0026 (the layout) is not measured for
those shapes. The step, on the device, under v28:

| per step, 856 tokens, mixed form, v28 | ms | launches |
|---|---|---|
| runtime gemm on the repacked Q4_K set | 30.2 + 0.9 | 272 + 272 |
| K-quant Q6_K: 31 down projections at 239 µs, 24 at 123, the lm_head 2.56, the rest 0.4 | **13.3** (23.7 under 0024) | 65 |
| K-quant Q5_K, 48 at 68.6 µs | 3.2 | 48 |
| everything else | 4.7 | ~1,470 |
| device busy | **52.3** (62.7 under 0024) | 2,125 |
| the untraced step (§7.0.2bj's cell) | 59.8 | |
| idle in the untraced step | **7.5** | |

Against the bar's 51.8 ms the idle is the largest single item left,
and it is host time around the K-quant nodes (§7.0.2bg: 0 nodes
2.3 ms, 113 nodes 8.4); what the host does per execution of our node
is the next instrument.

Retracted here (§7.0.1): §7.0.2bj's own "the K = 5,120 Q6_K shapes
untouched" (above; the served timeline shows 229 → 123 µs and the
lm_head 5.35 → 2.56 ms), and §7.0.2bh's "about 5 ms per step" for
this layout, which was the long-K figure applied to the whole set.

#### 7.0.2bk 0.4.1, the host idle around the K-quant nodes named: the plugin's runtime fusion check rejects our kernel, every fused residual add runs through the unfused-subgraph fallback, and that fallback drains the queue — 79 times per step; patch 0027 (2026-09-07)

§7.0.2bg had the mixed form's step at 62.7 ms of device time in 71.1,
and the idle scaling with the K-quant node count (0 nodes 2.3 ms, 113
nodes 8.4, 401 nodes 14.2) with a per-execution allocation as the
hypothesis; §7.0.2bj's v28 step had 52.3 of 59.8. The instrument that
named it was the intercept layer's host call log on one served step
(`tools/cl_timeline_steps.py --host`): the mixed form issues 20,894
OpenCL calls per step against the IR's 14,809, and among them
**81 `clFinish` against the IR's 2** — 25 ms of the step spent inside
a full queue drain, one after every K-quant projection: the sequence
is the K-quant launch, a `generic_eltwise_ref` launch, a flush, a
finish, the next norm. No allocation call appears per step: the
allocation hypothesis of §7.0.2bg is retracted.

Naming the caller took three instruments, in order: an environment-
gated print at the plugin's two per-node drain sites (the shape-
inference sync and the gather skip check) — zero hits; a thread-local
"current primitive" printed from the stream's own `finish()` — the
residual `Add` of every layer, 80 per forward, plus the graph's
Result; and gdb on a relink of the plugin without `-s`, breaking on
`clFinish` after the load's 1,450 drains — the stack:
`clFinish ← ocl_stream::finish ← primitive_inst::execute ←
network::execute_impl`, through the one inline `finish` in the
plugin's headers: `network_output::get_memory()`, which drains the
stream on an in-order queue, called at the end of the **unfused-
subgraph** path of `primitive_inst::execute`.

The mechanism, read from the source once the stack pointed at it.
The fusing pass fuses the residual add (and the MLP's Swish and
Multiply) into the K-quant node at build, because patch 0021 declared
`FUSED_OPS` support and the kernel carries it on its store. At run
time, for a dynamic node, `primitive_inst::is_valid_fusion()` accepts
a fused eltwise on an OCL fully-connected node only when the selected
kernel's name contains `fully_connected_gpu_bf_tiled` or
`fully_connected_gpu_bfyx_ref` ("only these are verified for fused
eltwise"); ours is `fully_connected_gpu_kquant`, so every K-quant node
with a fused eltwise is declared invalid every step and executed
through `get_unfused_subgraph()`: a separate sub-network of the node
and its fused primitives, whose output memory is read back through
`network_output::get_memory()` — the drain. So the K-quant kernel's
fused path had never executed in any served number of this milestone;
the residual add ran as its own launch (the 79 `generic_eltwise_ref`
of §7.0.2bg's census), and the card idled at each of the 79 drains.
The repack form has no K-quant node and 2.3 ms of idle; the native
form has fused eltwise on 401 nodes and 14.2.

**Patch 0027**: the runtime check accepts the K-quant kernel; the
kernel's fused ops take the output-typed value (the sum rounded to
f16, what the unfused add saw) so the fusion changes no bit; the
correctness set gains five cases with a residual add fused into the
op (both variants, both Q6_K layouts, Q5_K, Q4_K), each checking that
the eltwise is no longer a primitive of its own and that the output
matches the unfused rounding.

**Measured.** The plugin's correctness set 21/21 on both cards (the
five fused cases among them). Served on the 24 GB card, mixed form
with the 224-byte Q6_K layout, `u8` KV, one fresh process per cell,
against §7.0.2bj's cells on the same runtime base:

| mixed form | prefill | decode (64 tokens) | step | Prüfstand | greedy output |
|---|---|---|---|---|---|
| 856 tokens, 0026 | 531 t/s | 15.5 t/s | 59.8 ms | 10/10 at 17.0 t/s | 23e06c37e0d6 |
| 856 tokens, 0027 | 551 | **17.2** | **54.7** | **10/10 at 18.4** | 23e06c37e0d6 |
| 71,727 tokens, 0026 | 335 | 10.3 | 77.7 | — | 086d5e71ad47 |
| 71,727 tokens, 0027 | 341 | 11.1 | **73.3** | — | 086d5e71ad47 |

Byte-identical at both depths: the fused add rounds where the unfused
one did. Five milliseconds off the step at 1k and four at depth, and
the call log of the same served process on the fixed plugin: 1,064 `clFinish` in the whole trace against 6,663 with 0026 — the load's own drains being the same, the per-step count from 81 to the IR's 2.
Against the operator's decode bar the mixed form stands at 54.7 ms
against 51.8 — 17.2 t/s against 19.3 at 1k, 11.1 against 13.8 at
depth — with the augmentation's packing (§7.0.2bg, about 1.7 ms) and
Q5_K's rate (0.6) the items left on the device, and the host's share
now measured rather than inferred. On the timing test's side, a note
for the next reader: the type-114 rows of the first run after this
build read 2.7 ms per launch and 202 µs on the rerun — §7.0.2bd's
cold-cache stall on a freshly compiled kernel, not the kernel.

Retracted here (§7.0.1): §7.0.2bg's per-execution allocation as the
host cost per K-quant node (no allocation call per step; the cost was
the fallback's drain), and its "30–50 µs of host time per K-quant
execution" as a mechanism (it was ~95 µs of drained idle per fused
node); §7.0.2bg's and the handoff's "the MLP's Swish and Multiply are
fused into the K-quant kernel" as a statement about execution (fused
at build, unfused at run time through the fallback); §7.0.2be's
"fused post-op … removes 160 launches" stands corrected in the other
direction: the launches existed because the fusion never ran.

#### 7.0.2bl 0.4.1, the augmentation's packing as an option: exactness has a price and the price is a flag (`--gguf-mins exact|shared|nibble`) (2026-09-07)

The operator's order after §7.0.2bk: start the augmentation's packing,
Q5_K's rate and the tiled variant, and — on the first — "exactness cost
needs to be an option". The augmentation (§7.0.2ba, `core/gguf_repack.h`)
carries each group's 6-bit min as two u4 columns, hi and lo, under the
super-block's own f16 `dmin` as the augmented group's scale, so that
the min term falls out of the same fully-connected exactly; a
super-block's eight groups take sixteen of an augmented group's
thirty-two columns and the other sixteen are zeros: 12.5 % on the
Q4_K set, 1.3 GB per token of the mixed form (§7.0.2bg). One scale
per group of thirty-two columns is the runtime's layout, and two
super-blocks have two `dmin`s, so no exact packing shares an
augmented group. What can be shared is a *chosen* scale, with the
mins that do not divide by it requantised:

- **shared**: two super-blocks per augmented group under the larger
  of their `dmin`s; the other block's mins become `round(mn · dmin /
  s)` in steps of `s`. +6.25 % on the set.
- **nibble**: one nibble per group, thirty-two groups (four
  super-blocks) per augmented group, the scale the largest min of the
  thirty-two at 15; every min is `round(dmin · mn / s)`. +3.1 %.

Both err by at most half the shared scale per group min, which the
repack tests bound per group of every row against the exact form
(`tests/test_gguf_repack.cpp`), and both are priced on the fixture's
Q4_K tensor by the deviation the load reports (in units of the group's
quantisation step; the exact form's bound is 1/64):

| packing | augmented columns | max deviation | rms | values over the exact bound |
|---|---|---|---|---|
| exact (default) | K/8 | 0.013 steps | 0.003 | 0 of 32,768 |
| shared | K/16 | 0.147 | 0.037 | 13,243 |
| nibble | K/32 | 0.717 | 0.212 | 29,273 |

The load reports an inexact packing's deviation in its summary instead
of refusing it (the exact form still refuses any value over its bound),
the verdict cache keys the packing, and the plugin side needs nothing:
the repacked tensor is the same u4-per-32 form with fewer augmented
columns, and the widening matrix follows the slot layout.

**Served** (the 24 GB card, mixed form with the 224-byte Q6_K layout,
`+p10`, `u8` KV, one fresh process per cell; the exact form's cells
from §7.0.2bk):

| mixed form, 856 tokens | resident | max ctx at `u8` | Q4_K set | prefill | decode (64) | step | Prüfstand | greedy output |
|---|---|---|---|---|---|---|---|---|
| exact (§7.0.2bk) | 16.54 GiB | 112k | 11,264 MiB | 551 t/s | 17.2 t/s | 54.7 ms | 10/10 at 18.4 (1,145 tokens) | 23e06c37e0d6 |
| shared | **15.88** | 131k | 10,638 | 559 | 17.4 | 53.7 | 10/10 at 18.8 (1,077) | 23e06c37e0d6 |
| nibble | **15.60** | 140k | 10,379 | 527 | 17.8 | **52.5** | 10/10 at 19.1 (**474** tokens) | 98c732f5e252 |

At 71,727 tokens the shared form serves 345 / 12.8 t/s at a 72.3 ms
step (the exact form 341 / 11.1 at 73.3) with the greedy output
5e3fb7a8d72c against the exact form's 086d5e71ad47 — a near-tie
flipped back to what the template-embedding runs of §7.0.2be produced.
The load reports what the packings cost on the served file: shared, a
maximum of 10.5 quantisation steps with 41 % of the 18.7 G values over
the exact bound; nibble, 15.3 steps. Two readings. The shared packing
is free at the acceptance gate: 10/10, the 1k output byte-identical,
0.66 GiB back, 19k more tokens of context at `u8`, one millisecond off
the step. The nibble packing reaches to 0.7 ms of the operator's
decode bar and passes the acceptance task — but its text is different
from the first token at 1k, and the acceptance run answers in 474
tokens where the exact form takes 1,145: the same score for a
different program. That is the reason the default stays exact and the
inexact forms are flags: the score is a floor, not a fingerprint.

One thing the option cost on the way. The runtime's int4
fully-connected walks K in pairs of groups; the nibble packing's
augmented part left the width with an odd group count (165 at
K = 5,120) and the first served forward faulted in the driver
(`CL_OUT_OF_RESOURCES` on the load's probe). One zero group more when
the count is odd (0.6 %), for every packing, and it serves.

Retracted here (§7.0.1): nothing; the first §7.0.2bg estimate of the
packing's worth ("about 1.7 ms") assumed the exact half, which does not
exist.

#### 7.0.2bm 0.4.1, the tiled variant's activation tile staged in the matrix unit's own layout: one block read per operand instead of eight gathers per lane (patch 0028) (2026-09-07)

The operator's order after §7.0.2bl: the tiled variant's register
spill and its 2D block loads. The ISA of the tiled Q4_K gate kernel
(`tools/igadis.cpp` over the dumps of §7.0.2bh; 1,978 instructions,
151 sends per loop body) says where its time is before any block-load
extension is touched: of the 151 sends, 138 go to local memory, and
128 of those are `load.slm.d16u32` — one 16-bit element per lane per
message. That is the A operand of every matrix multiply assembled lane
by lane from a row-major tile: for each sub-block, half and row group,
eight rows read at `xt[row * 256 + s * 32 + lane]`, eight gathers for
eight rows, against sixteen `dpas` in the same body. §7.0.2bc had
measured the staging and these reads at 28 of the 39.5 ms of a
2,048-row gate launch without splitting them. The spill the kernel
names carry (`SPILL=1216–1472`) has no scratch send in the
disassembly: whatever the compiler reports, no spill traffic runs in
the loop.

The change (v30, patch 0028) is the tile's layout in local memory,
nothing else. The 16-byte chunks each work-item stages are stored not
row by row but in the layout the matrix unit reads: for sub-block s,
half h (k 0–15 or 16–31 of it) and row group g, the eight rows' sixteen
halves contiguous, `[s][h][g][row][lane]`. A chunk of eight consecutive
k of one row lands whole in one such row, so the stores are the same
`vstore4`s at a different address; and an A operand is then one
`intel_sub_group_block_read_us8` of local memory — lane l takes halves
l + 16 t, which is the operand's definition — on a 16-wide subgroup,
or one `intel_sub_group_block_read8` of uints on Xe-HPG's 8-wide, the
same bytes read two halves per lane. Sixteen block reads per loop body
where there were 128 gathers; the decode of the weights, the B operand
and the accumulators untouched, so the arithmetic is unchanged and the
result must be bit-identical to v29's.

**Measured** (the plugin's correctness set 21/21 on both cards; the
streamed timing test, warmed, the gate projection Q4_K 17,408 × 5,120):

| tiled gate launch, µs | 24 GB card, 0023–0027 | 24 GB card, v30 | 16 GiB card, before | 16 GiB card, v30 |
|---|---|---|---|---|
| M = 32 | 656 | **510** | 2,965 | 2,674 |
| M = 256 | 5,030 | 4,717 | 10,400 | 9,152 |
| M = 2,048 | 39,500 | **34,577** | 78,400 | 69,433 |

Served on the 24 GB card, mixed form, `u8` KV, one fresh process per
cell, against §7.0.2bk's cells: prefill **551 → 672 t/s** at 856
tokens and **341 → 385** at 71,727, the decode step unchanged (54.8
and 73.2 ms), Prüfstand 10/10 at 18.4 t/s, both greedy outputs
byte-identical (23e06c37e0d6, 086d5e71ad47). Against the operator's
prefill bar the mixed form stands at 50 % at 1k (672 of 1,341) and
84 % at depth (385 of 460).

So the gathers were a fifth of the launch, not the 28 ms §7.0.2bc
had put on "the staging and the local-memory reads" together. What
the tile costs now is arithmetic and re-reading: with 32 rows per
tile the lane decodes its column of every super-block (the `float
y[32]`, thirty-two converts, the scale pairs) once per 32 rows, and
each 32-row tile reads the whole 44.6 MB gate matrix again — 64
times over a 2,048-row prefill, 2.85 GB of weights for one launch,
against 365 GFLOP that the matrix unit would do in a few
milliseconds. A larger row tile divides both by the same factor and
costs accumulator registers (one register per row of the tile); the
256-register mode was a loss on the decode kernel (§7.0.2bd) because
it halves occupancy on a bandwidth-bound kernel, and is the natural
thing to measure on this one. The sweep, the same window's
instrument, the gate on the 24 GB card, µs — **retracted in §7.0.2bn:
the instrument's activations were in host memory, and this table
timed the bus; the sweep redone in device memory reverses it** —

| rows per tile × GRF | M = 32 | M = 256 | M = 2,048 |
|---|---|---|---|
| **32 × 128 (the kernel)** | 516 | 4,589 | **34,638** |
| 32 × 256 | 538 | 5,046 | 63,545 |
| 64 × 128 | 952 | 8,169 | 111,749 (spills) |
| 64 × 256 | 527 | 4,366 | 43,164 |
| 128 × 128 | 1,825 | 11,091 | 79,275 (spills) |
| 128 × 256 | 1,055 | 12,137 | 90,273 |

Correctness 21/21 at 64 and 128 rows. The tile stays at 32: a larger
one spills at 128 registers and halves occupancy at 256, and neither
the halved decode count nor the halved weight re-reads bought the
difference back. So the re-reads and the per-tile decode *count* are
not what binds; the instruction stream is: the loop body's ~2,000
instructions per sixteen `dpas` are the decode of every value to f32
and its conversion back to f16 before it reaches the matrix unit (the
`float y[32]` and its thirty-two converts per sub-block). The
utilities carry a second form that never reached the tiled path — the
packers (`kq_pack_*`), which build the B operand by a shift, a mask
and an or per value as `f16(1024 + q)` and let the 1024 fall out of
the f32 sums with one extra multiply by ones — at a rounding that is
not the decode path's. That is the next lever, and by the operator's
rule it is an option.

Retracted here (§7.0.1): §7.0.2bc's "28 of the 39.5 ms are the
staging and the local-memory reads" as an attribution to the reads:
removing the gathers took 5 ms; the rest of that figure is the decode
and the weight re-reads per tile, above.


#### 7.0.2bn 0.4.1, the tiled variant's operands: two forms that lose, the instrument that had been timing the bus, the 2D block loads and the 64-row tile at 256 registers (patch 0029) (2026-09-07)

The operator's order after §7.0.2bm: the tiled variant's register
spill and its 2D block loads. Four forms of the inner loop went through
the plugin's timing test and the served endpoint in one afternoon; two
lost, the test turned out to have been measuring the wrong thing since
it was written, and the two that won are patch 0029.

**The packed B operand (a loss, not shipped).** §7.0.2az's form on the
tiled path: the B column as `f16(1024 + q)` straight from the nibbles,
the scale and the offset applied to the sub-block's two sums, the row's
Σx per sub-block half staged beside the tile by the same work-items
(one shuffle per pair of chunks). Correct on both cards (21/21) and
slower on every tiled shape: on the instrument of that hour the gate at
2,048 rows 34.8 → 92.5 ms, the Q5_K attention output 17.0 → 28.3, the
224-byte Q6_K down projection 108.5 → 113.9, the 16 GiB card 69.3 →
131.1 on the gate. The ISA (offline, `ocloc` over the plugin's own jit
source at the 17,408 × 5,120 shape, `tools/igadis.cpp` for the counts)
says why it could not have won and why it lost:

| the tiled kernel, one super-block body, Xe2 | decode form | packed form |
|---|---|---|
| instructions | 2,421 | 2,611 |
| `mov` (the converts among them) | 891 | 470 |
| `mad` | 289 | 553 |
| local-memory messages | 74 | 146 |
| global messages | 86 | 49 |

The packing trades the converts for fmas one for one: what the decode
spends turning 32 values into f16, the packed form spends applying the
scale and the offset to 32 partial sums per row group, and the Σx reads
come on top. And the fmas wait: in the decode form the 64 `dpas` of a
super-block run as eight chains into the same accumulators, which the
matrix unit pipelines; in the packed form every `dpas` starts from zero
and a `mad` reads its result two instructions later (`{$9}` on the
`dpas`, `{$9.dst}` on the `mad`), so each of the 64 pays the systolic
latency in full. The option was built (`--gguf-prefill`, the plugin's
`ARCINT_KQ_PREFILL`) and withdrawn with the numbers; nothing of it is
in the tree.

**The loads hoisted ahead of the decode (a loss, not shipped).** The
same disassembly issues every A block read two instructions before the
`dpas` that waits on it, and the Q6_K body carries 242 global messages
per super-block from its 16-bit gathers. So v33: the A operands read at
the top of the sub-block, the next weight pair (or the aligned Q6_K's
next half) loaded as dwords a sub-block ahead through register
decoders. Correct (21/21 both cards); served on the 24 GB card the
prefill fell 672 → 437 t/s at 856 tokens and 385 → 321 at 71,727, the
tiled launch on the device timeline 5.4 → 9.2 ms. The compiler had
already scheduled what it could; holding sixteen more registers live
through the decode cost more than the latency it hid.

**The 2D block loads (v32).** `cl_intel_subgroup_2d_block_io` on the
24 GB card (the compiler knows the plain reads of 8/16/32 rows × 16
columns at 8, 16 and 32 bits and the transposed 32-bit reads of 16 or
32 rows × 8 dwords; the 16 GiB card has none of them). Both operands go
through it: the A operand is one `2d_block_read_16b_8r16x1c` of the
activations from global memory — eight rows, sixteen halves, lane l
takes k = l, the matrix unit's layout without a tile staged, a barrier
run or a byte of local memory, the work-group's eight subgroups sharing
the block through L1 — and the subgroup's sixteen weight rows come by
`2d_block_read_transpose_32b_16r8x1c`: one message brings 32 bytes of
each of the sixteen rows, lane l its own row's eight dwords; five
messages per super-block for Q4_K, six for Q5_K, seven for the 224-byte
Q6_K, from where the decode runs on registers (`kq_dec_*_w`, the same
arithmetic). Rows past the matrix read as zeros and are dropped where
they always were. The message addresses dwords, so the row pitch must
be a multiple of 16 bytes and the block offsets of 4: Q4_K, Q5_K and
the aligned Q6_K qualify; the file's 210-byte Q6_K rows, Q8_0's 34-byte
blocks and everything on Xe-HPG keep the staged path.

Correct on both cards (21/21). And then the two instruments
disagreed: the timing test put the 2D form at 1.5–2× *slower* on every
shape (the gate at 2,048 rows 35.1 → 65.4 ms), while the served 24 GB
card ran the prefill *faster*, 672 → 766 t/s at 856 tokens and 385 →
410 at 71,727, outputs byte-identical, Prüfstand 10/10; and the served
device timeline (`tools/cl_timeline_steps.py`'s method, the request's
run of launches up to the lm_head slice) put the tiled launches at
5.4 ms (v30) against 4.1 (2D), the Q6_K down projection at 856 rows
11.1 → 7.6 ms:

| the served 856-token prefill, 24 GB card, mixed form | v30 (staged) | v33 (hoisted) | v32 (2D) |
|---|---|---|---|
| device span / busy | 1,369 / 1,221 ms | 1,823 / 1,670 | 1,129 / 1,070 |
| the tiled K-quant launches (112) | 612 ms | 1,059 | 463 |
| of which the Q6_K down projection (32 launches) | 11.1 ms each | — | 7.6 |
| the runtime's int4 gemm on the repacked set (448) | 419 | 419 | 418 |
| gated delta net | 76 | 77 | 76 |

**The instrument.** The timing test allocated its activation rows with
`engine.allocate_memory(layout)`, which on this runtime returns the
*lockable* allocation — host memory — and every tiled figure the test
had produced since §7.0.2bc timed a kernel reading its A operand over
the bus: the staged path once per column work-group, the 2D path once
per subgroup, which is the whole disagreement. With both operands in
device memory (`allocation_type::usm_device`, `copy_from`) the test
agrees with the served timeline to within 5 %, and the absolute figures
are a third of what the record carried:

| the timing test, 24 GB card, device memory, ms | v30 (staged) | 2D |
|---|---|---|
| gate Q4_K 17,408 × 5,120, M = 856 / 2,048 | 5.64 / 13.49 | 5.25 / 12.10 |
| attention output Q5_K 5,120 × 5,120, M = 856 / 2,048 | 1.72 / 3.98 | 1.71 / 3.90 |
| down Q6_K (224) 5,120 × 17,408, M = 856 / 2,048 | 10.88 / 25.12 | 7.22 / 16.91 |

(The M = 1 decode figures of §7.0.2bc–bj are untouched: their
activation is 10 KB and read once.) Retracted here (§7.0.1), as
absolutes: §7.0.2bc's 39.5 ms and §7.0.2bm's 34.6 ms per 2,048-row gate
launch (13.5 in device memory) and the row-tile × GRF table of
§7.0.2bm; the served rates those sections report stand, as do their
directions. The test now carries the served row count (856) and valid
scales in its rows.

**The row tile, swept on the fixed instrument** (2D form, µs, the 24 GB
card; 21/21 at 64 rows):

| rows per tile × registers | gate M = 856 / 2,048 | Q5_K M = 856 / 2,048 | Q6_K M = 856 / 2,048 |
|---|---|---|---|
| 32 × 128 (0028) | 5,262 / 12,099 | 1,705 / 3,907 | 7,175 / 16,914 |
| 32 × 256 | 3,945 / 9,369 | 1,224 / 2,771 | 6,358 / 14,523 |
| 64 × 128 | 4,452 / 10,092 | 1,581 / 3,428 | 7,255 / 16,725 |
| **64 × 256** | **3,371 / 7,229** | **1,043 / 2,213** | **4,412 / 9,964** |
| 128 × 128 | 17,206 / 40,134 | 6,642 / 18,492 | 26,895 / 73,736 |
| 128 × 256 | 6,870 / 16,428 | 973 / 1,870 | 9,239 / 19,218 |

The opposite of the retracted table, and the reason §7.0.2bm gave for
the sweep was right: the tile divides the per-tile decode and the
weight re-reads, and the accumulators are what it costs. 64 rows at
128 registers spill; 64 at 256 halves nothing that matters on a kernel
that is not bandwidth-bound (the decode kernel, which is, lost 15–45 %
in the same mode, §7.0.2bd, and keeps 128: the tiled kernel is compiled
in its own batch with `-cl-intel-256-GRF-per-thread`). The staged path
gains the same way (Q6_K at 856 rows 10.9 → 5.97 ms at 64 × 256), so
the mode is not a property of the 2D loads.

**Shipped (patch 0029):** the 2D loads on Xe2 for the aligned layouts,
64-row tiles in the 256-register mode on Xe2; Xe-HPG at 32 × 128 until
measured there (below). Served on the 24 GB card, mixed form, `u8` KV,
one fresh process per cell:

| mixed form, 24 GB card, `u8` KV | 0028 (§7.0.2bm) | 0029 at 32 × 128 (v32) | **0029 shipped, 64 × 256** | the bar |
|---|---|---|---|---|
| prefill, 856 tokens | 672 t/s | 766 | **907** | 1,341 (68 %) |
| decode step at 1k (64 tokens) | 54.8 ms | 54.8 | 54.9 | 51.8 |
| prefill, 71,727 tokens | 385 | 410 | **451** | 460 (98 %) |
| decode step at 71.7k | 73.2 | 73.0 | 73.3 | 72.5 |
| Prüfstand | 10/10 at 18.4 t/s | 10/10 at 18.5 | 10/10 at 18.4 | 10/10 |
| greedy outputs, 1k / 71.7k | 23e06c37e0d6 / 086d5e71ad47 | the same | the same | — |

The prefill bar at depth is met within 2 %; at 1k the gap is 32 %, and
the served timeline of the 2D form at 32 × 128 says what it is made
of: 1,070 ms of device time for 856 tokens, of which the tiled
launches were 463 (now about 300 at 64 × 256), the runtime's int4 gemm
on the repacked Q4_K set 418, the gated delta net 76, everything else
110 — the repacked set's gemm is now the larger half of the prefill,
and it is the runtime's kernel, not this one.

The 16 GiB card (Xe-HPG, no 2D block loads, the staged path) had its
own sweep on the fixed instrument: at 32 rows the 256-register mode
takes the gate at 856 rows 10.3 → 8.0 ms, Q5_K 3.18 → 2.73, the Q6_K
down projection 14.8 → 12.9; 64 rows at 256 wins the first two (6.8,
2.20) and loses the third (18.3), so it keeps 32 rows and gains the
mode (v37: 21/21, the timing test at 8.0 / 2.73 / 12.9 ms; its M = 1
decode figures unchanged, that kernel is compiled at 128).

**The native form on 0029.** With the tiled kernel now running the
gate at 3.4 ms per 856 rows (45 TFLOPS, against the runtime's int4
gemm at about 50 on the repacked set), the handoff's first question
was the form with no repack at all — every projection through the
K-quant kernel, no augmented columns, no min term in the runtime's
kernel. Served on the 24 GB card, the same cells:

| 0029, 24 GB card, `u8` KV | mixed (default) | native | the bar |
|---|---|---|---|
| resident | 16.54 GiB | **15.22** (max ctx 155k) | — |
| prefill, 856 tokens | **907** t/s | 662 | 1,341 |
| decode step at 1k | 54.9 ms | **51.3** | 51.8 |
| prefill, 71,727 | **451** | 395 | 460 |
| decode step at 71.7k | 73.3 | **70.3** | 72.5 |
| Prüfstand | 10/10 at 18.4 t/s | 10/10 at 19.8 | 10/10 |
| greedy outputs | 23e06c37e0d6 / 086d5e71ad47 | the same | — |

The native form meets the operator's decode bar at both depths and
the size by 1.3 GiB more, and pays for it at prefill (49 % and 86 % of
the bars against the mixed form's 68 % and 98 %): the repacked set's
gemm is still faster than the tiled kernel on Q4_K by about the
difference. The default stays mixed, the prefill gap being the larger
of the two; `--gguf-mode native` is the flag for a deployment that
wants decode and context over prefill. The outputs of the two forms
agree byte for byte at both depths, as §7.0.2be found for the
template-embedding runs.

Retracted here (§7.0.1): §7.0.2bm's "the kernel is instruction-issue
bound (~2,000 instructions per sixteen dpas)" as the mechanism — the
2,421 instructions per super-block would issue in about 9 ms of the
gate's 13.5 at 2,048 rows, and the packed form showed the count is not
what moves it; what the sweep says binds is the per-tile decode and
re-read count, divided by the tile.

#### 7.0.2bo 0.4.2, the runtime's int4 gemm on the repacked set: at the card's f16 rate, the int8 form measured dead at 32-wide groups, and the min term as a separate term (`--gguf-mins split`) (2026-09-07)

The 0.4.1 record left the 856-token prefill of the mixed form at 907
t/s against the operator's bar of 1,341 (§7.0.2bn) and named the
runtime's int4 gemm on the repacked Q4_K set as the larger half of it:
418 ms of the prefill's 1,070 ms of device time, the runtime's own
kernel. This section is that kernel on the device timeline per shape,
its disassembly beside the kernel Intel's int4 IR of the same model
runs, the int8 form of it measured, and the one host-side lever that
was left -- which turned out to be a decode lever.

**The instrument, per shape.** The intercept layer's kernel-name
tracking now carries the global and local work sizes
(`CLI_DevicePerformanceTimeGWSTracking`, `...LWSTracking`), which
splits the runtime's `gemm_kernel` launches by shape where the name
alone did not, and its transfer tracking marks each logits copy, on
which the prefill parser anchors: the request's prefill is the run of
launches between the previous logits copy and the 65th from the end of
a 64-token run (the load's probe forwards copy logits too, and the
request can follow the load by less than the one-second host gap the
earlier parser looked for -- it did on the IR, and the first table of
the day was the load's; a gap anchor inside the run then lost the
first layers of a request that compiled kernels). The kernel ISA
binaries are dumped in the same run (`CLI_DumpKernelISABinaries`),
and `tools/igadis.cpp` now counts `dpas` and writes the disassembly
text beside the binary (`IGADIS_DUMP=1`). Four cells on the 24 GB
card, arcint 0.4.1 on `+p11`, `u8` KV, chunk 2,048, one fresh process
each, the 856-token prompt, traced (traced rates are for shares; the
record's rates are untraced):

| the 856-token prefill, device time | the IR (int4, dyn-quant on) | mixed exact | mixed shared | mixed nibble |
|---|---|---|---|---|
| busy | 503 ms | 894 | 871 | 828 |
| the runtime's gemm | 325 (65 %) | **420** (47 %) | 397 | 390 |
| the tiled K-quant launches | -- | 285 | 285 | 271 |
| gated delta net | 78 | 76 | 77 | 77 |
| everything else | 100 | 113 | 112 | 90 |
| greedy output | 50815ed40613 | 23e06c37e0d6 | 23e06c37e0d6 | 98c732f5e252 |

The gemm follows K: the shared packing's 5.6 % shorter width takes
5.5 % off it, the nibble's another 2 %. Per launch at 856 rows, from
the launch sequence of one layer: the IR's gate and up projections
(17,408 × 5,120) at 1.08 and 1.26 ms, dispatched as 952 work-groups
of 512 work-items; the mixed form's (17,408 × 5,760) at 1.92 and 2.06
ms, dispatched as 160 work-groups of 128 -- 640 subgroups, a
persistent grid that walks 15 output tiles each. In FLOP terms the
mixed form's gate launch runs at 86 TFLOP/s, the IR's at 130.

**The two kernels, disassembled.** Both are oneDNN's generated gemm.
The IR's runs `dpas` on int8 operands into int32 accumulators (its
activations quantised per token and 128-wide group by the runtime's
dynamic quantisation, 14 ms of its prefill; its weights u4 with a u4
zero point per 128-wide group); ours runs `dpas` on f16 operands into
f32 (the activation f16, the u4 weights decompressed in the loop under
the runtime's f16 math mode -- the plugin sets `fpmath_mode::f16` for
an f16 activation on compressed weights, and the catalog's plain
f16 × f16 entry serves it). One `dpas.8x8` occupies the matrix unit
for 16 cycles on this part, 2,048 f16 or 4,096 int8 multiply-adds,
which is where the card's 98 TFLOP/s f16 and 197 int8 peaks come from
(160 vector engines at 2.4 GHz). The K-loop of each kernel, read
instruction by instruction:

| the K-loop, one subgroup | ours (f16 × u4/32, no zero point) | the IR's (int8 × u4/128 + zero point) |
|---|---|---|
| register tile | 48 rows × 32 columns (12 accumulator ranges, f32) | 64 × 16 (8 ranges, int32) |
| K per iteration / `dpas` per iteration | 128 / 96 | 128 / 32 |
| decompression per 32 weights | `and`, `shr`, three `mul` (two rescale the nibble read as an f16 denormal, one applies the group scale) -- on the weights, before the `dpas` | `and`, `shr`, one `add` (the zero point); the scales applied to the int32 accumulators once per weight group: `mov`, `mul` (the activation scale), `mad` (the weight scale) per accumulator register |
| non-`dpas` instructions per `dpas` | 7.4 | 14.0 |
| matrix-unit cycles / issue slots per iteration | 1,536 / 810 | 512 / 480 |
| loads | 2D block loads for both operands, six prefetches, no local memory, a barrier per iteration | the same, five prefetches, no barrier |
| bound by (a reading of the counts, not a counter) | the matrix unit: 810 slots against 1,536 matrix cycles (88 % of the f16 roof measured) | issue, on the reading that a `dpas` also takes an issue slot and the drain's three-deep dependency serialises (66 % of the int8 roof measured; the count alone, 480 against 512, would say the matrix unit) |

Ours is at the f16 matrix unit's rate: the decompression is free
beside it (1.9× issue headroom), and the 12 % it misses is the
barrier, the tile switches and the epilogue. Nothing host-side changes
that at f16 -- not the zero point (free either way in this kernel),
not u8 weights (fewer instructions the loop does not need, twice the
bytes), not folding the rescale into the scale. The IR's kernel has
twice the roof and spends a third of its issue slots draining int32
accumulators into f32 through the per-group scales, three
instructions per accumulator register per weight group. That cost
scales with the group count, and the exact repack's groups are 32
wide -- the K-quant's own sub-blocks; a Q4_K value has no exact
expression under a wider scale (§7.0.2ba). At 32-wide groups the
drain is four times the IR's and the int8 kernel is issue-bound at
about 124 multiply-adds per cycle per engine -- under the f16 kernel's
128.

**Measured.** The exact mixed form with `--dyn-quant on`, one traced
cell: the runtime switches the gate to the int8 class (`dpas` on u8
weights and s8 activations; 512 `dpas` with 4,101 `mad` and 5,130
`mov` per program against the IR's 2,571 and 3,710) and the launch
takes 1.91–2.66 ms where the f16 kernel took 1.92–2.06: the gemm 413
ms against 420, the dynamic quantisation 12 ms on top, the greedy
output §7.0.2bg's (3cba6c3128b2, 2/10 there). The disassembly named
one int8 variant that cell had not covered -- the activation scale per
token rather than per token and group takes the `mul` out of the
drain and would issue at 1.4× the f16 kernel if the runtime selected
it -- so the engine can now ask for the runtime's group-size hint
(`ARCINT_DYN_QUANT_GROUP=max`, a measurement switch; the runtime's own
default for a hybrid linear-attention model is 128): on the split
form below, the per-token setting serves at 709 t/s against the
128-wide setting's 677 and the f16 form's 668, the gemm 411 ms
traced. Dead, and with its mechanism measured rather than narrated:
the handoff's lever 2, int8 activations with the min term kept exact,
cannot pay on this runtime's kernels with 32-wide groups, whatever is
done about the min term. Retracted here (§7.0.1): §7.0.2bg's "the
share `--dyn-quant on` could halve" as an expectation.

**The lever that was left: the min term as a separate term
(`--gguf-mins split`).** The exact repack carries each group's min as
augmented columns of the same tensor -- K widened by an eighth (5,120
→ 5,760), the activation widened by its group sums through a small
matmul and a concat (§7.0.2ba). The gemm follows K, so the
augmentation is 11 % of it, and the widening's own launches (the
reduce, the matmul, two concats per distinct activation) another 16
ms. The split form keeps the main columns as they are (K = 5,120, the
same u4 values and f16 scales byte for byte) and carries the min term
as a second fully-connected on the group sums: `y = MatMul(x, W_main)
+ MatMul(sums(x), -M)` with `M[g][n] = f16(dmin_sb(g) · mn)`, one f16
value per group per row -- the same bytes as the augmented columns
(K/16 per row either way), one rounding of the min product where the
augmented form stores the integer under `dmin` exactly. That rounding
adds up to 2^-11 of the min, in steps, to the exact form's deviation
-- a term nothing bounds a priori (the min can be many steps where
the group's scale is small), so the split form's bound is a measured
one, not a derived one: twice the exact bound, which holds on the
fixture's Q4_K tensor (0.0158 steps against the exact form's 0.0133
and its 1/64; one value of 32,768 over 1/64, none over 2/64) and on
the served file, and which the load refuses over. The served
computation emulated on the host agrees with the f32 reference at
1.44× the exact form's deviation (`tests/test_gguf_repack.cpp`,
`tests/test_gguf_graph.cpp`) -- the emulation assumes two
f16-rounded partial products added in f16; the plugin may fuse the
add into either fully-connected as a post-op on its f32 accumulator
instead, which is one rounding fewer and is not measured. It is
exact-class: the mins are the file's, refused over the bound like the
exact form, not reported like the shared and nibble packings. On the
served file the load reports 0.0187 steps, none over the bound (the
exact form 0.0141).

**Served** (the 24 GB card, `u8` KV, chunk 2,048, `--mtp off`, one
fresh process per cell, the 0.4.2 tree on `+p11`; the exact form's
control in the same window):

| mixed form | exact (control) | **split** | split + dyn-quant (group 128) | split + dyn-quant per token |
|---|---|---|---|---|
| resident / max ctx at `u8` | 16.54 GiB / 111,776 | **16.30 / 120,240** | 16.30 / 118,832 | 16.30 / 118,832 |
| prefill, 856 tokens, first request | 824 t/s | 668 | 677 | 709 |
| decode step at 1k (64 tokens) | 55.05 ms | **53.53** | 53.36 | 53.43 |
| prefill, 71,727 tokens | 451 (§7.0.2bn) | **458** | -- | -- |
| decode step at 71.7k | 73.3 (§7.0.2bn) | **72.17** | -- | -- |
| Prüfstand | 10/10 (§7.0.2bn) | 10/10 at 18.6 t/s | 10/10 at 18.7 | 10/10 at 18.7 |
| greedy output, 1k | 23e06c37e0d6 | **23e06c37e0d6** | 23e06c37e0d6 | 23e06c37e0d6 |
| greedy output, 71.7k | 086d5e71ad47 | 5e3fb7a8d72c | -- | -- |

Three readings. The decode step gains 1.5 ms at 1k and 1.1 at 71.7k
(the bar is 51.8 / 72.5: the mixed form now 1.7 ms over at 1k and
under at depth), 0.24 GiB come back and with them 8.5k tokens of
context at `u8`; the 1k output is byte-identical to the exact form's,
the 71.7k output is the shared packing's (§7.0.2bl: the near-tie that
flips back to the template-embedding runs' text). Second, the split
form makes `--dyn-quant on` serve 10/10 with the 1k output
byte-identical -- the augmented columns were what broke it in
§7.0.2bg -- for no rate, which closes that lever on its measured
mechanism. Third, the first-request prefill is slower, and the device
timeline says where: 1,240 ms of span for 877 ms busy (the exact form
959 / 894). On the device the split form is neutral -- the main gemm
loses 41 ms at K = 5,120 (308.7 against 349.8 over the same 232
launches), the min term's 200 f16 gemm launches at 0.2 ms (`[856 ×
160] × [160 × N]`, 24 TFLOP/s) put 40 back, the widening's launches
are gone (−16), everything else the same: the min term costs the same
40 ms in either form, inside the main gemm at its rate on four times
the FLOPs (640 augmented columns against 160 sums) or beside it at
little over a quarter of the rate. The 300 ms of host
idle are the first request's, and so is a part of every 1k prefill
figure on this record: three 856-token requests to one process, no
prefix cache, same window --

| 856-token prefill, one process | request 1 | request 2 | request 3 | decode step (requests 2, 3) |
|---|---|---|---|---|
| exact | 826 t/s | 948 | 947 | 54.0 ms |
| **split** | 709 | **967** | **967** | **53.3** |

The first request of a process compiles the runtime's gemm kernels
for the request's row count (the load's probe forwards run other row
counts), about 130 ms of it in the exact form and 320 in the split
form with its second fully-connected per projection by the served
rates above (the traced first requests put the host idle at 65 and
363 ms -- the two instruments agree on the difference, 180–200 ms,
better than on the absolute); from the second
request on the split form prefills 2 % faster than the exact one and
decodes 1.3 % faster, byte-identical. Every 1k prefill rate in
§7.0.2ba–bn was a first request of a fresh process -- the method the
cells were pre-registered with, kept because it is what a fresh
deployment serves first -- and the 907 t/s of §7.0.2bn carries about
120 ms of that; against the bar it is 948 warm. Recorded here, not
corrected there. The warm split prefill on the device timeline
(the second request of a traced process): 896 ms of span for 869
busy, 27 ms of host idle, the same launches at the same durations as
the first request's -- the compilation is all the difference.

**What this closes.** The runtime's int4 gemm on the repacked set is
at the card's f16 rate; the 2× the IR has over it is int8 arithmetic
that the K-quant's 32-wide groups make unprofitable on this runtime's
int8 kernel, measured three ways (group 128, per token, and the
augmented form). The handoff's 0.4.2 gate -- 1,100 t/s at 856 tokens
on the mixed form through this gemm -- is not reachable by any
host-side form of the projection, and the record says so instead of
a narrower gate: what the prefill has left is the tiled K-quant
share (285 ms, 0.4.3) and the 100 ms of everything else (the 48
per-layer 4.7 MB host writes at 17 ms among them, the same in the IR).
The split form ships as a flag with its prices measured: warm, it is
faster than the exact form at everything measured (prefill +2 % at
1k, +1.5 % at 71.7k, the decode step −1.3 % and −1.5 %), 0.24 GiB
smaller, exact-class, byte-identical at 1k; what it costs is 180 ms
more on the first request of a process and a different 71.7k text (a
near-tie, §7.0.2bl). The default stays exact in this increment, and
moving it is the operator's call (§7.0.2bl's rule: the score is a
floor, not a fingerprint). Also fixed on the way: `--help` had never listed
`--gguf-mode`, `--gguf-mins` or `--gguf-q6k` and still described the
retired `--gguf-native` behaviour.

#### 7.0.2bp 0.4.3, the tiled variant's activation reads 32 rows at a time (patch 0030): the lever the record had not named, the ones it had measured against it, and the tiled share on the served prefill (2026-09-08)

The handoff's list for the tiled K-quant kernel after §7.0.2bn was the
work-group size, the 2D prefetch builtins, a per-type row tile and
three small items, with a gate of 30 % off the tiled launches. The
design pass counted the 2D path's messages from the source before
touching any of them and found the lever the record had not named: on
a 64-row tile a subgroup issues five to seven transposed weight
messages per super-block and 128 activation messages -- eight
sub-blocks × eight row groups × two 8-row reads -- and the 2D block
reads exist in 16- and 32-row forms whose destination holds the rows
in order, so one 32-row read is four row groups' operands unchanged.
Everything went into one build behind measurement knobs and one
timing window decided it.

**The sweep** (the 24 GB card, the plugin's timing test in device
memory, warm, µs per launch at 856 / 2,048 rows; 0029's constants are
the first row; Q5_K on its 128-row tile throughout, the one change
that was measured before the window, §7.0.2bn; correctness 22/22 on
both cards and 22/22 on every knob before a number was read):

| tiled kernel, Xe2 | gate Q4_K 17,408 × 5,120 | Q5_K 5,120² | Q6_K-224 down 5,120 × 17,408 | small Q6_K 1,024 × 5,120 (856) |
|---|---|---|---|---|
| 0029 (8-row A reads, 8 subgroups) | 3,373 / 7,178 | 969 / 1,864 | 4,400 / 9,973 | 295 |
| 16-row A reads | 3,082 / 6,600 | 1,492 / 3,055 | 4,191 / 9,484 | 274 |
| **32-row A reads** | **2,898 / 6,320** | 978 / 1,965 | **3,686 / 8,677** | **237** |
| 32-row, 16 subgroups per work-group | **2,859 / 6,158** | 990 / 1,983 | **3,585 / 8,216** | 244 |
| 8-row reads, 16 subgroups (Q5_K's shipped form) | 3,480 / 7,693 | 985 / 1,891 | 4,404 / 9,821 | 296 |
| 32-row, 4 subgroups | 3,053 / 6,971 | 977 / 1,963 | 3,809 / 9,102 | 245 |
| the next super-block's weight rows prefetched (8 subgroups, 8-row reads) | 4,016 / 8,737 | 2,322 / 4,861 | 4,516 / 10,141 | 303 |
| the same at 32-row reads | 2,822 / 6,042 | 1,020 / 2,023 | 3,725 / 8,801 | 254 |
| the next sub-block's activations prefetched (32-row reads) | 3,178 / 7,048 | 1,142 / 2,499 | 3,933 / 9,606 | 240 |
| both prefetches, 16 subgroups, 32-row reads | 2,793 / 5,999 | 1,002 / 2,130 | 3,736 / 8,683 | 250 |
| the two matrix-unit calls split over the row groups (8-row reads) | 3,037 / 6,665 | 1,856 / 3,848 | 4,085 / 9,616 | 254 |
| the staged path (`ARCINT_KQ_2D=0`), 4 / 8 / 16 subgroups | 13,387 / 3,401 / 3,254 | 3,942 / 1,067 / 1,021 | 20,493 / 5,934 / 5,961 | 891 / 365 / 413 |

The 32-row read is the lever: −14 % on the gate, −16 % on the Q6_K
down projection, −20 % on the small Q6_K, at 856 rows, −12 % and
−13 % at 2,048; the 16-row form wins less and loses half on Q5_K (a
reading: its 128-row tile holds sixteen row groups, and no counter was
taken). Sixteen subgroups per work-group -- by the source, the number
of subgroups that read the same activation block on a path with no
local memory and no barrier; a reading, not a counter -- take the Q6_K
down projection another 4 % and the gate 1 % at the tall read, and
alone cost the gate 3 %; four lose. The weight prefetch one super-block ahead,
the decode kernel's Q6_K win of §7.0.2bd, loses on every tiled type
before the tall read (Q5_K 2.4×) and after it gains 2–3 % on the gate
while costing Q5_K 4 %; the activation prefetch loses everywhere; the
split of the two matrix-unit calls is inert under the tall read and
1.9× slower on Q5_K without it. Q5_K itself: the 128-row tile gives it
7 % at 856 rows and 16 % at 2,048, and the tall read gives it nothing
(+1 % / +5 %), so it keeps the 8-row read on its tile. Shipped: 32-row
reads for Q4_K and the 224-byte Q6_K, 8-row for Q5_K, sixteen
subgroups on Xe2, no prefetch; the staged path (Xe-HPG) unchanged by
construction and re-timed in the window. Not shipped and recorded: the
gate's 2.3 % from both prefetches at sixteen subgroups -- a type that
is not on the served default's tiled path, from a lever that costs the
two that are.

The shipped build, re-timed in its own window (the sweep's cells are
the knob build's): gate 2,859 / 6,163, Q5_K 987 / 1,891, Q6_K down
3,590 / 8,229, the small Q6_K 250; the file-layout Q6_K (type 14, the
staged path on Xe2, which keeps 8 subgroups) 9,918 at 856 rows, as at
0029. Against the handoff's timing gate -- the Q6_K down projection at
856 rows ≤ 3.1 ms and the gate ≤ 2.4 -- the window ends at 3.59 and 2.86:
−18 % and −15 % where 30 % was asked. The 30 % was the record's reading
of what the per-tile decode and re-read count could give divided by
the tile; the sweep says the activation traffic was the larger part of
what remained, and after it the kernel's next structural cost is the
one §7.0.2bn named last: the weights decoded once per 64-row tile,
fourteen times at 856 rows, which a different accumulator scheme would
have to remove.

**Served** (the 24 GB card, arcint 0.4.2, the exact mixed form, `u8` KV,
chunk 2,048, `--mtp off`, one fresh process per cell, three 856-token
requests to a process -- the first compiles the request's kernels,
§7.0.2bo -- and one 71,727-token request; the plugin staged as 0029
and as 0030 on the same runtime; the 0030 column is the shipped build's
own run, the review-fixed one):

| exact mixed form | 0029 (`+p11`) | **0030** |
|---|---|---|
| prefill, 856 tokens, first request / warm | 903 / 940 t/s | 962 / **1,001** |
| decode step at 1k (64 tokens) | 54.8 ms | 54.8 |
| prefill, 71,727 tokens | 451 (§7.0.2bn) | **464** |
| decode step at 71.7k | 73.3 (§7.0.2bn) | 73.7 |
| Prüfstand | 10/10 (§7.0.2bn) | 10/10 at 18.4 t/s |
| greedy outputs, 1k / 71.7k | 23e06c37e0d6 / 086d5e71ad47 | the same |
| the 16 GiB card's timing test (staged path, 32 × 256) | 8.0 / 2.73 / 12.9 ms | 8.01 / 2.73 / 12.92 |

The warm 856-token prefill crosses 1,000 t/s on the exact form (the
split form of §7.0.2bo was not run on 0030; its 2 % is an
extrapolation); the decode step does not move, as it must -- the
decode kernel is untouched, and every cell of the sweep timed M = 1 on
every shape for a knob that leaked into it: 216 / 85 / 204 / 20 µs
throughout, within 1 %.

**What this closes.** The tiled share of the served 856-token prefill was 285 ms at
0029 (§7.0.2bo); the 0030 run was not traced, and scaling the 0029
shares by the timing test's per-type gains puts it near 245 -- with
one open item in that arithmetic: the served step's 47 small Q6_K
launches run 1.44 ms each on the 0029 timeline, and the 1,024 × 5,120
shape the timing test gained for them runs 0.3, so they are another
shape, not identified. The served wall clock gives 56 ms back at 856
tokens. The prefill's other parts are what they were: the runtime's
gemm at the f16 rate (§7.0.2bo), the gated delta net, the hundred
milliseconds of everything else. Against the operator's bar of 1,341
t/s the exact form stands at 75 % warm (68 % at the tag of 0.4.1,
first requests); at depth, where 0.4.1 stood at 98 % of the 460 bar,
0030's 464 is the first form to cross it. The kernel's next structural
cost is named above and is not a knob; the handoff's 0.4.4 (the Q5_K
decode rate) is the decode side's remainder. Patch 0030 ships in
`marfrit-openvino +p12` (patches 0003–0030), the runtime floor of
0.4.3.

Retracted here (§7.0.1): the 0029 review's small item, carried in the
internal notes after §7.0.2bn, that "Q5_K's six weight messages could
be five" -- 44 dwords in 8-dword messages is six, and Q4_K's 36 in five
and Q6_K's 53 in seven are the floor too; the item never reached this
record as a claim and is closed here so that it cannot.

#### 7.0.2bq 0.4.4, the Q5_K decode row: bit-identical with three fifths of its integer work gone and not one microsecond faster; the dispatch swept; the timing test's own step change read from the served traces (2026-09-08)

The handoff's third point release was the Q5_K decode rate: the
decode variant (M = 1, the vector unit, lanes along K, §7.0.2bc)
streamed Q4_K at 355 GB/s and the 224-byte Q6_K at 379, Q5_K at 316
(60 µs on the 5,120² shape of the timing test -- the test's figures of
§7.0.2bc–bj, before the instrument change below), and the three levers
were the header and high-bit reads merged, the fifth bit extracted per
sub-block instead of per value, and the dispatch. Three things had to
be settled before any of them, and the first was the number itself.

**The instrument, again.** The timing test's one-row figures moved
between two builds of 2026-09-07 -- the Q4_K gate 141 → 217 µs, Q5_K
60 → 85, the 224-byte Q6_K unchanged at 203 -- at the build that put
the test's operands in device memory (§7.0.2bn) and shipped 0029. The
served device timeline says the kernel did not: the decode-variant
launches of the mixed form are the same in the 0028, 0029 and 0030
traces to the microsecond (the Q6_K down projection 0.240 ms, the
small Q6_K 0.069, Q5_K 0.129, every one compiled at 128
registers by its name), and a build with the tiled kernel's
256-register mode forced off for the whole primitive leaves the
test's rows where they are: shipped / forced to 128 / shipped again,
the gate 216.0 / 215.6 / 218.8 µs, Q5_K 88.1 / 84.5 / 81.9, the 224-byte
Q6_K 203.5 / 201.2 / 203.5 -- Q5_K's own 7 % spread across the three is
the argument. So the test's one-row figures since 0029 are the
instrument's own state -- its weights' placement, or the valid scales
its rows gained, a reading -- and they drift between days by a tenth
on the gate and a fifth on Q5_K (193 / 69 on 2026-09-08); the served
launch is the gate and the test compares forms on one instrument.
Read from the served trace, the Q5_K tensor of the mixed form is
10,240 × 5,120 (36 MB of rows), about 32 launches a step by the
per-hash table of the traced request (an earlier table of §7.0.2bg
counted 23 in a shorter window) at 0.129 ms: 279 GB/s, 4.1 ms of the
54.8 ms step. At the
224-byte Q6_K's 325 GB/s that is 0.6 ms back; at the 453 GB/s
random-read ceiling, 1.6. The bar (51.8) is 3.0 away: this row
cannot reach it alone, and this section does not claim it can.

**The row, read from the code.** Q5_K's decode row is Q4_K's
function with the fifth bit on: a uniform 16-byte header load per
lane, one `uc8` block read for the 128 nibble bytes (lane l takes
bytes 48 + 16i + l), one `uc2` for the 32 high-bit bytes (bytes 16 + l
and 32 + l), then per value a shift, an and, a shift and an or to
place the fifth bit, and the same sixteen fmas Q4_K runs. The
handoff's first lever, merging the header and the high-bit reads, was
rejected by derivation before any build: the header must be resident
in every lane (the scale decode indexes all three scale dwords), so
any block read of it costs broadcasts and shuffles to save one
message that moves no bytes the `uc2` does not already pull; and 176
bytes over 16 lanes is 11 per lane, for which no block read exists.
The second lever was built: with the lane's eight nibble bytes as two
dwords and its two high-bit bytes as one 16-bit word, two masked
shift-or pairs per dword place all four fifth bits (the mapping
proved from the old code and checked by a 10,000-draw model before
the edit), the integer operations per row per super-block 80 → 31,
Q4_K's 16 → 6, and not one floating-point operation or its order
touched. It is bit-identical: a build with the two masks swapped
fails exactly the type-13 cases (the red case), the 22 cases pass on
both cards, and the test's row dump for the old row and the new one on
the same inputs (one row, K = 5,120, N = 256, f32; two processes, the
old row kept under a jit constant) compares byte for byte, types 12
and 13, both cards.

And it is not faster. The timing test, one row, the old row against
the new in the same binary, warm, both cards:

| M = 1, µs | the 24 GB card, old / new | the 16 GiB card, old / new |
|---|---|---|
| gate Q4_K 17,408 × 5,120 | 193.2 / 193.0 | 142.3 / 141.8 |
| Q5_K 5,120² | 69.1 / 71.3 | 68.4 / 68.1 |
| Q6_K-224 down 5,120 × 17,408 (the row is not this one) | 204.1 / 203.1 | 240.9 / 242.7 |

Sixty per cent of the row's integer instructions gone and the launch
where it was: the decode row is not issue-bound, on either
architecture, for either type. The dispatch sweep on the 24 GB card
(type 13, rows per work-group × subgroups, the same shape):

| rows × subgroups | 2 | 4 | 8 |
|---|---|---|---|
| 4 | 100.5 | **69.2** | 85.7 |
| 8 | 107.2 | 71.3 | 88.3 |
| 16 | 81.9 | 71.5 | 88.7 |

The shipped 4 × 4 is the optimum, against the design's pre-registered
expectation of 8 rows slightly ahead; read from the matrix, not
counted: more rows per subgroup amortise a few per cent of shared
work and lose it to the tail, more subgroups lose to the reduction.
With the integer work measured free, the derivation that rejected
the first lever no longer held on its cost side, so it was built too,
in two forms behind a knob: the header by one dword block read and
four broadcasts instead of the uniform 16-byte load (one message
fewer if the uniform load was sixteen), and the two high-bit bytes
taken from that same read by two shuffles instead of their own
message. Both bit-identical to the old row (the same dump comparison,
both types); both at the old row's time on the 24 GB card: Q5_K 69.3 /
69.1 / 69.1 µs for the three forms and 69.2 / 69.2 / 72.5 on the
repeat, the gate 193 / 193 / 194 and 193 / 192 / 192 -- the scatter
that makes a null a null. Messages do not bind it either.

**Served.** The exact mixed form on the 24 GB card, the decode
per-hash table of the traced request: Q5_K 0.129 ms on the new row
as on the old, the Q6_K launches within the trace's scatter (0.239 and
0.065 against 0.240 and 0.069), the greedy output 23e06c37e0d6 at 1k
in every cell; the split form's step 53.4 ms (53.5, §7.0.2bo).

**What this closes.** Three levers, three measured nulls, one instrument correction, and
no patch: the fifth-bit rewrite is bit-identical and changes nothing
a user can see, so it is recorded here and not carried (the campaign's
rule); the dev tree is back at 0030 and `+p12` stays the runtime
floor. The Q5_K decode row runs at 279 GB/s on the served shape
against the 224-byte Q6_K's 325 and the card's 453, and what binds it
is none of the three things the handoff named -- the reading that is
left, from §7.0.2bh's Q6_K work on the same card, is the read shape:
the 176-byte block delivers its bytes to the lanes in three shapes
(a uniform 16, a 2 per lane, an 8 per lane) where the 224-byte Q6_K
layout of 0026 delivers dwords -- the reads without shuffles (0025)
were worth 400 → 259 µs on that row and the layout (0026) the rest of
the way to its probe's prediction. A Q5_K layout in the shape the card's read probe of
§7.0.2bh names (a "type 113" beside 114, the same load-time
re-blocking) is the next lever, and it is a layout change with its
own bytes to price, not a window. Recorded as the handoff's open item
for the decode side; 0.4.4 ends without a rate, with the decode rows
of both cards measured bit-exact against their old forms, and with
the timing test's step change of 2026-09-07 attributed to the
instrument, not the kernel.

Retracted here (§7.0.1): the handoff's Q5_K decode figure, 316 GB/s
at 60 µs -- the timing test before 0029's instrument change; the
served launch, 0.129 ms and 279 GB/s on the served shape, is the
figure, and it has not moved since 0028.

#### 7.0.2br The handoff's open items after 0.4.4: the 16 GiB card's ceiling, the load's parallel repack, the equivalence suite on a GGUF-opened model, patch 0020's declined combination, and the production question (2026-09-08)

Five items the 0.4.x handoff carried outside any point release, taken
in turn after 0.4.4.

**The 16 GiB card.** Its random-read ceiling over incompressible
bytes, by the probe that gave the 24 GB card 453 GB/s (§7.0.2bd),
measured for the first time: **414–418 GB/s** (512 MB and 2 GB
buffers, work-groups of 256 and 512, best and mean within 1 %); the
24 GB card re-read 453–454 in the same window. The probe had only
ever enumerated the first OpenCL platform, and on this host each card
sits on its own, which is why the number was missing. The other half
of the item -- a second decode body per architecture with a served
16 GiB-card GGUF case -- has no model to serve: the dev host's smaller
GGUF files are of no allowlisted family with a template (a 2B, an
embedding model, a 35B MoE, the 27B's MTP-only file), and the 27B
does not fit. Closed as measured; the served case waits for a file.

**The load.** The handoff's reading -- "the verdict cache's load:
repack 35–60 s of the 88–168 s; parallel across tensors is the
obvious step" -- was measured before it was built. From the served
logs of sixteen loads of the same 15.3 GB file (the phase table is in
the session ledger; the figures here are its range): the repack 7–85 s
with the deviation check at zero when the verdict cache holds the
file (the first load of a new packing pays the check: 235 s over
18.7 G values), the compile 15–34 s, the rest 22–74 s -- and the
repack's own spread follows what ran on the host before it (7 s right
after another load of the same file, 85 s after nine minutes of
kernel tests), which is the page cache of a file the loader maps and
touches, not the code. The code did one thing at a time across the
288 repacked tensors (the rows of one tensor in up to sixteen
threads; the tensors in sequence, each with its graph edit). Now the
repack and the check run across tensors in a bounded pool -- the
workers each take whole tensors with the row threading reduced so the
count does not multiply -- and the graph edits follow in the original
order on the calling thread; the file's mapping is advised for
read-ahead at open. Byte-identical by test (the same tensors through
one and four workers, the verdict cache empty and warm), and the test
found a thread-order dependence in the deviation's root-mean-square
sum that the row-ordered reduction removed (no test had asserted that
figure exactly). Measured on the 24 GB card, the exact mixed form,
loads back to back (old = the installed 0.4.3, new = this tree):

| load of the served file | old | new |
|---|---|---|
| fresh verdict cache (the check runs) | 335 s (repack 23, check 261) | **135** (repack + check 75 on 8 workers) |
| warm cache, first pair | 75 (repack 31) | 102 (repack 36) |
| warm cache, second pair | 99 (repack 43) | 70 (repack 14) |

The first load of a packing is 2.5× faster; the warm loads are what
the page cache makes them, on either binary (the compile 20–30 s and
the rest are untouched). The item the record named -- the parallel
step -- is done and its worth is the check, not the repack.

**The equivalence suite on a GGUF-opened model.** The suite (`tests/equivalence/run.sh`, §5) could not run
on a GGUF-opened model because its one non-gated diagnostic -- the
stateful executor against the paged one -- starts a server that
`--gguf` refuses, and the driver ended there. `ARCINT_SKIP_STATEFUL=1`
skips that section and nothing else; every gated check is paged and
runs. Run on the 24 GB card, the exact mixed form (arcint 0.4.3 on
`+p12`): the prefix-cache gates pass (warm output byte-identical to
cold, the console's hit, a continuation of a cached prompt hits and
matches a cold run; MTP with the prefix cache warm = cold), the
copy-prompt drafter gate passes (42.5 % accepted), and four gates
fail. Three are the same finding and the fourth is a fact about the
template: `--mtp on` on a GGUF-opened model is inert -- the template's
MTP head accepts 0 % against the file's weights -- which the milestone
had carried as "untested".

The finding: **two greedy runs of the suite's 235-token prompt in one
process differ.** Twelve requests to four processes gave five texts,
with MTP and the logits slice on or off alike, while the 856-token
prompt has been byte-stable in every process of this record. The
bisect, four requests per process: the IR 4/4 one text; the native form
4/4; the mixed form at `--prefill-chunk 64` 4/4, and the native form's
text; the mixed form at the default chunk, exact and split, on 0029 and
0030 alike, not. The one element of every failing cell is the
runtime's f16 gemm on the repacked set at 235 rows. Its source says
why (oneDNN 3.13, `kernel_evaluator.cpp`): the selector scores a
k-parallel strategy -- split-K across work-groups, the partial sums
accumulated atomically -- best whenever the plain M × N tiling
underfills the device, and that reduction's order varies; the
catalog's f16 entries carry the tag; `attr->set_deterministic(true)`
scores every such entry out and pins the k-parallel-local work-group
count to one, and the plugin never set it. **Patch 0031** sets it in
the f16-activation branch of the compressed fully-connected, one line:
the 235-token prompt is one text 4/4, and the served rates do not
move (856 tokens warm 1,008 t/s and a 54.1 ms step against 1,001 /
54.8; 71,727 tokens 464 t/s and 73.0 ms against 464 / 73.7; the
greedy outputs the same at both depths). At 85 and 145 tokens the
text changes with the patch (the strategy pinned is a different
kernel; a near-tie moves, §3.2's class) -- chosen knowingly, the
alternative being the five texts. It ships as `+p13`.

Two things the fix does not cover, both measured and open. At 85
tokens two texts alternate across requests -- in the mixed form and
in the native form, whose projections never touch the runtime's gemm,
and not in the IR: the GGUF path's own, unattributed. It shows on the
0029 plugin (8-row reads) as on 0030, at `--prefill-chunk 64`, and
with the head unsliced (six requests each, both texts in every
cell); the split form gave one text six times, which at the observed
odds of the two texts is as likely chance as a lead. So it is not the
tall read, not the chunking, not the head slice, and the mins
packing's gemm width is unproven either way. And a
prompt of about 190 to 215 tokens can fault the process in the mixed
form at the default chunk: `CL_OUT_OF_RESOURCES` on a buffer map, and
in the host's kernel log an engine reset (the f16 process) and an
engine memory CAT error with a reset of the compute engine and a
timed-out job in the process (the `--dyn-quant on` one) -- a kernel
touching unmapped memory, at 16.5 GiB resident on the 24 GB card, so
not the VRAM-pressure class of §7.0.2ad. It reproduced at 205 tokens
in four processes (the shipped f16 form, `--dyn-quant on`, the
deterministic build), at 190 and 211 once each, and not at 154, 163,
175, 181, 196, 235 or 856; not in the native form, not at
`--prefill-chunk 64`, not in the IR. Non-monotonic in the row count,
so a placement reading rather than a tile boundary; the runtime's
compressed gemm at those row counts is the one element of every
faulting cell, and that is as far as the record goes without a
mechanism. Until it is found, a GGUF deployment that sees short
prompts serves them at `--prefill-chunk 64`, which is deterministic
and does not fault (its rate cost at 1k is not measured here).
*(Found the same day, §7.0.2bs: the mechanism is in the plugin's
micro-SDPA prefill, not in the gemm; the "one element of every
faulting cell" reading above was the wrong element, and the
non-monotonic pattern was placement after all -- of the pages behind
the K buffer.)*

**Patch 0020's declined combination** (4-bit values under BY_TOKEN
keys: NaN past 128 keys, declined by the selector, §7.0.2as). Read,
not run. BY_TOKEN keys are not a choice arcint makes: the plugin's
default is BY_CHANNEL and it forces BY_TOKEN only for a graph with
cache-block rotation, which arcint's graphs do not carry; and the
plugin's own paged-attention implementation already disables its
micro-SDPA path for 4-bit BY_TOKEN keys "due to accuracy issues" --
so the combination 0020 declines is one upstream declines too, on the
key side, and nothing served reaches it. The value-side code 0020
added is gated on the value precision alone (the page stride, the
per-token scale and zero point offsets all follow the plain block
size), so bug A's class -- a value stride computed from a key
parameter -- was not found again by reading; the two pre-existing
BY_TOKEN key-scale formulas (`ldkq = 1`, the scale pointer at the head
size minus the subgroup index) are the untested intersection with the
4-bit value unpack on that kernel, and the ranking is a reading. The
localising experiment is written down for whoever needs the path: the
0020 micro-SDPA test at BY_TOKEN, one head, 129 keys and 256 keys,
with the decline bypassed, the existing per-row NaN map naming the
first wrong element (head element 0 of every row past 128 says the
page index; a fixed interior offset says the key-scale formula). Not
run: the path serves nothing, and the record does not narrate a
mechanism it did not measure.

**A GGUF on a production unit.** The operator's call, with the numbers
beside it: the agent unit serves the IR at 13.06 GiB, 1,574 t/s and a
45 ms step at 856 tokens, 151,552 tokens of context at `u8:i4` with MTP
on. The same model's GGUF in the mixed form on the same card is 16.54
GiB (16.30 split), 1,001 t/s warm and 54.8 ms (53.5 split), 112k
tokens at `u8` (120k split), and `--mtp on` on a GGUF-opened model is
inert (above).
Slower on every axis and larger; its case is serving the file's own
quantisation. *(One of those axes is withdrawn, §7.0.2bw: the drafter was not
inert on a GGUF-opened model, it was switched off at load by a defect in the
hidden-state tap. With that fixed it accepts 73.0 % and is output-neutral, so
this verdict's drafter column no longer holds and the resident-size and rate
columns are what remain of it.)* Nothing was changed. If the file's weights are wanted
on a unit: `--gguf-mins split`, `--mtp off`, `--n-ctx` at or under
120k at `u8`, through the unit manager's rollout, not a hand edit.

#### 7.0.2bs The short-prompt fault: a K-tile prefetch in the micro-SDPA prefill ran 256 rows past the buffer; patch 0032, upstream's fix of the same day (2026-09-08)

The fault §7.0.2br left open -- a prompt of about 190 to 215 tokens
killing a mixed-form process with an engine memory CAT error -- has a
measured mechanism, a fix, and a regression test that is red without
it. It was never the gemm, and never the GGUF path: the plugin's
paged-attention prefill on the pinned nightly prefetches past the end
of every prompt's K buffer, and whether the pages behind that buffer
happen to be mapped decides between a served prompt and a dead
process. The forms and lengths that "never faulted" were the ones
whose neighbours happened to be mapped.

**How it was found.** The intercept layer with a finish after every
enqueue pinned the failing launch to the micro-SDPA prefill kernel at
two query tiles (GWS 32 × 768). Memory reuse off, USM off and the
control all faulted, which ruled out the allocator's placement of the
kernel's own buffers. A dump of the compiled sources showed the mixed
form running two variants of that kernel per layer, differing only in
whether V carries the fused projection's padding, and the native form
running the unpadded one everywhere -- the faulting one -- without
faulting. The micro-gemm bodies inside the kernel are not source (oneDNN
generates them at load time as machine code in an inline-asm block), so
they were extracted from the dump and disassembled: their K and V reads
are 2D block loads whose surface height is the key count the kernel
passes, hardware-bounded. The call log at the request's own launch (the
first sixteen two-tile launches in any log are the load ladder's
256-token pass) showed every argument at offset 0 of a live allocation
larger than the kernel's extent, in both forms; a dump of every buffer
argument of that launch, USM off, showed the subsequence table (0, 205),
the tile mapping (0, 0), (128, 0) and the shape information all correct.
The kernel faulted on correct inputs.

Then the primitive alone: a unit test in the plugin (24 heads, 4 KV
heads, head 256, block 16, u8 KV by channel -- the served geometry --
one subsequence of N new tokens on exact-size buffers) faults at 193,
202, 205, 208, 211, 214 and 217 tokens, hangs at 196 and 202 (a run each),
and passes at 256 and 856; 193 serves in arcint and faults here, so the
served pattern was the neighbours' slack. Environment switches in the
generator, one class of access each: the host-side cooperative K/V
prefetches off, 5/5 pass; the micro-gemm's own block-2D prefetches off,
still faults; the block Q loads off, still faults; the non-micro path,
pass. The source then reads plainly. The kernel prefetches the first K
tile with the geometry (row length d, row count = keys, stride 1
element) and the *next* K tile with the arguments in the other order
(row length = remaining keys, row count = d = 256, stride 1 element):
the pointer lands *inside row 0* of K (k0 + 128 elements, not rows),
and from there the helper walks 256 rows of up to 256 B whatever the
tile, its clamp computed from the same swapped geometry. So a prefill
chunk of N keys with a next tile to prefetch -- 129 to 255 keys -- reads
256 − N rows past the end of K; at 256 keys and beyond the walk is in
bounds, which is why 256 and 856 pass and every faulting length lies
below 256. The order is oneDNN's, for a
transposed K; the plugin's K is [tokens × d]. Upstream fixed exactly this
on 2026-09-08 (openvinotoolkit/openvino PR #37878, "Fix out-of-bounds
next-K-tile prefetch in micro SDPA"), eighteen days after the pinned
nightly: the stride ldk unless TRANSPOSE_K, row length d and row count
the remaining keys for both calls. **Patch 0032** is those two hunks, plus
the reproducer as a regression test (upstream's own test is coverage:
its header says the pre-fix order passes it too).

**Measured** on the 24 GB card. The reproducer 11/11 green with the patch
(193–217 unpadded, 205 padded, 256, 856); the plugin's paged-attention
and SDPA suites 264/264. Served (arcint 0.4.3, the exact mixed form,
u8 KV, `--mtp off`, n-ctx 8192, prefix cache off, the default prefill
chunk), one process: 190, 205, 211, 205, 190, 211 tokens, six requests
served, one text -- the same text the non-faulting lengths gave before
-- where before a fresh process died on its first request at 190, 202,
205, 208, 211 and 214 tokens, every time (205: six processes; 190: the
one process that tried it). Rates unchanged: 856 tokens warm
1,008–1,009 t/s against 1,010 on +p13, the decode step 54.5–54.7 ms
against 54.7 (3.49–3.50 s per 64 tokens against 3.50), the greedy text
the same (283b2c44), four requests each, one fresh process per runtime,
the patch staged into the +p13 runtime. It ships as `+p14`; arcint's
own runtime floor stays at +p12 as with 0031, the fix being the
runtime's.

Two readings of §7.0.2br are withdrawn: the gemm as "the one element
of every faulting cell" (the SDPA was the element, the gemm ran in the
non-faulting forms too and was never suspect by measurement), and
"placement of the kernel's own buffers" (the placement that mattered
was of whatever lay behind them). The workaround it named,
`--prefill-chunk 64`, worked for the right reason -- one query tile per
chunk has no next K tile to prefetch -- and is no longer needed. Not
touched by the patch: the K-scale and K-zero-point prefetches of the
2D-quantised key path keep the pinned order; they are bounded by a cap
of one group and no served configuration takes that path. The 85-token
two-text alternation stays open and is unrelated (it does not fault and
the native form shows it).

#### 7.0.2bt The kernel review, first pass: every kernel on the card counted, timed and disassembled; the recurrent-state rows zeroed on the device (2026-09-08)

The operator's directive after the 0032 fix: every kernel that reaches
the card -- the plugin's, oneDNN's, the micro-gemm blobs -- disassembled
and reviewed for efficiency. This section is the first pass: the census,
the ranked table, what the disassemblies say kernel by kernel, and the one
patch it produced so far. The raw material (call logs, traces, every ISA
binary and jit source, the parsers) stays on the dev host; the record here
is what a reader needs to rank the rest.

**The census.** One 856-token chat prompt with 64 output tokens, under the
intercept layer with the call log, the chrome trace, the per-kernel ISA
and jit-source dumps, on the 24 GB card. The tracer's overhead sits in the
span, not in the device durations (the card's own timestamps).

| form | prefill: launches, device busy | steady decode step: launches, busy | kernel variants |
|---|---|---|---|
| IR, the agent unit's config (u8:i4 KV, MTP on, chunk 512) | 4,315, 681 ms | 1,763, 48.8 ms | 233 |
| GGUF mixed exact (u8 KV, MTP off) | 2,406, 841 ms | 2,045, 52.0 ms | 117 |

**The ranked table**, by device time over one prefill plus 64 steps, with
what the ISA says. Bandwidth shares are against the 453 GB/s probe ceiling
(§7.0.2bd); the byte counts of the K-quant launches come from the jit
sources' filter dimensions.

| # | kernel | share | the reading | lever |
|---|---|---|---|---|
| 1 | oneDNN M=1 decode gemm | 68 % of the IR step, 58 % of the GGUF step | one 128-k chunk per iteration; one 2D block load of packed int4 per 32 k, 80 ALU instructions of unpack, zero point and scale per 512 weights, then two DPAS; a work-group barrier per chunk; eight-way k-parallel inside the work-group with an SLM reduction; 80 work-groups, one wave. 73–79 % of the ceiling; by the instruction count the dequant would fit under the loads, unmeasured | oneDNN strategy only |
| 2 | oneDNN lm_head gemm | 11.7 % of the IR step (two launches with MTP) | the same loop shape on a grid of 3,880 work-groups in 48 waves of ten short chunk iterations each: 48 latency chains in series, 49 % of the ceiling. The GGUF form's K-quant lm_head does the same job at 96 % | oneDNN strategy only |
| 3 | oneDNN small-M gemm (the MTP verify rows) | 5.9 % of the IR step | 40 work-groups, half the device; 3.6× the M=1 launch on the same weights | oneDNN strategy only |
| 4 | the K-quant decode kernel | 32 % of the GGUF step | 71–96 % of the ceiling by shape (o_proj 71, the down projection 72, the lm_head 96: the shapes with one to four waves pay ramp and drain). Its own row and unroll knobs measured: any row count but four leaves the kernel's path; the block unroll of two is slower on every shape at 128 registers, four spills to scratch | none from the tiling |
| 5 | oneDNN int8 prefill gemm | 66 % of the IR prefill | 32 DPAS per iteration against ~590 ALU instructions of int4-to-int8 dequant done per thread, no SLM sharing: 102 TFLOP/s against an int8 XMX roof of twice the f16 roof's 98 TFLOP/s (§7.0.2bo), about half | oneDNN strategy only |
| 6 | gated delta net prefill | 9–11 % of both prefills | one subgroup per head and value block walks tokens sequentially with ~10 cross-lane reductions per token; 1,536 subgroups on 1,280 thread slots; 1.86 µs per token per layer | two tokens per iteration; the chunked form (large) |
| 7 | host-to-device state writes | 3.2 % of the IR prefill, 2 % of the GGUF one | 48 uploads of 5–6 MB per request (the gated-delta-net tables; the 48 conv tables are small and below the table's cut): arcint's own zeroing of the recurrent-state rows | **this section's patch** |
| 8 | the MTP head's GQA broadcast | 0.9 % of the IR step | the drafter's attention is not paged; its graph materialises K and V from 4 heads to 24 over the whole context every step | export-side |
| 9 | the KV-cache append | 1.2 % of the IR step | by-channel u8 keys re-quantise the whole 16-token block per appended token; inherent to the scale scheme | none |
| 10 | the reference kernels (activation, eltwise, slice, reduce, concat, gather, rms) | ~5 % of the prefill, ~2 % of the step, a quarter of the launches | small each; fusion | later |
| 11 | the f16 prefill gemm; the tiled K-quant prefill | 48 % and 27 % of the GGUF prefill | at the f16 roof (§7.0.2bo); at the record's 50 % (§7.0.2bp–bq); the tiled kernel's epilogue stores a 64×16 tile with 64 scattered 16-bit stores, negligible | none new |

Spills, the review's first suspicion, carry under one percent: two
256-register variants (a small tiled K-quant shape, a rare generate-variant
micro-SDPA) and the micro-SDPA prefill's 35 loads and 18 stores to scratch.

**The reading behind the table**, and it is a reading: what the ISA
suggests, unmeasured until a strategy-override experiment runs. The big
items are oneDNN's strategy choices -- the decode gemm near its roof; the
lm_head and small-M gemms at half of it, where the grid shape is the
visible difference from the fast one; the int8 prefill gemm with eighteen
ALU instructions per DPAS, which reads as issue-bound but no counter has
said so. The plugin's fully-connected passes oneDNN a primitive
descriptor and attributes (§7.0.2br's deterministic flag is one) and no
strategy hint that this reading found; they are recorded for an upstream
offer or that experiment. The cheap levers are arcint's own, and the
first is done.

**Patch 1: the recurrent-state rows on the device.** A fresh request
zeroed every conv and gated-delta-net layer's per-lane state table by
uploading a zero-filled host copy of the whole table -- 48 layers × 6 MB
on the 27B hybrid, 22 ms of device time per request in the IR form, 17 ms
in the GGUF one -- and a checkpoint row's read or write staged the whole
table both ways. Now one resident zero row per distinct state shape is
filled once, and a ROI view of the target row (the same view the KV host
tier uses, §4.4) takes a device-side copy from it; a row read or write goes
through the same view; the write still zeroes the other rows first, the
old side effect kept. If the plugin refuses a remote-to-remote ROI copy the
host path takes over for the process, logged once (it did not). An
arcint-only change: the runtime floor stays at +p12; measured on the
installed +p14. Measured on the 24 GB card, the agent unit's config
(u8:i4 KV, MTP on, chunk 512, n-ctx 8192, prefix cache off), two fresh
processes of the new binary against the installed 0.4.3, three requests
each; "restore" is the prefill line's wall time for the cache lookup,
the restore and the row zeroing, which with the cache off is the zeroing:

| prompt | before | after |
|---|---|---|
| 130 tokens | 0.19 s, 667–669 t/s, restore 0.03 s | **0.17 s, 769–773 t/s**, restore 0.00 |
| 856 tokens | 0.71 s, 1,203–1,205 t/s, restore 0.03 s | **0.68 s, 1,247–1,252 t/s**, restore 0.00 |

Output-neutral: the same texts per request in the same order before and
after. The equivalence suite on the new binary passes every gate,
including the prefix-cache restores that go through the row read and
write (warm byte-identical to cold, with MTP and without, a continuation
restored from cache matching a cold run).

**A finding on the way**, open: the agent unit's own configuration with
MTP on gave a different greedy text for the same 130-token prompt at each
of three requests in one process -- and the same three texts in the same
order in every process (four of four, before and after the patch alike),
so a state carried across requests, deterministic in the request index,
not noise. The suite's own two-run gate passes with its settings; open,
the bracket (MTP off, the prefix cache on, u8 KV) is the next section's
subject.

#### 7.0.2bu The alternation: with four-bit values the verify pass read the value rows through the f16 row's alignment; patch 0033 (2026-09-08)

The finding §7.0.2bt left open, run to ground (the bracket it named -- MTP
off, the prefix cache on, u8 KV -- is measured below: each of the three
sides holds one text). The agent unit's own
configuration (the IR of the dense Qwen3.8-27B, u8 keys with i4 values, MTP
on) gave two greedy texts for one 130-token prompt, alternating by request
parity within a process, and neither was the text of MTP off or of symmetric
u8 -- so the first request was already wrong, and only the alternation made
it visible. Every measurement below is on the 24 GB card, one fresh process
per cell, sixteen output tokens, the text named by the first eight hex digits
of its SHA-256.

**Where it was not.** Each ruled out by a measurement, not by reading: the KV
pool's contents (zeroed per request: alternates), memory reuse, the prefix
cache (off and on: alternates, the warm requests hit the cache and still
alternate), the recurrent-state rows, the drafter's reset, oneDNN's
determinism attribute on the int8 gemm, the plugin's intermediates. The
plugin's tensor dumps of two consecutive requests differed in one input: the
block table. Request 1 held pages 0..8 ascending, request 2 pages 10..2
descending. The page pool is a stack -- release pushes a sequence's pages in
order, allocate pops from the back -- so odd requests get an ascending run and
even ones a descending run. Three pool experiments (development knobs, not
shipped) closed it: pages as the pool hands them, alternates; fresh pages
sorted ascending, the odd text moves to requests 3 and 5, exactly the requests
whose logical blocks 0 and 1 land on pages 0 and 2; the lowest free pages
first, so every request gets the first request's page set, six identical
texts. Page identity is the whole carrier.

**Where it was.** Every kernel that takes the block table (ten) indexes it;
the micro K tile is 16 keys in every paged configuration, the host splits
appended tokens at the page remainder, a paged tile is cut at past_len, and
the single-query micro SDPA passed a NaN-tail test (5/5). The plugin's u8:i4
mixed-micro test, given the served shape (two new tokens over 130) during the
search, was green with its fill of the time. The defect is one line in the
micro-SDPA generator: the V*S
micro-gemm's A operand gets its alignment from the packed row (head/2 + 4
bytes) only when the *key* precision is four-bit; under u8 keys with i4
values it kept the f16 row's, 128 for a 132-byte row (the helper returns the
lowest set bit of the row length, capped at 128; 68-byte rows at head 128 get
the same 128).
Patch 0020 keyed the operand's type on the value precision and left the
alignment on the key's. A gemm strategy told its rows are 128-byte aligned
addresses them accordingly, and what it reads then depends on the page's
address -- hence page identity, hence parity.

**Measured, served** (`/usr/bin/arcint` 0.4.3, the agent configuration, chunk
512, no prefix cache):

| plugin | 130 tokens, six requests | 8,005 tokens, twice | prefill 8k | decode at 8k |
|---|---|---|---|---|
| +p14 stage m53 | 410be5ff 009c9d5e 410be5ff 834cb18e 410be5ff 834cb18e | d2845b6c, d2845b6c (stops at 7 tokens) | 869, 874 t/s | 21.6 t/s |
| m53 + the line (m54) | 825b1747 x6 = the MTP-off text | e8010a09, e8010a09 | 865, 870 t/s | 20.2, 22.9 t/s (64 tokens) |

The line changes no rate (prefill within 1 %; the decode samples are too short
to separate) and makes MTP on byte-equal to MTP off on this prompt. The
equivalence suite on the fixed plugin passes every gate, including a new
one: MTP at u8:i4, the same text on three requests of one process.

**Why the plugin test was green.** Two blindnesses, both in the test harness
and both now part of the patch. The harness always built an ascending
contiguous block table and addressed cache pages as `start + j` in ten
places, the one shape a served pool never guarantees; it now goes through the
table, and a page order can be requested (reversed, or a gap after page 0 --
the served third request's shape). And the mixed-micro tests' fill made every
page look alike: every past token carried the same key and a zero value, the
new tokens' keys depended on the token's position within its page only, and
a query of 8 made the softmax one-hot on each page's last token, so a misread
page returned identical bytes. The fill now gives every token, head and page
its own key (on the sixteen-level grid u8 by-channel stores within 4e-4) and
value (on the sixteen integer levels i4 by-token stores exactly -- all
sixteen present in every token-and-head row, so the row's scale is 1 --
distinct per token, head and 16-dim group), with a query of 1/64 so every
token's weight is within a factor of five of the others.

With that fill, full statistics over every element against the float
reference: before the line the u8:i4 cases err by 2.0-2.3 on 98-99 % of
their elements (the value range is 15) and the served geometry with three
new tokens over 1,000 hangs the test binary (aborts under a reversed or
gapped table); with the line, exact: at most 0.001 on every case (head 128 and 256,
u8:i4 and symmetric u4, the three page orders), none over 1e-2. The same fill through f16 KV and
through symmetric u8 is exact to the tolerance, so the reference and the
fill are sound. The tests' tolerance stays at 1e-2.

**Retracted on the record (§7.0.1).** A first version of that fill put eight
consecutive levels in each token-and-head row instead of sixteen, and the
search read what followed as a kernel property: a residual of 0.11 with
the line, "the four-bit value path's own floor", located by an element map
in the middle half of the head dims and announced as a defect for its own
patch. The review's arithmetic showed it to be the fill's own quantisation:
a row of eight levels has range 7, a quantisation step of 7/15 instead of
1, so its dequantised levels miss the integers by 0, 0.067, 0.133, 0.2, 0.2, 0.133, 0.067, 0 across the
eight groups -- zero at the outer quarters, largest in the middle half --
and weighted by the nine rows in sixteen that do not wrap past level 15
this gives 0.075 and 0.1125 on the map's groups against 0.073 and 0.110
measured, and 0.0068 for u8 by token against 0.0068 measured. Sixteen levels
per row make the same rows exact, and the "floor" is gone (the line above).
No kernel defect stands behind that number; the earlier readings that it
was "not token mixing" and "unchanged by the line" were correct and are
explained by the same arithmetic.

**On the way, fixed.** The harness packed the past tokens' four-bit values
as (dim, dim + 16) pairs where the production writer and reader use adjacent
pairs; with 16-dim value groups the reader saw the other group's value on
half the dims. Production agrees with itself (the served text equals the
symmetric u8 text); the harness is corrected in the patch.

The ledger records every mechanism narrated during the search in order with
its measurement; the one that reached a document is retracted above. The
11:20 localisation to "the micro-SDPA MIXED stage with 4-bit values" was
right as far as it went; the alignment line was recorded as a hypothesis
before its measurement, as §7.0.1 requires.

#### 7.0.2bv 0.4.5, the small open items: `/props` reporting what is served, patch 0020's declined pairing re-measured and kept declined, MTP said out loud on a GGUF-opened model (2026-09-08)

Four items the 0.4.4 handoff carried, each closed by a test or a
measurement. None of them moves a served text: the 130-token prompt on the
agent configuration gives 825b1747 on three requests of one process, matching
§7.0.2bu's figure for the same prompt and configuration, and the equivalence suite passes on both units' configurations
(the 24 GB card at `u8:i4`, 13 gates; the 16 GiB card at `u8`, 9 gates and
the artifact's own MTP skip).

**`/props` reported the stateful defaults, on every server.** The `cache`
block was three Config fields and a hardcoded `false`, none of which the
paged path reads. Both served units answered with the identical block --
`kv_dtype "fp16"`, `prefix_cache false`, `kv_block_size 32` -- while one of
them served `u8:i4` with an 8 GiB prefix cache that its own `/health` showed
hitting, and the other `u8` with 2 GiB. `kv_dtype` named a precision neither
served; `prefix_cache` was the M3 placeholder; `kv_block_size` is the prefix
cache's checkpoint granularity and the stateful path's block, not the paged
page size, which the block did not report at all.

The values cannot be read off Config, which is why the fix is not in the
handler alone: `--paged-kv` is overridable at load by `ARCINT_PAGED_KV`, so
the spec that won is `effective_paged_kv` in `load_paged`, and whether a
prefix cache exists is decided where it is constructed. `ModelStatus` now
carries both -- the served KV precision and whether a cache is serving --
and the block reports `path`, `kv_dtype`, `kv_block_tokens`, `kv_block_size`,
`prefix_cache`, `prefix_cache_mib` and the GDN checkpoint budget. A stub
loads nothing, so its path, precision and page size read `null` rather than a
default it never ran. Measured on the served configurations of both units:

| unit | `/props` cache block, this tree |
|---|---|
| 24 GB, `--paged-kv u8:i4 --prefix-cache-mib 8192 --n-ctx 151552` | `path paged, kv_dtype u8:i4, kv_block_tokens 16, kv_block_size 32, prefix_cache true, prefix_cache_mib 8192` |
| 16 GiB, `--paged-kv u8 --prefix-cache-mib 2048 --n-ctx 98304` | `path paged, kv_dtype u8, kv_block_tokens 16, kv_block_size 32, prefix_cache true, prefix_cache_mib 2048` |

Each agrees with the reservation block beside it (28,928 and 11,600 bytes per
token). Reporting only; no engine change. Red first on both levels: the unit
test on the assembled JSON aborts on the missing `path` key against the old
block, and `tests/roundtrip.sh`'s stub assertion fails.

**Patch 0020's declined pairing: re-measured, and the decline stays.** The
handoff asked for the declined combination -- four-bit values under BY_TOKEN
keys -- to be re-tested with 0033's discriminating fill and the three page
orders, and lifted if green. It is not green, and the reason it is not green
is not the one the item expected. All on the 24 GB card, one fresh process
per cell, the served geometry (24 heads, 4 KV heads, head 256, block 16)
unless said otherwise, with the routing read from the dispatched kernel list
rather than assumed.

The by-token cases fail before anything is lifted, on the generic kernel the
pairing has always had, and they fail as total NaN rather than a wrong value:

| by-token case | kernel dispatched | NaN |
|---|---|---|
| i4 values, 36 keys | generic paged attention | 3,072 of 12,288 |
| i4 values, 102 keys | generic | 9,216 of 12,288 |
| i4 values, 126 and 128 keys | generic | every element |
| i4 values, 132-1,003 keys, three page orders | generic | every element |
| **u8 values**, 132-1,003 keys, three page orders | **micro SDPA** | every element |
| by-channel, every case of `patches_0020_paged_attention_u8i4_mixed_micro` | micro SDPA | none; max error 0.000488 |

The eight-bit row is what settles it. That pairing is not what 0020 declines,
it runs on a different kernel, and it fails identically. Two kernels
producing all-NaN from one fill point at their common input, not at either
kernel. And the fill is not simply unusable: at the test's default geometry
(32 heads, 2 KV heads, head 128) the by-token cases are exact at 36 keys --
max 0.002, no NaN, at both value precisions -- and go all-NaN at 132. So what
is measured is a NaN that tracks the causal length and the head size and
ignores both the value precision and the kernel:

| head size | 36 keys | 102 | 126 | 132 and up |
|---|---|---|---|---|
| 128 | exact | -- | -- | all NaN |
| 256 (served geometry) | 25 % NaN | 75 % | all | all |

The NaN counts are not scattered elements. At 24 query heads over 4 KV heads
and head 256, each row is 6,144 elements and each KV-head group is 6 query
heads: 3,072 NaN is one whole group, 9,216 is three, 12,288 is all four. So at
head 256 the by-token failure arrives a whole KV-head group at a time -- one
group at 36 keys, three at 102, all four from 126. That is arithmetic on the
counts above, not a mechanism, and it constrains all three candidates equally:
whatever the common element is, it fails per KV head, not per element.

No mechanism was claimed at the time of writing. The common element -- the
harness's own by-token page writing, its cache sizing, or the key
dequantisation both kernels share -- was not measured, and this section
named those three candidates rather than choosing one. What the record
could say is that the re-test cannot reach a verdict on this instrument,
because the instrument fails in a configuration the decline does not
govern; so **the decline stays**.

**Patch 0035 found the mechanism (2026-09-09).** It was candidate 1: the
harness's own by-token page writing. `std::fill_n` gave every dimension
of a token's key vector the same fp16 value; `quantize_data()` then saw
`min == max`, fell through to the `diff = 0.001` fallback, computed
`scale = 255000`, and derived a zero-point that overflows fp16 for tokens
whose base key value exceeds ~0.257 (zp ≈ −66430, below fp16 min −65504
→ −inf). The kernel correctly propagated −inf → inf, then softmax hit
inf − inf = NaN. The NaN counts from 0034's table match arithmetically:
head 3 at head 256 (base 0.26+) overflows from block 0, producing one
KV-head group of NaN at 36 keys; heads 1–3 overflow at 102 keys (three
groups); all four from 126 keys. At head 128 the base stays below 0.257
at 36 keys, so no overflow -- exact. Every row predicted, every row
confirmed. The fix replaces the constant fill with a per-dimension
linear ramp (±0.128 / ±0.256); all 6 cases pass enabled.
The decline itself remains: this was a test bug, not a kernel bug, but
the decline's own measurement is still unreproduced on the current fill.
Patch 0020's decline is now re-testable (FIX 2b).

Nothing served is affected. BY_TOKEN keys are not a choice arcint makes: the
plugin defaults to BY_CHANNEL and forces BY_TOKEN only for a graph with
cache-block rotation, which arcint's graphs do not carry (§7.0.2br), and
every by-channel case is exact.

Not retracted, but not reproduced either: 0020's own note recorded this
pairing as NaN "for every query whose causal context passes 128 keys, and
only those", measured on 2026-09-05 on the staged tree with that patch's own
by-token test and the fill of the time. The pattern above -- a quarter of the
elements already NaN at 36 keys at head 256 -- is not that pattern. The two
were taken on different fills and different geometries and are not the same
measurement; which of the differences accounts for it is unmeasured.

**MTP on a GGUF-opened model is inert, and now says so.** *(Retracted the next
day, §7.0.2bw: it was never inert. The head was switched off at load and never
drafted at all; "0 % accepted" was measured correctly and framed wrongly.)* §7.0.2br measured
`--mtp on` against a GGUF-opened model accepting 0 %: the head served is
always the template export's own IR, and its weights are not part of the
file's body, so the drafter proposes tokens the body rejects. The server said
nothing at load and the equivalence suite failed its acceptance gate, which
reads as a defect when it is a fact about the artifact pairing. A pure
decision function now returns the warning text for exactly that corner (a
GGUF file opened and MTP wanted), the paged load path logs it once -- the
stateful path is not touched, `--gguf` being refused there -- and the suite
detects GGUF-openness from its own `ARCINT_EXTRA_ARGS`, reports the
acceptance as `ACCEPTANCE-SKIP mtp-acceptance gguf-opened-head-inert` instead
of failing, and takes the same fact as implying the stateful-vs-paged skip
that previously needed an environment variable set by hand. The gate stays
able to fail on an IR-opened model: acceptance above the threshold passes as
before, and the copy-drafter gate, which does accept on GGUF weights (42.5 %,
§7.0.2br), is untouched. Red first: the four-corner unit test does not link
before the function exists.

**Measured on a GGUF-opened model**, which is what makes the two paragraphs
above more than a reading of the code. On the 24 GB card, the dense template
opened with `--gguf` over the 15.3 GB Q4_K_M file at `u8` KV: the load-time
warning fires (the server's own `mtp:` line names the head and the file), and
the suite passes every gate -- 12 ok, 0 failed, one `ACCEPTANCE-SKIP
mtp-acceptance gguf-opened-head-inert` at **0.0 % accepted**, with the
stateful section skipped for the reason the flag implies. The acceptance
figure confirms §7.0.2br's 0 % on this runtime; why it is exactly zero is
still not measured. The gate can still fail: it reports the skip only when the
acceptance line exists AND the server's warning is in the log, so a head that
never ran -- failing to compile, or losing the hidden state -- fails the gate
as it did before rather than being folded into the skip.

**The single-query micro-SDPA tail test** (§7.0.2bs's non-paged neighbour):
five cases asking whether the output depends on what lies past the sequence
length in the K/V allocation -- a 129-row view of a 151-row allocation whose
tail holds NaN, against the same rows in an exact allocation. Green 5/5 on
the 24 GB card, and it was in no patch until now; patch 0034 carries it. It
costs nothing and it guards the form patch 0032 fixed on the paged side.

#### 7.0.2bw 0.4.6, the hidden-state tap and the retired inert-MTP warning: one walk that did not know the K-quant head, and a warning that framed its symptom as a fact (2026-09-09)

Two gaps, one root cause.

**Gap A: `expose_hidden_state` failed on every GGUF-opened model.** The walk
from the first Result to the LM-head projection accepted only
`ov::op::v0::MatMul` as its terminator. A GGUF-opened model whose
`output.weight` stays in the file's rows has `FullyConnectedKQuant` in its
place (`exec/kquant_op.h`), so the walk failed at hop 0 on every such model,
`want_mtp_` was set to false, and the drafter never ran. The "0 % accepted"
measured in §7.0.2br was correct -- no tokens were accepted -- but the framing
("the drafter proposes tokens the body rejects") was wrong: it never proposed.
The walk now accepts `FullyConnectedKQuant` alongside `MatMul`, and a failure
logs the node the walk stopped on with its type and friendly name.

Root cause: the same backward walk exists twice in `backend_ov.cpp`, twenty
lines apart. `slice_logits_to_last_token` was taught the K-quant head on
2026-09-07 (§7.0.2bg); `expose_hidden_state` was not, in the same pass. The
duplication is a defect in its own right and is carried as a separate item.

**Gap B: the `gguf_mtp_inert_warning` is retired.** The load-time warning
introduced in §7.0.2bv ("the MTP head served is the template export's...
measured at 0 % accepted") was premised on the 0 % being a fact about the
artifact pairing. With the tap fixed, the same configuration measures **73.0 %
acceptance** (the 24 GB card, the dense template opened with `--gguf` over a
Q4_K_M file, `--paged-kv u8`), and the drafter's text matches the non-MTP arm
byte for byte. The 0 % was a defect, not a pairing property, so the warning,
its declaration (`config.h`), its definition (`config.cpp`), its four unit
tests (`test_config.cpp`), and the equivalence suite's `ACCEPTANCE-SKIP
mtp-acceptance gguf-opened-head-inert` path (`tests/equivalence/run.sh`) are
all removed. A GGUF-opened model's drafter is gated exactly like an IR-opened
model's: acceptance above the threshold passes, below it fails.

Measured on the 24 GB card, the dense template opened with `--gguf` over the
15.3 GB Q4_K_M file, `--paged-kv u8`:

- Red (before the fix): `hidden state walk stopped at hop 0 on
  FullyConnectedKQuant "__module.model.lm_head/ov_ext::linear/MatMul"`,
  `/props` `mtp enabled false`, `draft accept 0.0 %`.
- Green (with the fix): no warning, `mtp enabled true`, **`draft accept
  73.0 %`**, greedy text `283b2c44` identical across all four requests of both
  arms.

Red-first tests (`tests/test_gguf_graph.cpp`): the MatMul head (the control),
the K-quant head (the case that failed), and an `Add` node (neither terminator
-- must fail closed). The function is declared in `exec/graph_rewrites.h` and
defined outside the anonymous namespace, same as `slice_logits_to_last_token`.

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

**The corollary for an explicit `--n-ctx` that overshoots at allocation
time.** The replay loop's explicit retry (§7.0.2t; `fit.h`'s `pool_sizing`
and `explicit_retry_decision`) uses exactly this split: the reserve absorbs
the overshoot first, one KV page at a time at minimum, and the retry
refuses only once the reserve is at zero. That is checked directly rather
than assumed: of the four replay passes, the last is forced to a live-only
request regardless of how much a trim sequence would otherwise have left
standing, so "the reserve is exhausted" is a fact about that pass's own
`spare_blocks`, not an inference from having run out of passes. It is the
same "everything that costs bytes comes out of the reserve" rule as the
ceiling corollary below, applied per retry pass instead of per `--n-ctx`
value — the request itself is a fixed count and, same as live pages above,
cannot absorb it.

With `--n-ctx` omitted the split runs the other way: the fit pass adopts the
maximum admissible depth, the pool is all live pages, and the reserve for
cached prefixes is nil — the load says so in a warning (measured on both
served configurations, 2026-09-03: 0 spare pages at 155,376 and at 171,312).
`--prefix-cache-reserve PCT` (0.2.13) holds PCT of the affordable pages
spare under auto-fit and adopts the correspondingly lower depth; it is
refused with an explicit `--n-ctx`, whose reserve is whatever remains by
construction, and below the 4096-token floor.

The correction pass itself had a blind spot that only a pool with spare
pages could show (measured 2026-09-03: the coder at a 25% reserve and the
35B at auto-fit on the 24 GB card each sat at the ceiling for four passes,
at 100,224/100,080/100,064/100,048 and 262,144/260,080/260,064/260,048):
a sub-page overshoot trimmed one live page, the spare absorbed it, and the
pool total — what the driver actually rounds — never moved. The correction
now trims the pool total, spare first unless a reserve was asked for, with
a per-pass floor of 4/16/64/256 pages, which converges for any allocation
granule up to 84 pages when the first overshoot is itself below one granule
(the rounding case measured here) and refuses loudly, with the attempt
history, above that. Both cells come up: 100,080 with 2,086 pages spare and 262,144 with
6,816. The zero-spare cells cut 11–12 pages analytically on their first
pass, above the floor, and adopt the same 155,376 and 171,312 as before.

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
- **Drafting at depth: a decision, not yet an option** (noted 2026-09-03,
  from §7.0.2ab). Measured on the 24 GB card at 76k tokens: a one-token
  forward costs 104 ms and any isolated forward of two to eight tokens
  costs about 209 ms, while the served verify forward of two tokens runs
  430 ms, so a verify cycle pays two to four plain steps at depth and a
  drafter breaks even only at one to three accepted tokens per cycle on
  average; near depth 0 the same switch costs 10%. The record's advice is to serve deep
  contexts without a drafter. A future version could make that a runtime
  decision instead of an operator setting: switch the drafter off per lane
  once the served depth crosses the point where the measured acceptance no
  longer pays for the multi-token step (the cross-over is per model and per
  card and must be measured, not assumed), and back on when the context is
  reset. Deferred until the cause of the zero acceptance at depth is known,
  because a drafter that accepts nothing at 76k is a different defect from
  one that merely costs more there. The plugin-side alternative -- a
  small-batch attention stage that does not pay the full multi-token cost
  at depth -- is a kernel question for the plugin series.
- **Native 4-bit values in micro-SDPA** (backlog, 2026-09-03): the
  upstream fix for the u8:i4 prefill path — the microkernel path declines
  packed 4-bit values and the fallback sizes its scratch by depth
  (§7.0.2ab). Deferred behind the plugin-side patch that bounds the
  partition count and stores f16 partial outputs; the operator's decision
  was to take those two first and carry the upstream change in the patch
  series backlog.
- **Plugin framework gap: intermediates reallocated without an argument
  rebind** (found 2026-09-03 while chasing a served-path crash under patch
  0015). In the GPU plugin, `primitive_inst::realloc_intermediates` replaces
  an intermediate buffer's identity — a real reallocation or a reinterpreted
  wrapper — without raising the flags that make the kernel arguments rebind
  (those track outputs only), so a kernel can keep running on the previous
  handle after its allocation was freed. Patch 0015 exposed it because its
  bounded partials make the intermediate plateau and take the reuse path on
  every chunk of a long prefill; the patch carries the local fix (the paged
  attention implementation forces the rebind when its intermediates change).
  The framework-wide fix belongs upstream and in a patch of its own.
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
- **The hybrid-state transfer contract** (noted 2026-09-01, from the
  GLM-5.3-Flash release): the industry has converged on this repo's model
  class and its problem set. GLM-5.3-Flash is a 320B-A18B MoE with the same
  1-in-4 hybrid layout (34 of 45 layers linear attention, KDA — the same
  delta-rule family as GDN), an MTP draft layer in the checkpoint, and a
  community DFlash2 drafter (block 8, 7 drafts, 74.1% acceptance, 2.15x over
  MTP-4) — our 3.8 result reproduced independently on a different model.
  Z.ai serves it behind a separated Encode–Prefill–Decode architecture, and
  both SGLang and vLLM disaggregate prefill from decode by transferring the
  *pair* — paged KV plus the KDA recurrent state — between pools; vLLM makes
  the state-layout compatibility an explicit pinned contract
  (`VLLM_SSM_CONV_STATE_LAYOUT`, `VLLM_KV_CACHE_LAYOUT`) checked on both
  sides. Multi-GPU disaggregation stays a non-goal here, but our prefix
  cache's snapshot blob is the same pair in miniature: if that state ever
  crosses a process boundary (host-tier sharing, a warm handover), the
  layout-pinning assertion is the piece to copy — a declared, checkable
  contract rather than an assumed byte layout.
