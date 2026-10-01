# research — reference audit: what Strata and FreeToken do that arcint does not (2026-10-01)

Read-only audit of two expert engines' **source**, against arcint's record
(DESIGN.md, docs/campaigns, patches). The operator asked for it because a
faulty implementation, explained in enough jargon, can look like a proof that
something is impossible. The rule from now on: follow what the more expert
engines do, and treat an arcint "negative" as settled only if it tested the
same mechanism.

References, read at source:
- **Strata** (`github.com/Niko1221/Strata`), written for Qwen3.8-Flash-Next
  on one consumer GPU plus RAM. Its author reports 49–93 t/s decode and
  1,600–2,200 t/s prefill on an RTX 5070 12 GB with a Ryzen 5 7600 and 64 GB
  DDR5.
- **FreeToken** (arcint's long-standing design reference), an edge-MoE
  engine with CPU-resident experts and a GPU LRU cache.

arcint today, measured here (B60, dense d48q8, 20–27k tokens): 6.6 t/s decode,
~61–65 t/s prefill.

## 1. Negatives on arcint's record that did not test the reference mechanism

| arcint verdict | what the reference does (`code`) | what arcint built | same mechanism? |
|---|---|---|---|
| LRU expert tier rejected (DESIGN §7.0.2ae) | FreeToken: one LRU pool shared by all layers, slot rewrite on the GPU (`moe/offload_cache.py:169-184`). Strata: decayed usage counts, up to 96 swaps every 4 rounds; the old expert is evicted at once and the new one admitted when its copy lands (`generate.cpp:4414-4478`). | A per-layer, per-access LRU that also chose **which device computes** (GPU f16 vs CPU f32), so output depended on history. | **No.** It was rejected under §3.4, not for speed (the tier was faster, §7.0.2x). arcint's offline replay: LRU 55.6/69.2 % vs census 21.2/38.5 % at 32/64 slots per layer; a shared pool reaches 93.8 % at 16 GiB (`git show b0447b8:docs/campaigns/expert-hot-set-lru.md`, lines 447-455). |
| hybrid-expert-fetch "slower", closed as a verdict | FreeToken: miss fetches from **pinned** RAM, and **fetched experts stay cached**; the split is set by measured CPU vs link speed. Strata: the last ~20–55 % of misses go over PCIe from the pinned arena, issued at plan time. | Copies from the **pageable** bank (2.7 ms/layer stall); the pinned arm covered 16 of 512 experts (moved ~5 % of misses); **transient slots overwritten every step**, so nothing was cached; a fixed K=3. | **No.** Also, FreeToken's own rule (hybrid only when the CPU is > 2× the link; here ~1.2) would have chosen GPU "offload" mode on this host. |
| prefill-expert-streaming v1 "a loss as built", parked | FreeToken: two buffers, the next layer copied on its own copy stream with events while the current one computes, chunk 8,192. Strata: a 384-slot ring borrowed from the cache, every expert streamed at chunk ≥ 1,024, quantised (MMQ) kernels, chunks up to 8,192. | CPU threads memcpy'd into separate staging that **squeezed the bank** (disk faults); copies and compute on one queue, in series; no prefetch; chunk 512 (2,048 crashed, not investigated). | **No.** arcint's own ceiling with overlap: ~2.3× at chunk 512. Strata: IQ3_XXS 1,745 t/s at 32K. |
| 0075 zero-copy + spin dropped | Strata: a doorbell (a kernel writes x and the ids to mapped pinned memory with a fence; the CPU spins; a GPU wait kernel polls a host flag), and the whole pass is pre-recorded (`elementwise.cu:186-311`, `verify.cpp`). | Removed only the y-writeback copies. x readback hops and the per-layer host block remained. The spin's +6.4 % prefill was discarded by a conjunctive, decode-first gate. | **No.** arcint saw +45 % from device-side routing on the all-resident 35B (§7.0.2cr). |
| MTP on Flash-Next: "no head" (`serving-config-flash-next.md:73-79`) | Strata fetches the 31 `mtp.*` tensors (~5 GB) from the BF16 checkpoint (`tools/mtp_fetch.py`); up to 3 drafts, 50 % confidence gate; 1.6–1.8× with CPU experts. | Never built for Flash-Next; arcint's MTP path drafts 1 token. | **Never tried.** The premise was refuted: the HF index carries the head. |

## 2. Mechanisms arcint has never tried

From both references:
- a global, byte-sized expert-cache budget ranked across layers (Strata, +13 % decode);
- a decode-built seed with per-layer budgets (arcint's prefill-built seed underpredicts decode ~4×);
- a pinned host bank (the TTM cap was raised as out of scope, never tested);
- a benchmark that picks the mode (CPU vs link speed);
- graph capture or replayable command lists (`cl_khr_command_buffer`, Level Zero);
- multi-draft MTP with a reduced draft vocabulary;
- prompt lookup with a cost policy;
- prefill chunks above 2,048 on the B60;
- conversation-state checkpoints usable alongside the adaptive cache.

## 3. Rules in arcint that, read literally, block reference mechanisms

These are operator decisions, never agent decisions
([[feedback-invariant-vs-reference-escalate]]):
- **DESIGN §3.4** "byte-identical for any cache state". The 2026-10-01
  amendment relaxes it for the expert tier.
- **Amendment clause 2**, "a deterministic function of history since boot".
  This still forbids Strata's non-blocking admission (timing-dependent), its
  link-probed PCIe share and its timing-driven draft policy. Strata accepts
  2–5 % top-1 flips at equal perplexity
  (`bench/results/2026-09-27-cache-parity`).
- **Amendment clause 3** and `tier_prefix_cache_decision`: no prefix cache
  with an adaptive tier. Both references run the two together.
- **CLAUDE.md** "cold against warm cache … MTP on against off — must be
  byte-exact". This contradicts DESIGN 630-643 ("byte-identity under
  speculation not deliverable on this backend") and wasn't updated with the
  amendment.
- **Conjunctive, decode-first gates.** These dropped measured one-phase
  gains (0075's spin; 0074 at first).
- **QSA runtime refusals** of speculation and the prefix cache. Fixable:
  Strata appends to the indexer at commit time.
- **Campaign scope exclusions** removed exactly the parts that make the
  references' mechanisms pay: "Out: an LRU policy; pinning the bank" and
  "Out: an LRU across calls; raising the TTM cap".

[Status, 2026-10-01, later: DESIGN §3.4 Amendment 2 and `CLAUDE.md` lift the
first four items and the conjunctive gates (correctness judged at the
answer; deterministic replay a default; the prefix cache allowed with an
adaptive tier, its refusal in `src/config.cpp` owed a change; gates per
phase). The campaign rewrite of the same day dropped the scope exclusions.]

## 4. Ranked levers (plain language)

1. **Let the GPU's expert cache learn the conversation.** Use one pool across
   layers, batched swaps, never wait on a copy, and fill it from pinned RAM.
   Today about a third of the expert work runs on the GPU; both references
   and arcint's own replay say this can be reversed. (In progress.)
2. **Read prompts on the GPU, streaming the missing experts in big batches**,
   from pinned memory, with copies overlapping compute. Strata reads prompts
   15–30× faster than arcint; arcint's one attempt was a different design.
3. **Draft several tokens with the model's own prediction layer (MTP).**
   1.6–1.8× in Strata with CPU experts. The layer is in the original
   checkpoint.
4. **Raise the pinned-memory cap on the host** (`ttm.pages_limit`). Items 1,
   2 and 6 depend on it. (Done: the dev host's cap is 40 GiB.)
5. **Stop the 48 per-token hand-offs through the driver.** The GPU gets its
   work in advance and waits on a flag in shared memory.
6. **Choose who handles a missed expert (CPU or a copy to the GPU) from a
   measured speed comparison.** FreeToken's rule picks the GPU on this host;
   Strata sends ~30 % over a link like this one.
7. **Keep conversation state between requests**, for agents re-sending long
   histories.
8. **CPU worker housekeeping**: one pinned worker per core, the coordinator
   on its own core, spin before sleeping (+6.4 % prefill, measured and
   dropped).
