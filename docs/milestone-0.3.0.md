# 0.3.0 — the extension series (M7–M14)

0.2.x closed the correctness line: M0–M6 are done, the engine serves two cards,
the gates in DESIGN §5 hold, and every number in DESIGN §7 names its card and
configuration. 0.3.0 is the performance-extension line — the features that sit
on top of a correct engine the way [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
sits on top of llama.cpp: quant mixes, offload policy, KV codecs, drafting,
auto-fit. Writing OpenVINO plugin patches is **in scope** for this series; the
`patches/` series already ships in the source-built plugin package and this
plan adds to it.

## Ground rules

1. **Every item ends in a measurement**, with the card, depth, KV precision and
   configuration named. The §5 gates apply unchanged: Prüfstand 10/10 on the
   coder artifact, byte-exact equivalence wherever the change claims to be
   invisible, red case first.
2. **Plugin patches need a fusion-impact profile**, not a kernel
   micro-benchmark. The 8.6×-faster custom kernel that lost 1.6 ms end-to-end
   (DESIGN §8) is the standing warning: a patch is judged by what the graph
   optimiser does around it.
3. **No item closes as "hardware baseline reached" without a survey.** Closing
   a performance milestone requires a recorded look at what other stacks do on
   the same silicon — upstream OpenVINO release notes, vLLM/SGLang XPU,
   ik_llama.cpp's CPU-side tricks, independent testbeds — not just our own
   numbers. The one time this repository assumed the hunt was over, the answer
   was a 44.8 t/s drafter someone else had already trained. A clean survey
   ("nobody does better on this path, here is what I checked") is a valid
   close; an unsurveyed one is not.

## Where the bar sits (survey, 2026-09-01)

| stack | result on comparable silicon | source |
|---|---|---|
| arcint 0.2.12 | coder int4: 71.3 t/s plain / 44.8 t/s DFlash-assisted agent on the B60; 24.0→44.8 t/s drafter gain; 35B at `--offload-ratio 20` on the A770: ~1.9 t/s | DESIGN §7.0.2r/s |
| vLLM XPU | 83.5 t/s, 20B model, 2× Arc B580 | xda-developers survey |
| upstream OpenVINO 2026.2 | MoE disk offload, batched-MoE decode optimisation, PagedAttention tree-mask (EAGLE-3), XMX dynamic quantisation | release notes |
| ipex-llm (archived 2026-01) | FlashMoE served 235B-class MoE on 1–2 Arc cards before archival | project README |
| DFlash-TfM (Mirai Labs) | τ 8–10 tokens/cycle with tree drafting vs 2.67 chain, same DFlash checkpoint family (B200, SGLang) | arXiv:2607.06763 |

Two entries in that table are direct evidence against "we are done": the tree
drafting τ, and upstream's own MoE decode work landing in the very plugin we
patch.

## The template, mapped

| ik_llama.cpp feature | arcint 0.3.0 analog |
|---|---|
| auto-fit offloaded tensors to VRAM, per-GPU margin | M7 |
| quantized KV (`q8/q6/q4`), Hadamard K/V transforms | M8 |
| `--cpu-moe`, tensor overrides, hybrid offload strategy | M9 |
| SOTA sub-4-bit quants (IQ_K, trellis), custom quant mixes | M10 |
| ngram/suffix self-speculation, DFlash, MTP | M11 (MTP/DFlash already served in 0.2.x) |
| iqk AVX2/AVX-512 GEMM kernels (150–350% CPU prompt speedup vs upstream) | M14 |
| fused MoE FFN (`-fmoe`) | M12 (exporter lowering so `fuse_moe_experts` fires) |
| — (llama.cpp: `--no-mmproj-offload`) | M13 |
| MLA/FlashMLA, Bitnet, tensor parallel, sampler zoo | deliberately excluded (below) |

## Milestones

| # | milestone | exit criterion |
|---|---|---|
| M7 | **Auto-fit and the honest reservation** — trial-allocate weights, expert slots and KV at load (deferred commit); on failure, shrink and **replay** the placement until it fits; print a max context that is true | With `--n-ctx` omitted, the printed max context loads *and completes a generation at full depth* on both cards, with and without `--offload-ratio`; the 0.2.12 "reservation under-counts" warning is deleted because the condition is gone; promised-vs-actual free memory error ≤ 2%. **2026-09-02: met** — auto-adopt served at 155,488 vs the hand-tuned 155,648 on the B60 (one live overshoot-correct pass); 0.8% promised-vs-actual on the A770 at ratio 20; the slot pool measured host-side (GTT), so the reservation keeps device and host ledgers (DESIGN §7.0.2t) |
| M8 | **Asymmetric paged KV: q8 keys, q4 values** — plugin patch: PagedAttention takes per-tensor cache precision (u8 K / i4 V) instead of one for both; keys hold the rope structure and are the sensitive half | Prüfstand 10/10 on the coder (named card, u8-K/i4-V), KV bytes/token reduction measured (the −36% row figure predates the padding arithmetic — real span −18…25%, §o-m8 design), decode within 5% of u8/u8 at 32k depth; a KLD-style probe against u8/u8 recorded. Stretch, same slot: the int8 group-64 codec with fused Hadamard pre-rotation (DESIGN §8), with the required fusion-impact profile. **2026-09-02: arcint groundwork landed** — `--paged-kv KEY[:VALUE]`, bitwidth-true cost model (fixed a 2× i4 over-count and a silent env fallback), `VALUE_CACHE_PRECISION` plumbing with the fail-loud ladder proven on-card ("Option not found … refusing"), per-port bitwidth audit (u8↔i8 alias measured and accepted). Symmetric `u4` serves and prices (6.3 vs 11.3 KiB/token, 35B). **Reduced to asymmetric REFUSAL** (§7.0.2w): the decode kernels derive both operands' packing from one config value, so u8:i4 can only fail loud (proven on-card), not serve — the durable win is the arcint ladder. Plugin patches 0008/0009 **parked, measurement-blocked** by a layout-sensitive u8 miscompile, NOT applied to production; kernel rewrite + i4/i4 pricing + KLD probe are future work behind a cold-build A/B |
| M9 | **Expert offload v2: feed-forward tensors on their own budget** — the FFN expert tensors are the offloadable class, dense/attention tensors are device-resident by priority; async batched host→device slot uploads (pinned staging, ordered queue), prefetch on router output, LRU persistent across requests | The measured blocker falls: slot-upload time overlapped with compute instead of 309 µs/tensor synchronous (13.1 s per 16 tokens at ratio 20 today); 35B q4 on the A770 at ratio 20 reaches ≥ 8 t/s decode **or** the item closes with a profile naming the next blocker; LRU hit rate reported per run. **2026-09-02: substantially met** — patches/0004–0007 (device-tier pool + async uploads + finish-gating): 0.4 → 9.1 t/s at ratio 50 with an 8 GiB device pool (4.9 at ratio 25); the dial verdict of §7.0.2s overturned (§7.0.2v). Counters read after the flush fix: avg copy 309→5 µs/tensor, tensor_loads = misses×9 (reconciliation resolved), evictions 0 (hot-set pin deferred by the design's own rule). Owed: equivalence+Prüfstand gates before rollout |
| M10 | **Sub-4-bit expert weights** — mixed per-tensor quantisation: rarely-routed experts below int4 (Q3-class ~3.2–3.5 bpw), dense and shared tensors at int4/int8; route is NNCF mixed-precision export or GGUF K-quant import with dequant-to-int4-at-load, decided by measurement | A mixed artifact serves with ≥ 15% more max context than pure int4 on the same card at Prüfstand 10/10; per-expert bpw map recorded in the artifact; quality delta vs int4 measured, not asserted |
| M11 | **Drafting II: trees from marginals** — a Weaver-class conditional re-ranker (~57M adapter, top-K-restricted, trained per the TfM recipe) over the served DFlash2 head; upstream PA tree-mask (2026.2) for the attention layers; a GDN tree-verify (masked triangular solve over the ancestor partial order) as a plugin patch — **the state is never speculatively written; the accepted path is replayed at commit**, which is the invariant our chain integration already holds; an ngram/suffix table drafter for the drafter-less endpoints | Chain re-rank first: tokens/verify-cycle > the measured 3.13 (B60, int4, block 8) at byte-exact greedy equivalence; tree path after: τ and t/s vs chain on the same card, or a documented negative with the profile. Also closes the 0.2.x leftovers: acceptance under the thinking template and under prefix-cache hits, and a Prüfstand run with the drafter on |
| M12 | **Host-loop and exporter** — pin the dispatch thread; lower the exporter's MoE to the per-expert pattern upstream `fuse_moe_experts` matches, so the fused path (and upstream's batched-MoE decode work) applies to our IRs | `fuse_moe_experts` proven to fire on an exported IR (graph dump); M=2 verify host cost re-measured against DESIGN §7.0.2p's numbers; per-cycle host ms delta named. **2026-09-02, corrected on-card:** the pass that matters is `ConvertTiledMoeBlockToGatherMatmuls` (tiled/compressed contract, §7.0.2u), proven ×40 in the served 35B's graph; the exporter emits the tiled form (`--moe-lowering tiled`) and the real head exports and compiles at production dims (946 MB u8 vs 1.69 GB f16 batched) — fused-op demonstration outside the engine's own base-model compile still open (one named discriminator). Dispatch pinning measured a null on a quiet host (`--pin-dispatch` kept for the contended arm) |
| M13 | **Vision projector disabled** — load the `*ForConditionalGeneration` IRs with the vision tower and projector excluded (they are in every checkpoint we serve); reserve `--vision` for later; pay zero VRAM for the modality we don't serve | Multimodal IR loads with vision excluded; text path byte-identical to the text-only export; VRAM delta of the excluded tower measured; request parser still rejects image parts explicitly |
| M14 | **CPU compute tier** — kernels vendored from a license-compatible high-performance CPU engine (candidate of record: ik_llama.cpp's iqk GEMM/GEMV kernels, MIT — compatible with this repository's Apache-2.0 under the `THIRD_PARTY.md` convention), tuned for the Zen 3 host class (Ryzen 7 5700X: 8C/16T, AVX2+FMA3, **no** AVX-512 and no AVX-VNNI, so the int8 path is the `maddubs` AVX2 one, not `HAVE_FANCY_SIMD`; 32 MiB L3; dual-channel DDR4 ≈ 50 GB/s ceiling). Primary use: **offloaded expert FFNs computed in place on the host** — the `--n-cpu-moe` analog — so M9's spill class stops crossing the bus at all; secondary: a full-CPU fallback engine for drafter-less and card-down operation. Ties into M10: the iqk kernels natively compute the K-quant/IQ_K sub-4-bit formats a mixed artifact would store | Expert-FFN-on-host beats M9's upload path on the same 35B ratio-20 config (per-token expert compute vs measured 309 µs/tensor uploads, profile recorded); full-CPU Prüfstand 10/10 on one model with t/s reported against the DDR4 bandwidth ceiling, not against GPU numbers; vendored files land in `third_party/` with license and version pinned in `THIRD_PARTY.md` |

Ordering intent, not a promise: M7 and M12 first (they make every later
measurement honest and cheaper), M8/M9 next (memory levers), M10/M11 after
(they consume the levers), M14 alongside M9/M10 (it is the other half of the
expert-placement question and shares M10's quant formats), M13 whenever an
export forces the question.

## Deliberate exclusions

- **Tensor parallel / multi-GPU disaggregation** — the two cards serve two
  models; the state-transfer contract note (DESIGN §8) stands for the day this
  changes.
- **MLA/FlashMLA** — no DeepSeek-family target in the allowlist.
- **Sampler zoo** (XTC, top-n σ, adaptive-p) — the operator layer covers
  serving defaults; exotic samplers are demand-driven, not planned.
- **Bitnet** — no artifact, no demand.

## Provenance

The feature set collects every open thread on the record: DESIGN §8's
deferrals (KV codec, MoE placement, host tier, multimodal), §7.0.2s's offload
verdict ("a fit lever, not a context dial" — M7/M9 are the two halves of
making it a context dial after all), the enqueue-bound follow-ups (M12), the
DFlash serving leftovers and the Trees-from-Marginals result (M11,
arXiv:2607.06763), and the fit/deferred-commit/replay, asymmetric-KV,
FFN-offload, projector-disable and LRU techniques surveyed from the
llama.cpp/ollama fitting work and ik_llama.cpp (M7–M9, M13). M14 vendors
rather than reinvents: the donor must be license-compatible (MIT and
Apache-2.0 both qualify; ik_llama.cpp is MIT, copyright ggml/llama.cpp/
ik_llama.cpp authors), carried under the existing `THIRD_PARTY.md` rules with
licenses unchanged.
