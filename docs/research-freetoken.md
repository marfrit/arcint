# Research: FreeToken (arXiv 2608.16157) — mapped to arcint

Paper: Yang et al., "FreeToken: Efficient Edge-Native MoE Serving with
Bandwidth-Adaptive Execution," arXiv:2608.16157, 2026-08-17. Repo:
`github.com/FlashML-org/FreeToken` (Apache-2.0; CUDA-only per §4.2).

This file transcribes the paper's mechanisms that arcint's design doc
cites, one section at a time, and states arcint's deviation from each
in a separate paragraph. Paper text and arcint policy never share a
paragraph — a rule the design doc's own reviewer flagged before this
file existed.

---

## §3.2 The q* crossover — CONFIRMED

Paper (§3.2, "Measured bandwidths determine how misses are served",
Equations 1–4): let `S` be the size in bytes of one complete expert,
`m` the number of missing experts at a decode step, `B_P` the pinned
expert-transfer PCIe bandwidth, `B_H` the host-side expert-processing
bandwidth. The misses are split into a cache-fill set `F` (size `q`,
transferred and evaluated on the GPU) and a CPU-execution set `C`
(size `m − q`, evaluated in place on the host). PCIe fill and CPU
execution run concurrently and read from the same host-memory
subsystem, so a saturated transfer leaves residual host bandwidth
`B_R = max(B_H − B_P, 0)`. The two branches finish in

    T_fill(q)  ≈ q · S / B_P
    T_cpu(m-q) ≈ (m-q) · S / (B_H − B_P)

Balancing gives `q / (m-q) ≈ B_P / (B_H − B_P)`, so

    q* ≈ m · B_P / B_H         (Equation 4)

`q*` is rounded to an integer at runtime; the cache always keeps at
least one fill so it continues warming. `B_P` and `B_H` are
"empirically profiled on the target hardware at deployment" — the
formula is not applied from spec sheets.

**arcint deviation.** arcint's `B_P` is over PCIe 4.0 x16 to an Intel
Arc card (OpenCL host-to-device queue, not CUDA's pinned-memcpy path
FreeToken measures). arcint's `B_H` is the effective bandwidth of an
AVX2 dequant-and-GEMV kernel for K-quant (Q4_K/Q5_K/Q6_K) rather than
FreeToken's SIMD kernel for BF16/NVFP4/MXFP4 experts (§4.2). Both `S`
and the resulting `q*` differ per-format, so arcint calibrates one
`q*` per served type. The formula stays; the inputs are the deployed
machine's.

---

## §3.2 Semantic-aware expert caching (LRU) — CONFIRMED as design shape

Paper (§3.2, "Semantic-aware expert caching follows the model's
evolving computation"): a shared LRU residency table on GPU is
maintained across decode steps, exploiting "strong temporal expert
locality: across consecutive steps, the same MoE layer repeatedly
routes to overlapping or recently used experts" (citing Liang et al.,
2025). Cache hit refreshes recency, cache fill admits a newly
selected expert, eviction removes the least recently demanded expert.

**arcint deviation.** arcint's existing host-tier (patches
0011–0012, extended by 0017–0019) uses a static-partition residency
fix (F2 in patch 0018/0019 per the design doc), not an LRU. Adopting
the LRU shape is deferred until FIX A's export blocker clears (no
Flash-Next artifact to route against yet). The design doc's FIX E
section names the crossover with LRU as the target; today's arcint
does not yet run one.

---

## §3.3 CPU-resident expert pool as source of truth — CONFIRMED

Paper (§3.3, "Elastic Memory Management for Edge-Native Runtimes"):
"the CPU-resident expert pool remains the source of truth, GPU memory
affects only performance, never correctness." Runtime cache
reconfiguration divides the GPU-memory budget between KV cache pages
and complete-expert slots; this can be rebuilt at any scheduler safe
point without reloading the host pool.

**arcint deviation.** arcint's existing static-residency fix already
treats the host copy as the source of truth, matching this shape.
Runtime cache reconfiguration is not landed; arcint fixes the
allocation at load and does not resize mid-session.

---

## §3.1 Prefill: full-layer double buffering — CONFIRMED

Paper (§3.1, "Full-layer double buffering hides transfer behind
computation"): two full-layer buffers from the global slot pool; the
GPU computes routed experts of layer `l` from one buffer while a
transfer stream loads the full expert set of layer `l+1` into the
other. Falls back to on-demand load if two full layers do not fit.

**arcint deviation.** arcint has no prefill double-buffering today.
The FIX C kernel work does not overlap layer-l compute with
layer-(l+1) transfer. Adopting it is FIX E-scoped, not in this
release.

---

## §3.1 Semantic-anchored recurrent-state checkpoints — DEVIATION (out of scope for 0.5.0)

Paper (§3.1, "Semantic anchors preserve recurrent state across context
edits"): hybrid-attention models compress the past into an evolving
recurrent state that cannot be partially reused, so prefix reuse
depends on periodic checkpoints. FreeToken places these at "semantic
anchors" — thinking-block boundaries, tool calls, tool outputs,
conversation turns — because agent frameworks (OpenClaw, OpenCode,
SWE-agent) edit context along exactly those boundaries.

**arcint deviation.** arcint 0.5.0 does not restore recurrent-state
checkpoints across turns. The paged serve loop starts each new
request from scratch. This is out of scope for this release; not a
correctness gap for isolated requests, only a wasted-work gap for
long agent turns. The FIX 3–6 unit-side items address other prefix-
reuse gaps (`--prefix-cache-mib`) which are the KV-cache side of the
same problem.

---

## §4.2 Expert storage (FTW format, bank layout) — DEVIATION

Paper (§4.2, "Expert banks and the FTW format"): FreeToken normalises
checkpoint layouts into a small set of expert banks, each keyed by
the flattened `l · E + e` layer-expert identifier. The FreeToken
Weight (FTW) format merges expert weights into runtime bank layout
ahead of time; loading skips tensor discovery and repacking with
parallel direct I/O into exact-size host banks. `bytes/element`:
BF16/NVFP4/MXFP4, not GGUF (§4.2).

**arcint deviation.** arcint reads GGUF files directly (`gguf_apply_to_
template` in `src/exec/gguf_graph.cpp`); the K-quant (Q4_K/Q5_K/Q6_K)
row-major layout the file ships in *is* the runtime layout, no
repack ahead of time. When the incoming type is Q4_K, the repacker
runs at load (0.4.1's "repack" mode); K-quant blocks stay
256-element, per this repository's K-quant path. FreeToken's FTW
format is not adopted. arcint's format decisions are driven by GGUF
compatibility, not by an ahead-of-time bank pack.

---

## §4.2 Pinned-memory fallback (pure-CPU MoE) — CONFIRMED (analogue exists)

Paper (§4.2, "Platform adaptation"): when the complete expert pool
cannot be pinned or registered for DMA, FreeToken falls back to a
pure-CPU MoE backend where all routed experts execute on the CPU and
only activations, routing metadata, and aggregated outputs cross the
CPU-GPU boundary.

**arcint deviation.** arcint's existing MOE_CPU_TIER (patches
0011–0012) is the equivalent knob; it activates via `--host-tier
<ratio>` and is measured on the served 35B MoE at 15.0/15.5 t/s at
ratio 50 with the host tier on, versus 10.4/10.6 without (design doc
FIX C's own reference to §7.0.2 measurements). The mechanism is the
same shape; the trigger is user-set on arcint, driver-conditioned on
FreeToken.

---

## §5.1 Measurement protocol — CONFIRMED

Paper (§5.1, "Hardware"): "All bandwidths in Table 1 are measured on
the deployed tensor shapes rather than taken from platform
specifications." Rented dual-socket servers are CPU-capped to 6
threads and pinned to the GPU's NUMA node to match the effective
host bandwidth of two real edge machines (53.8 GB/s on a
16-core desktop, 47.5 GB/s on a 14-core laptop). The 5090 desktop
row: RTX 5090 (32 GB, PCIe 5.0 x16), B_P 49.0 GB/s, Ryzen 9 9950X3D,
DDR5 192 GiB, B_H 53.8 GB/s.

**arcint deviation.** arcint measures on Zen 3 with DDR4 dual-channel
(the dev host), Arc A770 16 GB with PCIe 4.0 x16, not on the 5090's
PCIe 5.0 x16 with DDR5. The two numbers are neither the same regime
(DDR4 ~44 GiB/s single-thread read vs FreeToken's DDR5 desktop
53.8 GB/s aggregate host) nor the same card class. Any arcint claim
against FreeToken's row must state this. The measurement discipline
— name every number's card, precision, depth, host config — is the
same.

---

## §5.2 Baseline claim: 77–83 tok/s on Qwen3.6-35B-A3B — CONFIRMED with a precision caveat

Paper (Abstract; §5.2, first sentence): "on an RTX 5090, FreeToken
sustains 77–83 tok/s on Qwen3.6-35B-A3B." Precision (§5.1 first
paragraph, §5.2 caption): **BF16**, not int4. RTX 5090 is 32 GiB VRAM,
PCIe 5.0 x16.

**arcint deviation.** arcint's served 35B MoE artifact is Q4_K_M
(int4 K-quant, ~18.6 GB weights on disk), not BF16 (~70 GB). The
served card is Arc A770 (16 GiB), not RTX 5090 (32 GiB). Comparing
arcint's Q4_K_M throughput on A770 against FreeToken's BF16 on 5090
is a cross-precision AND cross-hardware comparison; any claim table
must state both differences on the same row rather than leaving
either implied. On the 4060 laptop (8 GB, PCIe 4.0 x8, LPDDR5),
FreeToken serves the 35B at 39.3 tok/s in NVFP4 — nearer in
precision to arcint's int4 but on a different quantisation family.

---

## Not in FreeToken: per_layer_token_embd (FIX D's n-gram table)

The paper has zero references to n-gram embedding, per-layer token
embedding, PLE, or `per_layer_token_embd`. Grep over the extracted
paper text (`freetoken/paper.txt`, 9218 words) for `n-gram | per_layer
| embedding table | PLE | ple_` returns zero matches under FreeToken's
own mechanisms (the incidental hit at line 165 is a description of
gated DeltaNet in Qwen3.6, not FreeToken's own design). The n-gram
embedding table is a **Qwen Flash Next** architectural feature (config
keys `ngram_size`, `ngram_vocab_size_base`, `heads_per_ngram`,
`ple_layer_ids`, `ple_embed_dim` on the checkpoint), and vLLM's
`VLLM_PLE_CPU_OFFLOAD=1` is vLLM's own implementation of hosting it.

**arcint deviation.** FIX D (`docs/design-qwen-flash-next.md` §"FIX D
— N-gram table host-offload and dequantise-on-gather" and this file's
"Not in FreeToken" section together) is **arcint-original work**. The
kernel (`src/exec/ngram_gather.h`), the memory-budget arithmetic
(`src/exec/fit.h::ngram_table_bytes` + `host_ram_fit_must_refuse`),
and the Q4_0/Q4_1/Q8_0 32-element-block support in
`src/core/gguf_dequant.cpp` are not derived from FreeToken's paper or
repo; the two efforts happen to touch overlapping problems
(host-resident table + PCIe crossover) but the n-gram table is not
FreeToken's design. Any repository claim that attributes FIX D to
FreeToken is UNSUPPORTED and should be reattributed.

---

## Claim sweep across `docs/design-qwen-flash-next.md`

Fifteen FreeToken/q\* references, in file order, with disposition:

1. Line 84–89 (recon-summary paragraph): "**77–83 t/s decoding a 35 B
   MoE on a 32 GB card**. They serve Qwen3.6-35B-A3B — our own served
   family — so the comparison is direct." — **CONFIRMED with
   precision caveat**: paper §5.2 first sentence, but BF16 not int4;
   32 GB is RTX 5090 (PCIe 5.0), not A770 (16 GB, PCIe 4.0). The
   "comparison is direct" clause overstates: family-direct but
   precision-and-card-cross.
2. Line 191–196 (FIX E scope): "FreeToken design, re-implemented
   natively — extracted from the paper and docs, not their code" —
   **CONFIRMED as approach**.
3. Line 997 (FIX E scope): "the FreeToken comparison protocol" —
   **CONFIRMED**: paper §5.1 measurement protocol.
4. Line 1015–1017 (Why a new dequant path): "FreeToken's contribution
   is exactly that crossover heuristic" — **CONFIRMED**: paper §3.2
   Equation 4, this file above.
5. Line 1038–1041 (Bandwidth calibration): "FreeToken's `ft bench bw`
   heuristic (paper §4, 'Bandwidth-Adaptive Execution')" —
   **CONFIRMED**: paper §3.2 (not §4; the "at deployment" profile
   step is §3.2's own last paragraph). Section number in the design
   doc is off by one.
6. Line 1045–1046 (measurement discipline): "measured on arcint's own
   hardware and kernels rather than assumed from FreeToken's
   published figures" — **CONFIRMED**: matches paper §5.1's stance and
   this repository's own §7.0.1.
7. Line 1113–1116 (crossover point): "`q* ~= m·B_P/B_H`" —
   **CONFIRMED**: paper Equation 4 verbatim.
8. Line 1169 (What's landable): "the K-quant kernel and calibrated
   `q*` split as the only new inputs" — **CONFIRMED as target design**;
   both are FIX E work, not landed today.
9. Line 1193 (Overlap strategy): "the calibrated `q*` split allows,
   rather than treating GPU idle time" — **CONFIRMED** as the paper's
   §3.2 concurrent-branches shape.
10. Line 1214 (DRAM contention): "the crossover `q*` used at serving
    time is the contended one" — **DEVIATION**: FreeToken measures
    B_H under the deployed kernel (paper §5.1 "measured on the
    deployed tensor shapes"), which is contended-with-itself but not
    contended with an unrelated stream. arcint's FIX D + FIX E jointly
    contend a single DRAM budget, which is a stricter measurement the
    paper does not perform.
11. Line 1236 (Calibration profile): "the derived `q*` crossover per
    K-quant type, both contention cases" — **DEVIATION**: paper reports
    one `q*` regime per hardware (Table 1's per-machine B_H/B_P);
    per-type `q*` (arcint's Q4_K/Q5_K/Q6_K variants) is arcint-
    original, forced by GGUF's per-type dequant bandwidth differences.
12. Line 1247–1254 (Comparison methodology): "FreeToken's paper
    reports Qwen3.6-35B-A3B ... at 77–83 tok/s on an RTX 5090" —
    **CONFIRMED**: paper §5.2 first sentence.
13. Line 1250–1251 (memory citation): "the `reference-freetoken-edge-
    moe` memory note" — meta reference, no verifiable claim.
14. Line 1273–1276 (honesty constraint): "if FreeToken has no row at
    all comparable in card class or model to what arcint actually
    measures, the claim row must say that gap out loud" —
    **CONFIRMED**: paper's own Table 1 has no A770 / Arc-class GPU
    row (5090, 4090, 3090, RTX PRO 6000, plus 5090 desktop + 4060
    laptop). Any arcint row on the A770 has no FreeToken comparable.
15. Line 1319–1321 (What's blocked): "The final FreeToken claim row's
    arcint-side numbers for Flash-Next" — **DEVIATION**: FreeToken
    reports on Qwen3.6, DeepSeek-V4-Flash, and GLM-5.2 in its main
    tables; Flash-Next itself is not in the paper's evaluated model
    list. An arcint Flash-Next row against FreeToken is a
    cross-model comparison, and the design doc's own honesty
    constraint (item 14 above) applies.

**Not previously flagged but should be**:
- The "vLLM ships `VLLM_PLE_CPU_OFFLOAD=1`" comparison line in FIX D
  (design doc line 843–847) is CONFIRMED as vLLM's own PLE mechanism
  (per its released docs; not verified in this file's fetch). vLLM's
  PLE offload is not FreeToken's — that citation is correctly
  attributed to vLLM.
- FIX D's n-gram table itself (`per_layer_token_embd`) is
  **arcint-original**, not FreeToken-derived. The design doc does
  not currently misattribute it, but the WP-R sweep confirms this so
  a future editor does not accidentally add such a citation.

---

## Repo (github.com/FlashML-org/FreeToken) — not fetched here

The paper's abstract points to `flashml.ai` and the repository at
`github.com/FlashML-org/FreeToken` (Apache-2.0). The paper §4.2 is
explicit that the CUDA kernel path is not portable ("FreeToken
selects GPU kernels compatible with the expert representation, GPU
architecture, and CUDA environment") and arcint's runtime is
OpenVINO on Intel Arc, so the repo's kernel code cannot be lifted
without a full port. The design doc's FIX E already records this ("no
SYCL/OpenVINO path, and Apache-2.0 fork tax would be permanent",
design doc line 971). Repo fetch is deferred; the paper text is
enough to place each claim in this file's sweep.

---

Fetched 2026-09-10, arxiv HTML v1 (`arxiv.org/html/2608.16157v1`,
168 KB), abstract from `arxiv.org/abs/2608.16157`. The paper's own
publication date is 2026-08-17 (arXiv metadata `citation_date`);
authors Yang, Shuo; Fan, Xiaoze; Pan, Melissa; Xi, Haocheng; Wang, Zhe;
Sun, Shanlin; Keutzer, Kurt; Han, Song; Zaharia, Matei; Xu, Chenfeng;
Stoica, Ion.
