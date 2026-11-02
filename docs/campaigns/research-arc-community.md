# research-arc-community — what others run on Arc, in Chinese and English (2026-10-25)

A field survey for work on Arc that our references miss: searches in Chinese
(Zhihu, CSDN, LINUX DO, vendor blogs) and English (GitHub, forums, blogs) for
LLM serving and hand-written kernels on the A770 and the B-series. Evidence
class per row: `paper` (a post or README, not read at source), `code` (read
here), `snippet` (only a search result could be read).

## Sources

| source | what | class |
|---|---|---|
| `github.com/ef777/a770-qwen-xe1` (2026-10-08..11) | Qwen3.8-27B on one A770 16 GB (Windows, oneAPI 2026.1, llama.cpp SYCL at `889edf43` + three open SYCL PRs): SYCL/ESIMD kernels, MTP, launch settings; decode ~20 -> 40-50 t/s with 3 drafts, prefill ~413 -> 610-690 t/s at 12.7k | `paper` (README; the 8,000-line patch not read yet) |
| vLLM's Chinese blog, `blog.vllm.com.cn/2025/11/11/intel-arc-pro-b.html`; Zhihu `zhuanlan.zhihu.com/p/1971721550853014687` | Intel's vLLM on the Arc Pro B60: a persistent "zero-gap" MoE grouped GEMM claimed above 80 % of hardware efficiency, two SYCL groups per Xe core; sequence-parallel flash attention for long contexts | `paper` |
| `sanmai.github.io/slop/2026/05/08/llama-cpp-batch-size-arc-b60.html` | llama.cpp SYCL on a B60: `-b 4096 -ub 2048` against the defaults (2048 / 512) +25-30 % prompt throughput on a 35B-A3B MoE, +8 % dense | `paper` |
| `forum.level1techs.com/t/arc-pro-b60-for-local-llms/246971` | stock llama.cpp SYCL on a B60, a qwen35moe Q8_0: tg128 8.3 t/s, pp16384 111.6 t/s | `paper` |
| LINUX DO `linux.do/t/1834964` (2026-03-27, part 1), code `github.com/funkpopo/Anna` (`anna-serve`) | a PyTorch XPU serving engine (Windows): weights quantized to int4 at load, the context sized automatically below the VRAM limit, a text-only fast path that skips the multimodal wrapper; Qwen3.5-9B on an A770 11 -> 19 t/s with the fast path | `paper` (the operator's copy of the page; the repository not read) |
| LINUX DO `linux.do/t/topic/1849761` (2026-03-30, part 2 of a series) | a PyTorch XPU custom operator for Qwen3.5 on the A770/A750: one SYCL kernel for the gated delta-net's state update, its output, the per-head RMSNorm with `norm_weight` and the `silu(z)` gate; f16/bf16/f32 inputs, `beta`, `g`, the norm weight and the state in f32 (lower precision gave garbled text); Qwen3.5-9B BF16 12 t/s, first token 2 s (16 s the week before), 9.7 GB; part 3 below |
| LINUX DO, part 3 (2026-04-17), same repository | CPU-XPU hybrid MoE in the same engine: experts in host RAM, copied to the GPU when routed, decode bound by the copies; Qwen3.5-35B-A3B (Intel's int4 AutoRound) at 1 t/s on DDR4-3200; custom router, dispatch and scatter operators beside the GEMM ("routing, dispatch, scatter, the expert cache and the expert copies often matter more than the matrix multiply"); intermediates in f32 against garbled or looping text; operators registered with `TORCH_LIBRARY` (Meta + XPU), built with dpcpp | `paper` (the operator's copy of the page) | `paper` (a copy of the page from the operator: the forum's Cloudflare challenge stops headless fetchers) |
| CSDN (a 4 x B60 server with llama.cpp SYCL, Qwen3-32B), Tencent News (a 96 GB 4 x B60 review), CodeArts / AtomGit guides, EE Times China | multi-card setups and tuning guides; no kernel work | `paper` |
| Bilibili, Gitee | nothing found in two passes | — |

## Against arcint

The A770 project (`a770-qwen-xe1`, `paper`) arrived independently at much of
what our series holds: multi-column matvecs so one weight read serves an MTP
verify (our 0009), a frequency-ordered compact draft head (our 40,960-row
draft vocabulary), a GQA flash-decode kernel for 1-8 queries (our 0016-0018),
an ESIMD gated delta-net for decode and prefill (our 0003), async host input
uploads with one synchronization (our 0047), skipping an identical checkpoint
re-save. Its A770 numbers are a useful outside reference for our coder card:
raw DPAS ~200 int8 TOPS with 8 independent accumulators and ~37 with one
dependency chain; SLM gathers ~10 lanes a clock per Xe core, so an i-quant
matvec with a lookup per two weights is gather-bound (removing the gathers,
wrong results, IQ4_XS 188 -> 129 us); Q3_K matvec at ~340 GB/s, 91 % of the
measured read peak (~374 GB/s); an XMX matvec at 1-8 columns slower than
dp4a; an XMX flash-attention kernel correct but slower (SLM/barrier stalls).

## Leads, ranked

1. **The MTP head over the prompt's tail only** (`a770-qwen-xe1`,
   `LLAMA_MTP_PREFILL_TAIL=N`, `paper`): in prompt batches over 64 tokens only
   the last N go through the MTP head; N = 256: a 12k prefill 473 -> 570 t/s,
   code decode unchanged, prose decode -3 %. Our drafter runs every prompt row
   through the MTP context (`src/exec/llama_spec.cpp:7-21`), as Strata does
   (`research-strata-qwen36.md`); a served A/B on the agent and the coder
   (prefill, acceptance, the answer-level bar) is owed.
2. **Intel's persistent MoE grouped GEMM** (vLLM / llm-scaler on the B60,
   `paper`): the coder's matmuls are 9.9 ms of a 19.25 ms token
   (`research-strata-qwen36.md`, `measured-here`); read the kernel at source
   (vllm-xpu-kernels) before any claim.
3. **The ubatch for prefill** (the B60 blog, `paper`): our coder and agent run
   `n_ubatch = min(n_batch, 512)` (`src/exec/backend_llama.cpp:586`); a served
   sweep at 512 / 1,024 / 2,048 has never been run (checked against the record,
   2026-10-25); the agent's VRAM margin limits it.
4. **The A770 kernel ceilings above** as targets for the coder's kernels
   (DPAS with independent accumulators; gather counts in the IQ paths).
5. **The gated delta-net with its output norm and z-gate in one kernel**
   (the LINUX DO operator, `paper`; Strata's `k_gdn_seq` with the gated norm
   inside, `code`, `research-strata-qwen36.md`): the coder's re-count runs the
   delta-net (0.35 ms a token) and then separate norm and gate kernels; a
   fusion-campaign item, with the state kept in f32 as both do.

The LINUX DO series confirms from outside what the coder's re-count shows: on
Arc the MoE's routing and small ops cost as much as its matmuls. Its copy-on-
miss hybrid (1 t/s) is the mechanism Flash-Next's expert cache replaced (the
hot experts resident, the misses computed on the CPU: 33-38 t/s on the B60,
`measured-here`).

The stock path is far behind ours on the B-series: 8.3 t/s for a qwen35moe
Q8_0 on llama.cpp SYCL (`paper`) against arcint's coder at 44-79 t/s on the
A770 (different quant, card and configuration; not a like-for-like
comparison).

## Addendum 2026-11-02: Torchit1's llama.cpp `arc-b580` branch

Source: the fork's branch `arc-b580` (head `e04b0cd4`, 222 commits on upstream `5ea87dd`), its guide
`docs/bonsai-arc-b580.md` and `ggml/src/ggml-sycl/fattn-dec.cpp` (`code`, read 2026-11-02). Target: one Arc B580
(Xe2, the B60's family), Linux, oneAPI 2025.3, Level Zero 26.35, PrismML's ternary Bonsai 2 27B (a Qwen3.5-style hybrid
with an MTP head) and Gemma 4 12B, 128K context, q4_0 KV cache. The guide's numbers are the fork's (`paper` for us):
Bonsai fresh code 90 t/s with MTP drafts, 42 t/s plain, 44 t/s new code at ~115K context; Gemma 4 12B QAT q4_0 from
~40 to ~102 t/s on new code with its drafter.

What is portable to arcint's OpenCL kernels, by mechanism (`code`, fattn-dec.cpp's own comments and kernels):
- **Decode and verify attention straight from the quantized cache on XMX** (`GGML_SYCL_FA_DEC_DPAS`, opt-in): one
  work-group per KV head and a context slice serves all G query heads of the GQA group and all verify tokens from one
  read (flash-decoding); Q.K as s8 x s4 DPAS (Q rows quantized to int8 once a call, the q4_0 bytes XOR 0x88 as the B
  operand, inline vISA), P.V as f16 DPAS with V dequantized in registers; K/V tiles staged through SLM with coalesced
  loads, the next phase's keys prefetched, the 256-register file for 3+ query tokens (their measurement: 6-17 %
  faster per layer at 128K). Their scalar predecessor was f32-ALU bound at 4 tokens (Q.K 163 + P.V 121 of 436 us a
  layer at 16K). **Corrected the same day by the concept mining (`research-concept-mining/arc-b580/`, checked against
  `contrib/llama.cpp/README.md`):** arcint's 0016 already takes the GQA-shared DPAS form on the B60 from 4 rows (one
  work-group per KV head and split, QK and PV on XMX, after NInfer's small-T kernel); a quantized K is staged as f16 in
  local memory first, and 1-3 rows stay on 0015's split ALU kernel; the A770 keeps 0015. What the fork adds is the
  int8 DPAS straight on the cache's bytes (no f16 conversion; our direct-K form measured 30.7 against 52.6 t/s staged
  because every sub-group converted the same blocks) and a staged DPAS kernel for 1-3 rows. Our K is q8_0, so the Q.K
  would be an s8 x s8 DPAS.
- **Gated DeltaNet single-token fusions** (conv chain, gate block, fused state writes; `GGML_SYCL_CONV_CHAIN`,
  `GGML_SYCL_DN_GATES`): the hybrid Qwens' linear-attention layers. To compare against what 0059-0066 already fuse.
- **Two-pass top-k for long rows**, **same-input mat-vecs in one launch** (q/k/v, gate/up), **a sync trace that
  attributes every synchronize** (`GGML_SYCL_SYNC_TRACE`): the last is a cheap diagnostic worth copying.
- **Their MTP draft chain measured -0.2 to -0.8 % on the B580** and is off by default; "one slot per card, keep n-gram
  drafts on" (their serving notes, B70s 31-74 % acceptance at one slot against 21-31 % at four).

A record to re-test, not a mechanism: they run **SYCL on Linux on a B580, which only the xe kernel driver supports**, and
default to the legacy Level Zero adapter (`SYCL_UR_USE_LEVEL_ZERO_V2` opt-in) as faster. arcint's record says SYCL is
dead under xe (a memcpy abort). That negative predates this and may have been the V2 adapter or the L0 loader
(memory: oneAPI 2026.1 needs a newer loader); by the reference-audit rule it is not settled until the legacy adapter
is tried on the B60.
