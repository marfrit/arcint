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
| LINUX DO `linux.do/t/topic/1849761` (2026-03-30, part 2 of a series) | a PyTorch XPU custom operator for Qwen3.5 on the A770/A750: one SYCL kernel for the gated delta-net's state update, its output, the per-head RMSNorm with `norm_weight` and the `silu(z)` gate; f16/bf16/f32 inputs, `beta`, `g`, the norm weight and the state in f32 (lower precision gave garbled text); Qwen3.5-9B BF16 12 t/s, first token 2 s (16 s the week before), 9.7 GB; part 3 (2026-04-17, MoE, "write your own operators") not read | `paper` (a copy of the page from the operator: the forum's Cloudflare challenge stops headless fetchers) |
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

The stock path is far behind ours on the B-series: 8.3 t/s for a qwen35moe
Q8_0 on llama.cpp SYCL (`paper`) against arcint's coder at 44-79 t/s on the
A770 (different quant, card and configuration; not a like-for-like
comparison).
