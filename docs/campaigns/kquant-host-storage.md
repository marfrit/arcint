# kquant-host-storage — the CPU tier decodes each quantised block once for all tokens of a call

**Open.** The native-format half is built: the host tier computes the
checkpoint's own IQ3_XXS / IQ4_NL / IQ4_XS / Q8_0 / IQ2_S blocks (patch
0043), and decode-shaped calls (≤ 8 jobs) dot in the quantised domain with
AVX2 (patch 0074, `MOE_CPU_TIER_Q8_DOT`). The open lever is the multi-token
form for prefill-shaped and verify-window calls.

## Charter

For a call in which an expert serves several tokens, decode each 32-value
chunk of the weight row once (grid lookups, signs, scale) and apply it to
every token's int8 activation, instead of re-decoding per token or decoding
the row to f32.

## Reference to follow

**Strata** (`~/src/Strata-ref/src/kernels/cpu/`):
- **i-quants, several tokens at once on AVX2** (`code`: `iq_avx2.cpp`,
  `expert_multi`, `row_dot_multi`, the IQ4_NL path at `:387`). Written for
  AMD Zen 2/3 and Intel parts without AVX-512, the dev host's CPU class:
  `pshufb` sign broadcast, `vpsignb`, `maddubs`, `madd`, `add` per token;
  formats IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ4_NL.
- **K-quants, several tokens, bit-exact against ggml-cpu** (`code`:
  `kq_avx2.cpp`).
- **The pool** (`code`: `include/strata/kernels/cpu/pool.hpp`).
- **Effect** (`paper` §6 Finding 7): decoding once for several tokens is
  2.0–2.4× faster; the i-quant tier is CPU-arithmetic-bound.

## Gate

On the served Flash-Next arm (B60, `d48q8`, ratio 75 + census128, host bank,
the 20,085-token needle prompt), against shape-routed 0074 in the same window:
prefill faster, decode within the run-to-run spread, and the answer-level bar
(`CLAUDE.md`). Device-free first: a cell against the f32 tier within a stated
scale-relative bound, deterministic across runs, red on a named mutant.

## Current state

- **Per layer call** (`measured-here`, the dev host's CPU, Flash-Next shapes,
  H 2560, I 640): decode-shaped, 8 experts, ~651 µs with 0074 (microbench);
  prefill-shaped, chunk 512, 322 experts, ~152 ms in the served tier, where
  llama.cpp's `mul_mat_id` takes 116.5 ms on the same shape (microbench, 8
  physical cores, cold DRAM).
- **Served** (`measured-here`, B60, 20,085 tokens): prefill 63.1 t/s, decode
  6.5 t/s with shape-routed 0074; window-0 KL +0.0183 nats against the
  pre-0074 tier.

## Where it lives

Plugin tree: `moe_cpu_expert_avx2.cpp` (`decode_*_row_avx2`,
`native_dot_jobs_avx2`, `compute_stage_f32_jobs`, the 0074 quantised dot) and
the unit cells `moe_cpu_expert_native.*`.
Prior art: `research-sub4bit-weights.md`, `research-hybrid-expert-execution.md`.

Full history: `git show b0447b8:docs/campaigns/kquant-host-storage.md`.
