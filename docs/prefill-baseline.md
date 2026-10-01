# Prefill baseline, coder on the B60 (2026-08-29)

The first prefill record of the paged path (`qwen36-coder-b5-ov`, u8 KV, one
lane, `arcint 0.2.1`): chunk 2048 is the served optimum (1,878 t/s at 28,906
tokens; 4096 and 8192 are refused by the fit), and the depth curve peaks at
1,954.5 t/s near 15k tokens and falls to 473.8 t/s at 240,754 (L^1.52 overall,
`measured-here`). Attention prefill runs `sdpa_micro` with `dpas` in f16 and u8
(IGC shader dump, `measured-here`). `ARCINT_PROFILE_PAST=<n>` profiles a chunk
past position 0. Current served rates: `docs/benchmark-served-services.md`.

Full history: `git show b0447b8:docs/prefill-baseline.md`.
