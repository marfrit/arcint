# dense-q8-flash-next — Flash-Next's dense projections served in the checkpoint's Q8_0 form

**Closed 2026-09-28.** `tools/q4e/dense_q8.py` and the exporter's `--dense-q8`
(Q8_0 first, then `--dense-u8` over the rest) produce `d48q8`, the served
Flash-Next artifact: 328 Q8_0 projections 6.17 → 3.28 GiB, lm `.bin` 72.17 →
60.70 GiB, dense gemm 23.7 → 17.2 ms a decode token on the B60, and the freed
VRAM carries 128 resident experts a layer instead of 112 (`measured-here`). The
recovery is bit-identical to the plain graph on CPU
(`tests/python/test_dense_q8.py`, red on a 1.001 scale mutant). The served
KL/argmax row reads against the f32 reference below 2,051 now and above it
after the re-capture (qsa T8).

Full history: `git show b0447b8:docs/campaigns/dense-q8-flash-next.md`.
