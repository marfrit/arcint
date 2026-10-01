# window-054 — 0.5.4 LYON-001 acceptance

Serving-length prefill (context ≥ 32k, spanning the 2,051 sparse-attention
boundary) for Qwen3.8 Flash-Next. Closed with 0.5.4 (2026-10-01) against the
readable rows, under the operator's 2026-10-01 re-gating; row 2, row 3a and
the above-2,051 KLD are owed. Markers as in `docs/window-050.md`; `EMPTY` = no
readable measurement.

| row | value / status | marker | evidence class | on record |
|---|---|---|---|---|
| L1-new | the ~31k-token needle at `n_ctx 32,768` on dense `d48q8`: **PASS.** B60, plugin `55c432880f2d5ed0` (patch 0074), `qwen38-flash-next-d48q8-ov`, `--n-ctx 32768 --paged-kv u8 --mtp off --offload-ratio 75 --moe-cpu-tier --moe-per-expert-dispatch`, one lane, greedy. 27,603 prompt tokens; prefill **451.29 s (61.2 t/s)**; answer **`ORANGE-FALCON-77`** (80 tokens, EOS); decode **6.6 t/s**; load 170 s | `RUN@wt+cce3c946` | `measured-here` | 2026-10-01 LYON window |
| L-QSA | QSA served, non-default, `n_ctx ≤ 32,768`: **PASS.** Native `d48q8qsa` (`b21359a42c2c8633`, `.bin` 65,221,492,040 B, peak 51.05 GiB) loads and serves 12 compressed block-cache layers; the needle is answered at 20,085 tokens; B60: decode **0.87× dense** (5.1 vs 5.9 t/s), prefill **1.27×** (51.6 vs 65.6 t/s). Dense stays the served artifact. Limit: `n_ctx ≤ 32,768` (fixed `[8192, dh]` block cap) | `measured-here` | `measured-here` | `docs/campaigns/qsa.md` 2026-09-30 (commit `2f7f942`) |
| L-KLD-above | KLD above 2,051: **owed** until the reference is re-captured with the fixed BF16 feed (the old captures ran the sparse-attention indexer on raw bytes; rows below 2,051 stand) | `EMPTY` | `measured-here` / `code` | `docs/campaigns/qsa.md` T8; DESIGN §7.0.2bz, §7.0.2cy |
| row 1 | the 32k prompt answered on Flash-Next `d48n` hybrid, A770: **PASS** (32,768 tokens, 15.1 t/s, digest `d5942c7f…`) | `RUN@5a783b7` | `measured-here` | 2026-09-26 |
| row 2 | KLD gate, both regimes: **owed** (open numerics campaign) | `EMPTY` | — | `docs/campaigns/served-prefill-determinism.md` |
| row 3a | prefill graph node count invariant in T, one compile per process: **owed** (the property was read on the sequential core, not the chunked core) | `EMPTY` | `code` | `docs/design-lyon-stateful-prefill.md` |
| row 3b | `compile_s ≤ nodes × 2.38 ms`: **PASS as an upper bound on one run** (40.1 s against 45.2 s; two further loads of 70.9 and 45.6 s stay open) | `measured-here` | `measured-here` | 2026-09-27 |
| row 3c | prefill ≥ 460 t/s at 32k on the A770: **PASS on the Qwen3.6-35B-A3B by operator ruling** — 778.9 t/s at chunk 2048, 781.4 t/s at chunk 1024; `qwen3.6-35b-a3b-native-d40packed-u8`, all-resident, `--paged-kv u8`, `--n-ctx 36864`, plugin series 0003–0064 | `RUN@bdbb0aa` | `measured-here` | 2026-09-26 |

**Verdict.** PASS on L1-new, L-QSA, rows 1, 3b (one run) and 3c; L-KLD-above,
row 2 and row 3a owed.

Full history: `git show b0447b8:docs/window-054.md`.
