# window-051 — 0.5.1 BERLIN-001 acceptance

Qwen3.8 Flash-Next served at depth 48 on one card. Closed with 0.5.4
(2026-10-01) against the readable rows, under the operator's 2026-10-01
re-gating; B6 and B7 are owed. Markers as in `docs/window-050.md`; `EMPTY` =
no readable measurement.

| # | row | value / status | marker | evidence class | on record |
|---|---|---|---|---|---|
| B1 | depth 48 served on the B60 (`d48q8`) | first answer **36.56 s** (second prompt 33.78 s), B60, ratio 78 + tier + dispatch, 12e9 device pool, census112 seed, host bank 46 GiB, 257 greedy tokens; ratio 75 + census128 with the 15.4e9 pool reads **35.83 s** (7.2 t/s) / 32.11 s (8.0 t/s) | `measured-here` | `measured-here` | `docs/campaigns/dense-q8-flash-next.md` "B60 window" (commit `e8d4110`) |
| B2 | bound bytes per tier, from the load ledger | card tier: device weights+graph **15.82 GiB** (ratio 78) / **18.66 GiB** (ratio 75, 15.4e9 pool); host tier: CPU expert bank **30 GiB** (0074 gate, ratio 75 + census128) and 46 GiB at the 0072 gate | `measured-here` | `measured-here` | `docs/campaigns/dense-q8-flash-next.md` "B60 window"; `docs/campaigns/host-expert-bank.md` |
| B3 | Paris, answered at depth 48 | ` Paris. Paris is a city in France` (cold and warm byte-identical), B60, the corrected-fill full-depth artifact, ratio 99 + tier, KV u8, f16, one lane, 2026-09-18 | `measured-here` | `measured-here` | `docs/campaigns/sub4bit-vram-kernel.md` 2026-09-18 entry |
| B4 | the 20k needle, answered | `ORANGE-FALCON-77` answered by dense `d48q8`, B60, ratio 75 + census128, bank 30 GiB, u8 KV, chunk 2048, a 20,085-token prompt, three fresh processes | `measured-here` | `measured-here` | `docs/campaigns/host-expert-bank.md` 2026-10-01 "0074 served gate" |
| B5 | KL below 2,051 against the f32 reference, window 0 (report-only) | base `MOE_CPU_TIER_Q8_DOT=0` **0.3003**; shape-routed 0074 **0.3186** (delta **+0.0183**); argmax 0.7974 / 0.7981 | `measured-here` | `measured-here` | `docs/campaigns/host-expert-bank.md` 2026-10-01 (commit `940d69b`) |
| B6 | the original bar row (`bar_0.5.1` = 3.0905e-03 nats below 2,051) | **owed**: the bar sits below the B60's run-to-run floor of 0.1361 / 0.1512 nats, so the row is unreadable on that card | `EMPTY` | `measured-here` (floor) / `code` (bar) | `docs/campaigns/served-prefill-determinism.md`; DESIGN §7.0.2cb |
| B7 | a Paris cell on the dense `d48q8` export specifically | **owed**: no Paris cell on the dense `d48q8` artifact is on record (B3 ran on the earlier full-depth artifact; `d48q8qsa` answers Paris, `docs/campaigns/qsa.md` T5) | `EMPTY` | `measured-here` | — |

**Verdict.** PASS on B1–B5; B6 and B7 owed.

Full history: `git show b0447b8:docs/window-051.md`.
