# window-052 — 0.5.2 VENICE-001 acceptance

The expert hot set for Qwen3.8 Flash-Next. Closed with 0.5.4 (2026-10-01)
against the readable rows, under the operator's 2026-10-01 re-gating of the
speed row to the served native decode against the recorded `d48n` host-bound
baseline. `G = 1.10`. Markers as in `docs/window-050.md`; `EMPTY` = no
readable measurement.

| row | value / status | marker | evidence class | on record |
|---|---|---|---|---|
| speed | warm decode ≥ G × the host-bound baseline: **PASS.** Served `d48n` hybrid, A770 `GPU.1`, GT clock pinned 2000 MHz, patch 0068 against `+p20`, `--offload-ratio 75 --moe-cpu-tier --moe-per-expert-dispatch`, u8 KV, `--n-ctx 8192`, chunk 512, one lane; four arms, 32 greedy tokens: 0068 d1 **0.9 / 1.2 t/s**, d512 **1.2 / 1.3 t/s**; base d1 0.5 / 0.6, d512 0.6 / 0.6. Baseline = the recorded `d48n` host tier 0.5–0.8 t/s; threshold 1.10 × 0.8 = **0.88 t/s** (A770 same-day 1.10 × 0.526 = 0.579 t/s also passes). Digests byte-identical across the four arms: d1 `1ddebc829f218598`, d512 `e4b40e198c8f22a6` | `measured-here` | `measured-here` | DESIGN §7.0.2cx; CHANGELOG 0.5.4 (commit `0e0ef26`) |
| speed, the residency-policy arm | ratio-99 census seed **0.556 t/s** (< 0.579); ratio-75 census top-128 **0.842 t/s** against the same-config host control **0.465 t/s** = **1.81×** | `measured-here` | `measured-here` | `docs/campaigns/sub4bit-vram-kernel.md`, status 2026-09-22 |
| stale-byte zero proof | **owed**: needs an engine-side host/card readback of expert bytes | `EMPTY` | `code` | `docs/campaigns/expert-hot-set-lru.md` |
| convergence | `rounds_to_plateau = None`, `plateau = False` at S = 6 and S = 10, at 512 and 4,096 decode tokens | `measured-here` | `measured-here` | `docs/campaigns/expert-hot-set-lru.md` |
| quality under policy | non-dispatch path: byte-identical digest `2169836b…336f`; dispatch route: the answer depends on the resident seed (branches at greedy token 3 of 64) | `measured-here` | `measured-here` | DESIGN §7.0.2cf |

**Verdict.** The speed row reads PASS on the served route; the stale-byte row
is owed.

Full history: `git show b0447b8:docs/window-052.md`.
