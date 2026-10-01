# window-053 — 0.5.3 LISBON acceptance

The host expert bank and NVMe-staged expert serving for Qwen3.8 Flash-Next.
Closed with 0.5.4 (2026-10-01) against the readable rows, under the operator's
2026-10-01 re-gating; L-cold and the arcwell-arm determinism sub-row are owed.
Markers as in `docs/window-050.md`; `EMPTY` = no readable measurement.
Threshold `X = 139.5 s` is arithmetic (`code`: `T_boot` 136 s + `T_fill`
0.387 s + `T_prefill` 3.08 s).

| row | value / status | marker | evidence class | on record |
|---|---|---|---|---|
| L-host-bank | the host bank's served gate: **PASS.** B60, one fresh process per arm, 257 greedy tokens each: mapping arm **48.55 s**, bank 46 GiB **38.18 s**; greedy text byte-identical across arms (−18 to −21 %). At 46 GiB: ~16,000 experts filled, 2,161–2,506 demand reads, 0 evictions; the fill read 59.8 GiB sequentially and added ~21 s to the load | `measured-here` | `measured-here` | `docs/campaigns/host-expert-bank.md` 2026-09-28 window table |
| L-nvme, row 1: cold TTFT, depth 4 | **PASS.** One B60 two-arm window, ratio 86, plugin `ov-0049`: arcwell arm **92.492 s** ≤ host-fed **99.679 s** ≤ `X`; prefetch depth **4 batches in flight**; `AW_IOC_STATS` arcwell delta `bytes +697,958,400`, `reads +852`, `segments +871`, `batches +4`, `via_host_bounce 0→0`, `max_inflight 220`; host-fed `bytes +0`; decode arcwell 4.1 t/s vs host-fed 3.4 t/s | `measured-here` | `measured-here` | `docs/campaigns/nvme-direct-expert-tier.md` 2026-09-24 |
| L-nvme, row 2: RSS through boot | **PASS.** Boot-child `wait4` `ru_maxrss` **3.697 GiB** both arms (≤ 32 GiB); mid-run VmHWM 3.697 GiB ≤ `wait4`; physical `MemAvailable` minimum 45.86 GiB, 0 watchdog trips; PLE term staged at 2.884 MiB | `measured-here` | `measured-here` | same |
| L-nvme, row 3: restart determinism | **PASS on the A770 host-fed arm**: two cold boots byte-identical, `9a7e2e77cfa1a25a0ebdb653a54abb343987f977558e3bfd98a9752353e5969f`. The arcwell arm's sub-row is **owed**: arcwell runs on the B60 only | `measured-here` | `measured-here` | same |
| L-cold | cold start with nothing prebound: **owed**. The served cell prebinds the 71 slots/layer at load; no served number with nothing prebound exists | `EMPTY` | `code` | `docs/campaigns/nvme-direct-expert-tier.md` |

**Verdict.** PASS on L-host-bank and L-nvme rows 1–3 (A770 arm for row 3);
the arcwell-arm determinism sub-row and L-cold owed.

Full history: `git show b0447b8:docs/window-053.md`.
