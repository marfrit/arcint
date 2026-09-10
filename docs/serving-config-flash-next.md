# Flash-Next (qwen4exp) streaming serving config — single-A770 target

WP7 (0.5.0), dated 2026-09-10. One page: the resident-byte budget, the pinned
KV precision, the miss budget, and the projected decode rate with its regime
label, for serving `Qwen3.8-Flash-Next-UD-Q3_K_XL` off NVMe on one A770.

This sizes the **streaming (host-tier expert offload) plan**. It is a projection
fed by measured inputs, not a served measurement — the live expert gather that
reads NVMe-resident GGUF rows on a miss is **parked on the backbone-IR emission
(FIX A)**, the same gate FIX D Link 3's integration sits behind. Re-derive every
number here with the audit block at the foot.

## Card budget (measured target)

| Tier | Size | Role |
|------|------|------|
| VRAM (A770) | 15.0 GiB usable | backbone 2.3 + KV pool 3.0 + activations 2.0 + expert LRU (VRAM share) |
| DRAM | 44.0 GiB usable | **PLE table 26.82 GiB (must be resident)** + 1.0 overhead + expert LRU (DRAM share) |
| NVMe (ZFS pool) | miss tier | 1.68 GiB/s in-container (measured, WP6b) — the real miss feed |

Resident expert pool after the table+backbone+KV+activation reservations:
**≈ 23.9 GiB of the 56.25 GiB full pool (≈ 42 %)**, ≈ 217 slots/layer
(VRAM 7.7 + DRAM 16.2 GiB). The **PLE table is DRAM-resident by requirement**:
its per-token random hashed gather (FIX D) is seek-bound on any paged tier, so a
config that cannot hold it in DRAM is **refused**, not degraded
(`flash_next_offload_must_refuse`).

## KV precision — PIN IT EXPLICITLY

Set `--paged-kv u8` (or an explicitly **KLD-validated** `u8:i4`). The KV
geometry is GQA 24 query heads / 2 KV heads (from the GGUF metadata), so the KV
pool is small relative to the table, and u8 is the safe default here.

Do **not** rely on the plugin's u4 KV auto-drop (`execution_config.cpp:345`) to
choose the precision silently: **its default may not be used until it has its
own KLD cell** (the `tools/kld_harness.py` instrument, WP5, measures the
KLD-vs-BF16 acceptance a 4-bit KV default would have to clear). Until that cell
exists, an explicit `--paged-kv u8` is the pinned precision this budget assumes;
the 3.0 GiB KV reservation above is a u8 figure.

## Miss budget

- Per-token full-miss expert traffic: **1.0986 GiB** (10 routed × 48 layers ×
  2,457,600 B int4 slice).
- At the provisioned **per-layer** LRU hit-rate ~94.4 % (24 GiB-class resident
  pool): miss traffic ≈ 0.062 GiB/token → **≈ 37 ms/token on the 1.68 GiB/s
  NVMe feed**, which dominates the ≈ 23 ms/token DRAM-resident-hit term. The
  plan is therefore **NVMe-miss-bound**: cutting the miss stream (residency, or
  amortization) is the only lever that moves it.

## Cache model — per-layer, not global

The resident pool is a **per-layer LRU** (each of the 48 MoE layers holds its
own slots), matching arcint's own slot pool (`fit.h expert_slot_bytes` is
per-layer). The hit-rate the projection consumes MUST come from a per-layer
replay. A **global** shared LRU (FreeToken's paper shape) reads a materially
more optimistic ~93.8 % (flat) on the same trace vs the per-layer 88.1 % at a
16 GiB budget — a > 50 % t/s overstatement if adopted. WP6b's hit-rate table is
the per-layer model; `tools/expert_lru_replay.py --check` reproduces it within
~1.4 points on the sha-pinned trace. **The global figure is the FreeToken
comparison point, recorded, not adopted.**

## Projected decode rate (bandwidth-bound estimate, not served)

| per-layer hit | resident | t/s | regime |
|---------------|----------|-----|--------|
| 88.1 % (16 GiB) | ~16 GiB | ~10.0 | NVMe-miss-bound |
| 94.4 % (24 GiB) | ~24 GiB | ~16.7 | NVMe-miss-bound |
| 95 % (rounded) | ~24 GiB | ~17.8 | NVMe-miss-bound |
| 98 % (40 GiB) | ~40 GiB | ~27 | NVMe-miss-bound |
| 100 % (all resident) | — | ~40.4 | DRAM-resident-bound (ceiling) |

**The MTP amortization lever is unavailable as the model ships.** The
`UD-Q3_K_XL` GGUF carries **no MTP head** (block index range 0..47 = the 48
backbone layers, no `nextn`/`mtp`/`eh_proj` tensor, no `mtp.*` KV — verified
across all three shards, WP8). So amortization is **1×**: the 30–40 t/s figures
that earlier notes attached to a re-exported MTP head are **not reachable from
this artifact** and require the trained MTP head (upstream HF checkpoint, not on
the fleet — see WP8).

## Dry-run

    arcint --flash-next-offload-plan <measured-per-layer-hit>   # e.g. 0.944

prints this plan for the target budget and exits ADMIT (table fits) / REFUSE
(table cannot be DRAM-resident). The projected t/s is advisory; only the table
condition gates the exit code.

## Audit (re-runnable)

    # reproduce the WP6b hit-rate table (per-layer model) from the sha-pinned trace
    python3 tools/expert_lru_replay.py --check <full-trace>   # cbc4fe8c...; PASS
    # offline replay↔policy binding (committed 400-token fixture)
    python3 tools/test_expert_lru_replay.py                   # 6 tests OK
    # C++ policy (device-free) + config flag
    ./build/arcint-test flash_next                            # 11 cases, 0 failed
    ./build/arcint --flash-next-offload-plan 0.944            # 16.7 t/s, ADMIT
