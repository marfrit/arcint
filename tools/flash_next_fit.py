#!/usr/bin/env python3
"""Flash-Next (qwen4_exp) streaming-fit projection for the dev target.

This is the arithmetic half of the fit study demanded by the standing
directive: given a MEASURED tier hierarchy (VRAM, DRAM feed, NVMe miss feed)
and a MEASURED LRU hit-rate + MTP amortization from a routing trace, project
decode throughput for an on-demand (streaming) expert plan and report the
smallest working configuration.

It is a PROJECTION instrument, labelled as such: every input is a measured
number (bandwidths from dd/stream probes, hit-rate + reuse + amortization from
the routing trace), but the t/s it prints is an analytic estimate of the
bandwidth-bound decode rate, not an end-to-end served measurement. The repo's
rule stands -- a served t/s claim needs the served endpoint; this tool sizes
the envelope and says which knob is worth what, so the build is aimed before a
card-window is spent.

Model (per decoded token):
  - MoE: top-K of E experts on each of L layers. One expert-layer int4 slice is
    SLICE bytes. Full-miss expert traffic = K*L*SLICE.
  - A fraction h of those slices is LRU-resident across VRAM+DRAM (h comes from
    the trace at the resident capacity). Resident slices are served at the DRAM
    feed (the DRAM->VRAM move); missed slices stream from NVMe.
  - MTP/PLE amortization A = output tokens per expert-load window (a verify of
    a accepted drafts loads each needed expert once for all of them). A divides
    the *miss* traffic, since resident hits are already cheap.
  - time/token = [ h*traffic/DRAM_bw + (1-h)*traffic/(NVMe_bw*A) ] + compute_floor.
  - The PLE table must be DRAM-resident (per-token random hashed gather is
    seek-bound on any paged tier); table_gib is subtracted from DRAM first.

Residency is reported with its assumption named (directive rule #1): a bare
"fits"/"does not fit" is never emitted without "resident"/"streaming" attached.
"""
import argparse
import json
import sys

GIB = 2 ** 30

# Measured Flash-Next geometry (WP6, off the real GGUF).
SLICE_BYTES = 2_457_600      # one expert-layer int4 slice (gate/up/down fused)
K_ACTIVE = 10                # routed experts per token per layer
L_LAYERS = 48
E_EXPERTS = 512
EXPERT_POOL_GIB = E_EXPERTS * L_LAYERS * SLICE_BYTES / GIB   # 56.25
TABLE_GIB = 28_800_138_240 / GIB                            # 26.82 (IQ4_NL)
BACKBONE_GIB = 2.30          # attn+GDN+emb+lm_head+shared+norms, int4
TRAFFIC_GIB = K_ACTIVE * L_LAYERS * SLICE_BYTES / GIB        # 1.0986 per token


def resident_expert_gib(vram_gib, dram_gib, backbone_vram_gib, kv_gib,
                        activation_gib, table_gib=TABLE_GIB, dram_overhead_gib=1.0):
    """GiB of expert pool that can be kept LRU-resident across VRAM+DRAM, after
    the table (DRAM), backbone+KV+activations (VRAM), and overheads are paid."""
    vram_for_experts = max(0.0, vram_gib - backbone_vram_gib - kv_gib - activation_gib)
    dram_for_experts = max(0.0, dram_gib - table_gib - dram_overhead_gib)
    return min(EXPERT_POOL_GIB, vram_for_experts + dram_for_experts), \
        vram_for_experts, dram_for_experts


def project_tps(hit_rate, dram_bw_gibs, nvme_bw_gibs, amortization=1.0,
                compute_floor_ms=0.0, traffic_gib=TRAFFIC_GIB):
    """Projected decode t/s under the streaming model above. hit_rate in [0,1]."""
    h = max(0.0, min(1.0, hit_rate))
    hit_ms = (h * traffic_gib / dram_bw_gibs) * 1000.0
    miss_ms = ((1.0 - h) * traffic_gib / (nvme_bw_gibs * max(amortization, 1e-9))) * 1000.0
    ms = hit_ms + miss_ms + compute_floor_ms
    return 1000.0 / ms if ms > 0 else float("inf")


def table_residency_note(dram_gib, table_gib=TABLE_GIB):
    if table_gib <= dram_gib:
        return f"PLE table {table_gib:.2f} GiB fits DRAM-resident ({dram_gib} GiB); " \
               f"{dram_gib - table_gib:.1f} GiB DRAM left for an expert LRU"
    return f"PLE table {table_gib:.2f} GiB does NOT fit DRAM-resident ({dram_gib} GiB) " \
           f"-- the random hashed gather would be seek-bound; needs more DRAM or a " \
           f"table-sharding scheme"


def self_test():
    """Sanity-check the projection model against the regime ceilings, so a
    mis-edited formula is caught before it prints a plausible-looking t/s."""
    fails = []
    # Full resident (h=1) -> DRAM-bound ceiling.
    full_res = project_tps(1.0, 44.4, 4.66, amortization=1.0)
    expect_dram = 1000.0 / ((TRAFFIC_GIB / 44.4) * 1000.0)
    if abs(full_res - expect_dram) > 0.1:
        fails.append(f"h=1 should hit DRAM ceiling {expect_dram:.1f}, got {full_res:.1f}")
    # Full miss (h=0), no amortization -> NVMe ceiling.
    full_miss = project_tps(0.0, 44.4, 4.66, amortization=1.0)
    expect_nvme = 1000.0 / ((TRAFFIC_GIB / 4.66) * 1000.0)
    if abs(full_miss - expect_nvme) > 0.1:
        fails.append(f"h=0 should hit NVMe ceiling {expect_nvme:.1f}, got {full_miss:.1f}")
    # Amortization must raise full-miss throughput monotonically.
    if not project_tps(0.0, 44.4, 4.66, amortization=3.0) > full_miss:
        fails.append("amortization did not raise the miss-bound throughput")
    # HDD floor must be far below NVMe (the tier finding).
    hdd = project_tps(0.0, 44.4, 0.413, amortization=1.0)
    if not hdd < full_miss / 5:
        fails.append("HDD floor not far below NVMe -- tier model wrong")
    # Residency accounting: table must be subtracted from DRAM first.
    res, vfe, dfe = resident_expert_gib(15, 44, 2.3, 3.0, 2.0)
    if dfe > 44 - TABLE_GIB:
        fails.append("DRAM expert budget ignored the table reservation")
    print(f"  self-test DRAM ceiling (h=1):  {full_res:.1f} t/s")
    print(f"  self-test NVMe ceiling (h=0):  {full_miss:.1f} t/s")
    print(f"  self-test HDD floor (h=0):     {hdd:.2f} t/s")
    print(f"  self-test resident experts @ (15 VRAM,44 DRAM): {res:.1f} GiB "
          f"(VRAM {vfe:.1f} + DRAM {dfe:.1f})")
    if fails:
        print("SELF-TEST FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("SELF-TEST PASSED: regime ceilings and residency accounting hold.")
    return 0


def run(args):
    res, vfe, dfe = resident_expert_gib(
        args.vram, args.dram, args.backbone_vram, args.kv, args.activation)
    res_frac = res / EXPERT_POOL_GIB
    print(f"target: {args.vram} GiB VRAM + {args.dram} GiB DRAM; "
          f"NVMe {args.nvme_bw} GiB/s, DRAM {args.dram_bw} GiB/s; MTP amort {args.amort}")
    print(table_residency_note(args.dram))
    print(f"resident expert pool: {res:.1f} / {EXPERT_POOL_GIB:.2f} GiB "
          f"({res_frac*100:.0f}%) = VRAM {vfe:.1f} + DRAM {dfe:.1f}")
    print(f"per-token full-miss expert traffic: {TRAFFIC_GIB:.4f} GiB")
    print("projection (streaming plan; t/s is bandwidth-bound estimate, not served):")
    print(f"  {'hit-rate h':>10} | {'t/s (amort 1)':>13} | {'t/s (amort '+format(args.amort,'.1f')+')':>16}")
    hrs = args.hit_rates if args.hit_rates else [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    for h in hrs:
        t1 = project_tps(h, args.dram_bw, args.nvme_bw, 1.0, args.compute_floor_ms)
        ta = project_tps(h, args.dram_bw, args.nvme_bw, args.amort, args.compute_floor_ms)
        print(f"  {h:>10.2f} | {t1:>13.1f} | {ta:>16.1f}")
    print(json.dumps({
        "resident_expert_gib": round(res, 2),
        "resident_frac": round(res_frac, 3),
        "traffic_gib_per_token": round(TRAFFIC_GIB, 4),
        "expert_pool_gib": round(EXPERT_POOL_GIB, 2),
        "table_gib": round(TABLE_GIB, 2),
    }))
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--vram", type=float, default=15.0, help="usable VRAM GiB (1xA770 ~15)")
    ap.add_argument("--dram", type=float, default=44.0, help="usable DRAM GiB")
    ap.add_argument("--backbone-vram", type=float, default=2.3)
    ap.add_argument("--kv", type=float, default=3.0, help="KV pool GiB (pinned precision)")
    ap.add_argument("--activation", type=float, default=2.0)
    ap.add_argument("--dram-bw", type=float, default=44.4, help="DRAM read GiB/s (measured)")
    ap.add_argument("--nvme-bw", type=float, default=4.66, help="NVMe read GiB/s (measured)")
    ap.add_argument("--amort", type=float, default=1.0, help="MTP tokens-per-weight-load")
    ap.add_argument("--compute-floor-ms", type=float, default=0.0)
    ap.add_argument("--hit-rates", type=float, nargs="*",
                    help="explicit hit-rate points (from the trace's LRU replay)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
