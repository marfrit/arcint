# hybrid-expert-fetch — the GPU takes part of each decode step's host experts over PCIe, so the CPU tier and the link finish together

Charter: 0.5.x performance, the lever after `host-expert-bank`. **Proposed;
an operator decision is owed on the §3.4 question below before any code.**

## The defect, as measured

This is a lever, not a defect. With every host expert in RAM (`host-expert-bank`,
patch 0072), a Flash-Next decode token on the B60 costs ~118 ms, 2.46 ms per
layer (`measured-here`: d48s2, ratio 78 + tier + dispatch, 12e9 device slot
pool, census112 seed, the same prompt served twice, decode-only counters by
a one- vs two-request difference):

| part, per layer | time | evidence |
|---|---|---|
| CPU tier compute | 1.14 ms (7.57 experts; the tier bench's own figure) | `measured-here` |
| x readback + y writeback | 0.13 + 0.08 ms | `measured-here` |
| GPU device time | ~0.69 ms (33.3 ms per token; dense `gemm_kernel` 23.7 of it) | `measured-here`, CLIntercept, 129- minus 1-token process |
| the rest (launch, sync) | ~0.4 ms | by difference |

The GPU is busy under 30 % of the wall. While the tier computes, it has only
the layer's ~2.5 resident experts to run. The tier's 8 or 15 workers make no
difference: 39.45 vs 38.90 s (`measured-here`).

## Known against hypothesised

**Known:**
- **The link rate.** The B60 copies a 2.46 MB expert from pageable anonymous
  memory (the bank) at 11.2–13.2 GB/s, 186–219 µs each. Pinned memory
  reaches 13.9 GB/s (`measured-here`, an OpenCL microbench, 16–64 chunks).
- **The reference does this.** FreeToken's hybrid backend fetches a capped
  fraction of each step's misses over PCIe into the GPU slot cache, and the
  CPU computes the rest. The fraction is `pcie_bw / (pcie_bw + cpu_bw)`,
  measured with both running at once (`code`: `moe/bench_profile.py`
  `load_hybrid_fetch_fraction`, `moe/offload_cache.py` `hybrid_max_fetch` /
  `hybrid_fetch_fraction`). Its CPU pool is one pinned worker per physical
  core (`code`: `moe/cpu_executor.py` `physical_core_cpus`).

**Hypothesised** (arithmetic from the rows above, not measured):
- **The split.** The CPU tier's throughput is ~0.15 ms per expert and the
  link's ~0.19 ms. Sending ~3.4 of the 7.57 experts to the GPU would finish
  both in ~0.65 ms instead of 1.14, about −0.5 ms per layer, roughly −20 % per
  token. Two terms are missing from that arithmetic:
  - the fetched experts' own compute on the GPU's native per-expert kernel.
    The batched native kernels ran 26–40 µs per call in the d48q8 profile,
    but the per-expert figure is the first measurement owed;
  - the overlap itself. Today the GPU's work and the tier run serially within
    a layer, so the saving assumes the fetch and its compute overlap the
    tier. That is a scheduling change, and it is in scope.
- **Contention.** The DMA and the tier would read host DRAM at the same time
  (~14 + ~16 GB/s), which is unmeasured here.

## The §3.4 question (the operator's)

On this route the GPU's native per-expert kernel and the CPU tier are **not**
bit-identical (DESIGN §7.0.2cf, `measured-here`). So which device computes an
expert changes the bits of the answer.
- **FreeToken's LRU-shaped split** would make the answer depend on history
  (what the cache held), which §3.4 forbids.
- **A deterministic split keeps §3.4.** The GPU would take the first *k* of
  the step's misses in expert-id order, *k* fixed by configuration. The
  fetched experts are computed on the GPU whether or not a slot still holds
  their bytes; the cache saves only the transfer. The answer is then a pure
  function of the routing.
- **It is still a different answer** from today's CPU-only tier, the way the
  static partition's residency is. Its quality needs the same gate as any
  residency change.

## Gate (proposed)

On the configuration above, the first 257-token answer of a fresh process
decodes at least 10 % faster than the bank-only arm. Two cold processes give
byte-identical text (history independence), on the A770, whose forwards are
bit-reproducible. The B60 differs at f16-ulp every forward, so there the
comparand is the text, never a logits digest. A quality row against the
reference is owed at the next measurement window.

## Scope — in / out

In: a per-layer transient device slot region, H2D copies from the bank of
the chosen misses, their compute on the GPU's native per-expert kernels
overlapped with the CPU tier, and *k* as a knob. Out: an LRU policy; pinning the bank (pageable already reaches
~13 GB/s).

## Status

- 2026-09-28. Proposed. Numbers above measured; no code.
- 2026-09-28. **Operator decision:** the deterministic split. The LRU form's
  numeric difference is small (the native per-expert kernels sit in the CPU
  oracle band since patches 0056/0069), but it would make answers depend on
  earlier requests.
- 2026-09-28. **Built (patch 0073, uncommitted) and measured: history
  independence holds, the speed does not.**
  - **Code:** `moe/hybrid_split.hpp` holds the choice: first k of a decode
    step's host-tier experts by the bank seed's rank. Four cells, red on four
    mutants (an LRU-shaped choice among them). K transient slots follow the
    pinned ones, sized by ONE rule in `moe_pool_slots`: the first build sized
    the slot buffers from the ratio alone, and the first copy into a
    transient slot took a copy-engine page fault (`-ENOENT`) and an engine
    reset on the B60 (twice; the card recovered). The copies come from the
    host bank after the CPU tier is dispatched.
  - **B60 window** (`measured-here`; d48q8, ratio 75 + 15.4–15.8e9 pool,
    census128, bank 46 GiB, the same prompt twice in one process):

    | arm | first | second | text |
    |---|---|---|---|
    | K=0 (0073 off) | 35.76 s | 28.00 s | `e102dc17`, = 0072 |
    | K=3, staged copies | 56.60 s | 44.22 s | `19a9b934`, both identical |
    | K=3, weights copied straight from the bank | 51.62 s | 42.88 s | `19a9b934`, both identical |

  - **Why it is slower** (`measured-here`, the plugin counters): 73,218
    experts went to the card and 49,006 of them
    needed a copy (the slots hold only K). Between CPU-tier dispatch and the
    join, the main thread spends ~2.7 ms per layer call against 0.09 ms, and
    the CPU tier's per-expert time rises (memory traffic). A non-blocking
    copy from pageable memory costs the same host time as the explicit
    staging, so the runtime stages it itself.
  - **What would change it:** a pinned (usm_host) source, so the enqueue is
    instant and the DMA overlaps. For example, a small static host-pinned set
    per layer (the next ranks after the resident ones), with the choice
    restricted to it: still a pure function of the routing.
- 2026-09-28. **The pinned candidate set, built and measured: no gain.**
  - **Code:** each layer's next 16 experts by rank after the pinned ones
    (`MOE_HYBRID_CANDIDATES`) sit in the slot layout in one usm_host buffer,
    and a copy is one DMA from it. Total ~1.9 GB pinned, under TTM's cap on
    the dev host of 8,220,668 pages = 31.4 GiB (`measured-here`, the
    `pages_limit` parameter).
  - **B60** (`measured-here`; the configuration above except the bank, cut
    from 46 to 44 GiB for the pinned images' host memory; both arms at 44):

    | arm | first | second | text |
    |---|---|---|---|
    | K=0 | 36.39 s | 28.06 s | `e102dc17` |
    | K=3, 16 pinned candidates | 43.72 s | 29.04 s | `1aab87ee`, both identical |

  - **Host cost fixed:** the main-thread gap between CPU-tier dispatch and
    the join falls from ~2.7 ms to 0.31 ms per layer call (control 0.09 ms),
    so staging was that cost.
  - **Too few experts moved:** 16 candidates catch only ~0.4 of a layer's
    ~7.6 host-tier experts (10,044 moved over 514 tokens). Building the
    images costs the first answer ~7 s (1.9 GB read once).
  - **What it would take to pay:** covering most misses means pinning most
    of the host tier. That is the bank itself as usm_host, capped at 31.4 GiB
    by TTM against the 44–46 GiB bank. Even then the ceiling is bounded: the
    link moves an expert in ~0.19 ms and the tier computes one in ~0.15 ms,
    so a perfect overlap takes at most ~45 % of the tier's ~1.14 ms per layer.
    **Verdict for now: not worth it on this host;** patch 0073 stays
    uncommitted, the numbers are on the record here.
