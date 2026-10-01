# prefill-expert-streaming — Flash-Next prefill computes every expert on the card, streaming the host tier's experts per layer

Charter: 0.5.4 LYON ("the long road": serving-length prefill), with the
performance directive of 2026-09-27. This is the lever the measurements put
first for Flash-Next's prefill. LYON's L1 (a fused chunked GDN primitive) is
not the binding term on this model; see "Known" below.

## The defect, as measured

Flash-Next `d48q8` on the B60 (ratio 75 + tier + per-expert dispatch, 15.4e9
device pool, census128, host bank 46 GiB, chunk 512), a 4,096-token prompt
from the bench ids (`measured-here`):

- **Rate.** Prefill decodes 4,096 tokens in 83.7–94.9 s (43–49 t/s).
- **Device time.** 2.72 ms per prompt token (CLIntercept, a 4,096- minus a
  16-token process), 11 s of the wall:
  - the grouped native MoE kernels: 1.98 ms;
  - H2D copies: 0.54 ms;
  - the GDN core: 0.077 ms (**0.3 % of the wall**).
- **The CPU tier.** Counters by the same difference, per MoE layer call (336
  over the prefill): 322 host experts and 3,227 (token, expert) pairs.
  - The tier takes ~152 ms (join wait 152 ms) and the x readback plus y
    writeback another ~28 ms.
  - Over the prefill that is ~51 s of CPU tier and ~9 s of moves, of the
    ~80 s the 4,096-token run adds over the 16-token one (83.7 − 3.3 s, the
    counter runs).

The CPU tier's cost grows with the pairs, so with the prompt. The card's
share grows with the bytes it reads.

## Known against hypothesised

**Known:**
- **The reference streams whole layers** (`code`): FreeToken's prefill copies
  a layer's whole expert set into device slots, position = expert id
  (`moe/offload_cache.py` `copy_missing`, the "whole-layer ... prefill
  materialize"). With `prefill_overlap` it double-buffers across layers:
  `cache_size >= 2 * num_experts`, the next layer copied while this one
  computes. Every expert is computed on the GPU.
- **The link.** The B60 copies 2.46 MB chunks at 11.2–13.2 GB/s from
  pageable memory and 13.9 GB/s from pinned memory (`measured-here`,
  microbench). But a copy from pageable memory costs the calling thread the
  runtime's own staging, ~1.3 ms per expert under the CPU tier's load
  (`measured-here`, the hybrid-expert-fetch legs). So the source must be
  pinned, or the staging taken off the critical thread.
- **The pinned cap.** TTM caps pinned memory for the card at 31.4 GiB on the
  dev host (`measured-here`, `pages_limit`); the bank is 44–46 GiB.
- **L1 is not binding.** The GDN core is 0.3 % of Flash-Next's prefill wall
  (above) and was 1.5 % of the 35B's prefill device time
  (`design-lyon-stateful-prefill.md` §4h). A chunked GDN primitive moves
  neither.

**Hypothesised** (arithmetic from the rows above):
- **Compute.** All experts on the card cost ~8–9 ms of device time a prompt
  token, the grouped kernels scaled by 4 for the 75 % now on the host.
- **Copy.** ~0.8–1 GB per layer per chunk: ~6.6 ms a token at chunk 512,
  ~1.6 ms at chunk 2048.
- **Result.** Against today's ~20 ms a token of CPU tier, prefill rises to
  ~1.4x (chunk 512) and ~2.2x (chunk 2048). Measure before claiming either.

## Gate

The 4,096-token prompt above prefills at least 1.5x faster than today's arm
on the same configuration, at the chunk that serves best. The answer is
checked by the KLD harness (window 0) against the incumbent's row,
since the card and the CPU tier are not bit-identical. Streaming is decided
by the call's shape, not by history, so two cold runs must give identical
text on the A770.

## Scope — in / out

In:
- a per-layer streaming region on the card, double-buffered;
- the host-tier experts of a prefill call copied into it from pinned staging,
  filled by the idle CPU-tier threads from the host bank;
- the grouped native kernels run over resident and streamed experts;
- a knob that turns streaming on above a token count.

Out: decode (unchanged); an LRU across calls; raising the TTM cap.

## Entry criteria

Met: the defect measured (above), the reference's code read, the link rate
and the pinned cap measured. Owed before code: the recon of the grouped
prefill path (how streamed experts enter the grouped kernels' tables).

## Where it lives

The plugin's prefill route in `moe_3gemm_swiglu_opt.cpp` (the grouped native
kernels and the hybrid prefill path of patch 0037), the slot buffers of
`ops/moe_offload_constant.cpp`, the host bank of patch 0072
(`moe/host_expert_bank.hpp`) as the byte source. Knobs and counters are named
with the design note.

## Pipeline for this campaign

Recon of the grouped prefill path → design note → red-first cells (slot
mapping, the staging schedule) → implementation → one B60 window →
review → DESIGN record and CHANGELOG.

## Invariants

DESIGN §3.4: which device computes an expert depends on the call's shape (a
prefill chunk above the threshold), never on history. Decode is untouched.

## Status

- 2026-09-28. Opened with the measurements above; no code.
- 2026-09-28. **v1 built (uncommitted) and measured: a loss as built; the
  ceiling measured.**
  - **v1:** the call's CPU-tier experts are planned into a streaming region
    (`moe/prefill_stream.hpp`, two cells red on three mutants). The pool's
    threads fill a usm_host staging copy in the slot layout from the host
    bank, one copy per tensor moves it, and the grouped native stages run a
    second time on the region. One region per expert geometry: Flash-Next has
    three, 3.64 GiB of device memory in the load ledger and as much usm_host.
  - **B60, d48q8, 4,096 tokens** (`measured-here`; bank 40 GiB, 320 experts
    a layer, to leave host memory for the staging):

    | arm | prefill |
    |---|---|
    | off, chunk 512 | 89.70 s |
    | on, chunk 512 | 160.05 s (fill 204 ms a call) |
    | on, chunk 2048 | the process crashed (a user-space segfault; not investigated) |
    | off, chunk 2048 | 92.50 s (CPU tier 324 ms a call for 4x the tokens) |

  - **Why v1 loses:** CPU writes into usm_host run at 18–20 GB/s, the same
    as into anonymous memory (`measured-here`, microbench, 1–16 threads).
    The fill is slow because the bank held 320 of a layer's ~384 host
    experts, and a prefill chunk touches all of them, so every call faults
    ~64 experts in from the disk. The pinned staging is what squeezed the
    bank.
  - **The ceiling** (`measured-here`, CLIntercept with streaming on, a
    4,096- minus 16-token difference): 9.36 ms of device time a prompt token.
    Host-to-device copies take 4.88 ms (~59 ms a call, ~7.3 GB/s) and the
    grouped kernels computing every expert 3.64 ms (~44 ms a call).
    - With the host fill hidden and the copies overlapping compute on a
      second queue: ~59 ms a call against the CPU tier's ~138 ms (the off
      arm's join wait at bank 40 GiB, a process average, `measured-here`;
      152 ms at 46 GiB above was a decode-difference figure), about 2.3x at
      chunk 512.
    - Serialised on one queue: ~1.3x.
    - At chunk 2048 the CPU tier amortises its weight reads (324 ms a call),
      so streaming nears break-even.
  - **What v2 needs:** a batch-sized pinned ring, so the bank holds the
    whole host tier again; the fill pipelined with the copies; a second
    queue with cross-queue events (FreeToken's copy stream with ready and
    release events, `code`); and the chunk-2048 crash explained. **Parked**:
    Qwen Sparse Attention (LYON L2, required by the operator) goes first;
    the v1 diff is kept outside the repository.
