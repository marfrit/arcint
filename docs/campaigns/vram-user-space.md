# vram-user-space — every byte of the card in use, no eviction cliff

**Status: open (operator, 2026-10-30).** All units; Flash-Next on the B60 first, where the prefill slot loan runs
the card to its edge.

## Why

xe does not fail an allocation that does not fit: TTM evicts buffers to GTT (host memory over PCIe), and they can
stay there. On this card that is a cliff, not a slope: a Flash-Next unit with a thin post-load margin had ~15 GiB of
its buffers evicted and produced 24 tokens in 49 s with the B60 at 99 % busy (`measured-here`, 2026-10-11, the unit
history in the operator notes; DESIGN's memory sections). The operator's rule (2026-10-30): **every byte of the card
should be useful, not idle** - so the answer is not a fixed reserve but knowing the card's real state and never
asking the kernel for more than is free.

The slot loan showed why a fixed reserve is the wrong tool (`measured-here`, 2026-10-30, `prefill-slot-loan.md`):
at 103,638 prompt tokens the loan arm peaked at 24,401 of 24,480 MiB resident (79 MiB from the card's total) with
prefill +58.9 % and right answers, while a rule built on the sum of ggml backend buffers printed ~4.7 GiB spare: the
bank-stream ring, the backend's grow-only preallocations, the MTP draft context and the runtime's own surfaces are
outside that sum (`code`, review of the loan's round 19). The same arm's GTT grew ~4 GiB beyond the lend bank at
depth (47,442 against 43,449 MiB at 18,802 tokens, loan off 37,095 at both depths) - evicted VRAM or host-side growth,
not yet told apart.

## What the kernel offers (`code`: the installed xe 7.0.14+p1 DKMS tree; the live system, `measured-here`)

- **No knob hands eviction to user space.** TTM decides in the kernel.
- **The card's own numbers, from user space:** `DRM_XE_DEVICE_QUERY_MEM_REGIONS` on a render fd returns each VRAM
  region's `total` and `used` (`xe_query.c:275,293`; `used` is not gated by perfmon in this tree). That is the
  device-wide truth: every client, the kernel's own VRAM objects included - more than a process's fdinfo shows.
  A reply with `used = 0` beside a real total is no figure -- in an older tree `used` sits behind
  `if (perfmon_capable())` -- and is read as a failed query, not as an empty card (`code`: the loan's round 23).
- **The dmem cgroup controller is live:** xe registers each card's VRAM region (`xe_ttm_vram_mgr.c:317`,
  `drmm_cgroup_register_region`); `/sys/fs/cgroup/dmem.capacity` lists both cards (B60 25,669,140,480 bytes);
  the card host's container cgroup carries `dmem.max/min/low/current`, all unset. `dmem.min`/`low` protect a
  cgroup's VRAM from eviction by other cgroups (`paper`: the kernel's cgroup documentation); a process that
  overcommits its own allocations is still evicted by TTM.

## Stages

0. **The accounting workaround (in the slot loan, its next round).** Before every growth (the loan's prompt buffers,
   a grown lend), read the card's `used`/`total` from the xe query; grow only into what is free, minus a small
   reserve for the kernel's own objects that is measured, not guessed. Margin to the card: zero beyond that reserve
   (operator, 2026-10-30, replacing the fixed 1 GiB of the loan's round 17). An eviction guard on top: if the
   process's own buffers fall short of their expected VRAM, by at least a noise floor (64 MiB in the loan, `code`),
   the loan never sheds and never refuses for it: it grows the NEXT lend by that shortfall through the sizing's
   existing growth path (`code`: the loan's `prefill-slot-loan` round 23 - a shed lowers the lend and so raises the
   residency the rule bounds, which cannot relieve the eviction; GTT growth is judged against a begin baseline and
   logged, not acted on). First measurement: whether the ~4 GiB of extra GTT at depth is eviction.
  Round 22's attempted measurement did not engage (`measured-here`, the B60): the OpenCL runtime
  opens two xe clients per card in one process, so the loan's unique-pdev match fell back and
  neither the query, the fdinfo figure nor the guard ran; the loan's round 24 sums the device's
  distinct clients and takes the ioctl on any fd of its pdev (`code`), so the next window's
  guard line is the first read of this.
1. **The same guard for every growth in the process:** the compute reserve at long prompts, the stream ring's first
   touch, the KV at its high-water mark, the MTP draft's buffers.
2. **An arena:** allocate the card's usable VRAM once at load and sub-allocate everything inside it (slots, compute
   buffers, KV, the ring, the backend's preallocations); the loan then reassigns ranges inside memory the process
   already owns, the kernel sees no new allocation and has nothing to evict. Needs the backend's own allocations
   (plain `clCreateBuffer` today) routed through the arena; prior art to survey first (llama.cpp's backend buffer
   types, Strata's and Infernix's VRAM pools, read at source).
2a. **Route A - LEO (operator 2026-10-30: gate it).** Level Zero virtual memory reaches the OpenCL kernels only
   through the runtime's LEO mode (OpenCL implemented on Level Zero; experimental, `EnableLEO=1`; research-vram-arena.md
   section 8, `code`; `paper`: not all extensions, performance tuning ongoing in 26.27). Gate before any arena gain
   counts: test-backend-ops on both cards, the answers at the answer-level bar, a same-window speed A/B against the
   default driver - on the installed runtime and on the newest one (in the fresh-host container). Then the probe
   (`tools/l0_vm_arena_probe.c`).
2b. **Route B - a chunk pool on the default driver (if LEO falls short).** Allocate the usable VRAM once at load as a
   fixed set of large `cl_mem` chunks; a ggml buffer type hands out ranges inside them (the backend already addresses
   every tensor as buffer + offset, `data_device` set in one place, `code`); the backend's grow-only buffers and the
   stream ring route through it; the slot loan becomes a reassignment of chunk ranges (no free, no create: a constant
   driver footprint). Costs: no contiguity across chunks (a tensor or compute buffer fits one chunk, or the compute
   reserve is split), fragmentation managed by size classes, a chunk can still be evicted whole under outside pressure.
3. **Neighbours:** `dmem.min` for a unit's cgroup protects it from another cgroup's allocations (the image service,
   a TTS leg on the same card) - an operator decision, priced by a window with both on the card.

## Gate

Adopted when decode or prefill rises (or a unit gets a larger context/slot budget from the freed reserve) and no
other target degrades: the answers at the answer-level bar, zero GTT growth beyond the host banks in the depth legs,
no xe engine resets.
