# tier-handoff-doorbell — the per-layer GPU/CPU-tier hand-off through mapped memory, not the driver

**Built, gate passed 2026-10-02 (patch 0077).** Opened 2026-10-01 from
`research-reference-audit.md` §1, the "0075 hand-off" row; levers 5 and 8 of
the audit's §4.

## Charter

In each MoE layer the GPU publishes the activation and the routed ids into
mapped host memory with a fence, the CPU tier spins on that flag instead of
waiting on the driver, writes its rows back into device-visible host memory,
and the GPU waits on a host flag; the decode pass is recorded once and
replayed. The tier's workers are pinned one per physical core, with the
coordinator on its own core and a short spin before sleeping.

## Reference to follow

**Strata** (`~/src/Strata-ref`):
- **The doorbell** (`code`: `src/kernels/cuda/elementwise.cu:186-311`).
  `doorbell_publish_kernel` writes x, ids and weights with volatile stores and
  `__threadfence_system()`; `doorbell_ring_kernel` increments the sequence in
  memory (not a captured literal); `doorbell_wait_kernel` polls the host flag.
- **The buffers** (`code`: `src/core/layer.cpp`, `doorbell_init`): mapped
  pinned `x_f`, `ids`, `weights`, `seq`, `flag` with device aliases;
  `wait_flag_ge_kernel` (`src/kernels/cuda/verify_kernels.cu:426`).
- **One recorded pass** (`code`: `src/core/verify.cpp`): the 48-layer pass is
  a captured graph per window size, so the host makes no synchronising driver
  call inside it.
- **The worker pool** (`code`: `include/strata/kernels/cpu/pool.hpp`): one
  flat batch per layer, lock-free claims tagged by epoch, spin before sleep.
- **Effect** (`paper` §3.2): one layer ≈ 0.6 ms at 4K context.

**FreeToken** (`~/src/FreeToken-ref/python/freetoken/moe/cpu_executor.py`):
- mapped-pinned `ready` / `done` / `err` flags, raised by the GPU at submit
  and spun on by the host (`code`: `:1-14`, `:262-279`);
- one worker per physical core, pinned, the coordinator on its own core
  (`code`: `physical_core_cpus` `:92`, `resolve_threads_and_affinity`
  `:119`, `:250-255`).

Candidate OpenCL mechanisms on arcint's stack (to verify on the installed
runtime): device-accessible USM host allocations for the published buffers
and flags; `cl_khr_command_buffer` for the recorded pass.

## Gate

Per phase (`CLAUDE.md`): on the served Flash-Next arm (B60, `d48q8`, ratio 75
+ census128, host bank, the 20,085-token needle prompt), decode or prefill
faster than today's arm in the same window and the other phase within the
run-to-run spread; the answer-level bar (needle answered; window-0 KL no more
than 0.03 nats above the baseline arm's; argmax down at most 1 point).

## Current state

- **The hand-off today** (`measured-here`, B60, `d48q8`, 0074, decode on a
  short prompt): 136 ms a token, the device idle ~46 % of it; per token 362
  `clWaitForEvents`, ~98 DtoH + 48 MtoH + 12 MtoD readback and staging
  copies and 48 DtoM. Of the idle, 47.2 ms waits on the tier's writeback and
  31.3 ms is copy-to-copy round trips. The dense 27B control on the same card
  is ~87 % device-bound.
- **Measured pieces** (`measured-here`, B60, 20,085 tokens): a 200 µs worker
  spin before sleeping raised prefill 62.8 → 66.8 t/s (+6.4 %) with decode
  inside the spread; a zero-copy tier writeback into a usm_host output removed
  292.7 HtoD calls a token. Both were built as patch 0075, which is not in the
  series; under the per-phase gate the spin is adoptable.
- **Device-side routing exists** on the fully resident pool: patch 0067
  writes decode's pair table from `topk_id` on the device and removed the
  per-layer readback there (A770, 35B: 19.3 → 28.1 t/s, `measured-here`).

## Build (patch 0077, 2026-10-02)

- **Where the decode token went first** (`measured-here`, B60, `d48q8`,
  patch 0076 served configuration, CLIntercept device timing, decode delta
  of 393 tokens): ~70 ms a token, the device busy 32.6 ms of it (dense
  projection GEMMs 17.2 ms in 761 launches, routed experts 4.7, the router's
  arg-max 2.3, 216 tier-row copies 1.9), the CPU tier's join ~20 ms, the
  device idle ~53 %.
- **The handshake on this card** (`measured-here`, probe
  `doorbell_probe.cpp`, B60, xe): host USM reports access and atomics but
  not concurrent access. A kernel polling a host-written flag sees the write
  only through `__builtin_IB_lsc_load_global_uint(..., L1UC_L3UC)` with an
  acquire fence (system scope) inside the loop: 8 of 8 rounds, 4.5-5.4 us
  from the host's store to the kernel's completion. Acquire atomics saw it in
  2 of 8 rounds, volatile and read-modify-write loads in 0 of 8, and the
  uncached load without the fence was hoisted out of the loop (0 of 6). The
  other direction needs nothing special: a release store from a kernel is
  host-visible ~20-40 us after enqueue, even with a 5 s kernel queued
  behind it.
- **Why not OpenCL user events** (`code`, NEO): a command waiting on a user
  event puts the queue in blocked mode, and every later command is submitted
  by whichever thread sets the event -- the per-layer submission would move
  onto the CPU tier's critical path. Strata's GPU-side wait avoids exactly
  that.
- **The build** (`MOE_DOORBELL=1`, decode calls of up to 63 pairs and 8
  tokens on a partly resident pool with the CPU tier):
  - a route + publish kernel maps each routed id through the layer's
    residency table (device memory, rewritten after swaps by an enqueued
    copy) to its slot or -1, writes the pair table, and publishes the ids,
    the slots it chose, the routing weights and x to host memory before a
    release store of the call's sequence number;
  - the batched expert kernels skip the -1 rows;
  - a wait + merge kernel polls the host's done flag as above (bounded) and
    copies the CPU tier's rows into `y` before the reduce;
  - the host half (count the ids for the adaptive cache, run the tier on
    exactly the pairs the device marked -1, write the rows, ring done) runs
    on a coordinator thread in FIFO order, so the enqueueing thread submits
    the whole step ahead (stage B). A coordinator error still rings done
    and is rethrown on the next call.
- **Results** (`measured-here`, B60, `d48q8`, 2,076-token needle then a
  500-token decode, fresh process per arm, same plugin):

  | arm | needle decode | 500-tok decode | prefill 2k | needle |
  |---|---|---|---|---|
  | patch 0076 served configuration | 10.0 t/s | 13.2 t/s | 62.9 t/s | right |
  | + doorbell, host half on the enqueueing thread (stage A) | 10.2 | 14.1 | 61.8 | right |
  | + doorbell, coordinator thread (stage B) | **12.3** | **15.4** | 63.4 | right |

  Stage A was neutral: the enqueueing thread still ran the tier before it
  could submit the next layer. 27,936 doorbell calls, 0 wait timeouts.
- **Gate passed** (`measured-here`, B60, `d48q8`, patch 0076's served
  configuration in both arms, 20,085-token needle, fresh process per arm):
  without the doorbell prefill 63.5 / needle decode 10.7 / 500-token decode
  14.0 t/s, with it 60.8 / **12.7** / **16.0**; needle right; window-0 KL over
  three arms each 0.409 vs 0.412 nats mean, argmax 79.1 vs 79.0 % (one arm
  pair alone read +0.038, inside the control's own 0.041 spread). The gate
  ran on the build before the review's fixes; the final build (bounded poll
  ~1.3 s, NaN rows and a throw on a timeout or a failed host half, drain on
  re-entry and in the destructor, the never-filled pinned slots read in the
  background -- filling them synchronously cost the first decode tokens 7 s)
  read 9.8 / 13.5 t/s without and 11.6 / 16.2 with at 2k, needle right.
  Lands as plugin patch **0077** (runtime `+p27`).
- **Open:** the device is still idle while the coordinator computes a layer
  (the next layer waits on its rows); the dense projection GEMMs run at
  ~45 % of the card's bandwidth (17.2 ms a token); multi-draft MTP (P4)
  amortises both over more tokens per pass.

## Where it lives

`moe_3gemm_swiglu_opt.cpp` (`dispatch_cpu_tier`, `cpu_tier_join`, the
staging ring), the tier pool (`MOE_CPU_TIER_SPIN_US` in the 0075 build,
`--moe-cpu-tier-threads`), `moe/host_expert_bank.hpp`.
