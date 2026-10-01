# tier-handoff-doorbell — the per-layer GPU/CPU-tier hand-off through mapped memory, not the driver

**Open** (opened 2026-10-01 from `research-reference-audit.md` §1, the
"0075 hand-off" row). Levers 5 and 8 of the audit's §4.

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

## Where it lives

`moe_3gemm_swiglu_opt.cpp` (`dispatch_cpu_tier`, `cpu_tier_join`, the
staging ring), the tier pool (`MOE_CPU_TIER_SPIN_US` in the 0075 build,
`--moe-cpu-tier-threads`), `moe/host_expert_bank.hpp`.
