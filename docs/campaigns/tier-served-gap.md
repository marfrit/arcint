# tier-served-gap — the CPU tier's decode calls at bench speed in serving

**Open.**

## Charter

Flash-Next decode on the B60 is bound by the CPU tier: 48 tier calls a
token, about 0.8–0.95 ms each for a one-token call (3.8–3.9 experts), and a
four-token verify window costs ~3.6 plain steps, which is what blocks
multi-draft MTP (`mtp-cycle-wall.md`). The same experts run in ~0.37 ms on
the same host outside the server. This campaign finds where the served call
loses that time and removes it.

## Reference to follow

- Strata, the CPU expert pool (`include/strata/kernels/cpu/pool.hpp`,
  `src/kernels/cpu/pool.cpp`): workers pinned one per physical core, the host
  draining, flat epoch-tagged claims, the two-phase row split
  (`run_split_multi`). Strata measured 36.3 GB/s alone against 26.9 GB/s with
  an unpinned spinning host (`pool.hpp:95-102`; code).
- Strata, the RAM copy of the cache complement: every expert the card does
  not hold is copied into a locked RAM arena and the file's pages are
  released with `MADV_DONTNEED` + `POSIX_FADV_DONTNEED`
  (`src/core/expert_source.cpp:1215-1270`; code). Device copies come from that
  arena, PLE rows are read with `O_DIRECT` (`src/platform/direct_file.cpp`):
  nothing is cached twice.
- Strata's routing-aware lookahead (`src/core/expert_source.cpp:814-930`,
  `src/program/generate.cpp:3001-3030`): warms the next layer's predicted
  file pages; its dot is vectorised (`bf16_rows_dot_multi`; code).

## Gate

Speed, per phase: one-token tier call time and served decode t/s (500-token
answer, 20k needle) improve beyond the arm-to-arm spread, prefill within its
spread. Answers: the needle and the long answer stay right; the tier's bytes
are unchanged by construction for the pool (cell below), and page-cache
changes do not touch the arithmetic.

## Current state (measured-here, B60, d48q8, served configuration of `docs/serving-config-flash-next.md`)

- **The kernel is not the bound.** Standalone, served format (IQ3_XXS
  gate/up, IQ4_NL down, 2.36 MB an expert, cold): 528 µs an expert on one
  core (4.4 GB/s), linear to 6 cores, 30.6 GB/s on 8 and 31.5 GB/s with SMT.
  The existing pool: 148 / 370 / 1,378 µs for 1 / 4 / 16 experts.
- **The pool is not the bound.** A Strata-style flat pool (pinned workers on
  cores 1–7, host draining, flat CAS claims, two-phase row split; bytes equal
  to `moe_cpu_expert` for 1–16 experts and 1–3 jobs, a dropped-task mutant
  caught) measured 110 / 343 / 1,284 µs standalone, but served it moved
  nothing: 962 vs 922 µs a one-token call, 919 with the rest of the process
  on the SMT siblings. Neighbour load costs the kernel 12–28 % (a standalone
  single-thread bench beside a decoding server), not 2.5x.
- **Not the activations' memory type.** Copying x out of the usm_host buffer
  before the tasks: no change.
- **The loss is the file tier.** Split by whether a one-token call has an
  expert outside the host bank (26 % of tier experts are read through the
  file mapping): all banked 471–517 µs (4.4–4.7 experts), some unbanked
  1,334–2,324 µs (3.4–3.6 experts), across a dozen processes. The unbanked
  third of the calls is ~60 % of the tier's decode time.
- **Page-cache census** (mincore over every expert, by class): the cache
  holds ~19 GiB of the file; 4.6–6.9 GiB of it were second copies of experts
  on the card and 0.8–1.9 GiB of banked ones, while 2.4–3.6 GiB of the
  unbanked complement (14.7 GiB) were missing. Sources of the duplicates
  (code): slot fills and adaptive swap reads through buffered `pread`, and
  experts promoted to the card after the tier had read them through the
  mapping.
- **Dropping the duplicates** (`POSIX_FADV_DONTNEED` after every
  device-bound read, releasing a promoted expert's pages, a one-shot release
  pass after the bank fill): card duplicates 1.1–1.6 GiB; complement
  residency unchanged (10.7–12.3 GiB) — the freed cache is not refilled with
  what the tier will need. Speed within the spread.
- **Release plus warm** (the duplicate fixes and a one-shot
  `POSIX_FADV_WILLNEED` of all ~7,700 complement experts after the bank
  fill; three arms against a control): the complement settles at ~11.9 of
  14.7 GiB again, banked duplicates regrow to 2.5–2.9 GiB within the run;
  unbanked calls 1,371–1,685 µs vs 1,250, 500-token decode 13.8–14.2 vs
  14.0 t/s. On this host (52 GiB container, 30 GiB bank) the page cache does
  not keep the complement, however it is managed.
- **Negatives on record:** a one-shot release + `WILLNEED` of the whole
  complement before the duplicate fixes (the complement did not stay; calls
  up to 0.97 s while the reads queued); the lookahead's dot vectorised (runs
  stay ~1 ms, skips ~50 %; no change — the dot was not its cost).
- **Where the reference stands** (`paper`: a third-party video review of
  Strata, 2026-10-02, matching the code's 1,382,400-byte blob): Strata's
  headline numbers are its 2-bit build, 34 GB of experts on a 64 GB host —
  the whole complement in RAM. Its 3-bit build is quoted at ~2.1x llama.cpp,
  the 2-bit at up to 6.2x. Our served experts are the 3-bit native format,
  2.36 MB each (~58 GB), against a 30 GiB bank. The open decision for the
  operator: an expert build whose host complement fits the bank (2-bit or
  mixed), judged by the answer-level bar against the 3-bit build.
- A `drop2` arm died once during load: a userspace segfault in two threads
  at one address (0x100000020), the B60 faults after it from the process
  teardown; the next arm loaded normally. Unlocated; legs now keep cores.

## Where it lives

Dev plugin tree (uncommitted, env-gated): `moe_cpu_expert.{hpp,cpp}`
(`CpuTierFlatPool`, `cpu_tier_run_flat`; `MOE_CPU_TIER_FLAT`),
`moe_3gemm_swiglu_opt.cpp` (doorbell wiring, the diagnostic counters
`db_breakdown` / `db_split_1tok`, the page-cache pass), `moe_otd_runtime.hpp`
(`ParallelWeightReader::release`, drop after read;
`MOE_OTD_KEEP_PAGE_CACHE=1` restores), `expert_weight_providers.cpp`
(`otd_page_cache_complement`, `otd_page_cache_census`;
`MOE_PAGECACHE_CENSUS_S`).
