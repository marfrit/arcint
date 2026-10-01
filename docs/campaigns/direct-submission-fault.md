# direct-submission-fault — the runtime's direct-submission semaphore evicted under VRAM pressure

**Open (measurement, external fix).**

## Charter

Deep u8:i4 prompts on the 24 GB card crashed (bus error or
`CL_OUT_OF_RESOURCES`) at random depths under VRAM pressure with concurrent
load. Confirm on the record whether the host kernel in use carries the fix.

## The mechanism, as recorded (DESIGN §7.0.2ad)

- Every crash is a GPU page-fault storm at one fixed virtual address with an
  "Engine memory CAT error" and an engine reset (`measured-here`).
- The runtime's allocation log names that address as its own direct-submission
  `SEMAPHORE_BUFFER` in local memory (`measured-here`).
- The buffer is bound once as an ordinary evictable BO and never re-validated,
  so eviction under pressure leaves the next wait reading a non-present page
  (`code` and driver documentation, not measured).

## Reference to follow

The upstream ring-ordering fix in the stable `linux-7.1.y` branch, whose
commit message names the symptom "a hang or a spurious pagefault"; upstream
items `drm/xe` issue 8390, `intel/compute-runtime` issue 948 (with arcint's
allocation-log identification), `drm/xe` tracker item 9141. The host's kernel
state is operator-local (`CLAUDE.local.md`).

## Gate

N ≥ 5 runs of the recorded crash configuration (24 GB card, u8:i4, the
98,147-token prompt, `ARCINT_PREFILL_CHUNK_CAP=off`) on the current host
kernel with zero fault lines; the upstream items' status recorded.

## Where it lives

DESIGN §7.0.2ac, §7.0.2ad; `src/exec/backend_ov.cpp`
(`ARCINT_PREFILL_CHUNK_CAP`); CHANGELOG "Known defects".

Full history: `git show b0447b8:docs/campaigns/direct-submission-fault.md`.
