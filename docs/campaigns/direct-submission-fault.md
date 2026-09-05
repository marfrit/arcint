# direct-submission-fault — the runtime's own direct-submission semaphore, evicted under VRAM pressure; diagnosed and mitigated by a kernel backport, not closed

## The defect, as measured

Deep-prompt crashes on the 24 GB card at `--paged-kv u8:i4` — a bus
error or `CL_OUT_OF_RESOURCES` at random depths within one fixed
98,147-token prompt's own prefill (traced at ~3,600, 12,000 and 40,000
tokens on three separately crashing processes) — were first framed as
depth- or partition-count-triggered ("past 98k tokens", "384
partitions"). RETRACTED (DESIGN §7.0.2ad, per §7.0.1): no partition
count or chunk size ever bracketed is a threshold; the trigger is VRAM
headroom with concurrent host or other-card load, not depth.

**The mechanism, in three legs (§7.0.2ad).** (1) Every crash is a GPU
page-fault storm at one fixed virtual address, identical across
processes and builds (483 fault lines compute-engine, 8 copy-engine,
over 9 crash events), with "Engine memory CAT error" and an engine
reset at the same instant. (2) The runtime's allocation log identifies
that address as a `SEMAPHORE_BUFFER` in local memory — its own direct-
submission ring and semaphore; no model buffer, KV page, or plugin
intermediate sits there. (3) Read from source and driver docs, not
measured like legs 1–2: the runtime binds those buffers once as
ordinary evictable user BOs, never re-validated — under VRAM pressure
with concurrent load the driver can evict the buffer, direct submission
never rebinds it, the next wait reads a non-present page, engine resets.

Exonerated by tracing, not elimination: 0 stale intermediate-buffer
bindings over 5,867 dispatches; the crash depth varies within one fixed
prompt, which a kernel-side threshold cannot explain; the untouched,
unpatched plugin passes on a quiet host — 2 of 2 (845 s and 802 s) at the
identical crash configuration, its direct-submission-off partners also
passing (820 s and 840 s) — while it crashed repeatedly under the
diagnostic bypass (`ARCINT_PREFILL_CHUNK_CAP=off`) built to reproduce a
*different*, already-disambiguated fault (the u8:i4 out-of-resources
fault, `u8i4-deep-prefill-fault`). Separately, with the belt at default
and the fit's own scratch term active (a different configuration, not
the bypass), the identical 98k cell passed 2 of 2 at 830.7 s (§7.0.2ad's
first knob row).

A ring-ordering fix exists upstream in a stable kernel branch
(`linux-7.1.y`), fixing the exact write-ordering gap leg 3 names. DESIGN
records moving the dev host to a kernel carrying it as "an operator
decision, not made here" (§7.0.2ad) — no run of the crash configuration
against a kernel carrying that backport is on the public record, and no
such run is claimed here; the 830.7 s sample above is the DEFAULT-belt
headroom configuration, not the bypass every crash used, so it is not a
post-backport sample of the crash configuration either. Kernel version
and backport state are operator-local (CLAUDE.local.md), not repeated
here. The gate's first measurement is therefore N ≥ 5 runs of the bypass
crash configuration (24 GB card, u8:i4, the 98,147-token prompt,
`ARCINT_PREFILL_CHUNK_CAP=off`) on the current host kernel, whatever it
is at the time this campaign runs — not an assumed-backported one.
Upstream items (§7.0.2ad, cited by number, not URL): `drm/xe` issue 8390
(open, engine resets under sustained decode on four LLM stacks of this
card's generation); a comment on `intel/compute-runtime` issue 948 (the
same fault class, our own allocation-log identification added); `drm/xe`
tracker item 9141, linking issues 8390, 8651 and 7810 — a second,
not-yet-separated candidate mechanism among them is the same
`linux-7.1.y` write-ordering fix, whose own commit message states the
symptom as "a hang or a spurious pagefault".

## Known against hypothesised

Known: the fault address and its identification as the runtime's own
semaphore buffer (legs 1–2); the 0-stale-binding exoneration; every
crash on record used the diagnostic bypass, and the identical cell
passes with the belt at default. Hypothesised, marked as such in
§7.0.2ad: leg 3 (eviction, never revalidated), read, not measured;
whether the upstream `linux-7.1.y` write-ordering fix closes it at all —
no run against a kernel carrying that fix is on the public record; the
two upstream mechanisms as the same fault, not two — kept "not yet
separated" in the record's own wording.

## Gate

N ≥ 5 runs of the recorded crash configuration (24 GB card, u8:i4, the
98,147-token prompt, the bypass at unbounded) on the current host kernel
with zero fault lines. If the operator has since moved the host to a
kernel carrying the upstream write-ordering fix, N ≥ 5 there too, to
isolate the kernel as the variable rather than a quiet host, which
already passes without either. If it still faults on the current kernel:
an arcint-side headroom rule with its own measurement. Either way, the
upstream status is recorded.

## Entry criteria

Met. The mechanism (legs 1–3) and the exoneration trace are on the
record (§7.0.2ad); no post-backport measurement of the crash
configuration is on the record, and none is assumed here; the crash
config is named precisely enough to rerun; kernel version and whether
the upstream fix has been applied are operator-local (CLAUDE.local.md),
not repeated here.

## Scope — in / out

In: the N ≥ 5 (current kernel) / N ≥ 5 (fix-carrying kernel, if
available) comparison; recording upstream items' status; a headroom
rule only if the current host kernel still faults. Out: fixing
the driver or runtime (upstream's to resolve); the prefill speed price
and the out-of-resources fault at deep prefill (`u8i4-prefill-price`,
`u8i4-deep-prefill-fault` — sharing a bypass and a card, not a
mechanism); disambiguating the two upstream mechanisms (their own open
question, not chased here).

## Where it lives

DESIGN §7.0.2ac (the crash first surfacing under patch 0015's testing)
and §7.0.2ad (the closure — three legs, the two-knob table, the
upstream match; supersedes §7.0.2ac's rebind reading). `contrib/
packaging/marfrit-openvino/patches/0015-*.patch` and `0016-*.patch`
headers, recording neither patch causes or closes it. `src/exec/
backend_ov.cpp`'s `ARCINT_PREFILL_CHUNK_CAP` bypass (~3128, ~5117) —
the switch every crash used. CHANGELOG's "Known defects" entry.
Kernel/DKMS state: CLAUDE.local.md.

## Pipeline for this campaign

Recon (check the host kernel's current state and whether the upstream
write-ordering fix has been applied since this record; re-read
§7.0.2ad's two-knob table) → a card window: N ≥ 5 on the current kernel,
N ≥ 5 more on a fix-carrying kernel if one becomes available, same
config, quiet host → if the current kernel still faults, a design note
for a headroom rule, red-first implementation, a second window → review →
DESIGN §7.0.2x record (largely external, arcint measures/discloses);
CHANGELOG line; upstream status recorded.

## Invariants

No arcint-side change trades away the fit's own scratch-charging
reservation (`u8i4-deep-prefill-fault`'s term) as a substitute for this
comparison — separate mechanisms sharing one bypass flag, already
retracted once (§7.0.2ad). Any headroom rule is history-independent — a
pure function of VRAM state and the request, never of prior requests in
the same process. Marked as largely external (driver) work; arcint's
part is measurement and disclosure, not a fix.

## Status

- 2026-09-05 — opened from the 0.3.0 known-defect list; nothing started.
- 2026-09-05 — an event on the same card and prompt that is **not** this
  class by its own discriminator, recorded at DESIGN §7.0.2aj so nobody
  counts it here by mistake: during the acceptance target's depth ladder
  (24 GB card, u8:i4, the 98k prompt, belt at default chunk 128, partials
  unbounded), the host kernel log shows a GuC job timeout ("not started"),
  a device coredump, a GT reset and `Timedout job` lines in the ladder's
  arcint and in a second, unidentified process — no page-fault storm, no
  CAT-error line, and §7.0.2ad says explicitly that no `Timedout job` line
  accompanies any crash of this class. Class open; owed by the ladder's
  rerun on a quiet card (`u8i4-deep-prefill-fault` carries it).
