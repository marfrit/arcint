# Campaigns — one defect, one document, one session's worth of work

`docs/milestone-0.3.0.md` planned 0.3.0 as fourteen milestones in one
document, and its backlog table carried what 0.3.0 did not close as four
rows. That shape has stopped fitting: a backlog row bundles several levers
with different owners, and a session that picks one up has to read the
whole record to find its edges. From 0.3.1 on, every open defect or lever
is its own **campaign**: one document in this directory, self-contained
enough that a fresh session can carry it with nothing but that document
and the DESIGN sections it cites.

## Rules

- **One campaign = one defect or one lever, with its own gate.** If two
  levers have different owners or different gates, they are two campaigns,
  even when one backlog row named both. A campaign that turns out to have
  two problems inside is split, not stretched.
- **The document is sufficient.** It names the measurement that defines
  the defect (with card, depth, precision, configuration and the DESIGN
  section that recorded it), what is known against what is hypothesised,
  the gate, the entry criteria, the files and knobs and counters involved,
  and the acceptance cells that prove it. Nothing operator-local: hosts,
  paths and unit names stay in the git-ignored `*.local.md` files, as
  `CLAUDE.md` says.
- **The gate is on the record before the work starts**, copied from the
  backlog row or the DESIGN section that found the defect, and it is a
  measurement that can fail. A campaign may close as a *verdict* — the
  defect measured and found not worth fixing, or not a defect — but only
  with the measurement that says so.
- **Pipeline, every time:** recon (read the cited sections and the code,
  write down what is actually there) → design note (a short `docs/design-
  <slug>.md` when the change is more than a fix) → red-first
  implementation → one card window at the end, not many along the way →
  review before commit → a DESIGN `§7.0.2x` record and a CHANGELOG line
  when it closes. Reviews are not skippable.
- **Invariants stay non-negotiable:** DESIGN §3.4 (history-independent
  greedy output) and §3.8, the §5 ladder, the measurement discipline in
  `CLAUDE.md`. A campaign that would trade one for a number does not
  close; it records the trade as a finding and stops.
- **Status is a dated log at the bottom of the document**, appended, never
  rewritten. The milestone document's backlog rows point here and are not
  edited further.
- **Releases collect closed campaigns.** A point release ships whatever
  closed since the last tag; no release waits for a campaign, and no
  campaign is started to fill a release.

## Template

    # <slug> — <one-line charter>
    ## The defect, as measured
    ## Known against hypothesised
    ## Gate
    ## Entry criteria
    ## Scope — in / out
    ## Where it lives
    ## Pipeline for this campaign
    ## Invariants
    ## Status

## The campaigns

Ordered by what unblocks what; the order is advice, not a queue.

| campaign | charter | origin | size |
|---|---|---|---|
| [test-ladder-close](test-ladder-close.md) | close the 0.3.1 lead item: fill the acceptance references from the runners' own windows, record the first real run | 0.3.1 lead item, `docs/design-0.3.1-test-ladder.md` | small, closed 2026-09-05 |
| [prefill-fallback-tristate](prefill-fallback-tristate.md) | patch 0018's overloaded `false` return in the per-expert prefill loop becomes a three-way answer | DESIGN §7.0.2ae, patch 0018 header | small |
| [static-partition-cold-start](static-partition-cold-start.md) | a cold sequence's first processes pay minutes of warming under the static partition; find the owner, then remove it | DESIGN §7.0.2ai | medium |
| [static-partition-prefill](static-partition-prefill.md) | tier-ON prefill runs at a third of tier OFF because every layer takes the per-expert fallback; a grouped prefill on the resident subset | DESIGN §7.0.2ai | medium–large |
| [partition-seeding](partition-seeding.md) | seed the static partition from a fixed calibration routing histogram so the pinned half is the hot half — only if decode variance survives the two above | DESIGN §7.0.2ai, patch 0013 | medium, conditional |
| [u8i4-prefill-price](u8i4-prefill-price.md) | `--paged-kv u8:i4` costs +7/+25/+72 % prefill time with depth; measure the mechanism, then decide | DESIGN §7.0.2aa, M8 row | small–medium |
| [u8i4-deep-prefill-fault](u8i4-deep-prefill-fault.md) | the out-of-resources fault at deep u8:i4 prefill on the 16 GiB card; the chunk belt mitigates, the plugin-side cause is open | DESIGN §7.0.2ab, §7.0.2ac | medium |
| [direct-submission-fault](direct-submission-fault.md) | the runtime's direct-submission semaphore evicted under VRAM pressure; diagnosed, the kernel-side fix operator-local and unmeasured on the record, not closed | DESIGN §7.0.2ad | small (measurement), external |
| [mtp-cycle-wall](mtp-cycle-wall.md) | MTP never beats plain decoding at depth on the dense agent: a 390 ms cycle against a 130 ms break-even; cut the cycle or record the verdict as final | DESIGN §7.0.2ag | medium |
| [turnstile-wall-time](turnstile-wall-time.md) | the turnstile test orders threads by wall-clock sleeps and flaked once under build load | 0.3.1 window | small, closed 2026-09-05 |
| [pruefstand-cell-remote](pruefstand-cell-remote.md) | the Prüfstand acceptance cell can only skip by name where the harness does not live; make it runnable from the card window | 0.3.1 window | small, closed 2026-09-05 |
| [sub4bit-vram-kernel](sub4bit-vram-kernel.md) | VRAM-resident sub-4-bit expert weights: a new GPU kernel path with in-kernel dequant, the only route that pays M10's gate in its own currency | M10 re-scope, DESIGN §7.0.2ah | large |
| [kquant-host-storage](kquant-host-storage.md) | the host compute tier computing K-quant blocks natively, so offloaded experts are sub-4-bit on disk and in the host pool | M14 extension, DESIGN §7.0.2ah | large, after the kernel campaign's format measurement |
