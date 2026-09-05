# u8i4-deep-prefill-fault — the out-of-resources fault at deep u8:i4 prefill on the 16 GiB card; the chunk belt mitigates, the plugin-side cause is open

## The defect, as measured

A deep `--paged-kv u8:i4` prefill on the 16 GiB card fails with a GPU
out-of-resources error — `xe: VM worker error: -12` / `exec queue reset
detected`, surfacing as `CL_OUT_OF_RESOURCES` — and every later request
in the process fails the same way, where the identical prompt prefills
and decodes at u8 (DESIGN §7.0.2ab; CHANGELOG's "Known defects", "Still
open at 0.3.0"). The fault moves with the chunk, bracketed one process
per cell on the same 131,072-token pool:

| prefill chunk | prompt tokens | result |
|---|---|---|
| 128 | 119,074 | faults on the 171,312-token auto-fit pool; passes on the table's own 131,072-token pool |
| 64 | 119,074 | prefills, 605 s (197 t/s), decodes |
| 128 | 71,689 | passes |
| 256 | ≈72,000 | faults |
| 512 | 35,227 | prefills, 233 s (151 t/s), decodes |

Halving the chunk clears the 119k prompt, doubling it breaks the 72k one
— but chunk 512 passes with a larger raw proxy (chunk × query heads ×
head size × 4 B × partitions) than the chunk-128 cell that faults, so
the proxy is a tuned yardstick, not a threshold (§7.0.2ab retracts "one
scratch buffer is the price and the fault" as a mechanism). Sampled
every two seconds during the 119,074-token prefill at chunk 128: a
131,072-token pool's free VRAM falls to a 492 MiB floor and passes; the
171,312-token auto-fit pool (belt disabled, `ARCINT_PREFILL_CHUNK_CAP=
off`) reaches 0 MiB as the plugin's ≈500 MiB intermediate resizes, and
the driver logs the `-12` — a VRAM budget the fit did not charge: the
mixed-stage paged-attention intermediate (`exp_sums`, `max_logits`,
`tmp_out`, sized by query heads × head size × bytes × partitions) grows
with the past, absent at past 0 where the activation probe runs.

The chunk belt (`ARCINT_PREFILL_CHUNK_CAP`, `fit.h`'s `prefill_chunk_
cap_for_packed_values`) caps the served chunk under 4-bit values — 64 at
n_ctx 131,072, 32 at 171,312 — at ~12% of prefill throughput on short
prompts; `packed_values_prefill_scratch_bytes` charges the buffer into
the reservation (12 KiB/token at chunk 128, 6 at 64 — more than the
8.8 KiB/token u8:i4 KV itself). Patch 0015 (bounded partials, applied
for a different, now-closed crash) raises the ceiling further: bounded
at 32, auto-fit lands at 165,680 tokens at chunk 128, 48.5 MiB charged
(§7.0.2ac); with the belt alone, honestly charged, auto-fit lands at
101,984 (§7.0.2ab). The plugin-side cause stays open regardless: chunk
512's larger buffer (1,104 MiB) passes where chunk 128's smaller one
(932 MiB) faults — not a size threshold. Not to re-open here: the
*deep-prompt crash* (a bus error on the same fixed prompt at random
depths within its own prefill), traced to the OpenCL runtime's own
direct-submission semaphore evicted under VRAM pressure — a
driver/runtime interaction, not this plugin's kernels (DESIGN §7.0.2ad,
its own campaign `direct-submission-fault`); both reproduce at deep
u8:i4 prefill on fault-reporting cards and were run down together
(§7.0.2ac/§7.0.2ad), but that closure explains only the *crash*.

## Known against hypothesised

Known: the fault signatures; the bracket above; that free VRAM reaching
0 during the resize correlates with the fault; that patch 0015 raises
the ceiling; that the belt's proxy is a tuned yardstick, not a derived
threshold. Hypothesised, unconfirmed against the allocation code: that
the buffer's *size* is what the plugin should size instead of charging
depth-scaled VRAM around it — no patch removes the depth dependency;
"the proxy compared buffers, not headroom" (§7.0.2ab) is a reading, not
yet a measurement.

Prior art, surveyed 2026-09-05 and recorded with sources, licenses and
Arc applicability: `research-kv-quantisation.md` (same directory). Its
"what transfers" section is the recon's starting point, not a substitute
for it.

## Gate

The fault reproduced with one variable at a time (buffer size / chunk /
depth) on the 16 GiB card, its owner named with a measurement, not a
code reading (§7.0.2ab's own admission). Then either closed in the
plugin (a patch with the red case, per the 0015/0016 convention), or
recorded as a driver/runtime limit with the belt kept as the documented
mitigation and the scratch charge as the permanent price. Either way,
`depth-ladder` stays green at `u8` and `u8:i4` on both cards — its
fault-signature grep is this campaign's own regression test.

## Entry criteria

Met: the bracket and VRAM-sampler trace are on record (§7.0.2ab); patch
0015 is committed, its ceiling measured (§7.0.2ac); `depth-ladder`
already runs this check at 98,147 tokens on both cards, both precisions.

## Scope — in / out

In: attributing the fault to a plugin allocation path; a patch if
plugin-side and fixable; keeping the belt and scratch term as mitigation
either way. Out: the direct-submission crash (`direct-submission-fault`);
the prefill speed price (`u8i4-prefill-price`); the default KV precision.

## Where it lives

`src/exec/fit.h` (`packed_values_prefill_scratch_bytes_ex`, the charged
term; `prefill_chunk_cap_for_packed_values_ex`, the belt;
`fit_context_packed_values`, folds it into auto-fit); `src/exec/
backend_ov.cpp` (`ARCINT_PREFILL_CHUNK_CAP` and its bypass, `PAGED_
ATTENTION_MAX_PARTITIONS`); `contrib/packaging/marfrit-openvino/
patches/0015-*.patch` and `0016-*.patch` headers; `tests/test_fit.cpp`'s
`packed_values_prefill_scratch_bytes_*` family; `tests/acceptance/
cells.json`'s `depth-ladder`; DESIGN §7.0.2ab (bracket, belt), §7.0.2ac
(patch 0015 — only the crash it uncovered closes at §7.0.2ad).

## Pipeline for this campaign

Recon (re-read §7.0.2ab against plugin source; confirm 0015/0016
shipped) → one-variable-at-a-time reproduction, VRAM sampler running →
design note if a plugin fix is indicated → red-first plugin patch →
rebuild → plugin unit tests → `depth-ladder` on both cards, both
precisions → review → commit as a patch or a documented verdict →
DESIGN §7.0.2x record, CHANGELOG line, the backlog line closed/restated.

## Invariants

Byte-identical to the unpatched plugin at the unbounded setting
(§7.0.2ac); the belt never raises the chunk past what was requested,
never touches `u8`/`f16`, floors at `--kv-block-size`; `depth-ladder`
stays green on both cards at both KV precisions throughout.

## Status

- 2026-09-05 — opened from the 0.3.0 known-defect list; nothing started.
- 2026-09-05 — first real run of the acceptance target (DESIGN §7.0.2aj):
  the depth ladder's 24 GB-card u8:i4 cell faulted (`CL_OUT_OF_RESOURCES`
  on `clEnqueueMapBuffer`, then `clFinish`) on the *third* 98k-class
  prefill in one process (two sizing rounds of 78,867 and 98,187 tokens
  had completed at 144 and 121 t/s), with the belt at its default (chunk
  cut 2,048 → 128, scratch 606 MiB), partials unbounded, prefix cache off;
  the host kernel log shows a GuC job timeout ("not started"), a device
  coredump, a GT reset and `Timedout job` lines in that arcint and in a
  second, unidentified process — a signature that is not §7.0.2ad's
  (no page-fault or CAT lines there, and that class never shows a
  `Timedout job`), so its class is open. The record's passes of this
  prompt on this card at u8:i4 (§7.0.2ad's knob table) ran with the
  partition bound at 32 or the belt bypassed, one prefill per process;
  the host's state in those windows was not recorded. One sample; the
  bound, the request count within a process and the second client are
  three variables this campaign's bracket must separate. The 16 GiB cell was
  refused at load by the runner's own margin, not by the card (fit admits
  101,232 tokens unbounded at chunk 128) — a runner correction, not a
  finding here.
- 2026-09-05 — the scratch charge on the microkernel path, measured
  (DESIGN §7.0.2at): with the host VRAM sampler running, a 71,727-token
  u8:i4 prefill at chunk 128 on the 16 GiB card consumes ≤ 9 MiB of free
  VRAM on the patch-0020 plugin and 573 MiB on `+p4`'s generic kernel
  (the fit charged 436.5 MiB on both loads; the 256 MiB margin covered the
  generic path's excess — one more point on the yardstick's error, kept
  here). The fit now charges no scratch term and applies only the measured
  chunk cap when the GPU plugin's build number names patch level 6 or
  later and the pairing is eight-bit keys with four-bit values (fit.h's
  `packed_values_mixed_stage_on_micro`; the contract is the recipe's
  stamp, since 0020 adds no property). On the `+p6` package, auto-fit
  admits 171,392 tokens (101,824 charged) and a 118,454-token prefill at
  chunk 128 ran on that pool with free VRAM flat at 971–978 MiB, no fault
  — the pool size and depth class of this campaign's defining fault. The
  gate is unchanged for every plugin below `+p6` and for the pairings
  0020 does not admit (i4:i4, four-bit keys): the fault's owner on the
  generic path is still a reading, and the belt and charge stay there.
  Owed to this campaign: `depth-ladder` on both cards at both precisions
  against the `+p6` package.
