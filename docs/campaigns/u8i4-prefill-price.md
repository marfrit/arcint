# u8i4-prefill-price — `--paged-kv u8:i4` costs +7/+25/+72% prefill time with depth; measure the mechanism, then decide

## The defect, as measured

M8's asymmetric KV codec (u8 keys, i4 values) buys ~28% more admissible
context at parity decode: on the coder (16 GiB card, one lane, no offload)
auto-fit adopts 171,312 tokens against u8's 133,456 (+28.4%), and on the
24 GB dense agent (MTP on, prefix cache 8 GiB) 199,424 against 155,376
(+28.3%) — both within 0.1 pp of the 8.8-vs-11.3 KiB/token cost model,
greedy text byte-identical u8 vs u8:i4 at 16 tokens on both cards
(DESIGN §7.0.2y). What that gain does not carry is prefill speed: on the
coder, u8:i4 prefill against u8 at parity depth costs +7%, +25% and +72%
of prefill time at roughly 8.9k, 37.7k and 71.7k tokens (DESIGN §7.0.2aa;
also CHANGELOG.md's "Known defects", "Still open at 0.3.0", the M8-owed
entry). Decode is at parity across the same depths; only prefill moves.

By code reading (DESIGN §7.0.2aa), micro-SDPA declines the packing-class
mismatch (a symmetric fast path expects the same element type on both
sides) and prefill falls through to the mixed-stage opt attention path
instead, whose intermediate scratch (`exp_sums`/`max_logits`/`tmp_out`)
scales with depth. That is a reading of the source, not a measurement: no
profile or counter on the record attributes the +7/+25/+72% figures to a
named kernel or stage. Owed since 0.2.13, still owed on the M8 row of
`docs/milestone-0.3.0.md` as of the 0.3.0 release gate (DESIGN §7.0.2ai).

## Known against hypothesised

Known: the three prefill-time deltas at their three depths (DESIGN
§7.0.2aa); the context gain and its byte-identity (§7.0.2y); that decode
is at parity; that micro-SDPA selection is per device and precision, not
per chunk (ruled out in §7.0.2ab as the chunk-512 cell's explanation).
Hypothesised, not yet measured: that the opt-path scratch buffers named
above are the mechanism, and that their growth with depth traces to the
+7/+25/+72% figures — §7.0.2aa names this "by code reading", never closes
it with a profile. A related but distinct, already-tracked question not
to re-open here: the *fault* at deep u8:i4 prefill (`u8i4-deep-prefill-
fault`) and its scratch-charging fix bound VRAM, not time — the belt's
own price (~12% at short prompts, chunk 64/32 against 128, §7.0.2ab) is a
different, already-measured number.

Prior art, surveyed 2026-09-05 and recorded with sources, licenses and
Arc applicability: `research-kv-quantisation.md` (same directory). Its
"what transfers" section is the recon's starting point, not a substitute
for it.

## Gate

The mechanism measured: a profile or counter (`ARCINT_PROFILE_MWALL`-
class instrumentation, or a plugin-side counter if the profiler's own
depth blindness recurs — §7.0.2ab notes `ARCINT_PROFILE_PAST` cannot see
this class of cost) that attributes the extra prefill time to a named
kernel/stage, at two depths, on a named card. Then either a lever that
brings the +72% figure at ~71.7k tokens (16 GiB card, coder, u8:i4 vs u8,
§7.0.2aa) down to ≤ +25% at the same depth and card, byte-identity held
against u8, Prüfstand 10/10 on the coder artifact — a number that can
fail, not an unstated improvement — or the price documented on the M8 row
and in DESIGN as the format's cost, with the measurement that found it. A
verdict — no cheaper lever found — is a valid close. If a lever is found
but does not reach ≤ +25%, the design note fixes the actual target before
implementation and that number is the gate instead.

## Entry criteria

Met. The three-depth table is on the record (DESIGN §7.0.2aa); the coder
artifact and both cards are the ones every other 0.3.0/0.3.1 campaign
already exercises; `ARCINT_PROFILE_MWALL`/`ARCINT_PROFILE_CYCLE` already
exist for comparable depth-blind-profiler cases (§7.0.2ab, §7.0.2ag).

## Scope — in / out

In: measuring where the +7/+25/+72% goes at u8:i4 on the coder, at least
two of the three recorded depths; a lever if cheap enough within this
campaign's size; the M8 row's and DESIGN's closing entries.

Out: the deep-prefill fault and its VRAM-side charging (`u8i4-deep-
prefill-fault`); the direct-submission crash (`direct-submission-fault`);
the default KV precision (§7.0.3's "u8 is the paged default" stands
regardless); the M10 sub-4-bit expert-weight question (a separate kernel
milestone, DESIGN §7.0.2ah).

## Where it lives

`src/exec/fit.h` (`packed_values_prefill_scratch_bytes_ex`,
`packed_values_prefill_scratch_bytes_per_token_ex`, and the belt,
`prefill_chunk_cap_for_packed_values_ex` — these charge and cap the VRAM
term the *fault* campaign owns, and are the closest existing arcint-side
instrumentation of this scratch buffer's shape); `src/exec/backend_ov.cpp`
around the `ARCINT_PREFILL_CHUNK_CAP` env read (~line 3128) and the
`PAGED_ATTENTION_MAX_PARTITIONS` probe (~line 2580) for how chunk and
bound are chosen; `tests/test_fit.cpp`'s `packed_values_prefill_scratch_
bytes` test family (~line 1850 onward) — byte-shape tests only, none
measure time; DESIGN §7.0.2w (the codec and its two fixed bugs), §7.0.2y
(the context gain), §7.0.2aa (the price table and the code reading),
§7.0.3 (u8's own prefill cost curve against f16, the sibling precedent
this campaign's shape follows); `docs/milestone-0.3.0.md`'s M8 row and
CHANGELOG.md's "Known defects" paragraph for u8:i4 prefill price.

## Pipeline for this campaign

Recon (re-read §7.0.2aa and the opt-attention kernel source, confirm the
reading still holds) → design note if an instrument beyond `ARCINT_
PROFILE_MWALL`/`ARCINT_PROFILE_CYCLE` is needed → red-first: an instrument
that can fail to attribute before it is trusted to attribute correctly →
one card window at two depths → if a lever is found, red-first
implementation and a second window at Prüfstand 10/10 with byte-identity
→ review → DESIGN §7.0.2x record and a CHANGELOG line, either closing the
M8 owed item or restating it as a documented, measured cost.

## Invariants

Byte-identity between u8 and u8:i4 output holds regardless of any prefill
lever tried (DESIGN §7.0.2y's identity is the baseline to preserve, not
just re-check); no lever raises the u8:i4 auto-fit ceiling's cost model
beyond what §7.0.2y measured without re-verifying it; Prüfstand 10/10 on
the coder artifact.

## Status

- 2026-09-05 — opened from the 0.3.0 known-defect list; nothing started.
- 2026-09-05 — deepest sample so far, DESIGN §7.0.2aj: 24 GB card, coder,
  98,187 tokens, u8:i4 prefills at 144 t/s (78,867 tokens) and 121 t/s
  (98,187) against 1,018–1,026 t/s at u8 on the same card and prompt — but
  at the belt's chunk 128 against u8's 2,048, so the ratio prices the
  chunk cut as much as the format; §7.0.2aa's +72 % had both arms at the
  same pre-belt auto-fit chunk on the 16 GiB card. The campaign's first measurement must hold the chunk
  constant across the two precisions before attributing anything.
