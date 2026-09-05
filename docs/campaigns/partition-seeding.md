# partition-seeding — seed the static partition from a fixed calibration routing histogram so the pinned half is the hot half

## The defect, as measured

Not yet a measured defect: this campaign is conditional, per the 0.3.1
backlog row (`docs/milestone-0.3.0.md`) and DESIGN §7.0.2ai's own text —
it opens only "if decode variance survives" `static-partition-cold-start`
and `static-partition-prefill`. On the record today, decode does not show
a problem this lever would fix: at the M14 reference cell (16 GiB card,
35B int4, ratio 50, 8 GiB device pool, u8 KV, one lane, n_ctx 65,536, the
1,198-token prompt, 64 greedy tokens) the static partition already holds
the LRU-era record, 16.4 t/s on the second request against 15.0/15.5
(§7.0.2x) and tier OFF's 11.3–11.4 (§7.0.2ai) — with a partition whose
resident set is a `splitmix64(seed, layer_key, expert)` rank, independent
of actual routing frequency (patch 0018 header,
`static_partition.hpp`). The charter this document exists to hold open is
the hypothesis that a random-seed pin could sometimes miss the actually-hot
experts on a different artifact, corpus, or seed than the one measured so
far, not a defect this cell already shows.

## Known against hypothesised

Known: the ranking rule is a pure function of `(seed, layer_key, expert,
capacity)`, fixed once at `bind()`, with no term for how often an expert is
actually routed (patch 0018 header). Known separately: DESIGN §7.0.2ah (the
M10 re-scope) records that patch 0013's routing histogram
(`MOE_OTD_ROUTING_HIST`), run over the acceptance prompt alone, "left most
of 7,360 experts at 0–2 routings, no distribution to threshold
'rarely-routed' on" — i.e. the one corpus already exercised is too short to
tell "hot" from noise. Hypothesised: whether a frequency-informed partition
would move decode at all versus the current random seed, and whether any
gain would generalise past the one reference cell — neither is measured,
and per this campaign's own entry criteria neither should be built before
the histogram exists over a longer corpus.

Prior art, surveyed 2026-09-05 and recorded with URLs, licenses and
Arc applicability: `research-hybrid-expert-execution.md` (same directory). Its "what transfers"
section is the recon's starting point, not a substitute for it.

## Gate

Not decode merely at or above the historical record at the reference cell
— the random-seed partition already clears that bar (§7.0.2x/§7.0.2ai), so
that form of the gate passes trivially. Instead: a delta against the
random-seed partition measured in the same window (same card, flags,
prompt; both arms' second-request decode read side by side), with the
seeded arm's decode not below the random-seed arm's, and the decode
variance that motivated this campaign (entry criterion (1)'s own
measurement) reduced by a fraction set in the design note before this
campaign's implementation step — the number written there is the gate,
not a number invented after the fact. Byte-identical across processes and
requests (ON-vs-ON, E2) holds regardless, and the histogram and its
corpus are recorded — copied, with the record made falsifiable, from the
0.3.1 backlog row. The failable content also lives in the entry criteria
below: this campaign does not open at all unless entry criterion (1)
shows the variance survives.

## Entry criteria

Two, from the row, neither met yet: (1) `static-partition-cold-start` and
`static-partition-prefill` closed, or measured and found insufficient to
explain any remaining decode variance; (2) a routing histogram (patch
0013) over a corpus long enough that "hot" has a distribution to threshold
on — explicitly shared prework with the `sub4bit-vram-kernel` campaign
(DESIGN §7.0.2ah names the same corpus gap for both).

## Scope — in / out

In: a config- or artifact-embedded routing histogram derived from patch
0013's per-expert counters over a defined corpus; a seeding function that
replaces or augments the `splitmix64` rank with a frequency-informed one,
remaining a pure function of `(seed, histogram table)` and never of a
process's own request history; measuring decode at the reference cell
against the current random-seed baseline.

Out: the corpus-collection methodology itself beyond what
`sub4bit-vram-kernel` already needs; the cold-start and prefill fixes
(separate campaigns, and entry criteria here, not scope); the LRUCache /
`reserve_pinned` mechanics patch 0018 already built, which this campaign
reuses rather than changes.

## Where it lives

Patch 0013 (`contrib/packaging/marfrit-openvino/patches/
0013-moe-otd-routing-histogram.patch`) — `OffloadExpertWeightProvider::
routing_histogram()`/`layer_seq_id()`, `MOE_OTD_ROUTING_HIST`'s CSV dump
(`layer,expert,count`, one row per nonzero count, in layer-sequence order).
Patch 0018's `static_partition.hpp`
(`static_partition_resident_experts`, the `splitmix64(seed, layer_key,
expert)` rank this campaign's seeding function would replace or extend)
and its header's own account of `layer_key` (the layer's first OTD
weight-file offset, the same structural key patch 0013 already uses — the
two patches already share this identity, which is what makes a histogram
keyed the same way pluggable here). DESIGN §7.0.2ah for the M10 row's own
text on the histogram corpus's current insufficiency. The acceptance cell
`tier-reference-cell` (`tests/acceptance/cells/tier_reference.sh`) for the
decode gate (`decode-warm-2nd-on`) and the byte-identity checks
(`identity_group "tier ON"`, and E2 at the bottom of the script).

## Pipeline for this campaign

Recon (confirm the closing or verdict state of the two campaigns above;
read patch 0013's dump format and the M10 row's own corpus-insufficiency
finding in full) → design note (`docs/design-partition-seeding.md`, since
this changes the partition-selection algorithm, not a bounded fix) →
gather a longer-corpus routing histogram, as shared prework with
`sub4bit-vram-kernel` → red-first: a test on a synthetic skewed-routing
fixture asserting the seeded rank differs from, and prefers, the
histogram's own hot set over the random `splitmix64` rank → implement the
seeded rank as a pure function of `(seed, histogram table)` → one card
window: `tier-reference-cell` at the reference cell (decode-warm-2nd-on,
ON-vs-ON identity, E2) against the current random-seed baseline → review →
commit → DESIGN §7.0.2x record naming the histogram and its corpus,
CHANGELOG line, the backlog row's closing line — or, if decode does not
move at the reference cell, a verdict record stating the random seed
already captures the available win and the campaign closes without a
code change.

## Invariants

The seeding function stays a pure function of `(seed, histogram table)` —
never of the process's own request history (DESIGN §3.4, the same
invariant patch 0018 exists to hold). ON-vs-ON identity and E2 (§3.4 as
corrected: ON-vs-ON, OFF-vs-OFF and E2, not ON-vs-OFF — the Gate section
above) keep passing, not traded for a decode gain.

## Status

- 2026-09-05 — opened from the 0.3.1 backlog row; nothing started.
