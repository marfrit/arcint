# kquant-host-storage — the M14 host tier computing K-quant blocks natively, VRAM untouched

## The defect, as measured

Not a defect: an extension the M14 row named but did not build. M14's host
compute tier is scoped on two decisions of record (DESIGN §7.0.2x): the CPU
kernels are written for the plugin's *actual* grouped-int4 layout — no
repack, no GGUF/M10 coupling — and llama.cpp on CPU is not the bar. That
decision is enforced in the shipped code, not just stated: the decode-path
wiring in patch 0012 refuses anything but 4-bit compressed expert weights
("MOE_CPU_TIER supports 4-bit compressed expert weights only (u4/i4); got
...", `0012-moe-cpu-tier-decode-split.patch`). So today, an offloaded expert
that misses the device slot LRU and takes the host path is computed, and
stored in host memory, at int4 — the same resident format as the device
tier, one byte count, not sub-4-bit anywhere off the card. ik_llama.cpp's
iqk techniques are credited rather than vendored in M14 specifically
*because* "their headline speed is on K-quant blocks the plugin does not
store" (§7.0.2x) — this campaign is what would make that sentence false.

The reference cell is M14's own (16 GiB card, 35B int4, ratio 50, 8 GiB
device pool, u8 KV, one lane, n_ctx 65,536), most recently re-measured
under patch 0018's F2 static partition at the 0.3.0 release gate (DESIGN
§7.0.2ai): tier OFF decode 11.3–11.4 t/s on the second request, tier ON
16.4 t/s on the second request (15.4 already on the first request of the
warm-cache process) — the LRU-era record (15.0/15.5, §7.0.2x) holds. Tier
ON is byte-identical to itself across processes and requests, and E2
(history-independence after a prefix-cache restore) passes; that it also
matched tier OFF in that window was a property of the sample, not a
standing gate — the first 0.3.1 acceptance run found every tier-ON output
different from tier OFF while ON stayed identical to itself and E2 held
(`test-ladder-close.md`; device f16 and host f32 are not bit-equal, DESIGN
§7.0.2ae). Prefill loses 3× under the static partition
(26.6–26.7 vs 87.4–87.6 t/s, `grouped_fallbacks` one per layer) — a
separate, already-backlogged defect (`static-partition-prefill`), out of
scope here and not this campaign's gate.

## Known against hypothesised

Known: the two M14 decisions of record and the type guard enforcing them
(patch 0012); the K-quant byte counts per 256-weight block — Q3_K 110 B
(3.4375 bpw), IQ3_XXS 98 B (3.0625 bpw), Q2_K 84 B (2.625 bpw), the
i-quants indexing a compile-time grid an importer would vendor under MIT
(§7.0.2ah); that the plugin's mmap weight accessor and thread pool already
exist for the grouped-int4 layout (patch 0011) and would need a second,
K-quant-native decode path beside them, not a replacement; the M14
reference-cell numbers above, under the static partition, with the
corrected equivalence gate (test-ladder-close: the tier-reference cell's
runner originally gated ON == OFF and failed on valid numbers, corrected
the same day to ON-vs-ON, OFF-vs-OFF and E2 — the form this campaign's
gate uses, not ON-vs-OFF).

Hypothesised: everything about the win itself. Whether a host kernel that
decodes Q3_K/IQ3-class blocks in place beats the grouped-int4 host kernel
at 270/305 µs per expert (§7.0.2x) is the open question this campaign
exists to answer, "win or lose" by its own gate's wording — no K-quant
host kernel exists yet to measure. Also hypothesised: which of `u3`
group-quant or K-quant blocks is even the resident format, which this
campaign does not decide (see Entry criteria).

Prior art, surveyed 2026-09-05 and recorded with URLs, licenses and
Arc applicability: `research-hybrid-expert-execution.md` and `research-sub4bit-weights.md` (same directory). Its "what transfers"
section is the recon's starting point, not a substitute for it.

## Gate

The pass/fail clause is equivalence alone: byte-exact tier ON vs itself
across processes and E2 at the M14 reference cell (§3.4 as corrected:
ON-vs-ON and E2, not ON-vs-OFF). The host-pool byte count and the decode
number against the grouped-int4 host kernel (16 GiB card, 35B int4, ratio
50, 8 GiB device pool, u8 KV, one lane, n_ctx 65,536) are reported, win or
lose, per the backlog row — not a pass/fail threshold themselves
(`docs/milestone-0.3.0.md` backlog row 2; DESIGN §7.0.2ah/§7.0.2x/§7.0.2ai).

## Entry criteria

Not met; both block starting the kernel work. (1) `sub4bit-vram-kernel`'s
resident-format measurement — it decides whether K-quant blocks are the
format at all, or whether `u3` group-quant wins on the one expert layer
that campaign measures; unrun there, so unmet here. (2) The `iq3xxs_grid`
and sibling constants vendored under the `THIRD_PARTY.md` convention
(MIT, ik_llama.cpp lineage credited, licenses unchanged) — checked: not
present. `THIRD_PARTY.md` today lists only cpp-httplib, nlohmann/json and
minja, all MIT/vendored single-header; no i-quant grid constants are in
the tree, so this is a real, unstarted step, not a formality.

## Scope — in / out

In: the host kernel computing Q3_K/IQ3-class blocks natively for the
offload path (no dequant-to-int4 at load for the host pool); lifting
patch 0012's u4/i4-only guard for the new path without weakening it for
the existing one; the host-pool byte count and decode measurement at the
reference cell; vendoring the grid constants; re-verifying the corrected
equivalence gate on the new path.

Out: `sub4bit-vram-kernel`'s GPU kernel and VRAM claim — a different
currency (disk/host-pool bytes here, VRAM there) and a different gate,
separated by DESIGN §7.0.2ah on purpose; the static partition's prefill
and cold-start costs (`static-partition-prefill`, `static-partition-cold-
start`); the per-expert bpw map's artifact format (decided by the other
campaign); a full-CPU fallback engine (dropped from M14's scope already,
`docs/milestone-0.3.0.md` M14 row, "2026-09-02 — scoped, gate redefined").

## Where it lives

DESIGN §7.0.2x (M14's scoping, the two decisions of record, the iqk byte
counts, the reason K-quant is credited not vendored today), §7.0.2ai (the
M14 reference-cell numbers under the static partition), §7.0.2ah (K-quant
byte counts, the separation from `sub4bit-vram-kernel`); `docs/milestone-
0.3.0.md` M14 row and backlog row 2; `contrib/packaging/marfrit-openvino/
patches/0011-moe-cpu-expert-kernel.patch` (the AVX2/scalar kernels for the
grouped-int4 layout, mmap accessor, thread pool, `MOE_CPU_TIER` property);
`patches/0012-moe-cpu-tier-decode-split.patch` (the decode-path wiring and
its explicit 4-bit-only guard, the site a K-quant path extends);
`patches/0018-moe-cpu-tier-static-partition.patch` (F2, the partition the
reference cell now runs under); `THIRD_PARTY.md` (the vendoring
convention; `iq3xxs_grid` not yet listed); `test-ladder-close.md` (the
corrected tier-cell gate form this campaign's gate quotes).

## Pipeline for this campaign

Wait on `sub4bit-vram-kernel`'s resident-format decision (blocking entry
criterion — this campaign does not start the kernel work before it lands)
→ vendor `iq3xxs_grid` and sibling constants under `THIRD_PARTY.md` (MIT,
credited) → red-first: a host-kernel unit test that decodes a Q3_K/IQ3-
class block natively fails on the pre-change tree → implement the native
block-decode kernel for the host tier, lifting patch 0012's guard for the
new path only → equivalence re-verification at the corrected gate
(ON-vs-ON, OFF-vs-OFF, E2) → one card window at the M14 reference cell:
host-pool byte count, decode number against the grouped-int4 host kernel,
win or lose reported either way → review before commit → patch revision
and packaging-recipe bump → DESIGN §7.0.2x record, CHANGELOG line, the
milestone row's closing line.

## Invariants

DESIGN §3.4 in its corrected form (ON-vs-ON, OFF-vs-OFF and E2 — not
ON-vs-OFF, per `test-ladder-close`'s fix to the tier-reference-cell
runner): a K-quant host kernel is exactly the kind of arithmetic change
(f32 host vs f16 device, now also a different bit layout) that made the
LRU-era M14 tier history-sensitive before patch 0018's F2 static partition
fixed it (§7.0.2ae); this campaign runs under F2 and must not reopen that.
Ground rule 2 (fusion-impact profile) applies if the new path touches the
fused decode kernel's skip-mask, not just the host excursion. The two M14
decisions of record stand: grouped-int4 stays the primary layout, K-quant
is an added native path, not a repack of it; llama.cpp on CPU is still not
the bar for this campaign's win/lose call.

## Status

- 2026-09-05 — opened from the 0.3.1 backlog; nothing started.
