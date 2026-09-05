# mtp-cycle-wall — on the dense agent, MTP never beats plain decoding at depth; cut the cycle wall or record the verdict as final

## The defect, as measured

DESIGN §7.0.2ag (dense 27B agent, 24 GB card, u8 KV, one lane, auto-fit
depth, prefix cache off, 400 greedy tokens, one process per arm/depth).
At 8.9k tokens: plain 15.3 t/s; MTP 7.3 t/s (85% accepted, cycle 253 ms);
DFlash 6.0 t/s (73% accepted, 287 ms). At 76k (77,134 prompt tokens):
plain unchanged at 15.3 t/s; MTP 1.1 t/s, 0/372 accepted, 902 ms; DFlash
5.1 t/s, 0/2,604 accepted, 198 ms. Greedy output byte-identical across
all three arms at both depths (§3.4 held) — a cost/acceptance problem,
not a correctness one.

Two mechanisms, found and fixed, both §7.0.2ag. (1) The MTP layer's own
KV state (4 heads × 256 head_dim × 2 K+V × 4 bytes f32 = 8 KiB/token),
unpaged and primed over the whole prompt, was never charged against the
reservation — at 77,134 tokens it costs 0.59 GiB, past the arm's own
logged 22.46 GiB ceiling, producing the bimodal verify cost at 76k (min
441 ms, p95 1,622 ms). Fixed: `kMtpStateBytesPerToken = 8192`
(`src/exec/fit.h:165`), folded into a local copy of the per-token KV rate
at the reservation line (`src/exec/backend_ov.cpp:2901-2913`, `:3852`).
(2) Both drafters compute rotary in-graph (`inv_freq[0] = 1.0`) under the
plugin's default f16 precision hint, overflowing at 65,504 — a bisected
cliff (60,333 tokens 78% accepted; 70,403 tokens 0%). Fixed: each
drafter's rotary subgraph (16 nodes MTP, 6 DFlash) marked
`ov::disable_conversion`, on by default (`ARCINT_DRAFT_ROPE_F16=1`
reverts it).

Re-measured with both fixes at 77,134 tokens (`n_ctx` clamps to 127,536,
was 155,488 unclamped): MTP 90.8% accepted (177/195), 4.9 t/s, cycle
≈ 390 ms — still below plain, since MTP accepts at most 2 tokens/cycle
against a ≈130 ms break-even; DFlash 40.6% accepted (276/679), 18.8 t/s,
≈4.4 tokens/cycle — above plain, passing the M11 row's original gate
(>3.13 tokens/verify cycle, byte-exact — DFlash's own break-even, §7.0.2z
for the number, §7.0.2ag for the ruling that it is DFlash's break-even
and must not be applied to MTP unchanged). Both byte-identical to
plain at both depths. The M11 row closed at the 0.3.0 release gate
(DESIGN §7.0.2ai, `docs/milestone-0.3.0.md` M11 row, 2026-09-05): the
equivalence suite's MTP section passes on the `+p4` package (identical
to plain greedy, warm equals cold with a real prefix-cache hit, 68.4%
acceptance at 8k — the depth per the M11 row, `docs/milestone-0.3.0.md`;
§7.0.2ai itself names the figure without the depth), but the 76k cell
was **not** re-run on the release binary ("nothing since touched it
except gating the MTP-state term ... inert in the served
configuration") — that re-run is this campaign's open item.

## Known against hypothesised

Known: both mechanisms, the fixed constants, the verification-window
numbers, the M11 shallow gate passing on `+p4` (§7.0.2ai). Hypothesised:
nothing on the mechanism — §7.0.2ag's own caveat was discharged by that
window. Open: whether the 76k profile reproduces on `+p4` itself, and
whether any cycle stage can be cut below ≈130 ms without a new kernel.

Prior art, surveyed 2026-09-05 and recorded with URLs, licenses and
Arc applicability: `research-speculative-cycle.md` (same directory). Its "what transfers"
section is the recon's starting point, not a substitute for it.

## Gate

Either close is valid: (1) **Cycle cut** — the MTP cycle wall at depth
below its own break-even (~130 ms at 76k, §7.0.2ag), a step profile
(`ARCINT_PROFILE_CYCLE`) naming the cut stage and its size, byte-identity
of MTP-on vs plain held, the equivalence suite's MTP section and the
`agent-dense` acceptance cell ("MTP identity ... and MTP acceptance
> 10%") green. (2) **Verdict as final** — the M11 verdict (DFlash with a
depth gate, or plain below the crossover; MTP off) restated as final on
the M11 row of `docs/milestone-0.3.0.md` and in CHANGELOG.md, citing
DESIGN §7.0.2aa's close ("cause not established... serve deep contexts
without a drafter until it is") as superseded by §7.0.2ag. §7.0.2aa
itself stays as written — a dated record, not edited, per §7.0.1's
retraction culture (later sections supersede, earlier ones are not
edited). `docs/model_requirements.md` §4 and README.md lines 116–133
already carry the corrected reading (§7.0.2ag's 90.8% at 77k, ~390 vs
~130 ms, and point at this campaign); no update to either is owed by
this close.

## Entry criteria

Met. `ARCINT_PROFILE_CYCLE`, `kMtpStateBytesPerToken` and the rotary
marking are on `main` (§7.0.2ag); the `+p4` release package exists
(CHANGELOG.md 0.3.0 known-defects; §7.0.2ai).

## Scope — in / out

In: the step profile at 8.9k/76k on the release binary, §7.0.2ag's
protocol; if cutting, the stage's mechanism, a red-first regression test
where `fit.h`/the reservation line is touched, the two acceptance cells
at that depth; if recording the verdict instead, the M11 row's closing
line confirmed and a DESIGN §7.0.2x record citing §7.0.2aa as superseded
by §7.0.2ag.

Out: DFlash's own tuning (§7.0.2z closed the tree path; the adapter is an
operator decision); any `--paged-kv`/offload-tier change; the 35B MoE's
reconstructed MTP head (§7.0.2o, a different artifact, a documented 30%
loss).

## Where it lives

`src/exec/fit.h` (`kMtpStateBytesPerToken` line 165, `mtp_state_bytes()`
line 172); `src/exec/backend_ov.cpp` (reservation site 2901–2913, 3852;
`ARCINT_PROFILE_CYCLE` near 2277; `ARCINT_DRAFT_F32`/
`ARCINT_DRAFT_ROPE_F16`, 2764–2867); the M11 test trio in
`tests/test_fit.cpp` from line 2920; `tests/equivalence/run.sh`'s MTP
section; `tests/acceptance/cells.json`'s `agent-dense` cell; DESIGN
§7.0.2ag/§7.0.2z/§7.0.2aa/§7.0.2ai; `docs/milestone-0.3.0.md`'s M11 row;
CHANGELOG.md's "Known defects in 0.2.13"; README.md lines 99–133, 432;
`docs/model_requirements.md` §4.

## Pipeline for this campaign

Recon (confirm release-binary identity against the verification build) →
one card window: step profile at 8.9k/76k, release bits, all three
arms → decide cut vs verdict against ~130 ms → if cutting, red-first test
plus the two acceptance cells; if recording, the M11 row and CHANGELOG
close → review → DESIGN §7.0.2x record, CHANGELOG line, the M11 row
updated if the wording changes.

## Invariants

§3.4: every arm at every depth on record is byte-identical to plain; a
cut that changes served text does not close this campaign. The M11 gate
keeps MTP's own break-even (a cycle under ~130 ms, at most 2 accepted
tokens/cycle) distinct from DFlash's (3.13 tokens/verify cycle) —
conflating them was the error §7.0.2ag already corrected once.

## Status

- 2026-09-05 — opened from the 0.3.0 known-defect list / the 0.3.1 window;
  nothing started.
