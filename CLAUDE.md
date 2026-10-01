# arcint — session rules

Source of truth: README.md (scope), DESIGN.md (architecture, invariants,
milestones), llm.txt (machine summary). Read DESIGN.md before touching
anything. The invariants in §3.4/§3.8 and the gates in §5 are not negotiable
**by an agent**; the operator may amend them, and has (§3.4, 2026-10-01).

## Correctness is judged at the answer, not at the bit (operator, 2026-10-01)

The bar is the **result**: the right facts ("the capital of France is
Paris"), the needle found, the acceptance task passing, the answer quality
unchanged. Bit-identity of the **output** across configurations, placements,
timing or request history is not required anywhere. Integrity checks on
copied data (expert stores, slots, staged tables) are a different thing and
stay exact. A mechanism that changes rounding, placement or timing is
acceptable when the answers stay right.

**The answer-level bar**, used wherever an equivalence or quality gate is
asked for:
- the answers stay right: the facts, the needle, the task battery;
- the candidate's mean KL against the reference is at most **0.03 nats**
  worse than the baseline arm's, on the same card and the same window;
- argmax agreement drops by at most **1 point**.

The tolerances are a default and the operator may change them. The bar is
comparative because a card's own floor is no bar: on the A770 it is 0 (any
moved byte would fail), and on the B60 it is 0.136 nats (nothing would).

Deterministic replay (the same request sequence giving the same output) is a
debugging convenience: keep it when it is cheap, and drop it when it costs
measured speed.

## References first: expert engines over our own negatives

Expertise on inference engines lives in the references, and a faulty
implementation explained in enough jargon looks like a proof. So:

- **Follow what the expert engines do.** Strata (`~/src/Strata-ref`, written
  for this model on one GPU + RAM), FreeToken (`~/src/FreeToken-ref`),
  llama.cpp. Their source is the design reference; see
  `docs/campaigns/research-reference-audit.md`.
- **An arcint "negative" is settled only if it tested the reference's
  mechanism.** Before recording a verdict against a reference technique,
  write down how the build differs from the reference's (source memory,
  blocking vs not, caching, batch sizes, queues). A materially different
  build is not a verdict on the technique.
- **A conflict with an invariant goes to the operator.** When a reference
  mechanism conflicts with a DESIGN invariant or a rule in this file, do not
  drop the mechanism and do not rewrite the rule: record the conflict and
  ask, with the measured price of each side. A deviation from a reference
  that rests on an invariant must name the invariant.
- **Scoping a reference's mechanism out is a deviation from it**, so the
  same rule applies: name the reason, and ask. "Out: an LRU policy; pinning
  the bank" removed exactly the parts that make the reference mechanism pay.

## RTFM MANDATE — first, every turn

`AGENTS.md` carries this mandate in full and binds OpenCode and Pi; it binds
Claude Code identically. Before the first substantive tool call of any turn
that touches a design decision, a mechanism, a milestone or a campaign:

1. Read `docs/campaigns/README.md`, and the campaign document covering the
   work. Each is written to be sufficient on its own.
2. Check `docs/campaigns/research-*.md` — the prior art may already be
   surveyed.
3. If the task cites an external project, read its SOURCE, not only its
   paper (`~/src/Strata-ref`, `~/src/FreeToken-ref`, `~/src/ninfer` are
   checked out). Where
   paper and code disagree, the code wins.
4. State the EVIDENCE CLASS of every disposition you write — `paper`,
   `code`, or `measured-here`. A row without one is not a disposition.
5. A label is not evidence. `CONFIRMED`/`DEVIATION`/`UNSUPPORTED` records
   what someone concluded from the evidence they had; check what that was.
6. A negative that was never tried is not verified.
7. Touching a GPU? `docs/sop-card-window.md` first — sampler before the
   leg, card identity by PCI id (DRM numbering is INVERTED vs OpenVINO),
   zombie sweep by pid.

The cost of ignoring this is on the record in `AGENTS.md` and in
`docs/research-freetoken-code.md`: a 0.5.1 serving route measured at 7.06x
the device residency per layer and 623x the warm forward, whose root cause
sat in `docs/campaigns/` for ten days before it was built.

## This repository is public

Every commit lands where anyone can read it. That is an authoring rule, not a
release step:

- No host names, no addresses, no credentials — not in code comments, not in
  DESIGN.md, not in a commit message. "the dev host", "GPU.0", "the 24 GB card".
  Until 0.2.3 a separate tree was sanitised on every release; that step is gone,
  and with it the safety net that used to catch this.
- Anything operator-local — which machine, which unit manager, how to reach it —
  belongs in `CLAUDE.local.md`, which is git-ignored and read alongside this
  file. If you need infrastructure detail that is not there, ask rather than
  writing it down here.
- Uncommitted work is invisible and cannot be published on purpose or by
  accident. Commit before a release; a dirty tree at release time is how someone
  else's work-in-progress nearly went out once.

## Measurement discipline

- Quality gate: the acceptance task (a Lua CSV parser to RFC 4180, scored by
  executing the candidate code) at 10/10 for the coder artifact is the bar.
  Equivalence (cold against warm cache, one lane against two, paged against
  stateful, MTP on against off, adaptive expert placement against static)
  is judged by the answer-level bar above. Byte-exactness is not required
  (amended 2026-10-01; it was before). Byte cells in the suites stay as
  tripwires: a red one on a change that passes the answer-level bar is
  reported, not a veto.
- Gates are per phase. A change that measurably improves prefill or decode
  and does not regress the other beyond the run-to-run spread is adopted,
  provided it also passes the answer-level bar. A
  conjunctive "both must improve" gate discarded measured gains (0075's
  +6.4 % prefill).
- A test must be able to fail. Run the red case first; a check that was green
  before the change measures nothing.
- Measure at the endpoint that matters, and name the card, the depth, the KV
  precision and the configuration whenever a number moves. Every correction in
  this repository's history came from one of those being implied instead of
  stated.
- A claimed defect or explanation is accepted only with a measurement of its
  root cause. A mechanism that is narrated and not measured gets retracted on
  the record rather than edited away — see DESIGN §7.0.1.
- Profiles taken at a past-0 chunk overstate every node share: chunk k attends
  to everything before it, so the first chunk is the cheapest in any run. Two
  retracted headlines came from exactly that.
- Report what actually happened, including the parts that did not work.

## Model selection for agents / subagents

* Daily coding and implementation: Sonnet 5 for feature work, routine
  debugging, and most subagent execution.
* Complex architecture: Opus 5 for thorough code reviews, subtle bug detection
  and architectural trade-offs — high-value, low-volume work that benefits from
  deeper reasoning.
* Simple subagent tasks: Haiku 4.5 for bounded searches, summaries or
  mechanical file operations, to keep latency and cost down.
* Hardest agentic work: Fable 5 for code reviews before pushing.

## Language

Code and docs in English. Conversation follows the user.
