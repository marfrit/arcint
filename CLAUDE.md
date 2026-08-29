# arcint — session rules

Source of truth: README.md (scope), DESIGN.md (architecture, invariants,
milestones), llm.txt (machine summary). Read DESIGN.md before touching
anything; the invariants in §3.4/§3.8 and the gates in §5 are not negotiable.

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
  Equivalence — cold against warm cache, one lane against two, paged against
  stateful, MTP on against off — must be byte-exact.
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
