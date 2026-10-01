# AGENTS.md — arcint

Read by OpenCode and Pi. Claude Code reads `CLAUDE.md`, which carries the
same mandate. All three are bound by everything below.

`CLAUDE.md` is the project's rules (scope, publishing, measurement
discipline, model selection). This file adds the mandate below on top of it,
and it is not optional.

---

## THE RTFM MANDATE — before the first substantive tool call of a turn

Any turn that touches a design decision, a mechanism, a milestone, or a
campaign STARTS by reading the repository's own record. Not after the
measurement. Not after the implementation. First.

**1. Read the campaign index.** `docs/campaigns/README.md` lists every open
defect and lever, each with a charter, a gate, and a status. If a campaign
covers the work, read that campaign document before touching code. Each one
is written to be sufficient on its own — that is the directory's stated rule.

**2. Check whether the prior art was already surveyed.**
`docs/campaigns/research-*.md` exist for exactly that. Their "what transfers"
sections are the starting point.

**3. If the task cites an external project, read its SOURCE, not only its
paper.** Checked out here: `~/src/Strata-ref`, `~/src/FreeToken-ref`,
`~/src/ninfer`. A paper
describes intent; the code is what the project does. Where they disagree,
the code wins.

**4. State the EVIDENCE CLASS of every disposition you write** — `paper`,
`code`, or `measured-here`. A row without one is not a disposition.

**5. A label is not evidence.** `CONFIRMED` / `DEVIATION` / `UNSUPPORTED` in
an existing document records what someone concluded *from the evidence they
had*. Before you build on one, check what that evidence was. Labels travel;
their basis does not.

**6. A negative that was never tried is not verified.** Already the rule in
`.claude/agents/fix-implementer.md`; it is general. "Not in the paper", "not
on the host", "no code path needs it" are claims that require an attempt.

**7. Touching a GPU? `docs/sop-card-window.md` first.** Sampler before the
leg, card identity by PCI id (DRM numbering is INVERTED vs OpenVINO), zombie
sweep by pid. Every rule there has a dated incident behind it.

**8. Expert engines over our own negatives.** The references are Strata
(`~/src/Strata-ref`, written for Qwen3.8-Flash-Next on one GPU + RAM),
FreeToken (`~/src/FreeToken-ref`) and llama.cpp; read
`docs/campaigns/research-reference-audit.md` first.
- An arcint negative counts against a reference technique only if it tested
  the same mechanism. Write down how the build differed before recording a
  verdict.
- A conflict between a reference mechanism and a DESIGN invariant or a
  `CLAUDE.md` rule is the **operator's** decision, with the measured price of
  each side. Never resolve it by dropping the mechanism, and never by
  rewriting the rule.

**9. Correctness is judged at the answer, not at the bit** (operator,
2026-10-01): the answers stay right (facts, needle, task battery), mean KL
against the reference no more than 0.03 nats worse than the baseline arm's on
the same card and window, and argmax agreement down by at most 1 point.
Bit-identical output across configurations or history is not required;
integrity checks on copied data stay exact. Details in `CLAUDE.md`.

---

## Why this file exists

**2026-09-15, documents not read.** 0.5.1 spent five days building a
segmented serving route whose expert bodies were whole-tensor `Parameter`s
outside the MoE fusion, so every expert computed for every token: on the B60
it measured 7.06× the device residency per layer and 623× the warm forward.
The correct mechanism (route only the selected experts, split the batch by
residency, as FreeToken's source does) was already scoped in
`docs/campaigns/`, and the reference's source had never been read. Rules 1–6
follow: read the record and the reference's code before the first tool call.

**2026-10-01, a reference mechanism dropped under an invariant.** The
FreeToken-style LRU expert cache was built, ruled a violation of DESIGN §3.4,
and replaced by a static partition without the price being put to the
operator: a 36 % GPU hit rate (`measured-here`) against Strata's ~0.72
(`paper`). The same audit found three more recorded negatives that had
tested a different mechanism from the reference's, and one premise that was
never tried (the Flash-Next MTP head). Rules 8–9 follow: follow the expert
engines, settle a negative only against the reference's own mechanism, take
every conflict between a reference mechanism and an invariant to the
operator with the measured price of each side, and judge correctness at the
answer.

**RTFM first. Every turn.**
