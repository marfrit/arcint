# AGENTS.md — arcint

Read by OpenCode and Pi. Claude Code reads `CLAUDE.md`, which carries the
same mandate. All three are bound by everything below.

`CLAUDE.md` is the project's rules (scope, publishing, measurement
discipline, model selection). This file adds one thing on top of it, and it
is not optional.

---

## THE RTFM MANDATE — before the first substantive tool call of a turn

Any turn that touches a design decision, a mechanism, a milestone, or a
campaign STARTS by reading the repository's own record. Not after the
measurement. Not after the implementation. First.

The documents named below — `docs/campaigns/`, the design notes, the
milestone and window records, DESIGN.md's full dated §7.0.x entries — live on
the development branch, `qfndev`. `main` carries the current state only; work
happens on `qfndev`.

**1. Read the campaign index.** `docs/campaigns/README.md` (on `qfndev`) lists every open
defect and lever, each with a charter, a gate, and a status. If a campaign
covers the work, read that campaign document before touching code. Each one
is written to be sufficient on its own — that is the directory's stated rule.

**2. Check whether the prior art was already surveyed.**
`docs/campaigns/research-*.md` exist for exactly that. Their "what transfers"
sections are the starting point.

**3. If the task cites an external project, read its SOURCE, not only its
paper.** Checked out here: `~/src/FreeToken-ref`, `~/src/ninfer`. A paper
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

---

## Why this file exists

2026-09-15. Five days of 0.5.1 built a segmented serving route whose expert
bodies are whole-tensor u8 `Parameter`s unpacked to f32 in-graph. Measured on
the B60: **7.06× the device residency per layer and 623× the warm forward**
against the same graph with the bodies as `Constant`s; no 12-layer segment
compiles at all.

The cause was in this repository the whole time:

- The MoE fusion matcher requires `u4` **Constants**
  (DESIGN §7.0.2ah, the full entry on `qfndev`). Leaving Constant-land leaves the fusion, and the fusion
  is what applies the **routing** — `window-051.md` (on `qfndev`) §2 says so in the design's
  own words: "every expert computes for every token".
- Flash-Next activates **10 of 512** experts per token.
  `design-qwen-flash-next.md`:84 dispositioned `num_experts_per_tok` as
  "**not read**". That one table cell is where the design stopped being
  FreeToken-shaped; it is 51.2× the required work.
- `docs/campaigns/research-hybrid-expert-execution.md` (on `qfndev`), dated **2026-09-05**,
  had already scoped the fix correctly: "the missing piece is the *split* …
  **That is a plugin change, not an engine change** — none of the surveyed
  systems replaced their serving engine, they added a kernel-dispatch
  branch." 0.5.1 replaced the serving engine.
- The two campaigns that own the real fix — `sub4bit-vram-kernel` (in-kernel
  dequant + a fusion matcher for the new element type) and
  `static-partition-prefill` (split the batch by residency; `moe_cpu_expert`
  already exists) — were both opened 2026-09-05 and both still read
  **"nothing started"**.
- `docs/research-freetoken.md` carries **10 section verdicts**, every one
  from the paper text alone. One is disproven by the code: it declares the
  n-gram/PLE table "Not in FreeToken … arcint-original work", while
  `~/src/FreeToken-ref` ships `models/qwen4_exp/ple_disk.py` with
  `ple_backend = "disk"` as the **default**. arcint pins the same table as
  26.82 GiB of USM host for the life of the process.

No instruction ever forbade checking any of this. There is no standing order
in `CLAUDE.md`, `CLAUDE.local.md`, the fleet `CLAUDE.md`, or any agent
definition against questioning the operator or the design premises — that was
searched. The contract required the opposite. The documents were simply never
opened, across roughly ten dispositions and several sessions, by more than one
agent.

Reading them costs minutes. Not reading them cost a 60 GB artifact, an 82
minute fill, four watchdog-killed card legs, a downed dev host, and a
mechanism that cannot work.

**RTFM first. Every turn.**
