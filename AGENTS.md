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

2026-10-01, the second time: the FreeToken-style LRU expert cache was built,
ruled "a violation of §3.4 as written", and replaced by a static partition.
The price was never put to the operator: a 36 % GPU hit rate and 308
CPU-tier experts per token, against Strata's ~0.72 hit rate (paper §3.4,
which leaves ~134 of 480 routed experts per token for the CPU before its
PCIe share). The operator: "I requested that so many times." The same audit
found three more recorded negatives that had tested a different mechanism
from the reference's: hybrid-expert-fetch (pageable memory, nothing cached),
prefill-expert-streaming v1 (staging, one queue, small chunks) and the 0075
hand-off (half of the wait removed). It also found one premise that was never
tried, "Flash-Next has no MTP head". Rules 8 and 9 above come from that.

The first time:

2026-09-15. Five days of 0.5.1 built a segmented serving route whose expert
bodies are whole-tensor u8 `Parameter`s unpacked to f32 in-graph. Measured on
the B60: **7.06× the device residency per layer and 623× the warm forward**
against the same graph with the bodies as `Constant`s; no 12-layer segment
compiles at all.

The cause was in this repository the whole time:

- The MoE fusion matcher requires `u4` **Constants**
  (`DESIGN.md`:4489). Leaving Constant-land leaves the fusion, and the fusion
  is what applies the **routing** — `window-051.md` §2 says so in the design's
  own words: "every expert computes for every token".
- Flash-Next activates **10 of 512** experts per token.
  `design-qwen-flash-next.md`:84 dispositioned `num_experts_per_tok` as
  "**not read**". That one table cell is where the design stopped being
  FreeToken-shaped; it is 51.2× the required work.
- `docs/campaigns/research-hybrid-expert-execution.md`, dated **2026-09-05**,
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
