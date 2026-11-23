# Contributing to arcint

## The three branches

| Branch | What it holds | How work arrives |
|---|---|---|
| `main` | Tested, released code. Every release tag is on `main`. | `development` -> `main`, only after the full release gate and an outside review. |
| `development` | Features that are finished and card-proven but have not passed a release gate. | Clean, curated commits, one per feature. Each commit names its card evidence, in the message or in the CHANGELOG's "Unreleased" section. |
| `qfndev` | The commit collector: raw work, working notes, experiments and the campaign documents (`docs/campaigns`). | Directly, as work happens. |

`development` starts from the latest release tag. A feature moves from `qfndev` to `development` only when all of these hold:

- it is measured on the cards it serves on, not just on a CPU host. The CPU cannot run the OpenCL kernels, so a CPU-only pass does not count;
- it meets the answer-level bar (right facts, needle and task battery; mean KL to the CPU reference at most 0.03 nats worse than the baseline; argmax agreement at most 1 point lower), including at micro-batch 1 where the change touches a decode path;
- its tests can fail: a red case was run, and it failed;
- the patch builds with the recipe from the pinned llama.cpp plus the series, and the unit tests pass.

Working notes, campaign documents and experiments stay on `qfndev`. A `development` commit refers to a campaign document as "(on the `qfndev` branch)".

## Releases

A release takes `development` as it stands, runs the full release gate (`DESIGN.md` §5 and the CHANGELOG's gate paragraphs), gets an outside review, and merges into `main` with a tag. The CHANGELOG's "Unreleased" section becomes the release's entry.

## Patches to llama.cpp

`contrib/llama.cpp/patches` is applied in C-locale order onto the pinned llama.cpp commit. Each patch starts with a line `NNNN: <one sentence>`, then an empty line, then a plain diff (no mail header). Every patch has a section in `contrib/llama.cpp/README.md`.

Parallel work takes patch numbers from disjoint ranges, assigned before the work starts.

## Work done without a card

Some contributors cannot run the cards. For them:

- anything already proven on a card is frozen. It changes only together with the card evidence the change asks for;
- a suspected bug in a frozen piece is written up as a question, with the evidence (for example the graph-dump lines and the expected against the observed tensor), and the code is left alone.

## Public repository

Commits, patches and documents never contain host names, user names, addresses, local paths or session identifiers.
