# prefill-fallback-tristate — the per-expert prefill loop's weight answer is three-way

**Closed 2026-09-05.** Patch 0019 replaces patch 0018's overloaded `false`
with `ExpertWeightsSide` (no tier / device / host tier); the loop runs the
device path for both device answers and an assertion guards the downcast. A
plugin unit test (resident load, both fast prefill paths off) is red on the
0018 tree and green with 0019 (`measured-here`). Packaged from `+p5`;
DESIGN §7.0.2ap.

Full history: `git show b0447b8:docs/campaigns/prefill-fallback-tristate.md`.
