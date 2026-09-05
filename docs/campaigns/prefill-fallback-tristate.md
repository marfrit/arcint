# prefill-fallback-tristate — patch 0018's overloaded `false` in the per-expert prefill loop becomes a three-way answer

## The defect, as measured

Found by review of patch 0018 (`contrib/packaging/marfrit-openvino/
patches/0018-moe-cpu-tier-static-partition.patch`, header section "Not
fixed here, found in the same investigation and reported instead of
guessed at", ~line 219), recorded at DESIGN §7.0.2ae. In the plugin's
per-expert prefill fallback (`moe_3gemm_swiglu_opt.cpp`, the loop near
line 2924 in the patched tree), the weight provider's
`on_load_expert_weights` returns `false` for two different reasons: the
offload tier is off entirely, or this expert is not resident under the
static partition. Its single caller reads `false` as "not resident" and
takes the host path. On a load with no offload at all, a caller reaching
that branch would cast a `ResidentExpertWeightProvider*` to the offload
provider type — undefined behaviour. Every path that reaches the caller
checks for offload first, except the top-level fallback that runs only
when a plugin flag is forced off, which arcint never does. No measurement
shows it firing; it is dormant in every configuration arcint drives, and
was left alone on purpose with the reasoning in the patch header.

## Known against hypothesised

Known: the two meanings, the one caller, the guard order, and that arcint
never forces the flag (patch header, traced at the time). Hypothesised:
nothing — the fix is mechanical; the only open question is whether a red
case can reach the branch at all without forcing the flag, which the
recon answers by reading the callers again on the current tree.

## Gate

A plugin unit test (the `moe_3gemm_static_partition.*` family the patch
already carries) that reaches the caller on a resident-only load with the
flag forced off and observes the three-way answer taken correctly — red on
the unpatched tree (a crash, a sanitizer report, or an assertion on the
cast) and green after; the four existing static-partition tests still
green; the tier reference cell and the coder offload cells of the
acceptance target byte-identical to their previous run; no rate moves
(the branch is off the hot path). Ships as a `+p5` patch level with the
packaging recipes updated in the same commit.

## Entry criteria

The patch header's trace re-read against the current patched tree; the
plugin's own test target building on the dev host (it did for patch
0018's four `moe_3gemm_static_partition.*` cases, plus the two
`execution_config` cases).

## Scope — in / out

In: the return type (an enum or optional with three states: tier off /
resident / not resident), the one caller, the unit test, the patch header
and `patches/README.md`, the `+p5` bump across the packaging recipes and
`docs/model_requirements.md`.
Out: the fallback path's performance (that is `static-partition-prefill`);
any change to which experts are resident.

## Where it lives

Patch 0018 (the hunk touching `expert_weight_providers.hpp/.cpp` and the
`~line 1358` / `~line 2924` sites it names); `contrib/packaging/
marfrit-openvino/{build-openvino.sh,build-deb.sh,debian/changelog,
patches/README.md}` for the patch level; the acceptance cells
`tier-reference-cell`, `coder-offload-1lane`, `coder-offload-2lane` as
the no-change proof.

## Pipeline for this campaign

Recon (one read of the two sites and the callers) → red-first unit test
in the plugin tree → the change → plugin rebuild on the dev host →
plugin unit tests → the three acceptance cells → review → commit as
patch 0019 or a revision of 0018 (the packaging README decides the
convention: 0018 is released in `+p4`, so a new patch 0019 on top) →
DESIGN §7.0.2x line, CHANGELOG line under "Dependencies".

## Invariants

No behaviour change on any configuration arcint drives: byte identity on
the three cells is the gate's teeth. §3.4 untouched.

## Status

- 2026-09-05 — opened from the 0.3.0 known-defect list; nothing started.
- 2026-09-05, closed (patch 0019, DESIGN §7.0.2ap). Recon on the staged
  tree confirmed the two meanings and found the second consequence: with
  the tier off the caller took the host path for weights that were on the
  device, not only the downcast. Fix: `ExpertWeightsSide` (no tier /
  device acquired or pinned / host tier); the loop runs the device path
  for both device answers; an assertion guards the downcast. Red first:
  a new plugin unit test (resident load, 40 tokens, both fast prefill
  paths off via the internal properties) failed on the 0018 tree with the
  misread provider trying to map a nonexistent weight file; green with
  0019, 21 of 21 MoE cases on each card, units running. Recipes bumped to
  `+p5` (package not built). Not run, on the operator's word for a quick
  functional test: the three acceptance cells; the branch is on none of
  their paths. The campaign's gate is therefore met on the unit-test half
  and waived on the cell half, recorded as such.
