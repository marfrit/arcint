<!-- 0.3.1, lead item. Produced by the campaign's design pass on 2026-09-05
against docs/milestone-0.3.0.md's backlog row "Unit tests and acceptance
tests differentiated" and DESIGN §5.1; accepted for implementation in the
increments of §6. The 0.3.0 release gate (DESIGN §7.0.2ai) was run from
prose and ad-hoc window scripts; this note exists so the next one is run
from an enumeration. -->

# Design note: differentiating unit and acceptance tests (0.3.1)

Spec: `docs/milestone-0.3.0.md:91` — device-free `ctest` is the unit gate, a second target is the card-requiring acceptance set with its cells enumerated in one place, and the release checklist is generated from that enumeration. Ladder: `DESIGN.md:1399-1418` (§5.1). Today's assembly: `CMakeLists.txt:167-216`.

## 0. What is actually true today

- `ctest` registers three tests: `unit` (`CMakeLists.txt:202`), `roundtrip` (`:204`), `stress` (`:212`). All three are device-free; §5.1 puts all three in the Unit class.
- Only `tests/test_rope_precision.cpp` is compile-gated on `ARCINT_OPENVINO` (`:9`, `:213`); it holds 4 `TEST()` blocks, exactly the 413−409 of `DESIGN.md:1273`. It builds a synthetic `ov::Model` and never touches `ov::Core` — it needs OV *headers*, not a card. `tests/test_config.cpp:47-58` switches branch on the same macro. Nothing else in the unit binary is OpenVINO- or card-dependent.
- `contrib/packaging/arcint/build-deb.sh:82` runs `ctest -R unit`, which selects the single test named `unit` — `roundtrip` and `stress` are silently outside the package gate, and the comment above it ("GPU-dependent suites skip themselves when no card is present") describes nothing that exists.
- The motivating example, `tests/test_affinity.cpp`, pinned to CPU 0 by literal until `eae32d1` and failed in a container whose cpuset excludes it — a unit case asserting a host property.
- Prose drifts: `llm.txt:790-792` says "306 unit cases … 48 curl checks" against DESIGN's 409/64. This is the argument for generating, not re-reading.

## 1. Mechanism: labels, plus a configure-time guard

**Chosen.** `LABELS unit` on `unit`/`roundtrip`/`stress`; acceptance cells registered only inside `if(ARCINT_ACCEPTANCE)` (new option, default OFF, refused unless `ARCINT_OPENVINO` is ON), each with `LABELS acceptance`. Bare `ctest` in the default build therefore runs exactly the unit set — not by convention but because nothing else is registered. The labels make the partition *assertable* (`ctest -N -L acceptance` must list zero tests in a default build).

**Rejected:** labels alone — bare `ctest` would then run acceptance cells on a card-less machine, which is the gate's own counter-example. Separate targets/directories — duplicates ctest's timeouts and reporting and gives the recipe nothing to select on. A bare CMake option without labels — no way to assert the partition.

**Recipe change:** `build-deb.sh:82` becomes `ctest --output-on-failure --no-tests=error -L unit`, and its comment is corrected. This widens the package gate from one test to three, which is the honest reading of §5.1's Unit row; cost and risk in §7.

## 2. Card-requiring cells: parameters and skip semantics

**Parameters** come from CMake cache variables resolved at configure time (`ARCINT_ACCEPTANCE_MODEL_ROOT`, `ARCINT_ACCEPTANCE_DEVICE_LARGE`, `ARCINT_ACCEPTANCE_DEVICE_SMALL`) into a generated run manifest that also carries `$<TARGET_FILE:arcint>`. Environment is rejected as the primary channel: it does not appear in the ctest log, and §7.0.2ai already shows an env term (`ARCINT_MOE_DEVICE_POOL_BYTES` at 8 GiB per offload arm) that has to be *stated* per cell. A checked-in config file alone is rejected because it cannot know the build's binary path.

**Skip semantics.** ctest's `SKIP_RETURN_CODE` marks a skip as not-failed, so `ctest -L acceptance` on a card-less machine would go **green with every cell skipped**. That is the failure this row exists to close. So the acceptance target is `tests/acceptance/run.py --all`, not bare ctest: a cell exits 0 / 1 / 77(skip, with a printed reason), and the runner exits non-zero unless every 77 is named on the command line as `--allow-skip <cell>`. Symmetrically it exits non-zero when `--allow-skip` names a cell that ran, or a name absent from the enumeration — so the allow-list cannot rot. The per-cell ctest tests call the runner, so its exit code is ctest's.

The suites carry the same defect internally: `tests/equivalence/run.sh:225` prints "no MTP head in $MODEL; skipping the MTP gates" and still exits 0 at `:267`. Increment 2 makes it print a machine-readable `ACCEPTANCE-SKIP <check> <reason>` line that the runner promotes to a named skip -- not a bare `SKIP`, because `tests/harness.h`'s unit-test harness already prints its own `SKIP %s: %s` lines (`tests/test_main.cpp`), and the sanitizers cell tails its ctest log to its own stdout, so a bare `SKIP` would let a unit-test skip get promoted as an acceptance one by accident.

## 3. The enumeration

One file, `tests/acceptance/cells.json` (JSON, matching the `models/allowlist-raw.json` precedent that `tests/test_provenance.cpp:18` already reads from the source tree). Per cell: `name`, `runner` + args, `artifact_class`, `card_class`, `flags`, `design` (defining section), `gates`, `reports`, `external`.

| cell | runner | artifact / card | flags | defines | gates vs reports |
|---|---|---|---|---|---|
| `coder-offload-1lane` | `tests/equivalence/run.sh` | coder int4, 24 GB | `--offload-ratio 20 --paged-kv u8:i4`, 8 GiB pool | §5, §3.4 | gates byte-equality; reports chunk sweep |
| `coder-offload-2lane` | same, `ARCINT_EXTRA_ARGS="--parallel 2"` | " | " | §5 | gates |
| `coder-offload-concurrency` | `tests/concurrency/run.py` | " | " | §4.1, §5 | gates |
| `coder-served-large` | equivalence + decode sample | coder, 24 GB, no offload | `--paged-kv u8` | §5 | gates ≥ 60 t/s at steady state (53.4 cold / 69.2 warm, §7.0.2ai) |
| `coder-served-small` | equivalence + concurrency | coder, 16 GiB | `--paged-kv u8 --n-ctx 98304`, 2 GiB prefix cache | §5 | gates suites; reports 48.0/49.5 t/s |
| `agent-dense` | equivalence (incl. MTP section) + concurrency | dense 27B, 24 GB | `--paged-kv u8` | §3.5.2, §5 | gates MTP identity + acceptance > 10% |
| `tier-reference-cell` | window script | 35B int4, 16 GiB | ratio 50, 8 GiB pool, u8 KV, 1 lane, n_ctx 65536, 1,198-tok prompt, 64 greedy tokens, 2 requests/process | §3.4, §7.0.2ai | gates tier ON/OFF byte-identity and E2 (CONT after history = CONT fresh); reports 16.4 t/s decode, 26.6 prefill |
| `ngram-determinism-repeat` | window script | 35B tier config | `--draft 4 --draft-ngram 3`, six fresh processes | §7.0.2ae | gates six identical outputs |
| `depth-ladder` | window script | both cards | 98,147-token prefill at u8 and u8:i4 | §5.1 | gates load; reports t/s |
| `pruefstand` | **external** | coder, through the deployed package | greedy, thinking off | §5 | gates 10/10 |
| `sanitizers` | ASan+UBSan (x86_64), UBSan elsewhere, over unit+roundtrip+stress | none | `-fno-sanitize-recover` | §5 | gates zero reports |
| `package-build` | `contrib/packaging/arcint/build-deb.sh` | none | version stamp (`:60`), RPATH probe | §5.1 | gates; post-deploy smoke follows |

**External dependency, honestly.** The Prüfstand harness is fleet tooling outside this repository. Its cell carries `"external": {"env": "ARCINT_PRUEFSTAND", "invocation": …, "score_parse": …}` and the reference score — never the harness. Unset env ⇒ the cell reports `SKIPPED external-harness-not-configured`, which must then be named in `--allow-skip`, so a release that ran without it says so in its own log rather than passing.

**DESIGN §5.1 and the checklist are generated.** `tools/acceptance_manifest.py` renders the §5.1 acceptance block and `docs/release-checklist.md` between marker comments; `--check` regenerates and diffs. Rejected: hand-maintained plus a consistency test — `llm.txt:790` is the measured outcome of that policy.

## 4. Unit-side hygiene

Rule: **a unit case may assert only properties of arcint's own code; where it needs a host capability it probes first and SKIPs with a reason.** Present offenders:

- `tests/test_affinity.cpp:10-35` — post-`eae32d1` it picks the first allowed core, but still fails where `sched_setaffinity` is denied outright. Gets `SKIP_UNLESS(pin permitted, …)`.
- `tests/test_turnstile.cpp:59,63,84` — 30/50 ms sleeps; timing-sensitive on a loaded builder and under ASan. Keep, but documented as the one timing-dependent case.
- `tests/test_artifact.cpp:27` reads `TMPDIR`; `tests/test_provenance.cpp:18` reads the source tree via `ARCINT_SOURCE_DIR` (`CMakeLists.txt:198-199`) — both fine in-tree, both recorded.
- `tests/roundtrip.sh:14-15` and `tests/concurrency/stress.sh:20` exit 2 when `curl`/`python3` are missing, which ctest reports as a *failure*. Becomes exit 77 with a reason.

**Harness SKIP primitive** (`tests/harness.h`): `t::skip(reason)` and `SKIP_UNLESS(cond, reason)`, recording `{case, reason}`. `tests/test_main.cpp:47` prints `N cases run, M failed, K skipped` plus one line per skip. Exit stays 0 on skips, but ctest registers `unit` with `--max-skips 0`: a skip is a failure unless the case is named with `--allow-skip <case>` — the same rule as the acceptance runner, so there is one rule in two places.

## 5. Red-first plan

| mechanism | red case (fails before) | green after |
|---|---|---|
| enumeration ↔ §5.1/checklist | remove `ngram-determinism-repeat` from `cells.json`; separately, hand-edit the generated block in `DESIGN.md` | `acceptance-enumeration` (`--check`) red in both directions, green when list and generated files agree |
| runner skip discipline | fixture manifest `tests/acceptance/fixtures/cells-fake.json` with `/bin/true`, `/bin/false`, and a 77-exiting script: run before the allow-skip logic exists → unnamed 77 returns 0 | five assertions: all-pass ⇒ 0; unnamed 77 ⇒ ≠0; named ⇒ 0; naming a passed cell ⇒ ≠0; naming an unknown cell ⇒ ≠0 |
| label partition | register an acceptance cell outside the `ARCINT_ACCEPTANCE` guard | `ctest -N -L acceptance` lists 0 in a default build |
| harness SKIP | a case calling `SKIP_UNLESS(false, …)` under `--max-skips 0` while the flag is stubbed out | exits non-zero; named in `--allow-skip` ⇒ 0 |
| recipe gate | `ctest --no-tests=error -L unit` before labels exist | selects exactly 3 today |

Every one of these runs device-free.

## 6. Scope and order

**Increment 1 — meets the row's gate, no card needed.**

- `CMakeLists.txt:167-216`: labels, `ARCINT_ACCEPTANCE` option + configure-time refusal without `ARCINT_OPENVINO`, per-cell `add_test` from the manifest, `--max-skips 0` on `unit`, two new unit-labelled tests (`acceptance-enumeration`, `acceptance-runner`). ~45 lines.
- `tests/acceptance/cells.json` — the 12 cells above. ~220 lines.
- `tests/acceptance/run.py` — manifest loader, `--cell`/`--all`/`--allow-skip`, skip accounting, exit rules. ~250 lines.
- `tests/acceptance/selftest.py` + `fixtures/cells-fake.json` — ~120 lines.
- `tools/acceptance_manifest.py` — generator with `--check`. ~150 lines.
- `docs/release-checklist.md` — generated, committed. ~80 lines.
- `tests/harness.h`, `tests/test_main.cpp` — SKIP primitive and accounting. ~50 lines.
- `tests/test_affinity.cpp`, `tests/roundtrip.sh`, `tests/concurrency/stress.sh` — skip-with-reason. ~15 lines.
- `contrib/packaging/arcint/build-deb.sh:82` and its comment; `DESIGN.md` §5.1 acceptance row → generated block; `DEVELOPMENT.md:23-38`, `README.md:26-28`, `llm.txt:790-792`.

**Increment 2 — one GPU window.** Move the 0.3.0 gate's window scripts into `tests/acceptance/cells/` as each cell's runner, and add the `SKIP` line to `tests/equivalence/run.sh`. One window, both card classes, per the ritual for stopping and restoring the resident services; the only thing being confirmed is that each runner starts and its console parse matches, not new numbers.

**Increment 3.** Reference values in `reports` so a regression is caught rather than merely recorded.

## 7. Risks

- **Wrongly green.** "ctest green" must never be quoted as a release gate — it says nothing about acceptance by construction. The checklist names the target that produced each line. The larger hazard is ctest's `SKIP_RETURN_CODE`: if acceptance cells are ever registered with it and run via bare `ctest -L acceptance`, a card-less machine goes green with everything skipped. The runner is the only entry point for that reason.
- **Wrongly red.** `--no-tests=error` requires a recent enough ctest; check the package builder's CMake before relying on it, and fall back to asserting the count from `ctest -N -L unit`. Widening the recipe to three tests adds up to ~7 minutes and binds an ephemeral port twice — a builder with a restrictive network namespace turns a green package red. That failure is real and should stand.
- **`ARCINT_OPENVINO`.** The OV build has 4 more cases and `test_config.cpp:47-58` on the other branch. Never assert a bare case *total*; assert skips. `ARCINT_ACCEPTANCE=ON` with `ARCINT_OPENVINO=OFF` must be refused at configure time — a stub binary would "pass" a card cell by serving stub bytes, the worst available outcome.
- **python3** becomes a hard dependency of two more unit tests. It already is one for `roundtrip` and `stress`, so this changes degree, not kind.
