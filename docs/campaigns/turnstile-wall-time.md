# turnstile-wall-time — tests/test_turnstile.cpp orders its threads by wall-clock sleeps, not synchronisation

## The defect, as measured

`tests/test_turnstile.cpp` (89 lines) exercises `lgc::Turnstile`
(`src/core/turnstile.h`, DESIGN §4.1's ticket lock over the device:
"whoever asks first runs first"). Two of its three tests decide an
ordering or timing assertion with `std::this_thread::sleep_for`, not with
a synchronisation primitive:

- `turnstile_serves_in_arrival_order` (lines 38–72): each contender sets
  an atomic flag; the main thread busy-yields until it sees the flag,
  then **sleeps 30 ms**, assuming that is enough wall time for the
  contender's own `gate.take()` to reach its mutex and register a ticket
  (`Turnstile::take()`, `src/core/turnstile.cpp:22–30`,
  `next_ticket_++` under `mutex_`) before the next contender starts. Two
  such sleeps (30 ms, 30 ms) gate `CHECK_EQ(order[0], 2)` /
  `CHECK_EQ(order[1], 3)`.
- `turnstile_reports_the_wait_it_imposed` (lines 74–89): a single 50 ms
  sleep before releasing the held turn, then
  `CHECK(waited.load() >= 0.04)` — the same assumption, that the other
  thread has already entered `cv_.wait()` before the sleep elapses.

Both are the same race, in the same direction: the assertion is decided
by whether the scheduler gets the other thread far enough within the
sleep window, not by anything the test itself observes.
`turnstile_lets_one_lane_through_at_a_time` (lines 13–36) is not in this
class — it counts overlaps and a total, not order, and carries no sleep.

Separately, the `roundtrip` ctest entry (`tests/roundtrip.sh`, 64 checks,
DESIGN §5.1's Unit row) flaked once under build load in the 0.3.1 window.
This is **not on DESIGN's or CHANGELOG's record** — no other flake of
this entry is documented anywhere in DESIGN.md, CHANGELOG.md or
`docs/milestone-0.3.0.md` (the only documented test flake in this period
is the n-gram-drafter acceptance check, DESIGN §7.0.1/§7.0.2ae, a
different suite entirely) — so it is carried here as a single observed
event, not a characterised mechanism, until it either reproduces or is
shown not to.

## Known against hypothesised

Known: the two sleep-gated assertions and their line numbers; the
`Turnstile` API surface (`take()`, `served()`, `Turn::waited_seconds()` —
no ticket number or "enqueued" signal is exposed today); the one
roundtrip flake is a single event with no logged cause. Hypothesised:
that it shares a mechanism with the turnstile sleeps — `roundtrip.sh` has
its own polling sleeps (lines 63, 324, 343, 369, 415, 455) that wait for
server readiness rather than decide an assertion, so the resemblance may
be coincidental; the recon should check each rather than assume it.

## Gate

The turnstile test orders its threads by synchronisation — each thread
signals when it has taken its ticket (or is blocked waiting for one), the
next thread waits on that signal — so no wall-clock sleep decides an
assertion. Shown red-first: on the *old* test, a deliberately slowed
thread (an injected delay between setting the flag and calling `take()`)
must be able to fail the ordering check on bad luck; the rewrite closes
that window. `ctest -L unit` green 20 times in a row under a parallel
build (`-j`) on the aarch64 build host (the same host class DESIGN §5's
Sanitizers paragraph already runs UBSan on, since ASan itself cannot
start there). The roundtrip flake either reproduces under matched build
load and is fixed, or is recorded as not reproduced in N runs (N stated,
with the load conditions approximated as closely as they can be
reconstructed from the single observed event).

## Entry criteria

Met. The test file, the `Turnstile` class and its one production call
site (`src/exec/backend_ov.cpp:5811–5812`) are all on `main`; charter
marks this "Small," no design note required first.

## Scope — in / out

In: `tests/test_turnstile.cpp`'s two order/timing-sensitive tests,
rewritten to synchronise rather than sleep; whatever minimal, test-only
observability `Turnstile`/`Turnstile::Turn` needs for that (a hook or
accessor, not a behaviour change — `served()` already exists "exposed
for the tests" per its own comment); the 20-repeat `ctest -L unit` run
under parallel build load; the roundtrip flake's reproduction attempt.

Out: `Turnstile`'s own runtime behaviour (DESIGN §4.1's stall bound);
`roundtrip.sh`'s own polling sleeps, unless the recon finds they share
the turnstile tests' race — then that is this campaign's finding, not a
silent fix; any other flaky test not named here.

## Where it lives

`tests/test_turnstile.cpp` (all three tests); `src/core/turnstile.h`/`.cpp`
(`Turnstile::take()`/`release()`/`served()`, `Turn::waited_seconds()`);
`src/exec/backend_ov.cpp:5811–5812` (the one production call site);
`CMakeLists.txt:209` (registers the test file), `:229–230` (`unit`,
`LABELS unit`), `:232–238` (`roundtrip`, 64 checks, `LABELS unit`,
`TIMEOUT 120`); `tests/roundtrip.sh` (467 lines; polling sleeps at lines
63, 324, 343, 369, 415, 455); DESIGN §4.1 (ticket-lock rationale) and
§5.1 (the Unit-class table, where `roundtrip` is listed).

## Pipeline for this campaign

Recon (read the two sleep-gated tests and `Turnstile`'s public surface;
check `roundtrip.sh` for the same pattern) → red-first: the
deliberately-slowed-thread case against the *current* test, showing it
can fail on bad luck → the synchronising rewrite (and its enabling
accessor, if any) → 20 green `ctest -L unit` repeats under parallel build
load on the aarch64 host → the roundtrip flake's reproduction attempt
under matched load → review → commit → CHANGELOG line (test
infrastructure, since no served behaviour changes).

## Invariants

No production behaviour changes: `Turnstile`'s ticket-lock semantics and
DESIGN §4.1's stall bound are untouched; any accessor added for the test
is test-only, the same way `served()` already is. A test that passes by
construction (synchronised) rather than by timing is the entire point —
a rewrite that still has a sleep deciding an assertion has not closed
this campaign.

## Status

- 2026-09-05 — opened from the 0.3.1 window; nothing started.
- 2026-09-05, closed (DESIGN §7.0.2am). Red first: a 100 ms delay between
  the contender's flag and its `take()` failed both sleep-gated cases
  three of three (order {3, 2}; a zero wait). Rewrite: `Turnstile::issued()`
  (test-only, beside `served()`); the ordering case waits for the ticket
  count to advance before starting the next contender, with the second
  contender dawdling 20 ms on purpose; the wait case bounds the reported
  wait by an interval it measures itself, which holds by construction.
  Removing either wait fails its case ten of ten; fifty local runs green.
  Gate: `ctest -L unit --repeat until-fail:20` under a continuous `-j4`
  clean rebuild (load 2.6–5.1), twenty of twenty; `roundtrip` forty more
  under the same load, forty of forty — sixty runs, the flake not
  reproduced, on the record as observed once. Finding: `roundtrip.sh`'s
  cancellation check was the same class (a fixed 0.5 s sleep before the
  abort-line grep); measured 11–32 ms latency under load, so not shown to
  be the flake; replaced by a bounded poll, red first with a never-logged
  pattern. No served behaviour changed; CHANGELOG line under Unreleased.
