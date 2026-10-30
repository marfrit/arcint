# expert-foresight round 6 — handover to the card window (2026-10-30)

Code at `c7af811` (branch `ligence-foresight`), llama tree
`4487f99` (branch `trace`), patch 0069 regenerated from it. The recipe on the
branch's HEAD (this handover is a doc-only commit after `c7af811`):
100% tests passed, 0 tests failed out of 6, EXIT 0; the reply quotes its
`RECIPE` line.
Device-free cells: `python3 -m pytest tests/python/test_expert_route_trace.py -q`
= 28 passed, plus the `EXPERT_ROUTE_TRACE_SRC` compile cells. Every claim here
is `code` until the A/B section, which is the open `measured-here` gate.

The seven review findings are closed in
[expert-foresight.md](expert-foresight.md#round-6--the-trace-safe-for-a-month-2026-10-30-code--red-tests).
Nothing here is a serving result: the trace's month-safety and the MTP map are
tested offline only. The A/B below is what turns them into a served number.

## What Claude runs (its announced window, SOP sampler first)

**Corrected 2026-10-30 (round 7, `code`):** the arm below passed
`--env LLAMA_EXPERT_ROUTE_TRACE="$DIR"`, and the backend unsets that variable
whenever `--llama-expert-route-trace` is empty
(`src/exec/backend_llama.cpp:391-392`, before the model's load) — that arm ran
the trace **off**. The trace is set by its flag, not by the env; only
`LLAMA_EXPERT_ROUTE_TRACE_MAX_MB` stays an env. The current handover with the
full command and the post-run checks is
[expert-foresight-handover-round7.md](expert-foresight-handover-round7.md).

## What to read from the on arm's trace

```bash
DIR=/flash-model/wp-tmp/round6-on
find "$DIR" -maxdepth 1 -name '*.ert1' | LC_ALL=C sort
du -sh "$DIR"

# the whole directory as one history, with its sidecars
python3 tools/expert_cache_replay.py "$DIR" --bounds

# each rotated file alone: header, LAYERS snapshot, its graphs
for f in "$DIR"/*.ert1; do echo "== $f"; python3 tools/expert_cache_replay.py "$f" --single >/dev/null && echo "replays alone"; done

# sidecar lines a file (a file with no decode graph may have no .accept)
for f in "$DIR"/*.ert1; do if [ -f "$f.accept" ]; then printf '%s trace=%s accept=%s\n' "$(basename "$f")" "$(stat -c%s "$f")" "$(wc -l < "$f.accept")"; else printf '%s trace=%s accept=none\n' "$(basename "$f")" "$(stat -c%s "$f")"; fi; done

# the first record a file: must be LAYERS (0), not a graph
python3 - "$DIR" <<'PY'
import sys
sys.path.insert(0, "tools")
import expert_cache_replay as r
for f in r.expand_traces(sys.argv[1]):
    tr = r.load_route(f)
    print(f.split("/")[-1], "first", tr["events"][0]["kind"], "torn", tr["torn"], "events", len(tr["events"]))
PY
```

Read them as:

- `find` shows more than one `.ert1` and a `.000`/`.001` sequence, so the 8 MB
  cap rotated (`code`).
- The replay header says `ERT1 v2`, `%d MTP draft records` **greater than 0**
  (Flash-Next's MTP head ran; 0 means the async DRAFT path was not exercised),
  and no `TORN tail`.
- `unwritten_last_graph` is absent from the replay output. The flush is what
  makes it 0; the round-3 build printed it once. If it appears, the request-end
  `TraceFlush` did not run under `mu_`.
- Every file replays alone and its first record is `LAYERS` (0) with `torn
  False`: that is the rotation snapshot, not a header-only stub (`code`).
- Each `.001`, `.002`, ... has its own sidecar beside it; a verify's line and
  its graph record in one file is the rotation-before-the-line fix (`code`).
- The final answer md5s and `SWEEP` lines are the A/B's answer and speed
  evidence; paste them into the campaign document with the card by PCI id, the
  build sha256, the cap and the six rates.

## If something fails

The trace can never fail the run: a write error is one `WARN` and tracing
stops; a read-back stops with it. A missing DIR is created. A stuck or absent
trace is a replay finding, not a serving failure. The reader resynchronises on
a later `ERT1` header, so an old pre-round-6 file still replays; a rotation line
before the accept line is the one red case the test forces.

No GPU was touched for this round; the card stays Claude's.
