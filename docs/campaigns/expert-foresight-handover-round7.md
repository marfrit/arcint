# expert-foresight round 7 — handover to the card window (2026-10-30)

**Superseded by [round 8](expert-foresight-handover-round8.md)**: the pending guard was per cache
(a layer split), the red cell mirrored the call order, and check 4 below read the server log for a
line `fn_sweep` prints on its stdout. Use round 8's commands and checks.

Supersedes [round 6](expert-foresight-handover-round6.md), whose on-arm ran
with the trace **off**: it passed `--env LLAMA_EXPERT_ROUTE_TRACE="$DIR"`, and
the backend unsets that variable whenever `--llama-expert-route-trace` is
empty (`src/exec/backend_llama.cpp:391-392`, before the model's load). The
trace is set by its flag; only `LLAMA_EXPERT_ROUTE_TRACE_MAX_MB` is an env.

Arcint side at `b0df2b0`; llama tree at `611ecaa` (branch `trace`), patch
`contrib/llama.cpp/patches/0069-expert-route-trace.patch` regenerated from it
(the trace branch's diff from its arcint-series base `43fefb0` to
`611ecaa`, the round-6 header kept and extended). The recipe on `b0df2b0`:

    RECIPE b0df2b0: 100% tests passed, 0 tests failed out of 6 EXIT 0 deb arcint_1.2.6-1_amd64.deb

so the regenerated patch applies to the pin and the package builds. Round 7's three residuals and two lows are closed in
[expert-foresight.md](expert-foresight.md#round-7--the-three-residuals-of-the-round-6-review-2026-10-30-code--red-tests).
Every claim here is `code` with device-free red cells until the A/B section,
which is the open `measured-here` gate. **No GPU was touched for this round;
the cards stay Claude's.**

Device-free at this commit (reproduced by whoever reviews):

```bash
cd ~/src/ligence-foresight
EXPERT_ROUTE_TRACE_SRC=<the llama tree> python3 -m pytest tests/python/test_expert_route_trace.py -q   # 30 passed
```

The two round-7 cells and the one-line mutations each was seen failing:
`test_pending_record_never_rotates_away_from_its_accept_line` (red arm rotates
at advance, or the decision function drops `graph_pending` — three files, the
line in `.000`, its record in `.001`) and
`test_write_failure_stops_tracing_with_one_warn` (drop `opened = false;` in the
writer's error path — `write-fail opened 1`).

## What Claude runs (its announced window, SOP sampler first)

One lane, the `fn_sweep.py` unit flags (11,600 MiB slots, CPU misses, bank
stream), greedy answers, a 20k-token corpus prompt, the six answers. Two fresh
processes, alternated if a second round is possible. **The unit's served file
and asset directory**, not the older XXS paths:

```bash
cd ~/src/ligence-foresight
ARCINT=/path/to/the/recipe/arcint       # the package binary of this commit's recipe
MODEL=/flash-model/gguf-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
ASSETS=/flash-model/wp-tmp/w1/assets    # expert-profile.bin, mtp-flash-next.gguf, mtp-vocab.json, corpus.txt
DIR=/flash-model/wp-tmp/round7-on

# arm off (the control), one fresh process
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" off --prompt-tokens 20000

# arm on: the trace through its FLAG (an env is unset by the backend when the
# flag is empty), the cap through its env. A small cap forces several
# rotations inside the 20k prompt and across the answers.
rm -rf "$DIR"
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" trace-on --prompt-tokens 20000 \
  --set --llama-expert-route-trace "$DIR" \
  --env LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=8
```

Raise `MAX_MB` to 64 for a gentler run if 8 distorts the timing; the rotation
path is still exercised by the 20k prompt and the answer verifies.

`fn_sweep.py` prints `SWEEP <tag> decode-mean <t/s> accept-mean <%> hits-mean
<%>`, the slot's decode lines and the answer md5s; its server log is
`/tmp/fn-sweep-<tag>.log`.

## Post-run checks on the on arm (all four are findings, not optional)

```bash
DIR=/flash-model/wp-tmp/round7-on

# 1. no trace warning: 0 expected, a WRITE FAILURE says tracing stopped (the
#    run itself is still valid, but the trace is not a run's trace)
grep -c "WARN: expert route trace" /tmp/fn-sweep-trace-on.log

# 2. size sanity: the 20k prompt alone is n_tokens x top-k x n_layer x 4 B
#    = 20000 x 10 x 48 x 4 ~ 38 MB; the answers add ~1.9 KB a decode row
du -sh "$DIR"

# 3. the arm really traced: zero .ert1 is the trace-off arm, NOT a clean run
find "$DIR" -maxdepth 1 -name '*.ert1' | wc -l

# 4. the arm's own command line, as fn_sweep printed it
head -1 /tmp/fn-sweep-trace-on.log
```

- **Zero `.ert1` files means the arm ran with the trace off**, even when the
  command line shows the flag: not a clean run. Check the printed `== trace-on:`
  flag line first, then the log for a `WARN: expert route trace: tracing is off`
  (a refused/missing DIR, or a build without 0069).
- **A `WARN` means the trace stopped mid-run** (a full disk, a closed fd): the
  records are a prefix, every check below still runs, and the A/B speed is
  valid — but the trace's coverage is not the run's.

## What to read from the on arm's trace

The reading steps of [round 6](expert-foresight-handover-round6.md#what-to-read-from-the-on-arms-trace)
still apply (rotations, `ERT1 v2`, per-file LAYERS first record, DRAFT count >
0, each file replays alone, sidecars a file, answer md5s). Round 7 adds one
property:

```bash
DIR=/flash-model/wp-tmp/round7-on
python3 tools/expert_cache_replay.py "$DIR" --bounds
python3 tools/expert_cache_replay.py "$DIR" --accept "$DIR"   # each file's sidecar
for f in "$DIR"/*.ert1; do [ -f "$f.accept" ] && printf '%s trace=%s accept=%s\n' "$(basename "$f")" "$(stat -c%s "$f")" "$(wc -l < "$f.accept")"; done
```

- **No accept-line split across a rotation**: the replay's `--accept DIR` walk
  must not exit on "out of step" / "no line", and a file that carries a
  `VERIFY` record must have its line in the same `<file>.accept`. A DRAFT
  record crossing the cap between `after_graph` and the next `advance` is the
  case round 7 fixed: the rotation waits for the next `after_graph`, so the
  line and its record never part. A split here is the round-7 defect back.
- **`N MTP draft records` > 0** is also the first served exercise of the DRAFT
  read path: round 7 reads the ids through the OpenCL batch hook
  (`ggml_backend_opencl_get_tensors_async`) instead of the blocking fallback.
  The records alone do not price it; the A/B's decode-mean does.
- Every `.ert1` with a decode graph has a `.accept` beside it; a file with only
  LAYERS (the last rotation, or a request that ended at the cap) may have no
  line — that is not an unwritten last graph.

Gate (`measured-here` when it comes): decode-mean within the off arm's
run-to-run spread, accept-mean not lower beyond the spread, the six answer md5s
identical, the capital and the prompt's answer right. The round-7 mechanisms do
not change routing or sampling (the read hook is a device read; the rotation is
a file split), so a mismatch is a defect, not a trade.

## If something fails

The trace can never fail the run: a write error is one `WARN` and tracing
stops; a read-back stops with it. A missing DIR is created. A stuck or absent
trace is a replay finding, not a serving failure. The reader resynchronises on
a later `ERT1` header, so an old pre-round-6 file still replays. A missing
`.accept` beside a file with a VERIFY graph is the one red case the device-free
cells force (`test_pending_record_never_rotates_away_from_its_accept_line`).
