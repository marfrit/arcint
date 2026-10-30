# expert-foresight round 8 — handover to the card window (2026-10-30)

Supersedes [round 7](expert-foresight-handover-round7.md). Round 8 corrects the
review of round 7's code: the pending guard was per cache (MEDIUM), the red
cell mirrored the call order instead of running it (LOW), and check 4 of the
round-7 handover read the wrong stream while the gate priced decode only
(LOW). Nothing here ran on a GPU: the cards stay in the operator's announced window; every code claim is
`code` with device-free red cells.

Code: llama tree `fc4439f` (branch `trace`), patch
`contrib/llama.cpp/patches/0069-expert-route-trace.patch` regenerated from its
diff against the arcint-series base `43fefb0`. Arcint side at `9964462`; this
handover is the doc-only commit after it. The round-7 code was `b0df2b0` after
the round-7 handover commit `f03a2ba`. The recipe run before this handover:

    RECIPE 9964462: 100% tests passed, 0 tests failed out of 6 EXIT 0 deb arcint_1.2.6-1_amd64.deb

so the regenerated patch applies to the pin and the package builds.

Device-free at this commit (reproduced by whoever reviews):

```bash
cd ~/src/ligence-foresight
EXPERT_ROUTE_TRACE_SRC=<the llama tree> python3 -m pytest tests/python/test_expert_route_trace.py -q   # 32 passed
```

Round 8's three cells and the one-line mutation each was seen failing:

- `test_layer_split_pending_is_process_wide_and_snapshot_has_every_card`:
  `rotate_now` drops the process-wide count (`t.pending_graphs > 0` -> `false`)
  -- the selftest exits 8 or 9, the first prepare rotates while the other
  cache's record waits. The same cell's snapshot half: `event_layers_all`
  writes only the last registered set (`for (const auto & s : layer_sets)` ->
  `for (const auto & s : { layer_sets.back() })`) -- the new file reads two
  layers where four are expected.
- `test_advance_never_prepares_and_after_graph_does`: insert
  `trace_prepare();` before `progress();` in `llama_expert_cache::advance` --
  the source-order check fails. This is the coverage round 7 did not have:
  its cells emulated the order, so that insertion passed all 30.
- Round 7's own two cells still pass: the pending-order red arm
  (`EXPERT_ROUTE_SELFTEST_ROTATE_ON_PENDING=1`) and the write-fail red cell
  (`opened = false;` dropped prints `write-fail opened 1`).

What the round-8 fix is, in one paragraph (`code`): the trace singleton holds
a process-wide pending count and every registered cache's residents. A graph
record increments the count at `after_graph` and its write in `advance` or
`flush_trace` decrements it (a write failure releases it); `rotate_now(t)`
tests the count, so no cache rotates while any cache's record for the same
graph is pending. Each cache refreshes its residents at init and at every
TABLES record; a rotation writes one LAYERS record over all of them, so every
file replays alone under a layer split. The replay's reader merges several
LAYERS records in one file into one cache list. The trace flag and
`LLAMA_EXPERT_ROUTE_TRACE_MAX_MB` are unchanged.

## What the card window runs (announced window, SOP sampler first)

One lane, the `fn_sweep.py` unit flags (11,600 MiB slots, CPU misses, bank
stream), greedy answers, a 20k-token corpus prompt, the six answers. Two fresh
processes, alternated if a second round is possible. **The unit's served file
and asset directory**, not the older XXS paths. `fn_sweep`'s own stdout is
teed per arm: its `==` line is the arm's command line, and its `prompt` line is
the arm's prefill rate (the trace's known cost is one scheduler synchronize a
prompt ubatch, so prefill is recorded beside decode). **Round 10's amendment
(`code`): this build's DRAFT path was dead — the MTP ids' buffer was null, so
no read was issued, no pending record and no decode-start synchronize — so
both arms' numbers exclude the DRAFT read-back and its one wait; the served
A/B with that path live is owed, see
[expert-foresight-handover-round10.md](expert-foresight-handover-round10.md).**

```bash
cd ~/src/ligence-foresight
set -o pipefail                    # tee must not hide fn_sweep's exit code
ARCINT=/path/to/the/recipe/arcint  # the package binary of this commit's recipe
MODEL=/flash-model/gguf-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
ASSETS=/flash-model/wp-tmp/w1/assets    # expert-profile.bin, mtp-flash-next.gguf, mtp-vocab.json, corpus.txt
DIR=/flash-model/wp-tmp/round8-on

# arm off (the control), one fresh process
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" off --prompt-tokens 20000 \
  2>&1 | tee /tmp/fn-sweep-off.out

# arm on: the trace through its FLAG (the backend unsets an env when the flag
# is empty), the cap through its env. A small cap forces several rotations
# inside the 20k prompt and across the answers.
rm -rf "$DIR"
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" trace-on --prompt-tokens 20000 \
  --set --llama-expert-route-trace "$DIR" \
  --env LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=8 \
  2>&1 | tee /tmp/fn-sweep-trace-on.out
```

Raise `MAX_MB` to 64 for a gentler run if 8 distorts the timing; the rotation
path is still exercised by the 20k prompt and the answer verifies. The server
log stays `/tmp/fn-sweep-<tag>.log`; the teed stdout is
`/tmp/fn-sweep-<tag>.out`.

Each arm's last line is now:

    SWEEP <tag> decode-mean <t/s> accept-mean <%> hits-mean <%> prefill-mean <t/s>

## Post-run checks on the on arm (all six are findings, not optional)

```bash
DIR=/flash-model/wp-tmp/round8-on

# 1. no trace warning: 0 expected, a WRITE FAILURE says tracing stopped (the
#    run itself is still valid, but the trace is not a run's trace)
grep -c "WARN: expert route trace" /tmp/fn-sweep-trace-on.log

# 2. size sanity: the 20k prompt alone is n_tokens x top-k x n_layer x 4 B
#    = 20000 x 10 x 48 x 4 ~ 38 MB; the answers add ~1.9 KB a decode row
du -sh "$DIR"

# 3. the arm really traced: zero .ert1 is the trace-off arm, NOT a clean run
find "$DIR" -maxdepth 1 -name '*.ert1' | wc -l

# 4. the arm's own command line and its prefill rate, from fn_sweep's stdout
head -1 /tmp/fn-sweep-trace-on.out
grep '^  prompt ' /tmp/fn-sweep-trace-on.out /tmp/fn-sweep-off.out

# 5. both arms' gate numbers on one line each
grep '^SWEEP ' /tmp/fn-sweep-off.out /tmp/fn-sweep-trace-on.out
```

- **Check 4 was wrong in round 7**: it read `/tmp/fn-sweep-trace-on.log`, the
  arcint server log, for the `==` line, which `fn_sweep` prints on its own
  stdout. Without the tee that line lives only in the terminal.
- **Zero `.ert1` files means the arm ran with the trace off**, even when the
  command line shows the flag: not a clean run. Check the printed `== trace-on:`
  line first, then the log for a `WARN: expert route trace: tracing is off`
  (a refused/missing DIR, or a build without 0069).
- **A `WARN` means the trace stopped mid-run** (a full disk, a closed fd): the
  records are a prefix, every check below still runs, and the A/B speed is
  valid -- but the trace's coverage is not the run's.

## What to read from the on arm's trace

The reading steps of [round 6](expert-foresight-handover-round6.md#what-to-read-from-the-on-arms-trace)
still apply (rotations, `ERT1 v2`, per-file LAYERS first record, DRAFT count >
0, each file replays alone, sidecars a file, answer md5s). Round 7 adds one
property; round 8 leaves it intact on one card.

```bash
DIR=/flash-model/wp-tmp/round8-on
python3 tools/expert_cache_replay.py "$DIR" --bounds
python3 tools/expert_cache_replay.py "$DIR" --accept "$DIR"   # each file's sidecar
for f in "$DIR"/*.ert1; do [ -f "$f.accept" ] && printf '%s trace=%s accept=%s\n' "$(basename "$f")" "$(stat -c%s "$f")" "$(wc -l < "$f.accept")"; done
```

- **No accept-line split across a rotation**: the replay's `--accept DIR` walk
  must not exit on "out of step" / "no line", and a file that carries a
  `VERIFY` record must have its line in the same `<file>.accept`. A DRAFT
  record crossing the cap between `after_graph` and the next `advance` is the
  case round 7 fixed, and round 8's process-wide count keeps it fixed when a
  second cache is in the process.
- **`N MTP draft records` > 0** is the first served exercise of the DRAFT read
  path: round 7 reads the ids through the OpenCL batch hook
  (`ggml_backend_opencl_get_tensors_async`) instead of the blocking fallback.
  The records alone do not price it; the A/B's decode-mean does.
- Every `.ert1` with a decode graph has a `.accept` beside it; a file with only
  LAYERS (the last rotation, or a request that ended at the cap) may have no
  line -- that is not an unwritten last graph.

Gate (`measured-here` when it comes): decode-mean within the off arm's
run-to-run spread, **prefill-mean not below the off arm beyond the spread**
(the trace's one synchronize a prompt ubatch is its known cost), accept-mean
not lower beyond the spread, the six answer md5s identical, the capital and
the prompt's answer right. The round-7/8 mechanisms do not change routing or
sampling (the read hook is a device read; the rotation is a file split;
the count is an ordering guard), so a mismatch is a defect, not a trade.

## Layer-split A/B for later (two cards, not this round)

Run only after the [layer-split charter](layer-split-two-cards.md) reaches its
own gate and the two cards are in an announced window; the trace branch must be the
pin this recipe builds. Same lane and assets, both cards named in `--device`
order, K layers on the first card, a cache budget per card (MiB, 0 keeps none
on that card), and the MTP model on the card the charter measured best (B60,
`--llama-mtp-card 0`).

```bash
cd ~/src/ligence-foresight
set -o pipefail
ARCINT=/path/to/the/recipe/arcint
MODEL=/flash-model/gguf-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
ASSETS=/flash-model/wp-tmp/w1/assets
DIR=/flash-model/wp-tmp/round8-split-on

# one-card baseline on the B60, same K's layers as the split arm's first card
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" one --prompt-tokens 20000 \
  2>&1 | tee /tmp/fn-sweep-one.out

# split arm: --device GPU.0,GPU.1, --llama-layer-split K (arcint sets
# tensor_split exactly; K runs 1..n_run-1), --llama-expert-cache B60,A770.
# The two cards' log lines show one cache a card. The trace flag reaches both
# caches' init; the cap is one env.
rm -rf "$DIR"
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" split-on --prompt-tokens 20000 \
  --set --device GPU.0,GPU.1 \
  --set --llama-layer-split 36 \
  --set --llama-expert-cache 10500,6000 \
  --set --llama-mtp-card 0 \
  --set --llama-expert-route-trace "$DIR" \
  --env LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=8 \
  2>&1 | tee /tmp/fn-sweep-split-on.out
```

The same arm with the trace off is the split control:

```bash
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" split-off --prompt-tokens 20000 \
  --set --device GPU.0,GPU.1 --set --llama-layer-split 36 \
  --set --llama-expert-cache 10500,6000 --set --llama-mtp-card 0 \
  2>&1 | tee /tmp/fn-sweep-split-off.out
```

K and the budgets are the charter's measured values for the quant in the
window; re-read `layer-split-two-cards.md`'s eviction note and keep each card's
margin above what a request allocates late. Checks beyond the single-card set:

- per-card `expert cache:` load lines and per-card stats after each request
  (one cache a card);
- the first `.ert1`'s LAYERS record holds every card's layers (round 8); a
  later rotation's LAYERS record also holds all of them;
- a verify graph carries both caches' records, and the replay's
  `--accept` walk pairs one line with the merged graph (the reader merges the
  records a graph);
- the A770's PCIe 3.0 x4 link: report its miss share next to prefill and
  decode, as the charter's gate asks.

## If something fails

The trace can never fail the run: a write error is one `WARN` and tracing
stops; a read-back stops with it. A missing DIR is created. A stuck or absent
trace is a replay finding, not a serving failure. The reader resynchronises on
a later `ERT1` header, so an old pre-round-6 file still replays. A missing
`.accept` beside a file with a VERIFY graph is the one red case the device-free
cells force (`test_pending_record_never_rotates_away_from_its_accept_line`);
under a layer split the same cell's round-8 companion covers the second cache.
