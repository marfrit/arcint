# expert-foresight round 9 — handover to the card window (2026-10-30)

Supersedes [round 8](expert-foresight-handover-round8.md) for the DRAFT check;
round 8's run and post-run checks still apply. Nothing here ran on a GPU: the
cards stay in the operator's announced window; every code claim is `code` with a
device-free red cell.

Code: llama tree `b12a4ab` (branch `trace`), patch
`contrib/llama.cpp/patches/0069-expert-route-trace.patch` regenerated from its
diff against the arcint-series base `43fefb0`. The round-8 code was `fc4439f`.
The arcint side at `cf709e2`; the recipe run before this handover:

    RECIPE cf709e2: 100% tests passed, 0 tests failed out of 6 EXIT 0 deb arcint_1.2.6-1_amd64.deb

## The defect and the fix, one paragraph (`code`)

Round 8's card run wrote 0 MTP DRAFT records beside a ~65 % accept rate. The
trace's MTP block marks its routed ids an output and stores them for decode's
read-back, but nothing in the FFN consumes the tensor; the scheduler allocates
only nodes the forward walk reaches, so `ids_out->buffer` stayed null, decode's
read-back skipped it (`rids->buffer == nullptr`), and no DRAFT record was
written. One `ggml_build_forward_expand(gf, ids_out)` in that block fixes it.
The cache's ids_out below is reachable through `ids_flat` and never needed one.
Off (`LLAMA_EXPERT_ROUTE_TRACE` unset) still adds no tensor and no node.

Device-free repro (`code`): a CPU qwen4exp MTP context, one file, two draft
decodes a token, the writer opened without a cache: `DRAFT 0` before, `DRAFT 2`
with the line. The repro's synthetic MTP model is a merged file the stock loader
refuses (`done_getting_tensors` requires every tensor used), so it is cited as
evidence, not committed.

Device-free at this commit (reproduced by whoever reviews):

```bash
cd ~/src/ligence-foresight
EXPERT_ROUTE_TRACE_SRC=<the llama tree> python3 -m pytest tests/python/test_expert_route_trace.py -q   # 33 passed
```

Red cell and its one-line mutation, seen failing:
`test_mtp_route_ids_are_expanded_into_the_graph` reads `src/llama-graph.cpp`,
names the trace's MTP block, and asserts the expand sits between the output mark
and `set_route_ids`; delete `ggml_build_forward_expand(gf, ids_out);` and the
cell fails (the CPU repro then reported `DRAFT 0`).

## What the card window runs for this check (tiny input, not 20k)

One lane, the `fn_sweep.py` unit flags (they include `--llama-mtp 3` and
`--llama-mtp-gguf`, so the MTP head is loaded), a **64-token** prompt, the six
answers. One fresh process, the trace on.

```bash
cd ~/src/ligence-foresight
set -o pipefail
ARCINT=/path/to/the/recipe/arcint
MODEL=/flash-model/gguf-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
ASSETS=/flash-model/wp-tmp/w1/assets
DIR=/flash-model/wp-tmp/round9-on

rm -rf "$DIR"
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" mtp-trace --prompt-tokens 64 \
  --set --llama-expert-route-trace "$DIR" \
  --env LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=8 \
  2>&1 | tee /tmp/fn-sweep-mtp-trace.out
```

Checks (all are findings):

```bash
DIR=/flash-model/wp-tmp/round9-on

# 1. the trace opened: one or more .ert1 files, no WARN
find "$DIR" -maxdepth 1 -name '*.ert1' | wc -l
grep -c "WARN: expert route trace" /tmp/fn-sweep-mtp-trace.out

# 2. DRAFT records > 0: the replay's first line says "N MTP draft records"
python3 tools/expert_cache_replay.py "$DIR"

# 3. the mtp predictor has rows: the report has an "mtp" row beside
#    "oracle", "draft ids", "prompt prior", "learned" (a row of zeros or a
#    missing row means the DRAFT records did not pair with a verify)
python3 tools/expert_cache_replay.py "$DIR" --sweep
```

The first replay line also gives the graph kind counts; the DRAFT count must be
comparable to the verify count (one DRAFT record a draft step that fed a
verify), not zero. A tiny prompt writes few records, so `MAX_MB=8` never
rotates here; rotation is round 8's check, not this one.

A `WARN` means tracing stopped mid-run: the records are a prefix, checks 2 and 3
then read a prefix, and the run's speed is still valid.
