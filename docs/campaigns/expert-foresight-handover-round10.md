# expert-foresight round 10 — handover to the card window (2026-10-30)

Supersedes [round 9](expert-foresight-handover-round9.md) for the DRAFT check
and adds the fusion check; round 8's run and its rotation checks still apply.
**Superseded by [round 11](expert-foresight-handover-round11.md)** for the
fusion line's delivery (name after `cb`, prefix compare, WARN once per
process) and for the log path in checks 1 and 4. Nothing here ran on a GPU: the cards stay in the operator's announced window;
every code claim is `code` with device-free red cells, and the card checks
below are `measured-here` owed.

Code: llama tree `7e0ba9e` (branch `trace`), patch
`contrib/llama.cpp/patches/0069-expert-route-trace.patch` regenerated from its
diff against the arcint-series base `43fefb0`. The round-9 code was `b12a4ab`.
The arcint side at the commit this handover ships; the recipe run before it:

    RECIPE 4ed26ba: 100% tests passed, 0 tests failed out of 6 EXIT 0 deb arcint_1.2.6-1_amd64.deb

## What changed, one paragraph each (`code`)

**HIGH — the MTP block pushed its router out of the fusion.** Round 9's
`ggml_build_forward_expand(gf, ids_out)` stood inside the MTP block where the
block was, before `build_moe_ffn`'s `call early so that topk-moe can be used`
expand of `weights`. Node order is expand order, so the trace's expand pushed
SOFT_MAX, ARGSORT, VIEW, CONT first and the weights expand followed with the
rest; 0064's `ggml_opencl_can_fuse_topk_moe` wants ten consecutive nodes from
SOFT_MAX (RESHAPE, ARGSORT, VIEW, GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV,
RESHAPE) and met ARGSORT at n[1], so the MTP draft's router ran unfused with
the trace on. Per `fusion-coder-night.md` the fused kernel's softmax sum order
changes greedy texts and draft acceptance, so that drafter was not the served
one. Fix: the block moved after the weights expand; the CONT lands after the
ten nodes and the fusion survives (`ggml_can_fuse_subgraph` exempts the nodes
in `outs`; the VIEW the CONT consumes is `outs[0]`, the same placement the
cache's own `ids_out` has). Off, still no tensor, no node, no name.

**The card check's fusion line** (`code`; round 11 corrected the delivery): the
ids CONT is named `ffn_moe_ids_trace` **after** the graph callback `cb` (the
callback renames a layer's tensor `"%s-%d"`, so a name set before it is
overwritten), and the OpenCL fused dispatch prints, once per process at WARN,

    opencl: topk-moe fused in the MTP trace graph (n_expert N, n_tok T)

WARN because ggml INFO is dropped at the units' default level; once because a
per-dispatch line floods a month-long trace. The backend compares the 18-byte
prefix. That line present = the trace-on drafter's router is the served fused
one; absent = the node order broke again or the trace is off. Only the MTP
trace graph is reported.

**Red cell, extended** (`code`, seen failing): `test_mtp_route_ids_are_expanded_into_the_graph`
now asserts `ggml_build_forward_expand(gf, weights);` appears before the trace
block's start in `src/llama-graph.cpp`. On round 9's placement it failed with
the block's offset before the weights expand; green after the move.

**Round 8's "trace cost not measurable" now carries its caveat** (`code`, then
`measured-here` owed): that on arm never ran the DRAFT read-back — the round-9
buffer defect meant no read was issued, no pending record, no decode-start
synchronize — so the number excludes the read-back and the one wait. The served
A/B with the DRAFT path live is owed; command below.

## The cells and who runs them (`code`)

The recipe's ctest runs six `unit`-labelled tests and **none of this round's
python cells**: `unit` (the C++ `arcint-test`), `roundtrip` (`tests/roundtrip.sh`),
`stress` (`tests/concurrency/stress.sh`), `acceptance-enumeration`
(`tools/acceptance_manifest.py --check`), `acceptance-runner`
(`tests/acceptance/selftest.py`) and `acceptance-ab`
(`tests/acceptance/ab_selftest.py`). The RECIPE line's "6 tests passed" is
therefore **not** evidence for the trace cells; they run separately:

```bash
cd ~/src/ligence-foresight
EXPERT_ROUTE_TRACE_SRC=<the llama tree> python3 -m pytest tests/python/test_expert_route_trace.py -q   # 33 passed
python3 -m pytest tests/python/test_expert_route_trace.py -q                                          # 23 passed, 10 skipped
```

With `EXPERT_ROUTE_TRACE_SRC` unset, 10 cells skip by name (the source/compile
ones): the rotation, last-graph, pending-order, process-wide pending, split
snapshot, `advance`-order, write-fail, open-without-append, non-numeric max and
the MTP-expand cell. The other 23 are device-free replay cells. Round 10's
extended assertion is inside the MTP-expand cell, so it skips without the
variable too.

## What the card window runs for this check (tiny input, not 20k)

One lane, the `fn_sweep.py` unit flags (they include `--llama-mtp 3` and
`--llama-mtp-gguf`, so the MTP head is loaded), a **64-token** prompt, the six
answers. One fresh process, the trace on. The fusion line needs nothing
beyond the trace being on; `GGML_OPENCL_FUSE_TOPK_MOE` stays at its default.

```bash
cd ~/src/ligence-foresight
set -o pipefail
ARCINT=/path/to/the/recipe/arcint
MODEL=/flash-model/gguf-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
ASSETS=/flash-model/wp-tmp/w1/assets
DIR=/flash-model/wp-tmp/round10-on

rm -rf "$DIR"
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" mtp-trace --prompt-tokens 64 \
  --set --llama-expert-route-trace "$DIR" \
  --env LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=8 \
  2>&1 | tee /tmp/fn-sweep-mtp-trace.out
```

Checks (all are findings):

```bash
DIR=/flash-model/wp-tmp/round10-on
LOG=/tmp/fn-sweep-mtp-trace.log   # the server writes its own log; the .out holds fn_sweep's summary

# 1. the trace opened: one or more .ert1 files, no WARN from the trace itself
find "$DIR" -maxdepth 1 -name '*.ert1' | wc -l
grep -c "WARN: expert route trace" "$LOG"
#    (the fused-router line is a WARN too; it does not contain "expert route trace")

# 2. DRAFT records > 0: the replay's first line says "N MTP draft records"
python3 tools/expert_cache_replay.py "$DIR"

# 3. the mtp predictor has rows: the report has an "mtp" row beside
#    "oracle", "draft ids", "prompt prior", "learned"
python3 tools/expert_cache_replay.py "$DIR" --sweep

# 4. the fusion is intact with the trace on: exactly one fused line in the server log
#    (0 means the node order broke again or the trace was off; >1 means the once-guard is gone)
grep -c "topk-moe fused in the MTP trace graph" "$LOG"
grep -m3 "topk-moe fused in the MTP trace graph" "$LOG"
```

Checks 2–4 together are the round's gate: DRAFT records exist **and** the row
exists **and** the drafter was the fused one, with **exactly one** fused line in
the server log. The first replay line also gives
the graph kind counts; the DRAFT count must be comparable to the verify count.
A tiny prompt writes few records, so `MAX_MB=8` never rotates; rotation is
round 8's check, not this one. A `WARN` means tracing stopped mid-run: checks
2 and 3 then read a prefix, check 4 still holds for the graphs that ran.

## Owed: the served A/B with the DRAFT path live (round 8's caveat)

Round 8's on/off numbers priced a trace whose DRAFT read-back never ran. The
A/B that prices it, same lane and prompt size, two fresh processes:

```bash
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" live-off --prompt-tokens 64 2>&1 | tee /tmp/fn-sweep-live-off.out
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" live-on  --prompt-tokens 64 \
  --set --llama-expert-route-trace /flash-model/wp-tmp/round10-on 2>&1 | tee /tmp/fn-sweep-live-on.out
```

Both arms must show the MTP accept rate (the `SWEEP` line's `accept-mean`) and
the on arm must pass check 2 above (DRAFT > 0); if it does not, the A/B is a
dead-path A/B again and the price is not claimable.
