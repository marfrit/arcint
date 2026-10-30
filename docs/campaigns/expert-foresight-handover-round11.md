# expert-foresight round 11 — handover to the card window (2026-10-30)

Supersedes [round 10](expert-foresight-handover-round10.md) for the
fused-router evidence line and for the log path in checks 1 and 4; round 8's
run and its rotation checks, and round 10's DRAFT/`mtp`-row checks, still
apply. Nothing here ran on a GPU: the cards stay in the operator's announced
window; every code claim is `code` with device-free red cells, and the card
check below is `measured-here` owed.

Code: llama tree `ec08c6d` (branch `trace`), patch
`contrib/llama.cpp/patches/0069-expert-route-trace.patch` regenerated from its
diff against the arcint-series base `43fefb0` and measured to apply on that
base (`git apply --check`, `code`). The round-10 code was `7e0ba9e`. The
arcint side at the commit this handover ships; the recipe run before it:

    RECIPE e3edd88: 100% tests passed, 0 tests failed out of 6 EXIT 0 deb arcint_1.2.6-1_amd64.deb

## What changed, one paragraph each (`code`)

**HIGH 1 — the name never matched.** `ggml_set_name(ids_out,
"ffn_moe_ids_trace")` stood before `cb(ids_out, ..., il)`, and
`llama_context::graph_get_cb` renames a layer's tensor with
`ggml_format_name(cur, "%s-%d", name, il)`, so the node carried
`ffn_moe_ids_trace-<il>`. The backend then compared
`strncmp(..., sizeof("ffn_moe_ids_trace"))`, 18 bytes **with** the NUL, against
that 18-byte name - a match could never succeed. Fix: the name is set **after**
`cb`, so the node keeps the exact name, and the scan compares the 18-byte
prefix (`sizeof(...) - 1`), which also matches the callback's suffixed form.

**HIGH 2 — the line was INFO and per dispatch.** arcint routes ggml INFO to
verbose and drops it at the units' default level, and the line printed per
fused dispatch of every MTP decode (about 100 lines a second). Fix: the same
text at `GGML_LOG_WARN`, held to **once per process** by a static
`mtp_trace_logged` flag; trace off costs a static read-once env guard and a
compare.

**HIGH 3 — the handover grepped the wrong file.** `fn_sweep.py` writes the
server's own log to `/tmp/fn-sweep-<tag>.log`; the `.out` only holds fn_sweep's
summary. Checks 1 and 4 now read the `.log` (round 8's handover had it right).

**MEDIUM 4 — the scan was per fused dispatch.** With the trace on, every fused
dispatch scanned `cgraph->n_nodes`, including the target graph's 48 routers
that never hold the name, so the trace-on A/B priced the scan as "the trace".
Fix: the scan is decided once per graph (`mtp_trace_scanned_graph`; graph reuse
keeps a graph's node set) and stops entirely after the one line.

**LOW 5 — the docs.** `contrib/llama.cpp/README.md` and the campaign now say
the name is set after `cb`, the compare is a prefix, the line is a WARN once
per process, and that absence means the node order broke **or** the trace is
off. [Round 10](expert-foresight-handover-round10.md) was corrected in place
where it claimed INFO semantics and the `.out` path.

## The cells and who runs them (`code`, red mutations seen failing)

Two new cells in `tests/python/test_expert_route_trace.py`, both source cells
needing `EXPERT_ROUTE_TRACE_SRC`:

- `test_fused_router_name_survives_the_graph_callback`: red on moving
  `ggml_set_name(ids_out, ...)` above `cb(ids_out, ...)` (seen failing: the
  order assertion), and red on `sizeof("ffn_moe_ids_trace")` without `- 1`
  (seen failing: the prefix assertion).
- `test_fused_router_line_is_warned_once_per_process`: red on
  `GGML_LOG_WARN(` -> `GGML_LOG_INFO(` in the fused-router block (seen
  failing), and red on `if (!mtp_trace_logged)` -> `if (mtp_trace_logged)`
  (seen failing).

The recipe's ctest runs six `unit`-labelled tests and **none of this round's
python cells**: `unit` (the C++ `arcint-test`), `roundtrip`
(`tests/roundtrip.sh`), `stress` (`tests/concurrency/stress.sh`),
`acceptance-enumeration` (`tools/acceptance_manifest.py --check`),
`acceptance-runner` (`tests/acceptance/selftest.py`) and `acceptance-ab`
(`tests/acceptance/ab_selftest.py`). The RECIPE line's "tests passed" is
therefore **not** evidence for the trace cells; they run separately:

```bash
cd ~/src/ligence-foresight
EXPERT_ROUTE_TRACE_SRC=<the llama tree> python3 -m pytest tests/python/test_expert_route_trace.py -q   # 35 passed
python3 -m pytest tests/python/test_expert_route_trace.py -q                                          # 23 passed, 12 skipped
```

With `EXPERT_ROUTE_TRACE_SRC` unset, 12 cells skip by name (the source/compile
ones, including this round's two); the other 23 are device-free replay cells.

## What the card window runs for this check (tiny input, not 20k)

One lane, the `fn_sweep.py` unit flags (they include `--llama-mtp 3` and
`--llama-mtp-gguf`, so the MTP head is loaded), a **64-token** prompt, the six
answers. One fresh process, the trace on. The fusion line needs nothing beyond
the trace being on; `GGML_OPENCL_FUSE_TOPK_MOE` stays at its default.
The card window runs this, in its announced window:

```bash
cd ~/src/ligence-foresight
set -o pipefail
ARCINT=/path/to/the/recipe/arcint
MODEL=/flash-model/gguf-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
ASSETS=/flash-model/wp-tmp/w1/assets
DIR=/flash-model/wp-tmp/round11-on
LOG=/tmp/fn-sweep-round11-on.log

rm -rf "$DIR"
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" round11-on --prompt-tokens 64 \
  --set --llama-expert-route-trace "$DIR" \
  --env LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=8 \
  2>&1 | tee /tmp/fn-sweep-round11-on.out
```

Checks, all findings, on the server log:

```bash
DIR=/flash-model/wp-tmp/round11-on
LOG=/tmp/fn-sweep-round11-on.log

# 1. the trace opened and did not stop: one or more .ert1 files, no trace WARN
#    (the fused-router line is a WARN too; it does not contain "expert route trace")
find "$DIR" -maxdepth 1 -name '*.ert1' | wc -l
grep -c "WARN: expert route trace" "$LOG"        # expect 0

# 2. DRAFT records > 0: the replay's first line says "N MTP draft records"
python3 tools/expert_cache_replay.py "$DIR"

# 3. the mtp predictor has rows: the report has an "mtp" row beside
#    "oracle", "draft ids", "prompt prior", "learned"
python3 tools/expert_cache_replay.py "$DIR" --sweep

# 4. the fusion is intact with the trace on, and logged once:
grep -c "topk-moe fused in the MTP trace graph" "$LOG"   # expect exactly 1
grep -m3 "topk-moe fused in the MTP trace graph" "$LOG"
```

Checks 2-4 together are the gate: DRAFT records exist **and** the `mtp` row
exists **and** exactly one fused line in the server log. `0` fused lines means
the name still does not reach the backend, the node order broke, or the trace
was off; `> 1` means the once-guard is gone. A `WARN: expert route trace`
means tracing stopped mid-run and checks 2 and 3 read a prefix. A tiny prompt
writes few records, so `MAX_MB=8` never rotates; rotation is round 8's check.

## Owed: the served A/B with the DRAFT path live (round 8's caveat, still open)

Round 8's on/off numbers priced a trace whose DRAFT read-back never ran. The
A/B that prices it, same lane and prompt size, two fresh processes:

```bash
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" live-off --prompt-tokens 64 2>&1 | tee /tmp/fn-sweep-live-off.out
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" live-on  --prompt-tokens 64 \
  --set --llama-expert-route-trace /flash-model/wp-tmp/round11-on 2>&1 | tee /tmp/fn-sweep-live-on.out
```

Both arms must show the MTP accept rate (the `SWEEP` line's `accept-mean`) and
the on arm must pass check 2 above (DRAFT > 0); if it does not, the A/B is a
dead-path A/B again and the price is not claimable.
