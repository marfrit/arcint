# concept-mining 14, round 3 — handover

State: round 2's review fixes are built and tested device-free. The save path is Strata's xe rule
in Strata's unit now, the recipe runs without pytest, the replay's `profile_slots` applies the
engine's global walk, and the training capture no longer routes the gate's topics or its capital
question. Evidence: `code` for every reference claim (Strata's `sycl/src/program/generate.cpp`
`save_profile`, its per-routing-event device counter, `tools/make_profile.py`); `measured-here` for
the red cells and the recipe. No card run was made here: the served A/B and the warm-start arms
stay owed, and the GPU commands below are for Claude's announced window.

## What changed

0. **The recipe's 7th cell runs on python3 alone.** `tests/python/test_expert_profile_save.py` is
   stdlib `unittest` now (pytest can still collect it); CTest's `expert-profile-save` invokes
   `python3 <file>` with `PYTHONDONTWRITEBYTECODE=1` and `EXPERT_PROFILE_SRC`, not `-m pytest`.
   The cell skips with a message when the profile sources are absent. Verified on this host with
   site-packages disabled (`python3 -S`: `import pytest` -> ModuleNotFoundError), the same cell
   passes, skip 1 (the byte-equality import needs numpy, which `-S` has not).
1. **The xe prior's unit.** The save's `heat` is now raw routing counts, one a routed pair,
   counted in the cache's `count()` beside `usage += 1` (`llama_expert_profile_add_heat`, Strata's
   `usage_dev_srv` unit), in `double`. Round 2's `heat += usage` before each adapt decay was
   1/(1-decay) larger (3.33 at decay 0.7), so the 800,000-entry prior weighed like ~240k entries.
   The xe `WARN` line prints the routed entries counted next to the prior, as Strata's stderr line.
2. **Adapt off still counts.** The heat call lives in `count()`, not `adapt()`, so
   `LLAMA_EXPERT_CACHE_ADAPT_EVERY=0` accumulates it. A source cell asserts the location; moving
   the call back to `adapt()` makes it fail (mutation j).
3. **The save says it happened.** The success line is `WARN` (was `INFO`, dropped at the units'
   default level); the failure line stays `WARN`. The card commands read the file back, not the
   log alone.
4. **The training capture and the gate.** Five overlapping prompts were replaced: hash tables ->
   bit counting, GC -> blocked matrix multiply, compilers -> mRNA vaccine prose, CSV -> hour-bucket
   time series, WAL/B-trees -> a backtracking Sudoku solver; the compiler-adjacent
   recursive-descent parser became Dijkstra. None matches the gate's six topics (strata/CSV, hash
   tables, compilers, TCP, B-trees, GC) or another set member. `fn_sweep.py --no-capital` leaves
   the capital question out of the capture; the served arms keep it, which is what makes the gate's
   capital check shared.
5. **The replay's walk.** `profile_slots` gives each layer exactly the slots the global walk spent
   on it and sets `n_slots` to that count, instead of truncating every layer back to the trace's
   per-layer counts. A test with a 5-frame budget that does not fit the trace's 2+3 split lands
   3+2 (mutation k truncates: 2+2, one frame unspent).
6. **Patch order (documented, not rebased).** 0069 and 0070 both touch
   `llama-expert-cache.{h,cpp}` (the destructor, the layer struct). When the prefill slot loan
   integrates as its 0068, 0069 and 0070 need regeneration on top of it; the patch README and this
   handover say so.
7. **Path fix.** The audit cites Strata's `tools/make_profile.py` at upstream with the mirror's
   `~/src/strata-profile-ref/src/tools/make_profile.py` beside it.

Patch 0070 was regenerated against 0001-0067 + 0069 and applies cleanly in sequence on the pin
(`git apply`, `code`). Its new cells: heat unit (`FAIL: heat counts one a routed pair, raw` under
`+= 1.0 / (1.0 - 0.7)`) and heat width (`FAIL: heat keeps a count past float's 2^24` under a float
narrowing), both run and seen failing before restored, `PYTHONDONTWRITEBYTECODE=1`.

Recipe line and the `git log` identity are in the final reply of round 3, and in the audit §14a.

## The card window commands (Claude runs these in the announced window)

One lane, `fn_sweep.py`'s unit flags. The training capture uses the training prompts only and no
capital question; the served gate uses the six standard prompts (the unseen side).

```bash
cd ~/src/ligence-foresight
set -o pipefail
ARCINT=/path/to/the/recipe/arcint
MODEL=/flash-model/gguf-iq3s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
ASSETS=/flash-model/wp-tmp/w1/assets
TRAIN=/flash-model/wp-tmp/item14-train

# 1. the training trace, from the training prompt set only, without the gate's capital question
rm -rf "$TRAIN"
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" item14-train \
  --prompts tools/fn_sweep_prompts.json --no-capital \
  --set --llama-expert-route-trace "$TRAIN" \
  --env LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=64 \
  2>&1 | tee /tmp/fn-sweep-item14-train.out
find "$TRAIN" -maxdepth 1 -name '*.ert1' | wc -l      # >= 1
grep -c "WARN: expert route trace" /tmp/fn-sweep-item14-train.log   # expect 0

# 2. the profile from that trace only (Strata's make_profile.rank_profile)
python3 tools/expert_profile_from_trace.py "$TRAIN" --out /tmp/item14-trace.bin
python3 - <<'PY'
import struct
b = open('/tmp/item14-trace.bin', 'rb').read()
print('STRP', b[:4], struct.unpack_from('<5I', b, 4))
PY
```

The served A/B, adapt on (the default), alternated twice, then the adapt-off pair. Each arm is one
fresh process; the standard prompts are the unseen gate.

```bash
UNIT="$ASSETS/expert-profile.bin"
TRACE=/tmp/item14-trace.bin
for tag in g1-unit g1-trace g2-trace g2-unit; do
  case "$tag" in
    *unit)  SET=(--set --llama-expert-profile "$UNIT") ;;
    *trace) SET=(--set --llama-expert-profile "$TRACE") ;;
  esac
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" "$tag" "${SET[@]}" \
    2>&1 | tee "/tmp/fn-sweep-$tag.out"
done
for tag in off-unit off-trace; do
  case "$tag" in
    *unit)  SET=(--set --llama-expert-profile "$UNIT") ;;
    *trace) SET=(--set --llama-expert-profile "$TRACE") ;;
  esac
  python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" "$tag" "${SET[@]}" \
    --env LLAMA_EXPERT_CACHE_ADAPT_EVERY=0 2>&1 | tee "/tmp/fn-sweep-$tag.out"
done
grep -h '^SWEEP ' /tmp/fn-sweep-g1-unit.out /tmp/fn-sweep-g1-trace.out \
  /tmp/fn-sweep-g2-trace.out /tmp/fn-sweep-g2-unit.out \
  /tmp/fn-sweep-off-unit.out /tmp/fn-sweep-off-trace.out
```

Report per arm: decode-mean, hits-mean, accept-mean, and the answers' capital check. The gate is
the answer-level bar (the facts stay right; the six standard prompts were never trained on); the
speed comparison is decode-mean within its spread, alternated twice.

The save path's warm-start arms (also owed), xe default and cu option, distinct paths. The save
check reads the file back, and greps the WARN line (INFO is dropped at the units' default level),
including the counted entries beside the prior:

```bash
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" save-xe \
  --set --llama-expert-profile "$UNIT" \
  --set --llama-expert-cache-save /tmp/item14-saved-xe.bin \
  --env STRATA_PROFILE_PRIOR_ENTRIES=800000 2>&1 | tee /tmp/fn-sweep-save-xe.out
test -s /tmp/item14-saved-xe.bin && python3 - <<'PY'
import struct, sys
b = open('/tmp/item14-saved-xe.bin', 'rb').read()
print('STRP', b[:4], struct.unpack_from('<5I', b, 4), len(b))
PY
grep "WARN: .*saved the expert profile" /tmp/fn-sweep-save-xe.log   # names xe, counted, prior
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" restart-xe \
  --set --llama-expert-profile /tmp/item14-saved-xe.bin 2>&1 | tee /tmp/fn-sweep-restart-xe.out
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" save-cu \
  --set --llama-expert-profile "$UNIT" \
  --set --llama-expert-cache-save /tmp/item14-saved-cu.bin \
  --set --llama-expert-cache-save-rule cu 2>&1 | tee /tmp/fn-sweep-save-cu.out
test -s /tmp/item14-saved-cu.bin
grep "WARN: .*saved the expert profile" /tmp/fn-sweep-save-cu.log   # names cu
python3 tools/fn_sweep.py "$ARCINT" "$MODEL" "$ASSETS" restart-cu \
  --set --llama-expert-profile /tmp/item14-saved-cu.bin 2>&1 | tee /tmp/fn-sweep-restart-cu.out
```

The warm-start gate: a restart on the saved file is at or above the arm it was saved from, within
the window's spread, and the answers stay right. `xe` is the default; `cu` is named only in the
`save-cu` arm, so the record can say which rule each arm used. A save made with adapt off is the
case item 2 exists for: the file must carry counts, not an empty ranking.

## Owed

- the training capture, the trace-built profile and the four alternated arms plus the adapt-off
  pair;
- the warm-start arms above;
- the served confirmation of item 6: when the slot loan lands as 0068, regenerate 0069 and 0070
  on top of it.
