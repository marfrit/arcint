# expert-foresight — stage the experts the next tokens will need, before they need them

**Status: open (operator, 2026-10-29).** Flash-Next on the B60; the A770 or a CPU core as a predictor's home.

## The lever

The expert cache decides which ~24 % of Flash-Next's experts sit in the B60's slots (5,915 slots at the unit's 11,600
MiB, ~123 of 512 a layer) from **past** routing. Replaying the 2026-10-15 routing trace through the cache's own
mechanism (`research-strata-history.md`, `measured-here` offline):

- every past-based policy (Strata's adapt, LFRU, windows, the best fixed set in hindsight) lands at ~89-90 % of
  decode's routed experts on the card;
- a planner that **knows** the next 24 tokens' routing reaches **96.6 %** inside the same cadence, swap cap and
  landing (next 6: 91.8 %, next 96: 97.6 %; no cap: 99.8 %).

Those ~7 points are foresight. On the card, the expert share and decode move together (today: 71.5 % -> 81.2 % of
routed experts came with 31.08 -> 34.30 t/s, `measured-here`, `research-concept-mining.md`), so a predictor that
recovers part of the oracle's 7 points is worth several percent of decode.

## Where foresight can come from

1. **The MTP window.** The verify pass computes every drafted token's routing at every layer; the drafts are the
   model's own guess of the next tokens. Free, already on the card.
2. **The prompt.** The experts a prompt routes most are likely to recur in its answer. Free at decode's start.
3. **A small learned predictor** ("a matrix, no LLM magic"): per layer, recent routing (+ 1, + 2) -> the expert
   probabilities for the next ~20 tokens; a linear map or a tiny MLP, trained offline on our own traces. Its output
   is a swap plan of a few hundred expert ids, so it is **off the critical path**: late means today's behaviour, not
   a stall. That makes the A770 (or a CPU core) a natural home: the B60 serves, the predictor plans.

Prior art: Strata predicted one layer ahead within a token (recall 27-34 %, precision ~8 %) and dropped its
foresight slots (decode x0.91-1.00, `paper`, 14f9e4c5, ef362cc4); cross-token foresight is untested by Strata and by
us. Published work on learned expert prefetching for MoE offloading exists; survey it (and read the code where there
is code) before stage 3.

## Stages

0. **Trace:** a diagnostic patch that records decode and prefill routing (and the MTP drafts' routing) per token and
   layer from the served unit (the 2026-10-15 trace knob is not in the served series any more).
1. **Replay with predictors** (`tools/expert_cache_replay.py`): past-based (the baseline), MTP-draft routing, prompt
   prior, both, and a trained linear/small model; hit rate inside the real swap budget, against the oracle. Train on
   some requests, score on others.
2. **Build what the replay justifies**: the planner fed with predicted routing; the predictor where its measured cost
   is lowest (B60, A770 or CPU).
3. **Served A/B** under the adoption rule.

Alongside: re-measure the two-card layer split on today's stack (`layer-split-two-cards.md`; CPU-computed misses
arrived after its 2026-10-08 numbers and keep the A770's misses off its x4 link).

## Continuous trace (operator, 2026-10-29)

The learned map and the token -> expert table were fitted on 5 requests (~400 steps, `measured-here`); a stable
workload logged over weeks would give them orders of magnitude more data, and a per-workload expert profile (audit
candidate 14) needs only routing counts. Operator decision:

- **An option, off by default.** The served unit may run with a continuous routing trace (the 0069 recorder made a
  production option: rotation, a size cap, no extra decode sync).
- **Record the detail, aggregate later.** The raw records (token ids, routed ids per layer, the MTP window's drafts
  and acceptances) are kept, not decayed counts: "I'd rather have a bit more detail than needed to aggregate later
  than finding out after a month the statistics were optimized too early." Aggregation (decayed counts, profiles,
  training sets) is an offline step over the raw log.
- **The documentation warns:** a trace with token ids is the prompt and the answer in another form. Whoever enables
  it stores every request's content on the serving host; the option's help text and the README section say so.

Measurement before building on it: held-out hit rate (and the learned map's recall) against training-set size, from
traces captured over several days; the oracle ceiling bounds the prize.

## Gate (operator's rule, 2026-10-29)

Adopted if decode (or prefill) rises at all and no other target variable degrades: the other phase within its
run-to-run spread, the answers at `CLAUDE.md`'s answer-level bar, peak VRAM and context unchanged.

## The operator's decision: the trace is an option, not a diagnostic switch (2026-10-29, quoted)

> "I'd rather have a bit more detail than needed to aggregate later than finding out after a month the
> statistics were optimized too early."

So the routing trace is a **production option**, off by default, and what it writes is raw: every request's
records stay as written, nothing is decayed or pre-aggregated on the serving path. Aggregation is an **offline
step** over the files (`tools/expert_cache_replay.py` and whatever comes next), run wherever the operator wants,
so a statistic not yet thought of can still be computed from a month of traces. Consequences recorded here:
`--llama-expert-route-trace DIR` (arcint), one file a process, rotated at a size cap with a per-file header and a
resident snapshot, nothing deleted by arcint, the last graph of every request written at its end. The size cap is
the operator's disk, not a retention rule. The warning stands in the flag's help and the README: a trace with
token ids is the prompt and the answer in another form — whoever enables it stores every request's content on the
serving host.

## What we learned and what would change it

Every outcome is recorded here, with its evidence class: what a predictor reached against the oracle, what it cost,
and the conditions under which a result would change - more VRAM (a larger card raises the slot share and the
baseline), a faster link for a second card (the A770 sits on PCIe 3.0 x4, 1.83 GB/s), a larger swap budget, or a
better predictor. A ceiling found here is a boundary with its numbers, for whoever comes back with the hardware or the
idea that moves it.

### Round 3 fixes, before the real-trace run (2026-10-29, `code`)

- **The accept sidecar is aligned per request.** Every request restarts at position 0, so the same position recurs in
  several requests; the old global position set refused the 9-request trace at `two lines for position 71`. The replay
  now walks the lines in request order and checks each against its graph's position **and** row count, so a prompt's
  checkpoint tail (a PROMPT record, no line) cannot shift the walk and an out-of-step line fails loudly. A process's
  last graph is not written (below), so one trailing line is kept apart as that graph.
- **A prompt's last-token alone-decode is relabeled.** arcint's non-spec loop prefills all but the last prompt token
  and runs the last one alone for its logits (`src/exec/backend_llama.cpp`, `decode_locked(..., want_last)`), so a trace
  records it as a 1-token DECODE; the sampler writes no accept line for it. The replay relabels a 1-token DECODE that
  sits at the preceding PROMPT's end and has no line as that prompt's last token, keeps it out of the scored steps and
  reports the count. On a trace with no sidecar the relabel is applied only when no VERIFY graph exists (a non-spec
  run): in a speculative trace a 1-token DECODE is a one-row verify and the adjacency alone cannot tell them apart.
  Test: `test_prompt_last_token_decode_is_relabeled`.
- **The default report is held out.** The token table is fitted on the even requests. The replay now scores the odd
  requests whenever they exist and never prints an all-steps section beside them; with no odd request it scores all
  steps and marks every fitted row (foresight, both, learned) as `(in-sample)`. `--sweep` does the same.
  Tests: `test_default_report_is_held_out`, `test_sweep_marks_one_request_in_sample`.
- **The kind decision has a Python mirror.** `route_graph_kind` in the replay is the serializer's decision
  (`llama_expert_route_graph_kind`), tested without `EXPERT_ROUTE_TRACE_SRC`: the 4-token prompt tail is PROMPT, a
  2..decode_max all-outputs call VERIFY, one token DECODE, a longer batch, a non-all-outputs call and an embedding
  batch PROMPT. Test: `test_kind_decision_mirrors_the_serializer`.
- **A trace ends one step short.** The cache writes a graph when the *next* graph's `advance` runs, so the process's
  last graph is never written (`src/llama-expert-cache.cpp`). The replay scores the written steps and keeps the
  sidecar's extra last line apart; the note is printed. This was a trace-property of the round-3 build, not a replay
  choice; round 5 flushes the last graph at the request's end, so a traced run no longer ends one step short.

### Round 3 — the served public-corpus trace (2026-10-29, `measured-here`, offline replay)

Trace: the B60, Flash-Next IQ3_S, the unit's flags, the slot loan off, built at `937145d`; 79.5 MB ERT1 plus its
accept sidecar, served at 34.40 t/s decode and 81.2 % of routed experts on the card. It is our own routing on public
corpus text, so no data rule constrains it. The file has 48 layers, 512 experts, top-10, 5,915 slots (93-155 a
layer), 842 graphs: 36 prompt, 101 one-token decode, 705 verify, 0 draft. 806 decode steps are written (1,265,280
routed pairs); the process's last graph is the one missing step. The replay's baseline rebuilds the trace's own
resident set 5,915 of 5,915 slots (100.0 %, a fidelity line). Accept alignment: 0 prompt-last relabels on this trace
(its 101 one-token decodes are one-row verifies with sidecar lines); 1 trailing line for the unwritten last graph.

Requests, in order: 9. Even (training) requests 0, 2, 4, 6, 8: the 24-token capital question (1 step), three of the
six answers (153/123/121 steps, 558/415/399 step tokens), the second long prompt (9 steps). Odd (held-out) requests
1, 3, 5, 7: the first long corpus prompt (12 steps, 29 step tokens), the other three answers (117/133/137 steps,
383/412/410 step tokens). Split: **407 training steps (672,960 routed pairs), 399 held-out steps (592,320 routed
pairs)**. The report below is the held-out set only, delay 2, 96 swaps a plan every 6 tokens.

| predictor (held-out, 399 steps) | hits | swaps | against the baseline |
|---|---:|---:|---:|
| strata (the baseline) | 83.88 % | 12,819 | — |
| foresight: draft ids -> experts | 61.37 % | 14,112 | −22.51 points |
| the prompt's routing as a prior | 53.23 % | 4,753 | −30.65 |
| both | 61.37 % | 15,840 | −22.51 |
| learned linear map (6 coefficients a layer, H=24) | 80.28 % | 16,128 | −3.60 |
| oracle, next 24 tokens | 85.87 % | 15,690 | **+1.99** |

Horizon sweep, same held-out steps (`--sweep --learned`):

| predictor / oracle | H=6 | H=12 | H=24 | H=48 |
|---|---:|---:|---:|---:|
| strata (the baseline) | 83.9 % | 83.9 % | 83.9 % | 83.9 % |
| foresight: draft ids -> experts | 61.4 % | 61.4 % | 61.4 % | 61.4 % |
| the prompt's routing as a prior | 53.2 % | 53.2 % | 53.2 % | 53.2 % |
| both | 61.4 % | 61.4 % | 61.4 % | 61.4 % |
| learned linear map | 80.3 % | 80.3 % | 80.3 % | 80.3 % |
| oracle, next H tokens | 78.1 % | 82.4 % | 85.9 % | 89.3 % |

What the numbers mean for building (`measured-here`):

- **No predictor recovers the oracle's gap on this trace; they all lose to the usage baseline.** The positive oracle
  gap is small here (+1.99 points at 24 tokens, +5.4 at 48), unlike the 2026-10-15 trace's ~7 points. Foresight reads
  the draft ids only (a 3-4 token window), the prompt prior is a constant per-request distribution, and neither is
  precise: over the held-out top slots, foresight is 40.4 % recall / 24.3 % precision, the prompt prior 38.5 / 23.3,
  both 45.3 / 27.2. The plan has no gain floor and no usage fallback, so a mediocre prediction actively evicts
  usage-hot residents for predicted-hot non-residents — degradation, not foresight. The linear map is close
  (80.3 %) but underfits: it does not improve with a longer target (same at H=6 and H=48).
- **Even a perfect short-horizon planner loses to the baseline inside this cadence.** The exact oracle at H=6 is
  78.1 % and at H=12 82.4 %, both below 83.9 %; it overtakes only at H>=24. So the MTP window (about three drafts)
  cannot pay by construction here, and the lever worth building is not "the drafts' routing" but a predictor with a
  window of at least ~24 tokens that ranks the held-out slots above the baseline's usage set.
- **The next test is a confidence/gain floor with a usage fallback** in the plan rule, replayed on this same trace;
  a predictor that cannot show its held hits above 83.88 % toward 85.87 % inside the same swap cap does not justify
  the A770/CPU home in stage 2. The oracle's 85.87 % is also the ceiling the mechanism allows here: the swap cap and
  the 6-token cadence, not prediction quality, bound the remaining ~2 points.
- **What would change it:** a larger slot share (more VRAM raises the baseline), a larger swap budget or a shorter
  cadence (the oracle's ceiling moves), a longer verified span (more drafts per verify lengthen the free window), or a
  learned predictor whose top-slot precision clears the baseline. The 2026-10-15 trace's 7-point gap does not
  reproduce on public corpus text; a result here is a boundary with its numbers, not a verdict on foresight.

### Round 4 — the blend: use stays the base, prediction only adds a conditional bonus (2026-10-29, `measured-here`, offline replay)

Trace and split: the same served public-corpus trace (B60, Flash-Next IQ3_S, `937145d`, 79.5 MB ERT1 + accept
sidecar), 407 training steps (even requests) and 399 held-out steps (odd requests), delay 2, 96 swaps a plan every 6
tokens, 5,915 slots. Nothing was read from the host or the card for this round; the replay is the same offline walk as
round 3.

The round-3 failure was structural: every predictor **replaced** the usage ranking, so a mediocre prediction evicted
usage-hot residents. The round-4 rule (`blend_plan` in `tools/expert_cache_replay.py`, policy `simulate(..., "blend")`)
keeps usage as the base and adds only a conditional bonus:

- `share = normalize(prediction)` a layer (sums to 1, so layers are comparable);
- `bonus = weight * share * max(usage)` a layer, **only** where `share >= threshold`; an expert is worth at most
  that weight times the layer's hottest resident. This puts the bonus in usage's own units, so weight 0 replays
  the usage plan exactly;
- candidate admission is the baseline's `usage >= 2` **or** a bonus above zero (a confidently predicted cold
  expert may compete);
- a swap runs only while `score[candidate] - score[resident] > margin`, with the engine's own gain `1.5` as the
  margin; a guess below the threshold adds nothing, so it cannot evict a hot resident.

The oracle is blended through the same rule, so its number is the mechanism's ceiling **inside** the blend rule,
not the rule-free planner's ceiling.

Tuning protocol: the threshold/weight/margin/H grid was selected jointly on the **even (training) requests** and
the odd requests were scored once with the selected cell (`--blend-sweep`, `--blend-tune even`). Comparing against a
selection on the odd requests, the cells move only within the grid's ties: foresight 83.99 either way (weight 1.0 vs
0.3), learned 83.98 vs 84.03, prompt and both unchanged, the blended oracle 89.30 at H=24 in both protocols. The
predictor rows are therefore stable, and the oracle's ceiling below is selected out of sample.

| predictor (held-out, 399 steps) | best blend (threshold, weight, margin, H) | hits | swaps | against the baseline | against the blended oracle |
|---|---|---:|---:|---:|---:|
| strata (the baseline) | — | 83.88 % | 12,819 | — | — |
| foresight: draft ids -> experts | 0.005 / 1.0 / 1.5 / 24 | 83.99 % | 13,820 | **+0.11** | −5.31 |
| the prompt's routing as a prior | 0.005 / 0.1 / 1.5 / 24 | 83.39 % | 13,379 | −0.50 | −5.91 |
| both | 0.005 / 0.1 / 1.5 / 24 | 83.66 % | 13,427 | −0.23 | −5.64 |
| learned linear map | 0.005 / 10.0 / 1.5 / 24 | 83.98 % | 16,002 | **+0.10** | −5.31 |
| blended oracle (the upper bound inside the rule) | 0.005 / 10.0 / 1.5 / 24 | 89.30 % | 14,459 | **+5.41** | 0 |

The rule-free oracle on the same held-out steps, for the ceiling comparison: H=24 85.87 % (+1.99), H=48 89.34 %
(+5.46), H=96 91.18 % (+7.30), all with 15,235-15,690 swaps. The blend rule raises the oracle at H=24 (85.87 ->
89.30) because the usage base still orders what the prediction leaves tied, but at H=48 and H=96 the rule-free
oracle stays ahead (89.34 and 91.18 against the blended best 89.30): the threshold and the 1.5 gain margin cost the
long window its aggressive swaps. The oracle's best horizon inside the rule is therefore H=24.

**Does any blend beat the baseline on held-out steps?** Yes, two, by very little: foresight **+0.11 points** (83.99
against 83.88, about 650 of the 592,320 routed pairs) and the learned map **+0.10 points** (83.98, about 590 pairs).
The prompt prior and both stay below the baseline (−0.50 and −0.23). The oracle's +5.41 is the same rule with the
true future and is not a predictor result. No blend recovers even a tenth of the oracle's gap; the confidence/bonus
rule stops the *degradation* of round 3 (foresight 61.37 -> 83.99, learned 80.28 -> 83.98) but does not turn a
weak predictor into foresight.

**What would change it:** a predictor whose top-slot precision clears the baseline's (both still pay more swaps:
foresight 13,820, learned 16,002 against 12,819), a longer verified span, a shorter cadence or a larger swap cap
(the oracle's inside-rule ceiling is set by those, not by prediction quality), or a larger slot share. The margin
3.0 was measured and is worse on every arm tested (e.g. the blended oracle at H=24, weight 10: 87.31 against 89.30),
so the engine's 1.5 gain stands. The stage-2 gate (decode rises, nothing else degrades) still needs a served A/B;
+0.1 point of offline hit rate is not that evidence.

### Round 5 — the trace as an option, DRAFT records, a multi-file history and the training curve (2026-10-29, `code` + `measured-here`)

Built: the production option (above), DRAFT records from the MTP block's own
routing (its fact: the section below), the replay reading several files as
one ordered history (a directory or a file list, rotated names sorted by
mtime then name; a later file's LAYERS snapshot updates the first file's
residency, a request that spans a rotation stays one request), and
`--train-curve`.

The MTP predictor (`mtp` in `providers`, blended as round 4) maps a verify's
draft tokens through the **MTP block's own routing** of those draft tokens,
taken from the DRAFT records of the training requests; per target layer, the
same expert ids. It adds no row when the trace has no DRAFT record, which is
the case for the 2026-10-29 trace (captured before DRAFT records existed;
served copy below is not re-captured).

**The training curve on today's single trace** (`measured-here`, offline;
`python3 tools/expert_cache_replay.py <trace> --train-curve`). Same trace,
split and mechanism as round 4: 399 held-out decode steps of the 4 odd
requests, 407 training steps of the 5 even requests; sizes 1, 2, 4, 5
training requests (chronological); each predictor's blend cell chosen on the
training steps, the odd steps scored once per size.

| training requests | training steps | baseline | draft-id table, blended | learned map, blended |
|---:|---:|---:|---:|---:|
| 1 | 1 | 83.88 % | 83.87 % (0.005/0.10/1.5) | 83.88 % (0.005/0.10/1.5) |
| 2 | 154 | 83.88 % | 83.86 % (0.005/1.00/1.5) | 83.88 % (0.005/1.00/1.5) |
| 4 | 398 | 83.88 % | 83.99 % (0.005/1.00/1.5) | 83.97 % (0.010/1.00/1.5) |
| 5 | 407 | 83.88 % | 84.02 % (0.005/1.00/1.5) | 83.98 % (0.010/1.00/1.5) |

The curve is short because this trace has nine requests: the first size is
one step, then 154, 398, 407. The direction is the recorded one: the draft-id
table moves 83.87 -> 84.02 as training grows past four requests, the learned
map 83.88 -> 83.98; both stay within a tenth of a point of the baseline, as
round 4's full-training rows. A longer curve needs more captured requests,
which the production option will now collect.

Device-free cells added with the round:
1. **Rotation**: a 314-byte cap splits one event stream into two files; each
   file replays alone (`--single`), the second file's LAYERS snapshot carries
   the residency at the rotation, and the concatenation equals the unrotated
   trace. Red mutation, seen failing: the rotated writer's `at_cap` never
   fires (one file, the "two files" assertion).
2. **The last graph at the request's end**: a complete trace has the decode at
   position 203 and the replay reports no unwritten last graph; the red case
   (`EXPERT_ROUTE_SELFTEST_SKIP_LAST`) drops that emit and the accept sidecar's
   line is left apart (unwritten last graph 1).
3. **DRAFT round trip**: the fixture's DRAFT record at 204 carries the drafted
   tokens `[4000, 4001]` and the block's ids in rank order; `route_to_steps`
   keeps it out of the scored steps, and the reader rebuilds the bytes.
4. **The curve**: a synthetic trace where the token table from the first
   training request already predicts every held-out draft exactly; the curve's
   training steps must grow and the held-out hits must not fall below the first
   row. Red mutation, seen failing: fit a size on the last `len - n` training
   requests (`max(1, ...)`) — the steps column becomes 24, 12, 12.

## Stage 0 — the trace, now a production option (2026-10-29, `code`)

`contrib/llama.cpp/patches/0069-expert-route-trace.patch` in the served series;
arcint's `--llama-expert-route-trace DIR` sets `LLAMA_EXPERT_ROUTE_TRACE`
before the model's load. Unset: one `getenv` at the cache's init, one cached
bool the graph builder and the accept writer ask, no allocation, no read-back,
no write. **Off is the default** (operator, above).

- **One file a process, rotated.** `DIR` gives
  `expert-route-<pid>-<unix-time>.000.ert1`, `.001.ert1`, ...; a cap at
  `LLAMA_EXPERT_ROUTE_TRACE_MAX_MB` (default 256; a non-numeric value falls
  back with a `WARN`). Every file repeats the header `ERT1`, version, the
  model's `n_layer`/`n_expert`/top-k, and after a rotation a `LAYERS` record of
  the residents at that moment, so **each file replays alone**; concatenating
  the files gives the unrotated trace's records. Nothing is deleted by arcint.
  Files are `O_EXCL` and never appended; a missing `DIR` is created or refused
  (`code`). Tests: `test_rotation_each_file_replays_alone_and_concatenation_equals`,
  `test_open_never_appends_and_creates_a_missing_directory` (red: no `O_EXCL`).
- **The last graph is written at the request's end.** The cache writes a graph
  at the next graph's `advance`, so the process's final graph was absent
  (round 3's "one step short"). `llama_expert_route_trace_flush` now writes
  the pending last graph from arcint's generate (on every return and on a
  throw) and from the backend's destructor: one sync, only when the trace is
  on and a graph is pending. Round 3's note no longer applies to a run with
  the flush. Test: `test_last_graph_written_at_request_end`, red mutation the
  selftest's `EXPERT_ROUTE_SELFTEST_SKIP_LAST` (the sidecar then has a line
  with no graph, the replay reports the unwritten last graph).
- **Cost (`code`).** A decode step writes the ids the cache already reads back
  for its usage counts (`ids_pending`, the read issued behind the graph by
  `after_graph`), so no new device sync; `LLAMA_EXPERT_CACHE_ASYNC_IDS=0` is
  respected, with the wait it asks. A prompt batch is not counted by the cache,
  so a traced prompt gets one host read and one wait at the next ubatch
  boundary. An MTP DRAFT record is an async read beside the logits copy,
  written at that context's `synchronize`; a decode that starts before a
  synchronize waits once. The request-end flush is one sync, taken under `mu_`,
  only with a pending graph. A failed write logs one `WARN` and stops the
  read-backs; no read-back continues. A record is a few KB in a small window,
  but a prompt record is `n_tokens × top-k × n_layer × 4 B` (~1.9 KB a prompt
  token; 100k ~190 MB), so DIR belongs on a dataset with room. **Off: none of
  these paths run.**
- **MTP drafts are recorded.** An MTP draft context has no expert cache, so its
  block's `selected_experts` is a graph output while the trace is on
  (`llama-graph.cpp`, `llama_context` reads it back), and a `DRAFT` record
  carries the MTP block's INPUT tokens (step 0 the accepted target token, then
  the previous draft) — not the drafted tokens — and the block's routed ids
  (`code`). The accept sidecar follows the current trace file
  (`<file>.accept`), and the cache rotates at `after_graph` before the line is
  written, so the line and graph record stay in one file (`code`; red: write
  the line before the rotation, `test_rotation_keeps_the_accept_line_and_graph_in_one_file`).
- **What it does not cover.** A two-card layer split writes one LAYERS record
  and one graph record a card; the replay merges the graphs and refuses the
  residency (one card's trace is what it scores). A multi-lane run scrambles
  the accept walk (refused by the replay). The 2026-10-15 trace knob
  (`ECT1`) is still read by the replay for old files, not served.

### Round 6 — the trace safe for a month (2026-10-30, `code` + red tests)

Round-5 review findings on the continuous log, all closed here. Every claim is
`code` unless marked; the device-free cells are in
`tests/python/test_expert_route_trace.py` and each named mutation was seen
failing.

- **No append, a fresh name, a real DIR.** `O_EXCL` and `expert-route-<pid>-
  <unix-time>`; an existing file base advances `.001`. A missing DIR is
  created or refused with a `WARN`. Several processes' files replay as one
  history. Red: drop `O_EXCL` and the file base gains a header.
- **Reader resync.** An old file that got a second header at its tail replays:
  on a record error the reader finds a later `ERT1` header with the same
  geometry and continues. `resynced` reports it. Red: drop the search, the file
  stops at "unknown record kind 69".
- **Rotation before the accept line.** `trace_prepare` now runs at
  `after_graph`, so a rotate and its `LAYERS` snapshot precede the verify's
  line; the line and graph record land in the new file. The rotation cell
  forces the rotation at a line graph; red: line before rotation, the new file
  has a graph and no line.
- **Flush under `mu_`.** The request-end `TraceFlush` takes the backend's
  `mu_`, like every other `llama_context` call; `generate_spec`'s catch
  releases it first.
- **Flush and read-backs after a failed write.** One `WARN`, `trace_on`
  cleared, no further ids reads. The size doc: prompt `n_tokens × top-k ×
  n_layer × 4 B`.
- **`ASYNC_IDS=0` respected** on the trace path; the ids are read with the
  wait the switch asks, and `count` uses them without a second copy.
- **MTP predictor is a fitted map.** `mtp_expert_matrix` pairs the DRAFT input
  at t with the verify's target row t+1, per target layer, fitted only on even
  requests; no identity. A verify without a matched DRAFT gets zeros. **No
  serving result tests this mechanism yet**: no trace with DRAFT records has
  been replayed. Red: identity map, or fitting all requests (`mat[0,0,3]`
  gains the odd pair).
- **Replay lows.** A directory spanning an unclean restart tolerates one
  missing last graph a file (not only at the end); a later `LAYERS` updates
  `residents`; `.000` finds `.001` siblings; `--accept DIR` reads its
  sidecars; `--single DIR` is refused; an invalid max falls back.
- **DRAFT read is async**, beside the logits copy, written after the context's
  synchronize; if a caller decodes before a synchronize, decode's start takes
  one wait. The A/B prices it. **(corrected in round 7: the Async API call used
  there is a blocking fallback on OpenCL, `code`)**
- **Still device-free unexplained:** the engine-path cells (`TraceFlush`
  present, a `LAYERS` record in `.001` after a rotation, sidecar lines a file,
  `unwritten_last_graph == 0`) need Claude's card A/B; the handover says what
  to read.

### Round 7 — the three residuals of the round-6 review (2026-10-30, `code` + red tests)

All three are engine-path findings (`code`) found by reading the round-6 diff
against the backend sources; each has a device-free red cell, and the served
A/B below is still the open `measured-here` gate.

- **The handover's on arm ran with the trace OFF (`code`).** It passed
  `--env LLAMA_EXPERT_ROUTE_TRACE="$DIR"`, and the backend unsets that
  variable whenever `--llama-expert-route-trace` is empty
  (`src/exec/backend_llama.cpp:391-392`, before the model's load), so the arm
  was the off arm with an inherited variable removed. The corrected arm sets
  the flag (`--set --llama-expert-route-trace "$DIR"`); the MAX_MB stays an
  env. The round-6 handover is amended and round 7's handover carries the
  command; a run whose DIR holds no `.ert1` is an untraced arm, not a clean
  one.
- **The DRAFT read was not async on OpenCL (`code`).**
  `ggml_backend_tensor_get_async` is a synchronize plus a blocking get on this
  series' OpenCL backend: the backend's `iface.get_tensor_async` slot is NULL
  (`ggml-opencl.cpp`, the iface table), beside the empty stub
  `ggml_backend_opencl_get_tensor_async`; `ggml_backend_tensor_get_async` in
  the core then takes its NULL-slot path (a synchronize and a blocking get),
  so the round-6 claim that the read overlapped the draft graph was false.
  The line numbers belong to a tree not checked out here, so the functions are
  named (`code`). The MTP context now reads the
  ids through the backend's own batch hook,
  `ggml_backend_reg_get_proc_address(reg,
  "ggml_backend_opencl_get_tensors_async")` (the entry point
  `llama-expert-cache.cpp:314` already uses for the cache's ids), grouping the
  route tensors by backend so the hook's one-queue assertion holds; the dsts
  live in the pending record until the synchronize that writes it. A backend
  without the hook keeps the blocking call. **What still prices it is the A/B**:
  no card ran here (`code`).
- **The accept line split when DRAFT records crossed the cap (`code`).**
  Round 6 rotated in `after_graph`, before arcint's accept line -- but the
  record is written only at the *next* `advance`/flush, and an MTP DRAFT record
  (written at that context's synchronize, which does not prepare) could push
  the file over the cap in between. `advance()` then called `trace_prepare()`
  again, rotated, and put the verify's record in the new file while its line
  stayed in the old one. Fix: the rotation decision is the pure function
  `llama_expert_route_trace_rotate_now(t, graph_pending)` -- never while a
  graph record is pending -- and `advance`/`flush_trace` no longer prepare; a
  rotation therefore waits for the next `after_graph`. Red cell
  `test_pending_record_never_rotates_away_from_its_accept_line` drives the
  cache's real order in the selftest (after_graph, line, four DRAFT records
  past the cap, advance) and has a red arm that rotates while pending. Red
  mutation, seen failing: drop `graph_pending` from the decision -- the green
  arm rotates at advance (three files, the line in `.000` and its record in
  `.001`).
- **The write-failure path has a red cell (`code`).**
  `test_write_failure_stops_tracing_with_one_warn` closes the trace's fd in the
  selftest: one `WARN`, `opened` cleared, no sidecar, and the next record is a
  no-op. Red mutation, seen failing: drop `opened = false;` in the writer's
  error path -- the arm prints `write-fail opened 1`.
- **Two stale comments (`code`).** The header said "an existing non-directory
  is the first file" (it is a base; the existing file is skipped by `O_EXCL`
  and the sequence starts at `.001`), and `flush()` said "O_APPEND keeps it
  whole" (the file is opened `O_WRONLY|O_EXCL` with one writer, short writes
  resumed in a loop). Both corrected in `llama-expert-route-trace.{h,cpp}`.

The card window's exact commands and post-run checks (the trace by its flag,
the unit's served file and asset directory, no `.ert1` = an untraced arm):
[expert-foresight-handover-round7.md](expert-foresight-handover-round7.md).

### Round 8 — the guard is process-wide, the rotation carries every card (2026-10-30, `code` + red tests)

Round 7's guard was one cache's `trace_pending`, and `trace_prepare()` is also
reached from `progress()` inside `advance` and from `adapt()` via `plan`. With a
cache a card (`llama-model.cpp`'s one cache per GPU that runs MoE layers) cache
A wrote its record and rotated at its `progress()`, while cache B's record for
the same graph was still pending: B's record landed in `.001` and the accept
line in `.000`. The fix, `code`:

- **A process-wide pending count in the trace singleton.** `after_graph`
  increments it when the graph's record becomes pending; the record write in
  `advance`/`flush_trace` decrements it, and the write-failure path releases it.
  `llama_expert_route_trace_rotate_now(t)` tests that count, so no cache rotates
  while any cache's record is pending. Red cell
  `test_layer_split_pending_is_process_wide_and_snapshot_has_every_card`; red
  mutation, seen failing: `rotate_now` ignores the count -- the selftest exits 8
  or 9, the first prepare rotates while A's record waits.
- **A rotation's LAYERS record carries every cache's residents.** Each cache
  keeps its current set in the trace singleton (`set_residents`, refreshed at
  init and at every TABLES record), and `trace_prepare` writes one
  `event_layers_all` record over all of them. Before this, a file started from
  the rotating cache's layers only and could not replay alone under a split.
  The replay's reader also merges several LAYERS records in one file into one
  cache list, so the first file replays alone too. Red mutation, seen failing:
  `event_layers_all` writes only the last registered set -- the split cell reads
  two layers where four are expected.
- **Round 7's cells mirrored the call order instead of running it.** Restoring
  `trace_prepare()` into `advance()` passed all 30 cells. `advance` must never
  prepare (the selftest's order cannot see it), so
  `test_advance_never_prepares_and_after_graph_does` reads the cache's source,
  names the function landmarks, and asserts `advance` has no `trace_prepare`
  call, `after_graph` has exactly one, and the decision tests the process-wide
  count. Red mutation, seen failing: insert `trace_prepare();` before
  `progress();` in `advance`. The selftest's pending cell keeps its own red arm,
  but it drives an emulated order; this source cell is the one that covers the
  real call sites, and the round-7 handover no longer claims the selftest does.

The card window's exact commands and post-run checks (the trace by its flag,
the unit's served file and asset directory, fn_sweep's stdout teed so its own
`==` line and the prefill rate are recorded, no `.ert1` = an untraced arm):
[expert-foresight-handover-round8.md](expert-foresight-handover-round8.md). A
layer-split A/B for later is in the same handover (`code` commands; no run yet).


The MTP block is **a full MoE layer**, not a dense head:

- the loader creates it with `load_block(il, mtp_flags)` for `il = n_layer`,
  which creates `ffn_gate_inp`, `ffn_down_exps` and the gate/up expert tensors
  (`src/models/qwen4exp.cpp:289-291`, `:276-278`);
- the MTP graph calls `build_layer_ffn(cur, il)` (`:597`), which asserts
  `ffn_gate_inp` and calls `build_moe_ffn` with `n_expert`, `n_expert_used`
  (`:1163-1180`);
- the served MTP-only GGUF carries them: the quantize recipe names
  `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` (`tools/flash_next_mtp_gguf.sh`,
  the 31 `mtp.*` tensors of the checkpoint).

It runs **without an expert cache** (the MTP model is loaded with default
params and every tensor on its card, `src/exec/backend_llama.cpp`), so its
routing was invisible to the cache; the DRAFT record above is how it is
captured now. The draft model does not compete for the target's slots.

### Round 8's card window (2026-10-30, `measured-here`)

One unit, the round-8 build, Flash-Next IQ3_S, `--llama-mtp 3` with
`--llama-mtp-gguf <the MTP head>`, a small prompt a request: decode with the
trace off 33.33 / 34.30 t/s and on 33.77 / 34.47; prefill off 436.9 / 442.2
and on 441.5 / 441.6; the MTP accept rate 63.6 - 66.6 % on both arms. Trace
cost is **not measurable at this spread** (the on/off difference is smaller
than the arm-to-arm spread), so no price is claimed — **but that on arm never
ran the DRAFT read-back**: the round-9 defect (the MTP ids' buffer was null)
meant no read was issued, no pending record was written and decode's start
never synchronized, so the on arm paid only the graph builder's cached bool
and its records. The read-back's cost is therefore **not** in that number
(`code`: the read-back is issued behind the graph and written at a
synchronize; round 9 fixed the buffer). The served A/B with the DRAFT path
live — the trace on and off, DRAFT records > 0 on the on arm — is **owed**
(`measured-here`, Claude's window; the command is in
[expert-foresight-handover-round10.md](expert-foresight-handover-round10.md)).
The on arm at
`LLAMA_EXPERT_ROUTE_TRACE_MAX_MB=8` wrote five files (12, 12, 9, 8, 2 MB) with
an accept sidecar beside each of the three files that hold verify graphs, and
no `WARN`. The directory replays: 848 graphs, 823 decode steps, held-out
baseline 84.24 %, draft ids 70.32 %, prompt prior 56.11 %, oracle H24 88.46 %;
each file replays alone with `--single` and the resident set rebuilds 100 %.

**The anomaly that opened this round:** the same run wrote **0 MTP DRAFT
records** while the MTP head was drafting and accepting ~65 %. The `mtp`
predictor rows had no source.

### Round 9 — the MTP ids were never allocated, so DRAFT records were never written (2026-10-30, `code`)

Cause, from the code and reproduced device-free:

- the trace's MTP block marks its routed ids an output
  (`ggml_set_output(ids_out)`) and stores the tensor for decode's read-back,
  but **nothing else in the FFN consumes it**. `ggml_backend_sched_alloc_graph`
  allocates only the nodes the forward walk reaches; the expand never reached
  this tensor, so `ids_out->buffer` stayed null.
- decode's read-back skips an id whose buffer is null
  (`llama-context.cpp`, the `rids->buffer == nullptr` guard) and then writes no
  `DRAFT` record, so the run's trace had none although the graph ran.
- the cache's own `ids_out` below is reachable through `ids_flat`, which is why
  the cache's records always worked and this one did not.
- fix: one `ggml_build_forward_expand(gf, ids_out)` on the trace's tensor
  (`contrib/llama.cpp/patches/0069-expert-route-trace.patch`, regenerated from
  `b12a4ab`). Unset `LLAMA_EXPERT_ROUTE_TRACE` still adds no tensor and no
  node.

Device-free repro (`code`): a CPU qwen4exp MTP context, one file, two draft
decodes a token, the writer opened without a cache. Before the fix: `DRAFT 0`;
with the line: `DRAFT 2`. The repro loads a merged synthetic model, which the
stock loader refuses (its `done_getting_tensors` requires every tensor used);
the served layout is two files, so the repro needed a scratch-only tolerance
and is cited as evidence, not committed.

Red cell: `test_mtp_route_ids_are_expanded_into_the_graph` reads
`src/llama-graph.cpp`, names the trace's MTP block, and asserts the expand
between the output mark and `set_route_ids` (round 10 added the placement
assertion below). Red mutation, seen failing: delete
`ggml_build_forward_expand(gf, ids_out);` (the CPU repro then reported
`DRAFT 0`).

Card gate still open (`measured-here`, Claude's window): one small prompt, the
trace on and MTP on, then `DRAFT > 0` in the trace, `mtp` predictor rows in
the replay, and one fused-router line for the MTP context. The tiny-prompt
command and its checks:
[expert-foresight-handover-round10.md](expert-foresight-handover-round10.md).

### Round 10 — the trace's MTP block pushed the router out of the fusion (2026-10-30, `code`)

Round 9's fix put `ggml_build_forward_expand(gf, ids_out)` inside the MTP block
where the block stood — **before** the FFN's `call early so that topk-moe can
be used` expand of the weights. A graph's node order is the expand order, so
the trace's expand appended SOFT_MAX, ARGSORT, VIEW and CONT first, and the
weights expand then appended GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE
after them. 0064's `ggml_opencl_can_fuse_topk_moe` wants ten consecutive nodes
from the router's SOFT_MAX — RESHAPE(src n[0]), ARGSORT(src n[0]), VIEW, then
GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE — and the scan met ARGSORT at
n[1]. So **the MTP draft's router ran unfused whenever the trace was on**, and
`fusion-coder-night.md` records that the fused kernel's softmax sum order
changes greedy texts and draft acceptance: the trace-on drafter was not the
served drafter, so the DRAFT records priced a drafter the unit never serves.

Fix (`code`, llama tree `7e0ba9e`): the block moved after the weights expand.
The CONT then lands after the ten nodes and the fusion survives —
`ggml_can_fuse_subgraph` exempts the nodes listed in `outs` from its use-count
and output-flag checks, and the VIEW the CONT consumes is `outs[0]` (the
cache's own `ids_out` below is placed exactly that way). Unset
`LLAMA_EXPERT_ROUTE_TRACE` still adds no tensor, no node and no name.

Evidence on the card (`code`, the line's print is unmeasured): the fusion is
a backend decision nothing else logged, so the ids CONT is named
`ffn_moe_ids_trace` and the OpenCL fused dispatch prints
`opencl: topk-moe fused in the MTP trace graph (n_expert N, n_tok T)` when the
trace is on and a graph holds that name. Round 11's review (`code`) corrected
that line's delivery: the name is set after the graph callback `cb` (which
renames to `"%s-%d"`, so a name set first is `ffn_moe_ids_trace-<il>`), the
backend compares the 18-byte prefix, and the line is a **WARN once per
process** (ggml INFO goes to verbose and is dropped at the units' default
level; per fused dispatch it would flood a month-long trace). The graph scan
is decided once per graph, so the trace-on A/B does not price the scan as the
trace. The card check reads the line: present = fused router in the MTP
context, absent = the trace-on drafter is again a different drafter, or the
trace is off. No GPU run of it yet (`measured-here` owed).

Red cell, extended in the same cell: `test_mtp_route_ids_are_expanded_into_the_graph`
now asserts `ggml_build_forward_expand(gf, weights);` appears **before** the
trace block's start in `src/llama-graph.cpp`. Seen failing on round 9's
placement (the assertion reported the block's offset before the weights
expand), green after the move. The line's presence is a `code` claim; only
the card run can call it `measured-here`.

**Round 11's cells** (`code`, red mutations seen failing):
`test_fused_router_name_survives_the_graph_callback` asserts the name is set
after `cb` and the scan is the prefix (`sizeof(...) - 1`) -- red on a name set
before `cb`, red on `sizeof` without `- 1`;
`test_fused_router_line_is_warned_once_per_process` asserts `GGML_LOG_WARN`
with a static once-flag and a per-graph scan -- red on `GGML_LOG_INFO`, red on
a truthy once-guard. Both need `EXPERT_ROUTE_TRACE_SRC`.

## Stage 1 — the replay (`tools/expert_cache_replay.py`, `code`)

Reads both formats (`ECT1` from the 2026-10-15 knob, `ERT1` above) and feeds
plans into the engine's own mechanism — a plan every 6 decode tokens, at most
96 swaps, landing 2 steps later, no plan while swaps are in flight:

- `strata` the baseline as the replay computes it today (usage decayed 0.7,
  a swap while the bank expert beats the slot by 1.5);
- `oracle` the next `--horizon` tokens (default 24) known, token by token;
- `foresight` the verify's draft token ids, known before the verify (the draft
  pass produces them), mapped through a per-layer token->expert table fitted on
  the even requests. This replaces the first stage's `mtp`, which used the
  accepted drafts' *routing* from the verify that had already processed them:
  that was the last step's routing with a plan's label on it, not foresight.
- `prompt` the request's prompt routing, normalized a layer, as the prior;
- `both` the two normalized and summed;
- `mtp` the MTP block's own routing of the verify's draft tokens, from the
  DRAFT records of the training requests (only where they exist; no row
  otherwise); `mtp_both` that plus the draft-id table, summed;
- `learned` (stage 1b) a per-layer linear map, 6 coefficients, in **step**
  space: the routed counts of the last 1, 2, 6 and 12 steps of the same request
  (each normalized), the request's prompt prior and the foresight counts, to
  the next `--horizon` tokens' occurrence from the landing (`si + delay`, as
  the oracle); least squares on the even requests, scored on the odd ones,
  windows cut at request ends so nothing of a scored request enters a training
  row.

The accept sidecar is walked in request order and checked by position and row
count (every request restarts at position 0, so a global position set collides);
a prompt-last 1-token DECODE with no line is relabeled PROMPT.

The reader takes **several files as one ordered history** (`--trace` a
directory, or a first file and its `.001` rotations; `--single` replays one
file), and their accept sidecars in the same order. `--train-curve` is the
size curve above.

Reported: hits on decode's routed pairs and swaps executed, the baseline and
the oracle beside each predictor, and the trace's own resident set against the
baseline's replay (a fidelity line, not a verdict). The default report is the
**held-out** odd requests; without an odd request the fitted rows are marked
in-sample. `--sweep` runs every predictor and the oracle at horizons 6, 12, 24
and 48 and prints the table; the horizon is a sweep, not a constant (20 was a
guess, 24 the default so far).
Planned and not built
(stage 1b's alternative): the same features into a tiny MLP (one hidden layer,
32 units a layer) replacing the linear map — no reason on the record yet to
prefer it over the linear fit.

Device-free tests (`tests/python/test_expert_route_trace.py`, red first):
1. round trip, in a default pytest — the committed fixture is the serializer's
   output (`tools/expert_route_trace_selftest.cpp`), the replay's reader must
   reproduce every field and its writer the same bytes; with the llama tree set
   the selftest is compiled and must match the fixture. Red mutation: the
   LAYERS record writes `n_slots + 1`, seen failing (misread residents).
2. a prompt's 4-token tail at position 10 beside a verify at 200 and a decode
   at 203 — the tail is a PROMPT graph, the steps are the two decode graphs,
   the accept sidecar aligns by position. Red mutation: the kind decision drops
   `!all_outputs`, the tail classifies VERIFY, the selftest exits 3 (seen
   failing).
3. foresight on a synthetic two-request trace: the even request trains a
   token->expert table, the odd request's draft token ids are the same but its
   verify rows are garbage, so only a token-id predictor reaches the oracle
   (horizon 1, delay 1). Red mutations, seen failing: the predictor's plan
   ignored (zeros must not reach the oracle), and the old row-based source (the
   verify's own accepted rows) must miss.

## Item 14's save path and the card arms (2026-10-30, `code` + `measured-here`)

The concept-mining item 14 build rides beside the trace: patch 0070 adds
`--llama-expert-cache-save FILE` (`--llama-expert-cache-save-rule xe|cu`), and
`tools/expert_profile_from_trace.py` writes the same STRP file from a trace.
Both features are **off by default**; `--llama-expert-route-trace` and
`--llama-expert-cache-save` each cost one getenv and nothing else when unset.
The save is regenerated on top of 0068's slot loan, because both touch
`llama-expert-cache.{h,cpp}`; 0069 keeps its number and the series is
0001-0070.

Claude's card runs (B60, Flash-Next, `measured-here`): the trace-built profile
against the unit profile is +0.35 % decode with adapt on and +5.7 % with adapt
off; a warm start on a saved profile is +2.5 % in-sample and -1.9 %
out-of-sample; profiles change answers through the CPU miss's rounding, a
rounding-only class; the MTP draft routing is the best predictor at 76.2 %
replacing and +0.26 blended; the blend rule is the limit, the oracle with the
rule off +6.52 against blended +1.28. Audit 14a carries the same figures.

## Where it lives

`contrib/llama.cpp/patches/0069-expert-route-trace.patch` (the trace),
`contrib/llama.cpp/patches/0070-expert-cache-save.patch` (the save),
`tools/expert_cache_replay.py` (the replay's modes),
`tools/expert_profile_from_trace.py` (a trace into STRP), `src/exec/llama_spec.cpp`
(the accept sidecar), `tests/python/test_expert_route_trace.py`,
`tests/python/test_expert_profile_save.py`, and
`tools/expert_route_trace_selftest.cpp` (the device-free cells).
