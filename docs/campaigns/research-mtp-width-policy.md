# research-mtp-width-policy: the MTP window width (concept-mining Wave 2, item 13)

Status: research + build plan; section 6 step 1 (the trace and the replay tool) built on branch `mtpw`
(2026-10-30/31, section 7; rounds 2 and 3 fixed the reviews' findings). The serving path is unchanged. Date 2026-10-30, tree qfndev at d725dd6; the round-1 line
numbers are read at branch base qfndev `450eca9`.
Evidence classes: `code` (path:line, read in this session), `paper` (a document or commit message says so),
`inferred` (reasoned here, not measured), `measured-here` (arcint's own window). References are cited as
`Strata-ref:`, `Infernix-ref:`, `llama-port:` (the checked-out trees, files read at origin/HEAD or the working tree).

## 1. Summary

- arcint drafts a fixed chain of up to K tokens and stops it early at one test: the head's top probability of the
  draft just made is below `--llama-mtp-min-p` (`code`: `src/exec/llama_spec.cpp:562-575`).
- Strata's engine replaces that test, when `STRATA_SPEC_COST=1`, by a cost-aware rule: keep a draft while the
  expected tokens per millisecond of the window still rise, with the per-width round cost measured live and the
  draft-probability-to-acceptance map learned live (`code`: `Strata-ref:src/spec/draft_policy.cpp:236-264`).
  It only chooses how many drafts are verified; every draft is still verified.
- Measured gains in the references are small-to-moderate and engine-specific: Strata +4.7 % decode on one GB10 in the port's first run of 26 October (`paper`: `docs/DGX_SPARK_LINUX.md:124`; the
  table comment `src/core/nv_defaults.cpp:74` says 1.047x [1.019, 1.079] on IQ3_S), which the reference's own later
  check of the whole four-switch set did not reproduce: 0.996x [0.97, 1.02] on IQ3_S, 1.02x [1.00, 1.05] on NVFP4
  (`paper`: `DGX_SPARK_LINUX.md:101-103`), Infernix adaptive MTP +2.0-2.1 % geomean over the best fixed length on two Qwen3.5-family models, and
  a loss on one (`paper`). arcint's own oracle ceiling on Flash-Next is +2.5 % (`measured-here`).
- Recommendation: build Strata's rule (probability calibration + measured per-width cost + best-width truncation)
  as a pure policy class beside `verify_walk.h`, ~200 lines, arcint-only (no llama.cpp patch), opt-in flag first.
  The reference's re-check did not reproduce its first-run decode gain, so price it on our cards before building on
  it (`inferred`): perhaps +1-2 % on Flash-Next, unknown on the agent and the coder; the gate is a same-window A/B per
  unit.

## 2. What arcint does now (item 2 first, the baseline the references are compared to)

- `--llama-mtp N` is the draft ceiling, 0..7 (`code`: `src/config.cpp:383-385`); the unit values are 5 (agent),
  4 (coder) and 3 (Flash-Next, with the 0.5 stop) (`code`: `contrib/systemd/arcint-agent.service:55`, `arcint-coder.service:31`, `arcint-flashnext.service:80-81`;
  concept-mining audit item 13).
- `--llama-mtp-min-p P` is the only policy (`code`: `src/config.cpp:416-418`, help `:290-292`). In `draft_multi` each
  draft step ends with: pick the token, compute its softmax top probability (`DraftHead::top_prob`,
  `llama_spec.cpp:304-313`, over the draft-vocabulary rows when `--llama-mtp-vocab` is set, else the full head), and
  if `prob < min_p_` the lane stops drafting and the token is left out of the verify (`code`: `:562-575`).
  A draft that passes is pushed and the next draft step runs (`:575`). The ceiling per lane is
  `min(p.n_max, n_draft_)` (`:513`); `n_max` is `min(n_draft_, budget - 1)` (`code`: `backend_llama.cpp:1734`).
- The probability is computed only when `min_p_ > 0` (`:562, :567`). It is not stored, not calibrated, not compared
  with the cost of the next verify row, and not used by anything after the stop test.
- The verify is one target decode of `1 + drafts` rows per lane; several lanes share the draft step and the verify
  (`code`: `backend_llama.cpp:2169-2220`, `draft_multi`, `decode_multi`). The units run one lane (`code`: none of the three unit files above carries a lane flag; the lanes campaign is the
  lanes-agent-subagent row of `docs/campaigns/README.md`, `:77`).
- Timings already exist per step: `stats.draft_propose_seconds`, `stats.draft_verify_seconds`, the walk and the
  rollback (`code`: `backend_llama.cpp:1744-1745`, walk `:1794-1795`, rollback `:1814`). The data a cost model needs is measured today
  and thrown away each step.
- Quality class of the walk: `walk_verify` draws each row with the request's own sampler and keeps a draft only
  when the drawn token equals it (`code`: `src/exec/verify_walk.h:31-77`; the draw `:36`, the accept test `:72-76`). Greedy: the plain loop's tokens, up to
  rounding. Sampling: every emitted token is a draw from the target's own distribution at that row, so the
  distribution is the plain loop's whatever the width (`code`: `verify_walk.h:36`, `:47-49`; the file's own header says the same).
  arcint does not use rejection sampling against the drafter's q (that is Strata's opt-in `STRATA_SPEC_PROB`, below).
  A width policy changes which rows are verified, not the rule that accepts them: quality class **rounding-only**
  (the verify batch shape moves the target's rounding, answers judged at the answer-level bar, CLAUDE.md).

## 3. The references' width policies

### 3.1 Strata (`Strata-ref`, origin/HEAD)

Three separate things, easily confused:

**(a) The min-p stop, the default (`--spec-min-p 0.5`).** The MTP chain runs step by step and continues while
`pj >= min_p` (`code`: `Strata-ref:src/core/mtp.cpp:1742`, `for (j = 1; j < max_steps && (chain_rule_ ? chain_rule_(j-1, pj) : pj >= min_p); ++j)`).
This is arcint's rule, taken from there. "`--spec 4 --spec-min-p 0.5` adopted; deeper windows did not pay"
(`paper`: `docs/campaigns/research-strata-history.md`).

**(b) The cost-aware window, `STRATA_SPEC_COST=1`.** `chain_rule_` is replaced by `DraftPolicy::keep_draft`
(`code`: `Strata-ref:src/program/generate.cpp:18126-18205` and `:12292-12299`, two call sites; `include/strata/core/mtp.hpp:115`). The rule
(`code`: `src/spec/draft_policy.cpp:236-264`, comment `include/strata/spec/draft_policy.hpp:78-85`):
- Inputs. (1) The probability `p_j` of each draft as it is made. (2) `a(p)`: the measured acceptance of drafts that
  arrived with probability `p`, an 8-bucket table `pa_ok_/pa_n_`, decayed 0.995 a round, prior = `p` itself worth 4
  observations (`:221-234`; "measured acceptance of the MTP chain is not the same as its own softmax: code and
  thinking text are over-confident"). (3) `cost(t)`: the measured round time of a window of `t` rows, draft
  included, an EMA (alpha 0.1, a running mean for the first samples) kept per context-length bucket
  (<6K, <24K, longer) (`:13-26, :41-53`, `kCtxBuckets = 3`).
- Rule. After draft `j`: `q *= a(p_j)`, `E += q` (expected tokens of a window of `j + 2` rows), value
  `v = E / cost(rows)`. The window to verify is the best `v` seen (`best_rows`), margin 3 % /3. The chain goes on
  only while one or two more drafts could still beat `best_v` (`v1`, `v2`: "the cost is not convex: on a card
  with experts in RAM a window of 3 costs about a window of 2", `:236-264`).
- Guards. An unmeasured width is probed up to 3 times when the chain is likely to hold that far (`kProbes`);
  a width's first measurement is replaced by the guess scaled from the other widths, and later samples are capped at
  twice the known value (`clamp_sample`, `draft_policy.cpp:55-65`; the 0.1 is `kCostAlpha`, the EMA weight, not a
  sample weight); costs of unmeasured widths are scaled from measured ones by a fixed shape
  `{1, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6}` (`:13, :41-53`); an optional per-card seed table
  (`STRATA_SPEC_COST_TABLE`, `generate.cpp:16495-16511`) starts the costs known.
- Separate from `keep_draft`: `stale()` (`draft_policy.cpp:116`; a width unmeasured for 300 rounds is re-tried) is used
  only in `choose()` (`:142`, `:147`) and `chain()` (`:287`, `:291`), the lookup/suffix paths, not in the MTP rule.
- It reads only earlier rounds and the current chain's own probabilities, never the target's draws.
- Measured (`paper`): decode **+4.7 %** (`docs/DGX_SPARK_LINUX.md:124`, GB10, first run 26 Oct), and in the table's
  own comment 1.047x [1.019, 1.079] on IQ3_S (`nv_defaults.cpp:74`); the reference's own re-check on 27 Oct saw no
  decode change from the four-switch set, 0.996x [0.97, 1.02] on IQ3_S and 1.02x [1.00, 1.05] on NVFP4
  (`DGX_SPARK_LINUX.md:101-103`), so the decode gain was not reproduced there. Prompt speed is what that re-check
  confirmed. The comment's 24 of 24 greedy answers byte-identical, "every draft is verified so the
  answers cannot drift" (`src/core/nv_defaults.cpp:73-76`). Default on sm_121 only. Off elsewhere, so no
  card but one carries a measured default. A cost on the other side: with `"parallel": 8` it costs about 3.5 %,
  set 0 there (`docs/DGX_SPARK_LINUX.md:137`). arcint's units are single-request.
- Landing history: 68781055 (best_rows), 5f49ec45 (step-by-step), d232721d (look-ahead and probes, 2026-10-21),
  b1ebc3d7 (per-context cost table), bf8285b8 merge, 9a398bd5 (sm_121 default, 2026-10-27) (`paper`: commit log).
  The history-doc line "A cost-aware window width is opt-in" is right outside sm_121 (`paper`).

**(c) Other gates.** The drafter's cumulative confidence cut for DeepSeek (`STRATA_DS4_DSPARK_CUM`, default 0.5 on
HIP only): verify while the product of the confidences stays above the cut; 71.7 -> 76.9 tok/s on 4x R9700 with
the same ids, flat on CUDA, so CUDA keeps 0 (`paper`: e83b82af). A different model's drafter, not arcint's head.
`STRATA_SPEC_PROB=1` is rejection sampling for sampled requests, opt-in (`code`: `include/strata/core/spec_prob.hpp`):
not the width policy and not needed here (point-mass drafts under arcint's equality walk are already the standard
rule, said in the header itself).

### 3.2 Infernix (`Infernix-ref`)

- `--draft-tokens auto` means different things per path. For MTP on the Qwen3.5 family it is the **adaptive-mtp**
  fast path: before each all-neural round the policy picks the draft length with the most expected committed
  tokens per unit of round time (`code`: `src/models/qwen3_5/program/speculative/mtp_draft_policy.h:14-29`).
  Inputs: a per-position conditional acceptance, decayed every round and shrunk to the pooled acceptance
  (`:43-47`); a round cost learned by an exponentially weighted least-squares fit `t = a + b*k` over consecutive
  start-to-start cycles (`:134-141`), the fixed model `1 + 0.2*K` until the fit has 24 rounds of spread
  (`kModelDraftCost = 0.2`, `kCostMinRounds = 24`, `:108-112`); a switch margin of 2 % (`kSwitchMargin`, `:99`);
  a probe of the next longer length every 16 choices (`:103`). It reads only earlier rounds, never the draws
  (`:26-29`). Each length is its own captured round family, which arcint's llama.cpp path does not have.
- Per-draft gate: `--spec-min-p 0.5` for the Dense8 MTP path, run on the device as a chain of IF nodes so steps past
  the cut do not run; +2.2 % on mixed topics with MTP 4, +3.5 % IQ3_S with auto (`paper`: 140faefc).
- Block drafters (DFlash2): a calibrated decile map from draft probability to acceptance, then the count that
  maximises expected tokens per round time, cost `1 + proposal + per_draft*k` (`code`:
  `src/runtime/speculative/draft_length.h:3-64`). Same idea as Strata's `a(p)`; the drafter proposes the whole
  block first, a different shape from arcint's serial MTP chain.
- Measured (`paper`, RTX 5090, INT8 KV): 3d1931c7 auto vs fixed 7 on the NVFP4 27B DFlash2: +17 % on code at one
  request, -6.5..+1.2 % batched, geomean +1.6 %; 35B-A3B +6.1 % (worst -1.5 %). 3d299bce: adaptive-mtp +2.1 % and
  +2.0 % geomean against the best fixed length on two models, and no gain beyond spread on the groupwise 27B.
  `src/models/qwen3_5/program/planning/fast_paths.cpp:529-530` (re-timed 2026-10-25): groupwise 27B adaptive-mtp +10 % on code at one request, -1.5 % on
  docs and -4.8 % at four requests, so it keeps fixed 3. Offloaded-expert 122B (verify columns bring their own
  misses): two gated drafts +14.0 % over plain decode, adaptive at a ceiling of 2 tied (13.9 %); with two requests
  about 6 % slower than plain (`src/models/qwen3_5/program/planning/fast_paths.cpp:509-518`). The "+17 %" in the audit table is the DFlash2 block
  drafter on code at one request, not an MTP chain gain.

### 3.3 FreeToken, llama.cpp

- FreeToken (`FreeToken-ref`): no draft-width policy; `docs/models.md:157` says speculative decoding is not served
  for the model it names (`code`/`paper`: grep over code and docs found no draft-length code).
- llama.cpp (`llama-port`): a fixed `draft.n_max` (default 3) and `draft.n_min`, and one greedy gate `p_min`
  (default 0), the first-token probability test `cur_p->data[0].p < params.p_min` (`code`: `common/common.h:328-333`,
  `common/speculative.cpp:421, :439, :871, :1348-1375`). No acceptance history, no cost model. The server logs
  per-position acceptance rates (`tools/server/server-context.cpp:705-725`), which is a measurement, not a policy.
  arcint's `--llama-mtp-min-p` is the same class of rule.

## 4. How arcint's fixed K + min-p stop differs

| aspect | arcint now | Strata cost-aware | Infernix adaptive-mtp |
|---|---|---|---|
| width decision | fixed K, early stop at one low draft | best `E/cost` over the chain so far, 1-2 draft look-ahead | best expected tokens per round time over lengths 1..ceiling |
| acceptance model | none (the probability is a threshold only) | `a(p)` learned per probability bucket | per-position, learned, shrunk to pooled |
| cost model | none | measured per width and context bucket, shape fallback | `a + b*k` fit, `1 + 0.2K` until measured |
| a draft below the line | stops the chain, left out of the verify | stops the chain only when no later width can win | chosen before the round, per history |
| calibration of an over-confident head | none: one fixed 0.5 | learned | learned |
| build shape | serial draft step per token, verify batch per lane | same (`mtp.cpp:1742`) | captured graph per length (different) |

Consequences (`inferred` unless noted):
- A fixed min-p cut is a probability test, not a cost test: it cannot lengthen the chain where the verify row is
  almost free (a resident MoE) nor shorten it where it is dear (expert misses). The Flash-Next sweep shows that cost:
  the best fixed width per answer is 2 for prose and 3 for code, and widths 4 and 5 lose 5.5 and 8.4 % (`measured-here`,
  `research-concept-mining.md`, Wave 1 window 1).
- The head's probability is over the draft vocabulary rows when `--llama-mtp-vocab` is set (`code`: `llama_spec.cpp:300`
  passes the subset's logits to `top_prob`), so it is normalised over fewer tokens and reads higher than the full-vocabulary
  value: the fixed 0.5 means a different thing per unit. A learned `a(p)` absorbs it.
- The agent's and coder's verify cost has a cliff between 4 and 5 rows (about 100 ms against 73, stage 3c, `paper`:
  audit item 13) and the coder lost at 5 (decode -3.6 to -9.2 %, acceptance 37.9 -> 33.6 %, `paper`:
  `fusion-coder-night.md:29`). A measured per-width cost sees a cliff; a fixed K does not.

## 5. Build plan

### 5.1 The rule to build first

Strata's `keep_draft` + `observe_probs` + a measured per-width cost table, in its single-request form. It is the
rule whose first run measured a gain on an MTP chain in a build of arcint's shape (serial drafts) - a gain the
reference's own re-check did not reproduce (`paper`, `DGX_SPARK_LINUX.md:101-103`), so it is a candidate to price on
our cards, not a proven one - the smallest, and it needs no model-specific code (`code` and `paper` above). Not built first: Infernix's `a + b*k` fit (needs spread lengths
and a captured-graph assumption; the fit would starve if the min-p stop keeps lengths narrow), and the cumulative
confidence cut (a different drafter, flat on CUDA).

Scoped out, each a deviation from the reference, named per the repository rule: the lookup and suffix drafters
(`DraftPolicy::choose`, `chain`): arcint has no such source, n-gram lookup lives in a different 0.x campaign;
`STRATA_SPEC_PROB`: arcint's walk is exact-match, no q is kept; DFlash2 block widths: separate flag
(`--llama-dflash`, `config.cpp:295-300`) and its own min-p; Strata's multi-lane/parallel treatment: arcint's units are
one lane. Do not extend to lanes without the lanes' own measurement (Strata and Infernix both lost 3.5-6 % batched).

### 5.2 Where it hooks

1. New header `src/exec/draft_policy.h`: a pure class (no libllama), tested like `tests/test_verify_walk.cpp`.
   State: `cost_[ctx_bucket][rows]` EMA and counts, `pa_ok_/pa_n_[8]`, `rounds_`. API modelled on Strata's (whose observer is `observe(bool lookup, int t, int accepted, int match, double round_ms)`,
   `draft_policy.hpp:40`, plus `observe_chain`, not an `observe_round`):
   `keep_draft(ChainState&, idx, p, cap)`, `observe_probs(p, n, accepted)`, `observe_round(rows, ms, ctx)`,
   `set_context(pos)`, optional `seed_costs`. Port the constants (`kShape`, alphas, probes, margin) as a starting
   point; each is `paper`, none measured on the units.
2. `llama_spec.cpp`, `draft_multi` (the loop at `:531-579`). At `:562-567` compute the probability whenever the policy
   or `min_p_` is on. At `:571` replace `prob < min_p_` by `!policy.keep_draft(st[k].chain, i, prob, st[k].n)` when the
   policy is on (the min-p value stays a floor: a draft below it is never kept). Record `prob` per kept draft in the
   `DraftPart` (a new `std::vector<float>* probs`, `llama_spec.h:33`). After the loop, truncate `out` to
   `best_rows - 1` (Strata verifies the best width, not the whole chain it drafted). Cost of the extra drafts that are
   computed and dropped: the look-ahead of 1-2 steps (a draft step plus its head, ~4 ms of a 36.7 ms coder step before
   `--llama-mtp-vocab`, `paper`: `fusion-coder-night.md:24`); Strata's look-ahead is the same.
3. `backend_llama.cpp`, step loop `:1734-1818`. After `walk_verify` (called at `:1783`; its timing `:1794-1795`, rollback `:1814`): call
   `policy.observe_probs(probs, draft.size(), w.accepted)` and `policy.observe_round(rows = draft.size() + 1, ms =
   req.t_propose + req.t_verify (+ walk + rollback), pos = have.size())`. Use the round's start-to-start time as
   Strata and Infernix do if the sum proves short of it; the stats fields exist (`:1744-1745`). A policy instance per
   lane in `Lane`, as `pending` and `mtp_end` are.
4. Flags in `config.cpp`: `--llama-mtp-policy off|cost` (default `off`, then flipped per unit by a gate), keep
   `--llama-mtp N` as the ceiling and `--llama-mtp-min-p` as the floor. `/props` reports the policy and the learned
   cost table at request end, like Strata's `cost_table_text()` (`code`: `draft_policy.cpp:98-113`).
5. Lanes: when `lanes_ > 1` the policy is off (fixed K), because a combined step verifies uneven lengths and the
   cost depends on batch composition (`code`: `n_uneven_` counter, `backend_llama.cpp:1826, :2226`; both references lose
   batched, `paper`).
6. No llama.cpp patch. If the `probs` of a draft are needed before the draft vocabulary head, they come from
   the same `top_prob` already computed.

### 5.3 Per-unit defaults and why the cost model matters

The cost table is the unit's own and starts seeded, not from the generic shape. Seed it from the first rounds' measured
times by width (Strata's `STRATA_SPEC_COST_TABLE` form, `generate.cpp:16495-16511`) and let the EMA take over.

- **Agent (dense 27B, B60, ceiling 5, min-p as the unit has).** Row cost near linear with a cliff between 4 and 5 rows
  (`paper`, stage 3c). Expect the policy to settle at 3-4 on prose and 5 on code and tool calls. Default: ceiling stays
  5; arm ceiling 6 only if the table shows 5 below 4 (`inferred`). Context buckets matter: attention per row grows
  with depth (a 55k to 232k prefill range, `paper`), so `<6K/<24K/more` buckets are likely too coarse for 245k; add
  `<64K`, `<128K` before the first agent measurement (`inferred`).
- **Coder (small MoE, A770, resident, ceiling 4).** A verify row is cheap relative to the draft head's step (draft
  step 4.2 ms of 36.7 ms before the vocabulary, `paper`), but the fixed 5 lost 3.6-9.2 % with acceptance falling
  37.9 -> 33.6 % (`paper`). The policy's job here is to find the gain between 3 and 4 per answer. Keep ceiling 4
  first; arm 5 with the policy as a test (it can only truncate). The A770 sits behind the chipset, where patch 0047 cut the coder's input copies from 1.93-2.02 ms to 133-255 us an
  input (`paper`: `research-concept-mining/audit.md:371`, README 0047); that cost is flat per round and absorbed by
  `cost(t)`.
- **Flash-Next (large MoE, B60, expert cache, ceiling 3).** The cost model matters most here and is the reason to
  measure rather than to assume: a verify of 4 tokens routes 40 (token, expert) pairs per layer and ~19 % of those
  are missed (`measured-here`, window 1), so each added row buys more misses, over a CPU-miss path with a cap
  (`measured-here`: caps 32-128 inside spread, 16 binds). The row cost is not linear and depends on the cache
  state, so a measured cost, which includes the misses, is the only honest input. The ceiling stays 3 with the
  0.5 floor; arm ceiling 4 (the fixed width 4 lost 5.5 %, but the policy would verify it only where `E/cost`
  rises, `inferred`). Prize: the oracle per answer is +2.5 % over width 3 (`measured-here`); a policy that learns
  within an answer collects part, so plan for 1-2 % and do not gate on more (an adoption does not need a minimum
  gain beyond the spread; CLAUDE.md: any measured gain with no other target degrading).

### 5.4 Quality class and gate

- Class: **rounding-only**, per section 2. Greedy: tokens differ from the fixed-K run only through the target's
  rounding at a different verify batch size (arcint measured this: text differs, answers right, DESIGN 3.5, `paper`);
  byte cells in the suites stay as tripwires, reported not vetoes (CLAUDE.md). Sampling: the distribution is the
  target's own at every row, the policy reads only earlier rounds and the current chain's probabilities (Infernix says
  the same of its rule: `mtp_draft_policy.h:26-29`; `inferred` that the sampler's draw sequence is consumed one draw
  per emitted token whatever the width, since `walk_verify` samples a row only after the previous token was accepted).
- Gate per unit, same card, same window, both arms in one session, one fresh process an arm, two rounds, the second
  reversed (`measured-here` method of Wave 1): decode t/s of the six-answer mean (capital, Strata's RFC 4180 request,
  four 300-token answers) rises beyond the run-to-run spread; the capital and the needle stay right; acceptance and
  tokens a round reported; no unit's prefill touched. Answer-level bar for equivalence if asked: mean KL within
  0.03 nats of the baseline arm's and argmax agreement within 1 point (CLAUDE.md); the acceptance tasks: coder
  Lua CSV 10/10 (a Prüfstand 9/10 is inside the history), agent 10/10.
- Red cases (a test must be able to fail): (1) the policy forced to width 1: decode must regress; (2) a policy that
  truncates after the draft and verifies the full chain anyway: the unit test on the chain fails; (3) `keep_draft`
  fed a constant cost: it must reproduce the fixed K path and the unit test equate them; (4) the walk test
  (`tests/test_verify_walk.cpp`) with a width change, a mutation that accepts past the target's disagreement.
  Each asserted in the built artifact (CLAUDE.md red-case rule), not assumed.

## 6. Card measurements to run (not run here)

All with `LLAMA_DECODE_TIMING`-class output (the existing propose/verify split per step) and the served log's
rounds, tokens a round and accepted drafts. Honor the card SOP before each leg (`docs/sop-card-window.md`).

1. **Trace first, no code (one window per card).** Existing binary, `-vv` verify lines (`backend_llama.cpp:1799-1803`
   logs `accepted a/n` per verify), width ceiling 5 / 5 / 4 and min-p 0: record, per round, rows, accepted, the
   probability of each draft and the time of propose and verify. A new 10-line trace switch in `llama_spec.cpp` is
   worth it (Strata has the same: `STRATA_SPEC_TRACE`, one line a round, `generate.cpp:18124` comment / `:18131` `fopen`). Offline replay of
   `keep_draft` over the trace gives the policy's expected tokens a round and the width histogram before any
   card leg, and the `a(p)` map per unit: whether the draft-vocabulary probability is calibrated at 0.5.
2. **Per-width round cost table per unit**: fixed widths 1..ceiling+1 at context depths 4k, 16k, 64k (agent also 128k,
   232k), same answers: round ms by rows. Shows the cliff on the agent, the miss slope on Flash-Next.
3. **Policy vs best fixed**, per unit: `--llama-mtp-policy cost` against the best fixed width of step 2, ceilings as
   in 5.3, six answers, two rounds reversed. Flash-Next also at ceiling 4. Read mean t/s, spread, acceptance, the
   policy's width histogram, the capital.
4. **Answers**: coder Lua CSV task 10/10 at greedy and at 0.7, agent task 10/10, Flash-Next needle; for any greedy md5
   changes the answer-level bar, not byte equality.
5. **Single-lane assumption check**: confirm the units run one lane before enabling the policy (`n_uneven_` log).

Open before the build, to settle with step 1: whether `t_propose + t_verify` alone matches the start-to-start round time
(the walk, emit and rollback add to it, and the `cost(t)` that Strata learns is the whole round); whether per-lane
state in `Lane` survives preemption/checkpoint restore the way `pending` does (`backend_llama.cpp:1683-1684`).

## 7. The trace and the replay (built; rounds 2 and 3 fixed the reviews' findings)

Built on branch `mtpw` from qfndev `450eca9`, arcint only, no llama.cpp patch: the off-by-default trace
(`src/exec/mtp_trace.h`, `src/exec/llama_spec.{h,cpp}`, `src/exec/backend_llama.cpp`), the stdlib-only replay
(`tools/mtp_width_replay.py`) and the device-free tests (`tests/test_mtp_trace.cpp`,
`tests/python/test_mtp_width_replay.py`, the fixture `tests/python/fixtures/mtp_trace_lines.txt`). It does not
touch the serving path's policy hook (§5.2).

### 7.1 Where the code disagrees with this document (`code`, the reference's source wins)

Read this session: Strata-ref `4b9e05a2`, Infernix-ref `b741710c`, arcint `mtpw`. A label in §1-§6 is not
evidence; these are the citations and mechanisms that differed when the code was read:

1. **`keep_draft`'s call sites** are `generate.cpp:16513` (the MTP chain) and `:11221` (the suffix path), not
   `:18126-18205` / `:12292-12299`; the chain loop's replacement of the min-p test is `:16512-16515`.
2. **The rule's own lines**: `draft_policy.cpp:199-224` (`keep_draft`), `:154-170` (`observe`), `:190-196`
   (`observe_probs`), `:182-188` (`draft_rate`), `:174-179` (`p_bucket`), `:98-113` (`cost_table_text`), `:40-53`
   (`cost_ms`), `:55-65` (`clamp_sample`), `:66-74` (`seed_costs`), `:76-93` (`set_context`); the constants
   `:11-24` and `draft_policy.hpp:25-27, :57-78`. §3.1's `:236-264` is the suffix's rule, not the MTP one's range.
3. **The trace's shape**: `generate.cpp:16461-16469` (the comment and the `fopen`), `:16768-16770` (the write).
   It is `"<T> <a> <sfx> <round_ms>"` then `o.spec - 1` probabilities at `%.4f`; it opens `w`, truncating, and the
   list is zero-padded to the ceiling, so a replay cannot tell a never-drafted row from `p = 0`. arcint's trace
   (7.2) omits instead, which is what §6 step 1 wanted.
4. **Strata's round time is start-to-end** (`t0` at `:16752` to `:16765`), not start-to-start: §6's open question
   is not answered by the reference (`code`). arcint's trace records the round's own duration - its start to the
   next round's start, a request's last round to the decode loop's exit - *and* `t_propose` / `t_verify`, so
   both ends are priced. The first format (`mtp1`, round 1) carried the previous round's duration on this
   round's rows; `mtp2` carries each round's own.
5. **The reference's cost arm has no floor**: `keep_draft` replaces the `pj >= min_p` test outright, so a draft
   below the floor is kept when the cost rule wants it. §5.2's "the min-p value stays a floor" is therefore a
   deviation from the reference (`code`), and the replay defaults to floor-off (`--respect-floor` models ours).
   The card arm must include min-p 0 for a faithful price.
6. **arcint's line numbers moved** from d725dd6 through both rounds: the min-p stop is
   `llama_spec.cpp:628` (§1's `:562-575`), `top_prob` `:354` (`:304-313`), the probability computed only when
   min-p or the trace asks (`want_p`) `:565-567, :622`, the ceiling `min(p.n_max, n_draft_)` `:568` (§5.2's
   `:562-567`), the config flags `config.cpp:410-412, :453-454` (§5.2's `:383-385, :416-418`), `StepReq`'s
   `probs` member and the combined step's `DraftPart` `backend_llama.cpp:2199, :2269`, `walk_verify` called at
   `:1847` (§5.2's `:1734-1818`), the trace write `:1883-1896` and the buffered last write `:1902-1907`, the
   request counter `:1745`, `n_uneven_` `:2317`. Cite these by symbol where
   the numbers are read again: the line numbers have moved once already in this round.
7. **The gate's answer set**: §5.4 says "four 300-token answers", the code posts five 300-token answers (hash,
   compiler, tcp, btree, gc) plus the 500-token RFC 4180 request, and its `SWEEP` line means over those six while
   the capital is checked separately (`code`: `tools/fn_sweep.py:109-118, :203-226`). The code wins; the gate uses
   fn_sweep's set.
8. **`fn_sweep.py` had only Flash-Next's flags** and `GPU.0`; to price the agent and the coder in the same window
   it now takes `--unit flashnext|agent|coder`, `--device`, and `--reverse` for §5.4's reversed second round.
   Defaults are unchanged (no flag: Flash-Next, GPU.0). The three flag tables disagree and the code says which
   is which (`code`): `fn_sweep.py`'s flashnext is Wave 1's arm table (11,600 MiB slots, `--llama-mtp-vocab`,
   min-p 0.5) and does **not** carry the unit's slot-loan flags; `served_check.py`'s flashnext is the release
   floor (12,000 MiB slots, no draft vocabulary, no loan flags); the systemd unit file carries the loan flags.
   `fn_sweep.py`'s agent and coder tables equal both `served_check.py`'s and the unit files'. The trace arms of
   7.4 therefore run the arm's own flags, and `served_check.py` stays the answers' check.

### 7.2 The trace (`code`: `src/exec/mtp_trace.h`)

`ARCINT_MTP_TRACE=<file>` turns it on; unset is OFF. Off is the default and costs one `getenv` a draft step
(and a round) and one pointer comparison; with it off no probability is computed beyond what min-p already
needs (`llama_spec.cpp:565-567`: `record`/`want_p` in the chain's `St`, and `pick` gets the null it got
before), so a run without the variable takes the same code path as before (`code`; that branch's own cost is
not measured here, and 7.4 runs a trace-off arm per unit next to a trace-on one). On, the trace computes the
head's top probability per draft, one host pass over the head's logits a draft (~0.8 ms a draft at the code's
own comment for a 248,320-row head, `llama_spec.cpp:354-361`): a cost the serving path does not pay, which
prices a trace-on arm's widths high - limitation 6 in 7.3. One line per verify round, appended, in a fixed
field order (format `mtp2`):

```
mtp2 req=<request counter> seq=<lane index> pos=<context depth> rows=<verified rows> acc=<accepted>
     nmax=<ceiling> minp=<floor> tp=<propose ms> tv=<verify ms> ms=<the round's own duration ms>
     p=<kept probs>[,...]
```

- `p` holds the head's top probability of every draft the chain kept; when the min-p floor stopped the chain, the
  stopping draft's probability is appended (it is the draft the floor rejected, the one event the replay needs
  that the kept list cannot carry). With no drafts, `p=` is empty.
- `rows` is the verify batch's length (the accepted drafts plus the target's row), `acc` how many were accepted.
- `ms` is the round's own duration: its start to the next round's start, the request's last round to the decode
  loop's exit, which answers §6's open question; `tp` and `tv` are the round's own propose and verify halves, so
  a replay can price the sum against it. The record is buffered one round and written when the next round starts
  (`backend_llama.cpp:1762, :1789-1795, :1883-1896, :1902-1907`); `ms` <= 0 is unknown and the replay learns no
  cost from such a line.
- `req` counts the requests the process served; `seq` is the lane's index, not a request id. A file opens once a
  process: clearing the variable later stops the writes, changing it later does not redirect them.
- Off-by-default proof: `tests/test_mtp_trace.cpp` renders records and matches `tests/python/fixtures/
  mtp_trace_lines.txt` byte for byte, checks the field order, the ceiling-end and floor-stop shapes, and that
  the variable unset writes nothing; a fifth case turns it on and checks the file opens, appends one line a
  call and is flushed before the process ends.
- Scope: MTP only, not DFlash2 (the record is built inside `if (mtp_trace_on() && !dflash_)`); lanes: one file
  per process, `seq` distinguishes lanes, and a combined step's `tp` is the shared draft step's time for every
  lane in it, so a multi-lane trace prices per-lane only if the lanes' steps are separate (`code`:
  `backend_llama.cpp:2269`).

### 7.3 The replay (`code`: `tools/mtp_width_replay.py`, standard library only)

Ports Strata's MTP half of `DraftPolicy`: `cost_ms` with `kShape`, `kCostAlpha` 0.1, `clamp_sample` at 2x the
known cost, `seed_costs`, `set_context`, `mtp_tokens` (prior `1 + 0.7*(t-1)`, `kPriorN` 4), `observe`'s
committed-token EMA (`kTokAlpha` 0.05, alpha `max(kCostAlpha, 1/(n+1))`), `draft_rate`, `observe_probs` with
`kCalDecay` 0.995, `keep_draft` (verbatim, `margin/3`, `kProbes` its own probe), `cost_table_text`; constants
cited at their `draft_policy.cpp` lines. Per trace it reports the rounds and requests, the fixed width's tokens
a round and tokens per ms, the policy's expected tokens a round, its rate as the mean of its own E/cost at the
width it would verify (not E over the mode width's cost), the width histogram, the truncated round's inferred
tokens and rate (its tokens over its chosen width's cost, summed), the `a(p)` calibration table by 8
probability buckets (reached/held), and the `tp+tv` against the round's own duration.

`observe_probs` receives only the probabilities the round recorded (`min(rows - 1, len(p))`, `code`:
`draft_policy.cpp:190-196`): a draft an arm never made never enters the `a(p)` table, and the prior
`1 + 0.7*(t-1)` (`draft_policy.cpp:118-121`) stands for its width until a round measures one. Not modelled:
Strata's `observe_chain`/`observe` call *sequence*, which is Strata's own choice of one call a round; the
replay feeds one per round (`code`: `generate.cpp:16776-16778`).

Limits to name on every report (the tool prints them): (1) an arm at ceiling `ceiling+1` with **min-p 0** is
needed to see the drafts the policy would have wanted past the ceiling or below the floor; (2) a trace has only
the widths it ran, so a width's cost is estimated from rounds that ran at it, not measured;
(3) a truncated round's `acc` is bounded by the chosen width, inferred as `min(acc, chosen-1)+1`, not
observed; (4) the round time is that round's own, which the reference's policy never saw (Strata's is
start-to-end); (5) the process-wide state means arms interleave only when one process a trace; (6) **the trace
pays a host cost**: the head's top probability per draft (~0.8 ms a draft, `llama_spec.cpp:354-361`), absent
from the trace-off serving path, so a trace-on arm's per-width costs are inflated and the replay leans
narrower - cover with the trace-off arm per unit (7.4); (7) **T1 is never measured by a trace** (`code`): a
round's `rows` is 1 + its drafts, and a chunk drafts at least one while the budget allows, so width 1 appears
only at a budget end, and `--llama-mtp 0` writes no trace - cover with the MTP-off leg's decode t/s, T1's
round ms = 1000/t_s (`inferred`), passed by `--seed-costs`; and the T1 seed lands beside measured costs that
were paid with the trace on, so T1 is under-priced against them by limitation 6's host cost (`code`: trace
writes only with `--llama-mtp N > 0`; the MTP-off leg has no trace); (8) the policy's state is process-wide
and the trace's `req` restarts per process, so the replay keys a request by (file, req).

Its `--seed-costs` takes Strata's `STRATA_SPEC_COST_TABLE` form (`generate.cpp:16495-16511`, weight 6 rounds,
`draft_policy.cpp:66-74`); without a seeded or measured cost a width is Strata's `kShape` guess, which the
report warns about by name. `--costs-from <leg traces>` measures the fixed-width legs' per-width mean round
ms by context bucket (`measured_costs`, weight 6) and seeds those; an explicit `--seed-costs` value for a
width wins, and both lose to a round of the arm's own that measured the width. The legs are replayed **alone**
first to read their per-width ms, then the arm with `--costs-from`: an arm and its legs in one history would
learn the arm's first rounds by `kShape` and pour the legs' rows into the arm's histogram, E and `a(p)`
(`code`: the replay's own `seed_costs`, `draft_policy.cpp:66-74`). The `--json` report carries the
per-width sums under `tp_tv_vs_round_duration` (renamed from an earlier `tp_tv`): the round's own ms against
propose + verify, which the open question of §5.4 measures.

### 7.4 Card commands (one window each card; honor `docs/sop-card-window.md` first)

Order: Flash-Next on the B60 (GPU.0) first - its row cost is the least linear - then the agent (B60, GPU.0),
then the coder on the A770 (GPU.1). Ceilings are the unit's +1 (§6 step 1: it needs the drafts the ceiling would
truncate), min-p 0 and the unit's own. One fresh process an arm, no other lane. Four arm classes a unit: the
ceiling+1 arm (the drafts the ceiling would truncate), the unit's own floor at the same ceiling, a
**trace-off** decode next to a trace-on one (limitation 6: the trace pays ~0.8 ms a draft on the host), and an
**MTP-off** decode (limitation 7: T1 has no trace round; its decode t/s is T1's ms a token). The
**fixed-width legs** (widths N -> rows N+1, so N runs 1..ceiling+1, one process a width, trace on: the
rows the floor-free ceiling+1 arm drafts and then truncates) measure the per-width round times
Strata's cost table needs - without them a width the arms never ran keeps Strata's `kShape` guess, which the
report warns about. `<arcint>` is the recipe build's binary. Assets as in 5.4's method. The replay procedure
after the sweeps, per unit: replay the legs **alone** for their per-width ms, then replay the arms with
`--costs-from` those legs and `--seed-costs` carrying T1 - an arm and its legs in one history would learn the
arm's first rounds by `kShape` and pour the legs' rows into the arm's histogram, E and `a(p)` (7.3).

Flash-Next (unit: ceiling 3, min-p 0.5; four arms, four legs, the MTP-off leg):

```
ARCINT_MTP_TRACE=/tmp/mtp-fn-4-p0.txt  tools/fn_sweep.py <arcint> <flash-next.gguf> <assets> fn-4-p0 --unit flashnext --set --llama-mtp 4 --set --llama-mtp-min-p 0
ARCINT_MTP_TRACE=/tmp/mtp-fn-4-p05.txt tools/fn_sweep.py <arcint> <flash-next.gguf> <assets> fn-4-p05 --unit flashnext --set --llama-mtp 4
ARCINT_MTP_TRACE=/tmp/mtp-fn-3-p0.txt  tools/fn_sweep.py <arcint> <flash-next.gguf> <assets> fn-3-p0 --unit flashnext --set --llama-mtp-min-p 0
tools/fn_sweep.py <arcint> <flash-next.gguf> <assets> fn-3-traceoff --unit flashnext
tools/fn_sweep.py <arcint> <flash-next.gguf> <assets> fn-mtp-off --unit flashnext --no-mtp
for N in 1 2 3 4 5; do ARCINT_MTP_TRACE=/tmp/mtp-fn-w$N.txt tools/fn_sweep.py <arcint> <flash-next.gguf> <assets> fn-w$N --unit flashnext --set --llama-mtp $N --set --llama-mtp-min-p 0; done
```

Agent (unit: ceiling 5, no min-p on the unit - so its own floor is 0; the ceiling+1 arm, six legs):

```
ARCINT_MTP_TRACE=/tmp/mtp-agent-6-p0.txt tools/fn_sweep.py <arcint> <agent.gguf> <assets> agent-6-p0 --unit agent --set --llama-mtp 6
tools/fn_sweep.py <arcint> <agent.gguf> <assets> agent-5-traceoff --unit agent
tools/fn_sweep.py <arcint> <agent.gguf> <assets> agent-mtp-off --unit agent --no-mtp
for N in 1 2 3 4 5 6; do ARCINT_MTP_TRACE=/tmp/mtp-agent-w$N.txt tools/fn_sweep.py <arcint> <agent.gguf> <assets> agent-w$N --unit agent --set --llama-mtp $N; done
```

Coder (unit: ceiling 4, no min-p; the A770, five legs):

```
ARCINT_MTP_TRACE=/tmp/mtp-coder-5-p0.txt tools/fn_sweep.py <arcint> <coder.gguf> <assets> coder-5-p0 --unit coder --device GPU.1 --set --llama-mtp 5
tools/fn_sweep.py <arcint> <coder.gguf> <assets> coder-4-traceoff --unit coder --device GPU.1
tools/fn_sweep.py <arcint> <coder.gguf> <assets> coder-mtp-off --unit coder --device GPU.1 --no-mtp
for N in 1 2 3 4 5; do ARCINT_MTP_TRACE=/tmp/mtp-coder-w$N.txt tools/fn_sweep.py <arcint> <coder.gguf> <assets> coder-w$N --unit coder --device GPU.1 --set --llama-mtp $N; done
```

Then, per unit, the report. First the legs alone - the `fixed` lines give each width's mean round ms, and a
WARNING says which width the replay would still guess:

```
python3 tools/mtp_width_replay.py /tmp/mtp-fn-w1.txt /tmp/mtp-fn-w2.txt /tmp/mtp-fn-w3.txt /tmp/mtp-fn-w4.txt
python3 tools/mtp_width_replay.py /tmp/mtp-agent-w1.txt /tmp/mtp-agent-w2.txt /tmp/mtp-agent-w3.txt /tmp/mtp-agent-w4.txt /tmp/mtp-agent-w5.txt /tmp/mtp-agent-w6.txt
python3 tools/mtp_width_replay.py /tmp/mtp-coder-w1.txt /tmp/mtp-coder-w2.txt /tmp/mtp-coder-w3.txt /tmp/mtp-coder-w4.txt /tmp/mtp-coder-w5.txt
```

Then the arms, with those measured legs as the seed and T1 from the MTP-off `SWEEP <tag> decode-mean <t/s>`
line: T1's round ms = 1000 / t_s (`inferred`), the same for each context bucket until the legs measured a
`T1` of their own:

```
python3 tools/mtp_width_replay.py /tmp/mtp-fn-4-p0.txt /tmp/mtp-fn-4-p05.txt --costs-from /tmp/mtp-fn-w1.txt /tmp/mtp-fn-w2.txt /tmp/mtp-fn-w3.txt /tmp/mtp-fn-w4.txt --seed-costs "<1000/t_s>;<1000/t_s>;<1000/t_s>"
python3 tools/mtp_width_replay.py /tmp/mtp-agent-6-p0.txt --costs-from /tmp/mtp-agent-w1.txt /tmp/mtp-agent-w2.txt /tmp/mtp-agent-w3.txt /tmp/mtp-agent-w4.txt /tmp/mtp-agent-w5.txt /tmp/mtp-agent-w6.txt --seed-costs "<1000/t_s>;<1000/t_s>;<1000/t_s>"
python3 tools/mtp_width_replay.py /tmp/mtp-coder-5-p0.txt --costs-from /tmp/mtp-coder-w1.txt /tmp/mtp-coder-w2.txt /tmp/mtp-coder-w3.txt /tmp/mtp-coder-w4.txt /tmp/mtp-coder-w5.txt --seed-costs "<1000/t_s>;<1000/t_s>;<1000/t_s>"
```

An arm's own rounds overwrite the seed where it ran that width, so the seed is a starting value, not a lock;
the report's `learned cost table` shows which entry is which. §5.4's reversed second round is
`fn_sweep.py --reverse` on the second sweep; replay those traces the same way (legs to `--costs-from`, arms in
the report).

and per unit the answers' check (`served_check.py` asks the capital, the needle, the answers; 5.4's items 4 and
5), single-lane check from the log's `n_uneven_` line.

Not run here: no card was touched this round; every command above waits for Claude's window.

### 7.5 Rounds 2 and 3: the review's findings, all fixed in the one commit

**Round 2**

1. *(high)* `mtp1` priced a round with the **previous** round's duration. Fixed: the record is buffered one round
   and written at the next round's start; a request's last round measures to the decode loop's exit; the format
   tag is `mtp2`. Evidence: `code` (`backend_llama.cpp:1762, :1789-1795, :1902-1907`).
2. *(high)* the first round wrote microseconds (the clock was set just before the first start) and the replay
   learned it as a width cost. Fixed in 1; the replay treats `ms <= 0` as unknown and learns no cost from it,
   reported as `unknown_ms`. Evidence: `code`, device-free red test `UnknownTimes`.
3. *(medium)* at min-p 0 the trace computes `DraftHead::top_prob` per draft (~0.8 ms a draft on the host,
   `llama_spec.cpp:354-361`), absent from the serving path: a trace-on arm's width costs are inflated toward
   narrower widths. Fixed by naming it as limitation 6 in 7.3 and adding a trace-off arm per unit in 7.4.
   Evidence: `code` for the host pass; the inflation's size is `inferred`, not measured this round.
4. *(medium)* the replay had no way to take Strata's measured cost table. Fixed: `--seed-costs` parses
   `STRATA_SPEC_COST_TABLE` (`Strata-ref:src/program/generate.cpp:16495-16511`) into the three buckets at weight
   6 (`draft_policy.cpp:66-74`), and 7.4 runs fixed-width legs so the table is measured, not guessed. Evidence:
   `code` both sides.
5. *(medium)* `exp_v` divided the mean expected tokens by the **mode** width's cost, mixing two widths. Fixed:
   the policy's rate is the mean of its own `best_v` (E over the cost of the width it would verify, `code`:
   `draft_policy.cpp:199-224`); the truncated rate is its tokens over its chosen width's cost, summed. Red
   test `FixedWidthNumbers.test_the_policy_rate_is_its_own_e_over_cost_per_round`.
6. *(medium)* the doc and the tool's docstring said the `--unit` tables equal `contrib/systemd` **and**
   `served_check.py` for Flash-Next - they do not (7.1 item 8). Fixed: 7.1 names what each table is.
7. *(low)* `kProbes` is `keep_draft`'s own probe (`draft_policy.cpp:216, :223`), now named at each of the
   replay's ported functions; the fixture's first line had a kept draft at `p = 0.42` below a min-p 0.5 floor,
   which the writer cannot make - now `0.66`, and `parse_line` rejects a kept draft below the floor (red test);
   `mtp_trace_off_by_default` cleared the variable it depends on; a fifth C++ case turns the writer on and
   checks the file opens, appends and flushes. Evidence: `code`.

**Round 3**

1. *(medium)* §7.4's replay command put a policy arm and its fixed-width legs in ONE history: the costs are
   learned online in file order, so the arm's first rounds priced by `kShape` and the legs' rows flowed into
   the arm's histogram, E and `a(p)` (`code`: `replay`'s per-record `observe`). Fixed: the legs are replayed
   alone for their per-width ms, then the arm with the new `--costs-from <leg traces>`, which builds the seed
   table from the legs' measured per-width means by context bucket; requests are keyed by (file, `req`) as the
   counter restarts per process. Red tests `CostsFrom` (the legs' means, the CLI wiring, the request keying).
2. *(medium)* `exp_tok` took `st.e` after the whole chain; `keep_draft` explores one or two drafts past
   `st.best_rows`, so E was overstated where the probe was rejected. Fixed: `ChainState` keeps `best_e` beside
   `best_rows` and the report's E is the chosen width's (`exp_tok`, JSON `expected_e_rounds`). The reviewer's
   chain: E 1.90, 2.62, 2.764, `best_rows` 3, so E = 2.62. Red test
   `PolicyExpectedTokens.test_e_is_at_the_chosen_width_not_the_chain_end`.
3. *(medium, window design)* T1 has no trace round (7.3 limitation 7) and `--llama-mtp 0` writes none. Fixed:
   named as limitation 7 and the report prints it; `fn_sweep.py --no-mtp` drops the four MTP flags (default
   unchanged, `--print-flags` for the device-free check) and 7.4 gives each unit an MTP-off leg whose decode
   t/s is T1's ms a token, passed by `--seed-costs`. Red tests `T1Unmeasured`, `FlagTable`; the t/s-to-ms
   conversion is `inferred`.
4. *(low)* named `kProbes` (`KPROBES` at `draft_policy.cpp:22`, used in `keep_draft`); the unknown-`ms`
   comments say `observe` is skipped entirely; the stale `generate.cpp:18176` citations now read
   `:16495-16511`; `cost_table_text` names a bucket once per bucket as Strata does (`draft_policy.cpp:105-111`,
   red test `CostTableText`); 7.1 item 6's refs are the current ones (`:2199, :2269, :1847, :2317`); this
   section sits after 7.4; the reversed-round sentence names `fn_sweep.py`. Evidence: `code` both sides.
