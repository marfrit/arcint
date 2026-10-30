# research-mtp-width-policy: the MTP window width (concept-mining Wave 2, item 13)

Status: research + build plan, no code. Date 2026-10-30. Tree: qfndev at d725dd6.
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
  (`STRATA_SPEC_COST_TABLE`, `generate.cpp:18176`) starts the costs known.
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
times by width (Strata's `STRATA_SPEC_COST_TABLE` form, `generate.cpp:18176`) and let the EMA take over.

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
