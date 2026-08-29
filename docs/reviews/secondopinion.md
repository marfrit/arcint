# Second opinion — outside review at f96d49e

Reviewer context: the 2026-08 fleet campaigns (ARCstory B5–B15), the OpenVINO
issue archaeology (#37607, #4367, CVS-162891), and the production baselines on
both cards. Read-only review of HEAD `f96d49e`; the dirty worktree files were
not reviewed. Nothing in the tree was touched except this file.

## Verdict

This is disciplined work of a kind the upstream projects it replaces do not
practice: every claim measured, every dead end recorded with its evidence, the
equivalence suite has anti-inertness sub-gates (non-zero acceptance, cache-hit
presence) that most CI never thinks of, and graph surgery refuses to proceed
rather than guessing (`slice_logits_to_last_token`'s operand checks are how a
rewrite should look). The MTP head reconstruction with acceptance-as-oracle and
the ablation table is publishable. M0–M5 by exit criteria is credible.

The findings below are ordered by how much they matter. The first one is the
only one I would stop the line for.

## Findings

### 1. The chunked-prefill story contradicts itself at HEAD — in your favour

Three artifacts disagree:

- `tests/equivalence/run.sh` (§ chunked): "not bit-exact on the OpenVINO GPU
  backend … a property of the backend, not of arcint", chunks 1/7/64
  *reported, not gated*, "default is unchunked".
- DESIGN §3.2: same claim, `--prefill-chunk` "defaults to 0".
- `backend_ov.cpp` `prefill()`: "Verified byte-identical to a single call at
  chunk sizes 32/64/128 … safe to use by default", and `config.cpp` ships
  **default 2048**.

The resolution is visible in `forward()`: the divergence had a root cause, and
it was arcint's — the language model was handed the embeddings request's own
output buffer, two InferRequests aliasing one tensor. Since that fix the C++
path is byte-identical at 32/64/128. Which means:

- The "backend property" claim is very likely a **mis-attribution of your own
  (fixed) aliasing bug**. The pure-Python driver that produced the ±2.8 logit
  deltas plausibly aliased the same way; that measurement should be re-run
  before the claim survives anywhere.
- Chunked prefill should be **gated again, not reported**: the suite currently
  documents an expected failure that your own code comment says no longer
  exists. A gate that expects failure and sees success stays green — this one
  narrates a wrong world.
- DESIGN §3.2 and run.sh need the correction; keeping the old text means the
  next reader configures away a feature that is both exact and (your own
  measurement) necessary for activation memory.
- Bonus once re-verified: if chunking is exact, the strongest objection to the
  §3.4 warm-restore story disappears — a prefix hit is mathematically a chunk
  boundary, and exact chunking is what makes warm-equals-cold hold *by
  construction* instead of by margin.

### 2. Features are gated alone, never together

run.sh gates prefix cache, drafts, MTP, slice — each against a plain baseline.
No gate runs **combinations**: MTP × prefix-cache (the rope/KV-length split in
`mtp_seek` exists precisely for that path and is untested), draft × chunked
prefill, prefix-cache × chunked, MTP × stop-mid-draft. The single-feature
equalities do not compose automatically — the mtp_seek bookkeeping is exactly
the kind of code that is right alone and wrong under composition. A small
matrix (even 4 combinations) would close the biggest remaining hole in an
otherwise excellent suite.

### 3. Your #37607 measurement disagrees with the fleet's — that is upstream gold

Fleet B14 (A770, GenAI VLMPipeline, `CACHE_MODE=OPTIMIZE_SPEED`, 0.9 GB blob):
warm import **works**, 63 s. Your finding (B60, `core.compile_model` with
`weights_path`, OPTIMIZE_SIZE): the only fast import observed was the **broken**
one, so you prove every cached load with a real forward and discard on failure.
Same bug family, different device/API path, opposite workaround outcome. That
device/path dependence is exactly what the assigned Intel maintainers need on
[openvinotoolkit/openvino#37607] — a short comment with your table would make
the fleet's existing comment materially stronger. Independent of upstream: your
warmup-prove-and-discard guard is the right pattern and strictly better than
trusting any mode; keep it even after a fix lands.

### 4. Cancel mid-accept leaves absorbed drafts in the state — safe today, by luck of structure

In the accept loop, `commit()` returning cancel breaks out *after* the verify
pass has already advanced the graph state over the full draft tail, and no
rollback runs on that exit. Today this is harmless because every entry path
(`prefill` after `reset_state()` or a cache restore) rebuilds the state before
reuse. That invariant is one refactor away from silently breaking — a
one-line `restore_tensors(rollback_)` on the early exits (or an assertion at
generate() entry that the state is fresh) turns "safe by current structure"
into "safe by construction". Same class, smaller: a failed `restore()` inside
the accept path leaves `mtp_len_`/`mtp_pos_` advanced past what the base model
kept.

### 5. The per-prompt snapshot tax is real and unmeasured in the stats

With the cache enabled, every prompt crossing one block serialises the full
state (GDN floor: 69.9 MiB MoE / 171.3 MiB dense, plus growing KV) through
`get_state` host copies — the same PCIe-class cost your M4 measurement proved
is 66% of speculative decode, paid here once per request on the *prefill* side.
`stats` has no field for it, so it hides inside prefill_seconds. Surface it
(snapshot_seconds) before someone benchmarks "prefix caching on" against
"off" and mis-attributes the delta. The paged path retires this too; until
then the console should show what the cache costs, not only what it saves.

### 6. Doc drift, minor

README's feature list still carries `[M0]`/`[planned]` markers from the stub
era (prefix caching "[planned]" — it is implemented and gated); llm.txt says
M0–M5 met. One of them is lying at any given moment; the README markers lost.
Also `status_.mtp_enabled = false;  // M4` is now wrong at HEAD when a head
loads.

## The Kniffe catalogue — what maps onto these two cards

Requested explicitly: which tricks from NInfer / vLLM / llama.cpp / OpenVINO
are reachable on A770/B60. Ranked by expected payoff over effort.

1. **Static-shape decode graph** (NInfer "exact-batch CUDA-graph decode" ≙ OV
   static shapes). Your own kernel excursion already proved the mechanism: at
   static `[1,512,32,128]` the plugin picks `permute_f_y_axes`, at `S=1` it
   deletes the transpose *entirely*, and 18–27% of decode is `permute_ref`
   copies that exist only because shapes are dynamic. A second compiled model
   with static decode shapes (S=1, and one per draft width if MTP is on),
   sharing weights via `ov::CompiledModel` from the same `ov::Model`, is the
   OV analogue of a captured CUDA graph — it attacks the same waste your
   custom kernel could not (fusion barrier) from the side the optimiser
   likes. This is the cheapest large win visible in your own profile.
2. **PagedAttention for the ten SDPA layers** (vLLM). Your depth measurement
   (IndirectSDPA L^1.46, +13.7 ms at 4k) is the motivation; OV ships
   `ov::pass::SDPAToPagedAttention` and the GPU plugin's paged kernels are
   what GenAI's CB path uses. This also unlocks `KV_CACHE_PRECISION` (your
   retype surgery becomes unnecessary) including **fp8 KV on Xe2** — and it
   retires findings 4 and 5 and the M4 rollback tax in one move. Every road
   in this review converges here.
3. **Hot/cold expert placement for the 35B on the A770** (llama.cpp `-ncmoe`,
   PowerInfer's insight, and the fleet's own data). The M5 gap is 17.4 GiB
   against 15.1: only ~2.5 GiB must live elsewhere. The fleet already owns
   per-expert activation statistics for exactly these models — the imatrix
   corpus used to prune the coder from 256 to 184 experts. Pin the coldest
   ~15% of expert FFNs host-side (CPU-compiled subgraph or streamed weights)
   and the 35B fits the A770 without a new artifact. PCIe x4 hurts only on
   cold-expert hits, which the statistics say are rare by construction.
4. **Host-tier state retention** (NInfer "Device/Host State"). Your prefix
   cache already serialises to host memory — a second, larger LRU ring on
   host RAM for evicted entries costs little and turns the 64 GiB of data's
   host memory (ARC-capped, verified) into cache depth. Only worth it after
   the paged path makes snapshots cheap.
5. **lm-head drafting** (NInfer `--lm-head-draft`) as a weightless drafter for
   the MoE pair, where the MTP head does not exist and your measurement says
   verify does not amortise (1.37×) anyway — a draft source with zero verify
   width cost may still lose; measure once, keep the ngram drafter otherwise.
6. **Batched console audit** (NInfer's 225-response audit). The Prüfstand is
   10 items; NInfer's honesty artifact is the *scale* of its audit. A nightly
   200-prompt greedy sweep with sha256s, diffed run-over-run, would catch
   regressions the 10-pointer cannot — and it is pure scripting.

### B60 extra instructions (Xe2) — verify before believing

The fleet record says A770/XMX: locked out on every path tried; ceiling is the
card. Xe2 (BMG) added instructions and reworked the EU ISA. What matters here
is not the datasheet but one observable: **whether oneDNN/OCL kernels on the
B60 engage the systolic path where the A770 could not**. Your
`ARCINT_PROFILE` already prints `exec_type` per kernel — grep a B60 profile
against an A770 profile of the same model for `dpas`/`systolic`/`xmx` markers
in the GEMM kernels (`jit:gemm:*`). If the B60 engages XMX:

- weight-heavy ops (lm_head, expert FFNs) should be *expected* to scale
  differently per card, and the static-shape decode graph (Kniff 1) should be
  tuned per card, not shared;
- fp8 storage (KV via the paged path, possibly activations) becomes a real
  Xe2 option where the A770 has none — worth one measurement, not a port.

If it does not engage, that is also worth knowing: then the B60's advantage
stays bandwidth+capacity, the 1.35× card-over-card numbers are already the
ceiling, and no ISA chase will move them.

## Priority recommendation

One sentence: **re-verify and re-gate chunked prefill (finding 1), add the
combination gates (finding 2), then spend everything on the paged path — items
2, and with it findings 4/5 and the M4 rollback — with the static-shape decode
graph (Kniff 1) as the parallel quick win.** The kernel excursion is correctly
closed; do not reopen it. The 35B/A770 expert-spill (Kniff 3) is the only item
here that needs fleet data rather than engine work — the imatrix statistics
live on a fleet host/the dev host and the fleet session can hand them over on request.

---

## Addendum after the session's response to finding 1

Conceded: chunked prefill is genuinely not bit-exact on this backend — now
demonstrated in-engine, not inferred, and my mis-attribution hypothesis (the
embeddings aliasing bug) is dead. The demonstration is better evidence than my
reading. But that settles the *cause* question and sharpens the *consequence*
question, because two things now hold simultaneously that the review let the
hypothesis excuse:

1. **The shipped default (`--prefill-chunk 2048`) violates the house rule.**
   DESIGN's own §3.4 discipline — "a path that cannot meet the equality gate is
   configured out, not papered over" — now applies to the default
   configuration. Either the default goes back to unchunked (and deep prompts
   pay the activation-memory price the logits slice already softened), or the
   invariant's reference point is redefined honestly (see 2). What cannot
   stand is a default that the suite's own narration calls non-exact.

2. **Every "byte-identical" gate that mixes multi-token and single-token
   forwards is margin-backed, not construction-backed.** If chunk boundaries
   change bytes, then: warm-restore-vs-cold differs whenever the cold run's
   boundary set differs from the warm run's (a prefix hit *is* a boundary);
   and the draft path's forward(1+k) + re-forward differs from the plain
   path's k × forward(1). These gates pass today because greedy argmax gaps
   absorb the deltas — margin, not construction. Margins hold until the day
   they do not, and that day looks like a flaky CI gate nobody can reproduce.

   The constructive fix is a **canonical boundary grid**: fix the chunk
   boundaries at absolute positions (multiples of the chunk size from
   position 0, with snapshot/hit points restricted to grid-compatible
   positions), and make every path — cold prefill, warm continuation,
   speculative re-forward — traverse the *same* boundary set for the same
   sequence. Then equality holds by construction again, on a backend that is
   allowed to be chunk-sensitive, because no two paths ever present the model
   different chunkings of the same tokens. The block-edge snapshot logic in
   `prefill()` is already half of this; the missing half is aligning the
   chunk grid to absolute positions instead of starting it at `past`.

3. Failing that, the fallback is to *state* the contract honestly: equality
   gates hold for the canonical configuration, and margin-backed equality
   elsewhere is monitored statistically (the 200-prompt audit sweep from the
   Kniffe list is the right instrument — sha256 drift over runs is exactly
   how you catch a margin eroding).

The demonstration also deserves an upstream artifact: a minimal reproducer of
chunk-boundary non-exactness on the stateful GPU path is a better-evidenced
cousin of the fleet's #4367 (stateful vs CB divergence) and likely shares its
root. Two measured data points from one fleet beat either alone.
