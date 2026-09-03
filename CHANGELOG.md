# Changelog

The human record of what each release changed and what it depends on. The
package record lives in `contrib/packaging/arcint/debian/changelog` (one
entry per built package) and the plugin patch record in
`contrib/packaging/marfrit-openvino/patches/README.md`; this file names the
same releases by tag and adds what a reader upgrading needs first: the
runtime dependency. Every number here was measured; the measurement
protocol and the retractions are in `DESIGN.md` §7.

Runtime dependency throughout: `marfrit-openvino`, a source build of
OpenVINO at the pinned upstream commit `71640275` (the 2026.4.0 nightly of
2026-08-21) with the patch series in `patches/` applied. The patch level
is part of the package version (`+p1`, `+p2`) and a release names the level
it was built and measured against; the dependency is strict because a
different nightly is a different ABI.

## Unreleased

### Added since 0.2.13 (unreleased)
- `--paged-attention-max-partitions N` (default 0, unbounded): engine side
  of plugin patch 0015. The plugin side is patch 0015
  (`patches/0015-paged-attention-bounded-partials.patch`), written and
  under device test; it is in no built package yet, so the bound is inert
  on every released plugin. Passed to the GPU plugin as its own
  RW config key `PAGED_ATTENTION_MAX_PARTITIONS` when a 4-bit-values load's
  compiled plugin accepts it -- detected by reading the key back before
  compile (an unrecognised-key throw is the ORDINARY case on every
  released plugin today, not a rare failure: this is not the same
  situation as the two known, typed GPU properties this file already reads
  elsewhere, which essentially never throw; logged ("plugin does not carry
  the bound; scratch term stays depth-scaled") and N is ignored, the load
  proceeds exactly as it does today) rather than the
  VALUE_CACHE_PRECISION pattern of setting-and-letting-compile-fail, since
  the default (0, no bound requested) must never refuse a load against an
  unpatched plugin. A compile failure with the key accepted names
  PAGED_ATTENTION_MAX_PARTITIONS explicitly now (and the compound case,
  together with an accepted VALUE_CACHE_PRECISION, names both rather than
  guessing which the plugin actually rejected) -- an earlier version of
  this catch could misattribute that failure to VALUE_CACHE_PRECISION
  alone. DETECTION CONTRACT: `element_bytes` follows the RW bound key's
  own acceptance directly, not a second probe -- a plugin exposing
  `PAGED_ATTENTION_MAX_PARTITIONS` carries the f16 host-sizing fix too,
  both ship together in patch 0015; a plugin with the key but without the
  sizing fix would be under-charged by half (4-byte elements priced as
  2-byte), and no such plugin exists in this series. (A round of review
  tried a second, independent read-only key for this and retracted it: the
  plugin implementer keeps 0015 as one unit and will not carry a separate
  key.) The plugin declares `PAGED_ATTENTION_MAX_PARTITIONS` as
  `Property<size_t, RW>`; the value handed to the compile-time property
  map is now cast to `size_t` explicitly (`cfg.paged_attention_max_
  partitions` is `int`, and an `int` payload inside `ov::Any` can fail
  this plugin's own typed read-back where the same bit pattern read as
  `size_t` succeeds -- the device run that accepted this property predates
  the explicit cast, so its success is not evidence an `int` payload is
  safe on a future plugin build).

  UPGRADE NOTE: this plugin level changes the GPU model-cache blob schema
  -- 0015 inserts an option into a positionally-serialised property list --
  so clear the GPU model cache (`--cache-dir`, if set) when upgrading to a
  plugin level carrying patch 0015. (This engine build itself always
  compiles with `ov::cache_dir("")`, i.e. the blob cache is off for the
  paged graph regardless -- see backend_ov.cpp -- so this note is for any
  operator-set `--cache-dir` on a non-paged path sharing the same plugin
  install.)

  `exec/fit.h`'s packed-4-bit-values scratch term (`packed_values_prefill_
  scratch_bytes`, `prefill_chunk_cap_for_packed_values`) is generalized to
  take `partitions` and `element_bytes` directly instead of deriving
  `ceil(n_ctx/256)` and hardcoding 4 bytes unconditionally -- `_ex`
  variants carry the new parameters, and the old signatures are thin
  wrappers reproducing today's unbounded, 4-byte behaviour byte for byte
  (regression-tested directly against the already-established fixtures).
  A second additive term joins tmp_out: exp_sums + max_logits, two f32
  buffers (`2 x chunk x heads x 4 B x partitions`, NOT scaled by the 1.5x
  margin -- that margin covers tmp_out's own measured resize alone),
  charged in both arms. When the plugin accepts the bound key: partitions
  = `min(ceil(n_ctx/256), N)` when N > 0, else unbounded. Bounded, the
  whole term (tmp_out + the exp/max pair, at the bound) is a FLAT amount
  once served depth passes a small, fixed partition count (8,192 tokens at
  N = 32) instead of scaling with `n_ctx` forever -- the reservation still
  always charges the 1.5x margin over the buffer's own true size (nothing
  here is "exact" to the byte; what changes is that the MARGINED value
  stops growing past the bound) -- and, because the buffer is per infer
  request rather than per compiled model, the flat charge is multiplied by
  `lanes` explicitly (an earlier version of this term charged it once
  regardless of lane count). The belt (`prefill_chunk_cap_for_packed_
  values`) is generalized the same way and now prices the SAME bounded
  buffer the term charges at both call sites (the fit's own climb and the
  post-Phase-E belt call), or it would halve the chunk for a buffer that no
  longer exists at that size.

  Worked example (N = 32, n_ctx 171,312, chunk 128, the coder's
  16-query-head/256-head_dim shape, one lane): tmp_out alone, bounded --
  32 partitions x 128 x 16 x 256 x 2 = 32 MiB, x1.5 = 48 MiB -- against the
  UNBOUNDED path's tmp_out at the same depth, 1,340 MiB before the 1.5x
  margin and 1.96 GiB after it; comfortably under the 512 MiB belt proxy
  budget at chunk 128 either way the bounded arm counts it, so the fixed
  point never needs the belt to shrink chunk at all (an engineered
  fixture, tmp_out + exp/max together: 1,227,936 tokens/chunk 32 unbounded
  -> 1,643,200 tokens/chunk 128 bounded, same budget). Logged at load when
  the key is accepted: `"4-bit values: plugin bounds attention partials at
  N (f16 partials); scratch term X MiB at chunk C"` when N > 0,
  `"...plugin carries the bound key; N = 0 leaves partials unbounded (f16
  partials)..."` when N = 0 (an earlier version read "bounds attention
  partials at 0," which is not what N = 0 means). Key absent: verified
  byte-for-byte identical to the pre-0015 unbounded path (the engine's own
  arithmetic -- the bound is inert on every RELEASED plugin, since patch
  0015 is in no built package yet). The design note's own §B recon -- the
  plugin's mixed-stage buffer is already f16 in the kernel, only the
  host-side sizing hardcodes 4-byte elements -- is read from the plugin
  source, not measured; nothing in this engine-side change measures it
  either, only prices what the note claims.

  MEASURED (2026-09-03, 16 GiB card, coder, u8:i4, the patched plugin
  under device test, `--paged-attention-max-partitions 32`): auto-fit
  lands at n_ctx 165,680 with chunk 128, 48.5 MiB of scratch charged --
  against 101,824 at chunk 64 unbounded on the same card, same binary. An
  8,417-token and a 17,126-token prompt both prefill and decode. A longer
  prompt run is still under investigation (a host-side crash, not a memory
  fault) -- 165,680 is NOT yet confirmed usable at that full depth;
  report it as measured-so-far, not as a cleared ceiling.

### Fixed since 0.2.13 (unreleased)
- Activations charged at the wrong chunk could refuse an explicit --n-ctx
  that auto-fit itself would adopt for the identical request. MEASURED
  (unpatched plugin, coder, u8:i4): --n-ctx omitted adopts 101,824 at belt
  chunk 64; the SAME 101,824 given explicitly was refused. Two rounds of
  fix, the first retracted: a "re-probe activations at the smaller served
  chunk" attempt could not work -- this file's own record is that the
  plugin's intermediate pool grows to the largest shape it has ever seen
  and never shrinks, so a probe taken after a bigger one returns the
  STALE, bigger reading regardless of what chunk is asked for next, and
  the card, on that binary, still refused the identical request a second
  time. The actual fix determines the SERVED CHUNK before the one real
  activation probe ever runs, so it is only ever taken at the chunk that
  will actually be served: an explicit --n-ctx's served chunk is a PURE
  function of the requested depth (the belt reads only depth and
  geometry, never activations or budget), computed with no GPU call at
  all -- via the new `fit_context_packed_values_at_depth` (fit.h), an
  exact, no-search evaluation at a known depth.

  RETRACTED (round-10 review, a REAL defect this repository's own record
  keeps rather than silently editing away, per DESIGN §7.0.1): an earlier
  version of this same fix claimed auto-fit could use the identical
  `fit_context_packed_values_at_depth` call, evaluated at `wanted` (the
  artifact's own train maximum) -- "BOTH paths call the SAME new
  primitive... with no search and no activation estimate." False:
  `wanted` is an upper bound to search FROM, not auto-fit's own served
  depth, and evaluating the belt there gives the smallest, most
  conservative chunk possible. Measured on the card's own constants:
  `wanted` = 262,144 evaluates to chunk 32, while the depth auto-fit
  actually adopts (103,040, well short of the train maximum once weights,
  activations and margin are accounted for) only needs chunk 64 -- an
  explicit request for that SAME 103,040, evaluated honestly at its own
  depth, priced chunk 64 and admitted it, while auto-fit -- pricing chunk
  32 the whole time -- served a different, more expensive reservation for
  the identical depth. The corrected design: `fit_context_packed_values_
  at_depth` at `wanted` remains step one, the conservative ceiling the
  one real activation probe must not explore past; auto-fit's own served
  chunk then comes from running `fit_context_packed_values` (the belt's
  own chunk-driven fixed-point SEARCH, unchanged, seeded from the
  operator's own requested/default chunk) to find the depth D* it
  actually adopts, confirmed by evaluating `fit_context_packed_values_at_
  depth` AT D* -- the same primitive the explicit path uses. When the
  search's served chunk is bigger than what the ceiling-bounded probe
  measured activations at, the engine re-probes activations UPWARD at the
  bigger chunk and re-runs the search (valid, unlike the round-7 defect
  below: the plugin's pool only ever grows to the largest shape it has
  seen, so a probe at a BIGGER chunk than before is a real measurement,
  not a stale downward re-read) -- bounded to terminate in a handful of
  rounds, since the served chunk only grows and is capped above by the
  operator's own requested chunk. New tests drive the real search and the
  real at_depth call directly (not two calls to the same function, and
  not a locally-modelled slack) and assert the served chunk is 64, not
  32, at the card's own weights/total/margin constants WITH a specific
  engineered activation figure of 1,200 MiB (this repository's own
  invented number, not a card reading -- the "64" result depends on
  which activation level is chosen, not on the card constants alone; a
  second activation level, 1,800 MiB, in the same test settles at chunk
  128 instead) (`tests/test_fit.cpp`).

  (An intermediate design that DID estimate activations analytically, to
  drive a chunk-ceiling SEARCH for auto-fit, was tried and retracted in
  an earlier round: it failed open when the estimate left no budget at
  all, and even fixed, the belt's own "return the floor chunk even if it
  does not truly fit" behaviour let the search wander to chunks and
  depths several times too deep when checked against the card's own
  constants -- see fit_context_packed_values_at_depth's own retraction
  comment in fit.h for the numbers.)

  A THIRD card refusal, after this fix, surfaced a separate, smaller gap:
  the fit-level admissibility check (`depth <= max_ctx`) is necessary but
  not sufficient -- Phase E's real allocation can still overshoot the
  analytic prediction by a small amount (measured: a 10 MiB, 0.07%
  shortfall for an explicit request equal to what auto-fit itself adopted
  after ITS OWN Phase E trim gave it that margin for free). One KV page of
  slack is now subtracted from `max_ctx` before EITHER path's
  admissibility check, so the auto-fit depth this produces is admissible
  again if resubmitted as an explicit request for the same depth -- proved
  with a fixture that runs exactly that round trip (`tests/test_fit.cpp`).
  Round-10 review, finding 5: `fit.admissible` itself was still computed
  from the PRE-slack max_ctx, so a fit landing exactly at the 4,096-token
  floor could report admissible while the slack-adjusted depth it goes on
  to adopt (4,064) is actually below that floor; `fit.admissible` is now
  recomputed against the same slack-adjusted figure every other check in
  the load path reads.
  Also fixed: the load-time detail log's "per token" figure read a FitTerm
  member that `fit_context_packed_values_at_depth` always leaves 0 (an
  exact-at-one-depth evaluation has no per-token slope to report), so it
  printed "per token 0.0 KiB" on every explicit-n_ctx 4-bit-values load; it
  now computes and prints the per-token rate at the served chunk directly,
  labelled INFORMATIONAL (round-10 review, finding 7) rather than "per
  token" -- it does not, in general, reconstruct the printed MiB figure by
  multiplying against the served depth (the charge is priced at the
  budget ceiling, the rate at whichever chunk was actually served, and the
  bounded arm has no per-token slope of its own charge at all). The same
  detail line's "for n_ctx" field used to print the raw, unclamped budget
  ceiling (`max_ctx`) rather than what the load actually goes on to serve
  -- a 24 GB-card trace showed "n_ctx 929456" on a load whose adopted
  depth was orders of magnitude smaller (bounded-path review) -- now
  `min(wanted, max_ctx)`. Also (round-10 review, finding 4): Phase B's
  own expert-slot-pool plateau probe (`--offload-ratio > 0` only) ran at
  a fixed 128-token floor regardless of the packed-values belt's own
  ceiling, so it could itself already exceed the served chunk the same
  way the activation climb's floor probe was fixed for in an earlier
  round; both probes now share one floor, bounded by the belt's ceiling.
  None of this round's fixes are independently re-measured on a card (no
  ssh); the underlying arithmetic is pinned with engineered fixtures in
  `tests/test_fit.cpp` (`fit_context` itself is unchanged throughout; the
  defect and every correction live in backend_ov.cpp's GPU-measurement
  orchestration and fit.h's new exact-at-depth primitive).
- The packed-4-bit-values belt's own `fits()` check compared the RAW,
  unmargined proxy buffer size against its 512 MiB budget, while the
  reservation actually charges 1.5x that plus the exp_sums/max_logits
  pair -- a bounded-plugin run (chunk 2048, N = 32) showed the belt
  reporting "fits exactly at 512 MiB" while the actual charge was 776 MiB,
  50%+ over budget. A first fix priced the SAME margined formula the
  reservation charges inside `fits()` too.

  RETRACTED in the same release (a REAL defect this repository's own
  record keeps rather than silently editing away, per DESIGN §7.0.1): the
  512 MiB budget constant was calibrated against the RAW proxy (chunk 64
  measured PASSING at n_ctx ~119k, chunk 128 measured FAULTING there),
  not against the margined formula the fix above substituted in --
  margining `fits()` moved the belt off that calibration and made it
  actively WRONG in the direction that matters: `at_depth(101,824)` (an
  explicit request at that depth) now computed chunk 32, while the card
  itself measured and SERVED chunk 64 at that exact depth. `fits()` is
  restored to the RAW proxy the 512 MiB budget was actually calibrated
  against; the 1.5x margin (and the exp_sums/max_logits pair) stay
  exactly where they always were -- in the CHARGED term, not the belt's
  own chunk-choice predicate. The 50%+-over-budget measurement that
  prompted the first fix was real; fixing it needs a separately
  re-measured budget constant for the margined case, which this round
  does not have, not folding margin into a predicate calibrated without
  it.

  Also added, and NOT retracted: a hard, `kMaxMeasuredPackedValuesChunk`
  = 128-token ceiling applied AFTER the budget-based shrink, independent
  of whichever budget predicate is active -- chunk 2048 (the operator's
  own `--prefill-chunk` default) has never been validated on any plugin
  or card for this scratch path at all, so the budget math alone (raw or
  margined) is not sufficient to stop the belt from picking an
  unvalidated chunk when the budget happens to allow one that large. In
  practice, WHEN the belt is active: an explicit `--prefill-chunk` above
  128 under 4-bit paged VALUES is reduced at least to 128 (the largest
  chunk this repository's own record has a passing card measurement
  for), even at a shallow served depth where the RAW budget check alone
  would have left it unchanged. This is NOT unconditional --
  `ARCINT_PREFILL_CHUNK_CAP=off` disables the belt (and this cap with
  it) entirely; an earlier version of this entry said "is reduced to
  128" without that qualifier, which reads as if the cap always applies
  -- corrected here rather than silently, per this file's own
  convention. The load-time log, at the point the chunk is actually
  decided (see the retraction below), names which of three limits
  actually bit -- "budget" (the belt's own 512 MiB proxy shrank it to
  something that fits), "bound" (the budget-driven halving ladder ran
  all the way down to the KV block floor and STILL does not fit -- the
  floor is served anyway, not a refusal), or "measured cap" (the
  128-token ceiling fired after the other two left something bigger) --
  instead of a single "prefill chunk X -> Y" line that left an operator
  guessing which of the three actually moved the number.

  RETRACTED history (three rounds, kept on the record rather than
  silently corrected, per DESIGN §7.0.1): (1) the measurement switch
  (`ARCINT_PREFILL_CHUNK_CAP=off`) was, for a time, a NO-OP on the
  auto-fit path -- the served-chunk ceiling that bounds the one real
  activation probe applied the belt UNCONDITIONALLY regardless of the
  switch, and the served-chunk seed separately re-applied the 128-token
  cap unconditionally too, so a load run with the switch set logged "the
  belt is disabled for this load" and then served chunk 128 anyway;
  fixed by gating both on the switch. (2) The reduction log above used
  to sit at the Phase E belt call site (after --n-ctx clamping and the
  auto-fit trim/replay) -- but by the time that site runs, the served
  chunk is already the fixed point the search or the explicit
  evaluation settled on, and that site's own belt call can only ever
  reproduce the SAME value (the belt never raises its output past its
  capped input, and every path to the final served depth only ever
  lowers it from the depth the term was priced at), so the log there was
  UNREACHABLE; moved to where the chunk is actually decided. (3) The
  explicit --n-ctx path had the SAME "switch is a no-op" bug a third
  time: the at-depth evaluation that path uses had no way to disable the
  belt at all, so the switch was ignored there even after (1) fixed the
  auto-fit path -- fixed by giving `fit_context_packed_values_at_depth`
  the same belt-disable parameter the auto-fit search already had.
  Separately (not a retraction, a related correctness fix): the search's
  own 8-round exhaustion fallback used to re-seed its final evaluation
  from the operator's full requested chunk, which could converge to a
  chunk bigger than any chunk activations were ever actually probed at
  (an under-reservation risk); it now re-seeds from the last chunk
  actually probed, bounding the result by construction rather than by a
  post-hoc clamp.
- `--prefill-chunk` under 4-bit paged VALUES: an engine-side belt
  (`exec/fit.h`'s `prefill_chunk_cap_for_packed_values`) now lowers the
  served chunk from the SERVED pool depth (after --n-ctx clamping,
  --prefix-cache-reserve and the auto-fit replay/trim all run) whenever the
  requested paged-KV value precision is 4-bit -- the class of load the
  "Known defects" `--paged-kv u8:i4` entry below describes. Empirical, not
  a derived mechanism: a proxy formula (chunk x query heads x head_dim x
  4 bytes x ceil(depth/256), a 512 MiB budget) is chosen so that the
  budget yields the chunk measured to pass at the deepest fault (64 at
  119,074 tokens, where 128 faults; 128 passes at 71,689 and 512 at
  35,227, so the proxy is a yardstick, not a threshold); query heads come from the artifact's own
  `num_attention_heads` (GQA-correct -- an earlier version of this fix read
  `num_key_value_heads` first, which undercounts the plugin's own
  query-head-sized buffer 8x on this family and never actually fired the
  cap). Never raises the chunk, never touches u8/f16, floors to
  `--kv-block-size`. Measured on the card (16 GiB-class, coder, u8:i4, the
  171,312-token auto-fit): the belt lowers the chunk 128 -> 32 and the
  119,074-token prompt that faulted at chunk 128 prefills in 705 s
  (169 t/s) and decodes; the root cause in the plugin stays open, so the
  "Known defects" entry below stays as a disclosure of the fault line. Known
  residual: the activation reservation is probed at the PRE-cap chunk (the
  auto-fit climb runs before this belt does), so when the belt fires the
  reservation over-charges activations for a chunk larger than what is
  actually served -- conservative, never unsafe; `/status` reports
  `activation_bytes` (pre-cap) and `prefill_chunk` (post-cap) side by side
  rather than reconciling them. Measured cost (16 GiB-class card, coder,
  u8:i4, n_ctx 131,072, 8,417-token prompt): 495 t/s prefill at chunk 128,
  437 at 64 (-12%), 440 at 32 -- chunk 128 only fits the 512 MiB proxy
  budget up to n_ctx 65,536 (the proxy scales with pool depth, not prompt
  length), so every deeper u8:i4 pool pays this price even on short
  prompts.
  `ARCINT_PREFILL_CHUNK_CAP=off` keeps the requested chunk for
  measuring the fault line (logged loudly; measurement only).
- The reservation itself now charges the packed-4-bit-values prefill
  scratch buffer the belt above only caps -- consistent with (16 GiB card,
  coder, u8:i4, host-side VRAM allocator sampled every 2 s) the same
  mixed-stage attention buffer the belt targets (the sampler measured
  aggregate free VRAM against a proxy formula, not a direct trace of that
  one buffer): it does not exist at past 0, where the activation probe
  runs, so the pre-fix reservation never charged it, and a 171,312-token
  auto-fit pool ran free VRAM to 0 MiB during a 119,074-token prefill
  (`xe: VM worker error: -12`, `exec queue reset detected`,
  `CL_OUT_OF_RESOURCES`) while a 131,072-token pool passed the same prompt
  with 492 MiB to spare. `exec/fit.h`'s `packed_values_prefill_scratch_
  bytes` prices the buffer at the SERVED chunk and depth, x1.5 for the
  measured overlap bound -- the same VRAM sampler shows the buffer held for
  ~0.9x of the whole 119,074-token prefill (round up to 1.0x); at the
  resize itself free VRAM traced 60 -> 562 -> 56 MiB, a RELEASE then a
  REALLOCATE (the old-size buffer dropped, then the new, larger one
  committed), not two buffers held at once, so the resize's own peak stays
  at or below that same ~1.0x rather than adding to it; 1.5x is margin
  ABOVE that observed <=1.0x peak. An earlier version of this term charged
  an unmeasured 2x (a full second copy) before the sampler data was read,
  and is retracted to 1.5x here -- and `fit_context_packed_values` folds it
  into the auto-fit climb as a per-token term (12 KiB/token at chunk 128
  for the coder's 16 query-head/256-head_dim shape, more than the
  8.8 KiB/token u8:i4 KV itself; 6 KiB/token at chunk 64, the belt's own
  choice at this depth) plus a one-partition fixed margin -- charged only
  when the requested paged-KV value precision is 4-bit, at whatever chunk
  the belt (`prefill_chunk_cap_for_packed_values`) would pick for each
  candidate depth tried, so the term and the belt never disagree about
  which chunk is being priced -- clamped to the chunk the term was
  actually priced at at the belt's OWN call site after Phase E (round-2
  review, F1: every path to the served depth only ever lowers it, and the
  belt is non-increasing in depth, so a trim crossing one of the belt's own
  step boundaries could otherwise hand back a chunk larger than the one
  priced -- measured shape: max_ctx 131,104 prices chunk 32, Phase E trims
  one page to 131,072, and the belt asked fresh from the unclamped
  requested chunk would pick 64, a ~383 MiB shortfall against the charged
  term). `ARCINT_PREFILL_CHUNK_CAP=off` prices the uncapped requested chunk
  instead, matching the belt's own disablement for that switch. Logged at
  load ("4-bit values: prefill scratch charged ... at chunk ... for
  n_ctx ... (per token ... + KV ...)") and folded into the one-line
  reservation summary ("+ prefill scratch N GiB (4-bit values, chunk C)").
  Deliberately EXCLUDED from the allocation-time audit's `predicted_total`
  (round-2 review, F4): that figure is what should already be resident the
  instant the KV pool is allocated, and the scratch buffer is not -- it
  grows later, once the first real prefill runs past this point in the
  load -- so including it there made the "deferred commit" log fire on
  every single 4-bit-values load, unconditionally, calling a buffer that
  had not been allocated yet a deferred commit of something that should
  already be there. Phase E's own acceptance ceiling (round-3 review,
  finding 2) is now `total - margin - scratch term`, the scratch term
  re-derived every pass at that pass's own served depth -- so a 4-bit-
  values load can now log "reservation overshoot ... correcting" and trim
  n_ctx, or an explicit --n-ctx can be refused, in a case where the
  PREVIOUS binary (ceiling = `total - margin` alone, not holding the term
  back at all) served silently and left the scratch buffer's own later
  allocation to find whatever was left over. The deferred-commit
  comparison (`observed < predicted_total`, just above) still excludes the
  term -- that check is unchanged and asks a different question ("is it
  resident yet", not "will there be room").

  MEASURED (2026-09-03, 16 GiB card, coder, u8:i4, auto-fit -- this
  milestone's own binary, BEFORE the Phase E ceiling narrowed for the
  scratch term above): the fit lands at n_ctx 101,984 with the belt at
  chunk 64, `"prefill scratch charged 599.1 MiB at chunk 64 for n_ctx
  101984 (per token 6.0 KiB + KV 8.8 KiB)"` -- BELOW the same card's own u8
  auto-fit of 133,456 (M8, §7.0.2 table). u8:i4 no longer widens the
  auto-fit context over plain u8 for this shape once the scratch buffer is
  honestly charged; it still wins on KV bytes/token in isolation, but
  auto-fit is what an operator without an explicit --n-ctx actually gets.
  (An earlier estimate in this entry, before the term's chunk-clamp fix
  above landed and before this card run, said 127,680 tokens at chunk 32 --
  wrong on both counts: 127,680 is not even a fixed point of the belt
  (128 -> 72,320 -> belt 64 -> 6 KiB/token -> ~101,728 -> belt 64 again),
  and the measured figure supersedes it.) Verified at that depth with the
  host's VRAM allocator sampled every two seconds: a 97,727-token prompt
  prefills (443 s, 220 t/s) with a free-VRAM floor of 818 MiB, where the
  uncharged 171,312 pool reached 0 MiB and faulted -- MEASURED BEFORE THE
  CEILING NARROWED; the ceiling fix above can only pull the adopted n_ctx
  down further (or refuse where it used to adopt), never up, so 101,984
  and everything verified against it here are an upper bound, not the
  current number. Re-measured below.
  Re-measured with the narrowed ceiling (same card, same prompt): the overshoot correction fires once at load and the served depth is n_ctx 101,824 (10 pages fewer); the 97,727-token prompt prefills in 442 s (221 t/s) with a free-VRAM floor of 814 MiB, no driver fault.
- The DFlash2 drafter past 2,048 tokens: plugin patch 0014 lets an Assign
  adopt a same-type, same-rank output layout instead of asserting (the
  served head now drafts at 17k-token prompts and through 3,000-token
  decodes); the export's state trim takes a runtime slice start and the
  export tool gained `--compress int4` with the recipe verified
  byte-identical against the served head; the engine's drafter disable is
  per lane and re-armed on the next request, with a feed cap one row below
  the window for plugins without 0014.

### Known defects in 0.2.13, found by the long-context window (2026-09-03)
- `--dflash` at depth: on the same card and model the DFlash2 drafter
  reaches 1.69 / 1.43 / 1.00 tokens per cycle at 8.9k / 37.7k / 76k tokens
  and decodes below plain at each (19.8 vs 22.3, 11.0 vs 19.9, 5.3 vs 16.3
  t/s); the zero acceptance at 76k is not yet explained. Serve deep
  contexts without a drafter.
- `--paged-kv u8:i4` prefill price (the M8 item owed since 0.2.13): +7% /
  +25% / +72% of prefill time at 8.9k / 37.7k / 71.7k tokens on the coder,
  decode at parity; by code reading it is the opt attention path u8:i4 is
  forced onto, not yet measured as such.
- `--mtp on` at depth: on the 24 GB card the dense 27B agent's MTP verify
  accepts 79% of drafts at 37.7k tokens (12.6 t/s) but 0% at 76k and 143k
  tokens, where decode falls to 1.0 and 0.3 t/s against plain decoding's
  16.3 at 76k, and its prefill runs 13% slower than plain there and at
  half plain's rate at 143k; the cause
  is not yet measured (by code reading the reconstructed MTP layer keeps an
  unpaged state and a dense mask over the whole context). Timed at 76k
  (64 greedy tokens, one process per arm): the verify forward of the main
  graph takes 763 ms per step against 97 ms for a plain decode step, and
  the per-node profile at that depth over-counts the step about 35x and is
  flat across one, two and four tokens, so it cannot see this;
  `ARCINT_PROFILE_MWALL` (new) times the forward per token count at depth
  instead: at 76k a 1-token forward takes 104 ms and 2, 4 or 8 tokens all
  take about 209 ms (at depth 1: 80 vs 86-88 ms), so every verify forward
  costs two plain steps at depth -- the drafter tax that puts DFlash at
  half plain with 0% acceptance -- and the MTP-serving forward's 763 ms
  against the plain graph's 211 ms for the same two tokens is not yet
  explained. Serve deep contexts with `--mtp off` until it is.
- `--paged-kv u8:i4`: a 141,902-token prefill on the coder (16 GiB card,
  auto-fit 171,312) failed with a GPU out-of-resources error and every
  later request in that process failed with it; prompts up to 8,909
  tokens succeed at every depth setting tried. The failing depth lies
  between those two numbers; bracketed one process per depth: 35,227 and
  71,689 tokens prefill and decode, and the same 119,074-token prompt that
  u8 prefills on this card fails at u8:i4, so the fault is specific to the
  u8:i4 prefill path. An earlier
  version of this entry said "every prompt over about 2,048 tokens", which
  was an artifact of running the depths deepest-first in one process and
  is retracted. The fault moves with the prefill chunk (one process per
  cell, the same 131,072-token pool): 128 faults and 64 passes at
  119,074 tokens, 256 faults and 128 passes at about 72k, and 512 passes
  at 35,227 -- so the chunk belt in "Fixed" above keeps the card serving,
  while the plugin-side cause (a mixed-stage intermediate buffer sized by
  query heads is the reading; a larger one passes where a smaller faults,
  so it is not a threshold) stays open.
- `--dflash`: the exported draft head carries a state variable fixed at
  2,048 rows; the first draft after a prompt longer than that fails and the
  drafter disables itself for the process (decode continues without
  drafts). Every DFlash number on the record was taken with prompts under
  2,048 tokens.

- Drafting II measured (M11, `DESIGN.md` §7.0.2z): four host-side levers on
  the DFlash2 chain, all free of training. Viterbi over the selector's
  lattice accepts fewer drafts than the greedy commit on every probe;
  blocks of 12 and 16 gain a few percent of tokens per verify cycle and
  lose 6 to 21% of throughput; the ngram drafter falls below plain decoding
  on the dense 27B. The offline oracle puts a floor of +0.74 accepted
  drafts per cycle under any re-ranker restricted to the same candidates.
- New flags: `--dflash-block N`, `--dflash-select greedy|viterbi` (greedy
  stays the default and byte-for-byte the previous selector),
  `--dflash-lambda X`, `--dflash-topk K`; the `ARCINT_DFLASH_DUMP` cycle
  dump and `tools/dflash_oracle.py`.

## 0.2.13 — 2026-09-03

Requires `marfrit-openvino 2026.4.0~dev20260821+p2` (patches 0003–0013).

### Added
- Measured reservation (M7): the fit pass adopts the maximum admissible
  `--n-ctx` when the flag is omitted and prints the ledger it used; an
  explicit `--n-ctx` is verify-only and never lowered. On the 24 GB card
  the dense 27B agent at u8 with an 8 GiB prefix cache admits 155,568
  tokens; a unit hand-set above that is refused by design.
- `--paged-kv KEY[:VALUE]` (M8): asymmetric KV, u8 keys with i4 values, at
  8.8 KiB per token against u8's 11.3. Auto-fit context on the coder
  (16 GiB card) 133,456 → 171,312, on the dense agent (24 GB card) 155,376
  → 199,424; acceptance task 10/10 on the coder at u8:i4. Still owed:
  the u8:i4 prefill price and prefix byte-exactness.
- `--offload-ratio` device-tier expert slot pool with asynchronous uploads
  (M9, plugin patches 0004–0007): the 35B on the 16 GiB card from 0.4 t/s
  (ratio 25, unpatched) to 9.1 (ratio 50, 8 GiB pool, 16-token probe) and
  10.4 (64-token probe).
- `--moe-cpu-tier` and `--moe-cpu-tier-threads` (M14, plugin patches
  0011–0012): experts that would evict a device slot are computed on the
  host instead. 35B on the 16 GiB card: 15.0 / 15.5 t/s against 10.4 / 10.6
  at ratio 50 with an 8 GiB pool, 14.1 / 14.8 against 7.4 / 7.5 at ratio
  75 with 5 GiB; greedy text byte-identical over 64 tokens; 10/10.
- `--prefix-cache-reserve PCT`: under auto-fit, hold that share of the
  affordable KV pages spare for cached prefixes (default 0, the previous
  behaviour: all pages live). The dense agent at 25% adopts 116,528 with
  2,428 pages spare.
- `--pin-dispatch` (M12, measured null on a quiet host, opt-in) and the
  exporter's `--moe-lowering tiled` form.
- `--vision`: reserved and refused (M13). The vision IRs a VLM export ships
  are reported at load and never read (coder: 6 files, 428.3 MiB on disk).
- Per-expert routing histogram in the plugin (patch 0013, env
  `MOE_OTD_ROUTING_HIST=<path>`), the input for any expert placement or
  bit-width policy.
- `llm.txt` carries the complete `--help`.

### Fixed
- An explicit `--n-ctx` below the admissible maximum was refused whenever a
  prefix cache was configured: the cache's reserve pages were sized from
  the whole budget and the overshoot was blamed on the request. The
  reserve is trimmed first now, the request never; the refusal is itemized
  in pages.
- The auto-fit correction could not converge when the pool held spare
  pages: a sub-page overshoot moved live pages into spare and the pool
  total never changed. It trims the pool total with a 4/16/64/256-page
  floor and refuses loudly, with the attempt history, above an allocation
  granule of 84 pages.
- Two help-text defects (a literal `%%`, a paragraph under the wrong flag).

### Dependencies
- `marfrit-openvino +p2`: ten new plugin patches (0004–0013) on the same
  upstream commit; the runtime reports `...-marfrit-p2`. Patches 0001 and
  0002 remain deliberately unapplied.
- Unit files ship `--n-ctx` as a multiple of 4096 below the admissible
  number the fit pass prints: coder 131072 (u8), agent 262144 (the 35B's
  trained maximum, verified), the dense 27B with MTP 131072.

## 0.2.12 — 2026-09-01
- DFlash2 block-diffusion drafter: `--dflash DIR` serves the public
  Qwen3.8-27B draft head, seven drafts per verify pass; 44.8 t/s on the
  24 GB card against 24.0 plain and 33.0 with the MTP head;
  `--dflash-device` parks the draft on the other card with byte-identical
  output. `tools/export_dflash.py` and the pairing-probe instruments.

## 0.2.11 — 2026-09-01
- Operator serving defaults: `--temp`, `--top-p`, `--top-k`,
  `--repetition-penalty`, `--presence-penalty`, `--chat-template-kwarg`;
  precedence request > flags > artifact > family card.
- `usage.completion_tokens_details` reports accepted and rejected
  prediction tokens per response.

## 0.2.10 — 2026-09-01
- Review fixes on the paged path (index-tensor map pruning, embeddings
  byte check, MTP argmax guard, a 16-token page tripwire);
  `ARCINT_PA_HOST_INPUTS` experiment. Requires `marfrit-openvino +p1`
  (patch 0003: the MoE subbuffer churn at token_num > 1). The -3 package
  corrected the self-reported version.

## 0.2.9 — 2026-08-30
- `/props` publishes `chat_template_caps`; `reasoning_effort` accepted as
  the template's on/off switch.

## 0.2.8 — 2026-08-30
- `reasoning_content` split from `content` for templates that open the
  think block in the prompt.

## 0.2.7 — 2026-08-30
- Tool-call arguments reach the template as the type its capability flag
  asks for (the 500 on tool-using turns).

## 0.2.6 — 2026-08-30
- `--gate-pad N` (shared-expert gate widened for a JIT GEMM: −13% prefill
  wall, −5..6% decode, off by default); profiler tables name their capture.

## 0.2.5 — 2026-08-30
- The logits slice fix: +27% prefill, six device-to-host copies per
  prefill gone, 2.09 GiB of activation reservation returned.

## 0.2.4 — 2026-08-29
- The public repository becomes the upstream (no sanitised export);
  u8-vs-f16 KV corrected at depth (f16 7.8% faster at 53.5k tokens).

## 0.2.3 — 2026-08-29
- `--paged-kv u8|f16`; u8 default with its prefill price on the record;
  two retractions kept in `DESIGN.md`.

## 0.2.2 — 2026-08-29
- `--served-model-name`; `/props` carries the context length; the unit
  template installs from CMake.

## 0.2.1 — 2026-08-29
- `/v1/models` carries `n_ctx`, `n_ctx_train`, quant, lanes; the shipped
  unit starts on a package-only host. The -2 package was a rebuild of a
  stale archive.

## 0.2.0 — 2026-08-29
- First packaged release: amd64, trixie, strict dependency on
  `marfrit-openvino` at the pinned nightly.
