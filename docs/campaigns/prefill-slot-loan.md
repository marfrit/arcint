# prefill-slot-loan — lend the expert cache's slots to a prompt, give them back for decode

**Status: open (2026-10-30).** Flash-Next on the B60. Lever, not defect. The
operator adopted the measured gain (2026-10-29); round 12 settled the exit
fault's mechanism on the card, round 13 closed the review's lows and prepared the
two gates (KL and depth, commands at the end), round 14 made both gates able to
fail, round 15 makes the lent branch's red scorable (the override reaches
every table producer and every ex-slot expert of the layer), scores the served
8,192-token shape in row windows under one loan (the 4,096-row window's 4.07 GB
is more than the lend's plan), fixes the Q8_0-down loan row and the false tool
comments. **Round 16 records both gates' card results** (`measured-here`): the
KL gate passes at the bar, the red fails it by 0.007 nats (thin), the control
sits at the twin floor; the depth leg passes the speed gate at +56 % but its
peak clause fails — the loan arm peaked 42 MiB under the card's total — and the
VRAM margin is analysed from the code (design §24a/24b). The unit's flags are
prepared, and the integration is squashed onto one commit above the base the
loan branch started from, so the branch's history does not reach a public branch.

**Round 17 (2026-10-30) records the operator's decision ("Margin ok") and its
implementation; round 19 puts the margin against the card's total, round 20
replaces that card figure with the process's own measured residency from its xe
render fds' fdinfo, round 21 (the vram-user-space campaign's stage 0) replaces the
fixed margin with the kernel's device-wide state, adds the eviction guard, and
removes the per-swap lend mirror. Round 22 repairs round 21's two HIGH design
defects found in review: the kernel `used` figure already holds every other
client and the kernel's own objects, so subtracting `used - fdinfo` counted them
twice (`code`); the rule is now `used - lend + need + first_touch + allowance <=
total` with the operator's zero margin, one optional prefill-growth allowance
(default 0), and `used - fdinfo` logged as information. The eviction guard now
judges the end against a per-process GTT baseline taken at the begin (minus the
host banks and any grown bank), acts on the VRAM shortfall alone, logs the GTT
growth, clears the shed after a clean loan, and has a knob
(`LLAMA_SLOT_LOAN_EVICTION_GUARD=0`); the end's expected VRAM now includes the
loan's compute reserve and the non-sum residency measured at the begin; the load
probe runs only for a cache with a lend. R84-R93 red (`code`, each seen failing
device-free). Round 23 gives the end guard a noise floor and turns a true
shortfall into the NEXT lend's growth instead of a shed or a refusal, treats a
kernel `used` of 0 as no figure, and makes the query's `why` an owned string
(`code`, R94-R98 red; the sound `shed_lend` is gone). Round 24 fixes the fdinfo
discovery the round-22 card window caught (`measured-here`: the runtime opens two
xe clients per card in one process, so round 20's unique-pdev match never
engaged; neither the query, the fdinfo figure nor the guard ran): a card's figure
is the SUM over the distinct client ids whose drm-pdev is the device's, and the
ioctl takes any fd of that pdev (`code`, R99-R102 red).** Round 25 makes the
carried shortfall's growth two-step: first the demand (`need + shortfall`), then,
when the tail cannot reach it, the tail's full growth when it still meets the
loan's own `need` -- so a short tail never refuses a loan the `need` rule allows
(R103 red), a claim round 23's wording made before the code did (`code`). The rule
keeps the 8,192 chunk and closes gate 5 as **accepted with a fixed margin**;
the lending patch grows the lend layer by layer until the margins hold and
refuses into the resident path when the tail cannot reach them (`code`, design
§24b/§24c, §26; R58-R60, R65-R67 red). A 27th layer (~+238 MiB) was round 17's
stated way; round 19's arithmetic says the 6,400 ask grows nothing (design
§26b), and the window decides. The owed window is
the ~100k-token prompt with the six answers, the sampler's 2 s peak `vram0`/GTT,
and the lend line's layers, slots and margin (§ round 17, "The owed window").

## Current state (`measured-here`, 2026-10-29, the B60)

The unit's binary, Flash-Next IQ3_S, the unit's flags plus `--llama-slot-loan
6400 --llama-loan-chunk 8192 --prefill-chunk 8192`, an 18,802-token prompt, two
rounds, loan off and on in turn:

- prefill **485.2 / 485.0 t/s with the loan against 436.9 / 436.9 without
  (+11.1 %)**; decode on the six answers 34.62 / 34.58 against 34.67 / 34.65
  (within the run-to-run spread); the answers byte-identical across the unit's
  binary, loan off and loan on.
- the lend: 26 layers (il 22-47), 3,094 slots, 6.20 GiB freed, bank in host
  memory; table check 0 of 192 rows; the return's check 0 of 9,360 rows, also
  after adapt swaps; peak 24,242 MiB (the unit's 23,519; the card 24,480).
- the refill: **2.66-2.86 s for 6.20 GiB (~2.3 GB/s)** against the card's link
  14.3 GB/s (measured with USM host memory) — the cost round 7 removes (§15).
- red switches: 1 and 4 exit 70 with the cause; 2 undoes the lend;
  `LLAMA_SLOT_LOAN_MUTATE=3` did not fire (it removed a comparison the healthy run
  never entered) — fixed in round 7 (§15b), again in round 10 (§18).

## The lever

A prompt and decode want the same VRAM for different things. Decode wants expert slots (the share of routed
experts served from the card); a prompt wants a large ubatch, whose compute buffer arcint reserves for the
whole session at load (`src/exec/backend_llama.cpp:577-588`: with the expert cache the ubatch is the prefill
chunk). Today the split is static, so a larger chunk costs decode its slots for good.

Measured (`measured-here`, 2026-10-29, `research-concept-mining.md`, Wave 1 window 2: the B60, Flash-Next
IQ3_S with the unit's flags, a 19,196-token prompt, one fresh process an arm):

- chunk 2,048 at the unit's 11,600 MiB of slots: prefill 442.1 t/s, peak 23,519 MiB;
- chunk 8,192 at 8,000 MiB: **569.9 / 572.1 t/s (+29 %)**, peak 23,632 MiB; chunk 4,096 at 9,600 MiB: 532.9
  (+20.5 %); the same chunks with more slots sit at the card's edge (24,400-24,437 MiB of 24,480) and scatter
  or lose (393-499 t/s, 364.1 t/s);
- the static price: decode on the six answers at 8,000 MiB of slots 31.08 t/s against 34.30 at 11,600
  (-9.4 %; routed experts on the card 71.5 against 81.2 %).
- why the larger chunk pays: the grouped GEMM's 32-token tiles hold +396.5 % (gate/up, bank experts) and
  +192.5 % (slot experts) more rows than were routed at 2,048, +115.8 % and +48.6 % at 8,192.

The prize is the larger chunk's prefill with the unit's decode: borrow slots for the prompt, refill them
before decode.

## The reference

Strata's prompt path borrows the tail slots of the GPU expert cache, a "lend region", for its buffers, and
refills them after the prompt (`code`, `~/src/Strata-ref/src/core/expert_source.cpp:247` (the lend region's
start), `:3571` (experts taken from the last slot down, the order the borrowing uses), `:3649-3655` (every
part's cache lends its tail; the refill copies them back)). History (`paper`): 68072d30 "refill the prompt
path's lent cache slots with queued copies and one wait"; d5733321 the refill from the GGUF mapping's pieces;
2838cbae "never write into the borrowable cache slots while laying out (a resident expert was corrupted)";
219ae436 Battlemage's borrowing configuration with a 8,192 prefill (B70 16K 1,222 -> 1,347 t/s, 32K 1,187 ->
1,361). Strata picks the largest chunk the lent bytes allow (`--prefill auto`).

## arcint today (`code`)

- The slots: one buffer a cache, `bslots` (`llama-src/src/llama-expert-cache.cpp:471`), three tensors a layer;
  the bank (host) beside it (`:473`). The cache's `verify` compares slot bytes with the file
  (`:132-140`).
- The compute buffer: reserved once for the prefill chunk and the reserve cells (patch 0052,
  `--llama-reserve-cells`); decode needs ~93-770 MiB of it (`paper`, the Flash-Next record, stage 3c).
- MTP's draft context reserves its own buffers beside it.

## Design constraints

- **Exact:** a lent slot is never read as an expert while lent (the cache treats it as empty: misses go to
  the CPU rows or the bank stream), and after the refill every lent slot's bytes equal the file's. The
  hazard Strata recorded (2838cbae) is the red case.
- **Whole graphs:** the loan is taken before a prompt's first ubatch and returned after its last; the cache's
  adapt is paused while the loan is out, and decode starts only after the refill's wait.
- **No eviction thrash:** the prefill graph's peak plus the remaining slots stay under the unit's measured
  peak (23,519 MiB); the record shows what the edge costs.
- **One lane** (Flash-Next serves one), so no second request can see a lent slot.

## Gate (on the record before the work)

Against the unit's configuration (chunk 2,048, 11,600 MiB), same binary family, same window, alternated fresh
processes, `tools/fn_sweep.py`:

1. prefill at the 19,196-token prompt **>= +20 %**;
2. decode on the six answers within the unit arm's run-to-run spread (window 1: 0.0-1.7 %);
3. the answers: the capital and the 20k needle right; the answer-level bar (mean KL at most 0.03 nats worse
   than the baseline arm against the reference, argmax agreement at most 1 point lower); the chunk change is
   rounding-only;
4. the slots' bytes equal the file's after every refill (the cache's verify, exact);
5. peak VRAM at or under the unit arm's; the refill's wall time reported (estimate ~0.25 s for ~3.6 GiB at
   the B60's 14.3 GB/s, inferred). **Closed 2026-10-30 as accepted with a fixed margin**: the depth arm's
   peak is 42 MiB under the card's total (`measured-here`, round 16), and the operator accepted the loan's
   transient peak under `lend - need >= 384 MiB + 1 GiB`, reached by lending a tail layer more (§24b). The
   clause is a fixed-margin rule, not the unit arm's peak number.

## Stages

0. Design note: where the prefill graph's compute buffer lives during a loan (a buffer over the lent slot
   range, or a second reservation the scheduler switches to), and how the scheduler's reserve stays valid for
   decode. Size the loan from the measured peaks.
1. The loan in the cache: lend the tail set (the separate tail buffer needs no contiguity), mark it empty,
   pause adapt, refill with queued copies and one wait, verify. Red cases: a graph that writes into a lent range is refused (mutation: the bound
   check removed); a lent slot not refilled fails the verify (mutation: one slot skipped).
2. The prefill chunk chosen from the lent bytes (Strata's `--prefill auto`), with the unit's chunk as the
   floor.
3. The served A/B against the gate.

## Implementation

The design is revised after the stage-0 review in
`docs/campaigns/prefill-slot-loan-design.md` (evidence class on every
disposition; findings 1-8 checked in the code). Stage 1 is patch 0068 of the
llama.cpp series, which is built in this tree, with the arcint hooks and the
device-free rules in `src/exec/slot_loan.h` (`tests/test_slot_loan.cpp` marks
its red cases); the card runs of stage 1 and the A/B are Claude's.

DeepSeek implements (patch 0068+ and `src/exec/backend_llama.cpp`), Claude runs the card windows and the A/B.

**The first card run (2026-10-29, the B60, the served IQ3_S) found the loan
does not lend on this model and crashes at load; fixed, no card re-run yet.**
Every layer's `ffn_down_exps` is IQ4_NL and `no_split()` excluded it, so no
layer was lendable; the selection then left the non-chosen eligible layers
marked and created tensors in a null tail context (`ggml_new_tensor_impl` <-
`ggml_new_tensor_3d` <- `llama_expert_cache::init`). The loan now lends
IQ4_NL through `get_slices`/`set_slices`, computes `lendable` from scratch per
layer over a non-contiguous set, logs `the slot loan lends nothing: <reason>`
and runs off when nothing is chosen, and the operator decided the poison policy
(2026-10-29): one ERROR line and a non-zero exit, so the unit restarts clean.
The design and its red cases are `prefill-slot-loan-design.md` §13; Claude's
re-run commands are below.

**Stage 1's fifth review round (2026-10-29, before the card re-run): one
blocker and one small fix, no card run yet.** Both are in
`prefill-slot-loan-design.md` §14 with the code lines that prove each and the
test that sees it. The blocker: the refill wrote IQ4_NL slot rows through
`set_tensor_slices`, whose card SOA path casts `tensor->extra` to
`ggml_tensor_extra_cl_iq4_nl` -- but the refill's re-allocated tail buffer
installed the plain extra (only a whole-tensor `set_tensor` write builds the
d/q sub-buffers, as init does at load), so a CL_CHECK abort or silent
corruption on every lent layer of the served model (its down is IQ4_NL), and
the same cast sat in the verify, the adapt copies and the kq weight args. The
refill now assembles the full `(n_slots+1) x nb2` image per layer and writes it
whole with `ggml_backend_tensor_set`, rebuilding the extra; the type rule
(`refill_whole_write`) covers the whole set_tensor-built set, the planar types
keep the slice path. The small fix: `poison_exit` used `std::exit`, so static
destructors (the OpenCL device contexts, the CPU-miss coordinator) ran on the
generation thread under the mutex and could hang; it is `std::_Exit(70)`, with
the line emitted through the llama log callback first and on stderr after.
Reported: one mirror swap costs ~2.05 MB read + written (~0.5-1 ms on the
decode thread, ~35-70 ms a 96-swap adapt) for the card A/B to confirm. Device-
free, each red case was seen failing: `the_refill_writes_a_rebuilt_iq4_nl_slot_
whole`, `the_whole_write_set_names_the_set_tensor_types`, and the extended
`the_poison_policy_exits_non_zero_with_the_cause` (a `LOG:ERROR` capture and an
`atexit` marker).

**Stage 1's third review round (2026-10-29, before the first card run): fixed,
no card run yet.** Eight more defects, each checked in the code first, are in
`prefill-slot-loan-design.md` §11 with their evidence class and the test that
sees each: a planned, unissued adapt phase is now issued and settled instead of
cleared (an aborted ubatch left it so; the spares leaked and the refill compared
the orphaned slot against -1); the settle fences before the phase it issues and
between the phase-1 tables and the phase-2 copies; the lend bank's mirror is
once a swap, not once a k (0.14 GiB a 96-swap adapt, not 0.42); a failed
resident re-reserve and a lend whose undo still differs poison the context and
the cache, so the request-start check refuses; a cache with no lend set is
skipped instead of refusing the loan under a layer split; the refill writes and
checks the zero slot too; `lend_undo` re-checks the tables; and the two
device-free cases that could not fail were rebuilt (production's call shape and
a source check on the cache's init; the flag rule on `init` and
`llama_model_default_params`). The rule cases are `tests/test_loan_book.cpp`
against patch 0068's `src/llama-loan-book.h`; each was seen failing under its
one-line mutation. The card proof for findings 1-7 is the trace at §11.

**Stage 1's second review round (2026-10-29, after the qfndev rebase): fixed, no
card run yet.** Eight defects, each checked in the code first, are in
`prefill-slot-loan-design.md` §10 with their evidence class and the test that
sees each: the lent branch had to precede the <= 8-token decode branch (a reserve
over NULL slot tensors and a 4-token last batch); the lend bank is now the slots'
mirror at both swap sites; the refill reads back through `get_slices` (planes);
a failed return cannot throw from the destructor; the cell budget is
`ub_cap x n_kv_reserve`; a failed lend puts its tables back; the tail buffer has
its own owner; the request-start check is under the mutex and the tool's flag wins
over `LLAMA_SLOT_LOAN_MIB`. Device-free red cases: `test_slot_loan.cpp`
(finding 4, the flag rule) and `test_loan_book.cpp` (findings 6 and 7, against
patch 0068's own `src/llama-loan-book.h`). The card proof for findings 1, 2, 3, 5
is the trace at §10/§11: a loan's `refill ... verify 0 of N rows differ`, the
return's `refill+verify ok`, and no `reallocating` line on a deep follow-up.

**Round 7 (2026-10-29, before the card run): the refill's speed, red switch 3,
the fifth review's nits, the peak's margin.** The design and its red cases are
`prefill-slot-loan-design.md` §15. The refill no longer stages and converts on the
host: the lend bank holds the card's layout, so one non-blocking region copy a
tensor from it (three new ggml-opencl entry points), no whole-image write, no
temporary device buffer, one wait at the end instead of one a layer. The check is
exact over every row either way: by default against the lend bank in the card's
layout (no conversion, no GGUF read a loan), whose provenance to the file is the
load's new lend bank check over every lendable row and every adapt's mirror;
`LLAMA_SLOT_LOAN_VERIFY=file` keeps the direct file form for the gate arms. The
refill's trace line now prints its phases (`stage, write, convert, tables,
wait`) and the check's time, so the card window shows where the time goes.
`LLAMA_SLOT_LOAN_MUTATE=3` now removes the lent graph's measurement — the thing
the size check depends on — instead of a comparison the measured run never
entered; the reserve then fails on the card and the slots are put back. The size
check keeps a 384 MiB peak reserve.

**Round 8 (2026-10-29, before the next card run): the seventh review, two card
findings.** The design and its red cases are `prefill-slot-loan-design.md` §16.
The card's round-7 numbers (`measured-here`, marker `mutate1-driver`): the refill
0.514-0.541 s for 6.20 GiB (was 2.7 s); the load's bank check 0 of 9282 rows in
1.5 s; prefill `=file` 498.4 t/s (verify 3.295 s), `=0` 546.4/546.6, loan off
436.6 -- the verify's cost decides the gain, so the default check was made exact
and cheap. The default bank check's per-row form read the slot's q region one
extra row ahead of the bank's (regions come one after the other, not
interleaved): ~8/9 rows "differed" on a correct copy and exited 70. It now reads
the same `n` on both sides and compares whole buffers, the zero slot separately
against zeros. `MUTATE=3` is deterministic: after the reserve, the actual buffer
sum is compared with the sizing's expectation (the switch perturbs the expected
bytes by skipping the lent measurement), and the mismatch puts the slots back.
Lendable is now only what the region copy is exact for (plain-no-extra, planar,
IQ4_NL): the bank stores the specialised types raw while the card builds a
split/extra or converts BF16, so a copy there is the old crash or wrong weights
with a passing bank check; `region_copyable` names the set, per type tested. The
384 MiB margin moved to the device's free memory after the lend's free and before
the prompt's reserve, logged there. `MUTATE=1` now corrupts one data byte of a
lent row after the copies (the round-7 skip left an unbuilt split extra and the
`=file` get_slices cast a plain extra: an Intel driver segfault). The poison exit
drains the compute, adapt and bank-stream copy queues first, bounded by a 5 s
watchdog, so `_Exit` does not tear the address space down under in-flight work (the
`xe` page fault and GuC engine reset on both exit-70 arms).

**Round 9 (2026-10-29, before the next card run): the eighth review's four
findings.** The design and its red cases are `prefill-slot-loan-design.md` §17.
The 384 MiB headroom check could not fail: `ggml_backend_opencl_device_get_memory`
returns `free = total - 1 GiB` (a constant), so round 8's `headroom_fits` only
tested the card's total. The loan is back on round 7's real rule,
`need + 384 MiB <= lend`; no device free figure is consulted. `reserve_matches`
is `actual <= expect`, not equality (a shared buft counts differently in the
sizing and the buffer sum; only actual above the expectation is the defect).
The bank check now compares row by row inside each region, so `bad` is a row
count like `n` and the first mismatch names layer, region and row (the old
`0 of 9360 rows` becomes `0 of 12480` on the served model). The watchdog's
`std::thread` construction is guarded (`watchdog_spawn`): a throw there would
have reached `std::terminate` and an exit that is not 70. Device-free, each red
case was seen failing: `a_size_check_leaves_a_peak_margin`,
`the_lent_sizing_is_measured_unless_mutated`,
`a_region_order_read_compares_row_by_row`,
`a_watchdog_construction_failure_still_exits`.

**Round 10 (2026-10-29, before the next card run): the two card findings from
round 8.** The design and its red cases are `prefill-slot-loan-design.md` §18.
(1) `MUTATE=4` exited 70 with the right line but the B60 faulted at that exit
(`xe: Fault response -ENOENT`, a GuC reset of the compute and copy engines,
timed to the second): the tail buffer was freed at the lend while the adapt and
bank-stream queues could still hold copies over it, and the exit's drain ran
them against released memory. Every tail-buffer free now drains first
(`drain_now` in `drop_tail_buffer`), and the return's poison path drains before
poisoning. (2) `MUTATE=3` printed the red line but the loan then lent, refilled
and returned normally: round 8's mutation skipped the lent flag on the premise
of one more MUL_MAT_ID dst, but the lent branch replaces the slot branch and
drops both down tables (`llama-graph.cpp:2510-2521`), so the skip RAISES `need`
and `actual <= expect` passes; `need` is the compared value and the comparison
is reached -- the mutation simply did not under-count. `MUTATE=3` now sizes the
graph at the resident chunk while the reserve allocates the prompt's chunk
(`measure_tokens`), a real under-count the rule rejects. Device-free, both red
cases were seen failing: `a_poison_path_drains_before_any_free` and
`the_lent_sizing_is_measured_unless_mutated`.

## The re-run's card commands (Claude's window)

Same binary family as the measured run, one fresh process an arm. The measured
configuration, the loan on and off in turn: `--llama-slot-loan 6400
--llama-loan-chunk 8192 --prefill-chunk 8192` (26 layers, 3,094 slots, 6.20 GiB,
the 18,802-token prompt). `LLAMA_SLOT_LOAN_VERIFY` unset (the default check
against the bank) for the timing arms, `=file` for the gate arm, `=0` for the
refill's floor. The proof lines are the patch's own trace:

    arcint --engine llama --gguf <flash-next-iq3_s-00001-of-00002>.gguf --device GPU.0 \
      --n-ctx 131072 --llama-cpu-moe 48 --llama-threads 8 --llama-expert-cache 11600 \
      --llama-reserve-cells 20480 --llama-expert-profile <expert-profile>.bin \
      --prefill-chunk 8192 --llama-slot-loan 6400 --llama-loan-chunk 8192 \
      --llama-kv q8_0 --queue-timeout 30

    llama_expert_cache::init: the slot loan lends nothing: ...            (must NOT appear)
    llama_expert_cache::init: lend bank check, ...: 0 of N rows differ from the file (T s)
    llama_expert_cache::lend, ...: N layers (il 22-47), S slots, X GiB freed ... table check: 0 of R rows differ
    llama_context::slot_loan_begin: loan out at C tokens: N layers, S slots, X GiB freed, prompt buffers Y GiB; size+lend+reserve took Z ms
    llama_expert_cache::refill: 3094 slots, 6.20 GiB in <T> s (stage <a>, write <b> from 78 tensors, convert 0.000, tables <c>, wait <d>), verify on <e> s: 0 of 12480 rows differ from the lend bank
    llama_context::slot_loan_end: loan returned: N layers, S slots, X GiB; refill+verify ok

What each line proves:

- **item 1 (the refill's speed).** `write <b>` + `wait <d>` is the number this
  round changes: expected 0.45-0.55 s for 6.20 GiB at the card's 14.3 GB/s with
  USM host memory; `convert 0.000` shows no layout conversion is left (a band
  near 2.7 s means the driver stages the USM writes through the CPU, and the
  kernel path named in design §15a is the next lever). `verify on <e>` is the
  check's own cost (~0.9-1.1 s expected: one read-back plus the memcmp);
  `=0` gives the floor. The prefill and decode numbers are the same A/B as the
  measured run (485.2 t/s against 436.9, decode 34.6 both, answers identical).
- **item 1's exactness.** `0 of 12480 rows differ from the lend bank` over every
  row (round 9: same `n` on both sides, row by row inside each region with its own
  stride, the zero slot in the slot image's extra row; a mismatch appends
  `(first: layer N region R (name) row S)`), and the load's `lend bank check ... 0 of N rows differ from the file` over every
  lendable row, once; `=file` is the gate's direct form (its line says
  `from the file`).
- **item 3.** `MUTATE=3` prints, in this order,
  `LLAMA_SLOT_LOAN_MUTATE=3: the prompt's graph is sized at the resident chunk (2048), not the prompt's (8192) (the red case)`,
  `the prompt's reserve allocated X GiB against the sized expectation Y GiB
  (LLAMA_SLOT_LOAN_MUTATE=3 sizes at the resident chunk); putting the slots back`,
  the refill's put-back with `0 of ... rows differ from the lend bank`,
  `refill+verify ok`, and the answer at the resident chunk. Round 10 (§18): round
  8's mutation skipped only the lent flag, which raises `need` (the lent branch
  replaces the slot branch and drops both down tables), so the check passed and
  the loan ran as healthy (`measured-here`); it now sizes at the resident chunk,
  a real under-count that `actual <= expect` rejects.
- **item 4.** The fifth-review half is the same `0 of 12480 rows differ from the
  lend bank` on IQ4_NL layers: `lend_write` rebuilt the d/q split extra the
  re-allocation lost. The planar list and BF16 are source-side (design §15d).
- **item 5 / round 9's item 1.** `the lend frees X GiB, the prompt's buffers need
  Y GiB, the peak reserve is 0.375 GiB` after the lend's free; the plan needs
  `need + 0.375 GiB <= lend` (`size_fits`, round 7's rule over real bytes). No
  device free figure is read. A refusal prints need, the lend and the reserve;
  the margin is then the operator's size decision.
- **red switch 1** still fires: with `LLAMA_SLOT_LOAN_MUTATE=1` the refill prints
  `LLAMA_SLOT_LOAN_MUTATE=1: one data byte of the last lent layer's down row 0
  corrupted after the copies (the red case)`, then exits non-zero (70, no static
  destructor) with `the refill did not verify: 1 of 12480 rows differ from the lend
  bank (first: layer N region R (quants) row 0)` (or the file) and one
  `ERROR llama:` line through the log callback -- and no Intel driver segfault.
- **item 7 (drain-before-exit).** On both exit-70 arms (`MUTATE=1`, `MUTATE=4`)
  the kernel log shows no `xe` fault and no GuC reset of the compute or copy
  engine at the exit second; the drain finishes the queues first, bounded by 5 s.
  Round 10 (§18): `MUTATE=4` faulted anyway -- the tail buffer was freed at the
  lend while adapt/bank-stream copies could still reference it, and the exit's
  drain ran them against released memory. Every tail-buffer free now drains
  before it (`drain_now`), and the return's poison path drains before poisoning.
  Round 12 retracts the queued-work explanation of the fault that survived
  (§19's note, §20): the review found no adapt-queue command over a galloc
  buffer and OpenCL's `cl_mem` retention bars a release fault, and the card's
  discriminator is idle time (round 11: `MUTATE=1` hundreds of ms, no fault;
  `MUTATE=4` microseconds, fault 3 of 3; direct submission off: no fault, 0 of
  3). The proof line is now at WARN, and a `MUTATE=4` log carries two drain
  lines with the `ERROR llama:` line between them.

**Round 11 (2026-10-29, before the next card run): the MUTATE=4 exit's remaining
fault, the sizing's throw, the drain's scope. RETRACTED in round 12: the
fault's explanation was not in the code (§20).** The design and its red cases are
`prefill-slot-loan-design.md` §19. The round-10 card run (b4bca38) still faulted
`MUTATE=4` at its exit 70 (`xe: Fault response -ENOENT/-EINVAL`, a GuC reset of
the compute and copy engines, timed to the exit second). The trace is `code`:
every free and every scheduler replace now follows a full drain (compute, adapt,
bank-stream), not only the backend's `synchronize` that leaves the adapt queue
out: `slot_loan_reserve` drains before `sched_reserve`'s `sched_new` and again on
a throw, `slot_loan_begin` drains before its own `sched_new`, and a failed/skipped
return synchronizes the scheduler and drains before the poison. A throw inside
`slot_loan_begin`'s sizing now takes the resident re-reserve and the rebind instead
of reaching `Scope::begin` with an unreserved scheduler and a dangling
`owner_sched`; the cache destructor and `drop_tail_buffer` drain whenever
`lend_bytes_ != 0` (before freeing anything; unchanged with the loan off), and a
failed tail allocation clears its dangling tensors before the poison. The drain
prints `ggml_opencl: drain: N device(s), compute, adapt and bank-stream queues
finished`, the card's proof line. Device-free, each red case was seen failing:
`a_re_reserve_finishes_the_queues_before_sched_new`,
`a_lent_sizing_throw_takes_the_resident_refusal`,
`a_lent_cache_drains_before_every_free_in_the_destructor`, and
`a_failed_tail_allocation_leaves_no_dangling_tensor` (R36-R39).

The card proof (round 12's form): `MUTATE=4` exits 70 with two drain lines and
the `ERROR llama:` line between them, then the exit's wait line, and
`dmesg -T | grep -E "xe|GuC"` shows no fault and no reset at the exit second. Round 11's alternative for a surviving fault -- the driver's teardown of
the live USM lend bank -- is retracted too: round 12's card discriminator names
the direct-submission rings (`measured-here`: the same arm with direct
submission off has no fault, 0 of 3), and the exit's bounded idle is the fix
(§20).

**Round 12 (2026-10-29, before the next card run): the exit's live rings, the
retraction, the drain line's level.** The design and its red cases are
`prefill-slot-loan-design.md` §20. Round 11's explanation -- a free or a
scheduler replace under queued work -- is retracted on the record, checked in
the code first (`code`): no adapt-queue command targets a galloc buffer, the
router-ids read is on the compute queue, and OpenCL retains a `cl_mem` until its
commands finish. The drains stay (harmless, loan-path only). The card separated
the mechanism instead (`measured-here`, the B60, round 11): `MUTATE=1` exits
hundreds of ms after the last GPU work (the refill verify's read-back and
memcmp) and does not fault; `MUTATE=4` exits microseconds after it and faults
3 of 3; the same arm with direct submission off exits 70 without a fault, 0 of
3. `_Exit` runs no destructor, so the runtime's direct-submission rings are
still live when `xe` closes the fd. The fix: `poison_exit` waits a bounded idle
after its drain and before `_Exit`, `LLAMA_SLOT_LOAN_EXIT_IDLE_MS` (default
2,000 ms, clamped to 4,000 so the 5 s watchdog keeps 1 s for the drain; the
default is twice `MUTATE=1`'s measured idle upper bound, since the runtime's own
idle period is not readable here). Direct submission is not switched off
globally: it is every unit's fast submit path. The cleaner runtime call
(`DirectSubmissionController::stopRingBuffers()`) is internal C++, not an
OpenCL entry point, and the source is not checked out here, so the idle is what
the public surface allows. The drain line is WARN now (INFO never reached the
card; `measured-here`, round 11's window had no line). The review's lows:
R36 checks the order (resident reserve before the rebind in the `!sized` block);
the `MUTATE=4` log form is two drain lines with the `ERROR llama:` line between
them; R36-R39 are marked text-order tests in §19. Device-free, each red case was
seen failing under its one-line mutation: `the_exit_waits_for_the_rings_to_stop`
(R40/R42), `the_exit_idle_knob_is_bounded` (R41),
`the_drain_line_is_at_warn` (R43), and R36's order half.

**Round 12's card results (`measured-here`, 2026-10-29, Claude's window 11, the
B60, Flash-Next IQ3_S, the unit's flags, d6fb3d4).** The idle discriminator is
settled on the card:

| arm | exit | log | kernel log | count |
|---|---|---|---|---|
| `MUTATE=4`, idle 2000 (default) | 70 | drain / `ERROR llama:` / drain / `the exit waits 2000 ms` (the documented order) | no `xe` line in the diff | 3 of 3 |
| `MUTATE=4`, `LLAMA_SLOT_LOAN_EXIT_IDLE_MS=0` | 70 | as above | `xe` fault at the exit second (bcs engine reset, -ENOENT) | 1 of 1 |
| `MUTATE=4`, idle 100 | 70 | as above | no fault | 1 of 1 |

So the rings are live at `_Exit` for microseconds of idle and stopped by 100
ms; the 2,000 ms default is a margin over a measured passing value, not the
value. `NEOReadDebugKeys=1 PrintDebugSettings=1` printed nothing into the
server log: the runtime's own controller period is not measured here (`code`:
the runtime is not checked out). The mechanism is settled (idle 0 faults, 100
and 2,000 do not).

**Round 12's loan arm (`measured-here`, same window).** Loan on, the measured
configuration: **515.5 t/s at the 18,802-token prompt** against round 11's off
arm 436.7 (**+18.0 %**); rounds 11, 8-10 are 521.7 and 518.8-521.3, inside the
run-to-run band. Refill **0.514 s**, verify 1.607 s, `0 of 12,250 rows differ`,
peak VRAM 24,229 MiB and GTT 43,445 MiB; a loaned request logs 6-8 drain lines
(that is WARN's drain per queue phase, not a fault). An off arm with
`--prefill-chunk 8192` and no loan ran **275.2 t/s** at peak 24,452 / GTT
49,731 MiB — the static chunk-8192 spill the loan exists to avoid (Claude's
trap arm, not a regression; recorded as such).

**Round 13 (2026-10-30, before the unit): the review's lows, the two gates.**
The code is `prefill-slot-loan-design.md` §21. `poison_exit` now prints a second
line after the idle with the measured elapsed ms; the watchdog, when it fires,
writes **one** line with `write(2)` (async-signal-safe: no `snprintf`, no log
callback, no flush, the hung drain may hold the log lock); and the knob reads
`off`/`OFF`/`false`/`False`/`no`/`NO` as 0 (round 12's `strcmp("off")` silently
took the default for an `OFF` arm). The docs say what was wrong: nothing bounds
the drain but the 5 s watchdog (the old "leaves 3 s for the drain" read as a
drain bound); the drain line is invisible at the unit's default verbosity
(`INFO` needs `-v`, the line is `WARN`); abort paths (`GGML_ASSERT`, `CL_CHECK`)
bypass the idle and would fault the same way — no change, a sentence. Device-free,
each new red case was seen failing under its one-line mutation: R44 the idle's
second line (emission removed), R45 the off words (the case-fold removed), R46
the watchdog's line (the `write` removed), R47 the KL driver's threshold (`>` to
`>=`), R48 the driver's begin/end order (the begin moved after the decode).

**The two gates (`code`, commands below).**

- **KL, answer-level bar.** `llama-perplexity` built from the served series does
  **not** take the loan: `LLAMA_SLOT_LOAN_MIB` arms the load (which layers and
  slots are lendable) but no tool calls `llama_context_slot_loan_begin/end` —
  the only caller is arcint's `src/exec/backend_llama.cpp`, and a perplexity
  run's "loan on" arm would be identical to its off arm. Arcint's endpoint has
  no logprobs (`src/api/handlers.cpp`: `"logprobs", nullptr`), so the served side
  cannot take it either. The route is `contrib/llama-loan-kld`: it reads the
  loan-off capture for the exact tokens and decodes them with the loan; the
  third `MUL_MAT_ID` branch is exercised by a chunk above 2,048 tokens (the
  lend threshold; a 512-token window would measure nothing).
- **Depth.** The loan was measured at 18,802 tokens; the unit runs 131,072. The
  KV is n_ctx-sized at load and does not follow the prompt; the compute buffers
  hold `ubatch x n_kv_reserve` cells and the reserve holds `ub_cap x
  n_kv_reserve`. The cell cut applies to **both** arms and scales with `ub_cap`
  (`code`, `llama-context.cpp:2429-2454`): the off arm's `ub_cap` is 2,048, the
  on arm's is 8,192, so at depth the off arm runs 419 tokens a ubatch at ~100k
  (320 at 131,072) and the on arm 1,676 (1,280) -- 4:1, not "the on arm at the
  off arm's rate". The measured 18,802-token arm used the full 8,192, which is
  why its +18.0 % is a full-chunk gain; at depth the gain's size is the open
  cell (round 14 rewrote this prediction; design §22a has the table).
**Round 14 (2026-10-30, before the unit): the two gates made able to fail.** The
code is `prefill-slot-loan-design.md` §22. Five fixes and a build gap, all
device-free here; the GPU work stays Claude's.

- **The depth prediction was backwards** (`code`, `llama-context.cpp:2429-2454`):
  the cell cut applies to both arms and scales with `ub_cap`, so the off arm runs
  419 tokens a ubatch at ~100k and the on arm 1,676 (320 / 1,280 at 131,072); the
  on arm stays faster at depth, and the gain's size is the open cell. Rewritten
  here and in design §22a, with the table.
- **The KL driver's host memory** (`code`): round 13 held 8.14 GB of logits copy,
  2.03 GB of row buffer and 8.14 GB of output reservation at Flash-Next's 248,320
  vocab. Rows now stream from `llama_get_logits(ctx)` before the loan's end, one
  row (0.47 MiB) is held, `n_outputs_max` is `n_ctx - n_ctx/2` (4,096 rows). The
  forecast for both arms is in the KL gate section.
- **A red case for the lent branch** (`code`): `LLAMA_SLOT_LOAN_MUTATE=5` moves one
  lent layer's lend-bank row index in the card's table during the prefill; no
  verify compares a table row, so the run refills, returns and scores -- and it
  runs first, before the control and the on arm, and must fail the bar. The WARN
  line names layer, expert and both rows, so the log proves the mutation was in
  the artefact the prefill used. R49. **Round 15 corrects this (`measured-here`:
  the review's run): the one host row was not in `check_tables`' producer, so the
  lend was undone and the driver exited 2 un-lent; the override now lives in
  `fill_tables`, every ex-slot expert of the layer rotates, and the log names the
  layer and the count.**
- **The served shape** (`code`): `-ub 2048` ran four ubatches of 2,048; the driver
  now sets `llama_context_params.n_ubatch_reserve` (common has no option) and the
  on arm runs `-ub 8192 --ubatch-reserve 2048`, one 8,192 ubatch under the
  resident 2,048 and the loan's 8,192 reserve. The off arm keeps perplexity's
  split. R53.
- **The control, the quantiser, the partial file, the cold chunk** (`code`): the
  driver without `--loan` is the control and must sit at the twin's floor; the
  capture's quantiser is perplexity's `nearest_int` (half-to-even, not
  `std::lround`); a partial `--out` is unlinked on an exit-2 refusal; the driver
  logs that its first chunk's rate is cold (no warmup decode). R50, R52.
- **The two tools on the card host** (`code`): the recipe builds arcint's deb with
  `LLAMA_BUILD_TOOLS OFF`, so `contrib/llama-loan-kld/build-tools.sh <recipe's
  llama tree>` builds `llama-perplexity` and `llama-loan-kld` from the same
  patched series, with `-DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
  -DGGML_OPENCL_EMBED_KERNELS=ON`, and prints both `TOOL ... sha256`.
- Device-free, each new red case was seen failing under its one-line mutation:
  the suite runs 61 cases green, and R49's rule and its `lend()` call site, R50
  (`std::lround`), R51 (the `n_outputs_max` assignment), R52 (the partial file's
  `std::remove`) and R53 (`served_shape`) each fail exactly one case when
  mutated.

## Round 15 (2026-10-30, before the unit): the red that scores, the served shape's row windows

A review of round 14's code. The design is `prefill-slot-loan-design.md` §23;
all fixes are device-free and each red case was seen failing before it.

- **The red** (`code`, R54): `MUTATE=5` shifted one host table row outside
  `fill_tables`, so `check_tables` rebuilt `want` from `loan_of`, the lend was
  undone and the driver exited 2 before scoring -- no red reached the bar. The
  override is now a layer member consulted in `fill_tables` while lent: the
  upload, the check, the card and the refill's restore agree, and `lend()`
  rotates EVERY ex-slot expert's lend-bank row in that layer. The WARN line names
  the layer and the count. `refill()` clears it before `slot_of` is rebuilt.
- **The served shape's rows** (`code` + arithmetic, R56): the loan's sizing takes
  `n_outputs = min(measure_ub, n_outputs_max)`, so the round-14 driver's 4,096
  rows put ~4.07 GB in the lent graph's compute buffer -- more than the 6.20 GiB
  lend's `need + 0.375 GiB <= lend` -- and the same rows are the context's
  output buffer, a plain CPU buffer on OpenCL, adding ~4.07 GB of anon RSS. The
  driver caps `n_outputs_max` at `--rows-per-decode` (default 2,048) and scores
  the served 8,192-token ubatch in disjoint row windows under ONE loan: 2,048
  rows at positions `first..first+2047`, 2,047 at `first+2048..n_ctx-2`. Each
  window is the full 8,192-token graph from an empty KV; a causal row at p
  depends only on tokens 0..p, so the rows equal `llama-perplexity`'s. The
  forecast is corrected below: 2.03 GB of output rows (anon and graph), not
  4.07; the base-need bound at a window width is in §22b.
- **The Q8_0-down loan row** (`code`, R55): init's `tab_loan` view added base
  twice (`tab_row + 6` of a six-row layer) while `fill_tables` and the mutation
  write row +5; a model whose last cached layer has a Q8_0 down asserted at load,
  loan on or off. The view is now the layer's last row.
- **The false comments/build script** (`code`, R57): the driver's "the end frees
  the buffer the logits live in" is false (`--ubatch-reserve` makes the resident
  re-reserve keep a buffer the logits are read from), the flag is required, and
  `build-tools.sh` now says the two tools resolve their libs by RPATH into
  `build-tools/bin` (do not copy them alone) and that `--fresh` needs an empty
  work dir.
- **The public-repo rule**: the four paragraphs quoting the brief's data rules
  with client names are deleted from the design document and are not written
  into any file again.

## The gate commands (Claude's window, the B60)

Both gates run in a Claude window; the GPU work is Claude's. One fresh process an
arm, SOP (`docs/sop-card-window.md`): the sampler before the leg, the card by PCI
id, a hard timeout a leg, no unit running. Same binary family, named with its
sha256. The loan-off reference capture is **data, not a check**: it is produced
once (step 0, below), then the red runs first, then the control, the off twins
and the on arm -- the floor is compared against in step 5.

### The two tools (once, on the card host)

The recipe builds arcint's deb with `LLAMA_BUILD_TOOLS OFF` (`CMakeLists.txt`),
so neither tool of the series is built -- and the recipe's llama tree does not
survive it (`contrib/packaging/arcint/build-deb.sh:50-51`: `mktemp -d` with
`trap "rm -rf $work" EXIT`). `contrib/llama-loan-kld/build-tools.sh` gets one:
`--fresh` assembles it exactly as the recipe does (the pin tarball, the
recipe's `LLAMA_SRC_TARBALL` override if present, build-deb.sh's sha256, then
`git apply` of `contrib/llama.cpp/patches/*.patch` in order); or give the script
a patched tree that already exists on the host. Then it builds both tools with
the recipe's OpenCL settings and the embedded kernels, and prints each
`TOOL <name>: <sha256>`:

    contrib/llama-loan-kld/build-tools.sh --fresh /tmp/loan-tools
    TREE=/tmp/loan-tools/llama.cpp-3e4dfa8f39f904e681737d9f512a6e6338d848aa
    PERPLEXITY=$TREE/build-tools/bin/llama-perplexity
    LOAN_KLD=$TREE/build-tools/bin/llama-loan-kld

The cmake line is the script's own (design §22d): `-DGGML_OPENCL=ON
-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF -DGGML_OPENCL_EMBED_KERNELS=ON
-DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF`, the tree's kernels
embedded from the patched series. The script's log is
`<tree>/build-tools/build.log`; a failure prints its path and the tail. The two
binaries are not standalone: both resolve their shared libs by RPATH into
`<tree>/build-tools/bin` (libllama, libggml, the OpenCL backend), so they are
run there -- do not copy a binary alone. `--fresh` needs an empty work dir; an
existing tree or stale extraction is refused.

### The KL gate (answer-level bar, `CLAUDE.md`)

The window is **1 x 8,192** (a 512-token window cannot cross the 2,048-token
resident chunk and would measure nothing; the driver refuses it). The
reference is the loan-off capture `off-ref.kld`; the twin is `off-a`/`off-b`
(same binary, same configuration) and its mean KL is the floor.

**Round 16 corrects every command below**: the round-15 form lacked
`--n-cpu-moe 48` and pinned no device, so every expert body was placed on the
card beside the expert cache's slots and the runs thrashed until the leg's
hard timeout killed them (`measured-here`, that window). The tools take
`--device GPU.0` (`-dev`, the pin: SOP, the sampler is not a selection) and
`--n-cpu-moe 48` (`-ncmoe`: llama.cpp's placement, the same 48 layers arcint's
`--llama-cpu-moe 48` names), as every command below now does.

0. **the reference capture, once** -- it is data, not a check; every arm below
   reads its exact tokens, so it is the precondition of the red arm too:

    GGML_OPENCL_BANK_STREAM=1 LLAMA_EXPERT_CACHE_CPU_MISSES=1 \
    LLAMA_EXPERT_CACHE_MIB=11600 LLAMA_EXPERT_CACHE_PROFILE=<assets>/expert-profile.bin \
    LLAMA_SLOT_LOAN_MIB=0 \
      $PERPLEXITY -m <flash-next-iq3_s-00001-of-00002>.gguf --device GPU.0 --n-cpu-moe 48 \
        -f <assets>/corpus.txt \
        -c 8192 -b 4096 -ub 2048 --chunks 1 -t 8 -ngl 99 -lm mmap -nr \
        --save-all-logits off-ref.kld

1. **the red, first** -- the lent branch's own corruption, default verify, the
   served shape; it must score and fail the bar:

    GGML_OPENCL_BANK_STREAM=1 LLAMA_EXPERT_CACHE_CPU_MISSES=1 \
    LLAMA_EXPERT_CACHE_MIB=11600 LLAMA_EXPERT_CACHE_PROFILE=<assets>/expert-profile.bin \
    LLAMA_SLOT_LOAN_MIB=6400 LLAMA_SLOT_LOAN_MUTATE=5 \
      $LOAN_KLD -m <flash-next-iq3_s-00001-of-00002>.gguf --device GPU.0 --n-cpu-moe 48 \
        -c 8192 -b 8192 -ub 8192 \
        --ubatch-reserve 2048 --rows-per-decode 2048 -t 8 -ngl 99 -lm mmap -nr \
        --base off-ref.kld --out red.kld --loan

2. **the control** -- the driver without `--loan`, perplexity's split; its rows
   must sit at the twin's floor (the driver's scorer and quantiser are
   perplexity's):

    GGML_OPENCL_BANK_STREAM=1 LLAMA_EXPERT_CACHE_CPU_MISSES=1 \
    LLAMA_EXPERT_CACHE_MIB=11600 LLAMA_EXPERT_CACHE_PROFILE=<assets>/expert-profile.bin \
    LLAMA_SLOT_LOAN_MIB=0 \
      $LOAN_KLD -m <flash-next-iq3_s-00001-of-00002>.gguf --device GPU.0 --n-cpu-moe 48 \
        -c 8192 -b 8192 -ub 2048 \
        --ubatch-reserve 2048 --rows-per-decode 2048 -t 8 -ngl 99 -lm mmap -nr \
        --base off-ref.kld --out ctl.kld

3. **the off twins** -- two more runs of the reference's binary and
   configuration; their mean KL against `off-ref.kld` is the floor:

    for t in a b; do
      GGML_OPENCL_BANK_STREAM=1 LLAMA_EXPERT_CACHE_CPU_MISSES=1 \
      LLAMA_EXPERT_CACHE_MIB=11600 LLAMA_EXPERT_CACHE_PROFILE=<assets>/expert-profile.bin \
      LLAMA_SLOT_LOAN_MIB=0 \
      $PERPLEXITY -m <flash-next-iq3_s-00001-of-00002>.gguf --device GPU.0 --n-cpu-moe 48 \
        -f <assets>/corpus.txt \
        -c 8192 -b 4096 -ub 2048 --chunks 1 -t 8 -ngl 99 -lm mmap -nr \
        --save-all-logits off-$t.kld
    done

`-b 4096`, not 8,192: perplexity's `n_outputs_max` is `n_batch`, and at this
vocab 8,192 rows reserve 8.14 GB of output where the capture reads 4,096 rows
(4.07 GB). The split is the tool's own and the bar is answer-level; the rows are
the same. `--save-all-logits` is the option `--kl-divergence-base` aliases
(`common/arg.cpp:2516`).

4. **the on arm** -- the loan, the served shape:

    GGML_OPENCL_BANK_STREAM=1 LLAMA_EXPERT_CACHE_CPU_MISSES=1 \
    LLAMA_EXPERT_CACHE_MIB=11600 LLAMA_EXPERT_CACHE_PROFILE=<assets>/expert-profile.bin \
    LLAMA_SLOT_LOAN_MIB=6400 \
      $LOAN_KLD -m <flash-next-iq3_s-00001-of-00002>.gguf --device GPU.0 --n-cpu-moe 48 \
        -c 8192 -b 8192 -ub 8192 \
        --ubatch-reserve 2048 --rows-per-decode 2048 -t 8 -ngl 99 -lm mmap -nr \
        --base off-ref.kld --out on.kld --loan

5. **the comparisons** (now that the floor exists), one line each:

    python3 tools/capture_vs_capture.py off-ref.kld --baseline off-a.kld off-b.kld \
      --candidate red.kld --window 0            # must FAIL: mean KL > twin + 0.03
    python3 tools/capture_vs_capture.py off-ref.kld --baseline off-a.kld off-b.kld \
      --candidate ctl.kld --window 0            # must be at the floor (expected 0.0000)
    python3 tools/capture_vs_capture.py off-ref.kld --baseline off-a.kld off-b.kld \
      --candidate on.kld --window 0             # the gate

Gate: the candidate's mean KL at most **0.03 nats worse** than the twin and
argmax agreement at most **1 point** below. A run without the lend lines does not
count.

What the log must show before the numbers count:

- red: the driver's `2048 scored rows in 2 row windows of 2048 (n_outputs_max
  2048)`, `lend ... table check: 0 of R rows differ` (the override is in every
  producer, so the check passes), the WARN
  `LLAMA_SLOT_LOAN_MUTATE=5: layer N: K ex-slot experts read the lend bank's
  rows one entry over during the prefill`, `refill ... verify 0 of N rows
  differ` (no slot byte was touched), `loan returned ... refill+verify ok`,
  `1 chunks, 1 loans taken, 0 refused`, and both windows scored;
- control: `2048 scored rows in 2 row windows of 2048`, `wrote ctl.kld (1
  chunks, 4095 rows a chunk)`, no `loan out` line;
- on: the load's `lend bank check ... 0 of N rows differ from the file`, the
  begin line, both windows' decodes, the refill/verify and the return, `1
  chunks, 1 loans taken, 0 refused`.

The printed line is `3094 slots` -- no thousands separator; a grep for the comma
form `3,094` misses it (the prose in this document writes `3,094` for reading,
the log text does not).

The shapes each arm runs (`code`, design §22c/§23): off twins `-b 4096 -ub 2048`
(two batches, four ubatches of 2,048); control `-b 8192 -ub 2048
--ubatch-reserve 2048 --rows-per-decode 2048` (two row windows, each batch split
into four ubatches of 2,048); red and on `-b 8192 -ub 8192 --ubatch-reserve
2048 --rows-per-decode 2048` (two row windows, each one ubatch of 8,192 under
one loan -- the served shape). The driver logs the shape; a `--loan` at `-ub`
below the chunk is a WARN.

Host-memory forecast (design §22b; the card host: container limit 96 GiB, host
RAM 125.7 GiB, TTM pinned cap 48 GiB):

| arm | anon RSS | host bank, pinned | lend bank, pinned | stream ring | pinned total of 48 GiB |
|---|---|---|---|---|---|
| off (perplexity) | ~8.2 GB (`logits` copy 4.07 GB + `log_probs` 4.07 GB) `(inferred)` | ~37.2 GiB `(inferred)` | 0 | 751 MiB VRAM (`measured-here`) | ~37.2 |
| on (driver, `--loan`, 2,048 rows a window) | ~2.0 GB (2.03 GB output window + one 0.47 MiB row) `(inferred, arithmetic)` | ~37.2 GiB `(inferred)` | 6.20 GiB (`measured-here`) | 751 MiB VRAM | ~43.4 (`measured-here`: round 12's 43,445 MiB) |
| on, `--rows-per-decode 512` | ~0.5 GB `(inferred, arithmetic)` | ~37.2 GiB | 6.20 GiB | 751 MiB VRAM | ~43.4 |

The container limit sees the anon (8.2 GB off, ~2.0 GB on at 2,048 rows, ~0.5 GB
at 512) far below; the pinned banks are not charged to the container's cgroup
(`measured-here`, stage 9). The binding limit is the 48 GiB pinned cap: ~4.6 GiB
of slack on the loan arm. On the **device**, the lent graph's logits rows are the
plan's addend over the unit's one-output prefill shape (round 12's arm, which
fit the 6.20 GiB lend): `need = need(8192 tokens, one row) + (width - 1) x
n_vocab x 4`; the plan allows `need + 0.375 GiB <= 6.20 GiB`. At 2,048 rows the
addend is 2.03 GB and the base need must be at or below 3.93 GiB; at 1,024 rows
(1.02 GB) below 4.88 GiB; at 512 rows (0.51 GB) below 5.35 GiB. The base need is
not measured here; the card's `size_fits` line (need, lend, reserve) decides,
and a refusal is recorded as one. The choice between fewer slots (a larger
pinned bank), a smaller chunk (no longer the served shape), or the narrower
`--rows-per-decode` window is the operator's.

What it exercises (`code`): the lent layers' experts from the host lend bank via
`mul_mm_kq_f16.cl`'s `MUL_MAT_ID` (`llama-graph.cpp:2510-2521`'s third branch);
the prefill of one 8,192-token chunk with the loan out, the refill returning the
slots before the decode. The red arm is the same branch with one table row off by
one -- the only arm here that shows the KL check can fail.

### The depth gate

The loan was measured at 18,802 tokens; the unit serves 131,072. One leg, two
arms, the same corpus and the same binary, ~100,000 tokens
(`tools/fn_sweep.py --prompt-tokens`, the unit's flags):

    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> depth-off \
      --prompt-tokens 100000 --no-answers --env LLAMA_KV_RESERVE_LOG=1
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> depth-on \
      --prompt-tokens 100000 --no-answers --env LLAMA_KV_RESERVE_LOG=1 \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400

Record per arm: the prompt's prefill line (tokens, t/s), the sampler's peak
resident `vram0` and `drm-resident-gtt`, the loan's `loan out` / `refill` /
`loan returned` lines, and the `compute buffers reserved for ... cells` /
`ubatches of N (n_kv_reserve ...)` line the cell budget prints
(`LLAMA_KV_RESERVE_LOG=1` on both arms). No spill arm: round 12 measured the
static chunk-8192 trap at 18k (275.2 t/s, 24,452 MiB, GTT 49,731 -- a thrash),
and the cell cut changes the ubatch but not the 8,192 reserve, so at ~100k it
would carry the same slots-plus-8,192-reserve peak the record already has.

The prediction and its table (`code`, design §22a): both arms are cut by the
cell budget, and the cut scales with the arm's `ub_cap` -- off 2,048, on 8,192:

| prompt tokens | cells | off ubatch | on ubatch |
|---|---|---|---|
| 20,480 | 20,480 | 2,048 | 8,192 |
| 40,960 | 40,960 | 1,024 | 4,096 |
| ~100,000 | 100,096 | 419 | 1,676 |
| 131,072 | 131,072 | 320 | 1,280 |

The cells are `GGML_PAD(p_end, 256)`; the budget is `ub_cap x n_kv_reserve`
(20,480 at the unit's `--llama-reserve-cells`); the floor is 6 and binds only
past ~6.99M cells. The 4:1 ratio holds at every row, past 20,480 cells too.

Prediction the leg can fail (`code` + arithmetic): the on arm's prefill is
**faster** than the off arm's at ~100k (1,676 against 419 tokens a ubatch, the
same mechanism the 18k arm measured), its peak at or under the off arm's (the
6.20 GiB lend pays for the 1,676-token graph), its GTT above the off arm's by
the lend bank. The gain's **size** is the open cell, not its sign: the on arm is
at 20 % of its chunk, and attention at depth grows for both arms. A sign at or
below (on not faster) fails the leg; the magnitude is recorded.

## Round 16 (2026-10-30): both gates scored on the card, the margin, the unit

All numbers below are `measured-here` (Claude's window, the B60, Flash-Next
IQ3_S, the tree at fc218db) unless marked. The commands are the round-16 forms
above (both tools pinned to `--device GPU.0` and `--n-cpu-moe 48`).

### The KL gate (`measured-here`)

8,192 ctx, 1 chunk, 4,095 rows, every arm from the one loan-off capture, the
twin floor from two runs of the reference binary:

| arm | mean KL | against the twin | argmax | against the twin | verdict |
|---|---|---|---|---|---|
| twins `off-a`/`off-b` (the floor) | 0.0060 | — | 97.13 % | — | floor |
| red `MUTATE=5` (layer 22, 147 ex-slot experts one row over) | — | **+0.0372 nats** | — | **-4.95 points** | **fails the bar** |
| control (driver, no loan) | — | -0.0028 nats | — | +1.06 points | at the floor |
| on (served shape, `-ub 8192`, 2 row windows of 2,048, one loan) | — | **-0.0030 nats** | — | **+1.79 points** | **passes** |

The bar is mean KL at most 0.03 nats worse than the twin and argmax at most 1
point below. The red is red on both halves, but it fails the KL half **by only
0.007 nats** (0.0372 against the 0.030 bar) — the lent branch's table-off-by-one
degrades the answer measurably, and the bar separates it with a thin margin; a
second, wider red (a slot-byte corruption, `MUTATE=1`) exits non-zero instead of
scoring and is the exactness red, not the quality one. The control at -0.0028
and the on arm at -0.0030 sit on the twin's own run-to-run floor, so the
served-shape change is rounding-only at the answer level (`measured-here` +
`code`: the rows equal perplexity's, design §22c).

### The depth leg (`measured-here`, `tools/fn_sweep.py`, the unit's flags)

103,638 prompt tokens, one fresh process an arm, no answers (the depth leg's
form):

| arm | prefill | peak VRAM | GTT |
|---|---|---|---|
| loan off | 217.9 t/s | 23,566 MiB | 37,099 MiB |
| loan on (`--prefill-chunk 8192 --llama-loan-chunk 8192 --llama-slot-loan 6400`) | **340.5 t/s (+56.0 %)** | **24,438 MiB** | 45,190 MiB |

The lend: 26 layers (il 22-47), 3,094 slots, 6.20 GiB freed; the refill 0.539 s;
the verify 0 of 12,250 rows differ from the lend bank. At 18,802 tokens
(round 12, the same card): off 436.7 t/s, on 515.5-521.7, peak 23,492 -> 24,229
MiB (`measured-here`). The card is 24,480 MiB, so the depth arm's peak is
**42 MiB under the card's total**.

Gate verdicts on this leg:

1. prefill >= +20 %: **passes** (+56.0 %). The gain is larger than the 18k arm's
   +11-18 % because both arms are cut by the cell budget at depth and the cut
   scales with `ub_cap` (1,280 against 320 tokens a ubatch at 131,072; design
   §22a), so the loan's 8,192-token reserve runs four times the off arm's
   ubatch.
2. decode within the unit arm's spread: not measured on this leg
   (`--no-answers`); round 12's 18k arm measured decode within the spread.
3. the answer-level bar: the KL gate above is that check; the depth leg has no
   answers.
4. the lent slots' bytes equal the file's: **passes** (0 of 12,250 rows differ).
5. peak at or under the unit arm's: **fails** — the on arm's peak is 872 MiB
   above the off arm's and 42 MiB under the card. The prediction in design §22a
   ("the on arm's peak at or under the off arm's") is refuted
   (`measured-here`); what the peak is made of and whether the rule bounds it is
   design §24a/24b, and the margin is the operator's size decision.

The static trap is not repeated here: round 12 measured the chunk-8,192 arm
without the loan at 275.2 t/s and a thrash (peak 24,452, GTT 49,731), which is
why the off arm keeps the resident chunk.

### The unit and the margin (`code` + the measured arm)

The unit's flags to add: `--prefill-chunk 8192 --llama-slot-loan 6400
--llama-loan-chunk 8192` (design §24c: what each does, and the fixed-margin
alternative with its price). What is measured and what is open:

- measured: the served-shape KL at the bar, the control at the floor, the depth
  prefill at +56 %, the refill's 0.539 s, the exactness of the refill's rows,
  the depth arm's peak at 42 MiB under the card;
- open: whether the 42 MiB margin survives a unit's *decode* after the refill
  and a second request at 131,072 (no window), whether the served answers at
  depth stay right (the KL gate is 8,192 ctx), the thin 0.007-nat red margin,
  and the size decision the margin forces (design §24b).

### The integration (`code`)

One commit on branch `loan-integration` above the base the loan branch started
from, carrying the final tree only (0068, the tools, the tests, the docs): the
loan branch's history -- which once held text that must not be published --
never reaches a public branch. The tree was scanned for host names, addresses,
user names and client terms before the commit; the recipe line is in design
§24d.

## Round 17 (2026-10-30): the fixed margin decided and implemented, gate 5 closed

Operator's decision (2026-10-30): **"Margin ok"** -- the loan's transient VRAM
peak during a long prefill is acceptable as long as a fixed margin holds. The
rule adopted is round 16's: `lend - need >= 384 MiB + kCardMargin` with
`kCardMargin` = 1 GiB, and the way to reach it is **option (a), more lent
layers**: a 27th tail layer, about +238 MiB, at the unchanged 8,192 chunk and
the measured +56 % depth rate.

### The rule (`code`, red R58-R60)

- `size_fits(need, lend)` is `lend >= kPeakReserve + kCardMargin && need <=
  lend - kPeakReserve - kCardMargin` (`llama-loan-book.h`: `kCardMargin` = 1
  GiB, `kLendMargin` = 1.375 GiB). The 384 MiB peak reserve stays.
- the lend-set choice keeps the ask-bounded pick (`choose_lend_layers`), and
  `extend_lend_layers` grows it one candidate tail layer at a time, toward the
  earlier layers, stopping at the first layer where the margin holds. The tail
  bounds it; when the tail runs out the pick carries `why` and the loan is
  refused into the resident path with one WARN naming `need`, the lend and the
  fixed margin.
- in the cache (`extend_lend`): a grown layer's lend bank is allocated in host
  memory and **filled from the file** with one whole-tensor `set_tensor` per
  region (round 18: the fresh USM bank has no IQ4_NL split, and only the
  whole-tensor USM write builds it; `set_slices` reads that map and threw on
  every ffn_down_exps layer). The bank's provenance is the file. The read is one
  GGUF read of the grown layer's slot rows, ~238 MiB a layer at the unit's
  11,600 MiB budget, paid once per process; the growth line prints the bytes and
  the seconds. The grown layer's resident slot buffer is **not** freed in
  `extend_lend`: the lend is not published yet, and a failed lend's undo would
  resume on released memory. `drop_tail_buffer` frees it after the lend, after
  the drain and after nulling the tensors; the refill allocates it back. Failure
  paths free the staged banks and leave every resident buffer in place; the
  growth call sits in the sizing's `try`, so a throw takes the resident
  re-reserve and the rebind.
- served figures to check (`code`): the Flash-Next unit is 11,600 MiB of slots
  and `--llama-slot-loan 6400`, so the ask's pick is the measured arm's 26
  layers, 3,094 slots, 6.20 GiB. Round 17 said the rule adds the 27th
  (~+238 MiB); round 19 corrects that to an arithmetic prediction of no growth
  at 6,400 (the lend-vs-need margin is 1.65 GiB, and the card's bound clears by
  4.7 GiB on the recorded parts, design §26b). `need` and the card figures are
  measured at request in the sizing and printed on the `loan out` line; the
  window decides
  (`measured-here`, pending) -- no arithmetic claim is made for it.

Red cases, each a one-line mutation seen failing (`device-free`, tests in
`tests/test_loan_book.cpp`, the llama tree built on the host):

| red | mutation | case |
|---|---|---|
| R58 | `extend_lend_layers` returns the ask unchanged | `a_short_margin_grows_the_lend_by_one_tail_layer` (chosen {2} -> {1, 2}) |
| R59 | `why` left null when the candidates run out | `the_tail_running_out_refuses_the_growth` |
| R60 | `kCardMargin` = 0 (or `size_fits` dropping the term) | `the_fixed_margin_is_the_documented_one`; the 1.375 GiB boundary in `a_size_check_leaves_a_peak_margin` |

### Gate 5

**Closed 2026-10-30 as accepted with a fixed margin.** Round 16's depth leg put
the on arm's peak 42 MiB under the card's total (`measured-here`), which failed
the old "at or under the unit arm's" clause; the operator accepted the peak
under the rule above. The clause is now the fixed margin, and the growth is what
buys it; the unit arm's peak number is no longer the gate.

### The unit

`contrib/systemd/arcint-flashnext.service` keeps the three loan flags and its
header states the margin rule (the 27th-layer growth, the refusal) instead of
the old caveat that the size decision was open.

### The owed window (`measured-here`, Claude's window, the B60)

The served shape at 131,072 ctx, `tools/fn_sweep.py`'s defaults -- the unit's
own flags, `--llama-expert-cache 11600` among them, so no `--set` of the cache
-- and the three loan flags, a prompt near 100k tokens, six answers decoded
after the refill. The loan-on arm and its loan-off twin:

    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-margin \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400 \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1

    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-margin-off \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1

A tiny run first, the loan-on command with `--prompt-tokens 3000` and tag
`fn3k-margin` (the answers and the md5s are the same tripwire, the prefill is
short). The answers' md5s are compared between the on and off arms.

The sampler runs before the leg (SOP §1, on the physical host, not the
container), every 2 s, recording the leg process's `fdinfo`
`drm-resident-vram0` / `drm-resident-gtt`, `VmRSS`, and the host's
`MemAvailable` / `Shmem` / ZFS `arcstats size`, with the `SIGKILL` watchdog at
4 GiB; it aborts the leg if its log is not growing after 5 s. The leg's card
identity is by PCI id (`8086:e211`, `card1`, arcint `GPU.0`), the arm pinned to
`--device GPU.0`, no unit running.

Record per run: the sampler's peak `vram0` and GTT (2 s), the prompt's prefill
line (tokens, t/s), the `loan out` line (layers, slots, freed GiB, `need`,
margin), the `refill` line (slots, GiB, seconds, every row equal to the lend
bank), and `loan returned`; the six answers' decode lines and md5s. Pass: the
margin in `lend - need >= 384 MiB + 1 GiB` holds on the line, the refill's rows
are 0 differing, the answers' md5s match the loan-off arm, and the peak sits
above the card's total only transiently (recorded, not gated). The 3,000-token
run is a smoke check of the same lines before the ~100k window.

### The integration (`code`)

Round 17 rebases the one squashed commit onto `qfndev`'s tip (e31e5f7) and keeps
one commit on top of it; the message names no borrowed rule. The tree is scanned
for host names, addresses, user names and client terms before the commit, and
the recipe line is quoted in the handover. Design §24d keeps the integration
shape; §24b/§24c carry the decision and the unit's flags.

## Round 18 (2026-10-30): the growth path repaired (`code`, card arms owed)

A review of round 17's code found the growth broken (the design is
`prefill-slot-loan-design.md` §25). `code`, device-free:

- a grown layer's lend bank is filled **from the file** with one whole-tensor
  `set_tensor` a region (the ask's pick's path); `set_slices` on the fresh USM
  bank read an `usm_split` that only the whole write builds and threw on every
  IQ4_NL `ffn_down_exps` layer. Cost: ~238 MiB of GGUF a grown layer at the
  unit's 11,600 MiB budget, once per process; the growth line prints it.
- the growth call is inside the sizing's `try`; a throw takes the resident
  re-reserve and the rebind, and `extend_lend` frees the banks it staged.
- `extend_lend` no longer frees the grown layer's spare slot buffer before the
  lend is published; `drop_tail_buffer` does, after the drain and the tensor
  clears.
- init's failed spare allocation nulls each freed `spare_buf[j]`;
  `n_bank_bytes` includes a grown bank.
- the unit is IQ3_S with `--llama-expert-cache 11600`; the service and the
  commands below say so. The card commands use `fn_sweep`'s defaults (the
  unit's flags), no `--set` of the cache.

Red cases R61-R64, each a one-line mutation seen failing on the host (design
§25's table). The card arms, the operator's announced card window, one fresh
process an arm, SOP:

    # 1. the tiny loan smoke run (3,000 tokens), the unit's flags + the loan's three
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn3k-margin \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400 \
      --prompt-tokens 3000 --env LLAMA_KV_RESERVE_LOG=1

    # 2. FORCED GROWTH at 18,802 tokens: the ask is short, so extend_lend runs on the served
    #    file; the whole-tensor fill is proven by LLAMA_SLOT_LOAN_VERIFY=file
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn18k-grow \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 5000 \
      --prompt-tokens 18802 --env LLAMA_SLOT_LOAN_VERIFY=file

    # 3. the owed window at ~100k, loan on and loan off, six answers
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-margin \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400 \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-margin-off \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1

Why 5,000 MiB forces the growth (`arithmetic`, corrected in round 19): the
ask's pick is roughly 21 layers at the record's ~238 MiB a layer (≈4.9-5.0 GiB
freed), and the recorded `need` at the 8,192 chunk is 4.55 GiB; `lend - need` is
then ≈0.26-0.34 GiB, far under the reserve's own sanity of 1.375 GiB, so the
growth runs regardless of the card. The `lend` line prints the exact grown
count. The ask arrives at the size the reserve's rule needs (~6.03 GiB freed,
i.e. 26 layers at 238 MiB a layer), not at a 27th layer by construction.

At 6,400 MiB the ask is the measured 26 layers. Round 19's arithmetic
(design §26b) grew nothing and left 4.7 GiB of spare on the ggml sum, while the
measured peak sat 42 MiB under the card's total: the sum did not see the peak's
outside parts (the ring, the backend's grow-only preallocations, the runtime's
kernels, scratch and heaps). **Round 20 replaces it with the fdinfo form
(design §27b)**: against the recorded 24,438 MiB peak at 103,638 tokens the
rule's prediction is 982 MiB over the limit and the ask grows 5 tail layers to
31 layers (~7,539 MiB lent), no refusal; the prefill price stays the 8,192
chunk's +56 % (340.5 t/s), the growth adds ~0.10 s refill and ~1.19 GiB of file
read once per process. The window decides, its `loan out` fdinfo figure being
the real number.

What each log must show:

- run 1: `loan out`, the refill with `0 of N rows differ from the lend bank`,
  `loan returned ... refill+verify ok`, six answers and their md5s;
- run 2: the growth WARN `grows the lend by K layer(s) ... (X GiB read from the
  file in T s, once per process)`, the `lend` line with the grown layer count,
  the refill with `0 of N rows differ from the file`, `loan returned`; the
  prefill line and the sampler's peak VRAM/GTT;
- run 3: both arms' prefill lines, the loan-on `loan out` (layers, slots, freed
  GiB, `need`, margin), refill and return, both arms' six answer lines and md5s
  (compared), the sampler's peak `vram0` and GTT.

## Round 19 (2026-10-30): the margin against the card's total, the failed growth's tensors

A review of round 18 found nothing blocking and two items that matter for the
operator's decision ("Margin ok"). The design is §26. `code`, device-free:

- **the rule holds a margin to the card.** Round 17's `size_fits` compared
  `need` against the lent bytes only; at the 6,400 ask the margin was 1.65 GiB
  against 1.375 GiB, so nothing grew, and the served shape stayed round 16's
  26 layers. The measured depth peak then sat 42 MiB under the card's total
  (`measured-here`), made of allocations outside `need` (§24b). Round 19 adds
  the card's own bound: after the lend frees its slots, the process's
  device-resident buffers on the card plus the prompt's `need` must stay at or
  below the card's total minus the margin (`card_fits`, `CardFacts`). The
  resident figure sums `ggml_backend_buffer_get_size` over the model's weights,
  the KV memory, the expert cache's slot buffers and gather mirror and the
  output rows; it leaves out the main context's compute term (replaced by
  `need`), the stream ring and gather pages, the driver's objects and the
  host-USM bank (design §26a). The target context's `ctx_other` is never set for
  Flash-Next's MTP (arcint sets it on the draft's params), so that branch was
  dead and round 20 removed it; the draft context's VRAM is the same process's
  and is in the fdinfo figure (design §27a). The card's total is
  OpenCL's global mem size via the device props, not the free query (which is
  `total - 1 GiB`); `LLAMA_SLOT_LOAN_CARD_MARGIN_MIB` moves the 1 GiB margin but
  a 0 or a typo keeps it. The growth path runs until **both** rules hold, one
  WARN on a card refusal names the resident bytes before/after the lend, the
  `need`, the card's total and the margin, and the `loan out` line prints them.
  The reserve's own sanity (`lend - need`, 1.375 GiB) stays as its own check.
- **a failed growth leaves no dangling lend tensors.** In `extend_lend`'s `!ok`
  branch every add layer's `lend[k]` has `buffer`/`data`/`extra` nulled before
  the staged banks are freed (as `refill` does for `tail_slots`); the old
  comment is gone. Without it the next attempt's alloc plan skips a tensor with
  `data != NULL` (growth dead for the process) and a partly placed set would
  write through freed USM.
- **the prediction at 6,400 on the unit** (`arithmetic` over `measured-here`,
  design §26b; **superseded by round 20's fdinfo form, §27b**): resident 20,430 MiB (the IQ3_S unit's 23,478 peak at the
  resident chunk minus its 3,048 compute reserve) - the lend's 6,349 MiB = 14,081,
  plus `need` 4,659 MiB (4.55 GiB) = 18,740 MiB against `24,480 - 1,024` =
  23,456 MiB: **no growth, no refusal; the served shape stays 26 layers**, with
  4,716 MiB of spare on that accounting. Caveat: the parts do not reconcile with
  the measured 24,438 peak (5,698 MiB unaccounted, §24a), so the window reads
  the line instead of trusting the arithmetic.
- **docs corrected**: the round-18 text's "6,400 grows the 27th" and the `need`
  band derived from it were not measurements; they are replaced by the
  prediction above and the 5,000 arm is explained by the reserve's own sanity,
  not by the card. The service header carries the card rule.

Red cases R65-R67, each a one-line mutation seen failing on the host (design
§26d's table): `card_fits` dropping the `need` term or the margin constant 0 or
the growth's card check removed -> `the_card_rule_bounds_resident_plus_need_against_the_card`,
`a_card_short_lend_grows_until_the_card_bounds_it`; `resolve_card_margin`
accepting 0 -> `the_card_margin_knob_moves_the_margin_but_not_removes_it`; the
null writes removed -> `a_failed_growth_nulls_the_lend_tensors_of_every_add_layer`.

The card arms are round 18's, one fresh process an arm, SOP:

    # 1. the tiny loan smoke run (3,000 tokens), the unit's flags + the loan's three
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn3k-margin \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400 \
      --prompt-tokens 3000 --env LLAMA_KV_RESERVE_LOG=1

    # 2. FORCED GROWTH at 18,802 tokens: the ask is short, so the reserve's own sanity
    #    runs extend_lend on the served file; the whole-tensor fill is proven by VERIFY=file
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn18k-grow \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 5000 \
      --prompt-tokens 18802 --env LLAMA_SLOT_LOAN_VERIFY=file

    # 3. the owed window at ~100k, loan on and off, six answers
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-card \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400 \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-card-off \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1

What each `loan out` line must show (run 1, 2, 3's on arm; round 20's form): the
layer/slot count and freed GiB (run 2: the grown count, a higher one than run
1's), `prompt buffers` = the measured `need`, `margin` = `lend - need` with the
1.375 GiB reserve sanity, `fdinfo vram0 X` (the process's own measured
residency), `ggml buffer sum Y` (the fallback and the tripwire's reference),
`first touch Z` (the cold ring's K x largest bank tensor; 0 when the ring is
already there), `the card's total` (run 1-3: 24,480 MiB/23.93 GiB), the card
margin (1.024 GiB), `resident after A + need + first touch = B <= total - margin
(...) by C GiB`, and the sizing/refill seconds. Run 1's 3,000-token loan and
run 2's 5,000-ask growth are predicted to show the ask/growth lines; run 3 is
round 20's prediction below.

The sampler (SOP §1) and the pass rule are round 17's owed window's: the
sampler's `drm-resident-vram0` peak against `24,480 - 1,024` MiB, the answers'
md5s compared between on and off, the refill's rows 0 differing, the card's
bound holding on the line, the peak recorded not gated. **Round 20's prediction
(§27b, `arithmetic` over `measured-here`): the 6,400 ask grows 5 tail layers
(31 layers, ~7,539 MiB lent) against the recorded 24,438 MiB peak at 103,638
tokens; it does not refuse, and the prefill price stays the 8,192 chunk's +56 %
(340.5 against 217.9 t/s), with ~0.10 s and ~1.19 GiB more refill/file read. The
window's `loan out` fdinfo figure is the number that decides; if it reads lower
than that peak's residency, no growth and the peak's residue stays uncaught.**
No 18.74 GiB threshold: under the old ggml sum it was unreachable, and under
round 20's fdinfo form the line carries the real figure.

## Round 20 (2026-10-30): the measured residency, the dead draft branch

A review of round 19 (`code`) found the card rule sums only ggml backend buffers,
while the peak lives largely outside them: the bank-stream ring (K x the largest
bank tensor, a plain `clCreateBuffer`, 751 MiB on the record), the backend's
grow-only preallocations (`prealloc_moe_qa`, `kq_xq`, `kq_xh`, `kq_fa`, `kq2_xq`,
`f32_splitk`, ...) and the runtime's own surfaces (kernel ISA, scratch/private,
heaps). On the shape measured 42 MiB from the card's edge the rule printed
~4.7 GiB of spare, and the "draft context counted" claim was false: the target
context's `cparams.ctx_other` is never set for Flash-Next's MTP (arcint sets it
on the draft's params), so that branch was dead. Round 20 implements the measured
form; every disposed item is `code` unless it names a `measured-here` figure.

- **the process's own residency.** `slot_loan_begin` reads
  `/proc/self/fdinfo` of its xe render fds: `drm-resident-vram0`, deduped by
  `drm-client-id`, matched to the OpenCL device by `drm-pdev` against
  `CL_DEVICE_PCI_BUS_INFO_KHR` (the device props' `device_id`, filled at init).
  One xe client is a trivial match; several without a unique pdev match fall
  back to the ggml sum with a WARN. The read is after `sched_new` freed the old
  compute buffers and after the drain the code already does, before the lend.
- **the rule.** `fdinfo_vram0 - lend_bytes_on_card + need_card + first_touch <=
  total - margin`, with `first_touch` = K x the largest bank tensor while the
  stream ring is not yet allocated (a cold process; K from
  `GGML_OPENCL_BANK_STREAM_RING`, default 4; the backend's ring bytes tell
  whether it exists, and a warm ring is already in the fdinfo figure). The ggml
  sum stays as the fallback and the tripwire, and `size_fits` stays as the
  reserve's own sanity.
- **the tripwire.** `fdinfo_vram0 < ggml sum` refuses the loan with a WARN: the
  process's own buffers are already evicted to GTT, and the loan's reserve would
  build on an evicted process.
- **the line.** the `loan out` WARN names fdinfo vram0, the ggml buffer sum, the
  prompt's `need`, the lend, the first touch, the card's total and the margin,
  one line a card (not `cards.front()`); a card refusal's WARN names the same.
- **the lows.** the reserve-sanity WARN no longer calls the second GiB the
  "card" margin; the card-margin override is clamped at 16 GiB so a huge value
  cannot wrap `mib << 20`.
- **the dead branch.** the `cparams.ctx_other` resident sum is removed (design
  §27a); the draft context's VRAM is the same process's and is in fdinfo.

**The prediction at the 6,400 ask on the unit** (`arithmetic` over
`measured-here`, design §27b): the recorded depth peak is 24,438 MiB at 103,638
tokens of the card's 24,480, and the limit is `24,480 - 1,024` = 23,456 MiB. The
peak is 982 MiB over the limit; a grown tail layer frees ~238 MiB on the card,
so 5 layers (982 / 238 = 4.13 -> 5) bring the prediction to 23,248 MiB, 208 MiB
under, with 7,539 MiB lent over 31 layers. The ask therefore **grows 5 layers
and does not refuse** by this arithmetic; the 8,192 chunk and the depth rate stay
(the recorded +56 % at 103,638 tokens, 340.5 against 217.9 t/s), the growth pays
~0.10 s more refill (~1.19 GiB at the measured 11.5 GiB/s) and ~1.19 GiB of file
read once per process. Caveat: this prediction assumes the read carries the
peak's outside parts; if the fdinfo figure at the read is lower (an allocation
the reserve makes later), the rule may print spare and the window decides.

**Red cases** (`device-free`, `tests/test_loan_book.cpp`; each a one-line
mutation seen failing):

| red | mutation | case |
|---|---|---|
| R68 | the scan's `drm-client-id` dedupe removed (or the xe filter removed) | `the_fdinfo_scan_discovers_the_xe_clients_and_counts_one_a_client` |
| R69 | the `drm-pdev` compare dropped (the front client taken) | `the_fdinfo_client_is_matched_to_the_device_by_drm_pdev` |
| R70 | the tripwire check removed (`if (false)`), or the fallback returning the client | `the_fdinfo_figure_is_the_rule_and_the_ggml_sum_is_the_fallback_and_tripwire` |
| R71 | `first_touch` dropped from `card_fits`' comparison | `the_card_rule_counts_the_fdinfo_figure_and_the_cold_rings_first_touch` |
| R72 | the fdinfo scan removed from the sizing | `the_context_reads_the_fdinfo_after_sched_new_and_drain_and_refuses_eviction` |
| R73 | the card-margin clamp removed | `the_card_margin_knob_moves_the_margin_but_not_removes_it` |

The card arms stay round 19's commands: the 3k smoke at 6,400, the forced growth
at 5,000 with `VERIFY=file` at 18,802, and the ~100k loan on and off with the
answers; the pass rule is the sampler's `drm-resident-vram0` peak against
`24,480 - 1,024` MiB and the `loan out` line's fdinfo figure beside it.

## Round 21 (2026-10-30): zero margin with the kernel's own state, the eviction guard, the dirty lend mirror

Stage 0 of `vram-user-space.md` (operator, 2026-10-30: every byte of the card
useful, not idle). The fixed 1 GiB card margin is replaced by the card's real
state and an eviction guard. Every disposition below is `code` unless it names a
`measured-here` figure; the prices over measured figures are `(inferred)`.

**Round 19's card results** (the announced card window, the 24 GB card, Flash-Next IQ3_S,
unit 1134fac; `measured-here`). Recorded here because they are the numbers this
round's rule and guard are built against:

- 3k loan OK; the forced growth at 5,000 MiB grew **6 layers to 26** (1.48 GiB
  read from the file in 0.674 s), `VERIFY=file` 0 of 9,360 rows.
- 103,638 tokens: prefill **348.0 t/s loan on against 219.0 off (+58.9 %)**; peak
  `vram0` 24,401 MiB of the card's 24,480 (79 MiB from the edge) against 23,566
  off. GTT 47,442 against 37,095 MiB; at 18,802 tokens the on arm had 43,449 MiB
  (~4 GiB extra at depth; eviction or host growth, not told apart then).
- decode mean 34.12 on against 34.48 off; the first answer after the refill 38.4
  against 41.1 t/s.
- the cache's swap-write times after a loan, per answer: **2.21 / 1.67 / 1.03 /
  1.20 s** against off 0.16 / 0.11 / 0.09 / 0.12 s (~10x). That is the per-swap
  host mirror, and §"the mirror fix" below removes it.

**The operator's decision** (`paper`: the operator, 2026-10-30): every byte of the
card useful, not idle; **zero margin** beyond what the kernel and other clients
actually hold, replaced by knowing the card's real state and an eviction guard.
The 1 GiB fixed margin (`kCardMargin`, round 17-20) is no longer the rule when the
kernel query works; it stays only as the fallback when the query fails.

**1. The card's state from the kernel** (`code`). `DRM_XE_DEVICE_QUERY_MEM_REGIONS`
on the process's xe render fd -- the same fd the fdinfo scan found, matched by
`drm-pdev` against the OpenCL device's `CL_DEVICE_PCI_BUS_INFO_KHR` -- returns each
VRAM region's `total_size` and `used`. `used` is the device-wide figure, every
client and the kernel's own VRAM objects included; on the served driver it is not
perfmon-gated (campaign `vram-user-space`, `code`). The rule at the loan's begin
and at every growth step is

    used_now - lend_bytes_on_card + need_card + first_touch <= total - kernel_reserve

(**Round 22 supersedes this formula**: it double-counted the other clients and the
kernel's objects -- see the Round 22 section, `used - lend + need + first_touch +
allowance <= total`.)

The reserve is **measured, not guessed**: it is `used - fdinfo_vram0` at the load
probe and again at each loan, logged every time; `LLAMA_SLOT_LOAN_KERNEL_RESERVE_MIB`
overrides it, and **0 is a valid explicit value** (the zero-margin decision). **Round
22 corrects this paragraph**: the measured difference is information, not a rule
term -- `used` already contains those bytes, and subtracting them counted every
other client and the kernel's objects twice. The rule is at round 22 below. The
fdinfo `drm-resident-vram0` stays the per-process figure and the ggml backend
buffer sum stays the fallback when the query fails (then the round-20
fdinfo + fixed-margin form decides). The card's total comes from the query when it
works, else OpenCL's global mem size via the device props. A failed query is not
fatal: one WARN names the reason.

**2. The eviction guard** (`code`). At the loan's begin and at the loan's end --
before the resident re-reserve and the refill -- the process's fdinfo is read
again: `drm-resident-gtt` above the cache's known host-USM bank bytes
(`n_bank_bytes`), or `drm-resident-vram0` below the ggml sum it should have (the
sum at the begin; the sum minus the card's lend at the end, because the slots are
freed then), means TTM moved the process's buffers. **Round 22 corrects this
paragraph**: the GTT term above the host banks is the process's other legitimate
GTT as well, and it can fire at every begin; the end now judges GTT growth against
a begin baseline and acts on the VRAM shortfall alone (round 22 below). One WARN names both numbers
and the over/short bytes. The reply is shed: the lend is returned early (the
refill path) and the next loan is **sized down by the measured amount**: the
highest grown lend layers are dropped first (the ask's pick's slots are one
buffer); if the grown tail cannot cover it, the next loan refuses into the
resident path instead of overcommitting an evicted process. **Round 23 supersedes
this reply** (see its section): a shed only lowered the headroom the loan creates
(`used - lend + need` rises as lend falls), so it could not relieve the eviction,
and a remainder refused the loan for the process's lifetime. The reaction is now
the next lend's growth by the true shortfall, never a refusal.

**3. The lend mirror's decode cost** (`code`; the ~10x is `measured-here`, round
19). The per-swap host write is gone. A swap now marks its slot dirty
(`DirtyTrack`, one flag a slot); at the next loan's begin, after `finish_pending`
and the caller's synchronize, `flush_lend_dirty` reads **only the dirty runs** from
the card's slot tensors (`get_slices`) into the lend bank (`set_slices`, the bank's
own IQ4_NL split / planar layout). The refill's verify against the lend bank stays
valid. Price from the code: at most one row a lent slot, three tensors -- at the
unit's 26 layers / 3,094 slots / 6.20 GiB the worst case is 3,094 slots x 3,
**6.20 GiB** copied once a loan, ~**0.43 s** at the link's measured 14.3 GB/s
(`(inferred)`); one dirty slot's ~2.0 MiB is ~0.15 ms (`(inferred)`). The old path
wrote ~1.5 MiB a swap at swap time, which is the 2.21/1.67/1.03/1.20 s the card
measured (`measured-here`). Typical decode dirty rows are far below the worst
case; the window's swap-write times say how far.

**What the window decides** (`arithmetic` over `measured-here` where not stated):
the load probe's measured `used - fdinfo` (whether it is stable, and its size);
whether the 6,400 ask under the kernel rule grows the same 5 layers round 20
predicted, or fewer/more, or refuses; whether the 103,638-token on arm stays at or
under the card's total with zero margin (`used - lend + need + first_touch +
allowance <= total`, round 22's form, on the `loan out` line beside the sampler's
`vram0`/GTT peak); whether the guard fires at depth (the ~4 GiB extra GTT) and
what it sheds, and whether a clean loan clears it; and the
decode answers' swap-write times on against off after the mirror fix. The 6,400
prediction of round 20 (§27b) was made under the 1 GiB margin; under zero margin
the kernel rule's own figure decides.

**Red cases** (`device-free`, `tests/test_loan_book.cpp`; each a one-line
mutation seen failing):

| red | mutation | case |
|---|---|---|
| R74 | `DirtyTrack::mark` writing 0 instead of 1 | `a_swap_marks_the_slot_dirty` |
| R75 | the flush iterating every slot instead of `L.dirty.runs()` | `the_lend_copies_exactly_the_dirty_rows` |
| R76 | a host `set_slices` comment/line back in `mark_lend_dirty`, or a swap site back to the old write | `no_per_swap_host_write_remains_for_lendable_layers` |
| R77 | the `drm-resident-gtt` parse removed | `the_guard_reads_gtt_and_the_query_from_a_fake_fdinfo` |
| R78 | **round 23 retires this red**: the shed function is gone (`shed_lend` removed); a true shortfall grows the next lend | `no_shortfall_state_survives_a_clean_loan_or_a_refused_begin` |
| R79 | the end guard's expected VRAM without the lend (`sum_dev.second`, no lend_card) | `the_guard_fires_at_the_loan_end_before_the_refill` |
| R80 | `vram_from_regions` counting sysmem (`if (false) continue`) | `the_kernel_query_sums_only_vram_regions_and_falls_back` |
| R81 | **round 22 retires this red**: the measured reserve is no longer a rule term; its case is the allowance's default | `the_prefill_growth_allowance_is_the_only_addend_and_defaults_to_zero` |
| R82 | **round 22 retires this red**: the kernel rule's limit back to `total - (used - fdinfo)` (the double count); the case is R84's | `the_card_rule_is_used_minus_lend_plus_need_against_the_total_with_zero_margin` |
| R83 | **round 22 retires this red**: the GTT term acting again (`bytes = max(vram_short, gtt_over)`); the case is R93's | `the_eviction_guard_acts_on_the_vram_shortfall_and_logs_the_gtt_growth` |

**Card commands** (round 20's, plus the kernel rule's env and the guard's
readings; Claude's announced window, one fresh process an arm, SOP, sampler
before the leg):

    # 1. the tiny loan smoke at 6,400 (the unit's flags)
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn3k-kernel \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400 \
      --prompt-tokens 3000 --env LLAMA_KV_RESERVE_LOG=1

    # 2. forced growth at 18,802, the file verify
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn18k-grow-kernel \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 5000 \
      --prompt-tokens 18802 --env LLAMA_SLOT_LOAN_VERIFY=file

    # 3. the ~100k owed window, loan on and off, six answers; the on arm read
    #    against the sampler every 2 s (vram0/gtt); the answers' swap-write times
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-kernel \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400 \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-kernel-off \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1

    # 4. the guard's knob (round 22's arm: the guard off, everything else the unit)
    python3 tools/fn_sweep.py <arcint> <flash-next-iq3_s-00001-of-00002>.gguf <assets> fn100k-guard-off \
      --set --prefill-chunk 8192 --set --llama-loan-chunk 8192 --set --llama-slot-loan 6400 \
      --prompt-tokens 100000 --env LLAMA_KV_RESERVE_LOG=1 \
      --env LLAMA_SLOT_LOAN_EVICTION_GUARD=0

The `loan out` line carries (round 22): `kernel used`, the card's total, `fdinfo
vram0`, `need`, `lend`, `first touch`, `prefill growth allowance`, `used - fdinfo`
(information) and the inequality `used after ... + need + first touch + allowance
<= total` with its spare bytes. The load probe's line is `card probe at load`, and
it now runs only for a cache with a lend. The end guard prints its decision every
loan -- `SHEDS` or `does not shed` -- with the expected VRAM (begin sum minus the
lend, plus the reserve and the non-sum residency), the GTT against its begin
baseline and the growth beyond the banks. The pass rule: the answers' md5s on
against off, the refill's 0 differing rows, the guard's `SHEDS` absent on the
healthy depth arm (or recorded with what it shed), and the sampler's `vram0` peak
read against the `loan out` line's bound every 2 s. Nothing is gated on the
transient peak.

## Round 22 (2026-10-30): the kernel reserve counted twice, the guard's false surplus and its lifetime shed

A review of rounds 20-21's code (`code`) found the fdinfo match, the ioctl and the
dirty tracking right and two HIGH design defects against the operator's
zero-margin rule. Every disposition below is `code` (the figures it argues over
are `measured-here`, round 19's card results, and marked where used); the
arithmetic over them is `(inferred)`.

**1 HIGH - the kernel reserve, counted twice** (`code`). `used` from
`DRM_XE_DEVICE_QUERY_MEM_REGIONS` is the device-wide buddy usage
(`xe_ttm_vram_mgr`: `mm.size - mm.avail`): it already holds this process's
objects, **every other client's and the kernel's own**, plus allocator rounding.
Round 21's rule subtracted `used - fdinfo_vram0` as a reserve as well, so with
another client on the card (another process on the card, ~13.4 GiB in the review's example)
the loan refused although there was room. The rule is now

    used - lend + need + first_touch + allowance <= total

with the operator's **zero margin** beyond `used`. `used - fdinfo_vram0` is
logged at the load probe and on the `loan out` line as information (what other
clients and the kernel hold), never a term. The one optional addend is the
**prefill's growth allowance**, named for what it is, default **0**,
`LLAMA_SLOT_LOAN_PREFILL_GROWTH_MIB` (clamped at 16 GiB; a non-numeric value keeps
0 -- a typo is not a band): allocations that first appear after the begin's read
(a grow-only preallocation first used by the prompt, the cold ring's pages, the
driver's own objects) are in no figure yet, and an operator who needs that band
sets it. The round-20 fdinfo + `kCardMargin` form stays the fallback when the
query fails.

**2 HIGH - the eviction guard's false GTT surplus and its lifetime shed** (`code`).
`gtt_over = gtt - n_bank_bytes` had no allowance for the process's other
legitimate GTT -- the runtime's command buffers and heaps, the stage-inputs USM
ring, `ids_host` -- so it could fire at every begin; and a shed never decayed
(`slot_loan_shed` only shrank by grown layers `shed_lend` removes), so one trip
(e.g. the unexplained ~4 GiB of extra GTT at 103k tokens, `measured-here`, round
19) disabled the loan for the process's lifetime. The guard now:

- takes a **per-process GTT baseline** at the loan's begin (`fc->gtt`,
  `host_bank_bytes`) and judges the end against it, minus the host banks and any
  **grown** bank added in between (`bank_growth`); `gtt_over` is the growth beyond
  that ledger and is **logged**, not acted on.
- acts on **`vram_short`** as the primary signal: the process's own buffers below
  the expected VRAM. The expected figure at the end is the begin's ggml buffer sum
  minus the card's lend, **plus** the loan's compute reserve (`need`, resident
  then) **plus** the non-sum residency measured at the begin
  (`vram0_begin - sum_begin`: the ring, the runtime's kernels and heaps). Round
  21's `sum - lend` alone was blind to up to ~9 GiB of eviction.
- **decays**: a loan that completes without a trip clears `slot_loan_shed`
  (`shed_after_loan`), so one trip cannot shrink the lend for the process's
  lifetime; a trip during the loan keeps the measured bytes.
- has a knob: `LLAMA_SLOT_LOAN_EVICTION_GUARD=0` (also `off`, `false`, `no`, any
  case) disables the guard; absent is on, a typo keeps it on.

**3 MEDIUM - the end guard's blind VRAM term** (`code`): fixed by the expected
figure above (`+ need + non_sum`); the end's WARN names the begin's sum, the lend,
the reserve and the non-sum residency, so the band it sees is explicit.

**4 LOW** (`code`): the load probe (an fdinfo scan, an ioctl and a WARN line) now
runs only when the cache has a lend (`lend_bytes_ != 0`), so the agent's and the
coder's processes pay nothing with the loan off; `fdinfo_parse_bytes` handles
`" TiB"`; a failed query's `why` carries errno (`card_why_errno`:
`<what>: <strerror> (errno <n>)`), so ENOTTY/EPERM/EINVAL are visible.

**Red cases** (`device-free`, `tests/test_loan_book.cpp`; each a one-line mutation
seen failing, then green on the restored line:

| red | mutation | case |
|---|---|---|
| R84 | the dispatch subtracting `used - fdinfo` from the limit again | `the_card_rule_is_used_minus_lend_plus_need_against_the_total_with_zero_margin` |
| R85 | the allowance resolver's default non-zero | `the_prefill_growth_allowance_is_the_only_addend_and_defaults_to_zero` |
| R86 | the begin's GTT baseline passed as 0 (host banks only) | `a_false_gtt_surplus_at_the_begin_does_not_shed` |
| R87 | `shed_after_loan` ignoring `tripped` | `a_shed_decays_after_a_clean_loan` |
| R88 | `resolve_eviction_guard` false for every value | `the_eviction_guard_knob_turns_the_guard_off` |
| R89 | the `TiB` line removed from `fdinfo_parse_bytes` | `fdinfo_reads_tib_and_the_query_why_carries_errno` |
| R90 | the errno text dropped from `card_why_errno` | the same case's errno CHECK |
| R91 | `init`'s probe gate removed (`if (true)`) | `the_load_probe_runs_only_for_a_cache_with_a_lend` |
| R92 | the end's `+ base.need + non_sum` dropped | `the_end_guard_sees_the_reserve_and_the_non_sum_residency` |
| R93 | `g.bytes = max(vram_short, gtt_over)` (the GTT term acts again) | `the_eviction_guard_acts_on_the_vram_shortfall_and_logs_the_gtt_growth` |

The suite runs 468 cases green with all ten restored in place. The card arms are
round 21's commands (Claude's announced window; the zero-margin arm is now the
default, arm 4 is the guard's knob), and the window reads the `card probe at
load` line, the `loan out` line's `used`, total, `vram0`, `need`, `lend`, `first
touch`, `allowance` and the guard's `SHEDS`/`does not shed` line against the 2 s
sampler; the commands and pass rule are the Round 21 section's, corrected above.

## Round 23 (2026-10-30): the guard's noise floor, the shortfall grows the next lend, `used == 0` is no figure

A review of round 22's code found the zero-margin rule and the loan-off path
verified, and one HIGH mechanism defect, one MEDIUM figure defect and three lows.
Every disposition below is `code` unless it names a `measured-here` figure; the
arithmetic over measured figures is `(inferred)`. Device-free only: no card arm
is added, and the owed window's lines carry the new readings (§ below).

**1 HIGH -- the end guard had no noise floor, and its reaction could not work
(`code`).** Two facts made round 22's shed wrong on a healthy card:

- its expected figure is exact tensor bytes while `vram0` is BO residency:
  allocator rounding, the reserve's allowed under-run and `reserve_matches`'
  allowance push `vram0` a few MiB below `expected`, so a 1-byte shortfall is not
  evidence of an eviction; round 22 acted on any byte.
- the loan's own residency is `used - lend + need + first_touch`, so shedding
  lend bytes RAISES it: a shed cannot relieve the pressure that caused the
  eviction. And a remainder the grown tail could not cover refused the loan for
  the process's lifetime (round 22 cleared only after a clean loan; a refused
  begin has no end).

The reaction is now: a **true** shortfall -- `vram_short >= kShedFloor`, default
**64 MiB**, `LLAMA_SLOT_LOAN_SHED_FLOOR_MIB` moves it (0 allowed, clamped at
16 GiB, a typo keeps 64 MiB; `resolve_shed_floor`, `growth_from_shortfall`) --
feeds the NEXT loan's sizing as **extra lend bytes** through the existing growth
path (`need + grow_by` as the demand, the card's rule carrying it in
`card.need`), logged as one WARN. The refusal checks keep the actual `need`, so
the shortfall is **never a refusal**; if the tail cannot cover it, one WARN says
so and the loan runs at the rules it holds. **Round 25 corrects this premise**
(`code`): the growth was all-or-nothing on the demand, so a tail that could not
reach `need + grow_by` grew nothing, and the `need` check then refused a loan a
partial growth would have allowed. The growth is two steps now; the shortfall is
a WARN either way. A clean loan stores 0; the begin
TAKES the state and zeroes it at its top, so every refused begin -- for any
reason -- leaves nothing behind. `llama_expert_cache::shed_lend` is removed with
its last caller (R78 retires).

**2 MEDIUM -- `used == 0` is no figure (`code`).** A kernel without
`CAP_PERFMON` (`if (perfmon_capable())` around `used` in older trees,
campaign `vram-user-space`, `code`) reports `used = 0` with a real total, which
made the zero-margin rule vacuous: every loan fits. `query_is_a_figure(used,
total)` requires a total above 0 and a used above 0; a reply failing it is a
failed query, so the fdinfo fallback with its fixed margin decides, and the
caller's existing one WARN names the reason. The query's answer is not silently
half-accepted.

**3 LOW (`code`).**
- `CardQuery::why` is a `std::string` (was a `const char *` into the query's own
  `why_buf`; a query is returned by value, so a copy kept a pointer into the
  source). `card_why_errno` returns the owned string; the unused out-param is
  gone.
- the campaign's Round 21 red table no longer names cases that do not exist:
  R78 retires with `shed_lend`, R81-R83 point at round 22's R85/R84/R93 cases.
- "the image service" is replaced by "another process on the card" in the four
  places it stood (this campaign, the design, the test, the header).
- the guard-knob case's last CHECK was tautological (`!off("0") || on("08")`,
  true by its first half); it now asserts the whole-word rule: `"08"` and `"0x0"`
  keep the guard on, only `"0"` (and the words) turn it off.

**Red cases** (`device-free`, `tests/test_loan_book.cpp`; each a one-line mutation
seen failing, then green on the restored line):

| red | mutation | case |
|---|---|---|
| R94 | `growth_from_shortfall` returning the raw shortfall (the floor dropped) | `a_sub_floor_shortfall_does_nothing` |
| R95 | the card's rule not carrying the demand (`card.need = need` back) | `a_true_shortfall_grows_the_next_lend_and_the_need_rule_refuses_alone` |
| R96 | the begin's `slot_loan_shed = 0;` removed (a refused begin keeps the state) | `no_shortfall_state_survives_a_clean_loan_or_a_refused_begin` |
| R97 | `query_is_a_figure` without the `used > 0` term (used 0 passes) | `used_zero_with_a_total_is_no_card_figure` |
| R98 | the reason's old own-buffer shape back in `CardQuery` (`why_buf`) | `the_query_reason_is_a_copy_safe_string` |

Each mutation was applied to the patched tree, `arcint-test` rebuilt, the named
case seen failing, the line restored and the case green again. The suite runs
**472 cases green** with the fixes in place. R98's first draft used one
`has_between` list; `has_between` requires ALL of its words, so it passed on a
revert that brought back the buffer alone -- the case now checks each token on
its own (`code`, found during this round's red-first).

**What the owed window reads now (`code` + the round-21 commands).** The card
arms are unchanged: the 3k smoke at 6,400, the forced growth at 5,000 with
`VERIFY=file`, and the ~100k on/off arms with the six answers; arm 4 is the
guard's knob. The end guard's line now prints the shortfall **beside the floor**
and reads `FEEDS THE NEXT LEND (grows it by the shortfall; no shed, no refusal)`
or `does not grow the lend (below the floor)`; a next begin that carries a true
shortfall logs it and, if the tail cannot cover it, one WARN (never a refusal).
Round 25: the begin's growth is two steps, so a tail short of the demand still
grows as far as it allows and the loan runs at the `need` rule (R103).
The `loan out` line and the pass rule are round 22's. The window decides whether
a healthy depth arm reads "does not grow the lend" and whether a shortfall that
fires is covered by the growth; nothing is gated on the transient peak.

## Round 24 (2026-10-30): the card figure never engaged -- the runtime's two xe clients per card

**The round-22 window's card result (`measured-here`, Claude's window 20, the
B60, Flash-Next IQ3_S, `0ea2a6c`).** Every arm logged `card probe at load failed
(no xe render fd)`; the `loan out` line printed `fdinfo vram0 18.01 GiB`, equal
to the ggml sum (the fallback), and the loan's end logged `the eviction guard has
no fdinfo client at the loan's end (several xe clients on the same drm-pdev)`.
Neither the kernel query nor the fdinfo figure nor the guard ran in any arm.

The cause is the process's fdinfo itself (`measured-here`, a running arcint's
`/proc/<pid>/fd` and fdinfo): the OpenCL runtime opens **two xe clients per card
in one process** -- e.g. fd 4 -> `renderD129`, client 849, drm-pdev
`0000:0f:00.0`, vram0 2,056 KiB (a probe client) and fd 6 -> the same node,
client 851, same pdev, vram0 16,074,808 KiB (the real one); the same pair on the
other card. Round 20's rule treated two clients on one pdev as a conflict
("several xe clients on the same drm-pdev") and fell back, so the rule's own
figure was never read.

**The fix (`code`, R99-R102 red).** Every xe client in `/proc/self` is this
process's, so the card's figure is the **SUM** of `drm-resident-vram0` and
`drm-resident-gtt` over the distinct `drm-client-id`s whose `drm-pdev` is the
device's (dedupe by client id only -- the scan's job; the runtime's probe client
and the real one are two terms, not a conflict). The `DRM_XE_DEVICE_QUERY_MEM_REGIONS`
ioctl is taken on **any** render fd of that pdev (the smallest, deterministically;
`used`/`total` are device-wide, so it does not need the client the figure is
summed from). A card with clients only on another pdev, or several clients with
no device pdev, is the ggml-sum fallback with a WARN; one xe client stays
trivial. `match_fdinfo` is gone; `sum_fdinfo` returns `FdinfoAgg` (vram0, gtt,
fd, clients, measured, why), and `read_resident` keeps its tripwire (summed vram0
below the ggml sum: refused as evicted).

**The load probe's order (`code`).** The probe sits in `llama_expert_cache::init`
after the cache's own tail/slot allocations (`ggml_backend_buft_alloc_buffer_n`)
and after `load_tensors` loaded the weights, so the backend's xe render fds are
open when the scan runs; it stays there. A text-order case pins it (a scan moved
before the allocation finds no fd and falls back). Its line now prints
`fdinfo vram0 ... over N client(s)`; a successful probe reads `card probe at
load, <device>: used X of Y GiB ..., fdinfo vram0 Z, gtt G, used - fdinfo ...`.

**The window's other round-22 readings (`measured-here`).** The dirty-row lend
copy works: 108 rows, 71.92 MiB in 0.030 s at the loan's begin. Prefill at
103,638 tokens: loan on 348.0 t/s against off 218.8 (+59.0 %). Decode: on 33.67,
off 33.92, the guard-off arm 34.57 (the post-loan mirror cost is gone). Peak
vram0 (sampler, max over the process's fds): 24,400 MiB, the guard-off arm
24,535. GTT 47,442 MiB against the guard-off arm 44,204 MiB -- the ~4 GiB of
extra GTT at depth is **not deterministic across identical arms**, which points
at intermittent eviction at the card's edge (the arm with it decoded slower).
The end guard did not run (the fallback above), so these guard arms are the
mechanism's absence, not its verdict.

**What the next window reads (`code`, the same four commands as round 21's
section: tiny gate at 6,400, forced growth at 18,802 with `VERIFY=file`, ~100k
loan on/off, guard off).** The pass rule is round 22's plus: the **tiny gate's
probe line is a success with a used/total figure** (no `failed (no xe render
fd)`), and its `loan out` line's `fdinfo vram0` **differs from the ggml buffer
sum** on a card that has other clients or runtime objects, and the end guard
names a figure instead of "no fdinfo client". The depth arms repeat: prefill on
against off, decode within the run-to-run spread, the six answers, the refill's
0 differing rows, the 2 s sampler's peak against the `loan out` bound, and the
GTT growth logged; the ~4 GiB non-determinism is looked at, not gated. Nothing
is gated on the transient peak. The exact commands are in the handover and are
round 21's command block unchanged.

## Round 25 (2026-10-30): the carried shortfall's growth in two steps

The final polish before the push; the base `qfndev` `10338c8`, one commit on
`loan-integration`. Device-free only (`code`), no card arm added: the owed
window's commands are round 21's block unchanged.

**1 MEDIUM -- a shortfall could refuse a loan that would run (`code`).** Round
23 feeds the guard's true shortfall into the next lend as `need + grow_by`, and
`extend_lend` was all-or-nothing on that demand: `extend_lend_layers` grows the
pick only while the demand's rules hold, returns `why` when the tail runs out,
and `extend_lend` then committed nothing. The `need` check that follows could
therefore refuse a loan whose ask pick was short of the `need` margin but whose
partial growth would have met it -- the shorter tail made the refusal, not the
shortfall's size. The growth is now two steps (`extend_lend_toward`,
`code`): step 1 asks the tail for the demand; when the tail cannot reach it, step
2 keeps the tail's full growth (as far as the tail allows toward the demand) if
that still meets the loan's own `need` and the card's rule at that `need`. Only
the `need` rule refuses, and it refuses exactly the loan it would refuse with no
carried shortfall; a demand the tail cannot cover costs one WARN. The card copy
in step 2 carries `need`, not the demand, so the shortfall's addend never
reaches a refusal term. The claims corrected: the round-23 text above, the
sizing's and `extend_lend`'s code comments, design §30a and §32a, and the R95
case's old name.

**The value case (R103, `code`).** 26 layers lent (238 MiB each = 6,188 MiB),
`need` 5 GiB (5,120 MiB), a carried 2 GiB shortfall (demand 7,168 MiB) and 5
tail layers (+1,190 MiB = 7,378 MiB). The demand's rule needs 8,576 MiB, so the
tail cannot reach it, and the ask alone needs 6,528 MiB, so the old growth left
it refused; the tail's full growth is 31 layers and meets `need`, so the loan
runs. The test asserts the ask's 26 layers and bytes, the demand's two refusals,
the 31-layer 7,378 MiB outcome, `reached_demand` false and `why` null, and that
one tail layer is short while two meet the rule. **The mutation** -- step 2
dropped (`g.pick = ask;` when the demand is out of reach) -- was applied to the
tree, `arcint-test` rebuilt, the case seen failing on `why == nullptr`, the 31
layers and the bytes, then the line restored and the suite's 474 cases green.
R95's case now names the need rule as the only refusal term and carries value
assertions for the demand (`need + shortfall` against a lend that fits `need`
alone and not the demand); R96 asserts the carry's values (`shed_after_loan`
keeps the measured bytes on a trip, zeroes them on a clean loan, and floor 0
lets every byte count); R101 asserts the probe's summed figure and query fd in
values beside its text-order checks.

**R104, the card rule's copy (`code`, red first).** The same two-step growth
under the kernel's card rule: `need` 5,120 MiB and a carried 3,072 MiB
shortfall (demand 8,192 MiB); the ask is 26 layers (6,188 MiB), the 31-layer
tail is 7,378 MiB, and the card reads used 24,000 of 24,480 MiB. The demand as
the card's `need` does not fit (24,814 MiB); `need` alone does (21,742 MiB),
and step 2's card copy carries `need`, not the demand. The case asserts the
values and the 31-layer commit. **The mutation** -- `own.need = need;` dropped
in `extend_lend_toward` -- was applied, `arcint-test` rebuilt, the case seen
failing (the step-2 card check refuses the tail: `why` set, the pick is the
ask), then the line restored and the case green.

**The WARN's wording (`code`).** The round-23 WARN ended `the loan runs at the
rules it holds, not at a refusal`, which can contradict the card rule's
refusal right after it. It now reads `the demand is not a refusal term, the
loan's own rules still decide`; the text case asserts the new phrase.

**The unit's header (`code`).** `contrib/systemd/arcint-flashnext.service`'s loan
paragraph described round 19/20's fixed 1 GiB margin. It now states the current
rule: the kernel's device-wide `used`/`total` from the xe memory-regions query on
the process's render fd, at zero margin; the process's own figure is the SUM over
the distinct `drm-client-id`s whose `drm-pdev` is the device's; the eviction
guard's true shortfall (64 MiB floor) feeds the NEXT lend's growth, never a shed
or a refusal; the prefill growth allowance is the only addend and defaults to 0;
the loan flags in the example unit are opt-in.

**Commit and identity (`code`).** One commit on the integration branch above
`qfndev`'s tip, the subject under 72 characters, and the round history in the
body. The attribution is two parts with one deliberate blank line between
them -- the implementer line, then the trailer block:

```
Implemented by meseeks (a private DeepSeek-V4.1-Flash quant, run locally); briefed, reviewed and measured on the cards by Claude.

Co-Authored-By: meseeks (private DeepSeek-V4.1-Flash quant) <noreply@example.invalid>
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

so `git interpret-trailers --parse` reads both `Co-Authored-By` lines.

The integration commit and the loan follow-ups are the two commits of the
branch; nothing is pushed.
