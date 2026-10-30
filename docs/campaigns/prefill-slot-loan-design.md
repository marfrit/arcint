# prefill-slot-loan — design, revision 2 (after the stage-0 review)

Revision of `.design-02-slot-loan.md` (stage 0). Every disposition carries its
evidence class: `code` (read in the tree at the branch named), `paper` (as a
commit, a record or a review states it), `measured-here` (a number taken on the
card host), `(inferred)` (arithmetic over the above).
Trees: `~/src/llama-loan` (`loan`, ff1d928 = the pin + arcint's 0001-0067),
`~/src/ligence-loan` (`loan`, d15de58), `~/src/Strata-ref`.

**Review findings 1-8 were each checked in the code before anything below rests
on them.** The verdicts are in §9.

---

## 1. Route: the tail slot buffer, not an alias (`code`; finding 8)

The prefill graph's compute buffer must land in the VRAM the lent slots give up.
Three shapes were on the table: an alias buffer type over the slot range (a1/a2
of the stage-0 note), a second scheduler (b), or a separate slot buffer that the
loan frees (`code`, this route).

**Chosen: the separate tail buffer.** `ggml_backend_buft_alloc_buffer_n(buft,
tensors, n)` allocates a buffer holding exactly the tensors passed
(`ggml-backend.h:39`, `ggml-backend.cpp:94-158`); the default plan groups by
buffer type and splits only at `CL_DEVICE_MAX_MEM_ALLOC_SIZE`
(`ggml-backend.cpp:58-90`). So at load the expert cache's tail layers' slot
tensors go into their own buffer, and a loan is `free that buffer ->
sched_reserve() at the prompt chunk -> run`. The return is `sched_reserve() at
the resident chunk -> re-allocate the tail buffer -> refill -> verify`.

Why this beats the alias:

- **No ggml-opencl change at all** (`code`): no buffer type through
  `ggml_backend_supports_buft` (`ggml-backend.cpp:2040-2050`; the OpenCL device
  accepts only its own type, `ggml-opencl.cpp:15363-15372`), no armed
  allocation, no fake-base offsets (`ggml-opencl.cpp:11974-11990`), no nesting
  against the Q8_0 tensor carve (`ggml-opencl.cpp:12236-12262`).
- **Tensor pointers survive** (`code`): the slot tensors are the same
  `ggml_tensor`s; only their `buffer`/`data`/`extra` move, and
  `g_miss_by_slot`, `g_cache_pairs`, `g_registry` key on the tensor pointer
  (`ggml-opencl.cpp:15482`, `:11515`, `llama-expert-cache.cpp:611`).
- **No bound check in an arm**: the volume is decided by VRAM the driver
  actually grants, not by arithmetic the design guessed.

Three requirements this route carries, all `code`:

1. `ggml_backend_opencl_buffer_free_buffer` only deletes the buffer context
   (`ggml-opencl.cpp:11946-11949`); it does **not** clear the tensors. So the
   loan must set each tail slot tensor's `buffer`, `data` and `extra` to null
   before freeing, because `ggml_backend_tensor_alloc` asserts both are null
   (`ggml-backend.cpp:2284-2295`). Re-allocation runs `init_tensor`, which
   rebuilds the tensor's device extra (`ggml-opencl.cpp:12006-12036`) — the
   finding-8 requirement.
2. The re-allocated buffer must be set to
   `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` again: the scheduler's weight tests read
   it (`ggml-backend.cpp:1116`, `:1483`, `:1820`), and the staged input write
   refuses weight buffers (`ggml-opencl.cpp:12166`).
3. The lend range is **derived from the tensor list**, never from layer
   arithmetic: `n_slots` varies per layer by the profile's rank
   (`llama-expert-cache.cpp:385-392`) and a Q8_0 bank lives in the same context
   and therefore in the same allocation order (`:450-452`). Finding 3 is
   absorbed by the route: there is no byte range to compute.

Peak: the compute buffer is allocated after the tail buffer is freed, so the
peak is the sum of the same parts as today, with the freed slots paying for the
larger chunk (`code`; the byte figures §4).

---

## 2. The lent layers' experts: a lend bank, under the 512 cap (finding 2)

A lent layer cannot read its slots. Its ex-slot experts must stay reachable
during the prompt. The campaign requires exactness and no thrash, so they live
in host memory that the GPU reads over the link — Strata's pinned mirror shape
(`paper`: `expert_source.cpp:1614-1696`).

**Option e1 — extend the existing bank tensor by `n_slots`.** For Flash-Next
(512 experts, ~170 slots a layer) the bank goes from `n_bank+1+n_spare` =
342+1+4 = 347 to 512+1+4 = **517 matrices**. The grouped kernel's cap is
`MAX_EXPERT 512`, a **compile-time local-memory table**, not a runtime count:
`local int cnt[MAX_EXPERT], base[MAX_EXPERT]`
(`mul_mm_kq_f16.cl:2052-2059`), and the host dispatch bounds it at
`n_expert <= 512` (`ggml-opencl.cpp:30845`). So e1 needs (a) a
`-DMAX_EXPERT=` compile define and the `mm_opts` build to pass it
(`ggml-opencl.cpp:2720-2740`), (b) the host bound moved to match, (c) both
re-verified on both cards — and the bigger bank tensor also grows the
bank-stream ring, whose slot size is frozen at learn time from the largest bank
tensor (`ggml-opencl.cpp:27720-27728`): 4 x 347 x an expert's bytes -> 4 x 517
x the same, **~+0.38 GiB inside the loan window** `(inferred, IQ3_S gate bytes
from the record's 1.465 MiB a slot over three tensors)`, and the gather mirror
reserve grows with it (`llama-expert-cache.cpp:487-505`), `~+0.10 GiB`
`(inferred)`. §4 shows the loan window's slack is ~0.36 GiB, so that ~0.48 GiB
does not fit.

**Option e2 — a lend bank tensor beside the bank (chosen).** For each lent
layer, a host-USM tensor `lend[k]` of `n_slots` entries, filled with the very
rows the slots hold (the same file reads, Strata's `choose_resident_keep_from`
shape, `paper`: `expert_source.cpp:245-260`). While lent, the graph builds the
bank branch **and** a lend branch for that layer, `ggml_add`ed like the other
two; `tab_loan` maps an ex-slot expert to its entry in `lend`, or -1. So every
tensor a loan graph sees is under the cap: the bank stays 347, the lend bank is
`n_slots` = ~170, the slot branch is *absent*. `code`:

- the two-branch form is already the prefill shape (`n_tokens > 8`,
  `llama-graph.cpp:2513-2520`), so a third branch is the same construction;
- a bank tensor is gathered into the existing mirror when the bank stream does
  not take it (`ggml-opencl.cpp:30914-30958`): the lend bank is not in the
  stream's learned order, so it gathers — copying per *routed* expert into the
  mirror reserved for the largest **bank** tensor. At a chunk each layer routes
  all of its experts, so the lend bank's gather is the same bytes the stream
  would copy; at the campaign's 8,192 chunk it is `~4.7 GiB over the link per
  ubatch for 19 layers`, 0.33 s at the B60's 14.3 GB/s `(inferred)` — the same
  bytes the stream copies, and the mirror is already paid for. No ring growth,
  no compile define, no dispatch change.
- `cache_src` is the tensor name's `.bank` suffix
  (`ggml-opencl.cpp:30843`); the lend tensors are named `…lend.bank` so the
  zero-fill and the cache-bank token rules apply (`:30884-30890`).

**No file fallback.** Strata's uncovered-experts path reads the GGUF during the
prompt and measured **-26 % prompt with 18 % uncovered** (`paper`, `#1389`).
e2 covers every lent expert, so that price is not paid; e1 would pay ~0.48 GiB
of VRAM and two card re-verifications instead. Both alternatives are worse on
the campaign's peak gate, so the decision is e2.

Cost `(inferred, arithmetic over the record)`: the lend bank is host USM equal
to the loan, ~4.6 GiB at 19 layers, filled once from the NVMe (the refills read
it, not the file, so the file is read once and the refill is a host->card copy,
~0.32 s at 14.3 GB/s).

---

## 3. Scheduler, reserve and the rebind (findings 1, 5, 6, 7)

`code` facts:

- `sched_reserve()` calls `sched_new()`, which *replaces* the scheduler
  (`llama-context.cpp:714-716`, `:755`).
- The expert cache binds to the first scheduler that decodes and never rebinds:
  `advance()` sets `owner_sched` once and returns false for another
  (`llama-expert-cache.cpp:905-907`), and `after_graph`'s unconsumed ids read
  synchronizes that scheduler (`:941-942`). After a re-reserve, adapt is off for
  the process, and an `ids_pending` read would synchronize a destroyed scheduler.
- The reserve's graph shape is `n_tokens = min(n_ctx, n_ubatch)`
  (`llama-context.cpp:743`), so today's reserve *is* the prefill buffer.
- A graph larger than the reserve reallocates through the buffer type
  (`ggml-alloc.c:927-941`), which escapes any loan plan (finding 5).

So the loan needs three context-side pieces:

**a. A ubatch cap.** `llama_context_params.n_ubatch_reserve` (0 = n_ubatch):
the reserve and the effective ubatch are `min(n_ubatch, n_ubatch_reserve)`
(`llama-context.cpp:1986-1998` already caps the effective ubatch by the KV cell
budget; the cap joins that min). At the unit's settings arcint passes
`n_ubatch = chunk`, `n_ubatch_reserve = 2048` with the loan on, so the resident
reserve is the unit's 2,048 and the loan's reserve is the chunk.

**b. Re-reserve at both ends, then rebind.** `slot_loan_begin()`: finish the
pending adapt (`pend_phase == 0`, `adapt_finish` when it is not — finding 7's
"the pending swap must finish, not only poll", `llama-expert-cache.cpp:945-959`,
`:69`), synchronize, relay the tables and wait for them, free the tail buffer,
set the cap to the loan chunk, `sched_need_reserve = true; sched_reserve()`, and
then `rebind(sched)` on each cache (synchronize the old scheduler if
`ids_pending != 0`, zero it, adopt the new one). `slot_loan_end()`: the same
with the resident cap, then re-allocate the tail buffer at the *smaller* reserve
(the re-reserve frees the VRAM the allocation needs), refill, verify, and
rebind again.

**c. The length rule.** A loan is offered only when the prompt still to run
exceeds the resident chunk, and only when the prompt-shaped buffers fit the
lend bytes §4 allows. The prompt need is measured *before* lending, without
allocating: `graph_reserve(chunk, 1, n_outputs, mctx, true, sizes)` ->
`ggml_backend_sched_reserve_size` (`llama-context.cpp:2769-2773`,
`ggml-backend.cpp:2099-2111`) fills `sizes[]`; the sum over the buffers is the
pp need. If it exceeds the lend bytes, the loan is refused, the ubatch stays at
the resident cap, and the prompt runs at 2,048 — no realloc escape, no failed
allocation (finding 5's red run is the mutation that skips this check). A
prompt of 5,000 tokens at a chunk of 8,192 whose loan is refused therefore runs
correctly at 2,048 ubatches instead of dying in a realloc.

**d. Short prompts (finding 6).** The rule is the one in (c): `remaining >
resident chunk`. A 1,000-token follow-up never pays a reserve, a refill or a
lend-bank fill; it runs in the resident reserve the previous request left
behind. The measured reason is `paper`: patch 0043's `~1.9 s` short-follow-up at
a large chunk (`docs/campaigns/flash-next-llama-engine.md:1450`).

---

## 4. Size, and why 19 layers

Estimates from the stage-0 note stand (`(inferred)`, arithmetic over `paper`
numbers): the pp compute need at a chunk is ~2,023 MiB at 4,096 and ~4,238 MiB
at 8,192 under the unit's cell budget; a slot layer is ~242 MiB at 11,600 MiB
over 48 layers. At 8,192: 19 layers lend 4,598 MiB, leaving the compute need
~360 MiB of slack for the refill staging and other small buffers. At 4,096: 10
layers lend 2,420 MiB against 2,023 MiB.

**The third branch is in the measurement** `(code)`: a lent layer's graph has
one more MUL_MAT_ID node than the two-branch graph, and its dst tensor is
`n_embd x n_used x n_tokens` f32 — 2,048 x 10 x 8,192 = **640 MiB** at the
campaign's chunk `(inferred from the model's shape)`. `slot_loan_begin` sizes
the graph with the caches' `lent` flag on (`set_measure_lent`), so the need it
compares with the lend bytes includes it. 19 layers + 4,238 + 640 would be
refused; the operator's knob is `--llama-slot-loan MIB` (more tail layers), and
the trace prints need and lend bytes so the step is visible. Stage 2 picks the
chunk from the same numbers instead of the operator trying sizes.

`measured-here`: the B60's `CL_DEVICE_MAX_MEM_ALLOC_SIZE` is 22.71 GiB (clinfo
on the card host), so a 19-layer tail buffer of 4.6 GiB is one `cl_mem` there;
on a card with a smaller max-alloc `buft_alloc_buffer_n` splits it and the loan
frees the whole multi-buffer (`ggml-backend.cpp:58-158`). No layer arithmetic
is used (finding 3).

Peak: parts cancel between arms only if the compute buffer sits where the slots
were; the loan frees the slots first, so the peak is `unit peak - freed + pp
need + lend-bank mirror delta` = 23,519 - 4,598 + 4,238 + ~0 = **23,159 MiB**
`(inferred)`; the gate is ≤ 23,519.

---

## 5. The cache side: lend, refill, verify, adapt pause

Per layer (all `code` shapes in `llama-expert-cache.cpp`):

- `lendable`: `n_tab == 3` (host banks; a Q8_0 down's bank lives in the slot
  buffer, `:450-452`, and would be freed by the loan, so such a layer is not
  lendable), and in the tail of the `layers` vector under the lend budget.
- **Lend**: set `slot_of[e] = -1` for every ex-slot expert e, `bank_of[e] =
  slot index s` (its row in `lend[k]`), `tab_loan[e] = s`, `lent = true`; the
  other experts keep their bank entries. `upload_tables/flush_tables`
  (`:160-180`) and one blocking `get_slices` of `tab_all` as the wait —
  `check_tables()` (`:183-198`) is 0 after it. Ordered before any graph.
- **Adapt paused**: `loan_out` makes `advance()` count nothing and
  `step()`/`after_graph()` issue no swap and no ids read. Any swap in flight was
  finished before the lend (finding 7).
- **Refill**: the tail buffer is re-allocated, then for each lent layer and k
  one device-side region copy from the lend bank — no host staging, no layout
  conversion, one wait at the end (round 7, §15; §14's whole write and §11
  finding 3's staged read-back are history). Then the tables restored and
  flushed, one blocking read as the wait, and the check over every lent slot and
  k (`:132-138`) — the campaign's exact gate, now against the lend bank by
  default, against the file with `LLAMA_SLOT_LOAN_VERIFY=file`.
- **Ordering with the free**: relay first (tables say "not resident"), then
  free, then reserve. A write into the freed range before the relay would be
  Strata's 2838cbae hazard (`paper`); the loan's relay precedes every graph.

---

## 6. The arcint side: the scope, the request-start check

`code`: `prefill_locked` throws (`src/exec/backend_llama.cpp:1764`, and the last
prompt token's decode at `:1015`), and requests are now cancelled at ubatch
boundaries (a client gone, a stop signal) — `paper`, the review: that change is
on branch wave2a, also touching `prefill_locked`, so the loan's scope must be
an RAII object whose destructor refills and restores on **every** exit.

- `src/exec/slot_loan.h`: the pure rules and the scope (`LoanScope` over two
  hooks), testable without a card: `loan_wanted(remaining, resident_chunk)`,
  `loan_layers(bytes_per_layer, budget)` (whole tail layers), the scope's
  normal/exception exit, the nested-begin refusal and the "a loan is out" test.
- `backend_llama.cpp`: `--llama-slot-loan [MiB]` (lend budget at load, so the
  tail buffer split happens then), `--llama-loan-chunk N` (default
  `--prefill-chunk`); a scope around the `prefill_locked(...)` call only, so the
  last prompt token's decode runs after the refill; a request-start check
  (`llama_slot_loan_out`) that refuses a request while a previous loan is out;
  one lane in stage 1 (`--parallel`/lanes > 1 refuse the loan).

---

## 7. Trace lines and the red switch

The lines, `code`, as the build prints them:

- load: `slot loan: <N> tail layers, <S> expert slots, <X> GiB lendable in host memory`
  (plus `llama_expert_cache::init: ... bank ... GiB in host memory` with the lend bank included);
- begin: `llama_expert_cache::lend, <dev>: <K> layers (il a-b), <S> slots, <X> GiB freed, lend bank <X> GiB in host memory; table check: 0 of <R> rows differ`
  then `llama_context::slot_loan_begin: loan out at <C> tokens: <K> layers, <S> slots, <X> GiB freed, prompt buffers <Y> GiB; size+lend+reserve took <Z> ms`
  (or the refusal line with need and lend bytes);
- load: `llama_expert_cache::init: lend bank check, <dev>: 0 of <N> rows differ from the file (<T> s)`
  (round 7, §15: every lendable row against a read of the file, once, so the refill's default
  check may compare against the bank);
- return: `llama_expert_cache::refill: <S> slots, <X> GiB in <T> s (stage <a>, write <b> from <D> tensors,
  convert 0.000, tables <c>, wait <d>), verify on <e> s: 0 of <N> rows differ from the lend bank`
  (`<N>` counts the zero slot's rows too: `n_slots+1` a layer a k, §11 finding 6; `... from the file`
  with `LLAMA_SLOT_LOAN_VERIFY=file`, and `verify off` with `=0`)
  and `llama_context::slot_loan_end: loan returned: <K> layers, <S> slots, <X> GiB; refill+verify ok`;
- each re-reserve: `llama_expert_cache::rebind: <dev>: rebound to the scheduler after a re-reserve (adapt back on)`.

`LLAMA_SLOT_LOAN_MUTATE=N`, beside the `LLAMA_EXPERT_CACHE_MUTATE` precedent
(1 and 2 there are the adapt's red cases, so the loan has its own switch):
1 the last lent layer's down slice is not copied into the fresh tail buffer
(the check differs, the request throws / the process exits 70, §14/§15);
2 the lend's last table row is dropped (the table check fails, the lend refused);
3 the lent graph's measurement is left out of the sizing (§15): need is
under-counted, the margin check passes, and the card's reserve at the prompt's
chunk fails; the trace prints
`LLAMA_SLOT_LOAN_MUTATE=3: the prompt's graph is sized WITHOUT the lent branch (the red case)`
then `llama_context::slot_loan_begin: the loan's reserve failed; putting the slots back`
and the request answers at the resident chunk (the traces at §15b). Before round 7 the switch
removed the bound comparison instead, which the measured run never entered (need <= lend there),
so it did not fire;
4 the return's resident re-reserve is skipped (the refill's tail buffer alloc
fails: the loan's reserve still holds the VRAM).

---

## 8. Stages after this one

Stage 2: the chunk chosen from the lent bytes with the unit's 2,048 as floor
(Strata's `--prefill auto`, `paper`: `generate.cpp:8524-8575`). Stage 3: the
served A/B against the gate, whose commands are in `.design-02-slot-loan.md` §6
with the loan chunk now set by `--llama-loan-chunk`.

---

## 9. The review findings, and what each one changed

| # | checked in the code | effect here |
|---|---|---|
| 1 | `code`: `llama-context.cpp:714-716`,`:755`; `llama-expert-cache.cpp:905-907`, `:941-942` | §3b: rebind (synchronize, zero `ids_pending`, adopt); a card test runs a second request to prove adapt is back |
| 2 | `code`: `mul_mm_kq_f16.cl:2052-2059` (`MAX_EXPERT 512`, local table), `ggml-opencl.cpp:30845`; the cap is a compile-time table, not runtime-sized | §2: e2 keeps every loan tensor under 512; e1 (raise the define) rejected with its VRAM price |
| 3 | `measured-here` (clinfo, card host), `code`: `:385-392`, `:450-452` | §1 requirement 3, §4: the route uses tensor lists, never ranges |
| 4 | `code`: `backend_llama.cpp:1764`, `:1015`; wave2a cancellation is `paper` | §6: RAII scope + request-start check |
| 5 | `code`: `ggml-alloc.c:927-941`; sizing without allocation exists, `llama-context.cpp:2769-2773` | §3c: measure before lending, refuse instead of realloc |
| 6 | `paper`: 0043's 1.9 s short follow-up | §3d: lend only above the resident chunk |
| 7 | `code`: `ggml-opencl.cpp:27720-27728` (ring frozen at learn), `llama-expert-cache.cpp:945-959` | §3b: finish the swap, not poll; §2: the lend bank is not in the ring |
| 8 | `code`: `ggml-backend.h:39`, `ggml-backend.cpp:58-158`, `2284-2295`; `llama-expert-cache.cpp:471` | §1: the route chosen, with the three requirements it carries |

**Not verified here, still open:** nothing built or run on this host; the two
sizes are `(inferred)`; the reserve's wall time per request is unmeasured; the
lend bank's USM allocation has to fit the card host's pinned budget next to the
bank (see §10c), and the load-time fill's added NVMe read is unmeasured
`(inferred)`; named lanes are refused in stage 1.

---

## 10. The second review round (2026-10-29), after the qfndev rebase

A second review of the built patch (measurements not yet: stage 1 has no card
run) found eight more defects. Each was checked in the code before it was fixed
-- the verdicts below are `code` unless named otherwise -- and the fixes are
commits of `~/src/llama-loan` `loan` and the regenerated
`contrib/llama.cpp/patches/0068`.

| # | checked in the code | what changed | tested by |
|---|---|---|---|
| 1 CRITICAL | `llama-graph.cpp:2511`: the `else if (ecl->lent)` branch sat after `if (n_tokens <= 8)`; while lent the slot tensors have `buffer == data == NULL` (`drop_tail_buffer`), and `ggml_gallocr_is_allocated` treats a NULL-data leaf as allocatable (`ggml-alloc.c:593-596`), so a 1-token tg graph (every `sched_reserve`, `llama-context.cpp:809`) allocated the whole lent slot set into the compute buffer and the refill's `buft_alloc_buffer_n` had no room; a prompt's 4-token last batch (`backend_llama.cpp:1843`) read uninitialised bytes. | the lent branch is tested first; the two-branch form is valid at any `n_tokens` | card: trace of a loan of a 4-token tail + a 1-token reserve graph, `refill ... verify 0 of N` (commands §11) |
| 2 CRITICAL | `llama-expert-cache.cpp:1029` (exchange), `:1223-1225` (adapt queue): `lend[k]` was written once at load and nothing mirrored a swap; the unused `loan_bank_of`/`loan_expert_of` marked the gap | both swap sites write the incoming expert's rows into `lend[k]` at its slot (host-to-host, ~1.5 MiB a swap); a pending adapt is settled, not polled | card: adapt swaps before a loan, then `refill ... verify 0 of N` |
| 3 HIGH | `ggml-opencl.cpp:12097-12110` (`ggml_cl_lb_planar`), `:16745` (`set_tensor_slices` converts device-side); the refill passed raw USM bytes (planes) | the refill reads back through `get_slices` into one reused staging buffer, a wait a layer | card: `refill ... verify 0 of N` on the IQ3_S unit |
| 4 HIGH | `llama-context.cpp:801`, `:820`, `:843` throw; `slot_loan.h`'s destructor called `end()` | `llama_context::slot_loan_reserve` reports failure; the `Scope` destructor and `end()`/`begin()` catch and record `ok() == false`; a failed return leaves the loan out and the request-start check refuses | device-free: `a_throwing_return_does_not_terminate` (fork, the red case is SIGABRT), `a_throwing_begin_is_reported_not_propagated` |
| 5 HIGH | `llama-context.cpp:2239`: `budget = cparams.n_ubatch * n_kv_reserve` against a `reserve_ubatch` reserve | `budget = ub_cap * n_kv_reserve`; the log compares against `ub_cap` | card: a deep follow-up, `LLAMA_KV_RESERVE_LOG=1`, no `reallocating` line |
| 6 MEDIUM | `llama-expert-cache.cpp:lend` returned false with `slot_of = -1`, `lent = true`, `loan_out_ = false`; `slot_loan_begin`'s recovery ran `refill` only `if (loan_out())` | `llama_loan_book::undo_lend` (the patch's own header) restores `slot_of` from `loan_of`, clears `lent`, re-uploads and flushes the tables on every failure path | device-free: `a_failed_lend_puts_the_layers_back` (red: the restore loop removed) |
| 7 MEDIUM | `llama-expert-cache.cpp:87-89`, `:560`: `b_lend` in `bufs`, the destructor freeing every entry | `take_tail` takes the tail buffer out of the list and never adds it; the destructor frees it separately and asserts it is not listed | device-free: `the_tail_buffer_is_not_the_destructor_list` (red: `take_tail` without the erase) |
| 8 LOW | `backend_llama.cpp:1030`, `:1626`: `llama_context_slot_loan_out` read outside `mu_`; two `INFO` lines with `%zu` on an `int32_t`; `LLAMA_SLOT_LOAN_MIB` could enable a loan a tool's flag left off; the bank stream learned the lend tensors | the check is under `mu_` in both request paths; `%d`; `slot_loan_bytes`'s "no parameter set" sentinel (`LLAMA_SLOT_LOAN_BYTES_ENV`), so 0 is off (`slot_loan::lend_param`); `.lend.bank` names are left out of the stream's order; the `R1`/`R3` labels in `test_slot_loan.cpp` corrected | device-free: `a_zero_flag_stays_off` (red: the sentinel for 0) |

### 10a. The graph over a lent layer, at any size (`code`)

While lent, `build_experts` is called twice: over `bank[]` with `ids_bank` and
over `lend[]` with `ids_loan` (`tab_loan`), summed by `ggml_add`. The slot
branch is absent, so no NULL-data tensor reaches the allocator and no CPU-miss
publish (`g_miss_by_slot`) can fire on it. The lend branch's dst tensor is
`n_embd x n_used x n_tokens` f32, 640 MiB at the campaign's 8,192 chunk in the
loan's sizing (`set_measure_lent`), and 80 KiB in a 1-token tg graph.

### 10b. The lend bank is the slots' mirror (`code`)

`lend[k]` row `s` is expert `expert_of[s]`'s bytes in the slot's own layout
(planes for a planar type, GGUF blocks otherwise -- a Q8_0 layer is not
lendable, `init`; IQ4_NL is, through the slice conversion pair). IQ4_NL is lendable since 2026-10-29: its split layout is not copied
raw, the swap mirror reads the bank slice through `get_slices` (GGUF blocks
again) and writes the lend slice through `set_slices` (the tensor's own split),
the pair the refill already used; a Q8_0 layer stays non-lendable because its
bank lives in the slot buffer. The swap sites copy the incoming expert's row
from its bank entry (host USM) into `lend[k][slot]`: the exchange path before
the exchange is queued (its host half rewrites that bank entry), and the adapt
queue's phase-2 issue point. ~1.5 MiB a swap (`(inferred)`, the record's
1.465 MiB a slot over three tensors), ~0.14 GiB over a 96-swap adapt.

### 10c. Host memory the lend bank pins (`code`, sizes `(inferred)`)

- `lend[k]` tensors are allocated once at load, in one USM host buffer
  (`clHostMemAllocINTEL`, `ggml-opencl.cpp:15163-15175`), and live until the
  cache is destroyed: **pinned for the process life, not freed per loan** (the
  refill reads it).
- Its size is `lend_bytes_` = sum over the chosen tail layers of
  `n_slots x (nb2[0]+nb2[1]+nb2[2])`, and `lend_bytes_ <= --llama-slot-loan MiB`
  (`init`). So it is **sized to the loan the operator configures**, never to
  every lendable layer: whole tail layers only, so the rounding is at most one
  layer (~242 MiB at the record's 11,600 MiB over 48 layers).
- At the design's 19-layer loan that is 4,598 MiB = **~4.6 GiB**, on top of the
  bank's ~40 GiB already pinned on the card host (TTM pinned cap 48 GiB,
  `measured-here` for the cap, the bank's 40 GiB from the campaign's records):
  44.6 GiB of 48 GiB, ~3.4 GiB left for the model load's own host pages.
- A larger `--llama-slot-loan` therefore does not fail at the loan, it fails at
  **load** (the USM allocation), which is the visible place: the operator picks
  the MiB with the 48 GiB cap in mind. Default 0 pins nothing (`lend_ctx_` is
  not created).

### 11. The third review round (2026-10-29), before the first card run

A third review of the built patch found eight more defects. Each was checked in
the code before it was fixed -- the verdicts are `code` unless named otherwise --
and the fixes are commits of `~/src/llama-loan` `loan` and the regenerated
`contrib/llama.cpp/patches/0068`. The rule cases live in the patch's own
`src/llama-loan-book.h` and arcint's `tests/test_loan_book.cpp`; each one was
seen failing under its one-line mutation before the fix was accepted.

| # | checked in the code | what changed | tested by |
|---|---|---|---|
| 1 HIGH | `llama-expert-cache.cpp:780-784` cleared `pend` when `!pend_issued`; `decode` returns before `after_graph` on an aborted or failed `process_ubatch` (`llama-context.cpp:2348-2390`), and the cancellation abort is a normal path. Phase 1 unissued: `adapt()` already popped the spares (`:1101-1103`) -- the clear leaked them and adapt stopped at that layer. Phase 2 unissued: `progress()` already applied `pend_tables_out` (`:1248`: `expert_of[slot] = -1`, the victim in spare `x`) -- the clear orphaned the slot: `lend[k]` row `s` still holds the victim, the refill copies it back and the verify compares it against `expert_of[s] = -1` (zeros, `:947-949`, `:151-153`), so the loan stuck out and every request was refused. | the pending phase and its transitions are `llama-loan-book.h`'s `PhaseState`/`issue_phase`/`settle`/`progress_phase`, which the cache embeds (`book_`): a planned phase is **issued** (fence, copies, record) and then settled, never cleared | device-free: `a_settled_unissued_phase_one_returns_its_spare`, `a_settled_unissued_phase_two_applies_its_tables` (red: `settle` clearing at `!issued`) |
| 2 MEDIUM | `:786-799`: the settle issued phase-2 copies with no fence; `adapt_copy` is unordered against the compute queue (`ggml-opencl.cpp:16579-16582`), `progress()` put `adapt_fence` before exactly these copies (`:1251`), and the compute-queue sync came after (`:810`, `slot_loan_begin` synchronizes at `llama-context.cpp:941`, after `finish_pending` at `:937`). A graph still reading the victim slot raced the newcomer's write. | `settle` fences before a phase nobody issued and again between the phase-1 tables and the phase-2 copies (`progress_phase` does the same at the poll) | device-free: `the_settle_fences_the_second_phase`, `a_settled_unissued_phase_is_fenced_before_its_copies` (red: the fence removed) |
| 3 MEDIUM | `:1338-1347` (after_graph), `:793-796` (settle): `lend_mirror` copies all three k per call but sat inside the for-k loops -- 3x the bytes (4.5 MiB a swap, 0.42 GiB over a 96-swap adapt) on the decode thread, for every tail-layer adapt whether a loan happens or not. | the mirror is outside the k loop in `issue_phase`, once a swap, as the exchange path `:1135` | device-free: `the_mirror_is_written_once_a_swap` (red: the mirror inside the loop, 9 for 3 swaps) |
| 4 MEDIUM | `llama-context.cpp:945`: `sched_new` replaces the live scheduler before `need > lend` is known; the resident re-reserves on the refusal paths (`:998`, `:1017`, `:1028`) ignored a false return, and `slot_loan_end` likewise when `back` was false but the refill succeeded (`:1088`) -- the next request prefilled on a scheduler with no reserve for its shape (a realloc through the buffer type, or ALLOC_FAILED). | a failed resident re-reserve **poisons** the context (`slot_loan_poison`, one ERROR line with the reason); `slot_loan_out` reports it and the request-start check refuses; a false `back` at the return is a failure, not a slower path | card: the trace of a loan whose resident re-reserve fails (the refusal line at the next request), §7's `LLAMA_SLOT_LOAN_MUTATE=4`; arcint's `request_start_refuses_while_a_loan_is_out` holds the check's side |
| 5 LOW | `:1008-1010`: `lent = ec->lend() && lent` -- a cache with no lend set returns false (`:820`), so under a layer split the loan was refused every time; the return's refill loop had the same shape. | caches with `lend_bytes() == 0` are skipped at the lend, the refill and the drop (`loan_cache`) | device-free: `a_cache_without_a_lend_is_not_lent` (red: `loan_cache` true), and the call site `the_lend_loop_skips_a_cache_without_a_lend` (red: its `continue` removed) |
| 6 LOW | `:909-916`: the refill wrote rows `[0, n_slots)` but init wrote and verified `n_slots+1` (the zero slot, `:655-664`, `:693`) -- and the slot branch reads that row for a bank expert; the new tail buffer's bytes are undefined. | the refill writes the zero row too and the verify covers `s <= n_slots` (`refill_zero_row`, `refill_check_rows`) | device-free: `the_refill_restores_the_zero_slot_too` (red: the zero row naming `-1`); card: `verify 0 of N+1 rows differ` |
| 7 LOW | `:849-852`: `lend_undo` after a failed table write did not re-check the tables | `check_tables()` runs after the undo, its read is the wait; a still-differing table marks the cache unusable (`loan_broken`, reported by `slot_loan_out`) | device-free: `a_failed_lend_whose_undo_still_differs_is_broken` (red: `lend_broken` false) |
| 8 LOW | `the_tail_buffer_is_not_the_destructor_list` took `take_tail` on a list the tail buffer was pushed into, while production passes `{bslots, bbank}` (it never is): the erase loop was dead and reintroducing `bufs.push_back(b_lend)` after the call passed the test. `a_zero_flag_stays_off` tested `mib << 20`, but the rule lives in `init` (`:286-292`) and `llama_model_default_params`. The comment at `:817` said the tables' buffer is a weight buffer and its writes blocking -- `set_tensor_slices` is non-blocking, the wait is `check_tables`' read. | the tail-buffer case tests production's shape and a source check reads the cache's own init for an entry (the destructor's assert is the runtime half); the flag rule is `resolve_lend_bytes` in the book header, called by `init`, plus a case on `llama_model_default_params`; the lend's comment corrected | device-free: `the_tail_buffer_is_not_the_destructor_list`, `the_destructor_list_gets_no_tail_entry`, `an_explicit_zero_flag_stays_off`, `the_default_params_ask_for_the_environment` (each red under its one-line mutation, the last in `llama-model.cpp`'s default) |

**The per-adapt host copy that remains** (fifth review, F3; `code` for the
paths, `(inferred)` for the times): after the hoist the mirror is one call over
the three tensors, each one slice through `get_slices` (bank USM, a planar type
to GGUF blocks) and `set_slices` (GGUF blocks into the lend USM's own split),
so a swap is **2.05 MB read + 2.05 MB written** for the served file's
IQ3_S+IQ3_XXS+IQ4_NL layer (704,000 + 422,400 + 921,600 B). The traffic alone
is ~0.3 ms at the card host's measured 14.3 GB/s; the two conversions are
small-block scalar loops (a plane shuffle for IQ3_S/IQ3_XXS, the 2+16-byte
interleave for IQ4_NL) and land at a few GB/s, so a swap is **~0.5-1 ms on the
decode thread** `(inferred)`. A 96-swap adapt is ~197 MB a direction (~0.19 GiB,
~0.37 GiB both ways) = **~35-70 ms** of the adapt's wall time, against the
~10 ms the third review counted from the bytes alone. The card A/B confirms it
without a new trace line: the same plan's adapt on a tail layer, the loan armed
and off, decode t/s and the refill's `in <T> s`; the mirror has no timing line
of its own and none is added (F3: no change unless cheap).

**The card commands this round needs** (Claude's window; the loan's own lines
are §7's): a prompt above the floor, one deep follow-up, and the red switch in
turn -- `LLAMA_SLOT_LOAN_MUTATE=1` must throw on the refill's verify (now over
`N+1` rows a layer), `=2` must refuse the lend, `=4` must refuse the next
request (the poisoned state). The mutation for finding 1 is not on the switch:
it is the cancellation path itself -- a client that leaves during prefill, then
one fresh follow-up request, whose trace must show `loan returned ... ok`.

### 12. The card commands that prove the fixes

Claude's window; the commands are in the campaign's status
(`prefill-slot-loan.md`) as `tools/fn_sweep.py` arms, and the lines quoted there
are the patch's own trace:
`llama_expert_cache::lend ... table check: 0 of N rows differ`,
`llama_context::slot_loan_begin: loan out at ...`,
`llama_expert_cache::refill: N slots in T s, verify 0 of N rows differ from the
file`, `llama_context::slot_loan_end: loan returned ... refill+verify ok`, and
the absence of `reallocating` in `LLAMA_KV_RESERVE_LOG=1` on a deep follow-up.

### 13. The fourth review round and the first card run (2026-10-29)

The first card run (`measured-here`, the B60, the served Flash-Next IQ3_S, this
branch's build) found a segfault during load and the root cause below; the
same binary with the loan off loaded and served normally (437 t/s prefill at
18,802 tokens). The verdicts here are `code` unless named otherwise.

**Root cause, `code` + `measured-here`.** Every one of the 48 layers of the
served file has `ffn_down_exps` in IQ4_NL (gate/up IQ3_S, IQ3_XXS or IQ4_XS).
`no_split()` excluded IQ4_NL, so no layer was lendable. The selection loop then
`break`ed at the first non-lendable layer from the tail, reset `lendable` only
on the chosen entries, and left the non-chosen eligible layers `lendable`
while `lend_bytes_ == 0`: `tail_ctx_` was never created and those layers'
`ggml_new_tensor_3d()` ran in a null context (`ggml_new_tensor_impl` <-
`ggml_new_tensor_3d` <- `llama_expert_cache::init`).

**Option chosen: (a), lend IQ4_NL too.** `code`: `get_tensor_slices` and
`set_tensor_slices` already handle IQ4_NL in both the USM and the card paths
(the scales-then-quants split, `ggml-opencl.cpp:16462`, `:16709`), and the
refill has used exactly that pair since the third review. Only two things were
missing: `no_split()` still excluded IQ4_NL, and the swap mirror's raw `nb2`
row memcpy would copy scales into the quants' place. Both are fixed: the mirror
now reads one bank slice through `get_slices` and writes it through
`set_slices`, and the gather already has IQ4_NL's separate scales/quants path
(`ggml-opencl.cpp:30989`). No new table rows, no mixed graph, no second lend
set.

Why not (b), per tensor (gate/up lent, down resident): `code` says the graph
can mix it -- `build_experts` takes `down_t` separately -- but the cache's
residency tables are per expert, not per projection; the down branch would need
its own ids (from `slot_of`) beside the lane's `tab_loan` ids, another table
row, and a slot tensor kept alive while the rest of the layer is lent. It frees
less: with the file's slot geometry (below) gate/up are 0.81-1.66 MiB of a
2.05-2.54 MiB layer slot. At the unit's 11,600 MiB over 48 layers (mean
~241.7 MiB a layer) option (b) needs ~35 tail layers for the prize's ~4.6 GiB,
leaving ~3.8 GiB of down slots resident plus 13 fully resident layers; option
(a) lends 19 layers for the same bytes (the record's ~4.6 GiB). (a) is chosen
for the smaller mechanism and the same freed bytes.

**Bytes a layer lends for this file** (`code` geometry: gate/up K 2,560, 640
rows; down K 640, 2,560 rows -- `docs/campaigns/flash-next-llama-engine.md:598`;
sizes `(inferred)` from the block sizes). Per slot, per tensor:

| tensor type | a slot tensor per expert | MiB |
|---|---|---|
| IQ3_S (gate/up) | 704,000 B | 0.6714 |
| IQ3_XXS (gate/up) | 422,400 B | 0.4028 |
| IQ4_XS (gate/up) | 870,400 B | 0.8301 |
| IQ4_NL (down) | 921,600 B | 0.8789 |
| Q8_0 (down, not lendable) | 1,740,800 B | 1.6602 |

A full-layer lend is `n_slots x (gate + up + down)`: 1.953 MiB a slot for
IQ3_S+IQ3_XXS+IQ4_NL (2,048,000 B), 2.222 MiB for IQ3_S+IQ3_S (2,329,600 B),
2.380 MiB for IQ3_S+IQ4_XS (2,496,000 B), 1.685 MiB for IQ3_XXS+IQ3_XXS
(1,766,400 B), 2.539 MiB for IQ4_XS+IQ4_XS (2,662,400 B). The selection sums
each chosen layer's own `n_slots` (the profile varies it), and `init`/`lend`
print the exact sum. At the unit's 11,600 MiB over 48 layers, 19 tail layers
are ~4.6 GiB; (b)'s gate+up only is 1.074, 1.343, 1.501, 0.806 or 1.660 MiB a
slot for the same pairs.

**Selection, from scratch (`code`).** `choose_lend_layers()` in
`llama-loan-book.h`: every layer starts `lendable = false`, the scan starts at
the tail, skips a non-lendable layer or a layer whose bytes do not fit the
remaining budget, and continues earlier instead of breaking; the set is
ascending. The separate tail buffer is a tensor list, so it needs no
contiguity. Every loop that assumed `[lend_lo_, lend_hi_)` (lend, undo,
measure, refill, restore, verify, logging) now walks `lend_set_`. Nothing
chosen means no loan: the tail/lend contexts are not created and every slot
tensor stays in the ordinary buffer, with one line
`llama_expert_cache::init: the slot loan lends nothing: <reason> (N GiB asked)`.
Device-free red cases: `a_tail_with_no_lendable_layers_lends_nothing` (red:
the non-lendable skip removed), `a_loan_smaller_than_one_layer_lends_nothing`
(red: the budget check removed), `a_mixed_set_skips_a_non_lendable_layer_and_a_
layer_that_does_not_fit` (red: `break` at a non-lendable layer), and
`a_loan_skips_an_oversized_tail_layer_and_takes_an_earlier_one` (red: `break`
at a layer that does not fit).

**The fourth review's leftovers (`code`).**
- (i) `slot_loan_begin`'s `!mctx` refusal ran after `sched_new`; it now
  initializes the memory module first, before the scheduler is replaced, so the
  early return leaves the live scheduler and its reserve untouched.
- (ii) the failed `slot_loan_reserve(cap)` path put the slots back, but its
  first reserve was sized with the lent graph; it now reserves the resident
  chunk again after the refill, as `slot_loan_end` does (and only when every
  cache's refill said ok).
- (iii) the call sites, not only the one-line helpers: `the_lend_loop_skips_a_
  cache_without_a_lend` reads `slot_loan_begin`'s `continue` (red: removed);
  `the_refill_writes_and_checks_the_zero_slot` reads the refill's zero-row
  block (`refill_zero_row`, `zeros.data()`, `refill_check_rows`; red: the row
  naming -1); `lend_undo_checks_the_tables_again` reads `check_tables()` and
  `lend_broken(` in `lend_undo` (red: the check removed); and
  `the_destructor_list_gets_no_tail_entry` no longer depends on one spelling --
  any `take_tail(` counts, and only `push_back(b_lend)`/`push_back(b_tail)`
  count as a tail entry (red: the push reintroduced).
- (iv) a refusal names the actual state: `llama_context_slot_loan_state()`
  ("off", "armed", "out", "poisoned", "broken") is what the request-start
  message prints, and `/props` publishes it under `slot_loan.state`
  (device-free: `props_reports_the_slot_loan_state`; the red case is the key
  removed from `/props`).
- (v) the duplicate `### 11` headings: the card commands are §12, the third
  review is §11, and its row 5 now names the call-site case too.

**The poison policy: `poison-exit` (operator, 2026-10-29).** When the loan's
state is poisoned or broken -- a failed resident re-reserve, a refill whose
verify differs or whose tail buffer cannot be allocated, an undo whose tables
still differ -- `llama_loan_book::poison_exit()` writes one ERROR line naming
the step and the counts/layers and exits non-zero (70), so the unit manager
restarts a clean process. Nothing serves in the poisoned state. Staying up and
refusing every request is the alternative and is that one function's body (drop
the `exit` and return). `slot_loan_poison` (the context's reserve paths) and
the cache's refill/undo call it; the red case
`the_poison_policy_exits_non_zero_with_the_cause` is a forked child that must
exit non-zero with the cause in its line (red: the `exit` removed), carry the
line through the llama log callback too (red: `log_line` removed), and run no
static destructor (red: `std::exit` instead of `std::_Exit`, the ATEXIT marker;
§14 finding 2).

**Default-off unchanged (`code`, asserted device-free).** With the loan off
(the default `--llama-slot-loan 0`, an explicit off) the dense agent (no expert
cache) and the coder (all experts resident, no cache) never enter
`build_moe_ffn`'s `ecl` branch -- `llama_expert_cache_lookup()` finds nothing,
so `experts = build_experts(cur, up_exps, ...)` is the same path
(`the_plain_moe_path_is_unchanged_without_an_expert_cache`). Flash-Next's
normal decode keeps `reserve_ubatch == n_ubatch` (0 is resolved to n_ubatch,
`llama-context.cpp:341`), so `ubatch_cap()` returns `n_ubatch` and the cell
budget is `n_ubatch x n_kv_reserve`, as before
(`the_decode_budget_uses_the_capped_ubatch`, red: the budget off `n_ubatch`).
The tables' extra `tab_loan` row is allocated only inside an expert cache's
`init`; the plain path reads none of it. `floor_chunk()` returns the prefill
chunk when the loan is off (the existing case).

### 14. The fifth review round (2026-10-29): the refilled slot tensor's extra, the poison exit

Two findings (one blocker, one small) and one report. Each was checked in the
code first; the verdicts here are `code` unless named otherwise. The fixes are
commits of `~/src/llama-loan` `loan` and the regenerated
`contrib/llama.cpp/patches/0068`.

**The blocker's evidence (`code`, verified here).** The card-side IQ4_NL layout
is not the plane the buffer holds: `ggml_backend_opencl_buffer_set_tensor`'s
whole-tensor write builds `ggml_tensor_extra_cl_iq4_nl` with the scales and
quants as separate sub-buffers of the buffer's `data_device`
(`ggml-opencl.cpp:13113-13177`, `tensor->extra = extra` at `:13174`) and
returns. init writes every slot tensor whole, so the load-time extras are the
split ones (`llama-expert-cache.cpp`'s init fill, `ggml_backend_tensor_set(t,
host.data(), 0, host.size())`). `drop_tail_buffer()` clears `buffer`, `data`
and `extra` (`:906-916`), and the refill's
`ggml_backend_buft_alloc_buffer_n` re-allocation runs `init_tensor`, which
installs the **plain** `ggml_tensor_extra_cl` only
(`ggml-opencl.cpp:11956`, `:11986-11991`). The refill's `set_slices(L.slot[k], ...)`
then takes the card SOA path and casts that plain extra:
`q = data_device, d = (cl_mem) actual_size` (`:16738-16747`) -- a CL_CHECK abort
or silent corruption. The verify's `get_slices` (`:16494`), `ggml_cl_regions`
for an adapt copy or an exchange (`:16534`) and the kq kernels' weight args
(`:27861-27865`) make the same cast. Every lent layer of the served model has
an IQ4_NL down, so this fires on every loan.

**The same property on other types (`code`).** `set_tensor` builds a
specialised extra for Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, MXFP4, Q8_0, IQ4_NL, Q4_K,
Q5_K and Q6_K (`ggml-opencl.cpp:12222-13710`); all but Q8_0 are lendable, and
a slice write after the re-allocation would cast the plain extra for every one
of them. A planar type (IQ3_S, IQ3_XXS, IQ4_XS, IQ2_*, Q2_0) keeps the plain
extra and converts inside the slice path (`ggml_cl_lb_planar`, `:16766`), so it
is outside the set.

| # | checked in the code | what changed | tested by |
|---|---|---|---|
| 1 BLOCKER | `ggml-opencl.cpp:13113-13177` (whole write builds the split extra), `:11956`/`:11986-11991` (the re-allocation's plain extra), `:16495`, `:16534`, `:16747`, `:27861` (the casts); `llama-expert-cache.cpp:906-916`, `:923`, `:954/964` | the refill assembles the full `(n_slots+1) x nb2` image per layer and k -- rows 0..n_slots-1 through `get_slices` from the lend bank, the zero slot's row zeroed -- and writes it whole with a blocking `ggml_backend_tensor_set(L.slot[k], img, 0, ggml_nbytes(L.slot[k]))`, which rebuilds the extra; `refill_whole_write(type)` names the set above (the extra's own type list, not just IQ4_NL), the planar and plain types keep the slice path | device-free: `the_refill_writes_a_rebuilt_iq4_nl_slot_whole` and `the_whole_write_set_names_the_set_tensor_types` (red: the whole branch's condition forced false; IQ4_NL dropped from the set -- both seen failing); card: the refill's `verify 0 of N rows differ` on an IQ4_NL layer, then right answers (§12's arms) |
| 2 SMALL | `llama-loan-book.h:158-162` (`std::exit(70)`); `llama-impl.cpp:61-82` (the log callback: logger state read, call, no mutex, no device); `arcint src/util/log.cpp` (its own mutex, never the context's) | `std::_Exit(70)` after the message, no static destructors; the line goes through the llama log callback first (`log_line`, defined in `llama-expert-cache.cpp` with `LLAMA_LOG_ERROR`) and then on stderr with a flush | device-free: `the_poison_policy_exits_non_zero_with_the_cause`, extended with a `llama_log_set` capture (`LOG:ERROR`) and an `atexit` marker (red: `log_line` removed; `std::exit` for `std::_Exit` -- both seen failing) |
| 3 report | the mirror's paths (`llama-expert-cache.cpp:747-772`): per swap three `get_slices` from the bank USM and three `set_slices` into the lend USM | no change; the cost is reported (§11's paragraph): 2.05 MB read + written a swap for the served file's layer, ~0.5-1 ms a swap, ~35-70 ms a 96-swap adapt | the card A/B confirms it (decode t/s on a tail-layer adapt with the loan armed and off) |

**The red switch moves with finding 1.** `LLAMA_SLOT_LOAN_MUTATE=1` now flips
a byte inside a real row of the whole-write image (row `n_slots-1` of the last
lent layer's last tensor), because the whole write replaces every row and
skipping one would not be the failure any more; a planar type's slice path
keeps the old "last lent slot unrefilled" form. On the card the mutation must
still exit 70 with `the refill did not verify: ... rows differ`.

**Round 7 supersedes the two paragraphs above** (the whole image and "no
GGML_OPENCL change"): §15 — the refill copies the lend bank's card layout by
regions through three new proc addresses, and the default check compares against
the bank, with the file form kept for gate arms. This section stands as the
record of the sixth round.

**The load path is unchanged.** init's fill still writes each tensor whole, so
the extras at load stay the split ones; the loan clears them before the free
(the buffer context's destructor deletes the temp extras and releases their
sub-buffers, `ggml-opencl.cpp:11557-11620`), and the refill rebuilds them. No
GGML_OPENCL change, no new proc address.

---

## 15. Round 7 (2026-10-29): the refill approaches the link, the two gaps, the margin

Measured before this round (`measured-here`, B60, Flash-Next IQ3_S, the loan's
flags, an 18,802-token prompt): the refill took 2.66-2.86 s for 6.20 GiB in
26 layers (il 22-47), 3,094 slots — **~2.3 GB/s**, while the card's link moves
14.3 GB/s with USM host memory. That time is the trace's `in <T> s`, which
excludes the verify (the timer is read before the check loop), so the return's
real cost was larger by the check, itself 9,360 blocking row reads.

The four costs, `code`, and what each became:

| phase | before | after | evidence |
|---|---|---|---|
| stage | a whole `(n_slots+1) x nb2` host image a tensor (238 MiB), assembled per layer | the tail buffer's allocation only; the IQ4_NL split extra built without bytes | `code`: `llama-expert-cache.cpp`'s refill; `ggml-opencl.cpp`'s `ggml_cl_iq4_nl_extra` |
| write | `get_slices` of the lend bank into host staging (a plane/GGUF conversion on the CPU), then a blocking `set_slices`/`tensor_set` a tensor, the IQ4_NL whole write with a temp device buffer and its conversion kernel | one non-blocking region copy a tensor straight from the USM bank into the new slots (`ggml_backend_opencl_lend_write`), no conversion, no temp buffer | `code`: the function; `measured-here` pending |
| wait | one blocking tables read a layer (78 waits, each releasing a staging buffer) | one blocking tables read after all layers (`code`) | `code` |
| verify | 9,360 blocking row reads through `get_slices`, each followed by a read of the GGUF row | default: every row against the lend bank in the card's layout, one blocking region read a tensor (`lend_read`) plus a memcmp; `file`: the same batching, GGUF bytes against the file | `code` |

**Why a region copy is exact.** The lend bank is not GGUF bytes: it is filled
through `set_tensor`/`set_slices` at load, so it holds the CARD'S layout — planes
for the planar types, the d/q split for IQ4_NL, f16 for BF16 — and an adapt's
mirror copies a bank entry's bytes that are already in that layout
(`llama-expert-cache.cpp`'s `lend_mirror`). `ggml_cl_regions` names each region
with its per-slice size and base, host or device; the copy is
`clEnqueueWriteBuffer` from the USM pointer into the slot buffer (IQ4_NL: two
regions). Nothing converts, so nothing can be converted wrong.

**Why the default check is not weaker.** The check is exact, byte for byte, over
every row of every lent tensor including the zero slot; what changes is the
reference: the lend bank. The chain to the file is proven where it can be proven
cheaply —
- at load, `llama_expert_cache::init` checks every lendable row of the bank
  against a fresh read of the file (`lend bank check, 0 of N rows differ from the
  file`), a one-time 6.20 GiB read, not a per-prompt one;
- every mirror writes bank bytes, i.e. the file's bytes, and no swap runs while
  a loan is out (`loan_out` pauses adapt), so the bank's bytes at a refill are
  still the file's;
- `LLAMA_SLOT_LOAN_VERIFY=file` keeps the campaign gate's direct form (GGUF
  bytes against the file) for the arms that want it; `=0`/`off` is the timing
  arm. A word that is neither is the default, never silently off.

The residual risk this takes: a mirror that writes the wrong bank entry would
put garbage into the slot and the bank's same garbage would compare equal; the
load check and the `file` arms catch it, the default does not per refill. A
per-swap file check of the mirrored rows (1.5 MiB a swap) would close it; it is
named for later, not built (`inferred` cost ~0.5-1 ms a swap on NVMe).

### 15a. Expected numbers (`inferred`)

- stage: ~0.01-0.05 s (an allocation and 78 tiny sub-buffer builds);
- write: 6.20 GiB from USM host memory over the link at 14.3 GB/s = **0.43 s**,
  if the driver's `clEnqueueWriteBuffer` from a USM pointer is a DMA. If it
  stages through the CPU instead, the line shows ~2.3 GB/s (2.7 s) again, and the
  next lever is a kernel reading the USM rows (the bank's pointers are already
  schemes this backend passes to kernels, `usm_set_arg_ptr`); the per-phase line
  decides it in one card run;
- verify (bank, default): a 6.20 GiB read-back over the link (0.43 s) plus a
  memcmp of 6.20 GiB on the host (~0.3-0.6 s) = **~0.8-1.1 s**;
- verify (`file`): the read-back, a planar conversion and 6.20 GiB of GGUF reads
  = several seconds (`measured-here` next window), which is why it is not the
  default;
- **the refill's total: ~0.5 s with `=0`, ~1.3-1.6 s with the default check,
  down from 2.66-2.86 s plus a check that was never timed.**

### 15b. Red switch 3, and why it did not fire before

`code`: the switch removed the comparison `need > lend` at
`llama-context.cpp`'s size check. In the measured configuration that comparison
was false in a healthy run too (`need <= lend`), so removing it changed nothing:
the lend, the reserve and the return ran as usual. The switch now removes what
the comparison depends on — the lent graph's measurement (`set_measure_lent`,
the third MUL_MAT_ID branch, ~0.64 GiB at 8,192) — so `need` is under-counted,
the margin check passes on it, and the card's reserve at the prompt's chunk
fails. The lines that prove it:
`LLAMA_SLOT_LOAN_MUTATE=3: the prompt's graph is sized WITHOUT the lent branch (the red case)`,
then `llama_context::slot_loan_begin: the loan's reserve failed; putting the slots back`,
then the refill's `put back` run and the answer at the resident chunk.

### 15c. Overlap: not this round (`code`, `inferred`)

- **While the prompt's last ubatches run:** no. The refill's first act is the
  tail buffer's allocation, which needs the resident re-reserve, which frees the
  buffers the running ubatches use; the VRAM cannot hold both (the measured peak
  is 24,242 MiB of 24,480).
- **Early-returned layers:** no. The lend frees ONE tail buffer for all 26 layers
  (`drop_tail_buffer`/`take_tail` act on the whole tensor list), so no layer is
  "back" before the others. A per-layer-group buffer would let the loan release
  groups progressively, but the reserve is one scheduling unit and the graph
  sizing would have to move with it — not cheap, named for later.
- **With the last token's decode:** no. That decode reads the slots, so the
  copies must precede it; they do (in-order queue), but the check must precede
  the answer, and moving the check behind the first decode graph would delay a
  poison exit past an answer that may already be wrong. The writes themselves
  are already off the critical path by being non-blocking; the wait is the check.

### 15d. The fifth review's nits

- the comment's planar list: the planar set is ggml-opencl's `ggml_cl_lb_fields`
  cases — IQ2_XXS, IQ2_XS, IQ2_S, Q2_0, IQ3_S; IQ3_XXS and IQ4_XS are not
  planar. The refill's comment now names exactly those, and
  `the_refill_comment_names_the_planar_types` reads both the source of truth and
  the comment.
- BF16: the whole-write rule is gone, not widened. The refill copies the bank's
  card layout for every type, so BF16's f16-on-card conversion at load is already
  in the bank's bytes and no type can be forgotten. (The rule the review named,
  "whole write unless planar", would have needed BF16 added; the region copy
  needs no list at all.)
- the IQ4_NL whole write's transient device buffer (one slot tensor's nbytes,
  ~78 MiB on the served model) no longer exists: the copy writes the split
  regions directly. The margin below covers whatever a later path allocates
  beside the plan.

### 15e. The peak and the margin

`measured-here`: the loan run peaked 24,242 MiB of the card's 24,480; the unit
arm peaks 23,519. Where the loan's peak sits: the prefill graph's compute
buffers at the 8,192 chunk (the scheduler's reserve), plus the slots of the
layers the loan did NOT lend, plus the scheduler's other buffers — and NOT the
lent slot buffer, which is freed, and NOT the refill's transient, which no
longer exists. The size check now requires `need + 384 MiB <= lend`
(`llama_loan_book::size_fits`, `kPeakReserve`), and the refusal line prints
need, the reserve and the lend bytes, so one card run settles whether the
measured configuration still fits (if it refuses, need is within 384 MiB of the
lend and the margin is the operator's size decision — the measured arm's own
headroom was 238 MiB, so the reserve is what keeps the plan off that edge).

---

## 16. Round 8 (2026-10-29): the seventh review's five findings, the two card findings

Card results for round 7 (`measured-here`, B60, the campaign's flags and an
18,802-token prompt, marker `mutate1-driver`): the region refill takes
0.514-0.541 s for 6.20 GiB (write 0.50 s, wait 0.015 s; was 2.7 s); the load's
lend bank check reads 0 of 9282 rows differ from the file in 1.5 s; prefill with
`LLAMA_SLOT_LOAN_VERIFY=file` 498.4 t/s (its verify 3.295 s, 0 of 9360 rows
differ), with `=0` 546.4/546.6 t/s, loan off 436.6 t/s. So the verify's cost
decides the gain: the default (bank) form must be exact AND cheap. The region
copy's exactness itself was confirmed by the review (`code`): `lend_write`
rebuilds IQ4_NL's split extra on the fresh buffer, planar slices carry per slice.

| # | checked in the code | what changed | tested by (each seen RED under its mutation) |
|---|---|---|---|
| 1 BLOCKER | `llama-expert-cache.cpp`'s refill check: `lend_read(L.slot[k], 0, n_slots+1)` and `lend_read(L.lend[k], 0, n_slots)`, then a per-row `memcmp` at `s*nb2`; `ggml-opencl.cpp` `ggml_backend_opencl_lend_read` emits regions one after the other (`[d x n][q x n]`), so the slot's q region sat one extra row ahead of the bank's and the zero row was compared inside that q region: ~8/9 rows "differed" on a correct copy, exit 70 | both sides are read with the SAME `n` (`n_slots`) and compared as whole buffers; the zero slot is read on its own (`i2 = n_slots, n = 1`) against zeros; row count stays `n_slots + 1` (`code`) | `a_region_order_read_compares_whole_and_not_per_row` (a device-free model of the region order: round 7's form reports all rows differing, the whole form 0; plus the source scan). RED: `L.n_slots` -> `nr` in the slot read |
| 2 SMALL | `llama-context.cpp`'s size check and `MUTATE=3`: skipping `set_measure_lent` lowers `need`, but `lend()` sets `L.lent` anyway, so the reserve allocates the healthy buffers and round 7's red arm ran normally | after the reserve, the actual buffer sum (`ggml_backend_sched_get_buffer_size` over `backend_ptrs`) is compared with the sized expectation (`reserve_matches`); MUTATE=3 perturbs the expectation (the skipped lent measurement) so the mismatch is the deterministic red case (`code`) | `the_lent_sizing_is_measured_unless_mutated`'s scan. RED: `reserve_matches(need, actual)` -> `true` |
| 3 BLOCKER | `set_tensor` builds a specialised extra for Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, MXFP4, Q8_0, IQ4_NL, Q4_K, Q5_K, Q6_K and converts BF16 to f16 (`ggml-opencl.cpp:12254-13912`); the USM bank (`ggml_cl_usm_set_tensor`) splits only IQ4_NL and planes the planar set, else raw memcpy. A region copy for the others writes GGUF bytes through a plain extra (the old crash); BF16 feeds f16 kernels raw BF16 and the bank check passes (it compares the wrong bytes with themselves) | `llama_loan_book::region_copyable(ggml_type)`: plain-no-extra, the planar set (IQ2_XXS, IQ2_XS, IQ2_S, Q2_0, IQ3_S) and IQ4_NL only; init's lend selection calls it for all three banks (`code`) | `a_lendable_type_is_one_the_region_copy_is_exact_for`: behavioral per type in both lists, plus the source scan that every excluded name is a `set_tensor` branch. RED: Q4_0 removed from the exclusion, and `region_copyable(Q4_0)` -> true |
| 4 SMALL | the round-7 margin compared `need` with the lend bytes; the headroom is set by the resident reserve and the non-lent buffers | `size_fits` is `need <= lend`; the 384 MiB margin is `headroom_fits(free)` on `ggml_backend_dev_memory` AFTER `drop_tail_buffer()` (the lend's free) and BEFORE `slot_loan_reserve(cap)`, and logged (`code`) | `a_size_check_leaves_a_peak_margin`, `headroom_fits` plus the context's scan. RED: `headroom_fits(free_min)` -> `true` |
| 5 nits | `iq4_nl_split_tensors` is not cleared by a weights buffer context reset; `lend_zero`'s first-use `clFinish` drained the queue once mid-refill (the first zero call followed the first write) | the comment names the staleness and its guard (the tensor's own extra); the refill calls `lend_zero` before `lend_write` in the loop, so the zero page's fill+finish precedes the copies (`code`) | `the_split_set_names_its_staleness`, `the_zero_page_drain_precedes_the_copies` (the loop's line order). RED: "stale entry" replaced; the two calls swapped |
| 6 BLOCKER (card) | `MUTATE=1`: round 7 skipped the last lent layer's down copy; the slot's split extra was then never built and `VERIFY=file`'s `get_slices` cast a plain extra -- a segfault inside the Intel driver, no refill or ERROR line (`measured-here`) | the switch flips one DATA byte of a lent row AFTER the copies, a device-side read-modify-write (`ggml_backend_opencl_lend_perturb`, the last region's byte at row 0 of the last lent layer's down); either reference catches it, exit 70 (`code`) | `the_red_switch_corrupts_one_lent_row_after_the_copies`: the refill's call, the old skip form gone, the backend's read/xor/write. RED: the perturb line removed |
| 7 BLOCKER (card) | `std::_Exit` tore the address space down under in-flight kernels/copies: a GPU page fault (`xe: Fault response: Unsuccessful -ENOENT/-EINVAL`) and a GuC reset of the compute AND copy engines, timed to the exit second, on both exit-70 arms (`measured-here`) | `poison_exit` drains every queue the process uses first -- `ggml_backend_opencl_drain` finishes the compute, adapt and bank-stream copy queues -- bounded by a 5 s watchdog thread that then `_Exit(70)` (`code`) | `the_poison_policy_exits_non_zero_with_the_cause` now registers a drain marker and requires it before the exit; `the_poison_drain_finishes_the_device_queues_first` scans the hook, the three `clFinish`es and the registration. RED: `drain();` removed |

**Card proof lines (round 8, Claude's window, the campaign's command):**

- item 1: the default refill line reads `verify on <e> s: 0 of 9360 rows differ
  from the lend bank` on the served model (0 of N), with no exit 70; `=file`
  keeps `0 of 9360 ... from the file`.
- item 2: `MUTATE=3` prints `LLAMA_SLOT_LOAN_MUTATE=3: the prompt's graph is sized
  WITHOUT the lent branch (the red case)`, then
  `the prompt's reserve allocated X GiB against the sized expectation Y GiB
  (LLAMA_SLOT_LOAN_MUTATE=3 skips the lent measurement); putting the slots back`,
  then the refill's `put back` with `0 of ... rows differ`, `refill+verify ok`,
  and the answer at the resident chunk.
- item 3: the load's loan line names what it lends; on the served model the 26
  lent layers stay (IQ4_NL down, planar up/gate); a model with a Q4_K/Q6_K/BF16
  bank layer prints `the slot loan lends nothing: <why>` instead of lending it.
- item 4: `llama_context::slot_loan_begin: after the lend's free: X GiB free of
  Y GiB on <dev>; the peak reserve is 0.375 GiB`, then the loan-out line with
  `size+lend+reserve took Z ms`; a refusal prints the free bytes and the reserve.
- item 5: no new line; the default verify line is unchanged.
- item 6: `MUTATE=1` prints `LLAMA_SLOT_LOAN_MUTATE=1: one data byte of the last
  lent layer's down row 0 corrupted after the copies (the red case)`, then
  `the refill did not verify: 1 of 9360 rows differ from the lend bank` (or the
  file), one `ERROR llama:` line, exit 70 -- and no segfault.
- item 7: on both exit-70 arms the kernel log has no `xe` fault and no GuC
  engine reset at the exit second (`dmesg -T | grep -E "xe|GuC"` before/after).

## 19. Round 11 (2026-10-29): the MUTATE=4 exit's remaining fault, the sizing's throw, the drain's scope

**RETRACTED (round 12, 2026-10-29).** This round's explanation of the
remaining fault -- "a free or a scheduler replace follows queued work, so the
exit's drain runs a command against released memory" -- is retracted; the
round-12 review found it not in the code (`code`): no adapt-queue command
targets a galloc buffer, the router-ids read is on the compute queue, and
OpenCL retains a `cl_mem` until its commands finish, so a release under queued
work cannot fault. What the card separated instead (`measured-here`, the B60,
round 11): `MUTATE=1` exits after the verify's read-back and memcmp --
hundreds of ms of idle -- and never faults; `MUTATE=4` exits microseconds after
the last GPU work and faults 3 of 3; the same arm with
`NEOReadDebugKeys=1 EnableDirectSubmission=0` exits 70 without a fault, 0 of 3.
The drains this round added stay: they are harmless, loan-path only, they keep
the queues empty under the exit and they do the `MUTATE=3`/`MUTATE=1` work.
§20 records the discriminator and the fix.

The round-10 card run (`measured-here`, b4bca38) left one fault: `MUTATE=4`
exited 70 with the right line and still showed the B60's
`xe: Fault response: Unsuccessful -ENOENT/-EINVAL` and a GuC reset of the
compute and copy engines at the exit second. The trace of the return path is
`code`: the lend relays the tables and frees the tail buffer (`drop_tail_buffer`,
drained since round 10); the prompt runs on the lent branch (the lend bank and
the cache's banks; no slot tensor, `llama-graph.cpp:2510-2521`); the return
either re-reserves the resident chunk (`slot_loan_reserve` -> `sched_reserve` ->
`sched_new`, which replaces the live scheduler and frees the galloc's buffers
after its own `synchronize`) or, on `MUTATE=4`, skips it, drains and poisons;
`poison_exit` drains and `_Exit`s. Two ordering gaps in that trace are fixed
here; both were checked in the code first.

| # | checked in the code | what changed | tested by (RED under its mutation) |
|---|---|---|---|
| 1 BLOCKER | `sched_reserve`'s `synchronize()` calls `ggml_backend_sched_synchronize` -> each backend's `synchronize`, which finishes the compute queue and `bstream.q` but **not** `astream.q` (`ggml-opencl.cpp:9062`). `sched_new` then replaces the scheduler and `ggml_gallocr_free` frees its buffers at once; `slot_loan_begin` calls `sched_new` itself before the sizing, and the exit path skips the return's re-reserve, so its `synchronize` never ran. A throw inside `sched_reserve` after `sched_new` leaves the galloc with buffers freed as well | `slot_loan_reserve` drains every queue (compute, adapt, bank-stream) **before** `sched_reserve`, and again in the catch; `slot_loan_begin` drains before its own `sched_new`; the return's failed/skipped path runs `ggml_backend_sched_synchronize(sched)` **and** the full drain before the poison. No replace or free follows queued work; `_Exit` then tears down an idle device (`code`) | `a_re_reserve_finishes_the_queues_before_sched_new` (text order: drain before `sched_reserve()`, drain in the catch), `a_poison_path_drains_before_any_free` (unchanged, still green) |
| 2 BLOCKER | `slot_loan_begin` calls `sched_new(max_nodes)` and then sizes via `graph_reserve(..., split_only=true)` -> `ggml_backend_sched_reserve_size` -> gallocr. A C++ throw there (a `bad_alloc`, a failed split) leaves `llama_context_slot_loan_begin` and reaches `Scope::begin`, which catches it and calls `end()`; no loan is out, so `end()` returns true, and the context stays on the replacement scheduler with no reserve while every cache's `owner_sched` points at the destroyed one | the resolution, `sched_new` and the sizing are in a try/catch; the catch logs and takes the refusals' own recovery: `slot_loan_reserve(resident)` and `rebind` each cache, then false (`code`) | `a_lent_sizing_throw_takes_the_resident_refusal` (text order: the catch, `sized = false`, the `!sized` block with the resident reserve before the rebind) |
| 3 SMALL (review item 2) | the cache destructor drained only when `b_tail` existed. After a lend `b_tail` is null (the loan freed it) while `bslots`, the USM bank and the lend bank are freed -- and the adapt/bank-stream queues can still read the USM banks; `drop_tail_buffer`'s drain had the same `b_tail != nullptr` key | both drain once whenever `lend_bytes_ != 0` (the loan armed), before clearing or freeing anything; with the loan off (0) nothing is added, so the loan-off form is unchanged (`code`) | `a_lent_cache_drains_before_every_free_in_the_destructor` (text order: the `lend_bytes_ != 0` drain precedes the tail free and the `bufs` walk) |
| 4 SMALL | `ggml_backend_buft_alloc_buffer_n_default` frees the buffers it built when a later one fails (`ggml-backend.cpp:94-158`), but the tensors it placed keep `buffer`/`data`/`extra` pointing at the released object; the refill's failure path then poisoned | the `b_tail == nullptr` branch clears `tail_slots` exactly as `drop_tail_buffer` does, before the poison line (`code`) | `a_failed_tail_allocation_leaves_no_dangling_tensor` (text order: the clears before `poison_exit`) |

The drain now prints its own line (`ggml_opencl: drain: N device(s), compute,
adapt and bank-stream queues finished`). Round 12 corrected two things about
it: the line was at INFO and arcint shows WARN and above, so the card never saw
it (now WARN), and a `MUTATE=4` log carries **two** drain lines with the
`ERROR llama:` line between them -- the return path's own drain
(`slot_loan_end`'s skipped path) and then `poison_exit`'s (the policy logs its
ERROR line first and drains after). A log with one drain line or with the two
on the same side of the ERROR line is not the round-12 form. Device-free, each red case was seen
failing under its one-line mutation (R36-R39 in `tests/test_loan_book.cpp`).

**What the card must prove.** `MUTATE=4` prints `the resident re-reserve at the
loan's return failed at 2048 tokens`, then the drain line, then one
`ERROR llama:` line and exits 70, and `dmesg -T | grep -E "xe|GuC"` shows no
fault and no reset at the exit second. If the fault survives the drain line, the
queues were idle and the fault is in the driver's teardown of live USM (the lend
bank) rather than in queued work; that is the next trace, not this round's.

## 20. Round 12 (2026-10-29): the exit's live rings, the retraction, the drain line's level

Round 11's explanation is retracted (§19's opening note, and the campaign
document). The round-12 review of the round's code (`code`) says the premise is
not in the code: no adapt-queue command targets a galloc buffer, the router-ids
read is on the compute queue (`ggml-opencl.cpp`'s `get_tensors`, the compute
`queue`), and OpenCL retains a `cl_mem` until its commands finish, so a release
under queued work cannot fault. Its alternative was measured on the card
(`measured-here`, round 11, the B60, Flash-Next IQ3_S, the unit's flags):

| arm | idle between the last GPU work and the exit | outcome |
|---|---|---|
| `MUTATE=1` | the refill verify's read-back (`lend_read`, one `clEnqueueReadBuffer` a region, `CL_TRUE`) plus the memcmp of 12,480 rows -- hundreds of ms | caught, exit 70, no xe fault |
| `MUTATE=4` | microseconds (the skipped re-reserve, then the drain) | exit 70 every run, xe fault at the exit second 3 of 3 (`-ENOENT`, GuC resets on bcs and ccs) |
| `MUTATE=4` + `NEOReadDebugKeys=1 EnableDirectSubmission=0` | as above | exit 70, no fault, 0 of 3 |

`MUTATE=1`'s natural idle, from the code path (`code`): its red case flips one
data byte after the refill's copies; the refill then verifies row by row --
`lend_read` is a blocking `clEnqueueReadBuffer` a region and
`lend_compare_rows` memcmp's the rows between reads
(`llama-expert-cache.cpp:1185-1190`) -- so the last GPU read-back is followed by
the remaining compare, the line's formatting and `poison_exit`'s drain before
the exit: hundreds of ms (`measured-here`, round 11). `MUTATE=4` has none of
that after its last GPU work.

The mechanism, `code` (the round-11 review's reading; the runtime's source is
not checked out in this tree and the machine has no network, so this is not a
read here): `_Exit` skips the Intel runtime's teardown, and its
direct-submission rings -- a hardware context an engine spins on a semaphore,
stopped by the runtime's own controller after a device idle period **or by its
destructor** -- are still live when `xe` closes the fd. `MUTATE=1`'s idle is
long enough; `MUTATE=4`'s is not.

**The fix.** `poison_exit` waits, after its drain and before `_Exit`, a bounded
idle: `LLAMA_SLOT_LOAN_EXIT_IDLE_MS`, default 2,000 ms, clamped to 4,000 ms (so
1 s of the 5 s watchdog is left for the drain). The wait is a device idle
period, not an OpenCL queue: `clFinish` says the queues are empty and nothing
about the rings. Direct submission stays on -- it is the runtime's fast submit
path for every unit; turning it off globally is not this defect's price.
No device was drained (no registered drain hook) means no ring: no wait.
The line is at WARN and on stderr, so the card's log shows the arm and its
value: `the exit waits N ms after the drain, for the runtime's direct-submission
rings to stop (LLAMA_SLOT_LOAN_EXIT_IDLE_MS, default 2000)`.

How the default was chosen (`measured-here` + `code`):
- the only exit path measured to pass is `MUTATE=1`, idle for the verify's
  read-back plus memcmp -- hundreds of ms on the card (the record's estimate
  for the 12,480-row read is 0.9-1.1 s). 2,000 ms is twice that upper bound,
  not the passing value itself.
- the runtime's own idle period is not readable here (see above), so the
  default is a margin over the measured passing idle, not the controller's
  constant; the knob exists so the window can test it.
- the watchdog bound is 5 s; its thread exits 70 at that bound whatever the
  drain does. So nothing bounds the drain but the watchdog: 2,000 ms of idle
  leaves the drain the watchdog's remaining time, and the clamp (4,000 ms)
  guarantees at least 1 s after any setting -- not a drain bound, a margin.
- "0", "off", "false" and "no" are the red (the round-11 form; round 13
  reads all cases of those words). A typo or a negative is the default, never
  a silent 0: a stray knob must not turn the red arm healthy.

**A cleaner call, named and not taken.** The runtime's own
`DirectSubmissionController::stopRingBuffers()` (compute-runtime
`shared/source/direct_submission/direct_submission_controller.cpp`, with
`checkPeriodicTimeout()` as the thread's loop) is the call that stops the rings
on demand. It is internal C++ in the runtime, not an OpenCL entry point and not
on any `cl_*` path, so a process cannot reach it without linking the runtime's
internals (`code`, not read here: the source is not checked out). The operator's
policy (one ERROR line, a drain, a non-zero exit) stays; the bounded idle is
what the public surface allows.

**The drain line.** `ggml_backend_opencl_drain`'s line is
`GGML_LOG_WARN` (round 11 had INFO; arcint shows WARN and above, so the card
never saw it -- `measured-here`, the round-11 window). At the units' default
verbosity an `INFO` line is not shown (`-v` is what shows it), which is why
round 11's drain was invisible there. The line still states the queues' read;
it does not claim the rings are stopped.

**The review's lows.** R36 also checks the order in the `!sized` block: the
resident reserve before the rebind (a rebind first binds `owner_sched` to the
scheduler the failed reserve replaces). The docs' `MUTATE=4` log form is two
drain lines with the `ERROR llama:` line between them (§19). R36-R39 are
text-order tests: each reads the pinned tree's source in one block, and R36's
order half reads the block's line positions.

| # | checked in the code | what changed | tested by (RED under its mutation) |
|---|---|---|---|
| 5 SMALL | `std::_Exit` runs no destructor, and the direct-submission rings are stopped by the runtime's controller after a device idle period or by its destructor (`code`, the review; `measured-here`: the discriminator table above) | `poison_exit` waits `exit_idle_ms(getenv("LLAMA_SLOT_LOAN_EXIT_IDLE_MS"))` after the drain, under the watchdog, logs the wait at WARN and on stderr (`code`) | `the_exit_waits_for_the_rings_to_stop` (the elapsed time, the WARN line, the order after the drain), `the_exit_idle_knob_is_bounded` |
| 6 SMALL | round 11's drain line at `GGML_LOG_INFO`; arcint's log shows WARN and up, and the card's round-11 window had no line (`measured-here`) | `GGML_LOG_WARN` for the drain line (`code`) | `the_drain_line_is_at_warn` (text order) |
| 7 SMALL (review) | the `!sized` block rebinds after the resident reserve; a rebind on the unreplaced scheduler binds `owner_sched` to the one the reserve destroys (`code`) | no code change: the order is checked | `a_lent_sizing_throw_takes_the_resident_refusal`'s order half (text order) |

The knob's parse is `exit_idle_ms` in `llama-loan-book.h`; `log_warn` is the
WARN route in `llama-expert-cache.cpp` beside `log_line`, so the unit's own log
callback sees it. Device-free, each red case was seen failing under its
one-line mutation (R40-R43, and R36's order half): the sleep removed,
`exit_idle_ms()` 0 for every value, the wait before the drain, the clamp
removed, `GGML_LOG_INFO`, the rebind moved above the reserve.

**What the card proved (round 12's window, `measured-here`, 2026-10-29).**
`MUTATE=4` three times with the default idle: exit 70 every run, the log order
drain / `ERROR llama:` / drain / `the exit waits 2000 ms`, and no `xe` line in
the `dmesg` diff -- **3 of 3**. `LLAMA_SLOT_LOAN_EXIT_IDLE_MS=0`: exit 70 and
the `xe` fault at the exit second (bcs engine reset, -ENOENT) -- **1 of 1**.
Idle 100 ms: exit 70, **no fault**, 1 of 1. The mechanism is settled: the
rings are live at `_Exit` for microseconds of idle and stopped by 100 ms, and
the 2,000 ms default is a margin over a passing value, not the value.
`NEOReadDebugKeys=1 PrintDebugSettings=1` printed nothing into the server log,
so the runtime's controller period is not measured here.

**Abort paths, not covered (one sentence, `code`):** `GGML_ASSERT` and
`CL_CHECK` abort the process directly, bypassing `poison_exit`'s drain and
idle, so an abort microseconds after GPU work would fault the rings the same
way. They are crash paths the unit restarts from; no change is made here.

**What nothing bounds.** The drain is not bounded by anything but the 5 s
watchdog: a hung driver holds the exit until the watchdog's thread fires
(round 13 writes its own line before `_Exit`). The idle is bounded by the
clamp and shares the same watchdog, so `exit_idle_ms` + the drain's time may
exceed 5 s only by the watchdog's own scheduling.

## 21. Round 13 (2026-10-30): the review's lows, the two gates

Round 12's card results are recorded in §20 (`measured-here`: idle 0 faults,
100 and 2,000 ms do not; 3 of 3 at the default). This round closes the review's
lows in the code and prepares the two gates the unit needs before it ships.

**The code, and what each low was (`code`).**

| # | checked in the code | what changed | tested by (RED under its one-line mutation) |
|---|---|---|---|
| 1 SMALL | the wait line is written *before* the sleep, so the log states what was asked, not what happened; and the watchdog's thread `_Exit(70)`s without a line, so a cut idle looked like a finished one (`code`, `poison_exit`) | after the sleep, a second WARN line with the measured elapsed: `the exit's idle finished after N ms (asked M ms, LLAMA_SLOT_LOAN_EXIT_IDLE_MS)` | `the_exit_reports_the_idle_it_actually_waited` (R44: the second line's emission removed) |
| 2 SMALL | the watchdog thread runs while the drain may be hung and the log route sits on the lock it holds; `snprintf`/the log callback are not async-signal-safe (`code`) | the thread writes ONE constant line with `::write(STDERR_FILENO, ...)` before `_Exit`: `the exit's watchdog fired after 5000 ms (the drain or the idle did not finish); _Exit(70)` | `the_watchdog_fires_with_one_async_safe_line` (R46: the `write` removed; a hung drain holds 5 s, so the line is the watchdog's) |
| 3 SMALL | `strcmp(e, "off")`: an `OFF` or `False` arm silently took the 2,000 ms default, so a red arm ran slower than the arm it was asked to be (`code`) | `exit_idle_off_word` compares `off`/`false`/`no` without case (portable fold, no POSIX `strncasecmp`); a typo or negative stays the default | `the_idle_knob_reads_the_off_words_in_any_case` (R45: the case-fold removed) |

**The docs' corrections.** Three sentences were wrong or incomplete and are
fixed in §20 and the campaign document: the old "leaves 3 s for the drain" read
as a drain bound -- nothing bounds the drain but the 5 s watchdog; the drain
line was invisible at the units' default verbosity (`INFO` needs `-v`, the line
is `WARN`); and `GGML_ASSERT` / `CL_CHECK` abort paths bypass the idle and would
fault the rings the same way (no change, stated).

**The KL gate (`code`, design; commands in the campaign doc).**
`llama-perplexity` built from the served series cannot take the loan: the lend
budget is the load-time model parameter (`slot_loan_bytes`, `LLAMA_SLOT_LOAN_MIB`
when unset) and the cache's `lend()`/`refill()` run only through
`llama_context_slot_loan_begin/end`; **no** call site exists under `tools/`,
`common/` or `examples/` -- the only caller is arcint's
`src/exec/backend_llama.cpp`. A perplexity "-loan on" arm would be identical to
its off arm and would prove nothing. Arcint's endpoint returns no logprobs
(`code`, `src/api/handlers.cpp` emits `"logprobs", nullptr`), so the served side
cannot take the KL either. The route is `contrib/llama-loan-kld`: it reads the
loan-off capture's exact tokens, decodes them with the loan out, and writes a
same-shape capture that `tools/capture_vs_capture.py` compares. A batch of
n_ctx tokens at `-b/-ub 8192` would cross the lend threshold
(8,192 > the 2,048 resident chunk), but only a caller that begins the loan
crosses it; the driver is that caller, and it refuses a context at or below
2,048 (R47).

The window is **1 x 8,192**, not 16 x 512 (the 512 window cannot cross the
threshold); `capture_vs_capture.py` compares a capture's window against the
reference's window 0, so the arm is one chunk. The bar is `CLAUDE.md`'s: mean KL
at most 0.03 nats worse than the twin (two same-binary loan-off runs), argmax
agreement at most 1 point lower. The driver's refusal of a run without a loan
(R48: begin before the decode, end after it, and every chunk must take one) is
what makes the arm exercise the third `MUL_MAT_ID` branch instead of passing on
an off arm.

**The depth gate (`code`, design; the prediction is corrected in §22/round 14).**
The unit runs 131,072; the loan was measured at 18,802. The KV is `n_ctx`-sized
at load and does not follow the prompt's depth; the compute buffers hold
`ubatch x n_kv_reserve` cells (`llama-context.cpp:2429-2454`), and the reserve
holds `ub_cap x n_kv_reserve` cells. The cut applies to **both arms** and
scales with `ub_cap` (`n_ubatch_eff = max(floor, min(ub_cap, ub_cap x
n_kv_reserve / cells))`): the off arm's `ub_cap` is its 2,048 ubatch, the on
arm's is the 8,192 chunk, so the on arm's ubatch is four times the off arm's at
every depth past 20,480 cells -- the ratio that decides the prediction, and the
reason the earlier "on at the off's rate, no faster" was backwards. The
measured 18,802-token arm used the full 8,192 (`8192 x 20480 / 18816 = 8,916`),
which is why its +18.0 % is a full-chunk gain; at ~100k it is 1,676 against 419,
so the gain's size is the open cell, not its sign. Peak: the on arm's compute
need at 1,676 is paid by the 6.20 GiB lend, predicted at or under the off arm's
peak; its GTT sits above the off arm's by the lend bank. The spill arm
(`--prefill-chunk 8192`, no loan) is dropped from the leg: round 12 measured the
same mechanism at 18k (275.2 t/s, 24,452 MiB, GTT 49,731 -- a thrash), and the
cell cut changes the ubatch but not the 8,192 reserve, so at ~100k it would
carry the same slots-plus-8,192-reserve peak the record already has.

**What the gates need from a Claude window.** The exact commands are in the
campaign document; the GPU work is Claude's. Everything in this round is
device-free and its red cases were seen failing under their one-line mutations
(R44-R48).

## 22. Round 14 (2026-10-30): a red case for the lent branch, the driver's host RAM, the served shape, the corrected depth prediction

A review of round 13 found five defects and one build gap. All are device-free
here; the card commands are in the campaign document. **Round 15 corrects row 3
(the red did not score: the override never reached check_tables' producer) and
row 2's host/device figures (the output rows are 4.07 GB in the lent graph and on
OpenCL's CPU output buffer); see §23.**

| # | checked in the code | what changed | tested by (RED under its one-line mutation) |
|---|---|---|---|
| 1 HIGH | the depth prediction was backwards: the cell cut (`n_ubatch_eff = max(floor, min(ub_cap, ub_cap x n_kv_reserve / cells))`, `llama-context.cpp:2429-2454`) applies to **both** arms and scales with `ub_cap`. The off arm's `ub_cap` is 2,048, the on arm's 8,192, so the on arm runs 4x the tokens a ubatch at every depth past 20,480 cells (`code`) | the prediction is rewritten here and in the campaign doc, with the ubatch table below: the on arm stays faster at depth; the gain's size is the open cell, not its sign | the leg (card): the sign cell is on > off; a sign at or below fails |
| 2 HIGH | `contrib/llama-loan-kld/loan_kld.cpp` held the round-13 buffers at Flash-Next's 248,320-token vocab: `:247` `n_ctx x n_vocab` floats (8.14 GB, zero-filled, half used), `:158` `n_outputs_max = n_batch` (llama's output rows: another 8.14 GB where `n_ctx - n_ctx/2` suffices), `:137` the row buffer `n_rows x nv` (2.03 GB) (`code`) | rows are written straight from `llama_get_logits(ctx)` before the loan's end (the end frees their buffer), the copy is gone, the row buffer is one row (0.47 MiB), `n_outputs_max = output_rows(n_ctx)` (4,096 rows, 4.07 GB) | `the_kld_logits_reservation_is_the_row_window` (R51: the assignment or the one-row buffer or the write-before-end order removed) |
| 3 MEDIUM-HIGH | no red case reached the branch the KL gate tests: every red arm exits 70 before scoring, and `MUTATE=1` + `VERIFY=off` is a FALSE red -- the corruption lands after the prefill (`code`) | `LLAMA_SLOT_LOAN_MUTATE=5` (`mutate_loan_bank_row`, applied in `lend()` to the host table before its non-blocking write): one lent layer's lend-bank row index off by one during the prefill; no verify compares a table row, so the run refills, returns and scores, and the WARN line names layer, expert and both rows | `the_lent_table_can_be_put_off_by_one` and `the_lent_mutation_is_applied_at_the_lend_and_names_itself` (R49) |
| 4 MEDIUM | the on arm did not run the served graph: at `-ub 2048` the lent prefill ran 4 ubatches of 2,048 under the 8,192 loan reserve, while the unit runs one ubatch of 8,192 (`code`) | the driver sets `llama_context_params.n_ubatch_reserve` directly (common has no option for it; the context is built from common's own loaders) and logs the shape: `-ub 8192 --ubatch-reserve 2048` is the served shape; a smaller ubatch under `--loan` is a WARN | `the_kld_on_arm_runs_the_served_ubatch_shape` (R53) |
| 5 LOW | a control arm was missing, the quantiser differed, a partial `--out` survived a refusal, and the cold first chunk was undocumented (`code`) | the control is the driver without `--loan`; `nearest_int` transcribed from `perplexity.cpp:72-77` (half-to-even; `std::lround` is half away from zero); a partial `--out` is unlinked on an exit-2 refusal (`out_file`); the driver logs that its first chunk's rate is cold (`params.warmup = false` was inert in the round-13 form -- `common_init_from_params` does not warm up) | `the_kld_quantiser_is_perplexitys_half_to_even` (R50), `the_kld_driver_unlinks_a_partial_capture_and_says_the_rate_is_cold` (R52) |
| 6 | the recipe builds arcint's deb with `LLAMA_BUILD_TOOLS OFF`, so neither tool of the series is built (`code`, `CMakeLists.txt:121`) | `contrib/llama-loan-kld/build-tools.sh` builds both from the recipe's patched llama tree, one cmake line with `-DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF -DGGML_OPENCL_EMBED_KERNELS=ON`, and `TOOL <name>: <sha256>` | the build (card host) |

### 22a. The depth prediction and its table (`code`)

`cells` is the batch's padded last position (`GGML_PAD(p_end, 256)`);
`ub_cap = min(n_ubatch, n_ubatch_reserve)`; the budget is `ub_cap x n_kv_reserve`
(20,480 cells at the unit's `--llama-reserve-cells`); the floor is
`n_seq_max x (n_rs_seq + 1) + 1 = 6` here and binds only past ~6.99M cells.

| prompt tokens | cells | off arm (ub_cap 2,048): ubatch | on arm (ub_cap 8,192): ubatch |
|---|---|---|---|
| 20,480 | 20,480 | 2,048 | 8,192 |
| 40,960 | 40,960 | 1,024 | 4,096 |
| ~100,000 | 100,096 | **419** | **1,676** |
| 131,072 | 131,072 | **320** | **1,280** |

The ratio is 4:1 at every row; it persists past 20,480 cells because both sides
divide the same `cells`. The measured 18,802-token arm is the last full-chunk
arm (`8192 x 20480 / 18816 = 8,916` clamps to 8,192), which is why its +18.0 %
is a full-chunk gain and not a depth gain.

Prediction the leg can fail (`code` + arithmetic): at ~100k the on arm's prefill
is **faster** than the off arm's (1,676 against 419 tokens a ubatch, the same
mechanism the 18k arm measured); the size of the gain is **open** -- the on arm
is at 20 % of its chunk, and at depth the attention term grows for both arms.
The peak cell: the on arm's peak at or under the off arm's, its GTT above the
off arm's by the lend bank. A sign at or below fails the cell; a magnitude is
recorded, not barred.

### 22b. The driver's host memory, and the KL gate's forecast (round 15 corrected)

Round-13 driver, this vocab: 8.14 GB logits copy, 2.03 GB row buffer, 8.14 GB
output reservation. Round 14 fixed the copy and the row buffer but kept the
capture's 4,096-row output reservation: 4.07 GB in the context's output buffer,
and the loan's sizing took `n_outputs = min(measure_ub, n_outputs_max)` = 4,096,
so the same 4.07 GB sat in the lent graph's sched compute buffer as well. Round
15 caps `n_outputs_max` at `--rows-per-decode` (2,048 by default) and scores the
served 8,192-token ubatch in disjoint row windows under one loan: the rows are
positions `first..first+2047` and `first+2048..n_ctx-2` (2,048 + 2,047 = 4,095 at
n_ctx 8,192), each window a full 8,192-token graph from an empty KV.

`width x n_vocab x 4` at Flash-Next's 248,320-token vocab (`code` + arithmetic):
2,048 rows = 2.03 GB; 1,024 rows = 1.02 GB; 512 rows = 0.51 GB; 256 rows =
0.25 GB. Both the context's output buffer (a plain CPU buffer on OpenCL -- the
device's `get_host_buffer_type` is NULL, `ggml-opencl.cpp:15418`) and the lent
graph's logits tensor hold that many rows.

| arm | anon RSS | host bank (pinned) | lend bank (pinned) | stream ring | pinned total (GTT) |
|---|---|---|---|---|---|
| off: `llama-perplexity -b 4096 -ub 2048` | ~8.2 GB (`logits` copy 4.07 GB touched + `log_probs` 4.07 GB zero-filled, the reserve is 8.14 GB virtual) `(inferred, arithmetic)` | ~37.2 GiB `(inferred: round 12's 43,445 MiB pinned minus the 6.20 GiB lend bank and KV/compute)` | 0 (no loan) | 751 MiB (VRAM, not host; `measured-here`, flash-next campaign) | ~37.2 GiB of 48 |
| on: the driver, `--loan`, 2,048 rows a window | ~2.0 GB (2.03 GB output window + one 0.47 MiB row, no copy, no `log_probs`) `(inferred, arithmetic)` | ~37.2 GiB `(inferred)` | 6.20 GiB (`measured-here`, round 12) | 751 MiB (VRAM) | ~43.4 GiB of 48 (`measured-here`: round 12's loan arm pinned 43,445 MiB at n_ctx 131,072; this gate's n_ctx 8,192 is lower) |
| on, `--rows-per-decode 512` | ~0.5 GB `(inferred, arithmetic)` | as above | 6.20 GiB | 751 MiB | ~43.4 GiB |

Against the card host: the container limit 96 GiB sees the arms' anon (8.2 GB
off, ~2.0 GB on at 2,048 rows, ~0.5 GB at 512) far below; the pinned banks are
**not charged to the container's cgroup** (`measured-here`, stage 9:
MemAvailable stayed 94.3-94.4 GiB while a process held its bank). The binding
limit is the TTM pinned cap, 48 GiB: ~37.2 off, ~43.4 on, ~4.6 GiB of slack on
the loan arm; the host RAM 125.7 GiB is not what binds.

Device side, not a host charge: the lent graph's logits rows are the plan's
addend over the unit's one-output prefill shape, which round 12's loan arm
measured as fitting the 6.20 GiB lend: `need = need(8192 tokens, one row) +
(width - 1) x n_vocab x 4`. The plan allows `need + 0.375 GiB <= 6.20 GiB`, so
the room for the base need is 5.825 GiB; at 2,048 rows (2.03 GB = 1.89 GiB) the
base need must be at or below **3.93 GiB**, at 1,024 rows (1.02 GB = 0.95 GiB)
below **4.88 GiB**, at 512 rows (0.51 GB = 0.47 GiB) below **5.35 GiB**. The
base need at 8,192 tokens is not measured here (`inferred` from round 12's
accepted loan); so whether the two-window arm fits is decided on the card by
`size_fits`, and the log line names need, lend and reserve. A refusal is
recorded as a refusal; the driver's `--rows-per-decode` then halves the addend
(the same rows, more windows, each still the served 8,192-token graph), and the
choice between fewer slots (a larger pinned bank), a smaller chunk (no longer
the served shape), or the narrower window is the operator's. No number here is a
claim that the arm fits.

Round 15's correction of this section: the old `<=0.5 GB` on arm and the old
`~4.07 GB` output addend were both wrong (the 4,096-row window's output buffer
on OpenCL is anon, and the same rows are in the sizing's need).

### 22c. What each KL arm runs (`code`)

- the off twins: `llama-perplexity -c 8192 -b 4096 -ub 2048`, no loan; four
  ubatches of 2,048 over two batches, `n_outputs_max` = 4,096 (the tool's
  `n_batch`, halved from 8,192 for the 248,320-token vocab). The cap cannot take
  `-b 8192` without 8.14 GB of output reservation, so the split is halved; the
  bar is answer-level, and the capture's rows are unchanged.
- the control: the driver WITHOUT `--loan` at `-b 8192 -ub 2048
  --ubatch-reserve 2048` -- the same four ubatches, the same `first`; its
  scorer and quantiser are perplexity's, so its rows must sit at the twin floor.
- the red: the driver `--loan` with `LLAMA_SLOT_LOAN_MUTATE=5`, the served shape;
  default verify (the mutation is not a slot-byte corruption), one chunk, it
  scores and must fail the bar.
- the on arm: the driver `--loan`, `-b 8192 -ub 8192 --ubatch-reserve 2048
  --rows-per-decode 2048` -- two decodes, each the full 8,192-token ubatch under
  the resident 2,048 and the loan's 8,192 reserve, under one loan; the first
  scores rows 4096..6143, the second 6144..8190. The rows equal
  `llama-perplexity`'s: each decode replays the chunk from an empty KV and a
  causal row at position p depends only on tokens 0..p.

### 22d. The two tools on the card host (`code`)

The recipe builds arcint's deb with `LLAMA_BUILD_TOOLS OFF`
(`CMakeLists.txt:121`) and does not leave a llama tree behind: build-deb.sh
works in `mktemp -d` with `trap "rm -rf $work" EXIT` (`:50-51`).
`contrib/llama-loan-kld/build-tools.sh` therefore has two ways in: `--fresh
<work-dir>` assembles the tree exactly as the recipe does -- the pin tarball
(the recipe's `LLAMA_SRC_TARBALL` override when set or at
`$HOME/src/llama-<pin7>.tar.gz`, else the pinned URL), build-deb.sh's
`LLAMA_TARBALL_SHA256`, then `git apply --whitespace=nowarn` of
`contrib/llama.cpp/patches/*.patch` in order -- or an existing patched tree as
the argument. Either way it copies the driver to `<tree>/tools/loan-kld`,
registers `add_subdirectory(loan-kld)` once, configures `<tree>/build-tools`
with

    -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF
    -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF

builds `llama-perplexity` and `llama-loan-kld`, and prints `TOOL <name>:
<sha256>` for each. The binaries are not standalone: both resolve their shared
libs by RPATH into `<tree>/build-tools/bin` (libllama, libggml, the OpenCL
backend), so they are run there and not copied alone. `--fresh` needs an empty
work dir: an existing tree or a stale extraction is refused. The tools' libllama is the same patched series the deb
carries, and the kernels are embedded from that tree (the recipe's own OpenCL
settings, `CMakeLists.txt:115-117`). Verified here (`code`): a fresh tree built
both targets, `TOOL llama-perplexity` and `TOOL llama-loan-kld` printed, and the
driver built without warnings after the comment-continuation fix.

### 22e. Red cases R49-R53

All five were seen failing device-free (`measured-here` in the sense of a run,
not a card number): `tests/test_loan_book.cpp` built against the patched series
runs 61 cases green, and under each one-line mutation exactly one case fails --
R49's rule (the shift removed) and its call site (the `lend()` block removed),
R50 (`std::lround` instead of `nearest_int`), R51 (the `n_outputs_max`
assignment removed), R52 (the `std::remove` of the partial file removed), R53
(`served_shape` returning true). The recipe's `ctest -L unit` run is the same
suite's gate on the package build.

## 23. Round 15 (2026-10-30): the scorable red, the served shape's row windows, the Q8_0 loan row, the false comments

A review of round 14's code (`code`) found five items. All fixes are device-free
here; the card commands are in the campaign document. Every row names its
evidence class; the round-14 review's refusal observation is `measured-here`
(that run), the arithmetic over it is `(inferred)`.

| # | checked in the code | what changed | tested by (each seen RED before the fix) |
|---|---|---|---|
| 1 HIGH | `MUTATE=5` shifted one row in `tab_host` only; `check_tables()` rebuilds `want` with `fill_tables` from `L.loan_of`, so it saw the difference, `lend_undo()` ran, `slot_loan_begin` refused and the driver exited 2 un-lent (`code`; the review ran it) | the override is a layer member, `lend_row_mutate`, consulted in `fill_tables` while lent: the upload, the check, the card after `flush_tables` and the refill's restore build one image. `lend()` sets it before re-uploading and rotates EVERY ex-slot expert's lend-bank row in that layer; the WARN line names the layer and the count, not one expert. `refill()` clears it before `slot_of` is rebuilt, so the post-loan tables are healthy (`code`) | `the_lent_override_reaches_every_table_producer` (R54): the upload and check producers agree (0 differ), all four ex-slot experts moved, the old one-host-row form differs in every row, the refill's cleared rows are -1, and the source lines in `fill_tables`/the tables loop |
| 2 MEDIUM | the loan's sizing takes `n_outputs = min(measure_ub, n_outputs_max)` (`llama-context.cpp:1068`); the round-14 driver's 4,096 rows put ~4.07 GB in the lent graph's sched compute buffer, more than the 6.20 GiB lend minus the 0.375 GiB reserve. The same rows are the context's output buffer, and OpenCL's `get_host_buffer_type` is NULL (`ggml-opencl.cpp:15418`), so it is a plain CPU buffer: ~4.07 GB anon (`code` + `(inferred)`) | the driver caps `n_outputs_max` at `--rows-per-decode` (default 2,048) and scores the served 8,192-token ubatch in disjoint row windows under ONE loan: 2,048 rows at positions `first..first+2047`, 2,047 at `first+2048..n_ctx-2`; each window is the full 8,192-token graph from an empty KV, and a causal row at p depends only on its prefix, so the rows equal `llama-perplexity`'s (`code`) | `the_kld_scores_the_served_ubatch_in_row_windows` (R56): the two-window loop, the per-window KV clear, the width helpers (2,048 + 2,047 = 4,095), the read before the end |
| 3 LOW | init's `tab_loan` view was `tab_row + 3 + base` with base 3 for a Q8_0 down: row +6 of a six-row layer, while `fill_tables` and the mutation write row +5; a model whose last cached layer has a Q8_0 down asserts at load, loan on or off (`code`) | the view is the layer's last row: `tab_row + 3 + (tab_slot_dn != nullptr ? 2 : 0)` (`code`) | `the_loan_row_is_the_layers_last_row` (R55): the init view and fill's offset name the same row |
| 4 LOW | `loan_kld.cpp`'s comment said the end "frees the buffer the logits live in" -- false, `--ubatch-reserve` makes the resident re-reserve keep a buffer the logits are read from; the comment at the flag's struct called it optional (`0: the ubatch`) while a missing value exits 2; `build-tools.sh` did not say the tools resolve their libs by RPATH into `build-tools/bin`, nor that `--fresh` needs an empty dir (`code`) | the comment states the read-before-end and the required reserve; the parser says `--ubatch-reserve N is required`; the script's header names the RPATH (do not copy a binary alone) and the empty work dir, and the script refuses a non-empty `--fresh` dir (`code`) | `the_kld_comments_state_what_the_end_and_the_reserve_do` (R57) |
| 5 | the public-repo rule (`CLAUDE.md`): four paragraphs quoted the brief's data rules with client names | deleted from this document, with the round 14 note that had replaced one host name; the rules are not project documentation and are not written into any file again (`code`) | the document's scan for the names/rules |

The card proof lines (Claude's window, the campaign document's commands):

- the red: `lend ... table check: 0 of R rows differ`, the WARN
  `LLAMA_SLOT_LOAN_MUTATE=5: layer N: K ex-slot experts read the lend bank's rows
  one entry over during the prefill`, `refill ... verify 0 of N rows differ`,
  `loan returned ... refill+verify ok`, `1 chunks, 1 loans taken, 0 refused`,
  and a KL above the twin's floor by more than 0.03 nats.
- the on arm: two windows, `2,048 scored rows in 2 row windows of 2048
  (n_outputs_max 2048)`, one `loan out`, both windows' decodes, the refill and
  the return, `1 chunks, 1 loans taken, 0 refused`; its KL at the twin floor.
- the Q8_0 item is device-free only here (`code`); a model with a Q8_0 down in
  its last cached layer is not on the card, and the fix is the row arithmetic.

The `--rows-per-decode` argument is the operator's lever when the two-window arm
is refused by the plan (§22b): 1,024 or 512 halves or quarters the output addend
and the anon budget, with the same rows and the same served graph shape.

Device-free, `tests/test_loan_book.cpp` runs 65 cases green (R54-R57 are the
four new ones) and each new case was seen failing before its fix; the driver's
source was syntax-checked against the patched tree. The card commands and their
expected lines are in the campaign document.

## 24. Round 16 (2026-10-30): the gates scored, the peak's parts and the rule's scope, the unit

The two gates' card results are in the campaign document (`measured-here`).
This section is the analysis the numbers ask for: what the loan arm's peak is
made of (`code` + the measured tables), how it moves at the unit's 131,072, what
the sizing rule does and does not bound, and the size rule proposed to the
operator with its price. Every disposition carries its class.

### 24a. The loan arm's peak, part by part (`code` + arithmetic over measured figures)

The measured depth peak is 24,438 MiB of the card's 24,480 (`measured-here`,
103,638 tokens). The parts, and how each moves at 131,072:

| part | what it is (`code`) | size | at 131,072 |
|---|---|---|---|
| KV | allocated for `n_ctx` at load (`llama-context.cpp`, the KV cache's size) | 2.0 GiB q8_0 (`measured-here`, flash-next record) | **fixed**: n_ctx-sized at load, does not follow the prompt |
| slots that stay | the cache's buffer minus the lent tail buffer's bytes (`lend_bytes_`, `llama-expert-cache.cpp`; only the tail is freed) | 11,600 MiB budget minus 6.20 GiB lent, ~5.4 GiB `(inferred, arithmetic)` | fixed by the lend, not by the prompt |
| the prompt's compute reserve (`need`) | the sum of the scheduler's buffer sizes for the graph a loan would build: `graph_reserve(measure_ub = the loan chunk, 1, n_outputs = min(chunk, n_outputs_max))` under the KV reserve cap (`llama-context.cpp` `slot_loan_begin`, patch 0052's `llama_graph_reserve` scope) | the dominant term is the QSA selection mask and head scores, `10.7 bytes x cell x token + 0.146 MiB a token` (`measured-here`, the flash-next 8-point fit; the tensors `code`, `qwen4exp.cpp` `build_qsa_sel`), ~3.0 GiB at 8,192 tokens x 20,480 cells, plus the rest of the graph | **exactly at the reserve's worst case, no growth**: the cell cut keeps `ubatch x cells <= ub_cap x n_kv_reserve`, and at 131,072 it is equality -- 1,280 x 131,072 = 8,192 x 20,480 = 167.8M cell-tokens -- so the mask term equals what the reserve sized, and the tokens-only terms shrink (1,280 against 8,192) |
| the output rows | `n_outputs = min(chunk, n_outputs_max)`; the unit's `n_outputs_max` is `(1 + n_spec) x lanes` = 4 (`code`, `backend_llama.cpp:664-665`) | 4 rows x 248,320 x 4 = ~3.8 MiB `(inferred)` | unchanged (the 2.08 GB window is the KL gate's driver at n_ctx 8,192, not the unit) |
| the stream ring and gather mirror | patch 0032's ring, pages made resident at the first prefill | 751 + 187.7 MiB (`measured-here`, flash-next record) | fixed, pages already resident after the first request |
| the MTP draft context | its own reserve beside the main context; the loan does not touch it (`code`: only expert caches are lent) | 770 MiB at 131,072 (`measured-here`, the same record) | fixed |
| the refill's transient | at the return the resident re-reserve runs **before** the refill allocates the tail buffer (`code`, `slot_loan_end`'s order), so the tail is allocated after the prompt's chunk reserve is freed | the return's peak is the all-slots shape at the resident chunk | unchanged |

So nothing above grows at 131,072: the KV is already n_ctx-sized, the slots that
stay are the lend's complement, the compute reserve is allocated at the loan's
chunk (the cell cut never lets the actual graph exceed it), and the transients
are the resident shape. The peak at 131,072 is therefore expected in the same
class as the measured one, not larger `(inferred from the parts)`.

**The measured delta is not explained by the parts, and that is recorded rather
than smoothed over.** The off arm peaked 23,566 and the on arm 24,438: +872 MiB,
while the rule guarantees the prompt's reserve is at least 384 MiB below the
lend. The two arms do not share their no-loan baseline: the off arm runs the
resident chunk (2,048) and the loan arm reserves the 8,192-token chunk, so the
no-loan peak at the 8,192 chunk is not the off arm's figure, and round 12's
static chunk-8,192 arm spilled instead of showing it (275.2 t/s, peak 24,452,
GTT 49,731 -- a thrash, not a baseline). The run's `loan out` line carries the
plan's `need` and `reserve_matches` prints the reserve's actual bytes; the
reconciliation by parts needs those two figures and is owed (no card here).

### 24b. What the rule bounds, and the size rule proposed (`code` + `measured-here`)

`size_fits(need, lend)` is `lend >= kPeakReserve && need <= lend - kPeakReserve`
(`contrib/llama.cpp` patch 0068, `src/llama-loan-book.h`; `kPeakReserve` = 384
MiB). It compares **the prompt's graph sum with the freed tail bytes**. It does
not read the card's total and cannot: OpenCL's device memory query returns
`total - 1 GiB` (a constant; round 8, kept in round 9's retraction, `code`).
So the rule bounds **the delta at the loan moment only**: the reserve the prompt
takes must not exceed the bytes the lend freed, minus 384 MiB. The process's
absolute peak is not in the rule at all -- the KV, the staying slots, the ring,
the draft context and the no-loan peak at the same chunk are outside it, and
the 384 MiB is a margin over the *lent* bytes, not a margin under the card's
total.

Measured here: the loan arm at depth peaked **42 MiB under the card's total**
(24,438 of 24,480, `measured-here`). That is inside the band where the record's
eviction cases live: 147 MiB of free VRAM and xe moved ~3.5 GiB to host
(`measured-here`, the layer-split campaign); GTT 15.4 GB at a 55k needle
(`measured-here`, dflash); and the non-loan path's prefill needs about 3 GiB
more than the post-load free because its compute buffer grows with the context
fill (988 MiB at 32k, 1,631 at 64k, 3,048 at 131k at chunk 2,048,
`measured-here`). The loan path takes its reserve up front, so that particular
growth does not apply; what still allocates after it: the ring's pages on first
touch, the refill's tail re-allocation at the return (6.20 GiB freed and
re-allocated), the first decode's lazily grown buffers (1.4 MiB,
`measured-here`), the MTP snapshots and the driver's own BOs. Under 42 MiB of
free pages, any of them makes xe evict a resident buffer to GTT, and an evicted
compute or slot buffer is the 0.05 t/s / 275.2 t/s class (`measured-here`,
recorded). **So the loan's peak at 131,072 can trigger xe eviction.** The depth
arm measured did not (the prefill reached 340.5 t/s, the GTT growth is the lend
bank plus the host bank), but it sits at a margin the record has seen bite.

**Decided (operator, 2026-10-30, "Margin ok"): the rule is
`lend - need >= 384 MiB + kCardMargin`, with `kCardMargin` = 1 GiB, and the way
to reach it is (a), more lent layers at the unchanged 8,192 chunk.** A fixed
margin under the card's total is acceptable; 768 MiB is the floor the record's
eviction cases suggest, and 1 GiB is the operator's size. The rule is `code`
(`llama-loan-book.h`: `kCardMargin`, `size_fits`, `extend_lend_layers`); the
lend grows one candidate tail layer at a time and stops at the layer where the
margin holds, bounded by the tail (`llama-expert-cache.cpp`
`extend_lend`, called from `llama-context.cpp`'s sizing before the refusal). A
lend that cannot reach it is refused into the resident path with one WARN naming
`need`, the lend and the margin. The ways that were priced:

- **(a) more lent layers, chunk unchanged.** One more layer at the measured
  238 MiB a layer (6.20 GiB over 26 layers) buys `lend - need` +238 MiB, keeps
  the 8,192 chunk and the measured +56 % depth rate, and costs ~0.02 s of refill
  at 0.539 s / 6.20 GiB (~11.5 GiB/s measured here). Decode is unaffected: the
  slots are refilled before it. This is the cheap side; it needs the 27th layer
  to be lendable (`code`: `lendable` is computed per layer over the region-
  copyable types) and the slot budget's tail to hold it. The unit's cache is
  11,600 MiB, the measured arm's own, so the tail's size should still be read
  from the unit's own `lend` line, not this arithmetic.
- **(b) chunk smaller, lend unchanged.** With 6.20 GiB lent, `need <= 6.20 -
  0.384 - 1.0 = 4.82 GiB`. `need` is dominated by ~0.355 MiB a token at the
  20,480-cell cap plus the graph's other terms, but its measured value is the
  `loan out` line, not this arithmetic; the largest chunk whose `need` fits is
  read there. The price is the rate: the static arms measured 4,096 at +20.5 %
  and 8,192 at +29 % (`measured-here`), and the depth arm's 8,192 is the +56 %
  figure, so a chunk cut below 8,192 lands between the two -- the exact figure
  needs the measured `need`.
- **(c) fewer lent bytes** is (b)'s shadow: the lend shrinks, so the chunk must
  shrink too; the peak falls only by the chunk's smaller need while the lend
  takes less of the slots out, and decode still gets everything back. It is the
  same trade with less margin and no separate price.

The operator chose (a) on 2026-10-30; (b) and (c) stay recorded as the
unchosen price. The unit's flags below are the measured set and stay as they are:
the rule grows the lend inside the same ask, so no flag changes.

### 24c. The unit's flags (`code`)

The Flash-Next unit adds exactly three flags, the measured set:

    --prefill-chunk 8192 --llama-slot-loan 6400 --llama-loan-chunk 8192

- `--prefill-chunk 8192`: `n_batch` and (with the expert cache) `n_ubatch` are
  the chunk (`backend_llama.cpp:631,636,640`); with the loan armed the *resident*
  reserve stays 2,048 (`slot_loan::floor_chunk`, `kResidentChunk`), so a short
  follow-up pays neither a second reserve nor a refill.
- `--llama-slot-loan 6400`: the ask, in MiB (`slot_loan::lend_param`); the cache
  lends what the tail's lendable layers give and logs it. The measured arm's ask
  freed 6.20 GiB over 26 layers (il 22-47); the unit's cache is 11,600 MiB, the
  same as that arm, so the unit's `lend` line should read 26 layers plus the
  margin rule's growth; the line is the number to read before the first window. Round 17: the ask is the floor
  of a fixed-margin lend -- `slot_loan_begin` grows it a tail layer at a time
  (a 27th layer, ~+238 MiB) until `lend - need >= 384 MiB + 1 GiB`, and the ask
  flag itself is unchanged.
- `--llama-loan-chunk 8192`: the context's `n_ubatch_reserve` and the chunk a
  loan asks for (`backend_llama.cpp:686-687`; the request's chunk, capped at the
  prompt's remaining tokens), the reserve the loan takes up front.

Nothing else of the unit's flags changes: the rule is in the lending patch, not
in a flag. What is measured and what is open is the campaign document's round-17
summary: the fixed margin decided and implemented (the growth stops at the layer
the margin needs, R58-R60 red), the KL gate at the bar with a thin 0.007-nat red
margin, the depth leg at +56 % with the old peak clause failed and now closed as
accepted with a fixed margin, and the owed card window that reads `need`, the
lend line and the margin at ~100k tokens with the six answers.

### 24d. The integration (`code`)

The loan branch's history is not published: it once held text that must not be
published outside this repository. The integration is therefore **one commit on
branch `loan-integration`, above the shared tip** (`qfndev`, round 17 rebases
the one commit onto it; the earlier base was `6e463b3`), carrying the final tree
only -- patch 0068, the two tools, the tests and the docs -- so no earlier commit
message or file version of the loan branch reaches a public branch. The tree was
scanned for host names, addresses, user names and client terms before the
commit. `~/bin/arcint-recipe.sh` was run on the squashed branch: the package
build passed, the unit tests passed and the `.deb` was produced; the exact
`RECIPE` line is quoted in the handover and in the recipe's log.

## 25. Round 18 (2026-10-30): the growth path's review (`code`)

A review of round 17's code found the growth path broken; the fixes below are
`code`, device-free on the host, and each red case was seen failing under its
one-line mutation. The card arms are owed (the operator's announced card
window; the campaign's round-18 section carries the commands).

- **A grown layer's bank from the file** (`code`, round-17 items 1 and 3): the
  growth filled the fresh USM lend bank with `get_slices` -> `set_slices`, and
  the OpenCL USM branch of `set_tensor_slices` does `bctx->usm_split.at(tensor)`
  for IQ4_NL; `usm_split` is populated only by the whole-tensor set
  (`ggml_cl_usm_set_tensor`). So it threw on every layer of the served file (all
  `ffn_down_exps` are IQ4_NL). It is now filled from the file with one
  whole-tensor `ggml_backend_tensor_set` a region, the path init uses for the
  ask's pick (`read_expert` rows into a host image, then the whole set). The
  bank's provenance is then the file. Cost: one GGUF read of the grown layer's
  slot rows, ~238 MiB a layer at the unit's 11,600 MiB budget, paid once per
  process; the growth line prints the bytes read and its seconds.
- **A throw takes the refusal path** (`code`, round-17 item 1): `extend_lend`
  was called outside the sizing's `try`, after `sched_new` replaced the
  scheduler, so a throw reached `Scope::begin` on an unreserved scheduler with a
  dangling `owner_sched`, and the growth's host banks leaked. The call is now
  inside the `try`; `extend_lend` stages every bank and frees them on a failure,
  so the cache is unchanged when the catch below takes the `!sized` block -- the
  resident re-reserve and the rebind.
- **No free before publish** (`code`, round-17 item 2): `extend_lend` freed the
  grown layer's spare slot buffer before the lend was published, leaving
  `L.slot[k]` on a released `cl_mem`; a lend whose check failed then undid and
  the request resumed on freed memory. The free is removed; `drop_tail_buffer`
  frees `spare_buf` for every lend-set layer after the drain and after nulling
  the tensors, and the refill allocates it back.
- **The failed spare allocation** (`code`, round-17 item 4): init's failure path
  freed `spare_buf[j]` without nulling them, so the destructor freed them again.
  Each freed entry is nulled.
- **The host-bank figure** (`code`, round-17 item 5): `n_bank_bytes` now adds a
  grown bank's bytes, so the init info line and the snapshot do not understate
  the host bank.
- **The unit** (`code`): the production unit serves the IQ3_S file with
  `--llama-expert-cache 11600`; every place claiming IQ3_XXS or 12,000 MiB is
  corrected (`contrib/systemd/arcint-flashnext.service`, the campaign's round-17
  and owed-window text, §24b/§24c). The card commands use `fn_sweep`'s defaults
  (the unit's flags), with no `--set` of the cache.

Red cases (`device-free`, `tests/test_loan_book.cpp`; each mutation is one line
and was seen failing the named case):

| red | mutation | case |
|---|---|---|
| R61 | the whole set replaced by `set_slices` (or the `n_bank_bytes` line dropped) | `a_grown_lend_bank_is_filled_by_the_whole_tensor_path` |
| R62 | the staged-bank free removed from `extend_lend`'s catch | `a_growth_throw_takes_the_resident_refusal` |
| R63 | the `spare_buf` free put back into `extend_lend` | `a_grown_layer_is_not_freed_before_the_lend_is_published` |
| R64 | `spare_buf[j] = nullptr;` removed from init's failure path | `a_failed_spare_allocation_nulls_every_freed_buffer` |

## 26. Round 19 (2026-10-30): the card's own bound, the failed growth's tensors (`code`)

A review of round 18 (`code`, round 17's rule included) found two things that
matter: the rule as implemented held no margin to the **card** (it compared
`need` against the lent bytes only), and a failed growth left the grown layers'
lend tensors naming freed memory. Round 19 fixes both on the host; the card arms
are owed. Every disposition below carries its class.

### 26a. What the rule bounded, and what round 19 adds (`code` + `measured-here`)

Round 17's `size_fits(need, lend)` is `lend >= kLendMargin && need <= lend -
kLendMargin` (`code`, `llama-loan-book.h`): it compares the prompt's graph sum
with the freed tail bytes. At the 6,400 MiB ask and the 8,192 chunk that rule is
satisfied on the recorded figures -- `lend - need` = 1.65 GiB against the 1.375
GiB margin (`arithmetic` over `measured-here` figures: 6.20 GiB over 26 layers,
4.55 GiB need) -- so round 17 grew nothing and the served shape stayed round
16's 26 layers. The measured depth peak then sat 42 MiB under the card's total
(24,438 of 24,480, `measured-here`), and its parts (§24a: KV, the staying slots,
the ring, the draft context, the output rows) are allocations **outside** `need`.

Round 19 implements the operator's "Margin ok" where it belongs, against the
card:

    resident_after_lend + need <= card_total - kCardMargin

- **Round 20 replaces this figure** (see §27): the card rule now reads the
  process's own residency from its xe render fds' `/proc/self/fdinfo`
  (`drm-resident-vram0`), which counts the ring, the backend's grow-only
  preallocations and the runtime's surfaces. The round-19 paragraph below is
  kept as the record of what it then summed and what it did not. The
  `cparams.ctx_other` branch was dead on the target -- arcint sets it on the
  MTP draft's params, never on the target context for Flash-Next -- and round 20
  removes it; the draft context's VRAM is the same process's and is in the
  fdinfo figure.
- round 19's `resident_after_lend` = the sum of `ggml_backend_buffer_get_size`
  over every backend buffer this context and the model hold on that card, minus
  the bytes the lend frees there (`code`). Summed: the model's weights
  (`model.memory_breakdown()`), the KV memory's buffers
  (`memory->memory_breakdown()`), the expert cache's device bytes --
  `bslots` + the tail buffer + candidate layers' spare buffers + the gather
  mirror's reserve (`llama_expert_cache::device_bytes()`, `code`), and the
  output rows (`buf_output`). **Left out**, deliberately: the main context's
  own compute term (the sizing runs after `sched_new`, so no compute buffer is
  resident; `need` is what the loan's reserve will allocate in its place), the
  stream ring and the gather pages the OpenCL backend makes resident on first
  touch (backend allocations, not buffers; 751 + 187.7 MiB, `measured-here`),
  the driver's own objects, and the host-USM expert bank (not device memory;
  `n_bank_bytes` is its figure). The 1 GiB margin is for those.
- `card_total` = OpenCL's `CL_DEVICE_GLOBAL_MEM_SIZE`, read as the device props'
  `memory_total` (`ggml_backend_dev_get_props`); NOT the free query, which is
  `total - 1 GiB` and could never fail (`code`, ggml-opencl.cpp). 0 means no
  figure: the rule is not checked (`code`).
- the margin is `kCardMargin` = 1 GiB, overridable by
  `LLAMA_SLOT_LOAN_CARD_MARGIN_MIB` (absent, unreadable or non-positive keeps
  1 GiB: the knob moves the margin, it does not remove it) (`code`).
- the growth path is unchanged in shape: `extend_lend_layers` adds one candidate
  tail layer at a time until **both** rules hold (`code`); each grown layer's
  freed slot bytes lower the card's resident, so one growth serves both. When
  the tail runs out the loan is refused into the resident path. The reserve's
  own sanity (`size_fits`) stays as its own check and its own WARN (`code`); a
  card that fails gets **one** WARN naming the resident bytes before and after
  the lend, the `need`, the card's total and the card margin (`code`). The
  `loan out` line prints the card's total, the resident before the lend, the
  resident after it + `need` against `total - margin`, and the spare (`code`).

### 26b. The prediction at the 6,400 ask on the unit (`arithmetic` over `measured-here`; superseded by §27b)

The unit: IQ3_S, 11,600 MiB of slots, 131,072 ctx, the 6,400 MiB ask, the
8,192 chunk. The recorded figures:

| figure | value | class |
|---|---|---|
| the card's total | 24,480 MiB | `measured-here` (B60) |
| resident device bytes, no compute, 11,600 MiB of slots | 23,478 MiB peak at the resident chunk minus the 3,048 MiB resident compute reserve = **20,430 MiB** | `arithmetic` over `measured-here` (the IQ3_S unit's 23,478 peak, flash-next record) |
| the lend at the ask (26 layers) | **6,349 MiB** (6.20 GiB) | `measured-here` (round 16) |
| `need` at the 8,192 chunk | **4,659 MiB** (4.55 GiB) | `measured-here` (the MUTATE=3 red arm) |
| the card margin | 1,024 MiB | `code` (kCardMargin) |

Resident after the lend 14,081 MiB + need 4,659 MiB = **18,740 MiB**, against
`24,480 - 1,024` = 23,456 MiB: **4,716 MiB of spare**. The new rule therefore
**does not grow and does not refuse at the 6,400 ask**: the served shape stays
round 16's 26 layers. Growth would need the sum to cross 23,456 MiB, i.e.
4,745 MiB more resident or need; one grown layer buys ~238 MiB, so it would take
about 20 layers -- the tail has ~22 candidate layers beyond the ask, so the
refusal is the outcome of such a demand.

**The honest caveat: the parts do not reconcile with the measured peak.** The
round-16 depth arm peaked 24,438 MiB (`measured-here`), 5,698 MiB above what the
parts above predict; §24a already recorded that the delta is not explained and
owed the `need`/`reserve_matches` figures. So the card rule's accounting may
understate the process's actual peak by that residue. The rule would then refuse
or grow only once the residue shows in `need` or in the resident sum, which is
why the owed window reads the `loan out` line rather than trusting this
arithmetic. If the residue is real and does not grow with the prompt, the card
rule still holds with 4.7 GiB to spare on its own accounting and the measured
peak stays at the edge -- recorded, not gated (`measured-here`, pending).

### 26c. The failed growth's tensors (`code`, review item 2)

`alloc_buffer_n` placed each add layer's `lend[k]` in the staged USM banks;
when the growth failed (a bank allocation, an image, a file read), the `!ok`
branch freed the banks and left every tensor naming the released buffer, its
`data` and its `extra` (a USM pointer). The next attempt's alloc plan skips a
tensor with `data != NULL`, so growth would be dead for the process, and a partly
placed set would write through freed USM (`code`). The branch now nulls
`buffer`/`data`/`extra` for every add layer's `lend[k]` **before** the free, as
`refill` does for `tail_slots` (`code`); the old comment ("tensors are left for
the next attempt to place again") is gone.

### 26d. Red cases (`device-free`; each mutation is one line and was seen failing)

| red | mutation | case |
|---|---|---|
| R65 | `card_fits` dropping the `need` term (`return after <= limit;`), the margin constant 0, or the growth's card check removed (`if (true) return true;` in `extend_lend_layers`) | `the_card_rule_bounds_resident_plus_need_against_the_card`, `a_card_short_lend_grows_until_the_card_bounds_it` |
| R66 | `resolve_card_margin` accepting 0 (`mib >= 0`) | `the_card_margin_knob_moves_the_margin_but_not_removes_it` |
| R67 | the null writes removed from `extend_lend`'s `!ok` branch | `a_failed_growth_nulls_the_lend_tensors_of_every_add_layer` |

### 26e. Where §24b/§24c said "the 27th layer"

§24b's option (a) and §24c's ask paragraph stated the round-17 form: the lend
grows a 27th layer until `lend - need >= 384 MiB + 1 GiB`. That is **not a
measurement**; with round 19's card rule the arithmetic above says no growth at
the 6,400 ask on the unit, and the window decides. §24b's "6,400 grows the 27th"
claim and the `need` band derived from it in the campaign's round-18 section are
corrected there.

## 27. Round 20 (2026-10-30): the process's measured residency, the dead draft branch

A review of round 19 (`code`) found the card rule sums only ggml backend buffers,
while the peak lives largely outside them: the bank-stream ring (K x the largest
bank tensor, a plain `clCreateBuffer`, 751 MiB on the record), the backend's
grow-only preallocations (`prealloc_moe_qa`, `kq_xq`, `kq_xh`, `kq_fa`,
`kq2_xq`, `f32_splitk`, ...), and the runtime's own surfaces (kernel ISA,
scratch/private, heaps). The rule printed ~4.7 GiB of spare on the shape measured
42 MiB from the card's edge. The "draft context counted" claim was false:
`cparams.ctx_other` is never set on the target context for Flash-Next's MTP
(arcint sets it on the draft's params), so that branch was dead. Every
disposition below is `code` unless it names a `measured-here` figure.

### 27a. The measured form (`code`)

`slot_loan_begin` reads `/proc/self/fdinfo` of its xe render fds:

- **discovery**: every numeric fdinfo entry whose `drm-driver` is `xe`, with a
  `drm-client-id` and `drm-resident-vram0` (units KiB/MiB/GiB/B; `code`).
- **dedupe**: one entry a `drm-client-id` -- several render fds of one client
  repeat the same residency, and summing them would double it (`code`).
- **match** (**round 24 supersedes this**: §31a): the device's `drm-pdev` from
  `CL_DEVICE_PCI_BUS_INFO_KHR` (`cl_khr_pci_bus_info`), formatted
  `domain:bus:device.function` and carried in the device props' `device_id`. One
  xe client is trivial; several needed a unique pdev match; no unique match fell
  back to the ggml sum with a WARN (`code`).
- **the read point**: after `sched_new` freed the old compute buffers and after
  the drain the code already does, before the lend (`code`). Reading earlier
  would count the pre-sized compute buffer the loan replaces.
- **the rule**: `fdinfo_vram0 - lend_bytes_on_card + need_card + first_touch <=
  total - margin`, with `first_touch` = K x the largest bank tensor while the
  stream ring is not yet allocated (a cold process; K from
  `GGML_OPENCL_BANK_STREAM_RING`, default 4, the backend's ring bytes saying
  whether it exists; a warm ring is already in the fdinfo figure) (`code`).
- **the tripwire**: `fdinfo_vram0 < ggml sum` refuses with a WARN -- the
  process's own buffers are already evicted to GTT, and the loan's reserve would
  build on an evicted process (`code`).
- **kept**: the ggml sum as the fallback and the tripwire's reference, and
  `size_fits` as the reserve's own sanity (`code`).
- **the line**: the `loan out` WARN names fdinfo vram0, the ggml buffer sum, the
  prompt's `need`, the lend, the first touch, the card's total and the margin,
  one line a card; a card refusal's WARN names the same (`code`).
- **the lows**: the reserve-sanity WARN no longer calls the second GiB the
  "card" margin; the card-margin override is clamped at 16 GiB, so a huge value
  cannot wrap `mib << 20` (`code`).
- **the dead branch**: the `cparams.ctx_other` resident sum is removed; the MTP
  draft context's VRAM is the same process's and is in the fdinfo figure
  (`code`).

### 27b. The prediction at the 6,400 ask, against the recorded peak (`arithmetic` over `measured-here`)

The recorded depth peak is 24,438 MiB of the card's 24,480 at 103,638 tokens
(round 16, `measured-here`); the limit is `24,480 - 1,024` = 23,456 MiB. The
peak is **982 MiB over the limit**. A grown tail layer frees ~238 MiB on the
card (the measured 6,349 MiB over the ask's 26 layers, round 16), so 5 layers
(982 / 238 = 4.13 -> 5) bring the prediction to 23,248 MiB, 208 MiB under, with
7,539 MiB lent over 31 layers. The 6,400 ask therefore **grows 5 layers and does
not refuse** by this arithmetic; the 8,192 chunk and the depth rate stay (the
recorded +56 % at 103,638 tokens, 340.5 against 217.9 t/s), the growth pays
~0.10 s more refill (~1.19 GiB at the measured 11.5 GiB/s) and ~1.19 GiB of file
read once per process. Caveat (`arithmetic`): this assumes the read carries the
peak's outside parts; an allocation the reserve makes after the read (a
grow-only preallocation first used then) is not in it, so the rule may print
spare and the window's fdinfo figure decides.

### 27c. Red cases (`device-free`; each mutation is one line and was seen failing)

| red | mutation | case |
|---|---|---|
| R68 | the scan's `drm-client-id` dedupe removed (or the `drm-driver` xe filter removed) | `the_fdinfo_scan_discovers_the_xe_clients_and_counts_one_a_client` |
| R69 | **round 24 retires this red**: the unique-pdev match is gone; the device's clients are summed (`sum_fdinfo`) | `the_device_s_drm_pdev_clients_are_summed_and_the_query_takes_any_of_their_fds` |
| R70 | the tripwire check removed (`if (false)`), or the fallback returning the client | `the_fdinfo_figure_is_the_rule_and_the_ggml_sum_is_the_fallback_and_tripwire` |
| R71 | `first_touch` dropped from `card_fits`' comparison | `the_card_rule_counts_the_fdinfo_figure_and_the_cold_rings_first_touch` |
| R72 | the fdinfo scan removed from the sizing | `the_context_reads_the_fdinfo_after_sched_new_and_drain_and_refuses_eviction` |
| R73 | the card-margin clamp removed | `the_card_margin_knob_moves_the_margin_but_not_removes_it` |

## 28. Round 21 (2026-10-30): the kernel's card state, zero margin, the guard, the dirty mirror

Stage 0 of `vram-user-space.md` (operator 2026-10-30). Dispositions are `code`
unless they name a `measured-here` figure; prices over those are `(inferred)`.

### 28a. The kernel rule and its measured reserve (`code`)

- **query**: `llama_loan_book::xe_query_card(fd)` issues
  `DRM_XE_DEVICE_QUERY_MEM_REGIONS` (`size = 0` first, then the reply) on an xe
  render fd of the device's `drm-pdev` (round 24: any fd of that pdev; the query
  is device-wide), found by the fdinfo scan (`CL_DEVICE_PCI_BUS_INFO_KHR`, the
  props' `device_id`). `vram_from_regions` sums
  only `DRM_XE_MEM_REGION_CLASS_VRAM`; a reply without a VRAM region is a failed
  figure, never a zero that would pass.
- **the rule** (**round 22 corrects this bullet**, §29a): round 21's form was
  `used - lend + need + first_touch <= total - kernel_reserve`, checked at the
  begin and in every growth step (`card_state_fits`, dispatched through
  `card_fits(CardFacts, lend)`). `used` is device-wide -- other clients and the
  kernel's own VRAM objects are inside it -- so subtracting a measured reserve of
  `used - fdinfo_vram0` counted those bytes **twice**. The corrected rule is
  `used - lend + need + first_touch + allowance <= total`.
- **the resident figure** (**round 24 supersedes §27a's match**, §31a): round
  20 matched ONE xe client by `drm-pdev`; the runtime opens two per card, so the
  figure is the SUM over the device's distinct client ids. The guard's baseline
  is the same sum's GTT.
- **the reserve** (**round 22**: removed as a term): `resolve_kernel_reserve`
  returned `used - fdinfo_vram0` when the query works (measured at the load probe
  and each loan), or the knob `LLAMA_SLOT_LOAN_KERNEL_RESERVE_MIB`; 0 was valid
  explicit. Round 22 keeps the difference as `used_beyond_fdinfo`, printed as
  information, and replaces the knob with the prefill-growth allowance (§29a); a
  failed query has no figure, and the round-20 fdinfo + fixed margin form
  (`kCardMargin`) is the fallback.
- **the read points**: the load probe is in the cache's `init` and logs
  `card probe at load`; the loan's read is after `sched_new` freed the old compute
  buffers and after the drain, before the lend (round 20's point). fdinfo remains
  the per-process figure, the ggml sum the fallback and the tripwire.
- **the lines**: `loan out` (`used`, `fdinfo vram0`, `gtt`, `kernel reserve`,
  `first touch`, the total, the inequality and the spare) and every card refusal
  name the same figures; the guard's WARN names its numbers.

### 28b. The eviction guard (`code`)

**Round 22 corrects this section's rule** (§29b): round 21's
`eviction_guard(vram0, gtt, expected_vram, host_banks)` shed the larger of
`gtt - host_banks` and `expected_vram - vram0`. At the begin `expected_vram` was
the ggml sum; at the loan's end, before the resident re-reserve and the refill
(the slots are freed then), it was the sum minus the card's lend. `host_banks` is
the cache's `n_bank_bytes` (the USM bank bytes GTT should carry). The process's
other legitimate GTT made the first term fire at every begin, and the end's
expected figure was blind to the loan's compute reserve and the non-sum
residency. A shed returned the lend early (the refill path) and stored the bytes
in `slot_loan_shed`; the next begin's `shed_lend` drops the highest grown layers
(their banks are freed after a drain, their tensors nulled first), and a
remainder the tail cannot cover refuses the next loan. The ask's pick is not
dropped: its slot tensors are one buffer (`b_tail`). Round 22's form and decay
are §29b. **Round 23 supersedes the reaction** (§30a): a shed raised
`used - lend + need` (it lowers the lend), so it could not relieve the eviction,
and a remainder refused the process's lifetime. The shortfall now grows the next
lend; `shed_lend` is gone.

### 28c. The dirty lend mirror (`code`)

`DirtyTrack` (one flag a slot, `runs()` returning [start, count) pairs) replaces
the per-swap `lend_mirror`: both swap sites call `mark_lend_dirty`. At the lend's
start -- after `finish_pending` and the caller's synchronize, so the swap copies
landed, and before any tail buffer is freed (the context frees after every
cache's lend) -- `flush_lend_dirty` reads exactly the dirty runs from the card's
slot tensors (`get_slices`) and writes them into the lend bank (`set_slices`, the
bank's own IQ4_NL split / planar layout), then clears the flags. The refill's
verify against the lend bank is unchanged and stays valid. Price from the code,
with round 19's measured figures (`measured-here`): at 26 layers / 3,094 slots /
6.20 GiB, the worst case is one row a lent slot x 3 tensors = **6.20 GiB** a loan,
**~0.43 s** at the link's measured 14.3 GB/s (`(inferred)`); a dirty slot is
~2.0 MiB, ~0.15 ms (`(inferred)`). The old path's ~1.5 MiB a swap at swap time is
the measured 2.21/1.67/1.03/1.20 s per answer against off 0.16/0.11/0.09/0.12 s
(~10x, `measured-here`, round 19).

### 28d. Red cases (`device-free`; each mutation is one line and was seen failing)

| red | mutation | case |
|---|---|---|
| R74 | `DirtyTrack::mark` writes 0, not 1 | `a_swap_marks_the_slot_dirty` |
| R75 | the flush iterates every slot instead of `L.dirty.runs()` | `the_lend_copies_exactly_the_dirty_rows` |
| R76 | a `set_slices` line back in `mark_lend_dirty` (or a swap site back to the old write) | `no_per_swap_host_write_remains_for_lendable_layers` |
| R77 | the `drm-resident-gtt` parse removed | `the_guard_reads_gtt_and_the_query_from_a_fake_fdinfo` |
| R78 | **round 23 retires this red**: the shed function is gone; a true shortfall grows the next lend | `no_shortfall_state_survives_a_clean_loan_or_a_refused_begin` |
| R79 | the end guard's expected VRAM without the lend | `the_guard_fires_at_the_loan_end_before_the_refill` |
| R80 | `vram_from_regions` counting sysmem (`if (false) continue`) | `the_kernel_query_sums_only_vram_regions_and_falls_back` |
| R81 | **round 22 replaces this red**: the measured reserve is no longer a term; its case is the allowance's default | `the_prefill_growth_allowance_is_the_only_addend_and_defaults_to_zero` |
| R82 | **round 22 replaces this red**: the kernel rule's limit back to `total - (used - fdinfo)` (the double count) | `the_card_rule_is_used_minus_lend_plus_need_against_the_total_with_zero_margin` |
| R83 | **round 22 replaces this red**: the GTT term acting again (`bytes = max(vram_short, gtt_over)`) | `the_eviction_guard_acts_on_the_vram_shortfall_and_logs_the_gtt_growth` |

The fake query in the tests is `vram_from_regions` over a constructed region
list, and the fake fdinfo is a temporary directory (the scan's `dir` argument),
so the rules are exercised without a card. The card window that fixes the
measured reserve's size and the guard's thresholds is in the campaign's Round 21
commands.

## 29. Round 22 (2026-10-30): the rule's double count and the guard's baseline

A review of rounds 20-21's `code` found the ioctl, the fdinfo match and the dirty
tracking right, and two HIGH defects against the operator's zero-margin rule.
Every disposition is `code`; figures it argues over are `measured-here` (round
19's card results) and the arithmetic over them is `(inferred)`.

### 29a. The rule, zero margin beyond `used` (`code`)

`used` from `DRM_XE_DEVICE_QUERY_MEM_REGIONS` is xe's device-wide buddy usage
(`xe_ttm_vram_mgr`: `mm.size - mm.avail`). It holds this process's objects, every
other client's, the kernel's own VRAM objects and allocator rounding. Round 21
also subtracted `used - fdinfo_vram0` as a reserve, so with another client on the
card (the review's example: another process on the card, ~13.4 GiB) the left side and the
limit both carried those bytes and the loan refused with room on the card. The
rule, at the begin and in every growth step, is now

    used - lend + need + first_touch + allowance <= total

- **zero margin beyond `used`**: `kCardMargin` is not in this branch; the card's
  total is the only line.
- **`used - fdinfo_vram0` is information** (`used_beyond_fdinfo`): printed at the
  load probe and on the `loan out` line as what other clients and the kernel
  hold. It is never a term.
- **the one addend**: the prefill's **growth allowance**, named for what it is,
  default **0** (`resolve_prefill_allowance`, `LLAMA_SLOT_LOAN_PREFILL_GROWTH_MIB`,
  clamped at 16 GiB, a non-numeric value keeps 0). It covers allocations that
  first appear after the begin's read: a grow-only preallocation first used by the
  prompt, the cold ring's pages, the driver's own objects. The operator's rule is
  zero margin, so the default is zero; the window's arms run the default.
- **the fallback** when the query fails stays round 20's `fdinfo - lend + need +
  first_touch <= total - kCardMargin`. `card_fits(CardFacts, lend)` dispatches
  between the two.
- **the lines**: the load probe and `loan out` name `used`, the card's total,
  `fdinfo vram0`, `need`, `lend`, `first_touch`, the allowance, `used - fdinfo` and
  the inequality with its spare. The card refusal names the same.

### 29b. The guard's baseline, its primary signal and its decay (`code`)

- **the baseline**: at the loan's begin, per card, `slot_loan_card_base` records
  the ggml sum, the process's fdinfo `vram0` and `gtt`, the cache's host-bank
  bytes and this card's `need`. At the begin the guard is passed its own
  `gtt`/`host_banks` as the baseline, so it sees no growth and cannot shed: the
  process's other legitimate GTT -- the runtime's command buffers and heaps, the
  stage-inputs USM ring, `ids_host` -- is in the baseline, not a surplus.
- **the end**: expected VRAM is the begin's ggml sum minus the card's lend, plus
  the loan's compute reserve (`need`) and the non-sum residency measured at the
  begin (`vram0 - sum`: the ring, the runtime's kernels and heaps). Round 21's
  `sum - lend` alone was blind to up to ~9 GiB of eviction (the review's MEDIUM).
  The GTT term is the growth since the baseline minus the **bank growth** added in
  between (`n_bank_bytes` after the growth) -> `gtt_over`; it is logged, not acted
  on.
- **the primary signal**: `vram_short` (`expected > vram0`). Only it sets the shed
  amount and `shed`; `gtt_over` cannot fire alone.
- **the decay**: `shed_after_loan(shed, tripped)` keeps the measured bytes when
  the end tripped and clears them after a loan that completes without a trip; the
  end stores the result in `slot_loan_shed`. One trip no longer disables the loan
  for the process's lifetime. Round 23 adds the floor and makes the carried bytes
  the next lend's growth (§30a).
- **the knob**: `LLAMA_SLOT_LOAN_EVICTION_GUARD` absent is on; `0`, `off`,
  `false`, `no` (any case) turn it off (one `off_word` predicate shared with the
  exit-idle knob); a typo keeps it on. Off clears the shed and logs one line.

### 29c. The LOW items (`code`)

- the load probe (fdinfo scan + ioctl + WARN line) runs only when
  `lend_bytes_ != 0`: the agent's and the coder's caches, loan off, pay nothing.
- `fdinfo_parse_bytes` reads `" TiB"` (a figure reported in TiB would otherwise be
  raw bytes, understated by 2^40).
- a failed query's `why` carries errno (`card_why_errno`): `<what>: <strerror>
  (errno <n>)`, so ENOTTY / EPERM / EINVAL are visible in the window's log.
  Round 23 makes the reason itself an owned `std::string` (§30c).

### 29d. Red cases (`device-free`; each mutation is one line and was seen failing)

| red | mutation | case |
|---|---|---|
| R84 | the dispatch subtracting `used - fdinfo` from the limit | `the_card_rule_is_used_minus_lend_plus_need_against_the_total_with_zero_margin` |
| R85 | the allowance resolver's default non-zero | `the_prefill_growth_allowance_is_the_only_addend_and_defaults_to_zero` |
| R86 | the begin's GTT baseline passed as 0 | `a_false_gtt_surplus_at_the_begin_does_not_shed` |
| R87 | `shed_after_loan` ignoring `tripped` | `a_shed_decays_after_a_clean_loan` |
| R88 | `resolve_eviction_guard` false for every value | `the_eviction_guard_knob_turns_the_guard_off` |
| R89 | the `TiB` line removed | `fdinfo_reads_tib_and_the_query_why_carries_errno` |
| R90 | the errno text dropped from `card_why_errno` | the same case's errno CHECK |
| R91 | `init`'s probe gate removed (`if (true)`) | `the_load_probe_runs_only_for_a_cache_with_a_lend` |
| R92 | the end's `+ base.need + non_sum` dropped | `the_end_guard_sees_the_reserve_and_the_non_sum_residency` |
| R93 | `g.bytes = max(vram_short, gtt_over)` | `the_eviction_guard_acts_on_the_vram_shortfall_and_logs_the_gtt_growth` |

Each was applied to the tree, the suite built, exactly the named case failed, and
the restored line ran all 468 cases green. The card heights are owed (the
campaign's Round 21 commands, corrected): the zero-margin arm is the default and
the window reads the `loan out` line's `used`, total, `vram0`, `need`, `lend`,
`first_touch` and allowance, the `card probe at load` line, and the guard's
`SHEDS`/`does not shed` line against the 2 s sampler.

## 30. Round 23 (2026-10-30): the guard's noise floor, the shortfall as growth, `used == 0`

A review of round 22's `code` found the zero-margin rule and the loan-off path
verified. What follows repairs one HIGH mechanism defect, one MEDIUM figure
defect and three lows. Every disposition is `code`; the figures it argues over
are `measured-here` (round 19's card results) and the arithmetic over them is
`(inferred)`. No card arm is added; §30d says what the owed window reads.

### 30a. The noise floor, and the reaction that can work (`code`)

Round 22 reacted to `vram_short = expected - vram0` with a shed: `shed_lend`
dropped the highest grown lend layers and a remainder refused the next loan.
Two facts say that reaction cannot work:

- `expected` is exact tensor bytes, `vram0` is BO residency. Allocator rounding,
  the reserve's allowed under-run (`reserve_matches`) and the reserve's own
  rounding put `vram0` a few MiB below `expected` on a healthy loan, so a 1-byte
  shortfall is not evidence. Round 22 acted on any byte.
- after the lend the process's residency is `used - lend + need + first_touch`.
  Shedding lend bytes RAISES that left side, so a shed deepens the pressure that
  caused the eviction. A remainder the grown tail could not cover refused the
  loan for the process's lifetime, because `shed_after_loan` cleared it only at
  a clean END and a refused begin has no end.

The form now:

    true shortfall = vram_short >= kShedFloor   (default 64 MiB)
    next loan     = sizing runs growth with demand = need + true shortfall

- `kShedFloor` = 64 MiB (`resolve_shed_floor`,
  `LLAMA_SLOT_LOAN_SHED_FLOOR_MIB`: digits only, 0 allowed as the operator's
  decision, clamped at 16 GiB, a typo keeps 64 MiB; `growth_from_shortfall`
  returns 0 below the floor).
- the shortfall feeds the growth path as extra demand: the context passes
  `need + grow_by` to `extend_lend` and sets the card copy's `card.need` to the
  demand, so `size_fits` and the card rule both hold with that much more
  headroom. Each grown layer lowers `used` by its slot bytes, which is the one
  action that reduces the left side. The refusal checks keep the actual `need`:
  the shortfall is never a refusal, and a tail that cannot cover it costs one
  WARN. **Round 25 corrects the premise** (`code`): the growth was all-or-nothing
  on the demand, so a tail that could not reach `need + grow_by` grew nothing,
  and the `need` check then refused a loan a partial growth would have allowed.
  §32a makes that two steps; only the `need` rule refuses.
- the begin TAKES `slot_loan_shed` and zeroes it at its top, before every early
  return, so a refused begin leaves no state; a clean end stores 0
  (`shed_after_loan(short_beyond, guard_tripped)`).
- `llama_expert_cache::shed_lend` is removed with its last caller (R78 retires
  from §28d).

### 30b. `used == 0` is no figure (`code`)

`DRM_XE_DEVICE_QUERY_MEM_REGIONS` reports `used = 0` on a kernel without
`CAP_PERFMON` (the `if (perfmon_capable())` around `used` in older trees;
campaign `vram-user-space`, `code`). Taking it as a figure made the zero-margin
rule vacuous -- `used - lend + need` is then 0 for any card. `query_is_a_figure
(used, total)` requires `total > 0 && used > 0`; a reply failing it sets
`ok = false` with the reason ("reports used 0 with a total of N bytes"), so the
round-20 fdinfo + fixed-margin fallback decides and the caller's existing WARN
names it. One WARN, no silent half-figure.

### 30c. The lows (`code`)

- `CardQuery::why` is a `std::string`; `card_why_errno` returns it. The old
  `const char *` pointed into the query's own `why_buf`, and the query is
  returned by value, so a copy's pointer pointed into the source. The unused
  out-param `xe_query_card(fd, &why)` is gone (no caller passed it).
- the campaign's Round 21 red table: R78 retires with `shed_lend`, R81-R83 name
  the round-22 cases (R85/R84/R93) that exist.
- "the image service" -> "another process on the card" in the four places
  (header comment, campaign, design, test).
- the guard-knob case's last CHECK (`!off("0") || on("08")`) was tautological
  by its first half; it now asserts `"08"` and `"0x0"` keep the guard on.

### 30d. Red cases (`device-free`; each mutation is one line and was seen failing)

| red | mutation | case |
|---|---|---|
| R94 | `growth_from_shortfall` returns the raw shortfall (floor gone) | `a_sub_floor_shortfall_does_nothing` |
| R95 | `card.need = need` back (the rule does not carry the demand) | `a_true_shortfall_grows_the_next_lend_and_the_need_rule_refuses_alone` |
| R96 | the begin's `slot_loan_shed = 0;` removed | `no_shortfall_state_survives_a_clean_loan_or_a_refused_begin` |
| R97 | `query_is_a_figure` without `used > 0` | `used_zero_with_a_total_is_no_card_figure` |
| R98 | the old own-buffer shape back in `CardQuery` (`why_buf`) | `the_query_reason_is_a_copy_safe_string` |

Seen failing per case with `arcint-test` rebuilt, then green on the restored
line; the suite is 472 cases green. R98's first draft used one `has_between` list
for both words, which passes when only one of them returns -- the case now checks
each token on its own (`code`, found in this round's red-first). The owed window
reads the end guard's `FEEDS THE NEXT LEND` / `does not grow the lend (below the
floor)` line with its shortfall and floor, and the next begin's one WARN when it
carries a true shortfall; the `loan out` line and the pass rule are round 22's.

## 31. Round 24 (2026-10-30): the fdinfo discovery -- the runtime's two xe clients per card

The round-22 card window found the loan's card figure never engaged
(`measured-here`: every arm's `card probe at load failed (no xe render fd)`, the
`loan out` line's fdinfo equal to the ggml sum, the guard's end line saying `no
fdinfo client ... several xe clients on the same drm-pdev`). Every disposition
below is `code`; the cause is `measured-here`; the arithmetic is `(inferred)`.

### 31a. The figure is a card's, not a client's (`code`)

A running arcint's `/proc/<pid>/fd` and fdinfo show the OpenCL runtime opens
**two xe clients per card in one process** (`measured-here`): a probe client
(vram0 ~2 MiB) and the real one (vram0 16 GiB), same `drm-pdev`, different
`drm-client-id`. Round 20 read one client as "the" client, so two on a pdev was
a conflict and the fallback ran. Every xe client in `/proc/self` is this
process's, so:

- the card's `vram0` and `gtt` are the **SUM** over every distinct
  `drm-client-id` whose `drm-pdev` equals the device's -- the scan dedupes by
  client id alone (two fds of one client report the same bytes once), and two
  distinct ids on one pdev are two terms;
- `DRM_XE_DEVICE_QUERY_MEM_REGIONS` runs on **any** render fd of that pdev
  (`used`/`total` are device-wide, `code`); `FdinfoAgg` carries the smallest fd
  deterministically;
- one xe client stays trivial; clients only on another pdev, or several with no
  device pdev, remain the ggml-sum fallback with a WARN;
- `read_resident` keeps the tripwire on the summed figure.

### 31b. The load probe's order (`code`)

`llama_expert_cache::init` allocates the cache's tail/slot buffers
(`ggml_backend_buft_alloc_buffer_n`) and `load_tensors` loads the weights before
the probe's scan runs, so the backend's xe render fds exist. The probe stays
there; a text-order case pins scan after allocation.

### 31c. Red cases (`device-free`; each mutation is one line and was seen failing)

| red | mutation | case |
|---|---|---|
| R99 | the sum loop breaks at its first pdev match | `the_device_s_drm_pdev_clients_are_summed_and_the_query_takes_any_of_their_fds` |
| R100 | `a.gtt += c.gtt;` removed | the same case |
| R101 | the probe's `sum_fdinfo(scan, props.device_id)` loses its pdev (falls back) | `the_probe_reads_the_device_s_sum_after_the_backend_opened_its_fds` |
| R102 | the loop's `c.pdev != pdev` compare removed (another card's client summed) | `the_device_s_drm_pdev_clients_are_summed_and_the_query_takes_any_of_their_fds` |

Seen failing per case with `arcint-test` rebuilt, then green on the restored
line; the suite is 473 cases green. The owed window's pass addition: the tiny
gate's probe line is a SUCCESS with `used`/`total`, and its `loan out` line's
`fdinfo vram0` differs from the ggml sum; the end guard names a figure. The
round-21 command block (§28d) is unchanged.

## 32. Round 25 (2026-10-30): the carried shortfall's growth in two steps

The final polish before the push. Every disposition below is `code`; the
numbers are arithmetic over `code` constants (`(inferred)` where they name
bytes from the unit's record). No card arm is added; §28d's round-21 command
block stands.

### 32a. The two-step growth (`code`)

Round 23's growth was all-or-nothing on the demand `need + grow_by`
(`extend_lend_layers` stops only when the demand's `size_fits` and `card_fits`
hold, and returns `why` when the tail runs out; `extend_lend` then committed
nothing). The refusal that followed compared the actual `need`, so a tail that
could not reach the demand left the ask pick ungrown and the `need` check
refused a loan whose partial growth would have met it. That made the carried
shortfall a refusal term, which round 23's wording denied.

`extend_lend_toward` (`llama-loan-book.h`, `extend_lend` calls it) is the
repair:

- step 1 is the demand: `extend_lend_layers(cand, ask, demand, beside, card)`
  with `card.need = demand`.
- step 2 runs only when step 1's tail cannot reach the demand: the tail's full
  growth -- as far as the tail allows toward the demand -- is kept when it still
  holds `size_fits(need, ...)` and `card_fits(card.need = need, ...)`. Otherwise
  nothing grows and the caller's `need` checks decide as before.
- `LendGrowth::reached_demand` records which step the set came from; the
  context's WARN on an unmet demand is unchanged, and only the `need` rule and
  the card's rule at `need` can refuse.

So the shortfall can raise the lend but never the refusal threshold: a loan the
`need` rule allows runs, at least as far as the tail can take it toward the
demand.

### 32b. The red case (`device-free`; R103)

| red | mutation | case |
|---|---|---|
| R103 | step 2 dropped (`g.pick = ask;` when the demand is out of the tail's reach) | `a_short_tail_still_runs_the_loan_at_what_need_alone_requires` |

The case's numbers (`code`, `(inferred)` arithmetic): 26 layers lent at 238 MiB
(6,188 MiB), `need` 5 GiB (5,120 MiB), a carried 2 GiB shortfall (demand
7,168 MiB), 5 tail layers (+1,190 MiB = 7,378 MiB). The demand needs 8,576 MiB
with the 1.375 GiB margins, so the tail cannot reach it; the ask needs 6,528 MiB
and does not meet it either, so the old growth left a refusal; the tail's full
growth, 31 layers, meets `need` and the loan runs. The case asserts the 26-layer
ask, the two failing demand sums, the 31-layer 7,378 MiB outcome with
`reached_demand` false and `why` null, and that one grown layer is short while
two meet the rule. The mutation was applied to the tree, `arcint-test` rebuilt,
the case seen failing (`why != nullptr`, 26 layers, 6,188 MiB), the line restored
and the suite's 474 cases green.

R95's case is renamed `a_true_shortfall_grows_the_next_lend_and_the_need_rule_refuses_alone`
and carries value assertions (the demand `need + shortfall` against a lend that
fits `need` alone and not the demand; the `extend_lend(demand, beside, card,
need)` call and the `need` comparison as its text half). R96 adds the carry's
values (`shed_after_loan` keeps the measured bytes on a trip, zeroes them on a
clean loan; floor 0 lets every byte count). R101 adds the probe's values -- the
summed `vram0` over two client ids on the device's pdev and the smallest pdev fd
-- beside its text-order checks. Each of those value assertions fails on the
behaviour it names (`code`), so the cases are not text-only.
