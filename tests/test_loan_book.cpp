// The prefill-slot-loan's bookkeeping rules, without a card (second review
// findings 6 and 7, third review findings 1, 2, 3, 5, 6, 7 and 8). The functions
// are the patch's own (llama.cpp src/llama-loan-book.h): llama-expert-cache.cpp
// calls exactly these -- the phase state it plans and settles is this header's
// -- so a mutation of a rule fails a case here before any card run.
//
// Red cases (each is a one-line edit to llama-loan-book.h or to the pinned
// tree's llama-model.cpp; the named test then fails):
//   R1 settle() clearing a planned phase instead of issuing it (the bug) ->
//      a_settled_unissued_phase_one_returns_its_spare,
//      a_settled_unissued_phase_two_applies_its_tables
//   R2 settle() without the fence between the phase-1 tables and the phase-2
//      copies -> the_settle_fences_the_second_phase
//   R3 the mirror inside the k loop (three calls a swap) ->
//      the_mirror_is_written_once_a_swap
//   R5 loan_cache() true for lend_bytes 0 -> a_cache_without_a_lend_is_not_lent
//   R6 refill_zero_row() naming row -1 (the bug: the zero slot left unwritten) ->
//      the_refill_restores_the_zero_slot_too
//   R7 lend_broken() false whatever the check said ->
//      a_failed_lend_whose_undo_still_differs_is_broken
//   R8 resolve_lend_bytes() reading the environment for an explicit 0 (the bug)
//      -> an_explicit_zero_flag_stays_off; llama-model.cpp's default at 0
//      instead of the sentinel -> the_default_params_ask_for_the_environment
//   R6/R7 (second review) as before: `undo_lend()` without the restore loop ->
//      a_failed_lend_puts_the_layers_back; the tail buffer listed in the
//      destructor's list -> the_tail_buffer_is_not_the_destructor_list
//   R12 choose_lend_layers() breaking at a non-lendable layer (the tail-only bug)
//      -> a_mixed_set_skips_a_non_lendable_layer_and_a_layer_that_does_not_fit;
//      breaking at a layer that does not fit -> a_loan_skips_an_oversized_tail_layer...
//   R9 poison_exit() without the exit (or the log) ->
//      the_poison_policy_exits_non_zero_with_the_cause;
//      std::exit instead of std::_Exit (static destructors run from the
//      generation thread under the mutex: a hang) -> the same case's no-ATEXIT
//      check; log_line removed -> its LOG: prefix check (fifth review findings 1, 2)
//   R17 the refill's region copy replaced by the old staged path (get_slices into
//      host rows, then set_slices) -> the_refill_copies_regions_from_the_bank...
//   R18 lend_write's split-extra build removed (the re-allocated IQ4_NL slot keeps
//      the plain extra and a region cast reads garbage) ->
//      the_refill_builds_the_split_extra_before_it_copies
//   R19 verify_ref()'s default is the file (the 6.2 GiB read a loan comes back) ->
//      the_refill_check_defaults_to_the_lend_bank; "file" not mapped ->
//      the_refill_check_can_be_the_file_for_a_gate_arm
//   R20 the load's lend bank check removed -> the_lend_bank_is_checked_at_load;
//      size_fits() without the peak reserve or with the comparison inverted ->
//      a_size_check_leaves_a_peak_margin
//   R34 measure_tokens() returning the prompt chunk for MUTATE=3 (the round-9 bug:
//      the lent-flag skip raises need instead of lowering it) ->
//      the_lent_sizing_is_measured_unless_mutated
//   R35 drain_now() a no-op / the drain removed from drop_tail_buffer or the
//      return's poison path (a free before queued work finished: the B60's xe
//      -ENOENT at the MUTATE=4 exit) -> a_poison_path_drains_before_any_free
//   R36 slot_loan_begin's sizing unwrapped (a throw after sched_new leaves an
//      unreserved scheduler and a dangling owner_sched) ->
//      a_lent_sizing_throw_takes_the_resident_refusal; and the !sized block's
//      rebind before its resident re-reserve (owner_sched moved onto the
//      scheduler the failed reserve leaves) -> the same case's order check
//      (text order: the source lines, not a call)
//   R37 slot_loan_reserve without the pre-sched_new drain (a replace frees
//      buffers the adapt queue may still read) ->
//      a_re_reserve_finishes_the_queues_before_sched_new (text order)
//   R38 the cache destructor's drain keyed on b_tail (after a lend it is null,
//      the USM banks are freed undrained) ->
//      a_lent_cache_drains_before_every_free_in_the_destructor (text order)
//   R39 the failed tail allocation's dangling tensors (alloc_buffer_n frees the
//      partial buffers, the tensors keep the pointers) ->
//      a_failed_tail_allocation_leaves_no_dangling_tensor (text order)
//   R40 poison_exit's idle removed (or exit_idle_ms() 0 for every value: _Exit
//      microseconds after the last GPU work, the round-11 MUTATE=4 fault) ->
//      the_exit_waits_for_the_rings_to_stop
//   R41 exit_idle_ms() without the watchdog clamp (a wait that eats the bound a
//      hung drain needs) -> the_exit_idle_knob_is_bounded
//   R42 the idle placed before poison_exit's drain (waits while the queues still
//      hold work, then _Exit) -> the_exit_waits_for_the_rings_to_stop's order
//   R43 the drain's line back at GGML_LOG_INFO (arcint shows WARN and above, so
//      the card never sees it) -> the_drain_line_is_at_warn (text order)
//   R44 the idle's second line removed (or written before the sleep: the elapsed
//      falls under the asked value) -> the_exit_reports_the_idle_it_actually_waited
//   R45 exit_idle_ms() case-sensitive again (an OFF arm silently takes the
//      default) -> the_idle_knob_reads_the_off_words_in_any_case
//   R46 the watchdog's one write(2) removed (a cut idle leaves no line) ->
//      the_watchdog_fires_with_one_async_safe_line
//   R47 the loan-KL driver's threshold >= instead of > (a 2,048-token chunk asks
//      for a loan the context's wanted() refuses) ->
//      the_kld_driver_takes_the_loan_above_the_resident_chunk
//   R48 the driver's begin moved after its decode (the capture then holds logits
//      of an off prefill) -> the_kld_driver_begins_before_its_decode_and_ends_after
//   R21 the refill's comment's planar list off ggml_cl_lb_fields' cases ->
//      the_refill_comment_names_the_planar_types
//   R10 slot_loan_begin's lend loop without `continue` (a cache with lend_bytes 0
//      fails the AND) -> the_lend_loop_skips_a_cache_without_a_lend
//   R11 decode's budget off n_ubatch instead of ub_cap ->
//      the_decode_budget_uses_the_capped_ubatch
//   R22 the 384 MiB margin off the freed lend bytes (round 8's device free figure
//      is `total - 1 GiB`, a constant) -> a_size_check_leaves_a_peak_margin
//   R23 the reserve's actual bytes not compared with the sizing's expectation
//      (MUTATE=3 runs as a healthy loan) -> the_lent_sizing_is_measured_unless_mutated//   R24 the poison exit without the drain / the drain unregistered ->
//      the_poison_policy_exits_non_zero_with_the_cause,
//      the_poison_drain_finishes_the_device_queues_first
//   R25 the bank check's per-row form across region-order buffers ->
//      a_region_order_read_compares_row_by_row
//   R26 region_copyable() naming a type set_tensor builds an extra for ->
//      a_lendable_type_is_one_the_region_copy_is_exact_for
//   R27 MUTATE=1 skipping a copy (an unbuilt split extra, a driver segfault) ->
//      the_red_switch_corrupts_one_lent_row_after_the_copies
//   R28 lend_zero after lend_write (the zero page's finish between the copies) ->
//      the_zero_page_drain_precedes_the_copies
//   R29 the split set's staleness comment removed -> the_split_set_names_its_staleness
//   R30 reserve_matches() equality instead of actual <= expect -> the same
//      sizing test's budget (a shared buft legitimately sums lower)
//   R31 the region-row loop stopping after the first region -> the row check
//      counts/names only one region (the per-tensor form)
//   R32 the watchdog's spawn unguarded (std::thread throws: terminate, no 70) ->
//      a_watchdog_construction_failure_still_exits
//   R33 the size check off the freed lend bytes (the fake device free figure) ->
//      a_size_check_leaves_a_peak_margin
// Round 22's red cases (R84-R93):
//   R84 the kernel rule's limit back to `total - (used - fdinfo)` (the other
//      client's and the kernel's bytes subtracted a second time) ->
//      the_card_rule_is_used_minus_lend_plus_need_against_the_total_with_zero_margin
//   R85 the prefill growth allowance's default non-zero (or used - fdinfo kept
//      as a rule term) -> the_prefill_growth_allowance_is_the_only_addend_and_defaults_to_zero
//   R86 the begin judging GTT against the host banks alone (a false surplus sheds)
//      -> a_false_gtt_surplus_at_the_begin_does_not_shed
//   R87 shed_after_loan() ignoring `tripped` (one trip shrinks the lend for the
//      process's lifetime) -> a_shed_decays_after_a_clean_loan
//   R88 resolve_eviction_guard() false for every value -> the knob test
//   R89 fdinfo_parse_bytes' TiB line removed -> fdinfo_reads_tib_and_the_query_why_carries_errno
//   R90 card_why_errno() without the errno text -> the same case's errno CHECK
//   R91 init's probe gate removed (the scan and the ioctl run with the loan off)
//      -> the_load_probe_runs_only_for_a_cache_with_a_lend
//   R92 the end guard's `need` and non-sum addends dropped (up to ~9 GiB unseen)
//      -> the_end_guard_sees_the_reserve_and_the_non_sum_residency
//   R93 g.bytes = max(vram_short, gtt_over) (the GTT term acts again) ->
//      the_eviction_guard_acts_on_the_vram_shortfall_and_logs_the_gtt_growth
// Round 23's red cases (R94-R98):
//   R94 the end guard's noise floor dropped (`growth_from_shortfall` returning the
//      raw shortfall, or `kShedFloor` 0) -> a_sub_floor_shortfall_does_nothing
//   R95 the shortfall not fed to the next lend (the begin's `grow_by` dropped from
//      the sizing, or `card.need = demand` removed) ->
//      a_true_shortfall_grows_the_next_lend_and_the_need_rule_refuses_alone
//   R96 a refused begin keeping the carried shortfall (the begin's `slot_loan_shed = 0`
//      removed, or `shed_after_loan` keeping the bytes on a clean loan) ->
//      no_shortfall_state_survives_a_clean_loan_or_a_refused_begin
//   R97 used == 0 with a total taken as a figure (`query_is_a_figure` removed) ->
//      used_zero_with_a_total_is_no_card_figure
//   R98 CardQuery's reason a `const char *` into its own buffer again (a returned
//      query's copy points into the source) -> the_query_reason_is_a_copy_safe_string
//   R49 mutate_loan_bank_row() without the shift (MUTATE=5 then mutates nothing
//      and the KL gate's red arm would pass) -> the_lent_table_can_be_put_off_by_one;
//      the shift removed from lend() -> the_lent_mutation_is_applied_at_the_lend_and_names_itself
//   R50 the driver's nearest_int() replaced by std::lround (half away from zero,
//      the control arm's rows then differ) -> the_kld_quantiser_is_perplexitys_half_to_even
//   R51 output_rows() back to n_batch, or the driver's n_outputs_max assignment /
//      the one-row buffer / the write-before-end order removed ->
//      the_kld_logits_reservation_is_the_row_window
//   R52 the partial --out guard removed, or the cold-rate line removed ->
//      the_kld_driver_unlinks_a_partial_capture_and_says_the_rate_is_cold
//   R53 served_shape() accepting a smaller ubatch, or the n_ubatch_reserve
//      assignment removed -> the_kld_on_arm_runs_the_served_ubatch_shape
//   R54 MUTATE=5's shift written into tab_host outside fill_tables (one expert, then
//      check_tables rebuilds want from loan_of and the lend refuses: no scorable
//      red) -> the_lent_override_reaches_every_table_producer
//   R55 the Q8_0 down's tab_loan view at tab_row + 6 (fill_tables and the mutation
//      write row + 5; a model whose last cached layer has a Q8_0 down asserts at
//      load) -> the_loan_row_is_the_layers_last_row
//   R56 the driver keeping 4,096 output rows in one lent ubatch (4.07 GB in the
//      sizing and on OpenCL's plain CPU output buffer) ->
//      the_kld_scores_the_served_ubatch_in_row_windows
//   R57 the driver's false "the end frees the logits" comment, or the
//      --ubatch-reserve wording calling it optional ->
//      the_kld_comments_state_what_the_end_and_the_reserve_do
//   R58 extend_lend_layers() returning the ask unchanged (the lend never grows:
//      the fixed margin is short and the growth is the operator's option a) ->
//      a_short_margin_grows_the_lend_by_one_tail_layer
//   R59 `why` left null when the candidates run out (a pick that does not fit is
//      read as a fit) -> the_tail_running_out_refuses_the_growth
//   R60 kCardMargin 0, or size_fits dropping the card term (the 1.375 GiB
//      boundary moves) -> the_fixed_margin_is_the_documented_one, and
//      a_size_check_leaves_a_peak_margin's boundaries
//   R61 a grown layer's lend bank filled with get_slices -> set_slices on the
//      fresh USM tensor (IQ4_NL's usm_split only a whole set_tensor builds, so
//      .at() throws on every ffn_down_exps layer) ->
//      a_grown_lend_bank_is_filled_by_the_whole_tensor_path
//   R62 the growth call moved outside the sizing's try (a throw after sched_new
//      leaves an unreserved scheduler and a dangling owner_sched) ->
//      a_growth_throw_takes_the_resident_refusal
//   R103 (round 25) the two-step growth's second step dropped (`g.pick = ask;` when
//      the demand is out of the tail's reach): a carried shortfall then refuses a
//      loan the `need` rule allows ->
//      a_short_tail_still_runs_the_loan_at_what_need_alone_requires
//   R63 the grown layer's spare slot buffer freed inside extend_lend before the
//      lend is published (a failed lend_undo resumes on released memory) ->
//      a_grown_layer_is_not_freed_before_the_lend_is_published
//   R64 init's failed spare allocation leaves the freed pointers in spare_buf
//      (the destructor frees them again) ->
//      a_failed_spare_allocation_nulls_every_freed_buffer
//   R65 card_fits dropping the need or the resident term, or the growth's card
//      check removed from extend_lend_layers' fits() (a card-short lend is
//      returned unchanged and the loan runs at the card's edge) ->
//      the_card_rule_bounds_resident_plus_need_against_the_card,
//      a_card_short_lend_grows_until_the_card_bounds_it
//   R66 resolve_card_margin returning 0 for an absent/0/typo value (the knob
//      silently removes the operator's margin), or its default read elsewhere ->
//      the_card_margin_knob_moves_the_margin_but_not_removes_it
//   R67 a failed growth leaving the add layers' lend[k] buffer/data/extra after
//      the staged banks are freed (the next attempt's alloc plan skips the tensor;
//      growth is dead and a partly placed set writes through freed USM) ->
//      a_failed_growth_nulls_the_lend_tensors_of_every_add_layer
//
// The card commands that prove the mechanism on device are in the campaign
// documents; these cases are what a device-free build can see.

#ifdef ARCINT_LLAMA_SLOT_LOAN

#include "llama-loan-book.h"

#include "llama.h"   // LLAMA_SLOT_LOAN_BYTES_ENV, llama_model_default_params
#include "loan_kld_book.h"   // round 13: the loan-KL driver's rules

#include "harness.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

// a layer's loan tables, as llama_expert_cache_layer holds them: slot_of,
// loan_of, lent. 4 experts, 2 slots: experts 0 and 2 in the slots, 1 and 3 in
// the bank (before a lend).
struct Layer {
    bool lent = false;
    std::vector<int32_t> slot_of{0, -1, 1, -1};
    std::vector<int32_t> loan_of{-1, -1, -1, -1};
};

// the lend's table half, as llama_expert_cache::lend writes it: per ex-slot
// expert its slot row into loan_of, and slot_of = -1
void lend_tables(Layer & L) {
    for (size_t e = 0; e < L.slot_of.size(); ++e) {
        if (L.slot_of[e] < 0) continue;
        L.loan_of[e] = L.slot_of[e];
        L.slot_of[e] = -1;
    }
    L.lent = true;
}

}  // namespace

// ---- third review: the pending phases (llama-loan-book.h's PhaseState) ----

namespace {

// the cache's pending_swap, as the cache pushes it: the victims' slot and spare,
// the newcomers' bank entry
struct Swap { int layer = 0, e_in = 0, e_out = 0, slot = 0, b_in = 0, x = 0; };

// the cache's callbacks, as data: the table halves applied to one layer's
// residency arrays (what pend_tables_out/in write), the spares a phase-2 table
// half returns, and an ordered log of what ran, so the fence's place is provable.
struct Ops {
    // one layer's tables: expert -> slot, slot -> expert (the cache's slot_of,
    // expert_of); the spare entries a settle returns
    std::vector<int> slot_of{0, -1, 1, -1};
    std::vector<int> expert_of{0, -1, -1, -1};   // expert 0 in slot 0; the plan's victim leaves
    std::vector<int> spares;

    std::vector<std::string> log;
    int copies_out = 0, copies_in = 0, mirrors = 0, records = 0, finishes = 0, checks = 0;
    bool landed = true;   // adapt_done: the issued copies landed

    void fence()  { log.push_back("fence"); }
    int  record() { log.push_back("record"); return ++records; }
    bool done()   { return landed; }
    void finish() { log.push_back("finish"); ++finishes; }
    void copy_out(const Swap &, int) { log.push_back("copy_out"); ++copies_out; }
    void copy_in (const Swap &, int) { log.push_back("copy_in");  ++copies_in; }
    void mirror  (const Swap &)      { log.push_back("mirror");   ++mirrors; }
    void tables_out(const std::vector<Swap> & p) {
        log.push_back("tables_out");
        for (const Swap & s : p) {
            int op = -1;
            for (size_t e = 0; e < slot_of.size(); ++e) {
                if (slot_of[e] == s.slot) op = (int) e;
            }
            expert_of[s.slot] = -1;
            if (op >= 0) slot_of[(size_t) op] = -1;
        }
    }
    void tables_in(const std::vector<Swap> & p) {
        log.push_back("tables_in");
        for (const Swap & s : p) {
            expert_of[s.slot] = s.e_in;   // the newcomer serves from the slot again
            slot_of[s.e_in]   = s.slot;
            spares.push_back(s.b_in);     // its old bank entry is a spare again
        }
    }
    void check_swap() { log.push_back("check"); ++checks; }
};

Swap one_swap() {   // expert 2 (bank entry b_in 5) into slot 0 (holding expert 0), spare x 7
    Swap s;
    s.e_in = 2; s.e_out = 0; s.slot = 0; s.b_in = 5; s.x = 7;
    return s;
}

// where "a" first appears relative to "b" in the log; -1 when absent
int at(const std::vector<std::string> & log, const std::string & what, int from = 0) {
    for (size_t i = (size_t) from; i < log.size(); ++i) {
        if (log[i] == what) return (int) i;
    }
    return -1;
}

}  // namespace

TEST(a_settled_unissued_phase_one_returns_its_spare) {
    // finding 1: adapt() popped the spare and named the victim; the ubatch failed
    // or was aborted before after_graph, so pend was never issued. The loan's
    // finish_pending settles it now. The old code cleared the plan: the spare the
    // adapt popped was never returned and adapt stopped at its first layer.
    Ops ops;
    std::vector<Swap> plan{one_swap()};
    llama_loan_book::PhaseState<Swap> book;
    book.pend = plan;
    book.phase = 1;

    llama_loan_book::settle(book, ops);

    CHECK_EQ(ops.copies_out, 3);   // the victims' three k, once
    CHECK_EQ(ops.copies_in, 3);    // the newcomers' three k, once
    CHECK_EQ(ops.mirrors, 1);      // finding 3: once a swap, not per k
    CHECK_EQ(ops.spares, std::vector<int>({5}));   // R1's red case: the spare is back
    CHECK_EQ(ops.expert_of[0], 2);                 // and the newcomer serves from its slot
    CHECK_EQ(book.phase, 0);
    CHECK(book.pend.empty());
}

TEST(a_settled_unissued_phase_two_applies_its_tables) {
    // finding 1, phase 2: progress() applied the phase-1 tables (expert_of[slot]
    // = -1, the victim in spare x) and moved to phase 2 without issuing its copies.
    // The old clear orphaned the slot: the refill copied the lend row back and the
    // verify compared it against expert_of = -1 (zeros), so the loan stuck out.
    Ops ops;
    ops.expert_of[0] = -1;   // the phase-1 tables, already applied
    ops.slot_of[0]   = -1;
    std::vector<Swap> plan{one_swap()};
    llama_loan_book::PhaseState<Swap> book;
    book.pend  = plan;
    book.phase = 2;

    llama_loan_book::settle(book, ops);

    CHECK_EQ(ops.copies_in, 3);
    CHECK_EQ(ops.mirrors, 1);
    CHECK_EQ(ops.expert_of[0], 2);   // R1's red case: the slot is not orphaned
    CHECK_EQ(book.phase, 0);
    CHECK(book.pend.empty());
}

TEST(the_settle_fences_the_second_phase) {
    // finding 2: the phase-2 copies go on the adapt queue, unordered against the
    // compute queue; progress places a fence before exactly these copies, and the
    // settle must as well (a graph still reading the victim slot would race the
    // newcomer's write)
    Ops ops;
    std::vector<Swap> plan{one_swap()};
    llama_loan_book::PhaseState<Swap> book;
    book.pend   = plan;
    book.phase  = 1;
    book.issued = true;   // after_graph issued phase 1; its copies landed

    llama_loan_book::settle(book, ops);

    const int out = at(ops.log, "tables_out");
    const int f   = at(ops.log, "fence");
    const int in  = at(ops.log, "copy_in");
    CHECK(out >= 0 && f >= 0 && in >= 0);
    CHECK(out < f);   // R2's red case: the fence after the phase-1 tables ...
    CHECK(f < in);    // ... and before the phase-2 copies
}

TEST(a_settled_unissued_phase_is_fenced_before_its_copies) {
    // finding 2 at the other entry: a phase nobody issued is fenced here, so its
    // copies wait for every compute command so far -- including the graph that
    // failed or was aborted
    Ops ops;
    std::vector<Swap> plan{one_swap()};
    llama_loan_book::PhaseState<Swap> book;
    book.pend  = plan;
    book.phase = 2;

    llama_loan_book::settle(book, ops);

    const int f  = at(ops.log, "fence");
    const int in = at(ops.log, "copy_in");
    CHECK(f >= 0 && in >= 0);
    CHECK(f < in);
}

TEST(the_mirror_is_written_once_a_swap) {
    // finding 3: lend_mirror copies all three k itself; inside the k loop it was
    // three host copies a swap (4.5 MiB instead of 1.5, 0.42 GiB instead of 0.14
    // over a 96-swap adapt) on the decode thread, for every tail-layer adapt
    Ops ops;
    std::vector<Swap> plan{one_swap(), one_swap(), one_swap()};
    llama_loan_book::PhaseState<Swap> book;
    book.pend  = plan;
    book.phase = 2;

    llama_loan_book::settle(book, ops);

    CHECK_EQ(ops.mirrors, 3);      // one a swap, three swaps -- R3's red case: 9
    CHECK_EQ(ops.copies_in, 9);
}

TEST(a_polled_phase_one_leaves_phase_two_behind_a_fence) {
    // progress(): the phase-1 copies landed, so its tables change; the phase 2
    // copies are due and must wait for the graphs that still read the victims
    Ops ops;
    std::vector<Swap> plan{one_swap()};
    llama_loan_book::PhaseState<Swap> book;
    book.pend   = plan;
    book.phase  = 1;
    book.issued = true;

    CHECK(llama_loan_book::progress_phase(book, ops));
    CHECK_EQ(book.phase, 2);
    CHECK(!book.issued);
    const int out = at(ops.log, "tables_out");
    const int f   = at(ops.log, "fence");
    CHECK(out >= 0 && f >= 0 && out < f);
}

TEST(a_polled_phase_two_clears_only_after_its_tables) {
    // progress(): the newcomers landed; the tables say so, the swap is checked
    // (the cache's log), then the plan clears
    Ops ops;
    std::vector<Swap> plan{one_swap()};
    llama_loan_book::PhaseState<Swap> book;
    book.pend   = plan;
    book.phase  = 2;
    book.issued = true;
    book.check  = true;

    CHECK(llama_loan_book::progress_phase(book, ops));
    CHECK_EQ(ops.expert_of[0], 2);
    CHECK_EQ(ops.checks, 1);
    CHECK_EQ(book.phase, 0);
    CHECK(book.pend.empty());
}

TEST(a_phase_in_flight_changes_nothing_when_polled) {
    Ops ops;
    std::vector<Swap> plan{one_swap()};
    llama_loan_book::PhaseState<Swap> book;
    book.pend   = plan;
    book.phase  = 2;
    book.issued = true;
    ops.landed  = false;   // the copies have not landed

    CHECK(!llama_loan_book::progress_phase(book, ops));
    CHECK_EQ(book.phase, 2);
    CHECK_EQ(book.pend.size(), 1u);
    CHECK(ops.log.empty());
}

TEST(the_destructor_list_gets_no_tail_entry) {
    // finding 7 (second review), third review's other half: the data test above
    // proves take_tail's rule, but the list itself is built inside the cache's
    // init -- a function a unit build cannot call. This reads that init and holds
    // the invariant there: between the fixed list and its first use no entry for
    // the tail buffer is added (the bug: `bufs.push_back(b_lend)` after the call,
    // a double free of the loan's buffer and a leak of the live one). The
    // destructor's own GGML_ASSERT is the runtime half, on the card. The check
    // does not depend on one spelling of the call: any take_tail( counts, and
    // only the tail buffer's own names count as an entry.
    std::ifstream f(std::string(ARCINT_LLAMA_SRC_DIR) + "/src/llama-expert-cache.cpp");
    CHECK(f.good());
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) lines.push_back(l);
    int start = -1, end = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (start < 0 && lines[i].find("bufs = { bslots, bbank }") != std::string::npos) start = (int) i;
        if (start >= 0 && end < 0 && lines[i].find("ggml_backend_buffer_set_usage(bslots") != std::string::npos) end = (int) i;
    }
    CHECK(start >= 0);
    CHECK(end > start);
    bool tail_entry = false;
    bool took_tail  = false;
    for (int i = start; i < end; ++i) {
        const std::string & l = lines[(size_t) i];
        if (l.find("take_tail(") != std::string::npos) took_tail = true;
        // the tail buffer's own names: `b_lendbank` is the lend bank, a bufs entry
        if (l.find("push_back(b_lend)") != std::string::npos ||
            l.find("push_back(b_tail)") != std::string::npos) tail_entry = true;
    }
    CHECK(took_tail);     // the ownership hand-off is there at all
    CHECK(!tail_entry);   // R7's red case: an entry for the tail buffer
}

// ---- the lend selection: from scratch over the tail, no contiguity required ----

TEST(a_tail_with_no_lendable_layers_lends_nothing) {
    // the first card run's model: every down is IQ4_NL and no_split marked every
    // layer non-lendable; the selection left them `lendable` and made tensors in
    // a null tail context (the segfault). From scratch, nothing chosen.
    std::vector<llama_loan_book::LendCandidate> c(3, {false, 100});
    const auto p = llama_loan_book::choose_lend_layers(c, 8ull << 30);
    CHECK(p.chosen.empty());
    CHECK_EQ(p.bytes, 0);
    CHECK(p.why != nullptr);
    CHECK(p.why != nullptr && std::string(p.why).find("lendable") != std::string::npos);
}

TEST(a_loan_smaller_than_one_layer_lends_nothing) {
    // a configured budget under one layer's slot bytes: the loan must log why and
    // run off, not create a tail context with no tensors.
    std::vector<llama_loan_book::LendCandidate> c{{true, 4096}};
    const auto p = llama_loan_book::choose_lend_layers(c, 4095);
    CHECK(p.chosen.empty());
    CHECK(p.why != nullptr);
    CHECK(p.why != nullptr && std::string(p.why).find("smaller than one") != std::string::npos);
}

TEST(a_mixed_set_skips_a_non_lendable_layer_and_a_layer_that_does_not_fit) {
    // the set need not be contiguous: the separate tail buffer is a tensor list.
    // From the tail: 3 fits; 2 is not lendable and is skipped; 1 fits; 0 does not
    // fit the remaining budget. {1, 3} -- R12's red case: break at 2 -> {3} only.
    std::vector<llama_loan_book::LendCandidate> c{{true, 100}, {true, 100}, {false, 0}, {true, 100}};
    const auto p = llama_loan_book::choose_lend_layers(c, 250);
    CHECK_EQ(p.chosen, std::vector<int>({1, 3}));
    CHECK_EQ(p.bytes, 200);
    CHECK(p.why == nullptr);
}

TEST(a_loan_skips_an_oversized_tail_layer_and_takes_an_earlier_one) {
    // budget under the tail layer but enough for an earlier smaller one: skip,
    // then continue -- R12's other half: break at the tail would lend nothing.
    std::vector<llama_loan_book::LendCandidate> c{{true, 50}, {true, 100}};
    const auto p = llama_loan_book::choose_lend_layers(c, 50);
    CHECK_EQ(p.chosen, std::vector<int>({0}));
    CHECK_EQ(p.bytes, 50);
}

// ---- the poison policy (operator 2026-10-29: poison-exit) ----

namespace {
// the llama log route: a llama_log_set callback as the unit sets one (arcint
// routes libllama's lines into its own log), writing to the pipe the parent reads
void ln_capture_log(ggml_log_level, const char * text, void *) {
    const std::string s = std::string("LOG:") + text;
    (void) !write(STDERR_FILENO, s.data(), s.size());
}
// a C++ static destructor's stand-in: with std::exit it runs, with std::_Exit it
// does not (the fifth review's finding: exit can hang under the mutex)
void ln_atexit_marker() {
    const char * s = "ATEXIT ";
    (void) !write(STDERR_FILENO, s, 7);
}
// round 8's drain-before-exit: the registered drain runs before _Exit
void ln_drain_marker() {
    const char * s = "DRAIN ";
    (void) !write(STDERR_FILENO, s, 6);
}
// round 10: drain_now() runs the registered drain (a counter for the behavioural case)
int g_drain_calls = 0;
void ln_count_drain() { ++g_drain_calls; }
}  // namespace

TEST(the_poison_policy_exits_non_zero_with_the_cause) {
    // llama_loan_book::poison_exit is the one policy function: one ERROR line
    // naming the nature and a non-zero exit, so the unit manager restarts clean.
    // The child must exit non-zero (R9's red case: the exit removed -> _exit(0)),
    // carry the line both through the llama log callback and on stderr (R9's
    // log_line red case), and run no static destructor (R9's std::exit red case,
    // the ATEXIT marker).
    int fds[2];
    CHECK_EQ(pipe(fds), 0);
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        llama_log_set(ln_capture_log, nullptr);
        std::atexit(ln_atexit_marker);
        llama_loan_book::register_drain(ln_drain_marker);
        setenv("LLAMA_SLOT_LOAN_EXIT_IDLE_MS", "0", 1);   // round 12: this case is the policy, not the idle
        llama_loan_book::poison_exit("the refill did not verify: 3 of 246 rows differ from the file over 2 lent layers (246 slots)");
        _exit(0);   // unreachable while the policy exits
    }
    close(fds[1]);
    std::string msg;
    char buf[512];
    for (ssize_t n; (n = read(fds[0], buf, sizeof(buf))) > 0;) msg.append(buf, (size_t) n);
    close(fds[0]);
    int st = 0;
    CHECK_EQ(waitpid(pid, &st, 0), pid);
    CHECK(WIFEXITED(st));          // an exit, not a signal
    CHECK(WEXITSTATUS(st) != 0);   // R9's red case: 0
    CHECK(msg.find("LOG:ERROR") != std::string::npos);   // the log route (R9: log_line removed)
    CHECK(msg.find("ERROR") != std::string::npos);
    CHECK(msg.find("unusable") != std::string::npos);
    CHECK(msg.find("3 of 246") != std::string::npos);   // the counts survive
    CHECK(msg.find("ATEXIT") == std::string::npos);     // R9: std::exit runs it
    CHECK(msg.find("DRAIN") != std::string::npos);      // R24's red case: the drain not run
}

// ---- round 12: the exit's idle, the knob's bound, the drain line's level ----

namespace {
// a level-carrying capture: "LOG3:" is WARN, "LOG4:" ERROR (ggml.h's enum), so a
// case can tell the idle line from the poison line
void ln_capture_level(ggml_log_level lvl, const char * text, void *) {
    const std::string s = std::string("LOG") + std::to_string((int) lvl) + ":" + text;
    (void) !write(STDERR_FILENO, s.data(), s.size());
}
}  // namespace

TEST(the_exit_waits_for_the_rings_to_stop) {
    // Round 12: after the drain nothing is queued, but the runtime's
    // direct-submission rings are still live until its own controller sees the
    // idle engine; std::_Exit runs no destructor, so they are live when xe closes
    // the fd (round 11, measured-here: MUTATE=4 exits microseconds after the last
    // GPU work and faults 3 of 3; MUTATE=1 is idle for the verify's read-back and
    // memcmp and never faults). poison_exit therefore waits exit_idle_ms after the
    // drain and before _Exit, bounded by the watchdog.
    // R40's red case: the sleep removed, or exit_idle_ms() 0 for every value (the
    // elapsed time falls back under the idle and the wait line vanishes).
    // R42's red case: the wait placed before drain() -- the DRAIN marker then
    // appears after the wait line, while the queues still hold work.
    int fds[2];
    CHECK_EQ(pipe(fds), 0);
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        llama_log_set(ln_capture_level, nullptr);
        llama_loan_book::register_drain(ln_drain_marker);
        setenv("LLAMA_SLOT_LOAN_EXIT_IDLE_MS", "250", 1);
        llama_loan_book::poison_exit("the resident re-reserve at the loan's return failed at 2048 tokens");
        _exit(0);   // unreachable while the policy exits
    }
    close(fds[1]);
    // the clock starts before the read: the pipe reaches EOF when the child exits,
    // so a clock taken after it measures nothing
    const auto t0 = std::chrono::steady_clock::now();
    std::string msg;
    char buf[512];
    for (ssize_t n; (n = read(fds[0], buf, sizeof(buf))) > 0;) msg.append(buf, (size_t) n);
    close(fds[0]);
    int st = 0;
    CHECK_EQ(waitpid(pid, &st, 0), pid);
    const long ms = (long) std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 70);
    CHECK(ms >= 200);   // R40's red case: no wait (250 ms asked, fork/wait overhead aside)
    // the wait line is at WARN, names the knob and the value, and follows the drain
    const size_t drain_at = msg.find("DRAIN");
    const size_t wait_at  = msg.find("the exit waits 250 ms");
    CHECK(drain_at != std::string::npos);
    CHECK(wait_at != std::string::npos);   // R40's red case: no line
    CHECK(msg.find("LOG3:") != std::string::npos);   // the WARN route (arcint shows WARN and above)
    CHECK(msg.find("LLAMA_SLOT_LOAN_EXIT_IDLE_MS") != std::string::npos);
    CHECK(drain_at < wait_at);   // R42's red case: the wait before the drain
}

TEST(the_exit_idle_knob_is_bounded) {
    // Round 12: the knob's parse and bound, as plain data. The default is the
    // measured passing idle (MUTATE=1) with margin; "0"/"off" is the red; a
    // value above the bound is clamped so a wait cannot eat the watchdog's room;
    // a typo or a negative is the default, never a silent 0 (that would turn the
    // red arm's own knob into the healthy one). R41's red case: the clamp removed
    // (a 99999 ms wait holds the watchdog's bound but the reason it exists).
    CHECK_EQ(llama_loan_book::kDefaultExitIdleMs, 2000);
    CHECK_EQ(llama_loan_book::kWatchdogBoundMs, 5000);
    CHECK(llama_loan_book::kExitIdleMaxMs < llama_loan_book::kWatchdogBoundMs - 500);   // drain room left
    CHECK_EQ(llama_loan_book::exit_idle_ms(nullptr), llama_loan_book::kDefaultExitIdleMs);
    CHECK_EQ(llama_loan_book::exit_idle_ms(""), llama_loan_book::kDefaultExitIdleMs);
    CHECK_EQ(llama_loan_book::exit_idle_ms("0"), 0);      // the red arm
    CHECK_EQ(llama_loan_book::exit_idle_ms("off"), 0);
    CHECK_EQ(llama_loan_book::exit_idle_ms("250"), 250);
    CHECK_EQ(llama_loan_book::exit_idle_ms("99999"), llama_loan_book::kExitIdleMaxMs);   // R41's red case
    CHECK_EQ(llama_loan_book::exit_idle_ms("abc"), llama_loan_book::kDefaultExitIdleMs);
    CHECK_EQ(llama_loan_book::exit_idle_ms("-5"), llama_loan_book::kDefaultExitIdleMs);
}


// ---- round 13: the idle's second line, the watchdog's own line, the off words ----

TEST(the_exit_reports_the_idle_it_actually_waited) {
    // Round 13, review low: the wait line is written BEFORE the sleep, so the log
    // says a wait was asked for, not that one happened -- and the watchdog can cut
    // the idle silently (no line is ever printed after it). poison_exit therefore
    // prints a second line after the sleep with the elapsed time.
    // R44's red case: the second line removed, or written before the sleep (the
    // elapsed then falls under the asked value and the order check fails).
    int fds[2];
    CHECK_EQ(pipe(fds), 0);
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        llama_log_set(ln_capture_level, nullptr);
        llama_loan_book::register_drain(ln_drain_marker);
        setenv("LLAMA_SLOT_LOAN_EXIT_IDLE_MS", "300", 1);
        llama_loan_book::poison_exit("the resident re-reserve at the loan's return failed at 2048 tokens");
        _exit(0);   // unreachable while the policy exits
    }
    close(fds[1]);
    std::string msg;
    char buf[512];
    for (ssize_t n; (n = read(fds[0], buf, sizeof(buf))) > 0;) msg.append(buf, (size_t) n);
    close(fds[0]);
    int st = 0;
    CHECK_EQ(waitpid(pid, &st, 0), pid);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 70);
    const size_t wait_at = msg.find("the exit waits 300 ms");
    const size_t done_at = msg.find("the exit's idle finished after");
    CHECK(wait_at != std::string::npos);
    CHECK(done_at != std::string::npos);   // R44's red case: no second line
    CHECK(wait_at < done_at);              // R44's red case: the line before the sleep
    // the elapsed is the measured sleep, at least the asked value
    const size_t ms_at = msg.find("after ", done_at);
    CHECK(ms_at != std::string::npos);
    const long ms = std::atol(msg.c_str() + ms_at + 6);
    CHECK(ms >= 300);                      // R44's red case: the asked value printed for an unslept wait
    CHECK(msg.rfind("LOG3:", done_at) != std::string::npos);   // the WARN route, like the wait line
}

TEST(the_idle_knob_reads_the_off_words_in_any_case) {
    // Round 13, review low: "off" was case-sensitive, so OFF or False silently
    // meant the default idle and the red arm was far slower than the arm asked
    // for. off/OFF/false/False/no/NO (any case) are 0; everything else keeps the
    // rule (a typo is the default, never a silent 0).
    // R45's red case: the case-fold removed (only the lowercase word is the red).
    for (const char * v : {"off", "OFF", "Off", "false", "FALSE", "False", "no", "NO", "No"}) {
        CHECK_EQ(llama_loan_book::exit_idle_ms(v), 0);
    }
    CHECK_EQ(llama_loan_book::exit_idle_ms("off "), llama_loan_book::kDefaultExitIdleMs);   // not an off word
    CHECK_EQ(llama_loan_book::exit_idle_ms("none"), llama_loan_book::kDefaultExitIdleMs);
    CHECK_EQ(llama_loan_book::exit_idle_ms("250"), 250);
}

TEST(the_watchdog_fires_with_one_async_safe_line) {
    // Round 13, review low: the watchdog cut the idle silently -- _Exit(70) with no
    // line, so a log looked like a clean policy exit that never finished. It now
    // writes ONE line with write(2) (async-signal-safe: no snprintf, no log
    // callback, no flush; the drain may be hung and the log route on the mutex).
    // The drain here hangs, so the fired line is the watchdog's, not the idle's.
    // R46's red case: the write removed (or a plain write replaced by log_warn).
    int fds[2];
    CHECK_EQ(pipe(fds), 0);
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        llama_log_set(ln_capture_level, nullptr);
        llama_loan_book::register_drain([] {
            const char * s = "DRAIN-HUNG ";
            (void) !write(STDERR_FILENO, s, 12);
            std::this_thread::sleep_for(std::chrono::seconds(60));   // the hung driver
        });
        setenv("LLAMA_SLOT_LOAN_EXIT_IDLE_MS", "0", 1);
        llama_loan_book::poison_exit("the resident re-reserve at the loan's return failed at 2048 tokens");
        _exit(0);   // unreachable while the policy exits
    }
    close(fds[1]);
    const auto t0 = std::chrono::steady_clock::now();
    std::string msg;
    char buf[512];
    // a bounded read loop: the pipe closes when the watchdog exits; if it never
    // does, the test fails instead of hanging the suite
    int st = 0;
    pid_t got = 0;
    for (;;) {
        const pid_t w = waitpid(pid, &st, WNOHANG);
        if (w == pid) { got = w; break; }
        if ((long) std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count() > 12000) {
            kill(pid, SIGKILL);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for (ssize_t n; (n = read(fds[0], buf, sizeof(buf))) > 0;) msg.append(buf, (size_t) n);
    close(fds[0]);
    CHECK_EQ(got, pid);   // the watchdog exits on its own
    const long ms = (long) std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 70);   // round 9's rule: an exit, not a signal
    CHECK(ms >= 4500 && ms < 11000);   // the 5 s bound, not the hung drain
    CHECK(msg.find("DRAIN-HUNG") != std::string::npos);   // the drain began
    CHECK(msg.find("the exit's watchdog fired") != std::string::npos);   // R46's red case: no line
    CHECK(msg.find("the exit's idle finished") == std::string::npos);   // the idle never ran
}

// ---- round 13: the loan-KL driver's rules (contrib/llama-loan-kld) ----

TEST(the_kld_driver_takes_the_loan_above_the_resident_chunk) {
    // Round 13's KL gate: the driver asks the context for a loan only when its
    // chunk's tokens beat the resident chunk the loan-off arm's context reserves
    // (2,048). A 512-token KL window (the 16 x 512 arms) would never cross it, so
    // that window would measure nothing; the gate uses a chunk above the floor.
    // R47's red case: `>=` instead of `>` (a 2,048-token chunk asks for a loan
    // that the context's own wanted() refuses).
    CHECK(loan_kld_book::loan_wanted(8192));
    CHECK(loan_kld_book::loan_wanted(2049));
    CHECK(!loan_kld_book::loan_wanted(2048));   // R47's red case
    CHECK(!loan_kld_book::loan_wanted(512));
    CHECK_EQ(loan_kld_book::chunk_tokens(8192), 8192u);
    CHECK_EQ(loan_kld_book::kResidentChunk, 2048u);   // arcint's slot_loan.h floor
    // the capture's row window equals llama's perplexity loop
    CHECK_EQ(loan_kld_book::first_row(8192), 4096);
    CHECK_EQ(loan_kld_book::n_rows(8192), 8192 - 1 - 4096);
    CHECK_EQ(loan_kld_book::row_width(1000), 2 * 500 + 4);
    CHECK_EQ(loan_kld_book::row_width(1001), 2 * 501 + 4);
}

TEST(the_kld_driver_begins_before_its_decode_and_ends_after) {
    // Round 13: the loan's scope in the driver is arcint's Scope around a prefill
    // chunk -- begin before the chunk's first batch, end after its last, before
    // the next chunk's decode. R48's red case (text order, one line moved): the
    // begin after the decode, or the end before it.
    std::ifstream f(std::string(ARCINT_SOURCE_DIR) + "/contrib/llama-loan-kld/loan_kld.cpp");
    CHECK(f.good());
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) lines.push_back(l);
    auto pos = [&](const char * needle) {
        for (size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].find(needle) != std::string::npos) return (long) i;
        }
        return -1L;
    };
    const long begin = pos("llama_context_slot_loan_begin(ctx");
    const long decode = pos("llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())");
    const long end   = pos("llama_context_slot_loan_end(ctx");
    CHECK(begin >= 0);
    CHECK(decode >= 0);
    CHECK(end >= 0);
    CHECK(begin < decode);   // R48's red case: the begin after the decode
    CHECK(decode < end);     // R48's red case: the end before the decode
    // the run is refused when a chunk ran without a loan (a capture that is not a
    // loan arm would pass a KL check without having exercised the branch)
    CHECK(pos("llama_context_slot_loan_out(ctx") >= 0);
    CHECK(pos("ran without a loan") >= 0);
}

// ---- the call sites the helpers alone cannot prove ----

// a source read of the patch's own function, between two landmarks; false when
// either landmark is missing (the shape moved and the case must be rewritten)
// a scan of a file of the pinned tree, given its path under the tree (llama/src/
// for the patch's own files, ggml/src/ggml-opencl/ for the backend this round adds
// the refill's entry points to)
bool scan_between(const char * rel, const char * a, const char * b, const std::vector<std::string> & want) {
    std::ifstream f(std::string(ARCINT_LLAMA_SRC_DIR) + "/" + rel);
    if (!f.good()) return false;
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) lines.push_back(l);
    int start = -1, end = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (start < 0 && lines[i].find(a) != std::string::npos) start = (int) i;
        if (start >= 0 && end < 0 && lines[i].find(b) != std::string::npos) end = (int) i;
    }
    if (start < 0 || end <= start) return false;
    for (const std::string & w : want) {
        bool found = false;
        for (int i = start; i < end && !found; ++i) found = lines[(size_t) i].find(w) != std::string::npos;
        if (!found) return false;
    }
    return true;
}

bool has_between(const char * file, const char * a, const char * b, const std::vector<std::string> & want,
                 std::vector<std::string> * lines_out = nullptr) {
    std::ifstream f(std::string(ARCINT_LLAMA_SRC_DIR) + "/src/" + file);
    if (!f.good()) return false;
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) lines.push_back(l);
    if (lines_out) *lines_out = lines;
    int start = -1, end = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (start < 0 && lines[i].find(a) != std::string::npos) start = (int) i;
        if (start >= 0 && end < 0 && lines[i].find(b) != std::string::npos) end = (int) i;
    }
    if (start < 0 || end <= start) return false;
    for (const std::string & w : want) {
        bool found = false;
        for (int i = start; i < end && !found; ++i) found = lines[(size_t) i].find(w) != std::string::npos;
        if (!found) return false;
    }
    return true;
}

TEST(the_drain_line_is_at_warn) {
    // Round 12, review low: round 11's drain line was GGML_LOG_INFO and arcint's
    // log shows WARN and above, so the card never saw it (measured-here, round 11:
    // the line never appeared). The backend's drain prints it at WARN, and it is
    // still inside the drain itself. R43's red case: GGML_LOG_INFO.
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "static void ggml_backend_opencl_drain(void) {",
                       "// Reads n small plain tensors",
                       {"GGML_LOG_WARN(\"ggml_opencl: drain: %d device(s)", "devices"}));
}

TEST(the_plain_moe_path_is_unchanged_without_an_expert_cache) {
    // the dense agent and the coder (all experts resident, no expert cache) take
    // build_moe_ffn's else branch: the loan's tables and the lent branch only
    // exist when llama_expert_cache_lookup() finds the layer's down tensor. The
    // dense agent has no cache at all, the coder's cache is empty.
    CHECK(has_between("llama-graph.cpp", "llama_expert_cache_layer * ecl =", "if (!weight_before_ffn)",
                      {"} else {", "build_experts(cur, up_exps, gate_exps, gate_up_exps, down_exps, selected_experts)"}));
}

TEST(the_lend_loop_skips_a_cache_without_a_lend) {
    // finding 5's call site: the `continue` in slot_loan_begin's lend loop. A
    // cache with lend_bytes 0 (the other card under a layer split) returns false
    // from lend(); without the skip the AND refuses a good loan. The helper test
    // above stays green when the call site loses its continue, so this reads it.
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_begin", "bool llama_context::slot_loan_end",
                      {"if (!llama_loan_book::loan_cache(ec->lend_bytes())) continue;"}));   // R10's red case
}

TEST(the_refill_copies_regions_from_the_bank_without_staging) {
    // Round 7: the lend bank holds the lent rows in the CARD'S layout (set_tensor /
    // set_slices built it at load, the mirrors copy it), so a refill is a device-side
    // region copy from USM host memory, one enqueue a tensor, no host staging and no
    // conversion -- the 2.7 s the card measured was the staging copies, the plane and
    // GGUF conversions and the wait a layer. Red case R17: the staged path back
    // (get_slices into host rows then set_slices) -> this scan fails.
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::refill", "void llama_expert_cache::rebind",
                      {"lend_write(L.slot[k], 0, L.lend[k], 0, L.n_slots)",
                       "lend_zero(L.slot[k], llama_loan_book::refill_zero_row(L.n_slots)",
                       "lend_read(L.slot[k]", "lend_read(L.lend[k]"}));
}

TEST(the_refill_builds_the_split_extra_before_it_copies) {
    // Fifth review, finding 1, in its round-7 form: drop_tail_buffer() clears
    // `extra`, the re-allocated tail buffer's init_tensor installs the PLAIN
    // ggml_tensor_extra_cl, and IQ4_NL's card layout needs the d/q sub-buffers.
    // set_tensor used to rebuild it with a whole write (the temp device buffer and
    // the conversion kernel, round 7's cost); lend_write now builds it without
    // bytes, and the `iq4_nl_split_tensors` set is how it knows the tensor lacks it
    // (set_tensor inserts on its whole write). Red case R18: the build block
    // removed -> the region cast reads a plain extra.
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "// The prefill-slot-loan's refill (docs/campaigns/prefill-slot-loan-design.md",
                       "static void ggml_backend_opencl_lend_zero(ggml_tensor * dst, int64_t row) {",
                      {"bctx->iq4_nl_split_tensors.count(dst) == 0",
                       "ggml_cl_iq4_nl_extra(backend_ctx, bctx, dst, 0)",
                       "bctx->iq4_nl_split_tensors.insert(dst)"}));
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "ggml_cl_iq4_nl_extra(backend_ctx, ctx, tensor, offset);",
                       "GGML_UNUSED(size_d); GGML_UNUSED(size_q);",
                       {"ctx->iq4_nl_split_tensors.insert(tensor)"}));
}

TEST(the_refill_check_defaults_to_the_lend_bank) {
    // Round 7: the default compares every row with the lend bank in the card's
    // layout -- byte for byte, no conversion, no read of the GGUF. The bank's
    // provenance to the file is the load's lend bank check and the mirrors. Red
    // case R19: the default becomes the file (the 6.2 GiB file read a loan).
    CHECK(llama_loan_book::verify_ref(nullptr) == llama_loan_book::VerifyRef::Bank);
    CHECK(llama_loan_book::verify_ref("") == llama_loan_book::VerifyRef::Bank);
    CHECK(llama_loan_book::verify_ref("1") == llama_loan_book::VerifyRef::Bank);
    CHECK(llama_loan_book::verify_ref("bank") == llama_loan_book::VerifyRef::Bank);   // not atoi 0 -> off
    CHECK(llama_loan_book::verify_ref("0") == llama_loan_book::VerifyRef::Off);
    CHECK(llama_loan_book::verify_ref("off") == llama_loan_book::VerifyRef::Off);
}

TEST(the_refill_check_can_be_the_file_for_a_gate_arm) {
    // the campaign's gate form stays reachable, and it is never silently off
    CHECK(llama_loan_book::verify_ref("file") == llama_loan_book::VerifyRef::File);
    CHECK(llama_loan_book::verify_ref("typo") != llama_loan_book::VerifyRef::Off);
    // the context's on/off and the cache's mode are one rule: "file" is on
    CHECK(has_between("llama-context.cpp", "static const bool verify_rows", "bool ok = back",
                      {"llama_loan_book::verify_ref(getenv(\"LLAMA_SLOT_LOAN_VERIFY\")) != llama_loan_book::VerifyRef::Off"}));
}

TEST(the_lend_bank_is_checked_at_load) {
    // Round 7's exactness chain: the refill compares the slots with the bank, so the
    // bank's provenance to the file is proven at load, over EVERY lendable row, and
    // a bank that differs turns the loan off instead of being copied back a prompt.
    // Red case R20: the block removed.
    CHECK(has_between("llama-expert-cache.cpp", "Round 7 (design §15): the refill's default check",
                      "g_registry[L.src[2]] = &L;",
                      {"lend bank check", "get_slices(L.lend[k], img.data(), 0, L.n_slots)",
                       "read_expert(L, L.expert_of[s], k, ref.data())", "lend_set_.clear()"}));
}

TEST(a_size_check_leaves_a_peak_margin) {
    // Round 9, review item 1: round 8 read ggml_backend_dev_memory, whose OpenCL
    // form is `free = global_mem_size - 1 GiB` -- a constant (the source scan
    // below reads it), so the 384 MiB headroom check could never fail. The rule is
    // round 7's over real plan bytes: need + the peak reserve <= the freed lend
    // bytes. R22/R33's red case: the reserve term dropped from size_fits.
    // Round 17 adds the fixed card margin: need + both margins <= the freed bytes.
    const size_t mib = 1u << 20;
    const size_t m = llama_loan_book::kLendMargin;
    CHECK(llama_loan_book::kPeakReserve >= 256u * mib && llama_loan_book::kPeakReserve <= 512u * mib);
    CHECK(llama_loan_book::size_fits(6400 * mib - m, 6400 * mib));    // exactly both margins clear
    CHECK(!llama_loan_book::size_fits(6400 * mib - m + 1, 6400 * mib)); // one byte into the margins
    CHECK(!llama_loan_book::size_fits(6400 * mib, 6400 * mib));
    CHECK(!llama_loan_book::size_fits(SIZE_MAX, 6400 * mib));                 // a failed sizing refuses
    CHECK(llama_loan_book::size_fits(0, m));      // no prompt buffers, the margins fit
    CHECK(!llama_loan_book::size_fits(0, m - 1)); // a lend under the margins is no underflow

    // the freed bytes are what the loan can use; the margins are in their arithmetic
    CHECK(has_between("llama-loan-book.h", "inline bool size_fits", "// Round 8, review item 2",
                      {"need <= lend - kLendMargin"}));
    // no device free figure in the loan's path: OpenCL's is a constant
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "static void ggml_backend_opencl_device_get_memory",
                       "static enum ggml_backend_dev_type",
                      {"opencl_extra_margin", "*total - opencl_extra_margin"}));
    CHECK(!has_between("llama-context.cpp", "auto put_back = [&]()", "if (!slot_loan_reserve(cap))",
                       {"ggml_backend_dev_memory", "headroom_fits"}));   // R33's red case: the fake free
    CHECK(has_between("llama-context.cpp", "if (!llama_loan_book::size_fits(need, lend))", "bool llama_context::slot_loan_end",
                      {"the fixed margin is %.3f GiB", "kLendMargin"}));
}

// ---- round 17: the fixed-margin rule (the lend grows layer by layer) ----

TEST(a_short_margin_grows_the_lend_by_one_tail_layer) {
    // The ask-bounded pick is tail layer 2 (1 GiB); with no prompt buffers the
    // fixed margin needs 1.375 GiB, so exactly ONE earlier candidate (layer 1) is
    // grown in and the growth stops. R58's red case: extend_lend_layers returns
    // the ask unchanged -> chosen {2} only.
    const int64_t GiB = (int64_t) (1u << 30);
    std::vector<llama_loan_book::LendCandidate> c{{true, GiB}, {true, GiB}, {true, GiB}};
    const auto ask = llama_loan_book::choose_lend_layers(c, (size_t) GiB);
    CHECK_EQ(ask.chosen, std::vector<int>({2}));
    const auto p = llama_loan_book::extend_lend_layers(c, ask, 0);
    CHECK_EQ(p.chosen, std::vector<int>({1, 2}));
    CHECK_EQ(p.bytes, GiB * 2);
    CHECK(p.why == nullptr);
    CHECK(llama_loan_book::size_fits(0, (size_t) p.bytes));
}

TEST(the_tail_running_out_refuses_the_growth) {
    // Two candidates, the ask picked layer 1 (1 GiB). With need 1 GiB the margin
    // is 2 GiB - 1 GiB = 1 GiB, under the fixed 1.375 GiB; layer 0's 1 GiB does
    // not reach it either, so the pick names the reason and the sizing refuses.
    // R59's red case: `why` left null when the tail is exhausted (the caller then
    // reads a fit out of an unfitting pick).
    const int64_t GiB = (int64_t) (1u << 30);
    std::vector<llama_loan_book::LendCandidate> c{{true, GiB}, {true, GiB}};
    const auto ask = llama_loan_book::choose_lend_layers(c, (size_t) GiB);
    const auto p = llama_loan_book::extend_lend_layers(c, ask, (size_t) GiB);
    CHECK_EQ(p.chosen, std::vector<int>({0, 1}));
    CHECK(p.why != nullptr);
    CHECK(p.why != nullptr && std::string(p.why).find("margin") != std::string::npos);
    CHECK(!llama_loan_book::size_fits((size_t) GiB, (size_t) p.bytes));
}

TEST(a_grown_lend_bank_is_filled_by_the_whole_tensor_path) {
    // Round 18, review item 1 and item 5: extend_lend fills a grown layer's lend
    // bank FROM THE FILE with one whole-tensor set per region, the path init uses
    // for the ask's pick. The OpenCL USM branch of set_tensor is the only writer
    // that builds IQ4_NL's usm_split; set_slices reads that map and throws on the
    // fresh bank, on every layer of the served file (all ffn_down_exps IQ4_NL).
    // The bank's provenance is then the file (item 3). A growth also adds its
    // bank to the host-bank figure (item 5). R61's red case: set_tensor replaced
    // by set_slices; the bank figure's line removed for item 5.
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::extend_lend",
                      "void llama_expert_cache::drop_tail_buffer() {",
                      {"read_expert(L, L.expert_of[s]", "ggml_backend_tensor_set(L.lend[k], img.data(), 0,",
                       "n_bank_bytes += ggml_backend_buffer_get_size(banks[a])"}, &lines));
    int start = -1, slices = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (start < 0 && lines[i].find("bool llama_expert_cache::extend_lend") != std::string::npos) start = (int) i;
        if (start >= 0 && slices < 0 && lines[i].find("set_slices(L.lend") != std::string::npos) slices = (int) i;
    }
    CHECK(slices < 0);   // a fresh bank has no usm_split: set_slices would .at()
}

TEST(a_growth_throw_takes_the_resident_refusal) {
    // Round 18, review item 1: the growth runs inside the sizing's try, so a
    // throw (a bank allocation, an image, a driver call) follows the catch into
    // the resident re-reserve and the rebind instead of reaching Scope::begin on
    // the scheduler sched_new replaced. extend_lend itself stages every bank and
    // frees them on failure, so no host USM bank leaks on an attempt. R62's red
    // case: the growth block moved below the catch (or the catch's staged-bank
    // free removed).
    CHECK(has_between("llama-context.cpp", "bool sized = true;", "catch (const std::exception & e)",
                      {"ec->extend_lend(demand, beside, card, need)"}));
    CHECK(!has_between("llama-context.cpp", "if (!sized) {", "if (!llama_loan_book::size_fits(need, lend))",
                       {"ec->extend_lend"}));   // not in the refusal block
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::extend_lend",
                      "void llama_expert_cache::drop_tail_buffer() {",
                      {"catch (...)", "for (ggml_backend_buffer_t x : banks)",
                       "ggml_backend_buffer_free(x)"}));
    CHECK(has_between("llama-context.cpp", "if (!sized) {", "if (!llama_loan_book::size_fits(need, lend))",
                      {"slot_loan_reserve(cparams.n_ubatch_reserve)", "ec->rebind(sched.get())"}));
}

TEST(a_grown_layer_is_not_freed_before_the_lend_is_published) {
    // Round 18, review item 2: extend_lend used to free the grown layer's spare
    // slot buffer before lend() published the tables. If the lend's table check
    // then failed, lend_undo put the slots back and the request resumed on the
    // released cl_mem. drop_tail_buffer frees the spare only after the drain,
    // after nulling every lent tensor and only once the lend is out. R63's red
    // case: the free block back in extend_lend.
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::extend_lend",
                      "void llama_expert_cache::drop_tail_buffer() {", {}, &lines));
    int starts = -1, ends = -1, free_in_extend = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (starts < 0 && lines[i].find("bool llama_expert_cache::extend_lend") != std::string::npos) starts = (int) i;
        if (starts >= 0 && ends < 0 && lines[i].find("void llama_expert_cache::drop_tail_buffer() {") != std::string::npos) ends = (int) i;
        if (starts >= 0 && ends < 0 && free_in_extend < 0 &&
            lines[i].find("ggml_backend_buffer_free(spare_buf") != std::string::npos) free_in_extend = (int) i;
    }
    CHECK(free_in_extend < 0);   // nothing freed before the lend is published
    CHECK(has_between("llama-expert-cache.cpp", "void llama_expert_cache::drop_tail_buffer",
                      "bool llama_expert_cache::refill",
                      {"t->extra  = nullptr;", "ggml_backend_buffer_free(spare_buf[(size_t) i])"}));
}

TEST(a_failed_spare_allocation_nulls_every_freed_buffer) {
    // Round 18, review item 4: init's failed per-layer spare allocation freed
    // spare_buf[j] but left the pointers; the destructor frees every non-null
    // entry again. The failure path nulls each freed entry. R64's red case: the
    // null write removed.
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "spare_buf.assign(layers.size(), nullptr);",
                      "llama_loan_book::take_tail(bufs, b_tail, b_lend);",
                      {"ggml_backend_buffer_free(spare_buf[j])", "spare_buf[j] = nullptr;"}, &lines));
    int f = -1, n = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (f < 0 && lines[i].find("ggml_backend_buffer_free(spare_buf[j])") != std::string::npos) f = (int) i;
        if (f >= 0 && n < 0 && lines[i].find("spare_buf[j] = nullptr;") != std::string::npos) n = (int) i;
    }
    CHECK(f >= 0 && n > f);   // the pointer is cleared after the free
}


// ---- round 19: the card's own bound (resident + need against the card's total) ----

TEST(the_card_rule_bounds_resident_plus_need_against_the_card) {
    // Round 19, review item 1: the rule as implemented compared `need` against the
    // lent bytes only. The operator's margin is under the CARD's total:
    // `resident_after_lend + need <= card_total - card_margin`, with the resident
    // figure the process's own device buffers there (not a free query: OpenCL's
    // free is `total - 1 GiB`, a constant). Boundaries below.
    const size_t MiB = 1u << 20;
    const size_t GiB = 1u << 30;
    CHECK_EQ(llama_loan_book::kCardMargin, (size_t) GiB);           // the operator's 1 GiB
    const size_t total = 24480 * MiB;                               // the B60's total
    const size_t limit = total - llama_loan_book::kCardMargin;      // 23,456 MiB
    // resident after the lend + need exactly at the limit: fits; one byte over: not
    CHECK(llama_loan_book::card_fits(limit - 100 * MiB, 100 * MiB, total, llama_loan_book::kCardMargin, 0));
    CHECK(!llama_loan_book::card_fits(limit - 100 * MiB + 1, 100 * MiB, total, llama_loan_book::kCardMargin, 0));
    CHECK(!llama_loan_book::card_fits(limit, 1, total, llama_loan_book::kCardMargin, 0));
    // the lend is subtracted from the resident first
    CHECK(llama_loan_book::card_fits(2000 * MiB, 500 * MiB, total, llama_loan_book::kCardMargin, 1500 * MiB));
    CHECK(!llama_loan_book::card_fits(24000 * MiB, 1000 * MiB, total, llama_loan_book::kCardMargin, 1500 * MiB));
    // a lend larger than the resident does not underflow (it clamps to 0)
    CHECK(llama_loan_book::card_fits(500 * MiB, 500 * MiB, total, llama_loan_book::kCardMargin, 900 * MiB));
    // no card figure (0): the rule is not checked; a failed sizing never fits
    CHECK(llama_loan_book::card_fits(9999 * GiB, 9999 * GiB, 0, llama_loan_book::kCardMargin, 0));
    CHECK(!llama_loan_book::card_fits(0, SIZE_MAX, total, llama_loan_book::kCardMargin, 0));
    // a card smaller than the margin: a limit of 0, so only an empty need fits
    CHECK(!llama_loan_book::card_fits(0, 1, GiB / 2, llama_loan_book::kCardMargin, 0));
    CHECK(llama_loan_book::card_fits(0, 0, GiB / 2, llama_loan_book::kCardMargin, 0));

    // the recorded depth shape on the unit (arithmetic over measured figures, not
    // this round's measurement): the IQ3_S unit holds ~20,430 MiB of device
    // buffers with 11,600 MiB of slots (23,478 MiB peak at the resident chunk
    // minus the 3,048 MiB resident compute reserve, `measured-here`); the 6,400
    // MiB ask frees 6,349 MiB (6.20 GiB), and the recorded need at the 8,192 chunk
    // is 4,659 MiB (4.55 GiB, the MUTATE=3 arm) -> 14,081 + 4,659 = 18,740 MiB,
    // 4,716 MiB under the card's total minus the margin. So the new rule does not
    // grow at 6,400: the served shape stays round 16's 26 layers.
    const size_t resident = 20430 * MiB, lend = 6349 * MiB, need = 4659 * MiB;
    CHECK(llama_loan_book::card_fits(resident, need, total, llama_loan_book::kCardMargin, lend));
    CHECK(llama_loan_book::size_fits(need, lend));
    const size_t after_need = resident - lend + need;
    CHECK(after_need <= limit);
    CHECK_EQ(limit - after_need, (size_t) (4716 * MiB));   // the margin the rule leaves

    // the sizing reads the card's total (the device props' memory_total, the
    // global mem size) and the process's buffers, and logs both
    CHECK(has_between("llama-context.cpp", "bool sized = true;", "the lend: tables relayed",
                      {"ggml_backend_dev_get_props", "memory_total", "resident_on", "lend_on_card",
                       "loan_fits_card", "card_margin"}));
    CHECK(has_between("llama-context.cpp", "resident_on = [&]", "lend_on_card = [&]",
                      {"ggml_backend_buft_get_device", "model.memory_breakdown()",
                       "memory->memory_breakdown()", "ec->device_bytes()",
                       "ggml_backend_buffer_get_size(buf_output.get())"}));
    // the refusal WARN names used, the ggml sum, fdinfo, the allowance and the
    // card's total (round 22: used - fdinfo is information, not a reserve)
    CHECK(has_between("llama-context.cpp", "Round 22: the card's rule per card", "the lend: tables relayed",
                      {"fdinfo vram0", "ggml buffer sum", "fdinfo gtt", "used - fdinfo %.2f GiB",
                       "prefill growth allowance", "the card's total is %.2f GiB",
                       "the prompt's compute buffers need %.2f GiB"}));
    // the loan-out line carries the same figures, the kernel query's used and the
    // guard's fdinfo/gtt beside it (round 21; round 22: the allowance and used -
    // fdinfo, and the limit is the card's total)
    CHECK(has_between("llama-context.cpp", "LLAMA_LOG_WARN(\"%s: loan out at", "return true;",
                      {"the card's total is %.2f GiB", "ggml buffer sum %.2f GiB",
                       "fdinfo vram0 %.2f GiB", "fdinfo gtt %.2f GiB",
                       "used - fdinfo %.2f GiB, first touch %.2f GiB, prefill growth allowance %.2f GiB",
                       "allowance = %.2f GiB <= total"}));
    // R65's red case: card_fits dropping a term, or the growth's card check removed
    CHECK(has_between("llama-loan-book.h", "inline bool card_fits(size_t resident, size_t need, size_t card_total, size_t margin, size_t lend, size_t first_touch) {",
                      "// Round 8, review item 2",
                      {"card_total >= margin", "resident - lend", "card_total == 0", "need > limit",
                       "first_touch > limit - need", "after <= limit - need - first_touch"}));
    // round 22: the kernel rule's limit is the card's total and its one addend is
    // the allowance (the double count is gone)
    CHECK(has_between("llama-loan-book.h", "inline bool card_state_fits(size_t used, size_t total",
                      "// The off words", {"after <= total - need - first_touch - allowance",
                                           "allowance > total - need - first_touch"}));
    CHECK(!has_between("llama-loan-book.h", "inline bool card_state_fits(size_t used, size_t total",
                       "// The off words", {"total - kernel_reserve"}));
}

TEST(the_card_margin_knob_moves_the_margin_but_not_removes_it) {
    // Round 19: LLAMA_SLOT_LOAN_CARD_MARGIN_MIB overrides the fixed margin. An
    // absent, unreadable or non-positive value keeps the operator's 1 GiB: a typo
    // is not a decision to remove the margin. R66's red case: the 0/typo path
    // returning 0, or the default read from anywhere but kCardMargin.
    const size_t MiB = 1u << 20;
    CHECK_EQ(llama_loan_book::resolve_card_margin(nullptr), llama_loan_book::kCardMargin);
    CHECK_EQ(llama_loan_book::resolve_card_margin("512"), (size_t) (512 * MiB));
    CHECK_EQ(llama_loan_book::resolve_card_margin("0"), llama_loan_book::kCardMargin);
    CHECK_EQ(llama_loan_book::resolve_card_margin("-1"), llama_loan_book::kCardMargin);
    CHECK_EQ(llama_loan_book::resolve_card_margin("abc"), llama_loan_book::kCardMargin);
    // Round 20's clamp: a huge value cannot wrap the shift (16u << 30 is 0 in a
    // 32-bit unsigned; the ceiling is a size_t), it lands at the ceiling.
    CHECK_EQ(llama_loan_book::resolve_card_margin("999999999"), llama_loan_book::kCardMarginMax);
    CHECK_EQ(llama_loan_book::kCardMarginMax, (size_t) 16 << 30);
    CHECK(has_between("llama-loan-book.h", "inline size_t resolve_card_margin", "struct FdinfoClient",
                      {"kCardMargin", "mib <= 0", "kCardMarginMax"}));
    CHECK(has_between("llama-context.cpp", "bool sized = true;", "the lend: tables relayed",
                      {"LLAMA_SLOT_LOAN_CARD_MARGIN_MIB", "resolve_card_margin"}));
}

TEST(a_card_short_lend_grows_until_the_card_bounds_it) {
    // Round 19, review item 1: the card's own bound can force the growth where the
    // lend-vs-need rule alone would not. Three 1 GiB candidates, the 2 GiB ask
    // picks the last two (entries 1 and 2); need 0, so size_fits(0, 2 GiB) holds.
    // resident 8 GiB, card total 7 GiB, margin 1 GiB -> after the ask 6 GiB sits
    // exactly at the limit (fits); one byte more resident does not, and ONE grown
    // layer (entry 0, 1 GiB) brings it back under. R65's red case: the growth's
    // card term dropped -> the ask is returned unchanged.
    const int64_t GiB = (int64_t) (1u << 30);
    std::vector<llama_loan_book::LendCandidate> c{{true, GiB}, {true, GiB}, {true, GiB}};
    const auto ask = llama_loan_book::choose_lend_layers(c, (size_t) (2 * GiB));
    CHECK_EQ(ask.chosen, std::vector<int>({1, 2}));
    CHECK(llama_loan_book::size_fits(0, (size_t) ask.bytes));   // the reserve's rule holds

    llama_loan_book::CardFacts card;
    card.total  = 7 * GiB;
    card.resident = 8 * GiB;
    card.need   = 0;
    card.margin = 1 * GiB;
    card.lend   = 0;   // no other cache on the card
    const auto p0 = llama_loan_book::extend_lend_layers(c, ask, 0, 0, card);
    CHECK_EQ(p0.chosen, ask.chosen);   // exactly at the limit: no growth
    CHECK(p0.why == nullptr);

    card.resident = 8 * GiB + 1;       // one byte over the limit
    const auto p1 = llama_loan_book::extend_lend_layers(c, ask, 0, 0, card);
    CHECK_EQ(p1.chosen, std::vector<int>({0, 1, 2}));
    CHECK_EQ(p1.bytes, 3 * GiB);
    CHECK(p1.why == nullptr);
    CHECK(llama_loan_book::card_fits(card.resident, card.need, card.total, card.margin, (size_t) p1.bytes));

    // the tail runs out: resident far over the card, the pick names the reason
    card.resident = 20 * GiB;
    const auto p2 = llama_loan_book::extend_lend_layers(c, ask, 0, 0, card);
    CHECK(p2.why != nullptr);
    CHECK(!llama_loan_book::card_fits(card.resident, card.need, card.total, card.margin, (size_t) p2.bytes));
}

TEST(a_failed_growth_nulls_the_lend_tensors_of_every_add_layer) {
    // Round 19, review item 2: alloc_buffer_n placed the add layers' lend tensors in
    // the banks extend_lend staged; when the growth fails, the banks are freed but
    // each tensor still names the released buffer, its `data` and its `extra` (a USM
    // pointer for the OpenCL host bank). The next attempt's alloc plan skips a tensor
    // with `data != NULL`, so growth would be dead for the whole process, and a
    // partly placed set would write through freed USM. Every add layer's lend[k] is
    // nulled in the !ok branch, as refill does for tail_slots. R67's red case: the
    // null loop removed (or the comment claiming the tensors are left in place).
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "if (!ok) {",
                      "for (ggml_backend_buffer_t x : banks)",
                      {"for (int i : add)", "L.lend[k]", "t->buffer = nullptr", "t->data   = nullptr",
                       "t->extra  = nullptr"}, &lines));
    int nulls = -1, free_at = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (nulls < 0 && lines[i].find("t->buffer = nullptr") != std::string::npos) nulls = (int) i;
        if (nulls >= 0 && free_at < 0 && lines[i].find("ggml_backend_buffer_free(x)") != std::string::npos) free_at = (int) i;
    }
    CHECK(nulls >= 0 && free_at > nulls);   // nulled before the banks are freed
    // the old comment said the tensors were left for the next attempt to place
    CHECK(!has_between("llama-expert-cache.cpp", "bool llama_expert_cache::extend_lend",
                       "void llama_expert_cache::drop_tail_buffer() {",
                       {"tensors are left for the next attempt to place again"}));
}


// ---- round 20: the process's own residency from fdinfo, the cold ring ----

namespace {
// a fake fdinfo directory: one file a fd number, the text its /proc/self/fdinfo
// entry would hold. The scan takes the directory, so this is device-free.
std::string fake_fdinfo_dir(const std::vector<std::pair<std::string, std::string>> & files) {
    char t[] = "/tmp/arcint-fdinfo-XXXXXX";
    char * d = mkdtemp(t);
    if (d == nullptr) return {};
    for (const auto & [name, text] : files) {
        std::ofstream f(std::string(d) + "/" + name);
        f << text;
    }
    return d;
}
std::string fdinfo_text(const char * driver, const char * pdev, const char * client, const char * vram0,
                        const char * gtt = nullptr) {
    std::string s = std::string("drm-driver:\t") + driver + "\ndrm-pdev:\t" + pdev +
                    "\ndrm-client-id:\t" + client + "\ndrm-resident-vram0:\t" + vram0 + "\n";
    if (gtt != nullptr) s += std::string("drm-resident-gtt:\t") + gtt + "\n";
    return s;
}
}  // namespace

TEST(the_fdinfo_scan_discovers_the_xe_clients_and_counts_one_a_client) {
    // Round 20, review item 1: the process reads its own residency from the xe
    // render fds' fdinfo, deduped by drm-client-id (two render fds of one client
    // report the same bytes; summing them doubles the figure). R68's red case: the
    // dedupe block removed (two entries), or the drm-driver xe filter removed.
    const std::string dir = fake_fdinfo_dir({
        {"3", fdinfo_text("xe", "0000:03:00.0", "7", "1024 KiB")},
        {"4", fdinfo_text("xe", "0000:03:00.0", "7", "1024 KiB")},   // one client, a second fd
        {"5", fdinfo_text("xe", "0000:03:00.0", "8", "2048 KiB")},   // a second client
        {"6", fdinfo_text("i915", "0000:41:00.0", "9", "4096 KiB")}, // not xe: skipped
        {"README", "not a fd"},
    });
    llama_loan_book::FdinfoScan scan;
    llama_loan_book::scan_fdinfo(dir.c_str(), scan);
    CHECK_EQ(scan.entries, 3u);       // the xe entries; the i915 entry is skipped
    CHECK_EQ(scan.deduped, 1u);       // the second fd of client 7
    CHECK_EQ(scan.clients.size(), 2u);
    const llama_loan_book::FdinfoClient * c7 = nullptr, * c8 = nullptr;
    for (const auto & c : scan.clients) {
        if (c.client_id == 7) c7 = &c;
        if (c.client_id == 8) c8 = &c;
    }
    CHECK(c7 != nullptr && c7->vram0 == (size_t) (1024 * 1024));   // the merged fd, counted once
    CHECK(c8 != nullptr && c8->vram0 == (size_t) (2048 * 1024));
    CHECK_EQ(llama_loan_book::fdinfo_parse_bytes("2048 KiB"), (size_t) (2048 * 1024));
    CHECK_EQ(llama_loan_book::fdinfo_parse_bytes("2 MiB"), (size_t) (2 * 1024 * 1024));
    CHECK_EQ(llama_loan_book::fdinfo_parse_bytes("1 GiB"), (size_t) (1u << 30));
    std::filesystem::remove_all(dir);
}

TEST(the_device_s_drm_pdev_clients_are_summed_and_the_query_takes_any_of_their_fds) {
    // Round 24, the round-22 card window's finding (`measured-here`): the OpenCL
    // runtime opens TWO xe clients on one card in one process (a probe client and
    // the real one), so round 20's unique-pdev match failed and neither the query,
    // the fdinfo figure nor the guard ran. Every xe client in /proc/self is this
    // process's: SUM drm-resident-vram0 and drm-resident-gtt over the distinct
    // client ids whose drm-pdev is the device's (dedupe by client id only), and
    // take any fd of that pdev for the ioctl (used/total are device-wide). A card's
    // figure is its clients', never one client's. R99: the sum loop breaking at its
    // first match; R100: gtt not summed; R102: the loop without the pdev compare.
    const std::string dir = fake_fdinfo_dir({
        {"3", fdinfo_text("xe", "0000:03:00.0", "849", "2056 KiB", "2 GiB")},
        {"6", fdinfo_text("xe", "0000:03:00.0", "851", "16074808 KiB", "41 GiB")},
        {"8", fdinfo_text("xe", "0000:03:00.0", "851", "16074808 KiB", "41 GiB")},  // a second fd of client 851
        {"7", fdinfo_text("xe", "0000:41:00.0", "900", "17 GiB", "9 GiB")},
        {"9", fdinfo_text("i915", "0000:43:00.0", "901", "5 GiB")},                // not xe: skipped
    });
    llama_loan_book::FdinfoScan scan;
    llama_loan_book::scan_fdinfo(dir.c_str(), scan);
    const size_t KiB = 1024, GiB = 1u << 30;
    const llama_loan_book::FdinfoAgg a = llama_loan_book::sum_fdinfo(scan, "0000:03:00.0");
    CHECK(a.measured);
    CHECK_EQ(a.clients, 2);                   // ids 849 and 851; 851's second fd is not a third client
    CHECK_EQ(a.vram0, (size_t) (2056 * KiB + 16074808 * KiB));
    CHECK_EQ(a.gtt, 43 * GiB);
    CHECK_EQ(a.fd, 3);                        // the smallest pdev fd; the query is device-wide
    // the fd really is one of this pdev's, not the other card's
    const char * fdpdev = nullptr;
    for (const auto & c : scan.clients) if (c.fd == a.fd) fdpdev = c.pdev.c_str();
    CHECK(fdpdev != nullptr && std::string(fdpdev) == "0000:03:00.0");
    // one card's figure is its clients', not another card's
    const llama_loan_book::FdinfoAgg b = llama_loan_book::sum_fdinfo(scan, "0000:41:00.0");
    CHECK(b.measured && b.clients == 1 && b.vram0 == 17 * GiB && b.gtt == 9 * GiB && b.fd == 7);
    CHECK(!llama_loan_book::sum_fdinfo(scan, "0000:99:00.0").measured);   // fallback
    CHECK(!llama_loan_book::sum_fdinfo(scan, nullptr).measured);          // several clients, no pdev
    // one xe client: trivial, no pdev needed
    const std::string one = fake_fdinfo_dir({{"3", fdinfo_text("xe", "0000:03:00.0", "7", "3 GiB")}});
    llama_loan_book::FdinfoScan s1;
    llama_loan_book::scan_fdinfo(one.c_str(), s1);
    const llama_loan_book::FdinfoAgg c1 = llama_loan_book::sum_fdinfo(s1, nullptr);
    CHECK(c1.measured && c1.clients == 1 && c1.vram0 == 3 * GiB);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(one);
}

TEST(the_fdinfo_figure_is_the_rule_and_the_ggml_sum_is_the_fallback_and_tripwire) {
    // Round 20, review item 1: the rule uses fdinfo vram0; the ggml sum is the
    // fallback (no unique xe match) and the tripwire (fdinfo below the sum: the
    // process's own buffers are already evicted to GTT -> refuse). R70's red
    // cases: the tripwire check removed, or the fallback returning the client.
    const size_t GiB = 1u << 30;
    const std::string dir = fake_fdinfo_dir({
        {"3", fdinfo_text("xe", "0000:03:00.0", "7", "12 GiB")},
        {"4", fdinfo_text("xe", "0000:03:00.0", "8", "8 GiB")},   // one card, two clients
        {"5", fdinfo_text("xe", "0000:41:00.0", "9", "17 GiB")},
    });
    llama_loan_book::FdinfoScan scan;
    llama_loan_book::scan_fdinfo(dir.c_str(), scan);
    const size_t sum = 18 * GiB;   // the ggml sum: below the card's summed 20 GiB
    auto rd = llama_loan_book::read_resident(scan, "0000:03:00.0", sum);
    CHECK(rd.measured && !rd.evicted && rd.bytes == 20 * GiB);   // the summed fdinfo, above the sum
    rd = llama_loan_book::read_resident(scan, "0000:99:00.0", sum);
    CHECK(!rd.measured && !rd.evicted && rd.bytes == sum);        // fallback
    CHECK(std::string(rd.why) == "no xe client on the device's drm-pdev");
    rd = llama_loan_book::read_resident(scan, "0000:41:00.0", sum);
    CHECK(rd.measured && rd.evicted && rd.bytes == 17 * GiB);     // the tripwire
    std::filesystem::remove_all(dir);
}

TEST(the_probe_reads_the_device_s_sum_after_the_backend_opened_its_fds) {
    // Round 24: the load probe must find the clients. It sits in
    // llama_expert_cache::init AFTER the cache's own tail/slot buffer allocations
    // (the backend's xe render fds are open by then) and uses the device's summed
    // figure with a pdev fd for the query; no unique-client match remains anywhere.
    // R101: the probe's summed lookup replaced by a unique match, or its device pdev
    // dropped (the clients then are not summed on this device and it falls back), or
    // the probe's scan moved before the allocation, where no fd exists yet.
    CHECK(has_between("llama-expert-cache.cpp", "Round 21: the card's real state at load", "return true;",
                      {"sum_fdinfo(scan, props.device_id)", "xe_query_card(agg.fd)"}));
    CHECK(!has_between("llama-expert-cache.cpp", "Round 21: the card's real state at load", "return true;",
                       {"match_fdinfo"}));
    CHECK(!has_between("llama-context.cpp", "bool llama_context::slot_loan_begin",
                       "bool llama_context::slot_loan_end", {"match_fdinfo"}));
    // R101's values: the figure the probe reads is the device's clients summed,
    // and the query fd is that pdev's smallest -- not one client's figure, not
    // another card's
    const std::string dir = fake_fdinfo_dir({
        {"3", fdinfo_text("xe", "0000:03:00.0", "849", "2056 KiB", "2 GiB")},
        {"6", fdinfo_text("xe", "0000:03:00.0", "851", "16074808 KiB", "41 GiB")},
    });
    llama_loan_book::FdinfoScan scan;
    llama_loan_book::scan_fdinfo(dir.c_str(), scan);
    const auto agg = llama_loan_book::sum_fdinfo(scan, "0000:03:00.0");
    const size_t KiB = 1024;
    CHECK(agg.measured && agg.clients == 2);
    CHECK_EQ(agg.vram0, (size_t) (2056 * KiB + 16074808 * KiB));
    CHECK_EQ(agg.fd, 3);
    std::filesystem::remove_all(dir);
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::init", "return true;",
                      {"scan_fdinfo(\"/proc/self/fdinfo\", scan)"}, &lines));
    // the probe's scan comes after the tail buffer's allocation in init
    int alloc = -1, scan_at = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (alloc < 0 && lines[i].find("ggml_backend_buft_alloc_buffer_n(tail_buft") != std::string::npos) alloc = (int) i;
        if (alloc >= 0 && scan_at < 0 && lines[i].find("scan_fdinfo") != std::string::npos) scan_at = (int) i;
    }
    CHECK(alloc >= 0 && scan_at > alloc);
}

TEST(the_card_rule_counts_the_fdinfo_figure_and_the_cold_rings_first_touch) {
    // Round 20, review item 1: the rule is
    // `fdinfo_vram0 - lend_bytes_on_card + need_card + first_touch <= total - margin`,
    // with first_touch = K x the largest bank tensor while the stream ring is not
    // allocated (a cold process). The ggml sum is the fallback and the tripwire.
    // R71's red case: first_touch dropped from card_fits (or from the context's
    // call) -> a cold ring's pages are counted as free.
    const size_t MiB = 1u << 20;
    const size_t total = 24480 * MiB;
    const size_t margin = 1024 * MiB;
    const size_t limit = total - margin;   // 23,456 MiB
    const size_t lend = 6349 * MiB;
    const size_t need = 4659 * MiB;
    const size_t first_touch = 751 * MiB;  // the recorded ring: 4 x 187.7 MiB
    // resident after the lend + need + first touch exactly at the limit: fits;
    // one byte over: not. Without the first touch this resident fits -- so the
    // term is what the boundary sees.
    const size_t resident = limit + lend - need - first_touch;
    CHECK(llama_loan_book::card_fits(resident, need, total, margin, lend, first_touch));
    CHECK(!llama_loan_book::card_fits(resident + 1, need, total, margin, lend, first_touch));
    CHECK(llama_loan_book::card_fits(resident + first_touch, need, total, margin, lend, 0));
    CHECK(!llama_loan_book::card_fits(resident + first_touch, need, total, margin, lend, 1));
    // the cold ring's first touch: K x the largest bank tensor while the ring is
    // not allocated; 0 once the ring is there (a warm process's fdinfo has it).
    CHECK_EQ(llama_loan_book::first_touch_bytes(4, 0, 1877 * 1024 * 1024 / 10),
             (size_t) (4 * (1877 * 1024 * 1024 / 10)));
    CHECK_EQ(llama_loan_book::first_touch_bytes(4, 751 * MiB, 1877 * 1024 * 1024 / 10), 0u);
    CHECK_EQ(llama_loan_book::resolve_bank_stream_ring(nullptr), 4);
    CHECK_EQ(llama_loan_book::resolve_bank_stream_ring("8"), 8);
    CHECK_EQ(llama_loan_book::resolve_bank_stream_ring("1"), 4);
}

TEST(the_context_reads_the_fdinfo_after_sched_new_and_drain_and_refuses_eviction) {
    // Round 20, review item 1: the read is after sched_new freed the old compute
    // buffers and after the drain, before the lend; the device's drm-pdev comes
    // from the props' device_id; the tripwire refuses an evicted process; the loan
    // out line prints fdinfo, the sum, need, lend, first touch, total and margin.
    // R68/R70's red cases on the call sites.
    std::vector<std::string> lines;
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_begin", "bool llama_context::slot_loan_end",
                       {"llama_loan_book::drain_now()", "sched_new(max_nodes", "scan_fdinfo(\"/proc/self/fdinfo\"",
                        "llama_loan_book::read_resident(fdinfo, props.device_id, sum)",
                        "rd.evicted", "already evicted to GTT", "first_touch_bytes(ring_k, ring, max_bank)",
                        "GGML_OPENCL_BANK_STREAM_RING", "llama_loan_book::card_fits(c.second, lend_on_card(c.first))",
                        "llama_loan_book::xe_query_card", "llama_loan_book::used_beyond_fdinfo",
                        "llama_loan_book::eviction_guard"}, &lines));
    int sched = -1, drain = -1, scan = -1, lend = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (sched < 0 && lines[i].find("sched_new(max_nodes") != std::string::npos) sched = (int) i;
        if (drain < 0 && lines[i].find("llama_loan_book::drain_now()") != std::string::npos) drain = (int) i;
        if (scan < 0 && lines[i].find("scan_fdinfo(\"/proc/self/fdinfo\"") != std::string::npos) scan = (int) i;
        if (lend < 0 && lines[i].find("the lend: tables relayed") != std::string::npos) lend = (int) i;
    }
    CHECK(drain >= 0 && scan > drain && scan > sched && lend > scan);   // after sched_new and the drain
    CHECK(has_between("llama-context.cpp", "LLAMA_LOG_WARN(\"%s: loan out at", "return true;",
                      {"%s %.2f GiB, ggml buffer sum %.2f GiB, fdinfo vram0 %.2f GiB, fdinfo gtt %.2f GiB",
                       "used - fdinfo %.2f GiB, first touch %.2f GiB, prefill growth allowance %.2f GiB",
                       "allowance = %.2f GiB <= total", "after %.2f GiB + need"}));
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp", "static void ggml_backend_opencl_device_get_props",
                       "static ggml_backend_t ggml_backend_opencl_device_init",
                       {"props->device_id", "pci_id"}));
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp", "static ggml_backend_opencl_context * ggml_cl_init(ggml_backend_dev_t dev) {",
                       "size_t ext_str_size;",
                       {"CL_DEVICE_PCI_BUS_INFO_KHR", "pci_domain", "pci_bus", "pci_device", "pci_function"}));
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp", "static void ggml_backend_opencl_bank_stream_ring(",
                       "static void ggml_backend_opencl_drain(void) {",
                       {"bs.ring", "ring_bytes"}));
    // no stale claim: the target context's ctx_other never holds the MTP draft
    // (arcint sets it on the draft's params), so the resident sum does not name it
    CHECK(!has_between("llama-context.cpp", "auto resident_on = [&]", "lend_on_card = [&]",
                       {"cparams.ctx_other"}));
}

TEST(the_fixed_margin_is_the_documented_one) {
    // The operator's decision 2026-10-30: 384 MiB peak reserve + 1 GiB fixed under
    // the card's total. R60's red case: kCardMargin 0 (or size_fits dropping it) ->
    // the 1.375 GiB boundary below moves.
    CHECK_EQ(llama_loan_book::kPeakReserve, (size_t) (384u << 20));
    CHECK_EQ(llama_loan_book::kCardMargin, (size_t) (1u << 30));
    CHECK_EQ(llama_loan_book::kLendMargin, llama_loan_book::kPeakReserve + llama_loan_book::kCardMargin);
    const size_t lend = 6400u << 20;
    CHECK(llama_loan_book::size_fits(lend - llama_loan_book::kLendMargin, lend));
    CHECK(!llama_loan_book::size_fits(lend - llama_loan_book::kLendMargin + 1, lend));
    // the sizing grows the lend before it refuses, and the refusal WARN names need,
    // lend and the reserve margin
    CHECK(has_between("llama-context.cpp", "bool sized = true;", "the lend: tables relayed",
                      {"extend_lend", "kLendMargin", "the fixed margin is %.3f GiB"}));
}

TEST(the_lent_sizing_is_measured_unless_mutated) {
    // Red case 3, deterministic since round 10: round 8 skipped the lent flag in
    // the sizing on the premise of one more MUL_MAT_ID dst; the lent branch
    // instead REPLACES the slot branch and drops both down tables, so the skip
    // RAISES `need` and actual <= expect passes on the card (MUTATE=3 lent and
    // returned normally, measured-here round 8). The red case now sizes at the
    // resident chunk while the reserve allocates the prompt's chunk: `need` is
    // under-counted for real and reserve_matches rejects it.
    CHECK(llama_loan_book::measure_lent(0));
    CHECK(llama_loan_book::measure_lent(1));
    CHECK(!llama_loan_book::measure_lent(3));
    CHECK_EQ(llama_loan_book::measure_tokens(0, 8192, 2048), 8192u);
    CHECK_EQ(llama_loan_book::measure_tokens(4, 8192, 2048), 8192u);
    CHECK_EQ(llama_loan_book::measure_tokens(3, 8192, 2048), 2048u);   // R34's red case: cap
    CHECK(llama_loan_book::reserve_matches(1200, 1200));
    CHECK(llama_loan_book::reserve_matches(1200, 900));       // R30's red case: a shared buft counts lower
    CHECK(llama_loan_book::reserve_matches(1200, 0));
    CHECK(!llama_loan_book::reserve_matches(1200, 1200 + 640));  // MUTATE=3: actual > the expectation
    CHECK(has_between("llama-context.cpp", "size_t need = 0;", "if (!llama_loan_book::size_fits(need, lend))",
                      {"llama_loan_book::measure_tokens(mutate_loan_ctx_measure, cap, cparams.n_ubatch_reserve)",
                       "ec->set_measure_lent(true)",
                       "LLAMA_SLOT_LOAN_MUTATE=3: the prompt's graph is sized at the resident chunk"}));
    CHECK(has_between("llama-context.cpp", "if (!slot_loan_reserve(cap))", "if (info) {",
                      {"ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i])",
                       "llama_loan_book::reserve_matches(need, actual)",
                       "LLAMA_SLOT_LOAN_MUTATE=3 sizes at the resident chunk"}));   // R23's red case
}

TEST(a_poison_path_drains_before_any_free) {
    // Round 10, card finding 1: the tail buffer was freed at the lend while the
    // adapt/bank-stream queues could still hold copies over it; the exit-70 drain
    // then executed them against released memory -- the B60's xe -ENOENT and GuC
    // reset, timed to the exit second (MUTATE=4, measured-here). A free is safe
    // only after the queues are finished; drain_now() runs the registered drain.
    // R35's red case: drain_now() a no-op, or the drain in drop_tail_buffer gone.
    llama_loan_book::DrainFn prev = llama_loan_book::drain_hook();
    llama_loan_book::register_drain(nullptr);
    llama_loan_book::drain_now();   // no hook: must not crash
    llama_loan_book::register_drain(ln_count_drain);
    llama_loan_book::drain_now();
    CHECK_EQ(g_drain_calls, 1);   // R35's red case: drain_now() a no-op
    llama_loan_book::register_drain(prev);

    // the free site drains first, and only when there is a buffer to free
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "void llama_expert_cache::drop_tail_buffer",
                      "bool llama_expert_cache::refill",
                      {"lend_bytes_ != 0 || b_tail != nullptr", "llama_loan_book::drain_now()",
                       "ggml_backend_buffer_free(b_tail)"}, &lines));
    int d = -1, f = -1, s0 = -1, e0 = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (s0 < 0 && lines[i].find("void llama_expert_cache::drop_tail_buffer() {") != std::string::npos) s0 = (int) i;
        if (s0 >= 0 && e0 < 0 && lines[i].find("bool llama_expert_cache::refill") != std::string::npos) e0 = (int) i;
    }
    for (int i = s0; s0 >= 0 && i < e0; ++i) {
        if (d < 0 && lines[(size_t) i].find("llama_loan_book::drain_now()") != std::string::npos) d = i;
        if (f < 0 && lines[(size_t) i].find("ggml_backend_buffer_free(b_tail)") != std::string::npos) f = i;
    }
    CHECK(d >= 0 && f >= 0 && d < f);   // the free after the drain
    // the return's poison path drains before it poisons
    std::vector<std::string> plines;
    CHECK(has_between("llama-context.cpp", "const bool back = mutate_loan_ctx != 4",
                      "for (const auto & ec : model.expert_caches)",
                      {"llama_loan_book::drain_now()", "slot_loan_poison(p)"}, &plines));
    int dn = -1, pz = -1, ps = -1, pe = -1;
    for (size_t i = 0; i < plines.size(); ++i) {
        if (ps < 0 && plines[i].find("const bool back = mutate_loan_ctx != 4") != std::string::npos) ps = (int) i;
        if (ps >= 0 && pe < 0 && plines[i].find("for (const auto & ec : model.expert_caches)") != std::string::npos) pe = (int) i;
    }
    for (int i = ps; ps >= 0 && i < pe; ++i) {
        if (dn < 0 && plines[(size_t) i].find("llama_loan_book::drain_now()") != std::string::npos) dn = i;
        if (pz < 0 && plines[(size_t) i].find("slot_loan_poison(p)") != std::string::npos) pz = i;
    }
    CHECK(dn >= 0 && pz >= 0 && dn < pz);   // R35's red case: the poison first
}

// ---- round 11: the MUTATE=4 exit, the sizing's throw, the drain scope ----

TEST(a_lent_sizing_throw_takes_the_resident_refusal) {
    // Round 11, review item 2: slot_loan_begin calls sched_new before the sizing
    // (graph_reserve -> sched_reserve_size -> gallocr). A C++ throw there reached
    // Scope::begin, which catches it and returns -- the context stayed on the
    // replacement scheduler with no reserve and every cache's owner_sched pointing
    // at the destroyed one. The sizing is wrapped; the catch falls into the same
    // recovery a refused size check takes: slot_loan_reserve(resident) (which
    // drains before sched_new) and the rebind. R36's red case: the try/catch
    // removed, or the resident re-reserve/rebind dropped from the catch.
    // Round 12, review low: the ORDER matters -- a rebind before the reserve binds
    // owner_sched to the scheduler the failed reserve replaces, so the text order
    // is checked too (R36's order half).
    std::vector<std::string> lines;
    CHECK(has_between("llama-context.cpp", "bool sized = true;",
                      "if (!llama_loan_book::size_fits(need, lend))",
                      {"try {", "sched_new(max_nodes, cparams.pipeline_parallel)",
                       "catch (const std::exception & e)", "sized = false;",
                       "if (!sized) {", "slot_loan_reserve(cparams.n_ubatch_reserve)",
                       "ec->rebind(sched.get())"}, &lines));
    // the !sized block only: end at the next landmark, so a rebind in the refused
    // size check's block cannot stand in for the catch's one
    int start = -1, end = -1, rsv = -1, rb = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (start < 0 && lines[i].find("if (!sized) {") != std::string::npos) start = (int) i;
        if (start >= 0 && end < 0 && lines[i].find("if (!llama_loan_book::size_fits(need, lend))") != std::string::npos) end = (int) i;
        if (start < 0) continue;
        if (rsv < 0 && lines[i].find("slot_loan_reserve(cparams.n_ubatch_reserve)") != std::string::npos) rsv = (int) i;
        if (rsv >= 0 && rb < 0 && lines[i].find("ec->rebind(sched.get())") != std::string::npos) rb = (int) i;
    }
    CHECK(end > start && rsv > start && rsv < end && rb > rsv && rb < end);
    // R36's order half: the resident reserve before the rebind, both in the block
}

TEST(a_re_reserve_finishes_the_queues_before_sched_new) {
    // Round 11, card finding 1: sched_reserve's sched_new replaces the live
    // scheduler and the galloc frees the buffers it held; its own synchronize()
    // finishes the compute and bank-stream queues but NOT the adapt queue. Both
    // the reserve and the exit path drain every queue first, and a throw during
    // the reserve drains again before the caller records the failure. R37's red
    // case: the pre-reserve drain removed, or the catch's drain removed.
    std::vector<std::string> lines;
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_reserve",
                      "bool llama_context::slot_loan_begin",
                      {"llama_loan_book::drain_now()", "sched_reserve()",
                       "catch (const std::exception & e)", "llama_loan_book::drain_now()"}, &lines));
    int start = -1, d0 = -1, rs = -1, c = -1, d1 = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (start < 0 && lines[i].find("bool llama_context::slot_loan_reserve") != std::string::npos) start = (int) i;
        if (start < 0) continue;
        if (d0 < 0 && lines[i].find("llama_loan_book::drain_now()") != std::string::npos) d0 = (int) i;
        if (rs < 0 && lines[i].find("sched_reserve()") != std::string::npos) rs = (int) i;
        if (c < 0 && lines[i].find("catch (const std::exception & e)") != std::string::npos) c = (int) i;
        if (c >= 0 && d1 < 0 && lines[i].find("llama_loan_book::drain_now()") != std::string::npos) d1 = (int) i;
    }
    CHECK(d0 >= 0 && rs > d0);          // the drain before the reserve replaces sched
    CHECK(c > rs && d1 > c);            // a throw drains again before the poison
}

TEST(a_lent_cache_drains_before_every_free_in_the_destructor) {
    // Round 11, review item 2: the destructor drained only when b_tail existed.
    // After a lend b_tail is null (the loan freed it), while bslots, the USM bank
    // and the lend bank are freed -- and the adapt/bank-stream queues can still
    // read the USM banks. With the loan armed (lend_bytes_ != 0) it drains once,
    // before freeing anything; with the loan off nothing is added. R38's red case:
    // the drain back under `if (b_tail != nullptr)`, or keyed on b_tail.
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "llama_expert_cache::~llama_expert_cache",
                      "for (auto b : bufs)",
                      {"if (lend_bytes_ != 0) {", "llama_loan_book::drain_now()"}, &lines));
    int start = -1, d = -1, bt = -1, bufs = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (start < 0 && lines[i].find("llama_expert_cache::~llama_expert_cache") != std::string::npos) start = (int) i;
        if (start < 0) continue;
        if (d < 0 && lines[i].find("llama_loan_book::drain_now()") != std::string::npos) d = (int) i;
        if (bt < 0 && lines[i].find("ggml_backend_buffer_free(b_tail)") != std::string::npos) bt = (int) i;
        if (bufs < 0 && lines[i].find("for (auto b : bufs) {") != std::string::npos) bufs = (int) i;
    }
    CHECK(d >= 0 && bt > d && bufs > d);   // the drain precedes both frees
}

TEST(a_failed_tail_allocation_leaves_no_dangling_tensor) {
    // Round 11, review item 1: ggml-backend's alloc_buffer_n_default frees the
    // buffers it built when a later one fails, but the tensors it placed in them
    // keep buffer/data/extra pointing at the released object. The refill's failure
    // path clears them exactly as drop_tail_buffer does, so the slot image holds no
    // dangling reference when the poison's drain runs and _Exit follows. R39's red
    // case: the clears removed from the `b_tail == nullptr` branch.
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::refill",
                      "ggml_backend_buffer_set_usage(b_tail, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)",
                      {"if (b_tail == nullptr) {", "t->buffer = nullptr;", "t->data   = nullptr;",
                       "t->extra  = nullptr;", "poison_exit(why)"}, &lines));
    int start = -1, clear = -1, poison = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (start < 0 && lines[i].find("if (b_tail == nullptr) {") != std::string::npos) start = (int) i;
        if (start < 0) continue;
        if (clear < 0 && lines[i].find("t->extra  = nullptr;") != std::string::npos) clear = (int) i;
        if (poison < 0 && lines[i].find("poison_exit(why)") != std::string::npos) poison = (int) i;
    }
    CHECK(clear >= 0 && poison > clear);   // no dangling tensor reaches the exit
}

TEST(the_refill_comment_names_the_planar_types) {
    // The fifth review's nit: the comment named IQ3_XXS and IQ4_XS as planar. The
    // planar set is ggml-opencl.cpp's ggml_cl_lb_fields cases -- IQ2_XXS, IQ2_XS,
    // IQ2_S, Q2_0, IQ3_S -- and the rules below rest on it, so the comment names
    // every one of them. Red case R21: one name off (or a non-planar type added to
    // the list) in the refill's comment.
    const std::vector<std::string> planar = {"GGML_TYPE_IQ2_XXS", "GGML_TYPE_IQ2_XS", "GGML_TYPE_IQ2_S",
                                            "GGML_TYPE_Q2_0", "GGML_TYPE_IQ3_S"};
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "static int ggml_cl_lb_fields(ggml_type t, ggml_cl_lb_field f[5]) {",
                       "static bool ggml_cl_lb_planar", planar));
    CHECK(has_between("llama-expert-cache.cpp", "Round 7, the refill's types:", "void llama_expert_cache::set_measure_lent",
                      {"IQ2_XXS, IQ2_XS, IQ2_S, Q2_0, IQ3_S --", "BF16"}));
}

TEST(the_refill_writes_and_checks_the_zero_slot) {
    // finding 6's call site: refill's zero-row block. The helper (refill_zero_row)
    // alone cannot fail when the block is removed; the loop must call it and
    // write zeros at that row -- round 7: from a device zero page, not host bytes.
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::refill", "void llama_expert_cache::rebind",
                      {"llama_loan_book::refill_zero_row(L.n_slots)", "lend_zero(L.slot[k]", "refill_check_rows(L.n_slots)"}));   // R6's red case
}

TEST(lend_undo_checks_the_tables_again) {
    // finding 7's call site: the undo's writes are not waited for; without the
    // check the cache could be served from tables that never landed.
    CHECK(has_between("llama-expert-cache.cpp", "void llama_expert_cache::lend_undo", "void llama_expert_cache::finish_pending",
                      {"check_tables()", "llama_loan_book::lend_broken("}));   // R7's red case
}

TEST(the_decode_budget_uses_the_capped_ubatch) {
    // the Flash-Next normal path, device-free: with the loan off the context's
    // reserve equals n_ubatch, so ubatch_cap is n_ubatch (byte-identical to
    // before). The call site's budget must read the capped value, not n_ubatch:
    // under a loan the two differ (2048 reserved against an 8192 ubatch).
    CHECK_EQ(llama_loan_book::ubatch_cap(8192, 8192), 8192u);
    CHECK_EQ(llama_loan_book::ubatch_cap(8192, 2048), 2048u);
    CHECK_EQ(llama_loan_book::cell_budget(8192, 4), 8192u * 4u);
    CHECK(has_between("llama-context.cpp", "const uint32_t ub_cap =", "n_ubatch_eff = ub_cap",
                      {"llama_loan_book::ubatch_cap(cparams.n_ubatch, reserve_ubatch)"}));
    CHECK(has_between("llama-context.cpp", "const uint64_t budget =", "n_ubatch_eff = (uint32_t)",
                      {"llama_loan_book::cell_budget(ub_cap, cparams.n_kv_reserve)"}));   // R11's red case
}

// ---- third review: the remaining rules ----

TEST(a_cache_without_a_lend_is_not_lent) {
    // finding 5: under a layer split the other card's cache has lend_bytes 0; its
    // lend() returns false, so the AND over every cache refused the loan every
    // time. The caller skips it.
    CHECK(!llama_loan_book::loan_cache(0));            // R5's red case: true
    CHECK(llama_loan_book::loan_cache(8ull << 30));
}

TEST(the_refill_restores_the_zero_slot_too) {
    // finding 6: init wrote and verified row n_slots (zeros); the refill wrote
    // only [0, n_slots) into a fresh, undefined buffer and the slot branch reads
    // the zero row for a bank expert. The refill writes it and the check covers
    // s <= n_slots.
    const int64_t n = 4;
    CHECK_EQ(llama_loan_book::refill_zero_row(n), n);        // R6's red case: -1
    CHECK_EQ(llama_loan_book::refill_check_rows(n), n + 1);
    CHECK_EQ(llama_loan_book::refill_zero_row(n), llama_loan_book::refill_check_rows(n) - 1);
}

TEST(a_failed_lend_whose_undo_still_differs_is_broken) {
    // finding 7: the undo's table writes are not waited for; the cache checks them
    // again (check_tables' read is the wait) and, if they still differ, is never
    // served from: every request is refused until restart
    CHECK(llama_loan_book::lend_broken(true));    // R7's red case: false
    CHECK(!llama_loan_book::lend_broken(false));
}

TEST(an_explicit_zero_flag_stays_off) {
    // finding 8: the rule lives in init, not in the tool. LLAMA_SLOT_LOAN_MIB is
    // read only for the sentinel llama_model_default_params sets; an explicit 0
    // -- a tool's flag off -- stays off whatever the environment says.
    const size_t sentinel = LLAMA_SLOT_LOAN_BYTES_ENV;
    CHECK_EQ(llama_loan_book::resolve_lend_bytes(0, sentinel, 4608), static_cast<size_t>(0));   // R8's red case
    CHECK_EQ(llama_loan_book::resolve_lend_bytes(sentinel, sentinel, 4608), static_cast<size_t>(4608) << 20);
    CHECK_EQ(llama_loan_book::resolve_lend_bytes(sentinel, sentinel, 0), static_cast<size_t>(0));
    CHECK(llama_loan_book::resolve_lend_bytes(4608ull << 20, sentinel, 0) == (static_cast<size_t>(4608) << 20));
}

TEST(the_default_params_ask_for_the_environment) {
    // finding 8: llama_model_default_params must hand init the "no parameter set"
    // sentinel; a default of 0 would make every environment setting dead
    const llama_model_params p = llama_model_default_params();
    CHECK_EQ(p.slot_loan_bytes, LLAMA_SLOT_LOAN_BYTES_ENV);   // R8's red case: 0
    CHECK(p.slot_loan_bytes != 0);
}

TEST(a_failed_lend_puts_the_layers_back) {
    Layer L;
    lend_tables(L);
    // while lent every ex-slot expert is a bank miss and its lend row is named
    CHECK_EQ(L.slot_of, std::vector<int32_t>({-1, -1, -1, -1}));
    CHECK_EQ(L.loan_of, std::vector<int32_t>({0, -1, 1, -1}));
    CHECK(L.lent);

    // the relay did not reach the card: the lend cannot be published
    CHECK(llama_loan_book::undo_lend(L));
    CHECK(!L.lent);
    CHECK_EQ(L.slot_of, std::vector<int32_t>({0, -1, 1, -1}));     // R6's red case
    CHECK_EQ(L.loan_of, std::vector<int32_t>({-1, -1, -1, -1}));
}

TEST(undo_lend_leaves_an_unlent_layer_alone) {
    Layer L;
    CHECK(!llama_loan_book::undo_lend(L));   // nothing was lent: no table write
    CHECK_EQ(L.slot_of, std::vector<int32_t>({0, -1, 1, -1}));
    CHECK_EQ(L.loan_of, std::vector<int32_t>({-1, -1, -1, -1}));
}

TEST(the_tail_buffer_is_not_the_destructor_list) {
    // finding 7 (second review), third review: test production's call shape.
    // init builds `bufs = {bslots, bbank}`, then takes the tail buffer; the tail
    // buffer is never in the list, so the destructor's walk cannot free it twice
    // and the loan's own pointer stays its owner. The bug is an entry added for
    // it -- before or after the call.
    void * bslots = reinterpret_cast<void *>(0x10);
    void * bbank  = reinterpret_cast<void *>(0x20);
    void * b_lend = reinterpret_cast<void *>(0x30);
    std::vector<void *> bufs = {bslots, bbank};
    void * tail = nullptr;

    llama_loan_book::take_tail(bufs, tail, b_lend);

    CHECK_EQ(tail, b_lend);
    CHECK(!llama_loan_book::listed(bufs, b_lend));     // R7's red case: a push after the call
    CHECK_EQ(bufs.size(), static_cast<size_t>(2));
    CHECK(llama_loan_book::listed(bufs, bslots) && llama_loan_book::listed(bufs, bbank));   // the others stay

    // a null tail is never "listed", and an erased entry of an older shape stays out
    std::vector<void *> older{b_lend, bslots};
    llama_loan_book::take_tail(older, tail, b_lend);
    CHECK(!llama_loan_book::listed(older, b_lend));
    CHECK(!llama_loan_book::listed(bufs, static_cast<void *>(nullptr)));
}


// ---- round 8: the seventh review's findings ----

TEST(a_region_order_read_compares_row_by_row) {
    // Round 9, review item 3: a split tensor's read image is region-major,
    // [d x n][q x n], each region's rows contiguous with the region's own stride.
    // Round 8 compared whole buffers (that avoided the region-order false
    // positives) but counted one per tensor while n counted rows and the line said
    // rows. The check now compares row by row INSIDE each region and names the
    // first mismatch's region and row. This models the helper on the same layout;
    // R31's red case stops the loop after the first region.
    const int n_slots = 3, d_per = 2, q_per = 16;   // IQ4_NL: one block, 2 scales, 16 quants
    const size_t per_slice = d_per + q_per;
    auto image = [&](int rows, uint8_t seed_d, uint8_t seed_q) {
        std::vector<uint8_t> out;
        for (int s = 0; s < rows; ++s) for (int i = 0; i < d_per; ++i) out.push_back((uint8_t) (seed_d + s + i));
        for (int s = 0; s < rows; ++s) for (int i = 0; i < q_per; ++i) out.push_back((uint8_t) (seed_q + s + i));
        return out;
    };
    const std::vector<uint8_t> bank = image(n_slots, 10, 40);
    std::vector<uint8_t> slot = image(n_slots + 1, 10, 40);
    // the zero slot: both regions' rows n_slots
    slot[(size_t) n_slots * d_per + 0] = 0;
    slot[(size_t) n_slots * d_per + 1] = 0;
    const size_t qbase = (size_t) (n_slots + 1) * d_per;
    for (int i = 0; i < q_per; ++i) slot[qbase + (size_t) n_slots * q_per + i] = 0;

    const auto regions = llama_loan_book::lend_region_bytes(GGML_TYPE_IQ4_NL, per_slice);
    CHECK_EQ(regions.nr, 2);
    CHECK_EQ(regions.per[0], (size_t) d_per);   // scales
    CHECK_EQ(regions.per[1], (size_t) q_per);   // quants

    // a correct copy: zero differing rows, 2 regions x (n_slots + 1) rows
    auto d0 = llama_loan_book::lend_compare_rows(slot.data(), bank.data(), n_slots,
                                                 GGML_TYPE_IQ4_NL, per_slice);
    CHECK_EQ(d0.bad, 0);
    CHECK_EQ(d0.rows, 2 * (n_slots + 1));
    CHECK_EQ(d0.first_row, -1);

    // one byte of the quants region, row 1: one row, and the name says quants/row 1
    std::vector<uint8_t> bad1 = slot;
    bad1[qbase + (size_t) 1 * q_per + 3] ^= 0xFF;
    auto d1 = llama_loan_book::lend_compare_rows(bad1.data(), bank.data(), n_slots,
                                                 GGML_TYPE_IQ4_NL, per_slice);
    CHECK_EQ(d1.bad, 1);                    // R31's red case: 0 when region 1 is not walked
    CHECK_EQ(d1.rows, 2 * (n_slots + 1));
    CHECK_EQ(d1.first_region, 1);
    CHECK_EQ(d1.first_row, 1);
    CHECK_EQ(std::string(llama_loan_book::lend_region_name(GGML_TYPE_IQ4_NL, 1)), "quants");

    // one byte of the zero row in the scales region names that region and row
    std::vector<uint8_t> badz = slot;
    badz[(size_t) n_slots * d_per] ^= 0xFF;   // the zero row of the scales region
    auto dz = llama_loan_book::lend_compare_rows(badz.data(), bank.data(), n_slots,
                                                 GGML_TYPE_IQ4_NL, per_slice);
    CHECK_EQ(dz.bad, 1);
    CHECK_EQ(dz.first_region, 0);
    CHECK_EQ(dz.first_row, n_slots);
    CHECK_EQ(std::string(llama_loan_book::lend_region_name(GGML_TYPE_IQ4_NL, 0)), "scales");

    // a plain tensor: one region, rows are the tensor's rows
    const auto plain = llama_loan_book::lend_region_bytes(GGML_TYPE_IQ3_S, 7);
    CHECK_EQ(plain.nr, 1);
    CHECK_EQ(plain.per[0], static_cast<size_t>(7));
    CHECK_EQ(std::string(llama_loan_book::lend_region_name(GGML_TYPE_IQ3_S, 0)), "row");

    // the refill's bank branch uses the row helper and reads both sides with the
    // slot's n_slots + 1 rows; the whole-buffer memcmp and the per-tensor count are
    // gone. R25's red case: the whole-buffer form back.
    CHECK(has_between("llama-expert-cache.cpp", "round 7's default", "if (verify_rows && bad)",
                      {"lend_compare_rows(a.data(), b.data()", "lend_read(L.slot[k], 0, L.n_slots + 1, a.data())",
                       "first_bad_row", "lend_region_name"}));
    CHECK(!has_between("llama-expert-cache.cpp", "round 7's default", "if (verify_rows && bad)",
                       {"bad += std::memcmp(a.data(), b.data(), a.size())", "bad += std::memcmp(a.data(), z.data()"}));
}

TEST(a_lendable_type_is_one_the_region_copy_is_exact_for) {
    // Round 8, review item 3: the USM bank stores most types raw while set_tensor
    // builds a specialised extra (Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, MXFP4, Q4_K, Q5_K,
    // Q6_K, Q8_0) or converts BF16 to f16 on the card. A region copy there writes
    // GGUF bytes through a plain extra (the old crash) or feeds f16 kernels raw
    // BF16 -- the bank check would compare the wrong bytes with themselves and
    // pass. Plain-no-extra types, the planar set and IQ4_NL are exact on both.
    const std::vector<ggml_type> exact = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q8_K,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, GGML_TYPE_IQ3_S};
    for (ggml_type t : exact) CHECK(llama_loan_book::region_copyable(t));
    const std::vector<ggml_type> excluded = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_MXFP4, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_BF16};
    for (ggml_type t : excluded) CHECK(!llama_loan_book::region_copyable(t));   // R26's red case: true
    // every excluded name is one set_tensor handles specially (the rule's basis)
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "static void ggml_backend_opencl_buffer_set_tensor(",
                       "static void ggml_backend_opencl_buffer_get_tensor(",
                      {"if (tensor->type == GGML_TYPE_Q1_0)", "if (tensor->type == GGML_TYPE_Q4_0)",
                       "if (tensor->type == GGML_TYPE_Q4_1)", "if (tensor->type == GGML_TYPE_Q5_0)",
                       "if (tensor->type == GGML_TYPE_Q5_1)", "if (tensor->type == GGML_TYPE_MXFP4)",
                       "if (tensor->type == GGML_TYPE_Q8_0)", "if (tensor->type == GGML_TYPE_IQ4_NL)",
                       "if (tensor->type == GGML_TYPE_Q4_K)", "if (tensor->type == GGML_TYPE_Q5_K)",
                       "if (tensor->type == GGML_TYPE_Q6_K)", "if (tensor->type == GGML_TYPE_BF16)"}));
    // the loan's selection calls the rule, not the old != Q8_0 test
    CHECK(has_between("llama-expert-cache.cpp", "A layer is lendable when its three banks", "llama_loan_book::choose_lend_layers",
                      {"llama_loan_book::region_copyable"}));
}

TEST(the_red_switch_corrupts_one_lent_row_after_the_copies) {
    // Round 8, review item 1 of the card finding: MUTATE=1 skipped a copy, so the
    // slot's split extra was never built and the check's get_slices cast a plain
    // extra -- an Intel driver segfault with no ERROR line. It now flips one DATA
    // byte of a lent row after the copies; the check catches it and exits 70.
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::refill", "const int64_t t_write =",
                      {"lend_perturb(L.slot[2], 0)", "one data byte of the last lent layer's down row 0"}));
    CHECK(!has_between("llama-expert-cache.cpp", "bool llama_expert_cache::refill", "const int64_t t_write =",
                       {"rows stay at what the fresh"}));   // R27's old skip form gone
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "static void ggml_backend_opencl_lend_perturb(",
                       "// Reads n small plain tensors",
                      {"clEnqueueReadBuffer(queue, r[i].mem, CL_TRUE, at, 1, &b",
                       "b ^= 0xFF",
                       "clEnqueueWriteBuffer(queue, r[i].mem, CL_TRUE, at, 1, &b"}));
}

TEST(the_poison_drain_finishes_the_device_queues_first) {
    // Round 8, card finding: std::_Exit tore the address space down under in-flight
    // kernels/copies (xe page fault, GuC reset of the compute and copy engines,
    // timed to the exit). The poison policy drains every queue first, bounded.
    // Round 12: the bound is a named constant now, and the idle's wait follows the
    // drain (checked in the_exit_waits_for_the_rings_to_stop); this case keeps the
    // drain and the watchdog's bound in place.
    CHECK(has_between("llama-loan-book.h", "inline void poison_exit", "inline uint32_t ubatch_cap",
                      {"drain_hook()", "drain();", "std::this_thread::sleep_for(std::chrono::seconds(kWatchdogSeconds))"}));
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "static void ggml_backend_opencl_drain(void)",
                       "// Reads n small plain tensors",
                      {"clFinish(ctx->queue)", "clFinish(ctx->astream.q)", "clFinish(ctx->bstream.q)"}));
    CHECK(has_between("llama-expert-cache.cpp", "ggml_backend_opencl_drain", "set_alloc_host = (void (*)(bool))",
                      {"llama_loan_book::register_drain(drain)"}));   // R24's red case: no drain registered
}

TEST(a_watchdog_construction_failure_still_exits) {
    // Round 9, review item 4: std::thread's constructor can throw. Uncaught in the
    // poison policy it reaches std::terminate -- the process exits non-zero by
    // signal, not by the policy's 70, and the unit's restart sees a crash. The
    // spawn is guarded; the policy still drains and _Exit(70). R32's red case:
    // watchdog_spawn's catch returning true (or the try gone).
    bool ran = false;
    CHECK(!llama_loan_book::watchdog_spawn([&] { ran = true; throw 1; }));   // threw: no terminate
    CHECK(ran);
    CHECK(llama_loan_book::watchdog_spawn([] {}));                          // the normal spawn
    CHECK(has_between("llama-loan-book.h", "inline void poison_exit", "inline uint32_t ubatch_cap",
                      {"watchdog_spawn([]", "drain();", "std::_Exit(70)"}));
    CHECK(has_between("llama-loan-book.h", "template <typename Spawn>", "inline void poison_exit",
                      {"catch (...)", "return false"}));
}

TEST(the_zero_page_drain_precedes_the_copies) {
    // Round 8, review item 5: lend_zero's first use fills the zero page and
    // clFinish()es once; in round 7 that finish sat between the copies (the first
    // zero call came after the first write). The zero call is now first in the loop.
    std::vector<std::string> lines;
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::refill", "const int64_t t_write =",
                      {"lend_zero(L.slot[k]", "lend_write(L.slot[k]"}, &lines));
    int z = -1, w = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (z < 0 && lines[i].find("lend_zero(L.slot[k]") != std::string::npos) z = (int) i;
        if (w < 0 && lines[i].find("lend_write(L.slot[k]") != std::string::npos) w = (int) i;
    }
    CHECK(z >= 0 && w >= 0 && z < w);   // R28's red case: the write first
}

TEST(the_split_set_names_its_staleness) {
    // Round 8, review item 5's comment: iq4_nl_split_tensors is not cleared by a
    // weights buffer context reset, so a reset that rebuilds extras leaves a stale
    // entry; the comment says so and names the guard (the tensor's own extra).
    CHECK(scan_between("ggml/src/ggml-opencl/ggml-opencl.cpp",
                       "// IQ4_NL tensors whose specialised extra",
                       "std::unordered_set<const ggml_tensor *> iq4_nl_split_tensors;",
                      {"stale entry", "weights buffer", "the tensor's own extra"}));
}

// ---- round 14: the lent branch's red case, the driver's host memory ----

TEST(the_lent_table_can_be_put_off_by_one) {
    // Round 14, review item 3: the KL gate tests the lent branch, but no red arm
    // reached it -- every one exits 70 before scoring, and MUTATE=1 with
    // VERIFY=off is a FALSE red (its corruption lands after the prefill). MUTATE=5
    // moves one lent layer's lend-bank row index in the table the card reads during
    // the prefill; nothing compares a table row (the refill's check compares slot
    // bytes with the bank's), so the run refills, returns and scores. R49's red
    // case: the shift removed (row returned unchanged) -> the first CHECK fails.
    CHECK_EQ(llama_loan_book::mutate_loan_bank_row(0, 5, 10), 5);    // healthy: no shift
    CHECK_EQ(llama_loan_book::mutate_loan_bank_row(4, 5, 10), 5);    // round 13's MUTATE=4 is not this one
    CHECK_EQ(llama_loan_book::mutate_loan_bank_row(5, 5, 10), 6);    // R49's red case
    CHECK_EQ(llama_loan_book::mutate_loan_bank_row(5, 9, 10), 8);    // the last row moves inward, never past n_slots
    CHECK_EQ(llama_loan_book::mutate_loan_bank_row(5, -1, 10), -1);  // a skip entry stays a skip
    CHECK_EQ(llama_loan_book::mutate_loan_bank_row(5, 0, 1), 0);     // a one-row bank cannot move
}

TEST(the_lent_mutation_is_applied_at_the_lend_and_names_itself) {
    // The rule alone cannot prove the call site: the shift must be applied to the
    // host table before its non-blocking write (so the card reads it) and the line
    // must name the layer, expert and rows -- a red arm is only a red arm if its log
    // says the mutation was in the artefact the prefill used. R49's call-site red:
    // the block removed -> this scan fails.
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::lend()",
                      "// Round 7, the refill's types",
                      {"mutate_loan == 5", "L.lend_row_mutate = mutate_loan",
                       "LLAMA_SLOT_LOAN_MUTATE=5: layer %d: %d ex-slot experts",
                       "read the lend bank's rows", "one entry over"}));
}

TEST(the_lent_override_reaches_every_table_producer) {
    // Round 15, review item 1 (R54): MUTATE=5 shifted only tab_host, so
    // check_tables rebuilt `want` from loan_of, saw the difference and the lend
    // refused before the prefill -- no red could score. The override is a layer
    // member that fill_tables consults while lent: the upload, the check, the card
    // and the refill's restore then agree. It rotates EVERY ex-slot expert's row
    // in the layer, so the graph reads a wrong row for every routed slot expert.
    struct L {
        int32_t n_slots = 4;
        bool    lent = false;
        int32_t lend_row_mutate = 0;
        std::vector<int32_t> slot_of{0, 1, 2, 3};
        std::vector<int32_t> loan_of{-1, -1, -1, -1};
    };
    // the one producer: fill_tables' loan row, with the layer's override
    auto loan_row = [](const L & l, size_t e) {
        if (l.loan_of[e] < 0) return int32_t(-1);
        return llama_loan_book::mutate_loan_bank_row(l.lend_row_mutate, l.loan_of[e], l.n_slots);
    };
    // the lend's table half, then the override, as lend() sets it
    L l;
    for (size_t e = 0; e < l.slot_of.size(); ++e) {
        l.loan_of[e] = l.slot_of[e];
        l.slot_of[e] = -1;
    }
    l.lent = true;
    l.lend_row_mutate = 5;
    // the upload's host image and check_tables' want, both through fill_tables
    std::vector<int32_t> host(4), want(4);
    for (size_t e = 0; e < 4; ++e) host[e] = loan_row(l, e);
    for (size_t e = 0; e < 4; ++e) want[e] = loan_row(l, e);
    int bad = 0;
    for (size_t e = 0; e < 4; ++e) bad += host[e] != want[e];
    CHECK_EQ(bad, 0);                                    // the check would publish the lend
    CHECK_EQ(host, std::vector<int32_t>({1, 2, 3, 2}));  // every row moved (0->1, 1->2, 2->3, 3 in by one)
    int moved = 0;
    for (size_t e = 0; e < 4; ++e) moved += loan_row(l, e) != l.loan_of[e];
    CHECK_EQ(moved, 4);                                  // all four ex-slot experts, not one
    // the old form: the host image filled without the override, then one row
    // written outside fill_tables -> check_tables' want disagrees and refuses
    std::vector<int32_t> old(want);
    for (size_t e = 0; e < 4; ++e) old[e] = l.loan_of[e];
    bad = 0;
    for (size_t e = 0; e < 4; ++e) bad += old[e] != want[e];
    CHECK_EQ(bad, 4);                                    // round 14's refusal
    // the refill clears the override before slot_of is rebuilt: rows healthy
    l.lend_row_mutate = 0;
    for (size_t e = 0; e < 4; ++e) {
        l.slot_of[e] = l.loan_of[e];
        l.loan_of[e] = -1;
    }
    for (size_t e = 0; e < 4; ++e) CHECK_EQ(loan_row(l, e), int32_t(-1));
    // and the producers in the tree do this
    CHECK(has_between("llama-expert-cache.h", "struct llama_expert_cache_layer",
                      "struct llama_expert_cache_swap", {"lend_row_mutate"}));
    CHECK(has_between("llama-expert-cache.cpp", "void llama_expert_cache::fill_tables",
                      "void llama_expert_cache::upload_tables", {"L.lend_row_mutate"}));
    CHECK(has_between("llama-expert-cache.cpp", "// ---- tables: relayed before any graph reads them",
                      "flush_tables();", {"L.lend_row_mutate = 0"}));
}

TEST(the_loan_row_is_the_layers_last_row) {
    // Round 15, review item 3 (R55): init's tab_loan view used base 3 twice --
    // tab_row + 3 + 3 = row 6 of a six-row layer -- while fill_tables and the
    // mutation write row 5. A last cached layer with a Q8_0 down then asserts at
    // load, loan on or off. The view must be the layer's last row.
    CHECK(has_between("llama-expert-cache.cpp", "const int base = L.src[2]->type",
                      "ggml_format_name(L.tab_slot,", {"L.tab_loan = tab_view(L.tab_row + 3 + (L.tab_slot_dn != nullptr ? 2 : 0))"}));
    CHECK(has_between("llama-expert-cache.cpp", "void llama_expert_cache::fill_tables",
                      "void llama_expert_cache::upload_tables",
                      {"r + 3 * ne + (L.tab_slot_dn != nullptr ? 2 * ne : 0)"}));
}

TEST(the_kld_quantiser_is_perplexitys_half_to_even) {
    // Round 14, review item 5: perplexity.cpp:72-77's nearest_int rounds half to
    // even through the f32 + 12582912 trick; std::lround rounds half away from
    // zero, so the control arm could not compare row for row. R50's red case: the
    // trick replaced by std::lround (0.5 -> 1) -> the first CHECK fails.
    CHECK_EQ(loan_kld_book::nearest_int(0.5f), 0);
    CHECK_EQ(loan_kld_book::nearest_int(1.5f), 2);
    CHECK_EQ(loan_kld_book::nearest_int(2.5f), 2);
    CHECK_EQ(loan_kld_book::nearest_int(3.5f), 4);
    CHECK_EQ(loan_kld_book::nearest_int(0.0f), 0);
}

TEST(the_kld_logits_reservation_is_the_row_window) {
    // Round 14, review item 2: Flash-Next's vocab is 248,320, so n_batch (8,192)
    // rows of logits reserved 8.14 GiB while the capture reads n_ctx - n_ctx/2
    // rows. R51's red cases: output_rows() back to n_batch, or the driver's own
    // assignment removed; the row buffer back to n_rows rows.
    CHECK_EQ(loan_kld_book::output_rows(8192), 4096u);
    CHECK((int) loan_kld_book::output_rows(8192) >= loan_kld_book::n_rows(8192));
    // round 15: the context's reservation is one row window, not the whole capture
    CHECK_EQ(loan_kld_book::decode_width(loan_kld_book::n_rows(8192), 2048), 2048u);
    CHECK_EQ(loan_kld_book::n_decodes(loan_kld_book::n_rows(8192), 2048), 2);
    CHECK_EQ(loan_kld_book::rows_in_decode(loan_kld_book::n_rows(8192), 2048, 0), 2048);
    CHECK_EQ(loan_kld_book::rows_in_decode(loan_kld_book::n_rows(8192), 2048, 1), 2047);
    CHECK_EQ(loan_kld_book::rows_in_decode(loan_kld_book::n_rows(8192), 2048, 0) +
             loan_kld_book::rows_in_decode(loan_kld_book::n_rows(8192), 2048, 1),
             loan_kld_book::n_rows(8192));
    CHECK_EQ(loan_kld_book::row_buffer_bytes(248320), 496648u);   // 248,324 uint16 = one row
    std::ifstream f(std::string(ARCINT_SOURCE_DIR) + "/contrib/llama-loan-kld/loan_kld.cpp");
    CHECK(f.good());
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(src.find("const uint32_t width = loan_kld_book::decode_width") != std::string::npos);   // R56
    CHECK(src.find("params.n_outputs_max = width;") != std::string::npos);
    CHECK(src.find("std::vector<float> logits") == std::string::npos);   // the 8.14 GiB copy is gone
    CHECK(src.find("std::memcpy(logits.data()") == std::string::npos);
    CHECK(src.find("row.resize(nv)") != std::string::npos);   // one row, not n_rows x nv
    CHECK(src.find("row.resize((size_t) n_rows * nv)") == std::string::npos);
    CHECK(src.find("write_chunk_rows(out.stream, n_vocab, batch_logits") != std::string::npos);   // rows straight from the context
    // the rows are written before the loan's end (the end frees their buffer)
    const long w = (long) src.find("write_chunk_rows(out.stream");
    const long e = (long) src.find("llama_context_slot_loan_end(ctx.get())");
    CHECK(w >= 0 && e >= 0 && w < e);   // R51's order half
}

TEST(the_kld_scores_the_served_ubatch_in_row_windows) {
    // Round 15, review item 2 (R56): the loan's sizing takes n_outputs =
    // min(measure_ub, n_outputs_max), so a 4,096-row window puts 4.07 GB in the
    // lent graph's compute buffer; the lend is 6.20 GiB and the plan refuses. The
    // driver now caps n_outputs_max at a row window and scores the served 8,192-
    // token ubatch in two decodes, each under one loan: the rows are disjoint and
    // equal llama-perplexity's because each replay starts from an empty KV and a
    // causal row at position p depends only on tokens 0..p.
    std::ifstream f(std::string(ARCINT_SOURCE_DIR) + "/contrib/llama-loan-kld/loan_kld.cpp");
    CHECK(f.good());
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(src.find("loan_kld_book::n_decodes(n_rows, width)") != std::string::npos);   // R56
    CHECK(src.find("for (int p = 0; p < passes") != std::string::npos);
    CHECK(src.find("llama_memory_clear(llama_get_memory(ctx.get()), true);") != std::string::npos);
    CHECK(src.find("decode_width") != std::string::npos);
    CHECK(src.find("write_chunk_rows(out.stream, n_vocab, batch_logits") != std::string::npos);
    const long w = (long) src.find("write_chunk_rows(out.stream");
    const long e = (long) src.find("llama_context_slot_loan_end(ctx.get())");
    CHECK(w >= 0 && e >= 0 && w < e);   // the rows are read before the loan's end
}

TEST(the_kld_comments_state_what_the_end_and_the_reserve_do) {
    // Round 15, review item 4 (R57): the old comment said the end "frees the
    // buffer the logits live in" -- false: --ubatch-reserve makes the context's
    // resident re-reserve keep a buffer the logits can be read from; and the flag
    // parser treated --ubatch-reserve as optional while missing it exits 2.
    std::ifstream f(std::string(ARCINT_SOURCE_DIR) + "/contrib/llama-loan-kld/loan_kld.cpp");
    CHECK(f.good());
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(src.find("which frees the buffer the logits live in") == std::string::npos);   // R57
    CHECK(src.find("the rows are read BEFORE the loan's end") != std::string::npos);
    CHECK(src.find("--ubatch-reserve makes the") != std::string::npos);
    CHECK(src.find("end's resident re-reserve keep a buffer") != std::string::npos);
    CHECK(src.find("--ubatch-reserve N is required") != std::string::npos);
}

TEST(the_kld_driver_unlinks_a_partial_capture_and_says_the_rate_is_cold) {
    // Round 14, review item 5: a refusal after the header is written leaves a
    // truncated --out that the next window would compare as if it were a capture;
    // and warmup off means chunk 0's rate is cold, which the log must say.
    std::ifstream f(std::string(ARCINT_SOURCE_DIR) + "/contrib/llama-loan-kld/loan_kld.cpp");
    CHECK(f.good());
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) lines.push_back(l);
    auto pos = [&](const char * needle) {
        for (size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].find(needle) != std::string::npos) return (long) i;
        }
        return -1L;
    };
    const long guard = pos("struct out_file");          // R52's red case: the guard removed
    const long rm    = pos("std::remove(path.c_str())");
    const long cold  = pos("the first chunk's prefill rate is cold");
    const long commit = pos("out.committed = true;");
    const long loans = pos("loans != base_chunk");
    CHECK(guard >= 0 && rm >= 0);                       // R52
    CHECK(cold >= 0);                                   // R52's cold-rate line
    CHECK(commit > loans && loans > 0);                 // committed only after the last refusal
}

TEST(the_kld_on_arm_runs_the_served_ubatch_shape) {
    // Round 14, review item 4: -ub 2,048 under the loan's 8,192 reserve ran four
    // ubatches of 2,048, not the unit's one ubatch of 8,192. The driver sets the
    // resident reserve directly (llama_context_params.n_ubatch_reserve) and names
    // the shape; R53's red cases: served_shape() accepting a smaller ubatch, or
    // the reserve assignment removed.
    CHECK(loan_kld_book::served_shape(8192, 8192));
    CHECK(!loan_kld_book::served_shape(2048, 8192));   // R53
    std::ifstream f(std::string(ARCINT_SOURCE_DIR) + "/contrib/llama-loan-kld/loan_kld.cpp");
    CHECK(f.good());
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(src.find("cparams.n_ubatch_reserve = own.ubatch_reserve") != std::string::npos);   // R53's call site
    CHECK(src.find("served_shape((uint32_t) params.n_ubatch") != std::string::npos);
    CHECK(src.find("not the served shape") != std::string::npos);
}

// ---- round 22: the zero-margin rule, the guard's baseline and decay ----
// ---- round 21: the kernel query, the guard, the dirty mirror ----

TEST(the_kernel_query_sums_only_vram_regions_and_falls_back) {
    // Round 21: DRM_XE_DEVICE_QUERY_MEM_REGIONS is the card's own figure; the
    // pure rule sums ONLY VRAM regions (a sysmem region in the reply is not the
    // card's), and no VRAM region is a failed figure, never a zero that passes.
    std::vector<llama_loan_book::XeRegion> regs{
        {0, 100, 50},                              // sysmem: not counted
        {llama_loan_book::kXeVramClass, (uint64_t) 24 << 30, (uint64_t) 21 << 30},
        {llama_loan_book::kXeVramClass, (uint64_t) 1 << 30, (uint64_t) 2 << 30},
    };
    size_t total = 0, used = 0;
    CHECK(llama_loan_book::vram_from_regions(regs, total, used));
    CHECK_EQ(total, (size_t) 25 << 30);
    CHECK_EQ(used, (size_t) 23 << 30);
    CHECK(!llama_loan_book::vram_from_regions({{0, 100, 50}}, total, used));   // no VRAM
    // no driver header / no render fd: not a figure, the caller falls back
    const llama_loan_book::CardQuery q = llama_loan_book::xe_query_card(-1);
    CHECK(!q.ok);
    CHECK(!q.why.empty());
}

TEST(the_prefill_growth_allowance_is_the_only_addend_and_defaults_to_zero) {
    // Round 22 (R85): round 21's kernel reserve is gone from the rule. `used` is
    // the device-wide buddy usage and already holds every other client and the
    // kernel's own objects, so subtracting used - fdinfo counted them twice. The
    // only optional addend is the prefill's growth band, named for what it is,
    // default 0 (the operator's zero margin), env override in MiB. R85's red case:
    // the resolver's default non-zero -> the first three CHECKs fail; used - fdinfo
    // returned as a term -> the rule test below refuses a card with room.
    const size_t GiB = 1u << 30;
    const size_t MiB = 1u << 20;
    CHECK_EQ(llama_loan_book::resolve_prefill_allowance(nullptr), 0u);
    CHECK_EQ(llama_loan_book::resolve_prefill_allowance(""), 0u);
    CHECK_EQ(llama_loan_book::resolve_prefill_allowance("0"), 0u);      // the documented default
    CHECK_EQ(llama_loan_book::resolve_prefill_allowance("512"), (size_t) 512 * MiB);
    CHECK_EQ(llama_loan_book::resolve_prefill_allowance("sixty"), 0u); // a typo is 0, not a band
    CHECK_EQ(llama_loan_book::resolve_prefill_allowance("-8"), 0u);
    CHECK_EQ(llama_loan_book::resolve_prefill_allowance("999999"), llama_loan_book::kPrefillGrowthMax); // clamped
    // used - fdinfo is information (R84's other half): 0 without a query, the
    // difference with one, and never a wrap when the figures disagree
    CHECK_EQ(llama_loan_book::used_beyond_fdinfo(false, 24 * GiB, 23 * GiB), 0u);
    CHECK_EQ(llama_loan_book::used_beyond_fdinfo(true, 24 * GiB, 23 * GiB), 1 * GiB);
    CHECK_EQ(llama_loan_book::used_beyond_fdinfo(true, 3 * GiB, 4 * GiB), 0u);
}

TEST(the_card_rule_is_used_minus_lend_plus_need_against_the_total_with_zero_margin) {
    // Round 22 (R84): the device-wide `used` already holds every other client and
    // the kernel's own objects. The rule is
    // `used - lend + need + first_touch + allowance <= total`, and the margin
    // beyond `used` is zero. The round-21 form subtracted used - fdinfo as a
    // reserve, counting the other client (another process on the card, ~13.4 GiB in the
    // review's example) twice and refusing a card with room. R84's red case: the
    // limit back to `total - (used - vram0)` -> the first CHECK fails.
    const size_t GiB = 1u << 30;
    const size_t total = 24 * GiB;
    const size_t used  = 20 * GiB;    // 13.4 GiB of it is another client
    const size_t vram0 = 6 * GiB;     // this process's own residency
    const size_t lend  = 2 * GiB;
    const size_t need  = 1 * GiB;
    llama_loan_book::CardFacts f;
    f.total = total; f.used = used; f.need = need;
    f.query_ok = true; f.used_beyond = llama_loan_book::used_beyond_fdinfo(true, used, vram0);
    CHECK(llama_loan_book::card_fits(f, lend));            // after 18 + need 1 <= 24, 5 GiB spare
    // the round-21 form, kept as the record, refuses the same card: its limit was
    // total - (used - fdinfo) = 24 - 14 = 10 GiB, and after 18 alone exceeds it
    const size_t after   = used - lend;
    const size_t old_limit = total - f.used_beyond;
    CHECK(after + need > old_limit);                       // the double count's refusal
    // the boundary is the card's total, no margin below it: after + need at the
    // total fits, one byte over does not (after = used - lend)
    const size_t edge = total - need;                      // after at the line
    CHECK(llama_loan_book::card_state_fits(edge + lend, total, lend, need, 0, 0));
    CHECK(!llama_loan_book::card_state_fits(edge + lend + 1, total, lend, need, 0, 0));
    // the one allowance is a term, default 0
    CHECK(llama_loan_book::card_state_fits(edge + lend, total, lend, need, 0, 0));
    CHECK(!llama_loan_book::card_state_fits(edge + lend, total, lend, need, 0, 1));
    // the fallback (query failed): round 20's fdinfo + fixed margin form stays
    llama_loan_book::CardFacts g;
    g.total = total; g.resident = 20 * GiB; g.need = 4 * GiB; g.margin = 1 * GiB;
    g.query_ok = false;
    CHECK(!llama_loan_book::card_fits(g, 0));              // 20 + 4 > 23
    CHECK(llama_loan_book::card_fits(g, 1 * GiB));         // 19 + 4 <= 23
}

TEST(the_eviction_guard_acts_on_the_vram_shortfall_and_logs_the_gtt_growth) {
    // Round 22 (R93): the guard ACTS on vram_short alone and logs the GTT growth
    // (the same eviction's other face). Round 21 acted on GTT past the host banks,
    // which is the process's other legitimate GTT -- the stale ~4 GiB made every
    // begin shed. R93's red case: `g.bytes = std::max(vram_short, gtt_over)` back ->
    // the first CHECK sheds although the VRAM figure met the expectation.
    const size_t GiB = 1u << 30;
    // VRAM meets the expectation; GTT grew 4 GiB over the 2 GiB of banks added
    const auto g = llama_loan_book::eviction_guard(20 * GiB, 12 * GiB, 20 * GiB, 9 * GiB, 8 * GiB, 7 * GiB);
    CHECK(!g.shed && g.bytes == 0u);
    CHECK_EQ(g.vram_short, 0u);
    CHECK_EQ(g.gtt_growth, 4 * GiB);
    CHECK_EQ(g.bank_growth, 2 * GiB);
    CHECK_EQ(g.gtt_over, 2 * GiB);                        // logged, not acted on
    // VRAM below the expectation: the primary signal sheds exactly the shortfall
    const auto v = llama_loan_book::eviction_guard(19 * GiB, 8 * GiB, 20 * GiB, 7 * GiB, 8 * GiB, 7 * GiB);
    CHECK(v.shed && v.vram_short == 1 * GiB && v.gtt_growth == 0u && v.bytes == 1 * GiB);
    // no growth and no shortfall: nothing
    const auto n = llama_loan_book::eviction_guard(20 * GiB, 7 * GiB, 20 * GiB, 7 * GiB, 7 * GiB, 7 * GiB);
    CHECK(!n.shed && n.bytes == 0u);
}

TEST(a_false_gtt_surplus_at_the_begin_does_not_shed) {
    // Round 22 (R86): the begin takes its OWN GTT and host-bank bytes as the
    // baseline, so the process's legitimate GTT (the runtime's command buffers and
    // heaps, the stage ring, ids_host) is not a surplus and the begin cannot fire.
    // R86's red case: the begin judging against host_banks alone (base 0) -> the
    // CHECK fails with a false surplus of 2 GiB.
    const size_t GiB = 1u << 30;
    const auto b = llama_loan_book::eviction_guard(20 * GiB, 10 * GiB, 20 * GiB, 8 * GiB, 10 * GiB, 8 * GiB);
    CHECK(!b.shed && b.bytes == 0u);
    CHECK_EQ(b.gtt_growth, 0u);
    CHECK_EQ(b.gtt_over, 0u);
    // the context's begin passes exactly that baseline pair
    CHECK(has_between("llama-context.cpp", "Round 22's eviction guard at the loan's begin", "const size_t first_touch",
                      {"eviction_guard(vram0, gtt, sum, host_banks, gtt, host_banks)"}));
}

TEST(a_shed_decays_after_a_clean_loan) {
    // Round 22 (R87): round 21 carried a measured shed into every later loan, so
    // one trip (the unexplained ~4 GiB of extra GTT at 103k tokens) shrank the lend
    // for the process's lifetime. A loan that completes without a trip is clean:
    // the shed is cleared. R87's red case: shed_after_loan ignoring `tripped`
    // (returning shed always) -> the second CHECK fails.
    const size_t GiB = 1u << 30;
    CHECK_EQ(llama_loan_book::shed_after_loan(4 * GiB, true), 4 * GiB);
    CHECK_EQ(llama_loan_book::shed_after_loan(4 * GiB, false), 0u);
    CHECK_EQ(llama_loan_book::shed_after_loan(0u, false), 0u);
    // the end applies it with this loan's true shortfall (round 23: over the
    // floor) and the context keeps the result
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_end", "const bool back =",
                      {"growth_from_shortfall(g.vram_short, shed_floor)",
                       "shed_after_loan(short_beyond, guard_tripped)"}));
}

TEST(a_sub_floor_shortfall_does_nothing) {
    // Round 23 (R94): the end guard's expected VRAM is exact tensor bytes while
    // fdinfo vram0 is BO residency; allocator rounding and reserve_matches'
    // allowed under-run push vram0 a few MiB below `expected` on a healthy loan.
    // A 1-byte "shortfall" is no evidence of an eviction and must carry nothing.
    const size_t MiB = 1u << 20;
    const size_t GiB = 1u << 30;
    CHECK_EQ(llama_loan_book::kShedFloor, (size_t) 64 * MiB);
    CHECK_EQ(llama_loan_book::resolve_shed_floor(nullptr), (size_t) 64 * MiB);
    CHECK_EQ(llama_loan_book::resolve_shed_floor(""), (size_t) 64 * MiB);
    CHECK_EQ(llama_loan_book::resolve_shed_floor("128"), (size_t) 128 * MiB);   // the knob
    CHECK_EQ(llama_loan_book::resolve_shed_floor("0"), 0u);                     // explicit zero is a decision
    CHECK_EQ(llama_loan_book::resolve_shed_floor("typo"), (size_t) 64 * MiB);   // a typo is not a decision
    CHECK_EQ(llama_loan_book::resolve_shed_floor("999999"), llama_loan_book::kShedFloorMax);   // clamped
    const size_t floor = llama_loan_book::resolve_shed_floor(nullptr);
    CHECK_EQ(llama_loan_book::growth_from_shortfall(1, floor), 0u);       // the 1-byte trip
    CHECK_EQ(llama_loan_book::growth_from_shortfall(floor - 1, floor), 0u);
    CHECK_EQ(llama_loan_book::growth_from_shortfall(floor, floor), floor);  // at the floor: a true one
    CHECK_EQ(llama_loan_book::growth_from_shortfall(2 * GiB, floor), 2 * GiB);
    // the end guard reads the floor and keeps only a true shortfall
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_end", "const bool back =",
                      {"resolve_shed_floor", "growth_from_shortfall(g.vram_short, shed_floor)"}));
}

TEST(a_true_shortfall_grows_the_next_lend_and_the_need_rule_refuses_alone) {
    // Round 23 (R95): a true shortfall feeds the NEXT loan's sizing as extra lend
    // bytes through the existing growth path (`need + grow_by` as the demand, the
    // card's rule carrying it in `card.need`). Round 25 makes that two steps, so
    // the demand is never a refusal term: R103's case proves a short tail still
    // runs the loan at the `need` rule, and the refusal checks compare the actual
    // `need`.
    const size_t MiB = 1u << 20, GiB = 1u << 30;
    CHECK_EQ(llama_loan_book::growth_from_shortfall(512 * MiB, 64 * MiB), 512 * MiB);
    CHECK_EQ(llama_loan_book::growth_from_shortfall(2 * GiB, 64 * MiB), 2 * GiB);
    // values of the addend: need 5 GiB + a 2 GiB shortfall = 7 GiB of demand; the
    // same 7,000 MiB of lend fits `need` alone (6,528 MiB) and not the demand
    // (8,576 MiB), which is what the growth asks for and what the WARN may report
    const size_t need = 5 * GiB, shortfall = 2 * GiB, lend = 7000 * MiB;
    CHECK_EQ(need + shortfall, (size_t) 7 * GiB);   // the sizing's own addend
    CHECK(llama_loan_book::size_fits(need, lend));
    CHECK(!llama_loan_book::size_fits(need + shortfall, lend));
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_begin", "bool llama_context::slot_loan_end",
                      {"growth_from_shortfall(", "extend_lend(demand, beside, card, need)", "card.need = demand",
                       "not at a refusal"}));
    // the refusal checks still compare `need`, the shortfall's demand is not one
    CHECK(has_between("llama-context.cpp", "if (!llama_loan_book::size_fits(need, lend))",
                      "bool llama_context::slot_loan_end", {"size_fits(need, lend)", "the resident reserve comes back"}));
}

TEST(a_short_tail_still_runs_the_loan_at_what_need_alone_requires) {
    // Round 25 (R103), the carried shortfall's two-step growth: 26 layers lent
    // (5..30, 238 MiB each = 6,188 MiB), need 5 GiB (5,120 MiB) and a carried
    // 2 GiB shortfall -> demand 7,168 MiB. The 5 tail layers (0..4, +1,190 MiB)
    // cannot reach it: 7,378 MiB is under the demand's 8,576 MiB, so the old
    // all-or-nothing growth grew nothing. The ask alone does not meet the `need`
    // rule either (it needs 6,528 MiB), so the loan was refused; the tail's full
    // growth (31 layers, as far as the tail allows) does meet it and the loan
    // runs. Red case: step 2 dropped (`g.pick = ask;` when the demand is out of
    // reach) -> the pick is the ask, `size_fits(need)` fails and this case fails.
    const int64_t MiB = 1u << 20, layer = 238 * MiB;
    std::vector<llama_loan_book::LendCandidate> c;
    for (int i = 0; i < 31; ++i) c.push_back({true, layer});
    const auto ask = llama_loan_book::choose_lend_layers(c, (size_t) (6400 * MiB));
    CHECK_EQ(ask.chosen.size(), 26u);                 // the 26 layers lent
    CHECK_EQ(ask.bytes, 26 * layer);
    const size_t need = 5120 * MiB, shortfall = 2048 * MiB;
    const size_t demand = need + shortfall;
    CHECK(!llama_loan_book::size_fits(need, (size_t) ask.bytes));          // the ask alone is refused
    CHECK(!llama_loan_book::size_fits(demand, (size_t) (ask.bytes + 5 * layer)));   // the demand is out of reach
    const auto g = llama_loan_book::extend_lend_toward(c, ask, demand, need);
    CHECK(!g.reached_demand);                         // the demand is the WARN, not a refusal
    CHECK(g.pick.why == nullptr);                     // step 2 committed the tail's full growth
    CHECK_EQ(g.pick.chosen.size(), 31u);              // 26 + the 5 tail layers
    CHECK_EQ(g.pick.bytes, 31 * layer);
    CHECK(llama_loan_book::size_fits(need, (size_t) g.pick.bytes));   // the loan runs at what need requires
    // one tail layer is not enough by the rule's arithmetic, two are
    CHECK(!llama_loan_book::size_fits(need, (size_t) (ask.bytes + 1 * layer)));
    CHECK(llama_loan_book::size_fits(need, (size_t) (ask.bytes + 2 * layer)));
}

TEST(no_shortfall_state_survives_a_clean_loan_or_a_refused_begin) {
    // Round 23 (R96): round 21's shed survived a refused begin (a remainder the
    // grown tail could not cover refused the loan for the process's lifetime). The
    // begin now takes the shortfall and zeroes it before every refusal path, a
    // clean end stores 0, and no shed path or permanent refusal remains.
    const size_t MiB = 1u << 20, GiB = 1u << 30;
    CHECK_EQ(llama_loan_book::shed_after_loan(
                 llama_loan_book::growth_from_shortfall(1 * MiB, 64 * MiB), false), 0u);   // a sub-floor trip
    CHECK_EQ(llama_loan_book::shed_after_loan(
                 llama_loan_book::growth_from_shortfall(512 * MiB, 64 * MiB), false), 0u);  // a clean loan
    // values of the carry: a trip keeps the measured bytes, a clean loan zeroes them
    CHECK_EQ(llama_loan_book::shed_after_loan(512 * MiB, true), 512 * MiB);
    CHECK_EQ(llama_loan_book::shed_after_loan(2 * GiB, true), 2 * GiB);
    CHECK_EQ(llama_loan_book::shed_after_loan(2 * GiB, false), 0u);
    CHECK_EQ(llama_loan_book::growth_from_shortfall(1, 0), 1u);   // floor 0: every byte counts
    // the begin clears the state at its top, before any refusal returns
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_begin", "if (slot_loan_out()) {",
                      {"const size_t grow_by = llama_loan_book::growth_from_shortfall(", "slot_loan_shed = 0;"}));
    std::ifstream fc(std::string(ARCINT_LLAMA_SRC_DIR) + "/src/llama-expert-cache.cpp");
    CHECK(fc.good());
    std::string src((std::istreambuf_iterator<char>(fc)), std::istreambuf_iterator<char>());
    CHECK(src.find("shed_lend") == std::string::npos);   // the shed itself is gone
    CHECK(src.find("shed_at_begin") == std::string::npos);
    std::ifstream fx(std::string(ARCINT_LLAMA_SRC_DIR) + "/src/llama-expert-cache.h");
    std::string hx((std::istreambuf_iterator<char>(fx)), std::istreambuf_iterator<char>());
    CHECK(hx.find("shed_lend") == std::string::npos);
    CHECK(!has_between("llama-context.cpp", "bool llama_context::slot_loan_begin", "bool llama_context::slot_loan_end",
                       {"shed_lend", "shed_at_begin"}));
}

TEST(used_zero_with_a_total_is_no_card_figure) {
    // Round 23 (R97): a kernel without CAP_PERFMON reports used = 0 (`if
    // (perfmon_capable())` around used in older trees) while the total is real.
    // Accepting 0 as a figure makes the zero-margin rule vacuous -- every loan
    // fits. It is no figure: the query fails and the fdinfo fallback with its
    // margin decides, with one WARN (the caller's existing fallback line).
    const size_t GiB = 1u << 30;
    CHECK(!llama_loan_book::query_is_a_figure(0, 24 * GiB));   // R97's red case
    CHECK(!llama_loan_book::query_is_a_figure(0, 0));
    CHECK(!llama_loan_book::query_is_a_figure(24 * GiB, 0));   // no total: no rule
    CHECK(llama_loan_book::query_is_a_figure(1, 24 * GiB));
    CHECK(llama_loan_book::query_is_a_figure(20 * GiB, 24 * GiB));
    // the query path turns it into a failed query, so the fallback is taken
    CHECK(has_between("llama-loan-book.h", "inline CardQuery xe_query_card", "return q;",
                      {"query_is_a_figure(q.used, q.total)", "q.ok = false", "not a figure"}));
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_begin", "bool llama_context::slot_loan_end",
                      {"the kernel's memory-regions query failed", "the card rule falls back to"}));
}

TEST(the_query_reason_is_a_copy_safe_string) {
    // Round 23 (R98): CardQuery is returned by value; its reason lived in its own
    // `why_buf` behind a `const char * why`, so a copy's pointer pointed into the
    // source. The reason is a std::string now and the copy carries it.
    const llama_loan_book::CardQuery q = llama_loan_book::xe_query_card(-1);
    CHECK(!q.why.empty());
    const llama_loan_book::CardQuery copy = q;
    CHECK(copy.why == q.why);                    // R98's red case: the old copy's pointer dangled
    CHECK(std::string(copy.why).find("xe render fd") != std::string::npos);
    // no struct field points into a buffer of its own: the reason is owned, and
    // the old pointer-plus-buffer shape is gone (each token checked on its own --
    // has_between requires ALL of its words, so one list would pass on a revert
    // that brought back the buffer alone)
    CHECK(has_between("llama-loan-book.h", "struct CardQuery", "inline std::string card_why_errno",
                      {"std::string why;"}));
    CHECK(!has_between("llama-loan-book.h", "struct CardQuery", "inline std::string card_why_errno",
                       {"const char * why"}));
    CHECK(!has_between("llama-loan-book.h", "struct CardQuery", "inline std::string card_why_errno",
                       {"why_buf"}));
}

TEST(the_eviction_guard_knob_turns_the_guard_off) {
    // Round 22 (R88): LLAMA_SLOT_LOAN_EVICTION_GUARD=0 disables the guard; absent
    // is on, and any other word (a typo) keeps it on. R88's red case: the resolver
    // returning false for every value -> the first CHECK fails.
    CHECK(llama_loan_book::resolve_eviction_guard(nullptr));
    CHECK(llama_loan_book::resolve_eviction_guard("1"));
    CHECK(!llama_loan_book::resolve_eviction_guard("0"));
    CHECK(!llama_loan_book::resolve_eviction_guard("off"));
    CHECK(!llama_loan_book::resolve_eviction_guard("OFF"));
    CHECK(!llama_loan_book::resolve_eviction_guard("False"));
    CHECK(!llama_loan_book::resolve_eviction_guard("no"));
    CHECK(llama_loan_book::resolve_eviction_guard("nope"));            // a typo keeps it on
    CHECK(llama_loan_book::resolve_eviction_guard("08"));              // only the whole word 0 is off
    CHECK(llama_loan_book::resolve_eviction_guard("0x0"));
}

TEST(the_end_guard_sees_the_reserve_and_the_non_sum_residency) {
    // Round 22 (R92): round 21's end expected was the ggml sum minus the lend, blind
    // to the loan's compute reserve (still resident then) and to the process's
    // non-sum residency (the ring, the runtime's kernels and heaps) -- up to ~9 GiB
    // of eviction unseen. The expected now adds `need` and (vram0_begin - sum_begin),
    // and the GTT term judges the growth against the begin's baseline.
    // R92's red case: the `+ base.need + non_sum` addends dropped -> the scan fails.
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_end", "const bool back =",
                      {"const size_t non_sum = base.vram0 > base.sum ? base.vram0 - base.sum : 0",
                       "+ base.need + non_sum",
                       "eviction_guard(agg.vram0, agg.gtt, expected, host_banks, base.gtt, base.banks)"}));
    // the begin records the non-sum and the reserve for the end, per card
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_begin", "bool llama_context::slot_loan_end",
                      {"slot_loan_card_base{dev, sum, vram0, gtt, host_banks, 0, rd.measured}", "b.need = c.second.need;"}));
}

TEST(fdinfo_reads_tib_and_the_query_why_carries_errno) {
    // Round 22 (R89/R90). R89: fdinfo_parse_bytes' unit list lacked TiB, so a
    // figure reported in TiB was read as raw bytes and understated by 2^40 (a GTT
    // or a card above 1 TiB). Red case: the TiB line removed -> the CHECKs fail.
    // R90: a failed ioctl's why was a bare reason, hiding ENOTTY/EPERM/EINVAL,
    // which ask different next steps. Red case: the errno suffix removed from
    // card_why_errno -> the errno CHECK fails.
    CHECK_EQ(llama_loan_book::fdinfo_parse_bytes("1 TiB"), (size_t) 1 << 40);
    CHECK_EQ(llama_loan_book::fdinfo_parse_bytes("2 TiB"), (size_t) 2 << 40);
    CHECK_EQ(llama_loan_book::fdinfo_parse_bytes("12 TiB"), (size_t) 12 << 40);
    CHECK_EQ(llama_loan_book::fdinfo_parse_bytes("1 GiB"), (size_t) 1 << 30);   // the neighbour unit
    // a TiB figure through the scan: the fake fdinfo's gtt reads it
    const std::string dir = fake_fdinfo_dir({
        {"7", fdinfo_text("xe", "0000:03:00.0", "7", "20 GiB", "2 TiB")},
    });
    llama_loan_book::FdinfoScan scan;
    llama_loan_book::scan_fdinfo(dir.c_str(), scan);
    CHECK(scan.clients.size() == 1u && scan.clients[0].gtt == (size_t) 2 << 40);
    std::filesystem::remove_all(dir);
    // Round 23: the reason is a std::string the query owns (R98: a copy of the
    // returned query must keep its reason -- the old const char * pointed into
    // the source's why_buf after the copy).
    const std::string w = llama_loan_book::card_why_errno("DRM ioctl failed", 22);
    CHECK(w.find("DRM ioctl failed") == 0);
    CHECK(w.find("errno 22") != std::string::npos);
    CHECK(w.find("Invalid argument") != std::string::npos);   // strerror's text
}

TEST(the_load_probe_runs_only_for_a_cache_with_a_lend) {
    // Round 22 (R91): the load probe -- an fdinfo scan, an ioctl and a WARN line --
    // ran for every cache, including the agent's and the coder's with the loan off.
    // It is gated on lend_bytes_ != 0. Red case: the gate removed -> the scan fails
    // (the probe body would sit outside the brace).
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::init", "return true;",
                      {"if (lend_bytes_ != 0) {", "card probe at load", "used_beyond_fdinfo(q.ok, q.used, vram0)"}));
    // and no rule term names the old reserve
    CHECK(!has_between("llama-expert-cache.cpp", "bool llama_expert_cache::init", "return true;",
                       {"kernel reserve"}));
}

TEST(a_swap_marks_the_slot_dirty) {
    // Round 21 red-first, R74: a swap marks its slot (not a host write); runs()
    // gives the exact contiguous rows the next lend copies. Mutation: mark()
    // writing nothing -> rows 0, this case fails.
    llama_loan_book::DirtyTrack d;
    d.reset(4);
    CHECK(!d.any());
    CHECK_EQ(d.rows(), 0u);
    d.mark(2);
    CHECK(d.any() && d.rows() == 1u);
    const auto r1 = d.runs();
    CHECK(r1.size() == 1u && r1[0].first == 2 && r1[0].second == 1);
    d.mark(3);
    const auto r2 = d.runs();
    CHECK(r2.size() == 1u && r2[0].first == 2 && r2[0].second == 2);   // one run, not two
    d.mark(0);
    const auto r3 = d.runs();
    CHECK(r3.size() == 2u && r3[0].first == 0 && r3[1].first == 2);
    d.clear();
    CHECK(!d.any());
}

TEST(the_lend_copies_exactly_the_dirty_rows) {
    // R75: the flush reads the dirty runs from the card's slot tensors through
    // get_slices and writes them into the lend bank through set_slices -- nothing
    // else. Mutation: `L.dirty.runs()` replaced by every slot -> the exact-run
    // scan fails; `L.dirty.clear()` removed -> the run is copied twice.
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::flush_lend_dirty",
                      "// The book's callbacks",
                      {"L.dirty.runs()", "get_slices(L.slot[k]", "set_slices(L.lend[k]", "L.dirty.clear()",
                       "img.resize((size_t) count * L.nb2[k])"}));
    // the flush is at the lend's start, after finish_pending, before the free
    CHECK(has_between("llama-expert-cache.cpp", "bool llama_expert_cache::lend()", "const bool mutate_tables",
                      {"finish_pending()", "flush_lend_dirty("}));
}

TEST(no_per_swap_host_write_remains_for_lendable_layers) {
    // R76: the ~1.5 MiB per-swap host write (lend_mirror) is gone from both swap
    // sites; they mark the slot. Mutation: one site back to lend_mirror -> the
    // name reappears and this scan fails. The old function itself is gone too.
    std::ifstream f(std::string(ARCINT_LLAMA_SRC_DIR) + "/src/llama-expert-cache.cpp");
    CHECK(f.good());
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(src.find("lend_mirror") == std::string::npos);                 // the mirror is gone
    CHECK(src.find("void llama_expert_cache::mark_lend_dirty") != std::string::npos);
    CHECK(src.find("ec->mark_lend_dirty(ec->layers[p.layer], p.slot)") != std::string::npos);  // the async site
    CHECK(src.find("mark_lend_dirty(L, c.slot)") != std::string::npos);  // the direct exchange site
    // the mark itself does no host read/write -- mutation: a set_slices there
    const size_t m0 = src.find("void llama_expert_cache::mark_lend_dirty");
    const size_t m1 = src.find("// The flush", m0 == std::string::npos ? 0 : m0);
    CHECK(m0 != std::string::npos && m1 != std::string::npos && m1 > m0);
    CHECK(src.substr(m0, m1 - m0).find("set_slices") == std::string::npos);
    CHECK(src.substr(m0, m1 - m0).find("get_slices") == std::string::npos);
    // the context frees every cache's tail after all lends, so the flush precedes
    // every free (the flush runs inside lend, before the tables relay)
    CHECK(has_between("llama-context.cpp", "the lend: tables relayed", "bool llama_context::slot_loan_end",
                      {"ec->lend()", "ec->drop_tail_buffer()"}));
}

TEST(the_guard_reads_gtt_and_the_query_from_a_fake_fdinfo) {
    // R77: device-free: a fake fdinfo directory carries drm-resident-gtt, the
    // scan parses it, and the guard reads the evicted bytes from those numbers.
    // Mutation: the gtt parse removed -> gtt 0, no shed.
    const size_t GiB = 1u << 30;
    const std::string dir = fake_fdinfo_dir({
        {"3", fdinfo_text("xe", "0000:03:00.0", "7", "20 GiB", "8 GiB")},
        {"4", fdinfo_text("xe", "0000:41:00.0", "8", "4 GiB", "2 GiB")},
    });
    llama_loan_book::FdinfoScan scan;
    llama_loan_book::scan_fdinfo(dir.c_str(), scan);
    const llama_loan_book::FdinfoAgg hit = llama_loan_book::sum_fdinfo(scan, "0000:03:00.0");
    CHECK(hit.measured && hit.clients == 1 && hit.gtt == 8 * GiB);
    // R77: the gtt parse feeds the guard; at a begin the baseline is the same
    // figure, so a surplus here is the process's own and is not acted on (R86)
    const auto b = llama_loan_book::eviction_guard(hit.vram0, hit.gtt, hit.vram0, 7 * GiB,
                                                   hit.gtt, 7 * GiB);
    CHECK(!b.shed && b.gtt_over == 0u && b.gtt_growth == 0u);
    // an end where GTT grew 2 GiB over the 1 GiB of banks added: logged, no shed
    const auto g = llama_loan_book::eviction_guard(hit.vram0, hit.gtt, hit.vram0, 8 * GiB,
                                                   6 * GiB, 7 * GiB);
    CHECK(!g.shed && g.gtt_over == 1 * GiB && g.gtt_growth == 2 * GiB && g.bank_growth == 1 * GiB);
    std::filesystem::remove_all(dir);
}

TEST(the_guard_fires_at_the_loan_end_before_the_refill) {
    // R79: the loan's end reads fdinfo and the query before the resident
    // re-reserve/refill; expected VRAM is the ggml sum minus the card's lend.
    // Mutation: the block moved after slot_loan_reserve -> the order scan fails.
    std::vector<std::string> lines;
    CHECK(has_between("llama-context.cpp", "bool llama_context::slot_loan_end", "const bool back = mutate_loan_ctx",
                      {"scan_fdinfo(\"/proc/self/fdinfo\"", "xe_query_card", "eviction_guard",
                       "base.sum > lend_card", "+ base.need + non_sum", "growth_from_shortfall(g.vram_short, shed_floor)"}, &lines));
}

#endif  // ARCINT_LLAMA_SLOT_LOAN
