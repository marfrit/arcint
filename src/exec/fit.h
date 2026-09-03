#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// M7 — auto-fit and the honest reservation (design: "M7 — Auto-fit and the
// honest reservation", 2026-09-01, §1/§2/§4). Pure arithmetic, no OpenVINO
// types, so it builds and runs on any host: the load path in backend_ov.cpp
// measures every term on the card and hands them here; this file only adds
// them up and floors the result to a page.
//
// The defect this removes: with --offload-ratio > 0 the expert slot pool
// commits physical memory at first touch (a deferred allocation, not a lie
// at request time), but nothing in the pre-M7 budget charged for it, so the
// printed max ctx was optimistic by roughly the resident expert share
// (DESIGN §7.0.2s: measured max ctx 1,172,016 on a card that could not serve
// it). `FitTerms::slot_pool` and `FitTerms::drafters` are the two terms that
// close that gap; everything else mirrors the budget load_paged() already
// computed.
namespace lgc {

// Every term is bytes already measured or already decided, never a
// percentage or a policy knob — see design §2's "budget = total - weights -
// drafters - slot_pool - activation_total - lanes*slab - margin".
struct FitTerms {
    uint64_t total          = 0;  // device_total_mem_size
    uint64_t weights        = 0;  // resident after the main model compile
    uint64_t drafters       = 0;  // resident delta from embeddings/MTP/DFlash compiles
    uint64_t slot_pool      = 0;  // expert slot pool (0 when --offload-ratio is 0)
    uint64_t activations    = 0;  // measured at the served chunk, all lanes together
    uint64_t slab_per_lane  = 0;  // GDN checkpoint rows, per lane
    uint64_t kv_bytes_token = 0;  // per lane, per token
    uint64_t margin         = 0;  // --fit-margin-mib

    int lanes           = 1;
    int kv_block_tokens = 1;  // KV page size in tokens; max_ctx floors to a multiple of this
    int n_ctx_floor      = 0;  // below this, a fit is not worth serving (see shrink_n_ctx callers)
};

struct FitResult {
    long long max_ctx       = 0;  // per lane, floored to a kv_block_tokens multiple
    uint64_t  reserved_total = 0;  // bytes committed if the pool is allocated at max_ctx
    bool      admissible    = false;  // max_ctx reaches n_ctx_floor and is > 0
};

// budget = total - weights - drafters - slot_pool - activations - lanes*slab - margin
// max_ctx = floor_to_page(budget / kv_bytes_token / lanes), floored again at 0.
inline FitResult fit_context(const FitTerms& t) {
    const int lanes      = std::max(1, t.lanes);
    const int kv_block    = std::max(1, t.kv_block_tokens);

    const long long fixed =
        static_cast<long long>(t.weights) + static_cast<long long>(t.drafters) +
        static_cast<long long>(t.slot_pool) + static_cast<long long>(t.activations) +
        static_cast<long long>(t.margin) +
        static_cast<long long>(lanes) * static_cast<long long>(t.slab_per_lane);
    const long long budget = static_cast<long long>(t.total) - fixed;

    long long max_ctx = 0;
    if (budget > 0 && t.kv_bytes_token > 0) {
        max_ctx = budget / static_cast<long long>(t.kv_bytes_token) / lanes;
        max_ctx = (max_ctx / kv_block) * kv_block;  // floor to a page multiple
    }
    if (max_ctx < 0) max_ctx = 0;

    FitResult r;
    r.max_ctx = max_ctx;
    r.reserved_total =
        static_cast<uint64_t>(fixed) +
        static_cast<uint64_t>(lanes) * static_cast<uint64_t>(max_ctx) * t.kv_bytes_token;
    r.admissible = max_ctx > 0 && max_ctx >= static_cast<long long>(t.n_ctx_floor);
    return r;
}

// slots(m) = ceil(num_expert * (100 - ratio_pct) / 100) -- see design §2: the
// plugin's own rounding lives in ops/moe.cpp, out of this repository's tree
// (patches/0003 here only touches expert-mask subbuffer caching and does not
// carry that expression), so this is the ceiling fallback the design names:
// it over-reserves rather than under-reserves, pending an on-card audit
// against MOE_OTD_PERF_LOG. `per_expert_bytes` is already the sum over one
// MoE layer's expert-weight inputs (gate/up/down + scales/zp) with the expert
// axis dropped; `moe_layers` is how many such layers the model has.
inline uint64_t expert_slot_bytes(int num_expert, int ratio_pct, uint64_t per_expert_bytes,
                                  int moe_layers) {
    if (num_expert <= 0 || per_expert_bytes == 0 || moe_layers <= 0) return 0;
    const int pct = std::clamp(ratio_pct, 0, 100);
    const uint64_t n = static_cast<uint64_t>(num_expert);
    const uint64_t slots = (n * static_cast<uint64_t>(100 - pct) + 99) / 100;  // ceil
    return slots * per_expert_bytes * static_cast<uint64_t>(moe_layers);
}

// M8 bug 1 (docs/design-m8-asymmetric-kv.md §3, first pickup check):
// backend_ov.cpp used to sum `ov::element::Type::size()` per KV pool port,
// and `size()` ceils a sub-byte width to a whole byte -- an i4 port reads
// back as 1 byte, the same as u8, so a KV pool with an i4 side was
// over-counted 2x.
//
// Each port is a physically separate device allocation, so each one's own
// byte size must ceil independently -- summing every port's bits into one
// total and dividing by 8 once (the first version of this fix, review
// 2026-09-02, F7) got that backwards: it can let one port's leftover bits
// complete another port's byte in the AGGREGATE, floors the whole sum, and
// silently undercounts the real allocation whenever any single port's own
// bit total is not itself byte-aligned (an i4 port with an odd element
// count, say) -- anti-conservative for the fit, the wrong direction to be
// wrong in. Each port is ceiled on its own instead: `(bitwidth * count + 7)
// / 8` still fixes the original over-count for the byte-aligned f16/u8/i8
// cases and for any i4/u4 port with an even element count (unchanged from
// before), and now also gets the odd-count case right instead of merely
// getting it right by accident.
//
// `ports` is (bitwidth, element_count) per key_cache/value_cache port --
// pure arithmetic, no OpenVINO types, so backend_ov.cpp's actual port walk
// can be tested here without a card.
inline uint64_t kv_block_bytes_from_bits(
    const std::vector<std::pair<uint64_t, uint64_t>>& ports) {
    uint64_t bytes = 0;
    for (const auto& [bitwidth, count] : ports) bytes += (bitwidth * count + 7) / 8;
    return bytes;
}

// RETRACTED: kv_cost_bitwidth, a requested-precision override for the KV
// cost model (measurement discipline, DESIGN §7.0.1: a mechanism that is
// narrated and not measured gets retracted on the record rather than
// edited away). Two on-card rounds, both wrong:
//
//   (1) First version: override the per-port bitwidth to the REQUESTED
//       precision whenever it was 4-bit (u4/i4), reasoning that this
//       plugin generation always types paged KV ports at 8 bits regardless
//       of packing. Wrong scope: it fired for every port on a 4-bit
//       request, including scale/auxiliary pool ports genuinely typed at
//       16 or 32 bits, charging them 4 bits too.
//   (2) Narrowed version: fire only when the actual port was also 8-bit
//       (the known 4-in-8 TYPE-level shape). --paged-kv u4 still printed
//       3.2 KiB/token after this narrowing -- the number did not move.
//       Solving KV_pure/4 + fixed = 3.2 against KV_pure + fixed = 11.3 (the
//       u8 baseline) gives KV_pure ~= 10.8 KiB/token, fixed ~= 0.5; WITHOUT
//       any override the predicted figure is KV_pure/2 + fixed ~= 5.9,
//       inside the expected ~5.65-5.9 band. This plugin generation encodes
//       4-bit packing in the port SHAPE (element count already halved for
//       a 4-bit request), not only the type -- so kv_block_bytes_from_bits
//       fed the compiled port's own (bitwidth, count) directly, with no
//       separate override, was already correct, and every override tried
//       here was double-halving. Corroborated separately (a prior window):
//       an honest i4-TYPED port attempt showed arcint's own tensor sized
//       exactly 2x the plugin's -- direct evidence of shape-level packing.
//
// The port AUDIT (config.h's kv_precision_bitwidth_matches /
// kv_precision_is_packed_four_bit) is unaffected by this retraction and
// stays as amended: TYPE-level 4-in-8 aliasing is real, measured, and must
// still pass rather than refuse a legitimate 4-bit request. Only the COST
// model's separate bitwidth override was wrong. backend_ov.cpp's port walk
// now feeds kv_block_bytes_from_bits the compiled port's own bitwidth
// unconditionally -- see the comment at that call site for the full
// history.

// The replay loop's correction (design §1 "The replay loop"): given the
// driver reported `overshoot` bytes more than the budget allowed, shrink
// n_ctx by the tokens needed to free that many bytes across every lane, then
// floor to a page. Termination is structural: `overshoot` is only ever
// nonzero, so the reduction is always >= 1 token, and flooring on a value
// that started page-aligned always drops at least one full page -- the
// caller does not have to prove that separately.
inline long long shrink_n_ctx(long long n_ctx, uint64_t overshoot, uint64_t kv_bytes_token,
                              int lanes, int kv_block_tokens) {
    if (overshoot == 0 || kv_bytes_token == 0) return n_ctx;
    const int lanes_c = std::max(1, lanes);
    const int block    = std::max(1, kv_block_tokens);

    const uint64_t per_token = static_cast<uint64_t>(kv_bytes_token) * static_cast<uint64_t>(lanes_c);
    const long long reduce_tokens =
        static_cast<long long>((overshoot + per_token - 1) / per_token);  // ceil(over / (kv*lanes))

    long long next = n_ctx - reduce_tokens;
    next           = (next / block) * block;  // floor to a page multiple
    if (next >= n_ctx) next = n_ctx - block;   // guarantee a strict decrease of >= one page
    if (next < 0) next = 0;
    return next;
}

// M7 defect fix (measured 2026-09-03, B60/24 GiB card -- 22.71 GiB usable,
// qwen38 27B dense int4, u8 paged KV at ~36.2 KiB/token, explicit --n-ctx
// 32768, --prefix-cache-mib 8192, one lane): an explicit --n-ctx well under
// the admissible maximum was refused at allocation time -- "reservation
// overshoot: 22.53 GiB resident against a 22.46 GiB ceiling after
// allocating n_ctx 32768" -- while the SAME artifact with --n-ctx omitted
// adopted 155,568 (corrected to 155,376, 0 spare, after the audit) and a
// second run that night adopted 171,488 (corrected to 171,312, also 0
// spare) -- both served at essentially the reported ~22.5 GiB residency.
// The residency did not track the explicit request because it was never
// sized from the request alone: DESIGN's "spare for cached prefixes"
// mechanism sizes the paged KV pool in BYTES from whatever budget the live
// request does not use, handing the rest to the prefix cache as reserve
// pages -- `backend_ov.cpp`'s allocate-audit-replay loop always computed
// this (the inline arithmetic this function replaces), but the loop's
// overshoot handling for an explicit request never read the split it
// produced: any overshoot at all was treated as the request not fitting,
// when in the reported case the live request was provably admissible (the
// `wanted > max_ctx` check above this pool is ever sized already
// guarantees it) and the pages that overshot were the spare reserve.
//
// Round-two on-card cells, same log format the retry line below prints,
// "(spare of blocks pages)": f1b (this card, explicit 32,768) pass 1 was
// 6,503 of 8,554 pages, trimmed 129 to the accepted 6,374 of 8,425; f2b
// (same request, DFlash resident) pass 1 was 4,700 of 6,751, trimmed 129
// to 4,571 of 6,622.
//
// Only the B60/24 GiB cells above are measurements of the defect itself. A
// second card (16 GiB) was checked only AFTER this fix landed, not
// independently root-caused to the same mechanism beforehand: with the fix
// applied, an explicit --n-ctx 65,536 on it (f4b) had pass 1 at 3,188 of
// 7,286 pages, trimmed 129 to the accepted 3,059 of 7,157 -- consistent
// with the same mechanism, not a second measurement of the defect.
//
// `live_blocks` is what the request itself needs (n_ctx/kv_block_tokens,
// summed over lanes, plus the drafter/lookahead margin the caller already
// adds); `wanted_spare_blocks` is how many extra pages the prefix cache
// would like (derived from its own host-side budget, independent of
// n_ctx); `budget_remaining_bytes` is the fit budget minus whatever prior
// replay passes have already learned overshot. Pure arithmetic mirroring
// backend_ov.cpp's per-pass block count exactly, so the spare-vs-live split
// -- and the retry that must be tried before an explicit request is ever
// refused -- is testable without a card.
struct PoolSizing {
    size_t blocks       = 0;  // live + spare this pass would request
    size_t spare_blocks = 0;  // the part beyond live_blocks (cache reserve)
};

inline PoolSizing pool_sizing(size_t live_blocks, size_t wanted_spare_blocks,
                              long long budget_remaining_bytes, uint64_t kv_block_bytes) {
    PoolSizing r;
    r.blocks = live_blocks;
    if (budget_remaining_bytes > 0 && kv_block_bytes > 0) {
        const size_t affordable = static_cast<size_t>(budget_remaining_bytes) / kv_block_bytes;
        const size_t spare_room = affordable > live_blocks ? affordable - live_blocks : 0;
        r.spare_blocks = std::min(spare_room, wanted_spare_blocks);
    }
    r.blocks += r.spare_blocks;
    return r;
}

// Round-3 review, defect 1: a driver allocation granule wider than one KV
// page (measured: coder/16 GiB, --prefix-cache-reserve 25 -- pass 1
// overshoot 14.86 GiB against a 14.86 GiB ceiling at n_ctx 100224, then
// 100080/100064/100048 with residency UNCHANGED at 14.86 GiB, exhausted;
// and the 35B/24 GB card's plain auto-fit cell, same shape at 262,144 ->
// 260080/260064/260048) means a trim sized only from the measured `over`
// can come out smaller than what it takes to move the driver's own
// rounding -- the accepted bytes stay pinned at the same rounded value
// and the replay loop never converges within its four passes. Both
// spare-trimming call sites (the explicit retry's `explicit_retry_spare_
// cap` below and the auto-fit branch's `auto_fit_trim`, further down)
// need a trim that grows on its own when the analytic figure is not
// enough -- `pool_pages_to_trim` is that shared floor.
//
// `pass` is the index (0-based) of the pass that just failed.
// `auto_fit_backoff_pages` escalates geometrically, base 4 STARTING AT 4
// (not 1): pass 0 floors at 4 pages, pass 1 at 16, pass 2 at 64, pass 3
// at 256.
//
// Round-4 review correction: an earlier version of this schedule started
// at 1 (pass 0 floors at 1, pass 1 at 4, pass 2 at 16, pass 3 at 64) and
// its own comment claimed convergence "up to 32 pages" -- false. The
// replay loop (backend_ov.cpp's Phase E, mirrored exactly by
// converge_passes in test_fit.cpp) checks convergence at the START of
// each pass, against whatever the PREVIOUS pass's trim already applied;
// the trim computed on the pass that fails the fourth and final check is
// never allocated or re-checked, so only the trims from the first THREE
// failed passes ever land. With the pass-0 schedule that is pass 0 + pass
// 1 + pass 2's floors -- 1 + 4 + 16 = 21 pages -- not the 64-page pass-3
// floor the old comment (wrongly) treated as the guarantee. 21 pages
// cannot guarantee crossing an arbitrary 32-page driver rounding
// boundary; the round-3 test's own "up to 32" result held only because
// its particular fixture's boundary happened to sit 16 pages below the
// starting pool, not because the schedule guaranteed it for any
// alignment.
//
// The corrected schedule floors this same three-trim, pre-last-pass
// total at 4 + 16 + 64 = 84 pages -- CONVERGES for any driver rounding
// granule up to and including 84 pool pages, for any alignment, provided the
// first pass's overshoot is itself below one granule (a pure rounding
// artifact; an overshoot of a granule plus a page spends pass 0 on the
// analytic excess and the guarantee drops to 81); a larger
// granule genuinely cannot be guaranteed within four passes and refuses
// loudly instead (the pre-existing "replay exhausted" throw, itemized --
// see the throw's own comment, round-4). A genuine analytic trim larger
// than the floor (the 129-page f1b case, an actual measured overshoot,
// not a rounding artifact) is left alone either way: this only RAISES a
// trim that would otherwise come out smaller than the driver's own
// granule, it never lowers one that is already big enough -- and the
// measured zero-spare auto-fit cells (analytic 11- and 12-page trims on
// pass 0) are untouched too, since 11 and 12 both exceed the new pass-0
// floor of 4.
inline size_t auto_fit_backoff_pages(int pass) {
    size_t trim = 4;
    for (int i = 0; i < pass && i < 8; ++i) trim *= 4;
    return trim;
}

inline size_t pool_pages_to_trim(uint64_t over, uint64_t kv_block_bytes, int pass) {
    const size_t analytic = kv_block_bytes > 0
        ? std::max<size_t>(static_cast<size_t>(1),
                           static_cast<size_t>((over + kv_block_bytes - 1) / kv_block_bytes))
        : static_cast<size_t>(1);
    return std::max(analytic, auto_fit_backoff_pages(pass));
}

// F1 (review finding, HIGH): the explicit-n_ctx retry must shrink the
// spare pool by at least one KV page every pass, the same guarantee
// shrink_n_ctx already gives the auto-fit path. Feeding `budget_remaining
// -= over` into pool_sizing alone does not have this property: `affordable
// = budget_remaining / kv_block_bytes` is a floor division, and a residual
// `over` smaller than one page can leave `budget_remaining` inside the SAME
// page bucket pass to pass, so `blocks` comes out byte-identical to the
// pass that just overshot -- the retry burns all four passes into the
// terminal "arithmetic above is wrong" refusal of an admissible request.
//
// This function does not take `live_blocks` or `n_ctx` at all -- by
// construction it can only ever narrow the spare bound the caller hands
// `pool_sizing` next pass, never the request. Called from backend_ov.cpp's
// explicit-n_ctx retry as the wanted-spare bound for the NEXT pass (capped
// against whatever the prefix cache already wanted, so a shrinking cache
// budget is still respected).
//
// `pass` (round-3 review, defect 1): the trim now floors at
// `pool_pages_to_trim`'s geometric backoff, not just the analytic
// ceil(over/kv_block_bytes) -- the f1b record (129-page trim on pass 0)
// is unaffected: `auto_fit_backoff_pages(0) == 4`, strictly smaller than
// 129, so `max(129, 4) == 129` reproduces the same trim exactly (kept
// test: explicit_retry_spare_cap_reproduces_the_f1b_trim).
inline size_t explicit_retry_spare_cap(size_t spare_blocks, uint64_t over,
                                       uint64_t kv_block_bytes, int pass) {
    const size_t trim = pool_pages_to_trim(over, kv_block_bytes, pass);
    return spare_blocks > trim ? spare_blocks - trim : 0;
}

// F4 (review finding, MEDIUM): a raw allocation exception on an explicit
// --n-ctx (fragmentation, no residency reading -- there is no measured
// `over` to size a trim from) used to fall straight through to refusal
// whenever spare pages were sitting right there to give up -- the same
// misattribution the overshoot branch had, blaming --n-ctx for the cache's
// reserve pages. Mirrors the auto-fit branch's own synthesized retreat
// (backend_ov.cpp's `over == 0` guard: a fixed-fraction retry instead of
// repeating the identical failed request): halve the spare pool, trimming
// at least one page, same termination shape as explicit_retry_spare_cap.
inline size_t synthesize_spare_retreat(size_t spare_blocks) {
    if (spare_blocks == 0) return 0;
    const size_t trim = std::max<size_t>(static_cast<size_t>(1), spare_blocks / 2);
    return spare_blocks > trim ? spare_blocks - trim : 0;
}

// The replay loop's per-pass refuse-or-retry decision for an explicit
// --n-ctx allocation that failed (backend_ov.cpp, the "could not be
// honoured" throw).
//
// THE DEFECT, pinned: production's rule here was unconditional -- the
// branch simply threw every time this pass wasn't accepted, never reading
// `over` or any live/spare split at all (see the round-1 review, finding
// F2: "return measured_overshoot" was never production's rule -- there was
// no `measured_overshoot` parameter for production to read in the first
// place; "return true" unconditionally is the honest characterization).
//
// Round-3 review, finding 1 -- a correction to round 2's own record: rebuilt
// device-free against `return true;` (round 2's file, 39 tests), TWO tests
// go red, not one -- `39 cases run, 2 failed`, `explicit_overshoot_2026_09_
// 02_defect_must_not_refuse_when_spare_can_absorb_it` AND `explicit_
// overshoot_unmeasured_failure_with_spare_must_not_refuse` (added in round
// 2; it asserts the same `!refuse(spare > 0)` shape, so it fails the same
// way against an unconditional rule). Round 2's fit.h claimed "exactly
// ONE... not two" -- that was true of round 1's file, at a time when the
// second test did not exist yet, and the sentence was carried forward
// instead of re-checked after the F4 test was added.
//
// Checked again against round 3's OWN file (43 tests, after this round's
// additions below and in test_fit.cpp): THREE go red, not two --
// `43 cases run, 3 failed`. The third is
// `explicit_retry_decision_refuse_agrees_with_explicit_overshoot_must_
// refuse` (new this round), and it fails for a different, expected reason:
// patching only `explicit_overshoot_must_refuse` to `return true;` breaks
// the AGREEMENT that test checks against `explicit_retry_decision`, which
// computes its own `refuse` directly from `spare_blocks` and does not call
// `explicit_overshoot_must_refuse` at all -- so the two functions
// legitimately disagree once one of them is wrong, which is exactly what
// that test exists to catch. It is not a repeat of round 2's mistake: the
// two findings-1-named tests are still exactly those two. The verbatim red
// output for all three is in the round-3 commit/report, not narrated here.
//
// Round 1 fixed the case where a MEASURED overshoot (a residency-vs-ceiling
// reading) still had spare pages to give up: refuse only when
// `!measured_overshoot || spare_blocks == 0`. Round 2 (review finding F4)
// narrows this further: an UNMEASURED failure (a raw allocation exception)
// is no longer an automatic refusal either, as long as spare pages exist
// to retreat -- synthesize_spare_retreat (above) gives that retry
// something to trim, mirroring the auto-fit branch's own synthesized
// retreat on its `over == 0` guard. Refusal now fires on exactly one
// condition: `spare_blocks == 0` -- there is nothing left to trim, measured
// or not, and the failure must be the request itself. The "never lowered
// automatically" promise (DESIGN §7.0.2t) still applies to n_ctx: trimming
// spare -- measured or synthesized -- never touches it.
inline bool explicit_overshoot_must_refuse(size_t spare_blocks) {
    return spare_blocks == 0;
}

// Round-3 review, finding 3 + finding 4: the explicit branch's whole
// per-pass decision, extracted so backend_ov.cpp's loop and test_fit.cpp
// exercise the exact same logic instead of a test merely restating
// pool_sizing's own arithmetic (finding 4's complaint about the round-2
// invariant test, which could not fail for any implementation).
//
// Deliberately takes no `n_ctx` or `live_blocks` parameter at all --
// `n_ctx_unchanged` is not computed from anything, it is pinned `true` by
// construction, which is the point: nothing in this function's signature
// COULD move the request, so a call site that swapped it for shrink_n_ctx
// would have to reach past this return value entirely to do it.
//
// `pass` is the index of the pass that JUST FAILED (0-indexed); `last_pass`
// is the index of the final pass in the loop (3, for four passes 0..3).
//
// `refuse` fires on either of two conditions: `spare_blocks == 0` (nothing
// left to trim, same as explicit_overshoot_must_refuse above -- this
// field and that free function agree on every input by construction), OR
// `pass >= last_pass` (finding 3: the pass that just failed WAS the last
// one -- there is no pass 5 to retry into, so continuing would silently
// waste the cap this function would otherwise compute and fall through to
// an un-itemized "replay exhausted" throw instead of the itemized one this
// decision drives; a `pass >= last_pass` failure IS the reserve-exhausted
// case, not a separate one).
//
// When not refusing, `next_spare_cap` is the wanted-spare bound for pass
// `pass + 1`: forced to 0 when THAT upcoming pass is itself the last one
// (`pass + 1 >= last_pass`) -- the one direct test of whether --n-ctx
// itself is honourable, tried before the loop can run out on a pool that
// was never reduced to just the request -- otherwise computed the normal
// way (explicit_retry_spare_cap for a measured overshoot,
// synthesize_spare_retreat for an unmeasured one).
struct ExplicitRetryDecision {
    bool   refuse           = true;   // spare_blocks == 0, or `pass` was already the last pass
    size_t next_spare_cap   = 0;      // wanted-spare bound for pass `pass + 1`
    bool   n_ctx_unchanged  = true;   // pinned true -- not computed from any n_ctx input
};

inline ExplicitRetryDecision explicit_retry_decision(bool measured_overshoot, uint64_t over,
                                                      size_t spare_blocks, uint64_t kv_block_bytes,
                                                      int pass, int last_pass) {
    ExplicitRetryDecision d;
    if (spare_blocks == 0 || pass >= last_pass) return d;  // refuse; next_spare_cap stays 0
    d.refuse = false;
    if (pass + 1 >= last_pass) {
        d.next_spare_cap = 0;  // finding 3: force the LAST pass live-only
    } else {
        d.next_spare_cap = measured_overshoot
            ? explicit_retry_spare_cap(spare_blocks, over, kv_block_bytes, pass)
            : synthesize_spare_retreat(spare_blocks);
    }
    return d;
}

// Round-3 review, defect 1: the auto-fit branch's own per-pass retry
// decision, mirroring explicit_retry_decision's role for the explicit
// branch. THE DEFECT, pinned: the pre-fix auto-fit correction shrank
// ONLY `paged_n_ctx_` (via shrink_n_ctx) and never capped `wanted_spare`
// -- so whenever spare_blocks > 0, pool_sizing's own spare_room
// (`affordable - live_blocks`, pool_sizing above) grew by exactly what
// live_blocks just lost, `blocks` (live + spare, the number that
// actually governs what the driver allocates) reassembled to the SAME
// total every pass, and a sub-page overshoot (a driver allocation
// granule wider than one KV page) never converged -- measured on both
// the reserve case (coder/16 GiB, --prefix-cache-reserve 25) and the
// no-reserve case (35B/24 GB card, plain auto-fit, spare from the train
// maximum capping live below what the budget affords). Fixed the same
// way the explicit branch already fixes it for spare alone: the CALLER
// (backend_ov.cpp) carries a persistent spare cap across passes and
// applies it to `wanted_spare` before every `pool_sizing` call --
// `next_spare_cap` here is that cap, pinned at (or below) the pool's
// OWN currently realized spare_blocks every pass, which by construction
// stops pool_sizing's room-growth from ever re-inflating it even when
// this pass's cut went entirely to `next_n_ctx` instead.
//
// The live/spare split: `protect_spare` is `reserve_applied` at the call
// site -- true only when an explicit --prefix-cache-reserve was
// honoured for this load. When true, LIVE is cut first (via
// `next_n_ctx`) and the spare cap is left exactly where it was (pinned,
// not reduced) for as long as live has room to give -- the operator
// explicitly asked to keep that headroom. Spare only gives once live has
// nothing left before the floor (`live_room < this pass's trim budget`);
// the shortfall that leaves, if any, is what the existing post-hoc
// `prefix_cache_reserve_shortfall` warning reports, not this function.
// When false (no explicit reserve -- either no --prefix-cache-reserve at
// all, or spare arose only because the train maximum capped live below
// what the budget affords), SPARE is cut first instead: nobody asked to
// keep it, cutting it is invisible to the operator, and cutting live
// changes the served context length, so it is the last resort.
//
// `n_ctx_floor`, `kv_block_tokens` and `lanes` mirror shrink_n_ctx's own
// parameters; `spare_blocks` is a POOL-WIDE page count (pool_sizing's own
// units, i.e. summed over lanes). Live's own room to give is derived from
// `n_ctx` and `n_ctx_floor` alone (see `live_room` below) rather than
// taking the caller's `live_blocks` as a separate parameter -- the two
// are equivalent to first order (live_blocks is itself a function of
// n_ctx via the caller's per_lane_blocks formula, and the fixed per-lane
// guard overhead that formula adds cancels out of a DIFFERENCE between
// the current and floor page counts), and deriving it here means this
// function does not need to know that formula's shape at all. A
// pool-wide page cut is translated back to a PER-LANE token reduction
// for `next_n_ctx` (rounding the per-lane share of the cut UP, so the
// per-lane page count strictly decreases even when the cut does not
// divide `lanes` evenly, the same guarantee shrink_n_ctx gives).
struct AutoFitTrim {
    bool      refuse         = false;  // floor reached with nothing left to cut
    long long next_n_ctx     = 0;      // n_ctx (tokens, per lane) for the next pass
    size_t    next_spare_cap = 0;      // spare_blocks cap (pool-wide pages) for the next pass
};

inline AutoFitTrim auto_fit_trim(long long n_ctx, size_t spare_blocks, uint64_t over,
                                 uint64_t kv_block_bytes, int lanes, int kv_block_tokens,
                                 int n_ctx_floor, int pass,
                                 bool protect_spare) {
    AutoFitTrim d;
    const int  lanes_c = std::max(1, lanes);
    const int  block    = std::max(1, kv_block_tokens);
    const long long floor_ll = static_cast<long long>(n_ctx_floor);

    const size_t budget = pool_pages_to_trim(over, kv_block_bytes, pass);

    // How many pool-wide pages n_ctx could still give up before the
    // floor -- a lower-bound estimate (integer division, ignoring the
    // caller's own per-lane guard overhead, which is a fixed additive
    // term on both the current and floor page counts and cancels out of
    // a difference like this to first order); conservative in the safe
    // direction, since it can only make this function ask for LESS room
    // than technically exists, never more.
    const long long room_tokens = n_ctx > floor_ll ? n_ctx - floor_ll : 0;
    const size_t    live_room =
        (static_cast<size_t>(room_tokens) / static_cast<size_t>(block)) *
        static_cast<size_t>(lanes_c);

    size_t live_cut = 0, spare_cut = 0;
    if (protect_spare) {
        live_cut  = std::min(budget, live_room);
        spare_cut = std::min(budget - live_cut, spare_blocks);
    } else {
        spare_cut = std::min(budget, spare_blocks);
        live_cut  = std::min(budget - spare_cut, live_room);
    }

    if (live_cut == 0 && spare_cut == 0) {
        d.refuse = true;
        return d;
    }

    long long next_n_ctx = n_ctx;
    if (live_cut > 0) {
        // live_cut is pool-wide (summed over lanes); the per-lane share
        // rounds UP so per_lane_blocks strictly decreases even when
        // live_cut does not divide lanes_c evenly.
        const size_t per_lane_cut =
            (live_cut + static_cast<size_t>(lanes_c) - 1) / static_cast<size_t>(lanes_c);
        const long long tokens_cut =
            static_cast<long long>(per_lane_cut) * static_cast<long long>(block);
        next_n_ctx = n_ctx - tokens_cut;
        next_n_ctx = (next_n_ctx / block) * block;              // floor to a page multiple
        if (next_n_ctx >= n_ctx) next_n_ctx = n_ctx - block;     // guarantee a strict decrease
        if (next_n_ctx < floor_ll) next_n_ctx = floor_ll;        // never below the floor
    }
    d.next_n_ctx     = next_n_ctx;
    d.next_spare_cap = spare_blocks > spare_cut ? spare_blocks - spare_cut : 0;
    return d;
}

// --prefix-cache-reserve PCT (M9): an option under auto-fit that holds back
// PCT percent of the pool's own budget-affordable pages as spare for cached
// prefixes, instead of auto-fit adopting the whole budget as live pages
// (PCT 0, today's behaviour) and leaving the prefix cache zero reserve --
// the case the "accepted pool has 0 spare pages" warning above exists to
// name. Verify-only for an explicit --n-ctx: config.cpp refuses that
// combination outright (an explicit depth already defines the reserve as
// whatever budget remains), so this function is only ever called from the
// auto-fit path, before the pool is sized.
//
// `affordable_pages` is fit_context's own max_ctx, in pages: fit_context
// already floors max_ctx to a kv_block_tokens multiple, so `max_ctx /
// kv_block_tokens` is exact, and affordable_pages IS "what the budget
// affords" -- PCT 0's live_pages == affordable_pages reproduces fit_context's
// unreserved max_ctx exactly, term for term.
//
// live_pages = floor(affordable_pages * (100 - PCT) / 100) -- integer
// arithmetic, so the result is exact and reproducible pass to pass, which is
// what the caller's own "the correction passes keep the reserve" depends on.
//
// Round-4 correction: this comment used to describe the correction as
// calling shrink_n_ctx "on the live portion only, never touching
// spare_blocks or wanted_spare" -- true of the ORIGINAL (defective)
// implementation, not of auto_fit_trim (fit.h, above), which replaced it
// in round 3 specifically because that description was the bug: with
// spare_blocks > 0, a live-only correction left pool_sizing's own
// spare_room (`affordable - live_blocks`) growing right back to absorb
// the cut, and the pool total never moved. What actually protects the
// reserve now: backend_ov.cpp passes `protect_spare = reserve_applied`
// into auto_fit_trim, which cuts LIVE first (`next_n_ctx`) and pins the
// spare cap exactly where it was for as long as `live_room >= this
// pass's trim budget` (`budget = pool_pages_to_trim(...)`, at most
// max(the analytic trim, 256 pages -- the last pass's backoff floor) --
// so spare stays untouched while `n_ctx` is more than roughly
// `budget * kv_block_tokens / lanes` tokens above the floor, and only
// gives once it is closer than that, or the analytic trim from a
// genuinely large `over` exceeds live's remaining room outright).
//
// One consequence worth stating plainly: because live shrinks pass to
// pass while a protected spare cap stays FLAT, the REALIZED reserve
// fraction (spare_blocks / pool total) drifts UP as the correction
// proceeds -- this function's own PCT is honoured at adoption time, not
// held constant through every retry; a load that needed several
// correction passes ends up with a somewhat MORE generous spare share
// than PCT asked for, never less, until live hits the floor and spare
// itself has to start giving (reported by the post-hoc
// `prefix_cache_reserve_shortfall` warning when it does).
//
// Flooring live DOWN (rather than rounding) is what guarantees
// spare_pages = affordable_pages - live_pages is never less than PCT
// percent of affordable_pages: live_pages <= affordable_pages * (100-PCT) /
// 100 as a real number, so spare_pages >= affordable_pages * PCT / 100
// follows directly, for any PCT in (0, 100).
//
// `admissible` mirrors FitResult::admissible's own rule (max_ctx > 0 and >=
// n_ctx_floor) applied to the RESERVED adopted_n_ctx rather than the
// unreserved one -- DESIGN's "never silently lowered" stance extends here:
// a PCT that would push the adopted depth below the floor refuses (the
// caller's admissible check), it does not silently float the reserve back
// up to fit inside the floor.
struct PrefixCacheReserve {
    long long live_pages    = 0;  // affordable_pages * (100-PCT) / 100, floored
    long long spare_pages   = 0;  // affordable_pages - live_pages (>= PCT% by construction)
    long long adopted_n_ctx = 0;  // live_pages * kv_block_tokens -- already page-aligned
    bool      admissible    = false;  // adopted_n_ctx > 0 and >= n_ctx_floor
};

inline PrefixCacheReserve prefix_cache_reserve(long long affordable_pages, int reserve_pct,
                                               int kv_block_tokens, int n_ctx_floor) {
    PrefixCacheReserve r;
    if (affordable_pages < 0) affordable_pages = 0;
    const int pct   = std::clamp(reserve_pct, 0, 90);
    const int block = std::max(1, kv_block_tokens);

    r.live_pages  = affordable_pages * (100 - pct) / 100;
    r.spare_pages = affordable_pages - r.live_pages;
    r.adopted_n_ctx = r.live_pages * static_cast<long long>(block);
    r.admissible    = r.adopted_n_ctx > 0 && r.adopted_n_ctx >= static_cast<long long>(n_ctx_floor);
    return r;
}

// Round-3 review, defect 2 (units): `PrefixCacheReserve::spare_pages` is
// PER-LANE, pure-KV pages -- it comes from fit_context's own max_ctx,
// which fit_context's comment states is already "per lane". The pool's
// own `spare_blocks` (pool_sizing, Phase E in backend_ov.cpp) is
// POOL-WIDE (summed over lanes) and, on the LIVE side, ALSO carries the
// per-lane guard overhead `per_lane_blocks` bakes in -- 2 lookahead/
// rollback pages plus ceil(drafts_max/kv_block_tokens) drafter pages,
// per lane -- overhead the reserve arithmetic never priced in, so it
// eats into what would otherwise be spare. Comparing `spare_pages`
// directly against `spare_blocks` (the round-2 H2 fix's own call site)
// was wrong on both axes: units (per-lane vs pool-wide -- under-detects
// a real shortfall by a factor of `lanes` on more than one lane) and
// scale (missing the guard subtraction -- a false alarm even on ONE
// lane: measured 18,313 asked vs 18,311 accepted with nothing actually
// short).
//
// `reserve_ask_pool_pages` converts `spare_pages` into the SAME units
// `spare_blocks` is in -- pool-wide pages, net of the per-lane guard
// overhead the adopted live pages already spend -- so the two are
// finally comparable. Clamped at 0: the guard overhead can exceed a very
// small reserve, and a negative ask is not meaningful.
inline long long reserve_ask_pool_pages(long long reserve_spare_pages, int lanes, int drafts_max,
                                        int kv_block_tokens) {
    const long long lanes_c = std::max(1, lanes);
    const long long block    = std::max(1, kv_block_tokens);
    const long long drafts   = std::max(0, drafts_max);
    const long long guard_per_lane = 2 + (drafts + block - 1) / block;

    const long long ask = reserve_spare_pages * lanes_c - guard_per_lane * lanes_c;
    return ask > 0 ? ask : 0;
}

// H2 (round-2 release review, HIGH; units corrected round-3, defect 2 --
// see reserve_ask_pool_pages above) `prefix_cache_reserve`'s own
// `spare_pages` is provably > 0 for any reserve_pct in [1, 90] with a
// nonzero affordable_pages (see the comment above -- it is a strict
// inequality on the BUDGET split, not a promise about what actually got
// allocated), so a caller that gated a warning on "spare_pages > 0" was
// gating on something that can never be false whenever the reserve was
// requested at all -- the warning went silent unconditionally, including
// the case it exists to catch: the allocation-time replay loop (Phase E in
// backend_ov.cpp) coming back with LESS spare than the reserve asked for,
// under real memory pressure or a --kv-pool-pages test cap.
//
// The real question is comparative: does the pool actually accepted at
// allocation time (`accepted_spare_blocks`, POOL-WIDE pages) fall short of
// what the reserve computed it should hold back (`reserve_spare_pages` --
// the caller must pass `reserve_ask_pool_pages`'s output here, NOT
// `PrefixCacheReserve::spare_pages` directly; the two are in different
// units, see above)? True exactly when there is a shortfall worth naming;
// false both when nothing was reserved (`reserve_spare_pages <= 0` -- the
// no-reserve case the caller's own pre-existing zero-spare warning already
// covers) and when the accepted pool met or exceeded the ask.
inline bool prefix_cache_reserve_shortfall(size_t accepted_spare_blocks,
                                           long long reserve_spare_pages) {
    return reserve_spare_pages > 0 &&
           static_cast<long long>(accepted_spare_blocks) < reserve_spare_pages;
}

}  // namespace lgc
