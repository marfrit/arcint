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
inline size_t explicit_retry_spare_cap(size_t spare_blocks, uint64_t over,
                                       uint64_t kv_block_bytes) {
    const size_t trim = kv_block_bytes > 0
        ? std::max<size_t>(static_cast<size_t>(1),
                           static_cast<size_t>((over + kv_block_bytes - 1) / kv_block_bytes))
        : static_cast<size_t>(1);
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
            ? explicit_retry_spare_cap(spare_blocks, over, kv_block_bytes)
            : synthesize_spare_retreat(spare_blocks);
    }
    return d;
}

}  // namespace lgc
