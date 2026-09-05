#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

// M7 — auto-fit and the honest reservation (design: "M7 — Auto-fit and the
// honest reservation", 2026-09-01, §1/§2/§4). Pure arithmetic, no OpenVINO
// types, so it builds and runs on any host: the load path in backend_ov.cpp
// measures every term on the card and hands them here; this file only adds
// them up and floors the result to a page. `packed_values_scratch_geometry`
// below is the one exception to "pure arithmetic": it reads an artifact's
// config.json (nlohmann::json, already a build dependency of this
// repository's core, not an OpenVINO type) so the query-heads/head_dim
// lookup it replaces in backend_ov.cpp is testable without a card too.
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

// Patch 0018 / MOE_CPU_TIER_STATIC_PARTITION (DESIGN §7.0.2ae "F2"): under
// the plugin's static residency partition, each expert's host-or-device
// placement is a pure function of expert id, layer and pool configuration,
// fixed for the life of the process. Under the LRU-mode plateau probe
// (backend_ov.cpp Phase B), this same arithmetic -- `expert_slot_bytes`,
// unchanged -- only ever prices a CEILING for the host-side ledger, because
// device residency there is history-dependent and only a probe can measure
// what is actually resident. Under the static partition that history
// dependence is gone, so the identical count is the EXACT device-resident
// figure, not an estimate. Named separately from `expert_slot_bytes` only so
// the load-time call site and this function's own test read as "the static-
// mode device figure" rather than "the LRU-mode host ceiling that happens to
// share arithmetic" -- the formula is deliberately the same one. The
// signature carries no probe/eviction/history state (there is nothing to
// carry): num_expert, ratio_pct, per_expert_bytes and moe_layers are the
// whole input, so this is a pure function of configuration alone, same as
// `expert_slot_bytes` itself.
inline uint64_t expert_slot_bytes_static(int num_expert, int ratio_pct, uint64_t per_expert_bytes,
                                         int moe_layers) {
    return expert_slot_bytes(num_expert, ratio_pct, per_expert_bytes, moe_layers);
}

// The load-time logits-slice verification (backend_ov.cpp, the activation
// floor probe): the slice keeps the LAST `keep_rows` rows of the token axis,
// and a forward of `tokens` tokens cannot return more rows than it was given
// tokens, so the row count the graph must hand back is min(keep_rows,
// tokens) -- not keep_rows. Found at the 0.3.0 release gate (DESIGN
// §7.0.2ai): with MTP's 2-row slice and --prefill-chunk 1 the probe runs one
// token, the check expected two rows, and a configuration that serves
// correctly (every serve-time read indexes by the tensor's actual row
// count) was refused at load with a message blaming the token axis. Callers
// gate on keep_rows > 0 first, as backend_ov.cpp does; an unsliced graph
// returns `tokens` rows and is not this function's claim.
inline size_t logits_slice_rows_expected(size_t keep_rows, size_t tokens) {
    return std::min(keep_rows, tokens);
}

// M11 §1.3 (DESIGN §7.0.2ag, "the fix design: MTP's verify cost and zero
// acceptance at depth"): the MTP layer's own state -- `mtp_layer`, driven by
// mtp_prime_paged in backend_ov.cpp -- is a STATEFUL paged KV pair,
// KV_HEADS 4 x HEAD_DIM 256 (tools/export_mtp.py:31), K+V, f32, and it is
// primed over the WHOLE prompt, not a bounded window: 4 * 256 * 2 * 4 bytes
// = 8 KiB per token, per lane. Pre-fix, `drafter_bytes` in backend_ov.cpp
// (the resident delta taken right after the drafters COMPILE) charged only
// the drafter graphs' WEIGHTS -- priming had not run yet at that point, so
// this per-token growth was charged nowhere, and a served MTP arm could
// overcommit the card (measured: "resident 22.47 GiB against a 22.46 GiB
// ceiling" at the admitted n_ctx 155,488, DESIGN's own §1.3 finding 3).
//
// DFlash's own drafter state is deliberately NOT priced by this function:
// it is windowed to kDflashWindow (2,048 rows, backend_ov.cpp / see
// tests/test_dflash_window.cpp) and so stays a small, DEPTH-INDEPENDENT
// ~84 MiB regardless of n_ctx -- unlike the MTP layer, it never needs a
// per-token term at all, which is exactly why this fix touches only one of
// the two drafters.
//
// `mtp_state_bytes_per_token` is folded into the FIT'S OWN per-token rate
// at the backend_ov.cpp call site (added to a local copy of
// FitTerms::kv_bytes_token, never to the real `kv_bytes_token_` member
// Phase E's allocation math still needs at the true KV-pool byte size) --
// the same "per lane, per token" shape kv_bytes_token already has, so
// fit_context's existing `budget / kv_bytes_token / lanes` division prices
// it exactly where the design's §3 Fix A asks: subtracted before max_ctx is
// derived, not after.
constexpr uint64_t kMtpStateBytesPerToken = 8192;  // 4 heads * 256 head_dim * 2 (K+V) * 4B (f32)

// Pure arithmetic mirror of the term above, for tests and for the
// reservation log line (backend_ov.cpp reports this next to "drafters"):
// bytes for `n_ctx` tokens of ONE lane's MTP state. `n_ctx <= 0` is "nothing
// primed yet" and returns 0, matching every other term in this file that
// treats a non-positive extent as absent rather than as an error.
inline uint64_t mtp_state_bytes(long long n_ctx) {
    if (n_ctx <= 0) return 0;
    return static_cast<uint64_t>(n_ctx) * kMtpStateBytesPerToken;
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

// M9 engine-side belt for packed 4-bit paged VALUES -- EMPIRICAL, not a
// derived mechanism (round-2 review, finding 3: the first version of this
// comment asserted a single scratch-byte threshold, and a later measurement
// contradicts that reading directly -- chunk 512 at a 35,227-token prompt
// PASSES, at ~1,104 MiB of the very buffer named below, while chunk 128 at a
// 119,074-token prompt FAULTS at ~932 MiB: the larger buffer is the one that
// worked, so "total scratch bytes <= threshold" is not what actually decides
// pass/fail, and this comment no longer claims it is).
//
// What IS measured (16 GiB-class card, coder artifact, `--paged-kv u8:i4`,
// one process per cell, pool n_ctx 131,072 unless noted): faults observed
// only at past >= ~72k for chunk 256 and past >= ~119k for chunk 128; never
// at chunk 64 in any cell tried, up to a 119,074-token prompt (605 s,
// 197 t/s); the same chunk 128 with plain `--paged-kv u8` (no 4-bit value
// side) prefills a 119,074-token prompt without incident, so the fault is
// specific to 4-bit-packed values, not chunk size alone. Read-only recon of
// the plugin (record, not something this repository derives or verifies):
// under i4-packed values, prefill routes through an "opt" paged-attention
// path with a MIXED-stage (past > 0) intermediate buffer sized
//   chunk x query_heads x head_size x 4 bytes x ceil(past / 256)
// -- query heads (`heads_num` in the plugin's own buffer-size call), not KV
// heads; on this GQA artifact (16 query heads, 2 KV heads) the two differ by
// 8x, which is exactly what an earlier version of the call site got wrong
// (fed KV heads, undercounted the buffer 8x, and the cap never fired). This
// formula is used here as a PROXY the empirical bound is expressed in --
// convenient because it is monotonic in chunk and in depth, matching the
// "smaller chunk, or shallower pool, is safer" shape the measurements show
// -- not as a claim that it is the plugin's exact allocation arithmetic
// (the contradiction above rules that out as a complete account).
//
// kPrefillScratchBudgetBytesPackedValues is chosen, not derived: 512 MiB is
// the figure that makes this proxy formula land on chunk 64 -- the largest
// chunk measured to pass at every depth tried, including the deepest
// (119,074 past) -- when evaluated at pool depth 131,072 (the served n_ctx
// during measurement): 64 x 16 x 256 x 4 x ceil(131072/256) = 64 x 16 KiB x
// 512 = exactly 512 MiB. The margin at the passing 71,689-token cell (562
// MiB at chunk 128, per the proxy) is incidental to this choice, not a
// second data point it was fit to.
constexpr uint64_t kPrefillScratchBudgetBytesPackedValues = 512ull << 20;  // 512 MiB

// Round-10 review (Opus, a PATCHED-plugin card run): the largest chunk
// this entire body of work has ever explicitly validated on a card for
// the packed-4-bit-values prefill path, UNPATCHED plugin (128 -- the
// belt's own measured cells above top out there; the card's own bounded-
// plugin run at chunk 2048 crashed, cause under investigation, and 2048
// has never passed on ANY plugin, patched or not). Fitting inside the
// 512 MiB scratch proxy is NECESSARY but not SUFFICIENT for a chunk to be
// safe -- the proxy prices one buffer's own size, not whatever else at a
// huge chunk could go wrong (dispatch overhead, a plugin-side limit
// nothing here can see) -- so `prefill_chunk_cap_for_packed_values_ex`
// additionally never serves more than this, regardless of what the
// budget alone would allow, unless the operator's own `requested_chunk`
// was already smaller (never widened, same as every other guard in this
// function).
constexpr int kMaxMeasuredPackedValuesChunk = 128;

// The two numbers the proxy formula above needs from the artifact's own
// geometry: QUERY heads (`num_attention_heads` -- the plugin's MIXED-stage
// buffer is sized by query heads, not KV heads; see the round-2 review
// finding this fixes, quoted above) and `head_dim`. text_config-aware, the
// same convention backend_ov.cpp's own inline JSON reads use elsewhere
// (moe_intermediate_size, geometry fields) -- when `config` carries a
// `text_config` object, that is read instead of the top level.
//
// Deliberately narrow: no `num_key_value_heads` fallback for heads (that IS
// the bug this function exists to fix -- KV heads undercount the query-head
// buffer on any GQA artifact), and no `hidden_size / num_attention_heads`
// fallback for `head_size` (not a safe identity for every architecture in
// this family; the config's own explicit `head_dim`, when absent, means
// "this function cannot price the cap," not "guess"). Either field missing
// returns nullopt -- the caller's job is to warn and skip the cap, not to
// invent geometry.
struct PackedValuesScratchGeometry {
    int heads     = 0;  // query heads (num_attention_heads)
    int head_size = 0;  // head_dim
};

inline std::optional<PackedValuesScratchGeometry> packed_values_scratch_geometry(
    const nlohmann::json& config) {
    const nlohmann::json& tc =
        config.contains("text_config") && config["text_config"].is_object()
            ? config["text_config"]
            : config;
    PackedValuesScratchGeometry g;
    if (tc.contains("num_attention_heads") && tc["num_attention_heads"].is_number_integer()) {
        g.heads = tc["num_attention_heads"].get<int>();
    }
    if (tc.contains("head_dim") && tc["head_dim"].is_number_integer()) {
        g.head_size = tc["head_dim"].get<int>();
    }
    if (g.heads <= 0 || g.head_size <= 0) return std::nullopt;
    return g;
}

// Returns the chunk the HALVING LADDER lands on: starting from
// `requested_chunk`, repeatedly halve and round UP to the next multiple of
// `block_size` (the KV page granularity config.cpp:738 already requires
// --prefill-chunk to divide) until a candidate satisfies
//   chunk * heads * head_size * 4 * ceil(max_ctx_tokens / 256) <= scratch_budget_bytes
// (the proxy above), or `requested_chunk` unchanged when it already
// satisfies that bound. This is the FIRST ladder step that fits, never
// below `block_size` -- NOT the largest block-size multiple that fits, by
// design (round-2 review residual 2): the ladder only ever visits values
// derived from `requested_chunk` by repeated halving (128, 64, 32, ... --
// the chunks this belt's own measurements were actually taken at), so an
// intermediate multiple the halving step skips over can satisfy the bound
// and still not be returned. Concretely, at the coder's own shape (16
// heads, 256 head_size, 512 MiB budget) and depth 71,689: 96 is a multiple
// of block_size 32, and 96 x 16 KiB x 281 partitions = 442 MiB fits under
// budget -- but halving from 128 goes straight to 64 and stops there
// (64 fits too), so 96 is never tried and never returned (see
// prefill_chunk_cap_packed_values_71689_skips_a_larger_fitting_multiple in
// tests/test_fit.cpp). Deliberate: halving keeps the served chunk on the
// values that were measured to pass or fault (128, 64, 32...), never on an
// arbitrary intermediate multiple nothing on the card ever actually ran at.
// Never raises the chunk: halving only ever starts from the request and
// stops the moment a candidate fits. `max_ctx_tokens` is the SERVED pool
// depth this load actually committed to -- backend_ov.cpp's final
// `paged_n_ctx_`, after --n-ctx clamping, --prefix-cache-reserve, and
// Phase E's replay/trim have all run -- not the pre-trim auto-fit `max_ctx`
// and not an individual request's prompt length (round-2 review, finding 4:
// an explicit --n-ctx well under a deep auto-fit ceiling must not be capped
// for a depth it will never reach). Any missing/non-positive input (heads,
// head_size, max_ctx_tokens, requested_chunk, scratch_budget_bytes,
// block_size) is "nothing to reason about" and passes `requested_chunk`
// through unchanged, same as the caller not calling this at all -- this
// function never widens a refusal into a silent no-op cap, it only ever
// narrows.
//
// Halving rounds UP to the next `block_size` multiple, not down (round-2
// review, finding 5): requested 96 with block_size 32 must land on 64, not
// 48 -- 96/2 = 48 floors to 32 (one block below the true half) but ceils to
// 64 (one block at or above it), and 64 is what a card that faulted at 96
// but never at 64 actually needs tried next, not a value smaller again.
//
// No guard against the rounded-up `next` reaching or passing `chunk` is
// needed (round-2 review residual 2: an earlier version carried one,
// `if (next >= chunk) next = chunk - block_size;`, and it was DEAD --
// removed rather than kept unreachable). Proof: the loop body only runs
// when `chunk > block_size`, i.e. `chunk >= block_size + 1`. Let
// `raw = chunk / 2` (integer division, so `raw <= (chunk - 1) / 2` when
// `chunk` is odd and `raw = chunk / 2` when even; either way
// `2 * raw <= chunk - (chunk mod 2) <= chunk`) and `next = ceil(raw /
// block_size) * block_size <= raw + block_size - 1`. Substituting,
// `next <= chunk / 2 + block_size - 1`, and this is `< chunk` exactly when
// `block_size - 1 < chunk / 2`, i.e. `chunk > 2 * block_size - 2` --
// which `chunk >= block_size + 1` satisfies for every `block_size >= 1`
// (the two bounds cross only below block_size 1, which `block_size <= 0`
// already refuses above). Checked exhaustively for block_size 1..2000 at
// the tightest case `chunk = block_size + 1` (this file's own test suite
// does not re-derive real numbers from this proof; it is recorded here so
// a future change to the rounding has something to re-check against).
//
// A byte-exact ladder (DESIGN's measurement-discipline gate) that crosses
// this cap between two runs served a DIFFERENT chunk in each -- chunk
// boundaries change where the graph slices logits/samples, so equivalence
// across a change here must record the EFFECTIVE chunk
// (status_.reservation.prefill_chunk, post-cap) alongside the run, not
// assume the requested one. The activation reservation itself is NOT
// re-measured at the post-cap chunk (backend_ov.cpp's own comment at the
// call site): it stays probed at the pre-cap chunk, which over-charges
// activations when the belt fires -- conservative, never unsafe.
// Round-5 (0015 engine side): generalized -- `partitions` and `element_bytes`
// are inputs now, rather than derived unconditionally from `max_ctx_tokens`
// at a hardcoded 4 bytes. `prefill_chunk_cap_for_packed_values`, below,
// becomes a thin wrapper (`partitions = ceil(max_ctx_tokens / 256)`,
// `element_bytes = 4`) reproducing today's unbounded, f32-buffer belt
// exactly, so every existing caller and test needs no change. The patched
// plugin's own `PAGED_ATTENTION_MAX_PARTITIONS` bound (design note
// o-0015-design.md §C) caps `partitions` below `ceil(max_ctx_tokens/256)`
// and its f16-corrected host sizing halves `element_bytes` -- backend_ov.cpp
// computes both from `packed_values_bounded_partitions` and the accepted-key
// probe before calling this.
// Forward declarations: the two functions this belt's own `fits()` check
// needs (round-10 review, finding 1) are defined further down this file
// (they need the belt's own halving-ladder logic to already exist for
// their OWN doc comments to reference it) -- pure arithmetic, no
// dependency cycle, just declaration order.
inline uint64_t packed_values_prefill_scratch_bytes_ex(int chunk, long long partitions, int heads,
                                                        int head_size, int element_bytes);
inline uint64_t packed_values_partials_exp_max_bytes(int chunk, long long partitions, int heads);

// Round-11 review (Opus), finding 3: whether `chunk` fits the raw-proxy
// budget on its own, exposed as its own query -- backend_ov.cpp's
// load-time log uses this (against the FLOOR, `block_size`) to tell
// "the halving ladder shrank to something that actually fits" (budget)
// apart from "the ladder ran all the way down to the floor and STILL
// does not fit" (bound -- the floor is returned anyway, not a refusal;
// see prefill_chunk_cap_for_packed_values_ex's own comment) without
// duplicating the raw-proxy arithmetic a second time at the call site.
inline bool packed_values_scratch_fits_budget(int chunk, long long partitions, int heads,
                                              int head_size, int element_bytes,
                                              uint64_t scratch_budget_bytes) {
    if (chunk <= 0 || partitions <= 0 || heads <= 0 || head_size <= 0 || element_bytes <= 0) {
        return true;  // nothing to price -- matches the callers' own no-op guard direction
    }
    const uint64_t raw = static_cast<uint64_t>(chunk) * static_cast<uint64_t>(heads) *
                         static_cast<uint64_t>(head_size) * static_cast<uint64_t>(element_bytes) *
                         static_cast<uint64_t>(partitions);
    return raw <= scratch_budget_bytes;
}

// Round-11 review (Opus), finding 3: the budget-driven halving ladder,
// WITHOUT the `kMaxMeasuredPackedValuesChunk` ceiling applied on top --
// extracted from `prefill_chunk_cap_for_packed_values_ex` below (which
// now calls this, then applies the ceiling; behaviour unchanged) so
// backend_ov.cpp's load-time log can compare the two and name which of
// the load's three possible limits actually shrank the requested chunk:
// this function's own output differing from `_ex`'s means the measured
// ceiling fired; this function returning `block_size` while
// `packed_values_scratch_fits_budget(block_size, ...)` is false means
// the ladder bottomed out without ever fitting (bound/floor); anything
// else is an ordinary budget-driven shrink (or no shrink at all).
inline int prefill_chunk_cap_for_packed_values_budget_only_ex(
    int requested_chunk, long long partitions, int heads, int head_size, int element_bytes,
    uint64_t scratch_budget_bytes, int block_size) {
    if (requested_chunk <= 0 || partitions <= 0 || heads <= 0 || head_size <= 0 ||
        element_bytes <= 0 || scratch_budget_bytes == 0 || block_size <= 0) {
        return requested_chunk;
    }
    // Round-10 review, finding 1: this predicate used to check the RAW,
    // un-margined single-buffer size (`chunk * heads * head_size *
    // element_bytes * partitions`) against `scratch_budget_bytes`, which
    // is not what the reservation actually charges -- the real term
    // (`packed_values_prefill_scratch_bytes_ex` + `packed_values_partials_
    // exp_max_bytes`) is 1.5x that raw size PLUS the exp_sums/max_logits
    // pair, so a chunk the raw check called "fits" could -- and, measured
    // once, did -- charge 50%+ more than the nominal budget (2048 x 16 x
    // 256 x 2 B x 32 partitions = exactly 512 MiB raw; the actual charge
    // was 776 MiB). Round-10's fix priced the margined formula here too.
    //
    // RETRACTED (round-11 review, a REAL defect this repository's own
    // record keeps rather than silently editing away, per DESIGN
    // §7.0.1): `kPrefillScratchBudgetBytesPackedValues` (512 MiB) was
    // calibrated against the RAW proxy -- chunk 64 measured PASSING at
    // n_ctx ~119k, chunk 128 measured FAULTING there (fit.h's own
    // calibration comment, above the constant) -- not against the
    // margined formula round-10 substituted in. Margining `fits()` moved
    // the belt off that calibration: measured on the card, `wanted` =
    // 262,144 now gives ceiling 32 and `at_depth(101,824)` also gives 32,
    // while the card itself measured and ADOPTED chunk 64 at n_ctx
    // 101,824 -- the margined belt is now WRONG in the direction that
    // matters (refusing/underserving a chunk the hardware has actually
    // proven), not merely conservative. `fits()` is restored to the RAW
    // proxy the 512 MiB budget was actually calibrated against; the 1.5x
    // margin (and the exp_sums/max_logits pair) stay exactly where they
    // were before this predicate ever touched them -- in the CHARGED term
    // (`packed_values_prefill_scratch_bytes_ex` /
    // `packed_values_partials_exp_max_bytes`, both still called at the
    // belt's OUTPUT chunk by every reservation call site) -- so the
    // reservation still charges margin, only the belt's own chunk choice
    // no longer double-counts it against a budget that was never measured
    // with margin included. The round-10 "50%+ over budget" measurement
    // was real; the fix for it is a future, separately re-measured
    // budget constant for the margined case, not folding margin into a
    // predicate calibrated without it.
    auto fits = [&](int chunk) {
        const uint64_t raw = static_cast<uint64_t>(chunk) * static_cast<uint64_t>(heads) *
                             static_cast<uint64_t>(head_size) *
                             static_cast<uint64_t>(element_bytes) *
                             static_cast<uint64_t>(partitions);
        return raw <= scratch_budget_bytes;
    };

    int chunk = requested_chunk;
    while (!fits(chunk) && chunk > block_size) {
        int next = chunk / 2;
        next     = ((next + block_size - 1) / block_size) * block_size;  // round UP to a block
        // `next` is always `< chunk` here -- see the no-guard-needed proof
        // above (round-2 review residual 2 removed the guard that used to
        // sit on this line). `next < block_size` is kept as a defensive
        // floor even though the same proof shows `raw >= 1` whenever the
        // loop runs (so `next >= block_size` already): it costs one
        // comparison and does not depend on `block_size` staying far from
        // overflowing `next + block_size - 1` above, which the removed
        // guard's sibling did not protect against either.
        if (next < block_size) next = block_size;
        chunk = next;
    }
    return chunk;  // block_size is returned even if it does not fit -- the floor, not a refusal
}

inline int prefill_chunk_cap_for_packed_values_ex(int requested_chunk, long long partitions,
                                                   int heads, int head_size, int element_bytes,
                                                   uint64_t scratch_budget_bytes, int block_size) {
    int chunk = prefill_chunk_cap_for_packed_values_budget_only_ex(
        requested_chunk, partitions, heads, head_size, element_bytes, scratch_budget_bytes,
        block_size);
    if (requested_chunk <= 0 || partitions <= 0 || heads <= 0 || head_size <= 0 ||
        element_bytes <= 0 || scratch_budget_bytes == 0 || block_size <= 0) {
        return chunk;  // the no-op guard above already passed the request straight through
    }
    // Round-10 review, finding 3: fitting the scratch proxy is NECESSARY
    // but not SUFFICIENT -- a chunk this large has never been measured to
    // actually pass on ANY plugin (the card's own patched-plugin run at
    // chunk 2048 crashed; the cause is under investigation and may or may
    // not be this same proxy gap). Applied AFTER the budget-driven shrink
    // above, never before it (a chunk the budget check already shrunk
    // below the ceiling is left alone) and never WIDENING anything (only
    // ever pulls `chunk` down, same direction every guard in this
    // function already goes): the served chunk under 4-bit values never
    // silently exceeds the largest chunk actually validated,
    // `kMaxMeasuredPackedValuesChunk`.
    if (chunk > kMaxMeasuredPackedValuesChunk) {
        int capped = kMaxMeasuredPackedValuesChunk;
        capped     = (capped / block_size) * block_size;
        if (capped < block_size) capped = block_size;
        chunk = capped;
    }
    return chunk;
}

inline int prefill_chunk_cap_for_packed_values(int requested_chunk, long long max_ctx_tokens,
                                               int heads, int head_size,
                                               uint64_t scratch_budget_bytes, int block_size) {
    if (requested_chunk <= 0 || max_ctx_tokens <= 0 || heads <= 0 || head_size <= 0 ||
        scratch_budget_bytes == 0 || block_size <= 0) {
        return requested_chunk;
    }
    // ceil(max_ctx_tokens / 256), the plugin's own partition width (recon,
    // above) -- not this repository's KV page size (kv_block_tokens_) or
    // `block_size` (the --prefill-chunk granule this function floors to).
    const long long partitions = (max_ctx_tokens + 255) / 256;
    return prefill_chunk_cap_for_packed_values_ex(requested_chunk, partitions, heads, head_size,
                                                  /*element_bytes=*/4, scratch_budget_bytes,
                                                  block_size);
}

// M9 fit-side charge for the packed-4-bit-values prefill scratch buffer
// (consistent with the measured fault, 2026-09-03, 16 GiB-class card, coder
// artifact, `--paged-kv u8:i4`, host-side VRAM allocator sampled every 2 s
// -- the sampler measured AGGREGATE free VRAM against a proxy formula, not
// a direct trace of this one plugin buffer, so "consistent with" rather
// than "root cause": round-3 review, finding 1): the long
// prefill under 4-bit packed values grows the plugin's mixed-stage
// paged-attention scratch buffer as `past` grows (see
// kPrefillScratchBudgetBytesPackedValues's own comment for the buffer
// formula and its provenance) -- but the activation probe above runs at
// past 0, where the mixed stage allocates nothing, so the reservation never
// charged it. With the auto-fit pool sized to 171,312 tokens (1.44 GiB KV)
// free VRAM reached 0 MiB during a 119k-token prefill and the driver's
// rebind worker failed a page-table allocation ("VM worker error: -12",
// "exec queue reset detected") -- CL_OUT_OF_RESOURCES at the runtime. A
// 131,072-token pool (340 MiB more headroom) passed the same prompt with a
// minimum of 492 MiB free. The fit must charge this term, not just the belt
// that caps the prefill chunk after the fact (prefill_chunk_cap_for_packed_
// values, above): the belt only protects a load that already reserved
// enough room to survive the correction, and 171,312 was not one.
//
// The buffer, at the SERVED chunk and the SERVED pool depth: `chunk x heads
// x head_size x 4 bytes x ceil(n_ctx_tokens / 256)`, x1.5 for the resize
// overlap -- a MEASURED bound, not a round number: the same host-side VRAM
// sampler (16 GiB card, chunk 128, 119,074-token prompt, 131,072 pool) that
// pins the buffer's own size also pins how much of it the prefill actually
// holds and how much a resize costs. Free VRAM fell 1,319 -> 492 MiB over
// the whole prefill against a 932 MiB buffer proxy at that depth -- ~830
// MiB consumed, ~0.9x of the buffer, i.e. the buffer is held for very
// close to the whole prefill (round up to 1.0x: the proxy is a yardstick,
// not exact, per this file's own caveat on kPrefillScratchBudgetBytesPacked
// Values, and 1.0x is the conservative side of 0.9x). At the fault itself
// (171,312-token auto-fit pool) free VRAM during a resize traced 60 -> 562
// -> 56 MiB -- a RELEASE then a REALLOCATE (free rises as the old-size
// buffer is dropped, then falls again as the new, larger one is
// committed), not two buffers held at once, so the resize's own PEAK
// residency stays at or below the same ~1.0x the buffer is held at
// everywhere else in the prefill -- it is not a second occupancy stacked
// on top of the first. 1.5x is margin ABOVE that observed <=1.0x peak (not
// 1.0x-held plus a separate swing, which would not even add to 1.5x --
// 1.0 + 0.55 is 1.55), replacing the earlier, unmeasured 2x (a full second
// copy, chosen before the sampler data existed). `heads` is QUERY
// heads, `head_size` is `head_dim` -- the same geometry packed_values_
// scratch_geometry resolves for the belt above; this function takes them
// as given so it is testable without config.json at all.
// Round-5 (0015 engine side): generalized the same way the belt above was
// -- `partitions` and `element_bytes` are inputs, `packed_values_prefill_
// scratch_bytes` below is the thin, byte-identical wrapper (`partitions =
// ceil(n_ctx_tokens/256)`, `element_bytes = 4`) every existing caller and
// test keeps using unmodified. The patched plugin's own bound
// (`PAGED_ATTENTION_MAX_PARTITIONS`, design note o-0015-design.md §C) caps
// partitions below `ceil(n_ctx_tokens/256)` -- `packed_values_bounded_
// partitions`, below, computes that -- and its f16-corrected host sizing
// (the note's §B recon: `tmp_out` was ALREADY f16 in the kernel, only the
// host-side buffer descriptor hardcoded 4-byte elements) halves
// `element_bytes` to 2, together, whenever the plugin accepts the key at
// all (the key's mere presence is how the patched plugin is detected --
// see backend_ov.cpp's load path).
inline uint64_t packed_values_prefill_scratch_bytes_ex(int chunk, long long partitions, int heads,
                                                        int head_size, int element_bytes) {
    if (chunk <= 0 || partitions <= 0 || heads <= 0 || head_size <= 0 || element_bytes <= 0) {
        return 0;
    }
    const uint64_t single_buffer = static_cast<uint64_t>(chunk) * static_cast<uint64_t>(heads) *
                                   static_cast<uint64_t>(head_size) *
                                   static_cast<uint64_t>(element_bytes) *
                                   static_cast<uint64_t>(partitions);
    // ceil(1.5 * single_buffer) == ceil(3 * single_buffer / 2) ==
    // (3 * single_buffer + 1) / 2 in integer arithmetic (b - 1 == 1 for
    // b == 2 in the general ceil(a/b) == (a + b - 1) / b identity). The
    // measured 1.5x overlap bound is unchanged by 0015 -- the resize this
    // margin covers is the SAME buffer growing between partition counts,
    // whatever its element width, and the note's own measurement plan (§D)
    // does not claim otherwise.
    return (3ull * single_buffer + 1ull) / 2ull;
}

inline uint64_t packed_values_prefill_scratch_bytes(int chunk, long long n_ctx_tokens, int heads,
                                                     int head_size) {
    if (chunk <= 0 || n_ctx_tokens <= 0 || heads <= 0 || head_size <= 0) return 0;
    const long long partitions = (n_ctx_tokens + 255) / 256;
    return packed_values_prefill_scratch_bytes_ex(chunk, partitions, heads, head_size,
                                                  /*element_bytes=*/4);
}

// The bound itself (design note o-0015-design.md §A/§C): `min(ceil(n_ctx_
// tokens/256), max_partitions)` when `max_partitions > 0` (the plugin
// accepted the key and it asked for a real bound), else the unbounded
// `ceil(n_ctx_tokens/256)` this file always computed before 0015 -- `0` is
// "no bound", not "zero partitions", the same convention `PAGED_ATTENTION_
// MAX_PARTITIONS`'s own default carries on the plugin side.
inline long long packed_values_bounded_partitions(long long n_ctx_tokens,
                                                   long long max_partitions) {
    if (n_ctx_tokens <= 0) return 0;
    const long long raw = (n_ctx_tokens + 255) / 256;
    return max_partitions > 0 ? std::min(raw, max_partitions) : raw;
}

// Round-6 review, finding 4: the design note's own §C formula has a SECOND
// additive term this file did not charge until now -- exp_sums and
// max_logits, two f32 buffers the mixed-stage kernel writes alongside
// tmp_out, each `chunk x heads x partitions x 4 bytes` (read from the
// recon tree's own source, o-0015-design.md §B: "buffers 5/6 (exp_sums/
// max_logits, genuinely f32, line 31) are correctly 4 B" already, on both
// an unpatched and a patched plugin -- unlike tmp_out, this pair's element
// width does NOT change with 0015, so `element_bytes` is not a parameter
// here). NOT scaled by the 1.5x overlap margin: that margin covers tmp_out
// alone (design note §B/§C: the measured resize swing this file's own
// packed_values_prefill_scratch_bytes prices is tmp_out's own growth, and
// the note's measurement plan does not claim anything about exp_sums/
// max_logits resizing the same way). Applies in BOTH arms -- bounded
// (`partitions = max_partitions`) and unbounded (`partitions =
// ceil(n_ctx/256)`, via the per-token wrapper below) -- since the pair
// exists on every mixed-stage paged-attention dispatch, patched plugin or
// not; the design note's own bound (§A) shrinks it exactly the way it
// shrinks tmp_out.
inline uint64_t packed_values_partials_exp_max_bytes(int chunk, long long partitions, int heads) {
    if (chunk <= 0 || partitions <= 0 || heads <= 0) return 0;
    return 2ull * static_cast<uint64_t>(chunk) * static_cast<uint64_t>(heads) * 4ull *
           static_cast<uint64_t>(partitions);
}

// The per-token slope for the pair above, same ceiling-conservative
// construction packed_values_prefill_scratch_bytes_per_token_ex uses for
// tmp_out (see that function's own proof -- identical shape, `per_token =
// ceil(one_partition_bytes / 256)` so `per_token * n_ctx + one_partition_
// bytes` never undercounts `packed_values_partials_exp_max_bytes(chunk,
// ceil(n_ctx/256), heads)` for any n_ctx > 0).
inline uint64_t packed_values_partials_exp_max_bytes_per_token(int chunk, int heads) {
    const uint64_t one_partition = packed_values_partials_exp_max_bytes(chunk, 1, heads);
    if (one_partition == 0) return 0;
    return (one_partition + 255) / 256;
}

// The per-token SLOPE the term above climbs at, for a FIXED chunk -- what
// lets the fit solve for max_ctx in closed form instead of also iterating
// on n_ctx (see fit_context_packed_values below, which still has to iterate
// on CHUNK, because the belt's own chunk choice depends on the depth being
// tried). At chunk 128, 16 query heads, 256 head_size: 128 x 16 KiB / 256 =
// 8 KiB of buffer per token of context, x1.5 for the measured overlap bound
// = 12 KiB/token -- more than the u8:i4 KV term itself (8.8 KiB/token, M8).
//
// Deliberately `ceil(one_partition_bytes / 256)`, not `one_partition_bytes
// / 256` floored: charging the CEILING of the true per-token rate, plus
// `one_partition_bytes` again as a flat term (fit_context_packed_values
// folds that into `activations`), makes the closed-form charge
// `per_token * n_ctx + one_partition_bytes` provably >=
// `packed_values_prefill_scratch_bytes(chunk, n_ctx, heads, head_size)` for
// EVERY `n_ctx > 0`, not merely the depths this file's tests happen to
// check: `packed_values_prefill_scratch_bytes`'s own `ceil(n_ctx/256) <=
// n_ctx/256 + 1`, so the exact total is at most
// `one_partition_bytes * n_ctx / 256 + one_partition_bytes` (real
// arithmetic); `per_token >= one_partition_bytes / 256` by construction
// (ceiling), so `per_token * n_ctx >= one_partition_bytes * n_ctx / 256`,
// and adding `one_partition_bytes` to both sides gives the charge as an
// upper bound on the exact total. Conservative, matching the direction
// every other over-charge in this file is deliberately wrong in (never
// unsafe).
// Round-5 (0015 engine side): generalized to take `element_bytes` too --
// `packed_values_prefill_scratch_bytes_per_token`, below, is the thin,
// byte-identical wrapper (`element_bytes = 4`) every existing caller keeps.
// Only meaningful for the UNBOUNDED case (fit_context_packed_values, below,
// never calls this when `max_partitions > 0`: a BOUNDED term stops growing
// with n_ctx past a small, fixed depth, so it is charged as a flat amount
// instead of a per-token slope -- see that function's own comment).
inline uint64_t packed_values_prefill_scratch_bytes_per_token_ex(int chunk, int heads,
                                                                  int head_size,
                                                                  int element_bytes) {
    const uint64_t one_partition =
        packed_values_prefill_scratch_bytes_ex(chunk, 1, heads, head_size, element_bytes);
    if (one_partition == 0) return 0;
    return (one_partition + 255) / 256;
}

inline uint64_t packed_values_prefill_scratch_bytes_per_token(int chunk, int heads,
                                                               int head_size) {
    return packed_values_prefill_scratch_bytes_per_token_ex(chunk, heads, head_size,
                                                             /*element_bytes=*/4);
}

// The fit's own climb, packed-4-bit-values case: `fit_context` alone cannot
// price `packed_values_prefill_scratch_bytes_per_token`'s term, because the
// chunk the M9 belt (`prefill_chunk_cap_for_packed_values`, above) would
// pick depends on the SERVED DEPTH -- exactly the `max_ctx` `fit_context`
// is solving for. This resolves that the same way the belt itself is
// meant to run (its own comment: "the chunk this load will actually
// serve"): try a candidate depth, ask the belt what chunk IT would pick
// for that depth, recharge the term at that chunk, and repeat until the
// belt's answer stops moving.
//
// Monotone by construction, so the loop cannot oscillate:
//   Lemma 1 (term is non-decreasing in chunk): both factors in
//   `packed_values_prefill_scratch_bytes_per_token` are non-negative
//   multiplicative terms of `chunk`, so a SMALLER chunk charges no MORE --
//   and `fit_context`'s own `budget = total - fixed - ...` means a
//   smaller charge can only report an equal-or-LARGER `max_ctx`. So the
//   map "chunk -> fit_context's max_ctx at that chunk's term" is
//   non-increasing in chunk.
//   Lemma 2 (the belt is non-increasing in depth): prefill_chunk_cap_for_
//   packed_values's own `fits()` predicate grows with `partitions`, which
//   grows with `max_ctx_tokens` -- so a LARGER candidate depth can only
//   force the belt to an equal-or-SMALLER chunk (its own doc comment
//   states this too: "never raises the chunk").
// Composing a non-increasing map (chunk -> max_ctx) with a non-increasing
// map (max_ctx -> chunk) gives a NON-DECREASING map chunk -> chunk' on the
// belt's own halving ladder (a finite, strictly decreasing chain from
// `requested_chunk` down to `block_size`). Iterating a non-decreasing
// self-map on a finite chain from either end converges monotonically to a
// fixed point without oscillating (the first two iterations settle
// whether the sequence is climbing or holding; a non-decreasing sequence
// bounded above by `requested_chunk` on a finite chain cannot climb
// forever) -- so this loop needs at most as many rounds as the ladder has
// rungs (a --prefill-chunk of a few thousand halved down to a
// --kv-block-size of 16 or 32 is under a dozen), and the fixed iteration
// cap below is a structural belt on top of that proof, not a reliance on
// it never being wrong.
//
// `base` must NOT already carry a packed-values term in `kv_bytes_token`
// or `activations` -- this function adds its own on top of whatever `base`
// already has (the honest u8/i4 KV cost, the measured activation probe,
// and every other M7/M9 term), the same way the belt call site adds its
// cap on top of the auto-fit `chunk` rather than replacing it.
//
// Round-5 (0015 engine side, design note o-0015-design.md §C): two more
// trailing parameters, both defaulted so every existing call site and test
// reproduces today's UNBOUNDED, 4-byte-element behaviour byte for byte.
// `max_partitions > 0` means the plugin accepted `PAGED_ATTENTION_MAX_
// PARTITIONS` and asked for a real bound; `element_bytes` is 2 whenever
// the plugin accepted the key at all -- DETECTION CONTRACT (round-7
// review, reverting a round-6 attempt at a second, independent read-only
// key: the plugin implementer keeps 0015 as one unit and will not carry a
// separate key for this): a plugin exposing PAGED_ATTENTION_MAX_PARTITIONS
// carries the f16 host-sizing fix too, both ship together in patch 0015,
// unconditionally. A plugin with the bound key but without the sizing fix
// would be under-charged by half (4-byte elements priced as 2); no such
// plugin exists in this series. 4 otherwise (the key not accepted at all).
//
// BOUNDED pricing is not the same shape as unbounded: `packed_values_
// bounded_partitions` is non-decreasing in n_ctx up to `max_partitions` and
// FLAT past it (`min(ceil(n_ctx/256), max_partitions)`), so the term itself
// is bounded ABOVE, for every n_ctx, by its own value at partitions ==
// `max_partitions`. Charging exactly that value, with NO per-token slope,
// is not "exact" in the sense of matching the plugin's own allocation byte
// for byte -- the 1.5x overlap margin (packed_values_prefill_scratch_
// bytes_ex's own comment) is charged here the same as everywhere else in
// this file, so the reservation is always ABOVE what the buffer itself
// needs, by design. What changes with the bound is that the margined
// value STOPS GROWING past `max_partitions * 256` served tokens (true of
// essentially every auto-fit scenario an operator would set
// --paged-attention-max-partitions for -- the note's own P_MAX=32 example
// crosses this at 8,192 tokens), instead of climbing with `n_ctx` forever
// the way the unbounded term does. This is why `per_token_bytes` is 0
// whenever `max_partitions > 0`: the per-token/fixed-margin split the
// unbounded path needs (to fold a growing term into `fit_context`'s
// closed-form division) has nothing to do in the bounded case, where the
// whole margined charge is already a known constant.
//
// Round-6 review, finding 1 (REAL defect): the buffer this term prices is
// per INFER REQUEST, not per compiled model -- concurrent lanes each need
// their own copy, the same reasoning that already makes the UNBOUNDED
// per-token part lane-multiplied (it is folded into `kv_bytes_token`,
// which `fit_context` itself multiplies by `lanes`). The bounded charge,
// though, goes into `t.activations`, which `fit_context` does NOT
// multiply by `lanes` -- so it must be multiplied here, explicitly, before
// it is added, or a multi-lane bounded load under-reserves by a factor of
// `lanes`. The UNBOUNDED arm's own `fixed` (a small one-partition ceiling-
// rounding margin, not a real per-request buffer) is deliberately left as
// it was reviewed before this round -- unmultiplied, charged once -- since
// nothing in this round's review asked that margin to change and it was
// never the buffer in question.
struct PackedValuesFitTerm {
    FitResult fit             = {};  // fit_context()'s own result, term already included
    int       chunk           = 0;   // the belt's chunk this fit is consistent with
    uint64_t  per_token_bytes = 0;   // charged per token, per lane (folded into kv_bytes_token);
                                     // 0 when bounded -- the whole term is `fixed_bytes` instead
    uint64_t  fixed_bytes     = 0;   // UNBOUNDED: the one-partition rounding margin, folded
                                     // into activations, ALREADY multiplied by lanes (round-8
                                     // review, item 5 -- it is part of the same per-request
                                     // buffer the bounded arm's own term is, not a model-wide
                                     // constant). BOUNDED: the WHOLE term (tmp_out + exp_sums/
                                     // max_logits at the bound), also already lane-multiplied --
                                     // see this struct's own comment, round-6 review finding 1.
    int       iterations      = 0;   // fixed-point rounds this took (>= 1 whenever geometry is valid)
};

inline PackedValuesFitTerm fit_context_packed_values(const FitTerms& base, int requested_chunk,
                                                      int heads, int head_size,
                                                      uint64_t scratch_budget_bytes, int block_size,
                                                      bool belt_enabled = true,
                                                      int max_partitions = 0,
                                                      int element_bytes = 4) {
    PackedValuesFitTerm r;
    if (requested_chunk <= 0 || heads <= 0 || head_size <= 0 || element_bytes <= 0) {
        // Nothing to price (mirrors prefill_chunk_cap_for_packed_values's
        // own no-op guard): the term stays zero and `chunk` passes through
        // unchanged, same as the caller never having called this at all.
        r.fit   = fit_context(base);
        r.chunk = requested_chunk;
        return r;
    }

    // Round-15 review (Opus, REAL defect, RETRACTS the paragraph this
    // replaces -- kept on the record rather than silently edited away,
    // per DESIGN §7.0.1): `belt_enabled = false` used to still price the
    // FULL term at the uncapped `requested_chunk` -- one pass, no ladder,
    // but still a real (and, at a large requested chunk, large) charge
    // subtracted from the budget. Measured on the card:
    // `ARCINT_PREFILL_CHUNK_CAP=off` with `--n-ctx 101824 --prefill-chunk
    // 128` still charged 1,200.2 MiB at chunk 128 and the load was
    // REFUSED (admits 32,256, far short of 101,824) -- the switch bypassed
    // the belt and the measured cap, exactly as designed, but the
    // depth-scaled term it still charged was enough on its own to refuse
    // the load before the plugin's own kernel ever ran a prefill at that
    // chunk. A switch whose whole purpose is reproducing the plugin's
    // OWN fault line cannot do that if THIS repository's own budget math
    // refuses the load first -- the fault this switch exists to find
    // lives in the pool the term forbids. `belt_enabled = false` is now a
    // FULL measurement bypass: no belt, no cap, AND no scratch term --
    // the reservation proceeds exactly as if this load were not under
    // 4-bit paged VALUES at all, `chunk` stays pinned at `requested_chunk`
    // (so the actual served prefill chunk is still the operator's own
    // request, unclamped), and whatever admits or refuses is Phase E's
    // real, on-card allocation -- not this file's own conservative
    // estimate of one buffer it cannot see the plugin's own true size
    // for.
    if (!belt_enabled) {
        r.fit   = fit_context(base);
        r.chunk = requested_chunk;
        r.per_token_bytes = 0;
        r.fixed_bytes      = 0;
        r.iterations       = 1;
        return r;
    }

    int chunk      = requested_chunk;
    int prev_chunk = 0;
    for (int i = 0; i < 32 && chunk != prev_chunk; ++i) {
        prev_chunk = chunk;
        FitTerms t = base;
        uint64_t per_token = 0;
        uint64_t fixed     = 0;
        if (max_partitions > 0) {
            // Bounded: one constant, no per-token growth -- tmp_out
            // (packed_values_prefill_scratch_bytes_ex) plus the exp_sums/
            // max_logits pair (packed_values_partials_exp_max_bytes,
            // round-6 review finding 4 -- always 4-byte, unaffected by
            // `element_bytes`), both evaluated AT the bound, then the
            // whole per-lane total multiplied by `lanes` (round-6 review
            // finding 1: this buffer is per infer request).
            const uint64_t per_lane =
                packed_values_prefill_scratch_bytes_ex(chunk, max_partitions, heads, head_size,
                                                        element_bytes) +
                packed_values_partials_exp_max_bytes(chunk, max_partitions, heads);
            const uint64_t lanes_c = static_cast<uint64_t>(std::max(1, base.lanes));
            fixed = per_lane * lanes_c;
        } else {
            per_token = packed_values_prefill_scratch_bytes_per_token_ex(chunk, heads, head_size,
                                                                          element_bytes) +
                        packed_values_partials_exp_max_bytes_per_token(chunk, heads);
            // Round-8 review (Opus), item 5: this one-partition margin is
            // ALSO part of the per-infer-request buffer (the same buffer
            // the bounded arm's own `per_lane` above prices), not a
            // model-wide constant -- an earlier version left it
            // unmultiplied, inconsistent with the bounded arm's own
            // round-6 review finding 1 fix. Lane-multiplied here too.
            const uint64_t one_partition =
                packed_values_prefill_scratch_bytes_ex(chunk, 1, heads, head_size,
                                                       element_bytes) +
                packed_values_partials_exp_max_bytes(chunk, 1, heads);
            const uint64_t lanes_c = static_cast<uint64_t>(std::max(1, base.lanes));
            fixed = one_partition * lanes_c;
        }
        t.kv_bytes_token += per_token;
        t.activations     += fixed;
        r.fit             = fit_context(t);
        r.per_token_bytes = per_token;
        r.fixed_bytes      = fixed;
        r.iterations       = i + 1;
        // `belt_enabled` is always true past the early return above --
        // this loop no longer runs at all when it is false.
        {
            // The belt's own proxy must price the SAME buffer this term
            // just charged -- partitions bounded the same way, the same
            // element size -- or it halves the chunk for a buffer that no
            // longer exists at that size (design note o-0015-design.md §C:
            // "prefill_chunk_cap_for_packed_values keeps its shape and
            // simply stops firing at the served depths").
            const long long partitions_for_belt =
                max_partitions > 0 ? packed_values_bounded_partitions(r.fit.max_ctx, max_partitions)
                                   : (r.fit.max_ctx + 255) / 256;
            chunk = prefill_chunk_cap_for_packed_values_ex(requested_chunk, partitions_for_belt,
                                                            heads, head_size, element_bytes,
                                                            scratch_budget_bytes, block_size);
        }
    }
    r.chunk = chunk;
    return r;
}

// The packed-values scratch term's total reservation contribution at a
// SERVED depth and lane count -- the same shape fit_context_packed_values
// charges internally inside its own climb (`fixed_bytes` once, folded into
// `activations`; `per_token_bytes` times lanes times n_ctx, folded into
// `kv_bytes_token`, which fit_context itself multiplies by both `lanes` and
// `max_ctx`). Extracted (round-3 review, findings 2 and 4) so Phase E's
// ceiling check (phase_e_ceiling_bytes, below) and backend_ov.cpp's
// load-time summary/detail log lines share ONE formula instead of three
// independent reimplementations of "lanes * n_ctx * per_token + fixed" --
// the load-time log line's own earlier version (round-2 review) omitted
// the `lanes` factor entirely, which this shared formula cannot do by
// construction once every caller routes through it.
inline uint64_t packed_values_scratch_reservation_bytes(uint64_t fixed_bytes,
                                                         uint64_t per_token_bytes, int lanes,
                                                         long long n_ctx) {
    const uint64_t lanes_c = static_cast<uint64_t>(std::max(1, lanes));
    const uint64_t n_ctx_c = static_cast<uint64_t>(std::max<long long>(n_ctx, 0));
    return fixed_bytes + lanes_c * n_ctx_c * per_token_bytes;
}

// Round-3 review, finding 2 (REAL defect): the F4 fix (round-2 review)
// excluded the packed-values scratch term from Phase E's `predicted_total`
// -- correctly, since that figure asks "what should be resident the
// instant the KV pool is allocated," and the scratch buffer is not, yet.
// But the SAME round's comment claimed the ceiling check "already reflects
// [the term] -- `total` itself was sized net of the term back at the
// climb": false. `total` is `device_total_mem_size`, a constant read once
// at load and never adjusted for anything the climb charges (the climb
// shrinks `max_ctx`, not `total`) -- so a bare `ceiling = total - margin`
// never held the scratch term back at all. The result: an observed
// residency landing just under that ceiling is accepted as-is, the pool
// keeps every one of those bytes, and the scratch buffer -- which has not
// allocated yet at this point in the load -- still has to fit in whatever
// is left over once the first real prefill grows it. When there is not
// enough left, that is the ORIGINAL fault this whole term exists to
// prevent, now reintroduced silently by the ceiling check that was
// supposed to be the last line of defence.
//
// The fix: the pool may only be ACCEPTED (not merely predicted) up to
// `total - margin - scratch_term_bytes`, where `scratch_term_bytes` is
// `packed_values_scratch_reservation_bytes` evaluated at the depth THIS
// pass actually requested (`paged_n_ctx_`) -- the same per-pass depth
// `predicted_total` already reads elsewhere in Phase E, so the ceiling
// tightens and loosens in step with the very n_ctx the replay loop is
// trying. `scratch_term_bytes` is 0 whenever the load is not under 4-bit
// values (both `packed_values_scratch_fixed_bytes_` and `_per_token_bytes_`
// default to 0), so this is a no-op arithmetic-wise on every other path --
// it only ever narrows the ceiling, never widens it.
inline uint64_t phase_e_ceiling_bytes(uint64_t total, uint64_t margin,
                                      uint64_t scratch_term_bytes) {
    const uint64_t floor_bytes = margin + scratch_term_bytes;
    return total > floor_bytes ? total - floor_bytes : 0;
}

// Round-3 review, finding 3: the belt call site's own chunk choice
// (backend_ov.cpp, after Phase E -- round-2 review, F1) extracted into a
// pure, named function so a test exercises the SAME implementation the
// call site calls, not a second, independently-typed copy of `std::min`
// that cannot fail when the call site regresses (the round-2 review's own
// three tests reimplemented the choice inline and would keep passing even
// if backend_ov.cpp reverted to the unclamped `prefill_chunk` alone).
// `priced_chunk` is `packed_values_scratch_chunk_` -- the chunk the fit's
// own fixed point (fit_context_packed_values, above) actually priced the
// term at; `prefill_chunk` is the served-chunk candidate before the belt
// runs. 0 or negative `priced_chunk` (no packed-values term charged this
// load) passes `prefill_chunk` through unchanged, the same "nothing to
// price" convention every other guard in this file uses.
inline int belt_requested_chunk(int prefill_chunk, int priced_chunk) {
    return priced_chunk > 0 ? std::min(prefill_chunk, priced_chunk) : prefill_chunk;
}

// Round-8 review (Opus, folding the card's second refusal): the round-7
// "re-probe at the belt's smaller chunk" fix could not work -- this file's
// own measured record (backend_ov.cpp, the activation-probe climb's own
// comments) is that the plugin's intermediate pool grows to the largest
// shape it has ever seen and NEVER SHRINKS, so a probe taken AFTER a
// bigger one has already run returns the stale, bigger reading -- the
// round-7 loop was a no-op in practice, which is exactly why the SAME
// card, on the SAME (round-7) binary, still refused 101,824 explicitly
// while auto-fit adopted it. The honest fix does not probe twice at two
// chunks at all: the SERVED CHUNK must be known BEFORE the one real
// activation probe ever runs, so the probe is only ever taken at the
// chunk that will actually be served.
//
// `fit_context_packed_values_at_depth` is the primitive this requires: an
// EXACT (not iteratively-solved-for) evaluation of the packed-values term
// at a KNOWN depth. Unlike `fit_context_packed_values` above (which
// SOLVES for `max_ctx` by iterating candidate depths until the belt's own
// chunk choice stops moving -- correct for auto-fit, where the served
// depth is exactly what is being searched for, but WRONG for an explicit
// --n-ctx, where the depth is already given and the only question is
// whether THAT depth is admissible), this function asks a narrower
// question with no search at all: at depth `depth`, what chunk would the
// belt pick, what does the term cost there, and is `depth` itself
// admissible under that pricing? Both `chunk` and the term's exact value
// are pure functions of `depth` alone (`packed_values_bounded_partitions`
// and the belt do not read `base` at all), so there is no ambiguity about
// which of possibly several self-consistent fixed points to report -- the
// measured defect (`fit_context_packed_values` seeded from a chunk the
// activation climb happened to explore first, e.g. 1024, converging to a
// DIFFERENT self-consistent point -- 60,880 tokens at chunk 128 -- than
// the one that actually admits the request) does not exist here, because
// there is nothing to converge: `depth` is the caller's own input, not a
// search variable.
//
// `base.activations` must already be whatever this load has decided to
// charge for activations (the real, measured figure once probed, or an
// analytic pre-probe estimate before that -- see the chunk-ceiling search
// below); this function does not touch it. The term itself IS lane-
// multiplied here (round-6 review, finding 1: the buffer is per infer
// request), matching `fit_context_packed_values`'s own bounded arm.
//
// Round-14 review (Opus), finding 1 (REAL defect, third instance of this
// class -- round-12's ceiling and round-11's `prefill_chunk_` seed were
// the first two, both in backend_ov.cpp): this function had no
// `belt_enabled` parameter, so `ARCINT_PREFILL_CHUNK_CAP=off` (the
// measurement switch `fit_context_packed_values` above already honours
// via its own `belt_enabled`) was silently ignored on the EXPLICIT
// --n-ctx path -- an operator running `--n-ctx 102384
// ARCINT_PREFILL_CHUNK_CAP=off` still had the belt (and the measured-cap
// ceiling on top of it) applied unconditionally, while the load's own
// log claimed the belt was disabled.
//
// Round-15 review (Opus, REAL defect, RETRACTS this paragraph's own
// original claim -- kept on the record rather than silently edited away,
// per DESIGN §7.0.1): `belt_enabled=false` used to still price
// `requested_chunk` unchanged -- no ladder, no cap, but still a real
// term. Measured on the card (see `fit_context_packed_values`'s own
// retraction, above, for the full account): a depth-scaled term charged
// at the uncapped chunk was, on its own, enough to refuse the load
// before the plugin's own kernel ever ran a prefill -- the switch bypassed
// the belt and the cap exactly as designed, but not the thing that
// actually stood between the load and the fault line it exists to find.
// `belt_enabled=false` is now a FULL bypass here too, matching
// `fit_context_packed_values` exactly: no belt, no cap, no scratch term
// (`fixed_bytes` and `per_token_bytes` both 0) -- `chunk` still pins at
// `requested_chunk` (the served chunk is still the operator's own
// request, unclamped), but nothing is subtracted from the budget for it.
inline PackedValuesFitTerm fit_context_packed_values_at_depth(
    const FitTerms& base, int requested_chunk, int heads, int head_size,
    uint64_t scratch_budget_bytes, int block_size, long long depth, int max_partitions,
    int element_bytes, bool belt_enabled = true) {
    PackedValuesFitTerm r;
    if (requested_chunk <= 0 || heads <= 0 || head_size <= 0 || element_bytes <= 0 ||
        depth <= 0) {
        r.fit   = fit_context(base);
        r.chunk = requested_chunk;
        return r;
    }
    if (!belt_enabled) {
        r.fit   = fit_context(base);
        r.chunk = requested_chunk;
        r.fixed_bytes      = 0;
        r.per_token_bytes = 0;
        r.iterations       = 1;
        return r;
    }
    const long long partitions = packed_values_bounded_partitions(depth, max_partitions);
    const int       chunk      = prefill_chunk_cap_for_packed_values_ex(
        requested_chunk, partitions, heads, head_size, element_bytes, scratch_budget_bytes,
        block_size);
    const uint64_t single_lane =
        packed_values_prefill_scratch_bytes_ex(chunk, partitions, heads, head_size,
                                               element_bytes) +
        packed_values_partials_exp_max_bytes(chunk, partitions, heads);
    const uint64_t lanes_c = static_cast<uint64_t>(std::max(1, base.lanes));

    FitTerms t = base;
    t.activations += single_lane * lanes_c;
    r.fit             = fit_context(t);
    r.chunk           = chunk;
    r.fixed_bytes      = single_lane * lanes_c;  // the exact term at (chunk, depth); no
                                                 // per-token slope concept applies here
    r.per_token_bytes = 0;
    r.iterations       = 1;
    return r;
}

// RETRACTED: fit_context_packed_values_chunk_ceiling, an analytic,
// no-probe SEARCH for auto-fit's own chunk ceiling (round-8 review).
// Round-9 review (Opus), finding 1 (REAL defect): it failed OPEN -- when
// the analytic activation estimate at a candidate chunk left no budget at
// all (`no_term_fit.max_ctx <= 0`), it ACCEPTED that candidate and
// returned it unchanged, rather than treating "no budget" as a reason to
// try a smaller chunk. Checked against the card's own constants (16 GiB,
// weights 12.83 GiB, margin 256 MiB, KV 9,011 B/token, slope 0.62 GiB at
// chunk 128): `--prefill-chunk 2048` (the default) made the whole pre-pass
// a no-op (ceiling 2048, i.e. "no ceiling at all"). A halve-on-failure fix
// was tried and found to have a SECOND problem: the belt's own "return
// block_size even if it does not fit" floor behaviour, fed back into a
// self-consistent search, lets the search accept a floor chunk that does
// not actually satisfy the scratch budget and keep climbing to
// increasingly optimistic (and wrong) depths from there -- checked on the
// card's constants, the search does not stabilize near the real answer
// (chunk 64 at n_ctx ~101,824) at all, landing instead near chunk 32 at a
// depth several times too deep.
//
// The retraction: auto-fit does not need an analytic SEARCH for the
// CEILING here -- the belt's own chunk choice is a PURE function of the
// depth and the fixed geometry (packed_values_bounded_partitions and the
// belt itself take no activation or budget input), so `wanted` (the
// artifact's train maximum) is knowable up front and `fit_context_
// packed_values_at_depth` evaluated there gives a real, no-search ceiling
// with no probing required to compute it.
//
// Round-10 review (Opus), finding 1 (REAL defect -- retracts THIS
// comment's own round-9 conclusion, "evaluated at `wanted` is correct
// for BOTH paths, literally the same call"): `wanted` is the artifact's
// train MAXIMUM, not auto-fit's served depth -- auto-fit typically adopts
// something considerably shallower once the rest of the budget (weights,
// activations, margin) is accounted for. Evaluating the belt AT `wanted`
// therefore gives the SMALLEST, most conservative chunk the belt could
// ever need, correct as a safety ceiling for the one real activation
// probe this file can afford (see fit_context_packed_values_at_depth's
// own comment, above), but WRONG as the packed-values term's own final
// priced chunk: measured on the card's own constants, `wanted` =
// 262,144 evaluates to chunk 32, while the depth auto-fit actually goes
// on to adopt (103,040) only needs chunk 64 -- an explicit request for
// that SAME 103,040, evaluated honestly with `depth` = 103,040, priced
// chunk 64 and admitted it; auto-fit, pricing chunk 32 the whole time,
// served a different, more expensive reservation for the identical
// depth. (An earlier draft of this same paragraph conflated the two
// numbers outright, writing "evaluating at wanted = 101,824" -- 101,824
// is a DIFFERENT fixture's explicit request depth, used nowhere near
// this artifact's `wanted`; kept here, corrected, as a record of the
// mistake rather than silently fixed away.)
//
// The corrected design: `fit_context_packed_values_at_depth` at `wanted`
// is still step one -- the conservative ceiling the one real activation
// probe must not explore past. Auto-fit's own served chunk then comes
// from running `fit_context_packed_values` (the belt's own chunk-driven
// fixed-point SEARCH, unchanged, seeded from the operator's own
// requested/default chunk, not from the ceiling) to find the depth D* it
// actually adopts, then confirming via `fit_context_packed_values_at_
// depth(depth = D*)` -- the SAME primitive the explicit path uses --
// that D*'s own served chunk agrees. When the search's served chunk
// turns out bigger than what the ceiling-bounded probe measured
// activations at, backend_ov.cpp re-probes upward (valid: the plugin's
// pool only grows) and repeats; see backend_ov.cpp's own comment at the
// auto-fit call site for the loop. `fit_context_packed_values_at_depth`
// remains, unchanged, the single evaluation the explicit path needs
// (depth is already given there, no search required) -- this retraction
// is only about what auto-fit itself must do before reaching for it.
// (no replacement function needed for the CEILING search itself -- see
// fit_context_packed_values_at_depth, used by the explicit path directly
// and by auto-fit's own re-probe loop in backend_ov.cpp, and fit_context_
// packed_values, the unchanged belt search, which auto-fit now runs for
// its own served chunk instead of a second analytic pass.)

}  // namespace lgc
