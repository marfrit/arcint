#include "exec/fit.h"
#include "harness.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace lgc;

namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kMiB = 1ull << 20;

// The 2026-09-01 configuration DESIGN §7.0.2s measured and this milestone
// exists to fix: a 16 GiB-class card, --offload-ratio 20, weights 1.20 GiB
// resident regardless of ratio, u8 paged KV (~11.3 KiB/token). `total` below
// is constructed, not read off a card -- chosen so that feeding it through
// TODAY'S formula (slot_pool = 0, drafters = 0, the arithmetic load_paged had
// before M7) reproduces the exact printed number from that section:
// max ctx 1,172,016. That reproduction is the fixture's own self-check
// (fit_2026_09_01_zero_slot_reproduces_the_defect, below); the honest case
// then asks the same question with the expert-slot term included.
constexpr uint64_t kKvBytesToken = 11571;  // ~11.3 KiB/token, u8 paged KV
constexpr int       kKvBlockTokens = 16;    // backend_ov.cpp's kv_block_tokens_ default
constexpr uint64_t kWeights   = 1288490189;  // 1.20 GiB, round(1.20 * 2^30)
constexpr uint64_t kMargin    = 256 * kMiB;   // today's hardcoded fit margin
constexpr uint64_t kActivation = 697932186;   // ~0.65 GiB, order of the A770 128-tok probe
constexpr uint64_t kSlab      = 64 * kMiB;    // a plausible per-lane GDN slab
constexpr uint64_t kTotal     = 15883363831ull;  // engineered: see comment above
constexpr long long kDefectiveMaxCtx = 1172016;

FitTerms base_terms() {
    FitTerms t;
    t.total          = kTotal;
    t.weights        = kWeights;
    t.drafters       = 0;
    t.slot_pool      = 0;
    t.activations    = kActivation;
    t.slab_per_lane  = kSlab;
    t.kv_bytes_token = kKvBytesToken;
    t.margin         = kMargin;
    t.lanes           = 1;
    t.kv_block_tokens = kKvBlockTokens;
    t.n_ctx_floor     = 4096;
    return t;
}

// A 35B-q4-shaped MoE geometry, ratio 20 -- round numbers, not a specific
// artifact's real dimensions; only the ceiling arithmetic in
// expert_slot_bytes is under test here.
constexpr int      kNumExpert     = 128;
constexpr int      kRatioPct      = 20;
constexpr uint64_t kPerExpertBytes = 1 * kMiB;
constexpr int      kMoeLayers     = 48;

// Two ledgers (defect-3 rework, on-card measurement: 35B q4, 16 GiB card, u8
// KV, 1 lane, --offload-ratio 20). The plateau probe -- or a forced
// ARCINT_FIT_SLOT_BYTES -- prices the DEVICE (VRAM) charge: the LRU working
// set actually resident -- the probe itself measured 0.11 GiB (0.14 GiB was
// a separate forced ARCINT_FIT_SLOT_BYTES experiment value in that on-card
// run, not a measurement) -- against a promised 6.81 GiB
// total (0.8% off). The (100-ratio)% ceiling estimate above
// (kNumExpert/kRatioPct/kPerExpertBytes/kMoeLayers, run through
// expert_slot_bytes) prices the HOST (GTT) pool instead -- confirmed via
// /proc/<pid>/fdinfo, ~12 GiB of the real card's ~13.17 GiB GTT total -- and
// must never be assigned to FitTerms.slot_pool: fit_context's budget only
// ever sees the device figure. kDeviceSlotPool below mirrors the probe
// figure at this fixture's scale.
constexpr uint64_t kDeviceSlotPool = 118111601;  // ~0.11 GiB, round(0.11 * 2^30)

// §7.0.2w quality gate: for the same VRAM budget and the same model shape,
// switching --paged-kv precision must widen the admitted context by the
// amount the cost model (kv_block_bytes_from_bits) predicts. The port
// geometry below is round numbers, not a specific artifact's shape (same
// convention as kNumExpert/kPerExpertBytes/kMoeLayers above): kKvPortLayers
// layers, two ports (key_cache, value_cache) per layer, kKvPortElems
// elements per port per kv_block_tokens-token page -- the same (bitwidth,
// count) shape backend_ov.cpp's port walk feeds kv_block_bytes_from_bits,
// reproduced here device-free. kKvPortElems is even (and a multiple of 4)
// so halving under i4 stays exact -- no ceiling noise to separate from the
// precision effect under test.
constexpr int      kKvPortLayers = 48;
constexpr uint64_t kKvPortElems  = 1920;

uint64_t kv_bytes_token_for_precision(int key_bits, int value_bits) {
    std::vector<std::pair<uint64_t, uint64_t>> ports;
    ports.reserve(2 * static_cast<size_t>(kKvPortLayers));
    for (int layer = 0; layer < kKvPortLayers; ++layer) {
        ports.emplace_back(static_cast<uint64_t>(key_bits), kKvPortElems);
        ports.emplace_back(static_cast<uint64_t>(value_bits), kKvPortElems);
    }
    // Mirrors backend_ov.cpp: kv_bytes_token_ = kv_block_bytes / kv_block_tokens_.
    return kv_block_bytes_from_bits(ports) / static_cast<uint64_t>(kKvBlockTokens);
}

}  // namespace

// ---------------------------------------------------------------- fit_context

// The fixture reproduces the pre-M7 defect: with slot_pool left at 0 (today's
// arithmetic), fit_context must land on exactly the number DESIGN §7.0.2s
// printed. If this fails, the fixture itself is wrong and every other case
// below is measuring nothing.
TEST(fit_2026_09_01_zero_slot_reproduces_the_defect) {
    const FitResult r = fit_context(base_terms());
    CHECK_EQ(r.max_ctx, kDefectiveMaxCtx);
}

// The failing-first case (design §4): with the honest DEVICE slot term
// included (the probe/forced figure -- see the two-ledger comment above),
// max ctx must NOT be the number a card that cannot serve it was promised.
// The drop is small on purpose: charging only the true device-resident
// working set barely touches the budget, which is the whole point of
// separating it from the much larger host estimate (see
// fit_device_and_host_ledgers_are_independent below).
TEST(fit_2026_09_01_honest_slot_pool_lowers_max_ctx) {
    FitTerms t    = base_terms();
    t.slot_pool   = kDeviceSlotPool;
    CHECK(t.slot_pool > 0);

    const FitResult r = fit_context(t);
    // 1,161,808: (kTotal - kWeights - kDeviceSlotPool - kActivation - kSlab -
    // kMargin) / kKvBytesToken, floored to a kKvBlockTokens page.
    CHECK_EQ(r.max_ctx, 1161808LL);
    CHECK(r.max_ctx < kDefectiveMaxCtx);
    CHECK(r.max_ctx > 0);  // still a servable configuration, just an honest one
    CHECK(r.admissible);
}

// A client asking for the old, dishonest number must be refused once the
// slot term is in the budget: the honest max ctx no longer reaches it.
TEST(fit_requested_1172016_is_inadmissible_once_honest) {
    FitTerms t  = base_terms();
    t.slot_pool = kDeviceSlotPool;
    const FitResult r = fit_context(t);
    CHECK(kDefectiveMaxCtx > r.max_ctx);
}

// The essence of defect 3: charging the device budget with the host-side
// estimate (what this rework removes) collapses max ctx far more than
// charging the true device figure does. Two ledgers, not one, and
// fit_context's budget must only ever see the device one.
TEST(fit_device_and_host_ledgers_are_independent) {
    const uint64_t host_estimate =
        expert_slot_bytes(kNumExpert, kRatioPct, kPerExpertBytes, kMoeLayers);
    CHECK(host_estimate > kDeviceSlotPool * 20);  // host is >> device, as measured

    FitTerms device_only    = base_terms();
    device_only.slot_pool   = kDeviceSlotPool;
    const FitResult honest  = fit_context(device_only);

    FitTerms host_charged    = base_terms();
    host_charged.slot_pool   = kDeviceSlotPool + host_estimate;  // the mistake, reproduced
    const FitResult mistaken = fit_context(host_charged);

    CHECK(mistaken.max_ctx < honest.max_ctx);
    // Substantially smaller, not just technically smaller -- under 70% of
    // the honest figure (measured: 713,776 against 1,161,808, ~61%).
    CHECK(mistaken.max_ctx * 10 < honest.max_ctx * 7);
}

TEST(fit_context_floors_to_a_page_multiple) {
    FitTerms t = base_terms();
    // Nudge the budget so the raw division is not already page-aligned.
    t.margin += 7;
    const FitResult r = fit_context(t);
    CHECK_EQ(r.max_ctx % kKvBlockTokens, 0);
    CHECK(r.max_ctx <= kDefectiveMaxCtx);
}

TEST(fit_context_lanes_two_roughly_halves) {
    FitTerms t1 = base_terms();
    t1.slot_pool = kDeviceSlotPool;
    const FitResult one = fit_context(t1);

    FitTerms t2 = t1;
    t2.lanes    = 2;
    const FitResult two = fit_context(t2);

    // Slab and KV are per lane, activations and slot_pool are not (design
    // §1 "lanes > 1"): two lanes pay slab twice and split what remains of
    // the KV budget two ways, so the per-lane max ctx is noticeably less
    // than half of one lane's, never more than half.
    CHECK(two.max_ctx > 0);
    CHECK(two.max_ctx <= one.max_ctx / 2);
    CHECK(two.max_ctx > one.max_ctx / 4);  // sane order of magnitude, not a collapse to floor
}

TEST(fit_context_budget_non_positive_refuses) {
    FitTerms t = base_terms();
    t.weights  = t.total;  // consumes everything before activations or KV are even considered
    const FitResult r = fit_context(t);
    CHECK_EQ(r.max_ctx, 0LL);
    CHECK(!r.admissible);
}

TEST(fit_context_below_floor_is_inadmissible) {
    FitTerms t     = base_terms();
    t.n_ctx_floor  = 2000000;  // above what this budget can ever reach
    const FitResult r = fit_context(t);
    CHECK(r.max_ctx > 0);      // it did fit something...
    CHECK(!r.admissible);      // ...just not enough to be worth serving
}

// The claim the next card window will measure (§7.0.2w): at least 15% more
// context at --paged-kv u8:i4 than at u8, and strictly more again at u4 than
// at u8:i4, for the identical VRAM budget and model shape -- driven only by
// the cost model's per-token byte figure, never by touching the budget
// terms. Written RED first: feeding the fit pass the u8 cost for all three
// "precisions" (as if --paged-kv never reached the cost model) fails both
// CHECKs below (the >= 115% one and the u4 > u8:i4 one) -- exactly the "no
// precision effect" defect this test exists to catch. Note what the 15%
// threshold does and does not verify: the fixture's u8/u8:i4 byte ratio is
// 4/3 by construction (the layer and element counts cancel), so any bar
// under 33% passes here; the bar is the milestone row's figure, and it is
// the ratio test below, not this threshold, that checks scaling.
TEST(fit_context_kv_precision_widens_admitted_ctx_per_cost_model) {
    const uint64_t kv_u8   = kv_bytes_token_for_precision(8, 8);
    const uint64_t kv_u8i4 = kv_bytes_token_for_precision(8, 4);
    const uint64_t kv_u4   = kv_bytes_token_for_precision(4, 4);

    FitTerms t   = base_terms();
    t.slot_pool  = kDeviceSlotPool;  // the honest device ledger, as elsewhere in this file

    t.kv_bytes_token       = kv_u8;
    const FitResult r_u8   = fit_context(t);
    t.kv_bytes_token       = kv_u8i4;
    const FitResult r_u8i4 = fit_context(t);
    t.kv_bytes_token       = kv_u4;
    const FitResult r_u4   = fit_context(t);

    // Page multiples: fit_context's own floor rule, checked explicitly for
    // all three precisions, not just assumed from the u8 fixtures above.
    CHECK_EQ(r_u8.max_ctx % kKvBlockTokens, 0LL);
    CHECK_EQ(r_u8i4.max_ctx % kKvBlockTokens, 0LL);
    CHECK_EQ(r_u4.max_ctx % kKvBlockTokens, 0LL);

    CHECK(r_u8i4.max_ctx * 100 >= r_u8.max_ctx * 115);
    CHECK(r_u4.max_ctx > r_u8i4.max_ctx);
}

// Cross-check via an independent computation: budget is identical across the
// three runs above (only kv_bytes_token changes), so the *unfloored*
// division scales exactly with the cost model's byte ratio; flooring both
// sides to a page can only move the adopted figure by less than one page
// from that prediction. Written RED first with the same u8-for-everything
// bug as above (kv_u8i4 and kv_u4 both fed the (8, 8) cost): the guard
// `CHECK(kv_u4 < kv_u8i4)` caught it immediately, since the cost model's own
// ladder collapses to a tie before fit_context is even asked. The cross-check
// reuses fit.h's budget expression, so it is independent of the code under
// test only in the ratio step, not in the budget term.
TEST(fit_context_kv_precision_ratio_matches_cost_model_byte_ratio) {
    const uint64_t kv_u8i4 = kv_bytes_token_for_precision(8, 4);
    const uint64_t kv_u4   = kv_bytes_token_for_precision(4, 4);
    CHECK(kv_u4 < kv_u8i4);  // the cost model's own ladder must be strict

    FitTerms t  = base_terms();
    t.slot_pool = kDeviceSlotPool;

    const long long fixed =
        static_cast<long long>(t.weights) + static_cast<long long>(t.drafters) +
        static_cast<long long>(t.slot_pool) + static_cast<long long>(t.activations) +
        static_cast<long long>(t.margin) +
        static_cast<long long>(t.lanes) * static_cast<long long>(t.slab_per_lane);
    const long long budget = static_cast<long long>(t.total) - fixed;

    t.kv_bytes_token         = kv_u8i4;
    const FitResult r_u8i4   = fit_context(t);
    t.kv_bytes_token         = kv_u4;
    const FitResult r_u4     = fit_context(t);
    CHECK_EQ(r_u8i4.max_ctx % kKvBlockTokens, 0LL);
    CHECK_EQ(r_u4.max_ctx % kKvBlockTokens, 0LL);

    const long long raw_u8i4 = budget / static_cast<long long>(kv_u8i4);
    const long long predicted_u4 =
        ((raw_u8i4 * static_cast<long long>(kv_u8i4) / static_cast<long long>(kv_u4)) /
         kKvBlockTokens) * kKvBlockTokens;
    CHECK(std::llabs(predicted_u4 - r_u4.max_ctx) <= kKvBlockTokens);
}

// -------------------------------------------------------------- expert_slot_bytes

// slots(m) = num_expert * (100 - ratio) / 100 (design §2): ratio is the
// share kept OFF the card, so ratio 0 means nothing is offloaded and every
// expert still gets a slot (100% of num_expert) -- Phase B in the backend
// skips this function entirely when offload_ratio_ is 0 (there is nothing to
// reserve), but the arithmetic itself, asked directly, has no such gate.
// ratio 100 is the only ratio at which the formula's own slot count is zero.
TEST(expert_slot_bytes_full_ratio_is_zero) {
    CHECK_EQ(expert_slot_bytes(kNumExpert, 100, kPerExpertBytes, kMoeLayers), 0ull);
}

TEST(expert_slot_bytes_zero_ratio_slots_every_expert) {
    const uint64_t want = static_cast<uint64_t>(kNumExpert) * kPerExpertBytes *
                          static_cast<uint64_t>(kMoeLayers);
    CHECK_EQ(expert_slot_bytes(kNumExpert, 0, kPerExpertBytes, kMoeLayers), want);
}

TEST(expert_slot_bytes_guards) {
    CHECK_EQ(expert_slot_bytes(0, kRatioPct, kPerExpertBytes, kMoeLayers), 0ull);
    CHECK_EQ(expert_slot_bytes(kNumExpert, kRatioPct, 0, kMoeLayers), 0ull);
    CHECK_EQ(expert_slot_bytes(kNumExpert, kRatioPct, kPerExpertBytes, 0), 0ull);
}

TEST(expert_slot_bytes_ceiling_rounds_up) {
    // 10 experts, ratio 95 -> 10*5/100 = 0.5 experts kept as slots; the
    // design requires ceiling (over-reserve, never under), so 1 slot, not 0.
    const uint64_t got = expert_slot_bytes(10, 95, 1024, 1);
    CHECK_EQ(got, 1024ull);
}

TEST(expert_slot_bytes_exact_division_stays_exact) {
    // 128 experts, ratio 20 -> 128*80/100 = 102.4 -> ceil 103 (kNumExpert
    // case used throughout this file).
    const uint64_t slots = 103;
    const uint64_t want  = slots * kPerExpertBytes * static_cast<uint64_t>(kMoeLayers);
    CHECK_EQ(expert_slot_bytes(kNumExpert, kRatioPct, kPerExpertBytes, kMoeLayers), want);
}

// --------------------------------------------------------- kv_block_bytes_from_bits

// M8 bug 1: the pre-fix arithmetic summed `Type::size()` (bytes, ceiled) per
// port, so an i4 port -- 4 bits/element -- counted as a full byte, same as
// u8. A u8/u8 pool and a u8/i4 pool with identical element counts must NOT
// land on the same byte figure once the value side is i4; this is the red
// case the pre-fix code could not pass (it summed `size()` = 1 byte for
// every u8 AND every i4 port alike).
TEST(kv_block_bytes_from_bits_i4_port_halves_against_u8) {
    constexpr uint64_t kElems = 128;  // an even count, as any real KV layout is
    const uint64_t u8_u8 = kv_block_bytes_from_bits({{8, kElems}, {8, kElems}});
    const uint64_t u8_i4 = kv_block_bytes_from_bits({{8, kElems}, {4, kElems}});
    // u8 side unchanged, i4 side exactly half of what a u8 side of the same
    // element count would have cost.
    CHECK_EQ(u8_u8, 2 * kElems);
    CHECK_EQ(u8_i4, kElems + kElems / 2);
    CHECK(u8_i4 < u8_u8);
}

TEST(kv_block_bytes_from_bits_matches_type_size_for_byte_aligned_types) {
    // f16 (16 bits) and u8 (8 bits) are already byte-aligned, so bitwidth
    // arithmetic must reproduce plain `Type::size()` summation exactly --
    // this milestone changes nothing for the types that were never wrong.
    constexpr uint64_t kElems = 64;
    CHECK_EQ(kv_block_bytes_from_bits({{16, kElems}}), 2 * kElems);
    CHECK_EQ(kv_block_bytes_from_bits({{8, kElems}}), kElems);
}

// F7 (review 2026-09-02): each port is a physically separate device
// allocation, so it must be ceiled on its OWN bit total, not summed into one
// aggregate and floored once -- the first version of this fix did exactly
// that, and could let one port's leftover bits complete another port's byte
// in the aggregate, undercounting the real allocation. Red case: a single
// i4 port with an odd element count (4 bits x 3 = 12 bits) needs 2 bytes on
// its own; the pre-fix arithmetic (floor(12/8)) landed on 1.
TEST(kv_block_bytes_from_bits_ceils_per_port_odd_i4_count) {
    CHECK_EQ(kv_block_bytes_from_bits({{4, 3}}), 2ull);  // ceil(12/8) = 2, not floor(12/8) = 1
}

// Two odd-by-themselves i4 ports must each ceil independently: 2 + 1 = 3,
// not the pre-fix aggregate answer ((12 + 4) bits / 8 = 2) that let the
// second port's leftover 4 bits complete the first port's byte in the sum.
TEST(kv_block_bytes_from_bits_does_not_let_ports_borrow_bits_from_each_other) {
    CHECK_EQ(kv_block_bytes_from_bits({{4, 3}, {4, 1}}), 3ull);
}

TEST(kv_block_bytes_from_bits_empty_is_zero) {
    CHECK_EQ(kv_block_bytes_from_bits({}), 0ull);
}

// ---------------------------------------------- kv_cost_bitwidth (RETRACTED)

// kv_cost_bitwidth -- a requested-precision override for the cost model --
// is deleted (see the retraction note in fit.h, next to
// kv_block_bytes_from_bits, for the full two-round history). A third
// on-card measurement found the SECOND version still wrong: --paged-kv u4
// kept printing 3.2 KiB/token even after the override was narrowed to fire
// only on an actual 8-bit-typed port. The arithmetic that found why: solve
// KV_pure/4 + fixed = 3.2 against KV_pure + fixed = 11.3 (the u8 baseline)
// -> KV_pure ~= 10.8 KiB/token, fixed ~= 0.5; WITHOUT any override the
// predicted figure is KV_pure/2 + fixed ~= 5.9, inside the expected
// ~5.65-5.9 band. This plugin generation packs a 4-bit request into the
// port's SHAPE (the element count is already halved), not only its type --
// so kv_block_bytes_from_bits, fed the compiled port's own (bitwidth,
// count) with no override at all, was correct from the start. The two test
// cases below replace the deleted kv_cost_bitwidth_* tests: the case that
// used to assert the packed-4-in-8 port charges 4 bits now asserts it
// charges 8 (its own type, unconditionally); the separate "does not
// override a 16/32-bit port" cases are now moot -- there is no override
// left to exempt them from -- and fold into the same "no override, ever"
// point.
TEST(kv_block_bytes_from_bits_charges_the_ports_own_type_for_a_packed_four_bit_request) {
    // The exact scenario the retracted override targeted: a 4-bit (u4/i4)
    // request that this plugin generation packs into an 8-bit-TYPED port by
    // halving the port's own element count. `kPackedCount` stands in for
    // that already-halved shape. Correct: kPackedCount bytes (the port's
    // real 8-bit type, unconditionally) -- NOT kPackedCount/2, which is what
    // the retracted override would have additionally charged, double-
    // halving what the shape already halved once.
    constexpr uint64_t kPackedCount = 100;
    const uint64_t      bytes        = kv_block_bytes_from_bits({{8, kPackedCount}});
    CHECK_EQ(bytes, kPackedCount);        // 100: the port's own type, no override
    CHECK(bytes != kPackedCount / 2);     // != 50: the retracted double-halving bug
}

TEST(kv_block_bytes_from_bits_charges_the_ports_own_type_unconditionally) {
    // No requested-precision input exists anywhere in this function's
    // signature any more -- every port, whatever precision it happens to
    // hold, is charged by its own compiled (bitwidth, count) alone.
    CHECK_EQ(kv_block_bytes_from_bits({{8, 100}}), 100ull);
    CHECK_EQ(kv_block_bytes_from_bits({{16, 100}}), 200ull);
    CHECK_EQ(kv_block_bytes_from_bits({{32, 100}}), 400ull);
    CHECK_EQ(kv_block_bytes_from_bits({{4, 100}}), 50ull);  // an honest 4-bit port, if one ever exists
}

// ------------------------------------------------------------------- shrink_n_ctx

TEST(shrink_n_ctx_strictly_decreases_and_terminates) {
    long long               n_ctx = kDefectiveMaxCtx;  // page-aligned (see the fixture comment)
    std::vector<long long>  trace{n_ctx};
    // A fixed overshoot, as the replay loop would see it if the driver kept
    // reporting the same excess after each correction -- the pathological
    // case for "does this ever stop". Cap mirrors the load path's own: 4
    // passes, then the caller refuses. The loop bound (`pass < 4`) is not
    // itself the property under test -- a stray `CHECK(passes <= 4)` here
    // would just restate that loop condition and could never fail. What the
    // test asserts instead is the trace: every value strictly below the one
    // before it, checked over the whole recorded sequence below, which is
    // termination proved by a test rather than a bound taken on faith.
    const uint64_t overshoot = 2ull * kGiB;
    for (int pass = 0; pass < 4; ++pass) {
        const long long next = shrink_n_ctx(n_ctx, overshoot, kKvBytesToken, /*lanes=*/1,
                                            kKvBlockTokens);
        CHECK(next < n_ctx);                   // strictly decreasing, every pass
        CHECK_EQ(next % kKvBlockTokens, 0LL);  // stays page-aligned
        n_ctx = next;
        trace.push_back(n_ctx);
        if (n_ctx <= 0) break;
    }
    for (size_t i = 1; i < trace.size(); ++i) {
        CHECK(trace[i] < trace[i - 1]);
    }
}

TEST(shrink_n_ctx_zero_overshoot_is_a_no_op) {
    CHECK_EQ(shrink_n_ctx(kDefectiveMaxCtx, 0, kKvBytesToken, 1, kKvBlockTokens),
             kDefectiveMaxCtx);
}

TEST(shrink_n_ctx_never_goes_negative) {
    const long long r = shrink_n_ctx(32, 1ull << 40, kKvBytesToken, 1, kKvBlockTokens);
    CHECK(r >= 0);
    CHECK(r < 32);
}

// ------------------------------------------------------------- pool_sizing

// The M7 defect, reproduced device-free (measured 2026-09-03, B60/24 GiB
// card -- 22.71 GiB usable, qwen38 27B dense int4, u8 paged KV at ~36.2
// KiB/token / ~579.6 KiB per 16-token page, --prefix-cache-mib 8192, one
// lane, explicit --n-ctx 32768: pass 1 residency 22.53 GiB against a 22.46
// GiB ceiling). `kBudgetRemainingM7Defect` is NOT read off the card -- like
// kTotal above, it is engineered so that feeding it through pool_sizing
// reproduces the measured pass-1 pool to the page.
//
// Round-3 review, finding 5 -- a correction to round 2's own record: the
// retry log line prints `(%zu of %zu pages)` = (spare, blocks), and round
// 2's fixture (8426 blocks / 6375 spare, a one-page trim) had that right
// as arithmetic but wrong as a reproduction -- the real cell needed a much
// deeper trim than one page. Cell f1b (this run): pass 1 was 6,503 of
// 8,554 pages, trimmed 129 to the accepted 6,374 of 8,425. Cell f2b (same
// request, DFlash resident): pass 1 was 4,700 of 6,751, trimmed 129 to
// 4,571 of 6,622. This fixture pins f1b's PASS-1 split, 8,554 blocks /
// 6,503 spare -- not DESIGN §7.0.2t's itemization, that section carries no
// such numbers; this is tonight's own measurement.
// `kWantedSpareUnbounded` stands in for the prefix cache's own entries
// count (host budget / la_row_bytes_) whenever --prefix-cache-mib is
// configured at all: it scales with the HOST budget, never the device one,
// so it is essentially always larger than whatever room the device budget
// has left over -- as it was here.
constexpr uint64_t kKvBlockBytesM7Defect      = 593920;      // ~579.6 KiB/block, order of 36.2 KiB/token * 16
// backend_ov.cpp's per_lane_blocks formula: (n_ctx + drafts_max + kv_block_tokens - 1)
// / kv_block_tokens + 2, one lane -- 32768 + 3 (drafts_max_) + 15, /16, +2 = 2051.
constexpr size_t   kLiveBlocksM7Defect        = 2051;
constexpr long long kBudgetRemainingM7Defect  = 5080391680;  // ~4.73 GiB, engineered so pool_sizing reproduces f1b's measured pass-1 split (8554 pages, 6503 spare) to the page
constexpr size_t   kWantedSpareUnbounded      = 100000;      // "the prefix cache always wants more"
// The measured order of f1b's pass-1 overshoot (22.53 GiB resident against
// a 22.46 GiB ceiling, ~0.07 GiB): the smallest value whose ceil-to-a-page
// is the record's own 129-page trim (128 full pages is one byte short of
// it), so pool_sizing + explicit_retry_spare_cap reproduce the record's
// accepted 8,425 / 6,374 exactly -- see
// explicit_retry_spare_cap_reproduces_the_f1b_trim below.
constexpr uint64_t kF1bOvershootBytes         = 76300000;

// The root cause, stated as arithmetic: the pool this configuration would
// allocate is NOT sized to the 2051 blocks the 32768-token request needs --
// it is sized to 8554 blocks, with the spare component alone (6503 blocks)
// dwarfing the live one. This is why residency did not track the explicit
// request: with --n-ctx omitted, auto-fit runs the same night adopted
// 155,568 (corrected to 155,376, 0 spare) and, on a second run, 171,488
// (corrected to 171,312, also 0 spare) -- the explicit request's pool was
// never actually being sized from the request either way.
TEST(pool_sizing_2026_09_03_defect_balloons_to_near_the_full_budget) {
    const PoolSizing sizing = pool_sizing(kLiveBlocksM7Defect, kWantedSpareUnbounded,
                                          kBudgetRemainingM7Defect, kKvBlockBytesM7Defect);
    CHECK_EQ(sizing.spare_blocks, 6503ull);
    CHECK_EQ(sizing.blocks, 8554ull);
    CHECK(sizing.spare_blocks > sizing.blocks - sizing.spare_blocks);  // spare dwarfs live
}

// The RED case (task deliverable 2): given that split, the load must NOT
// refuse -- the overshoot has 6503 blocks of spare capacity to give up
// before the live 2051-block request is ever touched, and fit_context (via
// the `wanted > max_ctx` check that runs before this pool is ever sized)
// already proved the live request alone is admissible.
//
// Honestly stated (round-1 review, finding F2): production's rule in this
// branch was unconditional -- the code simply threw every time this pass
// was not accepted, reading neither `over` nor any live/spare split (there
// was no `measured_overshoot` parameter for it to read in the first
// place; "return measured_overshoot" was never production's rule --
// "return true" unconditionally is the honest characterization).
//
// Round-3 review, finding 1 -- a correction to round 2's own record: rerun
// against round 2's file (39 tests) and that characterization, TWO tests
// go red, not one -- this one, and `explicit_overshoot_unmeasured_failure_
// with_spare_must_not_refuse` below (added in round 2; it asserts the same
// `!refuse(spare > 0)` shape against a different `measured_overshoot`
// value, so it fails the same way). Round 2's claim of exactly one was
// true when written -- the second test did not exist in round 1's file --
// and was carried forward without being re-checked once round 2 added it.
// Rerun again against round 3's OWN file (43 tests, after this round's
// additions), a THIRD test also goes red --
// `explicit_retry_decision_refuse_agrees_with_explicit_overshoot_must_
// refuse` below -- for a different, expected reason (see fit.h's comment
// on explicit_overshoot_must_refuse for why). See the round-3 report for
// the verbatim `39 cases run, 2 failed` and `43 cases run, 3 failed`
// output.
TEST(explicit_overshoot_2026_09_03_defect_must_not_refuse_when_spare_can_absorb_it) {
    const PoolSizing sizing = pool_sizing(kLiveBlocksM7Defect, kWantedSpareUnbounded,
                                          kBudgetRemainingM7Defect, kKvBlockBytesM7Defect);
    CHECK(sizing.spare_blocks > 0);
    CHECK(!explicit_overshoot_must_refuse(sizing.spare_blocks));
}

// F1 (HIGH, red-first): a residual overshoot smaller than one KV page must
// still trim at least one page per retry pass. Round 1's retry shrank only
// `budget_remaining` (by `overshoot_accum += over;`) and re-ran pool_sizing
// with the SAME unbounded wanted_spare -- `affordable = budget_remaining /
// kv_block_bytes` is a floor division, so a one-byte overshoot leaves
// `budget_remaining` inside the same page bucket and `blocks` comes out
// byte-identical. Fails against that: `pass2_round1_retry` below asserts
// the defect (no trim at all) is real, then `explicit_retry_spare_cap`
// (this fix) is shown to trim exactly the guaranteed minimum of one page.
TEST(explicit_retry_must_trim_one_page_when_residual_overshoot_is_smaller_than_a_page) {
    constexpr size_t   live         = 100;
    constexpr uint64_t block        = 1000;     // a round KV block size for this fixture
    constexpr long long budget_pass1 = 105600;  // affordable = 105, spare = 5, blocks = 105
    const PoolSizing pass1 =
        pool_sizing(live, /*wanted_spare_blocks=*/10000, budget_pass1, block);
    CHECK_EQ(pass1.blocks, 105ull);
    CHECK_EQ(pass1.spare_blocks, 5ull);

    constexpr uint64_t over = 1;  // smaller than one KV page (block = 1000)
    const long long budget_pass2 = budget_pass1 - static_cast<long long>(over);

    // The defect, reproduced: round 1's retry (pool_sizing alone, budget
    // shrunk by `over`, wanted_spare unchanged) does not trim anything.
    const PoolSizing pass2_round1_retry =
        pool_sizing(live, /*wanted_spare_blocks=*/10000, budget_pass2, block);
    CHECK_EQ(pass2_round1_retry.blocks, pass1.blocks);  // no trim at all -- the defect

    // The fix: cap wanted_spare with explicit_retry_spare_cap first,
    // guaranteeing a real (not just >= 1 page) trim regardless of how
    // small `over` is. pass=0: this is the retry computed for the FIRST
    // failed pass, same as every other call site in this file. Round-4
    // review: the backoff floor's own baseline moved from 1 to 4 pages at
    // pass 0 (auto_fit_backoff_pages(0) == 4 now, was 1 in round 3) --
    // this fixture's analytic trim here (1 page, from `over` = 1 against
    // block = 1000) is smaller than that floor, so the floor is what
    // actually binds, and the guaranteed trim is 4 pages, not 1.
    const size_t cap = explicit_retry_spare_cap(pass1.spare_blocks, over, block, /*pass=*/0);
    const PoolSizing pass2_fixed = pool_sizing(live, cap, budget_pass2, block);
    CHECK(pass2_fixed.blocks < pass1.blocks);
    CHECK_EQ(pass1.blocks - pass2_fixed.blocks, 4ull);  // the pass-0 backoff floor, not 1
}

// Ties the fixture to the actual record: f1b's real overshoot (~0.07 GiB,
// kF1bOvershootBytes) trims exactly 129 pages off the pass-1 split
// (8,554 / 6,503), landing on the record's own accepted pool -- 8,425
// blocks / 6,374 spare, live unchanged at 2,051.
TEST(explicit_retry_spare_cap_reproduces_the_f1b_trim) {
    const PoolSizing pass1 = pool_sizing(kLiveBlocksM7Defect, kWantedSpareUnbounded,
                                         kBudgetRemainingM7Defect, kKvBlockBytesM7Defect);
    CHECK_EQ(pass1.blocks, 8554ull);
    CHECK_EQ(pass1.spare_blocks, 6503ull);

    // pass=0: kept exactly as before -- auto_fit_backoff_pages(0)
    // == 4 is smaller than the 129-page analytic trim this overshoot
    // produces, so the floor never binds and the record's own 129-page
    // trim is unchanged (round-3 review: "its 129-page case must still
    // trim exactly 129 pages on pass 1").
    const size_t cap = explicit_retry_spare_cap(pass1.spare_blocks, kF1bOvershootBytes,
                                                kKvBlockBytesM7Defect, /*pass=*/0);
    CHECK_EQ(pass1.spare_blocks - cap, 129ull);  // the record's own trim count

    const PoolSizing accepted =
        pool_sizing(kLiveBlocksM7Defect, cap, kBudgetRemainingM7Defect, kKvBlockBytesM7Defect);
    CHECK_EQ(accepted.blocks, 8425ull);
    CHECK_EQ(accepted.spare_blocks, 6374ull);
}

// Round-3 review, finding 4: replaces `explicit_retry_trims_spare_not_the_
// live_request`, which could not fail for any implementation --
// `blocks - spare_blocks == live` is an identity of pool_sizing for any
// input, `live` was a constant the test itself supplied, and the
// `shrink_n_ctx` comparison tested `shrink_n_ctx`, not the retry. A
// backend that called `shrink_n_ctx` in the explicit branch would have
// stayed green against it.
//
// The real invariant -- the explicit retry's decision never depends on
// n_ctx or live_blocks -- is now checked against explicit_retry_decision
// (fit.h), the actual per-pass decision backend_ov.cpp calls, not a
// restatement of pool_sizing's arithmetic. What this group does NOT prove:
// that backend_ov.cpp's call site wires this decision's output to
// paged_n_ctx_ correctly (i.e. that nothing downstream re-applies
// shrink_n_ctx on top of it). That is pinned by the on-card log lines --
// the retry logs print the SAME n_ctx pass to pass, and the accepted pool
// line prints the n_ctx the request asked for -- and by reading the call
// site itself, not by a device-free test: a pure function's signature can
// guarantee it cannot compute a new n_ctx, it cannot guarantee the caller
// ignores that guarantee and computes one anyway.
TEST(explicit_retry_decision_never_depends_on_n_ctx_or_live_blocks) {
    // Two different n_ctx-derived live values (2,051 for --n-ctx 32768,
    // one lane; 4,098 for --n-ctx 65536, one lane) with budgets tuned so
    // BOTH land on the same spare_blocks, 6503 -- then explicit_retry_
    // decision, fed that shared spare_blocks, must produce an identical
    // result either way, because live/n_ctx are not among its parameters.
    constexpr size_t live_32k = 2051;
    constexpr size_t live_64k = 4098;
    const long long budget_32k =
        (static_cast<long long>(live_32k) + 6503) * static_cast<long long>(kKvBlockBytesM7Defect);
    const long long budget_64k =
        (static_cast<long long>(live_64k) + 6503) * static_cast<long long>(kKvBlockBytesM7Defect);
    const PoolSizing sizing_32k =
        pool_sizing(live_32k, kWantedSpareUnbounded, budget_32k, kKvBlockBytesM7Defect);
    const PoolSizing sizing_64k =
        pool_sizing(live_64k, kWantedSpareUnbounded, budget_64k, kKvBlockBytesM7Defect);
    CHECK_EQ(sizing_32k.spare_blocks, 6503ull);
    CHECK_EQ(sizing_64k.spare_blocks, 6503ull);
    CHECK(sizing_32k.blocks != sizing_64k.blocks);  // the pools themselves are NOT identical

    const ExplicitRetryDecision d_32k = explicit_retry_decision(
        /*measured_overshoot=*/true, kF1bOvershootBytes, sizing_32k.spare_blocks,
        kKvBlockBytesM7Defect, /*pass=*/0, /*last_pass=*/3);
    const ExplicitRetryDecision d_64k = explicit_retry_decision(
        /*measured_overshoot=*/true, kF1bOvershootBytes, sizing_64k.spare_blocks,
        kKvBlockBytesM7Defect, /*pass=*/0, /*last_pass=*/3);
    CHECK_EQ(d_32k.next_spare_cap, d_64k.next_spare_cap);  // identical despite different live/n_ctx
    CHECK(d_32k.n_ctx_unchanged);
    CHECK(d_64k.n_ctx_unchanged);
}

TEST(explicit_retry_decision_forces_the_last_pass_live_only) {
    // Finding 3: the cap computed for the pass AFTER pass 2 (0-indexed) --
    // i.e. for pass 3, the last one -- is always 0, regardless of
    // over/spare_blocks/measured: the one direct test of whether --n-ctx
    // itself is honourable must be tried before the loop can run out.
    const ExplicitRetryDecision d = explicit_retry_decision(
        /*measured_overshoot=*/true, /*over=*/1, /*spare_blocks=*/6503, kKvBlockBytesM7Defect,
        /*pass=*/2, /*last_pass=*/3);
    CHECK(!d.refuse);
    CHECK_EQ(d.next_spare_cap, 0ull);
    CHECK(d.n_ctx_unchanged);
}

TEST(explicit_retry_decision_refuses_when_the_last_pass_itself_fails) {
    // The last pass failing IS the exhausted-reserve case (DESIGN's
    // corollary): there is no pass 5 to retry into, so this refuses even
    // with spare_blocks > 0 here (a pathological --kv-pool-pages scenario,
    // finding 7, where the test-only cap can leave spare nonzero on any
    // pass) -- there is nowhere left for a computed cap to be used.
    const ExplicitRetryDecision d = explicit_retry_decision(
        /*measured_overshoot=*/true, /*over=*/1, /*spare_blocks=*/6503, kKvBlockBytesM7Defect,
        /*pass=*/3, /*last_pass=*/3);
    CHECK(d.refuse);
    CHECK_EQ(d.next_spare_cap, 0ull);
}

TEST(explicit_retry_decision_refuse_agrees_with_explicit_overshoot_must_refuse) {
    CHECK_EQ(explicit_retry_decision(true, 1, 0, kKvBlockBytesM7Defect, 0, 3).refuse,
             explicit_overshoot_must_refuse(0));
    CHECK_EQ(explicit_retry_decision(true, 1, 6503, kKvBlockBytesM7Defect, 0, 3).refuse,
             explicit_overshoot_must_refuse(6503));
}

// A genuine overshoot (task deliverable 3): once prior replay passes have
// accumulated enough measured overshoot that the remaining budget covers
// only the live request -- spare_room is 0 before wanted_spare is even
// consulted -- the pool_sizing split has nothing left to trim, and refusal
// must still fire. This is the case that must NEVER be silently lowered.
TEST(pool_sizing_budget_exhausted_leaves_no_spare) {
    const long long budget_remaining_exhausted =
        static_cast<long long>(kLiveBlocksM7Defect) * static_cast<long long>(kKvBlockBytesM7Defect);
    const PoolSizing sizing = pool_sizing(kLiveBlocksM7Defect, kWantedSpareUnbounded,
                                          budget_remaining_exhausted, kKvBlockBytesM7Defect);
    CHECK_EQ(sizing.spare_blocks, 0ull);
    CHECK_EQ(sizing.blocks, kLiveBlocksM7Defect);
}

TEST(explicit_overshoot_genuine_when_spare_already_zero_must_refuse) {
    CHECK(explicit_overshoot_must_refuse(/*spare_blocks=*/0));
}

// F4 (MEDIUM, red-first, rule change): a raw allocation exception
// (fragmentation, no residency reading -- `measured_overshoot` false in
// round 1's rule) used to refuse immediately regardless of spare_blocks.
// Round 1's own comment at this call site called that "matching
// backend_ov.cpp's `over == 0` guard" -- backwards: that guard RETRIES on
// an unmeasured failure with a synthesized retreat, it does not refuse;
// only the explicit-n_ctx branch refused outright. Fails against round 1's
// rule (`!measured_overshoot || spare_blocks == 0`, unconditionally true
// whenever measured_overshoot is false) -- run red before the fix lands;
// see the round-2 report for the verbatim red output. `6503` is f1b's own
// measured pass-1 spare (round-3 review, finding 8: round 2 used an
// invented 4691 here, flagged as not a measurement).
TEST(explicit_overshoot_unmeasured_failure_with_spare_must_not_refuse) {
    CHECK(!explicit_overshoot_must_refuse(/*spare_blocks=*/6503));
}

// The refuse rule now reads spare_blocks alone: an unmeasured failure with
// spare already at zero still refuses, same as a measured one -- there is
// nothing left to retreat to either way.
TEST(explicit_overshoot_unmeasured_failure_with_zero_spare_must_refuse) {
    CHECK(explicit_overshoot_must_refuse(/*spare_blocks=*/0));
}

// synthesize_spare_retreat (F4): the retry's retreat when there is no
// residency reading to size a trim from -- halve spare, minimum one page
// trimmed, mirroring explicit_retry_spare_cap's >= 1 page guarantee and the
// auto-fit branch's own `over == 0` synthesis.
TEST(synthesize_spare_retreat_halves_and_strictly_decreases) {
    CHECK_EQ(synthesize_spare_retreat(100), 50ull);
    CHECK_EQ(synthesize_spare_retreat(2), 1ull);   // trim max(1, 1) = 1
    CHECK_EQ(synthesize_spare_retreat(1), 0ull);   // trim max(1, 0) = 1
    CHECK_EQ(synthesize_spare_retreat(0), 0ull);   // nothing to retreat from
}

TEST(synthesize_spare_retreat_reaches_zero_and_terminates) {
    size_t spare      = 6503;
    int    iterations = 0;
    while (spare > 0 && iterations < 32) {
        const size_t next = synthesize_spare_retreat(spare);
        CHECK(next < spare);  // strict decrease every call
        spare = next;
        ++iterations;
    }
    CHECK_EQ(spare, 0ull);  // reaches the refuse condition in finite steps
}

// A small prefix-cache budget (a small wanted_spare) is capped by what it
// asks for, not by the whole remaining device budget -- the mechanism only
// balloons the pool when the cache genuinely wants more than the budget can
// spare (as it usually does, per the defect above).
TEST(pool_sizing_caps_spare_at_what_the_cache_actually_wants) {
    const PoolSizing sizing =
        pool_sizing(kLiveBlocksM7Defect, /*wanted_spare_blocks=*/10, kBudgetRemainingM7Defect,
                   kKvBlockBytesM7Defect);
    CHECK_EQ(sizing.spare_blocks, 10ull);
    CHECK_EQ(sizing.blocks, kLiveBlocksM7Defect + 10);
}

TEST(pool_sizing_no_prefix_cache_is_live_only) {
    const PoolSizing sizing = pool_sizing(kLiveBlocksM7Defect, /*wanted_spare_blocks=*/0,
                                          kBudgetRemainingM7Defect, kKvBlockBytesM7Defect);
    CHECK_EQ(sizing.spare_blocks, 0ull);
    CHECK_EQ(sizing.blocks, kLiveBlocksM7Defect);
}

TEST(pool_sizing_non_positive_budget_is_live_only) {
    const PoolSizing sizing =
        pool_sizing(kLiveBlocksM7Defect, kWantedSpareUnbounded, /*budget_remaining_bytes=*/0,
                   kKvBlockBytesM7Defect);
    CHECK_EQ(sizing.spare_blocks, 0ull);
    CHECK_EQ(sizing.blocks, kLiveBlocksM7Defect);
}

TEST(shrink_n_ctx_reduction_scales_with_lanes) {
    // The same overshoot is freed by a smaller per-lane token reduction when
    // more lanes each give some up (design's replay: "n_ctx = floor_to_page(
    // n_ctx - ceil(over / kv_bytes_token / lanes))").
    const long long n_ctx     = 100000 - (100000 % kKvBlockTokens);
    const uint64_t  overshoot = 4ull * kMiB;
    const long long one_lane  = shrink_n_ctx(n_ctx, overshoot, kKvBytesToken, 1, kKvBlockTokens);
    const long long two_lanes = shrink_n_ctx(n_ctx, overshoot, kKvBytesToken, 2, kKvBlockTokens);
    CHECK(two_lanes >= one_lane);  // two lanes need to give up fewer tokens each
    CHECK(two_lanes < n_ctx);
    CHECK(one_lane < n_ctx);
}

// M9: --prefix-cache-reserve PCT, an option under auto-fit. Reuses the
// section's own defect-reproduction fixture (base_terms() / kDefectiveMaxCtx
// = 1,172,016, kKvBlockTokens = 16) as "what the budget affords" -- the
// fixture's own self-check above already proves fit_context lands there, so
// affordable_pages = kDefectiveMaxCtx / kKvBlockTokens is exact (1,172,016
// is a kKvBlockTokens multiple by construction: fit_context floors to one).
constexpr long long kAffordablePages = kDefectiveMaxCtx / kKvBlockTokens;  // 73,251

// PCT 0 must reproduce fit_context's own unreserved adoption exactly --
// today's behaviour, unchanged.
TEST(prefix_cache_reserve_pct_zero_reproduces_current_adoption_exactly) {
    const FitResult          fit = fit_context(base_terms());
    const PrefixCacheReserve r =
        prefix_cache_reserve(fit.max_ctx / kKvBlockTokens, /*reserve_pct=*/0, kKvBlockTokens,
                             /*n_ctx_floor=*/4096);
    CHECK_EQ(r.adopted_n_ctx, fit.max_ctx);
    CHECK_EQ(r.spare_pages, 0ll);
    CHECK(r.admissible);
}

// PCT 25 yields a smaller adopted depth, with spare >= 25% of the total
// affordable pages -- "at least PCT percent... remain spare", not exactly.
TEST(prefix_cache_reserve_pct_25_yields_smaller_depth_with_at_least_25_pct_spare) {
    const PrefixCacheReserve r =
        prefix_cache_reserve(kAffordablePages, /*reserve_pct=*/25, kKvBlockTokens,
                             /*n_ctx_floor=*/4096);
    CHECK(r.adopted_n_ctx < kDefectiveMaxCtx);
    CHECK(r.admissible);
    // spare_pages * 100 >= 25 * affordable_pages, avoiding floating point.
    CHECK(r.spare_pages * 100 >= 25 * kAffordablePages);
    CHECK_EQ(r.live_pages + r.spare_pages, kAffordablePages);
}

// The adopted n_ctx is live_pages * kv_block_tokens by construction, so it
// floors to a page multiple for any PCT in range -- checked at a PCT that
// does not divide the affordable-page count evenly (73,251 * 63 / 100 is
// not an integer).
TEST(prefix_cache_reserve_adopted_n_ctx_floors_to_a_page_multiple) {
    for (const int pct : {1, 7, 25, 50, 63, 89, 90}) {
        const PrefixCacheReserve r =
            prefix_cache_reserve(kAffordablePages, pct, kKvBlockTokens, /*n_ctx_floor=*/4096);
        CHECK_EQ(r.adopted_n_ctx % kKvBlockTokens, 0ll);
    }
}

// PCT 90 at this fixture's scale still clears the 4096 floor -- the reserve
// does not refuse just because it is large, only when it pushes the
// adopted depth below the floor (the next test).
TEST(prefix_cache_reserve_pct_90_clears_the_floor_at_this_scale) {
    const PrefixCacheReserve r =
        prefix_cache_reserve(kAffordablePages, /*reserve_pct=*/90, kKvBlockTokens,
                             /*n_ctx_floor=*/4096);
    CHECK(r.admissible);
    CHECK(r.adopted_n_ctx >= 4096);
    CHECK(r.adopted_n_ctx < kDefectiveMaxCtx);
}

// The floor rule, stated and tested: a PCT that would push the adopted
// depth below n_ctx_floor REFUSES (admissible = false) rather than
// silently floating the adopted depth back up to the floor -- consistent
// with fit_context's own admissible rule and with DESIGN's "never silently
// lowered/raised" stance. 500 affordable pages at 16 tok/page = 8,000
// tokens; PCT 90 leaves floor(500 * 10 / 100) = 50 pages = 800 tokens,
// below the 4096 floor.
TEST(prefix_cache_reserve_pct_90_below_the_floor_refuses) {
    const PrefixCacheReserve r =
        prefix_cache_reserve(/*affordable_pages=*/500, /*reserve_pct=*/90, kKvBlockTokens,
                             /*n_ctx_floor=*/4096);
    CHECK_EQ(r.adopted_n_ctx, 800ll);
    CHECK(!r.admissible);
}

// reserve_pct is clamped to [0, 90] inside the pure function too (defence
// in depth -- config.cpp already refuses out-of-range PCT before this ever
// runs), so a value above 90 cannot silently reserve more than the
// documented maximum.
TEST(prefix_cache_reserve_pct_clamps_above_90) {
    const PrefixCacheReserve at_90 =
        prefix_cache_reserve(kAffordablePages, 90, kKvBlockTokens, 4096);
    const PrefixCacheReserve above_90 =
        prefix_cache_reserve(kAffordablePages, 200, kKvBlockTokens, 4096);
    CHECK_EQ(at_90.adopted_n_ctx, above_90.adopted_n_ctx);
}

// H2 (round-2 review, HIGH): reserve.spare_pages is always > 0 for any
// PCT in [1, 90] with affordable_pages > 0 (proved above), so gating a
// warning on "spare_pages > 0" can never fire -- the real question is
// whether the pool actually ACCEPTED at allocation time fell short of
// what the reserve asked it to hold back.
TEST(prefix_cache_reserve_shortfall_is_never_gated_by_spare_pages_alone) {
    // spare_pages > 0 always holds for this fixture (25% of 73,251 pages);
    // demonstrating the bug the old `!reserve_honoured` gate had: it is
    // NOT a usable proxy for "did the accepted pool meet the ask".
    const PrefixCacheReserve r =
        prefix_cache_reserve(kAffordablePages, /*reserve_pct=*/25, kKvBlockTokens, 4096);
    CHECK(r.spare_pages > 0);
}

// True: the accepted pool holds fewer spare pages than the reserve asked
// for -- the real alarm (a memory-pressure retry, or --kv-pool-pages,
// trimmed the reserve after it was computed).
TEST(prefix_cache_reserve_shortfall_true_when_accepted_spare_is_below_the_ask) {
    CHECK(prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/10, /*reserve_spare_pages=*/18313));
    CHECK(prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/0, /*reserve_spare_pages=*/1));
}

// False: the accepted pool met or exceeded the ask -- the reserve was
// actually honoured, nothing to warn about.
TEST(prefix_cache_reserve_shortfall_false_when_accepted_spare_meets_or_exceeds_the_ask) {
    CHECK(!prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/18313,
                                          /*reserve_spare_pages=*/18313));
    CHECK(!prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/20000,
                                          /*reserve_spare_pages=*/18313));
}

// False: no reserve was ever requested (reserve_spare_pages <= 0) -- the
// pre-existing zero-spare warning (gated on `prefix_cache_ != nullptr`
// alone in backend_ov.cpp, not on this function) covers that case
// instead, so this function must stay quiet rather than double-warn.
TEST(prefix_cache_reserve_shortfall_false_when_nothing_was_reserved) {
    CHECK(!prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/0, /*reserve_spare_pages=*/0));
    CHECK(!prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/5, /*reserve_spare_pages=*/0));
}

// ------------------------------------------------------------------
// Round 3, defect 2: reserve_ask_pool_pages -- unit conversion between
// PrefixCacheReserve::spare_pages (per-lane) and spare_blocks (pool-wide,
// net of the per-lane guard overhead).
// ------------------------------------------------------------------

// The review finding's own numbers, RED before reserve_ask_pool_pages
// exists: 1 lane, drafts_max 0 (guard = 2 + ceil(0/16) = 2 pages), a
// reserve that asked 18,313 per-lane KV pages. Compared directly (the
// round-2 H2 call site), 18,313 vs an accepted 18,311 reads as a
// 2-page shortfall on every load -- a false alarm. Converted to pool
// units first, the ask is 18,313 - 2 = 18,311, matching the accepted
// figure exactly: no shortfall.
TEST(reserve_ask_pool_pages_removes_the_per_lane_guard_overhead) {
    const long long ask = reserve_ask_pool_pages(/*reserve_spare_pages=*/18313, /*lanes=*/1,
                                                 /*drafts_max=*/0, /*kv_block_tokens=*/16);
    CHECK_EQ(ask, 18311ll);
    CHECK(!prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/18311, ask));
    // The pre-fix comparison (raw spare_pages, no conversion) would have
    // flagged this as a shortfall -- demonstrating why the conversion is
    // load-bearing, not cosmetic.
    CHECK(prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/18311,
                                         /*reserve_spare_pages=*/18313));
}

// Two lanes: the per-lane ask must scale by `lanes` to become pool-wide,
// net of `lanes` lanes' worth of guard pages -- the round-2 fix's own
// bug on more than one lane (it under-detected a real shortfall by a
// factor of `lanes`, since it never multiplied by lanes at all).
TEST(reserve_ask_pool_pages_scales_with_lanes) {
    // guard_per_lane = 2 + ceil(3/16) = 2 + 1 = 3; ask = 1000*2 - 3*2 = 1994.
    const long long ask = reserve_ask_pool_pages(/*reserve_spare_pages=*/1000, /*lanes=*/2,
                                                 /*drafts_max=*/3, /*kv_block_tokens=*/16);
    CHECK_EQ(ask, 1994ll);
}

// A very small reserve can be entirely consumed by the guard overhead --
// clamped at 0 rather than going negative (a negative "ask" is not
// meaningful, and prefix_cache_reserve_shortfall's own `> 0` guard would
// otherwise need to special-case it).
TEST(reserve_ask_pool_pages_clamps_at_zero) {
    // guard_per_lane = 2 + ceil(100/16) = 2 + 7 = 9; 1 - 9 would be -8.
    const long long ask = reserve_ask_pool_pages(/*reserve_spare_pages=*/1, /*lanes=*/1,
                                                 /*drafts_max=*/100, /*kv_block_tokens=*/16);
    CHECK_EQ(ask, 0ll);
    CHECK(!prefix_cache_reserve_shortfall(/*accepted_spare_blocks=*/0, ask));
}

// ------------------------------------------------------------------
// Round 3, defect 1: the auto-fit correction must shrink the POOL
// TOTAL, not just live pages, or a driver allocation granule wider than
// one KV page never converges within the replay loop's four passes.
// ------------------------------------------------------------------

// Simulates the driver rounding a requested pool (in pages) up to the
// next multiple of `granule` -- standing in for the measured defect: the
// driver's own allocation granule can be wider than one KV page, so a
// pool that shrinks by less than one granule between passes reports the
// SAME residency every time.
size_t simulate_driver_pages(size_t pool_pages, size_t granule) {
    if (granule <= 1) return pool_pages;
    return ((pool_pages + granule - 1) / granule) * granule;
}

constexpr size_t kConvergeWantedSpare     = 1000000000;  // "the cache always wants more"
constexpr size_t kConvergePoolMultiplier  = 2000;         // pool0 = kConvergePoolMultiplier * granule

// Drives the SAME sequence backend_ov.cpp's Phase E loop runs (pool_
// sizing, then auto_fit_trim on overshoot), purely in pool-page units:
// lanes = 1 and kv_block_tokens = 1 so tokens, per-lane pages and
// pool-wide pages all coincide -- the token<->page conversion is
// auto_fit_trim's own concern (exercised for real by backend_ov.cpp),
// this isolates the live/spare split and the backoff floor. `budget_
// remaining` shrinks pass to pass by the accumulated measured overshoot,
// mirroring backend_ov.cpp's own `overshoot_accum` -- this is what makes
// pool_sizing's spare_room GROW as live shrinks (the actual mechanism
// the pre-fix defect exploited: affordable moves toward live only
// slightly, live moves down more, so spare_room = affordable - live
// grows by roughly what live lost).
//
// Round-4 review correction: the round-3 version of this fixture used a
// FIXED pool0 (110000) for every granule, which is an exact multiple of
// some tested granules and not others -- for granule 32 specifically,
// 110000 is NOT a multiple of 32, so the fixture's starting pool
// happened to sit only 16 pages above a rounding boundary rather than at
// the worst-case 32 pages, and convergence at pass 3 (21 cumulative
// pages trimmed: the OLD schedule's 1+4+16) looked like it proved "up to
// 32 pages" when it had only exercised a much easier case. The TRUE
// worst case for a driver that rounds up to the next multiple of
// `granule` is a pool that starts EXACTLY AT a granule boundary: from
// there, no reduction smaller than a full `granule` pages can move the
// rounded value AT ALL (ceil((k*G - d)/G)*G stays k*G for every 0 < d <
// G), so the ceiling must be crossed by one clean granule-wide step.
// `pool0 = kConvergePoolMultiplier * granule` is an exact multiple of
// `granule` BY CONSTRUCTION, for every granule tested -- this is that
// worst case, not a fixture that happens to converge.
int converge_passes(size_t granule, bool protect_spare) {
    const size_t     g     = std::max<size_t>(1, granule);
    const size_t     pool0 = kConvergePoolMultiplier * g;
    const size_t     live0 = pool0 / 2;  // the other half is realized as spare via pool_sizing below
    long long        n_ctx = static_cast<long long>(live0);
    size_t           spare_cap = static_cast<size_t>(-1);
    long long        overshoot_accum = 0;
    const long long  budget = static_cast<long long>(pool0);
    // pool0 is an exact multiple of g, so simulate_driver_pages(pool0, g)
    // == pool0 -- ceiling is one page short of that, the worst-case bound
    // derived above.
    const size_t ceiling = simulate_driver_pages(pool0, g) - 1;

    for (int pass = 0; pass < 4; ++pass) {
        const size_t     live_blocks  = static_cast<size_t>(n_ctx);
        const size_t     capped_spare = std::min(kConvergeWantedSpare, spare_cap);
        const long long  budget_remaining = budget - overshoot_accum;
        const PoolSizing sizing =
            pool_sizing(live_blocks, capped_spare, budget_remaining, /*kv_block_bytes=*/1);
        const size_t observed = simulate_driver_pages(sizing.blocks, g);
        if (observed <= ceiling) return pass;

        const uint64_t over = observed - ceiling;
        overshoot_accum += static_cast<long long>(over);
        const AutoFitTrim trim =
            auto_fit_trim(n_ctx, sizing.spare_blocks, over, /*kv_block_bytes=*/1, /*lanes=*/1,
                         /*kv_block_tokens=*/1, /*n_ctx_floor=*/1, pass, protect_spare);
        if (trim.refuse) return -1;
        n_ctx     = trim.next_n_ctx;
        spare_cap = trim.next_spare_cap;
    }
    return -1;
}

// RED before auto_fit_trim exists (compile error). Once implemented: the
// HONEST bound (round-4 review correction of the round-3 claim) is that
// this loop's four passes can only ever apply the backoff floors from
// the first THREE failed passes before the fourth (last) check runs --
// the trim computed on the pass that fails that fourth check is never
// allocated or re-checked. With the corrected schedule (4, 16, 64, 256
// for passes 0..3), that is 4 + 16 + 64 = 84 pool pages, cumulative,
// available by the time the last check runs -- so a driver rounding
// granule up to and including 84 KV pages converges within the existing
// four-pass budget (checked here at the worst-case alignment for every
// value in the sweep, not merely at some particular pool size that
// happens to work); a granule of 85 or more genuinely cannot be
// guaranteed and refuses loudly instead (also checked here), which is
// the honest, not the false, half of the round-3 claim.
TEST(auto_fit_converges_within_four_passes_for_granules_up_to_84_and_refuses_beyond) {
    for (size_t granule : {size_t(1), size_t(2), size_t(4), size_t(8), size_t(16), size_t(32),
                           size_t(64), size_t(84)}) {
        const int no_reserve = converge_passes(granule, /*protect_spare=*/false);
        CHECK(no_reserve >= 0);
        const int with_reserve = converge_passes(granule, /*protect_spare=*/true);
        CHECK(with_reserve >= 0);
    }
    // Beyond the guaranteed bound: refuses loudly (does not converge
    // within four passes) rather than silently claiming a guarantee it
    // cannot back up, for either split rule.
    CHECK_EQ(converge_passes(/*granule=*/85, /*protect_spare=*/false), -1);
    CHECK_EQ(converge_passes(/*granule=*/85, /*protect_spare=*/true), -1);
}

// The mechanism the fix relies on, stated as its own invariant: the
// spare cap auto_fit_trim hands back for the next pass never exceeds
// what the pool actually realized as spare THIS pass -- if it could,
// pool_sizing's own room growth (affordable - live_blocks widening as
// live shrinks) would re-inflate spare right back, which is the whole
// defect. True regardless of protect_spare, since both branches compute
// `next_spare_cap = spare_blocks - spare_cut` with `spare_cut >= 0`.
TEST(auto_fit_trim_spare_cap_never_exceeds_current_spare_blocks) {
    for (bool protect_spare : {false, true}) {
        const AutoFitTrim trim =
            auto_fit_trim(/*n_ctx=*/100000, /*spare_blocks=*/10000,
                         /*over=*/1, /*kv_block_bytes=*/1, /*lanes=*/1, /*kv_block_tokens=*/1,
                         /*n_ctx_floor=*/1, /*pass=*/0, protect_spare);
        CHECK(trim.next_spare_cap <= 10000ull);
    }
}

// protect_spare = true never touches spare while live still has room:
// the reserve's own headroom must be the LAST thing to give, not the
// first.
TEST(auto_fit_trim_protects_spare_while_live_has_room) {
    const AutoFitTrim trim =
        auto_fit_trim(/*n_ctx=*/100000, /*spare_blocks=*/10000,
                     /*over=*/1, /*kv_block_bytes=*/1, /*lanes=*/1, /*kv_block_tokens=*/1,
                     /*n_ctx_floor=*/1, /*pass=*/0, /*protect_spare=*/true);
    CHECK_EQ(trim.next_spare_cap, 10000ull);  // unchanged -- spare untouched
    CHECK(trim.next_n_ctx < 100000);          // live gave up the trim instead
}

// protect_spare = false is the mirror image: spare gives first, live is
// untouched as long as spare alone can cover the trim.
TEST(auto_fit_trim_spends_spare_first_without_a_reserve) {
    const AutoFitTrim trim =
        auto_fit_trim(/*n_ctx=*/100000, /*spare_blocks=*/10000,
                     /*over=*/1, /*kv_block_bytes=*/1, /*lanes=*/1, /*kv_block_tokens=*/1,
                     /*n_ctx_floor=*/1, /*pass=*/0, /*protect_spare=*/false);
    CHECK_EQ(trim.next_n_ctx, 100000ll);      // unchanged -- live untouched
    CHECK(trim.next_spare_cap < 10000ull);    // spare gave up the trim instead
}

// The floor is never crossed: once live has no more room and spare is
// also exhausted, auto_fit_trim refuses rather than lowering n_ctx below
// n_ctx_floor.
TEST(auto_fit_trim_refuses_at_the_floor_with_nothing_left) {
    const AutoFitTrim trim =
        auto_fit_trim(/*n_ctx=*/4096, /*spare_blocks=*/0,
                     /*over=*/1'000'000, /*kv_block_bytes=*/1, /*lanes=*/1,
                     /*kv_block_tokens=*/16, /*n_ctx_floor=*/4096, /*pass=*/3,
                     /*protect_spare=*/false);
    CHECK(trim.refuse);
}

// ------------------------------------------------------------------
// Round 4 review, defect 2: the round-3 granule fixture above used
// kv_block_bytes = 1 and an `over` of a WHOLE page, so even the ORIGINAL
// (pre-round-3, shrink-only) correction already reduced the pool total
// there -- it never modelled the actual measured defect. The real
// measured cells overshot by a fraction of one KV page: kv_block_bytes
// is thousands of bytes (36.2 KiB/token-class numbers elsewhere in this
// file), and the driver's residency reading can land only a few bytes
// past the ceiling. With kv_block_bytes = 4096 and over = 1 BYTE,
// overshoot_accum stays inside the SAME `affordable = budget_remaining /
// kv_block_bytes` bucket every pass (1 byte a pass, 4096 bytes wide) --
// so `affordable` never moves, and a shrink-only correction's own
// textbook behaviour is exactly what DESIGN's own defect diagnosis
// named: live -1 page (shrink_n_ctx's guaranteed minimum), spare +1 page
// (pool_sizing's spare_room = affordable - live_blocks grows by exactly
// what live lost), pool TOTAL flat.
// ------------------------------------------------------------------

constexpr uint64_t kSubPageBlockBytes = 4096;  // kv_block_bytes for this fixture
constexpr uint64_t kSubPageOverBytes  = 1;     // over, in BYTES -- far below one page
constexpr size_t   kSubPageLive0      = 100000;
// `+ kSubPageBlockBytes / 2` is deliberate slack WITHIN the current
// `affordable = budget_remaining / kv_block_bytes` bucket: an exact
// multiple of kv_block_bytes sits ON a bucket boundary, so subtracting
// even 1 byte from an exact multiple drops `affordable` by a whole page
// immediately -- the opposite of the sub-page scenario this fixture
// means to model, where the budget stays inside the SAME bucket across
// all four passes (cumulative overshoot_accum here is at most 4 bytes,
// far under the 2048-byte slack).
constexpr long long kSubPageBudget =
    static_cast<long long>(kSubPageLive0 + 2000) * static_cast<long long>(kSubPageBlockBytes) +
    static_cast<long long>(kSubPageBlockBytes) / 2;

// A stub of the ORIGINAL (pre-round-3) auto-fit correction: shrink_n_ctx
// on n_ctx directly, spare cap left untouched forever (SIZE_MAX, i.e.
// never narrowed) -- exactly backend_ov.cpp's own logic before round 3
// introduced auto_fit_trim and a shared spare_cap. Kept ONLY in this test
// file for the red-first comparison below; production no longer has this
// code path (backend_ov.cpp's Phase E calls auto_fit_trim now).
std::vector<size_t> sub_page_blocks_history(bool use_fix) {
    long long  n_ctx           = static_cast<long long>(kSubPageLive0);
    size_t     spare_cap       = static_cast<size_t>(-1);
    long long  overshoot_accum = 0;
    std::vector<size_t> history;
    for (int pass = 0; pass < 4; ++pass) {
        const size_t     live_blocks      = static_cast<size_t>(n_ctx);
        const size_t     capped_spare     = std::min<size_t>(1000000000, spare_cap);
        const long long  budget_remaining = kSubPageBudget - overshoot_accum;
        const PoolSizing sizing =
            pool_sizing(live_blocks, capped_spare, budget_remaining, kSubPageBlockBytes);
        history.push_back(sizing.blocks);

        overshoot_accum += static_cast<long long>(kSubPageOverBytes);
        if (use_fix) {
            const AutoFitTrim trim =
                auto_fit_trim(n_ctx, sizing.spare_blocks, kSubPageOverBytes, kSubPageBlockBytes,
                             /*lanes=*/1, /*kv_block_tokens=*/1, /*n_ctx_floor=*/1, pass,
                             /*protect_spare=*/false);
            n_ctx     = trim.next_n_ctx;
            spare_cap = trim.next_spare_cap;
        } else {
            // The old, defective correction: live shrinks by shrink_n_ctx's
            // own guaranteed minimum (>= 1 page); spare_cap is simply
            // never touched -- the defect this whole round exists to fix.
            n_ctx = shrink_n_ctx(n_ctx, kSubPageOverBytes, kSubPageBlockBytes, /*lanes=*/1,
                                 /*kv_block_tokens=*/1);
        }
    }
    return history;
}

// RED (round-4 report, verbatim): wired to the OLD stub (use_fix=false),
// this fails at the very first comparison -- pass 1's pool total equals
// pass 0's exactly, not less, because spare absorbs live's entire cut.
// Wired to auto_fit_trim (use_fix=true, the assertion actually shipped
// here), it passes: every pass's pool total is strictly smaller than the
// one before it.
TEST(sub_page_overshoot_pool_total_strictly_decreases_every_pass) {
    const std::vector<size_t> history = sub_page_blocks_history(/*use_fix=*/true);
    CHECK_EQ(history.size(), 4ull);
    for (size_t i = 1; i < history.size(); ++i) {
        CHECK(history[i] < history[i - 1]);  // strictly decreasing, every pass
    }
}

// The defect, reproduced explicitly and by construction: the OLD stub's
// sequence is not merely "not guaranteed to decrease" but perfectly
// FLAT -- live -1 page each pass is exactly cancelled by spare +1 page,
// so the pool total does not move at all across all four passes.
TEST(sub_page_overshoot_old_shrink_only_logic_leaves_pool_total_flat) {
    const std::vector<size_t> history = sub_page_blocks_history(/*use_fix=*/false);
    CHECK_EQ(history.size(), 4ull);
    for (size_t i = 1; i < history.size(); ++i) {
        CHECK_EQ(history[i], history[0]);  // flat, the defect
    }
}

// ------------------------------------- packed_values_scratch_geometry

namespace {
// The coder's own GQA shape (verified on the artifact, round-2 review,
// CRITICAL finding): 16 query heads, 2 KV heads, head_dim 256. text_config
// wrapped, the export shape backend_ov.cpp actually reads.
nlohmann::json coder_text_config_fixture() {
    return nlohmann::json{{"text_config",
                          {{"num_attention_heads", 16},
                           {"num_key_value_heads", 2},
                           {"head_dim", 256}}}};
}

// Today's (pre-fix) reading order, reproduced as a stub for the red-first
// comparison below: prefers num_key_value_heads over num_attention_heads --
// exactly the CRITICAL finding (the call site fed KV heads first, so the
// cap priced 2 heads instead of the 16 the plugin's MIXED-stage buffer is
// actually sized by -- an 8x undercount that meant the cap effectively
// never fired: 128 MiB priced at n_ctx 131,072 instead of the true 1 GiB).
std::optional<PackedValuesScratchGeometry> packed_values_scratch_geometry_stub_kv_heads_first(
    const nlohmann::json& config) {
    const nlohmann::json& tc =
        config.contains("text_config") && config["text_config"].is_object()
            ? config["text_config"]
            : config;
    PackedValuesScratchGeometry g;
    if (tc.contains("num_key_value_heads") && tc["num_key_value_heads"].is_number_integer()) {
        g.heads = tc["num_key_value_heads"].get<int>();
    } else if (tc.contains("num_attention_heads") &&
               tc["num_attention_heads"].is_number_integer()) {
        g.heads = tc["num_attention_heads"].get<int>();
    }
    if (tc.contains("head_dim") && tc["head_dim"].is_number_integer()) {
        g.head_size = tc["head_dim"].get<int>();
    }
    if (g.heads <= 0 || g.head_size <= 0) return std::nullopt;
    return g;
}
}  // namespace

// The defect, reproduced explicitly and by construction: on the coder's
// own GQA shape, the old KV-heads-first stub above returns heads 2, not
// 16 -- swapped into packed_values_scratch_geometry_coder_gqa_reads_query_
// heads below in place of packed_values_scratch_geometry (mechanically:
// replace the callee, nothing else), it fails --
//   FAIL <file>:<line>
//            g->heads == 16
//            left:  2
//            right: 16
//   FAIL packed_values_scratch_geometry_coder_gqa_reads_query_heads
//   1 case run, 1 failed
// -- captured verbatim from that swap this round (exact line numbers are
// this comment's own line count and drift with every edit here, so only
// the file:line SHAPE is kept; the round's own report has the literal
// numbers from the run that produced this). Kept here as a standing
// assertion of the defect itself, the same convention this file already
// uses for prefill_chunk_stub_no_cap below.
TEST(packed_values_scratch_geometry_old_kv_heads_first_stub_undercounts) {
    const auto g = packed_values_scratch_geometry_stub_kv_heads_first(coder_text_config_fixture());
    CHECK(g.has_value());
    CHECK_EQ(g->heads, 2);   // the bug, reproduced: KV heads, not query heads
    CHECK_EQ(g->head_size, 256);
}

// CRITICAL (round-2 review): the plugin's MIXED-stage intermediate buffer
// (get_internal_buffer_descs, paged_attention_opt.cpp, read directly) is
// sized by QUERY heads (`heads_num`), not KV heads -- on this GQA artifact
// the two differ 8x (16 vs 2), and feeding KV heads undercounts the buffer
// 8x, which is why the cap effectively never fired in production before
// this fix.
TEST(packed_values_scratch_geometry_coder_gqa_reads_query_heads) {
    const auto g = packed_values_scratch_geometry(coder_text_config_fixture());
    CHECK(g.has_value());
    CHECK_EQ(g->heads, 16);   // query heads, NOT the 2 KV heads
    CHECK_EQ(g->head_size, 256);
}

TEST(packed_values_scratch_geometry_flat_config_without_text_config_wrapper) {
    const nlohmann::json flat = {
        {"num_attention_heads", 16}, {"num_key_value_heads", 2}, {"head_dim", 256}};
    const auto g = packed_values_scratch_geometry(flat);
    CHECK(g.has_value());
    CHECK_EQ(g->heads, 16);
    CHECK_EQ(g->head_size, 256);
}

TEST(packed_values_scratch_geometry_missing_heads_is_nullopt) {
    const nlohmann::json cfg = {{"text_config", {{"head_dim", 256}}}};
    CHECK(!packed_values_scratch_geometry(cfg).has_value());
}

TEST(packed_values_scratch_geometry_missing_head_dim_is_nullopt) {
    const nlohmann::json cfg = {{"text_config", {{"num_attention_heads", 16}}}};
    CHECK(!packed_values_scratch_geometry(cfg).has_value());
}

// No fallback: hidden_size / num_attention_heads happens to equal head_dim
// here (4096 / 16 = 256) but this function must not compute it -- an
// explicit head_dim is required (round-2 review, finding 1: that identity
// is not safe for every architecture in this family, so its absence means
// "cannot price the cap," not "guess from hidden_size").
TEST(packed_values_scratch_geometry_no_hidden_size_fallback) {
    const nlohmann::json cfg = {
        {"text_config", {{"num_attention_heads", 16}, {"hidden_size", 4096}}}};
    CHECK(!packed_values_scratch_geometry(cfg).has_value());
}

// ------------------------------------ prefill_chunk_cap_for_packed_values

// Coder shape (verified on the artifact, --paged-kv u8:i4): 16 QUERY heads,
// 256 head_dim -- packed_values_scratch_geometry_coder_gqa_reads_query_heads
// above is what actually resolves this from config.json in production; the
// tests below take heads/head_size as given, the way the pure cap function
// itself does. kCoderKvBlockSize mirrors config.h's --kv-block-size default
// (config.cpp:637 admits 16 or 32 only). kScratchBudget mirrors
// production's own constant rather than a fixture-local number, so a
// change to the constant is felt here too.
namespace {
constexpr int      kCoderHeads       = 16;
constexpr int      kCoderHeadSize    = 256;
constexpr int      kCoderKvBlockSize = 32;
constexpr uint64_t kScratchBudget    = kPrefillScratchBudgetBytesPackedValues;  // 512 MiB

// Today's production behaviour on the 4-bit-value prefill path, reproduced
// as a stub: the chunk the auto-fit climb decided is served as-is, never
// revisited for the plugin's opt-path scratch. Kept ONLY in this test file
// for the red-first comparison below -- production calls
// lgc::prefill_chunk_cap_for_packed_values instead (backend_ov.cpp, after
// Phase E, where the served pool depth is final).
int prefill_chunk_stub_no_cap(int requested_chunk, long long /*max_ctx_tokens*/, int /*heads*/,
                              int /*head_size*/, uint64_t /*scratch_budget_bytes*/,
                              int /*block_size*/) {
    return requested_chunk;
}
}  // namespace

// The defect, reproduced explicitly and by construction, not merely
// described: prefill_chunk_stub_no_cap above IS today's production
// behaviour on the packed-4-bit-value prefill path -- the chunk the climb
// decided is served as-is. Swapped into the two production tests below in
// place of prefill_chunk_cap_for_packed_values (mechanically: replace the
// callee, nothing else), each fails on its own --
//   FAIL <file>:<line>
//            capped == 64
//            left:  128
//            right: 64
//   FAIL prefill_chunk_cap_packed_values_131072_shrinks_128_to_64
//   1 case run, 1 failed
//
//   FAIL <file>:<line>
//            expected: capped <= 64
//   FAIL <file>:<line>
//            capped == 64
//            left:  128
//            right: 64
//   FAIL prefill_chunk_cap_packed_values_119074_caps_to_64
//   1 case run, 1 failed
// -- captured verbatim from that swap this round (each test run filtered on
// its own name; exact line numbers drift with every edit to this comment,
// so only the file:line SHAPE is kept here -- the round's own report has
// the literal numbers from the run that produced this). Kept here as a
// standing
// assertion of the defect itself (the stub never shrinks anything, for any
// input), the same convention
// sub_page_overshoot_old_shrink_only_logic_leaves_pool_total_flat uses
// above for the auto-fit spare-cap defect.
TEST(prefill_chunk_cap_packed_values_old_no_cap_stub_never_shrinks) {
    CHECK_EQ(prefill_chunk_stub_no_cap(128, 131072, kCoderHeads, kCoderHeadSize, kScratchBudget,
                                       kCoderKvBlockSize),
             128);
    CHECK_EQ(prefill_chunk_stub_no_cap(128, 119074, kCoderHeads, kCoderHeadSize, kScratchBudget,
                                       kCoderKvBlockSize),
             128);
}

// Round-10 review, finding 1 (REAL defect, card-measured): the belt's
// `fits()` predicate now prices the SAME formula the reservation actually
// charges (1.5x margin + the exp_sums/max_logits pair), not the raw,
// un-margined single-buffer size -- a chunk the OLD raw check accepted
// could charge 50%+ more than the nominal 512 MiB budget (measured:
// chunk 2048 at 32 bounded partitions priced 512 MiB raw, 776 MiB actual).
// This makes the belt MORE conservative than its original empirical
// calibration here too: chunk 128 is 128 x 16 KiB x 512 partitions =
// 1,030 MiB raw, over the 512 MiB budget; chunk 64 is 515 MiB raw --
// this depth's own exact-fit point (the constant kPrefillScratchBudget
// BytesPackedValues was calibrated against, per its own comment), so the
// belt shrinks exactly one step, to 64.
//
// Round-11 review (Opus, RETRACTS round-10's own margined-`fits()`
// version of this test): round-10 priced the margined+exp/max formula
// here and changed this test's expectation to chunk 32 -- but the 512
// MiB budget was never re-measured against that formula, only asserted
// to be "more conservative," and a direct card measurement (round-11)
// showed the margined belt actively WRONG: `at_depth(101,824)` gave
// chunk 32 while the card measured and served chunk 64 at that exact
// depth. `fits()` is restored to the RAW proxy this budget was actually
// calibrated against (fit.h); this test's expectation reverts to the
// pre-round-10 chunk 64 it always was before that detour.
TEST(prefill_chunk_cap_packed_values_131072_shrinks_128_to_64) {
    const int capped = prefill_chunk_cap_for_packed_values(
        128, 131072, kCoderHeads, kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(capped, 64);
}

// max_ctx 119,074: the measured FAULT cell (CL_OUT_OF_RESOURCES at chunk
// 128) -- and the measured PASSING cell at chunk 64, which is exactly
// what the raw-proxy belt (restored, round-11 review) serves here.
TEST(prefill_chunk_cap_packed_values_119074_caps_to_64) {
    const int capped = prefill_chunk_cap_for_packed_values(
        128, 119074, kCoderHeads, kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(capped, 64);
}

// max_ctx 171,312 (the u8:i4 auto-fit depth from fit.h's own M7 comment
// history, §"171,488 -- corrected to 171,312"), evaluated as a served pool
// depth in its own right. ceil(171312/256) = 670 partitions. Chunk 128 is
// 1.31 GiB, chunk 64 is 670 MiB (over budget), chunk 32 is 335 MiB -- under.
TEST(prefill_chunk_cap_packed_values_171312_caps_to_32) {
    const int capped = prefill_chunk_cap_for_packed_values(
        128, 171312, kCoderHeads, kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(capped, 32);
}

// n_ctx 165,680 (10,355 pages of 16 -- --kv-block-size 16, not this file's
// usual 32, so `block_size` is passed explicitly rather than via
// `kCoderKvBlockSize`): the depth the card's own auto-fit adopts under the
// PATCHED plugin with --paged-attention-max-partitions 32 (CHANGELOG's own
// MEASURED entry). This is the UNBOUNDED-arm reading of that same depth --
// what the term would cost with the bound key absent (element 4 B, the
// unpatched-plugin shape) -- ceil(165680/256) = 648 partitions: chunk 128
// is 1,296 MiB, chunk 64 is 648 MiB, chunk 32 is 324 MiB -- the first to
// land under the 512 MiB budget, so the belt shrinks two steps, not one.
TEST(prefill_chunk_cap_packed_values_165680_unbounded_shrinks_128_to_32) {
    const int capped = prefill_chunk_cap_for_packed_values(
        128, 165680, kCoderHeads, kCoderHeadSize, kScratchBudget, /*block_size=*/16);
    CHECK_EQ(capped, 32);
}

// The term itself at the chunk the belt above settled on: the per-token
// rate at chunk 32 is a property of the chunk alone (3,072 B = 3 KiB,
// the same figure packed_values_prefill_scratch_bytes_per_token_chunk32_
// is_3_kib already establishes independent of depth) and the EXACT total
// at this specific depth (165,680) is 509,607,936 B = 486.0 MiB (chunk 32
// x 16 query heads x 256 head_dim x 4 B x 648 partitions x1.5 margin,
// exact -- (3 x 339,738,624 + 1) / 2 with no remainder).
TEST(packed_values_prefill_scratch_bytes_at_165680_chunk32) {
    CHECK_EQ(packed_values_prefill_scratch_bytes_per_token(32, kCoderHeads, kCoderHeadSize),
             3072ull);
    CHECK_EQ(packed_values_prefill_scratch_bytes(32, 165680, kCoderHeads, kCoderHeadSize),
             509607936ull);
}

// n_ctx 8,192: a shallow served pool. Chunk 128 is 128 x 16 KiB x 32
// partitions (ceil(8192/256) = 32) = 64 MiB -- comfortably under the
// 512 MiB budget -- so the request already satisfies it and comes back
// unchanged, same as if the cap had never been applied.
TEST(prefill_chunk_cap_packed_values_8192_leaves_128_unchanged) {
    const int capped = prefill_chunk_cap_for_packed_values(
        128, 8192, kCoderHeads, kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(capped, 128);
}

// max_ctx 71,689: the measured PASSING cell at chunk 128 (see fit.h's own
// comment: faults are observed only at ~119k past for chunk 128, not at
// 71,689). ceil(71689/256) = 281 partitions -- 128 x 16 KiB x 281 = 562 MiB,
// above the 512 MiB budget despite this exact cell having passed on the
// card. The belt is conservative BY DESIGN (fit.h's own comment): 512 MiB
// was chosen to reproduce chunk 64 at the deepest measured pool, not to
// price every individual passing cell exactly, so a cell that happened to
// pass at 128 can still be trimmed.
TEST(prefill_chunk_cap_packed_values_71689_shrinks_128_to_64) {
    const int capped = prefill_chunk_cap_for_packed_values(
        128, 71689, kCoderHeads, kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(capped, 64);
}

// Round-2 review residual 2: the function returns the first HALVING LADDER
// step that fits, not the largest block_size multiple that fits -- these
// differ here. 96 is a multiple of kCoderKvBlockSize (96 / 32 = 3) and
// 96 x 16 KiB x 281 partitions = 441,974,784 bytes, UNDER the 512 MiB
// (536,870,912-byte) budget -- 96 fits, and 96 > 64. But halving from 128
// visits 64 directly (128 / 2 = 64 exactly) and stops there because 64
// already fits; 96 sits between 64 and 128 and is never visited. This is
// by design (fit.h's own comment on prefill_chunk_cap_for_packed_values):
// the ladder only ever returns chunks the belt's own measurements were
// taken at.
TEST(prefill_chunk_cap_packed_values_71689_skips_a_larger_fitting_multiple) {
    constexpr uint64_t kPartitions71689 = 281;  // ceil(71689 / 256)
    constexpr uint64_t kPerChunkTokenBytes =
        static_cast<uint64_t>(kCoderHeads) * kCoderHeadSize * 4ull * kPartitions71689;
    constexpr int kLargerMultiple = 96;
    CHECK(kLargerMultiple % kCoderKvBlockSize == 0);          // a legal block_size multiple
    CHECK(kLargerMultiple * kPerChunkTokenBytes <= kScratchBudget);  // and it fits
    CHECK(kLargerMultiple > 64);                               // strictly larger than what we get

    const int capped = prefill_chunk_cap_for_packed_values(
        128, 71689, kCoderHeads, kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(capped, 64);  // not 96, even though 96 also fits and is larger
}

// Floor case (round-2 review residual 2): when NOTHING on the ladder fits
// -- not even block_size itself -- the function still returns block_size
// rather than refusing or returning 0. Geometry is deliberately absurd
// (1000 heads x 1000 head_size, one partition) against a 1-byte budget so
// every ladder step, including the floor, is manifestly over budget:
// block_size (32) x 1000 x 1000 x 4 = 128,000,000 bytes, nowhere near
// fitting a 1-byte budget.
TEST(prefill_chunk_cap_packed_values_floor_case_returns_block_size_when_nothing_fits) {
    const int capped = prefill_chunk_cap_for_packed_values(
        128, /*max_ctx_tokens=*/256, /*heads=*/1000, /*head_size=*/1000,
        /*scratch_budget_bytes=*/1, kCoderKvBlockSize);
    CHECK_EQ(capped, kCoderKvBlockSize);  // the floor, not a refusal -- see the function's own comment
}

// The chunk is never RAISED, even when the budget has room to spare -- a
// request already below what the budget would admit comes back exactly as
// requested.
TEST(prefill_chunk_cap_packed_values_never_raises_the_chunk) {
    const int capped = prefill_chunk_cap_for_packed_values(
        64, 8192, kCoderHeads, kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(capped, 64);
}

// Round-11 review (Opus), finding 2: `kMaxMeasuredPackedValuesChunk` (the
// hard 128-token cap -- chunk 2048, the operator's own `--prefill-chunk`
// default, has never been validated on any plugin or card for this
// scratch path) had no test of its own -- every existing belt test's
// budget check ALSO happened to fire first, so a regression that deleted
// the cap entirely could pass every other test in this file. n_ctx 2,560
// (10 partitions) is shallow enough that the RAW proxy budget check
// alone would leave 256 unchanged and 2048 unchanged too (41.9 MiB and
// 335.5 MiB respectively, both comfortably under the 512 MiB budget) --
// the cap is the ONLY thing that can be shrinking either of these.
TEST(prefill_chunk_cap_packed_values_256_capped_to_128_by_the_measured_ceiling) {
    const int capped = prefill_chunk_cap_for_packed_values(
        256, /*max_ctx_tokens=*/2560, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kCoderKvBlockSize);
    CHECK_EQ(capped, 128);
}
TEST(prefill_chunk_cap_packed_values_2048_capped_to_128_by_the_measured_ceiling) {
    const int capped = prefill_chunk_cap_for_packed_values(
        2048, /*max_ctx_tokens=*/2560, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kCoderKvBlockSize);
    CHECK_EQ(capped, 128);
}

// The bounded arm (N = 32, 2-byte elements once the plugin accepts the
// key) is capped the same way -- the flat, past-the-bound charge stops
// growing with depth, so a shallow enough budget check alone would leave
// a big requested chunk untouched (1024 x 16 x 256 x 2 B x 32 partitions
// = 256 MiB, under the 512 MiB budget) while the measured cap still
// pulls it down to 128, never to 1024.
TEST(prefill_chunk_cap_packed_values_bounded_n32_capped_to_128_not_1024) {
    const int capped = prefill_chunk_cap_for_packed_values_ex(
        /*requested_chunk=*/1024, /*partitions=*/32, kCoderHeads, kCoderHeadSize,
        /*element_bytes=*/2, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(capped, 128);
}

// Round-12 review (Opus), finding 2: `prefill_chunk_cap_for_packed_
// values_budget_only_ex` and `packed_values_scratch_fits_budget` (fit.h)
// now have a live caller (backend_ov.cpp's own load-time reduction log,
// moved to where the search/at_depth result is decided) -- these two
// tests exercise them directly rather than only through `_ex`'s own
// refactor (which already proved the budget-only ladder's arithmetic
// unchanged, but never in isolation from the cap it now sits under).
TEST(prefill_chunk_cap_packed_values_budget_only_does_not_apply_the_measured_cap) {
    // 131,072 (partitions 512): _ex's own equivalent test settles at
    // chunk 64 with the cap applied -- the budget-only ladder alone must
    // agree here, since 64 is already under 128 and the cap never fires.
    const int budget_only = prefill_chunk_cap_for_packed_values_budget_only_ex(
        128, /*partitions=*/512, kCoderHeads, kCoderHeadSize, /*element_bytes=*/4, kScratchBudget,
        kCoderKvBlockSize);
    CHECK_EQ(budget_only, 64);
    // A shallow depth (2,560 tokens, 10 partitions) where the budget
    // alone would leave 2048 unchanged -- only the measured cap (not
    // exercised by this function) would shrink it, so the budget-only
    // ladder must return the full 2048 uncapped.
    const int budget_only_shallow = prefill_chunk_cap_for_packed_values_budget_only_ex(
        2048, /*partitions=*/10, kCoderHeads, kCoderHeadSize, /*element_bytes=*/4, kScratchBudget,
        kCoderKvBlockSize);
    CHECK_EQ(budget_only_shallow, 2048);
}

TEST(packed_values_scratch_fits_budget_matches_the_ladder_it_backs) {
    // 64 tokens, 512 partitions: the exact chunk the budget-driven ladder
    // above settles on at 131,072 -- must report "fits."
    CHECK(packed_values_scratch_fits_budget(64, 512, kCoderHeads, kCoderHeadSize, 4,
                                            kScratchBudget));
    // 128 at the same depth is the FAULT cell the ladder shrinks away
    // from -- must report "does not fit."
    CHECK(!packed_values_scratch_fits_budget(128, 512, kCoderHeads, kCoderHeadSize, 4,
                                             kScratchBudget));
}

// Granule (round-2 review, finding 5): the result must stay a multiple of
// the KV block size config.cpp:738 already enforces --prefill-chunk against
// -- requested 96 with block_size 32 must land on 64, not 48. 96 / 2 = 48
// FLOORS to 32 (one block below the true half) but CEILS to 64 (one block
// at or above it); the mitigation rounds up, so a request that fails at 96
// tries 64 next, not a smaller value that was never actually measured to
// help. Geometry is deliberately trivial (heads 1, head_size 1, one
// partition) so only the rounding direction is under test: 96 x 4 = 384
// bytes fails a 300-byte budget, 64 x 4 = 256 passes, and 48 is never
// tried.
// Round-10 review briefly raised `scratch_budget_bytes` from 300 to 900 to
// match a margined `fits()` -- retracted in round-11 review (`fits()`
// restored to the raw proxy the 512 MiB production budget was actually
// calibrated against; see prefill_chunk_cap_packed_values_131072_shrinks_
// 128_to_64's own comment for the measurement). Back to the original
// 300-byte budget against the raw formula.
TEST(prefill_chunk_cap_packed_values_rounds_up_to_the_block_granule) {
    const int capped =
        prefill_chunk_cap_for_packed_values(96, /*max_ctx_tokens=*/256, /*heads=*/1,
                                            /*head_size=*/1, /*scratch_budget_bytes=*/300,
                                            /*block_size=*/32);
    CHECK_EQ(capped, 64);
}

// u8 (symmetric, no 4-bit value side) is unaffected because backend_ov.cpp
// never calls this function on that path -- there is no 4-bit signal to
// gate it on, so there is nothing to unit-test about u8 here directly.
// Exercised instead as the function's own no-op guard: geometry the caller
// could not price (heads/head_size/depth/block_size 0, as if the caller
// had nothing to price) passes the request through unchanged rather than
// inventing a cap from nothing.
TEST(prefill_chunk_cap_packed_values_unset_geometry_is_a_no_op) {
    CHECK_EQ(prefill_chunk_cap_for_packed_values(128, 119074, /*heads=*/0, kCoderHeadSize,
                                                 kScratchBudget, kCoderKvBlockSize),
             128);
    CHECK_EQ(prefill_chunk_cap_for_packed_values(128, 119074, kCoderHeads, /*head_size=*/0,
                                                 kScratchBudget, kCoderKvBlockSize),
             128);
    CHECK_EQ(prefill_chunk_cap_for_packed_values(128, /*max_ctx_tokens=*/0, kCoderHeads,
                                                 kCoderHeadSize, kScratchBudget, kCoderKvBlockSize),
             128);
    CHECK_EQ(prefill_chunk_cap_for_packed_values(128, 119074, kCoderHeads, kCoderHeadSize,
                                                 kScratchBudget, /*block_size=*/0),
             128);
}

// Round-2 review, F1 (real defect, not yet re-measured on a card): the belt
// is non-increasing in DEPTH, but backend_ov.cpp's call site (after Phase E)
// always asks it starting from the UNCLAMPED `prefill_chunk_` -- not the
// chunk the fit's own climb already priced the term at
// (`packed_values_scratch_chunk_`). Every path to the served depth only
// ever LOWERS it from the climb's own settled max_ctx (min(wanted, max_ctx),
// the prefix-cache reserve, Phase E's trims), so a trim that crosses one of
// the belt's own halving-ladder step boundaries can hand back a chunk
// LARGER than the one the reservation priced, even though the served depth
// only went down: at max_ctx 131,104 the belt (started from 128) settles at
// chunk 32 (this is prefill_chunk_cap_packed_values_171312_caps_to_32's own
// neighbourhood -- one page short of the 131,072-token cliff below); Phase E
// trimming one KV page to 131,072 crosses that cliff, and the belt, asked
// AGAIN from the unclamped 128 at the new, slightly SMALLER depth, climbs
// back up to 64 (prefill_chunk_cap_packed_values_131072_shrinks_128_to_64,
// above) -- a served buffer bigger than what was charged.
// Round-10 review briefly re-derived this pair's boundary for a margined
// `fits()` (86,816/86,784) -- retracted in round-11 review along with the
// margined `fits()` itself (see prefill_chunk_cap_packed_values_131072_
// shrinks_128_to_64's own comment: the margined belt disagreed with a
// direct card measurement). `fits()` restored to the raw proxy, so this
// pair reverts to its original boundary too: 131,104 -> chunk 32,
// 131,072 -> chunk 64, one KV page apart.
TEST(prefill_chunk_cap_packed_values_at_131104_settles_chunk_32) {
    // Establishes the priced chunk for the fixture below: this is what
    // fit_context_packed_values's own fixed point would have settled on
    // had the climb's own settled max_ctx landed here.
    CHECK_EQ(prefill_chunk_cap_for_packed_values(128, 131104, kCoderHeads, kCoderHeadSize,
                                                 kScratchBudget, kCoderKvBlockSize),
             32);
}

// RED: the OLD call site's own shape -- prefill_chunk_ (128, unclamped) fed
// straight to the belt at the POST-TRIM depth (131,072, one page short of
// the priced depth above) -- reproduces the defect exactly: a chunk (64)
// LARGER than the one the reservation priced (32) at the depth just above.
TEST(prefill_chunk_cap_packed_values_old_call_site_shape_grows_past_the_priced_chunk) {
    const int served_at_old_call_site = prefill_chunk_cap_for_packed_values(
        /*prefill_chunk_ (unclamped)=*/128, /*paged_n_ctx_ (post-trim)=*/131072, kCoderHeads,
        kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(served_at_old_call_site, 64);
    CHECK(served_at_old_call_site > 32);  // > the chunk the term above priced -- the defect
}

// Round-3 review, finding 3: the two tests above (and an earlier version
// of the "GREEN" test below) exercised `prefill_chunk_cap_for_packed_
// values` directly with a HAND-COMPUTED `std::min` standing in for the
// call site's own choice -- correct arithmetic, but a second, independent
// copy of it: if backend_ov.cpp's belt call site regressed back to the
// unclamped `prefill_chunk_` alone (the defect the test above documents),
// nothing here would notice, because the test never calls the code the
// call site actually calls. `lgc::belt_requested_chunk` (fit.h) is that
// choice, extracted into one named, exported function; backend_ov.cpp's
// belt call site now calls it directly (no local reimplementation), and
// the tests below call the SAME function, not a copy of its arithmetic.
//
// RED, captured verbatim (belt_requested_chunk's body temporarily reverted
// to `return prefill_chunk;` -- the pre-F1 choice, ignoring priced_chunk --
// rebuilt, and re-run filtered to this test's own name; restored right
// after):
//   FAIL <file>:<line>
//            belt_requested_chunk(128, 32) == 32
//            left:  128
//            right: 32
//   FAIL belt_requested_chunk_bounds_prefill_chunk_by_the_priced_chunk
//   1 case run, 1 failed
TEST(belt_requested_chunk_bounds_prefill_chunk_by_the_priced_chunk) {
    // The ordinary case: the priced chunk is smaller than the unclamped
    // prefill chunk, so it wins.
    CHECK_EQ(belt_requested_chunk(128, 32), 32);
    // The priced chunk can never be LARGER than the prefill chunk it was
    // derived from (fit_context_packed_values only ever shrinks, never
    // grows, its own starting chunk) -- but the function is defensive
    // about it anyway: still the smaller of the two either way.
    CHECK_EQ(belt_requested_chunk(32, 128), 32);
    // 0 or negative priced_chunk: nothing was priced this load (not a
    // 4-bit-values load, or geometry was unavailable) -- passes
    // prefill_chunk through unchanged, the same "nothing to price"
    // convention prefill_chunk_cap_for_packed_values's own guards use.
    CHECK_EQ(belt_requested_chunk(128, 0), 128);
    CHECK_EQ(belt_requested_chunk(128, -5), 128);
}

// GREEN: the fix -- backend_ov.cpp's belt call site now passes
// `lgc::belt_requested_chunk(prefill_chunk_, packed_values_scratch_chunk_)`
// as the belt's own requested chunk, reproduced here by calling that exact
// function (not a reimplementation of it). The belt only ever shrinks its
// input (never raises it, per its own doc comment), so bounding the INPUT
// to the priced chunk (32) bounds the OUTPUT the same way, at the identical
// post-trim depth (131,072) the red case above used.
TEST(prefill_chunk_cap_packed_values_clamped_call_site_shape_stays_at_the_priced_chunk) {
    const int prefill_chunk              = 128;  // unclamped, activation-probe chunk
    const int packed_values_priced_chunk = 32;  // the term's own fixed point, at max_ctx 131,104
    const int requested_chunk_for_belt =
        belt_requested_chunk(prefill_chunk, packed_values_priced_chunk);
    CHECK_EQ(requested_chunk_for_belt, 32);

    const int served_at_fixed_call_site = prefill_chunk_cap_for_packed_values(
        requested_chunk_for_belt, /*paged_n_ctx_ (post-trim)=*/131072, kCoderHeads,
        kCoderHeadSize, kScratchBudget, kCoderKvBlockSize);
    CHECK_EQ(served_at_fixed_call_site, 32);
    CHECK(served_at_fixed_call_site <= packed_values_priced_chunk);  // never exceeds the priced chunk
}

// --------------------------- packed_values_prefill_scratch_bytes (M9 fit term)

// CONSISTENT WITH THE MEASURED FAULT (2026-09-03, 16 GiB card, coder,
// --paged-kv u8:i4, host-side VRAM allocator sampled every 2 s -- the
// sampler measured AGGREGATE free VRAM against a proxy formula, not a
// direct trace of the plugin's own buffer, so "consistent with" rather
// than "root cause": round-3 review, finding 1): the belt above caps the
// prefill chunk from the same buffer this term prices, but capping the
// chunk after the fact does not RESERVE anything -- a pool sized without
// this term can still run free VRAM to 0 during the prefill that grows the
// buffer (171,312-token auto-fit pool: free VRAM 0 MiB, xe "VM worker
// error: -12", "exec queue reset detected", CL_OUT_OF_RESOURCES; the same
// prompt on a 131,072-token pool passed with 492 MiB free). This section
// prices the buffer so the FIT itself charges for it, at the SERVED chunk
// and depth, x1.5 for the measured overlap bound (see packed_values_
// prefill_scratch_bytes's own comment: the sampler held the buffer for
// ~0.9x of the whole prefill, rounded up to 1.0x, plus a ~0.55x swing at a
// resize -- 1.5x covers both with margin, superseding the earlier,
// unmeasured 2x).
//
// Both hand-worked cases below reuse the coder shape already established
// above (kCoderHeads = 16 QUERY heads, kCoderHeadSize = 256 head_dim) and
// are independently corroborated by DESIGN.md's own arithmetic for the
// belt at these exact points ("chunk 32 is 335 MiB" at n_ctx 171,312;
// "8 KiB at chunk 128 ... 2 KiB at chunk 32" for the per-token, un-scaled
// rate) -- this section adds the x1.5 this file's belt tests do not need,
// and the fit-side wiring those tests do not exercise.
TEST(packed_values_prefill_scratch_bytes_chunk128_n_ctx131072_is_1024_mib_x1_5_to_1536) {
    // ceil(131072/256) = 512 partitions exactly (131072/256 = 512.0).
    // single buffer = 128 x 16 x 256 x 4 x 512
    //               = 128 x 16,384 (bytes/partition/head... folded) x 512
    // worked as the function does it: chunk(128) * heads(16) * head_size(256)
    // * 4 * partitions(512) = 128*16=2048; *256=524,288; *4=2,097,152;
    // *512=1,073,741,824 bytes = 1,024 MiB exactly (2^30). x1.5 for the
    // measured overlap bound: 1,610,612,736 bytes = 1,536 MiB exactly
    // (1,024 * 1.5). (The retracted 2x figure was 2,147,483,648 bytes,
    // 2,048 MiB -- this assertion is red against that multiplier.)
    const uint64_t bytes =
        packed_values_prefill_scratch_bytes(128, 131072, kCoderHeads, kCoderHeadSize);
    CHECK_EQ(bytes, 1610612736ull);
    CHECK_EQ(bytes, 3ull * (1024ull * kMiB) / 2ull);
}

TEST(packed_values_prefill_scratch_bytes_chunk32_n_ctx171312_is_335_mib_x1_5_to_502_5) {
    // ceil(171312/256): 171312 / 256 = 669.1875 -> 670 partitions.
    // single buffer = 32*16=512; *256=131,072; *4=524,288; *670=351,272,960
    // bytes. 351,272,960 / 1,048,576 = 335.0 exactly (1,048,576 * 335 =
    // 351,272,960) -- 335 MiB, matching this file's own belt-side comment
    // ("chunk 32 is 335 MiB") for the identical (chunk, n_ctx) pair.
    // x1.5 for the measured overlap bound: 526,909,440 bytes -- 502.5 MiB
    // exactly (335 * 1.5), not a whole MiB count but an exact byte value.
    // (The retracted 2x figure was 702,545,920 bytes, 670 MiB -- this
    // assertion is red against that multiplier.)
    const uint64_t bytes =
        packed_values_prefill_scratch_bytes(32, 171312, kCoderHeads, kCoderHeadSize);
    CHECK_EQ(bytes, 526909440ull);
    CHECK_EQ(bytes, 3ull * (335ull * kMiB) / 2ull);
}

// Any missing input (chunk, n_ctx, heads, head_size <= 0) is "nothing to
// price" and returns 0 -- the caller's job is to skip the term, not to
// invent a buffer from nothing (same convention prefill_chunk_cap_for_
// packed_values's own guards use).
TEST(packed_values_prefill_scratch_bytes_zero_guards) {
    CHECK_EQ(packed_values_prefill_scratch_bytes(0, 131072, kCoderHeads, kCoderHeadSize), 0ull);
    CHECK_EQ(packed_values_prefill_scratch_bytes(128, 0, kCoderHeads, kCoderHeadSize), 0ull);
    CHECK_EQ(packed_values_prefill_scratch_bytes(128, 131072, 0, kCoderHeadSize), 0ull);
    CHECK_EQ(packed_values_prefill_scratch_bytes(128, 131072, kCoderHeads, 0), 0ull);
}

// The per-token slope the climb below folds into kv_bytes_token: chunk x
// 8 KiB / 256, x1.5 for the measured overlap bound (chunk 128 -> 12
// KiB/token, chunk 32 -> 3 KiB/token; this file's own belt comment states
// the un-scaled 8 KiB/2 KiB figures) -- still more than the 8.8 KiB/token
// the u8:i4 KV itself costs (DESIGN.md) at chunk 128, which is why this
// term moves the auto-fit depth measurably rather than being noise against
// it. (The retracted 2x multiplier gave 16 KiB/4 KiB -- these assertions
// are red against that multiplier.)
TEST(packed_values_prefill_scratch_bytes_per_token_chunk128_is_12_kib) {
    CHECK_EQ(packed_values_prefill_scratch_bytes_per_token(128, kCoderHeads, kCoderHeadSize),
             12288ull);
}

TEST(packed_values_prefill_scratch_bytes_per_token_chunk32_is_3_kib) {
    CHECK_EQ(packed_values_prefill_scratch_bytes_per_token(32, kCoderHeads, kCoderHeadSize),
             3072ull);
}

TEST(packed_values_prefill_scratch_bytes_per_token_zero_guards) {
    CHECK_EQ(packed_values_prefill_scratch_bytes_per_token(0, kCoderHeads, kCoderHeadSize), 0ull);
    CHECK_EQ(packed_values_prefill_scratch_bytes_per_token(128, 0, kCoderHeadSize), 0ull);
    CHECK_EQ(packed_values_prefill_scratch_bytes_per_token(128, kCoderHeads, 0), 0ull);
}

// The conservativeness proof packed_values_prefill_scratch_bytes_per_
// token's own comment states algebraically, checked here at n_ctx values
// that do NOT fall on a 256-token partition boundary (where the linear
// charge and the exact ceil()-based total diverge the most): the closed-form
// charge (`per_token * n_ctx + fixed`, what fit_context_packed_values below
// actually charges) must never be LESS than the exact total a real load at
// that depth would need.
TEST(packed_values_prefill_scratch_linear_charge_never_undercounts_the_exact_total) {
    for (const int chunk : {32, 64, 128}) {
        for (const long long n_ctx : {1LL, 100LL, 255LL, 257LL, 769LL, 131072LL, 171312LL,
                                      171313LL}) {
            const uint64_t exact =
                packed_values_prefill_scratch_bytes(chunk, n_ctx, kCoderHeads, kCoderHeadSize);
            const uint64_t per_token =
                packed_values_prefill_scratch_bytes_per_token(chunk, kCoderHeads, kCoderHeadSize);
            const uint64_t fixed =
                packed_values_prefill_scratch_bytes(chunk, 1, kCoderHeads, kCoderHeadSize);
            const uint64_t charged = per_token * static_cast<uint64_t>(n_ctx) + fixed;
            CHECK(charged >= exact);
        }
    }
}

// ------------------------------------------- fit_context_packed_values

// The fixture DESIGN.md's own §7.0.2 measurement window used, reconstructed
// here as an M7-style engineered fixture (the file's established
// convention -- see kTotal's own comment at the top): a 16 GiB-class card,
// coder shape, --paged-kv u8:i4 (~8.8 KiB/token, DESIGN.md), one lane,
// --kv-block-size 32, requested --prefill-chunk 128. Round numbers for the
// fixed terms (weights/activation/margin/slab reuse the file's own top-of-
// file constants), because only the KV-vs-term arithmetic is under test.
namespace {
constexpr uint64_t kM9Total          = 16ull * kGiB;
constexpr uint64_t kM9KvBytesToken   = 9011;  // ~8.8 KiB/token, u8:i4 (DESIGN.md)
constexpr int       kM9KvBlockTokens = 32;     // matches kCoderKvBlockSize
constexpr int       kM9RequestedChunk = 128;

FitTerms m9_base_terms() {
    FitTerms t;
    t.total          = kM9Total;
    t.weights        = kWeights;
    t.drafters       = 0;
    t.slot_pool      = 0;
    t.activations    = kActivation;
    t.slab_per_lane  = kSlab;
    t.kv_bytes_token = kM9KvBytesToken;
    t.margin         = kMargin;
    t.lanes           = 1;
    t.kv_block_tokens = kM9KvBlockTokens;
    t.n_ctx_floor     = 4096;
    return t;
}
}  // namespace

// RED FIRST: `fit_context_packed_values` and `packed_values_prefill_
// scratch_bytes`/`_per_token` do not exist before this change -- reverting
// this milestone's diff against this test file does not merely fail an
// assertion, it fails to COMPILE, the same convention this file already
// uses for a function introduced alongside its own first test (e.g.
// auto_fit_converges_within_four_passes_for_granules_up_to_84_and_refuses_
// beyond, above: "RED before auto_fit_trim exists (compile error)").
//
// With the term ABSENT (today's production call, `fit_context(fterms)`
// alone -- literally the call site this milestone's backend_ov.cpp diff
// replaces for the 4-bit-values case): budget = total - fixed, divided by
// the bare KV rate. fixed = weights(kWeights) + activation(kActivation) +
// margin(kMargin) + lanes(1)*slab(kSlab) = 1,288,490,189 + 697,932,186 +
// 268,435,456 + 67,108,864 = 2,321,966,695. budget = 17,179,869,184 -
// 2,321,966,695 = 14,857,902,489. old_max_ctx = 14,857,902,489 / 9,011 /
// 1 = 1,648,935 tokens, floored to a 32-token page: 1,648,832.
//
// With the term CHARGED (fit_context_packed_values, this milestone, x1.5
// overlap bound, PLUS round-6 review finding 4's exp_sums/max_logits pair
// on top of tmp_out): the climb settles at chunk 32 after one belt
// correction (128 -> 32, the same shrink prefill_chunk_cap_packed_values_
// 171312_caps_to_32 above measures at this exact shape) and lands on
// max_ctx 1,227,936 -- strictly LOWER than the term-absent figure, and the
// reservation this fit computed (fit.reserved_total) still fits under
// kM9Total, so the depth it landed on is genuinely admissible, not merely
// smaller. (The retracted 2x multiplier landed on 1,133,504 here, and the
// pre-finding-4 figure -- tmp_out alone -- was 1,229,568; both are red
// against the current formula.)
TEST(fit_context_packed_values_drops_the_auto_fit_depth_versus_the_term_absent_fit) {
    const FitTerms  base          = m9_base_terms();
    const FitResult old_fit       = fit_context(base);  // today's call, term absent
    CHECK_EQ(old_fit.max_ctx, 1648832);

    const PackedValuesFitTerm term = fit_context_packed_values(
        base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens);

    CHECK(term.fit.max_ctx < old_fit.max_ctx);   // the depth the term must move
    CHECK(term.fit.max_ctx >= base.n_ctx_floor);  // still usable, not refused into the floor
    CHECK(term.fit.admissible);
    CHECK(term.fit.reserved_total <= kM9Total);   // the charge this fit made actually fits
    CHECK_EQ(term.fit.max_ctx, 1227936);
    CHECK_EQ(term.chunk, 32);
}

// Self-consistency (the whole point of iterating inside the climb instead
// of pricing the requested chunk once): the belt, asked directly for the
// chunk it would pick at the SETTLED depth, must agree with the chunk the
// climb actually priced the term at -- otherwise the reservation and the
// belt would be charging for two different buffers.
TEST(fit_context_packed_values_settles_on_the_chunk_the_belt_agrees_with) {
    const FitTerms           base = m9_base_terms();
    const PackedValuesFitTerm term = fit_context_packed_values(
        base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens);
    const int belt_chunk_at_settled_depth = prefill_chunk_cap_for_packed_values(
        kM9RequestedChunk, term.fit.max_ctx, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kM9KvBlockTokens);
    CHECK_EQ(term.chunk, belt_chunk_at_settled_depth);
    CHECK(term.iterations >= 1);
}

// Round-15 review (Opus, REAL defect, RETRACTS this test's own original
// premise -- kept on the record, per DESIGN §7.0.1): `belt_enabled =
// false` used to still PRICE the uncapped chunk (a real, and often
// large, charge) -- measured on the card, that charge alone was enough
// to refuse a load `ARCINT_PREFILL_CHUNK_CAP=off` was supposed to let
// through so the plugin's own kernel could be the thing that faults, not
// this repository's own budget estimate. `belt_enabled=false` is now a
// FULL measurement bypass: no belt, no cap, no scratch term at all --
// `chunk` stays pinned at the REQUESTED one (still uncapped, still what
// the load actually serves), but the fit runs exactly as if this term
// did not exist, so `belt_off.fit.max_ctx` must equal a PLAIN
// `fit_context(base)` (no packed-values term folded in at all) --
// strictly MORE than the belt-enabled climb's own max_ctx, not
// equal-or-less the way charging a real (if uncapped) term used to make
// it.
TEST(fit_context_packed_values_belt_disabled_charges_nothing) {
    const FitTerms base = m9_base_terms();
    const PackedValuesFitTerm belt_off =
        fit_context_packed_values(base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize,
                                  kScratchBudget, kM9KvBlockTokens, /*belt_enabled=*/false);
    const PackedValuesFitTerm belt_on =
        fit_context_packed_values(base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize,
                                  kScratchBudget, kM9KvBlockTokens, /*belt_enabled=*/true);
    const FitResult plain = fit_context(base);
    CHECK_EQ(belt_off.chunk, kM9RequestedChunk);  // pinned, unclamped -- still what is served
    CHECK_EQ(belt_off.iterations, 1);
    CHECK_EQ(belt_off.per_token_bytes, 0ull);
    CHECK_EQ(belt_off.fixed_bytes, 0ull);
    CHECK_EQ(belt_off.fit.max_ctx, plain.max_ctx);  // exactly the term-free fit, no term at all
    CHECK(belt_off.fit.max_ctx > belt_on.fit.max_ctx);  // strictly more room than charging
                                                        // ANY term, capped or not, would leave
}

// Round-12 review (Opus), finding 1 (retained under the round-15 full-
// bypass redesign, above -- the measured cap must stay bypassed too, not
// just the budget ladder, and `kM9RequestedChunk` (128, the test above)
// sits exactly AT the cap, too small to prove it on its own): 2048 (the
// operator's own --prefill-chunk default) is well past the 128-token
// ceiling, so a served chunk of exactly 2048 here proves BOTH that the
// cap did not fire AND that the belt did not run at all -- a capped-but-
// still-priced result would also show `chunk <= 2048`, but not a term of
// exactly zero.
TEST(fit_context_packed_values_belt_disabled_bypasses_the_measured_cap_too) {
    const FitTerms base = m9_base_terms();
    const PackedValuesFitTerm belt_off =
        fit_context_packed_values(base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize,
                                  kScratchBudget, kM9KvBlockTokens, /*belt_enabled=*/false);
    CHECK_EQ(belt_off.chunk, 2048);  // pinned, uncapped by budget OR the measured ceiling
    CHECK_EQ(belt_off.iterations, 1);
    CHECK_EQ(belt_off.per_token_bytes, 0ull);
    CHECK_EQ(belt_off.fixed_bytes, 0ull);
}

// u8/f16 values: backend_ov.cpp never calls this function on that path (it
// is gated on the requested paged-KV value precision being 4-bit, at the
// FitTerms/fit_context call site -- see backend_ov.cpp's `packed_values`
// local). Exercised here the same way this file already exercises the
// belt's own no-op guard, above: geometry the caller could not price
// (heads/head_size 0, as if config.json carried neither) is a pure
// passthrough to `fit_context(base)` -- no term, `chunk` unchanged -- not a
// second, silently-different code path.
TEST(fit_context_packed_values_unset_geometry_matches_plain_fit_context) {
    const FitTerms base = m9_base_terms();
    const FitResult plain = fit_context(base);

    const PackedValuesFitTerm no_heads = fit_context_packed_values(
        base, kM9RequestedChunk, /*heads=*/0, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens);
    CHECK_EQ(no_heads.fit.max_ctx, plain.max_ctx);
    CHECK_EQ(no_heads.per_token_bytes, 0ull);
    CHECK_EQ(no_heads.fixed_bytes, 0ull);
    CHECK_EQ(no_heads.chunk, kM9RequestedChunk);

    const PackedValuesFitTerm no_head_size = fit_context_packed_values(
        base, kM9RequestedChunk, kCoderHeads, /*head_size=*/0, kScratchBudget, kM9KvBlockTokens);
    CHECK_EQ(no_head_size.fit.max_ctx, plain.max_ctx);

    const PackedValuesFitTerm no_chunk = fit_context_packed_values(
        base, /*requested_chunk=*/0, kCoderHeads, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens);
    CHECK_EQ(no_chunk.fit.max_ctx, plain.max_ctx);
}

// The Phase E replay/backoff guarantee (auto_fit_converges_within_four_
// passes_for_granules_up_to_84_and_refuses_beyond, above) is parameterized
// entirely by the driver's rounding granule -- nothing about it reads how
// the STARTING n_ctx for that loop was chosen. This is the direct check
// that plugging the term-adjusted depth in as that starting point changes
// nothing about the guarantee: the same auto_fit_trim/pool_sizing loop,
// seeded from fit_context_packed_values's own settled max_ctx instead of
// the term-absent one, still converges within the existing four-pass
// budget for a representative driver granule (kM9KvBlockTokens itself,
// 32 -- comfortably under the proven 84-page bound).
TEST(fit_context_packed_values_output_still_converges_through_the_auto_fit_backoff) {
    const FitTerms            base = m9_base_terms();
    const PackedValuesFitTerm term = fit_context_packed_values(
        base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens);

    long long   n_ctx           = term.fit.max_ctx;
    size_t      spare_cap       = static_cast<size_t>(-1);
    long long   overshoot_accum = 0;
    const long long budget      = static_cast<long long>(term.fit.reserved_total);
    const size_t    granule     = static_cast<size_t>(kM9KvBlockTokens);
    // A synthetic driver-rounding ceiling one granule below the accepted
    // pool -- the same worst-case construction converge_passes (above)
    // uses, so pass 0 always needs at least one correction.
    const size_t ceiling = simulate_driver_pages(static_cast<size_t>(budget), granule) - 1;

    int converged_at = -1;
    for (int pass = 0; pass < 4; ++pass) {
        const size_t     live_blocks  = static_cast<size_t>(n_ctx);
        // Round-3 review, finding 5: `std::min<size_t>(0, spare_cap)` was
        // dead -- it is 0 for every `spare_cap` (a size_t, always >= 0),
        // so the whole call could have been written `capped_spare = 0`
        // directly. A real wanted-spare, the same "the cache always wants
        // more" constant converge_passes and sub_page_overshoot's own loop
        // use above (kConvergeWantedSpare / 1000000000), exercises
        // pool_sizing's spare arithmetic instead of trivially bypassing it.
        const size_t     capped_spare = std::min(kConvergeWantedSpare, spare_cap);
        const long long  budget_remaining = budget - overshoot_accum;
        const PoolSizing sizing =
            pool_sizing(live_blocks, capped_spare, budget_remaining, /*kv_block_bytes=*/1);
        const size_t observed = simulate_driver_pages(sizing.blocks, granule);
        if (observed <= ceiling) {
            converged_at = pass;
            break;
        }
        const uint64_t over = observed - ceiling;
        overshoot_accum += static_cast<long long>(over);
        const AutoFitTrim trim =
            auto_fit_trim(n_ctx, sizing.spare_blocks, over, /*kv_block_bytes=*/1, /*lanes=*/1,
                         /*kv_block_tokens=*/1, /*n_ctx_floor=*/1, pass, /*protect_spare=*/false);
        if (trim.refuse) break;
        n_ctx     = trim.next_n_ctx;
        spare_cap = trim.next_spare_cap;
    }
    CHECK(converged_at >= 0);
}

// ---------------------------------------------- 0015 engine side (bounded partials)

// Design note o-0015-design.md, section (C): a patched GPU plugin exposes
// an RW config key, PAGED_ATTENTION_MAX_PARTITIONS, bounding how many
// 256-token partitions the mixed-stage paged-attention kernel covers, and
// ships a corrected (f16, not f32) host sizing of the buffer this term
// prices whenever the key is accepted at all. Generalized here: the low-
// level formulas now take `partitions` and `element_bytes` directly rather
// than deriving `partitions = ceil(n_ctx/256)` and hardcoding
// `element_bytes = 4` unconditionally.
//
// RED FIRST: `packed_values_bounded_partitions`, `packed_values_prefill_
// scratch_bytes_ex`, `prefill_chunk_cap_for_packed_values_ex`, and
// `fit_context_packed_values`'s two new trailing parameters do not exist
// before this change -- reverting this milestone's diff against this test
// file is a compile error, the same convention this file already uses
// (e.g. auto_fit_converges_within_four_passes_for_granules_up_to_84_and_
// refuses_beyond's own "RED before auto_fit_trim exists").

// packed_values_bounded_partitions: min(ceil(n_ctx/256), max_partitions)
// when max_partitions > 0, else the unbounded ceil(n_ctx/256) this file
// always computed before 0015. Red-first verified by temporarily
// reverting the function's body to `return raw;` unconditionally (ignoring
// `max_partitions`), rebuilding and re-running the full suite -- the bound
// not propagating breaks every layer built on top of it, not just this
// function's own test, captured verbatim:
//   FAIL packed_values_bounded_partitions(171312, 32) == 32ll / left: 670 / right: 32
//   FAIL packed_values_bounded_partitions_matches_the_design_notes_own_numbers
//   FAIL partitions == 32ll / left: 670 / right: 32
//   FAIL bytes == 50331648ull / left: 1053818880 / right: 50331648
//   FAIL packed_values_prefill_scratch_bytes_ex_n32_at_171312_chunk128_is_48_mib
//   FAIL capped == 128 / left: 64 / right: 128
//   FAIL prefill_chunk_cap_packed_values_ex_bounded_does_not_shrink_where_unbounded_would
//   FAIL bounded.chunk == 128 / left: 32 / right: 128
//   FAIL bounded.iterations == 1 / left: 2 / right: 1
//   FAIL bounded.fixed_bytes == 50331648ull / left: 12582912 / right: 50331648
//   FAIL bounded.fit.max_ctx == 1643264 / left: 1647456 / right: 1643264
//   FAIL fit_context_packed_values_bounded_partitions_keeps_chunk_128
//   364 cases run, 4 failed
// restored right after (364 cases run, 0 failed).
TEST(packed_values_bounded_partitions_matches_the_design_notes_own_numbers) {
    CHECK_EQ(packed_values_bounded_partitions(171312, /*max_partitions=*/32), 32ll);
    // Below the bound: the raw, unbounded count wins -- the bound has not
    // been reached yet, so this is identical to today's ceil(n_ctx/256).
    CHECK_EQ(packed_values_bounded_partitions(1000, /*max_partitions=*/32), 4ll);  // ceil(1000/256)=4
    // max_partitions <= 0: unbounded, byte for byte what this file always
    // computed -- 0 is "no bound", not "zero partitions" (the plugin key's
    // own default convention).
    CHECK_EQ(packed_values_bounded_partitions(171312, /*max_partitions=*/0), 670ll);
    CHECK_EQ(packed_values_bounded_partitions(171312, /*max_partitions=*/-5), 670ll);
    // Nothing to price.
    CHECK_EQ(packed_values_bounded_partitions(0, /*max_partitions=*/32), 0ll);
    CHECK_EQ(packed_values_bounded_partitions(-100, /*max_partitions=*/32), 0ll);
}

// packed_values_prefill_scratch_bytes_ex: the coordinator's own worked
// example, N=32 at n_ctx 171,312, chunk 128, the coder shape (16 query
// heads, 256 head_dim), 2-byte elements (the plugin's f16-corrected
// sizing): 32 partitions x 128 x 16 x 256 x 2 = 128*16=2,048; *256=524,288;
// *2=1,048,576; *32=33,554,432 bytes = 32 MiB exactly. x1.5 (the measured
// overlap bound, unchanged by 0015): 50,331,648 bytes = 48 MiB exactly.
TEST(packed_values_prefill_scratch_bytes_ex_n32_at_171312_chunk128_is_48_mib) {
    const long long partitions = packed_values_bounded_partitions(171312, /*max_partitions=*/32);
    CHECK_EQ(partitions, 32ll);
    const uint64_t bytes = packed_values_prefill_scratch_bytes_ex(
        128, partitions, kCoderHeads, kCoderHeadSize, /*element_bytes=*/2);
    CHECK_EQ(bytes, 50331648ull);
    CHECK_EQ(bytes, 32ull * kMiB + 16ull * kMiB);  // 48 MiB, written as 32 + 16 to show the 1.5x
}

// Regression: `_ex` at element_bytes 4 and unbounded partitions must
// reproduce the plain wrapper byte for byte -- the two already-established
// belt-side fixtures above (131,072 and 171,312 tokens).
TEST(packed_values_prefill_scratch_bytes_ex_matches_the_old_wrapper_at_element_4) {
    for (const long long n_ctx : {131072ll, 171312ll}) {
        const long long partitions = (n_ctx + 255) / 256;
        CHECK_EQ(packed_values_prefill_scratch_bytes_ex(128, partitions, kCoderHeads,
                                                        kCoderHeadSize, /*element_bytes=*/4),
                 packed_values_prefill_scratch_bytes(128, n_ctx, kCoderHeads, kCoderHeadSize));
    }
}

// prefill_chunk_cap_for_packed_values_ex: the belt must price the SAME
// bounded buffer the term above charges, or it halves the chunk for a
// buffer that no longer exists (design note §C, quoted in this function's
// own fit.h comment). Contrasted directly against the UNBOUNDED belt at
// the identical depth, which this file already established shrinks 128 to
// 32 at 171,312 tokens (prefill_chunk_cap_packed_values_171312_caps_to_32,
// above) -- exactly the buffer-that-no-longer-exists case the bound-aware
// refactor exists to avoid.
TEST(prefill_chunk_cap_packed_values_ex_bounded_does_not_shrink_where_unbounded_would) {
    // Unbounded (today, element 4): the belt DOES shrink at this depth.
    CHECK_EQ(prefill_chunk_cap_for_packed_values(128, 171312, kCoderHeads, kCoderHeadSize,
                                                 kScratchBudget, kCoderKvBlockSize),
             32);
    // Bounded (0015, N=32, element 2): the same depth's buffer is capped at
    // 32 partitions and half the element width -- 48 MiB (the test above),
    // comfortably under the 512 MiB proxy budget at chunk 128 -- so the
    // belt never needs to shrink it at all.
    const long long partitions = packed_values_bounded_partitions(171312, /*max_partitions=*/32);
    const int capped = prefill_chunk_cap_for_packed_values_ex(
        128, partitions, kCoderHeads, kCoderHeadSize, /*element_bytes=*/2, kScratchBudget,
        kCoderKvBlockSize);
    CHECK_EQ(capped, 128);
}

// Regression: `_ex` at element_bytes 4 and unbounded partitions must
// reproduce the plain wrapper byte for byte.
TEST(prefill_chunk_cap_for_packed_values_ex_matches_the_old_wrapper_at_element_4) {
    for (const long long n_ctx : {131072ll, 171312ll, 119074ll, 71689ll}) {
        const long long partitions = (n_ctx + 255) / 256;
        CHECK_EQ(prefill_chunk_cap_for_packed_values_ex(128, partitions, kCoderHeads,
                                                        kCoderHeadSize, /*element_bytes=*/4,
                                                        kScratchBudget, kCoderKvBlockSize),
                 prefill_chunk_cap_for_packed_values(128, n_ctx, kCoderHeads, kCoderHeadSize,
                                                     kScratchBudget, kCoderKvBlockSize));
    }
}

// The end-to-end climb, bounded: the m9_base_terms() fixture already
// established above (fit_context_packed_values_drops_the_auto_fit_depth_
// versus_the_term_absent_fit: term absent -> 1,648,832; UNBOUNDED term ->
// 1,227,936, chunk shrinks to 32). Bounded at N=32, 2-byte elements -- the
// coordinator's own worked tmp_out figure, 48 MiB, plus round-6 review
// finding 4's exp_sums/max_logits pair AT the bound (2 x 128 x 16 x 4 x 32
// = 512 KiB), lanes-multiplied (finding 1 -- one lane here, so the
// multiplier is a no-op arithmetically but the code path is real): 48 MiB
// + 512 KiB = 50,855,936 bytes. No per-token growth at all
// (packed_values_bounded_partitions caps at 8,192 tokens, far below any
// depth this budget could possibly reach) -- the fixed point never needs
// the belt to shrink anything: chunk STAYS 128, in one round.
TEST(fit_context_packed_values_bounded_partitions_keeps_chunk_128) {
    const FitTerms base = m9_base_terms();

    // RED, by direct comparison: the UNBOUNDED call on this exact fixture
    // (already established above) shrinks chunk to 32 and settles at
    // 1,227,936 -- reproduced here so the contrast is in the same test.
    const PackedValuesFitTerm unbounded = fit_context_packed_values(
        base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens);
    CHECK_EQ(unbounded.chunk, 32);
    CHECK_EQ(unbounded.fit.max_ctx, 1227936);

    // GREEN: bounded (max_partitions=32, element_bytes=2).
    const PackedValuesFitTerm bounded = fit_context_packed_values(
        base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens,
        /*belt_enabled=*/true, /*max_partitions=*/32, /*element_bytes=*/2);
    CHECK_EQ(bounded.chunk, 128);           // the belt never needed to shrink it
    CHECK_EQ(bounded.iterations, 1);        // converges in one round -- 128 already fits
    CHECK_EQ(bounded.per_token_bytes, 0ull);  // bounded: no per-token slope, the whole
                                              // charge is a flat, depth-independent amount
    CHECK_EQ(bounded.fixed_bytes, 50855936ull);  // 48 MiB tmp_out + 512 KiB exp_sums/max_logits
    CHECK_EQ(bounded.fit.max_ctx, 1643200);
    CHECK(bounded.fit.max_ctx > unbounded.fit.max_ctx);  // bounding the term recovers depth
    CHECK(bounded.fit.reserved_total <= kM9Total);
    CHECK(bounded.fit.admissible);
}

// Round-6 review, finding 1's own lanes multiplier, isolated: two lanes
// must double the BOUNDED fixed charge (the per-request buffer, times
// lanes) while leaving the belt's own chunk choice and the UNBOUNDED
// path's tiny per-token margin alone -- lanes never entered that arm's
// arithmetic and nothing here asks it to. Red-first verified by
// temporarily reverting `fixed = per_lane * lanes_c;` to `fixed =
// per_lane;` (no lanes) -- rebuilt:
//   FAIL two.fixed_bytes == 2ull * 50855936ull / left: 50855936 / right: 101711872
//   FAIL fit_context_packed_values_bounded_fixed_bytes_scale_with_lanes
//   365 cases run, 1 failed
// restored right after (365 cases run, 0 failed).
TEST(fit_context_packed_values_bounded_fixed_bytes_scale_with_lanes) {
    FitTerms two_lanes  = m9_base_terms();
    two_lanes.lanes      = 2;
    const PackedValuesFitTerm one = fit_context_packed_values(
        m9_base_terms(), kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kM9KvBlockTokens, /*belt_enabled=*/true, /*max_partitions=*/32, /*element_bytes=*/2);
    const PackedValuesFitTerm two = fit_context_packed_values(
        two_lanes, kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kM9KvBlockTokens, /*belt_enabled=*/true, /*max_partitions=*/32, /*element_bytes=*/2);
    CHECK_EQ(one.fixed_bytes, 50855936ull);
    CHECK_EQ(two.fixed_bytes, 2ull * 50855936ull);
    CHECK_EQ(two.per_token_bytes, 0ull);
}

// "Key absent": nothing changes. This is the ONLY testable surface for
// that claim without a card -- the real, on-card baseline this repository
// records (CHANGELOG.md, "Re-measured with the narrowed ceiling") is
// n_ctx 101,824, from real total/weights/activation/margin figures this
// file's synthetic fixture does not have and cannot reproduce; what this
// test verifies instead is that the ENGINE'S OWN arithmetic for the
// key-absent path (max_partitions defaulted to 0, element_bytes defaulted
// to 4 -- exactly what backend_ov.cpp passes when `paged_attention_bound_
// accepted_` is false) is byte-for-byte identical to the pre-0015,
// already-committed unbounded path this file already tests above.
TEST(fit_context_packed_values_key_absent_reproduces_the_unbounded_path_exactly) {
    const FitTerms base = m9_base_terms();

    const PackedValuesFitTerm implicit_default = fit_context_packed_values(
        base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens);
    const PackedValuesFitTerm explicit_key_absent = fit_context_packed_values(
        base, kM9RequestedChunk, kCoderHeads, kCoderHeadSize, kScratchBudget, kM9KvBlockTokens,
        /*belt_enabled=*/true, /*max_partitions=*/0, /*element_bytes=*/4);

    CHECK_EQ(explicit_key_absent.chunk, implicit_default.chunk);
    CHECK_EQ(explicit_key_absent.per_token_bytes, implicit_default.per_token_bytes);
    CHECK_EQ(explicit_key_absent.fixed_bytes, implicit_default.fixed_bytes);
    CHECK_EQ(explicit_key_absent.fit.max_ctx, implicit_default.fit.max_ctx);
    // Pinned to the figure this file already established above (fit_
    // context_packed_values_drops_the_auto_fit_depth_versus_the_term_
    // absent_fit) -- a regression in either the defaults or the explicit
    // key-absent path moves this number, and this assertion is what would
    // catch it.
    CHECK_EQ(explicit_key_absent.fit.max_ctx, 1227936);
    CHECK_EQ(explicit_key_absent.chunk, 32);
}

// ------------------------ packed_values_scratch_reservation_bytes / phase_e_ceiling_bytes

// Round-3 review, finding 2 (REAL defect): round-2 review's own F4 comment
// claimed the ceiling check "already reflects" the packed-values term
// because "total itself was sized net of the term back at the climb" --
// false. `total` (backend_ov.cpp's `device_total_mem_size`) is a constant
// read once at load and never adjusted for anything the climb charges (the
// climb shrinks `max_ctx`, not `total`), so a bare `ceiling = total -
// margin` never held the term back at all -- an observed residency landing
// just under THAT ceiling was accepted, the pool kept every byte, and the
// scratch buffer (not yet allocated at this point) still had to fit in
// whatever was left once the first real prefill grew it: the original
// fault this whole term exists to prevent, reintroduced silently.
//
// Round numbers, not a card reading -- only the ceiling arithmetic is
// under test here. `packed_values_scratch_reservation_bytes` is exercised
// first on its own (also closes round-3 review, finding 4's regression:
// an earlier version of backend_ov.cpp's own summary log line multiplied
// the per-token part by `max_ctx` alone, omitting `lanes` -- this shared
// formula cannot omit it, since every caller, log line included, now
// routes through the same function).
namespace {
constexpr uint64_t kF2Total       = 1000000000ull;  // 1 GB
constexpr uint64_t kF2Margin      = 100000000ull;    // 100 MB
constexpr uint64_t kF2Fixed       = 10000000ull;     // 10 MB, the one-partition margin
constexpr uint64_t kF2PerToken    = 1000ull;         // 1,000 B/token
constexpr long long kF2PagedNCtx  = 50000;           // the FINAL served depth this pass
// scratch_term = fixed + lanes * n_ctx * per_token = 10,000,000 +
// 1 * 50,000 * 1,000 = 60,000,000 (one lane).
constexpr uint64_t kF2ScratchTermOneLane = 60000000ull;
// old_ceiling = total - margin = 900,000,000; new_ceiling = old_ceiling -
// scratch_term = 840,000,000. `kF2Observed` sits strictly BETWEEN the two
// -- inside the term's own size (60,000,000) below the old ceiling --
// which is exactly the band the old ceiling accepted and the new one must
// not.
constexpr uint64_t kF2Observed = 860000000ull;
}  // namespace

TEST(packed_values_scratch_reservation_bytes_multiplies_the_per_token_part_by_lanes) {
    CHECK_EQ(packed_values_scratch_reservation_bytes(kF2Fixed, kF2PerToken, /*lanes=*/1,
                                                     kF2PagedNCtx),
             kF2ScratchTermOneLane);
    // Round-3 review, finding 4's own regression, reproduced directly: two
    // lanes must double the per-token contribution, not leave it as if
    // there were still one.
    CHECK_EQ(packed_values_scratch_reservation_bytes(kF2Fixed, kF2PerToken, /*lanes=*/2,
                                                     kF2PagedNCtx),
             kF2Fixed + 2ull * static_cast<uint64_t>(kF2PagedNCtx) * kF2PerToken);
    // Zero/negative n_ctx or lanes: no per-token contribution, just the
    // fixed margin (mirrors fit_context's own std::max(1, lanes) floor and
    // shrink_n_ctx's "never negative" convention elsewhere in this file).
    CHECK_EQ(packed_values_scratch_reservation_bytes(kF2Fixed, kF2PerToken, /*lanes=*/1,
                                                     /*n_ctx=*/0),
             kF2Fixed);
    CHECK_EQ(packed_values_scratch_reservation_bytes(kF2Fixed, kF2PerToken, /*lanes=*/0,
                                                     kF2PagedNCtx),
             kF2ScratchTermOneLane);  // lanes floors to 1, same as one lane
}

// RED (documentation, not a compile-time or runtime failure against THIS
// file's own `phase_e_ceiling_bytes` -- against the OLD, pre-fix ceiling
// formula backend_ov.cpp used to compute inline, `total - margin`): the
// old ceiling does not even see the scratch term, so an observed residency
// well inside the term's own size still reads as "fits."
TEST(phase_e_ceiling_old_formula_accepts_an_observed_inside_the_scratch_term) {
    const uint64_t old_ceiling = kF2Total > kF2Margin ? kF2Total - kF2Margin : 0;
    CHECK_EQ(old_ceiling, 900000000ull);
    CHECK(kF2Observed <= old_ceiling);  // the old logic: NOT an overshoot
}

// GREEN: `phase_e_ceiling_bytes` (fit.h) subtracts the scratch term (priced
// at the FINAL served depth, via packed_values_scratch_reservation_bytes)
// from the ceiling -- the SAME `kF2Observed` that passed the old formula
// above now correctly overshoots the new, narrower one.
TEST(phase_e_ceiling_bytes_subtracts_the_scratch_term_and_catches_the_overshoot) {
    const uint64_t scratch_term =
        packed_values_scratch_reservation_bytes(kF2Fixed, kF2PerToken, /*lanes=*/1, kF2PagedNCtx);
    CHECK_EQ(scratch_term, kF2ScratchTermOneLane);

    const uint64_t new_ceiling = phase_e_ceiling_bytes(kF2Total, kF2Margin, scratch_term);
    CHECK_EQ(new_ceiling, 840000000ull);
    CHECK(new_ceiling < (kF2Total > kF2Margin ? kF2Total - kF2Margin : 0));  // strictly narrower

    CHECK(kF2Observed > new_ceiling);  // now correctly flagged as overshoot
}

// u8/f16 (or any load where the climb never charged the term): both
// scratch members default to 0, so the shared formula returns 0 and the
// ceiling is untouched -- `phase_e_ceiling_bytes` with a 0 scratch term
// reproduces the bare `total - margin` exactly, byte for byte.
TEST(phase_e_ceiling_bytes_zero_scratch_term_matches_the_bare_total_minus_margin) {
    CHECK_EQ(phase_e_ceiling_bytes(kF2Total, kF2Margin, /*scratch_term_bytes=*/0),
             kF2Total - kF2Margin);
}

// Round-4 review, finding 3: the floor. `margin + scratch_term_bytes >=
// total` means there is no room left for anything else at all -- a scratch
// term big enough (a very deep pool, or a chunk the belt could not shrink
// far enough) can genuinely exceed what `margin` alone left over. Must
// floor at 0, not underflow a `uint64_t` subtraction into a number in the
// exabytes (the same failure mode fit_context's own budget-underflow guard
// and shrink_n_ctx's floor exist to avoid elsewhere in this file).
TEST(phase_e_ceiling_bytes_floors_at_zero_when_margin_plus_term_reaches_total) {
    // Exactly at the boundary: margin + term == total.
    CHECK_EQ(phase_e_ceiling_bytes(kF2Total, kF2Margin, kF2Total - kF2Margin), 0ull);
    // Past the boundary: margin + term > total.
    CHECK_EQ(phase_e_ceiling_bytes(kF2Total, kF2Margin, kF2Total), 0ull);
    CHECK_EQ(phase_e_ceiling_bytes(kF2Total, kF2Margin, kF2Total * 2), 0ull);
    // One byte short of the boundary: still strictly positive, not
    // over-eagerly floored.
    CHECK_EQ(phase_e_ceiling_bytes(kF2Total, kF2Margin, kF2Total - kF2Margin - 1), 1ull);
}

// --------------------- round-7 review, item A: activation charged at the wrong chunk

// MEASURED ON CARD (unpatched plugin, coder, u8:i4, --n-ctx omitted vs an
// explicit --n-ctx equal to what auto-fit itself adopted): auto-fit lands
// on n_ctx 101,824 at belt chunk 64: the log line
// "prefill scratch charged 599 MiB at chunk 64 (per token 6.0 KiB)".
// The EXPLICIT request for the SAME 101,824, though, was refused: "activation
// fit: ... served chunk 1024 measured 0.25 GiB", then "prefill scratch
// charged 718.7 MiB at chunk 128 for n_ctx 61072 (per token 12.0 KiB + KV
// 8.8 KiB)", then "requested n_ctx 101824 ... needs 0.86 GiB of KV but the
// reservation admits 61072 per lane (... activations 0.25 ...)". The
// activation-probe climb (backend_ov.cpp, well before the packed-values
// fit runs) has its own budget check keyed on `wanted` alone -- it has
// never heard of the packed-values scratch term -- so on the explicit path
// (`wanted` = the request itself, smaller than the artifact's own train
// max auto-fit uses for that same check) it can climb activations to a
// FAR bigger chunk (1024) than the packed-values belt will actually serve
// (128), and the reservation is built from THAT stale, larger footprint.
//
// This is not a pure fit.h bug -- `fit_context` and `fit_context_packed_
// values` are unchanged by the round-7 review fix, which re-probes
// activations at the belt's own served chunk in backend_ov.cpp (a real
// GPU forward pass, not unit-testable without a card, per this round's own
// "no ssh" constraint). What IS testable here, and is the fixture this
// section exists to pin down, is the underlying arithmetic the fix
// depends on: charging `fit_context` the STALE (larger-chunk) activation
// figure, rather than the CORRECT (served-chunk) one, is what turns an
// admissible depth into a refused one -- the exact shape of the measured
// defect, reproduced with round numbers rather than the card's own
// unlabelled totals (which this file has no way to reconstruct without
// the real total/weights/margin the card run used).
namespace {
constexpr uint64_t kA7Total          = 1000000000ull;  // 1 GB
constexpr uint64_t kA7Weights        = 200000000ull;
constexpr uint64_t kA7Margin         = 50000000ull;
constexpr uint64_t kA7KvBytesToken   = 1000ull;  // 1 KB/token, round
constexpr int       kA7NCtxFloor     = 4096;
// The depth an explicit --n-ctx would ask for -- equal to what auto-fit
// itself would have adopted, per this round's own requirement ("an
// explicit --n-ctx equal to what auto-fit would adopt must be admitted").
constexpr long long kA7RequestedNCtx = 500000;
// Activations measured at the STALE, activation-climb-only chunk (1024 in
// the card trace) vs the CORRECT, served chunk (128 in the card trace) --
// round numbers standing in for "bigger" and "smaller," not the card's
// own 0.25 GiB (this fixture cannot reconstruct the real total/weights the
// card run used, so it is not trying to reproduce 101,824 or 61,072
// exactly, only the SHAPE of the defect: stale-larger refuses, correct-
// smaller admits, for the identical requested depth).
constexpr uint64_t kA7ActivationStale   = 300000000ull;
constexpr uint64_t kA7ActivationCorrect = 50000000ull;

FitTerms a7_terms(uint64_t activations) {
    FitTerms t;
    t.total          = kA7Total;
    t.weights        = kA7Weights;
    t.activations    = activations;
    t.margin         = kA7Margin;
    t.kv_bytes_token = kA7KvBytesToken;
    t.lanes           = 1;
    t.kv_block_tokens = 1;
    t.n_ctx_floor     = kA7NCtxFloor;
    return t;
}
}  // namespace

// RED: charged at the STALE (larger, pre-belt) chunk's own measured
// activation footprint -- exactly what the pre-round-7 code did on the
// explicit path -- the requested depth is refused.
TEST(activation_charged_at_stale_chunk_refuses_a_depth_auto_fit_would_admit) {
    const FitResult stale = fit_context(a7_terms(kA7ActivationStale));
    // fixed = 200,000,000 + 300,000,000 + 50,000,000 = 550,000,000.
    // budget = 1,000,000,000 - 550,000,000 = 450,000,000.
    // max_ctx = 450,000,000 / 1,000 = 450,000 -- BELOW the 500,000 requested.
    CHECK_EQ(stale.max_ctx, 450000);
    CHECK(stale.max_ctx < kA7RequestedNCtx);
    CHECK(!stale.admissible || stale.max_ctx < kA7RequestedNCtx);  // refuses this depth either way
}

// GREEN: charged at the CORRECT (served, post-belt) chunk's own measured
// activation footprint -- what round-7's re-probe fix produces -- the
// IDENTICAL requested depth is admissible.
TEST(activation_charged_at_the_served_chunk_admits_the_same_depth) {
    const FitResult correct = fit_context(a7_terms(kA7ActivationCorrect));
    // fixed = 200,000,000 + 50,000,000 + 50,000,000 = 300,000,000.
    // budget = 1,000,000,000 - 300,000,000 = 700,000,000.
    // max_ctx = 700,000,000 / 1,000 = 700,000 -- AT OR ABOVE the request.
    CHECK_EQ(correct.max_ctx, 700000);
    CHECK(correct.max_ctx >= kA7RequestedNCtx);
    CHECK(correct.admissible);
}

// --------------------------- round-8/9 review: evaluate at D, not at the search's own fixed point

// MEASURED ON CARD (round-7's binary, unpatched plugin, coder, u8:i4,
// explicit --n-ctx 101824 -- the SAME depth auto-fit itself adopts):
// STILL refused. "4-bit values: prefill scratch charged 720.2 MiB at
// chunk 128 for n_ctx 60880 (per token 12.1 KiB + KV 8.8 KiB)" -- the
// round-7 fix (re-probe at the belt's smaller chunk) could not work: this
// file's own record is that the plugin's intermediate pool grows to the
// largest shape it has ever seen and never shrinks, so a probe taken
// after a chunk-1024 probe returns the STALE, bigger reading regardless
// of what chunk is asked for next -- the round-7 loop was a no-op.
//
// The deeper bug round-8 review actually fixes: `fit_context_packed_
// values` (the auto-fit SEARCH) was being called on the explicit path
// too, seeded from whatever chunk the activation-probe climb explored
// (1024, per the card's own earlier trace) -- and it converged to A
// self-consistent (chunk, max_ctx) pair that has nothing to do with the
// REQUESTED depth (101,824) at all. `fit_context_packed_values_at_depth`
// (fit.h) replaces this for BOTH paths (round-9 review, finding 1's own
// retraction: auto-fit does not need a search here either -- the belt's
// chunk choice reads only depth and geometry, so `wanted` is exactly as
// evaluable up front as an explicit request's own depth): evaluate the
// belt AT the depth in question directly (398 partitions at 101,824 ->
// chunk 64, pure geometry, no activation dependence), price the term
// there, and check admissibility.
//
// Engineered fixture, the stated constants (weights 12.83 GiB, KV
// ~8.8 KiB/token, a 16 GiB card, 256 MiB margin, the coder's 16 x 256
// heads, the 512 MiB belt proxy budget, requested depth 101,824) --
// `kR8ActivationStale`/`kR8ActivationCorrect` are this file's own
// engineered numbers (this repository was not given the card's real
// activation reading): `kR8ActivationStale` stands in for activation
// measured at the climb's own uncapped chunk (1024, as the card trace
// showed) -- the shape the OLD (pre-round-8) code produced; `kR8Activation
// Correct` for activation measured at the depth's OWN served chunk (64)
// -- the re-ordering round-8's fix (backend_ov.cpp: determine chunk
// before the one real probe) makes true.
namespace {
constexpr uint64_t kR8Total        = 16ull * kGiB;
constexpr uint64_t kR8Weights      = 13776107601ull;  // 12.83 GiB, exact
constexpr uint64_t kR8Margin       = 256ull * kMiB;
constexpr uint64_t kR8KvBytesToken = 9011ull;  // ~8.8 KiB/token
constexpr int       kR8KvBlockTokens = 32;
constexpr long long kR8RequestedD    = 101824;
constexpr uint64_t kR8ActivationStale   = 1800ull * kMiB;  // at the climb's own uncapped chunk
constexpr uint64_t kR8ActivationCorrect = 268435456ull;    // 0.25 GiB, at the served chunk

FitTerms r8_terms(uint64_t activation) {
    FitTerms t;
    t.total          = kR8Total;
    t.weights        = kR8Weights;
    t.activations    = activation;
    t.margin         = kR8Margin;
    t.kv_bytes_token = kR8KvBytesToken;
    t.lanes           = 1;
    t.kv_block_tokens = kR8KvBlockTokens;
    t.n_ctx_floor     = 4096;
    return t;
}
}  // namespace

// CONTRAST, not a regression test of THIS round's own diff: `fit_context_
// packed_values` itself is unchanged. What it documents is WHY the
// pre-round-8 call site was wrong to use this function (the auto-fit
// SEARCH) on the explicit path at all: seeded from the activation
// climb's uncapped chunk (1024, matching the card's own trace), it
// self-consistently settles at chunk 64 and max_ctx 82,048 -- a fixed
// point that has nothing to do with the requested 101,824, and refuses
// it.
// Round-11 review (Opus, RETRACTS round-10's margined-`fits()` numbers
// for this fixture): `fits()` is restored to the raw proxy
// (kPrefillScratchBudgetBytesPackedValues's own calibration, see
// prefill_chunk_cap_packed_values_131072_shrinks_128_to_64's own
// comment). At `kR8ActivationStale` (1,800 MiB) seeded from chunk 1,024,
// the search now settles at chunk 128 (the hard-measured-cap ceiling,
// `kMaxMeasuredPackedValuesChunk`) and max_ctx 58,240, not chunk 64 -- a
// different fixed point, still nothing to do with the requested 101,824.
TEST(explicit_n_ctx_old_call_site_shape_self_consistent_search_refuses_the_requested_depth) {
    const FitTerms base = r8_terms(kR8ActivationStale);
    const PackedValuesFitTerm old_search = fit_context_packed_values(
        base, /*requested_chunk=*/1024, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kR8KvBlockTokens);
    CHECK_EQ(old_search.chunk, 128);
    CHECK_EQ(old_search.fit.max_ctx, 58240);
    CHECK(old_search.fit.max_ctx < kR8RequestedD);  // refuses 101,824
}

// GREEN: evaluate the belt AT the requested depth directly (no search),
// price the term there, and check that depth's own admissibility.
// Round-11 review (retracts round-10's margined-`fits()` number below):
// with the raw proxy restored, 398 partitions at depth 101,824 fits
// chunk 64 (351 MiB raw, under the 512 MiB budget) -- the belt lands on
// 64, the same chunk the card itself measured and served at this depth.
TEST(explicit_n_ctx_new_order_evaluate_at_depth_admits_the_requested_depth) {
    const FitTerms base = r8_terms(kR8ActivationCorrect);
    const PackedValuesFitTerm at_depth = fit_context_packed_values_at_depth(
        base, /*requested_chunk=*/1024, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kR8KvBlockTokens, kR8RequestedD, /*max_partitions=*/0, /*element_bytes=*/4);
    CHECK_EQ(at_depth.chunk, 64);
    CHECK_EQ(at_depth.fixed_bytes, 629260288ull);  // the term AT (chunk 64, depth 101,824)
    CHECK_EQ(at_depth.fit.max_ctx, 248320);
    CHECK(at_depth.fit.max_ctx >= kR8RequestedD);  // admits 101,824
    CHECK(at_depth.fit.admissible);
}

// Round-14 review (Opus), finding 1: `ARCINT_PREFILL_CHUNK_CAP=off` on
// the EXPLICIT --n-ctx path -- `belt_enabled=false` -- exactly what the
// auto-fit search already does via its own `belt_enabled` (this is the
// SAME real function backend_ov.cpp's explicit call site threads
// `cap_off` into, not a second implementation of the switch).
//
// Round-15 review (Opus, REAL defect, RETRACTS this test's own original
// premise -- kept on the record, per DESIGN §7.0.1): a card measurement
// with `ARCINT_PREFILL_CHUNK_CAP=off --n-ctx 101824 --prefill-chunk 128`
// showed the load REFUSED (1,200.2 MiB still charged at chunk 128, only
// 32,256 admitted) even with the belt and cap both bypassed -- the
// depth-scaled term alone, still priced at the uncapped chunk, was
// enough. `belt_enabled=false` now charges NOTHING here too (see
// `fit_context_packed_values_at_depth`'s own retraction comment): the
// requested chunk (2048) is still what gets served, but the fit runs as
// if the packed-values term did not exist, so a plain `fit_context(base)`
// is what admits or refuses -- not this file's own estimate of a buffer
// whose fault line is the whole reason the switch exists.
TEST(explicit_n_ctx_at_depth_belt_disabled_charges_nothing) {
    const FitTerms base = r8_terms(kR8ActivationCorrect);
    const PackedValuesFitTerm at_depth = fit_context_packed_values_at_depth(
        base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kR8KvBlockTokens, /*depth=*/102384, /*max_partitions=*/0, /*element_bytes=*/4,
        /*belt_enabled=*/false);
    const FitResult plain = fit_context(base);
    CHECK_EQ(at_depth.chunk, 2048);  // pinned, uncapped by budget OR the measured ceiling --
                                     // still what the load actually serves
    CHECK_EQ(at_depth.iterations, 1);
    CHECK_EQ(at_depth.fixed_bytes, 0ull);
    CHECK_EQ(at_depth.per_token_bytes, 0ull);
    CHECK_EQ(at_depth.fit.max_ctx, plain.max_ctx);  // exactly the term-free fit
}

// n_ctx 165,680 (10,355 pages of 16), the BOUNDED-arm reading: the depth
// the card's own auto-fit adopted under the patched plugin with
// --paged-attention-max-partitions 32 (CHANGELOG's own MEASURED entry:
// "auto-fit lands at n_ctx 165,680 with chunk 128, 48.5 MiB of scratch
// charged"). `r8_terms()` cannot be reused as-is here -- it fixes
// `kv_block_tokens` at 32 (`kR8KvBlockTokens`), and 165,680 is not a
// multiple of 32 (165,680 / 32 = 5,177.5; it IS a multiple of 16, the
// card's own --kv-block-size for this measurement) -- so the fixture is
// built inline with the SAME weights/total/margin/KV-bytes-per-token the
// rest of this file calls "the card constants," `kv_block_tokens = 16`.
// The bound caps partitions at 32 regardless of depth once
// ceil(n_ctx/256) exceeds it (ceil(165680/256) = 648 > 32), so the proxy
// the belt checks is chunk x 16 heads x 256 head_dim x 2 B x 32
// partitions = 32 MiB -- comfortably under the 512 MiB budget -- and the
// belt leaves chunk 128 untouched (no shrink needed at all, unlike the
// unbounded reading of this same depth two tests above, which shrinks to
// 32). The charged term is tmp_out (margined, 2-byte elements) 48.0 MiB
// + exp_sums/max_logits (always 4-byte, unaffected by the element width)
// 0.5 MiB = 48.5 MiB exactly -- the same figure the card logged.
namespace {
FitTerms r8_terms_kv16(uint64_t activation) {
    FitTerms t     = r8_terms(activation);
    t.kv_block_tokens = 16;
    return t;
}
}  // namespace

TEST(explicit_n_ctx_at_165680_bounded_n32_admits_with_the_card_constants) {
    const FitTerms base = r8_terms_kv16(kR8ActivationCorrect);
    const PackedValuesFitTerm at_depth = fit_context_packed_values_at_depth(
        base, /*requested_chunk=*/128, kCoderHeads, kCoderHeadSize, kScratchBudget,
        /*block_size=*/16, /*depth=*/165680, /*max_partitions=*/32, /*element_bytes=*/2);
    CHECK_EQ(at_depth.chunk, 128);              // the bound proxy (32 MiB) never needed to shrink
    CHECK_EQ(at_depth.fixed_bytes, 50855936ull);  // 48.0 MiB tmp_out + 0.5 MiB exp/max = 48.5 MiB,
                                                  // the exact figure the card logged
    CHECK_EQ(at_depth.fit.max_ctx, 312496);
    CHECK(at_depth.fit.max_ctx >= 165680);  // admits the depth the card actually adopted
    CHECK(at_depth.fit.admissible);
}

// The explicit round trip at 165,680, bounded arm: NOT two calls to the
// same function (F3's own standing objection to a trivial round trip) --
// a REAL SEARCH (`fit_context_packed_values`, seeded from the operator's
// own default chunk 2048, NOT 128, so the belt has to do real work to
// arrive there) at an activation level engineered so the search's own
// fixed point settles EXACTLY at max_ctx 165,680, compared against
// `fit_context_packed_values_at_depth` evaluated directly at that SAME
// depth -- the two independent primitives (search vs. exact-at-a-known-
// depth) must agree on both the served chunk and the charged term, the
// same "search vs. at_depth" contract every other round-trip test in
// this file drives through the real functions.
TEST(auto_fit_search_at_165680_bounded_agrees_with_the_explicit_round_trip) {
    // Engineered activation (~1,517.7 MiB) chosen so the bounded search's
    // flat 48.5 MiB term (unaffected by depth, once partitions hit the
    // N=32 bound) leaves a budget that divides out to exactly 165,680 at
    // this fixture's KV bytes/token (9,011) and 16-token KV page.
    constexpr uint64_t kActivationFor165680 = 1591455623ull;
    const FitTerms      base                = r8_terms_kv16(kActivationFor165680);

    const PackedValuesFitTerm search = fit_context_packed_values(
        base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
        /*block_size=*/16, /*belt_enabled=*/true, /*max_partitions=*/32, /*element_bytes=*/2);
    CHECK_EQ(search.chunk, 128);
    CHECK_EQ(search.fit.max_ctx, 165680);
    CHECK_EQ(search.fixed_bytes, 50855936ull);

    const PackedValuesFitTerm resubmitted = fit_context_packed_values_at_depth(
        base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
        /*block_size=*/16, /*depth=*/search.fit.max_ctx, /*max_partitions=*/32,
        /*element_bytes=*/2);
    CHECK_EQ(resubmitted.chunk, search.chunk);           // served == priced
    CHECK_EQ(resubmitted.fit.max_ctx, search.fit.max_ctx);  // the round trip reproduces itself
    CHECK_EQ(resubmitted.fixed_bytes, search.fixed_bytes);
    CHECK(resubmitted.fit.admissible);
}

// Round-10 review, finding 1 (REAL defect, retracts round-9's own claim
// that both paths evaluate at `wanted`): `wanted` is NOT the served
// depth for auto-fit, so evaluating the belt there gives the SMALLEST,
// most conservative chunk (32 at `wanted` = 262,144, requested_chunk
// 2048 -- see the CHECK below), not the one the adopted depth actually
// permits. F1's own required design: run the SEARCH
// (`fit_context_packed_values`) seeded from the operator's own
// requested/default chunk, let it find its own self-consistent depth D*,
// and confirm evaluating directly AT D* (`fit_context_packed_values_at_
// depth`, the SAME primitive the explicit path uses) reproduces the SAME
// chunk.
//
// Round-11 review (Opus), finding 1: backend_ov.cpp's own `prefill_
// chunk_` was found seeding from the activation climb's ceiling-bound
// `chunk` instead of the search's own converged `packed_values_scratch_
// chunk_` (c*) -- so the served chunk stayed pinned at the ceiling no
// matter how much bigger a chunk the search (and the upward re-probe
// that pays for it) legitimately found, making the whole re-probe loop
// pure waste. Fixed at the call site; this test is the fit.h-level proof
// that the real search's own served chunk (what `prefill_chunk_` must
// now be seeded from) is what the reservation was priced at ("served ==
// priced"), at two different activation levels landing on two different
// chunks -- not a single lucky case. `at_depth`, re-evaluated AT each
// search's own D*, reproduces the SAME chunk both times, confirming the
// two primitives agree (F3: real search vs. real at_depth, not two calls
// to the same function). (Round-10's own single-fixture version of this
// test used activation ~1.7 GiB and asserted chunk 64 against a margined
// `fits()`; retracted along with round-10's margined belt -- see
// prefill_chunk_cap_packed_values_131072_shrinks_128_to_64's own comment
// -- and replaced by the two-activation fixture below, against the
// restored raw proxy.)
//
// Round-12 review (Opus): stated plainly, because an earlier summary of
// this test (CHANGELOG.md) did not -- the "settles at chunk 64" result
// below is NOT a property of the card's weights/total/margin constants
// alone; it depends on the SPECIFIC engineered activation figure chosen
// (1,200 MiB, this repository's own invented number, not a card
// reading). The 1,800 MiB case right below it, same card constants,
// settles at chunk 128 instead -- the two cases together are the point
// (the served chunk is a function of activation too, not depth/geometry
// alone the way the BELT's own chunk choice is), not either number in
// isolation.
TEST(auto_fit_search_serves_the_priced_chunk_not_the_wanted_evaluated_ceiling) {
    const int ceiling_at_wanted = prefill_chunk_cap_for_packed_values(
        2048, /*max_ctx_tokens=*/262144, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kR8KvBlockTokens);
    CHECK_EQ(ceiling_at_wanted, 32);  // the conservative starting bound alone

    // activation 1,200 MiB: the search settles at chunk 64.
    {
        const FitTerms base = r8_terms(/*activation=*/1258291200ull);
        const PackedValuesFitTerm search = fit_context_packed_values(
            base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
            kR8KvBlockTokens);
        CHECK_EQ(search.chunk, 64);
        CHECK_EQ(search.fit.max_ctx, 123488);
        CHECK(search.chunk > ceiling_at_wanted);  // escapes the wanted-only ceiling

        const PackedValuesFitTerm reproduced = fit_context_packed_values_at_depth(
            base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
            kR8KvBlockTokens, search.fit.max_ctx, /*max_partitions=*/0, /*element_bytes=*/4);
        CHECK_EQ(reproduced.chunk, search.chunk);  // served == priced
    }

    // activation 1,800 MiB (kR8ActivationStale): shallower budget, fewer
    // partitions needed at the shallower depth the search converges to,
    // so the belt affords a BIGGER chunk here, not a smaller one -- the
    // search settles at chunk 128 (the hard measured-cap ceiling).
    {
        const FitTerms base = r8_terms(kR8ActivationStale);
        const PackedValuesFitTerm search = fit_context_packed_values(
            base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
            kR8KvBlockTokens);
        CHECK_EQ(search.chunk, 128);
        CHECK_EQ(search.fit.max_ctx, 58240);
        CHECK(search.chunk > ceiling_at_wanted);  // escapes the wanted-only ceiling

        const PackedValuesFitTerm reproduced = fit_context_packed_values_at_depth(
            base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
            kR8KvBlockTokens, search.fit.max_ctx, /*max_partitions=*/0, /*element_bytes=*/4);
        CHECK_EQ(reproduced.chunk, search.chunk);  // served == priced
    }
}

// Round-9 review (the card's third refusal): the FIT-LEVEL check alone
// (`depth <= fit.max_ctx`) is not enough -- Phase E's real allocation can
// still overshoot the analytic prediction by a small amount. One KV page
// of slack is subtracted from `max_ctx` before either path's
// admissibility check. Round-10 review, finding 3: this round trip now
// drives the REAL auto-fit path (the search) rather than modeling it as
// `at_depth(wanted)` -- search(requested_chunk=2048) settles at chunk 32,
// max_ctx 236,864; the depth it adopts under one page of slack,
// resubmitted through the SAME slack-adjusted explicit check, is
// admitted.
TEST(auto_fit_adopted_depth_with_slack_is_admissible_as_an_explicit_request) {
    const FitTerms base = r8_terms(kR8ActivationCorrect);
    constexpr long long kSlack = kR8KvBlockTokens;  // one KV page

    const PackedValuesFitTerm auto_fit = fit_context_packed_values(
        base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kR8KvBlockTokens);
    CHECK_EQ(auto_fit.chunk, 32);
    CHECK_EQ(auto_fit.fit.max_ctx, 236864);
    const long long d_auto = std::max<long long>(0, auto_fit.fit.max_ctx - kSlack);
    CHECK_EQ(d_auto, 236832);

    const PackedValuesFitTerm resubmitted = fit_context_packed_values_at_depth(
        base, /*requested_chunk=*/2048, kCoderHeads, kCoderHeadSize, kScratchBudget,
        kR8KvBlockTokens, d_auto, /*max_partitions=*/0, /*element_bytes=*/4);
    const long long resubmitted_max_ctx_slack =
        std::max<long long>(0, resubmitted.fit.max_ctx - kSlack);
    CHECK(d_auto <= resubmitted_max_ctx_slack);  // admitted -- the round trip holds
}

