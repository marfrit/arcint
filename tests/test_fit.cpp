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
