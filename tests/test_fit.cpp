#include "exec/fit.h"
#include "harness.h"

#include <cstdint>
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
