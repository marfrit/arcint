#include <cstdlib>
#include "core/spill_guard.h"
#include "harness.h"

#include <string>

using namespace lgc;

// The spill guard's rule, device-free (`measured-here` calibration points on the B60, xe's
// region total 24,480 MiB against OpenCL's 23,256; the margin is the growth beyond the FULL
// prediction: worst measured 294 (agent 255k needle) + 128 headroom = 422 MiB).
namespace {
constexpr size_t MiB = 1024ull * 1024ull;
constexpr size_t kB60Driver = 24480 * MiB;

SpillFacts facts(size_t demand, size_t host = 0, int n_ctx = 49152, size_t other = 0) {
    SpillFacts f;
    f.weights         = 0;          // the calibration records the whole device line, not its split
    f.context         = demand;
    f.compute         = 0;
    f.host_by_design  = host;
    f.card_total      = kB60Driver;
    f.card_from_driver = true;
    f.other_clients   = other;
    f.n_ctx           = n_ctx;
    return f;
}
}  // namespace

TEST(spill_guard_the_margin_is_the_worst_measured_growth_plus_headroom) {
    // worst growth 294 (agent 255k needle) + 128 headroom = 422
    CHECK_EQ(kSpillGrowthHeadroomBytes, static_cast<size_t>(128) * MiB);
    CHECK_EQ(kSpillGrowthMarginBytes, static_cast<size_t>(422) * MiB);
}

TEST(spill_guard_inkling_49152_cells_fits) {
    const SpillDecision d = spill_guard(facts(21732 * MiB, 0, 49152));
    CHECK(d.fits);
    CHECK(!d.refuse);
    CHECK_EQ(d.accounted, (21732 + 422) * MiB);
    CHECK(d.text.find("weights") != std::string::npos);
    CHECK(d.text.find("context 21732 MiB") != std::string::npos);
    CHECK(d.text.find("growth margin 422 MiB") != std::string::npos);
    CHECK(d.text.find("driver") != std::string::npos);
    CHECK(d.text.find("start fits") != std::string::npos);
}

TEST(spill_guard_inkling_65536_cells_is_a_confirmed_fit) {
    // card result (B60, full-size SWA, q8_0 KV): a 60,000-token needle was
    // answered RIGHT; VRAM after load 22,966 MiB, peak 23,686; GTT at most 308 MiB (no spill).
    // The breakdown is 23,460, so the growth beyond it is 226 MiB; the boundary under test is
    // 23,460 + 422 = 23,882, below the card, and the measured peak is below it too.
    const SpillDecision d = spill_guard(facts(23460 * MiB, 0, 65536));
    CHECK(d.fits);
    CHECK(!d.refuse);
    CHECK_EQ(d.accounted, (23460 + 422) * MiB);
    CHECK(d.accounted <= 24480 * MiB);
    CHECK(d.text.find("start fits") != std::string::npos);
}

TEST(spill_guard_inkling_131072_full_swa_is_truly_over_the_card) {
    // full-SWA at 131,072 cells: the breakdown alone (extrapolated from the 49k/65k slope,
    // 110.6 kB/cell) is 30,372 MiB, above the 24,480 card. This is the red case now.
    const SpillDecision d = spill_guard(facts(30372 * MiB, 0, 131072));
    CHECK(!d.fits);
    CHECK(d.refuse);
    CHECK(d.text.find("start refused") != std::string::npos);
    CHECK_EQ(static_cast<size_t>(d.accounted), (30372 + 422) * MiB);
}

TEST(spill_guard_the_allow_spill_flag_warns_but_starts) {
    SpillFacts f = facts(30372 * MiB, 0, 131072);
    f.allow_spill = true;
    const SpillDecision d = spill_guard(f);
    CHECK(!d.fits);
    CHECK(!d.refuse);
    CHECK(d.text.find("--llama-allow-spill") != std::string::npos);
    CHECK(d.text.find("does NOT fit") != std::string::npos);
}

TEST(spill_guard_inkling_262144_with_the_swa_window_fits) {
    // 21,800 full prediction, 22,031 after a 250k needle: growth 231, within the 422 margin.
    const SpillDecision d = spill_guard(facts(20918 * MiB, 0, 262144));
    CHECK(d.fits);
    CHECK(d.text.find("start fits") != std::string::npos);
}

TEST(spill_guard_the_agent_full_prediction_plus_margin_fits_its_peak) {
    // the agent at 262,144: full prediction 22,998 + 904 = 23,902 (draft term below), peak
    // 24,196 -> growth 294; plus the 422 margin the start is 24,324, under 24,480.
    SpillFacts f = facts(22998 * MiB, 0, 262144);   // the target breakdown, the draft added below
    f.draft_device = 904 * MiB;
    const SpillDecision d = spill_guard(f);
    CHECK(d.fits);
    CHECK_EQ(d.accounted, (22998 + 904 + 422) * MiB);
}

// Flash-Next on the B70, the card's two arms (`measured-here`):
// the guard's target figures were weights 3,983 + context 4,530 + compute 944 = 10,611 MiB,
// and the expert cache's own line named 13.05 GiB on the card. The driver's region total is
// 32,656 MiB. The unit flags must fit; --llama-expert-cache 40000 (39.46 GiB on the card)
// must refuse and name the cache size that would fit.
namespace {
// the cache's measured accounting at the B70: flag MiB -> on-card MiB, piecewise between the
// three measured pairs (13,000->13,363, 19,241->19,571, 40,000->40,409). This is the fixture
// the suggestion walks down over; production calls the cache's own projection API.
constexpr size_t kAnchors[3][2] = {{13000, 13363}, {19241, 19571}, {40000, 40409}};
size_t fixture_oncard(size_t flag_mib) {
    if (flag_mib <= kAnchors[0][0]) return kAnchors[0][1] * MiB;
    for (int i = 1; i < 3; ++i) {
        if (flag_mib <= kAnchors[i][0]) {
            const size_t f0 = kAnchors[i - 1][0], f1 = kAnchors[i][0];
            const size_t b0 = kAnchors[i - 1][1], b1 = kAnchors[i][1];
            return (b0 + (flag_mib - f0) * (b1 - b0) / (f1 - f0)) * MiB;
        }
    }
    return kAnchors[2][1] * MiB;
}

SpillFacts flash_next(size_t cache_mib, size_t draft_mib = 0, size_t flag_mib = 13000) {
    SpillFacts f;
    f.weights        = 3983 * MiB;
    f.context        = 4530 * MiB;
    f.compute        = 944 * MiB;
    f.cache_device   = cache_mib * MiB;
    f.cache_flag_mib = flag_mib;
    f.draft_device   = draft_mib * MiB;
    f.card_total     = 32656 * MiB;
    f.card_from_driver = true;
    f.process_vram0  = f.weights + f.context + f.compute + f.cache_device + f.draft_device;
    f.other_clients  = 1154 * MiB;   // the card's 10,611 MiB was 9,457 known + 1,154 used - fdinfo
    f.n_ctx          = 262144;
    f.cache_project  = fixture_oncard;   // flag MiB -> on-card bytes
    return f;
}
}  // namespace

TEST(spill_guard_flash_next_b70_unit_flags_fit) {
    const SpillDecision d = spill_guard(flash_next(13363));   // 13,000 MiB budget, 13.05 GiB on the card
    CHECK(d.fits);
    CHECK(!d.refuse);
    CHECK_EQ(d.accounted, (10611 + 13363 + 422) * MiB);   // 10,611 includes the 1,154 other-client share
    CHECK(d.text.find("expert cache 13363 MiB") != std::string::npos);
    CHECK(d.text.find("start fits") != std::string::npos);
}

TEST(spill_guard_flash_next_b70_cache_40000_is_the_red_case_and_is_refused) {
    // 39.46 GiB on the card: the cache alone is more than the card. Without counting it the
    // old rule saw 10,611 + margin and said "fits" -- the card's defect.
    SpillFacts without_cache = flash_next(0);
    CHECK(spill_guard(without_cache).fits);   // the pre-fix figure, the red case

    const SpillDecision d = spill_guard(flash_next(40407, 0, 40000));
    CHECK(!d.fits);
    CHECK(d.refuse);
    CHECK(d.text.find("expert cache 40407 MiB") != std::string::npos);
    CHECK(d.text.find("start refused") != std::string::npos);
    CHECK(d.fit_cache > 0);
    CHECK(d.fit_cache < 40407 * MiB);
    CHECK(d.text.find("--llama-expert-cache about") != std::string::npos);
}

TEST(spill_guard_a_deliberate_host_bank_never_refuses) {
    // Flash-Next's production unit flags on the B70 (--n-ctx 262144, --llama-expert-cache
    // 13000, the MTP head, q8_0 KV): ~40.94 GiB of experts pinned in host memory (GTT) and a
    // device demand of 24,000 MiB on the 32 GB driver figure. This must start; reading GTT as
    // spill was the first card guard's defect.
    SpillFacts f = facts(24000 * MiB, 40940 * MiB, 262144);
    f.card_total = 32768 * MiB;
    CHECK(spill_guard(f).fits);
    CHECK(spill_guard(f).text.find("host by design 40940 MiB (not counted)") != std::string::npos);
}

TEST(spill_guard_the_same_flags_with_a_device_demand_beyond_the_card_are_refused) {
    SpillFacts f = facts(24000 * MiB, 40940 * MiB, 262144);   // the Flash-Next unit's host shape
    f.card_total = 32768 * MiB;
    f.context    = 40000 * MiB;   // an expert cache far beyond the card's device demand
    CHECK(!spill_guard(f).fits);
    CHECK(spill_guard(f).refuse);
}

TEST(spill_guard_other_clients_count_against_the_total) {
    // the probe's used - fdinfo: another client on the card takes the room a load would use
    CHECK(spill_guard(facts(21000 * MiB, 0, 262144)).fits);
    CHECK(spill_guard(facts(21000 * MiB, 0, 262144, 3400 * MiB)).refuse);
}

TEST(spill_guard_the_opencl_fallback_is_named_in_the_message) {
    SpillFacts f = facts(21732 * MiB, 0, 49152);
    f.card_total       = 23256 * MiB;   // B60's OpenCL figure, the fallback
    f.card_from_driver = false;
    const SpillDecision d = spill_guard(f);
    CHECK(d.fits);   // 21,732 + 422 = 22,154 < 23,256
    CHECK(d.text.find("OpenCL") != std::string::npos);
    CHECK(d.text.find("driver") == std::string::npos);
}

TEST(spill_guard_no_fit_names_the_fixed_parts) {
    // weights, compute and other clients already exceed the budget: no --n-ctx is suggested
    SpillFacts f = facts(0, 0, 262144, 30000 * MiB);
    f.weights = 30000 * MiB;
    const SpillDecision d = spill_guard(f);
    CHECK(!d.fits);
    CHECK(d.refuse);
    CHECK_EQ(d.fit_n_ctx, 0);
    CHECK(d.text.find("no --n-ctx fits") != std::string::npos);
}

TEST(spill_guard_the_fdinfo_residual_counts_process_device_outside_the_breakdown) {
    // a projector, lazily grown scratch, the KV stream: fdinfo holds bytes the known demand does
    // not name. The residual term counts them, so a start with them can refuse.
    SpillFacts f = facts(21000 * MiB, 0, 262144);
    f.process_vram0 = 24400 * MiB;   // known 21,000; 3,400 outside the breakdown
    const SpillDecision d = spill_guard(f);
    CHECK(d.unlisted == 3400 * MiB);
    CHECK(!d.fits);
    CHECK(d.refuse);
    CHECK(d.text.find("process device outside the breakdown 3400 MiB") != std::string::npos);
}

// The B60 agent at 262,144 (`measured-here`): MTP-on target breakdown 22,998, the
// draft context+head outside it 904 (KV 416 + reserve/compute 316 + head rows 172), other
// clients 40, 255k needle peak 24,196 of 24,480.
namespace {
SpillFacts agent_unit() {
    SpillFacts f;
    f.weights        = 15622 * MiB;
    f.context        = 6978 * MiB;
    f.compute        = 398 * MiB;
    f.draft_device   = 904 * MiB;   // the draft context's own breakdown + head rows
    f.other_clients  = 40 * MiB;
    f.card_total     = 24480 * MiB;
    f.card_from_driver = true;
    f.process_vram0  = f.weights + f.context + f.compute + f.draft_device;
    f.n_ctx          = 262144;
    return f;
}
}  // namespace

TEST(spill_guard_the_agent_unit_fits_with_its_measured_peak) {
    const SpillDecision d = spill_guard(agent_unit());
    CHECK(d.fits);
    CHECK(!d.refuse);
    CHECK_EQ(d.accounted, (15622 + 6978 + 398 + 904 + 40 + 422) * MiB);
    CHECK(d.text.find("draft/head 904 MiB") != std::string::npos);
    CHECK(d.accounted < 24480 * MiB);
}

TEST(spill_guard_the_arex_unit_fits) {
    SpillFacts f = agent_unit();
    f.weights = 15000 * MiB;
    f.context = 7000 * MiB;
    f.compute = 677 * MiB;
    f.process_vram0 = f.weights + f.context + f.compute + f.draft_device;
    const SpillDecision d = spill_guard(f);
    CHECK(d.fits);
    CHECK(!d.refuse);
}

TEST(spill_guard_the_in_model_mtp_term_is_the_draft_breakdown) {
    // the draft term: the draft context's KV (416 at 262,144, q8_0:q4_0 one MTP layer)
    // + its reserve/compute (316) + the head rows (172) = 904, measured after the first request.
    SpillFacts f = agent_unit();
    CHECK_EQ(f.draft_device, static_cast<size_t>(416 + 316 + 172) * MiB);
}

TEST(spill_guard_the_margin_covers_every_measured_growth_and_has_headroom) {
    // growth beyond the FULL prediction at the measured peaks: agent 294, Inkling-with-switch
    // 231, Flash-Next 184 and 172, Inkling 65,536 full SWA 226. Worst 294 + 128 headroom = 422.
    CHECK(kSpillGrowthMarginBytes >= 294 * MiB);
    CHECK(kSpillGrowthMarginBytes >= 231 * MiB);
    CHECK(kSpillGrowthMarginBytes >= 184 * MiB);
    CHECK(kSpillGrowthMarginBytes >= 172 * MiB);
    CHECK(kSpillGrowthMarginBytes >= 226 * MiB);
    CHECK_EQ(kSpillGrowthMarginBytes, static_cast<size_t>(294 + 128) * MiB);
    CHECK_EQ(kSpillGrowthMarginBytes - kSpillGrowthHeadroomBytes, static_cast<size_t>(294) * MiB);
}

TEST(spill_guard_the_cache_suggestion_leaves_room_for_the_context) {
    const SpillDecision d = spill_guard(flash_next(40407, 0, 40000));
    CHECK_EQ(d.fit_cache / MiB, static_cast<size_t>(21623));   // 32,656 - weights/compute/context/other/margin
    CHECK(d.fit_cache_flag_mib > 0);
    CHECK(d.fit_cache_flag_mib < 40000);
    // feed the suggested FLAG back through the same measured accounting: it must fit
    SpillFacts f = flash_next(fixture_oncard(d.fit_cache_flag_mib) / MiB, 0, d.fit_cache_flag_mib);
    CHECK(spill_guard(f).fits);
}

TEST(spill_guard_the_n_ctx_suggestion_itself_passes) {
    SpillFacts f = facts(30372 * MiB, 0, 131072);
    const SpillDecision d = spill_guard(f);
    CHECK(!d.fits);
    CHECK(d.fit_n_ctx > 0);
    SpillFacts g = f;
    g.n_ctx   = d.fit_n_ctx;
    g.context = static_cast<size_t>(f.context) * static_cast<size_t>(d.fit_n_ctx) /
                static_cast<size_t>(f.n_ctx);
    CHECK(spill_guard(g).fits);
}

TEST(spill_guard_the_cache_accounting_reproduces_the_three_measured_pairs) {
    CHECK_EQ(fixture_oncard(13000), static_cast<size_t>(13363) * MiB);
    CHECK_EQ(fixture_oncard(19241), static_cast<size_t>(19571) * MiB);
    CHECK_EQ(fixture_oncard(40000), static_cast<size_t>(40409) * MiB);
}

TEST(spill_guard_the_b70_refusal_terms_suggest_a_fitting_cache) {
    // the card's refusal line: weights 3983, context 4530, compute 944, cache 40409,
    // draft/head 3298, other clients 40, margin 422 of 32656 -> 19439 free for the cache.
    SpillFacts f;
    f.weights        = 3983 * MiB;
    f.context        = 4530 * MiB;
    f.compute        = 944 * MiB;
    f.cache_device   = 40409 * MiB;
    f.cache_flag_mib = 40000;
    f.draft_device   = 3298 * MiB;
    f.other_clients  = 40 * MiB;
    f.card_total     = 32656 * MiB;
    f.card_from_driver = true;
    f.process_vram0  = f.weights + f.context + f.compute + f.cache_device + f.draft_device;
    f.n_ctx          = 262144;
    f.cache_project  = fixture_oncard;   // flag MiB -> on-card bytes
    const SpillDecision d = spill_guard(f);
    CHECK(!d.fits);
    CHECK(d.refuse);
    CHECK_EQ(d.fit_cache / MiB, static_cast<size_t>(19439));
    CHECK(d.fit_cache_flag_mib >= 18800);
    CHECK(d.fit_cache_flag_mib <= 19439);
    CHECK(d.text.find("--llama-expert-cache about") != std::string::npos);
    CHECK(d.text.find("no expert cache fits") == std::string::npos);
    // the suggested value's accounting fits the same free bytes
    SpillFacts g = f;
    g.cache_flag_mib = d.fit_cache_flag_mib;
    g.cache_device   = fixture_oncard(d.fit_cache_flag_mib) / MiB * MiB;
    g.process_vram0  = g.weights + g.context + g.compute + g.cache_device + g.draft_device;
    CHECK(spill_guard(g).fits);
}

TEST(spill_guard_the_flag_unit_is_converted_to_bytes_once) {
    CHECK_EQ(cache_flag_bytes(0), static_cast<size_t>(0));
    CHECK_EQ(cache_flag_bytes(40000), static_cast<size_t>(40000) * MiB);
}

TEST(spill_guard_the_cache_search_handles_a_non_monotonic_projection) {
    // a fit band [10,000, 12,000] with a larger projection above it: the descending walk must
    // find the band and refine to its upper edge.
    SpillFacts f;
    f.weights = 3983 * MiB; f.context = 4530 * MiB; f.compute = 944 * MiB;
    f.draft_device = 3298 * MiB; f.other_clients = 40 * MiB; f.margin = 422 * MiB;
    f.card_total = 32656 * MiB; f.cache_flag_mib = 20000; f.n_ctx = 262144;
    f.cache_device = 20000 * MiB;
    f.cache_project = [](size_t flag) {
        return flag >= 10000 && flag <= 12000 ? static_cast<size_t>(flag) * MiB
                                              : static_cast<size_t>(50000) * MiB;
    };
    const SpillDecision d = spill_guard(f);
    CHECK(d.refuse);
    CHECK(d.fit_cache_flag_mib >= 10000);
    CHECK(d.fit_cache_flag_mib <= 12000);
    CHECK(f.cache_project(d.fit_cache_flag_mib) <= d.fit_cache);
}

TEST(spill_guard_the_cache_search_reports_none_when_no_projection_fits) {
    SpillFacts f;
    f.weights = 3983 * MiB; f.context = 4530 * MiB; f.compute = 944 * MiB;
    f.draft_device = 3298 * MiB; f.other_clients = 40 * MiB; f.margin = 422 * MiB;
    f.card_total = 32656 * MiB; f.cache_flag_mib = 40000; f.n_ctx = 262144;
    f.cache_device = 40000 * MiB;
    f.cache_project = [](size_t) { return static_cast<size_t>(50000) * MiB; };
    const SpillDecision d = spill_guard(f);
    CHECK(d.refuse);
    CHECK_EQ(d.fit_cache_flag_mib, static_cast<size_t>(0));
    CHECK(d.text.find("no expert cache fits") != std::string::npos);
}
