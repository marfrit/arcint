#include "exec/flash_next_offload.h"
#include "harness.h"

#include <cstdint>

using namespace lgc;

namespace {

constexpr double kGiB = static_cast<double>(1ull << 30);
inline uint64_t gib(double x) { return static_cast<uint64_t>(x * kGiB + 0.5); }
inline double as_gib(uint64_t b) { return static_cast<double>(b) / kGiB; }

// Geometry constants must match the measured GGUF (WP6) and the Python
// instrument (tools/flash_next_fit.py). If a refactor drifts one of these, the
// projection silently moves -- pin them.
TEST(flash_next_geometry_constants_match_measured) {
    CHECK_EQ(kFlashNextSliceBytes, 2457600ull);
    CHECK_EQ(kFlashNextMoeLayers, 48);
    CHECK_EQ(kFlashNextExperts, 512);
    CHECK_EQ(kFlashNextActive, 10);
    CHECK_EQ(kFlashNextTrafficBytes, 1179648000ull);          // 10*48*2457600
    CHECK_EQ(kFlashNextTableBytes, 28800138240ull);           // IQ4_NL 160x320001536
    CHECK_NEAR(as_gib(kFlashNextTrafficBytes), 1.098633, 1e-4);
    CHECK_NEAR(as_gib(kFlashNextTableBytes), 26.8222, 1e-3);
    const uint64_t pool = 512ull * 48 * 2457600;
    CHECK_NEAR(as_gib(pool), 56.25, 1e-6);
}

// Cross-check the projection against tools/flash_next_fit.py's own values
// (computed 2026-09-10): the two implementations are kept numerically
// identical, so the analysis tool and the served-config header never disagree.
TEST(flash_next_projection_matches_python_instrument) {
    // DRAM ceiling (h=1, 44.4/4.66) and NVMe floor (h=0, 44.4/1.68).
    CHECK_NEAR(flash_next_project_tps(1.0, 44.4, 4.66, 1.0), 40.414, 0.02);
    CHECK_NEAR(flash_next_project_tps(0.0, 44.4, 1.68, 1.0), 1.529, 0.02);
    // Operating points at the measured NVMe miss feed (1.68 GiB/s).
    CHECK_NEAR(flash_next_project_tps(0.95,  44.4, 1.68, 1.0), 17.792, 0.02);
    CHECK_NEAR(flash_next_project_tps(0.944, 44.4, 1.68, 1.0), 16.672, 0.02);
    CHECK_NEAR(flash_next_project_tps(0.881, 44.4, 1.68, 1.0), 10.038, 0.02);  // per-layer 16 GiB
    CHECK_NEAR(flash_next_project_tps(0.938, 44.4, 1.68, 1.0), 15.685, 0.02);  // global 16 GiB
}

// The WP6b headline reproduced: at the provisioned single-A770 operating point
// -- ~24 GiB resident expert pool, per-layer hit ~95%, NVMe miss feed
// 1.68 GiB/s -- the projection is ~18 t/s and the regime is NVMe-miss-bound
// (the miss term dominates; the streaming plan's whole job is to keep it small).
TEST(flash_next_wp6b_operating_point_is_18tps_nvme_bound) {
    const double tps = flash_next_project_tps(0.95, kFlashNextDramBwGiBs,
                                              kFlashNextNvmeBwGiBs, 1.0);
    CHECK_NEAR(tps, 17.792, 0.05);
    CHECK(flash_next_regime(0.95, kFlashNextDramBwGiBs, kFlashNextNvmeBwGiBs, 1.0)
          == FlashNextRegime::NvmeMissBound);
    // At the DRAM ceiling (h=1) the regime flips to DRAM-resident-bound.
    CHECK(flash_next_regime(1.0, kFlashNextDramBwGiBs, kFlashNextNvmeBwGiBs, 1.0)
          == FlashNextRegime::DramResidentBound);
}

// THE WP7 FINDING, pinned as a test that can fail: the cache-model choice is
// not cosmetic. Feeding the PER-LAYER measured hit at a 16 GiB budget (88.1%,
// arcint's own slot-pool shape) projects ~10.0 t/s; feeding the GLOBAL-LRU
// figure on the same trace (93.8%, FreeToken's shape) projects ~15.7 t/s -- a
// >50% overstatement. A policy that adopted the global number would claim a
// throughput the served per-layer pool cannot reach. Assert the per-layer
// projection is materially LOWER, so a regression that swapped the model in is
// caught here.
TEST(flash_next_per_layer_model_is_not_the_optimistic_global_model) {
    const double per_layer = flash_next_project_tps(0.881, 44.4, 1.68, 1.0);
    const double global    = flash_next_project_tps(0.938, 44.4, 1.68, 1.0);
    CHECK(per_layer < global);
    CHECK(global - per_layer > 4.0);   // measured gap ~5.6 t/s; a real, not rounding, divergence
    CHECK_NEAR(per_layer, 10.038, 0.05);
}

// Residency accounting mirrors flash_next_fit.resident_expert_gib exactly:
// resident_expert_gib(15,44,2.3,3.0,2.0) = 23.8778 (VRAM 7.7 + DRAM 16.1778).
TEST(flash_next_resident_budget_mirrors_python) {
    const ResidentBudget b = flash_next_resident_budget(
        gib(15), gib(44), gib(2.3), gib(3.0), gib(2.0));
    CHECK_NEAR(as_gib(b.vram_for_experts), 7.7, 0.01);
    CHECK_NEAR(as_gib(b.dram_for_experts), 16.1778, 0.01);
    CHECK_NEAR(as_gib(b.expert_bytes), 23.8778, 0.01);
    CHECK(b.table_fits_dram);
}

// Per-layer slot count at a 24 GiB resident expert budget = floor(24GiB/slice)
// / 48 = 10485/48 = 218 -- the count tools/expert_lru_replay.py uses for the
// 24 GiB cell that reads ~94.4% per-layer.
TEST(flash_next_slots_per_layer_matches_replay) {
    CHECK_EQ(flash_next_slots_per_layer(gib(24)), 218);
    CHECK_EQ(flash_next_slots_per_layer(gib(16)), 145);
    CHECK_EQ(flash_next_slots_per_layer(gib(40)), 364);
}

// Refusal, condition 1: the PLE table cannot be DRAM-resident. With only 20 GiB
// DRAM the 26.82 GiB table does not fit; the plan must refuse regardless of the
// throughput floor (the random hashed gather would be seek-bound -- a latency
// cliff, not a slowdown).
TEST(flash_next_refuses_when_table_cannot_be_dram_resident) {
    const OffloadPlan p = flash_next_plan(gib(15), gib(20), gib(2.3), gib(3.0),
                                          gib(2.0), 0.95);
    CHECK(!p.table_fits_dram);
    CHECK(flash_next_offload_must_refuse(p, /*floor_tps=*/0.0));   // even with the floor off
}

// Refusal, condition 2: the projected decode rate is below the serving floor.
// A poor hit-rate (say 60%, a small resident pool) projects well under a 15 t/s
// floor and must refuse; the same plan with the floor disabled (<=0) does not.
TEST(flash_next_refuses_below_throughput_floor) {
    OffloadPlan p = flash_next_plan(gib(15), gib(44), gib(2.3), gib(3.0), gib(2.0), 0.60);
    CHECK(p.table_fits_dram);                       // table fits; only throughput is short
    CHECK(p.projected_tps < 15.0);
    CHECK(flash_next_offload_must_refuse(p, 15.0));
    CHECK(!flash_next_offload_must_refuse(p, 0.0));  // floor disabled -> admit
}

// A healthy plan (the provisioned operating point, ~95% hit) clears a 15 t/s
// floor and is admitted.
TEST(flash_next_admits_the_provisioned_operating_point) {
    const OffloadPlan p = flash_next_plan(gib(15), gib(44), gib(2.3), gib(3.0),
                                          gib(2.0), 0.95);
    CHECK(p.table_fits_dram);
    CHECK(p.projected_tps > 15.0);
    CHECK(!flash_next_offload_must_refuse(p, 15.0));
    CHECK(p.regime == FlashNextRegime::NvmeMissBound);
}

}  // namespace
