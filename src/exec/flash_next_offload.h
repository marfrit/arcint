#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

// WP7 (0.5.0) -- Flash-Next (qwen4exp) expert-offload serving policy. Pure
// arithmetic, no OpenVINO types (same discipline as fit.h), so it builds and is
// tested on any host: the load path in backend_ov.cpp measures the card budget
// and hands the terms here; this file sizes the resident expert pool, projects
// the bandwidth-bound decode rate for the streaming (host-tier offload) plan,
// and refuses a configuration that cannot clear a floor.
//
// This is the C++ mirror of tools/flash_next_fit.py's projection model and of
// the residency arithmetic the routing-trace replay (tools/expert_lru_replay.py)
// exercises. The two implementations are kept numerically identical on purpose
// (tests/test_flash_next_offload.cpp cross-checks the projection against the
// Python instrument's self-test constants); the Python side is the analysis
// tool, this header is what a served config consults.
//
// THE CACHE MODEL (WP7 finding, 2026-09-10, dated correction to WP6b's
// unstated model). The resident expert pool is PER-LAYER: each MoE layer holds
// its own LRU of `slots_per_layer` experts, exactly the shape arcint's offload
// slot pool already has (fit.h expert_slot_bytes = ceil(num_expert*(100-ratio)
// /100) slots PER LAYER, replicated across moe_layers). The measured hit-rate
// this file's projection consumes MUST come from a per-layer replay: a global
// shared LRU (FreeToken's paper shape) reads a materially different, more
// optimistic hit-rate on the same trace (~93.8% flat vs the per-layer 88.1% at
// a 16 GiB budget), and adopting it would overstate the projected t/s. WP6b's
// hit-rate table is the per-layer model (tools/expert_lru_replay.py --check
// reproduces it within ~1.4 points on the sha-pinned trace).
//
// This is a PROJECTION, labelled as such: every input is measured (bandwidths
// from dd/stream probes, hit-rate from the routing-trace replay), but the t/s
// is an analytic bandwidth-bound estimate, not an end-to-end served
// measurement. A served t/s claim still needs the served endpoint; the live
// expert-gather that would read NVMe-resident GGUF rows on a miss is PARKED on
// the backbone IR emission (FIX A), the same gate FIX D Link 3's integration
// sits behind. What this header lands is the policy the config uses to size the
// resident pool and to refuse an inadmissible plan before a window is spent.
namespace lgc {

// Measured Flash-Next geometry (WP6, off the real GGUF; mirrored in
// tools/flash_next_fit.py and tools/expert_lru_replay.py).
constexpr uint64_t kFlashNextSliceBytes  = 2'457'600;  // one expert-layer int4 slice (gate/up/down)
constexpr int      kFlashNextMoeLayers   = 48;
constexpr int      kFlashNextExperts     = 512;
constexpr int      kFlashNextActive      = 10;         // routed experts per token per layer
// PLE (n-gram) table: per_layer_token_embd, IQ4_NL (dtype 20), 160 x 320001536
// elements -> 28,800,138,240 B. Must be DRAM-resident (per-token random hashed
// gather is seek-bound on any paged tier -- FIX D).
constexpr uint64_t kFlashNextTableBytes  = 28'800'138'240ull;  // 26.82 GiB
// Per-token full-miss expert traffic = kActive * moe_layers * slice.
constexpr uint64_t kFlashNextTrafficBytes =
    static_cast<uint64_t>(kFlashNextActive) * kFlashNextMoeLayers * kFlashNextSliceBytes;  // 1.0986 GiB

// Measured tier bandwidths (single-A770 target). GiB/s.
//   DRAM  44.4  (WP2 window, single-thread stream read)
//   NVMe   1.68 (WP6b, in-container dd iflag=direct on the NVMe ZFS pool --
//               the REAL miss-feed rate, below the 2.3 GB/s raw partition)
//   HDD    0.413 (spinning-HDD pool, cold; NOT the Flash-Next miss path -- floor row)
constexpr double kFlashNextDramBwGiBs = 44.4;
constexpr double kFlashNextNvmeBwGiBs = 1.68;
constexpr double kFlashNextHddBwGiBs  = 0.413;

// Per-layer resident slot count for a resident expert budget of `resident_bytes`
// spread evenly across `moe_layers` layers -- the per-layer split arcint's slot
// pool uses. Floors (never claims a slot the budget cannot hold).
inline int flash_next_slots_per_layer(uint64_t resident_expert_bytes,
                                       uint64_t slice_bytes = kFlashNextSliceBytes,
                                       int moe_layers = kFlashNextMoeLayers) {
    if (slice_bytes == 0 || moe_layers <= 0) return 0;
    const uint64_t total_slots = resident_expert_bytes / slice_bytes;
    return static_cast<int>(total_slots / static_cast<uint64_t>(moe_layers));
}

// GiB of expert pool that can be kept LRU-resident across VRAM+DRAM, after the
// PLE table (DRAM), backbone+KV+activations (VRAM), and a DRAM overhead are
// paid. Mirrors flash_next_fit.resident_expert_gib exactly. All inputs bytes.
struct ResidentBudget {
    uint64_t expert_bytes = 0;  // min(expert pool, VRAM-for-experts + DRAM-for-experts)
    uint64_t vram_for_experts = 0;
    uint64_t dram_for_experts = 0;
    bool     table_fits_dram  = false;  // the PLE table is DRAM-resident (required)
};

inline ResidentBudget flash_next_resident_budget(
        uint64_t vram_bytes, uint64_t dram_bytes,
        uint64_t backbone_vram_bytes, uint64_t kv_bytes, uint64_t activation_bytes,
        uint64_t table_bytes = kFlashNextTableBytes,
        uint64_t dram_overhead_bytes = (1ull << 30)) {
    ResidentBudget b;
    const uint64_t vram_used = backbone_vram_bytes + kv_bytes + activation_bytes;
    b.vram_for_experts = vram_bytes > vram_used ? vram_bytes - vram_used : 0;
    const uint64_t dram_used = table_bytes + dram_overhead_bytes;
    b.dram_for_experts = dram_bytes > dram_used ? dram_bytes - dram_used : 0;
    b.table_fits_dram  = dram_bytes >= dram_used;
    const uint64_t pool = static_cast<uint64_t>(kFlashNextExperts) *
                          kFlashNextMoeLayers * kFlashNextSliceBytes;  // 56.25 GiB
    const uint64_t avail = b.vram_for_experts + b.dram_for_experts;
    b.expert_bytes = std::min(pool, avail);
    return b;
}

// Projected decode t/s under the streaming plan. hit_rate in [0,1], the
// PER-LAYER replay's measured hit at the resident capacity. Resident hits are
// served at the DRAM feed (the DRAM->VRAM move); misses stream from NVMe,
// divided by the MTP/PLE amortization A (output tokens per expert-load window;
// A=1 as the GGUF ships -- it carries no MTP head, see WP8). Identical to
// flash_next_fit.project_tps.
inline double flash_next_project_tps(double hit_rate, double dram_bw_gibs,
                                     double nvme_bw_gibs, double amortization = 1.0,
                                     double compute_floor_ms = 0.0,
                                     uint64_t traffic_bytes = kFlashNextTrafficBytes) {
    const double h = std::clamp(hit_rate, 0.0, 1.0);
    const double traffic_gib = static_cast<double>(traffic_bytes) / static_cast<double>(1ull << 30);
    const double amort = amortization > 1e-9 ? amortization : 1e-9;
    const double hit_ms  = (h * traffic_gib / dram_bw_gibs) * 1000.0;
    const double miss_ms = ((1.0 - h) * traffic_gib / (nvme_bw_gibs * amort)) * 1000.0;
    const double ms = hit_ms + miss_ms + compute_floor_ms;
    return ms > 0.0 ? 1000.0 / ms : 0.0;
}

// Which term dominates the per-token time -- the regime label every capacity
// statement must carry (measurement discipline). NVMe-miss-bound is the danger
// regime (the streaming plan's whole point is to keep misses off it);
// DRAM-resident-bound is the ceiling the plan approaches as the hit-rate rises.
enum class FlashNextRegime { DramResidentBound, NvmeMissBound };

inline FlashNextRegime flash_next_regime(double hit_rate, double dram_bw_gibs,
                                         double nvme_bw_gibs, double amortization = 1.0) {
    const double h = std::clamp(hit_rate, 0.0, 1.0);
    const double amort = amortization > 1e-9 ? amortization : 1e-9;
    const double hit_share  = h / dram_bw_gibs;
    const double miss_share = (1.0 - h) / (nvme_bw_gibs * amort);
    return miss_share >= hit_share ? FlashNextRegime::NvmeMissBound
                                   : FlashNextRegime::DramResidentBound;
}

inline const char* flash_next_regime_name(FlashNextRegime r) {
    return r == FlashNextRegime::NvmeMissBound ? "NVMe-miss-bound" : "DRAM-resident-bound";
}

// The whole plan: resident sizing + projection + regime. `hit_rate` is supplied
// by the caller from the per-layer replay at this resident capacity (it is data
// from the trace, not derivable here).
struct OffloadPlan {
    ResidentBudget  budget;
    int             slots_per_layer   = 0;
    double          resident_frac     = 0.0;  // expert_bytes / full pool
    double          projected_tps     = 0.0;
    FlashNextRegime regime            = FlashNextRegime::NvmeMissBound;
    bool            table_fits_dram   = false;
};

inline OffloadPlan flash_next_plan(uint64_t vram_bytes, uint64_t dram_bytes,
                                   uint64_t backbone_vram_bytes, uint64_t kv_bytes,
                                   uint64_t activation_bytes, double hit_rate,
                                   double dram_bw_gibs = kFlashNextDramBwGiBs,
                                   double nvme_bw_gibs = kFlashNextNvmeBwGiBs,
                                   double amortization = 1.0) {
    OffloadPlan p;
    p.budget = flash_next_resident_budget(vram_bytes, dram_bytes, backbone_vram_bytes,
                                          kv_bytes, activation_bytes);
    p.slots_per_layer = flash_next_slots_per_layer(p.budget.expert_bytes);
    const uint64_t pool = static_cast<uint64_t>(kFlashNextExperts) *
                          kFlashNextMoeLayers * kFlashNextSliceBytes;
    p.resident_frac = pool ? static_cast<double>(p.budget.expert_bytes) / static_cast<double>(pool) : 0.0;
    p.projected_tps = flash_next_project_tps(hit_rate, dram_bw_gibs, nvme_bw_gibs, amortization);
    p.regime = flash_next_regime(hit_rate, dram_bw_gibs, nvme_bw_gibs, amortization);
    p.table_fits_dram = p.budget.table_fits_dram;
    return p;
}

// Refuse a plan the machine should not serve rather than serve it below the
// floor -- same "refuse loudly, do not degrade silently" stance as fit.h's
// host_ram_fit_must_refuse. Two independent conditions:
//   (1) the PLE table cannot be DRAM-resident (its random hashed gather would
//       be seek-bound -- a correctness/latency cliff, not a slowdown);
//   (2) the projected decode rate is below `floor_tps` (the config cannot meet
//       the serving bar even with the resident pool it can afford).
// `floor_tps <= 0` disables the throughput half (table residency still checked).
inline bool flash_next_offload_must_refuse(const OffloadPlan& p, double floor_tps) {
    if (!p.table_fits_dram) return true;
    if (floor_tps > 0.0 && p.projected_tps < floor_tps) return true;
    return false;
}

}  // namespace lgc
