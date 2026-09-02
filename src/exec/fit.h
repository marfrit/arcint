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

}  // namespace lgc
