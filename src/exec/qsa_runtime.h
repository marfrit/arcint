#pragma once

#include <cstdint>
#include <optional>
#include <string>

// QSA (campaign qsa, step 3) runtime constraints under option A.
//
// The served QSA layer keeps the indexer's raw-key history in a plain state
// Variable, one f32 [1, past, indexer_head_dim] row per QSA layer. That state
// is neither the KV pool (paged ports) nor the GDN checkpoint slab (the
// conv/gated-delta state-table ports the loader allocates per lane), so every
// path that saves or restores "the state" has to either cover it or the load
// must refuse. This file is the pure decision, so the refusals are unit-testable
// without a card; backend_ov.cpp's load_paged builds the struct from the real
// Config and throws on the returned message.
//
// The geometry is read from the served graph, not from a config key, so it
// cannot drift from what the graph will store. That part needs OpenVINO types
// and lives behind the same guard as the rest of the OV path.

#ifdef ARCINT_OPENVINO

#include <memory>

#include <openvino/openvino.hpp>

#endif

namespace lgc {
namespace qsa {

// The indexer's raw-key history, per token per lane: `n_layer` layers, each
// one f32 row of `indexer_head_dim`. Flash-Next: 12 x 128 x 4 B = 6144 B =
// 6 KiB/token, 192 MiB at 32k.
inline uint64_t state_bytes_per_token(size_t n_layer, size_t indexer_head_dim) {
    return static_cast<uint64_t>(n_layer) * static_cast<uint64_t>(indexer_head_dim) *
           sizeof(float);
}

#ifdef ARCINT_OPENVINO

// One f32 [1, past, head_dim] Variable per QSA layer, id
// "cache_params.past.indexer_key.<layer>" (tools/q4e/serving_shape.py's
// `_indexer_variable`). The seq dim is dynamic and ignored; the trailing dims
// are the per-token row.
struct StateGeometry {
    size_t   n_layer         = 0;
    uint64_t bytes_per_token = 0;
};

inline StateGeometry state_geometry(const std::shared_ptr<ov::Model>& model) {
    StateGeometry g;
    if (!model) return g;
    for (const auto& var : model->get_variables()) {
        const ov::op::util::VariableInfo& info = var->get_info();
        if (info.variable_id.rfind("cache_params.past.indexer_key.", 0) != 0) continue;
        const ov::PartialShape& ps = info.data_shape;
        if (ps.rank().is_dynamic()) continue;
        const int64_t rank = ps.rank().get_length();
        if (rank < 3) continue;
        uint64_t per_token = 1;
        for (int64_t i = 2; i < rank; ++i) {
            if (ps[i].is_dynamic()) { per_token = 0; break; }
            per_token *= static_cast<uint64_t>(ps[i].get_length());
        }
        if (per_token == 0) continue;
        g.bytes_per_token += per_token * static_cast<uint64_t>(info.data_type.size());
        ++g.n_layer;
    }
    return g;
}

#endif  // ARCINT_OPENVINO

// The configurations option A cannot honour. Every field is already a fact of
// the Config (and the resolved drafter state) at load time.
struct RuntimeLimits {
    int  lanes        = 1;      // --parallel
    bool prefix_cache = false;  // --prefix-cache-mib > 0
    bool speculative  = false;  // paged speculation: --mtp on / --dflash active
};

// nullopt = QSA can be served under option A; otherwise the load must throw
// this message, rather than serve a mask built from empty or stale raw keys.
inline std::optional<std::string> runtime_refusal(const RuntimeLimits& c) {
    if (c.lanes > 1) {
        return "QSA under option A serves one lane (--parallel 1): the indexer's "
               "raw-key history is a plain per-layer state Variable, and the pass's "
               "mask rows are the flattened new tokens of ONE sequence. A second "
               "lane would not share that state safely, so it is refused. Option B "
               "(a paged indexer cache) is the route to multi-lane QSA.";
    }
    if (c.prefix_cache) {
        return "QSA cannot be served with a prefix cache: a reused prefix restores "
               "the KV pages and the GDN rows but NOT the indexer's raw-key Variable "
               "(it is not in the cache blob), so the selection would be built from "
               "an empty or stale history. Disable --prefix-cache-mib for a QSA "
               "artifact.";
    }
    if (c.speculative) {
        return "QSA cannot be served with paged speculative decoding under option A: "
               "a rejected draft has already appended raw keys to the indexer's "
               "Variable, and the paged rollback moves the committed GDN checkpoint "
               "row back without trimming that history -- the selection would read "
               "rejected tokens. Disable --mtp / --dflash for a QSA artifact.";
    }
    return std::nullopt;
}

}  // namespace qsa
}  // namespace lgc
