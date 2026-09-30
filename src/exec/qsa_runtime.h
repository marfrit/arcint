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
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <openvino/openvino.hpp>

#endif

namespace lgc {
namespace qsa {

// The indexer's per-token state rate: the compressed block cache stores ONE
// f32 row of `indexer_head_dim` per completed `ratio`-token block, so the
// row AMORTISES over `ratio` tokens. Flash-Next: 12 x 128 x 4 B / 4 = 1536 B =
// 1.5 KiB/token. This is the per-token term; the raw tail, the token counter
// and one bucket of capacity slack are fixed and reported beside it.
inline uint64_t state_bytes_per_token(size_t n_layer, size_t indexer_head_dim,
                                      size_t ratio) {
    if (ratio == 0) return 0;
    return static_cast<uint64_t>(n_layer) * static_cast<uint64_t>(indexer_head_dim) *
           sizeof(float) / ratio;
}

#ifdef ARCINT_OPENVINO

// One compressed block cache per QSA layer: a f32 [-1, head_dim] block-row
// Variable (`cache_params.past.indexer_block.<layer>`), plus the fixed raw
// tail (`...indexer_tail.<layer>`, [ratio-1, head_dim]) and the token counter
// (`...indexer_pos.<layer>`, [1] i32).
struct StateGeometry {
    size_t   n_layer         = 0;
    uint64_t block_row_bytes = 0;   // per layer, per completed block
    uint64_t fixed_bytes     = 0;   // tail + counter over every layer
    int64_t  block_cap       = 0;   // fixed block capacity per layer (rows)
};

inline StateGeometry state_geometry(const std::shared_ptr<ov::Model>& model) {
    StateGeometry g;
    if (!model) return g;
    for (const auto& var : model->get_variables()) {
        const ov::op::util::VariableInfo& info = var->get_info();
        const std::string&               id   = info.variable_id;
        const ov::PartialShape&          ps   = info.data_shape;
        if (ps.rank().is_dynamic()) continue;
        const int64_t rank = ps.rank().get_length();
        if (id.rfind("cache_params.past.indexer_block.", 0) == 0) {
            // The seq dim (index 0) is the FIXED capacity; dims 1.. are the
            // per-block row.
            uint64_t per_block = 1;
            bool     ok        = true;
            for (int64_t i = 1; i < rank; ++i) {
                if (ps[i].is_dynamic()) { ok = false; break; }
                per_block *= static_cast<uint64_t>(ps[i].get_length());
            }
            if (!ok) continue;
            g.block_row_bytes = per_block * static_cast<uint64_t>(info.data_type.size());
            if (ps[0].is_static()) g.block_cap = ps[0].get_length();
            ++g.n_layer;
        } else if (id.rfind("cache_params.past.indexer_tail.", 0) == 0 ||
                   id.rfind("cache_params.past.indexer_pos.", 0) == 0) {
            uint64_t bytes = 1;
            bool     ok    = true;
            for (int64_t i = 0; i < rank; ++i) {
                if (ps[i].is_dynamic()) { ok = false; break; }
                bytes *= static_cast<uint64_t>(ps[i].get_length());
            }
            if (ok) g.fixed_bytes += bytes * static_cast<uint64_t>(info.data_type.size());
        }
    }
    return g;
}

// QSA step 3 (2026-09-29). The exporter writes the marker on the MODEL as
// `qsa = {"boundary": N, "mask_nodes": ["attn3/qsa_mask", ...]}`. Node-level
// rt_info does NOT survive `ov.save_model` (the serializer writes only a fixed
// set of keys), which is exactly why the served 0.5.4 artifacts dropped the
// mask. Model rt_info does serialize, so this is the transport; load_paged
// calls it before SDPAToPagedAttention to re-apply the node tags the pass
// reads. Returns the number of nodes tagged (0 when the marker is absent or
// malformed).
inline size_t reapply_selection_tags(const std::shared_ptr<ov::Model>& model) {
    if (!model) return 0;
    const auto& rt = model->get_rt_info();
    const auto  it = rt.find("qsa");
    if (it == rt.end()) return 0;
    nlohmann::json j = nlohmann::json::parse(it->second.as<std::string>(), nullptr, false);
    if (j.is_discarded() || !j.is_object()) return 0;
    const int64_t boundary = j.value("boundary", static_cast<int64_t>(0));
    if (!j.contains("mask_nodes") || !j["mask_nodes"].is_array()) return 0;

    std::unordered_map<std::string, std::shared_ptr<ov::Node>> by_name;
    for (const auto& op : model->get_ops()) by_name[op->get_friendly_name()] = op;

    size_t applied = 0;
    for (const auto& name : j["mask_nodes"]) {
        if (!name.is_string()) continue;
        const auto n = by_name.find(name.get<std::string>());
        if (n == by_name.end()) continue;
        auto& node_rt = n->second->get_rt_info();
        node_rt["arcint"]       = std::string("qsa_selection");
        node_rt["qsa_boundary"] = std::string(std::to_string(boundary));
        ++applied;
    }
    return applied;
}

// After SDPAToPagedAttention: the number of PagedAttention nodes that carry
// the optional selection input (29 inputs). load_paged refuses when this is
// not the artifact's declared QSA layer count -- the very condition the served
// 0.5.4 artifacts silently violated.
inline size_t count_qsa_paged_attention(const std::shared_ptr<ov::Model>& model) {
    if (!model) return 0;
    size_t n = 0;
    for (const auto& op : model->get_ops()) {
        if (std::string(op->get_type_name()) == "PagedAttentionExtension" &&
            op->get_input_size() == 29)
            ++n;
    }
    return n;
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
