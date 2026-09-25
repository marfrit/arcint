#pragma once

// FEED-THE-PORTS (2026-09-13): the serving-shape IR carries the n-gram table
// as PARAMETER PORTS `ngram_table.K` -- one u8 [rows_K, row_bytes] per chunk
// under the card's per-object cap, bound once per request from host memory --
// and takes the hashed row for each token as two ports, `ngram_chunk_ids`
// (i32 [1, T, num_ngram_heads]) and `ngram_local_ids` (i64, same shape),
// split by the HOST at the port partition. Nothing in the graph does integer
// arithmetic on an id: the GPU plugin runs integer eltwise in f32 and was
// measured gathering the wrong row for every id not representable in f32
// (docs/window-050.md §4.7). The partition is read off the compiled model's
// port shapes -- no config carries it -- and the table's bytes are the
// GGUF's own `per_layer_token_embd.weight` (IQ4_NL, 90 bytes a 160-wide
// row), decoded in the graph after the gather.
//
// This header is the device-free half: recognise the ports, validate the
// partition and the source tensor, split ids. The OpenVINO half
// (backend_ov.cpp `bind_ngram_ports`, and the per-forward feed in
// `paged_forward`) allocates the USM-host chunks and hands the tensors over.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/gguf.h"
#include "util/log.h"

namespace lgc::ngram {

inline constexpr const char* kTablePortPrefix = "ngram_table.";
inline constexpr const char* kChunkIdsPort    = "ngram_chunk_ids";
inline constexpr const char* kLocalIdsPort    = "ngram_local_ids";
inline constexpr const char* kConvMaskPort    = "conv_mask";
inline constexpr const char* kTableTensor     = "per_layer_token_embd.weight";
// ggml's block_iq4_nl: an f16 scale + 16 nibble bytes per 32 elements.
inline constexpr int32_t kIq4NlType        = 20;
inline constexpr size_t  kIq4NlBlockElems  = 32;
inline constexpr size_t  kIq4NlBlockBytes  = 18;

struct TablePort {
    std::string name;   // "ngram_table.K"
    size_t      rows = 0;
};

struct PortPlan {
    std::vector<TablePort> chunks;      // in K order, contiguous from 0
    size_t row_bytes      = 0;          // static, every chunk the same
    size_t rows_per_chunk = 0;          // chunks[0].rows: the split's divisor
    size_t total_rows     = 0;
    bool   declares_ids       = false;  // both id ports present
    bool   declares_conv_mask = false;

    bool empty() const { return chunks.empty(); }
};

// A graph that declares NO ngram_table.K port is the qwen3_5_moe family's
// case: it carries no PLE and no n-gram table, and its n-gram binding must be
// INERT rather than required (--ngram-gguf must not be needed). But an
// artifact whose config DECLARES a table and whose IR declares no port is a
// mismatch: the PLE is silently absent and a table nothing reads may be
// admitted and held. Name that refusal here (the device-free half owns it) so
// the OpenVINO bind in backend_ov.cpp does not have to re-derive it, and so a
// unit cell can pin it. Returns "" when the pair is consistent.
inline std::string check_declared_table(int ngram_size, int ple_embed_dim,
                                        const PortPlan& plan) {
    if (!plan.empty()) return {};
    const bool declared = ngram_size >= 2 || ple_embed_dim > 0;
    if (!declared) return {};
    return log::format(
        "the artifact's config declares an n-gram table (ngram_size=%d ple_embed_dim=%d) but "
        "the IR declares no %s port to carry it; the PLE would be silently absent. Either the "
        "artifact is not the family its config says, or the table was dropped at export",
        ngram_size, ple_embed_dim, kTablePortPrefix);
}

// One compiled-model input as (name, dims), -1 for a dynamic dimension.
using PortDims = std::pair<std::string, std::vector<int64_t>>;

// Recognise the ports. Returns an empty plan when the model declares no
// `ngram_table.K`; throws, naming the port, when it declares them badly:
// a K missing from 0..n-1, a chunk that is not a static rank-2 [rows,
// row_bytes], row_bytes disagreeing between chunks, a chunk other than the
// last shorter than the first (the split divides by chunks[0].rows, so every
// chunk but the last must be exactly that long), or the id ports declared
// one without the other.
inline PortPlan plan_ngram_ports(const std::vector<PortDims>& inputs) {
    PortPlan plan;
    std::vector<std::pair<int, PortDims>> found;
    for (const auto& in : inputs) {
        const std::string& name = in.first;
        if (name == kChunkIdsPort) {
            plan.declares_ids = true;  // provisional; both required below
            continue;
        }
        if (name == kConvMaskPort) {
            plan.declares_conv_mask = true;
            continue;
        }
        if (name.rfind(kTablePortPrefix, 0) != 0) continue;
        const std::string tail = name.substr(std::string(kTablePortPrefix).size());
        if (tail.empty() || tail.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error(log::format(
                "ngram ports: '%s' is not 'ngram_table.<K>' with K a decimal", name.c_str()));
        }
        found.emplace_back(std::stoi(tail), in);
    }
    bool has_local = false;
    for (const auto& in : inputs) has_local |= in.first == kLocalIdsPort;
    if (plan.declares_ids != has_local) {
        throw std::runtime_error(log::format(
            "ngram ports: '%s' and '%s' must be declared together (found %s only)",
            kChunkIdsPort, kLocalIdsPort, plan.declares_ids ? kChunkIdsPort : kLocalIdsPort));
    }
    if (found.empty()) return plan;

    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t k = 0; k < found.size(); ++k) {
        const auto& [K, port] = found[k];
        if (K != static_cast<int>(k)) {
            throw std::runtime_error(log::format(
                "ngram ports: chunk %d missing -- the K's must be contiguous from 0 (found %s)",
                static_cast<int>(k), port.first.c_str()));
        }
        const auto& dims = port.second;
        if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0) {
            throw std::runtime_error(log::format(
                "ngram ports: '%s' must be a static rank-2 [rows, row_bytes] port (%zu dims, "
                "dims[0]=%lld dims[1]=%lld)",
                port.first.c_str(), dims.size(), static_cast<long long>(dims.empty() ? 0 : dims[0]),
                static_cast<long long>(dims.size() < 2 ? 0 : dims[1])));
        }
        const size_t rows = static_cast<size_t>(dims[0]);
        const size_t rb   = static_cast<size_t>(dims[1]);
        if (k == 0) {
            plan.row_bytes      = rb;
            plan.rows_per_chunk = rows;
        } else if (rb != plan.row_bytes) {
            throw std::runtime_error(log::format(
                "ngram ports: '%s' has %zu bytes a row, chunk 0 has %zu", port.first.c_str(), rb,
                plan.row_bytes));
        } else if (k + 1 < found.size() && rows != plan.rows_per_chunk) {
            throw std::runtime_error(log::format(
                "ngram ports: '%s' has %zu rows, chunk 0 has %zu -- every chunk but the last "
                "must be as long as the first (the split divides by it)",
                port.first.c_str(), rows, plan.rows_per_chunk));
        } else if (rows > plan.rows_per_chunk) {
            throw std::runtime_error(log::format(
                "ngram ports: last chunk '%s' has %zu rows, more than chunk 0's %zu",
                port.first.c_str(), rows, plan.rows_per_chunk));
        }
        plan.chunks.push_back({port.first, rows});
        plan.total_rows += rows;
    }
    return plan;
}

// Does the GGUF tensor match the ports? "" when it does; otherwise the
// refusal, naming what disagrees. The IR's rows are the GGUF's own bytes, so
// the type must be IQ4_NL, the row width (ggml dims[0]) must give exactly
// row_bytes per row, the row count (dims[1]) must be the ports' total, and
// the byte size must be rows * row_bytes.
inline std::string check_table_source(const gguf::TensorInfo& t, size_t tensor_bytes,
                                      const PortPlan& plan) {
    if (t.ggml_type != kIq4NlType) {
        return log::format("%s is %s, the ports carry IQ4_NL rows", t.name.c_str(),
                           gguf::type_name(t.ggml_type).c_str());
    }
    if (t.dims.size() != 2) {
        return log::format("%s has %zu dims, expected 2 [width, rows]", t.name.c_str(),
                           t.dims.size());
    }
    const size_t width = static_cast<size_t>(t.dims[0]);
    const size_t rows  = static_cast<size_t>(t.dims[1]);
    if (width % kIq4NlBlockElems != 0 ||
        width / kIq4NlBlockElems * kIq4NlBlockBytes != plan.row_bytes) {
        return log::format("%s rows are %zu elements = %zu IQ4_NL bytes, the ports take %zu",
                           t.name.c_str(), width, width / kIq4NlBlockElems * kIq4NlBlockBytes,
                           plan.row_bytes);
    }
    if (rows != plan.total_rows) {
        return log::format("%s has %zu rows, the ports partition %zu", t.name.c_str(), rows,
                           plan.total_rows);
    }
    if (tensor_bytes != rows * plan.row_bytes) {
        return log::format("%s is %zu bytes, %zu rows x %zu would be %zu", t.name.c_str(),
                           tensor_bytes, rows, plan.row_bytes, rows * plan.row_bytes);
    }
    return "";
}

// The host's half of the chunked-table contract: global row ids (from
// `row_ids`) -> (chunk id, local row) at the port partition. Exact integer
// arithmetic here, none in the graph. Throws when an id is outside the
// table -- a Gather does not, it reads something.
inline void split_by_partition(const std::vector<int64_t>& global, const PortPlan& plan,
                               std::vector<int32_t>& chunk_out, std::vector<int64_t>& local_out) {
    chunk_out.resize(global.size());
    local_out.resize(global.size());
    const int64_t per = static_cast<int64_t>(plan.rows_per_chunk);
    for (size_t i = 0; i < global.size(); ++i) {
        const int64_t g = global[i];
        if (g < 0 || g >= static_cast<int64_t>(plan.total_rows)) {
            throw std::runtime_error(log::format(
                "ngram row id %lld outside the table's %zu rows", static_cast<long long>(g),
                plan.total_rows));
        }
        chunk_out[i] = static_cast<int32_t>(g / per);
        local_out[i] = g % per;
    }
}

}  // namespace lgc::ngram
