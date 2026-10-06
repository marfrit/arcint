#pragma once

// ple-disk-backend (campaign docs/campaigns/ple-disk-backend.md, design note
// docs/design-ple-disk-backend.md): the n-gram table as a per-forward disk
// staging buffer.
//
// The served path (`backend_ov.cpp::bind_ngram_ports`) binds all 26.82 GiB of
// the table into USM host memory at load and keeps it. The reference's own
// default is a DISK backend (`code`: `~/src/FreeToken-ref/python/freetoken/
// engine/config.py`:32 `ple_backend: str = "disk"`; `models/qwen4_exp/
// ple_disk.py` `DiskRowTable`), which allocates bounded pinned staging
// (`max_graph_rows` decode lanes, `max_extend_tokens` prefill tokens, each
// `x heads x head_dim`) and fills only the rows a forward names.
//
// This header is the device-free half: given the port's static geometry
// (`staging_rows x row_bytes`, read off the compiled model like every other
// `ngram_table.K` port) and the on-disk table, it (a) validates the geometry
// against the source, (b) turns a forward's host-computed hashed row ids into
// the staging slots the graph gathers, and (c) `pread`s exactly those rows
// into the staging buffer. The rows and their bytes are identical to the
// pinned path's, in the same token order, decoded by the same
// `exec/ngram_gather.h` path -- so the output is byte-identical by
// construction, which is the campaign's gate.
//
// The first cut stages in token order with NO dedup (the reference's own
// `fill` order): slot i is the i-th named row, so the local ids are the
// arange and a repeated row is read twice. Dedup and the reference's io_uring
// (`FREETOKEN_PLE_IO_URING`) are named out of scope in the design note.

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/gguf.h"
#include "exec/ngram_ports.h"
#include "util/log.h"

namespace lgc::ngram {

// The staging port's contract: `staging_rows` is the port's static row count
// (`T_max x num_ngram_heads`), `row_bytes` the IQ4_NL row width the graph
// decodes, `table_rows` the on-disk table's own row count. Nothing carries a
// config: the port shapes are the contract (`exec/ngram_ports.h`), and the
// source's row count comes off the GGUF tensor / ARCINGRM header.
struct StagingGeometry {
    uint64_t staging_rows = 0;
    size_t   row_bytes    = 0;
    uint64_t table_rows   = 0;
};

// The served geometry's own helper: `heads` is `num_ngram_heads`, `max_tokens`
// the runtime's largest forward block, `table_rows` the source's row count.
inline StagingGeometry staging_geometry(uint64_t max_tokens, uint64_t heads, size_t row_bytes,
                                        uint64_t table_rows) {
    StagingGeometry g;
    g.staging_rows = max_tokens * heads;
    g.row_bytes    = row_bytes;
    g.table_rows   = table_rows;
    return g;
}

// Does the staging port match the GGUF tensor? "" when it does; otherwise the
// refusal, naming what disagrees. The served IR's staging port is a SMALL
// chunk -- its row count is the staging bound, NOT the table's -- so unlike
// `check_table_source` this validates the source's full shape (type, row
// width, row count, byte size) and requires the port's row count to be a
// bounded window into it.
inline std::string check_staging_geometry(const gguf::TensorInfo& t, size_t tensor_bytes,
                                          const StagingGeometry& g) {
    if (t.ggml_type != kIq4NlType) {
        return log::format("%s is %s, the staging port carries IQ4_NL rows", t.name.c_str(),
                           gguf::type_name(t.ggml_type).c_str());
    }
    if (t.dims.size() != 2) {
        return log::format("%s has %zu dims, expected 2 [width, rows]", t.name.c_str(),
                           t.dims.size());
    }
    const size_t width = static_cast<size_t>(t.dims[0]);
    const size_t rows  = static_cast<size_t>(t.dims[1]);
    if (width % kIq4NlBlockElems != 0 ||
        width / kIq4NlBlockElems * kIq4NlBlockBytes != g.row_bytes) {
        return log::format("%s rows are %zu elements = %zu IQ4_NL bytes, the staging port takes %zu",
                           t.name.c_str(), width, width / kIq4NlBlockElems * kIq4NlBlockBytes,
                           g.row_bytes);
    }
    if (rows != g.table_rows) {
        return log::format("%s has %zu rows, the staging geometry says %llu", t.name.c_str(), rows,
                           static_cast<unsigned long long>(g.table_rows));
    }
    if (tensor_bytes != rows * g.row_bytes) {
        return log::format("%s is %zu bytes, %zu rows x %zu would be %zu", t.name.c_str(),
                           tensor_bytes, rows, g.row_bytes, rows * g.row_bytes);
    }
    if (g.staging_rows == 0) {
        return log::format("the staging port holds 0 rows; it must cover at least one forward's "
                           "T x heads");
    }
    if (g.staging_rows > g.table_rows) {
        return log::format("the staging port holds %llu rows, more than the table's %llu",
                           static_cast<unsigned long long>(g.staging_rows),
                           static_cast<unsigned long long>(g.table_rows));
    }
    return "";
}

// One forward's staging plan: the host-computed hashed global row ids -> the
// staging slot the graph gathers (`local`, slot i = the i-th named row) and
// the on-disk row each slot is filled from (`rows`, = the global id). Refuses
// -- by name, before any bytes move -- an id outside the table (a Gather does
// not throw; it reads something, so silence here is a wrong row) and a forward
// that names more rows than the port holds.
inline void plan_staging_fill(const std::vector<int64_t>& global, const StagingGeometry& g,
                              std::vector<int64_t>& local, std::vector<uint64_t>& rows) {
    if (global.size() > g.staging_rows) {
        throw std::runtime_error(log::format(
            "ngram staging: this forward names %zu rows, the staging port holds %llu "
            "(T x heads overruns its static bound)", global.size(),
            static_cast<unsigned long long>(g.staging_rows)));
    }
    local.resize(global.size());
    rows.resize(global.size());
    for (size_t i = 0; i < global.size(); ++i) {
        const int64_t id = global[i];
        if (id < 0 || static_cast<uint64_t>(id) >= g.table_rows) {
            throw std::runtime_error(log::format(
                "ngram staging: row id %lld is outside the %llu-row table; refusing rather than "
                "gathering from the wrong row",
                static_cast<long long>(id), static_cast<unsigned long long>(g.table_rows)));
        }
        local[i] = static_cast<int64_t>(i);
        rows[i]  = static_cast<uint64_t>(id);
    }
}

// The reference's fill: `pread` exactly `rows` (in slot order) from the table
// payload of `fd`, row `r` at `payload_base + r * row_bytes`, into `dst` (slot
// i at `dst + i * row_bytes`). Synchronous `pread`; the reference's io_uring
// is out of scope. A short read is a named error, not a silently short buffer.
inline void pread_staging_rows(int fd, uint64_t payload_base, size_t row_bytes,
                               const std::vector<uint64_t>& rows, uint8_t* dst) {
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint64_t off = payload_base + rows[i] * row_bytes;
        size_t         done = 0;
        while (done < row_bytes) {
            const ssize_t n = ::pread(fd, dst + i * row_bytes + done, row_bytes - done,
                                      static_cast<off_t>(off + done));
            if (n <= 0) {
                throw std::runtime_error(log::format(
                    "ngram staging: read of %zu bytes at offset %llu returned %zd (row %llu)",
                    row_bytes, static_cast<unsigned long long>(off), n,
                    static_cast<unsigned long long>(rows[i])));
            }
            done += static_cast<size_t>(n);
        }
    }
}

// Plan + fill in one call for an open table file. Returns the local ids the
// graph gathers; `dst` must hold `g.staging_rows * g.row_bytes`. The caller
// owns the file descriptor and the USM staging tensor.
inline std::vector<int64_t> stage_from_file(int fd, uint64_t payload_base, size_t row_bytes,
                                            const std::vector<int64_t>& global,
                                            const StagingGeometry& g, uint8_t* dst) {
    std::vector<int64_t>  local;
    std::vector<uint64_t> rows;
    plan_staging_fill(global, g, local, rows);
    pread_staging_rows(fd, payload_base, row_bytes, rows, dst);
    return local;
}

}  // namespace lgc::ngram
