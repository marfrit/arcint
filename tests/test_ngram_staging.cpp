// ple-disk-backend (docs/campaigns/ple-disk-backend.md, design note
// docs/design-ple-disk-backend.md): the device-free, red-first cells for the
// staged n-gram table.
//
// The served path (`bind_ngram_ports`) pins all 26.82 GiB of the table into
// USM host memory. The staged path fills a bounded port with only the rows a
// forward names. These cases make the mechanism fail before it passes:
//
//   1. a forward that names more rows than the port holds is REFUSED, by name;
//   2. a hashed id outside the table is REFUSED, never gathered from the
//      wrong row;
//   3. the staged row i is the row the id names (row-order fidelity);
//   4. the staged gather output is BYTE-IDENTICAL to the pinned gather on the
//      same synthetic table -- the campaign's numeric gate, device-free;
//   5. the resident staging term is the forward, not the table, and the host
//      fit admits staged where it refuses pinned.
//
// `tests/test_ngram_gather.cpp` remains the untouched byte-exact oracle for
// the decode itself; this file tests the staging on top of it.
#include "core/gguf.h"
#include "core/ngram_header.h"
#include "exec/fit.h"
#include "exec/ngram_gather.h"
#include "exec/ngram_ports.h"
#include "exec/ngram_staging.h"
#include "exec/ngram_reader.h"
#include "harness.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lgc;

namespace {

constexpr int64_t  kTableRows  = 100000;
constexpr size_t   kRowBytes   = 90;    // 160 IQ4_NL elements, five 32-blocks
constexpr size_t   kRowElems   = 160;
constexpr uint64_t kHeads      = 16;    // (ngram_size - 1) * heads_per_ngram
constexpr uint64_t kMaxTokens  = 512;   // a forward's T for this fixture
constexpr uint64_t kStagingRows = kMaxTokens * kHeads;  // 8192

std::string temp_path(const char* stem) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "/tmp/arcint-ngram-staging-%s-%d.bin", stem, ::getpid());
    return std::string(buf);
}

// A deterministic, ROW-VARYING IQ4_NL payload: a constant fill would hide a
// wrong-row defect (the same BY_TOKEN NaN lesson the gather test cites). Row r,
// byte j is a function of both, so a staged row can be checked against the row
// its id names.
inline uint8_t table_byte(uint64_t r, size_t j) {
    return static_cast<uint8_t>((r * 131u + j * 37u + (r >> 3) * 7u + 11u) & 0xFFu);
}

void write_table(const std::string& path, uint32_t rows, size_t row_bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write("ARCINGRM", 8);
    const uint32_t type = ngram::kIq4NlType, cols = 160, reserved = 0;
    f.write(reinterpret_cast<const char*>(&type), 4);
    f.write(reinterpret_cast<const char*>(&cols), 4);
    f.write(reinterpret_cast<const char*>(&rows), 4);  // header.n_rows is uint32
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    std::vector<uint8_t> payload(static_cast<size_t>(rows) * row_bytes);
    for (uint64_t r = 0; r < rows; ++r)
        for (size_t j = 0; j < row_bytes; ++j) payload[r * row_bytes + j] = table_byte(r, j);
    f.write(reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
}

std::vector<uint8_t> read_payload(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return std::vector<uint8_t>(all.begin() + static_cast<long>(ngram::kHeaderBytes), all.end());
}

gguf::TensorInfo table_tensor(uint32_t rows = static_cast<uint32_t>(kTableRows), size_t row_bytes = kRowBytes) {
    gguf::TensorInfo t;
    t.name      = ngram::kTableTensor;
    t.dims      = {kRowElems, rows};
    t.ggml_type = ngram::kIq4NlType;
    t.n_elements = kRowElems * rows;
    (void)row_bytes;
    return t;
}

ngram::StagingGeometry geometry() {
    return ngram::staging_geometry(kMaxTokens, kHeads, kRowBytes, kTableRows);
}

// A forward's hashed row ids: every table edge plus a mid range, with a
// deliberate duplicate.
std::vector<int64_t> forward_ids() {
    std::vector<int64_t> ids;
    for (int64_t i = 0; i < 64; ++i) ids.push_back(i);                       // low edge
    ids.push_back(kTableRows - 1);                                           // top edge
    ids.push_back(0);                                                        // duplicate
    for (int64_t i = 0; i < 256; ++i) ids.push_back((i * 9973) % kTableRows);  // scattered
    return ids;
}

}  // namespace

TEST(ngram_staging_refuses_a_forward_that_overruns_the_port) {
    const auto g = geometry();
    std::vector<int64_t> over(static_cast<size_t>(g.staging_rows) + 1, 0);
    std::vector<int64_t> local;
    std::vector<uint64_t> rows;
    bool threw = false;
    try {
        ngram::plan_staging_fill(over, g, local, rows);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("staging port holds") != std::string::npos;
    }
    CHECK(threw);
    // exactly the bound is allowed
    std::vector<int64_t> at_bound(static_cast<size_t>(g.staging_rows), 0);
    ngram::plan_staging_fill(at_bound, g, local, rows);
    CHECK_EQ(local.size(), at_bound.size());
}

TEST(ngram_staging_refuses_an_out_of_range_table_id) {
    const auto g = geometry();
    std::vector<int64_t> local;
    std::vector<uint64_t> rows;
    const std::vector<int64_t> bad_ids = {static_cast<int64_t>(kTableRows),
                                          static_cast<int64_t>(kTableRows) + 7, -1LL};
    for (const int64_t bad : bad_ids) {
        bool threw = false;
        try {
            ngram::plan_staging_fill({bad}, g, local, rows);
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()).find("outside the") != std::string::npos &&
                    std::string(e.what()).find("wrong row") != std::string::npos;
        }
        CHECK(threw);
    }
    // a valid id is not refused and lands at slot 0
    ngram::plan_staging_fill({kTableRows - 1}, g, local, rows);
    CHECK_EQ(local, std::vector<int64_t>{0});
    CHECK_EQ(rows, std::vector<uint64_t>{static_cast<uint64_t>(kTableRows - 1)});
}

TEST(ngram_staging_places_row_i_at_the_row_the_id_names) {
    const std::string path = temp_path("order");
    write_table(path, static_cast<uint32_t>(kTableRows), kRowBytes);
    const auto payload = read_payload(path);

    const int fd = ::open(path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    const auto ids = forward_ids();
    const auto g   = geometry();
    std::vector<uint8_t> staging(static_cast<size_t>(g.staging_rows) * g.row_bytes, 0);
    const auto local = ngram::stage_from_file(fd, ngram::kHeaderBytes, kRowBytes, ids, g, staging.data());
    ::close(fd);

    CHECK_EQ(local.size(), ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        CHECK_EQ(local[i], static_cast<int64_t>(i));  // slot i is the i-th named row
        const uint8_t* staged = staging.data() + i * g.row_bytes;
        const uint8_t* named  = payload.data() + static_cast<size_t>(ids[i]) * g.row_bytes;
        CHECK(std::memcmp(staged, named, g.row_bytes) == 0);
    }
    ::unlink(path.c_str());
}

TEST(ngram_staging_output_matches_the_pinned_gather_byte_exact) {
    const std::string path = temp_path("exact");
    write_table(path, static_cast<uint32_t>(kTableRows), kRowBytes);
    const auto payload = read_payload(path);

    const auto ids = forward_ids();
    const auto g   = geometry();

    // Pinned path: gather directly from the whole in-memory table.
    std::vector<uint32_t> pinned_idx(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) pinned_idx[i] = static_cast<uint32_t>(ids[i]);
    std::vector<float> pinned(ids.size() * kRowElems);
    ngram::gather_dequant(ngram::kIq4NlType, payload.data(), kRowBytes, kRowElems, pinned_idx,
                          pinned.data());

    // Staged path: pread the named rows into a bounded buffer, gather by slot.
    const int fd = ::open(path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    std::vector<uint8_t> staging(static_cast<size_t>(g.staging_rows) * g.row_bytes, 0);
    const auto local = ngram::stage_from_file(fd, ngram::kHeaderBytes, kRowBytes, ids, g, staging.data());
    ::close(fd);
    std::vector<uint32_t> staged_idx(local.size());
    for (size_t i = 0; i < local.size(); ++i) staged_idx[i] = static_cast<uint32_t>(local[i]);
    std::vector<float> staged(local.size() * kRowElems);
    ngram::gather_dequant(ngram::kIq4NlType, staging.data(), kRowBytes, kRowElems, staged_idx,
                          staged.data());

    CHECK_EQ(pinned.size(), staged.size());
    CHECK(std::memcmp(pinned.data(), staged.data(), pinned.size() * sizeof(float)) == 0);
    ::unlink(path.c_str());
}

TEST(ngram_staging_geometry_refuses_a_source_that_is_not_the_ports_format) {
    const auto g   = geometry();
    const size_t bytes = static_cast<size_t>(kTableRows) * kRowBytes;
    CHECK_EQ(ngram::check_staging_geometry(table_tensor(), bytes, g), std::string(""));
    // wrong ggml type
    auto bad = table_tensor();
    bad.ggml_type = 2;  // Q4_0
    CHECK(ngram::check_staging_geometry(bad, bytes, g).find("IQ4_NL") != std::string::npos);
    // wrong row width
    auto narrow = table_tensor();
    narrow.dims = {128, static_cast<uint64_t>(kTableRows)};
    CHECK(ngram::check_staging_geometry(narrow, bytes, g).find("staging port takes 90") !=
          std::string::npos);
    // source row count disagrees with the geometry
    auto rows_off = geometry();
    rows_off.table_rows = kTableRows - 1;
    CHECK(ngram::check_staging_geometry(table_tensor(), bytes, rows_off).find("staging geometry says") !=
          std::string::npos);
    // a byte size that is not rows x row_bytes
    CHECK(ngram::check_staging_geometry(table_tensor(), bytes - 1, g).find("would be") !=
          std::string::npos);
    // a port longer than the table
    auto long_port = geometry();
    long_port.staging_rows = kTableRows + 1;
    CHECK(ngram::check_staging_geometry(table_tensor(), bytes, long_port).find("more than the table") !=
          std::string::npos);
}

TEST(ngram_staging_bytes_is_the_forward_not_the_table) {
    // The served geometry: T=512, H=16, 90 B a row over the 320,001,536-row
    // shipped table.
    const uint64_t staging = ngram_staging_bytes(kMaxTokens * kHeads, kRowBytes);
    CHECK_EQ(staging, static_cast<uint64_t>(512 * 16 * 90));  // 737,280 B = 720 KiB
    const uint64_t full = ngram_table_bytes(ngram::kIq4NlType, 320001536ULL * 160ULL);
    CHECK_EQ(full, 28800138240ULL);  // 26.82 GiB, the measured pin (window-050 §4.8)
    CHECK(staging * 1000 < full);    // three orders smaller

    // FIX D + FIX E contention, priced: a 30 GiB expert pool plus the pinned
    // table overruns a 48 GiB host; the staged term does not.
    constexpr uint64_t kHost = 48ULL << 30;
    constexpr uint64_t kPool = 30ULL << 30;
    CHECK(host_ram_fit_must_refuse(full, kPool, 0, kHost, 0));
    CHECK(!host_ram_fit_must_refuse(staging, kPool, 0, kHost, 0));
}

// Strata's PLE reader (exec/ngram_reader.h): O_DIRECT page reads on an I/O
// pool, page dedupe, a row cache. The staged bytes must be the synchronous
// pread's, byte for byte, for the fixture's edges, its duplicate and the last
// row of the file (a short last page). The file lives in the working
// directory, not /tmp: tmpfs refuses O_DIRECT, and the reader falls back to
// buffered reads there.
namespace {
std::string reader_path(const char* stem) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "arcint-ngram-reader-%s-%d.bin", stem, ::getpid());
    return std::string(buf);
}

std::vector<uint8_t> stage_by_pread(const std::string& path, const std::vector<int64_t>& ids) {
    const auto g = geometry();
    std::vector<uint8_t> out(static_cast<size_t>(g.staging_rows) * g.row_bytes, 0);
    const int fd = ::open(path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    ngram::stage_from_file(fd, ngram::kHeaderBytes, kRowBytes, ids, g, out.data());
    ::close(fd);
    return out;
}

std::vector<uint8_t> stage_by_reader(ngram::RowReader& r, const std::vector<int64_t>& ids) {
    const auto g = geometry();
    std::vector<uint8_t> out(static_cast<size_t>(g.staging_rows) * g.row_bytes, 0);
    std::vector<int64_t>  local;
    std::vector<uint64_t> rows;
    ngram::plan_staging_fill(ids, g, local, rows);
    r.collect(r.issue(rows, out.data()));
    return out;
}
}  // namespace

TEST(ngram_reader_stages_the_preads_bytes) {
    const std::string path = reader_path("bytes");
    write_table(path, static_cast<uint32_t>(kTableRows), kRowBytes);
    const auto ids = forward_ids();
    const auto want = stage_by_pread(path, ids);
    ngram::ReaderOptions opt;
    opt.cache_rows = 0;  // every row from the file
    {
        ngram::RowReader r(path, ngram::kHeaderBytes, kTableRows, kRowBytes, opt);
        CHECK(stage_by_reader(r, ids) == want);
        const auto s = r.stats();
        CHECK_EQ(s.requests, static_cast<uint64_t>(ids.size()));
        CHECK_EQ(s.cache_hits, static_cast<uint64_t>(0));
    }
    ::unlink(path.c_str());
}

TEST(ngram_reader_serves_a_repeated_forward_from_its_row_cache) {
    const std::string path = reader_path("cache");
    write_table(path, static_cast<uint32_t>(kTableRows), kRowBytes);
    const auto ids = forward_ids();
    const auto want = stage_by_pread(path, ids);
    {
        ngram::RowReader r(path, ngram::kHeaderBytes, kTableRows, kRowBytes);
        CHECK(stage_by_reader(r, ids) == want);
        const uint64_t reads = r.stats().reads;
        CHECK(stage_by_reader(r, ids) == want);
        const auto s = r.stats();
        CHECK_EQ(s.reads, reads);                                   // no new page read
        CHECK(s.cache_hits >= static_cast<uint64_t>(ids.size()));  // the second forward, all hits
    }
    ::unlink(path.c_str());
}

TEST(ngram_reader_reads_rows_that_share_a_page_once) {
    const std::string path = reader_path("dedup");
    write_table(path, static_cast<uint32_t>(kTableRows), kRowBytes);
    // 64 consecutive rows of 90 B from row 1000 (file bytes 90,024-95,783)
    // touch three pages: at most three page reads, not 64.
    std::vector<int64_t> ids;
    for (int64_t i = 1000; i < 1064; ++i) ids.push_back(i);
    const auto want = stage_by_pread(path, ids);
    ngram::ReaderOptions opt;
    opt.cache_rows = 0;
    {
        ngram::RowReader r(path, ngram::kHeaderBytes, kTableRows, kRowBytes, opt);
        CHECK(stage_by_reader(r, ids) == want);
        const auto s = r.stats();
        CHECK(s.reads <= 3);
        CHECK_EQ(s.reads + s.dedup_rows, static_cast<uint64_t>(ids.size()));
    }
    ::unlink(path.c_str());
}
