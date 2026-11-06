// The MTP head-fit capture's file format (src/exec/mtp_capture.h), device-free.
//
// The writer's bytes for a fixed synthetic capture (three prompt chunks on two
// lanes) are compared with tests/python/fixtures/mtp_capture_fixture.{h,z,tok}.bin,
// which tests/python/test_mtp_fit.py reads with the tool's own reader: one set
// of bytes pins both sides.
//
// Red mutations seen failing while this was written:
// (a) a chunk at position 0 not opening a request (req stuck at 0) -> the
//     token index differs from the fixture; (b) the count not rewritten after
//     an append -> the header bytes differ; (c) the z rows written into the h
//     file -> the h file differs; (d) O_EXCL dropped -> the refuses-an-existing-
//     file case fails; (e) the off case without its unsetenv is vacuous when
//     the caller exported ARCINT_MTP_CAPTURE (seen failing with the variable
//     set); (f) the file offset left at 0 after the header's pwrite -> the
//     first records overwrite the header: this one was a real defect of the
//     first version of the writer, found by the fixture comparison.
#include "exec/mtp_capture.h"
#include "harness.h"

#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::string fixture(const char* ext) {
    return slurp(std::string(ARCINT_SOURCE_DIR) + "/tests/python/fixtures/mtp_capture_fixture." + ext + ".bin");
}

// the fixture's rows: record k, value d: h = 10 k + d, z = -(10 k + d) - 0.5
void rows_of(size_t k0, size_t n, int dim, std::vector<float>& h, std::vector<float>& z) {
    h.clear();
    z.clear();
    for (size_t k = k0; k < k0 + n; ++k)
        for (int d = 0; d < dim; ++d) {
            h.push_back(static_cast<float>(10 * k + d));
            z.push_back(-static_cast<float>(10 * k + d) - 0.5f);
        }
}

std::string base_name(const char* tag) { return "/tmp/arcint-mtp-capture-" + std::to_string(getpid()) + "-" + tag; }

void remove_all(const std::string& base) {
    for (const char* e : { ".h", ".z", ".tok" }) std::remove((base + e).c_str());
}

bool write_fixture_capture(const std::string& base) {
    lgc::MtpCapture cap;
    std::string     err;
    if (!cap.open(base, 4, err)) return false;
    std::vector<float> h, z;
    const int          a[] = { 11, 12, 13 }, b[] = { 21, 22 }, c[] = { 14, 15 };
    rows_of(0, 3, 4, h, z);
    if (!cap.append(0, 0, a, 3, h.data(), z.data())) return false;
    rows_of(3, 2, 4, h, z);
    if (!cap.append(1, 0, b, 2, h.data(), z.data())) return false;
    rows_of(5, 2, 4, h, z);
    return cap.append(0, 3, c, 2, h.data(), z.data());
}

}  // namespace

TEST(mtp_capture_header_round_trips) {
    lgc::MtpCaptureHeader h;
    h.kind = lgc::kCaptureRowsZ; h.dim = 5120; h.dtype = lgc::kCaptureF32; h.count = 92000;
    unsigned char b[lgc::kMtpCaptureHeaderBytes];
    lgc::mtp_capture_header_encode(h, b);
    CHECK(std::memcmp(b, "ARCMTPC1", 8) == 0);
    lgc::MtpCaptureHeader g;
    CHECK(lgc::mtp_capture_header_decode(b, g));
    CHECK_EQ(g.kind, h.kind);
    CHECK_EQ(g.dim, h.dim);
    CHECK_EQ(g.count, h.count);
    b[0] = 'X';   // a file of another format is refused
    CHECK(!lgc::mtp_capture_header_decode(b, g));
}

TEST(mtp_capture_bytes_match_the_shared_fixture) {
    const std::string base = base_name("fx");
    remove_all(base);
    CHECK(write_fixture_capture(base));
    CHECK(slurp(base + ".h") == fixture("h"));
    CHECK(slurp(base + ".z") == fixture("z"));
    CHECK(slurp(base + ".tok") == fixture("tok"));
    remove_all(base);
}

TEST(mtp_capture_aligns_rows_tokens_and_requests) {
    // two lanes interleaved: the records of one request stay in position order,
    // (req, pos) is the key, and the second chunk of lane 0 continues request 0
    const std::string base = base_name("al");
    remove_all(base);
    CHECK(write_fixture_capture(base));
    const std::string idx = slurp(base + ".tok");
    CHECK_EQ(idx.size(), lgc::kMtpCaptureHeaderBytes + 7 * 3 * sizeof(int32_t));
    const int32_t* r = reinterpret_cast<const int32_t*>(idx.data() + lgc::kMtpCaptureHeaderBytes);
    const int32_t  want[7][3] = { {0,0,11}, {0,1,12}, {0,2,13}, {1,0,21}, {1,1,22}, {0,3,14}, {0,4,15} };
    for (int k = 0; k < 7; ++k)
        for (int j = 0; j < 3; ++j) CHECK_EQ(r[k * 3 + j], want[k][j]);
    remove_all(base);
}

TEST(mtp_capture_refuses_an_existing_file) {
    const std::string base = base_name("ex");
    remove_all(base);
    CHECK(write_fixture_capture(base));
    lgc::MtpCapture second;
    std::string     err;
    CHECK(!second.open(base, 4, err));              // never appended to or overwritten
    CHECK(err.find("new base name") != std::string::npos);
    CHECK(slurp(base + ".h") == fixture("h"));      // the first run's file is untouched
    remove_all(base);
}

TEST(mtp_capture_a_chunk_without_a_start_is_refused) {
    const std::string base = base_name("ns");
    remove_all(base);
    lgc::MtpCapture cap;
    std::string     err;
    CHECK(cap.open(base, 4, err));
    std::vector<float> h, z;
    rows_of(0, 1, 4, h, z);
    const int t[] = { 5 };
    CHECK(!cap.append(0, 7, t, 1, h.data(), z.data()));   // position 7 of a lane that never began
    CHECK_EQ(cap.records(), 0u);
    remove_all(base);
}

TEST(mtp_capture_off_by_default) {
    unsetenv("ARCINT_MTP_CAPTURE");
    CHECK(!lgc::mtp_capture_on());
    CHECK(lgc::mtp_capture_path() == nullptr);
    setenv("ARCINT_MTP_CAPTURE", "", 1);   // empty is off too
    CHECK(!lgc::mtp_capture_on());
    setenv("ARCINT_MTP_CAPTURE", "/tmp/x", 1);
    CHECK(lgc::mtp_capture_on());
    unsetenv("ARCINT_MTP_CAPTURE");
}
