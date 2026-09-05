// Stage 0 of docs/design-gguf-native.md: the GGUF v3 reader and host
// dequantizers, checked against tools/gguf_fixture.py's fixture (a tiny file
// with one tensor of each type the four decoders cover) and its reference
// dequantization (computed by gguf-py's own quants.dequantize, not by
// arcint, so this is an independent check).
#include "core/gguf.h"
#include "core/gguf_dequant.h"
#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace lgc;
using nlohmann::json;

namespace {

std::string fixture_path() {
    return std::string(ARCINT_SOURCE_DIR) + "/tests/fixtures/qwen35-tiny.gguf";
}

std::vector<uint8_t> read_whole_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

// A private, per-process temp file (mkstemp), removed on destruction --
// tests/test_artifact.cpp's TempArtifactDir pattern, for a single file.
class TempFile {
public:
    explicit TempFile(const std::vector<uint8_t>& bytes) {
        const char* tmpdir = std::getenv("TMPDIR");
        std::string tmpl_s =
            std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") + "/arcint-test-gguf-XXXXXX";
        std::vector<char> tmpl(tmpl_s.begin(), tmpl_s.end());
        tmpl.push_back('\0');
        const int fd = ::mkstemp(tmpl.data());
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        path_ = tmpl.data();
        FILE* f = ::fdopen(fd, "wb");
        if (f == nullptr) { ::close(fd); throw std::runtime_error("fdopen failed"); }
        if (!bytes.empty()) std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    }
    ~TempFile() { ::unlink(path_.c_str()); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

// Finds the (unique) byte offset of a length-prefixed GGUF string
// ("\x14\0\0\0\0\0\0\0nextn.eh_proj.weight" style: u64 LE length, then the
// bytes) inside `file`. Independent of gguf.cpp: a plain byte search, so
// this test does not lean on the reader under test to find what to corrupt.
size_t find_gguf_string(const std::vector<uint8_t>& file, const std::string& s) {
    std::vector<uint8_t> needle(8, 0);
    uint64_t len = s.size();
    std::memcpy(needle.data(), &len, 8);
    needle.insert(needle.end(), s.begin(), s.end());
    const auto it = std::search(file.begin(), file.end(), needle.begin(), needle.end());
    if (it == file.end()) throw std::runtime_error("string not found in fixture: " + s);
    return static_cast<size_t>(it - file.begin());
}

void patch_u32(std::vector<uint8_t>& file, size_t offset, uint32_t value) {
    std::memcpy(file.data() + offset, &value, 4);
}

struct RefTensor {
    std::vector<int64_t> shape;  // numpy [out, in] order
    std::string          qtype;
    size_t                byte_offset;
    size_t                n_floats;
};

// tools/gguf_fixture.py's manifest + sidecar .bin: {shape, qtype,
// byte_offset, n_floats} per tensor, and a flat float32 blob.
struct ReferenceDequant {
    std::vector<float>                    blob;
    std::map<std::string, RefTensor, std::less<>> tensors;
};

ReferenceDequant load_reference() {
    const std::string dir  = std::string(ARCINT_SOURCE_DIR) + "/tests/fixtures/";
    std::ifstream     jf(dir + "qwen35-tiny.dequant.json");
    if (!jf.good()) throw std::runtime_error("cannot read dequant manifest");
    json manifest;
    jf >> manifest;

    ReferenceDequant ref;
    const std::vector<uint8_t> raw = read_whole_file(dir + manifest.at("bin").get<std::string>());
    ref.blob.resize(raw.size() / sizeof(float));
    std::memcpy(ref.blob.data(), raw.data(), raw.size());

    for (const auto& [name, entry] : manifest.at("tensors").items()) {
        RefTensor t;
        for (const auto& d : entry.at("shape")) t.shape.push_back(d.get<int64_t>());
        t.qtype       = entry.at("qtype").get<std::string>();
        t.byte_offset = entry.at("byte_offset").get<size_t>();
        t.n_floats    = entry.at("n_floats").get<size_t>();
        ref.tensors.emplace(name, std::move(t));
    }
    return ref;
}

bool throws_containing(const std::string& path, const std::string& needle) {
    try {
        gguf::GgufFile::open(path);
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find(needle) == std::string::npos) {
            std::fprintf(stderr, "  (threw, but message %s did not contain \"%s\")\n",
                         msg.c_str(), needle.c_str());
            return false;
        }
        return true;
    }
    return false;
}

}  // namespace

// ---------------------------------------------------------------- metadata

TEST(gguf_opens_fixture) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    CHECK_EQ(f.alignment(), 32u);
    CHECK_EQ(f.tensors().size(), 6u);
}

TEST(gguf_metadata_string_and_int_and_float) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());

    // .value_or() with a deliberately-wrong default: a missing key still
    // fails the CHECK_EQ cleanly (no UB from dereferencing an empty
    // optional) while has_value() is checked right beside it.
    const auto arch = f.get_string("general.architecture");
    CHECK(arch.has_value());
    CHECK_EQ(arch.value_or("<absent>"), std::string("qwen35"));

    const auto align = f.get_int("general.alignment");
    CHECK(align.has_value());
    CHECK_EQ(align.value_or(-1), int64_t{32});

    const auto block_count = f.get_int("qwen35.block_count");
    CHECK(block_count.has_value());
    CHECK_EQ(block_count.value_or(-1), int64_t{2});

    const auto embd = f.get_int("qwen35.embedding_length");
    CHECK(embd.has_value());
    CHECK_EQ(embd.value_or(-1), int64_t{64});

    const auto freq = f.get_float("qwen35.rope.freq_base");
    CHECK(freq.has_value());
    CHECK_NEAR(freq.value_or(-1.0), 10000.0, 1e-6);

    const auto pre = f.get_string("tokenizer.ggml.pre");
    CHECK(pre.has_value());
    CHECK_EQ(pre.value_or("<absent>"), std::string("qwen2"));

    CHECK(!f.get_string("no.such.key").has_value());
    CHECK(!f.get_int("tokenizer.ggml.pre").has_value());  // wrong shape, not a throw
}

TEST(gguf_metadata_string_array) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());

    const auto tokens = f.get_string_array("tokenizer.ggml.tokens");
    CHECK(tokens.has_value());
    const std::vector<std::string> expected = {"<pad>", "<s>",  "</s>", "hello",
                                                ",",     " world", "!",    "\xe2\x96\x81tok"};
    CHECK_EQ(tokens.value_or(std::vector<std::string>{}), expected);
}

// ------------------------------------------------------------------ tensors

TEST(gguf_tensor_list_matches_fixture) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());

    struct Expect {
        const char*           name;
        std::vector<uint64_t> dims;  // ggml order: dims[0] fastest-varying
        int32_t                type;
        size_t                 n_elements;
        size_t                 bytes;
    };
    const std::vector<Expect> expect = {
        {"blk.0.ffn_gate.weight", {512, 64}, static_cast<int32_t>(gguf::GgmlType::Q4_K), 32768, 18432},
        {"blk.0.ffn_down.weight", {256, 512}, static_cast<int32_t>(gguf::GgmlType::Q6_K), 131072, 107520},
        {"blk.0.ssm_out.weight", {256, 32}, static_cast<int32_t>(gguf::GgmlType::Q5_K), 8192, 5632},
        {"nextn.eh_proj.weight", {64, 16}, static_cast<int32_t>(gguf::GgmlType::Q8_0), 1024, 1088},
        {"output_norm.weight", {64}, static_cast<int32_t>(gguf::GgmlType::F32), 64, 256},
        {"token_embd.weight", {64, 8}, static_cast<int32_t>(gguf::GgmlType::F16), 512, 1024},
    };

    for (const Expect& e : expect) {
        const gguf::TensorInfo* t = f.tensor(e.name);
        CHECK(t != nullptr);
        if (t == nullptr) continue;
        CHECK_EQ(t->dims, e.dims);
        CHECK_EQ(t->ggml_type, e.type);
        CHECK_EQ(t->n_elements, e.n_elements);
        CHECK_EQ(f.bytes(*t), e.bytes);
        CHECK(f.data(*t) != nullptr);
        // The tensor's bytes must lie inside the mapped file.
        CHECK(f.data_offset() + t->offset + f.bytes(*t) <= f.file_size());
    }

    CHECK(f.tensor("does.not.exist") == nullptr);
}

// -------------------------------------------------------------- dequantize

TEST(gguf_dequantize_matches_reference) {
    gguf::GgufFile   f   = gguf::GgufFile::open(fixture_path());
    ReferenceDequant ref = load_reference();

    for (const auto& [name, rt] : ref.tensors) {
        const gguf::TensorInfo* t = f.tensor(name);
        CHECK(t != nullptr);
        if (t == nullptr) continue;

        std::vector<float> out;
        gguf::dequantize_tensor(f, *t, out);
        CHECK_EQ(out.size(), rt.n_floats);

        // Exact: both sides compute the same ggml block formula in float32
        // from the same bytes, so this should be bit-identical, not merely
        // close -- a mismatch means an operation order differs somewhere.
        bool all_equal = true;
        double max_abs_diff = 0.0;
        for (size_t i = 0; i < rt.n_floats && i < out.size(); ++i) {
            const float expected = ref.blob[rt.byte_offset / sizeof(float) + i];
            if (expected != out[i]) {
                all_equal = false;
                max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(expected - out[i])));
            }
        }
        if (!all_equal) {
            t::fail(__FILE__, __LINE__,
                    name + ": dequantize_tensor (" + rt.qtype +
                        ") does not bit-match the reference, max abs diff " +
                        std::to_string(max_abs_diff));
        }
    }
}

TEST(gguf_dequantize_f32_and_f16_passthrough) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());

    const gguf::TensorInfo* norm = f.tensor("output_norm.weight");
    CHECK(norm != nullptr);
    if (norm != nullptr) {
        std::vector<float> norm_out;
        gguf::dequantize_tensor(f, *norm, norm_out);
        CHECK_EQ(norm_out.size(), size_t{64});
        // F32 is a byte-for-byte passthrough: compare against the mapped bytes.
        const float* raw = reinterpret_cast<const float*>(f.data(*norm));
        for (size_t i = 0; i < norm_out.size(); ++i) CHECK_EQ(norm_out[i], raw[i]);
    }

    const gguf::TensorInfo* embd = f.tensor("token_embd.weight");
    CHECK(embd != nullptr);
    if (embd != nullptr) {
        std::vector<float> embd_out;
        gguf::dequantize_tensor(f, *embd, embd_out);
        CHECK_EQ(embd_out.size(), size_t{512});
        const uint16_t* raw16 = reinterpret_cast<const uint16_t*>(f.data(*embd));
        for (size_t i = 0; i < embd_out.size(); ++i) {
            CHECK_EQ(embd_out[i], gguf::f16_to_f32(raw16[i]));
        }
    }
}

// ----------------------------------------------------------------- get_scale_min_k4
//
// A hand-built Q4_K block exercising both branches of ggml's
// get_scale_min_k4: sub-blocks 0-3 read their 6-bit scale/min directly
// (j < 4), sub-blocks 4-7 assemble it from two nibbles spread across the
// `scales[12]` field (j >= 4). Worked by hand in the commit that added this
// test (docs/design-gguf-native.md stage 0); see the comment below for the
// arithmetic each expected value comes from.
TEST(gguf_q4k_get_scale_min_k4_both_branches) {
    const uint8_t sc[8] = {5, 10, 15, 20, 25, 30, 35, 40};
    const uint8_t m[8]  = {1, 2, 3, 4, 5, 6, 7, 8};

    // Packing per Q4_K's scales[12] (ggml-common.h / gguf-py's
    // Q4_K.get_scale_min): d[i] = sc[i] | (top 2 bits of sc[i+4] << 6);
    // m[i] = m[i] | (top 2 bits of m[i+4] << 6); md[i] = low 4 bits of
    // sc[i+4] | (low 4 bits of m[i+4] << 4), for i in 0..3.
    uint8_t scales[12];
    for (int i = 0; i < 4; ++i) {
        scales[i]     = (sc[i] & 0x3F) | (((sc[i + 4] >> 4) & 0x3) << 6);
        scales[4 + i] = (m[i] & 0x3F) | (((m[i + 4] >> 4) & 0x3) << 6);
        scales[8 + i] = (sc[i + 4] & 0xF) | ((m[i + 4] & 0xF) << 4);
    }

    std::vector<uint8_t> block(144, 0);
    const uint16_t d_f16    = 0x4000;  // 2.0
    const uint16_t dmin_f16 = 0x3800;  // 0.5
    std::memcpy(block.data() + 0, &d_f16, 2);
    std::memcpy(block.data() + 2, &dmin_f16, 2);
    std::memcpy(block.data() + 4, scales, 12);
    // qs: every byte 0x93 -- low nibble 3 (even sub-blocks), high nibble 9
    // (odd sub-blocks), uniform within a sub-block so every element of it
    // decodes to the same value.
    for (int i = 0; i < 128; ++i) block[16 + i] = 0x93;

    float out[256];
    gguf::dequantize_row(static_cast<int32_t>(gguf::GgmlType::Q4_K), block.data(), 256, out);

    // y = d*sc[k]*q - dmin*m[k]; q = 3 for even k, 9 for odd k.
    const double expect[8] = {
        2.0 * 5 * 3 - 0.5 * 1,   2.0 * 10 * 9 - 0.5 * 2,  2.0 * 15 * 3 - 0.5 * 3,
        2.0 * 20 * 9 - 0.5 * 4,  2.0 * 25 * 3 - 0.5 * 5,  2.0 * 30 * 9 - 0.5 * 6,
        2.0 * 35 * 3 - 0.5 * 7,  2.0 * 40 * 9 - 0.5 * 8,
    };
    for (int k = 0; k < 8; ++k) {
        CHECK_NEAR(out[32 * k], expect[k], 1e-6);
        CHECK_NEAR(out[32 * k + 31], expect[k], 1e-6);  // uniform within the sub-block
    }
}

// -------------------------------------------------------------------- refusals

TEST(gguf_refuses_bad_magic) {
    std::vector<uint8_t> bytes = read_whole_file(fixture_path());
    bytes[0] = 'X';  // "GGUF" -> "XGUF"
    TempFile tmp(bytes);
    CHECK(throws_containing(tmp.path(), "magic"));
}

TEST(gguf_refuses_truncated_file) {
    std::vector<uint8_t> bytes = read_whole_file(fixture_path());
    bytes.resize(40);  // cuts off inside the key-value metadata
    TempFile tmp(bytes);
    CHECK(throws_containing(tmp.path(), "truncated"));
}

TEST(gguf_refuses_tensor_past_eof) {
    std::vector<uint8_t> bytes = read_whole_file(fixture_path());
    bytes.resize(bytes.size() - 100);  // whole tensor table parses, data does not fit
    TempFile tmp(bytes);
    CHECK(throws_containing(tmp.path(), "past"));
}

TEST(gguf_refuses_unknown_tensor_type) {
    std::vector<uint8_t> bytes = read_whole_file(fixture_path());

    // "nextn.eh_proj.weight": 2 dims. Layout right after the length-prefixed
    // name: u32 n_dims, n_dims * u64 dims, u32 type, u64 offset.
    const size_t name_off = find_gguf_string(bytes, "nextn.eh_proj.weight");
    const size_t type_off = name_off + 8 /* namelen */ + 20 /* "nextn.eh_proj.weight" */ +
                             4 /* n_dims */ + 2 * 8 /* dims */;
    uint32_t existing = 0;
    std::memcpy(&existing, bytes.data() + type_off, 4);
    CHECK_EQ(existing, uint32_t{8});  // Q8_0 -- sanity that we found the right field
    patch_u32(bytes, type_off, 9999u);

    TempFile tmp(bytes);
    CHECK(throws_containing(tmp.path(), "9999"));
}

// ------------------------------------------------------------- real-file check
//
// ARCINT_GGUF_REAL=<path> gates a spot check against a real GGUF on disk
// (not shipped with the repo -- deliverable 5's "later window" is the actual
// gguf-py comparison; this only exercises the decoders on real bytes without
// crashing and prints something a human can sanity-check by eye).
TEST(gguf_real_file_spot_check) {
    const char* path = std::getenv("ARCINT_GGUF_REAL");
    SKIP_UNLESS(path != nullptr && *path != '\0', "ARCINT_GGUF_REAL not set");

    gguf::GgufFile f = gguf::GgufFile::open(path);
    std::fprintf(stderr, "  ARCINT_GGUF_REAL: %zu tensors, alignment %u\n", f.tensors().size(),
                 f.alignment());

    const gguf::GgmlType wanted[] = {gguf::GgmlType::Q4_K, gguf::GgmlType::Q5_K,
                                      gguf::GgmlType::Q6_K, gguf::GgmlType::Q8_0};
    for (gguf::GgmlType want : wanted) {
        const gguf::TensorInfo* found = nullptr;
        for (const gguf::TensorInfo& t : f.tensors()) {
            if (t.ggml_type == static_cast<int32_t>(want)) { found = &t; break; }
        }
        if (found == nullptr) continue;  // not every real file carries every type

        std::vector<float> out;
        gguf::dequantize_tensor(f, *found, out);
        bool     finite = true;
        double   sum = 0.0, max_abs = 0.0;
        uint64_t checksum = 0;
        for (float v : out) {
            if (!std::isfinite(v)) finite = false;
            sum += v;
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(v)));
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            checksum = checksum * 1000003u + bits;
        }
        CHECK(finite);
        CHECK(max_abs < 1000.0);  // sanity band for a trained weight, not a proof
        std::fprintf(stderr, "  %s (%s): %zu values, mean %.6f, max|.| %.6f, checksum %016llx\n",
                     found->name.c_str(), gguf::type_name(found->ggml_type).c_str(), out.size(),
                     sum / out.size(), max_abs, static_cast<unsigned long long>(checksum));
    }
}
