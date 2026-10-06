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
    CHECK_EQ(f.tensors().size(), 9u);  // four quantized, two more for the pass test, three float
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

// ------------------------------------------------- the I-quant decoders (FIX F)
//
// IQ4_NL and IQ3_XXS are the formats the Flash-Next checkpoint ships its
// experts in (down / gate+up). The u4 grouped-affine repack of them costs
// 0.10-0.13 relative RMS per expert tensor and 0.73 nats at depth 48 (DESIGN
// 7.0.2bz; docs/design-routing-aware-expert-execution.md 2.3a), so the
// experts are computed from these blocks directly, and the host decoder is
// the first piece: the kernel's reference emulation and the host tier's
// decoder. Formulas transcribed from ggml-quants.c dequantize_row_iq4_nl /
// dequantize_row_iq3_xxs (pinned llama.cpp 56b9eb28).
//
// Red first: before the decoders existed, dequantize_row threw "not
// implemented" for both types and these cells failed on the throw.
TEST(gguf_iq4_nl_block_decodes_through_the_16_entry_table) {
    // one 32-value block: d = 1.0 (f16 0x3C00), nibble j in the low half of
    // byte j and 15-j in the high half, so y[j] = table[j], y[j+16] = table[15-j]
    static const int kTable[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                   1, 13, 25, 38, 53, 69, 89, 113};
    uint8_t block[18];
    block[0] = 0x00; block[1] = 0x3C;
    for (int j = 0; j < 16; ++j) block[2 + j] = static_cast<uint8_t>((j & 0xF) | ((15 - j) << 4));
    float y[32];
    gguf::dequantize_row(static_cast<int32_t>(gguf::GgmlType::IQ4_NL), block, 32, y);
    for (int j = 0; j < 16; ++j) {
        CHECK_EQ(y[j], static_cast<float>(kTable[j]));
        CHECK_EQ(y[j + 16], static_cast<float>(kTable[15 - j]));
    }
    // a second block with d = 0.5 (f16 0x3800) scales the same table by half
    uint8_t two[36];
    std::memcpy(two, block, 18);
    std::memcpy(two + 18, block, 18);
    two[18] = 0x00; two[19] = 0x38;
    float y2[64];
    gguf::dequantize_row(static_cast<int32_t>(gguf::GgmlType::IQ4_NL), two, 64, y2);
    CHECK_EQ(y2[32], -63.5f);
    CHECK_EQ(y2[63], static_cast<float>(kTable[0]) * 0.5f);
}

TEST(gguf_iq3_xxs_block_decodes_through_the_grid_and_the_sign_masks) {
    // one 256-value block (98 B): d = 1.0; the first 32-value sub-block's
    // four index pairs all point at grid entry 0 (0x04040404 -> 4,4,4,4)
    // except qs[0] = 1 (0x04040414 -> bytes 20,4,4,4); its scales-and-signs
    // word carries scale 3 (bits 28..31) and sign index 5 for l = 0 (bits
    // 0..6): ksigns[5] = 5 = 0b101 flips values 0 and 2 of the first eight.
    // db = d * (0.5 + 3) * 0.5 = 1.75, so the first eight values are
    // [-35, 7, -7, 7, 7, 7, 7, 7]; every other sub-block has scale 0 (db =
    // 0.25), index 0 and sign 0: all 1.0.
    uint8_t block[98];
    std::memset(block, 0, sizeof block);
    block[0] = 0x00; block[1] = 0x3C;
    uint8_t* qs = block + 2;
    qs[0] = 1;
    const uint32_t aux = (3u << 28) | 5u;
    std::memcpy(qs + 64, &aux, 4);
    float y[256];
    gguf::dequantize_row(static_cast<int32_t>(gguf::GgmlType::IQ3_XXS), block, 256, y);
    const float want[8] = {-35.f, 7.f, -7.f, 7.f, 7.f, 7.f, 7.f, 7.f};
    for (int j = 0; j < 8; ++j) CHECK_EQ(y[j], want[j]);
    for (int j = 8; j < 32; ++j) CHECK_EQ(y[j], 7.f);      // same sub-block scale, grid 0, no sign
    for (int j = 32; j < 256; ++j) CHECK_EQ(y[j], 1.f);    // scale 0 -> db 0.25, grid 0 -> 4
}

// IQ4_XS: layer 2's gate/up (the checkpoint mixes formats per layer; the
// census is in docs/design-routing-aware-expert-execution.md 2.3d). One
// 256-value block: d = 1.0; sub-block ib carries the 6-bit scale ls = ib + 30
// (low nibble in scales_l[ib/2], the two high bits in scales_h) so its scale
// is d * (ls - 32) = ib - 2; the nibbles as in the IQ4_NL cell. Red first:
// dequantize_row threw "not implemented" for IQ4_XS.
TEST(gguf_iq4_xs_block_decodes_with_its_six_bit_sub_block_scales) {
    uint8_t block[136] = {0};
    block[0] = 0x00; block[1] = 0x3C;                       // d = 1.0
    uint16_t scales_h = 0;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ib + 30;
        block[4 + ib / 2] |= static_cast<uint8_t>((ls & 0xF) << (4 * (ib % 2)));
        scales_h |= static_cast<uint16_t>(((ls >> 4) & 3) << (2 * ib));
    }
    block[2] = static_cast<uint8_t>(scales_h & 0xFF);
    block[3] = static_cast<uint8_t>(scales_h >> 8);
    for (int ib = 0; ib < 8; ++ib)
        for (int j = 0; j < 16; ++j)
            block[8 + ib * 16 + j] = static_cast<uint8_t>((j & 0xF) | ((15 - j) << 4));
    static const int8_t T[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    float y[256];
    gguf::dequantize_row(static_cast<int32_t>(gguf::GgmlType::IQ4_XS), block, 256, y);
    for (int ib = 0; ib < 8; ++ib) {
        const float dl = static_cast<float>(ib - 2);
        for (int j = 0; j < 16; ++j) {
            CHECK_EQ(y[ib * 32 + j], dl * static_cast<float>(T[j]));
            CHECK_EQ(y[ib * 32 + 16 + j], dl * static_cast<float>(T[15 - j]));
        }
    }
}

// The real shard: expert 0, row 0 of the two expert tensors, pinned to
// llama.cpp's own gguf-py dequantisation (`gguf.quants.dequantize`, read on
// the dev host 2026-09-18, the shard that carries blk.0's experts). The
// values are printed at 7 decimals there, the sums at 6.
TEST(gguf_iq_decoders_match_gguf_py_on_the_real_shard) {
    const char* path = std::getenv("ARCINT_GGUF_REAL");
    SKIP_UNLESS(path != nullptr && *path != '\0', "ARCINT_GGUF_REAL not set");
    gguf::GgufFile f = gguf::GgufFile::open(path);
    struct Pin { const char* name; gguf::GgmlType type; size_t row; double first8[8]; double sum; double abs_sum; };
    const Pin pins[] = {
        {"blk.0.ffn_down_exps.weight", gguf::GgmlType::IQ4_NL, 640,
         {-0.0030777, 0.0023675, -0.0267527, 0.0082862, 0.0023675, -0.0163357, -0.0089965, -0.0002367},
         -0.060845, 6.830881},
        {"blk.0.ffn_gate_exps.weight", gguf::GgmlType::IQ3_XXS, 2560,
         {-0.041832, 0.0134942, -0.0296872, 0.0134942, -0.0242895, -0.0188919, 0.0026988, 0.0080965},
         0.044130, 29.625160},
        // layer 2, the checkpoint's IQ4_XS gate over a Q8_0 down (2026-09-18)
        {"blk.2.ffn_gate_exps.weight", gguf::GgmlType::IQ4_XS, 2560,
         {-0.0013936, -0.0013936, -0.0030658, 0.0052955, 0.0001394, -0.0090581, 0.0018116, 0.0157472},
         0.092474, 26.058311},
        {"blk.2.ffn_down_exps.weight", gguf::GgmlType::Q8_0, 640,
         {-0.0019727, 0.0223570, -0.0032878, 0.0177541, -0.0065756, -0.0170965, -0.0003288, -0.0049317},
         0.155361, 6.580421},
    };
    int seen = 0;
    for (const Pin& pin : pins) {
        const gguf::TensorInfo* t = f.tensor(pin.name);
        if (t == nullptr) continue;                    // another shard
        ++seen;
        CHECK_EQ(t->ggml_type, static_cast<int32_t>(pin.type));
        std::vector<float> y(pin.row);
        gguf::dequantize_row(t->ggml_type, f.data(*t), pin.row, y.data());   // expert 0, row 0
        for (int j = 0; j < 8; ++j) CHECK_NEAR(static_cast<double>(y[j]), pin.first8[j], 2e-6);
        double sum = 0.0, abs_sum = 0.0;
        for (float v : y) { sum += v; abs_sum += std::fabs(v); }
        CHECK_NEAR(sum, pin.sum, 1e-5);
        CHECK_NEAR(abs_sum, pin.abs_sum, 1e-4);
        std::fprintf(stderr, "  %s: row 0 of expert 0 matches gguf-py (sum %.6f)\n", pin.name, sum);
    }
    SKIP_UNLESS(seen > 0, "this shard carries none of the pinned expert tensors");
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

// ---------------------------------------------------------------- core/gguf_map (0.4.0 stage 1)
#include "core/gguf_map.h"

// The converter's V-head reorder, inverted: with 16 key heads and 48 value
// heads (three per key head), HF head h = i*3 + j sits at the file's head
// j*16 + i. Red case: the identity would put HF head 1 at file head 1; the
// converter put it at 16.
TEST(gguf_v_head_map_inverts_the_converters_tiled_order) {
    const auto m = gguf_v_head_to_file_head(16, 48);
    CHECK_EQ(m.size(), static_cast<size_t>(48));
    CHECK_EQ(m[0], 0);
    CHECK_EQ(m[1], 16);
    CHECK_EQ(m[2], 32);
    CHECK_EQ(m[3], 1);
    CHECK_EQ(m[47], 47);
    // a bijection
    std::vector<int> seen(48, 0);
    for (auto v : m) { CHECK(v >= 0 && v < 48); seen[static_cast<size_t>(v)]++; }
    for (int s : seen) CHECK_EQ(s, 1);
    // equal head counts: no reorder
    const auto id = gguf_v_head_to_file_head(8, 8);
    for (size_t h = 0; h < id.size(); ++h) CHECK_EQ(id[h], static_cast<int64_t>(h));
    // counts that do not divide: no reorder either (the converter would not have)
    const auto odd = gguf_v_head_to_file_head(5, 12);
    for (size_t h = 0; h < odd.size(); ++h) CHECK_EQ(odd[h], static_cast<int64_t>(h));
}

TEST(gguf_module_map_names_every_projection_of_the_served_family) {
    const auto* qkv = gguf_module_map("linear_attn.in_proj_qkv");
    CHECK(qkv != nullptr);
    CHECK(qkv->gguf_tensor == "attn_qkv");
    CHECK(qkv->reorder == GgufReorder::RowsQKV);
    CHECK(gguf_module_map("linear_attn.out_proj")->reorder == GgufReorder::Columns);
    CHECK(gguf_module_map("mlp.down_proj")->reorder == GgufReorder::None);
    CHECK(gguf_module_map("self_attn.q_proj")->gguf_tensor == "attn_q");
    CHECK(gguf_module_map("self_attn.o_proj")->gguf_tensor == "attn_output");
    CHECK(gguf_module_map("linear_attn.in_proj_a")->gguf_tensor == "ssm_alpha");
    CHECK(gguf_module_map("something_else") == nullptr);
}

TEST(gguf_geometry_reads_the_fixtures_keys) {
    // The fixture carries the converter's key set at a toy geometry
    // (tools/gguf_fixture.py): two blocks of which one is the MTP block,
    // two GDN key heads of 64 against four value heads of 64.
    const auto f = gguf::GgufFile::open(fixture_path());
    const GgufGeometry g = gguf_geometry(f);
    CHECK_EQ(g.n_layers, 1);
    CHECK_EQ(g.hidden, 64);
    CHECK_EQ(g.n_heads, 4);
    CHECK_EQ(g.n_kv_heads, 2);
    CHECK_EQ(g.head_dim, 16);
    CHECK_EQ(g.linear_k_heads, 2);
    CHECK_EQ(g.linear_v_heads, 4);
    CHECK_EQ(g.linear_k_dim, 64);
    CHECK_EQ(g.linear_v_dim, 64);   // inner_size 256 / 4 value heads
    CHECK_EQ(g.full_attention_interval, 4);
    CHECK_EQ(g.vocab, 8);
    CHECK(gguf_geometry_mismatches(g, g).empty());
}

TEST(gguf_geometry_mismatches_name_every_differing_field_and_nothing_else) {
    GgufGeometry a; a.n_layers = 64; a.hidden = 5120; a.n_heads = 24; a.n_kv_heads = 4; a.head_dim = 256;
    a.linear_k_heads = 16; a.linear_v_heads = 48; a.linear_v_dim = 128; a.full_attention_interval = 4; a.vocab = 248320;
    GgufGeometry b = a;
    CHECK(gguf_geometry_mismatches(a, b).empty());
    b.vocab = 0;  // unknown on one side: not compared
    CHECK(gguf_geometry_mismatches(a, b).empty());
    b = a; b.n_layers = 65; b.linear_v_heads = 32;
    const auto m = gguf_geometry_mismatches(a, b);
    CHECK_EQ(m.size(), static_cast<size_t>(2));
    CHECK(m[0].find("layers") != std::string::npos);
    CHECK(m[1].find("GDN value heads") != std::string::npos);
}
