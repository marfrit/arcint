// FIX D Link 3 (docs/design-qwen-flash-next.md "Links 2 and 3"): the wiring
// proof for exec/ngram_table.h -- NGramLookup ties row_ids (the hashed index)
// to gather_dequant (the table read), the PLETableBackend.lookup contract.
//
// The wiring under test is "gather the HASHED rows, not the token-index rows".
// Every case here computes the expected embedding by dequantizing the rows the
// hash selects (ngram::row_ids) directly from the table bytes, independent of
// NGramLookup, and asserts NGramLookup produced the same bytes. If Link 3 were
// wired to the old "one row per token id" shape (gather row == token id), the
// gathered bytes would differ and these cases fail -- that is the deletion
// test the roadmap requires. `lookup_gathers_the_hashed_rows_not_token_indices`
// makes the two shapes' divergence explicit so the proof is not vacuous.
#include "core/artifact.h"
#include "core/gguf.h"
#include "core/gguf_dequant.h"
#include "core/ngram_header.h"
#include "exec/ngram_row_ids.h"
#include "exec/ngram_table.h"
#include "harness.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace lgc;

namespace {

constexpr int32_t  kQ8_0     = 8;    // ggml Q8_0
constexpr uint16_t kF16One   = 0x3C00;  // 1.0 in f16
constexpr size_t   kCols     = 160;  // 5 x 32-element blocks
constexpr size_t   kBlockBytes = 34;  // Q8_0: 2 (d) + 32 (int8)

// Small, self-contained PLE geometry: 4 heads (2 x 2-gram + 2 x 3-gram).
struct Fixture {
    int64_t vocab_size            = 257;
    int     ngram_size            = 3;
    int     heads_per_ngram       = 2;
    int64_t ngram_vocab_size_base = 17;
    int     ple_layer_index       = 0;
    int64_t eos                   = 0;
};

ngram::HashParams params(const Fixture& fx) {
    auto p = ngram::derive_hash_constants(fx.vocab_size, fx.ngram_size, fx.heads_per_ngram,
                                          fx.ngram_vocab_size_base, fx.ple_layer_index);
    p.eos_token_id = fx.eos;
    return p;
}

int8_t cell(uint32_t row, size_t elem) {
    // Distinct per (row, elem); spans the int8 range so a wrong row is caught.
    return static_cast<int8_t>(static_cast<int>((row * 7 + elem) % 251) - 125);
}

// One Q8_0 row (170 bytes): 5 blocks of [f16 d=1.0][32 int8]. Dequantizing it
// yields exactly cell(row, e) for e in [0,160).
void write_row(std::vector<uint8_t>& out, uint32_t row) {
    for (size_t block = 0; block < kCols / 32; ++block) {
        out.push_back(static_cast<uint8_t>(kF16One & 0xFF));
        out.push_back(static_cast<uint8_t>(kF16One >> 8));
        for (size_t k = 0; k < 32; ++k)
            out.push_back(static_cast<uint8_t>(cell(row, block * 32 + k)));
    }
}

std::vector<uint8_t> build_payload(uint32_t n_rows) {
    std::vector<uint8_t> p;
    p.reserve(static_cast<size_t>(n_rows) * kBlockBytes * (kCols / 32));
    for (uint32_t r = 0; r < n_rows; ++r) write_row(p, r);
    return p;
}

uint32_t required_rows(const Fixture& fx) {
    return static_cast<uint32_t>(ngram::ngram_required_rows(
        fx.vocab_size, fx.ngram_size, fx.heads_per_ngram, fx.ngram_vocab_size_base,
        /*num_ple_layers=*/1));
}

// The expected lookup output: for each hashed row id, dequantize that row's
// bytes directly (not through NGramLookup). This is the independent oracle.
std::vector<float> expected_from_hash(const ngram::HashParams& p, const std::vector<uint8_t>& payload,
                                      const std::vector<int64_t>& context,
                                      const std::vector<int64_t>& tokens) {
    const auto ids = ngram::row_ids(p, context, tokens);
    std::vector<float> out(ids.size() * kCols);
    for (size_t i = 0; i < ids.size(); ++i) {
        const uint8_t* row = payload.data() + static_cast<size_t>(ids[i]) * kBlockBytes * (kCols / 32);
        gguf::dequantize_row(kQ8_0, row, kCols, out.data() + i * kCols);
    }
    return out;
}

std::string temp_path(const char* stem) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "/tmp/arcint-ngram-lookup-%s-%d.bin", stem, ::getpid());
    return std::string(buf);
}

void write_ngram_file(const std::string& path, uint32_t n_rows) {
    std::ofstream f(path, std::ios::binary);
    f.write("ARCINGRM", 8);
    const uint32_t type = static_cast<uint32_t>(kQ8_0);
    const uint32_t cols = static_cast<uint32_t>(kCols);
    const uint32_t reserved = 0;
    f.write(reinterpret_cast<const char*>(&type), 4);
    f.write(reinterpret_cast<const char*>(&cols), 4);
    f.write(reinterpret_cast<const char*>(&n_rows), 4);
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    const auto payload = build_payload(n_rows);
    f.write(reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
}

Artifact artifact_for(const Fixture& fx) {
    Artifact a;
    a.model_type                         = "qwen4_exp";
    a.ov_arch                            = "Qwen4ExpForConditionalGeneration";
    a.ngram_config.ngram_size            = fx.ngram_size;
    a.ngram_config.heads_per_ngram       = fx.heads_per_ngram;
    a.ngram_config.ngram_vocab_size_base = static_cast<int>(fx.ngram_vocab_size_base);
    a.ngram_config.ple_embed_dim         = (fx.ngram_size - 1) * fx.heads_per_ngram * static_cast<int>(kCols);
    a.ngram_config.vocab_size            = static_cast<int>(fx.vocab_size);
    a.ngram_config.ngram_boundary_token_id = static_cast<int>(fx.eos);
    a.ngram_config.ple_layer_ids         = {1};   // one PLE layer (ordinal 0 == fx.ple_layer_index)
    return a;
}

}  // namespace

// The in-memory lookup: adopt a table, gather, and match the independent
// per-row dequant. This proves row_ids feeds gather_dequant correctly.
TEST(ngram_lookup_gathers_and_dequantizes_the_hashed_rows) {
    Fixture fx;
    const uint32_t n_rows = required_rows(fx);
    auto           payload = build_payload(n_rows);
    const auto     p       = params(fx);

    auto table = ngram::NGramLookup::adopt_owned(payload, kQ8_0, kCols, n_rows, {p});

    const std::vector<int64_t> context = {7, 9};
    const std::vector<int64_t> tokens  = {21, 22, 23, 24, 25};
    const auto got      = table.lookup(0, context, tokens);
    const auto expected = expected_from_hash(p, payload, context, tokens);

    CHECK_EQ(got.size(), expected.size());
    CHECK(std::memcmp(got.data(), expected.data(), got.size() * sizeof(float)) == 0);
}

// The wiring is genuinely "hashed rows, not token-index rows": prove the hashed
// output differs from what the old "one row per token id" shape would gather,
// so `ngram_lookup_gathers_and_dequantizes_the_hashed_rows` is not vacuous.
TEST(ngram_lookup_gathers_the_hashed_rows_not_token_indices) {
    Fixture fx;
    const uint32_t n_rows  = required_rows(fx);
    auto           payload = build_payload(n_rows);
    const auto     p       = params(fx);
    auto table = ngram::NGramLookup::adopt_owned(payload, kQ8_0, kCols, n_rows, {p});

    const std::vector<int64_t> context = {7, 9};
    const std::vector<int64_t> tokens  = {21, 22, 23, 24, 25};
    const int                  H       = p.num_ngram_heads();
    const auto                 got     = table.lookup(0, context, tokens);

    // The naive "one row per token id" embedding: every head of token i reads
    // row (token_i mod n_rows).
    bool differs = false;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<float> naive_row(kCols);
        const uint32_t     naive_id = static_cast<uint32_t>(tokens[i] % n_rows);
        const uint8_t*     row = payload.data() + static_cast<size_t>(naive_id) * kBlockBytes * (kCols / 32);
        gguf::dequantize_row(kQ8_0, row, kCols, naive_row.data());
        for (int h = 0; h < H; ++h) {
            const float* head = got.data() + (i * static_cast<size_t>(H) + h) * kCols;
            if (std::memcmp(head, naive_row.data(), kCols * sizeof(float)) != 0) differs = true;
        }
    }
    CHECK(differs);
}

// The load seam behind --flash-next-ngram: admit + mmap + derive + lookup, on a
// real ARCINGRM file. The same independent oracle checks the gathered bytes.
TEST(ngram_load_ngram_lookup_end_to_end_from_a_file) {
    Fixture fx;
    const uint32_t n_rows = required_rows(fx);
    const Artifact a      = artifact_for(fx);
    const auto     path   = temp_path("e2e");
    write_ngram_file(path, n_rows);

    std::string err;
    uint64_t    payload_bytes = 0;
    auto        lk = ngram::load_ngram_lookup(a, path, /*host_ram*/ (1ull << 30), /*other*/ 0,
                                              /*margin*/ 0, err, payload_bytes);
    CHECK(err.empty());
    CHECK(lk.has_value());
    CHECK(payload_bytes > 0);

    // The load path derives its own params; rebuild the same to form the oracle
    // (derive is deterministic on the config).
    const auto p = params(fx);
    const std::vector<int64_t> context = {3, 4};
    const std::vector<int64_t> tokens  = {31, 5, 33, 34};
    const auto memory_payload = build_payload(n_rows);  // identical bytes to the file's payload
    const auto got      = lk->lookup(0, context, tokens);
    const auto expected = expected_from_hash(p, memory_payload, context, tokens);
    ::unlink(path.c_str());
    CHECK_EQ(got.size(), expected.size());
    CHECK(std::memcmp(got.data(), expected.data(), got.size() * sizeof(float)) == 0);
}

// Cold path: no flag => no lookup, no error (serving without the table).
TEST(ngram_load_ngram_lookup_cold_path_when_flag_absent) {
    const Artifact a = artifact_for(Fixture{});
    std::string    err = "sentinel";
    uint64_t       payload_bytes = 999;
    auto           lk = ngram::load_ngram_lookup(a, /*path*/ "", 0, 0, 0, err, payload_bytes);
    CHECK(!lk.has_value());
    CHECK(err.empty());
    CHECK_EQ(payload_bytes, uint64_t{0});
}

// A table too small for the hashed row space is refused by name (admission's
// lower bound), and load returns no lookup.
TEST(ngram_load_ngram_lookup_refuses_a_table_too_small) {
    Fixture fx;
    const Artifact a    = artifact_for(fx);
    const auto     path = temp_path("small");
    write_ngram_file(path, required_rows(fx) - 1);  // one row short
    std::string err;
    uint64_t    payload_bytes = 999;
    auto        lk = ngram::load_ngram_lookup(a, path, 0, 0, 0, err, payload_bytes);
    ::unlink(path.c_str());
    CHECK(!lk.has_value());
    CHECK(!err.empty());
    CHECK(err.find("below") != std::string::npos);
    CHECK_EQ(payload_bytes, uint64_t{0});
}
