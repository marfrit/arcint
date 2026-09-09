// FIX D (0.5.0, docs/design-qwen-flash-next.md "FIX D — N-gram table:
// host-offload + dequantise-on-gather"): red-first coverage for
// exec/ngram_gather.h's gather-with-dequant kernel and exec/fit.h's
// host-RAM budget arithmetic for the n-gram embedding table
// (`per_layer_token_embd`).
//
// Fixture: a synthetic 1000-row x 160-column table (the checkpoint's own
// row width; not a whole tensor -- no Flash-Next artifact exists on this
// host to load, see HANDOFF-0.5.0.local.md), quantized here into all three
// 32-element-block formats the shipped/candidate precisions use
// (Q4_0/Q4_1/Q8_0) from known float values via test-local quantizers
// transcribed from ggml-quants.c's own quantize_row_q4_0/q4_1/q8_0 (the
// inverse of core/gguf_dequant.cpp's dequantize_row_q4_0/q4_1 and the
// existing dequantize_row_q8_0). This is deliberately NOT a constant-per-
// dimension fill (memory: feedback-gpu-test-zombies / the BY_TOKEN NaN
// record in HANDOFF-0.4.7 -- a constant fill makes min==max and hides
// exactly the kind of scale/zero-point bug a gather kernel could have).
#include "core/gguf.h"
#include "core/gguf_dequant.h"
#include "exec/fit.h"
#include "exec/ngram_gather.h"
#include "harness.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace lgc;

namespace {

constexpr size_t kRows = 1000;
constexpr size_t kCols = 160;  // the checkpoint's own n-gram row width
constexpr size_t kBlockElems = 32;
constexpr size_t kBlocksPerRow = kCols / kBlockElems;  // 5, exact -- no trailing partial block

// Minimal, round-to-nearest-even f32 -> f16 (test-fixture-only; the
// production direction, f16 -> f32, is core/gguf_dequant.cpp's own
// f16_to_f32, already reviewed and tested in tests/test_gguf.cpp -- this is
// its inverse, needed only to build a byte-valid fixture from float input,
// not exercised as production code anywhere).
uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t        exp  = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t       mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 1 : 0));
    if (exp <= 0) return static_cast<uint16_t>(sign);  // flush-to-zero: fixture values avoid this
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow to inf

    uint32_t mant16    = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (mant16 & 1u))) {
        ++mant16;
        if (mant16 == 0x400u) {
            mant16 = 0;
            ++exp;
            if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant16);
}

void write_u16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }

// ggml-quants.c's quantize_row_q4_0, transcribed (32-element block, one f16
// scale, 16 bytes of packed nibbles biased by +8).
void quantize_block_q4_0(const float* x, uint8_t* block /* 18 bytes */) {
    float amax = 0.0f, max = 0.0f;
    for (size_t j = 0; j < kBlockElems; ++j) {
        if (amax < std::fabs(x[j])) {
            amax = std::fabs(x[j]);
            max  = x[j];
        }
    }
    const float d  = max / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    write_u16(block, f32_to_f16(d));
    uint8_t* qs = block + 2;
    for (size_t j = 0; j < 16; ++j) {
        const float   x0  = x[j] * id;
        const float   x1  = x[j + 16] * id;
        const int32_t xi0 = std::min(15, static_cast<int32_t>(x0 + 8.5f));
        const int32_t xi1 = std::min(15, static_cast<int32_t>(x1 + 8.5f));
        qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
    }
}

// ggml-quants.c's quantize_row_q4_1: min/max range, one f16 scale + one f16
// min, 16 bytes of packed nibbles (no +8 bias -- Q4_1 is min-offset).
void quantize_block_q4_1(const float* x, uint8_t* block /* 20 bytes */) {
    float vmin = x[0], vmax = x[0];
    for (size_t j = 1; j < kBlockElems; ++j) {
        vmin = std::min(vmin, x[j]);
        vmax = std::max(vmax, x[j]);
    }
    const float d  = (vmax - vmin) / 15.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    write_u16(block, f32_to_f16(d));
    write_u16(block + 2, f32_to_f16(vmin));
    uint8_t* qs = block + 4;
    for (size_t j = 0; j < 16; ++j) {
        const float   x0  = (x[j] - vmin) * id;
        const float   x1  = (x[j + 16] - vmin) * id;
        const int32_t xi0 = std::min(15, static_cast<int32_t>(x0 + 0.5f));
        const int32_t xi1 = std::min(15, static_cast<int32_t>(x1 + 0.5f));
        qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
    }
}

// ggml-quants.c's quantize_row_q8_0: one f16 scale, 32 signed int8.
void quantize_block_q8_0(const float* x, uint8_t* block /* 34 bytes */) {
    float amax = 0.0f;
    for (size_t j = 0; j < kBlockElems; ++j) amax = std::max(amax, std::fabs(x[j]));
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    write_u16(block, f32_to_f16(d));
    auto* qs = reinterpret_cast<int8_t*>(block + 2);
    for (size_t j = 0; j < kBlockElems; ++j) {
        qs[j] = static_cast<int8_t>(std::lround(x[j] * id));
    }
}

// A smooth, per-dimension-and-per-row-varying source table -- deliberately
// not constant along either axis (see the file header comment). Range kept
// inside [-6, 6] so Q4_0's amax/-8 scale stays well away from f16 overflow
// or degenerate all-zero blocks.
std::vector<float> synth_table(size_t rows, size_t cols) {
    std::vector<float> t(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            t[r * cols + c] =
                6.0f * std::sin(0.013f * static_cast<float>(r) + 0.29f * static_cast<float>(c)) *
                (0.2f + 0.8f * static_cast<float>((r * 7 + c * 3) % 11) / 10.0f);
        }
    }
    return t;
}

std::vector<uint8_t> quantize_table(const std::vector<float>& src, size_t rows, size_t cols,
                                     size_t block_bytes,
                                     void (*quantize_block)(const float*, uint8_t*)) {
    const size_t         row_bytes = block_bytes * kBlocksPerRow;
    std::vector<uint8_t> out(rows * row_bytes);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t b = 0; b < kBlocksPerRow; ++b) {
            quantize_block(src.data() + r * cols + b * kBlockElems,
                            out.data() + r * row_bytes + b * block_bytes);
        }
    }
    return out;
}

std::vector<uint32_t> sample_indices(size_t n_rows, size_t how_many, uint32_t seed) {
    std::vector<uint32_t> idx = {0, static_cast<uint32_t>(n_rows - 1)};  // always cover the edges
    std::mt19937           rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(n_rows - 1));
    while (idx.size() < how_many) idx.push_back(dist(rng));
    return idx;
}

// Shared body for all three formats: quantize the synthetic table, gather a
// sample of rows through gather_dequant (the AVX2-or-scalar dispatcher) and
// through gather_dequant_scalar (forced scalar) independently, and assert
// byte-exact agreement -- plus a third independent check, gathering one row
// at a time via lgc::gguf::dequantize_row directly (the same reference
// dequant/gguf_dequant.cpp uses and test_gguf.cpp already checks against an
// external fixture), to catch a bug shared between gather_dequant_scalar and
// gather_dequant that a two-way comparison alone would miss.
void check_format(int32_t ggml_type, size_t block_bytes,
                   void (*quantize_block)(const float*, uint8_t*)) {
    const std::vector<float>   src   = synth_table(kRows, kCols);
    const std::vector<uint8_t> table = quantize_table(src, kRows, kCols, block_bytes, quantize_block);
    const size_t                row_bytes = block_bytes * kBlocksPerRow;

    // type_info's own block layout must agree with the fixture's -- if it
    // doesn't, every other check below is comparing against a
    // mis-addressed row and would pass or fail for the wrong reason.
    const auto info = gguf::type_info(ggml_type);
    CHECK_EQ(info.block_size, kBlockElems);
    CHECK_EQ(info.type_size, block_bytes);

    const std::vector<uint32_t> indices = sample_indices(kRows, 37, /*seed=*/12345u + ggml_type);

    std::vector<float> out_dispatch(indices.size() * kCols, 0.0f);
    std::vector<float> out_scalar(indices.size() * kCols, 0.0f);
    ngram::gather_dequant(ggml_type, table.data(), row_bytes, kCols, indices, out_dispatch.data(),
                          /*force_scalar=*/false);
    ngram::gather_dequant(ggml_type, table.data(), row_bytes, kCols, indices, out_scalar.data(),
                          /*force_scalar=*/true);

    // Byte-exact, not "close": memcmp, not CHECK_NEAR. This is the red case
    // -- an AVX2 kernel that rounds differently (an FMA contraction, a
    // wrong nibble/byte mapping, an off-by-one block stride) fails this
    // line even when the numbers "look right" to a tolerance-based check.
    CHECK(std::memcmp(out_dispatch.data(), out_scalar.data(), out_dispatch.size() * sizeof(float)) == 0);

    // Third, independent check: gather_dequant_scalar's own per-row output
    // against calling lgc::gguf::dequantize_row directly on the same row
    // bytes -- catches a bug shared between gather_dequant_scalar and
    // dequant_row_scalar (they currently both just forward to
    // dequantize_row, so this also pins that indirection stays a pure
    // forward, not a place a future refactor quietly diverges).
    for (size_t i = 0; i < indices.size(); ++i) {
        std::vector<float> direct(kCols);
        gguf::dequantize_row(ggml_type, table.data() + static_cast<size_t>(indices[i]) * row_bytes,
                             kCols, direct.data());
        CHECK(std::memcmp(direct.data(), out_scalar.data() + i * kCols, kCols * sizeof(float)) == 0);
    }

    // Sanity against the KNOWN float input: dequantized values must track
    // the source within one quantization step. Bounds are format-specific
    // (roughly amax/8 for Q4_0/Q4_1's 4-bit code, amax/127 for Q8_0's
    // 8-bit code); 1.0f is a loose bound that only catches a gross error
    // (wrong scale, wrong sign, a dropped row) -- the byte-exact checks
    // above are what actually pins the kernel, this is a second, cheap
    // "not obviously nonsense" gate on the reference dequant itself.
    for (size_t i = 0; i < indices.size(); ++i) {
        const size_t row = indices[i];
        for (size_t c = 0; c < kCols; ++c) {
            const float expected = src[row * kCols + c];
            const float got      = out_scalar[i * kCols + c];
            CHECK_NEAR(static_cast<double>(got), static_cast<double>(expected), 1.0);
        }
    }
}

}  // namespace

TEST(ngram_gather_q4_0_avx2_matches_scalar_reference_byte_exact) {
    SKIP_UNLESS(ngram::cpu_has_avx2(), "AVX2 not available on this host");
    check_format(static_cast<int32_t>(gguf::GgmlType::Q4_0), 18, quantize_block_q4_0);
}

TEST(ngram_gather_q4_1_avx2_matches_scalar_reference_byte_exact) {
    SKIP_UNLESS(ngram::cpu_has_avx2(), "AVX2 not available on this host");
    check_format(static_cast<int32_t>(gguf::GgmlType::Q4_1), 20, quantize_block_q4_1);
}

TEST(ngram_gather_q8_0_avx2_matches_scalar_reference_byte_exact) {
    SKIP_UNLESS(ngram::cpu_has_avx2(), "AVX2 not available on this host");
    check_format(static_cast<int32_t>(gguf::GgmlType::Q8_0), 34, quantize_block_q8_0);
}

// The scalar path alone, on any host regardless of AVX2 -- the same
// fixture and the same byte-exact-against-dequantize_row check, without the
// AVX2-vs-scalar comparison. Ensures the harness is not entirely gated
// behind SKIP_UNLESS on a host without AVX2 (this repository's ctest
// invocation runs with --max-skips 0: every case must either pass or be on
// the named allow-skip list, so a suite that only ever produces skips here
// would not actually be exercising anything on such a host).
TEST(ngram_gather_scalar_path_matches_dequantize_row_on_any_host) {
    const std::vector<float>   src   = synth_table(kRows, kCols);
    const std::vector<uint8_t> table = quantize_table(src, kRows, kCols, 18, quantize_block_q4_0);
    const size_t                row_bytes = 18 * kBlocksPerRow;
    const std::vector<uint32_t> indices   = sample_indices(kRows, 11, /*seed=*/777);

    std::vector<float> out(indices.size() * kCols, 0.0f);
    ngram::gather_dequant_scalar(static_cast<int32_t>(gguf::GgmlType::Q4_0), table.data(), row_bytes,
                                 kCols, indices, out.data());
    for (size_t i = 0; i < indices.size(); ++i) {
        std::vector<float> direct(kCols);
        gguf::dequantize_row(static_cast<int32_t>(gguf::GgmlType::Q4_0),
                             table.data() + static_cast<size_t>(indices[i]) * row_bytes, kCols,
                             direct.data());
        CHECK(std::memcmp(direct.data(), out.data() + i * kCols, kCols * sizeof(float)) == 0);
    }
}

// ------------------------------------------------------- host RAM budget

// FIX D item 2: the budget arithmetic, and its refusal red case -- a
// budget-exceeding configuration must be refused BY NAME
// (host_ram_fit_must_refuse), the machine must not silently swap.
namespace {
constexpr uint64_t kGiB = 1ull << 30;
}  // namespace

TEST(ngram_table_bytes_matches_the_recon_quoted_figures) {
    const uint64_t n = lgc::kFlashNextNgramElements;
    // Recon (HANDOFF-0.5.0.local.md): "q8_0 -> 51.9 GiB, q4_1 -> 30.5 GiB,
    // the shipped GGUF carries Q4_0 (~29 GB)" -- checked here to
    // one-hundredth of a GiB so a future change to type_info's byte layout
    // or to kFlashNextNgramElements's own derivation cannot silently drift
    // this file's own comment away from what it claims to reproduce.
    const uint64_t q4_0 = lgc::ngram_table_bytes(static_cast<int32_t>(gguf::GgmlType::Q4_0), n);
    const uint64_t q4_1 = lgc::ngram_table_bytes(static_cast<int32_t>(gguf::GgmlType::Q4_1), n);
    const uint64_t q8_0 = lgc::ngram_table_bytes(static_cast<int32_t>(gguf::GgmlType::Q8_0), n);
    CHECK_NEAR(static_cast<double>(q4_0) / kGiB, 27.48, 0.01);
    CHECK_NEAR(static_cast<double>(q4_1) / kGiB, 30.53, 0.01);
    CHECK_NEAR(static_cast<double>(q8_0) / kGiB, 51.90, 0.01);
}

TEST(ngram_table_bytes_unknown_type_prices_as_nothing) {
    CHECK_EQ(lgc::ngram_table_bytes(/*ggml_type=*/9999, 51'200'000'000ull), 0ull);
}

// RED CASE: the dev container (48 GiB RAM, per CLAUDE.local.md) cannot hold
// the Q8_0 n-gram table (~51.9 GiB) at all -- must refuse regardless of
// what else is resident.
TEST(host_ram_fit_refuses_q8_0_ngram_table_on_the_48gib_dev_container) {
    const uint64_t ngram_bytes =
        lgc::ngram_table_bytes(static_cast<int32_t>(gguf::GgmlType::Q8_0), lgc::kFlashNextNgramElements);
    const bool refuse = lgc::host_ram_fit_must_refuse(ngram_bytes, /*expert_pool_bytes=*/0,
                                                       /*other_resident_bytes=*/0,
                                                       /*host_ram_bytes=*/48ull * kGiB,
                                                       /*margin_bytes=*/0);
    CHECK(refuse);
}

// RED CASE: Q4_1 (~30.5 GiB) alone leaves ~17.5 GiB on the 48 GiB
// container -- HANDOFF's own framing is that this is "the same memory" the
// expert pool (FIX E) wants; an expert pool asking for more than that
// remaining headroom must be refused, not silently swapped.
TEST(host_ram_fit_refuses_q4_1_ngram_table_plus_an_oversized_expert_pool) {
    const uint64_t ngram_bytes =
        lgc::ngram_table_bytes(static_cast<int32_t>(gguf::GgmlType::Q4_1), lgc::kFlashNextNgramElements);
    const uint64_t host_ram = 48ull * kGiB;

    const lgc::HostRamFit ok = lgc::host_ram_fit(ngram_bytes, /*expert_pool_bytes=*/10ull * kGiB,
                                                 /*other_resident_bytes=*/2ull * kGiB, host_ram,
                                                 /*margin_bytes=*/1ull * kGiB);
    CHECK(!ok.refuse);
    CHECK(ok.headroom_bytes >= 0);

    const lgc::HostRamFit over = lgc::host_ram_fit(ngram_bytes, /*expert_pool_bytes=*/20ull * kGiB,
                                                   /*other_resident_bytes=*/2ull * kGiB, host_ram,
                                                   /*margin_bytes=*/1ull * kGiB);
    CHECK(over.refuse);
    CHECK(over.headroom_bytes < 0);
    CHECK_EQ(over.refuse, lgc::host_ram_fit_must_refuse(ngram_bytes, 20ull * kGiB, 2ull * kGiB,
                                                        host_ram, 1ull * kGiB));
}

// Non-refusal case, named for symmetry: a unit host with real headroom
// (128 GiB, well above the dev container) admits Q4_1 plus a generous
// expert pool -- the check must not refuse everything unconditionally.
TEST(host_ram_fit_admits_q4_1_ngram_table_on_a_128gib_unit_host) {
    const uint64_t ngram_bytes =
        lgc::ngram_table_bytes(static_cast<int32_t>(gguf::GgmlType::Q4_1), lgc::kFlashNextNgramElements);
    const lgc::HostRamFit r = lgc::host_ram_fit(ngram_bytes, /*expert_pool_bytes=*/60ull * kGiB,
                                                /*other_resident_bytes=*/4ull * kGiB,
                                                /*host_ram_bytes=*/128ull * kGiB,
                                                /*margin_bytes=*/4ull * kGiB);
    CHECK(!r.refuse);
    CHECK(r.headroom_bytes > 0);
}
