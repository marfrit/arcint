// 0.4.1 lever 2 (docs/design-gguf-native.md §3.6): the repack of K-quant
// tensors into the plugin's grouped compressed-weight form, checked on the
// host against ggml's dequantizer over the fixture. The gate is a bound on
// the deviation per weight in units of the group's quantisation step, the
// analytic worst case of the plugin's half arithmetic on each form (the
// stored integers and Q8_0's scales are the block's own; the rounding is the
// kernel's). A repack that exceeded the bound would be a wrong projection,
// not a slightly less precise one.
#include "core/gguf.h"
#include "core/gguf_dequant.h"
#include "core/gguf_repack.h"
#include "harness.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lgc;

namespace {

std::string fixture_path() { return std::string(ARCINT_SOURCE_DIR) + "/tests/fixtures/qwen35-tiny.gguf"; }

const gguf::TensorInfo& tensor_of_type(const gguf::GgufFile& f, int32_t type) {
    for (const auto& t : f.tensors())
        if (t.ggml_type == type && t.dims.size() == 2) return t;
    throw std::runtime_error("fixture has no 2-D tensor of type " + std::to_string(type));
}

}  // namespace

TEST(f32_to_f16_rounds_to_nearest_even_and_inverts_f16_to_f32) {
    // Every f16 value survives the round trip; a few halfway cases round to even.
    for (uint32_t h = 0; h < 0x10000; ++h) {
        const uint16_t bits = static_cast<uint16_t>(h);
        if ((bits & 0x7C00) == 0x7C00) continue;  // inf / nan
        CHECK_EQ(gguf::f32_to_f16(gguf::f16_to_f32(bits)), bits);
    }
    CHECK_EQ(gguf::f32_to_f16(1.0f), uint16_t{0x3C00});
    CHECK_EQ(gguf::f32_to_f16(1.0f + 1.0f / 2048.0f), uint16_t{0x3C00});   // halfway, rounds to even (down)
    CHECK_EQ(gguf::f32_to_f16(1.0f + 3.0f / 2048.0f), uint16_t{0x3C02});   // halfway, rounds to even (up)
    CHECK_EQ(gguf::f32_to_f16(65520.0f), uint16_t{0x7C00});                // just past max -> inf
    CHECK_EQ(gguf::f32_to_f16(-0.0f), uint16_t{0x8000});
}

// The bounds below are the analytic worst cases of the plugin's half
// arithmetic on the repacked form, in units of the group's quantisation
// step: every dequantized value is rounded to f16 (2^-11 relative, half an
// ulp of an 11-bit significand: at most |q - zp| * 2^-11 steps) -- the
// rounding every f16 weight in every served IR carries, and the one the
// native path's own tiled kernel applies to its f16 copies -- plus, for the
// K types, the f16 rounding of the group's scale (|q - zp| * 2^-11 again)
// and of the zero point. Q8_0's scale is the block's own f16, so only the
// value rounding remains: 127 * 2^-11, under 1/16 of a step.
TEST(q8_0_repacks_within_the_value_rounding_alone) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 8);
    const auto r = gguf::repack_tensor(f, t);
    CHECK(r.weights_type == gguf::RepackWeights::I8);
    CHECK(r.zp_type == gguf::RepackZeroPoint::None);
    CHECK_EQ(r.group, int64_t{32});
    // The stored bytes and scales are the block's own: exact before the kernel rounds.
    const auto* blk = f.data(t);
    CHECK_EQ(static_cast<int>(static_cast<int8_t>(r.weights[0])), static_cast<int>(static_cast<int8_t>(blk[2])));
    CHECK_EQ(r.scale[0], static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
    const auto dv = gguf::repack_deviation(f, t, r, 1.0 / 16.0);
    std::printf("  q8_0: max %.5f steps, rms %.5f steps, %zu of %zu over 1/16\n", dv.max_steps, dv.rms_steps, dv.over, dv.values);
    CHECK_EQ(dv.values, t.n_elements);
    CHECK(dv.max_steps <= 1.0 / 16.0);
    CHECK_EQ(dv.over, size_t{0});
}

TEST(q4_k_repacks_with_its_mins_as_augmented_columns_within_the_bound) {
    // q * scale in half: the value and the scale d*sc each round, at most
    // 15 * 2^-11 of a step each: 1/64. The mins are exact augmented columns.
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 12);
    const auto r = gguf::repack_tensor(f, t);
    CHECK(r.weights_type == gguf::RepackWeights::U4);
    CHECK(r.zp_type == gguf::RepackZeroPoint::None);
    CHECK_EQ(r.group, int64_t{32});
    CHECK_EQ(r.k_aug, r.k / 8);
    CHECK_EQ(r.groups_per_aug, int64_t{8});
    CHECK_EQ(r.weights.size(), static_cast<size_t>(r.n * r.width()) / 2);
    CHECK_EQ(r.scale.size(), static_cast<size_t>(r.n * r.width()) / 32);
    const auto dv = gguf::repack_deviation(f, t, r, 1.0 / 64.0);
    std::printf("  q4_k: max %.5f steps, rms %.5f steps, %zu of %zu over 1/64\n", dv.max_steps, dv.rms_steps, dv.over, dv.values);
    CHECK(dv.max_steps <= 1.0 / 64.0);
    CHECK_EQ(dv.over, size_t{0});
}

TEST(q5_k_repacks_with_its_mins_as_augmented_columns_within_the_bound) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 13);
    const auto r = gguf::repack_tensor(f, t);
    CHECK(r.weights_type == gguf::RepackWeights::U8);
    CHECK(r.zp_type == gguf::RepackZeroPoint::None);
    CHECK_EQ(r.k_aug, r.k / 8);
    const auto dv = gguf::repack_deviation(f, t, r, 1.0 / 32.0);
    std::printf("  q5_k: max %.5f steps, rms %.5f steps, %zu of %zu over 1/32\n", dv.max_steps, dv.rms_steps, dv.over, dv.values);
    CHECK(dv.max_steps <= 1.0 / 32.0);
    CHECK_EQ(dv.over, size_t{0});
}

TEST(q6_k_repacks_within_the_scale_rounding_bound) {
    // u8 with the zero point 32 (exact subtraction) and a scale per 16, ggml's
    // own grouping: d*sc and the value each round, at most 32 * 2^-11 of a
    // step each: 1/32.
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 14);
    const auto r = gguf::repack_tensor(f, t);
    CHECK(r.weights_type == gguf::RepackWeights::U8);
    CHECK(r.zp_type == gguf::RepackZeroPoint::U8Scalar);
    CHECK_EQ(static_cast<int>(r.zp_u8), 32);
    CHECK_EQ(r.group, int64_t{16});
    CHECK_EQ(r.scale.size(), t.n_elements / 16);
    const auto dv = gguf::repack_deviation(f, t, r, 1.0 / 32.0);
    std::printf("  q6_k: max %.5f steps, rms %.5f steps, %zu of %zu over 1/32\n", dv.max_steps, dv.rms_steps, dv.over, dv.values);
    CHECK(dv.max_steps <= 1.0 / 32.0 + 1e-9);
    CHECK_EQ(dv.over, size_t{0});
}

TEST(repack_refuses_a_type_it_does_not_serve) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const gguf::TensorInfo* f32 = nullptr;
    for (const auto& t : f.tensors()) if (t.ggml_type == 0) { f32 = &t; break; }
    CHECK(f32 != nullptr);
    CHECK(!gguf::repack_supported(0));
    bool threw = false;
    try { gguf::repack_tensor(f, *f32); } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("not a repacked type") != std::string::npos; if (!threw) std::printf("  threw: %s\n", e.what()); }
    CHECK(threw);
}

TEST(permute_rows_moves_whole_rows_of_the_repacked_form) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 12);
    const auto r = gguf::repack_tensor(f, t);
    std::vector<float> before; gguf::dequantize_repacked(r, before);
    // Reverse the rows.
    std::vector<int64_t> src_of(static_cast<size_t>(r.n));
    for (int64_t i = 0; i < r.n; ++i) src_of[static_cast<size_t>(i)] = r.n - 1 - i;
    auto p = r; gguf::permute_rows(p, src_of);
    std::vector<float> after; gguf::dequantize_repacked(p, after);
    bool same = true;
    for (int64_t dest = 0; dest < r.n && same; ++dest)
        for (int64_t c = 0; c < r.k; ++c)
            if (after[dest * r.k + c] != before[src_of[static_cast<size_t>(dest)] * r.k + c]) { same = false; break; }
    CHECK(same);
}

TEST(the_served_computation_matches_the_f32_reference_for_every_repacked_type) {
    // The widened activation through the augmentation matrix, the plain
    // multiply-accumulate over the repacked row: within the f16 roundings of the
    // f32 dot product with ggml's dequantized weights. This is the projection's
    // equivalence, end to end on the host.
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    for (int type : {8, 12, 13, 14}) {
        const auto& t = tensor_of_type(f, type);
        const auto r = gguf::repack_tensor(f, t);
        std::vector<float> w; gguf::dequantize_tensor(f, t, w);
        std::vector<float> x(static_cast<size_t>(r.k));
        for (int64_t c = 0; c < r.k; ++c) x[static_cast<size_t>(c)] = std::sin(0.37f * static_cast<float>(c) + static_cast<float>(type)) * 1.5f;
        std::vector<float> y; gguf::matvec_repacked_host(r, x, y);
        double max_rel = 0, max_abs = 0, max_ref = 0;
        for (int64_t row = 0; row < r.n; ++row) {
            double ref = 0, mag = 0;
            for (int64_t c = 0; c < r.k; ++c) { const double p = static_cast<double>(x[c]) * w[row * r.k + c]; ref += p; mag += std::fabs(p); }
            const double e = std::fabs(static_cast<double>(y[row]) - ref);
            max_abs = std::max(max_abs, e); max_ref = std::max(max_ref, std::fabs(ref));
            max_rel = std::max(max_rel, e / (1e-3 + mag * 0.002));  // 2^-9 of the magnitude sum: f16 roundings of x, the values and the sums
        }
        std::printf("  type %d: max |y - ref| %.3g (|ref| up to %.3g), worst %.3f of the rounding budget\n", type, max_abs, max_ref, max_rel);
        CHECK(max_rel <= 1.0);
    }
}

TEST(a_head_wise_column_order_is_applied_at_build_and_refused_when_it_splits_a_group) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 12);
    // Swap the two halves of every row in units of 128 columns (the fixture's K is 256).
    std::vector<int64_t> dest_of(static_cast<size_t>(t.dims[0]));
    for (size_t c = 0; c < dest_of.size(); ++c) dest_of[c] = static_cast<int64_t>((c + 128) % dest_of.size());
    const auto p = gguf::repack_tensor(f, t, &dest_of);
    CHECK_EQ(p.groups_per_aug, int64_t{4});
    const auto dvp = gguf::repack_deviation(f, t, p, 1.0 / 64.0);   // the check follows the order
    CHECK_EQ(dvp.over, size_t{0});
    CHECK(dvp.max_steps <= 1.0 / 64.0);
    const auto r = gguf::repack_tensor(f, t);
    std::vector<float> before; gguf::dequantize_repacked(r, before);
    std::vector<float> after; gguf::dequantize_repacked(p, after);
    bool same = true;
    for (int64_t row = 0; row < r.n && same; ++row)
        for (int64_t c = 0; c < r.k; ++c)
            if (after[row * r.k + dest_of[static_cast<size_t>(c)]] != before[row * r.k + c]) { same = false; break; }
    CHECK(same);
    // The served computation on a permuted input equals the unpermuted one.
    std::vector<float> x(static_cast<size_t>(r.k)), xp(x.size());
    for (size_t c = 0; c < x.size(); ++c) x[c] = std::cos(0.11f * static_cast<float>(c)) * 2.0f;
    for (size_t c = 0; c < x.size(); ++c) xp[static_cast<size_t>(dest_of[c])] = x[c];
    std::vector<float> y, yp; gguf::matvec_repacked_host(r, x, y); gguf::matvec_repacked_host(p, xp, yp);
    double md = 0; for (size_t i = 0; i < y.size(); ++i) md = std::max(md, static_cast<double>(std::fabs(y[i] - yp[i])));
    CHECK(md <= 1e-3);
    std::vector<int64_t> split(static_cast<size_t>(t.dims[0]));
    for (size_t c = 0; c < split.size(); ++c) split[c] = static_cast<int64_t>((c + 1) % split.size());
    bool threw = false;
    try { (void)gguf::repack_tensor(f, t, &split); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}
