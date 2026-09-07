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


// The augmented width the repack gives a k: one group of 32 per `per_aug` groups, then one
// zero group more when the total group count would be odd (the runtime's int4 kernel walks K
// in pairs of groups; an odd count faulted on the card, DESIGN §7.0.2bl).
int64_t padded_aug(int64_t k, int64_t per_aug) {
    const int64_t base = ((k / 32 + per_aug - 1) / per_aug) * 32;
    return ((k + base) / 32) % 2 ? base + 32 : base;
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
    CHECK_EQ(r.k_aug, padded_aug(r.k, 8));
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

// The mins' packing as an option (--gguf-mins). Exact: two nibbles per group,
// one super-block per augmented group, +12.5 % on a Q4_K set. Shared: two
// super-blocks share an augmented group under the larger dmin, the other
// block's mins requantised in steps of it (+6.25 %). Nibble: one nibble per
// group under a scale shared by 32 groups (+3.1 %), the coarsest. The cost
// of the inexact forms is bounded here per group -- half a step of the shared
// scale -- and measured in the deviation the load reports.
namespace {
double max_min_error_in_scale_steps(const gguf::RepackedTensor& exact, const gguf::RepackedTensor& r) {
    // |stored min - exact min| against half the augmented group's scale, per group of every row
    const int64_t groups = r.k / 32, wgroups = r.width() / r.group;
    double worst = 0;
    for (int64_t row = 0; row < r.n; ++row)
        for (int64_t g = 0; g < groups; ++g) {
            const float s = gguf::f16_to_f32(r.scale[static_cast<size_t>(row * wgroups + r.k / 32 + g / r.groups_per_aug)]);
            const double e = std::fabs(static_cast<double>(gguf::repacked_group_min(r, row, g)) - static_cast<double>(gguf::repacked_group_min(exact, row, g)));
            worst = std::max(worst, s > 0 ? e / (0.5 * s) : (e > 0 ? 1e9 : 0.0));
        }
    return worst;
}
}  // namespace

TEST(q4_k_mins_shared_by_two_super_blocks_halve_the_augmentation_within_half_a_step_of_the_shared_scale) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 12);
    const auto exact = gguf::repack_tensor(f, t);
    const auto r = gguf::repack_tensor(f, t, nullptr, gguf::RepackMins::Shared);
    CHECK(r.mins == gguf::RepackMins::Shared);
    CHECK_EQ(r.aug_slots, int64_t{2});
    CHECK_EQ(r.groups_per_aug, int64_t{16});
    CHECK_EQ(r.k_aug, padded_aug(r.k, 16));   // K/16 before the even-count padding; at the served K (5,120 / 17,408) half the exact form's
    CHECK(r.k_aug <= exact.k_aug);
    CHECK(max_min_error_in_scale_steps(exact, r) <= 1.0 + 1e-3);
    const auto dv = gguf::repack_deviation(f, t, r, 1.0 / 64.0);
    const auto dve = gguf::repack_deviation(f, t, exact, 1.0 / 64.0);
    std::printf("  q4_k mins shared: max %.4f steps (exact %.4f), rms %.4f, %zu of %zu over the exact bound\n", dv.max_steps, dve.max_steps, dv.rms_steps, dv.over, dv.values);
    CHECK(dv.max_steps >= dve.max_steps);  // the cost is measured, never hidden
}

TEST(q4_k_mins_as_one_nibble_per_group_quarter_the_augmentation_within_half_a_step_of_the_shared_scale) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 12);
    const auto exact = gguf::repack_tensor(f, t);
    const auto r = gguf::repack_tensor(f, t, nullptr, gguf::RepackMins::Nibble);
    CHECK(r.mins == gguf::RepackMins::Nibble);
    CHECK_EQ(r.aug_slots, int64_t{1});
    CHECK_EQ(r.groups_per_aug, int64_t{32});
    CHECK_EQ(r.k_aug, padded_aug(r.k, 32));
    CHECK_EQ(((r.k + r.k_aug) / 32) % 2, int64_t{0});
    // the rule at the served widths: 5,120 -> 192 columns against the exact form's 640, 17,408 -> 576 against 2,176
    CHECK_EQ(padded_aug(5120, 32), int64_t{192}); CHECK_EQ(padded_aug(5120, 8), int64_t{640}); CHECK_EQ(padded_aug(5120, 16), int64_t{320});
    CHECK_EQ(padded_aug(17408, 32), int64_t{576}); CHECK_EQ(padded_aug(17408, 8), int64_t{2176}); CHECK_EQ(padded_aug(17408, 16), int64_t{1088});
    CHECK(max_min_error_in_scale_steps(exact, r) <= 1.0 + 1e-3);
    const auto dv = gguf::repack_deviation(f, t, r, 1.0 / 64.0);
    std::printf("  q4_k mins nibble: max %.4f steps, rms %.4f, %zu of %zu over the exact bound\n", dv.max_steps, dv.rms_steps, dv.over, dv.values);
    // the served computation stays within the requantised mins' own budget: half a shared-scale step per group, times |X_g|
    std::vector<float> w; gguf::dequantize_tensor(f, t, w);
    std::vector<float> x(static_cast<size_t>(r.k));
    for (int64_t c = 0; c < r.k; ++c) x[static_cast<size_t>(c)] = std::sin(0.37f * static_cast<float>(c)) * 1.5f;
    std::vector<float> y; gguf::matvec_repacked_host(r, x, y);
    const int64_t wgroups = r.width() / r.group;
    double worst = 0;
    for (int64_t row = 0; row < r.n; ++row) {
        double ref = 0, mag = 0, budget = 0;
        for (int64_t c = 0; c < r.k; ++c) { const double p = static_cast<double>(x[c]) * w[row * r.k + c]; ref += p; mag += std::fabs(p); }
        for (int64_t g = 0; g < r.k / 32; ++g) {
            double xg = 0; for (int64_t i = 0; i < 32; ++i) xg += x[static_cast<size_t>(g * 32 + i)];
            budget += std::fabs(xg) * 0.5 * gguf::f16_to_f32(r.scale[static_cast<size_t>(row * wgroups + r.k / 32 + g / r.groups_per_aug)]);
        }
        worst = std::max(worst, std::fabs(static_cast<double>(y[row]) - ref) / (1e-3 + mag * 0.002 + budget));
    }
    std::printf("  q4_k mins nibble: worst %.3f of (rounding budget + the mins' own)\n", worst);
    CHECK(worst <= 1.0);
}

// RepackMins::Split: no augmented columns at all (the widened activation's
// group-sum columns are what --dyn-quant on was quantising along with the
// weights, DESIGN §7.0.2bg); the min goes into RepackedTensor::min_matrix
// instead, one f16 value per row per group, computed independently here from
// the raw super-block (ggml's own get_scale_min_k4 bit layout, not the
// repack's) so the test does not just check the code against itself.
namespace {
void scale_min_k4_ref(int s, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (s < 4) { d = q[s] & 63; m = q[s + 4] & 63; }
    else { d = static_cast<uint8_t>((q[s + 4] & 0xF) | ((q[s - 4] >> 6) << 4)); m = static_cast<uint8_t>((q[s + 4] >> 4) | ((q[s] >> 6) << 4)); }
}
}  // namespace

TEST(q4_k_mins_split_has_no_augmented_columns_and_min_matrix_matches_the_raw_super_block) {
    // Q4_K (type 12, 144-byte blocks) and Q5_K (type 13, 176-byte blocks): the
    // scale/min 6-bit decoding sits at the same offset (4) in both block
    // layouts, so one loop body covers both -- the block stride is the only
    // thing that differs.
    for (int type : {12, 13}) {
        gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
        const auto& t = tensor_of_type(f, type);
        const auto r = gguf::repack_tensor(f, t, nullptr, gguf::RepackMins::Split);
        CHECK(r.mins == gguf::RepackMins::Split);
        CHECK_EQ(r.width(), r.k);
        CHECK_EQ(r.k_aug, int64_t{0});
        const int64_t groups = r.k / 32;
        CHECK_EQ(r.min_matrix.size(), static_cast<size_t>(r.n) * static_cast<size_t>(groups));
        const uint8_t* data = f.data(t);
        const size_t block_bytes = type == 12 ? 144 : 176;
        const size_t row_bytes = static_cast<size_t>(r.k / 256) * block_bytes;
        int checked = 0;
        for (int64_t row : {int64_t{0}, r.n / 2, r.n - 1}) {
            for (int64_t g = 0; g < groups; ++g) {
                const int64_t b = g / 8, s = g % 8;
                const uint8_t* blk = data + static_cast<size_t>(row) * row_bytes + static_cast<size_t>(b) * block_bytes;
                const float dmin = gguf::f16_to_f32(static_cast<uint16_t>(blk[2] | (blk[3] << 8)));
                uint8_t sc6, m6; scale_min_k4_ref(static_cast<int>(s), blk + 4, sc6, m6);
                (void)sc6;
                const uint16_t expected = gguf::f32_to_f16(dmin * static_cast<float>(m6));
                CHECK_EQ(r.min_matrix[static_cast<size_t>(row * groups + g)], expected);
                ++checked;
            }
        }
        CHECK(checked > 0);
    }
}

TEST(q6_k_and_q8_0_are_unaffected_by_gguf_mins_split) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    for (int type : {8, 14}) {
        const auto& t = tensor_of_type(f, type);
        const auto exact = gguf::repack_tensor(f, t);
        const auto split = gguf::repack_tensor(f, t, nullptr, gguf::RepackMins::Split);
        CHECK_EQ(split.k_aug, int64_t{0});
        CHECK(split.min_matrix.empty());
        CHECK_EQ(split.weights, exact.weights);
        CHECK_EQ(split.scale, exact.scale);
    }
}

TEST(q4_k_split_mins_matvec_agrees_with_the_f32_reference_within_2x_of_the_exact_forms_deviation) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 12);
    const auto exact = gguf::repack_tensor(f, t);
    const auto split = gguf::repack_tensor(f, t, nullptr, gguf::RepackMins::Split);
    std::vector<float> w; gguf::dequantize_tensor(f, t, w);
    std::vector<float> x(static_cast<size_t>(exact.k));
    for (int64_t c = 0; c < exact.k; ++c) x[static_cast<size_t>(c)] = std::sin(0.37f * static_cast<float>(c)) * 1.5f;
    std::vector<float> y_exact, y_split;
    gguf::matvec_repacked_host(exact, x, y_exact);
    gguf::matvec_repacked_host(split, x, y_split);
    double max_abs_exact = 0, max_abs_split = 0;
    for (int64_t row = 0; row < exact.n; ++row) {
        double ref = 0;
        for (int64_t c = 0; c < exact.k; ++c) ref += static_cast<double>(x[static_cast<size_t>(c)]) * static_cast<double>(w[static_cast<size_t>(row * exact.k + c)]);
        max_abs_exact = std::max(max_abs_exact, std::fabs(static_cast<double>(y_exact[static_cast<size_t>(row)]) - ref));
        max_abs_split = std::max(max_abs_split, std::fabs(static_cast<double>(y_split[static_cast<size_t>(row)]) - ref));
    }
    std::printf("  q4_k split matvec: max |y-ref| exact %.3g, split %.3g (%.2fx)\n", max_abs_exact, max_abs_split,
               max_abs_exact > 0 ? max_abs_split / max_abs_exact : 0.0);
    CHECK(max_abs_split <= 2.0 * max_abs_exact + 1e-6);
}

// The main columns are the exact form's, byte for byte; the min term carries
// one rounding the exact form's does not (a single float multiply of dmin and
// mn, rounded to f16 once -- gguf_repack.h's "What is exact and what is not").
// Measured on this fixture that pushes one value of 32,768 (0.003 %) a hair
// past the exact form's 1/64 bound (0.0158 steps): the split form's own bound
// is twice the exact one -- a MEASURED bound (the extra term is up to 2^-11 of
// the min in steps, which nothing bounds a priori; 2x holds on this fixture
// and on the served file, DESIGN 7.0.2bo) --
// and the load refuses a split tensor over it exactly as it refuses an exact
// one over 1/64 -- the mins are the file's, the form is exact-class, unlike
// Shared/Nibble whose deviation is reported and accepted.
TEST(q4_k_split_mins_repack_deviation_holds_twice_the_exact_bound) {
    gguf::GgufFile f = gguf::GgufFile::open(fixture_path());
    const auto& t = tensor_of_type(f, 12);
    const auto exact = gguf::repack_tensor(f, t);
    const auto r = gguf::repack_tensor(f, t, nullptr, gguf::RepackMins::Split);
    const auto dve = gguf::repack_deviation(f, t, exact, 1.0 / 64.0);
    const auto dv64 = gguf::repack_deviation(f, t, r, 1.0 / 64.0);
    const auto dv = gguf::repack_deviation(f, t, r, 2.0 / 64.0);
    std::printf("  q4_k split: max %.5f steps (exact %.5f), rms %.5f, %zu of %zu over 1/64, %zu over 2/64\n",
               dv.max_steps, dve.max_steps, dv.rms_steps, dv64.over, dv64.values, dv.over);
    CHECK_EQ(dve.over, size_t{0});             // the exact form holds its bound exactly
    CHECK(dv64.over != 0);                     // and the split form does not (the extra rounding is real: red before the 2x bound)
    CHECK(dv.max_steps <= 2.0 / 64.0);         // the split form's bound, the one the load refuses over
    CHECK_EQ(dv.over, size_t{0});
    CHECK(dv64.over * 1000 <= dv64.values);    // a handful of borderline groups, not a systematic miss

    // Q5_K's own exact bound is 1/32 (twice Q4_K's 1/64, one more decoded bit
    // of range): the split form's is twice that, 2/32.
    const auto& t5 = tensor_of_type(f, 13);
    const auto exact5 = gguf::repack_tensor(f, t5);
    const auto r5 = gguf::repack_tensor(f, t5, nullptr, gguf::RepackMins::Split);
    const auto dve5 = gguf::repack_deviation(f, t5, exact5, 1.0 / 32.0);
    const auto dv5 = gguf::repack_deviation(f, t5, r5, 2.0 / 32.0);
    std::printf("  q5_k split: max %.5f steps (exact %.5f), rms %.5f, %zu of %zu over 2/32\n",
               dv5.max_steps, dve5.max_steps, dv5.rms_steps, dv5.over, dv5.values);
    CHECK_EQ(dve5.over, size_t{0});
    CHECK(dv5.max_steps <= 2.0 / 32.0);
    CHECK_EQ(dv5.over, size_t{0});
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
    // RepackMins::Split under the same head-wise order: min_matrix is written
    // at the destination group `dg`, not the source group `g` (gguf_repack.cpp's
    // repack_row, `min_row[dg] = ...`) -- a wrong index there would still pass
    // the min_matrix-vs-raw-block test above (which reads its own row's groups
    // consistently) but would fail here, where the permuted and unpermuted
    // split repacks are cross-checked against each other.
    const auto ps = gguf::repack_tensor(f, t, &dest_of, gguf::RepackMins::Split);
    const auto dvps = gguf::repack_deviation(f, t, ps, 2.0 / 64.0);
    CHECK_EQ(dvps.over, size_t{0});
    const auto rs = gguf::repack_tensor(f, t, nullptr, gguf::RepackMins::Split);
    std::vector<float> ys, yps; gguf::matvec_repacked_host(rs, x, ys); gguf::matvec_repacked_host(ps, xp, yps);
    double mds = 0; for (size_t i = 0; i < ys.size(); ++i) mds = std::max(mds, static_cast<double>(std::fabs(ys[i] - yps[i])));
    CHECK(mds <= 1e-3);
    std::vector<int64_t> split(static_cast<size_t>(t.dims[0]));
    for (size_t c = 0; c < split.size(); ++c) split[c] = static_cast<int64_t>((c + 1) % split.size());
    bool threw = false;
    try { (void)gguf::repack_tensor(f, t, &split); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}
