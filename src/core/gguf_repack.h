#pragma once

// Repacking a GGUF K-quant tensor into the plugin's own compressed-weight
// form at load (docs/design-gguf-native.md §3.6, 0.4.1 lever 2): grouped
// integer weights with an f16 scale per group and NO zero point -- the form
// the runtime's fastest kernels (oneDNN's int4 GEMM) take. Host-side,
// device-free, so the equivalence of the projection is measured here,
// against ggml's own dequantizer, before any of it reaches a card.
//
// The mins. Q4_K and Q5_K values are dl*q - ml with a per-group min ml =
// dmin*mn (dmin an f16 per super-block, mn a 6-bit integer). A zero point
// ml/dl is not an integer, and the runtime's fast path takes integer zero
// points only (an f16 one falls to kernels that prefill at a tenth of the
// rate and charge activations per token, measured). So the min term is
// carried EXACTLY as extra columns of the same tensor: for every group, the
// integer mn (as two u4 nibbles hi/lo for a u4 tensor, one u8 for a u8 one)
// under the super-block's own f16 dmin as the group scale; and the activation
// is widened by the matching group sums (-16 sum(x_g), -sum(x_g)), so that
//   sum_k x_k (dl q_k - ml) = sum_k x_k dl q_k + (-16 X_g) dmin hi + (-X_g) dmin lo
// falls out of one plain fully-connected. The widening is one reduce, one
// tiny matmul and one concat per distinct activation (augmentation_matrix).
//
// What is exact and what is not. The stored integers and Q8_0's scales are
// the block's own, the mins are exact. A K-quant group scale d*sc (an f16
// times a 6-bit integer, 17 bits) rounds to f16, and the plugin's kernels
// compute q * scale in half, which rounds every value at 2^-11 relative; the
// min term's group sums are f16 too. The deviation per weight (main columns)
// is bounded here in units of the group's quantisation step (under 1/64 for
// Q4_K, 1/32 for Q5_K and Q6_K, 1/16 for Q8_0) and checked by the tests over
// the fixture; the native path's own tiled kernel already multiplies f16
// copies of the decoded values (DESIGN §7.0.2ay), its Xe2 decode is f32-exact.
//
// RepackMins::Split's rounding sequence, on top of the above. The main
// columns are the exact form's, byte for byte -- same weights, same scale.
// The min itself is one float multiply of the file's f16 dmin (already
// exact, decoded once) and the integer mn, rounded to f16 once (min_matrix):
// one rounding the augmented-column form's mins do not carry (there, the
// stored integer is exact under the group's own f16 scale, with no second
// product to round). That rounding adds up to 2^-11 of the min, in steps,
// to the exact form's deviation -- a term nothing bounds a priori (the min
// can be many steps where the group's scale is small), so the split form's
// bound is MEASURED, not derived: twice the exact bound, which holds on the
// fixture's Q4_K tensor (0.0158 steps against the exact form's 0.0133 and
// 1/64) and on the served file; the load refuses a split tensor over it, the
// same way it refuses an exact one over its bound
// (tests/test_gguf_repack.cpp). Served as y = MatMul(act, W_main) +
// MatMul(sums, -mins) (gguf_graph.cpp). matvec_repacked_host emulates it as
// f16(f16(sum of f16(f16(q)*scale)*x)) + f16(sum of f16(group sum of x) *
// -min)) -- two independently-rounded partial dot products added once more
// in f16, not the exact form's single running accumulator. That is an
// ASSUMPTION about the served graph: the plugin may fuse the Add into either
// fully-connected as a post-op on its f32 accumulator (one rounding fewer);
// which it does is not measured.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/gguf.h"

namespace lgc::gguf {

enum class RepackWeights { U4, U8, I8 };
enum class RepackZeroPoint { None, U8Scalar };
// How the mins are packed into the augmented columns (--gguf-mins). Exact: the
// file's mins bit for bit, one super-block per augmented group (+12.5 % on a
// Q4_K set). Shared: two super-blocks per augmented group under the larger
// dmin, the other block's mins requantised in steps of it (+6.25 %). Nibble:
// one nibble per group under a scale shared by 32 groups, the largest min of
// the 32 at 15 (+3.1 %). The inexact forms err by at most half the shared
// scale per group min; the load reports their deviation instead of refusing it.
// Split: no augmented columns at all (k_aug == 0); every group's min goes into
// `RepackedTensor::min_matrix` instead, one f16 value per row per group, so
// the runtime can add the min term through a second, unquantised MatMul
// instead of folding it into the widened, quantised fully-connected (the
// widening's group-sum columns are what --dyn-quant on was destroying).
enum class RepackMins { Exact, Shared, Nibble, Split };

struct RepackedTensor {
    RepackWeights   weights_type = RepackWeights::U4;
    RepackZeroPoint zp_type      = RepackZeroPoint::None;
    int64_t n = 0;               // output rows
    int64_t k = 0;               // contraction, the model's
    int64_t k_aug = 0;           // extra columns carrying the mins (0 for Q6_K, Q8_0)
    int64_t group = 32;          // values per scale
    int64_t groups_per_aug = 8;  // groups whose mins share one augmented group (a super-block's 8; 4 under a head-wise column order; 16 / 32 for the inexact packings)
    RepackMins mins = RepackMins::Exact;
    int64_t aug_slots = 2;       // augmented columns per group: 2 (hi and lo nibbles of a 6-bit min) or 1 (one nibble, or one u8 byte)
    std::vector<uint8_t>  weights;   // [n][(k + k_aug) values]; u4: two per byte, even index low nibble
    std::vector<uint16_t> scale;     // f16 bits, [n][(k + k_aug)/group]
    uint8_t zp_u8 = 0;               // when zp_type == U8Scalar
    std::vector<int64_t> column_dest_of;  // the column order applied at build (empty: the file's); column dest_of[c] holds the file's column c
    // RepackMins::Split only: f16 bits, [n][k/32] -- min_matrix[row*  (k/32) + g]
    // is f16(dmin_of_the_group's_super_block * mn), one rounding, the min itself
    // (positive). Empty for every other packing and for the types without a min.
    std::vector<uint16_t> min_matrix;
    int64_t width() const { return k + k_aug; }
    size_t bytes() const { return weights.size() + 2 * scale.size() + 2 * min_matrix.size(); }
};

// True for the four block types this repack serves (Q8_0, Q4_K, Q5_K, Q6_K).
bool repack_supported(int32_t ggml_type);

// Repacks one whole tensor ([n rows of k], dims[0] == k). `column_dest_of`,
// when given, reorders the model's columns at build (column dest_of[src]
// of the result is column src of the file's row -- the inverse map a gather
// on the activation would apply); it must move whole groups of 32, and the
// mins then share an augmented group per 4 groups (a 128-wide head). Throws
// std::runtime_error for an unsupported type, a k that is not a whole number
// of blocks, or a permutation that splits a group. Rows are processed in
// parallel.
RepackedTensor repack_tensor(const GgufFile& file, const TensorInfo& t, const std::vector<int64_t>* column_dest_of = nullptr,
                             RepackMins mins = RepackMins::Exact);

// The min the repacked form carries for main group `g` of row `row` (0 for
// the types without one). Exact for RepackMins::Exact, the requantised value
// for Shared/Nibble, and min_matrix's f16-rounded value for Split -- in every
// case the stored integer or value under the relevant scale, not a column of
// `weights` when the packing is Split (there are none).
float repacked_group_min(const RepackedTensor& r, int64_t row, int64_t g);

// Rows reordered after the fact: row `dest` of the result is row
// `src_of[dest]` of the input.
void permute_rows(RepackedTensor& r, const std::vector<int64_t>& src_of);

// The [k/group][k_aug] f16 matrix (row-major bits) that turns a token's
// group sums into its augmented columns: -16 and -1 per group when a group
// has two augmented columns (hi and lo nibbles), -1 when it has one (a u8
// byte, or one nibble under RepackMins::Nibble), zeros elsewhere. Empty when
// k_aug == 0.
std::vector<uint16_t> augmentation_matrix(const RepackedTensor& r);

// The plugin's arithmetic on the main columns, emulated on the host in the
// precision the kernels use (half): value = f16(f16(q - zp) * scale) with
// each intermediate rounded to f16, MINUS the group's exact min for the
// augmented types. Fills `out` with n*k values, row-major.
void dequantize_repacked(const RepackedTensor& r, std::vector<float>& out);

// Round-to-nearest-even f32 -> f16 bits (the inverse of f16_to_f32).
uint16_t f32_to_f16(float f);

// The deviation of the repacked form from ggml's dequantized values over a
// tensor's main columns, in units of the group's quantisation step (the
// scale): the largest and the root-mean-square over all values, and the
// count of values whose deviation exceeds `step_fraction`.
struct RepackDeviation {
    double max_steps = 0.0;
    double rms_steps = 0.0;
    size_t over = 0;
    size_t values = 0;
};
RepackDeviation repack_deviation(const GgufFile& file, const TensorInfo& t, const RepackedTensor& r, double step_fraction);

// The served computation for one activation row on the host, f16 roundings
// emulated (the widened activation, the plain multiply-accumulate in f32):
// y[n] for every row of the tensor. The reference beside it is the plain
// f32 dot product with ggml's dequantized weights.
void matvec_repacked_host(const RepackedTensor& r, const std::vector<float>& x, std::vector<float>& y);

}  // namespace lgc::gguf
