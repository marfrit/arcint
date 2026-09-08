#include "core/gguf_repack.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <sys/stat.h>

#include "core/gguf_dequant.h"

namespace lgc::gguf {

namespace {

// Round-to-nearest-even f32 -> binary16, the plugin's own conversion.
uint16_t to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));  // inf / nan
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);                                       // overflow -> inf
    if (exp <= 0) {                                                                                    // subnormal or zero
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) ++half;  // may carry into the exponent, correctly
    return static_cast<uint16_t>(sign | half);
}

float r16(float f) { return f16_to_f32(to_f16(f)); }  // one f16 rounding

inline float f16at(const uint8_t* p) { return f16_to_f32(static_cast<uint16_t>(p[0] | (p[1] << 8))); }

inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

size_t block_bytes_of(int32_t t) {
    switch (t) {
        case 8:  return 34;
        case 12: return 144;
        case 13: return 176;
        case 14: return 210;
        default: return 0;
    }
}

// Where the model's group `g` (of the file's row order) lands in the repacked
// row: the main group, and for the augmented types the augmented group and
// the slot within it.
struct Placement {
    std::vector<int64_t> main_group_of;   // [k/32] -> destination main group
    std::vector<int64_t> aug_group_of;    // [k/32] -> destination augmented group (index within the aug part)
    std::vector<int64_t> aug_slot_of;     // [k/32] -> slot within its augmented group (0..groups_per_aug-1)
};

Placement placement_of(int64_t k, int64_t groups_per_aug, const std::vector<int64_t>* dest_of) {
    const int64_t groups = k / 32;
    Placement p;
    p.main_group_of.resize(static_cast<size_t>(groups));
    p.aug_group_of.resize(static_cast<size_t>(groups));
    p.aug_slot_of.resize(static_cast<size_t>(groups));
    for (int64_t g = 0; g < groups; ++g) {
        int64_t dg = g;
        if (dest_of) {
            const int64_t d0 = (*dest_of)[static_cast<size_t>(g * 32)];
            if (d0 % 32 != 0) throw std::runtime_error("gguf repack: column permutation is not group-aligned");
            for (int64_t i = 1; i < 32; ++i)
                if ((*dest_of)[static_cast<size_t>(g * 32 + i)] != d0 + i)
                    throw std::runtime_error("gguf repack: column permutation splits a group");
            dg = d0 / 32;
        }
        p.main_group_of[static_cast<size_t>(g)] = dg;
        // The augmented slot follows the DESTINATION group order, so that the
        // groups sharing an augmented group (and its dmin) are the ones that
        // sit together after the permutation -- a head's four groups under a
        // head-wise order, a super-block's eight otherwise.
        p.aug_group_of[static_cast<size_t>(g)] = dg / groups_per_aug;
        p.aug_slot_of[static_cast<size_t>(g)] = dg % groups_per_aug;
    }
    return p;
}

void repack_row(int32_t type, const uint8_t* row, RepackedTensor& r, int64_t n_row, const Placement& pl) {
    const int64_t k = r.k;
    const int64_t wgroups = r.width() / r.group;
    uint16_t* sc = r.scale.data() + n_row * wgroups;
    if (type == 8) {  // Q8_0: i8 with the block's f16 scale, exact
        int8_t* w = reinterpret_cast<int8_t*>(r.weights.data()) + n_row * k;
        for (int64_t b = 0; b < k / 32; ++b) {
            const uint8_t* blk = row + b * 34;
            sc[b] = static_cast<uint16_t>(blk[0] | (blk[1] << 8));
            std::memcpy(w + b * 32, blk + 2, 32);
        }
        return;
    }
    if (type == 12 || type == 13) {  // Q4_K (u4) / Q5_K (u8): the values, and the mins
        const bool u4 = type == 12;
        const bool split = r.mins == RepackMins::Split;
        const size_t gbytes = u4 ? 16 : 32;
        uint8_t* w = r.weights.data() + n_row * (static_cast<size_t>(r.width()) / (u4 ? 2 : 1));
        const int64_t main_groups = k / 32;
        // RepackMins::Split: min_matrix[row][g] takes the group's min directly
        // (one float multiply of the file's f16 dmin and the integer mn,
        // rounded to f16 once) -- no augmented group, no shared scale.
        uint16_t* min_row = split ? r.min_matrix.data() + n_row * main_groups : nullptr;
        const int64_t aug_first = main_groups;  // the augmented groups follow the main ones (Exact/Shared/Nibble only)
        // The mins are written after the row's main groups: an augmented group's scale is the
        // dmin of the one super-block it serves (Exact) or chosen over the groups it is shared by
        // (Shared: the largest dmin; Nibble: the largest min at 15), and every group's integer
        // is then expressed under that scale.
        struct GroupMin { int64_t ag; int64_t slot; float dmin; uint8_t m6; };
        std::vector<GroupMin> gmins;
        if (!split) gmins.resize(static_cast<size_t>(main_groups));
        for (int64_t b = 0; b < k / 256; ++b) {
            const uint8_t* blk = row + b * (u4 ? 144 : 176);
            const float d = f16at(blk), dmin = f16at(blk + 2);
            const uint8_t* qh = blk + 16;
            const uint8_t* qs = u4 ? blk + 16 : blk + 48;
            for (int s = 0; s < 8; ++s) {
                uint8_t s6, m6; scale_min_k4(s, blk + 4, s6, m6);
                const int64_t g = b * 8 + s;
                const int64_t dg = pl.main_group_of[static_cast<size_t>(g)];
                sc[dg] = to_f16(d * static_cast<float>(s6));
                uint8_t q[32];
                for (int l = 0; l < 32; ++l) {
                    uint8_t v = static_cast<uint8_t>((qs[(s >> 1) * 32 + l] >> ((s & 1) * 4)) & 0xF);
                    if (!u4) v = static_cast<uint8_t>(v + ((qh[l] & static_cast<uint8_t>(1u << s)) ? 16 : 0));
                    q[l] = v;
                }
                uint8_t* out = w + dg * gbytes;
                if (u4) for (int l = 0; l < 32; l += 2) out[l / 2] = static_cast<uint8_t>(q[l] | (q[l + 1] << 4));
                else std::memcpy(out, q, 32);
                if (split) min_row[dg] = to_f16(dmin * static_cast<float>(m6));
                else gmins[static_cast<size_t>(g)] = GroupMin{aug_first + pl.aug_group_of[static_cast<size_t>(g)], pl.aug_slot_of[static_cast<size_t>(g)], dmin, m6};
            }
        }
        if (split) return;  // no augmented columns: the mins are min_matrix's alone
        const int64_t aug_groups = r.k_aug / 32;
        std::vector<float> s(static_cast<size_t>(aug_groups), 0.0f);
        for (const auto& gm : gmins) {
            const size_t a = static_cast<size_t>(gm.ag - aug_first);
            if (r.mins == RepackMins::Nibble && u4) s[a] = std::max(s[a], gm.dmin * static_cast<float>(gm.m6) / 15.0f);
            else s[a] = std::max(s[a], gm.dmin);
        }
        for (int64_t a = 0; a < aug_groups; ++a) sc[aug_first + a] = to_f16(s[static_cast<size_t>(a)]);
        const int cap = (u4 && r.aug_slots == 1) ? 15 : (u4 ? 63 : 255);
        for (const auto& gm : gmins) {
            const float sa = f16_to_f32(sc[gm.ag]);   // the stored scale: the integer must match what the kernel multiplies
            long mn = sa > 0.0f ? std::lround(gm.dmin * static_cast<float>(gm.m6) / sa) : 0;  // Exact: dmin == sa, so m6 itself
            mn = std::max(0L, std::min(static_cast<long>(cap), mn));
            uint8_t* aug = w + gm.ag * gbytes;
            if (!u4) {
                aug[gm.slot] = static_cast<uint8_t>(mn);
            } else if (r.aug_slots == 2) {
                // columns 2*slot (hi) and 2*slot+1 (lo): one byte, hi in the low nibble (even column)
                aug[gm.slot] = static_cast<uint8_t>((mn >> 4) | ((mn & 0xF) << 4));
            } else {
                // one nibble at column slot: byte slot/2, the even column in the low nibble
                uint8_t& byte = aug[gm.slot / 2];
                if (gm.slot & 1) byte = static_cast<uint8_t>((byte & 0x0F) | (static_cast<uint8_t>(mn) << 4));
                else byte = static_cast<uint8_t>((byte & 0xF0) | static_cast<uint8_t>(mn));
            }
        }
        return;
    }
    if (type == 14) {  // Q6_K: u8 (0..63) with the zero point 32, scale per 16 (ggml's own grouping)
        uint8_t* w = r.weights.data() + n_row * k;
        for (int64_t b = 0; b < k / 256; ++b) {
            const uint8_t* blk = row + b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
            const float d = f16at(blk + 208);
            for (int hh = 0; hh < 2; ++hh) {
                const uint8_t* qlh = ql + hh * 64;
                const uint8_t* qhh = qh + hh * 32;
                const int8_t* sch = scales + hh * 8;
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const uint8_t q1 = static_cast<uint8_t>((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4));
                    const uint8_t q2 = static_cast<uint8_t>((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4));
                    const uint8_t q3 = static_cast<uint8_t>((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4));
                    const uint8_t q4 = static_cast<uint8_t>((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4));
                    const int64_t base = b * 256 + hh * 128;
                    w[base + l] = q1; w[base + 32 + l] = q2; w[base + 64 + l] = q3; w[base + 96 + l] = q4;
                    if (l % 16 == 0) {
                        const int64_t g = (base + l) / 16;   // run 0 of this half, then +2, +4, +6 for the runs below
                        sc[g]     = to_f16(d * static_cast<float>(sch[is + 0]));
                        sc[g + 2] = to_f16(d * static_cast<float>(sch[is + 2]));
                        sc[g + 4] = to_f16(d * static_cast<float>(sch[is + 4]));
                        sc[g + 6] = to_f16(d * static_cast<float>(sch[is + 6]));
                    }
                }
            }
        }
        return;
    }
    throw std::runtime_error("gguf repack: unsupported type " + std::to_string(type));
}

// The value at main column c of row `row`, decoded the way the plugin computes it
// (half arithmetic), before the min.
float main_value(const RepackedTensor& r, int64_t row, int64_t c) {
    const int64_t wgroups = r.width() / r.group;
    const float s = f16_to_f32(r.scale[row * wgroups + c / r.group]);
    const int64_t idx = row * r.width() + c;
    float q;
    if (r.weights_type == RepackWeights::U4) q = static_cast<float>((r.weights[idx / 2] >> ((idx & 1) * 4)) & 0xF);
    else if (r.weights_type == RepackWeights::U8) q = static_cast<float>(r.weights[idx]);
    else q = static_cast<float>(static_cast<int8_t>(r.weights[idx]));
    const float zp = r.zp_type == RepackZeroPoint::U8Scalar ? static_cast<float>(r.zp_u8) : 0.0f;
    return r16(r16(q - zp) * s);
}

// The min the repacked form carries for main group `g` of row `row` (0 for the
// types without one): the stored integer under the augmented group's scale,
// or min_matrix's own f16-rounded value under RepackMins::Split.
float min_of_group(const RepackedTensor& r, int64_t row, int64_t g) {
    if (r.mins == RepackMins::Split) {
        if (r.min_matrix.empty()) return 0.0f;
        const int64_t groups = r.k / 32;
        return f16_to_f32(r.min_matrix[static_cast<size_t>(row * groups + g)]);
    }
    if (r.k_aug == 0) return 0.0f;
    const int64_t wgroups = r.width() / r.group;
    const int64_t ag = r.k / 32 + g / r.groups_per_aug, slot = g % r.groups_per_aug;
    const float s = f16_to_f32(r.scale[row * wgroups + ag]);
    const bool u4 = r.weights_type == RepackWeights::U4;
    const size_t gbytes = u4 ? 16 : 32;
    const uint8_t* aug = r.weights.data() + row * (u4 ? static_cast<size_t>(r.width()) / 2 : static_cast<size_t>(r.width())) + ag * gbytes;
    int mn;
    if (!u4) mn = aug[slot];
    else if (r.aug_slots == 2) { const uint8_t byte = aug[slot]; mn = ((byte & 0xF) << 4) | (byte >> 4); }
    else mn = (aug[slot / 2] >> ((slot & 1) * 4)) & 0xF;
    return s * static_cast<float>(mn);
}

}  // namespace

uint16_t f32_to_f16(float f) { return to_f16(f); }
float repacked_group_min(const RepackedTensor& r, int64_t row, int64_t g) { return min_of_group(r, row, g); }

bool repack_supported(int32_t t) { return t == 8 || t == 12 || t == 13 || t == 14; }

RepackedTensor repack_tensor(const GgufFile& file, const TensorInfo& t, const std::vector<int64_t>* column_dest_of, RepackMins mins,
                             unsigned threads) {
    if (!repack_supported(t.ggml_type))
        throw std::runtime_error("gguf repack: " + t.name + " is " + type_name(t.ggml_type) + ", not a repacked type");
    if (t.dims.size() != 2) throw std::runtime_error("gguf repack: " + t.name + " is not 2-D");
    RepackedTensor r;
    r.k = static_cast<int64_t>(t.dims[0]);
    r.n = static_cast<int64_t>(t.dims[1]);
    const int64_t block = t.ggml_type == 8 ? 32 : 256;
    if (r.k % block != 0) throw std::runtime_error("gguf repack: " + t.name + "'s row is not a whole number of blocks");
    if (column_dest_of && static_cast<int64_t>(column_dest_of->size()) != r.k)
        throw std::runtime_error("gguf repack: column permutation size mismatch for " + t.name);
    switch (t.ggml_type) {
        case 8:  r.weights_type = RepackWeights::I8; r.group = 32; break;
        case 12: r.weights_type = RepackWeights::U4; r.group = 32; break;
        case 13: r.weights_type = RepackWeights::U8; r.group = 32; break;
        case 14: r.weights_type = RepackWeights::U8; r.zp_type = RepackZeroPoint::U8Scalar; r.group = 16; r.zp_u8 = 32; break;
    }
    if (t.ggml_type == 12 || t.ggml_type == 13) {
        // A u4 group's min takes two columns (hi, lo nibbles), a u8 group's one byte; the
        // Nibble packing one nibble. The groups sharing an augmented group of 32 columns:
        // a super-block's eight (Exact), sixteen or thirty-two for the inexact packings, four
        // (a 128-wide head) under a head-wise column order so that a head's groups keep one dmin.
        const bool u4 = t.ggml_type == 12;
        r.mins = mins;
        if (mins == RepackMins::Split) {
            // No augmented columns: min_matrix carries the mins instead (allocated below).
            r.k_aug = 0;
        } else {
            r.aug_slots = (u4 && mins != RepackMins::Nibble) ? 2 : 1;
            const int64_t per_group = 32 / r.aug_slots;                  // groups one augmented group can hold
            if (mins == RepackMins::Exact) r.groups_per_aug = column_dest_of ? 4 : 8;
            else if (mins == RepackMins::Shared) r.groups_per_aug = std::min<int64_t>(per_group, u4 ? 16 : 32);
            else r.groups_per_aug = per_group;                           // Nibble: 32 (u4), 32 (u8, the same byte as Shared)
            r.k_aug = ((r.k / 32 + r.groups_per_aug - 1) / r.groups_per_aug) * 32;   // one augmented group of 32 columns per groups_per_aug groups
            // The runtime's int4 fully-connected walks K in pairs of groups: a width with an odd
            // number of groups faulted on the card (CL_OUT_OF_RESOURCES on the first forward of
            // the Nibble packing at K = 5,120: 165 groups; DESIGN 7.0.2bl). One zero group more.
            if (((r.k + r.k_aug) / 32) % 2 != 0) r.k_aug += 32;
        }
    } else if (column_dest_of) {
        throw std::runtime_error("gguf repack: a column order is only applied to Q4_K/Q5_K tensors (" + t.name + ")");
    }
    const Placement pl = (t.ggml_type == 12 || t.ggml_type == 13) ? placement_of(r.k, r.groups_per_aug, column_dest_of) : Placement{};
    if (column_dest_of) r.column_dest_of = *column_dest_of;
    const size_t wgroups = static_cast<size_t>(r.width() / r.group);
    r.weights.assign(static_cast<size_t>(r.n * r.width()) / (r.weights_type == RepackWeights::U4 ? 2 : 1), 0);
    r.scale.assign(static_cast<size_t>(r.n) * wgroups, 0);
    if (mins == RepackMins::Split && (t.ggml_type == 12 || t.ggml_type == 13))
        r.min_matrix.assign(static_cast<size_t>(r.n * (r.k / 32)), 0);
    const size_t row_bytes = static_cast<size_t>(r.k / block) * block_bytes_of(t.ggml_type);
    const uint8_t* data = file.data(t);
    const unsigned n_threads = threads != 0 ? threads : std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    std::vector<std::thread> pool;
    for (unsigned th = 0; th < n_threads; ++th)
        pool.emplace_back([&, th]() {
            for (int64_t row = th; row < r.n; row += n_threads) repack_row(t.ggml_type, data + row * row_bytes, r, row, pl);
        });
    for (auto& th : pool) th.join();
    return r;
}

void permute_rows(RepackedTensor& r, const std::vector<int64_t>& src_of) {
    if (static_cast<int64_t>(src_of.size()) != r.n) throw std::runtime_error("gguf repack: row permutation size mismatch");
    const size_t wrow = r.weights.size() / static_cast<size_t>(r.n);
    const size_t groups = r.scale.size() / static_cast<size_t>(r.n);
    const size_t mgroups = r.min_matrix.empty() ? 0 : r.min_matrix.size() / static_cast<size_t>(r.n);
    std::vector<uint8_t> w(r.weights.size());
    std::vector<uint16_t> sc(r.scale.size());
    std::vector<uint16_t> mm(r.min_matrix.size());
    for (int64_t dest = 0; dest < r.n; ++dest) {
        const int64_t src = src_of[static_cast<size_t>(dest)];
        if (src < 0 || src >= r.n) throw std::runtime_error("gguf repack: row permutation out of range");
        std::copy_n(r.weights.data() + src * wrow, wrow, w.data() + dest * wrow);
        std::copy_n(r.scale.data() + src * groups, groups, sc.data() + dest * groups);
        if (mgroups) std::copy_n(r.min_matrix.data() + src * mgroups, mgroups, mm.data() + dest * mgroups);
    }
    r.weights.swap(w); r.scale.swap(sc);
    if (mgroups) r.min_matrix.swap(mm);
}

std::vector<uint16_t> augmentation_matrix(const RepackedTensor& r) {
    if (r.k_aug == 0) return {};
    const int64_t groups = r.k / 32;
    std::vector<uint16_t> m(static_cast<size_t>(groups * r.k_aug), 0);
    const uint16_t m16 = to_f16(-16.0f), m1 = to_f16(-1.0f);
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t ag = g / r.groups_per_aug, slot = g % r.groups_per_aug;
        if (r.aug_slots == 2) {
            m[static_cast<size_t>(g * r.k_aug + ag * 32 + 2 * slot)] = m16;
            m[static_cast<size_t>(g * r.k_aug + ag * 32 + 2 * slot + 1)] = m1;
        } else {
            m[static_cast<size_t>(g * r.k_aug + ag * 32 + slot)] = m1;
        }
    }
    return m;
}

void dequantize_repacked(const RepackedTensor& r, std::vector<float>& out) {
    out.resize(static_cast<size_t>(r.n * r.k));
    for (int64_t row = 0; row < r.n; ++row)
        for (int64_t c = 0; c < r.k; ++c)
            out[static_cast<size_t>(row * r.k + c)] = main_value(r, row, c) - min_of_group(r, row, c / 32);
}

RepackDeviation repack_deviation(const GgufFile& file, const TensorInfo& t, const RepackedTensor& r, double step_fraction,
                                 unsigned threads) {
    // Row by row in parallel (a whole tensor dequantized twice is gigabytes;
    // the served file has 25.6 G values and this runs at every load). A
    // column-reordered tensor is compared through the order it was built
    // with, so the reference stays the file's row.
    const int64_t wgroups = r.width() / r.group;
    const size_t row_bytes = static_cast<size_t>(r.k / (t.ggml_type == 8 ? 32 : 256)) * block_bytes_of(t.ggml_type);
    const uint8_t* data = file.data(t);
    const unsigned n_threads = threads != 0 ? threads : std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    // Every row's own max/sum-of-squares/over-count is exact and independent
    // of threading; only the ACROSS-row combination of the sum of squares is
    // a floating-point reduction, and summing threads' partials in thread
    // order (as the previous version did) makes that reduction's rounding
    // depend on how many threads happened to run -- observable as a
    // `rms_steps` that moved with `threads` alone, nothing else. Reducing by
    // row index instead, 0 to n-1, fixes the order regardless of the thread
    // count: gguf_repack_all runs the very same tensor through a smaller
    // pool than a lone call would, and its result must not differ for that.
    std::vector<double> row_max(static_cast<size_t>(r.n), 0.0);
    std::vector<double> row_sq(static_cast<size_t>(r.n), 0.0);
    std::vector<size_t> row_over(static_cast<size_t>(r.n), 0);
    std::vector<std::thread> pool;
    for (unsigned th = 0; th < n_threads; ++th)
        pool.emplace_back([&, th]() {
            std::vector<float> ref(static_cast<size_t>(r.k));
            for (int64_t row = th; row < r.n; row += n_threads) {
                dequantize_row(t.ggml_type, data + row * row_bytes, static_cast<size_t>(r.k), ref.data());
                double mx = 0.0, sq = 0.0;
                size_t over = 0;
                for (int64_t c = 0; c < r.k; ++c) {
                    const int64_t dc = r.column_dest_of.empty() ? c : r.column_dest_of[static_cast<size_t>(c)];  // where the file's column c sits
                    const float got = main_value(r, row, dc) - min_of_group(r, row, dc / 32);
                    const double step = std::fabs(static_cast<double>(f16_to_f32(r.scale[row * wgroups + dc / r.group])));
                    const double e = std::fabs(static_cast<double>(got) - static_cast<double>(ref[c])) / (step > 0 ? step : 1.0);
                    mx = std::max(mx, e);
                    sq += e * e;
                    if (e > step_fraction) ++over;
                }
                row_max[static_cast<size_t>(row)] = mx;
                row_sq[static_cast<size_t>(row)] = sq;
                row_over[static_cast<size_t>(row)] = over;
            }
        });
    for (auto& th : pool) th.join();
    RepackDeviation dv;
    double s = 0.0;
    for (int64_t row = 0; row < r.n; ++row) {
        dv.max_steps = std::max(dv.max_steps, row_max[static_cast<size_t>(row)]);
        dv.over += row_over[static_cast<size_t>(row)];
        s += row_sq[static_cast<size_t>(row)];
    }
    dv.values = static_cast<size_t>(r.n) * static_cast<size_t>(r.k);
    dv.rms_steps = dv.values ? std::sqrt(s / static_cast<double>(dv.values)) : 0.0;
    return dv;
}

void matvec_repacked_host(const RepackedTensor& r, const std::vector<float>& x, std::vector<float>& y) {
    if (static_cast<int64_t>(x.size()) != r.k) throw std::runtime_error("gguf repack: activation length mismatch");
    if (r.mins == RepackMins::Split && !r.min_matrix.empty()) {
        // The served split computation: two independent MatMuls, each rounded to
        // f16 on the way out (like every fully-connected on this path), added
        // once more in f16 -- not a single running accumulator over both terms.
        std::vector<float> xr(static_cast<size_t>(r.k));
        for (int64_t c = 0; c < r.k; ++c) xr[static_cast<size_t>(c)] = r16(x[static_cast<size_t>(c)]);
        const int64_t groups = r.k / 32;
        std::vector<float> gs(static_cast<size_t>(groups));
        for (int64_t g = 0; g < groups; ++g) {
            float s = 0; for (int64_t i = 0; i < 32; ++i) s += xr[static_cast<size_t>(g * 32 + i)];
            gs[static_cast<size_t>(g)] = r16(s);
        }
        y.assign(static_cast<size_t>(r.n), 0.0f);
        for (int64_t row = 0; row < r.n; ++row) {
            double main_acc = 0;
            for (int64_t c = 0; c < r.k; ++c) main_acc += static_cast<double>(xr[static_cast<size_t>(c)]) * static_cast<double>(main_value(r, row, c));
            const float main_t = r16(static_cast<float>(main_acc));
            double min_acc = 0;
            for (int64_t g = 0; g < groups; ++g)
                min_acc += static_cast<double>(gs[static_cast<size_t>(g)]) * static_cast<double>(-f16_to_f32(r.min_matrix[static_cast<size_t>(row * groups + g)]));
            const float min_t = r16(static_cast<float>(min_acc));
            y[static_cast<size_t>(row)] = r16(main_t + min_t);
        }
        return;
    }
    // The widened activation: x (f16), then the group sums through the augmentation matrix (f16).
    std::vector<float> xa(static_cast<size_t>(r.width()), 0.0f);
    for (int64_t c = 0; c < r.k; ++c) xa[static_cast<size_t>(c)] = r16(x[static_cast<size_t>(c)]);
    if (r.k_aug) {
        const auto m = augmentation_matrix(r);
        const int64_t groups = r.k / 32;
        std::vector<float> gs(static_cast<size_t>(groups), 0.0f);
        for (int64_t g = 0; g < groups; ++g) {
            float s = 0; for (int64_t i = 0; i < 32; ++i) s += xa[static_cast<size_t>(g * 32 + i)];
            gs[static_cast<size_t>(g)] = r16(s);
        }
        for (int64_t a = 0; a < r.k_aug; ++a) {
            float s = 0;
            for (int64_t g = 0; g < groups; ++g) s += gs[static_cast<size_t>(g)] * f16_to_f32(m[static_cast<size_t>(g * r.k_aug + a)]);
            xa[static_cast<size_t>(r.k + a)] = r16(s);
        }
    }
    y.assign(static_cast<size_t>(r.n), 0.0f);
    const int64_t wgroups = r.width() / r.group;
    for (int64_t row = 0; row < r.n; ++row) {
        double acc = 0;
        for (int64_t c = 0; c < r.width(); ++c) {
            const float s = f16_to_f32(r.scale[row * wgroups + c / r.group]);
            const int64_t idx = row * r.width() + c;
            float q;
            if (r.weights_type == RepackWeights::U4) q = static_cast<float>((r.weights[idx / 2] >> ((idx & 1) * 4)) & 0xF);
            else if (r.weights_type == RepackWeights::U8) q = static_cast<float>(r.weights[idx]);
            else q = static_cast<float>(static_cast<int8_t>(r.weights[idx]));
            const float zp = r.zp_type == RepackZeroPoint::U8Scalar ? static_cast<float>(r.zp_u8) : 0.0f;
            acc += static_cast<double>(xa[static_cast<size_t>(c)]) * static_cast<double>(r16(r16(q - zp) * s));
        }
        y[static_cast<size_t>(row)] = static_cast<float>(acc);
    }
}

double repack_bound_steps(int32_t ggml_type) {
    switch (ggml_type) {
        case 8:  return 1.0 / 16.0;   // Q8_0, |q| <= 127: the value rounding alone
        case 12: return 1.0 / 64.0;   // Q4_K, |q| <= 15: the value and the scale rounding
        default: return 1.0 / 32.0;   // Q5_K |q| <= 31; Q6_K |q - 32| <= 32
    }
}

namespace {

// The deviation verdict of one repacked projection, kept between loads
// (moved here, unchanged, from gguf_graph.cpp's anonymous namespace -- the
// caller no longer needs its own copy).
struct Verdict {
    double max_steps = 0.0;
    size_t values = 0;
};

// The packing is part of the key (gguf_graph.cpp's scheme before this cache
// moved here, unchanged): a numeric offset added to the bound, 0 for Exact so
// an Exact verdict file written by an old build is still found by a new one.
// Without this, an Exact load could read back a verdict a Shared/Nibble load
// of the same tensor/bound had written -- those never refuse and always
// write, so a bad Exact repack could be waved through on a stale, unrelated
// pass (tests/test_gguf_parallel.cpp's collision test).
double verdict_mins_offset(RepackMins mins) {
    switch (mins) {
        case RepackMins::Exact:  return 0.0;
        case RepackMins::Shared: return 1000.0;
        case RepackMins::Nibble: return 2000.0;
        case RepackMins::Split:  return 3000.0;
    }
    return 0.0;
}

std::string verdict_key(const std::string& file_path, const TensorInfo& t, double bound, RepackMins mins) {
    struct stat st {};
    if (file_path.empty() || ::stat(file_path.c_str(), &st) != 0) return "";
    std::ostringstream k;
    k << "repack-v1|" << st.st_size << '|' << st.st_mtim.tv_sec << '.' << st.st_mtim.tv_nsec << '|' << t.name << '|'
      << t.offset << '|' << t.ggml_type << '|' << (bound + verdict_mins_offset(mins));
    for (auto d : t.dims) k << 'x' << d;
    // FNV-1a over the key text: a file name, not a claim of uniqueness beyond the fields above.
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : k.str()) { h ^= c; h *= 1099511628211ull; }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

bool read_verdict(const std::string& dir, const std::string& key, Verdict& v) {
    if (dir.empty() || key.empty()) return false;
    std::ifstream in(std::filesystem::path(dir) / (key + ".verdict"));
    std::string tag;
    if (!(in >> tag >> v.max_steps >> v.values) || tag != "ok") return false;
    return true;
}

void write_verdict(const std::string& dir, const std::string& key, const Verdict& v) {
    if (dir.empty() || key.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return;
    std::ofstream out(std::filesystem::path(dir) / (key + ".verdict"));
    out.precision(17);
    out << "ok " << v.max_steps << ' ' << v.values << '\n';
}

}  // namespace

std::vector<RepackResult> gguf_repack_all(const GgufFile& file, const std::vector<RepackRequest>& requests,
                                          RepackMins mins, const std::string& verdict_dir,
                                          const std::string& file_path, unsigned threads) {
    std::vector<RepackResult> results(requests.size());
    std::vector<std::string> errors(requests.size());  // per-request; empty means no error

    // The cross-tensor pool: N workers (>= 1), each taking whole requests off
    // the list until none are left. The pool never grows past the number of
    // requests -- a lone request gets one worker, not `n_workers` of them
    // idling. Reduce the intra-tensor thread count from the pool size that
    // ACTUALLY runs (`actual_workers`), not the raw requested `threads`: a
    // single request on a 16-thread host used to compute intra from
    // n_workers == threads (== 16 by default) and repack single-threaded,
    // even though only one worker ever ran.
    const unsigned n_workers = std::max<unsigned>(1u, threads != 0 ? threads : std::thread::hardware_concurrency());
    const unsigned actual_workers = static_cast<unsigned>(std::min<size_t>(n_workers, std::max<size_t>(requests.size(), 1)));
    const unsigned pool_default = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    const unsigned intra = std::max(1u, pool_default / actual_workers);

    // Created once, up front, on the calling thread: write_verdict below also
    // calls create_directories, but several workers writing a verdict for
    // different tensors at once would otherwise race to create the same
    // directory concurrently. Ignored on failure the same way write_verdict
    // ignores it -- a verdict that cannot be written is a slower next load,
    // never a wrong one.
    if (!verdict_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(verdict_dir, ec);
    }

    auto process_one = [&](size_t i) {
        const RepackRequest& req = requests[i];
        if (req.tensor == nullptr) { errors[i] = "gguf: repack request has no tensor"; return; }
        const TensorInfo& t = *req.tensor;
        try {
            RepackedTensor packed = repack_tensor(file, t, req.column_dest_of.empty() ? nullptr : &req.column_dest_of, mins, intra);
            RepackResult res;
            const std::string key = verdict_key(file_path, t, req.bound, mins);
            Verdict v;
            if (read_verdict(verdict_dir, key, v)) {
                // Checked and passed by an earlier load of this very file (size, mtime,
                // offset, type, dims, bound); the repack itself is deterministic in the
                // bytes, so there is nothing left to measure.
                res.max_steps = v.max_steps;
                res.verdict_cached = true;
            } else {
                const auto dv = repack_deviation(file, t, packed, req.bound, intra);
                res.max_steps = dv.max_steps;
                res.over_bound = dv.over;
                res.checked = dv.values;
                // An inexact mins packing (--gguf-mins shared|nibble) is a chosen cost:
                // its deviation is reported, never refused (DESIGN 7.0.2bl).
                if (dv.over != 0 && (mins == RepackMins::Exact || mins == RepackMins::Split)) {
                    errors[i] = "gguf: " + t.name + " repacked outside its bound: " + std::to_string(dv.over) +
                                " value(s) over " + std::to_string(req.bound) +
                                " quantisation step(s), max " + std::to_string(dv.max_steps);
                    return;
                }
                write_verdict(verdict_dir, key, Verdict{dv.max_steps, dv.values});
            }
            res.packed = std::move(packed);
            results[i] = std::move(res);
        } catch (const std::exception& e) {
            errors[i] = e.what();
        }
    };

    // Largest tensor first: a handful of big projections (the dense model's
    // largest Q4_K sets) otherwise end up the tail, each finishing alone
    // after every small tensor's worker has gone idle. Sorting the request
    // INDICES (not `requests` itself) keeps every result at its own request's
    // slot in `results`/`errors` regardless of the order they are processed in.
    std::vector<size_t> order(requests.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        auto size_of = [&](size_t i) -> int64_t {
            const auto* t = requests[i].tensor;
            return (t && t->dims.size() >= 2) ? static_cast<int64_t>(t->dims[0]) * static_cast<int64_t>(t->dims[1]) : 0;
        };
        return size_of(a) > size_of(b);
    });

    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < actual_workers; ++w)
        pool.emplace_back([&]() {
            for (size_t j = next.fetch_add(1); j < order.size(); j = next.fetch_add(1)) process_one(order[j]);
        });
    for (auto& w : pool) w.join();

    // Rethrown on the calling thread, in the requests' own order, after every
    // one of them has been attempted -- not on the first failure, so a
    // tensor later in the list is not left unprocessed just because an
    // earlier one (running concurrently) happened to fail first.
    for (const auto& e : errors)
        if (!e.empty()) throw std::runtime_error(e);
    return results;
}

}  // namespace lgc::gguf
