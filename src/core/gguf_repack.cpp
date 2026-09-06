#include "core/gguf_repack.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

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
    if (type == 12 || type == 13) {  // Q4_K (u4) / Q5_K (u8): the values, and the mins as augmented columns
        const bool u4 = type == 12;
        const size_t gbytes = u4 ? 16 : 32;
        uint8_t* w = r.weights.data() + n_row * (static_cast<size_t>(r.width()) / (u4 ? 2 : 1));
        const int64_t main_groups = k / 32;
        const int64_t aug_first = main_groups;  // the augmented groups follow the main ones
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
                // The min: mn under dmin, in the augmented group this group shares.
                const int64_t ag = aug_first + pl.aug_group_of[static_cast<size_t>(g)];
                const int64_t slot = pl.aug_slot_of[static_cast<size_t>(g)];
                sc[ag] = to_f16(dmin);
                uint8_t* aug = w + ag * gbytes;
                if (u4) {
                    // columns 2*slot (hi) and 2*slot+1 (lo): one byte, hi in the low nibble (even column)
                    aug[slot] = static_cast<uint8_t>((m6 >> 4) | ((m6 & 0xF) << 4));
                } else {
                    aug[slot] = m6;
                }
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

// The exact min of main group `g` of row `row` (0 for the types without one).
float min_of_group(const RepackedTensor& r, int64_t row, int64_t g) {
    if (r.k_aug == 0) return 0.0f;
    const int64_t wgroups = r.width() / r.group;
    const int64_t ag = r.k / 32 + g / r.groups_per_aug, slot = g % r.groups_per_aug;
    const float dmin = f16_to_f32(r.scale[row * wgroups + ag]);
    const size_t gbytes = r.weights_type == RepackWeights::U4 ? 16 : 32;
    const uint8_t byte = r.weights[row * (r.weights_type == RepackWeights::U4 ? static_cast<size_t>(r.width()) / 2 : static_cast<size_t>(r.width())) + ag * gbytes + slot];
    const int mn = r.weights_type == RepackWeights::U4 ? ((byte & 0xF) << 4) | (byte >> 4) : byte;
    return dmin * static_cast<float>(mn);
}

}  // namespace

uint16_t f32_to_f16(float f) { return to_f16(f); }

bool repack_supported(int32_t t) { return t == 8 || t == 12 || t == 13 || t == 14; }

RepackedTensor repack_tensor(const GgufFile& file, const TensorInfo& t, const std::vector<int64_t>* column_dest_of) {
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
        r.groups_per_aug = column_dest_of ? 4 : 8;
        r.k_aug = (r.k / 32 / r.groups_per_aug) * 32;   // one augmented group of 32 columns per groups_per_aug groups
    } else if (column_dest_of) {
        throw std::runtime_error("gguf repack: a column order is only applied to Q4_K/Q5_K tensors (" + t.name + ")");
    }
    const Placement pl = (t.ggml_type == 12 || t.ggml_type == 13) ? placement_of(r.k, r.groups_per_aug, column_dest_of) : Placement{};
    if (column_dest_of) r.column_dest_of = *column_dest_of;
    const size_t wgroups = static_cast<size_t>(r.width() / r.group);
    r.weights.assign(static_cast<size_t>(r.n * r.width()) / (r.weights_type == RepackWeights::U4 ? 2 : 1), 0);
    r.scale.assign(static_cast<size_t>(r.n) * wgroups, 0);
    const size_t row_bytes = static_cast<size_t>(r.k / block) * block_bytes_of(t.ggml_type);
    const uint8_t* data = file.data(t);
    const unsigned threads = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    std::vector<std::thread> pool;
    for (unsigned th = 0; th < threads; ++th)
        pool.emplace_back([&, th]() {
            for (int64_t row = th; row < r.n; row += threads) repack_row(t.ggml_type, data + row * row_bytes, r, row, pl);
        });
    for (auto& th : pool) th.join();
    return r;
}

void permute_rows(RepackedTensor& r, const std::vector<int64_t>& src_of) {
    if (static_cast<int64_t>(src_of.size()) != r.n) throw std::runtime_error("gguf repack: row permutation size mismatch");
    const size_t wrow = r.weights.size() / static_cast<size_t>(r.n);
    const size_t groups = r.scale.size() / static_cast<size_t>(r.n);
    std::vector<uint8_t> w(r.weights.size());
    std::vector<uint16_t> sc(r.scale.size());
    for (int64_t dest = 0; dest < r.n; ++dest) {
        const int64_t src = src_of[static_cast<size_t>(dest)];
        if (src < 0 || src >= r.n) throw std::runtime_error("gguf repack: row permutation out of range");
        std::copy_n(r.weights.data() + src * wrow, wrow, w.data() + dest * wrow);
        std::copy_n(r.scale.data() + src * groups, groups, sc.data() + dest * groups);
    }
    r.weights.swap(w); r.scale.swap(sc);
}

std::vector<uint16_t> augmentation_matrix(const RepackedTensor& r) {
    if (r.k_aug == 0) return {};
    const int64_t groups = r.k / 32;
    std::vector<uint16_t> m(static_cast<size_t>(groups * r.k_aug), 0);
    const uint16_t m16 = to_f16(-16.0f), m1 = to_f16(-1.0f);
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t ag = g / r.groups_per_aug, slot = g % r.groups_per_aug;
        if (r.weights_type == RepackWeights::U4) {
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

RepackDeviation repack_deviation(const GgufFile& file, const TensorInfo& t, const RepackedTensor& r, double step_fraction) {
    // Row by row in parallel (a whole tensor dequantized twice is gigabytes;
    // the served file has 25.6 G values and this runs at every load). A
    // column-reordered tensor is compared through the order it was built
    // with, so the reference stays the file's row.
    const int64_t wgroups = r.width() / r.group;
    const size_t row_bytes = static_cast<size_t>(r.k / (t.ggml_type == 8 ? 32 : 256)) * block_bytes_of(t.ggml_type);
    const uint8_t* data = file.data(t);
    const unsigned threads = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    std::vector<RepackDeviation> part(threads);
    std::vector<double> sq(threads, 0.0);
    std::vector<std::thread> pool;
    for (unsigned th = 0; th < threads; ++th)
        pool.emplace_back([&, th]() {
            std::vector<float> ref(static_cast<size_t>(r.k));
            for (int64_t row = th; row < r.n; row += threads) {
                dequantize_row(t.ggml_type, data + row * row_bytes, static_cast<size_t>(r.k), ref.data());
                for (int64_t c = 0; c < r.k; ++c) {
                    const int64_t dc = r.column_dest_of.empty() ? c : r.column_dest_of[static_cast<size_t>(c)];  // where the file's column c sits
                    const float got = main_value(r, row, dc) - min_of_group(r, row, dc / 32);
                    const double step = std::fabs(static_cast<double>(f16_to_f32(r.scale[row * wgroups + dc / r.group])));
                    const double e = std::fabs(static_cast<double>(got) - static_cast<double>(ref[c])) / (step > 0 ? step : 1.0);
                    part[th].max_steps = std::max(part[th].max_steps, e);
                    sq[th] += e * e;
                    if (e > step_fraction) ++part[th].over;
                    ++part[th].values;
                }
            }
        });
    for (auto& th : pool) th.join();
    RepackDeviation dv;
    double s = 0;
    for (unsigned th = 0; th < threads; ++th) { dv.max_steps = std::max(dv.max_steps, part[th].max_steps); dv.over += part[th].over; dv.values += part[th].values; s += sq[th]; }
    dv.rms_steps = dv.values ? std::sqrt(s / static_cast<double>(dv.values)) : 0.0;
    return dv;
}

void matvec_repacked_host(const RepackedTensor& r, const std::vector<float>& x, std::vector<float>& y) {
    if (static_cast<int64_t>(x.size()) != r.k) throw std::runtime_error("gguf repack: activation length mismatch");
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

}  // namespace lgc::gguf
