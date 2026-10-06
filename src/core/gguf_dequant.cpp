#include "core/gguf_dequant.h"

#include <cstring>
#include <stdexcept>

#include "util/log.h"

// Host reference dequantizers (docs/design-gguf-native.md §3.4). Formulas
// transcribed from ggml's ggml-quants.c: dequantize_row_q8_0,
// dequantize_row_q4_K (get_scale_min_k4), dequantize_row_q5_K,
// dequantize_row_q6_K. Not the fast path -- see the design note's stage 1+
// kernels for that; this exists to be obviously correct.
namespace lgc::gguf {

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t       mant = h & 0x3FFu;
    uint32_t       bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // signed zero
        } else {
            // Subnormal half: normalize by shifting the mantissa left until
            // its implicit leading bit lands at bit 10, tracking the shift
            // to adjust the exponent (unbiased = -14 - shift; see the
            // derivation this test file's neighbors expect in review).
            int shift = 0;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                ++shift;
            }
            mant &= 0x03FFu;
            const int32_t unbiased = -14 - shift;
            bits = sign | (static_cast<uint32_t>(unbiased + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf (mant==0) or nan
    } else {
        const int32_t unbiased = static_cast<int32_t>(exp) - 15;
        bits = sign | (static_cast<uint32_t>(unbiased + 127) << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

float bf16_to_f32(uint16_t h) {
    // bf16 is the top 16 bits of a float32 (truncated, not rounded).
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float           f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

namespace {

uint16_t read_u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

// ggml-quants.c's get_scale_min_k4: unpacks the 8 6-bit (scale, min) pairs
// packed into Q4_K/Q5_K's 12-byte `scales` field. j in [0, 8).
void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// FIX D (docs/design-qwen-flash-next.md): the n-gram embedding table
// (`per_layer_token_embd`) ships in Q4_0/Q4_1 -- 32-element blocks, but NOT
// K-quant: no per-superblock scale/min packing, just one f16 scale (Q4_0) or
// one f16 scale plus one f16 min (Q4_1) per 32-element block of 4-bit
// nibbles. Transcribed from ggml-quants.c's dequantize_row_q4_0 /
// dequantize_row_q4_1.
void dequantize_row_q4_0(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kBlock = 32;
    for (size_t b = 0; b * kBlock < n_elements; ++b) {
        const uint8_t* block = p + b * 18;  // 2B d + 16B packed nibbles
        const float    d     = f16_to_f32(read_u16(block));
        const uint8_t* qs    = block + 2;
        float*         y     = out + b * kBlock;
        for (size_t l = 0; l < 16; ++l) {
            const int x0 = (qs[l] & 0x0F) - 8;
            const int x1 = (qs[l] >> 4) - 8;
            y[l]      = static_cast<float>(x0) * d;
            y[l + 16] = static_cast<float>(x1) * d;
        }
    }
}

void dequantize_row_q4_1(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kBlock = 32;
    for (size_t b = 0; b * kBlock < n_elements; ++b) {
        const uint8_t* block = p + b * 20;  // 2B d + 2B m + 16B packed nibbles
        const float    d     = f16_to_f32(read_u16(block));
        const float    m     = f16_to_f32(read_u16(block + 2));
        const uint8_t* qs    = block + 4;
        float*         y     = out + b * kBlock;
        for (size_t l = 0; l < 16; ++l) {
            const int x0 = qs[l] & 0x0F;  // no -8 offset: Q4_1 is (x*d + m), not centered
            const int x1 = qs[l] >> 4;
            y[l]      = static_cast<float>(x0) * d + m;
            y[l + 16] = static_cast<float>(x1) * d + m;
        }
    }
}

void dequantize_row_q8_0(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kBlock = 32;
    for (size_t b = 0; b * kBlock < n_elements; ++b) {
        const uint8_t* block = p + b * 34;
        const float    d     = f16_to_f32(read_u16(block));
        const auto*    qs    = reinterpret_cast<const int8_t*>(block + 2);
        float*         y     = out + b * kBlock;
        for (size_t l = 0; l < kBlock; ++l) y[l] = d * static_cast<float>(qs[l]);
    }
}

void dequantize_row_q4_k(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kSuper = 256;
    for (size_t b = 0; b * kSuper < n_elements; ++b) {
        const uint8_t* block  = p + b * 144;
        const float    d      = f16_to_f32(read_u16(block));
        const float    dmin   = f16_to_f32(read_u16(block + 2));
        const uint8_t* scales = block + 4;
        const uint8_t* qs     = block + 16;
        float*         y      = out + b * kSuper;

        for (int j = 0; j < 4; ++j) {
            const uint8_t* q32 = qs + 32 * j;
            uint8_t        sc, m;
            get_scale_min_k4(2 * j, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            for (int l = 0; l < 32; ++l) y[l] = d1 * static_cast<float>(q32[l] & 0xF) - m1;

            get_scale_min_k4(2 * j + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) y[32 + l] = d2 * static_cast<float>(q32[l] >> 4) - m2;

            y += 64;
        }
    }
}

void dequantize_row_q5_k(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kSuper = 256;
    for (size_t b = 0; b * kSuper < n_elements; ++b) {
        const uint8_t* block  = p + b * 176;
        const float    d      = f16_to_f32(read_u16(block));
        const float    dmin   = f16_to_f32(read_u16(block + 2));
        const uint8_t* scales = block + 4;
        const uint8_t* qh     = block + 16;
        const uint8_t* qs     = block + 48;
        float*         y      = out + b * kSuper;

        for (int j = 0; j < 4; ++j) {
            const uint8_t* q32 = qs + 32 * j;
            const uint8_t  u1  = static_cast<uint8_t>(1u << (2 * j));
            const uint8_t  u2  = static_cast<uint8_t>(2u << (2 * j));
            uint8_t        sc, m;

            get_scale_min_k4(2 * j, scales, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                const int q = (q32[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0);
                y[l]        = d1 * static_cast<float>(q) - m1;
            }

            get_scale_min_k4(2 * j + 1, scales, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) {
                const int q = (q32[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                y[32 + l]   = d2 * static_cast<float>(q) - m2;
            }

            y += 64;
        }
    }
}

void dequantize_row_q6_k(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kSuper = 256;
    for (size_t b = 0; b * kSuper < n_elements; ++b) {
        const uint8_t* block  = p + b * 210;
        const uint8_t* ql     = block;
        const uint8_t* qh     = block + 128;
        const auto*    scales = reinterpret_cast<const int8_t*>(block + 192);
        const float    d      = f16_to_f32(read_u16(block + 208));
        float*         y      = out + b * kSuper;

        for (int n = 0; n < 2; ++n) {
            const uint8_t* ql_h = ql + 64 * n;
            const uint8_t* qh_h = qh + 32 * n;
            const int8_t*  sc_h = scales + 8 * n;
            float*         y_h  = y + 128 * n;

            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql_h[l] & 0xF) | (((qh_h[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32;

                y_h[l]      = d * static_cast<float>(sc_h[is + 0]) * static_cast<float>(q1);
                y_h[l + 32] = d * static_cast<float>(sc_h[is + 2]) * static_cast<float>(q2);
                y_h[l + 64] = d * static_cast<float>(sc_h[is + 4]) * static_cast<float>(q3);
                y_h[l + 96] = d * static_cast<float>(sc_h[is + 6]) * static_cast<float>(q4);
            }
        }
    }
}

// --------------------------------------------------------------- the I-quants
//
// IQ4_NL and IQ3_XXS: the formats the Flash-Next checkpoint ships its experts
// in (down: IQ4_NL, gate/up: IQ3_XXS). Re-quantising them into the plugin's
// u4 grouped-affine codes costs 0.10-0.13 relative RMS per expert tensor and
// 0.73 nats at depth 48 (DESIGN 7.0.2bz), so the experts are computed from
// these blocks directly; this is the host decoder -- the per-expert kernel's
// reference emulation and the host tier's decode. Transcribed from
// ggml-quants.c dequantize_row_iq4_nl / dequantize_row_iq3_xxs and the
// tables of ggml-common.h (llama.cpp 56b9eb28, MIT).

// IQ4_NL: per 32 values one f16 scale and 16 bytes of nibbles; the nibble
// indexes a fixed 16-entry signed table (a non-uniform grid, which is why an
// affine (q - zp) * s cannot carry it).
static const int8_t kIq4NlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1,    13,   25,  38,  53,  69,  89,  113};

void dequantize_row_iq4_nl(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kBlock = 32;
    for (size_t b = 0; b * kBlock < n_elements; ++b) {
        const uint8_t* block = p + b * 18;  // 2B d + 16B nibbles
        const float    d     = f16_to_f32(read_u16(block));
        const uint8_t* qs    = block + 2;
        float*         y     = out + b * kBlock;
        for (size_t j = 0; j < 16; ++j) {
            y[j]      = d * static_cast<float>(kIq4NlValues[qs[j] & 0xF]);
            y[j + 16] = d * static_cast<float>(kIq4NlValues[qs[j] >> 4]);
        }
    }
}

// IQ4_XS: per 256 values one f16 d, eight 6-bit sub-block scales (the low
// nibble in scales_l[ib/2], the two high bits at 2*ib of scales_h), then 128
// bytes of nibbles read exactly as IQ4_NL's: value = d * (ls - 32) * table[q].
// Layer 2's gate/up in the shipped checkpoint (dequantize_row_iq4_xs).
void dequantize_row_iq4_xs(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kBlock = 256;
    for (size_t b = 0; b * kBlock < n_elements; ++b) {
        const uint8_t* block    = p + b * 136;  // 2B d + 2B scales_h + 4B scales_l + 128B nibbles
        const float    d        = f16_to_f32(read_u16(block));
        const uint16_t scales_h = read_u16(block + 2);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs       = block + 8;
        float*         y        = out + b * kBlock;
        for (int ib = 0; ib < 8; ++ib) {
            const int   ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * static_cast<float>(ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j]      = dl * static_cast<float>(kIq4NlValues[qs[j] & 0xF]);
                y[j + 16] = dl * static_cast<float>(kIq4NlValues[qs[j] >> 4]);
            }
            y += 32;
            qs += 16;
        }
    }
}

// IQ3_XXS: per 256 values one f16 scale d, then 64 bytes of 8-bit grid
// indices (each selects 8 values: two grid entries of 4 bytes) and 8 x u32
// of scales-and-signs, one per 32-value sub-block -- bits 0..27 four 7-bit
// sign-mask indices (one per 8 values), bits 28..31 the sub-block scale s:
// value = d * (0.5 + s) * 0.5 * grid_byte * sign.
static const uint8_t kIq2xsMask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
static const uint8_t kIq2xsSigns[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
static const uint32_t kIq3xxsGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

void dequantize_row_iq3_xxs(const uint8_t* p, size_t n_elements, float* out) {
    constexpr size_t kBlock = 256;
    for (size_t b = 0; b * kBlock < n_elements; ++b) {
        const uint8_t* block = p + b * 98;  // 2B d + 64B grid indices + 32B scales-and-signs
        const float    d     = f16_to_f32(read_u16(block));
        const uint8_t* qs    = block + 2;
        const uint8_t* ss    = qs + 64;
        float*         y     = out + b * kBlock;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, ss + 4 * ib32, sizeof aux32);
            const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = kIq2xsSigns[(aux32 >> (7 * l)) & 127];
                uint8_t g1[4], g2[4];
                std::memcpy(g1, &kIq3xxsGrid[qs[2 * l + 0]], 4);
                std::memcpy(g2, &kIq3xxsGrid[qs[2 * l + 1]], 4);
                for (int j = 0; j < 4; ++j) {
                    y[j]     = db * static_cast<float>(g1[j]) * ((signs & kIq2xsMask[j]) ? -1.f : 1.f);
                    y[j + 4] = db * static_cast<float>(g2[j]) * ((signs & kIq2xsMask[j + 4]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

}  // namespace

void dequantize_row(int32_t ggml_type, const uint8_t* block_bytes, size_t n_elements, float* out) {
    switch (static_cast<GgmlType>(ggml_type)) {
        case GgmlType::F32:
            std::memcpy(out, block_bytes, n_elements * sizeof(float));
            return;
        case GgmlType::F16:
            for (size_t i = 0; i < n_elements; ++i) out[i] = f16_to_f32(read_u16(block_bytes + 2 * i));
            return;
        case GgmlType::BF16:
            for (size_t i = 0; i < n_elements; ++i) out[i] = bf16_to_f32(read_u16(block_bytes + 2 * i));
            return;
        case GgmlType::Q8_0: dequantize_row_q8_0(block_bytes, n_elements, out); return;
        case GgmlType::Q4_0: dequantize_row_q4_0(block_bytes, n_elements, out); return;
        case GgmlType::Q4_1: dequantize_row_q4_1(block_bytes, n_elements, out); return;
        case GgmlType::Q4_K: dequantize_row_q4_k(block_bytes, n_elements, out); return;
        case GgmlType::Q5_K: dequantize_row_q5_k(block_bytes, n_elements, out); return;
        case GgmlType::Q6_K: dequantize_row_q6_k(block_bytes, n_elements, out); return;
        case GgmlType::IQ4_NL: dequantize_row_iq4_nl(block_bytes, n_elements, out); return;
        case GgmlType::IQ3_XXS: dequantize_row_iq3_xxs(block_bytes, n_elements, out); return;
        case GgmlType::IQ4_XS: dequantize_row_iq4_xs(block_bytes, n_elements, out); return;
        default:
            throw std::runtime_error(log::format(
                "gguf: dequantize_row: %s is not implemented (stage 0 covers Q8_0/Q4_K/Q5_K/Q6_K, "
                "FIX D adds Q4_0/Q4_1, FIX F adds IQ4_NL/IQ3_XXS/IQ4_XS, and F32/F16/BF16 passthrough)",
                type_name(ggml_type).c_str()));
    }
}

void dequantize_tensor(const GgufFile& file, const TensorInfo& t, std::vector<float>& out) {
    out.resize(t.n_elements);
    dequantize_row(t.ggml_type, file.data(t), t.n_elements, out.data());
}

}  // namespace lgc::gguf
