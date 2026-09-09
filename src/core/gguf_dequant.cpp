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
        default:
            throw std::runtime_error(log::format(
                "gguf: dequantize_row: %s is not implemented (stage 0 covers Q8_0/Q4_K/Q5_K/Q6_K, "
                "FIX D adds Q4_0/Q4_1, and F32/F16/BF16 passthrough)",
                type_name(ggml_type).c_str()));
    }
}

void dequantize_tensor(const GgufFile& file, const TensorInfo& t, std::vector<float>& out) {
    out.resize(t.n_elements);
    dequantize_row(t.ggml_type, file.data(t), t.n_elements, out.data());
}

}  // namespace lgc::gguf
