#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/gguf.h"

// Host reference dequantizers (docs/design-gguf-native.md §3.4, stage 0
// gate): ggml's own `dequantize_row_*` formulas, transcribed from
// ggml-quants.c, byte for byte. These exist to check a fixture and, later,
// real tensors -- not to be fast; the kernels in stage 1+ are the fast path.
namespace lgc::gguf {

// IEEE-754 binary16 -> binary32, exact (subnormals, inf, nan included).
float f16_to_f32(uint16_t h);

// bf16 -> f32: bf16 is the top 16 bits of f32, so this is a zero-extend.
float bf16_to_f32(uint16_t h);

// Dequantizes `n_elements` values starting at `block_bytes` into `out`.
// `n_elements` must be a multiple of the type's block size (true for any
// whole GGUF tensor row, and for a whole tensor since rows are packed
// contiguously with no inter-row padding). Supports F32, F16, BF16 (plain
// copy/convert) and Q8_0, Q4_K, Q5_K, Q6_K (ggml's block formats). Throws
// std::runtime_error for any other type -- stage 0 implements exactly the
// four K-quant types the design note's four decoders start with, plus
// pass-through for the file's own float types.
void dequantize_row(int32_t ggml_type, const uint8_t* block_bytes, size_t n_elements, float* out);

// Dequantizes a whole tensor's bytes (out.resize()'d to t.n_elements).
void dequantize_tensor(const GgufFile& file, const TensorInfo& t, std::vector<float>& out);

}  // namespace lgc::gguf
