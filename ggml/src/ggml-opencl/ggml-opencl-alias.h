// 0077: the VRAM alias check's sampled plan, a deviation from Strata.
//
// Strata's `sycl_queue.hpp:168-206, 259-295` tags EVERY 64 KiB block of an
// allocation of 32 MiB or more with GPU kernels and retries behind a spacer
// (`code`, audit item 19); two virtual blocks sharing one physical page make
// the later write visible in the earlier block. This plan samples at most
// GGML_CL_ALIAS_BLOCKS blocks an allocation, each 64 KiB, spread evenly by
// `j_i = i * n / k` over the block count (unique for n >= k); a smaller
// allocation is one block or none. The scan runs whenever the bank stream ring
// is (re)allocated at a learned order's completion; a relearn after a reset
// scans again, and it is not model load. The sample's
// hit probability is Strata's full-scan class only in part; the record names
// the price and the full-scan alternative. This header holds the plan and the
// pattern, so the device-free cell tests exactly what the backend runs.
#ifndef GGML_OPENCL_ALIAS_H
#define GGML_OPENCL_ALIAS_H

#include <cstddef>
#include <cstdint>

constexpr size_t GGML_CL_ALIAS_BLOCK  = 64u << 10;
constexpr size_t GGML_CL_ALIAS_BLOCKS = 16;

// the 64 KiB-aligned block offsets to check in an allocation of `bytes`
static inline size_t ggml_cl_alias_offsets(size_t bytes, size_t * out, size_t cap) {
    if (bytes < GGML_CL_ALIAS_BLOCK || cap == 0) {
        return 0;
    }
    const size_t n = bytes / GGML_CL_ALIAS_BLOCK;
    const size_t k = n < cap ? n : cap;
    // evenly spread and unique: j_i = i * n / k over the block count
    for (size_t i = 0; i < k; ++i) {
        out[i] = ((i * n) / k) * GGML_CL_ALIAS_BLOCK;
    }
    return k;
}

// the block's pattern: two 64-bit words carrying its id, then repeated
static inline void ggml_cl_alias_pattern(uint64_t id, void * block) {
    uint64_t * w = (uint64_t *) block;
    const size_t n = GGML_CL_ALIAS_BLOCK / sizeof(uint64_t);
    for (size_t i = 0; i < n; ++i) {
        w[i] = (i & 1) ? ~id : id;
    }
}

#endif  // GGML_OPENCL_ALIAS_H
