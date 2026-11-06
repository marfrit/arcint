// 0077: the slot-buffer alias check, a sampled tripwire over whole slices.
//
// Strata's `sycl_queue.hpp:168-206, 259-295` tags every 64 KiB block of an
// allocation of 32 MiB or more with GPU kernels, retries behind a growing
// spacer up to 8 times and then warns and proceeds (`code`, audit item 19).
// This check samples at most one region a slot tensor through the backend's
// blocking set/get, a deviation from the reference named in the record.
//
// The region is one whole slice of the tensor's type: its offset is a multiple
// of `nb[2]` and its size is exactly `nb[2]`, so the OpenCL set path's planar
// conversion (`offset % nb[2] == 0 && size % nb[2] == 0`, ggml-opencl.cpp) is
// satisfied and no per-type transform sees a partial row. One id-carrying
// pattern a tensor, written first for every tensor and read back after all
// writes, so a collision between any two regions shows as a differing
// read-back. Exact: it writes no number the model computes and the expert load
// overwrites the patterns.
//
// Device-free testable through the span arithmetic and the pattern
// (tests/test-expert-cache-alias.cpp).
#ifndef LLAMA_EXPERT_CACHE_ALIAS_H
#define LLAMA_EXPERT_CACHE_ALIAS_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

struct llama_cache_alias_span {
    size_t offset;
    size_t size;
    bool   ok;
};

// one whole `slice` row inside a tensor of `bytes`, at the middle row shifted
// by the tensor's index; not ok for an empty slice or a tensor shorter than one
static inline llama_cache_alias_span llama_cache_alias_region(size_t bytes, size_t slice, size_t index) {
    llama_cache_alias_span s = {0, 0, false};
    if (slice == 0 || bytes < slice) {
        return s;
    }
    const size_t rows = bytes / slice;
    const size_t start = (rows / 2 + index) % rows;
    s.offset = start * slice;
    s.size   = slice;
    s.ok     = true;
    return s;
}

// fill `size` bytes with the region's id (two 64-bit words alternating id and
// its complement, then the same over the tail): any two distinct regions differ
// at byte 0, so an alias's second write is visible when the first reads back
static inline void llama_cache_alias_pattern(uint64_t id, void * block, size_t size) {
    uint8_t * b = (uint8_t *) block;
    const uint64_t w[2] = {id, ~id};
    for (size_t i = 0; i + 8 <= size; i += 8) {
        std::memcpy(b + i, &w[(i / 8) & 1], 8);
    }
    for (size_t i = size & ~(size_t) 7; i < size; ++i) {
        b[i] = (uint8_t) (id >> ((i * 8) & 63));
    }
}

// whether a read-back equals the pattern of `id`, regenerated here: a plan
// holds the id and the span, not hundreds of MiB of expected bytes
static inline bool llama_cache_alias_reads_back(uint64_t id, const void * actual, size_t size) {
    std::vector<uint8_t> expected(size);
    llama_cache_alias_pattern(id, expected.data(), size);
    return std::memcmp(expected.data(), actual, size) == 0;
}

#endif  // LLAMA_EXPERT_CACHE_ALIAS_H
