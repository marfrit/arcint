// ARC-LAB 0022 / 0116: the planar layout of the low-bit types and Q3_K on the card, device-free (ggml.h
// only), so that arcint's unit tests run the same code as the backend.
//
// The low-bit types IQ2_XXS, IQ2_XS, IQ2_S, Q2_0, IQ3_S (0022, 0030), Q3_K (0116) and Q2_K (0117) live on the card in
// planes, per weight slice (one 2-D matrix, ne0 ne1 values: an expert): each field of every block back to
// back (mul_mv_iq_q8_1.cl, lb_expand), so a lane reads aligned 32-bit words. A slice keeps its size and
// place, so slot / bank copies and gathers stay byte copies; reads and writes of GGUF bytes convert here.
// The planes belong to the base tensor (the one that owns the memory): a view's bytes are the base's at
// the view's offset, whatever the view's own shape and strides.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

struct ggml_cl_lb_field { int off, size; };

static inline int ggml_cl_lb_fields(ggml_type t, ggml_cl_lb_field f[5]) {
    switch (t) {
        case GGML_TYPE_IQ2_XXS: f[0] = { 0, 2 }; f[1] = { 2, 64 }; return 2;
        case GGML_TYPE_IQ2_XS:  f[0] = { 0, 2 }; f[1] = { 2, 64 }; f[2] = { 66, 8 }; return 3;
        case GGML_TYPE_IQ2_S:   f[0] = { 0, 2 }; f[1] = { 2, 32 }; f[2] = { 34, 32 }; f[3] = { 66, 8 }; f[4] = { 74, 8 }; return 5;
        case GGML_TYPE_Q2_0:    f[0] = { 0, 2 }; f[1] = { 2, 16 }; return 2;
        case GGML_TYPE_IQ3_S:   f[0] = { 0, 2 }; f[1] = { 2, 64 }; f[2] = { 66, 8 }; f[3] = { 74, 32 }; f[4] = { 106, 4 }; return 5;
        // 0116: Q3_K's hmask[32] | qs[64] | scales[12] | d as d | hmask | qs | scales (mul_mv_iq_q8_1.cl)
        case GGML_TYPE_Q3_K:    f[0] = { 108, 2 }; f[1] = { 0, 32 }; f[2] = { 32, 64 }; f[3] = { 96, 12 }; return 4;
        // 0117: Q2_K's scales[16] | qs[64] | d | dmin as dm (d, dmin: one half pair) | scales | qs
        case GGML_TYPE_Q2_K:    f[0] = { 80, 4 }; f[1] = { 0, 16 }; f[2] = { 16, 64 }; return 3;
        default:                return 0;
    }
}

// the tensor that owns t's memory (ggml keeps view_src a non-view)
static inline const ggml_tensor * ggml_cl_lb_base(const ggml_tensor * t) {
    return t->view_src != nullptr ? t->view_src : t;
}

// blocks in one slice of t's base
static inline int64_t ggml_cl_lb_slice_blocks(const ggml_tensor * t) {
    const ggml_tensor * b = ggml_cl_lb_base(t);
    return b->ne[0] * b->ne[1] / ggml_blck_size(b->type);
}

// whether t's bytes live in planes: its base is a planar type with an even block count a slice (32-bit
// fields aligned); an odd one stays in GGUF blocks and its ops go to the CPU (supports_op). A view of a
// planar base is planar: its bytes are the base's.
static inline bool ggml_cl_lb_planar(const ggml_tensor * t) {
    ggml_cl_lb_field f[5];
    const ggml_tensor * b = ggml_cl_lb_base(t);
    return ggml_cl_lb_fields(b->type, f) > 0 && ggml_cl_lb_slice_blocks(b) % 2 == 0;
}

// 0116: t's bytes are one whole slice of a planar base, so a kernel that takes its block count from t
// (ne0 ne1) and its planes at t's offset addresses them right: t is the base, or a view with the base's
// ne0 ne1 at a slice-aligned offset (one 2-D slice, or the whole base reshaped with ne0 ne1 kept)
static inline bool ggml_cl_lb_whole_slice(const ggml_tensor * t) {
    const ggml_tensor * b = ggml_cl_lb_base(t);
    if (!ggml_cl_lb_planar(b)) {
        return false;
    }
    const size_t sb = (size_t) ggml_cl_lb_slice_blocks(b) * ggml_type_size(b->type);
    return b == t || (t->ne[0] * t->ne[1] == b->ne[0] * b->ne[1] && t->view_offs % sb == 0);
}

// n_slices slices of t's base, GGUF blocks <-> planes (to_planes)
static inline void ggml_cl_lb_convert(const ggml_tensor * t, const uint8_t * src, uint8_t * dst, int64_t n_slices, bool to_planes) {
    const ggml_tensor * bt = ggml_cl_lb_base(t);
    ggml_cl_lb_field f[5];
    const int     nf   = ggml_cl_lb_fields(bt->type, f);
    const size_t  blk  = ggml_type_size(bt->type);
    const int64_t nblk = ggml_cl_lb_slice_blocks(bt);
    GGML_ASSERT(nf > 0 && nblk % 2 == 0);
    const size_t  sb   = (size_t) nblk * blk;
    for (int64_t sl = 0; sl < n_slices; ++sl) {
        const uint8_t * a = src + sl * sb;
        uint8_t       * b = dst + sl * sb;
        size_t plane = 0;
        for (int k = 0; k < nf; ++k) {
            for (int64_t i = 0; i < nblk; ++i) {
                if (to_planes) memcpy(b + plane + i * f[k].size, a + i * blk + f[k].off, f[k].size);
                else           memcpy(b + i * blk + f[k].off, a + plane + i * f[k].size, f[k].size);
            }
            plane += (size_t) nblk * f[k].size;
        }
    }
}

// 0116: GGUF bytes [offset, offset + size) of t (t's view offset added: a view's range is its base's) to
// (to_planes) or from the planes of t's base. A range of whole slices of the base converts straight
// between `data` and the planes: through `direct` when the planes are host memory (a USM buffer; no copy
// at all), else through one uninitialized staging buffer and one read or write (the cost before 0116).
// A range that covers a slice in part reads the slices it touches, patches them and writes them back
// (read-modify-write), so any range is exact. read(boff, n, dst) and write(boff, n, src) move n plane
// bytes at byte boff of the base. Before, the callers converted with the view's own ne0 ne1 and nb[2]
// and needed whole slices: the permuted view [2048, 256] of a Q3_K [2048, 1, 256] weight, which
// test-backend-ops sets too, made 256 "slices" of the whole matrix, 57,671,680 bytes touched for a
// 225,280-byte write (measured-here, AddressSanitizer).
template <class R, class W>
static inline void ggml_cl_lb_io(const ggml_tensor * t, size_t offset, void * data, size_t size, bool to_planes, R && read, W && write,
                                 uint8_t * direct = nullptr) {
    if (size == 0) {
        return;
    }
    const ggml_tensor * b  = ggml_cl_lb_base(t);
    const size_t        sb = (size_t) ggml_cl_lb_slice_blocks(b) * ggml_type_size(b->type);
    const size_t        lo = t->view_offs + offset;
    GGML_ASSERT(sb > 0 && lo + size <= ggml_nbytes(b));
    const size_t s0 = lo / sb, s1 = (lo + size + sb - 1) / sb, n = s1 - s0;
    if (lo % sb == 0 && size % sb == 0) {
        // whole slices: no read-modify-write
        if (direct != nullptr) {
            if (to_planes) ggml_cl_lb_convert(b, (const uint8_t *) data, direct + lo, (int64_t) n, true);
            else           ggml_cl_lb_convert(b, direct + lo, (uint8_t *) data, (int64_t) n, false);
            return;
        }
        std::unique_ptr<uint8_t[]> planes(new uint8_t[size]);
        if (to_planes) {
            ggml_cl_lb_convert(b, (const uint8_t *) data, planes.get(), (int64_t) n, true);
            write(lo, size, (const uint8_t *) planes.get());
        } else {
            read(lo, size, planes.get());
            ggml_cl_lb_convert(b, planes.get(), (uint8_t *) data, (int64_t) n, false);
        }
        return;
    }
    std::unique_ptr<uint8_t[]> blocks(new uint8_t[n * sb]), planes(new uint8_t[n * sb]);
    read(s0 * sb, n * sb, planes.get());
    ggml_cl_lb_convert(b, planes.get(), blocks.get(), (int64_t) n, false);
    if (!to_planes) {
        memcpy(data, blocks.get() + (lo - s0 * sb), size);
        return;
    }
    memcpy(blocks.get() + (lo - s0 * sb), data, size);
    ggml_cl_lb_convert(b, blocks.get(), planes.get(), (int64_t) n, true);
    write(s0 * sb, n * sb, (const uint8_t *) planes.get());
}
