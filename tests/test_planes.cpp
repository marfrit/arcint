// The planar layout of the OpenCL backend's low-bit types and Q3_K (contrib/llama.cpp patches 0022,
// 0116): GGUF bytes written to or read from a planar tensor go through its base's planes, per slice of
// the base (ne0 ne1 values), whatever the tensor's own shape. The rule is the patch's own header
// (ggml/src/ggml-opencl/ggml-opencl-planes.h), which set_tensor / get_tensor call; here a host vector
// stands for the device buffer, and every plane read and write is bounds-checked against it.
//
// The defect this pins (measured on the cards, first 0116 build): test-backend-ops sets every tensor of
// a case, views too. The permuted view [K, M] of a [K, 1, M] Q3_K weight was converted with its own
// ne0 ne1 and nb[2]: 256 "slices" of the whole matrix, 57,671,680 bytes touched for a 225,280-byte
// write (the b6d0e532 code under AddressSanitizer on the recipe host: heap-buffer-overflow in memcpy),
// and on the cards a SIGSEGV in memcpy and once a copy-engine page fault.
//
// Red cases (each a one-line edit of the header; measured on the recipe host, the test built standalone
// against the header and libggml-base):
//   R3 ggml_cl_lb_base returning t (every geometry from the tensor itself, as the old call sites had it) ->
//      a_permuted_view_writes_through_its_base fails, then a later view case trips the header's range
//      assert (abort)
//   R2 ggml_cl_lb_io skipping the read of a partly covered slice ->
//      a_partial_range_is_read_modify_write, a_reshape_of_a_3d_base_converts_by_the_base_slices and
//      a_partial_range_across_a_slice_boundary_is_exact
//   R5 (0117) Q2_K's dm and scales fields swapped -> the_planes_follow_the_block_structs
//   R4 a field off in ggml_cl_lb_fields (Q3_K's d and hmask swapped, say) -> the_planes_follow_the_block_structs
//      and a_whole_slice_range_takes_the_direct_path
#include "harness.h"

#include <cstdint>
#include <random>
#include <vector>

#ifdef ARCINT_LLAMA_PLANES
#include "ggml.h"
#include "ggml-opencl-planes.h"
#include <cstddef>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"   // the block structs: the expected planes from them, not from the field table
#endif

#ifdef ARCINT_LLAMA
TEST(the_planar_layout_header_is_built) {
#ifndef ARCINT_LLAMA_PLANES
    t::fail(__FILE__, __LINE__, "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0116 (ggml-opencl-planes.h)");
#endif
}
#endif

#ifdef ARCINT_LLAMA_PLANES
namespace {

// a context for tensor metadata only (no data: the "device" is a vector)
struct meta_ctx {
    ggml_context * ctx;
    meta_ctx() {
        ggml_init_params p = { 16 * ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(p);
    }
    ~meta_ctx() { ggml_free(ctx); }
};

// the device: the base's plane bytes, every access bounds-checked
struct fake_device {
    std::vector<uint8_t> mem;
    size_t bad = 0;
    explicit fake_device(size_t n) : mem(n, 0xEE) {}
    void io(const ggml_tensor * t, size_t off, void * data, size_t size, bool to_planes) {
        ggml_cl_lb_io(t, off, data, size, to_planes,
                      [&](size_t o, size_t n, uint8_t * d) {
                          if (o + n > mem.size()) { ++bad; return; }
                          memcpy(d, mem.data() + o, n);
                      },
                      [&](size_t o, size_t n, const uint8_t * s) {
                          if (o + n > mem.size()) { ++bad; return; }
                          memcpy(mem.data() + o, s, n);
                      });
    }
};

std::vector<uint8_t> random_bytes(size_t n, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<uint8_t> v(n);
    for (auto & b : v) b = (uint8_t) g();
    return v;
}

// plane bytes of field (off, size) of block i of slice s, as the kernels address them: a slice of nblk
// blocks holds each field of every block back to back (Q3_K: d | hmask | qs | scales)
bool planes_hold(const std::vector<uint8_t> & planes, const std::vector<uint8_t> & gguf, ggml_type type, int64_t nblk, int64_t n_slices) {
    ggml_cl_lb_field f[5];
    const int    nf  = ggml_cl_lb_fields(type, f);
    const size_t blk = ggml_type_size(type), sb = (size_t) nblk * blk;
    for (int64_t s = 0; s < n_slices; ++s) {
        size_t plane = 0;
        for (int k = 0; k < nf; ++k) {
            for (int64_t i = 0; i < nblk; ++i) {
                if (memcmp(planes.data() + s * sb + plane + i * f[k].size, gguf.data() + s * sb + i * blk + f[k].off, f[k].size) != 0) {
                    return false;
                }
            }
            plane += (size_t) nblk * f[k].size;
        }
    }
    return true;
}

}  // namespace

TEST(a_permuted_view_writes_through_its_base) {
    // test-backend-ops' permuted MUL_MAT case: a [K, 1, M] weight and its view permute(0, 2, 1, 3) = [K, M]
    meta_ctx m;
    ggml_tensor * a  = ggml_new_tensor_4d(m.ctx, GGML_TYPE_Q3_K, 2048, 1, 256, 1);
    ggml_tensor * ap = ggml_permute(m.ctx, a, 0, 2, 1, 3);
    CHECK(ggml_cl_lb_planar(a));
    CHECK(ggml_cl_lb_planar(ap));
    const size_t n = ggml_nbytes(a);
    CHECK_EQ(ggml_nbytes(ap), n);
    fake_device dev(n);
    const std::vector<uint8_t> gguf = random_bytes(n, 1);
    dev.io(ap, 0, const_cast<uint8_t *>(gguf.data()), n, true);   // the case's setup writes the view
    CHECK_EQ(dev.bad, size_t{0});
    // the planes are the base's: 256 slices of 8 blocks
    CHECK(planes_hold(dev.mem, gguf, GGML_TYPE_Q3_K, 8, 256));
    std::vector<uint8_t> back(n);
    dev.io(a, 0, back.data(), n, false);
    CHECK(back == gguf);
    dev.io(ap, 0, back.data(), n, false);
    CHECK(back == gguf);
    CHECK_EQ(dev.bad, size_t{0});
}

TEST(a_partial_range_is_read_modify_write) {
    meta_ctx m;
    ggml_tensor * a = ggml_new_tensor_2d(m.ctx, GGML_TYPE_Q3_K, 512, 6);   // one slice of 12 blocks
    const size_t n = ggml_nbytes(a);
    fake_device dev(n);
    std::vector<uint8_t> gguf = random_bytes(n, 2);
    dev.io(a, 0, gguf.data(), n, true);
    // a range that starts and ends inside blocks
    const std::vector<uint8_t> patch = random_bytes(300, 3);
    const size_t at = n / 2 + 7;
    dev.io(a, at, const_cast<uint8_t *>(patch.data()), patch.size(), true);
    memcpy(gguf.data() + at, patch.data(), patch.size());
    CHECK(planes_hold(dev.mem, gguf, GGML_TYPE_Q3_K, 12, 1));
    std::vector<uint8_t> part(123);
    dev.io(a, 41, part.data(), part.size(), false);
    CHECK(memcmp(part.data(), gguf.data() + 41, part.size()) == 0);
    CHECK_EQ(dev.bad, size_t{0});
}

TEST(a_reshape_of_a_3d_base_converts_by_the_base_slices) {
    // [K, N, E] seen as [K, N E]: its elements, not its slices; also an offset view of rows of the base
    meta_ctx m;
    ggml_tensor * a  = ggml_new_tensor_3d(m.ctx, GGML_TYPE_Q3_K, 512, 64, 3);
    ggml_tensor * a2 = ggml_reshape_2d(m.ctx, a, 512, 64 * 3);
    const size_t n = ggml_nbytes(a);
    fake_device dev(n);
    std::vector<uint8_t> gguf = random_bytes(n, 4);
    dev.io(a2, 0, gguf.data(), n, true);
    CHECK(planes_hold(dev.mem, gguf, GGML_TYPE_Q3_K, 128, 3));
    ggml_tensor * rows = ggml_view_2d(m.ctx, a, 512, 5, a->nb[1], 70 * a->nb[1]);   // rows 70..74: slice 1
    const std::vector<uint8_t> patch = random_bytes(ggml_nbytes(rows), 5);
    dev.io(rows, 0, const_cast<uint8_t *>(patch.data()), patch.size(), true);
    memcpy(gguf.data() + 70 * a->nb[1], patch.data(), patch.size());
    CHECK(planes_hold(dev.mem, gguf, GGML_TYPE_Q3_K, 128, 3));
    std::vector<uint8_t> back(n);
    dev.io(a, 0, back.data(), n, false);
    CHECK(back == gguf);
    CHECK_EQ(dev.bad, size_t{0});
}

// the planes a slice of nblk blocks should hold, from ggml-common.h's struct, not from ggml_cl_lb_fields:
// d first, then the other members in the struct's order, each member of every block back to back
struct member { size_t off, size; };
std::vector<uint8_t> expected_planes(const std::vector<uint8_t> & gguf, size_t blk, int64_t nblk, int64_t n_slices,
                                     const std::vector<member> & order) {
    std::vector<uint8_t> out(gguf.size());
    const size_t sb = (size_t) nblk * blk;
    for (int64_t s = 0; s < n_slices; ++s) {
        size_t plane = s * sb;
        for (const member & m : order) {
            for (int64_t i = 0; i < nblk; ++i) memcpy(out.data() + plane + i * m.size, gguf.data() + s * sb + i * blk + m.off, m.size);
            plane += (size_t) nblk * m.size;
        }
    }
    return out;
}
#define MEMBER(T, f) member{ offsetof(T, f), sizeof(((T *) nullptr)->f) }

TEST(the_planes_follow_the_block_structs) {
    meta_ctx m;
    {
        ggml_tensor * a = ggml_new_tensor_3d(m.ctx, GGML_TYPE_Q3_K, 512, 6, 2);   // 2 slices of 12 blocks
        const size_t n = ggml_nbytes(a);
        fake_device dev(n);
        const std::vector<uint8_t> gguf = random_bytes(n, 6);
        dev.io(a, 0, const_cast<uint8_t *>(gguf.data()), n, true);
        CHECK(dev.mem == expected_planes(gguf, sizeof(block_q3_K), 12, 2,
                                         { MEMBER(block_q3_K, d), MEMBER(block_q3_K, hmask), MEMBER(block_q3_K, qs), MEMBER(block_q3_K, scales) }));
    }
    {
        // a non-Q3_K planar type through the same code: IQ2_XS
        ggml_tensor * a = ggml_new_tensor_3d(m.ctx, GGML_TYPE_IQ2_XS, 512, 6, 2);
        const size_t n = ggml_nbytes(a);
        fake_device dev(n);
        const std::vector<uint8_t> gguf = random_bytes(n, 7);
        dev.io(a, 0, const_cast<uint8_t *>(gguf.data()), n, true);
        CHECK(dev.mem == expected_planes(gguf, sizeof(block_iq2_xs), 12, 2,
                                         { MEMBER(block_iq2_xs, d), MEMBER(block_iq2_xs, qs), MEMBER(block_iq2_xs, scales) }));
        std::vector<uint8_t> back(n);
        dev.io(a, 0, back.data(), n, false);
        CHECK(back == gguf);
        CHECK_EQ(dev.bad, size_t{0});
    }
    {
        // 0117: Q2_K, its d and dmin one 4-byte member (ggml-common.h's union) ahead of scales and qs
        ggml_tensor * a = ggml_new_tensor_3d(m.ctx, GGML_TYPE_Q2_K, 512, 6, 2);
        const size_t n = ggml_nbytes(a);
        fake_device dev(n);
        const std::vector<uint8_t> gguf = random_bytes(n, 11);
        dev.io(a, 0, const_cast<uint8_t *>(gguf.data()), n, true);
        CHECK(dev.mem == expected_planes(gguf, sizeof(block_q2_K), 12, 2,
                                         { MEMBER(block_q2_K, data), MEMBER(block_q2_K, scales), MEMBER(block_q2_K, qs) }));
        std::vector<uint8_t> back(n);
        dev.io(a, 0, back.data(), n, false);
        CHECK(back == gguf);
        CHECK_EQ(dev.bad, size_t{0});
    }
}

TEST(a_partial_range_across_a_slice_boundary_is_exact) {
    // 3 slices; a write from inside slice 0 to inside slice 2, then reads across each boundary
    meta_ctx m;
    for (ggml_type type : { GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XS }) {
        ggml_tensor * a = ggml_new_tensor_3d(m.ctx, type, 512, 4, 3);
        const size_t n = ggml_nbytes(a), sb = a->nb[2];
        fake_device dev(n);
        std::vector<uint8_t> gguf = random_bytes(n, 8);
        dev.io(a, 0, gguf.data(), n, true);
        const size_t at = sb - 37, len = sb + 100;   // slice 0's tail, all of slice 1, slice 2's head
        const std::vector<uint8_t> patch = random_bytes(len, 9);
        dev.io(a, at, const_cast<uint8_t *>(patch.data()), len, true);
        memcpy(gguf.data() + at, patch.data(), len);
        std::vector<uint8_t> back(n);
        dev.io(a, 0, back.data(), n, false);
        CHECK(back == gguf);
        std::vector<uint8_t> part(80);
        dev.io(a, 2 * sb - 40, part.data(), part.size(), false);   // across the boundary of slices 1 and 2
        CHECK(memcmp(part.data(), gguf.data() + 2 * sb - 40, part.size()) == 0);
        CHECK_EQ(dev.bad, size_t{0});
    }
}

TEST(a_whole_slice_range_takes_the_direct_path) {
    // whole slices with host planes convert in place: the read and write callbacks are not called
    meta_ctx m;
    ggml_tensor * a = ggml_new_tensor_3d(m.ctx, GGML_TYPE_Q3_K, 512, 6, 2);
    const size_t n = ggml_nbytes(a), sb = a->nb[2];
    std::vector<uint8_t> planes(n, 0xEE);
    const std::vector<uint8_t> gguf = random_bytes(n, 10);
    int calls = 0;
    ggml_cl_lb_io(a, sb, const_cast<uint8_t *>(gguf.data()) + sb, sb, true,
                  [&](size_t, size_t, uint8_t *) { ++calls; }, [&](size_t, size_t, const uint8_t *) { ++calls; }, planes.data());
    CHECK_EQ(calls, 0);
    std::vector<uint8_t> back(sb);
    ggml_cl_lb_io(a, sb, back.data(), sb, false,
                  [&](size_t, size_t, uint8_t *) { ++calls; }, [&](size_t, size_t, const uint8_t *) { ++calls; }, planes.data());
    CHECK_EQ(calls, 0);
    CHECK(memcmp(back.data(), gguf.data() + sb, sb) == 0);
    // slice 1 holds the planes the block struct says, slice 0 is untouched by a write of slice 1
    const std::vector<uint8_t> want = expected_planes(gguf, sizeof(block_q3_K), 12, 2,
        { MEMBER(block_q3_K, d), MEMBER(block_q3_K, hmask), MEMBER(block_q3_K, qs), MEMBER(block_q3_K, scales) });
    CHECK(memcmp(planes.data() + sb, want.data() + sb, sb) == 0);
    bool untouched = true;
    for (size_t i = 0; i < sb; ++i) untouched = untouched && planes[i] == 0xEE;
    CHECK(untouched);
}

TEST(a_whole_slice_view_is_addressed_by_its_offset) {
    // supports_op's view rule: a view the kernels address right is a whole slice of a planar base
    meta_ctx m;
    ggml_tensor * a = ggml_new_tensor_3d(m.ctx, GGML_TYPE_Q3_K, 512, 64, 3);
    CHECK(ggml_cl_lb_whole_slice(a));
    CHECK(ggml_cl_lb_whole_slice(ggml_view_2d(m.ctx, a, 512, 64, a->nb[1], 2 * a->nb[2])));    // slice 2
    CHECK(!ggml_cl_lb_whole_slice(ggml_view_2d(m.ctx, a, 512, 64, a->nb[1], 5 * a->nb[1])));   // rows 5..68
    CHECK(!ggml_cl_lb_whole_slice(ggml_view_2d(m.ctx, a, 512, 32, a->nb[1], 0)));              // half a slice
    CHECK(!ggml_cl_lb_whole_slice(ggml_reshape_2d(m.ctx, a, 512, 64 * 3)));                   // all slices as one
    CHECK(!ggml_cl_lb_whole_slice(ggml_new_tensor_2d(m.ctx, GGML_TYPE_Q4_K, 512, 64)));        // not planar
}

TEST(an_odd_block_count_is_not_planar_and_the_low_bit_types_keep_their_fields) {
    meta_ctx m;
    CHECK(!ggml_cl_lb_planar(ggml_new_tensor_2d(m.ctx, GGML_TYPE_Q3_K, 256, 129)));
    CHECK(!ggml_cl_lb_planar(ggml_new_tensor_2d(m.ctx, GGML_TYPE_Q3_K, 768, 129)));   // 387 blocks
    CHECK(ggml_cl_lb_planar(ggml_new_tensor_2d(m.ctx, GGML_TYPE_Q3_K, 768, 130)));
    CHECK(!ggml_cl_lb_planar(ggml_new_tensor_2d(m.ctx, GGML_TYPE_Q4_K, 512, 4)));
    // every planar type's fields tile its block exactly
    for (ggml_type t : { GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, GGML_TYPE_IQ3_S, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K }) {
        ggml_cl_lb_field f[5];
        const int nf = ggml_cl_lb_fields(t, f);
        CHECK(nf > 0);
        std::vector<int> cover(ggml_type_size(t), 0);
        for (int k = 0; k < nf; ++k)
            for (int i = 0; i < f[k].size; ++i) cover[f[k].off + i]++;
        bool once = true;
        for (int c : cover) once = once && c == 1;
        CHECK(once);
    }
}
#endif
