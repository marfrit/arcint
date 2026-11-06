#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define FWHT_SHUFFLE_XOR(v, m) intel_sub_group_shuffle_xor((v), (m))
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#define FWHT_SHUFFLE_XOR(v, m) sub_group_shuffle_xor((v), (m))
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#else
// a compile fallback only: the indexing below is right at a 16-wide subgroup
// alone, and the host loads these kernels only where it can require one
#define REQD_SUBGROUP_SIZE_16
#endif

// The fast Walsh-Hadamard transform for MUL_MAT nodes that carry
// GGML_HINT_SRC0_IS_HADAMARD (the KV cache's attention rotations): dst row =
// H_n x src1 row with H_n the orthonormal Sylvester Hadamard matrix, so src0
// is never read. The reference is ggml-cuda/fwht.cu (one row per warp,
// shuffle stages inside the warp, register stages above it); here one row per
// 16-wide subgroup: element e of a row sits in lane e % 16, register e / 16.
// The stages commute, so the order of the butterflies is free.
#define FWHT_SG 16

#define FWHT_KERNEL(N)                                                              \
REQD_SUBGROUP_SIZE_16                                                               \
kernel void kernel_fwht_f32_##N(                                                    \
        global const char * src, ulong offs,                                        \
        global char * dst, ulong offd,                                              \
        int n_rows, float scale) {                                                  \
    const int r = get_global_id(0) / FWHT_SG;                                       \
    if (r >= n_rows) {                                                              \
        return;                                                                     \
    }                                                                               \
    const int lane = get_sub_group_local_id();                                      \
    global const float * s = (global const float *)(src + offs) + (long) r * N;     \
    global float *       d = (global float *)(dst + offd) + (long) r * N;           \
    float reg[N / FWHT_SG];                                                         \
    _Pragma("unroll")                                                               \
    for (int i = 0; i < N / FWHT_SG; ++i) {                                         \
        reg[i] = s[i * FWHT_SG + lane] * scale;                                     \
    }                                                                               \
    _Pragma("unroll")                                                               \
    for (int h = 1; h < FWHT_SG; h *= 2) {                                          \
        _Pragma("unroll")                                                           \
        for (int j = 0; j < N / FWHT_SG; ++j) {                                     \
            const float v  = reg[j];                                                \
            const float v2 = FWHT_SHUFFLE_XOR(v, (uint) h);                         \
            reg[j] = (lane & h) == 0 ? v + v2 : v2 - v;                             \
        }                                                                           \
    }                                                                               \
    _Pragma("unroll")                                                               \
    for (int h = FWHT_SG; h < N; h *= 2) {                                          \
        const int step = h / FWHT_SG;                                               \
        _Pragma("unroll")                                                           \
        for (int j = 0; j < N / FWHT_SG; j += 2 * step) {                           \
            _Pragma("unroll")                                                       \
            for (int k = 0; k < step; ++k) {                                        \
                const float x = reg[j + k];                                         \
                const float y = reg[j + k + step];                                  \
                reg[j + k]        = x + y;                                          \
                reg[j + k + step] = x - y;                                          \
            }                                                                       \
        }                                                                           \
    }                                                                               \
    _Pragma("unroll")                                                               \
    for (int i = 0; i < N / FWHT_SG; ++i) {                                         \
        d[i * FWHT_SG + lane] = reg[i];                                             \
    }                                                                               \
}

FWHT_KERNEL(64)
FWHT_KERNEL(128)
FWHT_KERNEL(256)
// no 512: on the A770 its 32 values a lane spill at SIMD16 (offline ocloc,
// dg2-g10) and the retry rolls the loops; no model here rotates at 512.
