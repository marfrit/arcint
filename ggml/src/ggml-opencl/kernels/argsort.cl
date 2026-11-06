#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define SWAP(x, y, T) { T tmp = (x); (x) = (y); (y) = tmp; }

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

kernel void kernel_argsort_f32_i32(
    global float * src0,
    ulong          offset0,
    global int   * dst,
    ulong          offsetd,
    const int      ne00,
    const int      ne00_pad,
    const int      order,
    local int    * dst_row
) {
    // bitonic sort
    int col = get_local_id(0);
    int row = get_group_id(1);

    if (col >= ne00_pad) {
        return;
    }

    src0 = (global char  *)((global char *)src0 + offset0);
    dst  = (global float *)((global char *)dst  + offsetd);

    global float * x_row = src0 + row * ne00;

    // initialize indices
    dst_row[col] = col;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= ne00_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ne00 ||
                        (dst_row[ixj] < ne00 && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj], int);
                    }
                } else {
                    if (dst_row[ixj] >= ne00 ||
                        (dst_row[col] < ne00 && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj], int);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // copy the result to dst without the padding
    if (col < ne00) {
        dst[row * ne00 + col] = dst_row[col];
    }
}

// TOP_K: the indices of the k largest values of each F32 row, unordered (ggml's
// GGML_OP_TOP_K). A work-group per row: four 8-bit radix passes over an
// order-preserving integer key find the k-th largest key T exactly, then the
// keys above T and enough of the keys equal to T are written. Integer keys,
// because these programs build with -cl-finite-math-only and -inf is common.
#define TOPK_WG 256
static inline uint topk_key(float f) {
    const uint u = as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
kernel void kernel_top_k_f32(
        global const char * src0, ulong offset0,
        global       char * dst,  ulong offsetd,
        int ne00, ulong nb01, ulong nb02, ulong nb03,
        int k,    ulong nb1,  ulong nb2,  ulong nb3) {
    local int hist[256];
    local uint s_prefix;
    local int  s_need;
    local int  s_gt, s_eq;
    const int lid = get_local_id(0);
    const int i1 = get_group_id(0), i2 = get_group_id(1), i3 = get_group_id(2);
    global const float * x = (global const float *)(src0 + offset0 + i1*nb01 + i2*nb02 + i3*nb03);
    global int * y = (global int *)(dst + offsetd + i1*nb1 + i2*nb2 + i3*nb3);
    if (lid == 0) { s_prefix = 0; s_need = k; }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const uint mask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int b = lid; b < 256; b += TOPK_WG) hist[b] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint prefix = s_prefix;
        for (int j = lid; j < ne00; j += TOPK_WG) {
            const uint key = topk_key(x[j]);
            if ((key & mask) == prefix) atomic_inc(&hist[(key >> shift) & 0xFF]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            // the bin, from the top, where the k-th largest lies
            int need = s_need, b = 255;
            for (; b > 0; --b) {
                if (hist[b] >= need) break;
                need -= hist[b];
            }
            s_prefix = prefix | ((uint)b << shift);
            s_need = need;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const uint T = s_prefix;
    const int n_eq = s_need;   // keys equal to T to take
    if (lid == 0) { s_gt = 0; s_eq = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int n_gt = k - n_eq;
    for (int j = lid; j < ne00; j += TOPK_WG) {
        const uint key = topk_key(x[j]);
        if (key > T) {
            y[atomic_inc(&s_gt)] = j;
        } else if (key == T) {
            const int e = atomic_inc(&s_eq);
            if (e < n_eq) y[n_gt + e] = j;
        }
    }
}
