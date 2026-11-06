#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Q4_0 x f32 on int8 DPAS for sub-group 8 (Alchemist), which has no 2D block
// loads. The 2D kernel's shape (mul_mm_kq_2d.cl) with the activations packed
// once a graph: [k / 32][token][32 bytes], so that an a operand of 8 tokens x 32
// k is one 256-byte sub-group block read and no local memory is involved. A lane
// is a weight row; its 32 codes of a block decode in registers.
//   sum = xd * sum_block d_block * ir_block
// with ir_block one i8 DPAS, the activations' scale xd per 256, the weight's f16
// scale d per 32, f32 accumulation.

#define QK_K 256
#ifndef WG
#define WG 8                  // sub-groups a work-group, 8 weight rows each
#endif
#ifndef TMM
#define TMM 32                // tokens a sub-group
#endif
#ifndef KSYNC
#define KSYNC 32
#endif
#define NGM (TMM / 8)
#ifdef Q40_RED
#define Q40_I8_OFF 0x7Cu
#else
#define Q40_I8_OFF 0x78u
#endif
#define Q40_S8(t) as_int((((t) & 0x0F0F0F0Fu) + Q40_I8_OFF * 0x01010101u) ^ 0x80808080u)

// one work-group of 256 a token, eight values a work-item (k % 256 == 0): a
// 256-block is 32 work-items, four sub-groups
__attribute__((intel_reqd_sub_group_size(8)))
kernel void kernel_quant_i8_sg8(global const char * src, ulong offs, global char * xq, global float * xd, int k, int tp) {
    const int tok = get_group_id(0), sg = get_sub_group_id();
    global const float * x = (global const float *)(src + offs) + (ulong)tok * k;
    local float sm[32];
    const int st = 8 * get_local_size(0), nr = (k + st - 1) / st;
    for (int r = 0; r < nr; ++r) {
        const int i = 8 * get_local_id(0) + r * st;
        const float8 v = i < k ? vload8(0, x + i) : (float8)(0.f);
        const float8 av = fabs(v);
        const float m8 = fmax(fmax(fmax(av.s0, av.s1), fmax(av.s2, av.s3)), fmax(fmax(av.s4, av.s5), fmax(av.s6, av.s7)));
        sm[sg] = sub_group_reduce_max(m8);
        barrier(CLK_LOCAL_MEM_FENCE);
        const int s0 = sg & ~3;
        const float bm = fmax(fmax(sm[s0], sm[s0 + 1]), fmax(sm[s0 + 2], sm[s0 + 3]));
        barrier(CLK_LOCAL_MEM_FENCE);
        // a block below 1e-30 quantizes to zeros (127 / bm would overflow)
        const float qs = bm / 127.f, iq = bm > 1e-30f ? 127.f / bm : 0.f;
        const int8 qv = convert_int8_rte(v * iq);
        if (i < k) {
            vstore8(convert_char8(qv), 0, xq + (ulong)(i / 32) * tp * 32 + (ulong)tok * 32 + (i & 31));
            if ((i & 255) == 0) xd[(ulong)(i / 256) * tp + tok] = qs;
        }
    }
}

#define AREAD(kb32, g) as_int8(intel_sub_group_block_read8((global const uint *)(xq + (ulong)(kb32) * tp * 32 + (ulong)(t0 + 8 * (g)) * 32)))

__attribute__((intel_reqd_sub_group_size(8)))
kernel void kernel_mul_mm_q4_0_i8_sg8(global const uchar * q, global const half * d, global const char * xq,
                                      global const float * xd, global char * dst, ulong offd, ulong nb1,
                                      int ne00, int ne01, int ne11, int tp) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int r0 = (get_group_id(1) * WG + sg) * 8;
    const int t0 = get_group_id(0) * TMM;
    const int nb = ne00 / QK_K;
    const int n = min(r0 + lane, ne01 - 1);
    const ulong rb = (ulong)n * nb * 8;     // the row's first block
    float8 acc[NGM];
    _Pragma("unroll")
    for (int g = 0; g < NGM; ++g) acc[g] = (float8)(0.f);
    for (int kb = 0; kb < nb; ++kb) {
        if (kb % KSYNC == 0) barrier(CLK_LOCAL_MEM_FENCE);
        const float8 dv = vload_half8(0, d + rb + kb * 8);
        const float da[8] = { dv.s0, dv.s1, dv.s2, dv.s3, dv.s4, dv.s5, dv.s6, dv.s7 };
        float8 xv[NGM], af[NGM];
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) {
            xv[g] = vload8(0, xd + (ulong)kb * tp + t0 + 8 * g);
            af[g] = (float8)(0.f);
        }
        _Pragma("unroll")
        for (int sb = 0; sb < 8; ++sb) {
            const uint4 w = vload4(0, (global const uint *)(q + (rb + kb * 8 + sb) * 16));
            int8 bq;
            bq.s0 = Q40_S8(w.s0); bq.s1 = Q40_S8(w.s1); bq.s2 = Q40_S8(w.s2); bq.s3 = Q40_S8(w.s3);
            bq.s4 = Q40_S8(w.s0 >> 4); bq.s5 = Q40_S8(w.s1 >> 4); bq.s6 = Q40_S8(w.s2 >> 4); bq.s7 = Q40_S8(w.s3 >> 4);
            _Pragma("unroll")
            for (int g = 0; g < NGM; ++g) {
                const int8 ir = intel_sub_group_i8_i8_matrix_mad_k32(AREAD(kb * 8 + sb, g), bq, (int8)(0));
                af[g] = fma(convert_float8(ir), (float8)(da[sb]), af[g]);
            }
        }
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) acc[g] = fma(af[g], xv[g], acc[g]);
    }
    if (r0 + lane < ne01) {
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) {
            const float v[8] = { acc[g].s0, acc[g].s1, acc[g].s2, acc[g].s3, acc[g].s4, acc[g].s5, acc[g].s6, acc[g].s7 };
            _Pragma("unroll")
            for (int m = 0; m < 8; ++m) {
                const int t = t0 + 8 * g + m;
                if (t < ne11) ((global float *)(dst + offd + (ulong)t * nb1))[r0 + lane] = v[m];
            }
        }
    }
}

// Q8_0: the same shape; a lane's 32 codes of a block are the b operand as they
// are. Q8_RED: the dwords negated (the red case of the suite).
#ifdef Q8_RED
#define Q8_B(v) (-(v))
#else
#define Q8_B(v) (v)
#endif
__attribute__((intel_reqd_sub_group_size(8)))
kernel void kernel_mul_mm_q8_0_i8_sg8(global const uchar * q, global const half * d, global const char * xq,
                                      global const float * xd, global char * dst, ulong offd, ulong nb1,
                                      int ne00, int ne01, int ne11, int tp) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int r0 = (get_group_id(1) * WG + sg) * 8;
    const int t0 = get_group_id(0) * TMM;
    const int nb = ne00 / QK_K;
    const int n = min(r0 + lane, ne01 - 1);
    const ulong rb = (ulong)n * nb * 8;
    float8 acc[NGM];
    _Pragma("unroll")
    for (int g = 0; g < NGM; ++g) acc[g] = (float8)(0.f);
    for (int kb = 0; kb < nb; ++kb) {
        if (kb % KSYNC == 0) barrier(CLK_LOCAL_MEM_FENCE);
        const float8 dv = vload_half8(0, d + rb + kb * 8);
        const float da[8] = { dv.s0, dv.s1, dv.s2, dv.s3, dv.s4, dv.s5, dv.s6, dv.s7 };
        float8 xv[NGM], af[NGM];
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) {
            xv[g] = vload8(0, xd + (ulong)kb * tp + t0 + 8 * g);
            af[g] = (float8)(0.f);
        }
        _Pragma("unroll")
        for (int sb = 0; sb < 8; ++sb) {
            const int8 bq = Q8_B(as_int8(vload8(0, (global const uint *)(q + (rb + kb * 8 + sb) * 32))));
            _Pragma("unroll")
            for (int g = 0; g < NGM; ++g) {
                const int8 ir = intel_sub_group_i8_i8_matrix_mad_k32(AREAD(kb * 8 + sb, g), bq, (int8)(0));
                af[g] = fma(convert_float8(ir), (float8)(da[sb]), af[g]);
            }
        }
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) acc[g] = fma(af[g], xv[g], acc[g]);
    }
    if (r0 + lane < ne01) {
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) {
            const float v[8] = { acc[g].s0, acc[g].s1, acc[g].s2, acc[g].s3, acc[g].s4, acc[g].s5, acc[g].s6, acc[g].s7 };
            _Pragma("unroll")
            for (int m = 0; m < 8; ++m) {
                const int t = t0 + 8 * g + m;
                if (t < ne11) ((global float *)(dst + offd + (ulong)t * nb1))[r0 + lane] = v[m];
            }
        }
    }
}
