#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

// K-quant x f16 matrix-matrix products on Xe2 with both operands read by
// 2D block loads (cl_intel_subgroup_2d_block_io), the shape of arcint's
// OpenVINO kernel (marfrit-openvino patch 0029, fully_connected_gpu_kquant.cl),
// moved onto ggml-opencl's flat planes:
//   - a lane is a weight row: the sub-group's sixteen rows are read by
//     transposed 32-bit block reads, one message bringing 32 bytes of each
//     row (lane l its own row's eight dwords), decoded in registers to the
//     fp16 DPAS b operand (lane = output column);
//   - the tokens are the a operand: blocks of 32 (or 8) activation rows x
//     16 halves read straight from global memory in the matrix unit's
//     layout, shared by the work-group's sub-groups through L1.
// The weights are decoded once per TM tokens, packed: two halves an
// instruction (hscale). f32 accumulation.
//
// Q4_K also has an int8 kernel at the end of this file (one i8 DPAS per 32
// weights, twice fp16's rate), the default on the B60.
//
// The super-block loop of mul_mm_kq_f16.cl (dequantization into local
// memory, the activations gathered per lane) kept its threads waiting on
// loads 52 % of the time on a B60 (unitrace VectorEngineStalls,
// measured-here); 0029 reached 58 TFLOPS on the same card and shapes.
//
// Rows past the matrix and tokens past the last read as zeros (the block
// loads clamp to the surface) and their results are dropped.

#define QK_K 256
#ifndef TM
#define TM 64                 // tokens a work-group
#endif
#ifndef WG
#define WG 16                 // sub-groups a work-group, 16 weight rows each
#endif
#ifndef AT
#define AT 32                 // activation rows a block read: 32 or 8
#endif
#define NG (TM / 8)           // 8-token groups
#define DPAS(a, b, c) intel_sub_group_f16_f16_matrix_mad_k16(a, b, c)

// a plane of nb blocks of `bb` bytes per row, ne01 rows: dwords dw .. dw + 7
// of block kb of the sub-group's sixteen rows, lane l row r0 + l
#define WREAD(plane, bb, dst, kb, dw) \
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((const global void *)(plane), nb * (bb), ne01, nb * (bb), \
        (int2)((int)((kb) * ((bb) / 4) + (dw)), r0), (dst))

// AT activation rows from row `row`, halves k .. k + 15
#if AT == 32
#define AREAD(dst, k, row) intel_sub_group_2d_block_read_16b_32r16x1c((const global void *)xh, ne00 * 2, ne11, ne00 * 2, (int2)((int)(k), (int)(row)), (dst))
#else
#define AREAD(dst, k, row) intel_sub_group_2d_block_read_16b_8r16x1c((const global void *)xh, ne00 * 2, ne11, ne00 * 2, (int2)((int)(k), (int)(row)), (dst))
#endif

// get_scale_min_k4 of ggml: the scale and min of sub-block j from the 12
// scale bytes, read as three dwords
static inline void scale_min_k4(int j, uint3 w, float * sc, float * m) {
    if (j < 4) {
        *sc = (float)((w.s0 >> (8 * j)) & 63);
        *m  = (float)((w.s1 >> (8 * j)) & 63);
    } else {
        const int jj = j - 4;
        const uint q8 = (w.s2 >> (8 * jj)) & 0xFF;   // scales[j + 4]
        const uint lo = (w.s0 >> (8 * jj)) & 0xFF;   // scales[j - 4]
        const uint hi = (w.s1 >> (8 * jj)) & 0xFF;   // scales[j]
        *sc = (float)((q8 & 0xF) | ((lo >> 6) << 4));
        *m  = (float)((q8 >> 4) | ((hi >> 6) << 4));
    }
}

// The activations are stored with k permuted inside every group of four
// (kernel_cvt_f32_f16_p4): slot s holds k = P4(s), P4 swapping 1 and 2. A
// dword of packed codes then yields a b-operand dword by one mask: bytes 0
// and 2 are the pair (k, k + 2) at slots (k, k + 1).
#define P4(s) (((s) & ~3) | (((s) & 1) << 1) | (((s) >> 1) & 1))

// Four codes a byte apart in w (bits 0..3, 8..11, 16..19, 24..27) as two
// b-operand dwords of halves 1024 + q: slots (k, k + 1) = codes (0, 2), (1, 3)
#define MAGIC 0x64006400u
#define PAIR0(w) (((w) & 0x000F000Fu) | MAGIC)
#define PAIR1(w) ((((w) >> 8) & 0x000F000Fu) | MAGIC)

// (p - 1024) * dl + nml on the 32 halves of a sub-group's dwords p, in place
// of what IGC makes of half2 per lane (two planar SIMD16 halves: a move
// apart and a move together around every operation). dl2 and nml2 hold
// each lane's scale twice. NoMask: a SIMD16 thread's dispatch mask covers
// 16 of the 32 channels; the kernels run full sub-groups in uniform flow.
static inline uint hscale(uint p, uint dl2, uint nml2) {
    uint r;
    __asm__("{\n"
            ".decl PH v_type=G type=hf num_elts=32 align=GRF alias=<%1, 0>\n"
            ".decl DH v_type=G type=hf num_elts=32 align=GRF alias=<%2, 0>\n"
            ".decl NH v_type=G type=hf num_elts=32 align=GRF alias=<%3, 0>\n"
            ".decl RH v_type=G type=hf num_elts=32 align=GRF alias=<%0, 0>\n"
            "add (M1_NM, 32) RH(0,0)<1> PH(0,0)<1;1,0> 0xE400:hf\n"
            "mad (M1_NM, 32) RH(0,0)<1> RH(0,0)<1;1,0> DH(0,0)<1;1,0> NH(0,0)<1;1,0>\n"
            "}\n" : "=rw"(r) : "rw"(p), "rw"(dl2), "rw"(nml2));
    return r;
}

// f32 -> f16 as kernel_cvt_f32_f16 (mul_mm_kq_f16.cl), k in P4 order and
// eight values a work-item (k % 8 == 0): 16-byte stores in place of 2-byte
// ones
kernel void kernel_cvt_f32_f16_p4(global const char * src, ulong offs, global half * dst, global float * scale, int k) {
    const int col = get_group_id(0);
    global const float * x = (global const float *)(src + offs) + (ulong)col * k;
    global half * y = dst + (ulong)col * k;
    float amax = 0.f;
    for (int i = 8 * get_local_id(0); i < k; i += 8 * get_local_size(0)) {
        const float8 v = fabs(vload8(0, x + i));
        amax = fmax(amax, fmax(fmax(fmax(v.s0, v.s1), fmax(v.s2, v.s3)), fmax(fmax(v.s4, v.s5), fmax(v.s6, v.s7))));
    }
    amax = work_group_reduce_max(amax);
    const float sc = amax > 16384.f ? ldexp(1.f, (int)ceil(log2(amax / 16384.f))) : 1.f;
    const float isc = 1.f / sc;
    for (int i = 8 * get_local_id(0); i < k; i += 8 * get_local_size(0)) {
        const float8 v = vload8(0, x + i) * isc;
        vstore_half8_rte(v.s02134657, 0, y + i);
    }
    if (get_local_id(0) == 0) scale[col] = sc;
}

// the multiply of one 32-k sub-block at k0 for the work-group's TM tokens
// b0 against every token group, then b1: a group's two dependent DPAS are
// TM / 8 apart, not adjacent (adjacent ones made IGC drain the matrix unit
// between them, 22 completion waits a super-block in the ISA)
#define MUL32(k0)                                                                           \
    _Pragma("unroll")                                                                       \
    for (int gb = 0; gb < TM / AT; ++gb) {                                                  \
        ushort a0[AT];                                                                      \
        AREAD(a0, (k0), t0 + gb * AT);                                                      \
        _Pragma("unroll")                                                                   \
        for (int u = 0; u < AT / 8; ++u)                                                    \
            acc[gb * (AT / 8) + u] = DPAS(as_short8(vload8(0, a0 + 8 * u)), b0, acc[gb * (AT / 8) + u]); \
    }                                                                                       \
    _Pragma("unroll")                                                                       \
    for (int gb = 0; gb < TM / AT; ++gb) {                                                  \
        ushort a1[AT];                                                                      \
        AREAD(a1, (k0) + 16, t0 + gb * AT);                                                 \
        _Pragma("unroll")                                                                   \
        for (int u = 0; u < AT / 8; ++u)                                                    \
            acc[gb * (AT / 8) + u] = DPAS(as_short8(vload8(0, a1 + 8 * u)), b1, acc[gb * (AT / 8) + u]); \
    }

// a work-group barrier every KSYNC blocks. The sub-groups share the
// activation tile through L1 and drift apart over a long k loop without
// one: at k = 17,408 the B60's load-store cache hit 86.5 % of reads against
// 92 % at 5,120, and a barrier every 16 blocks took the Q6_K down
// projection from 1,960 to 1,431 us (measured-here)
#ifndef KSYNC
#define KSYNC 16
#endif
#define KB_SYNC(kb) if ((kb) % KSYNC == 0) barrier(CLK_LOCAL_MEM_FENCE);
// the int8 kernel's interval: 32 (a genetic search over the tiles and the
// intervals, scored by the dense prefill: +0.5 % against 16, measured-here)
#ifndef KSYNC_I8
#define KSYNC_I8 32
#endif

#define PROLOGUE                                                                            \
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();                     \
    /* token tiles the fast grid dimension: a weight panel is read by its */                \
    /* token tiles in turn while it is in L2, not once per tile from memory */              \
    const int r0 = (get_group_id(1) * WG + sg) * 16;                                        \
    const int t0 = get_group_id(0) * TM;                                                    \
    const int nb = ne00 / QK_K;                                                             \
    const int n  = min(r0 + lane, ne01 - 1);                                                \
    float8 acc[NG];                                                                         \
    _Pragma("unroll")                                                                       \
    for (int g = 0; g < NG; ++g) acc[g] = (float8)(0.f);

// token t0 + 8 g + m of output row r0 + lane at dst + t * nb1
#define EPILOGUE                                                                            \
    if (r0 + lane < ne01) {                                                                 \
        _Pragma("unroll")                                                                   \
        for (int g = 0; g < NG; ++g) {                                                      \
            const float v[8] = { acc[g].s0, acc[g].s1, acc[g].s2, acc[g].s3,                \
                                 acc[g].s4, acc[g].s5, acc[g].s6, acc[g].s7 };              \
            _Pragma("unroll")                                                               \
            for (int m = 0; m < 8; ++m) {                                                   \
                const int t = t0 + 8 * g + m;                                               \
                if (t < ne11) ((global float *)(dst + offd + (ulong)t * nb1))[r0 + lane] = v[m] * xscale[t]; \
            }                                                                               \
        }                                                                                   \
    }

#define ARGS global const half * xh, global const float * xscale, global char * dst, ulong offd, ulong nb1, \
             int ne00, int ne01, int ne11

// Q4_K: a mask makes two halves 1024 + q, one half subtraction (exact)
// and one half fma scale them
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_kq2d_q4_K(global const uchar * q, global const uchar * s, global const half * d,
                                    global const half * dm, ARGS) {
    PROLOGUE
    for (int kb = 0; kb < nb; ++kb) {
        KB_SYNC(kb)
        const ulong b = (ulong)n * nb + kb;
        const uint3 sw = vload3(0, (global const uint *)(s + b * 12));
        const float D = vload_half(0, d + b), DM = vload_half(0, dm + b);
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) {
            uint qw[8];
            WREAD(q, 128, qw, kb, 8 * j);
            _Pragma("unroll")
            for (int hi = 0; hi < 2; ++hi) {
                const int sb = 2 * j + hi;
                float sc, mn;
                scale_min_k4(sb, sw, &sc, &mn);
                const uint dl2 = (uint)as_ushort((half)(D * sc)) * 0x10001u;
                const uint nml2 = (uint)as_ushort((half)(-DM * mn)) * 0x10001u;
                uint pw[16];
                _Pragma("unroll")
                for (int i = 0; i < 8; ++i) {
                    const uint w = hi ? qw[i] >> 4 : qw[i];
                    pw[2 * i] = hscale(PAIR0(w), dl2, nml2);
                    pw[2 * i + 1] = hscale(PAIR1(w), dl2, nml2);
                }
                const int8 b0 = as_int8(vload8(0, pw)), b1 = as_int8(vload8(0, pw + 8));
                MUL32(kb * QK_K + 32 * sb)
            }
        }
    }
    EPILOGUE
}

// Q5_K: Q4_K's decode with the fifth bit (bit sb of the high-bit bytes)
// put at bit 4 of each half
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_kq2d_q5_K(global const uchar * q, global const uchar * qh, global const uchar * s,
                                    global const half * d, global const half * dm, ARGS) {
    PROLOGUE
    for (int kb = 0; kb < nb; ++kb) {
        KB_SYNC(kb)
        const ulong b = (ulong)n * nb + kb;
        const uint3 sw = vload3(0, (global const uint *)(s + b * 12));
        const float D = vload_half(0, d + b), DM = vload_half(0, dm + b);
        uint qhw[8];
        WREAD(qh, 32, qhw, kb, 0);
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) {
            uint qw[8];
            WREAD(q, 128, qw, kb, 8 * j);
            _Pragma("unroll")
            for (int hi = 0; hi < 2; ++hi) {
                const int sb = 2 * j + hi;
                float sc, mn;
                scale_min_k4(sb, sw, &sc, &mn);
                const uint dl2 = (uint)as_ushort((half)(D * sc)) * 0x10001u;
                const uint nml2 = (uint)as_ushort((half)(-DM * mn)) * 0x10001u;
                uint pw[16];
                _Pragma("unroll")
                for (int i = 0; i < 8; ++i) {
                    const uint w = hi ? qw[i] >> 4 : qw[i];
                    // bit sb of bytes 0, 2 (1, 3) to bit 4 of the halves
                    const uint h0 = sb <= 4 ? qhw[i] << (4 - sb) : qhw[i] >> (sb - 4);
                    const uint h1 = qhw[i] >> (sb + 4);
                    pw[2 * i]     = hscale((w & 0x000F000Fu) | ((h0 & 0x00100010u) | MAGIC), dl2, nml2);
                    pw[2 * i + 1] = hscale(((w >> 8) & 0x000F000Fu) | ((h1 & 0x00100010u) | MAGIC), dl2, nml2);
                }
                const int8 b0 = as_int8(vload8(0, pw)), b1 = as_int8(vload8(0, pw + 8));
                MUL32(kb * QK_K + 32 * sb)
            }
        }
    }
    EPILOGUE
}

// (p - 1056) * sc on the 32 halves of p: Q6_K's 1024 + code to sc (code - 32)
static inline uint hscale6(uint p, uint sc2) {
    uint r;
    __asm__("{\n"
            ".decl PH v_type=G type=hf num_elts=32 align=GRF alias=<%1, 0>\n"
            ".decl SH v_type=G type=hf num_elts=32 align=GRF alias=<%2, 0>\n"
            ".decl RH v_type=G type=hf num_elts=32 align=GRF alias=<%0, 0>\n"
            "add (M1_NM, 32) RH(0,0)<1> PH(0,0)<1;1,0> 0xE420:hf\n"
            "mul (M1_NM, 32) RH(0,0)<1> RH(0,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "}\n" : "=rw"(r) : "rw"(p), "rw"(sc2));
    return r;
}

// Q6_K: ql 128 bytes a block, qh 64, scales 16 (int8), d. Sub-block sb
// (32 weights at 32 sb) = half n = sb / 4, part p = sb % 4: low/high
// nibble p / 2 of ql bytes 64 n + 32 (p & 1) .. + 31, two bits 2 p of qh
// bytes 32 n .. + 31, scales 8 n + 2 p (weights 0..15) and + 1 (16..31).
// Packed as Q4_K: the nibble and the two bits (at bits 4, 5) of bytes 0, 2
// (1, 3) as halves 1024 + code.
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_kq2d_q6_K(global const uchar * ql, global const uchar * qh, global const char * s,
                                    global const half * d, ARGS) {
    PROLOGUE
    for (int kb = 0; kb < nb; ++kb) {
        KB_SYNC(kb)
        const ulong b = (ulong)n * nb + kb;
        const float D = vload_half(0, d + b);
        uint scw[8];
        WREAD(s, 16, scw, kb, 0);   // the 16 scale bytes are dwords 0..3 (4..7 read past, unused)
        _Pragma("unroll")
        for (int nn = 0; nn < 2; ++nn) {
            uint qhw[8];
            WREAD(qh, 64, qhw, kb, 8 * nn);
            _Pragma("unroll")
            for (int pl = 0; pl < 2; ++pl) {
                uint qlw[8];
                WREAD(ql, 128, qlw, kb, 16 * nn + 8 * pl);
                _Pragma("unroll")
                for (int ph = 0; ph < 2; ++ph) {
                    const int p = pl + 2 * ph, sb = 4 * nn + p, is = 8 * nn + 2 * p;
                    const half s0 = (half)(D * (float)(char)((scw[is >> 2] >> (8 * (is & 3))) & 0xFF));
                    const half s1 = (half)(D * (float)(char)((scw[(is + 1) >> 2] >> (8 * ((is + 1) & 3))) & 0xFF));
                    const uint s02 = (uint)as_ushort(s0) * 0x10001u, s12 = (uint)as_ushort(s1) * 0x10001u;
                    uint pw[16];
                    _Pragma("unroll")
                    for (int i = 0; i < 8; ++i) {
                        const uint w = qlw[i] >> (4 * ph);
                        // bits 2p, 2p + 1 of bytes 0, 2 (1, 3) to bits 4, 5 of the halves
                        const uint h0 = p <= 2 ? qhw[i] << (4 - 2 * p) : qhw[i] >> (2 * p - 4);
                        const uint h1 = qhw[i] >> (2 * p + 4);
                        const uint sc2 = i < 4 ? s02 : s12;
                        pw[2 * i]     = hscale6((w & 0x000F000Fu) | ((h0 & 0x00300030u) | MAGIC), sc2);
                        pw[2 * i + 1] = hscale6(((w >> 8) & 0x000F000Fu) | ((h1 & 0x00300030u) | MAGIC), sc2);
                    }
                    const int8 b0 = as_int8(vload8(0, pw)), b1 = as_int8(vload8(0, pw + 8));
                    MUL32(kb * QK_K + 32 * sb)
                }
            }
        }
    }
    EPILOGUE
}

#ifndef GGML_OPENCL_NO_Q3K   // 0116: built without it when the program fails with it (ggml_cl_build_program_q3k)
#ifdef Q3K_BREAK_BUILD   // the red case of the build fallback (GGML_OPENCL_Q3K_OPTS): the Q3_K text does not compile
#error "Q3K_BREAK_BUILD"
#endif
// ARC-LAB 0116: (p - 1028) * sc on the 32 halves of p: Q3_K's 1024 + u to sc (u - 4)
static inline uint hscale3(uint p, uint sc2) {
    uint r;
    __asm__("{\n"
            ".decl PH v_type=G type=hf num_elts=32 align=GRF alias=<%1, 0>\n"
            ".decl SH v_type=G type=hf num_elts=32 align=GRF alias=<%2, 0>\n"
            ".decl RH v_type=G type=hf num_elts=32 align=GRF alias=<%0, 0>\n"
            "add (M1_NM, 32) RH(0,0)<1> PH(0,0)<1;1,0> 0xE404:hf\n"
            "mul (M1_NM, 32) RH(0,0)<1> RH(0,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "}\n" : "=rw"(r) : "rw"(p), "rw"(sc2));
    return r;
}

// scale is (0..15) of a Q3_K block's 12 scale bytes w, minus 32 (mul_mv_iq_q8_1.cl's q3k_scale; -DQ3K_RED
// drops the top two bits: the red case)
static inline int q3k_scale_2d(uint3 w, int is) {
    const uint v8 = is < 8 ? (is < 4 ? w.s0 : w.s1) : (is < 12 ? w.s0 : w.s1);
    const uint lo = is < 8 ? (v8 >> (8 * (is & 3))) & 0xFu : (v8 >> (8 * (is & 3) + 4)) & 0xFu;
#ifdef Q3K_RED
    const uint hi = 0;
#else
    const uint hi = (w.s2 >> (8 * (is & 3) + 2 * (is >> 2))) & 3u;
#endif
    return (int)(lo | (hi << 4)) - 32;
}

// Q3_K in the planes of mul_mv_iq_q8_1.cl (per slice d[nblk] | hmask[32 nblk] | qs[64 nblk] |
// scales[12 nblk], at q + offq): hmask and qs are surfaces of ne01 rows of nb * 32 and nb * 64 bytes
// (the host checks their 64-byte alignment), d and the scales are read per lane. Sub-block sb = 4 n + j
// (32 weights at 32 sb): bits 2j of qs bytes 32 n .. + 31, bit 4n + j of hmask bytes 0 .. 31, scales
// 8 n + 2 j (weights 0..15) and + 1 (16..31). Packed as Q6_K: the two bits (bits 0, 1) and the hmask
// bit (bit 2) of bytes 0, 2 (1, 3) as halves 1024 + u, u - 4 the weight.
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_kq2d_q3_K(global const uchar * q, ulong offq, global const half * dp_unused, ARGS) {
    PROLOGUE
    global const uchar * S  = q + offq;
    const ulong nblk = (ulong)ne01 * nb;
    global const half  * dq = (global const half *)S;
    global const uchar * hm = S + nblk * 2;
    global const uchar * qs = S + nblk * 34;
    global const uchar * sc = S + nblk * 98;
    for (int kb = 0; kb < nb; ++kb) {
        KB_SYNC(kb)
        const ulong b = (ulong)n * nb + kb;
        const float D = vload_half(b, dq);
        const uint3 sw = vload3(0, (global const uint *)(sc + b * 12));
        uint hmw[8];
        WREAD(hm, 32, hmw, kb, 0);
        _Pragma("unroll")
        for (int nn = 0; nn < 2; ++nn) {
            uint qw[8];
            WREAD(qs, 64, qw, kb, 8 * nn);
            _Pragma("unroll")
            for (int j = 0; j < 4; ++j) {
                const int sb = 4 * nn + j, is = 8 * nn + 2 * j, hb = 4 * nn + j;
                const half s0 = (half)(D * (float)q3k_scale_2d(sw, is));
                const half s1 = (half)(D * (float)q3k_scale_2d(sw, is + 1));
                const uint s02 = (uint)as_ushort(s0) * 0x10001u, s12 = (uint)as_ushort(s1) * 0x10001u;
                uint pw[16];
                _Pragma("unroll")
                for (int i = 0; i < 8; ++i) {
                    const uint w = qw[i] >> (2 * j);
                    const uint h = hmw[i] >> hb;
                    const uint sc2 = i < 4 ? s02 : s12;
                    pw[2 * i]     = hscale3((w & 0x00030003u) | ((h & 0x00010001u) << 2) | MAGIC, sc2);
                    pw[2 * i + 1] = hscale3(((w >> 8) & 0x00030003u) | (((h >> 8) & 0x00010001u) << 2) | MAGIC, sc2);
                }
                const int8 b0 = as_int8(vload8(0, pw)), b1 = as_int8(vload8(0, pw + 8));
                MUL32(kb * QK_K + 32 * sb)
            }
        }
    }
    EPILOGUE
}
#endif // GGML_OPENCL_NO_Q3K

#ifndef GGML_OPENCL_NO_Q2K   // 0117: built without it when the program fails with it (ggml_cl_build_program_q3k)
#ifdef Q2K_BREAK_BUILD   // the red case of the build fallback (GGML_OPENCL_Q2K_OPTS): the Q2_K text does not compile
#error "Q2K_BREAK_BUILD"
#endif
// ARC-LAB 0117: Q2_K (kq type index 13). A slice in planes (ggml-opencl-planes.h): dm[nblk] (d and dmin, a
// half pair) | scales[16 nblk] | qs[64 nblk]. Weight 128 n + 32 j + t of a block (n 0..1, j 0..3, t 0..31)
// is bits 2j of qs[32 n + t], a code q in 0..3, at scale byte is = 8 n + 2 j + t / 16: d (byte & 15) q -
// dmin (byte >> 4), ggml's dequantize_row_q2_K. -DQ2K_RED swaps the byte's two nibbles (the red case).
#define Q2K_SC(nblk) ((ulong)(nblk) * 4)
#define Q2K_QS(nblk) ((ulong)(nblk) * 20)

static inline uint q2k_byte(uint4 w, int k) {
    const uint v = k < 4 ? w.s0 : (k < 8 ? w.s1 : (k < 12 ? w.s2 : w.s3));
    return (v >> (8 * (k & 3))) & 0xFFu;
}

// the scale and the min of scale byte b
static inline void q2k_sm(uint b, float * sc, float * m) {
#ifdef Q2K_RED
    *sc = (float)(b >> 4); *m = (float)(b & 0xFu);
#else
    *sc = (float)(b & 0xFu); *m = (float)(b >> 4);
#endif
}

// Q2_K on its planes: qs is a surface of ne01 rows of nb x 64 bytes (the host checks its 64-byte
// alignment), dm and the 16 scale bytes are read per lane. Sub-block sb = 4 n + j (32 weights at 32 sb):
// bits 2j of qs bytes 32 n .. + 31 as halves 1024 + q, Q4_K's (p - 1024) dl + nml with dl = d sc, nml =
// -dmin m, per 16 weights
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_kq2d_q2_K(global const uchar * q, ulong offq, global const half * dp_unused, ARGS) {
    PROLOGUE
    global const uchar * S  = q + offq;
    const ulong nblk = (ulong)ne01 * nb;
    global const half  * dmp = (global const half *)S;
    global const uchar * sc  = S + Q2K_SC(nblk);
    global const uchar * qs  = S + Q2K_QS(nblk);
    for (int kb = 0; kb < nb; ++kb) {
        KB_SYNC(kb)
        const ulong b = (ulong)n * nb + kb;
        const float2 dm = vload_half2(b, dmp);
        const uint4  sw = vload4(0, (global const uint *)(sc + b * 16));
        _Pragma("unroll")
        for (int nn = 0; nn < 2; ++nn) {
            uint qw[8];
            WREAD(qs, 64, qw, kb, 8 * nn);
            _Pragma("unroll")
            for (int j = 0; j < 4; ++j) {
                const int sb = 4 * nn + j, is = 8 * nn + 2 * j;
                float sc0, m0, sc1, m1;
                q2k_sm(q2k_byte(sw, is), &sc0, &m0);
                q2k_sm(q2k_byte(sw, is + 1), &sc1, &m1);
                const uint dl02 = (uint)as_ushort((half)(dm.s0 * sc0)) * 0x10001u, nml02 = (uint)as_ushort((half)(-dm.s1 * m0)) * 0x10001u;
                const uint dl12 = (uint)as_ushort((half)(dm.s0 * sc1)) * 0x10001u, nml12 = (uint)as_ushort((half)(-dm.s1 * m1)) * 0x10001u;
                uint pw[16];
                _Pragma("unroll")
                for (int i = 0; i < 8; ++i) {
                    const uint w = qw[i] >> (2 * j);
                    const uint dl2 = i < 4 ? dl02 : dl12, nml2 = i < 4 ? nml02 : nml12;
                    pw[2 * i]     = hscale((w & 0x00030003u) | MAGIC, dl2, nml2);
                    pw[2 * i + 1] = hscale(((w >> 8) & 0x00030003u) | MAGIC, dl2, nml2);
                }
                const int8 b0 = as_int8(vload8(0, pw)), b1 = as_int8(vload8(0, pw + 8));
                MUL32(kb * QK_K + 32 * sb)
            }
        }
    }
    EPILOGUE
}
#endif // GGML_OPENCL_NO_Q2K

// (p - 1032) * d on the 32 halves of p: Q4_0's 1024 + code to d (code - 8),
// with the rounding of a dequantization to f16. Q40_RED: code - 4 (the red
// case of the suite).
#ifdef Q40_RED
#define Q40_SUB "0xE404"
#else
#define Q40_SUB "0xE408"
#endif
static inline uint hscale40(uint p, uint d2) {
    uint r;
    __asm__("{\n"
            ".decl PH v_type=G type=hf num_elts=32 align=GRF alias=<%1, 0>\n"
            ".decl SH v_type=G type=hf num_elts=32 align=GRF alias=<%2, 0>\n"
            ".decl RH v_type=G type=hf num_elts=32 align=GRF alias=<%0, 0>\n"
            "add (M1_NM, 32) RH(0,0)<1> PH(0,0)<1;1,0> " Q40_SUB ":hf\n"
            "mul (M1_NM, 32) RH(0,0)<1> RH(0,0)<1;1,0> SH(0,0)<1;1,0>\n"
            "}\n" : "=rw"(r) : "rw"(p), "rw"(d2));
    return r;
}

// Q4_0's flat planes (0096): 16 code bytes and one f16 scale per 32 weights;
// byte j holds weight j in its low nibble and weight j + 16 in its high one.
// 32 bytes are two blocks, so a dword read gives k 0..15 of a block from the
// low nibbles and k 16..31 from the high ones, in Q4_K's packed P4 order.
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_kq2d_q4_0(global const uchar * q, global const half * d, ARGS) {
    PROLOGUE
    for (int kb = 0; kb < nb; ++kb) {
        KB_SYNC(kb)
        const ushort8 dv = vload8(0, (global const ushort *)d + (ulong)n * nb * 8 + kb * 8);
        const ushort da[8] = { dv.s0, dv.s1, dv.s2, dv.s3, dv.s4, dv.s5, dv.s6, dv.s7 };
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) {
            uint qw[8];
            WREAD(q, 128, qw, kb, 8 * j);
            _Pragma("unroll")
            for (int bl = 0; bl < 2; ++bl) {
                const int sb = 2 * j + bl;
                const uint d2 = (uint)da[sb] * 0x10001u;
                uint pw[16];
                _Pragma("unroll")
                for (int i = 0; i < 4; ++i) {
                    const uint w = qw[4 * bl + i], wh = w >> 4;
                    pw[2 * i]     = hscale40(PAIR0(w), d2);
                    pw[2 * i + 1] = hscale40(PAIR1(w), d2);
                    pw[8 + 2 * i]     = hscale40(PAIR0(wh), d2);
                    pw[8 + 2 * i + 1] = hscale40(PAIR1(wh), d2);
                }
                const int8 b0 = as_int8(vload8(0, pw)), b1 = as_int8(vload8(0, pw + 8));
                MUL32(kb * QK_K + 32 * sb)
            }
        }
    }
    EPILOGUE
}

// ---------------------------------------------------------------------------
// Q4_K on int8 DPAS at full rate: one i8 x i8 DPAS a sub-block, twice the
// fp16 kernel's multiply rate. kernel_quant_i8 writes the activations as
// int8 per (token, 256) with an f32 scale xd [block][tp] and the integer sums
// of each 32, S [token][K / 32] (f16, exact to 2048; 4,064 at most).
//
// The codes go in as s8 (code - 8): a sub-block's product is then at most
// 32 * 127 * 8 = 32,512 in magnitude, a 16-bit value, and its 6-bit scale
// is applied by one integer mad on word sources (acc += ir * sc) into an
// int32 accumulator that spans the super-block. The constant part,
// 8 D sc - DM m, rides on one fp16 DPAS a super-block against S:
//   sum = xd (D sum_sb sc_sb ir_sb + sum_sb (8 D sc_sb - DM m_sb) S_sb)
// The activation tile and the weights are read a sub-block ahead (without
// that, 19-30 % slower, measured-here). TMM tokens a sub-group.
#ifndef TMM
#define TMM 32
#endif
#define NGM (TMM / 8)

// one work-group of 256 a token, eight values a work-item (k % 256 == 0):
// a 256-block is 32 work-items, the sub-groups 2 s and 2 s + 1 (Intel lays
// a work-group's sub-groups out in local-id order)
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_quant_i8(global const char * src, ulong offs, global char * xq, global float * xd,
                             global half * xs, int k, int tp) {
    const int tok = get_group_id(0), sg = get_sub_group_id(), lane = get_sub_group_local_id();
    global const float * x = (global const float *)(src + offs) + (ulong)tok * k;
    local float sm[64];
    const int st = 8 * get_local_size(0), nr = (k + st - 1) / st;
    for (int r = 0; r < nr; ++r) {
        const int i = 8 * get_local_id(0) + r * st;
        const float8 v = i < k ? vload8(0, x + i) : (float8)(0.f);
        const float8 av = fabs(v);
        const float m8 = fmax(fmax(fmax(av.s0, av.s1), fmax(av.s2, av.s3)), fmax(fmax(av.s4, av.s5), fmax(av.s6, av.s7)));
        sm[sg] = sub_group_reduce_max(m8);   // a 256-block is two sub-groups
        barrier(CLK_LOCAL_MEM_FENCE);
        const float bm = fmax(sm[sg & ~1], sm[sg | 1]);
        barrier(CLK_LOCAL_MEM_FENCE);
        // a block below 1e-30 quantizes to zeros (127 / bm would overflow)
        const float qs = bm / 127.f, iq = bm > 1e-30f ? 127.f / bm : 0.f;
        const int8 qv = convert_int8_rte(v * iq);
        int s8 = qv.s0 + qv.s1 + qv.s2 + qv.s3 + qv.s4 + qv.s5 + qv.s6 + qv.s7;
        s8 += intel_sub_group_shuffle_xor(s8, 1);
        s8 += intel_sub_group_shuffle_xor(s8, 2);
        if (i < k) {
            vstore8(convert_char8(qv), 0, xq + (ulong)tok * k + i);
            if ((lane & 3) == 0) xs[(ulong)tok * (k / 32) + i / 32] = (half)(float)s8;
            if ((i & 255) == 0) xd[(ulong)(i / 256) * tp + tok] = qs;
        }
    }
}

// get_scale_min_k4 as integers
static inline void scale_min_k4i(int j, uint3 w, uint * sc, uint * m) {
    if (j < 4) {
        *sc = (w.s0 >> (8 * j)) & 63;
        *m  = (w.s1 >> (8 * j)) & 63;
    } else {
        const int jj = j - 4;
        const uint q8 = (w.s2 >> (8 * jj)) & 0xFF, lo = (w.s0 >> (8 * jj)) & 0xFF, hi = (w.s1 >> (8 * jj)) & 0xFF;
        *sc = (q8 & 0xF) | ((lo >> 6) << 4);
        *m  = (q8 >> 4) | ((hi >> 6) << 4);
    }
}

// acc + (low word of ir) * sc
static inline int mad16(int acc, int ir, short sc) {
    int r;
    __asm__("{\n"
            ".decl IRW v_type=G type=w num_elts=32 align=GRF alias=<%2, 0>\n"
            "mad (M1, 16) %0(0,0)<1> IRW(0,0)<2;1,0> %3(0,0)<1;1,0> %1(0,0)<1;1,0>\n"
            "}\n" : "=rw"(r) : "rw"(acc), "rw"(ir), "rw"(sc));
    return r;
}

// the int8 activation tile: 32 tokens from t0 + 32 gb, bytes kk .. kk + 31
#define AREAD8(dst, kk, gb) intel_sub_group_2d_block_read_8b_32r32x1c((const global void *)xq, ne00, ne11, ne00, \
                                (int2)((kk), t0 + 32 * (gb)), (dst))

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_kq2d_q4_K_i8(global const uchar * q, global const uchar * s, global const half * d,
                                       global const half * dm, global const char * xq, global const float * xd,
                                       global const half * xs, global char * dst, ulong offd, ulong nb1,
                                       int ne00, int ne01, int ne11, int tp) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int r0 = (get_group_id(1) * WG + sg) * 16;
    const int t0 = get_group_id(0) * TMM;
    const int nb = ne00 / QK_K;
    const int n = min(r0 + lane, ne01 - 1);
    float8 acc[NGM];
    _Pragma("unroll")
    for (int g = 0; g < NGM; ++g) acc[g] = (float8)(0.f);
    // the int8 activation tile of the next sub-block, and the next 32 code bytes
    ushort acur[TMM / 32][32];
    _Pragma("unroll")
    for (int gb = 0; gb < TMM / 32; ++gb) AREAD8(acur[gb], 0, gb);
    uint qn[8];
    WREAD(q, 128, qn, 0, 0);
    for (int kb = 0; kb < nb; ++kb) {
        if (kb % KSYNC_I8 == 0) barrier(CLK_LOCAL_MEM_FENCE);
        const ulong b = (ulong)n * nb + kb;
        const uint3 sw = vload3(0, (global const uint *)(s + b * 12));
        const float D = vload_half(0, d + b), DM = vload_half(0, dm + b);
        // the block's sub-block sums and token scales, used after its DPAS
        ushort sa[NGM][8];
        float8 xv[NGM];
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) {
            intel_sub_group_2d_block_read_16b_8r16x1c((const global void *)xs, ne00 / 16, ne11, ne00 / 16,
                (int2)(8 * kb, t0 + 8 * g), sa[g]);
            xv[g] = vload8(0, xd + (ulong)kb * tp + t0 + 8 * g);
        }
        const int kn = min(kb + 1, nb - 1);
        int8 ai[NGM];
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) ai[g] = (int8)(0);
        half mh[16];
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) {
            uint qw[8];
            _Pragma("unroll")
            for (int i = 0; i < 8; ++i) qw[i] = qn[i];
            if (j < 3) WREAD(q, 128, qn, kb, 8 * (j + 1));
            else       WREAD(q, 128, qn, kn, 0);
            _Pragma("unroll")
            for (int hi = 0; hi < 2; ++hi) {
                const int sb = 2 * j + hi;
                uint sc, mn;
                scale_min_k4i(sb, sw, &sc, &mn);
                mh[sb] = (half)(8.f * D * (float)sc - DM * (float)mn);
                const short scs = (short)sc;
                int8 bq;
                _Pragma("unroll")
                for (int i = 0; i < 8; ++i) {
                    const uint t = (hi ? qw[i] >> 4 : qw[i]) & 0x0F0F0F0Fu;
                    bq[i] = as_int((t + 0x78787878u) ^ 0x80808080u);   // each byte code - 8
                }
                const int k0 = kb * QK_K + 32 * sb;
                ushort anx[TMM / 32][32];
                _Pragma("unroll")
                for (int gb = 0; gb < TMM / 32; ++gb) AREAD8(anx[gb], sb < 7 ? k0 + 32 : kn * QK_K, gb);
                _Pragma("unroll")
                for (int gb = 0; gb < TMM / 32; ++gb) {
                    int8 ir[4];
                    _Pragma("unroll")
                    for (int u = 0; u < 4; ++u)
                        ir[u] = intel_sub_group_i8_i8_matrix_mad_k32(as_short8(vload8(0, acur[gb] + 8 * u)), bq, (int8)(0));
                    _Pragma("unroll")
                    for (int u = 0; u < 4; ++u) {
                        int8 a = ai[gb * 4 + u];
                        a.s0 = mad16(a.s0, ir[u].s0, scs); a.s1 = mad16(a.s1, ir[u].s1, scs);
                        a.s2 = mad16(a.s2, ir[u].s2, scs); a.s3 = mad16(a.s3, ir[u].s3, scs);
                        a.s4 = mad16(a.s4, ir[u].s4, scs); a.s5 = mad16(a.s5, ir[u].s5, scs);
                        a.s6 = mad16(a.s6, ir[u].s6, scs); a.s7 = mad16(a.s7, ir[u].s7, scs);
                        ai[gb * 4 + u] = a;
                    }
                }
                _Pragma("unroll")
                for (int gb = 0; gb < TMM / 32; ++gb)
                    _Pragma("unroll")
                    for (int z = 0; z < 32; ++z) acur[gb][z] = anx[gb][z];
            }
        }
        // the constant part: S of the block's 8 sub-blocks (k slots 0..7, 8..15 zero)
        half16 mv;
        _Pragma("unroll")
        for (int sb = 0; sb < 16; ++sb) mv[sb] = sb < 8 ? mh[sb] : 0.h;
        const int8 bm = as_int8(mv);
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) {
            const float8 cst = DPAS(as_short8(vload8(0, sa[g])), bm, (float8)(0.f));
            acc[g] = fma(fma(convert_float8(ai[g]), (float8)(D), cst), xv[g], acc[g]);
        }
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

// Q4_0 on int8 DPAS: the codes as s8 (code - 8), one i8 DPAS per 32 weights, the
// block's f16 scale applied per lane in f32, the activations' scale per 256 as
// for Q4_K. No constant part: code - 8 is the weight's own integer.
#ifdef Q40_RED
#define Q40_I8_OFF 0x7Cu
#else
#define Q40_I8_OFF 0x78u
#endif
#define Q40_S8(t) as_int((((t) & 0x0F0F0F0Fu) + Q40_I8_OFF * 0x01010101u) ^ 0x80808080u)
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mul_mm_kq2d_q4_0_i8(global const uchar * q, global const half * d, global const char * xq,
                                       global const float * xd, global char * dst, ulong offd, ulong nb1,
                                       int ne00, int ne01, int ne11, int tp) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int r0 = (get_group_id(1) * WG + sg) * 16;
    const int t0 = get_group_id(0) * TMM;
    const int nb = ne00 / QK_K;
    const int n = min(r0 + lane, ne01 - 1);
    float8 acc[NGM];
    _Pragma("unroll")
    for (int g = 0; g < NGM; ++g) acc[g] = (float8)(0.f);
    ushort acur[TMM / 32][32];
    _Pragma("unroll")
    for (int gb = 0; gb < TMM / 32; ++gb) AREAD8(acur[gb], 0, gb);
    uint qn[8];
    WREAD(q, 128, qn, 0, 0);
    for (int kb = 0; kb < nb; ++kb) {
        if (kb % KSYNC_I8 == 0) barrier(CLK_LOCAL_MEM_FENCE);
        const float8 dv = vload_half8(0, d + (ulong)n * nb * 8 + kb * 8);
        const float da[8] = { dv.s0, dv.s1, dv.s2, dv.s3, dv.s4, dv.s5, dv.s6, dv.s7 };
        float8 xv[NGM], af[NGM];
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) {
            xv[g] = vload8(0, xd + (ulong)kb * tp + t0 + 8 * g);
            af[g] = (float8)(0.f);
        }
        const int kn = min(kb + 1, nb - 1);
        _Pragma("unroll")
        for (int j = 0; j < 4; ++j) {
            uint qw[8];
            _Pragma("unroll")
            for (int i = 0; i < 8; ++i) qw[i] = qn[i];
            if (j < 3) WREAD(q, 128, qn, kb, 8 * (j + 1));
            else       WREAD(q, 128, qn, kn, 0);
            _Pragma("unroll")
            for (int bl = 0; bl < 2; ++bl) {
                const int sb = 2 * j + bl;
                int8 bq;
                _Pragma("unroll")
                for (int i = 0; i < 4; ++i) {
                    const uint w = qw[4 * bl + i];
                    bq[i]     = Q40_S8(w);
                    bq[4 + i] = Q40_S8(w >> 4);
                }
                const int k0 = kb * QK_K + 32 * sb;
                ushort anx[TMM / 32][32];
                _Pragma("unroll")
                for (int gb = 0; gb < TMM / 32; ++gb) AREAD8(anx[gb], sb < 7 ? k0 + 32 : kn * QK_K, gb);
                _Pragma("unroll")
                for (int gb = 0; gb < TMM / 32; ++gb) {
                    int8 ir[4];
                    _Pragma("unroll")
                    for (int u = 0; u < 4; ++u)
                        ir[u] = intel_sub_group_i8_i8_matrix_mad_k32(as_short8(vload8(0, acur[gb] + 8 * u)), bq, (int8)(0));
                    _Pragma("unroll")
                    for (int u = 0; u < 4; ++u)
                        af[gb * 4 + u] = fma(convert_float8(ir[u]), (float8)(da[sb]), af[gb * 4 + u]);
                }
                _Pragma("unroll")
                for (int gb = 0; gb < TMM / 32; ++gb)
                    _Pragma("unroll")
                    for (int z = 0; z < 32; ++z) acur[gb][z] = anx[gb][z];
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
