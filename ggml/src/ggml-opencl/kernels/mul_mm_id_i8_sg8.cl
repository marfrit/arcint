#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// MUL_MAT_ID on int8 DPAS for sub-group 8 (Alchemist), its own program: a compile failure costs this arm
// only, ggml_cl_mul_mat_id_kq keeps the fp16 tile. The shape of mul_mm_q40_i8_sg8.cl (0099, 0102) with the
// route of kernel_moe_route_tiles: a work-group takes WG sub-groups of 8 weight rows and one tile of up to
// TMM (token, slot) pairs of one expert; a lane is a weight row.
//
// The activations. Routed pairs are not contiguous tokens. Three layouts were weighed, counted for one tile
// of T pairs, B 32-blocks of K and R rows, each sub-group needing all of the tile's a operands:
//   - columns in place, one 32-byte block read a pair: R/8 x B x T reads, no copy, no local memory;
//   - staged in local memory per 256 of K (0071, kq_i8_col): R/8 x B x T/8 reads, plus the stage's T x K
//     bytes per product (gate and up each pay it) and two barriers per 256;
//   - packed once per route in the tile's own order (this file): R/8 x B x T/8 reads, one global block read an
//     operand, no local memory, no barrier, plus T x K bytes copied once per (activation, route), so gate and
//     up share it. Strata's Xq has this form (sycl/include/strata/prefill/esimd_moe.hpp: rows back to back in
//     expert order, 8-row tiles, one 256-byte block load an A operand; no shared memory, no barrier).
// It costs the least reads, and the copy is a small part of them: a tile copies T x K bytes once and reads
// R/8 x K/32 x T x 32 = R x T x K / 8 bytes, 64 times more at R = 512. The route-major order also keeps a tile's
// reads inside T x K contiguous bytes.
// Layout, with the tile index ti, all padded to TMM pairs a tile (a part-full tile leaves its tail unwritten):
//   xq [ti][k / 32][TMM][32 bytes]    so pair group g of a block is the 256 bytes at offset 256 g
//   xd [ti][k / 256][TMM] float       the scale per 256
//   xs [ti][k / 256][TMM][16 halves]  the sums per 32 and 8 zeros (Q4_K's constant term)
//   oo [ti][TMM] ulong                the pair's output offset, t nb2 + i nb1, so that no kernel divides
// Column quantization is kernel_quant_i8_sg8 as it stands, once a graph per src1: a broadcast b (gate, up) has
// n_used pairs a column, so it quantizes one column for n_used copies. kernel_id_pack_i8_sg8 copies it into the
// tile order. A group past a tile's pairs is skipped (Strata's nvalid); its reads would stay inside the padded
// tile in any case, and a pair past te.z is never stored.
//
// The value form, IQ4_XS and IQ4_NL (dequantize_row_iq4_xs, dequantize_row_iq4_nl in ggml-quants.c; deq_iq4_xs
// in mul_mm_kq_f16.cl):
//   weight = d (ls - 32) kvalues_iq4nl[code]    IQ4_XS, d the f16 of the 256-block, ls 0..63 of the 32-block:
//                                               scales_l nibble | scales_h 2 bits << 4
//   weight = d kvalues_iq4nl[code]              IQ4_NL, d the f16 of the 32-block
// The code is not linear (no 2q - 15): kvalues_iq4nl is a table, -127..113, already an s8, so the b operand is
// the table entry, never rounded. Byte j of a 16-byte block holds k = j in its low nibble and k = 16 + j in its
// high one, so dwords 0..3 of the b operand take the low nibbles and 4..7 the high ones: k ascending, as xq.
// The lookup is the part to price. Strata (esimd_moe.dp.cpp) gathers a 256-entry byte-pair table, 16 gathers a
// 32-block; -DID_KV_TABLE is that form on a constant table. The default has no memory read: id_kv4 is the same
// table as a mux tree over the code's four bit planes, 4 bytes a dword, in about 15 selects and 4 masks a dword.
// A 32-block is 8 such dwords for one 16-byte read. Which of the two is faster on this card is not measured.
//
// Bounds and rounding. |a| <= 127 (kernel_quant_i8_sg8: convert_int8_rte of x 127 / amax), |b| <= 127, so a
// 32-block's dot is at most 32 x 127 x 127 = 516,128 in magnitude: no overflow in int32, and exact in f32 (below
// 2^24). Q4_K: |b| <= 8, at most 32,512. ws = d (ls - 32) is an 11-bit mantissa times 6 bits, exact in f32; IQ4_NL's
// d and Q4_K's D sc likewise. What rounds, and in which order: per 32-block one fma af += dot ws, k ascending;
// per 256 one fma acc += af xd, k ascending; each is one f32 rounding, relative 2^-24. The fp16 tile instead rounds
// each weight to f16 and each activation to f16; here a weight keeps its exact value and an activation has 8 bits
// per 256 of K, which is what the answer-level bar has to judge. Q4_K's constant term is the dense kernel's: one
// k16 fp16 DPAS a block, the sums exact to 2048 and even up to 4,064, mh rounded to f16.

#define QK_K 256
#ifndef WG
#define WG 8                  // sub-groups a work-group, 8 weight rows each
#endif
#ifndef TMM
#define TMM 32                // pairs a tile, a multiple of 8
#endif
#define NGM (TMM / 8)
#ifndef NR
#define NR 2                  // sets of 8 weight rows a sub-group: one A read serves them all
#endif
#ifndef ID_KEEP_AF
#define ID_NOAF               // no 32-block accumulator: see ID_XV
#endif

// ID_RED: every weight scale negated (the red case of the suite)
#ifdef ID_RED
#define ID_WS(x) (-(x))
#else
#define ID_WS(x) (x)
#endif

// One sub-group a pair of a tile: the column's codes, scales and sums into the tile's order, and the pair's
// output offset. tpc is the columns' row stride (kernel_quant_i8_sg8's tp); a stale pair (a route that skipped
// ids) is clamped to a column so that its read stays in bounds: no tile holds it.
__attribute__((intel_reqd_sub_group_size(8)))
kernel void kernel_id_pack_i8_sg8(global const char * xq, global const float * xd, global const half * xs, int tpc,
                                  global const int * pairs, global const int4 * tiles, global const int * n_tiles,
                                  int n_used, int ne11, int n_cols, global char * oq, global float * od, global half * os,
                                  global ulong * oo, ulong nb1, ulong nb2, int k) {
    const int m = get_group_id(0), ti = get_group_id(1);
    if (ti >= *n_tiles) return;
    const int4 te = tiles[ti];
    if (m >= te.z) return;
    const int p = pairs[te.y + m], t = p / n_used, i = p % n_used;
    const int col = clamp((i % ne11) + t * ne11, 0, n_cols - 1);
    const int nb = k / QK_K;
    const bool first = get_sub_group_local_id() == 0;
    if (first) oo[(ulong)ti * TMM + m] = (ulong)t * nb2 + (ulong)i * nb1;
    for (int kb = 0; kb < nb; ++kb) {
        const ulong b256 = ((ulong)ti * nb + kb) * TMM + m;
        if (first) od[b256] = xd[(ulong)kb * tpc + col];
        intel_sub_group_block_write((global uint *)(os + b256 * 16),
                                    intel_sub_group_block_read((global const uint *)(xs + ((ulong)kb * tpc + col) * 16)));
        _Pragma("unroll")
        for (int sb = 0; sb < 8; ++sb) {
            intel_sub_group_block_write((global uint *)(oq + (((ulong)ti * nb + kb) * 8 + sb) * TMM * 32 + (ulong)m * 32),
                                        intel_sub_group_block_read((global const uint *)(xq + (((ulong)kb * 8 + sb) * tpc + col) * 32)));
        }
    }
}

// ---- the b operand of IQ4_XS and IQ4_NL: 16 code bytes -> 32 s8 values, 8 dwords ----
#ifdef ID_KV_TABLE
// byte b -> kvalues_iq4nl[b & 15] in byte 0, kvalues_iq4nl[b >> 4] in byte 1
constant uint id_kv_pair[256] = {
    0x8181u, 0x8198u, 0x81ADu, 0x81BFu, 0x81CFu, 0x81DDu, 0x81EAu, 0x81F6u,
    0x8101u, 0x810Du, 0x8119u, 0x8126u, 0x8135u, 0x8145u, 0x8159u, 0x8171u,
    0x9881u, 0x9898u, 0x98ADu, 0x98BFu, 0x98CFu, 0x98DDu, 0x98EAu, 0x98F6u,
    0x9801u, 0x980Du, 0x9819u, 0x9826u, 0x9835u, 0x9845u, 0x9859u, 0x9871u,
    0xAD81u, 0xAD98u, 0xADADu, 0xADBFu, 0xADCFu, 0xADDDu, 0xADEAu, 0xADF6u,
    0xAD01u, 0xAD0Du, 0xAD19u, 0xAD26u, 0xAD35u, 0xAD45u, 0xAD59u, 0xAD71u,
    0xBF81u, 0xBF98u, 0xBFADu, 0xBFBFu, 0xBFCFu, 0xBFDDu, 0xBFEAu, 0xBFF6u,
    0xBF01u, 0xBF0Du, 0xBF19u, 0xBF26u, 0xBF35u, 0xBF45u, 0xBF59u, 0xBF71u,
    0xCF81u, 0xCF98u, 0xCFADu, 0xCFBFu, 0xCFCFu, 0xCFDDu, 0xCFEAu, 0xCFF6u,
    0xCF01u, 0xCF0Du, 0xCF19u, 0xCF26u, 0xCF35u, 0xCF45u, 0xCF59u, 0xCF71u,
    0xDD81u, 0xDD98u, 0xDDADu, 0xDDBFu, 0xDDCFu, 0xDDDDu, 0xDDEAu, 0xDDF6u,
    0xDD01u, 0xDD0Du, 0xDD19u, 0xDD26u, 0xDD35u, 0xDD45u, 0xDD59u, 0xDD71u,
    0xEA81u, 0xEA98u, 0xEAADu, 0xEABFu, 0xEACFu, 0xEADDu, 0xEAEAu, 0xEAF6u,
    0xEA01u, 0xEA0Du, 0xEA19u, 0xEA26u, 0xEA35u, 0xEA45u, 0xEA59u, 0xEA71u,
    0xF681u, 0xF698u, 0xF6ADu, 0xF6BFu, 0xF6CFu, 0xF6DDu, 0xF6EAu, 0xF6F6u,
    0xF601u, 0xF60Du, 0xF619u, 0xF626u, 0xF635u, 0xF645u, 0xF659u, 0xF671u,
    0x0181u, 0x0198u, 0x01ADu, 0x01BFu, 0x01CFu, 0x01DDu, 0x01EAu, 0x01F6u,
    0x0101u, 0x010Du, 0x0119u, 0x0126u, 0x0135u, 0x0145u, 0x0159u, 0x0171u,
    0x0D81u, 0x0D98u, 0x0DADu, 0x0DBFu, 0x0DCFu, 0x0DDDu, 0x0DEAu, 0x0DF6u,
    0x0D01u, 0x0D0Du, 0x0D19u, 0x0D26u, 0x0D35u, 0x0D45u, 0x0D59u, 0x0D71u,
    0x1981u, 0x1998u, 0x19ADu, 0x19BFu, 0x19CFu, 0x19DDu, 0x19EAu, 0x19F6u,
    0x1901u, 0x190Du, 0x1919u, 0x1926u, 0x1935u, 0x1945u, 0x1959u, 0x1971u,
    0x2681u, 0x2698u, 0x26ADu, 0x26BFu, 0x26CFu, 0x26DDu, 0x26EAu, 0x26F6u,
    0x2601u, 0x260Du, 0x2619u, 0x2626u, 0x2635u, 0x2645u, 0x2659u, 0x2671u,
    0x3581u, 0x3598u, 0x35ADu, 0x35BFu, 0x35CFu, 0x35DDu, 0x35EAu, 0x35F6u,
    0x3501u, 0x350Du, 0x3519u, 0x3526u, 0x3535u, 0x3545u, 0x3559u, 0x3571u,
    0x4581u, 0x4598u, 0x45ADu, 0x45BFu, 0x45CFu, 0x45DDu, 0x45EAu, 0x45F6u,
    0x4501u, 0x450Du, 0x4519u, 0x4526u, 0x4535u, 0x4545u, 0x4559u, 0x4571u,
    0x5981u, 0x5998u, 0x59ADu, 0x59BFu, 0x59CFu, 0x59DDu, 0x59EAu, 0x59F6u,
    0x5901u, 0x590Du, 0x5919u, 0x5926u, 0x5935u, 0x5945u, 0x5959u, 0x5971u,
    0x7181u, 0x7198u, 0x71ADu, 0x71BFu, 0x71CFu, 0x71DDu, 0x71EAu, 0x71F6u,
    0x7101u, 0x710Du, 0x7119u, 0x7126u, 0x7135u, 0x7145u, 0x7159u, 0x7171u,
};
#define ID_PAIR(w, lo, hi) {                                                                                  \
        const uint t0 = id_kv_pair[(w) & 0xFFu], t1 = id_kv_pair[((w) >> 8) & 0xFFu];                         \
        const uint t2 = id_kv_pair[((w) >> 16) & 0xFFu], t3 = id_kv_pair[(w) >> 24];                          \
        lo = as_int((t0 & 0xFFu) | ((t1 & 0xFFu) << 8) | ((t2 & 0xFFu) << 16) | (t3 << 24));                  \
        hi = as_int(((t0 >> 8) & 0xFFu) | (t1 & 0xFF00u) | ((t2 & 0xFF00u) << 8) | ((t3 & 0xFF00u) << 16));   \
    }
static inline int8 id_iq4_b(uint w0, uint w1, uint w2, uint w3) {
    int8 b;
    ID_PAIR(w0, b.s0, b.s4)
    ID_PAIR(w1, b.s1, b.s5)
    ID_PAIR(w2, b.s2, b.s6)
    ID_PAIR(w3, b.s3, b.s7)
    return b;
}
#else
// bit b of each byte of w -> 0xFF or 0x00 in that byte
static inline uint id_mask(uint w, int b) {
    const uint m = (w >> b) & 0x01010101u;
    return (m << 8) - m;
}
#define ID_K(v) ((uint)(uchar)(v) * 0x01010101u)
#define ID_SEL(x, y, m) ((x) ^ (((x) ^ (y)) & (m)))
// kvalues_iq4nl of the nibble at bits b..b + 3 of each byte of w
static inline uint id_kv4(uint w, int b) {
    const uint m0 = id_mask(w, b), m1 = id_mask(w, b + 1), m2 = id_mask(w, b + 2), m3 = id_mask(w, b + 3);
    const uint a0 = ID_SEL(ID_K(-127), ID_K(-104), m0), a1 = ID_SEL(ID_K(-83), ID_K(-65), m0);
    const uint a2 = ID_SEL(ID_K(-49), ID_K(-35), m0), a3 = ID_SEL(ID_K(-22), ID_K(-10), m0);
    const uint a4 = ID_SEL(ID_K(1), ID_K(13), m0), a5 = ID_SEL(ID_K(25), ID_K(38), m0);
    const uint a6 = ID_SEL(ID_K(53), ID_K(69), m0), a7 = ID_SEL(ID_K(89), ID_K(113), m0);
    const uint b0 = ID_SEL(a0, a1, m1), b1 = ID_SEL(a2, a3, m1), b2 = ID_SEL(a4, a5, m1), b3 = ID_SEL(a6, a7, m1);
    const uint c0 = ID_SEL(b0, b1, m2), c1 = ID_SEL(b2, b3, m2);
    return ID_SEL(c0, c1, m3);
}
static inline int8 id_iq4_b(uint w0, uint w1, uint w2, uint w3) {
    int8 b;
    b.s0 = as_int(id_kv4(w0, 0)); b.s1 = as_int(id_kv4(w1, 0)); b.s2 = as_int(id_kv4(w2, 0)); b.s3 = as_int(id_kv4(w3, 0));
    b.s4 = as_int(id_kv4(w0, 4)); b.s5 = as_int(id_kv4(w1, 4)); b.s6 = as_int(id_kv4(w2, 4)); b.s7 = as_int(id_kv4(w3, 4));
    return b;
}
#endif

// ---- the shared parts of the three kernels ----
// A sub-group owns NR sets of 8 weight rows and reads each A tile once, into registers, for all of them. Measured on
// the A770 (window 120, IQ4_XS, NR 1): the A reads were 50 % of the kernel, the decode 20, the DPAS and the rest 20,
// the weight reads 3, the stores 1. Strata's thread owns 16 rows for the same reason (esimd_moe.dp.cpp); Alchemist's
// DPAS takes 8, so here the 16 are two DPAS on one A operand. A set past the rows is skipped, so ne01 % (8 NR) may be
// anything: ne01 % 8 == 0 by the dispatch, and ne00 % 256 == 0. A sub-group past the rows, a work-group past the tiles
// return at once; nothing here synchronizes a work-group.
#define ID_COLS                                                                                       \
    const int ti = get_group_id(0);                                                                   \
    if (ti >= *n_tiles) return;                                                                       \
    const int r0 = (get_group_id(1) * WG + get_sub_group_id()) * 8 * NR;                              \
    if (r0 >= ne01) return;                                                                           \
    const int4 te = tiles[ti];                                                                        \
    const int nz = te.z, ng = (nz + 7) / 8, nb = ne00 / QK_K;                                         \
    const int nrv = min((int)NR, (ne01 - r0) / 8);                                                    \
    const ulong wr = (ulong)te.x * ne01 + r0 + get_sub_group_local_id();   /* the expert's row */     \
    const ulong tb = (ulong)ti * nb;                                                                  \
    global const char * xt = xq + tb * 8 * TMM * 32;                                                  \
    float8 acc[NR * NGM];                                                                             \
    _Pragma("unroll")                                                                                 \
    for (int i = 0; i < NR * NGM; ++i) acc[i] = (float8)(0.f);

// the tile's A operands of a 32-block, read once
#define ID_ALOAD(kb32)                                                                                \
    int8 at[NGM];                                                                                     \
    _Pragma("unroll")                                                                                 \
    for (int g = 0; g < NGM; ++g) {                                                                   \
        at[g] = (int8)(0);                                                                            \
        if (g < ng) at[g] = as_int8(intel_sub_group_block_read8((global const uint *)(xt + (ulong)(kb32) * TMM * 32 + g * 256))); \
    }

// the scales of a 256-block's pair groups, and its accumulators, f32. A float8 at sub-group 8 is 8 GRF: acc and af
// cost 16 GRF a set and pair group, which is the register budget of NR x NGM. ID_NOAF folds the activation's scale
// into each 32-block's fma instead (8 GRF, one more multiply a DPAS, the same value to one more f32 rounding).
#ifdef ID_NOAF
#define ID_AF_DECL
#define ID_AF_ZERO
#else
#define ID_AF_DECL float8 af[NR * NGM];
#define ID_AF_ZERO _Pragma("unroll") for (int i = 0; i < NR * NGM; ++i) af[i] = (float8)(0.f);
#endif
#define ID_XV                                                                                         \
        float8 xv[NGM];                                                                               \
        ID_AF_DECL                                                                                    \
        _Pragma("unroll")                                                                             \
        for (int g = 0; g < NGM; ++g) {                                                               \
            xv[g] = (float8)(0.f);                                                                    \
            if (g < ng) xv[g] = vload8(0, xd + (tb + kb) * TMM + 8 * g);                              \
        }                                                                                             \
        ID_AF_ZERO

#ifdef ID_NOAF
#define ID_DPAS_ACC(nr, g, F, WS) acc[(nr) * NGM + (g)] = fma(F, (float8)(WS) * xv[g], acc[(nr) * NGM + (g)]);
#else
#define ID_DPAS_ACC(nr, g, F, WS) af[(nr) * NGM + (g)] = fma(F, (float8)(WS), af[(nr) * NGM + (g)]);
#endif
// one DPAS a pair group for a 32-block of set nr, k ascending
#define ID_DPAS(nr, BQ, WS)                                                                           \
                _Pragma("unroll")                                                                     \
                for (int g = 0; g < NGM; ++g) {                                                       \
                    if (g < ng) {                                                                     \
                        const int8 ir = intel_sub_group_i8_i8_matrix_mad_k32(at[g], BQ, (int8)(0));   \
                        ID_DPAS_ACC(nr, g, convert_float8(ir), WS)                                    \
                    }                                                                                 \
                }

// the end of a 256-block
#ifdef ID_NOAF
#define ID_ACC
#else
#define ID_ACC                                                                                        \
        _Pragma("unroll")                                                                             \
        for (int nr = 0; nr < NR; ++nr) {                                                             \
            _Pragma("unroll")                                                                         \
            for (int g = 0; g < NGM; ++g) {                                                           \
                if (nr < nrv && g < ng) acc[nr * NGM + g] = fma(af[nr * NGM + g], xv[g], acc[nr * NGM + g]); \
            }                                                                                         \
        }
#endif

// a pair's 8 rows of a set are 8 contiguous floats of its output column: one block write
#define ID_STORE                                                                                      \
    _Pragma("unroll")                                                                                 \
    for (int g = 0; g < NGM; ++g) {                                                                   \
        if (g < ng) {                                                                                 \
            const ulong8 ov = vload8(0, oo + (ulong)ti * TMM + 8 * g);                                \
            const ulong oa[8] = { ov.s0, ov.s1, ov.s2, ov.s3, ov.s4, ov.s5, ov.s6, ov.s7 };           \
            _Pragma("unroll")                                                                         \
            for (int nr = 0; nr < NR; ++nr) {                                                         \
                if (nr < nrv) {                                                                       \
                    const float8 av = acc[nr * NGM + g];                                              \
                    const float v[8] = { av.s0, av.s1, av.s2, av.s3, av.s4, av.s5, av.s6, av.s7 };    \
                    _Pragma("unroll")                                                                 \
                    for (int m = 0; m < 8; ++m) {                                                     \
                        if (8 * g + m < nz) {                                                         \
                            intel_sub_group_block_write((global uint *)(dst + offd + oa[m]) + r0 + 8 * nr, as_uint(v[m])); \
                        }                                                                             \
                    }                                                                                 \
                }                                                                                     \
            }                                                                                         \
        }                                                                                             \
    }

#define ID_ARGS global const char * xq, global const float * xd, global const int4 * tiles, global const int * n_tiles, \
    global const ulong * oo, global char * dst, ulong offd, int ne00, int ne01

// IQ4_XS, 136-byte blocks as stored: d, scales_h, scales_l[4], qs[128]. The rolled loop over 32 code bytes keeps
// one copy of the decodes in the instruction cache; only sb moves in it, and no array is indexed by it.
__attribute__((intel_reqd_sub_group_size(8)))
kernel void kernel_mul_mm_id_iq4_xs_i8_sg8(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    global const uchar * b0 = q + offq + wr * nb * 136;
    for (int kb = 0; kb < nb; ++kb) {
        uint2 hw[NR];
        float dv[NR];
        _Pragma("unroll")
        for (int nr = 0; nr < NR; ++nr) {                       // s0: d | scales_h << 16, s1: scales_l
            hw[nr] = (uint2)(0u);
            if (nr < nrv) hw[nr] = vload2(0, (global const uint *)(b0 + ((ulong)nr * 8 * nb + kb) * 136));
            dv[nr] = (float)as_half((ushort)hw[nr].s0);
        }
        ID_XV
        for (int j = 0; j < 4; ++j) {
            uint8 wv[NR];
            _Pragma("unroll")
            for (int nr = 0; nr < NR; ++nr) {
                wv[nr] = (uint8)(0u);
                if (nr < nrv) wv[nr] = vload8(0, (global const uint *)(b0 + ((ulong)nr * 8 * nb + kb) * 136 + 8 + j * 32));
            }
            _Pragma("unroll")
            for (int hi = 0; hi < 2; ++hi) {
                const int sb = 2 * j + hi;
                ID_ALOAD(kb * 8 + sb)
                _Pragma("unroll")
                for (int nr = 0; nr < NR; ++nr) {
                    if (nr < nrv) {
                        const int ls = (int)(((hw[nr].s1 >> (4 * sb)) & 15u) | (((hw[nr].s0 >> (16 + 2 * sb)) & 3u) << 4));
                        const float ws = ID_WS(dv[nr] * (float)(ls - 32));
                        const int8 bq = hi ? id_iq4_b(wv[nr].s4, wv[nr].s5, wv[nr].s6, wv[nr].s7) : id_iq4_b(wv[nr].s0, wv[nr].s1, wv[nr].s2, wv[nr].s3);
                        ID_DPAS(nr, bq, ws)
                    }
                }
            }
        }
        ID_ACC
    }
    ID_STORE
}

// IQ4_NL, flat planes: 16 code bytes and one half a 32-block
__attribute__((intel_reqd_sub_group_size(8)))
kernel void kernel_mul_mm_id_iq4_nl_i8_sg8(global const uchar * q, ulong offq, global const half * dp, ID_ARGS) {
    ID_COLS
    const ulong rb = wr * nb * 8;     // the row's first 32-block
    for (int kb = 0; kb < nb; ++kb) {
        ID_XV
        for (int j = 0; j < 4; ++j) {
            uint8 wv[NR];
            float2 ds[NR];
            _Pragma("unroll")
            for (int nr = 0; nr < NR; ++nr) {
                wv[nr] = (uint8)(0u);
                ds[nr] = (float2)(0.f);
                if (nr < nrv) {
                    const ulong blk = rb + (ulong)nr * 8 * nb * 8 + kb * 8 + 2 * j;
                    wv[nr] = vload8(0, (global const uint *)(q + offq + blk * 16));
                    ds[nr] = vload_half2(0, dp + blk);
                }
            }
            _Pragma("unroll")
            for (int hi = 0; hi < 2; ++hi) {
                ID_ALOAD(kb * 8 + 2 * j + hi)
                _Pragma("unroll")
                for (int nr = 0; nr < NR; ++nr) {
                    if (nr < nrv) {
                        const float ws = ID_WS(hi ? ds[nr].s1 : ds[nr].s0);
                        const int8 bq = hi ? id_iq4_b(wv[nr].s4, wv[nr].s5, wv[nr].s6, wv[nr].s7) : id_iq4_b(wv[nr].s0, wv[nr].s1, wv[nr].s2, wv[nr].s3);
                        ID_DPAS(nr, bq, ws)
                    }
                }
            }
        }
        ID_ACC
    }
    ID_STORE
}

// Q4_K, as kernel_mul_mm_q4_K_i8_sg8: a lane's 32 bytes hold one sub-block's codes in the low nibbles and the next
// one's in the high ones, in as s8 (code - 8), the 6-bit scale per sub-block, the constant part on one fp16 DPAS a block:
//   sum = xd (D sum_sb sc_sb ir_sb + sum_sb (8 D sc_sb - DM m_sb) S_sb)
#define ID_QK_S8(t) as_int((((t) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u)
static inline void id_scale_min_k4(int j, uint3 w, float * sc, float * m) {
    if (j < 4) {
        *sc = (float)((w.s0 >> (8 * j)) & 63);
        *m  = (float)((w.s1 >> (8 * j)) & 63);
    } else {
        const int jj = j - 4;
        const uint q8 = (w.s2 >> (8 * jj)) & 0xFF, lo = (w.s0 >> (8 * jj)) & 0xFF, hi = (w.s1 >> (8 * jj)) & 0xFF;
        *sc = (float)((q8 & 0xF) | ((lo >> 6) << 4));
        *m  = (float)((q8 >> 4) | ((hi >> 6) << 4));
    }
}

// the 6-bit scale of sub-block j alone, for the rolled K loop
static inline float id_scale_k4(int j, uint3 w) {
    if (j < 4) return (float)((w.s0 >> (8 * j)) & 63);
    const int jj = j - 4;
    return (float)(((w.s2 >> (8 * jj)) & 0xF) | ((((w.s0 >> (8 * jj)) & 0xFF) >> 6) << 4));
}

__attribute__((intel_reqd_sub_group_size(8)))
kernel void kernel_mul_mm_id_q4_K_i8_sg8(global const uchar * q, global const uchar * s, global const half * d, global const half * dm,
                                         global const half * xs, ID_ARGS) {
    ID_COLS
    const ulong rb = wr * nb;
    for (int kb = 0; kb < nb; ++kb) {
        ulong bo[NR];
        uint3 sw[NR];
        float Dv[NR];
        int8 bm[NR];     // the constant term's B operand of a set: its 8 scale-and-min halves, then zeros
        _Pragma("unroll")
        for (int nr = 0; nr < NR; ++nr) {
            bo[nr] = rb + (ulong)nr * 8 * nb + kb;
            sw[nr] = (uint3)(0u);
            Dv[nr] = 0.f;
            float DM = 0.f;
            if (nr < nrv) {
                sw[nr] = vload3(0, (global const uint *)(s + bo[nr] * 12));
                Dv[nr] = vload_half(0, d + bo[nr]);
                DM = vload_half(0, dm + bo[nr]);
            }
            half16 mv = (half16)(0.f);
            _Pragma("unroll")
            for (int sb = 0; sb < 8; ++sb) {
                float sc, mn;
                id_scale_min_k4(sb, sw[nr], &sc, &mn);
                mv[sb] = (half)(8.f * Dv[nr] * sc - DM * mn);
            }
            bm[nr] = as_int8(mv);
        }
        ID_XV
        for (int j = 0; j < 4; ++j) {
            uint8 wv[NR];
            _Pragma("unroll")
            for (int nr = 0; nr < NR; ++nr) {
                wv[nr] = (uint8)(0u);
                if (nr < nrv) wv[nr] = vload8(0, (global const uint *)(q + bo[nr] * 128 + j * 32));
            }
            _Pragma("unroll")
            for (int hi = 0; hi < 2; ++hi) {
                const int sb = 2 * j + hi;
                ID_ALOAD(kb * 8 + sb)
                _Pragma("unroll")
                for (int nr = 0; nr < NR; ++nr) {
                    if (nr < nrv) {
                        int8 bq;
                        bq.s0 = ID_QK_S8(hi ? wv[nr].s0 >> 4 : wv[nr].s0);
                        bq.s1 = ID_QK_S8(hi ? wv[nr].s1 >> 4 : wv[nr].s1);
                        bq.s2 = ID_QK_S8(hi ? wv[nr].s2 >> 4 : wv[nr].s2);
                        bq.s3 = ID_QK_S8(hi ? wv[nr].s3 >> 4 : wv[nr].s3);
                        bq.s4 = ID_QK_S8(hi ? wv[nr].s4 >> 4 : wv[nr].s4);
                        bq.s5 = ID_QK_S8(hi ? wv[nr].s5 >> 4 : wv[nr].s5);
                        bq.s6 = ID_QK_S8(hi ? wv[nr].s6 >> 4 : wv[nr].s6);
                        bq.s7 = ID_QK_S8(hi ? wv[nr].s7 >> 4 : wv[nr].s7);
                        const float ds = ID_WS(Dv[nr] * id_scale_k4(sb, sw[nr]));
                        ID_DPAS(nr, bq, ds)
                    }
                }
            }
        }
        _Pragma("unroll")
        for (int g = 0; g < NGM; ++g) {
            if (g < ng) {
                const int8 sa = as_int8(intel_sub_group_block_read8((global const uint *)(xs + ((tb + kb) * TMM + 8 * g) * 16)));
                _Pragma("unroll")
                for (int nr = 0; nr < NR; ++nr) {
                    if (nr < nrv) {
                        const float8 cst = intel_sub_group_f16_f16_matrix_mad_k16(sa, bm[nr], (float8)(0.f));
#ifdef ID_NOAF
                        acc[nr * NGM + g] = fma(cst, xv[g], acc[nr * NGM + g]);
#else
                        acc[nr * NGM + g] = fma(af[nr * NGM + g] + cst, xv[g], acc[nr * NGM + g]);
#endif
                    }
                }
            }
        }
    }
    ID_STORE
}
