// Patch 0088/0089's device-free cell: the int8 Q K^T's arithmetic and block
// layout, read out of the kernel's own text (fa_gqa_i8_ops.inc, which the
// backend prepends to flash_attn_gqa_dpas.cl when the arm is built), so a
// mutation of the helper is a mutation of the kernel.
//
// What it checks:
//  - the q8_0 block offsets (34 bytes a block, codes at +2, the second half
//    behind byte 16, the staged row 68 / 34 dwords and its odd stride);
//  - the Q quantizer: one RNE scale per 32-block (amax / 127), the 32 codes
//    in dim order, round-to-nearest-even (the .5 corners), against an
//    independent nearbyint reference;
//  - (b) the int8-QK selftest: S over a 16-key tile and 8 query rows computed
//    the int8 way (integer contraction x dq[row][block] * dk[key][block]) and
//    the f16 way (exact Q x dequantized K), at head size 256; the stated bound
//    is max abs (int8 - f16) <= 1e-2 * max abs S, and the integer contraction
//    is exact against a double evaluation of the same codes.
//
// Red mutations, each run and seen failing before it was restored:
//  (a) fa_gqa_i8_qs reads 34 * b + 18 instead of + 2 (the block's second half
//      as its first): the offset cell and the S selftest fail;
//  (b) FA_GQA_I8_RINT truncated to (int) x: the RNE corner cell fails.
// The hardware arms ((a) every new FLASH_ATTN_EXT case at 1-6 rows, (b) DPAS
// against f16 DPAS on the card) are the card window's, commands in the README.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#include "fa_gqa_i8_ops.inc"

static int g_bad = 0;

static void fail(const char * what) {
    std::printf("FAIL %s\n", what);
    ++g_bad;
}

static void ok(const char * what) {
    std::printf("ok %s\n", what);
}

static uint32_t g_rng = 0x9e3779b9u;
static uint32_t next_u32() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
// [0, 1)
static float next_u01() {
    return (float) (next_u32() & 0xFFFFFFu) / 16777216.0f;
}

static void test_offsets() {
    for (int b : {0, 1, 3, 7}) {
        if (fa_gqa_i8_blk(b) != 34 * b) { fail("offset: block start"); return; }
        if (fa_gqa_i8_qs(b) != 34 * b + 2) { fail("offset: codes start"); return; }
    }
    if (fa_gqa_i8_half(0) != 16 || fa_gqa_i8_half(15) != 31) { fail("offset: second half"); return; }
    if (fa_gqa_i8_row_dw(256) != 68 || fa_gqa_i8_row_dw(128) != 34) { fail("offset: row dwords"); return; }
    if (fa_gqa_i8_stride_dw(256) != 69 || fa_gqa_i8_stride_dw(128) != 35) { fail("offset: staged stride"); return; }
    if (fa_gqa_i8_stride_dw(256) % 2 == 0) { fail("offset: stride must be odd"); return; }
    ok("offsets: q8_0 block, row and staged stride");
}

static void test_quantizer() {
    // the RNE corners, scale 1: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, -1.5 -> -2
    struct { float x; int want; } corners[] = {
        {0.5f, 0}, {1.5f, 2}, {2.5f, 2}, {3.5f, 4},
        {-0.5f, 0}, {-1.5f, -2}, {-2.5f, -2}, {-3.5f, -4},
    };
    for (auto c : corners) {
        if (fa_gqa_i8_code(c.x, 1.0f) != c.want) {
            std::printf("FAIL quant: %.1f -> %d, want %d (RNE)\n", c.x, fa_gqa_i8_code(c.x, 1.0f), c.want);
            ++g_bad;
            return;
        }
    }
    // random 32-blocks against an independent nearbyint reference
    for (int t = 0; t < 200; ++t) {
        float x[32];
        for (int i = 0; i < 32; ++i) x[i] = (next_u01() * 2.0f - 1.0f) * 0.7f;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[i]));
        const float d = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        float dq = -1.0f;
        int q[32];
        fa_gqa_i8_quant32(x, &dq, q);
        if (dq != d) { fail("quant: scale is not amax / 127"); return; }
        for (int i = 0; i < 32; ++i) {
            const int want = (int) std::nearbyint(x[i] * id);
            if (q[i] != want) {
                std::printf("FAIL quant: block %d dim %d code %d want %d\n", t, i, q[i], want);
                ++g_bad;
                return;
            }
            if (want > 127 || want < -127) { fail("quant: code outside int8"); return; }
        }
    }
    ok("quantizer: RNE corners, scale amax / 127, 32 codes in dim order");
}

// (b): the int8-QK selftest on one 16-key tile, 8 rows, head size 256.
static void test_s_tile() {
    const int D = 256, NB = D / 32, KEYS = 16, ROWS = 8;
    const float scale = 0.7f;
    std::vector<float> q(ROWS * D);
    for (auto & v : q) v = (next_u01() * 2.0f - 1.0f) * scale;

    // the keys as q8_0 rows: d and 32 code bytes a block, written at the
    // helper's offsets so the test reads what the kernel reads
    std::vector<float> kd(KEYS * NB);
    std::vector<signed char> kc(KEYS * NB * 32);
    // 64 bytes of padding past the rows: a block-offset mutation writes past a
    // row instead of past the buffer, so the cell fails instead of aborting
    std::vector<unsigned char> krow(KEYS * 272 + 64, 0);
    for (int k = 0; k < KEYS; ++k) {
        for (int b = 0; b < NB; ++b) {
            const float d = 0.005f + 0.015f * next_u01();
            kd[k * NB + b] = d;
            const int blk = k * 272 + fa_gqa_i8_blk(b);
            krow[blk] = 0;      // a half's bytes; the value is used from kd
            krow[blk + 1] = 0;
            for (int i = 0; i < 32; ++i) {
                const int c = (int) (next_u01() * 254.0f) - 127;
                kc[(k * NB + b) * 32 + i] = (signed char) c;
                krow[k * 272 + fa_gqa_i8_qs(b) + i] = (unsigned char) c;
            }
        }
    }

    double s_f16[ROWS][KEYS] = {};
    double s_i8[ROWS][KEYS] = {};
    double max_abs_s = 0.0;
    for (int r = 0; r < ROWS; ++r) {
        for (int b = 0; b < NB; ++b) {
            float x[32];
            for (int i = 0; i < 32; ++i) x[i] = q[r * D + b * 32 + i];
            float dq; int q[32];
            fa_gqa_i8_quant32(x, &dq, q);
            for (int k = 0; k < KEYS; ++k) {
                const float dk = kd[k * NB + b];
                const signed char * codes = &kc[(k * NB + b) * 32];
                // the int8 path: 32 codes in dim order against the raw block's
                // codes (read at the helper's offset; dim i + 16 is byte 16 + i)
                int sum = 0;
                for (int i = 0; i < 32; ++i) {
                    const signed char c = (signed char) krow[k * 272 + fa_gqa_i8_qs(b) + i];
                    sum += q[i] * c;
                }
                s_i8[r][k] += (double) dq * dk * sum;
                // the f16 path: the same Q values and the dequantized codes
                double sf = 0.0;
                for (int i = 0; i < 32; ++i) sf += (double) x[i] * dk * codes[i];
                s_f16[r][k] += sf;
            }
        }
    }
    for (int r = 0; r < ROWS; ++r)
        for (int k = 0; k < KEYS; ++k)
            max_abs_s = std::fmax(max_abs_s, std::fabs(s_f16[r][k]));

    double max_err = 0.0;
    for (int r = 0; r < ROWS; ++r) {
        for (int k = 0; k < KEYS; ++k) {
            max_err = std::fmax(max_err, std::fabs(s_i8[r][k] - s_f16[r][k]));
        }
    }
    std::printf("S tile: max|S| %.6f, max abs (int8 - f16) %.6f, relative %.4f %%\n",
                max_abs_s, max_err, 100.0 * max_err / max_abs_s);
    if (max_err > 1e-2 * max_abs_s) {
        std::printf("FAIL s-tile: max abs error %.6f above the stated bound 1e-2 * max|S| = %.6f\n",
                    max_err, 1e-2 * max_abs_s);
        ++g_bad;
        return;
    }
    ok("int8-QK selftest: S against the f16 form, inside 1e-2 * max|S|");
}

int main() {
    test_offsets();
    test_quantizer();
    test_s_tile();
    if (g_bad == 0) std::printf("test-fa-gqa-i8: all OK\n");
    return g_bad == 0 ? 0 : 1;
}
