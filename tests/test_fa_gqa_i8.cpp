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
//  - (b) the int8-QK selftest, modelled at the operands the kernel builds: the
//    staged a-operand tile (fa_gqa_i8_stage_rows' (row, block) pairs, poison
//    where the staging does not reach), the strided block read that gives lane
//    l dims (2l, 2l+1) of each row, the b operand as the lane's key (8 ints,
//    ascending dims) and the systolic dot4 pairing, then dq[row][block] *
//    dk[key][block]; against the f16 form over one 16-key tile, at the agent's
//    and the coder's row counts. The stated bound is max abs (int8 - f16)
//    <= 1e-2 * max abs S.
//
// Red mutations, each run and seen failing before it was restored:
//  (a) fa_gqa_i8_qs reads 34 * b + 18 instead of + 2 (the block's second half
//      as its first): the offset cell and the S selftest fail;
//  (b) FA_GQA_I8_RINT truncated to (int) x: the RNE corner cell fails;
//  (c) fa_gqa_i8_stage_rows returning the tile count (the pre-fix extent,
//      `tiles` without `* 8`): the S cells fail on every row the staging did
//      not reach -- the card's 135 of 135 FLASH_ATTN_EXT cases at ERR ~0.10.
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

// (b): the int8-QK selftest, modelled at the operands the kernel builds.
//
// The model is the kernel's own text, not an idealised dot product:
//  - the Q staging (fa_gqa_i8_stage_rows' (row, block) pairs) writes a row-major
//    [row][32 codes] a-operand tile, [tile][block][row][8 dwords], rows past
//    n_rows zeroed with dq 0 -- everything it does not reach holds the local
//    allocation's bytes (here a poison pattern, as the card's were);
//  - the a read is the DPAS layout: lane l, component (row) r is the strided
//    `intel_sub_group_block_read_us8` of the tile, bytes 2l + 32r, i.e. dims
//    (2l, 2l+1) of row r;
//  - the b operand is the lane's key: 8 ints, dims 4j..4j+3 ascending;
//  - the systolic pairing is dot4 per stage d: the a bytes of lanes 2d, 2d+1
//    (dims 4d..4d+3 of the row) against the b dword d, then scaled by
//    dq[row][block] * dk[key][block].
// Against the f16 form (Q x the block's dequantized codes) over one 16-key tile,
// at the agent's and the coder's row counts: 6 and 8 (one tile), 36 and 48 (five
// and six tiles). The bound is max abs (int8 - f16) <= 1e-2 * max|S|.
//
// Red: fa_gqa_i8_stage_rows returning the tile count (the pre-fix extent,
// `tiles` without `* 8`) stages only rows 0..tiles-1; every later row keeps the
// poison and its S fails the bound -- the card's 135 of 135 cases at ERR ~0.10.
static void test_s_tile(int n_rows, const char * name) {
    const int D = 256, QB = D / 32, KEYS = 16, TILES = 8, ROWS = TILES * 8;
    const float scale = 0.7f;

    // Q rows: rows past n_rows are the f16 path's zero row and the staged path's
    // zero codes with dq 0
    std::vector<float> q((size_t) ROWS * D, 0.0f);
    for (int r = 0; r < n_rows; ++r)
        for (int i = 0; i < D; ++i) q[(size_t) r * D + i] = (next_u01() * 2.0f - 1.0f) * scale;

    // the keys as q8_0 rows, at the helper's offsets
    std::vector<float> kd(KEYS * QB);
    std::vector<signed char> kc((size_t) KEYS * QB * 32);
    std::vector<unsigned char> krow(KEYS * 272 + 64, 0);
    for (int k = 0; k < KEYS; ++k) {
        for (int b = 0; b < QB; ++b) {
            kd[k * QB + b] = 0.005f + 0.015f * next_u01();
            for (int i = 0; i < 32; ++i) {
                const int c = (int) (next_u01() * 254.0f) - 127;
                kc[(size_t) (k * QB + b) * 32 + i] = (signed char) c;
                krow[(size_t) k * 272 + fa_gqa_i8_qs(b) + i] = (unsigned char) c;
            }
        }
    }

    // the local a-operand as the kernel builds it: poison everywhere, then the
    // staging loop's pairs
    const unsigned char POISON = 0x5A;   // as a signed char: 90
    const float POISON_DQ = 1.0f;
    std::vector<unsigned char> tile((size_t) TILES * QB * 8 * 32, POISON);   // [tile][block][row][32]
    std::vector<float> dqs((size_t) TILES * QB * 8, POISON_DQ);              // [tile][block][row]
    const int stage_rows = fa_gqa_i8_stage_rows(n_rows);
    for (int r = 0; r < stage_rows; ++r) {
        for (int b = 0; b < QB; ++b) {
            float x[32];
            for (int i = 0; i < 32; ++i) x[i] = r < n_rows ? q[(size_t) r * D + b * 32 + i] : 0.0f;
            float dq; int qc[32];
            fa_gqa_i8_quant32(x, &dq, qc);
            const int t = r / 8, rr = r % 8;
            dqs[(size_t) (t * QB + b) * 8 + rr] = dq;
            unsigned char * row = &tile[(size_t) ((t * QB + b) * 8 + rr) * 32];
            for (int i = 0; i < 32; ++i) row[i] = (unsigned char) (qc[i] & 0xFF);
        }
    }

    // S from the operands: per (tile, block) the four tiles a call, 16 keys a
    // group; component r is Q row tile*8 + r, lane l the key g*16 + l
    double s_f16[ROWS][KEYS] = {};
    double s_i8[ROWS][KEYS] = {};
    double max_abs_s = 0.0;
    for (int tile_i = 0; tile_i < (n_rows + 7) / 8; ++tile_i) {
        for (int b = 0; b < QB; ++b) {
            // the a operand read: lane l's component r = dims (2l, 2l+1) of row r
            int a_el[8][16][2];   // [row][lane][the short's two bytes]
            for (int r = 0; r < 8; ++r) {
                const unsigned char * row = &tile[(size_t) ((tile_i * QB + b) * 8 + r) * 32];
                for (int l = 0; l < 16; ++l) {
                    a_el[r][l][0] = row[2 * l];        // low byte: dim 2l
                    a_el[r][l][1] = row[2 * l + 1];    // high byte: dim 2l+1
                }
            }
            for (int k = 0; k < KEYS; ++k) {
                const float dk = kd[k * QB + b];
                int b_el[8][4];   // [stage][the dword's four bytes]
                for (int j = 0; j < 8; ++j) {
                    const unsigned char * word = &krow[(size_t) k * 272 + fa_gqa_i8_qs(b) + j * 4];
                    for (int m = 0; m < 4; ++m) b_el[j][m] = word[m];   // dims 4j..4j+3
                }
                for (int r = 0; r < 8; ++r) {
                    // the systolic pairing: stage d takes the shorts of lanes 2d,
                    // 2d+1 (dims 4d..4d+3) against the b dword d
                    int sum = 0;
                    for (int d = 0; d < 8; ++d) {
                        const int av[4] = { a_el[r][2 * d][0], a_el[r][2 * d][1],
                                            a_el[r][2 * d + 1][0], a_el[r][2 * d + 1][1] };
                        for (int m = 0; m < 4; ++m) {
                            sum += (int) (signed char) av[m] * (int) (signed char) b_el[d][m];
                        }
                    }
                    s_i8[tile_i * 8 + r][k] += (double) dqs[(size_t) (tile_i * QB + b) * 8 + r] * dk * sum;

                    double sf = 0.0;
                    for (int i = 0; i < 32; ++i) {
                        sf += (double) q[(size_t) (tile_i * 8 + r) * D + b * 32 + i] * dk *
                              (double) kc[(size_t) (k * QB + b) * 32 + i];
                    }
                    s_f16[tile_i * 8 + r][k] += sf;
                }
            }
        }
    }
    for (int r = 0; r < ROWS; ++r)
        for (int k = 0; k < KEYS; ++k)
            max_abs_s = std::fmax(max_abs_s, std::fabs(s_f16[r][k]));

    // every row the batch uses (0..n_rows-1), not only the staged ones: an
    // unstaged row is what the kernel computes with, and it must agree
    double max_err = 0.0;
    for (int r = 0; r < n_rows; ++r) {
        for (int k = 0; k < KEYS; ++k) {
            max_err = std::fmax(max_err, std::fabs(s_i8[r][k] - s_f16[r][k]));
        }
    }
    const double bound = 1e-2 * max_abs_s;
    std::printf("S tile n_rows %d (%s): stage rows %d, max|S| %.6f, max abs (int8 - f16) %.6f, relative %.4f %%\n",
                n_rows, name, stage_rows, max_abs_s, max_err, 100.0 * max_err / (max_abs_s > 0 ? max_abs_s : 1.0));
    if (max_err > bound) {
        std::printf("FAIL s-tile %s: max abs error %.6f above the stated bound 1e-2 * max|S| = %.6f\n",
                    name, max_err, bound);
        ++g_bad;
        return;
    }
    // the staging must cover every row of every tile the batch needs
    if (stage_rows < ((n_rows + 7) / 8) * 8) {
        std::printf("FAIL s-tile %s: the staging covers %d rows, the tiles need %d\n",
                    name, stage_rows, ((n_rows + 7) / 8) * 8);
        ++g_bad;
        return;
    }
    ok(name);
}

int main() {
    test_offsets();
    test_quantizer();
    // the agent's and the coder's row counts: one tile (6, 8), five and six (36, 48)
    test_s_tile(6,  "s-tile: agent decode, 6 rows");
    test_s_tile(8,  "s-tile: coder decode, 8 rows");
    test_s_tile(36, "s-tile: agent verify, 36 rows, five tiles");
    test_s_tile(48, "s-tile: coder verify, 48 rows, six tiles");
    if (g_bad == 0) std::printf("test-fa-gqa-i8: all OK\n");
    return g_bad == 0 ? 0 : 1;
}
