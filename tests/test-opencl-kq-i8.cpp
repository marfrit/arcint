// 0071: device-free tests for the int8 DPAS expert GEMM's pieces.
//   - the s8 decode of IQ3_S and IQ4_NL against ggml's own dequantizers
//     (dequantize_row_iq3_s / dequantize_row_iq4_nl), value for value;
//   - the block indices of an element offset kb against the fp16 tile
//     kernel's own indexing (sub-block j32 = kb / 32, LB_BI / LB_P);
//   - the q8_1-style 32-value activation quantizer (scale and codes);
//   - the dispatch rule (env, sub-group, crossover), the per-format list,
//     and the program gate;
//   - the activation's column mapping (per column, not per route pair);
//   - the ASSEMBLED int8 program as load_cl_kernels builds it: the .inc, then
//     mul_mm_kq_f16.cl, then mul_mm_kq_i8.cl, compiled as OpenCL C when a
//     clang is present, and text-checked always (no C++-only token, no
//     reserved OpenCL type name used as an identifier).
// One text: the helpers are ggml/src/ggml-opencl/kernels/kq_i8_ops.inc, which
// the int8 kernel also gets (prepended by ggml-opencl.cpp), so a mutation here
// is a mutation of the kernel. No GPU, no OpenCL runtime.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <string>
#include <random>
#include <vector>
#include <algorithm>
#include <cstdint>

#include "ggml-quants.h"
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#undef GGML_COMMON_IMPL_CPP
#include "kq_i8_ops.inc"

static int g_bad = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL: %s\n", what); ++g_bad; } } while (0)

// the little-endian word at a plane offset, as the kernel reads it
static inline kq_u32 rd32(const uint8_t * p) { return kq_i8_u32(p); }

// ---- IQ3_S: one random block, its 256 values against ggml's dequantizer ----
static void check_iq3s(std::mt19937 & rng, int rounds) {
    for (int t = 0; t < rounds; ++t) {
        std::vector<uint8_t> raw(sizeof(block_iq3_s));
        for (auto & b : raw) b = (uint8_t)(rng() & 0xFF);
        // a sane fp16 d and a non-zero grid index: any bytes are valid codes,
        // the low byte of d is kept arbitrary (a half), the high byte sane
        block_iq3_s * blk = (block_iq3_s *)raw.data();
        const ggml_half dh = ggml_fp32_to_fp16(0.001f + 0.0001f * (rng() % 60));
        std::memcpy(&blk->d, &dh, 2);
        float ref[QK_K];
        dequantize_row_iq3_s(blk, ref, QK_K);
        // the plane form the kernel reads: qs plane 64, qh plane 8, signs 32,
        // scales 4, d at the front (one row, nblk blocks = 1)
        uint8_t d2[2], qs[64], qh[8], signs[32], scales[4];
        std::memcpy(d2, &blk->d, 2);
        std::memcpy(qs, blk->qs, 64);
        std::memcpy(qh, blk->qh, 8);
        std::memcpy(signs, blk->signs, 32);
        std::memcpy(scales, blk->scales, 4);
        const float d = ggml_fp16_to_fp32(blk->d);
        for (int p = 0; p < 8; ++p) {   // the block's eight 32-weight sub-blocks
            kq_u32 w[8];
            kq_i8_decode_iq3s(rd32(qs + p * 8), rd32(qs + p * 8 + 4), rd32(signs + p * 4), qh[p], iq3s_grid, w);
            const int sc = kq_i8_iq3s_sc(scales[p >> 1], p);
            const float ws = d * (float)(1 + 2 * sc);
            for (int j = 0; j < 8; ++j) {
                const uint8_t * b = (const uint8_t *)&w[j];
                for (int m = 0; m < 4; ++m) {
                    const int k = p * 32 + j * 4 + m;
                    const int8_t code = (int8_t)b[m];
                    const float got = ws * (float)code;
                    const float want = ref[k];
                    if (got != want) {
                        std::printf("FAIL: IQ3_S k %d decode %d want %g got %g\n", k, code, want, got);
                        ++g_bad;
                        return;
                    }
                }
            }
        }
    }
}

// ---- IQ4_NL: one random block, 32 values ----
static void check_iq4nl(std::mt19937 & rng, int rounds) {
    for (int t = 0; t < rounds; ++t) {
        std::vector<uint8_t> raw(sizeof(block_iq4_nl));
        for (auto & b : raw) b = (uint8_t)(rng() & 0xFF);
        block_iq4_nl * blk = (block_iq4_nl *)raw.data();
        const ggml_half dh = ggml_fp32_to_fp16(0.001f + 0.0001f * (rng() % 60));
        std::memcpy(&blk->d, &dh, 2);
        float ref[QK4_NL];
        dequantize_row_iq4_nl(blk, ref, QK4_NL);
        const float d = ggml_fp16_to_fp32(blk->d);
        uint32_t qd[4];
        std::memcpy(qd, blk->qs, 16);
        kq_u32 w[8];
        kq_i8_decode_iq4nl(qd[0], qd[1], qd[2], qd[3], kvalues_iq4nl, w);
        for (int j = 0; j < 8; ++j) {
            const uint8_t * b = (const uint8_t *)&w[j];
            for (int m = 0; m < 4; ++m) {
                const int k = j * 4 + m;
                const int8_t code = (int8_t)b[m];
                const float got = d * (float)code;
                const float want = ref[k];
                if (got != want) {
                    std::printf("FAIL: IQ4_NL k %d decode %d want %g got %g\n", k, code, want, got);
                    ++g_bad;
                    return;
                }
            }
        }
    }
}

// ---- the block indices against the fp16 path ----
// The kernel's block loop counts kb in ELEMENTS of K (k0 += 256, kb += 32).
// The fp16 tile kernel indexes its 32-sub-block j32 = kb / 32 and reads it
// through LB_BI(T, r, nbr, j32) = r * nbr + j32 / LB_SUBS(T) and
// LB_P(T, j32) = j32 % LB_SUBS(T); LB_SUBS(IQ3_S) = 8, so IQ3_S's super-block
// is j32 / 8 = kb / 256 and its p is j32 % 8 = (kb / 32) & 7; IQ4_NL's flat
// 32-block is j32 = kb / 32. Every kb of K 640 and 2560 is checked; the old
// formulas (kb / 8, kb & 7, kb) fail on the first kb past a block.
static void check_indices(void) {
    for (const int K : { 640, 2560 }) {
        const int n32 = K / 32;
        for (int kb = 0; kb < K; kb += 32) {
            const int j32 = kb / 32;                    // the fp16 path's sub-block
            const int sb  = j32 / 8;                    // LB_BI: j32 / LB_SUBS(IQ3_S)
            const int p   = j32 % 8;                    // LB_P: j32 % LB_SUBS(IQ3_S)
            CHECK(kq_i8_iq3s_sb(kb) == sb, "index: IQ3_S super-block of kb");
            CHECK(kq_i8_iq3s_p(kb) == p, "index: IQ3_S sub-block of kb");
            CHECK(kq_i8_iq4nl_blk(kb) == j32, "index: IQ4_NL 32-block of kb");
        }
        CHECK(n32 == K / 32, "index: n32");
    }
    // the last element offset of each K, named: K 640's last block is
    // super-block 2 (a partial one), K 2560's is 9 with p 7
    CHECK(kq_i8_iq3s_sb(640 - 32) == 2 && kq_i8_iq3s_p(640 - 32) == 3, "index: K 640's last kb");
    CHECK(kq_i8_iq3s_sb(2560 - 32) == 9 && kq_i8_iq3s_p(2560 - 32) == 7, "index: K 2560's last kb");
}

// ---- the activation quantizer ----
static void check_quant(void) {
    float x[32];
    int8_t q[32];
    float sc;
    for (int i = 0; i < 32; ++i) x[i] = 1.0f;
    kq_i8_quant32(x, q, &sc);
    CHECK(sc == 1.0f / 127.0f, "quant: scale = amax / 127");
    for (int i = 0; i < 32; ++i) CHECK(q[i] == 127, "quant: 1.0 -> 127");
    // round half away from zero (q8_1's roundf): the .5 cases are the red line
    const float v[6] = { 0.5f, 1.5f, -0.5f, -1.5f, 126.5f, -126.5f };
    const int  want[6] = { 1, 2, -1, -2, 127, -127 };
    for (int i = 0; i < 32; ++i) x[i] = 0.f;
    for (int i = 0; i < 6; ++i) x[i] = v[i];
    // amax = 126.5 -> d = 126.5/127, id = 127/126.5; a code is round(x*id)
    kq_i8_quant32(x, q, &sc);
    CHECK(sc == 126.5f / 127.0f, "quant: scale from the .5 max");
    for (int i = 0; i < 6; ++i) CHECK(q[i] == want[i], "quant: round half away from zero");
    for (int i = 6; i < 32; ++i) CHECK(q[i] == 0, "quant: zero stays zero");
    // y exactly at a half integer (amax 127 -> id 1): 0.5 -> 1, -0.5 -> -1.
    // round-to-nearest-even gives 0 for both; this is the .5 red line.
    for (int i = 0; i < 32; ++i) x[i] = 0.f;
    x[0] = 0.5f; x[1] = 127.0f; x[2] = -0.5f; x[3] = -1.5f; x[4] = 126.5f; x[5] = -126.5f;
    kq_i8_quant32(x, q, &sc);
    CHECK(sc == 1.0f, "quant: amax 127 has scale 1");
    CHECK(q[0] == 1 && q[1] == 127 && q[2] == -1 && q[3] == -2, "quant: .5 away from zero at scale 1");
    CHECK(q[4] == 127 && q[5] == -127, "quant: 126.5 -> 127, -126.5 -> -127");
    // a block below 1e-30 quantizes to zeros with scale 0
    for (int i = 0; i < 32; ++i) x[i] = 0.f;
    kq_i8_quant32(x, q, &sc);
    CHECK(sc == 0.f, "quant: an all-zero block has scale 0");
    for (int i = 0; i < 32; ++i) CHECK(q[i] == 0, "quant: an all-zero block is zeros");
}

// ---- the dispatch rule ----
// this patch keeps int8 opt-in: unset is 0, auto and 1 only by the switch. The
// expected values here flip in the commit that makes auto the default.
static void check_dispatch(void) {
    CHECK(kq_i8_mode(nullptr, 16, 1) == 0, "unset -> off (this patch)");
    CHECK(kq_i8_mode("", 16, 1) == 0, "empty -> off (this patch)");
    CHECK(kq_i8_mode(nullptr, 8, 1) == 0, "SG 8 -> fp16");
    CHECK(kq_i8_mode(nullptr, 16, 0) == 0, "no kernels -> fp16");
    CHECK(kq_i8_mode("0", 16, 1) == 0, "=0 off");
    CHECK(kq_i8_mode("1", 16, 1) == 1, "=1 forced on SG 16 with the kernels");
    CHECK(kq_i8_mode("1", 8, 1) == 0, "dispatch: switch on, SG 8 -> fp16");
    CHECK(kq_i8_mode("1", 16, 0) == 0, "dispatch: switch on, no kernels -> fp16");
    CHECK(kq_i8_mode("auto", 16, 1) == 2, "=auto");
    CHECK(kq_i8_mode("auto", 8, 1) == 0, "dispatch: auto on SG 8 -> fp16");
    CHECK(kq_i8_mode("Auto", 16, 1) == 2, "=Auto");
    CHECK(kq_i8_mode("2", 16, 1) == 1, "any other number forces int8");
    CHECK(kq_i8_mode("off", 16, 1) == 0, "an unparsable word is 0");
    CHECK(kq_i8_mean_pairs(10240, 512) == 20, "10,240 pairs / 512 experts = 20");
    CHECK(kq_i8_mean_pairs(81920, 512) == 160, "81,920 pairs / 512 experts = 160");
    CHECK(kq_i8_mean_pairs(160, 512) == 1, "experts bound by the pairs: 160 pairs, 160 experts");
    CHECK(kq_i8_mean_pairs(0, 512) == 0, "no pairs -> 0");
    CHECK(kq_i8_auto_use(20, 128) == 1, "20 rows per expert: int8");
    CHECK(kq_i8_auto_use(128, 128) == 1, "at the crossover: int8");
    CHECK(kq_i8_auto_use(129, 128) == 0, "above the crossover: fp16");
    CHECK(kq_i8_auto_use(160, 128) == 0, "160 rows (an 8k chunk): fp16");
}

// ---- the per-format list ----
// GGML_OPENCL_KQ_MM_ID_I8_FMT: a comma list of iq3_s and iq4_nl; unset is
// both. A format not named takes the fp16 tile, so iq3_s-only makes IQ4_NL
// fp16 and the reverse. The dispatch conjunction multiplies this by
// kq_i8_mode, so the arm off stays off for every format.
static void check_formats(void) {
    CHECK(kq_i8_fmt_allowed(nullptr, 0) == 1 && kq_i8_fmt_allowed(nullptr, 1) == 1,
          "fmt: unset -> both");
    CHECK(kq_i8_fmt_allowed("", 0) == 1 && kq_i8_fmt_allowed("", 1) == 1,
          "fmt: empty -> both");
    CHECK(kq_i8_fmt_allowed(" ", 0) == 1 && kq_i8_fmt_allowed(" ", 1) == 1,
          "fmt: whitespace-only -> both");
    CHECK(kq_i8_fmt_allowed("\t", 0) == 1 && kq_i8_fmt_allowed("\t", 1) == 1,
          "fmt: a tab-only value -> both");
    CHECK(kq_i8_fmt_allowed("iq3_s", 0) == 1 && kq_i8_fmt_allowed("iq3_s", 1) == 0,
          "fmt: iq3_s only -> IQ4_NL fp16");
    CHECK(kq_i8_fmt_allowed("iq4_nl", 0) == 0 && kq_i8_fmt_allowed("iq4_nl", 1) == 1,
          "fmt: iq4_nl only -> IQ3_S fp16");
    CHECK(kq_i8_fmt_allowed("iq4_nl,iq3_s", 0) == 1 && kq_i8_fmt_allowed("iq4_nl,iq3_s", 1) == 1,
          "fmt: both named -> both");
    CHECK(kq_i8_fmt_allowed("iq3_s , iq4_nl", 0) == 1 && kq_i8_fmt_allowed("iq3_s , iq4_nl", 1),
          "fmt: spaces around the commas are trimmed");
    CHECK(kq_i8_fmt_allowed("iq3_s iq4_nl", 0) == 1 && kq_i8_fmt_allowed("iq3_s iq4_nl", 1) == 1,
          "fmt: whitespace is a separator too");
    CHECK(kq_i8_fmt_allowed("iq3_s\tiq4_nl", 0) == 1 && kq_i8_fmt_allowed("iq3_s\tiq4_nl", 1) == 1,
          "fmt: a tab separates tokens too");
    CHECK(kq_i8_fmt_allowed("\n", 0) == 1 && kq_i8_fmt_allowed("\n", 1) == 1,
          "fmt: a newline-only value -> both");
    CHECK(kq_i8_fmt_allowed("\r", 0) == 1 && kq_i8_fmt_allowed("\r", 1) == 1,
          "fmt: a carriage-return-only value -> both");
    CHECK(kq_i8_fmt_allowed("iq3_s\niq4_nl", 0) == 1 && kq_i8_fmt_allowed("iq3_s\niq4_nl", 1) == 1,
          "fmt: a newline separates tokens too");
    CHECK(kq_i8_fmt_allowed("IQ3_S", 0) == 1 && kq_i8_fmt_allowed("IQ3_S", 1) == 0,
          "fmt: names are case-insensitive (iq3_s)");
    CHECK(kq_i8_fmt_allowed("IQ4_NL", 0) == 0 && kq_i8_fmt_allowed("IQ4_NL", 1) == 1,
          "fmt: names are case-insensitive (iq4_nl)");
    CHECK(kq_i8_fmt_allowed(",", 0) == 1 && kq_i8_fmt_allowed(",", 1) == 1,
          "fmt: a separators-only value -> both");
    CHECK(kq_i8_fmt_allowed(" , , ", 0) == 1 && kq_i8_fmt_allowed(" , , ", 1) == 1,
          "fmt: spaces and commas only -> both");
    CHECK(kq_i8_fmt_allowed("iq2_xs", 0) == 0 && kq_i8_fmt_allowed("iq2_xs", 1) == 0,
          "fmt: an unlisted format takes fp16");
    CHECK(kq_i8_fmt_allowed("iq3_sx", 0) == 0, "fmt: a prefix is not a token");
    CHECK(kq_i8_fmt_allowed("iq4_nl,", 1) == 1 && kq_i8_fmt_allowed("iq4_nl,", 0) == 0,
          "fmt: a trailing comma is fine");
}

// ---- the int8 program is built only when the arm is on ----
// unset and 0 are off: the default path must not pay a second program compile a
// unit start. The gate reads the same switch the dispatch reads.
static void check_program(void) {
    CHECK(kq_i8_program_wanted(nullptr) == 0, "program: unset -> not built");
    CHECK(kq_i8_program_wanted("") == 0, "program: empty -> not built");
    CHECK(kq_i8_program_wanted("0") == 0, "program: =0 -> not built");
    CHECK(kq_i8_program_wanted("off") == 0, "program: an unparsable word -> not built");
    CHECK(kq_i8_program_wanted("1") == 1, "program: =1 -> built");
    CHECK(kq_i8_program_wanted("auto") == 1, "program: auto -> built");
}

// ---- the activation's column mapping ----
// the fp16 tile kernel's column, (i % ne11) + t * ne11, not the route pair: a
// broadcast b (ne11 = 1, the served gate/up) holds one activation a token.
static void check_columns(void) {
    CHECK(kq_i8_col(0, 10, 1) == 0, "col: slot 0 of token 0, broadcast b");
    CHECK(kq_i8_col(9, 10, 1) == 0, "col: slot 9 of token 0 shares the column");
    CHECK(kq_i8_col(10, 10, 1) == 1, "col: token 1's first slot is column 1");
    CHECK(kq_i8_col(13, 10, 2) == 3, "col: p 13 -> t 1, i 3, ne11 2 -> 3");
    CHECK(kq_i8_col(0, 10, 5) == 0 && kq_i8_col(7, 10, 5) == 2, "col: ne11 5 -> i % 5");
    // the served gate/up at an 8k chunk, K 2560: per column, not per pair
    const size_t n_cols = 8192, n_pairs = 10 * n_cols, k = 2560;
    CHECK(n_cols * k == 20971520, "col: 8k chunk, K 2560 -> 21 MB of int8 codes a column");
    CHECK(n_pairs * k == 209715200, "col: a per-pair buffer is 210 MB");
    CHECK(2 * n_cols * k == 41943040, "col: the fp16 path's activation is 42 MB");
}

// ---- the assembled int8 program is OpenCL C text too ----
// load_cl_kernels assembles it as kq_i8_ops.inc + mul_mm_kq_f16.cl (the weight
// tables the int8 kernel reads) + mul_mm_kq_i8.cl; this test compiles the same
// text, with a prelude for the Intel subgroup builtins clang's OpenCL header
// does not declare for the spir64 target (the device compiler does). The tests
// read the three files by their paths, so a mutation in any of them is the
// program's mutation.

// the Intel builtins the text uses, by the text's own types (as_short8 of a
// ushort8 block read, the s8 and f16 DPAS); the card compile gate needs the
// same declarations with Intel's compiler.
static const char * const kq_i8_prelude =
    "// prelude: the Intel subgroup builtins clang's OpenCL header leaves out\n"
    "ushort8 intel_sub_group_block_read_us8(const __local ushort * p);\n"
    "uint8   intel_sub_group_block_read8(const __local uint * p);\n"
    "int     intel_sub_group_shuffle_xor(int x, uint lane);\n"
    "int8    intel_sub_group_i8_i8_matrix_mad_k32(short8 a, int8 b, int8 c);\n"
    "__attribute__((overloadable)) float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 c);\n"
    "__attribute__((overloadable)) float8 intel_sub_group_f16_f16_matrix_mad_k16(int8 a, int8 b, float8 c);\n";

// every overloaded prelude declaration carries __attribute__((overloadable)):
// a clang front end (clang-18 was seen) rejects two same-name declarations
// without it as "conflicting types", which failed this test and exited 1. A
// name declared twice is the overload set; each line of it must say so. Red:
// dropping the attribute from one line fails here without needing a clang.
// the declaration without its attribute clauses, so the first '(' is the
// parameter list's and the token before it is the function's name
static std::string without_attrs(std::string d) {
    size_t p;
    while ((p = d.find("__attribute__")) != std::string::npos) {
        const size_t e = d.find("))", p);
        d.erase(p, e == std::string::npos ? std::string::npos : e + 2 - p);
    }
    return d;
}

static void check_prelude_overloads(void) {
    std::vector<std::string> lines;
    size_t at = 0;
    while (at < std::strlen(kq_i8_prelude)) {
        const size_t nl = std::strchr(kq_i8_prelude + at, '\n') - kq_i8_prelude;
        lines.emplace_back(kq_i8_prelude + at, nl - at);
        at = nl + 1;
    }
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string di = without_attrs(lines[i]);
        const size_t par = di.find('(');
        if (par == std::string::npos) continue;
        size_t b = par;
        while (b > 0 && (std::isalnum((unsigned char)di[b - 1]) || di[b - 1] == '_')) --b;
        const std::string name = di.substr(b, par - b);
        int n = 0;
        for (const std::string & l : lines) {
            const std::string dl = without_attrs(l);
            const size_t p2 = dl.find('(');
            if (p2 == std::string::npos) continue;
            size_t b2 = p2;
            while (b2 > 0 && (std::isalnum((unsigned char)dl[b2 - 1]) || dl[b2 - 1] == '_')) --b2;
            if (dl.substr(b2, p2 - b2) == name) ++n;
        }
        if (n > 1 && lines[i].find("overloadable") == std::string::npos) {
            std::printf("FAIL: opencl: overloaded prelude declaration %s has no overloadable attribute\n", name.c_str());
            ++g_bad;
        }
    }
    CHECK(true, "opencl: prelude overloads carry the overloadable attribute");
}

// nesting-aware: a conditional inside a kept branch is kept whole, and the
// __cplusplus branches are dropped with their own open/else/close directives, so
// a nested #if inside the host block can never end the skip early. The old
// scanner ended a skip on any #endif and emitted the initial block's #endif,
// which left the include guard unbalanced.
static std::string opencl_part(const std::string & text) {
    struct Frame { bool skip; bool ours; bool parent_skip; bool else_seen; };
    std::vector<Frame> st;
    bool skip = false;
    std::string out;
    size_t at = 0;
    while (at < text.size()) {
        const size_t nl = text.find('\n', at);
        const std::string line = text.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        const size_t q = line.find_first_not_of(" \t");
        const std::string d = q == std::string::npos ? std::string() : line.substr(q);
        const bool ours = d.rfind("#ifdef __cplusplus", 0) == 0 || d.rfind("#if defined(__cplusplus)", 0) == 0;
        const bool ifd = d.rfind("#if", 0) == 0;
        const bool els = d.rfind("#else", 0) == 0 || d.rfind("#elif", 0) == 0;
        const bool end = d.rfind("#endif", 0) == 0;
        if (ours) {
            st.push_back(Frame{ true, true, skip, false });
            skip = true;
        } else if (ifd) {
            st.push_back(Frame{ skip, false, skip, false });
            if (!skip) out += line, out += '\n';
        } else if (els) {
            if (!st.empty() && st.back().ours && !st.back().else_seen) {
                st.back().else_seen = true;
                skip = false;                       // the kept #else branch, directive dropped
            } else if (!(st.empty() ? skip : st.back().parent_skip)) {
                out += line, out += '\n';
            }
        } else if (end) {
            const bool emit = !st.empty() && !st.back().ours && !st.back().parent_skip;
            const bool parent = st.empty() ? skip : st.back().parent_skip;
            if (!st.empty()) st.pop_back();
            skip = parent;
            if (emit) out += line, out += '\n';
        } else if (!skip) {
            out += line;
            out += '\n';
        }
        at = nl == std::string::npos ? text.size() : nl + 1;
    }
    return out;
}

// opencl_part's nesting and its empty-stack #else: a conditional inside the
// dropped host block keeps its open/else/close together; a top-level #else
// must read `skip`, not an empty stack. Red: the ternary inverted -> the last
// case reads st.back() on an empty vector and the test exits non-zero.
static void check_opencl_part(void) {
    const std::string h =
        "#ifdef __cplusplus\n"
        "#if 0\n"
        "hidden nested\n"
        "#else\n"
        "hidden else\n"
        "#endif\n"
        "#endif\n"
        "open body\n"
        "#if SG == 8\n"
        "sg8\n"
        "#else\n"
        "sg16\n"
        "#endif\n"
        "tail\n";
    const std::string cl = opencl_part(h);
    CHECK(cl.find("open body") != std::string::npos, "opencl_part: a kept line survives a nested host block");
    CHECK(cl.find("hidden") == std::string::npos, "opencl_part: nested host lines are dropped");
    CHECK(cl.find("#if SG == 8") != std::string::npos && cl.find("#else") != std::string::npos &&
          cl.find("#endif") != std::string::npos, "opencl_part: a kept conditional keeps its directives");
    CHECK(cl.find("sg8") != std::string::npos && cl.find("sg16") != std::string::npos,
          "opencl_part: kept conditional branches survive");
    const std::string t = opencl_part("#else\ntop\n");
    CHECK(t.find("top") != std::string::npos, "opencl_part: a top-level else is emitted without an empty-stack read");
}

// drop the comments, so a comment that names a type never reads as an identifier
static std::string strip_comments(const std::string & text) {
    std::string out;
    size_t at = 0;
    while (at < text.size()) {
        if (text[at] == '/' && at + 1 < text.size() && text[at + 1] == '/') {
            const size_t nl = text.find('\n', at);
            at = nl == std::string::npos ? text.size() : nl;   // keep the newline
        } else if (text[at] == '/' && at + 1 < text.size() && text[at + 1] == '*') {
            const size_t end = text.find("*/", at + 2);
            at = end == std::string::npos ? text.size() : end + 2;
            out += ' ';
        } else {
            out += text[at];
            ++at;
        }
    }
    return out;
}

// a reserved OpenCL type name used as an identifier: the form `name =` with a
// one-token initializer (`==` is not one; `name *`, `name xs[...]`, `const
// name * p` are the type's uses). Round 2's blocker was `const int j = unit >>
// 1, half = unit & 1;` in I8_STAGE; this catches it without a compiler.
static bool reserved_ident(const std::string & cl, const char * name) {
    const std::string nm(name);
    size_t at = 0;
    while ((at = cl.find(nm, at)) != std::string::npos) {
        const size_t e = at + nm.size();
        const bool left_ok  = at == 0 || !(std::isalnum((unsigned char)cl[at - 1]) || cl[at - 1] == '_');
        const bool right_ok = e >= cl.size() || !(std::isalnum((unsigned char)cl[e]) || cl[e] == '_');
        if (left_ok && right_ok) {
            size_t k = e;
            while (k < cl.size() && (cl[k] == ' ' || cl[k] == '\t')) ++k;
            if (k < cl.size() && cl[k] == '=' && (k + 1 >= cl.size() || cl[k + 1] != '=')) return true;
        }
        at = e;
    }
    return false;
}

static std::string read_text(const char * path, bool * ok) {
    *ok = false;
    std::FILE * f = std::fopen(path, "rb");
    if (f == nullptr) return std::string();
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    std::fclose(f);
    *ok = true;
    return text;
}

// the identifier as a whole token in cl text
static bool has_ident(const std::string & cl, const std::string & nm) {
    size_t at = 0;
    while ((at = cl.find(nm, at)) != std::string::npos) {
        const size_t e = at + nm.size();
        const bool left  = at == 0 || !(std::isalnum((unsigned char)cl[at - 1]) || cl[at - 1] == '_');
        const bool right = e >= cl.size() || !(std::isalnum((unsigned char)cl[e]) || cl[e] == '_');
        if (left && right) return true;
        at = e;
    }
    return false;
}

// clang, clang-18, clang-19, or $CLANG (first that runs); empty when none
static std::string find_clang(void) {
    const char * env = std::getenv("CLANG");
    if (env != nullptr && env[0] != 0 &&
        std::system((std::string(env) + " --version >/dev/null 2>&1").c_str()) == 0) {
        return env;
    }
    for (const char * c : { "clang", "clang-18", "clang-19" }) {
        if (std::system((std::string(c) + " --version >/dev/null 2>&1").c_str()) == 0) return c;
    }
    return std::string();
}

static void check_opencl_text(void) {
    bool ok_inc = false, ok_f16 = false, ok_i8 = false;
    const std::string inc = read_text(KQ_I8_INC_PATH, &ok_inc);
    const std::string f16 = read_text(KQ_I8_F16_PATH, &ok_f16);
    const std::string kr  = read_text(KQ_I8_KR_PATH,  &ok_i8);
    CHECK(ok_inc, "opencl: kq_i8_ops.inc is readable");
    CHECK(ok_f16, "opencl: mul_mm_kq_f16.cl is readable");
    CHECK(ok_i8,  "opencl: mul_mm_kq_i8.cl is readable");
    if (!ok_inc || !ok_f16 || !ok_i8) return;
    // the backend's assembly order: helpers, then the mm source's tables, then the kernel
    const std::string prog = inc + f16 + kr;
    const std::string cl = strip_comments(opencl_part(prog));
    // the .inc's OpenCL part must carry no string literal: the host-only block
    // (the dispatch rule and the format list) is behind #ifdef __cplusplus and
    // opencl_part drops it. Round 6's blocker was a literal there -> OpenCL C
    // sees __constant char *, an address-space change both compilers refused.
    const std::string inc_cl = strip_comments(opencl_part(inc));
    if (inc_cl.find('"') != std::string::npos) {
        std::printf("FAIL: opencl: a string literal in the .inc's OpenCL part (it is __constant char *)\n");
        ++g_bad;
    }
    CHECK(inc_cl.find('"') == std::string::npos, "opencl: no string literal outside the .inc's host-only block");
    // every kq_i8_* identifier the .cl kernels call must be defined in the
    // .inc's OpenCL part. Round 7's blocker: kq_i8_col, called by I8_STAGE,
    // sat in the host-only block, so clang at SG16 reported 4 uses of an
    // undeclared identifier. This check needs no clang.
    const std::string kr_cl = strip_comments(kr);
    std::vector<std::string> called;
    for (size_t at2 = 0; (at2 = kr_cl.find("kq_i8_", at2)) != std::string::npos; ) {
        size_t e2 = at2 + 6;
        while (e2 < kr_cl.size() && (std::isalnum((unsigned char)kr_cl[e2]) || kr_cl[e2] == '_')) ++e2;
        const std::string nm = kr_cl.substr(at2, e2 - at2);
        if (std::find(called.begin(), called.end(), nm) == called.end()) called.push_back(nm);
        at2 = e2;
    }
    CHECK(!called.empty(), "opencl: the kernels call kq_i8_* helpers");
    for (const std::string & nm : called) {
        const bool defined = has_ident(inc_cl, nm);
        if (!defined) std::printf("FAIL: opencl: the kernel calls %s, not defined in the .inc's OpenCL part\n", nm.c_str());
        CHECK(defined, "opencl: every kq_i8_* the kernels call is in the OpenCL part");
    }
    check_prelude_overloads();
    // the C++-only tokens a CL front end does not know; nullptr is the red line
    for (const char * tok : { "nullptr", "std::", "template<", "constexpr", "static_cast", "reinterpret_cast" }) {
        const bool bad = cl.find(tok) != std::string::npos;
        if (bad) std::printf("FAIL: opencl: C++-only token %s in the OpenCL part\n", tok);
        CHECK(!bad, "opencl: no C++-only token in the OpenCL part");
    }
    // a reserved type name used as an identifier (the round-2 blocker: `half`)
    for (const char * nm : { "half", "float", "double", "int", "char", "short", "long",
                             "uchar", "ushort", "uint", "ulong", "bool", "void" }) {
        const bool bad = reserved_ident(cl, nm);
        if (bad) std::printf("FAIL: opencl: reserved type name %s used as an identifier\n", nm);
        CHECK(!bad, "opencl: no reserved type name used as an identifier");
    }
    // compile the assembled program, when a clang exists; without one the text
    // checks are its pre-run (the device compile is the card gate's job)
    const std::string cc = find_clang();
    if (cc.empty()) {
        std::printf("note: no clang (clang, clang-18, clang-19, $CLANG); the assembled program's compile is skipped\n");
        return;
    }
    std::FILE * w = std::fopen(KQ_I8_ASSEMBLED_PATH, "wb");
    CHECK(w != nullptr, "opencl: the assembled program is writable");
    if (w == nullptr) return;
    const std::string all = std::string(kq_i8_prelude) + prog;
    std::fwrite(all.data(), 1, all.size(), w);
    std::fclose(w);
    for (const char * sg : { "16", "8" }) {
        const std::string cmd = cc + " -target spir64 -x cl -cl-std=CL3.0 -DSG=" + sg +
                               " -fsyntax-only \"" + KQ_I8_ASSEMBLED_PATH + "\" >/dev/null 2>&1";
        const int rc = std::system(cmd.c_str());
        if (rc != 0) std::printf("FAIL: opencl: the assembled int8 program does not compile at SG%s (%s)\n", sg, cc.c_str());
        CHECK(rc == 0, "opencl: the assembled int8 program compiles as OpenCL C");
    }
}

int main() {
    std::mt19937 rng(11);
    check_iq3s(rng, 400);
    check_iq4nl(rng, 400);
    check_indices();
    check_quant();
    check_dispatch();
    check_formats();
    check_program();
    check_columns();
    check_opencl_part();
    check_opencl_text();
    if (g_bad == 0) std::printf("test-opencl-kq-i8: all OK\n");
    return g_bad == 0 ? 0 : 1;
}
