// Patch 0220's device-free cell: the dense Q8_0 2D-block GEMM (GGML_OPENCL_Q8_2D).
//
//  1. The gate (ggml_cl_q8_2d_gate, ggml-opencl-q8-2d.h, the function the OpenCL dispatch
//     calls) on Inkling 2 Mini's exported prefill graph (tests/q8_2d_export_lines.inc:
//     test-export-graph-ops lines, verbatim, at ub 512, 1024 and 2048). It must TAKE every
//     Q8_0 product of 16 or more columns the graph has (wq / wqkv_gate 3072 x 2304, wo 2304 x
//     3072, wk / wv 1024 x 2304, attn_rel 384 x 2304, attn_sink_proj 24 x 2304, the k/v conv
//     projections 32 x 2304) at modes 1 and 2, take nothing at mode 0, and leave the one-column
//     lines and the head (Q6_K) alone; and it must refuse the out-of-scope variants built from
//     a real line: K 64 and K 2880, one and 15 columns, K x columns at 2^31, a view, a weight not in the flat
//     planes, the head's 201,024 rows as Q8_0, a batched weight, strided columns, a strided
//     destination, a 32-byte base alignment. Red first, on every run: the same checks against
//     wrong rules (K % 32 in place of % 256, no row cap, accept-all, refuse-all) must FAIL.
//
//  2. The launch shape (ggml_cl_q8_2d_shape): every row covered, no idle work-group, the
//     work-group sized to the rows (24 and 32 rows: 2 sub-groups; 384: two of 12), and the
//     red (a fixed 16-sub-group work-group, the WG macro's shape) must fail those.
//
//  3. A scalar model of the arithmetic at the served K (2304, 3072, 256): Q8_0 weights as
//     ggml quantizes them, the int8 arm's activations as kernel_quant_i8 makes them (int8 per
//     (token, 256), scale amax / 127, round to nearest even), the fp16 arm's as
//     kernel_cvt_f32_f16 (halves, a column scale only above 16,384) with the weights rounded
//     once to f16; both against the CPU backend's product (activations as Q8_0 per 32, an f16
//     scale), the comparison test-backend-ops makes, on its uniform [-1, 1] fill. The bound is
//     the suite's NMSE for MUL_MAT, 5e-4. The two red builds of the kernel (Q8_2D_RED 1, the
//     8th block's codes dropped; 2, every lane the scales of its sub-group's first row) are
//     modelled too, and 3 (the int8 kernel: every token of a 32-token tile the activation
//     scale of the tile's first token), so the card's reds have a predicted size. On the
//     uniform fill every scale is nearly alike: red 1 is far above the bound, red 2 only
//     3-4x above it, and red 3 under it (asserted: the suite's uniform cases cannot see an
//     activation-scale error). On test_mul_mat_spread's fill (weights per (row, 32-block),
//     activations per (token, 256-block), eight magnitudes) all three must be far above it
//     and both green arms under it. This is a model of the arithmetic, not of the kernel's
//     text: the kernel is checked by the op cases on the card.
#include "ggml-opencl-q8-2d.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int  g_failures = 0;
static bool g_quiet    = false;

#define CHECK(cond, ...) do { \
        if (!(cond)) { if (!g_quiet) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } ++g_failures; } \
    } while (0)

// ---------------------------------------------------------------------------------------
// 1. the gate on the export lines
// ---------------------------------------------------------------------------------------
struct export_raw {
    const char * file;
    int          line;
    const char * text;
};
static const export_raw k_export_raw[] = {
#include "q8_2d_export_lines.inc"
};

struct export_line {
    const char * file;
    int          line;
    int          op, type;
    int64_t      ne[4];
    int          src_type[2];
    int64_t      src_ne[2][4];
    uint64_t     src_nb[2][4];
    std::string  name;
};

static std::vector<export_line> parse_export() {
    std::vector<export_line> out;
    for (const export_raw & r : k_export_raw) {
        std::istringstream in(r.text);
        export_line l = {};
        l.file = r.file;
        l.line = r.line;
        int n_params = 0, n_src = 0;
        in >> l.op >> l.type >> l.ne[0] >> l.ne[1] >> l.ne[2] >> l.ne[3] >> n_params;
        for (int i = 0; i < n_params; ++i) {
            int64_t p;
            in >> p;
        }
        in >> n_src;
        if (!in || n_src != 2) {
            printf("FAIL: unparsed export line %s:%d\n", r.file, r.line);
            ++g_failures;
            continue;
        }
        for (int s = 0; s < 2; ++s) {
            in >> l.src_type[s];
            for (int i = 0; i < 4; ++i) in >> l.src_ne[s][i];
            for (int i = 0; i < 4; ++i) in >> l.src_nb[s][i];
        }
        in >> l.name;
        if (!in) {
            printf("FAIL: unparsed export line %s:%d\n", r.file, r.line);
            ++g_failures;
            continue;
        }
        out.push_back(l);
    }
    return out;
}

// the gate's input as the dispatch builds it from the nodes: the export's types, shapes and
// strides; the destination ggml_mul_mat's own (F32, contiguous: ggml.c's ggml_mul_mat makes a
// new tensor); the weights are model tensors (inkling2.cpp's build_lora_mm(layer.wq, ...) and
// its siblings: no views), which set_tensor flattens (contiguous, not a view); the device's
// base alignment 128 bytes (any multiple of 64 passes)
static ggml_cl_q8_2d_in input_of(const export_line & l, int mode) {
    ggml_cl_q8_2d_in a = {};
    a.mode  = mode;
    a.type0 = l.src_type[0];
    a.type1 = l.src_type[1];
    a.typed = l.type;
    for (int i = 0; i < 4; ++i) {
        a.ne0[i] = l.src_ne[0][i];
        a.nb0[i] = (size_t) l.src_nb[0][i];
        a.ne1[i] = l.src_ne[1][i];
        a.nb1[i] = (size_t) l.src_nb[1][i];
    }
    a.nbd0      = sizeof(float);
    a.view0     = false;
    a.soa0      = a.type0 == GGML_CL_Q8_2D_TYPE_Q8_0;
    a.alignment = 128;
    a.max_rows  = 8192;
    return a;
}

typedef int (*rule_fn)(const ggml_cl_q8_2d_in &);

static int rule_real(const ggml_cl_q8_2d_in & a) { return ggml_cl_q8_2d_gate(a); }
static int rule_k32(const ggml_cl_q8_2d_in & a) { return ggml_cl_q8_2d_gate_k(a, 32); }
static int rule_nocap(const ggml_cl_q8_2d_in & a) {
    ggml_cl_q8_2d_in b = a;
    b.max_rows = (int64_t) 1 << 40;
    return ggml_cl_q8_2d_gate(b);
}
static int rule_all(const ggml_cl_q8_2d_in & a) {
    return a.mode != 0 && a.type0 == GGML_CL_Q8_2D_TYPE_Q8_0 ? GGML_CL_Q8_2D_TAKE : GGML_CL_Q8_2D_NOT_MINE;
}
static int rule_none(const ggml_cl_q8_2d_in &) { return GGML_CL_Q8_2D_LAYOUT; }

// a Q8_0 weight [k, m] and F32 columns [k, n], both contiguous, as the graph makes them
static ggml_cl_q8_2d_in make_input(int mode, int64_t k, int64_t m, int64_t n) {
    ggml_cl_q8_2d_in a = {};
    a.mode  = mode;
    a.type0 = GGML_CL_Q8_2D_TYPE_Q8_0;
    a.type1 = GGML_CL_Q8_2D_TYPE_F32;
    a.typed = GGML_CL_Q8_2D_TYPE_F32;
    const size_t row0 = (size_t) (k / 32) * 34;
    a.ne0[0] = k; a.ne0[1] = m; a.ne0[2] = 1; a.ne0[3] = 1;
    a.nb0[0] = 34; a.nb0[1] = row0; a.nb0[2] = row0 * (size_t) m; a.nb0[3] = a.nb0[2];
    a.ne1[0] = k; a.ne1[1] = n; a.ne1[2] = 1; a.ne1[3] = 1;
    a.nb1[0] = 4; a.nb1[1] = (size_t) k * 4; a.nb1[2] = a.nb1[1] * (size_t) n; a.nb1[3] = a.nb1[2];
    a.nbd0 = 4;
    a.soa0 = true;
    a.alignment = 128;
    a.max_rows = 8192;
    return a;
}

static int check_rule(const std::vector<export_line> & ex, rule_fn rule, bool quiet) {
    g_quiet = quiet;
    const int before = g_failures;
    int taken = 0;
    for (const export_line & l : ex) {
        CHECK(l.op == 29, "%s:%d is not MUL_MAT (op %d)", l.file, l.line, l.op);
        const bool q8 = l.src_type[0] == GGML_CL_Q8_2D_TYPE_Q8_0;
        const bool prompt = l.src_ne[1][1] >= 16;
        CHECK(rule(input_of(l, 0)) == GGML_CL_Q8_2D_OFF, "%s:%d %s taken or reported at mode 0", l.file, l.line, l.name.c_str());
        for (int mode : { 1, 2 }) {
            const int v = rule(input_of(l, mode));
            if (q8 && prompt) {
                CHECK(v == GGML_CL_Q8_2D_TAKE, "%s:%d %s [%lld x %lld, %lld columns] mode %d: refused (%s)", l.file, l.line,
                      l.name.c_str(), (long long) l.src_ne[0][1], (long long) l.src_ne[0][0], (long long) l.src_ne[1][1], mode,
                      ggml_cl_q8_2d_reason(v));
                taken += v == GGML_CL_Q8_2D_TAKE;
            } else {
                CHECK(v == GGML_CL_Q8_2D_NOT_MINE, "%s:%d %s mode %d: %s, want not-mine", l.file, l.line, l.name.c_str(), mode,
                      ggml_cl_q8_2d_reason(v));
            }
        }
    }
    // the export's Q8_0 prompt products: 6 shapes x 3 ubatch sizes, 2 modes
    CHECK(taken == 6 * 3 * 2, "%d of the export's 36 (line, mode) Q8_0 prompt products taken", taken);

    // the refusals, each from the 3072 x 2304 product at 1,024 tokens
    struct neg { const char * what; ggml_cl_q8_2d_in in; int want; };
    std::vector<neg> negs;
    negs.push_back({ "K 64 (the controllers' K)", make_input(2, 64, 2304, 1024), GGML_CL_Q8_2D_K });
    negs.push_back({ "K 2880 (% 32, not % 256)", make_input(2, 2880, 2880, 1024), GGML_CL_Q8_2D_K });
    negs.push_back({ "one column", make_input(2, 2304, 3072, 1), GGML_CL_Q8_2D_NOT_MINE });
    negs.push_back({ "15 columns", make_input(2, 2304, 3072, 15), GGML_CL_Q8_2D_NOT_MINE });
    negs.push_back({ "the head's 201,024 rows as Q8_0", make_input(2, 2304, 201024, 1024), GGML_CL_Q8_2D_ROWS });
    negs.push_back({ "8,193 rows", make_input(2, 2304, 8193, 1024), GGML_CL_Q8_2D_ROWS });
    negs.push_back({ "K 8,192 x 262,144 columns (2^31 values)", make_input(2, 8192, 1024, 262144), GGML_CL_Q8_2D_LIMIT });
    negs.push_back({ "K 8,192 x 262,143 columns", make_input(2, 8192, 1024, 262143), GGML_CL_Q8_2D_TAKE });
    { ggml_cl_q8_2d_in a = make_input(2, 2304, 3072, 1024); a.view0 = true;     negs.push_back({ "a view", a, GGML_CL_Q8_2D_VIEW }); }
    { ggml_cl_q8_2d_in a = make_input(2, 2304, 3072, 1024); a.soa0 = false;     negs.push_back({ "not in the flat planes", a, GGML_CL_Q8_2D_SOA }); }
    { ggml_cl_q8_2d_in a = make_input(2, 2304, 3072, 1024); a.alignment = 32;  negs.push_back({ "32-byte base alignment", a, GGML_CL_Q8_2D_ALIGN }); }
    { ggml_cl_q8_2d_in a = make_input(2, 2304, 3072, 1024); a.ne0[2] = 2; a.nb0[3] = a.nb0[2] * 2;
      negs.push_back({ "a batched weight", a, GGML_CL_Q8_2D_LAYOUT }); }
    { ggml_cl_q8_2d_in a = make_input(2, 2304, 3072, 1024); a.nb1[1] = 2 * a.nb1[1]; a.nb1[2] = a.nb1[1] * 1024; a.nb1[3] = a.nb1[2];
      negs.push_back({ "strided columns", a, GGML_CL_Q8_2D_LAYOUT }); }
    { ggml_cl_q8_2d_in a = make_input(2, 2304, 3072, 1024); a.nbd0 = 8;        negs.push_back({ "a strided destination", a, GGML_CL_Q8_2D_LAYOUT }); }
    { ggml_cl_q8_2d_in a = make_input(2, 2304, 3072, 1024); a.type1 = 1;       negs.push_back({ "F16 columns", a, GGML_CL_Q8_2D_NOT_MINE }); }
    { ggml_cl_q8_2d_in a = make_input(3, 2304, 3072, 1024);                    negs.push_back({ "mode 3", a, GGML_CL_Q8_2D_OFF }); }
    // and the edges that are taken
    negs.push_back({ "8,192 rows", make_input(2, 2304, 8192, 1024), GGML_CL_Q8_2D_TAKE });
    negs.push_back({ "K 256, 16 columns", make_input(1, 256, 24, 16), GGML_CL_Q8_2D_TAKE });
    negs.push_back({ "the bundle 6552 x 2304", make_input(2, 2304, 6552, 1024), GGML_CL_Q8_2D_TAKE });
    for (const neg & c : negs) {
        const int v = rule(c.in);
        CHECK(v == c.want, "%s: %s, want %s", c.what, ggml_cl_q8_2d_reason(v), ggml_cl_q8_2d_reason(c.want));
    }
    g_quiet = false;
    return g_failures - before;
}

// ---------------------------------------------------------------------------------------
// 2. the launch shape
// ---------------------------------------------------------------------------------------
typedef void (*shape_fn)(int64_t, int, int *, int *);
static void shape_fixed(int64_t rows, int max_sg, int * sg, int * nwg) {   // the red: WG sub-groups always
    *sg  = max_sg;
    *nwg = (int) ((rows + 16 * max_sg - 1) / (16 * max_sg));
}

static int check_shape(shape_fn f, bool quiet) {
    g_quiet = quiet;
    const int before = g_failures;
    for (int64_t rows = 1; rows <= 20000; ++rows) {
        int sg = 0, nwg = 0;
        f(rows, 16, &sg, &nwg);
        const int64_t nsg = (rows + 15) / 16;
        if (sg < 1 || sg > 16 || nwg < 1 || (int64_t) sg * nwg < nsg || (int64_t) sg * (nwg - 1) >= nsg ||
            (int64_t) sg * nwg - nsg >= nwg) {
            CHECK(false, "rows %lld: %d sub-groups x %d work-groups (rows need %lld sub-groups)", (long long) rows, sg, nwg, (long long) nsg);
            if (quiet) break;
        }
    }
    const struct { int64_t rows; int sg, nwg; } want[] = {
        { 24, 2, 1 }, { 32, 2, 1 }, { 384, 12, 2 }, { 1024, 16, 4 }, { 2304, 16, 9 }, { 3072, 16, 12 }, { 6552, 16, 26 },
    };
    for (const auto & w : want) {
        int sg = 0, nwg = 0;
        f(w.rows, 16, &sg, &nwg);
        CHECK(sg == w.sg && nwg == w.nwg, "rows %lld: %d x %d, want %d x %d", (long long) w.rows, sg, nwg, w.sg, w.nwg);
    }
    int sg = 0, nwg = 0;
    f(3072, 4, &sg, &nwg);   // a kernel that fits fewer sub-groups a work-group
    CHECK(sg == 4 && nwg == 48, "rows 3072 at 4 sub-groups: %d x %d, want 4 x 48", sg, nwg);
    g_quiet = false;
    return g_failures - before;
}

// ---------------------------------------------------------------------------------------
// 3. the arithmetic
// ---------------------------------------------------------------------------------------
// IEEE half, round to nearest even (normals, subnormals, overflow to inf)
static uint16_t f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    x &= 0x7FFFFFFFu;
    if (x >= 0x7F800000u) return (uint16_t) (sign | 0x7C00u | (x > 0x7F800000u ? 0x200u : 0));
    if (x >= 0x477FF000u) return (uint16_t) (sign | 0x7C00u);   // rounds past 65504
    if (x < 0x38800000u) {                                      // below the smallest normal half
        if (x < 0x33000000u) return (uint16_t) sign;            // below half the smallest subnormal
        // |f| = m 2^(e - 150) and a subnormal half's unit is 2^-24: r = round(m >> (126 - e))
        const uint32_t e = x >> 23, m = (x & 0x7FFFFFu) | 0x800000u;
        const uint32_t sh = 126 - e;   // 14 .. 24
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        uint32_t r = m >> sh;
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t) (sign | r);
    }
    uint32_t r = ((x >> 13) - (112u << 10));
    const uint32_t rem = x & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t) (sign | r);
}
static float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m | 0x400), (int) e - 25);
    uint32_t x;
    memcpy(&x, &v, 4);
    x |= sign;
    memcpy(&v, &x, 4);
    return v;
}
static float hround(float f) { return h2f(f2h(f)); }

// ggml's quantize_row_q8_0_ref: d = amax / 127 stored as f16, q = round(x / d)
static void quant_q8_0(const float * x, int k, int8_t * q, float * d) {
    for (int b = 0; b < k / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[32 * b + j]));
        const float dd = amax / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
        d[b] = hround(dd);
        for (int j = 0; j < 32; ++j) q[32 * b + j] = (int8_t) std::lround(x[32 * b + j] * id);
    }
}

struct model_out { double nmse_i8, nmse_f16, nmse_red1, nmse_red2, nmse_red3; };

// the products of one shape: m rows x k, n tokens. spread: test_mul_mat_spread's fill, weight row r,
// 32-block b scaled by 2^((r + b) % 8 - 4), activation token t, 256-block s by 2^((t + s) % 8 - 4)
static model_out model(int64_t m, int64_t k, int64_t n, bool spread, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> w(m * k), x(n * k);
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t i = 0; i < k; ++i) w[r * k + i] = u(rng) * (spread ? std::ldexp(1.f, (int) ((r + i / 32) % 8) - 4) : 1.f);
    }
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t i = 0; i < k; ++i) x[t * k + i] = u(rng) * (spread ? std::ldexp(1.f, (int) ((t + i / 256) % 8) - 4) : 1.f);
    }
    const int64_t nb32 = k / 32, nb256 = k / 256;
    std::vector<int8_t> wq(m * k);
    std::vector<float>  wd(m * nb32);
    for (int64_t r = 0; r < m; ++r) quant_q8_0(&w[r * k], (int) k, &wq[r * k], &wd[r * nb32]);
    // the CPU backend's activations: Q8_0 per 32
    std::vector<int8_t> cq(n * k);
    std::vector<float>  cd(n * nb32);
    for (int64_t t = 0; t < n; ++t) quant_q8_0(&x[t * k], (int) k, &cq[t * k], &cd[t * nb32]);
    // kernel_quant_i8: per (token, 256), qs = bm / 127, q = rte(v * 127 / bm)
    std::vector<int8_t> aq(n * k);
    std::vector<float>  ad(n * nb256);
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t b = 0; b < nb256; ++b) {
            float bm = 0.f;
            for (int j = 0; j < 256; ++j) bm = std::fmax(bm, std::fabs(x[t * k + 256 * b + j]));
            const float iq = bm > 1e-30f ? 127.f / bm : 0.f;
            ad[t * nb256 + b] = bm / 127.f;
            for (int j = 0; j < 256; ++j) aq[t * k + 256 * b + j] = (int8_t) std::nearbyint(x[t * k + 256 * b + j] * iq);
        }
    }
    double e_i8 = 0, e_f16 = 0, e_r1 = 0, e_r2 = 0, e_r3 = 0, s_ref = 0;
    for (int64_t r = 0; r < m; ++r) {
        const int64_t r_first = r - r % 16;   // red 2: the sub-group's first row
        for (int64_t t = 0; t < n; ++t) {
            double ref = 0;
            for (int64_t b = 0; b < nb32; ++b) {
                int32_t s = 0;
                for (int j = 0; j < 32; ++j) s += (int32_t) wq[r * k + 32 * b + j] * cq[t * k + 32 * b + j];
                ref += (double) s * wd[r * nb32 + b] * cd[t * nb32 + b];
            }
            // the int8 arm: a block's integer sum times the block's scale, summed over the
            // super-block in f32, times the activation scale
            float acc = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            const int64_t t_first = t - t % 32;   // red 3: the 32-token tile's first token
            for (int64_t sb = 0; sb < nb256; ++sb) {
                float af = 0.f, af1 = 0.f, af2 = 0.f;
                for (int bl = 0; bl < 8; ++bl) {
                    const int64_t b = sb * 8 + bl;
                    int32_t s = 0;
                    for (int j = 0; j < 32; ++j) s += (int32_t) wq[r * k + 32 * b + j] * aq[t * k + 32 * b + j];
                    af  = std::fma((float) s, wd[r * nb32 + b], af);
                    af1 = std::fma(bl == 7 ? 0.f : (float) s, wd[r * nb32 + b], af1);
                    af2 = std::fma((float) s, wd[r_first * nb32 + b], af2);
                }
                acc  = std::fma(af,  ad[t * nb256 + sb], acc);
                acc1 = std::fma(af1, ad[t * nb256 + sb], acc1);
                acc2 = std::fma(af2, ad[t * nb256 + sb], acc2);
                acc3 = std::fma(af,  ad[t_first * nb256 + sb], acc3);
            }
            // the fp16 arm: half(code x scale) times half(x), f32 sums
            float f16 = 0.f;
            for (int64_t i = 0; i < k; ++i) f16 += hround((float) wq[r * k + i] * wd[r * nb32 + i / 32]) * hround(x[t * k + i]);
            s_ref += ref * ref;
            e_i8  += (acc - ref) * (acc - ref);
            e_f16 += (f16 - ref) * (f16 - ref);
            e_r1  += (acc1 - ref) * (acc1 - ref);
            e_r2  += (acc2 - ref) * (acc2 - ref);
            e_r3  += (acc3 - ref) * (acc3 - ref);
        }
    }
    return { e_i8 / s_ref, e_f16 / s_ref, e_r1 / s_ref, e_r2 / s_ref, e_r3 / s_ref };
}

int main() {
    const std::vector<export_line> ex = parse_export();
    printf("export lines: %zu\n", ex.size());
    CHECK(ex.size() == 42, "%zu export lines, want 42 (14 a ubatch size)", ex.size());

    // ---- 1: red first, then the rule ----
    const struct { const char * what; rule_fn f; } reds[] = {
        { "K % 32 in place of % 256", rule_k32 },
        { "no row cap",              rule_nocap },
        { "accept every Q8_0",       rule_all },
        { "refuse everything",       rule_none },
    };
    for (const auto & r : reds) {
        const int saved = g_failures;
        const int f = check_rule(ex, r.f, true);
        g_failures = saved;   // the red's own failures are the point
        printf("red rule (%s): %d check(s) fail%s\n", r.what, f, f > 0 ? "" : " -- the checks cannot see it");
        if (f == 0) {
            printf("FAIL: the red rule \"%s\" passes every check\n", r.what);
            ++g_failures;
        }
    }
    const int fr = check_rule(ex, rule_real, false);
    printf("gate: %s\n", fr == 0 ? "every export line and refusal as expected" : "FAILED");

    // ---- 2: the shape ----
    {
        const int saved = g_failures;
        const int f = check_shape(shape_fixed, true);
        g_failures = saved;
        printf("red shape (16 sub-groups always): %d check(s) fail\n", f);
        if (f == 0) {
            printf("FAIL: the red shape passes every check\n");
            ++g_failures;
        }
        const int fs = check_shape(ggml_cl_q8_2d_shape, false);
        printf("shape: %s\n", fs == 0 ? "rows covered, no idle work-group, sized to the rows" : "FAILED");
    }

    // ---- 3: the arithmetic ----
    const double bound = 5e-4;   // test_mul_mat::max_nmse_err
    const struct { int64_t m, k; } shapes[] = {
        { 3072, 2304 }, { 2304, 3072 }, { 1024, 2304 }, { 384, 2304 }, { 24, 2304 }, { 32, 2304 }, { 6552, 2304 }, { 1024, 256 },
    };
    for (const auto & s : shapes) {
        // rows capped at 64 and 32 tokens: the error a value carries depends on K and the fill,
        // not on how many rows and tokens there are
        const int64_t mm = s.m < 64 ? s.m : 64;
        const model_out o = model(mm, s.k, 32, false, 20261125u + (uint32_t) s.k);
        printf("model %lld x %lld (%lld rows, 32 tokens), uniform fill: NMSE int8 %.2e, fp16 %.2e, red 1 %.3f, red 2 %.2e, red 3 %.2e\n",
               (long long) s.m, (long long) s.k, (long long) mm, o.nmse_i8, o.nmse_f16, o.nmse_red1, o.nmse_red2, o.nmse_red3);
        CHECK(o.nmse_i8 < bound, "%lld x %lld: the int8 arm's NMSE %.2e is not under %.0e", (long long) s.m, (long long) s.k, o.nmse_i8, bound);
        CHECK(o.nmse_f16 < bound, "%lld x %lld: the fp16 arm's NMSE %.2e is not under %.0e", (long long) s.m, (long long) s.k, o.nmse_f16, bound);
        CHECK(o.nmse_red1 > 100 * bound, "%lld x %lld: red 1's NMSE %.2e is not far above the bound", (long long) s.m, (long long) s.k, o.nmse_red1);
        CHECK(o.nmse_red2 > bound, "%lld x %lld: red 2's NMSE %.2e is under the bound", (long long) s.m, (long long) s.k, o.nmse_red2);
        // red 3 is invisible on the uniform fill: on the record, and a reason for the spread cases
        CHECK(o.nmse_red3 < bound, "%lld x %lld: red 3 is visible on the uniform fill (%.2e): the model changed", (long long) s.m, (long long) s.k, o.nmse_red3);
    }
    // test_mul_mat_spread's shapes, rows capped at 64, 32 tokens
    const struct { int64_t m, k; } spread_shapes[] = { { 3072, 2304 }, { 24, 2304 }, { 257, 2304 }, { 1024, 256 } };
    for (const auto & s : spread_shapes) {
        const int64_t mm = s.m < 64 ? s.m : 64;
        const model_out o = model(mm, s.k, 32, true, 7u + (uint32_t) s.m);
        printf("model %lld x %lld (%lld rows, 32 tokens), spread fill: NMSE int8 %.2e, fp16 %.2e, red 1 %.3f, red 2 %.3f, red 3 %.3f\n",
               (long long) s.m, (long long) s.k, (long long) mm, o.nmse_i8, o.nmse_f16, o.nmse_red1, o.nmse_red2, o.nmse_red3);
        CHECK(o.nmse_i8 < bound && o.nmse_f16 < bound, "%lld x %lld: the spread fill's green arms are not under the bound", (long long) s.m, (long long) s.k);
        CHECK(o.nmse_red1 > 100 * bound, "%lld x %lld spread: red 1's NMSE %.2e is not far above the bound", (long long) s.m, (long long) s.k, o.nmse_red1);
        CHECK(o.nmse_red2 > 100 * bound, "%lld x %lld spread: red 2's NMSE %.2e is not far above the bound", (long long) s.m, (long long) s.k, o.nmse_red2);
        CHECK(o.nmse_red3 > 100 * bound, "%lld x %lld spread: red 3's NMSE %.2e is not far above the bound", (long long) s.m, (long long) s.k, o.nmse_red3);
    }

    if (g_failures) {
        printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    printf("all checks pass\n");
    return 0;
}
