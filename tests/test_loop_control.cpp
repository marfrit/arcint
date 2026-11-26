// Patch 0221's device-free cell: Inkling 2 Mini's loop controller as one op (GGML_OP_LOOP_CONTROL).
//
// Every shape is read off the model's exported graphs (tests/loop_control_export_lines.inc:
// test-export-graph-ops lines, verbatim, at ub 512, 1024 and 2048, with and without the depth
// mix, and since 0226 at 1, 2, 3, 4, 5, 6 and 8 rows). The parts:
//
//  1. the rule (ggml_loop_control_refuse) on the exported calls: the tensors are built with the
//     graph's constructors and compared with the lines' ne and nb, and every exported call must be
//     accepted; out-of-scope variants must be refused, each by its reason. Red first, on every
//     run: a rule that accepts everything must fail the refusal checks.
//
//  2. the launch plan (ggml_loop_control_plan_make, which the OpenCL dispatch binds its arguments
//     from): each argument's reach equals its tensor's extent, the geometry for the B60 (20 Xe
//     cores) and the B70 (32), the local memory under the device's 64 KB.
//
//  3. the CPU op against the CHAIN the model emits without the knob (rms_norm x2, concat, the three
//     products, tanh x2, mul, sub, mul, add), built node for node as src/models/inkling2.cpp's
//     control() builds it, on the CPU backend: nmse <= 1e-7 (the design's bound for the CPU op).
//
//  4. a host replay of kernels/loop_control.cl ITSELF (the .cl text compiled as C++: 512 lanes as
//     threads, barrier() a real barrier, a sub-group's collectives - sub_group_reduce_add and the
//     bf16 DPAS - exchanged between its 16 lane threads, the local memory shared), every tensor at
//     its own non-zero offset of one buffer as on the backend, at the plan's arguments, against the
//     CPU op and against the chain. The f32 build must agree with both at nmse <= 1e-6 (the card's
//     op-case bound); the bf16 build at <= 1e-4 (its own bound). The reds of the same text
//     (LOOP_CONTROL_RED 1..4 in the f32 build, 5 in the bf16 build) must each miss the op on every
//     case by more than the bound.
//
//  5. (0226) the small pair, kernels/loop_control_small.cl, for 1 .. 8 rows - the reference's own
//     fused range, every decode step: the plan's small launch, the switch's row ranges
//     (ggml_loop_control_min_rows / _small_on), and a host replay of both kernels (the gate's
//     work-groups of 64 lane threads with their sub-group reduce and barrier, the out a lane a
//     column) through a scratch with guard floats, cur and msg at different padded row steps,
//     against the CPU op and the chain at every row count 1 .. 8: nmse <= 1e-6. Reds of the
//     same text: 1 - 4 must miss every case; 6 (the scratch's row step) and 7 (the tail row not
//     stored) are defects of rows past the first, so they must miss every case of 2 .. 8 rows
//     and must PASS the one-row case - which is why the one-row case alone proves nothing
//     about a row step.
//
// What this cannot see: the device's DPAS operand layout (the emulation below encodes the
// documented one, mul_mm_kq_f16.cl's card-proven SG16 layout), the device's tanh / sqrt, and
// anything about speed. Those are the card rows (contrib/llama.cpp/README.md, 0221).
//
// No GPU, no OpenCL runtime.

#define GGML_USE_CPU 1

#include "ggml.h"
#include "ggml-cpu.h"

#include <pthread.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------------------
// loop_control.cl on the host
// ---------------------------------------------------------------------------------------
namespace host_cl {
typedef uint32_t uint;
typedef uint16_t ushort;
typedef uint64_t ulong;

template <typename T, int N> struct vec {
    T s[N];
    T &       operator[](int i)       { return s[i]; }
    const T & operator[](int i) const { return s[i]; }
};
typedef vec<float, 4>  float4;
typedef vec<float, 8>  float8;
typedef vec<float, 16> float16;
typedef vec<short, 8>  short8;
typedef vec<int, 8>    int8;

struct wg_state {
    pthread_barrier_t wg;
    pthread_barrier_t sg[32];
    float  red[32][16];
    short8 a[32][16];
    int8   b[32][16];
};
thread_local int        g_group = 0;
thread_local int        g_gid[3] = {0, 0, 0};   // 0226: the small pair's kernels read global ids
thread_local int        g_lid   = 0;
thread_local wg_state * g_wg    = nullptr;

inline size_t get_group_id(int)    { return (size_t) g_group; }
inline size_t get_local_id(int)    { return (size_t) g_lid; }
inline size_t get_global_id(int d) { return (size_t) g_gid[d]; }
inline int    lc_host_sg()         { return g_lid / 16; }
inline int    lc_host_lane()       { return g_lid % 16; }
inline void   barrier(int)         { pthread_barrier_wait(&g_wg->wg); }
inline int    min(int a, int b)    { return a < b ? a : b; }
inline float  sqrt(float x)        { return std::sqrt(x); }
inline float  tanh(float x)        { return std::tanh(x); }
inline float  fma(float a, float b, float c) { return std::fma(a, b, c); }
inline uint   as_uint(float f)     { uint u; memcpy(&u, &f, 4); return u; }
inline void   prefetch(const float *, int) {}
inline float4  vload4 (size_t o, const float * p) { float4  v; for (int i = 0; i < 4;  ++i) v[i] = p[o*4  + i]; return v; }
inline float8  vload8 (size_t o, const float * p) { float8  v; for (int i = 0; i < 8;  ++i) v[i] = p[o*8  + i]; return v; }
inline float16 vload16(size_t o, const float * p) { float16 v; for (int i = 0; i < 16; ++i) v[i] = p[o*16 + i]; return v; }

// the sub-group's collectives: each lane posts its value, the 16 lanes meet, each reads
inline float sub_group_reduce_add(float x) {
    const int sg = lc_host_sg(), l = lc_host_lane();
    g_wg->red[sg][l] = x;
    pthread_barrier_wait(&g_wg->sg[sg]);
    float s = 0.0f;
    for (int i = 0; i < 16; ++i) s += g_wg->red[sg][i];
    pthread_barrier_wait(&g_wg->sg[sg]);
    return s;
}
inline float bf16_to_f(uint16_t h) { uint u = (uint) h << 16; float f; memcpy(&f, &u, 4); return f; }
// SIMD16 bf16 DPAS, k 16: a = short8, lane i holds k = i, component m is row m; b = int8, lane n
// is column n, its 16 k as 8 pairs, the low half the even k; result lane n column n, component m
// row m. The products are exact in f32; the 16-term sum runs k = 0 .. 15 here (the device's
// order is unspecified)
inline float8 intel_sub_group_bf16_bf16_matrix_mad_k16(short8 a, int8 b, float8 acc) {
    const int sg = lc_host_sg(), n = lc_host_lane();
    g_wg->a[sg][n] = a;
    g_wg->b[sg][n] = b;
    pthread_barrier_wait(&g_wg->sg[sg]);
    float8 r = acc;
    for (int m = 0; m < 8; ++m) {
        float s = acc[m];
        for (int k = 0; k < 16; ++k) {
            const float av = bf16_to_f((uint16_t) g_wg->a[sg][k][m]);
            const uint  bw = (uint) g_wg->b[sg][n][k/2];
            const float bv = bf16_to_f((uint16_t) (k % 2 == 0 ? (bw & 0xFFFFu) : (bw >> 16)));
            s += av*bv;
        }
        r[m] = s;
    }
    pthread_barrier_wait(&g_wg->sg[sg]);
    return r;
}
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wpedantic"
#define LOOP_CONTROL_HOST_REPLAY 1
#define CLK_LOCAL_MEM_FENCE 0
#define kernel
#define global
#define local
#define LC_REQD_SG
#define LC_SGID()   lc_host_sg()
#define LC_LANE()   lc_host_lane()
#define LC_ZERO8(v) (v) = float8{}
#define LC_KERNEL_NS(ns) namespace ns { using namespace host_cl;
#define LC_KERNEL_END }

LC_KERNEL_NS(k_f32)
#include "loop_control.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 1
LC_KERNEL_NS(k_red1)
#include "loop_control.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 2
LC_KERNEL_NS(k_red2)
#include "loop_control.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 3
LC_KERNEL_NS(k_red3)
#include "loop_control.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 4
LC_KERNEL_NS(k_red4)
#include "loop_control.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LC_BF16 1
LC_KERNEL_NS(k_bf16)
#include "loop_control.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 5
LC_KERNEL_NS(k_red5)
#include "loop_control.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#undef LC_BF16
// 0226: the small pair's text, green and its reds
LC_KERNEL_NS(ks_f32)
#include "loop_control_small.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 1
LC_KERNEL_NS(ks_red1)
#include "loop_control_small.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 2
LC_KERNEL_NS(ks_red2)
#include "loop_control_small.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 3
LC_KERNEL_NS(ks_red3)
#include "loop_control_small.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 4
LC_KERNEL_NS(ks_red4)
#include "loop_control_small.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 6
LC_KERNEL_NS(ks_red6)
#include "loop_control_small.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#define LOOP_CONTROL_RED 7
LC_KERNEL_NS(ks_red7)
#include "loop_control_small.cl"
LC_KERNEL_END
#undef LOOP_CONTROL_RED
#undef kernel
#undef global
#undef local
#pragma GCC diagnostic pop

namespace {

int  g_failures = 0;
bool g_quiet    = false;

#define CHECK(cond, ...) do { \
        if (!(cond)) { if (!g_quiet) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } ++g_failures; } \
    } while (0)

// ---------------------------------------------------------------------------------------
// 1. the rule on the export lines
// ---------------------------------------------------------------------------------------
struct export_src {
    int      type;
    int64_t  ne[4];
    uint64_t nb[4];
};
struct export_line {
    std::string file;
    int         line;
    int         op;
    int64_t     ne[4];
    std::vector<export_src> src;
};
struct export_raw {
    const char * file;
    int          line;
    const char * text;
};
const export_raw k_export_raw[] = {
#include "loop_control_export_lines.inc"
};

std::vector<export_line> parse_export() {
    std::vector<export_line> out;
    for (const export_raw & r : k_export_raw) {
        std::istringstream in(r.text);
        export_line l;
        l.file = r.file;
        l.line = r.line;
        int type = 0, n_params = 0, n_src = 0;
        in >> l.op >> type >> l.ne[0] >> l.ne[1] >> l.ne[2] >> l.ne[3] >> n_params;
        for (int i = 0; i < n_params; ++i) {
            int64_t p;
            in >> p;
        }
        in >> n_src;
        for (int s = 0; s < n_src; ++s) {
            export_src e;
            in >> e.type >> e.ne[0] >> e.ne[1] >> e.ne[2] >> e.ne[3] >> e.nb[0] >> e.nb[1] >> e.nb[2] >> e.nb[3];
            l.src.push_back(e);
        }
        CHECK(in && n_src == 5, "unparsed export line %s:%d", r.file, r.line);
        out.push_back(l);
    }
    return out;
}

typedef const char * (*rule_fn)(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
                                 const ggml_tensor *, const ggml_tensor *);
const char * rule_accept_all(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
                             const ggml_tensor *, const ggml_tensor *) {
    return nullptr;
}

bool same_layout(const ggml_tensor * t, const export_src & e) {
    for (int i = 0; i < 4; ++i) {
        if (t->ne[i] != e.ne[i] || t->nb[i] != e.nb[i]) return false;
    }
    return (int) t->type == e.type;
}

// the model's tensors for an exported call: cur and msg as the graph has them (the line's nb
// says whether a contiguous node or a mix column's view), the weights at their GGUF shapes
struct call_tensors {
    ggml_tensor * cur, * msg, * down, * gain, * forget;
};
call_tensors build_call(ggml_context * ctx, int64_t d, int64_t nt) {
    call_tensors c;
    c.cur    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, nt);
    // a mix column: ggml_depth_mix_column's view of a [d, nt, 4] result
    ggml_tensor * mixed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, nt, 4);
    c.msg    = ggml_view_2d(ctx, mixed, d, nt, mixed->nb[1], 2*mixed->nb[2]);
    c.down   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*d, 64);
    c.gain   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, d);
    c.forget = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, d);
    return c;
}

int check_rule(rule_fn rule, const std::vector<export_line> & ex) {
    const int before = g_failures;
    ggml_init_params p = { 512*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    for (const export_line & l : ex) {
        CHECK(l.op == GGML_OP_LOOP_CONTROL, "%s:%d is not op %d", l.file.c_str(), l.line, (int) GGML_OP_LOOP_CONTROL);
        call_tensors c = build_call(ctx, l.ne[0], l.ne[1]);
        CHECK(same_layout(c.cur, l.src[0]) && same_layout(c.msg, l.src[1]) && same_layout(c.down, l.src[2]) &&
              same_layout(c.gain, l.src[3]) && same_layout(c.forget, l.src[4]),
              "%s:%d: the constructors do not give the line's ne/nb", l.file.c_str(), l.line);
        // build them at the line's own strides as well (the export is what the graph had)
        const char * why = rule(c.cur, c.msg, c.down, c.gain, c.forget);
        CHECK(why == nullptr, "%s:%d refused (%s)", l.file.c_str(), l.line, why ? why : "");
        why = rule(c.msg, c.cur, c.down, c.gain, c.forget);
        CHECK(why == nullptr, "%s:%d with cur a view refused (%s)", l.file.c_str(), l.line, why ? why : "");
    }
    // out of scope, each refused by its reason
    struct variant { const char * what; const char * reason; };
    const int64_t d = 2304, nt = 1024;
    {
        call_tensors c = build_call(ctx, d, nt);
        ggml_tensor * f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, nt);
        const char * w = rule(f16, c.msg, c.down, c.gain, c.forget);
        CHECK(w != nullptr && strcmp(w, "not-f32") == 0, "f16 cur: %s", w ? w : "accepted");
    }
    {
        ggml_tensor * cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2300, nt);
        ggml_tensor * msg  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2300, nt);
        ggml_tensor * down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4600, 64);
        ggml_tensor * g    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 2300);
        const char * w = rule(cur, msg, down, g, g);
        CHECK(w != nullptr && strcmp(w, "d-lanes") == 0, "n_embd 2300: %s", w ? w : "accepted");
    }
    {
        call_tensors c = build_call(ctx, d, nt);
        ggml_tensor * down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*d, 32);
        ggml_tensor * g    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, d);
        const char * w = rule(c.cur, c.msg, down, g, g);
        CHECK(w != nullptr && strcmp(w, "rank") == 0, "rank 32: %s", w ? w : "accepted");
    }
    {
        call_tensors c = build_call(ctx, d, nt);
        ggml_tensor * big = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*d, nt);
        ggml_tensor * strided = ggml_view_2d(ctx, big, d, nt/2, 2*big->nb[1], 0);   // T mismatched and strided
        const char * w = rule(c.cur, strided, c.down, c.gain, c.forget);
        CHECK(w != nullptr && strcmp(w, "msg-shape") == 0, "msg of other rows: %s", w ? w : "accepted");
        ggml_tensor * tr = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nt, d));
        w = rule(tr, c.msg, c.down, c.gain, c.forget);
        CHECK(w != nullptr && strcmp(w, "cur-stride") == 0, "transposed cur: %s", w ? w : "accepted");
    }
    {
        call_tensors c = build_call(ctx, d, nt);
        ggml_tensor * dt = ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 2*d));
        const char * w = rule(c.cur, c.msg, dt, c.gain, c.forget);
        CHECK(w != nullptr && strcmp(w, "down-stride") == 0, "transposed down: %s", w ? w : "accepted");
        ggml_tensor * gt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, d/2);
        w = rule(c.cur, c.msg, c.down, gt, c.forget);
        CHECK(w != nullptr && strcmp(w, "gain-shape") == 0, "gain of half the rows: %s", w ? w : "accepted");
        w = rule(c.cur, c.msg, c.down, c.gain, gt);
        CHECK(w != nullptr && strcmp(w, "forget-shape") == 0, "forget of half the rows: %s", w ? w : "accepted");
        ggml_tensor * c3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, nt, 2);
        w = rule(c3, c.msg, c.down, c.gain, c.forget);
        CHECK(w != nullptr && strcmp(w, "cur-shape") == 0, "3-d cur: %s", w ? w : "accepted");
        w = rule(c.cur, c.msg, c.down, c.gain, nullptr);
        CHECK(w != nullptr && strcmp(w, "missing") == 0, "no forget: %s", w ? w : "accepted");
    }
    ggml_free(ctx);
    return g_failures - before;
}

// ---------------------------------------------------------------------------------------
// 2. the plan
// ---------------------------------------------------------------------------------------
void check_plan(const std::vector<export_line> & ex) {
    ggml_init_params p = { 512*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    for (const export_line & l : ex) {
        call_tensors c = build_call(ctx, l.ne[0], l.ne[1]);
        ggml_tensor * dst = ggml_loop_control(ctx, c.cur, c.msg, c.down, c.gain, c.forget, 1e-6f);
        for (int cores : {20, 32}) {
            ggml_loop_control_plan pl;
            ggml_loop_control_plan_make(dst, cores, &pl);
            CHECK(pl.reach_cur == ggml_nbytes(c.cur) && pl.reach_msg == ggml_nbytes(c.msg) &&
                  pl.reach_down == ggml_nbytes(c.down) && pl.reach_gain == ggml_nbytes(c.gain) &&
                  pl.reach_forget == ggml_nbytes(c.forget) && pl.reach_dst == ggml_nbytes(dst),
                  "%s:%d cores %d: a reach differs from its tensor", l.file.c_str(), l.line, cores);
            CHECK(pl.rows_wg % 8 == 0 && pl.rows_wg <= GGML_LOOP_CONTROL_ROWS_MAX && (int64_t) pl.n_wg*pl.rows_wg >= l.ne[1] &&
                  (int64_t) (pl.n_wg - 1)*pl.rows_wg < l.ne[1], "%s:%d cores %d: rows_wg %d n_wg %d", l.file.c_str(), l.line,
                  cores, pl.rows_wg, pl.n_wg);
            CHECK(pl.local_floats*4 <= 65536, "local memory %llu bytes", (unsigned long long) pl.local_floats*4);
            CHECK(pl.rsplit == 2 && (l.ne[0]/16*pl.rsplit) % GGML_LOOP_CONTROL_NSG == 0, "rsplit %d", pl.rsplit);
        }
    }
    // the geometry the README prices: 1,024 rows, 56 rows a group on the B60 (19 groups), 32 on
    // the B70 (32 groups); 2,048 rows cap at 64 a group
    call_tensors c = build_call(ctx, 2304, 1024);
    ggml_tensor * dst = ggml_loop_control(ctx, c.cur, c.msg, c.down, c.gain, c.forget, 1e-6f);
    ggml_loop_control_plan pl;
    ggml_loop_control_plan_make(dst, 20, &pl);
    CHECK(pl.rows_wg == 56 && pl.n_wg == 19, "B60 1024 rows: %d x %d", pl.rows_wg, pl.n_wg);
    ggml_loop_control_plan_make(dst, 32, &pl);
    CHECK(pl.rows_wg == 32 && pl.n_wg == 32, "B70 1024 rows: %d x %d", pl.rows_wg, pl.n_wg);
    printf("plan: 1024 rows B60 56 x 19, B70 32 x 32; local %llu bytes at 64 rows\n",
           (unsigned long long) ((64*64 + 32*16*16 + 32*16)*4));
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------------------
// 3 / 4. data, the CPU op, the chain, the replay
// ---------------------------------------------------------------------------------------
double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        num += d*d;
        den += (double) b[i]*b[i];
    }
    return den > 0 ? num/den : num;
}

float to_bf16_value(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u &= 0xFFFF0000u;
    memcpy(&f, &u, 4);
    return f;
}

// one case's inputs: cur / msg rows at amplitudes 0.25 .. 4, msg at 10x; weights bf16 values at
// scales that keep tanh's arguments near 1 (the op cases' initialization, test-backend-ops 0221)
struct case_data {
    int64_t d, nt;
    std::vector<float> cur, msg, down, gain, forget;
};
case_data make_case(int64_t nt, uint32_t seed) {
    case_data c;
    c.d = 2304;
    c.nt = nt;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    c.cur.resize((size_t) (c.d*nt));
    c.msg.resize((size_t) (c.d*nt));
    for (int64_t r = 0; r < nt; ++r) {
        const float ac = 0.25f*(float) (1 << ((r*7) % 5));
        const float am = 10.0f*0.25f*(float) (1 << ((r*7 + 2) % 5));
        for (int64_t e = 0; e < c.d; ++e) {
            c.cur[(size_t) (r*c.d + e)] = ac*u(gen);
            c.msg[(size_t) (r*c.d + e)] = am*u(gen);
        }
    }
    c.down.resize((size_t) (2*c.d*64));
    c.gain.resize((size_t) (64*c.d));
    c.forget.resize((size_t) (64*c.d));
    for (float & v : c.down)   v = to_bf16_value(0.03f*u(gen));
    for (float & v : c.gain)   v = to_bf16_value(0.2f*u(gen));
    for (float & v : c.forget) v = to_bf16_value(0.2f*u(gen));
    return c;
}

// the CPU op (chain = false) or the model's chain node for node (chain = true), on the CPU backend
std::vector<float> cpu_eval(const case_data & c, bool chain) {
    const size_t mem = (size_t) (c.d*c.nt*4)*16 + (size_t) (4*c.d*64*4) + 64*ggml_tensor_overhead() + ggml_graph_overhead() + (64 << 20);
    ggml_init_params p = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * cur    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.d, c.nt);
    ggml_tensor * msg    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.d, c.nt);
    ggml_tensor * down   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*c.d, 64);
    ggml_tensor * gain   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, c.d);
    ggml_tensor * forget = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, c.d);
    memcpy(cur->data, c.cur.data(), ggml_nbytes(cur));
    memcpy(msg->data, c.msg.data(), ggml_nbytes(msg));
    memcpy(down->data, c.down.data(), ggml_nbytes(down));
    memcpy(gain->data, c.gain.data(), ggml_nbytes(gain));
    memcpy(forget->data, c.forget.data(), ggml_nbytes(forget));
    ggml_tensor * out;
    if (!chain) {
        out = ggml_loop_control(ctx, cur, msg, down, gain, forget, 1e-6f);
    } else {
        ggml_tensor * gate = ggml_concat(ctx, ggml_rms_norm(ctx, cur, 1e-6f), ggml_rms_norm(ctx, msg, 1e-6f), 0);
        gate = ggml_mul_mat(ctx, down, gate);
        ggml_tensor * tg = ggml_tanh(ctx, ggml_mul_mat(ctx, gain,   gate));
        ggml_tensor * tf = ggml_tanh(ctx, ggml_mul_mat(ctx, forget, gate));
        out = ggml_add(ctx, ggml_sub(ctx, cur, ggml_mul(ctx, tf, cur)), ggml_mul(ctx, tg, msg));
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    std::vector<float> r((size_t) (c.d*c.nt));
    memcpy(r.data(), out->data, r.size()*4);
    ggml_free(ctx);
    return r;
}

typedef uint64_t u64;
typedef void (*kernel_fn)(const char *, u64, u64, const char *, u64, u64, const char *, u64, u64,
                          const char *, u64, u64, const char *, u64, u64, char *, u64, u64,
                          int, int, int, int, float, float *);

// the replay: one byte buffer, every tensor at its own non-zero offset (as the backend's buffers
// hold them), cur and msg at a padded row step (a view's), the plan's arguments, the work-groups
// one after the other with 512 lane threads each
std::vector<float> replay(kernel_fn kfn, const case_data & c, int cores) {
    const uint64_t pad  = 64;                                   // cur / msg row step past a row
    const uint64_t nbc1 = (uint64_t) c.d*4 + pad, nbm1 = (uint64_t) c.d*4 + 2*pad, nbd1 = (uint64_t) c.d*4;
    uint64_t off = 4096;
    auto place = [&](uint64_t bytes) { const uint64_t o = off; off += (bytes + 4095)/4096*4096 + 4096; return o; };
    const uint64_t oc = place(nbc1*c.nt), om = place(nbm1*c.nt);
    const uint64_t ow = place(c.down.size()*4), og = place(c.gain.size()*4), of = place(c.forget.size()*4);
    const uint64_t od = place(nbd1*c.nt) + 128;
    std::vector<char> buf(off + 4096 + 128, 0);
    for (int64_t r = 0; r < c.nt; ++r) {
        memcpy(&buf[oc + r*nbc1], &c.cur[(size_t) (r*c.d)], (size_t) c.d*4);
        memcpy(&buf[om + r*nbm1], &c.msg[(size_t) (r*c.d)], (size_t) c.d*4);
    }
    memcpy(&buf[ow], c.down.data(), c.down.size()*4);
    memcpy(&buf[og], c.gain.data(), c.gain.size()*4);
    memcpy(&buf[of], c.forget.data(), c.forget.size()*4);

    // the plan from a node of the same shapes
    ggml_init_params p = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * big_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (nbc1*c.nt/4));
    ggml_tensor * big_m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (nbm1*c.nt/4));
    ggml_tensor * cur    = ggml_view_2d(ctx, big_c, c.d, c.nt, nbc1, 0);
    ggml_tensor * msg    = ggml_view_2d(ctx, big_m, c.d, c.nt, nbm1, 0);
    ggml_tensor * down   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*c.d, 64);
    ggml_tensor * gain   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, c.d);
    ggml_tensor * forget = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, c.d);
    ggml_tensor * dst = ggml_loop_control(ctx, cur, msg, down, gain, forget, 1e-6f);
    ggml_loop_control_plan pl;
    ggml_loop_control_plan_make(dst, cores, &pl);
    ggml_free(ctx);
    CHECK(oc + pl.reach_cur <= buf.size() && od + pl.reach_dst <= buf.size(), "replay buffer too small");

    std::vector<float> slm((size_t) pl.local_floats);
    for (int g = 0; g < pl.n_wg; ++g) {
        host_cl::wg_state st;
        pthread_barrier_init(&st.wg, nullptr, (unsigned) pl.lsz);
        for (int s = 0; s < 32; ++s) pthread_barrier_init(&st.sg[s], nullptr, 16);
        std::fill(slm.begin(), slm.end(), std::nanf(""));   // a read before a write shows
        std::vector<std::thread> th;
        th.reserve((size_t) pl.lsz);
        for (int lid = 0; lid < pl.lsz; ++lid) {
            th.emplace_back([&, lid, g]() {
                host_cl::g_group = g;
                host_cl::g_lid   = lid;
                host_cl::g_wg    = &st;
                kfn(buf.data(), oc, pl.nbc1, buf.data(), om, pl.nbm1, buf.data(), ow, pl.nbw1,
                    buf.data(), og, pl.nbg1, buf.data(), of, pl.nbf1, buf.data(), od, pl.nbd1,
                    pl.ne0, pl.nt, pl.rows_wg, pl.rsplit, 1e-6f, slm.data());
            });
        }
        for (std::thread & t : th) t.join();
        pthread_barrier_destroy(&st.wg);
        for (int s = 0; s < 32; ++s) pthread_barrier_destroy(&st.sg[s]);
    }
    std::vector<float> r((size_t) (c.d*c.nt));
    for (int64_t t = 0; t < c.nt; ++t) {
        memcpy(&r[(size_t) (t*c.d)], &buf[od + t*nbd1], (size_t) c.d*4);
    }
    return r;
}

// ---------------------------------------------------------------------------------------
// 5. the small pair (0226): loop_control_small.cl's two kernels on the host
// ---------------------------------------------------------------------------------------
typedef void (*gate_fn)(const char *, u64, u64, const char *, u64, u64, const char *, u64, u64,
                        char *, u64, int, int, float, float *);
typedef void (*out_fn)(const char *, u64, u64, const char *, u64, u64, const char *, u64, u64,
                       const char *, u64, u64, const char *, u64, char *, u64, u64, int, int, int);

// The launch the dispatch makes, from the same plan: the gate's work-groups one after the other,
// GATE_W lane threads that walk the groups in step (a barrier ends each group, so no lane enters
// the next group's local memory early); then the out, a work-item a column (it has no barrier
// and no collective, so the items run one after the other). The scratch sits at a non-zero
// offset between NaN guards: a read before a write, or a write past nt*2*64 floats, shows.
std::vector<float> replay_small(gate_fn kg, out_fn ko, const case_data & c, bool * scratch_ok) {
    const uint64_t pad  = 64;
    const uint64_t nbc1 = (uint64_t) c.d*4 + pad, nbm1 = (uint64_t) c.d*4 + 2*pad, nbd1 = (uint64_t) c.d*4;
    uint64_t off = 4096;
    auto place = [&](uint64_t bytes) { const uint64_t o = off; off += (bytes + 4095)/4096*4096 + 4096; return o; };
    const uint64_t oc = place(nbc1*c.nt), om = place(nbm1*c.nt);
    const uint64_t ow = place(c.down.size()*4), og = place(c.gain.size()*4), of = place(c.forget.size()*4);
    const uint64_t od = place(nbd1*c.nt) + 128;
    std::vector<char> buf(off + 4096 + 128, 0);
    for (int64_t r = 0; r < c.nt; ++r) {
        memcpy(&buf[oc + r*nbc1], &c.cur[(size_t) (r*c.d)], (size_t) c.d*4);
        memcpy(&buf[om + r*nbm1], &c.msg[(size_t) (r*c.d)], (size_t) c.d*4);
    }
    memcpy(&buf[ow], c.down.data(), c.down.size()*4);
    memcpy(&buf[og], c.gain.data(), c.gain.size()*4);
    memcpy(&buf[of], c.forget.data(), c.forget.size()*4);

    ggml_init_params p = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * big_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (nbc1*c.nt/4));
    ggml_tensor * big_m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (nbm1*c.nt/4));
    ggml_tensor * cur    = ggml_view_2d(ctx, big_c, c.d, c.nt, nbc1, 0);
    ggml_tensor * msg    = ggml_view_2d(ctx, big_m, c.d, c.nt, nbm1, 0);
    ggml_tensor * down   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*c.d, 64);
    ggml_tensor * gain   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, c.d);
    ggml_tensor * forget = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, c.d);
    ggml_tensor * dst = ggml_loop_control(ctx, cur, msg, down, gain, forget, 1e-6f);
    ggml_loop_control_plan pl;
    ggml_loop_control_plan_make(dst, 20, &pl);
    ggml_free(ctx);
    CHECK(pl.small == 1, "replay_small at %lld rows: the plan is not small", (long long) c.nt);
    CHECK(oc + pl.reach_cur <= buf.size() && od + pl.reach_dst <= buf.size(), "replay buffer too small");

    // the scratch: 16 guard floats, the plan's floats, 64 guard floats
    const uint64_t offp = 16*4;
    std::vector<float> gp((size_t) (16 + pl.gp_floats + 64), std::nanf(""));

    const int W = (int) pl.gate_gws[0];
    std::vector<float> slm((size_t) pl.gate_local_floats);
    host_cl::wg_state st;
    pthread_barrier_init(&st.wg, nullptr, (unsigned) W);
    for (int s = 0; s < 32; ++s) pthread_barrier_init(&st.sg[s], nullptr, 16);
    const int n1 = (int) pl.gate_gws[1], n2 = (int) pl.gate_gws[2];
    std::vector<std::thread> th;
    th.reserve((size_t) W);
    for (int lid = 0; lid < W; ++lid) {
        th.emplace_back([&, lid]() {
            host_cl::g_lid = lid;
            host_cl::g_wg  = &st;
            for (int g2 = 0; g2 < n2; ++g2) {
                for (int g1 = 0; g1 < n1; ++g1) {
                    host_cl::g_gid[0] = lid;
                    host_cl::g_gid[1] = g1;
                    host_cl::g_gid[2] = g2;
                    kg(buf.data(), oc, pl.nbc1, buf.data(), om, pl.nbm1, buf.data(), ow, pl.nbw1,
                       (char *) gp.data(), offp, pl.ne0, pl.nt, 1e-6f, slm.data());
                    pthread_barrier_wait(&st.wg);   // the group ends for every lane before the next begins
                }
            }
        });
    }
    for (std::thread & t : th) t.join();
    pthread_barrier_destroy(&st.wg);
    for (int s = 0; s < 32; ++s) pthread_barrier_destroy(&st.sg[s]);

    bool ok = true;
    for (size_t i = 0; i < 16; ++i) ok = ok && std::isnan(gp[i]);
    for (size_t i = 0; i < (size_t) pl.gp_floats; ++i) ok = ok && !std::isnan(gp[16 + i]);
    for (size_t i = 0; i < 64; ++i) ok = ok && std::isnan(gp[(size_t) (16 + pl.gp_floats) + i]);
    if (scratch_ok != nullptr) *scratch_ok = ok;

    host_cl::g_lid = 0;
    for (int d = 0; d < (int) pl.out_gws; ++d) {
        host_cl::g_gid[0] = d;
        host_cl::g_gid[1] = 0;
        host_cl::g_gid[2] = 0;
        ko(buf.data(), oc, pl.nbc1, buf.data(), om, pl.nbm1, buf.data(), og, pl.nbg1, buf.data(), of, pl.nbf1,
           (const char *) gp.data(), offp, buf.data(), od, pl.nbd1, pl.ne0, pl.nt, pl.out_tr);
    }
    std::vector<float> r((size_t) (c.d*c.nt));
    for (int64_t t = 0; t < c.nt; ++t) {
        memcpy(&r[(size_t) (t*c.d)], &buf[od + t*nbd1], (size_t) c.d*4);
    }
    return r;
}

// the small launch's plan and the switch's ranges
void check_small_plan_and_switch() {
    ggml_init_params p = { 512*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(p);
    for (int64_t nt = 1; nt <= 16; ++nt) {
        call_tensors c = build_call(ctx, 2304, nt);
        ggml_tensor * dst = ggml_loop_control(ctx, c.cur, c.msg, c.down, c.gain, c.forget, 1e-6f);
        ggml_loop_control_plan pl;
        ggml_loop_control_plan_make(dst, 20, &pl);
        const int tr = nt == 1 ? 1 : nt == 2 ? 2 : nt <= 4 ? 4 : 8;
        CHECK(pl.small == (nt <= 8 ? 1 : 0), "nt %lld: small %d", (long long) nt, pl.small);
        if (nt <= 8) {
            CHECK(pl.out_tr == tr && pl.out_tr >= nt, "nt %lld: out_tr %d", (long long) nt, pl.out_tr);
            CHECK(pl.gate_gws[0] == GGML_LOOP_CONTROL_GATE_W && pl.gate_gws[1] == 128 && pl.gate_gws[2] == (uint64_t) nt,
                  "nt %lld: the gate's global size", (long long) nt);
            CHECK(pl.out_gws == 2304 && pl.out_gws % GGML_LOOP_CONTROL_OUT_W == 0, "nt %lld: the out's global size", (long long) nt);
            CHECK(pl.gp_floats == (uint64_t) nt*128 && pl.gp_floats <= (uint64_t) GGML_LOOP_CONTROL_SMALL_ROWS*128,
                  "nt %lld: the scratch %llu floats", (long long) nt, (unsigned long long) pl.gp_floats);
            CHECK(pl.gate_local_floats == 8, "the gate's local floats %llu", (unsigned long long) pl.gate_local_floats);
            // a lane's chunks: n_embd/4 float4 over GATE_W lanes, whole
            CHECK((2304/4) % GGML_LOOP_CONTROL_GATE_W == 0, "the gate's lanes do not divide the row");
            CHECK(pl.reach_cur == ggml_nbytes(c.cur) && pl.reach_msg == ggml_nbytes(c.msg) && pl.reach_dst == ggml_nbytes(dst),
                  "nt %lld: a reach differs from its tensor", (long long) nt);
        }
    }
    ggml_free(ctx);

    // the switch (ggml.h, 0226): mode, the model's least row count, whether <= 8 rows take the pair
    struct kv { const char * v; int mode; int min_rows; bool small; };
    const kv t[] = {
        { nullptr,      0, 0, false }, { "0",          0, 0, false }, { "off",       0, 0, false },
        { "1",          1, 1, true  }, { "f32",        1, 1, true  }, { "all-layers", 1, 1, true },
        { "f32-t2",     1, 2, true  }, { "f32-prompt", 1, 9, true  }, { "f32-onewg",  1, 1, false },
        { "bf16",       2, 9, false }, { "BF16",      -1, 0, false }, { "f32-t3",    -1, 0, false },
        { "F32",       -1, 0, false },
    };
    for (const kv & e : t) {
        CHECK(ggml_loop_control_mode(e.v) == e.mode && ggml_loop_control_min_rows(e.v) == e.min_rows &&
              ggml_loop_control_small_on(e.v) == e.small,
              "switch value %s: mode %d min_rows %d small %d", e.v ? e.v : "(unset)", ggml_loop_control_mode(e.v),
              ggml_loop_control_min_rows(e.v), (int) ggml_loop_control_small_on(e.v));
    }
    printf("small plan: 1 .. 8 rows small, out_tr 1 / 2 / 4 / 8, gate 64 x 128 x T; switch ranges hold\n");
}

// the pair against the CPU op and the chain, and its reds
void check_small_replay() {
    struct sarm { const char * name; gate_fn g; out_fn o; int kind; };   // kind 0 green, 1 red at every T, 2 red at T >= 2 only
    const sarm arms[] = {
        { "small f32",       ks_f32::kernel_loop_control_gate,  ks_f32::kernel_loop_control_out,  0 },
        { "small RED=1",     ks_red1::kernel_loop_control_gate, ks_red1::kernel_loop_control_out, 1 },
        { "small RED=2",     ks_red2::kernel_loop_control_gate, ks_red2::kernel_loop_control_out, 1 },
        { "small RED=3",     ks_red3::kernel_loop_control_gate, ks_red3::kernel_loop_control_out, 1 },
        { "small RED=4",     ks_red4::kernel_loop_control_gate, ks_red4::kernel_loop_control_out, 1 },
        { "small RED=6",     ks_red6::kernel_loop_control_gate, ks_red6::kernel_loop_control_out, 2 },
        { "small RED=7",     ks_red7::kernel_loop_control_gate, ks_red7::kernel_loop_control_out, 2 },
    };
    const double bound = 1e-6;
    double worst = 0.0, least[7];
    for (double & v : least) v = 1e30;
    for (int64_t nt : {1, 2, 3, 4, 5, 6, 7, 8}) {
        const case_data c = make_case(nt, 2000u + (uint32_t) nt);
        const std::vector<float> op    = cpu_eval(c, false);
        const std::vector<float> chain = cpu_eval(c, true);
        for (int a = 0; a < 7; ++a) {
            const sarm & A = arms[a];
            bool scratch_ok = false;
            const std::vector<float> k = replay_small(A.g, A.o, c, &scratch_ok);
            const double e_op = nmse(k, op), e_ch = nmse(k, chain);
            printf("replay %-12s nt %lld: nmse vs CPU op %.3g, vs chain %.3g (bound %.0e)\n", A.name, (long long) nt, e_op, e_ch, bound);
            CHECK(scratch_ok, "%s nt %lld: the scratch was read unwritten or written past its floats", A.name, (long long) nt);
            if (A.kind == 0) {
                worst = std::max(worst, std::max(e_op, e_ch));
                CHECK(e_op <= bound && e_ch <= bound, "%s nt %lld: above its bound", A.name, (long long) nt);
            } else if (A.kind == 1 || nt >= 2) {
                least[a] = std::min(least[a], e_op);
                CHECK(e_op > bound, "%s nt %lld: the red passes (nmse %.3g)", A.name, (long long) nt, e_op);
            } else {
                // a row-step or tail-row defect cannot show on one row: stated, and asserted
                CHECK(e_op <= bound, "%s nt 1: a defect of rows past the first shows on one row (nmse %.3g)", A.name, e_op);
            }
        }
    }
    printf("summary small: f32 worst %.3g; reds least", worst);
    for (int a = 1; a < 7; ++a) printf(" %s %.3g", arms[a].name + 6, least[a]);
    printf("\n");
}

}  // namespace

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && strcmp(argv[1], "quick") == 0;
    const std::vector<export_line> ex = parse_export();
    printf("export lines: %zu\n", ex.size());
    CHECK(ex.size() == 13, "expected 13 export lines (6 prompt, 7 of 1 .. 8 rows), got %zu", ex.size());
    {
        // 0226: the seven small-row lines are the rows the model now emits the op for
        int n_small = 0;
        for (const export_line & l : ex) n_small += l.ne[1] <= GGML_LOOP_CONTROL_SMALL_ROWS ? 1 : 0;
        CHECK(n_small == 7, "expected 7 export lines of at most 8 rows, got %d", n_small);
    }

    // 1. the rule, red first
    g_quiet = true;
    const int red = check_rule(rule_accept_all, ex);
    g_failures -= red;
    g_quiet = false;
    CHECK(red == 10, "the accept-all rule must fail the 10 refusal checks, failed %d", red);
    printf("rule: the accept-all red fails %d checks\n", red);
    const int green = check_rule(ggml_loop_control_refuse, ex);
    printf("rule: %d failures on the export's %zu calls and 10 refusals\n", green, ex.size());

    // 2. the plan
    check_plan(ex);

    // the knob's values: an unknown value is not a value (-1: the callers keep the arm off and warn),
    // so a mistyped "BF16" cannot run the f32 kernel (review F1)
    {
        struct kv { const char * v; int m; bool all; };
        const kv t[] = { {nullptr, 0, false}, {"0", 0, false}, {"off", 0, false}, {"1", 1, false}, {"f32", 1, false},
                         {"bf16", 2, false}, {"all-layers", 1, true}, {"BF16", -1, false}, {"on", -1, false}, {"2", -1, false} };
        for (const kv & e : t) {
            CHECK(ggml_loop_control_mode(e.v) == e.m && ggml_loop_control_all_layers(e.v) == e.all,
                  "knob value %s: mode %d all %d", e.v ? e.v : "(unset)", ggml_loop_control_mode(e.v), (int) ggml_loop_control_all_layers(e.v));
        }
    }

    // 3 / 4. the CPU op, the chain, the replay
    struct arm { const char * name; kernel_fn fn; bool green; double bound; };
    const arm arms[] = {
        { "f32",          k_f32::kernel_loop_control,  true,  1e-6 },
        { "bf16",         k_bf16::kernel_loop_control, true,  1e-4 },
        { "f32 RED=1",    k_red1::kernel_loop_control, false, 1e-6 },
        { "f32 RED=2",    k_red2::kernel_loop_control, false, 1e-6 },
        { "f32 RED=3",    k_red3::kernel_loop_control, false, 1e-6 },
        { "f32 RED=4",    k_red4::kernel_loop_control, false, 1e-6 },
        { "bf16 RED=5",   k_red5::kernel_loop_control, false, 1e-4 },
    };
    std::vector<int64_t> rows = { 1, 7, 9, 16, 57 };
    if (!quick) {
        rows.push_back(300);    // a partial 16-row tile at 20 cores (16 rows a group, the last 12)
        rows.push_back(1027);
    }
    double worst_cpu_chain = 0.0;
    double worst[7] = {0}, least_red[7];
    for (double & v : least_red) v = 1e30;
    for (int64_t nt : rows) {
        const case_data c = make_case(nt, 1000u + (uint32_t) nt);
        const std::vector<float> op    = cpu_eval(c, false);
        const std::vector<float> chain = cpu_eval(c, true);
        const double e_oc = nmse(op, chain);
        worst_cpu_chain = std::max(worst_cpu_chain, e_oc);
        CHECK(e_oc <= 1e-7, "nt %lld: the CPU op against the chain nmse %.3g > 1e-7", (long long) nt, e_oc);
        for (int a = 0; a < 7; ++a) {
            const arm & A = arms[a];
            if ((nt == 300 || nt == 1027) && !A.green) continue;   // the reds at the smaller row counts
            for (int cores : {20, 32}) {
                if (cores == 32 && !(A.green && (nt == 57 || nt == 300 || nt == 1027))) continue;
                const std::vector<float> k = replay(A.fn, c, cores);
                const double e_op = nmse(k, op), e_ch = nmse(k, chain);
                printf("replay %-10s nt %4lld cores %d: nmse vs CPU op %.3g, vs chain %.3g (bound %.0e)\n",
                       A.name, (long long) nt, cores, e_op, e_ch, A.bound);
                if (A.green) {
                    worst[a] = std::max(worst[a], std::max(e_op, e_ch));
                    CHECK(e_op <= A.bound && e_ch <= A.bound, "%s nt %lld: above its bound", A.name, (long long) nt);
                } else {
                    least_red[a] = std::min(least_red[a], e_op);
                    CHECK(e_op > A.bound, "%s nt %lld: the red passes (nmse %.3g)", A.name, (long long) nt, e_op);
                }
            }
        }
    }
    // 5. the small pair (0226)
    check_small_plan_and_switch();
    check_small_replay();

    printf("summary: CPU op vs chain worst %.3g; replay f32 worst %.3g, bf16 worst %.3g; reds least",
           worst_cpu_chain, worst[0], worst[1]);
    for (int a = 2; a < 7; ++a) printf(" %s %.3g", arms[a].name, least_red[a]);
    printf("\n");

    if (g_failures) {
        printf("loop-control: %d FAILED\n", g_failures);
        return 1;
    }
    printf("loop-control: all passed\n");
    return 0;
}
