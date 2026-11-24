// Patches 0204-0207's device-free cell: Inkling 2 Mini's depth mix as one op.
//
// Every shape here is read off the model's exported prefill graph
// (tests/depth_mix_export_lines.inc: test-export-graph-ops lines, verbatim, at ub 512,
// 1024 and 2048), and the tensors are built with the constructors the model calls, then
// compared against those lines' ne and nb. Four parts:
//
//  1. the eligibility rule (ggml_depth_mix_refuse) on the exported calls: it must ACCEPT
//     every prefill call the graph has (the main block store's k 1..4 with nq 1 and 3,
//     the nq 1 calls with the folded norm weight, the loop model's bank store with its
//     four message columns) and refuse the decode lines (T = 1) and the out-of-scope
//     variants. Red first, on every run: the same checks are run against two wrong rules
//     (the handed-over one, which refused a norm weight seen through a view, and one that
//     accepts everything) and each must FAIL them.
//
//  2. the launch plan (ggml_depth_mix_plan_make, the function the OpenCL dispatch binds
//     its arguments from): each argument's reach stays inside its tensor.
//
//  3. a host replay of kernels/depth_mix.cl ITSELF (the .cl text compiled as C++, 256
//     lanes as threads, barrier() a real barrier) at the plan's arguments, every tensor at
//     its own non-zero offset of one buffer as on the backend, against the CPU op. The
//     red build of the same text (DEPTH_MIX_RED_NW: the norm weight's offset dropped, the
//     card's ERR ~2 on the three with_norm op cases) must NOT match.
//
//  4. the card's page fault, named: the loop model's message() read the arm's dst
//     [n_embd, n_tokens, 4] with the chain's strides (row step nb[2] = n_embd*n_tokens*4
//     bytes: one row at the start of every [2304, T] slab, 512 slabs, the tensor ends
//     after 4). ggml_view_within_src flags that view and passes the fixed column view;
//     `arcint-depth-mix fault-before` computes the old read on the CPU backend (a crash:
//     under ASan a heap-buffer-overflow READ in the copy), `fault-after` the fixed one.
//
// No GPU, no OpenCL runtime.

#define GGML_USE_CPU 1

#include "ggml.h"
#include "ggml-cpu.h"

#include <pthread.h>

#include <math.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------------------
// depth_mix.cl on the host: the kernel's own text, twice (as shipped, and the red build)
// ---------------------------------------------------------------------------------------
namespace host_cl {
thread_local int g_group = 0;
thread_local int g_lane  = 0;
pthread_barrier_t g_bar;
inline int   get_group_id(int) { return g_group; }
inline int   get_local_id(int) { return g_lane; }
inline void  barrier(int)      { pthread_barrier_wait(&g_bar); }
inline float rsqrt(float x)    { return 1.0f/sqrtf(x); }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#define CLK_LOCAL_MEM_FENCE 0
#define kernel
#define global
#define local
#define ulong uint64_t
#define DEPTH_MIX_HOST_REPLAY 1
namespace k_green {
using namespace host_cl;
#include "depth_mix.cl"
}
#define DEPTH_MIX_RED_NW 1
namespace k_red_nw {
using namespace host_cl;
#include "depth_mix.cl"
}
#undef DEPTH_MIX_RED_NW
#undef kernel
#undef global
#undef local
#undef ulong
#pragma GCC diagnostic pop

namespace {

int g_failures = 0;
bool g_quiet = false;   // the red rules' runs count failures without printing them

#define CHECK(cond, what) do { \
        if (!(cond)) { if (!g_quiet) { printf("FAIL: %s\n", what); } ++g_failures; } \
    } while (0)

// ---------------------------------------------------------------------------------------
// the export lines
// ---------------------------------------------------------------------------------------
struct export_src {
    int      type;
    int64_t  ne[4];
    uint64_t nb[4];
};
struct export_line {
    const char * file;
    int          line;
    int          op;
    int          type;
    int64_t      ne[4];
    std::vector<export_src> src;
    std::string  name;
};
struct export_raw {
    const char * file;
    int          line;
    const char * text;
};
const export_raw k_export_raw[] = {
#include "depth_mix_export_lines.inc"
};

std::vector<export_line> parse_export() {
    std::vector<export_line> out;
    for (const export_raw & r : k_export_raw) {
        std::istringstream in(r.text);
        export_line l;
        l.file = r.file;
        l.line = r.line;
        int n_params = 0, n_src = 0;
        in >> l.op >> l.type >> l.ne[0] >> l.ne[1] >> l.ne[2] >> l.ne[3] >> n_params;
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
        std::getline(in, l.name);
        if (!in && l.src.empty()) {
            printf("FAIL: unparsed export line %s:%d\n", r.file, r.line);
            ++g_failures;
            continue;
        }
        out.push_back(l);
    }
    return out;
}

const char * file_of(int64_t nt) {
    return nt == 512 ? "inkling-ub512.txt" : nt == 1024 ? "inkling-ub1024.txt" : "inkling-ub2048.txt";
}

// the RMS_NORM line of a states view: [2304, states, T] at the store's slab step
const export_line * find_states(const std::vector<export_line> & ex, int64_t nt, int64_t t_line, int64_t states, uint64_t slab) {
    for (const export_line & l : ex) {
        if (strcmp(l.file, file_of(nt)) == 0 && l.op == GGML_OP_RMS_NORM && l.src.size() == 1 &&
            l.src[0].ne[0] == 2304 && l.src[0].ne[1] == states && l.src[0].ne[2] == t_line && l.src[0].nb[2] == slab) {
            return &l;
        }
    }
    return nullptr;
}
// the MUL_MAT line states x queries: [2304, states, T] x [2304, nq, T]
const export_line * find_scores(const std::vector<export_line> & ex, int64_t nt, int64_t states, int64_t nq) {
    for (const export_line & l : ex) {
        if (strcmp(l.file, file_of(nt)) == 0 && l.op == GGML_OP_MUL_MAT && l.src.size() == 2 &&
            l.src[0].ne[1] == states && l.src[0].ne[2] == nt && l.src[1].ne[1] == nq && l.src[1].ne[2] == nt) {
            return &l;
        }
    }
    return nullptr;
}
// the CPY line of cur into the store's last row: src0 is cur, src1 the row view
const export_line * find_cur(const std::vector<export_line> & ex, int64_t nt, uint64_t slab) {
    for (const export_line & l : ex) {
        if (strcmp(l.file, file_of(nt)) == 0 && l.op == GGML_OP_CPY && l.src.size() == 2 && l.src[1].nb[2] == slab) {
            return &l;
        }
    }
    return nullptr;
}
// the CONT line of a chain column read: src0 [2304, T] at row step nq * 9216
const export_line * find_column(const std::vector<export_line> & ex, int64_t nt, int64_t nq) {
    for (const export_line & l : ex) {
        if (strcmp(l.file, file_of(nt)) == 0 && l.op == GGML_OP_CONT && l.src.size() == 1 &&
            l.src[0].ne[1] == nt && l.src[0].nb[1] == (uint64_t) nq*9216) {
            return &l;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------------------
// the nodes of one mix call, built as src/models/inkling2.cpp's mix() builds them
// ---------------------------------------------------------------------------------------
struct call_shape {
    int64_t n_slot;   // the store's rows: 5 (res_stack_tensor) or 2 (the loop's bank_stack)
    int64_t k;        // block rows read
    int64_t nt;
    int64_t nq;
    bool    with_nw;
};

struct nodes {
    ggml_tensor * stack  = nullptr;
    ggml_tensor * blocks = nullptr;
    ggml_tensor * cur    = nullptr;
    ggml_tensor * q      = nullptr;
    ggml_tensor * nw     = nullptr;
};

nodes build_nodes(ggml_context * ctx, const call_shape & c, int64_t n_embd = 2304) {
    nodes r;
    r.stack = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, c.n_slot, c.nt);
    // mix(): the k block rows end one row before the store's last (cur's row in the chain)
    r.blocks = ggml_view_3d(ctx, r.stack, n_embd, c.k, c.nt, r.stack->nb[1], r.stack->nb[2],
                            (size_t) ((c.n_slot - 1) - c.k)*r.stack->nb[1]);
    r.cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, c.nt);
    // a weight of one column is a 1-D tensor in the model; mix() sees it through reshape_2d
    r.q = c.nq == 1 ? ggml_reshape_2d(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd), n_embd, 1)
                    : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, c.nq);
    if (c.with_nw) {
        // attn_norm / ffn_norm / output_norm: 1-D, through reshape_2d (a view)
        r.nw = ggml_reshape_2d(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd), n_embd, 1);
    }
    return r;
}

struct ctx_shapes {
    ggml_context * ctx;
    ctx_shapes() {
        ggml_init_params p = { (size_t) 4*1024*1024, nullptr, true };   // strides only: no storage
        ctx = ggml_init(p);
    }
    ~ctx_shapes() { ggml_free(ctx); }
};

// the prefill calls the exported graph has
std::vector<call_shape> production_calls() {
    std::vector<call_shape> v;
    for (int64_t nt : {512, 1024, 2048}) {
        for (int64_t k : {1, 2, 3, 4}) {
            v.push_back({5, k, nt, 1, false});   // a controller between the mix and the norm (the loop passes)
            v.push_back({5, k, nt, 1, true});    // an SWA layer's attn, the ffn, the output: the norm folded
            v.push_back({5, k, nt, 3, false});   // a full-attention layer's q, k, v
        }
        v.push_back({2, 1, nt, 4, false});       // the loop's messages on its bank store
    }
    return v;
}

typedef const char * (*rule_fn)(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *);

// the handed-over rule: it refused a norm weight seen through a view, which is every norm
// weight the model has (the defect it was meant to avoid was the kernel's missing offset)
const char * rule_handed_over(const ggml_tensor * b, const ggml_tensor * c, const ggml_tensor * q, const ggml_tensor * nw) {
    if (nw != nullptr && nw->view_src != nullptr) {
        return "nw-view";
    }
    return ggml_depth_mix_refuse(b, c, q, nw);
}
// a rule that accepts everything
const char * rule_accept_all(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) {
    return nullptr;
}

bool refused_as(const char * why, const char * want) {
    return why != nullptr && strcmp(why, want) == 0;
}

// part 1: returns the number of failed checks under `rule`
int check_rule(const std::vector<export_line> & ex, rule_fn rule, bool quiet) {
    const int before = g_failures;
    g_quiet = quiet;

    // ---- accept: every prefill call of the export, its nodes equal to the lines' ne / nb ----
    for (const call_shape & c : production_calls()) {
        ctx_shapes h;
        nodes r = build_nodes(h.ctx, c);
        const uint64_t slab = (uint64_t) c.n_slot*9216;
        char what[160];
        snprintf(what, sizeof(what), "store %d rows, k %d, T %d, nq %d, norm %d", (int) c.n_slot, (int) c.k, (int) c.nt, (int) c.nq, (int) c.with_nw);

        const export_line * st = find_states(ex, c.nt, c.nt, c.k + 1, slab);
        const export_line * sc = find_scores(ex, c.nt, c.k + 1, c.nq);
        const export_line * cu = find_cur(ex, c.nt, slab);
        if (st == nullptr || sc == nullptr || cu == nullptr) {
            if (!quiet) { printf("FAIL: no export line for %s\n", what); }
            ++g_failures;
            continue;
        }
        // the states view the chain norms is the block rows plus cur's row: same strides
        CHECK(r.blocks->ne[0] == st->src[0].ne[0] && r.blocks->ne[1] + 1 == st->src[0].ne[1] && r.blocks->ne[2] == st->src[0].ne[2],
              "the block rows are the exported states view less cur's row");
        CHECK(r.blocks->nb[0] == st->src[0].nb[0] && r.blocks->nb[1] == st->src[0].nb[1] && r.blocks->nb[2] == st->src[0].nb[2],
              "the block rows keep the exported strides (row 9216, slab the store's)");
        CHECK(r.blocks->view_offs + (uint64_t) c.k*r.blocks->nb[1] == (uint64_t) (c.n_slot - 1)*9216,
              "the block rows end where the chain's cur row starts");
        CHECK(r.cur->ne[0] == cu->src[0].ne[0] && r.cur->ne[1] == cu->src[0].ne[1] && r.cur->nb[1] == cu->src[0].nb[1],
              "cur is the exported copy's source: [2304, T], row step 9216");
        CHECK(cu->src[1].nb[1] == 9216 && cu->src[1].ne[1] == 1, "the exported copy's target is one row of the store");
        CHECK(r.q->ne[0] == sc->src[1].ne[0] && r.q->ne[1] == sc->src[1].ne[1], "the query weight is the exported repeat's column set");

        const char * why = rule(r.blocks, r.cur, r.q, r.nw);
        if (why != nullptr) {
            if (!quiet) { printf("FAIL: %s refused as %s (%s:%d %s)\n", what, why, st->file, st->line, st->name.c_str()); }
            ++g_failures;
        }
    }

    // ---- refuse: the export's decode lines (the same states views at T = 1) ----
    for (int64_t nt : {512, 1024, 2048}) {
        for (int64_t k : {1, 2, 3, 4}) {
            const export_line * st = find_states(ex, nt, 1, k + 1, 46080);
            CHECK(st != nullptr, "the export has the decode states line");
            ctx_shapes h;
            nodes r = build_nodes(h.ctx, {5, k, 1, 1, false});
            CHECK(refused_as(rule(r.blocks, r.cur, r.q, r.nw), "decode"), "an exported decode call (T = 1) is refused as decode");
        }
        const export_line * st = find_states(ex, nt, 1, 2, 18432);
        CHECK(st != nullptr, "the export has the bank's decode states line");
        ctx_shapes h;
        nodes r = build_nodes(h.ctx, {2, 1, 1, 4, false});
        CHECK(refused_as(rule(r.blocks, r.cur, r.q, r.nw), "decode"), "the bank's decode call is refused as decode");
    }

    // ---- refuse: out of scope, each by its own reason ----
    {
        ctx_shapes h;
        // a slab step shorter than the states' extent: two tokens' states overlap
        ggml_tensor * st  = ggml_new_tensor_3d(h.ctx, GGML_TYPE_F32, 2304, 5, 512);
        ggml_tensor * blk = ggml_view_3d(h.ctx, st, 2304, 4, 512, st->nb[1], 2*st->nb[1], 0);
        nodes r = build_nodes(h.ctx, {5, 4, 512, 1, false});
        CHECK(refused_as(rule(blk, r.cur, r.q, nullptr), "blocks-slab"), "an overlapping slab step is refused as blocks-slab");
        // the transposed states of the chain (the export's cont(transpose) source): rows not back to back
        ggml_tensor * tr = ggml_transpose(h.ctx, r.blocks);
        CHECK(rule(tr, r.cur, r.q, nullptr) != nullptr, "a transposed states view is refused");
        ggml_tensor * cur16 = ggml_new_tensor_2d(h.ctx, GGML_TYPE_F16, 2304, 512);
        CHECK(refused_as(rule(r.blocks, cur16, r.q, nullptr), "not-f32"), "an f16 cur is refused as not-f32");
        ggml_tensor * q16 = ggml_new_tensor_2d(h.ctx, GGML_TYPE_F16, 2304, 3);
        CHECK(refused_as(rule(r.blocks, r.cur, q16, nullptr), "not-f32"), "an f16 query weight is refused as not-f32");
        ggml_tensor * cur_short = ggml_new_tensor_2d(h.ctx, GGML_TYPE_F32, 2304, 256);
        CHECK(refused_as(rule(r.blocks, cur_short, r.q, nullptr), "cur-shape"), "a cur of another token count is refused as cur-shape");
        ggml_tensor * q5 = ggml_new_tensor_2d(h.ctx, GGML_TYPE_F32, 2304, 5);
        CHECK(refused_as(rule(r.blocks, r.cur, q5, nullptr), "q-shape"), "five query columns are refused as q-shape");
        ggml_tensor * nw3 = ggml_new_tensor_2d(h.ctx, GGML_TYPE_F32, 2304, 3);
        CHECK(refused_as(rule(r.blocks, r.cur, r.q, nw3), "nw-shape"), "three norm columns for one query are refused as nw-shape");
    }
    {
        ctx_shapes h;
        nodes r = build_nodes(h.ctx, {5, 4, 512, 1, false}, 1000);   // a width off the lane rule
        CHECK(refused_as(rule(r.blocks, r.cur, r.q, nullptr), "d-lanes"), "a width off 256 is refused as d-lanes");
        nodes w = build_nodes(h.ctx, {5, 4, 512, 1, false}, 2560);   // 10 floats a lane
        CHECK(refused_as(rule(w.blocks, w.cur, w.q, nullptr), "d-lanes"), "ten floats a lane are refused as d-lanes");
        nodes s = build_nodes(h.ctx, {8, 7, 512, 1, false});         // more block rows than the registers hold
        CHECK(refused_as(rule(s.blocks, s.cur, s.q, nullptr), "k"), "seven block rows are refused as k");
    }

    g_quiet = false;
    return g_failures - before;
}

// ---------------------------------------------------------------------------------------
// part 2 and 3: the plan, and depth_mix.cl replayed at it
// ---------------------------------------------------------------------------------------
uint32_t g_rng = 0x2545F491u;
float frand() {   // [-1, 1)
    g_rng = g_rng*1664525u + 1013904223u;
    return (float) (int32_t) g_rng*(1.0f/2147483648.0f);
}
void fill(ggml_tensor * t, float lo, float hi) {
    float * d = (float *) t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        d[i] = lo + (hi - lo)*(0.5f + 0.5f*frand());
    }
}

typedef void (*kernel_fn)(char *, uint64_t, uint64_t, uint64_t, char *, uint64_t, uint64_t, char *, uint64_t, uint64_t,
                          char *, uint64_t, uint64_t, int, char *, uint64_t, uint64_t, uint64_t, int, int, int,
                          float, float, float, float *);

struct launch {
    kernel_fn fn;
    char *    base;        // the one buffer every source tensor lives in
    uint64_t  offb, offc, offq, offn;
    char *    out;         // the replay's own dst, at offset 0
    ggml_depth_mix_plan plan;
    float     eps_states, eps_out, scale;
    float *   red;
};

// a work-group at a time, 256 lanes as threads; every lane passes the same barriers
void run_groups(const launch & L, const std::vector<int> & tokens) {
    pthread_barrier_init(&host_cl::g_bar, nullptr, 256);
    std::vector<std::thread> lanes;
    lanes.reserve(256);
    for (int lane = 0; lane < 256; ++lane) {
        lanes.emplace_back([&L, &tokens, lane] {
            for (int t : tokens) {
                host_cl::g_group = t;
                host_cl::g_lane  = lane;
                L.fn(L.base, L.offb, L.plan.nb1, L.plan.nb2, L.base, L.offc, L.plan.nbc1, L.base, L.offq, L.plan.nbq1,
                     L.base, L.offn, L.plan.nbn1, L.plan.has_nw, L.out, 0, L.plan.nbd1, L.plan.nbd2,
                     L.plan.ne0, L.plan.k, L.plan.nq, L.eps_states, L.eps_out, L.scale, L.red);
                pthread_barrier_wait(&host_cl::g_bar);   // the group ends for every lane before the next starts
            }
        });
    }
    for (std::thread & th : lanes) {
        th.join();
    }
    pthread_barrier_destroy(&host_cl::g_bar);
}

// normalized squared error of the replayed tokens against the CPU op's dst
double nmse_tokens(const float * got, const ggml_tensor * dst, const std::vector<int> & tokens) {
    double num = 0.0, den = 0.0;
    for (int64_t q = 0; q < dst->ne[2]; ++q) {
        for (int t : tokens) {
            const size_t off = ((size_t) q*dst->nb[2] + (size_t) t*dst->nb[1])/sizeof(float);
            const float * a = got + off;
            const float * b = (const float *) dst->data + off;
            for (int64_t i = 0; i < dst->ne[0]; ++i) {
                num += ((double) a[i] - b[i])*((double) a[i] - b[i]);
                den += (double) b[i]*b[i];
            }
        }
    }
    return den > 0.0 ? num/den : 1.0;
}

void check_plan_and_replay(const call_shape & c, bool replay) {
    char what[160];
    snprintf(what, sizeof(what), "store %d rows, k %d, T %d, nq %d, norm %d", (int) c.n_slot, (int) c.k, (int) c.nt, (int) c.nq, (int) c.with_nw);

    const size_t tensors = (size_t) 2304*4*(size_t) (c.n_slot*c.nt + c.nt + c.nq*c.nt + 2*c.nq + 16);
    ggml_init_params p = { (replay ? tensors : 0) + (size_t) 8*1024*1024, nullptr, !replay };
    ggml_context * ctx = ggml_init(p);

    // a weight is not the first thing in its buffer on the card either: something before it
    if (replay) {
        fill(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2304*3), -7.0f, 7.0f);
    }
    nodes r = build_nodes(ctx, c);
    const float eps_states = 1e-6f, eps_out = 1e-5f, scale = 1.0f/sqrtf(2304.0f);
    ggml_tensor * dst = ggml_depth_mix(ctx, r.blocks, r.cur, r.q, r.nw, eps_states, eps_out, scale);

    // ---- part 2: the plan ----
    ggml_depth_mix_plan plan;
    ggml_depth_mix_plan_make(dst, &plan);
    CHECK(plan.ne0 == 2304 && plan.k == c.k && plan.nq == c.nq && plan.nt == c.nt && plan.has_nw == (c.with_nw ? 1 : 0), "the plan's counts are the call's");
    CHECK(plan.nb1 == 9216 && plan.nb2 == (uint64_t) c.n_slot*9216 && plan.nbc1 == 9216, "the plan's strides are the export's");
    CHECK(plan.nbd1 == 9216 && plan.nbd2 == (uint64_t) 9216*c.nt, "dst is [n_embd, n_tokens, nq]: a column's token block contiguous");
    CHECK(r.blocks->view_offs + plan.reach_blocks <= ggml_nbytes(r.stack), "the block rows' reach stays in the store");
    CHECK(plan.reach_cur == ggml_nbytes(r.cur), "cur's reach is cur");
    CHECK(plan.reach_queries == ggml_nbytes(r.q), "the queries' reach is the weight");
    CHECK(plan.reach_dst == ggml_nbytes(dst), "dst's reach is dst");
    CHECK(!c.with_nw || plan.reach_norm_w == ggml_nbytes(r.nw), "the norm weight's reach is the weight");
    CHECK(plan.local_floats == (uint64_t) ((c.k + 1)*(1 + c.nq) + c.nq)*256, "local memory: no slot shared by the two reductions");

    if (replay) {
        // ---- part 3: the kernel's text at the plan, against the CPU op ----
        fill(r.stack, -1.0f, 1.0f);
        fill(r.cur, -1.0f, 1.0f);
        fill(r.q, -4.0f, 4.0f);
        if (r.nw) {
            fill(r.nw, 0.5f, 1.5f);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);
        ggml_graph_compute_with_ctx(ctx, gf, 4);

        const std::vector<int> tokens = { 0, 1, (int) c.nt/2 + 3, (int) c.nt - 1 };
        std::vector<float> out((size_t) ggml_nelements(dst), 0.0f);
        std::vector<float> red((size_t) plan.local_floats, 0.0f);

        launch L;
        L.base = (char *) ggml_get_mem_buffer(ctx);
        L.offb = (uint64_t) ((char *) r.blocks->data - L.base);
        L.offc = (uint64_t) ((char *) r.cur->data - L.base);
        L.offq = (uint64_t) ((char *) r.q->data - L.base);
        L.offn = r.nw ? (uint64_t) ((char *) r.nw->data - L.base) : 0;
        L.out  = (char *) out.data();
        L.plan = plan;
        L.eps_states = eps_states;
        L.eps_out    = eps_out;
        L.scale      = scale;
        L.red        = red.data();
        CHECK(L.offb > 0 && L.offc > 0 && L.offq > 0 && (!r.nw || L.offn > 0), "every source sits at a non-zero offset of the buffer");

        L.fn = k_green::kernel_depth_mix_f32;
        run_groups(L, tokens);
        const double e = nmse_tokens(out.data(), dst, tokens);
        if (!(e <= 1e-9)) {
            printf("FAIL: depth_mix.cl replay vs the CPU op, %s: nmse %.3g\n", what, e);
            ++g_failures;
        }

        if (r.nw) {
            // red: the same text with the weight's offset dropped must not match
            std::fill(out.begin(), out.end(), 0.0f);
            L.fn = k_red_nw::kernel_depth_mix_f32;
            run_groups(L, tokens);
            const double er = nmse_tokens(out.data(), dst, tokens);
            if (!(er > 1e-2)) {
                printf("FAIL: the red kernel (norm weight offset dropped) still matches, %s: nmse %.3g\n", what, er);
                ++g_failures;
            }
            printf("replay %s: nmse %.2g, red (offset dropped) %.3g\n", what, e, er);
        } else {
            printf("replay %s: nmse %.2g\n", what, e);
        }
    }
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------------------
// part 4: the page fault
// ---------------------------------------------------------------------------------------
// the read src/models/inkling2.cpp's message() made before the fix, on whatever mix() returned
ggml_tensor * message_view_before(ggml_context * ctx, ggml_tensor * messages, int64_t ctrl, int64_t n_embd, int64_t n_tokens) {
    return ggml_view_2d(ctx, messages, n_embd, n_tokens, messages->nb[2], ctrl*messages->nb[1]);
}

typedef bool (*within_fn)(const ggml_tensor *);
bool within_always(const ggml_tensor *) { return true; }

int check_fault_shapes(const std::vector<export_line> & ex, within_fn within, bool quiet) {
    const int before = g_failures;
    g_quiet = quiet;
    for (int64_t nt : {512, 1024, 2048}) {
        ctx_shapes h;
        // the chain's messages: mul_mat(cont(transpose(states)), scores) = [n_embd, 4, T]; its
        // column read is the export's CONT line (row step 4 * 9216)
        const export_line * col = find_column(ex, nt, 4);
        CHECK(col != nullptr, "the export has the message column's copy");
        ggml_tensor * chain = ggml_new_tensor_3d(h.ctx, GGML_TYPE_F32, 2304, 4, nt);
        ggml_tensor * cv = message_view_before(h.ctx, chain, 3, 2304, nt);
        CHECK(col != nullptr && cv->nb[1] == col->src[0].nb[1] && cv->ne[1] == col->src[0].ne[1], "on the chain's result the read is the exported one");
        CHECK(within(cv), "on the chain's result the read stays inside");

        // the arm's messages: the loop's bank store (2 rows, k 1) and its four queries
        nodes r = build_nodes(h.ctx, {2, 1, nt, 4, false});
        ggml_tensor * dst = ggml_depth_mix(h.ctx, r.blocks, r.cur, r.q, nullptr, 1e-6f, 1e-5f, 1.0f/48.0f);
        for (int64_t ctrl = 0; ctrl < 4; ++ctrl) {
            ggml_tensor * old = message_view_before(h.ctx, dst, ctrl, 2304, nt);
            // the fault's signature: a row a slab, the slab [2304, T] f32
            CHECK(old->nb[1] == (uint64_t) 2304*nt*4, "the old read steps one [2304, T] slab a token");
            CHECK(!within(old), "the old read of the arm's dst leaves the tensor: flagged");
            ggml_tensor * fixed = ggml_depth_mix_column(h.ctx, dst, ctrl);
            CHECK(within(fixed), "the fixed column view stays inside");
            CHECK(ggml_is_contiguous(fixed) && fixed->ne[0] == 2304 && fixed->ne[1] == nt, "the fixed column view is [n_embd, n_tokens], no copy");
            CHECK(fixed->view_offs == (size_t) ctrl*2304*nt*4, "column c starts at c token blocks");
        }
    }
    g_quiet = false;
    return g_failures - before;
}

// the loop's message path on the CPU backend at the export's ub 512 shapes; `before` reads
// the arm's dst as the old message() did
int run_fault(bool before) {
    const int64_t nt = 512, n_embd = 2304;
    const size_t bytes = (size_t) n_embd*4*(size_t) (2*nt + nt + 4*nt + 4*nt + 8);
    ggml_init_params p = { bytes + (size_t) 4*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(p);
    nodes r = build_nodes(ctx, {2, 1, nt, 4, false});
    fill(r.stack, -1.0f, 1.0f);
    fill(r.cur, -1.0f, 1.0f);
    fill(r.q, -4.0f, 4.0f);
    ggml_tensor * dst = ggml_depth_mix(ctx, r.blocks, r.cur, r.q, nullptr, 1e-6f, 1e-5f, 1.0f/48.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> msgs;
    for (int64_t ctrl = 0; ctrl < 4; ++ctrl) {
        ggml_tensor * m = before ? ggml_cont(ctx, message_view_before(ctx, dst, ctrl, n_embd, nt))
                                 : ggml_cont(ctx, ggml_depth_mix_column(ctx, dst, ctrl));
        ggml_build_forward_expand(gf, m);
        msgs.push_back(m);
    }
    printf("message path, %s: computing on the CPU backend\n", before ? "the old read (expected to crash)" : "the fixed read");
    fflush(stdout);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    int bad = 0;
    for (int64_t ctrl = 0; ctrl < 4; ++ctrl) {
        const float * want = (const float *) dst->data + (size_t) ctrl*n_embd*nt;
        bad += memcmp(msgs[ctrl]->data, want, (size_t) n_embd*nt*4) != 0;
    }
    printf("message path: %d of 4 columns differ from the op's\n", bad);
    ggml_free(ctx);
    return bad;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "fault-before") == 0) {
        return run_fault(true) == 0 ? 0 : 1;
    }
    if (argc > 1 && strcmp(argv[1], "fault-after") == 0) {
        return run_fault(false) == 0 ? 0 : 1;
    }

    const std::vector<export_line> ex = parse_export();
    const int unparsed = g_failures;

    // ---- part 1: the rule, red first ----
    const int red_old = check_rule(ex, rule_handed_over, true);
    const int red_all = check_rule(ex, rule_accept_all, true);
    g_failures = unparsed;
    CHECK(ex.size() == 69, "69 export lines: 23 a ubatch size");
    printf("rule red (handed-over, refuses a view-backed norm weight): %d checks fail\n", red_old);
    printf("rule red (accepts everything): %d checks fail\n", red_all);
    CHECK(red_old == 12, "the handed-over rule fails exactly the 12 folded-norm calls");
    CHECK(red_all >= 20, "an accept-all rule fails the refusals");
    const int green = check_rule(ex, ggml_depth_mix_refuse, false);
    printf("rule green: %d checks fail over %d exported prefill calls\n", green, (int) production_calls().size());

    // the switches
    CHECK(!ggml_depth_mix_switch(nullptr) && !ggml_depth_mix_switch("0") && !ggml_depth_mix_switch("off"), "unset, 0 and off are off");
    CHECK(ggml_depth_mix_switch("1") && ggml_depth_mix_switch("nofold"), "1 and nofold turn the arm on");
    CHECK(ggml_depth_mix_fold_switch("1") && !ggml_depth_mix_fold_switch("nofold") && !ggml_depth_mix_fold_switch(nullptr) &&
          !ggml_depth_mix_fold_switch("0"), "the fold is on with the arm, off at nofold");

    // ---- part 4: the fault's read, red first ----
    const int fred = check_fault_shapes(ex, within_always, true);
    g_failures -= fred;
    printf("view-reach red (a check that passes every view): %d checks fail\n", fred);
    CHECK(fred == 12, "a reach check that passes everything misses the 12 old reads");
    check_fault_shapes(ex, ggml_view_within_src, false);
    CHECK(run_fault(false) == 0, "the fixed message path returns the op's columns");

    // ---- parts 2 and 3: the plan on every call, the kernel replay on ub 512's calls and
    // three of ub 2048's ----
    for (const call_shape & c : production_calls()) {
        const bool replay = c.nt == 512 || (c.nt == 2048 && (c.k == 4 || c.n_slot == 2));
        check_plan_and_replay(c, replay);
    }

    if (g_failures != 0) {
        printf("%d depth-mix check(s) failed\n", g_failures);
        return 1;
    }
    printf("depth-mix: all OK device-free\n");
    return 0;
}
