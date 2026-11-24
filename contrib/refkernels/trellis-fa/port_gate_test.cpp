// port_gate_test.cpp - the port arm's eligibility rules (fa_port_gate.h), on the real graph export.
//
// Reads test-export-graph-ops files of the model (tests/test-export-graph-ops.cpp's format:
// op, type, ne[4], the op params, then each source as type, ne[4], nb[4], the node name last)
// and feeds the gate each FLASH_ATTN_EXT line's own tensors:
//   q, k, v, mask      the line's sources 0..3, ne AND nb as exported (the model's permuted views);
//   the bias table     the MUL_MAT whose output is [n_bins, n_head, rows] (line 91 / 92), with the
//                      strides it has as a source of the SUB that consumes it (line 11 / 12);
//   the per-query sink the MUL_MAT whose output is [n_head, rows] (line 79 / 80), an op output,
//                      so contiguous; its head stride is cross-checked against the SUB's source 1;
// and synthesizes only what patches 0193 / 0194 add to the graph and no export of today's graph
// can hold: the indices' shape [n_kv, rows], kpos / qpos, the group classes, and the op's own
// output [D, n_head, rows] as ggml_flash_attn_ext creates it.
//
// Expectations. The export holds four FLASH_ATTN_EXT lines a file, 220..223: 220 / 221 are the
// decode step (1 query row) and must be refused "decode"; 222 / 223 are the prompt batch (the
// SWA ring and the global cache) and must be ACCEPTED. Each accepted line is then bent, one
// property at a time, into a call the kernel cannot serve, and each must be refused with its
// reason. The contiguous layout (0199's former cases) must be accepted too: the prepass and
// the block reads take both.
//
// Red first: built with -DFA_PORT_GATE_RED the header carries 0198's original contiguity rule
// and this test must fail on 222 / 223 with "layout" (the card's the first card window decline).
//
//   g++ -O2 -I <llama-tree>/ggml/include -I <llama-tree>/ggml/src/ggml-opencl port_gate_test.cpp -o port_gate_test
//   ./port_gate_test inkling-ub512.txt inkling-ub1024.txt inkling-ub2048.txt
#include "fa_port_gate.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct src_t { int type; int64_t ne[4]; size_t nb[4]; };
struct line_t {
    int no = 0, op = 0, type = 0;
    int64_t ne[4] = {0, 0, 0, 0};
    std::vector<src_t> src;
    std::string name;
};

static std::vector<line_t> read_lines(const char * path) {
    std::vector<line_t> out;
    std::ifstream f(path);
    std::string s;
    int no = 0;
    while (std::getline(f, s)) {
        ++no;
        std::istringstream is(s);
        line_t l; l.no = no;
        int64_t np = 0, ns = 0, dummy = 0;
        if (!(is >> l.op >> l.type >> l.ne[0] >> l.ne[1] >> l.ne[2] >> l.ne[3] >> np)) continue;
        for (int64_t i = 0; i < np; ++i) is >> dummy;
        if (!(is >> ns)) continue;
        for (int64_t i = 0; i < ns; ++i) {
            src_t x;
            is >> x.type >> x.ne[0] >> x.ne[1] >> x.ne[2] >> x.ne[3] >> x.nb[0] >> x.nb[1] >> x.nb[2] >> x.nb[3];
            l.src.push_back(x);
        }
        is >> l.name;
        out.push_back(l);
    }
    return out;
}

static struct ggml_tensor tensor_of(int type, const int64_t * ne, const size_t * nb) {
    struct ggml_tensor t;
    memset(&t, 0, sizeof t);
    t.type = (enum ggml_type) type;
    for (int i = 0; i < 4; ++i) { t.ne[i] = ne[i]; t.nb[i] = nb[i]; }
    return t;
}
static struct ggml_tensor from_src(const src_t & s) { return tensor_of(s.type, s.ne, s.nb); }
static struct ggml_tensor contiguous(int type, int64_t n0, int64_t n1, int64_t n2, size_t esz) {
    const int64_t ne[4] = { n0, n1, n2, 1 };
    const size_t  nb[4] = { esz, esz*n0, esz*n0*n1, esz*n0*n1*n2 };
    return tensor_of(type, ne, nb);
}

struct call_t {
    struct ggml_tensor q, k, v, mask, dst, table, indices, sinks, kpos, qpos, cflag, rflag;
    size_t align = 4096, wg_max = 256;
    int intel = 1, paged = 0, bias = 1;
    float max_bias = 0.0f, softcap = 0.0f;
    int32_t window = 0;
    bool no_sinks = false;
};
static int gate(const call_t & c, const char ** why) {
    *why = "";
    return fa_port_gate_why(&c.q, &c.k, &c.v, &c.mask, &c.dst, &c.table, &c.indices, c.no_sinks ? nullptr : &c.sinks,
                            &c.kpos, &c.qpos, &c.cflag, &c.rflag, c.align, c.wg_max, c.intel, c.paged, c.bias,
                            c.max_bias, c.softcap, c.window, why);
}

static int fails = 0;
static void expect(const char * file, int no, const char * what, const call_t & c, bool want_ok, const char * want_why) {
    const char * why = "";
    const int ok = gate(c, &why);
    const bool good = want_ok ? ok != 0 : (ok == 0 && strcmp(why, want_why) == 0);
    printf("  %s line %d %-26s %s%s%s\n", file, no, what, ok ? "ACCEPT" : "refuse ", ok ? "" : why,
           good ? "" : (want_ok ? "   <-- FAIL, expected ACCEPT" : "   <-- FAIL, expected another verdict"));
    if (!good) ++fails;
}

enum { OP_SUB = 6, OP_MUL_MAT = 29, OP_FLASH_ATTN_EXT = 75, T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_I8 = 24, T_I32 = 26 };

int main(int argc, char ** argv) {
    if ((int) GGML_OP_FLASH_ATTN_EXT != OP_FLASH_ATTN_EXT || (int) GGML_OP_MUL_MAT != OP_MUL_MAT || (int) GGML_OP_SUB != OP_SUB ||
        (int) GGML_TYPE_I8 != T_I8 || (int) GGML_TYPE_I32 != T_I32 || (int) GGML_TYPE_Q8_0 != T_Q8_0) {
        printf("FAIL: the export's op / type numbers are not this tree's\n");
        return 1;
    }
    if (argc < 2) { printf("usage: port_gate_test <export.txt> ...\n"); return 1; }
    for (int a = 1; a < argc; ++a) {
        const char * file = strrchr(argv[a], '/') ? strrchr(argv[a], '/') + 1 : argv[a];
        const std::vector<line_t> lines = read_lines(argv[a]);
        int n_fa = 0, n_prefill = 0, n_decode = 0;
        for (const line_t & l : lines) {
            if (l.op != OP_FLASH_ATTN_EXT) continue;
            ++n_fa;
            if (l.no < 220 || l.no > 223 || l.src.size() < 4) {
                printf("  %s line %d: FAIL, a FLASH_ATTN_EXT line outside 220..223 or without q/k/v/mask\n", file, l.no);
                ++fails; continue;
            }
            const int64_t n_head = l.ne[1], rows = l.ne[2], n_kv = l.src[1].ne[1];
            // the table and the sink of this row count, from their own lines
            const line_t * tl = nullptr, * sl = nullptr, * ul = nullptr;
            for (const line_t & x : lines) {
                if (x.op == OP_MUL_MAT && x.ne[0] >= 1024 && x.ne[1] == n_head && x.ne[2] == rows) tl = &x;
                if (x.op == OP_MUL_MAT && x.ne[0] == n_head && x.ne[1] == rows && x.ne[2] == 1) sl = &x;
                if (x.op == OP_SUB && x.src.size() == 2 && x.src[0].ne[0] >= 1024 && x.src[0].ne[1] == n_head &&
                    x.src[0].ne[2] == rows) ul = &x;
            }
            if (!tl || !sl || !ul || ul->src[0].ne[0] != tl->ne[0] ||
                ul->src[1].nb[2] != (size_t) n_head * 4 || ul->src[1].ne[1] != n_head) {
                printf("  %s line %d: FAIL, no table / sink / consumer line for %lld rows\n", file, l.no, (long long) rows);
                ++fails; continue;
            }
            call_t c;
            c.q = from_src(l.src[0]); c.k = from_src(l.src[1]); c.v = from_src(l.src[2]); c.mask = from_src(l.src[3]);
            c.table   = from_src(ul->src[0]);                                  // [n_bins, n_head, rows] as consumed
            c.sinks   = contiguous(T_F32, sl->ne[0], sl->ne[1], 1, 4);         // [n_head, rows], an op output
            c.dst     = contiguous(T_F32, l.ne[0], l.ne[1], l.ne[2], 4);
            c.indices = contiguous(T_I32, n_kv, rows, 1, 4);
            c.kpos    = contiguous(T_I32, n_kv, 1, 1, 4);
            c.qpos    = contiguous(T_I32, rows, 1, 1, 4);
            c.cflag   = contiguous(T_I8, (n_kv + 15) / 16, 1, 1, 1);
            c.rflag   = contiguous(T_I8, (rows + 7) / 8, 1, 1, 1);
            c.window  = n_kv == rows + 1024 ? 1024 : 0;                        // the SWA ring is window + batch keys

            char tag[64];
            snprintf(tag, sizeof tag, "(q%lld x kv%lld)", (long long) rows, (long long) n_kv);
            if (rows <= 8) {
                ++n_decode;
                expect(file, l.no, tag, c, false, "decode");
                continue;
            }
            ++n_prefill;
            if (l.src[0].nb[1] != (size_t) n_head * 512 || l.src[0].nb[2] != 512 ||
                l.src[1].nb[1] != (size_t) l.src[1].ne[2] * 256 || l.src[1].nb[2] != 256) {
                printf("  %s line %d: FAIL, not the permuted views this test is about\n", file, l.no);
                ++fails;
            }
            expect(file, l.no, tag, c, true, "");
#ifndef FA_PORT_GATE_RED
            // out of scope, one property at a time
            { call_t m = c; m.k.type = GGML_TYPE_Q8_0;                  expect(file, l.no, "  K q8_0", m, false, "kv-type"); }
            { call_t m = c; m.q.ne[0] = m.k.ne[0] = m.v.ne[0] = 256; expect(file, l.no, "  head 256", m, false, "head"); }
            { call_t m = c; m.q.ne[3] = 2;                        expect(file, l.no, "  two streams", m, false, "streams"); }
            { call_t m = c; m.k.nb[1] = 300;                      expect(file, l.no, "  K row pitch 300", m, false, "layout-k"); }
            { call_t m = c; m.v.nb[2] = 200;                      expect(file, l.no, "  V head stride 200", m, false, "layout-v"); }
            { call_t m = c; m.k.view_offs = 32;                   expect(file, l.no, "  K view offset 32", m, false, "layout-k"); }
            { call_t m = c; m.v.view_offs = 32;                   expect(file, l.no, "  V view offset 32", m, false, "layout-v"); }
            { call_t m = c; m.k.view_offs = 4096 + 16;            expect(file, l.no, "  K view offset 4112", m, false, "layout-k"); }
            { call_t m = c; m.k.view_offs = 2048 * 512;           expect(file, l.no, "  K view 512 rows in", m, true, ""); }
            { call_t m = c; m.wg_max = 32;                        expect(file, l.no, "  a 32-lane work-group", m, false, "work-group"); }
            { call_t m = c; m.k.ne[2] = m.v.ne[2] = 1;            expect(file, l.no, "  24 heads on 1 (G 24)", m, false, "work-group"); }
            { call_t m = c; m.q.nb[0] = 8;                        expect(file, l.no, "  Q element stride 8", m, false, "layout-q"); }
            { call_t m = c; m.align = 32;                         expect(file, l.no, "  buffer alignment 32", m, false, "layout-align"); }
            { call_t m = c; m.table = contiguous(T_F32, n_head, c.table.ne[0], rows, 4);
                                                                  expect(file, l.no, "  table [head, bin, row]", m, false, "bias-table"); }
            { call_t m = c; m.table.nb[1] *= 2;                   expect(file, l.no, "  table strided", m, false, "bias-table-layout"); }
            { call_t m = c; m.no_sinks = true;                    expect(file, l.no, "  no per-query sink", m, false, "no-sinks"); }
            { call_t m = c; m.sinks = contiguous(T_F32, n_head, 1, 1, 4); expect(file, l.no, "  a per-head sink", m, false, "sinks"); }
            { call_t m = c; m.mask.nb[1] = (size_t) n_kv;         expect(file, l.no, "  mask rows overlap", m, false, "mask"); }
            { call_t m = c; m.paged = 1;                          expect(file, l.no, "  paged K/V", m, false, "paged"); }
            { call_t m = c; m.intel = 0;                          expect(file, l.no, "  not an Intel device", m, false, "device"); }
            { call_t m = c; m.softcap = 30.0f;                    expect(file, l.no, "  a logit softcap", m, false, "softcap"); }
            { call_t m = c; m.window = -1;                        expect(file, l.no, "  window -1", m, false, "window"); }
            { call_t m = c; m.cflag = contiguous(T_I8, n_kv / 16 - 1, 1, 1, 1); expect(file, l.no, "  short key classes", m, false, "flags"); }
            { call_t m = c; m.dst.nb[1] = 1024;                   expect(file, l.no, "  dst head stride 1024", m, false, "dst"); }
            { call_t m = c; m.dst.ne[1] = n_head / 3;             expect(file, l.no, "  dst of 8 heads", m, false, "dst"); }
            { call_t m = c; m.kpos.type = GGML_TYPE_F32;          expect(file, l.no, "  f32 key positions", m, false, "positions"); }
            { call_t m = c; m.qpos = contiguous(T_I32, rows - 1, 1, 1, 4); expect(file, l.no, "  short row positions", m, false, "positions"); }
            { call_t m = c; m.rflag = contiguous(T_I8, (rows + 7) / 8 - 1, 1, 1, 1); expect(file, l.no, "  short row classes", m, false, "flags"); }
            // the row boundary: 8 rows keep today's kernels, 9 are the port's
            { call_t m = c; m.q.ne[1] = 8;                        expect(file, l.no, "  8 query rows", m, false, "decode"); }
            { call_t m = c; m.q.ne[1] = 9;                        expect(file, l.no, "  9 query rows", m, true, ""); }
            // the op-level gate (what supports_op and the launch both call) is the same verdict,
            // and the routing: a 64-row call of this layout reaches the launch whole when the
            // port is taken first, and as 8-row blocks ("decode") in the dispatch order before
            {
                call_t m = c; m.q.ne[1] = 64;
                struct ggml_tensor op = m.dst;
                op.ne[2] = 64;
                op.src[0] = &m.q; op.src[1] = &m.k; op.src[2] = &m.v; op.src[3] = &m.mask;
                op.src[FA_PORT_SRC_TABLE] = &m.table; op.src[FA_PORT_SRC_IDX] = &m.indices; op.src[FA_PORT_SRC_SINK_Q] = &m.sinks;
                op.src[FA_PORT_SRC_KPOS] = &m.kpos; op.src[FA_PORT_SRC_QPOS] = &m.qpos;
                op.src[FA_PORT_SRC_CFLAG] = &m.cflag; op.src[FA_PORT_SRC_RFLAG] = &m.rflag;
                const int32_t one = 1;
                memcpy((char *) op.op_params + FA_PORT_PARAM_BIAS_FLAG * sizeof(int32_t), &one, sizeof one);
                memcpy((char *) op.op_params + FA_PORT_PARAM_WINDOW * sizeof(int32_t), &m.window, sizeof m.window);
                const char * w0 = "", * w1 = "", * w2 = "";
                const int sup  = fa_port_gate_op(&op, m.align, m.wg_max, 1, &w0);
                const int now  = fa_port_gate_at_launch(&op, 1, 0, 1, m.align, m.wg_max, 1, &w1);
                const int then = fa_port_gate_at_launch(&op, 0, 0, 1, m.align, m.wg_max, 1, &w2);
                const bool good = fa_port_carries(&op) && sup && now && !then && strcmp(w2, "decode") == 0;
                printf("  %s line %d   64 rows: supports_op %s, launch %s; the old order: %s%s%s\n", file, l.no,
                       sup ? "ACCEPT" : w0, now ? "ACCEPT" : w1, then ? "ACCEPT" : "refuse ", then ? "" : w2,
                       good ? "" : "   <-- FAIL");
                if (!good) ++fails;
            }
            // the contiguous layout is served too
            { call_t m = c;
              m.q = contiguous(T_F32, 128, rows, n_head, 4);
              m.k = contiguous(T_F16, 128, n_kv, c.k.ne[2], 2); m.v = m.k;
              expect(file, l.no, "  contiguous q/k/v", m, true, ""); }
#endif
        }
        if (n_fa != 4 || n_prefill != 2 || n_decode != 2) {
            printf("  %s: FAIL, %d FLASH_ATTN_EXT lines (%d prompt, %d decode), expected 4 (2, 2)\n", file, n_fa, n_prefill, n_decode);
            ++fails;
        }
    }
    printf("%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
