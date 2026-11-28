// dec27_gate_test.cpp - 0227: the decode FA arm's gate and key-split rule (fa_dec27_gate.h), on the
// model's own graph export.
//
// fixtures/q38-27b-q8q4-fa.txt holds the two FLASH_ATTN_EXT lines of the dense 27B hybrid's export
// (test-export-graph-ops on the CPU: context 4,096, ubatch 512, flash attention on, -ctk q8_0 -ctv
// q4_0; the format is op, type, ne[4], the op params, then each source as type, ne[4], nb[4]): the
// decode step (1 row) and the prompt batch (512 rows). q is [256, T, 24] nb [4, 24576, 1024], K q8_0
// [256, n_kv, 4] nb [34, 1088, 272], V q4_0 [256, n_kv, 4] nb [18, 576, 144], the mask f16 [n_kv, T].
// A verify of a draft has the decode line's tensors with 4 to 8 rows; an export cannot hold it (it
// reserves the decode and the prompt graph only), so the test sets the row count on the decode line.
//
// Expectations: both lines as exported are refused "rows"; the decode line at 4, 5, 6, 7 and 8 rows
// is ACCEPTED, at the export's 4,096 keys, at 77,297 and at 262,144; each accepted call is then
// bent, one property at a time, into a call the kernel cannot serve, and must be refused with its
// reason. Then the split rule's table and the mask slots.
//
// Red first, five builds, each must exit 1:
//   -DFA_DEC27_GATE_RED_STRIDE   an f16 cache's head-stride rule: refuses the model's lines;
//   -DFA_DEC27_GATE_RED_EXTRA    only the page table is looked at: accepts a call with an indexed
//                                bias or the port's sink;
//   -DFA_DEC27_SPLIT_RED_NO_WAVE the split without the wave bound (512 keys a split at any depth);
//   -DFA_DEC27_GATE_RED_ROW_BLOCK the gate without the row-block rule: an 8-row block of a 9 to 64-row
//                                call accepted;
//   -DFA_DEC27_GATE_RED_NO_MIN_KV the gate without the crossover: a call of 8,191 keys accepted.
//
//   g++ -O2 -I <llama-tree>/ggml/include -I <llama-tree>/ggml/src/ggml-opencl dec27_gate_test.cpp -o dec27_gate_test
//   ./dec27_gate_test fixtures/q38-27b-q8q4-fa.txt
#include "fa_dec27_gate.h"

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
    std::vector<int32_t> params;
    std::vector<src_t> src;
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
        int64_t np = 0, ns = 0;
        if (!(is >> l.op >> l.type >> l.ne[0] >> l.ne[1] >> l.ne[2] >> l.ne[3] >> np)) continue;
        l.params.resize(np);
        for (int64_t i = 0; i < np; ++i) is >> l.params[i];
        if (!(is >> ns)) continue;
        for (int64_t i = 0; i < ns; ++i) {
            src_t x;
            is >> x.type >> x.ne[0] >> x.ne[1] >> x.ne[2] >> x.ne[3] >> x.nb[0] >> x.nb[1] >> x.nb[2] >> x.nb[3];
            l.src.push_back(x);
        }
        out.push_back(l);
    }
    return out;
}

// the op of a line, its tensors on the stack of this struct
struct call_t {
    ggml_tensor q, k, v, mask, op, extra;
    void set(ggml_tensor & t, const src_t & s) {
        memset(&t, 0, sizeof t);
        t.type = (ggml_type) s.type;
        for (int i = 0; i < 4; ++i) { t.ne[i] = s.ne[i]; t.nb[i] = s.nb[i]; }
    }
    call_t(const line_t & l) {
        set(q, l.src[0]); set(k, l.src[1]); set(v, l.src[2]); set(mask, l.src[3]);
        memset(&op, 0, sizeof op); memset(&extra, 0, sizeof extra);
        op.type = (ggml_type) l.type;
        for (int i = 0; i < 4; ++i) op.ne[i] = l.ne[i];
        // the output as ggml_flash_attn_ext creates it: [D, n_head, rows], contiguous
        op.nb[0] = 4; op.nb[1] = 4 * (size_t) op.ne[0]; op.nb[2] = op.nb[1] * (size_t) op.ne[1]; op.nb[3] = op.nb[2] * (size_t) op.ne[2];
        memcpy(op.op_params, l.params.data(), std::min(sizeof op.op_params, l.params.size() * sizeof(int32_t)));
        link();
    }
    call_t(const call_t & o) { *this = o; link(); }
    call_t & operator=(const call_t & o) { q = o.q; k = o.k; v = o.v; mask = o.mask; op = o.op; extra = o.extra; link(); return *this; }
    void link() {
        const bool had_mask = op.src[3] != nullptr || op.src[0] == nullptr;
        for (auto & s : op.src) { if (s != nullptr && s != &extra) s = nullptr; else if (s != nullptr) s = &extra; }
        op.src[0] = &q; op.src[1] = &k; op.src[2] = &v; op.src[3] = had_mask ? &mask : nullptr;
    }
    void rows(int64_t t) {      // a verify of t rows: the row axis of q, the mask and the output
        q.ne[1] = t; mask.ne[1] = t; op.ne[2] = t;
    }
    void keys(int64_t n) {
        k.ne[1] = n; v.ne[1] = n; mask.ne[0] = n; mask.nb[1] = 2 * (size_t) n;
    }
};

static int n_fail = 0;
static void expect(const char * what, const call_t & c, const char * want, size_t align = 4096, int intel = 1) {
    const char * why = "";
    const int acc = fa_dec27_gate_op(&c.op, align, intel, &why);
    const bool ok = want == nullptr ? acc != 0 : (acc == 0 && strcmp(why, want) == 0);
    printf("  %-58s %s%s  %s\n", what, acc ? "ACCEPT" : "refuse ", acc ? "" : why, ok ? "ok" : "FAIL");
    n_fail += !ok;
}
static void expect_eq(const char * what, long got, long want) {
    printf("  %-58s %ld (want %ld)  %s\n", what, got, want, got == want ? "ok" : "FAIL");
    n_fail += got != want;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: dec27_gate_test <export>\n"); return 2; }
    const auto lines = read_lines(argv[1]);
    const line_t * dec = nullptr, * prompt = nullptr;
    for (const auto & l : lines) {
        if (l.src.size() != 4 || l.src[0].ne[0] != 256 || l.src[1].type != GGML_TYPE_Q8_0 || l.src[2].type != GGML_TYPE_Q4_0) continue;
        if (l.src[0].ne[1] == 1) dec = &l; else prompt = &l;
    }
    if (dec == nullptr || prompt == nullptr) { fprintf(stderr, "the export has no decode and prompt FLASH_ATTN_EXT line with a q8_0 K and a q4_0 V\n"); return 2; }
    printf("the export's lines %d (decode) and %d (prompt): q nb [%zu, %zu, %zu], K nb [%zu, %zu, %zu], V nb [%zu, %zu, %zu]\n",
           dec->no, prompt->no, dec->src[0].nb[0], dec->src[0].nb[1], dec->src[0].nb[2],
           dec->src[1].nb[0], dec->src[1].nb[1], dec->src[1].nb[2], dec->src[2].nb[0], dec->src[2].nb[1], dec->src[2].nb[2]);

    expect("the decode line as exported (1 row)", call_t(*dec), "rows");
    expect("the prompt line as exported (512 rows)", call_t(*prompt), "rows");
    for (int64_t n : { (int64_t) 4096, (int64_t) 77297, (int64_t) 262144 }) {
        for (int64_t t = 4; t <= 8; ++t) {
            call_t c(*dec); c.rows(t); c.keys(n);
            char what[96]; snprintf(what, sizeof what, "the decode line at %lld rows, %lld keys", (long long) t, (long long) n);
            expect(what, c, nullptr);
            n_fail += fa_dec27_in_scope(&c.op) == 0;
        }
    }
    call_t base(*dec); base.rows(6); base.keys(77297);
    { call_t c = base; c.rows(3); expect("3 rows", c, "rows"); }
    { call_t c = base; c.rows(9); expect("9 rows", c, "rows"); }
    { call_t c = base; c.v.type = GGML_TYPE_Q8_0; c.v.nb[0] = 34; expect("a q8_0 V", c, "kv-type"); }
    { call_t c = base; c.k.type = GGML_TYPE_F16; expect("an f16 K", c, "kv-type"); }
    { call_t c = base; c.k.type = GGML_TYPE_Q4_0; expect("a q4_0 K", c, "kv-type"); }
    { call_t c = base; c.q.ne[0] = c.k.ne[0] = c.v.ne[0] = 128; expect("head size 128", c, "head"); }
    { call_t c = base; c.q.ne[3] = 2; expect("two streams", c, "streams"); }
    { call_t c = base; c.op.src[4] = &c.extra; expect("a sink", c, "sinks"); }
    { call_t c = base; c.op.src[5] = &c.extra; expect("a page table", c, "extra-sources"); }
    { call_t c = base; c.op.src[6] = &c.extra; expect("a ring", c, "extra-sources"); }
    { call_t c = base; c.op.src[12] = &c.extra; expect("an indexed bias table", c, "extra-sources"); }
    { call_t c = base; c.op.src[18] = &c.extra; expect("the port's per-query sink", c, "extra-sources"); }
    { call_t c = base; c.op.src[19] = &c.extra; expect("the op's last source slot (19)", c, "extra-sources"); }
    { call_t c = base; c.op.op_params[15] = 1; expect("the indexed-bias flag", c, "extra-sources"); }
    { call_t c = base; float x = 8.0f; memcpy((char *) c.op.op_params + 4, &x, 4); expect("ALiBi", c, "alibi"); }
    { call_t c = base; float x = 30.0f; memcpy((char *) c.op.op_params + 8, &x, 4); expect("a logit softcap", c, "softcap"); }
    { call_t c = base; c.op.src[3] = nullptr; expect("no mask", c, "mask"); }
    { call_t c = base; c.mask.type = GGML_TYPE_F32; expect("an f32 mask", c, "mask"); }
    { call_t c = base; c.mask.ne[2] = 24; expect("a mask a head", c, "mask"); }
    { call_t c = base; c.mask.ne[0] = 4096; expect("a mask shorter than the keys", c, "mask"); }
    { call_t c = base; c.v.ne[1] = 4096; expect("V shorter than K", c, "shape"); }
    { call_t c = base; c.q.ne[2] = 25; c.op.ne[1] = 25; expect("25 query heads on 4", c, "shape"); }
    { call_t c = base; c.k.nb[1] = 272; c.k.nb[2] = 272 * 77297; expect("K contiguous (a key row is one kv head)", c, "layout-k"); }
    { call_t c = base; c.k.nb[1] = 1090; expect("a K row pitch off 16 bytes", c, "layout-k"); }
    { call_t c = base; c.k.view_offs = 2; expect("a K view 2 bytes in", c, "layout-k"); }
    { call_t c = base; c.k.nb[2] = 274; expect("a K head stride off the dword grid", c, "layout-k"); }
    { call_t c = base; c.v.nb[1] = 144; c.v.nb[2] = 144 * 77297; expect("V contiguous", c, "layout-v"); }
    { call_t c = base; c.v.view_offs = 2; expect("a V view 2 bytes in", c, "layout-v"); }
    { call_t c = base; c.q.nb[0] = 2; expect("a Q that is not floats", c, "layout-q"); }
    { call_t c = base; c.op.nb[0] = 2; expect("an output that is not floats", c, "dst"); }
    {   // an 8-row block of a 9 to 64-row call, as the dispatch's row-block split hands it on: a verify's shape,
        // accepted by the shape's gate and refused as a call
        call_t c = base; c.rows(8);
        const char * why = "";
        const int shape = fa_dec27_gate_op(&c.op, 4096, 1, &why), call = fa_dec27_gate_call(&c.op, 1, 1, 4096, 1, &why);
        const bool ok = shape != 0 && call == 0 && strcmp(why, "row-block") == 0;
        printf("  %-58s %s%s  %s\n", "an 8-row block of the row-block split", call ? "ACCEPT" : "refuse ", call ? "" : why, ok ? "ok" : "FAIL");
        n_fail += !ok;
        const int plain = fa_dec27_gate_call(&c.op, 0, 0, 4096, 1, &why);
        printf("  %-58s %s  %s\n", "the same 8 rows as the model's own call", plain ? "ACCEPT" : "refuse", plain ? "ok" : "FAIL");
        n_fail += !plain;
    }
    {   // the crossover (FA_DEC27_MIN_KV, a measured point): one key below it the call keeps today's kernels, at it the
        // arm takes the call, forced (the tests) the arm takes the short call too; the shape's gate knows no threshold
        struct { int64_t kv; int force; int want; const char * what; } rows[] = {
            { FA_DEC27_MIN_KV - 1, 0, 0, "8,191 keys: refused below-crossover (today's kernels)" },
            { FA_DEC27_MIN_KV,     0, 1, "8,192 keys: the arm" },
            { FA_DEC27_MIN_KV - 1, 1, 1, "8,191 keys, forced: the arm" },
            { 300,                 1, 1, "300 keys, forced: the arm" },
        };
        for (auto & r : rows) {
            call_t c(*dec); c.rows(6); c.keys(r.kv);
            const char * why = "";
            const int call = fa_dec27_gate_call(&c.op, 0, r.force, 4096, 1, &why), shape = fa_dec27_gate_op(&c.op, 4096, 1, &why);
            const char * w2 = ""; (void) fa_dec27_gate_call(&c.op, 0, r.force, 4096, 1, &w2);
            const bool ok = (call != 0) == (r.want != 0) && shape != 0 && (call != 0 || strcmp(w2, "below-crossover") == 0);
            printf("  %-58s %s%s  %s\n", r.what, call ? "ACCEPT" : "refuse ", call ? "" : w2, ok ? "ok" : "FAIL");
            n_fail += !ok;
        }
        expect_eq("the threshold is the sweep's measured point", FA_DEC27_MIN_KV, 8192);
    }
    expect("a backend alignment of 32", base, "layout-align", 32);
    expect("another vendor's device", base, "device", 4096, 0);

    // the tiles and the mask slots
    expect_eq("tiles, 24 heads on 4, 6 rows", fa_dec27_tiles(6, 6), 5);
    expect_eq("tiles, 4 rows", fa_dec27_tiles(6, 4), 3);
    expect_eq("tiles, 8 rows", fa_dec27_tiles(6, 8), 6);
    for (int t = 4; t <= 8; ++t) expect_eq("mask slots, G 6", fa_dec27_mask_slots(6, t), 2);
    expect_eq("mask slots, G 8", fa_dec27_mask_slots(8, 6), 1);
    expect_eq("mask slots, G 3, 6 rows (a tile of rows 8 .. 15 spans tokens 2 .. 5)", fa_dec27_mask_slots(3, 6), 4);
    expect_eq("mask slots, G 1, 8 rows", fa_dec27_mask_slots(1, 8), 8);
    expect_eq("mask slots, G 1, 4 rows (the rows bound them)", fa_dec27_mask_slots(1, 4), 4);

    // the split: one wave, a multiple of the core count; the floor of 16 blocks below it
    expect_eq("wave, B60 (160 compute units)", fa_dec27_wave(160), 640);
    expect_eq("wave, B70 (256)", fa_dec27_wave(256), 1024);
    expect_eq("B60, 6 rows (20 threads a split), 77,297 keys", fa_dec27_splits(20, 160, 77297, 0, 0), 32);
    expect_eq("B60, 6 rows, 32,768 keys", fa_dec27_splits(20, 160, 32768, 0, 0), 32);
    expect_eq("B60, 6 rows, 262,144 keys", fa_dec27_splits(20, 160, 262144, 0, 0), 32);
    expect_eq("B60, 6 rows, 4,096 keys (the floor: 16 blocks a split)", fa_dec27_splits(20, 160, 4096, 0, 0), 8);
    expect_eq("B60, 6 rows, 300 keys", fa_dec27_splits(20, 160, 300, 0, 0), 1);
    expect_eq("B60, 4 rows (12 threads): 53 fit, 50 is the core multiple", fa_dec27_splits(12, 160, 77297, 0, 0), 50);
    expect_eq("B60, 8 rows (24 threads): 26 fit, 25 is the core multiple", fa_dec27_splits(24, 160, 77297, 0, 0), 25);
    expect_eq("B60, 5 rows (16 threads)", fa_dec27_splits(16, 160, 77297, 0, 0), 40);
    expect_eq("B70, 6 rows: 51 fit, 48 is the core multiple", fa_dec27_splits(20, 256, 77297, 0, 0), 48);
    expect_eq("B60, 6 rows, 4,096 keys, a floor of 4 blocks", fa_dec27_splits(20, 160, 4096, 0, 4), 32);
    expect_eq("forced 7", fa_dec27_splits(20, 160, 77297, 7, 0), 7);
    expect_eq("forced 7 on 100 keys (4 blocks)", fa_dec27_splits(20, 160, 100, 7, 0), 4);
    expect_eq("no compute units known: the floor alone, capped", fa_dec27_splits(20, 0, 77297, 0, 0), 151);
    for (int64_t n : { (int64_t) 6, (int64_t) 1013, (int64_t) 4096, (int64_t) 77297, (int64_t) 262144 }) {
        for (int tps : { 12, 16, 20, 24 }) {
            const int p = fa_dec27_splits(tps, 160, n, 0, 0), sb = fa_dec27_split_blocks(n, p);
            const int64_t blocks = (n + 31) / 32;
            const bool ok = p >= 1 && (p == 1 || p * tps <= 640) && (int64_t) sb * p >= blocks && (int64_t) sb * (p - 1) < blocks + sb;
            if (!ok) { printf("  split %d of %lld keys at %d threads a split: blocks a split %d  FAIL\n", p, (long long) n, tps, sb); n_fail++; }
        }
    }
    printf("dec27 gate test: %s (%d failures)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
