// port_gate_test.cpp — the port arm's eligibility rules, on the real graph export.
//
// Reads the test-export-graph-ops files (tests/test-export-graph-ops.cpp's format: an
// int per field, the node name last), takes each FLASH_ATTN_EXT line's q/k/v/mask from
// the line itself - ne AND nb, so the permuted views are the model's, not one the test
// builds -, takes the bias table from the MUL_MAT line that produces [n_bins, n_head,
// rows], synthesizes only what the patched graph ADDS (indices I32, kpos/qpos I32,
// cflag/rflag I8, the per-query sink F32 [n_head, rows], the dst contiguous as
// ggml_flash_attn_ext creates it), and asks fa_port_gate_why.
//
// Expectations, per line: a FLASH_ATTN_EXT line with dst ne[2] <= 8 (the decode step)
// is refused "decode"; one with dst ne[2] >= 9 (the prefill) is ACCEPTED. The build
// order is red first: with a contiguity rule the prefill lines are refused "layout".
//
//   g++ -O2 -I <llama-tree>/ggml/include port_gate_test.cpp -o port_gate_test
//   ./port_gate_test <export.txt> [<export2.txt> ...]
#include "fa_port_gate.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

struct line_t {
    int op, type;
    int64_t ne[4];
    std::vector<int32_t> params;
    struct src_t { int type; int64_t ne[4]; size_t nb[4]; };
    std::vector<src_t> sources;
    std::string name;
};

static std::vector<line_t> read_lines(const char * path) {
    std::vector<line_t> out;
    std::ifstream f(path);
    std::string s;
    while (std::getline(f, s)) {
        std::istringstream is(s);
        line_t l; int64_t np;
        if (!(is >> l.op >> l.type >> l.ne[0] >> l.ne[1] >> l.ne[2] >> l.ne[3] >> np)) continue;
        l.params.resize(np);
        for (auto & p : l.params) is >> p;
        int64_t ns;
        if (!(is >> ns)) continue;
        for (int64_t i = 0; i < ns; ++i) {
            line_t::src_t src;
            is >> src.type >> src.ne[0] >> src.ne[1] >> src.ne[2] >> src.ne[3]
               >> src.nb[0] >> src.nb[1] >> src.nb[2] >> src.nb[3];
            l.sources.push_back(src);
        }
        is >> l.name;
        out.push_back(l);
    }
    return out;
}

// the gate takes ggml_tensors: only type/ne/nb are read
static struct ggml_tensor tensor_of(int type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                    size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    struct ggml_tensor t;
    memset(&t, 0, sizeof t);
    t.type = (enum ggml_type) type;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = nb0; t.nb[1] = nb1; t.nb[2] = nb2; t.nb[3] = nb3;
    return t;
}
static struct ggml_tensor contiguous_of(int type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t esz) {
    struct ggml_tensor t = tensor_of(type, ne0, ne1, ne2, ne3, esz, esz*ne0, esz*ne0*ne1, esz*ne0*ne1*ne2);
    return t;
}

int main(int argc, char ** argv) {
    int fails = 0;
    for (int a = 1; a < argc; ++a) {
        const std::vector<line_t> lines = read_lines(argv[a]);
        // the bias table: the MUL_MAT whose dst is [n_bins, n_head, rows] (n_bins >= 1024)
        const line_t * table_line = nullptr;
        for (const auto & l : lines) {
            if (l.op != 6 || l.ne[0] < 1024) continue;
            const int64_t rows = l.ne[2];
            for (const auto & lf : lines) {
                if (lf.op != 75) continue;
                if (lf.ne[2] == rows) { table_line = &l; break; }
            }
            if (table_line) break;
        }
        if (!table_line) { printf("%s: FAIL no bias-table MUL_MAT line\n", argv[a]); ++fails; continue; }
        const int64_t n_bins = table_line->ne[0], n_head_kv = table_line->sources.empty() ? 8 : table_line->sources[0].ne[1];
        (void) n_head_kv;

        int n_checked = 0, n_prefill = 0, n_decode = 0;
        for (const auto & l : lines) {
            if (l.op != 75) continue;
            const line_t::src_t * sq = nullptr, * sk = nullptr, * sv = nullptr, * sm = nullptr;
            for (const auto & s : l.sources) {
                if (s.type == 0 && s.ne[0] == 128 && sk == nullptr && s.ne[1] == l.ne[2]) { sq = &s; continue; }
                if (s.type == 1 && s.ne[0] == 128) {
                    if (sk == nullptr) sk = &s; else if (sv == nullptr) sv = &s;
                    continue;
                }
                if (s.type == 1 && s.ne[2] == 1 && sm == nullptr) sm = &s;
                if (s.type == 0 && s.ne[0] > 1 && s.ne[1] == 1) sq = sq;   // per-head sink, not the port's
            }
            if (!sq || !sk || !sv || !sm) {
                printf("%s line %s: missing a source\n", argv[a], l.name.c_str()); ++fails; continue;
            }
            // the per-query sink [n_head, rows] and the positions/flags the patched graph adds
            struct ggml_tensor sinks_q = contiguous_of(0, l.ne[1], l.ne[2], 1, 1, 4);
            struct ggml_tensor kpos = contiguous_of(2, sk->ne[1], 1, 1, 1, 4);
            struct ggml_tensor qpos = contiguous_of(2, l.ne[2], 1, 1, 1, 4);
            struct ggml_tensor cflag = contiguous_of(3, (sk->ne[1] + 15)/16, 1, 1, 1, 1);
            struct ggml_tensor rflag = contiguous_of(3, (l.ne[2] + 7)/8, 1, 1, 1, 1);
            struct ggml_tensor indices = contiguous_of(2, sk->ne[1], l.ne[2], 1, 1, 4);
            struct ggml_tensor table = tensor_of(0, table_line->ne[0], table_line->ne[1], table_line->ne[2], 1,
                                                table_line->sources.empty() ? 4 : table_line->sources[0].nb[0],
                                                table_line->sources.empty() ? 4100 : table_line->sources[0].nb[1],
                                                table_line->sources.empty() ? 98400 : table_line->sources[0].nb[2], 0);
            struct ggml_tensor dst = contiguous_of(0, l.ne[0], l.ne[1], l.ne[2], 1, 4);
            struct ggml_tensor q = tensor_of(sq->type, sq->ne[0], sq->ne[1], sq->ne[2], sq->ne[3], sq->nb[0], sq->nb[1], sq->nb[2], sq->nb[3]);
            struct ggml_tensor k = tensor_of(sk->type, sk->ne[0], sk->ne[1], sk->ne[2], sk->ne[3], sk->nb[0], sk->nb[1], sk->nb[2], sk->nb[3]);
            struct ggml_tensor v = tensor_of(sv->type, sv->ne[0], sv->ne[1], sv->ne[2], sv->ne[3], sv->nb[0], sv->nb[1], sv->nb[2], sv->nb[3]);
            struct ggml_tensor mask = tensor_of(sm->type, sm->ne[0], sm->ne[1], sm->ne[2], sm->ne[3], sm->nb[0], sm->nb[1], sm->nb[2], sm->nb[3]);

            const char * why = "";
            const int ok = fa_port_gate_why(&q, &k, &v, &mask, &dst, &table, &indices, &sinks_q,
                                            &kpos, &qpos, &cflag, &rflag,
                                            2, 1, 0, 1, 0.0f, 0.0f, 0, &why);
            const bool decode = l.ne[2] <= 8;
            ++n_checked;
            if (decode) {
                ++n_decode;
                const bool good = !ok && strcmp(why, "decode") == 0;
                if (!good) { printf("%s line %s (decode, n_q %lld): expected refuse decode, got %s%s\n",
                                    argv[a], l.name.c_str(), (long long) l.ne[2], ok ? "ACCEPT" : "", why); ++fails; }
            } else {
                ++n_prefill;
                if (!ok) { printf("%s line %s (prefill, n_q %lld): refused %s\n", argv[a], l.name.c_str(),
                                  (long long) l.ne[2], why); ++fails; }
            }
            (void) n_bins;
        }
        printf("%s: %d FLASH_ATTN_EXT line(s): %d prefill, %d decode, %d failure(s)\n",
               argv[a], n_checked, n_prefill, n_decode, fails);
    }
    return fails ? 1 : 0;
}
