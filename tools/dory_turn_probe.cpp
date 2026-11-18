// dory_turn_probe: the CPU probe behind the 0118 figures (contrib/llama.cpp/patches/0118-dory-loops-per-context.patch,
// README section of the same name). It measures whether one llama.cpp context whose Dory loop count is chosen per
// turn (llama_set_dory_loops, tables built for 4 loops through LLAMA_DORY_MAX_LOOPS) gives the logits of a model loaded
// at that count, and what continuing a cache written at another count costs.
//
// NOT part of the arcint build: CMake does not compile it and the package recipe does not run it. Build it by hand
// against a llama.cpp tree with the series applied (static CPU-only build of the `llama` target, GGML_OPENCL=OFF,
// BUILD_SHARED_LIBS=OFF), then link the static libraries of that build:
//
//   g++ -O2 -std=c++17 -I <llama-tree>/include -I <llama-tree>/ggml/include tools/dory_turn_probe.cpp -o dory_turn_probe \
//       -Wl,--start-group $(find <build-dir> -name 'lib*.a') -Wl,--end-group -fopenmp -lpthread -ldl -lm
//
// Run (the fixtures are tools/dory_turn_cases.txt: one shared history and six made-up conversations):
//
//   ./dory_turn_probe <dory-model.gguf> <threads> tools/dory_turn_cases.txt 4096 green [chain-max-tokens]   > probe.out
//   LLAMA_DORY_RED_REUSE=1 ./dory_turn_probe <dory-model.gguf> <threads> tools/dory_turn_cases.txt 4096 red [chain-max-tokens] > probe-red.out
//
// `green` is the real run (exit 0 when it finishes; 2 = a load, context, tokenize or decode failed or the arguments are
// short, 3 = a state restore failed, 4 = an API check failed, 6 = a probe failed). `red` stops after the
// PROBE section and is meant to be run with the library's red switch set: it exits 0 when at least one probe failed (the
// switch bit) and 5 when none did (the red case proved nothing). Aggregate a green output with
// tools/dory_turn_probe_agg.py. A run took up to about 90 minutes on 8 threads.
//
// Output lines (one per measurement; KL in nats against the reference named on the line):
//   MODE, LOAD, TOK, API, PREFILL   set-up: the loads (executed layers, tables' maximum), token counts, the API checks of
//                                   llama_set_dory_loops, prefill rates of the baseline and of the per-turn context.
//   SIZE                            llama.cpp's own buffer-size lines of each load (compute buffer, KV, tables).
//   PROBE x->y                      one token at count x then at y in the same context against a load at y; OK = bit-equal.
//                                   PROBE-SUMMARY n of 9 failed.
//   EXACT, FLOOR                    greedy answer of a load at each count; FLOOR = the same count, batched teacher-forcing
//                                   against step-by-step decode (the noise floor of the comparison).
//   IDENT                           history and final turn all at count y by the per-turn switch against a load at y:
//                                   same tokens, rows compared, bits_equal, KL.
//   CHAIN, CHAINTEXT                one conversation with the count changing every turn against a load at each turn's
//                                   count with the whole conversation prefilled again.
//   PAIR x->y                       history cached at x, final turn at y in one context ("mixed") against a load at y,
//                                   next to the pure-x arm; MIXED is its greedy text; TVT compares it with a state moved
//                                   between two loads.
// What it compares, section by section:
// Dory's loop count as a per-turn choice in ONE context, tables built for 4 loops (LLAMA_DORY_MAX_LOOPS=4),
// against the baseline: a model loaded at that count the way the server does it today (metadata override, the
// variable unset, so the tables are the stock ones). CPU only.
//   PROBE : one token at count X, then the same token at count Y in the same context, against the baseline at Y.
//           The deliberately wrong build (LLAMA_DORY_RED_REUSE=1: the graph of the earlier count is reused) must fail here.
//   IDENT : history + final turn, all at count Y by the turn switch, against the baseline at Y (expected: the same bits)
//   CHAIN : one conversation, the count changing every turn, the cache never rebuilt; each turn against the baseline
//           at that turn's count with the whole conversation so far prefilled again
//   PAIR  : history at X, the final turn at Y in the same context (no state transfer), against the baseline at Y;
//           TVT compares it with the earlier experiment's route (state moved between two loads)
#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using tok = llama_token;
typedef std::vector<std::vector<float>> rows;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Case { std::string name; int maxtok; std::string text; std::vector<tok> f; };
struct Mod  { int loops; llama_model * m; llama_context * c; std::vector<uint8_t> st; };
static const llama_vocab * V; static int NV;
static const char * TAG = "";
static void logcb(enum ggml_log_level l, const char * t, void *) {
    if (l >= GGML_LOG_LEVEL_ERROR) { fputs(t, stderr); return; }
    static const char * k[] = { "compute buffer", "graph nodes", "KV buffer", "Dory tables", "LLAMA_DORY", "layers," };
    for (auto s : k) if (strstr(t, s)) { printf("SIZE %s | %s", TAG, t); if (!strchr(t, '\n')) printf("\n"); return; }
}
static std::vector<tok> tokz(const std::string & s) {
    std::vector<tok> t(s.size() + 16);
    int n = llama_tokenize(V, s.data(), (int) s.size(), t.data(), (int) t.size(), false, true);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); exit(2); }
    t.resize(n); return t;
}
static std::string piece(tok t) { char b[256]; int n = llama_token_to_piece(V, t, b, sizeof b, 0, true); return n > 0 ? std::string(b, n) : ""; }
static std::string text(const std::vector<tok> & g) { std::string s; for (tok t : g) s += piece(t); return s; }
static std::string esc(const std::string & s) {
    std::string o; for (unsigned char ch : s) { if (ch == '\n') o += "\\n"; else if (ch == '\t') o += "\\t"; else if (ch == '"') o += "\\\""; else if (ch == '\\') o += "\\\\"; else if (ch < 32) o += ' '; else o += (char) ch; } return o;
}
static bool dec(llama_context * c, const std::vector<tok> & t, int pos0, int lf, rows * out, int nb) {
    for (size_t o = 0; o < t.size(); o += nb) {
        int n = (int) std::min<size_t>(nb, t.size() - o);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = t[o + i]; b.pos[i] = pos0 + (int) o + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = ((int) (o + i) >= lf) || (o + i + 1 == t.size());
        }
        b.n_tokens = n;
        int r = llama_decode(c, b);
        if (r != 0) { fprintf(stderr, "llama_decode rc=%d\n", r); llama_batch_free(b); return false; }
        if (out) for (int i = 0; i < n; ++i) if ((int) (o + i) >= lf) { const float * l = llama_get_logits_ith(c, i); out->emplace_back(l, l + NV); }
        llama_batch_free(b);
    }
    return true;
}
static int amax(const std::vector<float> & l) { int a = 0; for (int i = 1; i < NV; ++i) if (l[i] > l[a]) a = i; return a; }
static void lsm(const std::vector<float> & l, std::vector<double> & o) {
    double m = l[0]; for (float x : l) m = std::max<double>(m, x);
    double s = 0; for (float x : l) s += std::exp((double) x - m);
    double ls = m + std::log(s); o.resize(NV); for (int i = 0; i < NV; ++i) o[i] = (double) l[i] - ls;
}
struct Cmp { double mean = 0, mx = 0, mad = 0; int agree = 0, n = 0, bits = 0; };
static Cmp cmp(const rows & ref, const rows & arm) {
    Cmp r; std::vector<double> p, q; r.n = (int) std::min(ref.size(), arm.size());
    for (int j = 0; j < r.n; ++j) {
        lsm(ref[j], p); lsm(arm[j], q); double k = 0;
        for (int i = 0; i < NV; ++i) { double e = std::exp(p[i]); if (e > 0) k += e * (p[i] - q[i]); r.mad = std::max(r.mad, (double) std::fabs(ref[j][i] - arm[j][i])); }
        r.mean += k; r.mx = std::max(r.mx, k);
        if (amax(ref[j]) == amax(arm[j])) r.agree++;
        if (!memcmp(ref[j].data(), arm[j].data(), NV * sizeof(float))) r.bits++;
    }
    if (r.n) r.mean /= r.n; return r;
}
static bool restore(llama_context * c, const std::vector<uint8_t> & st) {
    llama_memory_clear(llama_get_memory(c), true);
    return llama_state_seq_set_data(c, st.data(), st.size(), 0) != 0;
}
static void save(llama_context * c, std::vector<uint8_t> & st) { st.resize(llama_state_seq_get_size(c, 0)); llama_state_seq_get_data(c, st.data(), st.size(), 0); }
static void setl(llama_context * c, int k) {
    if (!llama_set_dory_loops(c, k) || (k && (int) llama_get_dory_loops(c) != k)) { printf("SETLOOPS-FAILED %d\n", k); exit(4); }
}
// greedy from the context's current state: decode f at pos0, then up to maxtok tokens; a logits row per step
static bool greedy(llama_context * c, const std::vector<tok> & f, int pos0, int maxtok, std::vector<tok> & g, rows * lg, int nb) {
    rows l;
    if (!dec(c, f, pos0, (int) f.size() - 1, &l, nb)) return false;
    int pos = pos0 + (int) f.size();
    for (int i = 0; i < maxtok; ++i) {
        tok t = amax(l.back()); if (lg) lg->push_back(l.back());
        if (llama_vocab_is_eog(V, t)) break;
        g.push_back(t); l.clear();
        if (!dec(c, { t }, pos++, 0, &l, nb)) return false;
    }
    return true;
}
// the same decode pattern as greedy, the tokens taken from g: n rows
static bool forced(llama_context * c, const std::vector<tok> & f, const std::vector<tok> & g, int pos0, size_t n, rows & out, int nb) {
    rows l;
    if (!dec(c, f, pos0, (int) f.size() - 1, &l, nb)) return false;
    int pos = pos0 + (int) f.size(); out.push_back(l.back());
    for (size_t j = 0; out.size() < n && j < g.size(); ++j) { l.clear(); if (!dec(c, { g[j] }, pos++, 0, &l, nb)) return false; out.push_back(l.back()); }
    return true;
}
static std::vector<tok> tf_tokens(const Case & c, const std::vector<tok> & g, const rows & ref) {
    std::vector<tok> fg = c.f; if (!g.empty()) fg.insert(fg.end(), g.begin(), g.end() - (ref.size() > g.size() ? 0 : 1)); return fg;
}
static std::string answer(const std::string & t) { size_t i = t.find("</think>"); return i == std::string::npos ? "[no </think> in limit] ..." + t.substr(t.size() > 90 ? t.size() - 90 : 0) : t.substr(i + 8, 260); }

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: dory_turn_probe model.gguf threads cases.txt n_ctx green|red [chain maxtok]\n"); return 2; }
    const char * path = argv[1]; int T = atoi(argv[2]); int nctx = atoi(argv[4]); const bool red = !strcmp(argv[5], "red");
    const int chain_max = argc > 6 ? atoi(argv[6]) : 400; const int nb = 2048; const int LC[3] = { 1, 2, 4 };
    llama_log_set(logcb, nullptr); ggml_backend_load_all(); llama_backend_init();
    printf("MODE %s RED_REUSE=%s\n", red ? "red" : "green", getenv("LLAMA_DORY_RED_REUSE") ? getenv("LLAMA_DORY_RED_REUSE") : "(unset)");
    auto load = [&](int loops, const char * tag) {
        Mod m { loops, nullptr, nullptr, {} }; TAG = tag;
        llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        llama_model_kv_override ov[3]; memset(ov, 0, sizeof ov);
        ov[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT; strcpy(ov[0].key, "dory.num_loops"); ov[0].val_i64 = loops;
        ov[1].tag = LLAMA_KV_OVERRIDE_TYPE_STR; strcpy(ov[1].key, "dory.recurrent_kv_cache_mode"); strcpy(ov[1].val_str, "last_loop");
        mp.kv_overrides = ov;
        double t0 = now(); m.m = llama_model_load_from_file(path, mp);
        if (!m.m) { fprintf(stderr, "load failed (%s)\n", tag); exit(2); }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = nctx; cp.n_batch = nb; cp.n_ubatch = 512; cp.n_threads = T; cp.n_threads_batch = T; cp.no_perf = true;
        m.c = llama_init_from_model(m.m, cp);
        if (!m.c) { fprintf(stderr, "context failed (%s)\n", tag); exit(2); }
        printf("LOAD %s loops=%d executed_layers=%d loops_max=%u ctx_loops=%u state_empty=%zu %.1fs\n", tag, loops, llama_model_n_layer(m.m),
               llama_model_dory_loops_max(m.m), llama_get_dory_loops(m.c), llama_state_seq_get_size(m.c, 0), now() - t0); fflush(stdout);
        return m;
    };
    // the baseline: loaded at each count, the variable unset
    unsetenv("LLAMA_DORY_MAX_LOOPS");
    std::vector<Mod> S; for (int k : LC) { char tag[32]; snprintf(tag, sizeof tag, "base%d", k); S.push_back(load(k, strdup(tag))); }
    // the turn model: loaded once at the unit's 2 loops, tables for 4
    setenv("LLAMA_DORY_MAX_LOOPS", "4", 1);
    Mod Tm = load(2, "turn"); unsetenv("LLAMA_DORY_MAX_LOOPS"); TAG = "run";
    // the setter's bounds (one call a statement: the order of evaluation inside one printf is not defined)
    {
        const int t5 = llama_set_dory_loops(Tm.c, 5); const int t4 = llama_set_dory_loops(Tm.c, 4); const unsigned g4 = llama_get_dory_loops(Tm.c);
        const int t1 = llama_set_dory_loops(Tm.c, 1); const unsigned g1 = llama_get_dory_loops(Tm.c);
        const int t0 = llama_set_dory_loops(Tm.c, 0); const unsigned g0 = llama_get_dory_loops(Tm.c);
        const int b4 = llama_set_dory_loops(S[1].c, 4); const int b1 = llama_set_dory_loops(S[1].c, 1); const unsigned h1 = llama_get_dory_loops(S[1].c);
        const int b0 = llama_set_dory_loops(S[1].c, 0); const unsigned h0 = llama_get_dory_loops(S[1].c);
        printf("API turn: set5=%d set4=%d (now %u) set1=%d (now %u) set0=%d (now %u) | base2 (tables for 2): set4=%d set1=%d (now %u) set0=%d (now %u)\n",
               t5, t4, g4, t1, g1, t0, g0, b4, b1, h1, b0, h0);
        if (t5 || !t4 || g4 != 4 || !t1 || g1 != 1 || !t0 || g0 != 2 || b4 || !b0 || h0 != 2) { printf("API-FAILED\n"); return 4; }
    }
    V = llama_model_get_vocab(S[0].m); NV = llama_vocab_n_tokens(V);
    std::string H; std::vector<Case> C; { std::ifstream in(argv[3]); std::string ln; std::string * cur = nullptr;
        while (std::getline(in, ln)) {
            if (ln.rfind("@@@H", 0) == 0) { cur = &H; continue; }
            if (ln.rfind("@@@F", 0) == 0) { std::stringstream ss(ln.substr(4)); Case c; ss >> c.name >> c.maxtok; C.push_back(c); cur = &C.back().text; continue; }
            if (cur) { if (!cur->empty()) *cur += "\n"; *cur += ln; } } }
    std::vector<tok> h = tokz(H + "\n"); for (auto & c : C) c.f = tokz(c.text + "\n");
    const std::vector<tok> tail = tokz("<|im_end|>\n");
    printf("TOK vocab=%d history=%zu tail=%zu\n", NV, h.size(), tail.size());
    const int nh = (int) h.size();

    // histories: the baseline at each count, the turn context at each count (its own states)
    std::vector<std::vector<uint8_t>> ST(3);
    for (int y = 0; y < 3; ++y) {
        if (red && LC[y] != 2) continue;
        double t0 = now(); if (!dec(S[y].c, h, 0, nh, nullptr, nb)) return 2; double tb = now() - t0; save(S[y].c, S[y].st);
        setl(Tm.c, LC[y]); llama_memory_clear(llama_get_memory(Tm.c), true);
        t0 = now(); if (!dec(Tm.c, h, 0, nh, nullptr, nb)) return 2; double tt = now() - t0; save(Tm.c, ST[y]);
        printf("PREFILL loops=%d %d tok: baseline %.1f t/s, turn switch %.1f t/s | state bytes %zu / %zu\n", LC[y], nh, nh / tb, nh / tt, S[y].st.size(), ST[y].size()); fflush(stdout);
    }
    // PROBE: from the 2-loop history, one token at X and then at Y in the turn context; the baseline at Y from the 2-loop state
    {
        const std::vector<tok> one = { C[0].f[0] }; int bad = 0;
        for (int x = 0; x < 3; ++x) for (int y = 0; y < 3; ++y) {
            rows a, b, r;
            if (!restore(Tm.c, ST[1])) return 3; setl(Tm.c, LC[x]); if (!dec(Tm.c, one, nh, 0, &a, nb)) return 2;
            if (!restore(Tm.c, ST[1])) return 3; setl(Tm.c, LC[y]); if (!dec(Tm.c, one, nh, 0, &b, nb)) return 2;
            if (!restore(S[y].c, S[1].st)) return 3; if (!dec(S[y].c, one, nh, 0, &r, nb)) return 2;
            Cmp kb = cmp(r, b), ka = cmp(a, b);
            const bool ok = kb.bits == 1; bad += !ok;
            printf("PROBE %d->%d turn@%d vs baseline@%d: KL=%.3g maxabs=%.3g bits_equal=%d | turn@%d vs turn@%d: KL=%.3g bits_equal=%d | %s\n",
                   LC[x], LC[y], LC[y], LC[y], kb.mean, kb.mad, kb.bits, LC[x], LC[y], ka.mean, ka.bits, ok ? "OK" : "FAIL");
        }
        printf("PROBE-SUMMARY %d of 9 failed\n", bad); fflush(stdout);
        if (red) { printf("DTURN-RED-DONE\n"); return bad ? 0 : 5; }
        if (bad) { printf("DTURN-ABORT probes failed\n"); return 6; }
    }
    // IDENT: the final turn after a history at the same count; the baseline's greedy run is the reference (EXACT),
    // FLOOR as in the earlier experiment (batched against stepwise at one count)
    std::vector<std::vector<std::vector<tok>>> G(3, std::vector<std::vector<tok>>(C.size()));
    std::vector<std::vector<rows>> L(3, std::vector<rows>(C.size()));
    for (int y = 0; y < 3; ++y) for (size_t c = 0; c < C.size(); ++c) {
        if (!restore(S[y].c, S[y].st)) return 3;
        double t1 = now(); if (!greedy(S[y].c, C[c].f, nh, C[c].maxtok, G[y][c], &L[y][c], nb)) return 2; double tg = now() - t1;
        printf("EXACT loops=%d case=%s n=%zu %.1f t/s text=\"%s\"\n", LC[y], C[c].name.c_str(), G[y][c].size(), (G[y][c].size() + 1) / tg, esc(text(G[y][c])).c_str());
        std::vector<tok> fg = tf_tokens(C[c], G[y][c], L[y][c]);
        if (!restore(S[y].c, S[y].st)) return 3;
        rows lb; if (!dec(S[y].c, fg, nh, (int) C[c].f.size() - 1, &lb, nb)) return 2;
        Cmp k = cmp(L[y][c], lb); printf("FLOOR loops=%d case=%s meanKL=%.3g maxKL=%.3g agree=%d/%d\n", LC[y], C[c].name.c_str(), k.mean, k.mx, k.agree, k.n);
        if (!restore(Tm.c, ST[y])) return 3; setl(Tm.c, LC[y]);
        std::vector<tok> gt; rows lt; t1 = now(); if (!greedy(Tm.c, C[c].f, nh, C[c].maxtok, gt, &lt, nb)) return 2; double tt = now() - t1;
        Cmp ki = cmp(L[y][c], lt);
        printf("IDENT loops=%d case=%s same_tokens=%d rows=%d bits_equal=%d meanKL=%.3g maxKL=%.3g maxabs=%.3g agree=%d/%d | %.1f t/s (baseline %.1f)\n",
               LC[y], C[c].name.c_str(), (int) (gt == G[y][c]), ki.n, ki.bits, ki.mean, ki.mx, ki.mad, ki.agree, ki.n, (gt.size() + 1) / tt, (G[y][c].size() + 1) / tg);
        fflush(stdout);
    }
    // CHAIN: one context, the count changes every turn, nothing is rebuilt
    {
        struct Turn { const char * name; int loops; };
        const Turn plan[] = { { "needle", 1 }, { "fact", 2 }, { "weather", 4 }, { "code", 1 }, { "german", 4 }, { "tool", 2 } };
        std::vector<tok> P = h; setl(Tm.c, 2); if (!restore(Tm.c, ST[1])) return 3;
        int prev = 2;
        for (const Turn & t : plan) {
            const Case * cs = nullptr; for (auto & c : C) if (c.name == t.name) cs = &c;
            if (!cs) { printf("CHAIN no case %s\n", t.name); continue; }
            const int y = t.loops == 1 ? 0 : t.loops == 2 ? 1 : 2; const int p0 = (int) P.size();
            setl(Tm.c, t.loops);
            std::vector<tok> g; rows lg; double t0 = now(); if (!greedy(Tm.c, cs->f, p0, chain_max, g, &lg, nb)) return 2; double tt = now() - t0;
            // the baseline at this count: the whole conversation prefilled again, then the same decode pattern on the chain's tokens
            llama_memory_clear(llama_get_memory(S[y].c), true);
            t0 = now(); if (!dec(S[y].c, P, 0, p0, nullptr, nb)) return 2; double tp = now() - t0;
            std::vector<uint8_t> sb; save(S[y].c, sb);
            rows lb; if (!forced(S[y].c, cs->f, g, p0, lg.size(), lb, nb)) return 2;
            Cmp k = cmp(lb, lg);
            if (!restore(S[y].c, sb)) return 3;
            std::vector<tok> gb; if (!greedy(S[y].c, cs->f, p0, chain_max, gb, nullptr, nb)) return 2;
            printf("CHAIN turn=%s loops=%d (before: %d) prefix=%d tok | n=%d meanKL=%.4f maxKL=%.4f agree=%d/%d same_text=%d | turn %.1fs, baseline re-prefill alone %.1fs\n",
                   t.name, t.loops, prev, p0, k.n, k.mean, k.mx, k.agree, k.n, (int) (gb == g), tt, tp);
            printf("CHAINTEXT %s switch  : %s\n", t.name, esc(answer(text(g))).c_str());
            printf("CHAINTEXT %s baseline: %s\n", t.name, esc(answer(text(gb))).c_str());
            fflush(stdout);
            if (!dec(Tm.c, tail, p0 + (int) cs->f.size() + (int) g.size(), (int) tail.size(), nullptr, nb)) return 2;
            P.insert(P.end(), cs->f.begin(), cs->f.end()); P.insert(P.end(), g.begin(), g.end()); P.insert(P.end(), tail.begin(), tail.end());
            prev = t.loops;
        }
        printf("CHAIN-DONE final prefix %zu tok\n", P.size()); fflush(stdout);
    }
    // PAIR: history at X, the final turn at Y, one context
    for (int x = 0; x < 3; ++x) for (int y = 0; y < 3; ++y) { if (x == y) continue;
        for (size_t c = 0; c < C.size(); ++c) {
            const auto & g = G[y][c]; const auto & ref = L[y][c]; std::vector<tok> fg = tf_tokens(C[c], g, ref); const int lf = (int) C[c].f.size() - 1;
            if (!restore(Tm.c, ST[x])) return 3; setl(Tm.c, LC[y]);
            rows lm; if (!dec(Tm.c, fg, nh, lf, &lm, nb)) return 2; Cmp km = cmp(ref, lm);
            if (!restore(S[y].c, S[x].st)) { printf("TRANSFER-REFUSED\n"); return 3; }
            rows lt; if (!dec(S[y].c, fg, nh, lf, &lt, nb)) return 2; Cmp kt = cmp(lt, lm);
            if (!restore(S[x].c, S[x].st)) return 3;
            rows lp; if (!dec(S[x].c, fg, nh, lf, &lp, nb)) return 2; Cmp kp = cmp(ref, lp);
            if (!restore(Tm.c, ST[x])) return 3; setl(Tm.c, LC[y]);
            std::vector<tok> gm; if (!greedy(Tm.c, C[c].f, nh, C[c].maxtok, gm, nullptr, nb)) return 2;
            printf("PAIR %d->%d case=%s n=%d mixed: meanKL=%.4f maxKL=%.4f agree=%d/%d | pure%d: meanKL=%.4f maxKL=%.4f agree=%d/%d | same_text=%d\n",
                   LC[x], LC[y], C[c].name.c_str(), km.n, km.mean, km.mx, km.agree, km.n, LC[x], kp.mean, kp.mx, kp.agree, kp.n, (int) (gm == g));
            printf("TVT %d->%d case=%s turn switch vs state transfer: rows=%d bits_equal=%d meanKL=%.3g maxabs=%.3g\n", LC[x], LC[y], C[c].name.c_str(), kt.n, kt.bits, kt.mean, kt.mad);
            printf("MIXED %d->%d case=%s n=%zu text=\"%s\"\n", LC[x], LC[y], C[c].name.c_str(), gm.size(), esc(text(gm)).c_str());
            fflush(stdout);
        } }
    printf("DTURN-DONE\n");
    return 0;
}
