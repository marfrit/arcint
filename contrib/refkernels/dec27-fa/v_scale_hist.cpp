// v_scale_hist.cpp - 0227, context only (it decides nothing): what a q4_0 V cache's block scales look like on a
// real model. Runs the model on the CPU over a text, takes every attention layer's V projection ("Vcur-<layer>", the
// rows that are written to the V cache) through the eval callback, and for each 32-value block of each (token, kv
// head) takes the block's largest magnitude - 8 x the q4_0 scale |d| ggml's quantizer stores for it. Prints, a layer:
// the blocks, the smallest, the 0.1 % / 1 % / 50 % quantiles and the largest block maximum, and how many blocks lie
// under 1e-1, 1e-2, 1e-3, 1e-4, 1e-5.
//   g++ -O2 -I <llama-tree>/include -I <llama-tree>/ggml/include v_scale_hist.cpp -L <build>/bin -lllama -lggml -lggml-base -o v_scale_hist
//   ./v_scale_hist <model.gguf> <text file> [n_tokens = 1024] [n_threads = 4]
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::map<int, std::vector<float>> g_max;      // layer -> block maxima

static bool on_eval(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (strncmp(t->name, "Vcur-", 5) != 0) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] % 32 != 0 || !ggml_is_contiguous(t)) return true;
    const int il = atoi(t->name + 5);
    const float * x = (const float *) t->data;
    const int64_t n = ggml_nelements(t);
    std::vector<float> & v = g_max[il];
    for (int64_t i = 0; i + 32 <= n; i += 32) {
        float m = 0.0f;
        for (int j = 0; j < 32; ++j) m = std::max(m, std::fabs(x[i + j]));
        v.push_back(m);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: v_scale_hist <model.gguf> <text file> [n_tokens] [n_threads]\n"); return 2; }
    const int n_want = argc > 3 ? atoi(argv[3]) : 1024, n_thr = argc > 4 ? atoi(argv[4]) : 4;
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); const std::string text = ss.str();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n < 0) return 1;
    n = std::min(n, n_want); toks.resize(n);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 64; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_threads = n_thr; cp.n_threads_batch = n_thr;
    cp.cb_eval = on_eval; cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    for (int i = 0; i < n; i += 256) {
        const int nb = std::min(256, n - i);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, nb)) != 0) { fprintf(stderr, "decode failed at %d\n", i); return 1; }
        fprintf(stderr, "decoded %d of %d tokens\n", i + nb, n);
    }
    printf("V block maxima (8 x the q4_0 scale |d|), %d tokens, %zu attention layers\n", n, g_max.size());
    printf("layer   blocks        min     q0.1%%       q1%%     median        max | under 1e-1  1e-2  1e-3  1e-4  1e-5\n");
    std::vector<float> all;
    auto row = [](const char * name, std::vector<float> v) {
        std::sort(v.begin(), v.end());
        auto q = [&](double p) { return v[std::min(v.size() - 1, (size_t) (p * v.size()))]; };
        auto under = [&](float x) { return (long) (std::lower_bound(v.begin(), v.end(), x) - v.begin()); };
        printf("%5s %8zu %10.3g %10.3g %10.3g %10.3g %10.3g | %10ld %5ld %5ld %5ld %5ld\n", name, v.size(), v.front(), q(0.001), q(0.01), q(0.5), v.back(),
               under(1e-1f), under(1e-2f), under(1e-3f), under(1e-4f), under(1e-5f));
    };
    for (auto & kv : g_max) { row(std::to_string(kv.first).c_str(), kv.second); all.insert(all.end(), kv.second.begin(), kv.second.end()); }
    if (!all.empty()) row("all", all);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
