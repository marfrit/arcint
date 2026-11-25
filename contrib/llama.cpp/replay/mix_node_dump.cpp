// dump every graph node's output in execution order (llama's cb_eval), CPU backend
#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct dump_state {
    FILE * tsv = nullptr;
    FILE * bin = nullptr;
    long   idx = 0;
    size_t off = 0;
};

static size_t put_rows(dump_state * d, const ggml_tensor * t) {
    // any f32 tensor, by its strides, as contiguous floats
    size_t n = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
        const char * p = (const char *) t->data + i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1];
        if (t->nb[0] == sizeof(float)) {
            n += fwrite(p, sizeof(float), (size_t) t->ne[0], d->bin);
        } else {
            for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                n += fwrite(p + i0*t->nb[0], sizeof(float), 1, d->bin);
            }
        }
    }
    return n*sizeof(float);
}

static void put(dump_state * d, const char * kind, const ggml_tensor * t, const ggml_tensor * owner) {
    size_t bytes = 0, off = d->off;
    const bool data = t->type == GGML_TYPE_F32 && t->data != nullptr && ggml_nelements(t) > 0 &&
                      (size_t) ggml_nelements(t)*4 <= (size_t) 96*1024*1024;
    if (data) {
        bytes = put_rows(d, t);
        d->off += bytes;
    }
    std::string srcs;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; ++i) {
        srcs += (i ? ";" : "") + std::string(t->src[i]->name);
    }
    fprintf(d->tsv, "%ld\t%s\t%s\t%s\t%s\t%lld,%lld,%lld,%lld\t%zu,%zu,%zu,%zu\t%zu\t%s\t%s\t%zu\t%zu\n",
            d->idx, kind, ggml_op_name(t->op), t->name, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            t->nb[0], t->nb[1], t->nb[2], t->nb[3], t->view_offs,
            t->view_src ? t->view_src->name : "-", owner ? srcs.c_str() : "-", off, bytes);
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    dump_state * d = (dump_state *) ud;
    if (ask) {
        return true;
    }
    put(d, "node", t, t);
    // the mix's inputs as the node saw them: the arm's four sources, and the chain's states view
    if (strcmp(ggml_op_name(t->op), "DEPTH_MIX") == 0) {
        for (int i = 0; i < 4 && t->src[i]; ++i) {
            char k[16]; snprintf(k, sizeof(k), "src%d", i);
            put(d, k, t->src[i], nullptr);
        }
    } else if (t->op == GGML_OP_RMS_NORM && t->src[0] && t->src[0]->view_src && t->ne[2] > 1 && t->ne[0] == 2304) {
        put(d, "src0", t->src[0], nullptr);
    }
    d->idx++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model out-prefix n_tokens\n", argv[0]); return 2; }
    const int n_tok = atoi(argv[3]);
    dump_state d;
    d.tsv = fopen((std::string(argv[2]) + ".tsv").c_str(), "w");
    d.bin = fopen((std::string(argv[2]) + ".bin").c_str(), "wb");
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.cb_eval = cb; cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 1; }
    std::string text;
    for (int i = 0; i < 40; ++i) {
        text += "The depth mix reads the block store and the current state, scores them against learned queries and writes one weighted sum a column. ";
    }
    std::vector<llama_token> toks(2048);
    const int n = llama_tokenize(llama_model_get_vocab(model), text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (n < n_tok) { fprintf(stderr, "only %d tokens\n", n); return 1; }
    toks.resize((size_t) n_tok);
    const int rc = llama_decode(ctx, llama_batch_get_one(toks.data(), n_tok));
    fprintf(stderr, "decode rc=%d nodes=%ld bytes=%zu\n", rc, d.idx, d.off);
    fclose(d.tsv); fclose(d.bin);
    llama_free(ctx); llama_model_free(model);
    return rc;
}
