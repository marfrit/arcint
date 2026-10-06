// eval-dump: llama-eval-callback's sibling that writes WHOLE tensors instead of
// printing corners. Every graph tensor whose name is listed in LLAMA_DUMP_NAMES
// (comma-separated, exact ggml names such as "attn_output-0") is written as raw
// f32 to $LLAMA_DUMP_DIR/<name>#<k>.f32 (k = the k-th occurrence of that name
// in the graph, 0-based) and a line "<name>#<k> ne0 ne1 ne2 ne3 type" goes to
// $LLAMA_DUMP_DIR/index.txt. Non-contiguous tensors are skipped with a note.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

struct dump_state {
    std::set<std::string>      names;
    std::map<std::string, int> seen;
    std::string                dir;
    FILE *                     index = nullptr;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const std::string name = t->name;
    const bool want = st->names.count(name) > 0;
    if (ask) {
        return want;
    }
    if (!want) {
        return true;
    }
    const int k = st->seen[name]++;
    const std::string tag = name + "#" + std::to_string(k);
    if (!ggml_is_contiguous(t)) {
        fprintf(st->index, "%s SKIPPED non-contiguous\n", tag.c_str());
        fflush(st->index);
        return true;
    }
    const size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    } else {
        const auto * tt = ggml_get_type_traits(t->type);
        if (!tt || !tt->to_float) {
            fprintf(st->index, "%s SKIPPED type %s\n", tag.c_str(), ggml_type_name(t->type));
            fflush(st->index);
            return true;
        }
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        tt->to_float(raw.data(), buf.data(), n);
    }
    const std::string path = st->dir + "/" + tag + ".f32";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f || fwrite(buf.data(), sizeof(float), n, f) != n) {
        if (f) fclose(f);
        fprintf(st->index, "%s SKIPPED cannot write %s\n", tag.c_str(), path.c_str());
        fflush(st->index);
        return true;
    }
    fclose(f);
    fprintf(st->index, "%s %lld %lld %lld %lld %s\n", tag.c_str(),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            ggml_type_name(t->type));
    fflush(st->index);
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s : no input tokens\n", __func__);
        return false;
    }
    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    dump_state st;
    const char * names = getenv("LLAMA_DUMP_NAMES");
    const char * dir   = getenv("LLAMA_DUMP_DIR");
    if (!names || !dir) {
        fprintf(stderr, "set LLAMA_DUMP_NAMES (comma list of tensor names) and LLAMA_DUMP_DIR\n");
        return 1;
    }
    {
        std::string s = names;
        size_t p = 0;
        while (p <= s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) q = s.size();
            if (q > p) st.names.insert(s.substr(p, q - p));
            p = q + 1;
        }
    }
    st.dir = dir;

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }
    // the index is opened only now: a bad flag or a failed load must not
    // truncate a previous run's index in the same directory
    st.index = fopen((st.dir + "/index.txt").c_str(), "w");
    if (!st.index) {
        fprintf(stderr, "cannot write %s/index.txt\n", dir);
        return 1;
    }
    bool OK = run(ctx, params);
    fclose(st.index);
    if (!OK) {
        return 1;
    }
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
