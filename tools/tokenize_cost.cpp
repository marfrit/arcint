// audit 15: what a request's tokenisations cost, on the host or the card host.
//
// It loads a vocab-only GGUF (llama_model_load_from_file with vocab_only) and
// times llama_tokenize over a text that tokenises to a long agent prompt:
//   - the old encode: a null-output sizing pass, then an exact-sized pass;
//   - the one-pass encode: an upper-bound buffer (n bytes + 8), retried only if
//     the vocabulary reports more.
// The text is read from a file, or built deterministically from `--bytes`
// bytes of prose-like lines. Prints tokens, bytes, the two passes' ms and the
// ratio. Reproducible: same GGUF, same bytes, same toolchain.
//
// Build (ARCINT_LLAMA=ON), run on the host with the pinned vocab:
//   ./build/tokenize-cost models/ggml-vocab-qwen35.gguf --bytes 160000
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string make_text(size_t bytes) {
    static const char * const lines[] = {
        "the agent reads a file, runs the tests, and writes the result back to the caller.\n",
        "a prompt of forty thousand tokens is the long agent turn this measures.\n",
        "the tokenizer walks the text byte by byte and emits one token per piece.\n",
    };
    std::string s;
    size_t i = 0;
    while (s.size() < bytes) s += lines[i++ % 3];
    s.resize(bytes);
    return s;
}

// two llama_tokenize calls: size, then fill (the old encode)
std::vector<int> old_encode(const llama_vocab * v, const std::string & text) {
    const int need = -llama_tokenize(v, text.data(), static_cast<int32_t>(text.size()), nullptr, 0,
                                     /*add_special=*/false, /*parse_special=*/true);
    std::vector<int> out(static_cast<size_t>(need > 0 ? need : 0));
    if (need > 0) {
        llama_tokenize(v, text.data(), static_cast<int32_t>(text.size()),
                       reinterpret_cast<llama_token *>(out.data()), need, false, true);
    }
    return out;
}

// one call with an upper bound, a retry only if the vocab needs more
std::vector<int> one_pass(const llama_vocab * v, const std::string & text) {
    std::vector<int> out(text.size() + 8);
    int n = llama_tokenize(v, text.data(), static_cast<int32_t>(text.size()),
                           reinterpret_cast<llama_token *>(out.data()),
                           static_cast<int32_t>(out.size()), false, true);
    if (n < 0) {
        out.resize(static_cast<size_t>(-n));
        n = llama_tokenize(v, text.data(), static_cast<int32_t>(text.size()),
                           reinterpret_cast<llama_token *>(out.data()),
                           static_cast<int32_t>(out.size()), false, true);
    }
    out.resize(static_cast<size_t>(n > 0 ? n : 0));
    return out;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <vocab.gguf> [text file] [--bytes N] [--runs N]\n", argv[0]);
        return 2;
    }
    const char * gguf = argv[1];
    std::string  text;
    size_t       bytes = 160000;
    int          runs  = 5;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) bytes = static_cast<size_t>(std::atoll(argv[++i]));
        else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (argv[i][0] != '-') {
            std::FILE * f = std::fopen(argv[i], "rb");
            if (f == nullptr) {
                std::fprintf(stderr, "cannot open %s\n", argv[i]);
                return 2;
            }
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            text.resize(static_cast<size_t>(n > 0 ? n : 0));
            if (!text.empty()) std::fread(text.data(), 1, text.size(), f);
            std::fclose(f);
        }
    }
    if (text.empty()) text = make_text(bytes);

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(gguf, params);
    if (model == nullptr) {
        std::fprintf(stderr, "cannot load %s (a vocab-only GGUF)\n", gguf);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // warm, then take the best of `runs` for each form
    const std::vector<int> ids = one_pass(vocab, text);
    double old_best = 1e9, new_best = 1e9;
    for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        const std::vector<int> a = old_encode(vocab, text);
        old_best = std::min(old_best, ms_since(t0));
        t0 = std::chrono::steady_clock::now();
        const std::vector<int> b = one_pass(vocab, text);
        new_best = std::min(new_best, ms_since(t0));
        if (a != b) {
            std::fprintf(stderr, "FAIL: the two forms' ids differ\n");
            llama_model_free(model);
            return 1;
        }
    }
    std::printf("tokenize-cost: %zu bytes, %zu tokens, old 2-call %.3f ms, one-pass %.3f ms, ratio %.2f\n",
                text.size(), ids.size(), old_best, new_best, old_best / new_best);
    std::printf("             one request: old 3 encodes = 6 calls = %.2f ms, new 1 encode = 1 call = %.2f ms\n",
                old_best * 3.0, new_best);
    llama_model_free(model);
    return 0;
}
