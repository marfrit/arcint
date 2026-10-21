// clean-turns: what the think-block mask will see on THIS artifact, without a
// card. Loads the GGUF with vocab_only (metadata and vocabulary only, no
// tensors, no device, no card seat) and answers the three facts the mask's
// design rests on:
//   1. which ids llama_vocab_is_eog answers true for (the mask's set),
//   2. whether each think-tag of the family tokenizes to ONE id (so the
//      opener can be blocked by id) or to several (matched as text),
//   3. the token type the vocabulary gives those ids.
// Tag strings are composed from characters here, and the output prints ids,
// lengths and types only - never a marker literal. That is the defect this
// campaign tracks, so the tool does not carry it either.
//
// Build against the tree arcint was built against (no arcint headers needed):
//   g++ -std=c++20 -fopenmp -I<llama>/include -I<llama>/ggml/include \
//       tools/vocab_eog_probe.cpp -o vocab_eog_probe \
//       <build>/llama.cpp/src/libllama.a <build>/llama.cpp/ggml/src/libggml{,-base,-cpu}.a \
//       <build>/llama.cpp/ggml/src/ggml-opencl/libggml-opencl.a -lOpenCL -lm -ldl
// (the OpenCL library is only needed to satisfy ggml's backend registry; with
// vocab_only no device is opened, so this runs while the cards are busy.)
#include <llama.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string tag(const char * word, bool closing) {
    return std::string(1, '<') + (closing ? "/" : "") + word + std::string(1, '>');
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <file.gguf>\n", argv[0]); return 2; }
    gpt_params_unused:;
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;        // vocabulary only: no tensors, no device, no card
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8;
    cp.n_seq_max = 1;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) { std::fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * v = llama_model_get_vocab(m);
    const int n = llama_vocab_n_tokens(v);
    std::printf("vocabulary tokens: %d\n", n);

    std::vector<int> eog;
    for (int id = 0; id < n; ++id) if (llama_vocab_is_eog(v, id)) eog.push_back(id);
    std::printf("ids llama_vocab_is_eog answers true for: %zu ->", eog.size());
    for (int id : eog) std::printf(" %d", id);
    std::printf("\n");

    const char * words[] = {"think", "thinking", "reason"};
    for (const char * w : words) {
        for (int closing = 0; closing < 2; ++closing) {
            const std::string t = tag(w, closing);
            std::vector<llama_token> out(16, -1);
            const int need = llama_tokenize(v, t.data(), (int32_t) t.size(),
                                            out.data(), (int32_t) out.size(),
                                            /*add_special=*/false, /*parse_special=*/true);
            std::printf("%-10s %-7s chars=%zu -> ids=%d", w, closing ? "closer" : "opener",
                        t.size(), need);
            if (need > 0) for (int i = 0; i < need && i < (int) out.size(); ++i) std::printf(" %d", out[i]);
            // the type the vocabulary assigns to the single-id form
            if (need == 1) {
                const llama_token_attr a = llama_vocab_get_attr(v, out[0]);
                std::printf("   attr: control=%d unknown=%d user_defined=%d normal=%d",
                            (a & LLAMA_TOKEN_ATTR_CONTROL) != 0,
                            (a & LLAMA_TOKEN_ATTR_UNKNOWN) != 0,
                            (a & LLAMA_TOKEN_ATTR_USER_DEFINED) != 0,
                            (a & LLAMA_TOKEN_ATTR_NORMAL) != 0);
            }
            std::printf("\n");
        }
    }
    llama_model_free(m);
    return 0;
}
