// audit 15: the request's one tokenisation.
//   - the one-pass tokeniser: an upper-bound initial buffer makes one call and
//     returns the ids the old two-call path (a null-output sizing pass, then an
//     exact-sized pass) makes, on a mixed corpus of text and a merged special;
//     the retry runs only when the buffer was too small and then the ids are
//     still the same;
//   - the request path: prepare_completion tokenises a text prompt exactly once
//     and hands the tokens in `GenerationInput::prompt_ids_text`; a backend
//     that reads them through `prompt_ids_raw` tokenises nothing again (the
//     count is 1, not 2), and the ids are the text's bytes;
//   - the llama backend's own use: encode() calls tokenize_once, prompt_tokens
//     and the multimodal set_prompt read `prompt_ids_raw`, and no null-output
//     llama_tokenize sizing call remains.
#include "api/handlers.h"
#include "config.h"
#include "exec/backend.h"
#include "exec/tokenize-once.h"
#include "harness.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace lgc;

namespace {

// a vocabulary that merges "<tag>" into one special id and otherwise emits one
// id per byte; it reports -need when the output buffer is short, as
// llama_tokenize does
struct FakeVocab {
    int calls = 0;
    std::vector<int> ids(std::string_view text) const {
        std::vector<int> v;
        for (size_t i = 0; i < text.size();) {
            if (text.substr(i, 5) == "<tag>") {
                v.push_back(9000);
                i += 5;
            } else {
                v.push_back(static_cast<unsigned char>(text[i]));
                ++i;
            }
        }
        return v;
    }
    int run(std::string_view text, int * out, int cap) {
        ++calls;
        const std::vector<int> v = ids(text);
        if (cap < static_cast<int>(v.size())) return -static_cast<int>(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = v[i];
        return static_cast<int>(v.size());
    }
};

const std::vector<std::string> kCorpus = {
    "", "a", "hello world",
    "the template tag <tag> in the middle",
    "two <tag><tag> next to each other",
    std::string(4000, 'x'),
};

// the old encode path: a null-output pass for the size, then an exact buffer
std::vector<int> two_pass(FakeVocab & v, std::string_view text) {
    const int need = -v.run(text, nullptr, 0);
    std::vector<int> out(static_cast<size_t>(need));
    const int n = v.run(text, out.data(), need);
    out.resize(static_cast<size_t>(n));
    return out;
}

}  // namespace

TEST(tokenize_once_makes_the_same_ids_as_two_passes) {
    for (const std::string & text : kCorpus) {
        FakeVocab a, b;
        const std::vector<int> old_ids = two_pass(a, text);
        const std::vector<int> new_ids = tokenize_once(
            [&](const char * p, int32_t len, int * out, int32_t cap) {
                return b.run(std::string_view(p, static_cast<size_t>(len)), out, cap);
            },
            text, text.size() + 8);
        CHECK_EQ(new_ids.size(), old_ids.size());
        CHECK(new_ids == old_ids);
        CHECK_EQ(a.calls, 2);
        CHECK_EQ(b.calls, 1);   // the upper bound was enough
    }
}

TEST(tokenize_once_retries_only_when_the_bound_was_short) {
    FakeVocab v;
    const std::string text = "a text whose upper bound is deliberately short";
    const std::vector<int> ids = tokenize_once(
        [&](const char * p, int32_t len, int * out, int32_t cap) {
            return v.run(std::string_view(p, static_cast<size_t>(len)), out, cap);
        },
        text, 4);   // too small: the first call reports -n
    CHECK(ids == v.ids(text));
    CHECK_EQ(v.calls, 2);

    FakeVocab w;
    CHECK(tokenize_once(
              [&](const char * p, int32_t len, int * out, int32_t cap) {
                  return w.run(std::string_view(p, static_cast<size_t>(len)), out, cap);
              },
              text, text.size() + 8) == w.ids(text));
    CHECK_EQ(w.calls, 1);
}

namespace {

class CountingTokenizer final : public Tokenizer {
public:
    mutable int encodes = 0;
    std::vector<int> encode(std::string_view text) override {
        ++encodes;
        std::vector<int> v;
        for (char c : text) v.push_back(static_cast<unsigned char>(c));
        return v;
    }
    std::string decode(const std::vector<int> & ids) override {
        return std::string(ids.begin(), ids.end());
    }
    std::string decode_one(int id) override { return std::string(1, static_cast<char>(id)); }
    int         eos_id() const override { return 0; }
};

class ContractBackend final : public Backend {
public:
    ContractBackend() {
        st.id          = "stub";
        st.served_id   = "stub";
        st.n_ctx_train = kStubNCtxTrain;
        st.loaded      = true;
        st.stub        = true;
        st.n_ctx       = 4096;
    }
    const ModelStatus & status() const override { return st; }
    Tokenizer &         tokenizer() override { return tok; }
    std::string         render_chat(const ChatRequest &) const override { return {}; }
    FinishReason        generate(const GenerationInput & in, int, const TokenCallback &,
                                 GenerationStats & stats) override {
        // what the llama backend does: the API's tokens, nothing re-tokenised
        const std::vector<int> ids = prompt_ids_raw(in, tok);
        stats.prompt_tokens = static_cast<int>(ids.size());
        seen_ids = ids;
        return FinishReason::Stop;
    }

    CountingTokenizer tok;
    std::vector<int>   seen_ids;
    ModelStatus        st;
};

}  // namespace

TEST(a_text_request_tokenises_once_and_the_backend_reuses_the_tokens) {
    Config cfg;
    for (const std::string & prompt : {std::string("hello world"), std::string(200, 'a')}) {
        ContractBackend backend;
        api::Context    ctx{&cfg, &backend, nullptr};

        nlohmann::json body = {{"model", "stub"}, {"prompt", prompt}};
        api::PreparedCompletion prep;
        const auto err = api::prepare_completion(ctx, body, prep);
        CHECK(!err.has_value());

        CHECK_EQ(backend.tok.encodes, 1);   // the API's count, once
        std::vector<int> bytes;
        for (char c : prompt) bytes.push_back(static_cast<unsigned char>(c));
        CHECK(prep.input.prompt_ids_text == bytes);
        CHECK(prep.input.prompt_ids.empty());

        GenerationStats stats;
        backend.generate(prep.input, 0, [](std::string_view, int) { return Control::Continue; }, stats);
        CHECK_EQ(backend.tok.encodes, 1);   // the backend did not tokenise it again
        CHECK(backend.seen_ids == bytes);
        CHECK_EQ(stats.prompt_tokens, static_cast<int>(bytes.size()));
    }
}

TEST(an_explicit_id_prompt_is_not_tokenised) {
    Config cfg;
    ContractBackend backend;
    api::Context    ctx{&cfg, &backend, nullptr};
    nlohmann::json body = {{"model", "stub"}, {"prompt", {10, 11, 12}}};
    api::PreparedCompletion prep;
    const auto err = api::prepare_completion(ctx, body, prep);
    CHECK(!err.has_value());
    CHECK_EQ(backend.tok.encodes, 0);
    CHECK_EQ(prep.input.prompt_ids.size(), 3u);
    CHECK(prep.input.prompt_ids_text.empty());
}

TEST(the_llama_backend_reads_the_api_tokens_and_tokenises_once) {
    std::ifstream f(std::string(ARCINT_SOURCE_DIR) + "/src/exec/backend_llama.cpp");
    CHECK(f.good());
    const std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(src.find("return tokenize_once(") != std::string::npos);
    CHECK(src.find("prompt_ids_raw(in, *tokenizer_)") != std::string::npos);
    CHECK(src.find("sampler.set_prompt(prompt_ids_raw(in, *tokenizer_))") != std::string::npos);
    CHECK(src.find("llama_tokenize(v_, text.data(), static_cast<int32_t>(text.size()), nullptr, 0") ==
          std::string::npos);
}
