#include "exec/llama_toolgrammar.h"

#include <cmath>
#include <stdexcept>

#include "core/toolgrammar.h"
#include "exec/backend.h"
#include "util/log.h"

namespace lgc {
namespace {

// common/common.cpp's regex_escape at the pin (the same character set): a WORD trigger that is
// not one token becomes a pattern that matches the word literally.
std::string regex_escape(const std::string& s) {
    static const std::string special = "^$.|?*+()[]{}\\";
    std::string              out;
    for (char c : s) {
        if (special.find(c) != std::string::npos) out += '\\';
        out += c;
    }
    return out;
}

}  // namespace

LlamaToolGrammar::LlamaToolGrammar(const llama_vocab* vocab, const std::vector<ToolSpec>& tools, bool parallel) {
    try {
        gbnf_ = generate_tool_grammar(tools, parallel);
    } catch (const std::invalid_argument& e) {
        // the client's schema: a 400, before any token (handlers.cpp guarded())
        throw InputError(std::string("tool_grammar: ") + e.what());
    }
    if (gbnf_.empty()) throw std::runtime_error("tool grammar: no tools declared");

    const std::string opener = tool_grammar_trigger();
    std::vector<llama_token> ids(opener.size() + 4);
    const int n = llama_tokenize(vocab, opener.c_str(), static_cast<int32_t>(opener.size()), ids.data(),
                                 static_cast<int32_t>(ids.size()), /*add_special*/ false, /*parse_special*/ true);
    std::vector<llama_token> trigger_tokens;
    std::vector<std::string> patterns;
    if (n == 1) {
        trigger_tokens.push_back(ids[0]);
        trigger_is_token_ = true;
    } else {
        patterns.push_back(regex_escape(opener));
    }
    std::vector<const char*> patterns_c;
    for (const std::string& p : patterns) patterns_c.push_back(p.c_str());

    smpl_ = llama_sampler_init_grammar_lazy_patterns(vocab, gbnf_.c_str(), "root", patterns_c.data(), patterns_c.size(),
                                                     trigger_tokens.data(), trigger_tokens.size());
    if (smpl_ == nullptr) throw std::runtime_error("tool grammar: llama.cpp failed to parse the grammar");
}

LlamaToolGrammar::~LlamaToolGrammar() {
    if (smpl_ != nullptr) llama_sampler_free(smpl_);
}

bool LlamaToolGrammar::allows(int token) {
    if (broken_) return true;
    llama_token_data       d{ token, 1.0f, 0.0f };
    llama_token_data_array a{ &d, 1, -1, false };
    llama_sampler_apply(smpl_, &a);
    return d.logit != -INFINITY;
}

void LlamaToolGrammar::mask(float* logits, size_t vocab) {
    if (broken_) return;
    ++rejections_;
    // Only the tokens still in play: a logit already at -inf (the think mask)
    // needs no grammar test, and the apply decodes every candidate it is given.
    cand_.clear();
    cand_.reserve(vocab);
    for (size_t i = 0; i < vocab; ++i)
        if (logits[i] != -INFINITY) cand_.push_back({ static_cast<llama_token>(i), logits[i], 0.0f });
    llama_token_data_array a{ cand_.data(), cand_.size(), -1, false };
    llama_sampler_apply(smpl_, &a);
    for (const llama_token_data& d : cand_)
        if (d.logit == -INFINITY) logits[static_cast<size_t>(d.id)] = -INFINITY;
}

void LlamaToolGrammar::accept(int token) {
    if (broken_) return;
    // Never feed a token the grammar rejects: llama.cpp aborts the process on
    // an end-of-generation token outside a complete stack and throws on any
    // other. The sampler only hands over tokens that passed allows(), so this
    // is the guard for a path that bypassed it; the request then goes on
    // unconstrained and says so, rather than taking the server down.
    if (!allows(token)) {
        broken_ = true;
        log::warn("grammar", "token %d is outside the tool-call grammar; the rest of this turn is unconstrained", token);
        return;
    }
    try {
        llama_sampler_accept(smpl_, token);
    } catch (const std::exception& e) {
        broken_ = true;
        log::warn("grammar", "tool-call grammar refused token %d (%s); the rest of this turn is unconstrained", token,
                  e.what());
    }
}

}  // namespace lgc
