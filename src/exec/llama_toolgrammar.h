// clean-turns step 2: the tool-call grammar (core/toolgrammar.h) as the
// sampler's TokenConstraint, on libllama's lazy grammar sampler. Device-free:
// it needs a vocabulary, not a context, so a vocab-only load can drive it
// (tools/tool_grammar_probe.cpp).
#pragma once

#ifdef ARCINT_LLAMA

#include <llama.h>

#include <string>
#include <vector>

#include "core/chat.h"
#include "core/sampler.h"

namespace lgc {

class LlamaToolGrammar final : public TokenConstraint {
public:
    // The grammar for `tools`, triggered on the call opener: a trigger TOKEN
    // when this vocabulary holds the opener as one token (the coder's does),
    // else a regex-escaped pattern -- llama.cpp's
    // common_sampling_add_grammar_triggers. Throws std::runtime_error when
    // the generator or llama.cpp's grammar parser refuses (the reference
    // fails the request the same way: "failed to parse grammar").
    LlamaToolGrammar(const llama_vocab* vocab, const std::vector<ToolSpec>& tools, bool parallel);
    ~LlamaToolGrammar() override;
    LlamaToolGrammar(const LlamaToolGrammar&)            = delete;
    LlamaToolGrammar& operator=(const LlamaToolGrammar&) = delete;

    bool allows(int token) override;
    void mask(float* logits, size_t vocab) override;
    void accept(int token) override;

    // For the log and the probe: the GBNF, whether the opener became a
    // trigger token, how many full-row masks (rejections) this request took.
    const std::string& gbnf() const { return gbnf_; }
    bool               trigger_is_token() const { return trigger_is_token_; }
    int                rejections() const { return rejections_; }
    bool               broken() const { return broken_; }

private:
    llama_sampler*                smpl_ = nullptr;
    std::string                   gbnf_;
    bool                          trigger_is_token_ = false;
    bool                          broken_           = false;
    int                           rejections_       = 0;
    std::vector<llama_token_data> cand_;
};

}  // namespace lgc

#endif  // ARCINT_LLAMA
