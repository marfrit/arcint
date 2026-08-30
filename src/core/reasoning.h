#pragma once
#include <string>
#include <string_view>

namespace lgc {

// Qwen3-family templates open the think block in the prompt: the generation
// prompt ends with "<think>\n", the model answers from inside it, and the
// first thing it emits is reasoning followed by a bare "</think>". The server
// is the only party that knows the block was pre-opened, so the split is its
// job: everything before the first close is `reasoning_content`, the rest is
// `content` -- the convention vLLM and llama.cpp follow. Until 2026-08-30 the
// whole thing went back as content, "</think>" included.
struct ReasoningSplit {
    std::string reasoning;
    std::string content;
    bool        closed = false;   // a "</think>" was seen; false = the model never left the block
};

// `think_open`: the rendered prompt ended inside a think block. When it did
// not, a "<think>" the model emits itself at the very start is honoured too.
ReasoningSplit split_reasoning(std::string_view raw, bool think_open);

// The streaming form: pieces in, (reasoning delta, content delta) out, with a
// hold-back for a close tag that straddles two pieces. Once the block closes
// every further byte is content and goes straight through.
class ReasoningStreamer {
public:
    explicit ReasoningStreamer(bool think_open);
    struct Step {
        std::string reasoning;
        std::string content;
    };
    Step push(std::string_view piece);
    Step flush();
    bool in_reasoning() const { return in_reasoning_; }

private:
    bool        in_reasoning_;
    bool        undecided_;      // not opened by the template: wait for the first bytes
    std::string buffer_;
};

}  // namespace lgc
