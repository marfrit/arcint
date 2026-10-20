#pragma once
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    bool        closed = false;   // the block's closer was seen; false = the model never left the block
};

// `think_open`: the rendered prompt ended inside a think block ("<think>").
// When it did not, a block the model opens itself at the very start is
// honoured too: "<think>", and with `extended_tags` (ModelStatus::
// think_tags_extended) also "<thinking>" (Cydonia 24B) or "[THINK]"
// (Magistral tunes), each closed only by its own closer.
ReasoningSplit split_reasoning(std::string_view raw, bool think_open, bool extended_tags = false);

// The streaming form: pieces in, (reasoning delta, content delta) out, with a
// hold-back for a close tag that straddles two pieces. Once the block closes
// every further byte is content and goes straight through.
class ReasoningStreamer {
public:
    explicit ReasoningStreamer(bool think_open, bool extended_tags = false);
    struct Step {
        std::string reasoning;
        std::string content;
    };
    Step push(std::string_view piece);
    Step flush();
    bool in_reasoning() const { return in_reasoning_; }

private:
    bool             in_reasoning_;
    bool             undecided_;      // not opened by the template: wait for the first bytes
    std::string      buffer_;
    std::string_view close_ = "</think>";   // the closer of the block that is open
    bool             extended_;
};

// The tag family the split honours, in the order it tries them: the Qwen pair,
// and with `extended` also the Mistral-family pairs. This is the only place
// they are spelled; anything else that needs them (the sampler's think-block
// mask, clean-turns) asks for them here rather than writing them out. A
// reference to a table built once: no caller copies them, and no caller needs
// to compose one.
const std::vector<std::pair<std::string, std::string>>& think_tag_pairs(bool extended);

}  // namespace lgc
