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

// The lead of a generated text: everything from its first non-blank
// character, where blank is a space or a newline. A model may pad before a
// self-opened block; the pad is not part of the opener, so the opener rule
// is applied to the stripped text. split_reasoning and the sampler's
// think-block mask both use this one helper, so they cannot disagree about
// where the lead starts.
std::string_view lead_of(std::string_view s);

// The streaming form: pieces in, (reasoning delta, content delta) out, with a
// hold-back for a close tag that straddles two pieces. Once the block closes
// every further byte is content and goes straight through.
//
// `open_close_ids` is the per-pair id table from ThinkBlockIds, aligned with
// think_tag_pairs(): a pair whose opener/closer is one special token is
// decided by that id, and its text is not looked at. A -1 entry keeps the text
// logic for that role. Empty: the text logic throughout, as before.
class ReasoningStreamer {
public:
    explicit ReasoningStreamer(bool think_open, bool extended_tags = false,
                               const std::vector<std::pair<int, int>>& open_close_ids = {});
    struct Step {
        std::string reasoning;
        std::string content;
    };
    // `token_id` is the id of the token that decoded to `piece`; -1 for a
    // caller that only has text.
    Step push(std::string_view piece, int token_id = -1);
    Step flush();
    bool in_reasoning() const { return in_reasoning_; }
    bool closed() const { return closed_; }

private:
    bool             in_reasoning_;
    bool             undecided_;      // not opened by the template: wait for the first bytes
    std::string      buffer_;
    std::string_view close_ = "</think>";   // the closer of the block that is open
    bool             extended_;
    std::vector<std::pair<int, int>> ids_;
    int              close_id_ = -1;   // the closer id of the open block; -1 = text
    bool             closed_   = false;
    // The block just closed and the answer has not started: the line breaks
    // between the closer and the answer are layout and are dropped, as the
    // non-streaming split drops them.
    bool             content_start_ = false;
};

// The non-streaming split driven by the same (piece, id) sequence the streamer
// takes, so the two paths cannot disagree: a closer spelled from ordinary
// tokens stays reasoning, the special-token closer ends the block.
ReasoningSplit split_reasoning_tokens(const std::vector<std::pair<std::string, int>>& pieces,
                                      bool think_open, bool extended_tags,
                                      const std::vector<std::pair<int, int>>& open_close_ids);

// The tag family the split honours, in the order it tries them: the Qwen pair,
// and with `extended` also the Mistral-family pairs. This is the only place
// they are spelled; anything else that needs them (the sampler's think-block
// mask, clean-turns) asks for them here rather than writing them out. A
// reference to a table built once: no caller copies them, and no caller needs
// to compose one.
const std::vector<std::pair<std::string, std::string>>& think_tag_pairs(bool extended);

}  // namespace lgc
