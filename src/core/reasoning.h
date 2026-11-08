#pragma once
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/dialect.h"
#include "core/markers.h"

namespace lgc {

// Qwen3-family templates open the think block in the prompt: the generation
// prompt ends with the Qwen think opener and a newline, the model answers
// from inside it, and the first thing it emits is reasoning followed by the
// bare think closer. The server is the only party that knows the block was
// pre-opened, so the split is its job: everything before the first close is
// `reasoning_content`, the rest is `content` -- the convention vLLM and
// llama.cpp follow. Until 2026-08-30 the whole thing went back as content,
// the closer included.
struct ReasoningSplit {
    std::string reasoning;
    std::string content;
    bool        closed = false;   // the block's closer was seen; false = the model never left the block
};

// Where the rendered prompt leaves the model with respect to the think block.
//   Undecided  no block at the prompt's end: the model may open one itself
//   Open       the prompt ended inside the block (the Qwen templates, thinking on)
//   Closed     the prompt ended right after a block the template closed (the
//              Qwen templates' empty block with thinking off, Dory's too)
// Converts from bool for the callers that predate Closed: true meant Open.
//
// After a closed block, whichever closed it (the template, or the model with
// thinking on), the answer's lead is layout: its line breaks are dropped, and
// so is a closer the model repeats there. A fine-tune that writes the closer
// once more after the template's (measured: thinking off, the answer began
// with it) would otherwise put it into content. A closer later in the answer,
// away from the lead, is text the model wrote and stays content: the stream
// cannot take back bytes it has sent, and both paths must agree.
struct ThinkStart {
    enum Kind : unsigned char { Undecided, Open, Closed };
    Kind kind = Undecided;

    constexpr ThinkStart() = default;
    constexpr ThinkStart(Kind k) : kind(k) {}
    constexpr ThinkStart(bool open) : kind(open ? Open : Undecided) {}
    constexpr bool open() const { return kind == Open; }
    constexpr bool closed() const { return kind == Closed; }
};

// `start`: where the rendered prompt left the model (ThinkStart). Open: the
// rendered prompt ended inside the Qwen think block.
// Otherwise a block the model opens itself at the very start is
// honoured too: the same opener, and under the Extended dialect also the angle-bracket opener (Cydonia 24B) or
// the square-bracket opener (Magistral tunes), each closed only by its
// own closer. Under Gemma4 the only opener is that model's thought channel;
// a template-opened block closes on the dialect's first pair.
ReasoningSplit split_reasoning(std::string_view raw, ThinkStart start, ChatDialect dialect = {});

// The repeated-closer rule at the answer's lead after a closed block
// (ThinkStart), one function for the split and the sampler's think tracker so
// the two cannot disagree about where the answer starts. `buf` holds the
// lead's bytes not decided yet; each call adds one piece. A pair whose closer
// is one special token (its id in `ids`, aligned with `closers`) is matched by
// that id alone, after nothing but layout: the same text from ordinary tokens
// is answer, as everywhere else in the split. A pair without an id is matched
// by its text, also across pieces, and the lead is held while it is still a
// prefix of that closer. A matched closer leaves `buf` with the layout before
// it, and `dropped` is set. Hold: only layout so far, or a prefix of an id-less
// closer. Answer: the answer starts with what `buf` holds.
enum class LeadState { Hold, Answer };
LeadState repeated_closer_step(std::string& buf, std::string_view piece, int token_id,
                               const std::vector<std::string>& closers,
                               const std::vector<std::pair<int, int>>& ids, bool& dropped);

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
// feyntalk W6b (docs/campaigns/feyntalk.md D2): which side of the think split
// a token belongs to, decided by the token id and the streamer's state, not by
// scanning text. The closer belongs to reasoning; an opener id opens reasoning.
enum class TokenField { Reasoning, Text };

class ReasoningStreamer {
public:
    explicit ReasoningStreamer(ThinkStart start, ChatDialect dialect = {},
                               const std::vector<std::pair<int, int>>& open_close_ids = {});
    struct Step {
        std::string reasoning;
        std::string content;
    };
    // `token_id` is the id of the token that decoded to `piece`; -1 for a
    // caller that only has text.
    Step push(std::string_view piece, int token_id = -1);
    // The token's side BEFORE this push: reasoning while the block is open (the
    // closer included), an opener id while undecided. A piece whose text opens
    // the block is caught by the caller through the step's reasoning bytes.
    TokenField field_for(int token_id) const;
    Step flush();
    bool in_reasoning() const { return in_reasoning_; }
    bool closed() const { return closed_; }

private:
    bool             in_reasoning_;
    bool             undecided_;      // not opened by the template: wait for the first bytes
    std::string      buffer_;
    std::string_view close_ = markers::kThinkClose;   // the closer of the block that is open
    ChatDialect      dialect_;
    std::vector<std::pair<int, int>> ids_;
    int              close_id_ = -1;   // the closer id of the open block; -1 = text
    bool             closed_   = false;
    // The block just closed and the answer has not started: the line breaks
    // between the closer and the answer are layout and are dropped, as the
    // non-streaming split drops them, and so is a repeated closer (ThinkStart).
    // The bytes of a lead not decided yet wait in buffer_.
    bool             content_start_ = false;
    // ThinkStart::Closed and nothing decided yet: a closer at the lead is the
    // template's, repeated, and is dropped before the opener rule runs.
    bool             after_close_ = false;
    // A repeated closer was dropped before the answer: the answer's leading
    // line breaks are layout too (the undecided path keeps them otherwise).
    bool             drop_newlines_ = false;
    std::vector<std::string> closers_;   // this dialect's closers, aligned with ids_

    Step lead_after_close_(std::string_view piece, int token_id);
};

// The non-streaming split driven by the same (piece, id) sequence the streamer
// takes, so the two paths cannot disagree: a closer spelled from ordinary
// tokens stays reasoning, the special-token closer ends the block.
ReasoningSplit split_reasoning_tokens(const std::vector<std::pair<std::string, int>>& pieces,
                                      ThinkStart start, ChatDialect dialect,
                                      const std::vector<std::pair<int, int>>& open_close_ids);

// The tag family the split honours, in the order it tries them: the Qwen pair;
// for the Extended dialect also the Mistral-family pairs; for Gemma4 its
// thought channel alone. This is the only place
// they are spelled; anything else that needs them (the sampler's think-block
// mask, clean-turns) asks for them here rather than writing them out. A
// reference to a table built once: no caller copies them, and no caller needs
// to compose one.
const std::vector<std::pair<std::string, std::string>>& think_tag_pairs(ChatDialect dialect);

}  // namespace lgc
