#include "core/reasoning.h"

#include "core/dialect.h"
#include "core/markers.h"
#include "util/text.h"

#include <iterator>
#include <vector>

namespace lgc {
namespace {
// the think blocks a model may open itself: Qwen's, Cydonia 24B's (Mistral
// Small 3.2) and the Magistral tunes'
struct ThinkTags {
    std::string_view open, close;
};
constexpr ThinkTags kTags[] = {{markers::kThinkOpen, markers::kThinkClose},
                               {markers::kThinkOpenAngle, markers::kThinkCloseAngle},
                               {markers::kThinkOpenSquare, markers::kThinkCloseSquare},
                               {markers::kGemmaThoughtOpen, markers::kGemmaThoughtClose}};

// The slice of kTags a dialect honours: the Qwen pair alone, the Qwen pair and
// the Mistral pairs, or Gemma 4's channel alone (a Gemma model's answer that
// merely starts with the Qwen pair's text stays content).
struct TagRange {
    size_t begin, count;
    const ThinkTags& operator[](size_t i) const { return kTags[begin + i]; }
};
TagRange tag_range(ChatDialect d) {
    switch (d.kind) {
        case ChatDialect::Extended: return {0, 3};
        case ChatDialect::Gemma4: return {3, 1};
        default: return {0, 1};
    }
}
// the pair whose opener starts `lead`, or nullptr
const ThinkTags* opened_by(std::string_view lead, ChatDialect d) {
    const TagRange r = tag_range(d);
    for (size_t i = 0; i < r.count; ++i)
        if (lead.substr(0, r[i].open.size()) == r[i].open) return &r[i];
    return nullptr;
}
// `lead` could still grow into an opener
bool could_open(std::string_view lead, ChatDialect d) {
    const TagRange r = tag_range(d);
    for (size_t i = 0; i < r.count; ++i)
        if (lead.size() < r[i].open.size() && r[i].open.substr(0, lead.size()) == lead) return true;
    return false;
}

// The pairs as owning strings, for callers outside this file. Same table, so
// the mask and the split cannot disagree about what a block looks like.
std::vector<std::pair<std::string, std::string>> build_pairs(ChatDialect d) {
    std::vector<std::pair<std::string, std::string>> out;
    const TagRange r = tag_range(d);
    for (size_t i = 0; i < r.count; ++i) out.emplace_back(std::string(r[i].open), std::string(r[i].close));
    return out;
}

std::string_view strip_edge_newlines(std::string_view s) {
    while (!s.empty() && s.front() == '\n') s.remove_prefix(1);
    while (!s.empty() && s.back() == '\n') s.remove_suffix(1);
    return s;
}
std::string_view strip_leading_newlines(std::string_view s) {
    while (!s.empty() && s.front() == '\n') s.remove_prefix(1);
    return s;
}

// The answer's lead after a closed block (ThinkStart): a closer the model
// repeats there, after any layout. Dropped: the pad and the closer leave the
// front of `buf`. Hold: only layout so far, or a lead that could still grow
// into a closer. Answer: the answer starts here.
enum class Lead { Hold, Dropped, Answer };
Lead repeated_closer(std::string& buf, ChatDialect d) {
    const std::string_view lead = lead_of(buf);
    if (lead.empty()) return Lead::Hold;
    const TagRange r      = tag_range(d);
    bool           loomed = false;
    for (size_t i = 0; i < r.count; ++i) {
        const std::string_view c = r[i].close;
        if (lead.substr(0, c.size()) == c) {
            buf.erase(0, buf.size() - lead.size() + c.size());
            return Lead::Dropped;
        }
        if (lead.size() < c.size() && c.substr(0, lead.size()) == lead) loomed = true;
    }
    return loomed ? Lead::Hold : Lead::Answer;
}

// The whole-text form: every closer at the lead goes, with the pad before it.
std::string_view drop_repeated_closers(std::string_view s, ChatDialect d, bool& dropped) {
    const TagRange r = tag_range(d);
    for (bool hit = true; hit;) {
        hit = false;
        const std::string_view lead = lead_of(s);
        for (size_t i = 0; i < r.count && !hit; ++i) {
            if (lead.substr(0, r[i].close.size()) != r[i].close) continue;
            s       = lead.substr(r[i].close.size());
            hit     = true;
            dropped = true;
        }
    }
    return s;
}
}  // namespace

// One helper, two callers: split_reasoning below and the sampler's
// think-block mask both strip the lead this way, so they cannot
// disagree about where a self-opened block starts.
std::string_view lead_of(std::string_view s) {
    while (!s.empty() && (s.front() == '\n' || s.front() == ' ')) s.remove_prefix(1);
    return s;
}

const std::vector<std::pair<std::string, std::string>>& think_tag_pairs(ChatDialect dialect) {
    // Built once per shape and handed out by reference: a caller that needs a
    // tag asks for it here rather than writing one.
    static const std::vector<std::pair<std::string, std::string>> all   = build_pairs(ChatDialect::Extended);
    static const std::vector<std::pair<std::string, std::string>> qwen  = build_pairs(ChatDialect::Qwen);
    static const std::vector<std::pair<std::string, std::string>> gemma = build_pairs(ChatDialect::Gemma4);
    switch (dialect.kind) {
        case ChatDialect::Extended: return all;
        case ChatDialect::Gemma4: return gemma;
        default: return qwen;
    }
}

ReasoningSplit split_reasoning(std::string_view raw, ThinkStart start, ChatDialect dialect) {
    ReasoningSplit out;
    std::string_view body  = raw;
    std::string_view close = tag_range(dialect)[0].close;   // the template opened it: this dialect's first pair
    if (!start.open()) {
        // The template closed the block: a closer the model repeats at the
        // lead is the template's, and goes with the layout after it.
        bool dropped = false;
        if (start.closed()) raw = drop_repeated_closers(raw, dialect, dropped);
        const std::string_view lead = lead_of(raw);
        const ThinkTags*       tags = opened_by(lead, dialect);
        if (tags == nullptr) {
            out.content = std::string(dropped ? strip_leading_newlines(raw) : raw);
            return out;
        }
        body  = lead.substr(tags->open.size());
        close = tags->close;
    }
    const size_t at = body.find(close);
    if (at == std::string_view::npos) {
        // Never left the block (a length stop, typically): all of it is
        // reasoning, and there is no answer to give.
        out.reasoning = std::string(strip_edge_newlines(body));
        return out;
    }
    out.closed    = true;
    out.reasoning = std::string(strip_edge_newlines(body.substr(0, at)));
    bool repeated = false;   // a closer repeated at the answer's lead is layout too
    out.content   = std::string(
        strip_leading_newlines(drop_repeated_closers(body.substr(at + close.size()), dialect, repeated)));
    return out;
}

ReasoningSplit split_reasoning_tokens(const std::vector<std::pair<std::string, int>>& pieces,
                                      ThinkStart start, ChatDialect dialect,
                                      const std::vector<std::pair<int, int>>& open_close_ids) {
    // One implementation with the streaming path: run the streamer over the
    // same sequence and concatenate its steps. A spelled closer stays reasoning
    // and the special-token closer closes the block in both, by construction.
    ReasoningStreamer streamer(start, dialect, open_close_ids);
    ReasoningSplit    out;
    for (const auto& [piece, id] : pieces) {
        const ReasoningStreamer::Step step = streamer.push(piece, id);
        out.reasoning += step.reasoning;
        out.content   += step.content;
    }
    const ReasoningStreamer::Step tail = streamer.flush();
    out.reasoning += tail.reasoning;
    out.content   += tail.content;
    out.closed = streamer.closed();
    // The text split's edges: the line breaks around the tags are layout, not
    // reasoning or answer. The streamer cannot drop them (they arrive as their
    // own pieces after the step that closed the block); the whole text can.
    // A reply with no block keeps its content as generated, as the text split
    // does.
    out.reasoning = std::string(strip_edge_newlines(out.reasoning));
    if (out.closed) out.content = std::string(strip_leading_newlines(out.content));
    return out;
}

ReasoningStreamer::ReasoningStreamer(ThinkStart start, ChatDialect dialect,
                                     const std::vector<std::pair<int, int>>& open_close_ids)
    : in_reasoning_(start.open()), undecided_(!start.open()), dialect_(dialect), ids_(open_close_ids) {
    after_close_ = start.closed();
    // a template-opened block closes on this dialect's first pair, not the Qwen one
    close_ = tag_range(dialect_)[0].close;
    // A template-opened block: its closer is the first pair's close id when the
    // vocabulary spells it as one token; -1 keeps the text fallback.
    if (in_reasoning_ && !ids_.empty() && ids_.front().second >= 0) close_id_ = ids_.front().second;
}

bool ReasoningStreamer::is_close_id_(int token_id) const {
    if (token_id < 0) return false;
    for (const auto& pair : ids_)
        if (pair.second == token_id) return true;
    return false;
}

// The answer's lead after the block closed: layout and a repeated closer are
// held and dropped, the first byte of the answer releases the rest (its
// leading line breaks dropped, as the whole-text split drops them).
ReasoningStreamer::Step ReasoningStreamer::lead_after_close_(std::string_view piece, int token_id) {
    Step step;
    // the closer as its one token, after nothing but layout: the marker again
    if (is_close_id_(token_id) && lead_of(buffer_).empty()) {
        buffer_.clear();
        return step;
    }
    buffer_ += piece;
    for (;;) {
        const Lead l = repeated_closer(buffer_, dialect_);
        if (l == Lead::Hold) return step;
        if (l == Lead::Answer) break;
    }
    step.content = std::string(strip_leading_newlines(buffer_));
    buffer_.clear();
    content_start_ = false;
    return step;
}

TokenField ReasoningStreamer::field_for(int token_id) const {
    if (in_reasoning_) return TokenField::Reasoning;
    if (undecided_ && token_id >= 0) {
        for (const auto& pair : ids_)
            if (pair.first == token_id) return TokenField::Reasoning;
    }
    return TokenField::Text;
}

ReasoningStreamer::Step ReasoningStreamer::push(std::string_view piece, int token_id) {
    Step step;
    if (undecided_) {
        // The model opened the block itself, as one special token: that id
        // opens it and pairs with its own closer. The whole piece is the
        // marker, so none of its text is output.
        for (size_t i = 0; token_id >= 0 && i < ids_.size(); ++i) {
            if (ids_[i].first != token_id) continue;
            undecided_    = false;
            in_reasoning_ = true;
            close_id_     = ids_[i].second;
            buffer_.clear();
            return step;
        }
        // The template closed an empty block: a closer at the lead is that
        // closer repeated (the token after nothing but layout, or its text,
        // also across pieces). It is dropped with the layout around it; the
        // opener rule below then runs on what follows.
        if (after_close_) {
            if (is_close_id_(token_id) && lead_of(buffer_).empty()) {
                buffer_.clear();
                drop_newlines_ = true;
                return step;
            }
            buffer_ += piece;
            for (;;) {
                const Lead l = repeated_closer(buffer_, dialect_);
                if (l == Lead::Hold) return step;
                if (l == Lead::Answer) break;
                drop_newlines_ = true;
            }
            after_close_ = false;
            piece        = {};   // already in buffer_
        }
        // Not opened by the template. Hold the first bytes until they either
        // spell an opener or cannot: anything else is plain content. Only a
        // pair whose opener has no id takes part in the text rule; a pair with
        // an id is decided by that id alone.
        buffer_ += piece;
        const std::string_view lead = lead_of(buffer_);
        const TagRange         range = tag_range(dialect_);
        const ThinkTags*       tags = nullptr;
        size_t                 tag_i = 0;
        for (size_t i = 0; i < range.count; ++i) {
            if (i < ids_.size() && ids_[i].first >= 0) continue;
            if (lead.substr(0, range[i].open.size()) == range[i].open) {
                tags  = &range[i];
                tag_i = i;
                break;
            }
        }
        if (tags == nullptr) {
            bool loomed = false;
            for (size_t i = 0; i < range.count; ++i) {
                if (i < ids_.size() && ids_[i].first >= 0) continue;
                if (could_open(lead, dialect_) && lead.size() < range[i].open.size()
                    && range[i].open.substr(0, lead.size()) == lead)
                    loomed = true;
            }
            // Only layout so far (an empty lead): an opener token may still
            // come, exactly as the sampler's mask stays undecided on it.
            if (loomed || lead.empty()) return step;   // could still be a tag
            undecided_ = false;
            step.content = drop_newlines_ ? std::string(strip_leading_newlines(buffer_)) : buffer_;
            buffer_.clear();
            return step;
        }
        undecided_    = false;
        in_reasoning_ = true;
        close_        = tags->close;
        close_id_     = tag_i < ids_.size() ? ids_[tag_i].second : -1;
        std::string rest(lead.substr(tags->open.size()));
        buffer_.clear();
        return push(rest, -1);
    }
    if (!in_reasoning_) {
        // The line breaks between the closer and the answer are layout, not
        // answer: the non-streaming split (the reference) drops them, so the
        // stream drops them too -- otherwise the two paths disagree on the same
        // token stream. A closer repeated there is dropped the same way.
        if (content_start_) return lead_after_close_(piece, token_id);
        step.content = std::string(piece);
        return step;
    }
    // The block is open. With a closer id known, only that id closes it: no
    // text is scanned, so a closer spelled from ordinary tokens stays
    // reasoning.
    if (close_id_ >= 0) {
        if (token_id == close_id_) {
            in_reasoning_ = false;
            closed_       = true;
            content_start_ = true;
            buffer_.clear();
            return step;   // the closer's piece is the marker, not content
        }
        step.reasoning = std::string(piece);
        return step;
    }
    buffer_ += piece;
    const size_t at = buffer_.find(close_);
    if (at == std::string::npos) {
        // Emit what cannot be part of a straddling close tag.
        const size_t hold = text::partial_stop_suffix(buffer_, close_);
        const size_t safe = buffer_.size() - hold;
        step.reasoning = buffer_.substr(0, safe);
        buffer_.erase(0, safe);
        return step;
    }
    step.reasoning = buffer_.substr(0, at);
    const std::string rest = buffer_.substr(at + close_.size());
    buffer_.clear();
    in_reasoning_  = false;
    content_start_ = true;
    step.content   = lead_after_close_(rest, -1).content;
    return step;
}

ReasoningStreamer::Step ReasoningStreamer::flush() {
    Step step;
    if (buffer_.empty()) return step;
    if (in_reasoning_) step.reasoning = buffer_;   // never closed: it stays reasoning
    // a lead still held after a closed block: its line breaks are layout
    else if (content_start_ || drop_newlines_) step.content = std::string(strip_leading_newlines(buffer_));
    else step.content = buffer_;
    buffer_.clear();
    return step;
}

}  // namespace lgc
