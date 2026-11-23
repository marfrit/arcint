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
                               {markers::kGemmaThoughtOpen, markers::kGemmaThoughtClose},
                               {markers::kInkThinkOpen, markers::kInkEndMessage}};

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
        case ChatDialect::Inkling: return {4, 1};
        // Neutral: the Qwen pair's ids decide a block where the vocabulary holds them as one
        // token each; the text rule is absent (a "runs" model's tags are not guessed at).
        case ChatDialect::Neutral: return {0, 0};
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

// The whole-text form (a vocabulary with no tag ids: the text rule throughout): every closer at the lead goes, with the pad before it.
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

LeadState repeated_closer_step(std::string& buf, std::string_view piece, int token_id,
                               const std::vector<std::string>& closers,
                               const std::vector<std::pair<int, int>>& ids, bool& dropped) {
    auto has_id = [&](size_t i) { return i < ids.size() && ids[i].second >= 0; };
    if (token_id >= 0 && lead_of(buf).empty()) {
        for (size_t i = 0; i < closers.size(); ++i) {
            if (!has_id(i) || ids[i].second != token_id) continue;
            buf.clear();   // the closer token and the layout before it
            dropped = true;
            return LeadState::Hold;
        }
    }
    buf += piece;
    for (;;) {
        const std::string_view lead = lead_of(buf);
        if (lead.empty()) return LeadState::Hold;
        bool hit = false, loomed = false;
        for (size_t i = 0; i < closers.size() && !hit; ++i) {
            if (has_id(i)) continue;   // decided by its id alone
            const std::string_view c = closers[i];
            if (lead.substr(0, c.size()) == c) {
                buf.erase(0, buf.size() - lead.size() + c.size());
                dropped = hit = true;
            } else if (lead.size() < c.size() && c.substr(0, lead.size()) == lead) {
                loomed = true;
            }
        }
        if (!hit) return loomed ? LeadState::Hold : LeadState::Answer;
    }
}

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
    static const std::vector<std::pair<std::string, std::string>> inkling = build_pairs(ChatDialect::Inkling);
    // Neutral keeps the Qwen pair for the id table, while its text range is empty.
    static const std::vector<std::pair<std::string, std::string>> neutral = build_pairs(ChatDialect::Qwen);
    switch (dialect.kind) {
        case ChatDialect::Extended: return all;
        case ChatDialect::Gemma4: return gemma;
        case ChatDialect::Inkling: return inkling;
        case ChatDialect::Neutral: return neutral;
        default: return qwen;
    }
}

ReasoningSplit split_reasoning(std::string_view raw, ThinkStart start, ChatDialect dialect) {
    ReasoningSplit out;
    if (dialect.neutral()) {
        // no text scan: the neutral dialect splits only by single-token ids (the token path);
        // a text-only caller gets the bytes as they came
        out.content = std::string(raw);
        return out;
    }
    if (dialect.inkling()) {
        // The framed-block form: one streamer pass over the whole text, so the
        // streaming and non-streaming paths cannot disagree. A reply that never
        // opened a thinking block has content only, and no block is closed.
        ReasoningStreamer streamer(start, dialect);
        const ReasoningStreamer::Step step = streamer.push(raw);
        const ReasoningStreamer::Step tail = streamer.flush();
        out.reasoning = std::string(strip_edge_newlines(step.reasoning + tail.reasoning));
        out.content   = step.content + tail.content;
        out.closed    = streamer.closed();
        return out;
    }
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
    // the neutral dialect has no text tags: only the vocabulary's single-token ids decide
    // a block, so neither a text closer list nor a text close tag is armed
    const bool neutral = dialect_.neutral();
    if (!neutral) {
        for (const auto& pair : think_tag_pairs(dialect_)) closers_.push_back(pair.second);
        // a template-opened block closes on this dialect's first pair, not the Qwen one
        close_ = tag_range(dialect_)[0].close;
    } else {
        close_ = {};
    }
    // A template-opened block: its closer is the first pair's close id when the
    // vocabulary spells it as one token; -1 keeps the text fallback.
    if (in_reasoning_ && !ids_.empty() && ids_.front().second >= 0) close_id_ = ids_.front().second;
    if (dialect_.inkling() && !ids_.empty() && ids_.front().second >= 0) close_id_ = ids_.front().second;
}

// The answer's lead after the block closed: layout and a repeated closer are
// held and dropped, the first byte of the answer releases the rest (its
// leading line breaks dropped, as the whole-text split drops them).
ReasoningStreamer::Step ReasoningStreamer::lead_after_close_(std::string_view piece, int token_id) {
    Step step;
    bool dropped = false;
    if (repeated_closer_step(buffer_, piece, token_id, closers_, ids_, dropped) == LeadState::Hold) return step;
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
    if (dialect_.inkling()) return ink_push(piece, token_id);
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
        // closer repeated (repeated_closer_step: its token after nothing but
        // layout, or for an id-less pair its text, also across pieces). It is
        // dropped with the layout around it; the opener rule below then runs on
        // what follows, exactly as the sampler's tracker does.
        if (after_close_) {
            bool dropped = false;
            const LeadState l = repeated_closer_step(buffer_, piece, token_id, closers_, ids_, dropped);
            if (dropped) drop_newlines_ = true;
            if (l == LeadState::Hold) return step;
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

namespace {
bool ink_starts(std::string_view s, std::string_view m) {
    return s.size() >= m.size() && s.substr(0, m.size()) == m;
}
bool ink_prefix(std::string_view s, std::string_view m) {
    return s.size() < m.size() && m.substr(0, s.size()) == s;
}
// The layout between a block's closer and the next one, dropped as lead_of
// drops it before a block or an answer.
void ink_drop_lead(std::string& s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == '\n' || s[i] == ' ' || s[i] == '\t' || s[i] == '\r')) ++i;
    s.erase(0, i);
}
}  // namespace

ReasoningStreamer::Step ReasoningStreamer::ink_push(std::string_view piece, int token_id) {
    Step step;
    // Thinking with a known closer id: only that token closes, as everywhere
    // else, so a closer spelled from ordinary tokens stays reasoning.
    if (ink_stage_ == InkStage::Thinking && close_id_ >= 0 && token_id == close_id_) {
        ink_buf_.clear();
        in_reasoning_ = false;
        closed_       = true;
        ink_stage_    = InkStage::AfterEnd;
        return step;
    }
    ink_buf_.append(piece);
    for (;;) {
        if (ink_buf_.empty() && ink_stage_ != InkStage::Thinking) return step;
        switch (ink_stage_) {
            case InkStage::Lead: {
                if (ink_starts(ink_buf_, markers::kInkMessageModel)) {
                    ink_buf_.erase(0, markers::kInkMessageModel.size());
                    ink_stage_ = InkStage::NeedKind;
                    continue;
                }
                if (ink_starts(ink_buf_, markers::kInkContentThink)) {
                    ink_buf_.erase(0, markers::kInkContentThink.size());
                    in_reasoning_ = true;
                    ink_stage_    = InkStage::Thinking;
                    continue;
                }
                if (ink_starts(ink_buf_, markers::kInkContentText)) {
                    ink_buf_.erase(0, markers::kInkContentText.size());
                    ink_stage_ = InkStage::Answer;
                    continue;
                }
                if (ink_prefix(ink_buf_, markers::kInkMessageModel) ||
                    ink_prefix(ink_buf_, markers::kInkContentThink) ||
                    ink_prefix(ink_buf_, markers::kInkContentText))
                    return step;
                // Plain content with no framing at all.
                ink_stage_ = InkStage::Answer;
                continue;
            }
            case InkStage::NeedKind: {
                if (ink_starts(ink_buf_, markers::kInkContentThink)) {
                    ink_buf_.erase(0, markers::kInkContentThink.size());
                    in_reasoning_ = true;
                    ink_stage_    = InkStage::Thinking;
                    continue;
                }
                if (ink_starts(ink_buf_, markers::kInkContentText)) {
                    ink_buf_.erase(0, markers::kInkContentText.size());
                    ink_stage_ = InkStage::Answer;
                    continue;
                }
                if (ink_prefix(ink_buf_, markers::kInkContentThink) ||
                    ink_prefix(ink_buf_, markers::kInkContentText))
                    return step;
                // Not an answer marker: this header introduced a tool call. The
                // header goes out with the block, for the parser to find.
                step.content = std::string(markers::kInkMessageModel) + ink_buf_;
                ink_buf_.clear();
                ink_stage_ = InkStage::Tail;
                return step;
            }
            case InkStage::Thinking: {
                if (!ink_buf_.empty()) {
                    if (close_id_ >= 0) {
                        // The closer is a special token; text is not scanned.
                        step.reasoning += ink_buf_;
                        ink_buf_.clear();
                        return step;
                    }
                    const size_t at = ink_buf_.find(markers::kInkEndMessage);
                    if (at != std::string::npos) {
                        step.reasoning += ink_buf_.substr(0, at);
                        ink_buf_.erase(0, at + markers::kInkEndMessage.size());
                        in_reasoning_ = false;
                        closed_       = true;
                        ink_stage_    = InkStage::AfterEnd;
                        continue;
                    }
                    const size_t hold = text::partial_stop_suffix(ink_buf_, markers::kInkEndMessage);
                    const size_t safe = ink_buf_.size() - hold;
                    step.reasoning += ink_buf_.substr(0, safe);
                    ink_buf_.erase(0, safe);
                }
                return step;
            }
            case InkStage::AfterEnd: {
                ink_drop_lead(ink_buf_);
                if (ink_buf_.empty()) return step;
                if (ink_starts(ink_buf_, markers::kInkModelEnd)) {
                    ink_buf_.clear();
                    return step;
                }
                if (ink_starts(ink_buf_, markers::kInkMessageModel)) {
                    ink_buf_.erase(0, markers::kInkMessageModel.size());
                    ink_stage_ = InkStage::NeedKind;
                    continue;
                }
                if (ink_starts(ink_buf_, markers::kInkContentThink)) {
                    ink_buf_.erase(0, markers::kInkContentThink.size());
                    in_reasoning_ = true;
                    ink_stage_    = InkStage::Thinking;
                    continue;
                }
                if (ink_starts(ink_buf_, markers::kInkContentText)) {
                    ink_buf_.erase(0, markers::kInkContentText.size());
                    ink_stage_ = InkStage::Answer;
                    continue;
                }
                if (ink_prefix(ink_buf_, markers::kInkMessageModel) ||
                    ink_prefix(ink_buf_, markers::kInkContentThink) ||
                    ink_prefix(ink_buf_, markers::kInkContentText) ||
                    ink_prefix(ink_buf_, markers::kInkModelEnd))
                    return step;
                ink_stage_ = InkStage::Answer;
                continue;
            }
            case InkStage::Answer: {
                if (!ink_buf_.empty()) {
                    const size_t eos = ink_buf_.find(markers::kInkModelEnd);
                    const size_t end = ink_buf_.find(markers::kInkEndMessage);
                    if (eos != std::string::npos && (end == std::string::npos || eos < end)) {
                        step.content += ink_buf_.substr(0, eos);
                        ink_buf_.clear();
                        return step;
                    }
                    if (end != std::string::npos) {
                        step.content += ink_buf_.substr(0, end);
                        ink_buf_.erase(0, end + markers::kInkEndMessage.size());
                        ink_stage_ = InkStage::AfterEnd;
                        continue;
                    }
                    size_t hold = text::partial_stop_suffix(ink_buf_, markers::kInkEndMessage);
                    hold = std::max(hold, text::partial_stop_suffix(ink_buf_, markers::kInkModelEnd));
                    const size_t safe = ink_buf_.size() - hold;
                    step.content += ink_buf_.substr(0, safe);
                    ink_buf_.erase(0, safe);
                }
                return step;
            }
            case InkStage::Tail: {
                const size_t eos = ink_buf_.find(markers::kInkModelEnd);
                if (eos != std::string::npos) {
                    step.content += ink_buf_.substr(0, eos);
                    ink_buf_.clear();
                    return step;
                }
                // A call block is handed out whole (the parser needs it); hold
                // only a partial model-end marker, which a text-fallback token
                // stream may split. The tool parser gets the bytes when it has
                // them.
                const size_t hold = text::partial_stop_suffix(ink_buf_, markers::kInkModelEnd);
                const size_t safe = ink_buf_.size() - hold;
                step.content += ink_buf_.substr(0, safe);
                ink_buf_.erase(0, safe);
                return step;
            }
        }
    }
}

ReasoningStreamer::Step ReasoningStreamer::ink_flush() {
    Step step;
    switch (ink_stage_) {
        case InkStage::Thinking: step.reasoning = ink_buf_; break;
        case InkStage::NeedKind:
            // The turn ended right after a header: keep what arrived rather
            // than dropping it, as the tool parser keeps an unterminated block.
            step.content = std::string(markers::kInkMessageModel) + ink_buf_;
            break;
        case InkStage::AfterEnd: {
            ink_drop_lead(ink_buf_);
            const size_t eos = ink_buf_.find(markers::kInkModelEnd);
            step.content = eos == std::string::npos ? ink_buf_ : ink_buf_.substr(0, eos);
            break;
        }
        default: step.content = ink_buf_; break;
    }
    ink_buf_.clear();
    return step;
}

ReasoningStreamer::Step ReasoningStreamer::flush() {
    if (dialect_.inkling()) return ink_flush();
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
