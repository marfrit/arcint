#include "core/reasoning.h"

#include "util/text.h"

#include <iterator>
#include <vector>

namespace lgc {
namespace {
constexpr std::string_view kClose = "</think>";
// the think blocks a model may open itself: Qwen's, Cydonia 24B's (Mistral
// Small 3.2) and the Magistral tunes'
struct ThinkTags {
    std::string_view open, close;
};
constexpr ThinkTags kTags[] = {{"<think>", "</think>"}, {"<thinking>", "</thinking>"}, {"[THINK]", "[/THINK]"}};

// the pairs a model may open: <think> only, or all of them
size_t n_tags(bool extended) { return extended ? std::size(kTags) : 1; }
// the pair whose opener starts `lead`, or nullptr
const ThinkTags* opened_by(std::string_view lead, bool extended) {
    for (size_t i = 0; i < n_tags(extended); ++i)
        if (lead.substr(0, kTags[i].open.size()) == kTags[i].open) return &kTags[i];
    return nullptr;
}
// `lead` could still grow into an opener
bool could_open(std::string_view lead, bool extended) {
    for (size_t i = 0; i < n_tags(extended); ++i)
        if (lead.size() < kTags[i].open.size() && kTags[i].open.substr(0, lead.size()) == lead) return true;
    return false;
}

// The pairs as owning strings, for callers outside this file. Same table, so
// the mask and the split cannot disagree about what a block looks like.
std::vector<std::pair<std::string, std::string>> build_pairs(bool extended) {
    std::vector<std::pair<std::string, std::string>> out;
    for (size_t i = 0; i < n_tags(extended); ++i)
        out.emplace_back(std::string(kTags[i].open), std::string(kTags[i].close));
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
}  // namespace

// One helper, two callers: split_reasoning below and the sampler's
// think-block mask both strip the lead this way, so they cannot
// disagree about where a self-opened block starts.
std::string_view lead_of(std::string_view s) {
    while (!s.empty() && (s.front() == '\n' || s.front() == ' ')) s.remove_prefix(1);
    return s;
}

const std::vector<std::pair<std::string, std::string>>& think_tag_pairs(bool extended) {
    // Built once per shape and handed out by reference: a caller that needs a
    // tag asks for it here rather than writing one.
    static const std::vector<std::pair<std::string, std::string>> all  = build_pairs(true);
    static const std::vector<std::pair<std::string, std::string>> qwen = build_pairs(false);
    return extended ? all : qwen;
}

ReasoningSplit split_reasoning(std::string_view raw, bool think_open, bool extended_tags) {
    ReasoningSplit out;
    std::string_view body  = raw;
    std::string_view close = kClose;
    if (!think_open) {
        const std::string_view lead = lead_of(raw);
        const ThinkTags*       tags = opened_by(lead, extended_tags);
        if (tags == nullptr) {
            out.content = std::string(raw);
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
    out.content   = std::string(strip_leading_newlines(body.substr(at + close.size())));
    return out;
}

ReasoningSplit split_reasoning_tokens(const std::vector<std::pair<std::string, int>>& pieces,
                                      bool think_open, bool extended_tags,
                                      const std::vector<std::pair<int, int>>& open_close_ids) {
    // One implementation with the streaming path: run the streamer over the
    // same sequence and concatenate its steps. A spelled closer stays reasoning
    // and the special-token closer closes the block in both, by construction.
    ReasoningStreamer streamer(think_open, extended_tags, open_close_ids);
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

ReasoningStreamer::ReasoningStreamer(bool think_open, bool extended_tags,
                                     const std::vector<std::pair<int, int>>& open_close_ids)
    : in_reasoning_(think_open), undecided_(!think_open), extended_(extended_tags), ids_(open_close_ids) {
    // A template-opened block: its closer is the first pair's close id when the
    // vocabulary spells it as one token; -1 keeps the text fallback.
    if (in_reasoning_ && !ids_.empty() && ids_.front().second >= 0) close_id_ = ids_.front().second;
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
        // Not opened by the template. Hold the first bytes until they either
        // spell an opener or cannot: anything else is plain content. Only a
        // pair whose opener has no id takes part in the text rule; a pair with
        // an id is decided by that id alone.
        buffer_ += piece;
        const std::string_view lead = lead_of(buffer_);
        const ThinkTags*       tags = nullptr;
        size_t                 tag_i = 0;
        for (size_t i = 0; i < n_tags(extended_); ++i) {
            if (i < ids_.size() && ids_[i].first >= 0) continue;
            if (lead.substr(0, kTags[i].open.size()) == kTags[i].open) {
                tags  = &kTags[i];
                tag_i = i;
                break;
            }
        }
        if (tags == nullptr) {
            bool loomed = false;
            for (size_t i = 0; i < n_tags(extended_); ++i) {
                if (i < ids_.size() && ids_[i].first >= 0) continue;
                if (could_open(lead, extended_) && lead.size() < kTags[i].open.size()
                    && kTags[i].open.substr(0, lead.size()) == lead)
                    loomed = true;
            }
            // Only layout so far (an empty lead): an opener token may still
            // come, exactly as the sampler's mask stays undecided on it.
            if (loomed || lead.empty()) return step;   // could still be a tag
            undecided_ = false;
            step.content = buffer_;
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
        // token stream.
        std::string text(piece);
        if (content_start_) {
            text = std::string(strip_leading_newlines(text));
            if (!text.empty()) content_start_ = false;
        }
        step.content = std::move(text);
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
    std::string_view rest(buffer_);
    rest.remove_prefix(at + close_.size());
    step.content = std::string(strip_leading_newlines(rest));
    content_start_ = step.content.empty();
    buffer_.clear();
    in_reasoning_ = false;
    return step;
}

ReasoningStreamer::Step ReasoningStreamer::flush() {
    Step step;
    if (buffer_.empty()) return step;
    if (in_reasoning_) step.reasoning = buffer_;   // never closed: it stays reasoning
    else step.content = buffer_;
    buffer_.clear();
    return step;
}

}  // namespace lgc
