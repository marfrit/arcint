#include "core/reasoning.h"

#include "util/text.h"

#include <iterator>

namespace lgc {
namespace {
constexpr std::string_view kClose = "</think>";
// the think blocks a model may open itself: Qwen's, Cydonia 24B's (Mistral
// Small 3.2) and the Magistral tunes'
struct ThinkTags {
    std::string_view open, close;
};
constexpr ThinkTags kTags[] = {{"<think>", "</think>"}, {"<thinking>", "</thinking>"}, {"[THINK]", "[/THINK]"}};

std::string_view lead_of(std::string_view s) {
    while (!s.empty() && (s.front() == '\n' || s.front() == ' ')) s.remove_prefix(1);
    return s;
}
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

ReasoningStreamer::ReasoningStreamer(bool think_open, bool extended_tags)
    : in_reasoning_(think_open), undecided_(!think_open), extended_(extended_tags) {}

ReasoningStreamer::Step ReasoningStreamer::push(std::string_view piece) {
    Step step;
    if (undecided_) {
        // Not opened by the template. Hold the first bytes until they either
        // spell an opener or cannot: anything else is plain content.
        buffer_ += piece;
        const std::string_view lead = lead_of(buffer_);
        const ThinkTags*       tags = opened_by(lead, extended_);
        if (tags == nullptr) {
            if (could_open(lead, extended_)) return step;   // could still be a tag
            undecided_ = false;
            step.content = buffer_;
            buffer_.clear();
            return step;
        }
        undecided_    = false;
        in_reasoning_ = true;
        close_        = tags->close;
        std::string rest(lead.substr(tags->open.size()));
        buffer_.clear();
        return push(rest);
    }
    if (!in_reasoning_) {
        step.content = std::string(piece);
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
