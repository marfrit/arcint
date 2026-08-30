#include "core/reasoning.h"

#include "util/text.h"

namespace lgc {
namespace {
constexpr std::string_view kOpen  = "<think>";
constexpr std::string_view kClose = "</think>";

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

ReasoningSplit split_reasoning(std::string_view raw, bool think_open) {
    ReasoningSplit out;
    std::string_view body = raw;
    if (!think_open) {
        std::string_view lead = raw;
        while (!lead.empty() && (lead.front() == '\n' || lead.front() == ' ')) lead.remove_prefix(1);
        if (lead.substr(0, kOpen.size()) != kOpen) {
            out.content = std::string(raw);
            return out;
        }
        body = lead.substr(kOpen.size());
    }
    const size_t close = body.find(kClose);
    if (close == std::string_view::npos) {
        // Never left the block (a length stop, typically): all of it is
        // reasoning, and there is no answer to give.
        out.reasoning = std::string(strip_edge_newlines(body));
        return out;
    }
    out.closed    = true;
    out.reasoning = std::string(strip_edge_newlines(body.substr(0, close)));
    out.content   = std::string(strip_leading_newlines(body.substr(close + kClose.size())));
    return out;
}

ReasoningStreamer::ReasoningStreamer(bool think_open)
    : in_reasoning_(think_open), undecided_(!think_open) {}

ReasoningStreamer::Step ReasoningStreamer::push(std::string_view piece) {
    Step step;
    if (undecided_) {
        // Not opened by the template. Hold the first bytes until they either
        // spell "<think>" or cannot: anything else is plain content.
        buffer_ += piece;
        std::string_view lead = buffer_;
        while (!lead.empty() && (lead.front() == '\n' || lead.front() == ' ')) lead.remove_prefix(1);
        if (lead.size() < kOpen.size()) {
            if (kOpen.substr(0, lead.size()) == lead) return step;   // could still be the tag
            undecided_ = false;
            step.content = buffer_;
            buffer_.clear();
            return step;
        }
        undecided_ = false;
        if (lead.substr(0, kOpen.size()) == kOpen) {
            in_reasoning_ = true;
            buffer_       = std::string(lead.substr(kOpen.size()));
            std::string rest;
            rest.swap(buffer_);
            return push(rest);
        }
        step.content = buffer_;
        buffer_.clear();
        return step;
    }
    if (!in_reasoning_) {
        step.content = std::string(piece);
        return step;
    }
    buffer_ += piece;
    const size_t close = buffer_.find(kClose);
    if (close == std::string::npos) {
        // Emit what cannot be part of a straddling close tag.
        const size_t hold = text::partial_stop_suffix(buffer_, kClose);
        const size_t safe = buffer_.size() - hold;
        step.reasoning = buffer_.substr(0, safe);
        buffer_.erase(0, safe);
        return step;
    }
    step.reasoning = buffer_.substr(0, close);
    std::string_view rest(buffer_);
    rest.remove_prefix(close + kClose.size());
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
