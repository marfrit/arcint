#include "core/stop.h"

#include <algorithm>

#include "util/text.h"

namespace lgc {

StopMatcher::StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
    // Drop empties defensively; an empty stop matches at position 0 and would
    // end every generation before the first token. The request parser rejects
    // them too, but this class is used from tests and future call sites.
    stops_.erase(std::remove_if(stops_.begin(), stops_.end(),
                                [](const std::string& s) { return s.empty(); }),
                 stops_.end());
}

StopMatcher::Step StopMatcher::push(std::string_view piece, int token_id) {
    Step step;

    if (stops_.empty()) {
        step.emit.assign(piece);
        step.segments.push_back({step.emit, token_id});
        return step;
    }

    buffer_.append(piece);
    held_.emplace_back(piece.size(), token_id);

    // Earliest completed stop sequence wins, so that two overlapping stops
    // truncate at the same place regardless of declaration order.
    size_t cut = std::string::npos;
    for (const std::string& s : stops_) {
        const size_t pos = buffer_.find(s);
        if (pos != std::string::npos && pos < cut) cut = pos;
    }
    if (cut != std::string::npos) {
        step.emit     = buffer_.substr(0, cut);
        step.hit      = true;
        step.segments = take(cut);
        buffer_.clear();
        held_.clear();
        return step;
    }

    // Nothing matched yet — hold back the longest tail that could still grow
    // into a stop sequence.
    size_t hold = 0;
    for (const std::string& s : stops_) {
        hold = std::max(hold, text::partial_stop_suffix(buffer_, s));
    }

    step.emit     = buffer_.substr(0, buffer_.size() - hold);
    step.segments = take(buffer_.size() - hold);
    buffer_.erase(0, buffer_.size() - hold);
    return step;
}

std::vector<StopMatcher::Segment> StopMatcher::take(size_t n) {
    std::vector<Segment> out;
    size_t               at = 0;
    while (n > 0 && !held_.empty()) {
        auto& [len, id] = held_.front();
        if (len <= n) {
            out.push_back({buffer_.substr(at, len), id});
            at += len;
            n -= len;
            held_.erase(held_.begin());
        } else {
            // Part of a piece: the first part has no id, but the id rides on the
            // part that stays held -- the LAST part -- so a tag whose text a
            // stop string splits still reaches its consumer with its id. The
            // id-less first part is ordinary text; the tracker decides tags by
            // the id, never by the text, so nothing leaks.
            out.push_back({buffer_.substr(at, n), -1});
            len -= n;
            n  = 0;
        }
    }
    return out;
}

std::string StopMatcher::flush() {
    std::string out;
    out.swap(buffer_);
    held_.clear();
    return out;
}

std::vector<StopMatcher::Segment> StopMatcher::flush_segments() {
    std::vector<Segment> out = take(buffer_.size());
    buffer_.clear();
    held_.clear();
    return out;
}

}  // namespace lgc
