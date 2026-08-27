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

StopMatcher::Step StopMatcher::push(std::string_view piece) {
    Step step;

    if (stops_.empty()) {
        step.emit.assign(piece);
        return step;
    }

    buffer_.append(piece);

    // Earliest completed stop sequence wins, so that two overlapping stops
    // truncate at the same place regardless of declaration order.
    size_t cut = std::string::npos;
    for (const std::string& s : stops_) {
        const size_t pos = buffer_.find(s);
        if (pos != std::string::npos && pos < cut) cut = pos;
    }
    if (cut != std::string::npos) {
        step.emit = buffer_.substr(0, cut);
        step.hit  = true;
        buffer_.clear();
        return step;
    }

    // Nothing matched yet — hold back the longest tail that could still grow
    // into a stop sequence.
    size_t hold = 0;
    for (const std::string& s : stops_) {
        hold = std::max(hold, text::partial_stop_suffix(buffer_, s));
    }

    step.emit = buffer_.substr(0, buffer_.size() - hold);
    buffer_.erase(0, buffer_.size() - hold);
    return step;
}

std::string StopMatcher::flush() {
    std::string out;
    out.swap(buffer_);
    return out;
}

}  // namespace lgc
