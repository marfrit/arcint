#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace lgc {

// Streaming-correct stop-sequence matching.
//
// A stop string can straddle two emitted pieces. Sending a chunk that ends in
// the first half of a stop sequence leaks it to the client one fragment at a
// time, so any text that could still turn into a stop sequence is withheld
// until the next piece resolves it.
class StopMatcher {
public:
    StopMatcher() = default;
    explicit StopMatcher(std::vector<std::string> stops);

    struct Step {
        std::string emit;       // safe to send now
        bool        hit = false;  // a stop sequence completed; stop generating
    };

    Step push(std::string_view piece);

    // End of generation: release whatever is still held back.
    std::string flush();

    bool empty_config() const { return stops_.empty(); }

private:
    std::vector<std::string> stops_;
    std::string              buffer_;
};

}  // namespace lgc
