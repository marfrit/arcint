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

    // A part of the emitted text that is one generated piece, whole, with
    // that piece's token id, or part of one piece with id -1. When a stop
    // string splits a piece, the released first part has no id and the id
    // rides on the LAST part: a consumer that decides by id (the think block's
    // closer) sees the same tag id the backend produced, with or without
    // stops, even when the tag's text is cut by a held stop tail.
    struct Segment {
        std::string text;
        int         id = -1;
    };

    struct Step {
        std::string          emit;       // safe to send now
        bool                 hit = false;  // a stop sequence completed; stop generating
        std::vector<Segment> segments;   // `emit`, cut at the pieces it came from
    };

    Step push(std::string_view piece, int token_id = -1);

    // End of generation: release whatever is still held back.
    std::string flush();
    // The same, cut at the held pieces.
    std::vector<Segment> flush_segments();

    bool empty_config() const { return stops_.empty(); }

private:
    // The first `n` bytes of buffer_, cut at the held pieces; consumes them
    // from held_ (the caller trims buffer_).
    std::vector<Segment> take(size_t n);

    std::vector<std::string> stops_;
    std::string              buffer_;
    // buffer_'s bytes by the piece they came from, oldest first: {bytes, id};
    // a piece split by a hold loses its id on the released part and keeps it
    // on what remains held.
    std::vector<std::pair<size_t, int>> held_;
};

}  // namespace lgc
