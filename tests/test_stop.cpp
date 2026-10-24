#include "core/stop.h"
#include "harness.h"

using namespace lgc;

TEST(stop_passthrough_without_sequences) {
    StopMatcher m;
    const auto  step = m.push("anything at all");
    CHECK_EQ(step.emit, std::string("anything at all"));
    CHECK(!step.hit);
    CHECK_EQ(m.flush(), std::string(""));
}

TEST(stop_truncates_at_sequence) {
    StopMatcher m({"STOP"});
    const auto  step = m.push("before STOP after");
    CHECK_EQ(step.emit, std::string("before "));
    CHECK(step.hit);
}

TEST(stop_holds_back_partial_match_across_pieces) {
    StopMatcher m({"<end>"});

    auto a = m.push("hello <en");
    CHECK_EQ(a.emit, std::string("hello "));  // "<en" withheld
    CHECK(!a.hit);

    auto b = m.push("d> tail");
    CHECK_EQ(b.emit, std::string(""));
    CHECK(b.hit);
}

TEST(stop_releases_partial_that_never_completes) {
    StopMatcher m({"<end>"});

    auto a = m.push("hello <en");
    CHECK_EQ(a.emit, std::string("hello "));
    // Generation stopped for another reason: the held-back text is ordinary
    // output and must reach the client.
    CHECK_EQ(m.flush(), std::string("<en"));
}

TEST(stop_earliest_match_wins) {
    StopMatcher m({"world", "lo w"});
    const auto  step = m.push("hello world");
    CHECK_EQ(step.emit, std::string("hel"));
    CHECK(step.hit);
}

TEST(stop_byte_by_byte_matches_whole_string) {
    // The property that matters for SSE: feeding one byte at a time must give
    // the same visible output as feeding the whole string at once.
    const std::string text = "abc<end>def";

    StopMatcher whole({"<end>"});
    const std::string expected = whole.push(text).emit;

    StopMatcher drip({"<end>"});
    std::string got;
    bool        hit = false;
    for (char c : text) {
        auto step = drip.push(std::string_view(&c, 1));
        got += step.emit;
        if (step.hit) {
            hit = true;
            break;
        }
    }
    CHECK(hit);
    CHECK_EQ(got, expected);
    CHECK_EQ(got, std::string("abc"));
}

TEST(stop_empty_sequences_are_ignored) {
    StopMatcher m({"", "x"});
    const auto  step = m.push("aaa");
    CHECK_EQ(step.emit, std::string("aaa"));
    CHECK(!step.hit);
}

// The emitted text is cut at the pieces it came from: a whole piece keeps its
// token id, a part of one carries -1. A held tail released with a later piece
// keeps its own id rather than losing it or lending it to its neighbour.
TEST(stop_segments_keep_each_whole_piece_id) {
    StopMatcher m({"\nUser:"});

    auto a = m.push("\n", 14);   // could start the stop: held
    CHECK_EQ(a.emit, std::string(""));
    CHECK_EQ(a.segments.size(), size_t{0});

    auto b = m.push("X", 7);     // resolves the hold: both pieces, each whole
    CHECK_EQ(b.emit, std::string("\nX"));
    CHECK_EQ(b.segments.size(), size_t{2});
    CHECK_EQ(b.segments[0].text, std::string("\n"));
    CHECK_EQ(b.segments[0].id, 14);
    CHECK_EQ(b.segments[1].text, std::string("X"));
    CHECK_EQ(b.segments[1].id, 7);
}

TEST(stop_segments_mark_a_split_piece_with_no_id) {
    StopMatcher m({"\nUser:"});

    auto a = m.push("ab\n", 3);  // "ab" released, "\n" held: both parts lose the id
    CHECK_EQ(a.segments.size(), size_t{1});
    CHECK_EQ(a.segments[0].text, std::string("ab"));
    CHECK_EQ(a.segments[0].id, -1);

    const auto tail = m.flush_segments();
    CHECK_EQ(tail.size(), size_t{1});
    CHECK_EQ(tail[0].text, std::string("\n"));
    CHECK_EQ(tail[0].id, -1);
}

TEST(stop_segments_flush_whole_held_pieces_with_their_ids) {
    StopMatcher m({"<end>"});
    m.push("<", 5);
    m.push("en", 6);
    const auto tail = m.flush_segments();
    CHECK_EQ(tail.size(), size_t{2});
    CHECK_EQ(tail[0].id, 5);
    CHECK_EQ(tail[1].id, 6);
}

TEST(stop_segments_before_a_hit_keep_their_ids) {
    StopMatcher m({"STOP"});
    m.push("S", 1);
    auto b = m.push("kip", 2);   // the hold resolves as text
    CHECK_EQ(b.segments.size(), size_t{2});
    auto c = m.push("aSTOPz", 3);   // the hit cuts inside this piece
    CHECK(c.hit);
    CHECK_EQ(c.emit, std::string("a"));
    CHECK_EQ(c.segments.size(), size_t{1});
    CHECK_EQ(c.segments[0].id, -1);
}

// A hit that cuts inside a piece held from an earlier step: the part before
// the cut goes out without an id, and nothing stays held.
TEST(stop_segments_hit_inside_a_held_piece) {
    StopMatcher m({"abSTOP"});
    auto a = m.push("xab", 4);   // "ab" held, "x" released without the id
    CHECK_EQ(a.segments.size(), size_t{1});
    CHECK_EQ(a.segments[0].text, std::string("x"));
    CHECK_EQ(a.segments[0].id, -1);
    auto b = m.push("STOP", 5);
    CHECK(b.hit);
    CHECK_EQ(b.emit, std::string(""));
    CHECK_EQ(b.segments.size(), size_t{0});
    CHECK_EQ(m.flush_segments().size(), size_t{0});
}
