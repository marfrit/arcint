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
