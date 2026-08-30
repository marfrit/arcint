#include "core/reasoning.h"

#include "harness.h"

using namespace lgc;

TEST(reasoning_split_when_the_template_opened_the_block) {
    const auto s = split_reasoning("The user wants a review.\n</think>\n\nI'll start by exploring.", true);
    CHECK(s.closed);
    CHECK_EQ(s.reasoning, std::string("The user wants a review."));
    CHECK_EQ(s.content, std::string("I'll start by exploring."));
}

TEST(reasoning_split_empty_thinking_is_just_content) {
    const auto s = split_reasoning("</think>\n\nThe total on the invoice is 42.", true);
    CHECK(s.closed);
    CHECK(s.reasoning.empty());
    CHECK_EQ(s.content, std::string("The total on the invoice is 42."));
}

TEST(reasoning_split_never_closed_is_all_reasoning) {
    const auto s = split_reasoning("Let me think about this for a", true);
    CHECK(!s.closed);
    CHECK_EQ(s.reasoning, std::string("Let me think about this for a"));
    CHECK(s.content.empty());
}

TEST(reasoning_split_not_opened_leaves_content_alone) {
    const auto s = split_reasoning("Plain answer with no tags.", false);
    CHECK(!s.closed);
    CHECK(s.reasoning.empty());
    CHECK_EQ(s.content, std::string("Plain answer with no tags."));
    // ...unless the model opened one itself.
    const auto t = split_reasoning("<think>\nhmm\n</think>\n\nanswer", false);
    CHECK_EQ(t.reasoning, std::string("hmm"));
    CHECK_EQ(t.content, std::string("answer"));
}

TEST(reasoning_streamer_holds_back_a_straddling_close_tag) {
    ReasoningStreamer st(true);
    std::string reasoning, content;
    for (std::string_view piece : {"The user", " wants a review.</thi", "nk>\n\nI'll start", " by exploring."}) {
        const auto step = st.push(piece);
        reasoning += step.reasoning;
        content += step.content;
    }
    const auto tail = st.flush();
    reasoning += tail.reasoning;
    content += tail.content;
    CHECK_EQ(reasoning, std::string("The user wants a review."));
    CHECK_EQ(content, std::string("I'll start by exploring."));
    CHECK(!st.in_reasoning());
}

TEST(reasoning_streamer_undecided_start_resolves_both_ways) {
    ReasoningStreamer plain(false);
    auto a = plain.push("Hel");
    CHECK(a.content == "Hel" && a.reasoning.empty());
    ReasoningStreamer self_opened(false);
    std::string r, c;
    for (std::string_view p : {"<thi", "nk>\nplan</think>\n", "go"}) { auto s = self_opened.push(p); r += s.reasoning; c += s.content; }
    CHECK_EQ(r, std::string("\nplan"));
    CHECK_EQ(c, std::string("go"));
}
