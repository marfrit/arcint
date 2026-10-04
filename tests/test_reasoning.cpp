#include "core/reasoning.h"

#include "harness.h"

#include <string>
#include <utility>
#include <vector>

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

TEST(reasoning_split_other_think_tags) {
    // Cydonia 24B (Mistral Small 3.2) thinks in <thinking>, Magistral tunes in [THINK]
    const auto a = split_reasoning("<thinking>\nplot first\n</thinking>\n\nThe storm broke.", false, true);
    CHECK(a.closed);
    CHECK_EQ(a.reasoning, std::string("plot first"));
    CHECK_EQ(a.content, std::string("The storm broke."));
    const auto b = split_reasoning("[THINK]weigh it[/THINK]Answer.", false, true);
    CHECK_EQ(b.reasoning, std::string("weigh it"));
    CHECK_EQ(b.content, std::string("Answer."));
    // a closer of another pair does not close the block
    const auto c = split_reasoning("<thinking>a </think> b</thinking>c", false, true);
    CHECK_EQ(c.reasoning, std::string("a </think> b"));
    CHECK_EQ(c.content, std::string("c"));
}

TEST(reasoning_streamer_other_think_tags) {
    ReasoningStreamer st(false, true);
    std::string r, c;
    for (std::string_view p : {"<thi", "nking>\nplot", " first</think", "ing>\n\nThe storm", " broke."}) {
        auto s = st.push(p); r += s.reasoning; c += s.content;
    }
    const auto t = st.flush(); r += t.reasoning; c += t.content;
    CHECK_EQ(r, std::string("\nplot first"));
    CHECK_EQ(c, std::string("The storm broke."));
    ReasoningStreamer m(false, true);
    std::string r2, c2;
    for (std::string_view p : {"[TH", "INK]x[/TH", "INK]y"}) { auto s = m.push(p); r2 += s.reasoning; c2 += s.content; }
    CHECK_EQ(r2, std::string("x"));
    CHECK_EQ(c2, std::string("y"));
}

TEST(reasoning_other_tags_only_when_extended) {
    // the Qwens keep <think> alone: an answer that merely starts with the
    // other tags stays content (a swallowed answer would lose tool calls)
    const auto a = split_reasoning("<thinking>x</thinking>y", false);
    CHECK(!a.closed);
    CHECK_EQ(a.content, std::string("<thinking>x</thinking>y"));
    ReasoningStreamer q(false);
    std::string r, c;
    for (std::string_view p : {"[THI", "NK]x[/THINK]y"}) { auto s = q.push(p); r += s.reasoning; c += s.content; }
    const auto t = q.flush(); r += t.reasoning; c += t.content;
    CHECK(r.empty());
    CHECK_EQ(c, std::string("[THINK]x[/THINK]y"));
}

TEST(reasoning_extended_edges) {
    // the <think> / <thinking> boundary across pieces
    auto run = [](std::vector<std::string_view> pieces) {
        ReasoningStreamer st(false, true);
        std::pair<std::string, std::string> rc;
        for (auto p : pieces) { auto s = st.push(p); rc.first += s.reasoning; rc.second += s.content; }
        auto t = st.flush(); rc.first += t.reasoning; rc.second += t.content;
        return rc;
    };
    CHECK(run({"<think", ">x</think>y"}) == std::make_pair(std::string("x"), std::string("y")));
    CHECK(run({"<think", "ing>x</thinking>y"}) == std::make_pair(std::string("x"), std::string("y")));
    CHECK(run({"<think>", "ing>a</think>b"}) == std::make_pair(std::string("ing>a"), std::string("b")));
    // unclosed: all reasoning
    const auto u = split_reasoning("[THINK]still going", false, true);
    CHECK(!u.closed);
    CHECK_EQ(u.reasoning, std::string("still going"));
    CHECK(u.content.empty());
    // a pre-opened <think> is closed only by </think>
    const auto o = split_reasoning("a</thinking>b[/THINK]c</think>d", true, true);
    CHECK_EQ(o.reasoning, std::string("a</thinking>b[/THINK]c"));
    CHECK_EQ(o.content, std::string("d"));
    // leading whitespace, near misses, case
    CHECK_EQ(split_reasoning("\n [THINK]p[/THINK]q", false, true).content, std::string("q"));
    CHECK_EQ(split_reasoning("<thinkpad> is a laptop", false, true).content, std::string("<thinkpad> is a laptop"));
    CHECK_EQ(split_reasoning("[THINKING] no", false, true).content, std::string("[THINKING] no"));
    CHECK_EQ(split_reasoning("[think]x[/think]y", false, true).content, std::string("[think]x[/think]y"));
}
