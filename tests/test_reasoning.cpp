#include "core/reasoning.h"
#include "core/markers.h"

#include "harness.h"

#include <string>
#include <utility>
#include <vector>

using namespace lgc;

namespace {
// Markers come from the shared header; no test spells one out.
std::string qwen_open() { return std::string(markers::kThinkOpen); }
std::string qwen_close() { return std::string(markers::kThinkClose); }
std::string angle_open() { return std::string(markers::kThinkOpenAngle); }
std::string angle_close() { return std::string(markers::kThinkCloseAngle); }
std::string square_open() { return std::string(markers::kThinkOpenSquare); }
std::string square_close() { return std::string(markers::kThinkCloseSquare); }
}  // namespace

TEST(reasoning_split_when_the_template_opened_the_block) {
    const auto s = split_reasoning("The user wants a review.\n" + qwen_close() + "\n\nI'll start by exploring.", true);
    CHECK(s.closed);
    CHECK_EQ(s.reasoning, std::string("The user wants a review."));
    CHECK_EQ(s.content, std::string("I'll start by exploring."));
}

TEST(reasoning_split_empty_thinking_is_just_content) {
    const auto s = split_reasoning(qwen_close() + "\n\nThe total on the invoice is 42.", true);
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
    const auto t = split_reasoning(qwen_open() + "\nhmm\n" + qwen_close() + "\n\nanswer", false);
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
    for (const std::string& p : {qwen_open().substr(0, 4), qwen_open().substr(4) + "\nplan" + qwen_close() + "\n",
                           std::string("go")}) { auto s = self_opened.push(p); r += s.reasoning; c += s.content; }
    CHECK_EQ(r, std::string("\nplan"));
    CHECK_EQ(c, std::string("go"));
}

TEST(reasoning_split_other_think_tags) {
    // Cydonia 24B (Mistral Small 3.2) thinks in the angle-bracket tags, Magistral
    // tunes in the square-bracket ones
    const auto a = split_reasoning(angle_open() + "\nplot first\n" + angle_close() + "\n\nThe storm broke.", false, true);
    CHECK(a.closed);
    CHECK_EQ(a.reasoning, std::string("plot first"));
    CHECK_EQ(a.content, std::string("The storm broke."));
    const auto b = split_reasoning(square_open() + "weigh it" + square_close() + "Answer.", false, true);
    CHECK_EQ(b.reasoning, std::string("weigh it"));
    CHECK_EQ(b.content, std::string("Answer."));
    // a closer of another pair does not close the block
    const auto c = split_reasoning(angle_open() + "a " + qwen_close() + " b" + angle_close() + "c", false, true);
    CHECK_EQ(c.reasoning, "a " + qwen_close() + " b");
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
    // the Qwens keep the bare think tag alone: an answer that merely starts with
    // the other tags stays content (a swallowed answer would lose tool calls)
    const auto a = split_reasoning(angle_open() + "x" + angle_close() + "y", false);
    CHECK(!a.closed);
    CHECK_EQ(a.content, angle_open() + "x" + angle_close() + "y");
    ReasoningStreamer q(false);
    std::string r, c;
    for (const std::string& p : {square_open().substr(0, 4), square_open().substr(4) + "x" + square_close() + "y"}) {
        auto s = q.push(p);
        r += s.reasoning;
        c += s.content;
    }
    const auto t = q.flush(); r += t.reasoning; c += t.content;
    CHECK(r.empty());
    CHECK_EQ(c, square_open() + "x" + square_close() + "y");
}

TEST(reasoning_extended_edges) {
    // the bare / angle-bracket think boundary across pieces
    auto run = [](std::vector<std::string> pieces) {
        ReasoningStreamer st(false, true);
        std::pair<std::string, std::string> rc;
        for (auto p : pieces) { auto s = st.push(p); rc.first += s.reasoning; rc.second += s.content; }
        auto t = st.flush(); rc.first += t.reasoning; rc.second += t.content;
        return rc;
    };
    CHECK(run({angle_open().substr(0, 6), ">x" + qwen_close() + "y"}) == std::make_pair(std::string("x"), std::string("y")));
    CHECK(run({angle_open().substr(0, 6), angle_open().substr(6) + "x" + angle_close() + "y"}) ==
          std::make_pair(std::string("x"), std::string("y")));
    CHECK(run({qwen_open(), "ing>a" + qwen_close() + "b"}) == std::make_pair(std::string("ing>a"), std::string("b")));
    // unclosed: all reasoning
    const auto u = split_reasoning(square_open() + "still going", false, true);
    CHECK(!u.closed);
    CHECK_EQ(u.reasoning, std::string("still going"));
    CHECK(u.content.empty());
    // a pre-opened bare think block is closed only by its own closer
    const auto o = split_reasoning("a" + angle_close() + "b" + square_close() + "c" + qwen_close() + "d", true, true);
    CHECK_EQ(o.reasoning, "a" + angle_close() + "b" + square_close() + "c");
    CHECK_EQ(o.content, std::string("d"));
    // leading whitespace, near misses, case
    CHECK_EQ(split_reasoning("\n " + square_open() + "p" + square_close() + "q", false, true).content, std::string("q"));
    CHECK_EQ(split_reasoning("<thinkpad> is a laptop", false, true).content, std::string("<thinkpad> is a laptop"));
    CHECK_EQ(split_reasoning("[THINKING] no", false, true).content, std::string("[THINKING] no"));
    CHECK_EQ(split_reasoning("[think]x[/think]y", false, true).content, std::string("[think]x[/think]y"));
}

// ---------------------------------------------------------------------------
// clean-turns step 2: the split decides the block by token id too. A closer
// spelled from ordinary tokens is the same text as the real one, so text alone
// cannot tell them apart. Streaming and non-streaming must agree on a sequence
// carrying both: the spelled one stays reasoning, the special-token one ends
// the block, and both paths put the answer in content.

TEST(reasoning_streaming_and_nonstreaming_agree_on_a_spelled_and_a_real_closer) {
    // The Qwen pair, template-opened: its closer is one special token, id 7.
    // The opener has no id here; the template already opened the block.
    const std::string closer = think_tag_pairs(false)[0].second;
    const std::vector<std::pair<int, int>> ids = {{-1, 7}};
    std::vector<std::pair<std::string, int>> pieces;
    pieces.emplace_back("the marker is ", 30);
    pieces.emplace_back(closer.substr(0, 3), 31);
    pieces.emplace_back(closer.substr(3), 32);
    pieces.emplace_back(" quoted, not emitted; ", 33);
    pieces.emplace_back(closer, 7);   // the real closer, one special token
    pieces.emplace_back("answer", 34);

    ReasoningStreamer streamed(true, false, ids);
    std::string       reasoning, content;
    for (const auto& [piece, id] : pieces) {
        const auto step = streamed.push(piece, id);
        reasoning += step.reasoning;
        content += step.content;
    }
    const auto tail = streamed.flush();
    reasoning += tail.reasoning;
    content += tail.content;

    const ReasoningSplit split = split_reasoning_tokens(pieces, true, false, ids);
    CHECK(split.closed);
    CHECK_EQ(split.reasoning, reasoning);   // the two paths are one implementation
    CHECK_EQ(split.content, content);
    CHECK_EQ(content, std::string("answer"));
    // The spelled closer and the text after it stay in reasoning.
    CHECK(split.reasoning.find(closer) != std::string::npos);
    CHECK(split.reasoning.find(" quoted, not emitted; ") != std::string::npos);
}

// The line breaks between the closer and the answer are layout: the
// non-streaming split drops them (the reference), and the stream drops them
// too, so both paths give the same content for the same token stream.
TEST(reasoning_streaming_and_nonstreaming_drop_the_closer_line_breaks) {
    const std::string closer = think_tag_pairs(false)[0].second;
    const std::vector<std::pair<int, int>> ids = {{-1, 7}};
    const std::vector<std::pair<std::string, int>> pieces = {{"the reasoning", 30},
                                                             {closer, 7},
                                                             {"\n\n", 15},
                                                             {"Hello", 16}};

    ReasoningStreamer streamed(true, false, ids);
    std::string       reasoning, content;
    for (const auto& [piece, id] : pieces) {
        const auto step = streamed.push(piece, id);
        reasoning += step.reasoning;
        content += step.content;
    }
    const auto tail = streamed.flush();
    reasoning += tail.reasoning;
    content += tail.content;

    const ReasoningSplit split = split_reasoning_tokens(pieces, true, false, ids);
    CHECK(split.closed);
    CHECK_EQ(content, std::string("Hello"));   // no leading line breaks
    CHECK_EQ(split.content, content);
    CHECK_EQ(split.reasoning, reasoning);
}
