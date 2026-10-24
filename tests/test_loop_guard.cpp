// loop-guard: the reasoning-loop detector and the escalation ladder
// (nudge, slap, abort from a reasoning loop). Synthetic streams and rows; no
// card, no tokenizer, no chat-template markers (the tag strings come from the
// shared table in core/reasoning.h, the closer is forced by id).
#include "api/handlers.h"
#include "config.h"
#include "core/chat.h"
#include "core/reasoning.h"
#include "core/sampler.h"
#include "exec/backend.h"
#include "exec/verify_walk.h"
#include "harness.h"

#include <string>

using namespace lgc;

namespace {

SamplerParams base_params(LoopGuardMode mode, int window = 10, int grace = 5) {
    SamplerParams p;
    p.temperature        = 0.0f;
    p.top_k              = 0;
    p.top_p              = 1.0f;
    p.repetition_penalty = 1.0f;
    p.loop_guard          = mode;
    p.loop_guard_window   = window;
    p.loop_guard_threshold = 0.8f;
    p.loop_guard_grace    = grace;
    // A vocabulary where the closer is one special token: the slap forces it
    // by id. The opener has no id (the template opened the block).
    const auto& pairs = think_tag_pairs(false);
    p.think_ids.eog        = {0};
    p.think_ids.close_ids  = {5};
    p.think_ids.close_tags = {pairs[0].second};
    p.think_ids.open_tags  = {pairs[0].first};
    return p;
}

// A line that is long enough to count, with enough varied (and unique) tokens
// that the near rule's 3-gram Jaccard stays well below its bar for a
// non-looping stream.
std::string distinct_token(int i, int salt) {
    uint32_t x = static_cast<uint32_t>((i + 1) * 2654435761u + salt * 40503u);
    std::string out;
    for (int k = 0; k < 4; ++k) {
        out += static_cast<char>('a' + x % 26);
        x /= 26;
    }
    return out;
}

std::string distinct_line(int i) {
    const std::string a = distinct_token(i, 1), b = distinct_token(i, 2),
                      c = distinct_token(i, 3), d = distinct_token(i, 4),
                      e = distinct_token(i, 5);
    switch (i % 4) {
        case 0: return "step " + std::to_string(i) + ": " + a + " " + b + " " + c + " " + d + " " + e + "\n";
        case 1: return "under " + std::to_string(i) + ", " + b + " asks " + c + " or " + d + " fits " + e + " beside " + a + "\n";
        case 2: return "mark " + std::to_string(i) + ": choose " + c + ", drop " + d + ", keep " + a + " next to " + e + "\n";
        default: return "branch " + std::to_string(i) + " reads " + d + " against " + e + " while " + a + " waits for " + b + "\n";
    }
}

// A counting loop as seen live: one fixed line, one line whose only change is
// a number word. Exact repeats stay at about half; near catches both.
std::string counting_fixed() {
    return "Wait, the index remainder is still the same value and the same step.\n";
}
std::string counting_varying(int i) {
    static const char* nums[] = {"four", "five", "six", "seven", "eight", "nine", "ten",
                                 "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                                 "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                                 "twenty-one", "twenty-two", "twenty-three", "twenty-four"};
    return "If the index remainder is one thousand and " + std::string(nums[i % 20]) +
           " after " + std::to_string(i) + " steps, still the same question?\n";
}

std::string loop_line(int) {
    return "reasoning line: the same plan again and again and again, unchanged\n";
}

void feed_text(Sampler& s, const std::string& text) { s.observe_think_text(text); }

void feed_lines(Sampler& s, int from, int to, bool loop) {
    for (int i = from; i < to; ++i) feed_text(s, loop ? loop_line(i) : distinct_line(i));
}

std::vector<float> row_with_argmax(int id, size_t vocab = 8) {
    std::vector<float> r(vocab, -1.0f);
    r[static_cast<size_t>(id)] = 5.0f;
    return r;
}

// The nudge sentence's ids stand in for the real tokenizer's: each id maps to
// a word piece, so the test can assemble the reasoning text the stream would
// show and count the sentence's presence.
constexpr int kPieceFirst = 101;
const std::vector<std::string> kPiece = {"I", " am", " repeating", " myself.", " I", " will",
                                        " decide", " now", " and", " write", " the", " final",
                                        " answer."};

std::string piece_of(int id) {
    const size_t i = static_cast<size_t>(id - kPieceFirst);
    return i < kPiece.size() ? kPiece[i] : std::string{};
}

}  // namespace

// 1. Detection: a synthetic looping reasoning stream trips at the expected
// line, not before. Window 10, threshold 0.8: nine distinct lines keep the
// share at 0; after that the share rises 1/10, 2/10 ... and first reaches 0.8
// when the eighth repeating line arrives (line 17, counting from 1).
TEST(loop_guard_detector_fires_at_the_expected_line) {
    // ladder-nodoor: the ladder's old order, so the nudge is the first rung at
    // detection (the door ladder defers it).
    SamplerParams p = base_params(LoopGuardMode::LadderNoDoor);
    p.loop_guard_nudge_ids = {kPieceFirst};   // the ladder's first rung needs them
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    CHECK_EQ(s.loop_guard_counts().detections, 0);

    for (int i = 9; i < 17; ++i) {   // the 7th repeating line is still 0.7
        feed_text(s, loop_line(i));
        CHECK_EQ(s.loop_guard_counts().detections, 0);
    }
    feed_text(s, loop_line(17));   // the 8th: 8 of 10
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    CHECK_EQ(s.loop_guard_counts().nudges, 1);   // ladder: the first rung
}

// 2. The false-positive guard: a non-looping stream of the same length with
// normal variety never fires, in every mode.
TEST(loop_guard_varied_stream_never_fires) {
    for (LoopGuardMode mode : {LoopGuardMode::Detect, LoopGuardMode::Nudge,
                               LoopGuardMode::Slap, LoopGuardMode::Ladder}) {
        Sampler s(base_params(mode), 1);
        s.start_in_think_block(true);
        feed_lines(s, 0, 60, false);
        CHECK_EQ(s.loop_guard_counts().detections, 0);
        CHECK_EQ(s.loop_guard_counts().nudges, 0);
        CHECK_EQ(s.loop_guard_counts().slaps, 0);
        CHECK_EQ(s.loop_guard_counts().aborts, 0);
    }
}

// The near rule catches the counting loop that exact misses.
TEST(loop_guard_near_catches_a_counting_loop_that_exact_misses) {
    auto detections = [&](LoopGuardMatch match) {
        SamplerParams p = base_params(LoopGuardMode::Detect);
        p.loop_guard_match = match;
        Sampler s(p, 1);
        s.start_in_think_block(true);
        for (int i = 0; i < 40; ++i) {
            feed_text(s, i % 2 == 0 ? counting_fixed() : counting_varying(i));
        }
        return s.loop_guard_counts().detections;
    };
    CHECK_EQ(detections(LoopGuardMatch::Exact), 0);
    CHECK_EQ(detections(LoopGuardMatch::Near), 1);
}

// The false-positive guard under near: a genuinely varied stream of the same
// length never fires.
TEST(loop_guard_near_varied_stream_never_fires) {
    SamplerParams p = base_params(LoopGuardMode::Detect);
    p.loop_guard_match = LoopGuardMatch::Near;
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 60, false);
    CHECK_EQ(s.loop_guard_counts().detections, 0);
}

// Hysteresis: a dip just under the threshold after a nudge does not count a
// second detection; a real fall below threshold - 0.1 and a rise again does.
// Window 20, threshold 0.8: a repeated line stream detects at 0.8, five varied
// lines dip it to 0.75 (above the 0.7 low mark), and 16 repeats cross back to
// 0.8 without counting; seven varied lines fall to 0.65 (below the low mark),
// and 16 repeats cross back and count.
TEST(loop_guard_hysteresis_dip_does_not_add_a_detection) {
    SamplerParams p = base_params(LoopGuardMode::LadderNoDoor, 20, 1000);
    p.loop_guard_match = LoopGuardMatch::Exact;
    for (size_t i = 0; i < kPiece.size(); ++i) p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 26, true);   // 17 repeating lines: the share first reaches 0.8
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    // Deliver the nudge, then dip just under the threshold.
    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        s.observe_think(id, piece_of(id));
    }
    feed_lines(s, 9, 14, false);   // 5 varied lines: share 0.75
    feed_lines(s, 26, 42, true);   // 16 repeats: crosses back to 0.8
    CHECK_EQ(s.loop_guard_counts().detections, 1);   // the dip did not count

    // A real fall below threshold - 0.1, then a rise again: a new detection.
    feed_lines(s, 14, 21, false);  // 7 varied lines: share 0.65
    feed_lines(s, 42, 58, true);   // 16 repeats: crosses 0.8 again
    CHECK_EQ(s.loop_guard_counts().detections, 2);
}

// Lines are judged after leading/trailing ASCII whitespace is stripped, the
// same text the offline measurement used. Indentation alone is not a repeat;
// an indented line whose stripped text is <= 20 characters is not counted.
TEST(loop_guard_strips_whitespace_before_the_bar_and_the_comparison) {
    // (a) the same long line with and without leading indentation is an exact
    // repeat: after nine varied lines, eight alternations reach 0.8.
    {
        SamplerParams p = base_params(LoopGuardMode::Detect);
        p.loop_guard_match = LoopGuardMatch::Exact;
        Sampler s(p, 1);
        s.start_in_think_block(true);
        feed_lines(s, 0, 9, false);
        const std::string body = "the index remainder is still the same value now";
        for (int i = 9; i < 18; ++i)
            feed_text(s, (i % 2 == 0 ? std::string(8, ' ') : std::string{}) + body + "\n");
        CHECK_EQ(s.loop_guard_counts().detections, 1);
    }
    // (b) fifteen spaces + a ten-character comment is under the bar once
    // stripped, so ten repetitions are not counted and nothing fires.
    {
        SamplerParams p = base_params(LoopGuardMode::Detect);
        p.loop_guard_match = LoopGuardMatch::Exact;
        Sampler s(p, 1);
        s.start_in_think_block(true);
        feed_lines(s, 0, 9, false);
        for (int i = 0; i < 10; ++i)
            feed_text(s, std::string(15, ' ') + "# Correct." + "\n");
        CHECK_EQ(s.loop_guard_counts().detections, 0);
    }
}

// 3. Ladder: the nudge text appears once in the reasoning, and after K more
// looping lines the next drawn token is the closer id. Afterwards content flows.
TEST(loop_guard_ladder_nudges_once_then_forces_the_closer) {
    // ladder-nodoor: today's ladder order, unchanged by the door rung.
    SamplerParams p = base_params(LoopGuardMode::LadderNoDoor, 10, 5);
    for (size_t i = 0; i < kPiece.size(); ++i) p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    std::string reasoning;
    auto draw = [&] {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        reasoning += piece_of(id);
        // observe the forced token's text and id, exactly as a decode loop does
        s.observe_think(id, piece_of(id));
        return id;
    };

    feed_lines(s, 0, 9, false);   // nine varied lines: share 0
    feed_lines(s, 9, 18, true);   // nine repeating lines: the 8th reaches 0.8
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    // The nudge tokenises to the sentence: one id a draw.
    size_t sentence_count = 0;
    for (size_t i = 0; i < kPiece.size(); ++i) {
        const int id = draw();
        CHECK_EQ(id, kPieceFirst + static_cast<int>(i));
    }
    // "repeating myself" occurs once in the reasoning text, never twice.
    for (size_t at = reasoning.find("repeating"); at != std::string::npos;
         at = reasoning.find("repeating", at + 1))
        ++sentence_count;
    CHECK_EQ(sentence_count, 1u);
    CHECK_EQ(s.loop_guard_counts().nudges, 1);

    // Keep looping for exactly K more counted lines; the share stays over the
    // threshold and the next draw is the closer id.
    feed_lines(s, 18, 23, true);   // 5 lines == grace
    CHECK_EQ(s.loop_guard_counts().slaps, 1);
    std::vector<float> l = row_with_argmax(2);
    CHECK_EQ(s.sample(l.data(), l.size()), 5);

    // The closer has closed the block (by id); content now flows normally.
    s.observe(5, "closed");
    std::vector<float> l2 = row_with_argmax(3);
    CHECK_EQ(s.sample(l2.data(), l2.size()), 3);
}

// Grace: the ladder waits exactly K counted lines after the nudge. K-1 lines
// keep watching (no slap); the Kth trips the slap and the next draw is closer.
TEST(loop_guard_ladder_waits_the_full_grace_before_the_slap) {
    SamplerParams p = base_params(LoopGuardMode::LadderNoDoor, 10, 40);
    for (size_t i = 0; i < kPiece.size(); ++i) p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // the nudge rung fires
    CHECK_EQ(s.loop_guard_counts().nudges, 1);

    // Deliver the nudge sentence, one id a draw, observing each piece.
    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        CHECK_EQ(id, kPieceFirst + static_cast<int>(i));
        s.observe_think(id, piece_of(id));
    }

    // K-1 = 39 counted lines after the nudge: still no slap.
    feed_lines(s, 18, 57, true);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);

    // The Kth line (40): the slap is armed and the next drawn token is closer.
    feed_text(s, loop_line(57));
    CHECK_EQ(s.loop_guard_counts().slaps, 1);
    std::vector<float> l = row_with_argmax(2);
    CHECK_EQ(s.sample(l.data(), l.size()), 5);
}

// The answer phase is watched only after a slap. In detect and nudge modes the
// block closes naturally, and a looping content phase must not abort: no
// counters move, and the turn does not finish "loop".
TEST(loop_guard_answer_phase_unwatched_without_a_slap) {
    for (LoopGuardMode mode : {LoopGuardMode::Detect, LoopGuardMode::Nudge}) {
        SamplerParams p = base_params(mode, 10, 5);
        for (size_t i = 0; i < kPiece.size(); ++i)
            p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
        Sampler s(p, 1);
        s.start_in_think_block(true);

        feed_lines(s, 0, 9, false);
        feed_lines(s, 9, 18, true);   // a reasoning loop, detected in either mode
        if (mode == LoopGuardMode::Nudge) {
            for (size_t i = 0; i < kPiece.size(); ++i) {
                std::vector<float> l = row_with_argmax(2);
                const int id = s.sample(l.data(), l.size());
                s.observe_think(id, piece_of(id));
            }
        }
        CHECK_EQ(s.loop_guard_counts().slaps, 0);
        s.observe(5, "closed");   // the block closes on its own, no slap

        // A looping answer after a natural close is not watched: no abort.
        feed_lines(s, 0, 9, false);
        feed_lines(s, 9, 18, true);
        CHECK(!s.loop_abort());
        CHECK_EQ(s.loop_guard_counts().aborts, 0);

        std::vector<float> rows = row_with_argmax(3);
        const VerifyWalk   w = walk_verify(
            rows.data(), 1, rows.size(), {}, s, 100, [](int) { return false; },
            [&](int) { return Control::Continue; });
        CHECK(w.reason != FinishReason::Loop);
        CHECK_EQ(std::string(finish_reason_name(w.reason)), std::string("stop"));
    }
}

// 4. Abort: a looping answer phase after a slap ends the turn with a distinct
// finish reason.
TEST(loop_guard_looping_content_after_a_slap_ends_the_turn) {
    SamplerParams p = base_params(LoopGuardMode::LadderNoDoor, 10, 2);
    for (size_t i = 0; i < kPiece.size(); ++i) p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // reasoning loop -> nudge
    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        s.observe_think(id, piece_of(id));
    }
    feed_lines(s, 18, 20, true);   // grace 2 -> slap pending
    std::vector<float> l = row_with_argmax(2);
    CHECK_EQ(s.sample(l.data(), l.size()), 5);
    s.observe(5, "closed");   // the block is closed: content phase, after a slap

    // The reasoning window must not abort the answer's first lines: the slap
    // starts a fresh content window, so one varied line is not a loop.
    feed_text(s, distinct_line(0));
    CHECK(!s.loop_abort());
    CHECK_EQ(s.loop_guard_counts().aborts, 0);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // the answer repeats itself too
    CHECK(s.loop_abort());
    CHECK_EQ(s.loop_guard_counts().aborts, 1);

    // The decode loop reads the latch and ends on FinishReason::Loop, which
    // names itself "loop".
    std::vector<float> rows = row_with_argmax(3);
    int               emitted = 0;
    const VerifyWalk   w = walk_verify(
        rows.data(), 1, rows.size(), {}, s, 100, [](int) { return false; },
        [&](int) {
            ++emitted;
            return Control::Continue;
        });
    CHECK(w.done);
    CHECK(w.reason == FinishReason::Loop);
    CHECK_EQ(std::string(finish_reason_name(w.reason)), std::string("loop"));
}

// 5. Off / no flag: today's behaviour exactly. A looping stream is not watched,
// nothing is forced, and the draw is the plain argmax.
TEST(loop_guard_off_watches_nothing_and_forces_nothing) {
    SamplerParams p = base_params(LoopGuardMode::Off);
    p.loop_guard_nudge_ids = {kPieceFirst};   // set, but off must ignore it
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 40, true);
    CHECK_EQ(s.loop_guard_counts().detections, 0);
    CHECK_EQ(s.loop_guard_counts().nudges, 0);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);
    CHECK_EQ(s.loop_guard_counts().aborts, 0);
    CHECK(!s.loop_abort());
    std::vector<float> l = row_with_argmax(3);
    CHECK_EQ(s.sample(l.data(), l.size()), 3);
}

// 6. A request field overrides the operator's flag for that request only.
TEST(loop_guard_request_field_overrides_the_flag) {
    nlohmann::json body = {{"model", "m"},
                           {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                           {"loop_guard", "detect"}};
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.sampler.loop_guard.has_value());
    CHECK(*req.sampler.loop_guard == LoopGuardMode::Detect);

    // Applied to a sampler that started at the operator's ladder rung, only the
    // request's field moves.
    SamplerParams p = base_params(LoopGuardMode::Ladder);
    CHECK(!sampler_apply(p, req.sampler).has_value());
    CHECK(p.loop_guard == LoopGuardMode::Detect);
    CHECK_EQ(p.loop_guard_window, 10);   // the operator thresholds stay

    // Unknown names are refused, not silently ignored.
    body["loop_guard"] = "sometimes";
    ChatRequest bad;
    CHECK(parse_chat_request(body, bad).has_value());
}

// 7. Counters reach the stats struct's consumers: the decode line suffix and
// the response's loop_guard object.
TEST(loop_guard_counters_reach_the_line_and_the_response_object) {
    GenerationStats s;
    s.loop_detections = 2;
    s.loop_nudges     = 1;
    s.loop_slaps      = 1;
    s.loop_aborts     = 1;
    const nlohmann::json j = api::loop_guard_json(s);
    CHECK_EQ(j.at("detections").get<int>(), 2);
    CHECK_EQ(j.at("nudges").get<int>(), 1);
    CHECK_EQ(j.at("slaps").get<int>(), 1);
    CHECK_EQ(j.at("aborts").get<int>(), 1);
    const std::string line = api::decode_suffix_for(s);
    CHECK(line.find("loop guard") != std::string::npos);
    CHECK(line.find("detected 2") != std::string::npos);
    CHECK(line.find("slapped 1") != std::string::npos);
}

// The operator switch parses, validates and travels into the sampler defaults.
TEST(loop_guard_flags_parse_and_travel) {
    Config cfg;
    std::vector<const char*> args = {"--stub", "--loop-guard", "slap",
                                     "--loop-guard-match", "exact",
                                     "--loop-guard-window", "64",
                                     "--loop-guard-threshold", "0.6",
                                     "--loop-guard-grace", "12"};
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("arcint"));
    for (const char* a : args) argv.push_back(const_cast<char*>(a));
    CHECK(parse_args(static_cast<int>(argv.size()), argv.data(), cfg).ok);
    CHECK_EQ(cfg.loop_guard, std::string("slap"));

    SamplerDefaults d;
    CHECK(!apply_operator_defaults(cfg, d).has_value());
    CHECK(d.loop_guard == LoopGuardMode::Slap);
    CHECK(d.loop_guard_match == LoopGuardMatch::Exact);
    CHECK_EQ(d.loop_guard_window, 64);
    CHECK_EQ(d.loop_guard_grace, 12);
    CHECK(d.provenance == "operator");

    // Off stays today's defaults and does not mark them operator.
    Config stub;
    std::vector<const char*> offargs = {"--stub"};
    std::vector<char*> offargv;
    offargv.push_back(const_cast<char*>("arcint"));
    for (const char* a : offargs) offargv.push_back(const_cast<char*>(a));
    CHECK(parse_args(static_cast<int>(offargv.size()), offargv.data(), stub).ok);
    SamplerDefaults d2;
    CHECK(!apply_operator_defaults(stub, d2).has_value());
    CHECK(d2.loop_guard == LoopGuardMode::Off);
    CHECK(d2.loop_guard_match == LoopGuardMatch::Near);   // the default rule
    CHECK_EQ(d2.loop_guard_window, 200);
    CHECK(d2.provenance == "provisional");

    // Bad values are refused at boot.
    for (std::vector<std::string> bad : {std::vector<std::string>{"--stub", "--loop-guard", "sometimes"},
                                         std::vector<std::string>{"--stub", "--loop-guard-match", "loose"},
                                         std::vector<std::string>{"--stub", "--loop-guard-window", "0"},
                                         std::vector<std::string>{"--stub", "--loop-guard-threshold", "1.5"},
                                         std::vector<std::string>{"--stub", "--loop-guard-grace", "-1"}}) {
        std::vector<const char*> ba;
        for (const std::string& b : bad) ba.push_back(b.c_str());
        std::vector<char*> bav;
        bav.push_back(const_cast<char*>("arcint"));
        for (const char* a : ba) bav.push_back(const_cast<char*>(a));
        Config c;
        CHECK(!parse_args(static_cast<int>(bav.size()), bav.data(), c).ok);
    }
}

// The nudge sentence is not reasoning the detector holds: its last piece
// must not prefix the model's next line. A 20-character line after the nudge
// stays below the counting bar (lines longer than 20 count); with the
// sentence's tail glued on it would count, and the 40th counted line would
// arrive one line early.
TEST(loop_guard_nudge_tail_does_not_prefix_the_next_line) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 40);
    for (size_t i = 0; i < kPiece.size(); ++i) p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);
    CHECK_EQ(s.loop_guard_counts().nudges, 1);
    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        s.observe_think(id, piece_of(id));
    }

    feed_text(s, "abcdefghij klmnopqrs\n");   // 20 characters: not counted
    feed_lines(s, 18, 57, true);              // 39 counted lines
    CHECK_EQ(s.loop_guard_counts().slaps, 0);
    feed_text(s, loop_line(57));              // the 40th
    CHECK_EQ(s.loop_guard_counts().slaps, 1);
}

// A forced token in the middle of a speculative span. The walk draws row by
// row; the slap's closer differs from the draft at that row, so the walk
// emits it and stops there: the drafts behind it are never accepted.
TEST(loop_guard_slap_breaks_the_verify_walk_at_its_row) {
    SamplerParams p = base_params(LoopGuardMode::Slap, 10, 2);
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detected; slap mode waits the grace
    feed_lines(s, 18, 20, true);  // grace 2: the slap is pending

    const size_t       vocab = 8;
    std::vector<float> rows;
    for (int r = 0; r < 3; ++r) {
        const std::vector<float> row = row_with_argmax(2, vocab);
        rows.insert(rows.end(), row.begin(), row.end());
    }
    std::vector<int> out;
    const VerifyWalk w = walk_verify(
        rows.data(), 3, vocab, {2, 2}, s, 100, [](int) { return false; },
        [&](int t) {
            out.push_back(t);
            return Control::Continue;
        });
    CHECK_EQ(w.emitted, 1);
    CHECK_EQ(w.accepted, 0);
    CHECK_EQ(w.last, 5);
    CHECK_EQ(out, std::vector<int>({5}));
    CHECK_EQ(s.loop_guard_counts().slaps, 1);
}

// The nudge over a span: a forced id that equals the draft at its row is
// accepted as that draft, the next forced id that does not ends the walk.
TEST(loop_guard_nudge_inside_the_verify_walk_keeps_only_the_matching_prefix) {
    SamplerParams p = base_params(LoopGuardMode::Nudge, 10, 5);
    for (size_t i = 0; i < kPiece.size(); ++i) p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);
    CHECK_EQ(s.loop_guard_counts().nudges, 1);

    const size_t       vocab = 200;
    std::vector<float> rows;
    for (int r = 0; r < 3; ++r) {
        const std::vector<float> row = row_with_argmax(2, vocab);
        rows.insert(rows.end(), row.begin(), row.end());
    }
    std::vector<int> out;
    const VerifyWalk w = walk_verify(
        rows.data(), 3, vocab, {kPieceFirst, 7}, s, 100, [](int) { return false; },
        [&](int t) {
            out.push_back(t);
            return Control::Continue;
        });
    CHECK_EQ(w.accepted, 1);
    CHECK_EQ(w.emitted, 2);
    CHECK_EQ(out, std::vector<int>({kPieceFirst, kPieceFirst + 1}));
}

// A newline-free stretch longer than the open-line cap: the detector keeps
// judging the lines that follow, and still trips on a loop at the expected
// line.
TEST(loop_guard_long_open_line_does_not_disturb_the_detector) {
    SamplerParams p = base_params(LoopGuardMode::Detect);
    Sampler s(p, 1);
    s.start_in_think_block(true);
    const std::string chunk(1000, 'x');
    for (int i = 0; i < 50; ++i) feed_text(s, chunk);   // 50,000 bytes, one open line
    feed_text(s, "\n");                                  // ends it: one distinct line
    feed_lines(s, 1, 9, false);
    for (int i = 9; i < 17; ++i) feed_text(s, loop_line(i));
    CHECK_EQ(s.loop_guard_counts().detections, 0);
    feed_text(s, loop_line(17));
    CHECK_EQ(s.loop_guard_counts().detections, 1);
}

// Lines that arrive one byte a piece, as a byte-level tokenizer hands out a
// torn code point: the open line grows over many pieces and is judged whole
// once its newline comes, so the detector trips where it does for whole lines.
TEST(loop_guard_lines_fed_a_byte_at_a_time_are_judged_whole) {
    SamplerParams p = base_params(LoopGuardMode::Detect);
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    auto bytewise = [&](const std::string& line) {
        for (char c : line) feed_text(s, std::string(1, c));
    };
    for (int i = 9; i < 17; ++i) bytewise(loop_line(i));
    CHECK_EQ(s.loop_guard_counts().detections, 0);
    bytewise(loop_line(17));
    CHECK_EQ(s.loop_guard_counts().detections, 1);
}

// One piece that ends an open line and carries a whole short line after it:
// both line ends are seen, so the second line counts on its own.
TEST(loop_guard_a_piece_with_two_line_ends_counts_both_lines) {
    SamplerParams p = base_params(LoopGuardMode::Detect);
    Sampler s(p, 1);
    s.start_in_think_block(true);
    const std::string r = "same check again, once more\n";   // 27 characters: counted
    feed_lines(s, 0, 8, false);
    feed_text(s, "an open distinct line of forty characters");   // no newline yet
    feed_text(s, "\n" + r);                                       // ends it, then r whole
    for (int i = 0; i < 7; ++i) feed_text(s, r);
    CHECK_EQ(s.loop_guard_counts().detections, 0);
    feed_text(s, r);   // the 8th repeat of r: 8 of 10
    CHECK_EQ(s.loop_guard_counts().detections, 1);
}

// single newline instead. The masked draw is counted.
TEST(loop_guard_door_masks_multi_newlines_inside_an_open_block) {
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 5);
    p.loop_guard_door_ids = {6, 7};   // the double/triple newline ids; 4 is single
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // the detector fires; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    std::vector<float> row(8, -1.0f);
    row[4] = 1.0f;   // the single newline
    row[6] = 5.0f;   // a multi-newline token at the top
    const int tok = s.sample(row.data(), row.size());
    CHECK_EQ(tok, 4);
    CHECK(row[6] == -INFINITY);
    CHECK(row[7] == -INFINITY);
    CHECK(row[4] == 1.0f);   // the single newline was never masked
    CHECK_EQ(s.loop_guard_counts().door_masked, 1);
}

// The door never applies outside the block, never before detection, and never
// with the switch off. Those rows are the plain draws, byte-identical.
TEST(loop_guard_door_never_applies_outside_its_window) {
    // (a) off is today's behaviour: the switch ignores the door ids entirely.
    {
        SamplerParams p = base_params(LoopGuardMode::Off, 10, 5);
        p.loop_guard_door_ids = {6, 7};
        Sampler s(p, 1);
        s.start_in_think_block(true);
        feed_lines(s, 0, 40, true);
        std::vector<float> row(8, -1.0f);
        row[6] = 5.0f;
        const std::vector<float> before = row;
        CHECK_EQ(s.sample(row.data(), row.size()), 6);
        CHECK(row == before);
        CHECK_EQ(s.loop_guard_counts().door_masked, 0);
    }
    // (b) before detection: six varied lines, under the 10-line floor.
    {
        SamplerParams p = base_params(LoopGuardMode::Door, 10, 5);
        p.loop_guard_door_ids = {6, 7};
        Sampler s(p, 1);
        s.start_in_think_block(true);
        feed_lines(s, 0, 6, false);
        CHECK_EQ(s.loop_guard_counts().detections, 0);
        std::vector<float> row(8, -1.0f);
        row[6] = 5.0f;
        CHECK_EQ(s.sample(row.data(), row.size()), 6);
        CHECK_EQ(s.loop_guard_counts().door_masked, 0);
    }
    // (c) outside the block: the door opened, the model closed with the closer
    // id, and the content phase draws the multi-newline token again.
    {
        SamplerParams p = base_params(LoopGuardMode::Door, 10, 5);
        p.loop_guard_door_ids = {6, 7};
        Sampler s(p, 1);
        s.start_in_think_block(true);
        feed_lines(s, 0, 9, false);
        feed_lines(s, 9, 18, true);
        CHECK_EQ(s.loop_guard_counts().detections, 1);
        s.observe(5, "closed");
        CHECK(!s.think_block_open());
        std::vector<float> row(8, -1.0f);
        row[6] = 5.0f;
        CHECK_EQ(s.sample(row.data(), row.size()), 6);
        CHECK_EQ(s.loop_guard_counts().door_masked, 0);
    }
}

// The door lets the model close by its own draw: a stream whose top token is
// the double newline draws the closer once that is masked, counts one
// self-close, and no nudge or slap follows.
TEST(loop_guard_door_self_close_needs_no_nudge_or_slap) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 5);
    p.loop_guard_door_grace = 20;
    p.loop_guard_door_ids   = {6, 7};
    p.loop_guard_nudge_ids  = {kPieceFirst};   // armed, but it must not fire
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    CHECK_EQ(s.loop_guard_counts().nudges, 0);

    std::vector<float> row(8, -1.0f);
    row[6] = 10.0f;   // the model's likely pick: the double newline
    row[5] = 9.0f;    // the block's closer, legal, right behind it
    row[4] = 8.0f;    // the single newline
    const int tok = s.sample(row.data(), row.size());
    CHECK_EQ(tok, 5);
    s.observe(tok, "closed");
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);
    CHECK_EQ(s.loop_guard_counts().door_masked, 1);

    // The block is closed: no later line nudges or slaps.
    feed_lines(s, 18, 40, true);
    CHECK_EQ(s.loop_guard_counts().nudges, 0);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);
}

// Ladder order with the door grace: the door opens at detection, no nudge
// before K more counted lines, the nudge at the Kth, the slap after the nudge
// grace. ladder-nodoor keeps the old order: nudge at detection.
TEST(loop_guard_ladder_door_then_nudge_then_slap_or_nodoor_old_order) {
    auto detections_at = [&](LoopGuardMode mode, int door_grace, int slap_grace) {
        SamplerParams p = base_params(mode, 10, slap_grace);
        p.loop_guard_door_grace = door_grace;
        p.loop_guard_door_ids   = {6, 7};
        for (size_t i = 0; i < kPiece.size(); ++i)
            p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
        Sampler s(p, 1);
        s.start_in_think_block(true);
        feed_lines(s, 0, 9, false);
        feed_lines(s, 9, 18, true);   // detection at the 9th repeating line
        return s;
    };

    // ladder: door grace 4, slap grace 3.
    {
        Sampler s = detections_at(LoopGuardMode::Ladder, 4, 3);
        CHECK_EQ(s.loop_guard_counts().detections, 1);
        CHECK_EQ(s.loop_guard_counts().nudges, 0);   // no nudge at detection
        for (int i = 18; i < 21; ++i) {   // 3 more counted lines: still short of 4
            feed_text(s, loop_line(i));
            CHECK_EQ(s.loop_guard_counts().nudges, 0);
        }
        feed_text(s, loop_line(21));   // the 4th line after detection: the nudge
        CHECK_EQ(s.loop_guard_counts().nudges, 1);
        CHECK_EQ(s.loop_guard_counts().slaps, 0);

        // Deliver the nudge sentence, then wait out the slap grace.
        for (size_t i = 0; i < kPiece.size(); ++i) {
            std::vector<float> l = row_with_argmax(2);
            const int id = s.sample(l.data(), l.size());
            s.observe_think(id, piece_of(id));
        }
        feed_text(s, loop_line(22));
        feed_text(s, loop_line(23));   // 2 lines after the nudge: no slap yet
        CHECK_EQ(s.loop_guard_counts().slaps, 0);
        feed_text(s, loop_line(24));   // the 3rd: the slap
        CHECK_EQ(s.loop_guard_counts().slaps, 1);
        std::vector<float> l = row_with_argmax(2);
        CHECK_EQ(s.sample(l.data(), l.size()), 5);
    }

    // ladder-nodoor: the nudge is the first rung at detection, exactly as the
    // ladder before the door.
    {
        Sampler s = detections_at(LoopGuardMode::LadderNoDoor, 4, 3);
        CHECK_EQ(s.loop_guard_counts().detections, 1);
        CHECK_EQ(s.loop_guard_counts().nudges, 1);   // immediately at detection
        CHECK_EQ(s.loop_guard_counts().slaps, 0);
        for (size_t i = 0; i < kPiece.size(); ++i) {
            std::vector<float> l = row_with_argmax(2);
            const int id = s.sample(l.data(), l.size());
            s.observe_think(id, piece_of(id));
        }
        feed_text(s, loop_line(18));
        feed_text(s, loop_line(19));   // 2 lines after the nudge: no slap yet
        CHECK_EQ(s.loop_guard_counts().slaps, 0);
        feed_text(s, loop_line(20));   // the 3rd: the slap, as before
        CHECK_EQ(s.loop_guard_counts().slaps, 1);
    }
}

// The door counters reach the stats consumers: the decode suffix, the response
// object, and the /props defaults carry the door grace and the mode names.
TEST(loop_guard_door_counters_and_modes_parse) {
    GenerationStats st;
    st.loop_self_closes = 2;
    st.loop_door_masked = 3;
    const nlohmann::json j = api::loop_guard_json(st);
    CHECK_EQ(j.at("self_closes").get<int>(), 2);
    CHECK_EQ(j.at("door_masked").get<int>(), 3);
    const std::string line = api::decode_suffix_for(st);
    CHECK(line.find("self-closed 2") != std::string::npos);
    CHECK(line.find("door masked 3") != std::string::npos);

    // Mode names travel through the request field and the flag.
    CHECK(loop_guard_parse("door") == LoopGuardMode::Door);
    CHECK(loop_guard_parse("ladder-nodoor") == LoopGuardMode::LadderNoDoor);
    CHECK(loop_guard_uses_door(LoopGuardMode::Door));
    CHECK(loop_guard_uses_door(LoopGuardMode::Ladder));
    CHECK(!loop_guard_uses_door(LoopGuardMode::LadderNoDoor));

    nlohmann::json body = {{"model", "m"},
                           {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                           {"loop_guard", "ladder-nodoor"}};
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.sampler.loop_guard.has_value());
    CHECK(*req.sampler.loop_guard == LoopGuardMode::LadderNoDoor);

    SamplerParams p = base_params(LoopGuardMode::Ladder);
    CHECK(!sampler_apply(p, req.sampler).has_value());
    CHECK(p.loop_guard == LoopGuardMode::LadderNoDoor);   // the field wins

    Config cfg;
    std::vector<const char*> args = {"--stub", "--loop-guard", "door",
                                     "--loop-guard-door-grace", "7"};
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("arcint"));
    for (const char* a : args) argv.push_back(const_cast<char*>(a));
    CHECK(parse_args(static_cast<int>(argv.size()), argv.data(), cfg).ok);
    CHECK_EQ(cfg.loop_guard_door_grace, 7);
    SamplerDefaults d;
    CHECK(!apply_operator_defaults(cfg, d).has_value());
    CHECK(d.loop_guard == LoopGuardMode::Door);
    CHECK_EQ(d.loop_guard_door_grace, 7);
}
