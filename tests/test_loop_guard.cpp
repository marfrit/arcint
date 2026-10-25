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

#include <atomic>
#include <string>
#include <thread>

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
    // Today's ladder, without the door: the nudge lands at detection, so the
    // tail check does not also depend on the door grace. The door's own
    // ladder order is pinned in its tests.
    SamplerParams p = base_params(LoopGuardMode::LadderNoDoor, 10, 40);
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
    st.loop_self_closes    = 2;
    st.loop_door_masked    = 3;
    st.loop_door_eog_masked = 4;
    const nlohmann::json j = api::loop_guard_json(st);
    CHECK_EQ(j.at("self_closes").get<int>(), 2);
    CHECK_EQ(j.at("door_masked").get<int>(), 3);
    const std::string line = api::decode_suffix_for(st);
    CHECK(line.find("self-closed 2") != std::string::npos);
    CHECK(line.find("door masked 3") != std::string::npos);
    CHECK(line.find("door EOG masked 4") != std::string::npos);

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

// A slap's forced closer is not the model's own draw: with the door open, the
// ladder's slap must leave self_closes at 0 and count the slap. The card test
// reads exactly these two counters to tell a self-close from a slap.
TEST(loop_guard_door_slap_is_not_a_self_close) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 3);
    p.loop_guard_door_grace = 4;
    p.loop_guard_door_ids   = {6, 7};
    for (size_t i = 0; i < kPiece.size(); ++i)
        p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection: the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    for (int i = 18; i < 22; ++i) feed_text(s, loop_line(i));   // door grace 4
    CHECK_EQ(s.loop_guard_counts().nudges, 1);

    // Deliver the nudge sentence, then wait out the slap grace.
    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        s.observe_think(id, piece_of(id));
    }
    for (int i = 22; i < 25; ++i) feed_text(s, loop_line(i));   // 3 lines: slap
    CHECK_EQ(s.loop_guard_counts().slaps, 1);

    // The next draw is the forced closer, observed exactly as a decode loop
    // does. It closes the block, but it is not the model's own draw.
    std::vector<float> l(8, -1.0f);
    l[6] = 10.0f;   // the model's likely pick, forbidden by the door
    l[5] = 9.0f;    // the closer, legal
    CHECK_EQ(s.sample(l.data(), l.size()), 5);
    s.observe(5, "closed");
    CHECK_EQ(s.loop_guard_counts().self_closes, 0);
    CHECK_EQ(s.loop_guard_counts().slaps, 1);
}

// ladder-nodoor is the ladder before the door: it never opens the mask, so a
// row whose top token is a multi-newline token still draws it.
TEST(loop_guard_ladder_nodoor_never_opens_the_door) {
    SamplerParams p = base_params(LoopGuardMode::LadderNoDoor, 10, 5);
    p.loop_guard_door_grace = 4;
    p.loop_guard_door_ids   = {6, 7};   // set, but this rung must not read them
    p.loop_guard_nudge_ids  = {kPieceFirst};
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; today's order nudges at once
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    CHECK_EQ(s.loop_guard_counts().nudges, 1);

    // The nudge is in flight at the detection moment: deliver its one id first,
    // exactly as a decode loop does.
    std::vector<float> nudge_row = row_with_argmax(2);
    CHECK_EQ(s.sample(nudge_row.data(), nudge_row.size()), kPieceFirst);
    s.observe_think(kPieceFirst, piece_of(kPieceFirst));

    std::vector<float> row(8, -1.0f);
    row[4] = 1.0f;   // the single newline
    row[6] = 5.0f;   // a multi-newline token at the top
    CHECK_EQ(s.sample(row.data(), row.size()), 6);
    CHECK(row[6] == 5.0f);   // the multi-newline ids stay finite
    CHECK(row[7] == -1.0f);
    CHECK_EQ(s.loop_guard_counts().door_masked, 0);
}

// ── door polish: the scan and the counter ─────────────────────────────────────

// A stub-shaped tokenizer: pieces interned by id, like the stub's table, but
// with a multi-newline piece the stub's byte splitter never makes. It counts
// decode_one calls, so the test can see how many scans ran.
class DoorPieceTokenizer final : public Tokenizer {
public:
    explicit DoorPieceTokenizer(std::vector<std::string> pieces)
        : pieces_(std::move(pieces)) {}
    std::vector<int> encode(std::string_view text) override {
        std::vector<int> out;
        for (char c : text) out.push_back(intern(std::string(1, c)));
        return out;
    }
    std::string decode(const std::vector<int>& ids) override {
        std::string out;
        for (int id : ids) out += decode_one(id);
        return out;
    }
    std::string decode_one(int id) override {
        ++decode_calls;
        if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) return {};
        return pieces_[static_cast<size_t>(id)];
    }
    int eos_id() const override { return 0; }
    int vocab_size() const override { return static_cast<int>(pieces_.size()); }

    mutable std::atomic<int> decode_calls{0};

private:
    int intern(const std::string& piece) {
        for (size_t i = 0; i < pieces_.size(); ++i)
            if (pieces_[i] == piece) return static_cast<int>(i);
        pieces_.push_back(piece);
        return static_cast<int>(pieces_.size() - 1);
    }
    std::vector<std::string> pieces_;
};

// 1. The door-id scan is pure and thread-safe: the ids are the multi-newline
// set before any request, and two threads scanning at once see the same ids.
TEST(loop_guard_door_ids_scan_is_pure_and_seen_before_a_request) {
    DoorPieceTokenizer tok({"", "\n", "\n\n", "x", "\n\n\n"});
    const std::vector<int> once = scan_loop_guard_door_ids(tok);
    CHECK(once == std::vector<int>({2, 4}));   // the single newline (1) is not
    CHECK_EQ(tok.decode_calls.load(), 5);      // the scan happens, before any request

    tok.decode_calls = 0;
    std::vector<int> a, b;
    std::thread t1([&] { a = scan_loop_guard_door_ids(tok); });
    std::thread t2([&] { b = scan_loop_guard_door_ids(tok); });
    t1.join();
    t2.join();
    CHECK(a == once);
    CHECK(b == once);
    CHECK_EQ(tok.decode_calls.load(), 2 * tok.vocab_size());   // two independent passes
}

// 2. The counting change moves no draw: a fixed stream of rows under the open
// door draws exactly the same ids it drew before the change.
TEST(loop_guard_door_counting_change_leaves_the_draw_byte_identical) {
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 5);
    p.loop_guard_door_ids = {6, 7};
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection: the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    auto row = [](std::vector<float> v) { return v; };
    std::vector<std::vector<float>> rows = {
        row({-1, -1, -1, 1, 2, -1, 5, -1}),   // double newline on top -> single (4)
        row({-1, -1, -1, -1, -1, 6, 7, -1}),  // closer 5 (double 6 is under it)
        row({-1, -1, -1, 5, 3, -1, -1, 4}),   // triple newline 7 on top -> 3
        row({-1, -1, 1, -1, 5, -1, -1, -1}),  // single newline on top -> 4
        row({-1, -1, -1, -1, 4, 2, 3, -1}),   // double 6 masked -> 4
    };
    std::vector<int> drawn;
    for (auto& r : rows) drawn.push_back(s.sample(r.data(), r.size()));
    CHECK(drawn == std::vector<int>({4, 5, 3, 4, 4}));
}

// 3. door_masked counts the candidate membership and only that: a masked id
// inside the row's top-k counts, one below the floor does not.
TEST(loop_guard_door_masked_counts_topk_membership_not_any_mask) {
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 5);
    p.loop_guard_door_ids = {6, 7};
    p.temperature         = 0.7f;   // a candidate set, not the greedy argmax
    p.top_k               = 2;
    p.top_p               = 1.0f;
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    // The double newline is below the kept pair: 5 and 4.5 are in, 4.0 is out.
    std::vector<float> below(8, -1.0f);
    below[4] = 5.0f;
    below[3] = 4.5f;
    below[6] = 4.0f;
    const int t1 = s.sample(below.data(), below.size());
    CHECK(t1 == 4 || t1 == 3);
    CHECK_EQ(s.loop_guard_counts().door_masked, 0);

    // The double newline is above the kept pair's floor (4.0): it would have
    // been among the candidates the mask removed it from.
    std::vector<float> inside(8, -1.0f);
    inside[4] = 5.0f;
    inside[6] = 4.8f;
    inside[3] = 4.0f;
    const int t2 = s.sample(inside.data(), inside.size());
    CHECK(t2 == 4 || t2 == 3);
    CHECK_EQ(s.loop_guard_counts().door_masked, 1);
}

// ── 1.2.4 gaps: the answer after a door self-close, and the back door ────────

// Gap 1: the door opened, the model closed the block itself, and the answer
// then loops. The content phase is watched after a door-open turn exactly as
// after a slap: the turn ends on `loop`, with one self-close counted.
TEST(loop_guard_answer_after_a_door_self_close_aborts_on_a_loop) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 5);
    p.loop_guard_door_grace = 20;
    p.loop_guard_door_ids   = {6, 7};
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    // The model closes by its own draw: the multi-newline top is masked, the
    // closer is next.
    std::vector<float> row(8, -1.0f);
    row[6] = 10.0f;
    row[5] = 9.0f;
    CHECK_EQ(s.sample(row.data(), row.size()), 5);
    s.observe(5, "closed");
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);

    // The answer loops. It is watched now, from a clean window at the close.
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);
    CHECK(s.loop_abort());
    CHECK_EQ(s.loop_guard_counts().aborts, 1);
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);

    std::vector<float> rows = row_with_argmax(3);
    const VerifyWalk   w = walk_verify(
        rows.data(), 1, rows.size(), {}, s, 100, [](int) { return false; },
        [&](int) { return Control::Continue; });
    CHECK(w.done);
    CHECK(w.reason == FinishReason::Loop);
    CHECK_EQ(std::string(finish_reason_name(w.reason)), std::string("loop"));
}

// Gap 1, negative: an answer that restates a line or two of its reasoning once
// is normal. The window is clean at the block's close, so those lines are not
// repeats and no abort follows.
TEST(loop_guard_answer_restating_two_reasoning_lines_once_does_not_abort) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 5);
    p.loop_guard_door_grace = 20;
    p.loop_guard_door_ids   = {6, 7};
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    std::vector<float> row(8, -1.0f);
    row[6] = 10.0f;
    row[5] = 9.0f;
    CHECK_EQ(s.sample(row.data(), row.size()), 5);
    s.observe(5, "closed");
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);

    feed_lines(s, 0, 4, false);         // the answer's own varied lines
    feed_text(s, distinct_line(0));     // restates reasoning line 0, once
    feed_text(s, distinct_line(1));     // restates reasoning line 1, once
    feed_lines(s, 4, 12, false);        // and goes on, varied
    CHECK(!s.loop_abort());
    CHECK_EQ(s.loop_guard_counts().aborts, 0);
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);

    std::vector<float> rows = row_with_argmax(3);
    const VerifyWalk   w = walk_verify(
        rows.data(), 1, rows.size(), {}, s, 100, [](int) { return false; },
        [&](int) { return Control::Continue; });
    CHECK(w.reason == FinishReason::Stop);
}

// Gap 1, negative: a turn whose door never opened keeps today's behaviour. Its
// answer is unwatched, so a repeating answer does not abort; the scope stays
// the door-open turns only.
TEST(loop_guard_answer_unwatched_when_the_door_never_opened) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 5);
    p.loop_guard_door_grace = 20;
    p.loop_guard_door_ids   = {6, 7};
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 40, false);   // varied reasoning: no detection, door shut
    CHECK_EQ(s.loop_guard_counts().detections, 0);
    s.observe(5, "closed");        // the block closes, the door was never open
    CHECK_EQ(s.loop_guard_counts().self_closes, 0);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);    // a looping answer
    CHECK(!s.loop_abort());
    CHECK_EQ(s.loop_guard_counts().aborts, 0);

    std::vector<float> rows = row_with_argmax(3);
    const VerifyWalk   w = walk_verify(
        rows.data(), 1, rows.size(), {}, s, 100, [](int) { return false; },
        [&](int) { return Control::Continue; });
    CHECK(w.reason == FinishReason::Stop);
}

// Gap 2: the back door. With the door open and think_eog_mask off (the
// coder-style config), the model can draw an end-of-generation id inside the
// open think block and end the turn there: finish stop, empty content, no
// self-close. The turn-ending ids are illegal while the door is open and the
// block is open, applied to the logits before penalties and the draw. The
// closer stays drawable (review B3).
TEST(loop_guard_door_masks_eog_inside_the_open_block) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 5);
    p.loop_guard_door_grace = 20;
    p.loop_guard_door_ids   = {6, 7};
    p.think_eog_mask        = false;   // the coder-style config
    p.ignore_eos            = false;
    // The closer is also a turn-ending id in this stub row: B3 must leave it
    // offered so the block can close.
    p.think_ids.eog = {0, 5};
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    std::vector<float> row(8, -1.0f);
    row[0] = 10.0f;   // the end-of-generation id at the top
    row[5] = 9.0f;    // the block's closer, legal
    row[6] = 7.0f;    // a multi-newline token, masked by the door
    row[4] = 2.0f;    // the single newline
    const int tok = s.sample(row.data(), row.size());
    CHECK_EQ(tok, 5);                 // not the EOG, and the closer was drawable
    CHECK(row[0] == -INFINITY);       // the EOG id is masked
    CHECK(row[5] == 9.0f);            // B3: the closer is never masked
    CHECK(s.loop_guard_counts().door_eog_masked >= 1);
}

// Gap 2, negative: the door shut (no detection) keeps today's behaviour --
// with think_eog_mask off, an EOG inside the open block stays legal.
TEST(loop_guard_door_shut_keeps_eog_legal) {
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 5);
    p.loop_guard_door_ids = {6, 7};
    p.think_eog_mask      = false;
    p.ignore_eos          = false;
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 6, false);   // under the floor: no detection, door shut
    CHECK_EQ(s.loop_guard_counts().detections, 0);

    std::vector<float> row(8, -1.0f);
    row[0] = 5.0f;   // the EOG id at the top
    CHECK_EQ(s.sample(row.data(), row.size()), 0);
    CHECK(row[0] == 5.0f);
    CHECK_EQ(s.loop_guard_counts().door_eog_masked, 0);
}

// Gap 2 with ignore_eos: the request asked for the turn to run past EOG, so
// the door does not mask it, and the counter stays 0.
TEST(loop_guard_door_respects_ignore_eos_for_eog) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 5);
    p.loop_guard_door_grace = 20;
    p.loop_guard_door_ids   = {6, 7};
    p.think_eog_mask        = false;
    p.ignore_eos            = true;
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    std::vector<float> row(8, -1.0f);
    row[0] = 5.0f;   // the EOG id at the top
    CHECK_EQ(s.sample(row.data(), row.size()), 0);
    CHECK(row[0] == 5.0f);
    CHECK_EQ(s.loop_guard_counts().door_eog_masked, 0);
}

// Gap 1 on the text-closer path: models whose closer has no single token id
// close the block by spelling it. The answer still starts from a clean window
// there. The answer restates twelve distinct reasoning lines once each: from a
// clean window they are first occurrences and no abort follows, but a stale
// window (the reasoning loop still at the threshold) would abort on the first.
TEST(loop_guard_text_closer_answer_starts_from_a_clean_window) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 5);
    p.loop_guard_door_grace = 20;
    p.loop_guard_door_ids   = {6, 7};
    // The text fallback: no closer id, so only the spelled tag closes.
    p.think_ids.close_ids.clear();
    p.think_ids.open_close_ids.clear();
    Sampler s(p, 1);
    s.start_in_think_block(true);
    CHECK(s.think_block_open());

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    feed_text(s, think_tag_pairs(false)[0].second);   // the model spells the closer
    CHECK(!s.think_block_open());
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);

    for (int i = 0; i < 12; ++i) feed_text(s, distinct_line(i));   // restated once each
    CHECK(!s.loop_abort());
    CHECK_EQ(s.loop_guard_counts().aborts, 0);
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);

    std::vector<float> rows = row_with_argmax(3);
    const VerifyWalk   w = walk_verify(
        rows.data(), 1, rows.size(), {}, s, 100, [](int) { return false; },
        [&](int) { return Control::Continue; });
    CHECK(w.reason == FinishReason::Stop);
}

// Gap 2's counter on the sampling path (temperature > 0, top-k > 1), not only
// the greedy path: an end-of-generation id above the kept pair's floor is a
// finite candidate the door removed, so door_eog_masked counts it.
TEST(loop_guard_door_eog_masked_counts_on_the_sampling_path) {
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 5);
    p.loop_guard_door_ids = {6, 7};
    p.think_eog_mask      = false;
    p.ignore_eos          = false;
    p.temperature         = 0.7f;
    p.top_k               = 2;
    p.top_p               = 1.0f;
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    std::vector<float> row(8, -1.0f);
    row[0] = 5.0f;   // the EOG id above the kept pair's floor
    row[3] = 4.5f;
    row[4] = 4.2f;
    const int tok = s.sample(row.data(), row.size());
    CHECK(tok == 3 || tok == 4);
    CHECK(row[0] == -INFINITY);
    CHECK_EQ(s.loop_guard_counts().door_eog_masked, 1);
}

// ── 1.2.5: the door owns the turn's ending ────────────────────────────────

// (a) ladder: once the door is open, escalation goes by budget, not by the
// share. Detection at 9 of 10 lines, then varied lines bring the share below
// the threshold while the block stays open: the nudge still fires at the door
// grace's counted lines, the slap at the nudge's grace, and the turn ends by
// the forced closer instead of running to the token cap.
TEST(loop_guard_door_ladder_escalates_by_budget_when_the_share_falls) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 3);
    p.loop_guard_door_grace = 4;
    p.loop_guard_door_ids   = {6, 7};
    for (size_t i = 0; i < kPiece.size(); ++i)
        p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    CHECK_EQ(s.loop_guard_counts().nudges, 0);

    feed_text(s, distinct_line(18));   // 1 counted line since detection
    feed_text(s, distinct_line(19));   // 2
    feed_text(s, distinct_line(20));   // 3
    CHECK_EQ(s.loop_guard_counts().nudges, 0);
    feed_text(s, distinct_line(21));   // 4: the door grace, share now 0.7 < 0.8
    CHECK_EQ(s.loop_guard_counts().nudges, 1);
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    // Deliver the nudge sentence, then three varied counted lines: the nudge's
    // grace, whatever the share.
    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        s.observe_think(id, piece_of(id));
    }
    feed_text(s, distinct_line(22));
    feed_text(s, distinct_line(23));
    CHECK_EQ(s.loop_guard_counts().slaps, 0);
    feed_text(s, distinct_line(24));   // the third: the slap
    CHECK_EQ(s.loop_guard_counts().slaps, 1);

    // The turn ends on the forced closer, not at the token cap.
    std::vector<float> l = row_with_argmax(2);
    CHECK_EQ(s.sample(l.data(), l.size()), 5);
    s.observe(5, "closed");
    CHECK(!s.think_block_open());
}

// (a) the same with think_eog_mask on: the door's ladder is unchanged by the
// turn-ending mask, and the budget still ends the turn inside the open block.
TEST(loop_guard_door_ladder_escalates_by_budget_with_think_eog_mask) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 3);
    p.loop_guard_door_grace = 4;
    p.loop_guard_door_ids   = {6, 7};
    p.think_eog_mask        = true;
    for (size_t i = 0; i < kPiece.size(); ++i)
        p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    feed_text(s, distinct_line(18));
    feed_text(s, distinct_line(19));
    feed_text(s, distinct_line(20));
    CHECK_EQ(s.loop_guard_counts().nudges, 0);
    feed_text(s, distinct_line(21));   // the door grace: nudge, share 0.7
    CHECK_EQ(s.loop_guard_counts().nudges, 1);

    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        s.observe_think(id, piece_of(id));
    }
    feed_text(s, distinct_line(22));
    feed_text(s, distinct_line(23));
    CHECK_EQ(s.loop_guard_counts().slaps, 0);
    feed_text(s, distinct_line(24));   // the nudge grace: slap
    CHECK_EQ(s.loop_guard_counts().slaps, 1);
    std::vector<float> l = row_with_argmax(2);
    CHECK_EQ(s.sample(l.data(), l.size()), 5);
}

// (a) `door` (no ladder): a mask alone must not hold a turn to the cap. After
// door_grace + grace counted lines without a close, the door gives up its
// masks: the multi-newline ids and the turn-ending ids are drawable again, the
// door counters stop, and a later detection does not reopen the mask.
TEST(loop_guard_door_mode_lifts_its_masks_after_the_budget) {
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 2);
    p.loop_guard_door_grace = 2;
    p.loop_guard_door_ids   = {6, 7};
    p.think_eog_mask        = false;
    p.ignore_eos            = false;
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    // The block stays open and the share falls; 4 = door_grace + grace lines.
    feed_text(s, distinct_line(18));
    feed_text(s, distinct_line(19));
    feed_text(s, distinct_line(20));
    std::vector<float> still(8, -1.0f);
    still[0] = 5.0f;    // the EOG id, still masked before the budget
    still[6] = 6.0f;    // a multi-newline token, still masked
    still[4] = 1.0f;    // the single newline
    CHECK_EQ(s.sample(still.data(), still.size()), 4);
    CHECK(still[0] == -INFINITY);
    CHECK(still[6] == -INFINITY);
    CHECK_EQ(s.loop_guard_counts().door_masked, 1);
    CHECK_EQ(s.loop_guard_counts().door_eog_masked, 1);

    feed_text(s, distinct_line(21));   // the 4th: the door lifts
    std::vector<float> lifted(8, -1.0f);
    lifted[0] = 5.0f;
    lifted[6] = 6.0f;
    lifted[4] = 1.0f;
    CHECK_EQ(s.sample(lifted.data(), lifted.size()), 6);   // the door is gone
    CHECK(lifted[0] == 5.0f);   // the EOG id is drawable again
    CHECK(lifted[6] == 6.0f);
    // The counters stop at the lift: this draw removed nothing.
    CHECK_EQ(s.loop_guard_counts().door_masked, 1);
    CHECK_EQ(s.loop_guard_counts().door_eog_masked, 1);

    // A later crossing does not reopen a mask the guard gave up: the turn is
    // the switch-off path from here.
    feed_lines(s, 0, 12, true);
    CHECK_EQ(s.loop_guard_counts().detections, 2);
    std::vector<float> after(8, -1.0f);
    after[0] = 5.0f;
    CHECK_EQ(s.sample(after.data(), after.size()), 0);
    CHECK(after[0] == 5.0f);
}

// (a) ladder-nodoor keeps 1.2.3's semantics: escalation only on an over line,
// however long the turn goes on below the threshold (it never opens the door).
TEST(loop_guard_ladder_nodoor_does_not_escalate_without_an_over_line) {
    SamplerParams p = base_params(LoopGuardMode::LadderNoDoor, 10, 3);
    p.loop_guard_door_grace = 4;
    p.loop_guard_door_ids   = {6, 7};
    for (size_t i = 0; i < kPiece.size(); ++i)
        p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the nudge is the first rung here
    CHECK_EQ(s.loop_guard_counts().nudges, 1);

    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        s.observe_think(id, piece_of(id));
    }
    feed_lines(s, 18, 60, false);   // 42 varied lines: the share never rises
    CHECK_EQ(s.loop_guard_counts().nudges, 1);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);
}

// (a) a model that closes the block itself before the budget: no escalation
// follows, and the close is still counted as the model's own.
TEST(loop_guard_door_ladder_self_close_before_the_budget_nudges_not) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 3);
    p.loop_guard_door_grace = 4;
    p.loop_guard_door_ids   = {6, 7};
    p.loop_guard_nudge_ids  = {kPieceFirst};   // armed, must not fire
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    feed_text(s, distinct_line(18));
    feed_text(s, distinct_line(19));
    feed_text(s, distinct_line(20));   // 3 counted lines: one short of the grace
    CHECK_EQ(s.loop_guard_counts().nudges, 0);

    std::vector<float> row(8, -1.0f);
    row[6] = 10.0f;   // the double newline, masked by the door
    row[5] = 9.0f;    // the block's closer, legal
    CHECK_EQ(s.sample(row.data(), row.size()), 5);
    s.observe(5, "closed");
    CHECK_EQ(s.loop_guard_counts().self_closes, 1);

    feed_lines(s, 21, 60, false);   // a long varied answer after the close
    CHECK_EQ(s.loop_guard_counts().nudges, 0);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);
}

// (b) at top_k 0 the door counters read the candidate set the draw samples
// from -- the prefix that survives top-p -- not the whole row. A masked id
// below the cutoff is not a candidate; one inside the cutoff is.
TEST(loop_guard_door_masked_counts_the_truncated_set_at_top_k_zero) {
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 5);
    p.loop_guard_door_ids = {6, 7};
    p.temperature         = 0.7f;
    p.top_k               = 0;       // no top-k: the top-p cutoff is the set
    p.top_p               = 0.9f;
    Sampler s(p, 1);
    s.start_in_think_block(true);
    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    // The masked double newline sits below the surviving prefix's floor: 5.0
    // and 4.9 hold >= 0.9 of the unmasked mass, so the set is those two.
    std::vector<float> below(8, -1.0f);
    below[4] = 5.0f;
    below[3] = 4.9f;
    below[6] = 4.8f;   // masked, and below the floor the draw keeps
    const int t1 = s.sample(below.data(), below.size());
    CHECK_EQ(t1, 4);
    CHECK_EQ(s.loop_guard_counts().door_masked, 0);

    // The same row with the masked id above the floor: it would have been a
    // candidate of the same truncated set.
    std::vector<float> inside(8, -1.0f);
    inside[4] = 5.0f;
    inside[3] = 4.9f;
    inside[6] = 4.95f;   // masked, above the floor
    const int t2 = s.sample(inside.data(), inside.size());
    CHECK_EQ(t2, 4);
    CHECK_EQ(s.loop_guard_counts().door_masked, 1);

    // Observability only: the draw itself is unchanged. A fixed seed and these
    // rows give a fixed token sequence, so a change that moved the draw would
    // have to move this list too (the top-k 0, top-p 1 case is pinned by
    // loop_guard_door_counting_change_leaves_the_draw_byte_identical).
    std::vector<float> third(8, -1.0f);
    third[4] = 4.0f;
    third[3] = 4.9f;
    third[6] = 4.95f;
    const int t3 = s.sample(third.data(), third.size());
    CHECK_EQ(std::vector<int>({t1, t2, t3}), std::vector<int>({4, 4, 3}));
}

// ── 1.2.5 round 2: the door's budget in tokens, and the lifted-door latch ──

// (A) The door's budget counted ended lines only, so a turn that switched to
// newline-free text never advanced it and the masks could hold the turn to the
// token cap. Each stretch of tokens without a counted line now counts as one
// line-equivalent for the budget (never for the detector's window): the ladder
// nudges and slaps by budget on newline-free text exactly as on lines.
TEST(loop_guard_door_budget_advances_by_tokens_without_line_ends) {
    constexpr int kStretch = 64;   // the sampler's stretch length
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 1);
    p.loop_guard_door_grace = 3;
    p.loop_guard_door_ids   = {6, 7};
    for (size_t i = 0; i < kPiece.size(); ++i)
        p.loop_guard_nudge_ids.push_back(kPieceFirst + static_cast<int>(i));
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    CHECK_EQ(s.loop_guard_counts().nudges, 0);

    // A counted line is the anchor: it spends one grace line and restarts the
    // token stretch, so the numbers below are exact whatever the detection did.
    feed_text(s, distinct_line(18));
    CHECK_EQ(s.loop_guard_counts().nudges, 0);

    // The model writes only newline-free text: no line ever ends again.
    int tokens = 0;
    auto tok = [&] {
        s.observe_think(30, "word ");   // no newline in the piece
        ++tokens;
    };
    while (tokens < kStretch - 1) tok();   // one stretch short: grace 2 of 3
    CHECK_EQ(s.loop_guard_counts().nudges, 0);
    tok();                                 // the first stretch: grace 2
    CHECK_EQ(s.loop_guard_counts().nudges, 0);
    while (tokens < 2 * kStretch - 1) tok();
    CHECK_EQ(s.loop_guard_counts().nudges, 0);
    tok();                                 // the second stretch: grace 3, the nudge
    CHECK_EQ(s.loop_guard_counts().nudges, 1);
    CHECK_EQ(s.loop_guard_counts().detections, 1);   // no detection from tokens

    while (tokens < 3 * kStretch - 1) tok();   // the nudge's grace not yet
    CHECK_EQ(s.loop_guard_counts().slaps, 0);
    tok();                                     // the next stretch: the slap
    CHECK_EQ(s.loop_guard_counts().slaps, 1);

    // Deliver the nudge, then the forced closer: the turn ends, not the cap.
    for (size_t i = 0; i < kPiece.size(); ++i) {
        std::vector<float> l = row_with_argmax(2);
        const int id = s.sample(l.data(), l.size());
        s.observe_think(id, piece_of(id));
    }
    std::vector<float> l = row_with_argmax(2);
    CHECK_EQ(s.sample(l.data(), l.size()), 5);
    s.observe(5, "closed");
    CHECK(!s.think_block_open());
}

// (A) `door` mode on the same newline-free stream: after door_grace + grace
// line-equivalents the door lifts, and the multi-newline and turn-ending ids
// are drawable again. Before the lift the masks still hold.
TEST(loop_guard_door_mode_lifts_after_token_stretches_without_lines) {
    constexpr int kStretch = 64;
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 1);
    p.loop_guard_door_grace = 1;
    p.loop_guard_door_ids   = {6, 7};
    p.think_eog_mask        = false;
    p.ignore_eos            = false;
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    feed_text(s, distinct_line(18));   // the counted-line anchor: grace 1
    int tokens = 0;
    auto tok = [&] {
        s.observe_think(30, "word ");
        ++tokens;
    };
    while (tokens < kStretch - 1) tok();   // 1 short of door_grace + grace
    std::vector<float> still(8, -1.0f);
    still[0] = 5.0f;    // the EOG id
    still[6] = 6.0f;    // a multi-newline token
    still[4] = 1.0f;    // the single newline
    CHECK_EQ(s.sample(still.data(), still.size()), 4);
    CHECK(still[0] == -INFINITY);
    CHECK(still[6] == -INFINITY);

    tok();   // the stretch: door_grace + grace, the lift
    std::vector<float> lifted(8, -1.0f);
    lifted[0] = 5.0f;
    lifted[6] = 6.0f;
    lifted[4] = 1.0f;
    CHECK_EQ(s.sample(lifted.data(), lifted.size()), 6);
    CHECK(lifted[0] == 5.0f);
    CHECK(lifted[6] == 6.0f);
}

// (A2) The lifted-door latch, pinned: after `door` mode's budget lifted the
// masks, the share falls and rises again to a second detection -- the masks
// stay off and the counters do not grow. A lifted door is not reopened.
TEST(loop_guard_door_lifted_is_not_reopened_by_a_later_detection) {
    constexpr int kStretch = 64;
    SamplerParams p = base_params(LoopGuardMode::Door, 10, 2);
    p.loop_guard_door_grace = 1;
    p.loop_guard_door_ids   = {6, 7};
    p.think_eog_mask        = false;
    p.ignore_eos            = false;
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    feed_lines(s, 9, 18, true);   // detection; the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);

    feed_text(s, distinct_line(18));   // the anchor: grace 1
    // door_grace 1 + the ladder grace 2 = three line-equivalents: the anchor
    // plus two token stretches.
    for (int i = 0; i < 2 * kStretch; ++i) s.observe_think(30, "word ");
    std::vector<float> open_row(8, -1.0f);
    open_row[0] = 5.0f;
    open_row[6] = 6.0f;
    open_row[4] = 1.0f;
    CHECK_EQ(s.sample(open_row.data(), open_row.size()), 6);   // lifted
    CHECK_EQ(s.loop_guard_counts().door_masked, 0);
    CHECK_EQ(s.loop_guard_counts().door_eog_masked, 0);

    // The share falls (varied lines), then rises again to a second detection.
    // Feed line by line and stop at the detection, so the sample below is the
    // first draw after it: a reopened door is still masked here.
    feed_lines(s, 18, 30, false);
    for (int i = 0; i < 40 && s.loop_guard_counts().detections < 2; ++i) feed_text(s, loop_line(i));
    CHECK_EQ(s.loop_guard_counts().detections, 2);

    std::vector<float> after(8, -1.0f);
    after[0] = 5.0f;   // the EOG id: masked again only if the door reopened
    after[6] = 6.0f;   // a multi-newline token, likewise
    after[4] = 1.0f;
    CHECK_EQ(s.sample(after.data(), after.size()), 6);
    CHECK(after[0] == 5.0f);
    CHECK(after[6] == 6.0f);
    CHECK_EQ(s.loop_guard_counts().door_masked, 0);
    CHECK_EQ(s.loop_guard_counts().door_eog_masked, 0);
}

// ── 1.2.5 round 2: the detector and the closer in one piece ─────────────────

// (B4) Text-fallback vocabulary: a piece that carries a counted line and the
// closer closes the block in the same call. The line is a real repeat, but the
// ladder must not act on it -- a nudge injected then lands in the answer. The
// detector counts the line and the detection; nothing is nudged.
TEST(loop_guard_closer_in_the_same_piece_does_not_nudge_into_the_answer) {
    SamplerParams p = base_params(LoopGuardMode::Nudge, 10, 1);
    p.think_ids.close_ids.clear();            // text fallback only
    p.think_ids.open_close_ids.clear();
    p.loop_guard_nudge_ids = {kPieceFirst};
    Sampler s(p, 1);
    s.start_in_think_block(true);

    feed_lines(s, 0, 9, false);
    for (int i = 9; i < 17; ++i) feed_text(s, loop_line(i));   // 7 repeats = 0.7
    CHECK_EQ(s.loop_guard_counts().detections, 0);

    // The 8th repeat line ends inside a piece that also spells the closer.
    feed_text(s, loop_line(17) + think_tag_pairs(false)[0].second);
    CHECK_EQ(s.loop_guard_counts().detections, 1);   // the reasoning line counts
    CHECK_EQ(s.loop_guard_counts().nudges, 0);       // but no nudge into the answer
    CHECK(!s.think_block_open());
}

// ── 1.2.5 round 2: the slap's closer family, and the completions start ─────

// (B5) Two tag families, a block opened by TEXT for the family whose closer is
// text: the slap must not force the other family's closer id into the output.
TEST(loop_guard_slap_forces_no_closer_from_another_family) {
    SamplerParams p = base_params(LoopGuardMode::Slap, 10, 5);
    const auto& pairs = think_tag_pairs(true);   // Qwen and the wider family
    p.think_ids.open_tags  = {pairs[0].first, pairs[1].first, pairs[2].first};
    p.think_ids.close_tags = {pairs[0].second, pairs[1].second, pairs[2].second};
    // The flattened set: only family 0's closer is one special token.
    p.think_ids.open_close_ids.clear();
    p.think_ids.open_ids.clear();
    p.think_ids.close_ids  = {5};
    p.think_ids.close_tags = {pairs[0].second, pairs[1].second};
    p.think_ids.open_tags  = {pairs[0].first, pairs[1].first};
    p.think_ids.open_close_ids = {{-1, 5}, {-1, -1}};
    Sampler s(p, 1);
    s.start_in_think_block(false);
    feed_text(s, pairs[1].first);   // the block opens by text, family 1
    CHECK(s.think_block_open());

    // The reasoning rounds into a loop: the slap fires.
    for (int i = 0; i < 10; ++i) feed_text(s, loop_line(i));
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    CHECK_EQ(s.loop_guard_counts().slaps, 1);

    // No closer id belongs to the open family, so the slap forces none: the
    // draw is the model's own, and family 0's closer (5) is not injected.
    std::vector<float> row(8, -1.0f);
    row[1] = 4.0f;   // not any closer
    CHECK_EQ(s.sample(row.data(), row.size()), 1);
    CHECK_EQ(row[5], -1.0f);
}

