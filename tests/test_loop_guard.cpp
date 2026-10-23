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

// A line that is long enough to count, with a distinct tail so a non-looping
// stream stays varied.
std::string distinct_line(int i) {
    return "reasoning line " + std::to_string(i) + ": checking a different branch of the problem now\n";
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
    SamplerParams p = base_params(LoopGuardMode::Ladder);
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

// 3. Ladder: the nudge text appears once in the reasoning, and after K more
// looping lines the next drawn token is the closer id. Afterwards content flows.
TEST(loop_guard_ladder_nudges_once_then_forces_the_closer) {
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 5);
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
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 40);
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
    SamplerParams p = base_params(LoopGuardMode::Ladder, 10, 2);
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
    CHECK_EQ(d2.loop_guard_window, 200);
    CHECK(d2.provenance == "provisional");

    // Bad values are refused at boot.
    for (std::vector<std::string> bad : {std::vector<std::string>{"--stub", "--loop-guard", "sometimes"},
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
