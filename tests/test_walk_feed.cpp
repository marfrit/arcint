// The verify walk (speculative decoding: --llama-mtp, DFlash2) feeds the
// sampler exactly as the plain decode loop does: one observation, the id with
// the text it decoded to, per emitted token. The walk and the plain loop run
// over the same scripted rows with --loop-guard ladder, a thinking block long
// enough that the door opens and its budget advances by token stretches up to
// a nudge and a slap, and the context guard armed; at every walk round's end
// everything observe() feeds must equal the plain loop's state after the same
// number of tokens: the history counts, the think state, the context guard,
// the door's budget counters and the token k-gram window. A second case runs a
// text-closer vocabulary (no closer id: the pieces decide the block), the
// closer drawn by the model at an interior row of a span, and an answer
// presence penalty, so the answer's own history is live.
#include "core/reasoning.h"
#include "core/sampler.h"
#include "exec/verify_walk.h"
#include "harness.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

using namespace lgc;

namespace {

constexpr size_t kVocab      = 128;
constexpr int    kEog        = 0;
constexpr int    kCloser     = 5;
constexpr int    kNudgeFirst = 101;
const std::vector<std::string> kNudge = {"I", " am", " repeating", " myself.", " I", " will", " decide",
                                         " now", " and", " write", " the", " final", " answer."};

// distinct reasoning lines (ids 10..18): long enough to count, varied enough
// that neither the exact nor the near rule reads them as repeats
std::string word(int i, int salt) {
    uint32_t    x = static_cast<uint32_t>((i + 1) * 2654435761u + salt * 40503u);
    std::string out;
    for (int k = 0; k < 4; ++k) {
        out += static_cast<char>('a' + x % 26);
        x /= 26;
    }
    return out;
}
std::string distinct_line(int i) {
    return "step " + std::to_string(i) + ": " + word(i, 1) + " asks " + word(i, 2) + " or " + word(i, 3) +
           " beside " + word(i, 4) + "\n";
}

// The text case: the closer spelled by two ordinary tokens (50, 51), which the
// sampler knows by text only; newline-free reasoning words 60..69; answer
// words 70..72, repeated, which the answer's presence penalty reads.
std::string close_tag() { return think_tag_pairs(false)[0].second; }

std::string piece_of(int id) {
    if (id == kCloser) return close_tag();
    if (id == 50) return close_tag().substr(0, close_tag().size() / 2);
    if (id == 51) return close_tag().substr(close_tag().size() / 2);
    if (id >= 60 && id <= 72) return " t" + std::to_string(id);
    if (id >= 10 && id <= 18) return distinct_line(id - 10);
    if (id == 19) return "reasoning line: the same plan again and again and again, unchanged\n";
    if (id >= 20 && id <= 24) return " w" + std::to_string(id);   // a newline-free cycle
    const size_t n = static_cast<size_t>(id - kNudgeFirst);
    if (id >= kNudgeFirst && n < kNudge.size()) return kNudge[n];
    return {};
}

// What the model's row says at the k-th emitted token: nine distinct lines,
// nine copies of one line (the detection, the door opens), then a newline-free
// cycle of five ids, which the door's budget sees only as token stretches.
int script(size_t k) {
    if (k < 9) return 10 + static_cast<int>(k);
    if (k < 18) return 19;
    return 20 + static_cast<int>(k % 5);
}

// Rounds of widths 1..5 end at 1, 3, 6, 10, 15, 16, 18, 21, 25, 30: the
// closer at 26-27 sits inside the span 25..29, with answer tokens after it.
int script_text(size_t k) {
    if (k < 26) return 60 + static_cast<int>(k % 10);
    if (k == 26) return 50;
    if (k == 27) return 51;
    return 70 + static_cast<int>(k % 3);
}

std::vector<float> row_for(int id) {
    std::vector<float> r(kVocab, -1.0f);
    r[static_cast<size_t>(id)] = 5.0f;
    return r;
}

SamplerParams base_params() {
    SamplerParams p;
    p.temperature          = 0.0f;
    p.top_k                = 0;
    p.top_p                = 1.0f;
    p.repetition_penalty   = 1.0f;
    p.loop_guard           = LoopGuardMode::Ladder;
    p.loop_guard_window    = 10;
    p.loop_guard_threshold = 0.8f;
    p.loop_guard_grace     = 2;
    p.loop_guard_door_grace = 3;
    p.loop_guard_door_ids   = {6, 7};
    for (size_t i = 0; i < kNudge.size(); ++i) p.loop_guard_nudge_ids.push_back(kNudgeFirst + static_cast<int>(i));
    p.think_ctx_reserve    = 1;   // arms the context guard's countdown
    const auto& pairs      = think_tag_pairs(false);
    p.think_ids.eog        = {kEog};
    p.think_ids.close_ids  = {kCloser};
    p.think_ids.close_tags = {pairs[0].second};
    p.think_ids.open_tags  = {pairs[0].first};
    return p;
}

Sampler start(const SamplerParams& p, int ctx_budget) {
    Sampler s(p, 1);
    s.set_prompt({10, 11, 40, 41, 70});
    s.set_think_ctx_budget(ctx_budget);
    s.start_in_think_block(true);
    return s;
}

Sampler make_sampler() { return start(base_params(), 5000); }

// The text case: no closer id, an answer row with a presence penalty, and a
// context guard whose budget runs out inside the block (a text family has no
// closer to force: ctx_no_closer).
Sampler make_text_sampler() {
    SamplerParams p           = base_params();
    p.think_ids.close_ids     = {};
    p.answer_presence_penalty = 0.5f;
    p.thinking_turn           = true;
    return start(p, 10);
}

using State = Sampler::ObservedStateForTest;

// The fields that differ, with both values where they print.
std::string diff(const State& a, const State& b) {
    std::ostringstream d;
#define WF_SCALAR(f) \
    if (!(a.f == b.f)) d << " " #f "=" << a.f << "/" << b.f;
#define WF_SEQ(f) \
    if (!(a.f == b.f)) d << " " #f "(size " << a.f.size() << "/" << b.f.size() << ")";
    WF_SCALAR(emitted)
    WF_SEQ(seen)
    WF_SEQ(generated)
    WF_SEQ(answer_generated)
    WF_SCALAR(think_state)
    WF_SCALAR(think_block_ever_open)
    WF_SCALAR(ctx_budget)
    WF_SCALAR(ctx_pending)
    WF_SCALAR(ctx_settled)
    WF_SCALAR(door_open)
    WF_SCALAR(door_opened)
    WF_SCALAR(door_lifted)
    WF_SCALAR(door_tokens)
    WF_SCALAR(since_detect)
    WF_SCALAR(since_nudge)
    WF_SEQ(token_ids)
    WF_SEQ(token_grams)
    WF_SCALAR(token_pos)
    WF_SCALAR(token_positions)
    WF_SCALAR(token_repeats)
    WF_SCALAR(tokens_seen)
    WF_SEQ(open_line)
    WF_SEQ(window_lines)
    WF_SCALAR(counts.detections)
    WF_SCALAR(counts.nudges)
    WF_SCALAR(counts.slaps)
    WF_SCALAR(counts.aborts)
    WF_SCALAR(counts.self_closes)
    WF_SCALAR(counts.door_masked)
    WF_SCALAR(counts.door_eog_masked)
    WF_SCALAR(counts.ctx_closes)
    WF_SCALAR(counts.ctx_no_closer)
    WF_SCALAR(counts.think_closes)
    WF_SCALAR(counts.ctx_forced_index)
#undef WF_SCALAR
#undef WF_SEQ
    return d.str();
}

constexpr size_t kTokens = 420;   // past the slap: the answer's first tokens too

struct Case {
    std::function<Sampler()>   make;
    std::function<int(size_t)> script;
    size_t                     tokens;
};

// The plain loop (backend_llama.cpp generate): sample, emit, observe(id, piece).
struct Plain {
    std::vector<int>   out;
    std::vector<State> after;   // after[k]: the state after k + 1 tokens
};
Plain run_plain(const Case& c) {
    Sampler s = c.make();
    Plain   r;
    while (r.out.size() < c.tokens) {
        std::vector<float> row = row_for(c.script(r.out.size()));
        const int          tok = s.sample(row.data(), kVocab);
        if (s.loop_abort() || tok == kEog) break;
        const std::string piece = piece_of(tok);
        r.out.push_back(tok);
        s.observe(tok, piece);
        r.after.push_back(s.observed_state_for_test());
    }
    return r;
}

// The walk, round by round, widths 1..5 so the round ends fall on most
// positions; the rows are the script's, the drafts what the rows say, so every
// draft is accepted unless the guard forces a token. Each round's end is held
// against the plain loop's state after as many tokens.
struct Walked {
    std::vector<int>         out;
    std::string              first_diff;
    int                      compared = 0, door_compared = 0;
    Sampler::ThinkFeedCounts feed;
};
Walked run_walk(const Case& c, const Plain& plain) {
    Sampler s = c.make();
    Walked  r;
    int     round = 0;
    while (r.out.size() < c.tokens) {
        const size_t       width = std::min<size_t>(1 + static_cast<size_t>(round++ % 5), c.tokens - r.out.size());
        std::vector<float> rows;
        std::vector<int>   draft;
        for (size_t i = 0; i < width; ++i) {
            const std::vector<float> row = row_for(c.script(r.out.size() + i));
            rows.insert(rows.end(), row.begin(), row.end());
            if (i + 1 < width) draft.push_back(c.script(r.out.size() + i));
        }
        // the backend's emit (generate_spec_lane): decode the piece, hand it on
        const VerifyWalk w = walk_verify(
            rows.data(), width, kVocab, draft, s, static_cast<int>(width), [](int t) { return t == kEog; },
            [&](int tok, std::string& piece) {
                piece = piece_of(tok);
                r.out.push_back(tok);
                return Control::Continue;
            });
        if (r.out.empty() || plain.after.empty()) break;
        const State       now = s.observed_state_for_test();
        const std::string d   = diff(now, plain.after[std::min(r.out.size(), plain.after.size()) - 1]);
        if (!d.empty() && r.first_diff.empty())
            r.first_diff = "after " + std::to_string(r.out.size()) + " tokens (walk/plain):" + d;
        ++r.compared;
        if (now.door_open) ++r.door_compared;
        if (w.done || w.emitted == 0) break;
    }
    r.feed = s.think_feed_counts();
    return r;
}

}  // namespace

TEST(verify_walk_feeds_the_sampler_once_per_token_like_the_plain_loop) {
    const Case  c{ make_sampler, script, kTokens };
    const Plain plain = run_plain(c);
    // The scenario is the one the test is about: the door opened, its budget
    // advanced by token stretches, the ladder nudged and slapped, the block
    // closed, and the walk below has answer tokens to compare too.
    CHECK_EQ(plain.out.size(), kTokens);
    if (plain.out.size() != kTokens) return;
    const State& end = plain.after.back();
    CHECK(end.door_opened);
    CHECK_EQ(end.counts.detections, 1);
    CHECK_EQ(end.counts.nudges, 1);
    CHECK_EQ(end.counts.slaps, 1);
    CHECK_EQ(end.think_state, 2);   // content: the slap's closer landed
    CHECK(end.ctx_budget < 5000 - 300);

    const Walked walk = run_walk(c, plain);
    CHECK_EQ(walk.out, plain.out);
    CHECK_EQ(walk.first_diff, std::string());
    // the served -vv think-tracker line's numbers: one tracker call a token
    CHECK_EQ(walk.feed.tracker_calls, static_cast<int>(walk.out.size()));
    CHECK_EQ(walk.feed.emitted, static_cast<int>(walk.out.size()));
    CHECK(walk.feed.thinking > 300 && walk.feed.thinking < walk.feed.emitted);   // the block closed at the slap
    CHECK(walk.compared > 100);
    CHECK(walk.door_compared > 50);
}

TEST(verify_walk_feeds_a_text_closer_drawn_mid_span_like_the_plain_loop) {
    const Case  c{ make_text_sampler, script_text, 60 };
    const Plain plain = run_plain(c);
    CHECK_EQ(plain.out.size(), c.tokens);
    if (plain.out.size() != c.tokens) return;
    // the text closer closed the block at its second token, the answer's
    // history holds the answer words only, the context guard ran out inside
    // the block with no closer id to force
    CHECK_EQ(plain.after[26].think_state, 1);   // thinking: half a closer
    CHECK_EQ(plain.after[27].think_state, 2);   // content
    CHECK_EQ(plain.after.back().answer_generated.size(), 3u);
    CHECK_EQ(plain.after.back().counts.ctx_no_closer, 1);

    const Walked walk = run_walk(c, plain);
    CHECK_EQ(walk.out, plain.out);
    CHECK_EQ(walk.first_diff, std::string());
    CHECK_EQ(walk.feed.tracker_calls, static_cast<int>(walk.out.size()));
    CHECK_EQ(walk.feed.thinking, 27);   // the closer's second token is the answer's edge
    CHECK(walk.compared > 15);
}
