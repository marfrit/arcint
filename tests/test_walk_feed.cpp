// The verify walk (speculative decoding: --llama-mtp, DFlash2) feeds the
// sampler exactly as the plain decode loop does: one observation, the id with
// the text it decoded to, per emitted token. The walk and the plain loop run
// over the same scripted rows with --loop-guard ladder, a thinking block long
// enough that the door opens and its budget advances by token stretches up to
// a nudge and a slap, and the context guard armed; at every walk round's end
// everything observe() feeds must equal the plain loop's state after the same
// number of tokens: the history counts, the think state, the context guard,
// the door's budget counters and the token k-gram window.
#include "core/reasoning.h"
#include "core/sampler.h"
#include "exec/verify_walk.h"
#include "harness.h"

#include <algorithm>
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

std::string piece_of(int id) {
    if (id == kCloser) return think_tag_pairs(false)[0].second;
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

std::vector<float> row_for(int id) {
    std::vector<float> r(kVocab, -1.0f);
    r[static_cast<size_t>(id)] = 5.0f;
    return r;
}

Sampler make_sampler() {
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
    Sampler s(p, 1);
    s.set_prompt({10, 11, 40, 41});
    s.set_think_ctx_budget(5000);
    s.start_in_think_block(true);
    return s;
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
    WF_SCALAR(counts.ctx_closes)
    WF_SCALAR(counts.ctx_forced_index)
#undef WF_SCALAR
#undef WF_SEQ
    return d.str();
}

constexpr size_t kTokens = 420;   // past the slap: the answer's first tokens too

// The plain loop (backend_llama.cpp generate): sample, emit, observe(id, piece).
struct Plain {
    std::vector<int>   out;
    std::vector<State> after;   // after[k]: the state after k + 1 tokens
};
Plain run_plain() {
    Sampler s = make_sampler();
    Plain   r;
    while (r.out.size() < kTokens) {
        std::vector<float> row = row_for(script(r.out.size()));
        const int          tok = s.sample(row.data(), kVocab);
        if (s.loop_abort() || tok == kEog) break;
        const std::string piece = piece_of(tok);
        r.out.push_back(tok);
        s.observe(tok, piece);
        r.after.push_back(s.observed_state_for_test());
    }
    return r;
}

}  // namespace

TEST(verify_walk_feeds_the_sampler_once_per_token_like_the_plain_loop) {
    const Plain plain = run_plain();
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

    // The walk, round by round, widths 1..5 so the round ends fall on most
    // positions; the rows are the script's, the drafts what the rows say, so
    // every draft is accepted unless the guard forces a token.
    Sampler          s = make_sampler();
    std::vector<int> out;
    int              round = 0, compared = 0, door_compared = 0;
    std::string      first_diff;
    while (out.size() < kTokens) {
        const size_t       width = std::min<size_t>(1 + static_cast<size_t>(round++ % 5), kTokens - out.size());
        std::vector<float> rows;
        std::vector<int>   draft;
        for (size_t i = 0; i < width; ++i) {
            const std::vector<float> r = row_for(script(out.size() + i));
            rows.insert(rows.end(), r.begin(), r.end());
            if (i + 1 < width) draft.push_back(script(out.size() + i));
        }
        // the backend's emit (generate_spec_lane): decode the piece, hand it on
        const VerifyWalk w = walk_verify(
            rows.data(), width, kVocab, draft, s, static_cast<int>(width), [](int t) { return t == kEog; },
            [&](int tok, std::string& piece) {
                piece = piece_of(tok);
                out.push_back(tok);
                return Control::Continue;
            });
        if (out.empty()) break;
        const State now = s.observed_state_for_test();
        const State& ref = plain.after[std::min(out.size(), plain.after.size()) - 1];
        const std::string d = diff(now, ref);
        if (!d.empty() && first_diff.empty())
            first_diff = "after " + std::to_string(out.size()) + " tokens (walk/plain):" + d;
        ++compared;
        if (now.door_open) ++door_compared;
        if (w.done || w.emitted == 0) break;
    }
    CHECK_EQ(out, plain.out);
    CHECK_EQ(first_diff, std::string());
    // the served -vv think-tracker line's numbers: one tracker call a token
    const Sampler::ThinkFeedCounts f = s.think_feed_counts();
    CHECK_EQ(f.tracker_calls, static_cast<int>(out.size()));
    CHECK_EQ(f.emitted, static_cast<int>(out.size()));
    CHECK(f.thinking > 300 && f.thinking < f.emitted);   // the block closed at the slap
    CHECK(compared > 100);
    CHECK(door_compared > 50);
}
