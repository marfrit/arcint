// The MTP draft-depth cutoff (--llama-mtp-depth-max, src/exec/spec_depth.h):
// the round's draft width as the serving loop computes it, and the property
// the cutoff is allowed to have at greedy -- with no draft proposed, the
// emitted token stream is the drafting arm's.
//
// The loop site (`generate_spec_lane`, src/exec/backend_llama.cpp) hands the
// helper the round's depth `pos0 + 1` (the lane's slot tokens at the round's
// start, scope: the round, not the request), so a later, shorter request on
// the same slot drafts again.
#include "config.h"
#include "exec/spec_depth.h"
#include "exec/verify_walk.h"
#include "harness.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lgc;

namespace {

ArgParse parse(std::vector<const char*> args, Config& cfg) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("arcint"));
    for (const char* a : args) argv.push_back(const_cast<char*>(a));
    return parse_args(static_cast<int>(argv.size()), argv.data(), cfg);
}

constexpr size_t kVocab = 16;

// The target's own greedy pick at position k -- a deterministic sequence, so
// the two arms of a run compare tokens and not a random draw.
int target_at(size_t k) { return static_cast<int>((k * 7 + 3) % kVocab); }

// Rows whose argmax is `target_at(k)`: row i is the target's pick at the
// position draft i stands for, so row 0 is the token after the round's
// id_last and row i is the token the walk compares draft i-1 against.
std::vector<float> rows_at(size_t emitted, size_t n_rows) {
    std::vector<float> r(n_rows * kVocab);
    for (size_t i = 0; i < n_rows; ++i)
        for (size_t v = 0; v < kVocab; ++v)
            r[i * kVocab + v] = static_cast<int>(v) == target_at(emitted + i) ? 4.0f : 0.1f * static_cast<float>(v % 5);
    return r;
}

SamplerParams greedy() {
    SamplerParams p;
    p.temperature        = 0.0f;
    p.repetition_penalty = 1.0f;
    return p;
}

struct Arm {
    std::vector<int> out;
    int              proposed = 0;   // draft tokens handed to the verify
    int              rounds   = 0;
};

// The serving loop's round arithmetic, without a context: a lane whose slot
// holds `prompt_len` tokens and emits greedily until `gen` tokens exist. The
// oracle drafter proposes the target's own pick at every second draft and a
// wrong token at the others, so the walk has to reject drafts from the target's
// row alone -- what a cutoff arm emits cannot depend on where a draft came from.
Arm run(int depth_max, size_t prompt_len, int gen, int n_draft, int budget) {
    Arm a;
    size_t emitted = 0;
    while (emitted < static_cast<size_t>(gen)) {
        // the loop's `pos0` is have.size(); the depth the context would hold
        // once id_last stands in it is one more
        const size_t depth = prompt_len + emitted;
        const int    room  = std::min(budget, gen - static_cast<int>(emitted));
        const int    n_max = spec_width(n_draft, room, depth_max, depth);
        std::vector<int> draft;
        for (int i = 0; i < n_max; ++i) {
            const int truth = target_at(emitted + static_cast<size_t>(i));
            draft.push_back(i % 2 == 1 ? (truth + 5) % static_cast<int>(kVocab) : truth);
        }
        a.proposed += static_cast<int>(draft.size());
        ++a.rounds;
        Sampler sampler(greedy(), 1);
        VerifyWalk w;
        std::vector<float> rows = rows_at(emitted, static_cast<size_t>(n_max) + 1);
        w = walk_verify(rows.data(), static_cast<size_t>(n_max) + 1, kVocab, draft, sampler, room,
                        [](int) { return false; },
                        [&](int tok) {
                            a.out.push_back(tok);
                            return Control::Continue;
                        });
        emitted += static_cast<size_t>(w.emitted);
        if (w.done) break;
    }
    return a;
}

}  // namespace

// Cutoff 0 is today: the width is the ceiling clipped by the emit budget, the
// same integer the loop computed before the flag existed.
TEST(mtp_depth_off_is_the_width_as_before) {
    const int ceil[]   = {0, 1, 3, 5, 7};
    const int budgets[] = {-2, 0, 1, 2, 4, 8, 64};
    for (int c : ceil)
        for (int b : budgets)
            for (size_t d : {size_t(0), size_t(1), size_t(4096), size_t(212992)}) {
                const int want = std::max(0, std::min(c, b - 1));
                CHECK_EQ(spec_width(c, b, 0, d), want);
            }
}

TEST(mtp_depth_below_the_prompt_proposes_nothing) {
    // a 900-token prompt, cutoff 512: no round ever drafts
    const Arm a = run(512, 900, 40, 5, 1000);
    CHECK_EQ(a.proposed, 0);
    CHECK_EQ(a.out.size(), size_t(40));   // one token a round, no stop
}

TEST(mtp_depth_at_the_cutoff_still_drafts_one_past_it_does_not) {
    CHECK(spec_drafting_at(512, 512));   // prompt.n_tokens() > N is the reference's rule
    CHECK(!spec_drafting_at(513, 512));
    // 500-token prompt, 5 drafts: the second draft is wrong, so a drafting
    // round emits two tokens and the depth grows by 2; the rounds at depth
    // 500, 502, .., 512 draft (a 6-row verify each), the next one does not
    const Arm a = run(512, 500, 40, 5, 1000);
    CHECK_EQ(a.proposed, 7 * 5);         // 512 - 500 in steps of 2: seven rounds
    CHECK_EQ(a.out.size(), size_t(40));
}

TEST(mtp_depth_reenters_on_a_shorter_request_same_slot) {
    // one slot, two requests: the long one is all above the cutoff, the short
    // one below it. The decision is recomputed from the round's depth, so
    // drafting comes back with the second request.
    const Arm long_req  = run(512, 900, 40, 5, 1000);
    const Arm short_req = run(512, 64, 40, 5, 1000);
    CHECK_EQ(long_req.proposed, 0);
    CHECK(short_req.proposed > 0);
    // and the slot's second request is not drafted "off" by the first one's
    CHECK(short_req.rounds < long_req.rounds);   // 6 rows a round against 1
}

TEST(mtp_depth_greedy_stream_is_the_arms_with_drafts) {
    const Arm off   = run(0, 900, 40, 5, 1000);     // today: 6-row verifies
    const Arm below = run(512, 900, 40, 5, 1000);   // cutoff under the prompt: 1-row
    CHECK(off.proposed > 0);
    CHECK_EQ(below.proposed, 0);
    CHECK_EQ(below.out.size(), size_t(40));   // one token a round, no stop
    CHECK(off.out == below.out);   // the target's own picks, draft or not
}

// The loop's depth is the round's live slot count, not a request's prompt
// length or a cached flag: the cutoff has to re-enter on a shorter request.
// source scan, as /props' KV tripwire is (the loop needs a card).
TEST(mtp_depth_reads_the_round_depth_at_the_loop_site) {
    std::ifstream in(std::string(ARCINT_SOURCE_DIR) + "/src/exec/backend_llama.cpp");
    CHECK(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();
    CHECK(src.find("spec_width(n_draft_, budget, mtp_depth_max_, pos0 + 1)") != std::string::npos);
}

// The flag: 0 default (today), a token count past --llama-mtp, and its refusals
TEST(mtp_depth_flag_defaults_off_and_needs_mtp) {
    Config cfg;
    CHECK(parse({"--gguf", "/m/q.gguf", "--llama-mtp", "5"}, cfg).ok);
    CHECK_EQ(cfg.llama_mtp_depth_max, 0);
    Config c2;
    CHECK(parse({"--gguf", "/m/q.gguf", "--llama-mtp", "5", "--llama-mtp-depth-max", "32768"}, c2).ok);
    CHECK_EQ(c2.llama_mtp_depth_max, 32768);
    Config c3;
    CHECK(!parse({"--gguf", "/m/q.gguf", "--llama-mtp", "5", "--llama-mtp-depth-max", "-1"}, c3).ok);
    Config c4;
    CHECK(!parse({"--gguf", "/m/q.gguf", "--llama-mtp", "5", "--llama-mtp-depth-max", "x"}, c4).ok);
    Config c5;
    CHECK(!parse({"--gguf", "/m/q.gguf", "--llama-mtp-depth-max", "32768"}, c5).ok);   // needs the drafter
}
