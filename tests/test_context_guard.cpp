// feyntalk W3.4 — the context guard. One turn can think itself out of its
// context before the between-turn compaction can act; the sampler owns the
// reserve and forces the think-block's closer through the loop guard's own
// forced-closer seam. Synthetic rows, no card, no tokenizer.
//
// Every case here is sampler-level or reporting-level on purpose: the stub
// backend runs no sampler, so the arithmetic and the forced draw are tested
// where they live (src/core/sampler.cpp) and the event/counter plumbing where
// it is mapped (src/api/generate.cpp, responses_stream.cpp).
#include "api/generate.h"
#include "api/handlers.h"
#include "config.h"
#include "core/reasoning.h"
#include "core/sampler.h"
#include "core/sampling.h"
#include "exec/backend.h"
#include "exec/verify_walk.h"
#include "harness.h"

#include <string>
#include <vector>

using namespace lgc;

namespace {

SamplerParams ctx_params() {
    SamplerParams p;
    p.temperature        = 0.0f;
    p.top_k              = 0;
    p.top_p              = 1.0f;
    p.repetition_penalty = 1.0f;
    p.loop_guard         = LoopGuardMode::Off;
    // A vocabulary where the closer is one special token and the template
    // opened the block (the agent's shape). The opener has no id.
    const auto& pairs = think_tag_pairs(false);
    p.think_ids.eog        = {0};
    p.think_ids.close_ids  = {5};
    p.think_ids.close_tags = {pairs[0].second};
    p.think_ids.open_tags  = {pairs[0].first};
    return p;
}

std::vector<float> row_with_argmax(int id, size_t vocab = 8) {
    std::vector<float> r(vocab, -1.0f);
    r[static_cast<size_t>(id)] = 5.0f;
    return r;
}

// A decode step as the plain loop makes it: sample, then emit and observe.
int drive(Sampler& s, int fallback_id = 2, std::string_view piece = {}) {
    std::vector<float> l = row_with_argmax(fallback_id);
    const int id = s.sample(l.data(), l.size());
    s.observe(id, piece);
    return id;
}

ArgParse run(std::vector<const char*> args, Config& cfg) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("arcint"));
    for (const char* a : args) argv.push_back(const_cast<char*>(a));
    return parse_args(static_cast<int>(argv.size()), argv.data(), cfg);
}

class UnusedTokenizer final : public Tokenizer {
public:
    std::vector<int> encode(std::string_view) override { return {}; }
    std::string      decode(const std::vector<int>&) override { return {}; }
    std::string      decode_one(int) override { return {}; }
    int              eos_id() const override { return 0; }
};

// /props reads status() and template_caps(); nothing else is called.
class PropsBackend final : public Backend {
public:
    ModelStatus st;
    const ModelStatus& status() const override { return st; }
    Tokenizer&          tokenizer() override { return tok_; }
    std::string         render_chat(const ChatRequest&) const override { return {}; }
    FinishReason generate(const GenerationInput&, int, const TokenCallback&,
                          GenerationStats&) override {
        return FinishReason::Stop;
    }

private:
    UnusedTokenizer tok_;
};

class AllowOnly final : public TokenConstraint {
public:
    explicit AllowOnly(int only) : only_(only) {}
    bool allows(int token) override { return token == only_; }
    void mask(float* logits, size_t vocab) override {
        for (size_t i = 0; i < vocab; ++i)
            if (i != static_cast<size_t>(only_)) logits[i] = -INFINITY;
    }
    void accept(int) override {}
    int rejections() const { return 0; }

private:
    int only_;
};

}  // namespace

// 1. The budget arithmetic. The sampler gets n_ctx - prompt_tokens - reserve
// at request start and counts it down per EMITTED token inside the block; a
// speculative chain the verify walk accepts is the tokens it observes, so each
// draft counts once, not per proposed row.
TEST(context_guard_budget_arithmetic_counts_accepted_drafts_once) {
    CHECK_EQ(think_ctx_budget(4096, 100, 256), 3740);
    CHECK_EQ(think_ctx_budget(4096, 4000, 256), -160);   // prompt inside the reserve
    CHECK_EQ(think_ctx_budget(4096, 4, 0), 4092);        // arithmetic only: reserve 0 is off

    SamplerParams p = ctx_params();
    p.think_ctx_reserve = 8;
    Sampler s(p, 1);
    s.set_prompt({1, 2, 3, 4});
    s.set_think_ctx_budget(think_ctx_budget(64, 4, 8));   // 52
    s.start_in_think_block(true);
    CHECK_EQ(s.think_ctx_remaining(), 52);

    for (int i = 0; i < 5; ++i) drive(s);
    CHECK_EQ(s.think_ctx_remaining(), 47);   // five emitted reasoning tokens

    // Three accepted drafts in one verify chain: the walk observes each emitted
    // token once, so the budget moves by three, not by the step or the drafts.
    const int rows_n = 3;
    std::vector<float> rows(static_cast<size_t>(rows_n) * 8);
    for (int r = 0; r < rows_n; ++r) {
        std::vector<float> one = row_with_argmax(2);
        for (size_t i = 0; i < one.size(); ++i) rows[static_cast<size_t>(r) * 8 + i] = one[i];
    }
    const std::vector<int> draft = {2, 2, 2};
    const VerifyWalk w = walk_verify(rows.data(), static_cast<size_t>(rows_n), 8, draft, s, 32,
                                     [](int) { return false; }, [](int) { return Control::Continue; });
    CHECK_EQ(w.emitted, 3);
    CHECK_EQ(w.accepted, 3);
    CHECK_EQ(s.think_ctx_remaining(), 44);
    CHECK_EQ(s.loop_guard_counts().ctx_closes, 0);
}

// 2. The forced close fires exactly when the budget reaches 0, under the
// agent's think-block mask (--think-eog-mask). The forced token is returned
// before the mask is applied, so a row whose argmax is an EOG the mask removes
// still yields the closer; the mask never fights it (review B3).
TEST(context_guard_forced_close_fires_at_budget_zero_with_the_think_mask) {
    SamplerParams p = ctx_params();
    p.think_ctx_reserve = 32;
    p.think_eog_mask    = true;
    Sampler s(p, 1);
    s.set_prompt({1});
    s.set_think_ctx_budget(5);
    s.start_in_think_block(true);

    for (int i = 0; i < 5; ++i) {
        // Rows whose argmax is the EOG id 0: the mask kills it while the block
        // is open, so the drawn token is 2. The budget still counts 2.
        std::vector<float> l(8, -1.0f);
        l[0] = 5.0f;
        l[2] = 1.0f;
        const int id = s.sample(l.data(), l.size());
        CHECK_EQ(id, 2);
        s.observe(id);
        CHECK_EQ(s.think_ctx_remaining(), 5 - (i + 1));
    }
    CHECK_EQ(s.loop_guard_counts().ctx_closes, 0);

    // Budget 0: the next draw is the closer id, before the mask touches the row.
    std::vector<float> l(8, -1.0f);
    l[0] = 5.0f;
    l[2] = 1.0f;
    CHECK_EQ(s.sample(l.data(), l.size()), 5);
    CHECK_EQ(s.loop_guard_counts().ctx_closes, 1);
    s.observe(5);
    CHECK(!s.think_block_open());   // the answer flows now
}

// 3. The same forced close with the mask off and the door ladder open (the
// coder's shape). The door's mask is applied inside sample(), after the forced
// return, so a row whose argmax is a door-masked newline still yields the
// closer.
TEST(context_guard_forced_close_fires_with_the_door_ladder_and_no_think_mask) {
    SamplerParams p = ctx_params();
    p.think_ctx_reserve    = 16;
    p.think_eog_mask       = false;
    p.loop_guard           = LoopGuardMode::Ladder;
    p.loop_guard_window    = 2;
    p.loop_guard_threshold = 0.5f;
    p.loop_guard_door_grace = 1000;   // no nudge/slap before the reserve fires
    p.loop_guard_door_ids  = {4};
    Sampler s(p, 1);
    s.set_prompt({1});
    s.set_think_ctx_budget(4);
    s.start_in_think_block(true);

    const std::string line = "reasoning line: the same plan again and again, unchanged\n";
    s.observe(1, line);   // first counted line
    s.observe(1, line);   // second: share 1.0, the door opens
    CHECK_EQ(s.loop_guard_counts().detections, 1);
    CHECK_EQ(s.think_ctx_remaining(), 2);

    s.observe(2);
    s.observe(2);
    CHECK_EQ(s.think_ctx_remaining(), 0);

    std::vector<float> l(8, -1.0f);
    l[4] = 5.0f;   // the door would mask this newline token
    l[2] = 1.0f;
    CHECK_EQ(s.sample(l.data(), l.size()), 5);
    CHECK_EQ(s.loop_guard_counts().ctx_closes, 1);
    CHECK_EQ(s.loop_guard_counts().slaps, 0);   // no loop rung fired
    s.observe(5, "closed");
    CHECK_EQ(s.loop_guard_counts().self_closes, 0);   // a forced close is not a self-close
}

// 4. After a forced close with --tool-grammar, the answer is grammatical: the
// constraint was dormant while the block was open and is live once the closer
// has closed it, so a tool call inside the reserve still parses.
TEST(context_guard_grammar_constrains_the_answer_after_the_forced_close) {
    SamplerParams p = ctx_params();
    p.think_ctx_reserve = 8;
    p.tool_grammar      = true;
    Sampler s(p, 1);
    s.set_prompt({1});
    s.set_think_ctx_budget(3);
    s.start_in_think_block(true);

    AllowOnly grammar(7);
    s.set_constraint(&grammar);
    s.observe(2);
    s.observe(2);
    // While thinking the grammar is dormant, so the row's own argmax comes.
    std::vector<float> l = row_with_argmax(6);
    const int dormant = s.sample(l.data(), l.size());
    CHECK_EQ(dormant, 6);
    s.observe(dormant);   // the budget reaches 0: the next draw is forced

    std::vector<float> l2 = row_with_argmax(2);
    const int closer = s.sample(l2.data(), l2.size());
    CHECK_EQ(closer, 5);
    CHECK_EQ(s.loop_guard_counts().ctx_closes, 1);
    s.observe(closer);   // content: the grammar accepts the closer and triggers

    // The answer row's argmax (6) is outside the grammar; the sampler redraws
    // into the allowed id, exactly as before the guard existed.
    std::vector<float> l3 = row_with_argmax(6);
    CHECK_EQ(s.sample(l3.data(), l3.size()), 7);
}

// 5. A text-closer family has no closer id to force. The guard copies the
// slap's behaviour there: no forced token, one counter and one warn saying so.
TEST(context_guard_text_closer_family_forces_nothing_and_says_so) {
    SamplerParams p = ctx_params();
    p.think_ctx_reserve = 8;
    p.think_ids.close_ids.clear();   // the closer is text, not one id
    p.think_ids.open_close_ids.clear();
    Sampler s(p, 1);
    s.set_prompt({1});
    s.set_think_ctx_budget(2);
    s.start_in_think_block(true);

    s.observe(2);
    s.observe(2);
    CHECK_EQ(s.think_ctx_remaining(), 0);
    const int id = drive(s);   // no closer to force: the draw stands
    CHECK_EQ(id, 2);
    CHECK_EQ(s.loop_guard_counts().ctx_closes, 0);
    CHECK_EQ(s.loop_guard_counts().ctx_no_closer, 1);
    CHECK(s.think_block_open());   // the block is still open; the guard gave up
}

// 6. A prompt already inside the reserve (n_ctx - prompt_tokens <= reserve):
// the closer is forced at the first thinking token, so thinking is skipped and
// the model answers inside what is left. No refusal.
TEST(context_guard_prompt_inside_the_reserve_forces_at_the_first_token) {
    SamplerParams p = ctx_params();
    p.think_ctx_reserve = 256;
    Sampler s(p, 1);
    s.set_prompt(std::vector<int>(200, 1));
    s.set_think_ctx_budget(think_ctx_budget(256, 200, 256));   // -200
    s.start_in_think_block(true);

    std::vector<float> l = row_with_argmax(2);
    const int first = s.sample(l.data(), l.size());
    CHECK_EQ(first, 5);   // the closer, no thinking token emitted
    CHECK_EQ(s.loop_guard_counts().ctx_closes, 1);
    CHECK_EQ(s.loop_guard_counts().ctx_forced_index, 0);
    s.observe(5);
    CHECK(!s.think_block_open());
}

// 7. Reserve 0 is the switch off: a handed budget is ignored (never stored),
// no counter moves, no token is forced -- today's behaviour exactly.
TEST(context_guard_reserve_zero_is_off) {
    CHECK_EQ(think_ctx_budget(256, 10, 0), 246);
    SamplerParams p = ctx_params();
    p.think_ctx_reserve = 0;
    p.think_eog_mask    = true;   // the agent's shape, so the block is tracked
    Sampler s(p, 1);
    s.set_prompt({1});
    s.set_think_ctx_budget(0);   // ignored: reserve 0 is off
    s.start_in_think_block(true);
    CHECK_EQ(s.think_ctx_remaining(), -1);
    for (int i = 0; i < 50; ++i) CHECK_EQ(drive(s), 2);
    CHECK_EQ(s.loop_guard_counts().ctx_closes, 0);
    CHECK_EQ(s.loop_guard_counts().ctx_no_closer, 0);
    CHECK(s.think_block_open());
}

// 8. The counters reach the stats consumers: the loop-guard object (chat and
// Responses) and the slot line. The chat route has no context event.
TEST(context_guard_counters_reach_the_object_and_the_line) {
    GenerationStats st;
    st.loop_context_closes    = 1;
    st.loop_context_no_closer = 0;
    st.loop_context_token_index = 17;
    const nlohmann::json j = api::loop_guard_json(st);
    CHECK(j.contains("context_closes"));
    if (j.contains("context_closes")) CHECK_EQ(j.at("context_closes").get<int>(), 1);
    CHECK(j.contains("context_no_closer"));
    if (j.contains("context_no_closer")) CHECK_EQ(j.at("context_no_closer").get<int>(), 0);
    const std::string line = api::decode_suffix_for(st);
    CHECK(line.find("context guard") != std::string::npos);
    CHECK(line.find("closed 1") != std::string::npos);

    GenerationStats plain;
    CHECK(api::decode_suffix_for(plain).find("context guard") == std::string::npos);
}

// 9. The operator flag parses, validates, travels into the sampler defaults and
// is named in /props.
TEST(context_guard_flag_parses_and_reaches_props) {
    Config cfg;
    std::vector<const char*> args = {"--stub", "--think-ctx-reserve", "256"};
    CHECK(run(args, cfg).ok);
    CHECK_EQ(cfg.think_ctx_reserve, 256);

    Config plain;
    CHECK(run({"--stub"}, plain).ok);
    CHECK_EQ(plain.think_ctx_reserve, 0);

    Config bad;
    CHECK(!run({"--stub", "--think-ctx-reserve", "-1"}, bad).ok);

    SamplerDefaults d;
    CHECK(!apply_operator_defaults(cfg, d).has_value());
    CHECK_EQ(d.think_ctx_reserve, 256);
    CHECK(d.provenance == "operator");

    PropsBackend backend;
    backend.st.id        = "model.gguf";
    backend.st.served_id = backend.st.id;
    backend.st.loaded    = true;
    backend.st.n_ctx     = 4096;
    backend.st.sampler_defaults = d;   // what the load path serves
    api::SlotPool slots(1);
    api::Context  ctx{&cfg, &backend, &slots};
    const nlohmann::json p = api::props(ctx);
    const nlohmann::json sd = p.at("sampler_defaults");
    CHECK(sd.contains("think_ctx_reserve"));
    if (sd.contains("think_ctx_reserve")) CHECK_EQ(sd.at("think_ctx_reserve").get<int>(), 256);
}
