// Request fields that reach the generation input rather than stopping at the
// parser: cache_prompt on both completion surfaces, n_predict's mapping onto
// max_tokens, top_logprobs 0, and arcint.timings on a streamed completion.
// These pass as written today; each fails under the mutation named beside it.
#include "api/handlers.h"

#include "core/chat.h"
#include "exec/backend.h"
#include "harness.h"

#include <memory>
#include <string>

using namespace lgc;
using nlohmann::json;

namespace {

// The stub with generate() captured: what the route hands the backend, not
// what the parser stored. Delegates the reads prepare_* needs.
class CaptureBackend : public Backend {
public:
    explicit CaptureBackend(std::unique_ptr<Backend> inner) : inner_(std::move(inner)) {}

    const ModelStatus& status() const override { return inner_->status(); }
    Tokenizer&         tokenizer() override { return inner_->tokenizer(); }
    std::string        render_chat(const ChatRequest& req) const override {
        return inner_->render_chat(req);
    }
    nlohmann::json template_caps() const override { return inner_->template_caps(); }

    FinishReason generate(const GenerationInput& in, int, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        (void)on_piece;
        saw_          = true;
        cache_prompt_ = in.cache_prompt;
        stats.prompt_tokens     = 1;
        stats.completion_tokens = 0;
        return FinishReason::Stop;
    }

    bool saw_          = false;
    bool cache_prompt_ = true;

private:
    std::unique_ptr<Backend> inner_;
};

struct Env {
    Config                   cfg;
    std::unique_ptr<Backend> inner;
    CaptureBackend*          capture = nullptr;
    api::SlotPool            slots{1};
    api::Context             ctx;

    Env() : inner(make_stub_backend(4096, 0, {}, {}, {}, 1)) {
        capture = new CaptureBackend(std::move(inner));
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = capture;
        ctx.slots   = &slots;
    }
    ~Env() { delete capture; }
};

json chat_body(bool cache_prompt) {
    return json{{"model", "stub"},
                {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                {"cache_prompt", cache_prompt},
                {"max_tokens", 2}};
}

json completion_body(bool cache_prompt) {
    return json{{"model", "stub"}, {"prompt", "hi"}, {"cache_prompt", cache_prompt},
                {"max_tokens", 2}};
}

}  // namespace

TEST(request_shapes_cache_prompt_reaches_generation_input_on_chat) {
    // Mutation: drop `prep.input.cache_prompt = req.cache_prompt;` from
    // prepare_chat (the input keeps its default true) -- the false case fails.
    Env env;
    for (bool cp : {false, true}) {
        api::PreparedChat prep;
        CHECK(!api::prepare_chat(env.ctx, chat_body(cp), prep).has_value());
        env.capture->saw_ = false;
        const auto lease = env.slots.acquire_for(0);
        CHECK_EQ(api::run_chat(env.ctx, prep, lease.index()).status, 200);
        CHECK(env.capture->saw_);
        CHECK_EQ(env.capture->cache_prompt_, cp);
    }
}

TEST(request_shapes_cache_prompt_reaches_generation_input_on_completions) {
    // Mutation: drop the same assignment from prepare_completion.
    Env env;
    for (bool cp : {false, true}) {
        api::PreparedCompletion prep;
        CHECK(!api::prepare_completion(env.ctx, completion_body(cp), prep).has_value());
        env.capture->saw_ = false;
        const auto lease = env.slots.acquire_for(0);
        CHECK_EQ(api::run_completion(env.ctx, prep, lease.index()).status, 200);
        CHECK(env.capture->saw_);
        CHECK_EQ(env.capture->cache_prompt_, cp);
    }
}

TEST(request_shapes_n_predict_maps_to_max_tokens_and_explicit_max_wins) {
    // Mutation: assign n_predict unconditionally (`s.max_tokens = v`) -- the
    // both-given case fails; or never assign -- the n_predict-only case fails.
    CompletionRequest c;
    CHECK(!parse_completion_request(json{{"model", "stub"}, {"prompt", "hi"}, {"n_predict", 8}}, c)
               .has_value());
    CHECK(c.sampler.max_tokens.has_value());
    if (c.sampler.max_tokens) CHECK_EQ(*c.sampler.max_tokens, 8);

    CompletionRequest both;
    CHECK(!parse_completion_request(
               json{{"model", "stub"}, {"prompt", "hi"}, {"n_predict", 8}, {"max_tokens", 5}}, both)
               .has_value());
    CHECK(both.sampler.max_tokens.has_value());
    if (both.sampler.max_tokens) CHECK_EQ(*both.sampler.max_tokens, 5);

    // -1 is llama.cpp's default ("unset") and leaves the budget alone.
    CompletionRequest unset;
    CHECK(!parse_completion_request(
               json{{"model", "stub"}, {"prompt", "hi"}, {"n_predict", -1}}, unset)
               .has_value());
    CHECK(!unset.sampler.max_tokens.has_value());
}

TEST(request_shapes_top_logprobs_zero_is_accepted_on_both_routes) {
    // The OpenAI SDK's default and the compliance suite's value. Mutation:
    // make the top_logprobs predicate trip at 0 -- both cases fail.
    ChatRequest chat;
    CHECK(!parse_chat_request(json{{"model", "stub"},
                                   {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                                   {"top_logprobs", 0}},
                              chat)
               .has_value());
    CompletionRequest comp;
    CHECK(!parse_completion_request(
               json{{"model", "stub"}, {"prompt", "hi"}, {"top_logprobs", 0}}, comp)
               .has_value());
}

TEST(request_shapes_timings_are_on_a_streamed_completion) {
    // Mutation: drop the `final_extra["timings"]` branch and the usage branch's
    // timings from stream_completion -- both cases lose the object.
    Env env;
    for (bool usage : {false, true}) {
        json b = completion_body(true);
        b["stream"] = true;
        b["arcint"] = {{"timings", true}};
        if (usage) b["stream_options"] = {{"include_usage", true}};
        api::PreparedCompletion prep;
        CHECK(!api::prepare_completion(env.ctx, b, prep).has_value());
        std::string frames;
        api::stream_completion(env.ctx, prep, 0, [&](std::string_view f) {
            frames.append(f);
            return true;
        });
        CHECK(frames.find("\"timings\"") != std::string::npos);
        CHECK(frames.find("data: [DONE]") != std::string::npos);
        if (usage) CHECK(frames.find("\"usage\"") != std::string::npos);
    }
}
