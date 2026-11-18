#include "api/handlers.h"
#include "api/responses.h"

#include <algorithm>
#include <string>

#include "harness.h"

// --llama-dory-max-loops, the request path (contrib/llama.cpp 0118): prepare_chat turns the request's
// effective reasoning effort (the one the chat template is rendered with) into GenerationInput::dory_loops,
// refuses what is not an effort level, and does nothing at all with the flag off. A fake backend stands in
// for the model; the count that reaches generate() is the one in the prepared input.
namespace {

using namespace lgc;
using lgc::api::Context;
using lgc::api::SlotPool;
using json = nlohmann::json;

class ByteTokenizer final : public Tokenizer {
public:
    std::vector<int> encode(std::string_view text) override { return std::vector<int>(text.begin(), text.end()); }
    std::string      decode(const std::vector<int>& ids) override { return std::string(ids.begin(), ids.end()); }
    std::string      decode_one(int id) override { return std::string(1, static_cast<char>(id)); }
    int              eos_id() const override { return 0; }
};

class FakeBackend final : public Backend {
public:
    ModelStatus        st;
    const ModelStatus& status() const override { return st; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest& req) const override {
        std::string out;
        for (const ChatMessage& m : req.messages) out += m.content;
        return out;
    }
    FinishReason generate(const GenerationInput&, int, const TokenCallback&, GenerationStats&) override {
        return FinishReason::Stop;
    }

private:
    ByteTokenizer tok_;
};

// A Dory unit: 2 loops of its own, per-request counts up to `max_loops` (0: the flag is off).
struct DoryUnit {
    Config      cfg;
    FakeBackend backend;
    SlotPool    slots{1};
    Context     ctx;

    explicit DoryUnit(int max_loops, const char* default_effort = nullptr) {
        cfg.stub = true;
        if (default_effort) cfg.chat_template_kwarg_defaults.emplace_back("reasoning_effort", json(default_effort).dump());
        backend.st.id          = "dory";
        backend.st.served_id   = "dory";
        backend.st.arch        = "dory";
        backend.st.loaded      = true;
        backend.st.n_ctx       = 512;
        backend.st.n_ctx_train = 512;
        backend.st.dory_present      = true;
        backend.st.dory.loops        = 2;
        backend.st.dory.max_loops    = max_loops;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
    }
};

json chat_body(json extra = json::object()) {
    json b{{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}, {"max_tokens", 1}};
    for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
    return b;
}

// the count prepared for the body, or -1 and the 400's message
int prepared(DoryUnit& u, const json& body, std::string* msg = nullptr) {
    lgc::api::PreparedChat prep;
    const auto             err = lgc::api::prepare_chat(u.ctx, body, prep);
    if (err) {
        CHECK_EQ(err->status, 400);
        if (msg) *msg = err->body.at("error").at("message").get<std::string>();
        return -1;
    }
    return prep.input.dory_loops;
}

// the 400's `param` for the body, or "(prepared)" when it is not refused: a request that is wrongly
// accepted fails the comparison instead of throwing out of the case
std::string refusal_param(DoryUnit& u, const json& body) {
    lgc::api::PreparedChat prep;
    const auto             err = lgc::api::prepare_chat(u.ctx, body, prep);
    return err ? err->body.at("error").value("param", std::string("(none)")) : std::string("(prepared)");
}

}  // namespace

TEST(dory_request_each_level_reaches_the_input) {
    DoryUnit u(4);
    CHECK_EQ(prepared(u, chat_body()), 2);   // no effort: the unit's count
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "none"}})), 1);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "off"}})), 1);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "minimal"}})), 1);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "low"}})), 1);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "medium"}})), 2);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "high"}})), 4);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "xhigh"}})), 4);
    DoryUnit capped(3);
    CHECK_EQ(prepared(capped, chat_body({{"reasoning_effort", "xhigh"}})), 3);
}

// one effective effort: the explicit kwarg wins over the typed field, exactly as in the template
TEST(dory_request_kwarg_wins_over_the_typed_field_like_the_template) {
    DoryUnit u(4);
    const json both = chat_body({{"reasoning_effort", "low"}, {"chat_template_kwargs", {{"reasoning_effort", "high"}}}});
    CHECK_EQ(prepared(u, both), 4);
    const json kw_only = chat_body({{"chat_template_kwargs", {{"reasoning_effort", "medium"}}}});
    CHECK_EQ(prepared(u, kw_only), 2);
    // the template is handed the same value
    ChatRequest req;
    CHECK(!parse_chat_request(both, req));
    const EffectiveEffort eff = effective_reasoning_effort(req);
    CHECK(eff.present && eff.is_string);
    CHECK_EQ(eff.level, std::string("high"));
    std::string err;
    const json  ctx = template_context(req, err, "reasoning_effort 'low' 'high'");
    CHECK_EQ(ctx.at("reasoning_effort").get<std::string>(), eff.level);
}

TEST(dory_request_refuses_what_is_not_an_effort_level) {
    DoryUnit u(4);
    std::string msg;
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "HIGH"}}), &msg), -1);
    CHECK(msg.find("'HIGH'") != std::string::npos);
    CHECK(msg.find("none, off, minimal, low, medium, high, xhigh") != std::string::npos);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "ultra"}})), -1);
    // the 400 names the field the bad value came from
    CHECK_EQ(refusal_param(u, chat_body({{"reasoning_effort", "ultra"}})), std::string("reasoning_effort"));
    CHECK_EQ(refusal_param(u, chat_body({{"chat_template_kwargs", {{"reasoning_effort", "ultra"}}}})),
             std::string("chat_template_kwargs.reasoning_effort"));
    CHECK_EQ(prepared(u, chat_body({{"chat_template_kwargs", {{"reasoning_effort", "High"}}}})), -1);
    CHECK_EQ(prepared(u, chat_body({{"chat_template_kwargs", {{"reasoning_effort", 3}}}}), &msg), -1);
    CHECK(msg.find("must be a string") != std::string::npos);
    // a bad kwarg is refused even when the typed field is fine: the kwarg is what the template reads
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "low"}, {"chat_template_kwargs", {{"reasoning_effort", "HIGH"}}}})), -1);
}

// a unit default effort (--chat-template-kwarg reasoning_effort=...) is the effort of a request that names none
TEST(dory_request_unit_default_effort) {
    DoryUnit u(4, "high");
    CHECK_EQ(prepared(u, chat_body()), 4);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "low"}})), 1);
    CHECK_EQ(prepared(u, chat_body({{"chat_template_kwargs", {{"reasoning_effort", "medium"}}}})), 2);
}

// with the flag off nothing runs: no count, and no refusal of an odd effort
TEST(dory_request_flag_off_changes_nothing) {
    DoryUnit u(0);
    CHECK_EQ(prepared(u, chat_body()), 0);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "high"}})), 0);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", "HIGH"}})), 0);
    CHECK_EQ(prepared(u, chat_body({{"chat_template_kwargs", {{"reasoning_effort", 3}}}})), 0);
    // a model that is not Dory
    DoryUnit other(4);
    other.backend.st.dory_present = false;
    CHECK_EQ(prepared(other, chat_body({{"reasoning_effort", "HIGH"}})), 0);
}

// /v1/completions has no effort: it runs the unit's count
TEST(dory_completions_run_the_units_count) {
    DoryUnit u(4, "high");
    lgc::api::PreparedCompletion prep;
    CHECK(!lgc::api::prepare_completion(u.ctx, json{{"prompt", "hi"}, {"max_tokens", 1}}, prep));
    CHECK_EQ(prep.input.dory_loops, 0);
}

// the effort is decided before the render and the fit pass: its 400 wins over a context-overflow 400
TEST(dory_request_effort_400_wins_over_a_later_error) {
    DoryUnit    u(4);
    const std::string big(2000, 'x');   // the fake renders the content as is: far past n_ctx 512
    json        long_body = chat_body();
    long_body["messages"][0]["content"] = big;
    // without a bad effort the overflow is what is refused
    std::string msg;
    CHECK_EQ(prepared(u, long_body, &msg), -1);
    CHECK(msg.find("reasoning effort") == std::string::npos);
    long_body["reasoning_effort"] = "HIGH";
    CHECK_EQ(prepared(u, long_body, &msg), -1);
    CHECK(msg.find("reasoning effort 'HIGH'") != std::string::npos);
}

// a typed empty string and a kwarg null read as no effort, as for the template (the flag must not change
// how a request is read); a kwarg that is the empty string is present and is not a level
TEST(dory_request_empty_and_null_efforts_read_as_unset) {
    DoryUnit u(4);
    CHECK_EQ(prepared(u, chat_body({{"reasoning_effort", ""}})), 2);
    CHECK_EQ(prepared(u, chat_body({{"chat_template_kwargs", {{"reasoning_effort", nullptr}}}})), 2);
    DoryUnit d(4, "high");
    CHECK_EQ(prepared(d, chat_body({{"reasoning_effort", ""}})), 4);   // the unit default's count
    CHECK_EQ(prepared(u, chat_body({{"chat_template_kwargs", {{"reasoning_effort", ""}}}})), -1);
}

// Responses: reasoning.effort is the typed effort of the same request path
TEST(dory_response_reasoning_effort_reaches_the_input) {
    DoryUnit u(4);
    const struct { const char* effort; int want; } rows[] = {{"none", 1}, {"minimal", 1}, {"low", 1}, {"medium", 2}, {"high", 4}, {"xhigh", 4}};
    for (const auto& r : rows) {
        lgc::api::ResponseRequest req;
        CHECK(!lgc::api::parse_response_request(
                  json{{"model", "dory"}, {"input", "hi"}, {"max_output_tokens", 16}, {"reasoning", {{"effort", r.effort}}}}, req));
        lgc::api::PreparedResponse prep;
        CHECK(!lgc::api::prepare_response(u.ctx, std::move(req), prep));
        CHECK_EQ(prep.chat.input.dory_loops, r.want);
    }
    lgc::api::ResponseRequest none;
    CHECK(!lgc::api::parse_response_request(json{{"model", "dory"}, {"input", "hi"}, {"max_output_tokens", 16}}, none));
    lgc::api::PreparedResponse prep;
    CHECK(!lgc::api::prepare_response(u.ctx, std::move(none), prep));
    CHECK_EQ(prep.chat.input.dory_loops, 2);
    // the flag off: no count
    DoryUnit off(0);
    lgc::api::ResponseRequest hi;
    CHECK(!lgc::api::parse_response_request(
              json{{"model", "dory"}, {"input", "hi"}, {"max_output_tokens", 16}, {"reasoning", {{"effort", "high"}}}}, hi));
    lgc::api::PreparedResponse p2;
    CHECK(!lgc::api::prepare_response(off.ctx, std::move(hi), p2));
    CHECK_EQ(p2.chat.input.dory_loops, 0);
}
