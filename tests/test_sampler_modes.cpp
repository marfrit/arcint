#include "api/handlers.h"

#include "exec/backend.h"
#include "harness.h"

// The sampler row is chosen by the request's thinking mode (sampler-defaults
// order, chunk 1): the operator's --nothink-* row applies only to a request
// that resolves to thinking off, and only when at least one such flag is set.
// Explicit request fields still win. With no --nothink-* flag every request
// gets the normal row, byte-identically to before.
//
// Handler-level, with the stub backend, because finish_prepare is where the
// row is picked and where the request's own overrides are applied.
namespace {

using namespace lgc;
using nlohmann::json;

SamplerDefaults normal_row() {
    SamplerDefaults d;
    d.temperature = 0.6f;
    d.top_p       = 0.95f;
    d.top_k       = 20;
    d.min_p       = 0.0f;
    d.provenance  = "operator";
    return d;
}

SamplerDefaults with_nothink_row() {
    SamplerDefaults d = normal_row();
    d.nothink_temperature      = 0.7f;
    d.nothink_top_p            = 0.8f;
    d.nothink_top_k            = 35;
    d.nothink_min_p            = 0.05f;
    d.nothink_presence_penalty = 0.3f;
    d.nothink_row              = true;
    return d;
}

struct Fixture {
    Config                  cfg;
    std::unique_ptr<Backend> backend;
    api::SlotPool           slots{1};
    api::Context            ctx;

    explicit Fixture(const SamplerDefaults& d) {
        cfg.stub = true;
        backend  = make_stub_backend(4096, 0, {}, {}, d);
        ctx.cfg     = &cfg;
        ctx.backend = backend.get();
        ctx.slots   = &slots;
    }
};

// A chat request with the thinking switch stated (or not), and optionally an
// explicit top_p of its own.
json chat(bool state_thinking, bool thinking, std::optional<float> top_p = std::nullopt) {
    json body{{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
              {"max_tokens", 1}};
    if (state_thinking) body["chat_template_kwargs"] = json{{"enable_thinking", thinking}};
    if (top_p) body["top_p"] = *top_p;
    return body;
}

}  // namespace

TEST(sampler_modes_thinking_off_gets_the_nothink_row) {
    Fixture f(with_nothink_row());
    api::PreparedChat prep;
    CHECK(!api::prepare_chat(f.ctx, chat(true, false), prep));

    const SamplerParams& p = prep.input.sampler;
    CHECK(p.temperature == 0.7f);
    CHECK(p.top_p == 0.8f);
    CHECK_EQ(p.top_k, 35);
    CHECK(p.min_p == 0.05f);
    CHECK(p.presence_penalty == 0.3f);
}

TEST(sampler_modes_thinking_on_or_unspecified_gets_the_normal_row) {
    Fixture f(with_nothink_row());
    api::PreparedChat prep;

    // explicitly on
    CHECK(!api::prepare_chat(f.ctx, chat(true, true), prep));
    CHECK(prep.input.sampler.temperature == 0.6f);
    CHECK(prep.input.sampler.top_p == 0.95f);
    CHECK_EQ(prep.input.sampler.top_k, 20);

    // unspecified, the process default is thinking on: the normal row
    f.cfg.think_default = true;
    CHECK(!api::prepare_chat(f.ctx, chat(false, true), prep));
    CHECK(prep.input.sampler.temperature == 0.6f);
    CHECK(prep.input.sampler.top_p == 0.95f);
    CHECK_EQ(prep.input.sampler.top_k, 20);
}

TEST(sampler_modes_request_field_wins_over_the_nothink_row) {
    Fixture f(with_nothink_row());
    api::PreparedChat prep;
    CHECK(!api::prepare_chat(f.ctx, chat(true, false, 0.42f), prep));

    // the request's top_p wins; the nothink row still supplies the rest
    CHECK(prep.input.sampler.top_p == 0.42f);
    CHECK(prep.input.sampler.temperature == 0.7f);
    CHECK_EQ(prep.input.sampler.top_k, 35);
}

TEST(sampler_modes_without_nothink_flags_both_modes_get_the_normal_row) {
    Fixture f(normal_row());   // no --nothink-* flag: nothink_row is false
    api::PreparedChat prep;

    for (bool thinking : {true, false}) {
        CHECK(!api::prepare_chat(f.ctx, chat(true, thinking), prep));
        CHECK(prep.input.sampler.temperature == 0.6f);
        CHECK(prep.input.sampler.top_p == 0.95f);
        CHECK_EQ(prep.input.sampler.top_k, 20);
        CHECK(prep.input.sampler.min_p == 0.0f);
        CHECK(prep.input.sampler.presence_penalty == 0.0f);
    }

    // the completions path has no thinking switch: always the normal row
    api::PreparedCompletion cp;
    json body{{"prompt", "hi"}, {"max_tokens", 1}};
    CHECK(!api::prepare_completion(f.ctx, body, cp));
    CHECK(cp.input.sampler.temperature == 0.6f);
    CHECK(cp.input.sampler.top_p == 0.95f);
}

// The common case: a client sends neither chat_template_kwargs.enable_thinking
// nor reasoning_effort, and the server has no --chat-template-kwarg default.
// The template thinks by default, so the request must get the normal row even
// though a non-thinking row is configured.
TEST(sampler_modes_unspecified_with_no_operator_default_gets_the_normal_row) {
    Fixture f(with_nothink_row());
    CHECK(!f.cfg.think_default);   // no operator default fills the field in

    api::PreparedChat prep;
    CHECK(!api::prepare_chat(f.ctx, chat(false, true), prep));
    CHECK(!prep.req.has_enable_thinking);   // nothing was stated, nothing filled
    CHECK(prep.req.enable_thinking);        // the template's thinking default

    CHECK(prep.input.sampler.temperature == 0.6f);
    CHECK(prep.input.sampler.top_p == 0.95f);
    CHECK_EQ(prep.input.sampler.top_k, 20);
    CHECK(prep.input.sampler.min_p == 0.0f);
    CHECK(prep.input.sampler.presence_penalty == 0.0f);
}
