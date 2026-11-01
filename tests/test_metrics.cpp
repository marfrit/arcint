#include "api/generate.h"
#include "api/handlers.h"
#include "api/metrics.h"
#include "api/responses.h"

#include "harness.h"

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

// GET /metrics (--metrics) and the opt-in per-response timings (arcint.timings).
// Both are additive and off by default: without --metrics the route does not
// exist, and an absent arcint.timings keeps every response byte-identical.
// These cases pin the Prometheus text, the counters' source, the request label
// folding and the seven timing fields (llama-server's names).
namespace {

using namespace lgc;
using lgc::api::Context;
using lgc::api::Lane;
using lgc::api::Metrics;
using lgc::api::PreparedChat;
using lgc::api::SlotPool;
using json = nlohmann::json;

struct Fixture {
    Config                   cfg;
    std::unique_ptr<Backend> backend;
    SlotPool                 slots{1};
    Context                  ctx;

    Fixture() {
        backend     = make_stub_backend(4096);
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = backend.get();
        ctx.slots   = &slots;
    }
};

// The label-value escaping (§Prometheus text): backslash, quote and newline.
bool has(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

// ------------------------------------------------------------------- timings
TEST(timings_json_has_llama_servers_fields) {
    GenerationStats s;
    s.prompt_tokens     = 12;
    s.completion_tokens = 7;
    s.cache_hit_tokens  = 3;
    s.prefill_seconds   = 1.25;
    s.decode_seconds    = 2.5;
    s.draft_proposed    = 4;
    s.draft_accepted    = 2;

    const json t = api::timings_json(s);
    CHECK(t.is_object());
    CHECK_EQ(t.value("prompt_n", -1), 12);
    CHECK_EQ(t.value("predicted_n", -1), 7);
    CHECK_EQ(t.value("cache_n", -1), 3);
    CHECK_EQ(t.value("draft_n", -1), 4);
    CHECK_EQ(t.value("draft_accepted", -1), 2);
    CHECK_NEAR(t.value("prompt_ms", -1.0), 1250.0, 1e-6);
    CHECK_NEAR(t.value("predicted_ms", -1.0), 2500.0, 1e-6);
}

TEST(chat_request_reads_arcint_timings) {
    ChatRequest req;
    json body = json{{"model", "stub"},
                     {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                     {"arcint", {{"timings", true}}}};
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.timings);
}

TEST(chat_request_refuses_a_malformed_arcint_object) {
    auto rejected = [](const json& a) {
        ChatRequest req;
        json body = json{{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                         {"arcint", a}};
        return parse_chat_request(body, req).has_value();
    };
    CHECK(rejected(json{{"timings", "yes"}}));
    CHECK(rejected(json{{"spans", true}}));
    CHECK(rejected(json{{"v", 2}}));
    CHECK(!rejected(json{{"v", 1}}));
}

TEST(chat_run_adds_timings_only_when_asked) {
    Fixture f;
    const json base = json{{"model", "stub"},
                           {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};

    PreparedChat plain;
    CHECK(!api::prepare_chat(f.ctx, base, plain));
    const api::HttpResult r0 = api::run_chat(f.ctx, plain, 0);
    CHECK_EQ(r0.status, 200);
    CHECK(!r0.body.contains("timings"));

    json asked = base;
    asked["arcint"] = {{"timings", true}};
    PreparedChat timed;
    CHECK(!api::prepare_chat(f.ctx, asked, timed));
    const api::HttpResult r1 = api::run_chat(f.ctx, timed, 0);
    CHECK_EQ(r1.status, 200);
    CHECK(r1.body.contains("timings"));
    if (r1.body.contains("timings")) {
        CHECK(r1.body["timings"].contains("prompt_n"));
        CHECK(r1.body["timings"].contains("predicted_ms"));
        CHECK(r1.body["timings"].contains("draft_accepted"));
    }
}

TEST(responses_request_reads_arcint_timings) {
    api::ResponseRequest req;
    json body = json{{"model", "stub"}, {"input", "hi"}, {"arcint", {{"v", 1}, {"timings", true}}}};
    CHECK(!api::parse_response_request(body, req).has_value());
    CHECK(req.timings);
}

// ------------------------------------------------------------------- metrics
TEST(metrics_route_label_folds_to_bounded_values) {
    CHECK_EQ(api::metrics_route_label("/props"), std::string("/props"));
    CHECK_EQ(api::metrics_route_label("/v1/chat/completions"),
             std::string("/v1/chat/completions"));
    CHECK_EQ(api::metrics_route_label("/v1/responses/resp-7/cancel"),
             std::string("/v1/responses/:id/cancel"));
    CHECK_EQ(api::metrics_route_label("/v1/responses/resp-7"), std::string("other"));
    CHECK_EQ(api::metrics_route_label("/scanned/../../../etc/passwd"), std::string("other"));
}

TEST(metrics_render_carries_counters_and_gauges) {
    Fixture f;
    Metrics m;
    m.observe_request("/props", 200);
    m.observe_request("/props", 200);
    m.observe_request("/v1/chat/completions", 503);

    GenerationStats s;
    s.prompt_tokens     = 10;
    s.completion_tokens = 5;
    s.cache_hit_tokens  = 3;
    s.prefill_seconds   = 1.5;
    s.decode_seconds    = 2.0;
    s.draft_proposed    = 4;
    s.draft_accepted    = 2;
    m.observe_generation(s);

    const std::string text = m.render(f.ctx);
    CHECK(has(text, "arcint_requests_total{route=\"/props\",status=\"200\"} 2"));
    CHECK(has(text, "arcint_requests_total{route=\"/v1/chat/completions\",status=\"503\"} 1"));
    CHECK(has(text, "arcint_prompt_tokens_total 10"));
    CHECK(has(text, "arcint_completion_tokens_total 5"));
    CHECK(has(text, "arcint_prompt_cache_hit_tokens_total 3"));
    CHECK(has(text, "arcint_prefill_seconds_total 1.500000"));
    CHECK(has(text, "arcint_decode_seconds_total 2.000000"));
    CHECK(has(text, "arcint_draft_proposed_total 4"));
    CHECK(has(text, "arcint_draft_accepted_total 2"));
    CHECK(has(text, "arcint_queue_depth{lane=\"default\"} 0"));
    CHECK(has(text, "arcint_slots_busy{lane=\"default\"} 0"));
    CHECK(has(text, "arcint_slots_total{lane=\"default\"} 1"));
    CHECK(has(text, "arcint_model_loaded 1"));
}

TEST(metrics_render_labels_each_named_lane) {
    Fixture f;
    SlotPool lane_pool(1);
    f.ctx.lanes.push_back(Lane{"sub\"agent", 4096, 1, &lane_pool});
    Metrics m;
    const std::string text = m.render(f.ctx);
    CHECK(has(text, "arcint_queue_depth{lane=\"sub\\\"agent\"} 0"));
    CHECK(has(text, "arcint_slots_total{lane=\"sub\\\"agent\"} 1"));
    CHECK(has(text, "arcint_model_loaded 1"));
}
