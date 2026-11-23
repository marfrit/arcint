// Client-default requests must not be refused. Two shapes: the OpenAI SDK's
// default set on all three completion surfaces, and the field set the live pi
// client sends on /v1/chat/completions. Both are 200 cases, not parser cases:
// the route prepares and runs against the device-free stub.
//
// RED state: the /v1/responses case failed before response_format, logprobs, n
// and top_logprobs 0/null were read there; the chat and completions cases were
// already green and are pinned so the refusal table cannot grow over them.
#include "api/handlers.h"
#include "api/responses.h"

#include "core/chat.h"
#include "exec/backend.h"
#include "harness.h"

#include <memory>
#include <string>

using namespace lgc;
using api::ResponseRequest;
using nlohmann::json;

namespace {

// The chat and completions routes' device-free fixture.
struct ChatEnv {
    Config                  cfg;
    std::unique_ptr<Backend> backend;
    api::SlotPool           slots{1};
    api::Context            ctx;

    ChatEnv() : backend(make_stub_backend(4096, 0, {}, {}, {}, 1)) {
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = backend.get();
        ctx.slots   = &slots;
    }
};

struct ResponseEnv {
    Config                   cfg;
    std::unique_ptr<Backend> backend;
    api::SlotPool            slots{1};
    api::Context             ctx;

    ResponseEnv() : backend(make_stub_backend(4096, 0, {}, {}, {}, 1)) {
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = backend.get();
        ctx.slots   = &slots;
    }
};

api::HttpResult run_chat_body(ChatEnv& env, const json& body) {
    api::PreparedChat prep;
    if (auto err = api::prepare_chat(env.ctx, body, prep)) return *err;
    const auto lease = env.slots.acquire_for(0);
    return api::run_chat(env.ctx, prep, lease.index());
}

api::HttpResult run_completion_body(ChatEnv& env, const json& body) {
    api::PreparedCompletion prep;
    if (auto err = api::prepare_completion(env.ctx, body, prep)) return *err;
    const auto lease = env.slots.acquire_for(0);
    return api::run_completion(env.ctx, prep, lease.index());
}

api::HttpResult run_response_body(ResponseEnv& env, const json& body) {
    ResponseRequest req;
    if (auto err = api::parse_response_request(body, req)) return *err;
    api::PreparedResponse prep;
    if (auto err = api::prepare_response(env.ctx, std::move(req), prep)) return *err;
    const auto lease = env.slots.acquire_for(0);
    return api::run_response(env.ctx, prep, lease.index());
}

json sdk_chat_defaults() {
    return json{{"model", "stub"},
                {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                {"max_tokens", 8},
                {"response_format", {{"type", "text"}}},
                {"logprobs", false},
                {"n", 1},
                {"top_logprobs", nullptr}};
}

json sdk_completion_defaults() {
    return json{{"model", "stub"},
                {"prompt", "hi"},
                {"max_tokens", 8},
                {"response_format", {{"type", "text"}}},
                {"logprobs", false},
                {"n", 1},
                {"top_logprobs", nullptr}};
}

json sdk_response_defaults(int top_logprobs) {
    json b = {{"model", "stub"},
              {"input", "hi"},
              {"max_output_tokens", 16},
              {"response_format", {{"type", "text"}}},
              {"logprobs", false},
              {"n", 1}};
    b["top_logprobs"] = top_logprobs;
    return b;
}

}  // namespace

TEST(request_defaults_openai_sdk_set_is_200_on_chat_and_completions) {
    ChatEnv env;
    const auto chat = run_chat_body(env, sdk_chat_defaults());
    CHECK_EQ(chat.status, 200);
    const auto comp = run_completion_body(env, sdk_completion_defaults());
    CHECK_EQ(comp.status, 200);
}

TEST(request_defaults_openai_sdk_set_is_200_on_responses) {
    ResponseEnv env;
    // null and 0: the SDK's default and the compliance suite's exact value.
    for (int top : {0, -1}) {
        json b = sdk_response_defaults(top);
        if (top < 0) b["top_logprobs"] = nullptr;
        const auto r = run_response_body(env, b);
        CHECK_EQ(r.status, 200);
        if (r.status != 200) t::fail(__FILE__, __LINE__, "top_logprobs " + b["top_logprobs"].dump());
    }
}

TEST(request_defaults_pi_chat_field_set_is_200) {
    ChatEnv env;
    const json tools = json::array({{{"type", "function"},
                                    {"function",
                                     {{"name", "run_query"},
                                      {"description", "run one query"},
                                      {"parameters",
                                       {{"type", "object"},
                                        {"properties", {{"sql", {{"type", "string"}}}}}}}}}}});
    const json body = {{"model", "stub"},
                       {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                       {"stream", false},
                       {"stream_options", {{"include_usage", true}}},
                       {"store", false},
                       {"max_tokens", 8},
                       {"temperature", 0.7},
                       {"tools", tools},
                       {"tool_choice", "auto"},
                       {"reasoning_effort", "medium"},
                       {"chat_template_kwargs", {{"enable_thinking", true}}},
                       {"enable_thinking", true},
                       {"thinking", "medium"},
                       {"reasoning", {{"effort", "medium"}}},
                       {"priority", "normal"},
                       {"provider", "openai"},
                       {"prompt_cache_key", "cache-key-1"},
                       {"user", "benchmark"}};
    const auto r = run_chat_body(env, body);
    CHECK_EQ(r.status, 200);
    if (r.status != 200) CHECK(r.body.contains("error"));
}

TEST(request_defaults_max_completion_tokens_is_200_on_chat) {
    ChatEnv env;
    json body = sdk_chat_defaults();
    body.erase("max_tokens");
    body["max_completion_tokens"] = 8;
    CHECK_EQ(run_chat_body(env, body).status, 200);
}
