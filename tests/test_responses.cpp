#include "api/error.h"
#include "api/responses.h"

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness.h"
#include "exec/backend.h"

// Open Responses, W1 (docs/campaigns/feyntalk.md): the request mapping — every
// input item type, the flat tool shape, the tool_choice forms, the refusals
// with their spec error envelope, the sampler/instructions/thinking mapping —
// and the complete non-streamed ResponseResource (every required key of the
// 2026-04-24 OpenAPI document, its output items and usage). Red first: the
// cases were written and run against a skeleton that refused every request and
// returned an empty resource; they failed, then the mapping made them pass.
namespace {

using namespace lgc;
using lgc::api::Context;
using lgc::api::PreparedChat;
using lgc::api::PreparedResponse;
using lgc::api::ResponseRequest;
using lgc::api::SlotPool;
using json = nlohmann::json;

// A device-free fixture: the stub backend, one lane, the chat route's Config.
struct Fixture {
    Config                  cfg;
    std::unique_ptr<Backend> backend;
    SlotPool                slots{1};
    Context                 ctx;

    Fixture() {
        backend = make_stub_backend(4096, 0, {}, {}, {}, 1);
        cfg.stub = true;
        ctx.cfg  = &cfg;
        ctx.backend = backend.get();
        ctx.slots   = &slots;
    }
};

json parse_ok(const json& body, ResponseRequest& out) {
    auto err = api::parse_response_request(body, out);
    CHECK(!err.has_value());
    return err ? err->body : json::object();
}

// The refused field: the envelope's code, param and type.
void check_refusal(const json& body, const std::string& code, const std::string& param,
                   const char* where) {
    ResponseRequest out;
    auto            err = api::parse_response_request(body, out);
    if (!err.has_value()) {
        t::fail(__FILE__, __LINE__, std::string(where) + ": expected a refusal, parsed instead");
        return;
    }
    CHECK_EQ(err->status, 400);
    const json& e = err->body["error"];
    CHECK_EQ(e["code"].get<std::string>(), code);
    if (param.empty()) CHECK(e["param"].is_null());
    else CHECK_EQ(e["param"].get<std::string>(), param);
    CHECK_EQ(e["type"].get<std::string>(), std::string("invalid_request"));
}

json base() { return json{{"model", "stub"}, {"input", "hi"}}; }

}  // namespace

// ------------------------------------------------------------ request mapping
TEST(responses_maps_a_plain_string_input) {
    ResponseRequest r;
    parse_ok(base(), r);
    CHECK_EQ(r.chat.model, std::string("stub"));
    CHECK_EQ(r.chat.messages.size(), 1u);
    if (r.chat.messages.empty()) return;
    CHECK_EQ(r.chat.messages[0].role, std::string("user"));
    CHECK_EQ(r.chat.messages[0].content, std::string("hi"));
    CHECK(r.chat.tool_choice == "auto");
    // The spec defaults the resource echoes, not the chat request's.
    CHECK(r.store);
    CHECK(!r.background);
    CHECK_EQ(r.truncation, std::string("disabled"));
    CHECK_EQ(r.service_tier, std::string("default"));
    CHECK(r.tools.empty());
}

TEST(responses_maps_every_message_part_type) {
    json body = base();
    body["input"] = json::array({
        {{"type", "message"},
         {"role", "user"},
         {"content", json::array({{{"type", "input_text"}, {"text", "a"}},
                                  {{"type", "output_text"}, {"text", "b"}},
                                  {{"type", "text"}, {"text", "c"}}})}},
        {{"type", "message"}, {"role", "system"}, {"content", "sys"}},
        {{"type", "message"}, {"role", "developer"}, {"content", "dev"}},
        {{"type", "message"},
         {"role", "assistant"},
         {"content", json::array({{{"type", "output_text"}, {"text", "ans"}}})},
         {"phase", "commentary"}},   // R2: accepted, ignored by the template
    });
    ResponseRequest r;
    parse_ok(body, r);
    CHECK_EQ(r.chat.messages.size(), 4u);
    if (r.chat.messages.size() < 4) return;
    // Parts are concatenated with a newline between them (code: the chat path).
    CHECK_EQ(r.chat.messages[0].content, std::string("a\nb\nc"));
    CHECK_EQ(r.chat.messages[1].role, std::string("system"));
    CHECK_EQ(r.chat.messages[2].role, std::string("developer"));
    CHECK_EQ(r.chat.messages[3].role, std::string("assistant"));
    CHECK_EQ(r.chat.messages[3].content, std::string("ans"));
}

TEST(responses_maps_an_image_part_and_refuses_the_other_modalities) {
    json body = base();
    body["input"] = json::array({
        {{"type", "message"},
         {"role", "user"},
         {"content", json::array({{{"type", "input_text"}, {"text", "look "}},
                                  {{"type", "input_image"},
                                   {"image_url", "data:image/png;base64,AAAA"},
                                   {"detail", "auto"}},
                                  {{"type", "input_text"}, {"text", " at it"}}})}},
    });
    ResponseRequest r;
    parse_ok(body, r);
    CHECK_EQ(r.chat.messages.size(), 1u);
    if (r.chat.messages.empty()) return;
    CHECK_EQ(r.chat.messages[0].images.size(), size_t(1));
    CHECK_EQ(r.chat.messages[0].images[0].url, std::string("data:image/png;base64,AAAA"));

    for (const char* type : {"input_file", "input_video"}) {
        json b = base();
        b["input"] = json::array({
            {{"type", "message"},
             {"role", "user"},
             {"content", json::array({{{"type", type}}})}},
        });
        check_refusal(b, "unsupported_parameter", "input", type);
    }
}

TEST(responses_maps_function_call_items_to_the_replay_shape) {
    json body = base();
    body["input"] = json::array({
        {{"type", "function_call"},
         {"call_id", "call_7"}, {"name", "read_file"},
         {"arguments", "{\"path\":\"a.txt\"}"}},
        {{"type", "function_call_output"}, {"call_id", "call_7"}, {"output", "alpha"}},
    });
    ResponseRequest r;
    parse_ok(body, r);
    CHECK_EQ(r.chat.messages.size(), 2u);
    if (r.chat.messages.size() < 2) return;
    CHECK_EQ(r.chat.messages[0].role, std::string("assistant"));
    CHECK_EQ(r.chat.messages[0].tool_calls.size(), size_t(1));
    if (r.chat.messages[0].tool_calls.empty()) return;
    CHECK_EQ(r.chat.messages[0].tool_calls[0].id, std::string("call_7"));
    CHECK_EQ(r.chat.messages[0].tool_calls[0].name, std::string("read_file"));
    CHECK_EQ(r.chat.messages[0].tool_calls[0].arguments, std::string("{\"path\":\"a.txt\"}"));
    CHECK_EQ(r.chat.messages[1].role, std::string("tool"));
    CHECK_EQ(r.chat.messages[1].tool_call_id, std::string("call_7"));
    CHECK_EQ(r.chat.messages[1].content, std::string("alpha"));

    // Array output: input_text parts joined with a newline; an image part is
    // refused rather than silently dropped.
    json array_output = body;
    array_output["input"][1]["output"] = json::array({
        {{"type", "input_text"}, {"text", "one"}},
        {{"type", "input_text"}, {"text", "two"}},
    });
    ResponseRequest r2;
    parse_ok(array_output, r2);
    CHECK_EQ(r2.chat.messages.size(), 2u);
    if (r2.chat.messages.size() < 2) return;
    CHECK_EQ(r2.chat.messages[1].content, std::string("one\ntwo"));

    json image_output = body;
    image_output["input"][1]["output"] = json::array({
        {{"type", "input_text"}, {"text", "one"}},
        {{"type", "input_image"}, {"image_url", "data:image/png;base64,AAAA"}},
    });
    check_refusal(image_output, "unsupported_parameter", "input", "function_call_output image");
}

TEST(responses_accepts_reasoning_items_without_rendering_them) {
    json body = base();
    body["input"] = json::array({
        {{"type", "reasoning"}, {"summary", json::array()}, {"content", nullptr}},
        {{"type", "message"}, {"role", "user"}, {"content", "go on"}},
    });
    ResponseRequest r;
    parse_ok(body, r);
    // Kept for the store (W3), not rendered into the prompt: parity with the
    // chat route, which replays no reasoning either.
    CHECK_EQ(r.reasoning_items.size(), size_t(1));
    CHECK_EQ(r.chat.messages.size(), 1u);
    if (r.chat.messages.empty()) return;
    CHECK_EQ(r.chat.messages[0].content, std::string("go on"));
}

TEST(responses_refuses_item_reference_and_compaction) {
    for (const char* type : {"item_reference", "compaction"}) {
        json body = base();
        body["input"] = json::array({{{"type", type}}});
        check_refusal(body, "unsupported_parameter", "input", type);
    }
}

TEST(responses_refuses_the_unbuilt_request_fields) {
    auto refuse = [](const char* key, const json& value, const char* param) {
        json body = base();
        body[key] = value;
        check_refusal(body, "unsupported_parameter", param, key);
    };
    refuse("stream", true, "stream");
    refuse("include", json::array({"message.output_text.logprobs"}), "include");
    refuse("max_tool_calls", 4, "max_tool_calls");
    refuse("background", true, "background");
    refuse("service_tier", "flex", "service_tier");
    refuse("top_logprobs", 3, "top_logprobs");
    {
        json body = base();
        body["text"] = {{"format", {{"type", "json_object"}}}};
        check_refusal(body, "unsupported_parameter", "text.format", "json_object");
    }
    {
        json body = base();
        body["tool_choice"] = {{"type", "allowed_tools"},
                               {"tools", json::array({{{"type", "function"}, {"name", "f"}}})}};
        check_refusal(body, "unsupported_parameter", "tool_choice", "allowed_tools");
    }
    {
        json body = base();
        body["arcint"] = {{"v", 2}};
        check_refusal(body, "unsupported_parameter", "arcint.v", "unknown arcint.v");
    }
    {
        json body = base();
        body["arcint"] = {{"v", 1}, {"token_mode", true}};
        check_refusal(body, "unsupported_parameter", "arcint.token_mode", "arcint beyond v");
    }
    {
        json body = base();
        body["max_output_tokens"] = 8;
        check_refusal(body, "invalid_request", "max_output_tokens", "max_output_tokens < 16");
    }
    {
        json body = base();
        body["whatever"] = 1;
        check_refusal(body, "invalid_request", "whatever", "unknown top-level field");
    }
}

// ---------------------------------------------------------- field mapping
TEST(responses_maps_instructions_tools_and_tool_choice) {
    json body = base();
    body["instructions"] = "be terse";
    body["tools"] = json::array({
        {{"type", "function"},
         {"name", "read_file"},
         {"description", "read a file"},
         {"parameters", {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}}}}},
         {"strict", true}},   // accepted and ignored
    });
    body["tool_choice"] = {{"type", "function"}, {"name", "read_file"}};
    body["parallel_tool_calls"] = false;

    ResponseRequest r;
    parse_ok(body, r);
    CHECK(r.has_instructions);
    CHECK_EQ(r.instructions, std::string("be terse"));
    CHECK_EQ(r.chat.messages.size(), 2u);   // instructions as a leading system turn
    if (r.chat.messages.size() < 2) return;
    CHECK_EQ(r.chat.messages[0].role, std::string("system"));
    CHECK_EQ(r.chat.messages[0].content, std::string("be terse"));
    CHECK_EQ(r.chat.messages[1].role, std::string("user"));
    CHECK_EQ(r.chat.tools.size(), size_t(1));
    if (r.chat.tools.empty() || r.tools.empty()) return;
    CHECK_EQ(r.chat.tools[0].name, std::string("read_file"));
    CHECK_EQ(r.chat.tools[0].description, std::string("read a file"));
    CHECK_EQ(r.chat.tool_choice, std::string("required"));
    CHECK_EQ(r.chat.tool_choice_function, std::string("read_file"));
    CHECK(!r.chat.parallel_tool_calls);
    // The resource echoes the flat spec shape, not the chat shape.
    CHECK_EQ(r.tools[0]["type"].get<std::string>(), std::string("function"));
    CHECK_EQ(r.tools[0]["name"].get<std::string>(), std::string("read_file"));
    CHECK_EQ(r.tool_choice["type"].get<std::string>(), std::string("function"));
    CHECK_EQ(r.tool_choice["name"].get<std::string>(), std::string("read_file"));

    json by_name = body;
    by_name["tool_choice"] = "required";
    ResponseRequest r2;
    parse_ok(by_name, r2);
    CHECK_EQ(r2.chat.tool_choice, std::string("required"));
    CHECK_EQ(r2.tool_choice.get<std::string>(), std::string("required"));
}

TEST(responses_maps_the_sampler_and_the_thinking_switch) {
    json body = base();
    body["temperature"]   = 0.5;
    body["top_p"]         = 0.9;
    body["presence_penalty"]  = -0.1;
    body["frequency_penalty"] = 0.2;
    body["max_output_tokens"] = 64;
    body["reasoning"] = {{"effort", "none"}};

    ResponseRequest r;
    parse_ok(body, r);
    CHECK(r.chat.sampler.temperature.has_value());
    if (r.chat.sampler.temperature) CHECK_NEAR(*r.chat.sampler.temperature, 0.5, 1e-6);
    CHECK(r.chat.sampler.top_p.has_value());
    if (r.chat.sampler.top_p) CHECK_NEAR(*r.chat.sampler.top_p, 0.9, 1e-6);
    CHECK(r.chat.sampler.presence_penalty.has_value());
    CHECK(r.chat.sampler.frequency_penalty.has_value());
    CHECK_EQ(r.chat.sampler.max_tokens.value_or(-1), 64);
    CHECK(r.chat.has_enable_thinking);
    CHECK(!r.chat.enable_thinking);   // effort "none" is the fleet's think-off switch
    CHECK_EQ(r.max_output_tokens, 64);
    CHECK(r.reasoning.is_object());
    if (!r.reasoning.is_object()) return;
    CHECK(r.reasoning["effort"].get<std::string>() == "none");
    CHECK(r.reasoning["summary"].is_null());   // no summary is produced (1.3.x)

    json high = base();
    high["reasoning"] = {{"effort", "high"}, {"summary", "auto"}};
    ResponseRequest r2;
    parse_ok(high, r2);
    CHECK(r2.chat.enable_thinking);
    CHECK_EQ(r2.reasoning["effort"].get<std::string>(), std::string("high"));
    CHECK(r2.reasoning["summary"].is_null());
}

TEST(responses_accepts_text_format_text_and_verbosity) {
    json body = base();
    body["text"] = {{"format", {{"type", "text"}}}, {"verbosity", "low"}};
    ResponseRequest r;
    parse_ok(body, r);
    CHECK(r.text.is_object() && r.text["format"].is_object());
    if (!(r.text.is_object() && r.text["format"].is_object())) return;
    CHECK_EQ(r.text["format"]["type"].get<std::string>(), std::string("text"));
    CHECK(r.text.contains("verbosity") && r.text["verbosity"].is_string());
    if (!(r.text.contains("verbosity") && r.text["verbosity"].is_string())) return;
    CHECK_EQ(r.text["verbosity"].get<std::string>(), std::string("low"));

    json absent = base();
    absent.erase("text");
    ResponseRequest r2;
    parse_ok(absent, r2);
    CHECK(r2.text.is_object() && r2.text["format"].is_object());
    if (!(r2.text.is_object() && r2.text["format"].is_object())) return;
    CHECK_EQ(r2.text["format"]["type"].get<std::string>(), std::string("text"));
}

TEST(responses_refuses_a_previous_response_id_until_the_store_exists) {
    json body = base();
    body["previous_response_id"] = "resp-1";
    Fixture fx;
    ResponseRequest r;
    parse_ok(body, r);
    PreparedResponse prep;
    auto err = api::prepare_response(fx.ctx, std::move(r), prep);
    CHECK(err.has_value());
    if (!err.has_value()) return;
    CHECK_EQ(err->status, 400);
    CHECK(err->body["error"]["code"].is_string());
    if (!err->body["error"]["code"].is_string()) return;
    CHECK_EQ(err->body["error"]["code"].get<std::string>(), std::string("previous_response_not_found"));
    CHECK_EQ(err->body["error"]["param"].get<std::string>(), std::string("previous_response_id"));
}

// --------------------------------------------------------------- resource
TEST(responses_resource_carries_every_required_key) {
    // The 2026-04-24 OpenAPI ResponseResource.required array, frozen here: a
    // resource missing one of these is not a response a spec client can parse.
    static const char* const kRequired[] = {
        "id", "object", "created_at", "completed_at", "status", "incomplete_details",
        "model", "previous_response_id", "instructions", "output", "error", "tools",
        "tool_choice", "truncation", "parallel_tool_calls", "text", "top_p",
        "presence_penalty", "frequency_penalty", "top_logprobs", "temperature",
        "reasoning", "usage", "max_output_tokens", "max_tool_calls", "store",
        "background", "service_tier", "metadata", "safety_identifier", "prompt_cache_key"};

    Fixture fx;
    ResponseRequest r;
    parse_ok(base(), r);
    PreparedResponse prep;
    CHECK(!api::prepare_response(fx.ctx, std::move(r), prep).has_value());
    CHECK(prep.chat.id.rfind("resp-", 0) == 0);

    const auto lease = fx.slots.acquire_for(0);
    const api::HttpResult res = api::run_response(fx.ctx, prep, lease.index());
    CHECK_EQ(res.status, 200);
    if (res.status != 200) return;
    const json& body = res.body;
    for (const char* key : kRequired) CHECK(body.contains(key));
    CHECK_EQ(body["object"].get<std::string>(), std::string("response"));
    CHECK_EQ(body["status"].get<std::string>(), std::string("completed"));
    CHECK(body["completed_at"].is_number_integer());
    CHECK(body["incomplete_details"].is_null());
    CHECK(body["previous_response_id"].is_null());
    CHECK(body["instructions"].is_null());
    CHECK(body["error"].is_null());
    CHECK_EQ(body["tool_choice"].get<std::string>(), std::string("auto"));
    CHECK_EQ(body["truncation"].get<std::string>(), std::string("disabled"));
    CHECK(body["parallel_tool_calls"].get<bool>());
    CHECK_EQ(body["text"]["format"]["type"].get<std::string>(), std::string("text"));
    CHECK(body["reasoning"].is_null());
    CHECK(body["max_output_tokens"].is_null());
    CHECK(body["max_tool_calls"].is_null());
    CHECK(body["store"].get<bool>());
    CHECK(!body["background"].get<bool>());
    CHECK_EQ(body["service_tier"].get<std::string>(), std::string("default"));
    CHECK(body["metadata"].is_null());
    CHECK(body["safety_identifier"].is_null());
    CHECK(body["prompt_cache_key"].is_null());
    CHECK_EQ(body["top_logprobs"].get<int>(), 0);
    CHECK_EQ(body["model"].get<std::string>(), std::string("stub"));

    // The message item and the usage block.
    CHECK_EQ(body["output"].size(), size_t(1));
    const json& item = body["output"][0];
    CHECK_EQ(item["type"].get<std::string>(), std::string("message"));
    CHECK_EQ(item["role"].get<std::string>(), std::string("assistant"));
    CHECK_EQ(item["status"].get<std::string>(), std::string("completed"));
    CHECK_EQ(item["content"][0]["type"].get<std::string>(), std::string("output_text"));
    CHECK(!item["content"][0]["text"].get<std::string>().empty());
    CHECK(item["content"][0]["annotations"].is_array());
    const json& usage = body["usage"];
    CHECK(usage["input_tokens"].get<int>() > 0);
    CHECK_EQ(usage["total_tokens"].get<int>(),
             usage["input_tokens"].get<int>() + usage["output_tokens"].get<int>());
    CHECK_EQ(usage["input_tokens_details"]["cached_tokens"].get<int>(), 0);
    CHECK_EQ(usage["output_tokens_details"]["reasoning_tokens"].get<int>(), 0);
}

TEST(responses_resource_emits_reasoning_and_function_call_items) {
    // A hand-built generation, because the stub emits no reasoning block and
    // this resource shape must not depend on one.
    ResponseRequest req;
    PreparedChat    prep;
    prep.id      = "resp-1";
    prep.created = 100;
    prep.model   = "stub";

    api::ChatGeneration gen;
    gen.reasoning        = "weigh the options";
    gen.reasoning_closed = true;
    gen.content          = "final answer";

    const json body = api::response_resource(req, prep, gen);
    CHECK_EQ(body["output"].size(), size_t(2));
    if (body["output"].size() < 2) return;
    CHECK_EQ(body["output"][0]["type"].get<std::string>(), std::string("reasoning"));
    CHECK(body["output"][0]["summary"].is_array());
    CHECK_EQ(body["output"][0]["content"][0]["type"].get<std::string>(), std::string("reasoning_text"));
    CHECK_EQ(body["output"][0]["content"][0]["text"].get<std::string>(), std::string("weigh the options"));
    CHECK_EQ(body["output"][1]["type"].get<std::string>(), std::string("message"));

    // Length: incomplete, the reason is the token budget, and the open item is
    // the last one and marked incomplete (spec §Items).
    api::ChatGeneration cut = gen;
    cut.reason              = FinishReason::Length;
    const json incomplete = api::response_resource(req, prep, cut);
    CHECK_EQ(incomplete["status"].get<std::string>(), std::string("incomplete"));
    CHECK_EQ(incomplete["incomplete_details"]["reason"].get<std::string>(),
             std::string("max_output_tokens"));
    CHECK(incomplete["completed_at"].is_null());
    CHECK_EQ(incomplete["output"].size(), size_t(2));
    if (incomplete["output"].size() < 2) return;
    CHECK_EQ(incomplete["output"].back()["status"].get<std::string>(), std::string("incomplete"));

    api::ChatGeneration loop = gen;
    loop.reason              = FinishReason::Loop;
    const json guarded = api::response_resource(req, prep, loop);
    CHECK_EQ(guarded["incomplete_details"]["reason"].get<std::string>(),
             std::string("loop_guard_abort"));
}

TEST(responses_a_tool_turn_returns_a_function_call_item) {
    Fixture fx;
    json body = base();
    body["input"] = "call the tool";
    body["tools"] = json::array({
        {{"type", "function"}, {"name", "read_file"},
         {"parameters", {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}}}}}},
    });
    ResponseRequest r;
    parse_ok(body, r);
    PreparedResponse prep;
    CHECK(!api::prepare_response(fx.ctx, std::move(r), prep).has_value());

    const auto lease = fx.slots.acquire_for(0);
    const api::HttpResult res = api::run_response(fx.ctx, prep, lease.index());
    CHECK_EQ(res.status, 200);
    if (res.status != 200) return;
    const json& out = res.body["output"];
    bool saw_call = false;
    for (const json& item : out) {
        if (item["type"].get<std::string>() != "function_call") continue;
        saw_call = true;
        CHECK_EQ(item["name"].get<std::string>(), std::string("read_file"));
        CHECK(!item["call_id"].get<std::string>().empty());
        CHECK(item["arguments"].is_string());
        CHECK_EQ(item["status"].get<std::string>(), std::string("completed"));
    }
    CHECK(saw_call);
    CHECK_EQ(res.body["tools"][0]["name"].get<std::string>(), std::string("read_file"));
    CHECK_EQ(res.body["tools"][0]["type"].get<std::string>(), std::string("function"));
}

// -------------------------------------------------------------- error shape
TEST(responses_error_uses_the_spec_spellings) {
    const json e = api::responses_error("nope", "invalid_request", "unsupported_parameter",
                                        "stream");
    CHECK_EQ(e["error"]["type"].get<std::string>(), std::string("invalid_request"));
    CHECK_EQ(e["error"]["code"].get<std::string>(), std::string("unsupported_parameter"));
    CHECK_EQ(e["error"]["param"].get<std::string>(), std::string("stream"));
    CHECK_EQ(e["error"]["message"].get<std::string>(), std::string("nope"));
    // The chat envelope keeps its own type spelling.
    const json chat = api::error_body("nope", "internal_error");
    CHECK_EQ(chat["error"]["type"].get<std::string>(), std::string("internal_error"));
}
