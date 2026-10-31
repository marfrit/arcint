#include "api/error.h"
#include "api/responses.h"
#include "api/responses_state.h"
#include "api/responses_stream.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <memory>
#include <string>
#include <thread>
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

    explicit Fixture(int n_ctx = 4096) {
        backend = make_stub_backend(n_ctx, 0, {}, {}, {}, 1);
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

TEST(responses_maps_a_message_item_without_type) {
    // Found by the live-client test: a real pi client (openai-responses) sends
    // {"role":"user","content":"..."} without `type`. The OpenAPI document
    // requires type but defaults every message item to "message", and clients
    // rely on that default. A role makes it a message; neither stays a 400.
    json body = base();
    body["input"] = json::array({
        {{"role", "user"}, {"content", "hello"}},
        {{"role", "assistant"}, {"content", "earlier answer"}},
    });
    ResponseRequest r;
    parse_ok(body, r);
    CHECK_EQ(r.chat.messages.size(), 2u);
    if (r.chat.messages.size() < 2) return;
    CHECK_EQ(r.chat.messages[0].role, std::string("user"));
    CHECK_EQ(r.chat.messages[0].content, std::string("hello"));
    CHECK_EQ(r.chat.messages[1].role, std::string("assistant"));

    json none = base();
    none["input"] = json::array({json{{"content", "no role, no type"}}});
    check_refusal(none, "invalid_request", "input", "an item with no type and no role");
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

TEST(responses_accepts_stream_and_obfuscation_options) {
    json body = base();
    body["stream"] = true;
    body["stream_options"] = {{"include_obfuscation", false}};
    ResponseRequest r;
    parse_ok(body, r);
    CHECK(r.chat.stream);
    // The non-stream form and an absent stream are the JSON body.
    ResponseRequest r2;
    parse_ok(base(), r2);
    CHECK(!r2.chat.stream);
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

TEST(responses_overflow_uses_the_spec_envelope_with_the_numbers) {
    // §3.8 stays a hard 400; the Responses surface spells it with the spec's
    // type and carries prompt_tokens/n_ctx at the top level (campaign Errors).
    Fixture fx(16);
    ResponseRequest r;
    parse_ok({{"model", "stub"},
              {"input", "one two three four five six seven eight nine ten eleven twelve "
                        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"}},
             r);
    PreparedResponse prep;
    auto err = api::prepare_response(fx.ctx, std::move(r), prep);
    CHECK(err.has_value());
    if (!err.has_value()) return;
    CHECK_EQ(err->status, 400);
    const json& b = err->body;
    CHECK_EQ(b["error"]["type"].get<std::string>(), std::string("invalid_request"));
    CHECK_EQ(b["error"]["code"].get<std::string>(), std::string("context_length_exceeded"));
    CHECK_EQ(b["error"]["param"].get<std::string>(), std::string("input"));
    CHECK(b["prompt_tokens"].is_number_integer());
    CHECK_EQ(b["n_ctx"].get<int>(), 16);
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
    // F2: the arcint extension key is present only when a loop-guard counter
    // moved; this stub turn moved none.
    CHECK(!body.contains("arcint"));

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
    // The reasoning block closed before the budget ran out, so that item stays
    // completed; only the open item (the message) is the last and incomplete.
    CHECK_EQ(incomplete["output"][0]["status"].get<std::string>(), std::string("completed"));
    CHECK_EQ(incomplete["output"].back()["status"].get<std::string>(), std::string("incomplete"));

    // A budget cut inside the reasoning block: the reasoning item is the only
    // output item, is last, and is incomplete (spec §Items).
    api::ChatGeneration open = gen;
    open.reasoning_closed   = false;
    open.content            = "";
    open.reason             = FinishReason::Length;
    const json open_body = api::response_resource(req, prep, open);
    CHECK_EQ(open_body["output"].size(), size_t(1));
    if (open_body["output"].empty()) return;
    CHECK_EQ(open_body["output"][0]["type"].get<std::string>(), std::string("reasoning"));
    CHECK_EQ(open_body["output"][0]["status"].get<std::string>(), std::string("incomplete"));

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

// ------------------------------------------------------------ SSE writer (W2)
namespace {

// Frames as (event name, body, [DONE] seen). The `event:` line is checked to be
// the body's type by the callers; a frame without one is an error.
struct Sse {
    std::vector<std::pair<std::string, json>> events;
    bool done = false;
};

Sse parse_sse(const std::vector<std::string>& raw) {
    Sse out;
    for (const std::string& frame : raw) {
        if (frame == "data: [DONE]\n\n") {
            out.done = true;
            continue;
        }
        if (frame.rfind("event: ", 0) != 0) {
            t::fail(__FILE__, __LINE__, "SSE frame without an event: line: " + frame);
            continue;
        }
        const size_t nl = frame.find('\n');
        const std::string name = frame.substr(7, nl - 7);
        const size_t data = frame.find("data: ", nl);
        if (data == std::string::npos) {
            t::fail(__FILE__, __LINE__, "SSE frame without data: " + frame);
            continue;
        }
        std::string payload = frame.substr(data + 6);
        while (!payload.empty() && payload.back() == '\n') payload.pop_back();
        out.events.emplace_back(name, json::parse(payload));
    }
    return out;
}

// The sequence numbers are 1..N with no gap, and every event's `event:` line is
// its body's `type`.
void check_framing(const Sse& s) {
    for (size_t i = 0; i < s.events.size(); ++i) {
        const auto& [name, body] = s.events[i];
        CHECK_EQ(name, body["type"].get<std::string>());
        CHECK_EQ(body["sequence_number"].get<int>(), static_cast<int>(i + 1));
    }
}

std::string think_close_tag() { return std::string("</") + "thi" + "nk" + ">"; }

// A programmable backend: exact pieces (and ids) in, one FinishReason out. It
// checks the stop flag as the stub and the llama backend do, and turns a failed
// write into Abort/ClientGone.
class PiecesBackend final : public Backend {
public:
    PiecesBackend() {
        st_.id          = "stub";
        st_.served_id   = "stub";
        st_.loaded      = true;
        st_.stub        = true;
        st_.n_ctx       = 4096;
        st_.n_ctx_train = 262144;
    }

    std::vector<std::pair<std::string, int>> pieces;
    int             delay_ms       = 0;   // lets a cancel arrive mid-decode
    FinishReason    reason         = FinishReason::Stop;
    bool            throw_after    = false;
    size_t          throw_at       = 0;
    GenerationStats counters{};

    const ModelStatus& status() const override { return st_; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest&) const override { return "p"; }

    FinishReason generate(const GenerationInput& in, int, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        stats.prompt_tokens = 4;
        if (in.stop != nullptr && in.stop->load(std::memory_order_relaxed)) {
            stats.abort_cause = AbortCause::StopSignal;
            return FinishReason::Abort;
        }
        for (size_t i = 0; i < pieces.size(); ++i) {
            if (throw_after && i == throw_at) throw std::runtime_error("backend exploded");
            if (in.cancel && in.cancel->load(std::memory_order_relaxed)) {
                stats.abort_cause = AbortCause::Cancelled;
                return FinishReason::Abort;
            }
            if (delay_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            const Control c = on_piece(pieces[i].first, pieces[i].second);
            ++stats.completion_tokens;
            if (c == Control::Cancel) {
                stats.abort_cause = AbortCause::ClientGone;
                return FinishReason::Abort;
            }
            if (c == Control::Stop) return FinishReason::Stop;
        }
        stats.loop_detections = counters.loop_detections;
        stats.loop_nudges     = counters.loop_nudges;
        stats.loop_slaps      = counters.loop_slaps;
        stats.loop_aborts     = counters.loop_aborts;
        stats.loop_self_closes = counters.loop_self_closes;
        stats.loop_door_masked = counters.loop_door_masked;
        stats.loop_door_eog_masked = counters.loop_door_eog_masked;
        return reason;
    }

private:
    struct Tok final : public Tokenizer {
        std::vector<int> encode(std::string_view text) override {
            return std::vector<int>(text.begin(), text.end());
        }
        std::string decode(const std::vector<int>& ids) override {
            return std::string(ids.begin(), ids.end());
        }
        std::string decode_one(int id) override { return std::string(1, static_cast<char>(id)); }
        int         eos_id() const override { return 0; }
    } tok_;
    ModelStatus st_;
};

// A prepared turn built by hand: the stream tests control exactly what the
// backend emits, so they do not depend on the stub's fixed reply.
struct StreamFixture {
    Config       cfg;
    PiecesBackend backend;
    SlotPool     slots{1};
    Context      ctx;
    ResponseRequest req;
    PreparedResponse prep;

    explicit StreamFixture(bool think_open = false, bool by_ids = false, bool tools = false) {
        cfg.stub   = true;
        ctx.cfg    = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
        prep.req         = req;
        prep.chat.id      = "resp-test";
        prep.chat.created = 1;
        prep.chat.model   = "stub";
        prep.chat.input.sampler.temperature = 0.0f;
        prep.chat.think_open = think_open;
        if (by_ids) prep.chat.think_ids.open_close_ids = {{248068, 248069}};
        if (tools) {
            prep.chat.parse_tool_calls = true;
            prep.chat.schemas = tool_schemas({{"read_file", "", json::object()}});
        }
    }

    std::vector<std::string> run(int fail_at = -1) {
        std::vector<std::string> raw;
        const int slot = 0;
        api::stream_response(ctx, prep, slot, [&](std::string_view frame) {
            if (fail_at >= 0 && static_cast<int>(raw.size()) == fail_at) return false;
            raw.emplace_back(frame);
            return true;
        });
        return raw;
    }
};

}  // namespace

TEST(responses_sse_plain_turn_has_the_lifecycle_and_framing) {
    StreamFixture fx;
    fx.backend.pieces = {{"Hello", 1}, {" world", 2}};
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    CHECK(s.done);
    // created, in_progress, item added, part added, two deltas, text done,
    // part done, item done, completed.
    CHECK_EQ(s.events.size(), size_t(10));
    if (s.events.size() < 10) return;
    const std::vector<std::string> types = {"response.created", "response.in_progress",
        "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.delta",
        "response.output_text.done", "response.content_part.done",
        "response.output_item.done", "response.completed"};
    for (size_t i = 0; i < types.size() && i < s.events.size(); ++i) {
        CHECK_EQ(s.events[i].first, types[i]);
    }
    // The opening snapshots are in_progress with no output and no usage.
    CHECK_EQ(s.events[0].second["response"]["status"].get<std::string>(), std::string("in_progress"));
    CHECK(s.events[0].second["response"]["output"].is_array());
    CHECK(s.events[0].second["response"]["output"].empty());
    CHECK(s.events[0].second["response"]["usage"].is_null());
    // The terminal resource carries the full answer and the usage block.
    const json& term = s.events.back().second["response"];
    CHECK_EQ(term["status"].get<std::string>(), std::string("completed"));
    if (term["output"].empty()) return;
    CHECK_EQ(term["output"][0]["content"][0]["text"].get<std::string>(), std::string("Hello world"));
    CHECK_EQ(term["usage"]["output_tokens"].get<int>(), 2);
    CHECK_EQ(term["usage"]["output_tokens_details"]["reasoning_tokens"].get<int>(), 0);
}

TEST(responses_sse_reasoning_closes_before_the_message) {
    StreamFixture fx(/*think_open=*/true);
    fx.backend.pieces = {{"So", 1}, {" far", 2}, {think_close_tag(), 3}, {"OK", 4}};
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    // The first reasoning event opens its item; the reasoning deltas precede
    // any output_text delta.
    size_t first_reasoning_delta = s.events.size(), first_text_delta = s.events.size();
    bool saw_reasoning_done = false, message_after_reasoning = false;
    for (size_t i = 0; i < s.events.size(); ++i) {
        const std::string& t = s.events[i].first;
        if (t == "response.reasoning.delta" && first_reasoning_delta == s.events.size())
            first_reasoning_delta = i;
        if (t == "response.output_text.delta" && first_text_delta == s.events.size())
            first_text_delta = i;
        if (t == "response.reasoning.done") saw_reasoning_done = true;
        if (t == "response.output_item.added" && saw_reasoning_done &&
            s.events[i].second["item"]["type"].get<std::string>() == "message")
            message_after_reasoning = true;
    }
    CHECK(first_reasoning_delta < first_text_delta);
    CHECK(saw_reasoning_done);
    CHECK(message_after_reasoning);
    const json& term = s.events.back().second["response"];
    CHECK_EQ(term["output"].size(), size_t(2));
    if (term["output"].size() < 2) return;
    CHECK_EQ(term["output"][0]["type"].get<std::string>(), std::string("reasoning"));
    CHECK_EQ(term["output"][0]["content"][0]["text"].get<std::string>(), std::string("So far"));
    CHECK_EQ(term["output"][1]["content"][0]["text"].get<std::string>(), std::string("OK"));
}

TEST(responses_sse_function_call_name_before_arguments) {
    StreamFixture fx(false, false, /*tools=*/true);
    const std::string call = "Sure\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": "
                             "{\"path\": \"a\"}}\n</tool_call>";
    fx.backend.pieces = {{"Sure", 1}, {"\n", 2}, {"<tool", 3}, {"_call", 4}, {">", 5},
                         {"\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}\n</tool", 6},
                         {"_call>", 7}};
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    // The call item is added with name and call_id before the argument delta.
    int added = -1, delta = -1, done_args = -1, item_done = -1;
    for (size_t i = 0; i < s.events.size(); ++i) {
        const std::string& t = s.events[i].first;
        if (t == "response.output_item.added" &&
            s.events[i].second["item"]["type"].get<std::string>() == "function_call" &&
            added < 0) {
            added = static_cast<int>(i);
            CHECK_EQ(s.events[i].second["item"]["name"].get<std::string>(), std::string("read_file"));
            CHECK(!s.events[i].second["item"]["call_id"].get<std::string>().empty());
            CHECK_EQ(s.events[i].second["item"]["arguments"].get<std::string>(), std::string(""));
        }
        if (t == "response.function_call_arguments.delta" && delta < 0)
            delta = static_cast<int>(i);
        if (t == "response.function_call_arguments.done") done_args = static_cast<int>(i);
        if (t == "response.output_item.done" &&
            s.events[i].second["item"]["type"].get<std::string>() == "function_call")
            item_done = static_cast<int>(i);
    }
    CHECK(added >= 0);
    CHECK(added < delta);
    CHECK(delta < done_args);
    CHECK(done_args < item_done);
    if (s.events.back().second["response"]["output"].size() < 2) return;
    CHECK(s.events.back().second["response"]["output"][1]["type"] == json("function_call"));
    CHECK_EQ(s.events.back().second["response"]["status"].get<std::string>(), std::string("completed"));
}

TEST(responses_sse_length_is_incomplete_and_the_open_item_is_last) {
    StreamFixture fx;
    fx.backend.pieces = {{"cut", 1}};
    fx.backend.reason = FinishReason::Length;
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    CHECK(s.done);
    CHECK_EQ(s.events.back().first, std::string("response.incomplete"));
    const json& term = s.events.back().second["response"];
    CHECK_EQ(term["status"].get<std::string>(), std::string("incomplete"));
    CHECK_EQ(term["incomplete_details"]["reason"].get<std::string>(), std::string("max_output_tokens"));
    if (term["output"].empty()) return;
    CHECK_EQ(term["output"].back()["status"].get<std::string>(), std::string("incomplete"));
}

TEST(responses_sse_loop_emits_the_counters_then_incomplete) {
    StreamFixture fx;
    fx.backend.pieces = {{"x", 1}};
    fx.backend.reason = FinishReason::Loop;
    fx.backend.counters.loop_detections = 3;
    fx.backend.counters.loop_nudges     = 2;
    fx.backend.counters.loop_slaps      = 1;
    fx.backend.counters.loop_aborts     = 1;
    fx.backend.counters.loop_self_closes = 1;
    fx.backend.counters.loop_door_masked = 4;
    fx.backend.counters.loop_door_eog_masked = 5;
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    // The loop-guard event is before the terminal incomplete, and carries v and
    // the seven counters.
    int guard = -1, terminal = -1;
    for (size_t i = 0; i < s.events.size(); ++i) {
        if (s.events[i].first == "arcint:loop_guard") guard = static_cast<int>(i);
        if (s.events[i].first == "response.incomplete") terminal = static_cast<int>(i);
    }
    CHECK(guard >= 0);
    if (guard < 0) return;
    CHECK(guard < terminal);
    const json& g = s.events[guard].second;
    CHECK_EQ(g["v"].get<int>(), 1);
    for (const char* k : {"detections", "nudges", "slaps", "aborts", "self_closes",
                          "door_masked", "door_eog_masked"})
        CHECK(g.contains(k));
    if (guard < 0) return;
    CHECK_EQ(s.events.back().second["response"]["incomplete_details"]["reason"].get<std::string>(),
             std::string("loop_guard_abort"));
}

TEST(responses_sse_exception_is_error_then_failed) {
    StreamFixture fx;
    fx.backend.pieces     = {{"partial", 1}, {"more", 2}};
    fx.backend.throw_after = true;
    fx.backend.throw_at    = 1;
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    CHECK(s.done);
    // An open item closes incomplete first, then the error event, then failed.
    int error = -1, failed_i = -1, item_done = -1;
    for (size_t i = 0; i < s.events.size(); ++i) {
        const std::string& t = s.events[i].first;
        if (t == "error" && error < 0) error = static_cast<int>(i);
        if (t == "response.output_item.done" && item_done < 0) item_done = static_cast<int>(i);
        if (t == "response.failed") failed_i = static_cast<int>(i);
    }
    CHECK(item_done >= 0);
    CHECK(item_done < error);
    CHECK(error < failed_i);
    const json& err = s.events[error].second["error"];
    CHECK_EQ(err["type"].get<std::string>(), std::string("server_error"));
    CHECK_EQ(err["code"].get<std::string>(), std::string("internal_error"));
    const json& term = s.events.back().second["response"];
    CHECK_EQ(term["status"].get<std::string>(), std::string("failed"));
    CHECK_EQ(term["error"]["code"].get<std::string>(), std::string("internal_error"));
    if (term["output"].empty()) return;
    CHECK_EQ(term["output"].back()["status"].get<std::string>(), std::string("incomplete"));
}

TEST(responses_sse_stop_signal_fails_with_server_stopping) {
    StreamFixture fx;
    std::atomic<bool> stop{true};
    fx.ctx.stop_flag      = &stop;
    fx.prep.chat.input.stop = &stop;   // the route's finish_prepare sets this
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    CHECK(s.done);
    CHECK_EQ(s.events.back().first, std::string("response.failed"));
    CHECK_EQ(s.events.back().second["response"]["status"].get<std::string>(), std::string("failed"));
    CHECK_EQ(s.events.back().second["response"]["error"]["code"].get<std::string>(),
             std::string("server_stopping"));
}

TEST(responses_sse_client_gone_ends_the_stream_silently) {
    StreamFixture fx;
    fx.backend.pieces = {{"Hello", 1}, {" world", 2}, {" again", 3}};
    // The sink fails at the fourth frame (the content-part added), as a closed
    // socket's write does.
    const std::vector<std::string> raw = fx.run(/*fail_at=*/3);
    const Sse s = parse_sse(raw);
    check_framing(s);
    CHECK(!s.done);
    CHECK_EQ(s.events.size(), size_t(3));   // created, in_progress, item added
    for (const auto& [name, body] : s.events) {
        CHECK(name != "response.completed" && name != "response.failed" &&
              name != "response.incomplete");
    }
}

TEST(responses_sse_counts_reasoning_tokens_where_the_split_is_by_ids) {
    StreamFixture fx(/*think_open=*/true, /*by_ids=*/true);
    // Two reasoning tokens, the closer's id, then one answer token.
    fx.backend.pieces = {{"a", 10}, {"b", 11}, {think_close_tag(), 248069}, {"c", 12}};
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    const json& term = s.events.back().second["response"];
    if (term["output"].size() < 2) return;
    CHECK_EQ(term["output"][0]["type"].get<std::string>(), std::string("reasoning"));
    CHECK_EQ(term["output"][0]["content"][0]["text"].get<std::string>(), std::string("ab"));
    CHECK_EQ(term["output"][1]["content"][0]["text"].get<std::string>(), std::string("c"));
    // R7: the id split counts the two reasoning pieces; 0 only for text splits.
    CHECK_EQ(term["usage"]["output_tokens_details"]["reasoning_tokens"].get<int>(), 2);
}

TEST(responses_non_stream_counts_reasoning_tokens_from_the_split) {
    // The same id-split pieces through the non-streamed builder: R7 applies to
    // both paths, not only the stream.
    StreamFixture fx(/*think_open=*/true, /*by_ids=*/true);
    fx.backend.pieces = {{"a", 10}, {"b", 11}, {think_close_tag(), 248069}, {"c", 12}};
    const auto lease = fx.slots.acquire_for(0);
    const api::HttpResult r = api::run_response(fx.ctx, fx.prep, lease.index());
    CHECK_EQ(r.status, 200);
    if (r.status != 200) return;
    CHECK_EQ(r.body["usage"]["output_tokens_details"]["reasoning_tokens"].get<int>(), 2);
    if (r.body["output"].empty()) return;
    CHECK_EQ(r.body["output"][0]["content"][0]["text"].get<std::string>(), std::string("ab"));
}

TEST(responses_busy_refusal_keeps_the_slots_diagnostics) {
    // Review F1: the 429 carries the chat 503's slots (and lanes) numbers.
    Fixture fx;
    const auto held = fx.slots.acquire_for(0);
    CHECK_EQ(held.index(), 0);
    api::SlotPool::Lease lease;
    int slot = -1;
    auto err = api::acquire_slot(fx.ctx, -1, lease, slot);
    CHECK(err.has_value());
    if (!err.has_value()) return;
    CHECK_EQ(err->status, 503);
    CHECK(err->body.contains("slots"));
    const api::HttpResult busy = api::responses_busy(*err);
    CHECK_EQ(busy.status, 429);
    CHECK_EQ(busy.body["error"]["type"].get<std::string>(), std::string("too_many_requests"));
    CHECK_EQ(busy.body["error"]["code"].get<std::string>(), std::string("all_lanes_busy"));
    CHECK(busy.body.contains("slots"));
    if (!busy.body.contains("slots")) return;
    CHECK_EQ(busy.body["slots"]["total"].get<int>(), 1);
    CHECK_EQ(busy.body["slots"]["free"].get<int>(), 0);
}

// ------------------------------------------------------- state (W3)
namespace {

std::shared_ptr<api::StoredResponse> stored_entry(const std::string& id, int lane,
                                                  const std::string& prompt = "p") {
    auto e       = std::make_shared<api::StoredResponse>();
    e->id        = id;
    e->lane      = lane;
    e->status    = "completed";
    e->prompt    = prompt;
    e->prompt_ids.assign(prompt.begin(), prompt.end());
    e->prefix_hash = api::sha256_hex(prompt);
    e->output_items = json::array({json{{"type", "message"}, {"id", "msg-1"}, {"role", "assistant"},
                                        {"status", "completed"},
                                        {"content", json::array({{{"type", "output_text"},
                                                                  {"text", "ok"}}})}}});
    return e;
}

// A hand-built store for the W3 cases that do not need a server.
api::ResponseStore::Limits limits(int entries, uint64_t bytes = 1ull << 20, int64_t ttl = 600) {
    api::ResponseStore::Limits l;
    l.entries = entries;
    l.bytes   = bytes;
    l.ttl_s   = ttl;
    return l;
}

}  // namespace

TEST(responses_store_lru_and_bytes_limits_per_lane) {
    api::ResponseStore store(limits(2));
    store.put(stored_entry("a", 0));
    store.put(stored_entry("b", 0));
    store.put(stored_entry("c", 0));   // 3 entries, limit 2: `a` (LRU) goes
    CHECK_EQ(store.entries(), 2);
    CHECK(store.get("a", 0) == nullptr);
    CHECK(store.get("b", 0) != nullptr);
    CHECK(store.get("c", 0) != nullptr);

    // The byte limit binds first when the entries fit: two ~950-byte entries
    // under a 1000-byte limit leave one, and the older one is evicted.
    api::ResponseStore small(limits(32, 1000));
    std::string        text(100, 'x');
    small.put(stored_entry("a", 0, text));
    small.put(stored_entry("b", 0, text));
    CHECK_EQ(small.entries(), 1);
    CHECK(small.get("a", 0) == nullptr);
    CHECK(small.get("b", 0) != nullptr);
    CHECK(small.bytes() > 0);
}

TEST(responses_store_ttl_sweeps_expired_entries) {
    api::ResponseStore store(limits(32, 1ull << 20, /*ttl=*/10));
    store.put(stored_entry("a", 0));
    const int64_t now = static_cast<int64_t>(std::time(nullptr));
    store.sweep(now + 5);
    CHECK(store.get("a", 0) != nullptr);
    store.sweep(now + 11);
    CHECK_EQ(store.entries(), 0);
}

TEST(responses_store_lane_scoped_and_entries_zero) {
    api::ResponseStore store(limits(32));
    store.put(stored_entry("a", 0));
    CHECK(store.get("a", 1) == nullptr);   // wrong lane: not found
    CHECK(store.get("a", 0) != nullptr);

    // Review R4: --responses-store-entries 0 retains nothing.
    api::ResponseStore off(limits(0));
    off.put(stored_entry("a", 0));
    CHECK_EQ(off.entries(), 0);
    CHECK(off.get("a", 0) == nullptr);
}

TEST(responses_slot_try_acquire_is_non_blocking) {
    SlotPool pool(2);
    const auto first = pool.try_acquire(0);
    CHECK_EQ(first.index(), 0);
    const auto busy = pool.try_acquire(0);
    CHECK_EQ(busy.index(), -1);   // taken: refused, not waited for
    const auto other = pool.try_acquire(1);
    CHECK_EQ(other.index(), 1);
}

TEST(responses_a_two_turn_continuation_rebuilds_and_echoes) {
    Fixture fx;
    fx.ctx.responses = std::make_shared<api::ResponseStore>(
        api::ResponseStore::Limits{32, 1ull << 20, 600});

    // Turn 1: a tool turn, so the stored assistant turn carries a function_call.
    json b1 = {{"model", "stub"}, {"input", "read it"}, {"max_output_tokens", 256},
               {"tools", json::array({{{"type", "function"}, {"name", "read_file"},
                                       {"parameters", {{"type", "object"},
                                                       {"properties", {{"path", {{"type", "string"}}}}}}}}})}};
    ResponseRequest r1;
    parse_ok(b1, r1);
    PreparedResponse p1;
    CHECK(!api::prepare_response(fx.ctx, std::move(r1), p1).has_value());
    const api::HttpResult res1 = api::run_response(fx.ctx, p1, 0);
    CHECK_EQ(res1.status, 200);
    if (res1.status != 200) return;
    const std::string id = res1.body["id"].get<std::string>();
    std::string call_id;
    for (const json& item : res1.body["output"])
        if (item["type"].get<std::string>() == "function_call")
            call_id = item["call_id"].get<std::string>();
    CHECK(!call_id.empty());
    CHECK(fx.ctx.responses->get(id, -1) != nullptr);

    // Turn 2: previous_response_id plus the function_call_output.
    json b2 = {{"model", "stub"}, {"previous_response_id", id}, {"max_output_tokens", 256},
               {"tools", json::array({{{"type", "function"}, {"name", "read_file"},
                                       {"parameters", {{"type", "object"},
                                                       {"properties", {{"path", {{"type", "string"}}}}}}}}})},
               {"input", json::array({{{"type", "function_call_output"},
                                       {"call_id", call_id}, {"output", "alpha"}}})}};
    ResponseRequest r2;
    parse_ok(b2, r2);
    PreparedResponse p2;
    const auto err = api::prepare_response(fx.ctx, std::move(r2), p2);
    CHECK(!err.has_value());
    if (err.has_value()) return;
    // Stored input messages (1 user) + the rebuilt assistant turn + the new
    // tool output.
    CHECK_EQ(p2.chat.req.messages.size(), 3u);
    if (p2.chat.req.messages.size() < 3) return;
    CHECK_EQ(p2.chat.req.messages[1].role, std::string("assistant"));
    CHECK_EQ(p2.chat.req.messages[1].tool_calls.size(), size_t(1));
    CHECK_EQ(p2.chat.req.messages[2].role, std::string("tool"));
    CHECK_EQ(p2.chat.req.messages[2].tool_call_id, call_id);
    CHECK_EQ(p2.chat.req.messages[2].content, std::string("alpha"));
    CHECK(p2.affinity_slot >= 0);

    const api::HttpResult res2 = api::run_response(fx.ctx, p2, 0);
    CHECK_EQ(res2.status, 200);
    if (res2.status != 200) return;
    CHECK_EQ(res2.body["previous_response_id"].get<std::string>(), id);
}

TEST(responses_store_false_keeps_nothing) {
    Fixture fx;
    fx.ctx.responses = std::make_shared<api::ResponseStore>(
        api::ResponseStore::Limits{32, 1ull << 20, 600});
    json b1 = base();
    b1["store"] = false;
    ResponseRequest r1;
    parse_ok(b1, r1);
    PreparedResponse p1;
    CHECK(!api::prepare_response(fx.ctx, std::move(r1), p1).has_value());
    const api::HttpResult res1 = api::run_response(fx.ctx, p1, 0);
    CHECK_EQ(res1.status, 200);
    const std::string id = res1.body["id"].get<std::string>();
    CHECK(fx.ctx.responses->get(id, -1) == nullptr);

    json b2 = base();
    b2["previous_response_id"] = id;
    ResponseRequest r2;
    parse_ok(b2, r2);
    PreparedResponse p2;
    auto err = api::prepare_response(fx.ctx, std::move(r2), p2);
    CHECK(err.has_value());
    if (!err.has_value()) return;
    CHECK_EQ(err->body["error"]["code"].get<std::string>(), std::string("previous_response_not_found"));
}

TEST(responses_continuation_prefix_mismatch_is_not_found) {
    Fixture fx;
    fx.ctx.responses = std::make_shared<api::ResponseStore>(
        api::ResponseStore::Limits{32, 1ull << 20, 600});
    auto e = stored_entry("resp-x", -1, "a prompt this server no longer renders");
    e->prefix_hash = std::string(64, '0');   // a hash the current render never yields
    fx.ctx.responses->put(e);

    json b = base();
    b["previous_response_id"] = "resp-x";
    ResponseRequest r;
    parse_ok(b, r);
    PreparedResponse p;
    auto err = api::prepare_response(fx.ctx, std::move(r), p);
    CHECK(err.has_value());
    if (!err.has_value()) return;
    CHECK_EQ(err->body["error"]["code"].get<std::string>(), std::string("previous_response_not_found"));
    CHECK(fx.ctx.responses->get("resp-x", -1) == nullptr);   // a 400 evicts the chain
}

TEST(responses_r4_entries_zero_refuses_every_continuation) {
    // Review R4: entries=0 disables retention, so a continuation always answers
    // previous_response_not_found even though the store exists.
    Fixture fx;
    fx.ctx.responses = std::make_shared<api::ResponseStore>(
        api::ResponseStore::Limits{0, 1ull << 20, 600});
    json b = base();
    b["previous_response_id"] = "resp-anything";
    ResponseRequest r;
    parse_ok(b, r);
    PreparedResponse p;
    auto err = api::prepare_response(fx.ctx, std::move(r), p);
    CHECK(err.has_value());
    if (!err.has_value()) return;
    CHECK_EQ(err->body["error"]["code"].get<std::string>(), std::string("previous_response_not_found"));
}

TEST(responses_store_limits_are_per_lane) {
    api::ResponseStore store(limits(1));
    store.put(stored_entry("a", 0));
    store.put(stored_entry("b", 1));   // another lane: its own one-entry budget
    CHECK_EQ(store.entries(), 2);
    CHECK(store.get("a", 0) != nullptr);
    CHECK(store.get("b", 1) != nullptr);
}

TEST(responses_slot_affinity_tries_the_stored_slot_then_falls_back) {
    Fixture fx;
    auto other = fx.slots.try_acquire(0);
    CHECK_EQ(other.index(), 0);   // the affinity slot is taken
    api::SlotPool::Lease lease;
    int slot = -1;
    auto err = api::acquire_slot(fx.ctx, -1, lease, slot, /*affinity=*/0);
    CHECK(err.has_value());   // falls back to the ordinary acquire, which is busy
    if (err.has_value()) CHECK_EQ(err->status, 503);
    other = api::SlotPool::Lease();   // release

    // Free again: the affinity look now wins without the fallback.
    api::SlotPool::Lease lease2;
    int slot2 = -1;
    CHECK(!api::acquire_slot(fx.ctx, -1, lease2, slot2, /*affinity=*/0).has_value());
    CHECK_EQ(slot2, 0);
}

TEST(responses_bogus_previous_id_is_not_found) {
    Fixture fx;
    fx.ctx.responses = std::make_shared<api::ResponseStore>(
        api::ResponseStore::Limits{32, 1ull << 20, 600});
    json b = base();
    b["previous_response_id"] = "resp-never";
    ResponseRequest r;
    parse_ok(b, r);
    PreparedResponse p;
    auto err = api::prepare_response(fx.ctx, std::move(r), p);
    CHECK(err.has_value());
    if (!err.has_value()) return;
    CHECK_EQ(err->body["error"]["code"].get<std::string>(), std::string("previous_response_not_found"));
}

TEST(responses_a_failed_continuation_evicts_the_entry) {
    StreamFixture fx;   // PiecesBackend: it can throw on demand
    fx.ctx.responses = std::make_shared<api::ResponseStore>(
        api::ResponseStore::Limits{32, 1ull << 20, 600});
    fx.backend.pieces = {{"ok", 1}, {" done", 2}};
    json b1 = base();
    ResponseRequest r1;
    parse_ok(b1, r1);
    PreparedResponse p1;
    CHECK(!api::prepare_response(fx.ctx, std::move(r1), p1).has_value());
    const api::HttpResult res1 = api::run_response(fx.ctx, p1, 0);
    CHECK_EQ(res1.status, 200);
    if (res1.status != 200) return;
    const std::string id = res1.body["id"].get<std::string>();
    CHECK(fx.ctx.responses->get(id, -1) != nullptr);

    // Turn 2 continues and the engine throws: the server answered 500, so the
    // state it continued from is broken and must not be reused.
    fx.backend.throw_after = true;
    fx.backend.throw_at    = 0;
    json b2 = base();
    b2["previous_response_id"] = id;
    ResponseRequest r2;
    parse_ok(b2, r2);
    PreparedResponse p2;
    auto err = api::prepare_response(fx.ctx, std::move(r2), p2);
    CHECK(!err.has_value());
    if (err.has_value()) return;
    const api::HttpResult res2 = api::run_response(fx.ctx, p2, 0);
    CHECK_EQ(res2.status, 500);
    CHECK(fx.ctx.responses->get(id, -1) == nullptr);
}

// ------------------------------------------------------- F2 (W2 review)
TEST(responses_stream_sends_loop_guard_when_any_counter_moved) {
    StreamFixture fx;
    fx.backend.pieces = {{"x", 1}};
    fx.backend.counters.loop_detections = 1;   // moved, but the turn ended Stop
    const Sse s = parse_sse(fx.run());
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    bool guard = false;
    CHECK_EQ(s.events.back().first, std::string("response.completed"));
    for (size_t i = 0; i < s.events.size(); ++i) {
        if (s.events[i].first == "arcint:loop_guard") {
            guard = true;
            CHECK_EQ(s.events[i].second["detections"].get<int>(), 1);
        }
    }
    CHECK(guard);   // parity with the chat route's loop_guard object
}

TEST(responses_non_stream_resource_carries_arcint_loop_guard_when_moved) {
    StreamFixture fx;
    fx.backend.pieces = {{"x", 1}};
    fx.backend.counters.loop_nudges = 2;
    const auto lease = fx.slots.acquire_for(0);
    const api::HttpResult r = api::run_response(fx.ctx, fx.prep, lease.index());
    CHECK_EQ(r.status, 200);
    if (r.status != 200) return;
    CHECK(r.body.contains("arcint"));
    if (!r.body.contains("arcint")) return;
    CHECK_EQ(r.body["arcint"]["loop_guard"]["nudges"].get<int>(), 2);
}

// ------------------------------------------------ instructions (W3.1)
namespace {

// Turn 1 with `instructions`, stored on the stub; returns the response id.
std::string stored_with_instructions(Fixture& fx, const std::string& instructions) {
    fx.ctx.responses = std::make_shared<api::ResponseStore>(
        api::ResponseStore::Limits{32, 1ull << 20, 600});
    json b = {{"model", "stub"}, {"input", "first"}, {"instructions", instructions},
              {"max_output_tokens", 256}};
    ResponseRequest r;
    parse_ok(b, r);
    PreparedResponse p;
    CHECK(!api::prepare_response(fx.ctx, std::move(r), p).has_value());
    const api::HttpResult res = api::run_response(fx.ctx, p, 0);
    CHECK_EQ(res.status, 200);
    return res.status == 200 ? res.body["id"].get<std::string>() : std::string();
}

}  // namespace

TEST(responses_continuation_resending_the_same_instructions_keeps_one_head) {
    // OpenAI-compatible clients send their full system prompt as `instructions`
    // on every turn. Resending the same text must not duplicate it: the stored
    // head is replaced by the identical message, so the shared prefix bytes (and
    // the slot-affinity prefix hash) stay the same.
    Fixture fx;
    const std::string id = stored_with_instructions(fx, "be terse");
    CHECK(!id.empty());
    if (id.empty()) return;

    json b = {{"model", "stub"}, {"previous_response_id", id}, {"instructions", "be terse"},
              {"input", "second"}, {"max_output_tokens", 256}};
    ResponseRequest r;
    parse_ok(b, r);
    PreparedResponse p;
    const auto err = api::prepare_response(fx.ctx, std::move(r), p);
    CHECK(!err.has_value());
    if (err.has_value()) return;
    int systems = 0;
    for (const ChatMessage& m : p.chat.req.messages) if (m.role == "system") ++systems;
    CHECK_EQ(systems, 1);   // the head, once
    CHECK_EQ(p.chat.req.messages[0].role, std::string("system"));
    CHECK_EQ(p.chat.req.messages[0].content, std::string("be terse"));
    // The rendered head+stored half is still the stored prompt's bytes: the
    // current renderer reproduces the stored prefix (checked in prepare) and the
    // head is the same message in the same position.
    CHECK_EQ(fx.ctx.responses->get(id, -1)->prefix_hash,
             api::sha256_hex(fx.backend->render_chat(fx.ctx.responses->get(id, -1)->chat)));
}

TEST(responses_continuation_changed_instructions_replace_the_head) {
    // A new system prompt replaces the stored head in place (first), never
    // appended after the assistant turn: Qwen-family templates reject a system
    // message that is not first.
    Fixture fx;
    const std::string id = stored_with_instructions(fx, "be terse");
    CHECK(!id.empty());
    if (id.empty()) return;

    json b = {{"model", "stub"}, {"previous_response_id", id}, {"instructions", "be verbose"},
              {"input", "second"}, {"max_output_tokens", 256}};
    ResponseRequest r;
    parse_ok(b, r);
    PreparedResponse p;
    const auto err = api::prepare_response(fx.ctx, std::move(r), p);
    CHECK(!err.has_value());
    if (err.has_value()) return;
    CHECK_EQ(p.chat.req.messages[0].role, std::string("system"));
    CHECK_EQ(p.chat.req.messages[0].content, std::string("be verbose"));
    int systems = 0;
    for (const ChatMessage& m : p.chat.req.messages) if (m.role == "system") ++systems;
    CHECK_EQ(systems, 1);
    CHECK(p.chat.req.messages[1].content != std::string("be terse"));
}

TEST(responses_continuation_without_instructions_keeps_the_stored_head) {
    // No instructions on the new request means "no new guidance", not "drop the
    // stored system prompt": the stored head stays first, and no system message
    // is added.
    Fixture fx;
    const std::string id = stored_with_instructions(fx, "be terse");
    CHECK(!id.empty());
    if (id.empty()) return;

    json b = {{"model", "stub"}, {"previous_response_id", id}, {"input", "second"},
              {"max_output_tokens", 256}};
    ResponseRequest r;
    parse_ok(b, r);
    PreparedResponse p;
    const auto err = api::prepare_response(fx.ctx, std::move(r), p);
    CHECK(!err.has_value());
    if (err.has_value()) return;
    CHECK_EQ(p.chat.req.messages[0].role, std::string("system"));
    CHECK_EQ(p.chat.req.messages[0].content, std::string("be terse"));
    int systems = 0;
    for (const ChatMessage& m : p.chat.req.messages) if (m.role == "system") ++systems;
    CHECK_EQ(systems, 1);
}

// ---------------------------------------------------- cancel by id (W4)
TEST(responses_in_flight_registry_registers_and_removes) {
    api::ResponseStore store(api::ResponseStore::Limits{32, 1ull << 20, 600});
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    auto entry = std::make_shared<api::InFlightResponse>();
    entry->id     = "resp-flight";
    entry->lane   = 0;
    entry->cancel = cancel;
    entry->prep   = std::make_shared<api::PreparedResponse>();
    store.begin_in_flight(entry);

    const auto found = store.find_in_flight("resp-flight");
    CHECK(found != nullptr);
    if (!found) return;
    CHECK_EQ(found->lane, 0);
    found->cancel->store(true, std::memory_order_relaxed);
    CHECK(cancel->load(std::memory_order_relaxed));
    CHECK(store.find_in_flight("resp-other") == nullptr);

    // store=false registers while in flight and removes at the end: the
    // registry is not the store.
    CHECK_EQ(store.entries(), 0);
    store.end_in_flight("resp-flight");
    CHECK(store.find_in_flight("resp-flight") == nullptr);
}

TEST(responses_stream_cancel_emits_cancelled_then_incomplete) {
    StreamFixture fx;
    fx.backend.pieces   = {{"one", 1}, {" two", 2}, {" three", 3}, {" four", 4},
                           {" five", 5}, {" six", 6}};
    fx.backend.delay_ms = 15;
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    fx.prep.chat.input.cancel = cancel;
    fx.prep.chat.input.peer_alive = [cancel] {
        return !cancel->load(std::memory_order_relaxed);
    };

    std::vector<std::string> raw;
    api::stream_response(fx.ctx, fx.prep, 0, [&](std::string_view frame) {
        raw.emplace_back(frame);
        // The client asks to cancel after the first answer delta.
        if (std::string_view(frame).find("\"response.output_text.delta\"") != std::string_view::npos)
            cancel->store(true, std::memory_order_relaxed);
        return true;
    });
    const Sse s = parse_sse(raw);
    check_framing(s);
    CHECK(!s.events.empty());
    if (s.events.empty()) return;
    CHECK(s.done);
    int cancelled = -1, terminal = -1, item_done = -1;
    for (size_t i = 0; i < s.events.size(); ++i) {
        const std::string& t = s.events[i].first;
        if (t == "arcint:response.cancelled" && cancelled < 0) cancelled = static_cast<int>(i);
        if (t == "response.incomplete") terminal = static_cast<int>(i);
        if (t == "response.output_item.done") item_done = static_cast<int>(i);
    }
    CHECK(cancelled >= 0);
    if (cancelled < 0) return;
    CHECK_EQ(s.events[cancelled].second["v"].get<int>(), 1);
    CHECK(item_done < cancelled);   // the open item closes before the terminal
    CHECK(cancelled < terminal);
    CHECK_EQ(s.events[terminal].second["response"]["status"].get<std::string>(),
             std::string("incomplete"));
    CHECK_EQ(s.events[terminal].second["response"]["incomplete_details"]["reason"].get<std::string>(),
             std::string("cancelled"));
    CHECK_EQ(s.events[terminal].second["response"]["output"].back()["status"].get<std::string>(),
             std::string("incomplete"));
}

TEST(responses_non_stream_cancel_returns_the_incomplete_resource) {
    StreamFixture fx;
    fx.backend.pieces   = {{"one", 1}, {" two", 2}, {" three", 3}, {" four", 4}};
    fx.backend.delay_ms = 30;
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    fx.prep.chat.input.cancel = cancel;
    fx.prep.chat.input.peer_alive = [cancel] {
        return !cancel->load(std::memory_order_relaxed);
    };

    api::HttpResult result;
    std::thread run([&] { result = api::run_response(fx.ctx, fx.prep, 0); });
    std::this_thread::sleep_for(std::chrono::milliseconds(70));
    cancel->store(true, std::memory_order_relaxed);
    run.join();

    CHECK_EQ(result.status, 200);
    if (result.status != 200) return;
    CHECK_EQ(result.body["status"].get<std::string>(), std::string("incomplete"));
    CHECK_EQ(result.body["incomplete_details"]["reason"].get<std::string>(), std::string("cancelled"));
    CHECK_EQ(result.body["output"].back()["status"].get<std::string>(), std::string("incomplete"));
}

TEST(responses_stub_cancel_is_named_and_aborts_at_the_prefill_boundary) {
    // The stub proof: make_stub_backend(..., prefill_ubatches=4), the cancel
    // flag set before the run. The backend sees it at its first ubatch boundary
    // (the same check the real backend makes), names the cause Cancelled, and
    // the turn returns the incomplete resource.
    auto backend = make_stub_backend(4096, 0, {}, {}, {}, /*prefill_ubatches=*/4);
    SlotPool slots(1);
    Config   cfg;
    cfg.stub = true;
    Context ctx;
    ctx.cfg = &cfg;
    ctx.backend = backend.get();
    ctx.slots   = &slots;
    ctx.responses = std::make_shared<api::ResponseStore>(
        api::ResponseStore::Limits{32, 1ull << 20, 600});

    PreparedResponse prep;
    prep.req         = ResponseRequest{};
    prep.chat.id     = "resp-cancel";
    prep.chat.model  = "stub";
    prep.chat.input.prompt = "a prompt a prefill holds";
    auto cancel = std::make_shared<std::atomic<bool>>(true);
    prep.chat.input.cancel = cancel;
    prep.chat.input.peer_alive = [cancel] {
        return !cancel->load(std::memory_order_relaxed);
    };
    prep.affinity_slot = -1;

    const auto lease = slots.acquire_for(0);
    const api::HttpResult r = api::run_response(ctx, prep, lease.index());
    CHECK_EQ(r.status, 200);
    if (r.status != 200) return;
    CHECK(r.body.is_object() && r.body.contains("status"));
    if (!(r.body.is_object() && r.body.contains("status"))) return;
    CHECK_EQ(r.body["status"].get<std::string>(), std::string("incomplete"));
    CHECK_EQ(r.body["incomplete_details"]["reason"].get<std::string>(), std::string("cancelled"));
    // Nothing decoded: the abort was at the first prefill boundary, not after
    // the whole prompt had been forwarded.
    CHECK_EQ(r.body["usage"]["output_tokens"].get<int>(), 0);
}
