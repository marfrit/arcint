#include "core/chat.h"
#include "harness.h"

using namespace lgc;
using nlohmann::json;

namespace {
json minimal_chat() {
    return json{{"model", "qwen3.8-27b"},
                {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
}
}  // namespace

TEST(chat_parses_a_minimal_request) {
    ChatRequest req;
    CHECK(!parse_chat_request(minimal_chat(), req).has_value());
    CHECK_EQ(req.messages.size(), 1u);
    CHECK_EQ(req.messages[0].role, std::string("user"));
    CHECK_EQ(req.messages[0].content, std::string("hi"));
    CHECK(!req.stream);
    CHECK(req.enable_thinking);
    CHECK(!req.has_enable_thinking);
}

TEST(chat_rejects_structurally_broken_requests) {
    auto rejected = [](json body) {
        ChatRequest req;
        return parse_chat_request(body, req).has_value();
    };

    CHECK(rejected(json::array()));
    CHECK(rejected(json::object()));                                   // no messages
    CHECK(rejected(json{{"messages", json::array()}}));                // empty
    CHECK(rejected(json{{"messages", json::array({json::object()})}}));  // no role
    CHECK(rejected(json{{"messages", json::array({{{"role", "wizard"}, {"content", "x"}}})}}));
    CHECK(rejected(json{{"messages", json::array({{{"role", "tool"}, {"content", "x"}}})}}));
}

TEST(chat_accepts_typed_text_content_parts) {
    json body = minimal_chat();
    body["messages"][0]["content"] =
        json::array({{{"type", "text"}, {"text", "a"}}, {{"type", "text"}, {"text", "b"}}});

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.messages[0].content, std::string("ab"));
}

TEST(chat_rejects_non_text_content_parts) {
    // v1 is text-only; a dropped image part would silently change what the
    // model was asked.
    json body                      = minimal_chat();
    body["messages"][0]["content"] = json::array({{{"type", "image_url"}, {"image_url", "x"}}});

    ChatRequest req;
    CHECK(parse_chat_request(body, req).has_value());
}

TEST(chat_rejects_a_non_string_content_part_type) {
    // Must be a 400 from the parser, not a 500 from nlohmann throwing.
    json body                      = minimal_chat();
    body["messages"][0]["content"] = json::array({{{"type", 123}}});

    ChatRequest req;
    CHECK(parse_chat_request(body, req).has_value());
}

TEST(chat_rejects_out_of_range_integers_instead_of_wrapping) {
    // get<int>() on 2^33+4 wraps to 4: the client asks for billions of tokens
    // and silently gets four.
    auto rejects = [](const char* key, int64_t value) {
        json body  = minimal_chat();
        body[key]  = value;
        ChatRequest req;
        return parse_chat_request(body, req).has_value();
    };

    CHECK(rejects("max_tokens", 8589934596LL));   // 2^33 + 4
    CHECK(rejects("max_tokens", 2147483648LL));   // 2^31
    CHECK(rejects("top_k", 4294967296LL));
    CHECK(rejects("max_completion_tokens", 8589934596LL));

    json body               = minimal_chat();
    body["stop_token_ids"]  = json::array({8589934596LL});
    ChatRequest req;
    CHECK(parse_chat_request(body, req).has_value());
}

TEST(chat_accepts_an_empty_tool_result) {
    // A tool that legitimately returns "" must be replayable without the client
    // inventing content.
    json body = minimal_chat();
    body["messages"].push_back(
        json{{"role", "tool"}, {"tool_call_id", "call_0"}, {"content", ""}});

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.messages.size(), 2u);
    CHECK_EQ(req.messages[1].content, std::string(""));
}

TEST(chat_tool_choice_object_keeps_the_named_function) {
    json body           = minimal_chat();
    body["tool_choice"] = json{{"type", "function"}, {"function", {{"name", "edit"}}}};

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.tool_choice, std::string("required"));
    CHECK_EQ(req.tool_choice_function, std::string("edit"));

    // An object without function.name silently constrained nothing before.
    json bad           = minimal_chat();
    bad["tool_choice"] = json{{"type", "function"}};
    ChatRequest ignored;
    CHECK(parse_chat_request(bad, ignored).has_value());
}

TEST(completion_carries_stream_include_usage) {
    CompletionRequest req;
    CHECK(!parse_completion_request(
               json{{"prompt", "x"},
                    {"stream", true},
                    {"stream_options", {{"include_usage", false}}}},
               req)
               .has_value());
    CHECK(req.stream);
    CHECK(!req.stream_include_usage);

    CompletionRequest def;
    CHECK(!parse_completion_request(json{{"prompt", "x"}}, def).has_value());
    CHECK(def.stream_include_usage);
}

TEST(chat_reads_sampler_fields) {
    json body            = minimal_chat();
    body["temperature"]  = 0.2;
    body["top_k"]        = 5;
    body["max_tokens"]   = 64;
    body["stop"]         = json::array({"\n\n", "END"});
    body["seed"]         = 11;

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.sampler.temperature.has_value());
    CHECK_EQ(*req.sampler.top_k, 5);
    CHECK_EQ(*req.sampler.max_tokens, 64);
    CHECK_EQ(req.sampler.stop->size(), 2u);
    CHECK_EQ(static_cast<int>(*req.sampler.seed), 11);
}

TEST(chat_accepts_a_bare_stop_string) {
    json body    = minimal_chat();
    body["stop"] = "END";

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.sampler.stop->size(), 1u);
}

TEST(chat_rejects_an_empty_stop_string) {
    // An empty stop matches at position zero and would end every generation
    // before the first token.
    json body    = minimal_chat();
    body["stop"] = json::array({""});

    ChatRequest req;
    CHECK(parse_chat_request(body, req).has_value());
}

TEST(chat_rejects_n_greater_than_one) {
    json body = minimal_chat();
    body["n"] = 2;

    ChatRequest req;
    CHECK(parse_chat_request(body, req).has_value());
}

TEST(chat_reads_max_completion_tokens_as_an_alias) {
    json body                     = minimal_chat();
    body["max_completion_tokens"] = 32;

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(*req.sampler.max_tokens, 32);
}

TEST(chat_honours_enable_thinking) {
    json body                   = minimal_chat();
    body["chat_template_kwargs"] = json{{"enable_thinking", false}};

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.has_enable_thinking);
    CHECK(!req.enable_thinking);

    const std::string prompt = render_chatml_stub(req);
    CHECK(prompt.find("<think>") != std::string::npos);
}

TEST(chat_stream_options_include_usage) {
    json body             = minimal_chat();
    body["stream"]        = true;
    body["stream_options"] = json{{"include_usage", false}};

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.stream);
    CHECK(!req.stream_include_usage);
}

TEST(chat_parses_tools_and_extracts_their_schemas) {
    json body      = minimal_chat();
    body["tools"]  = json::array({{{"type", "function"},
                                  {"function",
                                   {{"name", "edit"},
                                    {"description", "edit a file"},
                                    {"parameters",
                                     {{"type", "object"},
                                      {"properties",
                                       {{"path", {{"type", "string"}}},
                                        {"line", {{"type", "integer"}}}}}}}}}}});

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.tools.size(), 1u);

    const ToolSchemas schemas = tool_schemas(req.tools);
    CHECK_EQ(schemas.size(), 1u);
    CHECK_EQ(schemas.at("edit").at("line"), std::string("integer"));
    CHECK_EQ(schemas.at("edit").at("path"), std::string("string"));
}

TEST(chat_rejects_a_malformed_tool) {
    json body     = minimal_chat();
    body["tools"] = json::array({json{{"type", "function"}}});

    ChatRequest req;
    CHECK(parse_chat_request(body, req).has_value());
}

TEST(chat_tool_choice_none_is_preserved) {
    json body            = minimal_chat();
    body["tool_choice"]  = "none";

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.tool_choice, std::string("none"));
}

TEST(chat_replays_assistant_tool_calls) {
    json body = minimal_chat();
    body["messages"].push_back(json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls", json::array({{{"id", "call_0"},
                                     {"type", "function"},
                                     {"function", {{"name", "f"}, {"arguments", "{}"}}}}})}});
    body["messages"].push_back(
        json{{"role", "tool"}, {"tool_call_id", "call_0"}, {"content", "42"}});

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.messages[1].tool_calls.size(), 1u);
    CHECK_EQ(req.messages[1].tool_calls[0].name, std::string("f"));
}

TEST(chat_render_stub_is_deterministic) {
    ChatRequest req;
    CHECK(!parse_chat_request(minimal_chat(), req).has_value());
    CHECK_EQ(render_chatml_stub(req), render_chatml_stub(req));
    CHECK(render_chatml_stub(req).find("<|im_start|>assistant") != std::string::npos);
}

TEST(completion_parses_prompt_forms) {
    CompletionRequest req;
    CHECK(!parse_completion_request(json{{"prompt", "hello"}}, req).has_value());
    CHECK_EQ(req.prompt, std::string("hello"));

    CompletionRequest arr;
    CHECK(!parse_completion_request(json{{"prompt", json::array({"hello"})}}, arr).has_value());
    CHECK_EQ(arr.prompt, std::string("hello"));
}

TEST(completion_rejects_batches_and_token_prompts) {
    CompletionRequest req;
    CHECK(parse_completion_request(json{{"prompt", json::array({"a", "b"})}}, req).has_value());
    CHECK(parse_completion_request(json{{"prompt", json::array({1, 2, 3})}}, req).has_value());
    CHECK(parse_completion_request(json::object(), req).has_value());
}

TEST(chat_tool_call_arguments_follow_the_template_contract) {
    // The wire format is a string; Qwen3.6's template iterates a mapping.
    const nlohmann::json obj = tool_call_arguments_for_template("{\"path\": \"invoice.txt\"}", true);
    CHECK(obj.is_object());
    CHECK_EQ(obj["path"].get<std::string>(), std::string("invoice.txt"));
    // A template that wants the string keeps the string, byte for byte.
    const nlohmann::json str = tool_call_arguments_for_template("{\"path\": \"invoice.txt\"}", false);
    CHECK(str.is_string());
    CHECK_EQ(str.get<std::string>(), std::string("{\"path\": \"invoice.txt\"}"));
    // Arguments that are not a JSON object stay a string even when an object is wanted:
    // rendering something is better than refusing, and the template's own error names it.
    CHECK(tool_call_arguments_for_template("not json", true).is_string());
    CHECK(tool_call_arguments_for_template("[1,2]", true).is_string());
}
