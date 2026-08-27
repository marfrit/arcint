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
