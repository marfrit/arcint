#include "core/chat.h"
#include "core/reasoning.h"
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

TEST(chat_accepts_image_url_content_parts) {
    // 1.2.0 takes the OpenAI image_url part (docs/milestone-1.2.0.md); the
    // marker joins the text in place and the part is recorded for the backend.
    json body = minimal_chat();
    body["messages"][0]["content"] =
        json::array({{{"type", "text"}, {"text", "look at "}},
                     {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AAAA"}}}},
                     {{"type", "text"}, {"text", " what is this"}}});

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.messages[0].content, std::string("look at <__media__> what is this"));
    CHECK_EQ(req.messages[0].images.size(), size_t(1));
    CHECK_EQ(req.messages[0].images[0].url, std::string("data:image/png;base64,AAAA"));
}

TEST(chat_rejects_audio_and_malformed_content_parts) {
    // Audio input is the transcription route's, not the chat's, and an
    // image_url part without a url object is still a hard 400.
    {
        json part;
        part["type"]              = "input_audio";
        part["input_audio"]       = json::object();
        part["input_audio"]["data"] = "x";
        json body                      = minimal_chat();
        body["messages"][0]["content"] = json::array({part});
        ChatRequest           req;
        CHECK(parse_chat_request(body, req).has_value());
    }
    {
        json part;
        part["type"]      = "image_url";
        part["image_url"] = "x";
        json body                      = minimal_chat();
        body["messages"][0]["content"] = json::array({part});
        ChatRequest           req;
        CHECK(parse_chat_request(body, req).has_value());
    }
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
    body["min_p"]        = 0.05;
    body["max_tokens"]   = 64;
    body["stop"]         = json::array({"\n\n", "END"});
    body["seed"]         = 11;

    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.sampler.temperature.has_value());
    CHECK_EQ(*req.sampler.top_k, 5);
    CHECK(req.sampler.min_p && *req.sampler.min_p == 0.05f);
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

TEST(completion_rejects_batches) {
    CompletionRequest req;
    CHECK(parse_completion_request(json{{"prompt", json::array({"a", "b"})}}, req).has_value());
    CHECK(parse_completion_request(json::object(), req).has_value());
}

// FULL-DEPTH (2026-09-13): the OpenAI token-id prompt form is accepted, one
// sequence, fed as given -- the KLD gate replays the reference capture's own
// token windows through it. This cell was `completion_rejects_batches_and_
// token_prompts` and asserted the refusal; it went red when the form landed.
TEST(completion_accepts_a_token_id_prompt_and_refuses_a_bad_one) {
    CompletionRequest req;
    CHECK(!parse_completion_request(json{{"prompt", json::array({760, 6511, 314})}}, req).has_value());
    CHECK_EQ(req.prompt_ids.size(), static_cast<size_t>(3));
    CHECK_EQ(req.prompt_ids[2], 314);
    CHECK(req.prompt.empty());
    CompletionRequest bad;
    CHECK(parse_completion_request(json{{"prompt", json::array({1, -2})}}, bad).has_value());
    CHECK(parse_completion_request(json{{"prompt", json::array({1, "x"})}}, bad).has_value());
    CHECK(parse_completion_request(json{{"prompt", json::array({1, 2})}, {"echo", true}}, bad)
              .has_value());
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

TEST(chat_reasoning_effort_maps_onto_the_template_switch) {
    ChatRequest req;
    CHECK(!parse_chat_request(nlohmann::json::parse(R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"medium"})"), req));
    CHECK(req.has_enable_thinking && req.enable_thinking);
    ChatRequest off;
    CHECK(!parse_chat_request(nlohmann::json::parse(R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"none"})"), off));
    CHECK(off.has_enable_thinking && !off.enable_thinking);
    // The precise switch wins over the coarse one.
    ChatRequest both;
    CHECK(!parse_chat_request(nlohmann::json::parse(R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":true}})"), both));
    CHECK(both.has_enable_thinking && !both.enable_thinking);
    ChatRequest bad;
    CHECK(parse_chat_request(nlohmann::json::parse(R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":3})"), bad).has_value());
}

TEST(chat_tool_grammar_and_parallel_tool_calls_are_read) {
    ChatRequest plain;
    CHECK(!parse_chat_request(minimal_chat(), plain).has_value());
    CHECK(!plain.sampler.tool_grammar.has_value());
    CHECK(plain.parallel_tool_calls);   // the OpenAI default

    json body                   = minimal_chat();
    body["tool_grammar"]        = true;
    body["parallel_tool_calls"] = false;
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.sampler.tool_grammar.has_value() && *req.sampler.tool_grammar);
    CHECK(!req.parallel_tool_calls);

    json bad           = minimal_chat();
    bad["tool_grammar"] = "yes";
    ChatRequest r1;
    CHECK(parse_chat_request(bad, r1).has_value());
    json bad2                   = minimal_chat();
    bad2["parallel_tool_calls"] = 1;
    ChatRequest r2;
    CHECK(parse_chat_request(bad2, r2).has_value());
}

// ------------------------- reasoning replay (docs/campaigns/feyntalk.md W3.2)
TEST(chat_parses_reasoning_content_on_assistant_messages) {
    // OpenAI-compatible clients send a past assistant turn's reasoning as
    // reasoning_content; it must reach ChatMessage::reasoning, not be dropped.
    json body = minimal_chat();
    body["messages"] = json::array({
        {{"role", "user"}, {"content", "q"}},
        {{"role", "assistant"}, {"content", "a"}, {"reasoning_content", "why"}},
        {{"role", "user"}, {"content", "next"}},
    });
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK_EQ(req.messages[1].reasoning, std::string("why"));
    CHECK_EQ(req.messages[0].reasoning, std::string(""));

    // A non-string reasoning_content is a 400, not a silent drop.
    json bad = body;
    bad["messages"][1]["reasoning_content"] = 7;
    ChatRequest r2;
    CHECK(parse_chat_request(bad, r2).has_value());
}

TEST(stub_render_keeps_reasoning_only_after_the_last_user_message) {
    // The Qwen3-family rule, mirrored by the stub so the path is tested: an
    // assistant turn after the last user message carries its reasoning block;
    // an earlier turn does not. Tags are built from the shared table, never
    // written out here.
    ChatRequest req;
    ChatMessage first;
    first.role      = "user";
    first.content   = "first question";
    ChatMessage one;
    one.role      = "assistant";
    one.content   = "EARLY-ANSWER";
    one.reasoning = "EARLY-REASONING";
    ChatMessage second;
    second.role     = "user";
    second.content  = "second question";
    ChatMessage two;
    two.role      = "assistant";
    two.content   = "LATE-ANSWER";
    two.reasoning = "LATE-REASONING";
    req.messages = {first, one, second, two};

    const std::string prompt = render_chatml_stub(req);
    CHECK(prompt.find("EARLY-ANSWER") != std::string::npos);
    CHECK(prompt.find("LATE-ANSWER") != std::string::npos);
    CHECK_EQ(prompt.find("EARLY-REASONING"), std::string::npos);   // before the last user
    CHECK(prompt.find("LATE-REASONING") != std::string::npos);      // after the last user
    // The late block is the template's own shape: opener, reasoning, closer.
    const auto& pairs = think_tag_pairs(false);
    CHECK(!pairs.empty());
    if (pairs.empty()) return;
    const std::string block = pairs[0].first + "\n" + "LATE-REASONING" + "\n" + pairs[0].second;
    CHECK(prompt.find(block) != std::string::npos);
}
