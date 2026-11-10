#include "api/handlers.h"
#include "core/chat.h"
#include "core/reasoning.h"
#include "core/markers.h"
#include "harness.h"

#include <stdexcept>
#include <string>
#include <vector>

using namespace lgc;
using nlohmann::json;

namespace {
json minimal_chat() {
    return json{{"model", "qwen3.8-27b"},
                {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
}

// A backend whose template render throws, as minja does on a value it cannot
// use: prepare_chat must answer a 400 (the request made the template fail), not
// propagate the exception and not a 500 (that is an engine fault).
class ThrowTokenizer final : public Tokenizer {
public:
    std::vector<int> encode(std::string_view) override { return {}; }
    std::string      decode(const std::vector<int>&) override { return {}; }
    std::string      decode_one(int) override { return {}; }
    int              eos_id() const override { return 0; }
};

class ThrowingTemplateBackend final : public Backend {
public:
    // true: an engine fault (logic_error) that must not be read as the client's
    // 400; false: a minja runtime_error, the template's own failure.
    bool logic_error = false;
    ThrowingTemplateBackend() {
        st.id          = "stub";
        st.served_id   = "stub";
        st.loaded      = true;
        st.stub        = true;
        st.n_ctx       = 4096;
        st.n_ctx_train = kStubNCtxTrain;
    }
    const ModelStatus& status() const override { return st; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest&) const override {
        if (logic_error) throw std::logic_error("an engine bug inside the render");
        throw std::runtime_error("the template rejected the request");
    }
    FinishReason generate(const GenerationInput&, int, const TokenCallback&,
                          GenerationStats&) override {
        return FinishReason::Stop;
    }

private:
    ThrowTokenizer tok_;
    ModelStatus    st;
};
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
    CHECK(prompt.find(std::string(markers::kThinkOpen)) != std::string::npos);
}

TEST(chat_preserve_thinking_is_read) {
    json body = minimal_chat();
    body["chat_template_kwargs"] = json{{"preserve_thinking", true}};
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.has_preserve_thinking);
    CHECK(req.preserve_thinking);
}

TEST(chat_preserve_thinking_must_be_a_boolean) {
    json body = minimal_chat();
    body["chat_template_kwargs"] = json{{"preserve_thinking", 3}};
    ChatRequest req;
    const auto err = parse_chat_request(body, req);
    CHECK(err.has_value());
    if (err.has_value())
        CHECK(err->find("chat_template_kwargs.preserve_thinking") != std::string::npos);
}

TEST(chat_unknown_template_kwargs_are_passed_through) {
    // llama-server's dialect: chat_template_kwargs is a passthrough, so an
    // unknown key is not refused. It reaches the template unchanged (the stub
    // ignores it, like a template without that key), and the typed switches
    // are still read.
    json body = minimal_chat();
    body["chat_template_kwargs"] = json{{"enable_thinking", false}, {"some_template_key", true}};
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    CHECK(req.has_enable_thinking && !req.enable_thinking);
    CHECK(req.chat_template_kwargs.contains("some_template_key"));
    if (req.chat_template_kwargs.contains("some_template_key"))
        CHECK(req.chat_template_kwargs["some_template_key"].get<bool>());
    ChatRequest without = req;
    without.chat_template_kwargs.erase("some_template_key");
    CHECK_EQ(render_chatml_stub(req), render_chatml_stub(without));
}

TEST(template_context_is_empty_without_a_request) {
    ChatRequest req;
    std::string err;
    const json  ctx = template_context(req, err);
    CHECK(err.empty());
    CHECK(ctx.empty());   // the render can then leave extra_context unset
}

TEST(template_context_carries_the_typed_switches_and_the_passthrough) {
    ChatRequest req;
    req.has_enable_thinking      = true;
    req.enable_thinking          = false;
    req.has_preserve_thinking    = true;
    req.preserve_thinking        = true;
    req.chat_template_kwargs     = json{{"some_template_key", 42}, {"another", "x"}};
    std::string err;
    const json  ctx = template_context(req, err);
    CHECK(err.empty());
    CHECK_EQ(ctx.value("enable_thinking", json()), json(false));
    CHECK_EQ(ctx.value("preserve_thinking", json()), json(true));
    CHECK_EQ(ctx.value("some_template_key", json()), json(42));   // unchanged
    CHECK_EQ(ctx.value("another", json()), json("x"));
    CHECK(!ctx.contains("messages"));
}

// OpenAI's top-level reasoning_effort: the level reaches the template as the
// variable `reasoning_effort`, at a value that template accepts. Two minimal
// templates stand in for the two dialects: Qwen3.8's (names 'xhigh', raises on
// anything outside xhigh/medium/low) and OpenAI's low/medium/high (gpt-oss).
namespace {
constexpr const char* kXhighTemplate =
    "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
    "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
    "{{- raise_exception('bad effort') }}{%- endif %}";
constexpr const char* kPlainTemplate = "Reasoning: {{ reasoning_effort|default('medium') }}";

json effort_context(const json& body, const char* tmpl) {
    ChatRequest req;
    const auto  perr = parse_chat_request(body, req);
    CHECK(!perr.has_value());
    std::string err;
    const json  ctx = template_context(req, err, tmpl);
    CHECK(err.empty());
    return ctx;
}
}   // namespace

TEST(reasoning_effort_medium_reaches_an_xhigh_template) {
    json body                = minimal_chat();
    body["reasoning_effort"] = "medium";
    const json ctx = effort_context(body, kXhighTemplate);
    CHECK_EQ(ctx.value("reasoning_effort", json()), json("medium"));
    CHECK_EQ(ctx.value("enable_thinking", json()), json(true));
}

TEST(reasoning_effort_high_maps_to_the_templates_top_level) {
    json body                = minimal_chat();
    body["reasoning_effort"] = "high";
    // A template that takes no "high" (it quotes only xhigh/medium/low): its top level is xhigh.
    CHECK_EQ(effort_context(body, kXhighTemplate).value("reasoning_effort", json()), json("xhigh"));
    // A low/medium/high template must not receive "xhigh".
    CHECK_EQ(effort_context(body, kPlainTemplate).value("reasoning_effort", json()), json("high"));
    body["reasoning_effort"] = "xhigh";
    CHECK_EQ(effort_context(body, kXhighTemplate).value("reasoning_effort", json()), json("xhigh"));
    CHECK_EQ(effort_context(body, kPlainTemplate).value("reasoning_effort", json()), json("high"));
    body["reasoning_effort"] = "minimal";
    // minimal is OpenAI's lowest level, not a switch (operator, 2026-11-04): thinking
    // stays on, at the template's low level, on this route as on /v1/responses.
    const json ctx = effort_context(body, kXhighTemplate);
    CHECK_EQ(ctx.value("reasoning_effort", json()), json("low"));
    CHECK_EQ(ctx.value("enable_thinking", json()), json(true));
    CHECK_EQ(effort_context(body, kPlainTemplate).value("reasoning_effort", json()), json("low"));
}

// The client's level passes through to every template that takes it, and the
// template translates (operator, 2026-11-04: the GGUF's template is the
// contract). Taking a level means quoting it as a token: the real Qwen3.8
// template quotes 'high' in its own high -> xhigh step, so it gets "high"; a
// template that names xhigh only inside a sentence does not take xhigh.
TEST(reasoning_effort_passes_through_to_a_template_that_takes_the_level) {
    constexpr const char* kQwen38Like =
        "{%- set r = reasoning_effort|default('xhigh') %}"
        "{%- if r == 'high' %}{%- set r = 'xhigh' %}{%- endif %}"
        "{%- if r not in ('xhigh', 'medium', 'low') %}"
        "{{- raise_exception('Unexpected reasoning effort. Supported types are xhigh (default), medium, and low.') }}"
        "{%- endif %}";
    constexpr const char* kProseXhigh =
        "{%- if reasoning_effort|default('medium') not in ('low', 'medium', 'high') %}"
        "{{- raise_exception('low, medium or high; xhigh is not a level here') }}{%- endif %}";
    constexpr const char* kBoth =
        "{%- if reasoning_effort|default('medium') not in ('low', 'medium', 'high', 'xhigh') %}"
        "{{- raise_exception('bad effort') }}{%- endif %}";
    json body                = minimal_chat();
    body["reasoning_effort"] = "high";
    CHECK_EQ(effort_context(body, kQwen38Like).value("reasoning_effort", json()), json("high"));
    CHECK_EQ(effort_context(body, kProseXhigh).value("reasoning_effort", json()), json("high"));
    CHECK_EQ(effort_context(body, kBoth).value("reasoning_effort", json()), json("high"));
    body["reasoning_effort"] = "xhigh";
    CHECK_EQ(effort_context(body, kBoth).value("reasoning_effort", json()), json("xhigh"));
    CHECK_EQ(effort_context(body, kProseXhigh).value("reasoning_effort", json()), json("high"));
}

// A template that does not read the variable gets no level: the key would be
// dead weight and its debug line would claim a level reached the template.
TEST(reasoning_effort_skips_a_template_without_the_variable) {
    json body                = minimal_chat();
    body["reasoning_effort"] = "high";
    CHECK(!effort_context(body, "{{ messages[0].content }}").contains("reasoning_effort"));
}

// The parse-time call (the reserved-key check, no template source) maps no
// level: only the render, which has the template, decides what reaches it.
TEST(reasoning_effort_parse_time_context_maps_no_level) {
    json body                    = minimal_chat();
    body["reasoning_effort"]     = "high";
    body["chat_template_kwargs"] = json{{"some_key", 1}};
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    std::string err;
    const json  ctx = template_context(req, err);
    CHECK(err.empty());
    CHECK(!ctx.contains("reasoning_effort"));
    CHECK_EQ(ctx.value("some_key", json()), json(1));
}

// The levels a template takes, as /props publishes them: the quoted ones of
// low/medium/high/xhigh, only for a template that reads reasoning_effort.
TEST(reasoning_effort_levels_of_a_template) {
    using V = std::vector<std::string>;
    CHECK(template_reasoning_effort_levels("{{ messages }}") == V{});
    CHECK(template_reasoning_effort_levels(
              "{%- set r = reasoning_effort|default('xhigh') %}{%- if r == 'high' %}{%- endif %}"
              "{%- if r not in ('xhigh', 'medium', 'low') %}{%- endif %}") ==
          (V{"low", "medium", "high", "xhigh"}));
    CHECK(template_reasoning_effort_levels("Reasoning: {{ reasoning_effort|default('medium') }}") ==
          V{"medium"});
}

TEST(reasoning_effort_none_passes_no_level) {
    json body                = minimal_chat();
    body["reasoning_effort"] = "none";
    const json ctx = effort_context(body, kXhighTemplate);
    CHECK(!ctx.contains("reasoning_effort"));
    CHECK_EQ(ctx.value("enable_thinking", json()), json(false));
    body["reasoning_effort"] = "off";
    CHECK(!effort_context(body, kPlainTemplate).contains("reasoning_effort"));
}

TEST(reasoning_effort_template_kwarg_wins_over_the_top_level_field) {
    json body                    = minimal_chat();
    body["reasoning_effort"]     = "medium";
    body["chat_template_kwargs"] = json{{"reasoning_effort", "low"}};
    const json ctx = effort_context(body, kXhighTemplate);
    CHECK_EQ(ctx.value("reasoning_effort", json()), json("low"));
    CHECK_EQ(ctx.value("enable_thinking", json()), json(true));
}

TEST(reasoning_effort_unknown_level_passes_no_level_and_no_error) {
    json body                = minimal_chat();
    body["reasoning_effort"] = "turbo";
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());   // accepted, as before
    std::string err;
    const json  ctx = template_context(req, err, kXhighTemplate);
    CHECK(err.empty());
    CHECK(!ctx.contains("reasoning_effort"));   // the template's default applies
    CHECK_EQ(ctx.value("enable_thinking", json()), json(true));
}

TEST(template_context_refuses_reserved_keys) {
    // The context keys minja sets itself: a client may not override them.
    static constexpr const char* kReserved[] = {"messages", "add_generation_prompt", "bos_token",
                                                "eos_token", "strftime_now", "tools"};
    for (const char* key : kReserved) {
        ChatRequest req;
        req.chat_template_kwargs = json{{key, "x"}};
        std::string err;
        const json  ctx = template_context(req, err);
        CHECK(!err.empty());
        if (!err.empty()) CHECK(err.find(key) != std::string::npos);
        CHECK(!ctx.contains(key));   // the engine's own value wins
    }
}

TEST(chat_refuses_reserved_template_context_kwargs) {
    static constexpr const char* kReserved[] = {"messages", "add_generation_prompt", "bos_token",
                                                "eos_token", "strftime_now", "tools"};
    for (const char* key : kReserved) {
        json body = minimal_chat();
        body["chat_template_kwargs"] = json{{key, "x"}};
        ChatRequest req;
        const auto  err = parse_chat_request(body, req);
        CHECK(err.has_value());
        if (err.has_value()) {
            CHECK(err->find("chat_template_kwargs.") != std::string::npos);
            CHECK(err->find(key) != std::string::npos);
        }
    }
}

// minja's Context::builtins() (third_party/minja/minja.hpp): the template's
// own functions. A chat_template_kwargs key that names one shadows it on the
// child context and makes a template that calls it throw, so the names are
// reserved too. The table is read from that function.
static constexpr const char* kMinjaBuiltins[] = {
    "==",       "count",  "default",   "dictsort", "e",        "equalto",
    "escape",   "in",     "indent",    "int",      "items",    "join",
    "joiner",   "last",   "length",    "list",     "lower",    "map",
    "namespace", "raise_exception", "range", "reject", "rejectattr", "safe",
    "select",   "selectattr", "string", "tojson",  "trim",     "unique",
    "upper",
};

TEST(template_context_refuses_builtin_keys) {
    for (const char* key : kMinjaBuiltins) {
        ChatRequest req;
        req.chat_template_kwargs = json{{key, 1}};
        std::string err;
        const json  ctx = template_context(req, err);
        CHECK(!err.empty());
        if (!err.empty()) CHECK(err.find(key) != std::string::npos);
        CHECK(!ctx.contains(key));
    }
}

TEST(chat_refuses_builtin_template_context_kwargs) {
    for (const char* key : kMinjaBuiltins) {
        json body = minimal_chat();
        body["chat_template_kwargs"] = json{{key, 1}};
        ChatRequest req;
        const auto  err = parse_chat_request(body, req);
        CHECK(err.has_value());
        if (err.has_value()) CHECK(err->find(key) != std::string::npos);
    }
}

TEST(chat_template_render_error_is_a_400) {
    Config                 cfg;
    ThrowingTemplateBackend backend;
    api::SlotPool          slots(1);
    api::Context           ctx{&cfg, &backend, &slots};

    json body = json{{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
    // The current tree lets the render exception escape prepare_chat; the case
    // pins the 400. A propagated throw leaves `err` unset, which is the red.
    std::optional<api::HttpResult> err;
    try {
        api::PreparedChat prep;
        err = api::prepare_chat(ctx, body, prep);
    } catch (const std::exception&) {
    }
    CHECK(err.has_value());
    if (err.has_value()) CHECK_EQ(err->status, 400);
}

TEST(chat_template_render_engine_fault_is_not_a_400) {
    // Only a minja runtime_error is the template's own failure. A logic_error
    // (or bad_alloc) is an engine fault: it must not be blamed on the request,
    // it propagates to the route's 500 handler.
    Config                 cfg;
    ThrowingTemplateBackend backend;
    backend.logic_error = true;
    api::SlotPool slots(1);
    api::Context  ctx{&cfg, &backend, &slots};

    json body = json{{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
    bool propagated = false;
    try {
        api::PreparedChat prep;
        const auto        err = api::prepare_chat(ctx, body, prep);
        CHECK(!err.has_value() || err->status != 400);
    } catch (const std::logic_error&) {
        propagated = true;
    }
    CHECK(propagated);
}

TEST(chat_render_stub_preserve_thinking_keeps_earlier_reasoning) {
    ChatRequest req;
    CHECK(!parse_chat_request(minimal_chat(), req).has_value());
    // An assistant turn with reasoning BEFORE a later user message: the rule
    // hides it unless preserve_thinking is set.
    ChatMessage a;
    a.role      = "assistant";
    a.content   = "first answer";
    a.reasoning = "REASON-OLD";
    ChatMessage u;
    u.role    = "user";
    u.content = "steer";
    req.messages.push_back(a);
    req.messages.push_back(u);

    const std::string off = render_chatml_stub(req);
    CHECK(off.find("REASON-OLD") == std::string::npos);
    req.preserve_thinking     = true;
    req.has_preserve_thinking = true;
    const std::string on = render_chatml_stub(req);
    CHECK(on.find("REASON-OLD") != std::string::npos);
    CHECK(off != on);
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
    CHECK(render_chatml_stub(req).find(std::string(markers::kTurnStart) + "assistant") != std::string::npos);
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

// ---------------------------------------------------------------------------
// --chat-template-kwarg defaults applied where the request is prepared
// (apply_template_kwarg_defaults). The request's own value always wins.
namespace {
ChatRequest parsed_chat(const nlohmann::json& extra) {
    nlohmann::json body{{"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})}};
    for (auto it = extra.begin(); it != extra.end(); ++it) body[it.key()] = it.value();
    ChatRequest req;
    const auto  err = parse_chat_request(body, req);
    if (err) t::fail(__FILE__, __LINE__, "parse_chat_request: " + *err);
    return req;
}
}  // namespace

TEST(chat_template_kwarg_default_fills_an_absent_key) {
    ChatRequest req = parsed_chat(nlohmann::json::object());
    apply_template_kwarg_defaults(req, {{"force_nonempty_content", "true"}, {"depth", "3"}, {"tone", "\"warm\""}});
    CHECK(req.chat_template_kwargs.value("force_nonempty_content", nlohmann::json()) == nlohmann::json(true));
    CHECK(req.chat_template_kwargs.value("depth", nlohmann::json()) == nlohmann::json(3));
    CHECK(req.chat_template_kwargs.value("tone", nlohmann::json()) == nlohmann::json("warm"));
    // ... and it reaches the template context like a request's own key
    std::string err;
    const nlohmann::json extra = template_context(req, err);
    CHECK(err.empty());
    CHECK(extra.value("depth", nlohmann::json()) == nlohmann::json(3));
}

TEST(chat_template_kwarg_default_loses_to_the_request) {
    // an explicit false beats a default true; a different value beats it too
    ChatRequest req = parsed_chat({{"chat_template_kwargs", {{"force_nonempty_content", false}, {"depth", 9}}}});
    apply_template_kwarg_defaults(req, {{"force_nonempty_content", "true"}, {"depth", "3"}});
    CHECK(req.chat_template_kwargs["force_nonempty_content"] == nlohmann::json(false));
    CHECK(req.chat_template_kwargs["depth"] == nlohmann::json(9));
}

TEST(chat_template_kwarg_default_treats_a_request_null_as_absent) {
    ChatRequest req = parsed_chat({{"chat_template_kwargs", {{"force_nonempty_content", nullptr}}}});
    apply_template_kwarg_defaults(req, {{"force_nonempty_content", "true"}});
    CHECK(req.chat_template_kwargs["force_nonempty_content"] == nlohmann::json(true));
}

TEST(chat_template_kwarg_default_preserve_thinking_is_the_typed_field) {
    ChatRequest a = parsed_chat(nlohmann::json::object());
    apply_template_kwarg_defaults(a, {{"preserve_thinking", "true"}});
    CHECK(a.has_preserve_thinking);
    CHECK(a.preserve_thinking);
    // the request's own false wins
    ChatRequest b = parsed_chat({{"chat_template_kwargs", {{"preserve_thinking", false}}}});
    apply_template_kwarg_defaults(b, {{"preserve_thinking", "true"}});
    CHECK(b.has_preserve_thinking);
    CHECK(!b.preserve_thinking);
}

TEST(chat_template_kwarg_default_none_configured_changes_nothing) {
    ChatRequest req = parsed_chat({{"chat_template_kwargs", {{"x", 1}}}});
    const nlohmann::json before = req.chat_template_kwargs;
    apply_template_kwarg_defaults(req, {});
    CHECK(req.chat_template_kwargs == before);
    CHECK(!req.has_enable_thinking);
}

TEST(chat_template_kwarg_parse_default_forms) {
    std::string    key;
    nlohmann::json v;
    CHECK(!parse_template_kwarg_default("a=true", key, v));
    CHECK_EQ(key, std::string("a"));
    CHECK(v == nlohmann::json(true));
    CHECK(!parse_template_kwarg_default("a=1.5", key, v));
    CHECK(v == nlohmann::json(1.5));
    CHECK(!parse_template_kwarg_default("a=\"x\"", key, v));
    CHECK(v == nlohmann::json("x"));
    CHECK(!parse_template_kwarg_default("a=bare words", key, v));
    CHECK(v == nlohmann::json("bare words"));
    CHECK(!parse_template_kwarg_default("a={\"k\":[1,2]}", key, v));
    CHECK(v.is_object());
    // an unterminated JSON-looking value is bare text, not half a parse
    CHECK(!parse_template_kwarg_default("a=[1,", key, v));
    CHECK(v == nlohmann::json("[1,"));
    // the first '=' splits; the value may hold more
    CHECK(!parse_template_kwarg_default("a=b=c", key, v));
    CHECK(v == nlohmann::json("b=c"));
    CHECK(parse_template_kwarg_default("a=null", key, v).has_value());
    CHECK(parse_template_kwarg_default("tools=1", key, v).has_value());
    CHECK(parse_template_kwarg_default("preserve_thinking=0", key, v).has_value());
}

// Review F2: a unit default reasoning_effort is a level a request would have
// sent. A request that states its own level (top-level reasoning_effort, the
// Responses reasoning.effort, or the kwarg) keeps it, and the default goes through
// the mapping a request's level gets.
namespace {
// A template that reads the variable and quotes low and xhigh (no 'high').
const char* kEffortTemplate =
    "{% if reasoning_effort == 'low' %}L{% elif reasoning_effort == 'xhigh' %}X{% endif %}";

nlohmann::json effort_context(ChatRequest& req) {
    std::string err;
    const nlohmann::json extra = template_context(req, err, kEffortTemplate);
    CHECK(err.empty());
    return extra;
}
}  // namespace

TEST(chat_template_kwarg_default_reasoning_effort_loses_to_the_requests_top_level_level) {
    ChatRequest req = parsed_chat({{"reasoning_effort", "low"}});
    apply_template_kwarg_defaults(req, {{"reasoning_effort", "\"high\""}});
    const nlohmann::json extra = effort_context(req);
    CHECK(extra.value("reasoning_effort", nlohmann::json()) == nlohmann::json("low"));
}

TEST(chat_template_kwarg_default_reasoning_effort_loses_to_the_requests_kwarg) {
    ChatRequest req = parsed_chat({{"chat_template_kwargs", {{"reasoning_effort", "low"}}}});
    apply_template_kwarg_defaults(req, {{"reasoning_effort", "\"high\""}});
    const nlohmann::json extra = effort_context(req);
    CHECK(extra.value("reasoning_effort", nlohmann::json()) == nlohmann::json("low"));
}

TEST(chat_template_kwarg_default_reasoning_effort_goes_through_the_mapping) {
    // no request level: the default high reaches a template that names xhigh as xhigh
    ChatRequest req = parsed_chat(nlohmann::json::object());
    apply_template_kwarg_defaults(req, {{"reasoning_effort", "\"high\""}});
    const nlohmann::json extra = effort_context(req);
    CHECK(extra.value("reasoning_effort", nlohmann::json()) == nlohmann::json("xhigh"));
    CHECK(req.has_enable_thinking && req.enable_thinking);   // a level other than none is thinking on
}

TEST(chat_template_kwarg_default_reasoning_effort_none_is_thinking_off_without_a_level) {
    ChatRequest req = parsed_chat(nlohmann::json::object());
    apply_template_kwarg_defaults(req, {{"reasoning_effort", "\"none\""}});
    CHECK(req.has_enable_thinking);
    CHECK(!req.enable_thinking);
    const nlohmann::json extra = effort_context(req);
    CHECK(!extra.contains("reasoning_effort"));   // none passes no level, as a request's none
    CHECK(extra.value("enable_thinking", nlohmann::json()) == nlohmann::json(false));
    // a request that states thinking keeps it
    ChatRequest on = parsed_chat({{"chat_template_kwargs", {{"enable_thinking", true}}}});
    apply_template_kwarg_defaults(on, {{"reasoning_effort", "\"none\""}});
    CHECK(on.enable_thinking);
}

TEST(chat_template_kwarg_parse_default_reasoning_effort_values) {
    std::string    key;
    nlohmann::json v;
    for (const char* l : {"none", "off", "minimal", "low", "medium", "high", "xhigh"})
        CHECK(!parse_template_kwarg_default(std::string("reasoning_effort=") + l, key, v));
    CHECK(parse_template_kwarg_default("reasoning_effort=extreme", key, v).has_value());
    CHECK(parse_template_kwarg_default("reasoning_effort=3", key, v).has_value());
}
