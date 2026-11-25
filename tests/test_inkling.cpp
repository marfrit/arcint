#include "core/chat.h"
#include "core/dialect.h"
#include "core/markers.h"
#include "core/reasoning.h"
#include "core/toolcall.h"
#include "exec/chat_json.h"
#include "harness.h"

#include <string>
#include <utility>
#include <vector>

// minja declares `using json = nlohmann::ordered_json` at global scope, which
// clashes with `using nlohmann::json`; this file keeps to ordered_json, as
// test_chat_render.cpp does.
#include <minja/chat-template.hpp>

using namespace lgc;

namespace {

// The artifact's own template, as shipped for Inkling 2 Mini (the GGUF carries
// it too). Embedded so the render is tested without a model load; the golden
// below was produced from this same text by mainline llama.cpp's
// common_chat_templates_apply (2026-11-22, CPU only), then by arcint's minja.
constexpr const char* kInklingTemplate = R"JINJA({%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}
{%- set role_token = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>"} -%}

{%- macro emit_thinking_effort() -%}
    {%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
    {%- if eff is string -%}
        {%- set key = eff | trim -%}
        {%- if key not in effort_map -%}
            {{- raise_exception("Unknown reasoning_effort: " ~ eff) -}}
        {%- endif -%}
        {%- set num = effort_map[key] -%}
    {%- else -%}
        {%- set num = eff | float -%}
    {%- endif -%}
    {%- if num < 0.0 or num > 0.99 -%}
        {{- raise_exception("reasoning_effort must be in [0.0, 0.99]") -}}
    {%- endif -%}
    {{- "<|message_system|><|content_text|>Thinking effort level: " -}}
    {%- if num == 0.0 -%}0{%- else -%}{{ num }}{%- endif -%}
    {{- "<|end_message|>" -}}
{%- endmacro -%}

{%- if tools -%}
    {%- set tool_state = namespace(specs=[]) -%}
    {%- for tool in tools -%}
        {%- set fn = tool.function if tool.function is defined else tool -%}
        {%- set spec = {
            "description": (fn.description if fn.description is defined and fn.description else ""),
            "name": fn.name,
            "parameters": (fn.parameters if fn.parameters is defined and fn.parameters else {}),
            "type": (tool.type if tool.type is defined and tool.type else "function"),
        } -%}
        {%- set tool_state.specs = tool_state.specs + [spec] -%}
    {%- endfor -%}
    {{- "<|message_system|>tool_declare<|content_xml|>" -}}
    {{- tool_state.specs | tojson(sort_keys=true, separators=(",", ":")) -}}
    {{- "<|end_message|>" -}}
{%- endif -%}

{%- set state = namespace(effort_emitted=false) -%}
{%- for message in messages -%}
    {%- if message.role not in role_token -%}
        {{- raise_exception("Unknown message role: " ~ message.role) -}}
    {%- endif -%}
    {%- if not state.effort_emitted and message.role != "system" -%}
        {{- emit_thinking_effort() -}}
        {%- set state.effort_emitted = true -%}
    {%- endif -%}

    {%- set header = namespace(text=role_token[message.role]) -%}
    {%- if message.role == "tool" -%}
        {%- if message.name is defined and message.name -%}
            {%- set header.text = header.text ~ message.name -%}
        {%- elif message.tool_call_id is defined and message.tool_call_id -%}
            {%- for prev in messages -%}
                {%- if prev.role == "assistant" and prev.tool_calls -%}
                    {%- for tc in prev.tool_calls -%}
                        {%- if tc.id is defined and tc.id == message.tool_call_id and tc.function.name is defined -%}
                            {%- set header.text = header.text ~ tc.function.name -%}
                        {%- endif -%}
                    {%- endfor -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif message.role == "assistant" and message.name is defined and message.name -%}
        {%- set header.text = header.text ~ message.name -%}
    {%- endif -%}
    {%- set rtok = header.text -%}
    {%- set rendered -%}

    {%- if message.role == "assistant" and message.reasoning_content is defined and message.reasoning_content -%}
        {{- rtok ~ "<|content_thinking|>" ~ message.reasoning_content ~ "<|end_message|>" -}}
    {%- endif -%}

    {%- if message.content is string and (message.content or not (message.role == "assistant" and (message.tool_calls or message.reasoning_content))) -%}
        {{- rtok ~ "<|content_text|>" ~ message.content ~ "<|end_message|>" -}}
    {%- elif message.content -%}
        {%- for part in message.content -%}
            {%- if part is string -%}
                {{- rtok ~ "<|content_text|>" ~ part ~ "<|end_message|>" -}}
            {%- elif part.type is not defined or part.type in ("text", "input_text") -%}
                {%- set text_part = (part.text if part.text is defined and part.text is string else "") -%}
                {{- rtok ~ "<|content_text|>" ~ text_part ~ "<|end_message|>" -}}
            {%- elif part.type in ("image", "input_image", "image_url") -%}
                {%- if message.role not in ("user", "tool") -%}
                    {{- raise_exception("Image content is only supported for user and tool messages") -}}
                {%- endif -%}
                {{- rtok ~ "<|content_image|><|unused_200054|><|end_message|>" -}}
            {%- elif part.type in ("audio", "input_audio", "audio_url") -%}
                {{- raise_exception("Inkling 2 does not accept audio input") -}}
            {%- else -%}
                {{- raise_exception("Unsupported content part type: " ~ part.type) -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}

    {%- if message.role == "assistant" and message.tool_calls -%}
        {%- for tc in message.tool_calls -%}
            {%- set fn = tc.function -%}
            {%- if fn.name is not defined or fn.name is not string -%}
                {{- raise_exception("tool call function name must be a string") -}}
            {%- endif -%}
            {%- if message.name is defined and message.name and message.name != fn.name -%}
                {{- raise_exception("assistant name " ~ message.name ~ " does not match its tool call " ~ fn.name) -}}
            {%- endif -%}
            {%- set args = fn.arguments if fn.arguments is defined and fn.arguments else {} -%}
            {%- if args is string -%}
                {{- raise_exception("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream") -}}
            {%- endif -%}
            {%- if args is not mapping -%}
                {{- raise_exception("tool call arguments must be an object") -}}
            {%- endif -%}
            {{- "<|message_model|>" ~ fn.name ~ "<|content_invoke_tool_json|>" -}}
            {{- '{"name":' ~ (fn.name | tojson(sort_keys=true, separators=(",", ":"))) ~ ',"args":' -}}
            {{- "{" -}}
            {%- for key, value in args | dictsort(case_sensitive=true) -%}
                {{- (key | tojson) ~ ":" ~ (value | tojson(separators=(",", ":"))) -}}
                {%- if not loop.last -%},{%- endif -%}
            {%- endfor -%}
            {{- "}}<|end_message|>" -}}
        {%- endfor -%}
    {%- endif -%}

    {%- if message.role == "assistant" -%}
        {{- "<|content_model_end_sampling|>" -}}
    {%- endif -%}
    {%- endset -%}
    {%- if message.role == "assistant" -%}
        {%- generation -%}{{- rendered -}}{%- endgeneration -%}
    {%- else -%}
        {{- rendered -}}
    {%- endif -%}
{%- endfor -%}

{%- if not state.effort_emitted -%}
    {{- emit_thinking_effort() -}}
{%- endif -%}
)JINJA";

nlohmann::ordered_json message_of(const char* role, const char* content) {
    return nlohmann::ordered_json{{"role", role}, {"content", content}};
}

// The same three steps the llama backend takes: a request's messages and tools
// as the template's JSON (chat_json.h), the template's context (template_context),
// and minja with polyfills off.
std::string render(const ChatRequest& req) {
    std::string      err;
    nlohmann::json   ctx = template_context(req, err, kInklingTemplate);
    CHECK(err.empty());
    minja::chat_template tmpl(kInklingTemplate, "", "");
    minja::chat_template_inputs in;
    in.messages              = chat_messages_json(req, ChatDialect(ChatDialect::Inkling).replays_arguments_as_object());
    in.add_generation_prompt = true;
    const nlohmann::json tools = chat_tools_json(req);
    if (!tools.is_null()) in.tools = nlohmann::ordered_json(tools);
    if (!ctx.empty()) in.extra_context = nlohmann::ordered_json(ctx);
    minja::chat_template_options opts;
    opts.apply_polyfills = false;
    return tmpl.apply(in, opts);
}

ChatRequest conversation(const char* effort, bool numeric = false) {
    ChatRequest req;
    req.model = "inkling2-mini";
    req.messages.push_back({"system", "You are concise.", {}, {}, {}, {}, {}});
    req.messages.push_back({"user", "Weather in Berlin?", {}, {}, {}, {}, {}});
    ChatMessage assistant;
    assistant.role             = "assistant";
    assistant.reasoning        = "Need the current weather.";
    assistant.tool_calls       = {{"call_1", "get_weather", R"({"city":"Berlin","units":{"temp":"c"}})"}};
    req.messages.push_back(assistant);
    req.messages.push_back({"tool", "12 C", {}, "", "call_1", {}, {}});
    req.messages.push_back({"user", "Thanks", {}, {}, {}, {}, {}});
    req.tools.push_back({"get_weather", "Current weather for a city",
                         nlohmann::json::parse(R"({"type":"object","properties":{"city":{"type":"string"},)"
                                               R"("units":{"type":"object","properties":{"temp":{"type":"string"}}}},)"
                                               R"("required":["city"]})")});
    if (numeric) {
        nlohmann::json body = {{"model", "inkling2-mini"},
                               {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                               {"reasoning_effort", 0.7}};
        CHECK(!parse_chat_request(body, req).has_value());
        req.messages = conversation(nullptr).messages;
        req.tools    = conversation(nullptr).tools;
    } else if (effort != nullptr) {
        nlohmann::json body = {{"model", "inkling2-mini"},
                               {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                               {"reasoning_effort", effort}};
        CHECK(!parse_chat_request(body, req).has_value());
        req.messages = conversation(nullptr).messages;
        req.tools    = conversation(nullptr).tools;
    }
    return req;
}

std::string script(std::string_view role, std::string_view kind, std::string_view body) {
    return std::string(markers::kInkMessageModel) + std::string(kind) + std::string(body) +
           std::string(markers::kInkEndMessage);
}

}  // namespace

// render_system_user_assistant_call_tool_user
// Mutation: remove the minja `separators` handling (tojson writes ", " and ": ")
// and the golden differs; remove `dictsort(case_sensitive=true)` and render throws.
TEST(inkling_render_system_user_assistant_call_tool_user) {
    const std::string got = render(conversation("medium"));
    const std::string expected =
        "<|message_system|>tool_declare<|content_xml|>"
        R"([{"description":"Current weather for a city","name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"units":{"properties":{"temp":{"type":"string"}},"type":"object"}},"required":["city"],"type":"object"},"type":"function"}])"
        "<|end_message|>"
        "<|message_system|><|content_text|>You are concise.<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>"
        "<|message_user|><|content_text|>Weather in Berlin?<|end_message|>"
        "<|message_model|><|content_thinking|>Need the current weather.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        R"({"name":"get_weather","args":{"city":"Berlin","units":{"temp":"c"}}})"
        "<|end_message|><|content_model_end_sampling|>"
        "<|message_tool|>get_weather<|content_text|>12 C<|end_message|>"
        "<|message_user|><|content_text|>Thanks<|end_message|>";
    CHECK_EQ(got, expected);
}

// is_inkling_template / dialect_from_template
// Mutation: drop the kInkContentImage test and the Inkling template stops being
// detected (the fallback Qwen dialect is returned).
TEST(inkling_template_detection_decides_the_dialect) {
    CHECK(is_inkling_template(kInklingTemplate));
    CHECK(!is_inkling_template("{%- if enable_thinking is defined -%}x{%- endif -%}"));
    CHECK(dialect_from_template(kInklingTemplate, ChatDialect::Gemma4).inkling());
    CHECK(dialect_from_template("plain", ChatDialect::Gemma4).gemma4());
}

// split_reasoning on the model's framing
// Mutation: remove the Inkling branch and the content keeps the framing markers.
TEST(inkling_reasoning_split_strips_the_framing) {
    const std::string raw = script("<", markers::kInkContentThink, "plan") +
                            script("<", markers::kInkContentText, "answer");
    const ReasoningSplit s = split_reasoning(raw, ThinkStart::Undecided, ChatDialect::Inkling);
    CHECK(s.closed);
    CHECK_EQ(s.reasoning, std::string("plan"));
    CHECK_EQ(s.content, std::string("answer"));
}

// The token-id path: the closer is one special token (Inkling 2 Mini's
// <|end_message|>, tokenizer_config.json) and the header is another
// (<|message_model|>); text from ordinary tokens cannot close the block.
// Mutation: drop the `token_id == close_id_` test in Thinking and the closer
// stays in reasoning.
TEST(inkling_reasoning_split_by_token_ids) {
    const std::vector<std::pair<std::string, int>> pieces = {
        {std::string(markers::kInkMessageModel), 200001},
        {std::string(markers::kInkContentThink), 200008},
        {"plan.", 10},
        {std::string(markers::kInkEndMessage), 200010},
        {std::string(markers::kInkMessageModel), 200001},
        {std::string(markers::kInkContentText), 200004},
        {"answer.", 11},
        {std::string(markers::kInkEndMessage), 200010},
        {std::string(markers::kInkModelEnd), 200006},
    };
    const ReasoningSplit s = split_reasoning_tokens(pieces, ThinkStart::Undecided, ChatDialect::Inkling,
                                                    {{-1, 200010}});
    CHECK(s.closed);
    CHECK_EQ(s.reasoning, std::string("plan."));
    CHECK_EQ(s.content, std::string("answer."));
}

// The model-end marker (the EOG token <|content_model_end_sampling|>) is not
// answer text.
// Mutation: remove the kInkModelEnd cut in the Answer stage and it appears in
// the content.
TEST(inkling_model_end_marker_is_stripped) {
    const std::string raw = script("<", markers::kInkContentText, "answer") +
                            std::string(markers::kInkModelEnd);
    const ReasoningSplit s = split_reasoning(raw, ThinkStart::Undecided, ChatDialect::Inkling);
    CHECK_EQ(s.content, std::string("answer"));
}

// tool-call parsing: one call with nested JSON arguments, and two calls in one
// turn
// Mutation: skip the balanced-brace scan and the nested object's inner `}` ends
// the block early (truncated, no call).
TEST(inkling_tool_calls_nested_arguments_and_two_calls) {
    const std::string t = std::string(markers::kInkMessageModel) + "search" +
                          std::string(markers::kInkContentToolJson) +
                          R"({"name":"search","args":{"query":"a}b{c","nested":{"x":[1,{"y":2}]}}})" +
                          std::string(markers::kInkEndMessage) +
                          std::string(markers::kInkMessageModel) + "lookup" +
                          std::string(markers::kInkContentToolJson) +
                          R"({"name":"lookup","args":{"id":7}})" +
                          std::string(markers::kInkEndMessage);
    const ToolCallParse p = parse_tool_calls(t, ChatDialect::Inkling);
    CHECK(!p.truncated);
    CHECK_EQ(p.calls.size(), 2u);
    CHECK_EQ(p.calls[0].name, std::string("search"));
    CHECK_EQ(p.calls[0].arguments, nlohmann::json::parse(R"({"query":"a}b{c","nested":{"x":[1,{"y":2}]}})").dump());
    CHECK_EQ(p.calls[1].name, std::string("lookup"));
    CHECK_EQ(p.calls[1].arguments, std::string(R"({"id":7})"));
    CHECK_EQ(p.calls[0].id, std::string("call_0"));
    CHECK_EQ(p.calls[1].id, std::string("call_1"));
    CHECK(p.content.empty());
}

// A block without its closer, a name mismatch and bad JSON are kept in the
// content, not turned into a call.
// Mutation: treat a missing <|end_message|> as a call and `truncated` stays
// false with one call.
TEST(inkling_tool_call_malformed_is_kept_and_truncated) {
    const std::string open = std::string(markers::kInkMessageModel) + "search" +
                             std::string(markers::kInkContentToolJson) +
                             R"({"name":"search","args":{"q":1}})";
    for (const std::string& t : {open,
                                 std::string(markers::kInkMessageModel) + "other" +
                                     std::string(markers::kInkContentToolJson) +
                                     R"({"name":"search","args":{"q":1}})" +
                                     std::string(markers::kInkEndMessage),
                                 std::string(markers::kInkMessageModel) + "search" +
                                     std::string(markers::kInkContentToolJson) +
                                     R"({"name":"search","args":{"q":}})" +
                                     std::string(markers::kInkEndMessage)}) {
        const ToolCallParse p = parse_tool_calls(t, ChatDialect::Inkling);
        CHECK(p.truncated);
        CHECK(p.calls.empty());
        CHECK_EQ(p.content, t);
    }
}

// reasoning_effort as a number in [0, 0.99] and as each of the seven names
// Mutation: map minimal -> low as the Qwen branch does and the line reads 0.2,
// not 0.1; reject numbers at parse and the request is a 400.
TEST(inkling_reasoning_effort_number_and_names) {
    const char* expected[] = {"0", "0.1", "0.2", "0.7", "0.9", "0.99", "0.99"};
    const char* names[]    = {"none", "minimal", "low", "medium", "high", "xhigh", "max"};
    for (size_t i = 0; i < 7; ++i) {
        const std::string got = render(conversation(names[i]));
        CHECK(got.find(std::string("<|content_text|>Thinking effort level: ") + expected[i] +
                       "<|end_message|>") != std::string::npos);
    }
    const std::string num = render(conversation(nullptr, true));
    CHECK(num.find("<|content_text|>Thinking effort level: 0.7<|end_message|>") != std::string::npos);
    // out of range is refused at parse
    nlohmann::json bad = {{"model", "inkling2-mini"},
                          {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                          {"reasoning_effort", 1.0}};
    ChatRequest req;
    CHECK(parse_chat_request(bad, req).has_value());
}

// reasoning_effort robustness for a template with its own effort map (1.8.0).
// Mutation: drop both the "off" -> "none" line in template_reasoning_effort and
// the enable_thinking=false fallback in template_context (either alone still
// yields none: the parser turns "off" into enable_thinking=false) and the "off"
// render falls to the template's default 0.9, as it did before 1.8.0.
TEST(inkling_reasoning_effort_off_is_none) {
    const std::string off  = render(conversation("off"));
    const std::string none = render(conversation("none"));
    const std::string dflt = render(conversation(nullptr));
    CHECK(off == none);
    CHECK(off != dflt);
}

// Mutation: drop the enable_thinking=false fallback in template_context and the
// render is the template default's (0.9), not none's.
TEST(inkling_enable_thinking_false_without_effort_is_none) {
    ChatRequest req = conversation(nullptr);
    nlohmann::json body = {{"model", "inkling2-mini"},
                           {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                           {"chat_template_kwargs", {{"enable_thinking", false}}}};
    ChatRequest parsed;
    CHECK(!parse_chat_request(body, parsed).has_value());
    CHECK(parsed.has_enable_thinking && !parsed.enable_thinking);
    req.has_enable_thinking = parsed.has_enable_thinking;
    req.enable_thinking     = parsed.enable_thinking;
    req.chat_template_kwargs = parsed.chat_template_kwargs;
    CHECK(render(req) == render(conversation("none")));
    // thinking left on and no level: the template's own default stays
    ChatRequest on = conversation(nullptr);
    on.has_enable_thinking = true;
    on.enable_thinking     = true;
    CHECK(render(on) == render(conversation(nullptr)));
}

// Mutation: make `known` accept every name and "turbo" passes; drop the range
// check and 1.5 passes.
TEST(inkling_strict_effort_refuses_unknown_names_with_the_list) {
    const std::vector<std::string> levels = template_reasoning_effort_levels(kInklingTemplate);
    CHECK(levels.size() == 7);
    for (const char* ok : {"none", "minimal", "low", "medium", "high", "xhigh", "max", "off"})
        CHECK(!check_strict_reasoning_effort(conversation(ok), levels).has_value());
    CHECK(!check_strict_reasoning_effort(conversation(nullptr), levels).has_value());
    CHECK(!check_strict_reasoning_effort(conversation(nullptr, true), levels).has_value());   // 0.7
    const auto bad = check_strict_reasoning_effort(conversation("turbo"), levels);
    CHECK(bad.has_value());
    CHECK(bad->find("\"turbo\"") != std::string::npos);
    CHECK(bad->find("\"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\", \"max\"") != std::string::npos);
    ChatRequest kw = conversation(nullptr);
    kw.chat_template_kwargs = {{"reasoning_effort", 1.5}};
    CHECK(check_strict_reasoning_effort(kw, levels).has_value());
    kw.chat_template_kwargs = {{"reasoning_effort", "fast"}};
    CHECK(check_strict_reasoning_effort(kw, levels).has_value());
    kw.chat_template_kwargs = {{"reasoning_effort", "off"}};
    CHECK(!check_strict_reasoning_effort(kw, levels).has_value());
    CHECK(render(kw) == render(conversation("none")));   // the kwarg's "off" is mapped too
}

// pi sends its thinkingLevelMap values as JSON strings: "0.7" is the number 0.7.
// Mutation: drop the numeric-string branch in template_context and "0.7" renders
// as the template default (0.9), not as 0.7; drop it in the strict check and
// "0.45" is refused.
TEST(inkling_reasoning_effort_numeric_string_is_the_number) {
    CHECK(render(conversation("0.7")) == render(conversation(nullptr, true)));   // 0.7 as a number
    CHECK(render(conversation("0.7")) != render(conversation(nullptr)));
    const std::vector<std::string> levels = template_reasoning_effort_levels(kInklingTemplate);
    CHECK(!check_strict_reasoning_effort(conversation("0.45"), levels).has_value());
    CHECK(!check_strict_reasoning_effort(conversation("0"), levels).has_value());
    for (const char* bad : {"1.5", "-0.1", "0x1p-2", "nan", "0.5abc"}) {
        nlohmann::json body = {{"model", "inkling2-mini"},
                               {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                               {"reasoning_effort", bad}};
        ChatRequest req;
        // kept as a string at parse (1.7.0's behaviour for every template); the strict check refuses
        // an out-of-range number and a name the template does not take alike
        CHECK(!parse_chat_request(body, req).has_value());
        CHECK(check_strict_reasoning_effort(req, levels).has_value());
    }
    // a template without its own effort map keeps ignoring a level it does not name
    ChatRequest q = conversation("0.45");
    std::string e;
    const nlohmann::json ctx = template_context(q, e, "{%- if reasoning_effort == 'low' -%}l{%- endif -%}");
    CHECK(!ctx.contains("reasoning_effort"));
}

// A request that names no effort is the template's own default (0.9), a thinking turn.
// Mutation: put Inkling back among the dialects that do not think by default and the
// chat route sets enable_thinking=false, which renders as none (0).
TEST(inkling_thinks_by_default) {
    CHECK(ChatDialect(ChatDialect::Inkling).thinks_by_default());
    CHECK(!ChatDialect(ChatDialect::Gemma4).thinks_by_default());
}

// 1.8.0 review S1: a typed numeric reasoning_effort survives a chat_template_kwargs object that
// does not name it. Mutation: let the client's object replace the kwargs whole and the render
// falls to the template default (0.9).
TEST(inkling_numeric_effort_survives_other_kwargs) {
    nlohmann::json body = {{"model", "inkling2-mini"},
                           {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                           {"reasoning_effort", 0.7},
                           {"chat_template_kwargs", {{"preserve_thinking", true}}}};
    ChatRequest parsed;
    CHECK(!parse_chat_request(body, parsed).has_value());
    CHECK(parsed.chat_template_kwargs.contains("reasoning_effort"));
    CHECK(parsed.chat_template_kwargs.contains("preserve_thinking"));
    ChatRequest req = conversation(nullptr);
    req.chat_template_kwargs = parsed.chat_template_kwargs;
    req.chat_template_kwargs.erase("preserve_thinking");
    CHECK(render(req) == render(conversation(nullptr, true)));   // 0.7, the numeric conversation
    // the client's own reasoning_effort in the object wins over the typed field
    nlohmann::json body2 = body;
    body2["chat_template_kwargs"] = {{"reasoning_effort", "low"}};
    ChatRequest p2;
    CHECK(!parse_chat_request(body2, p2).has_value());
    CHECK(p2.chat_template_kwargs["reasoning_effort"] == "low");
}

// Tool call ids unique across turns (the 1.8.1 hotfix), at the render. The template finds a
// tool result's name by tool_call_id over ALL messages and appends every match
// to the tool header, so two turns that both minted call_0 render the first
// result as "readbash": the earlier prompt text changes between turns and the
// lane's prefix misses. Each turn here is answered as the routes do: the model
// output parsed, then numbered after the history (core/toolcall.h).
// Mutation: next_call_index returning 0 (1.8.1's numbering) and both headers
// read <|message_tool|>readbash.
TEST(inkling_render_two_turn_tool_ids_do_not_merge_names) {
    auto model_call = [](const char* name) {
        return std::string(markers::kInkMessageModel) + name + std::string(markers::kInkContentToolJson) +
               "{\"name\":\"" + name + "\",\"args\":{}}" + std::string(markers::kInkEndMessage);
    };
    ChatRequest req;
    req.model = "inkling2-mini";
    req.messages.push_back({"user", "Read it, then run it.", {}, {}, {}, {}, {}});
    req.tools.push_back({"read", "Read a file", nlohmann::json::parse(R"({"type":"object","properties":{}})")});
    req.tools.push_back({"bash", "Run a command", nlohmann::json::parse(R"({"type":"object","properties":{}})")});
    for (const char* name : {"read", "bash"}) {
        ToolCallParse p = parse_tool_calls(model_call(name), ChatDialect::Inkling);
        CHECK_EQ(p.calls.size(), 1u);
        if (p.calls.size() != 1) return;
        number_calls(p.calls, next_call_index(req));
        ChatMessage assistant;
        assistant.role       = "assistant";
        assistant.tool_calls = p.calls;
        req.messages.push_back(assistant);
        req.messages.push_back({"tool", "ok", {}, "", p.calls[0].id, {}, {}});
    }
    CHECK_EQ(req.messages[1].tool_calls[0].id, std::string("call_0"));
    CHECK_EQ(req.messages[3].tool_calls[0].id, std::string("call_1"));
    const std::string got = render(req);
    const std::string read_hdr = std::string("<|message_tool|>read<|content_text|>ok");
    const std::string bash_hdr = std::string("<|message_tool|>bash<|content_text|>ok");
    CHECK(got.find(read_hdr) != std::string::npos);
    CHECK(got.find(bash_hdr) != std::string::npos);
    CHECK(got.find("<|message_tool|>readbash") == std::string::npos);
    // the turn-one render up to and including the first result is what turn two
    // renders too: the first result's header did not change under the new call
    ChatRequest after_one = req;
    after_one.messages.resize(3);
    const std::string one = render(after_one);
    CHECK(!one.empty() && got.compare(0, one.size(), one) == 0);
}
