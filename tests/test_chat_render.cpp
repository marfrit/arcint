#include "core/chat.h"
#include "core/dialect.h"
#include "exec/chat_json.h"
#include "harness.h"

#include <exception>
#include <string>

// minja declares `using json = nlohmann::ordered_json` at global scope, which
// clashes with the unit files' `using nlohmann::json`; the render case lives
// here, in its own translation unit, for that reason.
#include <minja/chat-template.hpp>

using namespace lgc;

namespace {
// Qwen3.8's effort line and its refusal, reduced to the fragment that matters,
// plus one output so the render shows the resolved level.
constexpr const char* kQwen38Fragment =
    "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
    "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}"
    "{{- raise_exception('bad effort') }}{%- endif %}"
    "effort={{ resolved_reasoning_effort }}";

nlohmann::json effort_context(const char* level) {
    nlohmann::json body = {{"model", "qwen3.8-27b"},
                           {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                           {"reasoning_effort", level}};
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    std::string err;
    nlohmann::json ctx = template_context(req, err, kQwen38Fragment);
    CHECK(err.empty());
    return ctx;
}
}  // namespace

// The consumer's path without a model: the context template_context builds,
// rendered by minja (vendored, header-only) the way the llama render_chat does
// (apply_polyfills off), through a Qwen3.8-style fragment that raises on any
// level outside xhigh/medium/low. The negative arm shows the fragment can fail:
// the client's raw "high" raises, so the mapping is what makes it render.
TEST(reasoning_effort_renders_through_a_qwen38_style_template) {
    minja::chat_template tmpl(kQwen38Fragment, "", "");
    auto render = [&](const nlohmann::json& extra) {
        minja::chat_template_inputs in;
        in.messages              = nlohmann::ordered_json::array({{{"role", "user"}, {"content", "hi"}}});
        in.add_generation_prompt = true;
        if (!extra.empty()) in.extra_context = extra;
        minja::chat_template_options opts;
        opts.apply_polyfills = false;
        return tmpl.apply(in, opts);
    };
    CHECK_EQ(render(nlohmann::json::object()), std::string("effort=xhigh"));   // the default

    CHECK_EQ(render(effort_context("medium")), std::string("effort=medium"));
    CHECK_EQ(render(effort_context("high")), std::string("effort=xhigh"));
    CHECK_EQ(render(effort_context("low")), std::string("effort=low"));

    bool raw_high_raises = false;
    try {
        (void)render(nlohmann::json{{"reasoning_effort", "high"}});
    } catch (const std::exception&) {
        raw_high_raises = true;
    }
    CHECK(raw_high_raises);
}

// The real Qwen3.8 template maps "high" to "xhigh" itself, so the client's
// level passes through unchanged and the template translates it.
TEST(reasoning_effort_high_passes_through_a_template_that_maps_it) {
    constexpr const char* kMapsHigh =
        "{%- set r = reasoning_effort|default('xhigh') %}"
        "{%- if r == 'high' %}{%- set r = 'xhigh' %}{%- endif %}"
        "{%- if r not in ('xhigh', 'medium', 'low') %}{{- raise_exception('bad effort') }}{%- endif %}"
        "effort={{ r }}";
    nlohmann::json body = {{"model", "qwen3.8-27b"},
                           {"messages", nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}})},
                           {"reasoning_effort", "high"}};
    ChatRequest req;
    CHECK(!parse_chat_request(body, req).has_value());
    std::string err;
    const nlohmann::json ctx = template_context(req, err, kMapsHigh);
    CHECK(err.empty());
    CHECK_EQ(ctx.value("reasoning_effort", nlohmann::json()), nlohmann::json("high"));
    minja::chat_template tmpl(kMapsHigh, "", "");
    minja::chat_template_inputs in;
    in.messages              = nlohmann::ordered_json::array({{{"role", "user"}, {"content", "hi"}}});
    in.add_generation_prompt = true;
    in.extra_context         = ctx;
    minja::chat_template_options opts;
    opts.apply_polyfills = false;
    CHECK_EQ(tmpl.apply(in, opts), std::string("effort=xhigh"));
}

// A template that declares its tools the way Granite 4.2's does,
// `value | tojson(ensure_ascii=False)`: the keyword is accepted (vendored minja,
// third_party/minja/PATCHES.md), non-ASCII text is written as it is, and the
// render with tools no longer throws. Synthetic fragment, same form.
TEST(tojson_takes_the_ensure_ascii_keyword) {
    minja::chat_template tmpl(
        "{%- for tool in tools -%}{{ '\\n' ~ (tool | tojson(ensure_ascii=False)) }}{%- endfor -%}", "", "");
    minja::chat_template_inputs in;
    in.messages = nlohmann::ordered_json::array({{{"role", "user"}, {"content", "hi"}}});
    in.tools    = nlohmann::ordered_json::parse(R"([{"name":"wetter","description":"Temperatur in M\u00fcnchen"}])");
    in.add_generation_prompt = true;
    minja::chat_template_options opts;
    opts.apply_polyfills = false;
    std::string out;
    try {
        out = tmpl.apply(in, opts);
    } catch (const std::exception& e) {
        CHECK_EQ(std::string(e.what()), std::string(""));   // a throw is a failure, not an abort
    }
    CHECK(out.find("\"name\": \"wetter\"") != std::string::npos);
    CHECK(out.find("M\xc3\xbcnchen") != std::string::npos);   // not escaped
}

// Gemma 4's template shape, reduced: a macro that wraps string values in a
// delimiter built by concatenation, dictsort over a mapping, a namespace, and a
// reasoning field read with .get(). The delimiter is a placeholder here, not
// the model's marker.
TEST(gemma_style_template_shapes_render) {
    minja::chat_template tmpl(
        "{%- macro fmt(v) -%}{%- if v is string -%}{{- '<q>' + v + '<q>' -}}{%- elif v is mapping -%}"
        "{{- '{' -}}{%- for k, x in v | dictsort -%}{{- k -}}:{{- fmt(x) -}}{%- if not loop.last %},{% endif -%}"
        "{%- endfor -%}{{- '}' -}}{%- else -%}{{- v -}}{%- endif -%}{%- endmacro -%}"
        "{%- set ns = namespace(n=0) -%}"
        "{%- for m in messages -%}{%- set ns.n = ns.n + 1 -%}{{- m['role'] ~ ':' -}}"
        "{{- m.get('reasoning_content') or '-' -}}{{- fmt({'b': 2, 'a': 'x'}) -}};{%- endfor -%}{{ ns.n }}",
        "", "");
    minja::chat_template_inputs in;
    in.messages = nlohmann::ordered_json::array({{{"role", "user"}, {"content", "hi"}}});
    in.add_generation_prompt = true;
    minja::chat_template_options opts;
    opts.apply_polyfills = false;
    CHECK_EQ(tmpl.apply(in, opts), std::string("user:-{a:<q>x<q>,b:2};1"));
}

// Gemma 4's template renders a mapping in its own form and a string by stripping
// braces, and minja's capability probe accepts a string: the dialect's rule is
// what keeps the replayed call in the model's syntax. Synthetic template, same
// branch: the first arm reads the arguments as a mapping.
TEST(gemma4_replays_tool_call_arguments_as_an_object) {
    minja::chat_template tmpl(
        "{%- for m in messages -%}{%- for tc in m.tool_calls or [] -%}"
        "{%- if tc.function.arguments is mapping -%}OBJ:{{ tc.function.arguments.city }}"
        "{%- else -%}STR:{{ tc.function.arguments }}{%- endif -%}{%- endfor -%}{%- endfor -%}",
        "", "");
    ChatRequest req;
    ChatMessage a;
    a.role = "assistant";
    ToolCall c;
    c.id        = "call_0";
    c.name      = "f";
    c.arguments = R"({"city":"Kassel"})";
    a.tool_calls.push_back(c);
    req.messages.push_back(a);
    minja::chat_template_options opts;
    opts.apply_polyfills = false;
    auto render = [&](ChatDialect d) {
        minja::chat_template_inputs in;
        in.messages = chat_messages_json(req, tmpl.original_caps().requires_object_arguments ||
                                                  d.replays_arguments_as_object());
        return tmpl.apply(in, opts);
    };
    CHECK(!tmpl.original_caps().requires_object_arguments);   // the probe alone would pass a string
    CHECK_EQ(render(ChatDialect::Gemma4), std::string("OBJ:Kassel"));
    CHECK_EQ(render(ChatDialect::Qwen), std::string("STR:{\"city\":\"Kassel\"}"));
    CHECK_EQ(render(ChatDialect::Extended), std::string("STR:{\"city\":\"Kassel\"}"));
}
