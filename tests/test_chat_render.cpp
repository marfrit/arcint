#include "core/chat.h"
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
