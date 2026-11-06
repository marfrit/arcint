// Renders a chat template read from a file with the engine's own renderer
// (vendored minja, polyfills off, the same context construction as the llama
// backend's render_chat) over a small matrix: with and without tools,
// enable_thinking unset / true / false, a plain exchange, a system message, and
// a tool round trip with an earlier assistant turn. Prints every prompt and a
// summary line per case; exits non-zero when a render throws.
//
// The template text is never part of the repository: pass the path.
//
//   arcint-template-probe <template-file> [bos-text] [eos-text] [objargs]
//
// Not a ctest case: it needs a template taken from a GGUF.
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <minja/chat-template.hpp>
#include <string>
#include <vector>

#include "core/chat.h"
#include "exec/chat_json.h"

using namespace lgc;

namespace {

ChatMessage msg(const std::string& role, const std::string& content) {
    ChatMessage m;
    m.role    = role;
    m.content = content;
    return m;
}

ToolSpec weather_tool() {
    ToolSpec t;
    t.name        = "get_weather";
    t.description = "Current weather in a city";
    t.parameters  = nlohmann::json::parse(R"({"type":"object","properties":{"city":{"type":"string","description":"the city"},"days":{"type":"integer"}},"required":["city"]})");
    return t;
}

std::vector<ChatMessage> exchange(const std::string& kind) {
    std::vector<ChatMessage> m;
    if (kind == "plain") {
        m.push_back(msg("user", "What is the capital of France?"));
    } else if (kind == "system") {
        m.push_back(msg("system", "You answer in one word."));
        m.push_back(msg("user", "What is the capital of France?"));
    } else if (kind == "tool_round_trip") {
        m.push_back(msg("system", "You may call tools."));
        m.push_back(msg("user", "Weather in Kassel?"));
        ChatMessage a = msg("assistant", "");
        a.reasoning   = "I should look it up.";
        ToolCall c;
        c.id        = "call_0";
        c.name      = "get_weather";
        c.arguments = R"({"city":"Kassel","days":2})";
        a.tool_calls.push_back(c);
        m.push_back(a);
        ChatMessage r = msg("tool", "12 C, cloudy");
        r.tool_call_id = "call_0";
        r.name         = "get_weather";
        m.push_back(r);
    } else {   // two turns
        m.push_back(msg("user", "Hello"));
        m.push_back(msg("assistant", "Hi."));
        m.push_back(msg("user", "What is the capital of France?"));
    }
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <template-file> [bos-text] [eos-text] [objargs]\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    const std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::string bos = argc > 2 ? argv[2] : "";
    const std::string eos = argc > 3 ? argv[3] : "";
    // "objargs": replay tool-call arguments as an object, as the backend does for
    // a Gemma 4 model (ChatDialect::replays_arguments_as_object)
    const bool force_object = argc > 4 && std::string(argv[4]) == "objargs";

    std::unique_ptr<minja::chat_template> tmpl;
    try {
        tmpl = std::make_unique<minja::chat_template>(src, bos, eos);
    } catch (const std::exception& e) {
        std::printf("PARSE FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("template: %zu bytes, requires_object_arguments=%d, effort levels:", src.size(),
                tmpl->original_caps().requires_object_arguments ? 1 : 0);
    for (const std::string& l : template_reasoning_effort_levels(src)) std::printf(" %s", l.c_str());
    std::printf("\n");

    int failures = 0, cases = 0;
    for (const char* kind : {"plain", "system", "two_turns", "tool_round_trip"}) {
        for (int with_tools = 0; with_tools < 2; ++with_tools) {
            for (int think = 0; think < 3; ++think) {   // 0 unset, 1 true, 2 false
                ChatRequest req;
                req.messages = exchange(kind);
                if (with_tools) req.tools.push_back(weather_tool());
                if (think != 0) {
                    req.has_enable_thinking = true;
                    req.enable_thinking     = think == 1;
                }
                ++cases;
                const std::string label = std::string(kind) + (with_tools ? " +tools" : " -tools") +
                                          (think == 0 ? " thinking=unset" : think == 1 ? " thinking=true" : " thinking=false");
                try {
                    minja::chat_template_inputs inputs;
                    inputs.messages = chat_messages_json(req, tmpl->original_caps().requires_object_arguments || force_object);
                    inputs.tools    = chat_tools_json(req);
                    inputs.add_generation_prompt = true;
                    std::string ctx_error;
                    const nlohmann::json extra = template_context(req, ctx_error, src);
                    if (!extra.empty()) inputs.extra_context = extra;
                    minja::chat_template_options opts;
                    opts.apply_polyfills = false;
                    const std::string prompt = tmpl->apply(inputs, opts);
                    std::printf("---- %s: OK, %zu bytes\n%s\n", label.c_str(), prompt.size(), prompt.c_str());
                } catch (const std::exception& e) {
                    ++failures;
                    std::printf("---- %s: FAIL: %s\n", label.c_str(), e.what());
                }
            }
        }
    }
    std::printf("==== %d cases, %d failures\n", cases, failures);
    return failures == 0 ? 0 : 1;
}
