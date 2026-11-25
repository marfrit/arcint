#pragma once

// A chat request as the JSON a Jinja chat template reads (messages and
// tools), shared by every executor that renders the artifact's own template.

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "core/chat.h"

namespace lgc {

// The function name a tool result answers: a call with its tool_call_id in the
// nearest assistant turn before it (scanning back from `index`). Within that turn
// the results pair with the calls by position: the k-th result carrying the id
// after the turn answers the k-th call carrying it (two calls both `call_0`, two
// results: the first result is the first call's, the second the second's). A
// result past the turn's last such call takes the first. Null when no earlier
// assistant turn carries the id.
inline const std::string* tool_result_call_name(const ChatRequest& req, size_t index) {
    const std::string& id = req.messages[index].tool_call_id;
    if (id.empty()) return nullptr;
    for (size_t i = index; i-- > 0;) {
        const ChatMessage& prev = req.messages[i];
        if (prev.role != "assistant") continue;
        std::vector<const std::string*> names;
        for (const ToolCall& c : prev.tool_calls)
            if (c.id == id) names.push_back(&c.name);
        if (names.empty()) continue;
        size_t k = 0;   // results with this id between the turn and this one
        for (size_t j = i + 1; j < index; ++j)
            if (req.messages[j].role == "tool" && req.messages[j].tool_call_id == id) ++k;
        return k < names.size() ? names[k] : names.front();
    }
    return nullptr;
}

// `name_tool_results` (ChatDialect::names_tool_results, Inkling): a tool result the
// client sent without a `name` gets the name of the call it answers, so the template
// does not resolve it over the whole history. A client's own `name` is kept. No
// default: every call site states it (a site that dropped it would render Inkling's
// repeated ids merged again, and the tests could not see it).
inline nlohmann::json chat_messages_json(const ChatRequest& req, bool object_arguments,
                                         bool name_tool_results) {
    nlohmann::json out = nlohmann::json::array();
    for (size_t index = 0; index < req.messages.size(); ++index) {
        const ChatMessage& m = req.messages[index];
        nlohmann::json msg{{"role", m.role}, {"content", m.content}};
        if (!m.name.empty()) msg["name"] = m.name;
        else if (name_tool_results && m.role == "tool")
            if (const std::string* n = tool_result_call_name(req, index); n != nullptr && !n->empty()) msg["name"] = *n;
        if (!m.tool_call_id.empty()) msg["tool_call_id"] = m.tool_call_id;
        // The template's own field: it decides what to show (Qwen3 keeps it for
        // turns after the last user message, drops it earlier; a template that
        // does not reference reasoning_content ignores it).
        if (!m.reasoning.empty()) msg["reasoning_content"] = m.reasoning;
        if (!m.tool_calls.empty()) {
            nlohmann::json calls = nlohmann::json::array();
            for (const ToolCall& c : m.tool_calls) {
                calls.push_back({{"id", c.id},
                                 {"type", "function"},
                                 {"function",
                                  {{"name", c.name},
                                   {"arguments", tool_call_arguments_for_template(c.arguments, object_arguments)}}}});
            }
            msg["tool_calls"] = std::move(calls);
            if (m.content.empty()) msg["content"] = nullptr;
        }
        out.push_back(std::move(msg));
    }
    return out;
}

inline nlohmann::json chat_tools_json(const ChatRequest& req) {
    if (req.tools.empty()) return nlohmann::json();
    nlohmann::json out = nlohmann::json::array();
    for (const ToolSpec& t : req.tools) {
        out.push_back({{"type", "function"},
                       {"function", {{"name", t.name}, {"description", t.description}, {"parameters", t.parameters}}}});
    }
    return out;
}

}  // namespace lgc
