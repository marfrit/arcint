#pragma once

// A chat request as the JSON a Jinja chat template reads (messages and
// tools), shared by every executor that renders the artifact's own template.

#include <nlohmann/json.hpp>

#include "core/chat.h"

namespace lgc {

inline nlohmann::json chat_messages_json(const ChatRequest& req, bool object_arguments) {
    nlohmann::json out = nlohmann::json::array();
    for (const ChatMessage& m : req.messages) {
        nlohmann::json msg{{"role", m.role}, {"content", m.content}};
        if (!m.name.empty()) msg["name"] = m.name;
        if (!m.tool_call_id.empty()) msg["tool_call_id"] = m.tool_call_id;
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
