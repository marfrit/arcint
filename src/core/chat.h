#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/sampling.h"
#include "core/toolcall.h"

namespace lgc {

struct ToolSpec {
    std::string    name;
    std::string    description;
    nlohmann::json parameters;  // JSON Schema object, as declared by the client
};

struct ChatMessage {
    std::string           role;
    std::string           content;
    std::string           name;          // function name, for role == "tool"
    std::string           tool_call_id;  // for role == "tool"
    std::vector<ToolCall> tool_calls;    // assistant turns replayed back to us
};

struct ChatRequest {
    std::string              model;
    std::vector<ChatMessage> messages;
    std::vector<ToolSpec>    tools;
    std::string              tool_choice = "auto";

    // Set when tool_choice named one function explicitly; empty otherwise.
    std::string tool_choice_function;

    bool stream = false;

    // stream_options.include_usage. Defaults on: DESIGN.md §3.7 requires usage
    // in the final chunk, so opting out is the client's explicit choice.
    bool stream_include_usage = true;

    // chat_template_kwargs.enable_thinking — the fleet standard switch (§4).
    bool enable_thinking     = true;
    bool has_enable_thinking = false;

    SamplerOverrides sampler;
};

struct CompletionRequest {
    std::string      model;
    std::string      prompt;
    bool             stream = false;
    bool             echo   = false;
    bool             stream_include_usage = true;
    SamplerOverrides sampler;
};

// Both return an error message on rejection and leave the output untouched.
std::optional<std::string> parse_chat_request(const nlohmann::json& body, ChatRequest& out);
std::optional<std::string> parse_completion_request(const nlohmann::json& body,
                                                    CompletionRequest& out);

// Parameter types from the declared tool schemas, for coercing the coder
// model's XML tool-call values (see core/toolcall.h).
ToolSchemas tool_schemas(const std::vector<ToolSpec>& tools);

// M0 stub prompt rendering.
//
// This is NOT the model's chat template. DESIGN.md §3.7 makes the artifact the
// single source of truth for the template, and M1 replaces this with the jinja
// template shipped in the IR directory. Until then the stub backend needs some
// deterministic rendering to count tokens against, and ChatML is the family's
// shape. Never let this run against a real model.
std::string render_chatml_stub(const ChatRequest& req);

}  // namespace lgc
