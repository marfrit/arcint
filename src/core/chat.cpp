#include "core/chat.h"

#include "util/log.h"
#include "util/text.h"

namespace lgc {
namespace {

using json = nlohmann::json;

bool known_role(const std::string& r) {
    return r == "system" || r == "developer" || r == "user" || r == "assistant" || r == "tool";
}

// OpenAI allows `content` to be either a string or an array of typed parts.
// v1 is text-only (DESIGN.md §8 defers multimodal), so text parts are joined
// and anything else is a hard error rather than a silent drop.
std::optional<std::string> read_content(const json& v, std::string& out, size_t index) {
    if (v.is_null()) {
        out.clear();
        return std::nullopt;
    }
    if (v.is_string()) {
        out = v.get<std::string>();
        return std::nullopt;
    }
    if (!v.is_array()) {
        return log::format("messages[%zu].content must be a string, an array, or null", index);
    }

    std::string joined;
    for (const json& part : v) {
        if (!part.is_object() || !part.contains("type")) {
            return log::format("messages[%zu].content parts must be typed objects", index);
        }
        const std::string type = part["type"].get<std::string>();
        if (type != "text") {
            return log::format(
                "messages[%zu].content: part type '%s' is not supported (v1 is text-only)",
                index, type.c_str());
        }
        if (!part.contains("text") || !part["text"].is_string()) {
            return log::format("messages[%zu].content: text part is missing 'text'", index);
        }
        joined += part["text"].get<std::string>();
    }
    out = std::move(joined);
    return std::nullopt;
}

std::optional<std::string> read_tool_calls(const json& v, std::vector<ToolCall>& out,
                                           size_t index) {
    if (!v.is_array()) return log::format("messages[%zu].tool_calls must be an array", index);

    for (const json& c : v) {
        if (!c.is_object() || !c.contains("function") || !c["function"].is_object()) {
            return log::format("messages[%zu].tool_calls entries need a 'function' object", index);
        }
        ToolCall call;
        if (c.contains("id") && c["id"].is_string()) call.id = c["id"].get<std::string>();

        const json& fn = c["function"];
        if (!fn.contains("name") || !fn["name"].is_string()) {
            return log::format("messages[%zu].tool_calls: function.name is required", index);
        }
        call.name = fn["name"].get<std::string>();
        if (fn.contains("arguments")) {
            const json& a = fn["arguments"];
            call.arguments = a.is_string() ? a.get<std::string>() : a.dump();
        } else {
            call.arguments = "{}";
        }
        out.push_back(std::move(call));
    }
    return std::nullopt;
}

std::optional<std::string> read_stop(const json& v, SamplerOverrides& s) {
    if (v.is_null()) return std::nullopt;

    std::vector<std::string> stop;
    if (v.is_string()) {
        stop.push_back(v.get<std::string>());
    } else if (v.is_array()) {
        for (const json& e : v) {
            if (!e.is_string()) return "stop entries must be strings";
            stop.push_back(e.get<std::string>());
        }
    } else {
        return "stop must be a string or an array of strings";
    }

    // An empty stop string matches everywhere and would end every generation
    // before the first token.
    for (const std::string& e : stop) {
        if (e.empty()) return "stop entries must not be empty";
    }
    s.stop = std::move(stop);
    return std::nullopt;
}

// Reads the sampler fields shared by both endpoints.
std::optional<std::string> read_sampler(const json& b, SamplerOverrides& s) {
    auto num = [&](const char* key, std::optional<float>& dst) -> std::optional<std::string> {
        if (!b.contains(key) || b[key].is_null()) return std::nullopt;
        if (!b[key].is_number()) return log::format("%s must be a number", key);
        dst = b[key].get<float>();
        return std::nullopt;
    };
    auto integer = [&](const char* key, std::optional<int>& dst) -> std::optional<std::string> {
        if (!b.contains(key) || b[key].is_null()) return std::nullopt;
        if (!b[key].is_number_integer()) return log::format("%s must be an integer", key);
        dst = b[key].get<int>();
        return std::nullopt;
    };

    if (auto e = num("temperature", s.temperature)) return e;
    if (auto e = num("top_p", s.top_p)) return e;
    if (auto e = num("repetition_penalty", s.repetition_penalty)) return e;
    if (auto e = num("presence_penalty", s.presence_penalty)) return e;
    if (auto e = num("frequency_penalty", s.frequency_penalty)) return e;
    if (auto e = integer("top_k", s.top_k)) return e;
    if (auto e = integer("max_tokens", s.max_tokens)) return e;
    // OpenAI's newer name for the same field; explicit max_tokens wins.
    if (!s.max_tokens) {
        if (auto e = integer("max_completion_tokens", s.max_tokens)) return e;
    }

    if (b.contains("seed") && !b["seed"].is_null()) {
        if (!b["seed"].is_number_integer()) return "seed must be an integer";
        s.seed = static_cast<uint64_t>(b["seed"].get<int64_t>());
    }
    if (b.contains("ignore_eos") && !b["ignore_eos"].is_null()) {
        if (!b["ignore_eos"].is_boolean()) return "ignore_eos must be a boolean";
        s.ignore_eos = b["ignore_eos"].get<bool>();
    }
    if (b.contains("stop")) {
        if (auto e = read_stop(b["stop"], s)) return e;
    }
    if (b.contains("stop_token_ids") && !b["stop_token_ids"].is_null()) {
        if (!b["stop_token_ids"].is_array()) return "stop_token_ids must be an array";
        std::vector<int> ids;
        for (const json& e : b["stop_token_ids"]) {
            if (!e.is_number_integer()) return "stop_token_ids entries must be integers";
            ids.push_back(e.get<int>());
        }
        s.stop_token_ids = std::move(ids);
    }

    if (b.contains("n") && !b["n"].is_null()) {
        if (!b["n"].is_number_integer() || b["n"].get<int>() != 1) {
            // No n > 1 in v1: one slot serves one sequence, and fanning out
            // silently would misreport usage.
            return "n must be 1 (multiple choices per request are not supported)";
        }
    }

    return std::nullopt;
}

std::optional<std::string> read_stream(const json& b, bool& stream, bool& include_usage) {
    if (b.contains("stream") && !b["stream"].is_null()) {
        if (!b["stream"].is_boolean()) return "stream must be a boolean";
        stream = b["stream"].get<bool>();
    }
    if (b.contains("stream_options") && !b["stream_options"].is_null()) {
        const json& so = b["stream_options"];
        if (!so.is_object()) return "stream_options must be an object";
        if (so.contains("include_usage") && !so["include_usage"].is_null()) {
            if (!so["include_usage"].is_boolean()) {
                return "stream_options.include_usage must be a boolean";
            }
            include_usage = so["include_usage"].get<bool>();
        }
    }
    return std::nullopt;
}

}  // namespace

std::optional<std::string> parse_chat_request(const json& body, ChatRequest& out) {
    if (!body.is_object()) return "request body must be a JSON object";

    ChatRequest req;
    if (body.contains("model") && body["model"].is_string()) {
        req.model = body["model"].get<std::string>();
    }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        return "messages is required and must be an array";
    }
    if (body["messages"].empty()) return "messages must not be empty";

    size_t index = 0;
    for (const json& m : body["messages"]) {
        if (!m.is_object()) return log::format("messages[%zu] must be an object", index);

        ChatMessage msg;
        if (!m.contains("role") || !m["role"].is_string()) {
            return log::format("messages[%zu].role is required", index);
        }
        msg.role = m["role"].get<std::string>();
        if (!known_role(msg.role)) {
            return log::format("messages[%zu].role '%s' is not a known role", index,
                               msg.role.c_str());
        }

        if (m.contains("content")) {
            if (auto e = read_content(m["content"], msg.content, index)) return e;
        }
        if (m.contains("name") && m["name"].is_string()) {
            msg.name = m["name"].get<std::string>();
        }
        if (m.contains("tool_call_id") && m["tool_call_id"].is_string()) {
            msg.tool_call_id = m["tool_call_id"].get<std::string>();
        }
        if (m.contains("tool_calls") && !m["tool_calls"].is_null()) {
            if (auto e = read_tool_calls(m["tool_calls"], msg.tool_calls, index)) return e;
        }

        if (msg.role == "tool" && msg.tool_call_id.empty()) {
            return log::format("messages[%zu]: a tool message needs tool_call_id", index);
        }
        if (msg.content.empty() && msg.tool_calls.empty() && msg.role != "assistant") {
            return log::format("messages[%zu]: content must not be empty for role '%s'", index,
                               msg.role.c_str());
        }

        req.messages.push_back(std::move(msg));
        ++index;
    }

    if (body.contains("tools") && !body["tools"].is_null()) {
        if (!body["tools"].is_array()) return "tools must be an array";
        size_t ti = 0;
        for (const json& t : body["tools"]) {
            if (!t.is_object()) return log::format("tools[%zu] must be an object", ti);
            if (t.contains("type") && t["type"].is_string() &&
                t["type"].get<std::string>() != "function") {
                return log::format("tools[%zu].type must be 'function'", ti);
            }
            if (!t.contains("function") || !t["function"].is_object()) {
                return log::format("tools[%zu].function is required", ti);
            }
            const json& fn = t["function"];
            if (!fn.contains("name") || !fn["name"].is_string()) {
                return log::format("tools[%zu].function.name is required", ti);
            }

            ToolSpec spec;
            spec.name = fn["name"].get<std::string>();
            if (fn.contains("description") && fn["description"].is_string()) {
                spec.description = fn["description"].get<std::string>();
            }
            spec.parameters = fn.contains("parameters") ? fn["parameters"] : json::object();
            req.tools.push_back(std::move(spec));
            ++ti;
        }
    }

    if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
        const json& tc = body["tool_choice"];
        if (tc.is_string()) {
            req.tool_choice = tc.get<std::string>();
            if (req.tool_choice != "auto" && req.tool_choice != "none" &&
                req.tool_choice != "required") {
                return "tool_choice must be 'auto', 'none', 'required', or a function object";
            }
        } else if (tc.is_object()) {
            req.tool_choice = "required";
        } else {
            return "tool_choice must be a string or an object";
        }
    }

    if (auto e = read_stream(body, req.stream, req.stream_include_usage)) return e;
    if (auto e = read_sampler(body, req.sampler)) return e;

    if (body.contains("chat_template_kwargs") && !body["chat_template_kwargs"].is_null()) {
        const json& kw = body["chat_template_kwargs"];
        if (!kw.is_object()) return "chat_template_kwargs must be an object";
        if (kw.contains("enable_thinking") && !kw["enable_thinking"].is_null()) {
            if (!kw["enable_thinking"].is_boolean()) {
                return "chat_template_kwargs.enable_thinking must be a boolean";
            }
            req.enable_thinking     = kw["enable_thinking"].get<bool>();
            req.has_enable_thinking = true;
        }
    }

    out = std::move(req);
    return std::nullopt;
}

std::optional<std::string> parse_completion_request(const json& body, CompletionRequest& out) {
    if (!body.is_object()) return "request body must be a JSON object";

    CompletionRequest req;
    if (body.contains("model") && body["model"].is_string()) {
        req.model = body["model"].get<std::string>();
    }

    if (!body.contains("prompt")) return "prompt is required";
    const json& p = body["prompt"];
    if (p.is_string()) {
        req.prompt = p.get<std::string>();
    } else if (p.is_array()) {
        // A single-element array of strings is the common client shape. Token-id
        // prompts and batches are not v1 (one slot, one sequence).
        if (p.size() != 1 || !p[0].is_string()) {
            return "prompt must be a string or a one-element array of strings";
        }
        req.prompt = p[0].get<std::string>();
    } else {
        return "prompt must be a string or a one-element array of strings";
    }

    if (body.contains("echo") && !body["echo"].is_null()) {
        if (!body["echo"].is_boolean()) return "echo must be a boolean";
        req.echo = body["echo"].get<bool>();
    }

    bool include_usage = true;
    if (auto e = read_stream(body, req.stream, include_usage)) return e;
    if (auto e = read_sampler(body, req.sampler)) return e;

    out = std::move(req);
    return std::nullopt;
}

ToolSchemas tool_schemas(const std::vector<ToolSpec>& tools) {
    ToolSchemas schemas;
    for (const ToolSpec& t : tools) {
        ParamTypes types;
        if (t.parameters.is_object() && t.parameters.contains("properties") &&
            t.parameters["properties"].is_object()) {
            for (const auto& [key, spec] : t.parameters["properties"].items()) {
                if (spec.is_object() && spec.contains("type") && spec["type"].is_string()) {
                    types[key] = spec["type"].get<std::string>();
                }
            }
        }
        schemas[t.name] = std::move(types);
    }
    return schemas;
}

std::string render_chatml_stub(const ChatRequest& req) {
    std::string out;

    if (!req.tools.empty()) {
        json declared = json::array();
        for (const ToolSpec& t : req.tools) {
            declared.push_back({{"type", "function"},
                                {"function",
                                 {{"name", t.name},
                                  {"description", t.description},
                                  {"parameters", t.parameters}}}});
        }
        out += "<|im_start|>system\n# Tools\n";
        out += declared.dump();
        out += "<|im_end|>\n";
    }

    for (const ChatMessage& m : req.messages) {
        out += "<|im_start|>";
        out += m.role == "developer" ? "system" : m.role;
        out += "\n";
        out += m.content;
        for (const ToolCall& c : m.tool_calls) {
            out += "\n<tool_call>\n";
            out += json{{"name", c.name}, {"arguments", c.arguments}}.dump();
            out += "\n</tool_call>";
        }
        out += "<|im_end|>\n";
    }

    out += "<|im_start|>assistant\n";
    if (!req.enable_thinking) out += "<think>\n\n</think>\n\n";
    return out;
}

}  // namespace lgc
