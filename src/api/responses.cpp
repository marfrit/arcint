#include "api/responses.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "api/error.h"
#include "api/generate.h"
#include "core/chat.h"
#include "util/log.h"

// Open Responses 2026-04-24 request mapping (docs/campaigns/feyntalk.md W1).
// The mapping rows and their refusals are the campaign's table; every row that
// 1.3.0 does not build is refused loudly with the spec's own code
// `unsupported_parameter`, never accepted and dropped.
namespace lgc::api {
namespace {

using json = nlohmann::json;

constexpr size_t kMaxInputChars = 10485760;

HttpResult invalid( std::string message, std::string param = {}) {
    return HttpResult{400, responses_error(std::move(message), "invalid_request",
                                           "invalid_request", param)};
}

HttpResult unsupported(std::string message, std::string param) {
    return HttpResult{400, responses_error(std::move(message), "invalid_request",
                                           "unsupported_parameter", param)};
}

// The top-level fields this surface knows. Anything else is refused rather
// than silently ignored: a field a client sent and the server dropped is the
// silent weakening the campaign forbids.
bool known_top_level(std::string_view key) {
    static const char* const kKnown[] = {
        "model", "input", "previous_response_id", "include", "tools", "tool_choice",
        "metadata", "text", "temperature", "top_p", "presence_penalty",
        "frequency_penalty", "parallel_tool_calls", "stream", "stream_options",
        "background", "max_output_tokens", "max_tool_calls", "reasoning",
        "safety_identifier", "prompt_cache_key", "truncation", "instructions",
        "store", "service_tier", "top_logprobs", "arcint"};
    for (const char* k : kKnown) if (key == k) return true;
    return false;
}

// One message content part list. Parts are concatenated with a newline between
// them (the campaign's mapping row); an image part records an ImagePart and
// puts the renderer's media marker into the text. The other modalities are
// refused: no projector path exists for files or video.
std::optional<HttpResult> read_message_content(const json& content, ChatMessage& msg,
                                               const std::string& where) {
    if (content.is_null()) return std::nullopt;
    if (content.is_string()) {
        msg.content = content.get<std::string>();
        return std::nullopt;
    }
    if (!content.is_array()) return invalid(where + ".content must be a string or an array", "input");

    std::string joined;
    for (const json& part : content) {
        if (!part.is_object() || !part.contains("type") || !part["type"].is_string()) {
            return invalid(where + ".content parts must be typed objects", "input");
        }
        const std::string type = part["type"].get<std::string>();
        if (type == "input_image") {
            if (!part.contains("image_url") || !part["image_url"].is_string()) {
                return invalid(where + ".content: input_image needs image_url", "input");
            }
            ImagePart img;
            img.url = part["image_url"].get<std::string>();
            msg.images.push_back(std::move(img));
        } else if (type == "input_text" || type == "output_text" || type == "text") {
            if (!part.contains("text") || !part["text"].is_string()) {
                return invalid(where + ".content: " + type + " part is missing 'text'", "input");
            }
        } else if (type == "input_file" || type == "input_video") {
            return unsupported("'" + type + "' is not served: images are the only media input "
                               "(docs/campaigns/feyntalk.md Scope)", "input");
        } else {
            return unsupported("'" + type + "' content part is not served in 1.3.0", "input");
        }
        if (!joined.empty()) joined += "\n";
        if (type == "input_image") joined += kMediaMarker;
        else joined += part["text"].get<std::string>();
    }
    msg.content = std::move(joined);
    return std::nullopt;
}

// One input item. `index` names the position in the error's message.
std::optional<HttpResult> read_item(const json& item, ResponseRequest& req, size_t index) {
    const std::string where = log::format("input[%zu]", index);
    if (!item.is_object()) return invalid(where + " must be an object", "input");
    if (!item.contains("type") || !item["type"].is_string()) {
        return invalid(where + ".type is required", "input");
    }
    const std::string type = item["type"].get<std::string>();

    if (type == "message") {
        if (!item.contains("role") || !item["role"].is_string()) {
            return invalid(where + ".role is required", "input");
        }
        ChatMessage msg;
        msg.role = item["role"].get<std::string>();
        if (msg.role != "user" && msg.role != "system" && msg.role != "developer" &&
            msg.role != "assistant") {
            return invalid(where + ".role '" + msg.role + "' is not a message role", "input");
        }
        if (item.contains("content")) {
            if (auto err = read_message_content(item["content"], msg, where)) return err;
        }
        // `phase` (2026-04-24) is accepted on an assistant message and ignored:
        // the template knows no phase, and the output carries none (optional
        // field). Review R2 of the campaign moved it out of the skip list.
        req.chat.messages.push_back(std::move(msg));
        return std::nullopt;
    }

    if (type == "function_call") {
        if (!item.contains("call_id") || !item["call_id"].is_string() ||
            item["call_id"].get<std::string>().empty()) {
            return invalid(where + ".call_id is required", "input");
        }
        if (!item.contains("name") || !item["name"].is_string() ||
            item["name"].get<std::string>().empty()) {
            return invalid(where + ".name is required", "input");
        }
        if (!item.contains("arguments") || !item["arguments"].is_string()) {
            return invalid(where + ".arguments must be a JSON string", "input");
        }
        ChatMessage msg;
        msg.role = "assistant";
        ToolCall call;
        call.id        = item["call_id"].get<std::string>();
        call.name      = item["name"].get<std::string>();
        call.arguments = item["arguments"].get<std::string>();
        msg.tool_calls.push_back(std::move(call));
        req.chat.messages.push_back(std::move(msg));
        return std::nullopt;
    }

    if (type == "function_call_output") {
        if (!item.contains("call_id") || !item["call_id"].is_string() ||
            item["call_id"].get<std::string>().empty()) {
            return invalid(where + ".call_id is required", "input");
        }
        if (!item.contains("output")) return invalid(where + ".output is required", "input");
        ChatMessage  msg;
        msg.role         = "tool";
        msg.tool_call_id = item["call_id"].get<std::string>();
        const json& out  = item["output"];
        if (out.is_string()) {
            msg.content = out.get<std::string>();
        } else if (out.is_array()) {
            std::string joined;
            for (const json& part : out) {
                if (!part.is_object() || !part.contains("type") || !part["type"].is_string()) {
                    return invalid(where + ".output parts must be typed objects", "input");
                }
                const std::string ptype = part["type"].get<std::string>();
                if (ptype != "input_text") {
                    return unsupported("function_call_output part '" + ptype +
                                       "' is not served: text only", "input");
                }
                if (!part.contains("text") || !part["text"].is_string()) {
                    return invalid(where + ".output: input_text part is missing 'text'", "input");
                }
                if (!joined.empty()) joined += "\n";
                joined += part["text"].get<std::string>();
            }
            msg.content = std::move(joined);
        } else {
            return invalid(where + ".output must be a string or an array", "input");
        }
        req.chat.messages.push_back(std::move(msg));
        return std::nullopt;
    }

    if (type == "reasoning") {
        // Accepted and kept for the store; never rendered into the prompt
        // (parity with /v1/chat/completions, which replays no reasoning).
        req.reasoning_items.push_back(item);
        return std::nullopt;
    }

    // item_reference and compaction are spec item types with no 1.3.0
    // machinery: the store has no item index and there is no compaction pass.
    if (type == "item_reference" || type == "compaction") {
        return unsupported("'" + type + "' input items are not served in 1.3.0 "
                           "(docs/campaigns/feyntalk.md Scope)", "input");
    }
    return invalid("input item type '" + type + "' is not a known item type", "input");
}

std::optional<HttpResult> read_tools(const json& v, ResponseRequest& req) {
    if (v.is_null()) return std::nullopt;
    if (!v.is_array()) return invalid("tools must be an array or null", "tools");
    size_t i = 0;
    for (const json& t : v) {
        if (!t.is_object()) return invalid(log::format("tools[%zu] must be an object", i), "tools");
        if (!t.contains("type") || !t["type"].is_string()) {
            return invalid(log::format("tools[%zu].type is required", i), "tools");
        }
        if (t["type"].get<std::string>() != "function") {
            return unsupported(log::format("tools[%zu].type '%s' is not served: arcint runs no "
                                           "hosted tools", i, t["type"].get<std::string>().c_str()),
                               "tools");
        }
        if (!t.contains("name") || !t["name"].is_string() ||
            t["name"].get<std::string>().empty()) {
            return invalid(log::format("tools[%zu].name is required", i), "tools");
        }
        ToolSpec spec;
        spec.name = t["name"].get<std::string>();
        if (t.contains("description") && t["description"].is_string()) {
            spec.description = t["description"].get<std::string>();
        }
        if (t.contains("parameters") && t["parameters"].is_object()) {
            spec.parameters = t["parameters"];
        } else {
            // The spec allows parameters null (EmptyModelParam is the empty
            // object); an absent schema is the empty one, not a refusal.
            spec.parameters = json::object();
        }
        // `strict` is accepted and ignored: the tool grammar is the operator's
        // separate opt-in (--tool-grammar), not a request-time enforcement.
        json echo = {{"type", "function"}, {"name", spec.name},
                     {"description", t.contains("description") && t["description"].is_string()
                                         ? t["description"] : json(nullptr)},
                     {"parameters", spec.parameters},
                     {"strict", t.contains("strict") && t["strict"].is_boolean()
                                    ? t["strict"] : json(nullptr)}};
        req.chat.tools.push_back(std::move(spec));
        req.tools.push_back(std::move(echo));
        ++i;
    }
    return std::nullopt;
}

std::optional<HttpResult> read_tool_choice(const json& v, ResponseRequest& req) {
    if (v.is_null()) return std::nullopt;
    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        if (s != "auto" && s != "required" && s != "none") {
            return invalid("tool_choice must be 'auto', 'required', 'none', or a function object",
                           "tool_choice");
        }
        req.chat.tool_choice = s;
        req.tool_choice      = s;
        return std::nullopt;
    }
    if (!v.is_object()) return invalid("tool_choice must be a string or an object", "tool_choice");
    if (!v.contains("type") || !v["type"].is_string()) {
        return invalid("tool_choice.type is required", "tool_choice");
    }
    if (v["type"].get<std::string>() == "allowed_tools") {
        return unsupported("tool_choice 'allowed_tools' is not served in 1.3.0 "
                           "(docs/campaigns/feyntalk.md Scope)", "tool_choice");
    }
    if (v["type"].get<std::string>() != "function") {
        return invalid("tool_choice.type must be 'function'", "tool_choice");
    }
    if (!v.contains("name") || !v["name"].is_string() || v["name"].get<std::string>().empty()) {
        return invalid("tool_choice function needs a name", "tool_choice");
    }
    // arcint's internal form is the chat shape: required plus the named
    // function, which the grammar keeps lazy (clean-turns deviation, unchanged).
    req.chat.tool_choice          = "required";
    req.chat.tool_choice_function = v["name"].get<std::string>();
    req.tool_choice               = {{"type", "function"}, {"name", req.chat.tool_choice_function}};
    return std::nullopt;
}

std::optional<HttpResult> read_text(const json& v, ResponseRequest& req) {
    json out = {{"format", {{"type", "text"}}}};
    if (!v.is_null()) {
        if (!v.is_object()) return invalid("text must be an object or null", "text");
        if (v.contains("format") && !v["format"].is_null()) {
            const json& f = v["format"];
            if (!f.is_object() || !f.contains("type") || !f["type"].is_string()) {
                return invalid("text.format.type is required", "text.format");
            }
            const std::string ft = f["type"].get<std::string>();
            if (ft != "text") {
                return unsupported("text.format '" + ft + "' is not served in 1.3.0 "
                                   "(docs/campaigns/feyntalk.md Scope)", "text.format");
            }
            out["format"] = {{"type", "text"}};
        }
        if (v.contains("verbosity") && !v["verbosity"].is_null()) {
            if (!v["verbosity"].is_string()) return invalid("text.verbosity is not a string", "text.verbosity");
            const std::string vb = v["verbosity"].get<std::string>();
            if (vb != "low" && vb != "medium" && vb != "high") {
                return invalid("text.verbosity must be low, medium or high", "text.verbosity");
            }
            // Accepted and ignored: no verbosity control exists; echoed.
            out["verbosity"] = vb;
        }
    }
    req.text = std::move(out);
    return std::nullopt;
}

std::optional<HttpResult> read_reasoning(const json& v, ResponseRequest& req) {
    if (v.is_null()) return std::nullopt;
    if (!v.is_object()) return invalid("reasoning must be an object or null", "reasoning");
    json out = {{"effort", nullptr}, {"summary", nullptr}};
    if (v.contains("effort") && !v["effort"].is_null()) {
        if (!v["effort"].is_string()) return invalid("reasoning.effort is not a string", "reasoning.effort");
        const std::string effort = v["effort"].get<std::string>();
        if (effort != "none" && effort != "low" && effort != "medium" && effort != "high" &&
            effort != "xhigh" && effort != "minimal") {
            return invalid("reasoning.effort must be none, minimal, low, medium, high or xhigh",
                           "reasoning.effort");
        }
        // The fleet's switch: "none" turns thinking off, every other level on.
        // The Qwen templates know on/off only, so the level is echoed unchanged.
        req.chat.enable_thinking     = effort != "none";
        req.chat.has_enable_thinking = true;
        out["effort"] = effort;
    }
    if (v.contains("summary") && !v["summary"].is_null()) {
        if (!v["summary"].is_string()) return invalid("reasoning.summary is not a string", "reasoning.summary");
        const std::string summary = v["summary"].get<std::string>();
        if (summary != "concise" && summary != "detailed" && summary != "auto") {
            return invalid("reasoning.summary must be concise, detailed or auto", "reasoning.summary");
        }
        // Accepted, echoed null: no summary is produced in 1.3.0.
    }
    req.reasoning = std::move(out);
    return std::nullopt;
}

std::optional<HttpResult> read_arcint(const json& v, ResponseRequest& req) {
    if (!v.is_object()) return invalid("arcint must be an object", "arcint");
    if (!v.contains("v")) return invalid("arcint.v is required when the arcint object is sent",
                                         "arcint.v");
    if (!v["v"].is_number_integer()) return invalid("arcint.v must be an integer", "arcint.v");
    const int64_t version = v["v"].get<int64_t>();
    if (version != 1) {
        return unsupported(log::format("arcint.v %lld is not a schema version this server knows "
                                       "(current 1)", static_cast<long long>(version)),
                           "arcint.v");
    }
    req.arcint_v = 1;
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (it.key() == "v") continue;
        // Everything beyond the version belongs to a later package (token mode,
        // spans, ...); accepting and dropping it would be a silent lie.
        return unsupported("arcint." + it.key() + " is not served in this build",
                           "arcint." + it.key());
    }
    return std::nullopt;
}

}  // namespace

std::optional<HttpResult> parse_response_request(const json& body, ResponseRequest& out) {
    if (!body.is_object()) return invalid("request body must be a JSON object");

    for (auto it = body.begin(); it != body.end(); ++it) {
        if (!known_top_level(it.key())) {
            return invalid("'" + it.key() + "' is not a field of a response request", it.key());
        }
    }

    ResponseRequest req;

    if (body.contains("model") && !body["model"].is_null()) {
        if (!body["model"].is_string()) return invalid("model must be a string or null", "model");
        req.chat.model = body["model"].get<std::string>();
    }

    // The input halves, in order. A string is one user message; an array is
    // the item list. Absent or null is an empty conversation (the spec allows
    // a null input).
    if (body.contains("input") && !body["input"].is_null()) {
        const json& input = body["input"];
        if (input.is_string()) {
            const std::string text = input.get<std::string>();
            if (text.size() > kMaxInputChars) {
                return invalid(log::format("input is %zu characters; the spec's limit is %zu",
                                           text.size(), kMaxInputChars),
                               "input");
            }
            ChatMessage msg;
            msg.role    = "user";
            msg.content = text;
            req.chat.messages.push_back(std::move(msg));
        } else if (input.is_array()) {
            size_t i = 0;
            for (const json& item : input) {
                if (auto err = read_item(item, req, i)) return err;
                ++i;
            }
        } else {
            return invalid("input must be a string, an array, or null", "input");
        }
    }

    // instructions: one system message at the head of messages; echoed in the
    // resource.
    if (body.contains("instructions") && !body["instructions"].is_null()) {
        if (!body["instructions"].is_string()) {
            return invalid("instructions must be a string or null", "instructions");
        }
        req.instructions     = body["instructions"].get<std::string>();
        req.has_instructions = true;
        ChatMessage sys;
        sys.role    = "system";
        sys.content = req.instructions;
        req.chat.messages.insert(req.chat.messages.begin(), std::move(sys));
    }

    if (body.contains("tools")) {
        if (auto err = read_tools(body["tools"], req)) return err;
    }
    if (body.contains("tool_choice")) {
        if (auto err = read_tool_choice(body["tool_choice"], req)) return err;
    }
    if (body.contains("text")) {
        if (auto err = read_text(body["text"], req)) return err;
    }
    if (body.contains("reasoning")) {
        if (auto err = read_reasoning(body["reasoning"], req)) return err;
    }

    // Sampler fields, the same ones the chat route reads.
    auto number = [&](const char* key, std::optional<float>& dst) -> std::optional<HttpResult> {
        if (!body.contains(key) || body[key].is_null()) return std::nullopt;
        if (!body[key].is_number()) return invalid(log::format("%s must be a number", key), key);
        dst = body[key].get<float>();
        return std::nullopt;
    };
    if (auto err = number("temperature", req.chat.sampler.temperature)) return err;
    if (auto err = number("top_p", req.chat.sampler.top_p)) return err;
    if (auto err = number("presence_penalty", req.chat.sampler.presence_penalty)) return err;
    if (auto err = number("frequency_penalty", req.chat.sampler.frequency_penalty)) return err;

    if (body.contains("max_output_tokens") && !body["max_output_tokens"].is_null()) {
        if (!body["max_output_tokens"].is_number_integer()) {
            return invalid("max_output_tokens must be an integer or null", "max_output_tokens");
        }
        const int64_t n = body["max_output_tokens"].get<int64_t>();
        if (n < 16) {
            return invalid("max_output_tokens must be at least 16 (the spec's minimum)",
                           "max_output_tokens");
        }
        req.chat.sampler.max_tokens = static_cast<int>(n);
        req.max_output_tokens       = static_cast<int>(n);
    }

    if (body.contains("parallel_tool_calls") && !body["parallel_tool_calls"].is_null()) {
        if (!body["parallel_tool_calls"].is_boolean()) {
            return invalid("parallel_tool_calls must be a boolean or null", "parallel_tool_calls");
        }
        req.chat.parallel_tool_calls = body["parallel_tool_calls"].get<bool>();
    }

    // stream is honoured in the mapping but the SSE writer is W2: a streaming
    // request is refused loudly until it exists, never answered as a JSON body.
    if (body.contains("stream") && !body["stream"].is_null()) {
        if (!body["stream"].is_boolean()) return invalid("stream must be a boolean", "stream");
        if (body["stream"].get<bool>()) {
            return unsupported("stream is not served yet: POST with stream=false for the complete "
                               "ResponseResource (the SSE writer is the next package)", "stream");
        }
    }
    if (body.contains("stream_options") && !body["stream_options"].is_null()) {
        const json& so = body["stream_options"];
        if (!so.is_object()) return invalid("stream_options must be an object or null", "stream_options");
        for (auto it = so.begin(); it != so.end(); ++it) {
            if (it.key() != "include_obfuscation") {
                return invalid("stream_options." + it.key() + " is not a field", "stream_options." + it.key());
            }
            if (!it.value().is_null() && !it.value().is_boolean()) {
                return invalid("stream_options.include_obfuscation must be a boolean",
                               "stream_options.include_obfuscation");
            }
            // Accepted and ignored: arcint emits no obfuscation.
        }
    }

    if (body.contains("store") && !body["store"].is_null()) {
        if (!body["store"].is_boolean()) return invalid("store must be a boolean", "store");
        req.store = body["store"].get<bool>();
    }
    if (body.contains("background") && !body["background"].is_null()) {
        if (!body["background"].is_boolean()) return invalid("background must be a boolean", "background");
        if (body["background"].get<bool>()) {
            return unsupported("background mode is not served in 1.3.0: no job layer exists "
                               "(docs/campaigns/feyntalk.md Scope)", "background");
        }
    }
    if (body.contains("service_tier") && !body["service_tier"].is_null()) {
        if (!body["service_tier"].is_string()) return invalid("service_tier must be a string", "service_tier");
        const std::string tier = body["service_tier"].get<std::string>();
        if (tier != "auto" && tier != "default" && tier != "flex" && tier != "priority") {
            return invalid("service_tier must be auto, default, flex or priority", "service_tier");
        }
        if (tier != "auto" && tier != "default") {
            return unsupported("service_tier '" + tier + "' is not served: only auto and default "
                               "exist here", "service_tier");
        }
        req.service_tier = tier;
    }
    if (body.contains("metadata") && !body["metadata"].is_null()) {
        const json& md = body["metadata"];
        if (!md.is_object()) return invalid("metadata must be an object or null", "metadata");
        if (md.size() > 16) return invalid("metadata carries more than 16 pairs", "metadata");
        for (auto it = md.begin(); it != md.end(); ++it) {
            if (it.key().size() > 64) return invalid("metadata key is longer than 64 characters", "metadata");
            if (!it.value().is_string() || it.value().get<std::string>().size() > 512) {
                return invalid("metadata values must be strings of at most 512 characters", "metadata");
            }
        }
        req.metadata = md;
    }
    auto short_string = [&](const char* key, std::string& dst, bool& has) -> std::optional<HttpResult> {
        if (!body.contains(key) || body[key].is_null()) return std::nullopt;
        if (!body[key].is_string()) return invalid(log::format("%s must be a string or null", key), key);
        dst = body[key].get<std::string>();
        if (dst.size() > 64) return invalid(log::format("%s is longer than 64 characters", key), key);
        has = true;
        return std::nullopt;
    };
    if (auto err = short_string("safety_identifier", req.safety_identifier,
                                req.has_safety_identifier))
        return err;
    if (auto err = short_string("prompt_cache_key", req.prompt_cache_key, req.has_prompt_cache_key))
        return err;

    if (body.contains("truncation") && !body["truncation"].is_null()) {
        if (!body["truncation"].is_string()) return invalid("truncation must be a string", "truncation");
        const std::string tr = body["truncation"].get<std::string>();
        if (tr != "auto" && tr != "disabled") {
            return invalid("truncation must be auto or disabled", "truncation");
        }
        // Both mean no truncation here: `auto` MAY truncate, not MUST, and
        // `disabled` MUST fail — the §3.8 400 is the same answer.
        req.truncation = tr;
    }

    if (body.contains("previous_response_id") && !body["previous_response_id"].is_null()) {
        if (!body["previous_response_id"].is_string()) {
            return invalid("previous_response_id must be a string or null", "previous_response_id");
        }
        req.previous_response_id     = body["previous_response_id"].get<std::string>();
        req.has_previous_response_id = true;
    }

    // The unbuilt fields, refused by their own names.
    if (body.contains("include")) {
        return unsupported("include is not served in 1.3.0: no encrypted reasoning and no "
                           "logprobs exist here (docs/campaigns/feyntalk.md Scope)", "include");
    }
    if (body.contains("max_tool_calls")) {
        return unsupported("max_tool_calls is not served in 1.3.0: no cross-call counting exists "
                           "(docs/campaigns/feyntalk.md Scope)", "max_tool_calls");
    }
    if (body.contains("top_logprobs")) {
        return unsupported("top_logprobs is not served in 1.3.0: logprobs are absent from this "
                           "surface (docs/campaigns/feyntalk.md Scope)", "top_logprobs");
    }
    if (body.contains("arcint")) {
        if (auto err = read_arcint(body["arcint"], req)) return err;
    }

    out = std::move(req);
    return std::nullopt;
}

std::optional<HttpResult> prepare_response(const Context& ctx, ResponseRequest req,
                                           PreparedResponse& out) {
    // W1 has no store, so a continuation cannot be resolved. The spec's answer
    // for an unknown id is this 400; accepting and dropping the id would serve
    // a fresh turn while telling the client it continued (W3 adds the lookup).
    if (req.has_previous_response_id) {
        return HttpResult{400, responses_error(
            "previous_response_id '" + req.previous_response_id +
                "' is not known to this server: no response store exists in this build "
                "(docs/campaigns/feyntalk.md W3)",
            "invalid_request", "previous_response_not_found", "previous_response_id")};
    }

    PreparedResponse prep;
    if (auto err = prepare_from_chat(ctx, req.chat, prep.chat)) return err;
    prep.req = std::move(req);
    // The Responses ids are resp-N; prepare_from_chat numbered it as a chat
    // response, so it is renumbered here rather than letting one turn carry two
    // id namespaces.
    prep.chat.id      = next_id("resp");
    prep.chat.created = now_seconds();
    out               = std::move(prep);
    return std::nullopt;
}

namespace {

// The item status the run ended in (the open item is the last one, spec
// §Items): completed for Stop, incomplete otherwise.
const char* item_status(FinishReason reason) {
    return reason == FinishReason::Stop ? "completed" : "incomplete";
}

}  // namespace

json response_resource(const ResponseRequest& req, const PreparedChat& prep,
                       const ChatGeneration& gen) {
    const bool completed = gen.reason == FinishReason::Stop;

    json output = json::array();
    if (!gen.reasoning.empty()) {
        output.push_back(json{{"type", "reasoning"},
                              {"id", next_id("rs")},
                              {"status", item_status(gen.reason)},
                              {"summary", json::array()},
                              {"content",
                               json::array({{{"type", "reasoning_text"},
                                             {"text", gen.reasoning}}})}});
    }
    // A message item for an answer. A turn that produced only tool calls has
    // no message: the markup around the call is not an answer (the chat route
    // reports content null on that turn, not "").
    if (!gen.content.empty() || gen.calls.empty()) {
        output.push_back(json{{"type", "message"},
                              {"id", next_id("msg")},
                              {"role", "assistant"},
                              {"status", item_status(gen.reason)},
                              {"content",
                               json::array({{{"type", "output_text"},
                                             {"annotations", json::array()},
                                             {"text", gen.content}}})}});
    }
    for (size_t i = 0; i < gen.calls.size(); ++i) {
        const ToolCall& c = gen.calls[i];
        output.push_back(json{{"type", "function_call"},
                              {"id", next_id("fc")},
                              {"call_id", c.id.empty() ? log::format("call_%zu", i) : c.id},
                              {"name", c.name},
                              {"arguments", c.arguments},
                              {"status", item_status(gen.reason)}});
    }

    json incomplete = nullptr;
    if (!completed) {
        incomplete = json{{"reason", gen.reason == FinishReason::Length ? "max_output_tokens"
                                      : gen.reason == FinishReason::Loop ? "loop_guard_abort"
                                                                        : "max_output_tokens"}};
    }

    const SamplerParams& sampler = prep.input.sampler;
    json usage = {
        {"input_tokens", gen.stats.prompt_tokens},
        {"output_tokens", gen.stats.completion_tokens},
        {"total_tokens", gen.stats.prompt_tokens + gen.stats.completion_tokens},
        {"input_tokens_details", {{"cached_tokens", gen.stats.cache_hit_tokens}}},
        // 0 in W1: no separate reasoning-token count exists yet (W2 counts it
        // from the split, review R7).
        {"output_tokens_details", {{"reasoning_tokens", 0}}},
    };

    // Every required key of the 2026-04-24 ResponseResource, in the order the
    // document lists them. Zero/null where the spec says so.
    return json{{"id", prep.id},
                {"object", "response"},
                {"created_at", prep.created},
                {"completed_at", completed ? json(now_seconds()) : json(nullptr)},
                {"status", completed ? "completed" : "incomplete"},
                {"incomplete_details", incomplete},
                {"model", prep.model},
                {"previous_response_id", nullptr},   // W1: no chain is served
                {"instructions", req.has_instructions ? json(req.instructions) : json(nullptr)},
                {"output", output},
                {"error", nullptr},
                {"tools", req.tools},
                {"tool_choice", req.tool_choice},
                {"truncation", req.truncation},
                {"parallel_tool_calls", req.chat.parallel_tool_calls},
                {"text", req.text},
                {"top_p", sampler.top_p},
                {"presence_penalty", sampler.presence_penalty},
                {"frequency_penalty", sampler.frequency_penalty},
                {"top_logprobs", 0},
                {"temperature", sampler.temperature},
                {"reasoning", req.reasoning},
                {"usage", usage},
                {"max_output_tokens", req.max_output_tokens > 0 ? json(req.max_output_tokens)
                                                                : json(nullptr)},
                {"max_tool_calls", nullptr},
                {"store", req.store},
                {"background", req.background},
                {"service_tier", req.service_tier},
                {"metadata", req.metadata},
                {"safety_identifier", req.has_safety_identifier ? json(req.safety_identifier)
                                                               : json(nullptr)},
                {"prompt_cache_key", req.has_prompt_cache_key ? json(req.prompt_cache_key)
                                                              : json(nullptr)}};
}

HttpResult run_response(const Context& ctx, const PreparedResponse& prep, int slot) {
    GenerationStats stats;
    std::string     raw;
    std::vector<std::pair<std::string, int>> pieces;
    FinishReason reason = FinishReason::Stop;
    try {
        reason = drive(*ctx.backend, prep.chat.input, slot,
                       [](std::string_view, int, const std::string&) { return true; }, stats, raw,
                       &pieces);
    } catch (const ContextOverflowError& e) {
        log::warn("req", "responses refused: prompt %d tokens with its images, lane %d",
                  e.prompt_tokens, e.n_ctx);
        json body = responses_error(
            log::format("prompt is %d tokens, context is %d: arcint does not truncate or shift "
                        "context; compact the history client-side and retry",
                        e.prompt_tokens, e.n_ctx),
            "invalid_request", "context_length_exceeded", "input");
        body["prompt_tokens"] = e.prompt_tokens;
        body["n_ctx"]         = e.n_ctx;
        return HttpResult{400, std::move(body)};
    } catch (const InputError& e) {
        log::warn("req", "responses refused: %s", e.what());
        return HttpResult{400, responses_error(e.what(), "invalid_request", "invalid_request",
                                               "input")};
    } catch (const std::exception& e) {
        log::error("req", "responses generation failed: %s", e.what());
        return HttpResult{500, responses_error(e.what(), "server_error", "internal_error")};
    }
    log_stats(slot, stats, reason);

    // A stop signal is a 503 an alive client reads; a gone client gets no body.
    if (reason == FinishReason::Abort) {
        if (stats.abort_cause == AbortCause::ClientGone)
            return HttpResult{200, json::object(), true};
        return HttpResult{503, responses_error(
            "the server is stopping; the request was cancelled at its last token or batch boundary",
            "server_error", "server_stopping")};
    }

    ChatGeneration gen;
    gen.stats  = stats;
    gen.reason = reason;
    finish_chat_generation(prep.chat, pieces, raw, gen);
    return HttpResult{200, response_resource(prep.req, prep.chat, gen)};
}

}  // namespace lgc::api
