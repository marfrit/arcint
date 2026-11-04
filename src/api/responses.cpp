#include "api/responses.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "api/error.h"
#include "api/generate.h"
#include "api/metrics.h"
#include "api/request_id.h"
#include "api/responses_state.h"
#include "core/chat.h"
#include "util/log.h"
#include "util/text.h"

// Open Responses 2026-04-24 request mapping (docs/campaigns/feyntalk.md W1).
// The mapping rows and their refusals are the campaign's table; every row that
// 1.3.0 does not build is refused loudly with the spec's own code
// `unsupported_parameter`, never accepted and dropped.
namespace lgc::api {
namespace {

using json = nlohmann::json;

constexpr size_t kMaxInputChars = 10485760;

HttpResult invalid(std::string message, std::string param = {}) {
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

// The reasoning text a reasoning input item carries: its reasoning_text parts,
// joined with a newline (the campaign's parts rule). A null content is the
// spec's input shape and contributes nothing.
std::optional<std::string> reasoning_text_of(const json& item, std::string& out) {
    if (!item.contains("content") || item["content"].is_null()) return std::nullopt;
    const json& content = item["content"];
    if (!content.is_array()) return std::string("reasoning.content must be an array or null");
    std::string joined;
    for (const json& part : content) {
        if (!part.is_object() || !part.contains("type") || !part["type"].is_string())
            return std::string("reasoning.content parts must be typed objects");
        const std::string type = part["type"].get<std::string>();
        if (type != "reasoning_text" && type != "text")
            return std::string("reasoning.content part '" + type + "' is not served: reasoning_text only");
        if (!part.contains("text") || !part["text"].is_string())
            return std::string("reasoning.content: part is missing 'text'");
        if (!joined.empty()) joined += "\n";
        joined += part["text"].get<std::string>();
    }
    out = std::move(joined);
    return std::nullopt;
}

// W6g (docs/campaigns/feyntalk.md D8): the one merge rule for an assistant run.
// A run of [reasoning]? [assistant message]? [function_call]+ is ONE assistant
// ChatMessage. The input mapping and the stored-output reconstruction both use
// this state machine, so the two paths cannot drift.
struct AssistantRun {
    std::vector<ChatMessage>& messages;
    std::string               pending;   // reasoning for the next assistant item
    bool                      open = false;   // the tail message accepts calls

    explicit AssistantRun(std::vector<ChatMessage>& m) : messages(m) {}

    // A reasoning item contributes to the NEXT assistant item and restarts the
    // run: the grammar puts it before the message item, never between calls.
    void add_reasoning(const std::string& text) {
        if (!text.empty()) {
            if (!pending.empty()) pending += "\n";
            pending += text;
        }
        open = false;
    }
    // An assistant message item opens the run the following calls join; it takes
    // the pending reasoning.
    void add_message(ChatMessage msg) {
        msg.reasoning = std::move(pending);
        pending.clear();
        messages.push_back(std::move(msg));
        open = true;
    }
    // A call joins the open run, or opens a new assistant turn carrying the
    // pending reasoning.
    void add_call(ToolCall call) {
        if (open && !messages.empty() && messages.back().role == "assistant") {
            messages.back().tool_calls.push_back(std::move(call));
        } else {
            ChatMessage msg;
            msg.role      = "assistant";
            msg.reasoning = std::move(pending);
            pending.clear();
            msg.tool_calls.push_back(std::move(call));
            messages.push_back(std::move(msg));
            open = true;
        }
    }
    // A boundary item (a user/system/developer message, a function_call_output)
    // or an item this grammar does not handle closes the run.
    void close() {
        open = false;
        pending.clear();
    }
    // Whether the open run already carries `call_id`: the input path's duplicate
    // refusal (the stored path cannot see duplicates; the engine mints the ids).
    bool open_run_has_call(const std::string& call_id) const {
        if (!open || messages.empty() || messages.back().role != "assistant") return false;
        for (const ToolCall& c : messages.back().tool_calls)
            if (c.id == call_id) return true;
        return false;
    }
};

// One input item. `index` names the position in the error's message; `run`
// carries the reasoning pending for the next assistant item and the W6g merge
// state (the pairing rule: a reasoning item pairs with the FOLLOWING assistant
// message or function_call; any other item in between drops it).
std::optional<HttpResult> read_item(const json& item, ResponseRequest& req, size_t index,
                                    AssistantRun& run) {
    const std::string where = log::format("input[%zu]", index);
    if (!item.is_object()) return invalid(where + " must be an object", "input");
    // The OpenAPI document marks `type` required but defaults every message
    // item param to "message"; real clients (the live pi client with
    // provider type openai-responses) omit it and send only a role. A role
    // makes it a message, an item with neither is still a 400.
    std::string type;
    if (!item.contains("type") || item["type"].is_null()) {
        if (!item.contains("role") || !item["role"].is_string())
            return invalid(where + ".type is required (a message item may omit it by its role)",
                           "input");
        type = "message";
    } else if (!item["type"].is_string()) {
        return invalid(where + ".type must be a string", "input");
    } else {
        type = item["type"].get<std::string>();
    }

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
        if (msg.role == "assistant") {
            // Opens a run the following function_calls join and takes the
            // reasoning that precedes it.
            run.add_message(std::move(msg));
        } else {
            // A user/system/developer turn is a boundary: it ends the pairing
            // and the merge run.
            run.close();
            req.chat.messages.push_back(std::move(msg));
        }
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
        ToolCall call;
        call.id        = item["call_id"].get<std::string>();
        call.name      = item["name"].get<std::string>();
        call.arguments = item["arguments"].get<std::string>();
        // A duplicate call_id inside one merged turn makes the
        // function_call_output pairing ambiguous, so it is refused here. The
        // stored-output path carries engine-minted ids and cannot see one.
        if (run.open_run_has_call(call.id)) {
            return invalid(where + ".call_id '" + call.id +
                               "' is duplicated in one assistant turn; the function_call_output "
                               "pairing would be ambiguous",
                           "input");
        }
        // W6g: a call that follows an assistant message (or a previous call in
        // the same run) joins that one message; only the run's first call opens
        // it, and only it takes the pending reasoning.
        run.add_call(std::move(call));
        return std::nullopt;
    }

    if (type == "function_call_output") {
        if (!item.contains("call_id") || !item["call_id"].is_string() ||
            item["call_id"].get<std::string>().empty()) {
            return invalid(where + ".call_id is required", "input");
        }
        if (!item.contains("output")) return invalid(where + ".output is required", "input");
        // A tool output is not an assistant item and is a turn boundary (W6g).
        run.close();
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
        // Kept for the record and paired with the following assistant item; the
        // template decides whether the turn shows it (Qwen3 keeps the turns after
        // the last user message, drops earlier ones).
        std::string text;
        if (auto e = reasoning_text_of(item, text)) return invalid(where + ".content: " + *e, "input");
        // W6g: a reasoning item restarts the run (the grammar puts it before
        // the message item, never between calls): the next function_call opens a
        // new turn carrying this reasoning.
        run.add_reasoning(text);
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
        // The fleet's switch: "none" turns thinking off, every other level on,
        // minimal included, as on the chat route. The level itself reaches the
        // template as the chat route's reasoning_effort does (core/chat.cpp
        // template_context: passed through where the template takes it). The
        // echo below stays the level as sent.
        req.chat.enable_thinking     = effort != "none";
        req.chat.has_enable_thinking = true;
        req.chat.reasoning_effort    = effort;
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

// A hash is "sha256:" and 64 lowercase hex digits — the form the material
// endpoint publishes and the request must echo. Nothing else is accepted; a
// short or upper-case hash would compare unequal for the wrong reason.
bool is_sha256_hash(std::string_view s) {
    if (s.size() != 71 || s.substr(0, 7) != "sha256:") return false;
    for (char c : s.substr(7))
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
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
    bool has_renderer_hash = false;
    bool has_render        = false;
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (it.key() == "v") continue;
        if (it.key() == "token_mode") {
            if (!it.value().is_boolean())
                return invalid("arcint.token_mode must be a boolean", "arcint.token_mode");
            req.token_mode = it.value().get<bool>();
            continue;
        }
        if (it.key() == "timings") {
            if (!it.value().is_boolean())
                return invalid("arcint.timings must be a boolean", "arcint.timings");
            req.timings = it.value().get<bool>();
            continue;
        }
        if (it.key() == "preserve_thinking") {
            if (!it.value().is_boolean())
                return invalid("arcint.preserve_thinking must be a boolean",
                               "arcint.preserve_thinking");
            req.chat.preserve_thinking     = it.value().get<bool>();
            req.chat.has_preserve_thinking = true;
            continue;
        }
        if (it.key() == "renderer_hash") {
            if (!it.value().is_string())
                return invalid("arcint.renderer_hash must be a string", "arcint.renderer_hash");
            const std::string h = it.value().get<std::string>();
            if (!is_sha256_hash(h))
                return invalid("arcint.renderer_hash must be 'sha256:' and 64 lowercase hex digits",
                               "arcint.renderer_hash");
            req.renderer_hash = h;
            has_renderer_hash = true;
            continue;
        }
        if (it.key() == "render") {
            if (!it.value().is_object()) return invalid("arcint.render must be an object",
                                                        "arcint.render");
            const json& r = it.value();
            for (auto rt = r.begin(); rt != r.end(); ++rt) {
                if (rt.key() != "prompt_sha256")
                    return unsupported("arcint.render." + rt.key() + " is not served in this build",
                                       "arcint.render." + rt.key());
                if (!rt.value().is_string())
                    return invalid("arcint.render.prompt_sha256 must be a string",
                                   "arcint.render.prompt_sha256");
                const std::string h = rt.value().get<std::string>();
                if (!is_sha256_hash(h))
                    return invalid("arcint.render.prompt_sha256 must be 'sha256:' and 64 lowercase "
                                   "hex digits", "arcint.render.prompt_sha256");
                req.render_prompt_sha256 = h;
            }
            has_render = req.render_prompt_sha256.empty() ? false : true;
            if (!has_render)
                return invalid("arcint.render.prompt_sha256 is required when arcint.render is sent",
                               "arcint.render.prompt_sha256");
            continue;
        }
        if (it.key() == "input_ids") {
            // D1: the renderer-owned prompt path. The field is part of the
            // schema, so it is not "not a field"; it is refused by its own name
            // until W6f, when the renderer's encoder has passed its corpus.
            if (!it.value().is_array())
                return invalid("arcint.input_ids must be an array of token ids", "arcint.input_ids");
            for (const json& id : it.value())
                if (!id.is_number_integer() || id.get<int64_t>() < 0)
                    return invalid("arcint.input_ids must hold non-negative integer token ids",
                                   "arcint.input_ids");
            return unsupported("arcint.input_ids is accepted by the schema but not served in this "
                               "build: the engine renders the prompt and checks its hash (W6f adds "
                               "the renderer-owned ids)", "arcint.input_ids");
        }
        // Everything beyond the version and the W6a fields belongs to a later
        // package (spans, ...); accepting and dropping it would be a silent lie.
        return unsupported("arcint." + it.key() + " is not served in this build",
                           "arcint." + it.key());
    }
    if (req.token_mode) {
        if (!has_renderer_hash)
            return invalid("arcint.renderer_hash is required with arcint.token_mode: true",
                           "arcint.renderer_hash");
        if (!has_render)
            return invalid("arcint.render.prompt_sha256 is required with arcint.token_mode: true",
                           "arcint.render.prompt_sha256");
    } else if (has_renderer_hash || has_render) {
        // A hash without the mode names no turn to check: refuse rather than
        // store a value nothing reads.
        return invalid("arcint.renderer_hash and arcint.render need arcint.token_mode: true",
                       "arcint.token_mode");
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
            AssistantRun run(req.chat.messages);   // W6g's merge state for the whole input
            size_t       i = 0;
            for (const json& item : input) {
                if (auto err = read_item(item, req, i, run)) return err;
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

    // stream is honoured: true selects the SSE writer, false the complete
    // ResponseResource as one JSON body.
    if (body.contains("stream") && !body["stream"].is_null()) {
        if (!body["stream"].is_boolean()) return invalid("stream must be a boolean", "stream");
        req.chat.stream = body["stream"].get<bool>();
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
        // W3.5: `auto` is the cheap compaction tier (prepare_response drops
        // replayed reasoning blocks until the prompt fits); `disabled` MUST
        // fail on overflow and keeps the §3.8 400.
        req.truncation = tr;
    }

    if (body.contains("previous_response_id") && !body["previous_response_id"].is_null()) {
        if (!body["previous_response_id"].is_string()) {
            return invalid("previous_response_id must be a string or null", "previous_response_id");
        }
        req.previous_response_id     = body["previous_response_id"].get<std::string>();
        req.has_previous_response_id = true;
    }

    // include (2026-04-24): accepted and ignored. arcint serves no encrypted
    // reasoning and no output logprobs, and OpenAI-compatible clients (the live
    // pi client) send it on every reasoning request; a 400 would lock them out.
    // Nothing is added to the output. A non-string entry is still a 400.
    if (body.contains("include")) {
        const json& inc = body["include"];
        if (!inc.is_array()) return invalid("include must be an array", "include");
        for (const json& entry : inc) {
            if (!entry.is_string())
                return invalid("include entries must be strings (the spec's enum values)",
                               "include");
        }
    }
    // The unbuilt fields, refused by their own names.
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
    if (req.token_mode) {
        // Token mode is stream-only: its events ride the SSE writer.
        if (!req.chat.stream)
            return unsupported("token mode requires stream: true (arcint:token rides the SSE "
                               "writer)", "stream");
        // Text-only: the gateway renders a text prompt, and an image would
        // leave the prompt the engine verifies.
        for (const ChatMessage& m : req.chat.messages)
            if (!m.images.empty())
                return unsupported("token mode is text-only: an image part is not served with "
                                   "token mode", "arcint.token_mode");
    }

    out = std::move(req);
    return std::nullopt;
}

namespace {

// prepare_from_chat speaks the chat route's envelope (its lanes' 404 and the
// §3.8 overflow are shared answers). The Responses surface gets the spec's type
// spellings and field paths here, not a second prepare.
HttpResult translate_prepare_error(const HttpResult& err) {
    const json& e = err.body["error"];
    const std::string message = e["message"].is_string() ? e["message"].get<std::string>() : "";
    const std::string code = (e.contains("code") && e["code"].is_string())
                                ? e["code"].get<std::string>()
                                : "invalid_request";
    if (err.status == 404) {
        return HttpResult{404, responses_error(message, "not_found", code,
                                               e["param"].is_string() ? e["param"].get<std::string>()
                                                                      : "model")};
    }
    if (err.status == 400) {
        // Overflow is a §3.8 400 with the numbers at the top level (the
        // campaign's extension keys); a malformed field names `input`.
        const bool overflow = code == "context_length_exceeded";
        json body = responses_error(message, "invalid_request", code, overflow ? "input" : "");
        if (overflow) {
            if (e.contains("prompt_tokens")) body["prompt_tokens"] = e["prompt_tokens"];
            if (e.contains("n_ctx")) body["n_ctx"] = e["n_ctx"];
            // W3.5b N4 / W3.5c R3: the fit numbers reach the client, but only
            // for the fit pass's own 400 (`auto`). `overflow` rides with them, so
            // the disabled/absent 400 keeps today's bytes: it has no top-level
            // `overflow` key, exactly as before W3.5.
            if (e.contains("fit_reserve")) {
                if (e.contains("overflow")) body["overflow"] = e["overflow"];
                body["fit_reserve"] = e["fit_reserve"];
                if (e.contains("reasoning_dropped_blocks"))
                    body["reasoning_dropped_blocks"] = e["reasoning_dropped_blocks"];
            }
        }
        return HttpResult{400, std::move(body)};
    }
    return err;
}

// W6a D1: a token-mode turn's hashes are checked before the slot is taken, so a
// stale gateway cannot drive a prompt the engine would render differently. Both
// refusals are 409 `renderer_mismatch`; the param names which side differed.
HttpResult renderer_conflict(std::string message, std::string param) {
    return HttpResult{409, responses_error(std::move(message), "invalid_request",
                                           "renderer_mismatch", std::move(param))};
}

HttpResult previous_not_found(const std::string& id, const std::string& why) {
    return HttpResult{400, responses_error(
        "previous_response_id '" + id + "' is not usable here: " + why,
        "invalid_request", "previous_response_not_found", "previous_response_id")};
}

// One assistant ChatMessage per spec output item run, in the item order and
// with the same reasoning pairing and the same W6g merge as the input mapping
// (both use AssistantRun), so a continuation renders like a full-history
// request. Reasoning items pair with the following message or function_call;
// reasoning parts join with a newline. A run of [reasoning]? [message]?
// [call]+ is ONE message (content from the message item, all calls in item
// order), exactly as read_item maps the same items. An item this grammar does
// not handle closes the run, the same way a boundary item does on the input
// path, so a corrupt stored item cannot let two calls merge across it.
std::vector<ChatMessage> assistant_turns_of(const nlohmann::json& output_items) {
    std::vector<ChatMessage> out;
    AssistantRun             run(out);
    for (const json& item : output_items) {
        const std::string type = item.value("type", "");
        if (type == "reasoning") {
            std::string text;
            // One rule with the input path: a reasoning item restarts the run
            // and contributes only its text. An empty item -- or a malformed
            // content the stored path cannot refuse -- therefore carries
            // nothing and does NOT erase the reasoning that preceded it, so the
            // call that follows keeps it. (read_item refuses a malformed content
            // outright and calls add_reasoning("") for an empty one.)
            if (reasoning_text_of(item, text)) text.clear();
            run.add_reasoning(text);
        } else if (type == "message") {
            ChatMessage msg;
            msg.role = "assistant";
            // The same content shapes the input mapping accepts: a string, or
            // typed parts joined with a newline. A stored output message
            // carries one output_text part, so this does not move a real
            // continuation's bytes; it keeps the two paths one grammar.
            if (item.contains("content")) {
                const json& content = item["content"];
                if (content.is_string()) {
                    msg.content = content.get<std::string>();
                } else if (content.is_array()) {
                    for (const json& part : content) {
                        if (!part.is_object() || !part.contains("text") || !part["text"].is_string())
                            continue;
                        if (!msg.content.empty()) msg.content += "\n";
                        msg.content += part["text"].get<std::string>();
                    }
                }
            }
            run.add_message(std::move(msg));
        } else if (type == "function_call") {
            ToolCall call;
            call.id        = item.value("call_id", "");
            call.name      = item.value("name", "");
            call.arguments = item.value("arguments", std::string("{}"));
            run.add_call(std::move(call));
        } else {
            run.close();
        }
    }
    return out;
}

}  // namespace

std::optional<HttpResult> prepare_response(const Context& ctx, ResponseRequest req,
                                           PreparedResponse& out, bool evict_on_error) {
    // Token mode: the material first, before any render and before any
    // continuation lookup. A unit without material cannot serve token mode.
    if (req.token_mode) {
        // W6a review findings 1/3: the capability first (no build), then the
        // material, which builds once on first use. A build that failed leaves
        // the unit servable with token mode off, and is a 409 here.
        if (!ctx.backend->renderer_capable()) {
            return renderer_conflict("token mode is not served here: this unit is not renderer "
                                     "capable (GET /props says token_mode: \"off\")",
                                     "arcint.token_mode");
        }
        // W6 follow-up 3/5: the hash accessor builds if needed and returns the
        // cached string; the JSON tree is not kept after serialization.
        const std::string served_hash = ctx.backend->renderer_hash();
        if (served_hash.empty()) {
            return renderer_conflict("token mode is not served here: this unit's renderer "
                                     "material could not be built (the load log has the reason)",
                                     "arcint.token_mode");
        }
        if (served_hash != req.renderer_hash) {
            return renderer_conflict("arcint.renderer_hash does not name the material this unit "
                                     "serves; the gateway must refresh its renderer material",
                                     "arcint.renderer_hash");
        }
    }
    const std::string prev_id =
        req.has_previous_response_id ? req.previous_response_id : std::string();
    std::shared_ptr<StoredResponse> stored;
    if (req.has_previous_response_id) {
        if (!ctx.responses)
            return previous_not_found(prev_id, "this process keeps no response store");
        // Lane-scoped: resolve the request's lane first, so a name that is a 404
        // stays a 404 and an entry from another lane is not found.
        int lane = -1;
        if (auto err = resolve_lane(ctx, req.chat.model, lane))
            return translate_prepare_error(*err);
        stored = ctx.responses->get(prev_id, lane);
        if (!stored) return previous_not_found(prev_id, "unknown id, another lane, or evicted");
        // The prefix must still be reproducible by the current renderer. A
        // mismatch means the continuation would not share the stored prompt.
        const std::string rendered = ctx.backend->render_chat(stored->chat);
        if (sha256_hex(rendered) != stored->prefix_hash) {
            // A 4xx on a continuation evicts the entry it referenced -- unless
            // the caller is only counting (/tokenize), where nothing was served.
            if (evict_on_error) evict_continuation(ctx, prev_id);
            return previous_not_found(prev_id, "the rendered prefix no longer matches the stored one");
        }

        // stored input messages + the stored assistant turn + the new items.
        // The new request's tools, tool_choice, sampler and thinking settings
        // are the turn's own; the template is the one process.
        //
        // Instructions (W3.1): an OpenAI-compatible client resends its full
        // system prompt as `instructions` on every turn, and the Qwen-family
        // templates reject a system message that is not first. The spec gives no
        // replacement rule; we read `instructions` as this turn's guidance, so a
        // request that carries them REPLACES the stored head system message in
        // place, and a request that carries none KEEPS the stored head (absent
        // means "no new guidance", not "drop the stored system prompt"). With the
        // same text resent, the head and the stored half render to the same bytes
        // as before, which is what slot affinity's prefix needs.
        std::vector<ChatMessage> new_items = req.chat.messages;
        ChatMessage              new_head;
        bool                     has_new_head = false;
        if (req.has_instructions && !new_items.empty() && new_items.front().role == "system" &&
            new_items.front().content == req.instructions) {
            new_head = std::move(new_items.front());
            new_items.erase(new_items.begin());
            has_new_head = true;
        }
        ChatRequest merged = req.chat;
        merged.messages.clear();
        merged.messages = stored->chat.messages;
        if (has_new_head) {
            if (!merged.messages.empty() && merged.messages.front().role == "system")
                merged.messages.front() = std::move(new_head);
            else
                merged.messages.insert(merged.messages.begin(), std::move(new_head));
        }
        for (ChatMessage& m : assistant_turns_of(stored->output_items))
            merged.messages.push_back(std::move(m));
        for (ChatMessage& m : new_items) merged.messages.push_back(std::move(m));
        req.chat = std::move(merged);
    }

    PreparedResponse prep;
    // Inherited from the caller's struct: the local replaces `out` at the end of
    // this function, and the request id has to survive that. The route lambda
    // set it before prepare ran, so the prepare-time refusals can log it.
    prep.chat.request_id = out.chat.request_id;
    prep.affinity_slot = stored ? stored->slot : -1;
    // feyntalk W3.5: `truncation: "auto"` is the cheap compaction tier,
    // reasoning first. The fit reserve is the operator's --think-ctx-reserve
    // when set; otherwise a small default of 256 tokens of answer room, or the
    // request's own max_output_tokens when that is smaller. A larger
    // max_output_tokens does not enlarge the reserve: the length stop still
    // bounds the answer, and reserving a client's 16k ask would drop nearly
    // every reasoning block instead of failing. `disabled` and an absent field
    // never build a PromptFit and keep today's §3.8 400.
    PromptFit   fit;
    PromptFit*  fitp = nullptr;
    if (req.truncation == "auto") {
        const int operator_reserve = ctx.backend->status().sampler_defaults.think_ctx_reserve;
        fit.drop_reasoning = true;
        fit.reserve        = operator_reserve > 0
                                 ? operator_reserve
                                 : (req.max_output_tokens > 0
                                        ? std::min(req.max_output_tokens, 256)
                                        : 256);
        fitp = &fit;
    }
    if (auto err = prepare_from_chat(ctx, req.chat, prep.chat, fitp)) {
        // A continuation the server answers 4xx is a broken chain: evict the
        // referenced entry so a retry cannot silently continue from it. The
        // count path does not evict (see the header).
        if (evict_on_error) evict_continuation(ctx, prev_id);
        return translate_prepare_error(*err);
    }
    if (req.token_mode) {
        // W6b: the sampler's span sink, one vector per token-mode turn. The
        // sampler appends at its own counter increments; the SSE writer drains
        // it right after the emit that follows. Null outside token mode, so the
        // off path allocates nothing.
        prep.chat.input.sampler.span_events = std::make_shared<std::vector<SamplerSpanEvent>>();
        // The engine renders the prompt it serves; the gateway's render hash is
        // over that same full history. A mismatch is a 409 before the slot, not
        // a cache miss later: the prompt is never echoed, only the hash.
        const std::string served = "sha256:" + sha256_hex(prep.chat.input.prompt);
        if (served != req.render_prompt_sha256) {
            // A continuation's stored entry is NOT evicted here: nothing of it
            // was served, no token left the engine, and the client can retry the
            // same previous_response_id once its render is fixed. Only a
            // prepare 4xx (the stored prefix itself is broken) evicts.
            return renderer_conflict("arcint.render.prompt_sha256 does not match the prompt this "
                                     "unit renders for the turn; the gateway's render and the "
                                     "engine's differ", "arcint.render.prompt_sha256");
        }
    }
    // One line for the quiet path too: a truncated turn's numbers, so an
    // operator reading the log knows what was served and what was dropped.
    if (prep.chat.truncation.blocks_dropped > 0) {
        log::warn("req", "truncation auto: dropped %d reasoning blocks, prompt %d -> %d tokens "
                         "(reserve %d, n_ctx %d, lane %d)%s",
                  prep.chat.truncation.blocks_dropped, prep.chat.truncation.tokens_before,
                  prep.chat.truncation.tokens_after, prep.chat.truncation.reserve,
                  lane_n_ctx(ctx, prep.chat.lane), prep.chat.lane,
                  rid_suffix(prep.chat.request_id).c_str());
    }
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
                       const ResponseParts& parts) {
    const SamplerParams& sampler = prep.input.sampler;
    // Every required key of the 2026-04-24 ResponseResource, in the order the
    // document lists them. Zero/null where the spec says so.
    json body = {{"id", prep.id},
                {"object", "response"},
                {"created_at", prep.created},
                {"completed_at", parts.completed ? json(now_seconds()) : json(nullptr)},
                {"status", parts.status},
                {"incomplete_details", parts.incomplete_details},
                {"model", prep.model},
                {"previous_response_id", req.has_previous_response_id
                                            ? json(req.previous_response_id)
                                            : json(nullptr)},
                {"instructions", req.has_instructions ? json(req.instructions) : json(nullptr)},
                {"output", parts.output},
                {"error", parts.error},
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
                {"usage", parts.usage},
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
    // Review F2: the loop-guard counters as an arcint key when one moved, on
    // the same condition as the chat route's loop_guard object.
    if (!parts.arcint_ext.is_null()) body["arcint"] = parts.arcint_ext;
    return body;
}

json response_resource(const ResponseRequest& req, const PreparedChat& prep,
                       const ChatGeneration& gen) {
    const bool completed = gen.reason == FinishReason::Stop;

    json output = json::array();
    if (!gen.reasoning.empty()) {
        output.push_back(json{{"type", "reasoning"},
                              {"id", next_id("rs")},
                              {"status", gen.reasoning_closed ? "completed" : item_status(gen.reason)},
                              {"summary", json::array()},
                              {"content",
                               json::array({{{"type", "reasoning_text"},
                                             {"text", gen.reasoning}}})}});
    }
    // A message item for a real answer. A turn that produced only tool calls
    // or only reasoning has no message: the markup around the call is not an
    // answer (the chat route reports content null on that turn, not "").
    const bool has_answer = !text::trim(gen.content).empty();
    if (has_answer || (gen.calls.empty() && gen.reasoning.empty())) {
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

    ResponseParts parts;
    parts.completed = completed;
    parts.status    = completed ? "completed" : "incomplete";
    if (!completed) {
        // Length is the token budget, Loop the guard, an Abort is a cancel
        // (StopSignal never reaches the resource builder: it answers 503).
        parts.incomplete_details =
            json{{"reason", gen.reason == FinishReason::Length ? "max_output_tokens"
                            : gen.reason == FinishReason::Loop ? "loop_guard_abort"
                            : gen.reason == FinishReason::Abort ? "cancelled"
                                                              : "max_output_tokens"}};
    }
    json ext = json{{"v", 1}};
    // feyntalk W3.5: the fit pass's numbers, on the same extension object the
    // loop guard's counters use.
    if (prep.truncation.blocks_dropped > 0) ext["truncated"] = truncation_json(prep.truncation);
    if (loop_guard_moved(gen.stats)) ext["loop_guard"] = loop_guard_json(gen.stats);
    // arcint.timings: the per-response timing object, only when the request
    // asked; the extension namespace is the spec-typed surface's own.
    if (req.timings) ext["timings"] = timings_json(gen.stats);
    if (ext.size() > 1) parts.arcint_ext = std::move(ext);
    parts.output = std::move(output);
    parts.usage  = json{{"input_tokens", gen.stats.prompt_tokens},
                        {"output_tokens", gen.stats.completion_tokens},
                        {"total_tokens", gen.stats.prompt_tokens + gen.stats.completion_tokens},
                        {"input_tokens_details", {{"cached_tokens", gen.stats.cache_hit_tokens}}},
                        // R7: from the split where it is id-exact, 0 where it is
                        // textual (see ChatGeneration::reasoning_tokens).
                        {"output_tokens_details", {{"reasoning_tokens", gen.reasoning_tokens}}}};
    return response_resource(req, prep, parts);
}

HttpResult responses_busy(const HttpResult& chat_refusal) {
    const json& e = chat_refusal.body["error"];
    const std::string message = e["message"].is_string() ? e["message"].get<std::string>() : "";
    json body = responses_error(message, "too_many_requests", "all_lanes_busy");
    // The diagnostics are numbers, not a mood: a client told "busy" without
    // slots/free cannot decide whether to retry now or later (F1, W1 review).
    if (chat_refusal.body.contains("slots")) body["slots"] = chat_refusal.body["slots"];
    if (chat_refusal.body.contains("lanes")) body["lanes"] = chat_refusal.body["lanes"];
    return HttpResult{429, std::move(body)};
}

HttpResult run_response(const Context& ctx, PreparedResponse& prep, int slot) {
    GenerationStats stats;
    std::string     raw;
    std::vector<std::pair<std::string, int>> pieces;
    FinishReason reason = FinishReason::Stop;
    try {
        reason = drive(*ctx.backend, prep.chat.input, slot,
                       [](std::string_view, int, const std::string&) { return true; }, stats, raw,
                       &pieces);
    } catch (const ContextOverflowError& e) {
        if (prep.req.has_previous_response_id)
            evict_continuation(ctx, prep.req.previous_response_id);
        log::warn("req", "responses refused: prompt %d tokens with its images, lane %d%s",
                  e.prompt_tokens, e.n_ctx, rid_suffix(prep.chat.request_id).c_str());
        json body = responses_error(
            log::format("prompt is %d tokens, context is %d: arcint does not truncate or shift "
                        "context; compact the history client-side and retry",
                        e.prompt_tokens, e.n_ctx),
            "invalid_request", "context_length_exceeded", "input");
        body["prompt_tokens"] = e.prompt_tokens;
        body["n_ctx"]         = e.n_ctx;
        return HttpResult{400, std::move(body)};
    } catch (const InputError& e) {
        if (prep.req.has_previous_response_id)
            evict_continuation(ctx, prep.req.previous_response_id);
        log::warn("req", "responses refused: %s%s", e.what(),
                  rid_suffix(prep.chat.request_id).c_str());
        return HttpResult{400, responses_error(e.what(), "invalid_request", "invalid_request",
                                               "input")};
    } catch (const std::exception& e) {
        if (prep.req.has_previous_response_id)
            evict_continuation(ctx, prep.req.previous_response_id);
        log::error("req", "responses generation failed: %s%s", e.what(),
                   rid_suffix(prep.chat.request_id).c_str());
        return HttpResult{500, responses_error(e.what(), "server_error", "internal_error")};
    }
    log_stats(slot, stats, reason, prep.chat.request_id);
    if (ctx.metrics != nullptr) ctx.metrics->observe_generation(stats);

    // A cancel-by-id is a 200 with the incomplete resource (the campaign's
    // Cancel section); a stop signal is a 503 an alive client reads; a gone
    // client gets no body.
    if (reason == FinishReason::Abort && stats.abort_cause != AbortCause::Cancelled) {
        if (stats.abort_cause == AbortCause::ClientGone)
            return HttpResult{200, json::object(), true};
        if (prep.req.has_previous_response_id)
            evict_continuation(ctx, prep.req.previous_response_id);
        return HttpResult{503, responses_error(
            "the server is stopping; the request was cancelled at its last token or batch boundary",
            "server_error", "server_stopping")};
    }

    ChatGeneration gen;
    gen.stats  = stats;
    gen.reason = reason;
    finish_chat_generation(prep.chat, pieces, raw, gen);
    json body = response_resource(prep.req, prep.chat, gen);
    // store=true retains the turn (default true; review R4's entries=0 makes
    // this a no-op inside the store). store=false keeps nothing.
    remember(ctx, prep, body, slot);
    return HttpResult{200, std::move(body)};
}

}  // namespace lgc::api
