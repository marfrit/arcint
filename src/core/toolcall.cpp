#include "core/toolcall.h"

#include <nlohmann/json.hpp>

#include "util/log.h"
#include "util/text.h"

namespace lgc {
namespace {

using json = nlohmann::json;

constexpr std::string_view kOpen      = "<tool_call>";
constexpr std::string_view kClose     = "</tool_call>";
constexpr std::string_view kFnOpen    = "<function=";
constexpr std::string_view kFnClose   = "</function>";
constexpr std::string_view kParamOpen = "<parameter=";
constexpr std::string_view kParamEnd  = "</parameter>";

// The coder template wraps every value in newlines. Strip exactly one on each
// side — a value that legitimately starts with a blank line keeps it.
std::string_view strip_one_newline(std::string_view v) {
    if (!v.empty() && v.front() == '\n') v.remove_prefix(1);
    if (!v.empty() && v.back() == '\n') v.remove_suffix(1);
    return v;
}

json coerce(std::string_view raw, const std::string& type) {
    const std::string s(raw);

    if (type == "string" || type.empty()) return s;

    if (type == "integer" || type == "number") {
        try {
            json parsed = json::parse(s);
            if (parsed.is_number()) return parsed;
        } catch (const json::exception&) {
        }
        return s;
    }
    if (type == "boolean") {
        const std::string_view t = text::trim(s);
        if (text::iequals(t, "true")) return true;
        if (text::iequals(t, "false")) return false;
        return s;
    }
    if (type == "object" || type == "array") {
        try {
            json parsed = json::parse(s);
            if ((type == "object" && parsed.is_object()) ||
                (type == "array" && parsed.is_array())) {
                return parsed;
            }
        } catch (const json::exception&) {
        }
        return s;
    }
    return s;
}

// <function=NAME> ... </function> with <parameter=KEY>VALUE</parameter> inside.
bool parse_xml_body(std::string_view body, const ToolSchemas* schemas, ToolCall& out) {
    const size_t fn = body.find(kFnOpen);
    if (fn == std::string_view::npos) return false;

    const size_t name_start = fn + kFnOpen.size();
    const size_t name_end   = body.find('>', name_start);
    if (name_end == std::string_view::npos) return false;

    out.name = std::string(text::trim(body.substr(name_start, name_end - name_start)));
    if (out.name.empty()) return false;

    size_t body_end = body.find(kFnClose, name_end);
    if (body_end == std::string_view::npos) body_end = body.size();

    const ParamTypes* types = nullptr;
    if (schemas != nullptr) {
        auto it = schemas->find(out.name);
        if (it != schemas->end()) types = &it->second;
    }

    json args = json::object();
    size_t cursor = name_end + 1;
    while (cursor < body_end) {
        const size_t p = body.find(kParamOpen, cursor);
        if (p == std::string_view::npos || p >= body_end) break;

        const size_t key_start = p + kParamOpen.size();
        const size_t key_end   = body.find('>', key_start);
        if (key_end == std::string_view::npos || key_end >= body_end) break;

        const std::string key(text::trim(body.substr(key_start, key_end - key_start)));

        const size_t val_start = key_end + 1;
        size_t       val_end   = body.find(kParamEnd, val_start);
        bool         closed    = val_end != std::string_view::npos && val_end <= body_end;
        if (!closed) val_end = body_end;

        const std::string_view raw =
            strip_one_newline(body.substr(val_start, val_end - val_start));

        std::string type;
        if (types != nullptr) {
            auto it = types->find(key);
            if (it != types->end()) type = it->second;
        }
        if (!key.empty()) args[key] = coerce(raw, type);

        cursor = closed ? val_end + kParamEnd.size() : body_end;
    }

    out.arguments = args.dump();
    return true;
}

// {"name": "f", "arguments": {...}} — `arguments` may also arrive as a string.
bool parse_json_body(std::string_view body, ToolCall& out) {
    json parsed;
    try {
        parsed = json::parse(body);
    } catch (const json::exception&) {
        return false;
    }
    if (!parsed.is_object()) return false;

    if (parsed.contains("name") && parsed["name"].is_string()) {
        out.name = parsed["name"].get<std::string>();
    } else {
        return false;
    }

    if (parsed.contains("arguments")) {
        const json& a = parsed["arguments"];
        out.arguments = a.is_string() ? a.get<std::string>() : a.dump();
    } else {
        out.arguments = "{}";
    }
    return true;
}

}  // namespace

ToolCallParse parse_qwen_tool_calls(std::string_view text, const ToolSchemas* schemas) {
    ToolCallParse result;

    size_t cursor = 0;
    while (cursor <= text.size()) {
        const size_t open = text.find(kOpen, cursor);
        if (open == std::string_view::npos) {
            result.content.append(text.substr(cursor));
            break;
        }

        result.content.append(text.substr(cursor, open - cursor));

        const size_t body_start = open + kOpen.size();
        const size_t close      = text.find(kClose, body_start);
        if (close == std::string_view::npos) {
            // Unterminated block: hand the raw remainder back rather than
            // swallowing what the model produced.
            result.truncated = true;
            result.content.append(text.substr(open));
            break;
        }

        const std::string_view body =
            text::trim(text.substr(body_start, close - body_start));

        ToolCall call;
        call.id = log::format("call_%zu", result.calls.size());

        const bool parsed = !body.empty() && body.front() == '{'
                                ? parse_json_body(body, call)
                                : parse_xml_body(body, schemas, call);

        if (parsed) {
            result.calls.push_back(std::move(call));
        } else {
            // Unrecognised body. Keeping the block verbatim in the content is
            // the lesser evil: the caller can see what arrived.
            result.content.append(text.substr(open, close + kClose.size() - open));
        }

        cursor = close + kClose.size();
    }

    return result;
}

}  // namespace lgc
