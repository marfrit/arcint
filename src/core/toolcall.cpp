#include "core/toolcall.h"

#include <nlohmann/json.hpp>

#include "core/markers.h"
#include "util/log.h"
#include "util/text.h"

namespace lgc {
constexpr std::string_view kOpen      = "<tool_call>";
constexpr std::string_view kClose     = "</tool_call>";
constexpr std::string_view kFnOpen    = "<function=";
constexpr std::string_view kFnClose   = "</function>";
constexpr std::string_view kParamOpen = "<parameter=";
constexpr std::string_view kParamEnd  = "</parameter>";
namespace {

using json = nlohmann::json;

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

// ---- Gemma 4's call form ---------------------------------------------------
//   <open>call:NAME{key:<str>value<str>,n:1,o:{a:true},l:[1,2]}<close>
// A recursive-descent reader over the text between the markers.

constexpr int kMaxDepth = 32;

struct Cursor {
    std::string_view s;
    size_t           i = 0;
    bool done() const { return i >= s.size(); }
    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
    }
    bool eat(std::string_view tok) {
        if (s.substr(i, tok.size()) != tok) return false;
        i += tok.size();
        return true;
    }
};

bool read_value(Cursor& c, json& out, int depth, const std::string* type_hint);

// A bare literal: runs to the next separator at this level.
json bare_value(std::string_view tok, const std::string* type_hint) {
    tok = text::trim(tok);
    const std::string raw(tok);
    if (type_hint != nullptr && *type_hint == "string") return raw;
    if (tok == "true") return true;
    if (tok == "false") return false;
    if (tok == "null") return nullptr;
    try {
        json parsed = json::parse(raw);
        if (parsed.is_number()) return parsed;
    } catch (const json::exception&) {
    }
    return raw;
}

bool read_key(Cursor& c, std::string& key) {
    c.skip_ws();
    if (c.eat(markers::kGemmaStrDelim)) {
        const size_t end = c.s.find(markers::kGemmaStrDelim, c.i);
        if (end == std::string_view::npos) return false;
        key = std::string(c.s.substr(c.i, end - c.i));
        c.i = end + markers::kGemmaStrDelim.size();
        c.skip_ws();
        return c.eat(":") && !key.empty();
    }
    const size_t colon = c.s.find(':', c.i);
    if (colon == std::string_view::npos) return false;
    key = std::string(text::trim(c.s.substr(c.i, colon - c.i)));
    c.i = colon + 1;
    return !key.empty() && key.find_first_of("{}[],") == std::string::npos;
}

bool read_object(Cursor& c, json& out, int depth, const ParamTypes* types) {
    if (depth > kMaxDepth || !c.eat("{")) return false;
    out = json::object();
    c.skip_ws();
    if (c.eat("}")) return true;
    for (;;) {
        std::string key;
        if (!read_key(c, key)) return false;
        const std::string* hint = nullptr;
        if (types != nullptr) {
            auto it = types->find(key);
            if (it != types->end()) hint = &it->second;
        }
        json v;
        if (!read_value(c, v, depth + 1, hint)) return false;
        out[key] = std::move(v);
        c.skip_ws();
        if (c.eat(",")) continue;
        return c.eat("}");
    }
}

bool read_array(Cursor& c, json& out, int depth) {
    if (depth > kMaxDepth || !c.eat("[")) return false;
    out = json::array();
    c.skip_ws();
    if (c.eat("]")) return true;
    for (;;) {
        json v;
        if (!read_value(c, v, depth + 1, nullptr)) return false;
        out.push_back(std::move(v));
        c.skip_ws();
        if (c.eat(",")) continue;
        return c.eat("]");
    }
}

bool read_value(Cursor& c, json& out, int depth, const std::string* type_hint) {
    c.skip_ws();
    if (c.done()) return false;
    if (c.eat(markers::kGemmaStrDelim)) {
        const size_t end = c.s.find(markers::kGemmaStrDelim, c.i);
        if (end == std::string_view::npos) return false;
        out = std::string(c.s.substr(c.i, end - c.i));
        c.i = end + markers::kGemmaStrDelim.size();
        return true;
    }
    if (c.s[c.i] == '{') return read_object(c, out, depth, nullptr);
    if (c.s[c.i] == '[') return read_array(c, out, depth);
    const size_t start = c.i;
    while (c.i < c.s.size() && c.s[c.i] != ',' && c.s[c.i] != '}' && c.s[c.i] != ']') ++c.i;
    if (c.i == start) return false;
    out = bare_value(c.s.substr(start, c.i - start), type_hint);
    return true;
}

// The text between the markers: call:NAME{...}
bool parse_gemma_body(std::string_view body, const ToolSchemas* schemas, ToolCall& out) {
    Cursor c{text::trim(body), 0};
    if (!c.eat("call:")) return false;
    const size_t brace = c.s.find('{', c.i);
    if (brace == std::string_view::npos) return false;
    out.name = std::string(text::trim(c.s.substr(c.i, brace - c.i)));
    if (out.name.empty()) return false;
    c.i = brace;
    const ParamTypes* types = nullptr;
    if (schemas != nullptr) {
        auto it = schemas->find(out.name);
        if (it != schemas->end()) types = &it->second;
    }
    json args;
    if (!read_object(c, args, 0, types)) return false;
    c.skip_ws();
    if (!c.done()) return false;   // text after the closing brace: not a call
    out.arguments = args.dump();
    return true;
}

ToolCallParse parse_gemma_tool_calls(std::string_view text, const ToolSchemas* schemas) {
    ToolCallParse result;
    const std::string_view open_m  = markers::kGemmaToolOpen;
    const std::string_view close_m = markers::kGemmaToolClose;
    size_t cursor = 0;
    while (cursor <= text.size()) {
        const size_t open = text.find(open_m, cursor);
        if (open == std::string_view::npos) {
            result.content.append(text.substr(cursor));
            break;
        }
        result.content.append(text.substr(cursor, open - cursor));
        const size_t body_start = open + open_m.size();
        const size_t close      = text.find(close_m, body_start);
        if (close == std::string_view::npos) {
            result.truncated = true;
            result.content.append(text.substr(open));
            break;
        }
        ToolCall call;
        call.id = log::format("call_%zu", result.calls.size());
        if (parse_gemma_body(text.substr(body_start, close - body_start), schemas, call)) {
            result.calls.push_back(std::move(call));
        } else {
            result.content.append(text.substr(open, close + close_m.size() - open));
        }
        cursor = close + close_m.size();
    }
    return result;
}

}  // namespace

std::string_view tool_call_open(ChatDialect dialect) {
    return dialect.gemma4() ? markers::kGemmaToolOpen : kOpen;
}
std::string_view tool_call_close(ChatDialect dialect) {
    return dialect.gemma4() ? markers::kGemmaToolClose : kClose;
}

const ToolCallTags& tool_call_tags() {
    static const ToolCallTags t{kOpen, kClose, kFnOpen, kFnClose, kParamOpen, kParamEnd};
    return t;
}


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

ToolCallParse parse_tool_calls(std::string_view text, ChatDialect dialect, const ToolSchemas* schemas) {
    return dialect.gemma4() ? parse_gemma_tool_calls(text, schemas) : parse_qwen_tool_calls(text, schemas);
}

}  // namespace lgc
