#include "core/toolgrammar.h"

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "core/toolcall.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "peg-parser.h"

namespace lgc {
namespace {

using common_json = ::common_json;

// The coder's call tags, from the one table the parser reads too
// (core/toolcall.h, tool_call_tags()), so the grammar and the parser cannot
// disagree about what a call looks like and no tag is written out here.
std::string tag_text(std::string_view t) { return std::string(t); }

// The one derived-class method the tool-call block needs that the base builder
// does not provide: permute (required args in any order). Copied verbatim from
// the pinned llama.cpp's common/chat-peg-parser.cpp (MIT). The tag wrappers are
// the trivial `tag()` calls from common_chat_peg_builder, inlined so this does
// not drag in chat-peg-parser.cpp (and its Jinja dependency).
class toolcall_peg_builder : public common_peg_parser_builder {
  public:
    static constexpr size_t MAX_PERMUTE = 6;

    common_peg_parser permute(const std::string & rule_prefix, const std::vector<common_peg_parser> & parsers) {
        if (parsers.empty()) {
            return eps();
        }
        if (parsers.size() == 1 || parsers.size() > MAX_PERMUTE) {
            return sequence(parsers);
        }
        std::map<uint32_t, common_peg_parser>      rules;
        std::function<common_peg_parser(uint32_t)> remaining_of;
        remaining_of = [&](uint32_t remaining) -> common_peg_parser {
            if (remaining == 0) {
                return eps();
            }
            auto cached = rules.find(remaining);
            if (cached != rules.end()) {
                return cached->second;
            }
            auto alternatives = choice();
            for (size_t i = 0; i < parsers.size(); i++) {
                const uint32_t bit = 1u << i;
                if (remaining & bit) {
                    alternatives |= parsers[i] + remaining_of(remaining & ~bit);
                }
            }
            return rules.emplace(remaining, rule(rule_prefix + "-" + std::to_string(remaining), alternatives)).first->second;
        };
        return remaining_of((1u << parsers.size()) - 1);
    }

    // tag wrappers (from common_chat_peg_builder)
    common_peg_parser tool(const common_peg_parser & p) { return tag("tool", p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag("tool-open", p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag("tool-close", p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag("tool-name", p)); }
    common_peg_parser tool_args(const common_peg_parser & p) { return tag("tool-args", p); }
    common_peg_parser tool_arg(const common_peg_parser & p) { return tag("tool-arg", p); }
    common_peg_parser tool_arg_open(const common_peg_parser & p) { return atomic(tag("tool-arg-open", p)); }
    common_peg_parser tool_arg_close(const common_peg_parser & p) { return atomic(tag("tool-arg-close", p)); }
    common_peg_parser tool_arg_name(const common_peg_parser & p) { return atomic(tag("tool-arg-name", p)); }
    common_peg_parser tool_arg_value(const common_peg_parser & p) { return tag("tool-arg-value", p); }
    common_peg_parser tool_arg_string_value(const common_peg_parser & p) { return tag("tool-arg-string-value", p); }
    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return tag("tool-arg-value", p); }
};

// The exact strings a property's value may be, when every branch of its schema
// that admits a string lists its strings: an enum (its string members), a
// const string, through anyOf / oneOf, a "type" array (the closure expands it
// to an anyOf) and $ref. Branches that admit no string (null, a number, an
// object, a non-string member of a mixed enum) add nothing. Returns false when
// a branch admits a free string (a plain string, an untyped schema, allOf, a
// $ref cycle), so the caller keeps the free value. Order is the schema's,
// duplicates dropped (a "type" array repeats the enum in each branch).
bool listed_strings(const common_chat_schema& s, std::vector<std::string>& out,
                    std::unordered_set<const common_chat_schema*>& visiting) {
    auto add = [&](const common_json& v) {
        if (!v.is_string()) return;
        std::string str = v.get<std::string>();
        for (const auto& have : out)
            if (have == str) return;
        out.push_back(std::move(str));
    };
    switch (s.kind()) {
        case common_chat_schema::KIND_CONST:
            add(static_cast<const common_chat_schema_const&>(s).value);
            return true;
        case common_chat_schema::KIND_ENUM:
            for (const auto& v : static_cast<const common_chat_schema_enum&>(s).values) add(v);
            return true;
        case common_chat_schema::KIND_ANY_OF:
            for (const auto& child : static_cast<const common_chat_schema_any_of&>(s).children)
                if (!listed_strings(*child, out, visiting)) return false;
            return true;
        case common_chat_schema::KIND_REF: {
            const auto* target = static_cast<const common_chat_schema_ref&>(s).target;
            if (target == nullptr || !visiting.insert(target).second) return false;
            const bool ok = listed_strings(*target, out, visiting);
            visiting.erase(target);
            return ok;
        }
        case common_chat_schema::KIND_NULL:
        case common_chat_schema::KIND_BOOLEAN:
        case common_chat_schema::KIND_NUMBER:
        case common_chat_schema::KIND_INTEGER:
        case common_chat_schema::KIND_ARRAY:
        case common_chat_schema::KIND_TUPLE:
        case common_chat_schema::KIND_OBJECT:
            return true;
        case common_chat_schema::KIND_ANY:
        case common_chat_schema::KIND_STRING:
        case common_chat_schema::KIND_ALL_OF:
            return false;
    }
    return false;
}

// The name a rule gets in the grammar: the builder maps every run of characters
// outside [A-Za-z0-9-] to one "-" (peg-parser.cpp rule_name), so parameters
// "a_b" and "a.b" would share a rule.
std::string sanitized_rule_name(const std::string& name) {
    std::string out;
    bool        in_run = false;
    for (const char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-';
        if (ok) {
            out.push_back(c);
            in_run = false;
        } else if (!in_run) {
            out.push_back('-');
            in_run = true;
        }
    }
    return out;
}

// `base`, or `base-2`, `base-3`, ... : the first whose sanitized form no rule of
// this function has taken. add_rule is last-wins, so a clash would make one
// parameter's value silently resolve to another's rule. Without a clash the
// name is `base` itself, so a request that never clashed gets the grammar it
// always got.
std::string unique_rule_name(std::unordered_set<std::string>& used, const std::string& base) {
    std::string cand = base;
    for (size_t n = 2; !used.insert(sanitized_rule_name(cand)).second; n++) cand = base + "-" + std::to_string(n);
    return cand;
}

// Whether the raw schema of a property says `nullable: true`, itself or in an
// anyOf / oneOf branch (an OpenAPI-style null the closure does not read: it is
// not a type, so the value types stay {string} and the parent grammar took
// `null` as free text). `$ref` targets are not followed.
bool declares_nullable(const common_json& s, int depth = 0) {
    if (depth > 8 || !s.is_object()) return false;
    if (s.contains("nullable") && s.at("nullable").is_boolean() && s.at("nullable").get<bool>()) return true;
    for (const char* key : { "anyOf", "oneOf" }) {
        if (!s.contains(key) || !s.at(key).is_array()) continue;
        for (size_t i = 0; i < s.at(key).size(); i++)
            if (declares_nullable(s.at(key).at(i), depth + 1)) return true;
    }
    return false;
}

// Build one function's tool-call block, mirroring the reference's per-function
// loop. `params` is the function's JSON Schema (the `parameters` object the
// client declared). Returns the `tool-N` rule.
common_peg_parser build_function(toolcall_peg_builder & p, size_t tool_index,
                                 const std::string & name, const common_json & params) {
    const ToolCallTags& t      = tool_call_tags();
    const std::string kFnOpen    = tag_text(t.fn_open);
    const std::string kFnClose   = tag_text(t.fn_close);
    const std::string kCallClose = tag_text(t.close);
    const std::string kParamOpen = tag_text(t.param_open);
    const std::string kParamEnd  = tag_text(t.param_end);
    // DEVIATION from the reference (qwen3-coder.cpp closes a value only with
    // the canonical "\n</parameter>\n"): the closer is `</parameter>` with or
    // without the newline before it, then an optional newline, and a
    // free-text value ends at the first `</parameter>` in either form -- where
    // the parser (core/toolcall.cpp) has always ended it. Reason, measured-here
    // on a served raw trace: AREX-2 closes short values inline
    // (`finding</parameter>`); under the canonical-only closer the rest of the
    // call stayed inside the value, the grammar never completed and masked the
    // end of generation. A value can therefore not carry the literal closer
    // text. Cells: tests/test_toolgrammar.cpp toolgrammar_llama_inline_closer_*,
    // toolgrammar_llama_mixed_closers_in_one_call, ..._canonical_closer_unchanged.
    auto nl_opt     = p.optional(p.literal("\n"));
    auto arg_close  = p.tool_arg_close(nl_opt + p.literal(kParamEnd)) + nl_opt;
    auto arg_string = p.rule("xml-arg-string",
                             p.ac(p.tool_arg_string_value(p.until_one_of({ "\n" + kParamEnd, kParamEnd })) +
                                      p.tool_arg_close(nl_opt + p.literal(kParamEnd)),
                                  kParamEnd) +
                                 nl_opt);

    // The client's schema: one the closure cannot read, or whose root is not
    // an object (the call form names parameters, so nothing else can be
    // constrained), is the request's fault. The reference casts the root to
    // an object unchecked; here it is checked, since a request reaches it.
    std::shared_ptr<const common_chat_schema_document> doc;
    try {
        doc = std::make_shared<const common_chat_schema_document>(common_chat_schema_from_json(params));
    } catch (const std::exception& e) {
        throw std::invalid_argument("tool '" + name + "': its parameters schema cannot be constrained: " + e.what());
    }
    if (!doc->root || doc->root->kind() != common_chat_schema::KIND_OBJECT)
        throw std::invalid_argument("tool '" + name + "': its parameters schema must be an object");

    std::vector<common_peg_parser> required_args;
    std::vector<common_peg_parser> optional_args;

    const auto & obj = static_cast<const common_chat_schema_object &>(*doc->root);
    std::unordered_set<std::string> used_rules;
    for (const auto & prop : obj.properties) {
        const auto rule_name = unique_rule_name(used_rules, "tool-" + std::to_string(tool_index) + "-arg-" + prop.name);
        auto arg_open  = p.tool_arg_open(kParamOpen + p.tool_arg_name(p.literal(prop.name)) + ">\n");

        auto types = prop.schema->value_types();

        // DEVIATION (operator decision 2026-11-09; core/toolgrammar.h): a
        // string value whose strings are listed (enum / const) is one of those
        // strings exactly, then the closer. A listed string is raw text, as
        // every value in this call form: `<` and newlines are data, written as
        // is. One that ends in a newline takes the canonical closer only (the
        // parser strips one newline before the closer). A list holding the
        // closer text itself cannot be read back by the parser, which ends a
        // value at its first `</parameter>`: such a property stays free. So
        // does one holding the end of the call (`</function>`, `</tool_call>`:
        // the parser cuts the call there first) or a NUL (not representable in
        // the grammar text, which is handed on as a C string), and one declared
        // `nullable` (the parent grammar took `null` as free text there).
        std::vector<std::string> listed;
        bool                     is_listed = false;
        if (types.has(common_chat_schema::TYPE_STRING)) {
            std::unordered_set<const common_chat_schema*> visiting;
            is_listed = listed_strings(*prop.schema, listed, visiting) && !listed.empty();
            for (const auto& v : listed) {
                if (v.find('\0') != std::string::npos) is_listed = false;
                for (const std::string& closer_text : { kParamEnd, kFnClose, kCallClose })
                    if (v.find(closer_text) != std::string::npos) is_listed = false;
            }
            if (is_listed && params.contains("properties") && params.at("properties").is_object() &&
                params.at("properties").contains(prop.name) && declares_nullable(params.at("properties").at(prop.name)))
                is_listed = false;
        }
        // built only for a listed value: every parser is an arena node, node
        // ids are in the rule names, so a node made for nothing would move the
        // names of a request that lists nothing
        std::optional<common_peg_parser> listed_value;
        if (is_listed) {
            auto alts = p.choice();
            for (const auto& v : listed) {
                auto closer = (!v.empty() && v.back() == '\n') ? p.literal("\n" + kParamEnd)
                                                                 : nl_opt + p.literal(kParamEnd);
                auto value  = v.empty() ? p.tool_arg_string_value(p.eps()) : p.tool_arg_string_value(p.literal(v));
                alts |= value + p.tool_arg_close(closer);
            }
            listed_value = p.rule(unique_rule_name(used_rules, rule_name + "-listed"), alts + nl_opt);
        }

        auto arg_value = p.eps();
        if (!types.has(common_chat_schema::TYPE_STRING)) {
            arg_value = p.tool_arg_json_value(p.schema(p.json(), unique_rule_name(used_rules, rule_name + "-schema"), doc, *prop.schema)) + arg_close;
        } else if (types.is_only(common_chat_schema::TYPE_STRING)) {
            arg_value = is_listed ? *listed_value : arg_string;
        } else if (is_listed) {
            // the non-string alternatives keep today's JSON values; the free
            // text that subsumed them is replaced by the listed strings
            auto json_value = p.choice();
            if (types.has(common_chat_schema::TYPE_OBJECT)) json_value |= p.json_object();
            if (types.has(common_chat_schema::TYPE_ARRAY))  json_value |= p.json_array();
            if (types.has(common_chat_schema::TYPE_NUMBER) || types.has(common_chat_schema::TYPE_INTEGER)) json_value |= p.json_number();
            if (types.has(common_chat_schema::TYPE_BOOLEAN)) json_value |= p.json_bool();
            if (types.has(common_chat_schema::TYPE_NULL))    json_value |= p.json_null();
            arg_value = p.atomic(p.tool_arg_json_value(json_value) + arg_close) | *listed_value;
        } else {
            auto json_value = p.choice();
            if (types.has(common_chat_schema::TYPE_OBJECT)) json_value |= p.json_object();
            if (types.has(common_chat_schema::TYPE_ARRAY))  json_value |= p.json_array();
            if (types.has(common_chat_schema::TYPE_NUMBER) || types.has(common_chat_schema::TYPE_INTEGER)) json_value |= p.json_number();
            if (types.has(common_chat_schema::TYPE_BOOLEAN)) json_value |= p.json_bool();
            if (types.has(common_chat_schema::TYPE_NULL))    json_value |= p.json_null();
            arg_value = p.gbnf(p.atomic(p.tool_arg_json_value(json_value) + arg_close) | arg_string, "xml-arg-string");
        }

        auto arg_rule = p.rule(rule_name, p.tool_arg(arg_open + arg_value));
        (prop.required ? required_args : optional_args).push_back(arg_rule);
    }

    auto args = p.permute("tool-" + std::to_string(tool_index) + "-args", required_args);
    if (!optional_args.empty()) {
        args = args + p.zero_or_more(p.choice(optional_args));
    }

    auto func = p.tool(p.tool_open(kFnOpen + p.tool_name(p.literal(name)) + ">\n") +
                       p.tool_args(args) +
                       p.tool_close(p.literal(kFnClose + "\n")));

    return p.rule("tool-" + std::to_string(tool_index), func);
}

}  // namespace

std::string generate_tool_grammar(const std::vector<ToolSpec>& tools, bool parallel) {
    if (tools.empty()) return {};

    const ToolCallTags& t      = tool_call_tags();
    const std::string   kOpen  = tag_text(t.open);
    const std::string   kClose = tag_text(t.close);

    toolcall_peg_builder p;

    auto tool_choice = p.choice();
    for (size_t i = 0; i < tools.size(); i++) {
        // arcint's ToolSpec carries the client's schema as nlohmann/json; the
        // vendored closure speaks common_json (a pimpl over nlohmann/ordered_json), so
        // the conversion is a dump/parse (cheap: one schema per declared function).
        // A function declared without parameters is an object with none.
        const nlohmann::json& decl = tools[i].parameters;
        auto params = common_json::parse(decl.is_object() && !decl.empty() ? decl.dump()
                                                                          : std::string("{\"type\":\"object\"}"));
        tool_choice |= build_function(p, i, tools[i].name, params);
    }

    // The call block: opener, the chosen function, closer, as the reference's
    // tool_call rule. The reference's Qwen3-Coder variant that may omit the
    // opener (tool-call-first) applies only to a template without a think block
    // (is_qwen3_coder = !supports_reasoning); arcint's trigger is the opener,
    // so the opener is always there. Parallel calls as the reference: the first
    // call, then any number more (parallel_tool_calls, the OpenAI default), or
    // exactly one. min 0, as the reference for tool_choice auto: the grammar is
    // lazy, so it is only ever entered through the opener.
    auto tool_call_body = tool_choice + kClose + p.space();
    auto tool_call      = p.rule("tool-call", kOpen + "\n" + tool_call_body);
    auto calls          = parallel ? tool_call + p.zero_or_more(tool_call) : tool_call;
    p.trigger_rule("tool-call-root", p.repeat(calls, 0, 1));

    auto arena = p.build();
    return build_grammar([&](const common_grammar_builder& builder) {
        arena.build_grammar(builder, /*lazy=*/true);
    });
}

std::string tool_grammar_trigger() { return tag_text(tool_call_tags().open); }

}  // namespace lgc
