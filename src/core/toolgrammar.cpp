#include "core/toolgrammar.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "json-schema-to-grammar.h"
#include "json.h"
#include "peg-parser.h"

namespace lgc {
namespace {

using common_json = ::common_json;

// The coder's call tags, spelled from parts (a "<" or "</" prefix plus the
// rest) so no complete tag stands contiguously in this source. The open/close
// are single special tokens on the coder vocabulary (measured); the inner tags
// are spelled as text.
const std::string kOpen      = std::string("<")  + "tool_call>";
const std::string kClose     = std::string("</") + "tool_call>";
const std::string kFnOpen    = std::string("<")  + "function=";
const std::string kFnClose   = std::string("</") + "function>";
const std::string kParamOpen = std::string("<")  + "parameter=";
const std::string kParamEnd  = std::string("</") + "parameter>";

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

// Build one function's tool-call block, mirroring the reference's per-function
// loop. `params` is the function's JSON Schema (the `parameters` object the
// client declared). Returns the `tool-N` rule.
common_peg_parser build_function(toolcall_peg_builder & p, size_t tool_index,
                                 const std::string & name, const common_json & params) {
    auto arg_close  = p.tool_arg_close(p.literal("\n" + kParamEnd + "\n"));
    auto arg_string = p.rule("xml-arg-string",
                             p.ac(p.tool_arg_string_value(p.until("\n" + kParamEnd + "\n")) + arg_close, "\n" + kParamEnd + "\n"));

    auto doc = std::make_shared<const common_chat_schema_document>(common_chat_schema_from_json(params));

    std::vector<common_peg_parser> required_args;
    std::vector<common_peg_parser> optional_args;

    const auto & obj = static_cast<const common_chat_schema_object &>(*doc->root);
    for (const auto & prop : obj.properties) {
        auto rule_name = "tool-" + std::to_string(tool_index) + "-arg-" + prop.name;
        auto arg_open  = p.tool_arg_open(kParamOpen + p.tool_arg_name(p.literal(prop.name)) + ">\n");

        auto types = prop.schema->value_types();

        auto arg_value = p.eps();
        if (!types.has(common_chat_schema::TYPE_STRING)) {
            arg_value = p.tool_arg_json_value(p.schema(p.json(), rule_name + "-schema", doc, *prop.schema) + arg_close);
        } else if (types.is_only(common_chat_schema::TYPE_STRING)) {
            arg_value = arg_string;
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

std::string generate_tool_grammar(const std::vector<ToolSpec>& tools) {
    if (tools.empty()) return {};

    toolcall_peg_builder p;

    auto tool_choice = p.choice();
    for (size_t i = 0; i < tools.size(); i++) {
        // arcint's ToolSpec carries the client's schema as nlohmann/json; the
        // vendored closure speaks common_json (a pimpl over nlohmann/ordered_json), so
        // the conversion is a dump/parse (cheap: one schema per declared function).
        auto params = common_json::parse(tools[i].parameters.dump());
        tool_choice |= build_function(p, i, tools[i].name, params);
    }

    // The call block: opener, the chosen function, closer. A lazy trigger rule
    // on the opener, so prose before the call is unconstrained.
    auto tool_call_body = tool_choice + kClose + p.space();
    auto tool_call      = p.rule("tool-call", kOpen + "\n" + tool_call_body);
    p.trigger_rule("tool-call-root", p.repeat(tool_call, 1, 1));

    auto arena = p.build();
    return build_grammar([&](const common_grammar_builder& builder) {
        arena.build_grammar(builder, /*lazy=*/true);
    });
}

std::string tool_grammar_trigger() {
    // The call opener, as the lazy grammar's trigger pattern. It is a single
    // special token on the coder vocabulary (measured) and carries no regex
    // metacharacter, so the pattern is the opener itself (llama.cpp's
    // common_sampling_add_grammar_triggers regex-escapes a WORD trigger; here
    // there is nothing to escape).
    return kOpen;
}

}  // namespace lgc
