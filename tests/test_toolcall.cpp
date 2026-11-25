#include "core/toolcall.h"
#include "core/chat.h"
#include "harness.h"

#include <nlohmann/json.hpp>

using namespace lgc;
using nlohmann::json;

TEST(toolcall_plain_text_survives_untouched) {
    const std::string text = "no tags here, just <angle> brackets";
    const auto        r    = parse_qwen_tool_calls(text);
    CHECK_EQ(r.content, text);
    CHECK(r.calls.empty());
    CHECK(!r.truncated);
}

TEST(toolcall_json_form) {
    const auto r = parse_qwen_tool_calls(
        "sure\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Kassel\"}}\n"
        "</tool_call>");

    CHECK_EQ(r.calls.size(), 1u);
    CHECK_EQ(r.calls[0].name, std::string("get_weather"));
    CHECK_EQ(r.calls[0].id, std::string("call_0"));
    CHECK_EQ(json::parse(r.calls[0].arguments)["city"].get<std::string>(), std::string("Kassel"));
    CHECK_EQ(r.content, std::string("sure\n"));
}

TEST(toolcall_json_form_with_string_arguments) {
    const auto r = parse_qwen_tool_calls(
        "<tool_call>{\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}</tool_call>");
    CHECK_EQ(r.calls.size(), 1u);
    CHECK_EQ(r.calls[0].arguments, std::string("{\"a\": 1}"));
}

TEST(toolcall_xml_form) {
    const auto r = parse_qwen_tool_calls(
        "<tool_call>\n"
        "<function=edit_file>\n"
        "<parameter=path>\nsrc/main.cpp\n</parameter>\n"
        "<parameter=line>\n42\n</parameter>\n"
        "</function>\n"
        "</tool_call>");

    CHECK_EQ(r.calls.size(), 1u);
    CHECK_EQ(r.calls[0].name, std::string("edit_file"));

    const json args = json::parse(r.calls[0].arguments);
    CHECK_EQ(args["path"].get<std::string>(), std::string("src/main.cpp"));
    // Without a declared schema every value stays a string: that is what the
    // wire format actually says.
    CHECK(args["line"].is_string());
    CHECK_EQ(args["line"].get<std::string>(), std::string("42"));
}

TEST(toolcall_xml_form_coerces_with_schema) {
    ToolSchemas schemas;
    schemas["edit_file"] = {{"line", "integer"}, {"dry_run", "boolean"}, {"tags", "array"}};

    const auto r = parse_qwen_tool_calls(
        "<tool_call>\n"
        "<function=edit_file>\n"
        "<parameter=line>\n42\n</parameter>\n"
        "<parameter=dry_run>\ntrue\n</parameter>\n"
        "<parameter=tags>\n[\"a\",\"b\"]\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        &schemas);

    const json args = json::parse(r.calls[0].arguments);
    CHECK_EQ(args["line"].get<int>(), 42);
    CHECK_EQ(args["dry_run"].get<bool>(), true);
    CHECK(args["tags"].is_array());
    CHECK_EQ(args["tags"].size(), 2u);
}

TEST(toolcall_coercion_falls_back_to_string) {
    ToolSchemas schemas;
    schemas["f"] = {{"n", "integer"}};

    const auto r = parse_qwen_tool_calls(
        "<tool_call>\n<function=f>\n<parameter=n>\nnot-a-number\n</parameter>\n</function>\n"
        "</tool_call>",
        &schemas);

    const json args = json::parse(r.calls[0].arguments);
    CHECK(args["n"].is_string());
}

TEST(toolcall_multi_line_value_keeps_inner_newlines) {
    const auto r = parse_qwen_tool_calls(
        "<tool_call>\n<function=write>\n<parameter=body>\nline1\nline2\n</parameter>\n"
        "</function>\n</tool_call>");
    const json args = json::parse(r.calls[0].arguments);
    CHECK_EQ(args["body"].get<std::string>(), std::string("line1\nline2"));
}

TEST(toolcall_multiple_blocks_get_sequential_ids) {
    const auto r = parse_qwen_tool_calls(
        "a<tool_call>{\"name\":\"one\"}</tool_call>b<tool_call>{\"name\":\"two\"}</tool_call>c");

    CHECK_EQ(r.calls.size(), 2u);
    CHECK_EQ(r.calls[0].id, std::string("call_0"));
    CHECK_EQ(r.calls[1].id, std::string("call_1"));
    CHECK_EQ(r.calls[0].arguments, std::string("{}"));
    CHECK_EQ(r.content, std::string("abc"));
}

TEST(toolcall_unterminated_block_is_returned_verbatim) {
    const std::string text = "text before <tool_call>\n{\"name\": \"f\"";
    const auto        r    = parse_qwen_tool_calls(text);

    CHECK(r.truncated);
    CHECK(r.calls.empty());
    CHECK_EQ(r.content, text);
}

TEST(toolcall_unparseable_body_is_kept_in_content) {
    const std::string text = "<tool_call>\nnot json, not xml\n</tool_call>";
    const auto        r    = parse_qwen_tool_calls(text);

    CHECK(r.calls.empty());
    CHECK_EQ(r.content, text);
}

TEST(toolcall_content_is_the_exact_concatenation_of_the_gaps) {
    // The streaming path sends the text before the first block and later sends
    // parsed content as a plain suffix, which only works if content is the
    // untrimmed concatenation of the non-block segments.
    const std::string text = "  before  <tool_call>{\"name\":\"f\"}</tool_call>  after  ";
    const auto        r    = parse_qwen_tool_calls(text);
    CHECK_EQ(r.content, std::string("  before    after  "));
    CHECK_EQ(r.content.rfind("  before  ", 0), 0u);
}

// The call form Dory1.0-4B's template teaches (its own instruction text, example names kept): the Qwen
// coder's function/parameter blocks inside one tool_call pair, with a multi-line value. The Qwen dialect's
// parser takes it as it is, so the model needs no parser change.
TEST(toolcall_dory_template_example_parses_under_the_qwen_dialect) {
    const std::string text =
        "I will call it.\n"
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n</function>\n</tool_call>";
    const auto r = parse_tool_calls(text, ChatDialect::Qwen);
    CHECK_EQ(r.calls.size(), 1u);
    CHECK_EQ(r.calls[0].name, std::string("example_function_name"));
    const json args = json::parse(r.calls[0].arguments);
    CHECK_EQ(args["example_parameter_1"].get<std::string>(), std::string("value_1"));
    CHECK_EQ(args["example_parameter_2"].get<std::string>(),
             std::string("This is the value for the second parameter\nthat can span\nmultiple lines"));
    CHECK_EQ(r.content, std::string("I will call it.\n"));
}

// `<` inside a value is data, not markup: a comparison, an inequality, markup
// and a value that ends in `<`. Only the parameter closer ends a value.
TEST(toolcall_values_keep_angle_brackets) {
    const std::string sql = "SELECT * FROM lines WHERE amount < 0.01 AND qty <> 0";
    const auto        r   = parse_qwen_tool_calls(
        "<tool_call>\n<function=run_query>\n<parameter=sql>\n" + sql + "\n</parameter>\n"
        "<parameter=html>\n<b>x</b>\n</parameter>\n<parameter=tail>\na <\n</parameter>\n</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), 1u);
    if (r.calls.size() != 1) return;
    const json args = json::parse(r.calls[0].arguments);
    CHECK_EQ(args["sql"].get<std::string>(), sql);
    CHECK_EQ(args["html"].get<std::string>(), std::string("<b>x</b>"));
    CHECK_EQ(args["tail"].get<std::string>(), std::string("a <"));
    CHECK_EQ(r.content, std::string(""));
    // the JSON form carries the same value
    const auto j = parse_qwen_tool_calls("<tool_call>\n{\"name\": \"run_query\", \"arguments\": {\"sql\": \"" + sql +
                                         "\"}}\n</tool_call>");
    CHECK_EQ(j.calls.size(), 1u);
    if (j.calls.size() == 1) CHECK_EQ(json::parse(j.calls[0].arguments)["sql"].get<std::string>(), sql);
}

// ---------------------------------------------- ids unique across turns (1.8.2)
// The parser numbers a turn's calls from call_0; the routes renumber them after
// the highest call_N in the request's history (next_call_index, number_calls).
namespace {

ToolCall named_call(const char* id) { return ToolCall{id, "read_file", "{}"}; }

ChatMessage tool_result(const char* id) {
    ChatMessage m;
    m.role         = "tool";
    m.tool_call_id = id;
    m.content      = "x";
    return m;
}

// user, an assistant turn calling call_0 and call_1, both results
ChatRequest two_call_history() {
    ChatRequest req;
    ChatMessage user;
    user.role    = "user";
    user.content = "read both";
    req.messages.push_back(user);
    ChatMessage a;
    a.role       = "assistant";
    a.tool_calls = {named_call("call_0"), named_call("call_1")};
    req.messages.push_back(a);
    req.messages.push_back(tool_result("call_0"));
    req.messages.push_back(tool_result("call_1"));
    return req;
}

std::vector<std::string> ids_of(const std::vector<ToolCall>& calls) {
    std::vector<std::string> out;
    for (const ToolCall& c : calls) out.push_back(c.id);
    return out;
}

const char* kTwoCalls =
    "<tool_call>{\"name\": \"read_file\", \"arguments\": {}}</tool_call>"
    "<tool_call>{\"name\": \"bash\", \"arguments\": {}}</tool_call>";

}  // namespace

// Mutation: next_call_index returning 0 (the 1.8.1 numbering) and the new calls
// are call_0, call_1 again.
TEST(toolcall_ids_follow_the_history) {
    const ChatRequest req = two_call_history();
    CHECK_EQ(next_call_index(req), 2u);
    ToolCallParse p = parse_qwen_tool_calls(kTwoCalls);
    CHECK(ids_of(p.calls) == (std::vector<std::string>{"call_0", "call_1"}));   // the parser's own
    number_calls(p.calls, next_call_index(req));
    CHECK(ids_of(p.calls) == (std::vector<std::string>{"call_2", "call_3"}));
    CHECK_EQ(p.calls[0].name, std::string("read_file"));   // only the ids move
    CHECK_EQ(p.calls[1].name, std::string("bash"));
}

TEST(toolcall_ids_without_history_start_at_zero) {
    ChatRequest req;
    ChatMessage user;
    user.role    = "user";
    user.content = "hi";
    req.messages.push_back(user);
    CHECK_EQ(next_call_index(req), 0u);
    CHECK_EQ(next_call_index(ChatRequest{}), 0u);
    ToolCallParse p = parse_qwen_tool_calls(kTwoCalls);
    number_calls(p.calls, next_call_index(req));
    CHECK(ids_of(p.calls) == (std::vector<std::string>{"call_0", "call_1"}));
}

// Only call_<decimal digits> counts; anything else is the client's own id.
TEST(toolcall_ids_ignore_foreign_ids) {
    ChatRequest req;
    ChatMessage a;
    a.role       = "assistant";
    a.tool_calls = {named_call("toolu_abc"), named_call("call_abc"), named_call("call_"),
                    named_call("call_-1"), named_call("call_+4"), named_call("call_ 3"),
                    named_call("call_3x"), named_call("Call_9"), named_call("xcall_9"),
                    named_call(""), named_call("call_99999999999999999999999999"),
                    named_call("call_18446744073709551615")};
    req.messages.push_back(a);
    req.messages.push_back(tool_result("toolu_abc"));
    CHECK_EQ(next_call_index(req), 0u);
    // a call_N among them still sets the offset
    req.messages[0].tool_calls.push_back(named_call("call_4"));
    CHECK_EQ(next_call_index(req), 5u);
}

// The highest N wins wherever it appears: out of order, in a later turn, or
// only as a tool message's tool_call_id (its call turn dropped by the client).
TEST(toolcall_ids_take_the_highest_anywhere) {
    ChatRequest req = two_call_history();
    ChatMessage a;
    a.role       = "assistant";
    a.tool_calls = {named_call("call_7"), named_call("call_5")};
    req.messages.push_back(a);
    CHECK_EQ(next_call_index(req), 8u);
    req.messages.push_back(tool_result("call_12"));
    CHECK_EQ(next_call_index(req), 13u);
    CHECK_EQ(next_call_index(req), 13u);   // deterministic: a pure function of the request
}
