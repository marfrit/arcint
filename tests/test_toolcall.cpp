#include "core/toolcall.h"
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
