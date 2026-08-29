#include "harness.h"
#include "util/text.h"

using namespace lgc;

TEST(text_trim) {
    CHECK_EQ(std::string(text::trim("  hi  ")), std::string("hi"));
    CHECK_EQ(std::string(text::trim("\n\tx\r\n")), std::string("x"));
    CHECK_EQ(std::string(text::trim("   ")), std::string(""));
    CHECK_EQ(std::string(text::trim("")), std::string(""));
}

TEST(text_iequals) {
    CHECK(text::iequals("True", "true"));
    CHECK(!text::iequals("true", "trues"));
}

TEST(text_split) {
    const auto parts = text::split("a,b,,c", ',');
    CHECK_EQ(parts.size(), 4u);
    CHECK_EQ(std::string(parts[2]), std::string(""));
    CHECK_EQ(text::split("", ',').size(), 1u);
}

TEST(text_human_bytes) {
    CHECK_EQ(text::human_bytes(0), std::string("0 B"));
    CHECK_EQ(text::human_bytes(512), std::string("512 B"));
    CHECK_EQ(text::human_bytes(380ull * 1024 * 1024), std::string("380 MiB"));
    // The two figures the design's memory-map line prints.
    CHECK_EQ(text::human_bytes(static_cast<uint64_t>(4.2 * 1024 * 1024 * 1024)),
             std::string("4.2 GiB"));
    CHECK_EQ(text::human_bytes(static_cast<uint64_t>(15.9 * 1024 * 1024 * 1024)),
             std::string("15.9 GiB"));
}

TEST(text_partial_stop_suffix) {
    CHECK_EQ(text::partial_stop_suffix("hello <to", "<tool_call>"), 3u);
    CHECK_EQ(text::partial_stop_suffix("hello", "<tool_call>"), 0u);
    // A complete match is not a *partial* one: the caller handles that case.
    CHECK_EQ(text::partial_stop_suffix("x<tool_call>", "<tool_call>"), 0u);
    CHECK_EQ(text::partial_stop_suffix("", "abc"), 0u);
    CHECK_EQ(text::partial_stop_suffix("aab", "ab"), 0u);  // ends in "b", not "a"
    CHECK_EQ(text::partial_stop_suffix("aba", "ab"), 1u);
    // Longest wins: "aa" is a longer prefix of "aab" than "a".
    CHECK_EQ(text::partial_stop_suffix("xaa", "aab"), 2u);
}
