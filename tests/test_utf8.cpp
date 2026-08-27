#include "harness.h"
#include "util/utf8.h"

using namespace lgc;

TEST(utf8_complete_prefix_ascii) {
    CHECK_EQ(utf8::complete_prefix(""), 0u);
    CHECK_EQ(utf8::complete_prefix("hello"), 5u);
}

TEST(utf8_complete_prefix_holds_back_truncated) {
    const std::string euro = "\xe2\x82\xac";  // U+20AC, three bytes

    CHECK_EQ(utf8::complete_prefix(euro), 3u);
    CHECK_EQ(utf8::complete_prefix(euro.substr(0, 1)), 0u);
    CHECK_EQ(utf8::complete_prefix(euro.substr(0, 2)), 0u);
    CHECK_EQ(utf8::complete_prefix("a" + euro.substr(0, 2)), 1u);
    CHECK_EQ(utf8::complete_prefix("a" + euro), 4u);
}

TEST(utf8_complete_prefix_four_byte) {
    const std::string emoji = "\xf0\x9f\xa7\xa9";  // U+1F9E9
    for (size_t n = 1; n < 4; ++n) CHECK_EQ(utf8::complete_prefix(emoji.substr(0, n)), 0u);
    CHECK_EQ(utf8::complete_prefix(emoji), 4u);
}

TEST(utf8_never_stalls_on_garbage) {
    // Five continuation bytes with no lead: malformed. Emitting beats hanging
    // the stream forever.
    const std::string junk("\x80\x80\x80\x80\x80", 5);
    CHECK_EQ(utf8::complete_prefix(junk), junk.size());
    // At most three bytes are ever withheld.
    const std::string mixed = std::string("ok") + "\xf0\x9f\xa7";
    CHECK_EQ(utf8::complete_prefix(mixed), 2u);
}

TEST(utf8_streamer_reassembles_split_codepoint) {
    utf8::Streamer s;
    const std::string emoji = "\xf0\x9f\xa7\xa9";

    CHECK_EQ(s.push("hi "), std::string("hi "));
    CHECK_EQ(s.push(emoji.substr(0, 2)), std::string(""));
    CHECK(!s.empty());
    CHECK_EQ(s.push(emoji.substr(2)), emoji);
    CHECK(s.empty());
    CHECK_EQ(s.flush(), std::string(""));
}

TEST(utf8_streamer_flush_returns_incomplete_tail) {
    utf8::Streamer s;
    CHECK_EQ(s.push("\xf0\x9f"), std::string(""));
    CHECK_EQ(s.flush(), std::string("\xf0\x9f"));
    CHECK(s.empty());
}

TEST(utf8_streamer_concatenation_is_lossless) {
    // The property the SSE path depends on: whatever order the bytes arrive in,
    // the pieces the client receives concatenate back to the original.
    const std::string text = "Gr\xc3\xbc\xc3\x9f""e \xe2\x82\xac 5 \xf0\x9f\xa7\xa9 ok";
    for (size_t chunk = 1; chunk <= 4; ++chunk) {
        utf8::Streamer s;
        std::string    got;
        for (size_t i = 0; i < text.size(); i += chunk) {
            got += s.push(std::string_view(text).substr(i, chunk));
        }
        got += s.flush();
        CHECK_EQ(got, text);
    }
}

TEST(utf8_is_valid) {
    CHECK(utf8::is_valid("plain ascii"));
    CHECK(utf8::is_valid("\xe2\x82\xac"));
    CHECK(!utf8::is_valid("\xe2\x82"));          // truncated
    CHECK(!utf8::is_valid("\xc0\xaf"));          // overlong '/'
    CHECK(!utf8::is_valid("\xed\xa0\x80"));      // surrogate
    CHECK(!utf8::is_valid("\x80"));              // stray continuation
}

TEST(utf8_count_codepoints) {
    CHECK_EQ(utf8::count_codepoints("abc"), 3u);
    CHECK_EQ(utf8::count_codepoints("\xe2\x82\xac"), 1u);
    CHECK_EQ(utf8::count_codepoints("a\xf0\x9f\xa7\xa9""b"), 3u);
}
