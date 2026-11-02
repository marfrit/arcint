#include "api/request_id.h"
#include "harness.h"

#include <regex>
#include <set>
#include <string>
#include <vector>

// The request id itself: what is kept, what is replaced, and the shape of a
// generated one. The HTTP half — the header on every response, `rid=` on the
// log lines, the replaced value never appearing anywhere — is tests/reqid.sh.
namespace {

using namespace lgc;

const std::regex k_generated{"^req-[0-9a-f]{16}$"};

}  // namespace

TEST(reqid_supplied_valid_is_kept_verbatim) {
    const auto kept = api::resolve_request_id("trace-7_2:AB.cd");
    CHECK(kept.id == "trace-7_2:AB.cd");
    CHECK(!kept.replaced);
}

TEST(reqid_absent_or_empty_is_replaced) {
    // An empty value is treated as absent: nothing was sent that could be
    // echoed or logged, so there is nothing to report as replaced.
    const auto r = api::resolve_request_id("");
    CHECK(!r.replaced);
    CHECK(std::regex_match(r.id, k_generated));
}

TEST(reqid_punctuation_of_the_grammar_is_kept) {
    // The grammar is [A-Za-z0-9._:-] literally: a value made only of the
    // allowed punctuation is a valid id and is taken over, not replaced.
    for (const std::string& ok : std::vector<std::string>{"....", "-_-", "___", "::", ".:_-"}) {
        const auto r = api::resolve_request_id(ok);
        CHECK(!r.replaced);
        CHECK(r.id == ok);
    }
}

TEST(reqid_invalid_is_replaced_and_never_echoed) {
    // A space, a CR, an LF, a tab, non-ASCII, a shell metacharacter, and the
    // 129-character value: none of them is kept, and none of them can reach a
    // log line or a response header, because what comes back is generated.
    for (const std::string& bad : std::vector<std::string>{
             "a b", "a\rb", "a\nb", "a\tb", "caf\xC3\xA9", "a/b", "a;b", "a|b", "a b",
             std::string(129, 'a')}) {
        const auto r = api::resolve_request_id(bad);
        CHECK(r.replaced);
        CHECK(r.id != bad);
        CHECK(std::regex_match(r.id, k_generated));
    }
}

TEST(reqid_128_is_the_longest_kept) {
    const std::string ok(128, 'a');
    CHECK(!api::resolve_request_id(ok).replaced);
    CHECK(api::resolve_request_id(ok).id == ok);
    CHECK(api::resolve_request_id(std::string(129, 'a')).replaced);
}

TEST(reqid_generated_is_random_not_a_counter) {
    std::set<std::string> seen;
    for (int i = 0; i < 500; ++i) {
        const std::string id = api::new_request_id();
        CHECK(std::regex_match(id, k_generated));
        seen.insert(id);
    }
    CHECK(seen.size() == 500);
}

TEST(reqid_suffix_is_empty_without_an_id) {
    // A line emitted outside a request keeps the shape it always had.
    CHECK(api::rid_suffix("").empty());
    CHECK(api::rid_suffix("req-0123456789abcdef") == " rid=req-0123456789abcdef");
}
