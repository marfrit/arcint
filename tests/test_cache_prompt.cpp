// `cache_prompt` (llama.cpp server's switch): a request with false evaluates
// its prompt in full -- no prefix reuse, no checkpoint resume. These cases are
// RED on the tree before the field existed: the parser left it at the default
// true, and kv_prefix_common reused the whole common prefix whatever the flag
// said.
//
// One-line mutation that turns the decision case red again after the fix:
// delete `if (!cache_prompt) return 0;` from kv_prefix_common in
// src/exec/kv_prefix.h (the clamps below then reuse the prefix).
//
// The parser halves are red under: not assigning req.cache_prompt in
// read_cache_prompt, or ignooring it in parse_completion_request / parse_chat_request.
#include "core/chat.h"
#include "exec/kv_prefix.h"
#include "harness.h"

#include <string>
#include <vector>

using namespace lgc;
using nlohmann::json;

namespace {

json chat_with(const json& extra) {
    json b = json{{"model", "stub"},
                  {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
    for (auto& [k, v] : extra.items()) b[k] = v;
    return b;
}

json completion_with(const json& extra) {
    json b = json{{"model", "stub"}, {"prompt", "hi"}};
    for (auto& [k, v] : extra.items()) b[k] = v;
    return b;
}

std::vector<int> ids(std::size_t n, int base = 100) {
    std::vector<int> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = base + static_cast<int>(i);
    return v;
}

}  // namespace

TEST(cache_prompt_defaults_true_on_both_routes) {
    ChatRequest chat;
    CHECK(!parse_chat_request(chat_with(json::object()), chat).has_value());
    CHECK(chat.cache_prompt);

    CompletionRequest comp;
    CHECK(!parse_completion_request(completion_with(json::object()), comp).has_value());
    CHECK(comp.cache_prompt);
}

TEST(cache_prompt_false_is_read_on_both_routes) {
    ChatRequest chat;
    CHECK(!parse_chat_request(chat_with({{"cache_prompt", false}}), chat).has_value());
    CHECK(!chat.cache_prompt);

    CompletionRequest comp;
    CHECK(!parse_completion_request(completion_with({{"cache_prompt", false}}), comp).has_value());
    CHECK(!comp.cache_prompt);
}

TEST(cache_prompt_must_be_a_boolean) {
    ChatRequest chat;
    CHECK(parse_chat_request(chat_with({{"cache_prompt", "no"}}), chat).has_value());
    CompletionRequest comp;
    CHECK(parse_completion_request(completion_with({{"cache_prompt", 0}}), comp).has_value());
}

TEST(cache_prompt_false_reuses_nothing_of_an_identical_prompt) {
    // The measured shape: 4,096 prompt tokens, the lane holding the same 4,096.
    const std::vector<int> have   = ids(4096);
    const std::vector<int> prompt = ids(4096);
    CHECK_EQ(kv_prefix_common(have, prompt, prompt.size(), false), std::size_t{0});
    // With the switch on, every token but the last is reused (the last runs for
    // its logits); the plain path decrements a full match itself.
    CHECK_EQ(kv_prefix_common(have, prompt, prompt.size(), true), std::size_t{4096});
}

TEST(cache_prompt_true_still_reuses_the_common_prefix) {
    // A shared system prompt and a changed user turn: the prefix to the first
    // difference is what a request that allows reuse gets.
    const std::vector<int> have   = ids(128);
    std::vector<int>       prompt = ids(128);
    prompt[100] = 999;   // the turn's change
    CHECK_EQ(kv_prefix_common(have, prompt, prompt.size(), true), std::size_t{100});
    CHECK_EQ(kv_prefix_common(have, prompt, prompt.size(), false), std::size_t{0});
}

TEST(cache_prompt_false_is_still_zero_at_the_drafting_paths_limit) {
    // The drafting path may reuse all but the last prompt token (`limit`);
    // false stays zero there too, as the plain path.
    const std::vector<int> have   = ids(64);
    const std::vector<int> prompt = ids(64);
    CHECK_EQ(kv_prefix_common(have, prompt, prompt.size() - 1, false), std::size_t{0});
    CHECK_EQ(kv_prefix_common(have, prompt, prompt.size() - 1, true), std::size_t{63});
}
