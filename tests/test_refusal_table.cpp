// One table, read by the refusal code and by this test, so the release notes
// can list every refused request field with the reason the server gives. The
// samples here only trigger each rule; the strings come from
// request_field_refusal_reason(), not from this file.
//
// RED state before the table existed: request_field_refusal_count() returned 0,
// so the table case failed; every name/message pair was unverified.
#include "core/chat.h"
#include "harness.h"

#include <map>
#include <string>

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

// A value that trips each field's rule. Kept next to the test, not in the
// server: the server reads only names and reasons from the table.
const std::map<std::string, json>& trigger_values() {
    static const std::map<std::string, json> m = {
        {"min_tokens", 32},
        {"cache_salt", "tenant-a"},
        {"n_probs", 5},
        {"logprobs", 3},
        {"min_keep", 4},
        {"top_logprobs", 5},
        {"prompt_logprobs", 1},
        {"samplers", json::array({"top_k"})},
        {"return_tokens", true},
        {"return_progress", true},
        {"timings_per_token", true},
        {"response_fields", json::array({"tokens"})},
        {"logit_bias", json{{"123", 5.0}}},
        {"grammar", "root ::= \"a\""},
        {"json_schema", json{{"type", "object"}}},
        {"response_format", json{{"type", "json_object"}}},
        {"best_of", 2},
        {"use_beam_search", true},
        {"bad_words", json::array({"x"})},
        {"allowed_token_ids", json::array({1})},
        {"truncate_prompt_tokens", 128},
        {"n_predict", 0},
        {"typical_p", 0.9},
        {"mirostat", 2},
        {"dynatemp_range", 0.2},
        {"dynatemp_exponent", 2.0},
        {"xtc_probability", 0.5},
        {"tfs_z", 0.9},
        {"top_n_sigma", 3},
        {"dry_multiplier", 0.8},
        {"include_stop_str_in_output", true},
        {"skip_special_tokens", false},
        {"spaces_between_special_tokens", false},
        {"chat_format", "qwen"},
    };
    return m;
}

}  // namespace

TEST(refusal_table_is_not_empty_and_every_entry_has_a_reason) {
    const size_t n = request_field_refusal_count();
    CHECK(n >= 30u);
    const RequestFieldRefusal* rows = request_field_refusals();
    for (size_t i = 0; i < n; ++i) {
        CHECK(rows[i].field != nullptr);
        CHECK(rows[i].reason != nullptr);
        if (!rows[i].field || !rows[i].reason) continue;
        CHECK(std::string(rows[i].reason).size() > 8u);
        // the name the refusal code looks up is the same row's name
        const char* r = request_field_refusal_reason(rows[i].field);
        CHECK(r != nullptr);
        if (r) CHECK_EQ(std::string(r), std::string(rows[i].reason));
    }
}

TEST(refusal_table_every_refused_field_is_refused_on_both_routes_with_its_reason_and_the_rest_parse) {
    const size_t n = request_field_refusal_count();
    const RequestFieldRefusal* rows = request_field_refusals();
    const auto& samples = trigger_values();
    size_t checked = 0;
    for (size_t i = 0; i < n; ++i) {
        const std::string field = rows[i].field ? rows[i].field : "";
        const auto it = samples.find(field);
        if (it == samples.end()) {
            t::fail(__FILE__, __LINE__, "no trigger value for table field " + field);
            continue;
        }
        ChatRequest chat;
        const auto chat_err = parse_chat_request(chat_with({{field, it->second}}), chat);
        CompletionRequest comp;
        const auto comp_err = parse_completion_request(completion_with({{field, it->second}}), comp);
        const char* reason = request_field_refusal_reason(rows[i].field);
        if (!rows[i].refused) {
            // accepted and ignored: the request parses as if the field were absent
            CHECK(!chat_err.has_value());
            CHECK(!comp_err.has_value());
            ++checked;
            continue;
        }
        CHECK(chat_err.has_value());
        CHECK(comp_err.has_value());
        for (const auto* err : {&chat_err, &comp_err}) {
            if (!err->has_value()) continue;
            CHECK((*err)->find(field) != std::string::npos);
            if (reason) CHECK((*err)->find(reason) != std::string::npos);
        }
        ++checked;
    }
    CHECK_EQ(checked, n);
}

TEST(refusal_table_split_is_the_operators) {
    // 1.8.1 (operator): 400 only where a silently ignored value would change the
    // answer the client expects; the rest are accepted with one warning per field.
    const char* const kRefused[] = {"min_tokens", "cache_salt", "prompt_logprobs", "logit_bias", "grammar",
                                    "json_schema", "best_of", "use_beam_search", "bad_words",
                                    "allowed_token_ids", "truncate_prompt_tokens", "n_predict",
                                    "include_stop_str_in_output", "skip_special_tokens", "chat_format"};
    size_t refused = 0;
    const size_t n = request_field_refusal_count();
    const RequestFieldRefusal* rows = request_field_refusals();
    for (size_t i = 0; i < n; ++i) if (rows[i].refused) ++refused;
    CHECK_EQ(refused, sizeof(kRefused) / sizeof(kRefused[0]));
    for (const char* f : kRefused) CHECK(request_field_refused(f));
    CHECK_EQ(n - refused, size_t{19});
    CHECK(!request_field_refused("response_format"));
    CHECK(!request_field_refused("user"));   // outside the table: not refused
}

TEST(refusal_table_every_tabled_field_has_a_trigger) {
    // The reverse direction: a trigger without a table row means the server
    // refuses a field the release notes would not list.
    const std::map<std::string, json>& samples = trigger_values();
    for (const auto& [field, value] : samples) {
        CHECK(request_field_refusal_reason(field.c_str()) != nullptr);
        ChatRequest chat;
        CHECK_EQ(parse_chat_request(chat_with({{field, value}}), chat).has_value(),
                 request_field_refused(field.c_str()));
    }
}
