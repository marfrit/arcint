// Fields benchmark clients send that arcint does not serve. A field that would
// change results is refused by name (the pattern this tree already uses for
// n > 1, `speed`, unknown `arcint.*`, `response_format` on speech), rather than
// accepted and dropped. Fields that are supported keep their behaviour
// (ignore_eos, seed); fields that do nothing at their default value are
// accepted at that value.
//
// RED on the tree before the checks existed: every case below returned no error
// (the parser dropped the field silently).
//
// One-line mutation after the fix: remove the named check from
// read_result_changing (or make it `return std::nullopt` early); the matching
// case fails again.
#include "core/chat.h"
#include "harness.h"

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

// The refusal is on both routes and names the field.
void refuses_on_both(const char* field, const json& value) {
    ChatRequest chat;
    const auto chat_err = parse_chat_request(chat_with({{field, value}}), chat);
    CHECK(chat_err.has_value());
    if (chat_err) CHECK(chat_err->find(field) != std::string::npos);

    CompletionRequest comp;
    const auto comp_err = parse_completion_request(completion_with({{field, value}}), comp);
    CHECK(comp_err.has_value());
    if (comp_err) CHECK(comp_err->find(field) != std::string::npos);
}

void accepts_on_both(const char* field, const json& value) {
    ChatRequest chat;
    CHECK(!parse_chat_request(chat_with({{field, value}}), chat).has_value());
    CompletionRequest comp;
    CHECK(!parse_completion_request(completion_with({{field, value}}), comp).has_value());
}

}  // namespace

TEST(ignored_fields_supported_sampler_switches_stay_supported) {
    // llama.cpp/OpenAI: both are read into the sampler today.
    ChatRequest chat;
    CHECK(!parse_chat_request(chat_with({{"ignore_eos", true}, {"seed", 42}}), chat).has_value());
    CHECK(chat.sampler.ignore_eos.value_or(false));
    CHECK_EQ(chat.sampler.seed.value_or(0), uint64_t{42});
    CompletionRequest comp;
    CHECK(!parse_completion_request(completion_with({{"ignore_eos", true}, {"seed", 42}}), comp).has_value());
}

TEST(ignored_fields_min_tokens_is_refused_when_it_would_change_results) {
    refuses_on_both("min_tokens", 32);
    // 0 is vLLM's default and demands nothing.
    accepts_on_both("min_tokens", 0);
}

TEST(ignored_fields_cache_salt_is_refused) {
    refuses_on_both("cache_salt", "tenant-a");
    // empty: no isolation was asked for
    accepts_on_both("cache_salt", "");
}

TEST(ignored_fields_n_probs_and_logprobs_are_refused_when_they_ask_for_probabilities) {
    refuses_on_both("n_probs", 5);
    refuses_on_both("logprobs", 3);
    accepts_on_both("n_probs", 0);
    accepts_on_both("logprobs", 0);
    accepts_on_both("logprobs", nullptr);
}

TEST(ignored_fields_samplers_order_is_refused) {
    refuses_on_both("samplers", json::array({"top_k", "top_p", "temperature"}));
    refuses_on_both("samplers", "kpp");
}

TEST(ignored_fields_return_tokens_refused_when_true) {
    // true asks for raw ids in a field this build never writes.
    refuses_on_both("return_tokens", true);
    accepts_on_both("return_tokens", false);
}

TEST(ignored_fields_llama_completion_switches_are_refused_when_they_change_the_response) {
    refuses_on_both("timings_per_token", true);
    refuses_on_both("return_progress", true);
    refuses_on_both("response_fields", json::array({"tokens"}));
    refuses_on_both("min_keep", 4);
}

TEST(ignored_fields_grammar_and_logit_bias_are_refused) {
    refuses_on_both("logit_bias", json{{"123", 5.0}});
    // llama.cpp's array form is the same request: [[id, bias], ...]
    refuses_on_both("logit_bias", json::array({{123, 5.0}}));
    refuses_on_both("json_schema", json{{"type", "object"}});
    refuses_on_both("grammar", "root ::= \"a\"");
}

TEST(ignored_fields_empty_forms_are_the_default_and_accepted) {
    // Writers emit these shapes for "nothing here": refused only when non-empty.
    accepts_on_both("grammar", "");
    accepts_on_both("json_schema", json::object());
    accepts_on_both("samplers", json::array());
    accepts_on_both("response_fields", json::array());
    accepts_on_both("logit_bias", json::array());
    // a Python client sends the float, not the integer
    accepts_on_both("top_n_sigma", -1.0);
    // llama.cpp's default: unset, not a zero budget
    accepts_on_both("n_predict", -1);
}

TEST(ignored_fields_response_format_other_than_text_is_refused) {
    // The chat and completions surfaces accept exactly the SDK default.
    accepts_on_both("response_format", json{{"type", "text"}});
    refuses_on_both("response_format", "text");
    refuses_on_both("response_format", json{{"type", "json_object"}});
    refuses_on_both("response_format", json{{"type", "json_schema"}});
    refuses_on_both("response_format", json{{"type", "made_up"}});
    refuses_on_both("response_format", json::object());
}

TEST(ignored_fields_vllm_completion_switches_are_refused) {
    refuses_on_both("best_of", 2);
    refuses_on_both("use_beam_search", true);
    refuses_on_both("bad_words", json::array({"x"}));
    refuses_on_both("allowed_token_ids", json::array({1, 2}));
    refuses_on_both("prompt_logprobs", 1);
    refuses_on_both("truncate_prompt_tokens", 128);
}

TEST(ignored_fields_llama_sampler_shaping_is_refused_outside_its_default) {
    // The rest of llama.cpp server's sampler fields: each accepted at its own
    // default (where it changes nothing) and refused otherwise.
    refuses_on_both("typical_p", 0.9);
    refuses_on_both("mirostat", 2);
    refuses_on_both("dynatemp_range", 0.2);
    refuses_on_both("dynatemp_exponent", 2.0);
    refuses_on_both("xtc_probability", 0.5);
    refuses_on_both("tfs_z", 0.9);
    refuses_on_both("top_n_sigma", 3);
    refuses_on_both("dry_multiplier", 0.8);
    accepts_on_both("typical_p", 1.0);
    accepts_on_both("mirostat", 0);
    accepts_on_both("dynatemp_range", 0.0);
    accepts_on_both("dynatemp_exponent", 1.0);
    accepts_on_both("xtc_probability", 0.0);
    accepts_on_both("tfs_z", 1.0);
    accepts_on_both("top_n_sigma", -1);
    accepts_on_both("dry_multiplier", 0.0);
}

TEST(ignored_fields_other_result_shaping_is_refused) {
    refuses_on_both("include_stop_str_in_output", true);
    accepts_on_both("include_stop_str_in_output", false);
    refuses_on_both("skip_special_tokens", false);
    accepts_on_both("skip_special_tokens", true);
    refuses_on_both("spaces_between_special_tokens", false);
    accepts_on_both("spaces_between_special_tokens", true);
    refuses_on_both("chat_format", "qwen");
    // vLLM's -1 is the no-truncation default; a count is a truncation.
    refuses_on_both("truncate_prompt_tokens", 128);
    accepts_on_both("truncate_prompt_tokens", -1);
}

TEST(ignored_fields_default_valued_benign_fields_keep_working) {
    // Clients send these routinely; they change no result at these values.
    accepts_on_both("best_of", 1);
    accepts_on_both("use_beam_search", false);
    accepts_on_both("min_keep", 0);
    accepts_on_both("user", "benchmark");
}
