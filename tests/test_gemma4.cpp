#include "harness.h"

#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

#include "api/handlers.h"
#include "api/responses.h"
#include "api/responses_stream.h"
#include "core/markers.h"
#include "core/reasoning.h"
#include "core/toolcall.h"
#include "exec/backend.h"

// Gemma 4's dialect (ChatDialect::Gemma4): its thought channel goes to
// reasoning_content and its tool-call form to tool_calls, only when that model
// is loaded. Every marker comes from core/markers.h.
using namespace lgc;
using nlohmann::json;

namespace {

const ChatDialect G = ChatDialect::Gemma4;

std::string t_open() { return std::string(markers::kGemmaThoughtOpen); }
std::string t_close() { return std::string(markers::kGemmaThoughtClose); }
// the channel-start marker alone: the opener without the word and the newline
std::string channel_start() { return t_open().substr(0, t_open().size() - std::string_view("thought\n").size()); }
std::string c_open() { return std::string(markers::kGemmaToolOpen); }
std::string c_close() { return std::string(markers::kGemmaToolClose); }
std::string q() { return std::string(markers::kGemmaStrDelim); }
std::string call(const std::string& name, const std::string& args) {
    return c_open() + "call:" + name + "{" + args + "}" + c_close();
}
json args_of(const ToolCall& c) { return json::parse(c.arguments); }

constexpr int kCloseId = 7;

class ScriptBackend final : public Backend {
public:
    explicit ScriptBackend(std::vector<std::pair<std::string, int>> script) : script_(std::move(script)) {
        status_.id          = "stub";
        status_.served_id   = "stub";
        status_.n_ctx_train = kStubNCtxTrain;
        status_.loaded      = true;
        status_.stub        = true;
        status_.n_ctx       = 4096;
        status_.dialect     = G;   // the loaded model is Gemma 4
    }
    const ModelStatus& status() const override { return status_; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest& r) const override {
        seen_has_enable = r.has_enable_thinking;
        seen_enable     = r.enable_thinking;
        return prompt;
    }
    std::string        prompt = "prompt";
    ModelStatus&       mutable_status() { return status_; }
    mutable bool       seen_has_enable = false;
    mutable bool       seen_enable     = true;
    // the channel-end marker is one special token; the opener is several pieces
    ThinkBlockIds think_block_ids(ChatDialect) const override {
        ThinkBlockIds ids;
        ids.open_close_ids = {{-1, kCloseId}};
        ids.close_ids      = {kCloseId};
        ids.open_tags      = {t_open()};
        ids.close_tags     = {t_close()};
        return ids;
    }
    FinishReason generate(const GenerationInput&, int, const TokenCallback& on_piece, GenerationStats& stats) override {
        for (const auto& [piece, id] : script_) {
            ++stats.completion_tokens;
            if (on_piece(piece, id) != Control::Continue) return FinishReason::Stop;
        }
        return FinishReason::Stop;
    }

private:
    struct Tok final : public Tokenizer {
        std::vector<int> encode(std::string_view) override { return {}; }
        std::string      decode(const std::vector<int>&) override { return {}; }
        std::string      decode_one(int) override { return {}; }
        int              eos_id() const override { return 0; }
    } tok_;
    std::vector<std::pair<std::string, int>> script_;
    ModelStatus                              status_;
};

api::PreparedChat prepared(ChatDialect d, bool think_open) {
    api::PreparedChat prep;
    prep.think_open               = think_open;
    prep.dialect                  = d;
    prep.parse_tool_calls         = true;
    prep.id                       = "test";
    prep.model                    = "stub";
    prep.req.stream_include_usage = false;
    // the channel-end marker is one special token; the opener is several pieces
    prep.think_ids.open_close_ids = {{-1, kCloseId}};
    return prep;
}

// reasoning, the call, then nothing else
std::vector<std::pair<std::string, int>> reasoning_then_call() {
    return {{channel_start(), 100}, {"thought", 5},      {"\n", 6},          {"Look it up.", 8},
            {"\n", 6},              {t_close(), kCloseId}, {c_open(), 48},      {"call:get_weather{city:", 9},
            {q(), 52},              {"Kassel", 10},        {q(), 52},           {",days:3}", 11},
            {c_close(), 49}};
}

std::string trimmed(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == '\n' || s[b] == ' ')) ++b;
    while (e > b && (s[e - 1] == '\n' || s[e - 1] == ' ')) --e;
    return s.substr(b, e - b);
}

struct Out {
    std::string reasoning, content;
    json        calls = json::array();
};

Out non_streamed(Backend& backend, const api::PreparedChat& prep) {
    api::Context ctx;
    ctx.backend = &backend;
    const api::HttpResult r = api::run_chat(ctx, prep, 0);
    const json& msg = r.body["choices"][0]["message"];
    Out o;
    if (msg.contains("reasoning_content")) o.reasoning = msg["reasoning_content"].get<std::string>();
    if (msg["content"].is_string()) o.content = msg["content"].get<std::string>();
    if (msg.contains("tool_calls")) o.calls = msg["tool_calls"];
    return o;
}

Out streamed(Backend& backend, const api::PreparedChat& prep) {
    api::Context ctx;
    ctx.backend = &backend;
    Out o;
    api::SseWriter write = [&](std::string_view frame) {
        if (frame.rfind("data: ", 0) != 0) return true;
        std::string payload(frame.substr(6));
        while (!payload.empty() && payload.back() == '\n') payload.pop_back();
        if (payload == "[DONE]") return true;
        const json j = json::parse(payload);
        if (!j.contains("choices")) return true;
        for (const json& c : j["choices"]) {
            if (!c.contains("delta")) continue;
            const json& d = c["delta"];
            if (d.contains("reasoning_content")) o.reasoning += d["reasoning_content"].get<std::string>();
            if (d.contains("content") && d["content"].is_string()) o.content += d["content"].get<std::string>();
            if (d.contains("tool_calls"))
                for (const json& tc : d["tool_calls"]) o.calls.push_back(tc);
        }
        return true;
    };
    api::stream_chat(ctx, prep, 0, write);
    return o;
}

}  // namespace

// ---- the thought channel ---------------------------------------------------

TEST(gemma4_self_opened_channel_goes_to_reasoning) {
    const auto s = split_reasoning(t_open() + "I reckon it is Paris.\n" + t_close() + "Paris.", false, G);
    CHECK(s.closed);
    CHECK_EQ(s.reasoning, std::string("I reckon it is Paris."));
    CHECK_EQ(s.content, std::string("Paris."));
}

TEST(gemma4_template_opened_channel) {
    // the prompt ended inside the channel (the template does this after a tool response)
    const auto s = split_reasoning("check the result\n" + t_close() + "It is done.", true, G);
    CHECK(s.closed);
    CHECK_EQ(s.reasoning, std::string("check the result"));
    CHECK_EQ(s.content, std::string("It is done."));
}

TEST(gemma4_empty_channel_is_just_content) {
    // a model with thinking off may still emit the empty channel
    const auto s = split_reasoning(t_open() + t_close() + "Paris.", false, G);
    CHECK(s.closed);
    CHECK_EQ(s.reasoning, std::string(""));
    CHECK_EQ(s.content, std::string("Paris."));
}

TEST(gemma4_text_without_a_channel_is_content) {
    const auto s = split_reasoning("Paris.", false, G);
    CHECK(!s.closed);
    CHECK_EQ(s.content, std::string("Paris."));
}

TEST(gemma4_never_closed_channel_is_all_reasoning) {
    const auto s = split_reasoning(t_open() + "still thinking", false, G);
    CHECK(!s.closed);
    CHECK_EQ(s.reasoning, std::string("still thinking"));
    CHECK_EQ(s.content, std::string(""));
}

TEST(gemma4_opener_needs_the_whole_word_and_newline) {
    std::string almost = t_open();
    almost.pop_back();   // no newline
    const std::string raw = almost + "ful day" + t_close() + "x";
    CHECK_EQ(split_reasoning(raw, false, G).content, raw);
}

TEST(gemma4_channel_is_honoured_for_gemma_only) {
    const std::string raw = t_open() + "plan" + t_close() + "answer";
    for (ChatDialect d : {ChatDialect(ChatDialect::Qwen), ChatDialect(ChatDialect::Extended)}) {
        const auto s = split_reasoning(raw, false, d);
        CHECK(!s.closed);
        CHECK_EQ(s.content, raw);
    }
}

TEST(gemma4_does_not_honour_the_other_families_pairs) {
    const std::string qwen = std::string(markers::kThinkOpen) + "x" + std::string(markers::kThinkClose) + "y";
    CHECK_EQ(split_reasoning(qwen, false, G).content, qwen);
    const std::string angle =
        std::string(markers::kThinkOpenAngle) + "x" + std::string(markers::kThinkCloseAngle) + "y";
    CHECK_EQ(split_reasoning(angle, false, G).content, angle);
}

TEST(gemma4_tag_table) {
    CHECK_EQ(think_tag_pairs(G).size(), 1u);
    CHECK_EQ(think_tag_pairs(G)[0].first, t_open());
    CHECK_EQ(think_tag_pairs(G)[0].second, t_close());
    // the earlier dialects keep exactly their tables
    CHECK_EQ(think_tag_pairs(ChatDialect(ChatDialect::Qwen)).size(), 1u);
    CHECK_EQ(think_tag_pairs(ChatDialect(ChatDialect::Qwen))[0].first, std::string(markers::kThinkOpen));
    CHECK_EQ(think_tag_pairs(ChatDialect(ChatDialect::Extended)).size(), 3u);
    CHECK_EQ(think_tag_pairs(true).size(), 3u);
    CHECK_EQ(think_tag_pairs(false).size(), 1u);
}

TEST(gemma4_prompt_ends_inside_the_channel) {
    const std::string head = "turn\n";
    CHECK(api::prompt_ends_inside_think_block(head + t_open(), G));
    CHECK(!api::prompt_ends_inside_think_block(head + "model\n", G));
    CHECK(!api::prompt_ends_inside_think_block(head + t_open(), ChatDialect(ChatDialect::Qwen)));
    CHECK(!api::prompt_ends_inside_think_block(head + t_open(), ChatDialect(ChatDialect::Extended)));
    // the Qwen rule is untouched, and Gemma ignores it
    const std::string qwen = head + std::string(markers::kThinkOpen) + "\n";
    CHECK(api::prompt_ends_inside_think_block(qwen, ChatDialect(ChatDialect::Qwen)));
    CHECK(!api::prompt_ends_inside_think_block(qwen, G));
}

TEST(gemma4_streaming_agrees_with_the_split_at_every_cut) {
    const std::string raw = t_open() + "weigh it\n" + t_close() + "\n\nParis.";
    const ReasoningSplit whole = split_reasoning(raw, false, G);
    for (size_t cut = 0; cut <= raw.size(); ++cut) {
        ReasoningStreamer st(false, G);
        std::string       reasoning, content;
        for (const std::string& piece : {raw.substr(0, cut), raw.substr(cut)}) {
            const auto step = st.push(piece);
            reasoning += step.reasoning;
            content += step.content;
        }
        const auto tail = st.flush();
        reasoning += tail.reasoning;
        content += tail.content;
        while (!reasoning.empty() && reasoning.back() == '\n') reasoning.pop_back();
        while (!content.empty() && content.front() == '\n') content.erase(0, 1);
        CHECK_EQ(reasoning, whole.reasoning);
        CHECK_EQ(content, whole.content);
    }
}

TEST(gemma4_streaming_by_closer_id) {
    ReasoningStreamer st(false, G, {{-1, kCloseId}});
    std::string      reasoning, content;
    for (const auto& [piece, id] : std::vector<std::pair<std::string, int>>{
             {channel_start(), 100}, {"thought", 5}, {"\n", 6}, {"hmm", 8}, {t_close(), kCloseId}, {"Paris", 9}}) {
        const auto step = st.push(piece, id);
        reasoning += step.reasoning;
        content += step.content;
    }
    CHECK(st.closed());
    CHECK_EQ(reasoning, std::string("hmm"));
    CHECK_EQ(content, std::string("Paris"));
}

// ---- the tool-call form ----------------------------------------------------

TEST(gemma4_tool_call_with_a_string_and_a_number) {
    const auto r = parse_tool_calls("Let me check.\n" + call("get_weather", "city:" + q() + "Kassel" + q() + ",days:3"), G);
    CHECK_EQ(r.calls.size(), 1u);
    if (r.calls.size() != 1u) return;
    CHECK_EQ(r.calls[0].name, std::string("get_weather"));
    CHECK_EQ(r.calls[0].id, std::string("call_0"));
    const json a = args_of(r.calls[0]);
    CHECK_EQ(a["city"].get<std::string>(), std::string("Kassel"));
    CHECK(a["days"].is_number_integer());
    CHECK_EQ(a["days"].get<int>(), 3);
    CHECK_EQ(r.content, std::string("Let me check.\n"));
    CHECK(!r.truncated);
}

TEST(gemma4_tool_call_scalars_objects_and_arrays) {
    const auto r = parse_tool_calls(
        call("f", "on:true,off:false,none:null,x:-1.5,o:{a:1,b:" + q() + "t" + q() + "},l:[1," + q() + "z" + q() + ",{k:2}]"), G);
    CHECK_EQ(r.calls.size(), 1u);
    if (r.calls.size() != 1u) return;
    const json a = args_of(r.calls[0]);
    CHECK(a["on"] == true);
    CHECK(a["off"] == false);
    CHECK(a["none"].is_null());
    CHECK(a["x"] == -1.5);
    CHECK_EQ(a["o"]["a"].get<int>(), 1);
    CHECK_EQ(a["o"]["b"].get<std::string>(), std::string("t"));
    CHECK_EQ(a["l"].size(), 3u);
    CHECK_EQ(a["l"][1].get<std::string>(), std::string("z"));
    CHECK_EQ(a["l"][2]["k"].get<int>(), 2);
}

TEST(gemma4_tool_call_string_may_hold_structure_characters) {
    const auto r = parse_tool_calls(call("f", "s:" + q() + "a,b}{c:d]" + q() + ",n:1"), G);
    CHECK_EQ(r.calls.size(), 1u);
    if (r.calls.size() != 1u) return;
    const json a = args_of(r.calls[0]);
    CHECK_EQ(a["s"].get<std::string>(), std::string("a,b}{c:d]"));
    CHECK_EQ(a["n"].get<int>(), 1);
}

TEST(gemma4_tool_call_quoted_keys_and_no_arguments) {
    const auto k = parse_tool_calls(call("f", q() + "key" + q() + ":" + q() + "v" + q()), G);
    CHECK_EQ(k.calls.size(), 1u);
    if (k.calls.size() != 1u) return;
    CHECK_EQ(args_of(k.calls[0])["key"].get<std::string>(), std::string("v"));
    const auto none = parse_tool_calls(call("ping", ""), G);
    CHECK_EQ(none.calls.size(), 1u);
    if (none.calls.size() != 1u) return;
    CHECK_EQ(none.calls[0].arguments, std::string("{}"));
}

TEST(gemma4_two_calls_in_one_turn) {
    const auto r = parse_tool_calls(call("a", "x:1") + call("b", "y:" + q() + "2" + q()), G);
    CHECK_EQ(r.calls.size(), 2u);
    if (r.calls.size() != 2u) return;
    CHECK_EQ(r.calls[0].name, std::string("a"));
    CHECK_EQ(r.calls[1].name, std::string("b"));
    CHECK_EQ(r.calls[1].id, std::string("call_1"));
    CHECK_EQ(r.content, std::string(""));
}

TEST(gemma4_declared_string_parameter_keeps_a_bare_value_as_text) {
    ToolSchemas schemas;
    schemas["f"] = {{"zip", "string"}, {"n", "integer"}};
    const auto with = parse_tool_calls(call("f", "zip:12345,n:5"), G, &schemas);
    CHECK_EQ(with.calls.size(), 1u);
    if (with.calls.size() != 1u) return;
    CHECK(args_of(with.calls[0])["zip"].is_string());
    CHECK(args_of(with.calls[0])["zip"] == "12345");
    CHECK_EQ(args_of(with.calls[0])["n"].get<int>(), 5);
}

TEST(gemma4_unterminated_call_is_returned_as_content) {
    const std::string raw = "go " + c_open() + "call:f{a:1";
    const auto r = parse_tool_calls(raw, G);
    CHECK(r.truncated);
    CHECK(r.calls.empty());
    CHECK_EQ(r.content, raw);
}

TEST(gemma4_malformed_call_stays_in_content) {
    for (const std::string& body : {std::string("nonsense"), std::string("call:f{a:"), std::string("call:{a:1}"),
                                    std::string("call:f{a:1} trailing"), std::string("call:f{a:" + q() + "open}")}) {
        const std::string raw = c_open() + body + c_close();
        const auto        r   = parse_tool_calls(raw, G);
        CHECK(r.calls.empty());
        CHECK_EQ(r.content, raw);
    }
}

TEST(gemma4_call_forms_do_not_cross_dialects) {
    const std::string gemma = call("f", "a:1");
    const auto        q_on_g = parse_tool_calls(gemma, ChatDialect(ChatDialect::Qwen));
    CHECK(q_on_g.calls.empty());
    CHECK_EQ(q_on_g.content, gemma);
    const std::string qwen = std::string(tool_call_tags().open) + "{\"name\": \"f\", \"arguments\": {}}" +
                             std::string(tool_call_tags().close);
    const auto g_on_q = parse_tool_calls(qwen, G);
    CHECK(g_on_q.calls.empty());
    CHECK_EQ(g_on_q.content, qwen);
    // every non-Gemma dialect is exactly the Qwen parser
    const auto direct = parse_qwen_tool_calls(qwen);
    for (ChatDialect d : {ChatDialect(ChatDialect::Qwen), ChatDialect(ChatDialect::Extended)}) {
        const auto r = parse_tool_calls(qwen, d);
        CHECK_EQ(r.calls.size(), direct.calls.size());
        CHECK_EQ(r.calls[0].name, direct.calls[0].name);
        CHECK_EQ(r.content, direct.content);
    }
}

TEST(gemma4_tool_markers_by_dialect) {
    CHECK_EQ(std::string(tool_call_open(G)), c_open());
    CHECK_EQ(std::string(tool_call_close(G)), c_close());
    CHECK_EQ(std::string(tool_call_open(ChatDialect(ChatDialect::Qwen))), std::string(tool_call_tags().open));
    CHECK_EQ(std::string(tool_call_close(ChatDialect(ChatDialect::Extended))), std::string(tool_call_tags().close));
}

// ---- the chat routes, end to end -------------------------------------------

TEST(gemma4_chat_nonstreaming_reasoning_and_tool_call) {
    ScriptBackend backend(reasoning_then_call());
    const Out     o = non_streamed(backend, prepared(G, false));
    CHECK_EQ(trimmed(o.reasoning), std::string("Look it up."));
    CHECK_EQ(o.content, std::string(""));
    CHECK_EQ(o.calls.size(), 1u);
    if (o.calls.size() != 1u) return;
    CHECK_EQ(o.calls[0]["function"]["name"].get<std::string>(), std::string("get_weather"));
    const json a = json::parse(o.calls[0]["function"]["arguments"].get<std::string>());
    CHECK_EQ(a["city"].get<std::string>(), std::string("Kassel"));
    CHECK_EQ(a["days"].get<int>(), 3);
}

TEST(gemma4_chat_streaming_reasoning_and_tool_call) {
    ScriptBackend backend(reasoning_then_call());
    const Out     o = streamed(backend, prepared(G, false));
    CHECK_EQ(trimmed(o.reasoning), std::string("Look it up."));
    CHECK_EQ(o.content, std::string(""));   // no marker text leaks into content
    CHECK_EQ(o.calls.size(), 1u);
    if (o.calls.size() != 1u) return;
    CHECK_EQ(o.calls[0]["function"]["name"].get<std::string>(), std::string("get_weather"));
    const json a = json::parse(o.calls[0]["function"]["arguments"].get<std::string>());
    CHECK_EQ(a["city"].get<std::string>(), std::string("Kassel"));
}

TEST(gemma4_chat_answer_after_the_channel_both_routes) {
    const std::vector<std::pair<std::string, int>> script = {
        {channel_start(), 100}, {"thought", 5}, {"\n", 6}, {"capital of France", 8}, {"\n", 6},
        {t_close(), kCloseId},  {"The capital", 9}, {" is Paris.", 10}};
    ScriptBackend     backend(script);
    const auto        prep = prepared(G, false);
    const Out         a    = non_streamed(backend, prep);
    const Out         b    = streamed(backend, prep);
    for (const Out* o : {&a, &b}) {
        CHECK_EQ(trimmed(o->reasoning), std::string("capital of France"));
        CHECK_EQ(o->content, std::string("The capital is Paris."));
        CHECK(o->calls.empty());
    }
}

TEST(gemma4_chat_template_opened_channel_both_routes) {
    const std::vector<std::pair<std::string, int>> script = {
        {"the tool said 4", 8}, {"\n", 6}, {t_close(), kCloseId}, {"It is 4.", 9}};
    ScriptBackend backend(script);
    const auto    prep = prepared(G, true);
    for (const Out& o : {non_streamed(backend, prep), streamed(backend, prep)}) {
        CHECK_EQ(trimmed(o.reasoning), std::string("the tool said 4"));
        CHECK_EQ(o.content, std::string("It is 4."));
    }
}

TEST(gemma4_call_text_is_left_alone_on_a_qwen_dialect_model) {
    // the same script on a model that is not Gemma 4: nothing is split, nothing parsed
    ScriptBackend backend(reasoning_then_call());
    const Out     o = non_streamed(backend, prepared(ChatDialect(ChatDialect::Qwen), false));
    CHECK(o.calls.empty());
    CHECK_EQ(o.reasoning, std::string(""));
    CHECK(o.content.find("call:get_weather") != std::string::npos);
}

TEST(chat_dialect_converts_from_the_old_bool_and_names_its_rules) {
    CHECK(ChatDialect(false) == ChatDialect::Qwen);
    CHECK(ChatDialect(true) == ChatDialect::Extended);
    CHECK(ChatDialect().kind == ChatDialect::Qwen);
    CHECK(ChatDialect(ChatDialect::Gemma4).gemma4());
    CHECK(!ChatDialect(true).gemma4());
    CHECK(ChatDialect(true).extended());
    CHECK(ChatDialect(ChatDialect::Gemma4).replays_arguments_as_object());
    CHECK(!ChatDialect(ChatDialect::Qwen).replays_arguments_as_object());
    CHECK(!ChatDialect(ChatDialect::Extended).replays_arguments_as_object());
}

// ---- the Responses route (the same dialect through prepare_response) -------

namespace {

json responses_body() {
    return {{"model", "stub"},
            {"input", "weather?"},
            {"tools", json::array({{{"type", "function"},
                                    {"name", "get_weather"},
                                    {"parameters", {{"type", "object"},
                                                    {"properties", {{"city", {{"type", "string"}}},
                                                                    {"days", {{"type", "integer"}}}}}}}}})},
            {"max_output_tokens", 256}};
}

struct RespFixture {
    Config        cfg;
    ScriptBackend backend;
    api::SlotPool slots{1};
    api::Context  ctx;
    explicit RespFixture(std::vector<std::pair<std::string, int>> script) : backend(std::move(script)) {
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
    }
    api::PreparedResponse prepare() {
        api::ResponseRequest r;
        CHECK(!api::parse_response_request(responses_body(), r).has_value());
        api::PreparedResponse p;
        CHECK(!api::prepare_response(ctx, std::move(r), p).has_value());
        return p;
    }
};

// the output items of a finished response, by type
json items_of(const json& output, const std::string& type) {
    json out = json::array();
    for (const json& it : output)
        if (it.value("type", "") == type) out.push_back(it);
    return out;
}

}  // namespace

TEST(gemma4_responses_route_parses_the_channel_and_the_call) {
    RespFixture   fx(reasoning_then_call());
    api::PreparedResponse p = fx.prepare();
    CHECK(p.chat.dialect == G);   // the route took the loaded model's dialect
    const api::HttpResult res = api::run_response(fx.ctx, p, 0);
    CHECK_EQ(res.status, 200);
    const json calls = items_of(res.body["output"], "function_call");
    CHECK_EQ(calls.size(), 1u);
    if (calls.size() != 1) return;
    CHECK_EQ(calls[0]["name"].get<std::string>(), std::string("get_weather"));
    const json a = json::parse(calls[0]["arguments"].get<std::string>());
    CHECK_EQ(a["city"].get<std::string>(), std::string("Kassel"));
    CHECK_EQ(a["days"].get<int>(), 3);
    CHECK_EQ(items_of(res.body["output"], "reasoning").size(), 1u);
}

TEST(gemma4_responses_route_streams_the_call_without_leaking_its_markers) {
    RespFixture   fx(reasoning_then_call());
    api::PreparedResponse p = fx.prepare();
    std::vector<std::string> frames;
    api::stream_response(fx.ctx, p, 0, [&](std::string_view frame) {
        frames.emplace_back(frame);
        return true;
    });
    json completed;
    std::string text_deltas;
    for (const std::string& f : frames) {
        const size_t data = f.find("data: ");
        if (f.rfind("event: ", 0) != 0 || data == std::string::npos) continue;
        std::string payload = f.substr(data + 6);
        while (!payload.empty() && payload.back() == '\n') payload.pop_back();
        const json j = json::parse(payload);
        if (j.value("type", "") == "response.completed") completed = j["response"];
        if (j.value("type", "") == "response.output_text.delta") text_deltas += j["delta"].get<std::string>();
    }
    CHECK_EQ(text_deltas, std::string(""));   // the call is not answer text
    CHECK(!completed.is_null());
    if (completed.is_null()) return;
    const json calls = items_of(completed["output"], "function_call");
    CHECK_EQ(calls.size(), 1u);
    if (calls.size() != 1) return;
    CHECK_EQ(calls[0]["name"].get<std::string>(), std::string("get_weather"));
}

// ---- the thinking default of an unset request -------------------------------
// Gemma 4's template reads `enable_thinking | default(false)`; the Qwen
// templates default to thinking. A request that says nothing must be ONE kind
// of turn for the template, the sampler row and the answer-row switch alike.

namespace {

struct ThinkCase {
    bool  enable;          // prep.req.enable_thinking
    bool  has;             // prep.req.has_enable_thinking
    bool  thinking_turn;   // prep.input.sampler.thinking_turn
    float temperature;     // the sampler row chosen: 0.9 thinking, 0.11 non-thinking
    bool  rendered_enable; // what the template saw
    bool  rendered_has;
};

ThinkCase prepare_think(ChatDialect d, json extra_body, std::optional<bool> think_default) {
    Config cfg;
    cfg.think_default = think_default;
    ScriptBackend backend({});
    backend.mutable_status().dialect = d;
    SamplerDefaults sd;
    sd.temperature           = 0.9f;
    sd.nothink_row           = true;
    sd.nothink_temperature   = 0.11f;
    backend.mutable_status().sampler_defaults = sd;
    api::SlotPool slots(1);
    api::Context  ctx{&cfg, &backend, &slots};
    json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
    for (auto& [k, v] : extra_body.items()) body[k] = v;
    api::PreparedChat prep;
    const auto err = api::prepare_chat(ctx, body, prep);
    CHECK(!err.has_value());
    return {prep.req.enable_thinking, prep.req.has_enable_thinking, prep.input.sampler.thinking_turn,
            prep.input.sampler.temperature, backend.seen_enable, backend.seen_has_enable};
}

const json kUnset = json::object();
const json kAsk   = {{"chat_template_kwargs", {{"enable_thinking", true}}}};
const json kOff   = {{"chat_template_kwargs", {{"enable_thinking", false}}}};

}  // namespace

TEST(gemma4_unset_request_is_a_non_thinking_turn_everywhere) {
    const ThinkCase c = prepare_think(G, kUnset, std::nullopt);
    CHECK(!c.enable);
    CHECK(!c.thinking_turn);
    CHECK_EQ(c.temperature, 0.11f);           // the non-thinking row
    CHECK(!c.rendered_enable);                // and the template is told so
    CHECK(c.rendered_has);
}

TEST(gemma4_request_and_operator_default_still_win) {
    const ThinkCase on = prepare_think(G, kAsk, std::nullopt);
    CHECK(on.enable && on.thinking_turn && on.rendered_enable);
    CHECK_EQ(on.temperature, 0.9f);
    const ThinkCase off = prepare_think(G, kOff, std::nullopt);
    CHECK(!off.enable && !off.thinking_turn);
    CHECK_EQ(off.temperature, 0.11f);
    const ThinkCase def = prepare_think(G, kUnset, true);   // --think-default on
    CHECK(def.enable && def.thinking_turn && def.rendered_enable);
    CHECK_EQ(def.temperature, 0.9f);
    const ThinkCase defoff = prepare_think(G, kAsk, false);   // the request beats --think-default off
    CHECK(defoff.enable && defoff.thinking_turn);
}

TEST(qwen_dialects_keep_thinking_as_the_unset_default) {
    for (ChatDialect d : {ChatDialect(ChatDialect::Qwen), ChatDialect(ChatDialect::Extended)}) {
        const ThinkCase c = prepare_think(d, kUnset, std::nullopt);
        CHECK(c.enable);
        CHECK(!c.has);                 // the template is not told: it defaults to thinking
        CHECK(c.thinking_turn);
        CHECK_EQ(c.temperature, 0.9f);
        CHECK(!c.rendered_has);
    }
}

// ---- review follow-ups ------------------------------------------------------

TEST(gemma4_value_in_ordinary_quotes_is_the_string_it_spells) {
    const auto r = parse_tool_calls(call("f", "city:\"Kassel\",n:1"), G);
    CHECK_EQ(r.calls.size(), 1u);
    if (r.calls.size() != 1u) return;
    CHECK(args_of(r.calls[0])["city"] == "Kassel");
    ToolSchemas schemas;
    schemas["f"] = {{"city", "string"}};
    const auto d = parse_tool_calls(call("f", "city:\"Kassel\""), G, &schemas);
    CHECK_EQ(d.calls.size(), 1u);
    if (d.calls.size() != 1u) return;
    CHECK(args_of(d.calls[0])["city"] == "Kassel");
}

TEST(gemma4_delimited_string_is_coerced_to_the_declared_type_as_in_the_qwen_form) {
    ToolSchemas schemas;
    schemas["f"] = {{"n", "integer"}, {"on", "boolean"}, {"s", "string"}, {"l", "array"}};
    const auto r = parse_tool_calls(
        call("f", "n:" + q() + "5" + q() + ",on:" + q() + "true" + q() + ",s:" + q() + "7" + q() + ",l:" + q() + "[1,2]" + q()), G,
        &schemas);
    CHECK_EQ(r.calls.size(), 1u);
    if (r.calls.size() != 1u) return;
    const json a = args_of(r.calls[0]);
    CHECK(a["n"] == 5);
    CHECK(a["on"] == true);
    CHECK(a["s"] == "7");           // a declared string stays one
    CHECK(a["l"].is_array());
    // undeclared: a delimited string stays a string
    const auto u = parse_tool_calls(call("g", "n:" + q() + "5" + q()), G, &schemas);
    CHECK_EQ(u.calls.size(), 1u);
    if (u.calls.size() != 1u) return;
    CHECK(args_of(u.calls[0])["n"] == "5");
}

TEST(gemma4_tool_opener_split_across_two_streamed_pieces) {
    const std::string open = c_open();
    const size_t      cut  = open.size() / 2;
    std::vector<std::pair<std::string, int>> script = {
        {t_open(), -1},           {"plan", -1},     {t_close(), kCloseId},
        {open.substr(0, cut), -1}, {open.substr(cut) + "call:f{a:1}", -1}, {c_close(), 49}};
    ScriptBackend backend(script);
    const Out     o = streamed(backend, prepared(G, false));
    CHECK_EQ(o.content, std::string(""));   // not even the first half of the marker leaks
    CHECK_EQ(o.calls.size(), 1u);
    if (o.calls.size() != 1u) return;
    CHECK_EQ(o.calls[0]["function"]["name"].get<std::string>(), std::string("f"));
}

TEST(gemma4_responses_stream_with_the_prompt_already_inside_the_channel) {
    RespFixture fx({{"check the result\n", 8}, {t_close(), kCloseId}, {"It is done.", 9}});
    fx.backend.prompt = "turn\n" + t_open();   // the template left the channel open
    api::PreparedResponse p = fx.prepare();
    CHECK(p.chat.think_open);
    std::vector<std::string> frames;
    api::stream_response(fx.ctx, p, 0, [&](std::string_view frame) {
        frames.emplace_back(frame);
        return true;
    });
    json completed;
    std::string text, reasoning;
    for (const std::string& f : frames) {
        const size_t data = f.find("data: ");
        if (f.rfind("event: ", 0) != 0 || data == std::string::npos) continue;
        std::string payload = f.substr(data + 6);
        while (!payload.empty() && payload.back() == '\n') payload.pop_back();
        const json j = json::parse(payload);
        const std::string type = j.value("type", "");
        if (type == "response.completed") completed = j["response"];
        if (type == "response.output_text.delta") text += j["delta"].get<std::string>();
        if (type.find("reasoning") != std::string::npos && type.find("delta") != std::string::npos)
            reasoning += j.value("delta", std::string());
    }
    CHECK_EQ(text, std::string("It is done."));
    CHECK_EQ(trimmed(reasoning), std::string("check the result"));
    CHECK(!completed.is_null());
}
