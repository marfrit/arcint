#include "harness.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "api/handlers.h"
#include "config.h"
#include "core/chat.h"
#include "core/markers.h"
#include "core/reasoning.h"
#include "exec/backend.h"
#include "util/utf8.h"

// force_nonempty_content (core/chat.h): the Dory1.0-4B model card's kwarg,
// acted on the way vLLM's Nemotron-V3 reasoning parser does (code,
// vllm/parser/nemotron_v3.py + abstract_parser.py, main 1479deb0392f).
//
//   non-streaming: after the reasoning split, empty or whitespace-only content
//                  and non-blank reasoning -> the reasoning is the content; the
//                  tool-call parse then runs on the promoted content.
//   streaming:     the think block never closed -> the whole reasoning text goes
//                  out once as a content delta before the final chunk, with no
//                  tool-call parse of it (the reference's stream has none).
//
// The pure decision is cased first, then both handler paths are driven against
// a scripted backend (no socket, no model).
namespace {

using namespace lgc;
using nlohmann::json;

const json kTrue  = json{{"force_nonempty_content", true}};
const json kFalse = json{{"force_nonempty_content", false}};

class ScriptedBackend final : public Backend {
public:
    explicit ScriptedBackend(std::vector<std::string> pieces, FinishReason reason = FinishReason::Stop)
        : pieces_(std::move(pieces)), reason_(reason) {
        status_.id          = "stub";
        status_.served_id   = "stub";
        status_.n_ctx_train = kStubNCtxTrain;
        status_.loaded      = true;
        status_.stub        = true;
        status_.n_ctx       = 4096;
    }
    const ModelStatus& status() const override { return status_; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest&) const override { return "prompt"; }
    FinishReason generate(const GenerationInput&, int, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        for (const std::string& piece : pieces_) {
            ++stats.completion_tokens;
            if (on_piece(piece, 0) != Control::Continue) return FinishReason::Stop;
        }
        return reason_;
    }

private:
    struct Tok final : public Tokenizer {
        std::vector<int> encode(std::string_view) override { return {}; }
        std::string      decode(const std::vector<int>&) override { return {}; }
        std::string      decode_one(int) override { return {}; }
        int              eos_id() const override { return 0; }
    } tok_;
    std::vector<std::string> pieces_;
    FinishReason             reason_;
    ModelStatus              status_;
};

api::PreparedChat make_prep(bool think_open, const json& kwargs, bool tools = false) {
    api::PreparedChat prep;
    prep.think_open       = think_open;
    prep.parse_tool_calls = tools;
    prep.id               = "test";
    prep.model            = "stub";
    prep.req.stream_include_usage = false;
    prep.req.chat_template_kwargs = kwargs;
    return prep;
}

json message_of(std::vector<std::string> pieces, bool think_open, const json& kwargs, bool tools = false) {
    ScriptedBackend backend(std::move(pieces));
    api::Context    ctx;
    ctx.backend = &backend;
    const api::HttpResult r = api::run_chat(ctx, make_prep(think_open, kwargs, tools), 0);
    return r.body["choices"][0]["message"];
}

json finish_of(std::vector<std::string> pieces, bool think_open, const json& kwargs) {
    ScriptedBackend backend(std::move(pieces));
    api::Context    ctx;
    ctx.backend = &backend;
    return api::run_chat(ctx, make_prep(think_open, kwargs), 0).body["choices"][0]["finish_reason"];
}

std::vector<json> stream_of(std::vector<std::string> pieces, bool think_open, const json& kwargs,
                            bool tools = false, FinishReason reason = FinishReason::Stop) {
    ScriptedBackend backend(std::move(pieces), reason);
    api::Context    ctx;
    ctx.backend = &backend;
    const api::PreparedChat prep = make_prep(think_open, kwargs, tools);
    std::vector<std::string> frames;
    api::SseWriter write = [&](std::string_view f) { frames.push_back(std::string(f)); return true; };
    api::stream_chat(ctx, prep, 0, write);
    std::vector<json> objs;
    for (const std::string& f : frames) {
        if (f.rfind("data: ", 0) != 0) continue;
        std::string payload = f.substr(6);
        while (!payload.empty() && payload.back() == '\n') payload.pop_back();
        if (payload == "[DONE]") continue;
        objs.push_back(json::parse(payload));
    }
    return objs;
}

std::vector<std::string> deltas_of(const std::vector<json>& objs, const char* key) {
    std::vector<std::string> out;
    for (const json& o : objs)
        for (const json& c : o["choices"])
            if (c.contains("delta") && c["delta"].contains(key)) out.push_back(c["delta"][key].get<std::string>());
    return out;
}

std::string finish_of_stream(const std::vector<json>& objs) {
    for (const json& o : objs)
        for (const json& c : o["choices"])
            if (c.contains("finish_reason") && !c["finish_reason"].is_null())
                return c["finish_reason"].get<std::string>();
    return "<none>";
}

size_t tool_call_deltas(const std::vector<json>& objs) {
    size_t n = 0;
    for (const json& o : objs)
        for (const json& c : o["choices"])
            if (c.contains("delta") && c["delta"].contains("tool_calls")) n += c["delta"]["tool_calls"].size();
    return n;
}

const std::string kCall =
    "<tool_call>\n<function=f>\n<parameter=n>\n1\n</parameter>\n</function>\n</tool_call>";

std::string closer() { return std::string(markers::kThinkClose); }

}  // namespace

// ------------------------------------------------------------ the pure decision
TEST(force_content_trigger_is_the_json_boolean_true) {
    CHECK(force_nonempty_content_requested(kTrue));
    CHECK(!force_nonempty_content_requested(kFalse));
    CHECK(!force_nonempty_content_requested(json::object()));
    CHECK(!force_nonempty_content_requested(json(nullptr)));
    CHECK(!force_nonempty_content_requested(json{{"force_nonempty_content", "true"}}));
    CHECK(!force_nonempty_content_requested(json{{"force_nonempty_content", 1}}));
    CHECK(!force_nonempty_content_requested(json{{"force_nonempty_content", nullptr}}));
    CHECK(!force_nonempty_content_requested(json{{"other", true}}));
}

TEST(force_content_promotes_only_empty_or_blank_content_with_reasoning) {
    CHECK(promote_reasoning_to_content(kTrue, "thinking", ""));
    CHECK(promote_reasoning_to_content(kTrue, "thinking", "   \n\t\r\v\f "));
    // real content: nothing changes
    CHECK(!promote_reasoning_to_content(kTrue, "thinking", "answer"));
    CHECK(!promote_reasoning_to_content(kTrue, "thinking", " a "));
    // nothing to promote
    CHECK(!promote_reasoning_to_content(kTrue, "", ""));
    CHECK(!promote_reasoning_to_content(kTrue, "", "  "));
    // blank reasoning is nothing to promote either (R1)
    CHECK(!promote_reasoning_to_content(kTrue, "\n\n", ""));
    CHECK(!promote_reasoning_to_content(kTrue, "   ", ""));
    CHECK(!promote_reasoning_to_content(kTrue, " \t\r\n ", "  "));
    // no kwarg, false, "true", 1
    CHECK(!promote_reasoning_to_content(json::object(), "thinking", ""));
    CHECK(!promote_reasoning_to_content(kFalse, "thinking", ""));
    CHECK(!promote_reasoning_to_content(json{{"force_nonempty_content", "true"}}, "thinking", ""));
    CHECK(!promote_reasoning_to_content(json{{"force_nonempty_content", 1}}, "thinking", ""));
}

// ------------------------------------------------------------- non-streaming
TEST(force_content_nonstream_unclosed_block_becomes_the_content) {
    const json msg = message_of({"still thinking about it"}, true, kTrue);
    CHECK_EQ(msg["content"].get<std::string>(), std::string("still thinking about it"));
    CHECK(!msg.contains("reasoning_content"));   // empty reasoning is not emitted
    CHECK(!msg.contains("reasoning_closed"));
    CHECK_EQ(finish_of({"still thinking about it"}, true, kTrue).get<std::string>(), std::string("stop"));
}

TEST(force_content_nonstream_closed_block_with_blank_content_is_swapped) {
    const json msg = message_of({"my reasoning", closer(), " \t "}, true, kTrue);
    CHECK_EQ(msg["content"].get<std::string>(), std::string("my reasoning"));
    CHECK(!msg.contains("reasoning_content"));   // the old content was blank: nothing to report as reasoning
    // empty content after a closed block
    const json msg2 = message_of({"my reasoning", closer()}, true, kTrue);
    CHECK_EQ(msg2["content"].get<std::string>(), std::string("my reasoning"));
    CHECK(!msg2.contains("reasoning_content"));
}

TEST(force_content_nonstream_without_the_trigger_nothing_changes) {
    for (const json& kw : {json::object(), kFalse, json{{"force_nonempty_content", "true"}},
                           json{{"force_nonempty_content", 1}}}) {
        const json msg = message_of({"still thinking about it"}, true, kw);
        CHECK_EQ(msg["content"].get<std::string>(), std::string(""));
        CHECK_EQ(msg["reasoning_content"].get<std::string>(), std::string("still thinking about it"));
        CHECK(!msg.value("reasoning_closed", true));
    }
}

TEST(force_content_nonstream_real_content_is_unchanged) {
    const json msg = message_of({"why", closer(), "the answer"}, true, kTrue);
    CHECK_EQ(msg["content"].get<std::string>(), std::string("the answer"));
    CHECK_EQ(msg["reasoning_content"].get<std::string>(), std::string("why"));
    CHECK(msg.value("reasoning_closed", false));
}

TEST(force_content_nonstream_a_plain_answer_without_a_block_is_unchanged) {
    const json msg = message_of({"just an answer"}, false, kTrue);
    CHECK_EQ(msg["content"].get<std::string>(), std::string("just an answer"));
    CHECK(!msg.contains("reasoning_content"));
}

TEST(force_content_nonstream_a_tool_call_in_the_content_blocks_the_swap) {
    const std::string call = "<tool_call>\n<function=f>\n<parameter=n>\n1\n</parameter>\n</function>\n</tool_call>";
    const json msg = message_of({"planning", closer(), call}, true, kTrue, /*tools=*/true);
    CHECK(msg.contains("tool_calls"));
    CHECK_EQ(msg["tool_calls"].size(), static_cast<size_t>(1));
    // the reasoning stays reasoning: the content was not blank
    CHECK_EQ(msg["reasoning_content"].get<std::string>(), std::string("planning"));
    CHECK(msg["content"].is_null());
}

// The reference's order: the tool-call parse sees the promoted content. A call
// written inside a block that never closed is therefore parsed (vLLM's
// parse(): extract_reasoning, then _extract_tool_calls).
TEST(force_content_nonstream_the_tool_parse_sees_the_promoted_content) {
    const std::string call = "<tool_call>\n<function=f>\n<parameter=n>\n1\n</parameter>\n</function>\n</tool_call>";
    const json msg = message_of({"planning ", call}, true, kTrue, /*tools=*/true);
    CHECK(msg.contains("tool_calls"));
    if (!msg.contains("tool_calls")) return;
    CHECK_EQ(msg["tool_calls"][0]["function"]["name"].get<std::string>(), std::string("f"));
    // ... and without the trigger the same turn has no call
    const json off = message_of({"planning ", call}, true, json::object(), /*tools=*/true);
    CHECK(!off.contains("tool_calls"));
}

// R1: blank reasoning never becomes content, on every path.
TEST(force_content_nonstream_newline_only_reasoning_is_not_promoted) {
    const json msg = message_of({"\n\n"}, true, kTrue);
    CHECK_EQ(msg["content"].get<std::string>(), std::string(""));
}

TEST(force_content_nonstream_spaces_only_reasoning_is_not_promoted) {
    const json msg = message_of({"   "}, true, kTrue);
    CHECK_EQ(msg["content"].get<std::string>(), std::string(""));
}

// --------------------------------------------------------------- streaming
TEST(force_content_stream_unclosed_block_sends_the_reasoning_once_as_content) {
    const std::vector<json> objs = stream_of({"part one, ", "part two"}, true, kTrue);
    const auto content   = deltas_of(objs, "content");
    const auto reasoning = deltas_of(objs, "reasoning_content");
    CHECK_EQ(content.size(), static_cast<size_t>(1));
    if (content.size() != 1) return;
    CHECK_EQ(content[0], std::string("part one, part two"));
    std::string joined;
    for (const std::string& r : reasoning) joined += r;
    CHECK_EQ(joined, std::string("part one, part two"));   // the reasoning deltas stay
    // the content comes before the final chunk, and the final chunk is the last data frame
    const json& last = objs.back()["choices"][0];
    CHECK(!last["finish_reason"].is_null());
    CHECK(!last["delta"].contains("content"));
    CHECK(!last["delta"].value("reasoning_closed", true));
}

TEST(force_content_stream_blank_reasoning_is_not_promoted) {
    for (const char* blank : {"\n\n", "   "}) {
        const std::vector<json> objs = stream_of({blank}, true, kTrue);
        CHECK(deltas_of(objs, "content").empty());
        CHECK(finish_of_stream(objs) != "<none>");   // the stream still ends with its final chunk
    }
}

TEST(force_content_stream_closed_block_sends_no_extra_content) {
    // real content: only the answer
    auto objs = stream_of({"why", closer(), "the answer"}, true, kTrue);
    const auto answer = deltas_of(objs, "content");
    CHECK_EQ(answer.size(), static_cast<size_t>(1));
    if (answer.size() == 1) CHECK_EQ(answer[0], std::string("the answer"));
    // blank content after a closed block: the reference does not promote on the stream
    objs = stream_of({"why", closer(), "  "}, true, kTrue);
    for (const std::string& c : deltas_of(objs, "content")) CHECK(c.find("why") == std::string::npos);
}

TEST(force_content_stream_without_the_trigger_nothing_is_added) {
    for (const json& kw : {json::object(), kFalse, json{{"force_nonempty_content", "true"}}}) {
        const auto objs = stream_of({"part one, ", "part two"}, true, kw);
        CHECK(deltas_of(objs, "content").empty());
    }
}

TEST(force_content_stream_multibyte_reasoning_arrives_intact) {
    // a 2-, 3- and 4-byte code point torn across pieces
    const std::string text = "x \xc3\xa9 y \xe2\x87\xa9 z \xf0\x9f\xa7\xa9 w";
    const std::vector<json> objs =
        stream_of({text.substr(0, 3), text.substr(3, 5), text.substr(8, 8), text.substr(16)}, true, kTrue);
    const auto content = deltas_of(objs, "content");
    std::string joined;
    for (const std::string& c : content) {
        CHECK(utf8::is_valid(c));
        joined += c;
    }
    CHECK_EQ(joined, text);
}

TEST(force_content_stream_a_plain_answer_without_a_block_gets_nothing_extra) {
    const auto objs = stream_of({"just an answer"}, false, kTrue);
    const auto content = deltas_of(objs, "content");
    CHECK_EQ(content.size(), static_cast<size_t>(1));
    if (content.size() == 1) CHECK_EQ(content[0], std::string("just an answer"));
}

// The stream does not tool-parse the promoted text (vLLM's finalize_generation
// appends it to the delta's content and runs no tool parser; operator,
// 2026-11-11: a call the model quoted inside a thinking block that was cut must
// not come back as a call). The markup stays text in the one content delta,
// there is no tool_calls delta, and the finish reason is the generation's own.
TEST(force_content_stream_a_tool_call_inside_the_unclosed_block_stays_text) {
    const std::vector<json> objs = stream_of({"planning ", kCall}, true, kTrue, /*tools=*/true);
    CHECK_EQ(tool_call_deltas(objs), static_cast<size_t>(0));
    CHECK_EQ(finish_of_stream(objs), std::string("stop"));
    const auto deltas = deltas_of(objs, "content");
    CHECK_EQ(deltas.size(), static_cast<size_t>(1));
    if (deltas.size() == 1) CHECK_EQ(deltas[0], "planning " + kCall);
}

TEST(force_content_stream_only_a_call_in_the_unclosed_block_stays_text) {
    const std::vector<json> objs = stream_of({kCall}, true, kTrue, /*tools=*/true);
    CHECK_EQ(tool_call_deltas(objs), static_cast<size_t>(0));
    CHECK_EQ(finish_of_stream(objs), std::string("stop"));
    const auto deltas = deltas_of(objs, "content");
    CHECK_EQ(deltas.size(), static_cast<size_t>(1));
    if (deltas.size() == 1) CHECK_EQ(deltas[0], kCall);
}

// The served shape (the card's coding-agent turn cut by the token budget inside
// its thinking): the generation ends on Length, tools are declared, the block
// holds call markup. The finish reason stays "length"; with the promoted text
// parsed it was "tool_calls".
TEST(force_content_stream_a_call_in_a_block_cut_by_length_keeps_the_length_finish) {
    const std::vector<json> objs =
        stream_of({"planning ", kCall}, true, kTrue, /*tools=*/true, FinishReason::Length);
    CHECK_EQ(tool_call_deltas(objs), static_cast<size_t>(0));
    CHECK_EQ(finish_of_stream(objs), std::string("length"));
    const auto deltas = deltas_of(objs, "content");
    CHECK_EQ(deltas.size(), static_cast<size_t>(1));
    if (deltas.size() == 1) CHECK_EQ(deltas[0], "planning " + kCall);
}

// The control: a call in the answer after a block that closed is still a call on
// the stream, with the kwarg set (the answer's own tool parse is untouched).
TEST(force_content_stream_a_tool_call_after_a_closed_block_is_still_a_call) {
    const std::vector<json> objs = stream_of({"planning", closer(), kCall}, true, kTrue, /*tools=*/true);
    CHECK_EQ(tool_call_deltas(objs), static_cast<size_t>(1));
    CHECK_EQ(finish_of_stream(objs), std::string("tool_calls"));
    for (const std::string& c : deltas_of(objs, "content")) CHECK(c.find("<tool_call>") == std::string::npos);
}

TEST(force_content_stream_tools_declared_without_markup_promotes_as_before) {
    const std::vector<json> objs = stream_of({"part one, ", "part two"}, true, kTrue, /*tools=*/true);
    const auto content = deltas_of(objs, "content");
    CHECK_EQ(content.size(), static_cast<size_t>(1));
    if (content.size() == 1) CHECK_EQ(content[0], std::string("part one, part two"));
    CHECK_EQ(tool_call_deltas(objs), static_cast<size_t>(0));
    CHECK_EQ(finish_of_stream(objs), std::string("stop"));
}

TEST(force_content_stream_markup_without_tools_declared_is_promoted_verbatim) {
    const std::vector<json> objs = stream_of({"planning ", kCall}, true, kTrue, /*tools=*/false);
    CHECK_EQ(tool_call_deltas(objs), static_cast<size_t>(0));
    std::string content;
    for (const std::string& c : deltas_of(objs, "content")) content += c;
    CHECK_EQ(content, "planning " + kCall);
    CHECK_EQ(finish_of_stream(objs), std::string("stop"));
}

// The unit default (--chat-template-kwarg force_nonempty_content=true) end to end:
// prepare_chat adds it to the request, the promotion fires on that one request,
// and the request's own false cancels it.
namespace {
json answer_with_unit_default(const json& request_kwargs) {
    ScriptedBackend backend({"still thinking about it"});
    Config          cfg;
    cfg.stub                         = true;
    cfg.chat_template_kwarg_defaults = {{"force_nonempty_content", "true"}};
    api::SlotPool slots{1};
    api::Context  ctx;
    ctx.cfg     = &cfg;
    ctx.backend = &backend;
    ctx.slots   = &slots;
    json body{{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}, {"max_tokens", 16}};
    if (!request_kwargs.is_null()) body["chat_template_kwargs"] = request_kwargs;
    api::PreparedChat prep;
    const auto        err = api::prepare_chat(ctx, body, prep);
    CHECK(!err.has_value());
    prep.think_open = true;   // the scripted prompt is not rendered by a template
    return api::run_chat(ctx, prep, 0).body["choices"][0]["message"];
}
}  // namespace

TEST(force_content_unit_default_promotes_one_request_and_the_requests_false_cancels_it) {
    const json on = answer_with_unit_default(json(nullptr));
    CHECK_EQ(on["content"].get<std::string>(), std::string("still thinking about it"));
    const json off = answer_with_unit_default(kFalse);
    CHECK_EQ(off["content"].get<std::string>(), std::string(""));
    CHECK_EQ(off["reasoning_content"].get<std::string>(), std::string("still thinking about it"));
}
