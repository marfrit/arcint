#include "harness.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "api/handlers.h"
#include "api/raw_trace.h"
#include "api/responses.h"
#include "api/responses_stream.h"
#include "core/markers.h"
#include "core/toolcall.h"
#include "exec/backend.h"

// ARCINT_RAW_TRACE: one JSON line a turn with the text the model generated
// before the split and the parse, and what the parse made of it. Off, nothing
// changes; on, the output is still the same bytes. Every marker comes from
// core/markers.h or tool_call_tags().
using namespace lgc;
using nlohmann::json;

namespace {

constexpr int kOpenId = 248068, kCloseId = 248069, kCallOpenId = 248058, kCallCloseId = 248059, kEogId = 248046;

std::string s(std::string_view v) { return std::string(v); }
std::string head() {
    return s(markers::kTurnStart) + "user\nhi" + s(markers::kTurnEnd) + "\n" + s(markers::kTurnStart) + "assistant\n";
}
std::string prompt_off() { return head() + s(markers::kThinkOpen) + "\n\n" + s(markers::kThinkClose) + "\n\n"; }
std::string prompt_on() { return head() + s(markers::kThinkOpen) + "\n"; }

using Script = std::vector<std::pair<std::string, int>>;

class TraceBackend final : public Backend {
public:
    TraceBackend(std::string prompt, Script script) : prompt_(std::move(prompt)), script_(std::move(script)) {
        status_.id          = "stub";
        status_.served_id   = "stub";
        status_.n_ctx_train = kStubNCtxTrain;
        status_.loaded      = true;
        status_.stub        = true;
        status_.n_ctx       = 4096;
        status_.dialect     = ChatDialect::Qwen;
    }
    const ModelStatus& status() const override { return status_; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest&) const override { return prompt_; }
    ThinkBlockIds      think_block_ids(ChatDialect) const override {
        ThinkBlockIds ids;
        ids.open_close_ids = {{kOpenId, kCloseId}};
        ids.open_ids       = {kOpenId};
        ids.close_ids      = {kCloseId};
        ids.open_tags      = {s(markers::kThinkOpen)};
        ids.close_tags     = {s(markers::kThinkClose)};
        ids.eog            = {kEogId};
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
        int              eos_id() const override { return kEogId; }
    } tok_;
    std::string prompt_;
    Script      script_;
    ModelStatus status_;
};

json chat_tools() {
    return json::array({{{"type", "function"},
                         {"function",
                          {{"name", "classify"},
                           {"parameters",
                            {{"type", "object"},
                             {"properties", {{"class", {{"type", "string"}}}}},
                             {"required", json::array({"class"})}}}}}}});
}

struct Fixture {
    Config        cfg;
    TraceBackend  backend;
    api::SlotPool slots{1};
    api::Context  ctx;
    Fixture(std::string prompt, Script script) : backend(std::move(prompt), std::move(script)) {
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
    }
    api::PreparedChat prepare(bool thinking, bool tools) {
        json b = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                  {"chat_template_kwargs", {{"enable_thinking", thinking}}}};
        if (tools) b["tools"] = chat_tools();
        api::PreparedChat prep;
        CHECK(!api::prepare_chat(ctx, b, prep).has_value());
        return prep;
    }
    std::string streamed(const api::PreparedChat& prep) {
        std::string frames;
        api::stream_chat(ctx, prep, 0, [&](std::string_view f) {
            frames.append(f);
            return true;
        });
        return frames;
    }
    api::PreparedResponse prepare_response(bool tools) {
        json b = {{"model", "stub"}, {"input", "classify it"}, {"reasoning", {{"effort", "none"}}},
                  {"max_output_tokens", 64}};
        if (tools)
            b["tools"] = json::array({{{"type", "function"},
                                       {"name", "classify"},
                                       {"parameters",
                                        {{"type", "object"}, {"properties", {{"class", {{"type", "string"}}}}}}}}});
        api::ResponseRequest r;
        CHECK(!api::parse_response_request(b, r).has_value());
        api::PreparedResponse p;
        CHECK(!api::prepare_response(ctx, std::move(r), p).has_value());
        return p;
    }
};

// One file a process (the writer opens it once): every case appends to it and
// reads its own lines back.
const std::string& trace_path() {
    static const std::string p =
        (std::filesystem::temp_directory_path() / ("arcint-raw-trace-" + std::to_string(::getpid()) + ".jsonl")).string();
    return p;
}
std::vector<std::string> trace_lines() {
    std::vector<std::string> out;
    std::ifstream            f(trace_path());
    for (std::string line; std::getline(f, line);) out.push_back(line);
    return out;
}
struct TraceOn {
    TraceOn() { ::setenv("ARCINT_RAW_TRACE", trace_path().c_str(), 1); }
    ~TraceOn() { ::unsetenv("ARCINT_RAW_TRACE"); }
};

// The malformed body the AREX-2 fine-tune is suspected of: a parameter opened
// inside another one's value, so the outer parameter's value is the inner
// opener's text.
std::string malformed_body() {
    const ToolCallTags& t = tool_call_tags();
    return "\n" + s(t.fn_open) + "classify>\n" + s(t.param_open) + "class>\n" + s(t.param_open) + "unclear>\n" +
           s(t.param_end) + "\n" + s(t.fn_close) + "\n";
}
Script malformed_call() {
    const ToolCallTags& t = tool_call_tags();
    return {{"Classifying.\n", 1}, {s(t.open), kCallOpenId}, {malformed_body(), 300}, {s(t.close), kCallCloseId}};
}
std::string concat(const Script& sc) {
    std::string out;
    for (const auto& [p, id] : sc) out += p;
    return out;
}

// the last line appended since `before` lines, parsed; null when none
json last_line_since(size_t before) {
    const std::vector<std::string> lines = trace_lines();
    if (lines.size() != before + 1) {
        t::fail(__FILE__, __LINE__,
                "expected one new trace line, got " + std::to_string(lines.size()) + " - " + std::to_string(before));
        return json();
    }
    return json::parse(lines.back());
}

void check_malformed_line(const json& l, const std::string& route, bool stream) {
    if (l.is_null()) return;
    CHECK_EQ(l["trace"].get<std::string>(), std::string("raw1"));
    CHECK_EQ(l["route"].get<std::string>(), route);
    CHECK_EQ(l["stream"].get<bool>(), stream);
    CHECK_EQ(l["think_start"].get<std::string>(), std::string("closed"));
    CHECK_EQ(l["finish"].get<std::string>(), std::string("stop"));
    // the generated text exactly, the malformed body in it
    CHECK_EQ(l["raw"].get<std::string>(), concat(malformed_call()));
    CHECK(l["raw"].get<std::string>().find(malformed_body()) != std::string::npos);
    CHECK(l["tool_parse"].get<bool>());
    CHECK_EQ(l["tool_calls"].size(), 1u);
    if (l["tool_calls"].size() != 1) return;
    CHECK_EQ(l["tool_calls"][0]["name"].get<std::string>(), std::string("classify"));
    // what the parser produced from it: the inner opener as the value
    const json args = json::parse(l["tool_calls"][0]["arguments"].get<std::string>());
    CHECK_EQ(args["class"].get<std::string>(), s(tool_call_tags().param_open) + "unclear>");
    CHECK_EQ(l["content"].get<std::string>(), std::string("Classifying.\n"));
}

}  // namespace

TEST(raw_trace_off_writes_nothing_and_changes_no_byte) {
    ::unsetenv("ARCINT_RAW_TRACE");
    CHECK(!api::raw_trace_on());
    const size_t before = trace_lines().size();
    Fixture      fx(prompt_off(), malformed_call());
    const api::PreparedChat prep = fx.prepare(false, true);
    const std::string body_off   = api::run_chat(fx.ctx, prep, 0).body.dump();
    const std::string sse_off    = fx.streamed(prep);
    CHECK_EQ(trace_lines().size(), before);   // off: no line, no file touched
    std::string body_on, sse_on;
    {
        TraceOn on;
        body_on = api::run_chat(fx.ctx, prep, 0).body.dump();
        sse_on  = fx.streamed(prep);
    }
    // the same prepared request: the response bytes do not depend on the switch
    CHECK_EQ(body_on, body_off);
    CHECK_EQ(sse_on, sse_off);
}

TEST(raw_trace_line_carries_the_raw_text_and_the_parsed_call) {
    TraceOn on;
    {
        Fixture      fx(prompt_off(), malformed_call());
        const size_t before = trace_lines().size();
        api::run_chat(fx.ctx, fx.prepare(false, true), 0);
        check_malformed_line(last_line_since(before), "chat", false);
    }
    {
        Fixture      fx(prompt_off(), malformed_call());
        const size_t before = trace_lines().size();
        fx.streamed(fx.prepare(false, true));
        check_malformed_line(last_line_since(before), "chat", true);
    }
    {
        Fixture               fx(prompt_off(), malformed_call());
        const size_t          before = trace_lines().size();
        api::PreparedResponse p      = fx.prepare_response(true);
        api::run_response(fx.ctx, p, 0);
        check_malformed_line(last_line_since(before), "responses", false);
    }
    {
        Fixture               fx(prompt_off(), malformed_call());
        const size_t          before = trace_lines().size();
        api::PreparedResponse p      = fx.prepare_response(true);
        api::stream_response(fx.ctx, p, 0, [](std::string_view) { return true; });
        check_malformed_line(last_line_since(before), "responses", true);
    }
}

TEST(raw_trace_line_keeps_the_reasoning_and_the_closer_in_raw) {
    // thinking on, no tools: raw is everything before the split, the closer's
    // text included; reasoning and content are the split's
    TraceOn      on;
    const Script sc = {{"weigh it", 5}, {s(markers::kThinkClose), kCloseId}, {"\n\n", 271}, {"Paris.", 9}};
    for (bool stream : {false, true}) {
        Fixture      fx(prompt_on(), sc);
        const size_t before = trace_lines().size();
        const api::PreparedChat prep = fx.prepare(true, false);
        if (stream) fx.streamed(prep);
        else api::run_chat(fx.ctx, prep, 0);
        const json l = last_line_since(before);
        if (l.is_null()) continue;
        CHECK_EQ(l["raw"].get<std::string>(), concat(sc));
        CHECK_EQ(l["think_start"].get<std::string>(), std::string("open"));
        CHECK_EQ(l["reasoning"].get<std::string>(), std::string("weigh it"));
        CHECK_EQ(l["content"].get<std::string>(), std::string("Paris."));
        CHECK(!l["tool_parse"].get<bool>());
        CHECK(l["tool_calls"].empty());
    }
}

TEST(raw_trace_line_format_is_fixed) {
    const std::string     raw = "a\xff";   // a turn cut inside a code point
    std::vector<ToolCall> calls(1);
    calls[0].name      = "f";
    calls[0].arguments = "{\"x\":1}";
    api::RawTraceRecord r;
    r.route       = "chat";
    r.request_id  = "rid";
    r.finish      = "length";
    r.think_start = ThinkStart::Closed;
    r.raw         = &raw;
    r.tool_parse  = true;
    r.calls       = &calls;
    r.content     = "c";
    CHECK_EQ(api::raw_trace_line(r, 3),
             std::string("{\"trace\":\"raw1\",\"n\":3,\"request_id\":\"rid\",\"route\":\"chat\",\"stream\":false,"
                         "\"token_mode\":false,\"finish\":\"length\",\"think_start\":\"closed\",\"raw\":\"a\xef\xbf\xbd\","
                         "\"reasoning\":\"\",\"tool_parse\":true,\"tool_calls\":[{\"name\":\"f\","
                         "\"arguments\":\"{\\\"x\\\":1}\"}],\"content\":\"c\"}"));
}
