#include "harness.h"

#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

#include "api/handlers.h"
#include "core/reasoning.h"
#include "exec/backend.h"

// The think block is decided by token id (the closer is one special token),
// and a request's stop strings hold text back between the backend and that
// decision. A held tail released together with the closer's piece must not
// cost the closer its id: when it did, a stop string beginning with a newline
// (the shape agent frameworks send, a role prefix after a line break) kept the
// block open for the whole turn, so the answer came back as reasoning and the
// content was empty. These cases run the chat path end to end, both the
// non-streaming and the streaming handler, against a backend that emits an
// exact (piece, id) script.
namespace {

using json = nlohmann::json;

constexpr int kOpenId  = 6;
constexpr int kCloseId = 7;

const std::string& opener() { return lgc::think_tag_pairs(false)[0].first; }
const std::string& closer() { return lgc::think_tag_pairs(false)[0].second; }

class IdScriptBackend final : public lgc::Backend {
public:
    explicit IdScriptBackend(std::vector<std::pair<std::string, int>> script) : script_(std::move(script)) {
        status_.id          = "stub";
        status_.served_id   = "stub";
        status_.n_ctx_train = lgc::kStubNCtxTrain;
        status_.loaded      = true;
        status_.stub        = true;
        status_.n_ctx       = 4096;
    }

    const lgc::ModelStatus& status() const override { return status_; }
    lgc::Tokenizer&         tokenizer() override { return tok_; }
    std::string             render_chat(const lgc::ChatRequest&) const override { return "prompt"; }

    lgc::FinishReason generate(const lgc::GenerationInput&, int, const lgc::TokenCallback& on_piece,
                               lgc::GenerationStats& stats) override {
        for (const auto& [piece, id] : script_) {
            ++stats.completion_tokens;
            if (on_piece(piece, id) != lgc::Control::Continue) return lgc::FinishReason::Stop;
        }
        return lgc::FinishReason::Stop;
    }

private:
    struct Tok final : public lgc::Tokenizer {
        std::vector<int> encode(std::string_view) override { return {}; }
        std::string      decode(const std::vector<int>&) override { return {}; }
        std::string      decode_one(int) override { return {}; }
        int              eos_id() const override { return 0; }
    } tok_;

    std::vector<std::pair<std::string, int>> script_;
    lgc::ModelStatus                         status_;
};

lgc::api::PreparedChat prepared(bool think_open, std::vector<std::string> stop) {
    lgc::api::PreparedChat prep;
    prep.think_open                  = think_open;
    prep.think_tags_extended         = false;
    prep.parse_tool_calls            = false;
    prep.id                          = "test";
    prep.model                       = "stub";
    prep.req.stream_include_usage    = false;
    prep.think_ids.open_close_ids    = {{kOpenId, kCloseId}};
    prep.input.sampler.stop          = std::move(stop);
    return prep;
}

struct Answer {
    std::string reasoning, content;
    bool        closed = false;
};

Answer non_streamed(lgc::Backend& backend, const lgc::api::PreparedChat& prep) {
    lgc::api::Context ctx;
    ctx.backend = &backend;
    const lgc::api::HttpResult r = lgc::api::run_chat(ctx, prep, 0);
    const json& msg = r.body["choices"][0]["message"];
    Answer a;
    if (msg.contains("reasoning_content")) a.reasoning = msg["reasoning_content"].get<std::string>();
    if (msg.contains("reasoning_closed")) a.closed = msg["reasoning_closed"].get<bool>();
    if (msg["content"].is_string()) a.content = msg["content"].get<std::string>();
    return a;
}

Answer streamed(lgc::Backend& backend, const lgc::api::PreparedChat& prep) {
    lgc::api::Context ctx;
    ctx.backend = &backend;
    Answer                   a;
    lgc::api::SseWriter write = [&](std::string_view frame) {
        if (frame.rfind("data: ", 0) != 0) return true;
        std::string payload(frame.substr(6));
        while (!payload.empty() && payload.back() == '\n') payload.pop_back();
        if (payload == "[DONE]") return true;
        const json o = json::parse(payload);
        if (!o.contains("choices")) return true;
        for (const json& c : o["choices"]) {
            if (!c.contains("delta")) continue;
            if (c["delta"].contains("reasoning_content"))
                a.reasoning += c["delta"]["reasoning_content"].get<std::string>();
            if (c["delta"].contains("content")) a.content += c["delta"]["content"].get<std::string>();
        }
        return true;
    };
    lgc::api::stream_chat(ctx, prep, 0, write);
    return a;
}

std::string trimmed(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == '\n' || s[b] == ' ')) ++b;
    while (e > b && (s[e - 1] == '\n' || s[e - 1] == ' ')) --e;
    return s.substr(b, e - b);
}

// The reasoning ends on a line break, then the closer token, then the answer,
// then the role prefix the stop string waits for.
std::vector<std::pair<std::string, int>> template_opened_turn() {
    return {{"Let", 10},   {" me", 11},     {" think", 12}, {".", 13},  {"\n", 14},  {closer(), kCloseId},
            {"\n\n", 15},  {"Hello", 16},   {"\n", 14},     {"User", 17}, {":", 18}, {" more", 19}};
}

}  // namespace

TEST(stop_with_a_leading_newline_keeps_the_closer_id_nonstreaming) {
    IdScriptBackend backend(template_opened_turn());
    const Answer    a = non_streamed(backend, prepared(true, {"\nUser:"}));
    CHECK(a.closed);
    CHECK_EQ(a.reasoning, std::string("Let me think."));
    CHECK_EQ(a.content, std::string("Hello"));
}

TEST(stop_with_a_leading_newline_keeps_the_closer_id_streaming) {
    IdScriptBackend backend(template_opened_turn());
    const Answer    a = streamed(backend, prepared(true, {"\nUser:"}));
    CHECK_EQ(trimmed(a.reasoning), std::string("Let me think."));
    CHECK_EQ(trimmed(a.content), std::string("Hello"));
}

// The same turn without a stop string: the reference both arms must match.
TEST(stop_absent_gives_the_same_split) {
    IdScriptBackend backend(template_opened_turn());
    const Answer    a = non_streamed(backend, prepared(true, {}));
    CHECK(a.closed);
    CHECK_EQ(a.reasoning, std::string("Let me think."));
    CHECK_EQ(trimmed(a.content), std::string("Hello\nUser: more"));
}

// A stop string that begins with the closer's own text holds the whole closer
// piece back; released with the next piece, it still carries its id.
TEST(stop_beginning_with_the_closer_text_keeps_the_closer_id) {
    IdScriptBackend backend(template_opened_turn());
    const std::vector<std::string> stop = {closer() + "\n\nX"};
    const Answer n = non_streamed(backend, prepared(true, stop));
    CHECK(n.closed);
    CHECK_EQ(n.reasoning, std::string("Let me think."));
    CHECK_EQ(n.content, std::string("Hello\nUser: more"));
    const Answer s = streamed(backend, prepared(true, stop));
    CHECK_EQ(trimmed(s.content), std::string("Hello\nUser: more"));
}

// A self-opened block: the opener token arrives after a held line break.
// Without its id the opener's text leaked into content.
TEST(stop_with_a_leading_newline_keeps_the_opener_id) {
    IdScriptBackend backend({{"\n", 14}, {opener(), kOpenId}, {"Hmm", 20}, {"\n", 14}, {closer(), kCloseId},
                             {"Hi", 21}, {"\n", 14}, {"User", 17}, {":", 18}});
    const Answer n = non_streamed(backend, prepared(false, {"\nUser:"}));
    CHECK(n.closed);
    CHECK_EQ(n.reasoning, std::string("Hmm"));
    CHECK_EQ(n.content, std::string("Hi"));
    CHECK(n.content.find(opener()) == std::string::npos);
    const Answer s = streamed(backend, prepared(false, {"\nUser:"}));
    CHECK_EQ(trimmed(s.reasoning), std::string("Hmm"));
    CHECK_EQ(trimmed(s.content), std::string("Hi"));
}

// No block at all (the template did not open one, the model did not either):
// the content is the reply as generated, line breaks included, on both paths
// and as the text split gives it.
TEST(no_block_keeps_the_content_as_generated) {
    IdScriptBackend backend({{"\n", 14}, {"\n", 14}, {"Hello", 16}, {"\n", 14}, {"User", 17}, {":", 18}});
    const Answer n = non_streamed(backend, prepared(false, {"\nUser:"}));
    CHECK(!n.closed);
    CHECK_EQ(n.reasoning, std::string(""));
    CHECK_EQ(n.content, std::string("\n\nHello"));
    const Answer s = streamed(backend, prepared(false, {"\nUser:"}));
    CHECK_EQ(s.content, std::string("\n\nHello"));
    CHECK_EQ(lgc::split_reasoning("\n\nHello", false, false).content, n.content);
}

// A stop string that begins with the LAST bytes of the closer's text splits the
// closer's piece: the first part goes out without an id and the held tail
// carries it. Before the id rode on the last part, the tail reached the tracker
// with id -1, so the block stayed open and the answer came back as reasoning.
TEST(stop_beginning_with_the_closer_tail_keeps_the_closer_id) {
    // The stop never completes: it shares the closer's tail and then adds a
    // byte the reply does not contain.
    const std::vector<std::string> stop = {closer().substr(closer().size() - 3) + "\\x01"};
    IdScriptBackend backend(template_opened_turn());
    const Answer    n = non_streamed(backend, prepared(true, stop));
    CHECK(n.closed);
    // The answer is content, not reasoning: the block closes on the tail part's
    // id even though the stop string split the closer's piece.
    CHECK_EQ(n.content, std::string("Hello\nUser: more"));
    CHECK(n.reasoning.rfind("Let me think.", 0) == 0);
    CHECK(n.reasoning.find("Hello") == std::string::npos);
    const Answer s = streamed(backend, prepared(true, stop));
    CHECK_EQ(trimmed(s.content), std::string("Hello\nUser: more"));
}
