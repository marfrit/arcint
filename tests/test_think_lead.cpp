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

// The answer's lead after a think block that is already closed: by the template
// (thinking off, the Qwen generation prompt ends in an empty block) or by the
// model (thinking on). A fine-tune that repeats the closer there must not put
// it into `content`. Measured on a Qwen3.8-27B fine-tune with thinking off: its
// content began with the closer and two line breaks before the needle.
//
// Every case runs the full route: prepare_chat renders the prompt (the script
// backend returns the generation prompt the real template renders), and the
// non-streamed and the streamed path read the same scripted (piece, id)
// sequence. Every marker comes from core/markers.h or tool_call_tags().
using namespace lgc;
using nlohmann::json;

namespace {

// The Qwen3.8 vocabulary's ids (code: the GGUF's token table; the coder's and
// the fine-tune's are the same): both think tags and both call tags are
// single added tokens.
constexpr int kOpenId      = 248068;
constexpr int kCloseId     = 248069;
constexpr int kCallOpenId  = 248058;
constexpr int kCallCloseId = 248059;
constexpr int kEogId       = 248046;

std::string open_tag() { return std::string(markers::kThinkOpen); }
std::string close_tag() { return std::string(markers::kThinkClose); }
std::string s(std::string_view v) { return std::string(v); }

// The generation prompts the Qwen3.8 / Qwen3.6 coder templates render (code:
// their add_generation_prompt branch), and Dory's thinking-off form.
std::string head() { return s(markers::kTurnStart) + "user\nhi" + s(markers::kTurnEnd) + "\n" + s(markers::kTurnStart) + "assistant\n"; }
std::string qwen_prompt_off() { return head() + open_tag() + "\n\n" + close_tag() + "\n\n"; }
std::string qwen_prompt_on() { return head() + open_tag() + "\n"; }
std::string dory_prompt_off() { return head() + open_tag() + close_tag(); }

using Script = std::vector<std::pair<std::string, int>>;

class LeadBackend final : public Backend {
public:
    LeadBackend(std::string prompt, Script script) : prompt_(std::move(prompt)), script_(std::move(script)) {
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
        ids.open_tags      = {open_tag()};
        ids.close_tags     = {close_tag()};
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
    std::string  prompt_;
    Script       script_;
    ModelStatus  status_;
};

json sql_tool() {
    return json::array({{{"type", "function"},
                         {"function",
                          {{"name", "run_query"},
                           {"parameters",
                            {{"type", "object"},
                             {"properties", {{"sql", {{"type", "string"}}}}},
                             {"required", json::array({"sql"})}}}}}}});
}

json body(bool thinking, bool tools) {
    json b = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
              {"chat_template_kwargs", {{"enable_thinking", thinking}}}};
    if (tools) b["tools"] = sql_tool();
    return b;
}

struct Out {
    std::string reasoning, content;
    json        calls = json::array();
};

struct Fixture {
    Config        cfg;
    LeadBackend   backend;
    api::SlotPool slots{1};
    api::Context  ctx;
    Fixture(std::string prompt, Script script) : backend(std::move(prompt), std::move(script)) {
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
    }
    api::PreparedChat prepare(bool thinking, bool tools) {
        api::PreparedChat prep;
        const auto        err = api::prepare_chat(ctx, body(thinking, tools), prep);
        CHECK(!err.has_value());
        return prep;
    }
    Out non_streamed(bool thinking, bool tools = false) {
        const api::PreparedChat prep = prepare(thinking, tools);
        const api::HttpResult   r    = api::run_chat(ctx, prep, 0);
        Out                     o;
        CHECK_EQ(r.status, 200);
        if (r.status != 200) return o;
        const json& msg = r.body["choices"][0]["message"];
        if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string())
            o.reasoning = msg["reasoning_content"].get<std::string>();
        if (msg["content"].is_string()) o.content = msg["content"].get<std::string>();
        if (msg.contains("tool_calls")) o.calls = msg["tool_calls"];
        return o;
    }
    Out streamed(bool thinking, bool tools = false) {
        const api::PreparedChat prep = prepare(thinking, tools);
        Out                     o;
        api::SseWriter          write = [&](std::string_view frame) {
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
};

// Both paths, one expectation: the streamed deltas add up to the message.
void expect_both(const std::string& prompt, const Script& script, bool thinking, const std::string& reasoning,
                 const std::string& content) {
    Fixture    a(prompt, script);
    const Out  n = a.non_streamed(thinking);
    CHECK_EQ(n.reasoning, reasoning);
    CHECK_EQ(n.content, content);
    Fixture    b(prompt, script);
    const Out  st = b.streamed(thinking);
    CHECK_EQ(st.reasoning, reasoning);
    CHECK_EQ(st.content, content);
}

std::string sql() { return "SELECT * FROM lines WHERE amount < 0.01 AND qty <> 0"; }

// The call the coder template's form spells, as pieces with the call tags as
// their single ids and the body as ordinary text, split right after a `<`.
Script call_pieces(const std::string& value) {
    const ToolCallTags& t = tool_call_tags();
    const std::string   body =
        "\n" + s(t.fn_open) + "run_query>\n" + s(t.param_open) + "sql>\n" + value + "\n" + s(t.param_end) + "\n" + s(t.fn_close) + "\n";
    const size_t lt = body.find('<', body.find("sql>\n") + 5);
    Script       out;
    out.emplace_back(s(t.open), kCallOpenId);
    if (lt != std::string::npos) {
        out.emplace_back(body.substr(0, lt + 1), 300);
        out.emplace_back(body.substr(lt + 1), 301);
    } else {
        out.emplace_back(body, 300);
    }
    out.emplace_back(s(t.close), kCallCloseId);
    return out;
}

std::string sql_of(const json& call) {
    return json::parse(call["function"]["arguments"].get<std::string>())["sql"].get<std::string>();
}

}  // namespace

// ---- thinking off: the template closed an empty block ----------------------

TEST(think_lead_off_a_repeated_closer_token_is_not_content) {
    // the measured case: the closer as its one token, two line breaks, the needle
    expect_both(qwen_prompt_off(), {{close_tag(), kCloseId}, {"\n\n", 271}, {"ORCHID-LANTERN-77", 1000}}, false, "",
                "ORCHID-LANTERN-77");
}

TEST(think_lead_off_a_repeated_closer_spelled_in_two_pieces) {
    // the same closer from ordinary tokens, split inside the tag
    const std::string c = close_tag();
    expect_both(qwen_prompt_off(), {{c.substr(0, 2), 30}, {c.substr(2), 31}, {"\n\n", 271}, {"ORCHID-LANTERN-77", 1000}},
                false, "", "ORCHID-LANTERN-77");
}

TEST(think_lead_off_a_repeated_closer_glued_to_the_answer) {
    // one text piece carrying the closer, the layout and the answer (no ids)
    expect_both(qwen_prompt_off(), {{close_tag() + "\n\nORCHID-LANTERN-77", -1}}, false, "", "ORCHID-LANTERN-77");
}

TEST(think_lead_off_dory_prompt_without_line_breaks) {
    // Dory's thinking-off generation prompt closes the block with no layout
    expect_both(dory_prompt_off(), {{close_tag(), kCloseId}, {"Paris.", 12}}, false, "", "Paris.");
}

TEST(think_lead_off_clean_answers_are_unchanged) {
    // what the base model writes: nothing to drop, the bytes as generated
    expect_both(qwen_prompt_off(), {{"ORCHID-LANTERN-77", 1000}}, false, "", "ORCHID-LANTERN-77");
    // markup at the lead is held only while it could still be a closer
    expect_both(qwen_prompt_off(), {{"<", 60}, {"b>x</b> done", 61}}, false, "", "<b>x</b> done");
    // a block the model opens itself is still honoured
    expect_both(qwen_prompt_off(), {{open_tag(), kOpenId}, {"plan", 5}, {close_tag(), kCloseId}, {"\n\n", 271}, {"go", 6}},
                false, "plan", "go");
}

TEST(think_lead_off_a_closer_mid_answer_stays_content) {
    // Without an opener and away from the lead, a closer is text the model
    // wrote: the stream cannot take back what it already sent as content, and
    // the non-streamed path must give the same answer. Pinned as content.
    expect_both(qwen_prompt_off(), {{"The tag is ", 1}, {close_tag(), kCloseId}, {" here.", 2}}, false, "",
                "The tag is " + close_tag() + " here.");
}

// ---- thinking on: the model closed the block --------------------------------

TEST(think_lead_on_a_leading_closer_gives_empty_reasoning) {
    expect_both(qwen_prompt_on(), {{close_tag(), kCloseId}, {"\n\n", 271}, {"ORCHID-LANTERN-77", 1000}}, true, "",
                "ORCHID-LANTERN-77");
}

TEST(think_lead_on_a_repeated_closer_after_the_block_is_not_content) {
    expect_both(qwen_prompt_on(),
                {{"check the needle", 5}, {close_tag(), kCloseId}, {"\n\n", 271}, {close_tag(), kCloseId}, {"\n\n", 271},
                 {"ORCHID-LANTERN-77", 1000}},
                true, "check the needle", "ORCHID-LANTERN-77");
}

TEST(think_lead_on_a_repeated_closer_split_across_pieces) {
    const std::string c = close_tag();
    expect_both(qwen_prompt_on(),
                {{"check", 5}, {c, kCloseId}, {"\n", 198}, {c.substr(0, 5), 30}, {c.substr(5), 31}, {"\n\nok", 32}}, true,
                "check", "ok");
}

// ---- tool calls with `<` in the arguments, around the block -----------------

TEST(think_lead_off_call_with_comparisons_after_a_repeated_closer) {
    Script script = {{close_tag(), kCloseId}, {"\n\n", 271}};
    for (auto& p : call_pieces(sql())) script.push_back(p);
    for (bool stream : {false, true}) {
        Fixture   fx(qwen_prompt_off(), script);
        const Out o = stream ? fx.streamed(false, true) : fx.non_streamed(false, true);
        CHECK_EQ(o.content, std::string(""));
        CHECK_EQ(o.calls.size(), 1u);
        if (o.calls.size() == 1) CHECK_EQ(sql_of(o.calls[0]), sql());
    }
}

TEST(think_lead_calls_keep_angle_brackets_in_their_arguments) {
    // `<` in the argument, markup in the argument, and reasoning that itself
    // holds a `<` and a call opener's text: the call parses from content only
    const ToolCallTags& t = tool_call_tags();
    const std::string   markup = "<b>x</b>";
    for (const std::string& value : {sql(), markup, std::string("a <"), std::string("x </b")}) {
        Script script = {{"is 3 < 5? I would write " + s(t.open) + " later", 5}, {close_tag(), kCloseId}, {"\n\n", 271},
                         {"Running it.\n", 7}};
        for (auto& p : call_pieces(value)) script.push_back(p);
        for (bool stream : {false, true}) {
            Fixture   fx(qwen_prompt_on(), script);
            const Out o = stream ? fx.streamed(true, true) : fx.non_streamed(true, true);
            CHECK_EQ(o.reasoning, "is 3 < 5? I would write " + s(t.open) + " later");
            CHECK_EQ(o.content, std::string("Running it.\n"));
            CHECK_EQ(o.calls.size(), 1u);
            if (o.calls.size() == 1) CHECK_EQ(sql_of(o.calls[0]), value);
        }
    }
}

TEST(think_lead_content_with_comparisons_streams_whole) {
    // prose with `<` before a call: held only while it could be a call opener
    Script script = {{"Rows where amount <", 8}, {" 0.01 and <b>bold</b>:\n", 9}};
    for (auto& p : call_pieces(sql())) script.push_back(p);
    for (bool stream : {false, true}) {
        Fixture   fx(qwen_prompt_off(), script);
        const Out o = stream ? fx.streamed(false, true) : fx.non_streamed(false, true);
        CHECK_EQ(o.content, std::string("Rows where amount < 0.01 and <b>bold</b>:\n"));
        CHECK_EQ(o.calls.size(), 1u);
        if (o.calls.size() == 1) CHECK_EQ(sql_of(o.calls[0]), sql());
    }
}

// ---- the Responses route reads the same split -------------------------------

TEST(think_lead_off_responses_route_drops_the_repeated_closer) {
    const Script script = {{close_tag(), kCloseId}, {"\n\n", 271}, {"ORCHID-LANTERN-77", 1000}};
    const json   rbody  = {{"model", "stub"}, {"input", "needle?"}, {"reasoning", {{"effort", "none"}}}, {"max_output_tokens", 64}};
    for (bool stream : {false, true}) {
        Fixture             fx(qwen_prompt_off(), script);
        api::ResponseRequest r;
        CHECK(!api::parse_response_request(rbody, r).has_value());
        api::PreparedResponse p;
        CHECK(!api::prepare_response(fx.ctx, std::move(r), p).has_value());
        std::string text;
        if (!stream) {
            const api::HttpResult res = api::run_response(fx.ctx, p, 0);
            CHECK_EQ(res.status, 200);
            for (const json& it : res.body["output"])
                if (it.value("type", "") == "message")
                    for (const json& c : it["content"])
                        if (c.value("type", "") == "output_text") text += c["text"].get<std::string>();
        } else {
            api::stream_response(fx.ctx, p, 0, [&](std::string_view frame) {
                const size_t data = frame.find("data: ");
                if (frame.rfind("event: ", 0) != 0 || data == std::string_view::npos) return true;
                std::string payload(frame.substr(data + 6));
                while (!payload.empty() && payload.back() == '\n') payload.pop_back();
                const json j = json::parse(payload);
                if (j.value("type", "") == "response.output_text.delta") text += j["delta"].get<std::string>();
                return true;
            });
        }
        CHECK_EQ(text, std::string("ORCHID-LANTERN-77"));
    }
}

// ---- the start state and the whole-text split -------------------------------

TEST(think_lead_prompt_start_state_per_template) {
    const ChatDialect q = ChatDialect::Qwen;
    // the Qwen templates (agent, coder, the fine-tune): off closes, on opens
    CHECK(api::prompt_ends_after_think_block(qwen_prompt_off(), q));
    CHECK(!api::prompt_ends_inside_think_block(qwen_prompt_off(), q));
    CHECK(api::prompt_ends_inside_think_block(qwen_prompt_on(), q));
    CHECK(!api::prompt_ends_after_think_block(qwen_prompt_on(), q));
    // Dory's bare pair
    CHECK(api::prompt_ends_after_think_block(dory_prompt_off(), q));
    // no block at the end: undecided, as before
    CHECK(!api::prompt_ends_after_think_block(head(), q));
    // Gemma 4's model turn neither opens nor closes its channel, and the Qwen
    // pair is not in Gemma's table
    const ChatDialect g = ChatDialect::Gemma4;
    CHECK(!api::prompt_ends_after_think_block("turn\nmodel\n", g));
    CHECK(!api::prompt_ends_after_think_block(qwen_prompt_off(), g));
    CHECK(api::prompt_ends_after_think_block("x" + s(markers::kGemmaThoughtClose), g));
}

TEST(think_lead_whole_text_split_agrees_with_the_stream_at_every_cut) {
    struct Case {
        ThinkStart  start;
        std::string raw, reasoning, content;
    };
    const std::vector<Case> cases = {
        {ThinkStart::Closed, close_tag() + "\n\nORCHID-LANTERN-77", "", "ORCHID-LANTERN-77"},
        {ThinkStart::Closed, "\n" + close_tag() + close_tag() + "\nok", "", "ok"},
        {ThinkStart::Closed, "\n\nkept as generated", "", "\n\nkept as generated"},
        {ThinkStart::Closed, "a " + close_tag() + " b", "", "a " + close_tag() + " b"},
        {ThinkStart::Closed, close_tag() + "\n" + open_tag() + "plan" + close_tag() + "go", "plan", "go"},
        {ThinkStart::Open, "plan" + close_tag() + "\n\n" + close_tag() + "\n\nok", "plan", "ok"},
        {ThinkStart::Open, "plan" + close_tag() + "\n\nok " + close_tag(), "plan", "ok " + close_tag()},
        {ThinkStart::Undecided, close_tag() + "x", "", close_tag() + "x"},   // no block closed: untouched
    };
    for (const Case& c : cases) {
        const ReasoningSplit whole = split_reasoning(c.raw, c.start);
        CHECK_EQ(whole.reasoning, c.reasoning);
        CHECK_EQ(whole.content, c.content);
        for (size_t cut = 0; cut <= c.raw.size(); ++cut) {
            ReasoningStreamer st(c.start);
            std::string       reasoning, content;
            for (const std::string& piece : {c.raw.substr(0, cut), c.raw.substr(cut)}) {
                const auto step = st.push(piece);
                reasoning += step.reasoning;
                content += step.content;
            }
            const auto tail = st.flush();
            reasoning += tail.reasoning;
            content += tail.content;
            while (!reasoning.empty() && reasoning.back() == '\n') reasoning.pop_back();
            CHECK_EQ(reasoning, c.reasoning);
            CHECK_EQ(content, c.content);
        }
    }
}
