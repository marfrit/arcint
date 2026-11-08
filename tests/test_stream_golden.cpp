#include "harness.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "api/handlers.h"
#include "api/responses.h"
#include "api/responses_stream.h"
#include "core/markers.h"
#include "core/toolcall.h"
#include "exec/backend.h"

// The streamed routes' frames, pinned byte for byte. The expectations were
// taken from the streamed chat and Responses paths as they stood before the
// raw trace reordered their last steps (fcc9e41e), so a change that moves a
// frame's bytes or their order fails here, whatever else it passes. Only the
// fields that differ run to run (ids, timestamps) are normalised.
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

struct Case {
    std::string  name;
    std::string  prompt;
    Script       script;
    FinishReason end      = FinishReason::Stop;   // what the backend reports after the script
    bool         tools    = false;
    bool         with_ids = true;
    bool         stop_end = false;                // chat only: the request stops at "END"
};

class GoldenBackend final : public Backend {
public:
    explicit GoldenBackend(const Case& c) : c_(c) {
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
    std::string        render_chat(const ChatRequest&) const override { return c_.prompt; }
    ThinkBlockIds      think_block_ids(ChatDialect) const override {
        ThinkBlockIds ids;
        if (c_.with_ids) {
            ids.open_close_ids = {{kOpenId, kCloseId}};
            ids.open_ids       = {kOpenId};
            ids.close_ids      = {kCloseId};
        } else {
            ids.open_close_ids = {{-1, -1}};
        }
        ids.open_tags  = {s(markers::kThinkOpen)};
        ids.close_tags = {s(markers::kThinkClose)};
        ids.eog        = {kEogId};
        return ids;
    }
    FinishReason generate(const GenerationInput&, int, const TokenCallback& on_piece, GenerationStats& stats) override {
        for (const auto& [piece, id] : c_.script) {
            ++stats.completion_tokens;
            if (on_piece(piece, id) != Control::Continue) return FinishReason::Stop;
        }
        return c_.end;
    }

private:
    struct Tok final : public Tokenizer {
        std::vector<int> encode(std::string_view) override { return {}; }
        std::string      decode(const std::vector<int>&) override { return {}; }
        std::string      decode_one(int) override { return {}; }
        int              eos_id() const override { return kEogId; }
    } tok_;
    const Case& c_;
    ModelStatus status_;
};

json chat_tools() {
    return json::array({{{"type", "function"},
                         {"function",
                          {{"name", "run_query"},
                           {"parameters",
                            {{"type", "object"},
                             {"properties", {{"sql", {{"type", "string"}}}}},
                             {"required", json::array({"sql"})}}}}}}});
}

std::string call_text(const std::string& sql) {
    const ToolCallTags& t = tool_call_tags();
    return "\n" + s(t.fn_open) + "run_query>\n" + s(t.param_open) + "sql>\n" + sql + "\n" + s(t.param_end) + "\n" +
           s(t.fn_close) + "\n";
}

std::vector<Case> cases() {
    const ToolCallTags& t = tool_call_tags();
    const std::string   c = s(markers::kThinkClose);
    std::vector<Case>   v;
    // reasoning, the closer, prose, a call with `<` in it
    v.push_back({"think_call", prompt_on(),
                 {{"weigh it", 5}, {c, kCloseId}, {"\n\n", 271}, {"Running it.\n", 7}, {s(t.open), kCallOpenId},
                  {call_text("SELECT 1 WHERE a < 2"), 300}, {s(t.close), kCallCloseId}},
                 FinishReason::Stop, true});
    // the same without tools: the call markup is content
    v.push_back({"think_call_notools", prompt_on(),
                 {{"weigh it", 5}, {c, kCloseId}, {"\n\n", 271}, {"Running it.\n", 7}, {s(t.open), kCallOpenId},
                  {call_text("SELECT 1"), 300}, {s(t.close), kCallCloseId}},
                 FinishReason::Stop, false});
    // a stop string cuts the answer, tools off and on
    v.push_back({"stop_cut", prompt_off(), {{"Paris", 8}, {" END", 9}, {" tail", 10}}, FinishReason::Stop, false,
                 true, true});
    v.push_back({"stop_cut_tools", prompt_off(), {{"Paris", 8}, {" EN", 9}, {"D tail", 10}}, FinishReason::Stop, true,
                 true, true});
    // a length cut inside a call: unterminated, returned as content
    v.push_back({"length_in_call", prompt_off(),
                 {{"Let me.\n", 7}, {s(t.open), kCallOpenId}, {"\n" + s(t.fn_open) + "run_q", 300}},
                 FinishReason::Length, true});
    // a length cut inside the reasoning, a code point split across pieces
    v.push_back({"length_in_reasoning", prompt_on(), {{"weigh ", 5}, {"it \xc3", 6}, {"\xa9t", 7}},
                 FinishReason::Length, false});
    // a text-only vocabulary: the reasoning ends on a held closer prefix
    v.push_back({"held_closer_tail", prompt_on(), {{"plan ", 5}, {c.substr(0, 4), 6}}, FinishReason::Length, true,
                 false});
    return v;
}

// ids and timestamps differ run to run; nothing else may
std::string normalise(std::string f) {
    static const std::regex ids(R"re("(id|item_id|response_id|previous_response_id)":"[^"]*")re");
    static const std::regex times(R"re("(created|created_at|completed_at)":[0-9]+)re");
    f = std::regex_replace(f, ids, "\"$1\":\"X\"");
    f = std::regex_replace(f, times, "\"$1\":0");
    return f;
}

struct Fixture {
    Config        cfg;
    GoldenBackend backend;
    api::SlotPool slots{1};
    api::Context  ctx;
    explicit Fixture(const Case& c) : backend(c) {
        cfg.stub    = true;
        ctx.cfg     = &cfg;
        ctx.backend = &backend;
        ctx.slots   = &slots;
    }
};

std::string chat_frames(const Case& c) {
    Fixture fx(c);
    json    b = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}, {"stream", true}};
    if (c.tools) b["tools"] = chat_tools();
    if (c.stop_end) b["stop"] = json::array({"END"});
    api::PreparedChat prep;
    CHECK(!api::prepare_chat(fx.ctx, b, prep).has_value());
    std::string frames;
    api::stream_chat(fx.ctx, prep, 0, [&](std::string_view f) {
        frames.append(f);
        return true;
    });
    return normalise(frames);
}

std::string response_frames(const Case& c) {
    Fixture fx(c);
    json    b = {{"model", "stub"}, {"input", "hi"}, {"stream", true}, {"max_output_tokens", 64}};
    if (c.tools)
        b["tools"] = json::array({{{"type", "function"},
                                   {"name", "run_query"},
                                   {"parameters", {{"type", "object"}, {"properties", {{"sql", {{"type", "string"}}}}}}}}});
    api::ResponseRequest r;
    CHECK(!api::parse_response_request(b, r).has_value());
    api::PreparedResponse p;
    CHECK(!api::prepare_response(fx.ctx, std::move(r), p).has_value());
    std::string frames;
    api::stream_response(fx.ctx, p, 0, [&](std::string_view f) {
        frames.append(f);
        return true;
    });
    return normalise(frames);
}

const std::map<std::string, std::string>& golden();

void check_golden(const std::string& key, const std::string& got) {
    const auto& g  = golden();
    const auto  it = g.find(key);
    const bool  ok = it != g.end() && it->second == got;
    if (ok) return;
    // the frames as they came, for a reader of the failure
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "arcint-golden";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / (key + ".txt"), std::ios::binary) << got;
    t::fail(__FILE__, __LINE__, "streamed frames moved: " + key + " (got written to " + (dir / (key + ".txt")).string() + ")");
}

}  // namespace

TEST(stream_golden_frames_are_pinned) {
    for (const Case& c : cases()) {
        const std::string chat = chat_frames(c);
        CHECK_EQ(chat_frames(c), chat);   // deterministic after normalising
        check_golden("chat_" + c.name, chat);
        if (c.stop_end) continue;          // the Responses route takes no stop strings
        const std::string resp = response_frames(c);
        CHECK_EQ(response_frames(c), resp);
        check_golden("responses_" + c.name, resp);
    }
}

namespace {
#include "stream_golden_frames.inc"
}  // namespace
