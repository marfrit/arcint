#include "harness.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "api/handlers.h"
#include "core/reasoning.h"
#include "core/markers.h"
#include "exec/backend.h"
#include "util/utf8.h"

// DESIGN §3.7: a multi-byte code point must never be split across two SSE
// chunks. The content path already routes every chunk through a utf8::Streamer
// (handlers.cpp's emit_content); the reasoning path did not, so a torn code
// point reached the client as replacement characters. These cases drive
// stream_chat directly (no socket) against a backend that tears multi-byte
// pieces exactly the way the stub does (src/exec/backend_stub.cpp's
// emit_split), so the hold-back path is exercised the way production hits it.
//
// The backend mirrors the stub's byte-level detokenizer: it groups word bytes
// (letters, digits, '_' and every non-ASCII byte) into pieces, then halves
// every multi-byte piece so a code point lands split across two callbacks.
// Unlike the stub, its reply is fixed by the test, so a think block with
// multi-byte reasoning and content can be driven through the split.
namespace {

using json = nlohmann::json;

class SplittingBackend final : public lgc::Backend {
public:
    explicit SplittingBackend(std::string reply) : reply_(std::move(reply)) {
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
        for (const std::string& piece : pieces_of(reply_)) {
            ++stats.completion_tokens;
            // The stub's emit_split: a byte-level detokenizer routinely hands
            // out half of a multi-byte code point, so tear every multi-byte
            // piece in half. Single-code-point pieces pass through whole.
            if (lgc::utf8::count_codepoints(piece) < piece.size()) {
                const size_t cut = piece.size() / 2;
                if (emit(on_piece, std::string_view(piece).substr(0, cut))) return lgc::FinishReason::Stop;
                if (emit(on_piece, std::string_view(piece).substr(cut))) return lgc::FinishReason::Stop;
            } else if (emit(on_piece, piece)) {
                return lgc::FinishReason::Stop;
            }
        }
        return lgc::FinishReason::Stop;
    }

private:
    static bool emit(const lgc::TokenCallback& on_piece, std::string_view piece) {
        return on_piece(piece, 0) != lgc::Control::Continue;
    }

    // The stub's word-byte grouping (src/exec/backend_stub.cpp): letters,
    // digits, '_' and every non-ASCII byte glue together; each other ASCII
    // byte stands alone.
    static bool is_word_byte(char c) {
        const unsigned char b = static_cast<unsigned char>(c);
        if (b >= 0x80) return true;
        return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') ||
               b == '_';
    }
    static std::vector<std::string> pieces_of(const std::string& s) {
        std::vector<std::string> out;
        size_t                   i = 0;
        while (i < s.size()) {
            const size_t start = i;
            if (is_word_byte(s[i])) {
                while (i < s.size() && is_word_byte(s[i])) ++i;
            } else {
                ++i;
            }
            out.push_back(s.substr(start, i - start));
        }
        return out;
    }

    struct Tok final : public lgc::Tokenizer {
        std::vector<int> encode(std::string_view) override { return {}; }
        std::string      decode(const std::vector<int>&) override { return {}; }
        std::string      decode_one(int) override { return {}; }
        int              eos_id() const override { return 0; }
    } tok_;

    std::string   reply_;
    lgc::ModelStatus status_;
};

// Runs stream_chat against `backend` and returns the parsed SSE chunk objects
// (the [DONE] sentinel dropped).
std::vector<json> run_stream_with(lgc::Backend& backend, bool think_open) {
    lgc::api::Context ctx;
    ctx.backend = &backend;

    lgc::api::PreparedChat prep;
    prep.think_open          = think_open;  // the prompt ended inside a think block
    prep.think_tags_extended = false;
    prep.parse_tool_calls    = false;
    prep.id                  = "test";
    prep.model               = "stub";
    prep.req.stream_include_usage = false;

    std::vector<std::string> frames;
    lgc::api::SseWriter write = [&](std::string_view frame) {
        frames.push_back(std::string(frame));
        return true;
    };
    lgc::api::stream_chat(ctx, prep, 0, write);

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

// The think block pre-opened, the reply torn the way the stub tears it.
std::vector<json> run_stream(const std::string& reply) {
    SplittingBackend backend(reply);
    return run_stream_with(backend, true);
}

// A backend that emits an exact piece script, byte for byte: no word grouping,
// so a piece can carry what one decode step really carries -- the tail of the
// reasoning, the close tag and the first content byte in a single step.
class ScriptedBackend final : public lgc::Backend {
public:
    explicit ScriptedBackend(std::vector<std::string> pieces) : pieces_(std::move(pieces)) {
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
        for (const std::string& piece : pieces_) {
            ++stats.completion_tokens;
            if (on_piece(piece, 0) != lgc::Control::Continue) return lgc::FinishReason::Stop;
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

    std::vector<std::string> pieces_;
    lgc::ModelStatus         status_;
};

std::vector<json> run_stream_pieces(std::vector<std::string> pieces) {
    ScriptedBackend backend(std::move(pieces));
    return run_stream_with(backend, true);
}

// Collects every delta carrying `key`, in order.
std::vector<std::string> deltas(const std::vector<json>& objs, const char* key) {
    std::vector<std::string> out;
    for (const json& o : objs) {
        if (!o.contains("choices") || !o["choices"].is_array()) continue;
        for (const json& c : o["choices"]) {
            if (c.contains("delta") && c["delta"].contains(key))
                out.push_back(c["delta"][key].get<std::string>());
        }
    }
    return out;
}

std::string joined(const std::vector<std::string>& v) {
    std::string out;
    for (const std::string& s : v) out += s;
    return out;
}

// Every delta must be valid UTF-8 on its own (a torn code point would fail
// here), and the deltas must reassemble to `expected` byte for byte.
void check_stream(const std::vector<std::string>& ds, const std::string& expected, const char* what) {
    for (const std::string& d : ds) {
        if (!lgc::utf8::is_valid(d)) {
            t::fail(__FILE__, __LINE__,
                    std::string(what) + " delta is not valid UTF-8: " + t::show(d));
            return;
        }
    }
    const std::string got = joined(ds);
    if (got != expected) {
        t::fail(__FILE__, __LINE__,
                std::string(what) + " does not reassemble to the original: got " + t::show(got) +
                ", want " + t::show(expected));
        return;
    }
}

// The order the commit's claim is about: every reasoning_content frame
// precedes the first content frame. Reassembly alone cannot see a violation --
// a reasoning byte that flushed after content had started would still land in
// the right stream and reassemble, only late, in the wrong order.
void check_order(const std::vector<json>& objs) {
    bool content_seen = false;
    for (const json& o : objs) {
        if (!o.contains("choices") || !o["choices"].is_array()) continue;
        for (const json& c : o["choices"]) {
            if (!c.contains("delta")) continue;
            if (c["delta"].contains("content")) content_seen = true;
            if (content_seen && c["delta"].contains("reasoning_content")) {
                t::fail(__FILE__, __LINE__, "a reasoning frame arrived after content had started");
                return;
            }
        }
    }
}

}  // namespace

using namespace lgc;

// Multi-byte code points of every width (2, 3 and 4 bytes) inside the think
// block and again after it closes, each isolated so the stub's emit_split
// tears it in half. Every reasoning_content and content delta must be valid
// UTF-8, and each must reassemble to the original byte for byte.
TEST(stream_reasoning_and_content_never_split_a_code_point) {
    const std::string reasoning = "x \xe2\x87\xa9 y \xc3\xa9 z \xf0\x9f\xa7\xa9 w";  // ⇒ é 🧩
    const std::string content   = "p \xe2\x87\xa9 q \xc3\xa9 r \xf0\x9f\xa7\xa9 s";
    const std::string reply     = reasoning + std::string(markers::kThinkClose) + content;

    const std::vector<json> objs = run_stream(reply);

    check_stream(deltas(objs, "reasoning_content"), reasoning, "reasoning");
    check_stream(deltas(objs, "content"), content, "content");
    check_order(objs);
}

// A code point split exactly at the think-closer boundary: the last reasoning
// byte is a torn code point immediately before the close tag. It must land in
// the reasoning stream (reassembled), not leak into content or be dropped.
TEST(stream_code_point_at_the_think_closer_boundary) {
    const std::string reasoning = "w \xe2\x87\xa9";  // ⇒ right before the close tag
    const std::string content   = "s";
    const std::string reply     = reasoning + std::string(markers::kThinkClose) + content;

    const std::vector<json> objs = run_stream(reply);

    check_stream(deltas(objs, "reasoning_content"), reasoning, "reasoning");
    check_stream(deltas(objs, "content"), content, "content");
    check_order(objs);
}

// The block never closes (a length stop, typically): the held bytes flush at
// end of stream into the reasoning stream, in order, and still reassemble.
TEST(stream_reasoning_flushed_at_end_of_stream) {
    const std::string reasoning = "x \xe2\x87\xa9 y";  // ⇒ torn, block never closes
    const std::vector<json> objs = run_stream(reasoning);

    check_stream(deltas(objs, "reasoning_content"), reasoning, "reasoning");
    CHECK(deltas(objs, "content").empty());
    check_order(objs);
}

// The order claim, driven where it can actually break: one decode step that
// carries the close tag and the first content byte, with the reasoning side of
// that step still holding a torn code point. The reasoning delta the step
// emits -- held byte included -- must reach the client before the step's
// content delta, not after. (The torn case: a piece ends mid code point and
// the next piece starts with its remainder, as a byte-level detokenizer does;)
// here the remainder and the close tag arrive in the same step.)
TEST(stream_reasoning_frames_precede_content_frames) {
    const auto& closer = think_tag_pairs(false)[0].second;
    const std::string first  = "w \xe2\x87";   // a space and a torn code point's first two bytes
    const std::string second = "\xa9" + closer + " s";  // the remainder, the close, the answer's start
    const std::vector<json> objs = run_stream_pieces({ first, second });

    check_stream(deltas(objs, "reasoning_content"), "w \xe2\x87\xa9", "reasoning");  // ⇒ reassembled
    check_stream(deltas(objs, "content"), " s", "content");
    check_order(objs);
}

// ---------------------------------------------------------------------------
// The non-streaming path's reasoning_closed (run_chat). Before this case the
// field was set only by the Python gate's stand-in server: deleting it from
// the message kept the C++ suite green, so the claim carried by the field was
// untested here.
namespace {

// Emits its fixed reply as one piece through the same drive() the handler runs.
class ReplyBackend final : public lgc::Backend {
public:
    explicit ReplyBackend(std::string reply) : reply_(std::move(reply)) {
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
        stats.completion_tokens = 1;
        if (on_piece(reply_, 0) != lgc::Control::Continue) return lgc::FinishReason::Stop;
        return lgc::FinishReason::Stop;
    }

private:
    struct Tok final : public lgc::Tokenizer {
        std::vector<int> encode(std::string_view) override { return {}; }
        std::string      decode(const std::vector<int>&) override { return {}; }
        std::string      decode_one(int) override { return {}; }
        int              eos_id() const override { return 0; }
    } tok_;

    std::string    reply_;
    lgc::ModelStatus status_;
};

json run_chat_reply(const std::string& reply, bool think_open) {
    ReplyBackend      backend(reply);
    lgc::api::Context ctx;
    ctx.backend     = &backend;

    lgc::api::PreparedChat prep;
    prep.think_open          = think_open;
    prep.think_tags_extended = false;
    prep.parse_tool_calls    = false;
    prep.id                  = "test";
    prep.model               = "stub";
    prep.req.stream_include_usage = false;
    return lgc::api::run_chat(ctx, prep, 0).body;
}

}  // namespace

TEST(chat_run_reports_reasoning_closed) {
    const auto& opener = think_tag_pairs(false)[0].first;
    const auto& closer = think_tag_pairs(false)[0].second;
    json        msg;

    // Pre-opened block that closes: the field is true and the split is right.
    msg = run_chat_reply("why\n" + closer + "\nanswer", true)["choices"][0]["message"];
    CHECK_EQ(msg["reasoning_content"].get<std::string>(), "why");
    CHECK(msg.value("reasoning_closed", false));
    CHECK_EQ(msg["content"].get<std::string>(), "answer");

    // Pre-opened block that never closes: the field is false.
    msg = run_chat_reply("why not", true)["choices"][0]["message"];
    CHECK(msg.contains("reasoning_content"));
    CHECK(!msg.value("reasoning_closed", true));

    // A block the model opens itself at the lead: closed too.
    msg = run_chat_reply(opener + "why" + closer + "answer", false)["choices"][0]["message"];
    CHECK_EQ(msg["reasoning_content"].get<std::string>(), "why");
    CHECK(msg.value("reasoning_closed", false));

    // No block at all: no reasoning, and the field is not invented.
    msg = run_chat_reply("just an answer", false)["choices"][0]["message"];
    CHECK(!msg.contains("reasoning_content"));
    CHECK(!msg.contains("reasoning_closed"));
}

// Review B6: the stream carries the block's state the non-streaming message
// does -- reasoning_closed on the final chunk -- so a stream client can tell
// a closed-block empty answer from a mid-block stop. On the same condition
// the non-streaming path uses (reasoning was non-empty); absent otherwise.
TEST(stream_final_chunk_carries_reasoning_closed) {
    const auto& closer = think_tag_pairs(false)[0].second;
    const std::vector<json> objs = run_stream("why" + closer + "answer");
    const json& closed = objs.back()["choices"][0]["delta"];
    CHECK(closed.value("reasoning_closed", false));

    const std::vector<json> still = run_stream("never left the block");
    CHECK(!still.back()["choices"][0]["delta"].value("reasoning_closed", true));

    SplittingBackend plain("just an answer");
    const std::vector<json> none = run_stream_with(plain, false);
    CHECK(!none.back()["choices"][0]["delta"].contains("reasoning_closed"));
}

// ---------------------------------------------------------------------------
// The loop_guard object on the wire (1.2.5). The response body and the stream's
// finish frame are both plain json, so key order as a contract was never what
// the client saw; the seven keys and their values are. Read them off the
// serialized response rather than the in-memory object.
namespace {

// Emits a fixed reply and reports a moved loop guard in the stats, so both wire
// surfaces carry the object.
class LoopStatsBackend final : public lgc::Backend {
public:
    LoopStatsBackend() {
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
        stats.completion_tokens     = 1;
        stats.loop_detections       = 1;
        stats.loop_nudges           = 2;
        stats.loop_slaps            = 3;
        stats.loop_aborts           = 4;
        stats.loop_self_closes      = 5;
        stats.loop_door_masked      = 6;
        stats.loop_door_eog_masked  = 7;
        if (on_piece("answer", 0) != lgc::Control::Continue) return lgc::FinishReason::Stop;
        return lgc::FinishReason::Stop;
    }

private:
    struct Tok final : public lgc::Tokenizer {
        std::vector<int> encode(std::string_view) override { return {}; }
        std::string      decode(const std::vector<int>&) override { return {}; }
        std::string      decode_one(int) override { return {}; }
        int              eos_id() const override { return 0; }
    } tok_;

    lgc::ModelStatus status_;
};

json run_chat_loop_stats() {
    LoopStatsBackend  backend;
    lgc::api::Context ctx;
    ctx.backend = &backend;
    lgc::api::PreparedChat prep;
    prep.think_open          = false;
    prep.think_tags_extended = false;
    prep.parse_tool_calls    = false;
    prep.id                  = "test";
    prep.model               = "stub";
    prep.req.stream_include_usage = false;
    return lgc::api::run_chat(ctx, prep, 0).body;
}

}  // namespace

TEST(chat_loop_guard_object_carries_all_seven_keys_on_the_wire) {
    const json expected = {{"detections", 1}, {"nudges", 2}, {"slaps", 3}, {"aborts", 4},
                           {"self_closes", 5}, {"door_masked", 6}, {"door_eog_masked", 7}};

    // The non-streaming response, read as serialized and parsed back: what a
    // client actually receives.
    const json body = run_chat_loop_stats();
    const json wire = json::parse(body.dump());
    CHECK(wire.contains("loop_guard"));
    CHECK_EQ(wire["loop_guard"].size(), 7u);
    CHECK(wire["loop_guard"] == expected);

    // The stream's finish frame, the same object out of the same path.
    LoopStatsBackend  backend;
    const std::vector<json> objs = run_stream_with(backend, true);
    const json& finish = objs.back();
    CHECK(finish.contains("loop_guard"));
    CHECK_EQ(finish["loop_guard"].size(), 7u);
    CHECK(finish["loop_guard"] == expected);
    CHECK_EQ(finish["loop_guard"].dump(), wire["loop_guard"].dump());
}
