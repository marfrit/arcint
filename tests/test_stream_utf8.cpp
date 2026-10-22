#include "harness.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "api/handlers.h"
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

// Runs stream_chat against a backend that emits `reply` (torn the way the
// stub tears it) with the think block pre-opened, and returns the parsed SSE
// chunk objects (the [DONE] sentinel dropped).
std::vector<json> run_stream(const std::string& reply) {
    SplittingBackend backend(reply);
    lgc::api::Context ctx;
    ctx.backend = &backend;

    lgc::api::PreparedChat prep;
    prep.think_open          = true;  // the prompt ended inside a think block
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

}  // namespace

using namespace lgc;

// Multi-byte code points of every width (2, 3 and 4 bytes) inside the think
// block and again after it closes, each isolated so the stub's emit_split
// tears it in half. Every reasoning_content and content delta must be valid
// UTF-8, and each must reassemble to the original byte for byte.
TEST(stream_reasoning_and_content_never_split_a_code_point) {
    const std::string reasoning = "x \xe2\x87\xa9 y \xc3\xa9 z \xf0\x9f\xa7\xa9 w";  // ⇒ é 🧩
    const std::string content   = "p \xe2\x87\xa9 q \xc3\xa9 r \xf0\x9f\xa7\xa9 s";
    const std::string reply     = reasoning + "</think>" + content;

    const std::vector<json> objs = run_stream(reply);

    check_stream(deltas(objs, "reasoning_content"), reasoning, "reasoning");
    check_stream(deltas(objs, "content"), content, "content");
}

// A code point split exactly at the think-closer boundary: the last reasoning
// byte is a torn code point immediately before the close tag. It must land in
// the reasoning stream (reassembled), not leak into content or be dropped.
TEST(stream_code_point_at_the_think_closer_boundary) {
    const std::string reasoning = "w \xe2\x87\xa9";  // ⇒ right before the close tag
    const std::string content   = "s";
    const std::string reply     = reasoning + "</think>" + content;

    const std::vector<json> objs = run_stream(reply);

    check_stream(deltas(objs, "reasoning_content"), reasoning, "reasoning");
    check_stream(deltas(objs, "content"), content, "content");
}

// The block never closes (a length stop, typically): the held bytes flush at
// end of stream into the reasoning stream, in order, and still reassemble.
TEST(stream_reasoning_flushed_at_end_of_stream) {
    const std::string reasoning = "x \xe2\x87\xa9 y";  // ⇒ torn, block never closes
    const std::vector<json> objs = run_stream(reasoning);

    check_stream(deltas(objs, "reasoning_content"), reasoning, "reasoning");
    CHECK(deltas(objs, "content").empty());
}
