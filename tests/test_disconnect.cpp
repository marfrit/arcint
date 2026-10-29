#include "harness.h"

#include <atomic>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "api/handlers.h"
#include "exec/backend.h"

// DESIGN §3.7: a dropped connection aborts at the next boundary and frees the
// lane. The boundary used to be "the next write that fails", and two phases
// write nothing for a long time: a long prompt's prefill, and a tool call
// buffered from its marker to the turn's end. These cases drive the handlers'
// seam with a peer probe that closes (the server's probe is cpp-httplib's
// is_writable, a peek for EOF).
namespace {

using namespace lgc;
using json = nlohmann::json;

// A peer probe that counts its calls and reports gone after `alive_for` of
// them, so a test decides at which boundary the client leaves.
struct Peer {
    std::atomic<int> calls{0};
    int              alive_for = 0;

    api::PeerProbe probe() {
        return [this] {
            const int n = ++calls;
            return n <= alive_for;
        };
    }
};

// The frames a streaming request wrote, parsed to their JSON payloads (the
// [DONE] sentinel dropped).
std::vector<json> frames_of(const std::vector<std::string>& raw) {
    std::vector<json> out;
    for (const std::string& frame : raw) {
        if (frame.rfind("data: ", 0) != 0) continue;
        std::string payload = frame.substr(6);
        while (!payload.empty() && payload.back() == '\n') payload.pop_back();
        if (payload == "[DONE]") continue;
        out.push_back(json::parse(payload));
    }
    return out;
}

// The finish_reason of the last frame that carries one, or "none".
std::string finish_of(const std::vector<json>& frames) {
    for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
        if (!it->contains("choices") || (*it)["choices"].empty()) continue;
        const json& c = (*it)["choices"][0];
        if (c.contains("finish_reason") && !c["finish_reason"].is_null())
            return c["finish_reason"].get<std::string>();
    }
    return "none";
}

std::string any_content(const std::vector<json>& frames) {
    std::string out;
    for (const json& f : frames) {
        if (!f.contains("choices")) continue;
        for (const json& c : f["choices"]) {
            if (c.contains("delta") && c["delta"].contains("content")) out += c["delta"]["content"].get<std::string>();
        }
    }
    return out;
}

api::PreparedChat prepared(const std::string& prompt) {
    api::PreparedChat prep;
    prep.parse_tool_calls = false;
    prep.id               = "test";
    prep.model            = "stub";
    prep.req.stream_include_usage = false;
    prep.input.prompt     = prompt;
    return prep;
}

// A backend for the buffered-tool-call case: its prefill and its decode obey
// the request's probe exactly as the stub and the libllama backend do, and its
// answer opens a tool call early, then streams argument text that the handlers
// buffer to the turn's end. The marker is built from parts: no marker is
// written literally.
class BufferedToolBackend final : public Backend {
public:
    BufferedToolBackend() {
        st_.id          = "stub";
        st_.served_id   = "stub";
        st_.n_ctx_train = kStubNCtxTrain;
        st_.loaded      = true;
        st_.stub        = true;
        st_.n_ctx       = 4096;
    }

    const ModelStatus& status() const override { return st_; }
    Tokenizer&         tokenizer() override { return tok_; }
    std::string        render_chat(const ChatRequest&) const override { return "prompt"; }

    FinishReason generate(const GenerationInput& in, int, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        stats.prompt_tokens = 8;
        for (int u = 0; u < 4; ++u) {   // the prefill's boundary
            if (gone(in)) return FinishReason::Abort;
        }
        stats.prefill_seconds = 0.001;

        std::string open;
        open += "<tool";     // the handlers' marker, built from parts
        open += "_call>";
        const std::vector<std::string> pieces = {"Sure", " here", "<tool", "_call", ">", "\n{\"name\":\"x\"",
                                                 ",\"arguments\":{", "}}", "\n</tool", "_call>"};
        std::vector<std::string> stream = pieces;
        for (int i = 0; i < 80; ++i) stream.push_back(" x");   // buffered argument text
        for (size_t i = 0; i < stream.size(); ++i) {
            if (gone(in)) return FinishReason::Abort;   // the decode's boundary, per token
            const Control c = on_piece(stream[i], static_cast<int>(i));
            if (c == Control::Cancel) return FinishReason::Abort;
            if (c == Control::Stop) return FinishReason::Stop;
            ++stats.completion_tokens;
        }
        return FinishReason::Stop;
    }

private:
    static bool gone(const GenerationInput& in) {
        return in.peer_alive && !in.peer_alive();
    }

    struct Tok final : public Tokenizer {
        std::vector<int> encode(std::string_view text) override {
            return std::vector<int>(text.begin(), text.end());
        }
        std::string decode(const std::vector<int>& ids) override {
            return std::string(ids.begin(), ids.end());
        }
        std::string decode_one(int id) override { return std::string(1, static_cast<char>(id)); }
        int         eos_id() const override { return 0; }
    } tok_;

    ModelStatus st_;
};

}  // namespace

// The prefill case: the stub's prefill takes six ubatches and the client is
// gone at the third boundary. The request ends Abort with no token decoded, so
// the lane is free long before the prefill it would have run.
TEST(client_gone_during_prefill_ends_the_request_early) {
    auto backend = make_stub_backend(4096, 0, {}, {}, {}, 6);
    api::SlotPool slots(1);
    api::Context  ctx;
    ctx.backend = backend.get();
    ctx.slots   = &slots;

    const auto lease = slots.acquire_for(0);
    CHECK_EQ(lease.index(), 0);
    CHECK_EQ(slots.free(), 0);

    Peer peer;
    peer.alive_for = 2;   // alive at ubatches 0 and 1, gone at 2
    std::vector<std::string> raw;
    api::stream_chat(ctx, prepared("a prompt that a prefill holds"), 0,
                     [&](std::string_view frame) {
                         raw.emplace_back(frame);
                         return true;
                     },
                     peer.probe());

    CHECK_EQ(peer.calls.load(), 3);   // six ubatches would have taken six checks
    CHECK_EQ(frames_of(raw).size(), 1);   // the role chunk only: no token was decoded
    CHECK_EQ(any_content(frames_of(raw)), std::string(""));
}

// The tool-call case: the client leaves while the call's text is buffered (the
// marker and the argument stream write nothing). The request ends Abort at the
// token boundary and the buffered text is not flushed as raw content.
TEST(client_gone_while_a_tool_call_is_buffered_ends_the_request_early) {
    BufferedToolBackend backend;
    api::SlotPool       slots(1);
    api::Context        ctx;
    ctx.backend = &backend;
    ctx.slots   = &slots;

    api::PreparedChat prep = prepared("call the tool");
    prep.parse_tool_calls  = true;

    Peer peer;
    peer.alive_for = 10;   // alive through the four prefill checks, the content and the marker
    std::vector<std::string> raw;
    api::stream_chat(ctx, prep, 0,
                     [&](std::string_view frame) {
                         raw.emplace_back(frame);
                         return true;
                     },
                     peer.probe());

    // 4 prefill checks and 7 of the answer's pieces; 89 pieces would have taken
    // 93 checks.
    CHECK_EQ(peer.calls.load(), 11);
    const auto frames = frames_of(raw);
    CHECK_EQ(finish_of(frames), std::string("none"));   // aborted requests send nothing more
    CHECK_EQ(any_content(frames), std::string("Sure here"));   // the marker's text never goes out
    for (const json& f : frames) {
        if (!f.contains("choices")) continue;
        for (const json& c : f["choices"])
            CHECK(!(c.contains("delta") && c["delta"].contains("tool_calls")));
    }
}
