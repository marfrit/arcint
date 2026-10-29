#include "harness.h"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include <httplib.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "api/handlers.h"
#include "exec/backend.h"
#include "http/server.h"

// DESIGN §3.7: a dropped connection aborts at the next boundary and frees the
// lane. The boundary used to be "the next write that fails", and two phases
// write nothing for a long time: a long prompt's prefill, and a tool call
// buffered from its marker to the turn's end. These cases drive the handlers'
// seam with a peer probe that closes (the server's probe is cpp-httplib's
// is_writable, a peek for EOF) and a stop signal that arrives mid-generation.
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
            if (gone(in)) return abort_stats(in, stats);
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
            if (gone(in)) return abort_stats(in, stats);   // the decode's boundary, per token
            const Control c = on_piece(stream[i], static_cast<int>(i));
            if (c == Control::Cancel) {
                stats.abort_cause = AbortCause::ClientGone;
                return FinishReason::Abort;
            }
            if (c == Control::Stop) return FinishReason::Stop;
            ++stats.completion_tokens;
        }
        return FinishReason::Stop;
    }

private:
    static bool gone(const GenerationInput& in) {
        if (in.stop != nullptr && in.stop->load(std::memory_order_relaxed)) return true;
        return in.peer_alive && !in.peer_alive();
    }

    // The cause the slot line reports: stop flag first, the probe second.
    static FinishReason abort_stats(const GenerationInput& in, GenerationStats& stats) {
        stats.abort_cause = in.stop != nullptr && in.stop->load(std::memory_order_relaxed)
                                ? AbortCause::StopSignal
                                : AbortCause::ClientGone;
        return FinishReason::Abort;
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
    api::PreparedChat prep = prepared("a prompt that a prefill holds");
    prep.input.peer_alive   = peer.probe();   // the route puts it into the input
    std::vector<std::string> raw;
    api::stream_chat(ctx, prep, 0,
                     [&](std::string_view frame) {
                         raw.emplace_back(frame);
                         return true;
                     });

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
    prep.input.peer_alive = peer.probe();
    std::vector<std::string> raw;
    api::stream_chat(ctx, prep, 0,
                     [&](std::string_view frame) {
                         raw.emplace_back(frame);
                         return true;
                     });

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

// The non-streamed half-buffered tool call, the card's other face: a client
// that leaves while the call's arguments are being collected must not get the
// raw text back as content, and its response is not built.
TEST(a_gone_client_mid_tool_call_on_the_non_streamed_path_returns_no_body) {
    BufferedToolBackend backend;
    api::SlotPool       slots(1);
    api::Context        ctx;
    ctx.backend = &backend;
    ctx.slots   = &slots;

    api::PreparedChat prep = prepared("call the tool");
    prep.parse_tool_calls  = true;
    Peer peer;
    peer.alive_for        = 10;   // alive through the prefill and `Sure here`, gone inside the arguments
    prep.input.peer_alive = peer.probe();

    const auto lease = slots.acquire_for(0);
    const api::HttpResult r = api::run_chat(ctx, prep, 0);
    CHECK(r.gone);
    CHECK(r.body.empty());   // no finish_reason, no raw content, no tool_calls
}

// The stop signal inside the buffered arguments: 503, not a 200 carrying the
// half-parsed call as content.
TEST(a_stop_signal_mid_tool_call_on_the_non_streamed_path_is_a_503) {
    BufferedToolBackend backend;
    api::SlotPool       slots(1);
    api::Context        ctx;
    ctx.backend = &backend;
    ctx.slots   = &slots;

    std::atomic<bool> stop{false};
    api::PreparedChat prep = prepared("call the tool");
    prep.parse_tool_calls  = true;
    prep.input.stop        = &stop;
    int checks = 0;
    prep.input.peer_alive = [&] {
        if (++checks >= 12) stop.store(true, std::memory_order_relaxed);   // the arguments are buffered
        return true;
    };

    const auto lease = slots.acquire_for(0);
    const api::HttpResult r = api::run_chat(ctx, prep, 0);
    CHECK_EQ(r.status, 503);
    CHECK_EQ(r.body["error"]["code"].get<std::string>(), std::string("server_stopping"));
    CHECK(!r.body.contains("choices"));   // no content, no tool_calls
}

// The stop signal: stop() from another thread while a generation runs. The
// request ends Abort at its next token boundary and run_chat returns well
// inside the bound, rather than waiting out the long decode.
TEST(stop_signal_aborts_a_running_generation) {
    Config cfg;
    cfg.stub = true;
    // 60 ms a token, and the stub's reply is ~100 tokens: its natural run is
    // over six seconds. Only the stop flag can end this inside the bound, so
    // dt < 0.5 is evidence for the stop and not for a short stub.
    auto backend = make_stub_backend(4096, 60, {}, {}, {}, 4);
    api::SlotPool slots(1);
    api::Context  ctx;
    ctx.cfg     = &cfg;
    ctx.backend = backend.get();
    ctx.slots   = &slots;

    HttpServer server(cfg, ctx);   // owns the stop flag ctx now points at

    json body{{"model", "stub"},
              {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
              {"max_tokens", 100000}};
    api::PreparedChat prep;
    if (auto err = api::prepare_chat(ctx, body, prep)) {
        CHECK(!err.has_value());
        return;
    }

    const auto lease = slots.acquire_for(0);
    std::thread stopper([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        server.stop();
    });

    const auto       t0 = std::chrono::steady_clock::now();
    const api::HttpResult r = api::run_chat(ctx, prep, 0);
    const double     dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stopper.join();

    // The bound first: >6 s natural, so only the stop flag ends this inside
    // 0.5 s. Then the answer: a stop signal is not an OpenAI finish reason, so
    // the non-streamed route answers 503 with an error body, never a 200 that
    // says "abort".
    CHECK(dt < 0.5);
    CHECK_EQ(r.status, 503);
    CHECK_EQ(r.body["error"]["code"].get<std::string>(), std::string("server_stopping"));
    CHECK(!r.gone);
}

// The cause reaches the stats: a stop signal is not logged as a gone client.
// The card showed "aborted: client gone" for a SIGTERM.
TEST(a_stop_abort_carries_the_stop_cause) {
    Config cfg;
    cfg.stub = true;
    auto backend = make_stub_backend(4096, 60, {}, {}, {}, 4);
    api::SlotPool slots(1);
    api::Context  ctx;
    ctx.cfg     = &cfg;
    ctx.backend = backend.get();
    ctx.slots   = &slots;
    HttpServer server(cfg, ctx);

    json body{{"model", "stub"},
              {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})},
              {"max_tokens", 100000}};
    api::PreparedChat prep;
    if (auto err = api::prepare_chat(ctx, body, prep)) {
        CHECK(!err.has_value());
        return;
    }

    std::thread stopper([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        server.stop();
    });
    const api::HttpResult r = api::run_chat(ctx, prep, 0);
    stopper.join();
    CHECK_EQ(r.status, 503);
}

// The stop flag names the cause in the stats, through the seam the slot line
// reads: a SIGTERM is not reported as a gone client.
TEST(a_stop_signal_carries_the_stop_cause_into_the_stats) {
    auto             backend = make_stub_backend(4096, 0, {}, {}, {}, 4);
    std::atomic<bool> stop{true};
    GenerationInput  in;
    in.prompt = "a b c";
    in.stop   = &stop;
    GenerationStats stats;
    const FinishReason reason = backend->generate(
        in, 0, [](std::string_view, int) { return Control::Continue; }, stats);
    CHECK(reason == FinishReason::Abort);
    CHECK(stats.abort_cause == AbortCause::StopSignal);
}

// The aborted prefill reports what it prefilled, not the whole prompt: the
// card showed "prefill 37723 tok in 9.53 s" for a prefill cut 8 s in.
TEST(an_aborted_prefill_reports_the_tokens_it_prefilled) {
    auto     backend = make_stub_backend(4096, 0, {}, {}, {}, 6);
    api::SlotPool slots(1);
    api::Context  ctx;
    ctx.backend = backend.get();
    ctx.slots   = &slots;

    // the stub tokenizer is word-based: 12 words, so 12 prompt tokens
    api::PreparedChat prep = prepared("a b c d e f g h i j k l");
    Peer peer;
    peer.alive_for        = 2;   // gone at the third ubatch boundary
    prep.input.peer_alive = peer.probe();

    GenerationStats stats;
    std::string     raw;
    const FinishReason reason = backend->generate(
        prep.input, 0, [](std::string_view, int) { return Control::Continue; }, stats);

    CHECK(reason == FinishReason::Abort);
    CHECK(stats.abort_cause == AbortCause::ClientGone);
    CHECK(stats.prompt_tokens > 6);                    // the prompt's own size
    CHECK_EQ(stats.prompt_tokens_prefilled, stats.prompt_tokens * 2 / 6);   // the `have` growth
    CHECK(stats.prompt_tokens_prefilled < stats.prompt_tokens);   // not the whole prompt
}

// The non-streamed path with a gone client: no body, and the route writes
// nothing (run_chat returns `gone`). Before the probe this kept the lane for
// the whole prefill.
TEST(a_gone_client_on_the_non_streamed_path_returns_no_body) {
    auto     backend = make_stub_backend(4096, 0, {}, {}, {}, 4);
    api::SlotPool slots(1);
    api::Context  ctx;
    ctx.backend = backend.get();
    ctx.slots   = &slots;

    api::PreparedChat prep = prepared("a long prompt");
    prep.input.peer_alive  = [] { return false; };   // gone from the first boundary

    const auto lease = slots.acquire_for(0);
    const api::HttpResult r = api::run_chat(ctx, prep, 0);
    CHECK(r.gone);
    CHECK(r.body.empty());
    CHECK_EQ(lease.index(), 0);
}

#ifndef _WIN32
namespace {

// Whether detail::is_socket_alive says a socketpair's peer is there, and how
// long the answer took. The probe is poll(0) + a MSG_PEEK: it must neither
// block (the sink's is_writable was a select_write under the 3600 s write
// timeout) nor misread an fd above FD_SETSIZE.
struct Pair {
    int fd[2] = {-1, -1};
    Pair() { CHECK_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fd), 0); }
    ~Pair() {
        if (fd[0] >= 0) close(fd[0]);
        if (fd[1] >= 0) close(fd[1]);
    }
};

double timed_alive(int fd) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool alive = httplib::detail::is_socket_alive(fd);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(alive);
    return dt;
}

}  // namespace

// The probe answers at once on a live, idle socket and reads gone after the
// peer half-closes.
TEST(the_client_probe_peeks_without_blocking) {
    Pair p;
    CHECK(timed_alive(p.fd[0]) < 0.05);   // poll(0): the answer is immediate
    shutdown(p.fd[1], SHUT_WR);           // the peer is gone
    CHECK(!httplib::detail::is_socket_alive(p.fd[0]));
}

// Above FD_SETSIZE select() returns -1 without setting EBADF, which the probe
// would read as "gone" on an alive socket. CPPHTTPLIB_USE_POLL (CMakeLists.txt)
// is what the probe needs; without it this case fails, and the server itself
// answers 500 to such a connection in process_request.
TEST(the_client_probe_works_above_fd_setsize) {
    std::vector<int> filler;
    while (filler.size() < 1100) {
        const int fd = open("/dev/null", O_RDONLY);
        if (fd < 0) { CHECK(fd >= 0); break; }
        filler.push_back(fd);
        if (fd >= FD_SETSIZE) break;
    }
    CHECK(filler.back() >= FD_SETSIZE);   // an fd beyond select()'s set

    Pair p;
    CHECK(p.fd[0] >= FD_SETSIZE);
    // non-blocking on the read side, so a select() build cannot hang the test;
    // it reads -1 (EAGAIN) and the probe would call that gone
    fcntl(p.fd[0], F_SETFL, O_NONBLOCK);
    CHECK(timed_alive(p.fd[0]) < 0.05);
    for (int fd : filler) close(fd);
}
#endif
