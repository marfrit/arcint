#include "http/server.h"

#include <atomic>

#include <httplib.h>

#include "api/error.h"
#include "api/responses.h"
#include "api/responses_state.h"
#include "api/responses_stream.h"
#include "util/log.h"

namespace lgc {
namespace {

using json = nlohmann::json;

constexpr size_t kMaxPayloadBytes = 64ull * 1024 * 1024;

void send_json(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(api::dump_json(body), "application/json");
}

bool read_json(const httplib::Request& req, httplib::Response& res, json& out) {
    try {
        out = json::parse(req.body);
    } catch (const json::exception& e) {
        send_json(res, 400,
                  api::invalid_request(log::format("request body is not valid JSON: %s",
                                                   e.what())));
        return false;
    }
    return true;
}

// The peer-liveness probe for a request: cpp-httplib's detail::is_socket_alive,
// a zero-timeout poll plus a MSG_PEEK, so it neither blocks nor writes a byte.
// It used to be the sink's is_writable, which is select_write under this
// server's 3600 s write timeout -- it could block under the backend's mutex --
// and without CPPHTTPLIB_USE_POLL it read EVERY fd >= 1024 as gone (this build
// defines poll for exactly that, see CMakeLists.txt). No socket (a direct
// call in a test): nothing is watched.
api::PeerProbe peer_probe(socket_t sock) {
    return [sock] {
        if (sock < 0) return true;
        return httplib::detail::is_socket_alive(sock);
    };
}

// One req: line per refusal at parse/prepare time (nothing answers it in the
// log today otherwise, and the live run had to reproduce a 400 by hand). The
// response bytes are unchanged; the numbers the envelope carries
// (prompt_tokens/n_ctx on an overflow) go on the line.
void log_refusal(const char* surface, const api::HttpResult& err) {
    const json& e = err.body["error"];
    const std::string code = (e.contains("code") && e["code"].is_string())
                                ? e["code"].get<std::string>() : "invalid_request";
    const std::string param = (e.contains("param") && e["param"].is_string())
                                ? e["param"].get<std::string>() : "";
    std::string numbers;
    if (err.body.contains("prompt_tokens") && err.body["prompt_tokens"].is_number_integer())
        numbers += log::format(" prompt_tokens=%d", err.body["prompt_tokens"].get<int>());
    if (err.body.contains("n_ctx") && err.body["n_ctx"].is_number_integer())
        numbers += log::format(" n_ctx=%d", err.body["n_ctx"].get<int>());
    log::warn("req", "%s refused: %d code=%s%s%s", surface, err.status, code.c_str(),
              param.empty() ? "" : log::format(" param=%s", param.c_str()).c_str(),
              numbers.c_str());
}

// The in-flight registry entry's life: removed on every exit, including an
// exception out of the route (a leaked registry entry would answer a cancel
// with 200 for a response that is gone).
struct FlightGuard {
    FlightGuard(api::ResponseStore* s, std::string i) : store(s), id(std::move(i)) {}
    api::ResponseStore* store = nullptr;
    std::string         id;
    ~FlightGuard() {
        if (store != nullptr) store->end_in_flight(id);
    }
};

}  // namespace

struct HttpServer::Impl {
    const Config&   cfg;
    api::Context&   ctx;
    httplib::Server svr;
    // The stop signal's flag: SIGTERM/SIGINT sets it through stop(), and every
    // request prepared while it is set carries it into the backend, which ends
    // at its next token or ubatch boundary (DESIGN §3.7). A stop is terminal:
    // nothing clears it for this process's life.
    std::atomic<bool> stopping{false};

    Impl(const Config& c, api::Context& x) : cfg(c), ctx(x) {
        ctx.stop_flag = &stopping;
    }

    void route();
};

void HttpServer::Impl::route() {
    svr.set_payload_max_length(kMaxPayloadBytes);
    svr.set_read_timeout(60, 0);
    // Decode can run for minutes at long contexts; a short write timeout would
    // cut a healthy stream off mid-generation.
    svr.set_write_timeout(3600, 0);

    if (cfg.http_threads > 0) {
        const size_t n     = static_cast<size_t>(cfg.http_threads);
        svr.new_task_queue = [n] { return new httplib::ThreadPool(n); };
    }

    svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        log::verbose("http", "%s %s -> %d", req.method.c_str(), req.path.c_str(), res.status);
    });

    svr.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            std::string what = "unknown error";
            try {
                std::rethrow_exception(ep);
            } catch (const std::exception& e) {
                what = e.what();
            } catch (...) {
            }
            log::error("http", "unhandled exception: %s", what.c_str());
            send_json(res, 500, api::error_body(what, "internal_error"));
        });

    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        // httplib calls this for every status >= 400, a handler's own 404 (a
        // named lane's unknown model, §4.2) included: only a 404 without a
        // body is a missing route
        if (res.status == 404 && res.body.empty()) {
            send_json(res, 404,
                      api::error_body(log::format("no route for %s %s", req.method.c_str(),
                                                  req.path.c_str()),
                                      "invalid_request_error", "not_found"));
        }
    });

    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, api::health(ctx));
    });

    svr.Get("/props", [this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, api::props(ctx));
    });

    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, api::models(ctx));
    });

    // The TTS surface (docs/milestone-1.2.0.md D4/D5). Both routes refuse from
    // prepare_speech with a clear message when the build has no --tts or no
    // audio-generation mmproj; the voice list is a names-only probe.
    svr.Get("/v1/audio/voices", [this](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, api::voices(ctx));
    });

    svr.Post("/v1/audio/speech", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        if (!read_json(req, res, body)) return;

        auto prep = std::make_shared<api::PreparedSpeech>();
        if (auto err = api::prepare_speech(ctx, body, *prep)) {
            send_json(res, err->status, err->body);
            return;
        }

        // Same lane machinery as the chat routes: admission is a status code
        // before any byte of the answer is committed.
        auto lease = std::make_shared<api::SlotPool::Lease>();
        int  slot  = -1;
        if (auto err = api::acquire_slot(ctx, prep->lane, *lease, slot)) {
            send_json(res, err->status, err->body);
            return;
        }

        const api::SpeechResult r = api::run_speech(ctx, *prep, slot);
        if (r.status != 200 || r.wav.empty()) {
            send_json(res, r.status, r.error);
            return;
        }
        res.status = 200;
        res.set_content(r.wav, "audio/wav");
    });

    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        if (!read_json(req, res, body)) return;

        auto prep = std::make_shared<api::PreparedChat>();
        if (auto err = api::prepare_chat(ctx, body, *prep)) {
            send_json(res, err->status, err->body);
            return;
        }

        // The lane is taken here, before a single response byte is committed —
        // a 503 has to be a status code, and once the SSE body has started it
        // can only be a message inside a 200. Named lanes: the lane prepare
        // resolved from the model field, and only that one.
        auto lease = std::make_shared<api::SlotPool::Lease>();
        int  slot  = -1;
        if (auto err = api::acquire_slot(ctx, prep->lane, *lease, slot)) {
            send_json(res, err->status, err->body);
            return;
        }

        if (!prep->req.stream) {
            // The probe rides the prepared input, so the backend runs it as it
            // is: a stream:false client that leaves during a long prefill is
            // seen at the next boundary too. A stop signal answers a 503; a
            // gone client's response is not built (r.gone).
            prep->input.peer_alive = peer_probe(req.socket);
            const api::HttpResult r = api::run_chat(ctx, *prep, slot);
            if (r.gone) return;
            send_json(res, r.status, r.body);
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        // The lease rides in the provider so the lane is held for exactly as
        // long as the stream lives, and freed when it ends however it ends.
        const socket_t sock = req.socket;
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, prep, lease, slot, sock](size_t, httplib::DataSink& sink) {
                // The handlers turn a generation's exceptions into an error
                // event; this is the last guard: nothing thrown here may leave
                // httplib's worker thread, which would end the process.
                try {
                    prep->input.peer_alive = peer_probe(sock);
                    api::stream_chat(ctx, *prep, slot, [&sink](std::string_view data) {
                        return sink.write(data.data(), data.size());
                    });
                } catch (const std::exception& e) {
                    log::error("http", "stream ended by an exception: %s", e.what());
                } catch (...) {
                    log::error("http", "%s", "stream ended by an unknown exception");
                }
                sink.done();
                // httplib reads `false` as Error::Canceled and tears the
                // connection down; the stream ended normally, so say so and let
                // keep-alive survive.
                return true;
            });
    });



    // POST /v1/responses/{id}/cancel (docs/campaigns/feyntalk.md Cancel, W4; an
    // arcint extension — the spec defines no cancel endpoint or state). An
    // in-flight response: set the flag and answer the in-progress snapshot with
    // arcint.cancelling. A stored terminal response is 409; an unknown id 404.
    svr.Post(R"(/v1/responses/([^/]+)/cancel)", [this](const httplib::Request& req,
                                                      httplib::Response& res) {
        const std::string id = req.matches[1];
        if (!ctx.responses) {
            send_json(res, 404, api::responses_error(
                "no cancel route without a response store: this process keeps none",
                "not_found", "response_not_found"));
            return;
        }
        if (auto inflight = ctx.responses->find_in_flight(id)) {
            inflight->cancel->store(true, std::memory_order_relaxed);
            api::ResponseParts parts;
            parts.status    = "in_progress";
            parts.output    = json::array();
            parts.usage     = nullptr;
            parts.completed = false;
            json body = api::response_resource(inflight->prep->req, inflight->prep->chat, parts);
            body["arcint"] = {{"v", 1}, {"cancelling", true}};
            send_json(res, 200, body);
            return;
        }
        if (ctx.responses->find_stored_any(id)) {
            send_json(res, 409, api::responses_error(
                "this response has already ended; a terminal response cannot be cancelled",
                "invalid_request", "response_not_cancellable"));
            return;
        }
        send_json(res, 404, api::responses_error(
            "no in-flight or stored response with this id", "not_found", "response_not_found"));
    });

    // Open Responses (docs/campaigns/feyntalk.md W1): one request path, one
    // generation path. parse_response_request maps the body onto a ChatRequest,
    // prepare_response hands it to the chat route's own prepare_from_chat, and
    // run_response drives it exactly as run_chat does. Non-stream only in W1:
    // stream:true is refused by the parser with the spec's envelope.
    svr.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception& e) {
            const api::HttpResult err{400, api::responses_error(
                log::format("request body is not valid JSON: %s", e.what()),
                "invalid_request", "invalid_request")};
            log_refusal("responses", err);
            send_json(res, err.status, err.body);
            return;
        }

        api::ResponseRequest parsed;
        if (auto err = api::parse_response_request(body, parsed)) {
            log_refusal("responses", *err);
            send_json(res, err->status, err->body);
            return;
        }

        auto prep = std::make_shared<api::PreparedResponse>();
        if (auto err = api::prepare_response(ctx, std::move(parsed), *prep)) {
            log_refusal("responses", *err);
            send_json(res, err->status, err->body);
            return;
        }

        auto lease = std::make_shared<api::SlotPool::Lease>();
        int  slot  = -1;
        if (auto err = api::acquire_slot(ctx, prep->chat.lane, *lease, slot, prep->affinity_slot)) {
            // The spec maps a busy service to 429 too_many_requests; the chat
            // route keeps its 503. The refusal's slots/lanes diagnostics ride
            // into the 429 body (review F1).
            const api::HttpResult busy = api::responses_busy(*err);
            send_json(res, busy.status, busy.body);
            return;
        }

        // The cancel flag and the registry entry exist BEFORE the first event:
        // `response.created` is what gives the client the id it cancels by. The
        // flag is ORed into the peer closure, so every boundary check the engine
        // already makes sees a cancel exactly as it sees a close.
        std::shared_ptr<std::atomic<bool>> cancel;
        std::shared_ptr<FlightGuard>       flight;
        if (ctx.responses) {
            cancel = std::make_shared<std::atomic<bool>>(false);
            auto entry = std::make_shared<api::InFlightResponse>();
            entry->id     = prep->chat.id;
            entry->lane   = prep->chat.lane;
            entry->cancel = cancel;
            entry->prep   = prep;
            ctx.responses->begin_in_flight(std::move(entry));
            flight = std::make_shared<FlightGuard>(ctx.responses.get(), prep->chat.id);
        }
        const socket_t sock = req.socket;
        auto peer = [sock, cancel] {
            if (cancel && cancel->load(std::memory_order_relaxed)) return false;
            return peer_probe(sock)();
        };

        if (!prep->req.chat.stream) {
            prep->chat.input.cancel     = cancel;
            prep->chat.input.peer_alive = peer;
            const api::HttpResult r = api::run_response(ctx, *prep, slot);
            if (r.gone) return;
            send_json(res, r.status, r.body);
            return;
        }

        // SSE (docs/campaigns/feyntalk.md W2): the writer emits one event per
        // frame, `event:` equal to the body's type, and a terminal event before
        // [DONE]. The lease rides in the provider so the lane is held for exactly
        // as long as the stream lives.
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, prep, lease, slot, sock, cancel, flight, peer](size_t, httplib::DataSink& sink) {
                try {
                    prep->chat.input.cancel     = cancel;
                    prep->chat.input.peer_alive = peer;
                    api::stream_response(ctx, *prep, slot, [&sink](std::string_view data) {
                        return sink.write(data.data(), data.size());
                    });
                } catch (const std::exception& e) {
                    log::error("http", "responses stream ended by an exception: %s", e.what());
                } catch (...) {
                    log::error("http", "%s", "responses stream ended by an unknown exception");
                }
                sink.done();
                return true;
            });
    });

    svr.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        if (!read_json(req, res, body)) return;

        auto prep = std::make_shared<api::PreparedCompletion>();
        if (auto err = api::prepare_completion(ctx, body, *prep)) {
            send_json(res, err->status, err->body);
            return;
        }

        auto lease = std::make_shared<api::SlotPool::Lease>();
        int  slot  = -1;
        if (auto err = api::acquire_slot(ctx, prep->lane, *lease, slot)) {
            send_json(res, err->status, err->body);
            return;
        }

        if (!prep->req.stream) {
            prep->input.peer_alive = peer_probe(req.socket);
            const api::HttpResult r = api::run_completion(ctx, *prep, slot);
            if (r.gone) return;
            send_json(res, r.status, r.body);
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        const socket_t sock = req.socket;
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, prep, lease, slot, sock](size_t, httplib::DataSink& sink) {
                // The handlers turn a generation's exceptions into an error
                // event; this is the last guard: nothing thrown here may leave
                // httplib's worker thread, which would end the process.
                try {
                    prep->input.peer_alive = peer_probe(sock);
                    api::stream_completion(ctx, *prep, slot, [&sink](std::string_view data) {
                        return sink.write(data.data(), data.size());
                    });
                } catch (const std::exception& e) {
                    log::error("http", "stream ended by an exception: %s", e.what());
                } catch (...) {
                    log::error("http", "%s", "stream ended by an unknown exception");
                }
                sink.done();
                return true;
            });
    });
}

HttpServer::HttpServer(const Config& cfg, api::Context& ctx)
    : impl_(std::make_unique<Impl>(cfg, ctx)) {
    impl_->route();
}

HttpServer::~HttpServer() = default;

bool HttpServer::listen() {
    return impl_->svr.listen(impl_->cfg.host.c_str(), impl_->cfg.port);
}

void HttpServer::stop() {
    // The running requests first, then the listener: a generation that writes
    // nothing (a long prefill, a buffered tool call) would otherwise outlive
    // the stop signal, and a stop during one timed out and was killed.
    impl_->stopping.store(true, std::memory_order_relaxed);
    impl_->svr.stop();
}

}  // namespace lgc
