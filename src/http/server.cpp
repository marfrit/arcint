#include "http/server.h"

#include <httplib.h>

#include "api/error.h"
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

}  // namespace

struct HttpServer::Impl {
    const Config&   cfg;
    api::Context&   ctx;
    httplib::Server svr;

    Impl(const Config& c, api::Context& x) : cfg(c), ctx(x) {}

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
        if (res.status == 404) {
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
        // One process serves one model (§2). The list has exactly one entry so
        // that OpenAI clients which probe this endpoint keep working.
        //
        // It carries the context length because this is the only place a
        // discovering proxy looks: it reads n_ctx (among a few spellings) from
        // the model object here and asks /props for nothing but the template
        // capabilities. A context published on /props alone reaches no client,
        // and a client with no context falls back to its own default against a
        // server configured for 262144. `n_ctx` is what this process is
        // actually running with; `n_ctx_train` is the artifact's ceiling, so a
        // caller can see both and tell them apart.
        const ModelStatus& st = ctx.backend->status();
        auto maybe_int = [](int v) { return v > 0 ? json(v) : json(nullptr); };
        json entry{{"id", st.served_id},
                   {"object", "model"},
                   {"owned_by", "arcint"},
                   {"n_ctx", maybe_int(st.n_ctx)},
                   {"n_ctx_train", maybe_int(st.n_ctx_train)},
                   {"quant", quant_name(st.quant)},
                   {"lanes", ctx.slots->total()}};
        // The artifact this endpoint actually serves, which --served-model-name
        // may have renamed. Always present, so a rename never hides identity.
        entry["canonical_id"] = st.id;
        send_json(res, 200, json{{"object", "list"}, {"data", json::array({std::move(entry)})}});
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
        // can only be a message inside a 200.
        auto lease = std::make_shared<api::SlotPool::Lease>();
        if (auto err = api::acquire_slot(ctx, *lease)) {
            send_json(res, err->status, err->body);
            return;
        }
        const int slot = lease->index();

        if (!prep->req.stream) {
            const api::HttpResult r = api::run_chat(ctx, *prep, slot);
            send_json(res, r.status, r.body);
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        // The lease rides in the provider so the lane is held for exactly as
        // long as the stream lives, and freed when it ends however it ends.
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, prep, lease, slot](size_t, httplib::DataSink& sink) {
                api::stream_chat(ctx, *prep, slot, [&sink](std::string_view data) {
                    return sink.write(data.data(), data.size());
                });
                sink.done();
                // httplib reads `false` as Error::Canceled and tears the
                // connection down; the stream ended normally, so say so and let
                // keep-alive survive.
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
        if (auto err = api::acquire_slot(ctx, *lease)) {
            send_json(res, err->status, err->body);
            return;
        }
        const int slot = lease->index();

        if (!prep->req.stream) {
            const api::HttpResult r = api::run_completion(ctx, *prep, slot);
            send_json(res, r.status, r.body);
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, prep, lease, slot](size_t, httplib::DataSink& sink) {
                api::stream_completion(ctx, *prep, slot, [&sink](std::string_view data) {
                    return sink.write(data.data(), data.size());
                });
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

void HttpServer::stop() { impl_->svr.stop(); }

}  // namespace lgc
