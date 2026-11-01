#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "api/auth.h"
#include "api/handlers.h"
#include "api/metrics.h"
#include "api/responses_state.h"
#include "build_info.h"
#include "config.h"
#include "exec/backend.h"
#include "http/server.h"
#include "util/log.h"
#include "util/sha256.h"

namespace {

// The stub's context when neither --n-ctx nor --lane-ctx is given: the same
// rule the llama engine applies to a GGUF (the trained context, at most this).
constexpr int kDefaultNCtxCap = 32768;

std::atomic<lgc::HttpServer*> g_server{nullptr};

void on_signal(int sig) {
    lgc::HttpServer* srv = g_server.exchange(nullptr);
    if (srv != nullptr) srv->stop();
    (void)sig;
}

lgc::log::Level level_for(int verbosity) {
    if (verbosity >= 2) return lgc::log::Level::Debug;
    if (verbosity >= 1) return lgc::log::Level::Verbose;
    return lgc::log::Level::Info;
}

}  // namespace

int main(int argc, char** argv) {
    lgc::Config cfg;
    const lgc::ArgParse parsed = lgc::parse_args(argc, argv, cfg);

    if (!parsed.ok) {
        std::fprintf(stderr, "arcint: %s\n", parsed.error.c_str());
        return 2;
    }
    if (cfg.show_help) {
        std::fputs(lgc::usage_text().c_str(), stdout);
        return 0;
    }
    if (cfg.show_version) {
        std::printf("arcint %s (%s) %s, %s\n", ARCINT_VERSION, ARCINT_GIT_SHA,
                    ARCINT_BUILD_TYPE, ARCINT_COMPILER);
        return 0;
    }
    lgc::log::set_level(level_for(cfg.verbosity));
    lgc::log::info("boot", "arcint %s (%s) %s, %s", ARCINT_VERSION, ARCINT_GIT_SHA,
                   ARCINT_BUILD_TYPE, ARCINT_COMPILER);

    // --api-key/--api-key-file: the key is read once here, never logged and
    // never published. The server keeps only its sha256 digest; the plaintext
    // is scrubbed before any request is served.
    const bool key_configured = !cfg.api_key_file.empty() || !cfg.api_key.empty();
    if (!cfg.api_key_file.empty()) {
        std::string err;
        auto        key = lgc::api::read_api_key_file(cfg.api_key_file, err);
        if (!key) {
            lgc::log::error("boot", "could not read --api-key-file: %s", err.c_str());
            return 2;
        }
        cfg.api_key = *key;
        lgc::api::secure_scrub(*key);   // the file-read buffer
    }
    if (key_configured) {
        std::string key_error, key_warning;
        if (!lgc::api::api_key_hygiene(cfg.api_key, key_error, key_warning)) {
            lgc::log::error("boot", "%s", key_error.c_str());
            return 2;
        }
        if (!key_warning.empty()) lgc::log::warn("boot", "%s", key_warning.c_str());
        // sha256_hex_secure wipes its own working copy of the key too.
        cfg.api_key_sha256 = lgc::sha256_hex_secure(cfg.api_key);
        lgc::api::secure_scrub(cfg.api_key);   // the parsed key
        lgc::log::info("boot", "%s", "an API key is configured: every route but /health needs "
                                 "`Authorization: Bearer <key>`");
    }

    std::unique_ptr<lgc::Backend> backend;
    if (cfg.stub) {
        int         n_ctx        = cfg.n_ctx;
        const char* n_ctx_source = "requested";
        if (cfg.named_lanes()) {
            // the first lane's cap is the served context; a cap over the
            // stub's trained context is refused here, as the libllama engine
            // refuses it at load
            for (size_t i = 0; i < cfg.lane_ctx.size(); ++i) {
                if (cfg.lane_ctx[i] > lgc::kStubNCtxTrain) {
                    lgc::log::error("boot", "--lane-ctx %d (lane '%s') exceeds the stub's trained context %d",
                                    cfg.lane_ctx[i], cfg.lane_names[i].c_str(), lgc::kStubNCtxTrain);
                    return 2;
                }
            }
            n_ctx        = cfg.lane_ctx[0];
            n_ctx_source = "--lane-ctx, the first lane";
        }
        if (n_ctx <= 0) {
            n_ctx        = std::min(lgc::kStubNCtxTrain, kDefaultNCtxCap);
            n_ctx_source = "default";
        }
        lgc::SamplerDefaults sampler;
        if (auto err = lgc::apply_operator_defaults(cfg, sampler)) {
            lgc::log::error("boot", "%s", err->c_str());
            return 2;
        }
        backend = lgc::make_stub_backend(n_ctx, cfg.stub_delay_ms, cfg.served_model_name, cfg.lane_ctx, sampler);

        lgc::log::warn("boot", "%s",
                       "stub backend: no model, no card, synthetic output. "
                       "Nothing measured here is a model result.");
        lgc::log::info("load", "stub | n_ctx %d (%s)", n_ctx, n_ctx_source);
        lgc::log::info("mem", "%s", "no KV cells are allocated");
    } else {
#ifdef ARCINT_LLAMA
        const int n_ctx = cfg.n_ctx;
        try {
            backend = lgc::make_llama_backend(cfg, n_ctx);
        } catch (const std::exception& e) {
            lgc::log::error("load", "could not bring up the llama.cpp executor: %s", e.what());
            return 1;
        }
        if (cfg.named_lanes()) {
            std::string ls;
            for (size_t i = 0; i < cfg.lane_names.size(); ++i)
                ls += (i ? ", " : "") + cfg.lane_names[i] + " " + std::to_string(cfg.lane_ctx[i]);
            lgc::log::info("load", "device %s | %zu named lanes: %s", cfg.device.c_str(), cfg.lane_names.size(), ls.c_str());
        } else {
            lgc::log::info("load", "n_ctx %d | device %s | %d lane%s", backend->status().n_ctx, cfg.device.c_str(),
                           cfg.parallel, cfg.parallel == 1 ? "" : "s");
        }
#else
        lgc::log::error("boot", "%s", "this build has no serving engine: configure with -DARCINT_LLAMA=ON");
        return 2;
#endif
    }

    if (cfg.named_lanes()) {
        lgc::log::info("load", "%zu named lanes; the model field picks the lane, an empty one or '%s' the "
                               "first, any other name a 404 (DESIGN 4.2):",
                       cfg.lane_names.size(), backend->status().id.c_str());
        for (size_t i = 0; i < cfg.lane_names.size(); ++i)
            lgc::log::info("load", "  lane %zu '%s': n_ctx %d", i, cfg.lane_names[i].c_str(), cfg.lane_ctx[i]);
    } else if (!cfg.served_model_name.empty()) {
        lgc::log::info("load", "served as '%s' (--served-model-name); the file is '%s'",
                       backend->status().served_id.c_str(), backend->status().id.c_str());
    }

    lgc::api::SlotPool slots(cfg.parallel);
    // --metrics owns the counters; without the flag nothing is recorded.
    lgc::api::Metrics metrics;
    lgc::api::Context  ctx{.cfg     = &cfg,
                           .backend = backend.get(),
                           .slots   = &slots,
                           .metrics = cfg.metrics ? &metrics : nullptr};
    ctx.responses = std::make_shared<lgc::api::ResponseStore>(lgc::api::ResponseStore::Limits{
        cfg.responses_store_entries, static_cast<uint64_t>(cfg.responses_store_bytes),
        cfg.responses_store_ttl_s});
    // named lanes: a pool each, admitting by name
    std::vector<std::unique_ptr<lgc::api::SlotPool>> lane_pools;
    ctx.lanes = lgc::api::make_lanes(cfg, lane_pools);

    // The TLS server construction can refuse a bad --tls-cert/--tls-key; that
    // is a boot error, not a bind error later.
    std::unique_ptr<lgc::HttpServer> server;
    try {
        server = std::make_unique<lgc::HttpServer>(cfg, ctx);
    } catch (const std::exception& e) {
        lgc::log::error("boot", "%s", e.what());
        return 2;
    }
    g_server.store(server.get());
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);

    // The endpoint-identity line: address, lanes, and the name this process
    // answers to. A journal that does not say which name is served is no help
    // when a roster discovered one from /v1/models and a client is using
    // another.
    std::string serving = "'" + backend->status().served_id + "'";
    if (!ctx.lanes.empty()) {
        serving.clear();
        for (const lgc::api::Lane& l : ctx.lanes)
            serving += lgc::log::format("%s'%s' (%d)", serving.empty() ? "" : ", ", l.name.c_str(), l.n_ctx);
    }
    lgc::log::info("http", "listening on %s:%d | %d slot%s | serving %s", cfg.host.c_str(),
                   cfg.port, slots.total(), slots.total() == 1 ? "" : "s", serving.c_str());

    if (!server->listen()) {
        lgc::log::error("http", "could not bind %s:%d", cfg.host.c_str(), cfg.port);
        g_server.store(nullptr);
        return 1;
    }

    g_server.store(nullptr);
    lgc::log::info("http", "%s", "stopped");
    return 0;
}
