#include <atomic>
#include <csignal>
#include <cstdio>
#include <memory>

#include "api/handlers.h"
#include "build_info.h"
#include "config.h"
#include "core/artifact.h"
#include "exec/backend.h"
#include "http/server.h"
#include "util/log.h"
#include "util/text.h"

namespace {

// Fallback only, for an entry whose trained context is not pinned. It is a
// working default for the skeleton, never a claim about a model.
constexpr int kStubDefaultNCtx = 4096;

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
        std::fprintf(stderr, "ligence: %s\n", parsed.error.c_str());
        return 2;
    }
    if (cfg.show_help) {
        std::fputs(lgc::usage_text().c_str(), stdout);
        return 0;
    }
    if (cfg.show_version) {
        std::printf("ligence %s (%s) %s, %s\n", LIGENCE_VERSION, LIGENCE_GIT_SHA,
                    LIGENCE_BUILD_TYPE, LIGENCE_COMPILER);
        return 0;
    }

    lgc::log::set_level(level_for(cfg.verbosity));
    lgc::log::info("boot", "ligence %s (%s) %s, %s", LIGENCE_VERSION, LIGENCE_GIT_SHA,
                   LIGENCE_BUILD_TYPE, LIGENCE_COMPILER);

    const lgc::ModelEntry* entry = nullptr;

    std::unique_ptr<lgc::Backend> backend;
    if (cfg.stub) {
        entry = lgc::find_model(cfg.model_id);
        if (entry == nullptr) {
            lgc::log::error("boot", "'%s' is not in the allowlist", cfg.model_id.c_str());
            return 2;
        }
        const char* n_ctx_source = "requested";
        int         n_ctx        = cfg.n_ctx;
        if (n_ctx <= 0) {
            n_ctx        = entry->n_ctx_train > 0 ? entry->n_ctx_train : kStubDefaultNCtx;
            n_ctx_source = entry->n_ctx_train > 0 ? "allowlist" : "stub fallback";
        }
        backend = lgc::make_stub_backend(*entry, cfg.quant, n_ctx, cfg.stub_delay_ms);

        lgc::log::warn("boot", "%s",
                       "stub backend: no model, no OpenVINO, synthetic output. "
                       "Nothing measured here is a model result.");
        lgc::log::info("load", "%s %s | %s | n_ctx %d (%s)", entry->id.c_str(),
                       lgc::quant_name(cfg.quant),
                       entry->layers_pinned()
                           ? lgc::log::format("%d GDN + %d attn layers", entry->n_gdn_layer,
                                              entry->n_attn_layer)
                                 .c_str()
                           : "layer split not pinned",
                       n_ctx, n_ctx_source);
    } else {
#ifdef LIGENCE_OPENVINO
        lgc::Artifact artifact;
        if (auto err = lgc::load_artifact(cfg.model_path, artifact)) {
            lgc::log::error("load", "%s", err->c_str());
            return 2;
        }
        if (!cfg.model_id.empty() && cfg.model_id != artifact.id) {
            lgc::log::error("load", "--model-id says '%s' but the artifact is '%s'",
                            cfg.model_id.c_str(), artifact.id.c_str());
            return 2;
        }
        cfg.model_id = artifact.id;

        entry = lgc::find_model(artifact.id);
        if (entry == nullptr) {
            lgc::log::error("load", "artifact resolves to '%s', which is not in the allowlist",
                            artifact.id.c_str());
            return 2;
        }

        // Validate before compiling: a two-minute MoE compile is an expensive
        // way to discover the wrong checkpoint (DESIGN.md §3.1).
        const lgc::ValidationResult v =
            lgc::validate_artifact(*entry, artifact.to_info(cfg.quant));
        for (const std::string& w : v.warnings) lgc::log::warn("load", "%s", w.c_str());
        if (!v.ok) {
            for (const std::string& e : v.errors) lgc::log::error("load", "%s", e.c_str());
            lgc::log::error("load", "%s", "artifact rejected by the allowlist");
            return 2;
        }

        lgc::log::info("load", "%s %s | %d GDN + %d attn layers | weights %s | %s",
                       artifact.id.c_str(), lgc::quant_name(cfg.quant), artifact.n_gdn_layer,
                       artifact.n_attn_layer,
                       lgc::text::human_bytes(artifact.weights_bytes).c_str(),
                       entry->status.c_str());
        lgc::log::info("load", "sampler defaults from %s: temp %.2f top_p %.2f top_k %d",
                       artifact.sampler.provenance.c_str(), artifact.sampler.temperature,
                       artifact.sampler.top_p, artifact.sampler.top_k);

        if (cfg.parallel > 1) {
            // Each sequence needs its own copy of the graph's 80 state
            // variables, and the attention KV alone is gigabytes at long
            // context. Real concurrency waits for the paged pool at M2.
            lgc::log::warn("boot", "--parallel %d ignored: the M1 executor serves one sequence "
                                   "at a time (paged KV lands at M2)", cfg.parallel);
            cfg.parallel = 1;
        }

        const int n_ctx = cfg.n_ctx > 0 ? cfg.n_ctx : artifact.n_ctx_train;
        try {
            backend = lgc::make_ov_backend(artifact, cfg.quant, n_ctx, cfg.device, cfg.cache_dir);
        } catch (const std::exception& e) {
            lgc::log::error("load", "could not bring up the OpenVINO executor: %s", e.what());
            return 1;
        }
        lgc::log::info("load", "n_ctx %d | device %s", n_ctx, cfg.device.c_str());
        lgc::log::info("mem", "%s",
                       "cache is inside the OV graph at M1; the paged KV pool and the GDN "
                       "ledger arrive at M2");
#else
        // Unreachable: parse_args refuses --model on a build without OpenVINO.
        lgc::log::error("boot", "%s", "no backend available for --model in this build");
        return 2;
#endif
    }

    if (cfg.stub && !entry->hashes_pinned()) {
        lgc::log::warn("load", "%s",
                       "allowlist has no arch/template hash for this entry yet; artifact "
                       "provenance is unverified until an IR has been inspected");
    }
    if (cfg.stub) {
        lgc::log::info("mem", "%s", "kv pool and GDN ledger are not allocated before M2");
    }

    lgc::api::SlotPool slots(cfg.parallel);
    lgc::api::Context  ctx{&cfg, backend.get(), &slots};

    lgc::HttpServer server(cfg, ctx);
    g_server.store(&server);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);

    lgc::log::info("http", "listening on %s:%d | %d slot%s", cfg.host.c_str(), cfg.port,
                   slots.total(), slots.total() == 1 ? "" : "s");

    if (!server.listen()) {
        lgc::log::error("http", "could not bind %s:%d", cfg.host.c_str(), cfg.port);
        g_server.store(nullptr);
        return 1;
    }

    g_server.store(nullptr);
    lgc::log::info("http", "%s", "stopped");
    return 0;
}
