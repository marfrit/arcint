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
        backend = lgc::make_stub_backend(*entry, cfg.quant, n_ctx, cfg.stub_delay_ms,
                                         cfg.served_model_name);

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
#ifdef ARCINT_OPENVINO
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

        if (cfg.parallel > 1 && !cfg.paged) {
            // The stateful graph has one internal state, so a second sequence
            // would overwrite the first's. The paged path is what made lanes
            // possible (M6): its state lives in arcint's own rows and pages.
            lgc::log::warn("boot", "--parallel %d with --no-paged: the stateful reference "
                                   "executor serves one sequence at a time and the other "
                                   "lane(s) will wait", cfg.parallel);
        }

        const int n_ctx = cfg.n_ctx > 0 ? cfg.n_ctx : artifact.n_ctx_train;
        try {
            backend = lgc::make_ov_backend(artifact, cfg, n_ctx);
        } catch (const std::exception& e) {
            lgc::log::error("load", "could not bring up the OpenVINO executor: %s", e.what());
            return 1;
        }
        // What the engine actually settled on, not what was asked for: the
        // reservation may have clamped n_ctx and halved the chunk to make the
        // configuration fit (§7.0.2a), and a line that repeats the request
        // instead of the outcome is how a clamped run gets read as the one that
        // was configured.
        const lgc::Reservation& res = backend->status().reservation;
        const int eff_ctx   = backend->status().n_ctx > 0 ? backend->status().n_ctx : n_ctx;
        const int eff_chunk = res.measured ? res.prefill_chunk : cfg.prefill_chunk;
        lgc::log::info("load", "n_ctx %d | device %s | prefill %s | %d lane%s", eff_ctx,
                       cfg.device.c_str(),
                       eff_chunk > 0
                           ? lgc::log::format("chunked at %d tok", eff_chunk).c_str()
                           : "unchunked",
                       cfg.parallel, cfg.parallel == 1 ? "" : "s");
        if (cfg.prefill_chunk > 0) {
            lgc::log::verbose("load",
                              "prompts over %d tokens are prefilled in chunks; chunk boundaries "
                              "are not bit-exact on this backend (DESIGN.md 3.2), shorter "
                              "prompts are unaffected",
                              cfg.prefill_chunk);
        }
        if (cfg.prefix_cache_mib > 0) {
            lgc::log::warn("load", "%s",
                           "the prefix cache checkpoints mid-prompt, which splits the prefill. "
                           "Repeating an identical prompt is gated byte-equal; a CONTINUATION "
                           "of a cached prompt takes a different prefill split than a cold run "
                           "and inherits this backend's chunk non-exactness (DESIGN.md 3.2).");
        }
        if (cfg.prefix_cache_mib > 0) {
            lgc::log::info("mem", "prefix cache %d MiB, block %d tok | state lives in the OV "
                                  "graph and is checkpointed whole (KV and GDN together)",
                           cfg.prefix_cache_mib, cfg.kv_block_size);
        } else {
            lgc::log::info("mem", "%s", "prefix cache off (--prefix-cache-mib enables it)");
        }
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

    if (!cfg.served_model_name.empty()) {
        lgc::log::info("load", "served as '%s' (--served-model-name); the artifact is '%s' and "
                               "the allowlist assertion is unchanged",
                       backend->status().served_id.c_str(), backend->status().id.c_str());
    }

    lgc::api::SlotPool slots(cfg.parallel);
    lgc::api::Context  ctx{&cfg, backend.get(), &slots};

    lgc::HttpServer server(cfg, ctx);
    g_server.store(&server);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);

    // The endpoint-identity line: address, lanes, and the name this process
    // answers to. A journal that does not say which name is served is no help
    // when a roster discovered one from /v1/models and a client is using
    // another.
    lgc::log::info("http", "listening on %s:%d | %d slot%s | serving '%s'", cfg.host.c_str(),
                   cfg.port, slots.total(), slots.total() == 1 ? "" : "s",
                   backend->status().served_id.c_str());

    if (!server.listen()) {
        lgc::log::error("http", "could not bind %s:%d", cfg.host.c_str(), cfg.port);
        g_server.store(nullptr);
        return 1;
    }

    g_server.store(nullptr);
    lgc::log::info("http", "%s", "stopped");
    return 0;
}
