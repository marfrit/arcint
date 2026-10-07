#include <atomic>
#include <csignal>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "api/handlers.h"
#include "build_info.h"
#include "config.h"
#include "core/artifact.h"
#include "exec/backend.h"
#include "exec/flash_next_offload.h"
#include "exec/segment_plan.h"
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

// --inspect-artifact: the artifact contract, printed. Every number here is
// on-disk arithmetic -- bytes, hashes, port names, the buffer set the expert
// index implies. No card is opened, no IR compiled, nothing measured: what a
// card makes of this budget is a separate commit and a separate number (the
// discipline that keeps a predicted GiB out of a measured column).
// Inside the OpenVINO branch because that is where an artifact is read at all;
// the stub backend serves no artifact to inspect.
#ifdef ARCINT_OPENVINO
void print_artifact_inspection(const lgc::Config& cfg, const lgc::Artifact& a) {
    const double kGiB = 1.0 / (1024.0 * 1024.0 * 1024.0);
    const auto   GiB  = [&](uint64_t bytes) { return static_cast<double>(bytes) * kGiB; };

    std::printf("artifact inspection: %s\n", a.dir.c_str());

    const lgc::ModelEntry* entry = lgc::find_by_artifact(a.directory_name);
    if (entry == nullptr) {
        std::printf("  allowlist: NOT admitted -- no entry claims the directory name "
                    "'%s'\n",
                    a.directory_name.c_str());
    } else {
        std::printf("  allowlist: admitted as %s (status: %s)\n", entry->id.c_str(),
                    entry->status.c_str());
    }

    std::printf("  geometry: %d layers = %d GDN + %d attn, hidden %d, hc_count %d, "
                "experts %d (%s), ctx %d\n",
                a.n_layer, a.n_gdn_layer, a.n_attn_layer, a.n_embd, a.hc_count, a.n_expert,
                a.moe ? "moe" : "dense", a.n_ctx_train);
    std::printf("  hashes: arch %s [%s], template %s, tokenizer %s\n", a.arch_hash.c_str(),
                a.segmented() ? "chain over every segment xml, in segment order"
                              : "the single language-model xml",
                a.template_hash.c_str(), a.tokenizer_hash.c_str());
    std::printf("  weights: %s (%llu B) in %zu segment .bin(s)\n",
                lgc::text::human_bytes(a.weights_bytes).c_str(),
                static_cast<unsigned long long>(a.weights_bytes), a.segments.size());

    if (a.segmented()) {
        std::printf("  segments: %zu (declared by serving-shape.json segment_layers), "
                    "hidden boundary port %d x %d = %d wide\n",
                    a.segments.size(), a.hc_count, a.n_embd, a.hc_count * a.n_embd);
        for (const lgc::ArtifactSegment& s : a.segments) {
            std::printf("    seg %d  layers [%2d,%2d)  dir %-9s  %s%s  in_w %5d  attn %d gdn %d  "
                        "bin %11llu B  xml %s\n",
                        s.index, s.layer_lo, s.layer_hi, s.dir.c_str(), s.first ? "first " : "",
                        s.last ? "last" : (s.has_ple ? "ple" : ""), s.inputs_embeds_width,
                        s.attn_layers, s.gdn_layers,
                        static_cast<unsigned long long>(s.lm_bin_bytes), s.xml_sha.c_str());
        }
    } else {
        std::printf("  segments: 1 (the whole model; no segment_layers in serving-shape.json)\n");
    }

    if (a.expert_bodies_path.empty()) {
        std::printf("  expert bodies: the artifact carries no expert_bodies blob\n");
        return;
    }
    std::printf("  expert bodies: %s, %llu B (%.2f GiB), %zu bodies indexed\n",
                a.expert_bodies_path.c_str(),
                static_cast<unsigned long long>(a.expert_bodies_bytes),
                GiB(a.expert_bodies_bytes), a.expert_bodies.size());
    if (!a.segmented()) return;

    // The graph contract is segplan's, not the loader's: it is the runtime's
    // first call, so it is printed here as the runtime would meet it.
    try {
        const lgc::segplan::Plan plan =
            lgc::segplan::plan_segments(a.serving_shape, a.n_layer, a.n_embd, a.hc_count);
        std::printf("  segment plan: accepted\n");
        for (size_t k = 0; k < lgc::segplan::kKinds.size(); ++k) {
            std::string dims = "[";
            for (size_t d = 0; d < plan.slot_shape[k].size(); ++d) {
                dims += (d ? " x " : "") + std::to_string(plan.slot_shape[k][d]);
            }
            dims += "]";
            std::printf("    kind %-4s slot shape %s  %llu B\n", lgc::segplan::kKinds[k],
                        dims.c_str(), static_cast<unsigned long long>(plan.buffer_bytes(k)));
        }
        std::printf("    one buffer set: %zu slots, %llu B (%.2f GiB) -- resident for EVERY segment\n",
                    plan.slots_per_segment,
                    static_cast<unsigned long long>(plan.buffer_set_bytes()),
                    GiB(plan.buffer_set_bytes()));
        for (const lgc::segplan::SegmentSpec& s : plan.segments) {
            uint64_t bytes = 0;
            size_t   ops   = 0;
            for (const lgc::segplan::RefillOp& op : lgc::segplan::refill_ops(plan, s.index)) {
                bytes += op.bytes;
                ++ops;
            }
            std::printf("    refill for segment %d: %zu bodies, %llu B (%.2f GiB)\n", s.index, ops,
                        static_cast<unsigned long long>(bytes), GiB(bytes));
        }
        std::printf("    per forward the chain reads the whole blob: %llu B (%.2f GiB)\n",
                    static_cast<unsigned long long>(a.expert_bodies_bytes),
                    GiB(a.expert_bodies_bytes));
    } catch (const std::exception& e) {
        std::printf("  SEGMENT PLAN REFUSED: %s\n", e.what());
    }

    if (entry == nullptr) return;
    const lgc::ValidationResult v = lgc::validate_artifact(*entry, a.to_info(cfg.quant));
    for (const std::string& w : v.warnings) std::printf("  validate WARN: %s\n", w.c_str());
    for (const std::string& err : v.errors) std::printf("  validate ERROR: %s\n", err.c_str());
    std::printf("  validate: %s\n", v.ok ? "OK" : "REJECTED by the allowlist");
}
#endif  // ARCINT_OPENVINO

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
    if (cfg.flash_next_offload_plan) {
        // WP7 dry-run: size the Flash-Next expert-offload serving plan for the
        // single-A770 target at the measured per-layer LRU hit-rate. Device-free
        // (no card, no served graph); the live expert gather it sizes is parked
        // on the backbone IR (FIX A). Target budget = docs/serving-config-flash-next.md.
        const double GiB = static_cast<double>(1ull << 30);
        const auto B = [&](double g) { return static_cast<uint64_t>(g * GiB + 0.5); };
        const double vram = 15.0, dram = 44.0, backbone = 2.3, kv = 3.0, act = 2.0;
        const lgc::OffloadPlan p = lgc::flash_next_plan(
            B(vram), B(dram), B(backbone), B(kv), B(act), cfg.flash_next_offload_hit);
        const bool refuse = lgc::flash_next_offload_must_refuse(p, /*floor_tps=*/0.0);
        std::printf("Flash-Next expert-offload plan (single-A770 target, dry-run):\n");
        std::printf("  card budget: VRAM %.1f GiB, DRAM %.1f GiB "
                    "(backbone %.1f + KV %.1f + activations %.1f reserved on VRAM)\n",
                    vram, dram, backbone, kv, act);
        std::printf("  measured feeds: DRAM %.1f GiB/s, NVMe miss %.2f GiB/s; MTP amort 1x (GGUF has no MTP head)\n",
                    lgc::kFlashNextDramBwGiBs, lgc::kFlashNextNvmeBwGiBs);
        std::printf("  PLE table %.2f GiB DRAM-resident: %s\n",
                    static_cast<double>(lgc::kFlashNextTableBytes) / GiB,
                    p.table_fits_dram ? "yes" : "NO (random hashed gather would be seek-bound)");
        std::printf("  resident expert pool: %.2f GiB (%.0f%% of %.2f GiB), %d slots/layer "
                    "(VRAM %.2f + DRAM %.2f GiB)\n",
                    static_cast<double>(p.budget.expert_bytes) / GiB, p.resident_frac * 100.0,
                    static_cast<double>(512ull * 48 * lgc::kFlashNextSliceBytes) / GiB,
                    p.slots_per_layer,
                    static_cast<double>(p.budget.vram_for_experts) / GiB,
                    static_cast<double>(p.budget.dram_for_experts) / GiB);
        std::printf("  per-layer LRU hit-rate (measured input): %.1f%%\n",
                    cfg.flash_next_offload_hit * 100.0);
        std::printf("  projected decode: %.1f t/s (%s) -- bandwidth-bound estimate, not served\n",
                    p.projected_tps, lgc::flash_next_regime_name(p.regime));
        std::printf("  verdict: %s\n", refuse ? "REFUSE (PLE table cannot be DRAM-resident)"
                                              : "ADMIT (table-residency only; the t/s above is advisory)");
        return refuse ? 1 : 0;
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
        if (cfg.named_lanes()) {
            // the first lane's cap is the served context; a cap over the
            // artifact's trained context is refused here, as the libllama
            // engine refuses it at load
            for (size_t i = 0; i < cfg.lane_ctx.size(); ++i) {
                if (entry->n_ctx_train > 0 && cfg.lane_ctx[i] > entry->n_ctx_train) {
                    lgc::log::error("boot", "--lane-ctx %d (lane '%s') exceeds %s's trained context %d",
                                    cfg.lane_ctx[i], cfg.lane_names[i].c_str(), entry->id.c_str(),
                                    entry->n_ctx_train);
                    return 2;
                }
            }
            n_ctx        = cfg.lane_ctx[0];
            n_ctx_source = "--lane-ctx, the first lane";
        }
        if (n_ctx <= 0) {
            n_ctx        = entry->n_ctx_train > 0 ? entry->n_ctx_train : kStubDefaultNCtx;
            n_ctx_source = entry->n_ctx_train > 0 ? "allowlist" : "stub fallback";
        }
        lgc::ModelEntry stub_entry = *entry;
        if (auto err = lgc::apply_operator_defaults(cfg, stub_entry.sampler)) {
            lgc::log::error("boot", "%s", err->c_str());
            return 2;
        }
        backend = lgc::make_stub_backend(stub_entry, cfg.quant, n_ctx, cfg.stub_delay_ms,
                                         cfg.served_model_name, cfg.lane_ctx);

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
    } else if (cfg.engine == "llama") {
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
        lgc::log::error("boot", "%s", "this build carries no llama.cpp executor (configure with -DARCINT_LLAMA=ON)");
        return 2;
#endif
    } else {
#ifdef ARCINT_OPENVINO
        lgc::Artifact artifact;
        if (auto err = lgc::load_artifact(cfg.model_path, artifact,
                                         /*require_allowlisted=*/!cfg.inspect_artifact)) {
            lgc::log::error("load", "%s", err->c_str());
            return 2;
        }
        if (cfg.inspect_artifact) {
            print_artifact_inspection(cfg, artifact);
            return 0;
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
        if (auto err = lgc::apply_operator_defaults(cfg, artifact.sampler)) {
            lgc::log::error("boot", "%s", err->c_str());
            return 2;
        }
        lgc::log::info("load", "sampler defaults from %s: temp %.2f top_p %.2f top_k %d",
                       artifact.sampler.provenance.c_str(), artifact.sampler.temperature,
                       artifact.sampler.top_p, artifact.sampler.top_k);

        // The allowlist says what an artifact IS; this says whether anything can
        // DRIVE it. A segmented artifact passes the allowlist (its pin is the
        // chain hash) and would otherwise compile ONE segment and answer under a
        // full-depth pin.
        if (const std::string refuse = lgc::serve_refusal_for(artifact); !refuse.empty()) {
            lgc::log::error("load", "%s", refuse.c_str());
            return 2;
        }

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

    if (cfg.named_lanes()) {
        lgc::log::info("load", "%zu named lanes; the model field picks the lane, an empty one or '%s' the "
                               "first, any other name a 404 (DESIGN 4.2):",
                       cfg.lane_names.size(), backend->status().id.c_str());
        for (size_t i = 0; i < cfg.lane_names.size(); ++i)
            lgc::log::info("load", "  lane %zu '%s': n_ctx %d", i, cfg.lane_names[i].c_str(), cfg.lane_ctx[i]);
    } else if (!cfg.served_model_name.empty()) {
        lgc::log::info("load", "served as '%s' (--served-model-name); the artifact is '%s' and "
                               "the allowlist assertion is unchanged",
                       backend->status().served_id.c_str(), backend->status().id.c_str());
    }

    lgc::api::SlotPool slots(cfg.parallel);
    lgc::api::Context  ctx{&cfg, backend.get(), &slots};
    // named lanes: a pool each, admitting by name
    std::vector<std::unique_ptr<lgc::api::SlotPool>> lane_pools;
    ctx.lanes = lgc::api::make_lanes(cfg, lane_pools);

    lgc::HttpServer server(cfg, ctx);
    g_server.store(&server);
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

    if (!server.listen()) {
        lgc::log::error("http", "could not bind %s:%d", cfg.host.c_str(), cfg.port);
        g_server.store(nullptr);
        return 1;
    }

    g_server.store(nullptr);
    lgc::log::info("http", "%s", "stopped");
    return 0;
}
