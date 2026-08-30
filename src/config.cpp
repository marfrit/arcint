#include "config.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "util/log.h"

namespace lgc {
namespace {

constexpr const char* kDefaultStubModel = "qwen3.6-27b-a3b-coder";

bool parse_int(std::string_view s, int& out) {
    if (s.empty()) return false;

    // The string must outlive `end`: strtol's out-pointer addresses the buffer
    // it parsed, so handing it a temporary's c_str() leaves it dangling.
    const std::string text(s);
    char*             end = nullptr;
    errno                 = 0;
    const long long   value = std::strtoll(text.c_str(), &end, 10);

    if (end == text.c_str() || end == nullptr || *end != '\0') return false;
    if (errno == ERANGE || value < INT32_MIN || value > INT32_MAX) return false;
    out = static_cast<int>(value);
    return true;
}

bool parse_double(std::string_view s, double& out) {
    if (s.empty()) return false;

    const std::string text(s);
    char*             end = nullptr;
    errno                 = 0;
    const double      value = std::strtod(text.c_str(), &end);

    if (end == text.c_str() || end == nullptr || *end != '\0') return false;
    // NaN and infinity both have to go: an infinite duration handed to
    // condition_variable::wait_for is undefined, and in practice saturates into
    // the past, which would silently turn "wait forever" into "do not wait" —
    // the exact ambiguity the bounded admission exists to remove.
    if (errno == ERANGE || !std::isfinite(value)) return false;
    out = value;
    return true;
}

}  // namespace

std::string usage_text() {
    return
        "usage: arcint [options]\n"
        "\n"
        "model\n"
        "  --model PATH              OpenVINO IR directory to serve\n"
        "  --stub                    serve the M0 stub backend (no model, no GPU)\n"
        "  --stub-delay-ms N         stub only: artificial latency per token\n"
        "  --model-id ID             allowlist entry to assert (default: from the\n"
        "                            artifact directory name)\n"
        "  --quant q4|q8             weight format (default: q4)\n"
        "  --device DEV              OpenVINO device (default: GPU.0)\n"
        "  --cache-dir PATH          compiled-blob cache directory\n"
        "\n"
        "server\n"
        "  --host ADDR               bind address (default: 127.0.0.1)\n"
        "  --port N                  bind port (default: 8090)\n"
        "  --served-model-name NAME  the name this endpoint reports on\n"
        "                            /v1/models and /props (default: the\n"
        "                            artifact's allowlist id). Presentation\n"
        "                            only: --model-id still decides which\n"
        "                            artifact is accepted\n"
        "  --parallel N              number of lanes (default: 1)\n"
        "  --queue-timeout S         seconds a request waits for a lane before a\n"
        "                            503 with the reservation numbers (default: 0)\n"
        "  --http-threads N          HTTP worker threads (default: library default)\n"
        "\n"
        "memory\n"
        "  --n-ctx N                 context length (default: the model's own)\n"
        "  --kv-block-size 16|32     KV page size in tokens (default: 32)\n"
        "  --prefill-chunk N         prefill chunk in tokens, 0 = unchunked\n"
        "                            (default: 2048; bounds activation memory)\n"
        "  --paged-kv u8|f16         KV precision on the paged path (default: u8).\n"
        "  --cache-grid N            prefix-cache snapshot grid in tokens (default: 128; 0 = the\n"
        "                            prefill chunk). A snapshot lands on the last multiple of N\n"
        "                            below the prompt length; a coarser grid re-prefills the\n"
        "                            remainder on every continuation. DESIGN 7.0.2j.\n"
        "  --gate-pad N              widen the shared-expert gate to N columns (default: 0,\n"
        "                            off). 16 is the measured setting: -13% prefill wall,\n"
        "                            -5% decode; pays off below ~500 answer tokens per 12k\n"
        "                            prompt tokens. See DESIGN 7.0.2g.\n"
        "                            u8 is 11.3 KiB/token against f16's 20.0, which\n"
        "                            is what makes two lanes at depth fit. It is the\n"
        "                            default because it is the setting that does not\n"
        "                            refuse, not because it is faster -- f16 wins on\n"
        "                            prefill and, past ~50k, on decode too. f16 also\n"
        "                            costs prefix-cache reserve: the pool is sized in\n"
        "                            bytes and live pages are a fixed count, so the\n"
        "                            whole difference lands there\n"
        "  --prefix-cache-mib N      prefix-cache budget in MiB (default: 0, off)\n"
        "  --no-logits-slice         compute logits for every prompt token (slower,\n"
        "                            and runs out of memory past a few thousand)\n"
        "  --kv-dtype fp16|fp32      stored KV element type (default: fp16;\n"
        "                            fp32 is what the artifact exports)\n"
        "  --gdn-checkpoint-budget N GDN checkpoint budget in MiB (default: 512)\n"
        "  --mtp on|off|auto         speculative decoding (default: auto)\n"
        "  --draft N                 speculative draft length, 0 = off\n"
        "  --draft-ngram K           drafter match length (default: 3)\n"
        "  --custom-kernels FILE     OpenVINO custom-layer XML (hand-written OpenCL)\n"
        "  --offload-ratio N         %% of MoE expert weights to stream, 0 = all resident\n"
        "  --no-paged                stateful reference path instead of the paged executor\n"
        "  --emb-device DEV          run the embeddings gather elsewhere (default: --device)\n"
        "  --mtp-device DEV          run the MTP head elsewhere (default: --device)\n"
        "\n"
        "misc\n"
        "  -v, -vv                   raise console verbosity\n"
        "  --version                 print version and exit\n"
        "  -h, --help                print this help and exit\n";
}

ArgParse parse_args(int argc, char** argv, Config& cfg) {
    auto fail = [](std::string msg) {
        ArgParse r;
        r.ok    = false;
        r.error = std::move(msg);
        return r;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        auto value = [&](std::string_view& out) {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };

        std::string_view v;

        if (arg == "-h" || arg == "--help") {
            cfg.show_help = true;
        } else if (arg == "--version") {
            cfg.show_version = true;
        } else if (arg == "-v") {
            cfg.verbosity = 1;
        } else if (arg == "-vv") {
            cfg.verbosity = 2;
        } else if (arg == "--stub") {
            cfg.stub = true;
        } else if (arg == "--model") {
            if (!value(v)) return fail("--model needs a path");
            cfg.model_path = std::string(v);
        } else if (arg == "--model-id") {
            if (!value(v)) return fail("--model-id needs a value");
            cfg.model_id = std::string(v);
        } else if (arg == "--quant") {
            if (!value(v)) return fail("--quant needs a value");
            const auto q = quant_parse(v);
            if (!q) return fail("--quant must be q4 or q8");
            cfg.quant = *q;
        } else if (arg == "--device") {
            if (!value(v)) return fail("--device needs a value");
            cfg.device = std::string(v);
        } else if (arg == "--cache-dir") {
            if (!value(v)) return fail("--cache-dir needs a path");
            cfg.cache_dir = std::string(v);
        } else if (arg == "--host") {
            if (!value(v)) return fail("--host needs a value");
            cfg.host = std::string(v);
        } else if (arg == "--port") {
            if (!value(v) || !parse_int(v, cfg.port)) return fail("--port needs an integer");
        } else if (arg == "--parallel") {
            if (!value(v) || !parse_int(v, cfg.parallel)) {
                return fail("--parallel needs an integer");
            }
        } else if (arg == "--cache-grid") {
            if (!value(v) || !parse_int(v, cfg.cache_grid)) return fail("--cache-grid needs an integer");
        } else if (arg == "--gate-pad") {
            if (!value(v) || !parse_int(v, cfg.gate_pad)) return fail("--gate-pad needs an integer");
        } else if (arg == "--paged-kv") {
            if (!value(v)) return fail("--paged-kv needs u8 or f16");
            cfg.paged_kv = std::string(v);
        } else if (arg == "--served-model-name") {
            // An empty value is refused rather than folded into "not given":
            // `--served-model-name ""` asks for a name, and silently serving the
            // canonical one instead is how a roster ends up pinned to a name
            // nobody chose.
            if (!value(v) || v.empty()) {
                return fail("--served-model-name needs a non-empty name");
            }
            cfg.served_model_name = std::string(v);
        } else if (arg == "--queue-timeout") {
            if (!value(v) || !parse_double(v, cfg.queue_timeout_s)) {
                return fail("--queue-timeout needs a number of seconds");
            }
        } else if (arg == "--stub-delay-ms") {
            if (!value(v) || !parse_int(v, cfg.stub_delay_ms)) {
                return fail("--stub-delay-ms needs an integer");
            }
        } else if (arg == "--http-threads") {
            if (!value(v) || !parse_int(v, cfg.http_threads)) {
                return fail("--http-threads needs an integer");
            }
        } else if (arg == "--n-ctx") {
            if (!value(v) || !parse_int(v, cfg.n_ctx)) return fail("--n-ctx needs an integer");
        } else if (arg == "--prefill-chunk") {
            if (!value(v) || !parse_int(v, cfg.prefill_chunk)) {
                return fail("--prefill-chunk needs an integer");
            }
        } else if (arg == "--prefix-cache-mib") {
            if (!value(v) || !parse_int(v, cfg.prefix_cache_mib)) {
                return fail("--prefix-cache-mib needs an integer");
            }
        } else if (arg == "--no-logits-slice") {
            cfg.slice_logits = false;
        } else if (arg == "--kv-block-size") {
            if (!value(v) || !parse_int(v, cfg.kv_block_size)) {
                return fail("--kv-block-size needs an integer");
            }
        } else if (arg == "--kv-dtype") {
            if (!value(v)) return fail("--kv-dtype needs a value");
            cfg.kv_dtype = std::string(v);
        } else if (arg == "--gdn-checkpoint-budget") {
            if (!value(v) || !parse_int(v, cfg.gdn_checkpoint_budget_mib)) {
                return fail("--gdn-checkpoint-budget needs an integer");
            }
        } else if (arg == "--draft") {
            if (!value(v) || !parse_int(v, cfg.draft_tokens)) {
                return fail("--draft needs an integer");
            }
        } else if (arg == "--draft-ngram") {
            if (!value(v) || !parse_int(v, cfg.draft_ngram)) {
                return fail("--draft-ngram needs an integer");
            }
        } else if (arg == "--custom-kernels") {
            if (!value(v)) return fail("--custom-kernels needs a path");
            cfg.custom_kernels = std::string(v);
        } else if (arg == "--no-paged") {
            cfg.paged = false;
        } else if (arg == "--emb-device") {
            if (!value(v)) return fail("--emb-device needs a device");
            cfg.emb_device = std::string(v);
        } else if (arg == "--mtp-device") {
            if (!value(v)) return fail("--mtp-device needs a device");
            cfg.mtp_device = std::string(v);
        } else if (arg == "--offload-ratio") {
            if (!value(v) || !parse_int(v, cfg.offload_ratio)) {
                return fail("--offload-ratio needs an integer");
            }
        } else if (arg == "--mtp") {
            if (!value(v)) return fail("--mtp needs a value");
            cfg.mtp = std::string(v);
        } else {
            return fail(log::format("unknown option '%s' (try --help)",
                                    std::string(arg).c_str()));
        }
    }

    if (cfg.show_help || cfg.show_version) return {};

    // ------------------------------------------------------------ validation
    if (cfg.model_path.empty() && !cfg.stub) {
        return fail("nothing to serve: pass --model PATH, or --stub for the M0 skeleton");
    }
    if (!cfg.model_path.empty() && cfg.stub) {
        return fail("--model and --stub are mutually exclusive");
    }
#ifndef ARCINT_OPENVINO
    if (!cfg.model_path.empty()) {
        return fail("this build has no OpenVINO backend (configure with -DARCINT_OPENVINO=ON)");
    }
#endif

    // With --model the allowlist entry comes from the artifact's own directory
    // name, so --model-id is optional there; when it is given it is a claim that
    // load_artifact checks against what it actually read.
    if (cfg.model_id.empty() && cfg.stub) cfg.model_id = kDefaultStubModel;

    const ModelEntry* entry = nullptr;
    if (!cfg.model_id.empty()) {
        entry = find_model(cfg.model_id);
        if (entry == nullptr) {
            std::string known;
            for (const std::string& id : model_ids()) {
                if (!known.empty()) known += ", ";
                known += id;
            }
            return fail(log::format("'%s' is not in the allowlist (known: %s)",
                                    cfg.model_id.c_str(), known.c_str()));
        }
        if (!entry->accepts(cfg.quant)) {
            return fail(log::format("%s does not allow quant %s", cfg.model_id.c_str(),
                                    quant_name(cfg.quant)));
        }
    }

    if (cfg.port < 1 || cfg.port > 65535) return fail("--port must be in [1, 65535]");
    // A served name that is empty or blank would put an unusable id on
    // /v1/models, and a roster discovering names from there would take it.
    if (!cfg.served_model_name.empty() &&
        cfg.served_model_name.find_first_not_of(" \t\r\n") == std::string::npos) {
        return fail("--served-model-name cannot be blank");
    }
    if (cfg.parallel < 1) return fail("--parallel must be >= 1");
    if (!(cfg.queue_timeout_s >= 0.0 && cfg.queue_timeout_s <= 3600.0)) {
        return fail("--queue-timeout must be between 0 and 3600 seconds");
    }
    if (cfg.http_threads < 0) return fail("--http-threads must be >= 0");
    if (cfg.stub_delay_ms < 0) return fail("--stub-delay-ms must be >= 0");
    if (cfg.device.empty()) return fail("--device must not be empty");
    if (cfg.stub_delay_ms > 0 && !cfg.stub) return fail("--stub-delay-ms only applies to --stub");
    if (cfg.n_ctx < 0) return fail("--n-ctx must be >= 0");
    if (cfg.kv_block_size != 16 && cfg.kv_block_size != 32) {
        return fail("--kv-block-size must be 16 or 32");
    }
    if (cfg.kv_dtype == "q8") {
        // Refused deliberately. Retyping the KV state to i8 is a numeric cast,
        // not quantisation: there are no scales, so every value rounds to an
        // integer. It does not crash or produce garbage — it produces a
        // plausible, quietly worse answer, which is the failure mode this
        // engine exists to refuse. Real q8 KV needs per-block scales, and the
        // paged path already gets them from the plugin (DESIGN.md §3.3).
        return fail("--kv-dtype q8 is not implemented: a plain cast to int8 has no scales and "
                    "silently degrades the answer. Use fp16, or the paged path once it lands.");
    }
    if (cfg.cache_grid < 0 || cfg.cache_grid > 65536) {
        return fail(log::format("--cache-grid must be 0 (the chunk) or 1..65536, not %d", cfg.cache_grid));
    }
    if (cfg.gate_pad < 0 || cfg.gate_pad == 1 || cfg.gate_pad > 4096) {
        return fail(log::format("--gate-pad must be 0 (off) or 2..4096, not %d", cfg.gate_pad));
    }
    if (cfg.paged_kv != "u8" && cfg.paged_kv != "f16") {
        return fail(log::format("--paged-kv must be u8 or f16, not '%s'", cfg.paged_kv.c_str()));
    }
    if (cfg.kv_dtype != "fp16" && cfg.kv_dtype != "fp32") {
        return fail("--kv-dtype must be fp16 or fp32");
    }
    if (cfg.offload_ratio < 0 || cfg.offload_ratio > 100) {
        return fail("--offload-ratio must be a percentage in [0, 100]");
    }
    if (cfg.prefill_chunk < 0) return fail("--prefill-chunk must be >= 0");

    // The prefill grid and the cache grid have to be the same grid. A cache hit
    // is where a warm run starts, and a warm run must present the model the same
    // chunk boundaries a cold one did, or the two compute different logits for
    // the same tokens (DESIGN §3.2). That requires every checkpoint position to
    // be a prefill boundary, so the chunk size must be a whole number of blocks.
    if (cfg.prefix_cache_mib > 0) {
        if (cfg.prefill_chunk == 0) {
            return fail("prefix caching needs a prefill grid to align to: "
                        "--prefill-chunk 0 (unchunked) cannot guarantee that a warm "
                        "run matches a cold one");
        }
        if (cfg.kv_block_size <= 0 || cfg.prefill_chunk % cfg.kv_block_size != 0) {
            return fail(log::format(
                "--prefill-chunk (%d) must be a multiple of --kv-block-size (%d) so that "
                "cache checkpoints land on prefill boundaries",
                cfg.prefill_chunk, cfg.kv_block_size));
        }
    }
    if (cfg.draft_tokens < 0) return fail("--draft must be >= 0");
    if (cfg.draft_ngram < 1) return fail("--draft-ngram must be >= 1");
    if (cfg.prefix_cache_mib < 0) return fail("--prefix-cache-mib must be >= 0");
    if (cfg.gdn_checkpoint_budget_mib < 0) {
        return fail("--gdn-checkpoint-budget must be >= 0");
    }
    if (cfg.mtp != "on" && cfg.mtp != "off" && cfg.mtp != "auto") {
        return fail("--mtp must be on, off, or auto");
    }
    if (cfg.mtp == "on" && entry != nullptr && !entry->has_mtp_head) {
        return fail(log::format("--mtp on: %s ships no MTP head", cfg.model_id.c_str()));
    }

    return {};
}

}  // namespace lgc
