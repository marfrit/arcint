#include "config.h"

#include <cerrno>
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

}  // namespace

std::string usage_text() {
    return
        "usage: ligence [options]\n"
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
        "  --parallel N              number of slots (default: 1)\n"
        "  --http-threads N          HTTP worker threads (default: library default)\n"
        "\n"
        "memory\n"
        "  --n-ctx N                 context length (default: the model's own)\n"
        "  --kv-block-size 16|32     KV page size in tokens (default: 32)\n"
        "  --prefill-chunk N         prefill chunk in tokens (0 = unchunked, the\n"
        "                            default; chunking is not bit-exact here)\n"
        "  --prefix-cache-mib N      prefix-cache budget in MiB (default: 0, off)\n"
        "  --no-logits-slice         compute logits for every prompt token (slower,\n"
        "                            and runs out of memory past a few thousand)\n"
        "  --kv-dtype fp16|q8        KV cache element type (default: fp16)\n"
        "  --gdn-checkpoint-budget N GDN checkpoint budget in MiB (default: 512)\n"
        "  --mtp on|off|auto         speculative decoding (default: auto)\n"
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
#ifndef LIGENCE_OPENVINO
    if (!cfg.model_path.empty()) {
        return fail("this build has no OpenVINO backend (configure with -DLIGENCE_OPENVINO=ON)");
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
    if (cfg.parallel < 1) return fail("--parallel must be >= 1");
    if (cfg.http_threads < 0) return fail("--http-threads must be >= 0");
    if (cfg.stub_delay_ms < 0) return fail("--stub-delay-ms must be >= 0");
    if (cfg.device.empty()) return fail("--device must not be empty");
    if (cfg.stub_delay_ms > 0 && !cfg.stub) return fail("--stub-delay-ms only applies to --stub");
    if (cfg.n_ctx < 0) return fail("--n-ctx must be >= 0");
    if (cfg.kv_block_size != 16 && cfg.kv_block_size != 32) {
        return fail("--kv-block-size must be 16 or 32");
    }
    if (cfg.kv_dtype != "fp16" && cfg.kv_dtype != "q8") {
        return fail("--kv-dtype must be fp16 or q8");
    }
    if (cfg.prefill_chunk < 0) return fail("--prefill-chunk must be >= 0");
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
