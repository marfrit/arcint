#include "config.h"

#include "core/sampling.h"

#include <array>
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

// Bit width per --paged-kv value, per docs/design-m8-asymmetric-kv.md §3:
// the same five element types the plugin's own KV_CACHE_PRECISION
// validation accepts. This is the
// single source of truth for both "is this a legal value at all"
// (is_paged_kv_value, parse_paged_kv) and "does an actual compiled port
// match what was requested" (kv_precision_bitwidth_matches, the M8 port
// audit in backend_ov.cpp) -- one table, so the two rules cannot quietly
// drift onto different alphabets.
struct PagedKvValue {
    const char* name;
    int         bits;
};
constexpr std::array<PagedKvValue, 5> kPagedKvValues = {{
    {"f16", 16},
    {"u8", 8},
    {"i8", 8},
    {"u4", 4},
    {"i4", 4},
}};

bool paged_kv_bits(std::string_view value, int& bits) {
    for (const auto& v : kPagedKvValues) {
        if (value == v.name) {
            bits = v.bits;
            return true;
        }
    }
    return false;
}

bool is_paged_kv_value(std::string_view s) {
    int bits = 0;
    return paged_kv_bits(s, bits);
}

// The one known packed-4-bit shape (on-card finding, 2026-09-02): a 4-bit
// request stored in an 8-bit-typed port. Int-only so both
// kv_precision_bitwidth_matches and kv_precision_is_packed_four_bit share
// one definition of "known packing" rather than two that could drift.
bool kv_precision_is_packed_four_bit_bits(int requested_bits, int actual_bits) {
    return requested_bits == 4 && actual_bits == 8;
}

}  // namespace

bool parse_paged_kv(const std::string& spec, std::string& key, std::string& value) {
    const size_t colon = spec.find(':');
    if (colon == std::string::npos) {
        if (!is_paged_kv_value(spec)) return false;
        key = value = spec;
        return true;
    }
    // At most one colon: "u8:i4:x" is not a two-sided spec, it is garbage.
    if (spec.find(':', colon + 1) != std::string::npos) return false;
    const std::string k = spec.substr(0, colon);
    const std::string v = spec.substr(colon + 1);
    if (!is_paged_kv_value(k) || !is_paged_kv_value(v)) return false;
    key   = k;
    value = v;
    return true;
}

// M8 port audit rule (docs/design-m8-asymmetric-kv.md §4b; red case found
// on the card 2026-09-02): a plugin is free to store a u8 request as i8 (or
// a u4 request as i4) -- that is its own signedness choice, not a precision
// downgrade, and comparing exact element types refused every symmetric u8
// load over exactly this (the plugin compiled key_cache/value_cache as i8
// for a u8 request). The actual hazard the audit exists to catch is a side
// that kept the OLD bitwidth when a DIFFERENT one was requested -- an
// asymmetric u8:i4 request silently served as u8:u8, say -- so the
// comparison is BITWIDTH only.
//
// Amended (on-card finding, 2026-09-02): this GPU plugin generation stores
// paged KV in 8-bit-TYPED ports always -- an honest 4-bit-typed port breaks
// the plugin's own compile, and the working §7.0.3 u4 deployment ran
// 8-bit-typed ports throughout. A 4-bit precision (u4/i4) is therefore
// config-side packing that is INVISIBLE at the port level: a 4-bit request
// landing on an 8-bit port is the expected shape on this plugin generation,
// not a downgrade, so it passes too (the caller logs it, see backend_ov.cpp).
//
// Consequence, stated honestly: for a 4-bit request this audit can no
// longer catch a silently-unpacked world (an i4 request served as plain u8
// data in an 8-bit container would look identical to correctly packed i4 at
// this port-type level). The plugin-side config-level guard (patches/0009
// v3) covers the asymmetric-corruption case instead, and symmetric packing
// is verified by measured KV footprint (kv_bytes_token_ / the cost model
// below), not by this audit.
//
// `requested` and `actual` are each one of --paged-kv's five values;
// anything outside that alphabet counts as a mismatch rather than passing
// by default.
bool kv_precision_bitwidth_matches(const std::string& requested, const std::string& actual) {
    int requested_bits = 0, actual_bits = 0;
    if (!paged_kv_bits(requested, requested_bits) || !paged_kv_bits(actual, actual_bits)) {
        return false;
    }
    if (requested_bits == actual_bits) return true;
    return kv_precision_is_packed_four_bit_bits(requested_bits, actual_bits);
}

// True exactly for the known packed-4-bit shape above: a 4-bit request
// (u4/i4) landing on this plugin generation's 8-bit-typed port. Split out
// from kv_precision_bitwidth_matches so the caller can pick the right log
// message (the "packed 4-bit" wording vs. the generic same-bitwidth-alias
// wording) without re-deriving the bit widths itself.
bool kv_precision_is_packed_four_bit(const std::string& requested, const std::string& actual) {
    int requested_bits = 0, actual_bits = 0;
    if (!paged_kv_bits(requested, requested_bits) || !paged_kv_bits(actual, actual_bits)) {
        return false;
    }
    return kv_precision_is_packed_four_bit_bits(requested_bits, actual_bits);
}

bool parse_u64_strict(const std::string& s, uint64_t& out) {
    if (s.empty()) return false;

    char*                    end = nullptr;
    errno                        = 0;
    const unsigned long long v   = std::strtoull(s.c_str(), &end, 10);

    // Full consumption: refuses trailing garbage ("8e9" parses 8 and leaves
    // "e9" unconsumed) and empty input (end == s.c_str()).
    if (end != s.c_str() + s.size()) return false;
    if (errno == ERANGE) return false;
    // Round-trip: refuses anything that would not print back as itself --
    // a leading '-' (strtoull wraps it to a huge unsigned value rather than
    // rejecting it), a leading '+', or leading zeros. This is what actually
    // catches the '-' case, not a sign check, because strtoull's own
    // consumption of a leading '-' already satisfies the check above.
    if (std::to_string(v) != s) return false;

    out = static_cast<uint64_t>(v);
    return true;
}

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
        "  --pin-dispatch N          pin each lane's dispatch thread to N+lane_index\n"
        "                            (default: -1, off). Linux only. The pin persists on\n"
        "                            the pooled OS thread, not the request -- it is never\n"
        "                            undone once set\n"
        "\n"
        "memory\n"
        "  --n-ctx N                 context length. Omitted: adopts whatever the\n"
        "                            paged path's auto-fit computes (M7) -- that\n"
        "                            adoption IS the fit, and it can come out below\n"
        "                            the model's own maximum. --n-ctx 0 (typed) is\n"
        "                            NOT the same as omitting the flag: it asks\n"
        "                            explicitly for the model's own maximum, checked\n"
        "                            against the fit like any other explicit value --\n"
        "                            it can refuse where omission would have adopted\n"
        "                            a smaller number instead\n"
        "  --kv-block-size 16|32     KV page size in tokens (default: 32)\n"
        "  --prefill-chunk N         prefill chunk in tokens, 0 = unchunked\n"
        "                            (default: 2048; bounds activation memory)\n"
        "  --paged-kv KEY[:VALUE]    KV precision on the paged path (default: u8). KEY and\n"
        "                            VALUE each one of f16, u8, i8, u4, i4; omitting :VALUE\n"
        "                            applies KEY to both the key and value cache (\"u8\" and\n"
        "                            \"f16\" mean exactly what they always did). A colon pair\n"
        "                            asks for asymmetric precision (M8) and is refused unless\n"
        "                            the compiled paged model actually carries the requested\n"
        "                            widths per side.\n"
        "  --cache-grid N            prefix-cache snapshot grid in tokens (default: 0 = the\n"
        "                            prefill chunk). A snapshot lands on the last multiple of N\n"
        "                            below the prompt length; a coarser grid re-prefills the\n"
        "                            remainder on every continuation. DESIGN 7.0.2j.\n"
        "  --cache-host-mib N        host tier for evicted prefixes: an entry evicted for its\n"
        "                            KV pages parks them in host RAM instead and comes back\n"
        "                            over the link on a hit (default: 0, off). DESIGN 4.4.\n"
        "  --kv-pool-pages N         cap the KV pool at N pages (default: 0, sized by memory).\n"
        "                            A test knob: the way to make the cache evict on demand.\n"
        "\n"
        "serving defaults (the operator layer: request > these > artifact > family card;\n"
        "any of them flips the served sampler provenance to 'operator')\n"
        "  --temp X                  default temperature (0 = greedy, which is also what\n"
        "                            lets the MTP drafter engage)\n"
        "  --top-p X                 default nucleus mass\n"
        "  --top-k N                 default top-k (0 disables)\n"
        "  --repetition-penalty X    default repetition penalty\n"
        "  --presence-penalty X      default presence penalty\n"
        "  --chat-template-kwarg enable_thinking=BOOL\n"
        "                            template default when the request sends neither\n"
        "                            chat_template_kwargs.enable_thinking nor\n"
        "                            reasoning_effort; only enable_thinking exists\n"
        "\n"
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
        "  --fit-margin-mib N        headroom the paged-path auto-fit budget leaves\n"
        "                            unclaimed (default: 256). The only policy term in\n"
        "                            the reservation; every other term is measured\n"
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
        "  --dflash DIR              DFlash2 block drafter directory (7 drafts per\n"
        "                            verify pass; greedy only, like --mtp). One\n"
        "                            drafter per server: conflicts with --mtp on\n"
        "                            and --draft N\n"
        "  --dflash-device DEV       run the drafter elsewhere (default: --device)\n"
        "  --mtp-layer WHICH         auto | reconstructed | exported: which MTP layer graph to\n"
        "                            draft with when the artifact carries both (default: auto,\n"
        "                            the reconstructed one when present)\n"
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
        } else if (arg == "--cache-host-mib") {
            if (!value(v) || !parse_int(v, cfg.cache_host_mib)) return fail("--cache-host-mib needs an integer");
        } else if (arg == "--kv-pool-pages") {
            if (!value(v) || !parse_int(v, cfg.kv_pool_pages)) return fail("--kv-pool-pages needs an integer");
        } else if (arg == "--temp") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--temp needs a number");
            cfg.temp = static_cast<float>(d);
        } else if (arg == "--top-p") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--top-p needs a number");
            cfg.top_p = static_cast<float>(d);
        } else if (arg == "--top-k") {
            int k = 0;
            if (!value(v) || !parse_int(v, k)) return fail("--top-k needs an integer");
            cfg.top_k = k;
        } else if (arg == "--repetition-penalty") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--repetition-penalty needs a number");
            cfg.repetition_penalty = static_cast<float>(d);
        } else if (arg == "--presence-penalty") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--presence-penalty needs a number");
            cfg.presence_penalty = static_cast<float>(d);
        } else if (arg == "--chat-template-kwarg") {
            if (!value(v)) return fail("--chat-template-kwarg needs key=value");
            const std::string_view kv = v;
            const size_t           eq = kv.find('=');
            if (eq == std::string_view::npos) return fail("--chat-template-kwarg needs key=value");
            const std::string_view key = kv.substr(0, eq);
            const std::string_view val = kv.substr(eq + 1);
            if (key != "enable_thinking") {
                return fail("--chat-template-kwarg understands only enable_thinking; an "
                            "accepted-but-ignored key would be a lie");
            }
            if (val == "true" || val == "1") {
                cfg.think_default = true;
            } else if (val == "false" || val == "0") {
                cfg.think_default = false;
            } else {
                return fail("--chat-template-kwarg enable_thinking needs true or false");
            }
        } else if (arg == "--gate-pad") {
            if (!value(v) || !parse_int(v, cfg.gate_pad)) return fail("--gate-pad needs an integer");
        } else if (arg == "--paged-kv") {
            if (!value(v)) return fail("--paged-kv needs KEY[:VALUE]");
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
            // Distinguishes "the operator asked for this" from "nothing was
            // asked, take the model's own context" -- M7's auto-fit adopts
            // its computed max only in the second case.
            cfg.n_ctx_explicit = true;
        } else if (arg == "--prefill-chunk") {
            if (!value(v) || !parse_int(v, cfg.prefill_chunk)) {
                return fail("--prefill-chunk needs an integer");
            }
        } else if (arg == "--prefix-cache-mib") {
            if (!value(v) || !parse_int(v, cfg.prefix_cache_mib)) {
                return fail("--prefix-cache-mib needs an integer");
            }
        } else if (arg == "--fit-margin-mib") {
            if (!value(v) || !parse_int(v, cfg.fit_margin_mib)) {
                return fail("--fit-margin-mib needs an integer");
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
        } else if (arg == "--dflash") {
            if (!value(v)) return fail("--dflash needs a draft-model directory");
            cfg.dflash = std::string(v);
        } else if (arg == "--dflash-device") {
            if (!value(v)) return fail("--dflash-device needs a device");
            cfg.dflash_device = std::string(v);
        } else if (arg == "--mtp-layer") {
            if (!value(v)) return fail("--mtp-layer needs auto, reconstructed or exported");
            cfg.mtp_layer = std::string(v);
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
        } else if (arg == "--pin-dispatch") {
            if (!value(v) || !parse_int(v, cfg.pin_dispatch)) {
                return fail("--pin-dispatch needs an integer");
            }
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
    if (cfg.cache_host_mib < 0) return fail("--cache-host-mib must be >= 0");
    if (cfg.kv_pool_pages < 0) return fail("--kv-pool-pages must be >= 0");
    if (cfg.cache_grid < 0 || cfg.cache_grid > 65536) {
        return fail(log::format("--cache-grid must be 0 (the chunk) or 1..65536, not %d", cfg.cache_grid));
    }
    if (cfg.gate_pad < 0 || cfg.gate_pad == 1 || cfg.gate_pad > 4096) {
        return fail(log::format("--gate-pad must be 0 (off) or 2..4096, not %d", cfg.gate_pad));
    }
    {
        std::string pk_key, pk_value;
        if (!parse_paged_kv(cfg.paged_kv, pk_key, pk_value)) {
            return fail(log::format(
                "--paged-kv must be KEY[:VALUE] with each side one of f16, u8, i8, u4, i4, "
                "not '%s'",
                cfg.paged_kv.c_str()));
        }
    }
    if (!cfg.dflash.empty() && cfg.mtp == "on") {
        return fail("--dflash and --mtp on are two drafters for one verify loop; pick one");
    }
    if (!cfg.dflash.empty() && cfg.draft_tokens > 0) {
        return fail("--dflash and --draft are two drafters for one verify loop; pick one");
    }
    if (cfg.kv_dtype != "fp16" && cfg.kv_dtype != "fp32") {
        return fail("--kv-dtype must be fp16 or fp32");
    }
    if (cfg.offload_ratio < 0 || cfg.offload_ratio > 100) {
        return fail("--offload-ratio must be a percentage in [0, 100]");
    }
    if (cfg.pin_dispatch < -1 || cfg.pin_dispatch > 1023) {
        return fail("--pin-dispatch must be -1 (off) or a core number in [0, 1023]");
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
    if (cfg.fit_margin_mib < 0) return fail("--fit-margin-mib must be >= 0");
    if (cfg.gdn_checkpoint_budget_mib < 0) {
        return fail("--gdn-checkpoint-budget must be >= 0");
    }
    if (cfg.mtp != "on" && cfg.mtp != "off" && cfg.mtp != "auto") {
        return fail("--mtp must be on, off, or auto");
    }
    if (cfg.mtp_layer != "auto" && cfg.mtp_layer != "reconstructed" && cfg.mtp_layer != "exported") {
        return fail("--mtp-layer must be auto, reconstructed or exported");
    }
    if (cfg.mtp == "on" && entry != nullptr && !entry->has_mtp_head) {
        return fail(log::format("--mtp on: %s ships no MTP head", cfg.model_id.c_str()));
    }

    return {};
}

std::optional<std::string> apply_operator_defaults(const Config& cfg, SamplerDefaults& d) {
    SamplerOverrides o;
    o.temperature        = cfg.temp;
    o.top_p              = cfg.top_p;
    o.top_k              = cfg.top_k;
    o.repetition_penalty = cfg.repetition_penalty;
    o.presence_penalty   = cfg.presence_penalty;
    return sampler_defaults_apply(d, o);
}

}  // namespace lgc
