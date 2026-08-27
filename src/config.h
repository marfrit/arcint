#pragma once

#include <string>

#include "core/model_registry.h"

namespace lgc {

struct Config {
    // Exactly one of these selects what gets served.
    std::string model_path;  // OpenVINO IR directory (M1+)
    bool        stub = false;

    std::string model_id;  // allowlist entry; defaults to the coder under --stub
    Quant       quant = Quant::Q4;

    // OpenVINO device string. GPU.0 is the B60 and GPU.1 the A770 on dirac;
    // CPU is useful for correctness work when a card is busy.
    std::string device = "GPU.0";

    // Compiled-blob cache. Empty disables it; a cold MoE compile is ~2 minutes.
    std::string cache_dir;

    std::string host = "127.0.0.1";
    int         port = 8090;

    int n_ctx    = 0;  // 0: take the model's trained context
    int parallel = 1;  // slots (DESIGN.md §4 /health reports free/total)

    int         kv_block_size             = 32;      // 16 or 32 — §8 benchmarks this
    std::string kv_dtype                  = "fp16";  // fp16 | q8
    int         gdn_checkpoint_budget_mib = 512;     // §3.3
    std::string mtp                       = "auto";  // on | off | auto

    // Stub-only: milliseconds of artificial latency per emitted token. Exists
    // so the cancellation path (§3.7) can be demonstrated against a backend
    // that would otherwise finish before a client could disconnect.
    int stub_delay_ms = 0;

    int http_threads = 0;  // 0: httplib default
    int verbosity    = 0;  // -v, -vv

    bool show_help    = false;
    bool show_version = false;
};

struct ArgParse {
    bool        ok = true;
    std::string error;
};

ArgParse    parse_args(int argc, char** argv, Config& cfg);
std::string usage_text();

}  // namespace lgc
