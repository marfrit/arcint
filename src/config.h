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

    // OpenVINO device string. GPU.0 is the B60 and GPU.1 the A770 on the dev host;
    // CPU is useful for correctness work when a card is busy.
    std::string device = "GPU.0";

    // Compiled-blob cache. Empty disables it; a cold MoE compile is ~2 minutes.
    std::string cache_dir;

    std::string host = "127.0.0.1";
    int         port = 8090;

    int n_ctx    = 0;  // 0: take the model's trained context
    int parallel = 1;  // slots (DESIGN.md §4 /health reports free/total)

    // Prefill chunk in tokens, 0 = one call for the whole prompt.
    //
    // Bounding the chunk is what bounds activation memory, and that — not a
    // bigger host — is how deep context is actually served. The reference on
    // this fleet (llama.cpp, 262144 context, 35B on a 16 GB A770) does exactly
    // this, at n_ubatch 512.
    //
    // 2048 is chosen so that ordinary prompts still land in a single chunk and
    // are therefore bit-identical to an unchunked run, while a long prompt is
    // split rather than allowed to scale activations without limit. Chunk
    // boundaries are not bit-exact on this backend (DESIGN.md §3.2), so this is
    // a real trade and the number is where it bites least.
    int prefill_chunk = 2048;

    // Prefix-cache budget in MiB, host-side. Zero disables it. A snapshot of
    // this architecture's state is tens of MiB (the GDN half is fixed-size), so
    // this is a real budget, not a formality (§3.3).
    int prefix_cache_mib = 0;

    // Slice the hidden state to its last row before the LM head, so prefill
    // computes one logit row rather than one per prompt token. On by default:
    // it is what makes deep prompts fit at all, it is faster, and it does not
    // change the output. The switch exists so the equivalence suite can prove
    // that last claim rather than assert it.
    bool slice_logits = true;

    // Path to an OpenVINO custom-layer XML. Empty = use the plugin's own
    // kernels only. Opt-in because it replaces graph nodes with hand-written
    // OpenCL and the win has to be measured per card, not assumed.
    std::string custom_kernels;

    // Percentage of MoE expert weights the GPU plugin may keep out of VRAM and
    // stream on demand. 0 = everything resident. This is what lets a model that
    // does not fit a card run on it at all; it is not free (§7).
    int offload_ratio = 0;

    // The paged execution path (DESIGN §3.5.3, §7.0): arcint-owned block
    // tables and LA state rows, speculative rollback as row promotion,
    // reservation-based admission. Default on; --no-paged selects the stateful
    // reference implementation the equivalence suite compares against.
    bool paged = true;

    // Where the embeddings gather and the MTP head run. Empty = same card as
    // --device. On a tight card the measured configuration parks both on the
    // other card: ~20 KB of activations cross per step, weights stay put.
    std::string emb_device;
    std::string mtp_device;

    int         kv_block_size             = 32;      // 16 or 32 — §8 benchmarks this
    std::string kv_dtype                  = "fp16";  // fp16 | q8
    int         gdn_checkpoint_budget_mib = 512;     // §3.3
    std::string mtp                       = "auto";  // on | off | auto

    // Speculative decoding through the external-drafter hook (DESIGN.md §3.5).
    // 0 disables it. No export currently carries an MTP head, so the drafter is
    // weightless n-gram lookup; swapping in a head later changes only the
    // source of the guesses.
    int draft_tokens = 0;
    int draft_ngram  = 3;

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
