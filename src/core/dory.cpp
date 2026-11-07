#include "core/dory.h"

#include <string>
#include <cstdio>

namespace lgc {

std::string dory_option_refusal(std::string_view arch, bool mtp, bool dflash, bool layer_split,
                                const DoryOptions& opt) {
    if (arch != "dory") {
        if (opt.loops > 0)
            return "--llama-dory-loops is the loop count of a 'dory' model; this GGUF is a '" + std::string(arch) + "'";
        if (!opt.kv_mode.empty())
            return "--llama-dory-kv is the recurrent KV mode of a 'dory' model; this GGUF is a '" + std::string(arch) + "'";
        return {};
    }
    if (mtp)
        return "--llama-mtp: a 'dory' model has no MTP head; start without --llama-mtp (and its -vocab/-gguf/-policy flags)";
    if (dflash)
        return "--llama-dflash-gguf: there is no DFlash2 drafter for a 'dory' model; start without it";
    if (layer_split)
        return "--llama-layer-split: a 'dory' model counts physical blocks and executed layers apart, and a split of "
               "that has not been run; serve it on one card";
    return {};
}

std::string dory_resolve(const DoryGguf& g, const DoryOptions& opt, DoryServed& out) {
    if (g.n_input <= 0 || g.n_recurrent <= 0 || g.n_output <= 0)
        return "dory: the GGUF's input/recurrent/output layer counts are missing or zero";
    DoryServed s;
    if (opt.loops > 0) {
        s.loops        = opt.loops;
        s.loops_source = "flag";
    } else if (g.loops > 0) {
        s.loops        = g.loops;
        s.loops_source = "gguf";
    } else {
        return "dory: the GGUF carries no dory.num_loops; pass --llama-dory-loops N (1 to 4)";
    }
    if (s.loops > kDoryMaxLoops)
        return "dory: " + std::to_string(s.loops) + " loops; the port runs 1 to " + std::to_string(kDoryMaxLoops);
    if (!opt.kv_mode.empty()) {
        if (!dory_kv_mode_valid(opt.kv_mode)) return "dory: --llama-dory-kv takes per_loop or last_loop";
        s.kv_mode   = opt.kv_mode;
        s.kv_source = "flag";
    } else if (!g.kv_mode.empty()) {
        if (!dory_kv_mode_valid(g.kv_mode))
            return "dory: the GGUF's dory.recurrent_kv_cache_mode is '" + g.kv_mode + "', not per_loop or last_loop";
        s.kv_mode   = g.kv_mode;
        s.kv_source = "gguf";
    } else {
        s.kv_mode   = "per_loop";   // libllama's default for an absent key (patch 0108)
        s.kv_source = "default";
    }
    if (static_cast<int>(g.pattern.size()) != g.n_input + g.n_recurrent + g.n_output ||
        g.pattern.find_first_not_of("S*-") != std::string::npos)
        return "dory: dory.layer_pattern is missing or does not give one of S, * or - for each of the " +
               std::to_string(g.n_input + g.n_recurrent + g.n_output) + " blocks";
    auto attn = [&](int from, int n) {
        int a = 0;
        for (int i = from; i < from + n; ++i) a += g.pattern[static_cast<size_t>(i)] != '-';
        return a;
    };
    s.kv_layers = attn(0, g.n_input) + attn(g.n_input, g.n_recurrent) * (s.kv_mode == "per_loop" ? s.loops : 1) +
                  attn(g.n_input + g.n_recurrent, g.n_output);
    s.physical_layers = g.n_input + g.n_recurrent + g.n_output;
    s.executed_layers = g.n_input + g.n_recurrent * s.loops + g.n_output;
    out               = s;
    return {};
}

std::string dory_check_loaded(const DoryServed& s, int n_layer_reported, int n_layer_physical_reported) {
    if (n_layer_physical_reported != s.physical_layers)
        return "dory: libllama reports " + std::to_string(n_layer_physical_reported) + " physical blocks, the file's counts say " +
               std::to_string(s.physical_layers);
    if (n_layer_reported != s.executed_layers)
        return "dory: libllama executes " + std::to_string(n_layer_reported) + " layers, " + std::to_string(s.loops) +
               " loops imply " + std::to_string(s.executed_layers) + "; the loop count did not take effect";
    return {};
}

std::string dory_check_kv(const DoryServed& s, int kv_layers_built) {
    if (kv_layers_built != s.kv_layers)
        return "dory: libllama built " + std::to_string(kv_layers_built) + " KV cache layers, recurrent KV " + s.kv_mode + " at " +
               std::to_string(s.loops) + " loops implies " + std::to_string(s.kv_layers) + "; the KV mode did not take effect";
    return {};
}

namespace {
double kv_elem_bytes(std::string_view t) {
    if (t == "q8_0") return 34.0 / 32.0;
    if (t == "q4_0") return 18.0 / 32.0;
    return 2.0;   // f16
}
}  // namespace

std::uint64_t dory_kv_bytes(int kv_layers, int n_kv_heads, int head_dim, std::int64_t tokens, std::string_view k_type,
                            std::string_view v_type) {
    const double per_token_layer = static_cast<double>(n_kv_heads) * head_dim * (kv_elem_bytes(k_type) + kv_elem_bytes(v_type));
    return static_cast<std::uint64_t>(per_token_layer * kv_layers * static_cast<double>(tokens));
}

std::string dory_fit_warning(std::uint64_t kv_bytes, std::uint64_t weights_bytes, std::uint64_t card_bytes,
                             std::int64_t tokens, int kv_layers, std::string_view k_type, std::string_view v_type) {
    if (card_bytes == 0) return {};
    const double gib = 1024.0 * 1024.0 * 1024.0;
    if (static_cast<double>(kv_bytes + weights_bytes) <= 0.92 * static_cast<double>(card_bytes)) return {};
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "dory: the KV cache (%.1f GiB: %d layers x %lld cells, %.*s:%.*s) and the weights (%.1f GiB) are forecast "
                  "past the card's %.1f GiB with room for the compute buffers; lower --n-ctx, use --llama-kv q8_0:q4_0, "
                  "fewer --llama-dory-loops, or --llama-dory-kv last_loop",
                  static_cast<double>(kv_bytes) / gib, kv_layers, static_cast<long long>(tokens),
                  static_cast<int>(k_type.size()), k_type.data(), static_cast<int>(v_type.size()), v_type.data(),
                  static_cast<double>(weights_bytes) / gib, static_cast<double>(card_bytes) / gib);
    return buf;
}

std::string dory_boot_line(const DoryServed& s) {
    return "dory: loops " + std::to_string(s.loops) + " (" + s.loops_source + "), recurrent KV " + s.kv_mode + " (" + s.kv_source +
           "), " + std::to_string(s.physical_layers) + " blocks, " + std::to_string(s.executed_layers) + " layers executed, " +
           std::to_string(s.kv_layers) + " KV cache layers";
}

}  // namespace lgc
