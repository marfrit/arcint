#include "core/dory.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

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

std::uint64_t dory_compute_bytes(std::int64_t mask_cells, int n_ubatch) {
    if (mask_cells <= 0 || n_ubatch <= 0) return 0;
    return static_cast<std::uint64_t>(n_ubatch) *
           (kDoryComputeRowBytes + kDoryMaskBytesPerCellRow * static_cast<std::uint64_t>(mask_cells));
}

std::uint64_t dory_card_weight_bytes(std::string_view load_log) {
    static constexpr std::string_view key = " model buffer size =";
    std::uint64_t sum = 0;
    size_t at = 0;
    while (at < load_log.size()) {
        size_t eol = load_log.find('\n', at);
        if (eol == std::string_view::npos) eol = load_log.size();
        const std::string_view line = load_log.substr(at, eol - at);
        at = eol + 1;
        const size_t k = line.find(key);
        if (k == std::string_view::npos) continue;
        // the buffer's name is the word before the key
        size_t ne = k;
        while (ne > 0 && line[ne - 1] == ' ') --ne;
        size_t ns = ne;
        while (ns > 0 && line[ns - 1] != ' ' && line[ns - 1] != ':') --ns;
        const std::string_view name = line.substr(ns, ne - ns);
        if (name.empty() || name.substr(0, 3) == "CPU") continue;
        const std::string rest(line.substr(k + key.size()));
        char* end = nullptr;
        const double mib = std::strtod(rest.c_str(), &end);
        if (end == rest.c_str() || !std::isfinite(mib) || mib <= 0.0) continue;
        while (*end == ' ') ++end;
        if (std::string_view(end).substr(0, 3) != "MiB") continue;
        sum += static_cast<std::uint64_t>(mib * 1024.0 * 1024.0 + 0.5);
    }
    return sum;
}

DoryCardForecast dory_card_forecast(std::uint64_t weights_on_card, std::uint64_t kv_bytes, std::int64_t mask_cells,
                                    int n_ubatch) {
    DoryCardForecast f;
    f.weights = weights_on_card;
    f.kv      = kv_bytes;
    f.compute = dory_compute_bytes(mask_cells, n_ubatch);
    f.backend = kDoryBackendPeakBytes;
    return f;
}

DoryStartForecast dory_start_forecast(std::uint64_t weights_on_card, int kv_layers, std::string_view k_type,
                                      std::string_view v_type, std::int64_t pool, int lanes, bool unified,
                                      std::int64_t reserve_cells, int n_ubatch) {
    DoryStartForecast s;
    s.cells      = dory_pool_cells(pool, lanes, unified);
    s.mask_cells = dory_mask_cells(pool, lanes, unified, reserve_cells);
    s.card = dory_card_forecast(weights_on_card, dory_kv_bytes(kv_layers, kDoryKvHeads, kDoryHeadDim, s.cells, k_type, v_type),
                                s.mask_cells, n_ubatch);
    return s;
}

namespace {
double mib(std::uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); }
}  // namespace

std::string dory_forecast_line(const DoryCardForecast& f, std::uint64_t card_bytes, int n_ubatch, bool weights_from_load) {
    char card[96];
    if (card_bytes > 0) std::snprintf(card, sizeof(card), " of the card's %.0f MiB (OpenCL's figure)", mib(card_bytes));
    else card[0] = '\0';
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "dory: card memory forecast %.0f MiB at the peak%s: weights %.0f (%s), KV %.0f, compute %.0f at micro-batch %d, "
                  "backend %.0f (measured on an A770)",
                  mib(f.total()), card, mib(f.weights),
                  weights_from_load ? "libllama's card buffers" : "the whole file: no card buffer line was read", mib(f.kv),
                  mib(f.compute), n_ubatch, mib(f.backend));
    return buf;
}

std::string dory_fit_warning(const DoryCardForecast& f, std::uint64_t card_bytes, std::int64_t tokens, int kv_layers,
                             std::string_view k_type, std::string_view v_type) {
    if (card_bytes == 0) return {};
    if (f.total() <= card_bytes) return {};
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "dory: %.0f MiB are forecast past the card's %.0f MiB (OpenCL's figure): the KV cache %.0f MiB (%d layers x "
                  "%lld cells, %.*s:%.*s), the weights %.0f, the compute buffer %.0f, the backend %.0f; lower --n-ctx, use "
                  "--llama-kv q8_0:q4_0, fewer --llama-dory-loops, --llama-dory-kv last_loop, or a smaller --prefill-ubatch",
                  mib(f.total()), mib(card_bytes), mib(f.kv), kv_layers, static_cast<long long>(tokens),
                  static_cast<int>(k_type.size()), k_type.data(), static_cast<int>(v_type.size()), v_type.data(), mib(f.weights),
                  mib(f.compute), mib(f.backend));
    return buf;
}

std::string dory_boot_line(const DoryServed& s) {
    return "dory: loops " + std::to_string(s.loops) + " (" + s.loops_source + "), recurrent KV " + s.kv_mode + " (" + s.kv_source +
           "), " + std::to_string(s.physical_layers) + " blocks, " + std::to_string(s.executed_layers) + " layers executed, " +
           std::to_string(s.kv_layers) + " KV cache layers";
}

}  // namespace lgc
