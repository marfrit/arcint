#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace lgc {

// Dory1.0-4B ('dory'): a looped nGPT. The file holds 72 physical blocks (16
// input, 40 recurrent, 16 output); the 40 recurrent ones run `loops` times, so
// libllama executes 16 + 40 * loops + 16 layers (112 at 2 loops, 192 at 4) over
// one set of weights. `llama_model_n_layer` reports the executed count, which
// is why admission matches the physical one (contrib/llama.cpp patch 0108).
//
// Two knobs change what is computed, and both are the operator's, never a
// silent default of this process: the loop count (`dory.num_loops`, the file
// ships 4) and the recurrent KV mode (`dory.recurrent_kv_cache_mode`:
// `per_loop` keeps one KV per loop and layer, `last_loop` one per layer, each
// loop overwriting the last). With neither flag the file's own value serves, and
// the boot line and /props name the value and where it came from.
constexpr int kDoryMaxLoops = 4;   // libllama's cap (patch 0108)

inline bool dory_kv_mode_valid(std::string_view m) { return m == "per_loop" || m == "last_loop"; }

// What the operator asked for (--llama-dory-loops, --llama-dory-kv); 0 and empty
// are "the file's own".
struct DoryOptions {
    int         loops = 0;
    std::string kv_mode;
    bool any() const { return loops > 0 || !kv_mode.empty(); }
};

// Options that a 'dory' model cannot run with, and the dory knobs on any other
// architecture. Empty when the combination is fine. Run before the load, from
// the GGUF's architecture string.
//   --llama-mtp / --llama-dflash: Dory has no MTP head and no DFlash2 drafter.
//   --llama-layer-split: the layer placement counts the physical blocks and the
//   executed ones apart (patch 0108), and a split of that has not been run.
std::string dory_option_refusal(std::string_view arch, bool mtp, bool dflash, bool layer_split,
                                const DoryOptions& opt);

// The GGUF's own metadata, read from the loaded model; 0 / empty = absent.
struct DoryGguf {
    int         n_input     = 0;
    int         n_recurrent = 0;
    int         n_output    = 0;
    int         loops       = 0;
    std::string kv_mode;
    std::string pattern;   // dory.layer_pattern: one of S (sliding attention), * (global attention), - (MLP) a physical block
};

// What serves, and where each value came from ("flag", "gguf", or "default":
// the key is absent and libllama's own default applies).
struct DoryServed {
    int         loops = 0;
    std::string kv_mode;
    std::string loops_source;
    std::string kv_source;
    int         physical_layers = 0;
    int         executed_layers = 0;
    // the layers that hold a KV cache once libllama has built the model: the attention blocks of the input
    // and output segments, and the recurrent segment's attention blocks once per loop (per_loop) or once
    // (last_loop). Derived from the pattern here and counted again from the built graph after the load.
    int         kv_layers = 0;
};

// The effective values. Returns an error text (and leaves `out` unset) for
// metadata the port does not run; empty on success.
std::string dory_resolve(const DoryGguf& g, const DoryOptions& opt, DoryServed& out);

// After the load: the layer count libllama reports must be the one the resolved
// loop count implies, which is the proof that the override (or the file's value)
// took effect rather than a fallback. Empty when it matches.
std::string dory_check_loaded(const DoryServed& s, int n_layer_reported, int n_layer_physical_reported);

// After the context: the number of KV cache layers libllama built (counted from the reserved graph's
// cache tensors) must be the number the resolved KV mode implies. The proof that --llama-dory-kv, or the
// file's value, took effect: 96 against 36 layers at 4 loops. Empty when it matches.
std::string dory_check_kv(const DoryServed& s, int kv_layers_built);

// The KV cache's size for `tokens` cells over `kv_layers` layers of `n_kv_heads` x `head_dim`, in bytes,
// for the --llama-kv types ("f16", "q8_0", "q4_0"). Sliding-window layers count at full length: the
// library default swa_full keeps them so (prefix reuse stays exact).
std::uint64_t dory_kv_bytes(int kv_layers, int n_kv_heads, int head_dim, std::int64_t tokens, std::string_view k_type,
                            std::string_view v_type);

// A start-time warning (empty when it fits): weights plus cache against the card's total memory, with
// the margin the compute buffers need. arcint cannot see the free VRAM of an OpenCL card, so this is a
// forecast against the total, not a refusal.
std::string dory_fit_warning(std::uint64_t kv_bytes, std::uint64_t weights_bytes, std::uint64_t card_bytes,
                             std::int64_t tokens, int kv_layers, std::string_view k_type, std::string_view v_type);

// The boot line: "loops 4 (gguf), recurrent KV per_loop (gguf), 72 blocks, 192 layers executed".
std::string dory_boot_line(const DoryServed& s);

}  // namespace lgc
