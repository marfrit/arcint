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

// The card memory a Dory start is forecast to hold at its peak, by term. Every term but the cache is
// measured-here (an A770, the Q5_K_M file, 2 loops, last_loop, q8_0:q4_0, 2026-11-12). The constants come from
// a series at micro-batch 256 and 512 over contexts of 8,192 to 131,072 cells: against the card's resident
// memory at the four peaks of that series at micro-batch 512 the sum is 24.9 / 16.8 / 0.7 / 0.6 MiB high (8,192
// / 32,768 / 65,536 / 131,072 cells) and never below one; against its one peak at micro-batch 256 (8,192 cells)
// it is 32.4 MiB high. A later run at 196,608 cells (micro-batch 512, prompts to 176,888 tokens) peaked 0.2 MiB
// under the sum. The margin over the three deepest peaks is under 1 MiB, and no prompt deeper than 176,888
// tokens has been run.
struct DoryCardForecast {
    std::uint64_t weights = 0;   // the weight buffers libllama placed on the card (the input embedding stays in host RAM)
    std::uint64_t kv      = 0;   // dory_kv_bytes; libllama's two KV lines read 0.04 MiB more at 131,072 cells (7,488.04)
    std::uint64_t compute = 0;   // dory_compute_bytes: the card's compute buffer
    std::uint64_t backend = 0;   // kDoryBackendPeakBytes: card memory outside libllama's buffers
    std::uint64_t total() const { return weights + kv + compute + backend; }
};

// The card's compute buffer, per micro-batch row (measured-here: libllama's "OpenCL compute buffer size"
// lines, 39.0109 / 78.0216 MiB at 8,192 cells and 159.0109 / 318.0216 MiB at 131,072 cells for micro-batch
// 256 / 512). The part that grows with the context is the attention mask: [cells, micro-batch] in f16 under
// flash attention, once for the global layers and once for the sliding ones (code: libllama's
// build_attn_inp_kq_mask and build_attn_inp_kv_iswa), 4 bytes a cell and row. The old forecast had no such
// term and read 2,048 bytes a cell low at micro-batch 512.
// `mask_cells` is the cell count of one sequence's mask (code: libllama's n_ctx_seq): the whole pool with one
// lane or a unified cache, one lane's share with --parallel N. Only one lane has been measured. With
// --llama-reserve-cells R libllama reserves the buffer for R cells and keeps it there: a batch that ends deeper
// in the cache takes smaller micro-batches instead (code: llama_context::decode's n_ubatch_eff, contrib
// llama.cpp 0052), so the mask's area stays at R cells x the micro-batch (code, not measured for Dory). The one
// exception in that code is its floor of 2 rows a micro-batch with one lane: a batch deeper than 256 x R cells
// (micro-batch 512) takes a mask wider than the reserve's; the forecast does not model that case.
constexpr std::uint64_t kDoryComputeRowBytes     = 127020;
constexpr std::uint64_t kDoryMaskBytesPerCellRow = 2 * 2;
std::uint64_t dory_compute_bytes(std::int64_t mask_cells, int n_ubatch);

// The cells libllama allocates for a pool of `pool` cells over `lanes` sequences (code: llama_context's
// constructor): the pool is padded to 256, and without a unified cache each lane's share is padded to 256 again
// and the pool becomes lanes x that. --n-ctx 197,889 allocates 198,144 cells, so the forecast takes these
// figures, not the flag's.
inline std::int64_t dory_pad256(std::int64_t n) { return n <= 0 ? 0 : (n + 255) / 256 * 256; }
inline std::int64_t dory_seq_cells(std::int64_t pool, int lanes, bool unified) {
    const std::int64_t p = dory_pad256(pool);
    return unified || lanes <= 1 ? p : dory_pad256(p / lanes);
}
inline std::int64_t dory_pool_cells(std::int64_t pool, int lanes, bool unified) {
    const std::int64_t s = dory_seq_cells(pool, lanes, unified);
    return unified || lanes <= 1 ? s : s * lanes;
}
inline std::int64_t dory_mask_cells(std::int64_t pool, int lanes, bool unified, std::int64_t reserve_cells = 0) {
    const std::int64_t s = dory_seq_cells(pool, lanes, unified);
    const std::int64_t r = dory_pad256(reserve_cells);
    return r > 0 && r < s ? r : s;
}

// Card memory the driver counts for the process outside libllama's weight, KV and compute buffers
// (measured-here, the A770, fdinfo's resident VRAM against libllama's lines): 134 MiB at load, 678 MiB more
// with the first request, and 14 to 46 MiB more by the deepest fill (14 at micro-batch 256). At the peaks it
// was 833.1 / 841.2 / 857.3 / 857.4 MiB at 8,192 / 32,768 / 65,536 / 131,072 cells (micro-batch 512), 825.6 MiB
// at 8,192 cells with micro-batch 256, and in a later run 857.8 MiB at 196,608 cells (857.5 resident after the
// deepest prompt); the constant is the largest of the six, rounded up, which leaves 0.2 MiB at the deepest.
// Prompt depth and context size moved together in these runs, so which of the two the last 14 to 46 MiB follow
// is not separated. What allocates this memory has not been measured, so it is carried as a constant of that
// card and build, not derived.
constexpr std::uint64_t kDoryBackendPeakBytes = 858ull << 20;

// The bytes of the weight buffers libllama placed on a card, summed from the load's own lines
// ("load_tensors:       OpenCL model buffer size =  2853.58 MiB"); buffers named CPU... are host memory and
// do not count. 0 when the text holds no card buffer line. Every other buffer name counts as the card's: an
// expert cache's host-side bank carries the backend's name too, which no Dory file reaches (it has no experts).
std::uint64_t dory_card_weight_bytes(std::string_view load_log);

DoryCardForecast dory_card_forecast(std::uint64_t weights_on_card, std::uint64_t kv_bytes, std::int64_t mask_cells, int n_ubatch);

// The forecast of one start, composed from what the start is given: the pool it asks libllama for (--n-ctx x
// the lanes, or the sum of --lane-ctx), the lanes, whether the cache is unified, --llama-reserve-cells and the
// micro-batch. `cells` is the pool libllama allocates (the count the cache term and the warning take),
// `mask_cells` the cells one sequence's mask spans (the count the compute term takes). The start chooses the
// weight figure (libllama's card buffer lines, or the whole file when none was read) and supplies the card's
// OpenCL figure; every cell count and every other term comes from this call, so the tests run the composition
// the served line comes from. The cache geometry is the published Dory1.0-4B file's: 4 KV heads of 256
// (DESIGN.md).
constexpr int kDoryKvHeads = 4;
constexpr int kDoryHeadDim = 256;
struct DoryStartForecast {
    std::int64_t     cells      = 0;
    std::int64_t     mask_cells = 0;
    DoryCardForecast card;
};
DoryStartForecast dory_start_forecast(std::uint64_t weights_on_card, int kv_layers, std::string_view k_type,
                                      std::string_view v_type, std::int64_t pool, int lanes, bool unified,
                                      std::int64_t reserve_cells, int n_ubatch);

// The start line, logged for every Dory start: the forecast and its terms in MiB against the card's
// OpenCL global memory. `weights_from_load` false names the fallback (the whole file's tensor bytes, which
// overstates the card's share by the input embedding).
std::string dory_forecast_line(const DoryCardForecast& f, std::uint64_t card_bytes, int n_ubatch, bool weights_from_load);

// A start-time warning (empty when it fits): the forecast against the card's OpenCL global memory, which
// Intel's runtime reports below the card's memory (measured-here, clinfo on the card host: 16,225,243,136 bytes
// of CL_DEVICE_GLOBAL_MEM_SIZE, 15,473.6 MiB, for the A770's 16,288 MiB; it is the query the start itself makes
// for the card figure; ggml prints it truncated as "15473 MB", the forecast line rounded as 15474). arcint
// cannot see the free VRAM of an OpenCL card, so this is a forecast against that total, not a refusal: a
// start past the figure is not stopped.
// The rule has no margin of its own: it fires when the sum passes the figure. The rule before it (the file's
// bytes plus the cache against 92 % of the figure) fired from about 187,996 cells in the measured
// configuration, where this forecast stands 593 MiB under the figure at micro-batch 512 (807 at 256); this one
// fires from 198,030 cells at micro-batch 512 (201,892 at 256) by the arithmetic, which is a --n-ctx above
// 197,888 (201,728), the next allocation being 198,144 cells (201,984).
std::string dory_fit_warning(const DoryCardForecast& f, std::uint64_t card_bytes, std::int64_t tokens, int kv_layers,
                             std::string_view k_type, std::string_view v_type);

// The boot line: "loops 4 (gguf), recurrent KV per_loop (gguf), 72 blocks, 192 layers executed".
std::string dory_boot_line(const DoryServed& s);

}  // namespace lgc
