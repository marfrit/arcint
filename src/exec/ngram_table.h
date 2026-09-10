#pragma once

// FIX D Link 3 (docs/design-qwen-flash-next.md "Links 2 and 3"): the
// decode-time consumer of the per_layer_token_embd table. This is the wiring
// that ties the three FIX D pieces together behind the --flash-next-ngram flag:
//
//   Link 1 (tools/synthetic_ngram_table.py)  the on-disk ARCINGRM table
//   Link 2 (admit_ngram_table_from_disk)      load-time admission + fit refusal
//   Link 3 (this file)                         row_ids -> gather_dequant lookup
//
// `NGramLookup` is arcint's implementation of the reference's frozen
// `PLETableBackend.lookup` contract (docs/research-freetoken.md "Code-side
// ground truth"): row ids in, dequantized rows out. It owns the host-resident
// (mmapped) table payload, holds the per-PLE-layer hash constants, and for a
// batch of tokens computes their hashed row ids (exec/ngram_row_ids.h) and
// gathers+dequantizes those rows (exec/ngram_gather.h). One `NGramLookup::lookup`
// is one PLE layer's table read for one request -- exactly PinnedUVATable.lookup.
//
// What this does NOT do, and why: the reference's PLELayer wraps this lookup in
// gated key/value projections, RMS norms, and a dilated depthwise conv, then
// adds the result to the residual stream (docs/research-freetoken.md). Those
// need the TRAINED backbone weights (key_proj/value_proj/norms/conv1d), which
// only a Flash-Next artifact carries -- the checkpoint fork in
// HANDOFF-0.5.0.local.md. Until that lands, the hash constants come from the
// reference's dummy-weight derivation (derive_hash_constants); a check of the
// real checkpoint's int64 buffers is still owed. The lookup itself -- the piece
// that is "wiring, not design" -- is complete and testable windowless.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/artifact.h"
#include "exec/ngram_row_ids.h"

namespace lgc::ngram {

class NGramLookup {
public:
    NGramLookup() = default;
    ~NGramLookup();
    NGramLookup(NGramLookup&&) noexcept;
    NGramLookup& operator=(NGramLookup&&) noexcept;
    NGramLookup(const NGramLookup&)            = delete;
    NGramLookup& operator=(const NGramLookup&) = delete;

    // The PLETableBackend.lookup contract for one PLE layer + one request.
    // `layer_ordinal` indexes the PLE layers in ple_layer_ids order.
    // `context` is the ngram_size-1 tokens before `tokens[0]` (eos for a fresh
    // sequence). Returns [tokens.size() * num_ngram_heads * n_cols] floats,
    // dequantized -- num_ngram_heads rows of n_cols each per token. Throws a
    // named error if a hashed row id falls outside the table (a table too small
    // for the hashed space, which admission's lower bound is meant to prevent).
    std::vector<float> lookup(size_t layer_ordinal, const std::vector<int64_t>& context,
                              const std::vector<int64_t>& tokens) const;

    size_t             num_layers() const { return layers_.size(); }
    const HashParams&  params(size_t layer_ordinal) const { return layers_.at(layer_ordinal); }
    int32_t            ggml_type() const { return ggml_type_; }
    size_t             n_cols() const { return n_cols_; }
    uint32_t           n_rows() const { return n_rows_; }

    // Build a lookup over an already-in-memory payload (no mmap; the buffer is
    // copied and owned). For tests that construct a table in memory and for any
    // caller that has already read the payload. `layers` must be non-empty and
    // each validate().
    static NGramLookup adopt_owned(std::vector<uint8_t> payload, int32_t ggml_type,
                                   size_t n_cols, uint32_t n_rows,
                                   std::vector<HashParams> layers);

    // Build a lookup by mmapping `path`'s ARCINGRM payload read-only. The header
    // is parsed and its (ggml_type, n_cols, n_rows) recorded; the mapping is
    // owned for the lookup's lifetime. Throws on open/mmap/header failure.
    static NGramLookup mmap_table(const std::string& path, std::vector<HashParams> layers);

private:
    void finish_construction();

    // Exactly one of `map_` / `owned_` backs `payload_`.
    void*                map_      = nullptr;   // mmap base (munmap on destroy), or nullptr
    size_t               map_size_ = 0;
    std::vector<uint8_t> owned_;                // owned copy when not mmapped
    const uint8_t*       payload_  = nullptr;   // first table byte (past the 24-byte header)
    int32_t              ggml_type_ = 0;
    size_t               n_cols_    = 0;
    uint32_t             n_rows_    = 0;
    size_t               row_stride_bytes_ = 0;
    std::vector<HashParams> layers_;            // one per PLE layer (ple_layer_ids order)
};

// The load seam the OpenVINO backend calls behind --flash-next-ngram. Given the
// admitted artifact and the flag's path, it:
//   1. returns nullopt with err="" when `path` is empty (flag not set: the
//      cold path -- serving without the table);
//   2. runs Link 2 admission (admit_ngram_table_from_disk: header, type, row
//      width, hashed-row-space bound, on-disk size, host-RAM fit) and returns
//      nullopt with err=<named refusal> when it refuses;
//   3. on admission, derives the per-PLE-layer dummy-weight hash constants,
//      mmaps the table, and returns a ready NGramLookup.
// `out_payload_bytes` receives the admitted payload size (0 on cold path or
// refusal) so the caller can account the host-resident reservation.
std::optional<NGramLookup> load_ngram_lookup(const Artifact& artifact, const std::string& path,
                                             uint64_t host_ram_bytes, uint64_t other_resident_bytes,
                                             uint64_t margin_bytes, std::string& err,
                                             uint64_t& out_payload_bytes);

}  // namespace lgc::ngram
