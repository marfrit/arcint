#pragma once

// Opening a GGUF in process on a template IR (docs/design-gguf-native.md
// §3.1): the served IR of the same architecture is the topology, read with
// memory-mapped weights that are never touched; this pass walks it and
// replaces every projection's decompression subgraph with the file's own
// K-quant rows as a tagged u8 constant feeding a FullyConnectedKQuant op
// (kquant_op.h), which the plugin at +p7 decodes inside its kernel.
// Nothing is unpacked here: the only bytes that reach the card are the
// file's, row-permuted where the converter had permuted them.
//
// Stage 1 scope (dense `qwen35`): every projection of every block and the
// head are taken from the file; the embedding gather, the norms, the GDN's
// own small tensors (A_log, dt bias, conv1d) and the MTP layer stay the
// template's. What was replaced and what was kept is the report, printed at
// load and returned to the caller, so a served model never silently mixes.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <openvino/core/model.hpp>

#include "core/gguf.h"
#include "core/gguf_map.h"

namespace lgc {

// How a projection's K-quant rows reach the graph (0.4.1 lever 2,
// docs/design-gguf-native.md §3.6):
//   Repack -- the default: the rows are repacked at load into the plugin's
//             grouped compressed-weight form (u4/u8/i8 with an f16 scale and,
//             for the K types with a min, an f16 zero point per group), the
//             form the served IRs use and the runtime's fastest kernels take;
//             a bounded deviation per weight (core/gguf_repack.h), measured
//             at load and reported.
//   Native -- the file's own bytes as a tagged u8 constant feeding
//             FullyConnectedKQuant, decoded in the plugin's kernel (0.4.0);
//             exact on the decode path, slower.
//   Mixed  -- Q4_K repacked, every other type native (0.4.1): the repack's
//             u8 form of Q5_K/Q6_K cost 2.7 GB of residency over the file's
//             own rows on the dense model (DESIGN 7.0.2ba), which is what
//             kept its 71.7k cell out of u8 KV.
enum class GgufWeightsMode { Repack, Native, Mixed };
// The mode a tensor of `ggml_type` takes under `mode`.
GgufWeightsMode gguf_tensor_mode(GgufWeightsMode mode, int32_t ggml_type);

struct GgufReplacement {
    std::string ir_name;    // the IR constant's friendly name
    std::string gguf_name;  // the tensor taken from the file
    int32_t     ggml_type = 0;
    int64_t     n = 0, k = 0;
    size_t      bytes = 0;
    bool        rows_permuted = false;      // the V-head un-reorder applied to the output (a gather on the rows' axis)
    bool        columns_gathered = false;   // the un-reorder applied to the activation (a gather on the columns' axis)
    bool        repacked = false;           // the runtime's compressed form (else the file's rows in the K-quant kernel)
};

struct GgufApplyReport {
    int q6k_aligned = 0;         // native Q6_K projections laid out in 224-byte blocks (type 114)
    std::vector<GgufReplacement> replaced;
    std::vector<std::string>     kept;      // IR constants deliberately left as the template's (by role)
    size_t bytes_from_file = 0;             // the K-quant bytes now in the graph (native) or the repacked bytes
    GgufWeightsMode mode = GgufWeightsMode::Repack;
    // Repack mode: the deviation of the repacked projections from ggml's
    // dequantized values, in units of the group's quantisation step -- the
    // largest over every weight and the count over the per-type bound.
    double repack_max_steps = 0.0;
    size_t repack_over_bound = 0;
    size_t repack_checked = 0;
    size_t repack_verdicts_cached = 0;      // projections whose deviation verdict came from an earlier load (gguf_check once)
    double repack_seconds = 0.0;            // wall time of the repacks
    double check_seconds = 0.0;             // wall time of the deviation checks
    // The exporter's AWQ folded per-channel scales into the graph as
    // activation-side multipliers (the attention gate's `awq_mul/scale`)
    // compensating weights it had divided; with the projections now the
    // file's raw rows those multipliers are set to one. Count of constants
    // so neutralized.
    size_t awq_scales_neutralized = 0;
    // Norm weights stay the template's; each one the file also carries is
    // compared and the largest absolute difference kept here, with the
    // count compared, so a template whose norms are not the file's is seen.
    size_t norms_compared = 0;
    double norms_max_abs_diff = 0.0;
    std::string summary() const;            // one line: counts per type and MiB
};

// Replaces the template model's projections with the file's tensors. The
// K-quant constants alias the file's map and hold `file` alive with them; the
// caller keeps its own reference for the compiled model's lifetime too.
// Throws, naming the tensor, when a projection the template has is missing
// from the file, has an unexpected shape or a type this stage does not serve.
// `verdict_dir`, when given with `file_path`, keeps the deviation verdict of
// every repacked projection (keyed by the file's size and mtime, the tensor's
// offset, type and dims, and the bound) so a later load of the same file
// skips the check it already passed -- the exhaustive check is most of a
// 406 s load on the dense model (DESIGN 7.0.2bd); a verdict is only ever
// written for a projection that passed.
// q6k_aligned: the native Q6_K rows laid out in 224-byte blocks (kquant type 114) instead of
// the file's 210-byte rows (+6.7 % on that set; DESIGN 7.0.2bj).
GgufApplyReport gguf_apply_to_template(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<gguf::GgufFile>& file,
                                       const GgufGeometry& geometry,
                                       GgufWeightsMode mode = GgufWeightsMode::Repack,
                                       const std::string& verdict_dir = "",
                                       const std::string& file_path = "",
                                       bool q6k_aligned = true);

}  // namespace lgc
