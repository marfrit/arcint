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

struct GgufReplacement {
    std::string ir_name;    // the IR constant's friendly name
    std::string gguf_name;  // the tensor taken from the file
    int32_t     ggml_type = 0;
    int64_t     n = 0, k = 0;
    size_t      bytes = 0;
    bool        rows_permuted = false;      // the V-head un-reorder applied to the output (a gather on the rows' axis)
    bool        columns_gathered = false;   // the un-reorder applied to the activation (a gather on the columns' axis)
};

struct GgufApplyReport {
    std::vector<GgufReplacement> replaced;
    std::vector<std::string>     kept;      // IR constants deliberately left as the template's (by role)
    size_t bytes_from_file = 0;             // the K-quant bytes now in the graph
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
GgufApplyReport gguf_apply_to_template(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<gguf::GgufFile>& file,
                                       const GgufGeometry& geometry);

}  // namespace lgc
