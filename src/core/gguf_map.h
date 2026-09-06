#pragma once

// The architecture-level facts a GGUF open needs, kept free of OpenVINO so
// they are unit-tested device-free (docs/design-gguf-native.md §3.1): the
// file's geometry against the artifact's, the converter's V-head reorder
// inverted, and the served IR's module names mapped to llama.cpp's tensor
// names for `qwen35`.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/gguf.h"

namespace lgc {

struct GgufGeometry {
    int64_t n_layers        = 0;  // the file's block_count minus its MTP ("nextn") blocks
    int64_t hidden          = 0;
    int64_t n_heads         = 0;
    int64_t n_kv_heads      = 0;
    int64_t head_dim        = 0;
    int64_t linear_k_heads  = 0;  // GDN key heads
    int64_t linear_v_heads  = 0;  // GDN value heads
    int64_t linear_k_dim    = 0;  // GDN key head dim
    int64_t linear_v_dim    = 0;  // GDN value head dim (inner_size / value heads)
    int64_t full_attention_interval = 0;
    int64_t vocab           = 0;  // 0 when unknown
};

// From the file's metadata; throws std::runtime_error naming the missing key
// or an architecture other than `qwen35`.
GgufGeometry gguf_geometry(const gguf::GgufFile& file);

// One line per differing field between the file's and the artifact's
// geometry; empty means the template is the file's architecture. A zero
// vocabulary on either side is "unknown" and not compared.
std::vector<std::string> gguf_geometry_mismatches(const GgufGeometry& file, const GgufGeometry& artifact);

// Inverse of the converter's V-head reorder (`_reorder_v_heads` in its
// qwen.py): with r = v_heads / k_heads value heads per key head, HF head
// h = i * r + j sits at the file's head j * k_heads + i. Returns, for every
// HF head, the file's head holding it; the identity when the counts do not
// call for a reorder.
std::vector<int64_t> gguf_v_head_to_file_head(int64_t k_heads, int64_t v_heads);

enum class GgufReorder { None, RowsV, RowsQKV, Columns };

struct GgufModuleMap {
    std::string_view gguf_tensor;  // after "blk.N."
    GgufReorder      reorder;
};

// The served IR's module name (after "layers.N.", before "._openvino_orig_weight")
// to the file's tensor for `qwen35`; nullopt when the module is not mapped.
const GgufModuleMap* gguf_module_map(std::string_view ir_module);

}  // namespace lgc
