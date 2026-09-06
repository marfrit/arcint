#include "core/gguf_map.h"

#include <stdexcept>

namespace lgc {

namespace {

int64_t need_int(const gguf::GgufFile& f, const std::string& key) {
    auto v = f.get_int(key);
    if (!v) throw std::runtime_error("gguf: metadata key missing: " + key);
    return *v;
}

struct Entry {
    std::string_view ir_module;
    GgufModuleMap    map;
};
// The converter (qwen.py) fuses nothing for qwen3.5's GDN in-projections and
// keeps HF's fused q+gate for attention; it reorders value heads on the
// tensors marked here (design note §1).
constexpr Entry kLayerModules[] = {
    {"linear_attn.in_proj_qkv", {"attn_qkv",    GgufReorder::RowsQKV}},
    {"linear_attn.in_proj_z",   {"attn_gate",   GgufReorder::RowsV}},
    {"linear_attn.in_proj_a",   {"ssm_alpha",   GgufReorder::RowsV}},
    {"linear_attn.in_proj_b",   {"ssm_beta",    GgufReorder::RowsV}},
    {"linear_attn.out_proj",    {"ssm_out",     GgufReorder::Columns}},
    {"mlp.gate_proj",           {"ffn_gate",    GgufReorder::None}},
    {"mlp.up_proj",             {"ffn_up",      GgufReorder::None}},
    {"mlp.down_proj",           {"ffn_down",    GgufReorder::None}},
    {"self_attn.q_proj",        {"attn_q",      GgufReorder::None}},
    {"self_attn.k_proj",        {"attn_k",      GgufReorder::None}},
    {"self_attn.v_proj",        {"attn_v",      GgufReorder::None}},
    {"self_attn.o_proj",        {"attn_output", GgufReorder::None}},
};

}  // namespace

GgufGeometry gguf_geometry(const gguf::GgufFile& f) {
    const std::string arch = f.get_string("general.architecture").value_or("");
    if (arch != "qwen35")
        throw std::runtime_error("gguf: architecture '" + arch + "' is not served (stage 1 serves qwen35)");
    GgufGeometry g;
    const int64_t blocks = need_int(f, arch + ".block_count");
    const int64_t nextn  = f.get_int(arch + ".nextn_predict_layers").value_or(0);
    g.n_layers       = blocks - nextn;
    g.hidden         = need_int(f, arch + ".embedding_length");
    g.n_heads        = need_int(f, arch + ".attention.head_count");
    g.n_kv_heads     = need_int(f, arch + ".attention.head_count_kv");
    g.head_dim       = need_int(f, arch + ".attention.key_length");
    // The converter's own key choices for this family (its Qwen3Next base's
    // set_gguf_parameters, unchanged for qwen35): add_ssm_group_count(linear_
    // num_key_heads), add_ssm_time_step_rank(linear_num_value_heads),
    // add_ssm_state_size(linear_key_head_dim), add_ssm_inner_size(linear_
    // value_head_dim * linear_num_value_heads), add_ssm_conv_kernel(linear_
    // conv_kernel_dim). Read back here under those names.
    g.linear_k_heads = need_int(f, arch + ".ssm.group_count");
    g.linear_v_heads = need_int(f, arch + ".ssm.time_step_rank");
    g.linear_k_dim   = need_int(f, arch + ".ssm.state_size");
    const int64_t inner = need_int(f, arch + ".ssm.inner_size");
    if (g.linear_v_heads <= 0 || inner % g.linear_v_heads != 0)
        throw std::runtime_error("gguf: ssm.inner_size is not a multiple of the value head count");
    g.linear_v_dim   = inner / g.linear_v_heads;
    g.full_attention_interval = need_int(f, arch + ".full_attention_interval");
    if (const auto* toks = f.raw_meta("tokenizer.ggml.tokens"))
        g.vocab = static_cast<int64_t>(toks->strings.size());
    return g;
}

std::vector<std::string> gguf_geometry_mismatches(const GgufGeometry& a, const GgufGeometry& b) {
    std::vector<std::string> out;
    auto cmp = [&](const char* name, int64_t x, int64_t y) {
        if (x != y) out.push_back(std::string(name) + ": file " + std::to_string(x) + ", artifact " + std::to_string(y));
    };
    cmp("layers", a.n_layers, b.n_layers);
    cmp("hidden size", a.hidden, b.hidden);
    cmp("attention heads", a.n_heads, b.n_heads);
    cmp("kv heads", a.n_kv_heads, b.n_kv_heads);
    cmp("head dim", a.head_dim, b.head_dim);
    cmp("GDN key heads", a.linear_k_heads, b.linear_k_heads);
    cmp("GDN value heads", a.linear_v_heads, b.linear_v_heads);
    cmp("GDN key head dim", a.linear_k_dim, b.linear_k_dim);
    cmp("GDN value head dim", a.linear_v_dim, b.linear_v_dim);
    cmp("full-attention interval", a.full_attention_interval, b.full_attention_interval);
    if (a.vocab && b.vocab) cmp("vocabulary", a.vocab, b.vocab);
    return out;
}

std::vector<int64_t> gguf_v_head_to_file_head(int64_t k_heads, int64_t v_heads) {
    std::vector<int64_t> to_file(static_cast<size_t>(std::max<int64_t>(v_heads, 0)));
    for (int64_t h = 0; h < v_heads; ++h) to_file[static_cast<size_t>(h)] = h;
    if (k_heads <= 0 || v_heads <= 0 || v_heads % k_heads != 0 || v_heads == k_heads) return to_file;
    const int64_t r = v_heads / k_heads;
    for (int64_t h = 0; h < v_heads; ++h) {
        const int64_t i = h / r, j = h % r;
        to_file[static_cast<size_t>(h)] = j * k_heads + i;
    }
    return to_file;
}

const GgufModuleMap* gguf_module_map(std::string_view ir_module) {
    for (const auto& e : kLayerModules)
        if (e.ir_module == ir_module) return &e.map;
    return nullptr;
}

}  // namespace lgc
