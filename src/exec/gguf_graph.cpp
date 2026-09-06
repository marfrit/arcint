#include "exec/gguf_graph.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/runtime/tensor.hpp>

#include "core/gguf_dequant.h"
#include "core/gguf_map.h"
#include "exec/kquant_op.h"

namespace lgc {

namespace {

constexpr const char* kLayerPrefix  = "self.model.model.language_model.layers.";
constexpr const char* kWeightSuffix = "._openvino_orig_weight";
constexpr const char* kHeadConst    = "self.model.lm_head._openvino_orig_weight";

// Follows the single-consumer chain from a weight constant to the MatMul it
// feeds (Const -> Convert -> Subtract -> Multiply -> [Reshape] -> Convert ->
// MatMul, the exporter's decompression pattern), or nullptr when the chain is
// not that shape.
std::shared_ptr<ov::op::v0::MatMul> matmul_of(const std::shared_ptr<ov::Node>& constant) {
    std::shared_ptr<ov::Node> cur = constant;
    for (int hop = 0; hop < 8; ++hop) {
        const auto& targets = cur->output(0).get_target_inputs();
        if (targets.size() != 1) return nullptr;
        auto next = targets.begin()->get_node()->shared_from_this();
        if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(next)) {
            // The op built in place of it assumes x[.., K] against W[N, K]: the
            // exporter's `transpose_b` form, refused by name otherwise.
            if (targets.begin()->get_index() != 1) return nullptr;
            if (mm->get_transpose_a() || !mm->get_transpose_b())
                throw std::runtime_error("gguf: " + constant->get_friendly_name() + " feeds a MatMul without the exporter's transpose_b form");
            return mm;
        }
        cur = next;
    }
    return nullptr;
}

// A u8 [N, row_bytes] constant over the file's own bytes: the constructor that
// takes an owner aliases the memory and keeps the file alive with the node
// (no copy; a Tensor-wrapping constructor would alias without owning).
std::shared_ptr<ov::op::v0::Constant> u8_constant(const uint8_t* bytes, int64_t n, int64_t row_bytes,
                                                  const std::shared_ptr<gguf::GgufFile>& owner) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::u8,
                                                  ov::Shape{static_cast<size_t>(n), static_cast<size_t>(row_bytes)},
                                                  static_cast<const void*>(bytes),
                                                  std::static_pointer_cast<void>(owner));
}

// Output-side inverse of the converter's value-head reorder for a projection
// whose ROWS were reordered: the op computes y_file = W_file · x, and HF head
// h's slice of y is the file's head to_file[h]. A Gather on the last axis with
// idx[h*d + e] = first + to_file[h]*d + e (identity below `first`) puts the
// rows back in HF order without touching the weight bytes.
std::shared_ptr<ov::Node> gather_rows_back(const ov::Output<ov::Node>& y, int64_t n, int64_t first,
                                           const std::vector<int64_t>& to_file, int64_t head_rows) {
    std::vector<int64_t> idx(static_cast<size_t>(n));
    for (int64_t i = 0; i < first; ++i) idx[static_cast<size_t>(i)] = i;
    for (size_t h = 0; h < to_file.size(); ++h)
        for (int64_t d = 0; d < head_rows; ++d)
            idx[static_cast<size_t>(first + static_cast<int64_t>(h) * head_rows + d)] = first + to_file[h] * head_rows + d;
    auto idx_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(n)}, idx);
    auto axis_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    return std::make_shared<ov::op::v8::Gather>(y, idx_c, axis_c);
}

}  // namespace

std::string GgufApplyReport::summary() const {
    std::unordered_map<int32_t, std::pair<int, size_t>> per_type;
    for (const auto& r : replaced) { per_type[r.ggml_type].first++; per_type[r.ggml_type].second += r.bytes; }
    std::ostringstream s;
    s << replaced.size() << " projection(s) from the file";
    for (const auto& [t, cb] : per_type)
        s << ", " << gguf::type_name(t) << " x" << cb.first << " (" << (cb.second >> 20) << " MiB)";
    s << "; " << kept.size() << " constant(s) kept from the template; " << awq_scales_neutralized
      << " AWQ activation scale(s) set to one; " << norms_compared << " norm(s) compared with the file, max |diff| "
      << norms_max_abs_diff;
    return s.str();
}

GgufApplyReport gguf_apply_to_template(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<gguf::GgufFile>& file,
                                       const GgufGeometry& g) {
    GgufApplyReport rep;
    const auto to_file = gguf_v_head_to_file_head(g.linear_k_heads, g.linear_v_heads);
    const int64_t v_rows = g.linear_v_heads * g.linear_v_dim;
    const int64_t qk_rows = g.linear_k_heads * g.linear_k_dim;

    // Snapshot the constants first: replacing nodes while iterating get_ops() is undefined.
    std::vector<std::shared_ptr<ov::op::v0::Constant>> weights;
    for (const auto& op : model->get_ops()) {
        auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
        if (!c) continue;
        const std::string& name = c->get_friendly_name();
        if (name.size() > std::strlen(kWeightSuffix) &&
            name.compare(name.size() - std::strlen(kWeightSuffix), std::string::npos, kWeightSuffix) == 0)
            weights.push_back(c);
    }

    for (const auto& c : weights) {
        const std::string name = c->get_friendly_name();
        std::string gguf_name;
        GgufReorder reorder = GgufReorder::None;
        int64_t layer = -1;
        if (name == kHeadConst) {
            gguf_name = "output.weight";
        } else if (name.rfind(kLayerPrefix, 0) == 0) {
            const std::string rest = name.substr(std::strlen(kLayerPrefix));
            const size_t dot = rest.find('.');
            layer = std::stoll(rest.substr(0, dot));
            const std::string module = rest.substr(dot + 1, rest.size() - dot - 1 - std::strlen(kWeightSuffix));
            const GgufModuleMap* found = gguf_module_map(module);
            if (!found) throw std::runtime_error("gguf: no tensor map for IR constant " + name);
            gguf_name = "blk." + std::to_string(layer) + "." + std::string(found->gguf_tensor) + ".weight";
            reorder = found->reorder;
        } else {
            rep.kept.push_back(name);
            continue;
        }

        auto mm = matmul_of(c);
        if (!mm) throw std::runtime_error("gguf: IR constant " + name + " does not feed a MatMul through the exporter's pattern");
        const auto* t = file->tensor(gguf_name);
        if (!t) throw std::runtime_error("gguf: tensor " + gguf_name + " missing from the file (needed for " + name + ")");
        if (t->dims.size() != 2) throw std::runtime_error("gguf: tensor " + gguf_name + " is not 2-D");
        const int64_t k = static_cast<int64_t>(t->dims[0]), n = static_cast<int64_t>(t->dims[1]);
        const auto ir_shape = c->get_shape();
        if (ir_shape.empty() || static_cast<int64_t>(ir_shape[0]) != n)
            throw std::runtime_error("gguf: " + gguf_name + " has " + std::to_string(n) + " rows, the template's " + name +
                                     " has " + std::to_string(ir_shape.empty() ? 0 : ir_shape[0]));

        ov::Output<ov::Node> act_out = mm->input_value(0);
        GgufReplacement r;
        r.ir_name = name; r.gguf_name = gguf_name; r.ggml_type = t->ggml_type; r.n = n; r.k = k;

        // The activation-side permutation for a column-reordered projection.
        if (reorder == GgufReorder::Columns) {
            std::vector<int64_t> idx(static_cast<size_t>(k));
            const int64_t d = g.linear_v_dim;
            for (int64_t hf = 0; hf < g.linear_v_heads; ++hf)
                for (int64_t e = 0; e < d; ++e) idx[to_file[hf] * d + e] = hf * d + e;
            auto idx_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(k)}, idx);
            auto axis_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
            auto gather = std::make_shared<ov::op::v8::Gather>(act_out, idx_c, axis_c);
            gather->set_friendly_name(mm->get_friendly_name() + "/gguf_v_head_order");
            act_out = gather->output(0);
            r.columns_gathered = true;
        }

        const int64_t row_bytes = FullyConnectedKQuant::row_bytes(t->ggml_type, k);
        std::shared_ptr<ov::Node> replacement;
        std::shared_ptr<ov::Node> post;  // an output-side gather when the rows were reordered
        if (row_bytes != 0) {
            auto w = u8_constant(file->data(*t), n, row_bytes, file);
            w->set_friendly_name(name + "/gguf");
            replacement = std::make_shared<FullyConnectedKQuant>(act_out, w, t->ggml_type, k, n);
            if (reorder == GgufReorder::RowsV || reorder == GgufReorder::RowsQKV) {
                const int64_t first = reorder == GgufReorder::RowsQKV ? 2 * qk_rows : 0;
                if (first + v_rows != n)
                    throw std::runtime_error("gguf: " + gguf_name + " has " + std::to_string(n) + " rows; expected " +
                                             std::to_string(first) + " q/k rows and " + std::to_string(v_rows) + " value rows");
                post = gather_rows_back(replacement->output(0), n, first, to_file, g.linear_v_dim);
                post->set_friendly_name(mm->get_friendly_name() + "/gguf_v_head_order");
                r.rows_permuted = true;
            }
            r.bytes = static_cast<size_t>(n) * static_cast<size_t>(row_bytes);
        } else if (t->ggml_type == static_cast<int32_t>(gguf::GgmlType::F32) ||
                   t->ggml_type == static_cast<int32_t>(gguf::GgmlType::F16)) {
            // A float projection (the dense file's GDN alpha/beta): an f16
            // constant feeding the template's MatMul, rows un-reordered.
            std::vector<float> f(static_cast<size_t>(n) * static_cast<size_t>(k));
            gguf::dequantize_tensor(*file, *t, f);
            if (reorder == GgufReorder::RowsV) {
                std::vector<float> p(f.size());
                const int64_t head_rows = n / g.linear_v_heads;
                for (int64_t hf = 0; hf < g.linear_v_heads; ++hf)
                    for (int64_t d = 0; d < head_rows; ++d)
                        std::copy_n(f.data() + (to_file[hf] * head_rows + d) * k, k, p.data() + (hf * head_rows + d) * k);
                f.swap(p);
                r.rows_permuted = true;
            }
            auto w32 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(n), static_cast<size_t>(k)}, f);
            w32->set_friendly_name(name + "/gguf");
            auto wconv = std::make_shared<ov::op::v0::Convert>(w32, mm->get_input_element_type(1));
            replacement = std::make_shared<ov::op::v0::MatMul>(act_out, wconv, mm->get_transpose_a(), mm->get_transpose_b());
            r.bytes = f.size() * sizeof(float);
        } else {
            throw std::runtime_error("gguf: tensor " + gguf_name + " is " + gguf::type_name(t->ggml_type) +
                                     ", which stage 1 does not serve");
        }
        replacement->set_friendly_name(mm->get_friendly_name());
        ov::copy_runtime_info(mm, replacement);
        ov::replace_node(mm, post ? post : replacement);
        rep.replaced.push_back(r);
        rep.bytes_from_file += r.bytes;
    }
    // AWQ's activation-side compensations and the norms, second pass over the
    // remaining constants (the graph now holds the new ops; constants are
    // untouched by replace_node).
    const std::string awq_tag = "awq_mul/scale";
    std::vector<std::shared_ptr<ov::op::v0::Constant>> consts;
    for (const auto& op : model->get_ops())
        if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(op)) consts.push_back(c);
    for (const auto& c : consts) {
        const std::string name = c->get_friendly_name();
        if (name.find(awq_tag) != std::string::npos) {
            std::shared_ptr<ov::op::v0::Constant> one;
            if (c->get_element_type() == ov::element::f16) {
                std::vector<ov::float16> ones(ov::shape_size(c->get_shape()), ov::float16(1.0f));
                one = ov::op::v0::Constant::create(ov::element::f16, c->get_shape(), ones);
            } else {
                std::vector<float> ones(ov::shape_size(c->get_shape()), 1.0f);
                one = ov::op::v0::Constant::create(ov::element::f32, c->get_shape(), ones);
            }
            one->set_friendly_name(name + "/gguf_one");
            ov::copy_runtime_info(c, one);
            ov::replace_node(c, one);
            rep.awq_scales_neutralized++;
            continue;
        }
        // A norm weight is an f32 constant whose one consumer is the norm's own multiply.
        if (c->get_element_type() != ov::element::f32) continue;
        const auto& targets = c->output(0).get_target_inputs();
        if (targets.size() != 1) continue;
        const std::string consumer = targets.begin()->get_node()->get_friendly_name();
        std::string gname;
        const size_t lp = consumer.find("layers.");
        if (lp != std::string::npos) {
            const size_t dot = consumer.find('.', lp + 7);
            const std::string li = consumer.substr(lp + 7, dot - lp - 7);
            if (consumer.find(".input_layernorm/") != std::string::npos) gname = "blk." + li + ".attn_norm.weight";
            else if (consumer.find(".post_attention_layernorm/") != std::string::npos) gname = "blk." + li + ".post_attention_norm.weight";
            else if (consumer.find(".self_attn.q_norm/") != std::string::npos) gname = "blk." + li + ".attn_q_norm.weight";
            else if (consumer.find(".self_attn.k_norm/") != std::string::npos) gname = "blk." + li + ".attn_k_norm.weight";
            else if (consumer.find(".linear_attn.norm/") != std::string::npos) gname = "blk." + li + ".ssm_norm.weight";
        } else if (consumer.find("language_model.norm/") != std::string::npos) {
            gname = "output_norm.weight";
        }
        if (gname.empty()) continue;
        const auto* t = file->tensor(gname);
        if (!t || t->n_elements != ov::shape_size(c->get_shape())) continue;
        std::vector<float> fv(t->n_elements);
        gguf::dequantize_tensor(*file, *t, fv);
        const auto iv = c->cast_vector<float>();
        double md = 0;
        for (size_t i = 0; i < fv.size(); ++i) md = std::max(md, static_cast<double>(std::fabs(fv[i] - iv[i])));
        rep.norms_compared++;
        rep.norms_max_abs_diff = std::max(rep.norms_max_abs_diff, md);
    }
    return rep;
}

}  // namespace lgc
