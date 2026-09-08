#include "exec/gguf_graph.h"

#include "core/gguf_repack.h"

#include <chrono>
#include <cstdlib>
#include <map>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>
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

GgufWeightsMode gguf_tensor_mode(GgufWeightsMode mode, int32_t ggml_type) {
    if (mode != GgufWeightsMode::Mixed) return mode;
    return ggml_type == static_cast<int32_t>(gguf::GgmlType::Q4_K) ? GgufWeightsMode::Repack : GgufWeightsMode::Native;
}

std::string GgufApplyReport::summary() const {
    std::unordered_map<int32_t, std::pair<int, size_t>> per_type;
    size_t repacked = 0;
    for (const auto& r : replaced) { per_type[r.ggml_type].first++; per_type[r.ggml_type].second += r.bytes; if (r.repacked) ++repacked; }
    std::ostringstream s;
    s << replaced.size() << " projection(s) from the file";
    if (mode == GgufWeightsMode::Repack) s << " (repacked)";
    else if (mode == GgufWeightsMode::Native) s << " (native rows)";
    else s << " (mixed: " << repacked << " repacked, " << (replaced.size() - repacked) << " native rows)";
    for (const auto& [t, cb] : per_type)
        s << ", " << gguf::type_name(t) << " x" << cb.first << " (" << (cb.second >> 20) << " MiB)";
    if (repacked != 0) {
        s << "; mins " << (mins == gguf::RepackMins::Exact ? "exact" : mins == gguf::RepackMins::Shared ? "shared (two super-blocks per augmented group, inexact)"
                          : mins == gguf::RepackMins::Nibble ? "nibble (one nibble per group, inexact)"
                          : "split (the min term a separate MatMul, no augmented columns; the file's mins under one f16 rounding, bound 2x)");
        s << "; repack deviation max " << repack_max_steps << " quantisation step(s) over " << repack_checked << " value(s), "
          << repack_over_bound << (mins == gguf::RepackMins::Exact || mins == gguf::RepackMins::Split ? " over bound" : " over the exact bound (accepted: --gguf-mins)");
        if (repack_verdicts_cached != 0) s << " (" << repack_verdicts_cached << " verdict(s) from an earlier load)";
        s << "; repack+check " << static_cast<int>(repack_seconds) << " s on " << repack_workers << " worker(s)";
    }
    if (q6k_aligned != 0) s << "; " << q6k_aligned << " Q6_K projection(s) in 224-byte blocks";
    s << "; " << kept.size() << " constant(s) kept from the template; " << awq_scales_neutralized
      << " AWQ activation scale(s) set to one; " << norms_compared << " norm(s) compared with the file, max |diff| "
      << norms_max_abs_diff;
    return s.str();
}

// The plugin's own decompression form: Const -> Convert(f16) -> [Subtract zp]
// -> Multiply scale -> Reshape [n, width] -> MatMul, the chain the exporter
// writes and the runtime folds into its compressed fully-connected primitive.
// For the augmented types (Q4_K, Q5_K: core/gguf_repack.h) the activation is
// widened first by its group sums through the augmentation matrix -- one
// reduce, one small matmul and one concat, built once per distinct
// activation and shared by every projection reading it. RepackMins::Split
// takes no augmented columns, but reads the same group sums for its own min
// term (activation_group_sums, below) -- the cache holds both, keyed apart.
struct Widened {
    ov::Output<ov::Node> out;
};

// [.., K] -> [.., K/32, 32] (special zero keeps the leading dims) -> sum over
// 32 -> [.., K/32], one reduce per distinct activation, cached and shared by
// every projection reading it (the augmented forms' widening and
// RepackMins::Split's min term alike).
static ov::Output<ov::Node> activation_group_sums(const ov::Output<ov::Node>& act, int64_t k,
                                                   std::map<std::string, Widened>& cache, const std::string& name) {
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(act.get_node())) + ":" + std::to_string(act.get_index()) +
                            ":sums:" + std::to_string(k);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.out;
    const int64_t groups = k / 32;
    const auto rank = act.get_partial_shape().rank();
    if (!rank.is_static() || (rank.get_length() != 2 && rank.get_length() != 3))
        throw std::runtime_error("gguf: the activation of " + name + " is not 2-D or 3-D");
    const bool three_d = rank.get_length() == 3;
    std::vector<int64_t> shape = three_d ? std::vector<int64_t>{0, 0, groups, 32} : std::vector<int64_t>{0, groups, 32};
    auto shape_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape);
    auto grouped = std::make_shared<ov::op::v1::Reshape>(act, shape_c, true);
    auto axis_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{three_d ? 3 : 2});
    auto sums = std::make_shared<ov::op::v1::ReduceSum>(grouped, axis_c, false);
    cache[key] = Widened{sums->output(0)};
    return sums->output(0);
}

static ov::Output<ov::Node> widen_activation(const ov::Output<ov::Node>& act, const gguf::RepackedTensor& r,
                                             std::map<std::string, Widened>& cache, const std::string& name) {
    if (r.k_aug == 0) return act;
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(act.get_node())) + ":" + std::to_string(act.get_index()) +
                            ":widened:" + std::to_string(r.k) + ":" + std::to_string(r.k_aug) + ":" + std::to_string(r.groups_per_aug) +
                            ":" + (r.weights_type == gguf::RepackWeights::U4 ? "u4" : "u8");
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.out;
    const int64_t groups = r.k / 32;
    const auto sums = activation_group_sums(act, r.k, cache, name);
    const bool three_d = act.get_partial_shape().rank().get_length() == 3;
    const auto mbits = gguf::augmentation_matrix(r);
    auto m_owner = std::make_shared<std::vector<uint16_t>>(mbits);
    auto m = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{static_cast<size_t>(groups), static_cast<size_t>(r.k_aug)},
                                                    static_cast<const void*>(m_owner->data()), std::shared_ptr<void>(m_owner, m_owner->data()));
    m->set_friendly_name(name + "/gguf_augmentation");
    std::shared_ptr<ov::Node> m_typed = m;
    if (act.get_element_type() != ov::element::f16)  // the toy template's f32 activations; the served model's are f16
        m_typed = std::make_shared<ov::op::v0::Convert>(m, act.get_element_type());
    auto aug = std::make_shared<ov::op::v0::MatMul>(sums, m_typed, false, false);
    auto wide = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{act, aug->output(0)}, three_d ? 2 : 1);
    wide->set_friendly_name(name + "/gguf_widened");
    cache[key] = Widened{wide->output(0)};
    return wide->output(0);
}

// RepackMins::Split's min term: y += MatMul(sums, -min_matrix^T), added to the
// main MatMul's output (the caller). NegMins[g][row] = -min_matrix[row][g],
// transposed and sign-flipped (exact: negating an f16 value is exactly
// flipping its sign bit, never a rounding).
static std::shared_ptr<ov::Node> split_min_term(const ov::Output<ov::Node>& act, const gguf::RepackedTensor& r,
                                                std::map<std::string, Widened>& cache, const std::string& name) {
    const int64_t groups = r.k / 32;
    auto neg = std::make_shared<std::vector<uint16_t>>(static_cast<size_t>(groups * r.n));
    for (int64_t row = 0; row < r.n; ++row)
        for (int64_t g = 0; g < groups; ++g)
            (*neg)[static_cast<size_t>(g * r.n + row)] = static_cast<uint16_t>(r.min_matrix[static_cast<size_t>(row * groups + g)] ^ 0x8000u);
    auto m = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{static_cast<size_t>(groups), static_cast<size_t>(r.n)},
                                                    static_cast<const void*>(neg->data()), std::shared_ptr<void>(neg, neg->data()));
    m->set_friendly_name(name + "/gguf_min_term");
    std::shared_ptr<ov::Node> m_typed = m;
    if (act.get_element_type() != ov::element::f16)
        m_typed = std::make_shared<ov::op::v0::Convert>(m, act.get_element_type());
    const auto sums = activation_group_sums(act, r.k, cache, name);
    return std::make_shared<ov::op::v0::MatMul>(sums, m_typed, false, false);
}

static std::shared_ptr<ov::Node> repacked_matmul(const ov::Output<ov::Node>& act, const gguf::RepackedTensor& r,
                                                 const std::shared_ptr<ov::op::v0::MatMul>& mm, const std::string& name,
                                                 std::map<std::string, Widened>& cache) {
    const size_t n = static_cast<size_t>(r.n), width = static_cast<size_t>(r.width()), gs = static_cast<size_t>(r.group), groups = width / gs;
    const ov::element::Type wt = r.weights_type == gguf::RepackWeights::U4 ? ov::element::u4
                               : r.weights_type == gguf::RepackWeights::U8 ? ov::element::u8 : ov::element::i8;
    auto owner = std::make_shared<gguf::RepackedTensor>(r);  // the constants alias these vectors
    auto w = std::make_shared<ov::op::v0::Constant>(wt, ov::Shape{n, groups, gs}, static_cast<const void*>(owner->weights.data()),
                                                    std::shared_ptr<void>(owner, owner->weights.data()));
    w->set_friendly_name(name + "/gguf_repacked");
    std::shared_ptr<ov::Node> x = std::make_shared<ov::op::v0::Convert>(w, ov::element::f16);
    if (r.zp_type == gguf::RepackZeroPoint::U8Scalar) {
        auto zp = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{1, 1, 1}, std::vector<uint8_t>{r.zp_u8});
        zp->set_friendly_name(name + "/gguf_repacked/zero_point");
        x = std::make_shared<ov::op::v1::Subtract>(x, std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16));
    }
    auto sc = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{n, groups, 1}, static_cast<const void*>(owner->scale.data()),
                                                     std::shared_ptr<void>(owner, owner->scale.data()));
    sc->set_friendly_name(name + "/gguf_repacked/scale");
    x = std::make_shared<ov::op::v1::Multiply>(x, sc);
    auto shape_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{static_cast<int64_t>(n), static_cast<int64_t>(width)});
    x = std::make_shared<ov::op::v1::Reshape>(x, shape_c, false);
    if (mm->get_input_element_type(1) != ov::element::f16)
        x = std::make_shared<ov::op::v0::Convert>(x, mm->get_input_element_type(1));
    auto main_mm = std::make_shared<ov::op::v0::MatMul>(act, x, mm->get_transpose_a(), mm->get_transpose_b());
    if (r.mins != gguf::RepackMins::Split || r.min_matrix.empty()) return main_mm;
    // RepackMins::Split: the min term is a second, unquantised MatMul added to
    // the main one, not augmented columns of it -- the point of the flag
    // (DESIGN §7.0.2bg's `--dyn-quant on` finding: the augmented columns ride
    // through the same quantised fully-connected as the weights).
    auto minterm = split_min_term(act, r, cache, name);
    // The min constant owns its own (negated, transposed) copy; the repacked tensor's
    // min_matrix has nothing left to alias (5.5 MB per gate on the served file).
    owner->min_matrix.clear(); owner->min_matrix.shrink_to_fit();
    auto added = std::make_shared<ov::op::v1::Add>(main_mm->output(0), minterm->output(0));
    added->set_friendly_name(name + "/gguf_split");
    return added;
}

GgufApplyReport gguf_apply_to_template(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<gguf::GgufFile>& file,
                                       const GgufGeometry& g,
                                       GgufWeightsMode mode,
                                       const std::string& verdict_dir,
                                       const std::string& file_path,
                                       bool q6k_aligned,
                                       gguf::RepackMins mins) {
    GgufApplyReport rep;
    rep.mode = mode;
    rep.mins = mins;
    using wall = std::chrono::steady_clock;
    std::map<std::string, Widened> widened;  // one widening per distinct activation (repack mode)
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

    // Phase (a), sequential and cheap: resolve every weight's target tensor,
    // its MatMul and (for a projection to repack) the column order and bound
    // its check runs under -- everything gguf_repack_all needs, and nothing
    // that touches a tensor's bytes. No graph edit is APPLIED here (replace_node
    // is not called until phase (c)); a column-reorder Gather may be built
    // (its node exists, unconnected) and then discarded below when the
    // tensor turns out to repack instead, same as before this was split.
    struct WeightPlan {
        std::shared_ptr<ov::op::v0::Constant> c;
        std::string name, gguf_name;
        GgufReorder reorder = GgufReorder::None;
        std::shared_ptr<ov::op::v0::MatMul> mm;
        const gguf::TensorInfo* t = nullptr;
        int64_t n = 0, k = 0;
        ov::Output<ov::Node> act_out;
        bool columns_gathered = false;
        int64_t row_bytes = 0;
        GgufWeightsMode tmode = GgufWeightsMode::Native;
        std::vector<int64_t> dest_of;  // repack mode only (empty: the file's own order)
        double bound = 0.0;            // repack mode only
        int repack_index = -1;         // index into repack_results; -1 when not repacked
    };
    std::vector<WeightPlan> plans;
    plans.reserve(weights.size());
    std::vector<gguf::RepackRequest> requests;

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

        WeightPlan p;
        p.c = c; p.name = name; p.gguf_name = gguf_name; p.reorder = reorder; p.mm = mm; p.t = t; p.n = n; p.k = k;
        p.act_out = mm->input_value(0);

        // The activation-side permutation for a column-reordered projection
        // (undone below instead, for a repacked one -- the mins then follow
        // the head-wise order at build, not a gather at every forward).
        if (reorder == GgufReorder::Columns) {
            std::vector<int64_t> idx(static_cast<size_t>(k));
            const int64_t d = g.linear_v_dim;
            for (int64_t hf = 0; hf < g.linear_v_heads; ++hf)
                for (int64_t e = 0; e < d; ++e) idx[to_file[hf] * d + e] = hf * d + e;
            auto idx_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(k)}, idx);
            auto axis_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
            auto gather = std::make_shared<ov::op::v8::Gather>(p.act_out, idx_c, axis_c);
            gather->set_friendly_name(mm->get_friendly_name() + "/gguf_v_head_order");
            p.act_out = gather->output(0);
            p.columns_gathered = true;
        }

        p.row_bytes = FullyConnectedKQuant::row_bytes(t->ggml_type, k);
        p.tmode = gguf_tensor_mode(mode, t->ggml_type);

        if (p.row_bytes != 0 && p.tmode == GgufWeightsMode::Repack) {
            // A column-reordered projection (the output projection) takes its
            // order at build; the mins' augmented groups then follow the heads.
            if (reorder == GgufReorder::Columns) {
                p.act_out = mm->input_value(0);  // not the gather built above: the columns themselves move
                p.columns_gathered = false;
                p.dest_of.resize(static_cast<size_t>(k));
                const int64_t d = g.linear_v_dim;
                for (int64_t hf = 0; hf < g.linear_v_heads; ++hf)
                    for (int64_t e = 0; e < d; ++e) p.dest_of[static_cast<size_t>(to_file[static_cast<size_t>(hf)] * d + e)] = hf * d + e;
            }
            // The split form's bound: the exact form's terms plus one f16 rounding of the
            // min product dmin * mn (up to 2^-11 of the min, in steps -- not bounded a
            // priori). Twice the exact bound is the MEASURED bound: it holds on the fixture
            // (0.0158 on Q4_K against 1/64) and on the served file. Refused over it like
            // the exact form: the mins are the file's, the form is exact-class.
            p.bound = gguf::repack_bound_steps(t->ggml_type) * (mins == gguf::RepackMins::Split ? 2.0 : 1.0);
            p.repack_index = static_cast<int>(requests.size());
            requests.push_back(gguf::RepackRequest{t, p.dest_of, p.bound});
        }
        plans.push_back(std::move(p));
    }

    // Phase (b): every tensor to repack, at once, cross-tensor, in a bounded
    // worker pool (core/gguf_repack.h's gguf_repack_all -- device-free,
    // tested without OpenVINO in tests/test_gguf_parallel.cpp). repack_seconds
    // is the wall time of this WHOLE phase, not a sum of per-tensor repack
    // calls: the repack and the deviation check of every requested tensor
    // (cache read; on a miss, the check over the bound and the verdict
    // write) happen inside it, so there is no separate check_seconds to time
    // any more -- it stays at 0 for callers of the struct. A tensor over its
    // bound (RepackMins::Exact/Split) throws from inside gguf_repack_all,
    // after every request has been attempted, naming the tensor -- the exact
    // message this loop used to throw inline.
    std::vector<gguf::RepackResult> repack_results;
    if (!requests.empty()) {
        const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
        const auto t0 = wall::now();
        repack_results = gguf::gguf_repack_all(*file, requests, mins, verdict_dir, file_path, workers);
        rep.repack_seconds = std::chrono::duration<double>(wall::now() - t0).count();
        rep.repack_workers = workers;
    }

    // Phase (c), sequential, in the file's own tensor order: exactly the
    // graph edits phase (a) used to do inline, now reading a repacked
    // tensor's RepackedTensor and verdict from `repack_results` instead of
    // computing them here.
    for (auto& p : plans) {
        const std::string& name = p.name;
        const std::string& gguf_name = p.gguf_name;
        const GgufReorder reorder = p.reorder;
        const auto& mm = p.mm;
        const auto* t = p.t;
        const int64_t n = p.n, k = p.k;
        ov::Output<ov::Node> act_out = p.act_out;

        GgufReplacement r;
        r.ir_name = name; r.gguf_name = gguf_name; r.ggml_type = t->ggml_type; r.n = n; r.k = k;
        r.columns_gathered = p.columns_gathered;

        const int64_t row_bytes = p.row_bytes;
        std::shared_ptr<ov::Node> replacement;
        std::shared_ptr<ov::Node> post;  // an output-side gather when the rows were reordered
        const GgufWeightsMode tmode = p.tmode;
        if (row_bytes != 0) {
            if (tmode == GgufWeightsMode::Repack) {
                r.repacked = true;
                gguf::RepackResult& rr = repack_results[static_cast<size_t>(p.repack_index)];
                gguf::RepackedTensor packed = std::move(rr.packed);
                rep.repack_max_steps = std::max(rep.repack_max_steps, rr.max_steps);
                if (rr.verdict_cached) {
                    // Checked and passed by an earlier load of this very file (size, mtime,
                    // offset, type, dims); the repack itself is deterministic in the bytes.
                    ++rep.repack_verdicts_cached;
                } else {
                    rep.repack_over_bound += rr.over_bound;
                    rep.repack_checked += rr.checked;
                }
                // The value-head reorder is undone in the repacked rows and column
                // groups themselves, not by gathers in the graph (a gather per
                // projection cost the plugin an activation buffer per token: the fit
                // measured 2.65 GiB at chunk 256 with them, DESIGN §7.0.2ba).
                if (reorder == GgufReorder::RowsV || reorder == GgufReorder::RowsQKV) {
                    const int64_t first = reorder == GgufReorder::RowsQKV ? 2 * qk_rows : 0;
                    if (first + v_rows != n)
                        throw std::runtime_error("gguf: " + gguf_name + " has " + std::to_string(n) + " rows; expected " +
                                                 std::to_string(first) + " q/k rows and " + std::to_string(v_rows) + " value rows");
                    std::vector<int64_t> src_of(static_cast<size_t>(n));
                    for (int64_t i = 0; i < n; ++i) src_of[static_cast<size_t>(i)] = i;
                    const int64_t d = g.linear_v_dim;
                    for (int64_t h = 0; h < g.linear_v_heads; ++h)
                        for (int64_t e = 0; e < d; ++e) src_of[static_cast<size_t>(first + h * d + e)] = first + to_file[static_cast<size_t>(h)] * d + e;
                    gguf::permute_rows(packed, src_of);
                    r.rows_permuted = true;
                } else if (reorder == GgufReorder::Columns) {
                    r.rows_permuted = true;  // the order lives in the constant
                }
                replacement = repacked_matmul(widen_activation(act_out, packed, widened, name), packed, mm, name, widened);
                r.bytes = packed.bytes();
            } else if (q6k_aligned && t->ggml_type == static_cast<int32_t>(gguf::GgmlType::Q6_K)) {
                // The file's 210-byte super-blocks are 2-aligned and a block read needs a dword
                // (DESIGN 7.0.2bc, 7.0.2bh): laid out at 224 bytes each -- the block's bytes, then
                // 14 zero bytes -- every block is dword-aligned and the decode row reads its own
                // words with no shuffles (kquant type 114, DESIGN 7.0.2bj). +6.7 % on the Q6_K set.
                const int64_t blocks = k / 256;
                auto owner = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(n) * static_cast<size_t>(blocks) * 224, uint8_t{0});
                const uint8_t* src = file->data(*t);
                for (int64_t rb = 0; rb < n * blocks; ++rb)
                    std::memcpy(owner->data() + static_cast<size_t>(rb) * 224, src + static_cast<size_t>(rb) * 210, 210);
                auto w = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{static_cast<size_t>(n), static_cast<size_t>(blocks) * 224},
                                                                static_cast<const void*>(owner->data()), std::shared_ptr<void>(owner, owner->data()));
                w->set_friendly_name(name + "/gguf_224");
                replacement = std::make_shared<FullyConnectedKQuant>(act_out, w, 114, k, n);
                r.bytes = owner->size();
                rep.q6k_aligned++;
            } else {
                auto w = u8_constant(file->data(*t), n, row_bytes, file);
                w->set_friendly_name(name + "/gguf");
                replacement = std::make_shared<FullyConnectedKQuant>(act_out, w, t->ggml_type, k, n);
                r.bytes = static_cast<size_t>(n) * static_cast<size_t>(row_bytes);
            }
            if (tmode == GgufWeightsMode::Native && (reorder == GgufReorder::RowsV || reorder == GgufReorder::RowsQKV)) {
                const int64_t first = reorder == GgufReorder::RowsQKV ? 2 * qk_rows : 0;
                if (first + v_rows != n)
                    throw std::runtime_error("gguf: " + gguf_name + " has " + std::to_string(n) + " rows; expected " +
                                             std::to_string(first) + " q/k rows and " + std::to_string(v_rows) + " value rows");
                post = gather_rows_back(replacement->output(0), n, first, to_file, g.linear_v_dim);
                post->set_friendly_name(mm->get_friendly_name() + "/gguf_v_head_order");
                r.rows_permuted = true;
            }
        } else if (t->ggml_type == static_cast<int32_t>(gguf::GgmlType::F32) ||
                   t->ggml_type == static_cast<int32_t>(gguf::GgmlType::F16)) {
            // A float projection (the dense file's GDN alpha/beta): an f16
            // constant feeding the template's MatMul, rows un-reordered.
            std::vector<float> f(static_cast<size_t>(n) * static_cast<size_t>(k));
            gguf::dequantize_tensor(*file, *t, f);
            if (reorder == GgufReorder::RowsV) {
                std::vector<float> fp(f.size());  // named apart from the outer WeightPlan `p`
                const int64_t head_rows = n / g.linear_v_heads;
                for (int64_t hf = 0; hf < g.linear_v_heads; ++hf)
                    for (int64_t d = 0; d < head_rows; ++d)
                        std::copy_n(f.data() + (to_file[hf] * head_rows + d) * k, k, fp.data() + (hf * head_rows + d) * k);
                f.swap(fp);
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
