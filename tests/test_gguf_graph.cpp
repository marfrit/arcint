// The template pass of a GGUF open (src/exec/gguf_graph.cpp), on a toy
// template built in code with the exporter's own naming and decompression
// pattern, against the repository's fixture GGUF. Needs OpenVINO's core
// library (the graph API) and no device: it checks what the pass builds --
// which op replaced which MatMul, that the K-quant constants alias the file's
// bytes instead of copying them, that the value-head inverse is the gather it
// should be on both sides, that the AWQ multiplier is set to one and the
// norm comparison sees the identical norm -- not what the kernel computes,
// which the plugin's own unit tests and the served Prüfstand cover.
#ifdef ARCINT_OPENVINO

#include "exec/gguf_graph.h"
#include "exec/graph_rewrites.h"
#include "exec/kquant_op.h"
#include "harness.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <openvino/core/model.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/subtract.hpp>

#include <cstdlib>
#include <memory>
#include <string>

using namespace lgc;

namespace {

std::string fixture() {
    if (const char* p = std::getenv("ARCINT_TEST_FIXTURES")) return std::string(p) + "/qwen35-tiny.gguf";
    return std::string(ARCINT_SOURCE_DIR) + "/tests/fixtures/qwen35-tiny.gguf";
}

// One projection in the exporter's form: Const u4 [N, K/64, 64] -> Convert ->
// Subtract(zp) -> Multiply(scale) -> Reshape [N, K] -> Convert -> MatMul(x, ., transpose_b).
ov::Output<ov::Node> projection(const ov::Output<ov::Node>& x, const std::string& module, size_t n, size_t k) {
    const std::string base = "self.model.model.language_model.layers.0." + module + "._openvino_orig_weight";
    auto w = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{n, k / 64, 64}, std::vector<uint8_t>(n * k, 2));  // one value per element; create() packs
    w->set_friendly_name(base);
    auto zp = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{n, k / 64, 1}, std::vector<uint8_t>(n * k / 64, 8));
    zp->set_friendly_name(base + "/zero_point");
    auto sc = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{n, k / 64, 1}, std::vector<ov::float16>(n * k / 64, ov::float16(0.01f)));
    sc->set_friendly_name(base + "/scale");
    auto cw = std::make_shared<ov::op::v0::Convert>(w, ov::element::f16);
    auto cz = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);
    auto sub = std::make_shared<ov::op::v1::Subtract>(cw, cz);
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, sc);
    auto rs = std::make_shared<ov::op::v1::Reshape>(mul, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {int64_t(n), int64_t(k)}), false);
    auto cv = std::make_shared<ov::op::v0::Convert>(rs, ov::element::f32);
    auto mm = std::make_shared<ov::op::v0::MatMul>(x, cv, false, true);
    mm->set_friendly_name("__module.model.model.language_model.layers.0." + module + "/ov_ext::linear/MatMul");
    return mm->output(0);
}

struct Toy {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v0::Constant> awq;
    std::shared_ptr<ov::op::v0::Constant> norm;
};

Toy toy_template() {
    // Four projections matching the fixture's blk.0 tensors, an AWQ multiplier
    // on the output projection's input, an input_layernorm multiply.
    auto x512 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 512});
    auto x256 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 256});
    std::vector<float> nv(512);
    for (size_t i = 0; i < 512; ++i) nv[i] = 1.0f + 0.01f * static_cast<float>(i);
    auto norm = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 512}, nv);
    auto normed = std::make_shared<ov::op::v1::Multiply>(x512, norm);
    normed->set_friendly_name("__module.model.model.language_model.layers.0.input_layernorm/aten::mul");
    auto gate = projection(normed, "mlp.gate_proj", 64, 512);
    auto awq = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 256}, std::vector<float>(256, 3.0f));
    awq->set_friendly_name("__module.model.model.language_model.layers.0.self_attn/aten::sigmoid/Sigmoid/awq_mul/scale");
    auto scaled = std::make_shared<ov::op::v1::Multiply>(x256, awq);
    auto out = projection(scaled, "linear_attn.out_proj", 32, 256);
    auto z = projection(x256, "linear_attn.in_proj_z", 256, 256);
    auto qkv = projection(x256, "linear_attn.in_proj_qkv", 512, 256);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gate), std::make_shared<ov::op::v0::Result>(out),
                             std::make_shared<ov::op::v0::Result>(z), std::make_shared<ov::op::v0::Result>(qkv)};
    return {std::make_shared<ov::Model>(results, ov::ParameterVector{x512, x256}, "toy"), awq, norm};
}

GgufGeometry toy_geometry() {
    GgufGeometry g;
    g.linear_k_heads = 2; g.linear_v_heads = 4; g.linear_k_dim = 64; g.linear_v_dim = 64;
    return g;
}

template <class T>
std::vector<std::shared_ptr<T>> ops_of(const std::shared_ptr<ov::Model>& m) {
    std::vector<std::shared_ptr<T>> out;
    for (const auto& op : m->get_ops())
        if (auto t = std::dynamic_pointer_cast<T>(op)) out.push_back(t);
    return out;
}

std::vector<int64_t> gather_indices(const std::shared_ptr<ov::op::v8::Gather>& g) {
    auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(g->input_value(1).get_node_shared_ptr());
    CHECK(c != nullptr);
    return c ? c->cast_vector<int64_t>() : std::vector<int64_t>{};
}

}  // namespace

TEST(gguf_pass_replaces_the_exporters_chains_with_kquant_ops_that_alias_the_file) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    Toy toy = toy_template();
    CHECK_EQ(ops_of<ov::op::v0::MatMul>(toy.model).size(), static_cast<size_t>(4));

    // The file's own rows for every type (--gguf-q6k file): this case is about aliasing.
    const GgufApplyReport rep = gguf_apply_to_template(toy.model, file, toy_geometry(), GgufWeightsMode::Native, "", "", false);
    CHECK_EQ(rep.replaced.size(), static_cast<size_t>(4));
    CHECK_EQ(ops_of<ov::op::v0::MatMul>(toy.model).size(), static_cast<size_t>(0));
    const auto kq = ops_of<FullyConnectedKQuant>(toy.model);
    CHECK_EQ(kq.size(), static_cast<size_t>(4));
    size_t bytes = 0;
    for (const auto& op : kq) {
        // The constant is the file's own bytes, not a copy.
        auto w = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
        CHECK(w != nullptr);
        if (!w) continue;
        const std::string irname = w->get_friendly_name();  // "<ir constant>/gguf"
        std::string gguf_name;
        for (const auto& r : rep.replaced) if (irname == r.ir_name + "/gguf") gguf_name = r.gguf_name;
        const auto* t = file->tensor(gguf_name);
        CHECK(t != nullptr);
        if (!t) continue;
        CHECK(w->get_data_ptr() == static_cast<const void*>(file->data(*t)));
        CHECK_EQ(w->get_byte_size(), file->bytes(*t));
        CHECK_EQ(op->get_rt_info().at("arcint_kquant_type").as<int64_t>(), static_cast<int64_t>(t->ggml_type));
        CHECK_EQ(op->get_rt_info().at("arcint_kquant_k").as<int64_t>(), static_cast<int64_t>(t->dims[0]));
        bytes += file->bytes(*t);
    }
    CHECK_EQ(rep.bytes_from_file, bytes);
}

TEST(gguf_pass_inverts_the_value_head_reorder_as_gathers_on_the_right_side) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    Toy toy = toy_template();
    (void)gguf_apply_to_template(toy.model, file, toy_geometry(), GgufWeightsMode::Native);
    // to_file for 2 key heads, 4 value heads: HF head h = i*2 + j -> file head j*2 + i:
    // 0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3.
    const auto to_file = gguf_v_head_to_file_head(2, 4);
    CHECK_EQ(to_file[1], 2);
    CHECK_EQ(to_file[2], 1);
    size_t input_side = 0, output_side = 0;
    for (const auto& g : ops_of<ov::op::v8::Gather>(toy.model)) {
        const auto idx = gather_indices(g);
        const bool feeds_op = std::dynamic_pointer_cast<FullyConnectedKQuant>(g->output(0).get_target_inputs().begin()->get_node()->shared_from_this()) != nullptr;
        if (feeds_op) {
            // out_proj: x_file[to_file[h]*64 + e] = x_hf[h*64 + e]
            input_side++;
            CHECK_EQ(idx.size(), static_cast<size_t>(256));
            CHECK_EQ(idx[to_file[1] * 64 + 5], 1 * 64 + 5);
            CHECK_EQ(idx[to_file[3] * 64 + 0], 3 * 64 + 0);
        } else {
            // in_proj_z (256 rows) or in_proj_qkv (512 rows, the first 256 identity):
            // y_hf[first + h*64 + e] = y_file[first + to_file[h]*64 + e]
            output_side++;
            const int64_t first = idx.size() == 512 ? 256 : 0;
            for (int64_t i = 0; i < first; ++i) CHECK_EQ(idx[i], i);
            CHECK_EQ(idx[first + 1 * 64 + 7], first + to_file[1] * 64 + 7);
            CHECK_EQ(idx[first + 2 * 64 + 0], first + to_file[2] * 64 + 0);
        }
    }
    CHECK_EQ(input_side, static_cast<size_t>(1));
    CHECK_EQ(output_side, static_cast<size_t>(2));
}

TEST(gguf_pass_neutralises_awq_multipliers_and_compares_norms) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    Toy toy = toy_template();
    const GgufApplyReport rep = gguf_apply_to_template(toy.model, file, toy_geometry(), GgufWeightsMode::Native);
    CHECK_EQ(rep.awq_scales_neutralized, static_cast<size_t>(1));
    bool ones = false;
    for (const auto& c : ops_of<ov::op::v0::Constant>(toy.model)) {
        if (c->get_friendly_name().find("awq_mul/scale") == std::string::npos) continue;
        const auto v = c->cast_vector<float>();
        ones = !v.empty();
        for (float f : v) ones = ones && f == 1.0f;
    }
    CHECK(ones);
    CHECK_EQ(rep.norms_compared, static_cast<size_t>(1));
    CHECK(rep.norms_max_abs_diff == 0.0);
}

TEST(gguf_pass_refuses_a_template_whose_module_the_file_lacks) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 512});
    auto y = projection(x, "mlp.up_proj", 64, 512);  // no blk.0.ffn_up in the fixture
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(y)}, ov::ParameterVector{x});
    bool threw = false;
    try { (void)gguf_apply_to_template(model, file, toy_geometry(), GgufWeightsMode::Native); }
    catch (const std::runtime_error& e) { threw = std::string(e.what()).find("ffn_up") != std::string::npos; }
    CHECK(threw);
}

// 0.4.1 lever 2: the default mode repacks every projection into the plugin's
// own decompression chain (Const -> Convert -> [Subtract] -> Multiply ->
// Reshape -> MatMul), leaves no K-quant op in the graph, and reports a
// deviation under the per-type bound over every value it packed.
TEST(gguf_pass_repacks_into_the_plugins_decompression_chain_by_default) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    Toy toy = toy_template();
    const GgufApplyReport rep = gguf_apply_to_template(toy.model, file, toy_geometry());
    CHECK(rep.mode == GgufWeightsMode::Repack);
    CHECK_EQ(rep.replaced.size(), size_t{4});
    CHECK_EQ(ops_of<FullyConnectedKQuant>(toy.model).size(), size_t{0});
    CHECK_EQ(rep.repack_over_bound, size_t{0});
    CHECK(rep.repack_max_steps > 0.0 && rep.repack_max_steps < 1.0 / 16.0);
    size_t values = 0;
    for (const auto& r : rep.replaced) values += static_cast<size_t>(r.n * r.k);
    CHECK_EQ(rep.repack_checked, values);
    // The value-head reorder lives in the constants: no gather in the graph, the report says permuted.
    size_t gathers = 0;
    for (const auto& g : ops_of<ov::op::v8::Gather>(toy.model))
        if (g->get_friendly_name().find("/gguf_v_head_order") != std::string::npos) ++gathers;
    CHECK_EQ(gathers, size_t{0});
    size_t permuted = 0;
    for (const auto& r : rep.replaced) { if (r.rows_permuted) ++permuted; CHECK(!r.columns_gathered); }
    CHECK_EQ(permuted, size_t{3});  // in_proj_qkv, in_proj_z (rows) and out_proj (columns)
    // The Q4_K projections read a widened activation (their mins ride as columns): a Concat per distinct activation.
    size_t widened = 0;
    for (const auto& c : ops_of<ov::op::v0::Concat>(toy.model))
        if (c->get_friendly_name().find("/gguf_widened") != std::string::npos) ++widened;
    CHECK(widened >= 1);
    // Every MatMul's weight input is the chain, ending in an integer constant of the repacked type.
    size_t chains = 0;
    for (const auto& mm : ops_of<ov::op::v0::MatMul>(toy.model)) {
        auto node = mm->input_value(1).get_node_shared_ptr();
        if (std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) node = node->input_value(0).get_node_shared_ptr();  // f32 template MatMul
        auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(node);
        if (!reshape) continue;
        auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(reshape->input_value(0).get_node_shared_ptr());
        CHECK(mul != nullptr);
        if (!mul) continue;
        auto sc = std::dynamic_pointer_cast<ov::op::v0::Constant>(mul->input_value(1).get_node_shared_ptr());
        CHECK(sc && sc->get_element_type() == ov::element::f16 && sc->get_shape().size() == 3 && sc->get_shape()[2] == 1);
        auto x = mul->input_value(0).get_node_shared_ptr();
        if (auto sub = std::dynamic_pointer_cast<ov::op::v1::Subtract>(x)) x = sub->input_value(0).get_node_shared_ptr();
        auto conv = std::dynamic_pointer_cast<ov::op::v0::Convert>(x);
        CHECK(conv != nullptr);
        if (!conv) continue;
        auto w = std::dynamic_pointer_cast<ov::op::v0::Constant>(conv->input_value(0).get_node_shared_ptr());
        CHECK(w != nullptr);
        if (!w) continue;
        const auto et = w->get_element_type();
        CHECK(et == ov::element::u4 || et == ov::element::u8 || et == ov::element::i8);
        CHECK_EQ(w->get_shape().size(), size_t{3});
        ++chains;
    }
    CHECK_EQ(chains, size_t{4});
}

TEST(gguf_pass_mixed_mode_repacks_q4k_and_keeps_the_other_types_native) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    Toy toy = toy_template();
    const GgufApplyReport rep = gguf_apply_to_template(toy.model, file, toy_geometry(), GgufWeightsMode::Mixed);
    CHECK(rep.mode == GgufWeightsMode::Mixed);
    CHECK_EQ(rep.replaced.size(), size_t{4});
    size_t q4 = 0, other = 0;
    for (const auto& r : rep.replaced) {
        const bool is_q4 = r.ggml_type == static_cast<int32_t>(gguf::GgmlType::Q4_K);
        CHECK_EQ(r.repacked, is_q4);
        if (is_q4) ++q4; else ++other;
    }
    CHECK(q4 >= 1);
    CHECK(other >= 1);
    // The native ones are K-quant ops in the graph, the repacked ones are not.
    CHECK_EQ(ops_of<FullyConnectedKQuant>(toy.model).size(), other);
    CHECK_EQ(rep.repack_over_bound, size_t{0});
    CHECK(rep.summary().find("mixed") != std::string::npos);
    CHECK(gguf_tensor_mode(GgufWeightsMode::Mixed, static_cast<int32_t>(gguf::GgmlType::Q6_K)) == GgufWeightsMode::Native);
    CHECK(gguf_tensor_mode(GgufWeightsMode::Repack, static_cast<int32_t>(gguf::GgmlType::Q6_K)) == GgufWeightsMode::Repack);
}

TEST(gguf_pass_keeps_a_passed_deviation_verdict_between_loads_of_the_same_file) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    char tmpl[] = "/tmp/arcint-verdicts-XXXXXX";
    const char* dir = ::mkdtemp(tmpl);
    CHECK(dir != nullptr);
    const std::string vdir = std::string(dir) + "/gguf-verdicts";
    Toy first = toy_template();
    const GgufApplyReport a = gguf_apply_to_template(first.model, file, toy_geometry(), GgufWeightsMode::Repack, vdir, fixture());
    CHECK_EQ(a.repack_verdicts_cached, size_t{0});
    CHECK(a.repack_checked > 0);
    CHECK(a.check_seconds >= 0.0 && a.repack_seconds >= 0.0);
    // Second load of the same file: every projection's verdict is read back, nothing is re-checked.
    Toy second = toy_template();
    const GgufApplyReport b = gguf_apply_to_template(second.model, file, toy_geometry(), GgufWeightsMode::Repack, vdir, fixture());
    CHECK_EQ(b.repack_verdicts_cached, a.replaced.size());
    CHECK_EQ(b.repack_checked, size_t{0});
    CHECK_NEAR(b.repack_max_steps, a.repack_max_steps, 1e-9);
    // Without a directory (--gguf-check always) the check runs every time.
    Toy third = toy_template();
    const GgufApplyReport c = gguf_apply_to_template(third.model, file, toy_geometry(), GgufWeightsMode::Repack, "", fixture());
    CHECK_EQ(c.repack_verdicts_cached, size_t{0});
    CHECK_EQ(c.repack_checked, a.repack_checked);
    std::filesystem::remove_all(dir);
}

// The mins' packing is an option of the repack (--gguf-mins): the inexact forms
// halve or quarter the augmentation and are reported, not refused.
TEST(gguf_pass_repacks_with_an_inexact_mins_packing_on_request_and_reports_its_deviation) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    Toy exact = toy_template();
    const GgufApplyReport a = gguf_apply_to_template(exact.model, file, toy_geometry(), GgufWeightsMode::Mixed, "", fixture(), true, gguf::RepackMins::Exact);
    Toy shared = toy_template();
    const GgufApplyReport b = gguf_apply_to_template(shared.model, file, toy_geometry(), GgufWeightsMode::Mixed, "", fixture(), true, gguf::RepackMins::Shared);
    CHECK(b.mins == gguf::RepackMins::Shared);
    CHECK(b.repack_max_steps >= a.repack_max_steps);
    CHECK(b.summary().find("mins shared") != std::string::npos);
    size_t wa = 0, wb = 0;
    for (const auto& r : a.replaced) if (r.repacked) wa += r.bytes;
    for (const auto& r : b.replaced) if (r.repacked) wb += r.bytes;
    CHECK(wb <= wa);  // fewer augmented columns at the served widths; at the fixture's K the even-count padding takes the saving back
}

// RepackMins::Split (0.4.2, DESIGN §7.0.2bo; §7.0.2bg's `--dyn-quant on` finding): the
// min term is a second, unquantised MatMul added to the main one instead of
// augmented columns of it, so the widened activation's group-sum columns
// never reach the runtime's per-token int8 activation quantization. No
// Concat, no widened activation for the projection, one MatMul over the
// file's main columns alone (width K, not K + K/8) plus one small MatMul.
TEST(gguf_pass_split_mins_adds_the_min_term_as_a_separate_matmul_with_no_augmented_columns) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    Toy toy = toy_template();
    const GgufApplyReport rep = gguf_apply_to_template(toy.model, file, toy_geometry(), GgufWeightsMode::Repack, "", fixture(), true, gguf::RepackMins::Split);
    CHECK(rep.mins == gguf::RepackMins::Split);
    // mlp.gate_proj (blk.0.ffn_gate.weight, Q4_K, N=64, K=512) has a min: Split's path.
    const GgufReplacement* gate = nullptr;
    for (const auto& r : rep.replaced) if (r.gguf_name == "blk.0.ffn_gate.weight") gate = &r;
    CHECK(gate != nullptr);
    if (!gate) return;
    CHECK(gate->repacked);
    CHECK_EQ(gate->k, int64_t{512});
    // No Concat / no widened activation anywhere in the graph under Split.
    for (const auto& c : ops_of<ov::op::v0::Concat>(toy.model))
        CHECK(c->get_friendly_name().find("/gguf_widened") == std::string::npos);
    // toy's first Result is mlp.gate_proj's: its replacement is the Add.
    auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(toy.model->get_results()[0]->input_value(0).get_node_shared_ptr());
    CHECK(add != nullptr);
    if (!add) return;
    auto main_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(add->input_value(0).get_node_shared_ptr());
    auto min_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(add->input_value(1).get_node_shared_ptr());
    CHECK(main_mm != nullptr);
    CHECK(min_mm != nullptr);
    if (!main_mm || !min_mm) return;
    // The main MatMul's weight chain ends in a u4 constant of N*K values (not N*(K+K/8)).
    auto node = main_mm->input_value(1).get_node_shared_ptr();
    if (auto outer = std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) node = outer->input_value(0).get_node_shared_ptr();  // toy's own f32 MatMul
    auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(node);
    CHECK(reshape != nullptr);
    if (!reshape) return;
    auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(reshape->input_value(0).get_node_shared_ptr());
    CHECK(mul != nullptr);
    if (!mul) return;
    auto x = mul->input_value(0).get_node_shared_ptr();
    if (auto sub = std::dynamic_pointer_cast<ov::op::v1::Subtract>(x)) x = sub->input_value(0).get_node_shared_ptr();  // Q5_K/Q6_K only
    auto conv = std::dynamic_pointer_cast<ov::op::v0::Convert>(x);
    CHECK(conv != nullptr);
    if (!conv) return;
    auto w = std::dynamic_pointer_cast<ov::op::v0::Constant>(conv->input_value(0).get_node_shared_ptr());
    CHECK(w != nullptr);
    if (!w) return;
    CHECK(w->get_element_type() == ov::element::u4);
    size_t values = 1;
    for (auto d : w->get_shape()) values *= d;
    CHECK_EQ(values, size_t{64 * 512});  // N * K exactly: no augmented columns
    // The min term's own constant: f16, [K/32, N] = [16, 64], named .../gguf_min_term.
    auto min_node = min_mm->input_value(1).get_node_shared_ptr();
    if (auto outer = std::dynamic_pointer_cast<ov::op::v0::Convert>(min_node)) min_node = outer->input_value(0).get_node_shared_ptr();  // toy's f32 activation
    auto min_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(min_node);
    CHECK(min_const != nullptr);
    if (!min_const) return;
    CHECK(min_const->get_friendly_name().find("/gguf_min_term") != std::string::npos);
    CHECK(min_const->get_element_type() == ov::element::f16);
    CHECK_EQ(min_const->get_shape().size(), size_t{2});
    CHECK_EQ(min_const->get_shape()[0], size_t{512 / 32});
    CHECK_EQ(min_const->get_shape()[1], size_t{64});
}

// The native Q6_K rows are laid out in 224-byte blocks by default (kquant type 114: the file's
// 210 bytes then 14 zero bytes per super-block, every block dword-aligned, DESIGN §7.0.2bj);
// --gguf-q6k file keeps the file's rows (type 14). The fixture's ffn_down is Q6_K with one
// block per row, so consecutive rows alternate the block parity the layout removes.
TEST(gguf_pass_lays_q6k_rows_out_in_224_byte_blocks_by_default_and_keeps_the_files_rows_on_request) {
    auto file = std::make_shared<gguf::GgufFile>(gguf::GgufFile::open(fixture()));
    const auto* t = file->tensor("blk.0.ffn_down.weight");
    CHECK(t != nullptr);
    if (!t) return;
    CHECK_EQ(t->ggml_type, static_cast<int32_t>(gguf::GgmlType::Q6_K));
    const uint8_t* src = file->data(*t);
    for (int aligned = 1; aligned >= 0; --aligned) {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 256});
        auto y = projection(x, "mlp.down_proj", 512, 256);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(y)}, ov::ParameterVector{x}, "down");
        const GgufApplyReport rep = gguf_apply_to_template(model, file, toy_geometry(), GgufWeightsMode::Native, "", fixture(), aligned != 0);
        CHECK_EQ(rep.q6k_aligned, aligned);
        const auto ops = ops_of<FullyConnectedKQuant>(model);
        CHECK_EQ(ops.size(), size_t{1});
        if (ops.empty()) continue;
        CHECK_EQ(ops[0]->kquant_type(), static_cast<int64_t>(aligned ? 114 : 14));
        auto w = std::dynamic_pointer_cast<ov::op::v0::Constant>(ops[0]->input_value(1).get_node_shared_ptr());
        CHECK(w != nullptr);
        if (!w) continue;
        const size_t stride = aligned ? 224 : 210;
        CHECK_EQ(w->get_shape()[0], size_t{512});
        CHECK_EQ(w->get_shape()[1], stride);
        const uint8_t* bytes = w->get_data_ptr<uint8_t>();
        for (size_t row : {size_t{0}, size_t{3}, size_t{511}}) {
            CHECK_EQ(std::memcmp(bytes + row * stride, src + row * 210, 210), 0);
            if (aligned) for (size_t i = 210; i < 224; ++i) CHECK_EQ(static_cast<int>(bytes[row * 224 + i]), 0);
        }
    }
}

// The logits slice (DESIGN §7.0.2e) walks from the first Result to the LM
// head and cuts its input to the last row. On a GGUF-opened model whose
// output.weight stays in the file's rows (mixed, native), the head is a
// FullyConnectedKQuant, not a MatMul: the walk must accept it, or every prefill
// chunk computes and copies [M, vocab] logits (an OpenCL timeline caught the
// 850 MB copy per 856-token prefill and the head's tiled kernel over every
// row, 2026-09-07).
TEST(gguf_pass_slices_the_logits_at_a_kquant_lm_head) {
    // A MatMul head, the case that always worked: the control.
    {
        Toy toy = toy_template();
        CHECK(slice_logits_to_last_token(toy.model, 1, 0));
        const auto head = toy.model->get_results()[0]->input_value(0).get_node_shared_ptr();
        CHECK_EQ(std::string(head->input_value(0).get_node()->get_type_name()), std::string("Slice"));
    }
    // A K-quant head in the paged layout [tokens, 1, hidden]: Q8_0, K = 256, N = 16.
    {
        const int64_t k = 256, n = 16;
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, k});
        auto w = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{size_t(n), size_t(FullyConnectedKQuant::row_bytes(8, k))},
                                              std::vector<uint8_t>(size_t(n) * size_t(FullyConnectedKQuant::row_bytes(8, k)), 0));
        auto head = std::make_shared<FullyConnectedKQuant>(x, w, 8, k, n);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(head)}, ov::ParameterVector{x}, "kq_head");
        CHECK(slice_logits_to_last_token(model, 1, 0));
        CHECK_EQ(std::string(head->input_value(0).get_node()->get_type_name()), std::string("Slice"));
        CHECK_EQ(head->input_value(1).get_node_shared_ptr().get(), static_cast<ov::Node*>(w.get()));  // the weights untouched
        CHECK_EQ(model->output(0).get_partial_shape()[2].get_length(), n);
    }
}

// expose_hidden_state (backend_ov.cpp, primes the MTP head, DESIGN's MTP
// section) walks the same MatMul-or-FullyConnectedKQuant terminator as
// slice_logits_to_last_token above. Before this fix the walk knew only
// MatMul: a GGUF export whose output.weight stays in the file's rows always
// has FullyConnectedKQuant there instead (exec/kquant_op.h), so --mtp on
// over such an export logged "could not expose the hidden state; MTP
// disabled" and drafted nothing, ever -- measured on the dense template
// (--gguf, a Q4_K_M file, --paged-kv u8). The three cases
// below are the two terminators the fix accepts and one node that is
// neither, which must still fail closed.
TEST(gguf_pass_exposes_the_hidden_state_at_a_kquant_or_matmul_head) {
    // A MatMul head, the case that always worked: the control.
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8});
        auto w = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 8}, std::vector<float>(32, 1.0f));
        auto head = std::make_shared<ov::op::v0::MatMul>(x, w, false, true);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(head)}, ov::ParameterVector{x}, "mm_head");
        CHECK(expose_hidden_state(model));
        CHECK_EQ(model->get_results().size(), size_t{2});
        if (model->get_results().size() != 2) return;
        CHECK_EQ(model->get_results()[1]->input_value(0).get_node_shared_ptr().get(), static_cast<ov::Node*>(x.get()));
        CHECK_EQ(model->get_results()[1]->get_output_tensor(0).get_names().count("hidden_states"), size_t{1});
    }
    // A K-quant head, the case that used to fail closed at hop 0: Q8_0, K = 256, N = 16.
    {
        const int64_t k = 256, n = 16;
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, k});
        auto w = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{size_t(n), size_t(FullyConnectedKQuant::row_bytes(8, k))},
                                              std::vector<uint8_t>(size_t(n) * size_t(FullyConnectedKQuant::row_bytes(8, k)), 0));
        auto head = std::make_shared<FullyConnectedKQuant>(x, w, 8, k, n);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(head)}, ov::ParameterVector{x}, "kq_head");
        CHECK(expose_hidden_state(model));
        CHECK_EQ(model->get_results().size(), size_t{2});
        if (model->get_results().size() != 2) return;
        CHECK_EQ(model->get_results()[1]->input_value(0).get_node_shared_ptr().get(), static_cast<ov::Node*>(x.get()));
        CHECK_EQ(model->get_results()[1]->get_output_tensor(0).get_names().count("hidden_states"), size_t{1});
    }
    // Neither: an Add straight into the Result. Must fail closed and leave the model untouched.
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8});
        auto c = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8}, std::vector<float>(8, 1.0f));
        auto head = std::make_shared<ov::op::v1::Add>(x, c);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(head)}, ov::ParameterVector{x}, "neither");
        CHECK(!expose_hidden_state(model));
        CHECK_EQ(model->get_results().size(), size_t{1});
    }
}

#endif  // ARCINT_OPENVINO
