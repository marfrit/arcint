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
#include "exec/kquant_op.h"
#include "harness.h"

#include <openvino/core/model.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
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

    const GgufApplyReport rep = gguf_apply_to_template(toy.model, file, toy_geometry());
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
    (void)gguf_apply_to_template(toy.model, file, toy_geometry());
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
    const GgufApplyReport rep = gguf_apply_to_template(toy.model, file, toy_geometry());
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
    try { (void)gguf_apply_to_template(model, file, toy_geometry()); }
    catch (const std::runtime_error& e) { threw = std::string(e.what()).find("ffn_up") != std::string::npos; }
    CHECK(threw);
}

#endif  // ARCINT_OPENVINO
