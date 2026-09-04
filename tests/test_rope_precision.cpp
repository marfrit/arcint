// M11 §2 RoPE follow-up (DESIGN §7.0.2ag): a unit test on a tiny synthetic
// ov::Model, exercising the exact production helper (exec/rope_precision.h)
// backend_ov.cpp calls on the real drafter graphs -- not a reimplementation
// of its logic. Needs the OpenVINO headers/opset to build a graph at all, so
// this whole file is gated behind ARCINT_OPENVINO and lives in the
// OpenVINO-enabled `arcint-test` target (tests/CMakeLists is unconditional;
// the guard below makes the file empty, and empty is fine under C++20, in a
// plain stub build).
#ifdef ARCINT_OPENVINO

#include "exec/backend.h"
#include "exec/rope_precision.h"
#include "harness.h"

#include <openvino/openvino.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/unsqueeze.hpp>

#include <algorithm>

using namespace lgc;

namespace {

// Builds the pattern the design names verbatim, plus enough either side of
// it to make "marks exactly those nodes and nothing else" a real assertion
// rather than a vacuous one:
//
//   positions (i64 Parameter) -> Convert(f32) -> Unsqueeze -> Multiply(inv_freq)
//   angle_multiply -> Cos -> Multiply(with data Parameter q) --> Add -> Result
//   angle_multiply -> Sin -> Multiply(with data Parameter k) -/
//
// plus a DECOY Multiply fed by an unrelated Parameter and an unrelated
// Constant, wired to a second Result -- structurally disconnected from the
// positions Parameter, so it must never be marked.
struct RopeFixture {
    std::shared_ptr<ov::Model>          model;
    std::shared_ptr<ov::Node>           convert;
    std::shared_ptr<ov::Node>           unsqueeze;
    std::shared_ptr<ov::Node>           angle_multiply;
    std::shared_ptr<ov::Node>           cos_node;
    std::shared_ptr<ov::Node>           sin_node;
    std::shared_ptr<ov::Node>           cos_multiply;
    std::shared_ptr<ov::Node>           sin_multiply;
    std::shared_ptr<ov::Node>           add_node;
    std::shared_ptr<ov::Node>           decoy_multiply;
};

RopeFixture build_fixture() {
    using ov::element::f32;
    using ov::element::i64;

    RopeFixture f;

    auto positions = std::make_shared<ov::op::v0::Parameter>(i64, ov::PartialShape{1, 4});
    positions->set_friendly_name("positions");
    // q/k: [1, seq=4, head_dim=4] -- head_dim matches the angle tensor's
    // trailing dim (4, from inv_freq below) so cos/sin ([1,1,4], after the
    // Unsqueeze inserts a size-1 seq axis) numpy-broadcasts against them
    // cleanly instead of failing shape inference.
    auto q = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, 4, 4});
    q->set_friendly_name("q");
    auto k = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, 4, 4});
    k->set_friendly_name("k");
    // Decoy inputs: not reachable from `positions` at all -- shape is
    // whatever, it only has to match itself.
    auto decoy_param = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, 4, 8});
    decoy_param->set_friendly_name("decoy_param");

    f.convert = std::make_shared<ov::op::v0::Convert>(positions, f32);
    f.convert->set_friendly_name("convert");

    auto axes = ov::op::v0::Constant::create(i64, ov::Shape{1}, {0});
    f.unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(f.convert, axes);
    f.unsqueeze->set_friendly_name("unsqueeze");

    auto inv_freq = ov::op::v0::Constant::create(f32, ov::Shape{1, 4}, std::vector<float>(4, 1.0f));
    f.angle_multiply = std::make_shared<ov::op::v1::Multiply>(f.unsqueeze, inv_freq);
    f.angle_multiply->set_friendly_name("angle_multiply");

    f.cos_node = std::make_shared<ov::op::v0::Cos>(f.angle_multiply);
    f.cos_node->set_friendly_name("cos");
    f.sin_node = std::make_shared<ov::op::v0::Sin>(f.angle_multiply);
    f.sin_node->set_friendly_name("sin");

    f.cos_multiply = std::make_shared<ov::op::v1::Multiply>(f.cos_node, q);
    f.cos_multiply->set_friendly_name("cos_multiply");
    f.sin_multiply = std::make_shared<ov::op::v1::Multiply>(f.sin_node, k);
    f.sin_multiply->set_friendly_name("sin_multiply");

    f.add_node = std::make_shared<ov::op::v1::Add>(f.cos_multiply, f.sin_multiply);
    f.add_node->set_friendly_name("add");

    auto decoy_const = ov::op::v0::Constant::create(f32, ov::Shape{1, 4, 8}, std::vector<float>(32, 2.0f));
    f.decoy_multiply = std::make_shared<ov::op::v1::Multiply>(decoy_param, decoy_const);
    f.decoy_multiply->set_friendly_name("decoy_multiply");

    auto result       = std::make_shared<ov::op::v0::Result>(f.add_node);
    auto decoy_result = std::make_shared<ov::op::v0::Result>(f.decoy_multiply);

    f.model = std::make_shared<ov::Model>(ov::ResultVector{result, decoy_result},
                                          ov::ParameterVector{positions, q, k, decoy_param});
    return f;
}

bool contains(const std::vector<std::shared_ptr<ov::Node>>& v, const std::shared_ptr<ov::Node>& n) {
    return std::find(v.begin(), v.end(), n) != v.end();
}

}  // namespace

TEST(find_rope_nodes_marks_exactly_the_rotary_subgraph) {
    const RopeFixture f    = build_fixture();
    const auto        found = find_rope_nodes(f.model);

    CHECK_EQ(found.size(), static_cast<size_t>(8));
    for (const auto& n : {f.convert, f.unsqueeze, f.angle_multiply, f.cos_node, f.sin_node,
                          f.cos_multiply, f.sin_multiply, f.add_node}) {
        CHECK(contains(found, n));
    }
    CHECK(!contains(found, f.decoy_multiply));
}

TEST(mark_rope_no_fp16_compression_disables_conversion_on_the_found_nodes_only) {
    const RopeFixture f       = build_fixture();
    const size_t       marked = mark_rope_no_fp16_compression(f.model);
    CHECK_EQ(marked, static_cast<size_t>(8));

    for (const auto& n : {f.convert, f.unsqueeze, f.angle_multiply, f.cos_node, f.sin_node,
                          f.cos_multiply, f.sin_multiply, f.add_node}) {
        CHECK(ov::is_conversion_disabled(n, ov::element::f16));
    }
    CHECK(!ov::is_conversion_disabled(f.decoy_multiply, ov::element::f16));
}

// The env-switch half of the red-first requirement: mirrors exactly what
// backend_ov.cpp's own `apply_rope_fix = !draft_rope_f16_enabled()` gate
// does at the call site -- ARCINT_DRAFT_ROPE_F16 set means the marker is
// never called at all, so nothing ends up marked.
TEST(draft_rope_f16_switch_leaves_the_rotary_subgraph_unmarked) {
    setenv("ARCINT_DRAFT_ROPE_F16", "1", 1);
    const bool apply_rope_fix = !draft_rope_f16_enabled();
    unsetenv("ARCINT_DRAFT_ROPE_F16");
    CHECK(!apply_rope_fix);

    const RopeFixture f = build_fixture();
    size_t             marked = 0;
    if (apply_rope_fix) marked = mark_rope_no_fp16_compression(f.model);  // not reached
    CHECK_EQ(marked, static_cast<size_t>(0));

    for (const auto& n : {f.convert, f.unsqueeze, f.angle_multiply, f.cos_node, f.sin_node,
                          f.cos_multiply, f.sin_multiply, f.add_node}) {
        CHECK(!ov::is_conversion_disabled(n, ov::element::f16));
    }
}

// M11 §2, the DFlash shape (measured on the real served DFlash IR,
// /models/gptq/qwen38-dflash2-int4/openvino_dflash_draft_stateful.xml):
// the angle Multiply feeds a Concat (duplicating the half-width frequency
// table to the full head_dim before Cos/Sin) rather than feeding Cos/Sin
// directly -- the MTP layer fixture above never exercises that hop.
// Confirms find_rope_nodes' candidate-confirmation walk sees THROUGH a
// Concat, not just Convert/Unsqueeze in front of the angle multiply.
TEST(find_rope_nodes_sees_through_a_concat_between_the_angle_multiply_and_trig) {
    using ov::element::f32;
    using ov::element::i64;

    auto positions = std::make_shared<ov::op::v0::Parameter>(i64, ov::PartialShape{1, 4});
    positions->set_friendly_name("positions");

    auto convert = std::make_shared<ov::op::v0::Convert>(positions, f32);
    convert->set_friendly_name("convert");
    auto axes = ov::op::v0::Constant::create(i64, ov::Shape{1}, {0});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(convert, axes);
    unsqueeze->set_friendly_name("unsqueeze");
    auto inv_freq = ov::op::v0::Constant::create(f32, ov::Shape{1, 4}, std::vector<float>(4, 1.0f));
    auto angle_multiply = std::make_shared<ov::op::v1::Multiply>(unsqueeze, inv_freq);
    angle_multiply->set_friendly_name("angle_multiply");

    // Concat(angle, angle) along the last axis -- the DFlash IR's own
    // "duplicate the half-width table to full width" step.
    auto concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{angle_multiply, angle_multiply}, 2);
    concat->set_friendly_name("concat");

    auto cos_node = std::make_shared<ov::op::v0::Cos>(concat);
    cos_node->set_friendly_name("cos");
    auto sin_node = std::make_shared<ov::op::v0::Sin>(concat);
    sin_node->set_friendly_name("sin");

    auto result_cos = std::make_shared<ov::op::v0::Result>(cos_node);
    auto result_sin = std::make_shared<ov::op::v0::Result>(sin_node);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result_cos, result_sin},
                                             ov::ParameterVector{positions});

    const auto found = find_rope_nodes(model);
    CHECK_EQ(found.size(), static_cast<size_t>(6));
    const std::vector<std::shared_ptr<ov::Node>> expect{convert,       unsqueeze, angle_multiply,
                                                         concat,        cos_node,  sin_node};
    for (const auto& n : expect) {
        CHECK(contains(found, n));
    }
}

#endif  // ARCINT_OPENVINO
