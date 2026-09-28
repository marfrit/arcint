// QSA (campaign qsa, step 3 T4): the runtime side of option A.
//
// The refusals are a pure decision (src/exec/qsa_runtime.h), so these cells
// run without a card. The geometry cell builds a tiny synthetic ov::Model with
// two indexer Variables and one KV Variable, so "counts the indexer state and
// nothing else" is a real assertion, not a vacuous one.
#ifdef ARCINT_OPENVINO

#include "core/artifact.h"
#include "exec/qsa_runtime.h"
#include "harness.h"

#include <openvino/openvino.hpp>
#include <openvino/op/assign.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/result.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace lgc;

namespace {

constexpr size_t kIndexerHeadDim = 128;   // Flash-Next's indexer_head_dim
constexpr size_t kFlashNextQsa   = 12;    // its full-attention (QSA) layers

// A model with two indexer Variables ([1, -1, 128] f32) and one KV Variable
// ([−1, kv, −1, 128]) whose id deliberately contains ".key." -- the loader's
// is_kv() predicate -- so a scan that classified by id substring would count
// it. The indexer's own id "cache_params.past.indexer_key.N" does not.
std::shared_ptr<ov::Model> build_mixed_state_model() {
    using ov::element::f32;
    using ov::op::util::Variable;
    using ov::op::util::VariableInfo;

    ov::SinkVector sinks;
    std::shared_ptr<ov::op::v6::ReadValue> first_read;

    auto add_state = [&](const std::string& id, const ov::PartialShape& shape,
                         const ov::Shape& init_shape) {
        auto info = VariableInfo{shape, f32, id};
        auto var  = std::make_shared<Variable>(info);
        auto init = std::make_shared<ov::op::v0::Constant>(f32, init_shape);
        auto rv   = std::make_shared<ov::op::v6::ReadValue>(init, var);
        auto as   = std::make_shared<ov::op::v6::Assign>(rv, var);
        sinks.push_back(as);
        if (!first_read) first_read = rv;
    };

    add_state("cache_params.past.indexer_key.0", ov::PartialShape{1, -1, kIndexerHeadDim},
              ov::Shape{1, 0, kIndexerHeadDim});
    add_state("cache_params.past.indexer_key.1", ov::PartialShape{1, -1, kIndexerHeadDim},
              ov::Shape{1, 0, kIndexerHeadDim});
    add_state("cache_params.past.0.key", ov::PartialShape{-1, 2, -1, kIndexerHeadDim},
              ov::Shape{0, 2, 0, kIndexerHeadDim});

    auto res   = std::make_shared<ov::op::v0::Result>(first_read);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{});
    model->add_sinks(sinks);
    return model;
}

}  // namespace

TEST(qsa_state_bytes_per_token_is_the_flash_next_geometry) {
    // 12 layers x 128 head_dim x 4 B = 6144 B = 6 KiB/token.
    CHECK_EQ(qsa::state_bytes_per_token(kFlashNextQsa, kIndexerHeadDim), 6144u);
    CHECK_EQ(qsa::state_bytes_per_token(0, kIndexerHeadDim), 0u);
}

TEST(qsa_state_geometry_counts_indexer_variables_and_not_kv) {
    const qsa::StateGeometry g = qsa::state_geometry(build_mixed_state_model());
    // The two indexer Variables only; the ".key." KV Variable (rank 4, dynamic
    // seq) must not be charged as indexer state.
    CHECK_EQ(g.n_layer, size_t{2});
    CHECK_EQ(g.bytes_per_token, 2u * kIndexerHeadDim * 4u);
}

TEST(qsa_option_a_allows_the_safe_config) {
    qsa::RuntimeLimits ok;
    ok.lanes        = 1;
    ok.prefix_cache = false;
    ok.speculative  = false;
    CHECK(!qsa::runtime_refusal(ok).has_value());
}

TEST(qsa_option_a_refuses_a_second_lane) {
    qsa::RuntimeLimits c;
    c.lanes = 2;
    const auto why = qsa::runtime_refusal(c);
    CHECK(why.has_value());
    CHECK(why->find("--parallel") != std::string::npos);
}

TEST(qsa_option_a_refuses_the_prefix_cache) {
    qsa::RuntimeLimits c;
    c.prefix_cache = true;
    const auto why = qsa::runtime_refusal(c);
    CHECK(why.has_value());
    CHECK(why->find("prefix cache") != std::string::npos);
}

TEST(qsa_option_a_refuses_paged_speculation) {
    qsa::RuntimeLimits c;
    c.speculative = true;
    const auto why = qsa::runtime_refusal(c);
    CHECK(why.has_value());
    CHECK(why->find("speculative") != std::string::npos);
    // The first refusal wins, in the load order lanes -> cache -> speculation.
    qsa::RuntimeLimits both;
    both.lanes        = 2;
    both.prefix_cache = true;
    const auto first = qsa::runtime_refusal(both);
    CHECK(first.has_value());
    CHECK(first->find("--parallel") != std::string::npos);
}

#endif  // ARCINT_OPENVINO
