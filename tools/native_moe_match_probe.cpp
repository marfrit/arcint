// native_moe_match_probe -- does the GPU pipeline's native MoE matcher take an IR?
//
// Runs ONLY ov::pass::ConvertTiledMoeBlockNativeToMoeCompressed (the plugin
// patch series' native-format matcher, 0043/0050/0052) on a model read from an
// IR and reports what it produced: the number of MOECompressed ops, each one's
// gate/up and down weight_format, and the element type, shape and producer of
// the twelve expert inputs. Nothing is compiled and no device is touched: the
// pass lives in libopenvino.so, so this runs on any host with the patched
// runtime.
//
// Why it exists: the IQ2_S-packed matcher's resolve() accepted the block and
// the callback's Constant guard then refused it (the block registered the
// scale Multiply, not the d Constant, as its `scale` anchor). A print inside
// resolve() read as "the matcher fires"; the unfused chain was then constant-
// folded on the host and on the card at compile. This probe asks the question
// at the pass's output instead.
//
// Output, one line each, machine-readable:
//   MOE_COMPRESSED <n>
//   MOE <i> gate_up_format=<f> down_format=<f> hidden=<h> inter=<i> experts=<e> top_k=<k>
//   IN <i> <slot> <node type> <element type> <shape>
//
// Build (against the patched tree's headers and runtime):
//   c++ -std=c++17 -O1 -o native_moe_match_probe tools/native_moe_match_probe.cpp \
//     -I<ov src>/src/core/include -I<ov src>/src/inference/include \
//     -I<ov src>/src/common/transformations/include -L<runtime lib> -lopenvino
// Use: native_moe_match_probe <model.xml> [has_batch_dim 0|1]

#include <iostream>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.xml> [has_batch_dim 0|1]\n";
        return 2;
    }
    const bool has_batch_dim = argc < 3 || std::string(argv[2]) != "0";
    ov::Core core;
    auto model = core.read_model(argv[1]);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertTiledMoeBlockNativeToMoeCompressed>(has_batch_dim);
    manager.run_passes(model);

    static const char* slots[] = {"hidden", "routing", "topk", "gate_w", "gate_s", "gate_zp",
                                  "up_w",   "up_s",    "up_zp", "down_w", "down_s", "down_zp"};
    size_t n = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto moe = ov::as_type_ptr<ov::op::internal::MOECompressed>(node);
        if (!moe) continue;
        const auto& c = moe->get_config();
        std::cout << "MOE " << n << " gate_up_format=" << c.gate_up_weight_format
                  << " down_format=" << c.down_weight_format << " hidden=" << c.hidden_size
                  << " inter=" << c.inter_size << " experts=" << c.num_expert << " top_k=" << c.top_k << "\n";
        for (size_t i = 0; i < moe->get_input_size() && i < 12; ++i) {
            auto src = moe->get_input_node_shared_ptr(i);
            std::cout << "IN " << n << " " << slots[i] << " " << src->get_type_name() << " "
                      << moe->get_input_element_type(i) << " " << moe->get_input_partial_shape(i) << "\n";
        }
        ++n;
    }
    std::cout << "MOE_COMPRESSED " << n << "\n";
    return 0;
}
