#pragma once

// The graph op arcint builds for a GGUF K-quant projection when it opens a
// GGUF in process (docs/design-gguf-native.md §3.2). The GPU plugin at patch
// level +p7 (patch 0021) recognises it by type name only -- "FullyConnectedKQuant"
// in "arcint_opset" -- and reads two runtime-info integers off the node, so
// no class crosses the library boundary:
//   arcint_kquant_type  the ggml type id (8 = Q8_0, 12 = Q4_K, 13 = Q5_K, 14 = Q6_K)
//   arcint_kquant_k     the logical contraction size K
// Inputs: (0) the activation [.., K], (1) the weights as a u8 constant
// [N, row_bytes] holding the file's bytes verbatim, (2) an optional bias [N].
// Output: [.., N] in the activation's element type. The plugin decodes the
// blocks inside its kernel; nothing is unpacked here or at load.

#include <cstdint>
#include <memory>
#include <string>

#include <openvino/op/op.hpp>

namespace lgc {

class FullyConnectedKQuant : public ov::op::Op {
public:
    OPENVINO_OP("FullyConnectedKQuant", "arcint_opset");

    FullyConnectedKQuant() = default;
    FullyConnectedKQuant(const ov::Output<ov::Node>& activation,
                         const ov::Output<ov::Node>& weights,
                         int64_t kquant_type,
                         int64_t k,
                         int64_t n)
        : ov::op::Op({activation, weights}), kquant_type_(kquant_type), k_(k), n_(n) {
        stamp();
        constructor_validate_and_infer_types();
    }
    FullyConnectedKQuant(const ov::Output<ov::Node>& activation,
                         const ov::Output<ov::Node>& weights,
                         const ov::Output<ov::Node>& bias,
                         int64_t kquant_type,
                         int64_t k,
                         int64_t n)
        : ov::op::Op({activation, weights, bias}), kquant_type_(kquant_type), k_(k), n_(n) {
        stamp();
        constructor_validate_and_infer_types();
    }

    // Bytes one row of a [N, K] weight occupies for a ggml type id; 0 when the
    // pair is not one this op serves.
    static int64_t row_bytes(int64_t kquant_type, int64_t k) {
        switch (kquant_type) {
            case 8:  return k % 32 ? 0 : (k / 32) * 34;
            case 12: return k % 256 ? 0 : (k / 256) * 144;
            case 13: return k % 256 ? 0 : (k / 256) * 176;
            case 14: return k % 256 ? 0 : (k / 256) * 210;
            default: return 0;
        }
    }

    void validate_and_infer_types() override {
        const int64_t rb = row_bytes(kquant_type_, k_);
        NODE_VALIDATION_CHECK(this, rb != 0, "unsupported ggml type ", kquant_type_, " or K ", k_);
        const auto& w = get_input_partial_shape(1);
        NODE_VALIDATION_CHECK(this, get_input_element_type(1) == ov::element::u8 && w.is_static() && w.size() == 2 &&
                                        w[0].get_length() == n_ && w[1].get_length() == rb,
                              "weights must be u8 [", n_, ", ", rb, "], got ", get_input_element_type(1), " ", w);
        auto out = get_input_partial_shape(0);
        NODE_VALIDATION_CHECK(this, out.rank().is_static() && out.rank().get_length() >= 2, "activation rank must be >= 2");
        NODE_VALIDATION_CHECK(this, out[out.size() - 1].is_dynamic() || out[out.size() - 1].get_length() == k_,
                              "activation's last dimension must be K = ", k_);
        out[out.size() - 1] = n_;
        set_output_type(0, get_input_element_type(0), out);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        std::shared_ptr<FullyConnectedKQuant> c;
        if (new_args.size() == 3)
            c = std::make_shared<FullyConnectedKQuant>(new_args[0], new_args[1], new_args[2], kquant_type_, k_, n_);
        else
            c = std::make_shared<FullyConnectedKQuant>(new_args[0], new_args[1], kquant_type_, k_, n_);
        return c;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("kquant_type", kquant_type_);
        visitor.on_attribute("k", k_);
        visitor.on_attribute("n", n_);
        return true;
    }

    int64_t kquant_type() const { return kquant_type_; }
    int64_t k() const { return k_; }
    int64_t n() const { return n_; }

private:
    // The plugin reads these, not the attributes: runtime info survives every
    // pass the plugin runs (copy_runtime_info) and needs no shared class.
    void stamp() {
        auto& rt = get_rt_info();
        rt["arcint_kquant_type"] = kquant_type_;
        rt["arcint_kquant_k"]    = k_;
    }
    int64_t kquant_type_ = 0;
    int64_t k_           = 0;
    int64_t n_           = 0;
};

}  // namespace lgc
