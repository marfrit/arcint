// The M1 executor: OpenVINO owns the math, arcint owns everything else
// (DESIGN.md §1).
//
// The exported IR is a *stateful* graph — 80 internal variables carry the KV of
// the 10 full-attention layers and the conv/ssm state of the 30 GDN layers — so
// at M1 the cache lives inside OpenVINO and each sequence gets its own
// InferRequest. §3.2's stateless graphs with explicit cache I/O, and with them
// the paged KV pool and the GDN ledger, are M2 work; the state is reachable
// through ov::VariableState, which is what makes that possible at all.
//
// The export is a VLM: the language model consumes `inputs_embeds`, not token
// ids, so the text-embeddings graph runs first. v1 is text-only and the vision
// graphs are never compiled.
//
// M6 adds the second lane. A lane is one sequence's worth of mutable execution
// state — its own InferRequest for the language model, the embeddings gather
// and the MTP head, its own GDN checkpoint rows, its own KV block table — and
// nothing in it is shared with the other lane. What *is* shared is immutable or
// refcounted: the compiled models (weights are shared between InferRequests of
// one CompiledModel, which is what makes a second lane cost activations rather
// than another 12.8 GiB), and the KV page pool, from which each lane draws its
// own pages. Two sequences therefore never meet inside one graph execution,
// which is the rule all three reference engines agree on for hybrid models.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <set>

#include <dirent.h>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>

#include "build_info.h"

#include <optional>
#include <unordered_map>
#include <fstream>
#include <queue>
#include <sstream>

#include <nlohmann/json.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/assign.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <openvino/runtime/remote_tensor.hpp>

#include <minja/chat-template.hpp>

#include "config.h"
#include "core/artifact.h"
#include "core/block_pool.h"
#include "core/drafter.h"
#include "core/prefix_cache.h"
#include "core/sampler.h"
#include "core/turnstile.h"
#include "exec/backend.h"
#include "util/log.h"

namespace lgc {
namespace {

using json = nlohmann::json;

constexpr const char* kInputsEmbeds  = "inputs_embeds";
constexpr const char* kAttentionMask = "attention_mask";
constexpr const char* kPositionIds   = "position_ids";
constexpr const char* kBeamIdx       = "beam_idx";

// Fixed process-wide key for the prefix hash chain. It is not a secret — the
// point is that a hash is never trusted on its own; every hit re-verifies the
// tokens before the state is reused.
constexpr uint64_t kPrefixCacheKey = 0x617263696e743031ull;  // "arcint01"

// The block hashes are keyed, and the key has to name the thing the cached state
// belongs to. Today the cache lives inside one process serving one artifact, so a
// constant would do; the day a process reloads a different artifact, or the MTP
// export changes what a blob even contains, a stale hit would be silent rather
// than a miss. vLLM carries the same idea as per-entry "extra keys".
uint64_t prefix_cache_key(const Artifact& a) {
    uint64_t k = kPrefixCacheKey;
    for (char c : a.arch_hash) k = k * 1099511628211ull ^ static_cast<uint64_t>(c);
    // The blob's own layout depends on this: with a head loaded it carries the
    // head's KV, cursor and pending row as well.
    k = k * 1099511628211ull ^ (a.has_mtp_head ? 0x4d545031ull : 0x4e4f4e45ull);
    return k;
}

std::string tokenizers_extension_path() {
    // Shipped alongside the OpenVINO wheel rather than with the runtime, so the
    // path is deployment configuration. The environment wins; otherwise use
    // whatever CMake found at configure time, so an installed binary starts
    // without a wrapper that knows where a virtualenv lives; otherwise let the
    // loader search.
    if (const char* env = std::getenv("ARCINT_TOKENIZERS_SO")) return env;
    if (std::strlen(ARCINT_TOKENIZERS_SO_DEFAULT) > 0) return ARCINT_TOKENIZERS_SO_DEFAULT;
    return "libopenvino_tokenizers.so";
}

// ------------------------------------------------------------------ tokenizer
class OvTokenizer final : public Tokenizer {
public:
    OvTokenizer(ov::Core& core, const Artifact& a, std::vector<int> eos_ids)
        : eos_ids_(std::move(eos_ids)) {
        tokenizer_   = core.compile_model(a.tokenizer_xml, "CPU");
        detokenizer_ = core.compile_model(a.detokenizer_xml, "CPU");
        tok_req_     = tokenizer_.create_infer_request();
        detok_req_   = detokenizer_.create_infer_request();
    }

    std::vector<int> encode(std::string_view text) override {
        std::lock_guard<std::mutex> guard(mutex_);

        ov::Tensor in(ov::element::string, ov::Shape{1});
        in.data<std::string>()[0] = std::string(text);
        tok_req_.set_input_tensor(in);
        tok_req_.infer();

        // The tokenizer emits input_ids plus attention_mask (and sometimes
        // token_type_ids); take the one named input_ids rather than index 0.
        const ov::Tensor ids = named_output(tok_req_, tokenizer_, "input_ids");
        const int64_t*   p   = ids.data<const int64_t>();
        const size_t     n   = ids.get_size();

        std::vector<int> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.push_back(static_cast<int>(p[i]));
        return out;
    }

    std::string decode(const std::vector<int>& ids) override {
        if (ids.empty()) return {};
        std::lock_guard<std::mutex> guard(mutex_);
        return detokenize(ids);
    }

    // Qwen uses byte-level BPE, where a token's bytes do not depend on its
    // neighbours, so a per-token piece is exact. A piece may still be half a
    // code point — utf8::Streamer is what keeps that off the wire.
    std::string decode_one(int id) override {
        std::lock_guard<std::mutex> guard(mutex_);
        return detokenize({id});
    }

    int eos_id() const override { return eos_ids_.empty() ? -1 : eos_ids_.front(); }

    const std::vector<int>& eos_ids() const { return eos_ids_; }

private:
    static ov::Tensor named_output(ov::InferRequest& req, const ov::CompiledModel& model,
                                   const std::string& name) {
        for (const auto& port : model.outputs()) {
            const auto& names = port.get_names();
            if (names.count(name) != 0) return req.get_tensor(port);
        }
        return req.get_output_tensor(0);
    }

    std::string detokenize(const std::vector<int>& ids) {
        ov::Tensor in(ov::element::i64, ov::Shape{1, ids.size()});
        int64_t*   p = in.data<int64_t>();
        for (size_t i = 0; i < ids.size(); ++i) p[i] = ids[i];

        detok_req_.set_input_tensor(in);
        detok_req_.infer();

        const ov::Tensor out = detok_req_.get_output_tensor(0);
        if (out.get_element_type() != ov::element::string || out.get_size() == 0) return {};
        return out.data<std::string>()[0];
    }

    std::mutex        mutex_;
    ov::CompiledModel tokenizer_;
    ov::CompiledModel detokenizer_;
    ov::InferRequest  tok_req_;
    ov::InferRequest  detok_req_;
    std::vector<int>  eos_ids_;
};

// ------------------------------------------------------------ state blobs
//
// A cached prefix must restore the attention KV *and* the GDN recurrent state
// at the same boundary — both or neither (§3.4). On this graph both live as
// OpenVINO variables, so a snapshot is simply every variable's tensor, and the
// "both" part is free: there is no way to restore one without the other.
//
// Each blob is self-describing (type name, shape, bytes) so a restore never
// depends on the live state happening to have the same shape it had when the
// snapshot was taken.
uint32_t get_u32(const std::vector<uint8_t>& in, size_t& off) {
    if (off + 4 > in.size()) throw std::runtime_error("state blob truncated");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(in[off + static_cast<size_t>(i)]) << (i * 8);
    off += 4;
    return v;
}

std::vector<uint8_t> serialise_state(const ov::Tensor& t) {
    const std::string type  = t.get_element_type().get_type_name();
    const ov::Shape&  shape = t.get_shape();

    // Sized once and filled by offset. Growing it with reserve() plus range
    // inserts is equivalent but makes GCC -O3 emit a spurious
    // -Wfree-nonheap-object, and a header this small does not need the
    // ceremony of proving that diagnostic wrong on every build.
    const size_t header = 4 + type.size() + 4 + 4 * shape.size();
    std::vector<uint8_t> out(header + t.get_byte_size());

    size_t off = 0;
    auto   u32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) out[off++] = static_cast<uint8_t>(v >> (i * 8));
    };

    u32(static_cast<uint32_t>(type.size()));
    std::memcpy(out.data() + off, type.data(), type.size());
    off += type.size();
    u32(static_cast<uint32_t>(shape.size()));
    for (size_t d : shape) u32(static_cast<uint32_t>(d));

    std::memcpy(out.data() + off, t.data(), t.get_byte_size());
    return out;
}

ov::Tensor deserialise_state(const std::vector<uint8_t>& blob) {
    size_t         off = 0;
    const uint32_t tlen = get_u32(blob, off);
    if (off + tlen > blob.size()) throw std::runtime_error("state blob truncated (type)");
    const std::string type(reinterpret_cast<const char*>(blob.data() + off), tlen);
    off += tlen;

    const uint32_t rank = get_u32(blob, off);
    ov::Shape      shape;
    shape.reserve(rank);
    for (uint32_t i = 0; i < rank; ++i) shape.push_back(get_u32(blob, off));

    ov::Tensor t(ov::element::Type(type), shape);
    if (off + t.get_byte_size() > blob.size()) {
        throw std::runtime_error("state blob truncated (payload)");
    }
    std::memcpy(t.data(), blob.data() + off, t.get_byte_size());
    return t;
}

// Only the final token's logits are ever sampled, but the graph computes them
// for every prompt token: `logits` is [tokens, 1, 248320], so an unchunked 8k
// prefill materialises 8.1 GiB of logits on top of 12.8 GiB of weights and the
// card answers CL_OUT_OF_RESOURCES. Slicing the hidden state to its last row
// just before the LM head makes prefill produce one row instead of `tokens`.
//
// This is a pure win, and it retires a trade-off rather than balancing one:
// measured on the coder, unchunked prefill went from failing at ~8k to 9156 tok
// at 2247 t/s and 27444 tok at 732 t/s, greedy output byte-identical to the
// unsliced graph. It is faster than chunking *and* keeps the equality gate
// chunking had to give up.
// The MTP head consumes the base model's final hidden state, which the graph
// only computes on its way into the LM head. Exposing it as a second output
// costs nothing; it has to happen *before* the logits slice, or the head would
// see only the rows the slice keeps and could not be primed over a prompt.
bool expose_hidden_state(const std::shared_ptr<ov::Model>& model) {
    const auto& results = model->get_results();
    if (results.empty()) return false;

    std::shared_ptr<ov::Node> node = results[0]->input_value(0).get_node_shared_ptr();
    for (int hop = 0; hop < 8 && node && node->get_input_size() > 0; ++hop) {
        if (ov::as_type_ptr<ov::op::v0::MatMul>(node) != nullptr) break;
        const std::string t = node->get_type_name();
        if (t != "Convert" && t != "Reshape") return false;
        node = node->input_value(0).get_node_shared_ptr();
    }
    if (ov::as_type_ptr<ov::op::v0::MatMul>(node) == nullptr) return false;

    const auto res = std::make_shared<ov::op::v0::Result>(node->input_value(0));
    res->get_output_tensor(0).set_names({"hidden_states"});
    model->add_results({res});
    model->validate_nodes_and_infer_types();
    return true;
}

// The DFlash2 drafter conditions on the residual stream after target layers
// {ids}: HF's hidden_states[id+1], which is the value entering layer id+1's
// input_layernorm. Tap each, concatenate on the hidden axis, expose as one
// output. Before the logits slice, for the same reason as hidden_states.
static std::string dflash_read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) throw std::runtime_error("cannot read " + path);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

bool expose_dflash_feats(const std::shared_ptr<ov::Model>& model,
                         const std::vector<int64_t>& layer_ids) {
    ov::OutputVector taps;
    for (int64_t id : layer_ids) {
        const std::string key =
            "language_model.layers." + std::to_string(id + 1) + ".input_layernorm";
        std::shared_ptr<ov::Node> found;
        for (const auto& op : model->get_ops()) {
            if (op->get_friendly_name().find(key) == std::string::npos) continue;
            std::shared_ptr<ov::Node> src = op;
            for (int hop = 0; hop < 8 && src->get_input_size() > 0 &&
                              src->get_friendly_name().find(key) != std::string::npos;
                 ++hop) {
                src = src->input_value(0).get_node_shared_ptr();
            }
            if (src->get_friendly_name().find(key) == std::string::npos) {
                found = src;
                break;
            }
        }
        if (!found) return false;
        taps.push_back(found->output(0));
    }
    const auto cat = std::make_shared<ov::op::v0::Concat>(taps, -1);
    const auto res = std::make_shared<ov::op::v0::Result>(cat);
    res->get_output_tensor(0).set_names({"dflash_feats"});
    model->add_results({res});
    model->validate_nodes_and_infer_types();
    return true;
}

// `keep_rows` is 1 for plain decoding. Speculative decoding needs the model's
// own prediction *at every draft position* to verify a draft, so it asks for
// the last 1 + draft_tokens rows. That keeps all of the memory win -- the wall
// was one row per prompt token (8.1 GiB at 8k), not nine rows (5.4 MiB) -- and
// the last row still means what it meant before, so greedy output is unchanged.
// A graph node whose *type name* is LgcPermute, which is how OpenVINO's GPU
// plugin binds a custom OpenCL kernel: the CustomLayer entry in the XML is
// matched against the op's type, so putting this node in the graph is what
// makes kernels/permute.cl run in its place.
//
// It carries no attributes on purpose. The permutation is (0,2,1,3), fixed, and
// lives in the kernel; a node that could express any permutation would need the
// order passed through a second buffer the custom-layer ABI does not give us.
class LgcPermute : public ov::op::Op {
public:
    OPENVINO_OP("LgcPermute");

    LgcPermute() = default;
    explicit LgcPermute(const ov::Output<ov::Node>& arg) : ov::op::Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const ov::PartialShape& in = get_input_partial_shape(0);
        ov::PartialShape        out = in;
        if (in.rank().is_static() && in.rank().get_length() == 4) {
            out = ov::PartialShape{in[0], in[2], in[1], in[3]};
        }
        set_output_type(0, get_input_element_type(0), out);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        check_new_args_count(this, args);
        return std::make_shared<LgcPermute>(args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor&) override { return true; }
};

// Replaces exactly the (0,2,1,3) rank-4 transposes -- the head-major swap the
// GDN layers do three times each -- and leaves every other Transpose alone.
// That selectivity is the point: the custom-layer binding is by type name, so
// without it one kernel would have to serve every permutation in the graph.
size_t route_head_swap_permutes(const std::shared_ptr<ov::Model>& model) {
    const std::vector<int64_t> head_swap{0, 2, 1, 3};
    size_t                     n = 0;

    for (const std::shared_ptr<ov::Node>& node : model->get_ordered_ops()) {
        const auto t = ov::as_type_ptr<ov::op::v1::Transpose>(node);
        if (t == nullptr) continue;

        const auto order =
            ov::as_type_ptr<ov::op::v0::Constant>(t->input_value(1).get_node_shared_ptr());
        if (order == nullptr || order->cast_vector<int64_t>() != head_swap) continue;

        const ov::PartialShape& in = t->get_input_partial_shape(0);
        if (in.rank().is_dynamic() || in.rank().get_length() != 4) continue;

        // Only the GDN head-major swap: [B, S, 32, 128]. A custom node is a
        // fusion barrier -- OpenVINO cannot pattern-match through an opaque op
        // -- so routing every (0,2,1,3) transpose in the graph costs more in
        // broken fusions than the faster kernel wins. Take the hot class only.
        if (in[2].is_dynamic() || in[3].is_dynamic()) continue;
        if (in[2].get_length() != 32 || in[3].get_length() != 128) continue;

        const auto rep = std::make_shared<LgcPermute>(t->input_value(0));
        rep->set_friendly_name(t->get_friendly_name());
        ov::replace_node(t, rep);
        ++n;
    }
    if (n > 0) model->validate_nodes_and_infer_types();
    return n;
}

// Pads the shared-expert gate's output width from 1 to `npad`.
//
// The gate is a [M, 2048] x [2048, 1] MatMul with a fused sigmoid. oneDNN has
// no catalogued GEMM for that shape at M >= 128 -- read from its own dispatch
// log on 2026-08-30: "matching kernel not found in catalog", with zero
// candidates considered, not a rejected one -- so every prefill chunk runs it
// on ocl:ref, 40 nodes per chunk at 12.2% of prefill wall on the device
// timeline. The catalog is matched on shape, so this widens the shape: the u4
// weight, its zero points and its scales are padded along the output axis with
// rows whose scale is zero (so the dequantised rows are exactly zero), the
// Reshape that flattens the groups is told the new width, and a Slice takes
// column 0 back out before the sigmoid. Rung zero of DESIGN §1.1: our graph,
// no plugin change, no upstream loop. Whether the widened shape lands on a
// JIT kernel and what that costs is measured, not assumed.
size_t pad_gate_matmuls(const std::shared_ptr<ov::Model>& model, size_t npad) {
    using ov::op::v0::Constant;
    if (npad < 2) return 0;
    size_t done = 0;
    for (const auto& node : model->get_ordered_ops()) {
        const auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(node);
        if (mm == nullptr || !mm->get_transpose_b()) continue;
        if (mm->get_friendly_name().find("shared_expert_gate") == std::string::npos) continue;
        const ov::PartialShape out_ps = mm->get_output_partial_shape(0);
        if (out_ps.rank().is_dynamic()) continue;
        const int64_t last = out_ps.rank().get_length() - 1;
        if (!out_ps[last].is_static() || out_ps[last].get_length() != 1) continue;

        // Everything upstream of the weight input must be the decompression
        // subgraph -- Convert / Reshape / Multiply / Subtract over constants.
        // Anything else and this gate is not the shape we understand: skip it
        // rather than guess.
        std::vector<std::shared_ptr<ov::Node>>  stack{mm->input_value(1).get_node_shared_ptr()};
        std::set<const ov::Node*>               seen;
        std::vector<std::shared_ptr<Constant>>  consts;
        std::vector<std::shared_ptr<ov::Node>>  chain;   // the decompression subgraph
        bool                                    ok = true;
        while (!stack.empty() && ok) {
            const std::shared_ptr<ov::Node> n = stack.back();
            stack.pop_back();
            if (!seen.insert(n.get()).second) continue;
            if (auto c = ov::as_type_ptr<Constant>(n)) {
                consts.push_back(c);
                continue;
            }
            const std::string t = n->get_type_name();
            if (t != "Convert" && t != "Reshape" && t != "Multiply" && t != "Subtract") {
                ok = false;
                break;
            }
            chain.push_back(n);
            for (size_t i = 0; i < n->get_input_size(); ++i) {
                stack.push_back(n->input_value(i).get_node_shared_ptr());
            }
        }
        if (!ok) continue;

        std::vector<std::pair<std::shared_ptr<Constant>, std::shared_ptr<Constant>>> repl;
        for (const auto& c : consts) {
            const ov::Shape sh = c->get_shape();
            if (sh.size() == 1 && c->get_element_type() == ov::element::i64) {
                // The Reshape target: [1, 2048] -> [npad, 2048].
                std::vector<int64_t> v = c->cast_vector<int64_t>();
                if (v.empty() || v[0] != 1) { ok = false; break; }
                v[0] = static_cast<int64_t>(npad);
                repl.emplace_back(c, Constant::create(ov::element::i64, sh, v));
            } else if (sh.size() >= 2 && sh[0] == 1) {
                // Weight [1, groups, gs], zero point and scale [1, groups, 1]:
                // the whole constant is one output row, so the padded tensor is
                // that row followed by zero rows. For the scale, zero rows make
                // the dequantised weight exactly zero whatever the nibbles say.
                ov::Shape nsh = sh;
                nsh[0]        = npad;
                const size_t row_bytes = c->get_byte_size();
                std::vector<uint8_t> buf(row_bytes * npad, 0);
                std::memcpy(buf.data(), c->get_data_ptr(), row_bytes);
                repl.emplace_back(c, std::make_shared<Constant>(c->get_element_type(), nsh,
                                                                buf.data()));
            } else {
                ok = false;
                break;
            }
        }
        if (!ok || repl.empty()) continue;
        for (auto& pr : repl) {
            pr.second->set_friendly_name(pr.first->get_friendly_name());
            ov::replace_node(pr.first, pr.second);
        }
        // Column 0 back out. Two choices here were measured (DESIGN 7.0.2g/i):
        // a Slice between the MatMul and the sigmoid becomes a launched
        // strided_slice kernel at ~20 us of host time each, forty per step, and
        // it un-fuses the sigmoid from the FC. So the cut goes AFTER the sigmoid
        // -- which then stays an FC post-op -- and is a VariadicSplit, which the
        // plugin lowers to a crop; a crop at offset 0 is the case its in-place
        // optimisation accepts, so at best it is a view and not a kernel at all.
        // The [M, npad-1] remainder has no consumer and is dropped at compile.
        // The constants changed a few nodes below the MatMul, and every node in
        // between still carries its old shape until re-inferred; the split
        // validates against the sigmoid's shape at construction. Re-infer the
        // decompression chain only -- a model-wide pass here would push the
        // widened shape into the sigmoid's consumers before they are rewired
        // and fail on the Multiply. Bottom-up, because a Reshape re-inferred
        // before its input throws on the spot rather than waiting for a later
        // pass; the walk above was a depth-first descent from the MatMul, so
        // its reverse is exactly the bottom-up order.
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) (*it)->revalidate_and_infer_types();
        mm->revalidate_and_infer_types();
        ov::Output<ov::Node> cut_from = mm->output(0);
        if (const auto ts = mm->output(0).get_target_inputs(); ts.size() == 1) {
            const auto user = ts.begin()->get_node()->shared_from_this();
            if (ov::as_type_ptr<ov::op::v0::Sigmoid>(user) != nullptr) {
                user->revalidate_and_infer_types();
                cut_from = user->output(0);
            }
        }
        const auto targets = cut_from.get_target_inputs();
        const auto ax      = Constant::create(ov::element::i64, {}, {last});
        const auto lens    = Constant::create(ov::element::i64, {2},
                                              {int64_t{1}, static_cast<int64_t>(npad) - 1});
        const auto split   = std::make_shared<ov::op::v1::VariadicSplit>(cut_from, ax, lens);
        split->set_friendly_name(mm->get_friendly_name() + "/gate_pad_split");
        for (auto t : targets) t.replace_source_output(split->output(0));
        ++done;
    }
    if (done > 0) model->validate_nodes_and_infer_types();
    return done;
}

bool slice_logits_to_last_token(const std::shared_ptr<ov::Model>& model,
                                int64_t keep_rows, int64_t token_axis) {
    const auto& results = model->get_results();
    if (results.empty()) return false;

    // Walk back through shape-only ops to the matmul that is the LM head.
    std::shared_ptr<ov::Node> node = results[0]->input_value(0).get_node_shared_ptr();
    for (int hop = 0; hop < 8 && node && node->get_input_size() > 0; ++hop) {
        if (ov::as_type_ptr<ov::op::v0::MatMul>(node) != nullptr) break;
        const std::string t = node->get_type_name();
        if (t != "Convert" && t != "Reshape") return false;
        node = node->input_value(0).get_node_shared_ptr();
    }
    const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
    if (matmul == nullptr) return false;

    // Rewriting the wrong operand would silently produce wrong logits with no
    // error, which is the one failure mode a graph rewrite must not have. Only
    // proceed when input 0 is unmistakably the activation: not transposed, and
    // not a constant weight.
    if (matmul->get_transpose_a()) return false;
    if (ov::as_type_ptr<ov::op::v0::Constant>(
            node->input_value(0).get_node_shared_ptr()) != nullptr) {
        return false;
    }

    const ov::Output<ov::Node> hidden = node->input_value(0);
    const ov::PartialShape&    ps     = hidden.get_partial_shape();
    if (ps.rank().is_dynamic() || ps.rank().get_length() < 2) return false;
    // The token axis is not at a fixed position, and it cannot be found from
    // the shape: the dense export is [1, tokens, hidden] and the paged export
    // is [tokens, 1, hidden], but the paged graph declares both leading axes
    // dynamic, so "the one that can exceed one row" matches both. Assuming
    // rank-2 sliced the singleton batch axis of the paged graph -- "last 1 of
    // 1", a no-op that returned true -- and every prefill chunk went on
    // computing and copying [M, vocab] f32 logits to the host: 1.9 GiB per
    // 2048-token chunk at the link's full rate, 17.6% of prefill wall, found
    // by an OpenCL timeline on 2026-08-30. So the caller states the layout,
    // and the reservation probe checks the claim against a real forward.
    const int64_t axis = token_axis >= 0 ? token_axis : ps.rank().get_length() - 2;
    if (axis >= ps.rank().get_length() - 1) return false;

    using ov::op::v0::Constant;
    const auto start = Constant::create(ov::element::i64, {1}, {-keep_rows});
    const auto stop  = Constant::create(ov::element::i64, {1},
                                       {std::numeric_limits<int64_t>::max()});
    const auto step  = Constant::create(ov::element::i64, {1}, {int64_t{1}});
    const auto ax    = Constant::create(ov::element::i64, {1}, {axis});

    const auto slice = std::make_shared<ov::op::v8::Slice>(hidden, start, stop, step, ax);
    node->input(0).replace_source_output(slice->output(0));
    model->validate_nodes_and_infer_types();
    return true;
}

// The attention KV is the only part of the state that grows with context: at
// 262144 tokens it is ~10.7 GiB in fp32 against 12.8 GiB of weights, which does
// not fit a 22.7 GiB card. The GDN conv/ssm variables are fixed-size and are
// left alone.
//
// OpenVINO's KV_CACHE_PRECISION property is accepted on the GPU but has no
// effect on this stateful graph (measured: state stays fp32, output unchanged) —
// it only governs the paged path. So the storage type is changed directly:
// Convert on each key/value variable's initialiser, on its read, and on its
// assign, then the Variable itself is relabelled. Compute stays fp32; only what
// is *stored* shrinks.
//
// Measured on the coder: KV 12.9 -> 6.0 MiB at 306 tokens, greedy output
// byte-identical to fp32, about 10% slower from the extra Converts.
size_t store_kv_state_as(const std::shared_ptr<ov::Model>& model, const ov::element::Type& type) {
    auto is_kv = [](const std::string& id) {
        return id.find(".key.") != std::string::npos || id.find(".value.") != std::string::npos;
    };

    std::vector<std::shared_ptr<ov::Node>> reads, assigns;
    for (const auto& node : model->get_ordered_ops()) {
        const std::string t = node->get_type_name();
        if (t != "ReadValue" && t != "Assign") continue;

        std::string id;
        if (auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(node)) {
            id = rv->get_variable_id();
        } else if (auto as = ov::as_type_ptr<ov::op::v6::Assign>(node)) {
            id = as->get_variable_id();
        } else {
            continue;
        }
        if (!is_kv(id)) continue;
        (t == "ReadValue" ? reads : assigns).push_back(node);
    }
    if (reads.empty() || reads.size() != assigns.size()) return 0;

    for (const auto& rv : reads) {
        if (rv->get_input_size() > 0) {
            auto to_store = std::make_shared<ov::op::v0::Convert>(rv->input_value(0), type);
            rv->input(0).replace_source_output(to_store->output(0));
        }
        // Snapshot the consumers before inserting, or the new Convert rewires
        // itself onto its own input.
        auto out       = rv->output(0);
        auto consumers = out.get_target_inputs();
        auto back      = std::make_shared<ov::op::v0::Convert>(out, ov::element::f32);
        for (auto& inp : consumers) inp.replace_source_output(back->output(0));
    }
    for (const auto& as : assigns) {
        auto to_store = std::make_shared<ov::op::v0::Convert>(as->input_value(0), type);
        as->input(0).replace_source_output(to_store->output(0));
    }

    size_t relabelled = 0;
    for (const auto& v : model->get_variables()) {
        auto info = v->get_info();
        if (!is_kv(info.variable_id)) continue;
        info.data_type = type;
        v->update(info);
        ++relabelled;
    }
    model->validate_nodes_and_infer_types();
    return relabelled;
}

// -------------------------------------------------------------------- backend
class OvBackend final : public Backend {
public:
    // One lane (DESIGN.md §4.1). Everything a running sequence writes to lives
    // here, indexed by the slot the HTTP layer leased; nothing here is touched
    // by another lane, so the decode loop needs no lock of its own.
    struct Lane {
        int              index = 0;
        ov::InferRequest req;        // the paged language model; unused on --no-paged
        ov::InferRequest embed;      // the embeddings gather
        // The MTP head carries its own attention KV over the prefix, so it
        // needs its own request per lane as well: a shared head would let one
        // sequence draft from the other's prefix, which is cross-slot bleed in
        // its most confusing form (the drafts are verified, so the output would
        // stay correct and only the acceptance rate would quietly collapse).
        ov::InferRequest mtp_layer;
        ov::InferRequest mtp_head;
        ov::Tensor       mtp_pending;      // h_t, awaiting x_{t+1}
        bool             mtp_has_pending = false;
        size_t           mtp_len = 0;      // positions in the head's KV
        size_t           mtp_pos = 0;      // true position, for rope

        // DFlash2 (docs/dflash-pairing-probe.md). The draft's context KV lives
        // in dflash_req's graph state and only ever receives ACCEPTED
        // positions, so it needs no rollback. `dflash_pending` holds accepted
        // rows' target features not yet fed; `dfeats` is the last forward's
        // feature output, copied out like `hidden`.
        ov::InferRequest   dflash_req;
        ov::InferRequest   dflash_head;
        ov::Tensor         dfeats;
        std::vector<float> dflash_pending;
        size_t             dflash_base = SIZE_MAX;   // abs pos of pending[0]

        // The GDN checkpoint rows this lane owns (its own device tensors, not a
        // window into a shared table: a row write copies the whole tensor back,
        // and sharing one would let a snapshot in one lane zero the other's
        // rows), and the KV pages of the sequence it is running.
        std::vector<ov::RemoteTensor> la_tensors;
        size_t                        committed_row = 0;
        std::vector<int32_t>          blocks;
        PrefixCache::EntryRef         hit_keep;   // pages mapped from the cache
        // The paged graph's small index inputs as USM-host tensors, keyed by
        // name and length: the plugin dereferences these instead of mapping a
        // device buffer (ARCINT_PA_HOST_INPUTS, 7.0.2p).
        std::unordered_map<std::string, std::pair<ov::RemoteTensor, void*>> host_idx;

        // The graph's outputs, copied out of the request while the lane still
        // holds the device. Measured 2026-08-29: a second InferRequest of the
        // same CompiledModel adds 0.00 GiB of activations, i.e. the GPU plugin
        // pools intermediate buffers per compiled model, not per request. The
        // consequence is that a request's own output tensor is only valid until
        // the next execution on that model — by anyone — so reading it after
        // the turn has passed to the other lane would read the other lane's
        // numbers. Copying is ~1 MB per step against an 11 ms step.
        ov::Tensor         logits;
        ov::Tensor         hidden;
        std::vector<float> logit_scratch;
        GenerationStats*   stats = nullptr;
        // Waits are accumulated per graph execution and flushed once per
        // emitted token, because the number a reader feels is the gap between
        // tokens, not the wait before one of the two or three executions that
        // produced it.
        double              stall_accum = 0.0;
        std::vector<double> stalls;
    };

    OvBackend(const Artifact& artifact, const Config& cfg, int n_ctx)
        : artifact_(artifact), prefill_chunk_(cfg.prefill_chunk) {
        const std::string& device    = cfg.device;
        std::string cache_dir = cfg.cache_dir;
        lane_count_           = std::max(1, cfg.parallel);
        if (cfg.draft_tokens > 0) {
            drafter_     = std::make_unique<NgramDrafter>(static_cast<size_t>(cfg.draft_ngram),
                                                          static_cast<size_t>(cfg.draft_tokens));
            draft_tokens_ = static_cast<size_t>(cfg.draft_tokens);
            drafting_     = true;
        }
        if (cfg.prefix_cache_mib > 0) {
            prefix_cache_ = std::make_unique<PrefixCache>(
                static_cast<size_t>(cfg.prefix_cache_mib) * 1024 * 1024, cfg.kv_block_size,
                prefix_cache_key(artifact), static_cast<size_t>(cfg.cache_host_mib) * 1024 * 1024);
            if (cfg.cache_host_mib > 0) {
                prefix_cache_->set_demote([this](PrefixCache::Entry& e) { return demote_entry(e); });
            }
        }
        core_.add_extension(tokenizers_extension_path());
        if (!cache_dir.empty()) core_.set_property(ov::cache_dir(cache_dir));

        tokenizer_ = std::make_unique<OvTokenizer>(core_, artifact, artifact.eos_ids);

        if (cfg.paged) {
            load_paged(cfg, n_ctx);
            template_ = std::make_unique<minja::chat_template>(artifact.chat_template,
                                                               artifact.bos_token,
                                                               artifact.eos_token);
            status_.id               = artifact.id;
            status_.served_id        = cfg.served_model_name.empty() ? artifact.id
                                                                     : cfg.served_model_name;
            status_.quant            = cfg.quant;
            status_.loaded           = true;
            status_.stub             = false;
            status_.n_ctx            = paged_n_ctx_;
            status_.n_ctx_train      = artifact.n_ctx_train;
            status_.n_layer          = artifact.n_layer;
            status_.n_gdn_layer      = artifact.n_gdn_layer;
            status_.n_attn_layer     = artifact.n_attn_layer;
            status_.mtp_enabled      = mtp_ready_;
            status_.weights_bytes    = artifact.weights_bytes;
            status_.sampler_defaults = artifact.sampler;
            return;
        }

        log::info("load", "compiling embeddings graph on %s", device.c_str());
        auto t0    = std::chrono::steady_clock::now();
        embeddings_ = core_.compile_model(artifact.text_embeddings_xml, device);
        log::info("load", "embeddings ready in %.1f s", seconds_since(t0));

        // One lane: the stateful graph has a single internal state, so this
        // path serves one sequence at a time whatever --parallel says. It stays
        // as the reference oracle the equivalence suite compares against.
        lanes_.push_back(std::make_unique<Lane>());
        lanes_.back()->index = 0;
        lanes_.back()->embed = embeddings_.create_infer_request();
        if (lane_count_ > 1) {
            log::warn("load", "--no-paged serves one sequence at a time; the other %d lane(s) "
                              "will wait rather than run",
                      lane_count_ - 1);
        }

        std::shared_ptr<ov::Model> model = core_.read_model(artifact.language_model_xml);

        if (cfg.kv_dtype != "fp32") {
            const ov::element::Type kv = ov::element::f16;  // config refuses anything else
            try {
                const size_t n = store_kv_state_as(model, kv);
                if (n > 0) {
                    log::info("load", "attention KV stored as %s (%zu variables); the GDN state "
                                      "is fixed-size and stays fp32",
                              cfg.kv_dtype.c_str(), n);
                } else {
                    log::warn("load", "%s", "could not retype the KV state; it stays fp32");
                }
            } catch (const std::exception& e) {
                log::warn("load", "KV retype to %s failed, staying fp32: %s", cfg.kv_dtype.c_str(),
                          e.what());
                model = core_.read_model(artifact.language_model_xml);
            }
        }

        // Order matters: the head must see every prompt position, so this runs
        // before the slice rewires the LM head's input.
        offload_ratio_ = cfg.offload_ratio;
        if (offload_ratio_ > 0) {
            log::info("load", "expert offload at %d%%: the plugin keeps that share of the MoE "
                              "expert weights off the card and streams them",
                      offload_ratio_);
        }
        want_mtp_ = cfg.mtp != "off" && artifact.has_mtp_head;
        if (want_mtp_ && !expose_hidden_state(model)) {
            log::warn("mtp", "%s", "could not expose the hidden state; MTP disabled");
            want_mtp_ = false;
        }

        if (cfg.slice_logits) {
            // The MTP head drafts one token even when --draft is 0, and
            // verification needs a logits row per drafted position, so the slice
            // has to be at least two rows wide whenever the head is loaded.
            const size_t  drafts_max = std::max<size_t>(draft_tokens_, want_mtp_ ? 1 : 0);
            const int64_t keep       = static_cast<int64_t>(1 + drafts_max);
            if (slice_logits_to_last_token(model, keep, -1)) {
                log::info("load", "logits sliced to the last %lld token(s): prefill computes "
                                  "%lld row(s) instead of one per prompt token",
                          static_cast<long long>(keep), static_cast<long long>(keep));
            } else {
                log::warn("load", "%s",
                          "could not find the LM head to slice; prefill will materialise logits "
                          "for every prompt token and will run out of memory at depth");
            }
        }

        if (!cfg.custom_kernels.empty()) {
            const size_t n = route_head_swap_permutes(model);
            core_.set_property(device, ov::AnyMap{{"CONFIG_FILE", cfg.custom_kernels}});
            // A blob cache keyed on the plugin config would otherwise let a
            // graph compiled with these kernels be served without them, or the
            // other way round. This is a measurement switch; make it cold.
            cache_dir.clear();
            log::info("load", "custom kernels from %s: %zu head-swap permutes routed to "
                              "LgcPermute (blob cache off)",
                      cfg.custom_kernels.c_str(), n);
        }

        log::info("load", "compiling language model on %s (%d layers, %d GDN + %d attn)",
                  device.c_str(), artifact.n_layer, artifact.n_gdn_layer, artifact.n_attn_layer);

        // A compiled MoE graph imported from a blob cache can come back with an
        // uninitialised expert weight provider (openvino#37607). It does not
        // throw at compile time — it throws on the first infer, which without a
        // guard means the server starts healthy and then 500s every request.
        //
        // Measured on this fleet, 2026-08-28, B60 + the b5 coder artifact: the
        // only genuinely fast import observed (15.6 s against ~45 s cold) was
        // the broken one. So the cache is used, then proven by a real forward
        // pass, and a cache that fails the proof is discarded rather than
        // served. DESIGN.md §3.2 expected weightless mode to make imports safe;
        // on this artifact it does not.
        bool ready = false;
        if (!cache_dir.empty()) {
            ov::AnyMap cfg;
            cfg[ov::cache_mode.name()]   = ov::CacheMode::OPTIMIZE_SIZE;
            cfg[ov::weights_path.name()] = artifact.language_model_bin;
            if (offload_ratio_ > 0) cfg["OFFLOAD_RATIO"] = offload_ratio_;
            if (std::getenv("ARCINT_PROFILE") != nullptr) cfg[ov::enable_profiling.name()] = true;

            // Was there anything to import, or is this the run that writes the
            // blob? Both take this branch, and reporting them the same way makes
            // a cold start read like a successful import.
            bool had_blob = false;
            if (DIR* d = ::opendir(cache_dir.c_str())) {
                while (dirent* e = ::readdir(d)) {
                    const std::string n(e->d_name);
                    if (n != "." && n != "..") { had_blob = true; break; }
                }
                ::closedir(d);
            }

            auto t_cached = std::chrono::steady_clock::now();
            try {
                language_    = core_.compile_model(model, device, cfg);
                logits_port_ = language_.output(0);
                lm_req_      = language_.create_infer_request();
                warmup();
                ready = true;
                log::info("load", had_blob
                                      ? "language model ready in %.1f s (blob cache imported, "
                                        "warmup passed)"
                                      : "language model ready in %.1f s (compiled, blob cache "
                                        "written)",
                          seconds_since(t_cached));
            } catch (const std::exception& e) {
                log::warn("load", "blob cache produced an unusable graph, discarding it: %s",
                          e.what());
            }
        }

        if (!ready) {
            // Compile from the IR with the cache switched off, so a poisoned
            // blob cannot be read back in on the retry.
            core_.set_property(ov::cache_dir(""));
            auto t_cold  = std::chrono::steady_clock::now();
            ov::AnyMap props;
            if (std::getenv("ARCINT_PROFILE") != nullptr) props[ov::enable_profiling.name()] = true;
            if (offload_ratio_ > 0) {
                // Expert offload needs the weights on disk to stream from: the
                // plugin loads them on demand rather than keeping them resident.
                props["OFFLOAD_RATIO"]        = offload_ratio_;
                props[ov::weights_path.name()] = artifact.language_model_bin;
            }
            language_    = core_.compile_model(model, device, props);
            logits_port_ = language_.output(0);
            lm_req_      = language_.create_infer_request();
            warmup();
            log::info("load", "language model ready in %.1f s (compiled from IR)",
                      seconds_since(t_cold));
        }

        if (want_mtp_) {
            try {
                // The cold path above clears the cache dir so a poisoned MoE
                // blob cannot be read back (openvino#37607). That guard does not
                // apply to these two: they are dense graphs this repository
                // generated. Without restoring it they recompile on every start,
                // and the LM head is a 248320 x 5120 MatMul.
                if (!cfg.cache_dir.empty()) core_.set_property(ov::cache_dir(cfg.cache_dir));
                const auto t0 = std::chrono::steady_clock::now();
                const std::string layer_xml = choose_mtp_layer(artifact, cfg.mtp_layer);
                mtp_layer_    = core_.compile_model(layer_xml, device);
                read_mtp_contract(mtp_layer_, layer_xml);
                mtp_head_     = core_.compile_model(artifact.mtp_lm_head_xml, device);
                lanes_[0]->mtp_layer = mtp_layer_.create_infer_request();
                lanes_[0]->mtp_head  = mtp_head_.create_infer_request();
                mtp_ready_           = true;
                log::info("mtp", "head ready in %.1f s; drafting one token per step",
                          seconds_since(t0));
            } catch (const std::exception& e) {
                log::warn("mtp", "could not load the MTP head, continuing without it: %s",
                          e.what());
                mtp_ready_ = false;
            }
        } else if (cfg.mtp == "on") {
            log::warn("mtp", "%s", "--mtp on, but this export carries no MTP head "
                                   "(run tools/export_mtp.py); continuing without it");
        }

        template_ = std::make_unique<minja::chat_template>(artifact.chat_template,
                                                           artifact.bos_token, artifact.eos_token);

        status_.id               = artifact.id;
        status_.served_id        = cfg.served_model_name.empty() ? artifact.id
                                                                 : cfg.served_model_name;
        status_.quant            = cfg.quant;
        status_.loaded           = true;
        status_.stub             = false;
        status_.n_ctx            = n_ctx > 0 ? n_ctx : artifact.n_ctx_train;
        status_.n_ctx_train      = artifact.n_ctx_train;
        status_.n_layer          = artifact.n_layer;
        status_.n_gdn_layer      = artifact.n_gdn_layer;
        status_.n_attn_layer     = artifact.n_attn_layer;
        status_.mtp_enabled      = mtp_ready_;
        status_.weights_bytes    = artifact.weights_bytes;
        status_.sampler_defaults = artifact.sampler;
    }

    // Explicit, because member destruction order alone would get this wrong:
    // a cache entry hands its KV pages back to the pool as it dies, and the
    // pool is declared after the cache, so it would be gone by then. Releasing
    // into a destroyed BlockPool is undefined behaviour that happens to look
    // like a clean shutdown most of the time.
    ~OvBackend() override {
        for (auto& lane : lanes_) {
            if (lane != nullptr) release_lane(*lane);
        }
        if (prefix_cache_ != nullptr) prefix_cache_->clear();
    }

    const ModelStatus& status() const override { return status_; }
    Tokenizer&         tokenizer() override { return *tokenizer_; }

    PrefixCacheStats cache_stats() const override {
        return prefix_cache_ != nullptr ? prefix_cache_->stats() : PrefixCacheStats{};
    }
    uint64_t free_blocks() const override {
        return pool_ != nullptr ? static_cast<uint64_t>(pool_->free_blocks()) : 0;
    }

    json template_caps() const override {
        if (template_ == nullptr) return json::object();
        const minja::chat_template_caps& c = template_->original_caps();
        // "preserve reasoning" is llama.cpp's name for a template that puts an
        // assistant turn's reasoning_content back into the prompt. Qwen3.6's
        // does; the source text says so, and no rendering probe is needed.
        const bool preserves_reasoning =
            artifact_.chat_template.find("reasoning_content") != std::string::npos;
        return json{{"supports_tools", c.supports_tools},
                    {"supports_tool_calls", c.supports_tool_calls},
                    {"supports_tool_responses", c.supports_tool_responses},
                    {"supports_system_role", c.supports_system_role},
                    {"supports_parallel_tool_calls", c.supports_parallel_tool_calls},
                    {"supports_object_arguments", c.requires_object_arguments},
                    {"supports_string_content", !c.requires_typed_content},
                    {"supports_typed_content", c.requires_typed_content},
                    {"supports_preserve_reasoning", preserves_reasoning}};
    }

    std::string render_chat(const ChatRequest& req) const override {
        minja::chat_template_inputs inputs;
        inputs.messages            = messages_json(req, template_->original_caps().requires_object_arguments);
        inputs.tools               = tools_json(req);
        inputs.add_generation_prompt = true;
        if (req.has_enable_thinking) {
            inputs.extra_context = json{{"enable_thinking", req.enable_thinking}};
        }

        minja::chat_template_options opts;
        // The artifact's template is the contract (§3.7). Polyfilling it would
        // silently render something the exporter never produced, which is the
        // template-drift failure this rule exists to prevent.
        opts.apply_polyfills = false;

        return template_->apply(inputs, opts);
    }

    FinishReason generate(const GenerationInput& in, int slot, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        if (paged_) {
            // No lock: the lane is the HTTP layer's lease, its state is its own,
            // and the only thing two lanes contend for is the device — which the
            // turnstile orders (§4.1).
            Lane& lane = *lanes_[static_cast<size_t>(
                std::min<int>(std::max(slot, 0), lane_count_ - 1))];
            return generate_paged(lane, in, on_piece, stats);
        }
        // The stateful reference path has one graph state, so it serves one
        // sequence at a time and says so at load.
        std::lock_guard<std::mutex> guard(mutex_);
        Lane&       lane  = *lanes_[0];
        using clock = std::chrono::steady_clock;

        const std::vector<int> prompt_ids = tokenizer_->encode(in.prompt);
        stats.prompt_tokens               = static_cast<int>(prompt_ids.size());
        if (prompt_ids.empty()) return FinishReason::Stop;

        // An unseeded request still gets a seed, it just gets a fresh one — and
        // it is logged, so any answer can be reproduced exactly (§3.6).
        // random_device yields 32 bits; two draws fill the 64-bit seed, and it
        // is logged at info so "reproducible" means reproducible by the operator
        // who has the default log level, not only by one running with -v.
        uint64_t seed = in.sampler.seed;
        if (!in.sampler.seeded) {
            std::random_device rd;
            seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        }
        Sampler sampler(in.sampler, seed);
        if (!in.sampler.greedy()) {
            log::info("sample", "temp %.2f top_p %.2f top_k %d seed %llu",
                      static_cast<double>(in.sampler.temperature),
                      static_cast<double>(in.sampler.top_p), in.sampler.top_k,
                      static_cast<unsigned long long>(seed));
        }

        // The prompt counts toward repetition_penalty only; the sampler keeps
        // the two scopes apart and maintains its counts incrementally.
        sampler.set_prompt(prompt_ids);

        // The drafter needs the token sequence itself, not the sampler's counts
        // — and it needs the prompt in it, because a prompt is exactly where a
        // lookup drafter finds its matches.
        std::vector<int> history;
        if (drafting_.load()) history = prompt_ids;

        // -------------------------------------------------------- prefill
        const auto t_prefill = clock::now();

        size_t past = 0;
        if (prefix_cache_ != nullptr) {
            const PrefixCache::Hit hit = prefix_cache_->lookup(prompt_ids);
            if (hit.matched_tokens > 0 && restore(lane, *hit.state)) {
                past                   = hit.matched_tokens;
                stats.cache_hit_tokens = static_cast<int>(past);
            } else {
                lm_req_.reset_state();
            }
        } else {
            lm_req_.reset_state();
        }
        if (past == 0) lm_req_.reset_state();

        // The head's own state follows the base model's. A cache hit restores it
        // along with everything else (see snapshot), so only a cold start needs
        // the cursor moved by hand.
        if (stats.cache_hit_tokens == 0) {
            mtp_reset(lane);
            mtp_seek(lane, past);
        }

        // The boundary worth remembering is the last block edge strictly inside
        // the prompt: a snapshot of the whole prompt is useless to this prompt
        // and only helps a continuation of it, which the same insert covers.
        // The checkpoint has to sit on the prefill grid, not merely on a KV block
        // edge: a hit is where a warm run starts, and a warm run must traverse
        // the same boundaries a cold one does. Restoring at a position the cold
        // path never stopped at is what made warm-vs-cold margin-backed.
        const size_t grid = prefill_chunk_ > 0 ? static_cast<size_t>(prefill_chunk_) : 0;
        const size_t snap_at =
            prefix_cache_ != nullptr && grid > 0 && prompt_ids.size() > grid
                ? ((prompt_ids.size() - 1) / grid) * grid
                : 0;

        snapshot_seconds_ = &stats.snapshot_seconds;
        step_stats_       = &stats;
        ov::Tensor logits = prefill(lane, prompt_ids, past, snap_at);
        snapshot_seconds_ = nullptr;
        stats.prefill_seconds = std::chrono::duration<double>(clock::now() - t_prefill).count();

        int next = pick(lane, sampler, logits);

        const std::vector<int>& eos = tokenizer_->eos_ids();
        auto is_eos = [&](int id) {
            return !in.sampler.ignore_eos &&
                   std::find(eos.begin(), eos.end(), id) != eos.end();
        };

        // --------------------------------------------------------- decode
        //
        // The step counters are for decode only; prefill goes through the same
        // forward() and would otherwise be counted twice.
        stats.decode_forward_seconds = 0.0;
        stats.decode_embed_seconds   = 0.0;
        stats.decode_sample_seconds  = 0.0;
        stats.decode_emit_seconds    = 0.0;
        const auto   t_decode = clock::now();
        FinishReason reason   = FinishReason::Stop;
        past                  = prompt_ids.size();

        // Emits one committed token; returns false when the caller wants to stop.
        const bool trace = std::getenv("ARCINT_TRACE_TOKENS") != nullptr;
        auto commit = [&](int tok, Control& out) {
            if (trace) log::info("trace", "commit pos=%zu tok=%d", past, tok);
            ++stats.completion_tokens;
            const auto t_emit = std::chrono::steady_clock::now();
            out = on_piece(tokenizer_->decode_one(tok), tok);
            stats.decode_emit_seconds += seconds_since(t_emit);
            sampler.observe(tok);
            if (drafting_.load()) history.push_back(tok);
            return out == Control::Continue;
        };

        while (true) {
            if (is_eos(next)) break;
            if (in.sampler.max_tokens >= 0 && stats.completion_tokens >= in.sampler.max_tokens) {
                reason = FinishReason::Length;
                break;
            }
            if (status_.n_ctx > 0 && static_cast<int>(past) + 1 >= status_.n_ctx) {
                reason = FinishReason::Length;
                break;
            }

            Control c = Control::Continue;
            if (!commit(next, c)) {
                reason = c == Control::Cancel ? FinishReason::Abort : reason;
                break;
            }

            // ------------------------------------------------ speculative step
            //
            // Only under greedy: acceptance is "the token the sampler would
            // have picked here equals the guess", which makes the output
            // identical to non-speculative greedy by construction. Under
            // sampling the same test would change the distribution, so drafting
            // stays off there rather than being approximately right.
            //
            // "The sampler would have picked" is not the raw argmax: penalties
            // are applied before greedy chooses, so repetition_penalty (1.05 by
            // default) can move the answer. Verifying on a raw argmax silently
            // diverged from non-speculative greedy at draft 8.
            std::vector<int> drafts;
            if (mtp_ready_ && in.sampler.greedy()) {
                // The head is one position behind: feeding it x_P turns its
                // pending h_{P-1} into a prediction of x_{P+1}, which is exactly
                // the token the verify pass is about to check.
                const int d = mtp_feed(lane, next, true);
                if (d >= 0) drafts.push_back(d);
            } else if (drafting_.load() && in.sampler.greedy()) {
                drafts = drafter_->draft(history, draft_tokens_);
            }

            if (drafts.empty()) {
                logits = forward(lane, {next}, past);
                ++past;
                next = pick(lane, sampler, logits);
                if (mtp_ready_) mtp_set_pending(lane, last_hidden_, 0);
                continue;
            }

            // Verify all drafts in ONE forward pass: rows 0..k predict the
            // tokens after next, d1, ... dk respectively.
            //
            // The snapshot is taken *before* the verify pass and checked here,
            // because once the pass has run the state has absorbed every draft
            // and there is no honest way back without it. Recovering after the
            // fact would mean decoding from a state that holds tokens the model
            // never committed to -- so if the snapshot fails, do not speculate.
            const auto t_snap       = clock::now();
            const bool rollbackable = snapshot_tensors(rollback_);
            stats.draft_rollback_seconds +=
                std::chrono::duration<double>(clock::now() - t_snap).count();

            if (!rollbackable) {
                log::warn("draft", "%s", "cannot snapshot the state, so a rejected draft could "
                                         "not be rolled back; disabling speculation");
                drafting_  = false;
                mtp_ready_ = false;
                logits = forward(lane, {next}, past);
                ++past;
                next = pick(lane, sampler, logits);
                continue;
            }

            std::vector<int> seq;
            seq.reserve(1 + drafts.size());
            seq.push_back(next);
            seq.insert(seq.end(), drafts.begin(), drafts.end());

            const auto t_verify = clock::now();
            logits              = forward(lane, seq, past);
            stats.draft_verify_seconds +=
                std::chrono::duration<double>(clock::now() - t_verify).count();
            stats.draft_proposed += static_cast<int>(drafts.size());

            // Verification needs one logits row per drafted position. If the
            // graph gives fewer, every draft would be rejected forever and the
            // only symptom would be a slow 0% -- so say so and stop drafting.
            const size_t rows = logits.get_size() / logits.get_shape().back();
            if (rows < seq.size()) {
                log::warn("draft",
                          "the graph returned %zu logits row(s) for %zu tokens; drafts cannot "
                          "be verified, disabling speculation",
                          rows, seq.size());
                drafting_  = false;
                mtp_ready_ = false;
                restore_tensors(rollback_);
                logits = forward(lane, {next}, past);
                ++past;
                next = pick(lane, sampler, logits);
                continue;
            }

            // Accept and commit in one pass, in the same order the plain path
            // uses. Committing as we go is what makes the penalty state correct
            // for the next verification: draft i+1 must be judged against a
            // sampler that has already seen draft i.
            size_t accepted = 0;
            bool   stop     = false;
            for (size_t i = 0; i < drafts.size(); ++i) {
                const int want = pick_row(lane, sampler, logits, i);
                if (trace) {
                    log::info("trace", "verify i=%zu next=%d draft=%d want=%d rows=%zu %s", i,
                              next, drafts[i], want, rows,
                              want == drafts[i] ? "ACCEPT" : "reject");
                }
                if (want != drafts[i]) break;
                ++accepted;

                // A drafted token clears the same gates, in the same order, as
                // one the loop head picked normally -- otherwise speculation
                // emits past EOS or past max_tokens and the answer differs.
                if (is_eos(drafts[i])) { stop = true; break; }
                if (in.sampler.max_tokens >= 0 &&
                    stats.completion_tokens >= in.sampler.max_tokens) {
                    reason = FinishReason::Length;
                    stop   = true;
                    break;
                }
                if (status_.n_ctx > 0 &&
                    static_cast<int>(past + i) + 2 >= status_.n_ctx) {
                    reason = FinishReason::Length;
                    stop   = true;
                    break;
                }
                if (!commit(drafts[i], c)) {
                    reason = c == Control::Cancel ? FinishReason::Abort : reason;
                    stop   = true;
                    break;
                }
            }
            stats.draft_accepted += static_cast<int>(accepted);
            if (stop) {
                // Leaving here means the state still holds the whole drafted
                // tail, including tokens that were never committed. Nothing
                // reads it today -- every entry path rebuilds the state first --
                // but "safe because of what the callers happen to do" is not a
                // property worth relying on, and the rollback buffer is already
                // in hand.
                restore_tensors(rollback_);
                break;
            }

            if (accepted < drafts.size()) {
                // Over-advanced by the rejected tail. Roll the state back and
                // re-run only what was accepted; without this the cache would
                // hold tokens the model never committed to.
                const auto t_restore = clock::now();
                restore_tensors(rollback_);
                stats.draft_rollback_seconds +=
                    std::chrono::duration<double>(clock::now() - t_restore).count();
                std::vector<int> keep(seq.begin(), seq.begin() + static_cast<long>(1 + accepted));
                const auto t_re = clock::now();
                logits          = forward(lane, keep, past);
                stats.draft_reforward_seconds +=
                    std::chrono::duration<double>(clock::now() - t_re).count();
            }

            // Sample before the head runs. `logits` is the language model
            // request's own output tensor, and mtp_feed() infers two other
            // requests; reading it afterwards returned a token the model never
            // predicted, which is how a drafted token got inserted into an
            // otherwise identical answer.
            past += 1 + accepted;
            next = pick(lane, sampler, logits);

            if (mtp_ready_) {
                // hidden here covers the positions actually committed: row 0 is
                // position `past`, row i is position `past + i`. It is a copy
                // (see forward), so feeding the head cannot disturb it.
                for (size_t i = 0; i < accepted; ++i) {
                    mtp_set_pending(lane, last_hidden_, i);
                    mtp_feed(lane, drafts[i], false);
                }
                mtp_set_pending(lane, last_hidden_, accepted);
            }
        }

        stats.decode_seconds = std::chrono::duration<double>(clock::now() - t_decode).count();
        return reason;
    }

private:
    // Chunked prefill (§3.2). NOT byte-identical to a single call: measured
    // 2026-08-28, forward([x, y]) and forward([x]) then forward([y]) differ at
    // the last position by up to 0.013 in the logits, so any change in how a
    // sequence is split can flip a near-tie. Chunking is used by default anyway
    // because the alternative is unbounded activation memory, and the size is
    // large enough (2048) that ordinary prompts are a single chunk.
    // `snapshot_seconds_` is where prefill reports what the cache checkpoint
    // cost; it points at the caller's stats for the duration of one request.
    double*          snapshot_seconds_ = nullptr;
    GenerationStats* step_stats_        = nullptr;

    ov::Tensor prefill(Lane& lane, const std::vector<int>& tokens, size_t past,
                       size_t snapshot_at) {
        ov::Tensor   logits;
        const size_t grid = prefill_chunk_ > 0 ? static_cast<size_t>(prefill_chunk_) : 0;

        while (past < tokens.size()) {
            // Chunk on an ABSOLUTE grid -- multiples of the chunk size counted
            // from position 0 -- rather than in steps of `chunk` from wherever
            // this call happens to start.
            //
            // This is what makes a warm run equal a cold one. §3.2 measured that
            // advancing the state by k tokens in one call differs from advancing
            // it k times by one, so a cache hit, which is a boundary, would
            // otherwise hand the model a different split of the same tokens than
            // the cold run saw. Measured on the dense model: cold-vs-warm logits
            // differed by up to 0.210 with a top-2 margin of 0.145 -- the gate
            // was passing on margin, not by construction. With both paths on the
            // same grid the difference is exactly zero.
            size_t take = tokens.size() - past;
            if (grid > 0) {
                const size_t edge = ((past / grid) + 1) * grid;
                take = std::min(edge, tokens.size()) - past;
            }

            const size_t start = past;
            logits = forward(lane, {tokens.begin() + static_cast<long>(past),
                              tokens.begin() + static_cast<long>(past + take)},
                             past);
            past += take;

            // Prime the head over the prompt: position t is fed once x_{t+1} is
            // known, which inside a chunk it always is.
            if (mtp_ready_) {
                for (size_t i = 0; i < take; ++i) {
                    if (lane.mtp_has_pending) mtp_feed(lane, tokens[start + i], false);
                    mtp_set_pending(lane, last_hidden_, i);
                }
            }

            if (prefix_cache_ != nullptr && past == snapshot_at) {
                // Serialising the whole graph state is expensive and the KV half
                // grows with the prefix, so ask first whether it could be kept
                // at all rather than copying hundreds of MiB to have it dropped.
                if (prefix_cache_->may_accept(estimated_state_bytes(lane))) {
                    const auto             t_snap = std::chrono::steady_clock::now();
                    PrefixCache::StateBlob blob;
                    const bool             took = snapshot(lane, blob);
                    if (snapshot_seconds_ != nullptr) *snapshot_seconds_ += seconds_since(t_snap);
                    if (took) prefix_cache_->insert(tokens, past, std::move(blob));
                }
            }
        }
        return logits;
    }

    // Rollback needs a copy of the state, not a *serialised* copy of it: the
    // prefix cache's blob format exists so an entry can be validated against a
    // shape it no longer knows, which a buffer that lives for one decode step
    // never needs. Skipping the header and the second memcpy makes speculation
    // pay only for the device read.
    bool snapshot_tensors(std::vector<ov::Tensor>& out) {
        try {
            std::vector<ov::VariableState> states = lm_req_.query_state();
            out.resize(states.size());
            for (size_t i = 0; i < states.size(); ++i) {
                const ov::Tensor src = states[i].get_state();
                if (!out[i] || out[i].get_element_type() != src.get_element_type() ||
                    out[i].get_shape() != src.get_shape()) {
                    out[i] = ov::Tensor(src.get_element_type(), src.get_shape());
                }
                src.copy_to(out[i]);
            }
            if (!logged_rollback_size_) {
                logged_rollback_size_ = true;
                size_t bytes = 0;
                for (const ov::Tensor& t : out) bytes += t.get_byte_size();
                log::info("draft", "speculation rolls back %.1f MiB of state per decode step "
                                   "across %zu variables",
                          static_cast<double>(bytes) / (1024.0 * 1024.0), out.size());
            }
            return true;
        } catch (const std::exception& e) {
            log::warn("draft", "state snapshot failed, speculation disabled: %s", e.what());
            out.clear();
            return false;
        }
    }

    bool restore_tensors(const std::vector<ov::Tensor>& in) {
        std::vector<ov::VariableState> states = lm_req_.query_state();
        if (states.size() != in.size()) return false;
        try {
            for (size_t i = 0; i < states.size(); ++i) {
                // set_state() copies into the request's own buffer, so the
                // scratch tensor stays ours to overwrite next round.
                states[i].set_state(in[i]);
            }
            return true;
        } catch (const std::exception& e) {
            log::warn("draft", "state restore failed: %s", e.what());
            return false;
        }
    }

    size_t estimated_paged_state_bytes(Lane& lane) {
        size_t n = la_row_bytes_ + 64 * la_state_names_.size();
        if (mtp_ready_) {
            try {
                for (ov::VariableState& v : lane.mtp_layer.query_state()) {
                    n += v.get_state().get_byte_size() + 64;
                }
            } catch (const std::exception&) {}
            n += 256 + static_cast<size_t>(artifact_.n_embd) * 4;
        }
        return n;
    }

    // The paged cache blob: [one row of every LA table][the head's variables,
    // cursor and pending row]. The KV is not in it and is not copied — the
    // entry holds references to the pages themselves (§3.4), which is what
    // replaced the single-slot pool-epoch tag at M6. The GDN half travels
    // host-side because it is fixed-size and small; the KV half stays on the
    // card because it is neither.
    bool snapshot_paged(Lane& lane, size_t row, PrefixCache::StateBlob& out) {
        try {
            out.clear();
            for (auto& b : read_paged_row(lane, row)) out.push_back(std::move(b));
            if (mtp_ready_) {
                for (ov::VariableState& v : lane.mtp_layer.query_state()) {
                    out.push_back(serialise_state(v.get_state()));
                }
                ov::Tensor cursor(ov::element::i64, ov::Shape{3});
                cursor.data<int64_t>()[0] = static_cast<int64_t>(lane.mtp_len);
                cursor.data<int64_t>()[1] = static_cast<int64_t>(lane.mtp_pos);
                cursor.data<int64_t>()[2] = lane.mtp_has_pending ? 1 : 0;
                out.push_back(serialise_state(cursor));
                out.push_back(serialise_state(
                    lane.mtp_has_pending ? lane.mtp_pending
                                     : ov::Tensor(ov::element::f32, ov::Shape{1, 1, 1})));
            }
            return true;
        } catch (const std::exception& e) {
            log::warn("cache", "paged snapshot failed, continuing without caching: %s", e.what());
            out.clear();
            return false;
        }
    }

    bool restore_paged(Lane& lane, size_t row, const PrefixCache::StateBlob& blob) {
        const size_t tables = la_state_names_.size();
        size_t want = tables;
        std::vector<ov::VariableState> mtp_states;
        if (mtp_ready_) {
            mtp_states = lane.mtp_layer.query_state();
            want += mtp_states.size() + 2;
        }
        if (blob.size() != want) return false;
        try {
            std::vector<std::vector<uint8_t>> rows(blob.begin(),
                                                   blob.begin() + static_cast<long>(tables));
            write_paged_row(lane, row, rows);
            if (mtp_ready_) {
                for (size_t i = 0; i < mtp_states.size(); ++i) {
                    mtp_states[i].set_state(deserialise_state(blob[tables + i]));
                }
                const ov::Tensor cursor = deserialise_state(blob[blob.size() - 2]);
                lane.mtp_len         = static_cast<size_t>(cursor.data<const int64_t>()[0]);
                lane.mtp_pos         = static_cast<size_t>(cursor.data<const int64_t>()[1]);
                lane.mtp_has_pending = cursor.data<const int64_t>()[2] != 0;
                if (lane.mtp_has_pending) lane.mtp_pending = deserialise_state(blob.back());
            }
            return true;
        } catch (const std::exception& e) {
            log::warn("cache", "paged restore failed, falling back to a cold prefill: %s",
                      e.what());
            return false;
        }
    }

    FinishReason generate_paged(Lane& lane, const GenerationInput& in,
                                const TokenCallback& on_piece, GenerationStats& stats) {
        using clock = std::chrono::steady_clock;

        const std::vector<int> prompt_ids = tokenizer_->encode(in.prompt);
        stats.prompt_tokens               = static_cast<int>(prompt_ids.size());
        if (prompt_ids.empty()) return FinishReason::Stop;

        uint64_t seed = in.sampler.seed;
        if (!in.sampler.seeded) {
            std::random_device rd;
            seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        }
        Sampler sampler(in.sampler, seed);
        if (!in.sampler.greedy()) {
            log::info("sample", "temp %.2f top_p %.2f top_k %d seed %llu",
                      static_cast<double>(in.sampler.temperature),
                      static_cast<double>(in.sampler.top_p), in.sampler.top_k,
                      static_cast<unsigned long long>(seed));
        }
        sampler.set_prompt(prompt_ids);
        std::vector<int> history;
        if (drafting_.load()) history = prompt_ids;

        // The lane's pages go back to the pool however this request ends —
        // finished, stopped, cancelled or thrown out of.
        LaneReset reset(*this, lane, stats);

        // ------------------------------------------------------------ prefill
        const auto t_prefill = clock::now();
        const auto t_restore = t_prefill;
        size_t&    c    = lane.committed_row;      // the committed LA row, lane-relative
        c               = 0;
        size_t     past = 0;
        bool       warm = false;
        if (prefix_cache_ != nullptr) {
            PrefixCache::Hit hit = prefix_cache_->lookup(prompt_ids);
            // Every page of a hit is a complete page, and a complete page is
            // never written again, so mapping one needs no copy — only a
            // reference, taken before the entry can be evicted (§3.3).
            // Exactly the pages of the matched prefix, no more: a longer list
            // would hand this lane a shared page *past* the prefix, which it
            // would then write into — and "a complete page is never written
            // again" is the invariant that makes sharing safe without copying.
            // A tiered hit has its pages on the host (DESIGN §4.4): allocate,
            // copy them back, and only then does the entry hold pages again.
            if (hit.matched_tokens > 0 && hit.tiered && hit.matched_tokens % kv_block_tokens_ == 0) {
                const auto t_promote = clock::now();
                if (promote_entry(hit)) {
                    stats.cache_promote_seconds =
                        std::chrono::duration<double>(clock::now() - t_promote).count();
                } else {
                    log::warn("cache", "host-tier entry of %zu tok could not be promoted; cold prefill",
                              hit.matched_tokens);
                    hit = PrefixCache::Hit{};
                }
            }
            if (hit.matched_tokens > 0 && hit.blocks != nullptr &&
                hit.matched_tokens % kv_block_tokens_ == 0 &&
                hit.blocks->size() == hit.matched_tokens / kv_block_tokens_) {
                pool_->ref(*hit.blocks);
                lane.blocks   = *hit.blocks;
                lane.hit_keep = hit.keep;
                if (restore_paged(lane, c, *hit.state)) {
                    past                   = hit.matched_tokens;
                    stats.cache_hit_tokens = static_cast<int>(past);
                    warm                   = true;
                } else {
                    pool_->release(lane.blocks);
                    lane.blocks.clear();
                    lane.hit_keep.reset();
                }
            }
        }
        dflash_reset(lane);
        if (!warm) {
            zero_paged_rows(lane);
            mtp_reset(lane);
        }
        stats.prefill_restore_seconds = std::chrono::duration<double>(clock::now() - t_restore)
                                            .count();

        const size_t grid = prefill_chunk_ > 0 ? static_cast<size_t>(prefill_chunk_) : 0;
        // The snapshot grid is finer than the chunk grid (--cache-grid): the
        // chunk that contains the snapshot point is cut there, so the state at
        // snap_at exists to be captured. That cut is the one place the warm
        // continuation's forward boundaries can differ from a cold run's, and
        // whether the paged kernels are exact across it is gated, not assumed
        // (DESIGN 7.0.2j; --cache-grid 0 restores the chunk grid).
        const size_t sgrid = cache_grid_ > 0 ? cache_grid_ : grid;
        const size_t snap_at =
            prefix_cache_ != nullptr && sgrid > 0 && prompt_ids.size() > sgrid
                ? ((prompt_ids.size() - 1) / sgrid) * sgrid
                : 0;

        const bool trace = std::getenv("ARCINT_TRACE_TOKENS") != nullptr;
        ov::Tensor logits;
        while (past < prompt_ids.size()) {
            size_t take = prompt_ids.size() - past;
            if (grid > 0) {
                const size_t edge = ((past / grid) + 1) * grid;
                take = std::min(edge, prompt_ids.size()) - past;
            }
            if (snap_at > past && snap_at < past + take) take = snap_at - past;
            const auto t_blocks = clock::now();
            const bool got      = ensure_blocks(lane, past + take);
            stats.prefill_blocks_seconds += seconds_since_tp(t_blocks);
            if (!got) {
                log::warn("slot", "KV pool exhausted at %zu tokens; %llu page(s) free",
                          past + take, static_cast<unsigned long long>(pool_->free_blocks()));
                stats.prefill_seconds =
                    std::chrono::duration<double>(clock::now() - t_prefill).count();
                return FinishReason::Length;
            }
            const std::vector<int> chunk(prompt_ids.begin() + static_cast<long>(past),
                                         prompt_ids.begin() + static_cast<long>(past + take));
            // Each phase net of what it spent queueing, so the terms add up to
            // work and the waiting is reported as waiting.
            double     waited = lane.stall_accum;
            const auto t_emb  = clock::now();
            const ov::Tensor emb  = embed_paged(lane, chunk);
            stats.prefill_embed_seconds +=
                seconds_since_tp(t_emb) - (lane.stall_accum - waited);
            waited                = lane.stall_accum;
            const auto t_fwd      = clock::now();
            const bool       last = past + take == prompt_ids.size();
            const bool wfeats =
                dflash_ready_ && past + take + kDflashWindow >= prompt_ids.size();
            const ov::Tensor out =
                paged_forward(lane, emb, past, {static_cast<int32_t>(c)}, 0, last, wfeats);
            stats.prefill_forward_seconds +=
                seconds_since_tp(t_fwd) - (lane.stall_accum - waited);
            if (last) logits = out;
            if (mtp_ready_) {
                mtp_prime_paged(lane, lane.hidden, emb, take);
            }
            if (dflash_ready_ && wfeats) {
                const size_t from = prompt_ids.size() > kDflashWindow &&
                                            past < prompt_ids.size() - kDflashWindow
                                        ? prompt_ids.size() - kDflashWindow - past
                                        : 0;
                dflash_append(lane, from, take - from, past + from);
            }
            past += take;

            if (prefix_cache_ != nullptr && past == snap_at &&
                prefix_cache_->may_accept(estimated_paged_state_bytes(lane))) {
                const auto             t_snap = clock::now();
                PrefixCache::StateBlob blob;
                const bool             took = snapshot_paged(lane, c, blob);
                stats.snapshot_seconds += std::chrono::duration<double>(clock::now() - t_snap)
                                              .count();
                if (took) {
                    // The pages of the prefix travel with the entry, and the
                    // cache holds its own reference to them: this lane may
                    // finish and free its table while the other is still using
                    // the entry.
                    std::vector<int32_t> kept(
                        lane.blocks.begin(),
                        lane.blocks.begin() + static_cast<long>(past / kv_block_tokens_));
                    pool_->ref(kept);
                    prefix_cache_->insert(prompt_ids, past, std::move(blob), std::move(kept));
                }
            }
        }
        stats.prefill_seconds = std::chrono::duration<double>(clock::now() - t_prefill).count();
        stats.prefill_wait_seconds = lane.stall_accum;

        int next = pick(lane, sampler, logits);

        const std::vector<int>& eos = tokenizer_->eos_ids();
        auto is_eos = [&](int id) {
            return !in.sampler.ignore_eos &&
                   std::find(eos.begin(), eos.end(), id) != eos.end();
        };

        // ------------------------------------------------------------- decode
        //
        // Prefill did its own waiting, and that is TTFT rather than a stall
        // between tokens; the accumulator starts clean here.
        lane.stall_accum             = 0.0;
        stats.decode_forward_seconds = 0.0;
        stats.decode_embed_seconds   = 0.0;
        stats.decode_sample_seconds  = 0.0;
        stats.decode_emit_seconds    = 0.0;
        const auto   t_decode = clock::now();
        FinishReason reason   = FinishReason::Stop;
        past                  = prompt_ids.size();

        auto commit = [&](int tok, Control& out) {
            if (trace) log::info("trace", "commit pos=%zu tok=%d", past, tok);
            // One sample per token: the gap a reader feels is the sum of the
            // waits of every execution that produced this token, not the wait
            // before one of them.
            if (lane.stall_accum > 0.0) {
                lane.stalls.push_back(lane.stall_accum);
                lane.stall_accum = 0.0;
            }
            ++stats.completion_tokens;
            const auto t_emit = clock::now();
            out = on_piece(tokenizer_->decode_one(tok), tok);
            stats.decode_emit_seconds += seconds_since_tp(t_emit);
            sampler.observe(tok);
            if (drafting_.load()) history.push_back(tok);
            return out == Control::Continue;
        };

        while (true) {
            if (is_eos(next)) break;
            if (in.sampler.max_tokens >= 0 && stats.completion_tokens >= in.sampler.max_tokens) {
                reason = FinishReason::Length;
                break;
            }
            if (status_.n_ctx > 0 && static_cast<int>(past) + 1 >= status_.n_ctx) {
                reason = FinishReason::Length;
                break;
            }
            Control c_ctl = Control::Continue;
            if (!commit(next, c_ctl)) {
                reason = c_ctl == Control::Cancel ? FinishReason::Abort : reason;
                break;
            }

            std::vector<int> drafts;
            if (dflash_ready_ && in.sampler.greedy()) {
                drafts = dflash_draft(lane, next, past);
            } else if (mtp_ready_ && in.sampler.greedy()) {
                const int d = mtp_feed(lane, next, true);
                if (d >= 0) drafts.push_back(d);
            } else if (drafting_.load() && in.sampler.greedy()) {
                drafts = drafter_->draft(history, draft_tokens_);
                if (drafts.size() > drafts_max_) drafts.resize(drafts_max_);
            }

            // One page every kv_block_tokens_ tokens, so the pool lock is
            // touched once per 16 or 32 decode steps rather than per step.
            if (!ensure_blocks(lane, past + 1 + drafts_max_)) {
                log::warn("slot", "KV pool exhausted at %zu tokens; ending the stream",
                          past);
                reason = FinishReason::Length;
                break;
            }

            if (drafts.empty()) {
                // Each phase is timed net of what it spent queueing for the
                // device, so the breakdown adds up to work and the waiting is
                // reported as waiting.
                double     waited = lane.stall_accum;
                const auto t_emb  = clock::now();
                const ov::Tensor emb1 = embed_paged(lane, {next});
                stats.decode_embed_seconds +=
                    seconds_since_tp(t_emb) - (lane.stall_accum - waited);
                waited           = lane.stall_accum;
                const auto t_fwd = clock::now();
                logits = paged_forward(lane, emb1, past, {static_cast<int32_t>(c)}, 0);
                stats.decode_forward_seconds +=
                    seconds_since_tp(t_fwd) - (lane.stall_accum - waited);
                ++past;
                const auto t_smp = clock::now();
                next = pick(lane, sampler, logits);
                stats.decode_sample_seconds += seconds_since_tp(t_smp);
                if (mtp_ready_) {
                    mtp_set_pending(lane, lane.hidden, 0);
                }
                continue;
            }

            // Verify with per-token checkpoints: rows [c, s0..sk], interval 1.
            // Rollback is deciding which checkpoint row is the new committed
            // row -- no state bytes move, no re-forward runs.
            std::vector<int> seq;
            seq.reserve(1 + drafts.size());
            seq.push_back(next);
            seq.insert(seq.end(), drafts.begin(), drafts.end());
            std::vector<int32_t> la_rows{static_cast<int32_t>(c)};
            for (size_t rrow = 0; la_rows.size() < seq.size() + 1; ++rrow) {
                if (rrow != c) la_rows.push_back(static_cast<int32_t>(rrow));
            }

            const auto t_verify = clock::now();
            logits = paged_forward(lane, embed_paged(lane, seq), past, la_rows, 1);
            stats.draft_verify_seconds += seconds_since_tp(t_verify);
            stats.draft_proposed += static_cast<int>(drafts.size());

            const size_t rows = logits.get_size() / logits.get_shape().back();
            if (rows < seq.size()) {
                log::warn("draft", "the graph returned %zu logits row(s) for %zu tokens; "
                                   "disabling speculation", rows, seq.size());
                drafting_  = false;
                mtp_ready_ = false;
                // the pass advanced the state; its last checkpoint is the truth
                c = static_cast<size_t>(la_rows.back());
                past += seq.size();
                next = pick(lane, sampler, logits);
                continue;
            }

            size_t accepted = 0;
            bool   stop     = false;
            for (size_t i = 0; i < drafts.size(); ++i) {
                const int want = pick_row(lane, sampler, logits, i);
                if (trace) {
                    log::info("trace", "verify i=%zu next=%d draft=%d want=%d %s", i, next,
                              drafts[i], want, want == drafts[i] ? "ACCEPT" : "reject");
                }
                if (want != drafts[i]) break;
                ++accepted;
                if (is_eos(drafts[i])) { stop = true; break; }
                if (in.sampler.max_tokens >= 0 &&
                    stats.completion_tokens >= in.sampler.max_tokens) {
                    reason = FinishReason::Length;
                    stop   = true;
                    break;
                }
                if (status_.n_ctx > 0 && static_cast<int>(past + i) + 2 >= status_.n_ctx) {
                    reason = FinishReason::Length;
                    stop   = true;
                    break;
                }
                if (!commit(drafts[i], c_ctl)) {
                    reason = c_ctl == Control::Cancel ? FinishReason::Abort : reason;
                    stop   = true;
                    break;
                }
            }
            stats.draft_accepted += static_cast<int>(accepted);
            if (stop) {
                c = static_cast<size_t>(la_rows[1 + accepted]);
                break;
            }

            if (mtp_ready_) {
                const ov::Tensor hid = lane.hidden;
                for (size_t i = 0; i < accepted; ++i) {
                    mtp_set_pending(lane, hid, i);
                    mtp_feed(lane, drafts[i], false);
                }
                mtp_set_pending(lane, hid, accepted);
            }
            // The accepted rows (anchor + accepted drafts) are now context; only
            // they enter the drafter's state, so there is nothing to roll back.
            if (dflash_ready_) dflash_append(lane, 0, accepted + 1, past);

            c = static_cast<size_t>(la_rows[1 + accepted]);   // promotion
            past += 1 + accepted;
            next = pick_row(lane, sampler, logits, accepted);
        }

        stats.decode_seconds = std::chrono::duration<double>(clock::now() - t_decode).count();
        return reason;
    }

    static double seconds_since_tp(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    // =================================================================== paged
    //
    // The measured path (tools/paged_spec.py is the oracle). Requirements it
    // proved, ported as invariants: state rows zeroed before first use (the
    // kernels read the committed row even at past_lens 0), tables allocated as
    // device-resident remote tensors set once, the big model compiled before
    // the embeddings (compile-time peak), speculative rollback as checkpoint-
    // row promotion with zero state bytes moved, admission by a reservation in
    // which every term is measured.

    size_t device_resident_bytes(const std::string& device) {
        size_t out = 0;
        for (const auto& [k, v] : core_.get_property(device, ov::intel_gpu::memory_statistics)) {
            if (k == "usm_device" || k == "cl_mem") out += v;
        }
        return out;
    }

    void load_paged(const Config& cfg, int req_n_ctx) {
        paged_ = true;
        const std::string& device  = cfg.device;
        const std::string  emb_dev = cfg.emb_device.empty() ? device : cfg.emb_device;
        const std::string  mtp_dev = cfg.mtp_device.empty() ? device : cfg.mtp_device;

        std::shared_ptr<ov::Model> model = core_.read_model(artifact_.language_model_xml);
        // --gate-pad N: widen the shared-expert gate (see pad_gate_matmuls). A
        // deployment choice with a known price, like --paged-kv: DESIGN 7.0.2g
        // has the break-even. Off by default for this fleet's answer lengths.
        if (cfg.gate_pad > 0) {
            const size_t n = pad_gate_matmuls(model, static_cast<size_t>(cfg.gate_pad));
            log::info("load", "shared-expert gate padded to N=%d on %zu MatMul(s)", cfg.gate_pad, n);
        }

        // The LA state geometry is read off the *stateful* graph's variables,
        // where it is static; the transformed ports leave some dims dynamic.
        std::vector<ov::Shape> conv_proto, gdn_proto;
        for (const auto& var : model->get_variables()) {
            const ov::PartialShape& ps = var->get_info().data_shape;
            if (ps.rank().is_dynamic()) continue;
            const int64_t rank = ps.rank().get_length();
            bool tail_static = true;
            for (int64_t i = 1; i < rank; ++i) tail_static &= ps[i].is_static();
            if (!tail_static) continue;                    // attention KV: dynamic seq dim
            ov::Shape sh;
            sh.push_back(1);
            for (int64_t i = 1; i < rank; ++i) sh.push_back(ps[i].get_length());
            if (rank == 3) conv_proto.push_back(sh);
            if (rank == 4) gdn_proto.push_back(sh);
        }

        {
            ov::pass::Manager pm;
            pm.register_pass<ov::pass::SDPAToPagedAttention>();
            pm.run_passes(model);
        }

        want_dflash_ = !cfg.dflash.empty();
        // One drafter per verify loop: an explicit --mtp on + --dflash is a
        // config error; mtp auto yields to the requested drafter.
        want_mtp_ = cfg.mtp != "off" && artifact_.has_mtp_head && !want_dflash_;
        if (want_mtp_ && !expose_hidden_state(model)) {
            log::warn("mtp", "%s", "could not expose the hidden state; MTP disabled");
            want_mtp_ = false;
        }
        if (want_dflash_) {
            try {
                const nlohmann::json dcfg =
                    nlohmann::json::parse(dflash_read_file(cfg.dflash + "/config.json"))
                        .at("dflash_config");
                dflash_block_      = dcfg.at("block_size").get<size_t>();
                dflash_mask_token_ = dcfg.at("mask_token_id").get<int>();
                dflash_topk_       = dcfg.at("selector_top_k").get<size_t>();
                std::vector<int64_t> ids = dcfg.at("target_layer_ids").get<std::vector<int64_t>>();
                if (!expose_dflash_feats(model, ids)) {
                    throw std::runtime_error("no tap for one of the target layers");
                }
            } catch (const std::exception& e) {
                log::warn("dflash", "cannot wire the drafter, continuing without it: %s",
                          e.what());
                want_dflash_ = false;
            }
        }
        drafts_max_ = std::max<size_t>(draft_tokens_, want_mtp_ ? 1 : 0);
        if (want_dflash_) drafts_max_ = std::max(drafts_max_, dflash_block_ - 1);
        rows_per_lane_ = drafts_max_ + 3;
        if (cfg.slice_logits) {
            const int64_t keep = static_cast<int64_t>(1 + drafts_max_);
            // Token axis 0: the paged export's hidden state is [tokens, 1, hidden].
            // Stated here, verified below by the probe's first forward.
            if (slice_logits_to_last_token(model, keep, 0)) {
                logits_keep_rows_ = static_cast<size_t>(keep);
                log::info("load", "logits sliced to the last %lld row(s)",
                          static_cast<long long>(keep));
            } else {
                log::warn("load", "%s", "logits NOT sliced: every prefill chunk will "
                                        "compute and copy [M, vocab] logits");
            }
        }
        // What the graph will actually hand back per execution. A 2 GiB
        // device-to-host copy per prefill chunk (2026-08-30) turned out to be
        // the full [M, vocab] logits leaving the card while this log claimed
        // they were sliced to one row -- so the claim is now checked against
        // the model's own output ports rather than against the rewrite's
        // return value.
        for (const auto& port : model->outputs()) {
            std::ostringstream os;
            os << port.get_partial_shape();
            log::info("load", "paged output '%s' %s %s", port.get_any_name().c_str(),
                      port.get_element_type().get_type_name().c_str(), os.str().c_str());
        }

        offload_ratio_ = cfg.offload_ratio;
        ov::AnyMap props;
        // u8 KV is the default, by §7.0.3's protocol run to completion on the
        // C++ endpoint (2026-08-29): 10/10 on the harness at base depth AND at
        // the ~30k depth probe, base answer bitwise identical to f16, never
        // slower (71.3 vs 68.6 t/s), and half the KV memory. ARCINT_PAGED_KV
        // stays as the measurement switch to pin f16 for A/B runs.
        // --paged-kv decides it; ARCINT_PAGED_KV still overrides, because an A/B
        // over a running deployment should not need a config edit.
        ov::element::Type kv_prec = cfg.paged_kv == "f16" ? ov::element::f16 : ov::element::u8;
        std::string       kv_src  = "--paged-kv";
        if (const char* env = std::getenv("ARCINT_PAGED_KV")) {
            kv_prec = std::string(env) == "f16" ? ov::element::f16 : ov::element::u8;
            kv_src  = "ARCINT_PAGED_KV";
        }
        // Said out loud at load, because it is a throughput decision now and not
        // only a memory one: u8 halves KV and costs up to 22% of prefill at
        // depth (§7.0.3 chose it on decode evidence, which did not see that).
        log::info("load", "paged KV precision %s (%s): %s",
                  kv_prec == ov::element::f16 ? "f16" : "u8", kv_src.c_str(),
                  kv_prec == ov::element::f16
                      ? "20 KiB/token, faster prefill at depth"
                      : "11.3 KiB/token, what makes two lanes at depth fit");
        props["KV_CACHE_PRECISION"] = kv_prec;
        // The served path could not be profiled at all until now: enable_profiling
        // was wired on the stateful reference path only, so every per-kernel number
        // in this document describes a graph the fleet does not run. ARCINT_PROFILE
        // turns it on here too. It is opt-in because PERF_COUNT inflates wall clock
        // (§7.0: 53.6 ms for a step whose node sum is 19.01 ms), so a profiled run
        // is for shares, never for rates.
        if (std::getenv("ARCINT_PROFILE") != nullptr) props[ov::enable_profiling.name()] = true;
        if (offload_ratio_ > 0) {
            props["OFFLOAD_RATIO"]        = offload_ratio_;
            props[ov::weights_path.name()] = artifact_.language_model_bin;
        }
        // The blob cache is off for the paged graph: its import path is
        // unproven and the #37607 class is exactly the kind of thing it would
        // hide. Revisit once the paged path is the only path.
        core_.set_property(ov::cache_dir(""));

        log::info("load", "compiling PAGED language model on %s (big model first by design)",
                  device.c_str());
        auto t0      = std::chrono::steady_clock::now();
        paged_model_ = core_.compile_model(model, device, props);
        const size_t resident_base = device_resident_bytes(device);
        log::info("load", "paged model ready in %.1f s; device-resident %.2f GiB",
                  seconds_since(t0), static_cast<double>(resident_base) / (1u << 30));

        // One InferRequest per lane, all from the one CompiledModel. Weights are
        // NOT duplicated by this (measured 2026-08-29: a second request adds
        // activations, a second *compile* adds 0.791 GiB of weights, §7.0.2),
        // which is the whole reason a second lane is affordable at all.
        for (int i = 0; i < lane_count_; ++i) {
            lanes_.push_back(std::make_unique<Lane>());
            lanes_.back()->index = i;
            lanes_.back()->req   = paged_model_.create_infer_request();
        }

        embeddings_ = core_.compile_model(artifact_.text_embeddings_xml, emb_dev);
        for (auto& lane : lanes_) lane->embed = embeddings_.create_infer_request();
        log::info("load", "embeddings on %s", emb_dev.c_str());

        if (want_mtp_) {
            try {
                const std::string layer_xml = choose_mtp_layer(artifact_, cfg.mtp_layer);
                mtp_layer_ = core_.compile_model(layer_xml, mtp_dev);
                read_mtp_contract(mtp_layer_, layer_xml);
                mtp_head_  = core_.compile_model(artifact_.mtp_lm_head_xml, mtp_dev);
                for (auto& lane : lanes_) {
                    lane->mtp_layer = mtp_layer_.create_infer_request();
                    lane->mtp_head  = mtp_head_.create_infer_request();
                }
                mtp_ready_ = true;
                log::info("mtp", "head on %s, drafting one token per step", mtp_dev.c_str());
            } catch (const std::exception& e) {
                log::warn("mtp", "could not load the MTP head, continuing without it: %s",
                          e.what());
                mtp_ready_ = false;
            }
        } else if (cfg.mtp == "on") {
            log::warn("mtp", "%s", "--mtp on, but this export carries no MTP head");
        }

        if (want_dflash_) {
            try {
                const std::string ddev =
                    cfg.dflash_device.empty() ? device : cfg.dflash_device;
                // Plain f16 execution: the export carries the residual-stream
                // range fixes (a 1/4 fold into the writers and per-norm
                // pre-scales with eps*pre^2 -- exact rms identities), because
                // the raw head peaks at ~128k and f16 tops out at 65504. The
                // GPU-f16 parity gate is cycle-exact with CPU (2026-09-01).
                dflash_model_ = core_.compile_model(
                    cfg.dflash + "/openvino_dflash_draft_stateful.xml", ddev);
                if (artifact_.mtp_lm_head_xml.empty()) {
                    throw std::runtime_error(
                        "the drafter scores through the target lm_head and this artifact "
                        "carries no extracted openvino_mtp_lm_head.xml");
                }
                dflash_head_model_ = core_.compile_model(artifact_.mtp_lm_head_xml, ddev);
                load_dflash_selector(cfg.dflash);
                for (auto& lane : lanes_) {
                    lane->dflash_req  = dflash_model_.create_infer_request();
                    lane->dflash_head = dflash_head_model_.create_infer_request();
                }
                // the mask token's embedding, once
                {
                    ov::Tensor ids(ov::element::i64, ov::Shape{1, 1});
                    ids.data<int64_t>()[0] = dflash_mask_token_;
                    lanes_[0]->embed.set_input_tensor(ids);
                    lanes_[0]->embed.infer();
                    const ov::Tensor src = lanes_[0]->embed.get_output_tensor(0);
                    dflash_mask_embed_ = ov::Tensor(src.get_element_type(), src.get_shape());
                    std::memcpy(dflash_mask_embed_.data(), src.data(), src.get_byte_size());
                }
                dflash_ready_ = true;
                log::info("dflash", "block-%zu drafter on %s (%zu drafts per verify pass)",
                          dflash_block_, ddev.c_str(), dflash_block_ - 1);
            } catch (const std::exception& e) {
                log::warn("dflash", "could not load the drafter, continuing without it: %s",
                          e.what());
                dflash_ready_ = false;
            }
        }

        // ---- ports: state tables and KV pools --------------------------------
        size_t conv_i = 0, gdn_i = 0, kv_block_bytes = 0;
        for (const auto& port : paged_model_.inputs()) {
            const std::string  name = port.get_any_name();
            const ov::PartialShape& ps = port.get_partial_shape();
            if (name.rfind("conv_state_table.", 0) == 0) {
                if (conv_i >= conv_proto.size()) throw std::runtime_error("conv table mismatch");
                la_state_names_.push_back(name);
                la_state_shapes_.push_back(conv_proto[conv_i++ % conv_proto.size()]);
            } else if (name.rfind("gated_delta_state_table.", 0) == 0) {
                if (gdn_i >= gdn_proto.size()) throw std::runtime_error("gdn table mismatch");
                la_state_names_.push_back(name);
                la_state_shapes_.push_back(gdn_proto[gdn_i++ % gdn_proto.size()]);
            } else if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0) {
                ov::Shape sh;
                sh.push_back(1);  // blocks dim, filled at allocation
                for (size_t i = 1; i < static_cast<size_t>(ps.rank().get_length()); ++i) {
                    sh.push_back(static_cast<size_t>(ps[static_cast<int64_t>(i)].get_length()));
                }
                kv_pool_names_.push_back(name);
                kv_pool_types_.push_back(port.get_element_type());
                kv_pool_shapes_.push_back(sh);
                size_t block_elems = 1;
                for (size_t i = 1; i < sh.size(); ++i) block_elems *= sh[i];
                kv_block_bytes += block_elems * port.get_element_type().size();
            } else if (name == kPositionIds) {
                paged_sections_ = static_cast<size_t>(ps[0].get_length());
            }
        }
        // Tripwire for re-exports: the byte arithmetic below assumes the
        // plugin's 16-token KV page. The value_cache ports carry that page
        // count as a bare dimension in every layout we serve (f16, and u8
        // where only the head dim is padded with scale/zeropoint bytes);
        // key_cache under u8 pads the token dim itself (16 -> 20), so it is
        // not checked. A re-export laid out in 32-token pages would carry no
        // bare 16 here, and dividing by the wrong page size fails confusingly
        // at decode time rather than loudly at load.
        for (size_t i = 0; i < kv_pool_shapes_.size(); ++i) {
            if (kv_pool_names_[i].rfind("value_cache.", 0) != 0) continue;
            const ov::Shape& sh = kv_pool_shapes_[i];
            if (std::find(sh.begin() + 1, sh.end(), kv_block_tokens_) == sh.end()) {
                std::string dims;
                for (size_t d = 1; d < sh.size(); ++d)
                    dims += (d > 1 ? "x" : "") + std::to_string(sh[d]);
                throw std::runtime_error(log::format(
                    "%s [%s] has no %zu-token page dimension; re-derive kv_block_tokens_ "
                    "for this export before serving it",
                    kv_pool_names_[i].c_str(), dims.c_str(), kv_block_tokens_));
            }
        }
        kv_bytes_token_ = kv_block_bytes / kv_block_tokens_;
        la_row_bytes_   = 0;
        for (const ov::Shape& sh : la_state_shapes_) {
            size_t n = 2;  // f16
            for (size_t i = 1; i < sh.size(); ++i) n *= sh[i];
            la_row_bytes_ += n;
        }

        // ---- allocate the state tables (rows), per lane ----------------------
        ov::RemoteContext rctx = core_.get_default_context(device);
        for (auto& lane : lanes_) alloc_la_rows(rctx, *lane);
        if (std::getenv("ARCINT_PA_HOST_INPUTS") != nullptr) {
            usm_ctx_        = rctx;
            pa_host_inputs_ = true;
            log::info("load", "paged index inputs in USM host memory (ARCINT_PA_HOST_INPUTS)");
        }

        // ---- reservation: measure the activation peak with a probe pool ------
        //
        // The chunk size is the knob that buys context (§7.0.2a: the peak is
        // linear in the chunk), so when the requested n_ctx does not fit at the
        // configured chunk, the engine halves the chunk and RE-MEASURES rather
        // than refusing outright. Refusal is what remains when the floor is hit.
        //
        // With N lanes every per-sequence term is paid N times: activations,
        // GDN checkpoint rows and KV. The weights are paid once, because the
        // lanes share one CompiledModel. Nothing here assumes the factor is N —
        // each lane's request is probed in turn and its own increment measured,
        // which is the only way to find out whether the plugin's per-request
        // buffers really do cost what a first lane cost.
        Lane&        lane0  = *lanes_[0];
        const size_t total  = core_.get_property(device, ov::intel_gpu::device_total_mem_size);
        const size_t slab   = la_row_bytes_ * rows_per_lane_;   // per lane
        const size_t margin = 256ull << 20;
        const int    wanted = req_n_ctx > 0 ? req_n_ctx : artifact_.n_ctx_train;
        const int    lanes  = lane_count_;

        // Probe SMALL first -- a probe at the configured chunk can itself OOM on
        // a tight card (observed on the A770: sometimes the driver spills,
        // sometimes CL_OUT_OF_RESOURCES kills the process). The peak is linear
        // in the chunk (§7.0.2a), so a 128-token probe fixes the slope, the
        // largest admissible chunk is computed, and one guarded probe verifies
        // it, stepping down on failure instead of dying.
        size_t probe_pool_blocks = 0;
        auto probe = [&](Lane& lane, size_t chunk_tokens) -> long long {
            const size_t probe_blocks =
                (chunk_tokens + kv_block_tokens_ - 1) / kv_block_tokens_ + 1;
            if (probe_blocks != probe_pool_blocks) {
                alloc_kv_pools(rctx, probe_blocks);
                probe_pool_blocks = probe_blocks;
            }
            lane.blocks.resize(probe_blocks);
            for (size_t i = 0; i < probe_blocks; ++i) lane.blocks[i] = static_cast<int32_t>(i);
            zero_paged_rows(lane);
            std::vector<int> zeros(chunk_tokens, 0);
            const long long before = static_cast<long long>(device_resident_bytes(device));
            paged_forward(lane, embed_paged(lane, zeros), 0, {0, 0}, 0);
            const long long after = static_cast<long long>(device_resident_bytes(device));
            lane.blocks.clear();
            (void)before;
            return after - static_cast<long long>(resident_base) -
                   static_cast<long long>(slab) * lanes -
                   static_cast<long long>(probe_blocks * kv_block_bytes);
        };
        // The peak is affine in the chunk, not linear from the origin: measured
        // on the A770, 0.62 GiB at 128 tokens and 0.77 at 256, so most of it is
        // a fixed cost that a single probe smears into a slope and then charges
        // again for every token.
        //
        // Probing UPWARD matters, and the reason is a property of the plugin
        // rather than a preference: its intermediate pool grows to the largest
        // shape it has ever seen and never shrinks, so an over-large probe is a
        // permanent tax that no later, smaller probe can undo. Measured the hard
        // way on the A770, 2026-08-29: predicting straight to chunk 1024 left
        // 1.87 GiB resident and a budget of zero, i.e. a card that could serve
        // nothing because of a measurement. So each step up is taken only when
        // the fit says it fits WITH headroom, and the fit is re-made from the
        // two most recent measurements as it climbs.
        const size_t configured = prefill_chunk_ > 0 ? static_cast<size_t>(prefill_chunk_) : 512;
        const size_t floor_c    = std::min<size_t>(configured, 128);

        long long act128 = 0;
        long long extra128 = 0;
        double    slope_extra = 0.0;
        {
            act128 = probe(lane0, floor_c);
            // The slice's layout claim, checked against what the graph returned
            // for this forward rather than against the rewrite's return value.
            if (logits_keep_rows_ > 0) {
                const ov::Tensor lg    = lane0.req.get_tensor("logits");
                const size_t     vocab = lg.get_shape().back();
                const size_t     rows  = vocab > 0 ? lg.get_size() / vocab : 0;
                if (rows != logits_keep_rows_) {
                    std::ostringstream os;
                    os << lg.get_shape();
                    throw std::runtime_error(log::format(
                        "logits slice did not take: %zu row(s) for a %zu-token forward, "
                        "shape %s -- the token axis is not where the slice assumed",
                        rows, floor_c, os.str().c_str()));
                }
                log::info("load", "logits slice verified: %zu row(s) for a %zu-token forward",
                          rows, floor_c);
            }
            // What a SECOND lane costs, measured rather than assumed to be
            // another full peak. On the B60 with the coder it costs nothing:
            // the plugin pools intermediates per compiled model, not per
            // request. Pricing an imaginary second peak would halve the chunk
            // and with it prefill throughput.
            for (size_t i = 1; i < lanes_.size(); ++i) {
                const long long before = static_cast<long long>(device_resident_bytes(device));
                probe(*lanes_[i], floor_c);
                const long long added =
                    static_cast<long long>(device_resident_bytes(device)) - before;
                extra128 += std::max<long long>(added, 0);
            }
            slope_extra = static_cast<double>(extra128) / static_cast<double>(floor_c);
            if (lanes > 1) {
                log::info("load",
                          "lane activations at a %zu-token probe: lane 0 %.3f GiB, the other %d "
                          "lane(s) %.3f GiB together (the plugin pools intermediates per compiled "
                          "model, so this is measured rather than multiplied)",
                          floor_c, static_cast<double>(act128) / (1u << 30), lanes - 1,
                          static_cast<double>(extra128) / (1u << 30));
            }
        }

        // Everything that is not activations, and does not move with the chunk.
        const long long fixed = static_cast<long long>(resident_base) +
                                static_cast<long long>(margin) +
                                static_cast<long long>(lanes) *
                                    (static_cast<long long>(slab) +
                                     static_cast<long long>(wanted) *
                                         static_cast<long long>(kv_bytes_token_));
        // A prediction that lands 11% low was measured; a quarter of headroom
        // buys the step back without giving up the climb.
        const double kHeadroom = 1.25;

        size_t    chunk      = floor_c;
        long long activation = act128;
        double    slope      = static_cast<double>(std::max<long long>(act128, 1)) /
                          static_cast<double>(floor_c);
        double    intercept  = 0.0;

        while (chunk * 2 <= configured) {
            const size_t next      = chunk * 2;
            const double predicted = intercept + (slope + slope_extra) *
                                                     static_cast<double>(next);
            if (fixed + static_cast<long long>(predicted * kHeadroom) >
                static_cast<long long>(total)) {
                break;
            }
            long long measured = 0;
            try {
                measured = probe(lane0, next);
            } catch (const std::exception& e) {
                log::warn("load", "probe at chunk %zu failed (%s); staying at %zu", next,
                          e.what(), chunk);
                reset_lane_request(lane0);
                break;
            }
            if (fixed + measured > static_cast<long long>(total)) {
                // Overshot despite the headroom. The pool has already grown, so
                // this is reported rather than hidden: the budget below uses the
                // measurement, and n_ctx is clamped or refused with it.
                log::warn("load",
                          "chunk %zu measured %.2f GiB of activations, more than the fit "
                          "predicted (%.2f); the plugin's pool does not shrink, so this is what "
                          "the reservation must now live with",
                          next, static_cast<double>(measured) / (1u << 30),
                          predicted / static_cast<double>(1u << 30));
                chunk      = next;
                activation = measured;
                break;
            }
            // Re-fit from the two most recent points: the line gets truer the
            // closer it is to where it is being used.
            if (measured > activation) {
                slope     = static_cast<double>(measured - activation) /
                        static_cast<double>(next - chunk);
                intercept = static_cast<double>(measured) - slope * static_cast<double>(next);
            }
            chunk      = next;
            activation = measured;
        }
        log::info("load", "activation fit: %.3f GiB fixed + %.1f KiB per chunk token; served "
                          "chunk %zu measured %.2f GiB",
                  intercept / static_cast<double>(1u << 30), slope / 1024.0, chunk,
                  static_cast<double>(activation) / (1u << 30));

        // The other lanes again, now at the chunk that will actually be served:
        // whatever they add on top of lane 0's peak is what the budget below
        // pays for.
        long long activation_total = activation;
        for (size_t i = 1; i < lanes_.size(); ++i) {
            const long long before = static_cast<long long>(device_resident_bytes(device));
            probe(*lanes_[i], chunk);
            const long long added = static_cast<long long>(device_resident_bytes(device)) - before;
            activation_total += std::max<long long>(added, 0);
            log::info("load", "lane %zu adds %.2f GiB of activations at chunk %zu (lane 0 "
                              "measured %.2f)",
                      i, static_cast<double>(added) / (1u << 30), chunk,
                      static_cast<double>(activation) / (1u << 30));
        }

        const long long budget = static_cast<long long>(total) -
                                 static_cast<long long>(resident_base) -
                                 static_cast<long long>(slab) * lanes - activation_total -
                                 static_cast<long long>(margin);
        // Per lane: the pool is shared, but every lane must be able to reach
        // n_ctx at the same time, which is what "two lanes of 30k" means.
        long long max_ctx = budget > 0 ? budget / static_cast<long long>(kv_bytes_token_) / lanes
                                       : 0;
        max_ctx = (max_ctx / static_cast<long long>(kv_block_tokens_)) *
                  static_cast<long long>(kv_block_tokens_);
        log::info("load",
                  "reservation: weights+graph %.2f GiB + activations %.2f (all %d lane%s, chunk "
                  "%zu) + margin 0.25 + %d x (GDN rows %.1f MiB + KV %.1f KiB/token) of %.2f "
                  "GiB -> max ctx %lld per lane",
                  static_cast<double>(resident_base) / (1u << 30),
                  static_cast<double>(activation_total) / (1u << 30), lanes,
                  lanes == 1 ? "" : "s", chunk, lanes,
                  static_cast<double>(slab) / (1u << 20),
                  static_cast<double>(kv_bytes_token_) / 1024.0,
                  static_cast<double>(total) / (1u << 30), max_ctx);
        prefill_chunk_ = static_cast<int>(chunk);
        // The snapshot grid. Tied to the chunk until 2026-08-30, which on the
        // agent re-prefilled ~1900 tokens per continuation where 128 would
        // re-prefill ~970 (replay of real sessions, DESIGN 7.0.2j). It must
        // divide a page, because an entry keeps whole pages, and be a multiple
        // of the cache's hash block; it cannot exceed the chunk.
        cache_grid_ = 0;
        if (cfg.cache_grid > 0 && prefill_chunk_ > 0) {
            const size_t unit = std::max<size_t>(kv_block_tokens_, 32);
            size_t g = (static_cast<size_t>(cfg.cache_grid) + unit - 1) / unit * unit;
            g = std::min<size_t>(g, static_cast<size_t>(prefill_chunk_));
            if (g != static_cast<size_t>(cfg.cache_grid)) {
                log::info("load", "--cache-grid %d rounded to %zu (page %zu, hash block 32, chunk %d)",
                          cfg.cache_grid, g, kv_block_tokens_, prefill_chunk_);
            }
            cache_grid_ = g;
        }
        log::info("load", "prefix-cache snapshot grid %zu tok", cache_grid_ > 0 ? cache_grid_ : static_cast<size_t>(prefill_chunk_));

        if (static_cast<long long>(wanted) > max_ctx) {
            if (req_n_ctx > 0) {
                throw std::runtime_error(log::format(
                    "requested n_ctx %d on %d lane%s needs %.2f GiB of KV but the reservation "
                    "admits %lld per lane (weights %.2f + activations %.2f + margin 0.25 + %d x "
                    "state %.3f of %.2f GiB). Lower --n-ctx, lower --parallel, or lower "
                    "--prefill-chunk.",
                    wanted, lanes, lanes == 1 ? "" : "s",
                    static_cast<double>(wanted) * kv_bytes_token_ * lanes / (1u << 30), max_ctx,
                    static_cast<double>(resident_base) / (1u << 30),
                    static_cast<double>(activation_total) / (1u << 30), lanes,
                    static_cast<double>(slab) / (1u << 30),
                    static_cast<double>(total) / (1u << 30)));
            }
            log::info("load", "n_ctx clamped to the admissible %lld (train maximum %d)", max_ctx,
                      wanted);
        }
        paged_n_ctx_ = static_cast<int>(std::min<long long>(wanted, max_ctx));

        // lanes x n_ctx of live pages, plus headroom for cached prefixes — but
        // only as much headroom as the prefix cache could ever hold references
        // to. Its host-side budget bounds how many entries exist, each entry
        // maps at most one lane's worth of pages, and pages nothing can point at
        // are just VRAM taken off the card for nothing.
        const size_t per_lane_blocks =
            (static_cast<size_t>(paged_n_ctx_) + drafts_max_ + kv_block_tokens_ - 1) /
                kv_block_tokens_ + 2;
        const size_t live_blocks = per_lane_blocks * static_cast<size_t>(lanes);
        size_t       blocks      = live_blocks;
        if (budget > 0 && prefix_cache_ != nullptr) {
            const size_t affordable = static_cast<size_t>(budget) / kv_block_bytes;
            const size_t entries =
                std::max<size_t>(1, prefix_cache_->budget_bytes() /
                                        std::max<size_t>(la_row_bytes_, 1));
            const size_t wanted_spare = entries * per_lane_blocks;
            const size_t spare_room   = affordable > live_blocks ? affordable - live_blocks : 0;
            blocks += std::min(spare_room, wanted_spare);
        }
        // A cap for tests: the only way to make the cache evict on demand at a
        // small context. Never below what the lanes themselves need.
        if (cfg.kv_pool_pages > 0) {
            const size_t cap = std::max<size_t>(static_cast<size_t>(cfg.kv_pool_pages), live_blocks);
            if (cap < blocks) {
                log::info("load", "--kv-pool-pages: pool capped at %zu pages (would have been %zu)",
                          cap, blocks);
                blocks = cap;
            }
        }
        alloc_kv_pools(rctx, blocks);
        pool_ = std::make_unique<BlockPool>(blocks);
        if (prefix_cache_ != nullptr) {
            BlockPool* pool = pool_.get();
            prefix_cache_->set_release(
                [pool](const std::vector<int32_t>& b) { pool->release(b); });
        }
        log::info("load",
                  "paged pool: %zu pages x %zu tokens (%.2f GiB KV) = %zu per lane x %d lane%s "
                  "+ %zu spare for cached prefixes | %zu GDN rows per lane",
                  blocks, kv_block_tokens_,
                  static_cast<double>(blocks * kv_block_bytes) / (1u << 30), per_lane_blocks,
                  lanes, lanes == 1 ? "" : "s", blocks - live_blocks, rows_per_lane_);

        status_.reservation.measured           = true;
        status_.reservation.device_total_bytes = total;
        status_.reservation.weights_bytes      = resident_base;
        // Reported as the total for all lanes, because that is what it is: the
        // plugin's intermediate pool is per compiled model. Dividing it by the
        // lane count would invent a per-lane cost that nobody pays.
        status_.reservation.activation_bytes = static_cast<uint64_t>(activation_total);
        status_.reservation.la_slab_bytes      = slab;
        status_.reservation.kv_bytes_per_token = kv_bytes_token_;
        status_.reservation.margin_bytes       = margin;
        status_.reservation.pool_blocks        = blocks;
        status_.reservation.kv_block_tokens    = static_cast<int>(kv_block_tokens_);
        status_.reservation.lanes              = lanes;
        status_.reservation.prefill_chunk      = prefill_chunk_;
        status_.reservation.n_ctx              = paged_n_ctx_;

        if (std::getenv("ARCINT_PROFILE") != nullptr) profile_paged(*lanes_[0]);
    }

    // The KV pool is one allocation shared by every lane: pages are handed out
    // by the block pool, and each lane's block table says which are its own.
    // Every lane's request is bound to the same tensors, which is exactly what
    // makes a prefix hit across lanes free.
    void alloc_kv_pools(ov::RemoteContext& rctx, size_t blocks) {
        kv_pool_tensors_.clear();
        for (size_t i = 0; i < kv_pool_names_.size(); ++i) {
            ov::Shape sh = kv_pool_shapes_[i];
            sh[0]        = blocks;
            ov::RemoteTensor t = rctx.create_tensor(kv_pool_types_[i], sh);
            kv_pool_tensors_.push_back(t);
        }
        for (auto& lane : lanes_) bind_kv_pools(*lane);
    }

    // A probe that hit the card's limit leaves its request in an unknown state;
    // a fresh one, rebound to this lane's own tensors, is the cheapest way back.
    void reset_lane_request(Lane& lane) {
        // The scratch block table of a probe that threw would otherwise survive
        // into the first real request, naming pages the BlockPool (created
        // later) still believes are free — two sequences on one KV page.
        lane.blocks.clear();
        lane.req = paged_model_.create_infer_request();
        for (size_t i = 0; i < la_state_names_.size(); ++i) {
            lane.req.set_tensor(la_state_names_[i], lane.la_tensors[i]);
        }
        bind_kv_pools(lane);
    }

    void bind_kv_pools(Lane& lane) {
        for (size_t i = 0; i < kv_pool_names_.size(); ++i) {
            lane.req.set_tensor(kv_pool_names_[i], kv_pool_tensors_[i]);
        }
    }

    // The GDN checkpoint rows are per lane rather than a window into one shared
    // table, and the reason is right below: a row write stages the whole tensor
    // host-side and copies it back, so two lanes sharing one tensor would zero
    // each other's rows on every prefix-cache restore.
    void alloc_la_rows(ov::RemoteContext& rctx, Lane& lane) {
        lane.la_tensors.clear();
        for (size_t i = 0; i < la_state_names_.size(); ++i) {
            ov::Shape sh = la_state_shapes_[i];
            sh[0]        = rows_per_lane_;
            ov::RemoteTensor t = rctx.create_tensor(ov::element::f16, sh);
            lane.req.set_tensor(la_state_names_[i], t);
            lane.la_tensors.push_back(t);
        }
    }

    void zero_paged_rows(Lane& lane) {
        for (size_t i = 0; i < lane.la_tensors.size(); ++i) {
            ov::Shape sh = la_state_shapes_[i];
            sh[0]        = rows_per_lane_;
            ov::Tensor z(ov::element::f16, sh);
            std::memset(z.data(), 0, z.get_byte_size());
            z.copy_to(lane.la_tensors[i]);
        }
    }

    std::vector<std::vector<uint8_t>> read_paged_row(Lane& lane, size_t row) {
        std::vector<std::vector<uint8_t>> out;
        for (size_t i = 0; i < lane.la_tensors.size(); ++i) {
            ov::Shape full = la_state_shapes_[i];
            full[0]        = rows_per_lane_;
            ov::Tensor host(ov::element::f16, full);
            lane.la_tensors[i].copy_to(host);
            const size_t row_bytes = host.get_byte_size() / rows_per_lane_;
            const uint8_t* base = static_cast<const uint8_t*>(host.data()) + row * row_bytes;
            out.emplace_back(base, base + row_bytes);
        }
        return out;
    }

    void write_paged_row(Lane& lane, size_t row, const std::vector<std::vector<uint8_t>>& blobs) {
        for (size_t i = 0; i < lane.la_tensors.size(); ++i) {
            ov::Shape full = la_state_shapes_[i];
            full[0]        = rows_per_lane_;
            ov::Tensor host(ov::element::f16, full);
            std::memset(host.data(), 0, host.get_byte_size());
            const size_t row_bytes = host.get_byte_size() / rows_per_lane_;
            if (blobs[i].size() != row_bytes) throw std::runtime_error("row blob size mismatch");
            std::memcpy(static_cast<uint8_t*>(host.data()) + row * row_bytes, blobs[i].data(),
                        row_bytes);
            host.copy_to(lane.la_tensors[i]);
        }
    }

    // --------------------------------------------------------- KV page table
    //
    // Grows the lane's block table so it covers `tokens`. Pages come from the
    // shared pool; when it is dry, cached prefixes are dropped first, because a
    // cached page is reclaimable and a live sequence's is not (§3.3). Only when
    // even that is not enough does this fail, and it fails as a clean end of
    // stream rather than as an allocation error on the card.
    // ------------------------------------------------------- host tier (§4.4)
    //
    // A page is a contiguous slab in each KV pool tensor (shape [pages, ...]),
    // so a run of consecutive page ids is one ROI copy per tensor. The
    // allocator hands ids out low-id-first, so a prefix filled in one
    // prefill is a few long ascending runs; after churn it
    // fragments and the copy count grows. Measured, not assumed: the hit line
    // prints the promotion time.
    struct PageRun { int32_t first; size_t count; size_t at; };  // `at`: page index in the entry's order
    static std::vector<PageRun> page_runs(const std::vector<int32_t>& blocks) {
        std::vector<PageRun> runs;
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (!runs.empty() && runs.back().first + static_cast<int32_t>(runs.back().count) == blocks[i]) {
                ++runs.back().count;
            } else if (!runs.empty() && runs.back().first - 1 == blocks[i] && runs.back().count == 1) {
                // descending pair: start a run going down is not a slab; keep it separate
                runs.push_back({blocks[i], 1, i});
            } else {
                runs.push_back({blocks[i], 1, i});
            }
        }
        return runs;
    }
    size_t pool_page_bytes(size_t i) const {
        size_t elems = 1;
        for (size_t d = 1; d < kv_pool_shapes_[i].size(); ++d) elems *= kv_pool_shapes_[i][d];
        return elems * kv_pool_types_[i].size();
    }
    // Copy the entry's pages out to host buffers and release them. Runs under
    // the cache lock; the copies are the cost this feature trades for prefill.
    bool demote_entry(PrefixCache::Entry& e) {
        if (e.blocks.empty() || kv_pool_tensors_.empty()) return false;
        try {
            const std::vector<PageRun> runs = page_runs(e.blocks);
            e.host_kv.assign(kv_pool_tensors_.size(), {});
            e.host_bytes = 0;
            for (size_t i = 0; i < kv_pool_tensors_.size(); ++i) {
                const size_t pb = pool_page_bytes(i);
                std::vector<uint8_t>& buf = e.host_kv[i];
                buf.resize(pb * e.blocks.size());
                ov::Shape sh = kv_pool_shapes_[i];
                for (const PageRun& r : runs) {
                    ov::Coordinate begin(sh.size(), 0), end(sh.begin(), sh.end());
                    begin[0] = static_cast<size_t>(r.first);
                    end[0]   = static_cast<size_t>(r.first) + r.count;
                    sh[0]    = r.count;
                    ov::Tensor host(kv_pool_types_[i], sh, buf.data() + r.at * pb);
                    // A ROI of a remote tensor is still remote; the plugin's
                    // offset copy is reached through the RemoteTensor view.
                    ov::Tensor(kv_pool_tensors_[i], begin, end).as<ov::RemoteTensor>().copy_to(host);
                }
                e.host_bytes += buf.size();
            }
            pool_->release(e.blocks);
            return true;
        } catch (const std::exception& ex) {
            log::warn("cache", "demotion failed (%s); dropping the entry instead", ex.what());
            e.host_kv.clear();
            e.host_bytes = 0;
            return false;
        }
    }
    // Allocate pages for a tiered hit, copy its host buffers back, and record
    // the pages on the entry. On failure the entry stays tiered and the caller
    // prefills cold.
    bool promote_entry(PrefixCache::Hit& hit) {
        const size_t need = hit.matched_tokens / kv_block_tokens_;
        std::vector<int32_t> pages = pool_->allocate(need);
        while (pages.empty() && prefix_cache_->evict_oldest()) pages = pool_->allocate(need);
        if (pages.empty()) return false;
        const PrefixCache::Entry& e = *hit.keep;
        if (e.host_kv.size() != kv_pool_tensors_.size()) { pool_->release(pages); return false; }
        try {
            const std::vector<PageRun> runs = page_runs(pages);
            for (size_t i = 0; i < kv_pool_tensors_.size(); ++i) {
                const size_t pb = pool_page_bytes(i);
                if (e.host_kv[i].size() != pb * need) throw std::runtime_error("host buffer size mismatch");
                ov::Shape sh = kv_pool_shapes_[i];
                for (const PageRun& r : runs) {
                    ov::Coordinate begin(sh.size(), 0), end(sh.begin(), sh.end());
                    begin[0] = static_cast<size_t>(r.first);
                    end[0]   = static_cast<size_t>(r.first) + r.count;
                    sh[0]    = r.count;
                    const ov::Tensor host(kv_pool_types_[i], sh,
                                          const_cast<uint8_t*>(e.host_kv[i].data()) + r.at * pb);
                    ov::RemoteTensor dst = ov::Tensor(kv_pool_tensors_[i], begin, end).as<ov::RemoteTensor>();
                    dst.copy_from(host);
                }
            }
        } catch (const std::exception& ex) {
            log::warn("cache", "promotion failed (%s)", ex.what());
            pool_->release(pages);
            return false;
        }
        if (!prefix_cache_->promote(hit.keep, pages)) { pool_->release(pages); return false; }
        hit.blocks = &e.blocks;   // now the promoted pages
        hit.tiered = false;
        return true;
    }

    bool ensure_blocks(Lane& lane, size_t tokens) {
        const size_t want = (tokens + kv_block_tokens_ - 1) / kv_block_tokens_ + 1;
        if (lane.blocks.size() >= want) return true;
        const size_t need = want - lane.blocks.size();

        std::vector<int32_t> fresh = pool_->allocate(need);
        while (fresh.empty() && prefix_cache_ != nullptr && prefix_cache_->evict_oldest()) {
            fresh = pool_->allocate(need);
        }
        if (fresh.empty()) return false;
        lane.blocks.insert(lane.blocks.end(), fresh.begin(), fresh.end());
        return true;
    }

    void release_lane(Lane& lane) {
        // pool_ can still be null here: a probe that threw during load leaves a
        // lane holding its scratch block table, and the destructor runs anyway.
        if (pool_ != nullptr && !lane.blocks.empty()) pool_->release(lane.blocks);
        lane.blocks.clear();
        lane.hit_keep.reset();
    }

    // The lane's pages and its stall record belong to one request. This puts
    // both back however the request leaves — return, stop sequence, client
    // disconnect, or an exception out of the graph.
    struct LaneReset {
        LaneReset(OvBackend& backend, Lane& lane, GenerationStats& stats)
            : backend_(backend), lane_(lane), stats_(stats) {
            lane_.blocks.clear();   // nothing survives from a previous request
            lane_.stalls.clear();
            lane_.stall_accum = 0.0;
            lane_.stats       = &stats;
        }
        ~LaneReset() {
            backend_.release_lane(lane_);
            std::vector<double>& w = lane_.stalls;
            if (!w.empty()) {
                std::sort(w.begin(), w.end());
                stats_.stalled_steps = static_cast<int>(w.size());
                stats_.stall_max_seconds = w.back();
                stats_.stall_p95_seconds =
                    w[std::min(w.size() - 1, static_cast<size_t>(0.95 * w.size()))];
                double total = 0.0;
                for (double x : w) total += x;
                stats_.stall_total_seconds = total;
                stats_.decode_wait_seconds = total;
            }
            lane_.stats = nullptr;
            w.clear();
            // The USM-host index tensors are keyed by (name, length); lengths
            // grow with context, so without this the map is a slow monotonic
            // pin of host memory (one tiny tensor per distinct block count).
            lane_.host_idx.clear();
        }
        LaneReset(const LaneReset&)            = delete;
        LaneReset& operator=(const LaneReset&) = delete;

        OvBackend&       backend_;
        Lane&            lane_;
        GenerationStats& stats_;
    };

    // Embeddings as a host [n, hidden] f32 tensor (the paged graph's input
    // layout, and the head's food -- it crosses host memory either way).
    ov::Tensor embed_paged(Lane& lane, const std::vector<int>& ids) {
        const size_t n = ids.size();
        ov::Tensor id_tensor(ov::element::i64, ov::Shape{1, n});
        int64_t*   idp = id_tensor.data<int64_t>();
        for (size_t i = 0; i < n; ++i) idp[i] = ids[i];
        lane.embed.set_input_tensor(id_tensor);
        return with_turn(lane, [&] { return embed_take(lane, n); });
    }

    ov::Tensor embed_take(Lane& lane, size_t n) {
        lane.embed.infer();
        const ov::Tensor src   = lane.embed.get_output_tensor(0);
        const size_t     width = src.get_shape().back();
        ov::Tensor out(ov::element::f32, ov::Shape{n, width});
        if (src.get_byte_size() != out.get_byte_size()) {
            throw std::runtime_error(log::format(
                "embeddings output is %s %zu bytes for %zu tokens, expected f32 %zu",
                src.get_element_type().get_type_name().c_str(), src.get_byte_size(),
                n, out.get_byte_size()));
        }
        std::memcpy(out.data(), src.data(), src.get_byte_size());
        return out;
    }

    // One pass over the paged graph, for one lane. `la_rows` is [committed] for
    // in-place (interval 0, duplicated to two entries) or [committed, s0..sk]
    // with interval 1 for a checkpointing verify pass; both are lane-relative
    // and are translated to the lane's own rows here, so nothing above this
    // line has to know that another lane exists.
    //
    // The KV block table comes from the lane too: physical pages, in logical
    // order, from the shared pool. Before M6 it was the identity map 0..n,
    // which is the one thing that made two sequences impossible.
    //
    // The device runs one execution at a time whatever the host does, so the
    // turnstile does not serialise anything that was parallel — it decides the
    // *order*, and measures what the other lane cost this one (§4.1).
    //
    // `want_logits` says whether the caller is going to read the logits at all.
    // A prefill chunk that is not the last one is never sampled from, and
    // reading that output back costs ~165 ms per chunk on this plugin (measured
    // 2026-08-29: prefill 1644 -> 1290 t/s at 30k depth when every chunk was
    // read, decode unchanged). So the copy happens exactly where the value is
    // consumed, which is also the only place it needs protecting from the
    // other lane.
    ov::Tensor paged_forward(Lane& lane, const ov::Tensor& embeds, size_t past,
                             std::vector<int32_t> la_rows, int interval,
                             bool want_logits = true, bool want_feats = true) {
        const size_t n   = embeds.get_shape()[0];
        const size_t tot = past + n;
        const size_t nblk = (tot + kv_block_tokens_ - 1) / kv_block_tokens_;

        auto i32 = [](std::vector<int32_t> v) {
            ov::Tensor t(ov::element::i32, ov::Shape{v.size()});
            std::memcpy(t.data(), v.data(), v.size() * 4);
            return t;
        };
        if (la_rows.size() == 1) la_rows.push_back(la_rows[0]);

        ov::Tensor pos(ov::element::i64, ov::Shape{paged_sections_, n});
        int64_t*   pp = pos.data<int64_t>();
        for (size_t sct = 0; sct < paged_sections_; ++sct) {
            for (size_t i = 0; i < n; ++i) pp[sct * n + i] = static_cast<int64_t>(past + i);
        }
        if (lane.blocks.size() < nblk) {
            throw std::runtime_error(log::format(
                "lane %d has %zu KV page(s) for %zu token(s)", lane.index, lane.blocks.size(),
                tot));
        }
        std::vector<int32_t> blocks(lane.blocks.begin(),
                                    lane.blocks.begin() + static_cast<long>(nblk));

        // The index inputs. The plugin's PagedAttention reads past_lens,
        // subsequence_begins and max_context_len on the host in every stage of
        // every layer once M > 1; on a device buffer each read is a blocking
        // map that drains the queue, ~105 per two-token forward (7.0.2p). In
        // USM host memory the read is a dereference.
        auto set_i32 = [&](const char* name, const std::vector<int32_t>& v, bool scalar = false) {
            if (!pa_host_inputs_) {
                if (scalar) {
                    ov::Tensor t(ov::element::i32, ov::Shape{});
                    *t.data<int32_t>() = v[0];
                    lane.req.set_tensor(name, t);
                } else {
                    lane.req.set_tensor(name, i32(v));
                }
                return;
            }
            const std::string key = std::string(name) + '#' + std::to_string(v.size());
            auto it = lane.host_idx.find(key);
            if (it == lane.host_idx.end()) {
                const ov::Shape sh = scalar ? ov::Shape{} : ov::Shape{v.size()};
                ov::RemoteTensor t = usm_ctx_.create_tensor(
                    ov::element::i32, sh,
                    {{ov::intel_gpu::shared_mem_type.name(),
                      ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}});
                void* ptr = t.get_params().at(ov::intel_gpu::mem_handle.name()).as<void*>();
                it = lane.host_idx.emplace(key, std::make_pair(t, ptr)).first;
            }
            std::memcpy(it->second.second, v.data(), v.size() * sizeof(int32_t));
            lane.req.set_tensor(name, it->second.first);
        };
        lane.req.set_tensor(kInputsEmbeds, embeds);
        lane.req.set_tensor(kPositionIds, pos);
        set_i32("past_lens", {static_cast<int32_t>(past)});
        set_i32("subsequence_begins", {0, static_cast<int32_t>(n)});
        set_i32("block_indices", blocks);
        set_i32("block_indices_begins", {0, static_cast<int32_t>(nblk)});
        set_i32("max_context_len", {static_cast<int32_t>(tot)}, true);
        set_i32("la.block_indices", la_rows);
        set_i32("la.block_indices_begins", {0, static_cast<int32_t>(la_rows.size())});
        set_i32("la.past_lens", {static_cast<int32_t>(past)});
        set_i32("la.cache_interval", {interval});
        with_turn(lane, [&] {
            lane.req.infer();
            if (want_logits) copy_out(lane.req.get_tensor("logits"), lane.logits);
            if (mtp_ready_) copy_out(lane.req.get_tensor("hidden_states"), lane.hidden);
            if (dflash_ready_ && want_feats) {
                copy_out(lane.req.get_tensor("dflash_feats"), lane.dfeats);
            }
        });
        return lane.logits;
    }

    // Every execution of a shared CompiledModel goes through here, and so does
    // every read of its output. The plugin pools intermediate buffers per
    // compiled model rather than per request (§7.2), so an output tensor is only
    // valid until the next execution on that model *by anyone* — which makes
    // this the boundary of correctness, not just of fairness. It applies to the
    // embeddings gather and the MTP head as much as to the language model: they
    // are shared compiled models too.
    template <typename F>
    auto with_turn(Lane& lane, F&& body) -> decltype(body()) {
        Turnstile::Turn turn = gate_.take();
        lane.stall_accum += turn.waited_seconds();
        return body();
    }

    // Into a tensor the lane owns, reshaping only when the graph's dynamic
    // output changes shape (one row for a decode step, 1+k for a verify pass).
    static void copy_out(const ov::Tensor& src, ov::Tensor& dst) {
        if (!dst || dst.get_shape() != src.get_shape() ||
            dst.get_element_type() != src.get_element_type()) {
            dst = ov::Tensor(src.get_element_type(), src.get_shape());
        }
        std::memcpy(dst.data(), src.data(), src.get_byte_size());
    }

    // Which layer graph drafts, and what it wants fed (see the members above).
    std::string choose_mtp_layer(const Artifact& a, const std::string& which) const {
        if (which == "exported") return a.mtp_exported_layer_xml;
        if (which == "reconstructed") return a.mtp_layer_xml;
        return !a.mtp_layer_xml.empty() ? a.mtp_layer_xml : a.mtp_exported_layer_xml;
    }
    void read_mtp_contract(const ov::CompiledModel& layer, const std::string& path) {
        mtp_embeds_name_ = "input_embeds";
        mtp_mask_2d_     = false;
        for (const auto& port : layer.inputs()) {
            const std::string name = port.get_any_name();
            if (name == "inputs_embeds") mtp_embeds_name_ = name;
            if (name == "attention_mask") {
                mtp_mask_2d_   = port.get_partial_shape().rank().get_length() == 2;
                mtp_mask_type_ = port.get_element_type();
            }
            if (name == "position_ids") mtp_pos_type_ = port.get_element_type();
        }
        log::info("mtp", "layer %s: %s, %s mask, %s positions",
                  path.find("openvino_mtp_model") != std::string::npos ? "(optimum-intel export)"
                                                                        : "(reconstructed)",
                  mtp_embeds_name_.c_str(), mtp_mask_2d_ ? "2-D ones" : "4-D additive",
                  mtp_pos_type_.get_type_name().c_str());
    }
    ov::Tensor mtp_positions(size_t start, size_t n) const {
        ov::Tensor t(mtp_pos_type_, ov::Shape{1, n});
        for (size_t i = 0; i < n; ++i) {
            if (mtp_pos_type_ == ov::element::i64) t.data<int64_t>()[i] = static_cast<int64_t>(start + i);
            else t.data<float>()[i] = static_cast<float>(start + i);
        }
        return t;
    }
    // `kv_before` tokens are already in the layer's cache; `n` new ones arrive.
    // The 4-D form is additive and causal across the new tokens; the 2-D form
    // is HF's ones-over-everything and the graph builds causality itself.
    ov::Tensor mtp_mask(size_t kv_before, size_t n) const {
        const size_t total = kv_before + n;
        if (mtp_mask_2d_) {
            ov::Tensor m(mtp_mask_type_, ov::Shape{1, total});
            if (mtp_mask_type_ == ov::element::i64) std::fill_n(m.data<int64_t>(), total, int64_t{1});
            else std::fill_n(m.data<float>(), total, 1.0f);
            return m;
        }
        ov::Tensor m(ov::element::f32, ov::Shape{1, 1, n, total});
        float*     mp = m.data<float>();
        std::fill_n(mp, n * total, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = kv_before + i + 1; j < total; ++j) {
                mp[i * total + j] = -std::numeric_limits<float>::infinity();
            }
        }
        return m;
    }

    // Batched head priming over one prefill chunk: pairs (h_t, x_{t+1}) within
    // the chunk, the chunk-boundary pair carried through the lane's pending row.
    void mtp_prime_paged(Lane& lane, const ov::Tensor& hidden, const ov::Tensor& embeds,
                         size_t take) {
        if (!mtp_ready_ || take == 0) return;
        try {
            const size_t width  = hidden.get_shape().back();
            const bool   carry  = lane.mtp_has_pending;
            const size_t pairs  = (take - 1) + (carry ? 1 : 0);
            if (pairs == 0) { mtp_set_pending(lane, hidden, take - 1); return; }

            ov::Tensor h(ov::element::f32, ov::Shape{1, pairs, width});
            ov::Tensor e(ov::element::f32, ov::Shape{1, pairs, width});
            float* hp = h.data<float>();
            float* ep = e.data<float>();
            const float* hv = hidden.data<const float>();
            const float* ev = embeds.data<const float>();
            size_t r = 0;
            if (carry) {
                std::memcpy(hp, lane.mtp_pending.data(), width * 4);
                std::memcpy(ep, ev, width * 4);
                ++r;
            }
            for (size_t i = 0; i + 1 < take; ++i, ++r) {
                std::memcpy(hp + r * width, hv + i * width, width * 4);
                std::memcpy(ep + r * width, ev + (i + 1) * width, width * 4);
            }

            // In slices: an MoE head (Qwen3.6) computes every expert for every
            // token and holds an [experts, tokens, 2*intermediate] intermediate,
            // 2 GB at a full 2048-pair chunk. 128 pairs keeps it at ~130 MB, and
            // sequential slices are exactly one batch: the head's KV grows the
            // same way and the mask is causal within each slice.
            constexpr size_t kPrimeSlice = 128;
            for (size_t at = 0; at < pairs; at += kPrimeSlice) {
                const size_t n = std::min(kPrimeSlice, pairs - at);
                ov::Tensor hs(ov::element::f32, ov::Shape{1, n, width}, hp + at * width);
                ov::Tensor es(ov::element::f32, ov::Shape{1, n, width}, ep + at * width);
                const ov::Tensor pos  = mtp_positions(lane.mtp_pos, n);
                const ov::Tensor mask = mtp_mask(lane.mtp_len, n);
                ov::Tensor beam(ov::element::i32, ov::Shape{1});
                beam.data<int32_t>()[0] = 0;
                lane.mtp_layer.set_tensor("hidden_states", hs);
                lane.mtp_layer.set_tensor(mtp_embeds_name_, es);
                lane.mtp_layer.set_tensor("position_ids", pos);
                lane.mtp_layer.set_tensor("attention_mask", mask);
                lane.mtp_layer.set_tensor("beam_idx", beam);
                with_turn(lane, [&] { lane.mtp_layer.infer(); });
                lane.mtp_len += n;
                lane.mtp_pos += n;
            }
            lane.mtp_has_pending = false;
            mtp_set_pending(lane, hidden, take - 1);
        } catch (const std::exception& e) {
            log::warn("mtp", "head priming failed, disabling it: %s", e.what());
            mtp_ready_ = false;
        }
    }

    // Mirrors restore()'s contract: a failure degrades to "no caching", never
    // to a 500 on an otherwise healthy request.
    bool snapshot(Lane& lane, PrefixCache::StateBlob& out) {
        try {
            out.clear();
            for (ov::VariableState& v : lm_req_.query_state()) {
                out.push_back(serialise_state(v.get_state()));
            }
            // The MTP head has its own attention KV over the same prefix. Left
            // out, a warm run would draft from an empty head while a cold run
            // drafts from a primed one -- different drafts, so possibly a
            // different answer, which is the one thing §3.4 does not allow.
            // Its cursor goes in as well: the head lags the base model by a
            // position, and that offset cannot be recovered from the tensors.
            if (mtp_ready_) {
                for (ov::VariableState& v : lane.mtp_layer.query_state()) {
                    out.push_back(serialise_state(v.get_state()));
                }
                ov::Tensor cursor(ov::element::i64, ov::Shape{3});
                cursor.data<int64_t>()[0] = static_cast<int64_t>(lane.mtp_len);
                cursor.data<int64_t>()[1] = static_cast<int64_t>(lane.mtp_pos);
                cursor.data<int64_t>()[2] = lane.mtp_has_pending ? 1 : 0;
                out.push_back(serialise_state(cursor));
                // ...and the row it is holding. The head consumes (h_t, x_{t+1}),
                // so at a checkpoint it always has one hidden row waiting for a
                // token that has not arrived yet. Dropping it would leave the
                // head a position behind after a restore.
                out.push_back(serialise_state(
                    lane.mtp_has_pending ? lane.mtp_pending
                                     : ov::Tensor(ov::element::f32, ov::Shape{1, 1, 1})));
            }
            return true;
        } catch (const std::exception& e) {
            log::warn("cache", "state snapshot failed, continuing without caching: %s", e.what());
            out.clear();
            return false;
        }
    }

    size_t estimated_state_bytes(Lane& lane) {
        if (state_bytes_ != 0) return state_bytes_;
        try {
            for (ov::VariableState& v : lm_req_.query_state()) {
                state_bytes_ += v.get_state().get_byte_size() + 64;
            }
            if (mtp_ready_) {
                for (ov::VariableState& v : lane.mtp_layer.query_state()) {
                    state_bytes_ += v.get_state().get_byte_size() + 64;
                }
                // The cursor plus the pending hidden row. Derived, not a baked
                // n_embd: this has to follow whatever model is loaded.
                state_bytes_ += 128;
                if (lane.mtp_pending) state_bytes_ += lane.mtp_pending.get_byte_size() + 64;
                else state_bytes_ += static_cast<size_t>(artifact_.n_embd) * 4 + 64;
            }
        } catch (const std::exception&) {
            state_bytes_ = 0;
        }
        return state_bytes_;
    }

    bool restore(Lane& lane, const PrefixCache::StateBlob& blob) {
        std::vector<ov::VariableState> states = lm_req_.query_state();
        std::vector<ov::VariableState> mtp_states;
        size_t                         want = states.size();
        if (mtp_ready_) {
            mtp_states = lane.mtp_layer.query_state();
            want += mtp_states.size() + 2;  // + the cursor and the pending row
        }
        if (want != blob.size()) return false;
        try {
            for (size_t i = 0; i < states.size(); ++i) {
                states[i].set_state(deserialise_state(blob[i]));
            }
            for (size_t i = 0; i < mtp_states.size(); ++i) {
                mtp_states[i].set_state(deserialise_state(blob[states.size() + i]));
            }
            if (mtp_ready_) {
                const ov::Tensor cursor = deserialise_state(blob[blob.size() - 2]);
                lane.mtp_len         = static_cast<size_t>(cursor.data<const int64_t>()[0]);
                lane.mtp_pos         = static_cast<size_t>(cursor.data<const int64_t>()[1]);
                lane.mtp_has_pending = cursor.data<const int64_t>()[2] != 0;
                if (lane.mtp_has_pending) lane.mtp_pending = deserialise_state(blob.back());
            }
        } catch (const std::exception& e) {
            log::warn("cache", "state restore failed, falling back to a cold prefill: %s",
                      e.what());
            lm_req_.reset_state();
            return false;
        }
        return true;
    }

    // One real forward pass. This is what turns a latent import fault into a
    // startup failure instead of a run of 500s.
    void warmup() {
        Lane& lane = *lanes_[0];
        forward(lane, {0}, 0);
        lm_req_.reset_state();
        if (std::getenv("ARCINT_BENCH_FORWARD") != nullptr) bench_forward();
        if (std::getenv("ARCINT_PROFILE") != nullptr) profile_step();
    }

    // Per-kernel breakdown of the paged graph, for one prefill chunk and one
    // decode step. Prefill has never been profiled on any path: every kernel
    // table in DESIGN describes a decode step, which is how the reference
    // Transpose and GDN kernels were found, and prefill was simply never looked
    // at. The two phases run the same graph with a different token count, so
    // they can and do select different kernels.
    //
    // Read the shares, not the wall clock: PERF_COUNT inflates the latter, and
    // get_profiling_info() reports the LAST inference, which is why each phase
    // is run immediately before its table is dumped.
    void profile_paged(Lane& lane) {
        // The profiler borrows lane 0 and hands its pages back on every path.
        // warmup() runs after this and calls forward() directly, without an
        // ensure_blocks of its own, so it relies on lane 0 still holding the
        // pages the reservation gave it. Leaving the lane empty made
        // ARCINT_PROFILE and serving mutually exclusive: bring-up died with
        // "lane 0 has 0 KV page(s) for 1 token(s)". Restore what we borrowed.
        const size_t pages_on_entry = lane.blocks.size();
        size_t      depth = static_cast<size_t>(std::max(1, prefill_chunk_));
        const char* env   = std::getenv("ARCINT_PROFILE");
        if (env != nullptr && env[0] >= '1' && env[0] <= '9') {
            depth = static_cast<size_t>(std::strtoul(env, nullptr, 10));
        }

        auto dump = [&](const char* what, size_t tokens) {
            struct Agg { double us = 0.0; int n = 0; };
            std::map<std::string, Agg> by_kernel;
            // A node identified by "nothing else is this size" is a conjecture.
            // Reference-kernel rows get their node names printed so the thing
            // can be named rather than inferred.
            std::map<std::string, std::vector<std::string>> ref_names;
            double                     total = 0.0;
            // ARCINT_PROFILE_NODES=1: every node of the decode step by name, so the
            // primitive count -- which is what binds decode -- can be bucketed by
            // layer and block rather than guessed from kernel families.
            const bool dump_nodes = std::getenv("ARCINT_PROFILE_NODES") != nullptr &&
                                    std::strcmp(what, "decode step") == 0;
            for (const ov::ProfilingInfo& p : lane.req.get_profiling_info()) {
                const double us = static_cast<double>(p.real_time.count());
                if (dump_nodes) {
                    log::info("nodes", "%s\t%s\t%s\t%.0f", p.node_name.c_str(),
                              p.node_type.c_str(), p.exec_type.c_str(), us);
                }
                if (us <= 0.0) continue;
                const std::string key = p.node_type + "  " + p.exec_type;
                Agg& a = by_kernel[key];
                a.us += us;
                a.n += 1;
                total += us;
                if (p.exec_type.find("ref") != std::string::npos) {
                    ref_names[key].push_back(p.node_name);
                }
            }
            std::vector<std::pair<std::string, Agg>> rows(by_kernel.begin(), by_kernel.end());
            std::sort(rows.begin(), rows.end(),
                      [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
            log::info("profile", "%s: %zu token(s), node total %.2f ms across %zu (op, kernel) "
                                 "pairs, %.1f us/token",
                      what, tokens, total / 1000.0, rows.size(),
                      tokens > 0 ? total / static_cast<double>(tokens) : 0.0);
            // Six denominator errors, three retracted headlines. A share whose
            // denominator is implicit is a share that will be wrong again, so
            // the denominator is named on the same line as the shares it
            // governs, every time, and so is what the numerator is not.
            log::info("profile", "  denominator: THIS capture only (%s, %zu token(s), "
                                 "%.2f ms) -- not prefill wall, not a served chunk",
                      what, tokens, total / 1000.0);
            log::info("profile", "%s", "  a past-0 capture is the cheapest chunk in any "
                                       "run; depth-independent rows overstate here");
            // Measured 2026-08-30 against an OpenCL device timeline on the same
            // run: PERF_COUNT reported 14.97 ms where the device spent 27.23 ms
            // on the same 30 conv executions, and 36.78 against 67.19 ms on the
            // same 30 GDN executions -- 1.82x and 1.83x. It also cannot see
            // memory transfers, which were 18.5% of a served prefill's device
            // time. Shares below are shares of what this counter reports.
            log::info("profile", "%s", "  numerator: PERF_COUNT under-reports device time "
                                       "~1.8x and omits transfers entirely");
            // The canonical reason to distrust it needs no domain knowledge: at
            // M=1 it once reported 36.5 ms for forty gate nodes inside a step
            // whose wall was 22 ms. A component cannot exceed the container it
            // is measured in. A figure that fails that test is rejected without
            // further argument -- and the per-node decode numbers do.
            if (std::strcmp(what, "decode step") == 0) {
                log::info("profile", "%s", "  decode per-node figures are not device time: a "
                                           "component here has exceeded its own step's wall");
            }
            // Every pair, not the top 20. A truncated table cannot answer
            // "is this op still in the graph at all", which is exactly the
            // question that arose about the transposes the paged transformation
            // was supposed to have removed.
            for (size_t i = 0; i < rows.size(); ++i) {
                log::info("profile", "  %8.1f us %5d  %5.1f%%  %s", rows[i].second.us,
                          rows[i].second.n, 100.0 * rows[i].second.us / total,
                          rows[i].first.c_str());
            }
            // Only the FullyConnected fallbacks, and only a few names: the
            // question is which projection, and forty lines of the same answer
            // do not make it truer.
            for (const auto& [key, names] : ref_names) {
                if (key.find("FullyConnected") == std::string::npos) continue;
                for (size_t i = 0; i < names.size() && i < 4; ++i) {
                    log::info("profile", "    ref node: %s", names[i].c_str());
                }
                log::info("profile", "    (%zu node(s) on %s)", names.size(), key.c_str());
            }
        };

        if (!ensure_blocks(lane, depth + 2)) {
            log::warn("profile", "%s", "not enough KV pages to profile");
            return;
        }

        // A sweep over the token count, not a single chunk, because the open
        // question is what changes between one token and many. Kernel selection
        // for a dynamic-shape node happens per runtime shape, so M=1, M=2 and
        // M=depth are three different questions asked of the same graph.
        // ARCINT_PROFILE_SWEEP=2,4,8,... bisects the token count at which kernel
        // selection changes, in ONE load: each M is a forward, not a compile,
        // so the whole search costs what one server start costs.
        std::vector<size_t> sweep{1, 2, depth};
        if (const char* list = std::getenv("ARCINT_PROFILE_SWEEP")) {
            sweep.clear();
            const std::string spec(list);
            size_t pos = 0;
            while (pos < spec.size()) {
                const size_t comma = spec.find(',', pos);
                const std::string tok = spec.substr(pos, comma - pos);
                if (!tok.empty()) sweep.push_back(std::strtoul(tok.c_str(), nullptr, 10));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        // ARCINT_PROFILE_PAST=<n> profiles the chunk after prefilling to depth n,
        // which is the only way to see the quadratic term: every capture at
        // past 0 is a chunk that attends to nothing, and the depth dimension was
        // simply not being sampled.
        size_t past = 0;
        if (const char* pv = std::getenv("ARCINT_PROFILE_PAST")) {
            past = static_cast<size_t>(std::strtoul(pv, nullptr, 10));
        }

        // The synthetic input is not neutral in a mixture of experts. A chunk of
        // identical token ids routes every token to the same experts, so the MoE
        // moves a fraction of the weight a real chunk moves, and every share in
        // the resulting table is a share of the wrong denominator.
        // ARCINT_PROFILE_TOKENS=random fills with deterministic pseudo-random ids
        // instead. It is an arm, not a new default: the two must be compared, and
        // the old behaviour has to stay reachable to compare against.
        const bool rnd_tokens = [] {
            const char* v = std::getenv("ARCINT_PROFILE_TOKENS");
            return v != nullptr && std::string(v) == "random";
        }();
        uint64_t rng = 88172645463325252ull;
        auto make_tokens = [&](size_t n) {
            std::vector<int> t(n, 0);
            if (rnd_tokens) {
                for (size_t i = 0; i < n; ++i) {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    t[i] = static_cast<int>(rng % 100000U) + 1;
                }
            }
            return t;
        };
        log::info("profile", "synthetic tokens: %s", rnd_tokens ? "pseudo-random" : "all zero");

        for (size_t m : sweep) {
            if (m == 0) continue;
            if (!ensure_blocks(lane, past + m + 2)) {
                log::warn("profile", "not enough KV pages for M=%zu at past %zu", m, past);
                continue;
            }
            zero_paged_rows(lane);
            std::vector<int> chunk = make_tokens(m);
            // Walk to `past` in chunks the served grid would use, so the state
            // and the block table are what a real prefill would present.
            size_t at = 0;
            while (at < past) {
                const size_t take = std::min<size_t>(past - at, static_cast<size_t>(
                    std::max(1, prefill_chunk_)));
                paged_forward(lane, embed_paged(lane, make_tokens(take)), at, {0}, 0,
                              false);
                at += take;
            }
            // Twice, and the second one is dumped. The first pass of any shape
            // carries kernel warm-up, and an ascending sweep turns that into a
            // decaying bias that reads as a U-shape in every op -- measured
            // 2026-08-30, and it is why an ascending single-pass sweep cannot be
            // used to argue about scaling.
            paged_forward(lane, embed_paged(lane, chunk), at, {0}, 0);
            paged_forward(lane, embed_paged(lane, chunk), at, {0}, 0);
            {
                const ov::Tensor lg = lane.req.get_tensor("logits");
                std::ostringstream os;
                os << lg.get_shape();
                log::info("profile", "logits tensor after M=%zu: %s, %.1f MiB", m,
                          os.str().c_str(), lg.get_byte_size() / (1024.0 * 1024.0));
            }
            dump(log::format("prefill M=%zu past=%zu", m, past).c_str(), m);
            release_lane(lane);
        }

        // Then one decode step at depth, which is a 1-token forward with a past
        // rather than without one. The sweep above released the lane, so this
        // needs its own pages rather than inheriting whatever the loop left.
        zero_paged_rows(lane);
        if (!ensure_blocks(lane, depth + 2)) {
            log::warn("profile", "not enough KV pages for the decode step at depth %zu", depth);
            return;
        }
        paged_forward(lane, embed_paged(lane, make_tokens(depth)), 0, {0}, 0);
        paged_forward(lane, embed_paged(lane, {0}), depth, {0}, 0);
        dump("decode step", 1);

        release_lane(lane);
        zero_paged_rows(lane);
        // Unconditionally, not restoring a snapshot: lane 0 holds no pages when
        // the profiler is entered, so there is nothing to snapshot, and warmup()
        // still needs one page for its single-token forward.
        const size_t want = std::max<size_t>(pages_on_entry * kv_block_tokens_, 2);
        if (!ensure_blocks(lane, want)) {
            log::warn("profile", "could not restore lane %d after profiling", lane.index);
        }
    }

    // Per-kernel breakdown of one decode step. Aggregated by (op, kernel) so
    // the reference-kernel fallbacks stand out, which is what the numbers are
    // usually for. Needs PERF_COUNT at compile time (ARCINT_PROFILE sets it).
    void profile_step() {
        Lane& lane = *lanes_[0];
        // Depth matters: a profile at 64 tokens of context is not the step a
        // real request takes. ARCINT_PROFILE=<n> sets the prefix length.
        size_t      depth = 64;
        const char* env   = std::getenv("ARCINT_PROFILE");
        if (env != nullptr && env[0] >= '1' && env[0] <= '9') {
            depth = static_cast<size_t>(std::strtoul(env, nullptr, 10));
        }
        lm_req_.reset_state();
        std::vector<int> warm(depth, 0);
        forward(lane, warm, 0);

        // Wall clock for the same step, so the node total can be compared
        // against what the step actually costs.
        const auto t0 = std::chrono::steady_clock::now();
        const int  reps = 10;
        for (int i = 0; i < reps; ++i) forward(lane, {0}, depth + static_cast<size_t>(i));
        const double wall_ms = 1000.0 * seconds_since(t0) / reps;
        log::info("profile", "decode step wall clock %.2f ms at depth %zu", wall_ms, depth);

        struct Agg { double us = 0.0; int n = 0; };
        std::map<std::string, Agg> by_kernel;
        double                     total = 0.0;
        for (const ov::ProfilingInfo& p : lm_req_.get_profiling_info()) {
            const double us = static_cast<double>(p.real_time.count());
            if (us <= 0.0) continue;
            Agg& a = by_kernel[p.node_type + "  " + p.exec_type];
            a.us += us;
            a.n += 1;
            total += us;
        }

        std::vector<std::pair<std::string, Agg>> rows(by_kernel.begin(), by_kernel.end());
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.second.us > b.second.us; });

        log::info("profile", "decode step %.2f ms across %zu (op, kernel) pairs",
                  total / 1000.0, rows.size());
        for (size_t i = 0; i < rows.size() && i < 20; ++i) {
            log::info("profile", "%8.1f us %5d  %5.1f%%  %s", rows[i].second.us, rows[i].second.n,
                      100.0 * rows[i].second.us / total, rows[i].first.c_str());
        }
        lm_req_.reset_state();
    }

    // Speculative decoding only pays when a k-token forward costs about what a
    // 1-token forward costs. On a hybrid GDN model that is an open question --
    // 30 of 40 layers are a recurrent scan -- so measure it rather than assume
    // it. Prints ms per forward and the cost relative to a single token.
    void bench_forward() {
        Lane& lane = *lanes_[0];
        const size_t past = 512;
        std::vector<int> warm(past, 0);
        double one = 0.0;
        for (size_t k : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{9},
                         size_t{17}, size_t{33}, size_t{65}}) {
            lm_req_.reset_state();
            forward(lane, warm, 0);
            std::vector<int> ids(k, 0);
            const auto t0 = std::chrono::steady_clock::now();
            const int reps = 5;
            for (int r = 0; r < reps; ++r) forward(lane, ids, past + k * static_cast<size_t>(r));
            const double ms = 1000.0 * seconds_since(t0) / reps;
            if (k == 1) one = ms;
            log::info("bench", "forward(%2zu tok) %7.2f ms  = %5.2fx one token  "
                               "(%6.2f ms/token)",
                      k, ms, one > 0.0 ? ms / one : 1.0, ms / static_cast<double>(k));
        }
        lm_req_.reset_state();
    }

    static double seconds_since(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    static json messages_json(const ChatRequest& req, bool object_arguments) {
        json out = json::array();
        for (const ChatMessage& m : req.messages) {
            json msg{{"role", m.role}, {"content", m.content}};
            if (!m.name.empty()) msg["name"] = m.name;
            if (!m.tool_call_id.empty()) msg["tool_call_id"] = m.tool_call_id;
            if (!m.tool_calls.empty()) {
                json calls = json::array();
                for (const ToolCall& c : m.tool_calls) {
                    calls.push_back({{"id", c.id},
                                     {"type", "function"},
                                     {"function",
                                      {{"name", c.name},
                                       {"arguments", tool_call_arguments_for_template(
                                                         c.arguments, object_arguments)}}}});
                }
                msg["tool_calls"] = std::move(calls);
                if (m.content.empty()) msg["content"] = nullptr;
            }
            out.push_back(std::move(msg));
        }
        return out;
    }

    static json tools_json(const ChatRequest& req) {
        if (req.tools.empty()) return json();
        json out = json::array();
        for (const ToolSpec& t : req.tools) {
            out.push_back({{"type", "function"},
                           {"function",
                            {{"name", t.name},
                             {"description", t.description},
                             {"parameters", t.parameters}}}});
        }
        return out;
    }

    // Runs the embeddings graph then the language model for `ids`, with `past`
    // tokens already in the graph's state.
    ov::Tensor forward(Lane& lane, const std::vector<int>& ids, size_t past) {
        const size_t n     = ids.size();
        const size_t total = past + n;

        ov::Tensor id_tensor(ov::element::i64, ov::Shape{1, n});
        int64_t*   idp = id_tensor.data<int64_t>();
        for (size_t i = 0; i < n; ++i) idp[i] = ids[i];

        const auto t_embed = std::chrono::steady_clock::now();
        lane.embed.set_input_tensor(id_tensor);
        lane.embed.infer();
        if (step_stats_ != nullptr) step_stats_->decode_embed_seconds += seconds_since(t_embed);

        // Copy the embeddings out instead of handing the language model the
        // embeddings request's own output buffer. Sharing it made chunked
        // prefill diverge from unchunked: two requests aliasing one tensor is
        // not something either plugin promises to serialise, and the
        // equivalence suite caught it as a byte difference rather than a crash.
        const ov::Tensor src = lane.embed.get_output_tensor(0);
        if (!embeds_ || embeds_.get_shape() != src.get_shape() ||
            embeds_.get_element_type() != src.get_element_type()) {
            embeds_ = ov::Tensor(src.get_element_type(), src.get_shape());
        }
        std::memcpy(embeds_.data(), src.data(), src.get_byte_size());
        const ov::Tensor& embeds = embeds_;

        ov::Tensor mask(ov::element::i64, ov::Shape{1, total});
        std::fill_n(mask.data<int64_t>(), total, int64_t{1});

        // mrope: position_ids is [sections, batch, seq]. For text-only input
        // every section carries the same linear position — verified against the
        // artifact on the dev host, where a first-section-only layout produces
        // identical greedy output.
        const size_t sections = position_sections();
        ov::Tensor   pos(ov::element::i64, ov::Shape{sections, 1, n});
        int64_t*     pp = pos.data<int64_t>();
        for (size_t s = 0; s < sections; ++s) {
            for (size_t i = 0; i < n; ++i) pp[s * n + i] = static_cast<int64_t>(past + i);
        }

        ov::Tensor beam(ov::element::i32, ov::Shape{1});
        beam.data<int32_t>()[0] = 0;

        lm_req_.set_tensor(kInputsEmbeds, embeds);
        lm_req_.set_tensor(kAttentionMask, mask);
        lm_req_.set_tensor(kPositionIds, pos);
        lm_req_.set_tensor(kBeamIdx, beam);
        const auto t_lm = std::chrono::steady_clock::now();
        lm_req_.infer();
        if (step_stats_ != nullptr) step_stats_->decode_forward_seconds += seconds_since(t_lm);

        if (mtp_ready_) {
            // Our own copy, for exactly the reason embeds_ above is a copy: this
            // is the request's output buffer, and the MTP head runs two more
            // inferences before these rows are all read. Reading a live request
            // buffer across another infer is what broke chunked prefill once
            // already.
            const ov::Tensor src = lm_req_.get_tensor("hidden_states");
            if (!hidden_copy_ || hidden_copy_.get_shape() != src.get_shape() ||
                hidden_copy_.get_element_type() != src.get_element_type()) {
                hidden_copy_ = ov::Tensor(src.get_element_type(), src.get_shape());
            }
            std::memcpy(hidden_copy_.data(), src.data(), src.get_byte_size());
            last_hidden_ = hidden_copy_;
        }
        return lm_req_.get_tensor(logits_port_);
    }

    // ------------------------------------------------------------------ MTP
    //
    // The head predicts x_{t+2} from the base model's hidden state at t and the
    // embedding of x_{t+1}, so it runs exactly one position behind the base
    // model and is fed one position per committed token. Its own attention KV
    // therefore never needs rolling back: every input it has consumed is a
    // token the model committed to.
    void mtp_reset(Lane& lane) {
        if (!mtp_ready_) return;
        lane.mtp_layer.reset_state();
        lane.mtp_len         = 0;
        lane.mtp_pos         = 0;
        lane.mtp_has_pending = false;
    }

    // A prefix-cache hit skips the prompt the head would have been primed on.
    // Its rope positions must still be the true ones, so the two counters part
    // company: the head attends to a shorter prefix than it is positioned in.
    void mtp_seek(Lane& lane, size_t position) {
        if (!mtp_ready_) return;
        lane.mtp_pos = position;
    }

    void mtp_set_pending(Lane& lane, const ov::Tensor& hidden, size_t row) {
        if (!mtp_ready_) return;
        const ov::Shape& shape = hidden.get_shape();
        const size_t     width = shape.back();
        const size_t     rows  = hidden.get_size() / width;
        if (row >= rows) return;
        if (!lane.mtp_pending || lane.mtp_pending.get_shape() != ov::Shape{1, 1, width} ||
            lane.mtp_pending.get_element_type() != hidden.get_element_type()) {
            lane.mtp_pending = ov::Tensor(hidden.get_element_type(), ov::Shape{1, 1, width});
        }
        std::memcpy(lane.mtp_pending.data(),
                    static_cast<const uint8_t*>(hidden.data()) +
                        row * width * hidden.get_element_type().size(),
                    width * hidden.get_element_type().size());
        lane.mtp_has_pending = true;
    }

    // Feeds (pending, embedding of `next`) at the head's current position.
    // Returns the drafted token when one is wanted, or -1.
    int mtp_feed(Lane& lane, int next, bool want_draft) {
        if (!mtp_ready_ || !lane.mtp_has_pending) return -1;
        // One turn for the whole head step: it runs three shared compiled models
        // back to back and reads each one's output, so splitting it would leave
        // exactly the windows §7.2 says are unsafe.
        return with_turn(lane, [&] { return mtp_feed_locked(lane, next, want_draft); });
    }

    int mtp_feed_locked(Lane& lane, int next, bool want_draft) {
        try {
            ov::Tensor ids(ov::element::i64, ov::Shape{1, 1});
            ids.data<int64_t>()[0] = next;
            lane.embed.set_input_tensor(ids);
            lane.embed.infer();
            const ov::Tensor src = lane.embed.get_output_tensor(0);
            ov::Tensor       emb(src.get_element_type(), src.get_shape());
            std::memcpy(emb.data(), src.data(), src.get_byte_size());

            const ov::Tensor pos  = mtp_positions(lane.mtp_pos, 1);
            // Nothing is masked: the head attends to its whole committed prefix.
            const ov::Tensor mask = mtp_mask(lane.mtp_len, 1);

            ov::Tensor beam(ov::element::i32, ov::Shape{1});
            beam.data<int32_t>()[0] = 0;

            lane.mtp_layer.set_tensor("hidden_states", lane.mtp_pending);
            lane.mtp_layer.set_tensor(mtp_embeds_name_, emb);
            lane.mtp_layer.set_tensor("position_ids", pos);
            lane.mtp_layer.set_tensor("attention_mask", mask);
            lane.mtp_layer.set_tensor("beam_idx", beam);
            lane.mtp_layer.infer();

            ++lane.mtp_len;
            ++lane.mtp_pos;
            lane.mtp_has_pending = false;
            if (!want_draft) return -1;

            lane.mtp_head.set_input_tensor(lane.mtp_layer.get_output_tensor(0));
            lane.mtp_head.infer();
            const ov::Tensor lg   = lane.mtp_head.get_output_tensor(0);
            const size_t     v    = lg.get_shape().back();
            const size_t     rows = v ? lg.get_size() / v : 0;
            if (rows == 0) return -1;
            return Sampler::argmax(lg.data<const float>() + (rows - 1) * v, v);
        } catch (const std::exception& e) {
            log::warn("mtp", "head failed, disabling it for this process: %s", e.what());
            mtp_ready_ = false;
            return -1;
        }
    }

    // ---------------------------------------------------------------- DFlash2
    //
    // The drafter's context KV lives in its graph state and only ever receives
    // accepted positions, so there is nothing to roll back: a rejected block's
    // K/V never entered the state (the graph computes them per call and the
    // state append holds context rows only). docs/dflash-pairing-probe.md.
    void dflash_reset(Lane& lane) {
        if (!dflash_ready_) return;
        lane.dflash_req.reset_state();
        lane.dflash_pending.clear();
        lane.dflash_base = SIZE_MAX;
    }

    // Rows [row_from, row_from+rows) of the last forward's feature output
    // belong to absolute positions [pos, pos+rows) and are now committed
    // context. A discontinuity means the bookkeeping is wrong, and the honest
    // reaction is to stop drafting, not to draft from misaligned context.
    void dflash_append(Lane& lane, size_t row_from, size_t rows, size_t pos) {
        if (!dflash_ready_ || rows == 0) return;
        const size_t width = dflash_feat_width_;
        const size_t have  = lane.dfeats.get_size() / width;
        if (row_from + rows > have) {
            log::warn("dflash", "feature output has %zu row(s), needed %zu; disabling",
                      have, row_from + rows);
            dflash_ready_ = false;
            return;
        }
        const size_t pend = lane.dflash_pending.size() / width;
        if (lane.dflash_base == SIZE_MAX) {
            lane.dflash_base = pos;
        } else if (lane.dflash_base + pend != pos) {
            log::warn("dflash", "feature gap: pending ends at %zu, append at %zu; disabling",
                      lane.dflash_base + pend, pos);
            dflash_ready_ = false;
            return;
        }
        const float* src = lane.dfeats.data<const float>();
        lane.dflash_pending.insert(lane.dflash_pending.end(), src + row_from * width,
                                   src + (row_from + rows) * width);
        const size_t total = lane.dflash_pending.size() / width;
        if (total > kDflashWindow) {
            const size_t drop = total - kDflashWindow;
            lane.dflash_pending.erase(lane.dflash_pending.begin(),
                                      lane.dflash_pending.begin() +
                                          static_cast<long>(drop * width));
            lane.dflash_base += drop;
        }
    }

    // One verification cycle's draft: feed the pending accepted features, run
    // the block over [anchor, mask x (block-1)], score the draft rows through
    // the target lm_head, and trace one path with the selector. One turn for
    // the whole step: three shared compiled models run back to back.
    std::vector<int> dflash_draft(Lane& lane, int anchor, size_t past) {
        if (!dflash_ready_) return {};
        try {
            const size_t width = dflash_feat_width_;
            const size_t pend  = lane.dflash_pending.size() / width;
            if (pend == 0 || lane.dflash_base + pend != past) {
                log::warn("dflash", "pending context ends at %zu but the anchor sits at %zu; "
                                    "disabling",
                          lane.dflash_base == SIZE_MAX ? 0 : lane.dflash_base + pend, past);
                dflash_ready_ = false;
                return {};
            }
            const size_t q = dflash_block_;
            const size_t h = dflash_mask_embed_.get_shape().back();

            return with_turn(lane, [&]() -> std::vector<int> {
                ov::Tensor ids(ov::element::i64, ov::Shape{1, 1});
                ids.data<int64_t>()[0] = anchor;
                lane.embed.set_input_tensor(ids);
                lane.embed.infer();
                const ov::Tensor ae = lane.embed.get_output_tensor(0);

                ov::Tensor noise(ov::element::f32, ov::Shape{1, q, h});
                float*     np = noise.data<float>();
                std::memcpy(np, ae.data(), h * 4);
                const float* me = dflash_mask_embed_.data<const float>();
                for (size_t i = 1; i < q; ++i) std::memcpy(np + i * h, me, h * 4);

                ov::Tensor feats(ov::element::f32, ov::Shape{1, pend, width},
                                 lane.dflash_pending.data());
                ov::Tensor pos(ov::element::i64, ov::Shape{pend + q});
                int64_t*   pp = pos.data<int64_t>();
                for (size_t i = 0; i < pend; ++i) {
                    pp[i] = static_cast<int64_t>(lane.dflash_base + i);
                }
                for (size_t i = 0; i < q; ++i) {
                    pp[pend + i] = static_cast<int64_t>(past + i);
                }

                lane.dflash_req.set_tensor("new_feats", feats);
                lane.dflash_req.set_tensor("noise", noise);
                lane.dflash_req.set_tensor("positions", pos);
                lane.dflash_req.infer();
                const ov::Tensor out = lane.dflash_req.get_output_tensor(0);   // [1,q,h]

                ov::Tensor rows(ov::element::f32, ov::Shape{1, q - 1, h});
                std::memcpy(rows.data(), out.data<const float>() + h, (q - 1) * h * 4);
                lane.dflash_head.set_input_tensor(rows);
                lane.dflash_head.infer();
                const ov::Tensor lg = lane.dflash_head.get_output_tensor(0);

                std::vector<int> drafts = dflash_select(
                    rows.data<const float>(), lg.data<const float>(), q - 1, h, anchor);
                lane.dflash_pending.clear();
                lane.dflash_base = past;
                return drafts;
            });
        } catch (const std::exception& e) {
            log::warn("dflash", "draft failed, disabling the drafter: %s", e.what());
            dflash_ready_ = false;
            return {};
        }
    }

    // The candidate-path selector: top-k tokens per position, one coherent
    // path traced greedily with the predecessor/successor codebooks.
    std::vector<int> dflash_select(const float* hid, const float* logits, size_t rows,
                                   size_t width, int anchor) {
        const size_t k = dflash_topk_, r = dflash_rank_, vocab = dflash_vocab_;
        std::vector<int>   path;
        std::vector<float> hp(r), t(r), unary(k);
        std::vector<int>   cand(k);
        int pred = anchor;
        path.reserve(rows);
        for (size_t row = 0; row < rows; ++row) {
            const float* lg = logits + row * vocab;
            // top-k: min-heap of size k over the vocab
            using P = std::pair<float, int>;
            std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
            for (size_t v = 0; v < vocab; ++v) {
                if (heap.size() < k) {
                    heap.emplace(lg[v], static_cast<int>(v));
                } else if (lg[v] > heap.top().first) {
                    heap.pop();
                    heap.emplace(lg[v], static_cast<int>(v));
                }
            }
            for (size_t j = k; j-- > 0;) {
                unary[j] = heap.top().first;
                cand[j]  = heap.top().second;
                heap.pop();
            }
            const float* hrow = hid + row * width;
            for (size_t d = 0; d < r; ++d) {
                const float* pr  = dflash_proj_.data() + d * width;
                float        acc = 0.0f;
                for (size_t x = 0; x < width; ++x) acc += pr[x] * hrow[x];
                hp[d] = acc;
            }
            const ov::float16* pc = dflash_pred_cb_.data() + static_cast<size_t>(pred) * r;
            for (size_t d = 0; d < r; ++d) t[d] = static_cast<float>(pc[d]) * hp[d];
            float best  = -std::numeric_limits<float>::infinity();
            int   bidx  = cand[0];
            for (size_t j = 0; j < k; ++j) {
                const ov::float16* sc = dflash_succ_cb_.data() +
                                        static_cast<size_t>(cand[j]) * r;
                float score = unary[j];
                for (size_t d = 0; d < r; ++d) score += t[d] * static_cast<float>(sc[d]);
                if (score > best) {
                    best = score;
                    bidx = cand[j];
                }
            }
            pred = bidx;
            path.push_back(pred);
        }
        return path;
    }

    void load_dflash_selector(const std::string& dir) {
        auto read_all = [](const std::string& path) {
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            if (!in.good()) throw std::runtime_error("cannot read " + path);
            const std::streamsize n = in.tellg();
            in.seekg(0);
            std::vector<char> buf(static_cast<size_t>(n));
            in.read(buf.data(), n);
            return buf;
        };
        dflash_feat_width_ = static_cast<size_t>(
            dflash_model_.input("new_feats").get_partial_shape()[2].get_length());
        const size_t hidden = dflash_feat_width_ / 5;
        const auto proj = read_all(dir + "/dflash_hidden_projection.f32.bin");
        dflash_rank_ = proj.size() / (4 * hidden);
        dflash_proj_.resize(proj.size() / 4);
        std::memcpy(dflash_proj_.data(), proj.data(), proj.size());
        const auto pred = read_all(dir + "/dflash_predecessor_codebook.f16.bin");
        const auto succ = read_all(dir + "/dflash_successor_codebook.f16.bin");
        if (pred.size() != succ.size() || pred.size() % (2 * dflash_rank_) != 0) {
            throw std::runtime_error("selector codebooks disagree about their shape");
        }
        dflash_vocab_ = pred.size() / (2 * dflash_rank_);
        dflash_pred_cb_.resize(pred.size() / 2);
        dflash_succ_cb_.resize(succ.size() / 2);
        std::memcpy(dflash_pred_cb_.data(), pred.data(), pred.size());
        std::memcpy(dflash_succ_cb_.data(), succ.data(), succ.size());
    }

    size_t position_sections() const {
        if (position_sections_ != 0) return position_sections_;
        for (const auto& port : language_.inputs()) {
            if (port.get_names().count(kPositionIds) == 0) continue;
            const ov::PartialShape& ps = port.get_partial_shape();
            if (ps.rank().is_static() && ps.rank().get_length() == 3 && ps[0].is_static()) {
                position_sections_ = static_cast<size_t>(ps[0].get_length());
                return position_sections_;
            }
            position_sections_ = 1;
            return position_sections_;
        }
        position_sections_ = 1;
        return position_sections_;
    }

    // Copies the last logit row out of the graph's output tensor and hands it
    // to the sampler. The copy is deliberate: the sampler mutates the row for
    // penalties, and the tensor belongs to the InferRequest.
    // The sampler's decision for one row of a [tokens, 1, vocab] logits tensor.
    // Verification and normal picking go through the same call so that they can
    // never drift apart.
    int pick_row(Lane& lane, Sampler& sampler, const ov::Tensor& logits, size_t row) {
        const size_t vocab = logits.get_shape().back();
        const size_t rows  = logits.get_size() / vocab;
        // Out of range is a graph/plumbing bug, not a draft miss. Returning -1
        // guarantees rejection (token ids are non-negative) so output stays
        // correct, and the caller reports it instead of silently accepting 0%.
        if (row >= rows) return -1;

        const float* p = logits.data<const float>() + row * vocab;
        lane.logit_scratch.assign(p, p + vocab);
        return sampler.sample(lane.logit_scratch.data(), vocab);
    }

    int pick(Lane& lane, Sampler& sampler, const ov::Tensor& logits) {
        const size_t vocab = logits.get_shape().back();
        const auto   t0    = std::chrono::steady_clock::now();
        const int    tok   = pick_row(lane, sampler, logits, logits.get_size() / vocab - 1);
        if (step_stats_ != nullptr) step_stats_->decode_sample_seconds += seconds_since(t0);
        return tok;
    }

    Artifact                       artifact_;
    ov::Core                       core_;
    ov::CompiledModel              embeddings_;
    ov::CompiledModel              language_;
    ov::InferRequest               lm_req_;
    ov::Output<const ov::Node>     logits_port_;
    std::unique_ptr<OvTokenizer>   tokenizer_;
    std::unique_ptr<minja::chat_template> template_;
    ModelStatus                    status_;
    std::mutex                     mutex_;
    ov::Tensor                     embeds_;
    int                            prefill_chunk_ = 512;
    size_t                         state_bytes_   = 0;
    std::unique_ptr<PrefixCache>   prefix_cache_;
    // --- MTP head (§3.5). Drafts exactly one token per decode step.
    bool                           want_mtp_ = false;
    // Written by whichever lane sees the head fail, read by both on every step.
    std::atomic<bool>              mtp_ready_{false};
    ov::CompiledModel              mtp_layer_;
    ov::CompiledModel              mtp_head_;
    ov::Tensor                     last_hidden_;    // the base model's, this step
    ov::Tensor                     hidden_copy_;    // owned; last_hidden_ points here
    int                            offload_ratio_ = 0;
    // --- lanes (§4.1). One per --parallel slot; the stateful reference path
    // uses lane 0 for its embeddings and MTP requests and serialises on
    // mutex_, because it has one graph state and cannot do better.
    std::vector<std::unique_ptr<Lane>> lanes_;
    int                            lane_count_ = 1;
    size_t                         rows_per_lane_ = 4;
    std::unique_ptr<BlockPool>     pool_;
    Turnstile                      gate_;
    // --- paged path (§3.5.3, §7.0): arcint-owned block tables + LA rows ---
    bool                           paged_ = false;
    ov::CompiledModel              paged_model_;
    ov::InferRequest               paged_req_;
    std::vector<std::string>       la_state_names_;   // conv + gdn table port names
    std::vector<ov::Shape>         la_state_shapes_;  // one row; the rows dim is filled in
    std::vector<std::string>       kv_pool_names_;
    std::vector<ov::element::Type> kv_pool_types_;
    std::vector<ov::Shape>         kv_pool_shapes_;   // with the blocks dim filled in
    std::vector<ov::RemoteTensor>  kv_pool_tensors_;
    // The plugin's own KV page size, not --kv-block-size: the paged graph's
    // key_cache/value_cache ports are laid out in 16-token pages and the byte
    // arithmetic below divides by it. --kv-block-size governs the prefix cache's
    // reuse granularity, which is a multiple of this and therefore compatible.
    size_t                         kv_block_tokens_  = 16;
    size_t                         kv_bytes_token_   = 0;
    size_t                         la_row_bytes_     = 0;
    size_t                         logits_keep_rows_ = 0;  // 0: unsliced
    size_t                         cache_grid_       = 0;  // paged snapshot grid; 0: the chunk
    ov::RemoteContext              usm_ctx_;              // for USM-host index inputs
    bool                           pa_host_inputs_   = false;  // ARCINT_PA_HOST_INPUTS
    // The MTP layer's input contract, read from the compiled graph: the
    // reconstructed layer takes input_embeds, an f32 4-D additive mask and f32
    // positions; optimum-intel's export takes inputs_embeds, an i64 2-D
    // ones-mask and i64 positions. Same hidden in, same hidden out.
    std::string                    mtp_embeds_name_  = "input_embeds";
    bool                           mtp_mask_2d_      = false;
    ov::element::Type              mtp_mask_type_    = ov::element::f32;
    ov::element::Type              mtp_pos_type_     = ov::element::f32;
    int                            paged_n_ctx_      = 0;
    size_t                         paged_sections_   = 4;    // position_ids dim 0
    size_t                         drafts_max_       = 0;
    std::vector<ov::Tensor>        rollback_;   // reused speculation scratch
    bool                           logged_rollback_size_ = false;
    // Kept alive for the life of the process and gated by a flag rather than
    // destroyed when speculation turns itself off: with two lanes, one lane's
    // `drafter_.reset()` would free the object the other lane is inside
    // (NgramDrafter::draft only reads, so sharing it is otherwise fine).
    // ---- DFlash2 drafter state (docs/dflash-pairing-probe.md) ----
    ov::CompiledModel        dflash_model_;
    ov::CompiledModel        dflash_head_model_;
    bool                     want_dflash_      = false;
    bool                     dflash_ready_     = false;
    size_t                   dflash_block_     = 8;
    int                      dflash_mask_token_ = 0;
    size_t                   dflash_topk_      = 16;
    size_t                   dflash_rank_      = 0;
    size_t                   dflash_vocab_     = 0;
    size_t                   dflash_feat_width_ = 0;
    static constexpr size_t  kDflashWindow     = 2048;   // the head's sliding window
    std::vector<float>       dflash_proj_;               // [rank, hidden] f32
    std::vector<ov::float16> dflash_pred_cb_;            // [vocab, rank]
    std::vector<ov::float16> dflash_succ_cb_;
    ov::Tensor               dflash_mask_embed_;

    std::unique_ptr<Drafter>       drafter_;
    std::atomic<bool>              drafting_{false};
    size_t                         draft_tokens_ = 0;
    mutable size_t                 position_sections_ = 0;
};

}  // namespace

std::unique_ptr<Backend> make_ov_backend(const Artifact& artifact, const Config& cfg, int n_ctx) {
    return std::make_unique<OvBackend>(artifact, cfg, n_ctx);
}

}  // namespace lgc
