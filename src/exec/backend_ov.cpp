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
#include <cctype>
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
#include "core/affinity.h"
#include "exec/fit.h"
#include "core/artifact.h"
#include "core/block_pool.h"
#include "core/dflash_select.h"
#include "core/dflash_window.h"
#include "core/drafter.h"
#include "core/prefix_cache.h"
#include "core/sampler.h"
#include "core/turnstile.h"
#include "exec/backend.h"
#include "exec/rope_precision.h"
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

// M8 (docs/design-m8-asymmetric-kv.md §3): the five element types
// --paged-kv / ARCINT_PAGED_KV accept on either side of the colon, mapped to
// the OpenVINO type the plugin's own KV_CACHE_PRECISION / VALUE_CACHE_PRECISION
// properties want. config.h's parse_paged_kv() has already refused anything
// outside this set by the time this runs, so the fallback below is dead code
// reached only if the two tables ever drift -- rather than let that drift
// resolve to a silent default, it refuses too.
ov::element::Type paged_kv_element(const std::string& s) {
    if (s == "f16") return ov::element::f16;
    if (s == "u8") return ov::element::u8;
    if (s == "i8") return ov::element::i8;
    if (s == "u4") return ov::element::u4;
    if (s == "i4") return ov::element::i4;
    throw std::runtime_error(log::format("paged_kv_element: unrecognised '%s'", s.c_str()));
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

// M7 §2 Phase B, analytic route (the M7 design, "Auto-fit and the honest
// reservation"): sizes the expert slot pool from the `read_model` the load
// path already holds, without materialising any weight data -- a weightless
// IR (ov::weights_path) still carries every constant's shape and element
// type in the XML.
//
// A MoE op is identified by its OpenVINO type name carrying "moe" (matching
// how tools/verify_moe_lowering.py's own fusion_check finds them in a
// compiled runtime graph); its expert-weight inputs are the Constant
// operands (or a Constant behind one Convert, for compressed weights) whose
// leading dimension equals `num_expert` -- the expert axis is dim 0, so
// gate/up/down and any packed scale/zero-point tensors are all picked up the
// same way, by shape rather than by name. Every guard the M7 design names
// (op type unrecognised, a weight input that is not a plain constant, zero
// bytes, n_expert == 0) drops out of the sum rather than being special-cased,
// so an unmatched graph simply returns nullopt and the caller falls back to
// the plateau probe -- this function never guesses.
struct SlotPoolIr {
    uint64_t total_bytes      = 0;  // summed over every matched MoE layer
    uint64_t per_expert_bytes = 0;  // from the first matched layer, for the load-time log line
    int      slots            = 0;  // per layer, from the first matched layer
    int      moe_layers       = 0;
};

std::optional<SlotPoolIr> slot_pool_from_ir(const std::shared_ptr<ov::Model>& model,
                                            int num_expert, int ratio_pct) {
    if (num_expert <= 0) return std::nullopt;
    SlotPoolIr out;
    for (const auto& node : model->get_ordered_ops()) {
        std::string tname = node->get_type_name();
        std::transform(tname.begin(), tname.end(), tname.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (tname.find("moe") == std::string::npos) continue;

        uint64_t per_expert_bytes = 0;
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            std::shared_ptr<ov::Node> src = node->input_value(i).get_node_shared_ptr();
            // Compressed weights sometimes ride through one Convert (u4/u8 ->
            // f16/f32) before the op that consumes them; the Constant
            // underneath still carries the real (compressed) shape and
            // element size, which is the byte count that is actually
            // resident once a slot is loaded.
            if (const auto conv = ov::as_type_ptr<ov::op::v0::Convert>(src)) {
                src = conv->input_value(0).get_node_shared_ptr();
            }
            const auto konst = ov::as_type_ptr<ov::op::v0::Constant>(src);
            if (konst == nullptr) continue;  // not a weight input we can size
            const ov::Shape& sh = konst->get_shape();
            if (sh.empty() || sh[0] != static_cast<size_t>(num_expert)) continue;
            uint64_t elems = 1;
            for (size_t d = 1; d < sh.size(); ++d) elems *= sh[d];
            per_expert_bytes += elems * konst->get_element_type().size();
        }
        if (per_expert_bytes == 0) continue;  // not weight-bearing, or shapes we can't read

        if (out.moe_layers == 0) {
            out.per_expert_bytes = per_expert_bytes;
            // Mirrors expert_slot_bytes' ceiling (exec/fit.h): the plugin's
            // own rounding lives in ops/moe.cpp, out of this repository's
            // tree (patches/0003 here only touches expert-mask subbuffer
            // caching and does not carry that expression), so this
            // over-reserves rather than under-reserves, pending an on-card
            // audit against MOE_OTD_PERF_LOG.
            out.slots = static_cast<int>(expert_slot_bytes(num_expert, ratio_pct, 1, 1));
        }
        out.total_bytes += expert_slot_bytes(num_expert, ratio_pct, per_expert_bytes, 1);
        ++out.moe_layers;
    }
    if (out.moe_layers == 0) return std::nullopt;
    return out;
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
        // Per-lane recoverable disable (measured 2026-09-03): a draft
        // failure on this lane's sequence turns its drafter off until the
        // next dflash_reset (a fresh request re-arms it) instead of the old
        // process-wide `dflash_ready_ = false`, which left every other
        // lane's drafter permanently off for the rest of the process after
        // one bad request. dflash_fail_count is cumulative across resets
        // (not cleared by dflash_reset) so a persistently failing lane stays
        // visible in the request-end log/stats rather than quietly resetting
        // to zero every time it re-arms.
        bool               dflash_off        = false;
        uint64_t           dflash_fail_count = 0;

        // M11 dump instrument (the M11 design note (not in the repository) O): the lattice
        // dflash_select built for the cycle currently in flight, stashed here
        // because the drafts/accepted/realized half of the dump line is only
        // known later, back in the decode loop's verify step. Only populated
        // when ARCINT_DFLASH_DUMP is set.
        nlohmann::json     dflash_dump_lattice;

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

        // M11 step profile (ARCINT_PROFILE_CYCLE, DESIGN §7.0.2aa row): the
        // last paged_forward's own t0..t3 split and the last propose phase's
        // four segments, mirrored here so generate_paged's cycle line can
        // read them without a new clock read of its own. Populated
        // unconditionally where the clock was already being read for
        // ARCINT_FORWARD_SPLIT / the MTP head step (tens of nanoseconds,
        // 5278); only the cycle line itself is gated.
        double last_fwd_ms_index  = 0.0;
        double last_fwd_ms_infer  = 0.0;
        double last_fwd_ms_logits = 0.0;
        double last_fwd_ms_hidden = 0.0;
        double mtp_step_ms_embed  = 0.0;
        double mtp_step_ms_mask   = 0.0;
        double mtp_step_ms_layer  = 0.0;
        double mtp_step_ms_head   = 0.0;
    };

    OvBackend(const Artifact& artifact, const Config& cfg, int n_ctx)
        : artifact_(artifact), prefill_chunk_(cfg.prefill_chunk) {
        const std::string& device    = cfg.device;
        std::string cache_dir = cfg.cache_dir;
        lane_count_           = std::max(1, cfg.parallel);
        pin_dispatch_         = cfg.pin_dispatch;
        moe_cpu_tier_         = cfg.moe_cpu_tier;
        moe_cpu_tier_threads_ = cfg.moe_cpu_tier_threads;
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
        // M11 dump instrument (the M11 design note (not in the repository) O): opt-in, read once. One
        // JSON line per verify cycle that actually drafted through DFlash;
        // dflash_select stashes the lattice for that cycle on the lane, and
        // the decode loop appends the drafts/accepted/realized-tokens half
        // once the verify forward's logits are in hand.
        //
        // Critical-path cost when set (review follow-up): inside
        // dflash_draft's with_turn (holding the device turn),
        // dflash_dump_build_lattice computes the K x K x r transition matrix
        // per row -- the same FLOP count as a second Viterbi pass -- and
        // serialises it plus drafts_proposed/realized to JSON; the decode
        // loop's realized-token pass adds `drafts.size()` extra pick_row
        // calls, each copying the whole vocab row into logit_scratch.
        // Estimated 1-2 ms against a ~70 ms verify cycle (K=16, r=128, block
        // 8) -- worth stating in the cell whenever a dump is on, not just
        // measuring. When unset the cost is one atomic bool load per cycle
        // and nothing else; the file is opened once here, not once per
        // cycle, and dflash_dump_write only ever appends to it under
        // dflash_dump_mutex_.
        if (const char* env = std::getenv("ARCINT_DFLASH_DUMP")) {
            dflash_dump_path_ = env;
            dflash_dump_file_.open(dflash_dump_path_, std::ios::app);
            if (dflash_dump_file_.good()) {
                dflash_dump_enabled_.store(true, std::memory_order_release);
            } else {
                log::warn("dflash", "cannot open %s for the dump; disabling it",
                          dflash_dump_path_.c_str());
            }
        }

        core_.add_extension(tokenizers_extension_path());
        if (!cache_dir.empty()) core_.set_property(ov::cache_dir(cache_dir));

        tokenizer_ = std::make_unique<OvTokenizer>(core_, artifact, artifact.eos_ids);

        // M13: a VLM export's vision tower and projector IRs sit on disk
        // beside the language model but are never compiled -- --vision is
        // reserved (config.cpp), not served. Named and sized here so a "why
        // is this artifact bigger than the served weights" question has an
        // answer in the log rather than requiring a directory listing.
        if (!artifact.unloaded_vision_irs.empty()) {
            uint64_t vision_bytes = 0;
            for (const auto& ir : artifact.unloaded_vision_irs) vision_bytes += ir.bytes;
            log::info("load", "vision IRs present and not loaded: %zu files, %.1f MiB on disk",
                      artifact.unloaded_vision_irs.size(),
                      static_cast<double>(vision_bytes) / (1024.0 * 1024.0));
        }

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
            // Measured 2026-09-01 (DESIGN 7.0.2s): the OTD slot buffers commit
            // physical memory lazily, so the residency this reservation reads
            // at load excludes them, and the max-ctx it derives is optimistic
            // by roughly the resident expert share. And the streaming path
            // costs ~30x decode on this plugin build. A fit-the-model lever,
            // not a context-for-VRAM dial.
            //
            // Stateful-path only, and stated as fact rather than advice
            // (review): M7's auto-fit (Phase A-E, exec/fit.h) lives on
            // the paged path and is not reachable from here -- this
            // executor's own reservation has no slot-pool term at all, so
            // "set --n-ctx" would be routing around the one real check that
            // exists on the path that has one, not fixing this one.
            log::warn("load", "%s",
                      "offload active on the stateful (--no-paged) reference executor: it has "
                      "no auto-fit (M7 is paged-only) and its reservation does not account for "
                      "the expert slot pool, which commits lazily, so the printed max ctx here "
                      "is optimistic. The paged path (the default) sizes and refuses on this "
                      "term; this one does not.");
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
                // M11 §2 (DESIGN §7.0.2ag, rope_precision.h): the same
                // rotary f32-precision fix the paged path applies to its own
                // MTP/DFlash compiles further down this file -- read_model()
                // first (not a direct path compile) so the RoPE marker below
                // has a graph to walk before the plugin's own ConvertPrecision
                // pass runs inside compile_model(); see the paged path's own
                // comment for ARCINT_DRAFT_F32 / ARCINT_DRAFT_ROPE_F16.
                const bool draft_f32 = draft_f32_enabled();
                ov::AnyMap draft_precision_props;
                if (draft_f32) {
                    draft_precision_props[ov::hint::inference_precision.name()] = ov::element::f32;
                }
                const bool apply_rope_fix = !draft_rope_f16_enabled();
                auto         mtp_layer_model = core_.read_model(layer_xml);
                const size_t mtp_rope_marked =
                    apply_rope_fix ? mark_rope_no_fp16_compression(mtp_layer_model) : 0;
                mtp_layer_    = core_.compile_model(mtp_layer_model, device, draft_precision_props);
                read_mtp_contract(mtp_layer_, layer_xml);
                mtp_head_     = core_.compile_model(artifact.mtp_lm_head_xml, device,
                                                     draft_precision_props);
                lanes_[0]->mtp_layer = mtp_layer_.create_infer_request();
                lanes_[0]->mtp_head  = mtp_head_.create_infer_request();
                mtp_ready_           = true;
                log::info("mtp",
                          "head ready in %.1f s; drafting one token per step%s; rope kept f32 on "
                          "%zu node(s)%s",
                          seconds_since(t0),
                          draft_f32 ? " (ARCINT_DRAFT_F32: f32 inference precision)" : "",
                          mtp_rope_marked,
                          apply_rope_fix ? "" : " (ARCINT_DRAFT_ROPE_F16: fix disabled)");
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
        // §M12b: this thread — whichever HTTP worker owns `slot` — is also the
        // one that calls every synchronous infer() below for the rest of the
        // request, so pinning it here covers the whole generation loop (paged
        // and stateful alike) from one spot. Opt-in and a no-op by default.
        //
        // The pin outlives this call (review): pthread_setaffinity_np sets
        // the thread's affinity mask, not a per-request scope, so a pooled
        // HTTP worker that served slot 0 here and is later handed slot 1 by
        // the pool stays pinned to slot 0's core until it is told otherwise
        // -- there is no unpin. --pin-dispatch is a testing knob for a
        // measured hypothesis, not a production affinity policy, and this is
        // the price of the one-call, whole-request-covering placement.
        if (pin_dispatch_ >= 0) {
            const int core = pin_dispatch_ + slot;
            if (pin_current_thread(core)) {
                log::debug(log::format("slot %d", slot), "dispatch pinned to core %d", core);
            } else {
                // Core offline, cpuset-vetoed, or a non-Linux build: silent
                // before this fix, which reads as "pinned" in a log full of
                // debug lines from the successful case right beside it.
                log::warn(log::format("slot %d", slot),
                          "dispatch pin to core %d failed; running unpinned this request", core);
            }
        }
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
            // forward() (step_stats_, ~6186/6224) increments
            // decode_embed_seconds/decode_forward_seconds on every call,
            // including this one, and this call is ALSO bracketed into
            // draft_verify_seconds right below -- both counting the same
            // verify pass. Save and restore so the paged path's rule holds
            // here too: the served decode line's "other" subtracts
            // draft_verify_seconds once, so this pass must not also land in
            // decode_forward_seconds/decode_embed_seconds, or it is counted
            // twice and `other` goes negative instead of naming the verify.
            const double saved_embed_seconds   = stats.decode_embed_seconds;
            const double saved_forward_seconds = stats.decode_forward_seconds;
            logits                             = forward(lane, seq, past);
            stats.decode_embed_seconds   = saved_embed_seconds;
            stats.decode_forward_seconds = saved_forward_seconds;
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

        // M11 step profile (DESIGN §7.0.2aa row, the M11 design note (not in
        // the repository)): one line per drafting cycle, off by default and
        // reachable only through this switch. Segments 2-15 of that note's
        // table had no timer at all before this; ARCINT_FORWARD_SPLIT
        // (5283) is unaffected and averages over 64 forwards, which mixes
        // depths -- this is a strict superset, one line per cycle, no
        // averaging.
        static const bool profile_cycle = profile_cycle_enabled();

        // Review follow-up: one id per call, so the dump instrument can
        // tell which cycles belong to the same request (a lane is reused
        // across requests, so lane id alone is not enough to chain cycles).
        const uint64_t request_id = next_request_id_.fetch_add(1);

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
                dflash_active(lane) && past + take + kDflashWindow >= prompt_ids.size();
            const ov::Tensor out =
                paged_forward(lane, emb, past, {static_cast<int32_t>(c)}, 0, last, wfeats);
            stats.prefill_forward_seconds +=
                seconds_since_tp(t_fwd) - (lane.stall_accum - waited);
            if (last) logits = out;
            if (mtp_ready_) {
                mtp_prime_paged(lane, lane.hidden, emb, take);
            }
            if (dflash_active(lane) && wfeats) {
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
            // M11 dump instrument (the M11 design note (not in the repository) O): only the DFlash
            // branch has a lattice to dump; ngram and MTP proposals carry no
            // candidate set for the offline oracle to bound.
            bool used_dflash = false;
            // The propose phase's own cost, previously untimed (M11 §1 row
            // 2-5): turnstile wait, embed.infer() for the anchor token, and
            // whichever drafter runs. Net of turnstile wait, same pattern as
            // decode_embed_seconds/decode_forward_seconds below.
            double     propose_waited = lane.stall_accum;
            const auto t_propose      = clock::now();
            // M11 step profile only: commit() (below, and in the accept loop)
            // flushes lane.stall_accum into lane.stalls and zeroes it every
            // time it runs, so by the time the cycle line prints, stall_accum
            // alone can read 0 even though this cycle waited -- the entries
            // this cycle pushed are still in lane.stalls, appended after this
            // mark.
            const size_t stalls_at_propose = profile_cycle ? lane.stalls.size() : 0;
            if (dflash_active(lane) && in.sampler.greedy()) {
                drafts = dflash_draft(lane, next, past);
                // Set AFTER the call, not before: a failed draft can disable
                // this lane from inside dflash_draft (dflash_lane_fail), and
                // the dump code below reads lane.dflash_dump_lattice, which
                // a failed call never repopulated -- it could still hold
                // whatever a PRIOR successful cycle left in it (possibly
                // already moved out of by that cycle's own dump write). A
                // lane still active after the call means dflash_select ran
                // and the lattice is this cycle's own.
                used_dflash = dflash_active(lane);
            } else if (mtp_ready_ && in.sampler.greedy()) {
                const int d = mtp_feed(lane, next, true);
                if (d >= 0) drafts.push_back(d);
            } else if (drafting_.load() && in.sampler.greedy()) {
                drafts = drafter_->draft(history, draft_tokens_);
                if (drafts.size() > drafts_max_) drafts.resize(drafts_max_);
            }
            const double propose_ms =
                1000.0 * (seconds_since_tp(t_propose) - (lane.stall_accum - propose_waited));
            stats.draft_propose_seconds += propose_ms / 1000.0;
            // M11 step profile only: the propose call above is mtp_feed's
            // want_draft=true invocation, which is the only one that reads
            // the head's logits; the promotion step below (mtp_ready_ block,
            // want_draft=false) calls mtp_feed again per accepted draft and
            // would otherwise overwrite lane.mtp_step_ms_* before the cycle
            // line gets to read this cycle's own propose numbers.
            const double cyc_mtp_embed_ms = lane.mtp_step_ms_embed;
            const double cyc_mtp_mask_ms  = lane.mtp_step_ms_mask;
            const double cyc_mtp_layer_ms = lane.mtp_step_ms_layer;
            const double cyc_mtp_head_ms  = lane.mtp_step_ms_head;

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

            // Net of turnstile wait, same pattern as the neighbouring
            // embed/forward timers above: decode_wait_seconds (LaneReset,
            // the sum of every lane.stalls entry) already counts this same
            // wait, so a gross verify duration would double it.
            double     verify_waited = lane.stall_accum;
            const auto t_verify      = clock::now();
            // M11 step profile only: an extra pair of clock reads around the
            // verify embed, gated so the always-served path (profile_cycle
            // off) makes exactly the same calls it made before this change.
            const auto te0 = profile_cycle ? clock::now() : clock::time_point{};
            const ov::Tensor vembed = embed_paged(lane, seq);
            const double vembed_ms =
                profile_cycle
                    ? std::chrono::duration<double, std::milli>(clock::now() - te0).count()
                    : 0.0;
            logits = paged_forward(lane, vembed, past, la_rows, 1);
            stats.draft_verify_seconds +=
                seconds_since_tp(t_verify) - (lane.stall_accum - verify_waited);
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

            // M11 dump instrument (the M11 design note (not in the repository) O): the target's
            // realized token for every verify row, PLUS one extra row at
            // index drafts.size() (review follow-up) -- `rows >=
            // seq.size() == 1+drafts.size()` was already checked above, so
            // row index drafts.size() is always in bounds. That extra row is
            // what `next = pick_row(lane, sampler, logits, accepted)` reads
            // further down whenever accepted==drafts.size() (every draft
            // accepted, nothing rejected): realized[accepted] is ALWAYS the
            // genuine next-anchor token this way, not only when a rejection
            // happened to produce a spare row to read it from. This is what
            // lets the oracle chain cycles of one request (realized[acc] is
            // exactly cycle c+1's anchor) instead of stopping at whichever
            // cycle first saw a full accept.
            //
            // Computed now -- before the accept loop below calls commit()
            // (and so sampler.observe(), which updates the
            // repetition/presence-penalty history) -- so a non-default
            // penalty cannot skew a row's realized token by the rows
            // accepted ahead of it. Standard measurement cells run
            // `--repetition-penalty 1.0` (apply_penalties is then a no-op),
            // where this ordering makes no difference; it matters once a
            // penalty is in play (the oracle script's own caveat comment).
            std::vector<int> dflash_dump_realized;
            if (used_dflash && dflash_dump_enabled_.load(std::memory_order_relaxed)) {
                dflash_dump_realized.resize(drafts.size() + 1);
                for (size_t i = 0; i <= drafts.size(); ++i) {
                    dflash_dump_realized[i] = pick_row(lane, sampler, logits, i);
                }
            }

            size_t accepted = 0;
            bool   stop     = false;
            // M11 step profile only: brackets the accept loop, which
            // otherwise had no timer at all (§1 row 12).
            const auto t_accept0 = profile_cycle ? clock::now() : clock::time_point{};
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
            const double accept_ms =
                profile_cycle
                    ? std::chrono::duration<double, std::milli>(clock::now() - t_accept0).count()
                    : 0.0;
            stats.draft_accepted += static_cast<int>(accepted);

            // M11 dump instrument: dflash_dump_realized (computed above,
            // before any commit() in this cycle) already covers every draft
            // row, not only the accepted prefix -- exactly what the offline
            // oracle needs (tools/dflash_oracle.py), which chains cycles of
            // one request using `lane`/`request_id`/`past` (
            // ) rather than trusting realized[] past the
            // rejection row of THIS cycle in isolation. `past` here is the
            // anchor's own position -- still unadvanced by this cycle's `past
            // += 1 + accepted` below -- so the oracle can assert
            // past_{c+1} == past_c + 1 + accepted_c across consecutive
            // records of the same (lane, request_id) and flag a gap
            // (a non-drafted decode step, or a dropped/reordered dump line)
            // instead of silently chaining through one. `accepted` on this
            // record is the true served count for THIS cycle; only the
            // LAST cycle of a request can have it cut short by max_tokens,
            // n_ctx or an EOS draft (the `stop` cases below) rather than by
            // rejection -- the oracle should not read a short final count as
            // evidence the drafter did worse than it did .
            if (used_dflash && dflash_dump_enabled_.load(std::memory_order_relaxed)) {
                nlohmann::json rec     = std::move(lane.dflash_dump_lattice);
                rec["lane"]            = lane.index;
                rec["request_id"]      = request_id;
                rec["past"]            = past;
                rec["arm"]             = {
                    {"mode", dflash_select_mode_ == DflashSelectMode::kViterbi ? "viterbi" : "greedy"},
                    {"lambda", dflash_lambda_},
                    {"topk", dflash_topk_},
                    {"block", dflash_block_},
                };
                rec["drafts_proposed"] = drafts;
                rec["accepted"]        = accepted;
                rec["realized"]        = dflash_dump_realized;
                dflash_dump_write(std::move(rec));
            }

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
            if (dflash_active(lane)) dflash_append(lane, 0, accepted + 1, past);

            // M11 step profile: one line per drafting cycle (ARCINT_PROFILE_CYCLE),
            // a strict superset of ARCINT_FORWARD_SPLIT=1 (which averages over 64
            // forwards and so mixes prefill's last chunks with decode's first
            // steps at exactly the depth of interest). Every number here was
            // already computed above under the same switch or is an
            // already-read timestamp (paged_forward's t0..t3, mirrored onto
            // the lane); no new default behaviour.
            if (profile_cycle) {
                int frag = 0;
                for (size_t i = 0; i + 1 < lane.blocks.size(); ++i) {
                    if (lane.blocks[i + 1] != lane.blocks[i] + 1) ++frag;
                }
                const double cycle_ms =
                    std::chrono::duration<double, std::milli>(clock::now() - t_propose).count();
                // lane.stall_accum alone can read 0 here (commit() flushes it
                // into lane.stalls and zeroes it every time it runs, and the
                // accept loop above may have called commit() one or more
                // times) -- add back whatever this cycle pushed there since
                // stalls_at_propose.
                double wait_ms = 1000.0 * lane.stall_accum;
                for (size_t i = stalls_at_propose; i < lane.stalls.size(); ++i) {
                    wait_ms += 1000.0 * lane.stalls[i];
                }
                // Residual by design: ensure_blocks, the drafter dump and the MTP promotion
                // loop sit inside cycle_ms but carry no field, so propose + verify + accept +
                // wait < cycle_ms; the cycle that ends a request breaks out before this line.
                const std::string line = format_profile_cycle_line(
                    past, seq.size(), accepted, propose_ms, cyc_mtp_embed_ms, cyc_mtp_mask_ms,
                    cyc_mtp_layer_ms, cyc_mtp_head_ms, vembed_ms, lane.last_fwd_ms_index,
                    lane.last_fwd_ms_infer, lane.last_fwd_ms_logits, lane.last_fwd_ms_hidden,
                    accept_ms, wait_ms, cycle_ms, frag);
                log::info("cycle", "%s", line.c_str());
            }

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
        // --dflash-select / --dflash-lambda: inert unless want_dflash_, but
        // cheap and config-only, so set unconditionally rather than only
        // inside the try block below.
        dflash_select_mode_ =
            cfg.dflash_select == "viterbi" ? DflashSelectMode::kViterbi : DflashSelectMode::kGreedy;
        dflash_lambda_ = cfg.dflash_lambda;
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
                // M11 (the M11 design note (not in the repository) Q): --dflash-block overrides the
                // json's block size. `noise` is dynamic in the exported graph
                // (tools/export_dflash.py), so nothing else here needs to
                // change -- drafts_max_ and the reservation's drafter term
                // below already read dflash_block_ after this point.
                if (cfg.dflash_block_set) {
                    log::info("dflash", "block size overridden by --dflash-block: %d "
                                        "(config.json said %zu)",
                              cfg.dflash_block, dflash_block_);
                    dflash_block_ = static_cast<size_t>(cfg.dflash_block);
                }
                dflash_mask_token_ = dcfg.at("mask_token_id").get<int>();
                dflash_topk_       = dcfg.at("selector_top_k").get<size_t>();
                if (cfg.dflash_topk_set) {
                    log::info("dflash", "selector top-k overridden by --dflash-topk: %d "
                                        "(config.json said %zu)",
                              cfg.dflash_topk, dflash_topk_);
                    dflash_topk_ = static_cast<size_t>(cfg.dflash_topk);
                }
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
        //
        // M8 (docs/design-m8-asymmetric-kv.md §3): --paged-kv now takes
        // KEY[:VALUE], and both the flag's own value and the env override go
        // through the identical parser (config.h's parse_paged_kv). The
        // env override used to map anything it did not recognise straight to
        // u8 -- ARCINT_PAGED_KV=i4 silently served u8 -- which is exactly the
        // quiet-downgrade class this engine exists to refuse; it now refuses
        // to load instead.
        std::string pk_key, pk_value;
        if (!parse_paged_kv(cfg.paged_kv, pk_key, pk_value)) {
            // Belt and braces: config.cpp's validate() already refuses this
            // before load_paged is ever reached.
            throw std::runtime_error(
                log::format("--paged-kv '%s' does not parse", cfg.paged_kv.c_str()));
        }
        std::string kv_src = "--paged-kv";
        if (const char* env = std::getenv("ARCINT_PAGED_KV")) {
            if (!parse_paged_kv(env, pk_key, pk_value)) {
                throw std::runtime_error(log::format(
                    "ARCINT_PAGED_KV='%s' does not parse: each side must be one of f16, u8, "
                    "i8, u4, i4 (KEY[:VALUE])",
                    env));
            }
            kv_src = "ARCINT_PAGED_KV";
        }
        const ov::element::Type kv_prec    = paged_kv_element(pk_key);
        const ov::element::Type value_prec = paged_kv_element(pk_value);
        const bool               asym_kv   = pk_key != pk_value;
        // The spec that actually won: cfg.paged_kv is the flag's own value,
        // but ARCINT_PAGED_KV may have overridden it above, and pk_key/
        // pk_value already reflect whichever source won. Diagnostics below
        // name THIS, not the flag (docs/design-m8-asymmetric-kv.md review,
        // ) -- an env override used to print the flag's value in the load
        // line and in both refusal messages, misnaming what was actually
        // requested.
        const std::string effective_paged_kv = asym_kv ? (pk_key + ":" + pk_value) : pk_key;
        // Said out loud at load, because it is a throughput decision now and not
        // only a memory one: u8 halves KV and costs up to 22% of prefill at
        // depth (§7.0.3 chose it on decode evidence, which did not see that).
        log::info("load", "paged KV precision %s (%s)%s", effective_paged_kv.c_str(),
                  kv_src.c_str(),
                  asym_kv ? ": asymmetric, VALUE_CACHE_PRECISION requested" : "");
        props["KV_CACHE_PRECISION"] = kv_prec;
        if (asym_kv) {
            // VALUE_CACHE_PRECISION does not exist on an unpatched plugin
            // (M8 patches/0004, not carried in this repository yet). Setting
            // it is still the right thing to attempt: the compile_model guard
            // below catches a plugin that rejects it outright, and the port
            // audit after compile catches the more dangerous case -- a
            // plugin that silently ignores the unknown property and mirrors
            // KV_CACHE_PRECISION onto both sides anyway.
            props["VALUE_CACHE_PRECISION"] = value_prec;
        }
        // 0015 engine side (design note o-0015-design.md, scratchpad,
        // section C): PAGED_ATTENTION_MAX_PARTITIONS is a patched plugin's
        // own RW config key bounding the mixed-stage paged-attention
        // kernel's partition count, default 0 = unbounded = today's
        // behaviour. Gated on `value_prec.bitwidth() == 4` -- this
        // repository's fit only ever charges the scratch term the bound
        // would shrink for packed-4-bit paged VALUES (packed_values_
        // prefill_scratch_bytes, exec/fit.h); there is nothing for a u8/f16
        // load to do with it here.
        //
        // Detection is a READ, not a set-and-let-compile-fail: unlike
        // VALUE_CACHE_PRECISION above (where a plugin that rejects the
        // property must fail the whole load -- an asymmetric KV request
        // silently downgraded to symmetric is a correctness bug, not a
        // missed optimisation), a plugin that cannot carry this bound must
        // NOT refuse the load -- --paged-attention-max-partitions defaults
        // to 0, so most loads never ask for a bound at all, and the honest
        // fallback is the depth-scaled term this file already charges.
        //
        // Round-6 review, finding 2 (comment correction): an earlier
        // version of this comment called `core_.get_property(device, ...)`
        // throwing on an unrecognised key "the same technique" as
        // device_resident_bytes's own read (above) and the total-memory
        // read in this function's own reservation (further down) --
        // overstated. Those two read KNOWN, TYPED OpenVINO GPU properties
        // (`ov::intel_gpu::memory_statistics`, `ov::intel_gpu::device_
        // total_mem_size`) that every GPU plugin instance supports, so
        // that call essentially never throws in practice; it is the same
        // FUNCTION, `core_.get_property`, but not the same situation --
        // this probe is a plain STRING key for a property that legitimately
        // does not exist on any plugin in this fleet today (no plugin
        // implementing this patch exists in this tree yet), so throwing is
        // the ORDINARY case here, not a rare failure mode. A typed
        // `ov::Property` object is not available for the same reason: this
        // repository carries no OpenVINO header declaring one for a plugin
        // patch that has not shipped.
        //
        // Probed and, if accepted, SET unconditionally -- even when
        // `cfg.paged_attention_max_partitions` is 0 (no bound requested):
        // the key's mere acceptance is how the patched plugin is detected.
        // `paged_attention_bound_accepted_` (a member, read again much
        // later at the fit-term climb and the belt call site, both well
        // after compile) is the one fact that call carries forward.
        //
        // Round-7 review (reverting round-6 review finding 2): a separate
        // read-only PAGED_ATTENTION_PARTIALS_ELEMENT_BYTES key was tried
        // here, to confirm the f16 host-sizing fix independently of the
        // bound key's own acceptance. Retracted: the plugin implementer
        // keeps 0015 as one unit and will not carry a second key for this.
        // DETECTION CONTRACT, stated plainly because nothing enforces it
        // in code: a plugin exposing PAGED_ATTENTION_MAX_PARTITIONS
        // carries the f16 host-sizing fix -- both ship together in patch
        // 0015, unconditionally. A plugin that exposed the bound key
        // without the sizing fix would be under-charged by half (4-byte
        // elements priced as 2-byte); no such plugin exists in this
        // series, so `element_bytes = 2` follows directly from
        // `paged_attention_bound_accepted_` below, with no second probe.
        paged_attention_bound_accepted_ = false;
        if (value_prec.bitwidth() == 4) {
            try {
                (void)core_.get_property(device, "PAGED_ATTENTION_MAX_PARTITIONS");
                paged_attention_bound_accepted_ = true;
            } catch (const std::exception&) {
                paged_attention_bound_accepted_ = false;
            }
            if (paged_attention_bound_accepted_) {
                // The plugin declares this key as Property<size_t, RW>;
                // `cfg.paged_attention_max_partitions` is `int` (config.h,
                // parsed and range-checked as a plain CLI integer). Cast
                // explicitly rather than let ov::Any hold the `int` --
                // this plugin's read-back is typed, and an `int` payload
                // can fail that typed read even where the same bit
                // pattern read back as `size_t` would succeed. The card
                // run that accepted this property did so before the cast
                // was made explicit; do not take that as proof an `int`
                // payload is safe on a future plugin build.
                props["PAGED_ATTENTION_MAX_PARTITIONS"] =
                    static_cast<size_t>(std::max(0, cfg.paged_attention_max_partitions));
            } else {
                log::info("load", "%s",
                          "4-bit values: plugin does not carry the bound; scratch term stays "
                          "depth-scaled");
            }
            // Patch 0020 engine side (DESIGN §7.0.2at): the level is read
            // from the plugin's own build number, not the core library's
            // -- a staged runtime can pair one level's core with another's
            // plugin, and the kernel path is the plugin's. `get_versions`
            // returns one entry per plugin the device name resolves to;
            // the highest stamped level found is taken (there is one).
            gpu_plugin_patch_level_             = 0;
            packed_values_mixed_stage_on_micro_ = false;
            try {
                for (const auto& kv : core_.get_versions(device)) {
                    const char* build = kv.second.buildNumber;
                    if (build == nullptr) continue;
                    gpu_plugin_patch_level_ =
                        std::max(gpu_plugin_patch_level_, marfrit_patch_level(build));
                }
            } catch (const std::exception&) {
                gpu_plugin_patch_level_ = 0;
            }
            packed_values_mixed_stage_on_micro_ = packed_values_mixed_stage_on_micro(
                gpu_plugin_patch_level_, static_cast<int>(kv_prec.bitwidth()),
                static_cast<int>(value_prec.bitwidth()));
            if (packed_values_mixed_stage_on_micro_) {
                log::info("load",
                          "4-bit values: plugin patch level %d runs the %s mixed prefill stage on "
                          "micro-SDPA (patch 0020); the generic kernel's scratch term is not "
                          "charged, the measured chunk cap (%d) stays",
                          gpu_plugin_patch_level_, effective_paged_kv.c_str(),
                          kMaxMeasuredPackedValuesChunk);
            } else {
                log::info("load",
                          "4-bit values: plugin patch level %d (%s); the %s mixed prefill stage "
                          "runs the generic kernel, scratch term charged",
                          gpu_plugin_patch_level_,
                          gpu_plugin_patch_level_ > 0 ? "below the 0020 level, or not its pairing"
                                                      : "no recipe stamp in the build number",
                          effective_paged_kv.c_str());
            }
        }
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
            // Experiment knob, ARCINT_FIT_SLOT_BYTES-style: forwards a device
            // pool budget to a plugin that understands it (patches/0005);
            // an unpatched plugin rejects the property and the load fails
            // loudly rather than silently measuring the wrong world.
            //
            // The env string used to go into the AnyMap as-is and the
            // plugin's own stoull() has no full-consumption check, so a typo
            // silently changed the request by orders of magnitude rather
            // than refusing (docs/design-m8-asymmetric-kv.md review,:
            // "8e9" became 8 bytes, "-1" wrapped to ULLONG_MAX). Parsed and
            // refused here instead, with a numeric size_t handed to the
            // plugin rather than a string for it to mis-parse a second way.
            if (const char* dp = std::getenv("ARCINT_MOE_DEVICE_POOL_BYTES")) {
                uint64_t bytes = 0;
                if (!parse_u64_strict(dp, bytes)) {
                    throw std::runtime_error(log::format(
                        "ARCINT_MOE_DEVICE_POOL_BYTES='%s' does not parse as a byte count "
                        "(need a plain base-10 integer: no sign, no trailing text)",
                        dp));
                }
                props["MOE_OTD_DEVICE_POOL_BYTES"] = static_cast<size_t>(bytes);
                log::info("load", "expert slot device pool budget forced to %llu bytes "
                          "(ARCINT_MOE_DEVICE_POOL_BYTES)",
                          static_cast<unsigned long long>(bytes));
            }
            // arcint M14. Same contract as MOE_OTD_DEVICE_POOL_BYTES above: an
            // unpatched plugin rejects the property and the load fails loudly
            // rather than silently serving the device-upload path while the run
            // is labelled "cpu tier on". A flag, not an env var, because this
            // changes what is served, not what is logged.
            if (moe_cpu_tier_) {
                // DESIGN §7.0.2ae's F0 used to refuse --moe-cpu-tier with
                // --prefix-cache-mib > 0 unconditionally, at config parse
                // time (config.cpp). F1 (a bit-equal host kernel) was
                // retired by review -- the device's per-lane scale/zp and
                // subgroup-tree reduction order make a faithful host match
                // unattainable at reasonable cost. Patch 0018 ships as F2
                // instead, and makes the refusal conditional a different
                // way: it turns the host/device split itself into a pure
                // function of the served configuration (a static
                // per-(layer, expert) partition) rather than
                // history-dependent LRU residency, and exposes a READ-ONLY
                // GPU-plugin property, MOE_CPU_TIER_STATIC_PARTITION, that
                // reports whether the tier actually serving this load is
                // that static partition (false when
                // MOE_CPU_TIER_PARTITION=lru selects the old,
                // history-dependent one; the property is simply absent on a
                // plugin without 0018).
                //
                // MEASURED (24 GB card, m18 plugin, clean equivalence gate
                // run, 2026-09-04): this fact is NOT answerable here, unlike
                // PAGED_ATTENTION_MAX_PARTITIONS (which the comment this one
                // replaced analogised to). Patch 0018 registers
                // MOE_CPU_TIER_STATIC_PARTITION only on CompiledModel
                // (compiled_model.cpp's property list; an ExecutionConfig
                // option, not a Plugin-level one) -- there is no such key on
                // `core_`/`device` at all, pre-compile.
                // `core_.get_property(device, "MOE_CPU_TIER_STATIC_PARTITION")`
                // always throws for this key, so the refusal that used to
                // live here always fired -- confirmed on real hardware: the
                // equivalence suite's own prefix-cache server refused to
                // load with "the plugin does not report a static residency
                // partition" even though the exact same m18 plugin, loaded
                // moments earlier without --prefix-cache-mib, correctly
                // logged "plugin reports a static residency partition" from
                // the OTHER call site below, which reads `paged_model_`
                // (the just-compiled model) instead of `core_`. The refusal
                // is applied there now, after compilation, the only point
                // this plugin fact is actually knowable -- see
                // "static_partition_reported" a few hundred lines down,
                // right after the plateau-probe-skip comment that first
                // diagnosed this exact mismatch without anyone circling
                // back to fix the earlier call it named.
                props["MOE_CPU_TIER"] = true;
                if (moe_cpu_tier_threads_ > 0)
                    props["MOE_CPU_TIER_THREADS"] = static_cast<size_t>(moe_cpu_tier_threads_);
                log::info("load", "MoE host compute tier enabled (threads=%s)",
                          moe_cpu_tier_threads_ > 0 ? std::to_string(moe_cpu_tier_threads_).c_str() : "auto");
            }
        }
        // The blob cache is off for the paged graph: its import path is
        // unproven and the #37607 class is exactly the kind of thing it would
        // hide. Revisit once the paged path is the only path.
        core_.set_property(ov::cache_dir(""));

        log::info("load", "compiling PAGED language model on %s (big model first by design)",
                  device.c_str());
        auto t0 = std::chrono::steady_clock::now();
        try {
            paged_model_ = core_.compile_model(model, device, props);
        } catch (const std::exception& e) {
            // Round-6 review, finding 3: the pre-0015 version of this catch
            // only ever named VALUE_CACHE_PRECISION -- correct while that
            // was the only speculative property this load could have set,
            // wrong once PAGED_ATTENTION_MAX_PARTITIONS joined it: a
            // plugin that reads the key back fine (the probe above
            // succeeded, so it went into `props`) but rejects it at
            // compile is a real, distinct failure this catch used to
            // misattribute to VALUE_CACHE_PRECISION whenever `asym_kv` was
            // also true, or report with no key named at all otherwise
            // (a bare `throw;`). Named explicitly now, and the compound
            // case (both properties speculatively set) says so rather than
            // guessing which one the plugin actually rejected.
            if (asym_kv && paged_attention_bound_accepted_) {
                throw std::runtime_error(log::format(
                    "asymmetric paged KV %s requested (VALUE_CACHE_PRECISION set) and "
                    "PAGED_ATTENTION_MAX_PARTITIONS was also set (this plugin's own read-back "
                    "probe accepted it), but compiling the paged model failed: %s -- refusing "
                    "rather than guessing which property the plugin actually rejected",
                    effective_paged_kv.c_str(), e.what()));
            }
            if (asym_kv) {
                // Red-first layer 1 (docs/design-m8-asymmetric-kv.md §4b): a
                // plugin that rejects an unknown VALUE_CACHE_PRECISION
                // outright must fail the load, never fall back to compiling
                // without it and quietly serving symmetric KV.
                throw std::runtime_error(log::format(
                    "asymmetric paged KV %s requested (VALUE_CACHE_PRECISION set), but "
                    "compiling the paged model failed: %s -- refusing rather than silently "
                    "falling back to symmetric %s KV",
                    effective_paged_kv.c_str(), e.what(), pk_key.c_str()));
            }
            if (paged_attention_bound_accepted_) {
                throw std::runtime_error(log::format(
                    "PAGED_ATTENTION_MAX_PARTITIONS was accepted by this plugin's read-back "
                    "probe and set to %d, but compiling the paged model failed: %s -- refusing "
                    "rather than silently falling back to the unbounded, depth-scaled scratch "
                    "term",
                    cfg.paged_attention_max_partitions, e.what()));
            }
            throw;
        }
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

        // M11 §2 (DESIGN §7.0.2ag): ARCINT_DRAFT_F32, read once here and
        // handed to both drafter compiles below -- presence-armed, like
        // every other ARCINT_* switch in this file; the decision itself
        // lives in draft_f32_enabled() (backend_stub.cpp), so it is testable
        // without a card or an OpenVINO build. No default change: props
        // stays empty unless the switch is set.
        const bool draft_f32 = draft_f32_enabled();
        ov::AnyMap draft_precision_props;
        if (draft_f32) {
            draft_precision_props[ov::hint::inference_precision.name()] = ov::element::f32;
        }

        // M11 §2 RoPE follow-up (DESIGN §7.0.2ag, rope_precision.h):
        // applied by DEFAULT, unlike every other switch in this file --
        // ARCINT_DRAFT_ROPE_F16 (presence-armed) is the A/B lever back to
        // the pre-fix state (rotary subgraph left in the plugin's default
        // f16), named for the state it restores rather than for the fix.
        const bool apply_rope_fix = !draft_rope_f16_enabled();

        if (want_mtp_) {
            try {
                const std::string layer_xml = choose_mtp_layer(artifact_, cfg.mtp_layer);
                // read_model() first (not a direct path compile) so the RoPE
                // marker below has a graph to walk before the plugin's own
                // ConvertPrecision pass runs inside compile_model().
                auto           mtp_layer_model = core_.read_model(layer_xml);
                const size_t   mtp_rope_marked =
                    apply_rope_fix ? mark_rope_no_fp16_compression(mtp_layer_model) : 0;
                mtp_layer_ = core_.compile_model(mtp_layer_model, mtp_dev, draft_precision_props);
                read_mtp_contract(mtp_layer_, layer_xml);
                mtp_head_  = core_.compile_model(artifact_.mtp_lm_head_xml, mtp_dev,
                                                 draft_precision_props);
                for (auto& lane : lanes_) {
                    lane->mtp_layer = mtp_layer_.create_infer_request();
                    lane->mtp_head  = mtp_head_.create_infer_request();
                }
                mtp_ready_ = true;
                log::info("mtp",
                          "head on %s, drafting one token per step%s; rope kept f32 on %zu "
                          "node(s)%s",
                          mtp_dev.c_str(),
                          draft_f32 ? " (ARCINT_DRAFT_F32: f32 inference precision)" : "",
                          mtp_rope_marked,
                          apply_rope_fix ? "" : " (ARCINT_DRAFT_ROPE_F16: fix disabled)");
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
                // M11 §2: ARCINT_DRAFT_F32 (draft_precision_props, above)
                // overrides this to f32 when set -- no default change. The
                // rotary subgraph is marked separately, unconditionally
                // (apply_rope_fix, above) -- see the MTP branch's own
                // comment for why read_model() runs first.
                const std::string dflash_xml =
                    cfg.dflash + "/openvino_dflash_draft_stateful.xml";
                auto         dflash_ir_model = core_.read_model(dflash_xml);
                const size_t dflash_rope_marked =
                    apply_rope_fix ? mark_rope_no_fp16_compression(dflash_ir_model) : 0;
                dflash_model_ =
                    core_.compile_model(dflash_ir_model, ddev, draft_precision_props);
                if (artifact_.mtp_lm_head_xml.empty()) {
                    throw std::runtime_error(
                        "the drafter scores through the target lm_head and this artifact "
                        "carries no extracted openvino_mtp_lm_head.xml");
                }
                dflash_head_model_ =
                    core_.compile_model(artifact_.mtp_lm_head_xml, ddev, draft_precision_props);
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
                log::info("dflash",
                          "block-%zu drafter on %s (%zu drafts per verify pass)%s; rope kept "
                          "f32 on %zu node(s)%s",
                          dflash_block_, ddev.c_str(), dflash_block_ - 1,
                          draft_f32 ? " (ARCINT_DRAFT_F32: f32 inference precision)" : "",
                          dflash_rope_marked,
                          apply_rope_fix ? "" : " (ARCINT_DRAFT_ROPE_F16: fix disabled)");
            } catch (const std::exception& e) {
                log::warn("dflash", "could not load the drafter, continuing without it: %s",
                          e.what());
                dflash_ready_ = false;
            }
        }

        // M7 §1 Phase A (the M7 design, "Auto-fit and the honest
        // reservation"): embeddings, the MTP head and the DFlash drafter all
        // compile after `resident_base` above, on this device by default, so
        // their weights were being charged to the activation probe further
        // down instead of to their own line -- not lost, just attributed to
        // the wrong one, and it inflated the slope that picks the chunk. A
        // second residency read isolates them into their own term.
        const size_t resident_with_drafters = device_resident_bytes(device);
        const size_t drafter_bytes = resident_with_drafters > resident_base
                                         ? resident_with_drafters - resident_base
                                         : 0;
        if (drafter_bytes > 0) {
            log::info("load", "drafters (embeddings/MTP/DFlash) resident: %.2f GiB",
                      static_cast<double>(drafter_bytes) / (1u << 30));
        }

        // M11 §1.3 (DESIGN §7.0.2ag): `resident_with_drafters` above is
        // read right after the drafter graphs COMPILE -- weights only, no
        // priming has run yet -- so it never charges for the MTP layer's
        // own stateful KV pair (KV_HEADS 4 x HEAD_DIM 256, K+V, f32, primed
        // over the WHOLE prompt by mtp_prime_paged), which grows with n_ctx
        // exactly like the paged KV pool itself and, uncharged, let a
        // served MTP arm overcommit the card ("resident 22.47 GiB against a
        // 22.46 GiB ceiling" at n_ctx 155,488). DFlash's own state is
        // windowed to kDflashWindow (2,048 rows) and stays a small,
        // depth-independent figure regardless of n_ctx, so it gets no term
        // here -- see fit.h's mtp_state_bytes for the full account of why
        // only the MTP layer needs one. Folded into the fit's own per-token
        // rate below (never into `kv_bytes_token_` itself, which Phase E's
        // allocation math still needs at the true KV-pool byte size).
        //
        // Gated on `mtp_dev == device`, not on `mtp_ready_` alone: --mtp-
        // device can place the MTP layer's compiled graph -- and therefore
        // its KV pair -- on a different device than the one this budget
        // pass is sizing. Charging every device's budget for state that
        // physically lives on only one of them double-counts it on every
        // OTHER device and can refuse a load that would actually fit.
        const uint64_t mtp_state_bytes_token =
            (mtp_ready_ && mtp_dev == device) ? lgc::kMtpStateBytesPerToken : 0;

        // ---- ports: state tables and KV pools --------------------------------
        size_t conv_i = 0, gdn_i = 0;
        // (bitwidth, element_count) per key_cache/value_cache port -- fed to
        // kv_block_bytes_from_bits (fit.h) below rather than summing
        // Type::size() per port, which ceils a sub-byte width to a whole byte
        // and would over-count an i4 side 2x (docs/design-m8-asymmetric-kv.md §3 bug 1).
        std::vector<std::pair<uint64_t, uint64_t>> kv_port_bits;
        // Port audit alias logging (below): once per side, not once per
        // layer -- a 40-layer model would otherwise print the same "i8 for a
        // u8 request" line 40 times.
        bool logged_key_alias = false, logged_value_alias = false;
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
                const bool is_value = name.rfind("value_cache.", 0) == 0;
                const ov::element::Type  actual         = port.get_element_type();
                const std::string        actual_name    = actual.get_type_name();
                const std::string&       requested_name = is_value ? pk_value : pk_key;
                // Port audit (docs/design-m8-asymmetric-kv.md §3 item 11 / §4b, the red case in the
                // unpatched world): VALUE_CACHE_PRECISION is a hint the
                // plugin may not know or may silently ignore, mirroring
                // KV_CACHE_PRECISION onto both ports instead of honouring the
                // asymmetric request. That is the quiet-downgrade class this
                // engine exists to refuse, so the compiled model's own port
                // types are checked against what was requested rather than
                // trusted on the strength of the property having been set.
                //
                // The check is BITWIDTH, not exact element type (config.h's
                // kv_precision_bitwidth_matches): measured on the card
                // 2026-09-02, an unpatched plugin compiles a u8 request as
                // i8 storage -- same 8 bits, its own signedness choice, not
                // a downgrade -- and an exact-type check refused every
                // symmetric u8 load over exactly that. Amended the same day:
                // this plugin generation types paged KV ports at 8 bits
                // ALWAYS, even for a 4-bit (u4/i4) request -- an honest
                // 4-bit-typed port breaks the plugin's own compile -- so a
                // 4-bit request landing on an 8-bit port also passes now
                // (logged below, not silently). A different bitwidth outside
                // that one known shape (a request silently kept at the old
                // width) still refuses.
                if (!kv_precision_bitwidth_matches(requested_name, actual_name)) {
                    throw std::runtime_error(log::format(
                        "paged KV port '%s' compiled as %s but --paged-kv '%s' requested "
                        "%s=%s; refusing to serve a KV precision the plugin did not actually "
                        "give us (likely: this plugin does not support asymmetric paged KV)",
                        name.c_str(), actual_name.c_str(), effective_paged_kv.c_str(),
                        is_value ? "value" : "key", requested_name.c_str()));
                }
                if (actual_name != requested_name) {
                    bool& logged = is_value ? logged_value_alias : logged_key_alias;
                    if (!logged) {
                        if (kv_precision_is_packed_four_bit(requested_name, actual_name)) {
                            log::info("load",
                                      "%s cache requested %s: stored in 8-bit-typed ports on "
                                      "this plugin generation (packing is config-side and not "
                                      "visible at port level; verify sub-byte packing by "
                                      "measurement, not by this audit)",
                                      is_value ? "value" : "key", requested_name.c_str());
                        } else {
                            log::info("load",
                                      "paged KV %s cache compiled as %s for requested %s (same "
                                      "bitwidth -- the plugin's own storage choice, not a "
                                      "downgrade)",
                                      is_value ? "value" : "key", actual_name.c_str(),
                                      requested_name.c_str());
                        }
                        logged = true;
                    }
                }
                ov::Shape sh;
                sh.push_back(1);  // blocks dim, filled at allocation
                for (size_t i = 1; i < static_cast<size_t>(ps.rank().get_length()); ++i) {
                    sh.push_back(static_cast<size_t>(ps[static_cast<int64_t>(i)].get_length()));
                }
                kv_pool_names_.push_back(name);
                kv_pool_types_.push_back(actual);
                kv_pool_shapes_.push_back(sh);
                size_t block_elems = 1;
                for (size_t i = 1; i < sh.size(); ++i) block_elems *= sh[i];
                // Cost model: the compiled port's own bitwidth x element
                // count, unconditionally -- no requested-precision override.
                //
                // RETRACTED (measurement discipline, DESIGN §7.0.1: a
                // mechanism that is narrated and not measured gets retracted
                // on the record rather than edited away). Two on-card
                // rounds tried a requested-precision override here and both
                // were wrong:
                //   (1) first version: override to the requested bitwidth
                //       whenever the request was 4-bit, regardless of the
                //       actual port width -- wrongly charged 4 bits to
                //       genuinely 16/32-bit scale/auxiliary pool ports too.
                //   (2) narrowed version: override only when actual was
                //       8-bit (the known 4-in-8 TYPE-level packing shape).
                //       Window N measured --paged-kv u4 STILL printing 3.2
                //       KiB/token after this narrowing -- the number did not
                //       move. Solving KV_pure/4 + fixed = 3.2 against
                //       KV_pure + fixed = 11.3 (the u8 baseline) gives
                //       KV_pure ~= 10.8 KiB/token, fixed ~= 0.5; WITHOUT any
                //       override the predicted figure is KV_pure/2 + fixed
                //       ~= 5.9, inside the expected ~5.65-5.9 band. So this
                //       plugin generation encodes 4-bit packing in the port
                //       SHAPE (element count already halved for a 4-bit
                //       request), not only the type -- type x shape
                //       accounting (this line, unmodified) was already
                //       correct, and the override was double-halving.
                //       Corroborated separately (Window L): an honest
                //       i4-TYPED port attempt showed arcint's own tensor
                //       sized exactly 2x the plugin's -- direct evidence of
                //       shape-level packing.
                // The AUDIT above (kv_precision_bitwidth_matches /
                // kv_precision_is_packed_four_bit) is unaffected and stays:
                // TYPE-level 4-in-8 aliasing is real and still has to pass
                // without refusing a legitimate request. Only the COST
                // model's separate override was wrong, and is gone.
                //
                // Port classification (unchanged): this line only runs for
                // ports whose name starts with "key_cache." or
                // "value_cache." (the `else if` above) -- the same
                // classification kv_pool_names_/kv_pool_types_ use just
                // below. conv_state_table./gated_delta_state_table./
                // position_ids ports are handled in separate branches and
                // never reach kv_port_bits at all.
                kv_port_bits.emplace_back(actual.bitwidth(), block_elems);
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
        const size_t kv_block_bytes = static_cast<size_t>(kv_block_bytes_from_bits(kv_port_bits));
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
        const size_t margin = static_cast<size_t>(cfg.fit_margin_mib) << 20;
        const int    wanted = req_n_ctx > 0 ? req_n_ctx : artifact_.n_ctx_train;
        const int    lanes  = lane_count_;

        // Chunk floor, hoisted here (out of the climb below) because Phase B
        // needs it too: the plateau probe fallback saturates the expert pool
        // at the SAME floor_c the activation climb starts from, and must run
        // first (§1 "B must precede C").
        const size_t configured = prefill_chunk_ > 0 ? static_cast<size_t>(prefill_chunk_) : 512;
        const size_t floor_c    = std::min<size_t>(configured, 128);

        // Round-8 review (Opus): moved EARLY, before any activation
        // probing -- the served chunk must be known before the one real
        // probe runs (see the activation climb's own comment, below, for
        // why). `cap_off` mirrors the belt call site's own
        // ARCINT_PREFILL_CHUNK_CAP=off measurement switch. `element_bytes`
        // follows the RW bound key's own acceptance directly (the
        // detection contract stated at the probe site above: a plugin
        // exposing PAGED_ATTENTION_MAX_PARTITIONS carries the f16 host
        // sizing too, both ship together in patch 0015) -- read from the
        // plugin source once that patch exists in this tree, not a second
        // probe.
        const char* cap_env       = std::getenv("ARCINT_PREFILL_CHUNK_CAP");
        const bool  cap_off       = cap_env != nullptr && std::string(cap_env) == "off";
        const bool  packed_values = value_prec.bitwidth() == 4;
        std::optional<PackedValuesScratchGeometry> packed_geom;
        if (packed_values) packed_geom = packed_values_scratch_geometry(artifact_.config);
        const int paged_attention_element_bytes =
            paged_attention_bound_accepted_ ? 2 : 4;
        const int paged_attention_max_partitions_effective =
            paged_attention_bound_accepted_ ? cfg.paged_attention_max_partitions : 0;

        // Round-9 review (Opus), finding 2: computed HERE, before the
        // floor probe below (`act128`) ever runs -- not merely before the
        // activation-doubling climb, which is what round-8's own version
        // of this comment claimed but did not do (the floor probe itself
        // still ran first, so "the one probe runs at the served chunk" was
        // false whenever the served chunk turned out smaller than the
        // floor). Round-9 also retracts round-8's own auto-fit ANALYTIC
        // PRE-PASS SEARCH (fit_context_packed_values_chunk_ceiling,
        // fit.h) -- it failed open on the card's own constants and, even
        // fixed, could not be made to land near the real answer (see that
        // function's own retraction comment for the full account).
        //
        // Round-10 review (Opus), finding 1 (REAL defect, retracts round-
        // 9's own claim just below this line in earlier revisions of this
        // comment, that "BOTH paths call fit_context_packed_values_at_
        // depth at wanted, the literal same call for both"): `wanted` is
        // the artifact's train MAXIMUM, not auto-fit's served depth --
        // evaluating the belt there gives the smallest, most conservative
        // chunk (measured: chunk 32 at wanted 262,144), while the depth
        // auto-fit actually goes on to adopt can be far shallower and
        // genuinely afford a bigger, cheaper chunk (measured: chunk 64 at
        // the adopted 103,040) -- an explicit request for that SAME
        // adopted depth priced the bigger chunk correctly, so auto-fit
        // was charging a different, more expensive reservation than an
        // identical explicit request would for the identical served
        // depth. `chunk_ceiling` computed here is still exactly this:
        // belt-at-wanted, the smallest safe starting point -- but it is
        // now used ONLY as the ceiling the one real activation probe
        // below must not explore past (this repository has no cheap way
        // to probe activations more than once per chunk before compile),
        // never as the packed-values term's own final priced chunk. The
        // auto-fit branch below (`fit_context_packed_values`, the belt's
        // own chunk-driven fixed-point SEARCH, seeded from `configured`)
        // finds the real served chunk c*, which is always >= this ceiling
        // (D* <= wanted, and the belt is non-increasing in depth for a
        // fixed requested_chunk) -- and re-probes activations UPWARD when
        // c* turns out bigger than what was measured here (valid: the
        // plugin's intermediate pool only ever grows to the largest shape
        // it has seen, so a probe at a bigger chunk than before is a real
        // measurement, not the round-7 defect, which probed downward into
        // a stale reading). See that branch's own comment, further down,
        // for the full loop.
        // Round-12 review (Opus), finding 1 (HIGH, REAL defect):
        // `ARCINT_PREFILL_CHUNK_CAP=off` is the measurement switch that
        // exists so an operator can reproduce the plugin's own fault line
        // at the FULL requested chunk, belt disabled entirely (see the
        // Phase E belt call site's own `cap_off` gate, further down) --
        // but this ceiling used to apply the belt UNCONDITIONALLY,
        // regardless of `cap_off`, so the one real activation probe below
        // (and Phase B's plateau probe, which shares this same ceiling)
        // never got to explore past whatever the belt picked at `wanted`
        // even with the switch set. Measured: `--prefill-chunk 2048` with
        // the switch on logged "the belt is disabled for this load" and
        // then still served chunk 128 -- the switch was a no-op. Gated on
        // `!cap_off` now; `chunk_ceiling` keeps its already-initialized
        // `configured` value (no ceiling narrower than the full requested
        // chunk) whenever the switch is set.
        size_t chunk_ceiling = configured;
        if (packed_values && packed_geom && !cap_off && packed_values_mixed_stage_on_micro_) {
            // Patch 0020 arm: no buffer to price at any depth, so the
            // ceiling is the measured cap alone (fit.h's own arm does the
            // same for the term below).
            chunk_ceiling = static_cast<size_t>(std::max(
                packed_values_measured_chunk_cap(static_cast<int>(configured), cfg.kv_block_size),
                1));
            log::info("load",
                      "4-bit values: served-chunk ceiling %zu (the measured cap; the mixed stage "
                      "runs on micro-SDPA, no scratch buffer to price at n_ctx %d)",
                      chunk_ceiling, wanted);
        } else if (packed_values && packed_geom && !cap_off) {
            const long long partitions_at_wanted = packed_values_bounded_partitions(
                wanted, paged_attention_max_partitions_effective);
            const int chunk_at_wanted = prefill_chunk_cap_for_packed_values_ex(
                static_cast<int>(configured), partitions_at_wanted, packed_geom->heads,
                packed_geom->head_size, paged_attention_element_bytes,
                kPrefillScratchBudgetBytesPackedValues, cfg.kv_block_size);
            chunk_ceiling = static_cast<size_t>(std::max(chunk_at_wanted, 1));
            log::info("load",
                      "4-bit values: served-chunk ceiling %zu (belt evaluated at n_ctx %d), "
                      "the activation probe below will not explore past it",
                      chunk_ceiling, wanted);
        }

        // Round-10 review (Opus), finding 4: bounded by `chunk_ceiling`
        // too, not just `configured` -- this is what BOTH Phase B's
        // plateau probe (immediately below, `--offload-ratio > 0` only)
        // and the activation climb's own floor probe (further down) run
        // at, so a 4-bit-values load whose belt ceiling is smaller than
        // 128 tokens gets ONE consistent floor across both probes, not
        // Phase B measuring at a bigger, unbounded floor_c while the
        // activation climb measures at the capped one. An earlier version
        // of this comment (round-9) said "floor_c itself is left alone,
        // Phase B's own plateau-probe fallback is not part of this
        // review" -- that was the actual defect this finding fixes: Phase
        // B's probe could itself already exceed the served chunk (the
        // same failure mode the floor probe below was already fixed for).
        const size_t probe_floor_c = std::min(floor_c, chunk_ceiling);

        size_t       probe_pool_blocks = 0;

        // ---- Phase B: expert slot pool (M7 §1/§2, reworked after on-card
        // measurement) ------------------------------------------------------
        //
        // Two ledgers, not one. Measured on the card (35B q4, 16 GiB, u8 KV,
        // 1 lane, --offload-ratio 20): device VRAM after load was 6.69 GiB
        // against a 6.81 GiB promise (0.8% off) while a ~12 GiB expert pool
        // sat in GTT -- host RAM mapped for the GPU, confirmed via
        // /proc/<pid>/fdinfo (drm-total-gtt=13.17 GiB vs drm-total-vram0=
        // 6.69 GiB). So the plateau probe's small figure IS the correct
        // DEVICE charge (the LRU working set actually resident in VRAM), and
        // the (100-ratio)% ceiling estimate -- from the IR walk or from
        // config.json -- prices the HOST-side pool, not the device one.
        // Charging the device budget with the ceiling estimate is exactly
        // the failure this rework removes: it is the same shape of mistake
        // M7 itself was written to fix, just moved one term over.
        //
        // Only when offload is on; otherwise neither ledger has anything to
        // reserve and both stay exactly zero, not an estimate of zero.
        uint64_t    slot_pool         = 0;  // device (VRAM) charge -- goes into the budget
        std::string slot_source;            // "forced" | "probe" | "probe-static" | "static"
        bool        probe_priced_device = false;  // the plateau probe ran and set slot_pool
        uint64_t    slot_host_bytes   = 0;  // host (GTT) estimate -- informational only
        std::string slot_host_source;       // "ir" | "config"
        if (offload_ratio_ > 0) {
            // §4 "the on-card red case": forces the DEVICE term for an A/B
            // against the exact bug M7 removes. Checked first and, when
            // valid, short-circuits the rest of Phase B entirely -- no
            // probe, no analytic walk, no host-side estimate -- because a
            // forced figure is a deliberate override, not something a
            // multi-minute probe should still be spent measuring around.
            // ARCINT_FIT_SLOT_BYTES=0 on an --offload-ratio run reproduces
            // the pre-M7 arithmetic on today's binary, so a green full-depth
            // run at the real figure and an OOM (or a refusal) at the forced
            // 0 are the same binary proving the fix does something. An env
            // var and not a flag, like ARCINT_PAGED_KV: an experiment knob,
            // not something an operator sets in production.
            bool forced = false;
            if (const char* env = std::getenv("ARCINT_FIT_SLOT_BYTES")) {
                char*                    end = nullptr;
                const unsigned long long v   = std::strtoull(env, &end, 10);
                if (end != env && *end == '\0') {
                    slot_pool   = v;
                    slot_source = "forced";
                    forced      = true;
                    log::info("load",
                              "expert slot pool forced to %.2f GiB by ARCINT_FIT_SLOT_BYTES "
                              "(source: forced) -- Phase B probe and analytic walk skipped",
                              static_cast<double>(v) / (1u << 30));
                } else {
                    log::warn("load", "ARCINT_FIT_SLOT_BYTES='%s' is not a valid integer; ignored",
                              env);
                }
            }

            // Patch 0018 (MOE_CPU_TIER_STATIC_PARTITION, DESIGN
            // §7.0.2ae "F2"): the plateau probe below saturates the
            // device pool by observing EVICTION -- it relies on the
            // driver evicting a cold slot to make room for a new one,
            // and reads the resulting high-water mark. The static
            // partition pins the entire device slot pool at bind time
            // (patches/0018-moe-cpu-tier-static-partition.patch: no
            // slot is ever evicted -- that is the whole point, it is
            // what makes output history-independent), so under that
            // partition every probe chunk would drive `evict_one` into
            // its own "no evictable (unpinned) slot" throw and the load
            // would die before this file gets a chance to run it. (That
            // throw was bug #2's -- acquire_one() reached from the probe's
            // own prefill chunks. With bug #2 fixed, non-resident experts
            // take the host tier without an upload and nothing evicts, so
            // the probe runs under the static partition too; see the probe
            // block below, and why it must.)
            // Detected the same READ-ONLY property key the
            // --prefix-cache-mib refusal (below, moved here 2026-09-04)
            // needs, but NOT the same call: patch 0018 registers
            // MOE_CPU_TIER_STATIC_PARTITION on CompiledModel (compiled_
            // model.cpp's own property list, ov::intel_gpu::moe_cpu_
            // tier_static_partition is an ExecutionConfig option, not a
            // Plugin-level one) -- there is no such key on `core_`/
            // `device` at all, pre-compile. `core_.get_property(device,
            // ...)` -- what the --prefix-cache-mib gate used to call,
            // before this fix -- always throws for this key, so that
            // refusal always fired regardless of the plugin (measured
            // on the 24 GB card, a clean equivalence-suite run,
            // 2026-09-04: refused with "the plugin does not report a
            // static residency partition" even though this exact m18
            // plugin, loaded moments earlier without
            // --prefix-cache-mib, correctly reported it true right
            // here). `paged_model_` -- the model this function has just
            // finished compiling, a few lines above -- is what actually
            // answers it, so the refusal moved to read it from this
            // point instead.
            //
            // This block sits OUTSIDE `if (!forced)` on purpose: the §3.4
            // refusal is a safety gate on the loaded plugin/config pair,
            // not part of the sizing strategy ARCINT_FIT_SLOT_BYTES
            // bypasses. Nesting it under `!forced` used to mean a forced
            // slot-pool figure could load `--moe-cpu-tier
            // --prefix-cache-mib` against a non-static (LRU) plugin with no
            // refusal at all -- fail-open on the exact check this section
            // exists to guarantee. `static_partition_reported` is still
            // needed unconditionally below, forced or not, to decide
            // whether the device-term guard a few lines down applies.
            bool static_partition_reported = false;
            if (moe_cpu_tier_) {
                try {
                    static_partition_reported =
                        paged_model_.get_property("MOE_CPU_TIER_STATIC_PARTITION").as<bool>();
                } catch (const std::exception&) {
                    static_partition_reported = false;
                }
                if (cfg.prefix_cache_mib > 0) {
                    if (const auto refusal = tier_prefix_cache_decision(
                            static_partition_reported, moe_cpu_tier_, cfg.prefix_cache_mib)) {
                        throw std::runtime_error(*refusal);
                    }
                    log::info("load", "%s",
                              "host tier: plugin reports a static residency partition; "
                              "the prefix cache is allowed");
                }
            }

            if (!forced) {
                {
                    // Device term: the plateau probe, under BOTH residency
                    // partitions. The analytic IR walk (slot_pool_from_ir) is
                    // not used to charge the device budget -- measured on the
                    // card, its ceiling formula lands on the HOST figure, not
                    // what the driver actually keeps in VRAM. That held for
                    // the LRU partition since M7, and it holds for the static
                    // partition as well, measured 2026-09-05 (16 GiB card,
                    // 35B, ratio 50, 8 GiB pool): charging the whole pinned
                    // pool analytically (7.50 GiB) refused a configuration
                    // the card demonstrably serves, because the driver keeps
                    // most of that pool host-mapped there -- the probe under
                    // LRU had read 0.11 GiB resident for the same pool.
                    // Admission is charged with what is resident, so the
                    // probe runs; under the static partition it saturates as
                    // the pinned set fills on first use instead of by
                    // eviction (bug #2's fix routes non-resident experts to
                    // the host tier, nothing evicts) and terminates on the
                    // same plateau. The analytic figure is the fallback only
                    // if the probe throws under the static partition, and is
                    // logged beside the probe's reading as the cross-check.
                    bool probe_ok = true;
                    try {
                        const size_t probe_blocks =
                            (probe_floor_c + kv_block_tokens_ - 1) / kv_block_tokens_ + 1;
                        if (probe_blocks != probe_pool_blocks) {
                            alloc_kv_pools(rctx, probe_blocks);
                            probe_pool_blocks = probe_blocks;
                        }
                        lane0.blocks.resize(probe_blocks);
                        for (size_t i = 0; i < probe_blocks; ++i) {
                            lane0.blocks[i] = static_cast<int32_t>(i);
                        }

                        long long high_water = 0;
                        long long prev       = 0;
                        int       plateaued  = 0;
                        // A 128-token prefill routes min(num_expert, 128*top_k)
                        // experts per layer (§2), so saturation takes few chunks;
                        // 8 distinct-token probes is headroom over that, not a
                        // tuning knob -- there is deliberately no flag for this
                        // count (§3, "no probe-tuning flag: termination is a
                        // plateau, not a count").
                        for (int iter = 0; iter < 8 && plateaued < 2; ++iter) {
                            zero_paged_rows(lane0);
                            std::vector<int> ids(probe_floor_c);
                            // Distinct ids, so routing actually varies -- the
                            // all-zero activation probe below would hit the same
                            // handful of experts every time and never saturate
                            // the pool. Kept well under any real tokenizer's
                            // vocabulary so this never indexes past the
                            // embedding table.
                            for (size_t i = 0; i < probe_floor_c; ++i) {
                                ids[i] = static_cast<int>(
                                    (static_cast<size_t>(iter) * probe_floor_c + i) % 1000);
                            }
                            paged_forward(lane0, embed_paged(lane0, ids), 0, {0, 0}, 0);
                            const long long now = static_cast<long long>(
                                                      device_resident_bytes(device)) -
                                                  static_cast<long long>(resident_with_drafters);
                            plateaued = now <= prev ? plateaued + 1 : 0;
                            prev       = now;
                            high_water = std::max(high_water, now);
                        }
                        lane0.blocks.clear();
                        slot_pool = static_cast<uint64_t>(std::max<long long>(high_water, 0));
                    } catch (const std::exception& e) {
                        log::warn("load", "expert slot pool plateau probe failed: %s", e.what());
                        reset_lane_request(lane0);
                        probe_ok = false;
                    }
                    if (probe_ok) {
                        slot_source = static_partition_reported ? "probe-static" : "probe";
                        log::info("load",
                                  "expert slots: plateau probe settled at %.2f GiB after "
                                  "distinct-token chunks (source: %s)",
                                  static_cast<double>(slot_pool) / (1u << 30),
                                  slot_source.c_str());
                    } else if (static_partition_reported) {
                        log::warn("load", "%s",
                                  "expert slot pool: the plateau probe failed under the static "
                                  "residency partition; falling back to the analytic pinned-pool "
                                  "figure below, an UPPER bound (it counts the whole pinned pool "
                                  "as VRAM-resident, which the driver need not honour)");
                    } else if (!cfg.n_ctx_explicit) {
                        throw std::runtime_error(
                            "--offload-ratio > 0: the engine could not size the expert slot pool "
                            "with a plateau probe, and no --n-ctx was given to hold it to "
                            "explicitly. Pass --n-ctx, or drop --offload-ratio.");
                    } else {
                        log::warn("load", "%s",
                                  "expert slot pool unpriced; proceeding on the explicit --n-ctx "
                                  "alone (the fit below is verify-only and cannot promise this "
                                  "term is honest)");
                    }
                    probe_priced_device = probe_ok;
                }

                // Host-side ledger (GTT): informational, never charged
                // against the device budget. The analytic IR walk gives the
                // real per-expert weight bytes when it can find the MoE ops;
                // otherwise 3 x hidden x moe_intermediate x bytes-per-weight
                // per expert from config.json, at the same ceiling slot
                // count. Neither is a second source of truth for the device
                // term -- this cross-check informs the host line only.
                if (const auto ir = slot_pool_from_ir(model, artifact_.n_expert, offload_ratio_)) {
                    slot_host_bytes  = ir->total_bytes;
                    slot_host_source = "ir";
                    log::info("load",
                              "expert slots host-side: %.2f GiB (GTT, source: ir, %d per MoE "
                              "layer x %d layers x %.2f MiB)",
                              static_cast<double>(slot_host_bytes) / (1u << 30), ir->slots,
                              ir->moe_layers, static_cast<double>(ir->per_expert_bytes) / (1u << 20));
                    if (static_partition_reported && !probe_priced_device) {
                        // Analytic fallback (the probe threw): the same total
                        // this ledger just priced as a host-side ceiling
                        // (ir->total_bytes is built from repeated
                        // expert_slot_bytes calls, one per matched MoE
                        // layer). Under the static partition it is the
                        // pinned pool's full size -- an upper bound on what
                        // the driver keeps in VRAM, not a measurement of it.
                        slot_pool   = ir->total_bytes;
                        slot_source = "static";
                        log::info("load",
                                  "expert slot pool: device figure %.2f GiB (static, analytic "
                                  "fallback, upper bound, source: ir)",
                                  static_cast<double>(slot_pool) / (1u << 30));
                    } else if (static_partition_reported) {
                        log::info("load",
                                  "expert slot pool: pinned-pool ceiling %.2f GiB (source: ir) "
                                  "against %.2f GiB measured resident (source: probe-static) -- "
                                  "the difference is what the driver keeps host-mapped",
                                  static_cast<double>(ir->total_bytes) / (1u << 30),
                                  static_cast<double>(slot_pool) / (1u << 30));
                    }
                } else {
                    const json& tc = artifact_.config.contains("text_config") &&
                                             artifact_.config["text_config"].is_object()
                                         ? artifact_.config["text_config"]
                                         : artifact_.config;
                    const size_t hidden = static_cast<size_t>(artifact_.n_embd);
                    size_t       moe_intermediate = 0;
                    if (tc.contains("moe_intermediate_size") &&
                        tc["moe_intermediate_size"].is_number_integer()) {
                        moe_intermediate = tc["moe_intermediate_size"].get<size_t>();
                    }
                    if (hidden > 0 && moe_intermediate > 0 && artifact_.n_expert > 0 &&
                        artifact_.n_layer > 0) {
                        const double bytes_per_weight = cfg.quant == Quant::Q8 ? 1.0 : 0.5;
                        const uint64_t per_expert_config = static_cast<uint64_t>(
                            3.0 * static_cast<double>(hidden) *
                            static_cast<double>(moe_intermediate) * bytes_per_weight);
                        slot_host_bytes = expert_slot_bytes(artifact_.n_expert, offload_ratio_,
                                                            per_expert_config, artifact_.n_layer);
                        slot_host_source = "config";
                        log::info("load",
                                  "expert slots host-side: %.2f GiB (GTT, source: config)",
                                  static_cast<double>(slot_host_bytes) / (1u << 30));
                        if (static_partition_reported && !probe_priced_device) {
                            // Analytic fallback (the probe threw): the same
                            // formula -- expert_slot_bytes_static is
                            // expert_slot_bytes verbatim, named separately so
                            // this call site reads as pricing the device
                            // term. An upper bound, see the ir branch above.
                            slot_pool = expert_slot_bytes_static(artifact_.n_expert, offload_ratio_,
                                                                 per_expert_config, artifact_.n_layer);
                            slot_source = "static";
                            log::info("load",
                                      "expert slot pool: device figure %.2f GiB (static, analytic "
                                      "fallback, upper bound, source: config)",
                                      static_cast<double>(slot_pool) / (1u << 30));
                        } else if (static_partition_reported) {
                            log::info("load",
                                      "expert slot pool: pinned-pool ceiling %.2f GiB (source: "
                                      "config) against %.2f GiB measured resident (source: "
                                      "probe-static) -- the difference is what the driver keeps "
                                      "host-mapped",
                                      static_cast<double>(slot_host_bytes) / (1u << 30),
                                      static_cast<double>(slot_pool) / (1u << 30));
                        }
                    }
                }

                // §4 sanity note (replaces the old "under half -> override"
                // rule, which assumed both figures priced the same memory).
                // They do not: the device probe and the host estimate are
                // two different pools now, so a probe far below the host
                // estimate is the EXPECTED shape -- the LRU working set is a
                // small fraction of everything that could be paged in --
                // and is logged rather than corrected.
                if (slot_host_bytes > 0 && slot_pool < slot_host_bytes / 20) {
                    log::info("load",
                              "expert slot pool: device figure %.2f GiB (%s) is under 5%% of "
                              "the host-side estimate %.2f GiB -- expected (the device term is "
                              "the LRU working set; the host term is what could be paged in); "
                              "keeping the device figure for the budget",
                              static_cast<double>(slot_pool) / (1u << 30),
                              slot_source.empty() ? "unpriced" : slot_source.c_str(),
                              static_cast<double>(slot_host_bytes) / (1u << 30));
                }

                // A static partition pins the device slot pool by
                // definition. When the probe could not run and the analytic
                // fallback ALSO left slot_pool at 0, this ledger's own
                // arithmetic (the IR walk or the config-derived formula,
                // above) could not price it -- not that the plugin has
                // nothing pinned. Charging the device budget 0 bytes for a
                // pool the plugin is actually filling is exactly the M7
                // under-count this section exists to remove, so that case is
                // a hard refusal. A probe that RAN and read ~0 is a
                // measurement (the driver kept the pool host-mapped), not an
                // unpriced term, and is charged as read.
                if (static_partition_reported && !probe_priced_device && slot_pool == 0) {
                    throw std::runtime_error(
                        "--moe-cpu-tier: the plugin reports a static residency "
                        "partition but this ledger could not price the device slot "
                        "pool (neither the IR walk nor the config-derived formula "
                        "found the model's MoE shape) -- refusing to load with an "
                        "unpriced, but non-zero, pinned device allocation. Pass "
                        "ARCINT_FIT_SLOT_BYTES to override with a known figure.");
                }
            }
        }

        // The baseline every activation probe below measures against:
        // resident_base plus everything Phase A and B have already
        // committed to the device (drafters always; the GDN slab rows via
        // alloc_la_rows above; the device slot pool too, whenever the
        // plateau probe -- not a forced value, which touches nothing --
        // priced it).
        // Using the ORIGINAL resident_base here would charge those bytes to
        // the first activation probe's delta instead of to their own
        // term -- exactly the M7 §0 under-count, reproduced for a second
        // term if this read were skipped.
        //
        // `baseline_probe_pool_blocks` is a second, narrower snapshot: how
        // large the KV probe pool already was at the instant
        // resident_committed was read (Phase B's probe_floor_c pool when the
        // plateau probe ran; 0 otherwise). The pool's own bytes are already
        // inside resident_committed and inside every later `after` reading,
        // so a probe call must cancel only the CHANGE in pool size since the
        // baseline, not the pool's whole current size -- see the review
        // finding this replaces : the previous code subtracted the full
        // probe_blocks*kv_block_bytes every call, which double-subtracted a
        // pool that a prior Phase B probe (or an earlier climb step) had
        // already folded into the baseline.
        const size_t resident_committed        = device_resident_bytes(device);
        const size_t baseline_probe_pool_blocks = probe_pool_blocks;

        // Probe SMALL first -- a probe at the configured chunk can itself OOM on
        // a tight card (observed on the A770: sometimes the driver spills,
        // sometimes CL_OUT_OF_RESOURCES kills the process). The peak is linear
        // in the chunk (§7.0.2a), so a 128-token probe fixes the slope, the
        // largest admissible chunk is computed, and one guarded probe verifies
        // it, stepping down on failure instead of dying.
        //
        // The slab is NOT subtracted here (fix): alloc_la_rows above
        // already committed every lane's GDN checkpoint rows before
        // resident_committed was read, so the slab is already inside the
        // baseline. Subtracting it a second time is what produced the
        // measured negative activation deltas (-0.27 GiB at slab 0.296 on a
        // 24 GB card, -0.09 at GDN 0.093 on a 16 GB card) that an earlier
        // pass here wrongly attributed to LRU eviction noise -- there is no
        // eviction in this code path; it was a baseline accounting error.
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
            // Cancel only the KV probe pool's CHANGE since the baseline
            // (current size minus what was already resident when
            // resident_committed was read), not its whole current size --
            // the difference is nonzero only when this call, or an earlier
            // one in this same climb, actually grew the pool past what
            // Phase B (or nothing) had already committed.
            return after - static_cast<long long>(resident_committed) -
                   static_cast<long long>(probe_blocks * kv_block_bytes) +
                   static_cast<long long>(baseline_probe_pool_blocks * kv_block_bytes);
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

        // Round-9 review (Opus), finding 2: this activation-probe climb's
        // OWN floor is capped at `chunk_ceiling` too (not just the
        // doubling climb below it) -- otherwise the floor probe itself,
        // taken at up to 128 tokens, can already exceed a smaller served
        // chunk (measured: ceiling 64), and the claim "the one probe runs
        // at the served chunk" stays false for the floor probe even
        // though the doubling climb above it is correctly bounded.
        // `probe_floor_c` itself is computed once, well above (right
        // after `chunk_ceiling`), and shared with Phase B's own
        // plateau-probe fallback (round-10 review, finding 4) -- not
        // redeclared here.
        long long act128 = 0;
        long long extra128 = 0;
        double    slope_extra = 0.0;
        {
            act128 = probe(lane0, probe_floor_c);
            // The slice's layout claim, checked against what the graph returned
            // for this forward rather than against the rewrite's return value.
            if (logits_keep_rows_ > 0) {
                const ov::Tensor lg    = lane0.req.get_tensor("logits");
                const size_t     vocab = lg.get_shape().back();
                const size_t     rows  = vocab > 0 ? lg.get_size() / vocab : 0;
                // A forward of T tokens yields min(T, keep) rows: the slice
                // keeps the LAST keep rows, and the graph cannot return more
                // rows than it was given tokens. This probe runs at the
                // served chunk, so with --prefill-chunk below the slice
                // (MTP's verifier keeps 2; the 0.3.0 release gate ran the
                // suite's chunk-1 sweep on an MTP artifact and this check
                // refused the load, blaming the token axis) the honest
                // expectation is the probe's own token count, not the slice
                // size. Serve-time reads index by the tensor's actual row
                // count, never by logits_keep_rows_, so a short forward is
                // safe there. Note the check's reach: at a served chunk at or
                // below the slice size any graph returns `tokens` rows, sliced
                // correctly or not, so the token-axis claim is only verified
                // when the probe runs above the slice (the usual chunk 128);
                // a chunk-1 load is admitted, not verified.
                const size_t expect = lgc::logits_slice_rows_expected(logits_keep_rows_, probe_floor_c);
                if (rows != expect) {
                    std::ostringstream os;
                    os << lg.get_shape();
                    throw std::runtime_error(log::format(
                        "logits slice did not take: %zu row(s) for a %zu-token forward "
                        "(the slice keeps the last %zu, so %zu expected), shape %s -- the "
                        "token axis is not where the slice assumed",
                        rows, probe_floor_c, logits_keep_rows_, expect, os.str().c_str()));
                }
                log::info("load",
                          "logits slice verified: %zu row(s) for a %zu-token forward (the "
                          "slice keeps the last %zu)",
                          rows, probe_floor_c, logits_keep_rows_);
            }
            // What a SECOND lane costs, measured rather than assumed to be
            // another full peak. On the B60 with the coder it costs nothing:
            // the plugin pools intermediates per compiled model, not per
            // request. Pricing an imaginary second peak would halve the chunk
            // and with it prefill throughput.
            for (size_t i = 1; i < lanes_.size(); ++i) {
                const long long before = static_cast<long long>(device_resident_bytes(device));
                probe(*lanes_[i], probe_floor_c);
                const long long added =
                    static_cast<long long>(device_resident_bytes(device)) - before;
                extra128 += std::max<long long>(added, 0);
            }
            slope_extra = static_cast<double>(extra128) / static_cast<double>(probe_floor_c);
            if (lanes > 1) {
                log::info("load",
                          "lane activations at a %zu-token probe: lane 0 %.3f GiB, the other %d "
                          "lane(s) %.3f GiB together (the plugin pools intermediates per compiled "
                          "model, so this is measured rather than multiplied)",
                          probe_floor_c, static_cast<double>(act128) / (1u << 30), lanes - 1,
                          static_cast<double>(extra128) / (1u << 30));
            }
        }

        // Everything that is not activations, and does not move with the chunk.
        // M7 §2: drafters and the expert slot pool are their own terms now,
        // not folded into resident_base or into the activation delta.
        const long long fixed = static_cast<long long>(resident_base) +
                                static_cast<long long>(drafter_bytes) +
                                static_cast<long long>(slot_pool) +
                                static_cast<long long>(margin) +
                                static_cast<long long>(lanes) *
                                    (static_cast<long long>(slab) +
                                     static_cast<long long>(wanted) *
                                         static_cast<long long>(kv_bytes_token_));
        // A prediction that lands 11% low was measured; a quarter of headroom
        // buys the step back without giving up the climb.
        const double kHeadroom = 1.25;

        size_t    chunk      = probe_floor_c;
        long long activation = act128;
        double    slope      = static_cast<double>(std::max<long long>(act128, 1)) /
                          static_cast<double>(probe_floor_c);
        double    intercept  = 0.0;

        while (chunk * 2 <= configured && chunk * 2 <= chunk_ceiling) {
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
        // A residency delta measuring negative here is retracted as "LRU
        // eviction noise" (that explanation was narrated, not measured, and
        // is now disproven --: the actual cause was the probe's own
        // baseline double-subtracting the GDN slab and, when Phase B had
        // already committed one, the KV probe pool). With that arithmetic
        // fixed this should not fire in the ordinary case; it stays as a
        // guard because a negative value here would both under-report the
        // reservation and, cast to the Reservation's uint64_t field below,
        // wrap around into a number in the exabytes -- and if it ever does
        // fire again, the honest statement is that the baseline accounting
        // is wrong somewhere, not that anything evicted anything.
        if (activation_total < 0) {
            log::warn("load",
                      "activation delta measured negative (%.3f GiB) -- a baseline accounting "
                      "error would be the cause; clamped to 0 and logged",
                      static_cast<double>(activation_total) / (1u << 30));
            activation_total = 0;
        }

        // Per lane: the pool is shared, but every lane must be able to reach
        // n_ctx at the same time, which is what "two lanes of 30k" means.
        // M7 §2/§4 (exec/fit.h, unit-tested there against no card at all):
        // budget = total - weights - drafters - slot_pool - activations -
        // lanes*slab - margin, floored to a KV page.
        const long long budget = static_cast<long long>(total) -
                                 static_cast<long long>(resident_base) -
                                 static_cast<long long>(drafter_bytes) -
                                 static_cast<long long>(slot_pool) -
                                 static_cast<long long>(slab) * lanes - activation_total -
                                 static_cast<long long>(margin);
        FitTerms fterms;
        fterms.total          = total;
        fterms.weights        = resident_base;
        fterms.drafters       = drafter_bytes;
        fterms.slot_pool      = slot_pool;
        fterms.activations    = static_cast<uint64_t>(std::max<long long>(activation_total, 0));
        fterms.slab_per_lane  = slab;
        // M11 §1.3: the real per-token KV-pool byte size, PLUS the MTP
        // layer's own per-token state term when the MTP head is loaded --
        // `kv_bytes_token_` itself is left untouched (Phase E's own
        // allocation-overshoot math, further down, still needs the true
        // KV-pool rate, not this fit-only inflation).
        fterms.kv_bytes_token = kv_bytes_token_ + mtp_state_bytes_token;
        fterms.margin         = margin;
        fterms.lanes           = lanes;
        fterms.kv_block_tokens = static_cast<int>(kv_block_tokens_);
        fterms.n_ctx_floor     = static_cast<int>(
            std::max<size_t>(kv_block_tokens_ * static_cast<size_t>(lanes), 4096));

        // M9/0015 fit-side charge for the packed-4-bit-values prefill
        // scratch buffer: folded HERE, before fit_context ever runs,
        // because it changes max_ctx itself. Round-8 review (Opus):
        // `cap_off`, `packed_values`, `packed_geom` and the
        // `paged_attention_*` detection facts are now resolved EARLY
        // (before the activation probe climb, above -- see that block's
        // own comment), because the served chunk itself must be known
        // before the one real activation probe runs; reused here rather
        // than re-read a second time.
        FitResult fit;
        int       packed_values_chunk = static_cast<int>(chunk);
        // Round-9 review (Opus), finding 4: the informational per-token
        // rate for the summary/detail log lines below -- computed
        // separately from the FIT arithmetic. `fit_context_packed_values_
        // at_depth` never sets a per-token slope (there is no linear
        // approximation left when the term is priced exactly at a known
        // depth), so the log used to read a member that was always 0 on
        // the explicit path ("per token 0.0 KiB") -- the RATE the term
        // grows at, AT the served chunk, is what the log means to show,
        // and that is well defined regardless of which primitive priced
        // the term.
        uint64_t packed_values_log_per_token_bytes = 0;
        if (packed_values && packed_geom) {
            // Round-8 review (Opus): no re-probing here any more --
            // round-7's own fix could not work (this file's own record is
            // that the plugin's intermediate pool grows to the largest
            // shape it has ever seen and never shrinks, so a probe taken
            // after a bigger one is a stale no-op; see fit.h's
            // fit_context_packed_values_at_depth for the full account).
            // `chunk` (the activation climb's own served chunk, above) was
            // already capped at `chunk_ceiling` BEFORE that climb ran, so
            // the real, final `activation_total` was measured at the
            // right place from the start -- what remains here is the
            // final admissibility arithmetic, using the primitive both
            // paths share.
            //
            // Round-10 review (Opus, REAL defect -- retracts round-9's own
            // "both paths evaluate at wanted" claim): `wanted` is NOT the
            // served depth for auto-fit -- it is the artifact's train
            // maximum, an UPPER BOUND to search from, and evaluating the
            // belt there (`chunk_ceiling`, above) is the SMALLEST,
            // most-conservative chunk possible -- correct as the activation
            // climb's own safety ceiling, wrong as the packed-values term's
            // FINAL chunk. Measured: with `wanted` = 262,144 the belt gives
            // chunk 32; the depth auto-fit actually adopts (103,040) only
            // needs chunk 64 -- an explicit request for that SAME 103,040
            // evaluated the honest way (this file's own `at_depth`, below)
            // priced chunk 64, and auto-fit -- pricing 32 the whole time --
            // served a different, more expensive reservation for the
            // identical depth.
            //
            // Required design (Opus, round-10): (1) `chunk_ceiling` above
            // is step 1 -- the conservative starting probe bound. (2) run
            // the packed-values SEARCH (`fit_context_packed_values`, its
            // own chunk-driven fixed point, unchanged) seeded from
            // `configured` (the operator's own default/requested chunk,
            // NOT the ceiling-capped activation climb's `chunk`) to find
            // the adopted depth D* and ITS OWN served chunk c*. (3) c* is
            // >= `chunk_ceiling` by construction (D* <= wanted, belt is
            // non-increasing in depth). (4) if c* is bigger than the chunk
            // activations were actually probed at, RE-PROBE upward at c*
            // (valid -- the plugin's pool only ever grows, so a probe at a
            // BIGGER chunk than before is a real, honest measurement, not
            // the round-7 defect, which probed downward into a stale
            // reading) and re-run the search with the corrected figure;
            // repeat. `term.chunk` only grows and is bounded above by
            // `configured`, off the belt's own finite halving ladder, so
            // this terminates in at most as many rounds as the ladder has
            // rungs (bounded defensively at 8 anyway).
            PackedValuesFitTerm term;
            if (cfg.n_ctx_explicit) {
                // (5): `wanted` IS the requested depth here, so
                // `chunk_ceiling` (belt at wanted = D) is already the
                // right answer -- no search, single evaluation.
                //
                // Round-14 review (Opus), finding 1 (REAL defect, third
                // instance of this class -- round-12's ceiling and
                // round-11's `prefill_chunk_` seed were the first two):
                // `fit_context_packed_values_at_depth` had no
                // `belt_enabled` of its own, so `ARCINT_PREFILL_CHUNK_
                // CAP=off` was silently ignored on this (explicit
                // --n-ctx) path -- the switch's own warn log claimed the
                // belt was disabled while the belt (and the measured cap
                // on top of it) still ran. Threaded through now: under
                // `cap_off`, `configured` (the operator's own requested
                // chunk, NOT `chunk` -- the activation climb's own
                // served chunk, which can stop short of `configured` for
                // reasons that have nothing to do with the belt, e.g. an
                // activation-budget prediction or a failed probe, so it
                // is not a safe stand-in for "the full requested chunk,
                // uncapped") is priced with `belt_enabled=false`,
                // matching the auto-fit search's own `!cap_off` gate
                // exactly.
                term = fit_context_packed_values_at_depth(
                    fterms, cap_off ? static_cast<int>(configured) : static_cast<int>(chunk),
                    packed_geom->heads, packed_geom->head_size,
                    kPrefillScratchBudgetBytesPackedValues, cfg.kv_block_size, wanted,
                    paged_attention_max_partitions_effective, paged_attention_element_bytes,
                    /*belt_enabled=*/!cap_off, packed_values_mixed_stage_on_micro_);
            } else {
                int  last_probed_chunk = static_cast<int>(chunk);
                bool converged         = false;
                for (int round = 0; round < 8; ++round) {
                    term = fit_context_packed_values(
                        fterms, static_cast<int>(configured), packed_geom->heads,
                        packed_geom->head_size, kPrefillScratchBudgetBytesPackedValues,
                        cfg.kv_block_size, !cap_off, paged_attention_max_partitions_effective,
                        paged_attention_element_bytes, packed_values_mixed_stage_on_micro_);
                    if (term.chunk <= last_probed_chunk) {
                        // c* did not grow past what is already measured --
                        // `term` is CONSISTENT with the activations it was
                        // just priced against (this round read `fterms.
                        // activations` and never re-probed), done.
                        converged = true;
                        break;
                    }
                    const size_t served_chunk = static_cast<size_t>(term.chunk);
                    long long    reprobed      = probe(lane0, served_chunk);
                    for (size_t i = 1; i < lanes_.size(); ++i) {
                        const long long before =
                            static_cast<long long>(device_resident_bytes(device));
                        probe(*lanes_[i], served_chunk);
                        const long long added =
                            static_cast<long long>(device_resident_bytes(device)) - before;
                        reprobed += std::max<long long>(added, 0);
                    }
                    if (reprobed < 0) reprobed = 0;
                    log::info("load",
                              "4-bit values: search wants chunk %zu (was priced at %d) -- "
                              "re-probed activations upward: %.2f GiB, was %.2f GiB",
                              served_chunk, last_probed_chunk,
                              static_cast<double>(reprobed) / (1u << 30),
                              static_cast<double>(activation_total) / (1u << 30));
                    activation_total   = reprobed;
                    fterms.activations = static_cast<uint64_t>(reprobed);
                    last_probed_chunk  = term.chunk;
                }
                if (!converged) {
                    // Round-11 review (Opus), finding 5: on exhaustion (8
                    // rounds without the chunk settling), the loop above
                    // exits right after re-probing at THIS round's own
                    // `term.chunk` and updating `fterms.activations` to
                    // match -- but `term` itself is still the STALE result
                    // computed one round earlier, from the activation
                    // figure BEFORE that last re-probe. Adopting it would
                    // serve a (chunk, depth, activation) triple that
                    // disagrees with the activation figure this file is
                    // about to charge the reservation with. One more
                    // evaluation, against the now-current (fully
                    // re-probed) `fterms`, makes `term` consistent with
                    // what was actually measured, at the cost of one more
                    // CPU-only fixed-point pass (no further GPU probe).
                    //
                    // Round-12 review (Opus), finding 3 (REAL defect,
                    // under-reservation risk): that one more evaluation
                    // used to re-seed from `configured` -- the operator's
                    // own full requested chunk -- not from `last_probed_
                    // chunk`, so the search was free to climb PAST the
                    // chunk activations were actually last measured at.
                    // Serving a chunk bigger than any chunk this load
                    // ever probed is exactly the risk the whole upward-
                    // re-probe design exists to avoid (a bigger chunk can
                    // legitimately need more resident activation memory
                    // than a smaller one measured). Re-seeded from
                    // `last_probed_chunk` instead: the belt never raises
                    // its output past its own input, so this evaluation's
                    // `term.chunk` is now bounded above by `last_probed_
                    // chunk` BY CONSTRUCTION, not by a post-hoc clamp on
                    // an already-priced (and possibly larger, and now
                    // wrong) term.
                    log::warn("load",
                              "4-bit values: search did not settle within 8 rounds -- adopting "
                              "the last activation-consistent chunk, clamped to the last "
                              "actually-probed chunk (%d), rather than one more round of "
                              "re-probing",
                              last_probed_chunk);
                    term = fit_context_packed_values(
                        fterms, last_probed_chunk, packed_geom->heads,
                        packed_geom->head_size, kPrefillScratchBudgetBytesPackedValues,
                        cfg.kv_block_size, !cap_off, paged_attention_max_partitions_effective,
                        paged_attention_element_bytes, packed_values_mixed_stage_on_micro_);
                }
            }
            fit                                    = term.fit;
            packed_values_chunk                    = term.chunk;
            packed_values_scratch_per_token_bytes_ = term.per_token_bytes;
            packed_values_scratch_fixed_bytes_     = term.fixed_bytes;
            packed_values_scratch_chunk_           = term.chunk;
            packed_values_log_per_token_bytes =
                packed_values_mixed_stage_on_micro_
                    ? 0
                    : packed_values_prefill_scratch_bytes_per_token_ex(
                          term.chunk, packed_geom->heads, packed_geom->head_size,
                          paged_attention_element_bytes) +
                          packed_values_partials_exp_max_bytes_per_token(term.chunk,
                                                                         packed_geom->heads);
            // Round-12 review (Opus), finding 2 (REAL defect): the
            // reduction log used to sit at the Phase E belt call site
            // (further down), naming which of three limits ("budget",
            // "bound", "measured cap") shrank the chunk -- but by the
            // time that site runs, `prefill_chunk_` (round-11 review) IS
            // already the fixed point `term.chunk` computed right here,
            // and `belt_requested_chunk` caps the belt's own input at
            // that SAME value there, so that log was UNREACHABLE: the
            // belt never raises its output past its input, and every
            // path to the FINAL served depth only ever LOWERS it from
            // the depth this term was priced at (the belt is
            // non-increasing in depth), so a shallower re-evaluation can
            // only ever re-derive the same chunk, never a smaller one.
            // Moved here, to where the chunk is actually decided --
            // `configured` (the operator's own requested/default chunk)
            // versus `term.chunk` (what the search or `at_depth` just
            // settled on), classified with the SAME three names, at the
            // depth this term was actually priced at (the search's own
            // converged `term.fit.max_ctx` for auto-fit, `wanted` for an
            // explicit request). Skipped entirely under `cap_off` (the
            // belt is off by design there -- see the ceiling and Phase E
            // call sites' own comments) and whenever nothing shrank at
            // all.
            if (!cap_off && term.chunk < static_cast<int>(configured)) {
                const long long depth_for_reason =
                    cfg.n_ctx_explicit ? static_cast<long long>(wanted) : term.fit.max_ctx;
                const char* limit = "budget";
                if (packed_values_mixed_stage_on_micro_) {
                    // Patch 0020 arm (§7.0.2at review): the budget ladder
                    // never ran, so classifying against it would name
                    // "budget" for a chunk that was the measured cap and
                    // nothing else (an explicit n_ctx between 64k and 128k
                    // on `+p6` makes the ladder's own answer coincide with
                    // 128).
                    limit = "measured cap";
                } else {
                    const long long partitions_for_reason = packed_values_bounded_partitions(
                        depth_for_reason, paged_attention_max_partitions_effective);
                    const int budget_only = prefill_chunk_cap_for_packed_values_budget_only_ex(
                        static_cast<int>(configured), partitions_for_reason, packed_geom->heads,
                        packed_geom->head_size, paged_attention_element_bytes,
                        kPrefillScratchBudgetBytesPackedValues, cfg.kv_block_size);
                    if (term.chunk != budget_only) {
                        limit = "measured cap";
                    } else if (budget_only == cfg.kv_block_size &&
                               !packed_values_scratch_fits_budget(
                                   budget_only, partitions_for_reason, packed_geom->heads,
                                   packed_geom->head_size, paged_attention_element_bytes,
                                   kPrefillScratchBudgetBytesPackedValues)) {
                        limit = "bound";
                    }
                }
                log::info("load",
                          "prefill chunk %zu -> %d for 4-bit values (query heads %d, head size "
                          "%d, n_ctx %lld): empirical belt, measured -- not a derived scratch "
                          "formula (see exec/fit.h) -- limit: %s",
                          configured, term.chunk, packed_geom->heads, packed_geom->head_size,
                          depth_for_reason, limit);
            }
        } else {
            fit = fit_context(fterms);
        }
        long long max_ctx = fit.max_ctx;
        // Round-9 review (Opus), finding 3 (measured on card, a third
        // time): the FIT-LEVEL admissibility check below (`wanted vs
        // max_ctx`) is necessary but not sufficient for the packed-values
        // case -- Phase E's own allocation-time replay can still overshoot
        // by a small, real amount past what this analytic division
        // predicts (measured: a 10 MiB, 0.07% shortfall at the ceiling for
        // an explicit --n-ctx equal to what auto-fit itself adopted after
        // ITS OWN Phase E trim gave it exactly this margin for free).
        // Explicit --n-ctx has no such margin unless one is built in
        // here: one KV page (`kv_block_tokens_`) of slack, subtracted from
        // `max_ctx` before the admissibility check below, so BOTH paths
        // (explicit's refusal test, and auto-fit's own `min(wanted,
        // max_ctx)` adoption) use the identical, already-slack-adjusted
        // ceiling -- the auto-fit depth this produces is then, by
        // construction, admissible again if resubmitted as an explicit
        // request for the same depth (see the round-9 test making exactly
        // that round trip).
        if (packed_values && packed_geom) {
            max_ctx -= static_cast<long long>(kv_block_tokens_);
            if (max_ctx < 0) max_ctx = 0;
            // Round-10 review (Opus), finding 5: `fit.admissible` was
            // computed by `fit_context`/`fit_context_packed_values[_at_
            // depth]` from the PRE-slack `fit.max_ctx` -- the subtraction
            // two lines up only ever touched the local `max_ctx` copy, so
            // a fit landing EXACTLY at `n_ctx_floor` (admissible = true,
            // by the closed-form rule `max_ctx >= n_ctx_floor`) could
            // still adopt a slack-adjusted depth BELOW the floor a few
            // lines later (`paged_n_ctx_ = min(wanted, max_ctx)`, further
            // down) while `auto_fit_admissible` (which reads `fit.
            // admissible` directly) kept reporting the load as usable.
            // Recomputed here, against the SAME slack-adjusted `max_ctx`
            // every later check in this function reads (the reserve gate,
            // the "no usable context" refusal, and `paged_n_ctx_` itself),
            // so `fit.admissible` never again disagrees with the floor
            // rule applied to the depth this file actually adopts. `fit.
            // max_ctx` itself is left untouched -- nothing downstream
            // reads it again (only the local, slack-adjusted `max_ctx`
            // is), so there is nothing there to keep in sync.
            fit.admissible = max_ctx > 0 && max_ctx >= static_cast<long long>(fterms.n_ctx_floor);
        }
        // Round-2 review, F5: the packed-values term (when charged) is
        // folded into THIS summary line too, not only the detail line
        // below -- a one-line reservation is what an operator scanning the
        // log actually reads, and the total this line prints must be the
        // one the fit actually solved (weights+drafters+slots+activations+
        // margin+slab+KV+scratch), not everything except the newest term.
        // `term_total_bytes` and `packed_values_clause` are computed once,
        // here, and reused by the detail line further down so the two
        // never disagree about the number.
        //
        // Round-3 review, finding 4: the fit charges the per-token part of
        // this term PER LANE -- fit_context_packed_values folds it into
        // `kv_bytes_token`, which fit_context itself multiplies by both
        // `lanes` and `max_ctx` -- but this line's own earlier version
        // multiplied by `max_ctx` alone, so the log under-reported the
        // term by a factor of `lanes` on more than one lane and the
        // line's own terms did not sum to the printed total any more.
        // `packed_values_scratch_reservation_bytes` (fit.h) is the shared
        // formula fit_context_packed_values, Phase E's ceiling check
        // (below), and this line all now read from -- one implementation
        // of "lanes * n_ctx * per_token + fixed", not three.
        const uint64_t term_total_bytes =
            (packed_values && packed_geom)
                ? packed_values_scratch_reservation_bytes(packed_values_scratch_fixed_bytes_,
                                                          packed_values_scratch_per_token_bytes_,
                                                          lanes, max_ctx)
                : 0;
        const std::string packed_values_clause =
            (packed_values && packed_geom)
                ? log::format(" + prefill scratch %.2f GiB (4-bit values, chunk %d)",
                              static_cast<double>(term_total_bytes) / (1u << 30),
                              packed_values_chunk)
                : std::string();
        // M11 §1.3: reported next to "drafters" -- the same term folded into
        // fterms.kv_bytes_token above, evaluated at the final `lanes *
        // max_ctx` this line is about to print, via fit.h's own
        // mtp_state_bytes (one formula, not a second copy of the
        // arithmetic).
        const uint64_t mtp_state_total_bytes =
            mtp_state_bytes_token > 0
                ? static_cast<uint64_t>(lanes) * lgc::mtp_state_bytes(max_ctx)
                : 0;
        const std::string mtp_state_clause =
            mtp_state_bytes_token > 0
                ? log::format(" + MTP state %.2f GiB (%.1f KiB/token)",
                              static_cast<double>(mtp_state_total_bytes) / (1u << 30),
                              static_cast<double>(mtp_state_bytes_token) / 1024.0)
                : std::string();
        log::info("load",
                  "reservation: weights+graph %.2f GiB + drafters %.2f%s + expert slots %.2f (%s) "
                  "+ activations %.2f (all %d lane%s, chunk %zu)%s + margin %.2f + %d x (GDN "
                  "rows %.1f MiB + KV %.1f KiB/token) of %.2f GiB -> max ctx %lld per lane",
                  static_cast<double>(resident_base) / (1u << 30),
                  static_cast<double>(drafter_bytes) / (1u << 30),
                  mtp_state_clause.c_str(),
                  static_cast<double>(slot_pool) / (1u << 30),
                  slot_source.empty() ? "off" : slot_source.c_str(),
                  static_cast<double>(activation_total) / (1u << 30), lanes,
                  lanes == 1 ? "" : "s", chunk, packed_values_clause.c_str(),
                  static_cast<double>(margin) / (1u << 30), lanes,
                  static_cast<double>(slab) / (1u << 20),
                  static_cast<double>(kv_bytes_token_) / 1024.0,
                  static_cast<double>(total) / (1u << 30), max_ctx);
        if (packed_values && packed_geom) {
            // Round-9 review (Opus), finding 4: `packed_values_log_per_
            // token_bytes` (computed above, right after the term itself
            // was priced) -- NOT `packed_values_scratch_per_token_bytes_`,
            // which `fit_context_packed_values_at_depth` always leaves 0
            // (an exact-at-one-depth evaluation has no per-token slope to
            // report) and which printed a misleading "per token 0.0 KiB"
            // on every explicit-n_ctx 4-bit-values load.
            //
            // Earlier review (bounded-path budget/log message): this line
            // used to print the raw, unclamped `max_ctx` -- the BUDGET
            // ceiling, not what the load actually serves -- as "for n_ctx"
            // (a 24 GB-card trace showed "n_ctx 929456" on a load that
            // went on to adopt a train maximum orders of magnitude
            // smaller). `paged_n_ctx_` itself is not computed until after
            // Phase E (min(wanted, max_ctx), further down, and possibly
            // trimmed again after that) so it is not yet known here; a
            // follow-up fix printed `std::min(wanted, max_ctx)` instead,
            // one step early, before any Phase E trim.
            //
            // RETRACTED (a later, card-measured defect this repository's
            // own record keeps rather than silently editing away, per
            // DESIGN §7.0.1): `min(wanted, max_ctx)` reads fine when the
            // request is admitted, but collapses to a misleading "n_ctx
            // 0" in the REFUSE case -- measured on the card,
            // `ARCINT_PREFILL_CHUNK_CAP=off` with `--n-ctx 101824` and no
            // `--prefill-chunk` priced the term at the full, unbelted
            // chunk 2048 (19,203.5 MiB), leaving no budget at all
            // (`max_ctx` 0 after the slack subtraction above), and the
            // load correctly went on to refuse -- but the log printed
            // "for n_ctx 0", which reads as "this load is about to serve
            // an empty context," not "this load is about to refuse a
            // 101,824-token request." `min()` conflates two different
            // numbers (what was asked for, and what the budget admits)
            // into one that is only informative when they happen to
            // agree. Printed separately now: the REQUESTED depth
            // (`wanted` -- this file's own name for it everywhere else,
            // including the refusal message itself) and what the
            // reservation ADMITS (`max_ctx`, already the slack-adjusted
            // figure the admissibility check just below uses) -- "for
            // requested n_ctx 101824 (admits 0)" says plainly that the
            // request is being refused, not served at depth 0.
            //
            // Round-10 review (Opus), finding 7: `packed_values_log_per_
            // token_bytes` is INFORMATIONAL only -- the marginal rate the
            // term would grow at, one token further, at the served chunk.
            // It does NOT reconstruct the printed MiB figure by simple
            // multiplication against either depth printed here: the
            // charged MiB is priced at the BUDGET ceiling `max_ctx` (this
            // line's own `term_total_bytes`, shared with the summary line
            // and Phase E's ceiling check above), while the rate is
            // quoted at whichever chunk was actually served, and the
            // bounded arm (`max_partitions > 0`) charges a flat amount
            // with no per-token slope at all (the rate there describes
            // what an UNBOUNDED reservation would have cost per token,
            // for comparison, not a component of the bounded charge
            // itself). Labelled "rate" rather than "per token" so the
            // line does not read as a component that should sum to the
            // total above it.
            log::info("load",
                      "4-bit values: prefill scratch charged %.1f MiB at chunk %d for requested "
                      "n_ctx %d (admits %lld) (informational rate %.1f KiB/token + KV %.1f "
                      "KiB/token)",
                      static_cast<double>(term_total_bytes) / (1u << 20), packed_values_chunk,
                      wanted, max_ctx,
                      static_cast<double>(packed_values_log_per_token_bytes) / 1024.0,
                      static_cast<double>(kv_bytes_token_) / 1024.0);
            // 0015 engine side: the plugin-support disclosure, separate
            // from the arithmetic line above -- an operator reading the log
            // should be able to tell WHY the scratch term above is small
            // (or not) without cross-referencing whether the plugin is
            // patched at all. Round-6 review, finding 6: N = 0 means the
            // plugin carries the bound key but the operator asked for no
            // bound at all -- "bounds attention partials at 0" reads as a
            // refusal (zero partitions), not what actually happened, so
            // that case gets its own wording. "(f16 partials)" follows the
            // detection contract stated at the probe site (a plugin that
            // carries the bound key carries the f16 host sizing too, both
            // ship in patch 0015) -- unconditional whenever the key was
            // accepted, not a separate confirmation.
            if (packed_values_mixed_stage_on_micro_) {
                // Patch 0020 arm: the term above printed 0.0 MiB; say why,
                // ahead of the 0015 bound disclosure (which +p6 also
                // carries, and which bounds buffers this pairing no longer
                // allocates).
                log::info("load",
                          "4-bit values: scratch term not charged -- plugin patch level %d runs "
                          "the mixed prefill stage on micro-SDPA (patch 0020); chunk %d is the "
                          "measured cap, not a budget choice%s",
                          gpu_plugin_patch_level_, packed_values_chunk,
                          (paged_attention_bound_accepted_ && cfg.paged_attention_max_partitions > 0)
                              ? "; the partition bound is set on the plugin and not priced (the "
                                "buffers it bounds are not allocated on this path)"
                              : "");
            } else if (paged_attention_bound_accepted_) {
                if (cfg.paged_attention_max_partitions > 0) {
                    log::info("load",
                              "4-bit values: plugin bounds attention partials at %d (f16 "
                              "partials); scratch term %.1f MiB at chunk %d",
                              cfg.paged_attention_max_partitions,
                              static_cast<double>(term_total_bytes) / (1u << 20),
                              packed_values_chunk);
                } else {
                    log::info("load",
                              "4-bit values: plugin carries the bound key; N = 0 leaves "
                              "partials unbounded (f16 partials); scratch term %.1f MiB at "
                              "chunk %d",
                              static_cast<double>(term_total_bytes) / (1u << 20),
                              packed_values_chunk);
                }
            }
        }
        // Round-11 review (Opus), finding 1 (REAL defect, HIGH): this used
        // to seed unconditionally from `chunk` -- the activation climb's
        // own served chunk, bounded above by `chunk_ceiling` (belt at
        // `wanted`, the conservative starting probe bound) -- never from
        // the packed-values SEARCH's own converged c* (`packed_values_
        // scratch_chunk_`, set above, the chunk the reservation was
        // actually priced at). Since `chunk <= chunk_ceiling <= c*`
        // always (the F1 design's own step 3), Phase E's belt call site
        // below (`belt_requested_chunk(prefill_chunk_, packed_values_
        // scratch_chunk_)`) then took `min(prefill_chunk_, packed_values_
        // scratch_chunk_)` == `prefill_chunk_` == the SMALLER
        // ceiling-bound chunk, unconditionally -- so whatever bigger
        // chunk the search (and the upward re-probe loop that pays for
        // it in real GPU round-trips) legitimately found was priced into
        // the reservation and then silently discarded at serve time:
        // worse than not looping at all, since the extra probes bought
        // nothing served. Seeded from `packed_values_scratch_chunk_`
        // instead whenever a 4-bit-values load actually priced one --
        // defensively re-clamped to `configured` and `kMaxMeasured
        // PackedValuesChunk` (the search's own belt already enforces
        // both; this is a second, explicit guard at the seam between the
        // climb and Phase E, not a new bound) -- so the belt call site's
        // own `min()` becomes the no-op it was designed to be, and only
        // actually shrinks the served chunk when a LATER depth trim
        // genuinely requires it.
        //
        // Round-12 review (Opus), finding 1 (HIGH, REAL defect): this
        // defensive re-clamp used to apply UNCONDITIONALLY, including
        // `kMaxMeasuredPackedValuesChunk` -- so `ARCINT_PREFILL_CHUNK_
        // CAP=off` (which correctly makes the search price `configured`
        // uncapped, via `belt_enabled=false`, so `packed_values_scratch_
        // chunk_` already legitimately EQUALS `configured` when the
        // switch is set) got silently re-capped to 128 right here anyway,
        // a second no-op bug stacked on the ceiling one above. Under
        // `cap_off`, seed from `packed_values_scratch_chunk_` directly --
        // it already IS what the term was priced at, uncapped by
        // construction -- and skip the defensive clamp entirely.
        prefill_chunk_ =
            (packed_values && packed_geom)
                ? (cap_off ? static_cast<int>(packed_values_scratch_chunk_)
                           : static_cast<int>(std::min<size_t>(
                                 {static_cast<size_t>(packed_values_scratch_chunk_), configured,
                                  static_cast<size_t>(kMaxMeasuredPackedValuesChunk)})))
                // Patch 0020 arm without geometry (§7.0.2at review): the
                // term was never priced (no heads/head_dim to price it
                // with), but the measured cap needs no geometry and holds
                // on this path as on every other 4-bit-values load.
                : (packed_values && packed_values_mixed_stage_on_micro_ && !cap_off)
                      ? packed_values_measured_chunk_cap(static_cast<int>(chunk), cfg.kv_block_size)
                      : static_cast<int>(chunk);
        // The M9 4-bit-values prefill-chunk belt (exec/fit.h's
        // prefill_chunk_cap_for_packed_values) and the snapshot grid it
        // feeds (cache_grid_, tied to the chunk) both need the SERVED pool
        // depth, not this raw auto-fit `chunk`/`max_ctx` pair -- --n-ctx
        // clamping, --prefix-cache-reserve and Phase E's replay/trim all
        // still run between here and that final depth. Both are computed
        // once, together, after Phase E below (round-2 review, finding 4).

        // M7 §1 "Explicit --n-ctx": explicit always wins in that it is never
        // silently lowered, but the fit still runs verify-only -- an explicit
        // n_ctx above what the reservation admits refuses at load, here and
        // again below if the allocation-time audit disagrees with the fit.
        //
        // Defect fix: `req_n_ctx` is what main.cpp resolved BEFORE calling
        // this constructor (0 -> artifact.n_ctx_train), so it is always
        // positive here whether or not the operator typed --n-ctx -- using
        // `req_n_ctx > 0` as the "explicit" signal made every omitted
        // --n-ctx look operator-requested, which turned the honest (smaller)
        // M7 max ctx into a load refusal instead of the adopted default.
        // `cfg.n_ctx_explicit` is set only when --n-ctx was actually parsed
        // (config.cpp), so it is the real signal.
        const bool explicit_n_ctx = cfg.n_ctx_explicit;
        auto itemized_terms = [&](int n_ctx_for_message) {
            return log::format(
                "weights %.2f + drafters %.2f + expert slots %.2f (%s) + activations %.2f + "
                "margin %.2f + %d x state %.3f of %.2f GiB (n_ctx %d)",
                static_cast<double>(resident_base) / (1u << 30),
                static_cast<double>(drafter_bytes) / (1u << 30),
                static_cast<double>(slot_pool) / (1u << 30),
                slot_source.empty() ? "off" : slot_source.c_str(),
                static_cast<double>(activation_total) / (1u << 30),
                static_cast<double>(margin) / (1u << 30), lanes,
                static_cast<double>(slab) / (1u << 30), static_cast<double>(total) / (1u << 30),
                n_ctx_for_message);
        };

        // M9 "--prefix-cache-reserve PCT": an option under auto-fit that
        // holds back PCT percent of the pool's own budget-affordable pages
        // as spare for cached prefixes, instead of auto-fit adopting the
        // whole budget as live pages (PCT 0, today's behaviour) and leaving
        // the prefix cache the "0 spare pages" case the warning near the
        // end of this function names. Applied to `max_ctx` HERE, before the
        // "wanted > max_ctx" check and the `paged_n_ctx_ = min(wanted,
        // max_ctx)` adoption below -- so the adopted depth still floors to
        // a page multiple (prefix_cache_reserve floors live_pages down,
        // fit.h), still respects the train maximum (the min() below still
        // runs against the RESERVED max_ctx, so an artifact whose own
        // default is smaller than the reserved budget still wins), and
        // still respects the 4096 floor (`reserve.admissible`, mirroring
        // fit.admissible's own rule -- a PCT that would push the adopted
        // depth below the floor refuses below rather than silently landing
        // on the floor itself). config.cpp already refuses this flag
        // together with an explicit --n-ctx (an explicit depth already
        // defines the reserve as whatever remains after the request), so
        // `explicit_n_ctx` is always false whenever `reserve_applied` is
        // true -- asserted explicitly at the top of Phase E below (L6,
        // round-2 review), since this comment is not itself a guarantee a
        // future edit here has to respect.
        //
        // L5 (round-2 review): gated on `fit.admissible` too -- when the
        // PLAIN (unreserved) fit already found nothing usable, applying a
        // reserve on top and then blaming the reserve in the refusal
        // message below would misattribute a failure the reserve did not
        // cause. `!fit.admissible` falls through unreserved to the
        // pre-existing "auto-fit found no usable context" throw further
        // down, with its original (non-reserve) wording.
        //
        // What keeps the reserve through the replay loop below (Phase E):
        // round-4 correction -- an earlier version of this comment said
        // the auto-fit overshoot-correction branch "calls shrink_n_ctx on
        // paged_n_ctx_ directly and never touches spare_blocks or
        // wanted_spare", which was true of the ORIGINAL (defective)
        // correction, not of `auto_fit_trim` (fit.h), which replaced it in
        // round 3 precisely because a live-only correction left
        // pool_sizing's own spare_room growing back to absorb the cut and
        // the pool total never moved (see auto_fit_trim's own comment for
        // the measured cells). The correction now DOES read and write
        // spare, via `spare_cap` -- what actually protects the reserve is
        // the `protect_spare = reserve_applied` argument passed to
        // auto_fit_trim below: with it set, live is cut FIRST and the
        // spare cap is pinned (never enlarged, never reduced) for as long
        // as live still has room before the floor; spare only starts
        // giving once live's remaining room is smaller than the pass's
        // own trim budget (at most 256 pages, the last pass's backoff
        // floor, unless a genuinely large measured `over` asks for more)
        // -- reported, if it happens, by the post-hoc `prefix_cache_
        // reserve_shortfall` warning near the end of this function. One
        // consequence worth stating: because live shrinks while a
        // protected spare cap stays flat, the REALIZED spare fraction
        // drifts UP across correction passes, never down, until live
        // bottoms out at the floor. The explicit-n_ctx retry
        // (`explicit_retry_decision`) remains structurally unreachable in
        // this configuration, since it only runs when `explicit_n_ctx` is
        // true and this branch requires `!explicit_n_ctx`.
        //
        // L4 (round-2 review): the "prefix-cache reserve: ..." log line
        // used to print here, before `paged_n_ctx_ = min(wanted, max_ctx)`
        // ever ran -- so it could name an "adopted n_ctx" the artifact's
        // own train maximum then immediately overrode (when `wanted <
        // max_ctx`). Only the reserve arithmetic happens here now; the log
        // line moved to just after that min(), where `paged_n_ctx_` is the
        // number actually adopted.
        bool               reserve_applied = false;
        PrefixCacheReserve reserve;
        if (!explicit_n_ctx && cfg.prefix_cache_reserve_pct > 0 && max_ctx > 0 &&
            fit.admissible) {
            const long long affordable_pages = max_ctx / static_cast<long long>(kv_block_tokens_);
            reserve = prefix_cache_reserve(affordable_pages, cfg.prefix_cache_reserve_pct,
                                           static_cast<int>(kv_block_tokens_), fterms.n_ctx_floor);
            reserve_applied = true;
            max_ctx          = reserve.adopted_n_ctx;
        }

        // (review finding): adopt mode (no explicit --n-ctx) must not
        // silently clamp to an inadmissible max_ctx. fit.admissible was
        // computed and never read -- with status_.n_ctx left at 0 the
        // handlers.cpp admission guard (`n_ctx > 0 && ...`) is disabled, and
        // requests would die at runtime instead of the load refusing
        // cleanly. Checked unconditionally here (not nested under "wanted >
        // max_ctx"), because the failure is "there is nothing to adopt",
        // independent of what the artifact's own default happened to be.
        // M9: when the reserve was applied, `reserve.admissible` replaces
        // `fit.admissible` -- the question is now whether the RESERVED
        // depth is usable, not the unreserved one.
        const bool auto_fit_admissible = reserve_applied ? reserve.admissible : fit.admissible;
        if (!explicit_n_ctx && !auto_fit_admissible) {
            if (reserve_applied) {
                throw std::runtime_error(log::format(
                    "auto-fit found no usable context on %d lane%s after holding back a %d%% "
                    "prefix-cache reserve (%s); --n-ctx was not given, so there is no request "
                    "to lower further. Lower --prefix-cache-reserve, lower --parallel, lower "
                    "--prefill-chunk, or free memory on the card.",
                    lanes, lanes == 1 ? "" : "s", cfg.prefix_cache_reserve_pct,
                    itemized_terms(static_cast<int>(max_ctx)).c_str()));
            }
            throw std::runtime_error(log::format(
                "auto-fit found no usable context on %d lane%s (%s); --n-ctx was not given, "
                "so there is no request to lower further -- this is the largest number the "
                "fit could compute and it is still not enough. Lower --parallel, lower "
                "--prefill-chunk, or free memory on the card.",
                lanes, lanes == 1 ? "" : "s",
                itemized_terms(static_cast<int>(max_ctx)).c_str()));
        }
        if (static_cast<long long>(wanted) > max_ctx) {
            if (explicit_n_ctx) {
                throw std::runtime_error(log::format(
                    "requested n_ctx %d on %d lane%s needs %.2f GiB of KV but the reservation "
                    "admits %lld per lane (%s). Lower --n-ctx, lower --parallel, or lower "
                    "--prefill-chunk.",
                    wanted, lanes, lanes == 1 ? "" : "s",
                    static_cast<double>(wanted) * kv_bytes_token_ * lanes / (1u << 30), max_ctx,
                    itemized_terms(wanted).c_str()));
            }
            log::info("load", "n_ctx clamped to the admissible %lld (train maximum %d)", max_ctx,
                      wanted);
        }
        paged_n_ctx_ = static_cast<int>(std::min<long long>(wanted, max_ctx));

        // L4 (round-2 review): logged AFTER adoption, from `paged_n_ctx_`
        // itself, not from `max_ctx` -- the reserve arithmetic above ran
        // against the BUDGET-derived max_ctx, but `wanted` (the artifact's
        // own train maximum, or an operator's smaller default) can still
        // win the min() just above, and the "adopted n_ctx" this line
        // reports has to be the number that actually won.
        if (reserve_applied) {
            // Round-3 review, defect 2 (units): explicit about "per lane"
            // here -- `reserve.spare_pages` is per-lane, pure-KV pages
            // (see reserve_ask_pool_pages in fit.h), not the pool-wide page
            // count Phase E's own "paged pool: ... + N spare" line prints
            // further down. Naming both would need the allocation-time
            // pool to already exist, which it does not yet at this point
            // in the load -- the pool-wide "N pool page(s)" wording lives
            // on the shortfall warning below instead, once Phase E has run.
            log::info("load",
                      "prefix-cache reserve: %d%% = %lld page%s per lane (%lld tokens per "
                      "lane) held spare; adopted n_ctx %d",
                      cfg.prefix_cache_reserve_pct, reserve.spare_pages,
                      reserve.spare_pages == 1 ? "" : "s",
                      reserve.spare_pages * static_cast<long long>(kv_block_tokens_),
                      paged_n_ctx_);
        }

        // ---- Phase E: allocate, audit, replay (M7 §1) -------------------------
        //
        // A driver allocation promises address space, not pages (§1 "Deferred
        // commit") -- the slot pool proved that once already, so the audit
        // below is what stops a second term from making the same promise
        // unverified. Explicit --n-ctx is verify-only here too, but a replay
        // pass on an explicit request does not simply refuse: it retries by
        // trimming the reserve pages held for the prefix cache first (see
        // pool_sizing/explicit_overshoot_must_refuse in fit.h), and refuses
        // only once that reserve is exhausted -- --n-ctx itself is never
        // lowered, silently or otherwise.
        size_t   per_lane_blocks   = 0;
        size_t   live_blocks       = 0;
        size_t   blocks            = 0;
        size_t   spare_blocks      = 0;
        bool     accepted          = false;
        // (review finding): `budget` is constant across passes, but
        // `live_blocks` shrinks every time paged_n_ctx_ is corrected below --
        // so, unchanged, `affordable - live_blocks` (the spare-cache room)
        // GROWS by exactly what live_blocks lost, and `blocks` (live + spare)
        // reassembles to the same total every pass. Four passes then burn
        // into the terminal refusal re-allocating the identical oversized
        // pool that overshot in the first place, when a smaller pool would
        // have fit. Fixed by deducting what replay has already learned this
        // reservation overshot by from the spare-room ceiling, so spare room
        // shrinks in step with the live pool instead of silently reclaiming
        // the space live_blocks gave up.
        uint64_t overshoot_accum   = 0;
        // (review finding): a bound the retry narrows on its own, in
        // pages, independent of overshoot_accum's byte-level budget shrink
        // -- std::min against SIZE_MAX below is a no-op until a retry sets
        // it. Round-3 (defect 1): shared by BOTH the explicit-n_ctx retry
        // (explicit_retry_decision) and the auto-fit correction (auto_fit_
        // trim) now -- exactly one of the two branches ever runs for a
        // given load (explicit_n_ctx is constant for the whole call), so
        // one persistent cap is enough. Before round 3, only the explicit
        // branch narrowed this; the auto-fit branch left it untouched
        // forever, which is exactly why auto-fit's correction could shrink
        // `paged_n_ctx_` pass after pass while `wanted_spare` below grew
        // right back to absorb the cut (pool_sizing's own spare_room =
        // affordable - live_blocks) and the POOL TOTAL never moved -- see
        // auto_fit_trim's own comment in fit.h for the measured cells.
        size_t spare_cap = static_cast<size_t>(-1);
        // release_kv_pools() (below) reports how much residency left the
        // plugin's statistics when the lanes dropped their requests. Measured
        // on both cards (2026-09-03): 1.0 to 5.5 GiB per pass -- that is the
        // previous KV pool itself, which the request bindings kept alive,
        // plus whatever per-request buffers the plugin held; the two cannot
        // be separated by this reading. An earlier version added the delta
        // back into `observed` to stay conservative about the request
        // buffers and double-counted the pool: every cell then refused at
        // the 4096-token floor. The delta is logged and NOT added back; the
        // request buffers that return on the first real forward are inside
        // the reservation margin and remain unmeasured on the record.
        size_t   released_total = 0;
        uint64_t last_over      = 0;
        constexpr int kLastPass = 3;  // the loop below runs passes 0..3

        // Round-4 review: the "replay exhausted" throw below used to name
        // `paged_n_ctx_` AFTER the loop's last iteration had already
        // overwritten it with `trim.next_n_ctx` -- a depth computed FOR a
        // fifth pass that never runs (the loop condition is `pass < 4`),
        // so the throw named a number that was never actually attempted
        // or allocated. `last_attempted_n_ctx` and `attempt_history` are
        // captured once per pass, right after `blocks` is finalized for
        // that pass (the value actually handed to alloc_kv_pools), so the
        // throw can name the last depth genuinely tried and itemize every
        // pass's own (n_ctx, pool pages) instead.
        int last_attempted_n_ctx = paged_n_ctx_;
        std::vector<std::pair<int, size_t>> attempt_history;  // (n_ctx, pool pages) per pass

        // L6 (round-2 review): the reserve and an explicit --n-ctx must
        // never both be live going into this replay loop -- the spare-
        // trimming explicit_n_ctx retry below assumes it owns the whole
        // spare budget, and reserve_applied assumes nothing downstream
        // trims the headroom it just carved out of max_ctx; the two
        // assumptions conflict; config.cpp refuses the combination at
        // parse time and the guard just above (`fit.admissible` gated
        // `reserve_applied`, which itself required `!explicit_n_ctx`)
        // never sets both, but a loud check here catches a future edit
        // that loosens either guard before it can silently produce an
        // undersized or oversized pool instead of a clean refusal. Not an
        // `assert()`: this build has no OPENVINO-free way to guarantee
        // NDEBUG is unset, and a reservation-loop invariant is exactly the
        // kind of thing this file fails loud on rather than compiling out.
        if (reserve_applied && explicit_n_ctx) {
            throw std::runtime_error(
                "internal: prefix-cache reserve applied together with an explicit --n-ctx -- "
                "config.cpp should have refused this combination before the reservation "
                "loop ever ran");
        }

        for (int pass = 0; pass < 4 && !accepted; ++pass) {
            // (review finding): release the previous pass's pool from
            // every lane before asking the driver for a new one -- a fresh
            // InferRequest drops the old KV pool binding immediately,
            // instead of the retry requesting a second full pool on top of
            // a resident one. Shared by both the explicit and auto-fit
            // retries (this call site is common to both), and a no-op on
            // pass 0 for the pool itself (nothing bound by THIS loop yet --
            // finding 9: the plateau probe's own pool, if any, is still
            // resident on pass 0, a pre-existing, small instance of the
            // same on-top-of-a-resident-pool case).
            if (pass > 0) released_total += release_kv_pools(device);
            // lanes x n_ctx of live pages, plus headroom for cached prefixes
            // -- but only as much headroom as the prefix cache could ever
            // hold references to. Its host-side budget bounds how many
            // entries exist, each entry maps at most one lane's worth of
            // pages, and pages nothing can point at are just VRAM taken off
            // the card for nothing.
            per_lane_blocks =
                (static_cast<size_t>(paged_n_ctx_) + drafts_max_ + kv_block_tokens_ - 1) /
                    kv_block_tokens_ + 2;
            live_blocks = per_lane_blocks * static_cast<size_t>(lanes);
            const long long budget_remaining =
                static_cast<long long>(budget) - static_cast<long long>(overshoot_accum);
            // M7 defect fix (measured 2026-09-03: an explicit --n-ctx well
            // under the admissible max refused at allocation time -- "22.53
            // GiB resident against a 22.46 GiB ceiling" at n_ctx 32768 --
            // while omitting --n-ctx served at the same ~22.5 GiB residency
            // regardless of the token count). `pool_sizing` (fit.h, tested
            // device-free in test_fit.cpp) is the extracted form of this
            // block count: the pool is sized in BYTES from whatever the
            // live request does not use, so `spare_blocks` can dwarf
            // `live_blocks` whenever the prefix cache's own host budget
            // wants more entries than the remaining device budget can give
            // -- exactly what made the pool balloon to near the auto-fit
            // maximum no matter how small the explicit request was, and
            // exactly what the overshoot handling below must now trim
            // before it ever refuses an admissible request.
            size_t wanted_spare = 0;
            if (prefix_cache_ != nullptr) {
                const size_t entries =
                    std::max<size_t>(1, prefix_cache_->budget_bytes() /
                                            std::max<size_t>(la_row_bytes_, 1));
                wanted_spare = entries * per_lane_blocks;
            }
            //: once a retry (explicit or, since round 3, auto-fit) has
            // learned a spare cap from a prior pass, it bounds every
            // subsequent pass's spare request too -- otherwise a shrinking
            // cache budget could still race a stale, larger cap.
            wanted_spare = std::min(wanted_spare, spare_cap);
            const PoolSizing sizing =
                pool_sizing(live_blocks, wanted_spare, budget_remaining, kv_block_bytes);
            blocks       = sizing.blocks;
            spare_blocks = sizing.spare_blocks;
            // A cap for tests: the only way to make the cache evict on demand
            // at a small context. Never below what the lanes themselves need.
            // Round-3 review, finding 7 (the M7 fit review, not this
            // milestone's round 3): this override is test-only (real loads
            // never set --kv-pool-pages) and does not read `spare_cap` at
            // all -- it resets `blocks` to a FIXED `cap` independent of
            // whatever a retry above just narrowed the spare bound to, so
            // under this flag a retry can re-request the identical capped
            // pool pass after pass until pool_sizing's own (uncapped)
            // result eventually drops below `cfg.kv_pool_pages` on its
            // own. Kept as-is rather than made to interact with
            // `spare_cap` -- the flag exists to force small pools for
            // cache-eviction tests, not to exercise a retry's own
            // narrowing.
            if (cfg.kv_pool_pages > 0) {
                const size_t cap =
                    std::max<size_t>(static_cast<size_t>(cfg.kv_pool_pages), live_blocks);
                if (cap < blocks) {
                    log::info("load",
                              "--kv-pool-pages: pool capped at %zu pages (would have been %zu)",
                              cap, blocks);
                    blocks       = cap;
                    spare_blocks = blocks - live_blocks;
                }
            }

            // Round-4 review: record what this pass actually attempts
            // (`blocks`, finalized above) before trying to allocate it --
            // `last_attempted_n_ctx` is read by the "replay exhausted"
            // throw below instead of the post-loop `paged_n_ctx_`, which
            // by then may already hold a depth computed for a pass that
            // never ran.
            last_attempted_n_ctx = paged_n_ctx_;
            attempt_history.emplace_back(paged_n_ctx_, blocks);

            bool     failed = false;
            uint64_t over   = 0;
            try {
                alloc_kv_pools(rctx, blocks);
            } catch (const std::exception& e) {
                // Design §5 "Fragmentation": the sums count bytes requested,
                // not address space consumed, so a real allocation can fail
                // at a budget that said it fits. Deliberately a different
                // sentence from the overshoot below -- the two causes have to
                // stay distinguishable on the record.
                log::warn("load",
                          "allocation failed at a budget that said it fits -- fragmentation, "
                          "not arithmetic (pass %d/4, n_ctx %d): %s",
                          pass + 1, paged_n_ctx_, e.what());
                failed = true;
            }
            if (!failed) {
                const size_t   observed = device_resident_bytes(device);
                // Round-3 review, finding 2 (REAL defect, corrects round-2
                // review's own F4 comment): that comment claimed `total`
                // "was sized net of the term back at the climb, so
                // `ceiling = total - margin` already reflects it" -- false.
                // `total` is `device_total_mem_size`, a constant read once
                // at load and never adjusted for anything the climb
                // charges (the climb shrinks `max_ctx`, not `total`), so a
                // bare `total - margin` never held the scratch term back at
                // all. An observed residency landing just under THAT
                // ceiling was accepted, the pool kept every one of those
                // bytes, and the scratch buffer -- not yet allocated at
                // this point in the load -- still had to fit in whatever
                // was left once the first real prefill grew it: the
                // ORIGINAL fault this whole term exists to prevent,
                // reintroduced silently by the one check that was supposed
                // to be the last line of defence against it.
                //
                // The fix: the pool may only be ACCEPTED up to `total -
                // margin - scratch_term_bytes`, where `scratch_term_bytes`
                // (packed_values_scratch_reservation_bytes, fit.h) is
                // priced at THIS pass's own `paged_n_ctx_` -- the same
                // per-pass depth `predicted_total` below already reads. 0
                // on every load that is not under 4-bit values (both
                // scratch members default to 0), so this only ever
                // narrows the ceiling on the loads that need it, never on
                // any other.
                //
                // Round-10 review (Opus), finding 6 (stale comment
                // correction): an earlier version of this comment said the
                // ceiling "tightens and loosens in step with" `paged_n_
                // ctx_` unconditionally -- true only when `packed_values_
                // scratch_per_token_bytes_` is nonzero, which is the
                // unbounded auto-fit arm's own linear approximation
                // (fit_context_packed_values, the search). Both the
                // explicit path (fit_context_packed_values_at_depth, an
                // exact evaluation at one known depth, per_token_bytes
                // always 0) and the bounded arm (max_partitions > 0, a
                // flat lane-multiplied charge folded into `fixed_bytes_`,
                // per_token_bytes also always 0) price a CONSTANT
                // scratch_term_bytes across every pass of this loop --
                // `paged_n_ctx_` moves, the ceiling does not. Still
                // correct arithmetic (the formula is `fixed + per_token *
                // lanes * n_ctx`, and 0 * anything is 0), just not the
                // "moves together" behaviour this comment used to claim
                // for every load under 4-bit values.
                const uint64_t scratch_term_bytes = packed_values_scratch_reservation_bytes(
                    packed_values_scratch_fixed_bytes_, packed_values_scratch_per_token_bytes_,
                    lanes, static_cast<long long>(paged_n_ctx_));
                const size_t ceiling = phase_e_ceiling_bytes(total, margin, scratch_term_bytes);
                // `predicted_total` is what should already be RESIDENT the
                // instant `alloc_kv_pools` above returns -- weights,
                // drafters, the slot pool, the probed activations, and the
                // KV pool this pass just requested. The scratch term is
                // deliberately NOT folded in here (round-2 review, F4,
                // unchanged by this round): it is not part of what
                // `alloc_kv_pools` commits -- it does not exist at past 0
                // (exec/fit.h's own M9 comment) and only grows once the
                // first real prefill runs PAST this point in the load.
                // Folding it in here would make `observed < predicted_
                // total` true on every 4-bit-values load unconditionally
                // (the buffer is never resident yet at this instant),
                // turning the "deferred commit" branch below into a line
                // that always fires and calls a buffer that has not been
                // allocated yet a deferred commit of something that should
                // already be there. `ceiling` (above) is the ONLY place
                // this pass holds room back for the term -- a stricter,
                // narrower check than `predicted_total`'s "is it resident
                // yet", which is the right asymmetry: the ceiling asks
                // "will there be room later", predicted_total asks "is it
                // here already".
                const uint64_t predicted_total =
                    static_cast<uint64_t>(resident_base) + drafter_bytes + slot_pool +
                    static_cast<uint64_t>(std::max<long long>(activation_total, 0)) + margin +
                    static_cast<uint64_t>(lanes) *
                        (static_cast<uint64_t>(slab) +
                         static_cast<uint64_t>(paged_n_ctx_) * static_cast<uint64_t>(kv_bytes_token_));
                if (observed > ceiling) {
                    over = observed - ceiling;
                    log::warn("load",
                              "reservation overshoot: %.2f GiB resident against a %.2f GiB "
                              "ceiling after allocating n_ctx %d (pass %d/4) -- correcting",
                              static_cast<double>(observed) / (1u << 30),
                              static_cast<double>(ceiling) / (1u << 30), paged_n_ctx_, pass + 1);
                    failed = true;
                } else if (static_cast<uint64_t>(observed) < predicted_total) {
                    // Design §1 Phase E: residency BELOW prediction is a
                    // deferred commit, not free memory -- the general form of
                    // the slot-pool bug this milestone exists to fix, so it
                    // must never read as headroom. Accept, but say so.
                    log::info("load",
                              "the driver reports %.2f GiB of the %.2f GiB requested -- "
                              "deferred commit; the reservation keeps the analytic figure",
                              static_cast<double>(observed) / (1u << 30),
                              static_cast<double>(predicted_total) / (1u << 30));
                    accepted = true;
                } else {
                    accepted = true;
                }
            }
            last_over = over;  // finding 3: for the itemized "replay exhausted" throw below
            if (accepted) break;

            if (explicit_n_ctx) {
                // M7 defect fix (round 1) + /(round 2) + finding 3/4
                // (round 3): explicit_retry_decision (fit.h, tested
                // device-free -- the extracted form of this whole branch)
                // decides refuse-or-retry AND, when retrying, the spare
                // cap for the NEXT pass, in one call that agrees with
                // explicit_overshoot_must_refuse on every input by
                // construction. `over > 0` only when this pass's failure
                // was the residency-vs-ceiling reading above; `over == 0`
                // means a raw allocation exception (fragmentation) with no
                // residency delta to measure. Passing `pass` (this pass,
                // 0-indexed) and `kLastPass` lets the function both force
                // the LAST pass live-only (finding 3: the one direct test
                // of whether --n-ctx itself is honourable, tried before
                // the loop can run out on a pool never reduced to just the
                // request) and refuse immediately if THIS was already the
                // last pass -- there is no pass 5 to retry into, so
                // finding 6's "don't print retrying (pass 4/4)" follows
                // structurally: that log line is only reached when
                // `decision.refuse` is false, which cannot happen on
                // `pass == kLastPass`. `decision.n_ctx_unchanged` is
                // pinned `true` by the function's own signature (it does
                // not take `paged_n_ctx_` or `live_blocks` at all) --
                // nothing below reads or writes paged_n_ctx_ either way.
                const ExplicitRetryDecision decision = explicit_retry_decision(
                    /*measured_overshoot=*/over > 0, over, spare_blocks, kv_block_bytes, pass,
                    kLastPass);
                if (decision.refuse) {
                    //: itemize the pool's own terms in pages, not just
                    // the fit's byte terms -- a genuine overshoot (over >
                    // the whole spare) and a reserve-caused one both reach
                    // this throw, and only the pool terms tell them apart.
                    // `spare_blocks` here is the ACTUAL value this pass
                    // saw (0 whenever refusal came from exhaustion or from
                    // the last-pass forcing; finding 3 -- the last pass
                    // failing is exactly the reserve-exhausted case, not a
                    // separate one).
                    const size_t over_pages =
                        kv_block_bytes > 0
                            ? static_cast<size_t>((over + kv_block_bytes - 1) / kv_block_bytes)
                            : 0;
                    throw std::runtime_error(log::format(
                        "--n-ctx %d could not be honoured at allocation time on %d lane%s (%s); "
                        "the reservation had %zu live page%s and %zu spare page%s left to trim "
                        "(%zu page%s over) -- --n-ctx is verify-only and is never lowered "
                        "automatically. Lower --n-ctx, lower --parallel, or lower "
                        "--prefill-chunk.",
                        paged_n_ctx_, lanes, lanes == 1 ? "" : "s",
                        itemized_terms(paged_n_ctx_).c_str(), live_blocks,
                        live_blocks == 1 ? "" : "s", spare_blocks, spare_blocks == 1 ? "" : "s",
                        over_pages, over_pages == 1 ? "" : "s"));
                }
                // Finding 7: "capping" rather than "trimming N of M" --
                // the realized spare on the next pass can come out even
                // lower than `decision.next_spare_cap` if that pass's own
                // `budget_remaining` is <= 0 (pool_sizing snaps spare to 0
                // in that case, independent of this cap), so this cap is a
                // bound this retry is asking for, not a delta guaranteed
                // to land exactly.
                if (over > 0) {
                    log::info("load",
                              "reservation overshoot on an explicit --n-ctx came from spare pages "
                              "reserved for the prefix cache (%zu of %zu pages), not the %d-token "
                              "request -- capping spare at %zu page%s for pass %d/4",
                              spare_blocks, blocks, paged_n_ctx_, decision.next_spare_cap,
                              decision.next_spare_cap == 1 ? "" : "s", pass + 2);
                } else {
                    //: fragmentation, not an arithmetic overshoot -- but
                    // spare_blocks > 0 here (decision.refuse already
                    // handled spare_blocks == 0), so retry with a
                    // synthesized retreat instead of refusing outright,
                    // same as the auto-fit branch's own `over == 0` guard.
                    log::info("load",
                              "allocation failed at a budget that said it fits, with %zu spare "
                              "pages still reserved for the prefix cache -- no residency reading "
                              "to size a trim from, so capping spare at %zu page%s for pass %d/4",
                              spare_blocks, decision.next_spare_cap,
                              decision.next_spare_cap == 1 ? "" : "s", pass + 2);
                }
                spare_cap = decision.next_spare_cap;
                overshoot_accum += over;
                continue;
            }

            if (over == 0) {
                // A raw allocation failure with no useful residency delta
                // (the fragmentation case): there is no measured `over` to
                // correct with, so retreat by a fixed fraction to guarantee
                // forward progress instead of retrying the same request.
                over = static_cast<uint64_t>(kv_bytes_token_) * static_cast<uint64_t>(lanes) *
                       static_cast<uint64_t>(
                           std::max<long long>(paged_n_ctx_ / 8, kv_block_tokens_));
            }
            //: feeds the spare-room ceiling above on the next pass, so it
            // shrinks with the live pool instead of silently reabsorbing the
            // space live_blocks just gave up.
            overshoot_accum += over;

            // Round-3 (defect 1): auto_fit_trim (fit.h) replaces a plain
            // shrink_n_ctx call here -- shrink_n_ctx alone only ever
            // touched `paged_n_ctx_` (live pages), and left `wanted_spare`
            // uncapped, so whenever spare_blocks > 0 pool_sizing's own
            // spare_room = affordable - live_blocks grew back by exactly
            // what live_blocks lost and the POOL TOTAL never moved --
            // measured: coder/16 GiB with --prefix-cache-reserve 25 (pass
            // 1 "14.86 GiB resident against a 14.86 GiB ceiling at n_ctx
            // 100224", then 100080/100064/100048 with residency
            // UNCHANGED, exhausted) and the 35B/24 GB card's plain
            // auto-fit cell (262,144 -> 260080/260064/260048, same
            // shape). auto_fit_trim shrinks the POOL TOTAL instead,
            // splitting the cut between live (`next_n_ctx`) and spare
            // (`next_spare_cap`, applied via `spare_cap` above, the same
            // variable the explicit branch narrows) according to
            // `reserve_applied`: with an explicit --prefix-cache-reserve
            // honoured, live gives first and spare is protected until
            // live has nothing left (any shortfall that leaves is what
            // the post-hoc `prefix_cache_reserve_shortfall` warning
            // below reports); without one, spare gives first since
            // nobody asked to keep it and cutting live changes the
            // served context length. Either way the cap it hands back
            // for `spare_cap` never exceeds the CURRENT pass's own
            // spare_blocks, which is what stops pool_sizing's room
            // growth from re-inflating it next pass.
            const AutoFitTrim trim =
                auto_fit_trim(paged_n_ctx_, spare_blocks, over, kv_block_bytes, lanes,
                             static_cast<int>(kv_block_tokens_), fterms.n_ctx_floor, pass,
                             /*protect_spare=*/reserve_applied);
            if (trim.refuse) {
                throw std::runtime_error(log::format(
                    "the reservation cannot be honoured even at the floor (%d tokens) on %d "
                    "lane%s (%s). Lower --parallel, lower --prefill-chunk, or free memory on "
                    "the card.",
                    fterms.n_ctx_floor, lanes, lanes == 1 ? "" : "s",
                    itemized_terms(paged_n_ctx_).c_str()));
            }
            paged_n_ctx_ = static_cast<int>(trim.next_n_ctx);
            spare_cap    = trim.next_spare_cap;
        }
        if (!accepted) {
            // Finding 3: itemized the same way as the explicit "could not
            // be honoured" throw above -- live/spare/over pages plus the
            // fit terms. For the explicit path this line is defensive: the
            // last pass (forced live-only, see explicit_retry_decision)
            // always either accepts or throws the itemized "could not be
            // honoured" message from inside the loop, so it never falls
            // through to here. The auto-fit path has no such forcing and
            // can still exhaust all 4 passes without reaching its own
            // floor, which is what this throw is for.
            //
            // Round-4 review: named from `last_attempted_n_ctx` (the depth
            // the LAST pass actually asked alloc_kv_pools for), not
            // `paged_n_ctx_` -- by this point `paged_n_ctx_` already holds
            // `trim.next_n_ctx` from the loop's final iteration, a depth
            // computed for a fifth pass that never runs and was never
            // tried. `attempt_history` lists every pass's own (n_ctx, pool
            // pages) so the whole trimmed-pages record survives past the
            // log lines above, which the message otherwise only points at.
            const size_t over_pages =
                kv_block_bytes > 0
                    ? static_cast<size_t>((last_over + kv_block_bytes - 1) / kv_block_bytes)
                    : 0;
            std::string history;
            for (size_t i = 0; i < attempt_history.size(); ++i) {
                history += log::format("%spass %zu: n_ctx %d (%zu pages)", i == 0 ? "" : ", ",
                                       i + 1, attempt_history[i].first,
                                       attempt_history[i].second);
            }
            throw std::runtime_error(log::format(
                "reservation replay exhausted 4 passes without a stable allocation on %d "
                "lane%s -- last attempted n_ctx %d (%s); that pass had %zu live page%s and "
                "%zu spare page%s left to trim (%zu page%s over) -- the card is contended or "
                "the arithmetic above is wrong -- see the log lines above for which. "
                "Attempts: %s.",
                lanes, lanes == 1 ? "" : "s", last_attempted_n_ctx,
                itemized_terms(last_attempted_n_ctx).c_str(), live_blocks,
                live_blocks == 1 ? "" : "s", spare_blocks, spare_blocks == 1 ? "" : "s",
                over_pages, over_pages == 1 ? "" : "s", history.c_str()));
        }

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
        // (review finding): zero spare is a legal accepted configuration
        // (the live request alone fit, no room left over), but with a
        // prefix cache configured it silently means every cached prefix
        // now competes with the live lanes and evicts on demand -- worth a
        // warn, not just the pool line above.
        //
        // H2 (round-2 review, HIGH -- corrects the M9 round): the first cut
        // here gated this warning's suppression on `reserve.spare_pages >
        // 0`, reasoning that a reserve which asked for headroom at all
        // should silence the generic zero-spare warning. But
        // `prefix_cache_reserve` PROVES `spare_pages > 0` for any PCT in
        // [1, 90] with a nonzero affordable-page count (fit.h's own
        // comment on that function) -- so that gate could never be false
        // whenever the reserve was requested, and the warning went silent
        // unconditionally instead of only when the reserve was actually
        // honoured. It never fired the one case it existed to catch: the
        // allocation-time replay loop above coming back with LESS spare
        // than the reserve computed (real memory pressure, or a
        // --kv-pool-pages test cap trimming it after the fact).
        //
        // Split into the two cases that were being conflated:
        //   - no reserve requested (`!reserve_applied`): the pre-existing
        //     generic warning, unchanged.
        //   - reserve requested: `prefix_cache_reserve_shortfall` (fit.h,
        //     tested device-free) compares what was ASKED against what
        //     was ACCEPTED (spare_blocks) and names both numbers -- the
        //     true alarm, and the only one of the two branches that can
        //     actually fire.
        //
        // Round-3 review, defect 2 (units): `reserve.spare_pages` is
        // PER-LANE, pure-KV pages (fit_context's own max_ctx, which its
        // comment states is already "per lane"); `spare_blocks` is
        // POOL-WIDE and also nets out the per-lane guard overhead
        // (lookahead/rollback + drafter pages) `per_lane_blocks` bakes
        // into the live side. Comparing them directly (the round-2 H2
        // fix's own call site) was wrong on both axes -- under-detected a
        // real shortfall by a factor of `lanes` on more than one lane,
        // and false-alarmed by the guard overhead even on one (measured:
        // 18,313 asked vs 18,311 accepted with nothing actually short).
        // `reserve_ask_pool_pages` (fit.h) converts the ask into
        // `spare_blocks`'s own units first.
        if (reserve_applied) {
            const long long ask_pool_pages = reserve_ask_pool_pages(
                reserve.spare_pages, lanes, static_cast<int>(drafts_max_),
                static_cast<int>(kv_block_tokens_));
            if (prefix_cache_reserve_shortfall(spare_blocks, ask_pool_pages)) {
                log::warn("load",
                          "prefix-cache reserve asked %lld pool page%s, the accepted pool "
                          "holds %zu page%s",
                          ask_pool_pages, ask_pool_pages == 1 ? "" : "s", spare_blocks,
                          spare_blocks == 1 ? "" : "s");
            }
        } else if (spare_blocks == 0 && prefix_cache_ != nullptr) {
            log::warn("load",
                      "accepted pool has 0 spare pages for cached prefixes -- "
                      "--prefix-cache-mib is configured but every cached prefix will now "
                      "compete with the live lanes and evict on demand");
        }

        // M9 engine-side belt for 4-bit paged VALUES, empirical (fit.h's
        // prefill_chunk_cap_for_packed_values / kPrefillScratchBudgetBytesPackedValues
        // carry the measurement and the "proxy, not a mechanism" caveat).
        // Runs HERE, after Phase E, so `paged_n_ctx_` is the SERVED depth --
        // --n-ctx clamping, --prefix-cache-reserve and every replay/trim
        // pass have already run (round-2 review, finding 4: applying this
        // at the pre-trim auto-fit `max_ctx` capped an explicit --n-ctx well
        // under a deep auto-fit ceiling for a depth it would never reach).
        // Gated on `value_prec` (what was REQUESTED), not the compiled
        // port's own type: this plugin generation always compiles paged KV
        // ports at 8 bits, even for a packed 4-bit request (the port audit
        // above), so the compiled type alone cannot tell a packed-4-bit
        // value side apart from a plain 8-bit one -- only the request can.
        // u8/f16 values never reach this branch, and the cap only ever
        // lowers `prefill_chunk_`, never raises it -- an explicit
        // --prefill-chunk is capped the same way, logged rather than
        // silently. A byte-exact ladder must record the EFFECTIVE chunk
        // (status_.reservation.prefill_chunk, set below, after this) rather
        // than assume the requested one: chunk boundaries move where the
        // graph slices logits.
        //
        // Round-2 review residual 3: the activation reservation above
        // (`activation_total`, `status_.reservation.activation_bytes`) was
        // already probed and fitted at the PRE-cap `chunk` -- the belt runs
        // after Phase E, well after that probe climb. When it fires, the
        // reservation is left over-charging activations for a chunk larger
        // than what is actually served -- conservative (a real budget term
        // this load will not spend), never unsafe. Not corrected by a
        // second probe here: re-probing at the capped chunk would cost
        // another forward pass on every load that hits this branch, for a
        // number that can only move the reservation DOWN. /status reports
        // both fields side by side (`activation_bytes` at the pre-cap
        // chunk, `prefill_chunk` post-cap) rather than silently reconciling
        // them, so the mismatch is legible instead of hidden.
        // ARCINT_PREFILL_CHUNK_CAP=off is the measurement switch: it keeps the
        // requested chunk so the fault line itself can be reproduced on a card
        // (the belt would otherwise hide it); logged loudly, never a default.
        // `cap_off` and `packed_geom` were already resolved once, before the
        // climb above (the fit-side term needs the same two facts) -- reused
        // here rather than re-read/re-parsed a second time.
        //
        // Round-15 review (Opus, REAL defect, RETRACTS this warning's own
        // original wording -- kept on the record, per DESIGN §7.0.1): the
        // switch used to disable only the belt (chunk-shrinking) and the
        // measured cap -- the fit.h fixed-point (`fit_context_packed_
        // values`/`_at_depth`, both above) still charged a real,
        // depth-scaled scratch term at whatever chunk that left, and a
        // card measurement showed that charge ALONE refusing a load
        // (`--n-ctx 101824 --prefill-chunk 128`: 1,200.2 MiB charged,
        // only 32,256 admitted) the switch was supposed to let through.
        // A switch whose purpose is reproducing the plugin's OWN fault
        // line cannot do that if this repository's own budget math
        // refuses the load first -- the fault lives in the very pool the
        // term forbade. Both fit.h primitives now treat `belt_enabled =
        // false` as a FULL bypass (no belt, no cap, no term -- see their
        // own retraction comments), so the warning is corrected to say
        // that plainly rather than naming only two of the three things
        // it actually does.
        if (packed_values && cap_off) {
            log::warn("load", "%s",
                      "ARCINT_PREFILL_CHUNK_CAP=off: the 4-bit-values prefill-chunk belt, its "
                      "measured cap, AND its scratch-buffer reservation term are all disabled "
                      "for this load (charged 0) -- measurement only, so the plugin's own "
                      "kernel is what faults, not this repository's own budget estimate; a "
                      "long prefill may fault or OOM with no reservation to catch it first");
        }
        if (packed_values && !cap_off) {
            const std::optional<PackedValuesScratchGeometry>& geometry = packed_geom;
            if (geometry) {
                // Round-2 review, F1: the belt is non-increasing in depth,
                // and every path to `paged_n_ctx_` (min(wanted, max_ctx),
                // the prefix-cache reserve, Phase E's own trims) only ever
                // LOWERS the served depth from the climb's own settled
                // max_ctx -- so a trim that crosses one of the belt's own
                // step boundaries can hand back a LARGER chunk here than
                // `packed_values_scratch_chunk_` (the chunk the reservation
                // above actually priced the term at), even though the
                // served depth only went down (measured shape: max_ctx
                // 131,104 -> term priced at chunk 32; Phase E trims one
                // page to 131,072 -> the belt, asked fresh from the
                // unclamped `prefill_chunk_`, would pick 64 -- a real
                // shortfall against the charged term, see the member's own
                // comment). `lgc::belt_requested_chunk` (fit.h -- extracted
                // round-3 review, finding 3, so a test exercises this exact
                // implementation rather than a second copy of the choice)
                // caps the belt's OWN requested chunk to
                // `min(prefill_chunk_, packed_values_scratch_chunk_)`,
                // keeping its answer bounded by the priced chunk no matter
                // what depth it is asked about: the belt only ever shrinks
                // its input, never grows it, so the served chunk can never
                // exceed the smaller of the two -- and with served n_ctx
                // <= the priced max_ctx too, the charged term stays >= what
                // is actually served, unconditionally.
                //
                // 0015 engine side: the belt's own proxy must price the
                // SAME buffer the term above charged -- the same bounded
                // partitions, the same element size -- or it halves the
                // chunk for a buffer that no longer exists at that size
                // (design note o-0015-design.md §C). `paged_attention_
                // element_bytes` / `_max_partitions_effective` were already
                // resolved once, at the climb above (reused here rather
                // than re-read a second time, same convention `cap_off` and
                // `packed_geom` already follow); `packed_values_bounded_
                // partitions` (fit.h) applies the SAME bound this pass's
                // own SERVED depth (`paged_n_ctx_`) that fit_context_
                // packed_values applied to its own candidate depths during
                // the climb.
                const long long belt_partitions = packed_values_bounded_partitions(
                    static_cast<long long>(paged_n_ctx_), paged_attention_max_partitions_effective);
                const int belt_input = belt_requested_chunk(prefill_chunk_, packed_values_scratch_chunk_);
                // Patch 0020 arm: the budget ladder prices a buffer the
                // microkernel path does not allocate, so only the measured
                // cap applies here too (the same choice the climb made).
                const int capped =
                    packed_values_mixed_stage_on_micro_
                        ? packed_values_measured_chunk_cap(belt_input, cfg.kv_block_size)
                        : prefill_chunk_cap_for_packed_values_ex(
                              belt_input, belt_partitions, geometry->heads, geometry->head_size,
                              paged_attention_element_bytes,
                              kPrefillScratchBudgetBytesPackedValues, cfg.kv_block_size);
                // Round-12 review (Opus), finding 2 (REAL defect): this
                // used to log a "prefill chunk X -> Y" reduction with its
                // own limit name whenever `capped < prefill_chunk_` --
                // but `belt_input` above is ALREADY `prefill_chunk_`
                // itself (round-11 review made `prefill_chunk_` the
                // search/at_depth fixed point c*, and `belt_requested_
                // chunk` caps the belt's own input at that same value),
                // and the belt never raises its output past its input,
                // so `capped <= belt_input == prefill_chunk_` always --
                // and since every path to `paged_n_ctx_` only ever
                // LOWERS the served depth from the depth the term was
                // priced at, and the belt is non-increasing in depth, a
                // shallower re-evaluation here can only re-derive the
                // SAME chunk, never a smaller one: `capped ==
                // prefill_chunk_`, always, in practice. This branch was
                // UNREACHABLE; the reduction log (and its limit naming)
                // moved to where the chunk is actually decided -- the
                // search/at_depth call site above, comparing against
                // `configured`. The assignment below stays as a
                // defensive no-op: the one thing that would make `capped
                // != prefill_chunk_` is a future change to the
                // depth-only-shrinks invariant this comment states but
                // does not enforce in code.
                if (capped != prefill_chunk_) {
                    prefill_chunk_ = capped;
                }
            } else if (packed_values_mixed_stage_on_micro_) {
                // Patch 0020 arm without geometry (§7.0.2at review): no
                // term to size, but the measured cap needs no geometry --
                // the seed above already applied it; re-applied here for
                // the same defensive reason the geometry branch keeps its
                // own assignment.
                prefill_chunk_ = packed_values_measured_chunk_cap(prefill_chunk_, cfg.kv_block_size);
            } else {
                log::warn("load", "%s",
                          "4-bit paged KV values requested, but this artifact's config.json "
                          "does not carry num_attention_heads and head_dim -- the "
                          "prefill-chunk scratch belt cannot be sized and is skipped; the "
                          "plugin's opt paged-attention path may still fault at depth on a "
                          "tight card");
            }
        }
        // The snapshot grid. Tied to the chunk until 2026-08-30, which on the
        // agent re-prefilled ~1900 tokens per continuation where 128 would
        // re-prefill ~970 (replay of real sessions, DESIGN 7.0.2j). It must
        // divide a page, because an entry keeps whole pages, and be a multiple
        // of the cache's hash block; it cannot exceed the chunk. Computed
        // here (not right after the auto-fit climb decided `chunk`) so it
        // reflects the belt above, when the belt fired -- nothing between
        // the auto-fit climb and here reads `cache_grid_` (grepped: its only
        // other reader is the runtime prefill() path, long after load
        // returns), so deriving it this late is safe.
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

        status_.reservation.measured           = true;
        status_.reservation.device_total_bytes = total;
        status_.reservation.weights_bytes      = resident_base;
        status_.reservation.drafter_bytes          = drafter_bytes;
        status_.reservation.expert_slot_bytes      = slot_pool;
        status_.reservation.expert_slot_host_bytes = slot_host_bytes;
        status_.reservation.slot_source            = slot_source;
        // Reported as the total for all lanes, because that is what it is: the
        // plugin's intermediate pool is per compiled model. Dividing it by the
        // lane count would invent a per-lane cost that nobody pays.
        //
        // Round-2 review residual 3: on a 4-bit-values load where the belt
        // above fired, this is the PRE-cap chunk's own probed figure --
        // `prefill_chunk` a few lines down is post-cap. The two are
        // deliberately not reconciled (see the belt's own comment above);
        // reading them together is how an operator sees the over-charge.
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

    // (review finding): before the replay loop re-allocates the pool on
    // a retry, drop the previous pass's binding first. `kv_pool_tensors_.
    // clear()` in alloc_kv_pools below only drops arcint's own handles --
    // every lane's InferRequest keeps its pass-N tensors bound until
    // bind_kv_pools runs again after create_tensor, so without this a
    // retry requests a whole second pool on top of a resident one instead
    // of replacing it. A fresh InferRequest per lane is the same remedy
    // reset_lane_request already uses for a probe that threw, just without
    // its final bind_kv_pools call -- there is no pool to bind to yet.
    //
    // Round-3 review, finding 2: the departing request is not only the KV
    // pool binding -- on the FIRST call, it is also the plateau probe's
    // request, still holding the plugin's per-request device buffers
    // (`inputs_embeds`, `position_ids`, the probe's own output buffer at
    // its largest probed chunk). Those are real, measured residency (the
    // probe's `activation_total` was fitted while they were resident), and
    // dropping the request frees them immediately -- they do not come back
    // until the first real forward re-allocates them, which happens after
    // this whole reservation completes. Read device_resident_bytes()
    // before and after so the caller can charge the freed amount back to
    // `observed` on every remaining pass (conservative: still owed, not
    // headroom) instead of it silently reading as freed VRAM.
    //
    // Finding 9: `lane->req = {};` first, THEN create the new request --
    // `lane->req = paged_model_.create_infer_request()` alone evaluates
    // the RHS (allocating the new request) before the assignment destroys
    // the old one, so for one instant both are live: a transient second
    // request per lane, unwanted on a card that is already full.
    size_t release_kv_pools(const std::string& device) {
        const size_t before = device_resident_bytes(device);
        for (auto& lane : lanes_) {
            lane->blocks.clear();
            lane->req = {};
            lane->req = paged_model_.create_infer_request();
            for (size_t i = 0; i < la_state_names_.size(); ++i) {
                lane->req.set_tensor(la_state_names_[i], lane->la_tensors[i]);
            }
        }
        kv_pool_tensors_.clear();
        const size_t after = device_resident_bytes(device);
        const size_t freed = after < before ? before - after : 0;
        if (freed > 0) {
            log::info("load", "released the previous pool and request buffers: %.2f MiB",
                      static_cast<double>(freed) / (1u << 20));
        }
        return freed;
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
            // Every exit path -- normal return, an early `return
            // FinishReason::Length` out of the prefill loop when the KV pool
            // is exhausted, or an exception unwinding out of the graph --
            // passes through here, which is why this (not a line before one
            // particular `return`) is where the lane's cumulative dflash
            // failure count gets into this request's stats.
            stats_.dflash_lane_failures = static_cast<int>(lane_.dflash_fail_count);
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
        // ARCINT_PROFILE_CYCLE: the index-build segment below (pos, blocks,
        // the 9 x set_i32 into USM host) is timed only under the switch --
        // this function runs on every prefill chunk too, and an unconditional
        // pair of clock reads there is not the "tens of nanoseconds against a
        // forward measured in milliseconds" case the file already accepts at
        // 5278 (a prefill chunk can be much cheaper than a decode step).
        static const bool profile_cycle = profile_cycle_enabled();
        const auto t_index0 = profile_cycle ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
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
        // ARCINT_FORWARD_SPLIT=1: wall clock of the three parts of a paged
        // forward, averaged over 64 forwards and logged, so a served step's
        // cost can be attributed to the graph, the logits readback or the
        // hidden-state readback instead of guessed. Measurement switch only.
        static const char* split_env    = std::getenv("ARCINT_FORWARD_SPLIT");
        static const bool  split_log    = split_env != nullptr;
        // ARCINT_FORWARD_SPLIT=2: also measurement only. Re-runs the just-
        // completed forward a second time, immediately after the hidden-
        // state copy, with the same inputs already set on lane.req (nothing
        // here changes them) -- the repeat re-writes the same KV rows the
        // first infer just wrote, which is harmless for a measurement run
        // (identical inputs, so an idempotent write). This isolates whether
        // the served forward's cost at depth (measured 2026-09-03, 76k: the
        // MTP verify step at 430 ms against 211 ms for an isolated plain
        // forward) is STATEFUL/queue-related (a repeat on an already-warm
        // request should land near the isolated 211 ms) or INPUT-related (a
        // repeat should still cost close to 430 ms).
        static const bool split_repeat = split_log && std::string(split_env) == "2";
        if (profile_cycle) {
            lane.last_fwd_ms_index = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - t_index0)
                                          .count();
        }
        with_turn(lane, [&] {
            // Four steady_clock::now() reads every served forward (t0..t3),
            // whether or not ARCINT_FORWARD_SPLIT is set: on this host
            // std::chrono::steady_clock::now() costs tens of nanoseconds,
            // negligible against a forward measured in milliseconds -- an
            // always-on read here is not a measurable tax.
            const auto t0 = std::chrono::steady_clock::now();
            lane.req.infer();
            const auto t1 = std::chrono::steady_clock::now();
            if (want_logits) copy_out(lane.req.get_tensor("logits"), lane.logits);
            const auto t2 = std::chrono::steady_clock::now();
            if (mtp_ready_) copy_out(lane.req.get_tensor("hidden_states"), lane.hidden);
            const auto t3 = std::chrono::steady_clock::now();
            // Mirrored onto the lane so generate_paged's cycle line (M11 step
            // profile) can read this forward's split without a clock read of
            // its own -- t0..t3 above are already unconditional.
            lane.last_fwd_ms_infer  = std::chrono::duration<double, std::milli>(t1 - t0).count();
            lane.last_fwd_ms_logits = std::chrono::duration<double, std::milli>(t2 - t1).count();
            lane.last_fwd_ms_hidden = std::chrono::duration<double, std::milli>(t3 - t2).count();
            // The repeat's own two reads are gated on split_repeat, not
            // unconditional like t0..t3 above: they exist only to time a
            // forward that itself only happens under ARCINT_FORWARD_SPLIT=2.
            double repeat_ms = 0.0;
            if (split_repeat) {
                const auto tr0 = std::chrono::steady_clock::now();
                lane.req.infer();
                const auto tr1 = std::chrono::steady_clock::now();
                repeat_ms = std::chrono::duration<double, std::milli>(tr1 - tr0).count();
            }
            if (dflash_active(lane) && want_feats) {
                copy_out(lane.req.get_tensor("dflash_feats"), lane.dfeats);
            }
            if (split_log) {
                static double a_inf = 0, a_log = 0, a_hid = 0, a_rep = 0;
                static int    a_n   = 0;
                static size_t a_tok = 0;
                a_inf += std::chrono::duration<double, std::milli>(t1 - t0).count();
                a_log += std::chrono::duration<double, std::milli>(t2 - t1).count();
                a_hid += std::chrono::duration<double, std::milli>(t3 - t2).count();
                if (split_repeat) a_rep += repeat_ms;
                a_tok += n;
                if (++a_n == 64) {
                    const std::string repeat_suffix =
                        split_repeat ? log::format(", repeat infer %.1f ms", a_rep / a_n)
                                     : std::string();
                    log::info("profile",
                              "paged forward split over %d forwards (%zu tokens, past now %zu): "
                              "infer %.1f ms, logits %.1f ms, hidden %.1f ms per forward%s",
                              a_n, a_tok, past, a_inf / a_n, a_log / a_n, a_hid / a_n,
                              repeat_suffix.c_str());
                    a_inf = a_log = a_hid = a_rep = 0;
                    a_n   = 0;
                    a_tok = 0;
                }
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

    // ARCINT_PROFILE_MWALL=<comma list> overrides the token counts the M-wall
    // sweep times in both profile_step (stateful) and profile_paged (paged);
    // default {1, 2, 4, 8}. Shared so the two paths answer the same question
    // with the same M values unless a caller asks otherwise.
    static std::vector<size_t> profile_mwall_list() {
        std::vector<size_t> mwall{1, 2, 4, 8};
        if (const char* list = std::getenv("ARCINT_PROFILE_MWALL")) {
            mwall.clear();
            const std::string spec(list);
            size_t pos = 0;
            while (pos < spec.size()) {
                const size_t comma = spec.find(',', pos);
                const std::string tok = spec.substr(pos, comma - pos);
                if (!tok.empty()) mwall.push_back(std::strtoul(tok.c_str(), nullptr, 10));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        return mwall;
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

        // M-wall sweep: the served MTP verify step forwards M>1 tokens per
        // step (measured 2026-09-03 on the 24 GB card at 76k depth, u8: the
        // MTP verify forward took ~763 ms/step against ~97 ms for a plain
        // 1-token decode step -- a step timing, not yet a per-M wall clock), and the per-node PERF_COUNT table
        // just dumped exceeds either wall by ~35x and is flat across M -- it
        // cannot see this scaling. Time M-token forwards directly, on this
        // paged path, because this is the path production serves
        // (--paged-kv); profile_step's stateful path gets the same sweep too,
        // for whichever build has --no-paged instead.
        // The sweep's depth is ARCINT_PROFILE_PAST when set (the first run of
        // this hook timed M=1..8 "at depth 1" because it read the capture
        // depth, which ARCINT_PROFILE=1 sets to one token), else ARCINT_PROFILE.
        const size_t mwall_depth = past > 0 ? past : depth;
        // ARCINT_PROFILE_MWALL_LA=1: the checkpoint-rows arm below needs
        // rows [0..M] -- M+1 rows -- from the lane's own checkpoint table.
        // Read once, before the sweep, so the skip check below and the row
        // list built further down agree on the same flag.
        static const bool la_arm = std::getenv("ARCINT_PROFILE_MWALL_LA") != nullptr;
        for (size_t m : profile_mwall_list()) {
            if (m == 0) continue;
            // Measured 2026-09-03 (24 GB card): M=8 against a 4-row lane
            // silently truncated the checkpoint list to 4 rows instead of
            // the 9 the served verify graph actually addresses -- GPU page
            // faults and a device coredump, not merely a wrong number.
            // Skip the M outright rather than pass a short list.
            if (la_arm && m + 1 > rows_per_lane_) {
                log::warn("profile",
                          "M-wall LA arm: M=%zu needs %zu checkpoint rows, lane has %zu -- "
                          "skipped",
                          m, m + 1, rows_per_lane_);
                continue;
            }
            const size_t depth = mwall_depth;
            release_lane(lane);
            zero_paged_rows(lane);
            int reps = 10;
            while (reps > 0 && !ensure_blocks(lane, depth + static_cast<size_t>(reps) * m)) {
                --reps;
            }
            if (reps <= 0) {
                log::warn("profile", "not enough KV pages for M-wall M=%zu at depth %zu", m,
                          depth);
                continue;
            }
            if (reps < 10) {
                log::warn("profile",
                          "M-wall M=%zu at depth %zu: reduced to %d rep(s) for KV pages", m,
                          depth, reps);
            }
            // Same reset-and-prefill the sweep above does per M, so every M
            // starts from the same depth instead of compounding on the last
            // M's growth.
            // Walk to `depth` in served-grid chunks, as the capture sweep does:
            // one forward of the whole past is a 76k-token chunk, which the
            // card refuses (measured: CL_OUT_OF_RESOURCES on the first run).
            for (size_t at = 0; at < depth;) {
                const size_t take = std::min<size_t>(
                    depth - at, static_cast<size_t>(std::max(1, prefill_chunk_)));
                paged_forward(lane, embed_paged(lane, make_tokens(take)), at, {0}, 0, false);
                at += take;
            }
            // ARCINT_PROFILE_MWALL_LA=1: pass the served verify's checkpoint
            // inputs (rows [0..M], interval 1, one GDN-state row written per
            // token) instead of the plain step's ({0}, 0). Measured 2026-09-03
            // at 76k: the served 2-token verify graph ran 430 ms against 211 ms
            // for the plain 2-token forward; this arm isolates those inputs.
            // `la_arm` itself is read once above the sweep, alongside the
            // skip guard that keeps this loop from ever reaching here with
            // fewer than m+1 rows available.
            std::vector<int32_t> la_rows{0};
            int                  la_interval = 0;
            if (la_arm) {
                la_rows.clear();
                for (size_t r = 0; r <= m && r < rows_per_lane_; ++r) {
                    la_rows.push_back(static_cast<int32_t>(r));
                }
                la_interval = 1;
            }
            const auto t_m0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) {
                paged_forward(lane, embed_paged(lane, make_tokens(m)),
                              depth + static_cast<size_t>(i) * m, la_rows, la_interval);
            }
            const double m_wall_ms = 1000.0 * seconds_since(t_m0) / reps;
            log::info("profile",
                      "forward wall clock M=%zu at depth %zu: %.2f ms/step, %.2f ms/token "
                      "(%d reps%s)",
                      m, depth, m_wall_ms, m_wall_ms / static_cast<double>(m), reps,
                      la_arm ? ", checkpoint rows + interval 1" : "");
        }

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

        // M-wall sweep: the served MTP verify step forwards M>1 tokens per
        // step (measured 2026-09-03 on the 24 GB card at 76k depth, u8: the
        // MTP verify forward took ~763 ms/step against ~97 ms for a plain
        // 1-token decode step -- a step timing, not yet a per-M wall clock), and the PERF_COUNT node total above
        // exceeds either wall by ~35x and is flat across M -- it cannot see
        // this scaling. This times M-token forwards directly instead.
        // ARCINT_PROFILE_TOKENS=random uses pseudo-random ids, the same knob
        // profile_paged's sweep honours; make_tokens there is a lambda local
        // to that function, so this is its own small xorshift rather than a
        // shared one.
        const bool rnd_tokens = [] {
            const char* v = std::getenv("ARCINT_PROFILE_TOKENS");
            return v != nullptr && std::string(v) == "random";
        }();
        uint64_t mwall_rng = 88172645463325252ull;
        auto     mwall_tokens = [&](size_t n) {
            std::vector<int> t(n, 0);
            if (rnd_tokens) {
                for (size_t i = 0; i < n; ++i) {
                    mwall_rng ^= mwall_rng << 13;
                    mwall_rng ^= mwall_rng >> 7;
                    mwall_rng ^= mwall_rng << 17;
                    t[i] = static_cast<int>(mwall_rng % 100000U) + 1;
                }
            }
            return t;
        };
        // The stateful graph carries its KV in internal variables, not a
        // page pool -- there is nothing here to guard the way profile_paged
        // guards lane.blocks against pool_, so none is added.
        for (size_t m : profile_mwall_list()) {
            if (m == 0) continue;
            // Same reset-and-prefill this function does above, so every M
            // starts from the same depth instead of compounding on the last
            // sweep's growth.
            lm_req_.reset_state();
            forward(lane, warm, 0);
            const auto t_m0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) {
                forward(lane, mwall_tokens(m), depth + static_cast<size_t>(i) * m);
            }
            const double m_wall_ms = 1000.0 * seconds_since(t_m0) / reps;
            log::info("profile",
                      "forward wall clock M=%zu at depth %zu: %.2f ms/step, %.2f ms/token "
                      "(%d reps)",
                      m, depth, m_wall_ms, m_wall_ms / static_cast<double>(m), reps);
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
            // M11 step profile: four steady_clock::now() reads per head step,
            // unconditional, on the same "tens of nanoseconds against a
            // forward measured in milliseconds" argument paged_forward
            // already makes for its own t0..t3.
            const auto pt0 = std::chrono::steady_clock::now();
            ov::Tensor ids(ov::element::i64, ov::Shape{1, 1});
            ids.data<int64_t>()[0] = next;
            lane.embed.set_input_tensor(ids);
            lane.embed.infer();
            const ov::Tensor src = lane.embed.get_output_tensor(0);
            ov::Tensor       emb(src.get_element_type(), src.get_shape());
            std::memcpy(emb.data(), src.data(), src.get_byte_size());
            const auto pt1 = std::chrono::steady_clock::now();
            lane.mtp_step_ms_embed =
                std::chrono::duration<double, std::milli>(pt1 - pt0).count();

            const ov::Tensor pos  = mtp_positions(lane.mtp_pos, 1);
            // Nothing is masked: the head attends to its whole committed prefix.
            const ov::Tensor mask = mtp_mask(lane.mtp_len, 1);
            const auto pt2 = std::chrono::steady_clock::now();
            lane.mtp_step_ms_mask =
                std::chrono::duration<double, std::milli>(pt2 - pt1).count();

            ov::Tensor beam(ov::element::i32, ov::Shape{1});
            beam.data<int32_t>()[0] = 0;

            lane.mtp_layer.set_tensor("hidden_states", lane.mtp_pending);
            lane.mtp_layer.set_tensor(mtp_embeds_name_, emb);
            lane.mtp_layer.set_tensor("position_ids", pos);
            lane.mtp_layer.set_tensor("attention_mask", mask);
            lane.mtp_layer.set_tensor("beam_idx", beam);
            lane.mtp_layer.infer();
            const auto pt3 = std::chrono::steady_clock::now();
            lane.mtp_step_ms_layer =
                std::chrono::duration<double, std::milli>(pt3 - pt2).count();

            ++lane.mtp_len;
            ++lane.mtp_pos;
            lane.mtp_has_pending = false;
            if (!want_draft) {
                lane.mtp_step_ms_head = 0.0;
                return -1;
            }

            lane.mtp_head.set_input_tensor(lane.mtp_layer.get_output_tensor(0));
            lane.mtp_head.infer();
            const ov::Tensor lg   = lane.mtp_head.get_output_tensor(0);
            const size_t     v    = lg.get_shape().back();
            const size_t     rows = v ? lg.get_size() / v : 0;
            if (rows == 0) {
                lane.mtp_step_ms_head =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pt3)
                        .count();
                return -1;
            }
            const int drafted = Sampler::argmax(lg.data<const float>() + (rows - 1) * v, v);
            lane.mtp_step_ms_head =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pt3)
                    .count();
            return drafted;
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

    // The check every dflash_* call site wants: the drafter loaded at all
    // (process-wide, set once at startup) AND this lane's own drafter has
    // not been disabled by a runtime failure since its last reset. Kept as
    // one call so no site can accidentally check only the process-wide half
    // and draft on a lane a failure meant to keep off until its reset.
    bool dflash_active(const Lane& lane) const {
        return dflash_ready_.load(std::memory_order_relaxed) && !lane.dflash_off;
    }

    // A draft failure disables only the failing lane's drafter, until its
    // next dflash_reset (a fresh request re-arms it) -- a corrupted feature
    // buffer or a misaligned append on one sequence says nothing about the
    // head's correctness for another lane's sequence, and the pre-fix
    // process-wide `dflash_ready_ = false` left the drafter permanently off
    // for the rest of the process after one bad request (measured
    // 2026-09-03, one lane: after a long prompt's failure, a later 763-token
    // request in the same process decoded 3,000 tokens with zero accepted
    // drafts; that every other lane would have been off as well is what the
    // old code did by reading, not a measurement). Per-lane recovery is the
    // whole point, so nothing in here disables the drafter process-wide any
    // more -- kDflashWarnThreshold consecutive failures with no successful
    // draft landing in between, across however many lanes and requests, log
    // a warning on that failure and on every further one until a draft
    // succeeds; see the member declaration for why even that stopped short
    // of a disable.
    void dflash_lane_fail(Lane& lane, const std::string& why) {
        lane.dflash_off = true;
        ++lane.dflash_fail_count;
        lane.dflash_pending.clear();
        lane.dflash_base = SIZE_MAX;
        log::warn("dflash", "lane %d: draft failed, disabling this lane's drafter until its "
                            "next reset: %s",
                  lane.index, why.c_str());
        const int consec = dflash_consec_failures_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (consec >= kDflashWarnThreshold) {
            log::warn("dflash", "%d consecutive dflash failures with no successful draft "
                                "landing in between; every lane still recovers on its own "
                                "next reset, but this many in a row is worth a look",
                      kDflashWarnThreshold);
        }
    }

    void dflash_reset(Lane& lane) {
        if (!dflash_ready_) return;
        lane.dflash_req.reset_state();
        lane.dflash_pending.clear();
        lane.dflash_base = SIZE_MAX;
        lane.dflash_off  = false;   // a new request re-arms this lane's drafter
    }

    // Rows [row_from, row_from+rows) of the last forward's feature output
    // belong to absolute positions [pos, pos+rows) and are now committed
    // context. A discontinuity means the bookkeeping is wrong, and the honest
    // reaction is to stop drafting this lane, not to draft from misaligned
    // context.
    void dflash_append(Lane& lane, size_t row_from, size_t rows, size_t pos) {
        if (!dflash_active(lane) || rows == 0) return;
        const size_t width = dflash_feat_width_;
        const size_t have  = lane.dfeats.get_size() / width;
        if (row_from + rows > have) {
            dflash_lane_fail(lane, log::format("feature output has %zu row(s), needed %zu",
                                               have, row_from + rows));
            return;
        }
        const size_t pend = lane.dflash_pending.size() / width;
        if (lane.dflash_base == SIZE_MAX) {
            lane.dflash_base = pos;
        } else if (lane.dflash_base + pend != pos) {
            dflash_lane_fail(lane, log::format("feature gap: pending ends at %zu, append at %zu",
                                               lane.dflash_base + pend, pos));
            return;
        }
        const float* src = lane.dfeats.data<const float>();
        lane.dflash_pending.insert(lane.dflash_pending.end(), src + row_from * width,
                                   src + (row_from + rows) * width);
        // src/core/dflash_window.h: `keep` is the same "most recent
        // kDflashWindow rows" trim this always applied; `new_rows` is 0 here
        // because the rows just inserted above are already reflected in the
        // buffer's current size.
        const size_t total = lane.dflash_pending.size() / width;
        const auto   plan  = dflash_window_plan(total, 0, kDflashWindow);
        if (plan.keep < total) {
            const size_t drop = total - plan.keep;
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
    //
    // The loop below feeds `pend` rows to the drafter's state in the exact
    // sequence src/core/dflash_window.h's feed_steps() returns -- that
    // function, not this loop, is what tests/test_dflash_window.cpp pins,
    // so the loop under test is the loop that runs. See the header for what
    // is actually measured about why this cap exists (feeds >= kDflashWindow
    // observed to fail on GPU plugins without patch 0014; a narrower and now
    // retracted claim about exactly kDflashWindow was falsified by a later
    // 1,615-row failure). Every step before the last is a priming feed whose
    // output is discarded (only new_feats ever becomes persistent state);
    // the last step also carries the actual draft block and is the one
    // scored. Below the cap (every prompt under the window) feed_steps
    // returns a single element equal to `pend`, so the loop runs its one
    // iteration exactly as the pre-fix code always did, and the
    // accepted-per-cycle counts of a record taken under the window do not
    // move.
    std::vector<int> dflash_draft(Lane& lane, int anchor, size_t past) {
        if (!dflash_active(lane)) return {};
        try {
            const size_t width = dflash_feat_width_;
            const size_t pend  = lane.dflash_pending.size() / width;
            if (pend == 0 || lane.dflash_base + pend != past) {
                dflash_lane_fail(
                    lane,
                    log::format("pending context ends at %zu but the anchor sits at %zu",
                                lane.dflash_base == SIZE_MAX ? 0 : lane.dflash_base + pend, past));
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

                const std::vector<size_t> steps = feed_steps(pend, kDflashWindow);
                size_t                    off   = 0;
                std::vector<int>          drafts;
                for (size_t s = 0; s < steps.size(); ++s) {
                    const size_t feed       = steps[s];
                    const bool   final_step = s + 1 == steps.size();

                    ov::Tensor feats(ov::element::f32, ov::Shape{1, feed, width},
                                     lane.dflash_pending.data() + off * width);
                    ov::Tensor pos(ov::element::i64, ov::Shape{feed + q});
                    int64_t*   pp = pos.data<int64_t>();
                    for (size_t i = 0; i < feed; ++i) {
                        pp[i] = static_cast<int64_t>(lane.dflash_base + off + i);
                    }
                    for (size_t i = 0; i < q; ++i) {
                        pp[feed + i] = static_cast<int64_t>(past + i);
                    }

                    lane.dflash_req.set_tensor("new_feats", feats);
                    lane.dflash_req.set_tensor("noise", noise);
                    lane.dflash_req.set_tensor("positions", pos);
                    lane.dflash_req.infer();

                    off += feed;
                    if (!final_step) continue;   // priming step: output unused

                    const ov::Tensor out = lane.dflash_req.get_output_tensor(0);   // [1,q,h]
                    ov::Tensor rows(ov::element::f32, ov::Shape{1, q - 1, h});
                    std::memcpy(rows.data(), out.data<const float>() + h, (q - 1) * h * 4);
                    lane.dflash_head.set_input_tensor(rows);
                    lane.dflash_head.infer();
                    const ov::Tensor lg = lane.dflash_head.get_output_tensor(0);

                    drafts = dflash_select(lane, rows.data<const float>(), lg.data<const float>(),
                                           q - 1, h, anchor);
                }
                lane.dflash_pending.clear();
                lane.dflash_base = past;
                // A draft call that reaches here completed without throwing
                // and without a bookkeeping failure -- exactly the "nothing
                // in between" the breaker's consecutive count tracks; reset
                // it so an old, unrelated lane's failures do not carry
                // forward into a run that is working.
                dflash_consec_failures_.store(0, std::memory_order_relaxed);
                return drafts;
            });
        } catch (const std::exception& e) {
            dflash_lane_fail(lane, e.what());
            return {};
        }
    }

    // The candidate-path selector: top-k tokens per position, one path
    // traced through the lattice with the configured strategy
    // (--dflash-select; src/core/dflash_select.h has the pure DP). The score
    // this builds, per row and candidate j, given the token `prev` chosen for
    // the row before it (or the anchor at row 0), is
    //
    //     unary[j] + lambda * pred_cb[prev] . ((P.hidden_row) elementwise* succ_cb[token[j]])
    //
    // -- a trilinear term in (pred_cb[prev], P.hidden_row, succ_cb[cand]).
    // Before M11 this was computed strictly greedily; the Lattice this
    // function builds carries both the ingredients that reproduce the legacy
    // association order bit-for-bit at the default (kGreedy, lambda=1) --
    // `hp` and `succ_key` kept separate, per dflash_select.h's note in
    // 
    // row), which kViterbi needs because it scores every row's candidates
    // against EVERY predecessor candidate, not just the one greedy already
    // committed to, and has no legacy order to match.
    std::vector<int> dflash_select(Lane& lane, const float* hid, const float* logits, size_t rows,
                                   size_t width, int anchor) {
        const size_t k = dflash_topk_, r = dflash_rank_, vocab = dflash_vocab_;

        Lattice lattice;
        lattice.anchor_key.resize(r);
        {
            const ov::float16* ak = dflash_pred_cb_.data() + static_cast<size_t>(anchor) * r;
            for (size_t d = 0; d < r; ++d) lattice.anchor_key[d] = static_cast<float>(ak[d]);
        }
        lattice.rows.resize(rows);

        for (size_t row = 0; row < rows; ++row) {
            const float* lg = logits + row * vocab;
            // top-k: min-heap of size k over the vocab (unchanged from the
            // pre-M11 selector).
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
            LatticeRow& lr = lattice.rows[row];
            lr.token.resize(k);
            lr.unary.resize(k);
            for (size_t j = k; j-- > 0;) {
                lr.unary[j] = heap.top().first;
                lr.token[j] = heap.top().second;
                heap.pop();
            }

            const float* hrow = hid + row * width;
            lr.hp.resize(r);
            for (size_t d = 0; d < r; ++d) {
                const float* pr  = dflash_proj_.data() + d * width;
                float        acc = 0.0f;
                for (size_t x = 0; x < width; ++x) acc += pr[x] * hrow[x];
                lr.hp[d] = acc;
            }

            lr.pred_key.assign(k, std::vector<float>(r));
            lr.succ_key.assign(k, std::vector<float>(r));
            lr.row_key.assign(k, std::vector<float>(r));
            for (size_t j = 0; j < k; ++j) {
                const size_t       tok = static_cast<size_t>(lr.token[j]);
                const ov::float16* pc  = dflash_pred_cb_.data() + tok * r;
                const ov::float16* sc  = dflash_succ_cb_.data() + tok * r;
                for (size_t d = 0; d < r; ++d) {
                    lr.pred_key[j][d] = static_cast<float>(pc[d]);
                    lr.succ_key[j][d] = static_cast<float>(sc[d]);
                    lr.row_key[j][d]  = lr.hp[d] * lr.succ_key[j][d];
                }
            }
        }

        if (dflash_dump_enabled_.load(std::memory_order_relaxed)) dflash_dump_build_lattice(lane, lattice, anchor);

        return dflash_select_path(lattice, dflash_select_mode_, dflash_lambda_);
    }

    // M11 dump instrument (the M11 design note (not in the repository) O): stashes `lattice` as JSON
    // on the lane for the decode loop to append to once it knows the drafts,
    // the accepted count and the target's realized tokens (all only known
    // after the separate verify forward that follows this draft). Per row:
    // candidate token ids, their unary scores, and the FULL transition score
    // (bilinear term only, lambda not applied) from every reachable
    // predecessor -- the anchor for row 0, the previous row's own candidates
    // after. Compact since K <= 64, and it lets tools/dflash_oracle.py
    // recompute any selector's reachable path without knowing the codebooks
    // or the rank r.
    void dflash_dump_build_lattice(Lane& lane, const Lattice& lattice, int anchor) {
        nlohmann::json rec;
        rec["anchor"] = anchor;
        nlohmann::json rows = nlohmann::json::array();
        for (size_t ridx = 0; ridx < lattice.rows.size(); ++ridx) {
            const LatticeRow& row = lattice.rows[ridx];
            nlohmann::json    jrow;
            jrow["tokens"] = row.token;
            jrow["unary"]  = row.unary;
            nlohmann::json trans = nlohmann::json::array();
            if (ridx == 0) {
                // One score per candidate: the transition from the anchor.
                for (size_t j = 0; j < row.token.size(); ++j) {
                    float d = 0.0f;
                    const size_t n = std::min(lattice.anchor_key.size(), row.row_key[j].size());
                    for (size_t x = 0; x < n; ++x) d += lattice.anchor_key[x] * row.row_key[j][x];
                    trans.push_back(d);
                }
            } else {
                // [K_prev][K]: one row per previous-row candidate.
                const LatticeRow& prevr = lattice.rows[ridx - 1];
                for (size_t p = 0; p < prevr.token.size(); ++p) {
                    nlohmann::json prow = nlohmann::json::array();
                    for (size_t j = 0; j < row.token.size(); ++j) {
                        float        d = 0.0f;
                        const size_t n =
                            std::min(prevr.pred_key[p].size(), row.row_key[j].size());
                        for (size_t x = 0; x < n; ++x) d += prevr.pred_key[p][x] * row.row_key[j][x];
                        prow.push_back(d);
                    }
                    trans.push_back(prow);
                }
            }
            jrow["transition"] = trans;
            rows.push_back(jrow);
        }
        rec["lattice_rows"] = rows;
        lane.dflash_dump_lattice = std::move(rec);
    }

    // M11 dump instrument: appends one JSON line to the already-open
    // dflash_dump_file_ (opened once at construction, not per cycle).
    // Mutex-protected because lanes draft concurrently and an interleaved
    // write would corrupt the file's line-per-record contract; the same lock
    // also guards the disable-on-failure store, so there is no window where
    // another thread can observe dflash_dump_enabled_ true while the stream
    // it is about to write to is already known bad.
    void dflash_dump_write(nlohmann::json rec) {
        rec["cycle"] = dflash_dump_cycle_.fetch_add(1);
        std::lock_guard<std::mutex> lock(dflash_dump_mutex_);
        if (!dflash_dump_enabled_.load(std::memory_order_relaxed)) return;
        dflash_dump_file_ << rec.dump() << "\n";
        if (!dflash_dump_file_.good()) {
            log::warn("dflash", "write to %s failed; disabling the dump",
                      dflash_dump_path_.c_str());
            dflash_dump_enabled_.store(false, std::memory_order_release);
        }
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
    int                            pin_dispatch_ = -1;  // --pin-dispatch; -1 = off
    bool                           moe_cpu_tier_ = false;         // --moe-cpu-tier
    int                            moe_cpu_tier_threads_ = 0;     // --moe-cpu-tier-threads
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
    // M9 (2026-09-03, 16 GiB card, coder, --paged-kv u8:i4, host-side VRAM
    // sampled every 2 s): the packed-4-bit-values prefill scratch term
    // (exec/fit.h's packed_values_prefill_scratch_bytes) charged into the
    // fit's climb -- see the FitTerms/fit_context_packed_values call site
    // below. Zero except on a load where value_prec is 4-bit; carried as
    // members (not locals) so the belt call site after Phase E (below) and
    // the summary log line can read what the climb already priced instead
    // of recomputing it.
    uint64_t                       packed_values_scratch_per_token_bytes_ = 0;
    uint64_t                       packed_values_scratch_fixed_bytes_     = 0;
    // Round-2 review, F1: the chunk fit_context_packed_values's own fixed
    // point actually priced the term at (PackedValuesFitTerm::chunk) --
    // NOT `prefill_chunk_`, which the belt call site below re-derives from
    // scratch at the SERVED depth after Phase E may have trimmed it. Every
    // path to that served depth only ever lowers it (min(wanted, max_ctx),
    // the prefix-cache reserve, Phase E's own trims), and the belt is
    // non-increasing in depth (a smaller depth can only ever pick an
    // EQUAL-OR-LARGER chunk) -- so a trim that crosses one of the belt's
    // own step boundaries can hand back a chunk LARGER than the one this
    // term was priced at, even though the served depth only went down.
    // Measured shape (16 GiB card, coder, u8:i4): max_ctx settles at
    // 131,104 with the term priced at chunk 32 (~385 MiB); Phase E trims
    // one page to 131,072, and the belt, asked fresh with the unclamped
    // `prefill_chunk_` at that new depth, picks 64 (~768 MiB) -- a ~383
    // MiB shortfall against the charged term, bigger than the 256 MiB
    // margin. The belt call site clamps its OWN requested chunk to
    // `min(prefill_chunk_, packed_values_scratch_chunk_)` so the served
    // chunk can never exceed the priced one.
    int                            packed_values_scratch_chunk_ = 0;
    // 0015 engine side: whether the compiled plugin accepted
    // PAGED_ATTENTION_MAX_PARTITIONS (probed once, before compile, well
    // before this -- see the compile-time props block's own comment). Set
    // only on a 4-bit-values load; false on every other load and on any
    // 4-bit-values load against an unpatched plugin. Read again at the
    // fit-term climb and the belt call site to decide `max_partitions`
    // (cfg.paged_attention_max_partitions when accepted, 0/unbounded
    // otherwise) and `element_bytes` (2 when accepted -- the detection
    // contract stated at the probe site: a plugin exposing this key
    // carries the f16 host sizing too, both ship together in patch 0015 --
    // 4 otherwise).
    bool                           paged_attention_bound_accepted_ = false;
    // Patch 0020 engine side (DESIGN §7.0.2at): the GPU plugin's own patch
    // level, read from its build number (`marfrit-p<N>`, the recipe's
    // stamp -- the detection contract is stated at fit.h's
    // `kPackedValuesMixedStageMicroPatchLevel`), and whether this load's
    // key/value pairing runs its mixed prefill stage on micro-SDPA there
    // (u8 keys, 4-bit values, level >= 6). When true the fit charges no
    // scratch term and the belt applies only the measured chunk cap; read
    // at the climb, the ceiling and the belt call site, like the two
    // members above.
    int                            gpu_plugin_patch_level_ = 0;
    bool                           packed_values_mixed_stage_on_micro_ = false;
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
    // Whether the drafter loaded at all -- written once at startup (single-
    // threaded, before any request is served) and never on the request
    // path; std::atomic<bool> because it is read from every lane's thread on
    // every dflash call site (dflash_active() below) and a plain bool with
    // even an occasional cross-thread writer is a data race, not merely a
    // stale read. NOT the per-lane runtime disable, which lives on
    // Lane::dflash_off; dflash_active() below is the combination call sites
    // actually want.
    std::atomic<bool>       dflash_ready_{false};
    size_t                   dflash_block_     = 8;
    int                      dflash_mask_token_ = 0;
    size_t                   dflash_topk_      = 16;
    size_t                   dflash_rank_      = 0;
    size_t                   dflash_vocab_     = 0;
    size_t                   dflash_feat_width_ = 0;
    static constexpr size_t  kDflashWindow     = 2048;   // the head's sliding window
    // A lane failing on its own request is ordinary (a corrupted feature
    // buffer on one sequence, recoverable at the next reset -- see
    // Lane::dflash_off) and never disables anything beyond that lane.
    // kDflashWarnThreshold (8, roughly a handful of bad requests -- one bad
    // prompt can plausibly fail more than one lane in flight, so this
    // requires several before it is worth a look) logs a warning on the
    // failure that reaches it and on every further one until a draft
    // succeeds; nothing disables the drafter process-wide, and every lane
    // still recovers on its own next dflash_reset regardless of how high
    // this count climbs.
    static constexpr int     kDflashWarnThreshold = 8;
    std::atomic<int>         dflash_consec_failures_{0};
    std::vector<float>       dflash_proj_;               // [rank, hidden] f32
    std::vector<ov::float16> dflash_pred_cb_;            // [vocab, rank]
    std::vector<ov::float16> dflash_succ_cb_;
    ov::Tensor               dflash_mask_embed_;

    // M11 selector options (the M11 design note (not in the repository) V/K/lambda; the pure scoring
    // function lives in src/core/dflash_select.h so it can be unit-tested
    // without OpenVINO).
    DflashSelectMode         dflash_select_mode_ = DflashSelectMode::kGreedy;
    float                    dflash_lambda_       = 1.0f;

    // M11 dump instrument (the M11 design note (not in the repository) O): ARCINT_DFLASH_DUMP, read
    // once at construction, file opened once then (
    // "open the file once per process, not per cycle" -- simple here, since
    // the process never needs to change dump destinations mid-run). Every
    // check of "is the dump on" reads dflash_dump_enabled_ (an atomic bool)
    // rather than dflash_dump_path_.empty() -- the earlier version mutated
    // the string under dflash_dump_mutex_ on an open/write failure while
    // dflash_select and the decode loop read .empty() unlocked, a data race
    // (UB) on that path; dflash_dump_path_ is now write-once at construction
    // (kept only so the failure log line can name the file) and every
    // runtime check/disable goes through the atomic instead.
    // dflash_dump_cycle_ is a process-wide monotonic counter (lanes draft
    // concurrently) and dflash_dump_mutex_ serializes the actual file writes
    // so concurrent lanes cannot interleave two JSON lines into garbage.
    std::string              dflash_dump_path_;
    std::ofstream            dflash_dump_file_;
    std::atomic<bool>        dflash_dump_enabled_{false};
    std::atomic<uint64_t>    dflash_dump_cycle_{0};
    std::mutex               dflash_dump_mutex_;
    // Review follow-up: a per-request id the dump chains cycles
    // against (tools/dflash_oracle.py groups by lane+request, since two
    // requests on the same lane at different times are unrelated streams).
    // Process-wide and monotonic; assigned once per generate_paged call.
    std::atomic<uint64_t>    next_request_id_{0};

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
