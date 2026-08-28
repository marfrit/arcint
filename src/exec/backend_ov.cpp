// The M1 executor: OpenVINO owns the math, ligence owns everything else
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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>

#include <openvino/openvino.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/assign.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/core/graph_util.hpp>

#include <minja/chat-template.hpp>

#include "config.h"
#include "core/artifact.h"
#include "core/drafter.h"
#include "core/prefix_cache.h"
#include "core/sampler.h"
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
constexpr uint64_t kPrefixCacheKey = 0x6c6967656e636531ull;  // "ligence1"

std::string tokenizers_extension_path() {
    // Shipped alongside the OpenVINO wheel. The path is configuration, not a
    // guess: LIGENCE_TOKENIZERS_SO overrides it for an unusual install.
    if (const char* env = std::getenv("LIGENCE_TOKENIZERS_SO")) return env;
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

bool slice_logits_to_last_token(const std::shared_ptr<ov::Model>& model,
                                int64_t keep_rows) {
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
    const int64_t axis = ps.rank().get_length() - 2;  // [..., tokens, hidden]

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
    OvBackend(const Artifact& artifact, const Config& cfg, int n_ctx)
        : artifact_(artifact), prefill_chunk_(cfg.prefill_chunk) {
        const std::string& device    = cfg.device;
        std::string cache_dir = cfg.cache_dir;
        if (cfg.draft_tokens > 0) {
            drafter_     = std::make_unique<NgramDrafter>(static_cast<size_t>(cfg.draft_ngram),
                                                          static_cast<size_t>(cfg.draft_tokens));
            draft_tokens_ = static_cast<size_t>(cfg.draft_tokens);
        }
        if (cfg.prefix_cache_mib > 0) {
            prefix_cache_ = std::make_unique<PrefixCache>(
                static_cast<size_t>(cfg.prefix_cache_mib) * 1024 * 1024, cfg.kv_block_size,
                kPrefixCacheKey);
        }
        core_.add_extension(tokenizers_extension_path());
        if (!cache_dir.empty()) core_.set_property(ov::cache_dir(cache_dir));

        tokenizer_ = std::make_unique<OvTokenizer>(core_, artifact, artifact.eos_ids);

        log::info("load", "compiling embeddings graph on %s", device.c_str());
        auto t0    = std::chrono::steady_clock::now();
        embeddings_ = core_.compile_model(artifact.text_embeddings_xml, device);
        log::info("load", "embeddings ready in %.1f s", seconds_since(t0));

        embed_req_ = embeddings_.create_infer_request();

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

        if (cfg.slice_logits) {
            const int64_t keep = static_cast<int64_t>(1 + draft_tokens_);
            if (slice_logits_to_last_token(model, keep)) {
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
            if (std::getenv("LIGENCE_PROFILE") != nullptr) cfg[ov::enable_profiling.name()] = true;

            auto t_cached = std::chrono::steady_clock::now();
            try {
                language_    = core_.compile_model(model, device, cfg);
                logits_port_ = language_.output(0);
                lm_req_      = language_.create_infer_request();
                warmup();
                ready = true;
                log::info("load", "language model ready in %.1f s (blob cache, warmup passed)",
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
            if (std::getenv("LIGENCE_PROFILE") != nullptr) props[ov::enable_profiling.name()] = true;
            language_    = core_.compile_model(model, device, props);
            logits_port_ = language_.output(0);
            lm_req_      = language_.create_infer_request();
            warmup();
            log::info("load", "language model ready in %.1f s (compiled from IR)",
                      seconds_since(t_cold));
        }

        template_ = std::make_unique<minja::chat_template>(artifact.chat_template,
                                                           artifact.bos_token, artifact.eos_token);

        status_.id               = artifact.id;
        status_.quant            = cfg.quant;
        status_.loaded           = true;
        status_.stub             = false;
        status_.n_ctx            = n_ctx > 0 ? n_ctx : artifact.n_ctx_train;
        status_.n_layer          = artifact.n_layer;
        status_.n_gdn_layer      = artifact.n_gdn_layer;
        status_.n_attn_layer     = artifact.n_attn_layer;
        status_.mtp_enabled      = false;  // M4
        status_.weights_bytes    = artifact.weights_bytes;
        status_.sampler_defaults = artifact.sampler;
    }

    const ModelStatus& status() const override { return status_; }
    Tokenizer&         tokenizer() override { return *tokenizer_; }

    std::string render_chat(const ChatRequest& req) const override {
        minja::chat_template_inputs inputs;
        inputs.messages            = messages_json(req);
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

    FinishReason generate(const GenerationInput& in, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        std::lock_guard<std::mutex> guard(mutex_);
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
        if (drafter_ != nullptr) history = prompt_ids;

        // -------------------------------------------------------- prefill
        const auto t_prefill = clock::now();

        size_t past = 0;
        if (prefix_cache_ != nullptr) {
            const PrefixCache::Hit hit = prefix_cache_->lookup(prompt_ids);
            if (hit.matched_tokens > 0 && restore(*hit.state)) {
                past                   = hit.matched_tokens;
                stats.cache_hit_tokens = static_cast<int>(past);
            } else {
                lm_req_.reset_state();
            }
        } else {
            lm_req_.reset_state();
        }
        if (past == 0) lm_req_.reset_state();

        // The boundary worth remembering is the last block edge strictly inside
        // the prompt: a snapshot of the whole prompt is useless to this prompt
        // and only helps a continuation of it, which the same insert covers.
        const size_t block = prefix_cache_ != nullptr
                                 ? static_cast<size_t>(prefix_cache_->block_size())
                                 : 0;
        const size_t snap_at =
            block > 0 && prompt_ids.size() > block ? ((prompt_ids.size() - 1) / block) * block : 0;

        ov::Tensor logits = prefill(prompt_ids, past, snap_at);
        stats.prefill_seconds = std::chrono::duration<double>(clock::now() - t_prefill).count();

        int next = pick(sampler, logits);

        const std::vector<int>& eos = tokenizer_->eos_ids();
        auto is_eos = [&](int id) {
            return !in.sampler.ignore_eos &&
                   std::find(eos.begin(), eos.end(), id) != eos.end();
        };

        // --------------------------------------------------------- decode
        const auto   t_decode = clock::now();
        FinishReason reason   = FinishReason::Stop;
        past                  = prompt_ids.size();

        // Emits one committed token; returns false when the caller wants to stop.
        auto commit = [&](int tok, Control& out) {
            ++stats.completion_tokens;
            out = on_piece(tokenizer_->decode_one(tok), tok);
            sampler.observe(tok);
            if (drafter_ != nullptr) history.push_back(tok);
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
            if (drafter_ != nullptr && in.sampler.greedy()) {
                drafts = drafter_->draft(history, draft_tokens_);
            }

            if (drafts.empty()) {
                logits = forward({next}, past);
                ++past;
                next = pick(sampler, logits);
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
                drafter_.reset();
                logits = forward({next}, past);
                ++past;
                next = pick(sampler, logits);
                continue;
            }

            std::vector<int> seq;
            seq.reserve(1 + drafts.size());
            seq.push_back(next);
            seq.insert(seq.end(), drafts.begin(), drafts.end());

            const auto t_verify = clock::now();
            logits              = forward(seq, past);
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
                drafter_.reset();
                restore_tensors(rollback_);
                logits = forward({next}, past);
                ++past;
                next = pick(sampler, logits);
                continue;
            }

            // Accept and commit in one pass, in the same order the plain path
            // uses. Committing as we go is what makes the penalty state correct
            // for the next verification: draft i+1 must be judged against a
            // sampler that has already seen draft i.
            size_t accepted = 0;
            bool   stop     = false;
            for (size_t i = 0; i < drafts.size(); ++i) {
                if (pick_row(sampler, logits, i) != drafts[i]) break;
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
                    static_cast<int>(past + i) + 1 >= status_.n_ctx) {
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
            if (stop) break;

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
                logits          = forward(keep, past);
                stats.draft_reforward_seconds +=
                    std::chrono::duration<double>(clock::now() - t_re).count();
            }

            past += 1 + accepted;
            next = pick(sampler, logits);
        }

        stats.decode_seconds = std::chrono::duration<double>(clock::now() - t_decode).count();
        return reason;
    }

private:
    // Chunked prefill (§3.2). Verified byte-identical to a single call at chunk
    // sizes 32/64/128 on the artifact, which is what makes it safe to use by
    // default rather than only for prompts that would otherwise not fit.
    ov::Tensor prefill(const std::vector<int>& tokens, size_t past, size_t snapshot_at) {
        ov::Tensor   logits;
        const size_t chunk =
            prefill_chunk_ > 0 ? static_cast<size_t>(prefill_chunk_) : tokens.size();

        while (past < tokens.size()) {
            size_t take = std::min(std::max<size_t>(chunk, 1), tokens.size() - past);

            // Stop exactly on the snapshot boundary so the checkpoint lands on a
            // block edge instead of wherever the chunking happened to land.
            if (snapshot_at > past && snapshot_at < past + take) take = snapshot_at - past;

            logits = forward({tokens.begin() + static_cast<long>(past),
                              tokens.begin() + static_cast<long>(past + take)},
                             past);
            past += take;

            if (prefix_cache_ != nullptr && past == snapshot_at) {
                // Serialising the whole graph state is expensive and the KV half
                // grows with the prefix, so ask first whether it could be kept
                // at all rather than copying hundreds of MiB to have it dropped.
                if (prefix_cache_->may_accept(estimated_state_bytes())) {
                    PrefixCache::StateBlob blob;
                    if (snapshot(blob)) prefix_cache_->insert(tokens, past, std::move(blob));
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

    // Mirrors restore()'s contract: a failure degrades to "no caching", never
    // to a 500 on an otherwise healthy request.
    bool snapshot(PrefixCache::StateBlob& out) {
        try {
            out.clear();
            for (ov::VariableState& v : lm_req_.query_state()) {
                out.push_back(serialise_state(v.get_state()));
            }
            return true;
        } catch (const std::exception& e) {
            log::warn("cache", "state snapshot failed, continuing without caching: %s", e.what());
            out.clear();
            return false;
        }
    }

    size_t estimated_state_bytes() {
        if (state_bytes_ != 0) return state_bytes_;
        try {
            for (ov::VariableState& v : lm_req_.query_state()) {
                state_bytes_ += v.get_state().get_byte_size() + 64;
            }
        } catch (const std::exception&) {
            state_bytes_ = 0;
        }
        return state_bytes_;
    }

    bool restore(const PrefixCache::StateBlob& blob) {
        std::vector<ov::VariableState> states = lm_req_.query_state();
        if (states.size() != blob.size()) return false;
        try {
            for (size_t i = 0; i < states.size(); ++i) {
                states[i].set_state(deserialise_state(blob[i]));
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
        forward({0}, 0);
        lm_req_.reset_state();
        if (std::getenv("LIGENCE_BENCH_FORWARD") != nullptr) bench_forward();
        if (std::getenv("LIGENCE_PROFILE") != nullptr) profile_step();
    }

    // Per-kernel breakdown of one decode step. Aggregated by (op, kernel) so
    // the reference-kernel fallbacks stand out, which is what the numbers are
    // usually for. Needs PERF_COUNT at compile time (LIGENCE_PROFILE sets it).
    void profile_step() {
        lm_req_.reset_state();
        std::vector<int> warm(64, 0);
        forward(warm, 0);
        forward({0}, 64);

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
        const size_t past = 512;
        std::vector<int> warm(past, 0);
        double one = 0.0;
        for (size_t k : {size_t{1}, size_t{2}, size_t{3}, size_t{5}, size_t{9},
                         size_t{17}, size_t{33}, size_t{65}}) {
            lm_req_.reset_state();
            forward(warm, 0);
            std::vector<int> ids(k, 0);
            const auto t0 = std::chrono::steady_clock::now();
            const int reps = 5;
            for (int r = 0; r < reps; ++r) forward(ids, past + k * static_cast<size_t>(r));
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

    static json messages_json(const ChatRequest& req) {
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
                                      {{"name", c.name}, {"arguments", c.arguments}}}});
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
    ov::Tensor forward(const std::vector<int>& ids, size_t past) {
        const size_t n     = ids.size();
        const size_t total = past + n;

        ov::Tensor id_tensor(ov::element::i64, ov::Shape{1, n});
        int64_t*   idp = id_tensor.data<int64_t>();
        for (size_t i = 0; i < n; ++i) idp[i] = ids[i];

        embed_req_.set_input_tensor(id_tensor);
        embed_req_.infer();

        // Copy the embeddings out instead of handing the language model the
        // embeddings request's own output buffer. Sharing it made chunked
        // prefill diverge from unchunked: two requests aliasing one tensor is
        // not something either plugin promises to serialise, and the
        // equivalence suite caught it as a byte difference rather than a crash.
        const ov::Tensor src = embed_req_.get_output_tensor(0);
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
        // artifact on dirac, where a first-section-only layout produces
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
        lm_req_.infer();

        return lm_req_.get_tensor(logits_port_);
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
    int pick_row(Sampler& sampler, const ov::Tensor& logits, size_t row) {
        const size_t vocab = logits.get_shape().back();
        const size_t rows  = logits.get_size() / vocab;
        // Out of range is a graph/plumbing bug, not a draft miss. Returning -1
        // guarantees rejection (token ids are non-negative) so output stays
        // correct, and the caller reports it instead of silently accepting 0%.
        if (row >= rows) return -1;

        const float* p = logits.data<const float>() + row * vocab;
        logit_scratch_.assign(p, p + vocab);
        return sampler.sample(logit_scratch_.data(), vocab);
    }

    int pick(Sampler& sampler, const ov::Tensor& logits) {
        const size_t vocab = logits.get_shape().back();
        return pick_row(sampler, logits, logits.get_size() / vocab - 1);
    }

    Artifact                       artifact_;
    ov::Core                       core_;
    ov::CompiledModel              embeddings_;
    ov::CompiledModel              language_;
    ov::InferRequest               embed_req_;
    ov::InferRequest               lm_req_;
    ov::Output<const ov::Node>     logits_port_;
    std::unique_ptr<OvTokenizer>   tokenizer_;
    std::unique_ptr<minja::chat_template> template_;
    ModelStatus                    status_;
    std::mutex                     mutex_;
    std::vector<float>             logit_scratch_;
    ov::Tensor                     embeds_;
    int                            prefill_chunk_ = 512;
    size_t                         state_bytes_   = 0;
    std::unique_ptr<PrefixCache>   prefix_cache_;
    std::vector<ov::Tensor>        rollback_;   // reused speculation scratch
    bool                           logged_rollback_size_ = false;
    std::unique_ptr<Drafter>       drafter_;
    size_t                         draft_tokens_ = 0;
    mutable size_t                 position_sections_ = 0;
};

}  // namespace

std::unique_ptr<Backend> make_ov_backend(const Artifact& artifact, const Config& cfg, int n_ctx) {
    return std::make_unique<OvBackend>(artifact, cfg, n_ctx);
}

}  // namespace lgc
