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
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>

#include <openvino/openvino.hpp>

#include <minja/chat-template.hpp>

#include "core/artifact.h"
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

// -------------------------------------------------------------------- backend
class OvBackend final : public Backend {
public:
    OvBackend(const Artifact& artifact, Quant quant, int n_ctx, const std::string& device,
              const std::string& cache_dir)
        : artifact_(artifact) {
        core_.add_extension(tokenizers_extension_path());
        if (!cache_dir.empty()) core_.set_property(ov::cache_dir(cache_dir));

        tokenizer_ = std::make_unique<OvTokenizer>(core_, artifact, artifact.eos_ids);

        log::info("load", "compiling embeddings graph on %s", device.c_str());
        auto t0    = std::chrono::steady_clock::now();
        embeddings_ = core_.compile_model(artifact.text_embeddings_xml, device);
        log::info("load", "embeddings ready in %.1f s", seconds_since(t0));

        embed_req_ = embeddings_.create_infer_request();

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

            auto t_cached = std::chrono::steady_clock::now();
            try {
                language_    = core_.compile_model(artifact.language_model_xml, device, cfg);
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
            language_    = core_.compile_model(artifact.language_model_xml, device);
            logits_port_ = language_.output(0);
            lm_req_      = language_.create_infer_request();
            warmup();
            log::info("load", "language model ready in %.1f s (compiled from IR)",
                      seconds_since(t_cold));
        }

        template_ = std::make_unique<minja::chat_template>(artifact.chat_template,
                                                           artifact.bos_token, artifact.eos_token);

        status_.id               = artifact.id;
        status_.quant            = quant;
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
        const uint64_t seed = in.sampler.seeded ? in.sampler.seed : std::random_device{}();
        Sampler        sampler(in.sampler, seed);
        if (!in.sampler.greedy()) {
            log::verbose("sample", "temp %.2f top_p %.2f top_k %d seed %llu",
                         in.sampler.temperature, in.sampler.top_p, in.sampler.top_k,
                         static_cast<unsigned long long>(seed));
        }

        // History feeds the penalties; the prompt counts, as it does upstream.
        std::vector<int> history = prompt_ids;

        lm_req_.reset_state();

        // -------------------------------------------------------- prefill
        const auto t_prefill = clock::now();
        ov::Tensor logits    = forward(prompt_ids, /*past=*/0);
        stats.prefill_seconds = std::chrono::duration<double>(clock::now() - t_prefill).count();

        int next = pick(sampler, logits, history);

        const std::vector<int>& eos = tokenizer_->eos_ids();
        auto is_eos = [&](int id) {
            return !in.sampler.ignore_eos &&
                   std::find(eos.begin(), eos.end(), id) != eos.end();
        };

        // --------------------------------------------------------- decode
        const auto   t_decode = clock::now();
        FinishReason reason   = FinishReason::Stop;
        size_t       past     = prompt_ids.size();

        while (true) {
            if (is_eos(next)) break;
            if (in.sampler.max_tokens >= 0 && stats.completion_tokens >= in.sampler.max_tokens) {
                reason = FinishReason::Length;
                break;
            }
            if (status_.n_ctx > 0 &&
                static_cast<int>(past) + 1 >= status_.n_ctx) {
                reason = FinishReason::Length;
                break;
            }

            ++stats.completion_tokens;
            const Control c = on_piece(tokenizer_->decode_one(next), next);
            if (c == Control::Stop) break;
            if (c == Control::Cancel) {
                reason = FinishReason::Abort;
                break;
            }

            history.push_back(next);
            logits = forward({next}, past);
            ++past;
            next = pick(sampler, logits, history);
        }

        stats.decode_seconds = std::chrono::duration<double>(clock::now() - t_decode).count();
        return reason;
    }

private:
    // One real forward pass. This is what turns a latent import fault into a
    // startup failure instead of a run of 500s.
    void warmup() {
        forward({0}, 0);
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
        const ov::Tensor embeds = embed_req_.get_output_tensor(0);

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
    int pick(Sampler& sampler, const ov::Tensor& logits, const std::vector<int>& history) {
        const ov::Shape& shape = logits.get_shape();
        const size_t     vocab = shape.back();
        const size_t     rows  = logits.get_size() / vocab;
        const float*     row   = logits.data<const float>() + (rows - 1) * vocab;

        logit_scratch_.assign(row, row + vocab);
        return sampler.sample(logit_scratch_.data(), vocab, history);
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
    mutable size_t                 position_sections_ = 0;
};

}  // namespace

std::unique_ptr<Backend> make_ov_backend(const Artifact& artifact, Quant quant, int n_ctx,
                                         const std::string& device,
                                         const std::string& cache_dir) {
    return std::make_unique<OvBackend>(artifact, quant, n_ctx, device, cache_dir);
}

}  // namespace lgc
