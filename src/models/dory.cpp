#include "models.h"
#include "llama-placement-report.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

void llama_model_dory::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_INPUT_LAYER_COUNT, n_input);
    ml.get_key(LLM_KV_RECURRENT_LAYER_COUNT, n_recurrent);
    ml.get_key(LLM_KV_OUTPUT_LAYER_COUNT, n_output);
    ml.get_key(LLM_KV_NUM_LOOPS, n_loops);
    ml.get_key(LLM_KV_RECURRENT_KV_CACHE_MODE, recurrent_kv_cache_mode, false);
    if (recurrent_kv_cache_mode != "per_loop" && recurrent_kv_cache_mode != "last_loop") {
        throw std::runtime_error("Dory recurrent_kv_cache_mode must be per_loop or last_loop");
    }
    ml.get_key(LLM_KV_LAYER_PATTERN, pattern);
    uint32_t n_profile = 0;
    ml.get_arr_n(LLM_KV_ROPE_PROFILE, n_profile);
    if (n_profile > LLAMA_MAX_LAYERS) {
        throw std::runtime_error("Dory RoPE profile is too large");
    }
    ml.get_arr(LLM_KV_ROPE_PROFILE, profile);
    ml.get_key(LLM_KV_INIT_STD, init_std);
    ml.get_key(LLM_KV_ALPHA_INIT, alpha_init);
    ml.get_key(LLM_KV_GATE_INIT, gate_init);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, theta_1);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_2, theta_2);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    const uint64_t n_physical = uint64_t(n_input) + n_recurrent + n_output;
    const uint64_t n_virtual = uint64_t(n_input) + uint64_t(n_recurrent)*n_loops + n_output;
    if (!n_input || !n_recurrent || !n_output || !n_loops || n_loops > 4 ||
            n_virtual > LLAMA_MAX_LAYERS || n_physical != hparams.n_layer_all ||
            pattern.size() != n_physical || profile.size() != n_physical ||
            pattern.find_first_not_of("S*-") != std::string::npos ||
            !(init_std > 0 && std::isfinite(init_std)) ||
            !(alpha_init > 0 && std::isfinite(alpha_init)) ||
            !(gate_init > 0 && std::isfinite(gate_init)) ||
            !(theta_1 > 0 && std::isfinite(theta_1)) || !(theta_2 > 0 && std::isfinite(theta_2)) ||
            !hparams.n_swa || pattern.find('S') == std::string::npos || pattern.find('*') == std::string::npos) {
        throw std::runtime_error("unsupported or invalid Dory recurrence metadata");
    }
    const auto nh = hparams.n_head(0);
    const auto nk = hparams.n_head_kv(0);
    if (!nh || !nk || nh % nk || !hparams.n_embd_head_k() ||
            hparams.n_embd_head_k() != hparams.n_embd_head_v() || hparams.n_rot() != hparams.n_embd_head_k()) {
        throw std::runtime_error("invalid Dory attention dimensions");
    }
    const auto nf = hparams.n_ff(0);
    if (!nf) {
        throw std::runtime_error("invalid Dory feed-forward dimension");
    }
    for (auto p : profile) {
        if (p < 1 || p > 2) {
            throw std::runtime_error("invalid Dory RoPE profile");
        }
    }
    for (uint32_t i = 0; i < n_input; ++i) {
        physical.push_back(i);
    }
    for (uint32_t k = 0; k < n_loops; ++k) {
        for (uint32_t i = 0; i < n_recurrent; ++i) {
            physical.push_back(n_input + i);
        }
    }
    for (uint32_t i = 0; i < n_output; ++i) {
        physical.push_back(n_input + n_recurrent + i);
    }
    hparams.n_layer_all = n_virtual;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_freq_base_train_swa = theta_2;
    for (uint32_t il = 0; il < n_virtual; ++il) {
        const uint32_t p = physical[il];
        const bool recurrent = p >= n_input && p < n_input + n_recurrent;
        cache_layer.push_back(recurrent && recurrent_kv_cache_mode == "last_loop" ? p : il);
        hparams.n_head_arr[il] = nh;
        hparams.n_head_kv_arr[il] = pattern[p] == '-' ? 0 : nk;
        hparams.n_ff_arr[il] = nf;
        hparams.is_swa_impl[il] = pattern[p] == 'S';
        hparams.rope_pattern[il] = profile[p] != 0 && pattern[p] != '-';
    }
    type = LLM_TYPE_UNKNOWN;
}

// ARC-LAB 0113: LLAMA_DORY_FOLD, read once a load: 0 the PR's graph; 1 the folded vectors only (the
// same values on the CPU, bit for bit); 2 (the default) also no second norm of a residual that is
// already a norm's output. Anything else warns and serves 2.
static int dory_fold_level() {
    const char * env = getenv("LLAMA_DORY_FOLD");
    if (env == nullptr) {
        return 2;
    }
    if (strcmp(env, "0") == 0 || strcmp(env, "1") == 0 || strcmp(env, "2") == 0) {
        return env[0] - '0';
    }
    LLAMA_LOG_WARN("%s: LLAMA_DORY_FOLD=%s is not 0, 1 or 2; 2 serves\n", __func__, env);
    return 2;
}

void llama_model_dory::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;
    fold_level = dory_fold_level();
    // folded, the graph never reads the BF16 sources: they stay in host memory, unprobed (0113)
    const int src_flags = fold_level >= 1 ? llama_model_loader::TENSOR_HOST_ONLY : 0;
    // the generic pass after this function loads optional per-tensor ".scale" tensors by EXECUTED layer
    // index, so a looped copy would pick up the scale of whichever physical block sits at that index
    // (the output blocks' for the recurrent copies). The published file has none; one that does is refused
    for (const auto & w : ml.weights_map) {
        const std::string & n = w.first;
        if (n.size() > 6 && n.compare(n.size() - 6, 6, ".scale") == 0) {
            throw std::runtime_error("Dory does not support per-tensor .scale tensors (" + n + ")");
        }
    }
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);
    logit_scale = create_tensor(tn(LLM_TENSOR_OUTPUT_SCALE, "weight"), {n_vocab}, src_flags);
    scale.resize(n_layer);
    std::vector<int> first(pattern.size(), -1);
    for (int il = 0; il < n_layer; ++il) {
        const int p = physical[il];
        if (first[p] >= 0) {
            layers[il] = layers[first[p]];
            scale[il] = scale[first[p]];
            continue;
        }
        first[p] = il;
        auto & l = layers[il];
        auto & s = scale[il];
        s.alpha = create_tensor(tn(LLM_TENSOR_RESIDUAL_ALPHA, "weight", p), {n_embd}, src_flags);
        if (pattern[p] == '-') {
            l.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", p), {n_embd, n_ff}, 0);
            l.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", p), {n_embd, n_ff}, 0);
            l.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", p), {n_ff, n_embd}, 0);
            s.suv_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE_SCALE, "weight", p), {n_ff}, src_flags);
            s.suv_up = create_tensor(tn(LLM_TENSOR_FFN_UP_SCALE, "weight", p), {n_ff}, src_flags);
        } else {
            const int64_t nq = n_head * n_embd_head_k;
            const int64_t nkv = hparams.n_head_kv(il) * n_embd_head_k;
            l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", p), {n_embd, nq}, 0);
            l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", p), {n_embd, nkv}, 0);
            l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", p), {n_embd, nkv}, 0);
            l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", p), {nq, n_embd}, 0);
            l.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", p), {n_embd, nq}, 0);
            s.sqk = create_tensor(tn(LLM_TENSOR_ATTN_QK_SCALE, "weight", p), {nq}, src_flags);
            s.gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE_SCALE, "weight", p), {nq}, src_flags);
        }
    }
}

// ARC-LAB 0113: the scale vectors' constant chains folded at load. The PR's graph recomputes each one
// from the BF16 weight on every forward (a cast, a scale and an abs or a square a vector, 3-6 nodes a
// sublayer); here each final per-channel vector is computed once, in F32, from the loaded source (kept
// in host memory, TENSOR_HOST_ONLY: its exact BF16 values), with the graph's operations in the graph's
// order, so the CPU backend sees the same values bit for bit:
//   alpha  |alpha * (alpha_init / init_std)|                         [n_embd]
//   sq     (sqk * (sqrt(hd) / init_std))^2 [* (1 / sqrt(hd))]         [hd, n_head]  (the q norm's
//          1/sqrt(hd) moved into it only when sqrt(hd) is a power of two, where the product is exact)
//   sg     (s_gate * (sqrt(gate_init) / init_std))^2                  [n_q]
//   cg/cu  suv_gate/suv_up * sqrt(n_embd)                             [n_ff]
//   logit  logit_scale * (1 / init_std)                               [n_vocab]
// Each fold is placed by its layer's device, probed as the loader probes a weight (an F32 MUL), and a
// fold the device declines into host memory is reported as 0111 reports a weight. The memory-fit
// pass (no_alloc) builds them too, sizes only, so the fit counts them and plans the folded graph.
void llama_model_dory::load_tensors_post(llama_model_loader & ml) {
    if (fold_level <= 0) {
        LLAMA_LOG_INFO("%s: Dory scale chains computed in the graph (LLAMA_DORY_FOLD=0)\n", __func__);
        return;
    }
    const int n_layer = hparams.n_layer();
    const int64_t n_embd = hparams.n_embd;
    const int64_t hd = hparams.n_embd_head_k();
    const float rhd = sqrtf((float) hd);
    const bool qnorm_in_sq = rhd == floorf(rhd) && rhd >= 1.0f && (((int64_t) rhd) & (((int64_t) rhd) - 1)) == 0;

    enum kind { K_ALPHA, K_SQ, K_SG, K_SUV, K_LOGIT };
    struct item { const ggml_tensor * src; ggml_tensor ** dst; kind k; int il; };
    std::vector<item> items;
    std::vector<folded_scales> f(n_layer);
    std::vector<int> first(n_layer_physical(), -1);
    for (int il = 0; il < n_layer; ++il) {
        const int p = physical[il];
        if (first[p] >= 0) {
            continue;
        }
        first[p] = il;
        const auto & s = scale[il];
        items.push_back({ s.alpha, &f[il].alpha, K_ALPHA, il });
        if (pattern[p] == '-') {
            items.push_back({ s.suv_gate, &f[il].cg, K_SUV, il });
            items.push_back({ s.suv_up,   &f[il].cu, K_SUV, il });
        } else {
            items.push_back({ s.sqk,  &f[il].sq, K_SQ, il });
            items.push_back({ s.gate, &f[il].sg, K_SG, il });
        }
    }
    ggml_tensor * logit = nullptr;
    items.push_back({ logit_scale, &logit, K_LOGIT, -1 });

    // the layer's buffer type for an F32 MUL operand; a CPU extra type (CPU_REPACK) only takes what it
    // can repack, so a CPU choice is the CPU's own buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctxs;
    for (auto & it : items) {
        if (it.src == nullptr) {
            throw std::runtime_error("Dory: a scale vector is missing");
        }
        const int64_t n = ggml_nelements(it.src);
        ggml_backend_buffer_type_t asked = nullptr;
        ggml_backend_buffer_type_t buft = select_buft_f32_mul(it.il, n, &asked);
        if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft)) {
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                buft = ggml_backend_dev_buffer_type(dev);
            }
        }
        ggml_context *& ctx = ctxs[buft];
        if (ctx == nullptr) {
            ggml_init_params params = { ggml_tensor_overhead() * (items.size() + 1), nullptr, true };
            ctx = ggml_init(params);
            if (ctx == nullptr) {
                throw std::runtime_error("Dory: failed to create the folded-scale context");
            }
        }
        ggml_tensor * t = it.k == K_SQ ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, n / hd)
                                       : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_format_name(t, "%s.fold", ggml_get_name(it.src));
        *it.dst = t;
        // 0111's report for what the graph reads: the layer asked for device memory, its device
        // declined the fold's MUL, and the fold went to host memory
        if (!ml.no_alloc && llama_placement::declined_to_host(asked, buft)) {
            const std::string k = llama_placement::key("MUL", "f32", ggml_get_name(t), ggml_backend_buft_name(buft));
            if (llama_placement::global().add(k)) {
                LLAMA_LOG_WARN("DOWNGRADE src=llama %s\n", k.c_str());
            }
        }
    }
    size_t bytes = 0;
    const char * failed = nullptr;
    for (auto & [buft, ctx] : ctxs) {
        ggml_backend_buffer_t buf = nullptr;
        if (!failed) {
            if (ml.no_alloc) {
                // the loader's dummy for a fit pass: the scheduler leaves the tensors alone, and
                // memory_breakdown sizes the context
                buf = ggml_backend_buft_alloc_buffer(buft, 0);
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); buf != nullptr && t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf;
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            }
        }
        if (buf == nullptr) {
            failed = failed ? failed : ggml_backend_buft_name(buft);
            ggml_free(ctx);   // this one and every later one: the model owns only the allocated ones
            continue;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bytes += ml.no_alloc ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft) : ggml_backend_buffer_get_size(buf);
        own_buffer(ctx, buf);
    }
    if (failed) {
        throw std::runtime_error(format("Dory: unable to allocate the folded scales in %s", failed));
    }

    if (!ml.no_alloc) {
        // the source values as F32, read from the loaded tensor (host memory: the file's BF16 values)
        std::vector<uint8_t> raw;
        std::vector<float> v, out;
        // the factors as the graph spells them (float arithmetic, the same expressions)
        const float c_alpha = alpha_init / init_std;
        const float c_sqk   = sqrtf(hd) / init_std;
        const float c_qnorm = 1.0f/sqrtf((float) hd);
        const float c_gate  = sqrtf(gate_init) / init_std;
        const float c_suv   = sqrtf(n_embd);
        const float c_logit = 1.0f / init_std;
        for (auto & it : items) {
            const int64_t n = ggml_nelements(it.src);
            raw.resize(ggml_nbytes(it.src));
            ggml_backend_tensor_get(it.src, raw.data(), 0, raw.size());
            v.resize(n);
            switch (it.src->type) {
                case GGML_TYPE_F32:  memcpy(v.data(), raw.data(), n * sizeof(float)); break;
                case GGML_TYPE_F16:  ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), v.data(), n); break;
                case GGML_TYPE_BF16: ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), v.data(), n); break;
                default: throw std::runtime_error(format("Dory: scale vector %s has type %s", ggml_get_name(it.src), ggml_type_name(it.src->type)));
            }
            out.resize(n);
            for (int64_t i = 0; i < n; ++i) {
                float x = v[i];
                switch (it.k) {
                    case K_ALPHA: x = fabsf(x * c_alpha); break;
                    case K_SQ:    x = x * c_sqk; x = x * x; if (qnorm_in_sq) { x = x * c_qnorm; } break;
                    case K_SG:    x = x * c_gate; x = x * x; break;
                    case K_SUV:   x = x * c_suv; break;
                    case K_LOGIT: x = x * c_logit; break;
                }
                out[i] = x;
            }
            ggml_backend_tensor_set(*it.dst, out.data(), 0, out.size() * sizeof(float));
        }
    }
    for (int il = 0; il < n_layer; ++il) {
        f[il] = f[first[physical[il]]];
    }
    fold = std::move(f);
    logit_fold = logit;
    folded = true;
    renorm_dropped = fold_level >= 2;
    sq_has_qnorm = qnorm_in_sq;
    if (!ml.no_alloc) {
        std::string where;
        for (auto & [buft, ctx] : ctxs) {
            where += (where.empty() ? "" : ", ") + std::string(ggml_backend_buft_name(buft));
        }
        LLAMA_LOG_INFO("%s: Dory: %zu scale vectors folded at load, %.2f MiB F32 in %s, residual renorm %s (LLAMA_DORY_FOLD=%d; 0 keeps the graph's chains)\n",
                       __func__, items.size(), bytes / (1024.0 * 1024.0), where.c_str(), renorm_dropped ? "dropped" : "kept", fold_level);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dory::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_dory::graph::graph(const llama_model_dory & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    ggml_tensor * pos = build_inp_pos();
    ggml_tensor * out_ids = build_inp_out_ids();
    auto * attn = build_attn_inp_kv_iswa();
    ggml_tensor * input_h = nullptr;
    const int recurrent_end = model.n_input + model.n_recurrent * model.n_loops;
    const auto norm = [&](ggml_tensor * x) { return build_gdn_l2_norm(ctx0, x, 1e-6f); };
    const auto f32 = [&](ggml_tensor * x) { return x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx0, x, GGML_TYPE_F32); };
    // ARC-LAB 0113 (a deviation from the PR's graph, at the answer-level bar): the scale chains come
    // folded from the load (load_tensors_post) and the SwiGLU is one GLU node, which on the CPU gives
    // the PR graph's values bit for bit at Dory's head size 256; at LLAMA_DORY_FOLD=2 also a residual that is already a norm's
    // output is not normalised again (every one but the embedding: the previous slerp's or the loop
    // seed's), nor is a loop's last state (the last slerp's output), which moves rounding only
    const bool folded = model.folded;
    const bool renorm = !model.renorm_dropped;

    for (int il = 0; il < n_layer; ++il) {
        const int p = model.physical[il];
        const auto & l = model.layers[il];
        const auto & s = model.scale[il];
        const auto * fs = folded ? &model.fold[il] : nullptr;
        if (il == (int) model.n_input) {
            input_h = cur;
        }
        if (il >= (int) model.n_input && il < recurrent_end && (il - model.n_input) % model.n_recurrent == 0) {
            cur = norm(ggml_add(ctx0, cur, input_h));
        }
        ggml_tensor * residual = cur;
        if (model.pattern[p] == '-') {
            auto * g = build_lora_mm(l.ffn_gate, cur);
            auto * u = build_lora_mm(l.ffn_up, cur);
            if (fs) {
                cur = norm(ggml_swiglu_split(ctx0, ggml_mul(ctx0, g, fs->cg), ggml_mul(ctx0, u, fs->cu)));
            } else {
                g = ggml_mul(ctx0, g, ggml_scale(ctx0, f32(s.suv_gate), sqrtf(n_embd)));
                u = ggml_mul(ctx0, u, ggml_scale(ctx0, f32(s.suv_up), sqrtf(n_embd)));
                cur = norm(ggml_mul(ctx0, ggml_silu(ctx0, g), u));
            }
            cur = build_lora_mm(l.ffn_down, cur);
        } else {
            const auto hd = hparams.n_embd_head_k(il);
            const auto nh = hparams.n_head(il);
            const auto nk = hparams.n_head_kv(il);
            auto [q, k, v] = build_qkv(l, cur, hd, nh, nk, il);
            if (model.profile[p] != 0) {
                const float theta = model.profile[p] == 2 ? model.theta_2 : model.theta_1;
                q = ggml_rope_ext(ctx0, q, pos, nullptr, hd, rope_type, n_ctx_orig, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                k = ggml_rope_ext(ctx0, k, pos, nullptr, hd, rope_type, n_ctx_orig, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            }
            if (fs && model.sq_has_qnorm) {
                // the norm's 1/sqrt(hd) is in sq (exact: sqrt(hd) a power of two): rms_norm then mul, the pair the backends fuse
                q = ggml_mul(ctx0, ggml_rms_norm(ctx0, q, 1e-6f / (float) q->ne[0]), fs->sq);
            } else if (fs) {
                q = ggml_mul(ctx0, norm(q), fs->sq);
            } else {
                auto * sq = ggml_sqr(ctx0, ggml_scale(ctx0, f32(s.sqk), sqrtf(hd) / model.init_std));
                q = ggml_mul(ctx0, norm(q), ggml_reshape_3d(ctx0, sq, hd, nh, 1));
            }
            k = norm(k);
            auto * gate = build_lora_mm(l.wqkv_gate, cur);
            if (fs) {
                gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, gate, fs->sg));
            } else {
                auto * sg = ggml_sqr(ctx0, ggml_scale(ctx0, f32(s.gate), sqrtf(model.gate_init) / model.init_std));
                gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, gate, sg));
            }
            // Repeated writes replace only the current chunk; older tokens retain final-loop KV.
            cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(hd), model.cache_layer[il]);
            cur = norm(ggml_mul(ctx0, cur, gate));
            cur = build_lora_mm(l.wo, cur);
        }
        auto * a = !renorm && il > 0 ? residual : norm(residual);
        auto * b = norm(cur);
        auto * alpha = fs ? fs->alpha : ggml_abs(ctx0, ggml_scale(ctx0, f32(s.alpha), model.alpha_init / model.init_std));
        // the nGPT update: RMS_NORM SCALE SUB MUL ADD RMS_NORM SCALE in this order is what OpenCL fuses (0114)
        cur = norm(ggml_add(ctx0, a, ggml_mul(ctx0, ggml_sub(ctx0, b, a), alpha)));
        if (renorm && il >= (int) model.n_input && il < recurrent_end && (il + 1 - model.n_input) % model.n_recurrent == 0) {
            cur = norm(cur);
        }
        cb(cur, "l_out", il);
    }
    if (out_ids) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    res->t_embd = cur;
    cb(cur, "result_norm", -1);
    cur = build_lora_mm(model.output, cur);
    cur = ggml_mul(ctx0, cur, folded ? model.logit_fold : ggml_scale(ctx0, f32(model.logit_scale), 1.0f / model.init_std));
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
    // Preserve F32 inputs through recurrent normalization; BF16 and quantized weights keep their normal kernels.
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F32) {
            ggml_prec_set_src(node, GGML_PREC_F32, 1);
        }
    }
}
