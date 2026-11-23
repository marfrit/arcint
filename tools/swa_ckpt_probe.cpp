// The window-sized SWA cache's checkpoint resume (--llama-swa-window), on the
// CPU: a partial state (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) must carry the
// sliding window, and a cut below the retained window must resume from a
// checkpoint instead of trimming.
//
// Needs a GGUF; not a ctest case (the unit gate is tests/test_swa_window.cpp for
// the rule and the flag). Build CPU only against the pinned tree:
//
//   g++ -O2 -std=c++17 -I include -I ggml/include -I src tools/swa_ckpt_probe.cpp
//       -o swa_ckpt_probe -L build/bin -lllama -Wl,-rpath,$PWD/build/bin
//
//   ./swa_ckpt_probe model.gguf [n_ctx]
//
// The arms:
//   A  append-only follow-up, no cut: must match a fresh full prefill
//   B  the last message edited below the window: the lane restores its
//      checkpoint (the rule in core/swa_window.h) and must match a fresh full
//      prefill at the answer (same top token, mean KL inside the CPU's
//      slot-order floor)
//   C  the same edit with the pre-change behaviour (trim only, no restore):
//      this arm is the red case, it must NOT match
// A partial state at a position above the window must be larger than one below
// it: without the SWA part both are the recurrent-only size (red the same way).
#include "llama.h"
#include "llama-ext.h"   // llama_memory_attn_seq_pos_min (contrib/llama.cpp 0153)
#include "core/swa_window.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Result {
    double kl       = 0.0;
    double dmax     = 0.0;
    int    top_ref  = 0;
    int    top_got  = 0;
    bool   same_top = false;
};

bool decode_chunk(llama_context* ctx, const std::vector<llama_token>& t, int from, int to, int p0, bool last) {
    llama_batch b = llama_batch_init(to - from, 0, 1);
    for (int i = from; i < to; ++i) {
        b.token[i - from]    = t[i];
        b.pos[i - from]      = p0 + i;
        b.n_seq_id[i - from] = 1;
        b.seq_id[i - from][0] = 0;
        b.logits[i - from]   = (last && i == to - 1) ? 1 : 0;
    }
    b.n_tokens = to - from;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

bool decode(llama_context* ctx, const std::vector<llama_token>& t, int p0, bool last, int chunk = 1024) {
    for (int i = 0; i < (int) t.size(); i += chunk) {
        const int to = std::min((int) t.size(), i + chunk);
        if (!decode_chunk(ctx, t, i, to, p0, last && to == (int) t.size())) return false;
    }
    return true;
}

std::vector<float> logits_of(llama_context* ctx) {
    const int n = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    std::vector<float> v(n);
    const float* l = llama_get_logits_ith(ctx, -1);   // the last row asked for
    std::copy(l, l + n, v.begin());
    return v;
}

Result compare(const std::vector<float>& ref, const std::vector<float>& got) {
    Result r;
    r.top_ref = 0;
    r.top_got = 0;
    for (int i = 0; i < (int) ref.size(); ++i) {
        if (ref[i] > ref[r.top_ref]) r.top_ref = i;
        if (got[i] > got[r.top_got]) r.top_got = i;
        r.dmax = std::max(r.dmax, (double) std::fabs(ref[i] - got[i]));
    }
    double za = 0.0, zb = 0.0;
    for (float x : ref) za += std::exp(x);
    for (float x : got) zb += std::exp(x);
    for (int i = 0; i < (int) ref.size(); ++i) {
        const double pa = std::exp(ref[i]) / za;
        const double pb = std::exp(got[i]) / zb;
        if (pa > 0.0) r.kl += pa * std::log(pa / (pb > 0.0 ? pb : 1e-300));
    }
    r.same_top = r.top_ref == r.top_got;
    return r;
}

llama_context* make_ctx(llama_model* m, int n_ctx, uint32_t n_ubatch = 512, uint32_t n_rs_seq = 0) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = (uint32_t) n_ctx;
    cp.n_batch    = 2048;
    cp.n_ubatch   = n_ubatch;
    cp.n_seq_max  = 1;
    cp.n_rs_seq   = n_rs_seq;
    cp.n_threads  = 4;
    cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cp.swa_full   = false;   // --llama-swa-window
    return llama_init_from_model(m, cp);
}

std::vector<uint8_t> save_partial(llama_context* ctx, int seq) {
    const size_t sz = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<uint8_t> out(sz);
    if (sz == 0) return out;
    if (llama_state_seq_get_data_ext(ctx, out.data(), sz, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != sz) out.clear();
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [n_ctx] [per_loop|last_loop]\n", argv[0]); return 2; }
    const int n_ctx = argc > 2 ? std::atoi(argv[2]) : 8192;
    const char* dory_kv = argc > 3 ? argv[3] : nullptr;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    // Dory's recurrent KV mode is a GGUF override (arcint's --llama-dory-kv);
    // the file's own is per_loop. last_loop aliases the passes' slots, and its
    // prefill is chunk-boundary dependent by design (upstream's own test says
    // so), which the report names where the KL is not zero.
    llama_model_kv_override overrides[2] = {};
    if (dory_kv != nullptr) {
        std::snprintf(overrides[0].key, sizeof(overrides[0].key), "%s", "dory.recurrent_kv_cache_mode");
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        std::snprintf(overrides[0].val_str, sizeof(overrides[0].val_str), "%s", dory_kv);
        mp.kv_overrides = overrides;   // overrides[1] is the zeroed terminator
    }
    llama_model* m = llama_model_load_from_file(argv[1], mp);
    if (m == nullptr) { std::fprintf(stderr, "load failed\n"); return 2; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const int n_swa   = llama_model_n_swa(m);
    const int n_layer = llama_model_n_layer(m);
    std::fprintf(stderr, "model: n_layer %d, n_swa %d, hybrid %d, recurrent %d, n_ctx %d%s%s\n",
                 n_layer, n_swa, (int) llama_model_is_hybrid(m), (int) llama_model_is_recurrent(m), n_ctx,
                 dory_kv ? ", kv " : "", dory_kv ? dory_kv : "");
    if (n_swa <= 0) {
        std::fprintf(stderr, "SKIP: the model has no sliding window; nothing to verify\n");
        llama_model_free(m);
        return 0;
    }

    // deterministic token ids, no tokenizer (the model's text is irrelevant here)
    auto tok = [&](int i) -> llama_token { return (llama_token) (1000 + (i * 7919) % (n_vocab - 2000)); };
    // sizes from the model's window: the SWA cache holds n_swa + n_ubatch cells
    // (n_ubatch 512, padded to 256), so the lane fills past that ring, the
    // checkpoint sits inside the ring and above the window, and the edited
    // prompt shares a prefix that ends a whole window after the cut, where the
    // ring can no longer reach the divergence (the restore arm)
    const int size_swa = (n_swa + 512 + 255) / 256 * 256;
    const int lane_end = std::min(size_swa + n_swa + 128, n_ctx - 64);
    const int edited   = std::min(lane_end - n_swa - 128, n_ctx - 64);
    const int cut      = std::min(n_swa + 128, edited);
    const int early    = std::min(n_swa / 2, cut - 64);   // below the window, for the size delta
    std::vector<llama_token> lane_tokens, edited_tokens, append_tokens;
    for (int i = 0; i < lane_end; ++i) lane_tokens.push_back(tok(i));
    for (int i = 0; i < edited; ++i) edited_tokens.push_back(tok(i));
    append_tokens = lane_tokens;
    for (int i = 0; i < 256; ++i) append_tokens.push_back(tok(lane_end + i));

    int failures = 0;
    auto gate = [&](bool ok, const char* what, const char* detail, const Result& r) {
        std::fprintf(stderr, "%s %-58s KL %.9f dmax %.6g top %d/%d%s%s\n",
                     ok ? "PASS" : "FAIL", what, r.kl, r.dmax, r.top_ref, r.top_got,
                     r.same_top ? " same-top" : " TOP-DIFFERS", detail);
        if (!ok) ++failures;
    };

    // Reference: a fresh full prefill of the edited prompt.
    auto* ref_ctx = make_ctx(m, n_ctx);
    if (!decode(ref_ctx, edited_tokens, 0, true)) { std::fprintf(stderr, "reference prefill failed\n"); return 2; }
    const std::vector<float> ref_edited = logits_of(ref_ctx);
    llama_free(ref_ctx);

    // D: the floor. The same fresh prompt prefilled in another batch shape: a
    // cache mechanism cannot be held to a smaller KL than the CPU's arithmetic
    // between two ubatch splits of the same tokens.
    const int floor_chunk = std::max(64, edited / 4);
    auto* floor_ctx = make_ctx(m, n_ctx);
    if (!decode(floor_ctx, edited_tokens, 0, true, floor_chunk)) { std::fprintf(stderr, "floor prefill failed\n"); return 2; }
    const Result floor_r = compare(ref_edited, logits_of(floor_ctx));
    llama_free(floor_ctx);
    const double floor_kl = floor_r.kl;
    std::fprintf(stderr, "%s %-58s KL %.9f dmax %.6g top %d/%d%s (chunk %d)\n",
                 floor_r.same_top ? "PASS" : "FAIL", "D batch-shape floor: same prompt, another ubatch split",
                 floor_r.kl, floor_r.dmax, floor_r.top_ref, floor_r.top_got,
                 floor_r.same_top ? " same-top" : " TOP-DIFFERS", floor_chunk);
    if (!floor_r.same_top) ++failures;

    // Reference: a fresh full prefill of the appended prompt.
    auto* ref_ctx2 = make_ctx(m, n_ctx);
    if (!decode(ref_ctx2, append_tokens, 0, true)) { std::fprintf(stderr, "append reference failed\n"); return 2; }
    const std::vector<float> ref_append = logits_of(ref_ctx2);
    llama_free(ref_ctx2);

    // A: append-only follow-up, no cut.
    {
        auto* ctx = make_ctx(m, n_ctx);
        if (!decode(ctx, lane_tokens, 0, true)) { std::fprintf(stderr, "append lane failed\n"); return 2; }
        std::vector<llama_token> extra(append_tokens.begin() + lane_end, append_tokens.end());
        if (!decode(ctx, extra, lane_end, true)) { std::fprintf(stderr, "append follow-up failed\n"); return 2; }
        const Result r = compare(ref_append, logits_of(ctx));
        // the top token must equal. An append-only follow-up has its tail in
        // another ubatch split than the single full prefill: on a chunk-boundary
        // dependent mode (Dory's last_loop) the floor arm D measures that, and
        // the follow-up must stay inside it; elsewhere it is zero.
        gate(r.same_top && r.kl <= std::max(floor_kl, 1e-3), "A append-only follow-up matches the fresh prefill", "", r);
        llama_free(ctx);
    }

    // The lane at the checkpoint, then its edited prompt.
    auto* lane = make_ctx(m, n_ctx);
    if (!decode(lane, std::vector<llama_token>(lane_tokens.begin(), lane_tokens.begin() + cut), 0, true)) { return 2; }
    const std::vector<uint8_t> ckpt = save_partial(lane, 0);
    if (!decode(lane, std::vector<llama_token>(lane_tokens.begin() + cut, lane_tokens.end() - 1), cut, false)) { return 2; }
    if (!decode(lane, {lane_tokens[lane_end - 1]}, lane_end - 1, true)) { return 2; }
    llama_memory_t mem = llama_get_memory(lane);
    // the lane's whole state, for the C arm: it needs the same cache at the
    // same positions without a second prefill
    const size_t full_sz = llama_state_seq_get_size_ext(lane, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> full(full_sz);
    if (llama_state_seq_get_data_ext(lane, full.data(), full_sz, 0, LLAMA_STATE_SEQ_FLAGS_NONE) != full_sz) { return 2; }

    // A partial state above the window must be larger than one below it: the
    // SWA part is what makes the difference; without it, both are equal.
    {
        const size_t full = ckpt.size();
        auto* probe = make_ctx(m, n_ctx);
        if (!decode(probe, std::vector<llama_token>(edited_tokens.begin(), edited_tokens.begin() + early), 0, true)) { return 2; }
        const size_t low = save_partial(probe, 0).size();
        llama_free(probe);
        std::fprintf(stderr, "checkpoint bytes: %d tokens %.1f MiB, %d tokens %.1f MiB, delta %.1f MiB\n",
                     cut, full / 1048576.0, early, low / 1048576.0, (full - low) / 1048576.0);
        if (!(full > low)) {
            std::fprintf(stderr, "FAIL %-58s the partial state does not grow with the window: no SWA part\n",
                         "the checkpoint carries the sliding-window state");
            ++failures;
        } else {
            std::fprintf(stderr, "PASS %-58s +%.1f MiB over a below-window checkpoint\n",
                         "the checkpoint carries the sliding-window state", (full - low) / 1048576.0);
        }
    }

    // The cut the edited prompt shares: everything below `common` matches.
    size_t common = 0;
    const std::vector<llama_token>& have = lane_tokens;
    while (common < have.size() && common < edited_tokens.size() && have[common] == edited_tokens[common]) ++common;
    if (common >= edited_tokens.size()) common = edited_tokens.size() - 1;
    std::fprintf(stderr, "edit: common %zu, lane %zu, window %d, pos_min %d\n",
                 common, have.size(), n_swa, (int) llama_memory_attn_seq_pos_min(mem, 0));

    // B: arcint's rule. The cache's retained history must reach the next
    // query's window; otherwise resume from the newest checkpoint inside the
    // prefix and prefill from it.
    {
        const bool complete = lgc::swa_window_trim_complete(n_swa, llama_memory_attn_seq_pos_min(mem, 0), common);
        size_t from = common;
        bool restore = false;
        if (!complete || !llama_memory_seq_rm(mem, 0, (llama_pos) common, -1)) {
            if (!ckpt.empty() && llama_state_seq_set_data_ext(lane, ckpt.data(), ckpt.size(), 0,
                                                              LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == ckpt.size() &&
                llama_memory_seq_rm(mem, 0, (llama_pos) cut, -1)) {
                from = cut;
                restore = true;
            } else {
                llama_memory_seq_rm(mem, 0, -1, -1);
                from = 0;
            }
        }
        std::fprintf(stderr, "B arm: trim-complete %d, resumed from %zu%s\n", (int) complete, from, restore ? " (checkpoint)" : "");
        if (!decode(lane, std::vector<llama_token>(edited_tokens.begin() + from, edited_tokens.end() - 1), from, false)) { return 2; }
        if (!decode(lane, {edited_tokens.back()}, edited - 1, true)) { return 2; }
        const Result r = compare(ref_edited, logits_of(lane));
        // the same floor: the restored SWA cells are re-found in the ring's slot
        // order and the tail is re-prefilled with another ubatch split, so the
        // CPU's FA sums in a different order; the card judges at the answer
        // same top token, mean KL inside the answer-level bar: the restored SWA
        // cells are re-found in the ring's slot order, so the CPU's FA sums them
        // in a different order than the original placement (the floor arm D above
        // holds the batch-shape part of that; measured 1.5e-3 to 2.3e-2)
        gate(r.same_top && r.kl <= 0.03, "B edited last message, resumed from the checkpoint", "", r);
    }
    llama_free(lane);

    // C: the pre-change behaviour on a lane at the same state: trim to `common`
    // and resume, trusting seq_rm's true. This is the red arm; it must differ.
    // It applies to an attention-only SWA model, where seq_rm always returns
    // true: the SWA cache is trimmed and its window keeps a hole. A hybrid's
    // recurrent memory refuses that rollback (n_rs_seq 0), so the trim never
    // reaches the attention cache there and the arm is skipped.
    if (llama_model_is_hybrid(m)) {
        std::fprintf(stderr, "SKIP %-58s a hybrid's recurrent refusal blocks the trim\n",
                     "C trim-only resume (the pre-change rule) is wrong");
    } else {
        auto* ctx = make_ctx(m, n_ctx);
        if (llama_state_seq_set_data_ext(ctx, full.data(), full_sz, 0, LLAMA_STATE_SEQ_FLAGS_NONE) != full_sz) {
            std::fprintf(stderr, "C arm: the full state did not load\n"); return 2;
        }
        llama_memory_t c_mem = llama_get_memory(ctx);
        if (!llama_memory_seq_rm(c_mem, 0, (llama_pos) common, -1)) { std::fprintf(stderr, "C arm: seq_rm refused\n"); return 2; }
        if (!decode(ctx, std::vector<llama_token>(edited_tokens.begin() + common, edited_tokens.end() - 1), common, false)) { return 2; }
        if (!decode(ctx, {edited_tokens.back()}, edited - 1, true)) { return 2; }
        const Result r = compare(ref_edited, logits_of(ctx));
        const bool red = !(r.same_top && r.kl <= 1e-3);
        std::fprintf(stderr, "%s %-58s KL %.9f dmax %.6g top %d/%d%s\n",
                     red ? "PASS" : "FAIL", "C trim-only resume (the pre-change rule) is wrong", r.kl, r.dmax,
                     r.top_ref, r.top_got, r.same_top ? " same-top" : " TOP-DIFFERS");
        if (!red) ++failures;
        llama_free(ctx);
    }


    llama_model_free(m);
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return failures == 0 ? 0 : 3;
}
