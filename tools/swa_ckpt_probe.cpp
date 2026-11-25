// The SWA cache's checkpoint resume: a partial state
// (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) must carry the sliding window, and a cut
// below the retained window must resume from a checkpoint instead of trimming.
// By default the window-sized cache (--llama-swa-window) on the CPU, as before.
//
// Needs a GGUF; not a ctest case (the unit gate is tests/test_swa_window.cpp for
// the rule and the flag). Built with arcint (target arcint-swa-ckpt-probe), or
// CPU only against the pinned tree:
//
//   g++ -O2 -std=c++17 -I include -I ggml/include -I src -I <arcint>/src <arcint>/tools/swa_ckpt_probe.cpp
//       -o swa_ckpt_probe -L build/bin -lllama -Wl,-rpath,$PWD/build/bin
//
//   ./swa_ckpt_probe model.gguf [n_ctx] [per_loop|last_loop] [options]
//
// Options (contrib/llama.cpp 0209: the card's q8_0 / q4_0 K/V restore):
//   --ngl N           offload N layers (0, the default: CPU; 99: all, on one
//                     device, split mode none; pick the card by the backend's env)
//   --type-k T        K cache type (f16, the default; q8_0, q4_0, bf16, f32)
//   --type-v T        V cache type (as --type-k)
//   --swa-full        the full-size SWA cache (no --llama-swa-window, 1.8.0's
//                     default) instead of the window-sized one
//   --exact           run only the exact-restore arms E0-E2 below (and exit)
// With --ngl > 0 the answer-level bar is comparative (CLAUDE.md): an arm passes
// at the same top token and a KL at most 0.03 nats above the floor arm D on the
// same card (the B60's per-forward floor is ~0.14 nats, an absolute bound would
// fail a right answer). On the CPU the bounds stay as they were.
// The B arm states whether the checkpoint restore was taken, and fails when it
// was expected (a hybrid, whose recurrent memory refuses the trim, or a window
// the trim cannot keep) and not taken: a pass by a fresh prefill proves nothing
// about the restore.
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
//
// B and D are comparative, not exact: B's tail is prefilled in other ubatches
// than the fresh reference's (and its last token alone), D is the same prompt in
// another split, so neither is zero on a card, and the last token's mean KL on
// these synthetic tokens barely moves when a few hundred SWA cells hold the wrong
// bytes. The E arms (--exact) hold a restore to what it must be: bit-identical to
// the cells never having left the device.
//   E0 two contexts prefill the same prefix in the same calls: their partial
//      states are byte-equal (the card's determinism, the E arms' premise)
//   E1 arcint's case: the lane runs on past the checkpoint (the SWA ring reuses
//      the cells), restores the partial checkpoint and trims to the cut. The
//      partial state read back must equal the checkpoint byte for byte, and the
//      tail's logits (every row of one batch, then one decode) must equal the
//      untouched context's, which prefilled the same prefix in the same calls
//   E2 the early edit: after a FULL state save at the cut the lane is cleared and
//      prefilled with another text from position 64 on, then the full state is
//      restored: the stale cells hold different text, so a restore that writes
//      nothing differs by construction. State bytes and tail logits as in E1.
// After a byte-equal restore the B60 gave other tail logits than the untouched
// context (deterministically, the same after E1 and E2; the CPU gave equal rows).
// The arms that separate the candidates, in the same invocation:
//   E0' a second untouched context, the same calls: partial and full state bytes
//      and every tail logit row (unequal: the card is not deterministic here)
//   R1 / R2 after E1's / E2's restore, seq 0's K / V rows read from the WHOLE
//      storage tensors (not the views the state I/O uses), in position order, vs U's
//   E3 the restore without a history: a context's own partial state restored into
//      it at once (the restored layout, none of the lane's history), vs U and E1
//   C  the lane's history without a restore: the lane runs on as E1's, is cleared
//      whole and prefills the prefix again in the same calls; its layout per cache
//      is compared with E1's (by reading its SWA cache sits as U's: a prefill keeps
//      position p in cell p, a restore refills the window from cell 0), its tail
//      with U's and E1's
//   layout lines: per attention cache the cells of seq 0 (index by position,
//      used, used_max_p1) and the recurrent memory's head / rs_z / seq 0's cell
// The expectations per candidate are printed first; the "E reading" line gives the
// one the arms point to.
#include "llama.h"
#include "llama-ext.h"   // llama_memory_attn_seq_pos_min (contrib/llama.cpp 0153)
#include "core/swa_window.h"
// the E arms read the cache's own layout and storage (the pinned tree's src/)
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Config {
    int       ngl      = 0;
    ggml_type type_k   = GGML_TYPE_F16;
    ggml_type type_v   = GGML_TYPE_F16;
    bool      swa_full = false;   // false: --llama-swa-window
    bool      exact    = false;   // --exact: the E arms only
};
Config g_cfg;

bool parse_type(const char* s, ggml_type& t) {
    static const struct { const char* name; ggml_type type; } types[] = {
        {"f16", GGML_TYPE_F16}, {"bf16", GGML_TYPE_BF16}, {"f32", GGML_TYPE_F32},
        {"q8_0", GGML_TYPE_Q8_0}, {"q4_0", GGML_TYPE_Q4_0},
    };
    for (const auto& e : types) {
        if (std::strcmp(s, e.name) == 0) { t = e.type; return true; }
    }
    return false;
}

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
    cp.swa_full   = g_cfg.swa_full;   // false: --llama-swa-window
    cp.type_k     = g_cfg.type_k;
    cp.type_v     = g_cfg.type_v;
    return llama_init_from_model(m, cp);
}

std::vector<uint8_t> save_state(llama_context* ctx, int seq, llama_state_seq_flags flags) {
    const size_t sz = llama_state_seq_get_size_ext(ctx, seq, flags);
    std::vector<uint8_t> out(sz);
    if (sz == 0) return out;
    if (llama_state_seq_get_data_ext(ctx, out.data(), sz, seq, flags) != sz) out.clear();
    return out;
}

std::vector<uint8_t> save_partial(llama_context* ctx, int seq) {
    return save_state(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
}

// the E arms: byte and logit equality
struct ByteDiff {
    size_t n_a = 0, n_b = 0, n_diff = 0, first = 0;
    bool equal() const { return n_a == n_b && n_a > 0 && n_diff == 0; }
};

ByteDiff diff_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    ByteDiff d;
    d.n_a = a.size();
    d.n_b = b.size();
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (d.n_diff == 0) d.first = i;
            ++d.n_diff;
        }
    }
    d.n_diff += std::max(a.size(), b.size()) - n;
    return d;
}

// every row's logits: one batch with a logit asked for each token
bool decode_rows(llama_context* ctx, const std::vector<llama_token>& t, int p0, std::vector<float>& rows) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_batch b = llama_batch_init((int) t.size(), 0, 1);
    for (int i = 0; i < (int) t.size(); ++i) {
        b.token[i]     = t[i];
        b.pos[i]       = p0 + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    b.n_tokens = (int) t.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) return false;
    for (int i = 0; i < (int) t.size(); ++i) {
        const float* l = llama_get_logits_ith(ctx, i);
        rows.insert(rows.end(), l, l + n_vocab);
    }
    return true;
}

struct RowDiff {
    int    rows = 0, rows_diff = 0, argmax_diff = 0;
    double max_abs = 0.0;
    bool equal() const { return rows > 0 && rows_diff == 0; }
};

RowDiff diff_rows(const std::vector<float>& a, const std::vector<float>& b, int n_vocab) {
    RowDiff d;
    if (a.size() != b.size() || a.empty()) { d.rows_diff = d.rows = 1; d.max_abs = INFINITY; return d; }
    d.rows = (int) (a.size() / n_vocab);
    for (int r = 0; r < d.rows; ++r) {
        const float* x = a.data() + (size_t) r * n_vocab;
        const float* y = b.data() + (size_t) r * n_vocab;
        if (std::memcmp(x, y, sizeof(float) * n_vocab) != 0) ++d.rows_diff;
        int ax = 0, ay = 0;
        for (int i = 0; i < n_vocab; ++i) {
            if (x[i] > x[ax]) ax = i;
            if (y[i] > y[ay]) ay = i;
            d.max_abs = std::max(d.max_abs, (double) std::fabs(x[i] - y[i]));
        }
        if (ax != ay) ++d.argmax_diff;
    }
    return d;
}

// The cache's own view of where a sequence's cells sit (contrib/llama.cpp internals: the probe is
// built against the pinned tree's src/), and the K / V bytes read from the whole tensors, not a view.
struct CacheLayout {
    std::string name;
    uint32_t size = 0, used = 0, used_max_p1 = 0;
    std::vector<std::pair<llama_pos, uint32_t>> cells;   // (pos, cell index) of seq 0, by position
    llama_kv_cache* kv = nullptr;
};
struct Layout {
    std::vector<CacheLayout> caches;
    std::string recr;   // the recurrent memory's head / used / rs_z and seq 0's cell
};

std::vector<std::pair<const char*, llama_kv_cache*>> attn_caches(llama_memory_t mem) {
    if (auto* h = dynamic_cast<llama_memory_hybrid_iswa*>(mem)) {
        return {{"base", h->get_mem_attn()->get_base()}, {"swa", h->get_mem_attn()->get_swa()}};
    }
    if (auto* i = dynamic_cast<llama_kv_cache_iswa*>(mem)) return {{"base", i->get_base()}, {"swa", i->get_swa()}};
    if (auto* h = dynamic_cast<llama_memory_hybrid*>(mem)) return {{"kv", h->get_mem_attn()}};
    if (auto* k = dynamic_cast<llama_kv_cache*>(mem)) return {{"kv", k}};
    return {};
}

llama_memory_recurrent* recr_of(llama_memory_t mem) {
    if (auto* h = dynamic_cast<llama_memory_hybrid_iswa*>(mem)) return h->get_mem_recr();
    if (auto* h = dynamic_cast<llama_memory_hybrid*>(mem)) return h->get_mem_recr();
    return dynamic_cast<llama_memory_recurrent*>(mem);
}

Layout layout_of(llama_context* ctx) {
    Layout L;
    llama_memory_t mem = llama_get_memory(ctx);
    for (const auto& [name, kv] : attn_caches(mem)) {
        CacheLayout c;
        c.name = name;
        c.kv   = kv;
        const llama_kv_cells& cells = kv->get_cells(0);
        c.size = cells.size();
        c.used = cells.get_used();
        c.used_max_p1 = cells.used_max_p1();
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, 0)) c.cells.emplace_back(cells.pos_get(i), i);
        }
        std::sort(c.cells.begin(), c.cells.end());
        L.caches.push_back(std::move(c));
    }
    if (auto* r = recr_of(mem)) {
        char buf[256];
        int cell = -1;
        for (uint32_t i = 0; i < r->size; ++i) {
            if (r->cells[i].has_seq_id(0)) { cell = (int) i; break; }
        }
        std::snprintf(buf, sizeof(buf), "head %u used %u size %u rs_z %d seq0 cell %d", r->head, r->used, r->size, r->rs_z, cell);
        std::string s = buf;
        if (cell >= 0) {
            const auto& mc = r->cells[cell];
            std::snprintf(buf, sizeof(buf), " (pos %d src %d src0 %d tail %d)", mc.pos, mc.src, mc.src0, mc.tail);
            s += buf;
        }
        L.recr = s;
    }
    return L;
}

bool layout_equal(const Layout& a, const Layout& b) {
    if (a.caches.size() != b.caches.size() || a.recr != b.recr) return false;
    for (size_t i = 0; i < a.caches.size(); ++i) {
        const auto& x = a.caches[i];
        const auto& y = b.caches[i];
        if (x.used != y.used || x.used_max_p1 != y.used_max_p1 || x.cells != y.cells) return false;
    }
    return true;
}

void print_layout(const char* who, const Layout& L) {
    std::fprintf(stderr, "layout %s:", who);
    for (const auto& c : L.caches) {
        // runs of consecutive cells for consecutive positions
        int runs = 0;
        for (size_t i = 0; i < c.cells.size(); ++i) {
            if (i == 0 || c.cells[i].second != c.cells[i - 1].second + 1 || c.cells[i].first != c.cells[i - 1].first + 1) ++runs;
        }
        std::fprintf(stderr, " | %s size %u used %u max_p1 %u, %zu cells, pos %d..%d at cells %u..%u in %d run(s)", c.name.c_str(),
                     c.size, c.used, c.used_max_p1, c.cells.size(), c.cells.empty() ? -1 : c.cells.front().first,
                     c.cells.empty() ? -1 : c.cells.back().first, c.cells.empty() ? 0u : c.cells.front().second,
                     c.cells.empty() ? 0u : c.cells.back().second, runs);
    }
    std::fprintf(stderr, " | recurrent %s\n", L.recr.c_str());
}

// per cache: for each layer, the K and V rows of seq 0's cells at positions [0, n_pos), in position order,
// read from the whole storage tensor (ggml_backend_tensor_get on the base, not a view)
struct RawKV {
    std::vector<std::string> names;
    std::vector<std::vector<uint8_t>> k, v;   // [cache][bytes], all layers concatenated
    std::vector<size_t> row_k, row_v, n_rows;  // per cache: bytes a row (all layers), rows
    std::string note;
};

RawKV raw_kv(llama_context* ctx, const Layout& L, int n_pos) {
    RawKV R;
    for (const auto& c : L.caches) {
        std::vector<uint8_t> ks, vs;
        size_t rk = 0, rv = 0;
        std::vector<std::pair<llama_pos, uint32_t>> sel;
        for (const auto& pc : c.cells) if (pc.first < n_pos) sel.push_back(pc);
        // one layer at a time, rows in position order: [layer][pos][row]
        std::vector<std::vector<uint8_t>> kl, vl;
        for (uint32_t il : c.kv->get_layer_ids()) {
            ggml_tensor* kt = c.kv->get_k_storage((int32_t) il);
            llama_kv_cache::slot_info si;
            si.s0 = 0; si.s1 = 0; si.strm = {0}; si.idxs = {{0}};
            ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
            ggml_context* tc = ggml_init(ip);
            ggml_tensor* vview = c.kv->get_v(tc, (int32_t) il, 1, si);
            ggml_tensor* vt = vview->view_src ? vview->view_src : vview;
            const bool v_trans = vview->nb[1] > vview->nb[2];
            ggml_free(tc);
            std::vector<uint8_t> kall(ggml_nbytes(kt)), vall(ggml_nbytes(vt));
            ggml_backend_tensor_get(kt, kall.data(), 0, kall.size());
            ggml_backend_tensor_get(vt, vall.data(), 0, vall.size());
            const size_t krow = kt->nb[1];
            const size_t vrow = v_trans ? ggml_type_size(vt->type) * (size_t) vt->ne[0] : vt->nb[1];
            rk += krow; rv += vrow;
            for (const auto& pc : sel) {
                ks.insert(ks.end(), kall.begin() + (size_t) pc.second * krow, kall.begin() + (size_t) (pc.second + 1) * krow);
                if (!v_trans) {
                    vs.insert(vs.end(), vall.begin() + (size_t) pc.second * vrow, vall.begin() + (size_t) (pc.second + 1) * vrow);
                } else {
                    const size_t es = ggml_type_size(vt->type), kvs = (size_t) vt->ne[1];
                    for (int64_t j = 0; j < vt->ne[0]; ++j) {
                        const size_t off = ((size_t) j * kvs + pc.second) * es;
                        vs.insert(vs.end(), vall.begin() + off, vall.begin() + off + es);
                    }
                }
            }
        }
        R.names.push_back(c.name);
        R.k.push_back(std::move(ks));
        R.v.push_back(std::move(vs));
        R.row_k.push_back(rk);
        R.row_v.push_back(rv);
        R.n_rows.push_back(sel.size());
    }
    return R;
}

bool report_raw(const char* what, const RawKV& a, const RawKV& b) {
    bool ok = a.names == b.names;
    std::string detail;
    for (size_t i = 0; ok && i < a.names.size(); ++i) {
        const ByteDiff dk = diff_bytes(a.k[i], b.k[i]);
        const ByteDiff dv = diff_bytes(a.v[i], b.v[i]);
        char buf[200];
        std::snprintf(buf, sizeof(buf), " %s: %zu cells, K %zu of %zu bytes differ, V %zu of %zu;", a.names[i].c_str(), b.n_rows[i],
                      dk.n_diff, std::max(dk.n_a, dk.n_b), dv.n_diff, std::max(dv.n_a, dv.n_b));
        detail += buf;
        ok = ok && dk.equal() && dv.equal() && a.n_rows[i] == b.n_rows[i];
    }
    std::fprintf(stderr, "%s %-58s%s\n", ok ? "PASS" : "FAIL", what, a.names == b.names ? detail.c_str() : " the caches differ");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    // the positional arguments as before; the options anywhere after the model
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--ngl" && i + 1 < argc) {
            g_cfg.ngl = std::atoi(argv[++i]);
        } else if ((a == "--type-k" || a == "--type-v") && i + 1 < argc) {
            if (!parse_type(argv[++i], a == "--type-k" ? g_cfg.type_k : g_cfg.type_v)) {
                std::fprintf(stderr, "unknown cache type %s\n", argv[i]);
                return 2;
            }
        } else if (a == "--swa-full") {
            g_cfg.swa_full = true;
        } else if (a == "--exact") {
            g_cfg.exact = true;
        } else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        } else {
            pos.push_back(argv[i]);
        }
    }
    if (pos.empty()) {
        std::fprintf(stderr, "usage: %s model.gguf [n_ctx] [per_loop|last_loop] [--ngl N] [--type-k T] [--type-v T] [--swa-full] [--exact]\n", argv[0]);
        return 2;
    }
    const int n_ctx = pos.size() > 1 ? std::atoi(pos[1]) : 8192;
    const char* dory_kv = pos.size() > 2 ? pos[2] : nullptr;
    const bool card = g_cfg.ngl > 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_cfg.ngl;
    if (card) {
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;   // one device: the one the backend's env selects
        mp.main_gpu   = 0;
    }
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
    llama_model* m = llama_model_load_from_file(pos[0], mp);
    if (m == nullptr) { std::fprintf(stderr, "load failed\n"); return 2; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const int n_swa   = llama_model_n_swa(m);
    const int n_layer = llama_model_n_layer(m);
    std::fprintf(stderr, "model: n_layer %d, n_swa %d, hybrid %d, recurrent %d, n_ctx %d%s%s\n",
                 n_layer, n_swa, (int) llama_model_is_hybrid(m), (int) llama_model_is_recurrent(m), n_ctx,
                 dory_kv ? ", kv " : "", dory_kv ? dory_kv : "");
    std::fprintf(stderr, "config: ngl %d (%s), K %s, V %s, SWA cache %s\n", g_cfg.ngl, card ? "card" : "CPU",
                 ggml_type_name(g_cfg.type_k), ggml_type_name(g_cfg.type_v),
                 g_cfg.swa_full ? "full-size (--swa-full)" : "window-sized (--llama-swa-window)");
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

    if (g_cfg.exact) {
        const int e_tail = std::min(128, lane_end - cut - 2);
        std::fprintf(stderr, "E arms: prefix %d tokens (the lane's calls), lane %d, tail %d rows + 1 decode, early edit at 64\n",
                     cut, lane_end, e_tail);
        // what each candidate predicts, written before anything runs
        std::fprintf(stderr,
            "expect: E0' (two untouched contexts, the same calls) differing = the card is not deterministic here; nothing else is readable\n"
            "expect: (a) a kernel reads another layout than set / get: E0' equal, E1 differs, R1 (raw K/V, the whole tensors, no view) differs\n"
            "expect: (b) a derived cache / state outside the serialized state is not invalidated: E0' equal, E1 differs, R1 equal\n"
            "expect: (c) the cell layout changes the summation order: E3 (the restore without a history) equals E1, C (the history without a restore, U's SWA layout) equals U, "
            "and with GGML_OPENCL_FA_DPAS2_SPLIT=1 E1 / E2 match U up to the decode row (its n_kv differs)\n"
            "expect: history (a derived state of the lane's decodes): C differs from U\n");
        const std::vector<llama_token> prefix(lane_tokens.begin(), lane_tokens.begin() + cut);
        const std::vector<llama_token> tail(lane_tokens.begin() + cut, lane_tokens.begin() + cut + e_tail);
        const llama_token next = lane_tokens[cut + e_tail];
        auto run_tail = [&](llama_context* c, std::vector<float>& rows) {
            return decode_rows(c, tail, cut, rows) && decode_rows(c, {next}, cut + e_tail, rows);
        };
        auto report = [&](const char* what, const ByteDiff* bd, const RowDiff* rd) {
            const bool ok = (bd == nullptr || bd->equal()) && (rd == nullptr || rd->equal());
            std::fprintf(stderr, "%s %-58s", ok ? "PASS" : "FAIL", what);
            if (bd) std::fprintf(stderr, " bytes %zu of %zu differ (first at %zu, sizes %zu/%zu)", bd->n_diff, std::max(bd->n_a, bd->n_b),
                                 bd->n_diff ? bd->first : 0, bd->n_a, bd->n_b);
            if (rd) std::fprintf(stderr, "%s rows %d of %d differ, argmax %d, max |d| %.6g", bd ? ";" : "", rd->rows_diff, rd->rows,
                                 rd->argmax_diff, rd->max_abs);
            std::fprintf(stderr, "\n");
            return ok;
        };
        auto prefill_prefix = [&](llama_context* c) { return decode(c, prefix, 0, true); };
        auto run_lane_on = [&](llama_context* c) {
            return decode(c, std::vector<llama_token>(lane_tokens.begin() + cut, lane_tokens.end() - 1), cut, false) &&
                   decode(c, {lane_tokens[lane_end - 1]}, lane_end - 1, true);
        };

        // U: the untouched reference
        auto* u = make_ctx(m, n_ctx);
        if (!prefill_prefix(u)) { std::fprintf(stderr, "E: untouched prefill failed\n"); return 2; }
        const std::vector<uint8_t> u_part = save_partial(u, 0);
        const std::vector<uint8_t> u_full = save_state(u, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        const Layout lay_u = layout_of(u);
        const RawKV raw_u = raw_kv(u, lay_u, cut);
        print_layout("U", lay_u);
        std::vector<float> rows_u;
        if (!run_tail(u, rows_u)) { std::fprintf(stderr, "E: untouched tail failed\n"); return 2; }
        llama_free(u);

        // E0': a second untouched context with the same calls: state bytes and tail logits
        bool e0p_ok = true;
        {
            auto* u2 = make_ctx(m, n_ctx);
            if (!prefill_prefix(u2)) { return 2; }
            const ByteDiff bp = diff_bytes(u_part, save_partial(u2, 0));
            const ByteDiff bf = diff_bytes(u_full, save_state(u2, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
            print_layout("U2", layout_of(u2));
            std::vector<float> rows;
            if (!run_tail(u2, rows)) { return 2; }
            const RowDiff rd = diff_rows(rows_u, rows, n_vocab);
            e0p_ok &= report("E0' two untouched contexts: partial states", &bp, nullptr);
            e0p_ok &= report("E0' two untouched contexts: full states", &bf, nullptr);
            e0p_ok &= report("E0' two untouched contexts: tail logits", nullptr, &rd);
            if (!e0p_ok) ++failures;
            llama_free(u2);
        }

        // E1: the lane runs on, restores the partial checkpoint, trims to the cut
        std::vector<float> rows_e1;
        bool e1_rows_ok = false, r1_ok = false, have_e1 = false;
        Layout lay_e1;
        {
            auto* l = make_ctx(m, n_ctx);
            if (!prefill_prefix(l)) { return 2; }
            const std::vector<uint8_t> ck = save_partial(l, 0);
            { const ByteDiff b0 = diff_bytes(u_part, ck); if (!report("E0 two prefills of the prefix: partial states equal", &b0, nullptr)) ++failures; }
            if (!run_lane_on(l)) { return 2; }
            llama_memory_t mem_l = llama_get_memory(l);
            if (ck.empty() || llama_state_seq_set_data_ext(l, ck.data(), ck.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != ck.size() ||
                !llama_memory_seq_rm(mem_l, 0, (llama_pos) cut, -1)) {
                std::fprintf(stderr, "FAIL %-58s the restore was refused\n", "E1 partial restore after the lane ran on");
                ++failures;
            } else {
                const ByteDiff bd = diff_bytes(ck, save_partial(l, 0));
                lay_e1 = layout_of(l);
                print_layout("E1", lay_e1);
                const RawKV raw = raw_kv(l, lay_e1, cut);
                r1_ok = report_raw("R1 E1's restored K/V, whole tensors (no view) vs U's", raw_u, raw);
                if (!run_tail(l, rows_e1)) { std::fprintf(stderr, "E1: tail failed\n"); return 2; }
                const RowDiff rd = diff_rows(rows_u, rows_e1, n_vocab);
                e1_rows_ok = rd.equal();
                have_e1 = true;
                if (!report("E1 partial restore after the lane ran on", &bd, &rd)) ++failures;
                if (!r1_ok) ++failures;
            }
            llama_free(l);
        }

        // E2: full state at the cut, the lane refilled with another text from 64 on, the full state restored
        {
            auto* l = make_ctx(m, n_ctx);
            if (!prefill_prefix(l)) { return 2; }
            const std::vector<uint8_t> ck = save_state(l, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
            llama_memory_t mem_l = llama_get_memory(l);
            llama_memory_seq_rm(mem_l, 0, -1, -1);
            std::vector<llama_token> other(lane_tokens.begin(), lane_tokens.end());
            for (int i = 64; i < (int) other.size(); ++i) {
                other[i] = (llama_token) (1000 + (i * 104729 + 17) % (n_vocab - 2000));
            }
            if (!decode(l, other, 0, true)) { std::fprintf(stderr, "E2: the other text failed\n"); return 2; }
            if (ck.empty() || llama_state_seq_set_data_ext(l, ck.data(), ck.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != ck.size()) {
                std::fprintf(stderr, "FAIL %-58s the restore was refused\n", "E2 full restore over another text (early edit)");
                ++failures;
            } else {
                const ByteDiff bd = diff_bytes(ck, save_state(l, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
                const Layout lay = layout_of(l);
                print_layout("E2", lay);
                const RawKV raw = raw_kv(l, lay, cut);
                if (!report_raw("R2 E2's restored K/V, whole tensors (no view) vs U's", raw_u, raw)) ++failures;
                std::vector<float> rows;
                if (!run_tail(l, rows)) { std::fprintf(stderr, "E2: tail failed\n"); return 2; }
                const RowDiff rd = diff_rows(rows_u, rows, n_vocab);
                if (!report("E2 full restore over another text (early edit)", &bd, &rd)) ++failures;
                if (have_e1) { const RowDiff r12 = diff_rows(rows_e1, rows, n_vocab); report("E1 vs E2 tail logits (the same after either restore?)", nullptr, &r12); }
            }
            llama_free(l);
        }

        // E3: the restore without a history: a context's own partial state restored into it at once. The
        // restore clears the sequence and refills its cells from cell 0, so E3 has the restored layout
        // (as E1 / E2) and none of the lane's history
        std::vector<float> rows_e3;
        bool e3_like_e1 = false, e3_lay_eq = false, have_e3 = false;
        {
            auto* l = make_ctx(m, n_ctx);
            if (!prefill_prefix(l)) { return 2; }
            const std::vector<uint8_t> ck = save_partial(l, 0);
            if (ck.empty() || llama_state_seq_set_data_ext(l, ck.data(), ck.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != ck.size() ||
                !llama_memory_seq_rm(llama_get_memory(l), 0, (llama_pos) cut, -1)) {
                std::fprintf(stderr, "FAIL %-58s the restore was refused\n", "E3 restore in place, no history");
                ++failures;
            } else {
                const ByteDiff bd = diff_bytes(ck, save_partial(l, 0));
                const Layout lay = layout_of(l);
                print_layout("E3", lay);
                e3_lay_eq = have_e1 && layout_equal(lay, lay_e1);
                std::fprintf(stderr, "E3 layout vs E1: %s; vs U: %s\n", have_e1 ? (e3_lay_eq ? "equal" : "DIFFERS") : "no E1",
                             layout_equal(lay, lay_u) ? "equal" : "differs");
                if (!run_tail(l, rows_e3)) { return 2; }
                const RowDiff rd = diff_rows(rows_u, rows_e3, n_vocab);
                if (!report("E3 restore in place, no history", &bd, &rd)) ++failures;
                if (have_e1) {
                    const RowDiff r31 = diff_rows(rows_e1, rows_e3, n_vocab);
                    e3_like_e1 = r31.equal();
                    report("E3 vs E1 tail logits", nullptr, &r31);
                }
                have_e3 = true;
            }
            llama_free(l);
        }

        // C: the lane's history without a restore. The lane runs on as E1's, is cleared whole (a hybrid
        // refuses a partial trim) and prefills the prefix again in the same calls. By reading, its base
        // cache sits as E1's (positions in their own cells) but its SWA cache as U's (a prefill keeps
        // position p in cell p; a restore refills the window from cell 0): the layout lines say which
        bool c_like_e1 = false, c_rows_ok = false, c_lay_eq = false;
        {
            auto* c = make_ctx(m, n_ctx);
            if (!prefill_prefix(c) || !run_lane_on(c)) { return 2; }
            llama_memory_seq_rm(llama_get_memory(c), 0, -1, -1);
            if (!prefill_prefix(c)) { return 2; }
            const ByteDiff bp = diff_bytes(u_part, save_partial(c, 0));
            const Layout lay = layout_of(c);
            print_layout("C", lay);
            c_lay_eq = have_e1 && layout_equal(lay, lay_e1);
            std::fprintf(stderr, "C layout vs E1: %s; vs U: %s\n", have_e1 ? (c_lay_eq ? "equal" : "DIFFERS") : "no E1",
                         layout_equal(lay, lay_u) ? "equal" : "differs");
            if (have_e1) {
                for (size_t i = 0; i < lay.caches.size() && i < lay_e1.caches.size(); ++i) {
                    const bool eq = lay.caches[i].cells == lay_e1.caches[i].cells && lay.caches[i].used_max_p1 == lay_e1.caches[i].used_max_p1;
                    std::fprintf(stderr, "C layout vs E1, %s cache: %s\n", lay.caches[i].name.c_str(), eq ? "equal" : "DIFFERS");
                }
            }
            std::fprintf(stderr, "E1 layout vs U: %s\n", have_e1 ? (layout_equal(lay_e1, lay_u) ? "equal" : "differs") : "no E1");
            std::vector<float> rows;
            if (!run_tail(c, rows)) { return 2; }
            const RowDiff rd = diff_rows(rows_u, rows, n_vocab);
            c_rows_ok = rd.equal();
            report("C the lane's history, no restore: partial state vs U's", &bp, nullptr);
            report("C the lane's history, no restore: tail logits vs U's", nullptr, &rd);
            if (have_e1) {
                const RowDiff rce = diff_rows(rows_e1, rows, n_vocab);
                c_like_e1 = rce.equal();
                report("C vs E1 tail logits", nullptr, &rce);
            }
            llama_free(c);
        }

        // the reading, by the expectations above
        std::string verdict;
        if (!e0p_ok) verdict = "nondeterministic (E0' differs: nothing else is readable)";
        else if (!have_e1) verdict = "unclear (E1 did not run)";
        else if (e1_rows_ok) verdict = "none (E1's tail logits equal U's)";
        else if (!r1_ok) verdict = "a (the raw K/V differ although the state bytes are equal)";
        else if (c_lay_eq && c_like_e1 && !c_rows_ok) verdict = "c (the layout: C, E1's layout without a restore, differs from U like E1)";
        else if (c_lay_eq && c_rows_ok) verdict = "b (raw K/V equal; C, E1's layout without a restore, equals U: a derived state of the restore)";
        else if (!c_rows_ok) verdict = "history (C, the lane's history without a restore, differs from U)";
        else if (have_e3 && e3_lay_eq && e3_like_e1)
            verdict = "c or b-by-restore (E3, the restore without a history, equals E1; C, the history without a restore and in U's SWA layout, equals U): "
                      "the restored layout or the restore itself; the one-split run decides (c: E1 / E2 match U there)";
        else verdict = "unclear (C equals U in another layout; E3 is not E1)";
        std::fprintf(stderr, "E reading: %s\n", verdict.c_str());
        llama_model_free(m);
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return failures == 0 ? 0 : 3;
    }

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
    // the bound an arm's KL is held to: on the CPU as before, on a card the
    // floor plus the answer-level bar's 0.03 nats
    auto kl_bound = [&](double cpu_bound) { return card ? floor_kl + 0.03 : cpu_bound; };

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
        gate(r.same_top && r.kl <= kl_bound(std::max(floor_kl, 1e-3)), "A append-only follow-up matches the fresh prefill", "", r);
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
        // the restore must be what this arm measures: a hybrid's recurrent
        // memory refuses the trim, and a window the trim cannot keep needs the
        // checkpoint; a fresh prefill from 0 would pass without testing it
        const bool want_restore = llama_model_is_hybrid(m) || !complete;
        if (want_restore) {
            std::fprintf(stderr, "%s %-58s %s (checkpoint %zu tokens, %.1f MiB)\n", restore ? "PASS" : "FAIL",
                         "B the checkpoint restore was taken", restore ? "RESTORE TAKEN" : "RESTORE NOT TAKEN",
                         (size_t) cut, ckpt.size() / 1048576.0);
            if (!restore) ++failures;
        } else {
            std::fprintf(stderr, "B arm: no restore expected (the trim keeps the window); %s\n",
                         restore ? "RESTORE TAKEN" : "RESTORE NOT TAKEN");
        }
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
        gate(r.same_top && r.kl <= kl_bound(0.03), "B edited last message, resumed from the checkpoint", "", r);
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
    } else if (g_cfg.swa_full) {
        std::fprintf(stderr, "SKIP %-58s a full-size SWA cache trims exactly\n",
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
