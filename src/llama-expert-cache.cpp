#include "llama-expert-cache.h"

#include "llama-loan-book.h"
#include "llama-expert-route-trace.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-expert-profile.h"
#include "llama-expert-cache-alias.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>
#include <unordered_map>

// The book's log route (llama-loan-book.h's poison_exit, fifth review finding 2):
// the llama log callback, so a llama_log_set user -- arcint routes libllama's
// lines into its own log -- sees the poison line before the exit. Defined here
// because this file owns LLAMA_LOG_ERROR; it reads the logger state and calls
// the callback under no lock and touches no device, so it is safe from the
// generation thread, arcint's mutex and an unwinding request.
namespace llama_loan_book {
void log_line(const char * text) {
    llama_log_internal(GGML_LOG_LEVEL_ERROR, "%s", text);
}

// Round 12: the exit's idle line goes at WARN, the level the unit's log shows
// without an ERROR (arcint routes WARN and above; GGML_LOG_INFO is dropped).
void log_warn(const char * text) {
    llama_log_internal(GGML_LOG_LEVEL_WARN, "%s", text);
}
}  // namespace llama_loan_book

// down_exps -> its layer, for build_moe_ffn (the graph context holds no model)
static std::mutex g_registry_mutex;
static std::unordered_map<const ggml_tensor *, llama_expert_cache_layer *> g_registry;

llama_expert_cache_layer * llama_expert_cache_lookup(const ggml_tensor * down_exps) {
    std::lock_guard<std::mutex> lk(g_registry_mutex);
    auto it = g_registry.find(down_exps);
    return it == g_registry.end() ? nullptr : it->second;
}

llama_expert_cache * llama_expert_cache_owner(const llama_expert_cache_layer * layer) {
    return layer ? layer->owner : nullptr;
}

static bool on_cpu(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr) {
        return false;
    }
    // a memory-mapped tensor's buffer type has no device
    // (ggml_backend_cpu_buffer_from_ptr_type) but is host memory; the CPU's
    // repack buffer is not host memory but the CPU's
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
    if (dev == nullptr) {
        return ggml_backend_buffer_is_host(t->buffer);
    }
    return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

// Strata's STRP profile: the reader patch 0070 adds (llama-expert-profile.cpp)
// patch 0077: the slot buffers' sampled alias tripwire. One whole-slice region
// a slot tensor is written with a pattern carrying its index, all writes first
// and all read-backs after, so two regions that share a physical page show as a
// differing read-back of the earlier one. The sampled switch covers the planar
// types (IQ2_XXS, IQ2_XS, IQ2_S, Q2_0, IQ3_S) and IQ3_XXS, whose set path is a
// plain clEnqueueWriteBuffer and accepts an arbitrary whole slice. The
// whole-tensor SOA readers (Q4_0, Q1_0, Q8_0, IQ4_NL, Q4_K) are excluded: a
// sampled write there reads beyond the caller's buffer. The pattern an entry
// is regenerated at its read-back, so the plan holds spans and ids, not the
// sampled bytes; only the write's own region is transient. The patterns are
// overwritten by the expert load; nothing the model computes changes. A
// finding warns and the load proceeds (Strata's warn-and-proceed; its spacer
// retry is not built).
static int llama_cache_alias_check(const std::vector<ggml_tensor *> & slots) {
    static const bool on = [] {
        const char * e = getenv("LLAMA_EXPERT_CACHE_ALIAS");
        return e == nullptr || atoi(e) != 0;   // off: LLAMA_EXPERT_CACHE_ALIAS=0
    }();
    if (!on) {
        return 0;
    }
    struct plan_t {
        ggml_tensor * t;
        size_t off;
        size_t size;
        uint64_t id;
    };
    std::vector<plan_t> plan;
    size_t sampled_bytes = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        ggml_tensor * t = slots[i];
        if (t == nullptr || t->data == nullptr) continue;
        switch (t->type) {
            case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
            case GGML_TYPE_Q2_0:    case GGML_TYPE_IQ3_S:  case GGML_TYPE_IQ3_XXS: break;
            default: continue;      // the whole-tensor SOA readers
        }
        const size_t slice = (size_t) t->nb[2];
        if ((t->ne[0] * t->ne[1] / ggml_blck_size(t->type)) % 2 != 0) continue;
        const llama_cache_alias_span sp = llama_cache_alias_region(ggml_nbytes(t), slice, i);
        if (!sp.ok) continue;
        std::vector<uint8_t> region(sp.size);
        llama_cache_alias_pattern((uint64_t) i, region.data(), region.size());
        ggml_backend_tensor_set(t, region.data(), sp.offset, region.size());
        plan.push_back({t, sp.offset, sp.size, (uint64_t) i});
        sampled_bytes += sp.size;
    }
    int bad = 0;
    for (const plan_t & p : plan) {
        std::vector<uint8_t> actual(p.size);
        ggml_backend_tensor_get(p.t, actual.data(), p.off, actual.size());
        if (!llama_cache_alias_reads_back(p.id, actual.data(), actual.size())) {
            ++bad;
            LLAMA_LOG_WARN("%s: alias check, slot tensor '%s' reads back a different region\n", __func__, p.t->name);
        }
    }
    LLAMA_LOG_WARN("%s: alias check, %d of %zu sampled slot regions differ, %zu bytes written and read back\n",
                   __func__, bad, plan.size(), sampled_bytes);
    return bad;
}

static bool read_profile(const char * path, int n_layers, int n_expert, std::vector<std::pair<int, int>> & ranked) {
    std::string err;
    return llama_expert_profile_read(path, n_layers, n_expert, ranked, err);
}

llama_expert_cache::~llama_expert_cache() {
    if (book_.phase != 0 && adapt_finish && !layers.empty()) {
        adapt_finish(layers[0].slot[0]);
    }
    // --llama-expert-cache-save F (LLAMA_EXPERT_CACHE_SAVE): the learned profile
    // from this cache's cumulative routing. The default rule is Strata's SYCL/xe
    // save (save_profile, generate.cpp, commit 3d6f1718): the counts mixed with
    // the start's ranking as prior mass, so a short conversation does not wipe
    // the prior; --llama-expert-cache-save-rule cu selects Strata's
    // rank_learned_profile instead (the resident head). Written at teardown after
    // adapt_finish has landed the pending copies; the host tables are still at
    // the phase the last progress() left them in, so a pending incoming expert is
    // marked resident here (Strata does) and a pending outgoing expert is kept
    // only in phase 1. A layer split has one cache a card: the same variable
    // would name one file for both cards, so it is not written then -- the WARN
    // says what is needed.
    if (const char * save = getenv("LLAMA_EXPERT_CACHE_SAVE"); save != nullptr && *save != '\0' && !layers.empty()) {
        const int n_caches = model_ != nullptr ? (int) model_->expert_caches.size() : 1;
        if (n_caches > 1) {
            LLAMA_LOG_WARN("%s: %d expert caches (a cache a card): one save path for all of them is wrong; "
                            "per-card files need distinct paths, nothing written\n", __func__, n_caches);
        } else {
            const int n_layers = model_ != nullptr ? (int) model_->layers.size() : layers.back().il + 1;
            const int n_expert = layers[0].n_expert;
            std::vector<uint8_t> resident((size_t) n_layers * (size_t) n_expert, 0);
            std::vector<double>  heat    ((size_t) n_layers * (size_t) n_expert, 0.0);
            double counted = 0.0;
            for (const auto & L : layers) {
                const size_t base = (size_t) L.il * (size_t) n_expert;
                for (int e = 0; e < L.n_expert; ++e) {
                    resident[base + (size_t) e] = L.slot_of[e] >= 0 ? 1 : 0;
                    // the raw routing counts, one a routed pair (Strata's device counter)
                    heat[base + (size_t) e] = L.heat.empty() ? 0.0 : L.heat[e];
                    counted += heat[base + (size_t) e];
                }
            }
            if (book_.phase != 0) {
                std::vector<llama_expert_profile_swap> pending;
                pending.reserve(book_.pend.size());
                for (const auto & p : book_.pend) {
                    pending.push_back({p.layer, p.e_in, p.e_out});
                }
                llama_expert_profile_mark_pending(n_expert, resident, pending, book_.phase);
            }
            const char * rule_env = getenv("LLAMA_EXPERT_CACHE_SAVE_RULE");
            const bool cu = rule_env != nullptr && std::strcmp(rule_env, "cu") == 0;
            if (rule_env != nullptr && *rule_env != '\0' && !cu && std::strcmp(rule_env, "xe") != 0) {
                LLAMA_LOG_WARN("%s: LLAMA_EXPERT_CACHE_SAVE_RULE=%s: not xe or cu, the xe rule\n", __func__, rule_env);
            }
            // Strata's constant and its environment variable, mirrored
            double prior_entries = 800000.0;
            if (const char * pe = getenv("STRATA_PROFILE_PRIOR_ENTRIES")) {
                prior_entries = std::max(0.0, std::atof(pe));
            }
            const auto ranked = cu ? llama_expert_profile_rank(n_layers, n_expert, resident, heat, profile_ranked)
                                  : llama_expert_profile_rank_xe(n_layers, n_expert, heat, profile_ranked,
                                                                 prior_entries, 2.0);
            std::string err;
            if (llama_expert_profile_write(save, n_layers, n_expert, ranked, err)) {
                // WARN, not INFO: the units' default level drops INFO, and a save the
                // operator asked for must say it happened (Strata writes stderr plainly)
                LLAMA_LOG_WARN("%s: saved the expert profile to %s (%zu pairs, %s rule, %.0f routed entries counted on "
                               "the device%s)\n", __func__, save, ranked.size(), cu ? "cu" : "xe", counted,
                               cu ? "" : format(", prior %.0f routed entries", prior_entries).c_str());
            } else {
                LLAMA_LOG_WARN("%s: the expert profile was not saved: %s\n", __func__, err.c_str());
            }
        }
    }
    if (n_deferred > 0) {
        LLAMA_LOG_INFO("%s: %s: %llu swaps put off for want of a spare bank entry in their layer\n", __func__,
                       dev_ ? ggml_backend_dev_name(dev_) : "-", (unsigned long long) n_deferred);
    }
    if (n_multi_steps > 0) {
        LLAMA_LOG_INFO("%s: %s: %llu multi-token decode steps: %.1f routed pairs, %.1f distinct experts (%.1f in the bank) a step over its layers\n",
                       __func__, dev_ ? ggml_backend_dev_name(dev_) : "-", (unsigned long long) n_multi_steps, (double) n_multi_pairs / n_multi_steps,
                       (double) n_multi_unique / n_multi_steps, (double) n_multi_unique_bank / n_multi_steps);
    }
    {
        std::lock_guard<std::mutex> lk(g_registry_mutex);
        for (auto & L : layers) {
            g_registry.erase(L.src[2]);
        }
    }
    // Round 11, review item 2: drain once whenever the loan is armed, before
    // freeing ANYTHING. After a lend b_tail is null but the slot, bank and lend-bank
    // buffers below (and the USM banks the adapt and bank-stream queues may still
    // read) are freed; the old guard drained only when b_tail existed. With the
    // loan off (lend_bytes_ 0) nothing is added: the destructor is the loan-off
    // form, unchanged.
    if (lend_bytes_ != 0) {
        llama_loan_book::drain_now();
    }
    if (b_tail != nullptr) {
        // finding 7: the tail buffer is the loan's own, never a `bufs` entry --
        // that would free a pointer the loan already freed and leak the live one
        GGML_ASSERT(!llama_loan_book::listed(bufs, b_tail));
        ggml_backend_buffer_free(b_tail);
        b_tail = nullptr;
    }
    // round 17: a spare candidate's slot buffer (resident, never lent) and a
    // grown layer's lend bank (host USM) are not in `bufs` either
    for (ggml_backend_buffer_t b : spare_buf) {
        if (b != nullptr) ggml_backend_buffer_free(b);
    }
    for (auto b : bufs) {
        ggml_backend_buffer_free(b);
    }
    if (ctx) {
        ggml_free(ctx);
    }
    if (bank_ctx_) {
        ggml_free(bank_ctx_);
    }
    if (lend_ctx_) {
        ggml_free(lend_ctx_);
    }
    if (tail_ctx_) {
        ggml_free(tail_ctx_);
    }
    for (int fd : fds) {
        if (fd >= 0) {
            close(fd);
        }
    }
}

bool llama_expert_cache::read_expert(const llama_expert_cache_layer & L, int expert, int k, uint8_t * dst) const {
    size_t    left = L.nb2[k];
    off_t     off  = (off_t) (L.file_offs[k] + (size_t) expert * L.nb2[k]);
    const int fd   = fds[L.file_idx[k]];
    while (left > 0) {
        const ssize_t r = pread(fd, dst, left, off);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        dst += r; off += r; left -= (size_t) r;
    }
    // the bytes live in the slots and the bank: not a third time in the page cache
    posix_fadvise(fd, (off_t) (L.file_offs[k] + (size_t) expert * L.nb2[k]), (off_t) L.nb2[k], POSIX_FADV_DONTNEED);
    return true;
}

void llama_expert_cache::write_slices(ggml_tensor * t, int64_t i2, const uint8_t * data, size_t nb2) {
    if (set_slices) {
        set_slices(t, data, i2, 1);
    } else {
        ggml_backend_tensor_set(t, data, (size_t) i2 * nb2, nb2);
    }
}

// A slot's or bank entry's bytes against the file's (a converted layout is
// restored by get_tensor before the range is read); expert -1: zeros
bool llama_expert_cache::verify(const llama_expert_cache_layer & L, ggml_tensor * t, int k, int i2, int expert) const {
    std::vector<uint8_t> dev(L.nb2[k]), ref(L.nb2[k], 0);
    get_slices(t, dev.data(), i2, 1);
    if (expert >= 0 && !read_expert(L, expert, k, ref.data())) {
        return false;
    }
    return memcmp(dev.data(), ref.data(), L.nb2[k]) == 0;
}

// a layer's residency tables, its n_tab rows of tab_all from `r` on
void llama_expert_cache::fill_tables(const llama_expert_cache_layer & L, int32_t * r) const {
    const int64_t ne = tab_all->ne[1];
    int32_t * ts = r, * tb = r + ne, * tc = r + 2 * ne;
    int32_t * tl = L.tab_loan != nullptr ? r + 3 * ne + (L.tab_slot_dn != nullptr ? 2 * ne : 0) : nullptr;
    for (int e = 0; e < L.n_expert; ++e) {
        ts[e] = L.slot_of[e];   // -1: the kernels skip the pair in this branch
        tb[e] = L.bank_of[e];
        tc[e] = L.slot_of[e] >= 0 ? L.slot_of[e] : (L.bank_of[e] >= 0 ? L.n_slots + L.bank_of[e] : -1);
        // the loan's lend-bank branch (read only while lent); round 15: the
        // MUTATE=5 override lives here, so every producer of this row (the upload,
        // check_tables and the card's flush) sees the same shifted image
        tl[e] = L.loan_of[e] < 0 ? -1 : llama_loan_book::mutate_loan_bank_row(L.lend_row_mutate, L.loan_of[e], L.n_slots);
    }
    if (L.tab_slot_dn) {
        int32_t * ts2 = r + 3 * ne, * tb2 = r + 4 * ne;
        for (int e = 0; e < L.n_expert; ++e) {
            ts2[e] = L.slot_of[e] >= 0 ? L.slot_of[e] : L.n_slots;
            tb2[e] = L.bank_of[e] >= 0 ? L.bank_of[e] : L.n_bank;
        }
    }
}

// a layer's tables into the host copy of tab_all; flush_tables writes them
void llama_expert_cache::upload_tables(llama_expert_cache_layer & L) {
    fill_tables(L, tab_host.data() + L.tab_row * tab_all->ne[1]);
    tab_lo = std::min(tab_lo, L.tab_row);
    tab_hi = std::max(tab_hi, L.tab_row + L.n_tab);
}

// the rows upload_tables changed, in one write without a wait (the host copy
// is next changed by a later adapt step, after a graph has passed)
void llama_expert_cache::flush_tables() {
    if (tab_lo >= tab_hi) {
        return;
    }
    const int32_t * src = tab_host.data() + tab_lo * tab_all->ne[1];
    if (set_slices) {
        set_slices(tab_all, src, tab_lo, tab_hi - tab_lo);
    } else {
        ggml_backend_tensor_set(tab_all, src, (size_t) tab_lo * tab_all->nb[2], (size_t) (tab_hi - tab_lo) * tab_all->nb[2]);
    }
    tab_lo = INT32_MAX;
    tab_hi = 0;
}

// the tables on the card (a blocking read) against slot_of / bank_of: the rows that differ
int llama_expert_cache::check_tables() const {
    const int64_t ne = tab_all->ne[1], n_rows = tab_all->ne[2];
    std::vector<int32_t> want((size_t) (n_rows * ne), 0), dev((size_t) (n_rows * ne), 0);
    for (const auto & L : layers) {
        fill_tables(L, want.data() + L.tab_row * ne);
    }
    get_slices(tab_all, dev.data(), 0, n_rows);
    int bad = 0;
    for (const auto & L : layers) {
        for (int k = 0; k < L.n_tab; ++k) {
            const size_t o = (size_t) ((L.tab_row + k) * ne);
            bad += memcmp(want.data() + o, dev.data() + o, (size_t) ne * sizeof(int32_t)) != 0;
        }
    }
    return bad;
}

// Layer il as init() caches it on `dev`: its experts in host buffers, the
// layer itself on `dev` (a GPU), types whose kernels take the cache, its
// tensors' bytes in a GGUF file. `out` (when not null) gets the layer.
static bool qualify(const llama_model & model, const llama_model_loader & ml, int il, ggml_backend_dev_t dev, bool warn,
                    llama_expert_cache_layer * out) {
    const llama_layer & ly = model.layers[il];
    ggml_tensor * t[3] = { ly.ffn_gate_exps, ly.ffn_up_exps, ly.ffn_down_exps };
    if (!t[0] || !t[1] || !t[2] || ly.ffn_gate_up_exps || !on_cpu(t[0]) || !on_cpu(t[1]) || !on_cpu(t[2])) {
        return false;
    }
    ggml_backend_dev_t d = model.dev_layer(il);
    if (d == nullptr || d != dev || ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;   // another card's layer: that card's cache
    }
    // the types whose kq kernels skip a pair (gate/up), or whose down
    // projection takes the zero-entry tables (Q8_0)
    // and the low-bit types of patches 0022 (IQ2_XXS, IQ2_S, IQ1_M, Q2_0, IQ3_S) and 0030 (IQ2_XS)
    auto kq_skip = [](ggml_type ty) {
        return ty == GGML_TYPE_IQ3_XXS || ty == GGML_TYPE_IQ4_XS || ty == GGML_TYPE_IQ4_NL || ty == GGML_TYPE_IQ2_XXS ||
               ty == GGML_TYPE_IQ2_S || ty == GGML_TYPE_IQ1_M || ty == GGML_TYPE_Q2_0 || ty == GGML_TYPE_IQ3_S ||
               ty == GGML_TYPE_IQ2_XS;
    };
    if (!kq_skip(t[0]->type) || !kq_skip(t[1]->type) || !(kq_skip(t[2]->type) || t[2]->type == GGML_TYPE_Q8_0)) {
        if (warn) {
            LLAMA_LOG_WARN("%s: layer %d: experts %s/%s/%s, not cached (IQ and low-bit types; Q8_0 down)\n", __func__, il,
                           ggml_type_name(t[0]->type), ggml_type_name(t[1]->type), ggml_type_name(t[2]->type));
        }
        return false;
    }
    llama_expert_cache_layer L;
    L.il       = il;
    L.n_expert = (int) t[2]->ne[2];
    if (L.n_expert > 512) {   // the OpenCL MoE GEMM takes up to 512 matrices
        return false;
    }
    for (int k = 0; k < 3; ++k) {
        auto it = ml.weights_map.find(ggml_get_name(t[k]));
        if (it == ml.weights_map.end() || it->second.idx >= ml.file_paths.size() || ml.file_paths[it->second.idx].empty() ||
            t[k]->ne[2] != L.n_expert || !ggml_is_contiguous(t[k])) {
            return false;
        }
        L.src[k]       = t[k];
        L.file_idx[k]  = it->second.idx;
        L.file_offs[k] = it->second.offs;
        L.nb2[k]       = t[k]->nb[2];
    }
    if (out) {
        *out = std::move(L);
    }
    return true;
}

bool llama_expert_cache::has_layers(const llama_model & model, const llama_model_loader & ml, ggml_backend_dev_t dev) {
    for (int il = 0; il < (int) model.layers.size(); ++il) {
        if (qualify(model, ml, il, dev, false, nullptr)) {
            return true;
        }
    }
    return false;
}

bool llama_expert_cache::init(llama_model & model, const llama_model_loader & ml, ggml_backend_dev_t dev, size_t budget, const char * profile,
                              size_t lend_bytes) {
    if (const char * m = getenv("LLAMA_EXPERT_CACHE_MUTATE")) {
        mutate = atoi(m);
    }
    if (const char * m = getenv("LLAMA_SLOT_LOAN_MUTATE")) {
        mutate_loan = atoi(m);   // the loan's red cases: 1 a slot left unrefilled, 2 a table row lost
    }
    // finding 8's flag rule (third review), in llama-loan-book.h so a unit case
    // sees it: LLAMA_SLOT_LOAN_MIB is read only when no parameter was set
    // (llama_model_default_params' sentinel); an explicit 0 is off, so a tool's
    // own flag cannot be overridden by the environment.
    {
        const char * v = getenv("LLAMA_SLOT_LOAN_MIB");
        lend_bytes = llama_loan_book::resolve_lend_bytes(lend_bytes, LLAMA_SLOT_LOAN_BYTES_ENV,
                                                         v != nullptr ? atoll(v) : 0);
    }
    // the adaptation's constants (Strata's by default), for measurement
    if (const char * v = getenv("LLAMA_EXPERT_CACHE_MAX_SWAPS")) max_swaps = atoi(v);
    if (const char * v = getenv("LLAMA_EXPERT_CACHE_MIN_GAIN"))  min_gain  = (float) atof(v);
    if (const char * v = getenv("LLAMA_EXPERT_CACHE_MIN_USAGE")) min_usage = (float) atof(v);
    if (const char * v = getenv("LLAMA_EXPERT_CACHE_DECAY"))     decay     = (float) atof(v);
    if (const char * a = getenv("LLAMA_EXPERT_CACHE_ADAPT_EVERY")) {
        adapt_every = (uint32_t) atoi(a);   // 0: the profile's placement, never adapted
    }
    dev_ = dev;
    model_ = &model;
    layers.reserve(model.layers.size());   // the registry holds pointers into it
    for (int il = 0; dev != nullptr && il < (int) model.layers.size(); ++il) {
        llama_expert_cache_layer L;
        if (!qualify(model, ml, il, dev, true, &L)) {
            continue;
        }
        L.owner = this;
        L.slot_of.assign(L.n_expert, -1);
        L.bank_of.assign(L.n_expert, -1);
        L.usage.assign(L.n_expert, 0.0f);
        // the save's raw routing counts, Strata's --expert-profile-save allocates
        // its counter only then; off, this vector stays empty and count() skips it
        if (const char * save = getenv("LLAMA_EXPERT_CACHE_SAVE"); save != nullptr && *save != '\0') {
            L.heat.assign(L.n_expert, 0.0);
        }
        layers.push_back(std::move(L));
    }
    if (layers.empty()) {
        const llama_layer & l0 = model.layers[0];
        ggml_backend_dev_t  d0 = model.dev_layer(0);
        LLAMA_LOG_WARN("%s: no MoE layer on %s with its experts in host memory (layer 0: gate %s/%s, down %s, gate_up %d, device %s)\n",
                       __func__, dev ? ggml_backend_dev_name(dev) : "-",
                       l0.ffn_gate_exps ? ggml_get_name(l0.ffn_gate_exps) : "-",
                       l0.ffn_gate_exps && l0.ffn_gate_exps->buffer ? ggml_backend_buffer_name(l0.ffn_gate_exps->buffer) : "-",
                       l0.ffn_down_exps && l0.ffn_down_exps->buffer ? ggml_backend_buffer_name(l0.ffn_down_exps->buffer) : "-",
                       l0.ffn_gate_up_exps != nullptr, d0 ? ggml_backend_dev_name(d0) : "-");
        return false;
    }
    fds.assign(ml.file_paths.size(), -1);
    for (auto & L : layers) {
        for (int k = 0; k < 3; ++k) {
            int & fd = fds[L.file_idx[k]];
            if (fd < 0) {
                fd = open(ml.file_paths[L.file_idx[k]].c_str(), O_RDONLY);
                if (fd < 0) {
                    LLAMA_LOG_WARN("%s: cannot open %s\n", __func__, ml.file_paths[L.file_idx[k]].c_str());
                    return false;
                }
            }
        }
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    set_slices  = (decltype(set_slices)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_set_tensor_slices");
    get_tensors = (decltype(get_tensors)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_get_tensors");
    get_tensors_async = (decltype(get_tensors_async)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_get_tensors_async");
    {
        const char * e = getenv("LLAMA_EXPERT_CACHE_ASYNC_IDS");
        async_ids = get_tensors_async != nullptr && (e == nullptr || atoi(e) != 0);
    }
    get_slices  = (decltype(get_slices)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_get_tensor_slices");
    exchange    = (decltype(exchange)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_exchange_slices");
    lend_write  = (decltype(lend_write)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_lend_write");
    lend_zero   = (decltype(lend_zero))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_lend_zero");
    lend_read   = (decltype(lend_read))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_lend_read");
    lend_perturb = (decltype(lend_perturb)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_lend_perturb");
    bank_stream_ring = (decltype(bank_stream_ring)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_bank_stream_ring");
    {
        // round 8, drain-before-exit: the poison policy finishes every queue the
        // process has work on before _Exit (the card's xen page fault and engine
        // reset on the exit-70 arms); the backend's own drain, registered once.
        auto drain = (llama_loan_book::DrainFn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_drain");
        if (drain != nullptr) {
            llama_loan_book::register_drain(drain);
        }
    }
    set_alloc_host = (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_set_alloc_host");
    auto reserve_gather = (void (*)(ggml_backend_dev_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_reserve_gather");
    auto pair_bank      = (void (*)(const ggml_tensor *, const ggml_tensor *, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_pair_bank");
    usm_ptr = (decltype(usm_ptr)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_usm_ptr");
    if (!set_slices || !get_tensors || !get_slices || !exchange || !set_alloc_host || !reserve_gather || !pair_bank) {
        LLAMA_LOG_WARN("%s: %s lacks the expert cache's entry points (contrib/llama.cpp patch 0021)\n", __func__, ggml_backend_dev_name(dev));
        return false;
    }
    // round 7: the loan's refill by region copies from the lend bank, which needs
    // the three entry points of patch 0068; a backend without them would have to
    // stage and convert on the host, the cost this round removes.
    if (!lend_write || !lend_zero || !lend_read) {
        LLAMA_LOG_WARN("%s: %s lacks the loan's refill entry points (lend_write/lend_zero/lend_read, patch 0068)\n", __func__,
                       ggml_backend_dev_name(dev));
        return false;
    }
    adapt_copy   = (decltype(adapt_copy))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_adapt_copy");
    adapt_fence  = (decltype(adapt_fence))  ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_adapt_fence");
    adapt_record = (decltype(adapt_record)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_adapt_record");
    adapt_done   = (decltype(adapt_done))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_adapt_done");
    adapt_finish = (decltype(adapt_finish)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_adapt_finish");
    {
        const char * e = getenv("LLAMA_EXPERT_CACHE_NB");
        nb = adapt_copy && adapt_fence && adapt_record && adapt_done && adapt_finish && (e == nullptr || atoi(e) != 0);
    }
    // spare bank entries a layer: Strata reserves 96 exchange buffers over all
    // layers (expert_source.cpp:1332-1357, min(adapt_swaps, 96)); here they
    // live in each layer's bank tensors, 4 a layer by default
    int n_spare = 0;
    if (nb) {
        const char * e = getenv("LLAMA_EXPERT_CACHE_SPARES");
        n_spare = e != nullptr ? std::max(1, atoi(e)) : 4;
    }

    // the ranking: Strata's profile, else expert-major round robin
    std::vector<int> layer_of(model.layers.size(), -1);
    for (int i = 0; i < (int) layers.size(); ++i) {
        layer_of[layers[i].il] = i;
    }
    std::vector<std::pair<int, int>> ranked;
    const int n_expert = layers[0].n_expert;
    bool from_profile = profile && *profile && read_profile(profile, (int) model.layers.size(), n_expert, ranked);
    if (profile && *profile && !from_profile) {
        LLAMA_LOG_WARN("%s: profile %s unreadable or for another geometry; round robin\n", __func__, profile);
    }
    // the profile the run started from, for the save's prior step; empty for
    // round robin (Strata's save test's no-prior order)
    profile_ranked = from_profile ? ranked : std::vector<std::pair<int, int>>();
    if (!from_profile) {
        for (int e = 0; e < n_expert; ++e) {
            for (auto & L : layers) {
                ranked.emplace_back(L.il, e);
            }
        }
    }
    // slots in that order until the budget is spent, each of its layer's size
    // (Strata generate.cpp:2691-2709), each layer's zero slot counted; every
    // layer keeps at least one slot (its bank then fits 512 matrices with its
    // zero entry)
    std::vector<std::vector<int>> chosen(layers.size());
    size_t used = 0;
    for (auto & L : layers) {
        used += L.nb2[0] + L.nb2[1] + L.nb2[2];
    }
    for (auto [il, e] : ranked) {
        if (il < 0 || il >= (int) layer_of.size() || layer_of[il] < 0 || e < 0 || e >= n_expert) {
            continue;
        }
        const int i = layer_of[il];
        auto &    L = layers[i];
        const size_t b = L.nb2[0] + L.nb2[1] + L.nb2[2];
        if (used + b > budget) {
            break;
        }
        if (L.slot_of[e] == -2 || (int) chosen[i].size() + 1 >= L.n_expert) {
            continue;
        }
        L.slot_of[e] = -2;   // taken (set to its slot below)
        chosen[i].push_back(e);
        used += b;
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        auto & L = layers[i];
        if (chosen[i].empty()) {
            chosen[i].push_back(0);
            L.slot_of[0] = -2;
        }
        L.n_slots = (int) chosen[i].size();
        L.dirty.reset((size_t) L.n_slots);   // round 21: one flag a slot, set by a swap
        L.n_bank  = L.n_expert - L.n_slots;
        // the bank with its zero entry and spares within the GEMM's 512 matrices
        // (ggml_cl_mul_mat_id_kq refuses more; a layer with few slots gets fewer)
        L.n_spare = std::min(n_spare, std::max(0, 512 - (L.n_bank + 1)));
        if (L.n_spare < n_spare) {
            LLAMA_LOG_WARN("%s: layer %d: %d spare bank entries (%d slots, the bank's 512-matrix cap)\n", __func__, L.il, L.n_spare, L.n_slots);
        }
        L.expert_of.assign(L.n_slots + 1, -1);
        L.loan_of.assign(L.n_expert, -1);
        L.bexpert_of.assign(L.n_bank + 1 + L.n_spare, -1);
        for (int x = L.n_bank + L.n_spare; x > L.n_bank; --x) {
            L.spare.push_back(x);   // taken from the back: the lowest first
        }
        for (int s = 0; s < L.n_slots; ++s) {
            L.expert_of[s]          = chosen[i][s];
            L.slot_of[chosen[i][s]] = s;
        }
        int b = 0;
        for (int e = 0; e < L.n_expert; ++e) {
            if (L.slot_of[e] < 0) {
                L.bexpert_of[b] = e;
                L.bank_of[e]    = b++;
            }
        }
    }

    // tensors: slots and tables on the card, the banks in host memory
    // the tables: each layer's rows of tab_all (slot, bank, comb, the _dn pair,
    // then the loan row)
    int    n_rows = 0;
    size_t n_ctx  = 2;   // tab_all, tab_id
    for (auto & L : layers) {
        GGML_ASSERT(L.n_expert == n_expert);
        L.tab_row = n_rows;
        L.n_tab   = (L.src[2]->type == GGML_TYPE_Q8_0 ? 5 : 3) + 1;   // the last row: tab_loan
        n_rows   += L.n_tab;
        n_ctx    += 3 + L.n_tab;   // slots, table views
        for (int k = 0; k < 3; ++k) {
            n_ctx += L.src[k]->type == GGML_TYPE_Q8_0;   // a Q8_0 bank lives on the card
        }
    }
    // from scratch: no layer is lendable until the selection below names it.
    // A layer is lendable when its three banks are host memory AND the region
    // copy is exact for every one of their types (round 8, review item 3). The
    // USM bank stores most types raw, while set_tensor builds a specialised extra
    // for Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/MXFP4/Q4_K/Q5_K/Q6_K and converts BF16 to
    // f16 on the card: a region copy there writes GGUF bytes through a plain
    // extra (the old crash) or feeds f16 kernels raw BF16 (the bank check would
    // compare the wrong bytes with themselves and pass). Plain-no-extra types,
    // the planar set and IQ4_NL are byte-identical on both sides (IQ4_NL split
    // on both -- the bank's usm_split, the tail slot's rebuilt extra).
    for (auto & L : layers) {
        L.lendable = false;
    }
    // the lend set: computed per layer from scratch over the tail (creation
    // order), whole layers while their slot bytes fit lend_bytes. The separate
    // tail buffer holds a tensor list, so a non-contiguous set is fine (a
    // non-lendable layer in the tail is skipped). Nothing chosen means no loan:
    // the tail and lend contexts are not created and every slot tensor stays in
    // the ordinary buffer (the crash was a NULL context for marked layers).
    {
        std::vector<llama_loan_book::LendCandidate> cand(layers.size());
        auto copy_ok = [](ggml_type ty) { return llama_loan_book::region_copyable(ty); };
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto & L = layers[i];
            cand[i].lendable = lend_bytes > 0 && usm_ptr != nullptr &&
                               copy_ok(L.src[0]->type) && copy_ok(L.src[1]->type) && copy_ok(L.src[2]->type);
            cand[i].bytes = cand[i].lendable
                                ? (int64_t) L.n_slots * (int64_t) (L.nb2[0] + L.nb2[1] + L.nb2[2]) : 0;
        }
        const auto pick = llama_loan_book::choose_lend_layers(cand, lend_bytes);
        cand_       = cand;
        lend_bytes_ = (size_t) pick.bytes;
        lend_set_   = pick.chosen;
        init_set_   = pick.chosen;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].candidate = cand[i].lendable;
        }
        for (int i : lend_set_) {
            layers[(size_t) i].lendable = true;
            lend_slots_ += layers[(size_t) i].n_slots;
        }
        if (lend_bytes > 0 && lend_bytes_ == 0) {
            LLAMA_LOG_WARN("%s: the slot loan lends nothing: %s (%.2f GiB asked)\n", __func__,
                           pick.why != nullptr ? pick.why : "no lendable layer", lend_bytes / 1073741824.0);
            // round 17: no pick means the tail and lend contexts are not created;
            // no candidate may aim a tensor at them (the first card run's crash)
            for (auto & L : layers) L.candidate = false;
            cand_.clear();
        }
    }
    ggml_init_params ip = { ggml_tensor_overhead() * n_ctx, nullptr, true };
    ctx       = ggml_init(ip);
    ip.mem_size = ggml_tensor_overhead() * layers.size() * 3;
    bank_ctx_ = ggml_init(ip);
    if (lend_bytes_ > 0) {
        ip.mem_size = ggml_tensor_overhead() * (size_t) layers.size() * 3;   // only the lend set is created
        lend_ctx_  = ggml_init(ip);   // the lend bank: host USM, read over the link like the bank
        tail_ctx_  = ggml_init(ip);   // the tail layers' slot tensors: their own card buffer
    }
    tab_all = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 1, n_expert, n_rows);
    ggml_set_name(tab_all, "expert_tables");
    tab_host.assign((size_t) n_rows * n_expert, 0);
    auto tab_view = [&](int row) {
        return ggml_view_2d(ctx, tab_all, 1, n_expert, tab_all->nb[1], (size_t) row * tab_all->nb[2]);
    };
    for (auto & L : layers) {
        // a loan's tail layers' slot tensors in their own buffer, the rest in bslots:
        // the loan frees that buffer and the refill re-allocates it (ggml-backend's
        // buft_alloc_buffer_n; no byte range is computed anywhere)
        ggml_context * sctx = L.candidate ? tail_ctx_ : ctx;
        for (int k = 0; k < 3; ++k) {
            L.slot[k] = ggml_new_tensor_3d(sctx, L.src[k]->type, L.src[k]->ne[0], L.src[k]->ne[1], L.n_slots + 1);
            ggml_format_name(L.slot[k], "%s.slots", ggml_get_name(L.src[k]));
            if (L.lendable) tail_slots.push_back(L.slot[k]);   // the ask's pick: b_tail's tensors
            if (L.candidate) {
                // the rows the slots hold, readable while they are lent; a bank
                // suffix so the backend's cache-bank rules apply (an expert cache's
                // bank tensor by its name, ggml-opencl.cpp's cache_src)
                L.lend[k] = ggml_new_tensor_3d(lend_ctx_, L.src[k]->type, L.src[k]->ne[0], L.src[k]->ne[1], L.n_slots);
                ggml_format_name(L.lend[k], "%s.lend.bank", ggml_get_name(L.src[k]));
            }
            // the bank in USM host memory, read by the kq kernels over the link;
            // a Q8_0 bank (the general MoE path, no pointer arguments) on the card
            ggml_context * bctx = L.src[k]->type == GGML_TYPE_Q8_0 ? ctx : bank_ctx_;
            L.bank[k] = ggml_new_tensor_3d(bctx, L.src[k]->type, L.src[k]->ne[0], L.src[k]->ne[1], L.n_bank + 1 + L.n_spare);
            ggml_format_name(L.bank[k], "%s.bank", ggml_get_name(L.src[k]));
        }
        L.tab_slot = tab_view(L.tab_row);
        L.tab_bank = tab_view(L.tab_row + 1);
        L.tab_comb = tab_view(L.tab_row + 2);
        ggml_format_name(L.tab_comb, "blk.%d.expert_slot_or_bank", L.il);
        if (L.n_tab == 6 || L.n_tab == 4) {
            const int base = L.src[2]->type == GGML_TYPE_Q8_0 ? 3 : 0;
            if (base) {
                L.tab_slot_dn = tab_view(L.tab_row + 3);
                L.tab_bank_dn = tab_view(L.tab_row + 4);
                ggml_format_name(L.tab_slot_dn, "blk.%d.expert_slot_of_down", L.il);
                ggml_format_name(L.tab_bank_dn, "blk.%d.expert_bank_of_down", L.il);
            }
            // round 15 (R55): the loan row is the layer's LAST row. The round-14
            // form added base twice (tab_row + 3 + 3 = +6 of a six-row layer) while
            // fill_tables and the mutation write row +5, so a last cached layer with
            // a Q8_0 down asserted at load, loan on or off.
            L.tab_loan = tab_view(L.tab_row + 3 + (L.tab_slot_dn != nullptr ? 2 : 0));
            ggml_format_name(L.tab_loan, "blk.%d.expert_lend_entry", L.il);
        }
        ggml_format_name(L.tab_slot, "blk.%d.expert_slot_of", L.il);
        ggml_format_name(L.tab_bank, "blk.%d.expert_bank_of", L.il);
    }
    tab_id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
    ggml_set_name(tab_id, "expert_id");
    ggml_backend_buffer_type_t buft   = ggml_backend_dev_buffer_type(dev);
    tail_buft = buft;
    ggml_backend_buffer_t      bslots = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    // the loan's tail slot tensors in their own buffer: the loan frees it and
    // the refill re-allocates it (buft_alloc_buffer_n, the tensor list, no range).
    // Round 17: the tail context holds every candidate's slot tensors, but the
    // buffer is the ask's pick only; the other candidates are allocated one small
    // buffer a layer below, so growth can free exactly the layer it lends.
    std::vector<ggml_tensor *> pick_slots = tail_slots;
    std::vector<ggml_tensor *> pick_lends;
    for (int i : init_set_) {
        for (int k = 0; k < 3; ++k) pick_lends.push_back(layers[(size_t) i].lend[k]);
    }
    set_alloc_host(false);
    ggml_backend_buffer_t      b_lend = tail_ctx_ != nullptr && !pick_slots.empty()
                                          ? ggml_backend_buft_alloc_buffer_n(tail_buft, pick_slots.data(), (int) pick_slots.size())
                                          : nullptr;
    set_alloc_host(true);
    ggml_backend_buffer_t      bbank  = ggml_backend_alloc_ctx_tensors_from_buft(bank_ctx_, buft);
    ggml_backend_buffer_t      b_lendbank = lend_ctx_ != nullptr && !pick_lends.empty()
                                              ? ggml_backend_buft_alloc_buffer_n(tail_buft, pick_lends.data(), (int) pick_lends.size())
                                              : nullptr;
    set_alloc_host(false);
    if (bslots == nullptr || bbank == nullptr || (tail_ctx_ != nullptr && b_lend == nullptr) ||
        (lend_ctx_ != nullptr && b_lendbank == nullptr)) {
        LLAMA_LOG_WARN("%s: could not allocate the slots (%.2f GiB on %s), the bank in host memory, or the loan's tail slots\n", __func__,
                       used / 1073741824.0, ggml_backend_dev_name(dev));
        if (bslots) bufs.push_back(bslots);
        if (b_lend) bufs.push_back(b_lend);
        if (b_lendbank) bufs.push_back(b_lendbank);
        if (bbank)  bufs.push_back(bbank);
        return false;
    }
    bufs = { bslots, bbank };
    // round 17: a candidate outside the ask's pick keeps its slot tensors resident
    // in one small buffer a layer; the growth frees that layer's buffer when it
    // lends it (the refill allocates it again)
    spare_buf.assign(layers.size(), nullptr);
    for (size_t i = 0; i < layers.size(); ++i) {
        if (!layers[i].candidate || layers[i].lendable) continue;
        ggml_tensor * ts[3] = { layers[i].slot[0], layers[i].slot[1], layers[i].slot[2] };
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer_n(tail_buft, ts, 3);
        if (b == nullptr) {
            for (size_t j = 0; j < spare_buf.size(); ++j) {
                if (spare_buf[j] != nullptr) {
                    ggml_backend_buffer_free(spare_buf[j]);
                    spare_buf[j] = nullptr;   // round 18: the destructor frees every non-null entry
                }
            }
            // bslots/bbank are already in `bufs` (the destructor frees them); the
            // two loan buffers are not yet, so they are freed here, once
            if (b_lend) ggml_backend_buffer_free(b_lend);
            if (b_lendbank) ggml_backend_buffer_free(b_lendbank);
            LLAMA_LOG_WARN("%s: could not allocate a candidate tail layer's slot buffer\n", __func__);
            return false;
        }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        spare_buf[i] = b;
    }
    // finding 7: the tail buffer is taken out of (never added to) the list the
    // destructor walks: the loan frees it and the refill re-allocates it
    llama_loan_book::take_tail(bufs, b_tail, b_lend);
    if (b_lendbank != nullptr) {
        ggml_backend_buffer_set_usage(b_lendbank, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.push_back(b_lendbank);
    }
    ggml_backend_buffer_set_usage(bslots, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(bbank, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (b_lend != nullptr) ggml_backend_buffer_set_usage(b_lend, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    n_bytes      = ggml_backend_buffer_get_size(bslots);
    if (b_lend != nullptr) n_bytes += ggml_backend_buffer_get_size(b_lend);
    for (ggml_backend_buffer_t b : spare_buf) {
        if (b != nullptr) n_bytes += ggml_backend_buffer_get_size(b);
    }
    n_bank_bytes = ggml_backend_buffer_get_size(bbank);
    if (b_lendbank != nullptr) n_bank_bytes += ggml_backend_buffer_get_size(b_lendbank);
    // patch 0077: the alias check at the cache init, over every slot tensor in its slot
    // buffer (the ask's bslots and b_lend, and the candidates' per-layer spares)
    {
        std::vector<ggml_tensor *> alias_slots;
        for (const auto & L : layers) {
            for (int k = 0; k < 3; ++k) {
                ggml_tensor * t = L.slot[k];
                if (t == nullptr) continue;
                bool in_slot_buf = t->buffer == bslots || (b_lend != nullptr && t->buffer == b_lend);
                for (size_t j = 0; !in_slot_buf && j < spare_buf.size(); ++j) {
                    in_slot_buf = spare_buf[j] != nullptr && t->buffer == spare_buf[j];
                }
                if (in_slot_buf) alias_slots.push_back(t);
            }
        }
        llama_cache_alias_check(alias_slots);   // warn and proceed
    }
    // round 20: one stream-ring slot holds one bank tensor whole; the largest
    // bank tensor is the cold ring's first-touch slot content
    for (const auto & L : layers) {
        for (int k = 0; k < 3; ++k) {
            if (L.bank[k] != nullptr) max_bank_tensor_ = std::max(max_bank_tensor_, ggml_nbytes(L.bank[k]));
        }
    }
    // the gather mirror (the largest USM bank tensor, IQ4_NL's scales apart)
    // on the card now, not inside the first request
    {
        size_t q = 0, d = 0;
        // the lend bank's rows count too: a lent layer's lend branch is gathered
        // like a bank branch below the GEMM threshold (a 1-token reserve's tg graph)
        for (auto & L : layers) {
            for (int k = 0; k < 3; ++k) {
                ggml_tensor * ts[2] = { L.bank[k], L.candidate ? L.lend[k] : nullptr };
                for (ggml_tensor * t : ts) {
                    if (t == nullptr || t->type == GGML_TYPE_Q8_0) continue;
                    const size_t n = (size_t) t->ne[2];
                    if (t->type == GGML_TYPE_IQ4_NL) {
                        const size_t bps = (size_t) L.src[k]->ne[0] * L.src[k]->ne[1] / 32;
                        q = std::max(q, n * bps * 16);
                        d = std::max(d, n * bps * 2);
                    } else {
                        q = std::max(q, n * L.nb2[k]);
                    }
                }
            }
        }
        reserve_gather(dev, q, d);
        n_bytes += q + d;
    }
    // the single branch: each kq slot tensor knows its bank and the split
    for (auto & L : layers) {
        for (int k = 0; k < 3; ++k) {
            if (L.slot[k]->type != GGML_TYPE_Q8_0) {
                pair_bank(L.slot[k], L.bank[k], L.n_slots);
            }
        }
    }
    // patch 0056 (LLAMA_EXPERT_CACHE_CPU_MISSES=1): a share of decode's missed experts
    // on the CPU, Strata's doorbell tier; the backend computes them from the bank in
    // host memory while the card computes the hits and reads the other misses (its
    // multi-token rows, or for other types ggml-cpu's dot products passed in here)
    if (const char * cm = getenv("LLAMA_EXPERT_CACHE_CPU_MISSES"); cm != nullptr && atoi(cm) != 0) {
        typedef void (*miss_layer_t)(const ggml_tensor * const *, const ggml_tensor * const *, int, int, float,
                                     void * const *, const int *, void * const *, int);
        auto miss_layer = (miss_layer_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_miss_layer");
        const char * te = getenv("LLAMA_EXPERT_CACHE_CPU_MISS_THREADS");
        const int n_th = te ? std::max(1, atoi(te)) : 8;   // 8 measured as 10 and 12
        int n_reg = 0;
        for (auto & L : layers) {
            bool ok = miss_layer != nullptr;
            void * vd[3] = {}; int vt[3] = {}; void * ff[3] = {};
            for (int k = 0; k < 3 && ok; ++k) {
                const auto * tr = ggml_get_type_traits_cpu(L.slot[k]->type);
                ok = L.slot[k]->type != GGML_TYPE_Q8_0 && tr != nullptr && tr->vec_dot != nullptr &&
                     ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float != nullptr;
                if (ok) { vd[k] = (void *) tr->vec_dot; vt[k] = (int) tr->vec_dot_type; ff[k] = (void *) ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float; }
            }
            if (!ok) continue;
            const float c = model.hparams.swiglu_clamp_exp[L.il];
            const ggml_tensor * sl[3] = { L.slot[0], L.slot[1], L.slot[2] };
            const ggml_tensor * bk[3] = { L.bank[0], L.bank[1], L.bank[2] };
            miss_layer(sl, bk, L.n_slots, L.n_expert, c > 1e-6f ? c : 0.0f, vd, vt, ff, n_th);
            ++n_reg;
        }
        LLAMA_LOG_INFO("%s: the misses of %d of %zu layers on the CPU (%d threads)\n", __func__, n_reg, layers.size(), n_th);
    }

    // fill: each tensor whole (a converted layout is set up by its first write)
    const int64_t t_fill = ggml_time_us();
    int64_t n_slots_all = 0, lo = INT64_MAX, hi = 0;
    for (auto & L : layers) {
        for (int k = 0; k < 3; ++k) {
            for (int part = 0; part < 2; ++part) {
                const std::vector<int32_t> & who = part == 0 ? L.expert_of : L.bexpert_of;   // -1: zeros (the zero slot / entry, spares)
                ggml_tensor * t = part == 0 ? L.slot[k] : L.bank[k];
                std::vector<uint8_t> host(who.size() * L.nb2[k], 0);
                for (size_t s = 0; s < who.size(); ++s) {
                    if (who[s] >= 0 && !read_expert(L, who[s], k, host.data() + s * L.nb2[k])) {
                        LLAMA_LOG_WARN("%s: read failed (layer %d expert %d)\n", __func__, L.il, who[s]);
                        return false;
                    }
                }
                ggml_backend_tensor_set(t, host.data(), 0, host.size());
                if (part == 0 && L.lendable) {
                    // the loan's lend bank holds the same rows from the same read:
                    // row s = expert_of[s], so the refill is one range a tensor
                    ggml_backend_tensor_set(L.lend[k], host.data(), 0, (size_t) L.n_slots * L.nb2[k]);
                }
            }
        }
        upload_tables(L);
        n_slots_all += L.n_slots;
        lo = std::min<int64_t>(lo, L.n_slots);
        hi = std::max<int64_t>(hi, L.n_slots);
    }
    flush_tables();
    LLAMA_LOG_INFO("%s: table check, %s: %d of %d rows differ (written as one)\n", __func__, ggml_backend_dev_name(dev),
                   check_tables(), (int) tab_all->ne[2]);
    {
        std::vector<int32_t> id(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            id[e] = e;
        }
        ggml_backend_tensor_set(tab_id, id.data(), 0, id.size() * sizeof(int32_t));
    }
    // the copied bytes are exact: every slot of the first layer and bank
    // entries at both ends, each tensor
    {
        auto & L   = layers[0];
        int    bad = 0, n = 0;
        for (int k = 0; k < 3; ++k) {
            for (int s = 0; s <= L.n_slots; ++s, ++n) {
                bad += verify(L, L.slot[k], k, s, L.expert_of[s]) ? 0 : 1;
            }
            for (int b : { 0, 1, L.n_bank - 1, L.n_bank }) {
                if (b >= 0 && b <= L.n_bank) {
                    bad += verify(L, L.bank[k], k, b, L.bexpert_of[b]) ? 0 : 1;
                    ++n;
                }
            }
        }
        LLAMA_LOG_WARN("%s: copy check, %s, layer %d: %d of %d slots and bank entries differ from the file\n", __func__,
                       ggml_backend_dev_name(dev), L.il, bad, n);
        if (bad) {
            return false;
        }
    }
    // Round 7 (design §15): the refill's default check compares the slots with the
    // lend bank instead of re-reading the GGUF a loan, so the bank's provenance to
    // the file is proven here, once: every lendable layer's bank rows against a
    // fresh read of the file, through the same conversion path the refill's file
    // mode would use. This checks the bank at load; it does not convert what a
    // refill copies. LLAMA_SLOT_LOAN_VERIFY_LOAD=0 skips it (a load-time timing
    // arm); the campaign's gate arms leave it on.
    {
        static const bool lend_check = [] {
            const char * e = getenv("LLAMA_SLOT_LOAN_VERIFY_LOAD");
            return e == nullptr || atoi(e) != 0;
        }();
        if (lend_check && !lend_set_.empty()) {
            const int64_t t0 = ggml_time_us();
            int bad = 0, n = 0;
            std::vector<uint8_t> img, ref;
            for (int i : lend_set_) {
                auto & L = layers[(size_t) i];
                for (int k = 0; k < 3; ++k) {
                    img.resize((size_t) L.n_slots * L.nb2[k]);
                    get_slices(L.lend[k], img.data(), 0, L.n_slots);
                    ref.resize((size_t) L.nb2[k]);
                    for (int s = 0; s < L.n_slots; ++s, ++n) {
                        if (!read_expert(L, L.expert_of[s], k, ref.data())) { ++bad; continue; }
                        bad += std::memcmp(img.data() + (size_t) s * L.nb2[k], ref.data(), (size_t) L.nb2[k]) == 0 ? 0 : 1;
                    }
                }
            }
            LLAMA_LOG_WARN("%s: lend bank check, %s: %d of %d rows differ from the file (%.3f s)\n", __func__,
                           ggml_backend_dev_name(dev), bad, n, (ggml_time_us() - t0) / 1e6);
            if (bad) {
                LLAMA_LOG_ERROR("%s: the lend bank is not the file's bytes; the loan is off\n", __func__);
                lend_set_.clear();
                lend_bytes_ = lend_slots_ = 0;
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_registry_mutex);
        for (auto & L : layers) {
            g_registry[L.src[2]] = &L;
        }
    }
    // the routing trace (campaign expert-foresight): unset, one
    // getenv and no read-back; set, one host read a graph and one write a record
    if (const char * tp = getenv("LLAMA_EXPERT_ROUTE_TRACE")) {
        auto & t = llama_expert_route_trace::instance();
        if (!t.opened) {
            if (!t.open(tp, model.hparams.n_layer(), model.hparams.n_expert, model.hparams.n_expert_used_max())) {
                LLAMA_LOG_WARN("%s: LLAMA_EXPERT_ROUTE_TRACE: cannot open %s: %s\n", __func__, tp, strerror(errno));
            }
        }
        if (t.opened) {
            trace_on = true;
            trace_snapshot();
        }
    }
    LLAMA_LOG_WARN("%s: %s: %zu layers (%d-%d), %lld expert slots (%lld-%lld a layer), %.2f GiB on the card (with the Q8_0 banks), "
                   "bank %.2f GiB in host memory, budget %.2f GiB, from %s; filled in %.1f s; swaps %s\n",
                   __func__, ggml_backend_dev_name(dev), layers.size(), layers.front().il, layers.back().il, (long long) n_slots_all,
                   (long long) lo, (long long) hi, n_bytes / 1073741824.0, n_bank_bytes / 1073741824.0, budget / 1073741824.0,
                   from_profile ? profile : "round robin", (ggml_time_us() - t_fill) / 1e6,
                   nb ? format("on their own queue, %d spare bank entries a layer", n_spare).c_str() : "on the compute queue");
    // Round 21: the card's real state at load. The kernel's device-wide used and
    // total from DRM_XE_DEVICE_QUERY_MEM_REGIONS on this process's xe render fd,
    // beside the process's fdinfo vram0/gtt. The difference used - vram0 is the
    // kernel's own objects plus other clients: INFORMATION, not a rule term (round
    // 22: subtracting it double-counted every other client). A failed query is not
    // fatal. Round 22, review low: the probe (an fdinfo scan, an ioctl and a WARN
    // line) runs only when this cache has a lend configured; a cache under the loan
    // off (the agent, the coder) pays nothing. Round 24: the probe reads this
    // process's fdinfo AFTER the cache's own tail/slot allocations above and
    // load_tensors' weights, so the backend's xe render fds are open; it must stay
    // after them (a scan before the fds exist finds no client and the figure falls
    // back). It sums the device's pdev clients (the runtime opens two per card).
    if (lend_bytes_ != 0) {
        llama_loan_book::FdinfoScan scan;
        llama_loan_book::scan_fdinfo("/proc/self/fdinfo", scan);
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev_, &props);
        const llama_loan_book::FdinfoAgg agg = llama_loan_book::sum_fdinfo(scan, props.device_id);
        const llama_loan_book::CardQuery q = llama_loan_book::xe_query_card(agg.fd);
        const size_t vram0 = agg.measured ? agg.vram0 : 0;
        const size_t gtt   = agg.measured ? agg.gtt : 0;
        if (q.ok) {
            LLAMA_LOG_WARN("%s: card probe at load, %s: used %.2f GiB of %.2f GiB (the kernel's own objects and other "
                           "clients included), fdinfo vram0 %.2f GiB over %d client%s, gtt %.2f GiB, used - fdinfo %.2f GiB "
                           "(information; the rule's margin beyond used is zero)\n",
                           __func__, ggml_backend_dev_name(dev_), q.used / 1073741824.0, q.total / 1073741824.0,
                           vram0 / 1073741824.0, agg.clients, agg.clients == 1 ? "" : "s",
                           gtt / 1073741824.0,
                           llama_loan_book::used_beyond_fdinfo(q.ok, q.used, vram0) / 1073741824.0);
        } else {
            LLAMA_LOG_WARN("%s: card probe at load failed (%s); the loan falls back to fdinfo and the ggml sum\n",
                           __func__, !q.why.empty() ? q.why.c_str() : (agg.why != nullptr ? agg.why : "-"));
        }
    }
    return true;
}

// ---- the prefill-slot-loan (docs/campaigns/prefill-slot-loan-design.md) ----

// Round 21: the per-swap host mirror is gone. A swap changed the slot's expert;
// the lend bank must hold that expert's rows again before the next lend, but at
// swap time the incoming rows only live in the card queue (and the host bank
// entry is a temporary spare). Marking the slot and reading it once at the next
// loan's begin removes the ~1.5 MiB host write from every swap -- the measured
// ~10x swap-write cost after a loan (round 19) -- and the read is one batched
// run a tensor instead of a row a swap.
void llama_expert_cache::mark_lend_dirty(llama_expert_cache_layer & L, int slot) {
    if (!L.lendable || usm_ptr == nullptr || slot < 0) {
        return;
    }
    L.dirty.mark(slot);
}

// The flush: every dirty slot's rows in the CARD's layout, copied from the card's
// slot tensor into the host lend bank. get_slices restores the GGUF layout and
// set_slices writes the bank's own layout (IQ4_NL's split, planar planes), the
// pair the old mirror used -- but now only for dirty rows, once a loan. Runs are
// contiguous slots, one read and one write a run. Called at the lend's start,
// after finish_pending and the caller's synchronize, so the swap copies landed;
// before the lend frees the slot buffer.
bool llama_expert_cache::flush_lend_dirty(size_t * rows, size_t * bytes, double * secs) {
    const int64_t t0 = ggml_time_us();
    size_t n_rows = 0, n_bytes = 0;
    std::vector<uint8_t> img;
    bool ok = true;
    for (int i : lend_set_) {
        auto & L = layers[(size_t) i];
        if (!L.lendable) continue;
        for (int k = 0; k < 3 && ok; ++k) {
            for (const auto & r : L.dirty.runs()) {
                const int count = r.second;
                img.resize((size_t) count * L.nb2[k]);
                try {
                    get_slices(L.slot[k], img.data(), r.first, count);   // the card -> host rows
                    set_slices(L.lend[k], img.data(), r.first, count);   // the host rows -> the bank's layout
                } catch (...) {
                    ok = false;   // a failed read/write: no lend is taken
                    break;
                }
                n_rows  += (size_t) count;
                n_bytes += (size_t) count * L.nb2[k];
            }
        }
        if (ok) L.dirty.clear();
    }
    const double s = (ggml_time_us() - t0) / 1e6;
    if (rows != nullptr)  *rows  = n_rows;
    if (bytes != nullptr) *bytes = n_bytes;
    if (secs != nullptr)  *secs  = s;
    return ok;
}

// The book's callbacks (llama-loan-book.h): the adapt queue, the mirror once a
// swap, the two table halves. Plain member calls, no std::function: the settle
// and the after_graph issue path build one per call.
struct llama_expert_cache::loan_ops {
    llama_expert_cache * ec;

    void fence()  { ec->adapt_fence(ec->layers[0].slot[0]); }
    int  record() { return ec->adapt_record(ec->layers[0].slot[0]); }
    bool done()   { return ec->adapt_done(ec->layers[0].slot[0], ec->book_.ticket); }
    void finish() { ec->adapt_finish(ec->layers[0].slot[0]); }

    void copy_out(const pending_swap & p, int k) {
        ec->adapt_copy(ec->layers[p.layer].bank[k], p.x, ec->layers[p.layer].slot[k], p.slot);
    }
    void copy_in(const pending_swap & p, int k) {
        ec->adapt_copy(ec->layers[p.layer].slot[k], p.slot, ec->layers[p.layer].bank[k], p.b_in);
    }
    void mirror(const pending_swap & p) {
        ec->mark_lend_dirty(ec->layers[p.layer], p.slot);   // round 21: mark, not write
    }
    void tables_out(const std::vector<pending_swap> & plan) { ec->pend_tables_out(plan); }
    void tables_in (const std::vector<pending_swap> & plan) { ec->pend_tables_in(plan); }
    void check_swap() { ec->check_swap(); }
};

void llama_expert_cache::issue_phase(bool fence) {
    loan_ops o{this};
    const int64_t t0 = ggml_time_us();
    llama_loan_book::issue_phase(book_, o, fence);
    n_apply_us += (uint64_t) (ggml_time_us() - t0);
}

// Finding 6: a lend that cannot be published must not leave the tables saying
// "occupied, and the slot is gone": lend() moves slot_of to loan_of (the lend
// bank) and sets slot_of = -1 before it knows the relay reached the card, and a
// failure returned with lent set and loan_out_ false -- so the recovery in
// slot_loan_begin (refill only if loan_out()) restored nothing. The tables go
// back here: slot_of from loan_of, lent cleared, the tables re-uploaded and
// flushed, before the false return.
void llama_expert_cache::lend_undo() {
    bool any = false;
    for (int i : lend_set_) {
        auto & L = layers[(size_t) i];
        if (!llama_loan_book::undo_lend(L)) {   // the tables' restore half (llama-loan-book.h)
            continue;
        }
        upload_tables(L);
        any = true;
    }
    if (any) {
        flush_tables();
    }
    // finding 7 (third review): the undo's writes are not waited for; the tables
    // are checked again here (check_tables' read is the wait) and, if they still
    // differ, this cache is never served from: every later request is refused.
    const int bad = check_tables();
    LLAMA_LOG_WARN("%s: the lend was not published; %s: the tables are back to the slots, %d of %d rows differ\n",
                   __func__, ggml_backend_dev_name(dev_), bad, (int) tab_all->ne[2]);
    if (llama_loan_book::lend_broken(bad != 0)) {
        loan_broken_ = true;
        char why[256];
        std::snprintf(why, sizeof(why),
                      "a lend's undo did not reach the card: %d of %d table rows differ over %d layers",
                      bad, (int) tab_all->ne[2], (int) lend_set_.size());
        llama_loan_book::poison_exit(why);
    }
}

// No adapt in flight and no unconsumed ids read, then the queue drains: the
// loan's precondition. A swap in flight is *settled*, not just polled
// (after_graph issues the copies behind any graph): waiting alone would leave
// the tables half-applied, and the lend reads them. Settling runs the copies
// and both table halves synchronously -- a loan pays this once, blocking.
void llama_expert_cache::finish_pending() {
    // finding 1: a phase that adapt() planned or progress() moved (the spares
    // popped, the phase-1 tables applied) and no graph issued -- decode returned
    // on an aborted or failed ubatch before after_graph. The settle issues it
    // behind a fence and applies both halves; it never clears a plan (a clear
    // leaks the spares and orphans the slot the refill then compares against -1).
    if (book_.phase != 0 && adapt_finish && !layers.empty()) {
        loan_ops o{this};
        llama_loan_book::settle(book_, o);
    }
    if (ids_pending != 0 && owner_sched != nullptr) {
        ggml_backend_sched_synchronize(owner_sched);
    }
    ids_pending = 0;
}

// The lend: the chosen tail layers' ex-slot experts relayed to the lend bank on
// the card's tables, then loan_out. The table writes are non-blocking
// (set_tensor_slices queues them); the wait is check_tables' read below.
bool llama_expert_cache::lend() {
    if (loan_out_ || lend_set_.empty() || b_tail == nullptr || usm_ptr == nullptr) {
        return false;
    }
    finish_pending();
    // round 21: the swap-written rows are the card's now (finish_pending settled
    // them, the caller synchronized); copy ONLY the dirty slots into the lend
    // bank before the lend frees the slot buffer. A failed flush takes no lend.
    size_t dirty_rows = 0, dirty_bytes = 0;
    double dirty_secs = 0.0;
    if (!flush_lend_dirty(&dirty_rows, &dirty_bytes, &dirty_secs)) {
        LLAMA_LOG_WARN("%s: %s: the dirty lend mirror could not be read from the card (%.2f GiB over %zu rows); "
                       "no lend is taken\n", __func__, ggml_backend_dev_name(dev_),
                       dirty_bytes / 1073741824.0, dirty_rows);
        return false;
    }
    if (dirty_rows > 0) {
        LLAMA_LOG_WARN("%s: %s: the lend mirror copied %zu dirty slot rows, %.2f MiB from the card in %.3f s "
                       "(one read a run; the per-swap host writes are gone)\n", __func__,
                       ggml_backend_dev_name(dev_), dirty_rows, dirty_bytes / 1048576.0, dirty_secs);
    }
    const bool mutate_tables = mutate_loan == 2;
    for (int i : lend_set_) {
        auto & L = layers[(size_t) i];
        if (!L.lendable) {
            lend_undo();   // the layers already relayed go back too
            return false;
        }
        for (int e = 0; e < L.n_expert; ++e) {
            if (L.slot_of[e] < 0) continue;
            L.loan_of[e] = L.slot_of[e];   // its row in lend[k]
            L.slot_of[e] = -1;             // not resident: the past-slot experts are bank misses
        }
        L.lent = true;
        upload_tables(L);
    }
    if (mutate_loan == 5 && !lend_set_.empty()) {
        // round 15 (R54): the lent branch's own red case, made to score. The
        // override is a layer member fill_tables consults while lent, so the upload
        // (already run above, re-run here), check_tables' `want`, the card's table
        // after flush_tables and the refill's restore all agree: the lend publishes,
        // and every routed slot expert of the layer reads the neighbour's rows
        // during the prefill. Round 14 shifted one host row outside fill_tables, so
        // check_tables rebuilt the wanted row from loan_of, saw the difference, ran
        // lend_undo and the driver exited 2 before scoring (measured-here, round
        // 14's review).
        auto & L = layers[(size_t) lend_set_.front()];
        L.lend_row_mutate = mutate_loan;
        upload_tables(L);   // the host image now carries the shifted rows
        int moved = 0;
        for (int e = 0; e < L.n_expert; ++e) {
            if (L.loan_of[e] < 0) continue;
            if (llama_loan_book::mutate_loan_bank_row(mutate_loan, L.loan_of[e], L.n_slots) != L.loan_of[e]) ++moved;
        }
        if (moved > 0) {
            LLAMA_LOG_WARN("%s: LLAMA_SLOT_LOAN_MUTATE=5: layer %d: %d ex-slot experts read the lend bank's rows "
                           "one entry over during the prefill (the red case; the refill's check compares slot bytes, "
                           "not table rows)\n", __func__, L.il, moved);
        } else {
            LLAMA_LOG_WARN("%s: LLAMA_SLOT_LOAN_MUTATE=5: layer %d has no lend bank row to shift (one-row bank); "
                           "nothing was mutated\n", __func__, L.il);
        }
    }
    if (mutate_tables) {
        mutate = 0;   // the red case: the relay's last row never reaches the card
        tab_hi--;
    }
    flush_tables();
    const int bad = check_tables();
    LLAMA_LOG_WARN("%s: lend, %s: %d layers (il %d-%d), %lld slots, %.2f GiB freed, lend bank %.2f GiB in host memory; "
                   "table check: %d of %d rows differ\n", __func__, ggml_backend_dev_name(dev_), (int) lend_set_.size(),
                   layers[(size_t) lend_set_.front()].il, layers[(size_t) lend_set_.back()].il, (long long) lend_slots_,
                   lend_bytes_ / 1073741824.0, lend_bytes_ / 1073741824.0, bad, (int) tab_all->ne[2]);
    if (bad) {
        LLAMA_LOG_ERROR("%s: the lend's tables did not reach the card: the slots would be read while lent\n", __func__);
        lend_undo();
        return false;
    }
    loan_out_ = true;
    return true;
}

// Round 7, the refill's types: the lend bank holds every lent row IN THE CARD'S
// LAYOUT -- it is filled through set_tensor / set_slices at load, and an adapt's
// mirror copies bank bytes that are already in that layout -- so the refill is one
// region copy a tensor from it (ggml_backend_opencl_lend_write) with no host
// staging and no layout conversion. There is no type rule to get wrong:
//   - the plain types: the whole slice range, byte for byte;
//   - the planar types (ggml-opencl.cpp's ggml_cl_lb_planar, patch 0022/0030:
//     IQ2_XXS, IQ2_XS, IQ2_S, Q2_0, IQ3_S -- NOT IQ3_XXS or IQ4_XS, which keep
//     their own layout and are not planar): the bank's planes are the card's
//     bytes, slice size and place unchanged;
//   - the split-extra types (IQ4_NL's d/q sub-buffers, Q8_0's SOA): two regions,
//     both contiguous across slices. The re-allocated tail buffer installs the
//     PLAIN extra (init_tensor), so lend_write rebuilds the split without bytes
//     first (fifth review, finding 1); Q8_0 is not lendable (its bank lives in the
//     slot buffer) and is handled for completeness.
//   - BF16 stays f16 on the card (set_tensor converts it), so the bank's f16 bytes
//     are the card's bytes and the copy is exact: the fifth review's "BF16 is a gap
//     of the whole-write rule" cannot bite, because the rule is gone, not widened.
// The check that follows compares the card's bytes with the bank's, byte for byte
// (lend_read on both), so every type round-trips through the same layout twice.
// The tail slot buffer is back (the caller has re-reserved the scheduler for the
// resident chunk first, which frees the VRAM this allocation needs): a new
// buffer for the same tensors, their device extras rebuilt by init_tensor, the
// rows copied from the lend bank, one wait, the tables restored, then every
// lent slot and k against the file.
void llama_expert_cache::set_measure_lent(bool on) {
    for (int i : lend_set_) {
        auto & L = layers[(size_t) i];
        if (L.lendable) L.lent = on;
    }
}

// Round 17, the fixed-margin rule (operator's decision 2026-10-30): grow the
// lend one candidate tail layer at a time until `lend - need >= kPeakReserve +
// kCardMargin` holds over this cache's bytes plus the other caches' lend, AND the
// card's own bound holds (round 19: the process's device bytes there, after the
// lend frees its slots, plus this card's share of the prompt's buffers, at or
// below the card's total minus the fixed margin). A grown layer's lend bank is
// allocated in host memory and filled FROM THE FILE with one whole-tensor set
// (round 18, the OpenCL USM branch of set_tensor builds IQ4_NL's usm_split;
// set_slices only reads it and threw on every grown layer). The read is one GGUF
// read of the grown layer's slot rows -- ~238 MiB a layer at the unit's 11,600 MiB
// budget -- paid once per process, when the growth first happens; the log line
// prints the bytes and the time. A failed growth frees the banks it allocated,
// nulls the tensors it placed in them (round 19: a left `data`/`extra` makes the
// next attempt's alloc plan skip the tensor and would let a partly placed set
// write through freed USM), and leaves the set and every resident buffer as they
// were; the caller then takes the refusal path. Round 25: the growth runs in the
// two steps of `extend_lend_toward` -- the demand first, then the tail's full
// growth if the loan's own need still fits -- so only the need rule can refuse.
// The resident slot buffer is NOT
// freed here: the lend is not published yet, so a failed lend_undo would resume
// on released memory. drop_tail_buffer frees it after the lend and after nulling
// the tensors.
bool llama_expert_cache::extend_lend(size_t need, size_t beside, const llama_loan_book::CardFacts & card,
                                     size_t sanity) {
    if (lend_set_.empty() || cand_.empty()) {
        return false;   // no loan armed, or no candidate to grow
    }
    // `card.lend` is the other caches' freed bytes on this card; this cache's pick
    // is added here, so the card's resident is taken after both.
    if (llama_loan_book::size_fits(need, beside + lend_bytes_) &&
        llama_loan_book::card_fits(card, lend_bytes_)) {
        return true;    // the ask's pick already holds both rules
    }
    llama_loan_book::LendPick ask;
    ask.chosen = lend_set_;
    ask.bytes  = (int64_t) lend_bytes_;
    const auto g = llama_loan_book::extend_lend_toward(cand_, ask, need, sanity, beside, card);
    const auto & p = g.pick;
    if (p.why != nullptr) {
        return false;   // the tail cannot reach a set the loan's own need allows
    }
    std::vector<int> add;
    for (int i : p.chosen) {
        if (std::find(lend_set_.begin(), lend_set_.end(), i) == lend_set_.end()) add.push_back(i);
    }
    if (add.empty()) {
        return false;
    }
    llama_loan_book::drain_now();
    // phase 1: every grown layer's lend bank, host USM, in the card's layout.
    // Everything here is staged: nothing enters lend_set_, bufs or n_bank_bytes
    // until every bank is filled, so a failed growth leaves the cache as it was.
    std::vector<ggml_backend_buffer_t> banks;
    banks.reserve(add.size());
    size_t read_bytes = 0;
    bool   ok = true;
    const int64_t t0 = ggml_time_us();
    try {
        for (int i : add) {
            auto & L = layers[(size_t) i];
            ggml_tensor * ts[3] = { L.lend[0], L.lend[1], L.lend[2] };
            set_alloc_host(true);
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer_n(tail_buft, ts, 3);
            set_alloc_host(false);
            if (b == nullptr) {
                ok = false;
                break;
            }
            ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            banks.push_back(b);
        }
        if (ok) {
            // phase 2: each grown layer's lend bank from THE FILE, one whole-tensor
            // set per region. The USM branch of set_tensor is the only path that
            // builds IQ4_NL's usm_split (ggml-opencl.cpp, ggml_cl_usm_set_tensor);
            // set_slices reads that map and would throw on this fresh bank.
            std::vector<uint8_t> img;
            for (int i : add) {
                auto & L = layers[(size_t) i];
                for (int k = 0; k < 3; ++k) {
                    img.assign((size_t) L.n_slots * L.nb2[k], 0);
                    bool rows = true;
                    for (int s = 0; s < L.n_slots; ++s) {
                        if (L.expert_of[s] >= 0 &&
                            !read_expert(L, L.expert_of[s], k, img.data() + (size_t) s * L.nb2[k])) {
                            rows = false;
                            break;   // the row stays zero; the growth below refuses
                        }
                    }
                    if (!rows) {
                        ok = false;
                        break;
                    }
                    ggml_backend_tensor_set(L.lend[k], img.data(), 0, (size_t) L.n_slots * L.nb2[k]);
                    read_bytes += (size_t) L.n_slots * L.nb2[k];
                }
                if (!ok) break;
            }
        }
    } catch (...) {
        ok = false;   // a std::bad_alloc on an image or a driver throw: staged only
    }
    if (!ok) {
        // Round 19, review item 2: alloc_buffer_n placed the add layers' lend tensors
        // in these banks; after the free each still names the released buffer, its
        // `data` and its `extra` (a USM pointer for the OpenCL host bank). The next
        // attempt's alloc plan skips a tensor with `data != NULL`, so growth would be
        // dead for the process, and a partly placed set would write through freed
        // USM. Null them first, as refill does for tail_slots, then free the banks.
        for (int i : add) {
            auto & L = layers[(size_t) i];
            for (int k = 0; k < 3; ++k) {
                ggml_tensor * t = L.lend[k];
                t->buffer = nullptr;
                t->data   = nullptr;
                t->extra  = nullptr;
            }
        }
        for (ggml_backend_buffer_t x : banks) {
            ggml_backend_buffer_free(x);
        }
        LLAMA_LOG_WARN("%s: %s: the margin rule could not grow the lend (a bank allocation or a file read failed); "
                       "the lend stays as it was\n", __func__, ggml_backend_dev_name(dev_));
        return false;
    }
    // phase 3: publish. No spare buffer is freed here (drop_tail_buffer frees it
    // after the lend is out and after nulling the tensors), so a lend whose check
    // fails can put the resident slots back on live memory.
    size_t n = 0;
    for (size_t a = 0; a < add.size(); ++a) {
        const int i = add[a];
        auto & L = layers[(size_t) i];
        L.lendable = true;
        bufs.push_back(banks[a]);
        n_bank_bytes += ggml_backend_buffer_get_size(banks[a]);
        lend_set_.push_back(i);
        lend_bytes_ += (size_t) L.n_slots * (size_t) (L.nb2[0] + L.nb2[1] + L.nb2[2]);
        lend_slots_ += L.n_slots;
        ++n;
    }
    std::sort(lend_set_.begin(), lend_set_.end());
    LLAMA_LOG_WARN("%s: %s: the fixed margin grows the lend by %zu layer(s) to %zu layers, %.2f GiB "
                   "(%.2f GiB read from the file in %.3f s, once per process); the prompt's buffers need %.2f GiB\n",
                   __func__, ggml_backend_dev_name(dev_), n, lend_set_.size(), lend_bytes_ / 1073741824.0,
                   read_bytes / 1073741824.0, (ggml_time_us() - t0) / 1e6,
                   need == SIZE_MAX ? -1.0 : need / 1073741824.0);
    return llama_loan_book::size_fits(need, beside + lend_bytes_) &&
           llama_loan_book::card_fits(card, lend_bytes_);
}

void llama_expert_cache::drop_tail_buffer() {
    // Round 10, card finding 1: drain BEFORE freeing. The lend frees this buffer
    // while the adapt and bank-stream queues may still hold copies over its slot
    // tensors (the previous request's last adapt is settled but not executed on
    // the exit path); the exit-70 drain then runs those commands against a released
    // object -- the B60's xe -ENOENT and GuC reset, timed to the exit second. A free
    // with nothing queued is the only safe free.
    // Round 11: the drain is keyed on the loan being armed, not on b_tail. At the
    // lend the tail is freed (b_tail non-null); at a refill after a lend it is
    // already null but the clearing below must still follow a drain, and the
    // refill's own failure path frees what the allocation built. With the loan off
    // (lend_bytes_ 0, b_tail null) the call adds nothing.
    if (lend_bytes_ != 0 || b_tail != nullptr) {
        llama_loan_book::drain_now();
    }
    // round 17: a grown layer's slot tensors are in their own buffer, not in
    // b_tail; clear and free every lent layer's tensors so the refill rebuilds them
    for (int i : lend_set_) {
        for (int k = 0; k < 3; ++k) {
            ggml_tensor * t = layers[(size_t) i].slot[k];
            t->buffer = nullptr;
            t->data   = nullptr;
            t->extra  = nullptr;   // init_tensor rebuilds it (ggml-backend.cpp:2284)
        }
    }
    if (b_tail != nullptr) {
        ggml_backend_buffer_free(b_tail);
        b_tail = nullptr;
    }
    for (int i : lend_set_) {
        if (spare_buf[(size_t) i] != nullptr) {
            ggml_backend_buffer_free(spare_buf[(size_t) i]);
            spare_buf[(size_t) i] = nullptr;
        }
    }
}

bool llama_expert_cache::refill(bool verify_rows) {
    if (!loan_out_) {
        return false;
    }
    if (lend_write == nullptr || lend_zero == nullptr || lend_read == nullptr) {
        // init refuses a backend without them (patch 0068); the check keeps a
        // mixed build from writing through a null pointer
        llama_loan_book::poison_exit("the refill has no region copy (lend_write/lend_zero/lend_read missing)");
        return false;
    }
    const int64_t t0 = ggml_time_us();
    // ---- stage: the tail slot buffer back, the extras rebuilt by init_tensor
    // (the caller has re-reserved the scheduler for the resident chunk first, which
    // frees the VRAM this allocation needs). No staging image, no whole (n_slots+1)
    // x nb2 host copy: that was 238 MiB a tensor and the round-7 cost.
    drop_tail_buffer();
    b_tail = ggml_backend_buft_alloc_buffer_n(tail_buft, tail_slots.data(), (int) tail_slots.size());
    if (b_tail == nullptr) {
        // Round 11, review item 1: the default allocator frees the buffers it did
        // build when a later one fails (ggml-backend.cpp's alloc_buffer_n_default),
        // but the tensors it placed in them keep their `buffer`/`data`/`extra`
        // pointing at the released object. Clear them here, exactly as
        // drop_tail_buffer does, so no dangling reference outlives the failure (the
        // poison's drain runs against what the queues hold, not against these).
        for (ggml_tensor * t : tail_slots) {
            t->buffer = nullptr;
            t->data   = nullptr;
            t->extra  = nullptr;
        }
        char why[256];
        std::snprintf(why, sizeof(why), "the refill could not allocate the tail slot buffer (%.2f GiB on %s); the slot bytes are not the file's",
                      lend_bytes_ / 1073741824.0, ggml_backend_dev_name(dev_));
        llama_loan_book::poison_exit(why);
        return false;   // the policy may stand up instead (one line)
    }
    ggml_backend_buffer_set_usage(b_tail, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // round 17: a layer the margin rule grew into the lend has its own slot buffer
    // (the ask's pick is in b_tail); allocate it back for each such lent layer
    for (int i : lend_set_) {
        if (!layers[(size_t) i].lent) continue;
        if (std::find(init_set_.begin(), init_set_.end(), i) != init_set_.end()) continue;
        auto & L = layers[(size_t) i];
        ggml_tensor * ts[3] = { L.slot[0], L.slot[1], L.slot[2] };
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer_n(tail_buft, ts, 3);
        if (b == nullptr) {
            for (int j : lend_set_) {
                for (int k = 0; k < 3; ++k) {
                    ggml_tensor * t = layers[(size_t) j].slot[k];
                    t->buffer = nullptr;
                    t->data   = nullptr;
                    t->extra  = nullptr;
                }
            }
            char why[256];
            std::snprintf(why, sizeof(why), "the refill could not allocate the grown layer %d's slot buffer on %s; the slot bytes are not the file's",
                          L.il, ggml_backend_dev_name(dev_));
            llama_loan_book::poison_exit(why);
            return false;
        }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        spare_buf[(size_t) i] = b;
    }
    const int64_t t_stage = ggml_time_us();
    // ---- write: every lent tensor's rows as a device-side region copy from the
    // lend bank (USM host memory the card reads over the link), non-blocking on the
    // in-order compute queue; then the zero slot (row n_slots, no bank row holds it)
    // from a device zero page. Nothing is staged in host memory, so no layer needs a
    // wait to release a buffer -- round 7 removed the one-wait-a-layer.
    const bool mutate = mutate_loan == 1;
    int n_tensors = 0;
    for (int i : lend_set_) {
        auto & L = layers[(size_t) i];
        if (!L.lent) continue;
        for (int k = 0; k < 3; ++k) {
            // the zero slot first: its first use fills the zero page and drains the
            // queue once, so that wait is before the copies instead of between them
            // (round 8, review item 5). Both calls build an IQ4_NL split extra
            // without bytes, so the write then finds it.
            lend_zero(L.slot[k], llama_loan_book::refill_zero_row(L.n_slots));
            lend_write(L.slot[k], 0, L.lend[k], 0, L.n_slots);
            ++n_tensors;
        }
    }
    if (mutate) {
        // round 8, red case 1: one data byte of a lent row corrupted AFTER the
        // region copies (a device-side read-modify-write). The round-7 form
        // skipped a copy, so the slot's split extra was never built; the check's
        // get_slices then cast a plain extra and the Intel driver segfaulted
        // (no ERROR line). A data byte fails the check under either reference.
        if (lend_perturb != nullptr) {
            auto & L = layers[lend_set_.back()];
            lend_perturb(L.slot[2], 0);
            LLAMA_LOG_WARN("%s: LLAMA_SLOT_LOAN_MUTATE=1: one data byte of the last lent layer's down row 0 "
                           "corrupted after the copies (the red case)\n", __func__);
        } else {
            LLAMA_LOG_WARN("%s: LLAMA_SLOT_LOAN_MUTATE=1: the backend has no lend_perturb; no byte corrupted\n", __func__);
        }
    }
    const int64_t t_write = ggml_time_us();
    // ---- tables: relayed before any graph reads them, queued behind the copies on
    // the same queue, so no wait of their own
    for (int i : lend_set_) {
        auto & L = layers[(size_t) i];
        if (!L.lent) continue;
        // round 15 (R54): clear the red case's override before slot_of is rebuilt,
        // so the post-loan tables are the healthy image (the refill's upload and its
        // check then agree with the file)
        L.lend_row_mutate = 0;
        for (int e = 0; e < L.n_expert; ++e) {
            if (L.loan_of[e] < 0) continue;
            L.slot_of[e] = L.loan_of[e];
            L.loan_of[e] = -1;
        }
        L.lent = false;
        upload_tables(L);
    }
    flush_tables();
    const int64_t t_tables = ggml_time_us();
    // ---- wait: one blocking read of the tables' first row, on the queue the copies
    // precede. The loan's last token's decode and every graph after it follow this.
    {
        std::vector<int32_t> one((size_t) tab_all->ne[1]);
        get_slices(tab_all, one.data(), 0, 1);
    }
    const int64_t t_wait = ggml_time_us();
    const int64_t secs = t_wait - t0;
    // ---- convert: nothing. A region copy moves the bank's bytes to the slot's; the
    // phase is printed so the card run shows the absence of a conversion rather than
    // reading it as missing.
    // ---- verify: exact, every row of every lent tensor, the zero slot included.
    int    bad    = 0, n = 0;
    int    first_bad_layer = -1, first_bad_region = -1, first_bad_row = -1;
    ggml_type first_bad_type = GGML_TYPE_COUNT;
    double s_verify = 0.0;
    const char * from = "off";
    if (verify_rows) {
        const int64_t v0 = ggml_time_us();
        const llama_loan_book::VerifyRef ref = llama_loan_book::verify_ref(std::getenv("LLAMA_SLOT_LOAN_VERIFY"));
        std::vector<uint8_t> a, b, img, frow;
        for (int i : lend_set_) {
            auto & L = layers[(size_t) i];
            if (!L.lendable) continue;
            const int nr = (int) llama_loan_book::refill_check_rows(L.n_slots);
            for (int k = 0; k < 3; ++k) {
                if (ref == llama_loan_book::VerifyRef::File) {
                    // the gate's direct form: the GGUF bytes (get_slices restores a
                    // planar layout, the IQ4_NL split) against a fresh read of the file
                    img.resize((size_t) nr * L.nb2[k]);
                    get_slices(L.slot[k], img.data(), 0, nr);
                    frow.resize((size_t) L.nb2[k]);
                    for (int s = 0; s < nr; ++s, ++n) {
                        if (L.expert_of[s] >= 0) {
                            if (!read_expert(L, L.expert_of[s], k, frow.data())) { ++bad; continue; }
                            bad += std::memcmp(img.data() + (size_t) s * L.nb2[k], frow.data(), (size_t) L.nb2[k]) == 0 ? 0 : 1;
                        } else {
                            std::vector<uint8_t> z((size_t) L.nb2[k], 0);
                            bad += std::memcmp(img.data() + (size_t) s * L.nb2[k], z.data(), (size_t) L.nb2[k]) == 0 ? 0 : 1;
                        }
                    }
                } else {
                    // round 7's default: the card's bytes against the bank's, byte for
                    // byte, both in the card's layout -- no conversion, and no read of
                    // the GGUF (the bank's provenance to the file is the load's lend
                    // bank check and every mirror's bank bytes; design §15).
                    // Round 8, review item 1: lend_read emits a tensor's regions one
                    // after the other ([d x n][q x n]), so neither side may be compared
                    // as one buffer across regions. Round 9, review item 3: compare row
                    // by row INSIDE each region with the region's own row stride. `bad`
                    // is then a row count like `n` (round 8 added one per tensor while
                    // the line said rows), and the first mismatch names its region and
                    // row. The slot image has n_slots + 1 rows a region, so the zero
                    // slot (no bank row holds it) is checked in the same pass.
                    a.resize((size_t) (L.n_slots + 1) * L.nb2[k]);
                    b.resize((size_t) L.n_slots * L.nb2[k]);
                    lend_read(L.slot[k], 0, L.n_slots + 1, a.data());
                    lend_read(L.lend[k], 0, L.n_slots, b.data());
                    const llama_loan_book::LendRowDiff d =
                        llama_loan_book::lend_compare_rows(a.data(), b.data(), (int) L.n_slots,
                                                           L.slot[k]->type, (size_t) L.nb2[k]);
                    bad += d.bad;
                    n += d.rows;
                    if (first_bad_row < 0 && d.first_row >= 0) {
                        first_bad_layer = i;
                        first_bad_region = d.first_region;
                        first_bad_row = d.first_row;
                        first_bad_type = L.slot[k]->type;
                    }
                }
            }
        }
        from = ref == llama_loan_book::VerifyRef::File ? "file" : "lend bank";
        s_verify = (ggml_time_us() - v0) / 1e6;
    }
    char first_bad_name[128] = "";
    if (first_bad_row >= 0) {
        std::snprintf(first_bad_name, sizeof(first_bad_name), " (first: layer %d region %d (%s) row %d)",
                      first_bad_layer, first_bad_region,
                      llama_loan_book::lend_region_name(first_bad_type, first_bad_region), first_bad_row);
    }
    if (verify_rows) {
        LLAMA_LOG_WARN("%s: refill: %lld slots, %.2f GiB in %.3f s (stage %.3f, write %.3f from %d tensors, "
                       "convert 0.000, tables %.3f, wait %.3f), verify on %.3f s: %d of %d rows differ from the %s%s\n",
                       __func__, (long long) lend_slots_, lend_bytes_ / 1073741824.0, secs / 1e6,
                       (t_stage - t0) / 1e6, (t_write - t_stage) / 1e6, n_tensors,
                       (t_tables - t_write) / 1e6, (t_wait - t_tables) / 1e6, s_verify, bad, n, from,
                       first_bad_row >= 0 ? first_bad_name : "");
    } else {
        LLAMA_LOG_WARN("%s: refill: %lld slots, %.2f GiB in %.3f s (stage %.3f, write %.3f from %d tensors, "
                       "convert 0.000, tables %.3f, wait %.3f), verify off\n",
                       __func__, (long long) lend_slots_, lend_bytes_ / 1073741824.0, secs / 1e6,
                       (t_stage - t0) / 1e6, (t_write - t_stage) / 1e6, n_tensors,
                       (t_tables - t_write) / 1e6, (t_wait - t_tables) / 1e6);
    }
    if (verify_rows && bad) {
        char why[256];
        std::snprintf(why, sizeof(why),
                      "the refill did not verify: %d of %d rows differ from the %s over %d lent layers (%lld slots)%s",
                      bad, n, from, (int) lend_set_.size(), (long long) lend_slots_, first_bad_name);
        llama_loan_book::poison_exit(why);
        return false;   // the policy may stand up instead (one line)
    }
    loan_out_ = false;
    return true;
}

// A re-reserve made a new scheduler: the old one is drained (an unconsumed ids
// read targets its memory) and the new one adopted, so adapt runs again.
void llama_expert_cache::rebind(ggml_backend_sched_t sched) {
    if (ids_pending != 0 && owner_sched != nullptr) {
        ggml_backend_sched_synchronize(owner_sched);
    }
    ids_pending = 0;
    owner_sched = sched;
    LLAMA_LOG_INFO("%s: %s: rebound to the scheduler after a re-reserve (adapt back on)\n", __func__,
                   dev_ ? ggml_backend_dev_name(dev_) : "-");
}

// the previous decode ubatch's routing: usage and the hit share
void llama_expert_cache::count(uint32_t n_tokens) {
    for (auto & L : layers) {
        if (L.ids_out == nullptr || L.ids_out->buffer == nullptr || L.ids_out->ne[1] != (int64_t) n_tokens ||
            ggml_backend_buffer_get_type(L.ids_out->buffer) != ggml_backend_buffer_get_type(L.slot[0]->buffer)) {
            return;   // not the graph that just ran, or the ids not on this cache's card
        }
    }
    std::vector<std::vector<int32_t>> ids_read;
    if (ids_pending == n_tokens && ids_host.size() == layers.size()) {
        // read behind the graph by after_graph: complete once the queue is
        ggml_backend_sched_synchronize(owner_sched);
        ids_pending = 0;
    } else {
        ids_pending = 0;
        std::vector<const ggml_tensor *> ts;
        std::vector<void *> dsts;
        ids_read.resize(layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            ids_read[i].resize(ggml_nelements(layers[i].ids_out));
            ts.push_back(layers[i].ids_out);
            dsts.push_back(ids_read[i].data());
        }
        get_tensors(ts.data(), dsts.data(), (int) ts.size());
    }
    std::vector<std::vector<int32_t>> & ids = ids_read.empty() ? ids_host : ids_read;
    for (size_t i = 0; i < layers.size(); ++i) {
        auto & L = layers[i];
        for (int32_t e : ids[i]) {
            if (e < 0 || e >= L.n_expert) {
                continue;
            }
            L.usage[e] += 1.0f;
            (L.slot_of[e] >= 0 ? n_hits : n_misses) += 1;
        }
        // the save's heat, Strata's per-routing-event device counter: raw counts,
        // unnormalized, so it is in the same unit as the prior's
        // STRATA_PROFILE_PRIOR_ENTRIES (a decayed sum would be 1/(1-decay) larger
        // and weigh like ~240k entries at decay 0.7). Counted here, not in adapt():
        // LLAMA_EXPERT_CACHE_ADAPT_EVERY=0 must still accumulate it.
        if (!L.heat.empty()) {
            llama_expert_profile_add_heat(L.heat, L.n_expert, ids[i]);
        }
        if (n_tokens > 1) {
            std::vector<int32_t> u(ids[i]);
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end()), u.end());
            n_multi_pairs  += ids[i].size();
            n_multi_unique += u.size();
            for (int32_t e : u) n_multi_unique_bank += e >= 0 && e < L.n_expert && L.slot_of[e] < 0;
        }
    }
    if (n_tokens > 1 && ++n_multi_steps % 200 == 0) {
        LLAMA_LOG_INFO("%s: %s: %llu multi-token decode steps: %.1f routed pairs, %.1f distinct experts (%.1f in the bank) a step over its layers\n",
                       __func__, dev_ ? ggml_backend_dev_name(dev_) : "-", (unsigned long long) n_multi_steps, (double) n_multi_pairs / n_multi_steps,
                       (double) n_multi_unique / n_multi_steps, (double) n_multi_unique_bank / n_multi_steps);
    }
}

// this cache's swap candidates (Strata's rule, generate.cpp:4433-4452): per
// layer, the hottest bank experts against the coldest slots while the gain
// holds
void llama_expert_cache::candidates(std::vector<llama_expert_cache_swap> & out) {
    for (int i = 0; i < (int) layers.size(); ++i) {
        auto & L = layers[i];
        std::vector<std::pair<float, int>> c, v;
        for (int e = 0; e < L.n_expert; ++e) {
            if (L.slot_of[e] < 0 && L.usage[e] >= min_usage) {
                c.emplace_back(L.usage[e], e);
            }
        }
        for (int s = 0; s < L.n_slots; ++s) {
            if (L.expert_of[s] >= 0) {
                v.emplace_back(L.usage[L.expert_of[s]], s);
            }
        }
        std::sort(c.begin(), c.end(), [](auto & a, auto & b) { return a.first > b.first; });
        std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.first < b.first; });
        for (size_t j = 0; j < std::min(c.size(), v.size()); ++j) {
            if (c[j].first < v[j].first + min_gain) {
                break;
            }
            out.push_back({ this, i, c[j].first - v[j].first, v[j].second, c[j].second });
        }
    }
}

// Strata plans one adapt over every stage's layers, the max_swaps largest
// gains in all (generate.cpp:4453-4454), and each swap lands in its layer's
// own cache on that card's stream (4462-4471): the same here, over the caches
void llama_expert_cache::plan(const std::vector<std::unique_ptr<llama_expert_cache>> & caches) {
    std::vector<llama_expert_cache_swap> all;
    for (const auto & ec : caches) {
        ec->candidates(all);
    }
    std::sort(all.begin(), all.end(), [](auto & a, auto & b) { return a.gain > b.gain; });
    if ((int) all.size() > caches.front()->max_swaps) {
        all.resize(caches.front()->max_swaps);
    }
    for (const auto & ec : caches) {
        ec->adapt(all);
    }
}

void llama_expert_cache::adapt(const std::vector<llama_expert_cache_swap> & all_caches) {
    std::vector<llama_expert_cache_swap> all;
    for (const auto & c : all_caches) {
        if (c.cache == this) {
            all.push_back(c);
        }
    }
    for (auto & L : layers) {
        // no heat here: it counts in count(), one a routed pair, so it does not
        // depend on this cadence (LLAMA_EXPERT_CACHE_ADAPT_EVERY=0 still counts)
        for (float & u : L.usage) {
            u *= decay;
        }
    }
    n_adapts++;
    if (all.empty()) {
        return;
    }
    if (nb) {
        // Strata's adapt stream: each victim's bytes first copied out of its
        // slot into a spare entry of its layer's bank (resident_stage_swaps);
        // the tables change once they have landed (progress)
        const int64_t t0 = ggml_time_us();
        for (auto & c : all) {
            auto & L = layers[c.layer];
            if (L.spare.empty()) {
                n_deferred++;   // as Strata's beyond the exchange buffers' room: a later adapt
                continue;
            }
            const int x = L.spare.back();
            L.spare.pop_back();
            book_.pend.push_back({ c.layer, c.expert, L.expert_of[c.slot], c.slot, L.bank_of[c.expert], x });
            n_swaps++;
        }
        if (book_.pend.empty()) {
            return;
        }
        // a spare entry was a newcomer's bank entry until a table upload: the
        // graphs before it may still read it. The copies themselves go out
        // after the next graph (after_graph): the copy engine also writes
        // that graph's inputs, which must not queue behind them
        adapt_fence(layers[0].slot[0]);
        book_.phase  = 1;
        book_.issued = false;
        book_.check  = n_swaps == book_.pend.size();   // the first adapt
        n_apply_us += (uint64_t) (ggml_time_us() - t0);
        return;
    }
    // an exchange on the card's queue: the newcomer from its bank entry into the
    // victim's slot, the victim into that bank entry (no host copy; the next
    // graph runs after it)
    const int64_t t0 = ggml_time_us();
    const bool check = n_swaps == 0;
    std::vector<char> touched(layers.size(), 0);
    int first_victim = -1, first_entry = -1;
    for (auto & c : all) {
        auto & L = layers[c.layer];
        const int entry = L.bank_of[c.expert], victim = L.expert_of[c.slot];
        if (first_victim < 0) { first_victim = victim; first_entry = entry; }
        // round 21: the slot will hold the newcomer -- mark it dirty. The next
        // lend reads its rows from the card into the lend bank, so no per-swap
        // host write runs on this decode path; `entry` is only the incoming
        // expert's bank entry, a temporary spare after the exchange.
        mark_lend_dirty(L, c.slot);
        for (int k = 0; k < 3; ++k) {
            exchange(L.slot[k], c.slot, L.bank[k], entry);
        }
        L.slot_of[c.expert] = c.slot;   L.expert_of[c.slot] = c.expert;
        L.bank_of[victim]   = entry;    L.bexpert_of[entry] = victim;
        L.slot_of[victim]   = -1;       L.bank_of[c.expert] = -1;
        touched[c.layer] = 1;
        if (trace_on) {
            trace_tables.push_back({ L.il, c.expert, 1 });
            trace_tables.push_back({ L.il, victim, 0 });
        }
        n_swaps++;
    }
    if (mutate == 1) {   // the red case: one swapped slot's bytes corrupted after the exchange
        auto & c = all.front();
        auto & L = layers[c.layer];
        std::vector<uint8_t> bad(L.nb2[0]);
        if (read_expert(L, c.expert, 0, bad.data())) {
            bad[bad.size() / 2] ^= 0x5a;
            write_slices(L.slot[0], c.slot, bad.data(), L.nb2[0]);
            get_slices(L.slot[0], bad.data(), c.slot, 1);   // a blocking read: the write has landed before `bad` goes
        }
        mutate = 0;
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        if (touched[i]) {
            upload_tables(layers[i]);
        }
    }
    if (mutate == 2) {   // the red case: the write leaves out the last changed row
        tab_hi--;
        mutate = 0;
    }
    flush_tables();
    if (!trace_tables.empty()) {
        trace_residents();   // the TABLES record's residency, for a later rotation
        trace_prepare();
        llama_expert_route_trace::instance().event_tables(trace_tables);
        trace_tables.clear();
    }
    if (check) {
        const auto & c   = all.front();
        const auto & L   = layers[c.layer];
        int bad = 0;
        for (int k = 0; k < 3; ++k) {
            bad += verify(L, L.slot[k], k, c.slot, c.expert) ? 0 : 1;
            bad += verify(L, L.bank[k], k, first_entry, first_victim) ? 0 : 1;
        }
        LLAMA_LOG_WARN("%s: table check, %s: %d of %d rows differ\n", __func__, ggml_backend_dev_name(dev_), check_tables(),
                       (int) tab_all->ne[2]);
        LLAMA_LOG_WARN("%s: swap check, %s, layer %d slot %d <- expert %d, bank %d <- expert %d: %d of 6 copies differ from the file\n",
                       __func__, ggml_backend_dev_name(dev_), L.il, c.slot, c.expert, first_entry, first_victim, bad);
    }
    n_apply_us += (uint64_t) (ggml_time_us() - t0);
}

// the phase-1 tables: the victims' bytes are in their spare entries, served from
// the bank from the next graph on, their slots unread (a borrowed phase marker)
void llama_expert_cache::pend_tables_out(const std::vector<pending_swap> & plan) {
    std::vector<char> touched(layers.size(), 0);
    for (auto & p : plan) {
        auto & L = layers[p.layer];
        L.slot_of[p.e_out]  = -1;   L.bank_of[p.e_out] = p.x;
        L.bexpert_of[p.x]   = p.e_out;
        L.expert_of[p.slot] = -1;
        touched[p.layer] = 1;
        if (trace_on) {
            trace_tables.push_back({ L.il, p.e_out, 0 });
        }
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        if (touched[i]) {
            upload_tables(layers[i]);
        }
    }
    flush_tables();
    if (!trace_tables.empty()) {
        trace_residents();   // the TABLES record's residency, for a later rotation
        trace_prepare();
        llama_expert_route_trace::instance().event_tables(trace_tables);
        trace_tables.clear();
    }
}

// the phase-2 tables: the newcomers are in their slots, served from them from
// the next graph on; their old bank entries hold no expert (spares, written only
// after a fence behind this upload)
void llama_expert_cache::pend_tables_in(const std::vector<pending_swap> & plan) {
    std::vector<char> touched(layers.size(), 0);
    for (auto & p : plan) {
        auto & L = layers[p.layer];
        L.slot_of[p.e_in]   = p.slot;   L.expert_of[p.slot] = p.e_in;
        L.bank_of[p.e_in]   = -1;       L.bexpert_of[p.b_in] = -1;
        L.spare.push_back(p.b_in);
        touched[p.layer] = 1;
        if (trace_on) {
            trace_tables.push_back({ L.il, p.e_in, 1 });
        }
    }
    if (mutate == 1) {   // the red case: one swapped slot's bytes corrupted after it landed
        const auto & p = plan.front();
        auto & L = layers[p.layer];
        std::vector<uint8_t> bad(L.nb2[0]);
        if (read_expert(L, p.e_in, 0, bad.data())) {
            bad[bad.size() / 2] ^= 0x5a;
            write_slices(L.slot[0], p.slot, bad.data(), L.nb2[0]);
            get_slices(L.slot[0], bad.data(), p.slot, 1);   // a blocking read: the write has landed before `bad` goes
        }
        mutate = 0;
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        if (touched[i]) {
            upload_tables(layers[i]);
        }
    }
    if (mutate == 2) {   // the red case: the write leaves out the last changed row
        tab_hi--;
        mutate = 0;
    }
    flush_tables();
    if (!trace_tables.empty()) {
        trace_residents();   // the TABLES record's residency, for a later rotation
        trace_prepare();
        llama_expert_route_trace::instance().event_tables(trace_tables);
        trace_tables.clear();
    }
}

// The swaps in flight, polled between ubatches without a wait (Strata's
// apply_pending(false), generate.cpp:4413-4428)
void llama_expert_cache::progress() {
    if (book_.phase == 0 || !book_.issued) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    loan_ops o{this};
    if (llama_loan_book::progress_phase(book_, o)) {
        n_apply_us += (uint64_t) (ggml_time_us() - t0);
    }
}

// the first adapt's swap: its bytes against the file, once they have landed
// (progress_phase calls this after the phase-2 tables, before the plan clears)
void llama_expert_cache::check_swap() {
    if (!book_.check || book_.pend.empty()) {
        return;
    }
    const auto & p = book_.pend.front();
    const auto & L = layers[p.layer];
    int bad = 0;
    for (int k = 0; k < 3; ++k) {
        bad += verify(L, L.slot[k], k, p.slot, p.e_in) ? 0 : 1;
        bad += verify(L, L.bank[k], k, p.x, p.e_out) ? 0 : 1;
    }
    LLAMA_LOG_WARN("%s: table check, %s: %d of %d rows differ\n", __func__, ggml_backend_dev_name(dev_), check_tables(),
                   (int) tab_all->ne[2]);
    LLAMA_LOG_WARN("%s: swap check, %s, layer %d slot %d <- expert %d, bank %d <- expert %d: %d of 6 copies differ from the file\n",
                   __func__, ggml_backend_dev_name(dev_), L.il, p.slot, p.e_in, p.x, p.e_out, bad);
    book_.check = false;
}

// Keep this cache's residents for the next LAYERS snapshot. Updated at init and
// at every TABLES record (a residency change), so a rotation taken from ANOTHER
// cache still writes this card's current set.
void llama_expert_cache::trace_residents() {
    auto & t = llama_expert_route_trace::instance();
    if (!t.opened) {
        return;
    }
    std::vector<int> il, ns;
    std::vector<std::vector<int32_t>> res;
    for (const auto & L : layers) {
        il.push_back(L.il);
        ns.push_back(L.n_slots);
        std::vector<int32_t> r;
        for (int s = 0; s < L.n_slots; ++s) {
            r.push_back(s < (int) L.expert_of.size() ? L.expert_of[s] : -1);
        }
        res.push_back(std::move(r));
    }
    t.set_residents((intptr_t) this, il, ns, res);
}

// The trace's LAYERS record of every cache's residents at the moment (init, or
// a rotation): under a layer split one record per cache would leave a rotated
// file unable to replay alone, so the rotation writes all registered sets.
void llama_expert_cache::trace_snapshot() {
    auto & t = llama_expert_route_trace::instance();
    if (!t.opened) {
        return;
    }
    trace_residents();
    t.event_layers_all();
}

// Called before every trace record: at the file's cap it opens the next file
// (its own header) and writes the residents that file starts from. The cache
// calls it from after_graph, BEFORE arcint writes the verify's accept line, so
// the line and the graph record land in the same file at a rotation (round 6's
// one-line-off defect). A graph record's rotation is taken HERE and nowhere
// else: while a record is pending (trace_pending), its accept line already
// sits in this file, so a rotation at advance/flush_trace or an MTP DRAFT
// record would split the line from the record (round 7's defect) -- the
// decision is llama_expert_route_trace_rotate_now, and the rotation then waits
// for the next after_graph.
void llama_expert_cache::trace_prepare() {
    auto & t = llama_expert_route_trace::instance();
    if (!trace_on) {
        return;
    }
    if (!t.opened) {
        trace_on = false;   // a write failed: the read-backs stop, the cost goes
        return;
    }
    if (!llama_expert_route_trace_rotate_now(t)) {
        return;
    }
    if (!t.rotate_next()) {
        trace_on = false;   // a full disk: stop tracing, do not fail the run
        return;
    }
    trace_snapshot();
}

// At a request's end: the pending last graph's ids are complete after one
// sync, and the graph is written before the next request's advance would
// count a different one. `count` keeps the usage the engine already counted
// from this graph at the next decode, exactly once (last_n is cleared).
void llama_expert_cache::flush_trace(ggml_backend_sched_t sched) {
    if (!trace_on) {
        return;
    }
    if (!llama_expert_route_trace::instance().opened) {
        trace_on = false;   // a failed write: no more read-backs, the cost goes
        if (trace_pending) {
            llama_expert_route_trace::instance().pending_graph(false);
        }
        trace_pending = false;
        trace_tokens.clear();
        last_n = 0;
        return;
    }
    if (!trace_pending || last_n == 0 || sched != owner_sched) {
        return;
    }
    if (last_n <= decode_max) {
        count(last_n);
    } else {
        ggml_backend_sched_synchronize(owner_sched);
        ids_pending = 0;
    }
    if (ids_host.size() == layers.size() && trace_n == last_n) {
        std::vector<int> il;
        for (const auto & L : layers) {
            il.push_back(L.il);
        }
        // no trace_prepare: this record's rotation was taken at after_graph,
        // and rotating now (an MTP DRAFT record may have crossed the cap since,
        // the line is already written) would split the accept line from its
        // record
        llama_expert_route_trace::instance().event_graph(trace_kind, trace_pos, trace_n, trace_tokens, il, ids_host);
    }
    if (trace_pending) {
        llama_expert_route_trace::instance().pending_graph(false);
    }
    trace_pending = false;
    trace_tokens.clear();
    last_n = 0;
}

void llama_expert_cache::flush_traces(const std::vector<std::unique_ptr<llama_expert_cache>> & caches,
                                      ggml_backend_sched_t sched) {
    for (const auto & ec : caches) {
        ec->flush_trace(sched);
    }
}

// The owner context's ubatch: counts the previous decode ubatch's routing;
// true when an adapt is due
bool llama_expert_cache::advance(ggml_backend_sched_t sched, uint32_t n_tokens) {
    if (loan_out_) {
        return false;   // the loan pauses adapt, and no swap is planned into a lent layer
    }
    // the first context to decode drives the cache (an MTP draft context on
    // the same model would count its drafts and shift the cadence)
    if (owner_sched == nullptr) {
        owner_sched = sched;
    } else if (sched != owner_sched) {
        return false;
    }
    if (trace_on && !llama_expert_route_trace::instance().opened) {
        trace_on = false;   // a failed write: stop the read-backs at the next call
    }
    if (last_n > 0 && trace_pending) {
        // the trace's graph: the ids read behind it are complete now. A decode
        // graph's count() already waits; a prompt graph needs one wait here,
        // only when the trace is on
        if (last_n <= decode_max) {
            count(last_n);   // waits for the queue; ids_host is the graph's
        } else {
            ggml_backend_sched_synchronize(owner_sched);
            ids_pending = 0;
        }
        // as in flush_trace: no trace_prepare, the pending record's rotation
        // was taken at after_graph and rotating here would split its accept
        // line from it
        if (ids_host.size() == layers.size() && trace_n == last_n) {
            std::vector<int> il;
            for (const auto & L : layers) {
                il.push_back(L.il);
            }
            llama_expert_route_trace::instance().event_graph(trace_kind, trace_pos, trace_n, trace_tokens, il, ids_host);
        }
        if (trace_pending) {
            llama_expert_route_trace::instance().pending_graph(false);
        }
        trace_pending = false;
        trace_tokens.clear();
    } else if (last_n > 0 && last_n <= decode_max) {
        count(last_n);   // waits for the queue
    }
    progress();
    last_n = n_tokens;
    // the exchanges and the next graph share the card's in-order queue
    if (n_tokens <= decode_max) {
        since_adapt += n_tokens;
        if (adapt_every > 0 && since_adapt >= adapt_every) {
            since_adapt = 0;
            return true;
        }
    }
    return false;
}

void llama_expert_cache::step(const std::vector<std::unique_ptr<llama_expert_cache>> & caches, ggml_backend_sched_t sched,
                              uint32_t n_tokens) {
    // every cache counts its own layers; they share the owner and the cadence
    bool due = false, busy = false;
    for (const auto & ec : caches) {
        if (ec->loan_out()) {
            busy = true;   // a loan out: no plan anywhere (one lane in stage 1)
            continue;
        }
        due  = ec->advance(sched, n_tokens) || due;
        busy = busy || ec->book_.phase != 0;
    }
    // Strata plans no adapt while the previous swaps are in flight (generate.cpp:4431)
    if (due && !busy) {
        plan(caches);
    }
}

void llama_expert_cache::after_graph(ggml_backend_sched_t sched, uint32_t n_tokens, const int32_t * tokens,
                                     int32_t pos, bool all_outputs) {
    if (loan_out_) {
        return;   // no swap is issued, and a lent graph's ids are not read
    }
    if (ids_pending != 0 && owner_sched != nullptr) {
        ggml_backend_sched_synchronize(owner_sched);   // an unconsumed read still targets ids_host
    }
    ids_pending = 0;
    if (trace_on && sched == owner_sched && n_tokens > 0) {
        if (!llama_expert_route_trace::instance().opened) {
            trace_on = false;   // a failed write: the read-backs below stop
        } else {
            // BEFORE arcint writes this verify's accept line (which follows the
            // target decode): a rotation here puts the line and the graph record
            // in the same file. trace_prepare also writes the new file's LAYERS.
            trace_prepare();
            // a prompt call outputs fewer rows than it has tokens (llama.cpp's
            // prompt evaluation), a decode or verify call outputs every row; the
            // count alone would call a prompt's 4-token checkpoint tail a verify.
            // An embedding batch has no token ids at all.
            trace_kind = llama_expert_route_graph_kind(n_tokens, decode_max, all_outputs, tokens != nullptr);
            trace_n      = n_tokens;
            trace_pos    = pos;
            if (tokens != nullptr) {
                trace_tokens.assign(tokens, tokens + n_tokens);
            } else {
                trace_tokens.clear();   // event_graph writes -1 a row
            }
        }
    }
    if (book_.phase != 0 && !book_.issued && sched == owner_sched) {
        // the swaps' copies on the adapt queue, behind the fence placed before
        // this graph (adapt for phase 1, progress for phase 2): they run while it
        // computes (patch 0042). The mirror once a swap, outside the k loop
        // (finding 3): 3x the bytes otherwise.
        issue_phase(false);
    }
    if ((!async_ids && !trace_on) || sched != owner_sched || n_tokens == 0 ||
        (n_tokens > decode_max && !trace_on)) {
        return;
    }
    // each id copy on this cache's card (its buffer type is the card's own
    // under 0023): the reads below go to one queue, which get_tensors asserts
    for (auto & L : layers) {
        if (L.ids_out == nullptr || L.ids_out->buffer == nullptr || L.ids_out->ne[1] != (int64_t) n_tokens ||
            ggml_backend_buffer_get_type(L.ids_out->buffer) != ggml_backend_buffer_get_type(L.slot[0]->buffer)) {
            return;
        }
    }
    ids_host.resize(layers.size());
    std::vector<const ggml_tensor *> ts;
    std::vector<void *> dsts;
    for (size_t i = 0; i < layers.size(); ++i) {
        ids_host[i].resize(ggml_nelements(layers[i].ids_out));
        ts.push_back(layers[i].ids_out);
        dsts.push_back(ids_host[i].data());
    }
    // the switch is respected: ASYNC_IDS=0 means a read with a wait, even when
    // the trace is on (the trace's ids are the cache's own). The wait is this
    // graph's, as the switch asks, and the read-back then costs no extra copy.
    if (async_ids) {
        get_tensors_async(ts.data(), dsts.data(), (int) ts.size());
    } else {
        get_tensors(ts.data(), dsts.data(), (int) ts.size());
    }
    ids_pending = n_tokens;   // the ids are valid either way; count() takes them
    trace_pending = trace_on;
    if (trace_pending) {
        // process-wide: another cache on another card must not rotate while
        // this graph's record is still waiting for its write
        llama_expert_route_trace::instance().pending_graph(true);
    }
}

llama_expert_cache_stats llama_expert_cache::stats() const {
    llama_expert_cache_stats s;
    s.decode_hits   = n_hits;
    s.decode_misses = n_misses;
    s.swaps         = n_swaps;
    s.adapts        = n_adapts;
    s.apply_us      = n_apply_us;
    for (auto & L : layers) {
        s.slots += L.n_slots;
    }
    s.bytes      = n_bytes;
    s.bank_bytes = n_bank_bytes;
    return s;
}
