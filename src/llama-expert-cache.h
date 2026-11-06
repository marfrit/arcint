#pragma once

// A per-expert GPU cache for MoE layers whose expert tensors stay in host
// memory, after Strata's (src/core/expert_cache.cpp, the adapt step of
// src/program/generate.cpp:4413-4484, the pinned arena the GPU reads misses
// from, src/core/expert_source.cpp:1614-1696). Each layer keeps
//   - slots on the card: its hot experts, in the expert tensors' own types,
//     sized in bytes, filled from a decode-routing profile;
//   - a bank in GPU-visible host memory: every other expert of the layer,
//     read by the GPU over the link;
//   - residency tables on the card (expert -> slot, expert -> bank entry).
// The router's ids go through the tables on the GPU (get_rows) and both
// branches run there, summed before the router weights (llama-graph.cpp,
// build_moe_ffn): no GPU-CPU hand-off. Decode's routed ids are read back once
// per step and counted; every adapt_every decode tokens hot bank experts swap
// with cold slot experts of the same layer. Patch 0042 (Strata's adapt stream,
// generate.cpp:122-158, 4396-4480): the swaps run on their own queue while the
// graphs run, in two steps polled between ubatches: the victim is copied out
// of its slot into a spare bank entry and then served from there, the
// newcomer is copied into the freed slot and served from it once that has
// landed; its old bank entry becomes a spare. The tables change only between
// graphs, so no graph reads a slice that is being written; the copies go out
// right after a graph is enqueued, so they run while it computes and do not
// hold up the next graph's input writes on the copy engine.
// (LLAMA_EXPERT_CACHE_NB=0: the exchange of the two on the card's queue
// ahead of the next graph, as before.)
//
// Under a layer split (LLAMA_SPLIT_MODE_LAYER) each GPU gets its own cache
// over the layers it runs (Strata's per-stage expert cache, generate.cpp
// 2286-2307): its slots and tables on that card, its bank in USM host memory
// of that card's context (a USM allocation belongs to one context), its own
// budget, tables, queue and counters.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-loan-book.h"
#include "llama-expert-route-trace.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct llama_model_loader;
struct llama_expert_cache;

struct llama_expert_cache_layer {
    int      il       = -1;
    int      n_expert = 0;
    int      n_slots  = 0;              // slot n_slots holds zeros: a bank expert's row in the slot branch
    int      n_bank   = 0;              // bank entry n_bank holds zeros: a slot expert's row in the bank branch
    ggml_tensor * src [3] = {};         // gate, up, down as llama.cpp loaded them (host memory, not computed)
    ggml_tensor * slot[3] = {};         // [ne0, ne1, n_slots + 1] on the card
    ggml_tensor * bank[3] = {};         // [ne0, ne1, n_bank + 1 + n_spare] in GPU-visible host memory
    int      n_spare  = 0;              // bank entries after the zero entry that hold no expert at load
    std::vector<int32_t> spare;         // the entries that hold no expert now (a swap's victim goes there)
    ggml_tensor * tab_slot = nullptr;   // I32 [1, n_expert] on the card: slot, or -1
    ggml_tensor * tab_bank = nullptr;   // I32 [1, n_expert] on the card: bank entry, or -1
    ggml_tensor * tab_loan = nullptr;   // I32 [1, n_expert] on the card: entry in lend[k], or -1
    // a prefill-slot-loan (docs/campaigns/prefill-slot-loan.md): this layer's
    // slots can be lent (host banks, no Q8_0 bank in the slot buffer) and the
    // experts the slots held are readable from lend[] in host memory while
    // the slots are gone. The prompt's graph reads bank[] and lend[] only.
    // a loan candidate: host banks and region-copyable types, so its slot
    // tensors live in the tail context and it may be lent; `lendable` below is
    // membership in the current lend set (growth moves a candidate into it)
    bool         candidate = false;
    bool         lendable  = false;
    bool         lent      = false;
    ggml_tensor * lend[3]  = {};        // [ne0, ne1, n_slots] in GPU-visible host memory, row s = expert_of[s]
    // a down projection outside the kq kernels (Q8_0: the general MoE path,
    // which cannot skip a pair) reads the zero slot / zero entry instead of -1
    ggml_tensor * tab_slot_dn = nullptr;
    ggml_tensor * tab_bank_dn = nullptr;
    // the single branch (decode): slot, or n_slots + bank entry; the slot
    // tensors' kernels read a bank expert from the gathered mirror
    ggml_tensor * tab_comb = nullptr;
    int      file_idx [3] = {};
    size_t   file_offs[3] = {};         // the tensor's data in its GGUF file
    size_t   nb2      [3] = {};         // one expert's bytes
    std::vector<int32_t> slot_of;       // expert -> slot, -1
    std::vector<int32_t> expert_of;     // slot -> expert
    std::vector<int32_t> bank_of;       // expert -> bank entry, -1
    std::vector<int32_t> loan_of;       // expert -> entry in lend[], -1 (the loan's tables)
    // round 15: MUTATE=5 -- when nonzero, fill_tables rotates this layer's every
    // ex-slot expert's lend-bank row by one while the layer is lent, so the upload,
    // check_tables, the card's table and the refill's restore all build the same
    // image; refill clears it before slot_of is rebuilt. Zero on every healthy run.
    int32_t lend_row_mutate = 0;
    // round 21: the slots a swap changed since the last loan. The per-swap host
    // mirror is gone: the next lend copies exactly these rows from the card into
    // the lend bank, one read a run, before anything is freed.
    llama_loan_book::DirtyTrack dirty;
    std::vector<int32_t> bexpert_of;    // bank entry -> expert, -1 (the zero entry, a spare)
    std::vector<float>   usage;         // decode routing counts, decayed each adapt
    std::vector<double>  heat;          // raw routing counts for the save (--llama-expert-cache-save), one
                                        // a routed pair in count(); empty unless the save is on. Double: a
                                        // month at ~33 t/s passes float's 2^24 on the hottest pairs
    ggml_tensor * ids_out = nullptr;    // the last graph's routed ids (a copy, marked output)
    int      tab_row  = 0;              // the first of its tables' rows in the cache's tab_all
    int      n_tab    = 0;              // its tables: 3, or 5 with the _dn pair
    llama_expert_cache * owner = nullptr;
};

// a planned swap: in `cache`, layers[layer]'s bank expert `expert` into slot `slot`
struct llama_expert_cache_swap {
    llama_expert_cache * cache;
    int   layer;
    float gain;
    int   slot;
    int   expert;
};

struct llama_expert_cache_stats {
    uint64_t decode_hits   = 0;
    uint64_t decode_misses = 0;
    uint64_t swaps         = 0;
    uint64_t adapts        = 0;
    uint64_t apply_us      = 0;         // the decoding thread's time enqueueing swaps and tables
    int64_t  slots         = 0;
    size_t   bytes         = 0;         // on the card: slots, the Q8_0 banks, tables, the gather mirror
    size_t   bank_bytes    = 0;         // bank in host memory
};

struct llama_expert_cache {
    ~llama_expert_cache();

    // After the weights are loaded: the layers on `dev` whose expert tensors
    // live in host buffers get slots on `dev`, up to `budget` bytes, in the
    // order of `profile` (Strata's STRP file, its pairs of these layers;
    // empty: expert-major round robin over these layers), and a bank of the
    // rest. False when nothing qualifies or the memory cannot be had.
    // `lend_bytes` (0: no loan): the tail layers' slot tensors go into their
    // own buffer and get a lend bank of the same rows in host memory, up to
    // that many bytes.
    bool init(llama_model & model, const llama_model_loader & ml, ggml_backend_dev_t dev, size_t budget, const char * profile,
              size_t lend_bytes = 0);

    // `dev` runs at least one layer init() would cache
    static bool has_layers(const llama_model & model, const llama_model_loader & ml, ggml_backend_dev_t dev);

    // The process's last graph, at the end of a request (or at shutdown): the
    // cache writes a graph when the NEXT graph's advance runs, so without this
    // a continuous log loses its last step. One sync, only when the trace is
    // on and a graph is pending.
    static void flush_traces(const std::vector<std::unique_ptr<llama_expert_cache>> & caches, ggml_backend_sched_t sched);

    // Between ubatches of llama_context::decode, before the next graph, over
    // the model's caches (one per card): each counts the previous decode
    // ubatch's routing of its layers; every adapt_every decode tokens the
    // swaps are planned over all of them (the max_swaps largest gains) and
    // each cache enqueues its own on its card's queue.
    static void step(const std::vector<std::unique_ptr<llama_expert_cache>> & caches, ggml_backend_sched_t sched, uint32_t n_tokens);

    // Right after a decode graph is enqueued: its routed ids are read behind
    // it (no wait), for step() to count once the host has synchronized
    // (LLAMA_EXPERT_CACHE_ASYNC_IDS=0: step() reads them with a wait)
    // `tokens` is the ubatch's token ids (null for an embedding batch), `pos`
    // its first position (-1 when absent), `all_outputs` whether the call
    // outputs every row (a decode/verify call) or a prompt call does not
    void after_graph(ggml_backend_sched_t sched, uint32_t n_tokens, const int32_t * tokens, int32_t pos,
                     bool all_outputs);

    llama_expert_cache_stats stats() const;

    ggml_backend_dev_t device() const { return dev_; }

    // ---- the prefill-slot-loan (init's lend_bytes; sec docs/campaigns) ----
    // No adapt in flight and no unconsumed ids read: the loan's precondition.
    void finish_pending();
    // The lend: the tail layers' tables say their ex-slot experts are in
    // lend[], one synchronize for the writes. `true` when the loan is out.
    bool lend();
    // The tail slot buffer is back: the lend rows -> the slots, one wait, the
    // tables restored, then every lent slot and k against the file.
    // (verify: llm_slot_loan_verify; false on any differing row)
    bool refill(bool verify_rows);
    // A new scheduler after a re-reserve: the old one is drained first.
    void rebind(ggml_backend_sched_t sched);
    // The loan: the tail slot tensors' buffer is freed (their pointers survive;
    // the refill allocates a new buffer and init_tensor rebuilds the extras).
    void drop_tail_buffer();
    // The graph a loan would build, for sizing it without lending: the tail
    // layers' `lent` flag (the graph builder reads it), the tables untouched.
    void set_measure_lent(bool on);
    bool loan_out() const { return loan_out_; }
    // finding 7 (third review): a lend whose undo left the tables differing; the
    // cache must never serve from it, so every request is refused until restart
    bool loan_broken() const { return loan_broken_; }
    // The fixed-margin rule: grow the lend one candidate tail layer at a time
    // until `lend - need >= kPeakReserve + kCardMargin` holds over the cache's own
    // bytes plus `beside` (the other caches' lend, under a layer split), and the
    // card's own bound holds (`card`: its total, this process's resident bytes
    // there before the lend, the other caches' freed bytes there, this card's
    // share of the prompt's buffers, the margin). `need` is the growth's demand:
    // the prompt's buffers, or those plus the carried shortfall. Round 25: when
    // the tail cannot reach that demand, the tail's full growth is taken when it
    // still meets `sanity` (the loan's own `need`), so the shortfall is never the
    // thing that refuses. Frees the grown layers' slot buffers and allocates
    // their lend banks. `true` when a grown set is committed (false: nothing that
    // meets `sanity` can be reached, or an allocation failed; nothing is lent.)
    bool extend_lend(size_t need, size_t beside, const llama_loan_book::CardFacts & card,
                     size_t sanity);
    int  lend_layers() const { return (int) lend_set_.size(); }   // entries of `layers`
    int64_t lend_slots() const { return lend_slots_; }
    size_t lend_bytes() const { return lend_bytes_; }          // the tail buffers' size
    // Round 19: the device-resident bytes this cache holds on its card -- bslots,
    // the tail buffer (freed by a lend), candidate layers' spare buffers, and the
    // gather mirror's reserve; the host USM banks are NOT device memory and are
    // left out (n_bank_bytes is their figure). Used by the card's own bound.
    size_t device_bytes() const { return n_bytes; }
    // round 21: the host-USM bank bytes GTT should carry (the eviction guard's
    // reference), and the rows a swap left dirty for the next lend.
    size_t host_bank_bytes() const { return n_bank_bytes; }
    size_t dirty_rows() const {
        size_t n = 0;
        for (const auto & L : layers) n += L.lendable ? L.dirty.rows() : 0;
        return n;
    }
    // the tail slot tensors' buffer: freed by the loan, allocated by the refill
    ggml_backend_buffer_t tail_buffer() const { return b_tail; }
    // Round 20: the largest bank tensor's bytes (a stream-ring slot holds one bank
    // tensor whole) and the backend's ring bytes (0: not yet allocated). The cold
    // ring's first touch is K x the largest bank tensor; a warm process's ring is
    // in its fdinfo figure already.
    size_t max_bank_tensor_bytes() const { return max_bank_tensor_; }
    size_t stream_ring_bytes() const {
        if (!bank_stream_ring) return 0;
        size_t ring = 0, max_bytes = 0;
        bank_stream_ring(dev_, &ring, &max_bytes);
        return ring;
    }

    std::vector<llama_expert_cache_layer> layers;
    ggml_tensor * tab_id = nullptr;     // I32 [1, n_expert] identity: the routed ids' copy for read-back
    // every layer's tables as rows of one tensor (the layers' tab_* are views),
    // uploaded in one write as Strata's residency table (generate.cpp:4406-4413)
    ggml_tensor * tab_all = nullptr;    // I32 [1, n_expert, rows]
    std::vector<int32_t> tab_host;      // its bytes, alive until their non-blocking write lands
    int      tab_lo = INT32_MAX, tab_hi = 0;   // the rows written since the last upload

    // Strata's constants (generate.cpp:358, :380, :4428-4484)
    float    min_usage   = 2.0f;
    float    min_gain    = 1.5f;
    int      max_swaps   = 96;
    float    decay       = 0.7f;
    // decode tokens between adapts. Strata: every 4 verify windows of ~3 tokens
    // (~13); here 6 measured best (B60, Flash-Next: 13.0 t/s, 78 % on the card,
    // against 12.4-12.6 / 75 % at 12, 12.8 at 4, 10.0 at 3)
    uint32_t adapt_every = 6;
    uint32_t decode_max  = 8;           // a ubatch up to this many tokens counts as decode (Strata MAXT 8)

private:
    // one adapt's swaps in flight: phase 1 the victims' copies out, phase 2 the
    // newcomers' copies in (declared here: the phase's declarations below use it)
    struct pending_swap { int layer, e_in, e_out, slot, b_in, x; };
    // the loan's state (the accessors above read them: declared here, before
    // their first use in an inline body)
    bool loan_out_    = false;
    bool loan_broken_ = false;   // finding 7: the undo's tables still differ (refuse every request)

    bool advance(ggml_backend_sched_t sched, uint32_t n_tokens);
    void count(uint32_t n_tokens);
    void trace_residents();                   // keep this cache's residents for the next snapshot
    void trace_snapshot();                    // LAYERS: every cache's residents now
    void trace_prepare();                     // a rotation at the cap, then the snapshot
    void flush_trace(ggml_backend_sched_t sched);
    void candidates(std::vector<llama_expert_cache_swap> & out);
    void adapt(const std::vector<llama_expert_cache_swap> & all);
    static void plan(const std::vector<std::unique_ptr<llama_expert_cache>> & caches);
    bool read_expert(const llama_expert_cache_layer & L, int expert, int k, uint8_t * dst) const;
    void write_slices(ggml_tensor * t, int64_t i2, const uint8_t * data, size_t nb2);
    bool verify(const llama_expert_cache_layer & L, ggml_tensor * t, int k, int i2, int expert) const;
    void upload_tables(llama_expert_cache_layer & L);
    void flush_tables();
    void fill_tables(const llama_expert_cache_layer & L, int32_t * r) const;
    int  check_tables() const;
    // round 21: a swap marks its slot dirty instead of mirroring ~1.5 MiB into
    // the host lend bank on the card's decode path; the next lend's flush copies
    // the dirty rows from the card.
    void mark_lend_dirty(llama_expert_cache_layer & L, int slot);
    // the dirty rows of every lendable layer into the lend bank, one read a
    // contiguous run, before the lend frees anything. false: a read or write
    // failed, and no lend is taken.
    bool flush_lend_dirty(size_t * rows = nullptr, size_t * bytes = nullptr, double * secs = nullptr);
    // a lend that cannot be published puts every table it changed back (finding 6),
    // then checks them again (finding 7) and marks the cache unusable if they differ
    void lend_undo();
    // a pending adapt's table halves, as progress() and the settle apply them
    void pend_tables_out(const std::vector<pending_swap> & plan);
    void pend_tables_in(const std::vector<pending_swap> & plan);
    // the first adapt's swap check, once its copies have landed (progress_phase)
    void check_swap();
    // the book's callbacks on the adapt queue and the tables, for llama-loan-book.h
    struct loan_ops;
    void issue_phase(bool fence);   // llama_loan_book::issue_phase, the cache's ops
    void progress();

    ggml_backend_dev_t dev_ = nullptr;  // the card of these layers: slots, tables, the bank's context, the queue
    std::vector<int>   fds;             // per GGUF file
    ggml_context *     ctx = nullptr;
    ggml_context *     bank_ctx_ = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;   // bslots, bbank, the lend bank; b_tail is NOT here:
                                               // the loan frees and re-allocates it, so it has its own
                                               // owner (finding 7)
    ggml_backend_buffer_t b_tail = nullptr;
    ggml_context *     lend_ctx_ = nullptr;    // the lend bank tensors (host USM)
    ggml_context *     tail_ctx_ = nullptr;    // the tail slot tensors (their own card buffer)
    std::vector<ggml_tensor *> tail_slots;     // the ask-chosen tail slot tensors, in b_lend
    ggml_backend_buffer_type_t tail_buft = nullptr;
    std::vector<int> lend_set_;                 // the chosen entries of `layers`, ascending
    std::vector<int> init_set_;                 // the ask's pick: b_tail's entries
    // round 17: a candidate not in the ask's pick has its slot tensors in its own
    // small card buffer, so growth can free exactly the layers it lends (one more
    // layer, not the whole tail); the grow also needs the candidates and the
    // backend's host-allocation switch for the lend bank it allocates then
    std::vector<llama_loan_book::LendCandidate> cand_;
    std::vector<ggml_backend_buffer_t> spare_buf;   // per layer entry; nullptr when freed or not a spare
    void (*set_alloc_host)(bool) = nullptr;   // the lend bank growth allocates in host USM
    int64_t     lend_slots_ = 0;
    size_t      lend_bytes_ = 0;
    void (*set_slices)(ggml_tensor *, const void *, int64_t, int64_t) = nullptr;
    // a USM host tensor's bytes as a host pointer: ggml_backend_opencl_usm_ptr.
    // Used where both sides are host memory (the lend mirror at a swap).
    const void * (*usm_ptr)(const ggml_tensor *) = nullptr;
    // Round 7, the refill: the lend bank holds the lent rows in the card's layout,
    // so the refill is a device-side region copy from it (ggml_backend_opencl_lend_write),
    // the zero slot from a device zero page (lend_zero), and the raw read its check
    // compares against (lend_read). No host staging, no conversion, no whole write.
    void (*lend_write)(ggml_tensor *, int64_t, ggml_tensor *, int64_t, int64_t) = nullptr;
    void (*lend_zero)(ggml_tensor *, int64_t) = nullptr;
    void (*lend_read)(const ggml_tensor *, int64_t, int64_t, void *) = nullptr;
    // round 8, red case 1: one data byte of a lent row, read-modify-write on the
    // card after the region copies -- a corrupt byte the check must catch (the
    // round-7 form skipped a copy, so the slot's split extra was never built and
    // the check's get_slices cast a plain extra: a driver segfault, no ERROR line)
    void (*lend_perturb)(ggml_tensor *, int64_t) = nullptr;
    void (*get_tensors)(const ggml_tensor * const *, void * const *, int) = nullptr;
    void (*get_tensors_async)(const ggml_tensor * const *, void * const *, int) = nullptr;
    std::vector<std::vector<int32_t>> ids_host;   // after_graph's reads, per layer
    uint32_t           ids_pending = 0;   // their graph's tokens; 0: none queued
    // the diagnostic trace (LLAMA_EXPERT_ROUTE_TRACE): the last graph's kind,
    // its token ids and whether their read is queued
    bool               trace_on = false;
    bool               trace_pending = false;
    uint8_t            trace_kind = 0;
    uint32_t           trace_n = 0;
    int32_t            trace_pos = -1;
    std::vector<int32_t> trace_tokens;
    std::vector<llama_expert_route_trace::change> trace_tables;
    bool               async_ids = true;
    void (*get_slices)(const ggml_tensor *, void *, int64_t, int64_t) = nullptr;
    void (*exchange)(ggml_tensor *, int64_t, ggml_tensor *, int64_t) = nullptr;
    // patch 0042: the adapt queue (null: an older backend, the exchange)
    void (*adapt_copy)(ggml_tensor *, int64_t, const ggml_tensor *, int64_t) = nullptr;
    void (*adapt_fence)(const ggml_tensor *) = nullptr;
    int  (*adapt_record)(const ggml_tensor *) = nullptr;
    bool (*adapt_done)(const ggml_tensor *, int) = nullptr;
    void (*adapt_finish)(const ggml_tensor *) = nullptr;
    bool nb = false;                    // the swaps off the compute queue
    // the phase's own state and its transitions (llama-loan-book.h): never clear a
    // planned phase, fence before an unissued phase's copies, the mirror once a swap
    llama_loan_book::PhaseState<pending_swap> book_;
    uint64_t n_deferred = 0;            // swaps put off for want of a spare entry in their layer

    uint32_t           since_adapt = 0;
    ggml_backend_sched_t owner_sched = nullptr;   // the context that drives the cache
    uint32_t           last_n = 0;      // the previous ubatch's tokens
    uint64_t           n_swaps = 0, n_adapts = 0, n_apply_us = 0, n_hits = 0, n_misses = 0;
    // multi-token decode steps (a verify): routed pairs and distinct experts, summed over the layers
    uint64_t           n_multi_pairs = 0, n_multi_unique = 0, n_multi_unique_bank = 0, n_multi_steps = 0;
    size_t             n_bytes = 0, n_bank_bytes = 0;
    size_t             max_bank_tensor_ = 0;   // round 20: the stream ring's largest slot content
    // round 20: the backend's bank-stream ring bytes (0 while not allocated)
    void (*bank_stream_ring)(ggml_backend_dev_t, size_t *, size_t *) = nullptr;
    int                mutate = 0;      // LLAMA_EXPERT_CACHE_MUTATE: corrupt one swapped slot (red case for the swap check)
    int                mutate_loan = 0; // LLAMA_SLOT_LOAN_MUTATE: 1 a tensor of the last lent layer left uncopied, 2 drop the lend's last table row, 3 the lent sizing skipped, 4 the return's re-reserve skipped (llama-context.cpp), 5 a lent layer's bank row off by one (lend, the prefill's red case)
    // --llama-expert-cache-save: the profile loaded in init (the prior of both
    // rules: the xe mixing's pseudo-counts and cu's tie-break), empty for round
    // robin; and the model, to get the geometry and the cache count at teardown
    std::vector<std::pair<int, int>> profile_ranked;
    const llama_model * model_ = nullptr;
};

// The layer whose down projection (as loaded) is `down_exps`, or null.
llama_expert_cache_layer * llama_expert_cache_lookup(const ggml_tensor * down_exps);
llama_expert_cache * llama_expert_cache_owner(const llama_expert_cache_layer * layer);
