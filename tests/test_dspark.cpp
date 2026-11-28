// The DSpark drafter's bookkeeping (src/exec/dspark_ring.h) without a model:
// the Markov context ring (keyed by position, cut on a rollback, carried by a
// checkpoint), the row the anchor reads, the block's shape and the verify's
// width, the drafts read from the block's logits, and a whole speculative cycle
// on a synthetic target (draft, verify walk, rollback) whose output must be the
// plain loop's, token for token, greedy and sampled.
//
// Each red case of dspark_ring.h (ARCINT_DSPARK_RED=1..10) fails at least one
// case here: tests/dflash/build-red.sh's form, run by hand.
#include "exec/dspark_ring.h"
#include "exec/verify_walk.h"
#include "harness.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace lgc;

namespace {

constexpr size_t kWidth = 4;   // a context row's width (2,304 on Inkling 2 Mini)
constexpr int    kBlock = 8;   // the trained block (Inkling 2 Mini's mtp_config.block_size)

// a row that names its position: every float is p + 0.25 * column
std::vector<float> row_of(int64_t p) {
    std::vector<float> r(kWidth);
    for (size_t i = 0; i < kWidth; ++i) r[i] = static_cast<float>(p) + 0.25f * static_cast<float>(i);
    return r;
}

bool is_row_of(const float* r, int64_t p) {
    if (r == nullptr) return false;
    const std::vector<float> want = row_of(p);
    for (size_t i = 0; i < kWidth; ++i)
        if (r[i] != want[i]) return false;
    return true;
}

}  // namespace

// ------------------------------------------------------------------ the ring

TEST(dspark_ring_answers_by_position_only) {
    DsparkRing ring(kBlock + 1, kWidth);
    for (int64_t p = 0; p < 20; ++p) ring.put(p, row_of(p).data());
    // the last 9 positions are held; an older one shares a slot with a newer one
    for (int64_t p = 11; p < 20; ++p) CHECK(is_row_of(ring.get(p), p));
    CHECK(ring.get(10) == nullptr);   // slot 10 % 9 = 1 now holds 19
    CHECK(ring.get(2) == nullptr);
    CHECK(ring.get(20) == nullptr);
    CHECK(ring.get(-1) == nullptr);
}

TEST(dspark_ring_rollback_drops_the_rejected_rows) {
    // a verify injected positions 10..15 (the anchor and five drafts); two were
    // accepted, so the sequence keeps 13 tokens and rows 13..15 leave
    DsparkRing ring(kBlock + 1, kWidth);
    for (int64_t p = 7; p < 16; ++p) ring.put(p, row_of(p).data());
    ring.trim(13);
    for (int64_t p = 7; p < 13; ++p) CHECK(is_row_of(ring.get(p), p));
    for (int64_t p = 13; p < 16; ++p) CHECK(ring.get(p) == nullptr);
    CHECK(ring.held() == std::vector<int64_t>({7, 8, 9, 10, 11, 12}));
}

TEST(dspark_ring_checkpoint_carries_the_rows_before_it) {
    DsparkRing ring(kBlock + 1, kWidth);
    for (int64_t p = 0; p < 12; ++p) ring.put(p, row_of(p).data());   // holds 3..11
    // a checkpoint at 10 tokens: rows 3..9 (within 9 of it, before it); 10 and 11 are later tokens'
    const std::vector<uint8_t> s = ring.save(10);
    DsparkRing back(kBlock + 1, kWidth);
    CHECK(back.load(10, s.data(), s.size()));
    CHECK(back.held() == std::vector<int64_t>({3, 4, 5, 6, 7, 8, 9}));
    for (int64_t p = 3; p < 10; ++p) CHECK(is_row_of(back.get(p), p));
    CHECK(back.get(10) == nullptr);
    // the draft after the restore reads the row of token 9
    CHECK(is_row_of(back.get(dspark_anchor_pos(10)), 9));
}

TEST(dspark_ring_load_refuses_a_state_of_later_tokens) {
    DsparkRing ring(kBlock + 1, kWidth);
    for (int64_t p = 0; p < 12; ++p) ring.put(p, row_of(p).data());
    const std::vector<uint8_t> s = ring.save(12);   // rows up to 11
    DsparkRing back(kBlock + 1, kWidth);
    CHECK(!back.load(8, s.data(), s.size()));      // a sequence of 8 tokens has no row 11
    CHECK(back.held().empty());
    CHECK(!back.load(12, s.data(), s.size() - 1));   // a truncated buffer
    CHECK(back.held().empty());
    DsparkRing other(kBlock + 1, kWidth + 1);       // another width
    CHECK(!other.load(12, s.data(), s.size()));
    CHECK(back.load(12, s.data(), s.size()));
    CHECK(back.held().size() == 9);
}

// ------------------------------------------------------------------ the anchor, the block, the width

TEST(dspark_anchor_reads_the_last_committed_token) {
    // the reference: f_ring[(start - 1) % ring], start the anchor's position
    CHECK_EQ(dspark_anchor_pos(100), int64_t{99});
    CHECK_EQ(dspark_anchor_pos(1), int64_t{0});
    CHECK_EQ(dspark_anchor_pos(0), int64_t{-1});   // nothing before the first token: no draft
}

TEST(dspark_block_is_the_trained_block_at_every_width) {
    // the reference runs [anchor, mask x 7] whatever num_speculative_tokens is
    for (int n = 1; n <= 7; ++n) CHECK_EQ(dspark_block_rows(kBlock, n), kBlock);
}

TEST(dspark_step_drafts_and_the_verify_width) {
    CHECK_EQ(dspark_step_drafts(7, 7, kBlock), 7);
    CHECK_EQ(dspark_step_drafts(5, 7, kBlock), 5);
    CHECK_EQ(dspark_step_drafts(7, 3, kBlock), 3);    // the step's budget (max_tokens, the context)
    CHECK_EQ(dspark_step_drafts(7, 0, kBlock), 0);
    CHECK_EQ(dspark_step_drafts(9, 9, kBlock), 7);    // never past the block's draft rows
    CHECK_EQ(dspark_step_drafts(7, -2, kBlock), 0);
    // the verify is 1 + drafts rows: 2 to 8 when it drafts (patch 0226's 1-8-row loop
    // controller), never more drafts than the target's n_rs_seq snapshots (the backend
    // sets n_rs_seq = --llama-dflash), so a rejection always rolls back
    for (int n = 1; n <= 7; ++n)
        for (int b = 0; b <= 9; ++b) {
            const int d = dspark_step_drafts(n, b, kBlock);
            CHECK(dspark_verify_rows(static_cast<size_t>(d)) >= 1);
            CHECK(dspark_verify_rows(static_cast<size_t>(d)) <= 8);
            CHECK(d <= n);
            CHECK(d <= b || d == 0);
        }
}

// ------------------------------------------------------------------ the drafts from the block's logits

namespace {

constexpr size_t kVocab = 16;

// block logits whose row i has its maximum at argmax[i]
std::vector<float> block_logits(const std::vector<int>& argmax) {
    std::vector<float> r(argmax.size() * kVocab);
    for (size_t i = 0; i < argmax.size(); ++i)
        for (size_t v = 0; v < kVocab; ++v)
            r[i * kVocab + v] = static_cast<int>(v) == argmax[i] ? 9.0f : 0.1f * static_cast<float>(v % 7);
    return r;
}

}  // namespace

TEST(dspark_pick_reads_the_draft_rows_not_the_anchor) {
    const std::vector<float> l = block_logits({15, 1, 2, 3, 4, 5, 6, 7});
    CHECK(dspark_pick(l.data(), 8, kVocab, 7) == std::vector<int>({1, 2, 3, 4, 5, 6, 7}));
    CHECK(dspark_pick(l.data(), 8, kVocab, 5) == std::vector<int>({1, 2, 3, 4, 5}));   // the prefix
    CHECK(dspark_pick(l.data(), 8, kVocab, 9) == std::vector<int>({1, 2, 3, 4, 5, 6, 7}));
    CHECK(dspark_pick(l.data(), 8, kVocab, 0).empty());
    CHECK(dspark_pick(l.data(), 1, kVocab, 7).empty());   // a block of the anchor alone drafts nothing
}

// The drafts and the chain's conditioning share one tie rule: ggml's ARGMAX, the LAST maximum (ggml-cpu/vec.h
// ggml_vec_argmax_f32; the OpenCL op of patch 0240 keeps it). A first-maximum pick would draft one id while the chain
// conditioned on the other. Red 11: the old first-maximum pick.
TEST(dspark_pick_takes_the_highest_id_on_a_tie) {
    std::vector<float> l = block_logits({0, 9, 9});
    l[1 * kVocab + 4] = 9.0f;   // row 1: ids 4 and 9 tie
    l[2 * kVocab + 12] = 9.0f;  // row 2: ids 9 and 12 tie
    CHECK(dspark_pick(l.data(), 3, kVocab, 2) == std::vector<int>({9, 12}));
    std::vector<float> e(3 * kVocab, -1.0f);   // every id equal: the last id, as ggml's ARGMAX
    CHECK(dspark_pick(e.data(), 3, kVocab, 2) == std::vector<int>({static_cast<int>(kVocab) - 1, static_cast<int>(kVocab) - 1}));
}

// The shortlist path (patch 0241) drafts the ids the graph returned: the budget's prefix, none for a lane the graph
// returned nothing for, and none at all when an id lies outside the vocabulary (a broken graph must not draft).
// Red 12: the vocabulary bound dropped (the out-of-range id is drafted).
TEST(dspark_ids_take_bounds_the_drafts) {
    const int vocab = 200058;
    const int32_t ids[4] = { 5, 9, vocab - 1, 12 };
    CHECK(dspark_ids_take(ids, 4, 3, vocab) == std::vector<int>({5, 9, vocab - 1}));   // the budget's prefix
    CHECK(dspark_ids_take(ids, 2, 7, vocab) == std::vector<int>({5, 9}));              // fewer returned than asked
    CHECK(dspark_ids_take(ids, -1, 7, vocab).empty());                                 // the graph drafted none
    CHECK(dspark_ids_take(ids, 4, 0, vocab).empty());
    CHECK(dspark_ids_take(nullptr, 4, 3, vocab).empty());
    const int32_t pad[3] = { 5, vocab, 7 };    // an id past the vocabulary (a pad row)
    CHECK(dspark_ids_take(pad, 3, 3, vocab).empty());
    const int32_t neg[2] = { -1, 7 };
    CHECK(dspark_ids_take(neg, 2, 2, vocab).empty());
}

TEST(dspark_chunk_rows_only_where_the_ring_reads) {
    const std::vector<int64_t> none, ck = {4096};
    // a verify's rows always
    CHECK(dspark_chunk_needs_rows(100, 106, true, 77000, none, 9));
    // a prompt chunk far from the end: no; the last chunk and one ending 9 rows short: yes
    CHECK(!dspark_chunk_needs_rows(0, 64, false, 77000, none, 9));
    CHECK(dspark_chunk_needs_rows(76992, 77000, false, 77000, none, 9));
    CHECK(dspark_chunk_needs_rows(76928, 76992, false, 77000, none, 9));
    CHECK(!dspark_chunk_needs_rows(76864, 76928, false, 77000, none, 9));
    // a chunk before a checkpoint the prefill saves
    CHECK(dspark_chunk_needs_rows(4032, 4096, false, 77000, ck, 9));
    CHECK(!dspark_chunk_needs_rows(4096, 4160, false, 77000, ck, 9));
    // no plan: every row
    CHECK(dspark_chunk_needs_rows(0, 64, false, -1, none, 9));
}

// ------------------------------------------------------------------ a whole cycle on a synthetic target

namespace {

// The target: the next token's logits depend on the two tokens before it (so a
// draft needs the row of the token before the anchor, as DSpark's Markov head
// needs f_rnn of pos0 - 1). Row v: a ramp with a peak at (3 * a + 5 * b + 1) % 16.
void target_row(int a, int b, float* out) {
    const int peak = (3 * a + 5 * b + 1) % static_cast<int>(kVocab);
    for (size_t v = 0; v < kVocab; ++v)
        out[v] = (static_cast<int>(v) == peak ? 2.5f : 0.0f) + 0.15f * static_cast<float>((v * 7 + static_cast<size_t>(a)) % 5);
}
int target_greedy(int a, int b) {
    std::vector<float> r(kVocab);
    target_row(a, b, r.data());
    int best = 0;
    for (size_t v = 1; v < kVocab; ++v)
        if (r[v] > r[static_cast<size_t>(best)]) best = static_cast<int>(v);
    return best;
}

// The context row the target's features give the drafter at position p: the
// token at p (in the one-hot-free form row_of(token)), as the ring stores it
struct Cycle {
    std::vector<int> out;          // the emitted tokens
    int              steps    = 0;
    int              drafted  = 0;
    int              accepted = 0;
    int              no_draft = 0;   // steps whose anchor row was missing
    int              max_rollback = 0;
};

// The plain loop: sample a row, observe it, the next row
std::vector<int> plain(const std::vector<int>& prompt, int n, const SamplerParams& sp, uint64_t seed) {
    Sampler          s(sp, seed);
    std::vector<int> h = prompt, out;
    std::vector<float> r(kVocab);
    for (int k = 0; k < n; ++k) {
        target_row(h[h.size() - 2], h.back(), r.data());
        const int t = s.sample(r.data(), kVocab);
        s.observe(t);
        out.push_back(t);
        h.push_back(t);
    }
    return out;
}

// The speculative loop as the backend runs it with the DSpark driver: the
// prompt's rows into the ring; then each step a block draft at pos0 (the
// anchor id_last there, the ring's row of pos0 - 1), a verify of
// [id_last, drafts] whose rows the target computes from the drafted prefix and
// whose context rows enter the ring (rejected ones too), the walk, and the
// rollback to the kept tokens (the ring cut there; the target's recurrent
// state, n_rs_seq = n_draft snapshots, rolls back the same rows).
// `smart`: the drafter knows the target's greedy rule (its drafts are right at
// temperature 0); else it drafts a fixed wrong token after the first.
Cycle speculative(const std::vector<int>& prompt, int n, int n_draft, const SamplerParams& sp, uint64_t seed,
                  bool smart = true) {
    Sampler          s(sp, seed);
    DsparkRing       ring(kBlock + 1, kWidth);
    std::vector<int> have(prompt.begin(), prompt.end() - 1);   // the cells; id_last stands next
    int              id_last = prompt.back();
    // the prefill (all but the last prompt token, as the backend's): every row's features enter the drafter
    for (size_t p = 0; p + 1 < prompt.size(); ++p) ring.put(static_cast<int64_t>(p), row_of(prompt[p]).data());
    Cycle c;
    while (static_cast<int>(c.out.size()) < n) {
        const size_t pos0   = have.size();   // id_last's position
        const int    budget = n - static_cast<int>(c.out.size()) - 1;
        const int    want   = dspark_step_drafts(n_draft, budget, kBlock);
        // the draft: the block's logits (row 0 the anchor's, never a draft)
        std::vector<int> draft;
        const float*     ctx = ring.get(dspark_anchor_pos(static_cast<int64_t>(pos0)));
        if (want > 0 && ctx == nullptr) ++c.no_draft;
        if (want > 0 && ctx != nullptr) {
            const int        prev = static_cast<int>(ctx[0]);   // the token before the anchor, from its row
            std::vector<int> argmax(static_cast<size_t>(dspark_block_rows(kBlock, n_draft)));
            argmax[0] = (id_last + 11) % static_cast<int>(kVocab);   // what a drafted anchor row would say
            int a = prev, b = id_last;
            for (size_t i = 1; i < argmax.size(); ++i) {
                const int nx = smart ? target_greedy(a, b) : (i == 1 ? target_greedy(a, b) : 0);
                argmax[i]    = nx;
                a = b;
                b = nx;
            }
            const std::vector<float> l = block_logits(argmax);
            draft = dspark_pick(l.data(), argmax.size(), kVocab, want);
        }
        // the verify: rows for [id_last, drafts] from the drafted prefix
        std::vector<int> batch = {id_last};
        batch.insert(batch.end(), draft.begin(), draft.end());
        std::vector<float> rows(batch.size() * kVocab);
        for (size_t i = 0; i < batch.size(); ++i) {
            const int a = i == 0 ? (have.empty() ? 0 : have.back()) : batch[i - 1];
            target_row(a, batch[i], rows.data() + i * kVocab);
            ring.put(static_cast<int64_t>(pos0 + i), row_of(batch[i]).data());   // the injection, rejected rows too
        }
        const VerifyWalk w = walk_verify(rows.data(), batch.size(), kVocab, draft, s, budget + 1,
                                         [](int) { return false; },
                                         [&](int t, std::string&) {   // the walk makes the observation
                                             c.out.push_back(t);
                                             return Control::Continue;
                                         });
        ++c.steps;
        c.drafted += static_cast<int>(draft.size());
        c.accepted += w.accepted;
        have.push_back(id_last);
        have.insert(have.end(), draft.begin(), draft.begin() + w.accepted);
        id_last = w.last;
        // the rollback: both contexts and the ring to have.size(); the rows it removes
        const int rolled = static_cast<int>(batch.size()) - 1 - w.accepted;
        c.max_rollback   = std::max(c.max_rollback, rolled);
        ring.trim(static_cast<int64_t>(have.size()));
        if (w.done) break;
    }
    return c;
}

SamplerParams sampler_at(float temp) {
    SamplerParams p;
    p.temperature        = temp;
    p.top_p              = 1.0f;
    p.top_k              = temp > 0.0f ? 50 : 0;   // the Inkling unit's sampler: T 1.0, top-k 50
    p.min_p              = 0.0f;
    p.repetition_penalty = 1.0f;
    return p;
}

}  // namespace

TEST(dspark_cycle_greedy_is_the_plain_loop_and_accepts_a_right_drafter) {
    const std::vector<int> prompt = {3, 9, 4, 1};
    for (int n_draft : {5, 7}) {
        const SamplerParams sp = sampler_at(0.0f);
        const Cycle         c  = speculative(prompt, 64, n_draft, sp, 1);
        CHECK(c.out == plain(prompt, 64, sp, 1));
        CHECK_EQ(c.no_draft, 0);
        // a drafter that knows the target is right at every draft: n_draft + 1 tokens a step
        CHECK_EQ(c.accepted, c.drafted);
        CHECK(c.steps <= (64 + n_draft) / (n_draft + 1) + 1);
        CHECK_EQ(c.max_rollback, 0);
    }
}

TEST(dspark_cycle_sampled_is_the_plain_loop_token_for_token) {
    // the verify walk samples each row from the target's own row with the
    // sampler's state as the plain loop has it: the emitted tokens are the
    // plain loop's for the same seed, whatever the drafts were (lossless)
    const std::vector<int> prompt = {7, 2, 5, 5};
    const SamplerParams    sp     = sampler_at(1.0f);
    int                    accepted = 0, drafted = 0, rolled = 0;
    for (uint64_t seed = 1; seed <= 30; ++seed)
        for (int n_draft : {5, 7})
            for (bool smart : {true, false}) {
                const Cycle c = speculative(prompt, 48, n_draft, sp, seed, smart);
                CHECK(c.out == plain(prompt, 48, sp, seed));
                CHECK(c.max_rollback <= n_draft);   // within the target's n_rs_seq snapshots
                accepted += c.accepted;
                drafted += c.drafted;
                rolled = std::max(rolled, c.max_rollback);
            }
    // the sampled run accepts some drafts and rejects others: both paths ran
    CHECK(accepted > 0);
    CHECK(accepted < drafted);
    CHECK(rolled > 0);
}

TEST(dspark_cycle_wrong_drafts_cost_steps_not_tokens) {
    // a drafter right only at its first draft: one draft accepted a step at
    // temperature 0, the output still the plain loop's
    const std::vector<int> prompt = {3, 9, 4, 1};
    const SamplerParams    sp     = sampler_at(0.0f);
    const Cycle            c      = speculative(prompt, 40, 7, sp, 1, /*smart=*/false);
    CHECK(c.out == plain(prompt, 40, sp, 1));
    CHECK(c.accepted >= c.steps - 1);   // the first draft right at each step (the last may draft less)
    CHECK(c.accepted < c.drafted);
    CHECK(c.max_rollback >= 1);
}

// ------------------------------------------------------------------ the target and the checkpoint state (review C1, C3)

TEST(dspark_refuses_a_target_that_is_not_inkling_2) {
    // the taps are Inkling 2's layer visits (0214): another model of the same width and vocabulary size would
    // reach the graph's tap assert at its first decode, so the load refuses it
    CHECK(dspark_target_refusal("inkling2").empty());
    CHECK(dspark_target_refusal("inkling2_loop").empty());
    CHECK(!dspark_target_refusal("qwen35moe").empty());
    CHECK(!dspark_target_refusal("dflash").empty());
    CHECK(!dspark_target_refusal("").empty());
}

TEST(dspark_state_split_refuses_a_broken_payload) {
    // [u64 window bytes][window][ring]
    std::vector<uint8_t> p(8 + 5 + 3, 0);
    const uint64_t w = 5;
    std::memcpy(p.data(), &w, 8);
    DsparkStateParts s;
    CHECK(dspark_state_split(p.data(), p.size(), s));
    CHECK_EQ(s.win_off, size_t{8});
    CHECK_EQ(s.win_size, size_t{5});
    CHECK_EQ(s.ring_off, size_t{13});
    CHECK_EQ(s.ring_size, size_t{3});
    CHECK(!dspark_state_split(p.data(), 4, s));   // shorter than its own header
    const uint64_t big = 100;
    std::memcpy(p.data(), &big, 8);
    CHECK(!dspark_state_split(p.data(), p.size(), s));   // a window past the payload
}

TEST(dspark_lane_counts_every_failed_restore) {
    // a failed restore leaves the drafter's global layer without the cells below the cut for the lane's life:
    // it is counted (the decode line's "dflash lane failures", cumulative) and warned, never silent
    DsparkLaneHealth h;
    CHECK(!h.on_restore(true));
    CHECK_EQ(h.restore_failures, uint64_t{0});
    CHECK(h.on_restore(false));
    CHECK(h.on_restore(false));
    CHECK(!h.on_restore(true));
    CHECK_EQ(h.restore_failures, uint64_t{2});
}

// ------------------------------------------------------------------ --llama-dflash-kv into the draft context

// 0243: the probe's line, as tools/dspark_probe_read.py reads it; without the shortlist f is d and in is empty
TEST(dspark_probe_line_is_the_readers_format) {
    DsparkProbePending p;
    p.set = true; p.pos0 = 812; p.id_last = 1234; p.sl = 256;
    p.d = { 5, 6, 7 }; p.f = { 5, 9, 7 }; p.in = { 1, 0, 1 };
    CHECK_EQ(dspark_probe_line(3, p, 1), std::string("{\"seq\": 3, \"pos0\": 812, \"id_last\": 1234, \"a\": 1, \"sl\": 256, "
                                                     "\"d\": [5, 6, 7], \"f\": [5, 9, 7], \"in\": [1, 0, 1]}\n"));
    p.sl = 0;
    CHECK_EQ(dspark_probe_line(0, p, 3), std::string("{\"seq\": 0, \"pos0\": 812, \"id_last\": 1234, \"a\": 3, \"sl\": 0, "
                                                     "\"d\": [5, 6, 7], \"f\": [5, 6, 7], \"in\": []}\n"));
}

// 0243: a probe line that does not reach its file is reported (red 13 ignores the results); /dev/full refuses every
// write, a file in the temporary directory takes it
TEST(dspark_probe_write_reports_a_short_write) {
    std::FILE* full = std::fopen("/dev/full", "w");
    if (full == nullptr) {
        std::printf("  (no /dev/full here: the short write is not exercised)\n");
    } else {
        CHECK(!dspark_probe_write(full, std::string(8192, 'x') + "\n"));
        std::fclose(full);
    }
    std::FILE* ok = std::tmpfile();
    CHECK(ok != nullptr);
    if (ok != nullptr) {
        CHECK(dspark_probe_write(ok, "{}\n"));
        std::fclose(ok);
    }
}

TEST(dspark_kv_plan_reaches_the_draft_context) {
    // the drafter context's K/V types are the flag's; a quantized cache takes flash attention on (llama.cpp
    // refuses a quantized V without it, so the context refuses at load instead of serving another type)
    const DsparkKvPlan f = dspark_kv_plan("f16", "f16");
    CHECK_EQ(f.type_k, std::string("f16"));
    CHECK_EQ(f.type_v, std::string("f16"));
    CHECK(!f.flash_attn_on);
    const DsparkKvPlan q = dspark_kv_plan("q8_0", "q8_0");
    CHECK_EQ(q.type_k, std::string("q8_0"));
    CHECK_EQ(q.type_v, std::string("q8_0"));
    CHECK(q.flash_attn_on);
    const DsparkKvPlan h = dspark_kv_plan("q8_0", "q4_0");
    CHECK_EQ(h.type_k, std::string("q8_0"));
    CHECK_EQ(h.type_v, std::string("q4_0"));
    CHECK(h.flash_attn_on);
}
