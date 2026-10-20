#include "core/reasoning.h"
#include "core/sampler.h"
#include "harness.h"

#include <map>

using namespace lgc;

namespace {

SamplerParams greedy_params() {
    SamplerParams p;
    p.temperature = 0.0f;
    p.top_k       = 0;
    p.top_p       = 1.0f;
    p.repetition_penalty = 1.0f;
    return p;
}

SamplerParams sampling_params(float temp, int top_k, float top_p) {
    SamplerParams p;
    p.temperature        = temp;
    p.top_k              = top_k;
    p.top_p              = top_p;
    p.repetition_penalty = 1.0f;
    return p;
}

std::vector<float> logits(std::vector<float> v) { return v; }

}  // namespace

TEST(sampler_greedy_is_argmax) {
    auto    l = logits({0.1f, 0.9f, 0.4f, -2.0f});
    Sampler s(greedy_params(), 1);
    CHECK_EQ(s.sample(l.data(), l.size()), 1);
    CHECK_EQ(Sampler::argmax(l.data(), l.size()), 1);
}

TEST(sampler_top_k_one_is_greedy_too) {
    // top_k 1 collapses the distribution exactly as temperature 0 does; both
    // must take the same path or "greedy" means two things.
    auto l = logits({0.1f, 0.9f, 0.4f});
    SamplerParams p = sampling_params(1.0f, 1, 1.0f);
    Sampler s(p, 12345);
    CHECK(p.greedy());
    CHECK_EQ(s.sample(l.data(), l.size()), 1);
}

TEST(sampler_is_reproducible_for_a_seed) {
    auto draw = [](uint64_t seed) {
        Sampler          s(sampling_params(1.0f, 0, 1.0f), seed);
        std::vector<int> out;
        for (int i = 0; i < 24; ++i) {
            auto l = logits({1.0f, 1.05f, 0.95f, 1.02f, 0.9f});
            out.push_back(s.sample(l.data(), l.size()));
        }
        return out;
    };
    CHECK_EQ(draw(42), draw(42));
    CHECK(draw(42) != draw(43));
}

TEST(sampler_respects_top_k) {
    // Only ids 0 and 1 may ever come out, however long we draw.
    Sampler s(sampling_params(2.0f, 2, 1.0f), 7);
    for (int i = 0; i < 200; ++i) {
        auto      l  = logits({5.0f, 4.9f, 1.0f, 0.5f, 0.1f});
        const int id = s.sample(l.data(), l.size());
        CHECK(id == 0 || id == 1);
    }
}

TEST(sampler_respects_top_p) {
    // One token holds essentially all the mass; a modest top_p must keep only it.
    Sampler s(sampling_params(1.0f, 0, 0.5f), 3);
    for (int i = 0; i < 100; ++i) {
        auto l = logits({12.0f, 0.0f, -1.0f, -2.0f});
        CHECK_EQ(s.sample(l.data(), l.size()), 0);
    }
}

TEST(sampler_never_returns_nothing_for_a_tiny_top_p) {
    // top_p 0 must degenerate to greedy, not to an empty candidate set.
    Sampler s(sampling_params(1.0f, 0, 0.0001f), 5);
    auto    l = logits({0.5f, 2.0f, 0.1f});
    CHECK_EQ(s.sample(l.data(), l.size()), 1);
}

TEST(sampler_temperature_widens_the_distribution) {
    auto spread = [](float temp) {
        Sampler       s(sampling_params(temp, 0, 1.0f), 99);
        std::map<int, int> seen;
        for (int i = 0; i < 400; ++i) {
            auto l = logits({2.0f, 1.0f, 0.0f, -1.0f});
            ++seen[s.sample(l.data(), l.size())];
        }
        return seen.size();
    };
    CHECK(spread(2.0f) >= spread(0.2f));
}

TEST(sampler_repetition_penalty_pushes_seen_tokens_down) {
    SamplerParams p = greedy_params();
    p.repetition_penalty = 4.0f;

    // Without history, id 1 wins. With id 1 in history it must lose.
    auto    fresh = logits({1.0f, 2.0f, 1.5f});
    Sampler a(p, 1);
    CHECK_EQ(a.sample(fresh.data(), fresh.size()), 1);

    auto    penalised = logits({1.0f, 2.0f, 1.5f});
    Sampler b(p, 1);
    b.set_prompt({1, 1, 1});
    CHECK_EQ(b.sample(penalised.data(), penalised.size()), 2);
}

TEST(sampler_penalties_apply_before_greedy_is_decided) {
    // A greedy request with a penalty must still feel the penalty; deciding
    // greedy first would skip it entirely.
    SamplerParams p    = greedy_params();
    p.presence_penalty = 5.0f;

    auto    l = logits({1.0f, 2.0f});
    Sampler s(p, 1);
    s.observe(1);
    CHECK_EQ(s.sample(l.data(), l.size()), 0);
}

TEST(sampler_frequency_penalty_scales_with_count) {
    SamplerParams p     = greedy_params();
    p.frequency_penalty = 0.4f;

    auto    once = logits({1.0f, 1.5f});
    Sampler a(p, 1);
    a.observe(1);
    CHECK_EQ(a.sample(once.data(), once.size()), 1);  // 1.5 - 0.4 = 1.1 still wins

    auto    twice = logits({1.0f, 1.5f});
    Sampler b(p, 1);
    b.observe(1);
    b.observe(1);
    CHECK_EQ(b.sample(twice.data(), twice.size()), 0);  // 1.5 - 0.8 = 0.7 loses
}

TEST(sampler_ignores_out_of_range_history_ids) {
    SamplerParams p = greedy_params();
    p.repetition_penalty = 2.0f;

    auto    l = logits({1.0f, 2.0f});
    Sampler s(p, 1);
    s.set_prompt({-1, 999999});
    CHECK_EQ(s.sample(l.data(), l.size()), 1);
}


// --------------------------------------------------------------------------
// Penalty scope. repetition_penalty spans prompt + output; presence and
// frequency are generated-only, as OpenAI and vLLM define them.

TEST(sampler_presence_penalty_ignores_the_prompt) {
    SamplerParams p    = greedy_params();
    p.presence_penalty = 5.0f;

    // Token 1 appears only in the prompt: presence must not touch it.
    auto    l = logits({1.0f, 2.0f});
    Sampler s(p, 1);
    s.set_prompt({1, 1, 1});
    CHECK_EQ(s.sample(l.data(), l.size()), 1);

    // Once generated, it is penalised.
    auto    l2 = logits({1.0f, 2.0f});
    Sampler t(p, 1);
    t.set_prompt({1, 1, 1});
    t.observe(1);
    CHECK_EQ(t.sample(l2.data(), l2.size()), 0);
}

TEST(sampler_frequency_penalty_ignores_the_prompt) {
    SamplerParams p     = greedy_params();
    p.frequency_penalty = 0.4f;

    auto    l = logits({1.0f, 1.5f});
    Sampler s(p, 1);
    s.set_prompt(std::vector<int>(50, 1));  // 50 prompt occurrences
    CHECK_EQ(s.sample(l.data(), l.size()), 1);  // untouched: 1.5 still wins
}

TEST(sampler_repetition_penalty_does_count_the_prompt) {
    SamplerParams p      = greedy_params();
    p.repetition_penalty = 4.0f;

    auto    l = logits({1.0f, 2.0f, 1.5f});
    Sampler s(p, 1);
    s.set_prompt({1});
    CHECK_EQ(s.sample(l.data(), l.size()), 2);
}

TEST(sampler_history_is_incremental_not_rebuilt) {
    // Behavioural check that observe() accumulates: three observations must
    // penalise like three, not like one.
    SamplerParams p     = greedy_params();
    p.frequency_penalty = 0.3f;

    auto    l = logits({1.0f, 1.8f});
    Sampler s(p, 1);
    s.observe(1);
    s.observe(1);
    s.observe(1);                                  // 1.8 - 0.9 = 0.9
    CHECK_EQ(s.sample(l.data(), l.size()), 0);
}

TEST(sampler_large_vocabulary_top_p_still_finds_the_mass) {
    // The candidate probe must not silently truncate the distribution: with a
    // flat 8k vocabulary, top_p 0.95 needs far more than the probe's head.
    SamplerParams p = sampling_params(1.0f, 0, 0.95f);
    Sampler       s(p, 5);

    std::vector<float> flat(8000, 1.0f);
    const int          id = s.sample(flat.data(), flat.size());
    CHECK(id >= 0);
    CHECK(id < 8000);
}

TEST(sampler_respects_min_p) {
    // llama.cpp's min-p: keep p >= min_p * p_max on the logits before
    // temperature. Logits 0, -1, -3, -8: ratios to the top 1, 0.37, 0.05,
    // 0.0003. min_p 0.1 keeps tokens 0 and 1 only, whatever the temperature.
    auto           l = logits({0.0f, -1.0f, -3.0f, -8.0f});
    std::map<int, int> seen;
    SamplerParams  p = sampling_params(2.0f, 0, 1.0f);
    p.min_p          = 0.1f;
    Sampler        s(p, 7);
    for (int i = 0; i < 400; ++i) {
        auto c = l;
        ++seen[s.sample(c.data(), c.size())];
    }
    CHECK(seen.count(0) && seen.count(1));
    CHECK(!seen.count(2) && !seen.count(3));
}

TEST(sampler_min_p_over_a_large_vocabulary_without_top_k) {
    // top_k 0 samples from a probe of the largest logits; min-p closes the
    // set inside the probe, so tokens far down the vocabulary never appear
    std::vector<float> l(200000, -20.0f);
    l[5] = 0.0f;
    l[9] = -0.5f;
    SamplerParams p = sampling_params(1.5f, 0, 1.0f);
    p.min_p         = 0.05f;
    Sampler s(p, 3);
    for (int i = 0; i < 200; ++i) {
        auto c = l;
        const int t = s.sample(c.data(), c.size());
        CHECK(t == 5 || t == 9);
    }
}

TEST(sampler_top_p_on_the_probe_uses_the_whole_vocabulary) {
    // top_k 0 takes a probe of the largest logits. 4000 equal logits: the
    // probe holds about half the mass, so top_p 0.9 needs tokens beyond it.
    // Normalised within the probe alone, the nucleus would stop inside it.
    std::vector<float> l(4000, 0.0f);
    SamplerParams p = sampling_params(1.0f, 0, 0.9f);
    Sampler s(p, 5);
    int beyond = 0;
    for (int i = 0; i < 400; ++i) {
        auto c = l;
        if (s.sample(c.data(), c.size()) >= 2048) ++beyond;
    }
    CHECK(beyond > 100);
}

TEST(sampler_min_p_with_top_p) {
    // both cuts keep a prefix of the sorted order: the result is the shorter
    // one. Logits 0, -0.1, -0.2, -5: min_p 0.5 keeps three, top_p 0.5 keeps
    // the first two (each about a third of the mass)
    auto           l = logits({0.0f, -0.1f, -0.2f, -5.0f});
    std::map<int, int> seen;
    SamplerParams  p = sampling_params(1.0f, 0, 0.5f);
    p.min_p          = 0.5f;
    Sampler        s(p, 9);
    for (int i = 0; i < 400; ++i) {
        auto c = l;
        ++seen[s.sample(c.data(), c.size())];
    }
    CHECK(seen.count(0) && seen.count(1));
    CHECK(!seen.count(2) && !seen.count(3));
}

// ---------------------------------------------------------------------------
// clean-turns: the think-block mask. Tag strings are assembled from parts here
// The mask's own cases are appended below; they take the tag strings from
// think_tag_pairs(), the same table split_reasoning() reads. Nothing in this
// file writes a tag out.

namespace {

// The mask's ids for a test: stand-in ids for the two specials, and the tags
// exactly as the reasoning table spells them.
ThinkBlockIds table_ids() {
    const auto& pairs = think_tag_pairs(false);
    ThinkBlockIds t;
    t.eog        = {0};      // stands in for the turn-ending id
    t.open_ids   = {1};      // stands in for the block's opener id
    t.close_tags = {pairs[0].second};
    t.open_tags  = {pairs[0].first};
    return t;
}

SamplerParams masked_params(bool on) {
    SamplerParams p  = greedy_params();
    p.think_eog_mask = on;
    p.think_ids      = table_ids();
    return p;
}

// The end-of-generation token is the argmax; the next best is id 3.
std::vector<float> eog_first() { return logits({5.0f, 4.0f, 1.0f, 3.0f}); }

}  // namespace

TEST(sampler_mask_off_keeps_the_end_of_generation_token) {
    // The red case: without the mask the turn ends inside the think block.
    auto    l = eog_first();
    Sampler s(masked_params(false), 1);
    s.start_in_think_block(true);
    CHECK_EQ(s.sample(l.data(), l.size()), 0);
}

TEST(sampler_mask_on_blocks_the_end_of_generation_token_while_open) {
    auto    l = eog_first();
    Sampler s(masked_params(true), 1);
    s.start_in_think_block(true);
    CHECK_EQ(s.sample(l.data(), l.size()), 3);
}

TEST(sampler_mask_on_blocks_a_second_opener_while_open) {
    // A reopened block is the same defect one step later: while the block is
    // open the opener id is not a legal next token either.
    auto    l = logits({1.0f, 5.0f, 0.5f, 3.0f});
    Sampler s(masked_params(true), 1);
    s.start_in_think_block(true);
    CHECK_EQ(s.sample(l.data(), l.size()), 3);
}

TEST(sampler_mask_clears_once_the_closer_has_been_emitted) {
    const auto& pairs = think_tag_pairs(false);
    Sampler     s(masked_params(true), 1);
    s.start_in_think_block(true);
    s.observe_think_text("some reasoning");
    CHECK_EQ(s.sample(eog_first().data(), 4), 3);            // still blocked
    s.observe_think_text(pairs[0].second);
    CHECK_EQ(s.sample(eog_first().data(), 4), 0);            // the turn may end
}

TEST(sampler_mask_engages_when_the_model_opens_the_block_itself) {
    // The template did not open it; the model's own opener, at the lead of the
    // generation, engages the mask.
    const auto& pairs = think_tag_pairs(false);
    Sampler     s(masked_params(true), 1);
    s.start_in_think_block(false);
    CHECK_EQ(s.sample(eog_first().data(), 4), 0);            // not open: allowed
    s.observe_think_text(pairs[0].first + "why");
    CHECK_EQ(s.sample(eog_first().data(), 4), 3);            // open from here
}

TEST(sampler_mask_ignores_an_opener_that_is_not_at_the_lead) {
    // Mirrors split_reasoning: only an opener at the very start opens a block.
    // An answer that quotes a tag later must still be able to end normally.
    const auto& pairs = think_tag_pairs(false);
    Sampler     s(masked_params(true), 1);
    s.start_in_think_block(false);
    s.observe_think_text("the template writes " + pairs[0].first + " first");
    CHECK_EQ(s.sample(eog_first().data(), 4), 0);
}

TEST(sampler_mask_off_is_the_old_behaviour_even_with_the_ids_set) {
    auto    l = eog_first();
    Sampler s(masked_params(false), 1);
    s.start_in_think_block(true);
    s.observe_think_text(think_tag_pairs(false)[0].first);
    CHECK_EQ(s.sample(l.data(), l.size()), 0);
}

TEST(sampler_mask_leaves_every_other_token_alone) {
    // The mask names two ids. Nothing else in the row moves, so a normal
    // answer is sampled exactly as it was before. (sample() rewrites its row:
    // each call gets a copy, as every decode loop does.)
    const std::vector<float> row = logits({5.0f, 4.0f, 2.5f, 3.0f});

    auto    a = row;
    Sampler s(masked_params(true), 1);
    s.start_in_think_block(true);
    CHECK_EQ(s.sample(a.data(), a.size()), 3);
    size_t changed = 0;
    for (size_t i = 0; i < row.size(); ++i)
        if (a[i] != row[i]) ++changed;
    CHECK_EQ(static_cast<int>(changed), 2);   // the turn-ender and the opener

    auto    b = row;
    Sampler s2(masked_params(false), 1);
    s2.start_in_think_block(true);
    CHECK_EQ(s2.sample(b.data(), b.size()), 0);
    CHECK(b == row);                          // off: the row is not touched
}

TEST(sampler_mask_notices_a_closer_split_across_pieces) {
    // Cut from the table's own string, never written out here: a closer that
    // straddles two emitted pieces still clears the mask.
    const std::string closer = think_tag_pairs(false)[0].second;
    const size_t      cut    = closer.size() / 2;
    Sampler           s(masked_params(true), 1);
    s.start_in_think_block(true);
    s.observe_think_text(closer.substr(0, cut));
    CHECK_EQ(s.sample(eog_first().data(), 4), 3);            // not closed yet
    s.observe_think_text(closer.substr(cut));
    CHECK_EQ(s.sample(eog_first().data(), 4), 0);
}

TEST(sampler_mask_notices_a_closer_inside_one_long_piece) {
    // The tail kept for the straddle is never shorter than the piece just
    // appended: a closer that lands wholly inside a long piece is still seen.
    const std::string closer = think_tag_pairs(false)[0].second;
    Sampler           s(masked_params(true), 1);
    s.start_in_think_block(true);
    s.observe_think_text(std::string(closer.size() * 3, 'x') + closer + std::string(closer.size() * 3, 'y'));
    CHECK_EQ(s.sample(eog_first().data(), 4), 0);
}

TEST(sampler_mask_engages_on_an_opener_split_across_pieces) {
    // Cut from the table's own string. The rule is one-sided on purpose: a lead
    // that could still grow into an opener is masked from the first character,
    // because masking the turn-ender for the two or three tokens of a partial
    // opener costs nothing, while releasing a turn the client still reads as
    // reasoning is the defect this is about.
    const std::string opener = think_tag_pairs(false)[0].first;
    const size_t      cut    = opener.size() / 2;
    Sampler           s(masked_params(true), 1);
    s.start_in_think_block(false);
    s.observe_think_text(opener.substr(0, cut));
    CHECK_EQ(s.sample(eog_first().data(), 4), 3);            // already masked
    s.observe_think_text(opener.substr(cut) + "because");
    CHECK_EQ(s.sample(eog_first().data(), 4), 3);            // and stays so
}

TEST(sampler_mask_lets_go_when_the_lead_is_plainly_not_an_opener) {
    // The other side of the same rule: an answer that starts as content is not
    // held back for the length of a tag.
    const std::string opener = think_tag_pairs(false)[0].first;
    Sampler           s(masked_params(true), 1);
    s.start_in_think_block(false);
    s.observe_think_text(opener.substr(0, 1) + "n ordinary sentence");
    CHECK_EQ(s.sample(eog_first().data(), 4), 0);
}
