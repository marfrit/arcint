#include "core/sampler.h"

#include "core/reasoning.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>

namespace lgc {
namespace {

// How many candidates to pull before falling back to a full sort. Softmax mass
// concentrates hard on a 248k vocabulary, so this covers top_p in practice
// while avoiding a quarter-million-element sort per token. Correctness does not
// depend on it: the probe's probabilities are normalised over the whole
// vocabulary, so a probe short of the top_p mass is seen and falls back.
constexpr size_t kCandidateProbe = 2048;

}  // namespace

Sampler::Sampler(const SamplerParams& params, uint64_t seed)
    : params_(params), seed_(seed), rng_(seed) {}

void Sampler::set_prompt(const std::vector<int>& prompt) {
    seen_.clear();
    generated_.clear();
    answer_generated_.clear();
    answer_closer_now_ = false;
    seen_.reserve(prompt.size());
    for (int id : prompt) {
        if (id >= 0) ++seen_[id];
    }
}

void TokenConstraint::note_desync(int token, const std::string& why) {
    if (!note_desynced_) {
        note_desynced_ = true;
        ++desyncs_;
    }
    // Not warn: a broken constraint is an incident, and the error line is what
    // the log grep keeps. Latched, not once: a second loss of sync in the same
    // request is re-logged, and only the count refuses to run away per token.
    log::error("grammar", "desync: %s (token %d); the rest of this request is unconstrained", why.c_str(), token);
}

void Sampler::observe(int token, std::string_view piece) {
    if (token < 0) return;
    ++seen_[token];
    ++generated_[token];
    // The think-block tracker sees the (id, text) pair of every generated
    // token, in generation order. The id decides when the vocabulary spells a
    // tag as one special token; the text is the fallback otherwise.
    observe_think_(piece, token);
    // A token inside an open think block is reasoning, not a call: arcint
    // parses calls from the content only (split_reasoning), so the grammar is
    // not fed it -- a call opener the model mentions while thinking does not
    // trigger it. A closer is fed (harmless: the lazy grammar is still
    // waiting) and the first token after it is the first the grammar can
    // trigger on.
    if (constraint_ != nullptr && think_state_ != ThinkState::thinking) constraint_->accept(token);
    // llava: the answer row's history starts at the closer, not with it. The
    // token that closed the block is reasoning's last token; it is not counted
    // here, and the flag is consumed so the next token is. Inert before the
    // switch and in a turn that never switches; filled only when the answer row
    // owns a penalty, so an answer row without one keeps today's history.
    const bool closer_here = answer_closer_now_;
    answer_closer_now_ = false;
    if (use_answer_row()
        && (params_.answer_presence_penalty || params_.answer_frequency_penalty)
        && !closer_here)
        ++answer_generated_[token];
}

void Sampler::observe_think_text(std::string_view piece) { observe_think_(piece, -1); }

void Sampler::observe_think(int token, std::string_view piece) { observe_think_(piece, token); }

void Sampler::set_think_mask_ids(ThinkBlockIds ids) {
    params_.think_ids = std::move(ids);
    measure_think_tags();
}

// The tail kept to notice a closer that straddles two pieces is bounded by the
// longest tag, so the check costs nothing per step. Both entry points measure:
// the ids may arrive in the params (a test, or a backend that already knows
// them) or through set_think_mask_ids.
void Sampler::measure_think_tags() {
    think_max_tag_  = 0;
    think_max_open_ = 0;
    for (const std::string& t : params_.think_ids.close_tags) think_max_tag_ = std::max(think_max_tag_, t.size());
    for (const std::string& t : params_.think_ids.open_tags) {
        think_max_tag_  = std::max(think_max_tag_, t.size());
        think_max_open_ = std::max(think_max_open_, t.size());
    }
}

// The id rule's table: which open ids close which close ids, from the
// per-pair field the backend fills (ThinkBlockIds::open_close_ids, aligned
// with the tag pairs). A block opened by one pair's opener is closed by that
// pair's closer, exactly as the split pairs them. For a caller that only fills
// the flattened id lists, the ids are zipped positionally.
void Sampler::build_think_id_table() {
    think_open_to_close_.clear();
    think_has_open_ids_ = false;
    if (!params_.think_ids.open_close_ids.empty()) {
        for (const auto& [oid, cid] : params_.think_ids.open_close_ids) {
            if (oid < 0) continue;
            think_has_open_ids_ = true;
            think_open_to_close_.emplace_back(oid, cid);
        }
        return;
    }
    for (size_t i = 0; i < params_.think_ids.open_ids.size(); ++i) {
        const int oid = params_.think_ids.open_ids[i];
        if (oid < 0) continue;
        think_has_open_ids_ = true;
        const int cid = i < params_.think_ids.close_ids.size() ? params_.think_ids.close_ids[i] : -1;
        think_open_to_close_.emplace_back(oid, cid);
    }
}

void Sampler::start_in_think_block(bool open) {
    measure_think_tags();
    build_think_id_table();
    // The template wrote the opener as the lead of the turn: the block is
    // already open before the first token. Otherwise the first tokens decide.
    think_state_ = (tracks_think() && open) ? ThinkState::thinking : ThinkState::undecided;
    // llava: the answer row needs a block that actually opened, not merely a
    // request that declared thinking.
    think_block_ever_open_ = think_state_ == ThinkState::thinking;
    answer_generated_.clear();
    answer_closer_now_ = false;
    think_tail_.clear();
    think_lead_.clear();
    // The closer of a template-opened block: the first pair's close id, or the
    // first flattened close id. -1 keeps the text fallback.
    int cid = -1;
    if (!params_.think_ids.open_close_ids.empty()) cid = params_.think_ids.open_close_ids.front().second;
    else if (!params_.think_ids.close_ids.empty()) cid = params_.think_ids.close_ids.front();
    think_close_id_ = think_state_ == ThinkState::thinking && cid >= 0 ? cid : -1;
}

// Feed one generated token -- its text and its id -- to the think-block
// tracker, in generation order. The states: undecided (the opening is not
// settled yet), thinking (the block is open) or content (settled: the block
// closed, or never opened).
//
// The id rule: when the vocabulary spells a tag as one special token, that
// role is decided by the id and the decoded text is not looked at for it. A
// model that QUOTES the closer, spelling it from ordinary sub-word tokens,
// produces identical text but a different id; the text path read that as the
// real closer (reasoning ended early, planning spilled into the answer). A tag
// with no id (the stub backend, a template whose vocabulary spells it as text)
// keeps the text fallback below, exactly as before.
//
// The undecided state examines the stripped lead exactly as lead_of
// split_reasoning does, so the mask and the reasoning split agree about where
// the block is. Once content the state is final for the turn: nothing
// re-opens the block.
void Sampler::observe_think_(std::string_view piece, int token) {
    if (!tracks_think()) return;

    // loop-guard: feed the detector from the same piece stream the tracker
    // sees. Reasoning is watched while the block is open; the answer phase is
    // watched only after a slap (the abort rung), never otherwise.
    if (params_.loop_guard != LoopGuardMode::Off) {
        // The last nudge id's piece is observed after sample() returned it; the
        // sentence is not reasoning the detector should hold, so the buffer is
        // dropped before that piece lands (the sample() side only marks it).
        if (loop_nudge_flush_) {
            loop_nudge_flush_ = false;
            loop_line_.clear();
        }
        if (think_state_ == ThinkState::thinking
            || (think_state_ == ThinkState::content && loop_slapped_))
            loop_observe_(piece);
    }

    // The closer id: the block that is open closes only on it. A spelled-out
    // closer is ordinary tokens and cannot get here.
    if (think_state_ == ThinkState::thinking && think_close_id_ >= 0 && token == think_close_id_) {
        think_state_ = ThinkState::content;
        think_tail_.clear();
        think_lead_.clear();
        answer_closer_now_ = true;   // llava: this token is the answer history's edge
        return;
    }

    // The opener id: the model opened the block itself with one special token.
    // It pairs with that tag's own closer.
    if (think_state_ == ThinkState::undecided && token >= 0) {
        for (const auto& [oid, cid] : think_open_to_close_) {
            if (token != oid) continue;
            think_state_    = ThinkState::thinking;
            think_block_ever_open_ = true;
            think_close_id_ = cid;
            think_lead_.clear();
            think_tail_.clear();
            return;   // the opener token's whole piece is the marker
        }
    }

    if (think_state_ == ThinkState::undecided) {
        // N1: err towards STAYING undecided. A lead that could still grow into
        // an opener keeps the state undecided; a full opener match opens it; a
        // lead that can no longer grow into any opener settles to content.
        // Committing a merely looming lead to thinking (the old `matched ||
        // loomed`) is the defect this settles: if the text then diverges, no
        // closer ever comes, every turn-ender stays masked, and the turn runs
        // to the length cap. A pair whose opener is one special id keeps its
        // id rule here: its text can neither open nor settle the block.
        think_lead_.append(piece);
        // Strip the lead exactly as lead_of does, so a pad of spaces and
        // newlines before a self-opened block is not part of the opener and
        // the mask engages where the split would. The lead is bounded by the
        // longest OPENER (a closer can never be the lead, so its length is
        // irrelevant here): the decision needs no more, so a long plain
        // answer costs nothing.
        const std::string_view lead = lead_of(think_lead_);
        if (lead.size() != think_lead_.size()) think_lead_.erase(0, think_lead_.size() - lead.size());
        if (think_lead_.size() > think_max_open_) think_lead_.resize(think_max_open_);

        bool matched = false;
        bool loomed  = false;
        size_t matched_pair = 0;
        for (size_t i = 0; i < params_.think_ids.open_tags.size(); ++i) {
            // A pair whose opener is a known id is decided by that id alone.
            if (!params_.think_ids.open_close_ids.empty() && i < params_.think_ids.open_close_ids.size()
                && params_.think_ids.open_close_ids[i].first >= 0)
                continue;
            const std::string& t = params_.think_ids.open_tags[i];
            if (think_lead_.size() >= t.size()) {
                if (think_lead_.compare(0, t.size(), t) == 0) {
                    matched      = true;
                    matched_pair = i;
                    break;
                }
            } else if (!think_lead_.empty() && t.compare(0, think_lead_.size(), think_lead_) == 0) {
                loomed = true;
            }
        }
        if (matched) {
            // A full text opener at the lead, for a pair with no id: open from
            // here, paired with its own closer id when that exists; otherwise
            // the closer scan below runs on the rest of this piece.
            think_state_ = ThinkState::thinking;
            think_block_ever_open_ = true;
            think_close_id_ =
                !params_.think_ids.open_close_ids.empty() && matched_pair < params_.think_ids.open_close_ids.size()
                    ? params_.think_ids.open_close_ids[matched_pair].second
                    : (!params_.think_ids.close_ids.empty() && matched_pair < params_.think_ids.close_ids.size()
                           ? params_.think_ids.close_ids[matched_pair]
                           : -1);
            think_lead_.clear();
        } else if (loomed || think_lead_.empty()) {
            return;   // nothing is open; no closer scan on this piece
        } else {
            think_state_ = ThinkState::content;
            think_lead_.clear();
            return;
        }
    }

    if (think_state_ != ThinkState::thinking) return;

    // thinking. A known closer id is the whole rule above: no text can close
    // the block, so a spelled-out closer stays reasoning and there is nothing
    // to scan. Only the text fallback watches the tail.
    if (think_close_id_ >= 0) return;

    think_tail_.append(piece);
    // Keep enough of the tail to see a closer that straddles two pieces, and
    // never less than the piece just appended: a closer wholly inside one long
    // piece must still be seen.
    const size_t keep = std::max(think_max_tag_ * 2 + 8, piece.size());
    if (think_tail_.size() > keep) think_tail_.erase(0, think_tail_.size() - keep);
    for (const std::string& t : params_.think_ids.close_tags) {
        if (think_tail_.find(t) != std::string::npos) {
            // Settled: content. Both windows are cleared, so neither the
            // stale lead nor a tail remnant can re-open the block later.
            think_state_ = ThinkState::content;
            think_tail_.clear();
            think_lead_.clear();
            answer_closer_now_ = true;   // llava: the text closer is the history's edge
            return;
        }
    }
}

// loop-guard: the detector. A line ends at a newline in the decoded text and
// counts only when it is longer than 20 characters (an empty or one-word line
// says nothing about repetition). A line is a repeat when the same text
// occurred earlier in this turn. The share is the repeat count over the newest
// window lines; it trips at/above the threshold once at least 10 lines (or the
// window, when smaller) are in hand, so two identical lines at the start of a
// turn cannot pass for a loop. Everything is incremental: appending a piece
// and testing the newline is the only per-token work.
void Sampler::loop_observe_(std::string_view piece) {
    loop_line_.append(piece);
    size_t nl = 0;
    while ((nl = loop_line_.find('\n')) != std::string::npos) {
        const std::string line = loop_line_.substr(0, nl);
        loop_line_.erase(0, nl + 1);
        loop_line_ended_(line);
    }
}

// The near rule's 3-grams, hashed with FNV-1a and kept sorted/deduped so the
// Jaccard of two lines is a merge over two small sorted vectors. One pass per
// ended line; the text is never rescanned.
Sampler::LoopGrams Sampler::loop_grams_(const std::string& line) const {
    LoopGrams grams;
    if (line.size() < 3) return grams;
    grams.reserve(line.size() - 2);
    for (size_t i = 0; i + 2 < line.size(); ++i) {
        uint32_t h = 2166136261u;
        for (size_t k = 0; k < 3; ++k)
            h = (h ^ static_cast<unsigned char>(line[i + k])) * 16777619u;
        grams.push_back(h);
    }
    std::sort(grams.begin(), grams.end());
    grams.erase(std::unique(grams.begin(), grams.end()), grams.end());
    return grams;
}

bool Sampler::loop_near_match_(const LoopGrams& grams) const {
    for (const LoopGrams& prev : loop_gram_memory_) {
        size_t inter = 0;
        size_t i = 0, j = 0;
        while (i < grams.size() && j < prev.size()) {
            if (grams[i] < prev[j]) ++i;
            else if (grams[i] > prev[j]) ++j;
            else { ++inter; ++i; ++j; }
        }
        const size_t uni = grams.size() + prev.size() - inter;
        if (uni > 0 && static_cast<float>(inter) >= kLoopGuardNearJaccard * static_cast<float>(uni))
            return true;
    }
    return false;
}

void Sampler::loop_line_ended_(const std::string& line) {
    if (line.size() <= 20) return;

    // loop_observe_ is only fed from the gates in observe_think_: reasoning
    // while the block is open, content only after a slap. The phase test here
    // must not carry a second copy of the slap condition, or the gate and the
    // phase latch can disagree about what the answer phase is.
    const bool thinking = think_state_ == ThinkState::thinking;
    const bool content  = think_state_ == ThinkState::content;
    if (!thinking && !content) return;

    if (thinking && loop_nudged_ && !loop_slapped_) ++loop_since_nudge_;
    const bool exact_repeat = loop_seen_[line]++ > 0;
    bool       repeated     = exact_repeat;
    if (!exact_repeat && params_.loop_guard_match == LoopGuardMatch::Near) {
        LoopGrams grams = loop_grams_(line);
        repeated        = loop_near_match_(grams);
        // Every distinct counted line joins the near memory, the near matches
        // too: a counting loop's number lines are distinct text and must be
        // comparable with the next number line. Exact repeats already sit in
        // the memory once.
        loop_gram_memory_.push_back(std::move(grams));
        while (loop_gram_memory_.size() > kLoopGuardNearMemory) loop_gram_memory_.pop_front();
    }
    loop_window_.push_back(LoopLine{line, repeated});
    const size_t window = static_cast<size_t>(std::max(1, params_.loop_guard_window));
    while (loop_window_.size() > window) loop_window_.pop_front();

    const size_t floor = std::min<size_t>(window, 10);
    if (loop_window_.size() < floor) return;
    size_t repeats = 0;
    for (const LoopLine& l : loop_window_) repeats += l.repeated ? 1 : 0;
    const float share = static_cast<float>(repeats) / static_cast<float>(loop_window_.size());
    const bool  over     = share >= params_.loop_guard_threshold;
    // Hysteresis: low is re-armed only below (threshold - 0.1). A nudge's own
    // new line pushes the share just under the threshold; that dip must not
    // make the next repeat count as a second detection. A real fall below the
    // low mark is a new loop onset when the share rises again.
    const float low_mark = params_.loop_guard_threshold - 0.1f;
    if (share < low_mark) loop_low_ = true;
    const bool crossing = over && loop_low_;

    if (thinking) {
        if (crossing) {
            loop_low_ = false;
            ++loop_detections_;
            log::info("loop-guard", "lane: reasoning loop detected: %zu of %zu recent lines repeat (%.2f), after %zu lines",
                      repeats, loop_window_.size(), share, loop_seen_.size());
            loop_on_detect_(false);
        }
        // The slap's grace keeps watching every over line, dips included: a
        // share at/above the threshold after K counted lines still slaps.
        if (over && params_.loop_guard == LoopGuardMode::Ladder && loop_nudged_ && !loop_slapped_
            && loop_since_nudge_ >= params_.loop_guard_grace) {
            loop_do_slap_();
        }
    } else if (content && over && !loop_abort_) {
        loop_do_abort_();
    }
}

void Sampler::loop_on_detect_(bool content_phase) {
    if (content_phase) {
        if (loop_slapped_) loop_do_abort_();
        return;
    }
    switch (params_.loop_guard) {
        case LoopGuardMode::Off:    return;
        case LoopGuardMode::Detect: return;   // counted and logged above
        case LoopGuardMode::Nudge:
        case LoopGuardMode::Ladder: loop_do_nudge_(); return;
        case LoopGuardMode::Slap:   loop_do_slap_(); return;
    }
}

void Sampler::loop_do_nudge_() {
    if (loop_nudged_) return;
    if (params_.loop_guard_nudge_ids.empty()) {
        log::warn("loop-guard", "the nudge rung has no tokenised sentence (the backend did not set it); no nudge");
        return;
    }
    loop_nudged_     = true;
    loop_nudge_cursor_ = 0;
    loop_since_nudge_  = 0;
    ++loop_nudges_;
    log::info("loop-guard", "lane: nudge injected (%zu tokens), watching for %d more lines",
              params_.loop_guard_nudge_ids.size(), params_.loop_guard_grace);
}

void Sampler::loop_do_slap_() {
    if (loop_slapped_) return;
    loop_slapped_   = true;
    loop_slap_pending_ = true;
    ++loop_slaps_;
    // The answer phase gets its own window: the reasoning loop that caused the
    // slap must not count as the first answer lines (that would abort before
    // the model has said anything). Content lines are counted from here.
    loop_window_.clear();
    loop_seen_.clear();
    loop_gram_memory_.clear();
    loop_line_.clear();
    loop_low_ = true;
    log::info("loop-guard", "lane: slap: the think-block's closer is forced as the next token");
}

void Sampler::loop_do_abort_() {
    if (loop_abort_) return;
    loop_abort_ = true;
    ++loop_aborts_;
    log::warn("loop-guard", "lane: the answer phase is looping after a slap; ending the turn (finish loop)");
}

// While the block is open the turn-ending ids and a second opener are not
// legal next tokens. The turn-ending set is exactly the one the decode loop's
// stop check ends on (is_stop): the vocabulary's EOG ids, unless the request
// ignores them, plus the client's stop ids. Both mechanisms read the same
// fields of the same SamplerParams -- the EOG list the backend fills from the
// vocabulary's predicate, the client list straight from the request -- so the
// two are one source, not two lists: a stop id the check honours is one the
// mask holds back while the block is open, and an id the check would pass
// (ignore_eos) is offered, not masked.
// Applied to the logits, not to the stop check: a filter in the stop check
// would let the token be sampled, take its probability mass, and then be
// ignored, so the sampled token would not be the one the distribution
// describes.
void Sampler::mask_illegal_while_thinking(float* logits, size_t vocab) const {
    if (!params_.think_eog_mask || think_state_ != ThinkState::thinking) return;
    // Review B3: the closer must never be in the masked set. If an artifact's
    // end-of-thinking marker were one of the turn-ending ids -- or a client
    // stop id -- the mask would make the block unclosable and the turn could
    // end only at the length cap. The guard strips it from every source the
    // mask kills, and says so once.
    auto is_closer = [&](int id) {
        return std::find(params_.think_ids.close_ids.begin(), params_.think_ids.close_ids.end(), id)
               != params_.think_ids.close_ids.end();
    };
    auto kill = [&](const std::vector<int>& ids) {
        for (int id : ids) {
            if (is_closer(id)) {
                if (!closer_stripped_) {
                    closer_stripped_ = true;
                    log::warn("mask", "the think block's closer id %d is in the turn-ending set; "
                                      "the mask leaves it offered so the block can close",
                              id);
                }
                continue;
            }
            if (id >= 0 && static_cast<size_t>(id) < vocab) logits[static_cast<size_t>(id)] = -INFINITY;
        }
    };
    if (!params_.ignore_eos) kill(params_.think_ids.eog);
    kill(params_.stop_token_ids);
    kill(params_.think_ids.open_ids);
}

int Sampler::argmax(const float* logits, size_t vocab) {
    size_t best = 0;
    float  top  = logits[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (logits[i] > top) {
            top  = logits[i];
            best = i;
        }
    }
    return static_cast<int>(best);
}

void Sampler::apply_penalties(float* logits, size_t vocab) const {
    // llava: the effective row decides both the values and the history. When
    // the answer row owns a penalty (either answer_* field is configured), its
    // presence/frequency counts read only the tokens drawn after the closer --
    // the identifiers the reasoning just worked out are not docked. With no
    // answer penalty configured the fields fall back to the thinking row's
    // values and read the whole generated history, byte-identically to before.
    // repetition_penalty is unchanged: it counts the prompt too, by design
    // (set_prompt), and is not part of the switch.
    const EffectiveRow row = effective_row();
    const bool rep  = params_.repetition_penalty != 1.0f;
    const bool pres = row.presence_penalty != 0.0f;
    const bool freq = row.frequency_penalty != 0.0f;
    const bool answer_hist = row.answer_row
                             && (params_.answer_presence_penalty || params_.answer_frequency_penalty);
    const auto& hist = answer_hist ? answer_generated_ : generated_;

    if (rep) {
        for (const auto& [id, n] : seen_) {
            (void)n;
            if (static_cast<size_t>(id) >= vocab) continue;
            float& l = logits[static_cast<size_t>(id)];
            // llama.cpp's convention: dividing a positive logit and multiplying
            // a negative one both push the token down.
            l = l > 0.0f ? l / params_.repetition_penalty : l * params_.repetition_penalty;
        }
    }
    if (pres || freq) {
        for (const auto& [id, n] : hist) {
            if (static_cast<size_t>(id) >= vocab) continue;
            float& l = logits[static_cast<size_t>(id)];
            if (pres) l -= row.presence_penalty;
            if (freq) l -= row.frequency_penalty * static_cast<float>(n);
        }
    }
}

// Fills scratch_ with the candidates worth considering, largest logit first.
// Returns how many are usable.
size_t Sampler::collect_candidates(const float* logits, size_t vocab, bool full, int top_k) {
    const size_t want = full ? vocab
                        : top_k > 0
                            ? std::min(static_cast<size_t>(top_k), vocab)
                            : std::min(kCandidateProbe, vocab);

    scratch_.clear();
    scratch_.reserve(vocab);
    for (size_t i = 0; i < vocab; ++i) scratch_.emplace_back(logits[i], static_cast<int>(i));

    auto by_logit = [](const auto& a, const auto& b) { return a.first > b.first; };
    if (want < vocab) {
        std::partial_sort(scratch_.begin(), scratch_.begin() + static_cast<long>(want),
                          scratch_.end(), by_logit);
        scratch_.resize(want);
    } else {
        std::sort(scratch_.begin(), scratch_.end(), by_logit);
    }
    return scratch_.size();
}

int Sampler::sample(float* logits, size_t vocab) {
    if (vocab == 0) return 0;

    // loop-guard: a forced token is returned at the draw boundary, before the
    // mask and the penalties -- it is not a draw from the model's row, and
    // changing that row would not change what this step emits. The caller
    // emits it, observes it (history and tracker) and decodes it into the KV
    // exactly as a drawn token, which is how the nudge lands in
    // reasoning_content, the stream and the history.
    if (params_.loop_guard != LoopGuardMode::Off) {
        if (loop_nudge_cursor_ >= 0) {
            const int id = params_.loop_guard_nudge_ids[static_cast<size_t>(loop_nudge_cursor_)];
            ++loop_nudge_cursor_;
            if (loop_nudge_cursor_ >= static_cast<int>(params_.loop_guard_nudge_ids.size())) {
                loop_nudge_cursor_ = -1;      // the sentence is delivered once
                loop_nudge_flush_  = true;    // observe_think_ clears before its piece
            }
            return id;
        }
        if (loop_slap_pending_) {
            loop_slap_pending_ = false;
            int closer = think_close_id_ >= 0 ? think_close_id_
                         : (!params_.think_ids.close_ids.empty() ? params_.think_ids.close_ids.front() : -1);
            if (closer >= 0 && static_cast<size_t>(closer) < vocab) return closer;
            if (!loop_closer_warned_) {
                loop_closer_warned_ = true;
                log::warn("loop-guard", "the slap has no closer id in this vocabulary; the block cannot be forced shut");
            }
        }
    }

    mask_illegal_while_thinking(logits, vocab);
    apply_penalties(logits, vocab);

    const int tok = draw(logits, vocab);
    if (constraint_ == nullptr || think_state_ == ThinkState::thinking || constraint_->allows(tok)) return tok;
    // Grammar-based rejection sampling, as the reference's
    // common_sampler_sample: the drawn token does not fit, so mask the row
    // (after the think mask and the penalties, as the reference re-runs its
    // chain over the masked candidates) and draw again. A rejection costs one
    // full-vocabulary grammar pass; a token that fits costs one token's test.
    // Guarded like allows(): the constraint's own guard handles its engine
    // (broken_, counted, fail open), and this backstop handles a constraint
    // whose mask() throws anyway -- counted once, the constraint detached so
    // the rest of the request runs unconstrained, and the redraw draws from
    // the unmasked row, instead of the throw escaping sample() as a 500
    // mid-stream. The backend reads the count through its own pointer, so the
    // stats line still carries it.
    try {
        constraint_->mask(logits, vocab);
    } catch (const std::exception& e) {
        constraint_->note_desync(tok, e.what());
        constraint_ = nullptr;
        return draw(logits, vocab);
    }
    return draw(logits, vocab);
}

Sampler::EffectiveRow Sampler::effective_row() const {
    const bool answer = use_answer_row();
    EffectiveRow r;
    r.answer_row  = answer;
    r.temperature = answer && params_.answer_temperature ? *params_.answer_temperature : params_.temperature;
    r.top_p       = answer && params_.answer_top_p ? *params_.answer_top_p : params_.top_p;
    r.top_k       = answer && params_.answer_top_k ? *params_.answer_top_k : params_.top_k;
    r.min_p       = answer && params_.answer_min_p ? *params_.answer_min_p : params_.min_p;
    r.presence_penalty  = answer && params_.answer_presence_penalty
                              ? *params_.answer_presence_penalty : params_.presence_penalty;
    r.frequency_penalty = answer && params_.answer_frequency_penalty
                              ? *params_.answer_frequency_penalty : params_.frequency_penalty;
    return r;
}

int Sampler::draw(float* logits, size_t vocab) {
    // llava: one row per draw. The state cannot move inside sample(), so the
    // row decided here is the row this token is drawn with -- also in a verify
    // walk, where observe() advances the state before the next row's sample().
    const EffectiveRow row = effective_row();
    // Penalties can reorder the distribution, so greedy is decided after them.
    if (row.temperature <= 0.0f || row.top_k == 1) return argmax(logits, vocab);

    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool   probing = attempt == 0 && row.top_k <= 0 && kCandidateProbe < vocab;
        const size_t n       = collect_candidates(logits, vocab, attempt > 0, row.top_k);
        if (n == 0) return argmax(logits, vocab);

        const float temp = row.temperature > 0.0f ? row.temperature : 1.0f;
        const float max  = scratch_.front().first;

        double              sum = 0.0;
        std::vector<double> probs(scratch_.size());
        for (size_t i = 0; i < scratch_.size(); ++i) {
            probs[i] = std::exp(static_cast<double>(scratch_[i].first - max) / temp);
            sum += probs[i];
        }
        // A probe holds part of the vocabulary: normalise over all of it, so
        // top_p measures the true mass and a short probe is noticed
        if (probing) {
            sum = 0.0;
            for (size_t i = 0; i < vocab; ++i) sum += std::exp(static_cast<double>(logits[i] - max) / temp);
        }
        for (double& p : probs) p /= sum;

        // min-p as llama.cpp applies it (src/llama-sampler.cpp,
        // llama_sampler_min_p_apply): keep the logits at or above
        // max + ln(min_p), i.e. p >= min_p * p_max before temperature. The
        // candidates are sorted, so it keeps a prefix; once a candidate falls
        // below the floor, so does the rest of the vocabulary, and the probe
        // needs no full sort.
        size_t min_p_cut    = probs.size();
        bool   min_p_closed = false;
        if (row.min_p > 0.0f) {
            const float floor = max + std::log(row.min_p);
            min_p_cut         = 1;
            while (min_p_cut < scratch_.size() && scratch_[min_p_cut].first >= floor) ++min_p_cut;
            min_p_closed = min_p_cut < scratch_.size();
        }

        // top-p over the already-sorted order. At least one candidate always
        // survives, so a tiny top_p degenerates to greedy rather than nothing.
        size_t cutoff = probs.size();
        bool   short_of_mass = false;
        if (row.top_p < 1.0f) {
            double acc = 0.0;
            cutoff     = probs.size();
            for (size_t i = 0; i < probs.size(); ++i) {
                acc += probs[i];
                if (acc >= static_cast<double>(row.top_p)) {
                    cutoff = i + 1;
                    break;
                }
            }
            // The probe held less mass than top_p asked for; only then is the
            // full sort actually needed.
            if (acc < static_cast<double>(row.top_p) && probing && !min_p_closed) short_of_mass = true;
        } else if (probing && scratch_.size() < vocab && !min_p_closed) {
            short_of_mass = true;
        }
        cutoff = std::min(cutoff, min_p_cut);

        if (short_of_mass) continue;   // the full set next pass (this token only)

        double renorm = 0.0;
        for (size_t i = 0; i < cutoff; ++i) renorm += probs[i];

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double                                 r = dist(rng_) * renorm;
        for (size_t i = 0; i < cutoff; ++i) {
            r -= probs[i];
            if (r <= 0.0) return scratch_[i].second;
        }
        return scratch_[cutoff - 1].second;
    }
    return argmax(logits, vocab);
}

}  // namespace lgc
