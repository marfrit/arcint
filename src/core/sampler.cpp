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
}

void Sampler::observe_think_text(std::string_view piece) { observe_think_(piece, -1); }

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

    // The closer id: the block that is open closes only on it. A spelled-out
    // closer is ordinary tokens and cannot get here.
    if (think_state_ == ThinkState::thinking && think_close_id_ >= 0 && token == think_close_id_) {
        think_state_ = ThinkState::content;
        think_tail_.clear();
        think_lead_.clear();
        return;
    }

    // The opener id: the model opened the block itself with one special token.
    // It pairs with that tag's own closer.
    if (think_state_ == ThinkState::undecided && token >= 0) {
        for (const auto& [oid, cid] : think_open_to_close_) {
            if (token != oid) continue;
            think_state_    = ThinkState::thinking;
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
            return;
        }
    }
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
    const bool rep  = params_.repetition_penalty != 1.0f;
    const bool pres = params_.presence_penalty != 0.0f;
    const bool freq = params_.frequency_penalty != 0.0f;

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
        for (const auto& [id, n] : generated_) {
            if (static_cast<size_t>(id) >= vocab) continue;
            float& l = logits[static_cast<size_t>(id)];
            if (pres) l -= params_.presence_penalty;
            if (freq) l -= params_.frequency_penalty * static_cast<float>(n);
        }
    }
}

// Fills scratch_ with the candidates worth considering, largest logit first.
// Returns how many are usable.
size_t Sampler::collect_candidates(const float* logits, size_t vocab, bool full) {
    const size_t want = full ? vocab
                        : params_.top_k > 0
                            ? std::min(static_cast<size_t>(params_.top_k), vocab)
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

int Sampler::draw(float* logits, size_t vocab) {
    // Penalties can reorder the distribution, so greedy is decided after them.
    if (params_.greedy()) return argmax(logits, vocab);

    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool   probing = attempt == 0 && params_.top_k <= 0 && kCandidateProbe < vocab;
        const size_t n       = collect_candidates(logits, vocab, attempt > 0);
        if (n == 0) return argmax(logits, vocab);

        const float temp = params_.temperature > 0.0f ? params_.temperature : 1.0f;
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
        if (params_.min_p > 0.0f) {
            const float floor = max + std::log(params_.min_p);
            min_p_cut         = 1;
            while (min_p_cut < scratch_.size() && scratch_[min_p_cut].first >= floor) ++min_p_cut;
            min_p_closed = min_p_cut < scratch_.size();
        }

        // top-p over the already-sorted order. At least one candidate always
        // survives, so a tiny top_p degenerates to greedy rather than nothing.
        size_t cutoff = probs.size();
        bool   short_of_mass = false;
        if (params_.top_p < 1.0f) {
            double acc = 0.0;
            cutoff     = probs.size();
            for (size_t i = 0; i < probs.size(); ++i) {
                acc += probs[i];
                if (acc >= static_cast<double>(params_.top_p)) {
                    cutoff = i + 1;
                    break;
                }
            }
            // The probe held less mass than top_p asked for; only then is the
            // full sort actually needed.
            if (acc < static_cast<double>(params_.top_p) && probing && !min_p_closed) short_of_mass = true;
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
