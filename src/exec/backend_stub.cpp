#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "exec/backend.h"
#include "util/log.h"
#include "util/utf8.h"

namespace lgc {
namespace {

// The stub's fixed identity (its plausible trained context is backend.h's
// kStubNCtxTrain).
constexpr const char* kStubId = "stub";

// ---------------------------------------------------------------- tokenizer
//
// A reversible, deterministic splitter — emphatically not a BPE. Token counts
// will differ from the real tokenizer (the llama engine uses the GGUF's own,
// DESIGN.md §3.7), which is why the stub claims no real context numbers.
class StubTokenizer final : public Tokenizer {
public:
    StubTokenizer() {
        // Id 0 is EOS and decodes to nothing.
        pieces_.emplace_back("");
        ids_.emplace("", 0);
    }

    std::vector<int> encode(std::string_view text) override {
        std::vector<int> out;
        size_t           i = 0;
        while (i < text.size()) {
            const size_t start = i;
            if (is_word_byte(text[i])) {
                while (i < text.size() && is_word_byte(text[i])) ++i;
            } else {
                ++i;
            }
            out.push_back(intern(text.substr(start, i - start)));
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) override {
        std::string out;
        for (int id : ids) out += decode_one(id);
        return out;
    }

    std::string decode_one(int id) override {
        std::lock_guard<std::mutex> guard(mutex_);
        if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) return {};
        return pieces_[static_cast<size_t>(id)];
    }

    int eos_id() const override { return 0; }

    int vocab_size() const override { return static_cast<int>(pieces_.size()); }

private:
    // Letters, digits, underscore and every non-ASCII byte glue together; each
    // remaining ASCII byte stands alone. Concatenating the pieces reproduces the
    // input byte for byte, which is the only property the rest of the engine
    // relies on at M0.
    static bool is_word_byte(char c) {
        const unsigned char b = static_cast<unsigned char>(c);
        if (b >= 0x80) return true;
        return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') ||
               b == '_';
    }

    int intern(std::string_view piece) {
        std::lock_guard<std::mutex> guard(mutex_);
        const std::string           key(piece);
        auto                        it = ids_.find(key);
        if (it != ids_.end()) return it->second;

        const int id = static_cast<int>(pieces_.size());
        pieces_.push_back(key);
        ids_.emplace(key, id);
        return id;
    }

    std::mutex                             mutex_;
    std::vector<std::string>               pieces_;
    std::unordered_map<std::string, int>   ids_;
};

// ------------------------------------------------------------------ backend
class StubBackend final : public Backend {
public:
    StubBackend(int n_ctx, int delay_ms, const std::string& served_name,
                const std::vector<int>& lane_ctx, const SamplerDefaults& sampler,
                int prefill_ubatches)
        : delay_ms_(delay_ms), prefill_ubatches_(std::max(1, prefill_ubatches)), lane_ctx_(lane_ctx) {
        status_.id               = kStubId;
        status_.served_id        = served_name.empty() ? std::string(kStubId) : served_name;
        status_.n_ctx_train      = kStubNCtxTrain;
        status_.loaded           = true;
        status_.stub             = true;
        status_.n_ctx            = n_ctx;
        status_.mtp_enabled      = false;
        status_.weights_bytes    = 0;
        status_.sampler_defaults = sampler;
    }

    const ModelStatus& status() const override { return status_; }
    Tokenizer&         tokenizer() override { return tokenizer_; }

    std::string render_chat(const ChatRequest& req) const override {
        return render_chatml_stub(req);
    }

    // The lane index is real here too: the stub has no per-lane state to index
    // with it, but it does have to prove the plumbing — the concurrency gates
    // (two lanes, cancellation of one, admission of a third) run against this
    // backend in CI, where there is no card.
    FinishReason generate(const GenerationInput& in, int slot, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        using clock = std::chrono::steady_clock;
        // a named lane stops at its own cap (--lane-ctx)
        const int n_ctx = lane_ctx_.empty()
                              ? status_.n_ctx
                              : lane_ctx_[static_cast<size_t>(std::clamp(slot, 0, static_cast<int>(lane_ctx_.size()) - 1))];

        const auto prefill_start = clock::now();
        const auto prompt_ids    = in.prompt_ids.empty() ? tokenizer_.encode(in.prompt) : in.prompt_ids;
        stats.prompt_tokens      = static_cast<int>(prompt_ids.size());
        // The prefill in its ubatches, as the real backend's loop: one cancel
        // check a boundary, so a gone client or a stop signal is seen there
        // and not only when the first token fails to write (DESIGN §3.7).
        for (int u = 0; u < prefill_ubatches_; ++u) {
            if (abort_requested(in)) {
                // the prefill stopped after `u` of its ubatches: report what it
                // prefilled, not the whole prompt, and which source ended it
                stats.prompt_tokens_prefilled =
                    static_cast<int>(stats.prompt_tokens) * u / prefill_ubatches_;
                stats.abort_cause   = cause(in);
                stats.prefill_seconds =
                    std::chrono::duration<double>(clock::now() - prefill_start).count();
                return FinishReason::Abort;
            }
        }
        stats.prefill_seconds =
            std::chrono::duration<double>(clock::now() - prefill_start).count();

        const std::string reply = synthesise(in, stats.prompt_tokens);
        const auto        ids   = tokenizer_.encode(reply);

        const auto decode_start = clock::now();
        FinishReason reason     = FinishReason::Stop;

        for (int id : ids) {
            // A stop signal or a gone client ends the request at the token
            // boundary, and the slot is free at once. Checked every token: the
            // stub has no step cost worth throttling.
            if (abort_requested(in)) {
                stats.abort_cause = cause(in);
                reason = FinishReason::Abort;
                break;
            }
            if (in.sampler.max_tokens >= 0 &&
                stats.completion_tokens >= in.sampler.max_tokens) {
                reason = FinishReason::Length;
                break;
            }
            if (n_ctx > 0 &&
                stats.prompt_tokens + stats.completion_tokens >= n_ctx) {
                reason = FinishReason::Length;
                break;
            }

            if (delay_ms_ > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
            }

            const std::string piece = tokenizer_.decode_one(id);
            ++stats.completion_tokens;

            // A byte-level detokenizer routinely hands out half of a multi-byte
            // code point. Splitting multi-byte pieces here keeps the streaming
            // hold-back path (§3.7) on the tested path rather than the hoped-for
            // one.
            const Control c = utf8::count_codepoints(piece) < piece.size()
                                  ? emit_split(piece, id, on_piece)
                                  : on_piece(piece, id);
            if (c == Control::Stop) {
                reason = FinishReason::Stop;
                break;
            }
            if (c == Control::Cancel) {
                stats.abort_cause = AbortCause::ClientGone;   // the write failed
                reason = FinishReason::Abort;
                break;
            }
        }

        stats.decode_seconds = std::chrono::duration<double>(clock::now() - decode_start).count();
        return reason;
    }

    // The stub loads no model and no projector: the route's own refusal has
    // already fired for a build without --tts (prepare_speech), so this is the
    // belt-and-braces answer for a caller that reached the seam directly.
    bool generate_speech(const SpeechInput&, int, std::string&, std::string& error,
                         GenerationStats&) override {
        error = "this backend has no TTS: the stub loads no model";
        return false;
    }

private:
    // The two cancellation sources, one check: the process's stop flag, then
    // the client's connection (the probe peeks the socket and writes nothing).
    static bool abort_requested(const GenerationInput& in) {
        if (in.stop != nullptr && in.stop->load(std::memory_order_relaxed)) return true;
        // The route ORs the cancel flag into peer_alive; reading it here too
        // makes the stub's own boundary the same check as the real backend's.
        if (in.cancel && in.cancel->load(std::memory_order_relaxed)) return true;
        return in.peer_alive && !in.peer_alive();
    }

    // Which source ended it: the stop flag first, then the cancel flag, then
    // the closed peer (the campaign's naming order).
    static AbortCause cause(const GenerationInput& in) {
        if (in.stop != nullptr && in.stop->load(std::memory_order_relaxed))
            return AbortCause::StopSignal;
        if (in.cancel && in.cancel->load(std::memory_order_relaxed)) return AbortCause::Cancelled;
        return AbortCause::ClientGone;
    }

    static Control emit_split(const std::string& piece, int id, const TokenCallback& on_piece) {
        const size_t cut = piece.size() / 2;
        const Control first = on_piece(std::string_view(piece).substr(0, cut), id);
        if (first != Control::Continue) return first;
        return on_piece(std::string_view(piece).substr(cut), id);
    }

    std::string synthesise(const GenerationInput& in, int prompt_tokens) const {
        std::string out = log::format(
            "arcint stub backend \xc2\xb7 no model loaded \xc2\xb7 %s (%s)\n"
            "prompt %d tokens, sampler %s. Gr\xc3\xbc\xc3\x9f""e aus der Werkstatt.",
            status_.served_id.c_str(), kStubId, prompt_tokens,
            in.sampler.greedy() ? "greedy"
                                : log::format("temp %.2f", in.sampler.temperature).c_str());

        // A four-byte code point, deliberately surrounded by ASCII so the
        // tokenizer hands it out as its own piece and emit_split() tears it in
        // half. Without one, nothing in the HTTP path ever exercises the
        // streaming hold-back against a full-width sequence — the rest of this
        // reply is two-byte Latin-1 supplement at most.
        out += " \xf0\x9f\xa7\xa9";

        if (!in.tool_names.empty()) {
            out += "\n<tool_call>\n{\"name\": \"";
            out += in.tool_names.front();
            out += "\", \"arguments\": {}}\n</tool_call>";
        }
        return out;
    }

    int           delay_ms_ = 0;
    int           prefill_ubatches_ = 1;
    std::vector<int> lane_ctx_;   // --lane-ctx, by slot
    ModelStatus   status_;
    StubTokenizer tokenizer_;
};

}  // namespace

const char* finish_reason_name(FinishReason r) {
    switch (r) {
        case FinishReason::Stop:   return "stop";
        case FinishReason::Length: return "length";
        case FinishReason::Abort:  return "abort";
        case FinishReason::Loop:   return "loop";
    }
    return "stop";
}

std::unique_ptr<Backend> make_stub_backend(int n_ctx, int delay_ms, const std::string& served_name,
                                           const std::vector<int>& lane_ctx,
                                           const SamplerDefaults& sampler,
                                           int prefill_ubatches) {
    return std::make_unique<StubBackend>(n_ctx, delay_ms, served_name, lane_ctx, sampler,
                                         prefill_ubatches);
}

}  // namespace lgc
