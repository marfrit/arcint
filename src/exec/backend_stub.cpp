#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "exec/backend.h"
#include "util/log.h"
#include "util/utf8.h"

namespace lgc {
namespace {

// ---------------------------------------------------------------- tokenizer
//
// A reversible, deterministic splitter — emphatically not a BPE. Token counts
// will differ from the real tokenizer, which is exactly why the allowlist ships
// no context numbers for the stub to pretend against. M1 replaces this with
// openvino_tokenizers out of the artifact (DESIGN.md §3.7).
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
    StubBackend(const ModelEntry& entry, Quant quant, int n_ctx, int delay_ms)
        : delay_ms_(delay_ms) {
        status_.id               = entry.id;
        status_.quant            = quant;
        status_.loaded           = true;
        status_.stub             = true;
        status_.n_ctx            = n_ctx;
        status_.n_layer          = entry.n_layer;
        status_.n_gdn_layer      = entry.n_gdn_layer;
        status_.n_attn_layer     = entry.n_attn_layer;
        status_.mtp_enabled      = false;
        status_.weights_bytes    = 0;
        status_.sampler_defaults = entry.sampler;
    }

    const ModelStatus& status() const override { return status_; }
    Tokenizer&         tokenizer() override { return tokenizer_; }

    FinishReason generate(const GenerationInput& in, const TokenCallback& on_piece,
                          GenerationStats& stats) override {
        using clock = std::chrono::steady_clock;

        const auto prefill_start = clock::now();
        const auto prompt_ids    = tokenizer_.encode(in.prompt);
        stats.prompt_tokens      = static_cast<int>(prompt_ids.size());
        stats.prefill_seconds =
            std::chrono::duration<double>(clock::now() - prefill_start).count();

        const std::string reply = synthesise(in, stats.prompt_tokens);
        const auto        ids   = tokenizer_.encode(reply);

        const auto decode_start = clock::now();
        FinishReason reason     = FinishReason::Stop;

        for (int id : ids) {
            if (in.sampler.max_tokens >= 0 &&
                stats.completion_tokens >= in.sampler.max_tokens) {
                reason = FinishReason::Length;
                break;
            }
            if (status_.n_ctx > 0 &&
                stats.prompt_tokens + stats.completion_tokens >= status_.n_ctx) {
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
                reason = FinishReason::Abort;
                break;
            }
        }

        stats.decode_seconds = std::chrono::duration<double>(clock::now() - decode_start).count();
        return reason;
    }

private:
    static Control emit_split(const std::string& piece, int id, const TokenCallback& on_piece) {
        const size_t cut = piece.size() / 2;
        const Control first = on_piece(std::string_view(piece).substr(0, cut), id);
        if (first != Control::Continue) return first;
        return on_piece(std::string_view(piece).substr(cut), id);
    }

    std::string synthesise(const GenerationInput& in, int prompt_tokens) const {
        std::string out = log::format(
            "ligence stub backend \xc2\xb7 no model loaded \xc2\xb7 %s %s\n"
            "prompt %d tokens, sampler %s. Gr\xc3\xbc\xc3\x9f""e aus der Werkstatt.",
            status_.id.c_str(), quant_name(status_.quant), prompt_tokens,
            in.sampler.greedy() ? "greedy"
                                : log::format("temp %.2f", in.sampler.temperature).c_str());

        if (!in.tool_names.empty()) {
            out += "\n<tool_call>\n{\"name\": \"";
            out += in.tool_names.front();
            out += "\", \"arguments\": {}}\n</tool_call>";
        }
        return out;
    }

    int           delay_ms_ = 0;
    ModelStatus   status_;
    StubTokenizer tokenizer_;
};

}  // namespace

const char* finish_reason_name(FinishReason r) {
    switch (r) {
        case FinishReason::Stop:   return "stop";
        case FinishReason::Length: return "length";
        case FinishReason::Abort:  return "abort";
    }
    return "stop";
}

std::unique_ptr<Backend> make_stub_backend(const ModelEntry& entry, Quant quant, int n_ctx,
                                           int delay_ms) {
    return std::make_unique<StubBackend>(entry, quant, n_ctx, delay_ms);
}

}  // namespace lgc
