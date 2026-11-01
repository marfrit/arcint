#include "api/tokenize.h"

#include <cstdint>
#include <string>
#include <vector>

#include "api/error.h"
#include "api/responses.h"
#include "util/utf8.h"

// ops item 3: the token surface. `/tokenize` has three body shapes -- the
// llama-server content shape, a chat body and a Responses body. The last two
// go through the request's own prepare path (`prepare_chat` /
// `prepare_response`, hence `finish_prepare`), and the answer is built from the
// id vector finish_prepare stored in `GenerationInput::prompt_ids_text`, never
// a second encode: the count is what the generation would decode, by
// construction. The model field resolves a lane exactly as the generation
// routes do.
namespace lgc::api {
namespace {

using json = nlohmann::json;

// A piece as llama-server serializes it: a JSON string when it is valid UTF-8
// on its own, otherwise the array of its byte values. A byte-level
// detokenizer hands out half a multi-byte character routinely, and a JSON
// string cannot carry those bytes (nlohmann's dump would throw or replace
// them), so the bytes are the honest representation.
json piece_json(const std::string& piece) {
    if (utf8::is_valid(piece)) return piece;
    json bytes = json::array();
    for (unsigned char c : piece) bytes.push_back(static_cast<int>(c));
    return bytes;
}

json tokens_json(Tokenizer& tok, const std::vector<int>& ids, bool with_pieces) {
    if (!with_pieces) return ids;   // a std::vector<int> serializes as an array
    json out = json::array();
    for (int id : ids) out.push_back(json{{"id", id}, {"piece", piece_json(tok.decode_one(id))}});
    return out;
}

std::string model_of(const json& body) {
    auto it = body.find("model");
    return it != body.end() && it->is_string() ? it->get<std::string>() : std::string();
}

// A boolean field: absent -> `dflt`; present but not a boolean -> `ok = false`
// (the caller answers 400 rather than let a thrown type_error become a 500).
bool bool_field(const json& body, const char* key, bool dflt, bool& ok) {
    auto it = body.find(key);
    if (it == body.end()) return dflt;
    if (!it->is_boolean()) {
        ok = false;
        return dflt;
    }
    return it->get<bool>();
}

HttpResult bad_field(std::string_view message, std::string_view param = {}) {
    return HttpResult{400, invalid_request(message, param)};
}

// An mmproj unit counts the image tokens with the projector's own tokenizer,
// which /tokenize does not run: the text tokenisation would be short by every
// image token, so an image part is refused instead of reported wrong.
std::optional<HttpResult> refuse_images(const ChatRequest& req) {
    for (const ChatMessage& m : req.messages) {
        if (m.images.empty()) continue;
        return HttpResult{400, invalid_request(
            "image parts cannot be counted by /tokenize: an mmproj unit's image tokens are known "
            "only when the projector runs, so use the generation's usage.prompt_tokens",
            "messages")};
    }
    return std::nullopt;
}

// The count result: the id vector finish_prepare stored, with the llama
// backend's BOS rule applied. That backend prepends BOS for a vocabulary that
// wants it when the rendered prompt did not already write it (prompt_tokens),
// so the count has to as well; otherwise /tokenize reports one token fewer than
// the generation's usage. The count is the final vector's size.
HttpResult count_result(const Context& ctx, const GenerationInput& in, bool with_pieces) {
    std::vector<int> ids = in.prompt_ids_text;
    Tokenizer&       tok = ctx.backend->tokenizer();
    const int        bos = tok.bos_id();
    const bool       add_bos = ctx.backend->template_caps().value("add_bos", false);
    if (add_bos && bos >= 0 && in.prompt_ids.empty() && (ids.empty() || ids.front() != bos))
        ids.insert(ids.begin(), bos);
    return HttpResult{200, json{{"tokens", tokens_json(tok, ids, with_pieces)},
                                {"count", static_cast<int>(ids.size())}}};
}

}  // namespace

HttpResult tokenize_request(const Context& ctx, const json& body) {
    if (!body.is_object()) return bad_field("the body must be a JSON object");
    bool       ok          = true;
    const bool with_pieces = bool_field(body, "with_pieces", false, ok);
    if (!ok) return bad_field("with_pieces must be a boolean", "with_pieces");

    // 1. llama-server's content shape.
    if (body.contains("content")) {
        int lane = -1;
        if (auto err = resolve_lane(ctx, model_of(body), lane)) return *err;
        const json& c = body["content"];
        if (!c.is_string()) return bad_field("content must be a string", "content");
        bool       special_ok  = true;
        const bool add_special = bool_field(body, "add_special", false, special_ok);
        if (!special_ok) return bad_field("add_special must be a boolean", "add_special");
        Tokenizer&             tok = ctx.backend->tokenizer();
        const std::vector<int> ids = tok.encode(c.get<std::string>(), add_special);
        return HttpResult{200, json{{"tokens", tokens_json(tok, ids, with_pieces)}}};
    }

    // 2. a chat body: the chat route's own prepare, so the count is the prompt
    //    that would be decoded.
    if (body.contains("messages")) {
        // Parse and refuse an image before the prepare renders the prompt: an
        // image's tokens cannot be counted here, and the render is the work, so
        // the cheaper refusal comes first. The 400 keeps prepare_chat's shape.
        ChatRequest req;
        if (auto err = parse_chat_request(body, req))
            return HttpResult{400, invalid_request(*err)};
        if (auto err = refuse_images(req)) return *err;
        PreparedChat prep;
        if (auto err = prepare_from_chat(ctx, std::move(req), prep)) return *err;
        return count_result(ctx, prep.input, with_pieces);
    }

    // 3. a Responses body: the same prepare the Responses route calls, but a
    //    count is not a turn -- a 4xx here must not evict the referenced
    //    stored entry (the count path is how a client asks whether the next
    //    turn fits).
    if (body.contains("input")) {
        ResponseRequest req;
        if (auto err = parse_response_request(body, req)) return *err;
        if (auto err = refuse_images(req.chat)) return *err;   // the new input
        PreparedResponse prep;
        if (auto err = prepare_response(ctx, std::move(req), prep, /*evict_on_error=*/false)) return *err;
        // A previous_response_id continuation merges the stored prefix inside
        // prepare, and a stored message keeps its image parts: refuse again on
        // the request that was actually prepared, or the count would be short
        // by every image token the projector would have added.
        if (auto err = refuse_images(prep.chat.req)) return *err;
        return count_result(ctx, prep.chat.input, with_pieces);
    }

    return bad_field("the body needs `content`, `messages` or `input`");
}

HttpResult detokenize_request(const Context& ctx, const json& body) {
    if (!body.is_object()) return bad_field("the body must be a JSON object");
    int lane = -1;
    if (auto err = resolve_lane(ctx, model_of(body), lane)) return *err;
    if (!body.contains("tokens") || !body["tokens"].is_array())
        return bad_field("tokens must be an array of integers", "tokens");
    // Per element, never get<std::vector<int>>(): that converts a float (1.5 ->
    // 1) and wraps an integer past int32 (2^32 + 5 -> 5), which then passes the
    // vocabulary check. is_number_integer() rejects a bool; the int64/uint64
    // range is checked before the vocabulary range. An id past the vocabulary
    // would read out of range in decode_one (the llama backend throws, which
    // the exception handler would turn into a 500 with an "unhandled exception"
    // line). A tokenizer that cannot enumerate (vocab_size 0) offers nothing to
    // validate against and keeps the old path.
    const int        vocab = ctx.backend->tokenizer().vocab_size();
    std::vector<int> ids;
    ids.reserve(body["tokens"].size());
    for (const json& t : body["tokens"]) {
        if (!t.is_number_integer())
            return bad_field("tokens must be an array of integers", "tokens");
        int64_t v = 0;
        if (t.is_number_unsigned()) {
            const uint64_t u = t.get<uint64_t>();
            if (u > static_cast<uint64_t>(INT32_MAX))
                return bad_field("tokens must be ids this vocabulary serves", "tokens");
            v = static_cast<int64_t>(u);
        } else {
            v = t.get<int64_t>();
        }
        if (v < 0 || v > static_cast<int64_t>(INT32_MAX) ||
            (vocab > 0 && v >= static_cast<int64_t>(vocab)))
            return bad_field("tokens must be ids this vocabulary serves", "tokens");
        ids.push_back(static_cast<int>(v));
    }
    // Tokenizer::decode concatenates decode_one, which renders a special token
    // as its text (llama-server's default unparse_special=true).
    return HttpResult{200, json{{"content", ctx.backend->tokenizer().decode(ids)}}};
}

}  // namespace lgc::api
