#include "api/tokenize.h"

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

// The count result: the prompt tokens finish_prepare stored, or -- never on the
// chat and Responses paths -- the one tokenisation of the input. The count is
// the stored vector's size, so it equals what the backend would decode.
HttpResult count_result(const Context& ctx, const GenerationInput& in, int count, bool with_pieces) {
    std::vector<int> ids = in.prompt_ids_text;
    if (ids.empty()) ids = prompt_ids_raw(in, ctx.backend->tokenizer());
    Tokenizer& tok = ctx.backend->tokenizer();
    return HttpResult{200, json{{"tokens", tokens_json(tok, ids, with_pieces)}, {"count", count}}};
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
        PreparedChat prep;
        if (auto err = prepare_chat(ctx, body, prep)) return *err;
        return count_result(ctx, prep.input, prep.prompt_tokens, with_pieces);
    }

    // 3. a Responses body: the same prepare the Responses route calls.
    if (body.contains("input")) {
        ResponseRequest req;
        if (auto err = parse_response_request(body, req)) return *err;
        PreparedResponse prep;
        if (auto err = prepare_response(ctx, std::move(req), prep)) return *err;
        return count_result(ctx, prep.chat.input, prep.chat.prompt_tokens, with_pieces);
    }

    return bad_field("the body needs `content`, `messages` or `input`");
}

HttpResult detokenize_request(const Context& ctx, const json& body) {
    if (!body.is_object()) return bad_field("the body must be a JSON object");
    int lane = -1;
    if (auto err = resolve_lane(ctx, model_of(body), lane)) return *err;
    if (!body.contains("tokens")) return bad_field("the body needs `tokens`", "tokens");
    std::vector<int> ids;
    try {
        ids = body["tokens"].get<std::vector<int>>();
    } catch (const json::exception&) {
        return bad_field("tokens must be an array of integers", "tokens");
    }
    // Tokenizer::decode concatenates decode_one, which renders a special token
    // as its text (llama-server's default unparse_special=true).
    return HttpResult{200, json{{"content", ctx.backend->tokenizer().decode(ids)}}};
}

}  // namespace lgc::api
