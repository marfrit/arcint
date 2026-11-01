#pragma once

#include <nlohmann/json.hpp>

#include "api/handlers.h"

// ops item 3 (arcint 1.4.0): POST /tokenize and POST /detokenize, the
// llama-server-compatible token surface, plus the arcint extension that counts
// a chat or Responses body through the request's own prepare path. The
// endpoints are behind the same key as every other route (the server's
// pre-routing handler); the model field picks the lane exactly as the
// generation routes do.
namespace lgc::api {

// POST /tokenize. Three body shapes:
//   - `{content, add_special?, with_pieces?, model?}` (llama-server's shape):
//     `{tokens: [ids]}` or, with `with_pieces`, `{tokens: [{id, piece}]}`;
//   - a chat body (`messages`, `tools`, …): `{tokens, count}` through
//     prepare_chat and the tokens finish_prepare stored in
//     `GenerationInput::prompt_ids_text`, so the count is what a real request
//     would decode, by construction;
//   - a Responses body (`input`, `instructions`, `tools`, …): the same through
//     prepare_response.
HttpResult tokenize_request(const Context& ctx, const nlohmann::json& body);

// POST /detokenize `{tokens, model?}` -> `{content}`: the concatenation of the
// tokens' pieces, with special tokens rendered as their text (llama-server's
// default `unparse_special=true`), which is `Tokenizer::decode`.
HttpResult detokenize_request(const Context& ctx, const nlohmann::json& body);

}  // namespace lgc::api
