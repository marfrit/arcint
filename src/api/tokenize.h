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
//     `{tokens: [ids]}` or, with `with_pieces`, `{tokens: [{id, piece}]}`.
//     `content` must be a string: llama-server also takes a mixed array of
//     strings and token ids (tokenize_mixed); arcint serves only the string
//     form and answers 400 on an array. llama-server's `parse_special` is
//     accepted and ignored: this tokenizer always parses specials;
//   - a chat body (`messages`, `tools`, …): `{tokens, count}` through
//     prepare_chat and the tokens finish_prepare stored in
//     `GenerationInput::prompt_ids_text`, so the count is what a real request
//     would decode — the llama backend's BOS rule included. An image part is
//     refused: an mmproj unit counts its image tokens with the projector,
//     which this route does not run;
//   - a Responses body (`input`, `instructions`, `tools`, …): the same through
//     prepare_response, but the referenced stored entry is not evicted (a count
//     is not a turn). It is still a lookup: ResponseStore::get refreshes the
//     entry's last_used and bumps its LRU rank like any other read.
//
// A prompt the lane would refuse is refused here too, in the same error body
// (the §3.8 overflow carries `prompt_tokens`), because the count goes through
// the request's own prepare.
HttpResult tokenize_request(const Context& ctx, const nlohmann::json& body);

// POST /detokenize `{tokens, model?}` -> `{content}`: the concatenation of the
// tokens' pieces, with special tokens rendered as their text (llama-server's
// default `unparse_special=true`), which is `Tokenizer::decode`. An id outside
// `[0, vocab_size)` is a 400 naming `tokens`.
HttpResult detokenize_request(const Context& ctx, const nlohmann::json& body);

}  // namespace lgc::api
