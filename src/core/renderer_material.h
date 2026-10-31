#pragma once

#include <nlohmann/json.hpp>

// feyntalk W6a (docs/campaigns/feyntalk.md, D4): the renderer material the
// gateway consumes. The backend reads its artifact once at load and fills the
// parts; this assembler adds the schema version and the canonical hash over the
// five hashable parts. `model` is volatile (names) and is excluded from the
// hash. Nothing here is in the generation path.
namespace lgc {

struct RendererMaterialParts {
    nlohmann::json model;           // served_id, canonical_id, quant, arch
    nlohmann::json decode;          // pieces_b64[id] == decode_one(id), eos, eog_ids
    nlohmann::json encode;          // raw tokens, merges, token_types, pre, add_bos, special_text
    nlohmann::json chat_template;   // text, source
    nlohmann::json special_tokens;  // think ids, tool-call ids and texts
    nlohmann::json parser;          // parser hints
};

// {schema, hash, model, decode, encode, chat_template, special_tokens, parser}
nlohmann::json assemble_renderer_material(const RendererMaterialParts& parts);

}  // namespace lgc
