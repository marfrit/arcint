#include "core/renderer_material.h"

#include "util/canonical_json.h"

namespace lgc {

nlohmann::json assemble_renderer_material(const RendererMaterialParts& parts) {
    // The hash covers the five hashable parts; `model` is volatile (a served
    // name changes without changing what the renderer must agree on).
    const nlohmann::json core = {{"decode", parts.decode},
                                 {"encode", parts.encode},
                                 {"chat_template", parts.chat_template},
                                 {"special_tokens", parts.special_tokens},
                                 {"parser", parts.parser}};
    return nlohmann::json{{"schema", "feyntalk-render/1"},
                          {"hash", canonical_sha256(core)},
                          {"model", parts.model},
                          {"decode", parts.decode},
                          {"encode", parts.encode},
                          {"chat_template", parts.chat_template},
                          {"special_tokens", parts.special_tokens},
                          {"parser", parts.parser}};
}

}  // namespace lgc
