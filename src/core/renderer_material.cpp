#include "core/renderer_material.h"

#include "util/canonical_json.h"
#include "util/utf8.h"

namespace lgc {

nlohmann::json assemble_renderer_material(const RendererMaterialParts& parts,
                                           std::string* why) {
    try {
        // The hash covers the five hashable parts; `model` is volatile (a
        // served name changes without changing what the renderer must agree on).
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
    } catch (const std::exception& e) {
        // Non-UTF-8 string text (a raw-byte vocabulary), a float, a value the
        // canonical form refuses: no material this load, one reason for the
        // caller's warn line.
        if (why != nullptr) *why = e.what();
        return nlohmann::json();
    } catch (...) {
        if (why != nullptr) *why = "unknown error";
        return nlohmann::json();
    }
}

bool renderer_text_utf8_ok(int count, const std::function<std::string_view(int)>& text_of,
                           std::string_view template_text) {
    for (int i = 0; i < count; ++i) {
        const std::string_view text = text_of(i);
        if (!text.empty() && !utf8::is_valid(text)) return false;
    }
    return utf8::is_valid(template_text);
}

}  // namespace lgc
