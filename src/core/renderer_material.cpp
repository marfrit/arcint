#include "core/renderer_material.h"

namespace lgc {

nlohmann::json assemble_renderer_material(const RendererMaterialParts&) {
    return nlohmann::json{{"skeleton", true}};
}

}  // namespace lgc
