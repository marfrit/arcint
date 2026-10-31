#include "util/canonical_json.h"

namespace lgc {

std::string canonical_dump(const nlohmann::json&) { return {}; }

std::string canonical_sha256(const nlohmann::json&) { return "sha256:" + std::string(64, '0'); }

}  // namespace lgc
