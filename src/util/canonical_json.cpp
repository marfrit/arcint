#include "util/canonical_json.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "util/sha256.h"

namespace lgc {
namespace {

// The canonical form: objects sorted by key at every level, arrays in order,
// UTF-8 raw, `,`/`:` separators. A float is refused rather than rounded.
void write_canonical(const nlohmann::json& v, std::string& out) {
    switch (v.type()) {
        case nlohmann::json::value_t::object: {
            std::vector<std::string> keys;
            keys.reserve(v.size());
            for (const auto& item : v.items()) keys.push_back(item.key());
            std::sort(keys.begin(), keys.end());
            out.push_back('{');
            for (size_t i = 0; i < keys.size(); ++i) {
                if (i != 0) out.push_back(',');
                out += nlohmann::json(keys[i]).dump();
                out.push_back(':');
                write_canonical(v.at(keys[i]), out);
            }
            out.push_back('}');
            break;
        }
        case nlohmann::json::value_t::array:
            out.push_back('[');
            for (size_t i = 0; i < v.size(); ++i) {
                if (i != 0) out.push_back(',');
                write_canonical(v[i], out);
            }
            out.push_back(']');
            break;
        case nlohmann::json::value_t::number_float:
            throw std::runtime_error("canonical json: a floating-point number is not reproducible");
        case nlohmann::json::value_t::null:
        case nlohmann::json::value_t::boolean:
        case nlohmann::json::value_t::string:
        case nlohmann::json::value_t::number_integer:
        case nlohmann::json::value_t::number_unsigned:
            out += v.dump();
            break;
        default:
            throw std::runtime_error("canonical json: a value type the material does not carry");
    }
}

}  // namespace

std::string canonical_dump(const nlohmann::json& v) {
    std::string out;
    write_canonical(v, out);
    return out;
}

std::string canonical_sha256(const nlohmann::json& v) {
    return "sha256:" + sha256_hex(canonical_dump(v));
}

}  // namespace lgc
