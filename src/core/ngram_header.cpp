#include "core/ngram_header.h"

#include <cstring>

#include "util/log.h"

namespace lgc::ngram {

std::string parse_header(const uint8_t* bytes, std::size_t n_bytes, Header& out) {
    if (n_bytes < kHeaderBytes) {
        return log::format("header truncated: %zu bytes, expected at least %zu",
                           n_bytes, kHeaderBytes);
    }
    if (std::memcmp(bytes, kMagic.data(), kMagicBytes) != 0) {
        return log::format("bad magic: expected \"%s\"", std::string(kMagic).c_str());
    }
    uint32_t fields[4];
    std::memcpy(fields, bytes + kMagicBytes, sizeof(fields));
    const uint32_t ggml_type = fields[0];
    const uint32_t n_cols    = fields[1];
    const uint32_t n_rows    = fields[2];
    const uint32_t reserved  = fields[3];
    if (bytes_per_block(ggml_type) == 0) {
        return log::format(
            "unknown ggml_type %u: pick one of 2 (Q4_0), 3 (Q4_1), 8 (Q8_0)",
            ggml_type);
    }
    if (n_cols == 0 || n_cols % kBlockElements != 0) {
        return log::format(
            "n_cols %u is not a positive multiple of %u", n_cols, kBlockElements);
    }
    if (n_rows == 0) return "n_rows must be positive";
    if (reserved != 0) {
        return log::format("reserved field must be zero, got %u", reserved);
    }
    out.ggml_type = ggml_type;
    out.n_cols    = n_cols;
    out.n_rows    = n_rows;
    return {};
}

std::size_t bytes_per_block(uint32_t ggml_type) {
    switch (ggml_type) {
        case 2:  return 18;   // Q4_0
        case 3:  return 20;   // Q4_1
        case 8:  return 34;   // Q8_0
        default: return 0;
    }
}

uint64_t payload_bytes(const Header& h) {
    const uint64_t blocks_per_row = h.n_cols / kBlockElements;
    const uint64_t bpb            = bytes_per_block(h.ggml_type);
    return static_cast<uint64_t>(h.n_rows) * blocks_per_row * bpb;
}

uint64_t file_bytes(const Header& h) { return kHeaderBytes + payload_bytes(h); }

}  // namespace lgc::ngram
