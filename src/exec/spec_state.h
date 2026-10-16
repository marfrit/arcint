// The drafter state a context checkpoint carries (--llama-checkpoints with
// --llama-mtp or --llama-dflash-gguf). llama.cpp's server stashes the draft
// side with each prompt checkpoint: the draft context's partial state and the
// speculative implementation's own (tools/server/server-context.cpp:2596-2599
// at the pin). Here it is opaque bytes behind a small header naming the
// drafter kind and the token count the state belongs to, so a restore refuses
// a state from another drafter or another position instead of loading it.
// Host memory only, never written to disk: native byte order. Pure, so the
// round trip is testable without a card (tests/test_spec_state.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace lgc {

enum class SpecStateKind : uint32_t {
    MtpRow       = 1,   // the MTP drafter's carried hidden row (floats)
    DflashWindow = 2,   // the DFlash2 drafter's sliding-window K/V (llama_state_seq_get_data_ext, PARTIAL_ONLY)
};

constexpr uint32_t kSpecStateMagic  = 0x53505341;   // "ASPS"
constexpr size_t   kSpecStateHeader = 4 + 4 + 8 + 8;   // magic, kind, n_tokens, payload size

// A state of `size` payload bytes for `kind` at n_tokens, the payload zeroed
// at data() + kSpecStateHeader for the caller to fill
inline std::vector<uint8_t> spec_state_alloc(SpecStateKind kind, uint64_t n_tokens, size_t size) {
    std::vector<uint8_t> s(kSpecStateHeader + size, 0);
    const uint32_t       magic = kSpecStateMagic, k = static_cast<uint32_t>(kind);
    const uint64_t       sz    = size;
    std::memcpy(s.data(), &magic, 4);
    std::memcpy(s.data() + 4, &k, 4);
    std::memcpy(s.data() + 8, &n_tokens, 8);
    std::memcpy(s.data() + 16, &sz, 8);
    return s;
}

inline std::vector<uint8_t> spec_state_pack(SpecStateKind kind, uint64_t n_tokens, const void* data, size_t size) {
    std::vector<uint8_t> s = spec_state_alloc(kind, n_tokens, size);
    if (size > 0) std::memcpy(s.data() + kSpecStateHeader, data, size);
    return s;
}

// The payload of `s` when it is a whole state of `kind` at n_tokens; false
// (and an empty view) for an empty state, another kind, another token count,
// or a truncated or padded buffer
struct SpecStateView {
    const uint8_t* data = nullptr;
    size_t         size = 0;
};

inline bool spec_state_view(const std::vector<uint8_t>& s, SpecStateKind kind, uint64_t n_tokens, SpecStateView& out) {
    out = {};
    if (s.size() < kSpecStateHeader) return false;
    uint32_t magic = 0, k = 0;
    uint64_t n = 0, sz = 0;
    std::memcpy(&magic, s.data(), 4);
    std::memcpy(&k, s.data() + 4, 4);
    std::memcpy(&n, s.data() + 8, 8);
    std::memcpy(&sz, s.data() + 16, 8);
    if (magic != kSpecStateMagic || k != static_cast<uint32_t>(kind) || n != n_tokens || sz != s.size() - kSpecStateHeader)
        return false;
    out.data = s.data() + kSpecStateHeader;
    out.size = static_cast<size_t>(sz);
    return true;
}

}  // namespace lgc
