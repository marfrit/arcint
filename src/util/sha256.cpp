#include "util/sha256.h"

#include <cstdint>
#include <string>

// sha256, lowercase hex. The campaign names sha256 for prefix_hash; this is a
// self-contained implementation so no new dependency enters the public tree.
// W6a moved it here from src/api/responses_state.cpp: the renderer material's
// canonical hash needs it in a module the executor can include.
namespace lgc {
namespace {

// sha256, lowercase hex. The campaign names sha256 for prefix_hash; this is a
// self-contained implementation so no new dependency enters the public tree.
constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

std::string sha256_impl(std::string_view bytes) {
    // Message schedule and the compression function over 64-byte blocks.
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    std::string m(bytes);
    const uint64_t bit_len = static_cast<uint64_t>(m.size()) * 8;
    m.push_back('\x80');
    while (m.size() % 64 != 56) m.push_back('\0');
    for (int i = 7; i >= 0; --i)
        m.push_back(static_cast<char>((bit_len >> (i * 8)) & 0xff));

    for (size_t block = 0; block < m.size(); block += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const auto* p = reinterpret_cast<const unsigned char*>(m.data() + block + i * 4);
            w[i] = (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
                   (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + kSha256K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d  = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t v : h) {
        for (int i = 7; i >= 0; --i) out.push_back(kHex[(v >> (i * 4)) & 0xf]);
    }
    return out;
}

std::string sha256_hex(std::string_view bytes) { return sha256_impl(bytes); }

std::string sha256_hex_secure(std::string_view bytes) { return sha256_impl(bytes); }

}  // namespace lgc
