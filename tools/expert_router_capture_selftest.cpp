// The device-free writer cell for the LAC1 router-input capture.
// tools/expert_router_capture.cpp writes captures with lac1::writer
// (tools/expert_router_capture_format.h); this program writes the same two
// synthetic tokens to argv[1] with the same writer, and
// tests/python/test_expert_router_capture.py compiles it with a plain C++
// compiler (no llama, no model) and requires its bytes to equal the Python
// mirror writer's (tools/expert_router_capture_read.py:write_capture),
// byte for byte, then reads both back with the Python reader. The geometry is
// a small stand-in (3 layers, 8 wide, 6 experts, 3 used), not Flash-Next's.
//
// The synthetic values are constants, so the fixture cannot drift:
//   inputs  f16 bits  0x3C00 + 0x40*il + k   (il 0..2, k 0..7)
//   ids     ((2*il + 7*k) % 6)               (il 0..2, k 0..2)
// token 0: position 0, token 100, phase 0 (prompt)
// token 1: position 7, token 42,  phase 1 (decode)
//
// RED MUTATIONS, each in the Python mirror writer, run and seen failing before
// the mutation was removed:
//  (a) the mirror writer drops the phase byte: token 1's position reads 8, not 7
//      (the reader shifts one byte; the byte-equality cell fails too).
//  (b) the mirror writer writes n_used+1 ids a layer: every later field shifts
//      two bytes; token 1's layers read wrong ids and inputs (byte equality
//      fails).
//
// Usage: expert_router_capture_selftest <out-path>
// Prints "expert_router_capture_selftest: OK" and the record geometry.
// Constants only: nothing here names a host, a user or a path from the run.

#include "expert_router_capture_format.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <out-path>\n", argv[0]);
        return 2;
    }

    const lac1::geometry g{3, 8, 6, 3};   // n_layer, n_embd, n_expert, n_used
    const uint64_t expect_token_bytes = lac1::token_bytes(g);
    if (lac1::HEADER_BYTES != 25 || expect_token_bytes != 8 + 1 + 3 * (2 * 8 + 2 * 3)) {
        std::fprintf(stderr, "format sizes: header %zu, token %llu\n",
                     lac1::HEADER_BYTES, (unsigned long long) expect_token_bytes);
        return 3;
    }
    // Flash-Next's record size, the number the handover quotes
    if (lac1::token_bytes(lac1::geometry{48, 2560, 512, 10}) != 246729) {
        std::fprintf(stderr, "Flash-Next record size %llu, not 246729\n",
                     (unsigned long long) lac1::token_bytes(lac1::geometry{48, 2560, 512, 10}));
        return 4;
    }

    std::vector<uint16_t> inputs((size_t) g.n_layer * g.n_embd);
    std::vector<int16_t>  ids((size_t) g.n_layer * g.n_used);
    for (uint32_t il = 0; il < g.n_layer; ++il) {
        for (uint32_t k = 0; k < g.n_embd; ++k) {
            inputs[(size_t) il * g.n_embd + k] = (uint16_t) (0x3C00 + 0x40 * il + k);
        }
        for (uint32_t k = 0; k < g.n_used; ++k) {
            ids[(size_t) il * g.n_used + k] = (int16_t) ((2 * il + 7 * k) % g.n_expert);
        }
    }

    lac1::writer w;
    if (!w.open(argv[1], g)) {
        std::fprintf(stderr, "cannot open %s: %s\n", argv[1], w.error() ? w.error() : "?");
        return 1;
    }
    if (!w.write_token(0, 100, lac1::PHASE_PROMPT, inputs.data(), ids.data())) {
        std::fprintf(stderr, "token 0: %s\n", w.error() ? w.error() : "?");
        return 1;
    }
    if (!w.write_token(7, 42, lac1::PHASE_DECODE, inputs.data(), ids.data())) {
        std::fprintf(stderr, "token 1: %s\n", w.error() ? w.error() : "?");
        return 1;
    }
    if (!w.close()) {
        std::fprintf(stderr, "close: %s\n", w.error() ? w.error() : "?");
        return 1;
    }
    if (w.records != 2) {
        std::fprintf(stderr, "%llu records, not 2\n", (unsigned long long) w.records);
        return 1;
    }

    std::printf("expert_router_capture_selftest: OK (n_layer %u n_embd %u n_used %u, "
                "%zu header bytes, %llu token bytes, %llu total)\n",
                g.n_layer, g.n_embd, g.n_used, lac1::HEADER_BYTES,
                (unsigned long long) expect_token_bytes,
                (unsigned long long) (lac1::HEADER_BYTES + 2 * expect_token_bytes));
    return 0;
}
