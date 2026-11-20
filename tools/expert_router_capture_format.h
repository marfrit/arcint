#pragma once

// The LAC1 file a router-input capture writes (campaign expert-foresight,
// "Queued 2026-11-19: a layer-ahead predictor", step 1: a trace with the
// hidden state at each MoE input). tools/expert_router_capture.cpp writes it;
// tools/expert_router_capture_read.py reads it; the device-free round trip is
// tests/python/test_expert_router_capture.py, which also compiles
// tools/expert_router_capture_selftest.cpp against THIS header alone (no llama,
// no model) and requires its bytes to equal the Python writer's.
//
// This header is deliberately dependency-free (standard C++ only) so that
// selftest compiles alone:
//
//   header   magic "LAC1" (4 bytes), u32 version, u32 n_layer, u32 n_embd,
//            u32 n_expert, u32 n_used, u8 dtype  = 25 bytes, little endian
//   token    i32 position, i32 token id, u8 phase, then n_layer x
//              { n_embd x u16 input (dtype 1: f16, little endian,
//                token-major, contiguous: the tensor's row for this token) }
//              { n_used x i16 ids, little endian, in rank order: the rows of
//                "ffn_moe_topk", descending, exactly as the router selected }
//   phase    0 = prompt, 1 = decode
//
// A token record is 8 + 1 + n_layer x (2*n_embd + 2*n_used) bytes; for
// Flash-Next (48 layers, n_embd 2560, 10 of 512 experts) 246,729 bytes
// (~241 KiB), prompt and decode alike. The file has no trailer and no
// padding; a torn last record is detected by the reader (a short record is
// dropped, never half-read).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace lac1 {

constexpr char     MAGIC[4]   = {'L', 'A', 'C', '1'};
constexpr uint32_t VERSION    = 1;
constexpr uint8_t  DTYPE_F16  = 1;
constexpr uint8_t  PHASE_PROMPT = 0;
constexpr uint8_t  PHASE_DECODE = 1;

constexpr size_t HEADER_BYTES = 4 + 4 + 4 + 4 + 4 + 4 + 1;   // 25

struct geometry {
    uint32_t n_layer = 0;
    uint32_t n_embd  = 0;
    uint32_t n_expert = 0;
    uint32_t n_used  = 0;   // top-k
};

inline uint64_t token_bytes(const geometry & g) {
    return 8 + 1 + (uint64_t) g.n_layer * (2 * (uint64_t) g.n_embd + 2 * (uint64_t) g.n_used);
}

namespace detail {

inline void put_u32(std::uint8_t * p, std::uint32_t v) {
    p[0] = (std::uint8_t) (v      );
    p[1] = (std::uint8_t) (v >>  8);
    p[2] = (std::uint8_t) (v >> 16);
    p[3] = (std::uint8_t) (v >> 24);
}

inline void put_u16(std::uint8_t * p, std::uint16_t v) {
    p[0] = (std::uint8_t) (v     );
    p[1] = (std::uint8_t) (v >> 8);
}

inline void put_i16(std::uint8_t * p, std::int16_t v) {
    put_u16(p, (std::uint16_t) v);
}

inline void put_i32(std::uint8_t * p, std::int32_t v) {
    put_u32(p, (std::uint32_t) v);
}

} // namespace detail

// Serialises the 25-byte header into `out` (at least HEADER_BYTES).
inline void header_bytes(const geometry & g, std::uint8_t * out) {
    std::memcpy(out, MAGIC, 4);
    detail::put_u32(out +  4, VERSION);
    detail::put_u32(out +  8, g.n_layer);
    detail::put_u32(out + 12, g.n_embd);
    detail::put_u32(out + 16, g.n_expert);
    detail::put_u32(out + 20, g.n_used);
    out[24] = DTYPE_F16;
}

// Serialises one token record into `out` (token_bytes(g) bytes). `inputs` is
// n_layer*n_embd f16 values already in the file's per-layer token-major order;
// `ids` is n_layer*n_used int16 values in rank order.
inline void token_bytes(const geometry & g, std::int32_t pos, std::int32_t token,
                        std::uint8_t phase,
                        const std::uint16_t * inputs, const std::int16_t * ids,
                        std::uint8_t * out) {
    detail::put_i32(out,     pos);
    detail::put_i32(out + 4, token);
    out[8] = phase;
    std::uint8_t * p = out + 9;
    for (uint32_t il = 0; il < g.n_layer; ++il) {
        const std::uint16_t * row = inputs + (uint64_t) il * g.n_embd;
        for (uint32_t k = 0; k < g.n_embd; ++k) {
            detail::put_u16(p, row[k]);
            p += 2;
        }
        const std::int16_t * ids_row = ids + (uint64_t) il * g.n_used;
        for (uint32_t k = 0; k < g.n_used; ++k) {
            detail::put_i16(p, ids_row[k]);
            p += 2;
        }
    }
}

// The file writer the tool uses: header once, then one token record a call.
// Buffered by std::FILE; close() reports a failed final flush.
class writer {
public:
    writer() = default;
    writer(const writer &) = delete;
    writer & operator=(const writer &) = delete;
    ~writer() { close(); }

    bool open(const char * path, const geometry & g) {
        close();
        geom = g;
        f = std::fopen(path, "wb");
        if (f == nullptr) {
            err = "cannot open the output file";
            return false;
        }
        std::uint8_t h[HEADER_BYTES];
        header_bytes(geom, h);
        if (std::fwrite(h, 1, sizeof(h), f) != sizeof(h)) {
            err = "cannot write the header";
            close();
            return false;
        }
        records = 0;
        return true;
    }

    bool write_token(std::int32_t pos, std::int32_t token, std::uint8_t phase,
                     const std::uint16_t * inputs, const std::int16_t * ids) {
        if (f == nullptr) {
            err = "the writer is not open";
            return false;
        }
        buf.resize(token_bytes(geom));
        token_bytes(geom, pos, token, phase, inputs, ids, buf.data());
        if (std::fwrite(buf.data(), 1, buf.size(), f) != buf.size()) {
            err = "cannot write a token record";
            close();
            return false;
        }
        ++records;
        return true;
    }

    bool close() {
        bool ok = true;
        if (f != nullptr) {
            if (std::fclose(f) != 0) {
                ok = false;
                err = "cannot close the output file";
            }
            f = nullptr;
        }
        return ok;
    }

    uint64_t records = 0;
    const char * error() const { return err.empty() ? nullptr : err.c_str(); }

private:
    std::FILE * f = nullptr;
    geometry geom;
    std::vector<std::uint8_t> buf;
    std::string err;
};

} // namespace lac1
