// The MTP head-fit capture: the target's row and the head's row at every
// position of a replayed text, off by default.
//
// Campaign mtp-head-fit, experiment E1: before a draft head is refitted, a
// card run records what the two contexts produce over generated text, so
// `tools/mtp_fit.py` can score and refit the head offline.
//
// ARCINT_MTP_CAPTURE=<base> turns it on. Unset or empty is OFF and costs one
// getenv at the drafter's construction and a pointer test a decode; nothing
// is opened and the serving path is unchanged. On, it needs the draft
// vocabulary (--llama-mtp-vocab: the MTP context then returns a row for every
// entry, which is what makes the head's row available at every position; the
// drafter refuses to start without it), it holds the 0067 skip off so both
// contexts produce rows everywhere, and the backend starts every request on an
// empty lane (no prefix reuse), so that every prompt token is decoded and
// captured. It records the PROMPT pass only (the all-but-last tokens of a
// request, decoded without logits); the verify rows are not captured. A
// replay sends the text as the prompt with max_tokens 1.
//
// Three files are created, named <base>.h, <base>.z and <base>.tok. A file
// that already exists is not opened: a capture never mixes two runs.
//
//   <base>.h    the target's row h(p)      f32, `dim` values a record
//   <base>.z    the MTP head's row z(p)    f32, `dim` values a record
//   <base>.tok  req, pos, tok              i32 x 3 a record
//
// Record k of the three files is one position. `pos` is the position in its
// request, `tok` the token AT that position, `req` the request's ordinal in
// this process (0, 1, ...; a chunk that starts at position 0 opens the next
// one for its lane). Both rows are post-norm hidden states, the vectors the
// output head multiplies, and both predict the token at pos + 1:
//
//   h(p)  the trunk's row at p            (target logits at p = W h(p))
//   z(p)  the head's output for the entry (tok(p), h(p - 1)); for p = 0 the
//         entry's row input is the zero row, as at a served sequence start
//
// so the next token of record k is tok of the record with the same req and
// pos + 1 (the next record of that request). A request's last record has no
// known next token (the prompt's last token is decoded in the verify, which is
// not captured); a consumer drops it. Records of one request are in position
// order; requests of different lanes may interleave, (req, pos) is the key.
//
// File layout: a 64-byte header, then the records, little endian (the build
// hosts are x86-64 and aarch64). Header: bytes 0-7 the magic "ARCMTPC1"; u32
// version (1) at 8; u32 kind (1 = h, 2 = z, 3 = tok) at 12; u32 dim at 16 (the
// row width; 3 for the index); u32 dtype at 20 (0 = f32, 1 = i32); u64 count
// at 24 (the records written); zeros to 64. The count is rewritten after every
// append, so a file read while the process runs, or after a kill, is
// consistent: records past `count` are not valid. File size = 64 + count *
// dim * 4 once the process is done.
//
// Device-free so the format is testable without a card:
// tests/test_mtp_capture.cpp writes a fixed capture and compares it, byte for
// byte, with tests/python/fixtures/mtp_capture_fixture.*, which
// tests/python/test_mtp_fit.py reads with the tool's own reader.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace lgc {

inline constexpr size_t   kMtpCaptureHeaderBytes = 64;
inline constexpr uint32_t kMtpCaptureVersion     = 1;

enum MtpCaptureKind : uint32_t { kCaptureRowsH = 1, kCaptureRowsZ = 2, kCaptureIndex = 3 };
enum MtpCaptureDtype : uint32_t { kCaptureF32 = 0, kCaptureI32 = 1 };

struct MtpCaptureHeader {
    uint32_t version = kMtpCaptureVersion;
    uint32_t kind    = 0;
    uint32_t dim     = 0;
    uint32_t dtype   = kCaptureF32;
    uint64_t count   = 0;
};

// ARCINT_MTP_CAPTURE=<base>; null when unset or empty
inline const char* mtp_capture_path() {
    const char* f = std::getenv("ARCINT_MTP_CAPTURE");
    return f != nullptr && f[0] != '\0' ? f : nullptr;
}

inline bool mtp_capture_on() { return mtp_capture_path() != nullptr; }

inline void mtp_capture_put32(unsigned char* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xffu);
}
inline void mtp_capture_put64(unsigned char* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xffu);
}
inline uint32_t mtp_capture_get32(const unsigned char* p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
inline uint64_t mtp_capture_get64(const unsigned char* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

inline void mtp_capture_header_encode(const MtpCaptureHeader& h, unsigned char out[kMtpCaptureHeaderBytes]) {
    std::memset(out, 0, kMtpCaptureHeaderBytes);
    std::memcpy(out, "ARCMTPC1", 8);
    mtp_capture_put32(out + 8, h.version);
    mtp_capture_put32(out + 12, h.kind);
    mtp_capture_put32(out + 16, h.dim);
    mtp_capture_put32(out + 20, h.dtype);
    mtp_capture_put64(out + 24, h.count);
}

inline bool mtp_capture_header_decode(const unsigned char in[kMtpCaptureHeaderBytes], MtpCaptureHeader& h) {
    if (std::memcmp(in, "ARCMTPC1", 8) != 0) return false;
    h.version = mtp_capture_get32(in + 8);
    h.kind    = mtp_capture_get32(in + 12);
    h.dim     = mtp_capture_get32(in + 16);
    h.dtype   = mtp_capture_get32(in + 20);
    h.count   = mtp_capture_get64(in + 24);
    return h.version == kMtpCaptureVersion;
}

// The writer. One per process (the drafter owns it); append() is the only
// call on the decode path and takes a lock, so lanes decoding at once do not
// interleave inside a chunk.
class MtpCapture {
public:
    ~MtpCapture() {
        for (int fd : fd_)
            if (fd >= 0) ::close(fd);
    }

    // Creates <base>.h, <base>.z and <base>.tok; false with `err` when one
    // exists already or cannot be created (the three are removed again then)
    bool open(const std::string& base, int dim, std::string& err) {
        if (dim <= 0) {
            err = "ARCINT_MTP_CAPTURE: the row width is not positive";
            return false;
        }
        dim_ = dim;
        static const char* const ext[3] = { ".h", ".z", ".tok" };
        static const uint32_t    kind[3] = { kCaptureRowsH, kCaptureRowsZ, kCaptureIndex };
        for (int i = 0; i < 3; ++i) {
            path_[i] = base + ext[i];
            // O_EXCL: a capture never appends to, or overwrites, another run's file
            fd_[i] = ::open(path_[i].c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
            if (fd_[i] < 0) {
                err = "ARCINT_MTP_CAPTURE: cannot create " + path_[i] + ": " + std::strerror(errno) +
                      (errno == EEXIST ? " (a capture file is never overwritten or appended to: use a new base name)" : "");
                abandon();
                return false;
            }
            MtpCaptureHeader h;
            h.kind  = kind[i];
            h.dim   = i == 2 ? 3u : static_cast<uint32_t>(dim);
            h.dtype = i == 2 ? kCaptureI32 : kCaptureF32;
            kind_[i] = h;
            // pwrite leaves the file offset at 0: the records start after the header
            if (!write_header(i) || ::lseek(fd_[i], static_cast<off_t>(kMtpCaptureHeaderBytes), SEEK_SET) < 0) {
                err = "ARCINT_MTP_CAPTURE: cannot write the header of " + path_[i];
                abandon();
                return false;
            }
        }
        return true;
    }

    // One decoded chunk of a prompt: `n` positions starting at `pos0` of lane
    // `seq`, their tokens, the target's rows h (n x dim) and the head's rows z
    // (n x dim). A chunk at position 0 opens the lane's next request. False
    // when a write fails (the capture is then incomplete and says so loudly).
    bool append(int seq, size_t pos0, const int* toks, size_t n, const float* h, const float* z) {
        if (n == 0) return true;
        std::lock_guard<std::mutex> lk(mu_);
        if (seq < 0) return false;
        if (static_cast<size_t>(seq) >= req_.size()) req_.resize(static_cast<size_t>(seq) + 1, -1);
        if (pos0 == 0) req_[static_cast<size_t>(seq)] = static_cast<int64_t>(n_req_++);
        if (req_[static_cast<size_t>(seq)] < 0) return false;   // a chunk of a request whose start was never seen
        std::vector<int32_t> idx(n * 3);
        for (size_t i = 0; i < n; ++i) {
            idx[i * 3 + 0] = static_cast<int32_t>(req_[static_cast<size_t>(seq)]);
            idx[i * 3 + 1] = static_cast<int32_t>(pos0 + i);
            idx[i * 3 + 2] = static_cast<int32_t>(toks[i]);
        }
        const size_t row = static_cast<size_t>(dim_) * sizeof(float);
        if (!write_all(0, h, n * row) || !write_all(1, z, n * row) || !write_all(2, idx.data(), idx.size() * sizeof(int32_t)))
            return false;
        for (int i = 0; i < 3; ++i) {
            kind_[i].count += n;
            if (!write_header(i)) return false;
        }
        return true;
    }

    uint64_t records() const { return kind_[0].count; }
    uint64_t requests() const { return n_req_; }

private:
    void abandon() {
        for (int i = 0; i < 3; ++i) {
            if (fd_[i] >= 0) {
                ::close(fd_[i]);
                fd_[i] = -1;
                // only the files this open() created are removed (O_EXCL: the one that failed was not ours)
                if (!path_[i].empty()) ::unlink(path_[i].c_str());
            }
        }
    }
    bool write_header(int i) {
        unsigned char b[kMtpCaptureHeaderBytes];
        mtp_capture_header_encode(kind_[i], b);
        return ::pwrite(fd_[i], b, sizeof b, 0) == static_cast<ssize_t>(sizeof b);
    }
    bool write_all(int i, const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        while (n > 0) {
            const ssize_t w = ::write(fd_[i], c, n);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) return false;
            c += w;
            n -= static_cast<size_t>(w);
        }
        return true;
    }

    int              dim_ = 0;
    int              fd_[3] = { -1, -1, -1 };
    std::string      path_[3];
    MtpCaptureHeader kind_[3];
    std::vector<int64_t> req_;   // per lane, the request ordinal of its current request (-1: none yet)
    uint64_t         n_req_ = 0;
    std::mutex       mu_;
};

}  // namespace lgc
