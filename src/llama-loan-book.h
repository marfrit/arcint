#pragma once

// The prefill-slot-loan's bookkeeping rules as plain data: no llama, no device
// (ggml's type enum only, for the region-copy rule). llama-expert-cache.cpp
// calls exactly these (third review, findings 1, 2, 3, 5, 6, 7, 8; round 8's
// review items), and arcint's device-free test exercises the same functions
// (tests/test_loan_book.cpp), so a mutation of the rule fails a unit case
// before any card run.

#include "ggml.h"

#include <algorithm>
#include <cerrno>   // round 22: the query's why carries errno
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>   // round 13: the watchdog's line is one write(2), async-signal-safe
#include <dirent.h>
#include <string>
#include <vector>
#include <sys/ioctl.h>   // round 21: DRM_XE_DEVICE_QUERY_MEM_REGIONS on the xe render fd

#if __has_include(<drm/xe_drm.h>)
#include <drm/xe_drm.h>
#define LLAMA_LOAN_HAVE_XE_QUERY 1
#else
#define LLAMA_LOAN_HAVE_XE_QUERY 0
#endif

namespace llama_loan_book {

// Finding 6: a lend that cannot be published puts its tables back. lend() moves
// each ex-slot expert's slot_of into loan_of and sets slot_of = -1 before it
// knows the relay reached the card; the failure return then left lent = true and
// loan_out_ false, so the caller's recovery (refill only while loan_out()) did
// nothing and every ex-slot expert was unreachable (tab_comb -1, adapt over
// slot_of -1). The restore half: loan_of -> slot_of, the loan entries cleared,
// lent cleared. false when the layer was not lent (nothing to put back).
template <typename LayerT>
inline bool undo_lend(LayerT & L) {
    if (!L.lent) {
        return false;
    }
    const size_t n = L.slot_of.size() < L.loan_of.size() ? L.slot_of.size() : L.loan_of.size();
    for (size_t e = 0; e < n; ++e) {
        if (L.loan_of[e] < 0) {
            continue;
        }
        L.slot_of[e] = L.loan_of[e];
        L.loan_of[e] = -1;
    }
    L.lent = false;
    return true;
}

// Finding 7 (third review): the undo is not enough on its own -- the tables it
// rewrote are written to the card without a wait, and a lend whose relay never
// landed leaves them short. The cache checks them again after the undo and, when
// they still differ, marks itself unusable: every later request is refused (a
// lent slot read as an expert would answer wrong bytes). This is that decision,
// as one value.
inline bool lend_broken(bool tables_still_differ) {
    return tables_still_differ;
}

// Finding 7 (second review): the loan's tail slot buffer is its own. The
// destructor walks the fixed buffer list and frees every entry; the loan frees
// the tail buffer and the refill re-allocates it, so an entry for it there is a
// double free of a dead pointer and a leak of the live one. take_tail() is how
// the cache takes ownership: it takes the buffer OUT of the list (a stale entry
// would be the bug) and never adds it. The destructor asserts the invariant.
template <typename BufT>
inline void take_tail(std::vector<BufT> & bufs, BufT & tail, BufT b) {
    for (auto it = bufs.begin(); it != bufs.end();) {
        it = (*it == b) ? bufs.erase(it) : it + 1;
    }
    tail = b;
}

template <typename BufT>
inline bool listed(const std::vector<BufT> & bufs, BufT tail) {
    if (tail == nullptr) {
        return false;
    }
    for (const BufT & b : bufs) {
        if (b == tail) {
            return true;
        }
    }
    return false;
}

// Finding 5 (third review): a cache set up for no loan (lend_bytes 0 -- under a
// layer split the other card) is not lent, and its refill is not asked. Its
// lend() and refill() return false, so the AND the caller runs over every cache
// would turn a good loan or a good return into a refusal. The caller skips it.
inline bool loan_cache(size_t lend_bytes) {
    return lend_bytes != 0;
}

// Finding 8 (third review): the flag rule lives in init, not in the tool.
// llama_model_params.slot_loan_bytes == LLAMA_SLOT_LOAN_BYTES_ENV is the
// "no parameter set" sentinel llama_model_default_params sets; only then is
// LLAMA_SLOT_LOAN_MIB read. An explicit value -- 0 included -- is what the tool
// asked for, so a stray environment variable cannot turn a loan on behind a
// flag that left it off.
inline size_t resolve_lend_bytes(size_t param, size_t sentinel, long long mib_env) {
    if (param != sentinel) {
        return param;
    }
    return mib_env > 0 ? (size_t) mib_env << 20 : 0;
}

// The lend set: computed from scratch over every layer. A layer is a candidate
// when its banks are host memory and it is not Q8_0 (a Q8_0 bank lives in the
// slot buffer and is freed by the loan). The separate tail buffer holds the
// chosen layers' slot tensors as one tensor list, so the set need not be a
// contiguous range: the scan starts at the tail and skips a non-lendable layer
// or one whose bytes do not fit the remaining budget, then continues earlier.
// Nothing chosen means the loan is off: a configured loan must say why and run
// with no loan, never create tensors in a null context (the first card run's
// segfault: the tail context was never created but the eligible layers were
// left marked).
struct LendCandidate {
    bool    lendable = false;   // host banks, not Q8_0
    int64_t bytes    = 0;       // n_slots x the three slot tensors' bytes
};

struct LendPick {
    std::vector<int> chosen;    // entries of `layers`, ascending
    int64_t          bytes = 0; // their slot bytes
    const char *     why   = nullptr;   // non-null when nothing was chosen
};

inline LendPick choose_lend_layers(const std::vector<LendCandidate> & cand, size_t budget) {
    LendPick p;
    if (budget == 0) {
        p.why = "the loan budget is 0";
        return p;
    }
    bool any = false;
    int64_t used = 0;
    for (int i = (int) cand.size() - 1; i >= 0; --i) {
        if (!cand[(size_t) i].lendable) {
            continue;
        }
        any = true;
        const int64_t b = cand[(size_t) i].bytes;
        if (b <= 0 || used + b > (int64_t) budget) {
            continue;   // this layer does not fit; an earlier smaller one may
        }
        used += b;
        p.chosen.push_back(i);
    }
    std::sort(p.chosen.begin(), p.chosen.end());
    p.bytes = used;
    if (p.chosen.empty()) {
        p.why = any ? "the loan budget is smaller than one lendable layer"
                    : "no layer has a host-bank lendable tail";
    }
    return p;
}

// The two fixed margins (operator's decision 2026-10-30, "Margin ok"): the
// 384 MiB peak reserve over the prompt's buffers and a fixed 1 GiB under the
// card's total. Round 17 put them both in the lend-vs-need rule; round 19 puts the
// card's 1 GiB where it belongs -- against the card's own total (see CardFacts).
constexpr size_t kPeakReserve = 384u << 20;
constexpr size_t kCardMargin  = 1u << 30;   // 1 GiB, fixed, under the card's total
constexpr size_t kLendMargin   = kPeakReserve + kCardMargin;

inline bool size_fits(size_t need, size_t lend);   // the fixed-margin rule, defined below
inline bool card_fits(size_t resident, size_t need, size_t card_total, size_t margin, size_t lend = 0, size_t first_touch = 0);
struct CardFacts;   // round 21: the dispatch below, declared before the growth uses it
inline bool card_fits(const CardFacts & f, size_t lend_extra);

// LLAMA_SLOT_LOAN_CARD_MARGIN_MIB overrides the fixed card margin, in MiB. A
// missing, unreadable or non-positive value keeps the operator's 1 GiB: the knob
// may move the margin, it may not silently remove it (a typo is not a decision).
constexpr size_t kCardMarginMax = (size_t) 16 << 30;   // the knob's ceiling: 16 GiB, above any card served
inline size_t resolve_card_margin(const char * env) {
    if (env == nullptr) {
        return kCardMargin;
    }
    long long mib = std::atoll(env);
    if (mib <= 0) return kCardMargin;
    if (mib > (long long) (kCardMarginMax >> 20)) mib = (long long) (kCardMarginMax >> 20);
    return (size_t) mib << 20;   // clamped above: the shift cannot wrap
}

// The process's own residency, read from /proc/self/fdinfo (round 20; round 24
// sums a device's clients). The xe driver reports the process's device-resident
// bytes there as `drm-resident-vram0`, per DRM client (`drm-client-id`) and
// device (`drm-pdev`); several render fds of one client repeat the same numbers,
// so the scan keeps one entry a client id. The OpenCL runtime opens TWO clients
// on one card in one process (a probe client and the real one, `measured-here`
// 2026-10-30, round 22's window), so a card's figure is the SUM over every
// distinct client id whose drm-pdev matches the device -- do not pick one
// client. The figure is the process's whole residency -- the OpenCL buffers, the
// runtime's kernels and scratch, the driver's objects, the bank-stream ring --
// which the ggml buffer sum cannot see. The ggml sum stays as the fallback and
// as the tripwire's reference.
struct FdinfoClient {
    long long   client_id = -1;
    int         fd        = -1;    // round 21: the render fd, for the kernel's region query
    std::string pdev;             // drm-pdev, "domain:bus:device.function"
    size_t      vram0     = 0;    // drm-resident-vram0 in bytes
    size_t      gtt       = 0;    // drm-resident-gtt in bytes (round 21's eviction guard)
};

struct FdinfoScan {
    std::vector<FdinfoClient> clients;
    size_t entries = 0;           // xe entries read
    size_t deduped = 0;           // extra fds of a client already counted
};

// One fdinfo value: "<n> KiB" (the driver's unit), also MiB, GiB, TiB or B. An
// unreadable value is 0, never a silent byte count. Round 22 adds TiB: a card
// above 1 TiB or a GTT figure reported in TiB would otherwise be read as raw
// bytes and understate the residency by a factor of 2^40.
inline size_t fdinfo_parse_bytes(const char * v) {
    if (v == nullptr) return 0;
    char * end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 10);
    if (end == v) return 0;
    while (*end == ' ' || *end == '\t') ++end;
    size_t shift = 0;
    if      (std::strncmp(end, "KiB", 3) == 0) shift = 10;
    else if (std::strncmp(end, "MiB", 3) == 0) shift = 20;
    else if (std::strncmp(end, "GiB", 3) == 0) shift = 30;
    else if (std::strncmp(end, "TiB", 3) == 0) shift = 40;
    else if (*end == 'B')                      shift = 0;
    else                                       shift = 0;
    if (shift != 0 && n > (SIZE_MAX >> shift)) return SIZE_MAX;
    return (size_t) n << shift;
}

// A xe client's entry: drm-driver xe, a drm-client-id and drm-resident-vram0.
// Other drivers' fdinfo (i915, amdgpu) is not this rule's figure; it is skipped.
inline bool fdinfo_parse_file(const char * path, FdinfoClient & out) {
    std::FILE * f = std::fopen(path, "r");
    if (f == nullptr) return false;
    std::string driver, pdev;
    long long   cid = -1;
    size_t      vram0 = 0, gtt = 0;
    bool        has_cid = false, has_vram = false;
    char        line[512];
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        char * colon = std::strchr(line, ':');
        if (colon == nullptr) continue;
        *colon = '\0';
        char * val = colon + 1;
        while (*val == ' ' || *val == '\t') ++val;
        char * e = val + std::strlen(val);
        while (e > val && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) --e;
        *e = '\0';
        if      (std::strcmp(line, "drm-driver") == 0)        driver = val;
        else if (std::strcmp(line, "drm-pdev") == 0)          pdev = val;
        else if (std::strcmp(line, "drm-client-id") == 0)   { cid = std::atoll(val); has_cid = true; }
        else if (std::strcmp(line, "drm-resident-vram0") == 0) { vram0 = fdinfo_parse_bytes(val); has_vram = true; }
        else if (std::strcmp(line, "drm-resident-gtt") == 0)   { gtt   = fdinfo_parse_bytes(val); }
    }
    std::fclose(f);
    if (driver != "xe" || !has_cid || !has_vram) return false;
    out.client_id = cid;
    out.pdev      = pdev;
    out.vram0     = vram0;
    out.gtt       = gtt;
    return true;
}

// The directory's xe entries, one a client id (two render fds of one client
// report the same residency; summing them would double it). `dir` is injectable
// so a device-free test can read a fake fdinfo directory; production passes
// /proc/self/fdinfo.
inline void scan_fdinfo(const char * dir, FdinfoScan & scan) {
    scan.clients.clear();
    scan.entries = 0;
    scan.deduped = 0;
    DIR * d = opendir(dir);
    if (d == nullptr) return;
    while (const dirent * e = readdir(d)) {
        char * end = nullptr;
        const long fd = std::strtol(e->d_name, &end, 10);
        if (end == e->d_name || *end != '\0' || fd < 0) continue;
        const std::string path = std::string(dir) + "/" + e->d_name;
        FdinfoClient c;
        if (!fdinfo_parse_file(path.c_str(), c)) continue;
        c.fd = (int) fd;   // /proc/self/fdinfo's entry name is the fd (round 21's query)
        ++scan.entries;
        bool merged = false;
        for (FdinfoClient & k : scan.clients) {
            if (k.client_id != c.client_id) continue;
            k.vram0 = std::max(k.vram0, c.vram0);
            k.gtt   = std::max(k.gtt,   c.gtt);
            if (k.pdev.empty()) k.pdev = c.pdev;
            ++scan.deduped;
            merged = true;
            break;
        }
        if (!merged) scan.clients.push_back(c);
    }
    closedir(d);
}

// This OpenCL device's figure: every xe client in /proc/self is this process's,
// so the card's residency is the SUM of the distinct client ids whose drm-pdev
// is the device's (CL_DEVICE_PCI_BUS_INFO_KHR via the device props' device_id).
// Two clients on one pdev (the runtime's probe client and the real one) are two
// terms, not a conflict; dedupe is by client id alone (the scan's job). The
// ioctl is device-wide, so ANY render fd of that pdev answers it. One xe client
// is trivial; several with no device drm-pdev are the fallback, with a WARN.
struct FdinfoAgg {
    size_t      vram0   = 0;   // the sum over the device's clients
    size_t      gtt     = 0;   // the sum over the device's clients (the guard's baseline)
    int         fd      = -1;  // any render fd of the device's pdev, for the region query
    int         clients = 0;   // distinct client ids summed
    bool        measured = false;
    const char * why    = nullptr;
};

inline FdinfoAgg sum_fdinfo(const FdinfoScan & scan, const char * pdev) {
    FdinfoAgg a;
    if (scan.clients.empty()) {
        a.why = "no xe client in the process's fdinfo";
        return a;
    }
    if (pdev == nullptr || *pdev == '\0') {
        if (scan.clients.size() == 1) {
            a.vram0    = scan.clients.front().vram0;
            a.gtt      = scan.clients.front().gtt;
            a.fd       = scan.clients.front().fd;
            a.clients  = 1;
            a.measured = true;
            a.why      = "one xe client";
        } else {
            a.why = "several xe clients and the OpenCL device reports no drm-pdev";
        }
        return a;
    }
    for (const FdinfoClient & c : scan.clients) {
        if (c.pdev.empty() || c.pdev != pdev) continue;
        a.vram0 += c.vram0;
        a.gtt   += c.gtt;
        if (a.fd < 0 || c.fd < a.fd) a.fd = c.fd;   // any fd of the pdev: the query is device-wide
        ++a.clients;
    }
    if (a.clients == 0) {
        a.why = "no xe client on the device's drm-pdev";
        return a;
    }
    a.measured = true;
    a.why = a.clients == 1 ? "drm-pdev match" : "the device's drm-pdev clients summed";
    return a;
}

// The figure the card rule uses and the tripwire. A matched client's vram0 below
// the ggml sum means the process's own buffers are already evicted to GTT (the
// driver moved them), and a loan must not be taken: it would build its reserve
// on an evicted process. `measured` false means the fallback (ggml sum).
struct ResidentRead {
    size_t      bytes    = 0;
    bool        measured = false;
    bool        evicted  = false;
    const char * why     = nullptr;
};

inline ResidentRead read_resident(const FdinfoScan & scan, const char * pdev, size_t ggml_sum) {
    const FdinfoAgg a = sum_fdinfo(scan, pdev);
    if (!a.measured) {
        return ResidentRead{ ggml_sum, false, false, a.why };
    }
    if (a.vram0 < ggml_sum) {
        return ResidentRead{ a.vram0, true, true, "fdinfo vram0 is below the ggml sum" };
    }
    return ResidentRead{ a.vram0, true, false, a.why };
}

// ---- Round 21: the kernel's own card state, the eviction guard, the dirty mirror ----

// The xe query's regions as plain data (one mem_class, its total and used). Kept
// outside the driver structs so a device-free test feeds the same rule a fake
// reply: tests/test_loan_book.cpp.
struct XeRegion {
    uint16_t mem_class = 0;
    uint64_t total     = 0;
    uint64_t used      = 0;
};
constexpr uint16_t kXeVramClass = 1;   // DRM_XE_MEM_REGION_CLASS_VRAM

// Sum only VRAM regions. A reply with no VRAM region is a failed query (the
// rule has no figure), never a zero total that would pass every check.
inline bool vram_from_regions(const std::vector<XeRegion> & regions, size_t & total, size_t & used) {
    total = 0;
    used  = 0;
    bool any = false;
    for (const XeRegion & r : regions) {
        if (r.mem_class != kXeVramClass) continue;
        any = true;
        if (r.total > SIZE_MAX - total || r.used > SIZE_MAX - used) {
            total = used = SIZE_MAX;
            return true;
        }
        total += (size_t) r.total;
        used  += (size_t) r.used;
    }
    return any;
}

// The kernel's device-wide figure, DRM_XE_DEVICE_QUERY_MEM_REGIONS on this
// process's xe render fd. `used` is the driver's ttm_resource_manager_usage --
// every client and the kernel's own VRAM objects, not a process's residency.
// `size == 0` asks the driver for the reply's size, then the same buffer is
// filled; a failed ioctl or a reply without VRAM is not a figure (the caller
// falls back to the per-process fdinfo figure). No perfmon capability is
// required on the served driver (campaign vram-user-space, `code`).
// Round 23: whether the kernel's used/total pair is a figure at all. used == 0
// with a total above it is the perfmon-less kernel's placeholder, not an empty
// card: the zero-margin rule would then fit every loan (campaign
// vram-user-space, `code`: the `if (perfmon_capable())` around used in older
// trees). No total, or a total of 0, is likewise no figure.
inline bool query_is_a_figure(size_t used, size_t total) {
    return total > 0 && used > 0;
}

struct CardQuery {
    bool        ok    = false;
    size_t      total = 0;
    size_t      used  = 0;
    std::string why;   // round 23: own storage, so a copied query keeps its reason
};

// A failed ioctl's reason with errno: "<what>: <strerror> (errno <n>)". A bare
// reason can hide ENOTTY, EPERM or EINVAL, which ask different next steps from
// the window. Round 23: the reason is a std::string, not a char buffer the
// struct points into -- a query is returned by value, and the old pointer kept
// pointing into the source's buffer after the copy.
inline std::string card_why_errno(const char * what, int err) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "%s: %s (errno %d)", what, std::strerror(err), err);
    return std::string(buf);
}

inline CardQuery xe_query_card(int fd) {
    CardQuery q;
#if LLAMA_LOAN_HAVE_XE_QUERY
    if (fd < 0) {
        q.why = "no xe render fd";
    } else {
        struct drm_xe_device_query dq = {};
        dq.query = DRM_XE_DEVICE_QUERY_MEM_REGIONS;
        dq.size  = 0;
        if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &dq) < 0) {
            q.why = card_why_errno("DRM_XE_DEVICE_QUERY_MEM_REGIONS size probe failed", errno);
        } else if (dq.size < sizeof(struct drm_xe_query_mem_regions)) {
            q.why = "DRM_XE_DEVICE_QUERY_MEM_REGIONS reply is too small";
        } else {
            std::vector<char> buf(dq.size);
            dq.data = (uintptr_t) buf.data();
            if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &dq) < 0) {
                q.why = card_why_errno("DRM_XE_DEVICE_QUERY_MEM_REGIONS read failed", errno);
            } else {
                const auto * mr = reinterpret_cast<const struct drm_xe_query_mem_regions *>(buf.data());
                const size_t need = sizeof(struct drm_xe_query_mem_regions) +
                                    (size_t) mr->num_mem_regions * sizeof(struct drm_xe_mem_region);
                if (need > buf.size()) {
                    q.why = "DRM_XE_DEVICE_QUERY_MEM_REGIONS reply is truncated";
                } else {
                    std::vector<XeRegion> rs;
                    rs.reserve(mr->num_mem_regions);
                    for (uint32_t i = 0; i < mr->num_mem_regions; ++i) {
                        const struct drm_xe_mem_region & r = mr->mem_regions[i];
                        rs.push_back(XeRegion{ r.mem_class, r.total_size, r.used });
                    }
                    if (vram_from_regions(rs, q.total, q.used)) {
                        q.ok = true;
                        // Round 23: a used of 0 with a total above it is no figure.
                        // A kernel without CAP_PERFMON (the `if (perfmon_capable())`
                        // around used in older trees) reports 0, which would make the
                        // zero-margin rule vacuous: a loan would fit any card. Treat
                        // it as a failed query, so the fdinfo fallback with its margin
                        // decides, and one WARN says why.
                        if (!query_is_a_figure(q.used, q.total)) {
                            q.ok = false;
                            char b[192];
                            std::snprintf(b, sizeof(b), "DRM_XE_DEVICE_QUERY_MEM_REGIONS reports used 0 with a total of %zu bytes: "
                                          "the kernel's used is not a figure (no CAP_PERFMON); the fdinfo fallback decides", q.total);
                            q.why = b;
                        }
                    } else {
                        q.why = "the query has no VRAM region";
                    }
                }
            }
        }
    }
#else
    (void) fd;
    q.why = "no drm/xe_drm.h at build";
#endif
    return q;
}


// Round 22: what the card's used figure holds beyond this process's fdinfo
// residency -- the kernel's own VRAM objects and every other client (another
// process on the card). It is INFORMATION: `used` already contains those
// bytes, so the rule must not subtract them as well. Round 21 did subtract them
// and counted every other client's residency twice, refusing a loan on a card
// that had room. Logged with the probe and the `loan out` line, never a term.
inline size_t used_beyond_fdinfo(bool query_ok, size_t used, size_t vram0) {
    if (!query_ok || used < vram0) return 0;
    return used - vram0;
}

// Round 22: the one optional allowance for growth while the prefill runs. The
// kernel's `used` is read before the lend; allocations that first appear after
// that read (a grow-only prealloc first used by the prompt, the cold ring's
// pages, the driver's own objects) are not in it. The operator's rule is zero
// margin, so the default is 0; an operator who needs a band for those growths
// sets LLAMA_SLOT_LOAN_PREFILL_GROWTH_MIB. A non-numeric value keeps the 0
// default (a typo is not a decision), and the value is clamped at 16 GiB.
constexpr size_t kPrefillGrowthMax = (size_t) 16 << 30;
inline size_t resolve_prefill_allowance(const char * env) {
    if (env == nullptr || env[0] == '\0') return 0;
    bool digits = true;
    for (const char * p = env; *p != '\0'; ++p) digits = digits && (*p >= '0' && *p <= '9');
    if (!digits) return 0;
    long long mib = std::atoll(env);
    if (mib <= 0) return 0;
    if (mib > (long long) (kPrefillGrowthMax >> 20)) mib = (long long) (kPrefillGrowthMax >> 20);
    return (size_t) mib << 20;
}

// The kernel rule (operator 2026-10-30: zero margin, knowing the card's real
// state). Round 22 corrects the round-21 form: `used` is the device-wide buddy
// usage (xe_ttm_vram_mgr: mm.size - mm.avail), so it ALREADY holds this
// process's objects, every other client's and the kernel's own, plus allocator
// rounding. After the lend frees its slots:
//
//     used - lend + need + first_touch + allowance <= total
//
// The margin beyond `used` is zero -- nothing is subtracted a second time. The
// only allowance is the prefill growth band above (default 0), named for what it
// is. `used - fdinfo_vram0` is logged as the other clients' and kernel objects'
// share of `used`, information. 0 total: no figure, the rule is not checked.
inline bool card_state_fits(size_t used, size_t total, size_t lend, size_t need,
                            size_t first_touch, size_t allowance) {
    if (total == 0) return true;
    if (need == SIZE_MAX || first_touch == SIZE_MAX || allowance == SIZE_MAX) return false;
    const size_t after = used >= lend ? used - lend : 0;
    if (need > total) return false;                            // even an empty card does not fit
    if (first_touch > total - need) return false;             // the cold ring alone over the total
    if (allowance > total - need - first_touch) return false;
    return after <= total - need - first_touch - allowance;   // no overflow: compared by subtraction
}

// The off words, compared without case: "off" is the documented one, "false" and
// "no" are the accepted spellings; "0" is the numeric one (round 22's guard knob
// writes 0). One predicate, so every knob reads the off words the same way.
inline bool off_word(const char * e) {
    if (e == nullptr) return false;
    if (e[0] == '0' && e[1] == '\0') return true;
    static const char * const words[] = {"off", "false", "no"};
    for (const char * w : words) {
        size_t i = 0;
        for (; e[i] != '\0' && w[i] != '\0'; ++i) {
            const char a = (e[i] >= 'A' && e[i] <= 'Z') ? (char) (e[i] - 'A' + 'a') : e[i];
            if (a != w[i]) {
                break;
            }
        }
        if (e[i] == '\0' && w[i] == '\0') {
            return true;   // the whole word, any case (no POSIX strncasecmp: portable)
        }
    }
    return false;
}

// The eviction guard's knob, LLAMA_SLOT_LOAN_EVICTION_GUARD: absent is on; 0,
// off, false, no (any case) turn it off. A typo keeps it on -- a guard a window
// asked to disable must be disabled by its documented word, not by a guess.
inline bool resolve_eviction_guard(const char * env) {
    return !off_word(env);
}

// Round 22: a shed decays. Round 21 carried the measured bytes into every later
// loan, so one trip (the unexplained ~4 GiB of extra GTT at 103k tokens) shrank
// the lend for the process's lifetime. A loan that completes without a trip is a
// clean loan: the carried figure is cleared. `tripped` is a true shortfall over
// this loan's end guard (round 23: one above the noise floor); anything else
// clears the ledger.
inline size_t shed_after_loan(size_t shed, bool tripped) {
    return tripped ? shed : 0;
}

// Round 23: the end guard's noise floor. vram_short is expected - fdinfo vram0,
// and the expected figure is exact tensor bytes while vram0 is BO residency:
// allocator rounding, the reserve's allowed under-run and reserve_matches'
// allowance push vram0 a few MiB below `expected` on a healthy loan, so a 1-byte
// shortfall is not evidence of an eviction. A shortfall at or above the floor is
// a true one; below it, the guard does nothing (no state is carried). The floor
// defaults to 64 MiB and LLAMA_SLOT_LOAN_SHED_FLOOR_MIB moves it (0 allowed:
// every byte counts); a non-numeric value keeps the default (a typo is not a
// decision), and the value is clamped at 16 GiB like the other knobs.
constexpr size_t kShedFloor = 64u << 20;
constexpr size_t kShedFloorMax = (size_t) 16 << 30;
inline size_t resolve_shed_floor(const char * env) {
    if (env == nullptr || env[0] == '\0') return kShedFloor;
    bool digits = true;
    for (const char * p = env; *p != '\0'; ++p) digits = digits && (*p >= '0' && *p <= '9');
    if (!digits) return kShedFloor;
    long long mib = std::atoll(env);
    if (mib < 0) return kShedFloor;
    if (mib > (long long) (kShedFloorMax >> 20)) mib = (long long) (kShedFloorMax >> 20);
    return (size_t) mib << 20;
}

// Round 23: the shortfall the next lend must grow by, or 0 for noise. The shed
// no longer drops grown layers: that only reduced the headroom the loan creates
// (`used - lend + need`), so it could not relieve the pressure that caused an
// eviction. A true shortfall feeds the NEXT loan's sizing as extra lend bytes
// (the existing growth path), and is never a refusal.
inline size_t growth_from_shortfall(size_t shortfall, size_t floor) {
    return shortfall >= floor ? shortfall : 0;
}

// The eviction guard (round 21, corrected in round 22). At the loan's begin the
// process's fdinfo GTT is taken as its BASELINE (`gtt_base`, `host_banks_base`);
// at the end the growth since it is judged. The process's other legitimate GTT --
// the runtime's command buffers and heaps, the stage-inputs USM ring, ids_host --
// is in the baseline and is not a surplus, so a false surplus at a begin cannot
// fire. `bank_growth` is the host-USM bank bytes added in between (a grown
// layer's bank), the part GTT growth should legitimately carry; `gtt_over` is the
// growth beyond it, logged. The guard ACTS on `vram_short` alone -- the process's
// own buffers below the expected VRAM, measured against the figure the begin
// recorded (ggml sum minus the lend, plus the loan's compute reserve and the
// non-sum residency measured at the begin). GTT growth is the same eviction's
// other face and is reported, not acted on: acting on it alone is what round 21
// got wrong. `shed` is false whenever `vram_short` is 0.
struct EvictionGuard {
    bool   shed        = false;
    size_t bytes       = 0;   // the amount to shed from the next loan (vram_short)
    size_t vram_short  = 0;   // the primary signal: expected VRAM not resident
    size_t gtt_growth  = 0;   // GTT since the begin's baseline (information)
    size_t bank_growth = 0;   // the host banks added since the baseline (the ledger)
    size_t gtt_over    = 0;   // GTT growth beyond the bank growth (logged, not acted on)
};
inline EvictionGuard eviction_guard(size_t vram0, size_t gtt, size_t expected_vram, size_t host_banks,
                                    size_t gtt_base, size_t host_banks_base) {
    EvictionGuard g;
    g.gtt_growth  = gtt >= gtt_base ? gtt - gtt_base : 0;
    g.bank_growth = host_banks >= host_banks_base ? host_banks - host_banks_base : 0;
    g.gtt_over    = g.gtt_growth > g.bank_growth ? g.gtt_growth - g.bank_growth : 0;
    g.vram_short  = expected_vram > vram0 ? expected_vram - vram0 : 0;
    g.bytes       = g.vram_short;
    g.shed        = g.bytes > 0;
    return g;
}

// The dirty slots of a lend bank. A swap changes which expert a slot holds; the
// per-swap mirror used to write the incoming rows into the host lend bank at
// once (~1.5 MiB a swap, ~10x the swap's own cost on the card). Round 21 marks
// the slot instead and copies only the dirty rows at the next loan's begin, one
// read a run, before the lend frees anything. `runs()` is [start, count) pairs in
// ascending slot order, so the flush reads contiguous runs and nothing else.
struct DirtyTrack {
    std::vector<uint8_t> flags;

    void reset(size_t n_slots) { flags.assign(n_slots, 0); }
    void mark(int slot) {
        if (slot >= 0 && (size_t) slot < flags.size()) flags[(size_t) slot] = 1;
    }
    bool any() const {
        for (uint8_t f : flags) if (f) return true;
        return false;
    }
    size_t rows() const {
        size_t n = 0;
        for (uint8_t f : flags) n += f ? 1 : 0;
        return n;
    }
    std::vector<std::pair<int, int>> runs() const {
        std::vector<std::pair<int, int>> out;
        int start = -1;
        for (int s = 0; s < (int) flags.size(); ++s) {
            if (flags[(size_t) s]) {
                if (start < 0) start = s;
            } else if (start >= 0) {
                out.emplace_back(start, s - start);
                start = -1;
            }
        }
        if (start >= 0) out.emplace_back(start, (int) flags.size() - start);
        return out;
    }
    void clear() { flags.assign(flags.size(), 0); }
};

// The bank-stream ring's K, as the backend parses it: below 2 gives 4.
inline int resolve_bank_stream_ring(const char * env) {
    const int v = env != nullptr ? std::atoi(env) : 4;
    return v >= 2 ? v : 4;
}

// The ring's pages the first prefill graph makes resident: K x the largest bank
// tensor, counted only while the ring is not yet allocated (a cold process; a
// warm one's ring is in its fdinfo figure already).
inline size_t first_touch_bytes(int ring_k, size_t ring_bytes, size_t largest_bank) {
    if (ring_bytes != 0 || ring_k <= 0) return 0;
    if ((size_t) ring_k != 0 && largest_bank > SIZE_MAX / (size_t) ring_k) return SIZE_MAX;
    return (size_t) ring_k * largest_bank;
}

// The card's own bound (operator's decision 2026-10-30, "Margin ok", round 19;
// round 20 moves the figure to the process's own residency):
//
//     fdinfo_vram0 - lend + need + first_touch <= card_total - margin
//
// Round 20: `resident` is the process's own residency from its xe render fds'
// `/proc/self/fdinfo` (`drm-resident-vram0`), deduped by `drm-client-id` and
// matched to the device by `drm-pdev` against OpenCL's
// `CL_DEVICE_PCI_BUS_INFO_KHR`; round 24 sums every distinct client id on the
// device's pdev (the runtime opens two). That figure includes what the ggml sum
// cannot: the bank-stream ring, the backend's grow-only preallocations, the
// runtime's kernels, scratch and heaps. The ggml sum over the backend buffers
// remains the fallback (several xe clients, no pdev known) and the tripwire (a
// process whose own buffers are already evicted to GTT is refused). The cold
// ring's pages are the one part the fdinfo figure cannot have yet: `first_touch`
// is K x the largest bank tensor while the ring is not allocated.
//
// `card_total` is OpenCL's `CL_DEVICE_GLOBAL_MEM_SIZE` as llama's device props
// report it (`ggml_backend_dev_get_props` -> `memory_total`); NOT the free query,
// which is `total - 1 GiB` and could never fail. 0 means no card figure: the rule
// is not checked, and the lend-vs-need rule alone decides.
struct CardFacts {
    size_t total    = 0;   // the card's total memory; 0: unknown, no card rule
    size_t resident = 0;   // the rule's figure there before the lend: fdinfo vram0, or the sum
    size_t lend     = 0;   // the freed bytes on the card this pick does not account for
    size_t need     = 0;   // this card's own share of the prompt's compute buffers
    size_t margin   = kCardMargin;   // the fallback rule's margin (query failed)
    size_t sum      = 0;   // the ggml backend buffers' sum: the fallback and the tripwire
    size_t first_touch = 0;   // the cold ring's K x largest bank tensor, else 0
    bool   measured = false;  // resident is the process's fdinfo figure
    // Round 21: the kernel's device-wide figure. Round 22: when query_ok the rule
    // is `used - lend + need + first_touch + allowance <= total` -- zero margin,
    // nothing subtracted twice. `used_beyond` (used - fdinfo_vram0) is logged, not
    // subtracted; `allowance` is the prefill growth band, default 0. Without the
    // query the round-20 fdinfo + `margin` fallback stays.
    size_t used        = 0;
    size_t allowance   = 0;   // the prefill growth band, default 0
    size_t used_beyond = 0;   // used - fdinfo_vram0: information, never a term
    bool   query_ok    = false;
    size_t gtt         = 0;
};

// The fixed-margin rule (operator's decision 2026-10-30, "Margin ok"): the
// prompt's reserve may take the freed lend bytes only with a margin left over on
// both sides of the loan -- the 384 MiB peak reserve and the card margin. A loan
// that does not reach it is grown one candidate layer at a time from the
// ask-bounded pick, toward the earlier tail layers, and stops as soon as the
// margin holds. That is the cheap side the operator chose: the chunk and the
// depth rate stay, the refill pays ~0.02 s a layer at the measured 11.5 GiB/s.
// The tail bounds the growth; when it runs out the pick is returned with `why`
// set, carrying the full grown set. Round 25: `extend_lend_toward` commits that
// set when the loan's own need still fits, so a tail that cannot reach a carried
// shortfall's demand is never the reason a runnable loan is refused.
//
// Round 19: a layer the growth lends frees its slot bytes on the card, so the
// card's resident figure falls with it. `card.lend` holds the freed bytes the
// pick does not account for (other caches on the same card); the pick's own
// bytes are added below, so each added layer lowers the card's resident.
inline LendPick extend_lend_layers(const std::vector<LendCandidate> & cand,
                                   const LendPick & ask, size_t need, size_t beside = 0,
                                   const CardFacts & card = CardFacts{}) {
    LendPick p = ask;
    auto fits = [&](const LendPick & q) {
        const size_t lend = beside + (size_t) q.bytes;
        if (!size_fits(need, lend)) return false;
        if (card.total == 0) return true;
        // the growth frees the added layers' bytes on the card
        const size_t after_lend = card.lend + (size_t) q.bytes;
        CardFacts c = card;
        c.lend      = after_lend;
        return card_fits(c, 0);
    };
    if (fits(p)) {
        p.why = nullptr;
        return p;
    }
    bool any = false;
    for (int i = (int) cand.size() - 1; i >= 0; --i) {
        if (!cand[(size_t) i].lendable) {
            continue;
        }
        if (std::find(p.chosen.begin(), p.chosen.end(), i) != p.chosen.end()) {
            continue;   // the ask already chose it
        }
        const int64_t b = cand[(size_t) i].bytes;
        if (b <= 0) {
            continue;
        }
        any = true;
        p.chosen.push_back(i);
        p.bytes += b;
        if (fits(p)) {
            std::sort(p.chosen.begin(), p.chosen.end());
            p.why = nullptr;
            return p;
        }
    }
    std::sort(p.chosen.begin(), p.chosen.end());
    p.why = any ? "the lendable tail does not reach the fixed margin"
                : "no tail layer is left to grow the lend";
    return p;
}

// Round 25: the growth in two steps. `demand` is what the lend should reach --
// the prompt's `need`, or that plus the carried shortfall -- and `need` is the
// loan's own rule, the floor a partial growth must keep. Step 1 asks the tail
// for the demand. When the tail cannot reach it, step 2 keeps the tail's full
// growth (as far as the tail allows toward the demand) if it still leaves the
// loan's own `need` clear; the card copy carries `need`, not the demand. So the
// shortfall never turns a loan that the `need` rule allows into a refusal: only
// the `need` rule refuses, and it refuses the same loan it would refuse without
// any carried shortfall.
struct LendGrowth {
    LendPick pick;
    bool     reached_demand = false;   // false: the tail's full growth, taken at the need rule
};

inline LendGrowth extend_lend_toward(const std::vector<LendCandidate> & cand, const LendPick & ask,
                                     size_t demand, size_t need, size_t beside = 0,
                                     const CardFacts & card = CardFacts{}) {
    LendGrowth g;
    g.pick = extend_lend_layers(cand, ask, demand, beside, card);
    if (g.pick.why == nullptr) {
        g.reached_demand = true;
        return g;
    }
    // step 2: the tail grew as far as it allows. Keep it when `need` alone is
    // still clear; otherwise nothing grows and the caller's `need` checks decide.
    CardFacts own = card;
    own.need = need;
    if (!size_fits(need, beside + (size_t) g.pick.bytes) ||
        !card_fits(own, (size_t) g.pick.bytes)) {
        g.pick = ask;
        g.pick.why = "the demand is beyond the tail and the loan's own need does not fit";
        return g;
    }
    g.pick.why = nullptr;   // step 2 commits the tail's full growth at the need rule
    return g;
}

// The poison policy, operator's decision 2026-10-29 ("poison-exit"): a poisoned
// or broken loan has no safe serving state -- a lent slot may be read as an
// expert, a refill's bytes are not the file's, or the resident reserve is gone.
// Log one ERROR line naming the nature and exit non-zero, so the unit manager
// restarts a clean process. Keeping the unit up and refusing every request is
// the alternative; that is this body's one line (drop the exit). It is
// called from the context's reserve paths and the cache's refill/undo paths,
// and a device-free subprocess test proves the exit (tests/test_loan_book.cpp).
//
// Fifth review, finding 2: std::exit runs static destructors from the generation
// thread (under arcint's mutex, possibly while the request unwinds), and those
// close the OpenCL device contexts and the CPU-miss coordinator while HTTP
// threads live and the device may be busy: a hang, which defeats the policy (the
// unit only restarts on an exit). std::_Exit skips every destructor, the exit is
// the policy. The line goes out twice on purpose: through the llama log callback
// first (the unit's own log routes libllama's lines; log_line is defined in
// llama-expert-cache.cpp, which owns the LLAMA_LOG_ERROR call) and then on
// stderr with a flush, so it survives whatever the callback is. Both take their
// own log lock and no device, so they are safe here.
void log_line(const char * text);
// The same route at WARN, for a line the unit's log shows without the process
// being poisoned (round 12's exit-idle line): arcint logs WARN and above.
void log_warn(const char * text);

// Round 8, drain-before-exit: std::_Exit tears the process's address space down
// while kernels and copies are still queued (the card's xen: page fault and a
// GuC reset of the compute and copy engines on both exit-70 arms, 2026-10-29).
// The backend's owner registers a function that finishes every queue the
// process has work on (compute, adapt, the bank stream's copy queue); the
// policy runs it before _Exit. The watchdog is bounded: a detached thread exits
// 70 after ~5 s whatever the drain does, so a hung driver cannot hold the unit.
//
// Round 12: the drain empties the queues, it does not stop the runtime's
// direct-submission rings. Those are a hardware context an engine (the compute
// and the copy engines) spins on a semaphore, stopped by the runtime's own
// controller after a device idle period or by its destructor -- and std::_Exit
// runs no destructor. Measured-here (round 11, the B60): an exit microseconds
// after the last GPU work faults the page of a live ring (xe -ENOENT, GuC reset
// 3 of 3) while an exit hundreds of ms after it does not (MUTATE=1, the refill
// verify's read-back and memcmp). So the policy waits, bounded, before _Exit;
// the idle is exit_idle_ms below (round 12 §20).
using DrainFn = void (*)(void);
inline DrainFn & drain_hook() {
    static DrainFn fn = nullptr;   // registered once, before any request
    return fn;
}
inline void register_drain(DrainFn fn) {
    drain_hook() = fn;
}

// Round 10, card finding 1: a buffer free must never follow queued work that
// still addresses it. The tail slot buffer is freed at the lend and every queue
// the process uses (compute, the adapt copies, the bank-stream copies) may still
// hold a command over it -- the exit-70 drain then executes that command against
// a released object (the B60's xe -ENOENT and GuC reset). Any path that frees the
// tail buffer drains the queues first through this accessor.
inline void drain_now() {
    if (DrainFn drain = drain_hook()) {
        drain();
    }
}

// Round 12, the exit's bounded idle after the drain. Nothing is queued at that
// point, so the only thing the wait buys is time in which the runtime's
// direct-submission controller (a thread of its own, not an OpenCL queue) can
// notice the idle engine and stop its rings; _Exit then closes the fd with no
// hardware context spinning. The knob is LLAMA_SLOT_LOAN_EXIT_IDLE_MS.
//
// The default is 2,000 ms, chosen here:
// - the one exit path measured to pass (MUTATE=1, `measured-here`, round 11)
//   is idle for the refill verify's read-back plus memcmp -- hundreds of ms on
//   the card, ~0.9-1.1 s estimated for 12,480 rows in the record. 2,000 ms is
//   twice that upper bound, not the passing value itself.
// - the runtime's own idle period is not readable here (compute-runtime is not
//   checked out in this tree and the machine has no network), so the default is
//   a margin over the measured passing idle rather than the controller's
//   constant; the knob exists for the window that tests it.
// - the watchdog bound is 5 s (see poison_exit). 2,000 ms leaves 3 s for the
//   drain, and no setting above 4,000 ms is accepted, so a wait can never eat
//   the bound a hung driver needs.
// "0", "off", "false" and "no" are the red: no idle, the round-11 form. Round
// 13: every case of those words counts (OFF / False / NO), since the knob was
// case-sensitive and an OFF arm silently took the 2,000 ms default -- a red arm
// that ran far slower than the arm it was asked to be. A typo or a negative is
// the default -- never silently 0, which would turn the red arm's own knob into
// the healthy one.
constexpr int kWatchdogSeconds   = 5;
constexpr int kWatchdogBoundMs   = kWatchdogSeconds * 1000;
constexpr int kExitIdleMaxMs     = kWatchdogBoundMs - 1000;   // 1 s of the bound left for the drain
constexpr int kDefaultExitIdleMs = 2000;

// the off words, compared without case: "off" is the documented one, "false"
// and "no" are the accepted spellings
inline bool exit_idle_off_word(const char * e) {
    return off_word(e);   // round 22: one predicate for every knob's off words
}
inline int exit_idle_ms(const char * e) {
    if (e == nullptr || e[0] == '\0') {
        return kDefaultExitIdleMs;
    }
    if (exit_idle_off_word(e)) {
        return 0;
    }
    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || end == nullptr || *end != '\0' || v < 0) {
        return kDefaultExitIdleMs;   // a typo is the default, never a silent 0
    }
    return v > kExitIdleMaxMs ? kExitIdleMaxMs : (int) v;
}

// Round 9, review item 4: std::thread's constructor can throw std::system_error
// (thread resources exhausted). Uncaught from the poison policy it reaches
// std::terminate, so the process does not exit 70 and the unit's restart policy
// sees a crash. The spawn is guarded; a failed watchdog is best-effort (the drain
// still runs and _Exit(70) follows).
template <typename Spawn>
inline bool watchdog_spawn(Spawn spawn) {
    try {
        spawn();
        return true;
    } catch (...) {
        return false;
    }
}

inline void poison_exit(const char * why) {
    char line[1024];
    std::snprintf(line, sizeof(line),
                  "ERROR llama: the prefill-slot-loan's state is unusable (%s); exiting non-zero so the unit restarts clean\n", why);
    log_line(line);
    std::fputs(line, stderr);
    std::fflush(stderr);
    if (DrainFn drain = drain_hook()) {
        // the drain is bounded: if the driver's finish hangs, the watchdog exits.
        // The watchdog's construction is guarded (watchdog_spawn): no watchdog, the
        // drain still runs and _Exit(70) follows.
        llama_loan_book::watchdog_spawn([] {
            std::thread([] {
                std::this_thread::sleep_for(std::chrono::seconds(kWatchdogSeconds));
                // Round 13, review low: the watchdog cut the idle silently, so a
                // log looked like a policy exit that had finished its wait. One
                // line, ONE write(2) of a constant string: async-signal-safe (no
                // snprintf, no log callback, no flush) -- the drain may be hung and
                // the log route runs on the lock the hung thread holds.
                static const char fired[] =
                    "ggml_opencl: the exit's watchdog fired after 5000 ms (the drain or the idle did not finish); _Exit(70)\n";
                (void) !::write(STDERR_FILENO, fired, sizeof(fired) - 1);
                std::_Exit(70);
            }).detach();
        });
        drain();
        // Round 12: the queues are empty, the rings are not stopped. Wait, under
        // the same watchdog, long enough for the runtime's direct-submission
        // controller to stop them (a device idle period, not an OpenCL queue), so
        // _Exit closes the fd with no hardware context spinning. No device was
        // drained (no hook) means no ring either: no wait.
        const int idle = exit_idle_ms(std::getenv("LLAMA_SLOT_LOAN_EXIT_IDLE_MS"));
        char wait[256];
        std::snprintf(wait, sizeof(wait),
                      "ggml_opencl: the exit waits %d ms after the drain, for the runtime's direct-submission "
                      "rings to stop (LLAMA_SLOT_LOAN_EXIT_IDLE_MS, default %d)\n", idle, kDefaultExitIdleMs);
        log_warn(wait);
        std::fputs(wait, stderr);
        std::fflush(stderr);
        if (idle > 0) {
            // Round 13: the wait line above is written BEFORE the sleep, so it says
            // what was asked for, not what happened; the watchdog could also cut
            // the idle with no line at all. The second line carries the measured
            // elapsed, so the log shows whether the wait ran to its value.
            const auto t0 = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(idle));
            const long elapsed = (long) std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::steady_clock::now() - t0).count();
            char done[256];
            std::snprintf(done, sizeof(done),
                          "ggml_opencl: the exit's idle finished after %ld ms (asked %d ms, LLAMA_SLOT_LOAN_EXIT_IDLE_MS)\n",
                          elapsed, idle);
            log_warn(done);
            std::fputs(done, stderr);
            std::fflush(stderr);
        }
    }
    std::_Exit(70);
}

// The decode ubatch's cap and its cell budget, so the loan's resident reserve
// (2048) and the prompt's reserve (the chunk) are the same arithmetic and the
// loan-off path is byte-identical to before (ub_cap == n_ubatch then).
inline uint32_t ubatch_cap(uint32_t n_ubatch, uint32_t reserve) {
    return std::min(n_ubatch, reserve);
}

inline uint64_t cell_budget(uint32_t ub_cap, uint32_t n_kv_reserve) {
    return (uint64_t) ub_cap * n_kv_reserve;
}

// Round 7, the refill's check (§15 of the design): the refill writes the lend
// bank's bytes into the re-allocated slots, so the default check compares every
// row of every lent tensor with the lend bank in the CARD'S layout -- byte for
// byte, no conversion, no read of the GGUF. The bank's provenance to the file is
// the load's lend bank check (every lendable row against a read of the file) and
// every adapt's mirror, which copies bank bytes that are the file's. "file" is the
// campaign's gate form: the GGUF bytes of the slots against a fresh read of the
// file (a planar conversion and a 6.2 GiB file read a loan, hence not the
// default). "0"/"off" turns it off for timing arms. An unknown word is the
// default, never silent off (a typo in a gate arm must not turn the check off).
enum class VerifyRef { Off, Bank, File };
inline VerifyRef verify_ref(const char * e) {
    if (e == nullptr || e[0] == '\0') {
        return VerifyRef::Bank;
    }
    if (std::atoi(e) == 0) {
        if (strcmp(e, "0") == 0 || strcmp(e, "off") == 0) {
            return VerifyRef::Off;
        }
        if (strcmp(e, "file") == 0) {
            return VerifyRef::File;
        }
    }
    return VerifyRef::Bank;
}

// Round 10, card finding 2: round 8's red case 3 skipped the lent flag in the
// sizing, on the premise that the lent graph has one MORE MUL_MAT_ID dst than the
// resident shape. The code says the opposite: the lent branch REPLACES the slot
// branch with the lend bank (llama-graph.cpp:2510-2521) and drops both down tables,
// so the non-lent measurement carries MORE tensors than the lent reserve. Skipping
// the lent measurement therefore RAISES `need`, and `actual <= expect` passes --
// the card's MUTATE=3 lent, refilled and returned normally (`measured-here`,
// round 8). The red case now measures the graph at the resident chunk while the
// reserve allocates the prompt's chunk, which under-counts `need` for real -- the
// defect the check exists for, made deterministic.
inline bool measure_lent(int mutate) {
    return mutate != 3;
}
inline uint32_t measure_tokens(int mutate, uint32_t cap, uint32_t resident) {
    return mutate == 3 ? resident : cap;   // MUTATE=3: sized for the resident chunk, reserved at cap
}

// Round 14, the KL gate's red case (docs/campaigns/prefill-slot-loan.md):
// every red arm before this one exits 70 before it scores, and MUTATE=1 with
// VERIFY=off is a FALSE red -- its corruption lands after the prefill, so the
// prefill's logits (what the KL reads) are the healthy ones. MUTATE=5 corrupts
// the thing the lent branch itself reads: one lent layer's lend-bank row index,
// off by one in the table the card uses during the prefill. Nothing verifies a
// table row (the refill's check compares slot bytes with the bank's), so the
// run continues, refills and scores -- and the score must fail the bar.
// The shift stays inside the lend bank: row + 1 when it exists, else row - 1;
// a -1 entry (an expert with no lent row, the kernels' skip) is left alone.
inline int32_t mutate_loan_bank_row(int mutate, int32_t row, int32_t n_slots) {
    if (mutate != 5 || row < 0) return row;
    if (row + 1 < n_slots) return row + 1;
    if (row > 0) return row - 1;
    return row;   // a one-row bank has no neighbour to move to
}

// Round 9, review item 1: round 8's margin read ggml_backend_dev_memory, whose
// OpenCL implementation (ggml-opencl.cpp's ggml_backend_opencl_device_get_memory)
// returns `free = global_mem_size - 1 GiB` -- a constant, so a headroom check
// against it could never fail. The rule is over real plan bytes: the prompt's
// compute buffers need at most the freed lend bytes minus the margins. The lend
// bytes are the VRAM the tail buffers actually released; no device free figure is
// consulted (OpenCL has none that is reliable).
//
// Round 17, the fixed margin (operator's decision 2026-10-30, "Margin ok"): the
// 42 MiB the depth arm left under the card's total is not enough headroom for the
// allocations beside the plan (the ring's first touch, the refill's tail
// re-allocation, the first decode's buffers), and the measured eviction cases
// put the floor at 768 MiB. The operator set a fixed 1 GiB under the card's
// total beside the 384 MiB peak reserve, so the lend's own sanity is
// `lend - need >= kPeakReserve + kCardMargin` (1.375 GiB). Round 19 keeps this
// check as the reserve's own sanity and puts the card margin against the card's
// total in card_fits, above.
inline bool size_fits(size_t need, size_t lend) {
    return lend >= kLendMargin && need <= lend - kLendMargin;   // need + both margins <= the freed bytes
}

// The card's bound as a free function: c.resident after c.lend, plus c.need and
// the cold ring's first touch, at or below c.total - c.margin. An unknown total
// (0) means the rule is not checked; a failed sizing (SIZE_MAX) never fits.
inline bool card_fits(size_t resident, size_t need, size_t card_total, size_t margin, size_t lend, size_t first_touch) {
    if (card_total == 0) return true;   // no card figure: the lend-vs-need rule alone
    if (need == SIZE_MAX || first_touch == SIZE_MAX) return false;
    const size_t limit  = card_total >= margin ? card_total - margin : 0;
    const size_t after  = lend >= resident ? 0 : resident - lend;
    if (need > limit) return false;                   // even an empty resident does not fit
    if (first_touch > limit - need) return false;     // the cold ring alone over the limit
    return after <= limit - need - first_touch;       // no overflow: compared by subtraction
}

// Round 21: the card rule's dispatch. With the kernel query the left side is
// the device-wide `used` minus the lend, and the margin beyond the card's total
// is zero -- `used` carries every other client and the kernel's objects already
// (round 22; the round-21 form subtracted them a second time). The prefill
// growth allowance is the only addend beside `need` and `first_touch`. Without
// the query the round-20 fdinfo + fixed margin form stays as the fallback (the
// operator's zero margin is only safe with the kernel's figure).
inline bool card_fits(const CardFacts & f, size_t lend_extra) {
    const size_t lend = f.lend + lend_extra;
    if (f.query_ok) {
        return card_state_fits(f.used, f.total, lend, f.need, f.first_touch, f.allowance);
    }
    return card_fits(f.resident, f.need, f.total, f.margin, lend, f.first_touch);
}

// Round 8, review item 2 / round 9, review item 2: the reserve is deterministic
// when its actual bytes are compared with the sizing's expectation. MUTATE=3 skips
// the lent measurement (need is under-counted) while lend() still sets the lent
// flags the reserve builds its graph from, so the actual sum is the healthy one.
// The comparison is `actual <= expect`, not equality: sched_reserve_size and
// ggml_backend_sched_get_buffer_size count a buffer shared between bufts
// differently (the same bytes can appear once or twice), so a healthy reserve can
// legitimately sum BELOW the expectation. Only actual > expectation is the defect
// -- the plan allocated more than it was sized for (MUTATE=3's red case).
inline bool reserve_matches(size_t expect, size_t actual) {
    return actual <= expect;
}

// Round 8, review item 3: a region copy is exact only where the lend bank's
// bytes and the card slot's layout are identical. The USM bank stores IQ4_NL
// split (ggml_cl_usm_set_tensor) and the planar types as planes, everything else
// raw. The card's set_tensor instead builds a specialised extra for Q1_0, Q4_0,
// Q4_1, Q5_0, Q5_1, MXFP4, Q8_0, IQ4_NL, Q4_K, Q5_K, Q6_K and converts BF16 to
// f16. For those, a region copy would write GGUF bytes through a plain extra
// (the old crash) or feed f16 kernels raw BF16 (which the bank check would pass,
// because it compares the wrong bytes with themselves). The plain-no-extra types
// are the GGUF bytes on both sides, the planar set is planes on both, IQ4_NL is
// split on both; Q8_0 is not lendable anyway (its bank is the slot buffer).
inline bool region_copyable(ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_IQ3_S:
            return true;
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_BF16:
            return false;   // set_tensor builds an extra or converts; the bank is raw
        default:
            return true;    // plain: the same GGUF bytes on both sides
    }
}

// Finding 6 (third review): init wrote and verified the zero slot too -- row
// n_slots of every slot tensor holds zeros, and a bank expert's slot-branch row
// reads it. drop_tail_buffer() leaves the new tail buffer's contents undefined,
// so the refill restores that row as well and the check covers it.
inline int64_t refill_zero_row(int64_t n_slots) {
    return n_slots;   // the row no lend-bank row holds
}

inline int64_t refill_check_rows(int64_t n_slots) {
    return n_slots + 1;   // rows 0 .. n_slots, inclusive
}

// Round 9, review item 3: the refill's bank check counts rows, not tensors.
// ggml_backend_opencl_lend_read emits a tensor's regions one after the other
// ([d x n][q x n] for IQ4_NL, one region for the others), each region's n rows
// contiguous with a per-region row stride. The check therefore compares row by
// row INSIDE each region: `bad` is a row count (the same unit as `n`) and the
// first mismatch names its region and row. Round 8 added one to `bad` per tensor
// while `n` counted rows and the line called both "rows".
struct LendRegionBytes { int nr; size_t per[2]; };
inline LendRegionBytes lend_region_bytes(ggml_type t, size_t per_slice) {
    LendRegionBytes b{1, {per_slice, 0}};
    if (t == GGML_TYPE_IQ4_NL) {
        const size_t blk = ggml_type_size(t), qb = blk - sizeof(ggml_fp16_t);
        b.nr = 2;
        b.per[0] = per_slice / blk * sizeof(ggml_fp16_t);   // scales (d)
        b.per[1] = per_slice / blk * qb;                    // quants (q)
    }
    return b;
}
inline const char * lend_region_name(ggml_type t, int region) {
    if (t == GGML_TYPE_IQ4_NL) return region == 0 ? "scales" : "quants";
    return "row";
}
struct LendRowDiff { int bad = 0; int rows = 0; int first_region = -1; int first_row = -1; };
// slot: the read image with n_slots + 1 rows a region (the zero slot included);
// bank: the lend bank's image with n_slots rows a region. Both region-major.
inline LendRowDiff lend_compare_rows(const uint8_t * slot, const uint8_t * bank, int n_slots,
                                     ggml_type t, size_t per_slice) {
    LendRowDiff d;
    const LendRegionBytes b = lend_region_bytes(t, per_slice);
    size_t sofs = 0, bofs = 0;   // a region follows the previous one, not a fixed stride
    for (int r = 0; r < b.nr; ++r) {
        const size_t per = b.per[r];
        const uint8_t * sd = slot + sofs;
        const uint8_t * bd = bank + bofs;
        for (int s = 0; s < n_slots; ++s) {
            ++d.rows;
            if (std::memcmp(sd + (size_t) s * per, bd + (size_t) s * per, per) != 0) {
                ++d.bad;
                if (d.first_row < 0) { d.first_region = r; d.first_row = s; }
            }
        }
        ++d.rows;   // the zero slot's row in this region
        bool zero = true;
        for (size_t i = 0; i < per; ++i) {
            if (sd[(size_t) n_slots * per + i] != 0) { zero = false; break; }
        }
        if (!zero) {
            ++d.bad;
            if (d.first_row < 0) { d.first_region = r; d.first_row = n_slots; }
        }
        sofs += ((size_t) n_slots + 1) * per;
        bofs += (size_t) n_slots * per;
    }
    return d;
}

// Finding 1 (third review): an adapt's two phases are planned before a graph
// (adapt: the spares popped, the victims named; progress: the phase-1 tables
// applied, the phase 2 copies due) and their copies are issued after_graph. A
// graph that fails or is aborted returns before after_graph, so the next loan's
// finish_pending() met a planned, unissued phase. Clearing it there leaks the
// phase 1 spares (adapt silently stops on a later layer) and, for phase 2,
// orphans a slot: expert_of[s] is -1, the victim sits in spare x, the lend bank
// row s still holds it, the refill copies it back and the verify compares
// against zeros -- the loan sticks out and every request is refused. So a
// planned phase is never cleared: it is issued (fence first, the copies, the
// record) and then settled, exactly as a completed graph would have done.
//
// Finding 2 (third review): the copies go on the adapt queue, which nothing
// orders against the compute queue but a fence; the settle therefore places one
// before it issues a phase the graph path did not fence, and again between the
// phase-1 tables and the phase-2 copies (progress does the same).
//
// Finding 3 (third review): the lend bank's mirror copies a swap's three
// tensors in one call, so it sits OUTSIDE the k loop -- 3x the bytes otherwise.
//
// The types are the cache's own: `SwapT` is its pending_swap, `OpsT` whatever
// provides the callbacks below (the cache's own, or the test's data-only ones).
template <typename SwapT>
struct PhaseState {
    std::vector<SwapT> pend;
    int  phase  = 0;      // 0 none, 1 victims out, 2 newcomers in
    int  ticket = -1;     // the adapt queue's mark for the issued copies
    bool issued = false;  // the phase's copies are enqueued
    bool check  = false;  // the first adapt's swap: its bytes against the file once landed
};

// OpsT needs: fence(), record() -> int, finish(), done() -> bool,
// copy_out(swap, k), copy_in(swap, k), mirror(swap),
// tables_out(pend), tables_in(pend).
template <typename SwapT, typename OpsT>
inline void issue_phase(PhaseState<SwapT> & b, OpsT & o, bool fence) {
    if (b.phase == 0) {
        return;
    }
    if (fence) {
        o.fence();   // the copies wait for every compute command so far
    }
    for (auto & p : b.pend) {
        if (b.phase == 1) {
            for (int k = 0; k < 3; ++k) {
                o.copy_out(p, k);   // the victim out of its slot into its spare
            }
        } else {
            o.mirror(p);            // once a swap, all three k (finding 3)
            for (int k = 0; k < 3; ++k) {
                o.copy_in(p, k);    // the newcomer from its bank entry into the slot
            }
        }
    }
    b.ticket = o.record();
    b.issued = true;
}

// A phase that was planned but not issued is issued first, behind a fence of its
// own; then the settle runs the copies, applies both table halves and clears the
// plan only once both are applied.
template <typename SwapT, typename OpsT>
inline void settle(PhaseState<SwapT> & b, OpsT & o) {
    if (b.phase == 0) {
        return;
    }
    if (!b.issued) {
        issue_phase(b, o, true);
    }
    o.finish();   // phase 1: the victims are in their spares; phase 2: the newcomers in their slots
    if (b.phase == 1) {
        o.tables_out(b.pend);
        o.fence();   // finding 2: the phase-2 copies wait for the graphs still reading the victims
        b.phase  = 2;
        b.issued = false;
        issue_phase(b, o, false);
        o.finish();
        o.tables_in(b.pend);
    } else {
        o.tables_in(b.pend);
    }
    b.pend.clear();
    b.phase  = 0;
    b.issued = false;
    b.ticket = -1;
}

// The poll between ubatches (no wait): the phase's copies landed, so its tables
// change. Phase 1 leaves the plan at phase 2, unissued, behind a fence; phase 2
// applies the newcomer tables and clears. Returns false while the copies are in
// flight (nothing changes).
template <typename SwapT, typename OpsT>
inline bool progress_phase(PhaseState<SwapT> & b, OpsT & o) {
    if (b.phase == 0 || !b.issued) {
        return false;
    }
    if (!o.done()) {
        return false;
    }
    if (b.phase == 1) {
        o.tables_out(b.pend);
        o.fence();   // the copies after it wait for the graphs that still read the victims
        b.phase  = 2;
        b.issued = false;
        b.ticket = -1;
        return true;
    }
    o.tables_in(b.pend);
    o.check_swap();   // the first adapt's swap check (the cache's log)
    b.pend.clear();
    b.phase  = 0;
    b.issued = false;
    b.ticket = -1;
    return true;
}

}  // namespace llama_loan_book
