#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace lgc {

// The spill guard: a start whose device-resident demand plus the growth a request adds beyond
// the full prediction does not fit the card is refused. Pure, so the rule and its message are
// unit-tested with no device; the backend supplies libllama's accounting and the driver's card
// figure.
//
// THE RULE, from the card's calibration points (`measured-here`, B60, xe's region total
// 24,480 MiB, not OpenCL's 23,256). The full prediction is the target context's breakdown by
// kind plus the draft context's own breakdown (the in-model MTP context at the agent's
// 262,144: KV 416 + reserve/compute 316 + head rows 172 = 904 MiB), the expert cache's own
// on-card bytes, this process's fdinfo bytes above that demand, and the other clients' share.
// The growth beyond it at measured peaks: agent 24,196 - (22,998 + 904) = 294 MiB; Inkling
// with --llama-swa-window 22,031 - 21,800 = 231; Flash-Next after its first request 26,306 -
// 26,122 = 184 and 26,294 - 26,122 = 172 (two measured legs); Inkling 65,536 full SWA
// 23,686 - 23,460 = 226. The worst is 294; the headroom
// is 128 MiB, because a 512 MiB headroom would refuse the agent, a config measured to run.
// Constant, not environment: a start rule's number is not a knob.
inline constexpr size_t kSpillGrowthHeadroomBytes = 128ull << 20;
inline constexpr size_t kSpillGrowthMarginBytes = 422ull << 20;   // 294 worst + 128 headroom


// The flag unit is MiB; the cache projection takes bytes. This is the one conversion, so a
// caller cannot shift twice.
inline constexpr size_t cache_flag_bytes(size_t flag_mib) { return flag_mib << 20; }

// Device-resident demand by kind: libllama's memory breakdown for the target (device buffer
// types only), the expert cache's own on-card figure (its slots, Q8_0 banks, gather mirror and
// loan tail are allocated outside the breakdown), the draft model/context/head, deliberate
// host-resident bytes (never a refusal reason: Flash-Next's 40.94 GiB pinned expert bank is
// by design), the card's total, and what other clients and the kernel hold. `other_clients` is
// the probe's used - fdinfo_vram0 when both are figures. `process_vram0` is this process's
// residency; bytes it holds above the known demand (a projector, lazily grown scratch, the KV
// stream, anything the grep list misses) are the `unlisted` term, so they count too.
struct SpillFacts {
    size_t weights       = 0;   // device weights
    size_t context       = 0;   // device KV and recurrent context
    size_t compute       = 0;   // device compute buffers
    size_t cache_device  = 0;   // the expert cache's on-card bytes (slots, Q8_0 banks, gather mirror, loan tail)
    size_t cache_flag_mib = 0;  // the --llama-expert-cache value that produced cache_device (0: no cache)
    // the cache's real accounting: flag MiB -> on-card bytes, no allocation (the llama
    // projection). The argument is the flag's MiB, not bytes; the backend shifts once.
    std::function<size_t(size_t)> cache_project;
    size_t draft_device  = 0;   // the MTP/DFlash draft model, its context and the draft head rows
    size_t host_by_design = 0;  // host buffers + pinned expert banks; logged, not summed
    size_t card_total     = 0;
    bool   card_from_driver = false;   // xe's region total (true) or OpenCL's global size
    size_t process_vram0  = 0;   // this process's fdinfo residency on the card; 0 when not known
    size_t other_clients  = 0;   // used - process vram0; 0 when not known
    size_t margin         = kSpillGrowthMarginBytes;
    int    n_ctx          = 0;   // the context being started
    bool   allow_spill    = false;   // --llama-allow-spill: warn, do not refuse
};

struct SpillDecision {
    bool        fits   = true;   // accounted <= card_total
    bool        refuse = false;  // !fits and not allow_spill
    size_t      accounted = 0;    // weights + context + compute + cache + draft + unlisted + other + margin
    int         fit_n_ctx = 0;    // an --n-ctx whose context share fits the headroom, 0 if none
    size_t      fit_cache = 0;    // the free on-card bytes the cache could take, 0 if none
    size_t      fit_cache_flag_mib = 0;  // the --llama-expert-cache flag value whose projection fits
    size_t      unlisted  = 0;    // process device bytes fdinfo holds above the known demand
    std::string text;             // the start line or the refusal, naming every part
};

// The decision and its one message. No device; ARCINT_GUARD_TRACE=1 logs the cache search's probes.
SpillDecision spill_guard(const SpillFacts& facts);

}  // namespace lgc
