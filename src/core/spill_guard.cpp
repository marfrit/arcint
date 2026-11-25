#include "core/spill_guard.h"

#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace lgc {
namespace {
constexpr size_t kMiB = 1024ull * 1024ull;

double mib(size_t bytes) { return static_cast<double>(bytes) / static_cast<double>(kMiB); }

std::string fmt(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

// The free on-card bytes the expert cache could take beside the fixed terms.
size_t fit_cache(const SpillFacts& f, size_t fixed) {
    if (fixed >= f.card_total) return 0;
    return f.card_total - fixed;
}

// The context's share is the only part proportional to cells at load; the weights, compute
// buffers and other clients are not. The cell count is padded to llama.cpp's 256-cell cache
// granularity, never above the context being started.
int fit_n_ctx(const SpillFacts& f, size_t fixed) {
    if (f.n_ctx <= 0) return 0;
    if (fixed >= f.card_total) return 0;
    const size_t headroom = f.card_total - fixed;
    if (f.context == 0) return f.n_ctx;
    const double raw = static_cast<double>(f.n_ctx) * static_cast<double>(headroom) /
                       static_cast<double>(f.context);
    int n = static_cast<int>(std::floor(raw));
    if (n <= 0) return 0;
    n = (n / 256) * 256;
    n = std::min(n, f.n_ctx);
    // the linear estimate is not the decision: the context section rounds and the margin
    // stands, so the suggestion is verified against the same budget before it is printed.
    while (n > 0) {
        const size_t ctx = static_cast<size_t>(f.context) * static_cast<size_t>(n) /
                           static_cast<size_t>(f.n_ctx);
        if (fixed + ctx <= f.card_total) return n;
        n -= 256;
    }
    return 0;
}
}  // namespace

SpillDecision spill_guard(const SpillFacts& f) {
    SpillDecision d;
    const size_t known = f.weights + f.context + f.compute + f.cache_device + f.draft_device;
    // fdinfo is this process's own residency on the card. Bytes it holds above the known
    // demand are other device allocations this process made (a projector, lazily grown
    // scratch, the KV stream): they count, and if the virtual breakdown is larger than the
    // resident figure (the calibration's 21,732 against 21,334) the gap is not subtracted.
    d.unlisted = f.process_vram0 > known ? f.process_vram0 - known : 0;
    const size_t fixed = f.weights + f.compute + f.cache_device + f.draft_device + d.unlisted +
                         f.other_clients + f.margin;
    d.accounted        = fixed + f.context;
    d.fits             = d.accounted <= f.card_total;
    d.refuse           = !d.fits && !f.allow_spill;
    d.fit_n_ctx        = fit_n_ctx(f, fixed);
    // the context is outside `fixed`: a cache suggestion must leave room for it too
    const size_t free_cache = f.cache_device > 0 ? fit_cache(f, fixed - f.cache_device + f.context) : 0;
    d.fit_cache = free_cache;
    // the flag->on-card conversion is not monotonic over the whole range: a Q8_0 bank holds
    // every expert at a low budget while slots grow with it, so a plain bisection can miss the
    // fit band. Walk down from the current flag in coarse steps, then refine upward from the
    // first fit; each step calls the cache's own projection.
    if (!d.fits && f.cache_project && f.cache_flag_mib > 0 && free_cache > 0) {
        const bool trace = [] { const char* e = getenv("ARCINT_GUARD_TRACE"); return e != nullptr && atoi(e) != 0; }();
        auto probe = [&](size_t flag) {
            const size_t p = f.cache_project(flag);
            if (trace)
                log::warn("guard", "cache probe: flag %zu MiB -> projected %.0f MiB, free %.0f MiB: %s",
                          flag, static_cast<double>(p) / (1024.0 * 1024.0),
                          static_cast<double>(free_cache) / (1024.0 * 1024.0), p <= free_cache ? "fits" : "over");
            return p;
        };
        const size_t hi   = f.cache_flag_mib;
        const size_t step = std::max<size_t>(1, hi / 256);
        size_t found = SIZE_MAX;
        for (size_t flag = hi; ; ) {
            if (probe(flag) <= free_cache) { found = flag; break; }
            if (flag < step) break;
            flag -= step;
        }
        if (found == SIZE_MAX && probe(0) <= free_cache) found = 0;
        if (found != SIZE_MAX) {
            size_t best = found;
            const size_t up = std::min(hi, found + step);
            for (size_t flag = found + 1; flag <= up; ++flag) {
                if (probe(flag) <= free_cache) best = flag;
                else break;   // from the fit's lower edge the slot growth is monotonic
            }
            d.fit_cache_flag_mib = best;
        }
    } else {
        d.fit_cache_flag_mib = 0;
    }

    const char* source = f.card_from_driver ? "driver" : "OpenCL";
    const std::string pct = f.card_total == 0
                               ? std::string("an unknown card")
                               : fmt("%.1f %% of the card", 100.0 * static_cast<double>(d.accounted) /
                                                            static_cast<double>(f.card_total));
    std::string suggestions;
    if (!d.fits) {
        const std::string nctx =
            d.fit_n_ctx > 0 && d.fit_n_ctx < f.n_ctx
                ? fmt("--n-ctx %d would fit (down from %d)", d.fit_n_ctx, f.n_ctx)
                : std::string("no --n-ctx fits: weights, compute, the cache and other clients already exceed the budget");
        const std::string cache =
            d.fit_cache_flag_mib > 0
                ? fmt("--llama-expert-cache about %zu MiB would fit (down from %zu MiB)",
                      d.fit_cache_flag_mib, f.cache_flag_mib)
                : std::string("no expert cache fits beside this start");
        suggestions = "; " + nctx;
        if (f.cache_device > 0) suggestions += "; " + cache;
    }
    const std::string head =
        fmt("spill guard: weights %.0f MiB, context %.0f MiB, compute %.0f MiB, expert cache %.0f MiB, "
            "draft/head %.0f MiB, process device outside the breakdown %.0f MiB, other clients %.0f MiB, "
            "growth margin %.0f MiB = %.0f MiB of the card's %.0f MiB (%s, %s); host by design %.0f MiB "
            "(not counted)%s",
            mib(f.weights), mib(f.context), mib(f.compute), mib(f.cache_device), mib(f.draft_device),
            mib(d.unlisted), mib(f.other_clients), mib(f.margin), mib(d.accounted), mib(f.card_total), source,
            pct.c_str(), mib(f.host_by_design), suggestions.c_str());
    if (d.fits) {
        d.text = head + "; start fits";
    } else if (f.allow_spill) {
        d.text = head + "; start does NOT fit, allowed by --llama-allow-spill (it may spill to GTT)";
    } else {
        d.text = head + "; start refused (pass --llama-allow-spill to run it anyway)";
        // Without the xe driver's figure the budget is OpenCL's global memory size, which can be
        // below the card (the B60: 23,256 against 24,480 MiB) and refuse a start that fits the card.
        if (!f.card_from_driver)
            d.text += "; the card figure is OpenCL's global memory size because the xe driver's was not available "
                      "(an i915 host, a build without drm/xe_drm.h, or several xe clients without a PCI bus id): "
                      "it can be below the card's real size, so this refusal may be the fallback's, not the card's";
    }
    return d;
}

}  // namespace lgc
