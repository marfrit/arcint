#pragma once

// The KV stream's start refusal (docs/campaigns/kv-streaming-design.md, section 4.3
// and increment I5): a start that asks for streaming on a card or a build without
// paged FLASH_ATTN_EXT is refused with a message, never degraded to the CPU
// reference and never silently ignored. The rules are plain data here so a
// device-free unit can drive them; the llama backend probes the card and calls
// the rule before a model loads (src/exec/backend_llama.cpp).
//
// Red switch (a test build only): ARCINT_KV_STREAM_REFUSE_OFF=1 makes the rule
// return no refusal -- the arm the unit must fail on.

#include <cstdlib>
#include <string>

namespace arcint {

struct KvStreamAsk {
    int         resident_mib      = 0;      // --llama-kv-resident
    int         decode_mib        = 0;      // --llama-kv-resident-decode
    bool        has_paged_fa      = false;  // the selected device's supports_op for a paged FLASH_ATTN_EXT
    bool        card_is_b60       = false;  // the lever's card (the A770's link is 1.83 GB/s)
    bool        flash_attn        = false;  // the build resolves flash attention on
    std::string kv_dtype;                   // --llama-kv, e.g. "q8_0:q4_0"
};

// the message a start must be refused with; empty = the flag may proceed
inline std::string kv_stream_refusal(const KvStreamAsk& ask) {
    if (ask.resident_mib <= 0 && ask.decode_mib <= 0) return {};   // off: nothing to check
    if (const char* off = std::getenv("ARCINT_KV_STREAM_REFUSE_OFF"); off != nullptr && off[0] == '1') {
        return {};   // RED: the rule accepts the flag it cannot honour
    }
    if (!ask.flash_attn) {
        return "--llama-kv-resident needs flash attention: the streamed pages are read by the "
               "paged FLASH_ATTN_EXT kernel";
    }
    if (ask.decode_mib > 0) {
        return "--llama-kv-resident-decode: the phase budget is not built in this round's patch "
               "(contrib/llama.cpp 0121b); the single --llama-kv-resident budget serves";
    }
    if (!ask.card_is_b60) {
        return "--llama-kv-resident: KV streaming is served on the 24 GB card only; this device's "
               "link cannot carry the streamed pages per verify round (1.83 GB/s on the 16 GB card)";
    }
    if (!ask.has_paged_fa) {
        return "--llama-kv-resident: this build's OpenCL attention does not take paged K/V (the "
               "paged GQA decode kernel, contrib/llama.cpp patch 0122); refusing rather than "
               "running the pages on the CPU";
    }
    if (ask.kv_dtype != "q8_0:q4_0") {
        return "--llama-kv-resident needs --llama-kv q8_0:q4_0: the paged kernel's K/V pair";
    }
    return {};
}

} // namespace arcint
