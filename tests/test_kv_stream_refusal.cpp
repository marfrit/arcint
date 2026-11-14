// The KV stream's start refusal (docs/campaigns/kv-streaming-design.md section 4.3,
// increment I5): the rules are plain data in src/exec/kv_stream_refusal.h, and this
// cell drives them. The llama backend probes the selected card with one paged
// FLASH_ATTN_EXT node before a model loads and calls the same rule.
//
// Green: the flag is honoured only with flash attention, the B60, a device whose
// supports_op takes paged K/V, and --llama-kv q8_0:q4_0; every other ask is refused
// with a message naming what is missing. No round-1 degradation: the stream never
// falls to the CPU reference.
//
// Red: the header's ARCINT_KV_STREAM_REFUSE_OFF=1 arm (a test build) accepts an ask
// it cannot honour; the refusals below then return empty and this cell fails. The
// red branch sets the variable in main before the cases run.

#include "exec/kv_stream_refusal.h"

#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

#define CHECK(cond, what)                                                              \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            printf("FAIL: %s\n", what);                                                \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

arcint::KvStreamAsk ask_ok() {
    arcint::KvStreamAsk a;
    a.resident_mib = 4096;
    a.decode_mib   = 0;
    a.has_paged_fa = true;
    a.card_is_b60  = true;
    a.flash_attn   = true;
    a.kv_dtype     = "q8_0:q4_0";
    return a;
}

bool contains(const std::string & s, const char * what) {
    return s.find(what) != std::string::npos;
}

} // namespace

int main() {
    // off: no refusal, whatever the device
    {
        arcint::KvStreamAsk a = ask_ok();
        a.resident_mib = 0;
        a.has_paged_fa = false;
        CHECK(arcint::kv_stream_refusal(a).empty(), "flag off is not refused");
    }
    // honoured: the one arm that proceeds
    CHECK(arcint::kv_stream_refusal(ask_ok()).empty(), "the B60 with paged FA and q8_0:q4_0 proceeds");

    // every other ask is refused, with a message that names the reason
    {
        arcint::KvStreamAsk a = ask_ok(); a.card_is_b60 = false;
        const std::string m = arcint::kv_stream_refusal(a);
        CHECK(!m.empty() && contains(m, "24 GB card"), "the 16 GB card is refused for the link");
    }
    {
        arcint::KvStreamAsk a = ask_ok(); a.has_paged_fa = false;
        const std::string m = arcint::kv_stream_refusal(a);
        CHECK(!m.empty() && contains(m, "does not take paged K/V") && contains(m, "0122"),
              "a build without the paged kernel is refused, naming the patch");
    }
    {
        arcint::KvStreamAsk a = ask_ok(); a.flash_attn = false;
        const std::string m = arcint::kv_stream_refusal(a);
        CHECK(!m.empty() && contains(m, "flash attention"), "flash attention off is refused");
    }
    {
        arcint::KvStreamAsk a = ask_ok(); a.kv_dtype = "q4_0:q4_0";
        const std::string m = arcint::kv_stream_refusal(a);
        CHECK(!m.empty() && contains(m, "q8_0:q4_0"), "another KV pair is refused");
    }
    {
        arcint::KvStreamAsk a = ask_ok(); a.decode_mib = 2048;
        const std::string m = arcint::kv_stream_refusal(a);
        CHECK(!m.empty() && contains(m, "decode"), "the unbuilt phase budget is refused, not ignored");
    }

    // the same asks under the red switch: the rule accepts them, and the checks above
    // are what fails (this arm is the red branch's, not the green tree's)
    if (const char * off = std::getenv("ARCINT_KV_STREAM_REFUSE_OFF"); off != nullptr && off[0] == '1') {
        arcint::KvStreamAsk a = ask_ok(); a.has_paged_fa = false;
        CHECK(!arcint::kv_stream_refusal(a).empty(), "RED: the refusal was switched off, the ask is accepted");
    }

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the KV stream's start refusal honours one arm and names the rest\n");
    return 0;
}
