#pragma once

#include <cstdint>
#include <string>
#include <optional>

#include "core/model_registry.h"

namespace lgc {

struct SamplerDefaults;  // core/model_registry.h

struct Config {
    // Exactly one of these selects what gets served.
    std::string model_path;  // OpenVINO IR directory (M1+)
    bool        stub = false;

    std::string model_id;  // allowlist entry; defaults to the coder under --stub

    // The name this endpoint answers to on /v1/models and /props. Empty means
    // the allowlist's canonical id. Presentation only: it does not weaken
    // --model-id, which is about artifact identity and keeps refusing a wrong
    // artifact either way (DESIGN.md §4.2).
    std::string served_model_name;
    Quant       quant = Quant::Q4;

    // OpenVINO device string. GPU.0 is the B60 and GPU.1 the A770 on the dev host;
    // CPU is useful for correctness work when a card is busy.
    std::string device = "GPU.0";

    // Compiled-blob cache. Empty disables it; a cold MoE compile is ~2 minutes.
    std::string cache_dir;

    std::string host = "127.0.0.1";
    int         port = 8090;

    int  n_ctx          = 0;      // 0: take the model's trained context
    // True only when --n-ctx was actually passed (any value, including 0),
    // never when n_ctx fell back to the artifact's own training context.
    // The auto-fit (M7) needs this distinction: an omitted --n-ctx adopts
    // whatever context the reservation computes -- that adoption IS the
    // fit -- while an operator-explicit value is verify-only and refuses
    // rather than being silently lowered.
    bool n_ctx_explicit = false;
    int parallel = 1;  // lanes (DESIGN.md §4 /health reports free/total)

    // How long a request waits for a lane before it is refused with the
    // reservation numbers (DESIGN.md §4.3). Zero refuses immediately, which is
    // the default because a lane count is a *memory* reservation: with N lanes
    // reserved, an N+1st concurrent sequence has nowhere to live, and a client
    // that waits behind a 30k-context session cannot tell a queue from a hang.
    // A non-zero value restores queueing for deployments that prefer it, and
    // /health reports the queue depth either way.
    double queue_timeout_s = 0.0;

    // Prefill chunk in tokens, 0 = one call for the whole prompt.
    //
    // Bounding the chunk is what bounds activation memory, and that — not a
    // bigger host — is how deep context is actually served. The reference on
    // this fleet (llama.cpp, 262144 context, 35B on a 16 GB A770) does exactly
    // this, at n_ubatch 512.
    //
    // 2048 is chosen so that ordinary prompts still land in a single chunk and
    // are therefore bit-identical to an unchunked run, while a long prompt is
    // split rather than allowed to scale activations without limit. Chunk
    // boundaries are not bit-exact on this backend (DESIGN.md §3.2), so this is
    // a real trade and the number is where it bites least.
    int prefill_chunk = 2048;

    // Prefix-cache budget in MiB, host-side. Zero disables it. A snapshot of
    // this architecture's state is tens of MiB (the GDN half is fixed-size), so
    // this is a real budget, not a formality (§3.3).
    int prefix_cache_mib = 0;

    // Percent of the auto-fit pool's own budget-affordable pages to hold
    // back as spare for cached prefixes (M9, exec/fit.h's
    // prefix_cache_reserve). 0 (default) is today's behaviour: auto-fit
    // adopts the whole budget as live pages and the prefix cache gets
    // whatever spare the allocation-time replay loop happens to leave,
    // which with --n-ctx omitted is usually zero (the "0 spare pages"
    // warning in backend_ov.cpp). Auto-fit only: an explicit --n-ctx
    // already defines the reserve as whatever remains after the request,
    // and needs --prefix-cache-mib > 0 -- there is nothing to reserve pages
    // for otherwise. Range [0, 90]; validated in config.cpp.
    int prefix_cache_reserve_pct = 0;

    // Headroom the auto-fit budget leaves unclaimed on the paged path
    // (M7, DESIGN §7.0.2s / the M7 design's §2-3). The only genuinely
    // *policy* term in the reservation -- every other term is measured, and a
    // flag for a measured term would be a way to lie to the arithmetic.
    // Replaces what used to be a hardcoded 256 MiB constant in load_paged.
    int fit_margin_mib = 256;

    // Slice the hidden state to its last row before the LM head, so prefill
    // computes one logit row rather than one per prompt token. On by default:
    // it is what makes deep prompts fit at all, it is faster, and it does not
    // change the output. The switch exists so the equivalence suite can prove
    // that last claim rather than assert it.
    bool slice_logits = true;

    // Path to an OpenVINO custom-layer XML. Empty = use the plugin's own
    // kernels only. Opt-in because it replaces graph nodes with hand-written
    // OpenCL and the win has to be measured per card, not assumed.
    std::string custom_kernels;

    // Percentage of MoE expert weights the GPU plugin may keep out of VRAM and
    // stream on demand. 0 = everything resident. This is what lets a model that
    // does not fit a card run on it at all; it is not free (§7).
    int offload_ratio = 0;

    // The paged execution path (DESIGN §3.5.3, §7.0): arcint-owned block
    // tables and LA state rows, speculative rollback as row promotion,
    // reservation-based admission. Default on; --no-paged selects the stateful
    // reference implementation the equivalence suite compares against.
    bool paged = true;

    // Where the embeddings gather and the MTP head run. Empty = same card as
    // --device. On a tight card the measured configuration parks both on the
    // other card: ~20 KB of activations cross per step, weights stay put.
    std::string emb_device;
    std::string mtp_device;

    // Operator serving defaults — the layer between the artifact and the
    // request (sampler-defaults order, 2026-09-01). Each flag overrides only
    // its own field of the artifact's sampler defaults; a request field still
    // wins over everything. Setting any of them turns the served provenance
    // string to "operator", so /props keeps telling the truth about where the
    // regime came from. Unset means "the artifact decides", exactly as before.
    std::optional<float> temp;
    std::optional<float> top_p;
    std::optional<int>   top_k;
    std::optional<float> repetition_penalty;
    std::optional<float> presence_penalty;

    // --chat-template-kwarg enable_thinking=BOOL: the default a request gets
    // when it sends neither chat_template_kwargs.enable_thinking nor
    // reasoning_effort. Only enable_thinking is accepted, because it is the
    // only kwarg the render path implements — an accepted-but-ignored key
    // would be a lie.
    std::optional<bool> think_default;

    int         kv_block_size             = 32;      // 16 or 32 — §8 benchmarks this
    std::string kv_dtype                  = "fp16";  // fp16 | q8 — the STATEFUL path

    // KV precision on the paged (served) path, where the plugin manages the
    // scales and q8 is real quantisation rather than a cast (DESIGN §7.0.3).
    // Default u8, and the reason is capability rather than speed: it halves KV
    // to 11.3 KiB/token, which is what lets two lanes reach agent depth on the
    // B60 and what lets the A770 reach depth at all. It is NOT free — measured
    // 2026-08-29, it costs up to 22% of prefill at 115k (§4.4) — so a one-lane
    // deep-context endpoint on a card with room should say f16.
    //
    // Grammar (M8, docs/design-m8-asymmetric-kv.md §3): KEY[:VALUE], each
    // side one of f16, u8, i8, u4, i4. No colon applies KEY to both sides, so
    // a bare "u8" or "f16" means exactly what it always did. Kept as one
    // string rather than two fields -- the pair is parsed where it is needed
    // (parse_paged_kv below), not stored twice.
    std::string paged_kv                  = "u8";    // KEY[:VALUE], see parse_paged_kv
    int         gate_pad                  = 0;       // 0 = off; 16 = the measured setting
    int         cache_grid                = 0;       // prefix-cache snapshot grid; 0 = the prefill chunk (see DESIGN 7.0.2j)
    int         cache_host_mib            = 0;       // host tier for evicted prefixes, MiB; 0 = off
    int         kv_pool_pages             = 0;       // cap the KV pool (tests: force eviction); 0 = sized by memory
    int         gdn_checkpoint_budget_mib = 512;     // §3.3
    std::string mtp                       = "auto";  // on | off | auto
    std::string mtp_layer                 = "auto";  // auto | reconstructed | exported

    // DFlash2 block-diffusion drafter (docs/dflash-pairing-probe.md): a
    // directory holding openvino_dflash_draft_stateful.xml, the selector
    // sidecars and config.json. Empty = off. Mutually exclusive with an
    // explicit --mtp on and with --draft N: one drafter per server.
    std::string dflash;
    std::string dflash_device;                       // empty = --device

    // Overrides the block size the drafter's config.json declares (M11,
    // the M11 design note (not in the repository) Q). dflash_block_set distinguishes "the flag was
    // typed" from "0 happens to be the value" the way n_ctx_explicit does for
    // --n-ctx elsewhere -- an explicit --dflash-block 0 is still out of range
    // and must refuse, not silently mean "take the json". When set, checked
    // against [2, 32] and refused without --dflash -- the block size means
    // nothing without a drafter to apply it to. `noise` in the exported graph
    // is dynamic (tools/export_dflash.py), so this needs no re-export;
    // drafts_max_ and the reservation's drafter term already read
    // dflash_block_ after it is set (backend_ov.cpp, load_paged).
    int  dflash_block     = 0;
    bool dflash_block_set = false;

    // Selector strategy over the DFlash lattice (M11, the M11 design note (not in the repository) V):
    // "greedy" commits row-by-row and never reconsiders (today's production
    // behaviour); "viterbi" finds the exact maximum-score path over the whole
    // lattice under the same additive score. Pure function:
    // src/core/dflash_select.h. Same "was it typed" distinction as
    // dflash_block_set (review follow-up): the default value
    // ("greedy") is indistinguishable from "not given" on its own, and this
    // flag is as inert without --dflash as --dflash-block/--dflash-topk are,
    // so it is refused the same way rather than silently accepted and
    // ignored.
    std::string dflash_select     = "greedy";
    bool        dflash_select_set = false;

    // Weight on the bilinear (codebook) term of the selector's score:
    // unary + lambda * bilinear. 1.0 reproduces today's scoring; 0.0 ignores
    // the codebooks entirely and scores on the target lm_head logit alone.
    // dflash_lambda_set: same reasoning as dflash_select_set above.
    float dflash_lambda     = 1.0f;
    bool  dflash_lambda_set = false;

    // Overrides the selector's top-k the drafter's config.json declares
    // (selector_top_k). Same "was it typed" distinction as dflash_block_set
    // above; when set, checked against [1, 64] and refused without --dflash.
    int  dflash_topk     = 0;
    bool dflash_topk_set = false;

    // Speculative decoding through the external-drafter hook (DESIGN.md §3.5).
    // 0 disables it. No export currently carries an MTP head, so the drafter is
    // weightless n-gram lookup; swapping in a head later changes only the
    // source of the guesses.
    int draft_tokens = 0;
    int draft_ngram  = 3;

    // Stub-only: milliseconds of artificial latency per emitted token. Exists
    // so the cancellation path (§3.7) can be demonstrated against a backend
    // that would otherwise finish before a client could disconnect.
    int stub_delay_ms = 0;

    int http_threads = 0;  // 0: httplib default
    int verbosity    = 0;  // -v, -vv

    // Pin each lane's dispatch thread to <core>+lane_index (core/affinity.h).
    // -1 = off. Testing knob for the measured hypothesis that plain decode is
    // enqueue-bound: infer() is synchronous, so the thread that calls it is
    // the HTTP request's own generation loop, and an idle core it never
    // migrates off of is what a pinned run measures against. Linux only.
    // The pin persists on the OS thread, not the request: a pooled HTTP
    // worker that picks up a different lane on its next request stays
    // pinned to this lane's core until the process exits -- there is no
    // unpin.
    int pin_dispatch = -1;

    // Compute MoE experts that would evict a resident device slot on the host
    // CPU instead of uploading them (needs --offload-ratio > 0: with every
    // expert resident there is nothing for the host tier to compute).
    bool moe_cpu_tier = false;   // --moe-cpu-tier
    int  moe_cpu_tier_threads = 0;  // --moe-cpu-tier-threads; 0 = plugin default

    bool show_help    = false;
    bool show_version = false;
};

struct ArgParse {
    bool        ok = true;
    std::string error;
};

ArgParse    parse_args(int argc, char** argv, Config& cfg);
std::string usage_text();

// --paged-kv's grammar: "<key>[:<value>]", each side one of f16, u8, i8, u4,
// i4. No colon means both sides take the same value -- "u8" and "f16" mean
// exactly what they meant before the pair syntax existed. On success `key`
// and `value` are set (equal, when `spec` had no colon) and the function
// returns true; on a spec that does not parse it returns false and leaves
// `key`/`value` untouched. Used by --paged-kv's own validation and by the
// ARCINT_PAGED_KV env override in backend_ov.cpp, so an unrecognised value
// is refused in exactly one place instead of the two drifting apart (the
// bug this milestone found: the env override used to map anything it did
// not recognise to u8, silently).
bool parse_paged_kv(const std::string& spec, std::string& key, std::string& value);

// The M8 port audit's comparison rule (backend_ov.cpp, load_paged): a
// compiled key_cache/value_cache port only has to match what was requested
// in BITWIDTH, not in exact element type -- a plugin that stores a u8
// request as i8 (or a u4 request as i4) is making its own signedness
// choice, not downgrading precision, and a mismatched bitwidth (a request
// silently kept at the old width) is the actual hazard. Amended (on-card
// finding, 2026-09-02): this plugin generation always types paged KV ports
// at 8 bits, even for a 4-bit (u4/i4) request -- the packing is config-side
// and invisible at the port level -- so a 4-bit request landing on an
// 8-bit port also passes; see kv_precision_is_packed_four_bit below for
// telling that case apart from a same-bitwidth signedness alias.
// `requested` and `actual` are each one of --paged-kv's five values
// (parse_paged_kv's alphabet); anything outside that set counts as a
// mismatch.
bool kv_precision_bitwidth_matches(const std::string& requested, const std::string& actual);

// True exactly when `requested` is 4-bit (u4/i4) and `actual` is 8-bit --
// the one shape kv_precision_bitwidth_matches now passes despite a bitwidth
// difference. The caller (backend_ov.cpp's port audit) uses this to choose
// its log wording: this is a known plugin-generation packing quirk, worth
// naming explicitly, not a generic signedness alias.
bool kv_precision_is_packed_four_bit(const std::string& requested, const std::string& actual);

// Strict base-10 uint64 parse: refuses empty input and trailing garbage
// (strtoull alone happily parses "8e9" as 8 and ignores "e9"), and refuses
// any value that does not round-trip through decimal formatting -- which
// also catches a leading '-' without special-casing it, since strtoull
// silently two's-complements a negative sign into a huge unsigned value
// ("-1" reads back as ULLONG_MAX) and that reads back as "18446744073709551615",
// not "-1". Used wherever an env var hands a byte count straight into a
// plugin property (docs/design-m8-asymmetric-kv.md review,:
// ARCINT_MOE_DEVICE_POOL_BYTES="8e9" used to silently become 8 bytes), so a
// typo refuses the load instead of changing what got requested by orders of
// magnitude.
bool parse_u64_strict(const std::string& s, uint64_t& out);

// The operator serving-defaults layer: applies the --temp/--top-p/--top-k/
// --repetition-penalty/--presence-penalty flags (only those that were set) to
// the artifact's sampler defaults and marks their provenance "operator".
// Returns the same refusal a request would get for an out-of-range value.
std::optional<std::string> apply_operator_defaults(const Config& cfg, SamplerDefaults& d);

}  // namespace lgc
