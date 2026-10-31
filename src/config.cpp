#include "config.h"

#include "core/sampling.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "util/log.h"

namespace lgc {
namespace {

bool parse_int(std::string_view s, int& out) {
    if (s.empty()) return false;

    // The string must outlive `end`: strtol's out-pointer addresses the buffer
    // it parsed, so handing it a temporary's c_str() leaves it dangling.
    const std::string text(s);
    char*             end = nullptr;
    errno                 = 0;
    const long long   value = std::strtoll(text.c_str(), &end, 10);

    if (end == text.c_str() || end == nullptr || *end != '\0') return false;
    if (errno == ERANGE || value < INT32_MIN || value > INT32_MAX) return false;
    out = static_cast<int>(value);
    return true;
}

// "A,B,...": comma-separated items, none empty
bool split_list(std::string_view s, std::vector<std::string>& out) {
    out.clear();
    for (size_t at = 0;;) {
        const size_t comma = s.find(',', at);
        out.emplace_back(s.substr(at, comma == std::string_view::npos ? std::string_view::npos : comma - at));
        if (out.back().empty()) return false;
        if (comma == std::string_view::npos) return true;
        at = comma + 1;
    }
}

bool parse_double(std::string_view s, double& out) {
    if (s.empty()) return false;

    const std::string text(s);
    char*             end = nullptr;
    errno                 = 0;
    const double      value = std::strtod(text.c_str(), &end);

    if (end == text.c_str() || end == nullptr || *end != '\0') return false;
    // NaN and infinity both have to go: an infinite duration handed to
    // condition_variable::wait_for is undefined, and in practice saturates into
    // the past, which would silently turn "wait forever" into "do not wait" —
    // the exact ambiguity the bounded admission exists to remove.
    if (errno == ERANGE || !std::isfinite(value)) return false;
    out = value;
    return true;
}

}  // namespace

bool is_retired_flag(const std::string& flag) {
    static const char* const kRetired[] = {
    "--model",
    "--model-id",
    "--quant",
    "--paged-kv",
    "--offload-ratio",
    "--moe-cpu-tier",
    "--moe-cpu-tier-threads",
    "--moe-per-expert-dispatch",
    "--dflash",
    "--dflash-device",
    "--dflash-block",
    "--dflash-select",
    "--dflash-lambda",
    "--dflash-topk",
    "--mtp",
    "--mtp-layer",
    "--mtp-device",
    "--emb-device",
    "--draft",
    "--draft-ngram",
    "--gguf-mode",
    "--gguf-native",
    "--gguf-mins",
    "--gguf-q6k",
    "--gguf-embed",
    "--gguf-check",
    "--ngram-gguf",
    "--flash-next-ngram",
    "--prefix-cache-mib",
    "--prefix-cache-reserve",
    "--cache-grid",
    "--cache-host-mib",
    "--cache-dir",
    "--fit-margin-mib",
    "--fit-ledger-dir",
    "--kv-block-size",
    "--kv-dtype",
    "--kv-pool-pages",
    "--no-paged",
    "--no-logits-slice",
    "--custom-kernels",
    "--pin-dispatch",
    "--gate-pad",
    "--dyn-quant",
    "--vision",
    "--inspect-artifact",
    "--flash-next-offload-plan",
    "--gdn-checkpoint-budget",
    "--paged-attention-max-partitions"
    };
    for (const char* r : kRetired)
        if (flag == r) return true;
    return false;
}

std::string llama_kv_dtype(const Config& cfg) {
    return cfg.llama_kv_k == cfg.llama_kv_v ? cfg.llama_kv_k : cfg.llama_kv_k + ":" + cfg.llama_kv_v;
}

std::string usage_text() {
    return
        "usage: arcint [options]\n"
        "\n"
        "model\n"
        "  --gguf FILE               serve this GGUF (the first shard of a split file) on\n"
        "                            the libllama engine: ggml's OpenCL backend. The\n"
        "                            tokenizer and the chat template come from the file\n"
        "  --mmproj FILE             the multimodal projector (libmtmd, docs/\n"
        "                            milestone-1.2.0.md): with it loaded the chat\n"
        "                            endpoint takes OpenAI image_url parts, and the\n"
        "                            audio endpoints its TTS/ASR counterparts\n"
        "  --engine llama            the executor; llama is the only value, kept for the\n"
        "                            units' sake (a bare --gguf selects it). The OpenVINO\n"
        "                            engine was retired in 0.8.0: docs/history.md, tag\n"
        "                            openvino-history\n"
        "  --tts                     serve the audio-generation endpoints: the enrolled\n"
        "                            voices' profile list (GET /v1/audio/voices) and\n"
        "                            speech synthesis (POST /v1/audio/speech). Needs\n"
        "                            --mmproj with an audio-generation gen half; it sets\n"
        "                            the context's embeddings=1 for the pipeline\n"
        "  --tts-profiles DIR        the server-side speaker profiles (one <DIR>/<name>/\n"
        "                            with reference.wav and profile.json, written by\n"
        "                            tools/tts_enrol.py). Operator-local, outside this\n"
        "                            repository (D5); default $ARCINT_TTS_PROFILES\n"
        "  --tts-lang LANG           the synthesised text's language, zh en de it pt es\n"
        "                            ja ko fr or ru (default de)\n"
        "  --tts-gen-device DEV      where the audio-generation mmproj's clip graphs\n"
        "                            run: gpu (default, on the card with the backbone)\n"
        "                            or cpu (the hybrid: backbone on the card, the\n"
        "                            speaker encoder, code predictor and code2wav on\n"
        "                            the CPU; the best measured TTS arm)\n"
        "  --stub                    serve the stub backend (no model, no GPU): the whole\n"
        "                            HTTP surface for the device-free suites\n"
        "  --stub-delay-ms N         stub only: artificial latency per token\n"
        "  --device DEV              the N-th GPU in OpenCL enumeration (GPU.N, default\n"
        "                            GPU.0) or a device-name substring (B60); two cards\n"
        "                            GPU.0,GPU.1 for --llama-layer-split\n"
        "\n"
        "server\n"
        "  --host ADDR               bind address (default: 127.0.0.1)\n"
        "  --port N                  bind port (default: 8090)\n"
        "  --served-model-name NAME  the name this endpoint reports on /v1/models and\n"
        "                            /props (default: the GGUF file's name; the stub:\n"
        "                            stub). There is no allowlist: the file is admitted\n"
        "                            by its architecture\n"
        "                            A,B (with --lane-ctx): named lanes, one per name;\n"
        "                            the request's model field picks the lane (empty:\n"
        "                            the first; unknown: 404)\n"
        "  --lane-ctx CA,CB          each named lane's context, paired with\n"
        "                            --served-model-name A,B by index; multiples of\n"
        "                            256, at most 4 lanes. The lanes share one KV pool\n"
        "                            of their sum (llama.cpp's kv_unified); --parallel\n"
        "                            and --n-ctx follow from them\n"
        "  --parallel N              number of lanes (default: 1)\n"
        "  --queue-timeout S         seconds a request waits for a lane before a 503\n"
        "                            with the lane's free count (default: 0, refuse at\n"
        "                            once)\n"
        "  --http-threads N          HTTP worker threads (default: library default)\n"
        "\n"
        "responses store\n"
        "  --responses-store-entries N  retained responses per lane (default: 32;\n"
        "                            0 disables retention entirely)\n"
        "  --responses-store-bytes B    retained bytes per lane (default: 16777216);\n"
        "                            0 disables the byte bound\n"
        "  --responses-store-ttl S      entry TTL in seconds (default: 86400)\n"
        "\n"
        "memory\n"
        "  --n-ctx N                 context length (default: the model's trained\n"
        "                            context, at most 32768). There is no auto-fit: the\n"
        "                            value is configuration, and whether it fits the\n"
        "                            card is measured at the deepest admitted prompt\n"
        "                            (DESIGN 3.3)\n"
        "  --prefill-chunk N         prefill chunk in tokens (llama.cpp's n_batch;\n"
        "                            default: 2048, 0 the same). Bounds activation\n"
        "                            memory; chunk boundaries are not bit-exact\n"
        "  --prefill-ubatch N        prefill micro-batch in tokens (llama.cpp's\n"
        "                            n_ubatch, at most the chunk; default 0: 512, or\n"
        "                            the chunk with --llama-expert-cache). Larger is\n"
        "                            faster and takes more VRAM for the compute buffer\n"
        "\n"
        "serving defaults (the operator layer: request > these > the GGUF's own;\n"
        "any of them flips the served sampler provenance to 'operator')\n"
        "  --temp X                  default temperature (0 = greedy)\n"
        "  --top-p X                 default nucleus mass\n"
        "  --top-k N                 default top-k (0 disables)\n"
        "  --min-p X                 default min-p: keep p >= X * p_max, before\n"
        "                            temperature, as llama.cpp (0 disables)\n"
        "  --repetition-penalty X    default repetition penalty\n"
        "  --presence-penalty X      default presence penalty\n"
        "  --answer-temp X           the answer row: temperature for tokens drawn\n"
        "                            after the think block closes in a thinking turn\n"
        "                            (--answer-top-p/--answer-top-k/--answer-min-p\n"
        "                            likewise; an unset field keeps the thinking row's\n"
        "                            value, none set keeps today's behaviour)\n"
        "  --answer-top-p X          the answer row: nucleus mass\n"
        "  --answer-top-k N          the answer row: top-k (0 disables)\n"
        "  --answer-min-p X          the answer row: min-p (0 disables)\n"
        "  --answer-presence-penalty X\n"
        "                            the answer row: presence penalty, counted over\n"
        "                            the tokens drawn after the closer only\n"
        "  --answer-frequency-penalty X\n"
        "                            the answer row: frequency penalty, same history\n"
        "  --nothink-temp X          the row a request gets when it resolves to\n"
        "                            thinking off (same range as --temp)\n"
        "  --nothink-top-p X         non-thinking row: default nucleus mass\n"
        "  --nothink-top-k N         non-thinking row: default top-k (0 disables)\n"
        "  --nothink-min-p X         non-thinking row: default min-p (0 disables)\n"
        "  --nothink-presence-penalty X\n"
        "                            non-thinking row: default presence penalty\n"
        "                            (unset nothink fields fall back to the row\n"
        "                            above; no --nothink-* flag leaves every\n"
        "                            request on the normal row)\n"
        "  --think-eog-mask          while a think block is open, do not offer the\n"
        "                            turn-ending token: a turn cannot then end with no\n"
        "                            answer inside an unfinished block. Off keeps the\n"
        "                            behaviour of every build up to this one; a request\n"
        "                            may set the same thing as think_eog_mask\n"
        "  --tool-grammar            constrain a tool call to the declared schemas\n"
        "  --think-ctx-reserve N     tokens of the lane's context reserved for what\n"
        "                            is left after the thought (default 0: off). The\n"
        "                            sampler forces the think-block's closer once\n"
        "                            n_ctx - prompt_tokens - N reasoning tokens have\n"
        "                            been emitted, through the loop guard's forced-\n"
        "                            closer seam; a prompt already inside the reserve\n"
        "                            forces it at the first thinking token. With a\n"
        "                            loop-guard nudge in flight the closer waits for\n"
        "                            the sentence's remaining ids: add that length to\n"
        "                            the reserve\n"
        "                            with a GBNF grammar: a call the model starts is\n"
        "                            one it finishes, so an unterminated call block is\n"
        "                            unreachable. A no-op when the request declared no\n"
        "                            tools; a request may set the same thing as\n"
        "                            tool_grammar\n"
        "  --loop-guard MODE         off|detect|nudge|slap|door|ladder|ladder-nodoor:\n"
        "                            a reasoning loop detector inside the think block.\n"
        "                            detect counts only; door masks the multi-newline\n"
        "                            tokens so the model can close by its own draw;\n"
        "                            nudge injects a fixed sentence once; slap forces\n"
        "                            the think-block's closer at the detection moment;\n"
        "                            ladder opens the door, nudges after the door grace\n"
        "                            if the loop persists, then slaps after the grace,\n"
        "                            then ends the turn if the answer loops;\n"
        "                            ladder-nodoor is the ladder before the door.\n"
        "                            off (default) is today's behaviour exactly; a\n"
        "                            request may set loop_guard for itself\n"
        "  --loop-guard-match RULE   exact|near (default near): near also counts a\n"
        "                            line whose character-3-gram set is Jaccard >=\n"
        "                            0.8 with one of the last 400 distinct counted\n"
        "                            lines; exact counts identical text only\n"
        "  --loop-guard-window N     recent reasoning lines whose repeat share is\n"
        "                            tested (default 200; only lines > 20 chars)\n"
        "  --loop-guard-threshold T  share of the window that trips the detector\n"
        "                            (default 0.8)\n"
        "  --loop-guard-grace K      more reasoning lines after a nudge before the\n"
        "                            ladder's slap (default 40)\n"
        "  --loop-guard-door-grace K more counted lines after detection before the\n"
        "                            ladder's nudge, while the door is open (default\n"
        "                            20)\n"
        "  --chat-template-kwarg enable_thinking=BOOL\n"
        "                            template default when the request sends neither\n"
        "                            chat_template_kwargs.enable_thinking nor\n"
        "                            reasoning_effort; only enable_thinking exists\n"
        "\n"
        "misc\n"
        "  -v, -vv                   raise console verbosity\n"
        "  --version                 print version and exit\n"
        "  -h, --help                print this help and exit\n"
        "\n"
        "libllama engine\n"
        "  --llama-cpu-moe N         the first N layers' experts stay in host memory,\n"
        "                            computed by the CPU (experts beyond VRAM)\n"
        "  --llama-threads N         CPU threads (default: half the hardware threads)\n"
        "  --llama-mtp N             draft up to N tokens a step with the GGUF's MTP\n"
        "                            head, verified in one pass (0-7; 0 off). Works for\n"
        "                            sampled requests as well as greedy ones\n"
        "  --llama-mtp-vocab FILE    the token ids the MTP head may draft (int32, or a\n"
        "                            JSON list): a draft step reads only those rows of\n"
        "                            the output head\n"
        "  --llama-mtp-gguf FILE     the MTP layer from this MTP-only GGUF (llama.cpp's\n"
        "                            converter --mtp) instead of the model's own file\n"
        "  --llama-mtp-min-p P       stop drafting at a token the MTP head gives less\n"
        "                            than P, and leave it out of the verify (0-1; 0 off:\n"
        "                            always --llama-mtp drafts)\n"
        "  --llama-mtp-policy off|cost\n"
        "                            cost: Strata's cost-aware window rule picks the\n"
        "                            kept drafts and the verify's rows from the measured\n"
        "                            per-width round time (default off: the fixed\n"
        "                            --llama-mtp width; one lane only)\n"
        "  --llama-mtp-cost-seed LIST the round ms per width, one list a context\n"
        "                            bucket: \"T1,T2,..;T1,..;T1,..\" (needs\n"
        "                            --llama-mtp-policy cost)\n"
        "  --llama-dflash-gguf FILE  draft with this DFlash2 drafter (llama.cpp's\n"
        "                            'dflash' GGUF for the served model) instead of\n"
        "                            an MTP head: one drafter pass proposes a block,\n"
        "                            verified in one pass (not with --llama-mtp, not\n"
        "                            under --llama-layer-split)\n"
        "  --llama-dflash N          drafts a verify with the DFlash2 drafter (1-7,\n"
        "                            default 7)\n"
        "  --llama-dflash-min-p P    stop the DFlash2 block at a position whose pick\n"
        "                            has less than P of its candidates' softmax (0-1;\n"
        "                            0 off: always --llama-dflash drafts)\n"
        "  --llama-dflash-ubatch N   the DFlash2 drafter's ubatch, 8-512 (default 64):\n"
        "                            its compute buffer holds N rows of logits\n"
        "  --llama-slot-loan MIB     lend the expert cache's tail layers' slots to a\n"
        "                            prompt of more than the resident chunk (2048) and\n"
        "                            reserve the compute buffers at the prompt's chunk\n"
        "                            (default 0, off; contrib/llama.cpp 0068)\n"
        "  --llama-loan-chunk N      the chunk a slot loan may ask for (0: the\n"
        "                            prefill chunk)\n"
        "  --llama-reserve-cells N   reserve the compute buffers for a prompt ubatch at N\n"
        "                            cache cells, not the whole context; deeper prompt\n"
        "                            chunks take smaller ubatches (0: the whole context)\n"
        "  --llama-checkpoints N     per lane, up to N snapshots of a hybrid model's\n"
        "                            recurrent state (near the prompt's end, at user\n"
        "                            messages), so a follow-up or an edited message\n"
        "                            resumes instead of re-prefilling (default 32; 0 off)\n"
        "  --llama-checkpoint-step T the least spacing of snapshots at earlier user\n"
        "                            messages (default 8192)\n"
        "  --llama-expert-cache MIB  MIB of expert slots on the card for the MoE layers\n"
        "                            --llama-cpu-moe keeps in host memory; the other\n"
        "                            experts in a bank in host memory (pinned) read by\n"
        "                            the card over the link; hot experts swapped in\n"
        "                            during decode. The card also holds a Q8_0 down\n"
        "                            projection's bank and a gather mirror beyond MIB\n"
        "                            (default 0, off; contrib/llama.cpp 0021). Two cards:\n"
        "                            A,B, a cache per card for its own layers, in\n"
        "                            --device order (0: none on that card; 0024)\n"
        "  --llama-layer-split K     two cards (--device GPU.0,GPU.1): layers [0, K) on\n"
        "                            the first, the rest, the output head and the MTP\n"
        "                            layer on the second (contrib/llama.cpp 0023)\n"
        "  --llama-mtp-card N        two cards: the card running the --llama-mtp-gguf\n"
        "                            model (default 0, the first)\n"
        "  --llama-expert-profile F  the expert ranking that fills the cache (Strata's\n"
        "                            STRP profile; default round robin)\n"
        "  --llama-expert-cache-save F\n"
        "                            save the learned STRP ranking to F at teardown,\n"
        "                            from this cache's cumulative routing counts\n"
        "                            (Strata's --expert-profile-save); needs\n"
        "                            --llama-expert-cache. A layer split has a cache a\n"
        "                            card and one path cannot name both: nothing is\n"
        "                            saved there (give a path a card). F must differ\n"
        "                            from --llama-expert-profile (the save would\n"
        "                            overwrite the file it points at). OFF by\n"
        "                            default: only this flag turns it on\n"
        "  --llama-expert-cache-save-rule R\n"
        "                            the save's ranking rule: xe (default, Strata's\n"
        "                            SYCL/xe save: prior mass by rank mixed with the\n"
        "                            counts, STRATA_PROFILE_PRIOR_ENTRIES, 800,000 by\n"
        "                            default) or cu (Strata's rank_learned_profile:\n"
        "                            resident pairs first)\n"
        "  --llama-expert-route-trace DIR\n"
        "                            binary routing trace into DIR (created if\n"
        "                            missing; one fresh file a process, rotated at\n"
        "                            256 MB; each file replays alone; nothing is\n"
        "                            deleted). OFF by default; when on it stores every\n"
        "                            request's token ids and the experts it routed,\n"
        "                            i.e. the prompt and the answer in another form,\n"
        "                            on the serving host. A prompt record is about\n"
        "                            1.9 KB a prompt token (100k: ~190 MB), so DIR\n"
        "                            belongs on a dataset with room\n"
        "  --llama-kv K[:V]          the attention cache types, f16, q8_0 or q8_0:q4_0\n"
        "                            (default f16): q8_0 takes the cache to 53 %,\n"
        "                            q8_0:q4_0 to 41 %\n";
}

ArgParse parse_args(int argc, char** argv, Config& cfg) {
    auto fail = [](std::string msg) {
        ArgParse r;
        r.ok    = false;
        r.error = std::move(msg);
        return r;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        auto value = [&](std::string_view& out) {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };

        std::string_view v;

        if (arg == "-h" || arg == "--help") {
            cfg.show_help = true;
        } else if (arg == "--version") {
            cfg.show_version = true;
        } else if (arg == "-v") {
            cfg.verbosity = 1;
        } else if (arg == "-vv") {
            cfg.verbosity = 2;
        } else if (arg == "--stub") {
            cfg.stub = true;
        } else if (arg == "--engine") {
            if (!value(v)) return fail("--engine needs llama");
            if (v == "ov")
                return fail("--engine ov was retired with the OpenVINO engine in 0.8.0: the last release "
                            "that has it is 0.7.12, its record the tag openvino-history (see docs/history.md)");
            if (v != "llama") return fail("--engine takes llama (the only engine)");
            cfg.engine = std::string(v);
        } else if (arg == "--llama-cpu-moe") {
            if (!value(v) || !parse_int(v, cfg.llama_cpu_moe) || cfg.llama_cpu_moe < 0)
                return fail("--llama-cpu-moe needs a layer count >= 0");
        } else if (arg == "--llama-threads") {
            if (!value(v) || !parse_int(v, cfg.llama_threads) || cfg.llama_threads < 0)
                return fail("--llama-threads needs a thread count >= 0");
        } else if (arg == "--llama-mtp") {
            if (!value(v) || !parse_int(v, cfg.llama_mtp) || cfg.llama_mtp < 0 || cfg.llama_mtp > 7)
                return fail("--llama-mtp needs a draft count from 0 to 7 (a verify of up to 8 rows)");
        } else if (arg == "--llama-checkpoints") {
            if (!value(v) || !parse_int(v, cfg.llama_checkpoints) || cfg.llama_checkpoints < 0)
                return fail("--llama-checkpoints needs a count >= 0 (0: off)");
        } else if (arg == "--llama-checkpoint-step") {
            if (!value(v) || !parse_int(v, cfg.llama_checkpoint_step) || cfg.llama_checkpoint_step < 1)
                return fail("--llama-checkpoint-step needs a token count >= 1");
        } else if (arg == "--llama-expert-cache") {
            std::vector<std::string> items;
            if (!value(v) || !split_list(v, items))
                return fail("--llama-expert-cache needs a size in MiB >= 0 (0: off), or one per --device card (A,B)");
            cfg.llama_expert_cache_mib_dev.clear();
            for (const std::string& it : items) {
                int mib = 0;
                if (!parse_int(it, mib) || mib < 0)
                    return fail("--llama-expert-cache needs a size in MiB >= 0 (0: off), or one per --device card (A,B)");
                cfg.llama_expert_cache_mib_dev.push_back(mib);
            }
            cfg.llama_expert_cache_mib = cfg.llama_expert_cache_mib_dev.front();
        } else if (arg == "--llama-layer-split") {
            if (!value(v) || !parse_int(v, cfg.llama_layer_split) || cfg.llama_layer_split < 1)
                return fail("--llama-layer-split needs a layer index K >= 1 (layers [0, K) on the first card)");
        } else if (arg == "--llama-mtp-card") {
            if (!value(v) || !parse_int(v, cfg.llama_mtp_card) || cfg.llama_mtp_card < 0)
                return fail("--llama-mtp-card needs a card index (0: the first --device card)");
        } else if (arg == "--llama-expert-profile") {
            if (!value(v)) return fail("--llama-expert-profile needs a profile file");
            cfg.llama_expert_profile = std::string(v);
        } else if (arg == "--llama-expert-cache-save") {
            if (!value(v)) return fail("--llama-expert-cache-save needs a profile file");
            cfg.llama_expert_cache_save = std::string(v);
        } else if (arg == "--llama-expert-cache-save-rule") {
            if (!value(v) || (std::string(v) != "xe" && std::string(v) != "cu"))
                return fail("--llama-expert-cache-save-rule takes xe or cu");
            cfg.llama_expert_cache_save_rule = std::string(v);
        } else if (arg == "--llama-expert-route-trace") {
            if (!value(v)) return fail("--llama-expert-route-trace needs a directory");
            cfg.llama_expert_route_trace = std::string(v);
        } else if (arg == "--llama-mtp-gguf") {
            if (!value(v)) return fail("--llama-mtp-gguf needs an MTP-only GGUF");
            cfg.llama_mtp_gguf = std::string(v);
        } else if (arg == "--llama-mtp-min-p") {
            if (!value(v) || !parse_double(v, cfg.llama_mtp_min_p) || cfg.llama_mtp_min_p < 0.0 || cfg.llama_mtp_min_p > 1.0)
                return fail("--llama-mtp-min-p needs a probability from 0 to 1");
        } else if (arg == "--llama-mtp-policy") {
            if (!value(v) || (std::string(v) != "off" && std::string(v) != "cost"))
                return fail("--llama-mtp-policy takes off or cost");
            cfg.llama_mtp_policy = std::string(v) == "cost";
        } else if (arg == "--llama-mtp-cost-seed") {
            if (!value(v)) return fail("--llama-mtp-cost-seed needs a cost table: \"T1,T2,..;T1,..;T1,..\"");
            cfg.llama_mtp_cost_seed = std::string(v);
        } else if (arg == "--llama-dflash-gguf") {
            if (!value(v)) return fail("--llama-dflash-gguf needs a DFlash2 drafter GGUF");
            cfg.llama_dflash_gguf = std::string(v);
        } else if (arg == "--llama-dflash") {
            if (!value(v) || !parse_int(v, cfg.llama_dflash) || cfg.llama_dflash < 1 || cfg.llama_dflash > 7)
                return fail("--llama-dflash needs a draft count from 1 to 7 (a block of up to 8, the drafter's trained size)");
        } else if (arg == "--llama-dflash-min-p") {
            if (!value(v) || !parse_double(v, cfg.llama_dflash_min_p) || cfg.llama_dflash_min_p < 0.0 ||
                cfg.llama_dflash_min_p > 1.0)
                return fail("--llama-dflash-min-p needs a probability from 0 to 1");
        } else if (arg == "--llama-dflash-ubatch") {
            if (!value(v) || !parse_int(v, cfg.llama_dflash_ubatch) || cfg.llama_dflash_ubatch < 8 ||
                cfg.llama_dflash_ubatch > 512)
                return fail("--llama-dflash-ubatch needs a ubatch from 8 to 512");
        } else if (arg == "--llama-slot-loan") {
            int mib = 0;
            if (!value(v) || !parse_int(v, mib) || mib < 0)
                return fail("--llama-slot-loan needs a size in MiB >= 1 (0: off)");
            cfg.llama_slot_loan_mib = mib;
        } else if (arg == "--llama-loan-chunk") {
            int n = 0;
            if (!value(v) || !parse_int(v, n) || n < 0)
                return fail("--llama-loan-chunk needs a token count >= 0 (0: the prefill chunk)");
            cfg.llama_loan_chunk = n;
        } else if (arg == "--llama-reserve-cells") {
            int n = 0;
            if (!value(v) || !parse_int(v, n) || n < 0)
                return fail("--llama-reserve-cells needs a cell count (0: the whole context)");
            cfg.llama_reserve_cells = static_cast<uint32_t>(n);
        } else if (arg == "--llama-mtp-vocab") {
            if (!value(v)) return fail("--llama-mtp-vocab needs a file of token ids");
            cfg.llama_mtp_vocab = std::string(v);
        } else if (arg == "--llama-kv") {
            if (!value(v)) return fail("--llama-kv needs K[:V] (f16, q8_0 or q8_0:q4_0)");
            const std::string kv(v);
            const size_t colon = kv.find(':');
            cfg.llama_kv_k = kv.substr(0, colon);
            cfg.llama_kv_v = colon == std::string::npos ? cfg.llama_kv_k : kv.substr(colon + 1);
            // the pairs contrib/llama.cpp 0015 has Intel attention kernels for
            // and that pass the answer-level bar: any other pair dequantizes all
            // of K and V to f32 on every call, and 4:4 drops the dense 27B's
            // top-1 agreement by 1.2 points (contrib/llama.cpp/README.md)
            const std::string& k = cfg.llama_kv_k;
            const std::string& vt = cfg.llama_kv_v;
            if (!(k == vt && (k == "f16" || k == "q8_0")) && !(k == "q8_0" && vt == "q4_0"))
                return fail("--llama-kv takes f16, q8_0 or q8_0:q4_0");
        } else if (arg == "--gguf") {
            if (!value(v)) return fail("--gguf needs a path");
            cfg.gguf_path = std::string(v);
        } else if (arg == "--mmproj") {
            if (!value(v)) return fail("--mmproj needs a path");
            cfg.mmproj_path = std::string(v);
        } else if (arg == "--tts") {
            cfg.tts = true;
        } else if (arg == "--tts-profiles") {
            if (!value(v)) return fail("--tts-profiles needs a directory");
            cfg.tts_profiles = std::string(v);
        } else if (arg == "--tts-lang") {
            if (!value(v)) return fail("--tts-lang needs a language code");
            cfg.tts_lang = std::string(v);
        } else if (arg == "--tts-gen-device") {
            if (!value(v)) return fail("--tts-gen-device needs a device");
            cfg.tts_gen_device = std::string(v);
        } else if (arg == "--device") {
            if (!value(v)) return fail("--device needs a value");
            cfg.device = std::string(v);
        } else if (arg == "--host") {
            if (!value(v)) return fail("--host needs a value");
            cfg.host = std::string(v);
        } else if (arg == "--port") {
            if (!value(v) || !parse_int(v, cfg.port)) return fail("--port needs an integer");
        } else if (arg == "--parallel") {
            if (!value(v) || !parse_int(v, cfg.parallel)) {
                return fail("--parallel needs an integer");
            }
            cfg.parallel_explicit = true;
        } else if (arg == "--lane-ctx") {
            // one context per --served-model-name, paired by index; checked
            // against the names below
            std::vector<std::string> items;
            if (!value(v) || !split_list(v, items))
                return fail("--lane-ctx needs a context per named lane (CA,CB in tokens)");
            cfg.lane_ctx.clear();
            for (const std::string& it : items) {
                int c = 0;
                if (!parse_int(it, c)) return fail("--lane-ctx needs a context per named lane (CA,CB in tokens), not '" + it + "'");
                cfg.lane_ctx.push_back(c);
            }
        } else if (arg == "--temp") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--temp needs a number");
            cfg.temp = static_cast<float>(d);
        } else if (arg == "--top-p") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--top-p needs a number");
            cfg.top_p = static_cast<float>(d);
        } else if (arg == "--min-p") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--min-p needs a number");
            cfg.min_p = static_cast<float>(d);
        } else if (arg == "--top-k") {
            int k = 0;
            if (!value(v) || !parse_int(v, k)) return fail("--top-k needs an integer");
            cfg.top_k = k;
        } else if (arg == "--repetition-penalty") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--repetition-penalty needs a number");
            cfg.repetition_penalty = static_cast<float>(d);
        } else if (arg == "--presence-penalty") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--presence-penalty needs a number");
            cfg.presence_penalty = static_cast<float>(d);
        } else if (arg == "--answer-temp") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--answer-temp needs a number");
            cfg.answer_temperature = static_cast<float>(d);
        } else if (arg == "--answer-top-p") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--answer-top-p needs a number");
            cfg.answer_top_p = static_cast<float>(d);
        } else if (arg == "--answer-top-k") {
            int k = 0;
            if (!value(v) || !parse_int(v, k)) return fail("--answer-top-k needs an integer");
            cfg.answer_top_k = k;
        } else if (arg == "--answer-min-p") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--answer-min-p needs a number");
            cfg.answer_min_p = static_cast<float>(d);
        } else if (arg == "--answer-presence-penalty") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--answer-presence-penalty needs a number");
            cfg.answer_presence_penalty = static_cast<float>(d);
        } else if (arg == "--answer-frequency-penalty") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--answer-frequency-penalty needs a number");
            cfg.answer_frequency_penalty = static_cast<float>(d);
        } else if (arg == "--nothink-temp") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--nothink-temp needs a number");
            cfg.nothink_temp = static_cast<float>(d);
        } else if (arg == "--nothink-top-p") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--nothink-top-p needs a number");
            cfg.nothink_top_p = static_cast<float>(d);
        } else if (arg == "--nothink-top-k") {
            int k = 0;
            if (!value(v) || !parse_int(v, k)) return fail("--nothink-top-k needs an integer");
            cfg.nothink_top_k = k;
        } else if (arg == "--nothink-min-p") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--nothink-min-p needs a number");
            cfg.nothink_min_p = static_cast<float>(d);
        } else if (arg == "--nothink-presence-penalty") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d)) return fail("--nothink-presence-penalty needs a number");
            cfg.nothink_presence_penalty = static_cast<float>(d);
        } else if (arg == "--think-eog-mask") {
            cfg.think_eog_mask = true;
        } else if (arg == "--think-ctx-reserve") {
            int n = 0;
            if (!value(v) || !parse_int(v, n) || n < 0)
                return fail("--think-ctx-reserve needs a token count >= 0 (0: off)");
            cfg.think_ctx_reserve = n;
        } else if (arg == "--tool-grammar") {
            cfg.tool_grammar = true;
        } else if (arg == "--loop-guard") {
            if (!value(v)) return fail("--loop-guard needs off|detect|nudge|slap|door|ladder|ladder-nodoor");
            if (!loop_guard_parse(v)) return fail("--loop-guard needs off|detect|nudge|slap|door|ladder|ladder-nodoor");
            cfg.loop_guard = std::string(v);
        } else if (arg == "--loop-guard-match") {
            if (!value(v)) return fail("--loop-guard-match needs exact|near");
            if (!loop_guard_match_parse(v)) return fail("--loop-guard-match needs exact|near");
            cfg.loop_guard_match     = std::string(v);
            cfg.loop_guard_match_set = true;
        } else if (arg == "--loop-guard-window") {
            int n = 0;
            if (!value(v) || !parse_int(v, n) || n < 1) return fail("--loop-guard-window needs an integer >= 1");
            cfg.loop_guard_window = n;
        } else if (arg == "--loop-guard-threshold") {
            double d = 0.0;
            if (!value(v) || !parse_double(v, d) || d < 0.0 || d > 1.0)
                return fail("--loop-guard-threshold needs a number in [0, 1]");
            cfg.loop_guard_threshold = static_cast<float>(d);
        } else if (arg == "--loop-guard-door-grace") {
            int k = 0;
            if (!value(v) || !parse_int(v, k) || k < 0)
                return fail("--loop-guard-door-grace needs an integer >= 0");
            cfg.loop_guard_door_grace = k;
        } else if (arg == "--loop-guard-grace") {
            int k = 0;
            if (!value(v) || !parse_int(v, k) || k < 0) return fail("--loop-guard-grace needs an integer >= 0");
            cfg.loop_guard_grace = k;
        } else if (arg == "--chat-template-kwarg") {
            if (!value(v)) return fail("--chat-template-kwarg needs key=value");
            const std::string_view kv = v;
            const size_t           eq = kv.find('=');
            if (eq == std::string_view::npos) return fail("--chat-template-kwarg needs key=value");
            const std::string_view key = kv.substr(0, eq);
            const std::string_view val = kv.substr(eq + 1);
            if (key != "enable_thinking") {
                return fail("--chat-template-kwarg understands only enable_thinking; an "
                            "accepted-but-ignored key would be a lie");
            }
            if (val == "true" || val == "1") {
                cfg.think_default = true;
            } else if (val == "false" || val == "0") {
                cfg.think_default = false;
            } else {
                return fail("--chat-template-kwarg enable_thinking needs true or false");
            }
        } else if (arg == "--served-model-name") {
            // An empty value is refused rather than folded into "not given":
            // `--served-model-name ""` asks for a name, and silently serving the
            // canonical one instead is how a roster ends up pinned to a name
            // nobody chose.
            if (!value(v) || v.empty()) {
                return fail("--served-model-name needs a non-empty name");
            }
            cfg.served_model_name = std::string(v);
        } else if (arg == "--queue-timeout") {
            if (!value(v) || !parse_double(v, cfg.queue_timeout_s)) {
                return fail("--queue-timeout needs a number of seconds");
            }
        } else if (arg == "--responses-store-entries") {
            if (!value(v) || !parse_int(v, cfg.responses_store_entries)) {
                return fail("--responses-store-entries needs an integer");
            }
        } else if (arg == "--responses-store-bytes") {
            if (!value(v) || !parse_int(v, cfg.responses_store_bytes)) {
                return fail("--responses-store-bytes needs an integer");
            }
        } else if (arg == "--responses-store-ttl") {
            if (!value(v) || !parse_int(v, cfg.responses_store_ttl_s)) {
                return fail("--responses-store-ttl needs an integer (seconds)");
            }
        } else if (arg == "--stub-delay-ms") {
            if (!value(v) || !parse_int(v, cfg.stub_delay_ms)) {
                return fail("--stub-delay-ms needs an integer");
            }
        } else if (arg == "--http-threads") {
            if (!value(v) || !parse_int(v, cfg.http_threads)) {
                return fail("--http-threads needs an integer");
            }
        } else if (arg == "--n-ctx") {
            if (!value(v) || !parse_int(v, cfg.n_ctx)) return fail("--n-ctx needs an integer");
            // Distinguishes "the operator asked for this" from "nothing was
            // asked, take the model's own context" -- M7's auto-fit adopts
            // its computed max only in the second case.
            cfg.n_ctx_explicit = true;
        } else if (arg == "--prefill-chunk") {
            if (!value(v) || !parse_int(v, cfg.prefill_chunk)) {
                return fail("--prefill-chunk needs an integer");
            }
        } else if (arg == "--prefill-ubatch") {
            if (!value(v) || !parse_int(v, cfg.prefill_ubatch)) {
                return fail("--prefill-ubatch needs an integer");
            }
        } else if (is_retired_flag(std::string(arg))) {
            return fail(std::string(arg) + " was retired with the OpenVINO engine in 0.8.0 (see docs/history.md)");
        } else {
            return fail(log::format("unknown option '%s' (try --help)",
                                    std::string(arg).c_str()));
        }
    }

    if (cfg.show_help || cfg.show_version) return {};

    // ------------------------------------------------------------ validation
    if (cfg.stub && !cfg.gguf_path.empty()) return fail("--gguf and --stub are mutually exclusive");
    if (!cfg.stub && cfg.gguf_path.empty())
        return fail("nothing to serve: pass --gguf FILE, or --stub for the device-free stub backend");
    // --tts (docs/milestone-1.2.0.md D4/D5): the operator-local profile
    // directory may come from the environment instead of the flag.
    if (cfg.tts && cfg.tts_profiles.empty()) {
        if (const char* env = std::getenv("ARCINT_TTS_PROFILES"); env != nullptr && *env != '\0')
            cfg.tts_profiles = env;
    }

    if (cfg.port < 1 || cfg.port > 65535) return fail("--port must be in [1, 65535]");
    // A served name that is empty or blank would put an unusable id on
    // /v1/models, and a roster discovering names from there would take it.
    if (!cfg.served_model_name.empty() &&
        cfg.served_model_name.find_first_not_of(" \t\r\n") == std::string::npos) {
        return fail("--served-model-name cannot be blank");
    }
    if (cfg.parallel < 1) return fail("--parallel must be >= 1");
    // --served-model-name A,B: named lanes (a single name, as before, may not
    // contain a comma). The first name is the served one.
    if (cfg.served_model_name.find(',') != std::string::npos) {
        std::vector<std::string> names;
        if (!split_list(cfg.served_model_name, names))
            return fail("--served-model-name A,B: a list of names, none empty");
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i].find_first_not_of(" \t\r\n") == std::string::npos)
                return fail("--served-model-name A,B: a name cannot be blank");
            if (names[i].find_first_of(" \t\r\n") != std::string::npos)
                return fail("--served-model-name A,B: '" + names[i] + "' contains whitespace (a request's model field would not reach it)");
            for (size_t j = 0; j < i; ++j)
                if (names[i] == names[j]) return fail("--served-model-name names " + names[i] + " twice: a name picks one lane");
        }
        cfg.served_model_name = names[0];
        cfg.lane_names        = std::move(names);
    }
    if (!(cfg.queue_timeout_s >= 0.0 && cfg.queue_timeout_s <= 3600.0)) {
        return fail("--queue-timeout must be between 0 and 3600 seconds");
    }
    if (cfg.http_threads < 0) return fail("--http-threads must be >= 0");
    if (cfg.responses_store_entries < 0)
        return fail("--responses-store-entries must be >= 0");
    if (cfg.responses_store_bytes < 0) return fail("--responses-store-bytes must be >= 0");
    if (cfg.responses_store_ttl_s < 0) return fail("--responses-store-ttl must be >= 0");
    if (cfg.stub_delay_ms < 0) return fail("--stub-delay-ms must be >= 0");
    if (cfg.device.empty()) return fail("--device must not be empty");
    if (cfg.stub_delay_ms > 0 && !cfg.stub) return fail("--stub-delay-ms only applies to --stub");
    if (cfg.n_ctx < 0) return fail("--n-ctx must be >= 0");
    if (cfg.prefill_chunk < 0) return fail("--prefill-chunk must be >= 0");
    if (cfg.prefill_ubatch < 0) return fail("--prefill-ubatch must be >= 0");
    // The libllama engine is the only one; a bare --gguf selects it.
    if (cfg.engine.empty()) cfg.engine = "llama";
    // Named lanes (DESIGN §4.2, amended 2026-10-07): the libllama engine's
    // shared KV pool (docs/campaigns/lanes-agent-subagent.md), and the stub's
    // admission for the device-free suites
    if (!cfg.lane_names.empty() || !cfg.lane_ctx.empty()) {
        constexpr size_t kMaxLanes = 4;
        if (cfg.lane_ctx.empty())
            return fail("--served-model-name A,B names lanes: give each its context with --lane-ctx CA,CB");
        if (cfg.lane_names.empty())
            return fail(cfg.served_model_name.empty()
                            ? "--lane-ctx needs the lanes' names: --served-model-name A,B"
                            : "--lane-ctx with one name: one lane's context is --n-ctx (named lanes are two or more)");
        if (cfg.lane_names.size() != cfg.lane_ctx.size())
            return fail(log::format("--served-model-name lists %zu names and --lane-ctx %zu contexts: they pair by index",
                                    cfg.lane_names.size(), cfg.lane_ctx.size()));
        if (cfg.lane_names.size() > kMaxLanes)
            return fail(log::format("%zu named lanes: at most %zu", cfg.lane_names.size(), kMaxLanes));
        long long sum = 0;
        for (int c : cfg.lane_ctx) {
            // llama.cpp pads the context to 256 (llama-context.cpp:291): a cap
            // off that grid would make the pool larger than the lanes' sum
            if (c <= 0 || c % 256 != 0)
                return fail(log::format("--lane-ctx %d: a lane's context is a positive multiple of 256", c));
            sum += c;
        }
        if (sum > INT32_MAX) return fail("--lane-ctx: the lanes' contexts sum past 2^31 tokens");
        if (cfg.n_ctx_explicit)
            return fail("--n-ctx and --lane-ctx: the lanes' contexts are the context; give --lane-ctx alone");
        if (cfg.parallel_explicit && cfg.parallel != static_cast<int>(cfg.lane_names.size()))
            return fail(log::format("--parallel %d disagrees with %zu named lanes (the lane count is the number of names)",
                                    cfg.parallel, cfg.lane_names.size()));
        cfg.parallel = static_cast<int>(cfg.lane_names.size());
    }
    if (cfg.stub) {
        if (cfg.llama_cpu_moe > 0 || cfg.llama_threads > 0 || cfg.llama_mtp > 0 || !cfg.llama_mtp_vocab.empty() ||
            !cfg.llama_mtp_gguf.empty() || cfg.llama_mtp_min_p > 0.0 || cfg.llama_mtp_policy ||
            !cfg.llama_mtp_cost_seed.empty() || cfg.llama_dflash_on() || cfg.llama_dflash > 0 ||
            cfg.llama_dflash_min_p > 0.0 || cfg.llama_dflash_ubatch > 0 ||
            cfg.llama_kv_k != "f16" || cfg.llama_kv_v != "f16" || cfg.llama_checkpoints != 32 || cfg.llama_checkpoint_step != 8192 ||
            cfg.llama_expert_cache_on() || !cfg.llama_expert_profile.empty() || cfg.llama_layer_split > 0 || cfg.llama_mtp_card >= 0 ||
            cfg.llama_reserve_cells > 0 || cfg.tts || !cfg.tts_profiles.empty() || cfg.tts_lang != "de" ||
            cfg.tts_gen_device != "gpu" || !cfg.mmproj_path.empty())
            return fail("--llama-cpu-moe, --llama-threads, --llama-mtp, --llama-mtp-vocab/-gguf/-min-p, --llama-dflash/-gguf/-min-p/-ubatch, "
                        "--llama-kv, --llama-checkpoints/-step, "
                        "--llama-expert-cache/-profile, --llama-layer-split, --llama-mtp-card, --llama-reserve-cells, --mmproj and --tts "
                        "(-profiles/-lang/-gen-device) are options of "
                        "the libllama engine (--gguf), not of --stub");
        return {};
    }
    // --tts: a mmproj with the audio-generation half and the enrolled voices'
    // directory. The flag without either is a boot error, not a request-time
    // one: the process could never answer the routes it was told to serve.
    if (cfg.tts && cfg.mmproj_path.empty())
        return fail("--tts needs --mmproj (the audio-generation projector: the speaker encoder and the acoustic decoder)");
    if (cfg.tts && cfg.tts_profiles.empty())
        return fail("--tts needs --tts-profiles DIR (or $ARCINT_TTS_PROFILES): the enrolled voices' profile "
                    "directory (tools/tts_enrol.py, docs/speech-service.md D5)");
    if (!cfg.tts && !cfg.tts_profiles.empty())
        return fail("--tts-profiles needs --tts (the audio-generation path)");
    if (!cfg.tts && cfg.tts_lang != "de")
        return fail("--tts-lang needs --tts");
    if (!cfg.tts && cfg.tts_gen_device != "gpu")
        return fail("--tts-gen-device needs --tts (the audio-generation path)");
    if (cfg.tts_gen_device != "gpu" && cfg.tts_gen_device != "cpu")
        return fail("--tts-gen-device takes gpu (the card, the default) or cpu (the hybrid: backbone on the card, the mmproj graphs on the CPU)");
    {
        static const char* const kTtsLangs[] = {"zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"};
        bool known = false;
        for (const char* l : kTtsLangs) if (cfg.tts_lang == l) known = true;
        if (!known)
            return fail("--tts-lang takes one of the backbone's codec_language codes: zh, en, de, it, pt, es, ja, ko, fr, ru");
    }
    // DFlash2: one drafter a context (the MTP layer is then not loaded)
    if (cfg.llama_dflash > 0 && !cfg.llama_dflash_on()) return fail("--llama-dflash needs --llama-dflash-gguf (the drafter)");
    if (cfg.llama_dflash_min_p > 0.0 && !cfg.llama_dflash_on())
        return fail("--llama-dflash-min-p needs --llama-dflash-gguf (the drafter)");
    if (cfg.llama_dflash_ubatch > 0 && !cfg.llama_dflash_on())
        return fail("--llama-dflash-ubatch needs --llama-dflash-gguf (the drafter)");
    if (cfg.llama_dflash_on() && (cfg.llama_mtp > 0 || !cfg.llama_mtp_vocab.empty() || !cfg.llama_mtp_gguf.empty() ||
                                  cfg.llama_mtp_min_p > 0.0 || cfg.llama_mtp_policy || !cfg.llama_mtp_cost_seed.empty()))
        return fail("--llama-dflash-gguf and --llama-mtp (-vocab, -gguf, -min-p) are exclusive: one drafter a context, "
                    "the DFlash2 drafter or the MTP head");
    if (cfg.llama_dflash_on() && cfg.llama_dflash == 0) cfg.llama_dflash = 7;   // the drafter's block of 8
    if (cfg.llama_dflash_on() && cfg.llama_dflash_ubatch == 0) cfg.llama_dflash_ubatch = 64;
    if (!cfg.llama_mtp_vocab.empty() && cfg.llama_mtp <= 0) return fail("--llama-mtp-vocab needs --llama-mtp");
    if (!cfg.llama_mtp_gguf.empty() && cfg.llama_mtp <= 0) return fail("--llama-mtp-gguf needs --llama-mtp");
    if (cfg.llama_mtp_min_p > 0.0 && cfg.llama_mtp <= 0) return fail("--llama-mtp-min-p needs --llama-mtp");
    if (cfg.llama_mtp_policy && cfg.llama_mtp <= 0) return fail("--llama-mtp-policy needs --llama-mtp");
    if (!cfg.llama_mtp_cost_seed.empty() && !cfg.llama_mtp_policy)
        return fail("--llama-mtp-cost-seed needs --llama-mtp-policy cost");
    if (cfg.llama_expert_cache_on() && cfg.llama_cpu_moe <= 0)
        return fail("--llama-expert-cache caches experts --llama-cpu-moe keeps in host memory: give --llama-cpu-moe");
    if (!cfg.llama_expert_profile.empty() && !cfg.llama_expert_cache_on())
        return fail("--llama-expert-profile needs --llama-expert-cache");
    if (!cfg.llama_expert_cache_save.empty() && !cfg.llama_expert_cache_on())
        return fail("--llama-expert-cache-save needs --llama-expert-cache");
    // the save would write over the profile the same run reads: refuse at config
    if (!cfg.llama_expert_cache_save.empty() && cfg.llama_expert_cache_save == cfg.llama_expert_profile)
        return fail("--llama-expert-cache-save must differ from --llama-expert-profile: the save would overwrite it");
    // --device: one card, or the two of a layer split (no automatic split:
    // the split point decides each card's VRAM, so it is the operator's)
    if (!split_list(cfg.device, cfg.llama_devices))
        return fail("--device GPU.0,GPU.1: a list of cards, none empty");
    for (size_t i = 0; i < cfg.llama_devices.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (cfg.llama_devices[i] == cfg.llama_devices[j]) return fail("--device names " + cfg.llama_devices[i] + " twice");
    if (cfg.llama_devices.size() > 2) return fail("--device takes one card, or two for --llama-layer-split");
    if (cfg.llama_devices.size() == 2 && cfg.llama_layer_split <= 0)
        return fail("two cards need --llama-layer-split K (layers [0, K) on the first card; there is no automatic split)");
    if (cfg.llama_layer_split > 0 && cfg.llama_devices.size() != 2)
        return fail("--llama-layer-split needs two cards: --device GPU.0,GPU.1");
    // the dense agent runs on one card, and no DFlash drafter for a split
    // model is known (docs/campaigns/dflash2-agent.md, operator 2026-10-16)
    if (cfg.llama_dflash_on() && cfg.llama_layer_split > 0)
        return fail("--llama-dflash-gguf is not supported under --llama-layer-split: the DFlash2 drafter runs with a "
                    "model on one card");
    if (cfg.llama_mtp_card >= 0 && (cfg.llama_devices.size() != 2 || cfg.llama_mtp_card >= static_cast<int>(cfg.llama_devices.size())))
        return fail("--llama-mtp-card N: a card of the layer split (0 or 1)");
    if (cfg.llama_mtp_card >= 0 && cfg.llama_mtp_gguf.empty())
        return fail("--llama-mtp-card places the --llama-mtp-gguf model (the GGUF's own MTP layer runs with the last layer)");
    if (cfg.llama_devices.size() == 2 && cfg.llama_mtp_card < 0) cfg.llama_mtp_card = 0;
    if (!cfg.llama_expert_cache_mib_dev.empty() && cfg.llama_expert_cache_mib_dev.size() != cfg.llama_devices.size())
        return fail("--llama-expert-cache takes one size per --device card (A,B in MiB; 0: none on that card)");

    return {};
}

std::optional<std::string> apply_operator_defaults(const Config& cfg, SamplerDefaults& d) {
    SamplerOverrides o;
    o.temperature        = cfg.temp;
    o.top_p              = cfg.top_p;
    o.top_k              = cfg.top_k;
    o.min_p              = cfg.min_p;
    o.repetition_penalty = cfg.repetition_penalty;
    o.presence_penalty   = cfg.presence_penalty;
    o.answer_temperature = cfg.answer_temperature;
    o.answer_top_p       = cfg.answer_top_p;
    o.answer_top_k       = cfg.answer_top_k;
    o.answer_min_p       = cfg.answer_min_p;
    o.answer_presence_penalty  = cfg.answer_presence_penalty;
    o.answer_frequency_penalty = cfg.answer_frequency_penalty;
    if (cfg.think_eog_mask) o.think_eog_mask = true;
    if (cfg.think_ctx_reserve > 0) o.think_ctx_reserve = cfg.think_ctx_reserve;
    if (cfg.tool_grammar)   o.tool_grammar   = true;
    // loop-guard: off changes nothing and does not mark the defaults operator;
    // any other rung carries its window/threshold/grace with it.
    if (const auto mode = loop_guard_parse(cfg.loop_guard); mode && *mode != LoopGuardMode::Off) {
        o.loop_guard           = *mode;
        o.loop_guard_window    = cfg.loop_guard_window;
        o.loop_guard_threshold = cfg.loop_guard_threshold;
        o.loop_guard_grace     = cfg.loop_guard_grace;
        o.loop_guard_door_grace = cfg.loop_guard_door_grace;
    }
    if (cfg.loop_guard_match_set) {
        if (const auto match = loop_guard_match_parse(cfg.loop_guard_match)) o.loop_guard_match = *match;
    }
    o.nothink_temperature      = cfg.nothink_temp;
    o.nothink_top_p            = cfg.nothink_top_p;
    o.nothink_top_k            = cfg.nothink_top_k;
    o.nothink_min_p            = cfg.nothink_min_p;
    o.nothink_presence_penalty = cfg.nothink_presence_penalty;
    return sampler_defaults_apply(d, o);
}

}  // namespace lgc
