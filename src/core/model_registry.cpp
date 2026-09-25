#include "core/model_registry.h"

#include <algorithm>

#include "util/log.h"

namespace lgc {
namespace {

// The Qwen3-family model-card sampler recommendation. Provisional by
// construction — models/allowlist-raw.json carries no sampler settings, so
// these are inherited, not measured. See SamplerDefaults::provenance.
SamplerDefaults qwen_card_defaults() {
    SamplerDefaults d;
    d.temperature        = 0.7f;
    d.top_p              = 0.8f;
    d.top_k              = 20;
    d.repetition_penalty = 1.05f;
    d.presence_penalty   = 0.0f;
    d.provenance         = "provisional";
    return d;
}

// One layer in `interval` is full attention; the remainder carry GDN state.
void split_layers(ModelEntry& e) {
    e.n_attn_layer = e.full_attention_interval > 0 ? e.n_layer / e.full_attention_interval : 0;
    e.n_gdn_layer  = e.n_layer - e.n_attn_layer;
}

std::vector<ModelEntry> build_registry() {
    std::vector<ModelEntry> r;

    {
        ModelEntry e;
        e.id                      = "qwen3.6-27b-a3b-coder";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-coder-b5-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = false;  // no MTP graph in the export
        e.mtp_head_pinned         = true;   // inspected 2026-08-28
        e.mtp_in_checkpoint       = true;   // config: mtp_num_hidden_layers 1
        e.n_embd                  = 2048;
        e.n_expert                = 184;  // pruned from 256
        e.full_attention_interval = 4;
        e.n_layer                 = 40;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4, Quant::Q8};
        e.arch_hash               = "6745cfe3d57e3f0f";
        e.template_hash           = "e84f32a23fdda276";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 13760293946ull;
        e.status                  = "production, 10/10 on the Prüfstand (b5 artifact)";
        e.sampler                 = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        ModelEntry e;
        e.id                      = "qwen3.6-35b-a3b";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-35b-a3b-int4-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = false;  // no MTP graph in the export
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2048;
        e.n_expert                = 256;
        e.full_attention_interval = 4;
        e.n_layer                 = 40;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4, Quant::Q8};
        e.arch_hash               = "21fe4d57d6d016f5";
        e.template_hash           = "c3f7038f278583e1";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 18646558274ull;
        // Measured 2026-08-30 as the fleet's agent endpoint (arcint-agent.service,
        // GPU.0 = B60, 262144 context, the reservation admitting 377552): 10/10
        // greedy on the acceptance task, 3/3 on the tool probe, clean German with
        // enable_thinking:false, 62.7 t/s decode at 53.5k and 1584 t/s prefill.
        e.status                  = "10/10 greedy, 3/3 tool probe (B60, 262144, 2026-08-30); "
                                    "62.7 t/s decode at 53.5k, 1584 t/s prefill";
        e.sampler                 = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        // The same 35B artifact with the reconstructed MTP head beside it, as a
        // separate directory of symlinks, so the production agent's directory is
        // never touched: --mtp auto turns drafting on the moment a head exists,
        // and a head goes into production only after it is measured.
        ModelEntry e;
        e.id                      = "qwen3.6-35b-a3b-mtp";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-35b-a3b-mtp-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = true;   // the reconstructed head (tools/export_mtp.py, MoE)
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2048;
        e.n_expert                = 256;
        e.full_attention_interval = 4;
        e.n_layer                 = 40;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "21fe4d57d6d016f5";
        e.template_hash           = "c3f7038f278583e1";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 18646558274ull;
        e.status                  = "head pairs: 93.9% / 75.4% draft acceptance (code / prose, B60, "
                                    "2026-08-30) but --mtp on decodes at 48-53 t/s against 71.5 off: "
                                    "the serving loop is host-bound and the f16 dense-expert head "
                                    "is a wash on device time; not for production";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        ModelEntry e;
        e.id                      = "qwen3.8-27b";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-b7c1-ov"};
        e.ov_arch                 = "Qwen3_5ForConditionalGeneration";
        e.model_type              = "qwen3_5";
        e.moe                     = false;  // dense (§2)
        // §3.5 says this one ships a native MTP head. optimum-intel's export
        // drops it, so for a long time the answer here was "no". It is now
        // "yes": tools/export_mtp.py reconstructs the head from the weights in
        // the checkpoint and writes openvino_mtp_{layer,lm_head}.xml beside the
        // model. An artifact without those files still loads -- the check
        // against this flag is what tells the user their export lacks them.
        e.has_mtp_head            = true;
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 5120;
        e.n_expert                = 0;
        e.full_attention_interval = 4;
        e.n_layer                 = 64;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4, Quant::Q8};
        e.arch_hash               = "5892b9b333bf0ab3";
        e.template_hash           = "c3cf9e34abf4f9e3";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 14405167394ull;
        // Measured through arcint's paged executor with the MTP head drafting
        // (2026-08-29): 10/10 under greedy at 36.2 t/s on the B60, 93.2% draft
        // acceptance. The earlier stateful measurement was 8/10 greedy; greedy
        // is deterministic per configuration, so both numbers are real -- the
        // paged path's near-tie landings simply score better on this task.
        e.status = "10/10 greedy (paged+MTP, B60); AWQ-only - SE calibration degenerates "
                   "greedy, do not re-add SE";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        // Intel's public export of the same checkpoint -- OpenVINO/Qwen3.8-27B-
        // int4-ov -- as its own entry, not an alias of qwen38-b7c1-ov: that entry
        // carries a measured status for our AWQ-only export, and an alias would
        // make the allowlist assert a result nobody obtained for this file.
        // Same geometry (hidden 5120, 64 layers, 24/4 heads, vocab 248320, one
        // MTP layer, untied embeddings), same chat template and tokenizer,
        // different quantisation of the body. The reconstructed MTP head from
        // tools/export_mtp.py is placed beside it; whether it pairs is what the
        // draft-acceptance measurement decides (DESIGN 7.0.2n).
        ModelEntry e;
        e.id                      = "qwen3.8-27b-intel-int4";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-intel-int4-ov"};
        e.ov_arch                 = "Qwen3_5ForConditionalGeneration";
        e.model_type              = "qwen3_5";
        e.moe                     = false;
        e.has_mtp_head            = true;   // the reconstructed head, copied beside it
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 5120;
        e.n_expert                = 0;
        e.full_attention_interval = 4;
        e.n_layer                 = 64;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "ab94a08ce150de6a";
        e.template_hash           = "c3cf9e34abf4f9e3";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 13929177378ull;
        // The reconstructed head pairs with this export (2026-08-30, B60, greedy,
        // thinking off): draft acceptance 96.3% (157/163) on a code prompt and
        // 77.3% (140/181) on prose, 34.6-37.4 t/s -- next to the 93.2% the same
        // head measures on our own export. A wrong head cannot raise acceptance,
        // only depress it, so the number is the oracle (tools/export_mtp.py).
        e.status                  = "10/10 greedy (paged+MTP, B60, 2026-08-30), 36.3 t/s at 90.8% draft "
                                    "acceptance vs 25.0 t/s plain; Intel's own MTP layer + our lm_head 37.7-38.1 t/s";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // Qwen3.5-2B: the smallest dense qwen3_5 checkpoint, exported for FIX 8.2
        // (GGUF serving on the 16 GiB card). Hybrid architecture: 18 linear-attention
        // + 6 full-attention layers. Hashes pinned 2026-09-10 off the real IR at
        // /models/ov/qwen35-2b-ov on the dev host (sha256 prefixes, byte-identical to the
        // server's load path): same tokenizer and chat template as qwen38-b7c1-ov;
        // the plain export carries no MTP graph (mtp_head_exported false).
        ModelEntry e;
        e.id                      = "qwen3.5-2b";
        e.family                  = "qwen3.5";
        e.artifact_aliases        = {"qwen35-2b-ov"};
        e.ov_arch                 = "Qwen3_5ForConditionalGeneration";
        e.model_type              = "qwen3_5";
        e.moe                     = false;
        e.has_mtp_head            = false;  // no MTP graph in the export
        e.mtp_head_pinned         = true;   // inspected 2026-09-10
        e.mtp_in_checkpoint       = true;   // config: mtp_num_hidden_layers 1
        e.n_embd                  = 2048;
        e.n_expert                = 0;
        e.full_attention_interval = 4;
        e.n_layer                 = 24;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "ec0078368fcce101";
        e.template_hash           = "c3cf9e34abf4f9e3";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 1884937298ull;
        e.status                  = "provisional; dense qwen35 marker export (FIX 8.2), "
                                    "served via GGUF on the 16 GiB card, not acceptance-scored";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // FULL-DEPTH (2026-09-13): the Qwen3.8 Flash-Next (qwen4exp) serving-
        // shape IR at DEPTH 4 of 48, real weights from the UD-Q3_K_XL shards
        // (tools/export_serving_artifact.py; serving-shape.json beside the
        // IR names the tree, the ports and the fill). It exists so the served
        // path -- the binary, its loader, bind_ngram_ports -- runs on a card
        // at all; until it, the only thing that had fed this IR on a card was
        // the labelled probe. Hashes read off the dev-host directory. The
        // n-gram table is NOT in the .bin: it binds to the IR's ngram_table.K
        // ports from the GGUF shard at load (--ngram-gguf). A measurement
        // artifact: 44 layers are missing and nothing it says is the model's
        // answer; the full-depth artifact is its own entry when it exists.
        ModelEntry e;
        e.id                      = "qwen3.8-flash-next-d4";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-flash-next-d4-ov"};
        e.ov_arch                 = "Qwen4ExpForConditionalGeneration";
        e.model_type              = "qwen4_exp";
        e.moe                     = true;
        e.has_mtp_head            = false;  // no MTP graph beside this IR
        e.mtp_head_pinned         = true;   // inspected 2026-09-13 (the export writes none)
        e.mtp_in_checkpoint       = true;   // the checkpoint ships an MTP head (staged, not served)
        e.n_embd                  = 2560;
        e.n_expert                = 512;
        e.full_attention_interval = 4;
        e.n_layer                 = 4;      // of 48: layer 3 is the one attention layer
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "2910a860bf9dc6bb";
        e.template_hash           = "12827f24b742ea4e";  // the GGUF's own chat template
        e.tokenizer_hash          = "87a7830d63fcf43b";  // passthrough; vocab == the GGUF's, id for id
        e.weights_bytes           = 9427885993ull;
        e.status                  = "measurement artifact: depth 4 of 48, served-path boot; "
                                    "not the model's answers";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // THE 12-LAYER RUNG (0.5.1 WP3, 2026-09-13): the serving-shape IR at
        // depth 12 of 48 -- three attention layers, nine GDN -- the first rung
        // of the depth ladder docs/window-051.md row (b) prices before the
        // segmented forward lands. Single model, expert bodies as constants
        // (the staging edge is what its compile measures). Hashes read off
        // the dev-host directory. A measurement artifact: 36 layers are
        // missing and nothing it says is the model's answer.
        ModelEntry e;
        e.id                      = "qwen3.8-flash-next-d12";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-flash-next-d12-ov"};
        e.ov_arch                 = "Qwen4ExpForConditionalGeneration";
        e.model_type              = "qwen4_exp";
        e.moe                     = true;
        e.has_mtp_head            = false;
        e.mtp_head_pinned         = true;   // the export writes none
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2560;
        e.n_expert                = 512;
        e.full_attention_interval = 4;
        e.n_layer                 = 12;     // of 48: layers 3, 7, 11 are attention
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "7738fa87cddca8e2";
        e.template_hash           = "12827f24b742ea4e";  // the GGUF's own chat template
        e.tokenizer_hash          = "87a7830d63fcf43b";  // passthrough; vocab == the GGUF's
        e.weights_bytes           = 22613492905ull;
        e.status                  = "measurement artifact: depth 12 of 48, the depth ladder's "
                                    "first rung; not the model's answers";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // THE FUSED-MoE REWRITE OF THE DEPTH-12 RUNG (2026-09-17): the d12
        // artifact above, rewritten in memory by tools/moe_tiled_rewrite.py
        // at tree 0b66c43 and written back with ov.save_model. Not an
        // export: the same u4 codes and zero-points, the scales rounded to
        // f16, the MoE block in the shape the GPU plugin's tiled matcher
        // accepts (two Reshapes, a one-input Swish, an f16 dequant chain with
        // a trailing Convert). The first Flash-Next serving-shape artifact
        // that compiles to moe_3gemm_fused_compressed; at --offload-ratio 99
        // with the CPU tier it is 3.00 GiB device-resident on the 24 GiB
        // card against 17.73 GiB unfused (campaign sub4bit-vram-kernel,
        // status 2026-09-17). Hashes read off the artifact with
        // --inspect-artifact. Superseded by a re-export from the shards.
        ModelEntry e;
        e.id                      = "qwen3.8-flash-next-d12r";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-flash-next-d12r-ov"};
        e.ov_arch                 = "Qwen4ExpForConditionalGeneration";
        e.model_type              = "qwen4_exp";
        e.moe                     = true;
        e.has_mtp_head            = false;
        e.mtp_head_pinned         = true;   // the export writes none
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2560;
        e.n_expert                = 512;
        e.full_attention_interval = 4;
        e.n_layer                 = 12;     // of 48: layers 3, 7, 11 are attention
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "d1d1005332c96fbc";
        e.template_hash           = "12827f24b742ea4e";  // the GGUF's own chat template
        e.tokenizer_hash          = "87a7830d63fcf43b";  // passthrough; vocab == the GGUF's
        e.weights_bytes           = 22141633733ull;
        e.status                  = "measurement artifact: depth 12 of 48, the fused-MoE "
                                    "rewrite of the first rung; not the model's answers";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // SEGMENTED (2026-09-14): ALL 48 layers as a chain of four 12-layer
        // compiled models (tools/export_serving_artifact.py --layers 48
        // --segment-layers 12, tree c00500f): segment0/..segment3/ each hold
        // their own openvino_language_model.{xml,bin}, and the 144 expert
        // bodies live beside them in one expert_bodies.u8 blob the runtime
        // refills one segment at a time (docs/window-051 §2). Allowlisted so a
        // runtime can be pointed at it; NO segmented runtime drives it yet, so
        // its answers are not the model's either way -- admission here is a
        // pin, not a capability claim.
        //
        // lm_xml_sha is NOT a file digest for this entry: a segmented artifact
        // hashes as segplan::chain_arch_hash over every segment's xml sha in
        // segment order (src/core/artifact.cpp), so re-exporting one segment,
        // or reordering the chain, changes it. Read off the directory with
        // `arcint --model <dir> --inspect-artifact`, which prints the chain
        // digest and each segment's own; template/tokenizer are the GGUF's
        // own, identical to the d4 and d12 rungs.
        ModelEntry e;
        e.id                      = "qwen3.8-flash-next-seg12";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-flash-next-seg12-ov"};
        e.ov_arch                 = "Qwen4ExpForConditionalGeneration";
        e.model_type              = "qwen4_exp";
        e.moe                     = true;
        e.has_mtp_head            = false;
        e.mtp_head_pinned         = true;   // the export writes none
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2560;
        e.n_expert                = 512;
        e.full_attention_interval = 4;
        e.n_layer                 = 48;     // the whole model, over four segments
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "32d3060ca30238d1";  // the CHAIN hash, not one file's
        e.template_hash           = "12827f24b742ea4e";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 21953654588ull;  // SUM over the four segment .bins
        e.status                  = "measurement artifact: 48 layers as a 4x12 segment "
                                    "chain; no segmented runtime yet; not the model's answers";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // FULL-DEPTH (2026-09-13): the same serving-shape IR at ALL 48 layers
        // (tools/export_serving_artifact.py --layers 48, tree 092df69): 1,030
        // dense f32 tensors, 144 u4 expert bodies, ~78 GiB .bin. Allowlisted
        // so the served binary can be pointed at it; window-050 §4.10 predicted
        // that neither card holds its 69 GiB of constants and measured, on the
        // A770, that the plugin stages every constant in USM HOST memory at
        // compile and the host runs out first (CL_OUT_OF_HOST_MEMORY at one
        // expert body). Hashes read off the dev-host directory.
        ModelEntry e;
        e.id                      = "qwen3.8-flash-next";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-flash-next-ov"};
        e.ov_arch                 = "Qwen4ExpForConditionalGeneration";
        e.model_type              = "qwen4_exp";
        e.moe                     = true;
        e.has_mtp_head            = false;
        e.mtp_head_pinned         = true;   // inspected 2026-09-13 (the export writes none)
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2560;
        e.n_expert                = 512;
        e.full_attention_interval = 4;
        e.n_layer                 = 48;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "3f574b776a8dd2c6";
        e.template_hash           = "12827f24b742ea4e";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 81948724009ull;
        e.status                  = "full depth, 48 of 48: built; the served-path compile on the A770 was "
                                    "refused by HOST memory (USM host staging of 69 GiB of constants, "
                                    "window-050 §4.10); not servable on this host";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // FULL DEPTH IN THE FUSED SHAPE, THROUGH THE CORRECTED FILL
        // (2026-09-18): the 48-layer serving-shape IR re-exported from tree
        // f91ea73 after the fill's three provenance defects were found and
        // fixed against llama.cpp's own tensors of the same GGUF (DESIGN
        // 7.0.2bz: the converter's folded norm gammas and -exp(A_log)
        // undone at the feed, the sigmoid output gate, the tiled key-head
        // pairing). Its depth-4 sibling agrees with llama.cpp at every cut
        // (layer 3 out corr 0.9987, campaign serving-shape-logits). The
        // first full-depth artifact (d48f, tree da52858, xml f89a1623) is
        // superseded: it served logits with no information about the model
        // (KL 12.4 nats against the model's own capture) and is not admitted
        // any more. Hashes read off the export log.
        ModelEntry e;
        e.id                      = "qwen3.8-flash-next-d48g";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-flash-next-d48g-ov"};
        e.ov_arch                 = "Qwen4ExpForConditionalGeneration";
        e.model_type              = "qwen4_exp";
        e.moe                     = true;
        e.has_mtp_head            = false;
        e.mtp_head_pinned         = true;   // the export writes none
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2560;
        e.n_expert                = 512;
        e.full_attention_interval = 4;
        e.n_layer                 = 48;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "a077e6e4bfa9b847";
        e.template_hash           = "12827f24b742ea4e";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 80061287197ull;
        e.status                  = "full-depth fused-MoE artifact through the corrected fill; "
                                    "the KLD gate against the model's own capture is its acceptance";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // qwen3.8-flash-next-d48n (2026-09-18): the 48-layer serving-shape IR
        // with the checkpoint's OWN expert blocks (IQ3_XXS / IQ4_XS gate-up,
        // IQ4_NL / Q8_0 down, per layer as the GGUF ships them) decoded in
        // standard ops, tree 69dfffd -- the native-format artifact of campaign
        // sub4bit-vram-kernel (design-routing-aware-expert-execution 2.3a-d).
        // Serves through marfrit-openvino +p19 (patch 0043): every routed
        // expert on the CPU tier. Its depth-4 sibling agrees with llama.cpp at
        // layer 0/1/2/3 out corr 0.99991 / 0.99989 / 0.99970 / 0.99953 (the u4
        // repack's d48g sibling: 0.99924 / 0.99918 / - / 0.99873). Hashes read
        // off the export log. d48g stays registered beside it.
        ModelEntry e;
        e.id                      = "qwen3.8-flash-next-d48n";
        e.family                  = "qwen3.8";
        e.artifact_aliases        = {"qwen38-flash-next-d48n-ov"};
        e.ov_arch                 = "Qwen4ExpForConditionalGeneration";
        e.model_type              = "qwen4_exp";
        e.moe                     = true;
        e.has_mtp_head            = false;
        e.mtp_head_pinned         = true;   // the export writes none
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2560;
        e.n_expert                = 512;
        e.full_attention_interval = 4;
        e.n_layer                 = 48;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};   // the registry's coarse label; the experts are the GGUF's IQ3_XXS/IQ4_XS/IQ4_NL/Q8_0
        e.arch_hash               = "641fcb1863f83629";
        e.template_hash           = "12827f24b742ea4e";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 77492280673ull;
        e.status                  = "full-depth artifact with the checkpoint's native expert formats (patch 0043, "
                                    "every routed expert on the CPU tier); the KLD gate against the model's own "
                                    "capture is its acceptance";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    {
        // THE NATIVE `qwen3_5_moe` SERVING-SHAPE RUNG (2026-09-25): the
        // Qwen3.6-35B-A3B serving-shape IR at DEPTH 4 of 40, emitted by
        // tools/export_serving_artifact.py --family qwen35moe --layers 4
        // --expert-format native, so its routed expert bodies are the
        // checkpoint's OWN IQ2_S (gate/up) and IQ3_XXS (down) blocks decoded
        // in standard ops -- the plugin's fourth native format (patch 0050).
        // It is the first admitted artifact of this family that is NOT an HF
        // export: it carries a plain pre-norm residual layer, no
        // hyper-connection and NO PLE / n-gram table. That absence is why
        // the n-gram binding is inert here and --ngram-gguf is not needed;
        // the artifact's config.json declares no n-gram keys, and
        // bind_ngram_ports must not force one (backend_ov.cpp).
        //
        // Hashes and weights_bytes read off the artifact's own manifest,
        // never guessed: `arcint --model <dir> --inspect-artifact` printed
        // arch 391bd21db6368d57 (the single language-model xml), template
        // 55d4931433fe502b, tokenizer 87a7830d63fcf43b, 4,284,499,713 B in one
        // segment -- 2026-09-25. A measurement artifact: 36 of the 40 layers
        // are missing and nothing it says is the model's answer.
        ModelEntry e;
        e.id                      = "qwen3.6-35b-a3b-native-d4";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-35b-a3b-d4n-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = false;  // the export writes none
        e.mtp_head_pinned         = true;   // inspected 2026-09-25
        e.mtp_in_checkpoint       = true;   // config: mtp_num_hidden_layers 1
        e.n_embd                  = 2048;
        e.n_expert                = 256;
        e.full_attention_interval = 4;
        e.n_layer                 = 4;      // of 40: layer 3 is the one attention layer
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};   // the coarse label; the experts are the GGUF's IQ2_S/IQ3_XXS
        e.arch_hash               = "391bd21db6368d57";
        e.template_hash           = "55d4931433fe502b";  // the GGUF's own chat template
        e.tokenizer_hash          = "87a7830d63fcf43b";  // passthrough; vocab == the GGUF's, id for id
        e.weights_bytes           = 4284499713ull;   // the one language-model .bin, off --inspect-artifact
        e.status                  = "measurement artifact: depth 4 of 40, the native-format "
                                    "(IQ2_S/IQ3_XXS) serving-shape rung; no PLE/n-gram table, served "
                                    "inertly; not the model's answers";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        // The FULL-DEPTH (40-layer) native rung, exported 2026-09-25 from the
        // same GGUF and emitter as d4 above. It exists to answer whether the
        // depth-4 artifact's degenerate greedy text is truncation or an
        // emitter defect; it is still a measurement artifact (native experts,
        // no PLE/n-gram table), and no quality claim rides on it.
        //
        // Hashes and weights_bytes read off its own serving-shape.json, not
        // guessed: arch b94ecc6ab6b200ac (the single language-model xml),
        // template 55d4931433fe502b, tokenizer 87a7830d63fcf43b, lm .bin
        // 23,429,144,641 B in one segment -- 2026-09-25. The 120 expert
        // bodies are the GGUF's own blocks (census: 40 IQ2_S gate + 40 IQ2_S
        // up + 37 IQ3_XXS down + 3 IQ4_XS down folded onto IQ4_NL).
        ModelEntry e;
        e.id                      = "qwen3.6-35b-a3b-native-d40";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-35b-a3b-d40n-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = false;  // the export writes none
        e.mtp_head_pinned         = true;   // inspected 2026-09-25
        e.mtp_in_checkpoint       = true;   // config: mtp_num_hidden_layers 1
        e.n_embd                  = 2048;
        e.n_expert                = 256;
        e.full_attention_interval = 4;
        e.n_layer                 = 40;     // 30 GDN + 10 attention (i % 4 == 3)
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};   // the coarse label; the experts are the GGUF's IQ2_S/IQ3_XXS/IQ4_XS
        e.arch_hash               = "b94ecc6ab6b200ac";
        e.template_hash           = "55d4931433fe502b";  // the GGUF's own chat template
        e.tokenizer_hash          = "87a7830d63fcf43b";  // passthrough; vocab == the GGUF's, id for id
        e.weights_bytes           = 23429144641ull;  // the one language-model .bin, off --inspect-artifact
        e.status                  = "measurement artifact: the full-depth (40-layer) native-format "
                                    "(IQ2_S/IQ3_XXS/IQ4_XS) serving-shape rung; no PLE/n-gram table, "
                                    "served inertly";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }
    {
        // The same 40-layer artifact with the dense/graph part stored f16
        // (`--dense-fp16`, 2026-09-25): lm .bin 19,482,424,091 B against the
        // f32 form's 23,429,144,641. The native expert bodies are u8/f16
        // already and byte-identical (sampled readback, same leg). A size
        // lever toward the all-resident fit; no quality claim here.
        ModelEntry e;
        e.id                      = "qwen3.6-35b-a3b-native-d40f16";
        e.family                  = "qwen3.6";
        e.artifact_aliases        = {"qwen36-35b-a3b-d40f16-ov"};
        e.ov_arch                 = "Qwen3_5MoeForConditionalGeneration";
        e.model_type              = "qwen3_5_moe";
        e.moe                     = true;
        e.has_mtp_head            = false;
        e.mtp_head_pinned         = true;
        e.mtp_in_checkpoint       = true;
        e.n_embd                  = 2048;
        e.n_expert                = 256;
        e.full_attention_interval = 4;
        e.n_layer                 = 40;
        e.n_ctx_train             = 262144;
        e.quants                  = {Quant::Q4};
        e.arch_hash               = "43d2e607941c77ea";   // its own lm xml, off --inspect-artifact
        e.template_hash           = "55d4931433fe502b";
        e.tokenizer_hash          = "87a7830d63fcf43b";
        e.weights_bytes           = 19482424091ull;
        e.status                  = "measurement artifact: the full-depth (40-layer) native-format "
                                    "(IQ2_S/IQ3_XXS/IQ4_XS) serving-shape rung, dense stored f16; "
                                    "no PLE/n-gram table, served inertly";
        e.sampler = qwen_card_defaults();
        split_layers(e);
        r.push_back(std::move(e));
    }

    return r;
}

void check_int(ValidationResult& res, const char* field, int pinned, int seen) {
    if (pinned == 0) {
        res.warnings.push_back(
            log::format("%s not pinned in the allowlist; artifact reports %d", field, seen));
        return;
    }
    if (pinned != seen) {
        res.errors.push_back(
            log::format("%s mismatch: allowlist %d, artifact %d", field, pinned, seen));
    }
}

void check_hash(ValidationResult& res, const char* field, const std::string& pinned,
                const std::string& seen) {
    if (pinned.empty()) {
        res.warnings.push_back(log::format(
            "%s not pinned in the allowlist; artifact reports %s", field,
            seen.empty() ? "nothing" : seen.c_str()));
        return;
    }
    if (seen.empty()) {
        res.errors.push_back(log::format("%s missing from artifact, allowlist pins %s", field,
                                         pinned.c_str()));
        return;
    }
    if (pinned != seen) {
        res.errors.push_back(log::format("%s mismatch: allowlist %s, artifact %s", field,
                                         pinned.c_str(), seen.c_str()));
    }
}

// A pinned byte count is the same kind of contract as a pinned hash: the entry
// says what the artifact IS. Before 2026-09-25 the allowlist pinned
// weights_bytes and nothing read it, so a wrong or re-exported .bin passed
// admission on the strength of its xml hash alone. A zero on either side is
// "not pinned" / "not reported" and is named rather than silently compared.
void check_u64(ValidationResult& res, const char* field, uint64_t pinned, uint64_t seen) {
    if (pinned == 0) {
        res.warnings.push_back(
            log::format("%s not pinned in the allowlist; artifact reports %llu", field,
                        static_cast<unsigned long long>(seen)));
        return;
    }
    if (seen == 0) {
        res.errors.push_back(log::format("%s missing from artifact, allowlist pins %llu", field,
                                         static_cast<unsigned long long>(pinned)));
        return;
    }
    if (pinned != seen) {
        res.errors.push_back(log::format("%s mismatch: allowlist %llu, artifact %llu", field,
                                         static_cast<unsigned long long>(pinned),
                                         static_cast<unsigned long long>(seen)));
    }
}

}  // namespace

const char* quant_name(Quant q) {
    switch (q) {
        case Quant::Q4: return "q4";
        case Quant::Q8: return "q8";
    }
    return "?";
}

std::optional<Quant> quant_parse(std::string_view s) {
    if (s == "q4" || s == "int4") return Quant::Q4;
    if (s == "q8" || s == "int8") return Quant::Q8;
    return std::nullopt;
}

bool ModelEntry::accepts(Quant q) const {
    return std::find(quants.begin(), quants.end(), q) != quants.end();
}

bool ModelEntry::matches_alias(std::string_view artifact_dir) const {
    return std::find(artifact_aliases.begin(), artifact_aliases.end(), artifact_dir) !=
           artifact_aliases.end();
}

const std::vector<ModelEntry>& registry() {
    static const std::vector<ModelEntry> kRegistry = build_registry();
    return kRegistry;
}

const ModelEntry* find_model(std::string_view id) {
    for (const ModelEntry& e : registry()) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

const ModelEntry* find_by_artifact(std::string_view artifact_dir) {
    for (const ModelEntry& e : registry()) {
        if (e.matches_alias(artifact_dir)) return &e;
    }
    return nullptr;
}

std::vector<std::string> model_ids() {
    std::vector<std::string> ids;
    ids.reserve(registry().size());
    for (const ModelEntry& e : registry()) ids.push_back(e.id);
    return ids;
}

ValidationResult validate_artifact(const ModelEntry& entry, const ArtifactInfo& seen) {
    ValidationResult res;

    if (entry.id != seen.id) {
        res.errors.push_back(
            log::format("id mismatch: allowlist %s, artifact %s", entry.id.c_str(),
                        seen.id.c_str()));
    }
    if (!entry.accepts(seen.quant)) {
        res.errors.push_back(log::format("quant %s not allowed for %s", quant_name(seen.quant),
                                         entry.id.c_str()));
    }
    if (entry.has_mtp_head != seen.has_mtp_head) {
        const std::string msg = log::format("MTP head mismatch: allowlist %s, artifact %s",
                                            entry.has_mtp_head ? "present" : "absent",
                                            seen.has_mtp_head ? "present" : "absent");
        // Only a refusal when the claim has a source. The allowlist's MTP flag
        // is currently design prose, and prose must not block a real artifact.
        if (entry.mtp_head_pinned) {
            res.errors.push_back(msg);
        } else {
            res.warnings.push_back(msg + " (allowlist value is unpinned; artifact wins)");
        }
    }

    check_int(res, "n_layer", entry.n_layer, seen.n_layer);
    check_int(res, "n_gdn_layer", entry.n_gdn_layer, seen.n_gdn_layer);
    check_int(res, "n_attn_layer", entry.n_attn_layer, seen.n_attn_layer);
    check_int(res, "n_ctx_train", entry.n_ctx_train, seen.n_ctx_train);

    check_hash(res, "arch_hash", entry.arch_hash, seen.arch_hash);
    check_hash(res, "template_hash", entry.template_hash, seen.template_hash);
    check_hash(res, "tokenizer_hash", entry.tokenizer_hash, seen.tokenizer_hash);
    check_u64(res, "weights_bytes", entry.weights_bytes, seen.weights_bytes);

    if (seen.n_layer > 0 && seen.n_gdn_layer + seen.n_attn_layer != seen.n_layer) {
        res.errors.push_back(log::format("layer split does not sum: %d GDN + %d attn != %d",
                                         seen.n_gdn_layer, seen.n_attn_layer, seen.n_layer));
    }

    res.ok = res.errors.empty();
    return res;
}

}  // namespace lgc
