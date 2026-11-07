// A weight the card declines at load, placed in host memory, is reported (contrib/llama.cpp
// patch 0111). Before it, Dory's 217 BF16 scale vectors went to CPU buffers on a build
// whose OpenCL CPY refused a BF16 source, decode fell from 23.7 to 14.7 t/s on the A770,
// and no DOWNGRADE line said so: the device declines inside the loader's probe
// (weight_buft_supported), before any backend path that 0100 counts.
//
// The rule and the key are the patch's own header (src/llama-placement-report.h), which
// the loader calls; the loader's call site is checked as text, since a unit build has no
// device that declines. The card case is in the README section of 0111.
//
// Red cases (each a one-line edit; the named test then fails):
//   P1 weight_pattern() returning the name unchanged -> a_layer_number_becomes_N
//   P2 declined_to_host() ignoring where the list starts (token_embd and the layers left
//      on the CPU would report) -> the_cpu_extra_buffer_types_are_not_device_memory
//   P3 the loader's `probed = true` moved to the override branch, or the report placed
//      before the mmap swap (the key would name the host buffer type the swap replaces)
//      -> the_loader_reports_only_probe_selections_after_the_mmap_swap
//   P4 collect_downgrades() reading the first source only (the 0100 text alone) ->
//      both_sources_reach_props
//   P5 read_downgrade_text() without the cut of a partial last line ->
//      a_cut_last_line_does_not_join_the_next_source
//   P7 in_device_memory() on is_host alone (0111's first form, measured on the cards: the CPU's
//      extra buffer type CPU_REPACK leaves is_host unset, heads the input layer's list, and every
//      token_embd and per_layer_token_embd reported) -> the_cpu_extra_buffer_types_are_not_device_memory,
//      a_device_decline_reports_wherever_on_the_host_it_lands; the input-layer guard removed from
//      the loader -> the_loader_reports_only_probe_selections_after_the_mmap_swap
//   P6 patch 0112 left out (LLM_TENSOR_PLE_CONV1D probed as SSM_CONV again: Flash-Next's f16
//      kernel back in a CPU buffer, the 0111 key back) -> the_ple_conv_kernel_is_probed_as_its_cast
#include "exec/backend.h"

#include "harness.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#ifdef ARCINT_LLAMA_PLACEMENT
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-placement-report.h"
#endif

namespace {

std::string g_first;
std::string g_second;
size_t      g_grow = 0;   // bytes the first source gains between its two calls

size_t write_text(const std::string& text, char* buf, size_t size) {
    if (buf != nullptr && size > 0) std::snprintf(buf, size, "%s", text.c_str());
    return text.size();
}
size_t first_source(char* buf, size_t size) {
    if (buf == nullptr) return g_first.size();
    return write_text(g_first + std::string(g_grow, 'x'), buf, size);
}
size_t second_source(char* buf, size_t size) { return write_text(g_second, buf, size); }

}  // namespace

TEST(both_sources_reach_props) {
    g_first  = "src=opencl op=MUL_MAT type=f32 k=512 cols=16+ path=generic reason=no-fast-path\t41\n";
    g_second = "src=llama op=CPY type=bf16 weight=blk.N.residual_alpha.weight buft=CPU reason=device-declines\t72\n";
    g_grow   = 0;
    const auto got = lgc::collect_downgrades({ first_source, second_source });
    CHECK_EQ(got.size(), size_t{2});
    if (got.size() != 2) return;
    CHECK_EQ(got[0].first, std::string("src=opencl op=MUL_MAT type=f32 k=512 cols=16+ path=generic reason=no-fast-path"));
    CHECK_EQ(got[0].second, uint64_t{41});
    CHECK_EQ(got[1].first, std::string("src=llama op=CPY type=bf16 weight=blk.N.residual_alpha.weight buft=CPU reason=device-declines"));
    CHECK_EQ(got[1].second, uint64_t{72});
    // a source that reports nothing adds nothing; a missing one neither
    g_second = "";
    CHECK_EQ(lgc::collect_downgrades({ first_source, second_source, nullptr }).size(), size_t{1});
}

TEST(a_cut_last_line_does_not_join_the_next_source) {
    // the first source's text grows between the length query and the read: the read is cut
    // inside its new last line, and that partial line must not run into the second source
    g_first  = "src=opencl a=1\t3\n";
    g_second = "src=llama b=2\t5\n";
    g_grow   = 6;   // "src=opencl a=1\t3\nxxxxxx" cut at the first text's length + 1
    g_first += "src=opencl c=3\t4";   // the length query sees this partial line too
    const auto got = lgc::collect_downgrades({ first_source, second_source });
    CHECK_EQ(got.size(), size_t{2});
    if (got.size() != 2) return;
    CHECK_EQ(got[0].first, std::string("src=opencl a=1"));
    CHECK_EQ(got[1].first, std::string("src=llama b=2"));
    CHECK_EQ(got[1].second, uint64_t{5});
}

#ifdef ARCINT_LLAMA
TEST(the_placement_report_is_built) {
    // a libllama build of the series carries 0111: without it a weight the card declines at
    // load lands in host memory with no line and no count, which is the gap this closes
#ifndef ARCINT_LLAMA_PLACEMENT
    t::fail(__FILE__, __LINE__, "ARCINT_LLAMA_DIR lacks contrib/llama.cpp patch 0111 (llama_placement_downgrades)");
#endif
}
#endif

#ifdef ARCINT_LLAMA_PLACEMENT
TEST(a_layer_number_becomes_N) {
    using llama_placement::weight_pattern;
    CHECK_EQ(weight_pattern("blk.17.residual_alpha.weight"), std::string("blk.N.residual_alpha.weight"));
    CHECK_EQ(weight_pattern("blk.0.ple_conv1d.weight"), std::string("blk.N.ple_conv1d.weight"));
    CHECK_EQ(weight_pattern("v.blk.3.attn_q.weight"), std::string("v.blk.N.attn_q.weight"));
    CHECK_EQ(weight_pattern("prism.hadamard.12"), std::string("prism.hadamard.N"));
    CHECK_EQ(weight_pattern("7.x"), std::string("N.x"));
    // digits inside a part are the part's name, not a layer
    CHECK_EQ(weight_pattern("blk.4.ffn_up_exps.weight"), std::string("blk.N.ffn_up_exps.weight"));
    CHECK_EQ(weight_pattern("output_hc_down.weight"), std::string("output_hc_down.weight"));
    CHECK_EQ(weight_pattern("conv2d.weight"), std::string("conv2d.weight"));
    CHECK_EQ(weight_pattern("token_embd.weight"), std::string("token_embd.weight"));
    CHECK_EQ(weight_pattern(""), std::string(""));
    CHECK_EQ(weight_pattern("a..1."), std::string("a..N."));
    // the count text is tab-separated: a tab or a space in a name cannot split a key
    CHECK_EQ(weight_pattern("odd\tname 2.1"), std::string("odd_name_2.N"));
}

TEST(only_device_memory_counts_as_the_device) {
    using llama_placement::in_device_memory;
    // a card's own buffer types: an OpenCL or a split buffer, an integrated GPU's
    CHECK(in_device_memory(false, GGML_BACKEND_DEVICE_TYPE_GPU));
    CHECK(in_device_memory(false, GGML_BACKEND_DEVICE_TYPE_IGPU));
    CHECK(in_device_memory(false, GGML_BACKEND_DEVICE_TYPE_META));
    // a card's pinned host buffer type is host memory
    CHECK(!in_device_memory(true, GGML_BACKEND_DEVICE_TYPE_GPU));
    // the CPU's buffer type, and its extra buffer types, which leave is_host unset (repacked data)
    CHECK(!in_device_memory(true, GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!in_device_memory(false, GGML_BACKEND_DEVICE_TYPE_CPU));
    CHECK(!in_device_memory(false, GGML_BACKEND_DEVICE_TYPE_ACCEL));
    CHECK(!in_device_memory(nullptr));
}

TEST(the_cpu_extra_buffer_types_are_not_device_memory) {
    // the buffer types that head the input layer's list on the cards (make_cpu_buft_list with
    // extra buffer types on): the CPU's own and its extra ones, read from the CPU backend the way
    // the loader gets them. CPU_REPACK is among them on x86 and reports is_host false; it is the
    // one 0111's first rule took for device memory
    ggml_backend_reg_t cpu = ggml_backend_cpu_reg();
    CHECK(cpu != nullptr);
    if (cpu == nullptr) return;
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(cpu, 0);
    CHECK(!llama_placement::in_device_memory(ggml_backend_cpu_buffer_type()));
    // a proc address to a function pointer, without the object-to-function cast -Wpedantic flags
    void*                             addr     = ggml_backend_reg_get_proc_address(cpu, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_dev_get_extra_bufts_t extra_fn = nullptr;
    static_assert(sizeof(addr) == sizeof(extra_fn), "pointer sizes");
    std::memcpy(&extra_fn, &addr, sizeof(addr));
    CHECK(extra_fn != nullptr);
    if (extra_fn == nullptr) return;
    size_t n_extra = 0, n_not_host = 0;
    for (ggml_backend_buffer_type_t* b = extra_fn(dev); b != nullptr && *b != nullptr; ++b) {
        ++n_extra;
        n_not_host += !ggml_backend_buft_is_host(*b);
        CHECK(!llama_placement::in_device_memory(*b));
        // the input layer's list headed by it, the CPU buffer chosen: no report
        CHECK(!llama_placement::declined_to_host(*b, ggml_backend_cpu_buffer_type()));
    }
    // the case must exist in this build, or this test proves nothing about it
    CHECK(n_extra >= 1);
    CHECK(n_not_host >= 1);
}

TEST(a_device_decline_reports_wherever_on_the_host_it_lands) {
    // a stand-in device buffer type: the rule reads only is_host and the owner's type, through
    // the decisions above; with no card in a unit build the device side is the boolean form
    using llama_placement::in_device_memory;
    const bool dev_first = in_device_memory(false, GGML_BACKEND_DEVICE_TYPE_GPU);
    CHECK(dev_first);
    // the CPU buffer and an extra buffer type (CPU_REPACK takes a Q4_0 product the card declines)
    // are both outside device memory: a decline into either reports
    CHECK(!in_device_memory(ggml_backend_cpu_buffer_type()));
    CHECK(!in_device_memory(false, GGML_BACKEND_DEVICE_TYPE_CPU));
    // a list that starts outside device memory (the input layer, layers not offloaded) never does
    CHECK(!llama_placement::declined_to_host(ggml_backend_cpu_buffer_type(), ggml_backend_cpu_buffer_type()));
}

TEST(the_key_names_op_type_weight_and_buffer) {
    CHECK_EQ(llama_placement::key("CPY", "bf16", "blk.17.residual_alpha.weight", "CPU"),
             std::string("op=CPY type=bf16 weight=blk.N.residual_alpha.weight buft=CPU reason=device-declines"));
    CHECK_EQ(llama_placement::key("SSM_CONV", "f16", "blk.40.ple_conv1d.weight", "CPU"),
             std::string("op=SSM_CONV type=f16 weight=blk.N.ple_conv1d.weight buft=CPU reason=device-declines"));
}

TEST(a_key_logs_once_and_counts_every_weight) {
    llama_placement::registry r;
    const std::string a = llama_placement::key("CPY", "bf16", "blk.1.attn_qk_scale.weight", "CPU");
    const std::string b = llama_placement::key("CPY", "bf16", "output_scale.weight", "CPU");
    CHECK(r.add(a));          // the first weight of a key: the caller logs the line
    CHECK(!r.add(llama_placement::key("CPY", "bf16", "blk.2.attn_qk_scale.weight", "CPU")));
    CHECK(r.add(b));
    CHECK(!r.add(a));
    // keys sorted, "src=llama " in front, the format lgc::parse_downgrades reads
    CHECK_EQ(r.text(), std::string("src=llama op=CPY type=bf16 weight=blk.N.attn_qk_scale.weight buft=CPU reason=device-declines\t3\n"
                                   "src=llama op=CPY type=bf16 weight=output_scale.weight buft=CPU reason=device-declines\t1\n"));
    const auto got = lgc::parse_downgrades(r.text());
    CHECK_EQ(got.size(), size_t{2});
    if (got.size() == 2) CHECK_EQ(got[0].second, uint64_t{3});
    CHECK_EQ(llama_placement::registry{}.text(), std::string());
}

#ifdef ARCINT_LLAMA_SRC_DIR
TEST(the_loader_reports_only_probe_selections_after_the_mmap_swap) {
    std::ifstream f(std::string(ARCINT_LLAMA_SRC_DIR) + "/src/llama-model-loader.cpp");
    CHECK(f.good());
    const std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t override_sel = src.find("buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);");
    const size_t probe_sel    = src.find("buft = select_weight_buft(hparams, t_meta, op, buft_list);");
    const size_t probed_set   = src.find("probed = true;");
    const size_t swap         = src.find("// avoid using a host buffer when using mmap");
    const size_t report       = src.find("llama_placement::declined_to_host(");
    const size_t moved        = src.find("if (buft != buft_list->front().second) {");
    CHECK(override_sel != std::string::npos && probe_sel != std::string::npos && probed_set != std::string::npos);
    CHECK(swap != std::string::npos && report != std::string::npos && moved != std::string::npos);
    // set only after the probe's own selection, not the override's (which comes first)
    CHECK(override_sel < probe_sel && probe_sel < probed_set && probed_set < swap);
    CHECK_EQ(src.find("probed = true;", probed_set + 1), std::string::npos);
    // the report reads the buffer type the mmap swap leaves, and is guarded by the probe and
    // by a real load (a memory-fit pass places nothing)
    CHECK(swap < report && report < moved);
    const std::string guard = src.substr(src.rfind("if (", report), report - src.rfind("if (", report));
    CHECK(guard.find("probed") != std::string::npos);
    CHECK(guard.find("!no_alloc") != std::string::npos);
    // the input layer is the CPU list by design: never reported, whatever heads that list
    CHECK(guard.find("info.layer != LLM_TENSOR_LAYER_INPUT") != std::string::npos);
    // the rule gets the buffer types themselves (device memory by owner, not by is_host)
    CHECK(src.find("llama_placement::declined_to_host(buft_list->front().second, buft)", report) == report);
    CHECK(src.find("LLAMA_LOG_WARN(\"DOWNGRADE src=llama %s\\n\"", report) != std::string::npos);
}

TEST(the_ple_conv_kernel_is_probed_as_its_cast) {
    // 0112: the loader probes a weight with its tensor's op (LLM_TENSOR_INFOS); Flash-Next's
    // f16 PLE conv kernel was probed as SSM_CONV, which the card declines for an f16 weight,
    // while qwen4exp's graph only takes column views of it and casts them (CONT, CPY), which the
    // card runs. The probe must be the cast, and the graph must still read the kernel that way.
    std::ifstream fa(std::string(ARCINT_LLAMA_SRC_DIR) + "/src/llama-arch.cpp");
    std::ifstream fm(std::string(ARCINT_LLAMA_SRC_DIR) + "/src/models/qwen4exp.cpp");
    CHECK(fa.good() && fm.good());
    const std::string arch((std::istreambuf_iterator<char>(fa)), std::istreambuf_iterator<char>());
    const std::string model((std::istreambuf_iterator<char>(fm)), std::istreambuf_iterator<char>());
    const size_t info = arch.find("{LLM_TENSOR_PLE_CONV1D,");
    CHECK(info != std::string::npos);
    if (info == std::string::npos) return;
    const std::string line = arch.substr(info, arch.find('\n', info) - info);
    CHECK(line.find("GGML_OP_CPY") != std::string::npos);
    // every read of the kernel in the graph is the column view, and the view is cast
    const std::string use = "model.layers[il].ple_conv1d";
    size_t n = 0, viewed = 0;
    for (size_t at = model.find(use); at != std::string::npos; at = model.find(use, at + 1)) {
        ++n;
        const size_t view = model.rfind("ggml_view_2d(", at);
        viewed += view != std::string::npos && at - view < 160;
    }
    CHECK(n >= 1);
    CHECK_EQ(viewed, n);
    CHECK(model.find("wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);") != std::string::npos);
    CHECK_EQ(model.find("ggml_ssm_conv(ctx0, padded"), std::string::npos);
}
#endif
#endif
