// Patch 0121's device-free cell: the KV stream reaches the cache and the graph, and
// stays inert with the flag off. Design: docs/campaigns/kv-streaming-design.md,
// increment I3's prerequisite.
//
// A stream-on run cannot be made here: arcint's llama engine constructs its OpenCL
// cards at start and throws without one (src/exec/backend_llama.cpp), and a GGUF
// model is never committed. So this cell reads the patched tree's source and checks
// the wiring's shape -- the parts a mutation would break before a run could:
//  - the context parameter defaults to 0 (off) and is copied into cparams;
//  - the cache only sizes the pool under `stream_resident_bytes > 0`, validates the
//    paged kernel's shapes (q8_0 K / q4_0 V, head 256, one stream, no window/SWA),
//    shrinks the K/V tensors to `stream_n_slots*page_size` rows, and constructs the
//    stream with its host store and ring slots;
//  - every ubatch's `apply_ubatch` places the pages through `prepare` before the
//    graph;
//  - the graph builds the page table input and per-layer ring views, and sets them
//    on the fused FLASH_ATTN node with the layer's table row;
//  - the table input is filled from the stream in `set_input`, and a moved layout
//    refuses graph reuse.
//
// The behavioural evidence that pages and the table read the same bytes as an
// unpaged run is tests/test_kv_stream_map.cpp (green) and its reds
// (kv-stream-red2 eviction, kv-stream-red3 a one-step page-table shift).
//
// Red case for this cell (a one-line mutation, run before the patch was accepted):
// the default in llama_context_default_params' `kv_resident_size` set to 1 -- the
// first check fails, and a normal start would then require a stream it was never
// asked for.

#include "llama-kv-stream.h" // the patch's presence and page size

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond, what)                                                              \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            printf("FAIL: %s\n", what);                                                \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        printf("FAIL: cannot read %s\n", path.c_str());
        ++g_failures;
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool has(const std::string & text, const std::string & needle) {
    return text.find(needle) != std::string::npos;
}

} // namespace

int main() {
#ifndef ARCINT_KV_STREAM_SRC_DIR
    printf("SKIP: no ARCINT_KV_STREAM_SRC_DIR\n");
    return 0;
#else
    const std::string dir = ARCINT_KV_STREAM_SRC_DIR;

    const std::string llama_h   = read_file(dir + "/../include/llama.h");
    const std::string cparams_h = read_file(dir + "/llama-cparams.h");
    const std::string context   = read_file(dir + "/llama-context.cpp");
    const std::string cache_c   = read_file(dir + "/llama-kv-cache.cpp");
    const std::string cache_h   = read_file(dir + "/llama-kv-cache.h");
    const std::string graph_c   = read_file(dir + "/llama-graph.cpp");
    const std::string graph_h   = read_file(dir + "/llama-graph.h");

    // 1. the parameter exists and defaults off
    CHECK(has(llama_h, "uint32_t kv_resident_size"), "include/llama.h has kv_resident_size");
    CHECK(has(cparams_h, "kv_resident_size"), "llama-cparams.h carries kv_resident_size");
    CHECK(has(context, "/*.kv_resident_size            =*/ 0,"), "the context default is 0 (off)");
    CHECK(has(context, "cparams.kv_resident_size = params.kv_resident_size;"), "cparams take the parameter");

    // 2. the cache sizes the pool only under the flag, with the kernel's shapes
    CHECK(has(cache_c, "if (stream_resident_bytes > 0) {"), "the pool is sized only with the flag on");
    CHECK(has(cache_c, "q8_0 K cache and a q4_0 V cache"), "the cache refuses other KV types");
    CHECK(has(cache_c, "head size of 256"), "the cache refuses other head sizes");
    CHECK(has(cache_c, "stream_n_rows = stream_n_slots*llama_kv_stream::page_size;"), "the tensors shrink to the pool");
    CHECK(has(cache_c, "ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, stream_n_rows, n_stream)"), "K rows are the pool's");
    CHECK(has(cache_c, "std::make_unique<llama_kv_stream>(sl, stream_n_slots, stream_n_ring, kv_size, stream_buft)"),
          "the stream is constructed with its host store and ring");

    // 3. every ubatch places its pages before the graph
    CHECK(has(cache_c, "stream->prepare(sinfo.idxs[0], get_n_kv(sinfo));"), "apply_ubatch prepares the pages");

    // 4. the graph builds the table and the ring views, and sets them on the FA node
    CHECK(has(graph_c, "inp->self_kv_pages = mctx_cur->build_input_kv_pages(ctx0);"), "the graph builds the page table input");
    CHECK(has(graph_c, "inp->self_kv_ring_k[il] = mctx_cur->build_ring_k(ctx0, il);"), "the graph builds the ring views");
    CHECK(has(graph_c, "ggml_flash_attn_ext_set_pages(fa.tensor, inp->self_kv_pages"), "the FA node gets the page table");
    CHECK(has(graph_c, "mctx_cur->kv_layer_index(il)"), "the FA node gets the layer's table row");
    CHECK(has(graph_c, "mctx->set_input_kv_pages(self_kv_pages);"), "set_input fills the table");
    CHECK(has(graph_c, "if (mctx->kv_stream_on()) {\n        res = false;"), "a stream-on graph is not reused");
    CHECK(has(graph_h, "ggml_tensor * self_kv_pages = nullptr;"), "the input struct carries the table");

    // 5. the decode views read the pool, not the logical context
    CHECK(has(cache_c, "if (stream) {\n        n_kv = k->ne[1];"), "get_k reads the whole pool");

    if (g_failures != 0) {
        printf("FAILED: %d\n", g_failures);
        return 1;
    }
    printf("OK: the KV stream's cache and graph wiring is present and off by default\n");
    return 0;
#endif
}
