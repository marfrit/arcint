// 0079: device-free tests for the DG2 + LEO image1d_buffer bound.
//   - the rule itself (ggml-opencl-image-limit.h, the header the backend
//     includes): every combination of LEO, DG2 and driver max, including the
//     16,384 / 16,385 boundary the abbreviation sits at;  the default driver
//     and the B60 stay permissive at the same sizes, so no default image is
//     refused;
//   - the two predicates: the LEO device-version suffix and the DG2 model name;
//   - text checks over ggml-opencl.cpp: the mxfp4 weight image and the image
//     pool (flash-attention's K view) are both asked before creation, and the
//     selection site that would take an image kernel only does so when the
//     image is still non-null; the FA site falls back to the buffer kernel on a
//     skipped image.
// No GPU, no OpenCL runtime.  A mutation of the header is a mutation of the
// backend's creation rule, and a mutation of either guard is caught here.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "ggml-opencl-image-limit.h"

static int g_bad = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL: %s\n", what); ++g_bad; } } while (0)

static ggml_cl_image_mode mode(int leo, int dg2, size_t driver_max) {
    ggml_cl_image_mode m = { leo, dg2, driver_max };
    return m;
}

static void check_rule(void) {
    // the driver's own bound is honoured on every device
    CHECK(!ggml_cl_image_allowed(mode(0, 0, 16384), 16385), "driver max: 16385 above 16384");
    CHECK(ggml_cl_image_allowed(mode(0, 0, 16384), 16384), "driver max: 16384 fits");
    CHECK(ggml_cl_image_allowed(mode(0, 0, 0), (size_t) 1 << 20), "driver max 0: no driver bound");

    // the default driver on the A770 (no LEO): images of every size as before
    CHECK(ggml_cl_image_allowed(mode(0, 1, 0), 16385), "A770 default driver: 16385 allowed");
    CHECK(ggml_cl_image_allowed(mode(0, 1, 0), 1036800), "A770 default driver: 1036800 allowed");

    // the B60 in LEO (DG2 false): images of every size as before
    CHECK(ggml_cl_image_allowed(mode(1, 0, 0), 16385), "B60 LEO: 16385 allowed");
    CHECK(ggml_cl_image_allowed(mode(1, 0, 0), 1036800), "B60 LEO: 1036800 allowed");

    // the A770 in LEO: the 14-bit width field's exact boundary
    CHECK(ggml_cl_image_allowed(mode(1, 1, 0), 1), "A770 LEO: one texel");
    CHECK(ggml_cl_image_allowed(mode(1, 1, 0), 16384), "A770 LEO: 16384 fits");
    CHECK(!ggml_cl_image_allowed(mode(1, 1, 0), 16385), "A770 LEO: 16385 refused");
    CHECK(!ggml_cl_image_allowed(mode(1, 1, 0), 1036800), "A770 LEO: a real weight image refused");
    CHECK(!ggml_cl_image_allowed(mode(1, 1, 0), 0), "zero texels is never an image");

    // a wider driver bound does not lift the DG2 + LEO bound, and a driver
    // bound below it still wins
    CHECK(!ggml_cl_image_allowed(mode(1, 1, 65536), 16385), "A770 LEO: driver max 65536 does not lift the bound");
    CHECK(!ggml_cl_image_allowed(mode(1, 1, 8192), 8193), "A770 LEO: driver max 8192 wins");
    CHECK(!ggml_cl_image_allowed(mode(1, 1, 8192), 16384), "A770 LEO: driver max below the DG2 bound refuses 16384");
    CHECK(ggml_cl_image_allowed(mode(1, 1, 8192), 8192), "A770 LEO: 8192 fits the driver max");
}

static void check_predicates(void) {
    CHECK(ggml_cl_version_is_leo("OpenCL 3.0 LEO") == 1, "LEO version suffix");
    CHECK(ggml_cl_version_is_leo("OpenCL 3.0 NEO") == 0, "default driver version is not LEO");
    CHECK(ggml_cl_version_is_leo("OpenCL 3.0") == 0, "bare version is not LEO");
    CHECK(ggml_cl_version_is_leo(NULL) == 0, "null version");

    CHECK(ggml_cl_name_is_dg2("Intel(R) Arc(TM) A770 Graphics") == 1, "A770 name");
    CHECK(ggml_cl_name_is_dg2("Intel(R) Arc(TM) A750 Graphics") == 1, "A750 name");
    CHECK(ggml_cl_name_is_dg2("Intel(R) Arc(TM) Pro A60 Graphics") == 1, "Pro A60 name");
    CHECK(ggml_cl_name_is_dg2("Intel(R) Arc(TM) Pro B60 Graphics") == 0, "B60 is not DG2");
    CHECK(ggml_cl_name_is_dg2("Adreno (TM) 740") == 0, "Adreno is not DG2");
    CHECK(ggml_cl_name_is_dg2(NULL) == 0, "null name");
}

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream b;
    b << f.rdbuf();
    return b.str();
}

static bool has(const std::string & text, const char * needle) {
    return text.find(needle) != std::string::npos;
}

static void check_source(const char * path) {
    const std::string src = read_file(path);
    CHECK(!src.empty(), "ggml-opencl.cpp reads");
    if (src.empty()) {
        return;
    }

    // the header and the one call site are part of the backend
    CHECK(has(src, "#include \"ggml-opencl-image-limit.h\""), "the backend includes the header");
    CHECK(has(src, "ggml_cl_image_texels_ok(backend_ctx, q_img_texels)"),
          "mxfp4 image creation asks the bound");

    // mxfp4: the image is created only inside the allowed branch, and the
    // refused branch leaves q_img null (the Intel kernels read the q buffer)
    const std::string mxfp4_anchor = "// Create image for Q, unless the DG2 + LEO bound refuses it.";
    const size_t mx = src.find(mxfp4_anchor);
    CHECK(mx != std::string::npos, "mxfp4 image site has the guard comment");
    if (mx != std::string::npos) {
        const std::string block = src.substr(mx, 1400);
        CHECK(has(block, "if (ggml_cl_image_texels_ok(backend_ctx, q_img_texels))"), "mxfp4 creates only when allowed");
        CHECK(has(block, "extra->q_img = nullptr;"), "mxfp4 refused path leaves q_img null");
        CHECK(has(block, "extra->q_img = clCreateImage"), "mxfp4 allowed path still creates the image");
    }

    // the image pool: FA's K view is an image1d_buffer too, and the pool's
    // nullptr answer makes the caller take its buffer kernel
    // the pool asks the bound before it creates the sub-buffer: a refusal after
    // clCreateSubBuffer would return without releasing it (a handle per call)
    const size_t pq = src.find("ggml_cl_image_texels_ok(backend_ctx, required_bytes / pixel_size)");
    CHECK(pq != std::string::npos, "the image pool asks the bound");
    if (pq != std::string::npos) {
        const size_t sb = src.find("clCreateSubBuffer(data_device, 0,", pq > 2000 ? pq - 2000 : 0);
        CHECK(sb != std::string::npos && sb > pq, "the image pool asks the bound before creating the sub-buffer");
    }

    // a selection site that assumes q_img must still see it: both weight-image
    // MoE selections read a live image, and the FA site falls back
    CHECK(has(src, "&& extra0_q4_K->q_img != nullptr"), "q4_K wimg selection needs a live image");
    CHECK(has(src, "&& extra0_mxfp4->q_img != nullptr"), "mxfp4 wimg selection needs a live image");
    const size_t fa = src.find("if (prefill_k_img == nullptr)");
    CHECK(fa != std::string::npos, "FA has a skipped-image branch");
    if (fa != std::string::npos) {
        const std::string block = src.substr(fa, 300);
        CHECK(has(block, "kernel = backend_ctx->fa.f32_f16_split.at(dk_dv)"),
              "FA takes the buffer kernel when the image is skipped");
        CHECK(has(block, "use_prefill_k_img = false"), "FA stops claiming an image argument");
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::printf("usage: %s <ggml-opencl.cpp>\n", argv[0]);
        return 2;
    }
    check_rule();
    check_predicates();
    check_source(argv[1]);

    if (g_bad != 0) {
        std::printf("image-limit: %d check(s) failed\n", g_bad);
        return 1;
    }
    std::printf("image-limit: rule, predicates and backend text OK\n");
    return 0;
}
