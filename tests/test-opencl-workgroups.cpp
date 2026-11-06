// 0072: device-free tests for the uniform-work-group rounding rule.
//   - the rounding helper itself (every dimension, idempotence, local 0/NULL,
//     a local size 1, the tail), and, in the Adreno build of this test
//     (-DGGML_OPENCL_USE_ADRENO_KERNELS), that it returns the input unchanged;
//   - the text check over ggml-opencl.cpp: every clEnqueueNDRangeKernel call
//     that passes a local size is preceded by ggml_cl_round_up_global_work_size,
//     so no launch site hands the driver a raw (ragged) count. The two calls
//     inside enqueue_ndrange_kernel are covered by that wrapper's own call.
//   - the kernel sources: a parameter-list line whose comment ends with a
//     comma swallowed a declaration separator (the ssm_conv round-2 blocker: a
//     comment sat before the comma, so two parameters merged), and, when a
//     clang exists, a syntax-only compile as OpenCL C (spir64, CL3.0, the
//     default header) of the sources in tests/opencl_clang_clean.txt, the list
//     measured clean under a generic clang; the others are counted, not compiled.
// No GPU, no OpenCL runtime. The helper is ggml/src/ggml-opencl/
// ggml-opencl-uniform.h, the same header ggml-opencl.cpp includes, so a
// mutation here is a mutation of the backend's launch rule.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <dirent.h>

#include "ggml-opencl-uniform.h"

static int g_bad = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::printf("FAIL: %s\n", what); ++g_bad; } } while (0)

static void check_rounding(void) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // the Adreno build: the helper is a no-op, the Adreno launches stay ragged
    CHECK(ggml_cl_round_up_global(10, 64) == 10, "adreno: round leaves the count");
    size_t g[3] = {10, 20, 30};
    const size_t l[3] = {64, 8, 1};
    ggml_cl_round_up_global_work_size(g, l, 3);
    CHECK(g[0] == 10 && g[1] == 20 && g[2] == 30, "adreno: no dimension rounds");
    std::printf("workgroups: Adreno build, the rounding helper is a no-op\n");
#else
    // one dimension: a ragged count rounds to a whole local size
    CHECK(ggml_cl_round_up_global(10, 64) == 64, "round 10/64");
    CHECK(ggml_cl_round_up_global(64, 64) == 64, "round 64/64 stays");
    CHECK(ggml_cl_round_up_global(65, 64) == 128, "round 65/64");
    CHECK(ggml_cl_round_up_global(0, 64) == 0, "round 0 stays");
    CHECK(ggml_cl_round_up_global(7, 1) == 7, "local 1 leaves the count");
    CHECK(ggml_cl_round_up_global(7, 0) == 7, "local 0 leaves the count");
    // a multi-dimension launch: each dimension rounds independently, and a
    // NULL local size (the driver picks) is left alone
    size_t g[3] = {10, 20, 30};
    const size_t l[3] = {64, 8, 1};
    ggml_cl_round_up_global_work_size(g, l, 3);
    CHECK(g[0] == 64 && g[1] == 24 && g[2] == 30, "round three dimensions");
    // idempotence: a launch that is already uniform is unchanged
    size_t g2[3] = {64, 24, 30};
    ggml_cl_round_up_global_work_size(g2, l, 3);
    CHECK(g2[0] == 64 && g2[1] == 24 && g2[2] == 30, "rounding is idempotent");
    size_t g3[2] = {10, 11};
    ggml_cl_round_up_global_work_size(g3, NULL, 2);
    CHECK(g3[0] == 10 && g3[1] == 11, "NULL local size is left alone");
#endif
}

// split the top-level comma arguments of a call, from the '(' after its name
static std::vector<std::string> split_args(const std::string & text, size_t open_paren) {
    std::vector<std::string> args;
    int depth = 0;
    std::string cur;
    for (size_t i = open_paren; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '(' || c == '[') depth++;
        if (c == ')' || c == ']') {
            if (depth == 0) { args.push_back(cur); return args; }
            depth--;
        }
        if (c == ',' && depth == 0) { args.push_back(cur); cur.clear(); }
        else cur += c;
    }
    args.push_back(cur);
    return args;
}

static void check_launch_sites(const std::string & src) {
    // the wrapper's own two calls: it rounds before both, at the top of the
    // function; test that separately, then check every other call site for a
    // rounding call between it and the previous launch site
    const size_t wstart = src.find("void enqueue_ndrange_kernel(");
    const size_t wend = src.find("const void * get_adreno_bin_kernel", wstart);
    const size_t wround = src.find("ggml_cl_round_up_global_work_size(global_work_size, local_work_size, work_dim);", wstart);
    CHECK(wstart != std::string::npos, "the launch wrapper exists");
    CHECK(wround != std::string::npos, "the launch wrapper rounds every dimension");

    size_t pos = 0;
    size_t prev = 0;
    int checked = 0;
    int raw = 0;
    int in_wrapper = 0;
    for (;;) {
        const size_t start = src.find("clEnqueueNDRangeKernel(", pos);
        if (start == std::string::npos) break;
        const size_t args_begin = start + std::strlen("clEnqueueNDRangeKernel(");
        const std::string call = src.substr(start, 1200);
        const std::vector<std::string> a = split_args(call, std::strlen("clEnqueueNDRangeKernel("));
        pos = args_begin;
        if (a.size() < 6) continue;
        std::string local = a[5];
        const size_t b = local.find_first_not_of(" \t\r\n");
        const size_t e = local.find_last_not_of(" \t\r\n");
        local = b == std::string::npos ? "" : local.substr(b, e - b + 1);
        if (local.empty() || local == "NULL" || local == "nullptr") continue;
        // the two calls inside the wrapper sit 1.5 KB after its top; the
        // wrapper's own rounding is before them, not between the previous site
        const bool wrapper_call = start > wstart && start < wend;
        if (!wrapper_call) {
            const std::string between = src.substr(prev, start - prev);
            if (between.find("ggml_cl_round_up_global_work_size(") == std::string::npos) {
                std::printf("FAIL: launch at byte %zu passes a local size without rounding: %s\n",
                            start, call.substr(0, 120).c_str());
                ++raw;
                ++g_bad;
            }
        } else {
            ++in_wrapper;
        }
        ++checked;
        prev = start;
    }
    CHECK(raw == 0, "no launch site passes a raw count with a local size");
    CHECK(checked > 20, "the text check saw the backend's launch sites");
    CHECK(in_wrapper == 2, "both wrapper calls are covered");
    std::printf("workgroups: %d launch sites with a local size checked (%d in the wrapper)\n", checked, in_wrapper);

    // the non-uniform fallback must be gone: no launch picks NULL local size
    // because the shape is ragged, and no local_work_size_ptr remains
    CHECK(src.find("local_work_size_ptr") == std::string::npos, "the local-size fallback is gone");
}

static std::string read_text(const std::string & path) {
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return std::string();
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}


// a comment inside a kernel's parameter list whose text ends with a comma: the
// comma belongs to the comment, so the declarations around it merge (the
// ssm_conv round-2 blocker, `int act /* ... */ int ne01`). Only parameter lists
// count: prose comments elsewhere may end with a comma.
static void check_kernel_comments(const char * dir, const std::vector<std::string> & files) {
    int bad = 0, lists = 0;
    for (const std::string & f : files) {
        const std::string text = read_text(std::string(dir) + "/" + f);
        size_t pos = 0;
        while ((pos = text.find("kernel void ", pos)) != std::string::npos) {
            const size_t c = text.find('(', pos);
            if (c == std::string::npos) break;
            // the parameter list, comment-aware: parens inside a comment or a
            // string do not count
            int depth = 1;
            size_t i = c + 1;
            for (; i < text.size(); ++i) {
                if (text.compare(i, 2, "//") == 0) {
                    const size_t nl = text.find('\n', i);
                    i = nl == std::string::npos ? text.size() : nl;
                    continue;
                }
                if (text.compare(i, 2, "/*") == 0) {
                    const size_t e = text.find("*/", i + 2);
                    i = e == std::string::npos ? text.size() : e + 1;
                    continue;
                }
                if (text[i] == '(') ++depth;
                if (text[i] == ')') {
                    if (--depth == 0) break;
                }
            }
            const std::string region = text.substr(c + 1, i - c - 1);
            ++lists;
            // per line inside the list: a comment whose text ends with ','
            size_t ls = 0, line = 1;
            for (size_t k = 0; k < c; ++k) if (text[k] == '\n') ++line;
            while (ls < region.size()) {
                const size_t le = region.find('\n', ls);
                const std::string ln = region.substr(ls, (le == std::string::npos ? region.size() : le) - ls);
                const size_t cc = ln.find("//");
                if (cc != std::string::npos) {
                    const std::string body = ln.substr(cc + 2);
                    const size_t lb = body.find_last_not_of(" \t\r");
                    if (lb != std::string::npos && body[lb] == ',') {
                        std::printf("FAIL: %s:%d a comma inside a parameter comment ends the line: %s\n",
                                    f.c_str(), (int)line, ln.c_str());
                        ++bad;
                    }
                }
                ++line;
                if (le == std::string::npos) break;
                ls = le + 1;
            }
            pos = i;
        }
    }
    CHECK(bad == 0, "no kernel parameter list hides a comma in a comment");
    CHECK(lists > 50, "the parameter lists were read");
}

// clang, clang-18, clang-19, or $CLANG (first that runs); empty when none
static std::string find_clang(void) {
    const char * env = std::getenv("CLANG");
    if (env != nullptr && env[0] != 0 &&
        std::system((std::string(env) + " --version >/dev/null 2>&1").c_str()) == 0) {
        return env;
    }
    for (const char * c : { "clang", "clang-18", "clang-19" }) {
        if (std::system((std::string(c) + " --version >/dev/null 2>&1").c_str()) == 0) return c;
    }
    return std::string();
}

// compile exactly the sources listed in tests/opencl_clang_clean.txt, when a
// clang exists. The list was measured with clang-18 (`-target spir64 -x cl
// -cl-std=CL3.0 -Xclang -finclude-default-header -fsyntax-only`) on this tree,
// both before the series' 0072 change and after: every name in it compiles
// with 0 errors under a generic front end. The rest of kernels/*.cl calls a
// vendor builtin or needs the backend's own -D defines, and fails identically
// on the pre-0072 text, so it is counted and reported and not compiled:
// auto-discovery would fail on any host that has a clang and catch nothing.
// To refresh the list, compile kernels/*.cl with that command and keep the
// names that report no error before and after the patch.
static void check_kernel_compile(const char * dir, const std::vector<std::string> & files,
                                 const std::string & list_path) {
    const std::string list = read_text(list_path);
    if (list.empty()) {
        std::printf("FAIL: the clang-clean list %s is not readable\n", list_path.c_str());
        CHECK(false, "tests/opencl_clang_clean.txt is readable");
        return;
    }
    std::vector<std::string> listed;
    for (size_t i = 0; i < list.size(); ) {
        size_t nl = list.find('\n', i);
        if (nl == std::string::npos) nl = list.size();
        std::string name = list.substr(i, nl - i);
        i = nl + 1;
        while (!name.empty() && std::isspace((unsigned char)name.back())) name.pop_back();
        size_t lead = 0;
        while (lead < name.size() && std::isspace((unsigned char)name[lead])) ++lead;
        name = name.substr(lead);
        if (name.empty() || name[0] == '#') continue;
        listed.push_back(name);
    }

    int missing = 0;
    std::set<std::string> names(files.begin(), files.end());
    for (const std::string & f : listed) {
        if (names.count(f) == 0 || read_text(std::string(dir) + "/" + f).empty()) {
            std::printf("FAIL: the listed kernel source %s is missing from %s\n", f.c_str(), dir);
            ++missing;
        }
    }
    int unlisted = static_cast<int>(files.size()) - static_cast<int>(listed.size());
    int listed_present = static_cast<int>(listed.size()) - missing;

    const std::string cc = find_clang();
    if (cc.empty()) {
        std::printf("note: no clang (clang, clang-18, clang-19, $CLANG); %d listed kernel sources are not "
                    "compiled, %d sources are not in the list\n", listed_present, unlisted);
        CHECK(missing == 0, "every kernel source listed in tests/opencl_clang_clean.txt exists");
        return;
    }
    int compiled = 0, failed = missing;
    for (const std::string & f : listed) {
        const std::string path = std::string(dir) + "/" + f;
        if (read_text(path).empty()) continue;   // reported above
        const std::string cmd = cc + " -target spir64 -x cl -cl-std=CL3.0"
                               " -Xclang -finclude-default-header -fsyntax-only \"" + path + "\" >/dev/null 2>&1";
        if (std::system(cmd.c_str()) != 0) {
            std::printf("FAIL: %s does not compile as OpenCL C (spir64, CL3.0, %s)\n", f.c_str(), cc.c_str());
            ++failed;
        }
        ++compiled;
    }
    std::printf("workgroups: %d of the %d listed kernel sources compiled as OpenCL C, %d kernel sources not "
                "in the list are not compiled\n", compiled, listed_present, unlisted);
    CHECK(failed == 0, "every kernel source in tests/opencl_clang_clean.txt compiles as OpenCL C");
    CHECK(compiled > 20, "the list names enough sources to be a real check");
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;
    check_rounding();
#ifdef GGML_OPENCL_SRC_PATH
    const char * path = GGML_OPENCL_SRC_PATH;
#else
    const char * path = argc > 1 ? argv[1] : nullptr;
#endif
#ifdef GGML_OPENCL_KERNELS_PATH
    const char * kdir = GGML_OPENCL_KERNELS_PATH;
#else
    const char * kdir = argc > 2 ? argv[2] : nullptr;
#endif
#ifdef GGML_OPENCL_CLANG_CLEAN_PATH
    const std::string clist = GGML_OPENCL_CLANG_CLEAN_PATH;
#else
    const std::string clist = argc > 3 ? argv[3] : std::string();
#endif
    if (path == nullptr) {
        std::printf("FAIL: no ggml-opencl.cpp path (pass it as argv[1])\n");
        return 1;
    }
    const std::string src = read_text(path);
    if (src.empty()) {
        std::printf("FAIL: cannot open %s\n", path);
        return 1;
    }
    check_launch_sites(src);
    if (kdir != nullptr && !clist.empty()) {
        std::vector<std::string> files;
        DIR * d = opendir(kdir);
        if (d != nullptr) {
            for (;;) {
                struct dirent * de = readdir(d);
                if (de == nullptr) break;
                const std::string n = de->d_name;
                if (n.size() > 3 && n.compare(n.size() - 3, 3, ".cl") == 0) files.push_back(n);
            }
            closedir(d);
        }
        std::sort(files.begin(), files.end());
        CHECK(!files.empty(), "the kernel sources are listed");
        check_kernel_comments(kdir, files);
        check_kernel_compile(kdir, files, clist);
    } else if (kdir == nullptr) {
        std::printf("FAIL: no kernels directory (pass it as argv[2])\n");
        ++g_bad;
    } else {
        std::printf("FAIL: no clang-clean list (pass it as argv[3])\n");
        ++g_bad;
    }
    if (g_bad) {
        std::printf("test-opencl-workgroups: %d failure(s)\n", g_bad);
        return 1;
    }
    std::printf("test-opencl-workgroups: ok\n");
    return 0;
}
