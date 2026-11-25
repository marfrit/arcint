// fa_ref_host.cpp — run trellis-serve's vendored ESIMD prefill-attention kernel
// (fa::Kern<128>, contrib/refkernels/trellis-fa/fa_kern.sycl) on an Intel GPU
// through OpenCL 3.0, from the SPIR-V build-spirv.sh produced, and check it
// against a CPU oracle of the reference's own formula.
//
//   fa_ref_host --spv fa_kern.precise.spv --abi abi.txt
//               [--ref-spv fa_kern.ref.spv] [--shape NAME[,NAME]] [--dry]
//               [--reps N] [--qtw N] [--dump FILE] [--dump-fixture PREFIX]
//               [--ref-tol-abs X] [--ref-tol-kl X] [--kernel NAME]
//
// Two modules, two arms:
//   --spv      the module built with -fp-model=precise: the correctness arm.
//   --ref-spv  the module built with trellis-serve's own flags
//              (-O3 -ffast-math, xpu/scripts/build_inkling2.sh:7-8): the timing
//              arm. Its output is compared against the same oracle, with a stated
//              tolerance, so a fast-math difference is seen and not assumed away.
//
// A module path ending in .spv is loaded with clCreateProgramWithIL (the default
// route). Any other path is loaded with clCreateProgramWithBinary: that is the
// AOT fallback for a driver that refuses the SPIR-V (README §AOT-fallback), and
// it is never the default.
//
// Shapes (all Hq 24, Hk 8, D 128, window 1,024, E 1,024, sinks, bf16 table,
// contiguous positions, qpos0 = Lk - Lq): the default is q512xkv1536; --shape
// adds q2048xkv3072 and q2048xkv2048, or `all`. For each shape: the host-side
// bounds check, the oracle comparison, and the median/min/max of >= 10 timed
// launches per module from clGetEventProfilingInfo.
//
// It is deliberately a *measurement and oracle* program, not a backend: it sets
// the kernel's arguments in the order the ABI dump read out of the module, and it
// never guesses. The kernel's own code is verbatim trellis-serve (MIT, see the
// header of fa_kern.sycl); the oracle below is the reference's formula written
// out in plain f32 arithmetic.
//
// Evidence for every build option and every bound this file relies on is in
// README.md next to this file; the lines that matter are cited in comments here.
//
// --dry does everything except the enqueue: platform and device, programs from
// IL or an AOT binary, builds, kernel names, kernel creation, arguments set,
// buffers filled, the bounds table, the oracle. On a machine without a device it stops at the
// platform step and says so.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <chrono>

// ---------------------------------------------------------------------------
// The invariants of every shape: Inkling 2 Mini's attention geometry. Only
// Lq and Lk vary per shape (see Shape below). qtw = 1 is the reference's default
// (xpu/csrc's prefill_attn signature), and the launch geometry is the
// reference's own arithmetic (xpu/csrc/inkling2_ops.sycl:665-671).
// ---------------------------------------------------------------------------
static constexpr int HQ = 24;
static constexpr int HK = 8;
static constexpr int D = 128;
static constexpr int WINDOW = 1024;
static constexpr int E = 1024;                 // bias table bins; Inkling's extent is 1,024 -> 1,025 bins, this arm uses 1,024
static constexpr int KPOS0 = 0;                // every shape: keys start at position 0
static constexpr int QR = 8;                   // fa::QR, the reference's rows per query tile
static constexpr float LOG2E = 1.4426950408889634f;   // fa::LOG2E
static constexpr int G = HQ / HK;              // 3 query heads per kv head
// ---------------------------------------------------------------------------
// Shapes: the comparison cases. Each is Hq 24, Hk 8, D 128, window 1,024, E
// 1,024, sinks, a bf16 table, contiguous positions with kpos0 = 0 and
// qpos0 = Lk - Lq (the chunk's queries at the end of its keys). The default is
// the single case of stage 1b; --shape adds the others.
// ---------------------------------------------------------------------------
struct Shape {
    const char* name;
    int lq, lk;
};

static const Shape kDefaultShape = {"q512xkv1536", 512, 1536};
static const Shape kShapes[] = {
    {"q2048xkv3072", 2048, 3072},
    {"q2048xkv2048", 2048, 2048},
    {"q512xkv1536", 512, 1536},
};

static bool shape_by_name(const std::string& n, Shape& out) {
    if (n == kDefaultShape.name) { out = kDefaultShape; return true; }
    for (const Shape& s : kShapes) if (n == s.name) { out = s; return true; }
    return false;
}

// qpos0 / kpos0 of a shape: positions contiguous from 0, the queries at the end.
static inline int qpos0_of(const Shape& s) { return (s.lk - s.lq); }

// bf16 as the reference sees it: uint16 with the f32's top 16 bits (the kernel
// does `simd<uint32_t>(bb) << 16` then reads it as f32 — exactly this).
static inline float bf16_to_f32(uint16_t b) {
    uint32_t w = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}

// f32 -> bf16 with round-to-nearest-even on the 16 discarded bits (the test
// fixture's own conversion; the kernel only reads what we put there).
static inline uint16_t f32_to_bf16(float f) {
    uint32_t w;
    std::memcpy(&w, &f, 4);
    uint32_t lsb = (w >> 16) & 1;
    uint32_t bias = 0x7fff + lsb;
    uint32_t rounded = w + bias;
    return (uint16_t)(rounded >> 16);
}

// f32 -> f16 with round-to-nearest-even; written out so the fixture does not
// depend on _Float16 support or a compiler's conversion mode.
static inline uint16_t f32_to_f16(float f) {
    uint32_t w;
    std::memcpy(&w, &f, 4);
    uint32_t sign = (w >> 16) & 0x8000u;
    int32_t exp = (int32_t)((w >> 23) & 0xff) - 127;
    uint32_t mant = w & 0x007fffffu;
    if (exp > 15) return (uint16_t)(sign | 0x7c00u);                   // inf
    if (exp < -24) return (uint16_t)(sign);                           // underflow -> 0
    if (exp < -14) {                                                   // subnormal
        uint32_t shift = (uint32_t)(-14 - exp);
        uint32_t v = (mant | 0x00800000u) >> (shift + 13);
        // round to nearest even on the bits shifted out
        uint32_t rest = (mant | 0x00800000u) & ((1u << (shift + 13)) - 1u);
        uint32_t half = 1u << (shift + 12);
        if (rest > half || (rest == half && (v & 1u))) v++;
        return (uint16_t)(sign | v);
    }
    uint32_t v = (mant >> 13);
    uint32_t rest = mant & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (v & 1u))) {
        v++;
        if (v >= 0x400u) { v = 0; exp++; if (exp > 15) return (uint16_t)(sign | 0x7c00u); }
    }
    return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | v);
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) ? 0x80000000u : 0u;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    float out;
    if (e == 0) {
        out = (float)m * std::ldexp(1.0f, -24);
    } else if (e == 31) {
        out = m ? std::numeric_limits<float>::quiet_NaN()
                : std::numeric_limits<float>::infinity();
    } else {
        out = std::ldexp((float)(m | 0x400u), (int)e - 25);
    }
    uint32_t bits = 0;
    std::memcpy(&bits, &out, 4);
    bits |= sign;
    std::memcpy(&out, &bits, 4);
    return out;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The oracle: the reference's formula, f32, no kernel involved.
//   score (base 2) = LOG2E * (scale*q.k + bias);  bias = table[i, h, min(dist, E-1)]
//   for dist < E else 0; valid = kpos >= 0 && dist >= 0 && dist < window (window>0);
//   the sink is the initial (m, l) = (sink * LOG2E, 1), i.e. a key whose value
//   row is zero and whose logit is sink*LOG2E.
// That is xpu/csrc/inkling2_ops.sycl:512-517 (the block's own comment) and
// xpu/exl3xpu/inkling2/attn_triton.py:142-162 (reference_attention), and the
// kernel's body: acc = dpas(...); S = acc*scale_log2; sv += bias*LOG2E; m/l
// start at (sink*LOG2E, 1).
// ---------------------------------------------------------------------------
struct OracleOut {
    std::vector<float> o;          // [Lq, Hq, D]
    std::vector<float> row_sum_l;  // [Lq, Hq]: the denominator, for reporting
};

static OracleOut oracle(const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
                        const std::vector<uint16_t>& v, const std::vector<uint16_t>& table,
                        const std::vector<float>& sink, float scale, const Shape& sh) {
    const int lq = sh.lq, lk = sh.lk, qpos0 = qpos0_of(sh);
    OracleOut r;
    r.o.assign((size_t)lq * HQ * D, 0.0f);
    r.row_sum_l.assign((size_t)lq * HQ, 0.0f);
    const float scale_log2 = LOG2E * scale;

    // f16 -> f32 once, up front: f16 values are exactly representable in f32, so
    // this changes no value, only the traffic (the row dot and the output
    // accumulation then read contiguous f32 instead of decoding per element, per
    // row and per head — the v row's decode was repeated for every query row).
    std::vector<float> qf(q.size()), kf(k.size()), vf(v.size());
    for (size_t i = 0; i < q.size(); ++i) qf[i] = f16_to_f32(q[i]);
    for (size_t i = 0; i < k.size(); ++i) kf[i] = f16_to_f32(k[i]);
    for (size_t i = 0; i < v.size(); ++i) vf[i] = f16_to_f32(v[i]);

    std::vector<float> s((size_t)lk), p((size_t)lk);

    for (int i = 0; i < lq; ++i) {
        const int qpos = qpos0 + i;
        // the valid keys are a contiguous range: kpos <= qpos and, with a
        // window, dist < window — both monotone in the key row.
        int lo = 0;
        if (WINDOW > 0) { lo = qpos - WINDOW + 1 - KPOS0; if (lo < 0) lo = 0; }
        int hi = qpos - KPOS0 + 1; if (hi > lk) hi = lk;
        for (int h = 0; h < HQ; ++h) {
            const int kvh = h / G;
            const float* qrow = &qf[((size_t)i * HQ + h) * D];
            const float sinkv = LOG2E * sink[(size_t)i * HQ + h];
            float m = sinkv;                                  // the sink as the initial m
            float* sp = s.data();
            for (int j = lo; j < hi; ++j) {
                const float* krow = &kf[((size_t)j * HK + kvh) * D];
                float dot = 0.0f;
                for (int c = 0; c < D; ++c) dot += qrow[c] * krow[c];
                const int dist = qpos - (KPOS0 + j);
                float bias = 0.0f;
                if (dist < E) bias = bf16_to_f32(table[((size_t)i * HQ + h) * E + std::min(dist, E - 1)]);
                const float sj = scale_log2 * dot + LOG2E * bias;
                sp[j] = sj;
                if (sj > m) m = sj;
            }
            // second pass: p = exp2(s - m), l = sum including the sink's exp2(sink-m)
            float sum = std::exp2(sinkv - m);
            for (int j = lo; j < hi; ++j) {
                const float e = std::exp2(sp[j] - m);
                p[(size_t)j] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            r.row_sum_l[(size_t)i * HQ + h] = sum;
            float* orow = &r.o[((size_t)i * HQ + h) * D];
            for (int j = lo; j < hi; ++j) {
                const float pj = p[(size_t)j] * inv;
                if (pj == 0.0f) continue;
                const float* vrow = &vf[((size_t)j * HK + kvh) * D];
                for (int c = 0; c < D; ++c) orow[c] += pj * vrow[c];
            }
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// The bounds check, before any enqueue. Each buffer's maximal index is computed
// from the kernel's own index expressions, quoted in README.md §bounds:
//   q:  q + ((q0 + r) * Hq + h) * D, r < QR, h < Hq  ->  (Lq-1)*Hq + (Hq-1)) * D + D
//   k/v: kb32 = k + kvh * D, 2D blocks of rows kb0+t*KT with kb0 < hi <= Lk and
//        surfW = D*2-1, pitch = Hk*D*2-1 -> the last row start (Lk-1)*Hk*D + kvh*D,
//        plus one row's width D, i.e. at most Lk*Hk*D elements of the allocation
//   o:  o + ((q0 + r) * Hq + h) * D, D floats
//   table: gather offset dc*2 + ((qrow*Hq + h)*E)*2, dc <= E-1, uint16
//   sink: sink[(q0 + r) * Hq + h], one float
// A failure here is a case error, not a kernel defect, and it stops before the
// driver sees anything.
// ---------------------------------------------------------------------------
struct BufBound {
    const char* name;
    unsigned long long max_byte_index;   // the last byte the kernel may touch
    unsigned long long bytes_alloc;      // what we allocated
};

static bool bounds_check(std::vector<BufBound>& rows, const Shape& sh) {
    const unsigned long long nq = (unsigned long long)sh.lq * HQ * D;
    const unsigned long long nkv = (unsigned long long)sh.lk * HK * D;
    const unsigned long long ntab = (unsigned long long)sh.lq * HQ * E;
    const unsigned long long nsink = (unsigned long long)sh.lq * HQ;
    rows = {
        {"q",     2ull * nq - 1,    2ull * nq},
        {"k",     2ull * nkv - 1,   2ull * nkv},
        {"v",     2ull * nkv - 1,   2ull * nkv},
        {"o",     4ull * nq - 1,    4ull * nq},
        {"table", 2ull * ntab - 1,  2ull * ntab},
        {"sink",  4ull * nsink - 1, 4ull * nsink},
    };
    bool ok = true;
    for (auto& r : rows) {
        if (r.max_byte_index >= r.bytes_alloc) {
            std::fprintf(stderr, "BOUNDS: %s: the kernel's maximal byte index %llu is outside the %llu-byte allocation\n",
                         r.name, r.max_byte_index, r.bytes_alloc);
            ok = false;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Fixture: deterministic, seeded, the shape the model has. q and k rows are L2
// normalised to norm 20 and stored as fp16 (the model clips q/k to an L2 norm;
// trellis-serve's own test fixture xpu/tests/t_inkling2_prefill_esimd.py:7-9
// does exactly this), v is fp16 noise, table is bf16 noise in [-2, 2] (same
// test), sink is f32 noise. The same seed gives the same bytes for any Lq/Lk.
// ---------------------------------------------------------------------------
struct Fixture {
    std::vector<uint16_t> q, k, v, table;
    std::vector<float> sink;
};

static void normalise_rows_to(std::vector<uint16_t>& x, int rows, int cols, float norm) {
    for (int r = 0; r < rows; ++r) {
        uint16_t* row = &x[(size_t)r * cols];
        float s = 0.0f;
        for (int c = 0; c < cols; ++c) s += f16_to_f32(row[c]) * f16_to_f32(row[c]);
        if (s > 0.0f) {
            const float mul = norm / std::sqrt(s);
            for (int c = 0; c < cols; ++c) row[c] = f32_to_f16(f16_to_f32(row[c]) * mul);
        }
    }
}

static Fixture make_fixture(const Shape& sh) {
    Fixture f;
    std::mt19937_64 gen(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    f.q.resize((size_t)sh.lq * HQ * D);
    for (auto& x : f.q) x = f32_to_f16(nd(gen));
    normalise_rows_to(f.q, sh.lq * HQ, D, 20.0f);
    f.k.resize((size_t)sh.lk * HK * D);
    for (auto& x : f.k) x = f32_to_f16(nd(gen));
    normalise_rows_to(f.k, sh.lk * HK, D, 20.0f);
    f.v.resize((size_t)sh.lk * HK * D);
    for (auto& x : f.v) x = f32_to_f16(nd(gen));
    f.table.resize((size_t)sh.lq * HQ * E);
    for (auto& x : f.table) x = f32_to_bf16(ud(gen));
    f.sink.resize((size_t)sh.lq * HQ);
    for (auto& x : f.sink) x = nd(gen);
    return f;
}
// OpenCL plumbing.
// ---------------------------------------------------------------------------
static const char* cl_err(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        default: return "other";
    }
}
// Byte-exact dumping of a name string: a length and a hex dump, so a trailing
// NUL, a ';', a space or a mangled difference is visible instead of being
// silently trimmed by stdio. This is what a CL_INVALID_KERNEL_NAME needs.
static void dump_bytes(const char* tag, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    std::printf("%s: %zu byte(s):", tag, n);
    for (size_t i = 0; i < n; ++i) std::printf(" %02x", b[i]);
    std::printf("  \"");
    for (size_t i = 0; i < n; ++i) {
        const unsigned char ch = b[i];
        if (ch == 0) std::printf("\\0");
        else if (ch == ';') std::printf("';'");
        else if (ch == ' ') std::printf("<sp>");
        else if (ch < 0x20 || ch > 0x7e) std::printf("\\x%02x", ch);
        else std::printf("%c", ch);
    }
    std::printf("\"\n");
}

#define CLCHECK(x) do { cl_int _e = (x); if (_e != CL_SUCCESS) { \
    std::fprintf(stderr, "%s:%d: %s -> %s (%d)\n", __FILE__, __LINE__, #x, cl_err(_e), _e); \
    std::exit(1); } } while (0)

static std::vector<unsigned char> read_file(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s: %s\n", p.c_str(), std::strerror(errno)); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> b((size_t)n);
    if (n > 0 && std::fread(b.data(), 1, (size_t)n, f) != (size_t)n) {
        std::fprintf(stderr, "short read on %s\n", p.c_str()); std::exit(1);
    }
    std::fclose(f);
    return b;
}

// ABI side-check: parse abi.txt (build-spirv.sh's machine-readable twin) and
// require the kernel's live argument count and types to match it. This is the
// only place the host "knows" the ABI, and it is asserted, not assumed.
struct AbiRow { std::string type; int bytes; };
static std::vector<AbiRow> read_abi(const std::string& p, std::string& entry_name) {
    std::vector<AbiRow> rows;
    FILE* f = std::fopen(p.c_str(), "r");
    if (!f) {
        // --abi is optional; a missing file must not stop a run
        std::fprintf(stderr, "note: no ABI file at %s (optional; build-spirv.sh writes out/abi.txt)\n", p.c_str());
        return rows;
    }
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        // entry_point\tNAME | arg\tN\tTYPE\tBYTES
        if (s.rfind("entry_point\t", 0) == 0) { entry_name = s.substr(12); continue; }
        if (s.rfind("arg\t", 0) == 0) {
            size_t a = s.find('\t', 4);
            size_t b = s.find('\t', a + 1);
            AbiRow r;
            r.type = s.substr(a + 1, b - a - 1);
            std::string bs = s.substr(b + 1);
            r.bytes = (bs == "-") ? -1 : std::atoi(bs.c_str());
            rows.push_back(r);
        }
    }
    std::fclose(f);
    if (rows.empty()) {
        // The two files a build can leave behind: the machine twin `abi.txt` (rows
        // "arg<TAB>i<TAB>type<TAB>bytes") and the SPIR-V dump / human table that is not
        // one. Say which, and how to regenerate, instead of failing on a count of 0.
        std::fprintf(stderr,
            "note: %s names no arguments - it is not build-spirv.sh's machine twin "
            "(abi.txt: \"arg<TAB>index<TAB>type<TAB>bytes\" rows). Regenerate with "
            "bash build-spirv.sh <outdir> and point --abi at <outdir>/abi.txt, "
            "or drop --abi: the ABI is then not asserted.\n", p.c_str());
    }
    return rows;
}

// A case: one shape's fixture and its oracle output, computed before the driver
// is asked anything (so a --dry on a host without a device still does the work).
struct Case {
    Shape shape;
    Fixture fx;
    OracleOut ref;
    double oracle_ms = 0.0;
};

// One device module: the precise module (the oracle arm) and, optionally, the
// module built with the reference's own flags (the timing arm).
struct Module {
    const char* label = nullptr;   // "precise" or "reference-flags"
    std::string path;
    bool is_reference_flags = false;
    cl_program prog = nullptr;
    cl_kernel kern = nullptr;
    size_t work_group_limit = 0;
};

// ---------------------------------------------------------------------------
// Comparison of one module's output against the oracle. Same measures as stage
// 1b: max abs error, max row relative L2, mean row KL (the row's 128 output
// components as a simplex — the kernel writes no attention weights, so a KL
// over keys is not observable). The reference-flags module gets a stated
// tolerance; exceeding it is reported, not hidden.
// ---------------------------------------------------------------------------
struct Compare {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_kl = 0.0;
    int nnan = 0;
    int nrows = 0;
};

static Compare compare_out(const std::vector<float>& out, const std::vector<float>& refo, int lq) {
    Compare c;
    for (int i = 0; i < lq; ++i) {
        for (int h = 0; h < HQ; ++h) {
            const float* a = &out[((size_t)i * HQ + h) * D];
            const float* b = &refo[((size_t)i * HQ + h) * D];
            double row_abs = 0.0, sa = 0.0, sb = 0.0;
            for (int e = 0; e < D; ++e) {
                if (std::isnan(a[e])) ++c.nnan;
                row_abs = std::max(row_abs, (double)std::fabs(a[e] - b[e]));
                sa += a[e];
                sb += b[e];
            }
            c.max_abs = std::max(c.max_abs, row_abs);
            const double ps = std::isnan(sb) ? 1.0 : (sb != 0.0 ? sb : 1.0);
            const double qs = std::isnan(sa) ? -1.0 : (sa != 0.0 ? sa : 1e-30);
            double kl = 0.0;
            for (int e = 0; e < D; ++e) {
                const double p = b[e] / ps;
                const double kq = a[e] / qs;
                if (p > 0.0) {
                    if (kq <= 0.0) kl += p * 30.0;
                    else kl += p * std::log(p / kq);
                }
            }
            c.mean_kl += kl;
            ++c.nrows;
            double na = 0.0, nb = 0.0;
            for (int e = 0; e < D; ++e) { na += (double)a[e] * a[e]; nb += (double)b[e] * b[e]; }
            if (nb > 0.0) c.max_rel = std::max(c.max_rel, std::fabs(std::sqrt(na / nb) - 1.0));
        }
    }
    if (c.nrows) c.mean_kl /= c.nrows;
    return c;
}
// ---------------------------------------------------------------------------
// One timed kernel: reps launches, each with its own event, the device time from
// CL_PROFILING_COMMAND_START/END. The queue holds CL_QUEUE_PROFILING_ENABLE.
// ---------------------------------------------------------------------------
struct Timed {
    std::vector<double> ms;
    double median = 0.0, min = 0.0, max = 0.0;
};

static bool time_kernel(cl_command_queue q, cl_kernel kern, size_t n, size_t wg, int reps, Timed& t) {
    t.ms.clear();
    for (int r = 0; r < reps; ++r) {
        cl_event ev;
        cl_int e = clEnqueueNDRangeKernel(q, kern, 1, nullptr, &n, &wg, 0, nullptr, &ev);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "timed launch %d -> %s (%d)\n", r, cl_err(e), e); return false; }
        if (clWaitForEvents(1, &ev) != CL_SUCCESS) { std::fprintf(stderr, "timed launch %d: wait failed\n", r); clReleaseEvent(ev); return false; }
        cl_ulong b0 = 0, b1 = 0;
        cl_int e0 = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof b0, &b0, nullptr);
        cl_int e1 = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof b1, &b1, nullptr);
        clReleaseEvent(ev);
        if (e0 != CL_SUCCESS || e1 != CL_SUCCESS || b1 < b0) {
            std::fprintf(stderr, "timed launch %d: no profiling (%s/%s)\n", r, cl_err(e0), cl_err(e1));
            return false;
        }
        t.ms.push_back((double)(b1 - b0) / 1e6);
    }
    std::sort(t.ms.begin(), t.ms.end());
    t.min = t.ms.front();
    t.max = t.ms.back();
    t.median = t.ms[t.ms.size() / 2];
    return true;
}

// the pairs the case actually multiplies: per query, keys within the window and causal
static double pairs_of(const Shape& sh) {
    double pairs = 0.0;
    const int qpos0 = qpos0_of(sh);
    for (int i = 0; i < sh.lq; ++i) {
        const int qpos = qpos0 + i;
        int lo = 0;
        if (WINDOW > 0) { lo = qpos - WINDOW + 1 - KPOS0; if (lo < 0) lo = 0; }
        int hi = qpos - KPOS0 + 1; if (hi > sh.lk) hi = sh.lk;
        pairs += std::max(0, hi - lo);
    }
    return pairs;
}

int main(int argc, char** argv) {
    std::string spv_path, ref_spv_path, kernel_name, abi_path, dump_path, fixture_prefix;
    bool dry = false;
    int reps = 20;
    int qtw = 1;
    double ref_tol_abs = 5e-3;    // stated tolerance for the reference-flags module
    double ref_tol_kl = 5e-4;     // (see README §tolerance for where the numbers come from)
    std::vector<Shape> shapes;
    bool shapes_all = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--spv") spv_path = next("--spv");
        else if (a == "--ref-spv") ref_spv_path = next("--ref-spv");
        else if (a == "--kernel") kernel_name = next("--kernel");
        else if (a == "--abi") abi_path = next("--abi");
        else if (a == "--dump") dump_path = next("--dump");
        else if (a == "--dump-fixture") fixture_prefix = next("--dump-fixture");
        else if (a == "--ref-tol-abs") ref_tol_abs = std::atof(next("--ref-tol-abs").c_str());
        else if (a == "--ref-tol-kl") ref_tol_kl = std::atof(next("--ref-tol-kl").c_str());
        else if (a == "--reps") reps = std::atoi(next("--reps").c_str());
        else if (a == "--qtw") qtw = std::atoi(next("--qtw").c_str());
        else if (a == "--dry") dry = true;
        else if (a == "--shape") {
            std::string v = next("--shape");
            if (v == "all") { shapes_all = true; continue; }
            size_t p = 0;
            while (p <= v.size()) {
                size_t c = v.find(',', p);
                std::string one = v.substr(p, c == std::string::npos ? std::string::npos : c - p);
                Shape s;
                if (!shape_by_name(one, s)) {
                    std::fprintf(stderr, "unknown shape '%s'. Known:", one.c_str());
                    for (const Shape& k : kShapes) std::fprintf(stderr, " %s", k.name);
                    std::fprintf(stderr, "\n");
                    return 2;
                }
                shapes.push_back(s);
                if (c == std::string::npos) break;
                p = c + 1;
            }
        }
        else if (a == "--help" || a == "-h") {
            std::printf("usage: %s --spv fa_kern.precise.spv [--ref-spv fa_kern.ref.spv] [--abi abi.txt]\n"
                        "       [--shape NAME[,NAME]|all] [--dry] [--reps N] [--qtw N]\n"
                        "       [--dump FILE] [--dump-fixture PREFIX] [--ref-tol-abs X] [--ref-tol-kl X]\n"
                        "shapes: q512xkv1536 (default), q2048xkv3072, q2048xkv2048\n", argv[0]);
            return 0;
        } else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (spv_path.empty()) { std::fprintf(stderr, "--spv is required (fa_kern.precise.spv)\n"); return 2; }
    if (qtw < 1) { std::fprintf(stderr, "--qtw must be >= 1\n"); return 2; }
    if (shapes.empty()) {
        if (shapes_all) for (const Shape& k : kShapes) shapes.push_back(k);
        else shapes.push_back(kDefaultShape);
    }
    if (reps < 10) std::fprintf(stderr, "warning: --reps %d is below the 10 the timing needs; the same %d run(s) are still reported\n", reps, reps);

    // --- per shape: bounds, fixture, oracle — all before any driver step ---
    std::vector<Case> cases;
    for (const Shape& sh : shapes) {
        std::vector<BufBound> bounds;
        std::printf("\n=== shape %s: Lq %d Lk %d qpos0 %d kpos0 %d (Hq %d Hk %d G %d D %d window %d E %d qtw %d) ===\n",
                    sh.name, sh.lq, sh.lk, qpos0_of(sh), KPOS0, HQ, HK, G, D, WINDOW, E, qtw);
        const int qtiles = (sh.lq + QR - 1) / QR;
        const int qgroups = (qtiles + qtw - 1) / qtw;
        const size_t n_global = (size_t)qgroups * qtw * HQ;
        const size_t wg = (size_t)(G * qtw);
        std::printf("launch (the reference's own arithmetic, xpu/csrc:665-671): qtiles %d qgroups %d global %zu work-group %zu\n",
                    qtiles, qgroups, n_global, wg);
        if (!bounds_check(bounds, sh)) {
            std::fprintf(stderr, "bounds check failed for %s; nothing was enqueued\n", sh.name);
            return 3;
        }
        std::printf("bounds (buffer, kernel's maximal byte index, allocation):\n");
        for (const auto& r : bounds)
            std::printf("  %-6s max %12llu  alloc %12llu  %s\n", r.name, r.max_byte_index, r.bytes_alloc,
                        r.max_byte_index < r.bytes_alloc ? "inside" : "OUTSIDE");

        Case c;
        c.shape = sh;
        c.fx = make_fixture(sh);
        std::printf("fixture: q %zu B, k %zu B, v %zu B, table %zu B, sink %zu B (seed 12345)\n",
                    c.fx.q.size() * 2, c.fx.k.size() * 2, c.fx.v.size() * 2, c.fx.table.size() * 2, c.fx.sink.size() * 4);
        const float scale = 1.0f / std::sqrt((float)D);
        const auto t0 = std::chrono::steady_clock::now();
        c.ref = oracle(c.fx.q, c.fx.k, c.fx.v, c.fx.table, c.fx.sink, scale, sh);
        c.oracle_ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
        double checksum = 0.0;
        for (float x : c.ref.o) checksum += x;
        std::printf("oracle: %.1f ms, scale_log2 %.9f, checksum %.6f (l: min %.6f max %.6f)\n",
                    c.oracle_ms, LOG2E * scale, checksum,
                    *std::min_element(c.ref.row_sum_l.begin(), c.ref.row_sum_l.end()),
                    *std::max_element(c.ref.row_sum_l.begin(), c.ref.row_sum_l.end()));

        if (!fixture_prefix.empty()) {
            const bool many = shapes.size() > 1;
            const std::string base = many ? fixture_prefix + "." + sh.name : fixture_prefix;
            auto wr = [&](const std::string& suf, const void* p, size_t n) {
                FILE* f = std::fopen((base + suf).c_str(), "wb");
                if (!f) { std::fprintf(stderr, "cannot write %s%s\n", base.c_str(), suf.c_str()); return; }
                std::fwrite(p, 1, n, f);
                std::fclose(f);
            };
            wr(".q.f16", c.fx.q.data(), c.fx.q.size() * 2);
            wr(".k.f16", c.fx.k.data(), c.fx.k.size() * 2);
            wr(".v.f16", c.fx.v.data(), c.fx.v.size() * 2);
            wr(".table.bf16", c.fx.table.data(), c.fx.table.size() * 2);
            wr(".sink.f32", c.fx.sink.data(), c.fx.sink.size() * 4);
            std::printf("fixture dumped: %s.{q,k,v}.f16, %s.table.bf16, %s.sink.f32\n", base.c_str(), base.c_str(), base.c_str());
        }
        if (dry && !dump_path.empty()) {
            const bool many = shapes.size() > 1;
            std::string path = many ? dump_path + "." + sh.name + ".bin" : dump_path;
            FILE* f = std::fopen(path.c_str(), "wb");
            if (f) {
                std::fprintf(f, "# oracle only (--dry): shape %s Lq %d Lk %d Hq %d Hk %d D %d window %d E %d qpos0 %d kpos0 %d qtw %d\n",
                             sh.name, sh.lq, sh.lk, HQ, HK, D, WINDOW, E, qpos0_of(sh), KPOS0, qtw);
                std::fwrite(c.ref.o.data(), 4, c.ref.o.size(), f);
                std::fclose(f);
                std::printf("dump: %s (oracle f32; --dry writes no kernel output)\n", path.c_str());
            } else {
                std::fprintf(stderr, "cannot write %s: %s\n", path.c_str(), std::strerror(errno));
            }
        }
        cases.push_back(std::move(c));
    }

    // --- the ABI, if given (build-spirv.sh writes it next to the modules) ---
    const int nargs = 17;
    if (!abi_path.empty()) {
        std::string abi_entry;
        std::vector<AbiRow> abi = read_abi(abi_path, abi_entry);
        std::printf("\nabi file: %zu argument(s), entry point %s\n", abi.size(), abi_entry.c_str());
        if (!abi.empty() && (int)abi.size() != nargs) {
            std::fprintf(stderr, "the abi file names %zu arguments; this host sets %d\n", abi.size(), nargs);
            return 9;
        }
        // the 6 pointers first (a cl_mem goes in; "ptr" from the direct SPIR-V
        // route, "ptr-untyped" from the llvm-spirv fallback), then 10 ints, then
        // the float
        const char* want[17] = {"ptr","ptr","ptr","ptr","ptr","ptr",
                                "int","int","int","int","int","int","int","int","int","int","float"};
        for (int i = 0; i < nargs; ++i) {
            bool ptrok = abi[i].type.rfind("ptr", 0) == 0 && std::string(want[i]) == "ptr";
            if (abi[i].type != want[i] && !ptrok)
                std::fprintf(stderr, "warning: abi arg %d is %s, the host sets a %s\n", i, abi[i].type.c_str(), want[i]);
        }
        if (kernel_name.empty()) kernel_name = abi_entry;
    }

    // --- platform / device ---
    cl_platform_id platform = nullptr;
    cl_uint np = 0;
    cl_int perr = clGetPlatformIDs(0, nullptr, &np);
    if (perr != CL_SUCCESS || np == 0) {
        // CL_PLATFORM_NOT_FOUND_KHR (-1001) on a host with the loader but no
        // vendor ICD: no device exists here, which is exactly the --dry limit.
        std::fprintf(stderr, "\nno OpenCL platform (clGetPlatformIDs -> %s (%d)); every step up to the driver is done, "
                             "the driver steps need the card host\n", cl_err(perr), perr);
        return 4;
    }
    std::vector<cl_platform_id> plats(np);
    CLCHECK(clGetPlatformIDs(np, plats.data(), nullptr));
    for (cl_uint i = 0; i < np; ++i) {
        char name[256] = {0}, ver[128] = {0};
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof name, name, nullptr);
        clGetPlatformInfo(plats[i], CL_PLATFORM_VERSION, sizeof ver, ver, nullptr);
        std::printf("platform %u: %s (%s)\n", i, name, ver);
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS && nd > 0 && !platform)
            platform = plats[i];
    }
    if (!platform) { std::fprintf(stderr, "no GPU device on any platform: nothing to do (this is the --dry limit)\n"); return 4; }
    cl_device_id dev = nullptr;
    CLCHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    {
        char name[256] = {0}, drv[256] = {0}, ilv[64] = {0};
        cl_uint bits = 0;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, nullptr);
        clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof drv, drv, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_IL_VERSION, sizeof ilv, ilv, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_ADDRESS_BITS, sizeof bits, &bits, nullptr);
        cl_ulong mem = 0;
        size_t maxwg = 0;
        clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof maxwg, &maxwg, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, nullptr);
        std::printf("device: %s, driver %s, OpenCL IL %s, address bits %u, max work-group %zu, global mem %.1f GiB\n",
                    name, drv, ilv, bits, maxwg, (double)mem / (1024.0 * 1024 * 1024));
        if (bits != 64) std::fprintf(stderr, "warning: CL_DEVICE_ADDRESS_BITS is %u; the ABI dump assumed 8-byte pointers\n", bits);
    }

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CLCHECK(err);
    const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qprops, &err);
    CLCHECK(err);

    // --- the modules ---
    std::vector<Module> mods;
    {
        Module m;
        m.label = "precise";
        m.path = spv_path;
        mods.push_back(m);
    }
    if (!ref_spv_path.empty()) {
        Module m;
        m.label = "reference-flags";
        m.path = ref_spv_path;
        m.is_reference_flags = true;
        mods.push_back(m);
    }

    // Build options. Each one is sourced in README.md §build-options:
    //   -vc-codegen                             required for an ESIMD image: the SYCL
    //                                           runtime adds it itself for isEsimdImage
    //                                           images (program_manager.cpp,
    //                                           appendCompileOptionsFromImage), and
    //                                           the exl3xpu issue #1 says so in words;
    //                                           without it IGC has no ESIMD variant.
    //   -cl-fp32-correctly-rounded-divide-sqrt  the kernel divides 1.0f / l; Intel's
    //                                           fp32 divide is 2.5 ulp by default
    //                                           (Strata src/sycl/CMakeLists.txt:69-72).
    //                                           Kept for the oracle arm's sake.
    //   -ze-opt-greater-than-4GB-buffer-required  keeps pointer arguments on 64-bit
    //                                           addresses; without it IGC turns them
    //                                           into 32-bit surface offsets and misreads
    //                                           past 4 GiB (Strata's measured defect,
    //                                           CMakeLists.txt:79-85).
    // Deliberately absent: -cl-fast-relaxed-math / -cl-finite-math-only /
    // -cl-mad-enable (llama.cpp's ggml-opencl compile_opts, ggml-opencl.cpp:1998-2000).
    // The reference-flags module fed -ffast-math *compile-time*; these build
    // options stay identical for both modules so the comparison isolates the
    // compiler's flags, not the driver's.
    std::string build_opts = "-vc-codegen -cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required";
    std::printf("\nbuild options (both modules): %s\n", build_opts.c_str());

    const size_t wg = (size_t)(G * qtw);
    for (Module& m : mods) {
        std::vector<unsigned char> blob = read_file(m.path);
        size_t blob_len = blob.size();
        cl_int bin_status = CL_SUCCESS;
        const bool is_il = m.path.size() >= 4 && m.path.compare(m.path.size() - 4, 4, ".spv") == 0;
        std::printf("\n--- module %s: %s (%zu bytes, %s) ---\n", m.label, m.path.c_str(), blob.size(),
                    is_il ? "clCreateProgramWithIL" : "clCreateProgramWithBinary (AOT)");
        if (is_il) {
            m.prog = clCreateProgramWithIL(ctx, blob.data(), blob.size(), &err);
            if (err != CL_SUCCESS) {
                std::fprintf(stderr, "clCreateProgramWithIL(%s) -> %s (%d); the driver's SPIR-V reader refused this module\n",
                             m.path.c_str(), cl_err(err), err);
                std::fprintf(stderr, "the documented fallback is an AOT binary for bmg-g21 on the card host (README §AOT-fallback),\n"
                                     "passed here as --spv <fa_kern.bmg_g21.bin>, not a quiet switch\n");
                return 5;
            }
        } else {
            const unsigned char* ptr = blob.data();
            m.prog = clCreateProgramWithBinary(ctx, 1, &dev, &blob_len, &ptr, &bin_status, &err);
            if (err != CL_SUCCESS) {
                std::fprintf(stderr, "clCreateProgramWithBinary(%s) -> %s (%d)\n", m.path.c_str(), cl_err(err), err);
                return 5;
            }
        }
        // a precompiled binary takes no build options (the implementation rejects
        // them for binaries); an IL program gets the ones README §build-options
        // names, -vc-codegen first
        err = clBuildProgram(m.prog, 1, &dev, is_il ? build_opts.c_str() : "", nullptr, nullptr);
        {
            size_t logsz = 0;
            if (clGetProgramBuildInfo(m.prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz) == CL_SUCCESS && logsz > 1) {
                std::vector<char> log(logsz + 1, 0);
                clGetProgramBuildInfo(m.prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
                std::printf("build log:\n%s\n", log.data());
            }
        }
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "clBuildProgram(%s) -> %s (%d)\n", m.label, cl_err(err), err);
            return 6;
        }
        if (!is_il) {
            cl_uint btype = 0;
            if (clGetProgramInfo(m.prog, CL_PROGRAM_BINARY_TYPE, sizeof btype, &btype, nullptr) == CL_SUCCESS)
                std::printf("CL_PROGRAM_BINARY_TYPE %u (0 none, 1 compiled object, 2 library, 4 executable)\n", btype);
        }
        size_t nsz = 0;
        CLCHECK(clGetProgramInfo(m.prog, CL_PROGRAM_KERNEL_NAMES, 0, nullptr, &nsz));
        std::string names(nsz, '\0');
        CLCHECK(clGetProgramInfo(m.prog, CL_PROGRAM_KERNEL_NAMES, nsz, &names[0], nullptr));
        std::printf("CL_PROGRAM_KERNEL_NAMES: %s\n", names.c_str());
        dump_bytes("CL_PROGRAM_KERNEL_NAMES bytes", names.data(), names.size());
        // the list is ';'-separated and NUL-terminated; each token is dumped so a
        // trailing NUL or space is seen. The tokens are the candidate names.
        std::vector<std::string> tokens;
        {
            size_t start = 0;
            while (start < names.size()) {
                size_t stop = names.find(';', start);
                if (stop == std::string::npos) stop = names.find('\0', start);
                if (stop == std::string::npos) stop = names.size();
                if (stop > start) tokens.push_back(names.substr(start, stop - start));
                if (names[stop] == '\0') break;
                start = stop + 1;
                if (start >= names.size()) break;
            }
            for (size_t i = 0; i < tokens.size(); ++i) {
                dump_bytes(("token " + std::to_string(i)).c_str(), tokens[i].data(), tokens[i].size());
            }
        }
        std::string kname;
        if (!kernel_name.empty()) {
            kname = kernel_name;      // --kernel wins
        } else if (!tokens.empty()) {
            kname = tokens.front();
        } else {
            kname = names.substr(0, names.find(';'));
        }
        std::printf("kernel: %s\n", kname.c_str());
        dump_bytes("name passed to clCreateKernel", kname.data(), kname.size());
        m.kern = clCreateKernel(m.prog, kname.c_str(), &err);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "clCreateKernel(%s) in %s -> %s (%d)\n", kname.c_str(), m.label, cl_err(err), err);
            // A name mismatch and a listed-but-not-creatable kernel look the same
            // from clCreateKernel alone. clCreateKernelsInProgram enumerates what
            // the driver will actually create and gives each one's own
            // CL_KERNEL_FUNCTION_NAME, so the two cases are told apart.
            cl_uint nk = 0;
            cl_int e2 = clCreateKernelsInProgram(m.prog, 0, nullptr, &nk);
            std::printf("clCreateKernelsInProgram(count only) -> %s (%d), %u kernel(s)\n", cl_err(e2), e2, nk);
            if (e2 == CL_SUCCESS && nk > 0) {
                std::vector<cl_kernel> all(nk, nullptr);
                cl_int e3 = clCreateKernelsInProgram(m.prog, nk, all.data(), nullptr);
                if (e3 != CL_SUCCESS) {
                    std::fprintf(stderr, "clCreateKernelsInProgram(%u) -> %s (%d)\n", nk, cl_err(e3), e3);
                    return 7;
                }
                int pick = -1;
                for (cl_uint i = 0; i < nk; ++i) {
                    size_t fn = 0;
                    cl_uint na = 0;
                    char attrs[512] = {0};
                    size_t attrsz = 0;
                    std::string fname;
                    if (clGetKernelInfo(all[i], CL_KERNEL_FUNCTION_NAME, 0, nullptr, &fn) == CL_SUCCESS && fn > 0) {
                        fname.assign(fn, '\0');
                        clGetKernelInfo(all[i], CL_KERNEL_FUNCTION_NAME, fn, &fname[0], nullptr);
                        dump_bytes("  CL_KERNEL_FUNCTION_NAME", fname.data(), fname.size());
                    }
                    clGetKernelInfo(all[i], CL_KERNEL_NUM_ARGS, sizeof na, &na, nullptr);
                    if (clGetKernelInfo(all[i], CL_KERNEL_ATTRIBUTES, 0, nullptr, &attrsz) == CL_SUCCESS && attrsz > 1) {
                        clGetKernelInfo(all[i], CL_KERNEL_ATTRIBUTES, std::min(attrsz, sizeof attrs), attrs, nullptr);
                    }
                    std::printf("  kernel %u: function name %s (tail trimmed), CL_KERNEL_NUM_ARGS %u, attributes \"%s\"\n",
                                i, fname.c_str(), na, attrs);
                    if (pick < 0 && (fname == kname || fname == kname + std::string(1, '\0'))) pick = (int)i;
                }
                if (pick < 0 && nk == 1) pick = 0;
                if (pick < 0) {
                    std::fprintf(stderr, "clCreateKernelsInProgram created %u kernel(s), none whose name matches %s\n", nk, kname.c_str());
                    return 7;
                }
                for (cl_uint i = 0; i < nk; ++i) if ((int)i != pick) clReleaseKernel(all[i]);
                m.kern = all[(size_t)pick];
                std::printf("clCreateKernel failed but clCreateKernelsInProgram created the kernel; using index %d\n", pick);
            } else {
                std::fprintf(stderr,
                             "clCreateKernelsInProgram created nothing: the module is listed but exposes no creatable kernel.\n"
                             "  That is not a string mismatch. The documented next step for the reference arm is the AOT\n"
                             "  route (README section \"The kernel-name question\"), which does not go through this path.\n");
                return 7;
            }
        }
        {
            size_t fn = 0;
            std::string got;
            if (clGetKernelInfo(m.kern, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &fn) == CL_SUCCESS && fn > 0) {
                got.assign(fn, '\0');
                clGetKernelInfo(m.kern, CL_KERNEL_FUNCTION_NAME, fn, &got[0], nullptr);
                dump_bytes("created kernel CL_KERNEL_FUNCTION_NAME", got.data(), got.size());
            }
        }
        size_t kwg = 0, pref = 0;
        cl_ulong lm = 0;
        size_t comp[3] = {0, 0, 0};
        clGetKernelWorkGroupInfo(m.kern, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof kwg, &kwg, nullptr);
        clGetKernelWorkGroupInfo(m.kern, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof pref, &pref, nullptr);
        clGetKernelWorkGroupInfo(m.kern, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof lm, &lm, nullptr);
        clGetKernelWorkGroupInfo(m.kern, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof comp, comp, nullptr);
        std::printf("kernel: CL_KERNEL_WORK_GROUP_SIZE %zu, preferred multiple %zu, local mem %llu B, compile work-group (%zu,%zu,%zu)\n",
                    kwg, pref, (unsigned long long)lm, comp[0], comp[1], comp[2]);
        cl_uint live = 0;
        if (clGetKernelInfo(m.kern, CL_KERNEL_NUM_ARGS, sizeof live, &live, nullptr) == CL_SUCCESS) {
            std::printf("CL_KERNEL_NUM_ARGS %u against %d\n", live, nargs);
            if (live != (cl_uint)nargs) { std::fprintf(stderr, "the live kernel takes %u arguments; the ABI dump said %d\n", live, nargs); return 9; }
        }
        if (kwg < wg) {
            std::fprintf(stderr, "the reference's work-group %zu exceeds the kernel's %zu on this device\n", wg, kwg);
            return 8;
        }
        m.work_group_limit = kwg;
    }

    // --- dry: everything but the enqueue, per shape and module ---
    if (dry) {
        for (const Case& c : cases) {
            for (const Module& m : mods) {
                const int qtiles = (c.shape.lq + QR - 1) / QR;
                const int qgroups = (qtiles + qtw - 1) / qtw;
                const size_t n = (size_t)qgroups * qtw * HQ;
                const size_t w = (size_t)(G * qtw);
                std::printf("[dry] %s / %s: global %zu work-group %zu would be enqueued\n", c.shape.name, m.label, n, w);
            }
        }
        std::printf("--dry: every step above done, the enqueue skipped. Nothing was timed, nothing compared.\n");
        return 0;
    }

    // --- per shape, per module: args, one checked launch, timing ---
    for (const Case& c : cases) {
        const Shape& sh = c.shape;
        const int qtiles = (sh.lq + QR - 1) / QR;
        const int qgroups = (qtiles + qtw - 1) / qtw;
        const size_t n_global = (size_t)qgroups * qtw * HQ;
        const double pairs = pairs_of(sh);
        const double flops = pairs * HQ * 4.0 * D;   // q.k and p.v, 2 mul-adds each

        for (const Module& m : mods) {
            std::printf("\n=== %s / module %s ===\n", sh.name, m.label);
            auto make_buf = [&](cl_mem_flags fl, size_t bytes, const void* host) {
                cl_int e = CL_SUCCESS;
                cl_mem b = clCreateBuffer(ctx, fl, bytes, const_cast<void*>(host), &e);
                if (e != CL_SUCCESS) { std::fprintf(stderr, "clCreateBuffer(%zu) -> %s\n", bytes, cl_err(e)); std::exit(10); }
                return b;
            };
            cl_mem dq  = make_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  c.fx.q.size() * 2,     c.fx.q.data());
            cl_mem dk  = make_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  c.fx.k.size() * 2,     c.fx.k.data());
            cl_mem dv  = make_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  c.fx.v.size() * 2,     c.fx.v.data());
            cl_mem do_ = make_buf(CL_MEM_READ_WRITE, (size_t)sh.lq * HQ * D * 4, nullptr);
            cl_mem dt  = make_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  c.fx.table.size() * 2, c.fx.table.data());
            cl_mem dsn = make_buf(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  c.fx.sink.size() * 4,  c.fx.sink.data());

            int idx = 0;
            CLCHECK(clSetKernelArg(m.kern, idx++, sizeof(cl_mem), &dq));
            CLCHECK(clSetKernelArg(m.kern, idx++, sizeof(cl_mem), &dk));
            CLCHECK(clSetKernelArg(m.kern, idx++, sizeof(cl_mem), &dv));
            CLCHECK(clSetKernelArg(m.kern, idx++, sizeof(cl_mem), &do_));
            CLCHECK(clSetKernelArg(m.kern, idx++, sizeof(cl_mem), &dt));
            CLCHECK(clSetKernelArg(m.kern, idx++, sizeof(cl_mem), &dsn));
            // Lq Lk Hq Hk G qpos0 kpos0 window E QTW, the functor's field order
            const int ints[10] = {sh.lq, sh.lk, HQ, HK, G, qpos0_of(sh), KPOS0, WINDOW, E, qtw};
            for (int v : ints) CLCHECK(clSetKernelArg(m.kern, idx++, sizeof(int), &v));
            const float scale_log2 = (float)(LOG2E / std::sqrt((double)D));
            CLCHECK(clSetKernelArg(m.kern, idx++, sizeof(float), &scale_log2));
            std::printf("arguments set: %d (6 cl_mem, 10 int, 1 float), scale_log2 %.9f, global %zu work-group %zu\n",
                        idx, scale_log2, n_global, wg);

            cl_event ev;
            err = clEnqueueNDRangeKernel(q, m.kern, 1, nullptr, &n_global, &wg, 0, nullptr, &ev);
            if (err != CL_SUCCESS) {
                std::fprintf(stderr, "clEnqueueNDRangeKernel -> %s (%d)\n", cl_err(err), err);
                return 11;
            }
            CLCHECK(clWaitForEvents(1, &ev));
            std::vector<float> out((size_t)sh.lq * HQ * D);
            CLCHECK(clEnqueueReadBuffer(q, do_, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));

            Compare cmp = compare_out(out, c.ref.o, sh.lq);
            std::printf("comparison: max abs error %.6g, max row relative L2 %.6g, mean row KL %.6g nats (%d rows, %d NaN)\n",
                        cmp.max_abs, cmp.max_rel, cmp.mean_kl, cmp.nrows, cmp.nnan);
            std::printf("  (row KL = the row's 128 output components normalised to a simplex - NOT a quality\n"
                "   number: it says nothing about the model's answer, it is not comparable across\n"
                "   shapes. Quality is the campaign's answer-level KL; here max abs error and row\n"
                "   relative L2 are the comparison)\n");
            if (m.is_reference_flags) {
                const bool ok = cmp.max_abs <= ref_tol_abs && cmp.mean_kl <= ref_tol_kl && cmp.nnan == 0;
                std::printf("tolerance (reference-flags module): max abs <= %.3g, mean row KL <= %.3g -> %s\n",
                            ref_tol_abs, ref_tol_kl, ok ? "inside" : "EXCEEDED");
                if (!ok) std::fprintf(stderr, "warning: the reference-flags module is outside the stated tolerance; its times are a separate arm\n");
            }

            Timed t;
            if (!time_kernel(q, m.kern, n_global, wg, reps, t)) return 12;
            std::printf("timing: %d launches, median %.3f ms (min %.3f, max %.3f, spread %.3f), %.2f TFLOP/s over %.0f pairs*H*4*D\n",
                        reps, t.median, t.min, t.max, t.max - t.min, flops / (t.median / 1000.0) / 1e12, pairs);

            if (!dump_path.empty()) {
                const bool many = cases.size() > 1 || mods.size() > 1;
                std::string path = dump_path;
                if (many) path += "." + std::string(sh.name) + "." + m.label + ".bin";
                FILE* f = std::fopen(path.c_str(), "wb");
                if (f) {
                    std::fprintf(f, "# %s / module %s: Lq %d Lk %d Hq %d Hk %d D %d window %d E %d qpos0 %d kpos0 %d qtw %d\n",
                                 sh.name, m.label, sh.lq, sh.lk, HQ, HK, D, WINDOW, E, qpos0_of(sh), KPOS0, qtw);
                    std::fwrite(out.data(), 4, out.size(), f);
                    std::fwrite(c.ref.o.data(), 4, c.ref.o.size(), f);
                    std::fclose(f);
                    std::printf("dump: %s (kernel f32 then oracle f32)\n", path.c_str());
                }
            }

            clReleaseMemObject(dq); clReleaseMemObject(dk); clReleaseMemObject(dv);
            clReleaseMemObject(do_); clReleaseMemObject(dt); clReleaseMemObject(dsn);
        }
    }

    for (const Module& m : mods) { clReleaseKernel(m.kern); clReleaseProgram(m.prog); }
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
