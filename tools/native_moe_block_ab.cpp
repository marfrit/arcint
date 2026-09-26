// native_moe_block_ab -- one block IR (an MoE block, or any [1,T,H]-input graph): the CPU
// plugin's exact run against a GPU compile.
//
// The C++ half of tests/python/test_native_lowering_gpu.py for a patched
// runtime the Python binding refuses (pyopenvino asserts that its own build
// number equals the runtime's, and the patch series stamps its own). Reads
// the IR, runs it on CPU (the unfused decode chain in f32: the oracle) and on
// the named GPU with the given properties, on the same random input, and
// prints what the cell asserts on:
//
//   RUNTIME moe_typed=<n> native_nodes=<n>
//   TYPES <runtime layer type>=<count> ...
//   DIFF max_abs=<d> max_want=<w> max_over_band=<r> corr=<c>
//   GOT fnv1a64=<hash of the GPU output bytes>
//   REPEAT n=<N> ms_mean=<m> ms_min=<m>   (only with ARCINT_BLOCK_AB_REPEAT=N)
//   HEAD rows=<k> fnv1a64=<hash of the first k output rows> (ARCINT_BLOCK_AB_HASH_ROWS=k)
//
// The input rows are drawn in order from one seeded stream, so row 0 is the
// same at every T: the first rows' hash at T = 1 against T = 17 asks whether a
// token's output bytes depend on the other tokens of its call (the matrix-unit
// kernel's tile-mates, docs/design-native-dpas-expert-kernel.md gate 5).
//
// ARCINT_BLOCK_AB_REPEAT=N runs the GPU request N more times after the
// checked run, for timing: the first launch after a compile runs before the
// card has clocked up, so one launch is not a kernel's time (2026-09-26: one
// block read 60.3 ms where repeats read about 49). Wall time per run is
// printed; a kernel's own time comes from an OpenCL intercept layer over the
// repeats.
//
// The band is the cell's: 2% of the element plus 1% of its row's RMS.
//
// Build: c++ -std=c++17 -O1 -o native_moe_block_ab tools/native_moe_block_ab.cpp \
//          -I<ov src>/src/core/include -I<ov src>/src/inference/include -L<runtime lib> -lopenvino
// Use:   native_moe_block_ab <moe.xml> <GPU.n> <T> <seed> [KEY=VALUE ...]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " <moe.xml> <GPU.n> <T> <seed> [KEY=VALUE ...]\n";
        return 2;
    }
    const std::string xml = argv[1], dev = argv[2];
    const size_t T = std::stoul(argv[3]);
    ov::AnyMap props;
    for (int i = 5; i < argc; ++i) {
        std::string kv = argv[i];
        auto eq = kv.find('=');
        if (eq == std::string::npos) {
            std::cerr << "bad property " << kv << "\n";
            return 2;
        }
        props[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    ov::Core core;
    auto cpu = core.compile_model(core.read_model(xml), "CPU");
    // the input's own rank: [1, T, H] (an emitter block) or [T, H] (the
    // token-flattened form the paged graph feeds its projections)
    const auto in_shape = cpu.input().get_partial_shape();
    const size_t H = in_shape[in_shape.size() - 1].get_length();
    ov::Tensor x(ov::element::f32, in_shape.size() == 2 ? ov::Shape{T, H} : ov::Shape{1, T, H});
    std::mt19937 rng(static_cast<unsigned>(std::stoul(argv[4])));
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t i = 0; i < x.get_size(); ++i) x.data<float>()[i] = nd(rng);

    auto rq_cpu = cpu.create_infer_request();
    rq_cpu.set_input_tensor(x);
    rq_cpu.infer();
    const ov::Tensor want = rq_cpu.get_output_tensor();

    auto gpu = core.compile_model(core.read_model(xml), dev, props);
    auto rq_gpu = gpu.create_infer_request();
    rq_gpu.set_input_tensor(x);
    rq_gpu.infer();
    ov::Tensor got_t = rq_gpu.get_output_tensor();

    size_t moe_typed = 0, native_nodes = 0;
    std::map<std::string, size_t> layer_types;
    for (const auto& n : gpu.get_runtime_model()->get_ordered_ops()) {
        const auto& rt = n->get_rt_info();
        std::string t = n->get_type_name();
        auto it = rt.find("layerType");
        if (it != rt.end()) t = it->second.as<std::string>();
        std::string tl = t;
        for (auto& c : tl) c = static_cast<char>(std::tolower(c));
        ++layer_types[t];
        if (tl.find("moe") != std::string::npos) ++moe_typed;
        if (n->get_friendly_name().find("MOECompressedNative") != std::string::npos) ++native_nodes;
    }
    std::cout << "RUNTIME moe_typed=" << moe_typed << " native_nodes=" << native_nodes << "\n";
    std::cout << "TYPES";
    for (const auto& [t, c] : layer_types) std::cout << " " << t << "=" << c;
    std::cout << "\n";

    // the GPU output may be f16 or f32; read both as double
    auto at = [](const ov::Tensor& t, size_t i) -> double {
        if (t.get_element_type() == ov::element::f16) return static_cast<float>(t.data<ov::float16>()[i]);
        return t.data<float>()[i];
    };
    const size_t n = want.get_size();
    if (got_t.get_size() != n) {
        std::cout << "SHAPE_MISMATCH want=" << n << " got=" << got_t.get_size() << "\n";
        return 1;
    }
    // the GPU output's own bytes, FNV-1a 64: two GPU runs compare exactly
    {
        uint64_t h = 1469598103934665603ull;
        const auto* b = static_cast<const unsigned char*>(got_t.data());
        for (size_t i = 0; i < got_t.get_byte_size(); ++i) h = (h ^ b[i]) * 1099511628211ull;
        std::cout << "GOT fnv1a64=" << std::hex << h << std::dec << "\n";
        if (const char* hr = std::getenv("ARCINT_BLOCK_AB_HASH_ROWS")) {
            const size_t k = std::min<size_t>(std::stoul(hr), T);
            const size_t row_bytes = got_t.get_byte_size() / T;
            uint64_t hh = 1469598103934665603ull;
            for (size_t i = 0; i < k * row_bytes; ++i) hh = (hh ^ b[i]) * 1099511628211ull;
            std::cout << "HEAD rows=" << k << " fnv1a64=" << std::hex << hh << std::dec << "\n";
        }
    }
    const size_t row = want.get_shape().back();
    double max_abs = 0, max_want = 0, max_ratio = 0, sw = 0, sg = 0, sww = 0, sgg = 0, swg = 0;
    for (size_t r = 0; r < n / row; ++r) {
        double ss = 0;
        for (size_t j = 0; j < row; ++j) ss += at(want, r * row + j) * at(want, r * row + j);
        const double rms = std::sqrt(ss / row);
        for (size_t j = 0; j < row; ++j) {
            const double w = at(want, r * row + j), g = at(got_t, r * row + j), d = std::fabs(g - w);
            max_abs = std::max(max_abs, d);
            max_want = std::max(max_want, std::fabs(w));
            max_ratio = std::max(max_ratio, d / (2e-2 * std::fabs(w) + 1e-2 * rms));
            sw += w, sg += g, sww += w * w, sgg += g * g, swg += w * g;
        }
    }
    const double corr = (n * swg - sw * sg) / std::sqrt((n * sww - sw * sw) * (n * sgg - sg * sg));
    std::cout << "DIFF max_abs=" << max_abs << " max_want=" << max_want << " max_over_band=" << max_ratio
              << " corr=" << corr << "\n";
    if (const char* rep = std::getenv("ARCINT_BLOCK_AB_REPEAT")) {
        const int reps = std::atoi(rep);
        double sum = 0, best = 1e300;
        for (int i = 0; i < reps; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            rq_gpu.infer();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            sum += ms;
            best = std::min(best, ms);
        }
        if (reps > 0) std::cout << "REPEAT n=" << reps << " ms_mean=" << sum / reps << " ms_min=" << best << "\n";
    }
    return 0;
}
