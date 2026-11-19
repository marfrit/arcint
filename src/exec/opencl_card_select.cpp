#include "exec/opencl_card_select.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef ARCINT_LLAMA
#include <CL/cl.h>
#endif

namespace lgc {
namespace {

// "P:D (name)" -- the address every message carries, so the operator can put
// it into GGML_OPENCL_DEVICES or read ggml's own log line.
std::string pd(const ClDeviceDesc& d) {
    return std::to_string(d.platform) + ":" + std::to_string(d.device) + " (" + d.name + ")";
}

std::string all_gpus(const std::vector<ClDeviceDesc>& all) {
    std::string s;
    for (const ClDeviceDesc& d : all) {
        if (!d.gpu) continue;
        s += (s.empty() ? "" : ", ") + pd(d);
    }
    return s.empty() ? "none" : s;
}

// GPU or GPU.N: the N-th GPU over every platform, in enumeration order, the
// non-GPU devices of a CPU platform skipped. -1 when the word is not an index.
int gpu_ordinal(const std::string& w, bool* is_index) {
    *is_index = w == "GPU" || (w.rfind("GPU.", 0) == 0 && w.size() > 4);
    if (!*is_index) return -1;
    if (w == "GPU") return 0;
    const std::string n = w.substr(4);
    for (char c : n)
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            *is_index = false;
            return -1;
        }
    return std::atoi(n.c_str());
}

ClPick refuse(std::string e) { return ClPick{{}, std::move(e)}; }

}  // namespace

std::vector<ClDeviceDesc> opencl_enumerate() {
    std::vector<ClDeviceDesc> out;
#ifdef ARCINT_LLAMA
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return out;
    std::vector<cl_platform_id> ps(np);
    if (clGetPlatformIDs(np, ps.data(), nullptr) != CL_SUCCESS) return out;
    for (cl_uint i = 0; i < np; ++i) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> ds(nd);
        if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_ALL, nd, ds.data(), nullptr) != CL_SUCCESS) continue;
        for (cl_uint j = 0; j < nd; ++j) {
            ClDeviceDesc d;
            d.platform = static_cast<int>(i);
            d.device   = static_cast<int>(j);
            cl_device_type t = 0;
            if (clGetDeviceInfo(ds[j], CL_DEVICE_TYPE, sizeof(t), &t, nullptr) == CL_SUCCESS)
                d.gpu = (t & CL_DEVICE_TYPE_GPU) != 0;
            char name[256] = {};
            if (clGetDeviceInfo(ds[j], CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr) == CL_SUCCESS) d.name = name;
            cl_ulong gm = 0;
            if (clGetDeviceInfo(ds[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gm), &gm, nullptr) == CL_SUCCESS)
                d.mem_bytes = static_cast<uint64_t>(gm);
            out.push_back(std::move(d));
        }
    }
#endif
    return out;
}

ClPick pick_cards(const std::vector<ClDeviceDesc>& all, const std::vector<std::string>& wanted) {
    if (wanted.empty()) return refuse("--device: no card named");

    // the GPUs in enumeration order: what GPU.N counts and what a name matches
    std::vector<const ClDeviceDesc*> gpus;
    for (const ClDeviceDesc& d : all)
        if (d.gpu) gpus.push_back(&d);

    ClPick pick;
    for (const std::string& w : wanted) {
        bool is_index = false;
        const int ord = gpu_ordinal(w, &is_index);
        const ClDeviceDesc* hit = nullptr;
        if (is_index) {
            if (ord < 0 || static_cast<size_t>(ord) >= gpus.size())
                return refuse("--device " + w + ": no OpenCL GPU matches; the GPUs are " + all_gpus(all));
            hit = gpus[static_cast<size_t>(ord)];
        } else {
            std::vector<const ClDeviceDesc*> m;
            for (const ClDeviceDesc* d : gpus)
                if (d->name.find(w) != std::string::npos) m.push_back(d);
            if (m.empty()) return refuse("--device " + w + ": no OpenCL GPU matches; the GPUs are " + all_gpus(all));
            if (m.size() > 1) {
                std::string s;
                for (const ClDeviceDesc* d : m) s += (s.empty() ? "" : ", ") + pd(*d);
                return refuse("--device " + w + ": " + std::to_string(m.size()) + " OpenCL GPUs match --device " + w + ": " + s);
            }
            hit = m[0];
        }
        for (const ClDeviceDesc& c : pick.cards)
            if (c.platform == hit->platform && c.device == hit->device)
                return refuse("--device: " + pd(*hit) + " is named twice");
        pick.cards.push_back(*hit);
    }
    return pick;
}

std::string ggml_opencl_devices_value(const std::vector<ClDeviceDesc>& cards) {
    std::string s;
    for (const ClDeviceDesc& c : cards)
        s += (s.empty() ? "" : ",") + std::to_string(c.platform) + ":" + std::to_string(c.device);
    return s;
}

}  // namespace lgc
