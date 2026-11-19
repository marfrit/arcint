#pragma once

// The card selection behind --device, and the GGML_OPENCL_DEVICES value it
// produces. One OpenCL platform may hold several cards (the B60 and the B70 on
// one platform), so the selection is by device, not by platform.
//
// opencl_enumerate() is the one function that calls OpenCL: every device in
// enumeration order -- platform order, then the platform's devices, every type,
// numbered as GGML_OPENCL_DEVICES numbers them ("P:D", contrib/llama.cpp patch
// 0023). With ZE_AFFINITY_MASK set the runtime hides the masked cards, so the
// enumeration is already the visible set.
//
// pick_cards() and ggml_opencl_devices_value() are pure: no OpenCL, no
// environment, for the device-free unit tests.

#include <cstdint>
#include <string>
#include <vector>

namespace lgc {

struct ClDeviceDesc {
    int         platform = -1;
    int         device   = -1;
    bool        gpu      = false;
    std::string name;
    uint64_t    mem_bytes = 0;   // CL_DEVICE_GLOBAL_MEM_SIZE: the card's total, not its free memory
};

// cards empty and error non-empty: the start is refused. Otherwise error is
// empty and cards holds the picked cards in the order the words were given.
struct ClPick {
    std::vector<ClDeviceDesc> cards;
    std::string               error;
};

// every OpenCL device in enumeration order (platform order, then the platform's
// devices, every type, numbered as GGML_OPENCL_DEVICES numbers them)
std::vector<ClDeviceDesc> opencl_enumerate();

// pure: no OpenCL, no environment
ClPick pick_cards(const std::vector<ClDeviceDesc>& all, const std::vector<std::string>& wanted);

// "P:D[,P:D]" in card order
std::string ggml_opencl_devices_value(const std::vector<ClDeviceDesc>& cards);

}  // namespace lgc
