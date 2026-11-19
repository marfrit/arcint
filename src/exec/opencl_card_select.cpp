#include "exec/opencl_card_select.h"

namespace lgc {

// stub: the interface only, the selection follows
std::vector<ClDeviceDesc> opencl_enumerate() { return {}; }

ClPick pick_cards(const std::vector<ClDeviceDesc>&, const std::vector<std::string>&) {
    return ClPick{{}, "opencl_card_select: the card selection is not built yet"};
}

std::string ggml_opencl_devices_value(const std::vector<ClDeviceDesc>&) { return {}; }

}  // namespace lgc
