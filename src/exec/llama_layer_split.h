#pragma once

// --llama-layer-split K as llama.cpp's LLAMA_SPLIT_MODE_LAYER over two
// devices: the tensor_split that puts layers [0, K) on the first and [K,
// n_layer) plus the output on the second. Device-free, for the unit tests.
//
// llama.cpp (src/llama-model.cpp, load_tensors, at bed0a85) places
// n_layer + 1 slots (the layers, then the output) with every layer on a GPU:
// slot il goes to the first device whose cumulative normalised split exceeds
// il / (n_layer + 1) (std::upper_bound, in float). n_layer is the GGUF's
// <arch>.block_count, MTP (nextn) layers included.
//
// Splits {K - 0.5, n_layer + 1 - (K - 0.5)} sum to n_layer + 1, so the first
// cut is (K - 0.5) / (n_layer + 1): half a slot from both neighbours, far
// beyond float's rounding (n_layer <= 512). Splits {K, n_layer - K} are not
// exact: with 48 layers and K = 24, layer 24 lands on the first device.

#include <algorithm>
#include <array>

namespace lgc {

// 1 <= k < n_layer: both devices run at least one layer
inline bool llama_layer_split_valid(int n_layer, int k) {
    return n_layer >= 2 && k >= 1 && k < n_layer;
}

inline std::array<float, 2> llama_layer_split_fractions(int n_layer, int k) {
    const float first = static_cast<float>(k) - 0.5f;
    return { first, static_cast<float>(n_layer + 1) - first };
}

// llama-model.cpp's assignment of slot il (0..n_layer; n_layer: the output)
// for n_dev splits, with every layer on a GPU (n_gpu_layers >= n_layer + 1)
inline int llama_layer_split_device(const float* split, int n_dev, int n_layer, int il) {
    float cum[16] = {};
    float sum     = 0.0f;
    for (int i = 0; i < n_dev && i < 16; ++i) {
        sum += split[i];
        cum[i] = sum;
    }
    for (int i = 0; i < n_dev && i < 16; ++i) cum[i] /= sum;
    const int act = n_layer + 1;
    const int dev = static_cast<int>(std::upper_bound(cum, cum + n_dev, static_cast<float>(il) / act) - cum);
    return std::min(dev, n_dev - 1);
}

}  // namespace lgc
