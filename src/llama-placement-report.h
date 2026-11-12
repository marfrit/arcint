#pragma once

// ARC-LAB 0111: a weight the device declined, placed in host memory, says so.
//
// The loader probes each weight's op on the buffer types of its layer's list
// (weight_buft_supported) and takes the first that the device accepts. When
// the layer asked for a device buffer and the device's supports_op declines the
// probe, the weight lands in a host buffer and the scheduler splits the graph
// around it: no backend path is ever taken for it, so the backends' own
// downgrade counts (ggml_backend_opencl_downgrades) never see it. This header
// holds the rule and the key; llama-model-loader.cpp calls it at the decision
// and llama_placement_downgrades (llama-ext.h) reads the counts.
//
// Device-free on purpose: arcint's unit tests include it (ggml-backend.h's queries only, no device
// is opened).

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace llama_placement {

// The tensor name with every dot-separated part that is all digits made "N"
// (blk.12.ffn_up.weight -> blk.N.ffn_up.weight), so a key covers every layer
// and the key count stays small. A space or a tab (never in a GGUF name, but
// the count text is tab-separated) becomes '_'.
inline std::string weight_pattern(const std::string & name) {
    std::string out;
    out.reserve(name.size());
    size_t i = 0;
    while (i <= name.size()) {
        const size_t dot  = name.find('.', i);
        const size_t end  = dot == std::string::npos ? name.size() : dot;
        bool         digits = end > i;
        for (size_t j = i; j < end && digits; ++j) {
            digits = name[j] >= '0' && name[j] <= '9';
        }
        if (digits) {
            out += 'N';
        } else {
            for (size_t j = i; j < end; ++j) {
                const char c = name[j];
                out += (c == ' ' || c == '\t' || c == '\n') ? '_' : c;
            }
        }
        if (dot == std::string::npos) {
            break;
        }
        out += '.';
        i = dot + 1;
    }
    return out;
}

// Whether a buffer type is a device's own memory, from what ggml says of it: not a host buffer, and
// owned by a GPU, an integrated GPU or the meta device of a split. is_host alone cannot tell: the CPU's
// extra buffer types (CPU_REPACK, AMX) are host memory with a repacked layout and leave is_host unset,
// which makes them "not host". They come first in the CPU list when extra buffer types are on, so a
// rule on is_host took the input layer's list for a device list and reported every token_embd
// (measured on the cards, 0111's first form).
inline bool in_device_memory(bool buft_is_host, enum ggml_backend_dev_type owner) {
    return !buft_is_host && (owner == GGML_BACKEND_DEVICE_TYPE_GPU || owner == GGML_BACKEND_DEVICE_TYPE_IGPU ||
                             owner == GGML_BACKEND_DEVICE_TYPE_META);
}

inline bool in_device_memory(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    return dev != nullptr && in_device_memory(ggml_backend_buft_is_host(buft), ggml_backend_dev_type(dev));
}

// Whether a selection is a device decline: the layer's list asked for device memory first, the probe
// on it failed, and the buffer type chosen instead is not device memory (a CPU buffer, a CPU extra
// buffer type, a device's pinned host buffer). A list that starts outside device memory (the input
// layer, which is always the CPU list: token_embd, per_layer_token_embd, every LLM_TENSOR_LAYER_INPUT
// tensor; layers and an output not offloaded) never reports; neither does a fall from one device
// buffer type to another (a split buffer to the device's own). The caller passes only selections made
// by the probe: a tensor_buft_override and a lazily read tensor are placements somebody asked for.
inline bool declined_to_host(ggml_backend_buffer_type_t first, ggml_backend_buffer_type_t chosen) {
    return in_device_memory(first) && !in_device_memory(chosen);
}

// The key without the "src=llama " that the log line and the count text put in
// front, as ggml-opencl's keys: the declined probe op, the weight's type and
// name pattern, the buffer type it got.
inline std::string key(const char * op, const char * type, const std::string & name, const char * buft) {
    return std::string("op=") + (op ? op : "?") + " type=" + (type ? type : "?") + " weight=" + weight_pattern(name) +
           " buft=" + (buft ? buft : "?") + " reason=device-declines";
}

// The counts: a key's count is the number of weights placed under it, summed
// over the models the process loaded (a draft model's weights add to the
// target's). Reset with the process.
struct registry {
    std::mutex                      mu;
    std::map<std::string, uint64_t> counts;

    // true the first time a key occurs: the caller logs it then
    bool add(const std::string & k) {
        std::lock_guard<std::mutex> lock(mu);
        return counts[k]++ == 0;
    }

    // "src=llama <key>\t<count>\n" a key, ggml_backend_opencl_downgrades' format
    std::string text() {
        std::lock_guard<std::mutex> lock(mu);
        std::string out;
        for (const auto & kv : counts) {
            out += "src=llama " + kv.first + "\t" + std::to_string(kv.second) + "\n";
        }
        return out;
    }
};

inline registry & global() {
    static registry r;
    return r;
}

} // namespace llama_placement
