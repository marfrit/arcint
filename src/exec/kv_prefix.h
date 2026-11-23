#pragma once

// The prefix a lane may reuse from its KV at a request's start. One function
// for the plain and the drafting prefill paths, so a request that asked for a
// full evaluation cannot reuse on one path and not the other.
//
// `have` is the lane's token history (the KV it still holds), `prompt` the
// request's ids, `limit` how far the path may reuse (the plain path's whole
// prompt, the drafting path's all-but-last), `cache_prompt` the request's
// switch. False: nothing is reused, so the prompt is evaluated in full.
//
// It returns the length of the common prefix. The caller keeps the lane's KV
// after a full prefill: the next request that allows reuse still finds it.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace lgc {

inline std::size_t kv_prefix_common(const std::vector<int>& have, const std::vector<int>& prompt,
                                    std::size_t limit, bool cache_prompt) {
    // A request that asked for a full evaluation reuses nothing.
    if (!cache_prompt) return 0;
    std::size_t common = 0;
    const std::size_t n = std::min({have.size(), prompt.size(), limit});
    while (common < n && have[common] == prompt[common]) ++common;
    return common;
}

}  // namespace lgc
