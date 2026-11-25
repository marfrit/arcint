// A window-sized sliding-window (SWA) KV cache (--llama-swa-window): after a
// prefix cut, what the cache still holds decides whether the cut is exact.
//
// With the library's full-size SWA cache (swa_full = true, the default until
// this flag) every prefix trim keeps the whole history, so a lane can trim to
// any shared prefix and resume. With swa_full = false the SWA layers hold only
// n_swa + n_ubatch cells, so llama.cpp's seq_rm() on the SWA cache succeeds
// while silently leaving a hole: the next query at position `common` attends
// over [common - n_swa + 1, common], and the cells below the cache's oldest
// retained cell are gone. The trim is exact only when the retained history
// starts at or below that oldest key; otherwise the lane must resume from a
// checkpoint (or clear and re-prefill), which is what the backend does.
#pragma once

#include <cstddef>

namespace lgc {

// `pos_min` is llama_memory_attn_seq_pos_min() of the lane before the cut (the SWA
// cache's oldest retained position, -1 when it holds nothing), `common` the
// shared prefix in tokens, `n_swa` the model's sliding window (0: no SWA, the
// ordinary attention cache trims exactly). True: trimming to `common` leaves
// every key the next query at `common` can attend to.
inline bool swa_window_trim_complete(int n_swa, long long pos_min, size_t common) {
    if (n_swa <= 0) return true;
    if (pos_min < 0) return false;                       // nothing retained: the window cannot be there
    const long long need_from = common > (size_t) n_swa ? (long long) common - n_swa + 1 : 0;
    return pos_min <= need_from;
}

// Whether this start must not save or restore checkpoints. A checkpoint
// (PARTIAL_ONLY) carries a sliding-window cache's cells (every swa type but none),
// and the restore writes them through views of the K / V tensors. On the pinned
// llama.cpp's OpenCL backend a view write into a q8_0 or q4_0 tensor returns
// without writing, so a restore would bring back the cells' positions and not
// their bytes: the window would attend stale keys and values. 1.8.1 and 1.8.2
// refused every such start. contrib/llama.cpp 0209 performs the write (1.8.3), so
// the refusal now stands only where the binary's llama.cpp lacks it.
// `on_opencl`: the model runs on an OpenCL device; `k_q4q8` / `v_q4q8`: the K / V
// cache type is q8_0 or q4_0; `view_writes`: the llama.cpp tree the binary was
// built against carries 0209 (found at configure, ARCINT_LLAMA_KV_VIEW_WRITES;
// the recipe builds the whole series, so a release always does). A build against
// a tree without it keeps 1.8.1's refusal instead of restoring wrong cells.
inline bool swa_ckpt_restore_refused(int n_swa, bool on_opencl, bool k_q4q8, bool v_q4q8, bool view_writes) {
    return n_swa > 0 && on_opencl && (k_q4q8 || v_q4q8) && !view_writes;
}

}  // namespace lgc
