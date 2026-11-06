// 0072: uniform work groups for the OpenCL backend.
//
// OpenCL 2.0+ lets a launch's global size be ragged (not a multiple of its
// local size) as a non-uniform work group. The individual driver accepts that,
// LEO (OpenCL over Level Zero, EnableLEO=1) does not: clEnqueueNDRangeKernel
// returns CL_INVALID_WORK_GROUP_SIZE (-54) at the first ragged global size, and
// every following launch of the op fails with it (the gate in
// docs/campaigns/vram-user-space.md, stage 2a).
//
// The fix is one rounding rule for the whole backend: every dimension of the
// global size is rounded up to the next multiple of its local size, so each
// work-group is full, and the work-items of the rounded tail exit in the
// kernel. Most kernels already carry the bound (an n/ne/n_blk argument they
// check); the ones that did not got one in this patch. The rounding lives in
// this one header, so the host launches and the device-free test share it.
//
// A launch whose global size is already a multiple (tile-derived grids, local
// size 1) is unchanged by the rule: rounding is idempotent.
//
// Adreno builds round nothing: `GGML_OPENCL_USE_ADRENO_KERNELS` makes both
// helpers return their input unchanged, so the 170 Adreno-only launches keep
// the pin's ragged global sizes and the Adreno kernels need no bound of their
// own (several decompose their axes by modulo, e.g. kernel_transpose_32). LEO
// is not an Adreno driver: it exists in Intel's Level Zero OpenCL
// implementation, which an Adreno device never runs through.
#ifndef GGML_OPENCL_UNIFORM_H
#define GGML_OPENCL_UNIFORM_H

#include <stddef.h>

// round one global dimension up to a whole number of local work-items
#ifdef __cplusplus
static inline
#endif
size_t ggml_cl_round_up_global(size_t global, size_t local) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // 0072: the Adreno class stays ragged (see above)
    (void) local;
    return global;
#else
    if (local == 0) {
        return global;
    }
    return ((global + local - 1) / local) * local;
#endif
}

// round every dimension of a launch; local_work_size == NULL (the driver picks
// it) is left alone, and so is a zero local dimension
#ifdef __cplusplus
static inline
#endif
void ggml_cl_round_up_global_work_size(size_t * global_work_size, const size_t * local_work_size, unsigned int work_dim) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // 0072: the Adreno class stays ragged (see above)
    (void) global_work_size;
    (void) local_work_size;
    (void) work_dim;
#else
    if (local_work_size == NULL) {
        return;
    }
    for (unsigned int i = 0; i < work_dim; ++i) {
        global_work_size[i] = ggml_cl_round_up_global(global_work_size[i], local_work_size[i]);
    }
#endif
}

#endif // GGML_OPENCL_UNIFORM_H
