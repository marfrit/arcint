// 0079: the DG2 + LEO limit for image1d_buffer weights.
//
// Under the runtime's LEO mode (OpenCL over Level Zero, EnableLEO=1) on a DG2
// device, an IMAGE1D_BUFFER (CL_R, UINT32) larger than 16,384 texels aborts the
// process in RENDER_SURFACE_STATE::setWidth: LEO's DG2 path writes the texel
// count into the 14-bit width field of the surface state instead of the
// buffer-surface encoding (docs/campaigns/vram-user-space.md, stage 2a; the
// reproducer is tools/leo_image1d_buffer_repro.c). 16,384 texels fit the field,
// 16,385 do not. The default driver is fine at every size and the B60 under
// LEO is fine at every size.
//
// The backend therefore asks this header before it creates a weight image: the
// image is allowed unless the device is DG2 in LEO mode and the texel count is
// above the field's 16,384. The driver's own CL_DEVICE_IMAGE_MAX_BUFFER_SIZE is
// still honoured; it does not report the DG2+LEO limit (the runtime accepts the
// allocation and aborts at the surface-state write).
//
// LEO is detected from CL_DEVICE_VERSION containing "LEO" (the device string
// reads "OpenCL 3.0 LEO"; measured-here, the gate in vram-user-space.md). DG2
// is detected from the Intel device id (CL_DEVICE_ID_INTEL, the A770 is 0x56a0;
// measured-here, the card host's PCI id 8086:56a0) or, without that extension,
// from the device name's Arc A-series / Arc Pro A-series models.
//
// One function, device-free: the rule, the LEO predicate and the DG2 name
// predicate are testable without a GPU; tests/test-opencl-image-limit.cpp does
// that, including the 16,384 / 16,385 boundary.
#ifndef GGML_OPENCL_IMAGE_LIMIT_H
#define GGML_OPENCL_IMAGE_LIMIT_H

#include <stddef.h>
#include <string.h>

// the largest texel count the 14-bit RENDER_SURFACE_STATE width can encode
#define GGML_CL_DG2_LEO_MAX_IMAGE_TEXELS ((size_t) 16384)

struct ggml_cl_image_mode {
    int    leo;          // CL_DEVICE_VERSION carried "LEO"
    int    dg2;          // the device is a DG2 (Intel Arc A-series)
    size_t driver_max;   // CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, 0 = not queried
};

// 1 when a weight image of `texels` may be created in this mode, 0 when not.
// The narrow form: on DG2 under LEO only images above the field bound are
// refused, so a per-tensor choice keeps every image that fits.  A zero texel
// count is never an image.
static inline int ggml_cl_image_allowed(const struct ggml_cl_image_mode mode, size_t texels) {
    if (texels == 0) {
        return 0;
    }
    if (mode.driver_max != 0 && texels > mode.driver_max) {
        return 0;
    }
    if (mode.leo && mode.dg2 && texels > GGML_CL_DG2_LEO_MAX_IMAGE_TEXELS) {
        return 0;
    }
    return 1;
}

// LEO's device string is "OpenCL 3.0 LEO" (measured-here); the default driver
// has no such suffix.
static inline int ggml_cl_version_is_leo(const char * version) {
    return version != NULL && strstr(version, "LEO") != NULL;
}

// DG2 models are Intel's Arc A-series and Arc Pro A-series.  The B60 (BMG) is
// "Arc(TM) Pro B60" and an Adreno device is not Intel.
static inline int ggml_cl_name_is_dg2(const char * name) {
    if (name == NULL || strstr(name, "Intel") == NULL) {
        return 0;
    }
    return strstr(name, "Arc(TM) A") != NULL ||
           strstr(name, "Arc A")    != NULL ||
           strstr(name, "Arc(TM) Pro A") != NULL ||
           strstr(name, "Arc Pro A")    != NULL;
}

#endif // GGML_OPENCL_IMAGE_LIMIT_H
