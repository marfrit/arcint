#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// LEAKY_RELU (patch 0060): the missing unary the ESRGAN upscaler's graph falls
// back to the CPU for, where the split-graph op segfaults. Same six variants as
// unary_ext.cl (f32, f32_4, f16, f16_4, f32_nc, f16_nc); the negative slope is
// the op's own parameter, an extra trailing argument.
//   x > 0: x;  x <= 0: x * slope.
// The CPU reference (ggml-cpu/ops.cpp) computes in f32; the f16 variants read
// and write half and convert, so they match within tolerance.
//------------------------------------------------------------------------------

#define LEAKY_RELU(NAME, SEXPR, VEXPR)                                          \
kernel void kernel_##NAME##_f32(                                               \
        global const float * src0, ulong offset0,                             \
        global       float * dst,  ulong offsetd, float slope, int n) {       \
    if ((int) get_global_id(0) >= n) { return; } /* 0072: rounded tail */ \
    src0 = (global float*)((global char*)src0 + offset0);                     \
    dst  = (global float*)((global char*)dst  + offsetd);                     \
    float x = src0[get_global_id(0)];                                         \
    dst[get_global_id(0)] = (SEXPR);                                          \
}                                                                             \
kernel void kernel_##NAME##_f32_4(                                            \
        global const float4 * src0, ulong offset0,                           \
        global       float4 * dst,  ulong offsetd, float slope, int n) {      \
    if ((int) get_global_id(0) >= n) { return; } /* 0072: rounded tail */ \
    src0 = (global float4*)((global char*)src0 + offset0);                   \
    dst  = (global float4*)((global char*)dst  + offsetd);                   \
    float4 x = src0[get_global_id(0)];                                       \
    dst[get_global_id(0)] = (VEXPR);                                         \
}                                                                             \
kernel void kernel_##NAME##_f16(                                             \
        global const half * src0, ulong offset0,                            \
        global       half * dst,  ulong offsetd, float slope, int n) {        \
    if ((int) get_global_id(0) >= n) { return; } /* 0072: rounded tail */ \
    src0 = (global half*)((global char*)src0 + offset0);                    \
    dst  = (global half*)((global char*)dst  + offsetd);                    \
    float x = src0[get_global_id(0)];                                       \
    dst[get_global_id(0)] = (SEXPR);                                        \
}                                                                            \
kernel void kernel_##NAME##_f16_4(                                          \
        global const half4 * src0, ulong offset0,                          \
        global       half4 * dst,  ulong offsetd, float slope, int n) {       \
    if ((int) get_global_id(0) >= n) { return; } /* 0072: rounded tail */ \
    src0 = (global half4*)((global char*)src0 + offset0);                  \
    dst  = (global half4*)((global char*)dst  + offsetd);                  \
    float4 x = convert_float4(src0[get_global_id(0)]);                     \
    dst[get_global_id(0)] = convert_half4(VEXPR);                          \
}                                                                           \
kernel void kernel_##NAME##_f32_nc(                                         \
        global const char * src0, ulong offset0,                          \
        global       char * dst,  ulong offsetd,                          \
        int ne00, ulong nb00, ulong nb01, ulong nb02, ulong nb03,         \
        ulong nb0, ulong nb1, ulong nb2, ulong nb3, float slope) {        \
    src0 = src0 + offset0; dst = dst + offsetd;                            \
    const int i3 = get_group_id(2);                                       \
    const int i2 = get_group_id(1);                                       \
    const int i1 = get_group_id(0);                                       \
    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {  \
        float x = *(global const float *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00); \
        *(global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = (SEXPR);            \
    }                                                                     \
}                                                                         \
kernel void kernel_##NAME##_f16_nc(                                       \
        global const char * src0, ulong offset0,                          \
        global       char * dst,  ulong offsetd,                          \
        int ne00, ulong nb00, ulong nb01, ulong nb02, ulong nb03,         \
        ulong nb0, ulong nb1, ulong nb2, ulong nb3, float slope) {        \
    src0 = src0 + offset0; dst = dst + offsetd;                            \
    const int i3 = get_group_id(2);                                       \
    const int i2 = get_group_id(1);                                       \
    const int i1 = get_group_id(0);                                       \
    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {  \
        float x = *(global const half *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00); \
        *(global half *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = (SEXPR);            \
    }                                                                     \
}

#define LEAKY_RELU_BODY(NAME) LEAKY_RELU(NAME, x > 0.0f ? x : x * slope, select(x * slope, x, x > (float4)0.0f))

LEAKY_RELU_BODY(leaky_relu)
