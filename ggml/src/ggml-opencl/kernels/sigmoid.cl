#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// sigmoid
//------------------------------------------------------------------------------

kernel void kernel_sigmoid_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int n
) {
    // 0072: exit the rounded uniform launch's tail
    if ((int) get_global_id(0) >= n) {
        return;
    }

    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = 1.0f / (1.0f + exp(-src0[get_global_id(0)]));
}

kernel void kernel_sigmoid_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int n
) {
    // 0072: exit the rounded uniform launch's tail
    if ((int) get_global_id(0) >= n) {
        return;
    }

    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = 1.0f / (1.0f + exp(-src0[get_global_id(0)]));
}

// sigmoid(src0) * src1, the same shape, contiguous (patch 0049, after llama.cpp
// CUDA's {UNARY, MUL} fusion with SIGMOID): kernel_sigmoid_f32's formula, then
// the product
kernel void kernel_sigmoid_mul_f32(
        global float * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);
    const size_t i = get_global_id(0);
    const float s = 1.0f / (1.0f + exp(-src0[i]));
    dst[i] = s * src1[i];
}
