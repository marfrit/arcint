#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// fill
//------------------------------------------------------------------------------
__kernel void kernel_fill_f32(
        __global float *dst,
        ulong offsetd,
        float v,
        int n

) {
    dst = (global float*)((global char*)dst + offsetd);
    if(get_global_id(0) < n){
        dst[get_global_id(0)] = v;
    }
}

// F16: the value's half bits from the host
__kernel void kernel_fill_f16(
        __global ushort *dst,
        ulong offsetd,
        ushort v,
        int n
) {
    dst = (global ushort*)((global char*)dst + offsetd);
    if(get_global_id(0) < n){
        dst[get_global_id(0)] = v;
    }
}
