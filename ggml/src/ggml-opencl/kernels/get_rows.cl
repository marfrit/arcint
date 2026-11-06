#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#define QK4_0                   32

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};


//------------------------------------------------------------------------------
// dequantize_q4_0_f32, dequantize_q4_0_f16
//------------------------------------------------------------------------------
void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}


//------------------------------------------------------------------------------
// get_rows
//------------------------------------------------------------------------------
kernel void kernel_get_rows_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nchunks = get_num_groups(0) / ne10;
    int g       = get_group_id(0);
    int i10     = g / nchunks;
    int chunk   = g - i10 * nchunks;
    int i11     = get_group_id(1);
    int i12     = get_group_id(2);

    int r = ((global int *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    global float * dst_row = (global float *) ((global char *) dst  + i12*nb3 + i11*nb2 + i10*nb1);
    global float * src_row = (global float *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03);

    int span  = (ne00 + nchunks - 1) / nchunks;
    int start = chunk * span;
    int end   = min(start + span, ne00);

    for (int ind = start + get_local_id(0); ind < end; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

// int32 rows, copied as ints (an expert cache's residency tables)
kernel void kernel_get_rows_i32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global int * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global int*)((global char*)dst + offsetd);

    int nchunks = get_num_groups(0) / ne10;
    int g       = get_group_id(0);
    int i10     = g / nchunks;
    int chunk   = g - i10 * nchunks;
    int i11     = get_group_id(1);
    int i12     = get_group_id(2);

    int r = ((global int *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    global int * dst_row = (global int *) ((global char *) dst  + i12*nb3 + i11*nb2 + i10*nb1);
    global int * src_row = (global int *) ((global char *) src0 + r*nb01 + i11*nb02 + i12*nb03);

    int span  = (ne00 + nchunks - 1) / nchunks;
    int start = chunk * span;
    int end   = min(start + span, ne00);

    for (int ind = start + get_local_id(0); ind < end; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_get_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        if (ind >= ne00) {
            return;
        }
        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
            ((global half *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
    }
}

kernel void kernel_get_rows_q4_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int NL = 2;

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16 temp;
        if (ind >= ne00) {
            return;
        }
        dequantize_q4_0_f32(
            ((global struct block_q4_0 *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03)) + ind/NL, ind%NL, &temp);
        *(((global float16 *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

//------------------------------------------------------------------------------
// GET_ROWS of q8_0 (patch 0036): a quantized KV or QSA indexer cache keeps
// ggml's plain 34-byte blocks (written by SET_ROWS / CPY); a q8_0 weight uploaded
// through set_tensor is flattened into a quant buffer and a scale buffer.
// One work-item an output element, so the reads and writes coalesce.
//------------------------------------------------------------------------------
#define QK8_0 32
struct block_q8_0
{
    half d;
    int8_t qs[QK8_0];
};

kernel void kernel_get_rows_q8_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int i10 = get_group_id(0);
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);

    const int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    global struct block_q8_0 * x = (global struct block_q8_0 *) ((global char *) src0 + r*nb01 + i11*nb02 + i12*nb03);
    global float * y = (global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1);

    for (int i = get_local_id(0); i < ne00; i += get_local_size(0)) {
#ifndef GET_ROWS_Q8_RED   // the red case of the test: the scale of the next block
        y[i] = vload_half(0, &x[i / QK8_0].d) * (float) x[i / QK8_0].qs[i % QK8_0];
#else
        y[i] = vload_half(0, &x[i / QK8_0 + 1].d) * (float) x[i / QK8_0].qs[i % QK8_0];
#endif
    }
}

// the flattened layout: quants at q + 32 b, scales at d + b, b the logical block index
kernel void kernel_get_rows_q8_0_flat(
        global char * q,
        global half * d,
        ulong block0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int i10 = get_group_id(0);
    const int i11 = get_group_id(1);
    const int i12 = get_group_id(2);

    const int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    // nb are in bytes of the logical 34-byte blocks
    const ulong b0 = block0 + (r*nb01 + i11*nb02 + i12*nb03) / (2 + QK8_0);
    global float * y = (global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1);

    for (int i = get_local_id(0); i < ne00; i += get_local_size(0)) {
        const ulong b = b0 + i / QK8_0;
#ifndef GET_ROWS_Q8_RED
        y[i] = vload_half(b, d) * (float) q[b * QK8_0 + i % QK8_0];
#else
        y[i] = vload_half(b + 1, d) * (float) q[b * QK8_0 + i % QK8_0];
#endif
    }
}
