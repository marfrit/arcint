#pragma OPENCL FP_CONTRACT OFF

kernel void kernel_rope_apply_f32(
        global const char * src0,
        ulong                    offset0,
        global const char * src1,
        ulong                    offset1,
        global       char * dst,
        ulong                    offsetd,
        int     ne00,
        int     ne01,
        int     ne02,
        int     ne03,
        ulong   nb00,
        ulong   nb01,
        ulong   nb02,
        ulong   nb03,
        ulong   nb10,
        ulong   nb11,
        ulong   nb12,
        ulong   nb13,
        int     ne0,
        int     half_dim,
        int     pairs,
        int     interleaved
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    const int gid = get_global_id(0);
    if (gid >= pairs) {
        return;
    }

    // one work-item handles one (pair j, row) -- what the fork's Vulkan kernel does,
    // ported to the dispatch style here. row = (batch, head, token) in that order.
    const int j     = gid % half_dim;
    const int row   = gid / half_dim;
    const int token = row % ne02;
    const int head  = (row / ne02) % ne01;
    const int batch = row / (ne02 * ne01);

    const int i0 = interleaved != 0 ? 2*j           : j;
    const int i1 = interleaved != 0 ? 2*j + 1       : j + half_dim;

    const ulong x_row  = head*nb01 + token*nb02 + batch*nb03;
    const ulong matrix = j*nb12 + token*nb13;

    const float a = *((global const float *) (src0 + x_row + i0*nb00));
    const float b = *((global const float *) (src0 + x_row + i1*nb00));
    const float c = *((global const float *) (src1 + matrix));
    const float s = *((global const float *) (src1 + matrix + nb11));

    float y0;
    float y1;
    if (interleaved != 0) {
        // pe stores [[cos, -sin], [sin, cos]]
        const float neg_s = *((global const float *) (src1 + matrix + nb10));
        const float c1    = *((global const float *) (src1 + matrix + nb10 + nb11));
        y0 = a*c + b*neg_s;
        y1 = a*s + b*c1;
    } else {
        y0 = a*c - b*s;
        y1 = a*s + b*c;
    }

    // dst is contiguous F32 [head_dim, tokens, heads*batch]: nb0 == sizeof(float),
    // row stride ne0 floats.
    const ulong out_row = (ulong) row * (ulong) ne0 * sizeof(float);
    *((global float *) (dst + out_row + i0*sizeof(float))) = y0;
    *((global float *) (dst + out_row + i1*sizeof(float))) = y1;
}
