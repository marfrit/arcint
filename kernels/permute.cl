// Permute (0,2,1,3) on BFYX, written against the macros the GPU plugin actually
// injects for custom layers: INPUT0_DIMS and INPUT0_PITCHES (arrays), plus
// INPUT0_TYPE / OUTPUT0_TYPE. The SIZE_X / FEATURE_NUM family that built-in
// kernels use is not available here.
//
// Dimension order in the arrays is BFYX, so DIMS[0..3] = B,F,Y,X.
__kernel void lgc_permute(__global const INPUT0_TYPE* src, __global OUTPUT0_TYPE* dst) {
    const uint IB = (uint)INPUT0_DIMS[0], IF = (uint)INPUT0_DIMS[1];
    const uint IY = (uint)INPUT0_DIMS[2], IX = (uint)INPUT0_DIMS[3];

    // out = in.transpose(0,2,1,3)  ->  OUT dims are (IB, IY, IF, IX)
    const uint OF = IY, OY = IF, OX = IX;

    // Indices are named after OUTPUT coordinates: y walks OY (= input F),
    // f walks OF (= input Y). The in_off below reads in[b][y][f][x], i.e.
    // input dims (B, F=y, Y=f, X) -- that inversion IS the transpose.
    const uint x  = (uint)get_global_id(0);
    const uint y  = (uint)get_global_id(1);
    const uint bf = (uint)get_global_id(2);
    const uint f  = bf % OF;
    const uint b  = bf / OF;
    if (x >= OX || y >= OY || b >= IB) return;

    // in[b][y][f][x] with the plugin's own pitches, so any padding is honoured
    const uint in_off = (uint)INPUT0_OFFSET
                      + b * (uint)INPUT0_PITCHES[0] + y * (uint)INPUT0_PITCHES[1]
                      + f * (uint)INPUT0_PITCHES[2] + x * (uint)INPUT0_PITCHES[3];
    const uint out_off = ((b * OF + f) * OY + y) * OX + x;
    dst[out_off] = (OUTPUT0_TYPE)src[in_off];
}
