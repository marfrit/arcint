#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// rms_norm
//------------------------------------------------------------------------------
// This kernel depends on subgroup size.
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm(
        global void * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        float eps,
        local float * sum, // Note, the size depends on number of subgroups
        float post,       // a following GGML_OP_SCALE folded in (1 when none, patch 0049)
        uint  opaque      // 0 from the host: the compiler cannot regroup (x * scale) * post through it
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float * x_scalar = (global float *) x;
    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    all_sum = sub_group_reduce_add(all_sum);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = all_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // broadcast
    for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
       if (get_local_id(0) < i) {
           sum[get_local_id(0)] += sum[get_local_id(0) + i];
       }
    }
    if (get_local_id(0) == 0) {
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {
            sum[0] += x_scalar[i];
        }
        sum[0] /= ne00;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const float mean  = sum[0];
    const float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    global float * y_scalar = (global float *) y;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        // rounded as the norm and the scale were: x * scale first (the norm's
        // store), then * post (the scale's); under -cl-fast-relaxed-math IGC
        // regrouped the plain form to x * (scale * post) and the text changed
        y[i00] = as_float4(as_uint4(x[i00] * scale) ^ opaque) * post;
    }
    if (get_local_id(0) == 0) {
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {
            y_scalar[i00] = as_float(as_uint(x_scalar[i00] * scale) ^ opaque) * post;
        }
    }
}

//------------------------------------------------------------------------------
// rms_norm_mul
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_mul(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float eps,
        local float * sum
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    // The size of sum is sizeof(float)*subgroup_size.
    // Each subgroup writes its partial sum to this array.
    // So the number of subgroups per workgroup for this kernel cannot exceed the subgroup size.
    // This is generally true -
    // for subgroup size 64, workgroup size should be less than 4096 (the max is usually 1024).
    if (get_sub_group_id() == 0) {
        sum[get_sub_group_local_id()] = 0.0f;
    }

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float4 * f = (global float4 *) (src1 + (i03%ne13)*nb13 + (i02%ne12)*nb12 + (i01%ne11)*nb11);

    float sumf = 0;

    // parallel sum
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = sub_group_reduce_add(sumf);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
    //   if (get_local_id(0) < i) {
    //       sum[get_local_id(0)] += sum[get_local_id(0) + i];
    //   }
    //}
    //if (get_local_id(0) == 0) {
    //    sum[0] /= ne00;
    //}

    //barrier(CLK_LOCAL_MEM_FENCE);

    sumf = sum[get_sub_group_local_id()];
    sumf = sub_group_reduce_add(sumf);

    float mean  = sumf / ne00;
    float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + i03*nb3 + i02*nb2 + i01*nb1);
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = (x[i00] * scale) * f[i00%(ne10/4)];
    }
}

//------------------------------------------------------------------------------
// rms_norm + mul (norm weight) + add (residual), fused. Mirrors
// kernel_rms_norm_mul with an extra residual operand src2: computes
//   y = (rmsnorm(x) * w) + g
// in one dispatch, removing one kernel launch + one global round-trip per
// residual block (the dominant per-layer adjacency on Gemma matformers).
//------------------------------------------------------------------------------
kernel void kernel_rms_norm_mul_add(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne20,
        int ne21,
        int ne22,
        int ne23,
        ulong nb21,
        ulong nb22,
        ulong nb23,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float eps,
        local float * sum
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    dst  = dst  + offsetd;

    if (get_sub_group_id() == 0) {
        sum[get_sub_group_local_id()] = 0.0f;
    }

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float4 * f = (global float4 *) (src1 + (i03%ne13)*nb13 + (i02%ne12)*nb12 + (i01%ne11)*nb11);
    global float4 * g = (global float4 *) (src2 + (i03%ne23)*nb23 + (i02%ne22)*nb22 + (i01%ne21)*nb21);

    float sumf = 0;

    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = sub_group_reduce_add(sumf);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    sumf = sum[get_sub_group_local_id()];
    sumf = sub_group_reduce_add(sumf);

    float mean  = sumf / ne00;
    float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + i03*nb3 + i02*nb2 + i01*nb1);
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = (x[i00] * scale) * f[i00%(ne10/4)] + g[i00%(ne20/4)];
    }
}

//------------------------------------------------------------------------------
// rms_norm + mul(norm weight) + add(residual) + mul(scalar scale), fused.
// Computes y = ((rmsnorm(x) * w) + g) * s, where s is a broadcast SCALAR (e.g.
// Gemma-4 layer_output_scale). Folds the trailing per-layer l_out scale-mul into
// the residual-norm kernel: one extra dispatch + global round-trip saved per
// layer. src3 points at the single scale value.
//------------------------------------------------------------------------------
kernel void kernel_rms_norm_mul_add_scale(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * src3,
        ulong offset3,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne20,
        int ne21,
        int ne22,
        int ne23,
        ulong nb21,
        ulong nb22,
        ulong nb23,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float eps,
        local float * sum
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    src3 = src3 + offset3;
    dst  = dst  + offsetd;

    const float sc = *((global float *) src3);

    if (get_sub_group_id() == 0) {
        sum[get_sub_group_local_id()] = 0.0f;
    }

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float4 * f = (global float4 *) (src1 + (i03%ne13)*nb13 + (i02%ne12)*nb12 + (i01%ne11)*nb11);
    global float4 * g = (global float4 *) (src2 + (i03%ne23)*nb23 + (i02%ne22)*nb22 + (i01%ne21)*nb21);

    float sumf = 0;

    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = sub_group_reduce_add(sumf);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    sumf = sum[get_sub_group_local_id()];
    sumf = sub_group_reduce_add(sumf);

    float mean  = sumf / ne00;
    float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + i03*nb3 + i02*nb2 + i01*nb1);
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = ((x[i00] * scale) * f[i00%(ne10/4)] + g[i00%(ne20/4)]) * sc;
    }
}

//------------------------------------------------------------------------------
// rms_norm · norm weight · interleaved M-RoPE, fused (patch 0074).
// The q/k norm and its rotation sit adjacent on every attention layer of the
// served models (qwen35/qwen4exp: a 1-D [head_dim] weight, mode 40). This
// computes y = rope(rmsnorm(x) * w) in one dispatch and drops the normed row's
// store/load between the two kernels. The epsilon, the weight factor and the
// rotation's cos/sin are kernel_rms_norm_mul's and kernel_rope_multi_f32's
// expressions. The normed value rounds in two steps here, (x * scale) then * w,
// through the `opaque` guard against IGC regrouping — kernel_rms_norm_mul has no
// such guard (`code`), so it may fold x * (scale * w) itself and the two paths'
// bits are not promised to agree either way; the answer-level bar and the
// md5/argmax tripwire are the gate. Only the interleaved M-RoPE form (mode
// GGML_ROPE_TYPE_IMROPE) is handled, which is what both served models build.
//------------------------------------------------------------------------------
float rope_yarn_ramp_fused(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// rope.cl's rope_yarn, rope_yarn_corr_factor and rope_yarn_corr_dims, verbatim
// in a second program (this one needs its subgroup reduction, rope.cl's does
// not), so the fused cos/sin is the rotation's own expression.
float2 rope_yarn_fused(float theta_extrap, float freq_scale, float2 corr_dims, int i0, float ext_factor, float mscale) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_fused(corr_dims.s0, corr_dims.s1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return (float2)(cos(theta) * mscale, sin(theta) * mscale);
}

float rope_yarn_corr_factor_fused(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

float2 rope_yarn_corr_dims_fused(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow) {
    return (float2)(
        max(0.0f,         floor(rope_yarn_corr_factor_fused(n_dims, n_ctx_orig, beta_fast, freq_base))),
        min(n_dims - 1.0f, ceil(rope_yarn_corr_factor_fused(n_dims, n_ctx_orig, beta_slow, freq_base)))
    );
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_mul_rope_multi_f32(
        global char * src0,
        ulong offset0,
        global char * wsrc,
        ulong offsetw,
        global int  * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections,
        int n_offs,
        float eps,
        uint opaque,
        local float * sum
) {
    src0 = src0 + offset0;
    global float * w = (global float *) (wsrc + offsetw);
    // the offsets are BYTES on every operand, as kernel_rope_multi_f32 does it:
    // `src1` is an int pointer, so `src1 + offset1` would step offset1 ELEMENTS
    // (x4 bytes) and read a wrong region -- window 43's diagnostic read zeros for
    // all four position streams (its print's offset and reads proved it)
    src1 = (global int *)((global char *) src1 + offset1);
    src2 = src2 + offset2;
    global float * ff = (global float *) src2;
    dst  = dst  + offsetd;

    if (get_sub_group_id() == 0) {
        sum[get_sub_group_local_id()] = 0.0f;
    }

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);

    float sumf = 0;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = sub_group_reduce_add(sumf);

    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    sumf = sum[get_sub_group_local_id()];
    sumf = sub_group_reduce_add(sumf);

    const float scale = 1.0f/sqrt(sumf / ne00 + eps);

    float2 corr_dims = rope_yarn_corr_dims_fused(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    // the host refuses a zero sections sum (the per-op path asserts it); this keeps
    // the kernel defined if the argument ever arrives empty: ic % 0 is undefined
    const int sect_sum = sections.s0 + sections.s1 + sections.s2 + sections.s3;
    const int sect_dims = sect_sum > 0 ? sect_sum : 1;
    const float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 >= n_offs && i0 < n_offs + n_dims) {
            int iw = i0 - n_offs; // relative idx
            int ic = iw/2;

            const int sector = ic % sect_dims;
            float theta_base = 0.0f;

            // the interleaved M-RoPE's position layout, kernel_rope_multi_f32's
            // is_imrope branch (the sections' t/h/w/e positions per token)
            if (sector % 3 == 1 && sector < 3 * sections.s1) { // h
                theta_base = (float) pos[i02 + ne02 * 1];
            } else if (sector % 3 == 2 && sector < 3 * sections.s2) { // w
                theta_base = (float) pos[i02 + ne02 * 2];
            } else if (sector % 3 == 0 && sector < 3 * sections.s0) { // t
                theta_base = (float) pos[i02 + ne02 * 0];
            } else { // e
                theta_base = (float) pos[i02 + ne02 * 3];
            }

            const float theta = theta_base * pow(freq_base, inv_ndims*iw);

            const float freq_factor = src2 != src0 ? ff[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn_fused(theta/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor);

            global float * src      = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + (n_offs + ic)*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i03*nb3  + i02*nb2  + i01*nb1  + (n_offs + ic)*nb0);

            // the norm's store then the weight, as kernel_rms_norm_mul rounds them
            const float x0 = as_float(as_uint(src[0] * scale) ^ opaque) * w[n_offs + ic];
            const float x1 = as_float(as_uint(src[n_dims/2] * scale) ^ opaque) * w[n_offs + ic + n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * const src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i03*nb3  + i02*nb2  + i01*nb1  + i0*nb0);

            dst_data[0] = as_float(as_uint(src[0] * scale) ^ opaque) * w[i0];
            dst_data[1] = as_float(as_uint(src[1] * scale) ^ opaque) * w[i0 + 1];
        }
    }
}

//------------------------------------------------------------------------------
// ARC-LAB 0114: the nGPT residual update of Dory's graph in one kernel,
//   b = rms_norm(x, eps1) * s1,  t = a + (b - a) * w,  y = rms_norm(t, eps2) * s2
// (the graph's RMS_NORM SCALE SUB MUL ADD RMS_NORM SCALE), one work-group a row,
// w [ne00] broadcast over the rows. Each element is read and written by the same
// work-item, after both reductions, so y may lie exactly over a or over x (the
// allocator's in-place placement); any other overlap is refused by the host.
// red != 0 (the test's mutation) reads w as 1.
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_ngpt_slerp_f32(
        global char * src_x,
        ulong off_x,
        global char * src_a,
        ulong off_a,
        global char * src_w,
        ulong off_w,
        global char * dst,
        ulong off_d,
        int ne00,
        ulong nb_x,
        ulong nb_a,
        ulong nb_d,
        float eps1,
        float s1,
        float eps2,
        float s2,
        int red,
        local float * sum
) {
    const int row = get_group_id(0);
    global const float4 * x = (global const float4 *) (src_x + off_x + row*nb_x);
    global const float4 * a = (global const float4 *) (src_a + off_a + row*nb_a);
    global const float4 * w = (global const float4 *) (src_w + off_w);
    global float4 * y = (global float4 *) (dst + off_d + row*nb_d);
    const int n4  = ne00/4;
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const uint sg  = get_sub_group_id();
    const uint nsg = get_local_size(0) / get_max_sub_group_size();   // the host launches a multiple of the sub-group size

    float4 acc = 0.0f;
    for (int i = lid; i < n4; i += lsz) {
        const float4 v = x[i];
        acc += v*v;
    }
    float p = sub_group_reduce_add(acc.s0 + acc.s1 + acc.s2 + acc.s3);
    if (get_sub_group_local_id() == 0) {
        sum[sg] = p;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float tot = 0.0f;
    for (uint k = 0; k < nsg; ++k) {
        tot += sum[k];
    }
    const float r1 = 1.0f/sqrt(tot/ne00 + eps1);
    // every work-item has read the partial sums before they are written again
    barrier(CLK_LOCAL_MEM_FENCE);

    acc = 0.0f;
    for (int i = lid; i < n4; i += lsz) {
        const float4 av = a[i];
        const float4 wv = red ? (float4)(1.0f) : w[i];
        const float4 t  = av + ((x[i]*r1)*s1 - av)*wv;
        acc += t*t;
    }
    p = sub_group_reduce_add(acc.s0 + acc.s1 + acc.s2 + acc.s3);
    if (get_sub_group_local_id() == 0) {
        sum[sg] = p;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    tot = 0.0f;
    for (uint k = 0; k < nsg; ++k) {
        tot += sum[k];
    }
    const float r2 = 1.0f/sqrt(tot/ne00 + eps2);

    for (int i = lid; i < n4; i += lsz) {
        const float4 av = a[i];
        const float4 wv = red ? (float4)(1.0f) : w[i];
        const float4 t  = av + ((x[i]*r1)*s1 - av)*wv;
        y[i] = (t*r2)*s2;
    }
}
