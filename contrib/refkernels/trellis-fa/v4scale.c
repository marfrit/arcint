// v4scale.c: the q4_0 V build's scale in P, one expression, on the host (no card). For one q4_0 block (ggml's
// quantize_row_q4_0_ref: d = max / -8, max the signed value of largest magnitude, d stored as f16) and one key's p,
// pe = f16(p * ds) with
//   old   ds = fmin(128 d, 65504)            (0225 at 218c97ed: q8_0's line)
//   clamp ds = clamp(128 d, -65504, 65504)   (the review's one-line fix alone)
//   fix   ds = clamp(8 d, -65504, 65504)     (0225 now: V4_PSCALE 8, PV_PSCALE in the final division)
// and the block's contribution as the kernel forms it: acc = sum_j pe (1024 + n_j) - 1032 pe (f32; the DPAS on
// 0x6400 | n and the ns term), out_j = acc_j / scale, against p d (n_j - 8) in double.
// Build: cc -O1 -std=c11 v4scale.c -lm   (x86-64: _Float16 since GCC 12)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float h(float x) { return (float) (_Float16) x; }

typedef struct { float d; int n[32]; } blk;

static blk quant(const float *x) {   // ggml quantize_row_q4_0_ref, one block
    float amax = 0, max = 0;
    for (int j = 0; j < 32; ++j) if (fabsf(x[j]) > amax) { amax = fabsf(x[j]); max = x[j]; }
    const float d = max / -8, id = d ? 1.0f / d : 0.0f;
    blk b; b.d = h(d);
    for (int j = 0; j < 32; ++j) { int q = (int) (x[j] * id + 8.5f); b.n[j] = q > 15 ? 15 : q; }
    return b;
}

// the worst relative error over the block's 32 outputs, or inf / nan when an output is not finite
static double run(const blk *b, float p, int arm) {
    float ds;
    if (arm == 0) ds = fminf(128 * b->d, 65504.0f);
    else if (arm == 1) ds = fmaxf(-65504.0f, fminf(128 * b->d, 65504.0f));
    else ds = fmaxf(-65504.0f, fminf(8 * b->d, 65504.0f));
    const float sc = arm == 2 ? 8.0f : 128.0f;
    const float pe = h(p * ds);
    double worst = 0, ref_max = 0;
    for (int j = 0; j < 32; ++j) ref_max = fmax(ref_max, fabs((double) p * b->d * (b->n[j] - 8)));
    for (int j = 0; j < 32; ++j) {
        const float acc = pe * (1024.0f + b->n[j]) + (-1032.0f * pe);
        const float out = acc / sc;
        if (!isfinite(out)) return out;
        const double ref = (double) p * b->d * (b->n[j] - 8);
        worst = fmax(worst, fabs(out - ref) / ref_max);
    }
    return worst;
}

int main(void) {
    static const char *name[3] = { "old   fmin(128 d)       ", "clamp clamp(128 d) alone", "fix   clamp(8 d), /8    " };
    // block maxima: inside the old range, at its edge, past it (positive max -> d < 0), and a negative max (d > 0)
    const float maxv[] = { 100.0f, 4000.0f, 4094.0f, 4200.0f, 6000.0f, 8000.0f, 65504.0f, -6000.0f, -65504.0f };
    const float ps[] = { 1.0f, 0.9f, 0.5f, 0.01f };
    int bad_fix = 0, old_nonfinite = 0;
    for (size_t m = 0; m < sizeof maxv / sizeof *maxv; ++m) {
        float x[32];
        for (int j = 0; j < 32; ++j) x[j] = maxv[m] * (float) ((j * 37 % 31) - 15) / 16.0f;   // values of both signs
        x[5] = maxv[m];                                                                         // the block's maximum
        const blk b = quant(x);
        for (size_t k = 0; k < sizeof ps / sizeof *ps; ++k) {
            printf("max %9.1f d %9.3f p %4.2f |", maxv[m], b.d, ps[k]);
            for (int arm = 0; arm < 3; ++arm) {
                const double e = run(&b, ps[k], arm);
                printf(" %s %-9.3g|", name[arm], e);
                if (arm == 0 && !isfinite(e)) old_nonfinite++;
                if (arm == 2 && !(e < 1e-3)) bad_fix++;
            }
            printf("\n");
        }
    }
    printf("old: %d non-finite cells; fix: %d cells above 1e-3 relative (f16 rounding of pe is ~5e-4)\n", old_nonfinite, bad_fix);
    return bad_fix != 0 || old_nonfinite == 0;   // red: the old line must give non-finite cells; green: the fix none
}
