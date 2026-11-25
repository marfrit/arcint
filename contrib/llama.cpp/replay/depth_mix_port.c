// Replay of kernels/depth_mix.cl's index math on Inkling 2 Mini's real prefill strides
// (B60 card evidence, 2026-11-24): a token a work-group, per-lane registers kept between
// phases, the two reductions over emulated local memory, and a range check on every global
// access. Fed the exported graph's strides: block store [2304, 5, tokens] with nb[1] 9216 and
// nb[2] 46080, cur nb[1] 9216, the query parent nb[1] 9216, dst [2304, tokens, nq].
// Compared against the CPU op (ggml-cpu/ops.cpp) on the same buffers.
//   gcc -O1 -g -fsanitize=address -std=c11 depth_mix_port.c -o port -lm
//   ASAN_OPTIONS=detect_leaks=0 ./port <tokens> <nq> <k> <with_norm>
// Recorded (all clean, OOB 0, fp32 agreement with the CPU op): 512 1 1 0/1, 512 3 3 0/1,
// 2048 1 4 1, 512 3 1 0. It is the evidence that the strides are not the card fault's cause;
// it gives every argument its own buffer at offset 0, so it could not see the defect the card
// showed (the kernel never added the norm weight's offset). Superseded by the replay of the
// kernel's own text in tests/test_depth_mix.cpp; kept as the record of the alias check.
// faithful host port of kernels/depth_mix.cl on the model's real strides: the same
// phases, the same local-memory layout, per-lane registers kept between phases, and a
// range check on every global access (ASan on the buffers is available but the checks
// are explicit so they also run without ASan).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define LS 256
#define NSTATE 7
#define NQMAX 4
#define EL 9

typedef struct { float * data; size_t nbytes; } buf_t;

static int g_bad = 0;
static void chk(const buf_t * b, const char * what, size_t off, size_t bytes) {
    if (off + bytes > b->nbytes) { fprintf(stderr, "OOB %s: %zu + %zu > %zu\n", what, off, bytes, b->nbytes); g_bad++; }
}

static float red[64*LS], tmp8[8*32];
static void reduce_sim_off(int nq_, int off);
static void reduce_sim(int nq_) {   // all lanes and the barriers, as the GPU runs it
    reduce_sim_off(nq_, 0);
}
static void reduce_sim_off(int np, int off) {
    for (int base = 0; base < np; base += 8) {
        for (int lane = 0; lane < LS; ++lane) {
            const int sub = lane / 32, sl = lane % 32, p = base + sub;
            float v = 0.0f;
            if (p < np) for (int j = sl; j < LS; j += 32) v += red[(off+p)*LS + j];
            tmp8[sub*32 + sl] = v;
        }
        for (int st = 16; st >= 1; st >>= 1)
            for (int lane = 0; lane < LS; ++lane) {
                const int sub = lane / 32, sl = lane % 32;
                if (sl < st) tmp8[sub*32 + sl] += tmp8[sub*32 + sl + st];
            }
        for (int lane = 0; lane < LS; ++lane) {
            const int sub = lane / 32, sl = lane % 32, p = base + sub;
            if (sl == 0 && p < np) red[(off+p)*LS] = tmp8[sub*32];
        }
    }
}

typedef struct { float x[NSTATE][EL]; float inv[NSTATE]; float pw[NQMAX][NSTATE];
                 float tot[NQMAX]; float outv[NQMAX][EL]; } lane_t;
static lane_t L[LS];

int main(int argc, char ** argv) {
    const int ne0 = 2304, nt = argc>1?atoi(argv[1]):512, nq = argc>2?atoi(argv[2]):1, k = argc>3?atoi(argv[3]):1, has_nw = argc>4?atoi(argv[4]):1;
    const size_t rows = 5;
    buf_t blk = { NULL, (size_t)ne0*rows*nt*4 };
    buf_t cur = { NULL, (size_t)ne0*nt*4 };
    buf_t qry = { NULL, (size_t)ne0*nq*4 };
    buf_t nw  = { NULL, (size_t)ne0*(nq>1?nq:1)*4 };
    buf_t dst = { NULL, (size_t)ne0*nt*nq*4 };
    buf_t * bs[] = { &blk, &cur, &qry, &nw, &dst };
    for (int i=0;i<5;i++){ bs[i]->data = malloc(bs[i]->nbytes); for (size_t j=0;j<bs[i]->nbytes/4;j++) bs[i]->data[j]=0.1f*((int)(j%97)-48); }
    buf_t dst_cpu = { NULL, dst.nbytes }; dst_cpu.data = calloc(dst.nbytes/4, sizeof(float));
    const size_t nb1=(size_t)ne0*4, nb2=rows*nb1, offb=(size_t)(rows-1-k)*nb1;
    const size_t nbd1=nb1, nbd2=(size_t)ne0*nt*4;
    const float eps_st=1e-6f, eps_out=1e-5f, scale=1.0f/sqrtf((float)ne0);
    const size_t nbq1=(nq>1?nb1:0), nbn1=(nq>1?nb1:0);
    const int ns=k+1, el=ne0/LS, np=ns*(1+nq);

    for (int t=0;t<nt;t++) {
        memset(red,0,sizeof red);
        // phase 0: states into registers, ss + dot partials into local
        for (int lane=0;lane<LS;lane++) {
            for (int s=0;s<ns;s++) {
                size_t off = (s<k)? offb+(size_t)s*nb1+(size_t)t*nb2 : (size_t)t*nb1;
                const buf_t * bb = (s<k)? (&blk) : (&cur);
                chk(bb, (s<k)?"blk":"cur", off+(size_t)lane*4, 4);
                chk(bb, (s<k)?"blk":"cur", off+(size_t)(lane+(el-1)*LS)*4, 4);
                for (int i=0;i<EL;i++) L[lane].x[s][i] = (i<el)? bb->data[(off+(size_t)(lane+i*LS)*4)/4] : 0.0f;
            }
            for (int s=0;s<ns;s++){ float v=0; for (int i=0;i<EL;i++) if (i<el) v+=L[lane].x[s][i]*L[lane].x[s][i]; red[s*LS+lane]=v; }
            for (int q=0;q<nq;q++){ size_t qoff=(size_t)q*nbq1;
                chk(&qry,"qry",qoff+(size_t)lane*4,4); chk(&qry,"qry",qoff+(size_t)(lane+(el-1)*LS)*4,4);
                for (int s=0;s<ns;s++){ float d=0; for (int i=0;i<EL;i++) if (i<el) d+=L[lane].x[s][i]*qry.data[(qoff+(size_t)(lane+i*LS)*4)/4];
                    red[(ns+q*ns+s)*LS+lane]=d; } }
        }
        if (getenv("DBG2") && t==73) { double ps=0; for (int l=0;l<LS;l++) ps+=red[0*LS+l]; printf("partials ss0 sum %.6f\n", ps); }
        // phase 1: reduce, inv
        reduce_sim(np);
        for (int lane=0;lane<LS;lane++) for (int s=0;s<ns;s++) L[lane].inv[s]=1.0f/sqrtf(red[s*LS]/(float)ne0+eps_st);
        // phase 2: weights, weighted sum, (norm partials)
        for (int lane=0;lane<LS;lane++) {
            for (int q=0;q<nq;q++) {
                float mx=-INFINITY;
                for (int s=0;s<ns;s++) mx=fmaxf(mx, scale*L[lane].inv[s]*red[(ns+q*ns+s)*LS]);
                float tot=0;
                for (int s=0;s<ns;s++){ L[lane].pw[q][s]=expf(scale*L[lane].inv[s]*red[(ns+q*ns+s)*LS]-mx); tot+=L[lane].pw[q][s]; }
                L[lane].tot[q]=tot;
                for (int i=0;i<EL;i++) if (i<el) { float o=0; for (int s=0;s<ns;s++) o+=L[lane].pw[q][s]*L[lane].x[s][i]; L[lane].outv[q][i]=o/tot; }
            }
            if (has_nw) { for (int q=0;q<nq;q++){ float v=0; for (int i=0;i<EL;i++) if (i<el) v+=L[lane].outv[q][i]*L[lane].outv[q][i]; red[(np+q)*LS+lane]=v; } }
        }
        if (getenv("DBG2") && t==73) { int l=5; printf("t73 lane5: inv %.6f %.6f pw %.6g %.6g tot %.6f x %.6f %.6f\n",
            L[l].inv[0], L[l].inv[1], L[l].pw[0][0], L[l].pw[0][1], L[l].tot[0], L[l].x[0][0], L[l].x[1][0]);
            printf("  red %.6g %.6g %.6g %.6g\n", red[0*LS], red[1*LS], red[2*LS], red[3*LS]);
            printf("  x0 lane5 e0..e3 %.4f %.4f %.4f %.4f\n", L[l].x[0][0], 0.f,0.f,0.f); }
        // phase 3: (norm reduce, write)
        if (has_nw) {
            reduce_sim_off(nq, np);
            for (int lane=0;lane<LS;lane++) for (int q=0;q<nq;q++) {
                const float onorm=1.0f/sqrtf(red[(np+q)*LS]/(float)ne0+eps_out);
                size_t woff=(size_t)q*nbn1, doff=(size_t)q*nbd2+(size_t)t*nbd1;
                chk(&nw,"nw",woff+(size_t)lane*4,4); chk(&nw,"nw",woff+(size_t)(lane+(el-1)*LS)*4,4);
                chk(&dst,"dst",doff+(size_t)lane*4,4); chk(&dst,"dst",doff+(size_t)(lane+(el-1)*LS)*4,4);
                for (int i=0;i<EL;i++) if (i<el) dst.data[(doff+(size_t)(lane+i*LS)*4)/4]=L[lane].outv[q][i]*onorm*nw.data[(woff+(size_t)(lane+i*LS)*4)/4];
            }
        } else {
            for (int lane=0;lane<LS;lane++) for (int q=0;q<nq;q++) {
                size_t doff=(size_t)q*nbd2+(size_t)t*nbd1;
                chk(&dst,"dst",doff+(size_t)lane*4,4); chk(&dst,"dst",doff+(size_t)(lane+(el-1)*LS)*4,4);
                for (int i=0;i<EL;i++) if (i<el) dst.data[(doff+(size_t)(lane+i*LS)*4)/4]=L[lane].outv[q][i];
            }
        }
    }

    // the CPU op (ops.cpp), same tensors
    for (int t=0;t<nt;t++) {
        float v[NSTATE][2304];
        for (int s=0;s<ns;s++){ size_t off=(s<k)? offb+(size_t)s*nb1+(size_t)t*nb2 : (size_t)t*nb1;
            const buf_t * bb=(s<k)?&blk:&cur; for (int i=0;i<ne0;i++) v[s][i]=bb->data[(off+(size_t)i*4)/4]; }
        for (int q=0;q<nq;q++) {
            float sc[NSTATE], ss[NSTATE];
            for (int s=0;s<ns;s++){ double a=0,d=0; for (int i=0;i<ne0;i++){ a+=(double)v[s][i]*v[s][i]; d+=(double)v[s][i]*qry.data[((size_t)q*nbq1+(size_t)i*4)/4]; } ss[s]=(float)a; sc[s]=(float)d; }
            float mx=-INFINITY;
            for (int s=0;s<ns;s++){ const float inv=1.0f/sqrtf(ss[s]/(float)ne0+eps_st); sc[s]=scale*inv*sc[s]; mx=fmaxf(mx,sc[s]); }
            double tot=0, pv[NSTATE]; for (int s=0;s<ns;s++){ pv[s]=exp((double)sc[s]-mx); tot+=pv[s]; }
            float o[2304]; for (int i=0;i<ne0;i++){ double acc=0; for (int s=0;s<ns;s++) acc+=pv[s]/tot*v[s][i]; o[i]=(float)acc; }
            if (getenv("DBG2") && t==73 && q==0) printf("cpu v0[0..3] %.4f %.4f %.4f %.4f  v1[0..3] %.4f %.4f %.4f %.4f\n", v[0][0],v[0][1],v[0][2],v[0][3], v[1][0],v[1][1],v[1][2],v[1][3]);
            if (getenv("DBG2") && t==73 && q==0) printf("cpu t73 ss %.6f %.6f sc %.6f %.6f tot %.6f\n", ss[0], ss[1], sc[0], sc[1], tot);
            if (has_nw) { double sq=0; for (int i=0;i<ne0;i++) sq+=(double)o[i]*o[i];
                const float inv=1.0f/sqrtf((float)sq/(float)ne0+eps_out);
                for (int i=0;i<ne0;i++) o[i]=o[i]*inv*nw.data[((size_t)q*nbn1+(size_t)i*4)/4]; }
            for (int i=0;i<ne0;i++) dst_cpu.data[((size_t)q*nbd2+(size_t)t*nbd1+(size_t)i*4)/4]=o[i];
        }
    }
    if (getenv("DBG2")) {
        int t=73; size_t off0=offb+(size_t)t*nb2, off1=(size_t)t*nb1;
        double a0=0,d0=0,a1=0,d1=0;
        for (int i=0;i<ne0;i++){ float x0=blk.data[(off0+(size_t)i*4)/4]; float x1=cur.data[(off1+(size_t)i*4)/4]; float qv=qry.data[(size_t)i*4/4];
            a0+=x0*x0; d0+=x0*qv; a1+=x1*x1; d1+=x1*qv; }
        printf("plain t73 ss %.4f %.4f dot %.4f %.4f  (scaled %.4f %.4f)\n", a0,a1,d0,d1,
               scale*(1.0/sqrt(a0/ne0+eps_st))*d0, scale*(1.0/sqrt(a1/ne0+eps_st))*d1);
    }
    // debug: token 0, q 0, lane 0's intermediates and the first few outputs
    if (getenv("DBG")) {
        printf("inv0 %.6f mx %.6f tot %.6f outv[0][0] %.6f\n", L[0].inv[0], 0.0, L[0].tot[0], L[0].outv[0][0]);
        for (int i=0;i<4;i++) printf("out[%d] port %.6f cpu %.6f\n", i, dst.data[i], dst_cpu.data[i]);
    }
    double maxd=0,maxv=0; size_t first=(size_t)-1; for (size_t j=0;j<dst.nbytes/4;j++){ double d=fabs((double)dst.data[j]-dst_cpu.data[j]); if (d>maxd) { maxd=d; first=j; } maxv=fmax(maxv,fabs((double)dst_cpu.data[j])); }
    if (maxd>1e-4){ printf("first mismatch at flat %zu = q %zu t %zu i %zu : port %.6f cpu %.6f\n", first, first/((size_t)ne0*nt), (first/(size_t)ne0)%nt, first%(size_t)ne0, dst.data[first], dst_cpu.data[first]); }
    printf("nt=%d nq=%d k=%d has_nw=%d : port vs CPU op max|diff| %.6g (max|value| %.6g) OOB %d\n",nt,nq,k,has_nw,maxd,maxv,g_bad);
    return g_bad?2:(maxd/(maxv+1e-30)<1e-4?0:1);
}
