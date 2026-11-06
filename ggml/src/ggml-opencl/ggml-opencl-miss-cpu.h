#pragma once
// patch 0056: the CPU half of the expert cache's miss path (ggml-opencl-miss-cpu.cpp)
#include "ggml.h"
#include <cstddef>
#include <cstdint>

bool   ggml_cl_miss_cpu_ok(void);                     // AVX2 + FMA + F16C at run time
bool   ggml_cl_miss_cpu_type(ggml_type t);            // IQ2_S, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, Q2_0, IQ4_NL (split bank: rows_iq4nl)
// a token's quantized activation for rows of type t over n values: its bytes, and the quantizer
size_t ggml_cl_miss_cpu_act_size(ggml_type t, int64_t n);
void   ggml_cl_miss_cpu_quant(ggml_type t, const float * x, int64_t n, void * y);
// out[k][r] = row r of the ne0 x ne1 matrix `slice` (in planes when `planar`) . act[k], r in [r0, r1), k < nt
void   ggml_cl_miss_cpu_rows(ggml_type t, const void * slice, bool planar, int64_t ne0, int64_t ne1,
                             const void * const * act, int nt, float * const * out, int r0, int r1);
// the same for an IQ4_NL matrix in the bank's split form (scales d, quants q over the whole tensor); its first block blk0
void   ggml_cl_miss_cpu_rows_iq4nl(const void * d, const void * q, int64_t blk0, int64_t ne0, int64_t ne1,
                                   const void * const * act, int nt, float * const * out, int r0, int r1);
// patch 0075: 0 off (the rows' block-ahead hints and the page-touch pass off),
// 1 the rows' hints of the next block's payload field two blocks ahead, 2 also
// the page-touch pass at publish (the bank is written at load, so its pages are
// resident: the pass is a measurement arm, the window's fault count decides)
void   ggml_cl_miss_cpu_set_prefetch(int level);
int    ggml_cl_miss_cpu_prefetch_ahead(void);   // blocks the row hint names ahead
// one byte a 4 KiB page of [p, p+bytes): the page-touch pass (see the .cpp's premise)
void   ggml_cl_miss_cpu_touch(const void * p, size_t bytes);
// n IQ4_NL blocks from i0 on, as GGUF blocks, out of a USM bank that holds the tensor's scales and its
// quants as two arrays (ggml_cl_usm_set_tensor): the rows ggml-cpu's dot products read
void   ggml_cl_miss_cpu_join_iq4nl(const void * d, const void * q, int64_t i0, int64_t n, void * dst);
