// LEO reproducer: an IMAGE1D_BUFFER wider than 16384 texels, sampled in a kernel (argv[1] = texel count)
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
static const char *src = "kernel void k(read_only image1d_buffer_t img, global uint *o, int n){int i=get_global_id(0); if(i<n) o[i]=read_imageui(img,i).x;}";
int main(int argc, char **argv) {
  size_t n = argc > 1 ? strtoul(argv[1], 0, 10) : 1036800; cl_int e; cl_platform_id p[4]; cl_uint np; clGetPlatformIDs(4, p, &np);
  int pi = argc > 2 ? atoi(argv[2]) : 1; cl_device_id d; clGetDeviceIDs(p[pi], CL_DEVICE_TYPE_GPU, 1, &d, 0);
  char name[128]; clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof name, name, 0); printf("device %s, texels %zu\n", name, n);
  cl_context c = clCreateContext(0, 1, &d, 0, 0, &e); cl_command_queue q = clCreateCommandQueueWithProperties(c, d, 0, &e);
  cl_uint *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = (cl_uint)(i * 2654435761u);
  cl_mem b = clCreateBuffer(c, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * 4, h, &e); if (e) { printf("buffer err %d\n", e); return 1; }
  cl_image_format f = {CL_R, CL_UNSIGNED_INT32}; cl_image_desc dd = {CL_MEM_OBJECT_IMAGE1D_BUFFER, n, 0, 0, 0, 0, 0, 0, 0, {b}};
  cl_mem img = clCreateImage(c, CL_MEM_READ_ONLY, &f, &dd, 0, &e); printf("clCreateImage %d\n", e); if (e) return 1;
  cl_mem o = clCreateBuffer(c, CL_MEM_WRITE_ONLY, n * 4, 0, &e);
  cl_program pr = clCreateProgramWithSource(c, 1, &src, 0, &e); e = clBuildProgram(pr, 1, &d, "", 0, 0); printf("build %d\n", e);
  cl_kernel k = clCreateKernel(pr, "k", &e); int ni = (int)n;
  clSetKernelArg(k, 0, sizeof img, &img); clSetKernelArg(k, 1, sizeof o, &o); clSetKernelArg(k, 2, sizeof ni, &ni);
  size_t g = (n + 255) / 256 * 256, l = 256; e = clEnqueueNDRangeKernel(q, k, 1, 0, &g, &l, 0, 0, 0); printf("enqueue %d\n", e);
  cl_uint *r = malloc(n * 4); e = clEnqueueReadBuffer(q, o, CL_TRUE, 0, n * 4, r, 0, 0, 0);
  size_t bad = 0; for (size_t i = 0; i < n; i++) bad += r[i] != h[i]; printf("read %d, mismatches %zu\n", e, bad); return bad != 0;
}
