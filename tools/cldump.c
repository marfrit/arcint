/* cldump.c -- portable OpenCL kernel-capture + serialization shim (campaign
 * served-prefill-determinism, 2026-09-20).
 *
 * The built intel_gpu plugin has no ENABLE_DEBUG_CAPS, so
 * OV_GPU_DUMP_SOURCES_PATH / OV_GPU_DUMP_TENSORS_PATH are inert. This shim
 * intercepts the OpenCL entry points the plugin uses:
 *   - clCreateProgramWithSource / WithIL / WithBinary and clBuildProgram /
 *     clCompileProgram / clLinkProgram -> .cl source, .names, .options, .isa
 *   - clEnqueueNDRangeKernel -> with CLDUMP_SERIALIZE set, calls clFinish()
 *     after EVERY enqueue, forcing full serialization (used to refute the
 *     inter-kernel-overlap hypothesis for the B60 GDN defect).
 *
 * Build:  gcc -shared -fPIC -O2 -o libcldump.so cldump.c -ldl
 * Use:    CLDUMP_DIR=/tmp/cldump LD_PRELOAD=$PWD/libcldump.so <python ...>
 *         CLDUMP_SERIALIZE=1 LD_PRELOAD=$PWD/libcldump.so <python ...>
 *
 * Measured 2026-09-20: with CLDUMP_SERIALIZE=1 the B60 layer0/mixer_out cut
 * still differs on all 7 repeats (233 enqueues each finished before the next),
 * so the defect is a within-kernel/execution-level effect, not overlap.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *cl_program;
typedef unsigned int cl_uint;
typedef void *cl_device_id;
typedef void (*cl_build_callback)(cl_program, void *);
typedef int cl_int;
typedef unsigned int cl_program_info;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_kernel;
typedef void *cl_event;

#define CL_PROGRAM_SOURCE          0x1161
#define CL_PROGRAM_BINARY_SIZES    0x1164
#define CL_PROGRAM_BINARIES        0x1165
#define CL_PROGRAM_NUM_DEVICES     0x1162
#define CL_PROGRAM_KERNEL_NAMES    0x1166

static cl_int (*real_build)(cl_program, cl_uint, const cl_device_id *,
                            const char *, cl_build_callback, void *);
static cl_int (*real_compile)(cl_program, cl_uint, const cl_device_id *,
                              const char *, cl_uint, const cl_program *,
                              const char **, cl_build_callback, void *);
static cl_program (*real_link)(void *, cl_uint, const cl_device_id *, const char *,
                               cl_uint, const cl_program *, cl_build_callback, void *,
                               cl_int *);
static cl_int (*real_getinfo)(cl_program, cl_program_info, size_t, void *, size_t *);
static cl_program (*real_create_src)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
static cl_program (*real_create_il)(cl_context, const void *, size_t, cl_int *);
static cl_program (*real_create_bin)(cl_context, cl_uint, const cl_device_id *, const size_t *,
                                      const unsigned char **, cl_int *, cl_int *);
static cl_int (*real_enqueue)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                              const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
static cl_int (*real_finish)(cl_command_queue);

static void *ocl_handle(void) {
    static void *h = NULL;
    if (!h) h = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    return h;
}

static void *sym(const char *name) {
    void *p = dlsym(RTLD_NEXT, name);
    if (!p) {
        void *h = ocl_handle();
        if (h) p = dlsym(h, name);
    }
    return p;
}

static void resolve(void) {
    if (!real_build)   real_build   = (cl_int (*)(cl_program, cl_uint, const cl_device_id *,
                            const char *, cl_build_callback, void *))sym("clBuildProgram");
    if (!real_compile) real_compile = (cl_int (*)(cl_program, cl_uint, const cl_device_id *,
                              const char *, cl_uint, const cl_program *,
                              const char **, cl_build_callback, void *))sym("clCompileProgram");
    if (!real_link)    real_link    = (cl_program (*)(void *, cl_uint, const cl_device_id *, const char *,
                               cl_uint, const cl_program *, cl_build_callback, void *,
                               cl_int *))sym("clLinkProgram");
    if (!real_getinfo) real_getinfo = (cl_int (*)(cl_program, cl_program_info, size_t, void *, size_t *))sym("clGetProgramInfo");
    if (!real_create_src) real_create_src = (cl_program (*)(cl_context, cl_uint, const char **, const size_t *, cl_int *))sym("clCreateProgramWithSource");
    if (!real_create_il)  real_create_il  = (cl_program (*)(cl_context, const void *, size_t, cl_int *))sym("clCreateProgramWithIL");
    if (!real_create_bin) real_create_bin = (cl_program (*)(cl_context, cl_uint, const cl_device_id *, const size_t *,
                                      const unsigned char **, cl_int *, cl_int *))sym("clCreateProgramWithBinary");
    if (!real_enqueue) real_enqueue = (cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                              const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *))sym("clEnqueueNDRangeKernel");
    if (!real_finish)  real_finish  = (cl_int (*)(cl_command_queue))sym("clFinish");
}

static int src_counter = 0, il_counter = 0;

static void dump_named(const char *fmt, const void *data, size_t len) {
    const char *dir = getenv("CLDUMP_DIR");
    if (!dir) dir = "/tmp/cldump";
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, fmt);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    if (data && len) fwrite(data, 1, len, f);
    fclose(f);
}

static void logline(const char *fmt, const char *a, long n) {
    const char *dir = getenv("CLDUMP_DIR");
    if (!dir) dir = "/tmp/cldump";
    char path[1024];
    snprintf(path, sizeof(path), "%s/log.txt", dir);
    FILE *f = fopen(path, "a");
    if (f) { fprintf(f, fmt, a, n); fclose(f); }
}

static int counter = 0;

static void dump(const char *suffix, const void *data, size_t len) {
    const char *dir = getenv("CLDUMP_DIR");
    if (!dir) dir = "/tmp/cldump";
    char path[1024];
    snprintf(path, sizeof(path), "%s/program_%03d.%s", dir, counter, suffix);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    if (data && len) fwrite(data, 1, len, f);
    fclose(f);
}

static void capture(cl_program program, const char *options) {
    resolve();
    if (!real_getinfo) { logline("capture: no clGetProgramInfo for %s (%ld)\n", "", 0); counter++; return; }
    logline("capture enter %s (%ld)\n", "", (long)counter);

    size_t sz = 0;
    cl_int rr = real_getinfo(program, CL_PROGRAM_SOURCE, 0, NULL, &sz);
    logline("  SOURCE size query rc/len: %s (%ld)\n", "", (long)(rr==0?(long)sz:-1));
    if (rr == 0 && sz) {
        char *src = calloc(sz + 1, 1);
        rr = real_getinfo(program, CL_PROGRAM_SOURCE, sz, src, NULL);
        logline("  SOURCE read rc: %s (%ld)\n", "", (long)rr);
        if (rr == 0) dump("cl", src, sz);
        free(src);
    }
    sz = 0;
    rr = real_getinfo(program, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &sz);
    if (rr == 0 && sz) {
        char *nm = calloc(sz + 1, 1);
        if (real_getinfo(program, CL_PROGRAM_KERNEL_NAMES, sz, nm, NULL) == 0)
            dump("names", nm, sz);
        free(nm);
    }
    dump("options", options, options ? strlen(options) : 0);
    if (getenv("CLDUMP_BINARIES")) {
        cl_uint ndev = 0;
        if (real_getinfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(ndev), &ndev, NULL) == 0 && ndev > 0) {
            size_t *sizes = calloc(ndev, sizeof(size_t));
            if (real_getinfo(program, CL_PROGRAM_BINARY_SIZES, ndev * sizeof(size_t), sizes, NULL) == 0) {
                char **bufs = calloc(ndev, sizeof(char *));
                for (cl_uint i = 0; i < ndev; ++i) bufs[i] = calloc(sizes[i] ? sizes[i] : 1, 1);
                if (real_getinfo(program, CL_PROGRAM_BINARIES, ndev * sizeof(char *), bufs, NULL) == 0)
                    dump("isa", bufs[0], sizes[0]);
                for (cl_uint i = 0; i < ndev; ++i) free(bufs[i]);
                free(bufs);
            }
            free(sizes);
        }
    }
    logline("capture done %s (%ld)\n", "", (long)counter);
    counter++;
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices,
                      const cl_device_id *device_list, const char *options,
                      cl_build_callback pfn, void *user_data) {
    resolve();
    logline("clBuildProgram called: %s (%ld)\n", options ? options : "", (long)num_devices);
    if (!real_build) { logline("clBuildProgram UNRESOLVED: %s (%ld)\n", "", 0); return -1; }
    cl_int r = real_build(program, num_devices, device_list, options, pfn, user_data);
    capture(program, options);
    return r;
}

cl_int clCompileProgram(cl_program program, cl_uint num_devices,
                        const cl_device_id *device_list, const char *options,
                        cl_uint num_headers, const cl_program *headers,
                        const char **header_names, cl_build_callback pfn, void *user_data) {
    resolve();
    logline("clCompileProgram called: %s (%ld)\n", options ? options : "", (long)num_devices);
    if (!real_compile) { logline("clCompileProgram UNRESOLVED: %s (%ld)\n", "", 0); return -1; }
    return real_compile(program, num_devices, device_list, options, num_headers,
                        headers, header_names, pfn, user_data);
}

cl_program clLinkProgram(void *context, cl_uint num_devices, const cl_device_id *device_list,
                         const char *options, cl_uint num_input_programs,
                         const cl_program *input_programs, cl_build_callback pfn,
                         void *user_data, cl_int *errcode_ret) {
    resolve();
    logline("clLinkProgram called: %s (%ld)\n", options ? options : "", (long)num_input_programs);
    if (!real_link) { logline("clLinkProgram UNRESOLVED: %s (%ld)\n", "", 0);
                      if (errcode_ret) *errcode_ret = -1; return NULL; }
    cl_program p = real_link(context, num_devices, device_list, options,
                             num_input_programs, input_programs, pfn, user_data, errcode_ret);
    if (p) capture(p, options);
    return p;
}


cl_program clCreateProgramWithSource(cl_context context, cl_uint count,
                                     const char **strings, const size_t *lengths,
                                     cl_int *errcode_ret) {
    resolve();
    if (!real_create_src) { if (errcode_ret) *errcode_ret = -1; return NULL; }
    cl_program p = real_create_src(context, count, strings, lengths, errcode_ret);
    if (p) {
        char name[64];
        snprintf(name, sizeof(name), "src_%03d.cl", src_counter++);
        const char *dir = getenv("CLDUMP_DIR"); if (!dir) dir = "/tmp/cldump";
        char path[1024]; snprintf(path, sizeof(path), "%s/%s", dir, name);
        FILE *f = fopen(path, "wb");
        if (f) {
            for (cl_uint i = 0; i < count; ++i) {
                size_t n = lengths && lengths[i] ? lengths[i] : (strings[i] ? strlen(strings[i]) : 0);
                if (strings[i]) fwrite(strings[i], 1, n, f);
            }
            fclose(f);
        }
        logline("clCreateProgramWithSource -> %s (%ld)\n", name, (long)count);
    }
    return p;
}

cl_program clCreateProgramWithIL(cl_context context, const void *il, size_t length,
                                 cl_int *errcode_ret) {
    resolve();
    if (!real_create_il) { if (errcode_ret) *errcode_ret = -1; return NULL; }
    cl_program p = real_create_il(context, il, length, errcode_ret);
    if (p) {
        char name[64];
        snprintf(name, sizeof(name), "il_%03d.spv", il_counter++);
        dump_named(name, il, length);
        logline("clCreateProgramWithIL -> %s (%ld)\n", name, (long)length);
    }
    return p;
}


cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices,
                                     const cl_device_id *device_list, const size_t *lengths,
                                     const unsigned char **binaries, cl_int *binary_status,
                                     cl_int *errcode_ret) {
    resolve();
    if (!real_create_bin) { if (errcode_ret) *errcode_ret = -1; return NULL; }
    cl_program p = real_create_bin(context, num_devices, device_list, lengths, binaries,
                                   binary_status, errcode_ret);
    if (p && binaries && lengths && num_devices) {
        char name[64];
        snprintf(name, sizeof(name), "bin_%03d.isa", il_counter);
        dump_named(name, binaries[0], lengths[0]);
        logline("clCreateProgramWithBinary -> %s (%ld)\n", name, (long)lengths[0]);
        il_counter++;
    }
    return p;
}


cl_int clEnqueueNDRangeKernel(cl_command_queue queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event) {
    resolve();
    logline("clEnqueueNDRangeKernel called (serialize=%s) (%ld)\n",
            getenv("CLDUMP_SERIALIZE") ? "on" : "off", (long)work_dim);
    if (!real_enqueue) return -1;
    cl_int r = real_enqueue(queue, kernel, work_dim, global_work_offset, global_work_size,
                            local_work_size, num_events_in_wait_list, event_wait_list, event);
    if (getenv("CLDUMP_SERIALIZE") && real_finish && r == 0)
        real_finish(queue);
    return r;
}
