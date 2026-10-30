/*
 * l0_vm_arena_probe.c - proof for docs/campaigns/research-vram-arena.md section 8.5.
 *
 * Reserves a large GPU virtual range with Level Zero virtual memory, maps device-local physical
 * chunks into it, wraps the range in an OpenCL cl_mem through LEO ("Level Zero Executing OpenCL")
 * interop, runs kernels across a chunk boundary, moves a chunk to another address (unmap + map,
 * never two addresses at once), destroys and re-creates a chunk, and prints the process's
 * drm-resident-vram0 (and the card's VRAM used, when xe_drm.h is available) at every step.
 *
 * Build (Level Zero headers from the level-zero dev package, or a prefix given by -I/-L):
 *   gcc -O1 -Wall -o l0_vm_arena_probe tools/l0_vm_arena_probe.c [-I<ze>/include -L<ze>/lib] \
 *       -lze_loader -lOpenCL
 * Run on the card host, B60 by PCI address (card window per docs/sop-card-window.md):
 *   NEOReadDebugKeys=1 EnableLEO=1 ./l0_vm_arena_probe 0000:0f:00.0 [chunk_mib [chunks]]
 * Control (must fail at step 1, exit 1):
 *   NEOReadDebugKeys=1 EnableLEO=0 ./l0_vm_arena_probe 0000:0f:00.0
 * Defaults: 512 MiB chunks, 8 chunks created, range = 32 chunk slots (16 GiB). Exit 0 = all checks.
 * Not compiled or run where it was written; checked against ze_api.h 1.x by reading only.
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/ioctl.h>
#if defined(__has_include)
#if __has_include(<drm/xe_drm.h>)
#include <drm/xe_drm.h>
#define HAVE_XE 1
#endif
#endif

/* LEO interop keys (private header cl_ext_private.h; HEAD renames them CL_CONTEXT_L0_HANDLE_INTEL /
 * CL_MEM_L0_HANDLE_INTEL with the same values): pinned by value. */
#define K_L0_CONTEXT 0x42B0 /* CL_L0_CONTEXT_HANDLE, context property */
#define K_L0_MEM_OBJ 0x42B3 /* CL_L0_MEM_OBJ_HANDLE, mem property */
#define K_L0_DEVICE  0x42B4 /* CL_L0_DEVICE_HANDLE, device info */

#define MIB (1ull << 20)
#define GIB (1ull << 30)

static int step_no;
static void fail(const char *msg, const char *detail) {
    fprintf(stderr, "FAIL step %d: %s%s%s\n", step_no, msg, detail ? ": " : "", detail ? detail : "");
    exit(1);
}
static void failn(const char *msg, long v) { char b[64]; snprintf(b, sizeof b, "%ld (0x%lx)", v, v); fail(msg, b); }
#define ZE(call) do { ze_result_t r_ = (call); if (r_ != ZE_RESULT_SUCCESS) failn(#call, (long)r_); } while (0)
#define CLC(err, what) do { if ((err) != CL_SUCCESS) failn(what, (long)(err)); } while (0)
static void step(const char *t) { step_no++; printf("\n== step %d: %s\n", step_no, t); fflush(stdout); }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

/* ---- memory accounting ---- */
/* Sum of drm-resident-vram0 over this process's xe clients, deduped by drm-client-id; bytes. */
static long long resident_vram0(void) {
    DIR *d = opendir("/proc/self/fdinfo"); if (!d) return -1;
    struct dirent *e; long long sum = 0; long long seen[64]; int ns = 0;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char p[128]; snprintf(p, sizeof p, "/proc/self/fdinfo/%s", e->d_name);
        FILE *f = fopen(p, "r"); if (!f) continue;
        char l[256]; int xe = 0; long long id = -1, res = 0;
        while (fgets(l, sizeof l, f)) {
            char u[16] = ""; long long v;
            if (!strncmp(l, "drm-driver:", 11)) xe = strstr(l, "xe") != NULL;
            else if (sscanf(l, "drm-client-id: %lld", &v) == 1) id = v;
            else if (sscanf(l, "drm-resident-vram0: %lld %15s", &v, u) >= 1)
                res = v * (!strcmp(u, "KiB") ? 1024 : !strcmp(u, "MiB") ? MIB : !strcmp(u, "GiB") ? GIB : 1);
        }
        fclose(f);
        if (!xe) continue;
        int dup = 0; for (int i = 0; i < ns; i++) if (seen[i] == id) dup = 1;
        if (dup) continue;
        if (ns < 64) seen[ns++] = id;
        sum += res;
    }
    closedir(d); return sum;
}
static int card_fd = -1;
static long long card_used(void) { /* VRAM used card-wide, bytes; -1 when unavailable */
#ifdef HAVE_XE
    if (card_fd < 0) return -1;
    struct drm_xe_device_query q = { .query = DRM_XE_DEVICE_QUERY_MEM_REGIONS };
    if (ioctl(card_fd, DRM_IOCTL_XE_DEVICE_QUERY, &q) || !q.size) return -1;
    struct drm_xe_query_mem_regions *m = calloc(1, q.size); q.data = (uintptr_t)m;
    long long used = -1;
    if (!ioctl(card_fd, DRM_IOCTL_XE_DEVICE_QUERY, &q)) {
        used = 0;
        for (unsigned i = 0; i < m->num_mem_regions; i++)
            if (m->mem_regions[i].mem_class == DRM_XE_MEM_REGION_CLASS_VRAM) used += (long long)m->mem_regions[i].used;
    }
    free(m); return used;
#else
    return -1;
#endif
}
static long long last_res, last_card;
static void state(const char *what) {
    long long r = resident_vram0(), c = card_used();
    printf("   [%s] resident-vram0 %lld MiB (%+lld)", what, r >> 20, (r - last_res) / (long long)MIB);
    if (c >= 0) printf(", card used %lld MiB (%+lld)", c >> 20, (c - last_card) / (long long)MIB);
    else printf(", card used n/a");
    printf("\n"); fflush(stdout);
}
static void mark(void) { last_res = resident_vram0(); last_card = card_used(); }
static long long dres(void) { return resident_vram0() - last_res; }

/* ---- OpenCL side ---- */
static const char *SRC =
"static uint pat(ulong i, uint seed) { uint x = (uint)i * 2654435761u ^ seed; x ^= x >> 15; x *= 2246822519u; return x ^ (x >> 13); }\n"
"kernel void fill(global uint *p, ulong off, uint seed, ulong n) {\n"
"  ulong g = get_global_id(0); if (g < n) p[off + g] = pat(g, seed); }\n"
"kernel void count_bad(global uint *p, ulong off, uint seed, ulong n, global uint *bad) {\n"
"  ulong g = get_global_id(0); if (g < n && p[off + g] != pat(g, seed)) atomic_inc(bad); }\n";

static cl_context cx; static cl_command_queue q; static cl_kernel kfill, kbad; static cl_mem badbuf;

static void run_fill(cl_mem b, cl_ulong off, cl_uint seed, cl_ulong n) {
    size_t gs = (size_t)n; cl_int e;
    CLC(clSetKernelArg(kfill, 0, sizeof b, &b), "arg0"); CLC(clSetKernelArg(kfill, 1, 8, &off), "arg1");
    CLC(clSetKernelArg(kfill, 2, 4, &seed), "arg2"); CLC(clSetKernelArg(kfill, 3, 8, &n), "arg3");
    e = clEnqueueNDRangeKernel(q, kfill, 1, NULL, &gs, NULL, 0, NULL, NULL); CLC(e, "fill enqueue");
    CLC(clFinish(q), "fill finish");
}
static unsigned count_bad(cl_mem b, cl_ulong off, cl_uint seed, cl_ulong n) {
    size_t gs = (size_t)n; cl_uint z = 0, bad = 0;
    CLC(clEnqueueWriteBuffer(q, badbuf, CL_TRUE, 0, 4, &z, 0, NULL, NULL), "bad clear");
    CLC(clSetKernelArg(kbad, 0, sizeof b, &b), "arg0"); CLC(clSetKernelArg(kbad, 1, 8, &off), "arg1");
    CLC(clSetKernelArg(kbad, 2, 4, &seed), "arg2"); CLC(clSetKernelArg(kbad, 3, 8, &n), "arg3");
    CLC(clSetKernelArg(kbad, 4, sizeof badbuf, &badbuf), "arg4");
    CLC(clEnqueueNDRangeKernel(q, kbad, 1, NULL, &gs, NULL, 0, NULL, NULL), "count enqueue");
    CLC(clEnqueueReadBuffer(q, badbuf, CL_TRUE, 0, 4, &bad, 0, NULL, NULL), "bad read");
    return bad;
}
static cl_mem sub(cl_mem whole, size_t origin, size_t size) {
    cl_buffer_region r = { origin, size }; cl_int e;
    cl_mem s = clCreateSubBuffer(whole, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &e);
    CLC(e, "clCreateSubBuffer"); return s;
}
static void expect_bad(unsigned bad, unsigned want, const char *what) {
    printf("   %s: bad = %u (want %u)\n", what, bad, want);
    if (bad != want) fail(what, "mismatch count wrong");
}

/* ---- Level Zero side ---- */
static ze_context_handle_t zc; static ze_device_handle_t zd; static size_t chunk;
static ze_physical_mem_handle_t phys_create(void) {
    ze_physical_mem_desc_t pd = { ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, NULL, 0, chunk };
    ze_physical_mem_handle_t h; ZE(zePhysicalMemCreate(zc, zd, &pd, &h)); return h;
}
static double map_chunk(char *base, size_t slot, ze_physical_mem_handle_t h) {
    double t = now_ms(); void *at = base + slot * chunk;
    ZE(zeVirtualMemMap(zc, at, chunk, h, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));
    ZE(zeVirtualMemSetAccessAttribute(zc, at, chunk, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));
    return now_ms() - t;
}
static double unmap_chunk(char *base, size_t slot) {
    double t = now_ms(); ZE(zeVirtualMemUnmap(zc, base + slot * chunk, chunk)); return now_ms() - t;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <pci-bdf, e.g. 0000:0f:00.0> [chunk_mib [chunks]]\n", argv[0]); return 2; }
    unsigned dom, bus, dev, fn;
    if (sscanf(argv[1], "%x:%x:%x.%x", &dom, &bus, &dev, &fn) != 4) { fprintf(stderr, "bad PCI address\n"); return 2; }
    chunk = (argc > 2 ? strtoull(argv[2], 0, 0) : 512) * MIB;
    int nch = argc > 3 ? atoi(argv[3]) : 8;
    if (nch < 5 || chunk < 8 * MIB) { fprintf(stderr, "need >= 5 chunks of >= 8 MiB\n"); return 2; }
    size_t slots = 4 * (size_t)nch, total = slots * chunk, lend_slot = slots / 2; /* default 32 slots = 16 GiB */

    /* ---- 1: device by PCI id, LEO context ---- */
    step("device by PCI id; OpenCL context over the Level Zero context (LEO)");
    ZE(zeInit(0));
    uint32_t nd = 0; ZE(zeDriverGet(&nd, NULL)); if (!nd) fail("no Level Zero driver", NULL);
    ze_driver_handle_t *drv = calloc(nd, sizeof *drv); ZE(zeDriverGet(&nd, drv));
    ze_driver_handle_t zdrv = NULL; zd = NULL;
    for (uint32_t i = 0; i < nd && !zd; i++) {
        uint32_t n = 0; if (zeDeviceGet(drv[i], &n, NULL) || !n) continue;
        ze_device_handle_t *ds = calloc(n, sizeof *ds); zeDeviceGet(drv[i], &n, ds);
        for (uint32_t j = 0; j < n; j++) {
            ze_pci_ext_properties_t pci = { ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES };
            if (zeDevicePciGetPropertiesExt(ds[j], &pci) != ZE_RESULT_SUCCESS) continue;
            if (pci.address.domain == dom && pci.address.bus == bus && pci.address.device == dev && pci.address.function == fn)
                { zd = ds[j]; zdrv = drv[i]; break; }
        }
        free(ds);
    }
    if (!zd) fail("no Level Zero device at that PCI address", argv[1]);
    ze_context_desc_t cd = { ZE_STRUCTURE_TYPE_CONTEXT_DESC };
    ZE(zeContextCreate(zdrv, &cd, &zc));

    cl_device_id cdev = NULL; cl_platform_id cplat = NULL;
    cl_uint np = 0; clGetPlatformIDs(0, NULL, &np);
    cl_platform_id *pl = calloc(np ? np : 1, sizeof *pl); clGetPlatformIDs(np, pl, NULL);
    for (cl_uint i = 0; i < np && !cdev; i++) {
        cl_uint n = 0; if (clGetDeviceIDs(pl[i], CL_DEVICE_TYPE_GPU, 0, NULL, &n) || !n) continue;
        cl_device_id *ds = calloc(n, sizeof *ds); clGetDeviceIDs(pl[i], CL_DEVICE_TYPE_GPU, n, ds, NULL);
        for (cl_uint j = 0; j < n; j++) {
            char ver[256] = ""; void *h = NULL;
            clGetDeviceInfo(ds[j], CL_DEVICE_VERSION, sizeof ver, ver, NULL);
            if (clGetDeviceInfo(ds[j], K_L0_DEVICE, sizeof h, &h, NULL) != CL_SUCCESS) continue; /* native NEO: unknown key */
            printf("   CL device '%s' L0 handle %p (ze device %p)\n", ver, h, (void *)zd);
            if (h == (void *)zd) {
                cdev = ds[j]; cplat = pl[i];
                if (!strstr(ver, "LEO")) printf("   note: CL_DEVICE_VERSION lacks 'LEO'\n");
                break;
            }
        }
        free(ds);
    }
    if (!cdev) fail("no OpenCL device maps to the ze device (CL_L0_DEVICE_HANDLE); is EnableLEO=1 with NEOReadDebugKeys=1 set?", NULL);
    cl_int err;
    cl_context_properties cp[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)cplat, K_L0_CONTEXT, (cl_context_properties)zc, 0 };
    cx = clCreateContext(cp, 1, &cdev, NULL, NULL, &err); CLC(err, "clCreateContext over ze_context");
    q = clCreateCommandQueueWithProperties(cx, cdev, NULL, &err); CLC(err, "queue");
    cl_program prog = clCreateProgramWithSource(cx, 1, &SRC, NULL, &err); CLC(err, "program");
    if (clBuildProgram(prog, 1, &cdev, "", NULL, NULL) != CL_SUCCESS) {
        char log[4096] = ""; clGetProgramBuildInfo(prog, cdev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL); fail("build", log);
    }
    kfill = clCreateKernel(prog, "fill", &err); CLC(err, "kernel fill");
    kbad = clCreateKernel(prog, "count_bad", &err); CLC(err, "kernel count_bad");
    badbuf = clCreateBuffer(cx, CL_MEM_READ_WRITE, 4, NULL, &err); CLC(err, "badbuf");
#ifdef HAVE_XE
    { char p[128]; snprintf(p, sizeof p, "/dev/dri/by-path/pci-%04x:%02x:%02x.%x-render", dom, bus, dev, fn); card_fd = open(p, O_RDWR); }
#endif
    printf("   OK; card-wide used %s\n", card_used() >= 0 ? "available" : "not available (no xe_drm.h or query failed)");
    mark(); state("after context");

    /* ---- 2: page size, reserve ---- */
    step("page size and reservation");
    size_t ps = 0; ZE(zeVirtualMemQueryPageSize(zc, zd, chunk, &ps));
    printf("   page size for a %zu MiB chunk: %zu KiB (expect 2048)\n", chunk >> 20, ps >> 10);
    if (!ps || chunk % ps) fail("chunk size is not a multiple of the page size", NULL);
    void *vbase = NULL; ZE(zeVirtualMemReserve(zc, NULL, total, &vbase));
    char *base = vbase; printf("   reserved %zu MiB at %p\n", total >> 20, vbase);
    state("after reserve (expect 0)");

    /* ---- 3: physical chunks ---- */
    step("create physical chunks");
    ze_physical_mem_handle_t *ph = calloc(nch, sizeof *ph); mark();
    for (int i = 0; i < nch; i++) ph[i] = phys_create();
    state("after create (expect ~0 under deferred backing)");

    /* ---- 4: map 0-3, cl_mem over the range, write/verify, red case ---- */
    step("map chunks 0-3; cl_mem over the whole range; fill and verify across a chunk boundary");
    mark(); double tm = 0;
    for (int i = 0; i < 4; i++) tm += map_chunk(base, i, ph[i]);
    printf("   map: %.2f ms per chunk\n", tm / 4); state("after map of 4 chunks");
    long long want = 4ll * chunk, got = dres();
    if (got < want - 2ll * MIB || got > want + 2ll * MIB) printf("   WARN: resident delta %lld MiB, expected %lld MiB\n", got >> 20, want >> 20);
    cl_mem_properties mp[] = { K_L0_MEM_OBJ, (cl_mem_properties)(uintptr_t)vbase, 0 };
    cl_mem whole = clCreateBufferWithProperties(cx, mp, CL_MEM_READ_WRITE, total, NULL, &err);
    CLC(err, "clCreateBufferWithProperties(L0 mem obj) over the range");
    cl_mem s3 = sub(whole, 3 * chunk, chunk);
    cl_ulong win = 2 * MIB / 4 * 2, woff = (2 * chunk - 4 * MIB) / 4; /* 8 MiB straddling the chunk 1/2 boundary (word units) */
    run_fill(whole, woff, 100, win);
    expect_bad(count_bad(whole, woff, 100, win), 0, "boundary window, right seed");
    expect_bad(count_bad(whole, woff, 101, win), (unsigned)win, "RED CASE: wrong seed must fail every word");
    cl_ulong cw = chunk / 4;
    run_fill(s3, 0, 3, cw);
    expect_bad(count_bad(s3, 0, 3, cw), 0, "chunk 3 full");

    /* ---- 5: lend: unmap chunk 3, map it at another address ---- */
    step("lend: move chunk 3 to another address");
    CLC(clFinish(q), "finish"); CLC(clReleaseMemObject(s3), "release sub"); mark();
    double tu = unmap_chunk(base, 3), tmp = map_chunk(base, lend_slot, ph[3]);
    printf("   unmap %.2f ms, map %.2f ms; chunk 3 now at slot %zu (%p)\n", tu, tmp, lend_slot, (void *)(base + lend_slot * chunk));
    state("after lend (expect unchanged within 2 MiB)");
    if (llabs(dres()) > 2ll * MIB) fail("resident changed by more than 2 MiB on a move", NULL);
    cl_mem s3b = sub(whole, lend_slot * chunk, chunk); /* the hole at slot 3 is never touched */
    expect_bad(count_bad(s3b, 0, 3, cw), 0, "moved chunk 3 at new address, old seed (same pages moved)");

    /* ---- 6: shrink and grow, then lend cycles ---- */
    step("shrink: unmap + destroy chunk 2; grow: create + map a new chunk there");
    mark(); unmap_chunk(base, 2); ZE(zePhysicalMemDestroy(zc, ph[2])); ph[2] = NULL;
    state("after destroy (expect -one chunk)");
    if (dres() > -(long long)chunk + 2ll * MIB || dres() < -(long long)chunk - 2ll * MIB) fail("resident did not drop by one chunk", NULL);
    mark(); ph[2] = phys_create(); map_chunk(base, 2, ph[2]);
    state("after create+map (expect +one chunk)");
    cl_mem s2 = sub(whole, 2 * chunk, chunk);
    run_fill(s2, 0, 22, cw); expect_bad(count_bad(s2, 0, 22, cw), 0, "new chunk 2");
    CLC(clReleaseMemObject(s2), "release s2");

    printf("   100 lend cycles of chunk 3 (unmap + map + clFinish), alternating slots %zu and 3\n", lend_slot);
    size_t cur = lend_slot; mark(); double t0 = now_ms();
    for (int i = 0; i < 100; i++) {
        size_t nxt = (cur == lend_slot) ? 3 : lend_slot;
        unmap_chunk(base, cur); map_chunk(base, nxt, ph[3]); CLC(clFinish(q), "finish"); cur = nxt;
    }
    printf("   %.3f ms per cycle\n", (now_ms() - t0) / 100); state("after 100 cycles");
    CLC(clReleaseMemObject(s3b), "release s3b");
    cl_mem s3c = sub(whole, cur * chunk, chunk);
    expect_bad(count_bad(s3c, 0, 3, cw), 0, "chunk 3 after 100 moves");

    /* ---- teardown ---- */
    step("teardown");
    CLC(clFinish(q), "finish");
    CLC(clReleaseMemObject(s3c), "rel"); CLC(clReleaseMemObject(whole), "rel whole");
    unmap_chunk(base, cur);
    unmap_chunk(base, 0); unmap_chunk(base, 1); unmap_chunk(base, 2);
    for (int i = 0; i < nch; i++) ZE(zePhysicalMemDestroy(zc, ph[i]));
    ZE(zeVirtualMemFree(zc, vbase, total)); state("after teardown");
    clReleaseMemObject(badbuf); clReleaseKernel(kfill); clReleaseKernel(kbad); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(cx); zeContextDestroy(zc);
    printf("\nPASS: all checks passed\n");
    return 0;
}
