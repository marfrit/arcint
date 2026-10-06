/* arcwell_cl_slot_proof.c -- LISBON D2/D3: the transport + OpenCL slot import,
 * proven end-to-end at the smallest scale on the B60.
 *
 * Campaign: nvme-direct-expert-tier (0.5.3 LISBON). This tool is the missing
 * half the byte-destination proof (tools/arcwell_bo_dma_proof.c) left open:
 * that proof registered ONE dma-buf and verified it by HOST readback through
 * the BO's own xe mapping. This one exercises the CONSUMER path the plugin
 * needs -- a per-tensor VRAM slot pool, the store's gate|up|down record
 * scattered into the three per-tensor regions, and the SAME dma-bufs imported
 * into OpenCL and read back by an OpenCL queue:
 *
 *   per-layer BLOCK (gate, up, down) -> one 64 KiB-rounded xe VRAM BO each
 *   -> DRM_IOCTL_PRIME_HANDLE_TO_FD -> AW_IOC_MAP_BUFFER (REQUIRE_P2P)
 *   -> one 2,457,600 B real store expert per slot, scattered as three
 *      page-aligned requests (gate@0, up@slot*T, down@2*slot*T)
 *   -> AW_IOC_SUBMIT_BATCH / AW_IOC_BATCH_WAIT
 *   -> clCreateBufferWithProperties(DMA_BUF) on every BO
 *   -> clEnqueueReadBuffer -> byte-compare against the store record.
 *
 * WHY PER-TENSOR REGIONS, NOT ONE CONCATENATED PER-EXPERT BO. The plugin's
 * device layout is three separate per-tensor memories; the kernel
 * (moe_3gemm_swiglu_mlp.cl) indexes each by `expert_id * expert_wei_size` with
 * a stride fixed to the per-tensor size. A single concatenated record would
 * need a kernel stride change; per-tensor regions keep the kernel untouched.
 * The store record is gate|up|down concatenated, so one expert becomes THREE
 * page-aligned requests (each 819,200 B = 200 pages), not one -- a dated
 * correction to the design note's "one expert = one request", which held only
 * for the synthetic store whose whole file mapped to one destination.
 *
 * RED-FIRST. Every --mutate-* leg MUST report RESULT=FAIL and exit non-zero:
 *   --mutate-no-part-offset   drop the ext4 partition start -> mismatch
 *   --mutate-offset-unaligned in_dest_offset not page-aligned -> refused
 *   --mutate-system-bo        a system-memory BO (host bounce) -> MAP refused
 *   --mutate-readback         corrupt the expected bytes -> mismatch
 *   --mutate-cl               corrupt the BO through OpenCL after DMA -> mismatch
 *
 * Build (system CL + drm headers + the arcwell uAPI):
 *   gcc -O2 -Wall -I/usr/include/drm -I<arcwell>/stub/include -o \
 *       arcwell_cl_slot_proof arcwell_cl_slot_proof.c -lOpenCL
 *
 * Usage:
 *   arcwell_cl_slot_proof --drm <render-node> --store <dir> \
 *       --part-start <sectors> --cl-name B60 [--n <experts>] \
 *       [--arcwell /dev/arcwell] [--readback-out <path>] [--mutate-...]
 *
 * No default names a host, a device node number, or a path; the render node is
 * identified by PCI id by the caller (docs/sop-card-window.md section 2) and
 * the OpenCL device by its CL_DEVICE_NAME.
 */
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <linux/fs.h>
#include <linux/fiemap.h>
#include <drm/xe_drm.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "aw_uapi.h"

#ifndef CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR
#define CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR 0x2067
#endif

#define SLOT_64K  65536u
#define N_TENSOR  3

static const char *opt_drm = NULL;
static const char *opt_store = NULL;
static const char *opt_arcwell = "/dev/arcwell";
static const char *opt_readback = NULL;
static const char *opt_cl_name = NULL;
static unsigned long long opt_part_start = 0;
static unsigned long long opt_tensor_bytes = 819200;
static unsigned opt_capacity = 4;
static unsigned opt_n = 2;
static int opt_no_part_offset = 0;
static int opt_offset_unaligned = 0;
static int opt_system_bo = 0;
static int opt_readback_mutate = 0;
static int opt_cl_mutate = 0;

static int fail(const char *what)
{
	fprintf(stderr, "RESULT=FAIL -- %s: %s\n", what, strerror(errno));
	return 1;
}

/* one plain extent -> absolute LBA (docs/USING_ARCWELL.md section 5). */
static int file_lba(const char *path, uint64_t *lba, uint64_t *len)
{
	struct stat st;
	int fd = open(path, O_RDONLY);
	if (fd < 0) return fail("open expert file");
	if (fstat(fd, &st) < 0) return fail("fstat");

	union {
		struct fiemap fm;
		char raw[sizeof(struct fiemap) + 8 * sizeof(struct fiemap_extent)];
	} u;
	memset(&u, 0, sizeof u);
	u.fm.fm_start = 0;
	u.fm.fm_length = (uint64_t)st.st_size;
	u.fm.fm_flags = FIEMAP_FLAG_SYNC;
	u.fm.fm_extent_count = 8;
	if (ioctl(fd, FS_IOC_FIEMAP, &u.fm) < 0) { close(fd); return fail("FS_IOC_FIEMAP"); }
	close(fd);

	if (u.fm.fm_mapped_extents != 1) {
		fprintf(stderr, "RESULT=FAIL -- %s has %u extents, the layout promises 1\n",
			path, u.fm.fm_mapped_extents);
		return 1;
	}
	struct fiemap_extent *ex = &u.fm.fm_extents[0];
	if (ex->fe_flags & (FIEMAP_EXTENT_UNKNOWN | FIEMAP_EXTENT_DELALLOC |
			    FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_INLINE |
			    FIEMAP_EXTENT_UNWRITTEN)) {
		fprintf(stderr, "RESULT=FAIL -- %s extent not plain (flags %#llx)\n",
			path, (unsigned long long)ex->fe_flags);
		return 1;
	}
	*lba = ex->fe_physical / 512 + (opt_no_part_offset ? 0 : opt_part_start);
	*len = (uint64_t)st.st_size;
	return 0;
}

struct cl_ctx {
	cl_context ctx;
	cl_command_queue q;
	cl_device_id dev;
	cl_platform_id plat;
	clEnqueueAcquireExternalMemObjectsKHR_fn acq;
	clEnqueueReleaseExternalMemObjectsKHR_fn rel;
	char name[256];
};

static int cl_init(struct cl_ctx *c)
{
	cl_platform_id plats[8]; cl_uint nplat = 0;
	clGetPlatformIDs(8, plats, &nplat);
	c->dev = NULL;
	for (cl_uint i = 0; i < nplat && !c->dev; i++) {
		cl_device_id d; cl_uint nd = 0;
		if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &d, &nd) || !nd) continue;
		char n[256] = {0};
		clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof n, n, NULL);
		if (opt_cl_name && strstr(n, opt_cl_name)) {
			c->dev = d; c->plat = plats[i];
			strncpy(c->name, n, sizeof c->name - 1);
		}
	}
	if (!c->dev) {
		fprintf(stderr, "RESULT=FAIL -- no OpenCL device matching '%s'\n",
			opt_cl_name ? opt_cl_name : "(null)");
		return 1;
	}
	cl_int err;
	c->ctx = clCreateContext(NULL, 1, &c->dev, NULL, NULL, &err);
	if (!c->ctx) return fail("clCreateContext");
	c->q = clCreateCommandQueueWithProperties(c->ctx, c->dev, NULL, &err);
	if (!c->q) return fail("clCreateCommandQueueWithProperties");
	c->acq = (clEnqueueAcquireExternalMemObjectsKHR_fn)
		clGetExtensionFunctionAddressForPlatform(c->plat, "clEnqueueAcquireExternalMemObjectsKHR");
	c->rel = (clEnqueueReleaseExternalMemObjectsKHR_fn)
		clGetExtensionFunctionAddressForPlatform(c->plat, "clEnqueueReleaseExternalMemObjectsKHR");
	printf("OpenCL device: %s (acquire/release %s)\n", c->name,
	       c->acq && c->rel ? "present" : "ABSENT");
	return 0;
}

static cl_mem cl_import(struct cl_ctx *c, int fd, size_t bytes)
{
	cl_mem_properties props[] = {
		(cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
		(cl_mem_properties)fd, 0
	};
	cl_int err;
	cl_mem m = clCreateBufferWithProperties(c->ctx, props, CL_MEM_READ_WRITE, bytes, NULL, &err);
	if (!m) { fprintf(stderr, "clCreateBufferWithProperties failed err=%d\n", err); return NULL; }
	if (c->acq && c->acq(c->q, 1, &m, 0, NULL, NULL) != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueAcquireExternalMemObjectsKHR failed\n");
		clReleaseMemObject(m);
		return NULL;
	}
	clFinish(c->q);
	return m;
}

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--drm") && i + 1 < argc) opt_drm = argv[++i];
		else if (!strcmp(argv[i], "--store") && i + 1 < argc) opt_store = argv[++i];
		else if (!strcmp(argv[i], "--arcwell") && i + 1 < argc) opt_arcwell = argv[++i];
		else if (!strcmp(argv[i], "--readback-out") && i + 1 < argc) opt_readback = argv[++i];
		else if (!strcmp(argv[i], "--cl-name") && i + 1 < argc) opt_cl_name = argv[++i];
		else if (!strcmp(argv[i], "--part-start") && i + 1 < argc)
			opt_part_start = strtoull(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--capacity") && i + 1 < argc)
			opt_capacity = (unsigned)strtoul(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--n") && i + 1 < argc)
			opt_n = (unsigned)strtoul(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--mutate-no-part-offset")) opt_no_part_offset = 1;
		else if (!strcmp(argv[i], "--mutate-offset-unaligned")) opt_offset_unaligned = 1;
		else if (!strcmp(argv[i], "--mutate-system-bo")) opt_system_bo = 1;
		else if (!strcmp(argv[i], "--mutate-readback")) opt_readback_mutate = 1;
		else if (!strcmp(argv[i], "--mutate-cl")) opt_cl_mutate = 1;
		else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
	}
	int mutated = opt_no_part_offset || opt_offset_unaligned || opt_system_bo ||
		      opt_readback_mutate || opt_cl_mutate;

	if (!opt_drm || !opt_store || !opt_cl_name) {
		fprintf(stderr, "usage: %s --drm <node> --store <dir> --cl-name <substr> "
			"--part-start <sectors> [--n N] [--capacity C] [--mutate-...]\n", argv[0]);
		return 2;
	}

	const size_t T = (size_t)opt_tensor_bytes;
	const size_t region = (size_t)opt_capacity * T;            /* bytes per per-tensor BO */
	const size_t bo_size = (region + (SLOT_64K - 1)) & ~((size_t)SLOT_64K - 1);

	printf("==== transport + OpenCL slot import %s ====\n",
	       mutated ? "[MUTATED - MUST FAIL]" : "[normal - must PASS]");
	printf("drm=%s store=%s n=%u capacity=%u tensor=%zu region=%zu bo=%zu\n",
	       opt_drm, opt_store, opt_n, opt_capacity, T, region, bo_size);
	if (opt_n > opt_capacity) { fprintf(stderr, "RESULT=FAIL -- n > capacity\n"); return 1; }

	int gfd = open(opt_drm, O_RDWR | O_CLOEXEC);
	if (gfd < 0) return fail("open render node");
	int afd = open(opt_arcwell, O_RDWR);
	if (afd < 0) return fail("open /dev/arcwell");

	struct aw_ioc_stats base; memset(&base, 0, sizeof base);
	if (ioctl(afd, AW_IOC_STATS, &base) < 0) return fail("AW_IOC_STATS baseline");

	/* 1. one BO per tensor, all registered. */
	struct bo {
		uint32_t gem;
		int prime_fd;
		uint32_t aw_handle;
		cl_mem cl;
		void *map;
	} bos[N_TENSOR];
	memset(bos, 0, sizeof bos);
	for (int t = 0; t < N_TENSOR; t++) {
		struct drm_xe_gem_create gc; memset(&gc, 0, sizeof gc);
		gc.size = bo_size;
		gc.placement = opt_system_bo ? (1u << DRM_XE_MEM_REGION_CLASS_SYSMEM)
					     : (1u << DRM_XE_MEM_REGION_CLASS_VRAM);
		gc.flags = opt_system_bo ? 0 : DRM_XE_GEM_CREATE_FLAG_NEEDS_VISIBLE_VRAM;
		gc.cpu_caching = DRM_XE_GEM_CPU_CACHING_WC;
		if (ioctl(gfd, DRM_IOCTL_XE_GEM_CREATE, &gc) < 0) return fail("GEM_CREATE");
		bos[t].gem = gc.handle;

		struct drm_xe_gem_mmap_offset mo; memset(&mo, 0, sizeof mo);
		mo.handle = gc.handle;
		if (ioctl(gfd, DRM_IOCTL_XE_GEM_MMAP_OFFSET, &mo) < 0) return fail("MMAP_OFFSET");
		bos[t].map = mmap(NULL, bo_size, PROT_READ | PROT_WRITE, MAP_SHARED, gfd, (off_t)mo.offset);
		if (bos[t].map == MAP_FAILED) return fail("mmap BO");
		memset(bos[t].map, 0xA5, bo_size);

		struct drm_prime_handle pr; memset(&pr, 0, sizeof pr);
		pr.handle = gc.handle;
		pr.flags = O_RDWR | O_CLOEXEC;
		if (ioctl(gfd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &pr) < 0) {
			if (opt_system_bo) return fail("[MUTATED] PRIME export of system BO");
			return fail("PRIME_HANDLE_TO_FD");
		}
		bos[t].prime_fd = pr.fd;

		struct aw_ioc_map_buffer mb; memset(&mb, 0, sizeof mb);
		mb.in_handle = (uint64_t)pr.fd;
		mb.in_source = AW_BUF_DMABUF;
		mb.in_length = (uint32_t)bo_size;
		if (ioctl(afd, AW_IOC_MAP_BUFFER, &mb) < 0) {
			if (opt_system_bo) {
				int saved = errno;
				struct aw_ioc_stats after; memset(&after, 0, sizeof after);
				ioctl(afd, AW_IOC_STATS, &after);
				printf("MAP_BUFFER refused system-memory BO: %s (via_host_bounce %u->%u)\n",
				       strerror(saved), base.via_host_bounce, after.via_host_bounce);
				fprintf(stderr, "RESULT=FAIL -- [MUTATED] host-bounce configuration refused\n");
				return 1;
			}
			return fail("AW_IOC_MAP_BUFFER");
		}
		if (opt_system_bo) {
			fprintf(stderr, "RESULT=FAIL -- [MUTATED] MAP_BUFFER accepted a system-memory BO\n");
			return 1;
		}
		if (!(mb.out_flags & AW_MAP_F_REQUIRE_P2P)) {
			fprintf(stderr, "RESULT=FAIL -- MAP_BUFFER lacked AW_MAP_F_REQUIRE_P2P\n");
			return 1;
		}
		bos[t].aw_handle = mb.out_handle;
		printf("BO[%d]: gem=%u prime_fd=%d aw_handle=%u size=%zu out_flags=%#x\n",
		       t, bos[t].gem, bos[t].prime_fd, bos[t].aw_handle, bo_size, mb.out_flags);
	}

	/* 2. build the batch: one expert (store ordinal == slot) -> three requests. */
	size_t nreq = (size_t)opt_n * N_TENSOR;
	struct aw_ioc_read_blocks *reqs = calloc(nreq, sizeof *reqs);
	if (!reqs) return fail("calloc reqs");

	for (unsigned s = 0; s < opt_n; s++) {
		char path[512];
		snprintf(path, sizeof path, "%s/expert_%04u.bin", opt_store, s);
		uint64_t lba = 0, len = 0;
		if (file_lba(path, &lba, &len)) return 1;
		if (len != N_TENSOR * (uint64_t)T) {
			fprintf(stderr, "RESULT=FAIL -- %s size %llu != record %llu\n",
				path, (unsigned long long)len,
				(unsigned long long)(N_TENSOR * (uint64_t)T));
			return 1;
		}
		for (int t = 0; t < N_TENSOR; t++) {
			struct aw_ioc_read_blocks *r = &reqs[s * N_TENSOR + t];
			r->in_buffer_handle = bos[t].aw_handle;
			r->in_start_block = lba + (uint64_t)t * (T / 512);
			r->in_block_count = T / 512;
			r->in_dest_offset = (uint64_t)s * T + (opt_offset_unaligned ? 512 : 0);
		}
	}

	struct aw_ioc_batch_submit sb; memset(&sb, 0, sizeof sb);
	sb.in_requests = (uint64_t)(uintptr_t)reqs;
	sb.in_count = (uint32_t)nreq;
	if (ioctl(afd, AW_IOC_SUBMIT_BATCH, &sb) < 0) {
		if (opt_offset_unaligned) {
			fprintf(stderr, "RESULT=FAIL -- [MUTATED] unaligned geometry refused by ioctl\n");
			return 1;
		}
		return fail("AW_IOC_SUBMIT_BATCH");
	}
	printf("SUBMIT: batch_id=%llu submitted=%u err=%d\n",
	       (unsigned long long)sb.out_batch_id, sb.out_submitted, sb.out_err);
	if (sb.out_err != 0 || sb.out_submitted != sb.in_count) {
		if (opt_offset_unaligned) {
			printf("SUBMIT refused unaligned in_dest_offset: submitted=%u err=%d\n",
			       sb.out_submitted, sb.out_err);
			fprintf(stderr, "RESULT=FAIL -- [MUTATED] unaligned transfer geometry refused\n");
			return 1;
		}
		fprintf(stderr, "RESULT=FAIL -- submission refused: %u of %u err=%d\n",
			sb.out_submitted, sb.in_count, sb.out_err);
		return 1;
	}

	struct aw_ioc_batch_wait bw; memset(&bw, 0, sizeof bw);
	bw.in_batch_id = sb.out_batch_id;
	bw.in_timeout_us = UINT64_MAX;
	if (ioctl(afd, AW_IOC_BATCH_WAIT, &bw) < 0) return fail("AW_IOC_BATCH_WAIT");
	printf("COLLECT: bytes=%llu completed=%u segments=%u err=%d (of %zu requests)\n",
	       (unsigned long long)bw.out_bytes, bw.out_completed, bw.out_segments,
	       bw.out_err, nreq);
	if (bw.out_err || bw.out_completed != (uint32_t)nreq ||
	    bw.out_bytes != (unsigned long long)(opt_n * N_TENSOR * T)) {
		fprintf(stderr, "RESULT=FAIL -- batch did not land every request\n");
		return 1;
	}

	/* 3. import every dma-buf into OpenCL and read it back. */
	static struct cl_ctx c;
	if (cl_init(&c)) return 1;
	for (int t = 0; t < N_TENSOR; t++) {
		bos[t].cl = cl_import(&c, bos[t].prime_fd, bo_size);
		if (!bos[t].cl) { fprintf(stderr, "RESULT=FAIL -- OpenCL import of BO[%d]\n", t); return 1; }
		printf("BO[%d] imported into OpenCL\n", t);
	}
	if (opt_cl_mutate) {
		/* corrupt the first byte of BO0 through the OpenCL object; a kernel/
		 * readback must then disagree. */
		unsigned char z = 0x00;
		clEnqueueWriteBuffer(c.q, bos[0].cl, CL_TRUE, 0, 1, &z, 0, NULL, NULL);
		printf("[MUTATED] wrote 0x00 over BO0 byte 0 through OpenCL\n");
	}

	/* expected: for each tensor, its slice of each expert record; got: the
	 * corresponding region of the imported BO read back through OpenCL. */
	size_t used = (size_t)opt_n * T;
	unsigned char *got = malloc((size_t)N_TENSOR * used);
	unsigned char *want = malloc((size_t)N_TENSOR * used);
	if (!got || !want) return fail("malloc");
	for (int t = 0; t < N_TENSOR; t++) {
		if (clEnqueueReadBuffer(c.q, bos[t].cl, CL_TRUE, 0, used,
					got + (size_t)t * used, 0, NULL, NULL) != CL_SUCCESS) {
			fprintf(stderr, "RESULT=FAIL -- clEnqueueReadBuffer BO[%d]\n", t);
			return 1;
		}
		for (unsigned s = 0; s < opt_n; s++) {
			char path[512];
			snprintf(path, sizeof path, "%s/expert_%04u.bin", opt_store, s);
			int fd = open(path, O_RDONLY);
			if (fd < 0) return fail("open expert for compare");
			uint64_t off = (uint64_t)t * T;
			if (pread(fd, want + (size_t)t * used + (size_t)s * T, T, off) != (ssize_t)T) {
				close(fd); return fail("pread expert");
			}
			close(fd);
		}
	}

	if (opt_readback_mutate) want[(size_t)used / 2] ^= 0xFF;

	int bad = -1;
	for (size_t i = 0; i < (size_t)N_TENSOR * used; i++)
		if (got[i] != want[i]) { bad = (int)i; break; }

	if (opt_readback) {
		int wfd = open(opt_readback, O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (wfd < 0) return fail("open readback-out");
		if (write(wfd, got, (size_t)N_TENSOR * used) != (ssize_t)((size_t)N_TENSOR * used)) {
			close(wfd); return fail("write readback-out");
		}
		close(wfd);
		printf("READBACK: wrote %zu B to %s\n", (size_t)N_TENSOR * used, opt_readback);
	}

	if (bad >= 0) {
		fprintf(stderr, "RESULT=FAIL -- readback mismatch at byte %d%s\n", bad,
			mutated ? " (expected for this mutation)" : "");
		return 1;
	}
	if (mutated) {
		fprintf(stderr, "RESULT=FAIL -- [MUTATED] the mutation did NOT change the outcome; "
				"this leg proves nothing\n");
		return 1;
	}

	/* 4. STATS delta on this leg's own work. */
	struct aw_ioc_stats st; memset(&st, 0, sizeof st);
	if (ioctl(afd, AW_IOC_STATS, &st) < 0) return fail("AW_IOC_STATS after");
	printf("STATS delta: via_host_bounce %u->%u max_inflight %u->%u batches %llu->%llu "
	       "batch_reads %llu->%llu segments %u->%u bytes %llu->%llu\n",
	       base.via_host_bounce, st.via_host_bounce, base.max_inflight, st.max_inflight,
	       (unsigned long long)base.batches, (unsigned long long)st.batches,
	       (unsigned long long)base.batch_reads, (unsigned long long)st.batch_reads,
	       base.segments, st.segments, (unsigned long long)base.bytes,
	       (unsigned long long)st.bytes);
	if (st.via_host_bounce != base.via_host_bounce) {
		fprintf(stderr, "RESULT=FAIL -- via_host_bounce moved\n"); return 1;
	}
	if (st.max_inflight <= 1) { fprintf(stderr, "RESULT=FAIL -- queue depth unused\n"); return 1; }
	if (st.batch_reads - base.batch_reads != nreq) {
		fprintf(stderr, "RESULT=FAIL -- batch_reads delta != requests\n"); return 1;
	}

	printf("READBACK: %zu B through OpenCL byte-identical to the store record\n",
	       (size_t)N_TENSOR * used);
	printf("RESULT=PASS -- %u real expert(s) DMA'd by controller into per-tensor VRAM BOs, "
	       "imported into OpenCL on %s, read back byte-identical, via_host_bounce delta 0, "
	       "max_inflight %u, %zu requests\n", opt_n, c.name, st.max_inflight, nreq);
	return 0;
}
