/* arcwell_bo_dma_proof.c -- LISBON D2/D3 byte-destination proof, smallest scale.
 *
 * Campaign: nvme-direct-expert-tier (0.5.3 LISBON), design note
 * docs/design-nvme-direct-expert-tier.md D2/D3. The fill's destination must be
 * a dma-buf exported from an xe VRAM BO meeting arcwell's mapping contract
 * (NEEDS_VISIBLE_VRAM, CPU_CACHING_WC, size a multiple of 64 KiB, page-aligned
 * offsets and lengths). This tool is the NON-arcint end-to-end proof of that
 * path at the smallest scale:
 *
 *   xe VRAM BO -> DRM_IOCTL_PRIME_HANDLE_TO_FD -> AW_IOC_MAP_BUFFER
 *   -> one real 2,457,600 B expert payload from an ext4 store
 *   -> host readback of the BO -> AW_IOC_STATS delta.
 *
 * It answers a concrete question the design note left open: arcwell provides
 * NO allocator/helper. `stub/src/arcwell.c` states the caller does it --
 * "userspace creates a host-visible VRAM BO on xe and exports it as a dma-buf,
 * then hands us the fd" -- and docs/USING_ARCWELL.md section 3 is the caller's
 * lifecycle. So this tool drives the raw xe DRM ioctls itself, exactly as
 * arcwell's own stub/test/aw_expert_test.c does, and only includes the arcwell
 * uAPI header for the ioctl contract.
 *
 * RED-FIRST. Every --mutate-* leg MUST report RESULT=FAIL and exit non-zero;
 * the normal leg must report RESULT=PASS. The mutations are the losing
 * configurations this path must not silently accept:
 *   --mutate-no-part-offset   drop the ext4 partition start -> payload mismatch
 *   --mutate-offset-unaligned in_dest_offset=512 (not page-aligned) -> refused
 *   --mutate-bo-size          BO size not a 64 KiB multiple -> GEM_CREATE refused
 *   --mutate-system-bo        a system-memory BO (would be a host bounce) ->
 *                             MAP_BUFFER must refuse / clear REQUIRE_P2P
 *   --mutate-readback         corrupt the expected bytes -> memcmp mismatch
 *
 * Build (system drm header + the arcwell uAPI; no libdrm needed):
 *   gcc -O2 -Wall -I/usr/include/drm -I$ARCWELL/stub/include \
 *       -o arcwell_bo_dma_proof arcwell_bo_dma_proof.c
 * where $ARCWELL is ~/src/arcwell (an established tracked convention).
 *
 * Usage:
 *   arcwell_bo_dma_proof --drm <render-node> --file <expert.bin> \
 *       --part-start <sectors> [--arcwell /dev/arcwell] \
 *       [--readback-out <path>] [--mutate-...]
 *
 * No defaults name a host, a device node number, or a path: the render node is
 * identified by PCI id by the caller (docs/sop-card-window.md section 2).
 */
#define _GNU_SOURCE
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
#include "aw_uapi.h"

#define SLOT_64K 65536u

static const char *opt_drm = NULL;
static const char *opt_file = NULL;
static const char *opt_arcwell = "/dev/arcwell";
static const char *opt_readback = NULL;
static unsigned long long opt_part_start = 0;
static int opt_no_part_offset = 0;
static int opt_offset_unaligned = 0;
static int opt_bo_size = 0;
static int opt_system_bo = 0;
static int opt_readback_mutate = 0;

static int fail(const char *what)
{
	fprintf(stderr, "RESULT=FAIL -- %s: %s\n", what, strerror(errno));
	return 1;
}

/* One plain extent, absolute LBA = fe_physical/512 + partition start. The same
 * once-at-open translation docs/USING_ARCWELL.md section 5 requires. */
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
	/* The mutation lives here: with --mutate-no-part-offset the relative
	 * physical block is used unshifted, which reads the wrong bytes. */
	*lba = ex->fe_physical / 512 + (opt_no_part_offset ? 0 : opt_part_start);
	*len = (uint64_t)st.st_size;
	printf("FIEMAP: %s extents=1 phys=%llu -> absolute LBA=%llu len=%llu%s\n",
	       path, (unsigned long long)ex->fe_physical, (unsigned long long)*lba,
	       (unsigned long long)*len, opt_no_part_offset ? "  [MUTATED: partition start dropped]" : "");
	return 0;
}

static unsigned char *read_file(const char *path, uint64_t want, int *ok)
{
	int fd = open(path, O_RDONLY);
	*ok = 0;
	if (fd < 0) return NULL;
	unsigned char *buf = malloc(want);
	if (!buf) { close(fd); return NULL; }
	if (read(fd, buf, want) != (ssize_t)want) { free(buf); close(fd); return NULL; }
	close(fd);
	*ok = 1;
	return buf;
}

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--drm") && i + 1 < argc) opt_drm = argv[++i];
		else if (!strcmp(argv[i], "--file") && i + 1 < argc) opt_file = argv[++i];
		else if (!strcmp(argv[i], "--arcwell") && i + 1 < argc) opt_arcwell = argv[++i];
		else if (!strcmp(argv[i], "--readback-out") && i + 1 < argc) opt_readback = argv[++i];
		else if (!strcmp(argv[i], "--part-start") && i + 1 < argc)
			opt_part_start = strtoull(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--mutate-no-part-offset")) opt_no_part_offset = 1;
		else if (!strcmp(argv[i], "--mutate-offset-unaligned")) opt_offset_unaligned = 1;
		else if (!strcmp(argv[i], "--mutate-bo-size")) opt_bo_size = 1;
		else if (!strcmp(argv[i], "--mutate-system-bo")) opt_system_bo = 1;
		else if (!strcmp(argv[i], "--mutate-readback")) opt_readback_mutate = 1;
		else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
	}
	int mutated = opt_no_part_offset || opt_offset_unaligned || opt_bo_size ||
		      opt_system_bo || opt_readback_mutate;

	if (!opt_drm || !opt_file) {
		fprintf(stderr, "usage: %s --drm <node> --file <expert.bin> --part-start <sectors> [--mutate-...]\n",
			argv[0]);
		return 2;
	}
	printf("==== arcwell BO -> dma-buf -> arcwell -> host readback %s ====\n",
	       mutated ? "[MUTATED - MUST FAIL]" : "[normal - must PASS]");
	printf("drm=%s arcwell=%s part_start=%llu\n", opt_drm, opt_arcwell, opt_part_start);

	struct stat fst;
	if (stat(opt_file, &fst) < 0) return fail("stat expert file");
	uint64_t flen = (uint64_t)fst.st_size;

	/* 1. xe VRAM BO, contract shape: VRAM placement, NEEDS_VISIBLE_VRAM, WC,
	 *    size rounded up to a 64 KiB multiple (KERNEL_FACTS.md "the target
	 *    buffer exists"). --mutate-system-bo uses system memory instead; the
	 *    map must then be refused rather than silently bounced. */
	int gfd = open(opt_drm, O_RDWR | O_CLOEXEC);
	if (gfd < 0) return fail("open render node");

	struct drm_xe_gem_create bo;
	memset(&bo, 0, sizeof bo);
	bo.size = opt_bo_size ? flen : ((flen + (SLOT_64K - 1)) & ~((uint64_t)SLOT_64K - 1));
	bo.placement = opt_system_bo ? (1u << DRM_XE_MEM_REGION_CLASS_SYSMEM)
				     : (1u << DRM_XE_MEM_REGION_CLASS_VRAM);
	bo.flags = opt_system_bo ? 0 : DRM_XE_GEM_CREATE_FLAG_NEEDS_VISIBLE_VRAM;
	bo.cpu_caching = DRM_XE_GEM_CPU_CACHING_WC;

	if (ioctl(gfd, DRM_IOCTL_XE_GEM_CREATE, &bo) < 0) {
		if (opt_bo_size) {
			printf("GEM_CREATE refused a non-64KiB size (%llu B): %s\n",
			       (unsigned long long)flen, strerror(errno));
			fprintf(stderr, "RESULT=FAIL -- [MUTATED] non-64KiB BO size rejected\n");
			return 1;
		}
		return fail("DRM_IOCTL_XE_GEM_CREATE");
	}
	if (opt_bo_size)
		printf("NOTE: [MUTATED] GEM_CREATE ACCEPTED a non-64KiB size (%llu B); "
		       "continuing to see whether the contract is enforced later\n",
		       (unsigned long long)flen);
	printf("BO: size=%llu B (%.2f x 64 KiB) handle=%u placement=%#x flags=%#x\n",
	       (unsigned long long)bo.size, (double)bo.size / SLOT_64K, bo.handle,
	       bo.placement, bo.flags);

	/* 2. map the BO so the CPU can poison and read it back. This is the
	 *    readback path the BO gate proved ("verified through its xe mapping"). */
	struct drm_xe_gem_mmap_offset mm;
	memset(&mm, 0, sizeof mm);
	mm.handle = bo.handle;
	if (ioctl(gfd, DRM_IOCTL_XE_GEM_MMAP_OFFSET, &mm) < 0) return fail("DRM_IOCTL_XE_GEM_MMAP_OFFSET");
	void *vram = mmap(NULL, bo.size, PROT_READ | PROT_WRITE, MAP_SHARED, gfd, (off_t)mm.offset);
	if (vram == MAP_FAILED) return fail("mmap BO");
	memset(vram, 0xA5, bo.size);   /* poison: a no-op DMA must show */

	/* 3. export the dma-buf fd. */
	struct drm_prime_handle pr;
	memset(&pr, 0, sizeof pr);
	pr.handle = bo.handle;
	pr.flags = O_RDWR | O_CLOEXEC;
	if (ioctl(gfd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &pr) < 0) {
		if (opt_system_bo)
			return fail("[MUTATED] PRIME export of a system BO");
		return fail("DRM_IOCTL_PRIME_HANDLE_TO_FD");
	}
	printf("PRIME: dma-buf fd=%d\n", pr.fd);

	/* 4. register with arcwell; REQUIRE_P2P is the contract, not optional. */
	int afd = open(opt_arcwell, O_RDWR);
	if (afd < 0) return fail("open /dev/arcwell");
	struct aw_ioc_stats base;
	memset(&base, 0, sizeof base);
	if (ioctl(afd, AW_IOC_STATS, &base) < 0) return fail("AW_IOC_STATS baseline");

	struct aw_ioc_map_buffer mb;
	memset(&mb, 0, sizeof mb);
	mb.in_handle = (uint64_t)pr.fd;
	mb.in_source = AW_BUF_DMABUF;
	mb.in_length = (uint32_t)bo.size;
	if (ioctl(afd, AW_IOC_MAP_BUFFER, &mb) < 0) {
		if (opt_system_bo) {
			int saved = errno;   /* the STATS read below must not clobber it */
			struct aw_ioc_stats after;
			memset(&after, 0, sizeof after);
			ioctl(afd, AW_IOC_STATS, &after);
			printf("MAP_BUFFER refused a system-memory BO: %s\n", strerror(saved));
			printf("STATS after refusal: via_host_bounce %u->%u (delta %u)\n",
			       base.via_host_bounce, after.via_host_bounce,
			       after.via_host_bounce - base.via_host_bounce);
			fprintf(stderr, "RESULT=FAIL -- [MUTATED] host-bounce configuration refused "
					"(no path bounces; via_host_bounce counts the refusal)\n");
			return 1;
		}
		return fail("AW_IOC_MAP_BUFFER");
	}
	if (opt_system_bo) {
		fprintf(stderr, "RESULT=FAIL -- [MUTATED] MAP_BUFFER ACCEPTED a system-memory BO; "
				"a host bounce would have been reported as success\n");
		return 1;
	}
	if (!(mb.out_flags & AW_MAP_F_REQUIRE_P2P)) {
		fprintf(stderr, "RESULT=FAIL -- MAP_BUFFER returned without AW_MAP_F_REQUIRE_P2P "
				"(the mapping is not peer-to-peer)\n");
		return 1;
	}
	printf("MAP_BUFFER: handle=%u out_flags=%#x (AW_MAP_F_REQUIRE_P2P honoured)\n",
	       mb.out_handle, mb.out_flags);

	/* 5. one real expert: FIEMAP -> LBA, one request at in_dest_offset=0. */
	uint64_t lba = 0, elen = 0;
	if (file_lba(opt_file, &lba, &elen)) return 1;
	if (elen != flen) { fprintf(stderr, "RESULT=FAIL -- size changed under us\n"); return 1; }

	struct aw_ioc_read_blocks req;
	memset(&req, 0, sizeof req);
	req.in_buffer_handle = mb.out_handle;
	req.in_start_block = lba;
	req.in_block_count = elen >> 9;              /* 4800 logical blocks */
	req.in_dest_offset = opt_offset_unaligned ? 512 : 0;

	struct aw_ioc_batch_submit sb;
	memset(&sb, 0, sizeof sb);
	sb.in_requests = (uint64_t)(uintptr_t)&req;
	sb.in_count = 1;
	if (ioctl(afd, AW_IOC_SUBMIT_BATCH, &sb) < 0) {
		if (opt_offset_unaligned) {
			printf("SUBMIT_BATCH ioctl refused the unaligned in_dest_offset: %s\n", strerror(errno));
			fprintf(stderr, "RESULT=FAIL -- [MUTATED] unaligned transfer geometry refused "
					"(USING_ARCWELL.md section 6: offsets and lengths must be page-aligned)\n");
			return 1;
		}
		return fail("AW_IOC_SUBMIT_BATCH");
	}
	printf("SUBMIT: batch_id=%llu submitted=%u err=%d\n",
	       (unsigned long long)sb.out_batch_id, sb.out_submitted, sb.out_err);

	/* `AW_IOC_SUBMIT_BATCH` returns 0 once the batch is queued, and reports a
	 * submission-time request error in @out_err with @out_submitted short of
	 * @in_count (observed: an unaligned in_dest_offset gives submitted=0,
	 * err=-22). The transport must therefore check the counts, not the ioctl
	 * return alone -- patch 0048's submit() has to do the same. */
	if (sb.out_err != 0 || sb.out_submitted != sb.in_count) {
		if (opt_offset_unaligned) {
			printf("SUBMIT_BATCH refused the unaligned in_dest_offset=512: "
			       "submitted=%u err=%d (out_err is the submission-time error; the ioctl "
			       "return is not enough)\n", sb.out_submitted, sb.out_err);
			fprintf(stderr, "RESULT=FAIL -- [MUTATED] unaligned transfer geometry refused "
					"(USING_ARCWELL.md section 6: offsets and lengths must be page-aligned)\n");
			return 1;
		}
		fprintf(stderr, "RESULT=FAIL -- submission refused: submitted=%u of %u, err=%d\n",
			sb.out_submitted, sb.in_count, sb.out_err);
		return 1;
	}

	/* 6. prove it is in flight (poll EAGAIN), then collect. */
	struct aw_ioc_batch_wait bw;
	int polls = 0, saw_eagain = 0, collected = 0;
	for (int i = 0; i < 10000; i++) {
		memset(&bw, 0, sizeof bw);
		bw.in_batch_id = sb.out_batch_id;
		bw.in_timeout_us = 0;
		if (ioctl(afd, AW_IOC_BATCH_WAIT, &bw) == 0) { collected = 1; break; }
		if (errno != EAGAIN) return fail("AW_IOC_BATCH_WAIT poll");
		saw_eagain = 1;
		polls++;
		usleep(100);
	}
	if (saw_eagain && !collected) {
		memset(&bw, 0, sizeof bw);
		bw.in_batch_id = sb.out_batch_id;
		bw.in_timeout_us = UINT64_MAX;
		if (ioctl(afd, AW_IOC_BATCH_WAIT, &bw) < 0) return fail("AW_IOC_BATCH_WAIT block");
	}
	printf("poll: saw_eagain=%d polls=%d collected=%d\n", saw_eagain, polls, collected);
	printf("COLLECT: bytes=%llu completed=%u segments=%u err=%d\n",
	       (unsigned long long)bw.out_bytes, bw.out_completed, bw.out_segments, bw.out_err);
	if (bw.out_err || bw.out_completed != 1 || bw.out_bytes != elen) {
		fprintf(stderr, "RESULT=FAIL -- batch did not land one full payload\n");
		return 1;
	}

	/* 7. AW_IOC_STATS as a DELTA (module-global counter). */
	struct aw_ioc_stats stats;
	memset(&stats, 0, sizeof stats);
	if (ioctl(afd, AW_IOC_STATS, &stats) < 0) return fail("AW_IOC_STATS after");
	printf("STATS delta: via_host_bounce %u->%u max_inflight %u->%u batches %llu->%llu "
	       "batch_reads %llu->%llu segments %u->%u bytes %llu->%llu\n",
	       base.via_host_bounce, stats.via_host_bounce, base.max_inflight, stats.max_inflight,
	       (unsigned long long)base.batches, (unsigned long long)stats.batches,
	       (unsigned long long)base.batch_reads, (unsigned long long)stats.batch_reads,
	       base.segments, stats.segments, (unsigned long long)base.bytes,
	       (unsigned long long)stats.bytes);

	/* 8. host readback: the BO's own xe mapping, byte-compared to the file. */
	int ok = 0;
	unsigned char *want = read_file(opt_file, elen, &ok);
	if (!ok || !want) return fail("read expert file for comparison");
	if (opt_readback_mutate) want[elen / 2] ^= 0xFF;  /* mutation: corrupt expected */
	unsigned char *got = malloc(elen);
	if (!got) return fail("malloc readback");
	memcpy(got, vram, elen);

	if (opt_readback) {
		int wfd = open(opt_readback, O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (wfd < 0) return fail("open readback-out");
		if (write(wfd, got, elen) != (ssize_t)elen) { close(wfd); return fail("write readback-out"); }
		close(wfd);
		printf("READBACK: wrote %llu B to %s\n", (unsigned long long)elen, opt_readback);
	}

	int bad = -1;
	for (uint64_t i = 0; i < elen; i++)
		if (got[i] != want[i]) { bad = (int)i; break; }

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
	if (stats.via_host_bounce != base.via_host_bounce) {
		fprintf(stderr, "RESULT=FAIL -- via_host_bounce moved by %u\n",
			stats.via_host_bounce - base.via_host_bounce);
		return 1;
	}
	/* max_inflight is a module-global HIGH-WATER MARK, not a per-run delta (the
	 * inherited loaded module already read 261 from an earlier leg), so the
	 * design note's requirement is the absolute `> 1`: the queue depth was used.
	 * This leg's own work is proved by the batches/batch_reads/bytes deltas. */
	if (stats.max_inflight <= 1) {
		fprintf(stderr, "RESULT=FAIL -- max_inflight is %u; the queue depth was not used\n",
			stats.max_inflight);
		return 1;
	}
	if (stats.batches <= base.batches || stats.batch_reads <= base.batch_reads) {
		fprintf(stderr, "RESULT=FAIL -- this leg's batch did not register in the counters\n");
		return 1;
	}
	if (stats.bytes - base.bytes != elen) {
		fprintf(stderr, "RESULT=FAIL -- byte delta %llu != payload %llu\n",
			(unsigned long long)(stats.bytes - base.bytes), (unsigned long long)elen);
		return 1;
	}

	printf("READBACK: %llu B byte-identical to the store file\n", (unsigned long long)elen);
	printf("RESULT=PASS -- xe VRAM BO dma-buf registered peer-to-peer, one real expert landed "
	       "by controller DMA, host readback byte-identical, via_host_bounce delta 0, "
	       "max_inflight %u\n", stats.max_inflight);

	close(pr.fd);
	close(afd);
	close(gfd);
	return 0;
}
