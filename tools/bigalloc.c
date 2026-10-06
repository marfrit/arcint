/*
 * bigalloc.c -- attribute a process's large host allocations to call stacks.
 *
 * The instrument behind docs/design-fit-levers.md: the full-depth all-resident
 * compile materialises ~2.9x the artifact's bytes in host memory, and which
 * code path owns those bytes is the question. An LD_PRELOAD shim interposes
 * malloc/calloc/realloc/free, the aligned allocators and anonymous mmap/munmap;
 * every allocation at or above BIGALLOC_MIN bytes (default 8 MiB) is recorded
 * with its call stack. It keeps, per distinct stack, the live bytes, the count
 * and the bytes ever allocated, and snapshots the per-stack live bytes each
 * time the process-wide live total sets a new peak (in BIGALLOC_STEP steps,
 * default 256 MiB), so the
 * dump says who held the memory AT the peak, not only who allocated most.
 *
 * Output: BIGALLOC_OUT (default bigalloc.<pid>.txt), rewritten every
 * BIGALLOC_PERIOD seconds (default 5) by a background thread and at exit, so a
 * run killed by a watchdog still leaves its last state. Frames are written as
 * `module+0xoffset`; symbolise offline with addr2line against unstripped
 * copies of the same builds (tools/bigalloc_report.py).
 *
 * Allocations below the threshold are not counted: this measures the large
 * buffers, which is where a multiple of the artifact's bytes has to live. The
 * `small` line in the dump reports how much sub-threshold traffic there was
 * only as a count, never as a claim about its size.
 *
 * Build: cc -O2 -fPIC -shared -o bigalloc.so tools/bigalloc.c -ldl -lpthread
 * Use:   LD_PRELOAD=./bigalloc.so BIGALLOC_OUT=/path/trace.txt [BIGALLOC_ONLY=prog] <cmd>
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h> /* program_invocation_short_name */
#include <malloc.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MAX_FRAMES 28
#define MAX_STACKS 8192
#define TABLE_BITS 18
#define TABLE_SIZE (1u << TABLE_BITS)

typedef struct {
    void*    frames[MAX_FRAMES];
    int      depth;
    uint64_t hash;
    uint64_t live;
    uint64_t live_at_peak;
    uint64_t total;
    uint64_t count;
    uint64_t max_single;
} stack_rec;

typedef struct {
    uintptr_t ptr; /* 0 = empty, 1 = tombstone */
    uint64_t  size;
    int       stack;
} alloc_rec;

static void* (*real_malloc)(size_t);
static void (*real_free)(void*);
static void* (*real_calloc)(size_t, size_t);
static void* (*real_realloc)(void*, size_t);
static int (*real_posix_memalign)(void**, size_t, size_t);
static void* (*real_aligned_alloc)(size_t, size_t);
static void* (*real_memalign)(size_t, size_t);
static void* (*real_mmap)(void*, size_t, int, int, int, off_t);
static int (*real_munmap)(void*, size_t);

static stack_rec       stacks[MAX_STACKS];
static int             n_stacks;
static alloc_rec       table[TABLE_SIZE];
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t        live_total, peak_total, next_snapshot, dropped;
static uint64_t        peak_step = 256ull << 20;
static atomic_ullong   small_count;
static size_t          min_bytes = 8u << 20;
static int             ready;
static __thread int    in_hook;
static char            out_path[512];
static double          t_start, t_peak;

/* dlsym itself may calloc before real_calloc is known. */
static char   boot_buf[1 << 16];
static size_t boot_used;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int is_boot(void* p) {
    return (char*)p >= boot_buf && (char*)p < boot_buf + sizeof boot_buf;
}

static uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

static int intern_stack(void** frames, int depth) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < depth; ++i) h = mix(h, (uint64_t)(uintptr_t)frames[i]);
    for (int i = 0; i < n_stacks; ++i)
        if (stacks[i].hash == h && stacks[i].depth == depth &&
            memcmp(stacks[i].frames, frames, sizeof(void*) * (size_t)depth) == 0)
            return i;
    if (n_stacks == MAX_STACKS) return -1;
    stack_rec* s = &stacks[n_stacks];
    memcpy(s->frames, frames, sizeof(void*) * (size_t)depth);
    s->depth = depth;
    s->hash  = h;
    return n_stacks++;
}

static uint32_t slot_of(uintptr_t p) {
    uint64_t x = (uint64_t)p * 0x9e3779b97f4a7c15ull;
    return (uint32_t)(x >> (64 - TABLE_BITS));
}

static void record(void* p, size_t size) {
    if (!p || !ready || size < min_bytes || in_hook) {
        if (p && ready && size < min_bytes) atomic_fetch_add(&small_count, 1);
        return;
    }
    in_hook = 1;
    void* frames[MAX_FRAMES + 2];
    int   depth = backtrace(frames, MAX_FRAMES + 2);
    /* drop record() and the hook itself */
    int skip = depth > 2 ? 2 : 0;
    pthread_mutex_lock(&lock);
    int sid = intern_stack(frames + skip, depth - skip);
    uint32_t i = slot_of((uintptr_t)p);
    int placed = 0;
    for (uint32_t n = 0; n < TABLE_SIZE; ++n, i = (i + 1) & (TABLE_SIZE - 1)) {
        if (table[i].ptr <= 1) {
            table[i].ptr   = (uintptr_t)p;
            table[i].size  = size;
            table[i].stack = sid;
            placed = 1;
            break;
        }
    }
    if (!placed || sid < 0) {
        ++dropped;
    } else {
        stack_rec* s = &stacks[sid];
        s->live += size;
        s->total += size;
        s->count += 1;
        if (size > s->max_single) s->max_single = size;
    }
    live_total += size;
    if (live_total > peak_total) {
        peak_total = live_total;
        if (peak_total >= next_snapshot) {
            for (int k = 0; k < n_stacks; ++k) stacks[k].live_at_peak = stacks[k].live;
            next_snapshot = peak_total + peak_step;
            t_peak        = now_s() - t_start;
        }
    }
    pthread_mutex_unlock(&lock);
    in_hook = 0;
}

static void forget(void* p) {
    if (!p || !ready) return;
    pthread_mutex_lock(&lock);
    uint32_t i = slot_of((uintptr_t)p);
    for (uint32_t n = 0; n < TABLE_SIZE; ++n, i = (i + 1) & (TABLE_SIZE - 1)) {
        if (table[i].ptr == 0) break;
        if (table[i].ptr == (uintptr_t)p) {
            if (table[i].stack >= 0) stacks[table[i].stack].live -= table[i].size;
            live_total -= table[i].size;
            table[i].ptr = 1;
            break;
        }
    }
    pthread_mutex_unlock(&lock);
}

static stack_rec snap[MAX_STACKS];

static void dump(void) {
    if (!out_path[0]) return;
    char tmp[600];
    snprintf(tmp, sizeof tmp, "%s.tmp", out_path);
    in_hook = 1;
    /* Copy under the lock, symbolise outside it: dladdr() takes the dynamic
     * loader's lock, and record() runs backtrace() (which can take it too)
     * before our lock -- holding ours across dladdr() would invert that order
     * against a thread allocating inside dlopen. */
    pthread_mutex_lock(&lock);
    const int      ns = n_stacks;
    const uint64_t live = live_total, peak = peak_total, drop = dropped;
    const double   tp = t_peak;
    memcpy(snap, stacks, sizeof(stack_rec) * (size_t)ns);
    pthread_mutex_unlock(&lock);
    FILE* f = fopen(tmp, "w");
    if (!f) {
        in_hook = 0;
        return;
    }
    fprintf(f, "# bigalloc pid=%d min_bytes=%zu t=%.1f live=%llu peak=%llu t_peak=%.1f stacks=%d dropped=%llu small=%llu\n",
            (int)getpid(), min_bytes, now_s() - t_start, (unsigned long long)live, (unsigned long long)peak, tp, ns,
            (unsigned long long)drop, (unsigned long long)atomic_load(&small_count));
    for (int k = 0; k < ns; ++k) {
        stack_rec* s = &snap[k];
        fprintf(f, "S %d live=%llu at_peak=%llu total=%llu count=%llu max=%llu\n", k,
                (unsigned long long)s->live, (unsigned long long)s->live_at_peak,
                (unsigned long long)s->total, (unsigned long long)s->count,
                (unsigned long long)s->max_single);
        for (int d = 0; d < s->depth; ++d) {
            Dl_info di;
            if (dladdr(s->frames[d], &di) && di.dli_fname)
                fprintf(f, "  F %s+0x%lx %s\n", di.dli_fname,
                        (unsigned long)((char*)s->frames[d] - (char*)di.dli_fbase),
                        di.dli_sname ? di.dli_sname : "-");
            else
                fprintf(f, "  F ?+%p\n", s->frames[d]);
        }
    }
    fclose(f);
    rename(tmp, out_path);
    in_hook = 0;
}

static void* dumper(void* arg) {
    unsigned period = (unsigned)(uintptr_t)arg;
    in_hook = 1; /* this thread's own allocations are not the process's */
    for (;;) {
        sleep(period);
        dump();
        in_hook = 1;
    }
    return NULL;
}

static void resolve(void) {
    if (real_malloc) return;
    real_malloc         = dlsym(RTLD_NEXT, "malloc");
    real_free           = dlsym(RTLD_NEXT, "free");
    real_calloc         = dlsym(RTLD_NEXT, "calloc");
    real_realloc        = dlsym(RTLD_NEXT, "realloc");
    real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    real_aligned_alloc  = dlsym(RTLD_NEXT, "aligned_alloc");
    real_memalign       = dlsym(RTLD_NEXT, "memalign");
    real_mmap           = dlsym(RTLD_NEXT, "mmap");
    real_munmap         = dlsym(RTLD_NEXT, "munmap");
}

__attribute__((constructor)) static void init(void) {
    resolve();
    /* BIGALLOC_ONLY=<name>: trace only the program with that short name, so a
     * driver that launches the traced process (and inherits LD_PRELOAD) does
     * not write the same dump. Images replaced by exec re-run this check. */
    const char* only = getenv("BIGALLOC_ONLY");
    if (only && strcmp(only, program_invocation_short_name) != 0) return;
    const char* m = getenv("BIGALLOC_MIN");
    if (m) min_bytes = (size_t)strtoull(m, NULL, 10);
    const char* st = getenv("BIGALLOC_STEP");
    if (st) peak_step = strtoull(st, NULL, 10);
    next_snapshot = peak_step;
    const char* o = getenv("BIGALLOC_OUT");
    if (o) snprintf(out_path, sizeof out_path, "%s", o);
    else snprintf(out_path, sizeof out_path, "bigalloc.%d.txt", (int)getpid());
    unsigned    period = 5;
    const char* per    = getenv("BIGALLOC_PERIOD");
    if (per) period = (unsigned)strtoul(per, NULL, 10);
    /* backtrace() loads libgcc_s on first use and that allocates: do it now. */
    void* warm[4];
    in_hook = 1;
    backtrace(warm, 4);
    in_hook = 0;
    t_start = now_s();
    pthread_t th;
    pthread_create(&th, NULL, dumper, (void*)(uintptr_t)(period ? period : 5));
    pthread_detach(th);
    ready = 1;
}

__attribute__((destructor)) static void fini(void) {
    dump();
}

void* malloc(size_t n) {
    if (!real_malloc) resolve();
    void* p = real_malloc(n);
    record(p, n);
    return p;
}

void* calloc(size_t a, size_t b) {
    if (!real_calloc) {
        /* dlsym bootstrap */
        size_t n = a * b;
        n = (n + 15) & ~(size_t)15;
        if (boot_used + n > sizeof boot_buf) return NULL;
        void* p = boot_buf + boot_used;
        boot_used += n;
        return p;
    }
    void* p = real_calloc(a, b);
    record(p, a * b);
    return p;
}

void free(void* p) {
    if (!p || is_boot(p)) return;
    /* Only a chunk at or above the threshold can be in the table; checking
     * the usable size first keeps the lock off the small-free hot path. */
    if (malloc_usable_size(p) >= min_bytes) forget(p);
    if (!real_free) resolve();
    real_free(p);
}

void* realloc(void* p, size_t n) {
    if (is_boot(p)) {
        /* a boot chunk's size is not kept: copy at most what the buffer holds */
        const size_t avail = (size_t)(boot_buf + sizeof boot_buf - (char*)p);
        void* q = malloc(n);
        if (q) memcpy(q, p, n < avail ? n : avail);
        return q;
    }
    if (!real_realloc) resolve();
    const size_t old = p ? malloc_usable_size(p) : 0;
    if (p && old >= min_bytes) forget(p);
    void* q = real_realloc(p, n);
    /* failed: p is still live. Re-recording keeps live/at_peak right; that
     * stack's total/count gain one phantom allocation per failure. */
    if (!q && n != 0 && p && old >= min_bytes) record(p, old);
    else record(q, n);
    return q;
}

int posix_memalign(void** out, size_t align, size_t n) {
    if (!real_posix_memalign) resolve();
    int rc = real_posix_memalign(out, align, n);
    if (rc == 0) record(*out, n);
    return rc;
}

void* aligned_alloc(size_t align, size_t n) {
    if (!real_aligned_alloc) resolve();
    void* p = real_aligned_alloc(align, n);
    record(p, n);
    return p;
}

void* memalign(size_t align, size_t n) {
    if (!real_memalign) resolve();
    void* p = real_memalign(align, n);
    record(p, n);
    return p;
}

void* mmap(void* addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) resolve();
    void* p = real_mmap(addr, len, prot, flags, fd, off);
    if (p != MAP_FAILED && (flags & MAP_ANONYMOUS) && !(flags & MAP_SHARED)) record(p, len);
    return p;
}

/* A partial munmap at a recorded base forgets the whole mapping, and tombstones
 * are not reclaimed: both bias toward UNDER-counting live bytes, never over. */
int munmap(void* addr, size_t len) {
    if (!real_munmap) resolve();
    if (len >= min_bytes) forget(addr);
    return real_munmap(addr, len);
}
