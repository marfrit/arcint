// The byte buffer a context checkpoint's state is saved into (src/exec/bytes.h):
// a llama state call writes every byte of it, so the allocation must write none.
// Red case: `using Bytes = std::vector<uint8_t>` in src/exec/bytes.h — its
// resize value-initialises the allocation, and the trait and the residency
// checks below both fail.
#include "exec/bytes.h"
#include "harness.h"

#include <sys/mman.h>   // mincore: whether the allocation touched its pages
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using namespace lgc;

namespace {

int constructed_default = 0;   // a construct with no value: default-initialised, no byte written
int constructed_value   = 0;   // a construct with a value: the value is written

// DefaultInitByteAlloc's shape with a tally instead of a no-op: what the
// resize path asks an element for. A value-initialising buffer passes a value
// here; this one does not.
template <typename T>
struct CountingByteAlloc {
    using value_type = T;

    CountingByteAlloc() = default;
    template <typename U> CountingByteAlloc(const CountingByteAlloc<U>&) noexcept {}

    T* allocate(std::size_t n) { return std::allocator<T>{}.allocate(n); }
    void deallocate(T* p, std::size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }

    void construct(T* p) {
        ++constructed_default;
        ::new (static_cast<void*>(p)) T;
    }
    template <typename... Args>
    void construct(T* p, Args&&... args) {
        ++constructed_value;
        ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
    }
};

// Resident pages of [p, p+n): the pages a write to the buffer would fault in.
// A page-aligned window is what mincore takes, so the window starts at the
// allocation's page.
long resident_pages(const uint8_t* p, std::size_t n) {
    const long    pg = sysconf(_SC_PAGESIZE);
    const uintptr_t at = reinterpret_cast<uintptr_t>(p) & ~static_cast<uintptr_t>(pg - 1);
    const std::size_t head = reinterpret_cast<uintptr_t>(p) - at;
    const std::size_t len  = head + n;
    std::vector<unsigned char> vec((len + static_cast<std::size_t>(pg) - 1) / static_cast<std::size_t>(pg));
    if (mincore(reinterpret_cast<void*>(at), len, vec.data()) != 0) return -1;
    long r = 0;
    for (const unsigned char c : vec) r += (c & 1);
    return r;
}

}  // namespace

TEST(bytes_are_the_init_free_buffer) {
    // the type the checkpoint's state is allocated in: the allocator that
    // default-initialises an element, not the std one that value-initialises
    CHECK((std::is_same_v<Bytes::allocator_type, DefaultInitByteAlloc<uint8_t>>));
}

TEST(bytes_resize_constructs_without_a_value) {
    // the property behind the allocator above: std::vector's resize brings no
    // argument for each element, so the no-argument construct (which writes no
    // byte) is the one that runs; a value-initialising buffer would bring one
    constructed_default = constructed_value = 0;
    std::vector<uint8_t, CountingByteAlloc<uint8_t>> v;
    v.resize(64);
    CHECK_EQ(v.size(), size_t{64});
    CHECK_EQ(constructed_default, 64);
    CHECK_EQ(constructed_value, 0);
}

TEST(bytes_allocation_touches_no_page) {
    // 64 MiB: above glibc's 32 MiB mmap ceiling, so the allocation is a fresh
    // mapping; a value-initialising allocation writes its pages, this one does
    // not. The untouched-fit vector next to it is the control: it shows what a
    // written allocation reports on this host.
    const std::size_t n     = 64u << 20;
    const std::size_t pages = n / static_cast<std::size_t>(sysconf(_SC_PAGESIZE));

    Bytes b;
    b.resize(n);
    CHECK_EQ(b.size(), n);
    CHECK(b.data() != nullptr);
    const long byte_pages = resident_pages(b.data(), n);

    std::vector<uint8_t> v;
    v.resize(n);
    const long written = resident_pages(v.data(), n);

    // without mincore the residency half cannot say anything: report a skip,
    // not a pass (bytes_construct_writes_no_byte holds the property anyway)
    SKIP_UNLESS(byte_pages >= 0 && written >= 0, "mincore unavailable on this host");
    CHECK(written >= static_cast<long>(pages));                 // the control: a written allocation is resident
    CHECK(byte_pages <= static_cast<long>(pages / 100));        // the buffer's: unmapped pages, allocation only
}

TEST(bytes_construct_writes_no_byte) {
    // the property itself, without the host's page accounting: memory filled
    // with a marker keeps it through DefaultInitByteAlloc's no-argument
    // construct, and through a resize and a sized construction of Bytes into
    // storage it reuses (a value-initialising construct writes 0 and fails)
    DefaultInitByteAlloc<uint8_t> a;
    uint8_t* p = a.allocate(64);
    for (int i = 0; i < 64; ++i) p[i] = 0xAA;
    for (int i = 0; i < 64; ++i) a.construct(p + i);
    int kept = 0;
    for (int i = 0; i < 64; ++i) kept += p[i] == 0xAA;
    a.deallocate(p, 64);
    CHECK_EQ(kept, 64);

    Bytes b;
    b.reserve(64);
    b.resize(64);                                 // default-insert into the reserved storage
    for (auto& c : b) c = 0xAA;
    b.clear();                                    // destroys, keeps the storage
    b.resize(64);                                 // default-insert again: the marker must survive
    int kept2 = 0;
    for (const uint8_t c : b) kept2 += c == 0xAA;
    CHECK_EQ(kept2, 64);
}
