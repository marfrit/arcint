// A byte buffer for a llama state blob: allocated, not written. Every byte of
// a context checkpoint's buffer is written by llama_state_seq_get_data_ext, so
// a fresh std::vector<uint8_t>'s resize would write `size` zero bytes first and
// throw them away (about 150 MiB a save on the 27B agent, one recurrent plane
// set, `inferred` from the state's size). This allocator's construct
// default-initialises the element instead of value-initialising it (a placement
// new without a value): it writes no byte, and with optimisation the no-op
// element loop compiles to nothing (at -O0 or under a sanitizer it is one call a
// byte). What that saves is the zeroing pass only: a fresh mapping's first-touch
// page faults still happen, now inside the state call. Measured on the recipe
// host's CPU (median of 11, 150 MiB, allocation plus the full copy a save
// does): 73.2-74.8 ms with std::vector, 66.3-68.6 ms with Bytes.
// tests/test_bytes.cpp holds the property, and it fails if a
// value-initialising buffer comes back.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace lgc {

// std::allocator's allocation and deallocation, its construct overridden to
// leave an element default-initialised. Only for bytes whose every value a
// state call writes: an element nobody writes would read as whatever the
// allocation left there.
template <typename T>
struct DefaultInitByteAlloc {
    using value_type      = T;
    using is_always_equal = std::true_type;

    DefaultInitByteAlloc() = default;
    template <typename U> DefaultInitByteAlloc(const DefaultInitByteAlloc<U>&) noexcept {}

    T* allocate(std::size_t n) { return std::allocator<T>{}.allocate(n); }
    void deallocate(T* p, std::size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }

    // std::vector's resize(n) and its sized constructor vector(n) bring no
    // argument for each element: default-initialise it, so the allocation
    // writes no byte. The state call that follows fills all of them.
    void construct(T* p) { ::new (static_cast<void*>(p)) T; }

    // A path that brings a value (vector(n, v), resize(n, v), assign(n, v))
    // still passes it and still writes it.
    template <typename... Args>
    void construct(T* p, Args&&... args) {
        ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
    }

    void destroy(T* p) { p->~T(); }
};

template <typename T, typename U>
bool operator==(const DefaultInitByteAlloc<T>&, const DefaultInitByteAlloc<U>&) noexcept { return true; }
template <typename T, typename U>
bool operator!=(const DefaultInitByteAlloc<T>&, const DefaultInitByteAlloc<U>&) noexcept { return false; }

// The bytes of a context checkpoint (exec/backend_llama.cpp) and of the
// drafter's state a checkpoint carries (exec/spec_state.h). Same interface as
// std::vector<uint8_t> at the call sites: resize, data, size, empty, clear.
using Bytes = std::vector<uint8_t, DefaultInitByteAlloc<uint8_t>>;

}  // namespace lgc
