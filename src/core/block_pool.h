#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

// DESIGN.md §3.3/§3.4. The KV pool as a refcounted set of pages.
//
// Until M6 there was one sequence on the card at a time, so a sequence could
// simply own physical blocks 0..n and a "pool epoch" counter was enough to
// tell a stale prefix-cache entry from a live one. With two lanes that is no
// longer true: two sequences hold disjoint pages at the same time, and a
// cached prefix is a third holder of pages that neither lane may overwrite.
//
// So pages are refcounted. A page is handed out with refcount 1, gains a
// reference for every sequence or cache entry that maps it, and returns to the
// free list when the last one goes. Two invariants make that safe without
// copy-on-write machinery in the hot path:
//
//   1. Only *complete* pages are ever shared. A prefix-cache hit lands on a
//      block boundary by construction (§3.4), so every page it maps is full.
//   2. A full page is never written again. The page a sequence writes into is
//      always one it allocated itself, with refcount 1.
//
// Together those mean a shared page is immutable, which is copy-on-write with
// the copy provably never needed rather than merely not implemented. The
// backend asserts (1) rather than assuming it: a hit that is not block-aligned
// falls back to a cold prefill instead of writing into someone else's page.
namespace lgc {

class BlockPool {
public:
    explicit BlockPool(size_t total_blocks);

    // All `n` blocks or none, so a partial allocation never has to be unwound
    // by the caller. Fresh blocks come back with refcount 1.
    std::vector<int32_t> allocate(size_t n);

    // One more holder of every listed block (a prefix-cache entry, or a second
    // lane mapping the same prefix).
    void ref(const std::vector<int32_t>& blocks);

    // One holder fewer. A block whose last reference goes is free again.
    void release(const std::vector<int32_t>& blocks);

    size_t   total() const { return refs_.size(); }
    size_t   free_blocks() const;
    uint32_t refcount(int32_t block) const;

private:
    mutable std::mutex    mutex_;
    std::vector<int32_t>  free_;   // ids with refcount 0; back() is next out
    std::vector<uint32_t> refs_;
};

}  // namespace lgc
