#include "core/block_pool.h"

namespace lgc {

BlockPool::BlockPool(size_t total_blocks) : refs_(total_blocks, 0) {
    free_.reserve(total_blocks);
    // Handed out low id first, which keeps a fresh sequence's pages contiguous
    // and its block table readable in a trace.
    for (size_t i = total_blocks; i > 0; --i) free_.push_back(static_cast<int32_t>(i - 1));
}

std::vector<int32_t> BlockPool::allocate(size_t n) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (n > free_.size()) return {};
    std::vector<int32_t> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const int32_t id = free_.back();
        free_.pop_back();
        refs_[static_cast<size_t>(id)] = 1;
        out.push_back(id);
    }
    return out;
}

void BlockPool::ref(const std::vector<int32_t>& blocks) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (int32_t id : blocks) {
        if (id < 0 || static_cast<size_t>(id) >= refs_.size()) continue;
        // A page at zero is on the free list, so nothing legitimately holds it
        // and this is a caller bug. Refusing is the safe half of the choice:
        // incrementing would resurrect a page allocate() is about to hand to
        // someone else, and two sequences would write one KV page. Mirrors the
        // guard release() has for the same mistake in the other direction.
        if (refs_[static_cast<size_t>(id)] == 0) continue;
        ++refs_[static_cast<size_t>(id)];
    }
}

void BlockPool::release(const std::vector<int32_t>& blocks) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (int32_t id : blocks) {
        if (id < 0 || static_cast<size_t>(id) >= refs_.size()) continue;
        uint32_t& r = refs_[static_cast<size_t>(id)];
        if (r == 0) continue;   // double release: ignored, never a negative count
        if (--r == 0) free_.push_back(id);
    }
}

size_t BlockPool::free_blocks() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return free_.size();
}

uint32_t BlockPool::refcount(int32_t block) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (block < 0 || static_cast<size_t>(block) >= refs_.size()) return 0;
    return refs_[static_cast<size_t>(block)];
}

}  // namespace lgc
