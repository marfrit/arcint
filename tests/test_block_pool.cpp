#include "core/block_pool.h"

#include <atomic>
#include <set>
#include <thread>

#include "harness.h"

using lgc::BlockPool;

TEST(block_pool_hands_out_distinct_blocks) {
    BlockPool pool(8);
    CHECK_EQ(pool.total(), size_t{8});
    CHECK_EQ(pool.free_blocks(), size_t{8});

    const std::vector<int32_t> a = pool.allocate(3);
    const std::vector<int32_t> b = pool.allocate(3);
    CHECK_EQ(a.size(), size_t{3});
    CHECK_EQ(b.size(), size_t{3});

    std::set<int32_t> all(a.begin(), a.end());
    all.insert(b.begin(), b.end());
    CHECK_EQ(all.size(), size_t{6});      // no page handed to two lanes
    CHECK_EQ(pool.free_blocks(), size_t{2});
    for (int32_t id : a) CHECK_EQ(pool.refcount(id), uint32_t{1});
}

TEST(block_pool_allocation_is_all_or_nothing) {
    BlockPool pool(4);
    const std::vector<int32_t> big = pool.allocate(5);
    CHECK(big.empty());
    // The failed request must not have consumed anything on its way out.
    CHECK_EQ(pool.free_blocks(), size_t{4});
    CHECK_EQ(pool.allocate(4).size(), size_t{4});
    CHECK_EQ(pool.free_blocks(), size_t{0});
}

TEST(block_pool_shared_pages_survive_one_holder_leaving) {
    // The agent+subagent case: one lane prefills, the prefix cache keeps the
    // pages, a second lane maps the same prefix. The first lane finishing must
    // not pull the pages out from under either of the others.
    BlockPool pool(8);
    const std::vector<int32_t> prefix = pool.allocate(3);

    pool.ref(prefix);      // the prefix-cache entry
    pool.ref(prefix);      // the second lane
    for (int32_t id : prefix) CHECK_EQ(pool.refcount(id), uint32_t{3});

    pool.release(prefix);  // lane one is done
    CHECK_EQ(pool.free_blocks(), size_t{5});
    for (int32_t id : prefix) CHECK_EQ(pool.refcount(id), uint32_t{2});

    pool.release(prefix);  // the cache entry is evicted
    for (int32_t id : prefix) CHECK_EQ(pool.refcount(id), uint32_t{1});
    CHECK_EQ(pool.free_blocks(), size_t{5});

    pool.release(prefix);  // lane two finishes: now, and only now, they are free
    CHECK_EQ(pool.free_blocks(), size_t{8});
    for (int32_t id : prefix) CHECK_EQ(pool.refcount(id), uint32_t{0});
}

TEST(block_pool_released_pages_come_back) {
    BlockPool pool(4);
    std::vector<int32_t> a = pool.allocate(4);
    CHECK(pool.allocate(1).empty());
    pool.release(a);
    CHECK_EQ(pool.free_blocks(), size_t{4});
    CHECK_EQ(pool.allocate(4).size(), size_t{4});
}

TEST(block_pool_double_release_is_not_a_negative_count) {
    // A lane that released its table twice would otherwise hand the same page
    // to two sequences later, which is exactly the cross-slot bleed M6 gates.
    BlockPool pool(2);
    const std::vector<int32_t> a = pool.allocate(1);
    pool.release(a);
    pool.release(a);
    CHECK_EQ(pool.free_blocks(), size_t{2});
    CHECK_EQ(pool.refcount(a.front()), uint32_t{0});
    const std::vector<int32_t> b = pool.allocate(2);
    CHECK_EQ(b.size(), size_t{2});
    CHECK(b[0] != b[1]);
}

TEST(block_pool_is_safe_under_two_lanes) {
    // Two threads allocating and releasing at once must never see the same id,
    // and the pool must return to full when both are done.
    BlockPool         pool(64);
    std::atomic<bool> collision{false};
    std::atomic<int>  ready{0};

    auto lane = [&] {
        ++ready;
        for (int i = 0; i < 500; ++i) {
            std::vector<int32_t> mine = pool.allocate(4);
            if (mine.empty()) continue;
            for (int32_t id : mine) {
                if (pool.refcount(id) != 1) collision = true;
            }
            pool.release(mine);
        }
    };
    std::thread t1(lane), t2(lane);
    t1.join();
    t2.join();

    CHECK(!collision.load());
    CHECK_EQ(pool.free_blocks(), size_t{64});
}

TEST(block_pool_will_not_resurrect_a_free_page) {
    // ref() on a page that is on the free list is a caller bug, and the unsafe
    // reading of it is the dangerous one: allocate() would hand the same page
    // to someone else and two sequences would write one KV page.
    BlockPool pool(4);
    const std::vector<int32_t> a = pool.allocate(2);
    pool.release(a);
    CHECK_EQ(pool.free_blocks(), size_t{4});

    pool.ref(a);                                  // stale reference, refused
    CHECK_EQ(pool.refcount(a.front()), uint32_t{0});
    CHECK_EQ(pool.free_blocks(), size_t{4});

    const std::vector<int32_t> b = pool.allocate(4);
    CHECK_EQ(b.size(), size_t{4});
    std::set<int32_t> distinct(b.begin(), b.end());
    CHECK_EQ(distinct.size(), size_t{4});
}
