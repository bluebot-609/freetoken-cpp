// Module 3.8 unit tests. dev_spec.md Section 8, step 4: unit-test the
// lock-free/SPSC queue under concurrent load BEFORE wiring it into the
// scheduler or server -- a concurrency bug is far easier to isolate here
// than after integration.
#include <gtest/gtest.h>

#include <thread>

#include "spsc_queue.h"

using freetoken::core::SpscQueue;

// Single-threaded sanity check first: no concurrency involved yet, just
// "does the ring-buffer bookkeeping work at all."
TEST(SpscQueue, SingleThreadedBasics) {
    SpscQueue<int> q(4);

    int out = 0;
    EXPECT_FALSE(q.pop(out)) << "pop() on empty queue should fail";

    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(q.push(i)) << "push() unexpectedly failed at i=" << i;
    }
    EXPECT_FALSE(q.push(999)) << "push() on full queue should fail";

    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(q.pop(out));
        EXPECT_EQ(out, i);
    }
    EXPECT_FALSE(q.pop(out)) << "pop() on empty queue (again) should fail";
}

// The real point of this whole file: an actual producer thread and an
// actual consumer thread, hammering a deliberately tiny queue so it wraps
// around constantly. If push()/pop()'s memory ordering were wrong, this is
// the kind of test that would catch it (a single-threaded test never would).
// Small capacity (16) so 200,000 items force ~12,500 wrap-arounds --
// deliberately stressing the boundary conditions, not just the happy path.
// Races are timing-dependent -- a single pass proves less than it looks
// like. Run this one repeatedly for real confidence:
//   ./spsc_queue_test --gtest_filter=SpscQueue.ConcurrentStressNoDropsOrReorders --gtest_repeat=8
TEST(SpscQueue, ConcurrentStressNoDropsOrReorders) {
    constexpr int kNumItems = 200000;
    constexpr size_t kQueueCapacity = 16;

    SpscQueue<int> q(kQueueCapacity);
    bool consumer_ok = true;

    std::thread producer([&] {
        for (int i = 0; i < kNumItems; ++i) {
            while (!q.push(i)) {
                // queue full — spin, don't block. This is a deliberate
                // busy-wait, not a bug: SpscQueue never blocks by design.
            }
        }
    });

    std::thread consumer([&] {
        for (int expected = 0; expected < kNumItems; ++expected) {
            int got = 0;
            while (!q.pop(got)) {
                // queue empty — spin.
            }
            if (got != expected) {
                consumer_ok = false;
            }
        }
    });

    producer.join();
    consumer.join();
    EXPECT_TRUE(consumer_ok) << "consumer saw an out-of-order or dropped item";
}
