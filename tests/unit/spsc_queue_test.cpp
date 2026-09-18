// Module 3.8 unit tests. dev_spec.md Section 8, step 4: unit-test the
// lock-free/SPSC queue under concurrent load BEFORE wiring it into the
// scheduler or server -- a concurrency bug is far easier to isolate here
// than after integration.
#include <cassert>
#include <cstdio>
#include <thread>

#include "spsc_queue.h"

using freetoken::core::SpscQueue;

// Single-threaded sanity check first: no concurrency involved yet, just
// "does the ring-buffer bookkeeping work at all."
static bool test_single_threaded_basics() {
    SpscQueue<int> q(4);

    int out = 0;
    if (q.pop(out)) { std::fprintf(stderr, "pop() on empty queue should fail\n"); return false; }

    for (int i = 0; i < 4; ++i) {
        if (!q.push(i)) { std::fprintf(stderr, "push() unexpectedly failed at i=%d\n", i); return false; }
    }
    if (q.push(999)) { std::fprintf(stderr, "push() on full queue should fail\n"); return false; }

    for (int i = 0; i < 4; ++i) {
        if (!q.pop(out) || out != i) { std::fprintf(stderr, "pop() returned %d, expected %d\n", out, i); return false; }
    }
    if (q.pop(out)) { std::fprintf(stderr, "pop() on empty queue (again) should fail\n"); return false; }

    return true;
}

// The real point of this whole file: an actual producer thread and an
// actual consumer thread, hammering a deliberately tiny queue so it wraps
// around constantly. If push()/pop()'s memory ordering were wrong, this is
// the kind of test that would catch it (a single-threaded test never would).
static bool test_concurrent_stress(int num_items, size_t queue_capacity) {
    SpscQueue<int> q(queue_capacity);
    bool producer_ok = true;
    bool consumer_ok = true;

    std::thread producer([&] {
        for (int i = 0; i < num_items; ++i) {
            while (!q.push(i)) {
                // queue full — spin, don't block. This is a deliberate
                // busy-wait, not a bug: SpscQueue never blocks by design.
            }
        }
    });

    std::thread consumer([&] {
        for (int expected = 0; expected < num_items; ++expected) {
            int got = 0;
            while (!q.pop(got)) {
                // queue empty — spin.
            }
            if (got != expected) {
                std::fprintf(stderr, "consumer: expected %d, got %d\n", expected, got);
                consumer_ok = false;
            }
        }
    });

    producer.join();
    consumer.join();
    return producer_ok && consumer_ok;
}

int main() {
    if (!test_single_threaded_basics()) {
        std::printf("test_single_threaded_basics: FAIL\n");
        return 1;
    }
    std::printf("test_single_threaded_basics: PASS\n");

    // Small capacity (16) so 200,000 items force ~12,500 wrap-arounds —
    // deliberately stressing the boundary conditions, not just the happy path.
    if (!test_concurrent_stress(/*num_items=*/200000, /*queue_capacity=*/16)) {
        std::printf("test_concurrent_stress: FAIL\n");
        return 1;
    }
    std::printf("test_concurrent_stress: PASS (200000 items, capacity 16, no drops/reorders)\n");

    return 0;
}
