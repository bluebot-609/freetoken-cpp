#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

namespace freetoken::core {

/// A fixed-capacity, lock-free single-producer/single-consumer queue
/// (dev_spec.md Module 3.8). Exactly one thread may call push(); exactly one
/// (possibly different) thread may call pop(). Calling push() from two
/// threads at once, or pop() from two threads at once, is undefined —
/// SPSC means what it says.
///
/// Cite: Anthony Williams, "C++ Concurrency in Action" (the SPSC
/// ring-buffer-with-monotonic-counters pattern used here is the standard
/// approach described there) — also listed in roadmap.md's concurrency
/// resources.
template <typename T>
class SpscQueue {
public:
    explicit SpscQueue(size_t capacity);

    /// Producer side. Returns false if the queue is full (does not block).
    bool push(const T& item);

    /// Consumer side. Returns false if the queue is empty (does not block).
    bool pop(T& out);

private:
    std::vector<T> buffer_;
    const size_t capacity_;

    /// Monotonically increasing counters, not wrapped indices — the classic
    /// way to avoid ambiguity between "empty" and "full" in a ring buffer.
    /// head_ is written only by the consumer, tail_ only by the producer.
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
};

template <typename T>
SpscQueue<T>::SpscQueue(size_t capacity) : buffer_(capacity), capacity_(capacity) {}

template <typename T>
bool SpscQueue<T>::push(const T& item) {
    const size_t current_tail = tail_.load(std::memory_order_relaxed);  // relaxed: only this thread ever writes tail_
    const size_t next_tail = current_tail + 1;

    /// memory_order_acquire here: we need to see the CONSUMER's most recent
    /// head_ update (from its own release store in pop(), below) to know
    /// how much room is really free — a relaxed load here could see a
    /// stale head_ and wrongly think the queue is full.
    if (next_tail - head_.load(std::memory_order_acquire) > capacity_) {
        return false;  // full
    }

    buffer_[current_tail % capacity_] = item;

    /// memory_order_release: this publishes both the new tail_ value AND
    /// (because release "happens after" everything before it in this
    /// thread) the buffer_ write above, so the consumer's acquire load of
    /// tail_ is guaranteed to see the item we just wrote, not stale data.
    tail_.store(next_tail, std::memory_order_release);
    return true;
}

template <typename T>
bool SpscQueue<T>::pop(T& out) {
    const size_t current_head = head_.load(std::memory_order_relaxed);  // relaxed: only this thread ever writes head_

    /// memory_order_acquire: mirrors push()'s release store of tail_ above —
    /// this is what guarantees that if we see the producer's updated
    /// tail_, we also see the buffer_ write it made just before that store.
    if (current_head == tail_.load(std::memory_order_acquire)) {
        return false;  // empty
    }

    out = buffer_[current_head % capacity_];

    /// release: publishes that this slot is now free, so a subsequent
    /// push()'s acquire load of head_ sees it (see the capacity check above).
    head_.store(current_head + 1, std::memory_order_release);
    return true;
}

}  // namespace freetoken::core
