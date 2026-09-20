#include "selection_policy.h"

#include <algorithm>
#include <limits>

namespace freetoken::core {

namespace {

QStarSplit split_by_order(const std::vector<uint32_t>& ordered, int q) {
    QStarSplit split;
    const size_t q_clamped = static_cast<size_t>(std::clamp(q, 0, static_cast<int>(ordered.size())));
    split.fill_set.assign(ordered.begin(), ordered.begin() + static_cast<long>(q_clamped));
    split.compute_set.assign(ordered.begin() + static_cast<long>(q_clamped), ordered.end());
    return split;
}

}  // namespace

QStarSplit select_experts(SelectionPolicy policy, const std::vector<uint32_t>& missing_experts, int q,
                           const HostResidentPool& pool, SelectionHistory& history) {
    ++history.clock;  // this decode step's tick -- read below for ranking, written after for next time

    QStarSplit result;

    if (policy == SelectionPolicy::FirstInOrder) {
        // No real logic -- the control. Whatever order the caller listed
        // them in, unchanged.
        result = split_by_order(missing_experts, q);
    } else if (policy == SelectionPolicy::MostRecentlyRequested) {
        // Prefer experts whose last miss was most recent -- a never-seen
        // expert gets tick 0 (lowest priority): no evidence it'll be
        // needed again soon, so it doesn't get prioritized over ones with
        // an actual recent-miss history.
        std::vector<uint32_t> ordered = missing_experts;
        std::stable_sort(ordered.begin(), ordered.end(), [&](uint32_t a, uint32_t b) {
            uint64_t tick_a = history.last_requested_tick.count(a) ? history.last_requested_tick[a] : 0;
            uint64_t tick_b = history.last_requested_tick.count(b) ? history.last_requested_tick[b] : 0;
            return tick_a > tick_b;  // most recent (highest tick) first
        });
        result = split_by_order(ordered, q);
    } else {  // SmallestFirst
        // Prefer the smallest experts -- more fills fit in the same PCIe
        // time budget. Anything not found in the pool (shouldn't happen)
        // sorts last, defensively, rather than crashing or guessing.
        std::vector<uint32_t> ordered = missing_experts;
        std::stable_sort(ordered.begin(), ordered.end(), [&](uint32_t a, uint32_t b) {
            const ExpertSlot* slot_a = pool.find(a);
            const ExpertSlot* slot_b = pool.find(b);
            size_t size_a = slot_a ? slot_a->size_bytes : std::numeric_limits<size_t>::max();
            size_t size_b = slot_b ? slot_b->size_bytes : std::numeric_limits<size_t>::max();
            return size_a < size_b;
        });
        result = split_by_order(ordered, q);
    }

    // Record that every expert in `missing_experts` was requested THIS
    // step, for future calls' ranking -- after selection, not before, so
    // this step's own misses don't influence this step's own ranking.
    for (uint32_t id : missing_experts) {
        history.last_requested_tick[id] = history.clock;
    }

    return result;
}

}  // namespace freetoken::core
