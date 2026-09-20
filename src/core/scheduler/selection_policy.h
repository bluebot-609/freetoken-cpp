#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "host_pool.h"

namespace freetoken::core {

// The result of selecting which experts go where (dev_spec.md Module 3.4).
// `fill_set` goes through the GPU cache (Module 3.3); `compute_set` is
// computed directly on the CPU against the Host-Resident Pool (Module 3.2),
// bypassing the GPU entirely for those experts. (Moved here from
// qstar_scheduler.h to avoid that header and this one including each other.)
struct QStarSplit {
    std::vector<uint32_t> fill_set;
    std::vector<uint32_t> compute_set;
};

// Which heuristic decides *which* q missing experts become the fill set,
// out of all m that missed. The paper delegates this choice to "the cache
// replacement policy" without specifying one -- these are our candidates,
// meant to be benchmarked against each other, not just one guessed choice.
enum class SelectionPolicy {
    FirstInOrder,          // no real logic -- whatever order the caller listed them in. The control.
    MostRecentlyRequested, // prefer experts whose last miss was most recent (temporal-locality bet)
    SmallestFirst,         // prefer the smallest experts (more fills fit in the same PCIe time budget)
};

// Missing experts, by definition, aren't tracked in GpuExpertCache's
// resident-slot state (they're not resident) -- so a recency-based policy
// needs its own small memory of "when was this expert last missed,"
// maintained by whoever calls select_experts() across repeated calls.
struct SelectionHistory {
    std::unordered_map<uint32_t, uint64_t> last_requested_tick;
    uint64_t clock = 0;
};

// Splits `missing_experts` into a fill set (first `q` by `policy`'s
// ranking) and a compute set (the rest), per dev_spec.md Module 3.4.
// Updates `history` with this call's misses, for future calls' ranking.
QStarSplit select_experts(SelectionPolicy policy, const std::vector<uint32_t>& missing_experts, int q,
                           const HostResidentPool& pool, SelectionHistory& history);

}  // namespace freetoken::core
