#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "host_pool.h"

namespace freetoken::core {

/// One expert's weights, cached in VRAM. Mirrors ExpertSlot, but `data` is a
/// CUDA device pointer, not host memory — never dereference it from CPU code.
struct GpuSlot {
    uint32_t expert_id  = 0;
    void*    data       = nullptr;  // device pointer (cudaMalloc'd)
    size_t   size_bytes = 0;
    uint64_t last_used  = 0;        // tick count, for LRU eviction (next chunk)
};

/// Exposed parameters for the GPU-Expert Cache (dev_spec.md Module 3.3).
/// Only LRU is implemented for the baseline — dev_spec.md explicitly says
/// start with LRU and add frequency-weighting only after LRU is benchmarked,
/// so there's no eviction_policy enum here yet; adding one now for a policy
/// that doesn't exist would be exactly the "abstraction for the future" this
/// project's own principles say not to do.
struct GpuCacheConfig {
    size_t max_bytes = 0;  // VRAM budget for the cache
};

/// A dynamic, shared cache of currently-hot experts in VRAM. On a miss,
/// synchronously copies from the Host-Resident Pool — this "stall and copy"
/// behavior is the deliberate naive baseline (dev_spec.md Section 8, step 3)
/// that the real q* scheduler (Phase 1's actual point) gets benchmarked
/// against later.
class GpuExpertCache {
public:
    explicit GpuExpertCache(GpuCacheConfig config);
    ~GpuExpertCache();

    GpuExpertCache(const GpuExpertCache&) = delete;
    GpuExpertCache& operator=(const GpuExpertCache&) = delete;

    /// Returns a device pointer to `expert_id`'s weights. On a cache hit,
    /// this is fast. On a miss, this blocks the caller while it evicts (if
    /// needed) and copies from `host_pool` — that stall is the point of the
    /// baseline. Returns nullptr if `expert_id` isn't in `host_pool` either.
    const void* get_or_fetch(uint32_t expert_id, const HostResidentPool& host_pool);

    /// True if `expert_id` is currently resident, WITHOUT fetching it or
    /// counting as a hit/miss, and without touching its LRU recency. A
    /// real caller needs this to even determine what "missing_experts" is
    /// in the first place, before calling QStarScheduler::run_step() --
    /// added when building the first thing that actually needed it (a
    /// multi-step selection-policy benchmark), not speculatively.
    bool contains(uint32_t expert_id) const;

    /// Ground truth for hit/miss behavior — roadmap.md lists "cache hit
    /// rate" as required instrumentation for Phase 1 anyway; timing alone
    /// isn't a reliable way to tell hit from miss apart (a tiny copy is
    /// fast enough that it can look just like a hit once CUDA is warmed up).
    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }

private:
    GpuCacheConfig config_;
    std::vector<GpuSlot> slots_;
    size_t used_bytes_ = 0;
    uint64_t clock_ = 0;  // incremented on every access, for LRU
    size_t hits_ = 0;
    size_t misses_ = 0;
};

}  // namespace freetoken::core
