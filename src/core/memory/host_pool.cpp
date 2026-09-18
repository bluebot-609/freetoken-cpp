#include "host_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

namespace freetoken::core {

HostResidentPool::HostResidentPool(HostPoolConfig config) : config_(config) {}

HostResidentPool::~HostResidentPool() {
    for (auto& slot : slots_) {
        // Must match whichever allocator actually produced slot.data —
        // cudaFreeHost() on a plain malloc() pointer (or vice versa) is
        // undefined behavior, not a harmless no-op like munlock() was.
        if (slot.pinned) {
            cudaFreeHost(slot.data);
        } else {
            std::free(slot.data);
        }
    }
}

bool HostResidentPool::register_expert(uint32_t expert_id, const void* src_data, size_t size_bytes) {
    void* dst = nullptr;
    bool pinned = false;

    if (config_.pin_memory) {
        // cudaHostAlloc page-locks this memory (same "can't be swapped out"
        // guarantee mlock gave) AND registers it with CUDA so the GPU's DMA
        // engine can transfer directly from it, skipping the internal
        // staging-buffer copy CUDA otherwise inserts for ordinary
        // ("pageable") memory — see docs/citations.md for the measured PCIe
        // bandwidth difference this makes. Failure is logged, not fatal:
        // falls back to plain malloc, same "surface the tradeoff, don't
        // hide it" pattern as before.
        if (cudaHostAlloc(&dst, size_bytes, cudaHostAllocDefault) == cudaSuccess) {
            pinned = true;
        } else {
            std::fprintf(stderr, "HostResidentPool: cudaHostAlloc failed for expert %u (%zu bytes) — falling back to malloc\n",
                         expert_id, size_bytes);
        }
    }
    if (!dst) {
        dst = std::malloc(size_bytes);
    }
    if (!dst) {
        return false;
    }
    std::memcpy(dst, src_data, size_bytes);

    ExpertSlot slot;
    slot.expert_id = expert_id;
    slot.data = dst;
    slot.size_bytes = size_bytes;
    slot.pinned = pinned;
    slots_.push_back(slot);
    return true;
}

const ExpertSlot* HostResidentPool::find(uint32_t expert_id) const {
    // Linear search: fine for a handful of experts (the naive baseline).
    // Revisit with a hash map if the expert count/lookup rate ever makes
    // this measurably slow — no evidence yet that it will.
    for (const auto& slot : slots_) {
        if (slot.expert_id == expert_id) {
            return &slot;
        }
    }
    return nullptr;
}

}  // namespace freetoken::core
