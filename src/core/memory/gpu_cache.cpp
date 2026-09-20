#include "gpu_cache.h"

#include <cstdio>
#include <limits>

#include <cuda_runtime.h>

namespace freetoken::core {

GpuExpertCache::GpuExpertCache(GpuCacheConfig config) : config_(config) {}

GpuExpertCache::~GpuExpertCache() {
    for (auto& slot : slots_) {
        cudaFree(slot.data);
    }
}

bool GpuExpertCache::contains(uint32_t expert_id) const {
    for (const auto& slot : slots_) {
        if (slot.expert_id == expert_id) {
            return true;
        }
    }
    return false;
}

const void* GpuExpertCache::get_or_fetch(uint32_t expert_id, const HostResidentPool& host_pool) {
    ++clock_;

    // Cache hit: already in VRAM, just update its LRU timestamp and return.
    for (auto& slot : slots_) {
        if (slot.expert_id == expert_id) {
            slot.last_used = clock_;
            ++hits_;
            return slot.data;
        }
    }
    ++misses_;

    // Miss: this expert must exist in the Host-Resident Pool, or there's
    // nothing to copy.
    const ExpertSlot* host_slot = host_pool.find(expert_id);
    if (!host_slot) {
        return nullptr;
    }

    // Evict LRU entries (naive baseline: linear scan for the oldest
    // last_used, same "fine for small N" precedent as HostResidentPool::find)
    // until there's room for the incoming expert.
    while (used_bytes_ + host_slot->size_bytes > config_.max_bytes && !slots_.empty()) {
        size_t victim_idx = 0;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].last_used < oldest) {
                oldest = slots_[i].last_used;
                victim_idx = i;
            }
        }
        cudaFree(slots_[victim_idx].data);
        used_bytes_ -= slots_[victim_idx].size_bytes;
        slots_.erase(slots_.begin() + static_cast<long>(victim_idx));
    }

    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, host_slot->size_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GpuExpertCache: cudaMalloc failed for expert %u: %s\n",
                     expert_id, cudaGetErrorString(err));
        return nullptr;
    }

    // The actual stall: this call does not return until the copy is done.
    err = cudaMemcpy(device_ptr, host_slot->data, host_slot->size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GpuExpertCache: cudaMemcpy failed for expert %u: %s\n",
                     expert_id, cudaGetErrorString(err));
        cudaFree(device_ptr);
        return nullptr;
    }

    GpuSlot slot;
    slot.expert_id = expert_id;
    slot.data = device_ptr;
    slot.size_bytes = host_slot->size_bytes;
    slot.last_used = clock_;
    slots_.push_back(slot);
    used_bytes_ += host_slot->size_bytes;
    return device_ptr;
}

}  // namespace freetoken::core
