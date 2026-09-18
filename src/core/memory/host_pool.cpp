#include "host_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>  // mlock/munlock — Linux-only, see dev_spec.md Section 2

namespace freetoken::core {

HostResidentPool::HostResidentPool(HostPoolConfig config) : config_(config) {}

HostResidentPool::~HostResidentPool() {
    for (auto& slot : slots_) {
        if (config_.use_mlock) {
            // munlock() on memory that isn't actually locked just returns an
            // error harmlessly — no need to track per-slot whether the
            // mlock() call in register_expert() below actually succeeded.
            munlock(slot.data, slot.size_bytes);
        }
        std::free(slot.data);
    }
}

bool HostResidentPool::register_expert(uint32_t expert_id, const void* src_data, size_t size_bytes) {
    void* dst = std::malloc(size_bytes);
    if (!dst) {
        return false;
    }
    std::memcpy(dst, src_data, size_bytes);

    if (config_.use_mlock) {
        // Pin these pages so the kernel can never swap them out — see the
        // explanation in docs/citations.md / CLAUDE.md discussion: the
        // pool's whole job is "always fast to read," and a swapped-out
        // page would silently turn a copy into a slow disk read. Failure
        // is logged, not fatal — mlock() commonly fails without elevated
        // rlimits, and that's a real, expected tradeoff to surface, not
        // hide (dev_spec.md Section 3.2).
        if (mlock(dst, size_bytes) != 0) {
            std::fprintf(stderr, "HostResidentPool: mlock failed for expert %u (%zu bytes) — continuing unpinned\n",
                         expert_id, size_bytes);
        }
    }

    ExpertSlot slot;
    slot.expert_id = expert_id;
    slot.data = dst;
    slot.size_bytes = size_bytes;
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
