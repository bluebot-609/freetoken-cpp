#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace freetoken::core {

// One expert's weights, as stored in the Host-Resident Pool (dev_spec.md
// Module 3.2). This is deliberately a plain data holder with no behavior
// yet — allocation, pinning, and lookup are separate concerns that get
// built on top of this in the next chunks.
struct ExpertSlot {
    uint32_t expert_id = 0;   // which expert this is (index into the model's expert list)
    void*    data      = nullptr;  // raw pointer to this expert's weight bytes
    size_t   size_bytes = 0;       // how many bytes `data` points to
};

// Exposed parameters for the Host-Resident Pool (dev_spec.md Module 3.2).
// `use_mlock` and `use_madvise_hints` are separate knobs, not one on/off
// switch — pinning helps transfer speed but reduces memory available to the
// OS, and dev_spec.md explicitly says expose that tradeoff, don't hide it.
struct HostPoolConfig {
    size_t max_bytes         = 0;      // 0 = no explicit cap yet (naive baseline)
    bool   use_mlock         = false;  // pin memory so it can't be paged out
    // MADV_WILLNEED/MADV_DONTNEED hinting. Stored but not yet exercised —
    // there's no eviction or predictive prefetch logic yet for it to hint
    // about (that's Phase 1b/GPU-Expert Cache work). Revisit then.
    bool   use_madvise_hints = false;
};

// Holds the *complete* set of an MoE model's expert weights in host RAM —
// the "source of truth" the GPU-Expert Cache (Module 3.3) copies from on a
// cache miss. dev_spec.md Module 3.2.
class HostResidentPool {
public:
    explicit HostResidentPool(HostPoolConfig config);
    ~HostResidentPool();

    // This class owns raw memory allocations (and, later, mlock'd pages).
    // Copying it would mean two objects trying to free the same memory —
    // so copying is disabled outright rather than writing a deep-copy that
    // nothing actually needs yet. (This is a common modern-C++ pattern for
    // any class that owns a resource: RAII — the constructor acquires it,
    // the destructor releases it, and copies are blocked unless you
    // deliberately implement one.)
    HostResidentPool(const HostResidentPool&) = delete;
    HostResidentPool& operator=(const HostResidentPool&) = delete;

    // Copies `size_bytes` from `src_data` into pool-owned memory, tagged
    // with `expert_id`. Returns false if allocation fails.
    bool register_expert(uint32_t expert_id, const void* src_data, size_t size_bytes);

    // Returns the slot for `expert_id`, or nullptr if it was never registered.
    const ExpertSlot* find(uint32_t expert_id) const;

private:
    HostPoolConfig config_;
    std::vector<ExpertSlot> slots_;
};

}  // namespace freetoken::core
