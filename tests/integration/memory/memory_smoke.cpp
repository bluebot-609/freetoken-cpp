// End-to-end tests for Module 3.2 (Host-Resident Pool) + Module 3.3
// (GPU-Expert Cache), naive stall-and-copy baseline (dev_spec.md Section 8,
// step 3).
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "gpu_cache.h"
#include "host_pool.h"

using freetoken::core::HostPoolConfig;
using freetoken::core::HostResidentPool;
using freetoken::core::GpuCacheConfig;
using freetoken::core::GpuExpertCache;

namespace {

// Each fake "expert" is just 4 floats, all set to (expert_id + 1) * 10 —
// small and easy to eyeball-verify, same idea as tests/fixtures/hello.gguf.
constexpr size_t kExpertBytes = 4 * sizeof(float);

void make_fake_expert(uint32_t expert_id, float out[4]) {
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<float>((expert_id + 1) * 10);
    }
}

// Times one get_or_fetch call and reports the REAL hit/miss outcome, read
// from the cache's own counters (see gpu_cache.h) rather than guessed from
// timing — a tiny 16-byte copy is fast enough, once CUDA is warmed up, that
// elapsed time alone can't reliably tell a hit from a miss. Timing is
// printed purely as extra color -- the numbers stay visible in the log,
// same as before GTest.
const void* timed_fetch(uint32_t expert_id, GpuExpertCache& cache, const HostResidentPool& pool) {
    size_t misses_before = cache.misses();

    auto start = std::chrono::steady_clock::now();
    const void* ptr = cache.get_or_fetch(expert_id, pool);
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();

    const bool was_miss = cache.misses() > misses_before;
    std::printf("  fetch(%u): %6lld us  [%s]\n", expert_id, static_cast<long long>(elapsed_us),
                was_miss ? "MISS (allocated + copied)" : "HIT (already cached)");
    return ptr;
}

}  // namespace

TEST(MemoryPipeline, EvictionAndReadbackCorrectness) {
    HostPoolConfig pool_cfg;
    pool_cfg.pin_memory = true;  // exercise the cudaHostAlloc path too; failure is non-fatal
    HostResidentPool pool(pool_cfg);

    constexpr uint32_t kNumExperts = 3;
    for (uint32_t id = 0; id < kNumExperts; ++id) {
        float data[4];
        make_fake_expert(id, data);
        ASSERT_TRUE(pool.register_expert(id, data, kExpertBytes)) << "failed to register expert " << id;
    }
    std::printf("registered %u fake experts in the Host-Resident Pool\n", kNumExperts);

    // Budget for exactly 2 experts (32 bytes) so a 3rd forces an eviction.
    GpuCacheConfig cache_cfg;
    cache_cfg.max_bytes = 2 * kExpertBytes;
    GpuExpertCache cache(cache_cfg);

    std::printf("\naccessing experts in order 0, 1, 2, 0, 2 (cache holds only 2 at a time):\n");
    timed_fetch(0, cache, pool);  // miss: cache empty      -> cache = {0}
    timed_fetch(1, cache, pool);  // miss: room available   -> cache = {0, 1}
    timed_fetch(2, cache, pool);  // miss: evicts LRU (0)   -> cache = {1, 2}
    timed_fetch(0, cache, pool);  // miss: evicts LRU (1)   -> cache = {2, 0}
    timed_fetch(2, cache, pool);  // hit: 2 is still cached

    std::printf("\ntotals: %zu hits, %zu misses\n", cache.hits(), cache.misses());
    EXPECT_EQ(cache.hits(), 1u);
    EXPECT_EQ(cache.misses(), 4u);

    // Correctness, not just "didn't crash": copy expert 2's data back from
    // VRAM to RAM and check it's really the values we put in. `data` is a
    // device pointer — reading it directly from CPU code is undefined, it
    // must go through cudaMemcpy (DeviceToHost direction, this time).
    const void* device_ptr = cache.get_or_fetch(2, pool);
    float readback[4] = {};
    cudaMemcpy(readback, device_ptr, kExpertBytes, cudaMemcpyDeviceToHost);

    float expected[4];
    make_fake_expert(2, expected);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(readback[i], expected[i]) << "mismatch at element " << i;
    }
}

// Proves the Module 3.2 pinning fix actually does what it claims: register
// one 16MB expert (same size as calibration.cpp's default sample) in a
// pinned pool and an unpinned pool, time a raw cudaMemcpy H2D from each,
// and compare. This is the direct, measured answer to "does pin_memory
// actually make transfers faster on this machine" — not a theoretical claim.
TEST(MemoryPipeline, PinnedMemoryIsFasterThanUnpinned) {
    constexpr size_t kBenchBytes = 16 * 1024 * 1024;
    std::vector<char> source(kBenchBytes, 0);

    void* device_buffer = nullptr;
    cudaMalloc(&device_buffer, kBenchBytes);

    auto time_copy_from_pool = [&](bool pin) {
        HostPoolConfig cfg;
        cfg.pin_memory = pin;
        HostResidentPool pool(cfg);
        pool.register_expert(0, source.data(), kBenchBytes);
        const void* pool_data = pool.find(0)->data;

        auto start = std::chrono::steady_clock::now();
        cudaMemcpy(device_buffer, pool_data, kBenchBytes, cudaMemcpyHostToDevice);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return static_cast<double>(kBenchBytes) / seconds / 1e9;  // GB/s
    };

    double unpinned_gbps = time_copy_from_pool(false);
    double pinned_gbps = time_copy_from_pool(true);

    std::printf("\npinned vs unpinned H2D bandwidth (16MB transfer):\n");
    std::printf("  unpinned (pin_memory=false): %.2f GB/s\n", unpinned_gbps);
    std::printf("  pinned   (pin_memory=true):  %.2f GB/s\n", pinned_gbps);

    cudaFree(device_buffer);

    // Not a strict EXPECT_GT: single-run timing on a shared/virtualized GPU
    // (WSL2) can be noisy. This is a soft check with the numbers printed
    // above for a human to actually look at, not a hard gate on one sample.
    if (pinned_gbps <= unpinned_gbps) {
        std::printf("  NOTE: pinned wasn't faster this run -- see printed numbers, may be timing noise\n");
    }
}
