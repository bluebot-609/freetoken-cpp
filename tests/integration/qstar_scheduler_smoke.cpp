// End-to-end smoke test + benchmark for Module 3.4 (q* scheduler),
// dev_spec.md Section 8 step 5 -- benchmarked against the Modules 3.2/3.3
// stall-and-copy baseline immediately, per dev_spec.md's own instruction.
#include <cstdio>
#include <chrono>
#include <vector>

#include "qstar_scheduler.h"

using namespace freetoken::core;

namespace {

constexpr size_t kExpertBytes = 4 * sizeof(float);

void make_fake_expert(uint32_t id, float out[4]) {
    for (int i = 0; i < 4; ++i) out[i] = static_cast<float>((id + 1) * 10);
}

}  // namespace

int main() {
    // --- Part 1: correctness, with known numbers (same case already
    // proven in qstar_scheduler_test.cpp: m=8, ratio 0.25 -> q=2) ---
    HostPoolConfig pool_cfg;
    HostResidentPool pool(pool_cfg);
    for (uint32_t id = 0; id < 8; ++id) {
        float data[4];
        make_fake_expert(id, data);
        pool.register_expert(id, data, kExpertBytes);
    }

    GpuCacheConfig cache_cfg;
    cache_cfg.max_bytes = 100 * kExpertBytes;  // generous -- no evictions here, keep this test focused
    GpuExpertCache cache(cache_cfg);

    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 10e9;
    bw.b_host_bytes_per_sec = 40e9;

    bool ok = true;
    {
        QStarScheduler scheduler(pool, cache);

        std::vector<uint32_t> missing = {0, 1, 2, 3, 4, 5, 6, 7};
        QStarStepResult result = scheduler.run_step(missing, bw);

        std::printf("step 1: m=%d q=%d gpu_fill=%.1fus cpu_compute=%.1fus\n",
                    result.m, result.q, result.gpu_fill_seconds * 1e6, result.cpu_compute_seconds * 1e6);

        if (result.m != 8) { std::fprintf(stderr, "expected m=8, got %d\n", result.m); ok = false; }
        if (result.q != 2) { std::fprintf(stderr, "expected q=2, got %d\n", result.q); ok = false; }

        // Fill set was experts {0, 1} (first q=2, per split_missing_experts'
        // documented order). They should now be real cache HITS.
        size_t hits_before = cache.hits();
        cache.get_or_fetch(0, pool);
        cache.get_or_fetch(1, pool);
        if (cache.hits() != hits_before + 2) {
            std::fprintf(stderr, "expected experts 0,1 to be cache hits after fill -- they weren't\n");
            ok = false;
        }

        // --- Part 2: reuse the SAME scheduler for a second step, proving
        // the persistent worker thread survives across calls. ---
        std::vector<uint32_t> missing2 = {8, 9};  // not registered -- exercises the "not found" path harmlessly
        pool.register_expert(8, missing.data(), kExpertBytes);  // reuse missing's buffer as filler data, contents don't matter here
        pool.register_expert(9, missing.data(), kExpertBytes);
        QStarStepResult result2 = scheduler.run_step(missing2, bw);
        std::printf("step 2: m=%d q=%d gpu_fill=%.1fus cpu_compute=%.1fus\n",
                    result2.m, result2.q, result2.gpu_fill_seconds * 1e6, result2.cpu_compute_seconds * 1e6);
        if (result2.m != 2) { std::fprintf(stderr, "expected m=2 on step 2, got %d\n", result2.m); ok = false; }

        // scheduler destructs here (end of block) -- proves clean worker
        // thread shutdown (join(), not abandoned) before the next section.
    }
    std::printf("scheduler shutdown: ok (no hang)\n");

    std::printf("\ncorrectness: %s\n", ok ? "PASS" : "FAIL");

    // --- Part 3: the actual benchmark dev_spec.md asks for -- q* split vs
    // naive stall-and-copy (everything through the GPU cache, no CPU
    // offload), on a larger, more realistic batch. ---
    constexpr size_t kBenchExpertBytes = 256 * 1024;  // 256KB/expert, closer to a real (if tiny) weight tensor
    constexpr uint32_t kBenchExpertCount = 40;

    HostPoolConfig bench_pool_cfg;
    HostResidentPool bench_pool(bench_pool_cfg);
    std::vector<char> filler(kBenchExpertBytes, 0);
    for (uint32_t id = 0; id < kBenchExpertCount; ++id) {
        bench_pool.register_expert(id, filler.data(), kBenchExpertBytes);
    }
    std::vector<uint32_t> all_ids;
    for (uint32_t id = 0; id < kBenchExpertCount; ++id) all_ids.push_back(id);

    // Naive baseline: every expert through the GPU cache, sequentially,
    // no CPU offload at all -- exactly Modules 3.2/3.3 alone.
    {
        GpuCacheConfig naive_cfg;
        naive_cfg.max_bytes = kBenchExpertCount * kBenchExpertBytes;
        GpuExpertCache naive_cache(naive_cfg);

        auto start = std::chrono::steady_clock::now();
        for (uint32_t id : all_ids) {
            naive_cache.get_or_fetch(id, bench_pool);
        }
        double naive_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("\nnaive baseline (all %u experts via GPU cache): %.2f ms\n",
                    kBenchExpertCount, naive_seconds * 1e3);
    }

    // q* split path: same experts, same "all missing" starting condition.
    {
        GpuCacheConfig split_cfg;
        split_cfg.max_bytes = kBenchExpertCount * kBenchExpertBytes;
        GpuExpertCache split_cache(split_cfg);
        QStarScheduler scheduler(bench_pool, split_cache);

        auto start = std::chrono::steady_clock::now();
        QStarStepResult result = scheduler.run_step(all_ids, bw);
        double split_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("q* split (q=%d fill / %d compute): %.2f ms\n",
                    result.q, result.m - result.q, split_seconds * 1e3);
    }

    return ok ? 0 : 1;
}
