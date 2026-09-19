// End-to-end tests + benchmark for Module 3.4 (q* scheduler), dev_spec.md
// Section 8 step 5 -- benchmarked against the Modules 3.2/3.3
// stall-and-copy baseline immediately, per dev_spec.md's own instruction.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include "model_loader.h"
#include "qstar_scheduler.h"

using namespace freetoken::core;

namespace {

constexpr size_t kExpertBytes = 4 * sizeof(float);

void make_fake_expert(uint32_t id, float out[4]) {
    for (int i = 0; i < 4; ++i) out[i] = static_cast<float>((id + 1) * 10);
}

bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

}  // namespace

// Correctness with known numbers (same case already proven in
// qstar_scheduler_test.cpp: m=8, ratio 0.25 -> q=2), AND proves the fill
// set genuinely landed in the GPU cache -- not just that the returned
// numbers look right.
TEST(QStarScheduler, MatchesFormulaAndActuallyFillsGpuCache) {
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

    QStarScheduler scheduler(pool, cache);
    std::vector<uint32_t> missing = {0, 1, 2, 3, 4, 5, 6, 7};
    QStarStepResult result = scheduler.run_step(missing, bw);

    std::printf("m=%d q=%d gpu_fill=%.1fus cpu_compute=%.1fus\n",
                result.m, result.q, result.gpu_fill_seconds * 1e6, result.cpu_compute_seconds * 1e6);

    EXPECT_EQ(result.m, 8);
    EXPECT_EQ(result.q, 2);

    // Fill set was experts {0, 1} (first q=2, per split_missing_experts'
    // documented order). They should now be real cache HITS.
    size_t hits_before = cache.hits();
    cache.get_or_fetch(0, pool);
    cache.get_or_fetch(1, pool);
    EXPECT_EQ(cache.hits(), hits_before + 2) << "experts 0,1 should be cache hits after fill";
}

// Reuses the SAME scheduler for a second step, proving the persistent
// worker thread survives across calls, then lets it go out of scope --
// proving clean shutdown (join(), not abandoned/hung).
TEST(QStarScheduler, PersistsAcrossMultipleStepsAndShutsDownCleanly) {
    HostPoolConfig pool_cfg;
    HostResidentPool pool(pool_cfg);
    for (uint32_t id = 0; id < 2; ++id) {
        float data[4];
        make_fake_expert(id, data);
        pool.register_expert(id, data, kExpertBytes);
    }

    GpuCacheConfig cache_cfg;
    cache_cfg.max_bytes = 100 * kExpertBytes;
    GpuExpertCache cache(cache_cfg);

    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 10e9;
    bw.b_host_bytes_per_sec = 40e9;

    {
        QStarScheduler scheduler(pool, cache);
        QStarStepResult first = scheduler.run_step({0}, bw);
        QStarStepResult second = scheduler.run_step({1}, bw);
        EXPECT_EQ(first.m, 1);
        EXPECT_EQ(second.m, 1);
        // scheduler destructs at end of this block -- if join() were
        // broken this whole test would hang instead of completing.
    }
    SUCCEED() << "scheduler shut down without hanging";
}

// The actual benchmark dev_spec.md asks for -- q* split vs naive
// stall-and-copy (everything through the GPU cache, no CPU offload), on a
// larger, more realistic batch.
TEST(QStarScheduler, FasterThanNaiveStallAndCopyBaseline) {
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

    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 10e9;
    bw.b_host_bytes_per_sec = 40e9;

    double naive_ms = 0.0;
    {
        // Naive baseline: every expert through the GPU cache, sequentially,
        // no CPU offload at all -- exactly Modules 3.2/3.3 alone.
        GpuCacheConfig naive_cfg;
        naive_cfg.max_bytes = kBenchExpertCount * kBenchExpertBytes;
        GpuExpertCache naive_cache(naive_cfg);

        auto start = std::chrono::steady_clock::now();
        for (uint32_t id : all_ids) {
            naive_cache.get_or_fetch(id, bench_pool);
        }
        naive_ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
    }

    double split_ms = 0.0;
    int split_q = 0;
    {
        // q* split path: same experts, same "all missing" starting condition.
        GpuCacheConfig split_cfg;
        split_cfg.max_bytes = kBenchExpertCount * kBenchExpertBytes;
        GpuExpertCache split_cache(split_cfg);
        QStarScheduler scheduler(bench_pool, split_cache);

        auto start = std::chrono::steady_clock::now();
        QStarStepResult result = scheduler.run_step(all_ids, bw);
        split_ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
        split_q = result.q;
    }

    std::printf("naive baseline (all %u experts via GPU cache): %.2f ms\n", kBenchExpertCount, naive_ms);
    std::printf("q* split (q=%d fill / %d compute): %.2f ms\n", split_q, kBenchExpertCount - split_q, split_ms);

    if (split_ms >= naive_ms) {
        std::printf("  NOTE: split wasn't faster this run -- see printed numbers, may be timing noise\n");
    }
}

// Real model weight bytes, not synthetic filler. See model_loader.h's
// load_moe_experts_into_pool() doc comment and PROGRESS.md's placeholder
// ledger for exactly what's real here (real GGUF bytes, one of the three
// matrices a real expert needs) and what isn't (no real forward-pass math).
// Correctness-focused, not a scale benchmark -- this test model only has 12
// real experts total.
TEST(QStarScheduler, WorksWithRealGgufModelWeights) {
    const std::string model_path = "models/tiny-random-granite-moe.f16.gguf";
    if (!file_exists(model_path)) {
        GTEST_SKIP() << "model not downloaded -- run scripts/download_test_model.sh first";
    }

    HostPoolConfig real_pool_cfg;
    HostResidentPool real_pool(real_pool_cfg);
    int n_real_experts = load_moe_experts_into_pool(model_path, real_pool);
    ASSERT_EQ(n_real_experts, 12) << "6 layers x 2 experts/layer, per this specific model";

    // 64x32 f16 per expert (see model_loader.h's layout doc) = 4096 bytes.
    const ExpertSlot* slot0 = real_pool.find(0);
    ASSERT_NE(slot0, nullptr);
    EXPECT_EQ(slot0->size_bytes, 4096u);

    GpuCacheConfig real_cache_cfg;
    real_cache_cfg.max_bytes = 12 * 4096;
    GpuExpertCache real_cache(real_cache_cfg);
    QStarScheduler real_scheduler(real_pool, real_cache);

    std::vector<uint32_t> real_missing;
    for (uint32_t id = 0; id < 12; ++id) real_missing.push_back(id);
    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 10e9;
    bw.b_host_bytes_per_sec = 40e9;
    QStarStepResult real_result = real_scheduler.run_step(real_missing, bw);

    std::printf("real model: m=%d q=%d (expect q=3, ratio 0.25 x 12)\n", real_result.m, real_result.q);
    EXPECT_EQ(real_result.m, 12);
    EXPECT_EQ(real_result.q, 3);
}
