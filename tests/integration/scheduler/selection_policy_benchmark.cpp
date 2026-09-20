// Compares all three SelectionPolicy choices against the SAME synthetic
// multi-step access trace, per the author's request: "have multiple
// policies... benchmark multiple at the same time."
//
// WHY A MULTI-STEP TRACE, NOT A SINGLE SHOT: every other benchmark in this
// codebase (qstar_scheduler_smoke.cpp) uses an all-distinct, single-shot
// set of missing experts -- every expert is a first-time miss, nothing
// repeats. Selection-policy choice literally cannot matter there: whichever
// q you pick to fill, none of them were ever going to be cache hits anyway,
// since nothing repeats. To see a real difference between policies, there
// has to be REUSE for a smarter policy to exploit -- so this trace has a
// skewed hot/cold access pattern (20 "hot" experts get most of the traffic,
// 80 "cold" experts get the rest) across many steps, matching the paper's
// own observation that MoE decode routing has real temporal locality.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "qstar_scheduler.h"

using namespace freetoken::core;

namespace {

constexpr uint32_t kNumExperts = 100;
constexpr uint32_t kNumHotExperts = 20;
constexpr size_t kBaseExpertBytes = 64 * 1024;
constexpr int kNumSteps = 80;
constexpr int kRequestsPerStep = 8;
// Cache sized off the AVERAGE expert size below (2.5x base) -- real
// contention either way, but varied sizes are what actually give
// SmallestFirst something to differentiate; uniform sizes make it
// degenerate to the same order as FirstInOrder (verified: it did, first
// version of this benchmark had that exact bug).
constexpr size_t kCacheBytes = static_cast<size_t>((kNumHotExperts / 2) * kBaseExpertBytes * 2.5);

// Deliberately varied per-expert size -- 1x, 2x, 3x, 4x the base, cycling
// by expert_id -- so SmallestFirst has a real signal to sort on.
size_t expert_size(uint32_t id) {
    return kBaseExpertBytes * (1 + (id % 4));
}

// Fixed seed -- reproducible across runs, so a result difference between
// policies is real, not RNG noise from run to run.
std::vector<std::vector<uint32_t>> generate_trace() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<uint32_t> hot_pick(0, kNumHotExperts - 1);
    std::uniform_int_distribution<uint32_t> cold_pick(kNumHotExperts, kNumExperts - 1);

    std::vector<std::vector<uint32_t>> trace;
    for (int step = 0; step < kNumSteps; ++step) {
        std::set<uint32_t> requested;  // de-duplicated -- one step shouldn't ask for the same expert twice
        while (requested.size() < kRequestsPerStep) {
            requested.insert(coin(rng) < 0.8 ? hot_pick(rng) : cold_pick(rng));
        }
        trace.emplace_back(requested.begin(), requested.end());
    }
    return trace;
}

struct TraceResult {
    size_t total_requests = 0;
    size_t trace_hits = 0;    // already resident when requested -- no work needed
    size_t trace_misses = 0;  // had to go through the scheduler
    double total_seconds = 0.0;
};

TraceResult run_trace(SelectionPolicy policy, HostResidentPool& pool,
                       const std::vector<std::vector<uint32_t>>& trace) {
    GpuCacheConfig cache_cfg;
    cache_cfg.max_bytes = kCacheBytes;
    GpuExpertCache cache(cache_cfg);
    QStarScheduler scheduler(pool, cache, policy);

    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 10e9;
    bw.b_host_bytes_per_sec = 40e9;

    TraceResult result;
    auto start = std::chrono::steady_clock::now();
    for (const auto& requested : trace) {
        std::vector<uint32_t> missing;
        for (uint32_t id : requested) {
            ++result.total_requests;
            if (cache.contains(id)) {
                ++result.trace_hits;
            } else {
                ++result.trace_misses;
                missing.push_back(id);
            }
        }
        if (!missing.empty()) {
            scheduler.run_step(missing, bw);
        }
    }
    result.total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

}  // namespace

TEST(SelectionPolicyBenchmark, ComparesAllThreePoliciesOnTheSameTrace) {
    HostPoolConfig pool_cfg;
    HostResidentPool pool(pool_cfg);
    std::vector<char> filler(kBaseExpertBytes * 4, 0);  // big enough for the largest size class
    for (uint32_t id = 0; id < kNumExperts; ++id) {
        pool.register_expert(id, filler.data(), expert_size(id));
    }

    const auto trace = generate_trace();

    std::printf("\nselection policy comparison (%d steps, %d requests/step, %u/%u hot experts cached):\n",
                kNumSteps, kRequestsPerStep, kNumHotExperts / 2, kNumHotExperts);

    for (auto [name, policy] : {
             std::pair{"FirstInOrder", SelectionPolicy::FirstInOrder},
             std::pair{"MostRecentlyRequested", SelectionPolicy::MostRecentlyRequested},
             std::pair{"SmallestFirst", SelectionPolicy::SmallestFirst},
         }) {
        TraceResult r = run_trace(policy, pool, trace);
        double hit_rate = 100.0 * static_cast<double>(r.trace_hits) / static_cast<double>(r.total_requests);

        std::printf("  %-22s hit_rate=%5.1f%% (%zu/%zu)  total=%.2fms\n",
                    name, hit_rate, r.trace_hits, r.total_requests, r.total_seconds * 1e3);

        // Sanity, not a "which policy wins" assertion -- we're MEASURING
        // that, not assuming an answer.
        EXPECT_EQ(r.trace_hits + r.trace_misses, r.total_requests);
        EXPECT_GE(hit_rate, 0.0);
        EXPECT_LE(hit_rate, 100.0);
    }
}
