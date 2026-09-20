#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "calibration.h"
#include "gpu_cache.h"
#include "host_pool.h"
#include "selection_policy.h"
#include "spsc_queue.h"

namespace freetoken::core {

/// Computes q* (dev_spec.md Module 3.4, paper §3.2 Eq. 4): of `m` cache-
/// missing experts this decode step, how many should go to the GPU-fill
/// path vs the CPU-direct-execute path.
///
///   q* ≈ m · (B_P / B_H)
///
/// Two edge cases the paper calls for explicitly, not left to chance:
///   - Always at least 1 fill when m >= 1 ("the cache continues warming
///     even when the CPU handles most misses" — paper §3.2).
///   - Never more than m (can't fill more experts than are actually missing).
int compute_q_star(int m, const CalibrationResult& bandwidths);

/// Partitions `missing_experts` into a fill set (first `q` — see
/// compute_q_star()) and a compute set (the rest). SIMPLIFICATION, stated
/// explicitly (CLAUDE.md principle #4): the paper delegates the choice of
/// *which* q experts become F to the cache replacement policy; we don't have
/// that decision logic yet, so this takes them in whatever order the caller
/// provides. Revisit once a real eviction/selection policy exists.
QStarSplit split_missing_experts(const std::vector<uint32_t>& missing_experts, int q);

/// What one call to run_step() measured — the raw material for benchmarking
/// against the Modules 3.2/3.3 stall-and-copy baseline (dev_spec.md's
/// "benchmark immediately" instruction).
struct QStarStepResult {
    int m = 0;  // total missing experts this step
    int q = 0;  // how many went to the GPU-fill path
    double gpu_fill_seconds = 0.0;     // wall-clock time of the fill-set loop
    double cpu_compute_seconds = 0.0;  // wall-clock time from dispatch to worker completion
};

/// Orchestrates one decode step's q* split: dispatches the compute set to a
/// persistent CPU worker thread (via the SPSC queue, Module 3.8) while the
/// calling thread handles the fill set through the GPU cache (Module 3.3) —
/// both running concurrently, not sequentially. dev_spec.md Module 3.4.
///
/// SCOPING NOTE: dev_spec.md §3.4 lists "CPU thread count" as an exposed
/// parameter, implying possibly multiple CPU workers. This baseline uses
/// exactly ONE persistent worker thread, because our SPSC queue (Module
/// 3.8) is strictly single-producer/single-consumer by design — that's the
/// primitive we built. Multiple workers would need either multiple queues
/// or a different (MPMC) queue; a real future enhancement, not faked here.
class QStarScheduler {
public:
    // `policy` defaults to FirstInOrder -- today's existing behavior --
    // so existing callers are unaffected unless they explicitly ask for
    // something else.
    QStarScheduler(HostResidentPool& pool, GpuExpertCache& cache,
                    SelectionPolicy policy = SelectionPolicy::FirstInOrder);
    ~QStarScheduler();

    QStarScheduler(const QStarScheduler&) = delete;
    QStarScheduler& operator=(const QStarScheduler&) = delete;

    QStarStepResult run_step(const std::vector<uint32_t>& missing_experts, const CalibrationResult& bandwidths);

private:
    void worker_loop();

    HostResidentPool& pool_;
    GpuExpertCache& cache_;
    SelectionPolicy policy_;
    SelectionHistory history_;

    SpscQueue<uint32_t> work_queue_;
    std::thread worker_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<int> completed_count_{0};
};

}  // namespace freetoken::core
