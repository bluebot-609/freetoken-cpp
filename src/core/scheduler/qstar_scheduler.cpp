#include "qstar_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "ggml-cpu.h"
#include "ggml.h"

namespace freetoken::core {

int compute_q_star(int m, const CalibrationResult& bandwidths) {
    if (m <= 0) {
        return 0;
    }
    if (bandwidths.b_host_bytes_per_sec <= 0.0) {
        // No usable CPU throughput measurement — can't run the CPU branch
        // at all, so everything has to go through the GPU-fill path.
        return m;
    }

    const double raw_q = m * (bandwidths.b_pcie_bytes_per_sec / bandwidths.b_host_bytes_per_sec);
    int q = static_cast<int>(std::lround(raw_q));

    q = std::max(q, 1);   // always at least one fill (paper §3.2)
    q = std::min(q, m);   // never more than m — nothing to over-fill
    return q;
}

QStarSplit split_missing_experts(const std::vector<uint32_t>& missing_experts, int q) {
    QStarSplit split;
    const size_t q_clamped = static_cast<size_t>(std::clamp(q, 0, static_cast<int>(missing_experts.size())));

    split.fill_set.assign(missing_experts.begin(), missing_experts.begin() + static_cast<long>(q_clamped));
    split.compute_set.assign(missing_experts.begin() + static_cast<long>(q_clamped), missing_experts.end());
    return split;
}

QStarScheduler::QStarScheduler(HostResidentPool& pool, GpuExpertCache& cache, SelectionPolicy policy)
    : pool_(pool), cache_(cache), policy_(policy), work_queue_(/*capacity=*/1024) {
    // Start the persistent worker now, once — matches the paper's own
    // description of its CPU workers as "a persistent C++ pool pinned to
    // physical cores," not something spun up fresh per decode step.
    worker_ = std::thread(&QStarScheduler::worker_loop, this);
}

QStarScheduler::~QStarScheduler() {
    // Tell the worker to stop, then actually WAIT for it (join), rather
    // than destroying this object while that thread might still be
    // reading pool_/cache_ — that would be a use-after-free race.
    stop_requested_.store(true, std::memory_order_relaxed);
    worker_.join();
}

void QStarScheduler::worker_loop() {
    while (!stop_requested_.load(std::memory_order_relaxed)) {
        uint32_t expert_id = 0;
        if (!work_queue_.pop(expert_id)) {
            continue;  // queue empty right now — spin, no work to do yet
        }

        // Placeholder "compute": same deliberate stand-in used since
        // model_loader.cpp (trivial ggml op) — we don't have a real MoE
        // expert kernel yet. What this proves is that computation happens
        // directly against the Host-Resident Pool's data, entirely on the
        // CPU, without ever touching the GPU cache — that's the actual
        // point of the compute-set branch (paper §3.2: "Experts in C
        // execute directly from the CPU-resident expert pool and leave
        // residency unchanged").
        const ExpertSlot* slot = pool_.find(expert_id);
        if (slot) {
            // THIRD time hitting this exact class of mistake (see
            // docs/citations.md: model_loader.cpp, calibration.cpp) --
            // ggml_new_graph() allocates a fixed-size default graph
            // regardless of how few nodes are used, and that fixed
            // overhead (tens of KB) swamps a small tensor's own size. A
            // size-relative formula like `2 * slot->size_bytes + 4096`
            // works for large experts but crashes on small ones. Always
            // use a large FLAT minimum, not something scaled to the data.
            const size_t mem_size = std::max<size_t>(2 * slot->size_bytes, 2 * 1024 * 1024) + 1024 * 1024;
            ggml_init_params params{
                /*.mem_size=*/ mem_size,
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc=*/ false,
            };
            ggml_context* ctx = ggml_init(params);
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                                 static_cast<int64_t>(slot->size_bytes / sizeof(float)));
            std::memcpy(t->data, slot->data, slot->size_bytes);
            ggml_tensor* out = ggml_scale(ctx, t, 1.0f);
            ggml_cgraph* graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            ggml_graph_compute_with_ctx(ctx, graph, /*n_threads=*/1);
            ggml_free(ctx);
        }

        completed_count_.fetch_add(1, std::memory_order_release);
    }
}

QStarStepResult QStarScheduler::run_step(const std::vector<uint32_t>& missing_experts,
                                          const CalibrationResult& bandwidths) {
    QStarStepResult result;
    result.m = static_cast<int>(missing_experts.size());
    result.q = compute_q_star(result.m, bandwidths);

    QStarSplit split = select_experts(policy_, missing_experts, result.q, pool_, history_);
    const int expected_completed = static_cast<int>(split.compute_set.size());

    // completed_count_ accumulates across every call to run_step() over
    // this scheduler's lifetime, not just this one step — so we snapshot
    // where it stood before dispatching, and wait for it to advance by
    // exactly this step's compute_set size, not for some absolute value.
    const int completed_before = completed_count_.load(std::memory_order_acquire);

    auto cpu_start = std::chrono::steady_clock::now();
    for (uint32_t expert_id : split.compute_set) {
        while (!work_queue_.push(expert_id)) {
            // queue momentarily full (default capacity 1024) — spin
            // rather than drop work.
        }
    }

    // GPU-fill runs on THIS (calling) thread. The worker thread is
    // concurrently draining the queue we just filled, in the background
    // — this loop and that worker are genuinely running at the same time,
    // not sequentially.
    auto gpu_start = std::chrono::steady_clock::now();
    for (uint32_t expert_id : split.fill_set) {
        cache_.get_or_fetch(expert_id, pool_);
    }
    auto gpu_end = std::chrono::steady_clock::now();
    result.gpu_fill_seconds = std::chrono::duration<double>(gpu_end - gpu_start).count();

    // Wait for the worker to actually FINISH this step's compute set —
    // not just "we handed it off." Same busy-wait pattern as the SPSC
    // queue's own tests: no locks, no condition variables.
    while (completed_count_.load(std::memory_order_acquire) < completed_before + expected_completed) {
        // spin
    }
    auto cpu_end = std::chrono::steady_clock::now();
    result.cpu_compute_seconds = std::chrono::duration<double>(cpu_end - cpu_start).count();

    return result;
}

}  // namespace freetoken::core
