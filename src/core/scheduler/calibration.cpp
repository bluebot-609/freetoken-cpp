#include "calibration.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "ggml-cpu.h"
#include "ggml.h"

namespace freetoken::core {

namespace {

// Measures B_P: real host-to-device cudaMemcpy bandwidth, averaged over
// several iterations to smooth out one-off scheduling noise (a single
// timed copy can be thrown off by, say, the OS scheduling something else
// on this core for a millisecond).
double measure_pcie_bandwidth(const CalibrationConfig& config) {
    std::vector<char> host_buffer(config.sample_bytes, 0);

    void* device_buffer = nullptr;
    cudaError_t err = cudaMalloc(&device_buffer, config.sample_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "calibrate: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 0.0;
    }

    double total_seconds = 0.0;
    for (int i = 0; i < config.iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        cudaMemcpy(device_buffer, host_buffer.data(), config.sample_bytes, cudaMemcpyHostToDevice);
        total_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    cudaFree(device_buffer);

    const double avg_seconds = total_seconds / config.iterations;
    return static_cast<double>(config.sample_bytes) / avg_seconds;
}

// Measures B_H: the effective bandwidth of a real ggml CPU compute op run
// over `config.sample_bytes` worth of data — this is compute-bound, not
// just a memcpy, matching the paper's own definition ("effective bandwidth
// of the CPU-side MoE expert kernel"). We don't have a real MoE expert
// kernel yet (that's Module 3.4's CPU branch), so a scale op stands in as
// a representative CPU-bound tensor op — same deliberate-placeholder
// pattern used in model_loader.cpp's forward-pass smoke test.
double measure_cpu_bandwidth(const CalibrationConfig& config) {
    const size_t n_elements = config.sample_bytes / sizeof(float);

    // ggml_scale() below allocates a SECOND tensor (its output), same size
    // as the input -- plus the graph structure itself needs headroom. Give
    // it a generous flat margin rather than computing the exact minimum
    // (got this wrong once already trying to be precise; see
    // docs/citations.md).
    ggml_init_params params{
        /*.mem_size=*/ 2 * n_elements * sizeof(float) + 4 * 1024 * 1024,
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc=*/ false,
    };
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(n_elements));
    ggml_tensor* out = ggml_scale(ctx, t, 1.0f);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    const int n_threads = static_cast<int>(std::thread::hardware_concurrency());

    double total_seconds = 0.0;
    for (int i = 0; i < config.iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        ggml_graph_compute_with_ctx(ctx, graph, n_threads);
        total_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    ggml_free(ctx);

    const double avg_seconds = total_seconds / config.iterations;
    return static_cast<double>(config.sample_bytes) / avg_seconds;
}

}  // namespace

CalibrationResult calibrate(const CalibrationConfig& config) {
    CalibrationResult result;
    result.b_pcie_bytes_per_sec = measure_pcie_bandwidth(config);
    result.b_host_bytes_per_sec = measure_cpu_bandwidth(config);
    return result;
}

}  // namespace freetoken::core
