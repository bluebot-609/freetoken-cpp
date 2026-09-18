#pragma once

#include <cstddef>

namespace freetoken::core {

// Exposed parameters for the Hardware Calibration Pass (dev_spec.md Module
// 3.6). Both knobs trade calibration accuracy against startup latency —
// dev_spec.md explicitly says expose that tradeoff, don't hide it.
struct CalibrationConfig {
    size_t sample_bytes = 16 * 1024 * 1024;  // size of the buffer timed for each measurement
    int    iterations   = 5;                  // repeat and average, to smooth out one-off noise
};

// The two measured bandwidths Module 3.4's q* split needs (bytes/second).
// Named to match the paper's own notation (§3.2) so the two stay easy to
// cross-reference against the formula, not because a different name
// wouldn't do — dev_spec.md's own citation discipline wants this legible
// against the source.
struct CalibrationResult {
    double b_pcie_bytes_per_sec = 0.0;  // B_P — measured host-to-device cudaMemcpy bandwidth
    double b_host_bytes_per_sec = 0.0;  // B_H — measured CPU expert-kernel bandwidth
};

// Runs both measurements on the current machine and returns the result.
// Takes a few tens of milliseconds to a few seconds depending on
// `config.sample_bytes`/`iterations` — dev_spec.md's exposed startup-cost
// tradeoff.
CalibrationResult calibrate(const CalibrationConfig& config);

}  // namespace freetoken::core
