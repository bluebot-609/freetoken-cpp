// Module 3.6 test: run the calibration pass on this machine and print what
// it measures, so the numbers can be eyeballed against known hardware
// specs (dev_spec.md Section 5's own testing philosophy — trust a
// measurement more once you've watched it produce a sane number).
#include <gtest/gtest.h>

#include "calibration.h"

using freetoken::core::calibrate;
using freetoken::core::CalibrationConfig;

TEST(HardwareCalibration, MeasuresPositiveBandwidths) {
    CalibrationConfig config;
    auto result = calibrate(config);

    std::printf("B_P (PCIe H2D): %.2f GB/s\n", result.b_pcie_bytes_per_sec / 1e9);
    std::printf("B_H (CPU expert kernel): %.2f GB/s\n", result.b_host_bytes_per_sec / 1e9);

    // A real measurement, not a stub -- both should come back as sane
    // positive numbers, not 0 (which would mean the measurement silently
    // failed) or negative (nonsensical for a bandwidth).
    EXPECT_GT(result.b_pcie_bytes_per_sec, 0.0);
    EXPECT_GT(result.b_host_bytes_per_sec, 0.0);
}
