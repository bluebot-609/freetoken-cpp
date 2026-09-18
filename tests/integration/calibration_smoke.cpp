// Module 3.6 smoke test: run the calibration pass on this machine and
// print what it measures, so the numbers can be eyeballed against known
// hardware specs (dev_spec.md Section 5's own testing philosophy — trust a
// measurement more once you've watched it produce a sane number).
#include <cstdio>

#include "calibration.h"

using freetoken::core::calibrate;
using freetoken::core::CalibrationConfig;

int main() {
    CalibrationConfig config;
    auto result = calibrate(config);

    std::printf("B_P (PCIe H2D): %.2f GB/s\n", result.b_pcie_bytes_per_sec / 1e9);
    std::printf("B_H (CPU expert kernel): %.2f GB/s\n", result.b_host_bytes_per_sec / 1e9);
    return 0;
}
