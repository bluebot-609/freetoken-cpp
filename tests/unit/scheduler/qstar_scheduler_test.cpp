// Module 3.4 unit tests: compute_q_star() and split_missing_experts()
// against known values, per dev_spec.md Module 3.4's formula (paper §3.2
// Eq. 4). Pure logic, no CUDA/threads -- see tests/integration/scheduler
// for the end-to-end/concurrent/real-model versions.
#include <gtest/gtest.h>

#include "qstar_scheduler.h"

using freetoken::core::compute_q_star;
using freetoken::core::split_missing_experts;
using freetoken::core::CalibrationResult;
using freetoken::core::QStarSplit;

TEST(ComputeQStar, ExactCase) {
    // m=8, B_P/B_H = 10/40 = 0.25 -> q* = 8 * 0.25 = 2 exactly.
    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 10e9;
    bw.b_host_bytes_per_sec = 40e9;
    EXPECT_EQ(compute_q_star(8, bw), 2);
}

TEST(ComputeQStar, ClampsToOneWhenRawResultWouldRoundToZero) {
    // Raw q* would round to 0 (PCIe is tiny relative to CPU throughput),
    // but the paper requires at least one fill.
    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 1e9;
    bw.b_host_bytes_per_sec = 100e9;
    EXPECT_EQ(compute_q_star(5, bw), 1);
}

TEST(ComputeQStar, ClampsToMWhenRawResultWouldExceedIt) {
    // Raw q* would exceed m (PCIe is huge relative to CPU throughput), but
    // there are only m experts to fill.
    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 100e9;
    bw.b_host_bytes_per_sec = 1e9;
    EXPECT_EQ(compute_q_star(3, bw), 3);
}

TEST(ComputeQStar, ZeroMissingExpertsGivesZero) {
    CalibrationResult bw;
    bw.b_pcie_bytes_per_sec = 10e9;
    bw.b_host_bytes_per_sec = 10e9;
    EXPECT_EQ(compute_q_star(0, bw), 0);
}

TEST(SplitMissingExperts, TakesFirstQAsFillSetRestAsComputeSet) {
    std::vector<uint32_t> missing = {10, 20, 30, 40, 50};
    QStarSplit split = split_missing_experts(missing, 2);
    EXPECT_EQ(split.fill_set, (std::vector<uint32_t>{10, 20}));
    EXPECT_EQ(split.compute_set, (std::vector<uint32_t>{30, 40, 50}));
}

TEST(SplitMissingExperts, ClampsQLargerThanTheListItself) {
    // q clamped to the full list when it's larger than available experts
    // (shouldn't happen given compute_q_star()'s own clamp, but this
    // function takes q as a plain int -- defend against a bad caller).
    std::vector<uint32_t> missing = {1, 2};
    QStarSplit split = split_missing_experts(missing, 5);
    EXPECT_EQ(split.fill_set, (std::vector<uint32_t>{1, 2}));
    EXPECT_TRUE(split.compute_set.empty());
}
