// Module 3.4 unit test: compute_q_star() against known values, per
// dev_spec.md Module 3.4's formula (paper §3.2 Eq. 4).
#include <cstdio>

#include "qstar_scheduler.h"

using freetoken::core::compute_q_star;
using freetoken::core::split_missing_experts;
using freetoken::core::CalibrationResult;
using freetoken::core::QStarSplit;

namespace {

bool check(int actual, int expected, const char* label) {
    if (actual != expected) {
        std::fprintf(stderr, "%s: expected %d, got %d\n", label, expected, actual);
        return false;
    }
    return true;
}

bool check_vec(const std::vector<uint32_t>& actual, const std::vector<uint32_t>& expected, const char* label) {
    if (actual != expected) {
        std::fprintf(stderr, "%s: sizes/contents differ (actual size %zu, expected size %zu)\n",
                     label, actual.size(), expected.size());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    bool ok = true;

    // Exact case: m=8, B_P/B_H = 10/40 = 0.25 -> q* = 8 * 0.25 = 2 exactly.
    {
        CalibrationResult bw;
        bw.b_pcie_bytes_per_sec = 10e9;
        bw.b_host_bytes_per_sec = 40e9;
        ok &= check(compute_q_star(8, bw), 2, "exact case");
    }

    // Clamp-to-1: raw q* would round to 0 (PCIe is tiny relative to CPU
    // throughput), but the paper requires at least one fill.
    {
        CalibrationResult bw;
        bw.b_pcie_bytes_per_sec = 1e9;
        bw.b_host_bytes_per_sec = 100e9;
        ok &= check(compute_q_star(5, bw), 1, "clamp-to-1 case");
    }

    // Clamp-to-m: raw q* would exceed m (PCIe is huge relative to CPU
    // throughput), but there are only m experts to fill.
    {
        CalibrationResult bw;
        bw.b_pcie_bytes_per_sec = 100e9;
        bw.b_host_bytes_per_sec = 1e9;
        ok &= check(compute_q_star(3, bw), 3, "clamp-to-m case");
    }

    // m=0: nothing missing, nothing to split.
    {
        CalibrationResult bw;
        bw.b_pcie_bytes_per_sec = 10e9;
        bw.b_host_bytes_per_sec = 10e9;
        ok &= check(compute_q_star(0, bw), 0, "m=0 case");
    }

    if (!ok) {
        std::printf("test_compute_q_star: FAIL\n");
        return 1;
    }
    std::printf("test_compute_q_star: PASS\n");

    bool split_ok = true;
    {
        std::vector<uint32_t> missing = {10, 20, 30, 40, 50};
        QStarSplit split = split_missing_experts(missing, 2);
        split_ok &= check_vec(split.fill_set, {10, 20}, "split fill_set");
        split_ok &= check_vec(split.compute_set, {30, 40, 50}, "split compute_set");
    }
    {
        // q clamped to the full list when it's larger than available experts
        // (shouldn't happen given compute_q_star()'s own clamp, but this
        // function takes q as a plain int — defend against a bad caller).
        std::vector<uint32_t> missing = {1, 2};
        QStarSplit split = split_missing_experts(missing, 5);
        split_ok &= check_vec(split.fill_set, {1, 2}, "split over-large q");
        split_ok &= check_vec(split.compute_set, {}, "split over-large q compute_set");
    }

    if (!split_ok) {
        std::printf("test_split_missing_experts: FAIL\n");
        return 1;
    }
    std::printf("test_split_missing_experts: PASS\n");
    return 0;
}
