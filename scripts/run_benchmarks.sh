#!/usr/bin/env bash
# Builds and runs every test via ctest, captures the full verbose output
# (including each test's own printed numbers -- bandwidths, timings,
# speedups) to benchmark/reports/, and prints a compact summary.
#
# dev_spec.md Section 1 names this file and benchmark/reports/ as planned
# repo structure -- this is that script.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="build"
REPORT_DIR="benchmark/reports"
mkdir -p "$REPORT_DIR"
REPORT_FILE="$REPORT_DIR/run_$(date +%Y%m%d_%H%M%S).txt"

echo "Building..."
if ! cmake --build "$BUILD_DIR" >>"$REPORT_FILE" 2>&1; then
    echo "BUILD FAILED -- see $REPORT_FILE"
    exit 1
fi

echo "Running tests (ctest)..."
# -V: verbose, keeps each test's printed numbers in the log, not just PASS/FAIL.
ctest --test-dir "$BUILD_DIR" -V >>"$REPORT_FILE" 2>&1
CTEST_EXIT=$?

echo ""
echo "==================== SUMMARY ===================="
# ctest's own final table (per-test PASS/FAIL + timing) plus every printed
# number line from each test -- pulled from the report we just wrote, no
# separate instrumentation needed.
grep -E '^\s*[0-9]+/[0-9]+ Test|tests passed|tests failed|GB/s|ms$|hits,|misses|q=[0-9]' "$REPORT_FILE"
echo "==================================================="
echo "Full report: $REPORT_FILE"

exit $CTEST_EXIT
