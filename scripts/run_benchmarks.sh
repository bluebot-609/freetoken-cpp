#!/usr/bin/env bash
# Builds and runs every test/benchmark binary in this repo, captures a
# full report (with real numbers, not just pass/fail) to
# benchmark/reports/, and prints a compact summary to stdout.
#
# dev_spec.md Section 1 names this file and benchmark/reports/ as planned
# repo structure -- this is that script, actually built.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="build"
REPORT_DIR="benchmark/reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="$REPORT_DIR/run_${TIMESTAMP}.txt"

echo "Building..."
if ! cmake --build "$BUILD_DIR" >>"$REPORT_FILE" 2>&1; then
    echo "BUILD FAILED -- see $REPORT_FILE"
    exit 1
fi

declare -a SUMMARY=()
FAIL_COUNT=0

# run_test NAME BINARY [ARGS...]
run_test() {
    local name="$1"; shift
    local binary="$1"; shift

    {
        echo ""
        echo "==================== $name ===================="
    } >>"$REPORT_FILE"

    if [ ! -x "$binary" ]; then
        echo "SKIPPED (not built: $binary)" >>"$REPORT_FILE"
        SUMMARY+=("$name: SKIPPED (not built)")
        return
    fi

    local output
    if output="$("$binary" "$@" 2>&1)"; then
        local status="PASS"
    else
        local status="FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo "$output" >>"$REPORT_FILE"

    # Pull out whatever the test itself already prints as its key metrics --
    # no new instrumentation needed, these lines already exist in each
    # test's own output.
    local metrics
    metrics="$(echo "$output" | grep -E 'PASS|FAIL|GB/s|ms$|hits,.*misses' | tr '\n' ' | ')"
    SUMMARY+=("$name [$status]: $metrics")
}

echo "Running tests..."
run_test "SPSC queue (unit)" "$BUILD_DIR/tests/unit/spsc_queue_test"
run_test "q* scheduler math (unit)" "$BUILD_DIR/tests/unit/qstar_scheduler_test"
run_test "Host pool + GPU cache (integration)" "$BUILD_DIR/tests/integration/memory_smoke"
run_test "Hardware calibration (integration)" "$BUILD_DIR/tests/integration/calibration_smoke"

if [ -f "models/tiny-random-granite-moe.f16.gguf" ]; then
    run_test "q* scheduler + real model (integration)" \
        "$BUILD_DIR/tests/integration/qstar_scheduler_smoke" \
        "models/tiny-random-granite-moe.f16.gguf"
else
    echo "q* scheduler + real model: SKIPPED (run scripts/download_test_model.sh first)"
    SUMMARY+=("q* scheduler + real model: SKIPPED (model not downloaded -- run scripts/download_test_model.sh)")
fi

echo ""
echo "==================== SUMMARY ===================="
for line in "${SUMMARY[@]}"; do
    echo "$line"
done
echo "==================================================="
echo "Full report: $REPORT_FILE"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "$FAIL_COUNT test(s) FAILED"
    exit 1
fi
echo "All tests passed."
