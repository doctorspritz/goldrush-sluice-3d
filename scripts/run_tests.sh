#!/bin/bash
# Simulation Test Suite Runner
#
# Runs all test levels in sequence. Exits on first failure.
#
# Usage:
#   ./scripts/run_tests.sh        # Run all levels
#   ./scripts/run_tests.sh 0      # Run specific level
#   ./scripts/run_tests.sh 0 2    # Run range of levels

set -e

echo "========================================"
echo " Sluice Simulation Test Suite"
echo "========================================"
echo ""

# Determine which levels to run
if [ $# -eq 0 ]; then
    LEVELS=(0)  # For now, only level 0 is implemented
elif [ $# -eq 1 ]; then
    LEVELS=($1)
else
    LEVELS=($(seq $1 $2))
fi

PASSED=0
FAILED=0
SKIPPED=0

for level in "${LEVELS[@]}"; do
    EXAMPLE="test_level_$level"

    echo "----------------------------------------"
    echo "Running Level $level..."
    echo "----------------------------------------"

    # Check if example exists
    if ! cargo build --example "$EXAMPLE" --release 2>/dev/null; then
        echo "SKIP: $EXAMPLE not found or failed to build"
        ((SKIPPED++))
        continue
    fi

    # Run the test
    if cargo run --example "$EXAMPLE" --release; then
        echo "PASS: Level $level"
        ((PASSED++))
    else
        echo "FAIL: Level $level"
        ((FAILED++))
        echo ""
        echo "========================================"
        echo " TEST SUITE FAILED at Level $level"
        echo "========================================"
        exit 1
    fi

    echo ""
done

echo "========================================"
echo " TEST SUITE COMPLETE"
echo "========================================"
echo ""
echo "Results:"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo ""

if [ $FAILED -gt 0 ]; then
    exit 1
fi
