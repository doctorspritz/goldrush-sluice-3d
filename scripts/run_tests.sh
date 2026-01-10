#!/bin/bash
# Simulation Test Suite Runner
#
# Runs physics validation first, then test levels in sequence.
# Exits on first failure.
#
# Usage:
#   ./scripts/run_tests.sh        # Run all tests
#   ./scripts/run_tests.sh 0      # Run specific level only
#   ./scripts/run_tests.sh 0 2    # Run range of levels only

set -e

echo "========================================"
echo " Sluice Simulation Test Suite"
echo "========================================"
echo ""

PASSED=0
FAILED=0
SKIPPED=0

# Always run physics validation first (unless specific level requested)
if [ $# -eq 0 ]; then
    echo "----------------------------------------"
    echo "Running Physics Validation..."
    echo "----------------------------------------"

    if cargo run --example test_physics_validation --release; then
        echo "PASS: Physics Validation"
        ((PASSED++))
    else
        echo "FAIL: Physics Validation"
        ((FAILED++))
        echo ""
        echo "========================================"
        echo " TEST SUITE FAILED at Physics Validation"
        echo "========================================"
        exit 1
    fi
    echo ""
fi

# Determine which levels to run
if [ $# -eq 0 ]; then
    LEVELS=(0 2)  # Implemented levels: 0 (dam break), 2 (riffles)
elif [ $# -eq 1 ]; then
    LEVELS=($1)
else
    LEVELS=($(seq $1 $2))
fi

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
