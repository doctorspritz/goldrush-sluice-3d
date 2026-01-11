#!/bin/bash
# DEM Example Test Runner
#
# Runs all DEM physics example tests and reports results.
# Examples exit with 0=pass, 1=fail.
#
# Usage:
#   ./scripts/run_example_tests.sh

set -o pipefail

# DEM test examples to run
EXAMPLES=(
    "test_dem_physics"
    "test_dem_stability"
    "test_dem_rolling"
    "test_dem_water_coupling"
)

PASSED=0
FAILED=0
TOTAL=${#EXAMPLES[@]}

echo "Running DEM example tests..."

for example in "${EXAMPLES[@]}"; do
    # Print test name with padding for alignment
    printf "  %-25s ... " "$example"

    # Run the example and capture exit code
    if cargo run --example "$example" --release >/dev/null 2>&1; then
        echo "PASS"
        ((PASSED++))
    else
        echo "FAIL"
        ((FAILED++))
    fi
done

echo ""
echo "$PASSED/$TOTAL tests passed"

# Exit with non-zero if any test failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
