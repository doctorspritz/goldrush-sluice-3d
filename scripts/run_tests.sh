#!/bin/bash
# Canonical Example Smoke Builder
#
# Builds the two canonical examples (washplant_editor, world_sim).
# Other examples have been archived under crates/game/examples/archive.
#
# Usage:
#   ./scripts/run_tests.sh

set -e

echo "========================================"
echo " Canonical Example Smoke Build"
echo "========================================"
echo ""

EXAMPLES=(washplant_editor world_sim)

for example in "${EXAMPLES[@]}"; do
    echo "----------------------------------------"
    echo "Building $example..."
    echo "----------------------------------------"
    cargo build -p game --example "$example" --release
    echo ""
done

echo "========================================"
echo " BUILD COMPLETE"
echo "========================================"
echo ""
echo "To run:"
echo "  cargo run -p game --example washplant_editor --release"
echo "  cargo run -p game --example world_sim --release"
