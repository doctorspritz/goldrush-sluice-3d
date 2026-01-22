#!/bin/bash
# Canonical Example Smoke Builder (alias)
#
# The DEM example tests are archived with other examples under
# crates/game/examples/archive.
#
# Usage:
#   ./scripts/run_example_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_tests.sh"
