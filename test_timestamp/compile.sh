#!/bin/bash
set -euo pipefail

# Build helper for read_timestamp
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

make -s clean
make -s all

echo "Build complete: $SCRIPT_DIR/read_timestamp"


