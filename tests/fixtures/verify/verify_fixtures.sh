#!/bin/bash
# Verify coverage fixtures against Musashi reference emulator
# Usage: ./verify_fixtures.sh [fixture.bin ...]
#        ./verify_fixtures.sh --all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="musashi-verify"

# Build the Docker image if not present
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Building Musashi verifier image..."
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile.musashi" "$FIXTURES_DIR"
fi

run_verify() {
    local fixture="$1"
    local name=$(basename "$fixture" .bin)
    local output=$(docker run --rm -v "$FIXTURES_DIR:/work" "$IMAGE_NAME" "/work/$fixture" 2>/dev/null)
    local result=$(echo "$output" | grep "passes=" || echo "NO OUTPUT")
    echo "$name: $result"
}

if [ "$1" = "--all" ]; then
    echo "=== Verifying all coverage fixtures against Musashi ==="
    for f in "$FIXTURES_DIR/extra/coverage/bin/"*.bin; do
        run_verify "extra/coverage/bin/$(basename $f)"
    done
elif [ $# -eq 0 ]; then
    echo "Usage: $0 [--all | fixture.bin ...]"
    echo ""
    echo "Examples:"
    echo "  $0 --all                           # Verify all coverage fixtures"
    echo "  $0 extra/coverage/bin/scc_all.bin  # Verify specific fixture"
    exit 1
else
    for f in "$@"; do
        run_verify "$f"
    done
fi
