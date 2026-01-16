#!/bin/bash
# Build extra test fixtures using Docker
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET="${1:-all}"

echo "=== Building Extra 68k Test Fixtures ==="
echo "Target: $TARGET"

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    exit 1
fi

IMAGE_NAME="rusashi-m68k-builder"

# Build Docker image if needed
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Building Docker image..."
    docker build --platform linux/amd64 -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile.m68k" "$SCRIPT_DIR"
fi

echo "Running build in Docker..."
docker run --rm --platform linux/amd64 \
    -v "$FIXTURES_ROOT:/work" \
    "$IMAGE_NAME" \
    bash -c "cd /work/extra && make $TARGET"

echo ""
echo "=== Build Complete ==="
echo "68040 Binaries:"
ls -la "$SCRIPT_DIR/m68040/bin/"*.bin 2>/dev/null | wc -l
echo "fixtures built"
echo ""
echo "68020 Binaries:"
ls -la "$SCRIPT_DIR/m68020/bin/"*.bin 2>/dev/null | wc -l
echo "fixtures built"
echo ""
M68030_COUNT=$(find "$SCRIPT_DIR/m68030/bin" -name "*.bin" -type f 2>/dev/null | wc -l | tr -d ' ')
M68010_COUNT=$(find "$SCRIPT_DIR/m68010/bin" -name "*.bin" -type f 2>/dev/null | wc -l | tr -d ' ')
PRIVILEGE_COUNT=$(find "$SCRIPT_DIR/privilege/bin" -name "*.bin" -type f 2>/dev/null | wc -l | tr -d ' ')
EXCEPTION_COUNT=$(find "$SCRIPT_DIR/exceptions/bin" -name "*.bin" -type f 2>/dev/null | wc -l | tr -d ' ')

echo "68030 Binaries:"
printf "%8d fixtures built\n" "$M68030_COUNT"

echo ""
echo "68010 Binaries:"
printf "%8d fixtures built\n" "$M68010_COUNT"

echo ""
echo "Privilege Binaries:"
printf "%8d fixtures built\n" "$PRIVILEGE_COUNT"

echo ""
echo "Exception Binaries:"
printf "%8d fixtures built\n" "$EXCEPTION_COUNT"
