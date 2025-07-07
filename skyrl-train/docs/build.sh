#!/bin/bash
set -e

# Build and serve the documentation with live reload
# Usage: ./build.sh [--build-only]
#   --build-only: Build docs without starting the live server

cd "$(dirname "$0")"  # Ensure we're in the docs directory

if [ "$1" = "--build-only" ]; then
    uv run --extra docs --isolated sphinx-build -b html . _build/html
else
    uv run --extra docs --isolated sphinx-autobuild . _build/html --port 8001
fi