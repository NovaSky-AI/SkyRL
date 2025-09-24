#!/bin/bash
set -e

# Ensure a valid UTF-8 locale to avoid "unsupported locale setting"
# Use en_US.UTF-8 on macOS, C.UTF-8 elsewhere
if [[ "$OSTYPE" == "darwin"* ]]; then
    export LC_ALL=${LC_ALL:-en_US.UTF-8}
    export LANG=${LANG:-en_US.UTF-8}
    export LANGUAGE=${LANGUAGE:-en_US.UTF-8}
else
    export LC_ALL=${LC_ALL:-C.UTF-8}
    export LANG=${LANG:-C.UTF-8}
    export LANGUAGE=${LANGUAGE:-C.UTF-8}
fi

# Build and serve the documentation with live reload
# Usage: ./build.sh [--build-only]
#   --build-only: Build docs without starting the live server

cd "$(dirname "$0")"  # Ensure we're in the docs directory

# Simple flag handling - if more flags are added, consider using case statement or getopts
BUILD_ONLY=false
ARGS=()

# Parse arguments to extract --build-only flag
for arg in "$@"; do
    if [ "$arg" = "--build-only" ]; then
        BUILD_ONLY=true
    else
        ARGS+=("$arg")
    fi
done

if [ "$BUILD_ONLY" = true ]; then
    CMD_AND_ARGS=("sphinx-build" "-b" "html")
else
    CMD_AND_ARGS=("sphinx-autobuild")
fi

uv run --extra docs --extra cpu --isolated "${CMD_AND_ARGS[@]}" . _build/html "${ARGS[@]}"
