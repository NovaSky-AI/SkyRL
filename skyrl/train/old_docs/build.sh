#!/bin/bash
set -e

# Ensure a valid UTF-8 locale to avoid "unsupported locale setting"
export LC_ALL=${LC_ALL:-C.UTF-8}
export LANG=${LANG:-C.UTF-8}
export LANGUAGE=${LANGUAGE:-C.UTF-8}

# Build and serve the documentation with live reload
# Usage: ./build.sh [--build-only]
#   --build-only: Build docs without starting the live server

cd "$(dirname "$0")"  # Ensure we're in the docs directory

if [ "$1" = "--build-only" ]; then
    uv run --extra docs --isolated sphinx-build -b html . _build/html $@
else
    uv run --extra docs --isolated sphinx-autobuild . _build/html $@
fi
