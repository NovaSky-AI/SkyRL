#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for the local router. No keys involved.
# Edit routes inside router_simple.py to point to your upstreams.

PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}

echo "Starting router on http://${HOST}:${PORT}/v1"
cd "$(dirname "$0")"
exec uvicorn router_simple:app --host "${HOST}" --port "${PORT}"

# curl -s http://127.0.0.1:8080/health | jq