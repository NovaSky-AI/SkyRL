#!/usr/bin/env bash
# Harbor + inference-capture, with capture resolved from a local checkout.
#
# Capture needs no database standing by: with no DATABASE_URL it starts its own
# PostgreSQL under ICAP_DATA_DIR and keeps the queue and payloads beside it.
set -euo pipefail

ICAP_PATH="${ICAP_PATH:-$HOME/dev/nscale-endpoints/anyscale-capture}"
export ICAP_DATA_DIR="${ICAP_DATA_DIR:-$PWD/icap-data}"
# Share one PostgreSQL download across runs that use a fresh data directory.
export ICAP_POSTGRES_CACHE="${ICAP_POSTGRES_CACHE:-$HOME/.cache/icap-postgres}"

if [[ ! -d "$ICAP_PATH" ]]; then
  echo "set ICAP_PATH to an anyscale-capture checkout (got: $ICAP_PATH)" >&2
  exit 1
fi

exec uv run --with-editable "$ICAP_PATH" \
  -m examples.train_integrations.harbor.icap.entrypoints.main_harbor_icap "$@"
