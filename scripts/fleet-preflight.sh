#!/usr/bin/env bash
# Pre-launch validation for SkyRL-Fleet jobs.
#
# Catches missing/invalid credentials and broken cloud auth *before*
# `sky launch` provisions GPU nodes — provisioning is slow and expensive,
# and the existing per-job env-var checks only fire inside the remote
# `setup:` block, after a VM is already up. This script is meant to run
# locally on the launching machine.
#
# Checks performed (in order):
#   1. Required env vars are set and non-empty:
#        FLEET_API_KEY, WANDB_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#      Plus any extras passed via --require <NAME>.
#   2. `gcloud` CLI is installed, has an active account, and has
#      application-default credentials configured (SkyPilot uses ADC for GCP).
#   3. `sky` CLI is installed and `sky check` reports at least one cloud
#      enabled.
#   4. Live API probes (via scripts/_fleet_preflight_probes.py): AWS STS,
#      Fleet API, W&B viewer. Probes degrade to "skip" if the relevant
#      Python SDK is not installed locally — presence is still required.
#
# Usage:
#   bash scripts/fleet-preflight.sh
#   bash scripts/fleet-preflight.sh --require OPENROUTER_API_KEY --require TINKER_API_KEY
#   bash scripts/fleet-preflight.sh --skip-gcloud      # for non-GCP launches
#   bash scripts/fleet-preflight.sh --skip-liveness    # presence checks only
#
# Exit codes:
#   0 — all enabled checks passed.
#   1 — at least one check failed; the offending check prints a remediation hint.
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

# --- Argument parsing ---
EXTRA_REQUIRED=()
SKIP_GCLOUD=false
SKIP_SKY=false
SKIP_LIVENESS=false
SKIP_AWS=false
SKIP_WANDB=false
SKIP_FLEET=false

# `require_value` errors out cleanly when a flag that takes a value is
# called at the end of argv (or with another flag as its value). Without
# this guard, `set -u` would surface the failure as the cryptic
# "$2: unbound variable", which has bitten callers passing args incorrectly.
require_value() {
  local flag="$1" value="${2-}"
  if [ -z "$value" ] || [[ "$value" == --* ]]; then
    echo "ERROR: $flag requires a value (got: '${value:-<missing>}')" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --require)        require_value "$1" "${2-}"; EXTRA_REQUIRED+=("$2"); shift 2 ;;
    --skip-gcloud)    SKIP_GCLOUD=true;       shift ;;
    --skip-sky)       SKIP_SKY=true;          shift ;;
    --skip-liveness)  SKIP_LIVENESS=true;     shift ;;
    --skip-aws)       SKIP_AWS=true;          shift ;;
    --skip-wandb)     SKIP_WANDB=true;        shift ;;
    --skip-fleet)     SKIP_FLEET=true;        shift ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "ERROR: Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# --- Output helpers (color only on TTY) ---
if [ -t 1 ]; then
  C_OK=$'\033[32m'; C_FAIL=$'\033[31m'; C_WARN=$'\033[33m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
  C_OK=""; C_FAIL=""; C_WARN=""; C_DIM=""; C_RST=""
fi

FAILURES=0
record_pass()  { printf '  %s✓%s %s\n'    "$C_OK"   "$C_RST" "$1"; }
record_fail()  { printf '  %s✗%s %s\n'    "$C_FAIL" "$C_RST" "$1"; FAILURES=$((FAILURES + 1)); }
record_warn()  { printf '  %s!%s %s\n'    "$C_WARN" "$C_RST" "$1"; }
record_hint()  { printf '    %sfix:%s %s\n' "$C_DIM" "$C_RST" "$1"; }

echo "=== Fleet Preflight ==="

# --- 1. Required env vars ---
echo "[1/4] Required environment variables"
# `${arr[@]+"${arr[@]}"}` is the standard bash idiom for safely expanding a
# possibly-empty array under `set -u`. Plain `${arr[@]}` errors as unbound.
REQUIRED=(FLEET_API_KEY WANDB_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY ${EXTRA_REQUIRED[@]+"${EXTRA_REQUIRED[@]}"})
for var in "${REQUIRED[@]}"; do
  # Presence: variable is set AND non-empty AND not pure whitespace.
  value="${!var:-}"
  trimmed="$(printf '%s' "$value" | tr -d '[:space:]')"
  if [ -z "$trimmed" ]; then
    record_fail "$var is unset or empty"
    record_hint "export $var=<your-key> (or pass --env $var=... to sky launch)"
  else
    record_pass "$var is set (${#value} chars)"
  fi
done

# --- 2. gcloud ---
echo "[2/4] Google Cloud authentication"
if [ "$SKIP_GCLOUD" = true ]; then
  record_warn "gcloud checks skipped (--skip-gcloud)"
elif ! command -v gcloud >/dev/null 2>&1; then
  record_fail "gcloud CLI not found on PATH"
  record_hint "install: https://cloud.google.com/sdk/docs/install — or pass --skip-gcloud if not launching to GCP"
else
  active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
  if [ -z "$active_account" ]; then
    record_fail "no active gcloud account"
    record_hint "run: gcloud auth login"
  else
    record_pass "active gcloud account: $active_account"
  fi

  # Application-default credentials are what SkyPilot/boto3-style libs read.
  # `print-access-token` is the cheapest way to confirm ADC are present and
  # not expired; it does no work beyond minting a token.
  if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    record_pass "application-default credentials present"
  else
    record_fail "application-default credentials missing or expired"
    record_hint "run: gcloud auth application-default login"
  fi
fi

# --- 3. SkyPilot ---
echo "[3/4] SkyPilot"
if [ "$SKIP_SKY" = true ]; then
  record_warn "sky check skipped (--skip-sky)"
elif ! command -v sky >/dev/null 2>&1; then
  record_fail "sky CLI not found on PATH"
  record_hint "install: pip install 'skypilot-nightly[gcp,runpod,lambda]' (or your providers)"
else
  # `sky check` exits 0 if at least one cloud is enabled, non-zero otherwise.
  # Capture output for display on failure so the user sees what's wrong.
  if sky_check_output="$(sky check 2>&1)"; then
    enabled_clouds="$(printf '%s\n' "$sky_check_output" | grep -E '^\s*([A-Za-z]+):\s*enabled' | awk -F: '{print $1}' | xargs || true)"
    if [ -n "$enabled_clouds" ]; then
      record_pass "sky check passed; enabled clouds: $enabled_clouds"
    else
      record_pass "sky check passed"
    fi
  else
    record_fail "sky check failed"
    printf '%s\n' "$sky_check_output" | sed 's/^/      /'
    record_hint "run: sky check  (and follow its setup instructions)"
  fi
fi

# --- 4. Liveness probes ---
echo "[4/4] API liveness"
if [ "$SKIP_LIVENESS" = true ]; then
  record_warn "liveness probes skipped (--skip-liveness)"
else
  PROBES=()
  [ "$SKIP_AWS"   = false ] && PROBES+=(aws)
  [ "$SKIP_FLEET" = false ] && PROBES+=(fleet)
  [ "$SKIP_WANDB" = false ] && PROBES+=(wandb)

  if [ ${#PROBES[@]} -eq 0 ]; then
    record_warn "all liveness probes individually skipped"
  else
    PYTHON_BIN="${PYTHON:-python3}"
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      record_warn "$PYTHON_BIN not found; skipping liveness probes"
    else
      probe_json="$("$PYTHON_BIN" scripts/_fleet_preflight_probes.py "${PROBES[@]}" 2>/dev/null || true)"
      if [ -z "$probe_json" ]; then
        record_warn "liveness probe runner returned no output"
      else
        # Parse JSON with Python (jq isn't guaranteed to be installed).
        while IFS=$'\t' read -r status probe detail; do
          case "$status" in
            ok)   record_pass "$probe — $detail" ;;
            fail) record_fail "$probe — $detail" ;;
            skip) record_warn "$probe — $detail" ;;
          esac
        done < <("$PYTHON_BIN" -c "
import json, sys
data = json.loads(sys.stdin.read())
for r in data['results']:
    print(f\"{r['status']}\t{r['probe']}\t{r['detail']}\")
" <<< "$probe_json")
      fi
    fi
  fi
fi

# --- Summary ---
echo ""
if [ "$FAILURES" -eq 0 ]; then
  printf '%s=== Preflight passed ===%s\n' "$C_OK" "$C_RST"
  exit 0
else
  printf '%s=== Preflight failed (%d issue%s) ===%s\n' "$C_FAIL" "$FAILURES" "$([ "$FAILURES" -eq 1 ] || echo s)" "$C_RST"
  echo "Fix the items above before launching, or pass the matching --skip-* flag if a check doesn't apply."
  exit 1
fi
