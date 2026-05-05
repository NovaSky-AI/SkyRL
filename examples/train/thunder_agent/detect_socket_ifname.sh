#!/usr/bin/env bash
set -euo pipefail

target_ip="${1:-}"

if [ -n "$target_ip" ] && command -v ip >/dev/null 2>&1; then
  ifname="$(ip route get "$target_ip" 2>/dev/null | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "dev" && (i + 1) <= NF) {
          print $(i + 1)
          exit
        }
      }
    }')"
  if [ -n "${ifname:-}" ]; then
    printf '%s\n' "$ifname"
    exit 0
  fi
fi

if command -v ip >/dev/null 2>&1; then
  ifname="$(ip route show default 2>/dev/null | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "dev" && (i + 1) <= NF) {
          print $(i + 1)
          exit
        }
      }
    }')"
  if [ -n "${ifname:-}" ]; then
    printf '%s\n' "$ifname"
    exit 0
  fi
fi

hostname -I | tr ' ' '\n' | grep -q '^172\.27\.' && {
  printf '%s\n' eth0
  exit 0
}

printf '%s\n' "$(ls /sys/class/net | grep -Ev '^(lo|docker|br-|veth)' | head -n1)"
