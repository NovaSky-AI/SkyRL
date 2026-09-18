#!/usr/bin/env python3
"""Probe Megatron-Bridge / megatron-core APIs for AMD SkyRL compatibility."""

from __future__ import annotations

import json
import sys

from skyrl.backends.skyrl_train.workers.megatron.bridge_compat import (
    inspect_bridge_capabilities,
)


def main() -> int:
    caps = inspect_bridge_capabilities()
    print(json.dumps(caps, indent=2, sort_keys=True))
    if not caps.get("auto_bridge"):
        print("FAIL: AutoBridge is not importable", file=sys.stderr)
        return 1
    if not (caps.get("from_hf_pretrained") or caps.get("from_hf") or caps.get("from_pretrained")):
        print("FAIL: no HF loader on AutoBridge", file=sys.stderr)
        return 1
    print("PASS: Megatron-Bridge exposes an HF loader SkyRL can use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
