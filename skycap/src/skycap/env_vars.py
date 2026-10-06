"""
Environment variable configuration for skycap.

All environment variables used by skycap should be defined here for discoverability.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Service Timeouts
# ─────────────────────────────────────────────────────────────────────────────

SKYCAP_START_TIMEOUT = float(os.environ.get("SKYCAP_START_TIMEOUT", 60.0))
"""
Timeout in seconds for ``CaptureService.start()`` to wait for the server to accept connections.

Default: 60 seconds. Set ``SKYCAP_START_TIMEOUT=120`` for slower environments.
"""

SKYCAP_START_EXPOSURE_TIMEOUT = float(os.environ.get("SKYCAP_START_EXPOSURE_TIMEOUT", 600.0))
"""
Timeout in seconds for ``CaptureService.start()`` when an exposure is configured.

With an exposure (e.g. Cloudflare tunnel), the server waits for both the listener and
the exposure to be ready, which can take longer (especially for tunnel setup).

Default: 600 seconds (10 minutes). Set ``SKYCAP_START_EXPOSURE_TIMEOUT=900`` for slow tunnel setups.
"""
