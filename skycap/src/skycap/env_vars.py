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

# ─────────────────────────────────────────────────────────────────────────────
# Exposure
# ─────────────────────────────────────────────────────────────────────────────

SKYCAP_EXPOSURE_STOP_GRACE = float(os.environ.get("SKYCAP_EXPOSURE_STOP_GRACE", 30.0))
"""
Seconds a stopped server waits for its exposure's ``start`` to give up, when the server is stopped while
the exposure is still opening. An exposure that ignores its stop is left behind after that.

Default: 30 seconds.
"""

SKYCAP_CLOUDFLARED_DOWNLOAD_TIMEOUT = float(os.environ.get("SKYCAP_CLOUDFLARED_DOWNLOAD_TIMEOUT", 60.0))
"""
Seconds the one-time cloudflared download (for ``cloudflare`` exposures, when cloudflared isn't on
``PATH``) may go without receiving data.

Default: 60 seconds.
"""

SKYCAP_CLOUDFLARED_DOWNLOAD_DEADLINE = float(os.environ.get("SKYCAP_CLOUDFLARED_DOWNLOAD_DEADLINE", 600.0))
"""
Seconds the one-time cloudflared download may take in all.

Default: 600 seconds (10 minutes).
"""
