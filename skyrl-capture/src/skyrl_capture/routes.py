"""Where the data plane lives in the URL space.

One process serves the control API and the per-trajectory proxy on one port, so
something has to decide which of the two gets a request.

It used to decide by looking for a ``/tr_`` prefix on the path, which is the
shape of an id capture generates. That quietly broke the one thing
``trajectory_id=`` exists for: a caller-supplied name like ``0_1`` did not match,
so every completion went to the control API and 404'd, after creation had
succeeded and handed back a URL with nothing visibly wrong with it.

Keying on the *data* side instead of the control side is what fixes it for good.
The prefix is chosen here and never changes, so the control API can grow new
endpoints at the root without any of them being mistaken for a trajectory, and a
trajectory may be named anything -- including ``v1`` or ``healthz`` -- because
the two no longer share a namespace.
"""

from __future__ import annotations

DATA_PLANE_PREFIX = "/route/"
