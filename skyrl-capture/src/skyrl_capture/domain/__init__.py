"""The domain: what capture decided, and the two shapes it keeps it in.

A dependency rule, enforced by `tests/test_boundaries.py`: nothing here imports
from the proxy, the persistence layer, the control plane or the runtime. The
domain is what those are built around.

* `records` -- `ActiveTrajectory`, the hot aggregate one process owns while a
  trajectory is in use, and `TrajectoryRecord`, the frozen artifact it
  compiles into when it finishes. Both project the same `TrajectoryDocument`.
* `graph` -- one `ConversationGraph` per trajectory, for both capture modes,
  and the `GraphDelta` that is the only way it changes.
* `timing` -- what the clocks on a set of exchanges say about each other, which
  is derived on read rather than stored on any one of them.
* `models` -- the request edge: identifiers, routes, export formats, and the
  validation a command runs before it decides anything.
* `hashing` -- every hash in the system, and the canonical JSON they are taken
  over. Node identity is a domain fact.
"""
