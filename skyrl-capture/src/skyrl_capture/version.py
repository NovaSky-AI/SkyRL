"""Version and schema identity.

``SCHEMA_VERSION`` is stamped into every persisted trajectory record and every
export record. Derivation logic that can evolve independently of the raw
exchange carries ``DERIVATION_VERSION`` instead.
"""

__version__ = "0.1.0"

# Bumped when the persisted capture shape changes.
#
# 2: the review batch. Nodes carry `author` and a chained `context_hash`;
#    labels and annotations are trajectory-scoped and unversioned; sampling
#    params and tools are first-class on an exchange; the export set is
#    graph/replay/text_samples/token_samples.
# 3: single tenant, single upstream. No tenant column anywhere, no `targets`
#    or `dataset_sinks` table; a trajectory carries `upstream_snapshot`
#    instead of a target name and a snapshot.
# 4: per-trajectory persistence. A trajectory carries `integrity` -- what
#    capture knows it does not know -- instead of loose counters, and carries
#    no expiry deadline: nothing ends a trajectory on a clock.
SCHEMA_VERSION = 4

# Bumped when prefix matching or graph derivation changes.
#
# 2: the wait before a call is measured from its parent rather than from the
#    previous arrival, which changes every gap on a branched trajectory.
DERIVATION_VERSION = 2

# Bumped when the active journal's framing changes. The framing is a
# cross-language contract (see docs/design/record-format.md), so this version
# is independent of SCHEMA_VERSION.
#
# 1: one append-only journal per trajectory, replacing the global event log.
JOURNAL_FORMAT_VERSION = 1

# Bumped when the record directory's layout, or the committed record in it,
# changes shape. The record is read by the exporters, by the viewer, and by
# anything that opens it later -- a public format, not a convenience dump -- so
# it carries its own version rather than borrowing SCHEMA_VERSION.
#
# 1: a serialized view per finished trajectory, under project/run directories,
#    with an index file per run.
# 2: one append-only global event log, replayed through the live reducer.
# 3: an export is written beside the record and served from there, so its
#    events carry no destination and no delivery result.
# 4: a trajectory carries no upstream overrides. Deployment and credentials
#    come from process configuration, per-inference settings come from the
#    inference request, and what is on a trajectory is capture metadata.
# 5: per trajectory. An `active/` journal per unfinished trajectory and a
#    compiled `committed/` record per finished one; no global log, no global
#    sequence, and a manifest that holds only static format information.
RECORD_FORMAT_VERSION = 5
