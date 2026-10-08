"""The ``sharded_rdt`` weight-transfer backend: RDMA/NIXL pull instead of push.

SkyRL owns the sender and receiver implementations, including the pull-backend
extensions vLLM does not provide: per-rank ownership and a group index.
``rdt_libfabric_shim`` prepares the RDT runtime. Factory registration lives in
``weight_sync/register.py``.

This ``__init__`` imports nothing, so importing ``weight_sync`` does not require
the optional vLLM dependency.
"""
