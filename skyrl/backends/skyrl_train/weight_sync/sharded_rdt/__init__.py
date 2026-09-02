"""The ``sharded_rdt`` weight-transfer backend: RDMA/NIXL pull instead of push.

``sharded_rdt_{common,engine,fake,trainer}.py`` are vendored from the
``vllm-rdt-weight-sync`` fork and are deleted once SkyRL's pinned vLLM carries
this backend natively. ``sharded_rdt_base`` is what remains of the vendored
trainer-side ABCs now that vLLM 0.28 ships them: the two channels a *pull*
backend needs and vLLM has no concept of (per-rank ownership, and a group index).
The rest is SkyRL glue: ``rdt_send`` (weight sources + the trainer init info),
``rdt_vllm_register``, ``rdt_libfabric_shim``.

This ``__init__`` imports nothing: ``sharded_rdt_engine`` and ``sharded_rdt_trainer``
import ``vllm`` at module scope, so a re-export here would pull vllm into every
``weight_sync`` import and break the CI job that runs without the wheel. Import the
submodules directly.
"""
