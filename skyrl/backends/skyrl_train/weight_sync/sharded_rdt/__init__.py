"""The ``sharded_rdt`` weight-transfer backend: RDMA/NIXL pull instead of push.

``sharded_rdt_{base,common,engine,fake,trainer}.py`` are vendored from the
``vllm-rdt-weight-sync`` fork and are deleted once SkyRL's pinned vLLM carries the
trainer-side transfer API natively. The rest is SkyRL glue: ``sharded_rdt_strategy``
(the ``WeightTransferStrategy`` adapter), ``rdt_send`` (trainer-side driver and weight
sources), ``rdt_control_plane``, ``rdt_vllm_register``, ``rdt_libfabric_shim``.

This ``__init__`` imports nothing: ``sharded_rdt_engine`` and ``sharded_rdt_trainer``
import ``vllm`` at module scope, so a re-export here would pull vllm into every
``weight_sync`` import and break the CI job that runs without the wheel. Import the
submodules directly.
"""
