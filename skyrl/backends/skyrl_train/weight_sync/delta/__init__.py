"""Checkpoint-delta weight sync: publish compressed deltas, fetch and reload them.

The trainer XORs new weights against a CPU snapshot of the base checkpoint,
compresses each patch and writes it to a shared directory or object store; each
inference worker fetches the delta, replays it into a local checkpoint copy and
reloads that. Nothing crosses the trainer-to-inference network fabric, which is
the point: this backend is for deployments where the two sides are not
NCCL-reachable.

    checkpoint.py  DeltaCheckpointPublisher, LocalCheckpointStore, manifest + XOR payloads
    payload.py     zstd compress/decompress + uint8 tensor <-> bytes helpers
    trainer.py     DeltaTrainerWeightTransferEngine (send side)
    engine.py      DeltaWeightTransferEngine (receive side, in the vLLM worker)

This ``__init__`` imports nothing: ``trainer`` and ``engine`` import ``vllm`` at
module scope, so a re-export would pull vllm into every import of this package
and break the CPU CI job that runs without the wheel. Import those modules at
their call sites, and register the engines through ``weight_sync/register.py``,
which names them by module path and never imports them.
"""
