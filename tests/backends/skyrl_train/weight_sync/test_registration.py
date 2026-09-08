"""Tests for registering SkyRL's weight-transfer engines into vLLM's factories.

There are two registries and they are populated by different code:

* ``WeightTransferTrainerFactory`` — the send side, populated by
  ``register.register_trainer_engines``.
* ``WeightTransferEngineFactory`` — the receive side, populated by
  ``register.register_receive_engines``. It must run in every vLLM worker
  process AND on the driver, which validates ``WeightTransferConfig.backend``.

Every test here **resolves** each entry rather than asserting the key is
present. Three of the registrations pass the engine as a module path and class
name in strings, which vLLM imports only when a worker builds the backend — so a
stale path or a renamed class satisfies a membership check and fails on a live
inference worker instead.
"""

import pytest

pytest.importorskip("vllm", reason="the factories under test come from vLLM")

pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.weight_sync import (  # noqa: E402
    get_vllm_receive_backend,
)


def _register_receive_side() -> None:
    """Everything the driver and each vLLM worker register (idempotent)."""
    from skyrl.backends.skyrl_train.weight_sync.register import register_receive_engines

    register_receive_engines()


def _resolve(registry, name):
    """Force the registry entry to produce its class.

    ``register_engine`` wraps both calling conventions (module path + class name,
    or a direct class) in a loader, so calling it is what actually imports a
    lazily-registered engine.
    """
    return registry[name]()


class TestTrainerFactory:
    def test_skyrl_backends_resolve_to_their_classes(self):
        from vllm.distributed.weight_transfer.factory import (
            WeightTransferTrainerFactory,
        )

        from skyrl.backends.skyrl_train.weight_sync.delta.trainer import (
            DeltaTrainerWeightTransferEngine,
        )
        from skyrl.backends.skyrl_train.weight_sync.register import (
            register_trainer_engines,
        )
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_trainer import (
            ShardedRDTTrainerWeightTransferEngine,
        )

        register_trainer_engines()
        registry = WeightTransferTrainerFactory._registry
        assert _resolve(registry, "delta") is DeltaTrainerWeightTransferEngine
        assert _resolve(registry, "sharded_rdt") is ShardedRDTTrainerWeightTransferEngine

    def test_vllms_own_engines_are_still_there(self):
        """SkyRL registers alongside vLLM's, never over them."""
        from vllm.distributed.weight_transfer.factory import (
            WeightTransferTrainerFactory,
        )

        from skyrl.backends.skyrl_train.weight_sync.register import (
            register_trainer_engines,
        )

        register_trainer_engines()
        for name in ("nccl", "ipc"):
            assert name in WeightTransferTrainerFactory._registry


class TestReceiveFactory:
    def test_skyrl_backends_resolve_to_their_classes(self):
        from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

        from skyrl.backends.skyrl_train.weight_sync.delta.engine import (
            DeltaWeightTransferEngine,
        )
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_engine import (
            ShardedRDTWeightTransferEngine,
        )
        from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
            get_skyrl_ipc_engine,
            get_skyrl_nccl_engine,
        )

        _register_receive_side()
        registry = WeightTransferEngineFactory._registry
        assert _resolve(registry, "skyrl_nccl") is get_skyrl_nccl_engine()
        assert _resolve(registry, "skyrl_ipc") is get_skyrl_ipc_engine()
        assert _resolve(registry, "delta") is DeltaWeightTransferEngine
        assert _resolve(registry, "sharded_rdt") is ShardedRDTWeightTransferEngine

    def test_skyrl_nccl_and_ipc_subclass_vllms_engines(self):
        """The new names exist because ``register_engine`` refuses a duplicate,
        not because the engines are unrelated to vLLM's."""
        from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
            SkyrlDrafterReloadMixin,
            get_skyrl_ipc_engine,
            get_skyrl_nccl_engine,
        )

        assert issubclass(get_skyrl_nccl_engine(), (NCCLWeightTransferEngine, SkyrlDrafterReloadMixin))
        assert issubclass(get_skyrl_ipc_engine(), (IPCWeightTransferEngine, SkyrlDrafterReloadMixin))


@pytest.mark.parametrize(
    "weight_sync_backend,colocate_all",
    [("nccl", False), ("nccl", True), ("delta", False), ("sharded_rdt", False), ("rdt", False)],
)
def test_every_selectable_receive_backend_is_registered(weight_sync_backend, colocate_all):
    """The name the driver puts in ``WeightTransferConfig`` must be a name the
    factory can build.

    ``build_vllm_cli_args`` configures the servers with
    ``get_vllm_receive_backend(...)`` and vLLM validates it at
    ``create_engine``, in the worker. A backend selectable in config but absent
    from the registry therefore fails only once a real engine is constructed.
    """
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    _register_receive_side()
    name = get_vllm_receive_backend(weight_sync_backend, colocate_all)
    assert _resolve(WeightTransferEngineFactory._registry, name) is not None


def test_weight_transfer_config_accepts_the_skyrl_backends():
    """``WeightTransferConfig.backend`` is typed ``Literal[...] | str``, so vLLM
    accepts SkyRL's names while still validating its own built-ins. The driver
    stamps one of these into the servers' config, so it has to construct.
    """
    from vllm.config import WeightTransferConfig

    _register_receive_side()
    for name in ("sharded_rdt", "delta", "skyrl_nccl", "skyrl_ipc"):
        assert WeightTransferConfig(backend=name).backend == name
    for name in ("nccl", "ipc"):
        assert WeightTransferConfig(backend=name).backend == name


def test_the_per_engine_registration_helpers_stand_alone():
    """``register_delta_weight_transfer_engine`` / ``register_rdt_weight_transfer_engine``
    are callable on their own, not only through ``register_receive_engines``."""
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    from skyrl.backends.skyrl_train.weight_sync.register import (
        register_delta_weight_transfer_engine,
        register_rdt_weight_transfer_engine,
    )

    register_delta_weight_transfer_engine()
    register_rdt_weight_transfer_engine()
    for name in ("delta", "sharded_rdt"):
        assert _resolve(WeightTransferEngineFactory._registry, name) is not None


def test_registration_is_idempotent():
    """``register_engine`` raises on a duplicate name, so every helper has to
    guard — the driver and the worker extension both call these."""
    from skyrl.backends.skyrl_train.weight_sync.register import register_trainer_engines

    for _ in range(3):
        _register_receive_side()
        register_trainer_engines()


def test_build_vllm_cli_args_registers_every_backend_it_can_select(monkeypatch):
    """The driver must register what it configures.

    ``build_vllm_cli_args`` stamps ``WeightTransferConfig(backend=...)`` from
    ``get_vllm_receive_backend``, and vLLM validates that name against the
    receive registry at ``create_engine``. This simulates a fresh driver
    process — registry cleared, module guards reset — so a backend the driver
    selects but forgets to register fails here rather than on a live worker.
    """
    import vllm.platforms
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
    from vllm.platforms.interface import UnspecifiedPlatform

    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
    from skyrl.backends.skyrl_train.weight_sync import register
    from skyrl.train.config import SkyRLTrainConfig

    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())
    # Fresh-process simulation; monkeypatch restores both on teardown.
    monkeypatch.setattr(register, "_RECEIVE_REGISTERED", False)
    for name in ("skyrl_nccl", "skyrl_ipc", "delta", "sharded_rdt"):
        if name in WeightTransferEngineFactory._registry:
            monkeypatch.delitem(WeightTransferEngineFactory._registry, name)

    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "delta"
    args = build_vllm_cli_args(cfg)

    backend = args.weight_transfer_config.backend
    assert backend == "delta"
    assert _resolve(WeightTransferEngineFactory._registry, backend) is not None
