"""CPU tests for ``patch_moe_expert_lora_rank``.

vLLM objects are built without ``__init__`` (``LoRAModelManager``, ``MoERunner``,
``FusedMoEWithLoRA``) so the wrappers run without a GPU or a model.
"""

import inspect
import types

import pytest
import torch


@pytest.mark.vllm
class TestManagerPatch:
    @pytest.fixture(autouse=True)
    def patched(self, monkeypatch):
        pytest.importorskip("vllm")
        import vllm.lora.model_manager as model_manager

        from skyrl.backends.skyrl_train.patches.vllm import patch_moe_expert_lora_rank

        cls = model_manager.LoRAModelManager
        # Restored by monkeypatch after the test, together with the wrappers installed below.
        for name in ("__init__", "_create_lora_modules", "_create_merged_loras_inplace", "get_dummy_lora_warmup_rank"):
            monkeypatch.setattr(cls, name, cls.__dict__[name])
        monkeypatch.setattr(model_manager, "from_layer", model_manager.from_layer)
        monkeypatch.setattr(patch_moe_expert_lora_rank, "_PATCHED", False)
        self.model_manager = model_manager
        self.patch = patch_moe_expert_lora_rank
        self.monkeypatch = monkeypatch

    def _lora_config(self, **kwargs):
        from vllm.config.lora import LoRAConfig

        return LoRAConfig(max_lora_rank=64, max_loras=1, **kwargs)

    def _manager(self, expert_rank, lora_config=None):
        manager = self.model_manager.LoRAModelManager.__new__(self.model_manager.LoRAModelManager)
        manager.lora_config = lora_config or self._lora_config()
        manager._skyrl_expert_lora_rank = expert_rank
        return manager

    def _moe_runner(self, tp_size=1):
        from vllm.model_executor.layers.fused_moe import MoERunner

        runner = MoERunner.__new__(MoERunner)
        parallel = types.SimpleNamespace(tp_size=tp_size)
        object.__setattr__(runner, "moe_config", types.SimpleNamespace(moe_parallel_config=parallel))
        return runner

    def test_vllm_contract(self):
        """The patch reads these from the installed vLLM; a rename here silently disables it."""
        cls = self.model_manager.LoRAModelManager
        assert "from_layer" in inspect.unwrap(cls._create_lora_modules).__code__.co_names
        params = inspect.signature(inspect.unwrap(cls.__init__)).parameters
        assert "lora_config" in params and "vllm_config" in params

    def test_read_expert_lora_rank(self):
        lora_config = self._lora_config()
        read = self.patch._read_expert_lora_rank
        assert read(lora_config, types.SimpleNamespace(additional_config={})) is None
        assert read(lora_config, types.SimpleNamespace(additional_config={"max_expert_lora_rank": 64})) is None
        assert read(lora_config, types.SimpleNamespace(additional_config={"max_expert_lora_rank": 4})) == 4
        with pytest.raises(ValueError, match="must be in"):
            read(lora_config, types.SimpleNamespace(additional_config={"max_expert_lora_rank": 128}))

    def test_with_rank_copies_config(self):
        lora_config = self._lora_config()
        # 4 is not in vLLM's MaxLoRARanks literal.
        expert_config = self.patch._with_rank(lora_config, 4)
        assert expert_config.max_lora_rank == 4
        assert lora_config.max_lora_rank == 64
        assert expert_config.max_loras == lora_config.max_loras

    def _create_modules(self, manager, layers):
        """Run the patched ``_create_lora_modules`` over ``layers``; return the ranks ``from_layer`` saw."""
        seen = []
        model_manager = self.model_manager

        def fake_from_layer(layer, max_loras, lora_config, *args, **kwargs):
            seen.append((layer, lora_config.max_lora_rank))

        # Looks up `from_layer` in the module globals, as the real method does.
        def fake_create_lora_modules(manager):
            for layer in layers:
                model_manager.from_layer(layer, 1, manager.lora_config, [], None)

        self.monkeypatch.setattr(self.model_manager, "from_layer", fake_from_layer)
        self.monkeypatch.setattr(self.model_manager.LoRAModelManager, "_create_lora_modules", fake_create_lora_modules)
        self.patch.apply_moe_expert_lora_rank_patch()
        manager._create_lora_modules()
        # The module-level from_layer is restored after the call.
        assert self.model_manager.from_layer is fake_from_layer
        return seen

    def test_moe_layers_get_expert_rank(self):
        runner, dense = self._moe_runner(), torch.nn.Linear(2, 2)
        seen = self._create_modules(self._manager(expert_rank=8), [dense, runner])
        assert seen == [(dense, 64), (runner, 8)]

    def test_unset_expert_rank_keeps_max_lora_rank(self):
        runner = self._moe_runner()
        seen = self._create_modules(self._manager(expert_rank=None), [runner])
        assert seen == [(runner, 64)]

    def test_fully_sharded_requires_divisible_expert_rank(self):
        manager = self._manager(expert_rank=4, lora_config=self._lora_config(fully_sharded_loras=True))
        with pytest.raises(ValueError, match="divisible by the MoE tensor parallel size"):
            self._create_modules(manager, [self._moe_runner(tp_size=8)])

    def _moe_lora_module(self):
        from vllm.lora.layers import FusedMoEWithLoRA

        return FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)

    def _packed_moe_weights(self, rank, num_experts=4, hidden=16, inter=8):
        # pack_moe layout: [w1, w2, w3], lora_A (E, rank, in), lora_B (E, out, rank).
        lora_a = [torch.zeros(num_experts, rank, hidden), torch.zeros(num_experts, rank, inter)]
        lora_a.append(torch.zeros(num_experts, rank, hidden))
        lora_b = [torch.zeros(num_experts, inter, rank), torch.zeros(num_experts, hidden, rank)]
        lora_b.append(torch.zeros(num_experts, inter, rank))
        return types.SimpleNamespace(lora_a=lora_a, lora_b=lora_b)

    @pytest.mark.parametrize("adapter_rank", [4, 8])
    def test_adapter_within_expert_rank_is_accepted(self, adapter_rank):
        manager = self._manager(expert_rank=8)
        manager.modules = {"model.layers.0.mlp.experts": self._moe_lora_module()}
        lora_model = types.SimpleNamespace(
            id=1, loras={"model.layers.0.mlp.experts": self._packed_moe_weights(adapter_rank)}
        )
        self.patch.validate_expert_adapter_ranks(manager, lora_model)

    def test_adapter_above_expert_rank_is_rejected(self):
        manager = self._manager(expert_rank=8)
        manager.modules = {
            "model.layers.0.self_attn.qkv_proj": object(),
            "model.layers.0.mlp.experts": self._moe_lora_module(),
        }
        lora_model = types.SimpleNamespace(
            id=1,
            loras={
                # Dense modules are not checked against the expert rank.
                "model.layers.0.self_attn.qkv_proj": types.SimpleNamespace(lora_a=torch.zeros(64, 16), lora_b=None),
                "model.layers.0.mlp.experts": self._packed_moe_weights(16),
            },
        )
        with pytest.raises(ValueError, match="model.layers.0.mlp.experts has rank 16, above the MoE expert LoRA"):
            self.patch.validate_expert_adapter_ranks(manager, lora_model)

    def test_rejection_happens_before_registration(self):
        calls = []
        self.monkeypatch.setattr(
            self.model_manager.LoRAModelManager, "_create_merged_loras_inplace", lambda self, lora: calls.append(lora)
        )
        self.patch.apply_moe_expert_lora_rank_patch()
        manager = self._manager(expert_rank=8)
        manager.modules = {"model.layers.0.mlp.experts": self._moe_lora_module()}
        manager._registered_adapters = {}
        # vLLM >= 0.32 runs _validate_moe_lora_format first in _add_adapter; it is not under test here.
        self.monkeypatch.setattr(
            self.model_manager.LoRAModelManager, "_validate_moe_lora_format", lambda self, lora: None, raising=False
        )
        lora_model = types.SimpleNamespace(id=1, loras={"model.layers.0.mlp.experts": self._packed_moe_weights(16)})
        with pytest.raises(ValueError, match="above the MoE expert LoRA slot rank"):
            manager._add_adapter(lora_model)
        assert calls == [lora_model]
        assert manager._registered_adapters == {}

    @pytest.mark.parametrize("expert_rank,expected", [(None, 8), (16, 8), (4, 4), (2, 2)])
    def test_dummy_lora_warmup_rank(self, expert_rank, expected):
        self.patch.apply_moe_expert_lora_rank_patch()
        assert self._manager(expert_rank).get_dummy_lora_warmup_rank(8) == expected
