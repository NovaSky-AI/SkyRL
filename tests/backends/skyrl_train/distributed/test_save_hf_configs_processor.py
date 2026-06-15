"""CPU tests for DistributedStrategy.save_hf_configs processor handling.

Verifies that a vision-language checkpoint export writes the HF processor
(``preprocessor_config.json`` etc.) and that text-only exports do not, without
requiring GPUs, distributed init, or network access.
"""

from unittest.mock import MagicMock, patch

from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy


class _StubStrategy(DistributedStrategy):
    """Concrete strategy that stubs every abstractmethod so the CPU-only
    ``save_hf_configs`` path can be exercised in isolation."""

    def setup_distributed(self):  # pragma: no cover - stub
        pass

    def backward(self, loss, model, optimizer, **kwargs):  # pragma: no cover - stub
        pass

    def optimizer_step(self, optimizer, model, scheduler, name="model", **kwargs):  # pragma: no cover - stub
        pass

    def save_checkpoint(self, model, ckpt_dir, node_local_rank, optimizer, scheduler, tokenizer):  # pragma: no cover
        pass

    def load_checkpoint(
        self, model, ckpt_dir, optimizer, scheduler, load_module_strict, load_optimizer_states, load_lr_scheduler_states
    ):  # pragma: no cover - stub
        pass

    def save_hf_model(self, model, output_dir, tokenizer=None, **kwargs):  # pragma: no cover - stub
        pass


def _make_model_config(name_or_path="some/base-model"):
    model_config = MagicMock()
    model_config.name_or_path = name_or_path
    # save_pretrained is a no-op for the test; we only care about the processor.
    model_config.save_pretrained = MagicMock()
    return model_config


_STRATEGY_MOD = "skyrl.backends.skyrl_train.distributed.strategy"


class TestSaveHfConfigsProcessor:
    def test_saves_processor_for_vlm(self, tmp_path):
        """A VLM export resolves the processor and saves it into the HF dir."""
        processor = MagicMock()
        model_config = _make_model_config()
        with (
            patch(f"{_STRATEGY_MOD}.check_is_vlm", return_value=True) as check_is_vlm,
            patch(f"{_STRATEGY_MOD}.get_processor", return_value=processor) as get_processor,
            patch(f"{_STRATEGY_MOD}.GenerationConfig"),
        ):
            _StubStrategy().save_hf_configs(model_config, str(tmp_path))

        # The check reuses the loaded config object (no redundant AutoConfig I/O).
        check_is_vlm.assert_called_once_with(model_config)
        get_processor.assert_called_once_with("some/base-model")
        processor.save_pretrained.assert_called_once_with(str(tmp_path))

    def test_no_processor_for_text_only(self, tmp_path):
        """A text-only export never resolves or saves a processor."""
        model_config = _make_model_config()
        with (
            patch(f"{_STRATEGY_MOD}.check_is_vlm", return_value=False) as check_is_vlm,
            patch(f"{_STRATEGY_MOD}.get_processor") as get_processor,
            patch(f"{_STRATEGY_MOD}.GenerationConfig"),
        ):
            _StubStrategy().save_hf_configs(model_config, str(tmp_path))

        check_is_vlm.assert_called_once_with(model_config)
        get_processor.assert_not_called()

    def test_vlm_check_failure_does_not_raise(self, tmp_path):
        """A VLM-detection error is swallowed (export must not crash)."""
        with (
            patch(f"{_STRATEGY_MOD}.check_is_vlm", side_effect=RuntimeError("config unavailable")),
            patch(f"{_STRATEGY_MOD}.get_processor") as get_processor,
            patch(f"{_STRATEGY_MOD}.GenerationConfig"),
        ):
            # Should not raise even though the VLM check blew up.
            _StubStrategy().save_hf_configs(_make_model_config(), str(tmp_path))

        get_processor.assert_not_called()

    def test_processor_failure_does_not_raise(self, tmp_path):
        """A processor resolution error is swallowed (export must not crash)."""
        with (
            patch(f"{_STRATEGY_MOD}.check_is_vlm", return_value=True),
            patch(f"{_STRATEGY_MOD}.get_processor", side_effect=RuntimeError("no processor")),
            patch(f"{_STRATEGY_MOD}.GenerationConfig"),
        ):
            # Should not raise.
            _StubStrategy().save_hf_configs(_make_model_config(), str(tmp_path))

    def test_skipped_when_no_name_or_path(self, tmp_path):
        """Models initialized without a base path skip processor + gen config."""
        with (
            patch(f"{_STRATEGY_MOD}.check_is_vlm") as check_is_vlm,
            patch(f"{_STRATEGY_MOD}.get_processor") as get_processor,
            patch(f"{_STRATEGY_MOD}.GenerationConfig"),
        ):
            _StubStrategy().save_hf_configs(_make_model_config(name_or_path=""), str(tmp_path))

        check_is_vlm.assert_not_called()
        get_processor.assert_not_called()
