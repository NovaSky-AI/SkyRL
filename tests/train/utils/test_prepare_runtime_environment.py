"""
uv run --isolated --extra dev pytest tests/train/utils/test_prepare_runtime_environment.py
"""

from types import SimpleNamespace

import pytest

from skyrl.train.utils.utils import prepare_runtime_environment


def _make_config(backend: str = "vllm", strategy: str = "fsdp"):
    """Minimal config-like object covering what prepare_runtime_environment reads."""
    return SimpleNamespace(
        trainer=SimpleNamespace(
            strategy=strategy,
            flash_attn=False,
            placement=SimpleNamespace(
                policy_num_gpus_per_node=1,
                critic_num_gpus_per_node=1,
                ref_num_gpus_per_node=1,
            ),
        ),
        generator=SimpleNamespace(
            inference_engine=SimpleNamespace(
                backend=backend,
                weight_sync_backend="nccl",
                tensor_parallel_size=1,
            )
        ),
    )


class TestVLLMCompileCache:
    """vLLM's torch.compile cache is left enabled unless the user opts out.

    It was force-disabled for every run in #95; re-enabling it roughly halves
    engine startup on a warm cache, and the corrupt-artifact failure that
    motivated the workaround is fixed upstream (pytorch/pytorch#162432, in the
    torch 2.11 we pin).
    """

    def test_cache_is_not_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)

        env_vars = prepare_runtime_environment(_make_config())

        assert "VLLM_DISABLE_COMPILE_CACHE" not in env_vars

    @pytest.mark.parametrize("value", ["1", "0"])
    def test_explicit_user_setting_is_forwarded(self, monkeypatch, value):
        """An explicit opt-out (or opt-in) reaches the Ray workers verbatim."""
        monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", value)

        env_vars = prepare_runtime_environment(_make_config())

        assert env_vars["VLLM_DISABLE_COMPILE_CACHE"] == value

    def test_not_set_for_non_vllm_backends(self, monkeypatch):
        monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

        env_vars = prepare_runtime_environment(_make_config(backend="sglang"))

        assert "VLLM_DISABLE_COMPILE_CACHE" not in env_vars
