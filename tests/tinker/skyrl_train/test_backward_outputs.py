from types import SimpleNamespace
from unittest.mock import Mock

import pytest

backend_module = pytest.importorskip("skyrl.backends.skyrl_train_backend")


@pytest.mark.parametrize("enabled", [True, False])
def test_backward_outputs_preserve_request_sample_counts(enabled):
    backend = object.__new__(backend_module.SkyRLTrainBackend)
    backend.config = backend_module.MegatronBackendOverrides(
        return_backward_per_token_outputs=enabled,
    )
    assert "return_backward_per_token_outputs" not in backend.config.model_extra
    assert backend_module.MegatronBackendOverrides().return_backward_per_token_outputs
    backend._cfg = SimpleNamespace(
        trainer=SimpleNamespace(strategy="megatron", micro_train_batch_size_per_gpu=1),
    )
    backend._get_batch_role = Mock(return_value="policy")
    backend._to_training_batch = Mock(return_value=object())
    backend._pad_batch = Mock(return_value=(object(), 1))
    backend._dispatch = Mock()
    backend._dispatch.forward_backward.return_value = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": [-0.5]} if enabled else {} for _ in range(4)],
        loss_fn_output_type="cross_entropy",
        metrics={"loss": 0.5},
    )
    prepared = SimpleNamespace(
        all_model_ids=["model"],
        all_loss_fns=["cross_entropy"],
        all_loss_fn_configs=[None],
        request_batch_slices=[("first", None, 0, 1), ("second", None, 1, 3)],
    )

    results = backend._forward_backward_single_model_batch(prepared)

    assert backend._dispatch.forward_backward.call_args.kwargs["return_per_token_outputs"] is enabled
    assert [len(results[key].loss_fn_outputs) for key in ("first", "second")] == [1, 2]
    assert results["first"].metrics == results["second"].metrics == {"total_loss:sum": 0.5}
    if enabled:
        assert results["first"].loss_fn_outputs[0]["logprobs"]["data"] == [-0.5]
    else:
        assert results["first"].loss_fn_outputs == [{}]
        assert results["second"].loss_fn_outputs == [{}, {}]
