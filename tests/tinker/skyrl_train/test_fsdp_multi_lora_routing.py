from types import SimpleNamespace

from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend
from skyrl.tinker import types


def _backend(implementation: str) -> SkyRLTrainBackend:
    backend = object.__new__(SkyRLTrainBackend)
    backend._cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp",
            policy=SimpleNamespace(model=SimpleNamespace(lora=SimpleNamespace(implementation=implementation))),
        )
    )
    backend._model_ids_to_role = {"a": "policy", "b": "policy"}
    backend._model_metadata = {
        "a": SimpleNamespace(adapter_index=0),
        "b": SimpleNamespace(adapter_index=1),
    }
    backend._tokenizer = SimpleNamespace(pad_token_id=0)
    backend._renderer = None
    return backend


def _mixed_batch() -> types.PreparedModelPassBatch:
    inputs = [
        types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2])]),
        types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[3, 4])]),
    ]
    return types.PreparedModelPassBatch(
        all_model_inputs=inputs,
        all_targets=[[2, 3], [4, 5]],
        all_token_weights=[[1.0, 1.0], [1.0, 1.0]],
        all_sampling_logprobs=[[], []],
        all_advantages=[[], []],
        all_values=[[], []],
        all_returns=[[], []],
        all_model_ids=["a", "b"],
        all_loss_fns=["cross_entropy", "cross_entropy"],
        all_loss_fn_configs=[None, None],
        request_batch_slices=[("req-a", "a", 0, 1), ("req-b", "b", 1, 2)],
    )


def test_concurrent_fsdp_keeps_mixed_adapter_batch_together():
    backend = _backend("concurrent")
    batch = _mixed_batch()

    assert backend._get_batch_role(batch.all_model_ids) == "policy"
    assert backend._split_model_pass_batch_by_model_id(batch) == [batch]


def test_single_adapter_implementation_retains_legacy_split():
    backend = _backend("single")
    batches = backend._split_model_pass_batch_by_model_id(_mixed_batch())

    assert [batch.all_model_ids for batch in batches] == [["a"], ["b"]]


def test_concurrent_fsdp_splits_different_loss_functions():
    backend = _backend("concurrent")
    batch = _mixed_batch()
    batch.all_loss_fns[1] = "importance_sampling"

    batches = backend._split_model_pass_batch_by_model_id(batch)

    assert [sub_batch.all_model_ids for sub_batch in batches] == [["a"], ["b"]]


def test_concurrent_fsdp_splits_different_loss_configs():
    backend = _backend("concurrent")
    batch = _mixed_batch()
    batch.all_loss_fn_configs[:] = [{"eps_clip": 0.1}, {"eps_clip": 0.2}]

    batches = backend._split_model_pass_batch_by_model_id(batch)

    assert [sub_batch.all_model_ids for sub_batch in batches] == [["a"], ["b"]]


def test_adapter_slots_follow_rows_through_conversion_and_padding():
    backend = _backend("concurrent")
    backend._dispatch = SimpleNamespace(get_lcm_dp_size=lambda: 4)

    batch = backend._to_training_batch(_mixed_batch(), role="policy")
    assert batch["adapter_indices"].tolist() == [0, 1]

    padded, pad_size = backend._pad_batch(batch)
    assert pad_size == 2
    assert padded["adapter_indices"].tolist() == [0, 1, 0, 0]
    assert padded["loss_mask"][2:].count_nonzero().item() == 0


def test_concurrent_step_applies_all_adapter_adam_settings():
    backend = _backend("concurrent")
    calls = []
    backend._dispatch = SimpleNamespace(
        set_optimizer_hparams=lambda *args, **kwargs: calls.append(("hparams", args, kwargs)),
        optim_step=lambda *args, **kwargs: 2.5,
    )
    request = types.OptimStepInput(
        adam_params=types.AdamParams(
            learning_rate=3e-4,
            beta1=0.8,
            beta2=0.95,
            eps=1e-6,
            weight_decay=0.1,
        )
    )

    output = backend.optim_step("a", request)

    assert calls == [
        (
            "hparams",
            ("policy",),
            {
                "learning_rate": 3e-4,
                "beta1": 0.8,
                "beta2": 0.95,
                "eps": 1e-6,
                "weight_decay": 0.1,
                "model_id": "a",
            },
        )
    ]
    assert output.metrics["skyrl.ai/grad_norm"] == 2.5
