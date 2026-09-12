import pytest

from skyrl.backends.skyrl_train.weight_sync.lora_transport import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRATransportRollbackError,
    LoRATransportServerLifecycle,
    LoRAUpdateRequest,
)


class _Engine:
    def __init__(self, fail_methods=()):
        self.calls = []
        self.fail_methods = set(fail_methods)

    async def collective_rpc(self, method, kwargs):
        self.calls.append((method, kwargs))
        if method in self.fail_methods:
            raise RuntimeError(f"forced {method} failure")


def _layout():
    key = "decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight"
    layout = LoRABridgeSourceLayout(
        "adapter",
        (
            LoRABridgeSource(
                key=key,
                source_rank=0,
                hf_param_names=("down_proj.lora_B.weight",),
                component="linear_out",
                transform="identity",
                shape=(1, 2),
                tensor_parallel_axis=0,
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                expert_parallel_axis=None,
                expert_parallel_rank=0,
                expert_parallel_size=1,
                transform_config=(),
            ),
        ),
    )
    return layout


def _request(generation=1):
    return LoRAUpdateRequest.from_json_dict(
        {
            "adapter_name": "adapter",
            "generation": generation,
            "layout_digest": _layout().layout_digest,
            "source_dtype": "float32",
        }
    )


async def _stage(lifecycle, engine, request, adapter_id):
    await lifecycle.stage_transport(
        engine,
        request,
        adapter_id,
        "stage_lora_nccl_adapter",
    )


@pytest.mark.asyncio
async def test_server_lifecycle_commits_only_an_activated_generation():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine()
    lifecycle._active_ids["adapter"] = 3

    await _stage(lifecycle, engine, _request(), 4)
    with pytest.raises(ValueError, match="has not been activated"):
        await lifecycle.commit(engine, _request(), 4)
    assert lifecycle.get_active_adapter_id("adapter") == 3

    await lifecycle.activate(engine, _request(), 4)
    assert await lifecycle.commit(engine, _request(), 4)
    assert lifecycle.get_active_adapter_id("adapter") == 4
    assert [call[0] for call in engine.calls] == [
        "stage_lora_nccl_adapter",
        "activate_lora_transport_adapter",
        "remove_lora_transport_adapter",
    ]


@pytest.mark.asyncio
async def test_server_lifecycle_rolls_back_without_replaying_cleanup():
    lifecycle = LoRATransportServerLifecycle()
    lifecycle._active_ids["adapter"] = 3
    engine = _Engine(fail_methods={"activate_lora_transport_adapter"})
    await _stage(lifecycle, engine, _request(), 4)

    with pytest.raises(RuntimeError, match="forced"):
        await lifecycle.activate(engine, _request(), 4)
    assert await lifecycle.rollback(engine, _request(), 4)
    assert not await lifecycle.rollback(engine, _request(), 4)

    assert lifecycle.get_active_adapter_id("adapter") == 3
    assert [call[0] for call in engine.calls] == [
        "stage_lora_nccl_adapter",
        "activate_lora_transport_adapter",
        "restore_lora_transport_adapter",
        "discard_lora_transport_adapter",
    ]


@pytest.mark.asyncio
async def test_terminal_replay_preserves_a_newer_staging_transaction():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine()
    await _stage(lifecycle, engine, _request(), 4)
    await lifecycle.activate(engine, _request(), 4)
    assert await lifecycle.commit(engine, _request(), 4)
    await _stage(lifecycle, engine, _request(2), 5)
    before = list(engine.calls)

    assert not await lifecycle.commit(engine, _request(), 4)
    with pytest.raises(ValueError, match="already committed"):
        await lifecycle.rollback(engine, _request(), 4)
    with pytest.raises(ValueError, match="does not match"):
        await lifecycle.rollback(engine, _request(), 5)

    assert engine.calls == before
    assert lifecycle.get_active_adapter_id("adapter") == 4
    assert lifecycle._staged["adapter"][:2] == (_request(2), 5)


@pytest.mark.asyncio
async def test_failed_collective_stage_cleans_successful_ranks_before_retry():
    class PartialStageEngine(_Engine):
        def __init__(self):
            super().__init__()
            self.staged = {}
            self.fail_stage = True

        async def collective_rpc(self, method, kwargs):
            await super().collective_rpc(method, kwargs)
            if method == "stage_lora_nccl_adapter":
                self.staged[0] = kwargs["adapter_id"]
                if self.fail_stage:
                    raise RuntimeError("rank 1 failed staging")
                self.staged[1] = kwargs["adapter_id"]
            elif method == "discard_lora_transport_adapter":
                self.staged = {
                    rank: adapter_id for rank, adapter_id in self.staged.items() if adapter_id != kwargs["adapter_id"]
                }

    engine = PartialStageEngine()
    lifecycle = LoRATransportServerLifecycle()
    lifecycle._active_ids["adapter"] = 3

    with pytest.raises(RuntimeError, match="rank 1 failed"):
        await _stage(lifecycle, engine, _request(), 4)

    assert engine.staged == {}
    assert lifecycle.get_active_adapter_id("adapter") == 3
    assert lifecycle._staged == {}

    engine.fail_stage = False
    await _stage(lifecycle, engine, _request(2), 5)

    assert engine.staged == {0: 5, 1: 5}
    assert lifecycle.get_active_adapter_id("adapter") == 3


@pytest.mark.asyncio
async def test_transport_rejects_changed_layout_before_worker_collective():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine()
    await _stage(lifecycle, engine, _request(), 4)
    await lifecycle.activate(engine, _request(), 4)
    await lifecycle.commit(engine, _request(), 4)
    before = list(engine.calls)
    changed = LoRAUpdateRequest.from_json_dict(
        {
            **_request(2).to_json_dict(),
            "layout_digest": "0" * 64,
        }
    )

    with pytest.raises(ValueError, match="changed the fixed source layout"):
        await lifecycle.stage_transport(
            engine,
            changed,
            5,
            "stage_lora_nccl_adapter",
        )

    assert engine.calls == before
    assert lifecycle._staged == {}
    assert lifecycle.get_active_adapter_id("adapter") == 4
    await lifecycle.unload(engine, "adapter")
    assert lifecycle.get_active_adapter_id("adapter") is None


@pytest.mark.asyncio
async def test_partial_stage_cleanup_failure_is_explicit():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine(fail_methods={"stage_lora_nccl_adapter", "discard_lora_transport_adapter"})

    with pytest.raises(LoRATransportRollbackError, match="could not discard partially staged"):
        await _stage(lifecycle, engine, _request(), 4)


@pytest.mark.asyncio
async def test_unload_releases_active_buffer_once_and_rejects_late_publication():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine()
    await _stage(lifecycle, engine, _request(), 4)
    await lifecycle.activate(engine, _request(), 4)
    await lifecycle.commit(engine, _request(), 4)

    await lifecycle.unload(engine, "adapter")
    calls = list(engine.calls)
    await lifecycle.unload(engine, "adapter")
    with pytest.raises(ValueError, match="unloaded"):
        await _stage(lifecycle, engine, _request(2), 5)

    assert engine.calls == calls
    assert calls[-1] == ("remove_lora_transport_adapter", {"adapter_id": 4})
    assert lifecycle.get_active_adapter_id("adapter") is None


@pytest.mark.asyncio
async def test_failed_unload_can_retry_without_admitting_another_generation():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine()
    await _stage(lifecycle, engine, _request(), 4)
    await lifecycle.activate(engine, _request(), 4)
    await lifecycle.commit(engine, _request(), 4)
    engine.fail_methods.add("remove_lora_transport_adapter")

    with pytest.raises(RuntimeError, match="forced"):
        await lifecycle.unload(engine, "adapter")
    with pytest.raises(ValueError, match="unloaded"):
        await _stage(lifecycle, engine, _request(2), 5)

    assert lifecycle.get_active_adapter_id("adapter") == 4
    engine.fail_methods.clear()
    await lifecycle.unload(engine, "adapter")
    assert lifecycle.get_active_adapter_id("adapter") is None


@pytest.mark.asyncio
async def test_unload_requires_pending_replacement_to_finish_before_removing_buffers():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine()
    await _stage(lifecycle, engine, _request(), 4)
    before = list(engine.calls)

    with pytest.raises(ValueError, match="unfinished replacement"):
        await lifecycle.unload(engine, "adapter")

    assert engine.calls == before
    await lifecycle.rollback(engine, _request(), 4)
    await lifecycle.unload(engine, "adapter")
    assert lifecycle.get_active_adapter_id("adapter") is None


@pytest.mark.asyncio
async def test_abort_before_delayed_stage_prevents_generation_revival():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine()
    assert not await lifecycle.abort(engine, _request())
    with pytest.raises(ValueError, match="completed transaction"):
        await _stage(lifecycle, engine, _request(), 4)
    assert engine.calls == []
    await _stage(lifecycle, engine, _request(2), 5)
    before = list(engine.calls)
    assert not await lifecycle.abort(engine, _request())
    assert engine.calls == before
    await lifecycle.activate(engine, _request(2), 5)
    assert lifecycle.get_active_adapter_id("adapter") == 5


@pytest.mark.asyncio
async def test_abort_preserves_committed_and_mismatched_generations():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine()
    await _stage(lifecycle, engine, _request(), 4)
    await lifecycle.activate(engine, _request(), 4)
    await lifecycle.commit(engine, _request(), 4)
    with pytest.raises(ValueError, match="already committed"):
        await lifecycle.abort(engine, _request())
    await _stage(lifecycle, engine, _request(2), 5)
    before = list(engine.calls)
    mismatch = LoRAUpdateRequest.from_json_dict(
        {
            **_request(2).to_json_dict(),
            "layout_digest": "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="not staged"):
        await lifecycle.abort(engine, mismatch)
    assert engine.calls == before
    assert lifecycle.get_active_adapter_id("adapter") == 4


@pytest.mark.asyncio
async def test_failed_stage_retains_cleanup_id_without_becoming_activatable():
    lifecycle = LoRATransportServerLifecycle()
    engine = _Engine(fail_methods={"stage_lora_nccl_adapter", "discard_lora_transport_adapter"})
    with pytest.raises(LoRATransportRollbackError):
        await _stage(lifecycle, engine, _request(), 4)
    before = list(engine.calls)
    with pytest.raises(ValueError, match="did not finish staging"):
        await lifecycle.activate(engine, _request(), 4)
    assert engine.calls == before
    engine.fail_methods.clear()
    assert await lifecycle.abort(engine, _request())
    assert engine.calls[-1] == ("discard_lora_transport_adapter", {"adapter_id": 4})
    assert not await lifecycle.abort(engine, _request())
    await _stage(lifecycle, engine, _request(2), 5)
