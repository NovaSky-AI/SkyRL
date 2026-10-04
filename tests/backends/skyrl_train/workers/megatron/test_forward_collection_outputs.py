"""Only collection ranks materialize token outputs after the distributed forward."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("megatron.core")

from skyrl.backends.skyrl_train.distributed.dispatch import MeshRank
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
    MegatronRefWorkerBase,
)


@pytest.mark.parametrize("worker_cls", [MegatronPolicyWorkerBase, MegatronRefWorkerBase])
@pytest.mark.parametrize("sp,tp,pp", [(0, 0, 1), (1, 0, 1), (0, 1, 1), (0, 0, 0)])
def test_forward_keeps_computation_but_only_collects_selected_rank(worker_cls, sp, tp, pp):
    worker = worker_cls.__new__(worker_cls)
    worker.mesh_rank = MeshRank(dp=0, sp=sp, tp=tp, pp=pp, world_size=8, dp_size=1, pp_size=2)
    collects = worker.mesh_rank.is_collection_dp_rank()
    # A non-collection rank must not even inspect/materialize its returned tensor.
    values = torch.tensor([[-0.5, -1.0]]) if collects else object()
    worker._forward_logprobs = Mock(return_value=values)
    data = object()

    output = worker.forward(data)

    worker._forward_logprobs.assert_called_once_with(data)
    assert output.loss_fn_outputs == ([{"logprobs": [-0.5, -1.0]}] if collects else [])
    assert output.metrics == {}
