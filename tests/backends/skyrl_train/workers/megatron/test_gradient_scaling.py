from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.gradient_scaling import scale_gradients


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_reject_invalid_gradient_scale(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        scale_gradients([], scale)


def test_accumulation_matches_full_batch_adam_with_expert_gradients():
    reference = [torch.nn.Parameter(torch.tensor([0.2, -0.3])) for _ in range(2)]
    streamed = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    optimizers = [torch.optim.AdamW(params, lr=0.03) for params in (reference, streamed)]
    for iteration in range(3):
        chunks = [torch.tensor([1.0, -2.0]), torch.tensor([3.0, 0.5]), torch.tensor([-0.3, 0.7])]
        for params, optimizer in zip((reference, streamed), optimizers):
            optimizer.zero_grad()
            for x in chunks:
                loss = sum(((p * x - iteration) ** 2).sum() for p in params)
                (loss / len(chunks) if params is reference else loss).backward()
            if params is streamed:
                scale_gradients(
                    [
                        SimpleNamespace(
                            buffers=[SimpleNamespace(grad_data=params[0].grad)],
                            expert_parallel_buffers=[SimpleNamespace(grad_data=params[1].grad)],
                        )
                    ],
                    1 / len(chunks),
                )
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
        for expected, actual in zip(reference, streamed):
            torch.testing.assert_close(actual, expected)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(optimizers[1].state[actual][key], optimizers[0].state[expected][key])
