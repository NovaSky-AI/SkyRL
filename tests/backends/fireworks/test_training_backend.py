import json
from types import SimpleNamespace

import pytest
import ray
import torch

from skyrl.backends.fireworks.training_backend import FireworksPolicyDispatch
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.trainer_utils import run_on_each_node


class _Future:
    def __init__(self, value):
        self.value = value
        self.timeout = None

    def result(self, timeout=None):
        self.timeout = timeout
        return self.value


class _TrainingClient:
    def __init__(self):
        self.forward_backward_calls = []
        self.optim_params = []
        self.saved_states = []
        self.loaded_states = []

    def forward_backward(self, datums, loss_fn):
        self.forward_backward_calls.append((datums, loss_fn))
        return _Future(
            SimpleNamespace(
                metrics={"loss:sum": 1.25, "response_tokens": 4.0},
                loss_fn_output_type="scalar",
            )
        )

    def optim_step(self, params):
        self.optim_params.append(params)
        return _Future(SimpleNamespace(metrics={"grad_norm": 0.75}))

    def save_state(self, name):
        self.saved_states.append(name)
        return _Future(SimpleNamespace(path=f"tinker://source/weights/{name}"))

    def resolve_checkpoint_path(self, checkpoint_name, source_job_id=None):
        return f"cross-job://{source_job_id}/{checkpoint_name}"

    def load_state_with_optimizer(self, path):
        self.loaded_states.append((path, True))
        return _Future(SimpleNamespace())

    def load_state(self, path):
        self.loaded_states.append((path, False))
        return _Future(SimpleNamespace())


def _cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.fireworks.max_seq_len = 128
    return cfg


def test_policy_dispatch_stages_and_submits_importance_sampling() -> None:
    training_client = _TrainingClient()
    runtime = SimpleNamespace(
        training_client=training_client, publish_sampler_weights=None
    )
    built_batches = []

    def datum_builder(batch, *, max_seq_len):
        built_batches.append((batch, max_seq_len))
        return ["datum"] * batch.batch_size

    dispatch = FireworksPolicyDispatch(_cfg(), runtime, datum_builder=datum_builder)
    batch = TrainingInputBatch({"sequences": torch.arange(12).reshape(4, 3)})
    staged = dispatch.stage_data("policy", batch, [(0, 2), (2, 4)])

    assert [part.batch_size for part in staged] == [2, 2]
    output = dispatch.forward_backward_from_staged("policy", staged[0])

    assert training_client.forward_backward_calls == [
        (["datum", "datum"], "importance_sampling")
    ]
    assert built_batches[0][1] == 128
    assert output.metrics["final_loss"] == pytest.approx(1.25)
    assert output.metrics["response_tokens"] == pytest.approx(4.0)


def test_policy_dispatch_optimizer_uses_skyrl_optimizer_config() -> None:
    cfg = _cfg()
    cfg.trainer.policy.optimizer_config.lr = 2e-5
    cfg.trainer.policy.optimizer_config.adam_betas = [0.8, 0.9]
    cfg.trainer.policy.optimizer_config.weight_decay = 0.1
    cfg.trainer.policy.optimizer_config.max_grad_norm = 2.0
    training_client = _TrainingClient()
    dispatch = FireworksPolicyDispatch(
        cfg, SimpleNamespace(training_client=training_client)
    )

    grad_norm = dispatch.optim_step("policy")

    params = training_client.optim_params[0]
    assert params.learning_rate == pytest.approx(2e-5)
    assert params.beta1 == pytest.approx(0.8)
    assert params.beta2 == pytest.approx(0.9)
    assert params.weight_decay == pytest.approx(0.1)
    assert params.grad_clip_norm == pytest.approx(2.0)
    assert grad_norm == pytest.approx(0.75)


def test_policy_dispatch_saves_and_cross_job_loads_dcp_checkpoint(tmp_path) -> None:
    training_client = _TrainingClient()
    runtime = SimpleNamespace(
        training_client=training_client,
        trainer_job_id="source-trainer",
    )
    cfg = _cfg()
    cfg.trainer.fireworks.request_timeout_s = 123
    dispatch = FireworksPolicyDispatch(cfg, runtime)
    ckpt_dir = tmp_path / "global_step_7" / "policy"

    dispatch.save_checkpoint("policy", str(ckpt_dir), tokenizer="unused")

    assert len(training_client.saved_states) == 1
    assert training_client.saved_states[0].startswith("skyrl-step-7-")
    manifest = json.loads(
        (ckpt_dir / "fireworks_checkpoint.json").read_text()
    )
    assert manifest["source_trainer_job_id"] == "source-trainer"
    assert manifest["includes_optimizer_state"] is True

    dispatch.load_checkpoint(
        "policy",
        str(ckpt_dir),
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )

    assert training_client.loaded_states == [
        (
            f"cross-job://source-trainer/{manifest['checkpoint_name']}",
            True,
        )
    ]


def test_policy_dispatch_serverless_load_uses_provider_path(tmp_path) -> None:
    training_client = _TrainingClient()
    runtime = SimpleNamespace(training_client=training_client, trainer_job_id=None)
    dispatch = FireworksPolicyDispatch(_cfg(), runtime)
    ckpt_dir = tmp_path / "global_step_2" / "policy"
    ckpt_dir.mkdir(parents=True)
    provider_path = "tinker://serverless-run/weights/step-2"
    (ckpt_dir / "fireworks_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint_kind": "fireworks_dcp",
                "checkpoint_name": "step-2",
                "provider_path": provider_path,
                "source_trainer_job_id": None,
                "includes_optimizer_state": True,
            }
        )
    )

    dispatch.load_checkpoint(
        "policy",
        str(ckpt_dir),
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )

    assert training_client.loaded_states == [(provider_path, False)]


def test_hosted_empty_node_cleanup_does_not_initialize_ray(monkeypatch) -> None:
    monkeypatch.setattr(
        ray,
        "remote",
        lambda *args, **kwargs: pytest.fail("ray.remote should not be called"),
    )

    assert run_on_each_node([], lambda: None) == []
