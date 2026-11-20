"""Startup ordering for managed fully async engines, using driver-side doubles."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyrl.backends.skyrl_train.inference_servers import setup
from skyrl.train import trainer as trainer_module
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.trainer import RayPPOTrainer


@pytest.mark.parametrize("mode", ["overlap", "inference failure", "training failure", "custom schedule", "opt out"])
def test_managed_async_model_startup(tmp_path, monkeypatch, mode):
    events = []
    cfg = SkyRLTrainConfig()
    cfg.trainer.export_path = cfg.trainer.ckpt_path = str(tmp_path)
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.overlap_model_startup = mode != "opt out"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.algorithm.use_kl_loss = cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.critic.model.path = None
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size = 2
    cfg.trainer.max_training_steps = 3

    exp = BasePPOExp.__new__(BasePPOExp)
    exp.cfg = cfg
    exp.tokenizer = SimpleNamespace(pad_token_id=7)
    exp.train_dataset = range(40)
    exp.eval_dataset = exp.colocate_pg = None
    exp.get_tracker = Mock()
    exp.get_trajectory_logger = Mock()
    policy = Mock()
    allocations = []

    def start_training(path, num_training_steps):
        allocations.append(num_training_steps)
        events.append("training launched")
        return ["training-init"]

    policy.async_init_model.side_effect = start_training
    policy.async_run_ray_method.return_value = ["pad"]
    monkeypatch.setattr(trainer_module, "PPORayActorGroup", Mock(return_value=policy))

    client = Mock()
    generator = Mock()
    server_setup = setup.InferenceServerSetup(proxy_url="http://router", server_urls=["http://server"])

    def start_inference(cfg, tokenizer, *, placement_group, on_servers_launched):
        events.append("inference launched")
        if on_servers_launched is not None:
            on_servers_launched()
        if mode == "inference failure":
            raise ValueError("inference failed")
        events.append("router healthy")
        return client, server_setup

    monkeypatch.setattr(setup, "build_new_inference_client", start_inference)

    def get_generator(cfg, tokenizer, inference_client):
        assert inference_client is client
        assert events[-1] == "router healthy"
        events.append("generator")
        return generator

    exp.get_generator = get_generator

    def get_trainer(**kwargs):
        assert kwargs["generator"] is generator
        assert kwargs["inference_engine_client"] is client
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.cfg = cfg
        trainer.colocate_all = False
        trainer.total_training_steps = 5 if mode == "custom schedule" else 3
        trainer.colocate_pg = None
        trainer.tokenizer = exp.tokenizer
        trainer.inference_engine_client = client
        trainer._vllm_metrics_scraper = None
        return trainer

    exp.get_trainer = get_trainer

    def wait(refs, label):
        if refs == ["training-init"]:
            events.append("training ready")
            if mode == "training failure":
                raise ValueError("training failed")

    monkeypatch.setattr(trainer_module.deadline, "ray_get", wait)
    dispatch = Mock()
    monkeypatch.setattr(trainer_module, "WorkerDispatch", dispatch)

    if mode.endswith("failure"):
        with pytest.raises(ValueError, match=mode.split()[0] + " failed"):
            exp._setup_trainer()
        policy.shutdown.assert_called_once_with()
        dispatch.assert_not_called()
        return

    trainer = exp._setup_trainer()
    if mode == "overlap":
        assert events == ["inference launched", "training launched", "router healthy", "generator", "training ready"]
    elif mode == "opt out":
        assert events == ["inference launched", "router healthy", "generator", "training launched", "training ready"]
    expected_steps = [3, 5] if mode == "custom schedule" else [3]
    assert allocations == [steps * cfg.trainer.update_epochs_per_batch for steps in expected_steps]
    assert trainer.policy_model is policy
    assert dispatch.call_args.kwargs["inference_engine_client"] is client
    policy.async_run_ray_method.assert_called_once_with("pass_through", "_set_pad_token_id", 7)
    assert policy.shutdown.call_count == int(mode == "custom schedule")
