import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.fireworks.sft import FireworksSFTDispatch
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SFTConfig, build_skyrl_config_for_sft
from skyrl.train.entrypoints.main_fireworks_sft import run
from skyrl.train.fireworks_sft_trainer import FireworksSFTTrainer
from skyrl.train.sft_trainer import SFTTrainer


class _Runtime:
    trainer_job_id = "skyrl-smoke-sft-trainer"

    def __init__(self):
        self.training_client = SimpleNamespace()
        self.closed = 0
        self.promotions = []
        self.promoted_resources = []
        self.deleted = 0

    async def promote_final_model(self, **kwargs):
        self.promotions.append(kwargs)
        return {
            "sampler_path": "snapshot://final",
            "checkpoint_resource": "accounts/test/rlorTrainerJobs/trainer/checkpoints/final",
            "checkpoint_type": "CHECKPOINT_TYPE_INFERENCE_BASE",
            "output_model_id": kwargs["output_model_id"],
            "model": {"name": f"accounts/test/models/{kwargs['output_model_id']}"},
        }

    async def promote_checkpoint_resource(self, **kwargs):
        self.promoted_resources.append(kwargs)
        checkpoint = kwargs["checkpoint"]
        return {
            "sampler_path": checkpoint.snapshot_path,
            "checkpoint_resource": checkpoint.checkpoint_resource,
            "checkpoint_type": checkpoint.checkpoint_type,
            "output_model_id": kwargs["output_model_id"],
            "model": {"name": f"accounts/test/models/{kwargs['output_model_id']}"},
        }

    async def delete_trainer(self):
        self.deleted += 1
        return "JOB_STATE_DELETED"

    async def close(self):
        self.closed += 1


class _Tracker:
    def __init__(self):
        self.finished = False

    def finish(self):
        self.finished = True


def _cfg() -> SFTConfig:
    cfg = SFTConfig(
        strategy="fireworks",
        max_length=512,
        remove_microbatch_padding=False,
        enable_ray_gpu_monitor=False,
    )
    cfg.model.path = "Qwen/Qwen3-4B"
    cfg.model.lora.rank = 8
    cfg.fireworks.base_model = "accounts/fireworks/models/qwen3-4b"
    cfg.fireworks.training_shape_id = "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"
    cfg.fireworks.trainer_job_id = "skyrl-smoke-sft-trainer"
    cfg.fireworks.max_seq_len = 32768
    return cfg


def _trainer() -> FireworksSFTTrainer:
    cfg = _cfg()
    trainer = FireworksSFTTrainer(cfg, skyrl_cfg=build_skyrl_config_for_sft(cfg))
    trainer.tokenizer = SimpleNamespace()
    return trainer


def test_fireworks_trainer_reuses_native_train_and_eval_loops() -> None:
    assert FireworksSFTTrainer.train is SFTTrainer.train
    assert FireworksSFTTrainer.train_step is SFTTrainer.train_step
    assert FireworksSFTTrainer.run_eval is SFTTrainer.run_eval


def test_native_train_step_uses_hosted_dispatch() -> None:
    trainer = _trainer()
    calls = []

    class Dispatch:
        def forward_backward(self, model, batch, loss_fn, *, return_per_token_outputs):
            calls.append(("forward_backward", model, loss_fn, return_per_token_outputs))
            return WorkerOutput("scalar", [], {"final_loss": 1.25})

        def optim_step(self, model):
            calls.append(("optim_step", model))
            return 0.75

    trainer.dispatch = Dispatch()
    result = trainer.train_step(TrainingInputBatch({"sequences": torch.tensor([[1, 2]])}), step=1)

    assert calls == [
        ("forward_backward", "policy", "cross_entropy", False),
        ("optim_step", "policy"),
    ]
    assert result["loss"] == pytest.approx(1.25)
    assert result["grad_norm"] == pytest.approx(0.75)


def test_setup_uses_tokenizer_only_path(monkeypatch) -> None:
    trainer = _trainer()
    seen = {}

    def tokenizer(path, **kwargs):
        seen.update(path=path, **kwargs)
        return "tokenizer"

    monkeypatch.setattr("skyrl.train.fireworks_sft_trainer.get_tokenizer", tokenizer)
    monkeypatch.setattr(trainer, "_build_collator", lambda value: ("collator", value))
    monkeypatch.setattr(trainer, "_init_tracker", lambda: seen.update(tracker=True))
    monkeypatch.setattr(trainer, "_init_workers", lambda: seen.update(workers=True))

    trainer.setup()

    assert trainer.is_vlm is False
    assert trainer.tokenizer == "tokenizer"
    assert trainer.collator == ("collator", "tokenizer")
    assert seen == {
        "path": "Qwen/Qwen3-4B",
        "trust_remote_code": True,
        "use_fast": True,
        "padding_side": "left",
        "tracker": True,
        "workers": True,
    }


def test_tracker_initialization_is_deferred() -> None:
    trainer = _trainer()

    trainer._init_tracker()

    assert trainer.tracker is None


def test_backend_setup_creates_no_local_workers(monkeypatch) -> None:
    trainer = _trainer()
    runtime = _Runtime()
    captured = {}

    def connect(**kwargs):
        captured.update(kwargs)
        return runtime

    def init_tracker(self):
        self.tracker = _Tracker()

    monkeypatch.setattr("skyrl.train.fireworks_sft_trainer.FireworksRuntime.connect", connect)
    monkeypatch.setattr(SFTTrainer, "_init_tracker", init_tracker)

    trainer._init_workers()

    assert captured["create_deployment"] is False
    assert isinstance(trainer.dispatch, FireworksSFTDispatch)
    assert isinstance(trainer.tracker, _Tracker)
    trainer.shutdown()
    assert runtime.closed == 1


def test_tracker_failure_closes_runtime(monkeypatch) -> None:
    trainer = _trainer()
    runtime = _Runtime()
    monkeypatch.setattr(
        "skyrl.train.fireworks_sft_trainer.FireworksRuntime.connect",
        lambda **kwargs: runtime,
    )

    def fail_tracker(self):
        raise RuntimeError("tracker failed")

    monkeypatch.setattr(SFTTrainer, "_init_tracker", fail_tracker)

    with pytest.raises(RuntimeError, match="tracker failed"):
        trainer._init_workers()

    assert runtime.closed == 1
    assert trainer._fireworks_runtime is None


def test_fireworks_trainer_reuses_native_checkpoint_methods() -> None:
    assert FireworksSFTTrainer.save_checkpoint is SFTTrainer.save_checkpoint
    assert FireworksSFTTrainer.load_checkpoint is SFTTrainer.load_checkpoint


def test_checkpoint_saves_provider_and_dataloader_state(tmp_path) -> None:
    trainer = _trainer()
    trainer.sft_cfg.ckpt_path = str(tmp_path)
    trainer.global_step = 7
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"position": 7})
    saved = []

    class Dispatch:
        def save_checkpoint(self, model, path, tokenizer):
            Path(path).mkdir(parents=True)
            saved.append((model, path, tokenizer))

    trainer.dispatch = Dispatch()

    checkpoint_path = trainer.save_checkpoint()

    assert checkpoint_path == str(tmp_path / "global_step_7")
    assert saved == [("policy", str(tmp_path / "global_step_7" / "policy"), trainer.tokenizer)]
    assert torch.load(tmp_path / "global_step_7" / "data.pt", weights_only=False) == {"position": 7}
    assert torch.load(tmp_path / "global_step_7" / "trainer_state.pt", weights_only=False)["global_step"] == 7
    assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "7"


def test_checkpoint_restores_provider_and_dataloader_state(tmp_path) -> None:
    trainer = _trainer()
    checkpoint_path = tmp_path / "global_step_7"
    (checkpoint_path / "policy").mkdir(parents=True)
    torch.save({"global_step": 7}, checkpoint_path / "trainer_state.pt")
    torch.save({"position": 7}, checkpoint_path / "data.pt")
    trainer.sft_cfg.resume_from = str(checkpoint_path)
    restored_data = []
    loaded = []
    trainer.train_dataloader = SimpleNamespace(load_state_dict=lambda state: restored_data.append(state))

    class Dispatch:
        def load_checkpoint(self, model, path, **kwargs):
            loaded.append((model, path, kwargs))

    trainer.dispatch = Dispatch()

    step = trainer.load_checkpoint()

    assert step == 7
    assert restored_data == [{"position": 7}]
    assert loaded == [
        (
            "policy",
            str(checkpoint_path / "policy"),
            {"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
    ]


def test_export_final_model_promotes_dcp_name_and_deletes_trainer(tmp_path) -> None:
    trainer = _trainer()
    trainer.global_step = 7
    trainer.sft_cfg.ckpt_path = str(tmp_path)
    trainer.sft_cfg.fireworks.output_model_id = "sft-final-step-7"
    trainer.sft_cfg.fireworks.delete_trainer_after_promotion = True
    runtime = _Runtime()
    trainer._fireworks_runtime = runtime
    policy_dir = tmp_path / "global_step_7" / "policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "fireworks_checkpoint.json").write_text(json.dumps({"checkpoint_name": "skyrl-step-7-deadbeef"}))

    manifest = trainer.export_final_model()

    assert runtime.promotions == [{"checkpoint_name": "skyrl-step-7-deadbeef", "output_model_id": "sft-final-step-7"}]
    assert runtime.deleted == 1
    assert manifest["trainer_state_after_delete"] == "JOB_STATE_DELETED"
    written = json.loads((tmp_path / "global_step_7" / "fireworks_final_model.json").read_text())
    assert written == {key: value for key, value in manifest.items() if key != "trainer_state_after_delete"}
    cleanup = json.loads((tmp_path / "global_step_7" / "fireworks_trainer_cleanup.json").read_text())
    assert cleanup == {
        "trainer_job_id": "skyrl-smoke-sft-trainer",
        "state": "JOB_STATE_DELETED",
    }


def test_export_final_model_promotes_saved_promotable_checkpoint(tmp_path) -> None:
    trainer = _trainer()
    trainer.global_step = 7
    trainer.sft_cfg.ckpt_path = str(tmp_path)
    trainer.sft_cfg.fireworks.output_model_id = "sft-final-step-7"
    runtime = _Runtime()
    trainer._fireworks_runtime = runtime
    policy_dir = tmp_path / "global_step_7" / "policy"
    policy_dir.mkdir(parents=True)
    checkpoint_resource = "accounts/test/rlorTrainerJobs/trainer/checkpoints/step-7-cafebabe"
    (policy_dir / "fireworks_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_name": "skyrl-step-7-deadbeef",
                "promotable_checkpoint": {
                    "snapshot_path": "snapshot://step-7-cafebabe",
                    "checkpoint_resource": checkpoint_resource,
                    "checkpoint_type": "CHECKPOINT_TYPE_INFERENCE_BASE",
                },
            }
        )
    )

    manifest = trainer.export_final_model()

    assert runtime.promotions == []
    assert len(runtime.promoted_resources) == 1
    promoted = runtime.promoted_resources[0]
    assert promoted["checkpoint"].checkpoint_resource == checkpoint_resource
    assert promoted["output_model_id"] == "sft-final-step-7"
    assert manifest["checkpoint_resource"] == checkpoint_resource


def test_export_final_model_falls_back_when_promotable_manifest_is_partial(tmp_path) -> None:
    trainer = _trainer()
    trainer.global_step = 7
    trainer.sft_cfg.ckpt_path = str(tmp_path)
    trainer.sft_cfg.fireworks.output_model_id = "sft-final-step-7"
    trainer.sft_cfg.fireworks.delete_trainer_after_promotion = False
    runtime = _Runtime()
    trainer._fireworks_runtime = runtime
    policy_dir = tmp_path / "global_step_7" / "policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "fireworks_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_name": "skyrl-step-7-deadbeef",
                "promotable_checkpoint": {"snapshot_path": "snapshot://step-7-cafebabe"},
            }
        )
    )

    trainer.export_final_model()

    assert runtime.promoted_resources == []
    assert runtime.promotions == [{"checkpoint_name": "skyrl-step-7-deadbeef", "output_model_id": "sft-final-step-7"}]


def test_promotion_failure_preserves_trainer(tmp_path) -> None:
    trainer = _trainer()
    trainer.global_step = 7
    trainer.sft_cfg.ckpt_path = str(tmp_path)
    trainer.sft_cfg.fireworks.output_model_id = "sft-final-step-7"
    runtime = _Runtime()

    async def fail_promotion(**kwargs):
        raise RuntimeError("promotion failed")

    runtime.promote_final_model = fail_promotion
    trainer._fireworks_runtime = runtime
    policy_dir = tmp_path / "global_step_7" / "policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "fireworks_checkpoint.json").write_text(json.dumps({"checkpoint_name": "skyrl-step-7-deadbeef"}))

    with pytest.raises(RuntimeError, match="promotion failed"):
        trainer.export_final_model()

    assert runtime.deleted == 0
    assert not (tmp_path / "global_step_7" / "fireworks_final_model.json").exists()


def test_shutdown_is_idempotent() -> None:
    trainer = _trainer()
    runtime = _Runtime()
    tracker = _Tracker()
    trainer._fireworks_runtime = runtime
    trainer.tracker = tracker

    trainer.shutdown()
    trainer.shutdown()

    assert tracker.finished is True
    assert runtime.closed == 1


def test_entrypoint_run_uses_hosted_trainer_lifecycle(monkeypatch) -> None:
    events = []

    class Trainer:
        tracker = None
        global_step = 0

        def __init__(self, cfg, skyrl_cfg):
            events.append("init")

        def setup(self):
            events.append("setup")

        def train(self):
            events.append("train")

        def export_final_model(self):
            events.append("export")
            return {"model": "promoted"}

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr("skyrl.train.entrypoints.main_fireworks_sft.FireworksSFTTrainer", Trainer)

    result = run(_cfg())

    assert result == {"model": "promoted"}
    assert events == ["init", "setup", "train", "export", "shutdown"]


def test_entrypoint_interrupts_without_error_logging(monkeypatch) -> None:
    events = []

    class Trainer:
        tracker = SimpleNamespace(log_exception=lambda *args, **kwargs: events.append("logged"))
        global_step = 0

        def __init__(self, cfg, skyrl_cfg):
            pass

        def setup(self):
            events.append("setup")

        def train(self):
            raise KeyboardInterrupt

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr("skyrl.train.entrypoints.main_fireworks_sft.FireworksSFTTrainer", Trainer)

    with pytest.raises(KeyboardInterrupt):
        run(_cfg())

    assert events == ["setup", "shutdown"]
