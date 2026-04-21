from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_backend_accepts_ppo_as_critic_request_loss_alias():
    backend_source = (REPO_ROOT / "skyrl/backends/skyrl_train_backend.py").read_text()

    assert 'if role == "critic" and loss_fn not in {"ppo", "ppo_critic"}:' in backend_source
    assert "Critic batches must use loss_fn='ppo' or 'ppo_critic'" in backend_source


def test_backend_and_worker_preserve_dedicated_critic_loss_path():
    backend_source = (REPO_ROOT / "skyrl/backends/skyrl_train_backend.py").read_text()
    worker_source = (REPO_ROOT / "skyrl/backends/skyrl_train/workers/worker.py").read_text()

    assert 'self._dispatch.set_algorithm_config(' in backend_source
    assert 'self._dispatch.forward_backward("critic", batch)' in backend_source
    assert 'if role == "critic":\n            return loss_fn, loss_fn_config' in backend_source
    assert "self.critic_loss_fn: Callable = ppo_critic_loss" in worker_source


def test_backend_normalizes_public_ppo_policy_loss_to_train_loss_name():
    backend_source = (REPO_ROOT / "skyrl/backends/skyrl_train_backend.py").read_text()

    assert 'if loss_fn != "ppo":\n            return loss_fn, loss_fn_config' in backend_source
    assert 'return "regular", normalized_config or None' in backend_source


def test_backend_translates_public_ppo_clip_thresholds_to_train_config_keys():
    backend_source = (REPO_ROOT / "skyrl/backends/skyrl_train_backend.py").read_text()

    assert 'clip_low_threshold = normalized_config.pop("clip_low_threshold", None)' in backend_source
    assert 'clip_high_threshold = normalized_config.pop("clip_high_threshold", None)' in backend_source
    assert 'normalized_config["eps_clip_low"] = 1.0 - clip_low_threshold' in backend_source
    assert 'normalized_config["eps_clip_high"] = clip_high_threshold - 1.0' in backend_source
