from unittest.mock import MagicMock

from skyrl.utils import log


def test_wandb_tracker_delegates_authentication_to_wandb(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    wandb = MagicMock()
    monkeypatch.setattr(log, "wandb", wandb)

    tracker = log.WandbTracker(config={"model": "test"}, project="project")

    wandb.init.assert_called_once_with(config={"model": "test"}, project="project")
    assert tracker.run is wandb.init.return_value
