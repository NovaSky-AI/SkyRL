import pytest

from skyrl.train.config import ResponseLengthAdaptiveLRConfig
from skyrl.train.utils.adaptive_lr import ResponseLengthAdaptiveLRController


def make_controller(**kwargs) -> ResponseLengthAdaptiveLRController:
    cfg = ResponseLengthAdaptiveLRConfig(
        enabled=True,
        ema_alpha=kwargs.get("ema_alpha", 0.5),
        trigger_ratio=kwargs.get("trigger_ratio", 1.2),
        decay_factor=kwargs.get("decay_factor", 0.5),
        cooldown_steps=kwargs.get("cooldown_steps", 0),
        min_monitor_steps=kwargs.get("min_monitor_steps", 0),
        min_scale=kwargs.get("min_scale", 0.125),
        max_decays=kwargs.get("max_decays", 3),
    )
    return ResponseLengthAdaptiveLRController(cfg, warmup_steps=kwargs.get("warmup_steps", 0))


def test_response_length_controller_waits_for_baseline():
    controller = make_controller()
    update = controller.update(step=1, avg_response_length=10.0)

    assert update.triggered is False
    assert update.scale == pytest.approx(1.0)
    assert controller.ema_response_length == pytest.approx(10.0)


def test_response_length_controller_triggers_decay_on_surge():
    controller = make_controller(trigger_ratio=1.1, decay_factor=0.5, cooldown_steps=0)
    controller.update(step=1, avg_response_length=10.0)

    update = controller.update(step=2, avg_response_length=20.0)

    assert update.triggered is True
    assert update.scale == pytest.approx(0.5)
    assert update.metrics["lr_scale"] == pytest.approx(0.5)
    assert update.metrics["response_length_ratio"] == pytest.approx(2.0)
    assert update.metrics["trigger_threshold"] == pytest.approx(1.1)


def test_response_length_controller_respects_cooldown():
    controller = make_controller(trigger_ratio=1.1, decay_factor=0.5, cooldown_steps=2)
    controller.update(step=1, avg_response_length=10.0)
    controller.update(step=2, avg_response_length=20.0)

    update = controller.update(step=3, avg_response_length=40.0)

    assert update.triggered is False
    assert update.scale == pytest.approx(0.5)


def test_response_length_controller_clamps_scale_and_decay_count():
    controller = make_controller(
        trigger_ratio=1.0 + 1e-6,
        decay_factor=0.5,
        cooldown_steps=0,
        min_scale=0.25,
        max_decays=2,
    )
    controller.update(step=1, avg_response_length=10.0)
    controller.update(step=2, avg_response_length=20.0)
    controller.update(step=3, avg_response_length=40.0)
    update = controller.update(step=4, avg_response_length=80.0)

    assert controller.decay_count == 2
    assert update.scale == pytest.approx(0.25)


def test_response_length_controller_state_dict_round_trip():
    controller = make_controller(trigger_ratio=1.1, decay_factor=0.5)
    controller.update(step=1, avg_response_length=10.0)
    controller.update(step=2, avg_response_length=20.0)

    restored = make_controller(trigger_ratio=1.1, decay_factor=0.5)
    restored.load_state_dict(controller.state_dict())

    assert restored.current_scale == pytest.approx(controller.current_scale)
    assert restored.ema_response_length == pytest.approx(controller.ema_response_length)
    assert restored.decay_count == controller.decay_count
