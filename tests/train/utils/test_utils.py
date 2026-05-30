"""
uv run --isolated --extra dev pytest tests/train/utils/test_utils.py
"""

from copy import deepcopy
from types import SimpleNamespace

from skyrl.train.utils.utils import update_model_config


def _make_config():
    """Build a config-like object with a nested sub-config attribute."""
    sub = SimpleNamespace(mtp_num_layers=4, hidden_size=128)
    return SimpleNamespace(num_nextn_predict_layers=4, sub_config=sub)


class TestUpdateModelConfigNonMutating:
    """Lock in the non-mutating contract of ``update_model_config``."""

    def test_input_is_not_mutated_at_top_level(self):
        cfg = _make_config()
        before = deepcopy(cfg)
        update_model_config(cfg, {"num_nextn_predict_layers": 0})
        assert cfg.num_nextn_predict_layers == before.num_nextn_predict_layers
        assert cfg.sub_config.mtp_num_layers == before.sub_config.mtp_num_layers

    def test_returned_copy_carries_top_level_overrides(self):
        cfg = _make_config()
        new_cfg = update_model_config(cfg, {"num_nextn_predict_layers": 0})
        assert new_cfg.num_nextn_predict_layers == 0
        assert new_cfg is not cfg

    def test_nested_overrides_do_not_leak_back_into_input(self):
        cfg = _make_config()
        new_cfg = update_model_config(cfg, {"sub_config": {"mtp_num_layers": 0}})
        assert new_cfg.sub_config.mtp_num_layers == 0
        assert cfg.sub_config.mtp_num_layers == 4
        assert new_cfg.sub_config is not cfg.sub_config
