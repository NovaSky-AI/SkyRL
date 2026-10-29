"""remote_code_identity.check_files without a spec: custom model code is refused unless the spec is set or
explicitly set to "none"; transformers' own model classes need no spec."""

import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.utils import remote_code_identity as rci


class _Custom(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)


_Custom.__module__ = "transformers_modules.ml4_ckpt.modeling_ml4"


def _builtin():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(n_layer=1, n_embd=8, n_head=2, vocab_size=16, n_positions=8)
    return transformers.GPT2LMHeadModel(cfg)


class TestSpecMissing:
    @pytest.mark.parametrize(
        "kind,env,refused",
        [
            ("custom", None, True),
            ("custom", "", True),
            ("custom", "none", False),
            ("builtin", None, False),
        ],
        ids=["custom-unset", "custom-empty", "custom-explicit-none", "builtin-unset"],
    )
    def test_check_files(self, monkeypatch, kind, env, refused):
        if env is None:
            monkeypatch.delenv(rci.ENV, raising=False)
        else:
            monkeypatch.setenv(rci.ENV, env)
        torch.manual_seed(0)
        model = _Custom() if kind == "custom" else _builtin()
        if refused:
            with pytest.raises(rci.RemoteCodeIdentityError, match=rci.ENV):
                rci.check_files(model)
        else:
            assert rci.check_files(model) is None
