"""remote_code_identity on the real ML4 model code, loaded the way trust_remote_code loads it (a package
holding the model dir's *.py copies). ML4_TREE_NEW / ML4_TREE_STALE: two ML4 model-code trees
(ml4_train/ + tests/), the current one and an older model-dir copy."""

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

from skyrl.backends.skyrl_train.utils import remote_code_identity as rci

TREES = {k: os.environ.get(f"ML4_TREE_{k.upper()}") for k in ("new", "stale")}
pytestmark = pytest.mark.skipif(not all(TREES.values()), reason="ML4_TREE_NEW / ML4_TREE_STALE not set")
CODE = ("configuration_ml4.py", "modeling_ml4.py", "ops.py", "loading.py", "lora.py", "qat.py")


def shas(tree: str) -> dict:
    return {f: hashlib.sha256(Path(tree, "ml4_train", f).read_bytes()).hexdigest() for f in CODE}


def load(tree: str, tmp: Path, name: str):
    d = tmp / "transformers_modules" / name
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("")
    for f in CODE:
        shutil.copyfile(Path(tree, "ml4_train", f), d / f)
    sys.path.insert(0, str(tmp / "transformers_modules"))
    try:
        return importlib.import_module(f"{name}.modeling_ml4"), importlib.import_module(f"{name}.configuration_ml4")
    finally:
        sys.path.pop(0)


def tiny_params(tree: str) -> dict:
    s = importlib.util.spec_from_file_location("ml4_fake_ckpt", Path(tree, "tests", "fake_ckpt.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    p = copy.deepcopy(m.TINY_PARAMS)
    p["dsa"]["fp8_kv_cache"] = True
    p["kv_lora_rank"] = 512  # fp8_ds_mla layout
    return p


@pytest.fixture
def spec(tmp_path, monkeypatch):
    def make(files: dict) -> Path:
        p = tmp_path / "spec.json"
        p.write_text(
            json.dumps(
                {
                    "files": files,
                    "fail_dir": str(tmp_path),
                    "proof_dir": str(tmp_path / "proof"),
                    "expect_fp8_latent": True,
                    "expect_r3_recompute": True,
                }
            )
        )
        monkeypatch.setenv(rci.ENV, str(p))
        monkeypatch.setattr(rci, "_PROOF", None)
        return p

    return make


def train_step(modeling, configuration, tree: str, reentrant: bool):
    torch.manual_seed(0)
    model = modeling.ML4ForCausalLM(configuration.config_from_params(tiny_params(tree))).train()
    rci.check_files(model)
    rci.install_proof(model)
    model.gradient_checkpointing_enable({"use_reentrant": reentrant})
    T = 12
    routers = [m for m in model.modules() if hasattr(type(m), "replay_topk_ids") and hasattr(m, "route")]
    assert routers
    g = torch.Generator().manual_seed(1)
    for r in routers:
        r.replay_topk_ids = torch.randint(0, r.gate.weight.shape[0], (T, r.top_k), generator=g)
    tok = torch.randint(0, 300, (1, T), generator=g)
    logits = model(tok).logits
    logits.float().pow(2).mean().backward()
    return rci.check_proof()


@pytest.mark.parametrize("reentrant", [False, True], ids=["non-reentrant", "reentrant"])
def test_frozen_code_passes(tmp_path, spec, reentrant):
    spec(shas(TREES["new"]))
    modeling, configuration = load(TREES["new"], tmp_path, f"new_{int(reentrant)}")
    rec = train_step(modeling, configuration, TREES["new"], reentrant)
    c = rec["counts"]
    assert rec["ok"] and rec["kv_layers"] > 0
    assert c["fp8_latent_forward"] == c["forwards"] * rec["kv_layers"]
    assert c["route_recompute_explicit"] > 0 and not c.get("route_recompute_attr") and not c.get("route_recompute_none")
    assert json.loads((tmp_path / "proof" / f"proof-{rec['host']}-rank{rec['rank']}.json").read_text())["ok"]


def test_stale_copy_fails_identity(tmp_path, spec):
    """The spec hashes the current code while the trainer imports an older model-dir copy."""
    spec(shas(TREES["new"]))
    modeling, configuration = load(TREES["stale"], tmp_path, "stale_id")
    model = modeling.ML4ForCausalLM(configuration.config_from_params(tiny_params(TREES["stale"])))
    with pytest.raises(rci.RemoteCodeIdentityError, match="modeling_ml4.py"):
        rci.check_files(model)
    assert list(tmp_path.glob("TUPLE_FAIL_*"))


def test_stale_code_fails_runtime_proof(tmp_path, spec):
    """Even with matching hashes, the stale code neither simulates the FP8 latent cache nor passes the R3
    ids into the checkpointed layer."""
    spec(shas(TREES["stale"]))
    modeling, configuration = load(TREES["stale"], tmp_path, "stale_proof")
    with pytest.raises(rci.RemoteCodeIdentityError) as e:
        train_step(modeling, configuration, TREES["stale"], reentrant=False)
    assert "fp8_ds_mla_latent ran 0 times" in str(e.value)
    assert "did not receive the replay ids explicitly" in str(e.value) or "no router call" in str(e.value)
