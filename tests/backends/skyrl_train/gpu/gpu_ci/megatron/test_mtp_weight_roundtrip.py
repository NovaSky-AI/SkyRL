"""Round-trip fidelity test for the native MTP head weights.

Hypothesis under test: vLLM MTP speculative decoding under SkyRL gives ~0
acceptance (vs ~0.9 when vLLM loads the HF checkpoint directly), and our
in-training ``mtp_loss`` is stuck at ~160 nats (>> uniform ln(V)=12.4). Both
point at the *weights* the MTP head sees being wrong.

SkyRL's weight sync sends ``bridge.export_hf_weights(actor_module)`` to vLLM,
and that same loaded Megatron MTP head drives our decoupled draft replay. So if
the HF -> Megatron load -> Megatron -> HF export round-trip does NOT reproduce
the checkpoint's ``mtp.*`` tensors, that single bug explains *both* symptoms.

This test:
  1. Builds a real Megatron *policy* worker for Qwen3.5-2B the SkyRL way
     (MTP enabled), exactly as training does.
  2. Exports HF-format weights through ``bridge.export_hf_weights`` -- the
     literal source of the weight sync -- and keeps the ``mtp.*`` tensors.
  3. Loads the raw ``mtp.*`` tensors straight from the HF safetensors (what a
     standalone ``vllm serve`` loads, the 0.9-acceptance reference).
  4. Asserts every ``mtp.*`` tensor round-trips (shape + values close).

Run with::
    uv run --isolated --extra megatron --extra dev pytest -s -vvv \
      tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_mtp_weight_roundtrip.py
"""

import glob
import json
import os

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as _megatron_worker_mod,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

MODEL_NAME = "Qwen/Qwen3.5-2B"


class _ProbeMegatronPolicyWorker(MegatronPolicyWorkerBase):
    """Policy worker that exposes the exported ``mtp.*`` HF tensors (the exact
    bytes the weight sync would push to vLLM), moved to CPU for the driver."""

    def probe_export_mtp_weights(self) -> dict:
        from megatron.core.utils import unwrap_model

        out = {}
        all_names = []
        for name, tensor in self.bridge.export_hf_weights(
            self.actor_module,
            show_progress=False,
            conversion_tasks=None,
        ):
            all_names.append(name)
            # HF naming for the native MTP head is top-level ``mtp.*``; be
            # liberal and also catch any nested ``.mtp.`` just in case.
            if name.startswith("mtp.") or ".mtp." in name:
                out[name] = tensor.detach().to(torch.float32).cpu()
            del tensor

        # --- diagnostics: was the MTP head actually built on the Megatron side? ---
        gm = unwrap_model(self.actor_module[0])
        # descend into VL nesting (language_model) like the capture does
        host = gm
        for _ in range(4):
            if getattr(host, "mtp", None) is not None:
                break
            host = getattr(host, "language_model", None)
            if host is None:
                host = gm
                break
        built_mtp = getattr(host, "mtp", None) is not None
        # collect the Megatron-side param names that mention mtp
        megatron_mtp_params = [n for n, _ in gm.named_parameters() if "mtp" in n.lower()]

        return {
            "mtp_tensors": out,
            "total_exported": len(all_names),
            "export_mtp_names": [n for n in all_names if "mtp" in n.lower()],
            "sample_export_names": all_names[:5] + all_names[-5:],
            "config_mtp_num_layers": getattr(getattr(host, "config", None), "mtp_num_layers", "NO_CONFIG"),
            "built_mtp": built_mtp,
            "host_type": type(host).__name__,
            "megatron_mtp_param_names": megatron_mtp_params[:20],
            "n_megatron_mtp_params": len(megatron_mtp_params),
        }


_ProbePolicyWorker = ray.remote(num_gpus=1)(_ProbeMegatronPolicyWorker)


def _make_policy_cfg(model_name: str) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    # TP=1/PP=1 so each exported tensor is whole (no sharded reassembly to muddy
    # the comparison) -- this is the simplest faithful round trip.
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    # This test only round-trips MTP head weights; packing is irrelevant and Qwen3.5's GDN layers
    # cannot sample-pack anyway (the wrapper rejects it -- see the 9B recipe's remove_microbatch_padding=false).
    cfg.trainer.remove_microbatch_padding = False
    # Enable MTP via the high-level knob, exactly as the training run does. The trained head count
    # stays None on purpose: the draft DEPTH (num_speculative_tokens) is inference-only, and the head
    # count is inferred from the checkpoint's HF config by the bridge. (_apply_mtp_config only forces
    # it to 0 when trainer.mtp.enabled is False.)
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = 1
    cfg.trainer.mtp.loss_weight = 0.1
    validate_cfg(cfg)
    assert cfg.trainer.policy.megatron_config.mtp_num_layers is None, (
        "expected mtp_num_layers=None (inferred from the checkpoint) after validate_cfg, got "
        f"{cfg.trainer.policy.megatron_config.mtp_num_layers}"
    )
    return cfg


def _load_hf_mtp_tensors(model_name: str) -> dict:
    """Load every ``mtp.*`` tensor straight from the HF safetensors snapshot."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    local = model_name
    if not os.path.isdir(local):
        local = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])

    index = os.path.join(local, "model.safetensors.index.json")
    if os.path.exists(index):
        weight_map = json.load(open(index))["weight_map"]
        files = sorted({os.path.join(local, f) for k, f in weight_map.items() if "mtp" in k.lower()})
    else:
        files = sorted(glob.glob(os.path.join(local, "*.safetensors")))

    out = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("mtp.") or ".mtp." in key:
                    out[key] = handle.get_tensor(key).to(torch.float32)
    return out


def _suffix(name: str) -> str:
    """Normalize to the ``mtp...`` suffix so exported and raw names align even
    if one carries an extra prefix (e.g. ``model.``)."""
    i = name.find("mtp.")
    return name[i:] if i >= 0 else name


@pytest.mark.megatron
def test_mtp_head_weights_roundtrip(ray_init_fixture):
    cfg = _make_policy_cfg(MODEL_NAME)

    hf = {_suffix(k): v for k, v in _load_hf_mtp_tensors(MODEL_NAME).items()}
    assert hf, "no mtp.* tensors found in the HF checkpoint"

    _orig = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _ProbePolicyWorker
    try:
        policy = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=1,
            cfg=cfg,
        )
        probes = ray.get(policy.async_run_ray_method("pass_through", "probe_export_mtp_weights"))
    finally:
        _megatron_worker_mod.PolicyWorker = _orig

    # --- diagnostics first ---
    p0 = probes[0]
    print("\n===== MTP build / export diagnostics (rank 0) =====")
    print("host_type             :", p0["host_type"])
    print("built_mtp (.mtp attr) :", p0["built_mtp"])
    print("config.mtp_num_layers :", p0["config_mtp_num_layers"])
    print("total exported params :", p0["total_exported"])
    print("n megatron mtp params :", p0["n_megatron_mtp_params"])
    print("megatron mtp param names:", p0["megatron_mtp_param_names"])
    print("export names w/ 'mtp' :", p0["export_mtp_names"])
    print("sample export names   :", p0["sample_export_names"])
    print("==================================================")

    # TP=PP=1 -> a single rank holds the whole MTP head.
    exported = {}
    for rank in probes:
        exported.update({_suffix(k): v for k, v in rank["mtp_tensors"].items()})

    assert p0["built_mtp"], (
        f"MTP head was NOT built on the Megatron side (config.mtp_num_layers="
        f"{p0['config_mtp_num_layers']}, n_megatron_mtp_params={p0['n_megatron_mtp_params']}). "
        "Test config did not enable MTP."
    )
    assert exported, (
        "MTP head IS built on Megatron (n_megatron_mtp_params="
        f"{p0['n_megatron_mtp_params']}) but bridge.export_hf_weights exported NONE of it -> "
        "the weight sync silently omits the MTP head, so vLLM never receives trained MTP weights."
    )

    print(f"\nHF mtp tensors: {len(hf)} | exported mtp tensors: {len(exported)}")
    missing = sorted(set(hf) - set(exported))
    extra = sorted(set(exported) - set(hf))
    print("missing from export:", missing)
    print("extra in export:", extra)

    report = []
    failures = []
    for key in sorted(hf):
        h = hf[key]
        if key not in exported:
            failures.append(f"{key}: MISSING from export")
            continue
        e = exported[key]
        if tuple(e.shape) != tuple(h.shape):
            failures.append(f"{key}: shape {tuple(e.shape)} != HF {tuple(h.shape)}")
            continue
        max_abs = (e - h).abs().max().item()
        denom = h.norm().item() or 1.0
        rel = (e - h).norm().item() / denom
        cos = torch.nn.functional.cosine_similarity(e.flatten(), h.flatten(), dim=0).item()
        report.append((key, tuple(h.shape), max_abs, rel, cos))
        # bf16 round-trip tolerance is loose; a real mismatch is off by orders
        # of magnitude (rel ~ O(1), cos far from 1), not by bf16 epsilon.
        if rel > 0.05 or cos < 0.99:
            failures.append(f"{key}: rel={rel:.4f} cos={cos:.4f} max_abs={max_abs:.4e}")

    print("\n%-62s %-18s %10s %8s %8s" % ("name", "shape", "max_abs", "rel", "cos"))
    for key, shape, max_abs, rel, cos in report:
        print("%-62s %-18s %10.3e %8.4f %8.5f" % (key, str(shape), max_abs, rel, cos))

    assert not failures, "MTP head weight round-trip mismatches:\n  " + "\n  ".join(failures)
