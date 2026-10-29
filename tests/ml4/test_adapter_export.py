"""FSDP in-memory adapter sync (``lora.sync_mode=memory``) on ML4.

Runs the FSDP worker's own adapter state (``_lora_adapter_state``, ``lora_adapter_config``)
through ``FsdpLoraAdapterSource`` and checks what would go on the wire: every key is in
ML4's vLLM adapter contract, lora_B carries ``lora_alpha / r`` folded exactly as the
Megatron source folds it, the config says ``lora_alpha == r``, and vLLM's math on the
folded adapter (scale 1) equals PEFT's LoRA delta.
"""

import json
import re
from types import SimpleNamespace

import torch


def test_memory_sync_stream_contract_and_alpha_fold(ml4_model_dir, vllm_weight_transfer):
    from conftest import LORA_ALPHA, LORA_RANK, load_ml4_wrapper
    from ml4_train.lora import (
        ADAPTER_KEY_REGEX,
        LORA_TARGET_REGEX,
        adapter_module_names,
    )
    from peft.tuners.lora import LoraLayer

    from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
        collect_lora_params,
        lora_adapter_config,
    )
    from skyrl.backends.skyrl_train.weight_sync.sources import FsdpLoraAdapterSource
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    wrapper, _ = load_ml4_wrapper(ml4_model_dir)
    peft_model = wrapper.model
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if ".lora_B." in n:
                p.copy_(torch.randn(p.shape, generator=g))
    worker = SimpleNamespace(model=wrapper, _is_multimodal_lm_only=False)
    raw = collect_lora_params(peft_model)  # what the disk path writes
    source = FsdpLoraAdapterSource(
        dtype=torch.bfloat16,
        adapter_state=lambda: FSDPPolicyWorkerBase._lora_adapter_state(worker),
        adapter_config=lora_adapter_config(peft_model),
    )
    source.set_lora_name("ml4")
    source.prepare()
    target = source.receive_target
    meta = source.metadata()
    sent = dict(iter(source))

    # Names: the PEFT keys unchanged, all inside the contract, every LoRA module present.
    assert [m.name for m in meta] == list(sent) and set(sent) == set(raw)
    assert all(re.fullmatch(ADAPTER_KEY_REGEX, k) for k in sent)
    lora_modules = {
        n.removeprefix("base_model.model.") for n, m in peft_model.named_modules() if isinstance(m, LoraLayer)
    }
    assert adapter_module_names(sent) == lora_modules
    assert all(t.dtype == torch.bfloat16 for t in sent.values())

    # Config: alpha folded, regex targets kept whole, JSON-serializable (it rides an RPC).
    cfg = target["adapter_config"]
    assert (cfg["r"], cfg["lora_alpha"]) == (LORA_RANK, LORA_RANK)
    assert cfg["target_modules"] == LORA_TARGET_REGEX
    assert target["aliases"] == {} and target["lora_name"] == "ml4"
    json.dumps(target)

    # Megatron's formula: lora_B * (alpha / r) in fp32, cast back; then the wire dtype.
    scale = LORA_ALPHA / LORA_RANK
    for k, t in sent.items():
        want = raw[k].float() * scale if ".lora_B." in k else raw[k]
        torch.testing.assert_close(t, want.to(torch.bfloat16), atol=0, rtol=0, msg=k)

    # vLLM applies x @ A^T @ B^T * (lora_alpha / r = 1); PEFT applies its own scaling.
    name = "language_model.model.layers.1.mlp.experts.0.down_proj"
    layer = peft_model.get_submodule(f"base_model.model.{name}")
    x = torch.randn(3, layer.in_features, generator=g)
    with torch.no_grad():
        peft_delta = layer(x) - layer.base_layer(x)
    a, b = (sent[f"base_model.model.{name}.lora_{s}.weight"].float() for s in "AB")
    torch.testing.assert_close(x @ a.T @ b.T, peft_delta, atol=5e-2, rtol=2e-2)


def test_memory_sync_stream_holds_one_tensor(ml4_model_dir, vllm_weight_transfer):
    """``prepare`` copies nothing, and the source keeps no reference to what it already yielded.

    Peak memory of the export is therefore the transport's bucket plus one tensor: no
    full-adapter copy, on the host or on the device.
    """
    import weakref

    from conftest import load_ml4_wrapper

    from skyrl.backends.skyrl_train.distributed.fsdp_utils import lora_adapter_config
    from skyrl.backends.skyrl_train.weight_sync.sources import FsdpLoraAdapterSource
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    wrapper, _ = load_ml4_wrapper(ml4_model_dir)
    worker = SimpleNamespace(model=wrapper, _is_multimodal_lm_only=False)
    source = FsdpLoraAdapterSource(
        dtype=torch.bfloat16,
        adapter_state=lambda: FSDPPolicyWorkerBase._lora_adapter_state(worker),
        adapter_config=lora_adapter_config(wrapper.model),
    )
    source.set_lora_name("ml4")
    source.prepare()
    meta = source.metadata()
    assert all(t.is_meta for t in source._prepared_state()[0].values())

    params = {p.data_ptr() for p in wrapper.model.parameters()}
    yielded = []
    for name, tensor in source:
        assert tensor.data_ptr() not in params  # a copy in the wire dtype, never the parameter
        assert all(ref() is None for ref in yielded), f"{name}: an earlier tensor is still alive"
        yielded.append(weakref.ref(tensor))
        del tensor
    assert len(yielded) == len(meta) and source._shards is None
