"""Runtime identity of trust_remote_code model code, and a runtime proof that it runs the expected paths.

Required for custom model code: a model whose class is not one of transformers' own (trust_remote_code, or any
other package) is refused when ``SKYRL_REMOTE_CODE_SPEC`` is unset or empty. ``SKYRL_REMOTE_CODE_SPEC=none``
runs it unchecked, by explicit choice. Models built from transformers' own classes need no spec.
``SKYRL_REMOTE_CODE_SPEC`` names a JSON file:

    {"files": {"modeling_ml4.py": "<sha256>", ...},    # the frozen code, by file name
     "fail_dir": "<dir>",                              # a failure also writes <dir>/TUPLE_FAIL_<host>-rank<r>
     "proof_dir": "<dir>",                             # proof-<host>-rank<r>.json after the first training step
     "expect_fp8_latent": true,                        # fp8_ds_mla_latent runs once per KV-producing layer per forward
     "expect_r3_recompute": true}                      # checkpoint recompute routes with explicitly passed replay ids

``check_files(model)``: after the model is built, every file named in ``files`` is resolved the way the
runtime resolves it (the module of that name in the package the model class was loaded from, e.g. the
transformers_modules copy of a trust_remote_code checkpoint dir), and its sha256 must equal the spec.

``install_proof(model)``: counts calls of the modeling module's ``fp8_ds_mla_latent`` and of the MoE
router's ``route`` (with or without an explicit ``replay`` argument, in the forward or in the backward
recompute) and the model forwards. ``check_proof()`` after the first training step: every forward ran
``fp8_ds_mla_latent`` once per KV-producing attention layer, and every route in the backward recompute
received the replay ids explicitly (none read the router attribute at recompute time).
Any failure raises ``RemoteCodeIdentityError`` after writing the fail marker.
"""

import hashlib
import inspect
import json
import os
import socket
import sys
from collections import Counter
from typing import Optional

import torch
from torch import nn

ENV = "SKYRL_REMOTE_CODE_SPEC"
#: explicit opt-out value of ``ENV``
UNCHECKED = "none"


class RemoteCodeIdentityError(RuntimeError):
    pass


def _spec() -> Optional[dict]:
    p = os.environ.get(ENV)
    if not p or p == UNCHECKED:
        return None
    with open(p) as f:
        return json.load(f)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _fail(spec: dict, msg: str) -> None:
    d = spec.get("fail_dir")
    if d:
        tag = f"{socket.gethostname().split('.')[0]}-rank{_rank()}"
        with open(os.path.join(d, f"TUPLE_FAIL_{tag}"), "a") as f:
            f.write(f"remote code: {msg}\n")
    raise RemoteCodeIdentityError(msg)


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _base(model: nn.Module) -> nn.Module:
    m = model
    for _ in range(8):  # PEFT / wrappers: PeftModel.base_model.model, HFModelWrapper.model
        inner = getattr(m, "base_model", None)
        inner = getattr(inner, "model", inner) if inner is not None else getattr(m, "model", None)
        if not isinstance(inner, nn.Module) or inner is m:
            break
        if type(m).__module__.split(".")[0] in ("peft", "skyrl"):
            m = inner
        else:
            break
    return m


def model_package(model: nn.Module) -> str:
    return type(_base(model)).__module__.rpartition(".")[0]


def _custom_code(model: nn.Module) -> bool:
    mod = type(_base(model)).__module__
    return not (mod == "transformers" or mod.startswith("transformers."))


def check_files(model: nn.Module) -> Optional[dict]:
    if not os.environ.get(ENV) and _custom_code(model):
        cls = type(_base(model))
        raise RemoteCodeIdentityError(
            f"{cls.__name__} is custom model code ({cls.__module__}) and {ENV} is not set: set it to the frozen "
            f"spec file, or to {UNCHECKED!r} to run this code unchecked"
        )
    spec = _spec()
    if spec is None:
        return None
    pkg = model_package(model)
    got, bad = {}, []
    # every module the runtime imported from the model's package (transformers copies only the files the
    # model imports) must be a frozen file with the frozen sha256
    loaded = sorted(n for n, m in list(sys.modules.items()) if pkg and n.startswith(pkg + ".") and m is not None)
    if not loaded:
        bad.append(f"no module of the model package {pkg!r} is loaded")
    for n in loaded:
        path = inspect.getsourcefile(sys.modules[n])
        name = os.path.basename(path or "")
        if name not in spec["files"]:
            bad.append(f"{n} ({path}) is not a frozen file")
            continue
        sha = _sha(path)
        got[name] = {"path": path, "sha256": sha}
        if sha != spec["files"][name]:
            bad.append(f"{name}: {path} sha256 {sha[:16]} != frozen {spec['files'][name][:16]}")
    for cls in (type(_base(model)), type(getattr(_base(model), "config", None))):
        if cls is type(None):
            continue
        f = os.path.basename(inspect.getsourcefile(cls) or "")
        if f not in spec["files"]:
            bad.append(f"{cls.__name__} is defined in {inspect.getsourcefile(cls)}, not in a frozen file")
    if bad:
        _fail(spec, "; ".join(bad))
    return got


class _Proof:
    def __init__(self, spec: dict, model: nn.Module):
        self.spec, self.counts, self.done = spec, Counter(), False
        base = _base(model)
        modeling = sys.modules[type(base).__module__]
        self.kv_layers = sum(
            1
            for m in base.modules()
            if hasattr(m, "shares_kv") and hasattr(m, "kv_a_proj_with_mqa") and not m.shares_kv
        )
        counts = self.counts

        def in_backward() -> bool:
            return torch._C._current_graph_task_id() != -1

        fn = getattr(modeling, "fp8_ds_mla_latent", None)
        if fn is not None:

            def fp8_ds_mla_latent(*a, **k):
                counts["fp8_latent_backward" if in_backward() else "fp8_latent_forward"] += 1
                return fn(*a, **k)

            modeling.fp8_ds_mla_latent = fp8_ds_mla_latent
        routers = [m for m in base.modules() if hasattr(type(m), "replay_topk_ids") and hasattr(m, "route")]
        if routers:
            cls = type(routers[0])
            route = cls.route

            def route_counted(self_, x, *a, **k):
                explicit = len(a) > 0 or "replay" in k
                replay = (a[0] if a else k.get("replay")) if explicit else self_.replay_topk_ids
                tensor = isinstance(replay, torch.Tensor)
                where = "recompute" if in_backward() else "forward"
                how = "explicit" if explicit and tensor else ("attr" if tensor else "none")
                counts[f"route_{where}_{how}"] += 1
                return route(self_, x, *a, **k)

            cls.route = route_counted
        base.register_forward_pre_hook(lambda *_: counts.__setitem__("forwards", counts["forwards"] + 1))

    def check(self) -> dict:
        c, spec = dict(self.counts), self.spec
        rec = {"host": socket.gethostname().split(".")[0], "rank": _rank(), "kv_layers": self.kv_layers, "counts": c}
        bad = []
        fw = c.get("forwards", 0)
        if spec.get("expect_fp8_latent"):
            want = fw * self.kv_layers
            if self.kv_layers == 0 or c.get("fp8_latent_forward", 0) != want:
                bad.append(
                    f"fp8_ds_mla_latent ran {c.get('fp8_latent_forward', 0)} times in {fw} forwards, "
                    f"want {want} ({self.kv_layers} KV layers)"
                )
        if spec.get("expect_r3_recompute"):
            if c.get("route_recompute_explicit", 0) == 0:
                bad.append("no router call in the backward recompute received the replay ids explicitly")
            n = c.get("route_recompute_attr", 0) + c.get("route_recompute_none", 0)
            if n:
                bad.append(f"{n} router calls in the backward recompute did not receive the replay ids explicitly")
        rec["ok"] = not bad
        rec["errors"] = bad
        d = spec.get("proof_dir")
        if d:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"proof-{rec['host']}-rank{rec['rank']}.json"), "w") as f:
                json.dump(rec, f, indent=1, sort_keys=True)
        if bad:
            _fail(spec, "runtime proof: " + "; ".join(bad))
        return rec


_PROOF: Optional[_Proof] = None


def install_proof(model: nn.Module) -> None:
    global _PROOF
    spec = _spec()
    if _PROOF is not None:  # one trained model per process; a second wrapper would count twice
        return
    if spec is not None and (spec.get("expect_fp8_latent") or spec.get("expect_r3_recompute")):
        _PROOF = _Proof(spec, model)


def check_proof() -> Optional[dict]:
    """After a training forward_backward: evaluate once; later calls are no-ops."""
    if _PROOF is None or _PROOF.done:
        return None
    _PROOF.done = True
    return _PROOF.check()
