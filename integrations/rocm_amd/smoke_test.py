#!/usr/bin/env python3
"""Compatibility smoke test for SkyRL Megatron on AMD ROCm GPUs.

Run inside a ROCm container with GPUs (e.g. rocm/primus:v26.4):

    python integrations/rocm_amd/smoke_test.py --all
    python integrations/rocm_amd/smoke_test.py --phase bridge --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


@dataclass
class CheckResult:
    name: str
    phase: str
    passed: bool
    detail: str = ""
    error: str | None = None


@dataclass
class SmokeReport:
    started_at: str
    hostname: str = field(default_factory=lambda: os.uname().nodename)
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] ({result.phase}) {result.name}: {result.detail or result.error or ''}")

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _try_import(module: str, attr: str | None = None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        if attr:
            getattr(mod, attr)
        version = getattr(mod, "__version__", None)
        return True, f"version={version}" if version else "ok"
    except Exception as exc:  # noqa: BLE001 - exploration script
        return False, f"{type(exc).__name__}: {exc}"


def _run_check(report: SmokeReport, phase: str, name: str, fn: Callable[[], tuple[bool, str]]) -> None:
    try:
        ok, detail = fn()
        report.add(
            CheckResult(
                name=name, phase=phase, passed=ok, detail=detail if ok else "", error=detail if not ok else None
            )
        )
    except Exception as exc:  # noqa: BLE001
        report.add(
            CheckResult(
                name=name,
                phase=phase,
                passed=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )


def phase_env(report: SmokeReport) -> None:
    def torch_hip() -> tuple[bool, str]:
        import torch

        hip = getattr(torch.version, "hip", None)
        cuda = torch.cuda.is_available()
        count = torch.cuda.device_count() if cuda else 0
        name = torch.cuda.get_device_name(0) if count else "n/a"
        return bool(hip), f"torch={torch.__version__} hip={hip} devices={count} name={name}"

    def ray() -> tuple[bool, str]:
        import ray

        return True, f"ray={ray.__version__}"

    _run_check(report, "env", "torch_rocm", torch_hip)
    _run_check(report, "env", "ray", ray)

    def gpu_support() -> tuple[bool, str]:
        from gpu_support import detect_gpu_info

        info = detect_gpu_info()
        return info.supported, info.detail

    _run_check(report, "env", "gpu_support", gpu_support)


def phase_megatron_core(report: SmokeReport) -> None:
    checks = [
        ("megatron.core", None),
        ("megatron.core.parallel_state", None),
        ("megatron.core.optimizer", None),
        ("megatron.core.distributed", None),
    ]
    for module, attr in checks:
        _run_check(report, "megatron_core", module, lambda m=module, a=attr: _try_import(m, a))


def phase_transformer_engine(report: SmokeReport) -> None:
    def te() -> tuple[bool, str]:
        ok, detail = _try_import("transformer_engine")
        if not ok:
            return ok, detail
        import transformer_engine.pytorch as te_pytorch  # noqa: F401

        return True, detail + " + transformer_engine.pytorch"

    _run_check(report, "transformer_engine", "transformer_engine", te)


def phase_megatron_bridge(report: SmokeReport, model: str | None) -> None:
    def bridge_import() -> tuple[bool, str]:
        ok, detail = _try_import("megatron.bridge", "AutoBridge")
        return ok, detail

    def bridge_from_hf(model_path: str) -> tuple[bool, str]:
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
        provider = bridge.to_megatron_provider()
        tp = getattr(provider, "tensor_model_parallel_size", "?")
        return True, f"bridge ok for {model_path}, provider.tp={tp}"

    _run_check(report, "megatron_bridge", "import", bridge_import)
    if model:
        _run_check(report, "megatron_bridge", f"from_hf({model})", lambda: bridge_from_hf(model))


def phase_skyrl_imports(report: SmokeReport) -> None:
    skyrl_checks = [
        ("skyrl", None),
        ("skyrl.backends.skyrl_train.workers.worker", "PolicyWorkerBase"),
        ("skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy", "MegatronStrategy"),
    ]
    for module, attr in skyrl_checks:
        _run_check(report, "skyrl", module, lambda m=module, a=attr: _try_import(m, a))

    def megatron_worker() -> tuple[bool, str]:
        # This import pulls megatron-bridge + TE; expected to fail if bridge unavailable.
        from skyrl.backends.skyrl_train.workers.megatron import (
            megatron_worker,  # noqa: F401
        )

        return True, "megatron_worker imported"

    _run_check(report, "skyrl", "megatron_worker", megatron_worker)


def phase_dist_init(report: SmokeReport) -> None:
    def single_gpu_dist() -> tuple[bool, str]:
        import torch
        import torch.distributed as dist

        if not torch.cuda.is_available():
            return False, "no cuda/hip device"

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

        torch.cuda.set_device(0)
        if dist.is_initialized():
            dist.destroy_process_group()

        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        dist.destroy_process_group()
        return rank == 0, f"nccl init ok rank={rank}"

    _run_check(report, "distributed", "single_gpu_nccl", single_gpu_dist)


def phase_vllm(report: SmokeReport) -> None:
    _run_check(report, "vllm", "import", lambda: _try_import("vllm"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        action="append",
        choices=(
            "env",
            "megatron_core",
            "transformer_engine",
            "megatron_bridge",
            "skyrl",
            "distributed",
            "vllm",
            "all",
        ),
        default=None,
        help="Run one or more phases (repeat flag). Default: all",
    )
    parser.add_argument("--all", action="store_true", help="Run all phases (alias for --phase all)")
    parser.add_argument("--model", default=None, help="HF model for megatron-bridge from_hf test")
    parser.add_argument("--report", default=None, help="Write JSON report to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = SmokeReport(started_at=datetime.now(UTC).isoformat())

    phases = {
        "env": lambda: phase_env(report),
        "megatron_core": lambda: phase_megatron_core(report),
        "transformer_engine": lambda: phase_transformer_engine(report),
        "megatron_bridge": lambda: phase_megatron_bridge(report, args.model),
        "skyrl": lambda: phase_skyrl_imports(report),
        "distributed": lambda: phase_dist_init(report),
        "vllm": lambda: phase_vllm(report),
    }

    if args.all or not args.phase or args.phase == ["all"]:
        selected = list(phases.keys())
    else:
        selected = args.phase
    for name in selected:
        phases[name]()

    summary = {
        "started_at": report.started_at,
        "hostname": report.hostname,
        "passed": report.passed,
        "total": len(report.checks),
        "failures": [asdict(c) for c in report.checks if not c.passed],
        "checks": [asdict(c) for c in report.checks],
    }

    print("\n=== Summary ===")
    print(
        json.dumps(
            {"passed": summary["passed"], "total": summary["total"], "failures": len(summary["failures"])}, indent=2
        )
    )

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote report to {args.report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
