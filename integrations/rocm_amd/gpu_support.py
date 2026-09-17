#!/usr/bin/env python3
"""AMD Instinct GPU detection and ROCm ISA helpers for SkyRL on ROCm."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

# CDNA3: MI300X, MI325X -> gfx942
# CDNA4: MI355X -> gfx950
# See https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
SUPPORTED_GPUS: dict[str, str] = {
    "MI300X": "gfx942",
    "MI325X": "gfx942",
    "MI355X": "gfx950",
}

DEFAULT_PYTORCH_ROCM_ARCH = "gfx942;gfx950"


@dataclass(frozen=True)
class GpuInfo:
    product_names: tuple[str, ...]
    gfx_archs: tuple[str, ...]
    supported: bool
    detail: str


def _normalize_gfx(name: str) -> str:
    return name.split(":")[0].strip().lower()


def _extract_product_names(text: str) -> list[str]:
    """Extract AMD Instinct MI product names without applying the allowlist."""
    return [f"MI{match.upper()}" for match in re.findall(r"\bMI\s*(\d{3}X)\b", text, re.IGNORECASE)]


def _gfx_from_torch() -> list[str]:
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    archs: list[str] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        gcn = getattr(props, "gcnArchName", None)
        if gcn:
            archs.append(_normalize_gfx(str(gcn)))
    return archs


def _product_names_from_rocm_smi() -> list[str]:
    if not shutil_which("rocm-smi"):
        return []
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []

    return _extract_product_names(out.stdout)


def shutil_which(cmd: str) -> str | None:
    import shutil

    return shutil.which(cmd)


def _product_names_from_torch() -> list[str]:
    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    names: list[str] = []
    for idx in range(torch.cuda.device_count()):
        names.extend(_extract_product_names(torch.cuda.get_device_name(idx)))
    return names


def detect_gpu_info() -> GpuInfo:
    products = list(dict.fromkeys(_product_names_from_rocm_smi() + _product_names_from_torch()))
    gfx_archs = list(dict.fromkeys(_gfx_from_torch()))

    if products:
        unsupported = [p for p in products if p not in SUPPORTED_GPUS]
        if unsupported:
            return GpuInfo(
                product_names=tuple(products),
                gfx_archs=tuple(gfx_archs),
                supported=False,
                detail=f"unsupported GPU product(s): {', '.join(unsupported)}",
            )
        expected = {SUPPORTED_GPUS[p] for p in products}
        if gfx_archs and not set(gfx_archs).issubset(expected):
            return GpuInfo(
                product_names=tuple(products),
                gfx_archs=tuple(gfx_archs),
                supported=False,
                detail=f"gfx arch mismatch: saw {gfx_archs}, expected subset of {sorted(expected)}",
            )
        return GpuInfo(
            product_names=tuple(products),
            gfx_archs=tuple(gfx_archs),
            supported=True,
            detail=f"supported: {', '.join(products)} ({', '.join(gfx_archs) or 'gfx unknown'})",
        )

    if gfx_archs:
        allowed = set(SUPPORTED_GPUS.values())
        if set(gfx_archs).issubset(allowed):
            return GpuInfo(
                product_names=(),
                gfx_archs=tuple(gfx_archs),
                supported=True,
                detail=f"supported gfx: {', '.join(gfx_archs)}",
            )
        return GpuInfo(
            product_names=(),
            gfx_archs=tuple(gfx_archs),
            supported=False,
            detail=f"unsupported gfx arch(s): {gfx_archs}; supported: {sorted(allowed)}",
        )

    return GpuInfo(
        product_names=(),
        gfx_archs=(),
        supported=False,
        detail="no ROCm GPU detected (torch.cuda unavailable or no supported arch)",
    )


def pytorch_rocm_arch_for_build() -> str:
    """Return PYTORCH_ROCM_ARCH covering the supported CDNA3 and CDNA4 GPUs."""
    return DEFAULT_PYTORCH_ROCM_ARCH


def check_gpu_supported(strict: bool = True) -> GpuInfo:
    """Return GPU info; exit non-zero when strict and GPU is missing or unsupported."""
    info = detect_gpu_info()
    if strict and not info.supported:
        raise SystemExit(f"Unsupported or undetected ROCm GPU: {info.detail}")
    return info


def main() -> int:
    info = detect_gpu_info()
    print(f"product_names={list(info.product_names)}")
    print(f"gfx_archs={list(info.gfx_archs)}")
    print(f"supported={info.supported}")
    print(f"detail={info.detail}")
    print(f"PYTORCH_ROCM_ARCH={pytorch_rocm_arch_for_build()}")
    return 0 if info.supported else 1


if __name__ == "__main__":
    raise SystemExit(main())
