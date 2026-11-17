"""CPU tests for ``docker/rocm_requirements.py``, which builds the install list of the AMD ROCm image.

uv run --isolated --extra tinker --extra jax --extra dev pytest tests/utils/test_rocm_requirements.py
"""

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rocm_requirements():
    spec = importlib.util.spec_from_file_location("rocm_requirements", _REPO_ROOT / "docker" / "rocm_requirements.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_drops_excluded_packages_and_keeps_the_rest(rocm_requirements):
    export = [
        "-e ./skyrl-gym",
        "# a comment",
        "",
        "torch==2.14.0 ; sys_platform == 'linux'",
        "Flash_Attn==2.8.3.post1",
        "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM@ed9c2c8c1cf4a5438922ae8d215181e7c8b9e0fe",
        "nvidia-resiliency-ext @ git+https://github.com/NVIDIA/nvidia-resiliency-ext@6c5f2a13c7688d92",
        "nvidia-ml-py==13.0.0",
        "torchdata==0.11.0",
    ]

    kept, dropped = rocm_requirements.filter_requirements(export, ["torch", "flash-attn"])

    assert dropped == ["torch", "flash-attn"]
    assert kept == [export[0], export[5], export[6], export[7], export[8]]


@pytest.mark.parametrize(
    "line",
    [
        "nvidia-cublas==13.1.0.3",
        "cuda-bindings==13.0.3",
        "some-kernel-cu13==1.0",
        "flashinfer-python==0.4",
        "nixl==1.0",
    ],
)
def test_filter_fails_on_a_cuda_package_nobody_listed(rocm_requirements, line):
    with pytest.raises(ValueError, match="CUDA-looking packages"):
        rocm_requirements.filter_requirements(["numpy==2.0.0", line], ["torch"])


def test_excluded_file_covers_every_cuda_package_of_the_megatron_lock(rocm_requirements):
    """The checked-in deny list and the lock agree: the Megatron export filters without an alarm."""
    import shutil
    import subprocess

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    export = subprocess.run(
        [uv, "export", "--frozen", "--no-hashes", "--no-annotate", "--no-emit-project", "--extra", "megatron", "-q"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if export.returncode != 0:
        pytest.skip(f"uv export failed: {export.stderr.strip()[:200]}")
    excluded = rocm_requirements.read_excluded(_REPO_ROOT / "docker" / "rocm-excluded-packages.txt")

    kept, dropped = rocm_requirements.filter_requirements(export.stdout.splitlines(), excluded)

    kept_names = {rocm_requirements.req_name(line) for line in kept}
    assert {"megatron-core", "megatron-bridge", "flash-linear-attention", "ray"} <= kept_names
    assert not kept_names & {"torch", "vllm", "triton", "transformer-engine", "flash-attn"}
    assert {"torch", "vllm", "transformer-engine"} <= set(dropped)


def test_git_pins_reads_name_and_commit(rocm_requirements):
    pins = rocm_requirements.git_pins(
        [
            "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM@ed9c2c8c1cf4a5438922ae8d215181e7c8b9e0fe ; sys_platform == 'linux'",
            "Megatron_Bridge @ git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@7171584bd7452caf2adfa6afd12fb89cc37fbc64",
            "vllm-router @ https://example.com/vllm_router-0.1.14-cp38-abi3-manylinux_2_35_x86_64.whl",
            "numpy==2.0.0",
        ]
    )

    assert pins == {
        "megatron-core": "ed9c2c8c1cf4a5438922ae8d215181e7c8b9e0fe",
        "megatron-bridge": "7171584bd7452caf2adfa6afd12fb89cc37fbc64",
    }
