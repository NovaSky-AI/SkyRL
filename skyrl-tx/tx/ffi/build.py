import os
import subprocess
import sys
import tempfile
from pathlib import Path

CUTLASS_REPO = "https://github.com/NVIDIA/cutlass.git"
CUTLASS_TAG = "v4.3.5"


def get_cutlass_dir(tmpdir):
    if cutlass_dir := os.environ.get("CUTLASS_DIR"):
        return Path(cutlass_dir)

    cutlass_dir = Path(tmpdir) / "cutlass"
    print(f"Cloning CUTLASS {CUTLASS_TAG}...")
    subprocess.run(
        ["git", "clone", "--depth=1", f"--branch={CUTLASS_TAG}", CUTLASS_REPO, str(cutlass_dir)],
        check=True,
    )
    return cutlass_dir


def build_ragged_dot():
    try:
        import jaxlib
        jax_include_dir = Path(jaxlib.__file__).parent / "include"
    except ImportError:
        print("jaxlib not installed, skipping ragged_dot_ffi build", file=sys.stderr)
        return

    nvcc_bin = os.environ.get("NVCC_BIN", "nvcc")
    nvcc_arch = os.environ.get("NVCC_ARCH", "90a")

    try:
        subprocess.run([nvcc_bin, "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"nvcc not found at {nvcc_bin}, skipping ragged_dot_ffi build", file=sys.stderr)
        return

    ffi_dir = Path(__file__).parent
    source_file = ffi_dir / "ragged_dot_ffi.cu"
    output_file = ffi_dir / "libragged_dot_ffi.so"

    with tempfile.TemporaryDirectory() as tmpdir:
        cutlass_dir = get_cutlass_dir(tmpdir)

        cmd = [
            nvcc_bin, "-O3", "-std=c++17", f"-arch=sm_{nvcc_arch}",
            "--expt-relaxed-constexpr",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DCUTLASS_ENABLE_GDC_FOR_SM90=1",
            "-shared", "-Xcompiler", "-fPIC",
            f"-I{jax_include_dir}",
            f"-I{cutlass_dir}/include",
            f"-I{cutlass_dir}/tools/util/include/",
            str(source_file), "-o", str(output_file),
        ]

        print(f"Building {output_file}...")
        subprocess.run(cmd, check=True)
        print(f"Built {output_file}")


from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CudaBuildHook(BuildHookInterface):
    PLUGIN_NAME = "cuda_build"

    def initialize(self, version, build_data):
        if self.target_name == "wheel":
            build_ragged_dot()
