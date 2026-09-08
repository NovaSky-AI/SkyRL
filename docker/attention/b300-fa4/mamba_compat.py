"""Preserve the validated Mamba2 path with FA4's newer TVM FFI dependency."""

import hashlib
import importlib.metadata
import json

VERSION = "2.3.2.post1+cu.13.0.torch.2.11"
EAGER_IMPORT = b"from mamba_ssm.modules.mamba3 import Mamba3\n"


def prepare_mamba():
    distribution = importlib.metadata.distribution("mamba-ssm")
    if distribution.version != VERSION:
        raise RuntimeError(f"Requalify Mamba compatibility for {distribution.version}")
    path = distribution.locate_file("mamba_ssm/__init__.py")
    original = path.read_bytes()
    if original.count(EAGER_IMPORT) != 1:
        raise RuntimeError("Expected the pinned Mamba3 eager import before image preparation")
    updated = original.replace(EAGER_IMPORT, b"")
    path.write_bytes(updated)
    return {
        "mamba-ssm": distribution.version,
        "path": str(path),
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "prepared_sha256": hashlib.sha256(updated).hexdigest(),
        "change": "Removed unused Mamba3 eager import; Mamba2 remains available",
    }


if __name__ == "__main__":
    print(json.dumps(prepare_mamba(), indent=2))
