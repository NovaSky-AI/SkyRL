"""Check the installed FA2/FA4 pair without importing CUDA during image builds."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import platform
import re
import sys
from pathlib import Path


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def verify_install(runtime=False, check_dependencies=True):
    distributions = list(importlib.metadata.distributions())
    pair = {}
    for name in ("flash-attn", "flash-attn-4"):
        matches = [dist for dist in distributions if normalize(dist.metadata["Name"]) == name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one installed {name}, found {len(matches)}")
        pair[name] = matches[0]
    combined, companion = pair["flash-attn"], pair["flash-attn-4"]
    provenance = json.loads(combined.read_text("skyrl-provenance.json") or "null")
    if provenance is None:
        raise RuntimeError("flash-attn is missing the combined-wheel provenance")
    build = provenance["build"]
    if combined.version != build["combined_version"] or companion.version != build["companion_version"]:
        raise RuntimeError("The FA2 and FA4 distributions are not a matching release pair")
    if companion.requires != [f"flash-attn=={combined.version}"]:
        raise RuntimeError("FA4 companion must depend on the exact combined wheel")
    root = Path(combined.locate_file("")).resolve()
    if Path(companion.locate_file("")).resolve() != root:
        raise RuntimeError("FA2 and FA4 metadata were installed in different environments")
    companion_files = [row[0] for row in csv.reader(io.StringIO(companion.read_text("RECORD") or ""))]
    if not companion_files or any(not Path(item).parts[0].endswith(".dist-info") for item in companion_files):
        raise RuntimeError("FA4 companion owns files outside its metadata directory")
    inventory = provenance["payload_sha256"]
    owned = {str(item) for item in combined.files or []}
    if not set(inventory).issubset(owned):
        raise RuntimeError("Combined wheel RECORD is missing payload files")
    for name, expected in inventory.items():
        path = Path(combined.locate_file(name))
        if not path.is_file():
            raise RuntimeError(f"Missing installed payload: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Installed payload hash mismatch: {name}")
    cute_root = root / "flash_attn/cute"
    unexpected = {
        str(path.relative_to(root))
        for path in cute_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    } - set(inventory)
    if unexpected:
        raise RuntimeError(f"Stale CuTe payload files: {sorted(unexpected)}")
    for dist in distributions:
        if dist is combined:
            continue
        overlaps = [
            str(item)
            for item in dist.files or []
            if str(item) in inventory and Path(dist.locate_file(item)).resolve() == root / str(item)
        ]
        if overlaps:
            raise RuntimeError(f"{dist.metadata['Name']} also owns payload: {overlaps}")
    report = {"flash-attn": combined.version, "flash-attn-4": companion.version, "payload_files": len(inventory)}
    for name, requirement in build["runtime_pins"].items() if check_dependencies else []:
        installed = importlib.metadata.version(name)
        if installed.split("+", 1)[0] != requirement.split("==", 1)[1]:
            raise RuntimeError(f"Expected {requirement}, found {installed}")
        report[name] = installed
    if runtime:
        if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
            raise RuntimeError("The B300 profile requires CPython 3.12 on Linux x86_64")
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available() or torch.version.cuda.split(".", 1)[0] != "13":
            raise RuntimeError("The B300 profile requires an available CUDA 13 GPU")
        devices = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
        if not devices or any(device != (10, 3) for device in devices):
            raise RuntimeError(f"The B300 profile requires SM103 devices, found {devices}")
        for name in ("flash_attn", "flash_attn.cute.interface"):
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(f"{name} is shadowed by {module.__file__}")
            report[name + ".path"] = module.__file__
        cutlass = importlib.import_module("cutlass")
        loaded_version = str(cutlass.__version__)
        if loaded_version != "4.6.2":
            raise RuntimeError(f"Loaded CUTLASS DSL is {loaded_version}, expected 4.6.2")
        report.update(cutlass_path=cutlass.__file__, cutlass_loaded_version=loaded_version, devices=devices)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--payload-only", action="store_true", help="Check file ownership without runtime dependencies")
    args = parser.parse_args()
    if args.runtime and args.payload_only:
        parser.error("--runtime cannot be combined with --payload-only")
    print(json.dumps(verify_install(args.runtime, not args.payload_only), indent=2))
