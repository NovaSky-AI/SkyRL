import importlib.metadata
import json
import sys
from pathlib import Path

import pytest
import tomlkit

PROFILE = Path(__file__).resolve().parents[2] / "docker/attention/b300-fa4"
sys.path.insert(0, str(PROFILE))
import build_wheels  # noqa: E402
import image_profile  # noqa: E402
import mamba_compat  # noqa: E402
import verify_install  # noqa: E402


@pytest.fixture
def wheel_pair(tmp_path):
    manifest = json.loads((PROFILE / "inputs.json").read_text())
    cache, output = tmp_path / "inputs", tmp_path / "outputs"
    cache.mkdir()
    for name, spec in manifest["inputs"].items():
        root = f"{'flash_attn' if name == 'fa2' else 'flash_attn_4'}-{spec['version']}.dist-info"
        files = {
            f"{root}/METADATA": (
                f"Metadata-Version: 2.4\nName: {'flash-attn' if name == 'fa2' else 'flash-attn-4'}\n"
                f"Version: {spec['version']}\nRequires-Dist: einops\nRequires-Dist: torch\n"
            ).encode(),
            f"{root}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-cp312-manylinux_2_24_x86_64\n",
            f"{root}/licenses/LICENSE": b"BSD-3-Clause test license\n",
            "flash_attn/cute/__init__.py": f"SOURCE = '{name}'\n".encode(),
            "flash_attn/cute/interface.py": f"SOURCE = '{name}'\n".encode(),
        }
        if name == "fa2":
            files.update(
                {
                    "flash_attn_2_cuda.so": b"\x7fELF FA2 binary\x00",
                    "flash_attn/__init__.py": b"",
                    "flash_attn/cute/stale.py": b"old",
                }
            )
        else:
            files["flash_attn/cute/new.py"] = b"new"
        path = cache / spec["filename"]
        build_wheels.write_wheel(path, files)
        spec["sha256"] = build_wheels.sha256(path)
    release = build_wheels.build_pair(manifest, cache, output)
    return manifest, cache, output, release


def test_one_owner_preserves_fa2_and_replaces_all_cute(wheel_pair):
    manifest, cache, output, release = wheel_pair
    combined, companion = [build_wheels.read_wheel(output / item["filename"]) for item in release["artifacts"]]
    original = build_wheels.read_wheel(cache / manifest["inputs"]["fa2"]["filename"])
    upstream = build_wheels.read_wheel(cache / manifest["inputs"]["fa4"]["filename"])
    assert not set(combined) & set(companion)
    assert combined["flash_attn_2_cuda.so"] == original["flash_attn_2_cuda.so"]
    assert "flash_attn/cute/stale.py" not in combined
    assert {name: data for name, data in combined.items() if name.startswith(build_wheels.CUTE)} == {
        name: data for name, data in upstream.items() if name.startswith(build_wheels.CUTE)
    }
    metadata = build_wheels.read_metadata(companion)
    assert metadata.get_all("Requires-Dist") == [f"flash-attn=={manifest['combined_version']}"]
    assert metadata.get_all("Provides-Extra") == ["cu13"]
    assert all(name.startswith(build_wheels.dist_info(companion) + "/") for name in companion)


def test_build_is_byte_reproducible(wheel_pair, tmp_path):
    manifest, cache, _, release = wheel_pair
    rebuilt = build_wheels.build_pair(manifest, cache, tmp_path / "again")
    assert rebuilt == release


@pytest.mark.parametrize("name", ["fa2", "fa4"])
def test_corrupt_input_is_rejected_before_output(wheel_pair, tmp_path, name):
    manifest, cache, _, _ = wheel_pair
    (cache / manifest["inputs"][name]["filename"]).write_bytes(b"wrong wheel")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_wheels.build_pair(manifest, cache, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_record_detects_tampered_binary(wheel_pair):
    _, _, output, release = wheel_pair
    files = build_wheels.read_wheel(output / release["artifacts"][0]["filename"])
    files["flash_attn_2_cuda.so"] += b"changed"
    with pytest.raises(ValueError, match="RECORD mismatch"):
        build_wheels.validate_record(files)


@pytest.fixture
def installed_pair(wheel_pair, tmp_path, monkeypatch):
    manifest, _, output, release = wheel_pair
    site = tmp_path / "site-packages"
    site.mkdir()
    for item in release["artifacts"]:
        for name, data in build_wheels.read_wheel(output / item["filename"]).items():
            path = site / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    for name, requirement in manifest["runtime_pins"].items():
        version = requirement.split("==", 1)[1]
        root = site / f"{name.replace('-', '_')}-{version}.dist-info"
        root.mkdir()
        (root / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n")
    monkeypatch.syspath_prepend(str(site))
    discover = importlib.metadata.distributions
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: discover(path=[str(site)]))
    return site


def test_verifies_actual_distribution_metadata(installed_pair):
    report = verify_install.verify_install()
    assert report["payload_files"] == 5


@pytest.mark.parametrize("change", ["stale", "corrupt", "missing", "overlap", "companion_code"])
def test_installed_payload_damage_is_detected(installed_pair, change):
    site = installed_pair
    if change == "stale":
        (site / "flash_attn/cute/stale.py").write_text("stale")
    elif change == "corrupt":
        (site / "flash_attn/cute/interface.py").write_text("different version")
    elif change == "missing":
        (site / "flash_attn_2_cuda.so").unlink()
    elif change == "overlap":
        root = site / "another-1.dist-info"
        root.mkdir()
        (root / "METADATA").write_text("Name: another\nVersion: 1\n")
        (root / "RECORD").write_text("flash_attn/cute/interface.py,,\n")
    else:
        record = next(site.glob("flash_attn_4-*.dist-info/RECORD"))
        record.write_text(record.read_text() + "alias.py,,\n")
    with pytest.raises(RuntimeError):
        verify_install.verify_install()


def test_projection_is_opt_in_and_keeps_relative_sources(wheel_pair):
    _, _, _, release = wheel_pair
    source = (PROFILE.parents[2] / "pyproject.toml").read_text()
    before = tomlkit.parse(source)
    after = tomlkit.parse(image_profile.project_manifest(source, release))
    assert before["tool"]["uv"]["sources"]["skyrl-gym"] == after["tool"]["uv"]["sources"]["skyrl-gym"]
    assert before["project"]["optional-dependencies"]["tinker"] == after["project"]["optional-dependencies"]["tinker"]
    for name in ("fsdp", "megatron", "skyrl-train"):
        assert any(
            release["build"]["combined_version"] in requirement
            for requirement in after["project"]["optional-dependencies"][name]
        )
    assert "flash-attn-4" not in before["tool"]["uv"]["sources"]
    assert after["tool"]["uv"]["sources"]["flash-attn-4"]["url"] == release["artifacts"][1]["url"]
    assert len(after["tool"]["uv"]["sources"]["fast-hadamard-transform"]) == 1


def test_source_drift_is_rejected(tmp_path, monkeypatch):
    profile, root = tmp_path / "profile", tmp_path / "root"
    profile.mkdir()
    root.mkdir()
    for name in ("pyproject.toml", "uv.lock", "release-manifest.json"):
        (profile / name).write_text(name)
    expected = {
        "source_manifest_sha256": "0" * 64,
        "generated_manifest_sha256": build_wheels.sha256(profile / "pyproject.toml"),
        "lock_sha256": build_wheels.sha256(profile / "uv.lock"),
        "release_manifest_sha256": build_wheels.sha256(profile / "release-manifest.json"),
    }
    (profile / "profile.json").write_text(json.dumps(expected))
    (root / "pyproject.toml").write_text("changed source")
    monkeypatch.setattr(image_profile, "HERE", profile)
    with pytest.raises(ValueError, match="source manifest changed"):
        image_profile.apply(root)
    assert not (root / "uv.lock").exists()


def test_mamba_preparation_keeps_mamba2_importable(tmp_path, monkeypatch):
    root = tmp_path / "mamba_ssm"
    root.mkdir()
    (root / "modules").mkdir()
    (root / "modules/mamba2.py").write_text("class Mamba2: pass\n")
    (root / "modules/mamba3.py").write_text("raise RuntimeError('incompatible TileLang TVM')\n")
    init = root / "__init__.py"
    init.write_bytes(b"from mamba_ssm.modules.mamba2 import Mamba2\n" + mamba_compat.EAGER_IMPORT)
    metadata = tmp_path / f"mamba_ssm-{mamba_compat.VERSION}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: mamba-ssm\nVersion: {mamba_compat.VERSION}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RuntimeError, match="incompatible TileLang"):
        __import__("mamba_ssm")
    report = mamba_compat.prepare_mamba()
    assert report["original_sha256"] != report["prepared_sha256"]
    assert __import__("mamba_ssm").Mamba2.__name__ == "Mamba2"
    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            sys.modules.pop(name)
