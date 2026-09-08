import importlib
import importlib.metadata
import sys

import skyrl


def test_installed_fa4_remains_importable(monkeypatch):
    def installed_version(distribution):
        assert distribution == "flash-attn-4"
        return "4.0.0b28"

    monkeypatch.delitem(sys.modules, "flash_attn.cute", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", installed_version)

    importlib.reload(skyrl)

    assert "flash_attn.cute" not in sys.modules


def test_missing_fa4_disables_bundled_cute(monkeypatch):
    def raise_package_not_found(distribution):
        assert distribution == "flash-attn-4"
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.delitem(sys.modules, "flash_attn.cute", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", raise_package_not_found)

    importlib.reload(skyrl)

    assert sys.modules["flash_attn.cute"] is None


def test_metadata_only_companion_enables_cute(tmp_path, monkeypatch):
    metadata = tmp_path / "flash_attn_4-4.0.0b28+skyrl1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.4\nName: flash-attn-4\nVersion: 4.0.0b28+skyrl1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "flash_attn.cute", raising=False)

    importlib.reload(skyrl)

    assert importlib.metadata.version("flash-attn-4") == "4.0.0b28+skyrl1"
    assert "flash_attn.cute" not in sys.modules
