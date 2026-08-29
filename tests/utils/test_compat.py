import sys
from importlib.metadata import PackageNotFoundError

from skyrl import _compat


def test_disable_flash_attn_cute_blocks_bundled_implementation(monkeypatch):
    def package_not_found(_package: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(_compat, "version", package_not_found)
    monkeypatch.delitem(sys.modules, "flash_attn.cute", raising=False)

    _compat.disable_flash_attn_cute()

    assert "flash_attn.cute" in sys.modules
    assert sys.modules["flash_attn.cute"] is None


def test_disable_flash_attn_cute_preserves_dedicated_fa4(monkeypatch):
    dedicated_fa4 = object()
    monkeypatch.setattr(_compat, "version", lambda _package: "4.0.0b27")
    monkeypatch.setitem(sys.modules, "flash_attn.cute", dedicated_fa4)

    _compat.disable_flash_attn_cute()

    assert sys.modules["flash_attn.cute"] is dedicated_fa4
