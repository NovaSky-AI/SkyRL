"""The integrations' public names import from their packages."""

import pytest


def test_harbor_exports():
    pytest.importorskip("harbor")
    from skyrl.integration import harbor
    from skyrl.integration.harbor import HarborGenerator, HarborTaskDataset

    assert HarborGenerator.__module__ == "skyrl.integration.harbor.harbor_generator"
    assert HarborTaskDataset.__module__ == "skyrl.integration.harbor.dataset"
    assert {"HarborGenerator", "HarborTaskDataset"} <= set(dir(harbor))
    with pytest.raises(AttributeError):
        harbor.NotAName


def test_harbor_skycap_exports():
    pytest.importorskip("harbor")
    pytest.importorskip("skycap")
    import skyrl.integration.harbor_skycap as harbor_skycap

    for name in harbor_skycap.__all__:
        assert getattr(harbor_skycap, name).__name__ == name


def test_record_index_imports_without_harbor():
    """`record_index pull` must run where Harbor isn't installed: import it in a fresh
    process that refuses any Harbor import."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'harbor' or name.startswith('harbor.'):\n"
        "            raise ImportError('harbor is blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import skyrl.integration.harbor_skycap as pkg\n"
        "from skyrl.integration.harbor_skycap.record_index import SkycapRecordIndex\n"
        "assert pkg.SkycapRecordIndex is SkycapRecordIndex\n"
        "assert not any(m == 'harbor' or m.startswith('harbor.') for m in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
