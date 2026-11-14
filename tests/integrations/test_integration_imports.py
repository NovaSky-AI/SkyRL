"""The integrations' public names import from their packages."""

import pytest


def test_harbor_exports():
    pytest.importorskip("harbor")
    from skyrl.integration import harbor
    from skyrl.integration.harbor import HarborGenerator, HarborTaskDataset

    assert HarborGenerator.__module__ == "skyrl.integration.harbor.harbor_generator"
    assert HarborTaskDataset.__module__ == "skyrl.integration.harbor.dataset"
    with pytest.raises(AttributeError):
        harbor.NotAName


def test_harbor_skycap_exports():
    pytest.importorskip("harbor")
    pytest.importorskip("skycap")
    import skyrl.integration.harbor_skycap as harbor_skycap

    for name in harbor_skycap.__all__:
        assert getattr(harbor_skycap, name).__name__ == name
