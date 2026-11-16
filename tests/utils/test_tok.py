from types import SimpleNamespace

from skyrl.utils.tok import check_is_vlm


def test_check_is_vlm_loads_pathlike_with_auto_config(tmp_path, monkeypatch):
    config = SimpleNamespace(vision_config={"image_size": 224})
    loaded_paths = []

    def load_config(path, **kwargs):
        loaded_paths.append((path, kwargs))
        return config

    monkeypatch.setattr("skyrl.utils.tok.AutoConfig.from_pretrained", load_config)

    assert check_is_vlm(tmp_path) is True
    assert loaded_paths == [(tmp_path, {"trust_remote_code": True})]
