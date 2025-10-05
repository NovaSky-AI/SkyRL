from pathlib import Path
from hydra import initialize, compose
import os

CONFIG_DIR = Path(__file__).parent  # skyrl-train/config
DEFAULT_CONFIG_NAME = "ppo_base_config.yaml"


def get_default_config():
    current_directory = Path(__file__).parent.absolute()
    abs_config_dir = Path(CONFIG_DIR).absolute()
    relative_config_dir = os.path.relpath(abs_config_dir, current_directory)
    with initialize(version_base=None, config_path=relative_config_dir):
        cfg = compose(config_name=DEFAULT_CONFIG_NAME)
    return cfg


if __name__ == "__main__":
    cfg = get_default_config()
