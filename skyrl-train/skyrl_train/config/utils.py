from pathlib import Path
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).parent  # skyrl-train/config
DEFAULT_CONFIG_NAME = "ppo_base_config.yaml"


def get_default_config():
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=DEFAULT_CONFIG_NAME)
    return cfg


if __name__ == "__main__":
    cfg = get_default_config()
