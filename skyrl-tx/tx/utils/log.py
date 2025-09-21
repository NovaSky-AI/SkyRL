import logging
from pathlib import Path

from rich.logging import RichHandler


def _setup_root_logger() -> None:
    logger = logging.getLogger("tx")
    logger.setLevel(logging.DEBUG)
    handler = RichHandler(
        show_time=False,
        show_level=False,
        markup=True,
    )
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    logger = logging.getLogger("tx")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")

_setup_root_logger()
logger = logging.getLogger("tx")


__all__ = ["logger"]
