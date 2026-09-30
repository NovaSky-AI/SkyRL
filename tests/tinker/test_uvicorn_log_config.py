from rich.logging import RichHandler

from skyrl.utils.log import get_uvicorn_log_config


def test_uvicorn_error_logger_uses_plain_handler():
    config = get_uvicorn_log_config()

    assert config["handlers"]["default"]["()"] is RichHandler
    assert config["handlers"]["error"]["class"] == "logging.StreamHandler"
    assert "()" not in config["handlers"]["error"]
    assert config["loggers"]["uvicorn.error"]["handlers"] == ["error"]
