"""
Helper to redirect Ray actor stdout/stderr to log file.

This prevents infrastructure logs from polluting the driver's stdout,
allowing only training progress to be displayed to the user.
"""

import os
import sys


def redirect_actor_output_to_file():
    """
    Redirect stdout and stderr to log file to prevent Ray from forwarding to driver.

    Call this at the very start of any Ray actor/remote function where you want
    to suppress output from appearing on the driver's stdout. The output will
    instead be written to the log file specified by SKYRL_LOG_FILE.

    In DEBUG mode, redirection is skipped so all logs appear on stdout.

    Note: Do NOT call this in skyrl_entrypoint() - training progress should
    go to stdout.
    """
    from skyrl_train.env_vars import SKYRL_LOG_LEVEL

    if SKYRL_LOG_LEVEL == "DEBUG":
        return

    log_file = os.getenv("SKYRL_LOG_FILE")
    if log_file:
        log_fd = open(log_file, "a", buffering=1)  # noqa: SIM115
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
        log_fd.close()  # original fd no longer needed; stdout/stderr hold copies
