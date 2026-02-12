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

    When SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1, redirection is skipped so all logs
    appear on stdout.

    Note: Do NOT call this in skyrl_entrypoint() - training progress should
    go to stdout.
    """
    from skyrl_train.env_vars import SKYRL_DUMP_INFRA_LOG_TO_STDOUT

    if SKYRL_DUMP_INFRA_LOG_TO_STDOUT:
        return

    log_file = os.getenv("SKYRL_LOG_FILE")
    if log_file:
        # Ensure the directory exists on this node (may differ from driver in multi-node)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", buffering=1) as log_f:
            os.dup2(log_f.fileno(), sys.stdout.fileno())
            os.dup2(log_f.fileno(), sys.stderr.fileno())
