"""
Slack alerting utilities for training metric thresholds.

Sends alerts to Slack channels when training metrics exceed configured thresholds.
Requires SLACK_BOT_TOKEN environment variable with chat:write scope.

All functions are fault-tolerant: they catch all exceptions internally and will
never raise, so they are safe to call in the training loop without risk of
crashing a training run.
"""

import logging
import os
from typing import Any, Dict, Optional
from urllib import request
import json

logger = logging.getLogger(__name__)

# Default channel for training alerts
DEFAULT_CHANNEL = "#fleet-training-runs"

# Track whether we've already alerted for a given metric+step to avoid spam
_alerted_steps: Dict[str, int] = {}


def send_slack_alert(
    message: str,
    channel: str = DEFAULT_CHANNEL,
    bot_token: Optional[str] = None,
) -> bool:
    """Post an alert message to a Slack channel.

    This function is fault-tolerant and will never raise an exception.

    Args:
        message: The message text to send.
        channel: Slack channel name or ID (default: #fleet-training-runs).
        bot_token: Slack bot token. Falls back to SLACK_BOT_TOKEN env var.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    try:
        token = bot_token or os.environ.get("SLACK_BOT_TOKEN")
        if not token:
            logger.warning("SLACK_BOT_TOKEN not set, skipping Slack alert")
            return False

        payload = json.dumps({"channel": channel, "text": message}).encode("utf-8")
        req = request.Request(
            "https://slack.com/api/chat.postMessage",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )

        with request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if not body.get("ok"):
                logger.warning(f"Slack API error: {body.get('error', 'unknown')}")
                return False
            return True
    except Exception as e:
        logger.warning(f"Failed to send Slack alert: {e}")
        return False


def check_tool_error_rate_alert(
    metrics: Dict[str, Any],
    global_step: int,
    threshold: float = 0.5,
    run_name: Optional[str] = None,
    channel: str = DEFAULT_CHANNEL,
) -> None:
    """Check if tool_error_rate exceeds threshold and send a Slack alert.

    Checks both per-environment and overall tool_error_rate metrics.
    Only alerts once per metric per step to avoid spam.

    This function is fault-tolerant and will never raise an exception.

    Args:
        metrics: Dictionary of training metrics (from self.all_metrics).
        global_step: Current training step.
        threshold: Alert if tool_error_rate exceeds this value (default 0.5 = 50%).
        run_name: Optional run name for the alert message.
        channel: Slack channel to alert.
    """
    try:
        alerts = []

        for key, value in metrics.items():
            if not key.endswith("tool_error_rate"):
                continue
            if not isinstance(value, (int, float)) or value <= threshold:
                continue

            # Deduplicate: only alert once per metric per step
            alert_key = f"{key}:{global_step}"
            if alert_key in _alerted_steps:
                continue
            _alerted_steps[alert_key] = global_step

            # Extract env name from key like "environment/outlook/tool_error_rate"
            parts = key.rsplit("/tool_error_rate", 1)
            env_name = parts[0] if parts[0] != key else "overall"

            alerts.append((env_name, value))

        if not alerts:
            return

        run_label = f" (run: `{run_name}`)" if run_name else ""
        lines = [f":warning: *Tool Error Rate Alert* — step {global_step}{run_label}"]
        for env_name, rate in alerts:
            lines.append(f"  • `{env_name}`: {rate:.1%} tool error rate (threshold: {threshold:.0%})")

        send_slack_alert("\n".join(lines), channel=channel)
    except Exception as e:
        logger.warning(f"Tool error rate alert check failed (non-fatal): {e}")
