#!/bin/bash
# Health check script for GitHub Actions self-hosted runners
# Runs via cron to monitor runner health and auto-restart on failure
#
# Install: Add to crontab with: */5 * * * * /opt/runner-health-check.sh
# Requires: SLACK_WEBHOOK_URL environment variable (optional)

set -euo pipefail

RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)}"
LOG_FILE="/var/log/runner-health-check.log"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
MAX_RESTART_ATTEMPTS=3
RESTART_ATTEMPTS_FILE="/tmp/runner_restart_attempts"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | sudo tee -a "$LOG_FILE" >/dev/null
}

notify_slack() {
    local message="$1"
    local emoji="${2:-warning}"
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -sf -X POST "$SLACK_WEBHOOK" \
            -H 'Content-type: application/json' \
            -d "{\"text\":\":${emoji}: [${RUNNER_NAME}] ${message}\"}" >/dev/null 2>&1 || true
    fi
}

get_restart_attempts() {
    if [ -f "$RESTART_ATTEMPTS_FILE" ]; then
        cat "$RESTART_ATTEMPTS_FILE"
    else
        echo "0"
    fi
}

increment_restart_attempts() {
    local current=$(get_restart_attempts)
    echo $((current + 1)) > "$RESTART_ATTEMPTS_FILE"
}

reset_restart_attempts() {
    echo "0" > "$RESTART_ATTEMPTS_FILE"
}

check_runner_service() {
    # Find the runner service (name varies by runner name)
    local service_name=$(systemctl list-units --type=service --all | grep "actions.runner" | awk '{print $1}' | head -1)

    if [ -z "$service_name" ]; then
        log "ERROR: Could not find runner service"
        return 1
    fi

    if systemctl is-active --quiet "$service_name"; then
        return 0
    else
        return 1
    fi
}

check_github_registration() {
    # Check if runner is registered as online with GitHub
    # Look for "Listening for Jobs" in recent logs (last 5 min)
    local recent_logs=$(journalctl -u "actions.runner.*" --since "5 minutes ago" 2>/dev/null)

    # Signs of healthy connection
    if echo "$recent_logs" | grep -q "Listening for Jobs\|Running job\|Successfully renew job"; then
        return 0
    fi

    # Signs of disconnection
    if echo "$recent_logs" | grep -q "Unable to read data from the transport\|SocketException\|Operation canceled"; then
        log "Detected GitHub connection errors in logs"
        return 1
    fi

    # No recent activity - check if service started recently
    local uptime=$(systemctl show "actions.runner.*" --property=ActiveEnterTimestamp 2>/dev/null | cut -d= -f2)
    if [ -n "$uptime" ]; then
        local uptime_secs=$(( $(date +%s) - $(date -d "$uptime" +%s 2>/dev/null || echo 0) ))
        if [ "$uptime_secs" -lt 300 ]; then
            # Service started less than 5 min ago, give it time
            return 0
        fi
    fi

    # No activity for 5+ min - suspicious
    log "No GitHub activity detected in last 5 minutes"
    return 1
}

check_github_connectivity() {
    if curl -sf --max-time 10 https://api.github.com >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_disk_space() {
    local usage=$(df "$RUNNER_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$usage" -gt 90 ]; then
        return 1
    fi
    return 0
}

cleanup_disk() {
    log "Disk usage high, running cleanup..."
    local before=$(df "$RUNNER_DIR" | tail -1 | awk '{print $5}')
    # Clean old SkyPilot logs (keep last 2 days)
    find ~/.sky/logs -type f -mtime +2 -delete 2>/dev/null || true
    find ~/.sky/logs -type d -empty -delete 2>/dev/null || true
    # Clean pip cache
    pip cache purge 2>/dev/null || true
    # Clean apt cache
    sudo apt-get clean 2>/dev/null || true
    # Clean tmp files older than 1 day
    find /tmp -type f -mtime +1 -not -name "runner_*" -delete 2>/dev/null || true
    # Clean old journal logs
    sudo journalctl --vacuum-time=2d 2>/dev/null || true
    local after=$(df "$RUNNER_DIR" | tail -1 | awk '{print $5}')
    log "Disk cleanup: $before -> $after"
}

restart_runner() {
    local service_name=$(systemctl list-units --type=service --all | grep "actions.runner" | awk '{print $1}' | head -1)
    log "Restarting runner service: $service_name"
    sudo systemctl restart "$service_name"
    sleep 5

    if check_runner_service; then
        log "Runner restarted successfully"
        reset_restart_attempts
        notify_slack "Runner restarted successfully" "white_check_mark"
        return 0
    else
        log "Runner restart failed"
        return 1
    fi
}

main() {
    local issues=()

    # Check 1: Runner service status
    if ! check_runner_service; then
        log "ALERT: Runner service is not running"
        issues+=("service_down")
    fi

    # Check 2: GitHub connectivity (network level)
    if ! check_github_connectivity; then
        log "ALERT: Cannot reach GitHub API"
        issues+=("network_issue")
    fi

    # Check 3: GitHub registration (runner is actually connected)
    if check_runner_service && ! check_github_registration; then
        log "ALERT: Runner service running but disconnected from GitHub"
        issues+=("github_disconnected")
    fi

    # Check 4: Disk space — auto-cleanup if above 90%
    if ! check_disk_space; then
        log "ALERT: Disk usage above 90%"
        issues+=("disk_full")
        cleanup_disk
        if ! check_disk_space; then
            notify_slack "Disk usage still above 90% after cleanup - manual intervention needed" "warning"
        else
            log "Disk cleanup resolved the issue"
            notify_slack "Disk usage was above 90%, auto-cleanup freed space" "white_check_mark"
        fi
    fi

    # Handle issues
    if [[ " ${issues[*]} " =~ " service_down " ]] || [[ " ${issues[*]} " =~ " github_disconnected " ]]; then
        local attempts=$(get_restart_attempts)
        local issue_type="Service down"
        [[ " ${issues[*]} " =~ " github_disconnected " ]] && issue_type="GitHub disconnected"

        if [ "$attempts" -lt "$MAX_RESTART_ATTEMPTS" ]; then
            increment_restart_attempts
            notify_slack "${issue_type} - attempting restart (attempt $((attempts+1))/$MAX_RESTART_ATTEMPTS)" "rotating_light"

            if restart_runner; then
                log "Runner recovered"
            else
                log "Runner restart attempt failed"
            fi
        else
            notify_slack "CRITICAL: Runner failed after $MAX_RESTART_ATTEMPTS restart attempts - manual intervention needed" "red_circle"
            log "CRITICAL: Max restart attempts reached"
        fi
    elif [ ${#issues[@]} -eq 0 ]; then
        # All healthy - reset restart counter
        reset_restart_attempts
        log "Health check passed"
    fi
}

main "$@"
