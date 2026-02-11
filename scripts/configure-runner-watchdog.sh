#!/bin/bash
# Configure systemd watchdog for GitHub Actions runner service
# This ensures the runner automatically restarts if it crashes or becomes unresponsive
#
# Usage: sudo ./configure-runner-watchdog.sh

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Must run as root
if [ "$EUID" -ne 0 ]; then
    error "Please run as root (sudo)"
fi

# Find the runner service
SERVICE_NAME=$(systemctl list-units --type=service --all | grep "actions.runner" | awk '{print $1}' | head -1)

if [ -z "$SERVICE_NAME" ]; then
    error "Could not find GitHub Actions runner service"
fi

log "Found runner service: $SERVICE_NAME"

# Create override directory
OVERRIDE_DIR="/etc/systemd/system/${SERVICE_NAME}.d"
mkdir -p "$OVERRIDE_DIR"

# Create watchdog override
OVERRIDE_FILE="$OVERRIDE_DIR/override.conf"
cat > "$OVERRIDE_FILE" << 'EOF'
[Service]
# Automatically restart if the service fails
Restart=always
RestartSec=10

# Kill process if it doesn't respond within 5 minutes
WatchdogSec=300

# Limit restart attempts: 5 times within 10 minutes
StartLimitIntervalSec=600
StartLimitBurst=5

# After hitting limit, wait 30 minutes before allowing restarts again
StartLimitAction=none

# Send SIGKILL after 90 seconds if graceful shutdown fails
TimeoutStopSec=90
EOF

log "Created watchdog configuration: $OVERRIDE_FILE"

# Reload systemd
log "Reloading systemd..."
systemctl daemon-reload

# Restart the runner service to apply changes
log "Restarting runner service..."
systemctl restart "$SERVICE_NAME"

sleep 3

# Verify service is running
if systemctl is-active --quiet "$SERVICE_NAME"; then
    log "Runner service restarted successfully with watchdog enabled"
    echo ""
    echo "Watchdog configuration:"
    echo "  - Auto-restart on failure: YES"
    echo "  - Restart delay: 10 seconds"
    echo "  - Watchdog timeout: 5 minutes"
    echo "  - Max restarts: 5 within 10 minutes"
else
    error "Runner service failed to restart"
fi

# Show current status
echo ""
log "Current service status:"
systemctl status "$SERVICE_NAME" --no-pager
