#!/bin/bash
# Setup script for GitHub Actions self-hosted runner with SkyPilot
# Usage: ./setup-gha-runner.sh <RUNNER_NAME> <GITHUB_TOKEN>
#
# Prerequisites:
#   - Ubuntu 22.04+ EC2 instance
#   - Cloud credentials (Lambda, RunPod, Vast, Nebius) available as env vars or files
#
# This script will:
#   1. Install system dependencies
#   2. Install Python and SkyPilot
#   3. Configure cloud credentials
#   4. Install and start GitHub Actions runner

set -e

RUNNER_NAME="${1:-fleet-runner-$(hostname)}"
GITHUB_TOKEN="${2:-}"
REPO_URL="https://github.com/fleet-ai/SkyRL"
RUNNER_VERSION="2.321.0"
RUNNER_LABELS="fleet-runner"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    error "Do not run as root. Run as a regular user with sudo access."
fi

# Check for GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    echo ""
    echo "GitHub Runner Token required."
    echo "Get it from: https://github.com/fleet-ai/SkyRL/settings/actions/runners/new"
    echo ""
    read -p "Enter GitHub Runner Token: " GITHUB_TOKEN
fi

log "Setting up GitHub Actions runner: $RUNNER_NAME"

# 1. Install system dependencies
log "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    curl \
    git \
    jq \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libssl-dev \
    libffi-dev \
    unzip

# Install AWS CLI (needed for S3 data validation)
log "Installing AWS CLI..."
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
cd /tmp && unzip -q awscliv2.zip && sudo ./aws/install
aws --version

# 2. Install SkyPilot
log "Installing SkyPilot..."
pip3 install --user "skypilot-nightly[lambda,runpod,vast]"
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Create symlink so GHA runner service can find sky
sudo ln -sf "$HOME/.local/bin/sky" /usr/local/bin/sky
sky --version

# 3. Configure cloud credentials
log "Configuring cloud credentials..."

# Lambda Cloud
if [ -n "$LAMBDA_API_KEY" ]; then
    mkdir -p ~/.lambda_cloud
    echo "api_key = $LAMBDA_API_KEY" > ~/.lambda_cloud/lambda_keys
    log "Lambda Cloud configured"
fi

# RunPod
if [ -n "$RUNPOD_API_KEY" ]; then
    pip3 install --user runpod
    runpod config "$RUNPOD_API_KEY" || true
    log "RunPod configured"
fi

# Vast.ai
if [ -n "$VAST_API_KEY" ]; then
    mkdir -p ~/.config/vastai
    echo "$VAST_API_KEY" > ~/.config/vastai/vast_api_key
    log "Vast.ai configured"
fi

# Nebius
if [ -n "$NEBIUS_CREDENTIALS_JSON" ]; then
    mkdir -p ~/.nebius
    echo "$NEBIUS_CREDENTIALS_JSON" > ~/.nebius/credentials.json
    log "Nebius configured"
fi

# AWS (for S3 trajectory storage)
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    mkdir -p ~/.aws
    cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
    log "AWS configured"
fi

# Verify SkyPilot can see clouds
log "Verifying cloud access..."
sky check || warn "Some clouds may not be configured"

# 4. Install GitHub Actions runner
log "Installing GitHub Actions runner..."
RUNNER_DIR="$HOME/actions-runner"
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

# Download runner
curl -o actions-runner-linux-x64.tar.gz -L \
    "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf actions-runner-linux-x64.tar.gz
rm actions-runner-linux-x64.tar.gz

# Configure runner
./config.sh \
    --url "$REPO_URL" \
    --token "$GITHUB_TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --unattended \
    --replace

# 5. Install and start as service
log "Installing runner as service..."
sudo ./svc.sh install
sudo ./svc.sh start

# Verify runner is running
sleep 2
if sudo ./svc.sh status | grep -q "active (running)"; then
    log "Runner installed and running successfully!"
else
    warn "Runner service may not be running. Check with: sudo ./svc.sh status"
fi

echo ""
echo "============================================"
echo "Runner setup complete!"
echo "Name: $RUNNER_NAME"
echo "Labels: $RUNNER_LABELS"
echo "Directory: $RUNNER_DIR"
echo ""
echo "Useful commands:"
echo "  Check status:  sudo $RUNNER_DIR/svc.sh status"
echo "  View logs:     sudo journalctl -u actions.runner.fleet-ai-SkyRL.$RUNNER_NAME -f"
echo "  Stop runner:   sudo $RUNNER_DIR/svc.sh stop"
echo "  Start runner:  sudo $RUNNER_DIR/svc.sh start"
echo "============================================"
