# GitHub Actions Self-Hosted Runner Setup

This guide explains how to set up self-hosted GitHub Actions runners for SkyRL training jobs.

## Why Self-Hosted Runners?

- **No 6-hour timeout**: GitHub-hosted runners have a 6-hour job limit; training runs can take 24+ hours
- **Persistent SkyPilot state**: Cloud credentials and SkyPilot configuration persist across jobs
- **Cost control**: Run on your own EC2 instances

## Current Runner Pool

| Runner Name | Instance ID | IP | Status |
|-------------|-------------|-----|--------|
| fleet-gha-runner | i-045e312899d63f130 | 3.88.43.218 | Active |

To check runner status: [GitHub Settings → Actions → Runners](https://github.com/fleet-ai/SkyRL/settings/actions/runners)

## Adding a New Runner

### 1. Launch EC2 Instance

**Recommended specs:**
- AMI: Ubuntu 22.04 LTS (ami-0c7217cdde317cfec in us-east-1)
- Instance type: t3.medium (2 vCPU, 4GB RAM) - runner just orchestrates, GPUs are on SkyPilot clusters
- Storage: 50GB gp3
- Security group: Allow outbound HTTPS (port 443)

**Launch via AWS CLI:**
```bash
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.medium \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --subnet-id subnet-xxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fleet-runner-2}]'
```

### 2. Get GitHub Runner Token

1. Go to [SkyRL → Settings → Actions → Runners](https://github.com/fleet-ai/SkyRL/settings/actions/runners)
2. Click "New self-hosted runner"
3. Copy the token from the configure command (starts with `A...`)

### 3. Run Setup Script

SSH into the new instance and run:

```bash
# Set cloud credentials as environment variables
export LAMBDA_API_KEY="your-lambda-key"
export RUNPOD_API_KEY="your-runpod-key"
export VAST_API_KEY="your-vast-key"
export NEBIUS_CREDENTIALS_JSON='{"...":"..."}'
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"

# Download and run setup script
curl -O https://raw.githubusercontent.com/fleet-ai/SkyRL/main/scripts/setup-gha-runner.sh
chmod +x setup-gha-runner.sh
./setup-gha-runner.sh fleet-runner-2 <GITHUB_TOKEN>
```

### 4. Verify Runner

Check the runner appears in [GitHub Settings](https://github.com/fleet-ai/SkyRL/settings/actions/runners) with status "Idle".

## Managing Runners

### Check Status
```bash
# On the runner instance
sudo ~/actions-runner/svc.sh status

# View logs
sudo journalctl -u actions.runner.fleet-ai-SkyRL.fleet-runner-2 -f
```

### Stop/Start Runner
```bash
sudo ~/actions-runner/svc.sh stop
sudo ~/actions-runner/svc.sh start
```

### Remove Runner
```bash
cd ~/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token <GITHUB_TOKEN>
```

Then terminate the EC2 instance.

## Scaling the Runner Pool

Each runner can handle **1 concurrent job**. For N parallel training runs, you need N runners.

**Current recommendation: 3 runners** for parallel experimentation.

### Quick Setup for Multiple Runners

```bash
# On each new EC2 instance, just change the runner name
./setup-gha-runner.sh fleet-runner-2 <TOKEN>
./setup-gha-runner.sh fleet-runner-3 <TOKEN>
# etc.
```

All runners with the `fleet-runner` label will automatically receive jobs from the workflow.

## Troubleshooting

### Runner shows "Offline"
1. Check if the service is running: `sudo ~/actions-runner/svc.sh status`
2. Check logs: `sudo journalctl -u actions.runner.fleet-ai-SkyRL.* -f`
3. Restart: `sudo ~/actions-runner/svc.sh restart`

### SkyPilot can't find clouds
1. Verify credentials: `sky check`
2. Re-run credential setup from the setup script

### Job stuck in "Queued"
- All runners are busy. Wait or add more runners.
- Check runner status in GitHub Settings.

## Cost Estimates

| Component | Cost |
|-----------|------|
| EC2 t3.medium (runner) | ~$30/month per runner |
| GPU clusters (SkyPilot) | Varies by cloud/GPU type |

Runners are cheap - the real cost is GPU time on the training clusters.
