# GitHub Actions Self-Hosted Runners

Self-hosted runners for long training jobs (24+ hours) that exceed GitHub's 6-hour limit.

## Runner Pool

| Runner | Instance ID | Type | RAM | Disk | Swap |
|--------|-------------|------|-----|------|------|
| fleet-runner-1 | i-04a15158610df980f | t3.medium | 4GB | 50GB | 4GB |
| fleet-runner-2 | i-0f80c703294413a4c | t3.xlarge | 16GB | 50GB | 4GB |
| fleet-runner-3 | i-09321b67952c1a208 | t3.xlarge | 16GB | 50GB | 4GB |
| fleet-runner-4 | i-0f2ccaa840ef29450 | t3.medium | 4GB | 50GB | 4GB |
| fleet-runner-5 | i-05ecd98e74949dc87 | t3.xlarge | 16GB | 50GB | 4GB |

> **Note:** Runners 1 and 4 are still t3.medium. They should be upgraded to t3.xlarge
> when idle (stop instance, change type, start instance).

**Cost:** ~$120/month per t3.xlarge runner, ~$30/month per t3.medium runner

[View runner status](https://github.com/fleet-ai/SkyRL/settings/actions/runners)

## Auto-Recovery

- **Watchdog:** Auto-restarts crashed/hung services (5-min timeout)
- **Health check:** Cron every 5 min detects GitHub disconnection
- **Central monitoring:** `runner-health.yaml` checks all runners every 15 min, auto-recovers unhealthy instances via EC2 stop/start, and alerts Slack

The central health check (`runner-health.yaml`) will automatically attempt to recover unhealthy runners by stopping and restarting the EC2 instance. It tracks recovery attempts per instance and escalates to a manual intervention Slack alert after 2 failed attempts.

## Installed Cloud Providers

The setup script installs SkyPilot with cloud plugins and additional CLIs:

| Provider | SkyPilot Plugin | Separate CLI | Config Location |
|----------|-----------------|--------------|-----------------|
| Lambda | ✓ | - | `~/.lambda_cloud/lambda_keys` |
| RunPod | ✓ | `runpod` | `~/.runpod/config.toml` |
| Vast | ✓ | - | `~/.config/vastai/vast_api_key` |
| Nebius | ✓ | - | `~/.nebius/credentials.json` |
| AWS | ✓ | `aws` | `~/.aws/credentials` |
| Prime Intellect | - | `prime` | via `prime config` |

Credentials are configured during workflow runs from GitHub secrets.

## Add a Runner

```bash
# 1. Launch EC2 (copy config from existing runner)
aws ec2 run-instances --image-id ami-0c7217cdde317cfec --instance-type t3.xlarge --key-name gha-runner-key --security-group-ids sg-00fefd8181d51909d --subnet-id subnet-03879810067f57f85 --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fleet-runner-N}]'

# 2. Get token
gh api -X POST repos/fleet-ai/SkyRL/actions/runners/registration-token --jq '.token'

# 3. SSH and setup
curl -O https://raw.githubusercontent.com/fleet-ai/SkyRL/main/scripts/setup-gha-runner.sh
chmod +x setup-gha-runner.sh && ./setup-gha-runner.sh fleet-runner-N <TOKEN>
```

## Troubleshooting

**Runner offline:**

The health check workflow will auto-recover most cases. If manual intervention is needed:

```bash
# Stop/start the instance (reboot does NOT fix impaired instances):
aws ec2 stop-instances --instance-ids <INSTANCE_ID>
aws ec2 wait instance-stopped --instance-ids <INSTANCE_ID>
aws ec2 start-instances --instance-ids <INSTANCE_ID>
```

> **Note:** `reboot-instances` does not work for instances with `impaired` status. Always use stop/start instead.

**Check instance health:**
```bash
aws ec2 describe-instance-status --instance-ids i-04a15158610df980f i-0f80c703294413a4c i-09321b67952c1a208 i-0f2ccaa840ef29450 i-05ecd98e74949dc87 --query 'InstanceStatuses[].[InstanceId,InstanceStatus.Status]' --output table
```

**Manual service restart (via SSH):**
```bash
sudo systemctl restart actions.runner.fleet-ai-SkyRL.*
```

**SkyPilot or RunPod not found (`sky: command not found` or `runpod: command not found`):**

The setup script installs tools to `~/.local/bin/` and creates symlinks at `/usr/local/bin/`. If symlinks are missing, the GHA service can't find them:

```bash
# Check if symlinks exist
ls -la /usr/local/bin/sky /usr/local/bin/runpod

# If missing, create them (via EC2 Instance Connect):
aws ec2-instance-connect send-ssh-public-key --instance-id <INSTANCE_ID> --instance-os-user ubuntu --ssh-public-key file://~/.ssh/<YOUR_KEY>.pub
ssh -i ~/.ssh/<YOUR_KEY> ubuntu@<IP> "sudo ln -sf /home/ubuntu/.local/bin/sky /usr/local/bin/sky && sudo ln -sf /home/ubuntu/.local/bin/runpod /usr/local/bin/runpod && sky --version"
```
