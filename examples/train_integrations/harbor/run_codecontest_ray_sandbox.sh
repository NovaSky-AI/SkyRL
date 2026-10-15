set -ex

# CodeContests training with Harbor trials in Ray sandboxes instead of Daytona.
# No DAYTONA_API_KEY needed. Nodes need runsc; build them from Dockerfile.ray-sandbox.
#
# Sandbox nodes advertise a custom resource so sandboxes stay off GPU nodes:
#   ray start --address=<head> --resources='{"sandbox": 1000}'
#
# IMAGE must be public and prebuilt: the task's base image plus tmux and asciinema.
# Otherwise terminus-2 apt-installs them in every trial, which needs network access.
IMAGE=${IMAGE:-python:3.12-slim}

bash "$(dirname "$0")/run_codecontest.sh" \
  harbor_trial_config.environment.import_path=examples.train_integrations.harbor.ray_sandbox_env:RaySandboxEnvironment \
  harbor_trial_config.environment.kwargs.image=$IMAGE \
  harbor_trial_config.environment.kwargs.resources.sandbox=0.001 \
  harbor_trial_config.environment.kwargs.ttl_seconds=3600 \
  "$@"
