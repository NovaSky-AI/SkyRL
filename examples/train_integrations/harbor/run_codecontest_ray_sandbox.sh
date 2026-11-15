set -ex

# CodeContests training with Harbor trials in Ray sandboxes instead of Daytona.
# No DAYTONA_API_KEY needed. Nodes need the sandbox tools in https://docs.ray.io/en/master/core/sandboxes.html
# and Ray master @ 78e4b1f713; docker/Dockerfile.ray-sandbox builds such a node image.
#
# Sandbox nodes advertise a custom resource so sandboxes stay off GPU nodes:
#   ray start --address=<head> --resources='{"sandbox": 1000}'
#
# Run this on the Ray cluster, e.g. as a Ray job: with RAY_ADDRESS=local the trainer
# starts its own one-node Ray and can't place sandboxes on the sandbox nodes.
#
# IMAGE must be public and prebuilt: the task's base image plus tmux and asciinema.
# Otherwise terminus-2 apt-installs them in every trial, which needs network access.
IMAGE=${IMAGE:-python:3.12-slim}

# The job needs Ray with the sandbox fixes, which SkyRL's pinned Ray lacks until 2.61: by default it
# layers on the wheel of Ray master 78e4b1f713. The node must run the same Ray commit.
# RAY_WHEEL= (empty) turns it off.
export RAY_WHEEL="${RAY_WHEEL-https://s3-us-west-2.amazonaws.com/ray-wheels/master/78e4b1f71314fa46ca9bd2907d6aab4ec536909c/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl}"

bash "$(dirname "$0")/run_codecontest.sh" \
  harbor_trial_config.environment.import_path=skyrl.integration.ray_sandbox.harbor_env:RaySandboxEnvironment \
  harbor_trial_config.environment.kwargs.image=$IMAGE \
  harbor_trial_config.environment.kwargs.resources.sandbox=0.001 \
  "$@"
