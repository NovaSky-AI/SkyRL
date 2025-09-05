# Helper that converts Environment Hub ID to required uv flags
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env_resolver.sh"

# Specify environment from Environments Hub in form "org/name@version" (e.g., will/wordle@0.1.4)
if [ -z "$1" ]; then
  echo "Usage: $(basename "$0") <ENV_ID>"
  exit 1
fi
ENV_ID="$1"
DATA_DIR="$HOME/data/$ENV_ID"

ENV_UV_INSTALL_FLAGS="$(verifiers_env_to_uv_flags "$ENV_ID")"
uv run --isolated --with verifiers $ENV_UV_INSTALL_FLAGS -- python "$SCRIPT_DIR/verifiers_dataset.py" --env_id $ENV_ID --output_dir $DATA_DIR