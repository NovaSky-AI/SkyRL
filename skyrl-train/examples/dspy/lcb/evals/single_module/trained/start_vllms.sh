#!/usr/bin/env bash
set -e

SESSION="vllm-trained"
ENV_NAME="assertion"
CONDA_SH="/home/eecs/jiashu.chen/.zshrc"

# declare -A SERVER0=(
#   [model]="Qwen/Qwen2.5-Coder-1.5B-Instruct"
#   [port]="7000"
#   [gpus]="0,1,2"
#   [dp]="3"
#   [max_seqs]="512"
#   [max_model_len]="8192"
#   [master_port]="29510"
#   [lora_path]="my_lora=/work/jiashu/.cache/models--Harryllh--lcb_test_generator_1.5b_100steps/snapshots/fa130d837ca5530214553405b6844e5c606c42b2"
# )

# declare -A SERVER0=(
#   [model]="Harryllh/lcb_test_generator_3b_100steps"
#   [port]="6999"
#   [gpus]="7"
#   [dp]="1"
#   [max_seqs]="512"
#   [max_model_len]="8192"
#   [master_port]="29510"
# )

declare -A SERVER0=(
  [model]="Qwen/Qwen2.5-Coder-1.5B-Instruct"
  [port]="7001"
  [gpus]="0"
  [dp]="1"
  [max_seqs]="512"
  [max_model_len]="8192"
  [master_port]="29510"
)

declare -A SERVER1=(
  [model]="Qwen/Qwen2.5-Coder-3B-Instruct"
  [port]="7002"
  [gpus]="1"
  [dp]="1"
  [max_seqs]="512"
  [max_model_len]="8192"
  [master_port]="29511"
)

declare -A SERVER2=(
  [model]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [port]="7003"
  [gpus]="2"
  [dp]="1"
  [max_seqs]="512"
  [max_model_len]="8192"
  [master_port]="29512"
)

declare -A SERVER3=(
  [model]="Harryllh/lcb_prog_generator_1.5b_100steps"
  [port]="7004"
  [gpus]="3"
  [dp]="1"
  [max_seqs]="512"
  [max_model_len]="8192"
  [master_port]="29513"
)

declare -A SERVER4=(
  [model]="Harryllh/lcb_prog_generator_3b_100steps"
  [port]="7005"
  [gpus]="4"
  [dp]="1"
  [max_seqs]="512"
  [max_model_len]="8192"
  [master_port]="29514"
)

declare -A SERVER5=(
  [model]="Harryllh/lcb_test_generator_1.5b_100steps"
  [port]="7006"
  [gpus]="5"
  [dp]="1"
  [max_seqs]="512"
  [max_model_len]="8192"
  [master_port]="29515"
)

# declare -A SERVER6=(
#   [model]="Harryllh/lcb_test_generator_3b_200steps"
#   [port]="7007"
#   [gpus]="7"
#   [dp]="1"
#   [max_seqs]="512"
#   [max_model_len]="8192"
#   [master_port]="29516"
# )

SERVERS=(SERVER0 SERVER1 SERVER2 SERVER3 SERVER4 SERVER5)

# Example for another:
# declare -A SERVER1=(
#   [model]="Qwen/Qwen2.5-Coder-7B-Instruct"
#   [port]="7001"
#   [gpus]="2,3"
#   [dp]="2"
#   [max_seqs]="384"
#   [max_model_len]="16384"
#   [master_port]="29511"
# )

# List of server dictionaries
# SERVERS=(SERVER0 SERVER1)

window_name () {
  local idx="$1"
  local model="$2"

  local short
  short=$(echo "$model" \
    | sed 's|.*/||' \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/-instruct//; s/[^a-z0-9]+/-/g')

  echo "${idx}-${short}"
}

echo "[start_vllms] Killing existing tmux session '$SESSION' (if any)..."
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

declare -n FIRST="${SERVERS[0]}"

FIRST_WINDOW=$(window_name 0 "${FIRST[model]}")
tmux new-session -d -s "$SESSION" -n "$FIRST_WINDOW"

for idx in "${!SERVERS[@]}"; do
  declare -n S="${SERVERS[$idx]}"

  WINDOW=$(window_name "$idx" "${S[model]}")

  # Create additional windows
  if [ "$idx" -ne 0 ]; then
    tmux new-window -a -t "$SESSION" -n "$WINDOW"
  fi

  echo "[start_vllms] Launching ${S[model]} on port ${S[port]} (GPUs ${S[gpus]}) max_len=${S[max_model_len]}"

  # Check if lora_path is set
  LORA_ARGS=""
  if [[ -n "${S[lora_path]+set}" ]]; then
    LORA_ARGS="--enable-lora --lora-modules \"${S[lora_path]}\""
  fi

  tmux send-keys -t "$SESSION:$WINDOW" "
    export CUDA_VISIBLE_DEVICES='${S[gpus]}';
    export MASTER_PORT='${S[master_port]}';
    source \"$CONDA_SH\" && \
    conda activate \"$ENV_NAME\" && \
    vllm serve \"${S[model]}\" \
      --host 127.0.0.1 \
      --port ${S[port]} \
      --max-num-seqs ${S[max_seqs]} \
      --max-model-len ${S[max_model_len]} \
      $LORA_ARGS \
      $( [ -z "$LORA_ARGS" ] && echo "--data-parallel-size ${S[dp]} \\" )
  " C-m
done

echo "[start_vllms] All servers launched."
echo "[start_vllms] Attaching to tmux session '$SESSION'..."
tmux attach -t "$SESSION"
