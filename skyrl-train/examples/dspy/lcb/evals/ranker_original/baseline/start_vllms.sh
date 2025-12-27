#!/usr/bin/env bash
set -e

SESSION="vllm"
ENV_NAME="assertion"
CONDA_SH="/home/eecs/jiashu.chen/.zshrc"
MAX_SEQS="512"
MAX_MODEL_LEN="8192"


declare -A SERVER0=(
  [model]="Qwen/Qwen2.5-Coder-7B-Instruct"
  [port]="7000"
  [gpus]="0"
  [dp]="1"
  [max_seqs]="$MAX_SEQS"
  [max_model_len]="$MAX_MODEL_LEN"
  [master_port]="29510"
)

declare -A SERVER1=(
  [model]="Qwen/Qwen2.5-Coder-1.5B-Instruct"
  [port]="7001"
  [gpus]="2"
  [dp]="1"
  [max_seqs]="$MAX_SEQS"
  [max_model_len]="$MAX_MODEL_LEN"
  [master_port]="29511"
)

declare -A SERVER2=(
  [model]="Qwen/Qwen2.5-Coder-3B-Instruct"
  [port]="7002"
  [gpus]="5"
  [dp]="1"
  [max_seqs]="$MAX_SEQS"
  [max_model_len]="$MAX_MODEL_LEN"
  [master_port]="29512"
)

# List of server dictionaries
SERVERS=(SERVER0 SERVER1 SERVER2)

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

  if [ "$idx" -ne 0 ]; then
    tmux new-window -a -t "$SESSION" -n "$WINDOW"
  fi

  echo "[start_vllms] Launching ${S[model]} on port ${S[port]} (GPUs ${S[gpus]}) dp=${S[dp]} max_len=${S[max_model_len]}"

  tmux send-keys -t "$SESSION:$WINDOW" "
    export CUDA_VISIBLE_DEVICES='${S[gpus]}';
    export MASTER_PORT='${S[master_port]}';
    source \"$CONDA_SH\" && \
    conda activate \"$ENV_NAME\" && \
    vllm serve \"${S[model]}\" \
      --host 127.0.0.1 \
      --port ${S[port]} \
      --data-parallel-size ${S[dp]} \
      --max-num-seqs ${S[max_seqs]} \
      --max-model-len ${S[max_model_len]} \
      --gpu-memory-utilization 0.95
  " C-m
donew

echo "[start_vllms] All servers launched."
echo "[start_vllms] Attaching to tmux session '$SESSION'..."
tmux attach -t "$SESSION"
