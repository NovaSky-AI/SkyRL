#!/usr/bin/env bash
set -e

SESSION_BASE="evals-baseline"
# NAME="prog-gen-3b"
NAME="prog-gen-1point5b"

SESSION="$SESSION_BASE-$NAME"
# -------------------------
# Model definitions
# -------------------------
TEST_GEN_7B="Qwen/Qwen2.5-Coder-7B-Instruct"
TEST_GEN_15B="Qwen/Qwen2.5-Coder-1.5B-Instruct"
TEST_GEN_3B="Qwen/Qwen2.5-Coder-3B-Instruct"

PORT_7B=7000
PORT_15B=7001
PORT_3B=7002

# PROG_MODEL="Qwen/Qwen2.5-Coder-3B-Instruct"
PROG_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
# PROG_PORT=$PORT_3B
PROG_PORT=$PORT_15B

# -------------------------
# Paths & env
# -------------------------
# BASE_PATH="/work/jiashu/assertion-data/livecodebench/evals/results/ranker-original/test"
BASE_PATH="/work/jiashu/assertion-data/livecodebench/evals/results/ranker-original/baseline/$NAME"

if [ ! -d "$BASE_PATH" ]; then
  mkdir -p "$BASE_PATH"
fi

LOG_DIR=$BASE_PATH/logs
echo "LOG_DIR: $LOG_DIR"
CONDA_SH="/home/eecs/jiashu.chen/.zshrc"
ENV_NAME="assertion"
NUM_WORKERS="10"
NUM_RUN="3"

# ------------------------------------------------------------
# Eval Tasks (dictionaries)
# ------------------------------------------------------------
# declare -A TASK0=(
#   [dataset]="train"
#   [gen_model]="$TEST_GEN_7B"
#   [gen_port]="$PORT_7B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# declare -A TASK1=(
#   [dataset]="train"
#   [gen_model]="$TEST_GEN_15B"
#   [gen_port]="$PORT_15B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# declare -A TASK2=(
#   [dataset]="train"
#   [gen_model]="$TEST_GEN_3B"
#   [gen_port]="$PORT_3B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

declare -A TASK3=(
  [dataset]="test"
  [gen_model]="$TEST_GEN_7B"
  [gen_port]="$PORT_7B"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

# declare -A TASK4=(
#   [dataset]="test"
#   [gen_model]="$TEST_GEN_15B"
#   [gen_port]="$PORT_15B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# declare -A TASK5=(
#   [dataset]="test"
#   [gen_model]="$TEST_GEN_3B"
#   [gen_port]="$PORT_3B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# List of tasks
# TASKS=(TASK0 TASK1 TASK2 TASK3 TASK4 TASK5)
TASKS=(TASK3)

# ------------------------------------------------------------
# Helper: derive simple model tag from model name
# Example: Qwen/Qwen2.5-Coder-7B-Instruct → 7b
# ------------------------------------------------------------
model_tag () {
  basename=$(echo "$1" | sed 's|.*/||')
  echo "$basename" | sed -E 's/.*-([0-9]+[a-zA-Z]*).*/\1/' | tr '[:upper:]' '[:lower:]'
}

# ------------------------------------------------------------
# Tmux session setup
# ------------------------------------------------------------
echo "[start_evals] Killing existing tmux session '$SESSION' (if any)..."
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

echo "[start_evals] Creating tmux session '$SESSION'..."
tmux new-session -d -s "$SESSION" -n "placeholder"

# ------------------------------------------------------------
# Launch each task
# ------------------------------------------------------------
idx=0

for task_name in "${TASKS[@]}"; do
  declare -n T="$task_name"

  TEST_GEN_TAG=$(model_tag "${T[gen_model]}")
  WINDOW="${idx}-${TEST_GEN_TAG}-${T[dataset]}"

  if [[ $idx -eq 0 ]]; then
    tmux rename-window -t "$SESSION:0" "$WINDOW"
  else
    tmux new-window -a -t "$SESSION" -n "$WINDOW"
  fi

  echo "[start_evals] Launching eval task $idx : $WINDOW"

  tmux send-keys -t "$SESSION:$WINDOW" "
    source \"$CONDA_SH\" && \
    conda activate $ENV_NAME && \
    python evals/ranker_original/eval_ranker_original_cli.py \
      --dataset \"${T[dataset]}\" \
      --base_path \"$BASE_PATH\" \
      --gen_model \"${T[gen_model]}\" \
      --prog_model \"$PROG_MODEL\" \
      --gen_port ${T[gen_port]} \
      --prog_port $PROG_PORT \
      --num_workers ${T[num_workers]} \
      --num_runs ${T[num_runs]} \
      --log_dir \"$LOG_DIR\" \
      --experiment_name \"$NAME\"
  " C-m

  idx=$((idx + 1))
done

echo "[start_evals] All evals launched."
echo "[start_evals] Attaching to tmux session '$SESSION'..."
tmux attach -t "$SESSION"
