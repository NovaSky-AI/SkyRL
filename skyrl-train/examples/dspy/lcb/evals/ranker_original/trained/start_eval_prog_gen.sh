#!/usr/bin/env bash
set -e

SESSION_BASE="evals-trained"
# NAME="prog-gen-3b-100steps"
NAME="prog-gen-1point5b-100steps"

SESSION="$SESSION_BASE-$NAME"
# -------------------------
# Model definitions
# -------------------------
TEST_GEN_7B="Qwen/Qwen2.5-Coder-7B-Instruct"
TEST_GEN_15B="Qwen/Qwen2.5-Coder-1.5B-Instruct"
TEST_GEN_3B="Qwen/Qwen2.5-Coder-3B-Instruct"
TEST_GEN_15B_100STEPS="Harryllh/lcb_test_generator_1.5b_100steps"
TEST_GEN_3B_200STEPS="Harryllh/lcb_test_generator_3b_200steps"

PROG_GEN_15B_100STEPS="Harryllh/lcb_prog_generator_1.5b_100steps"
PROG_GEN_3B_100STEPS="Harryllh/lcb_prog_generator_3b_100steps"

TEST_GEN_15B_PORT=7001
TEST_GEN_3B_PORT=7002
TEST_GEN_7B_PORT=7003

PROG_GEN_15B_100STEPS_PORT=7004
PROG_GEN_3B_100STEPS_PORT=7005
TEST_GEN_15B_100STEPS_PORT=7006
# TEST_GEN_3B_200STEPS_PORT=7007


# PROG_MODEL=$PROG_GEN_3B_100STEPS
PROG_MODEL=$PROG_GEN_15B_100STEPS
# PROG_PORT=$PROG_GEN_3B_100STEPS_PORT
PROG_PORT=$PROG_GEN_15B_100STEPS_PORT

# -------------------------
# Paths & env
# -------------------------
# BASE_PATH="/work/jiashu/assertion-data/livecodebench/evals/results/ranker-original/test"
BASE_PATH="/work/jiashu/assertion-data/livecodebench/evals/results/ranker-original/trained/$NAME"

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
declare -A TASK0=(
  [dataset]="train"
  [gen_model]="$TEST_GEN_15B"
  [gen_port]="$TEST_GEN_15B_PORT"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK1=(
  [dataset]="train"
  [gen_model]="$TEST_GEN_3B"
  [gen_port]="$TEST_GEN_3B_PORT"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK2=(
  [dataset]="train"
  [gen_model]="$TEST_GEN_7B"
  [gen_port]="$TEST_GEN_7B_PORT"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK3=(
  [dataset]="train"
  [gen_model]="$TEST_GEN_15B_100STEPS"
  [gen_port]="$TEST_GEN_15B_100STEPS_PORT"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

# declare -A TASK4=(
#   [dataset]="train"
#   [gen_model]="$TEST_GEN_3B_100STEPS"
#   [gen_port]="$TEST_GEN_3B_100STEPS_PORT"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

declare -A TASK5=(
  [dataset]="test"
  [gen_model]="$TEST_GEN_15B"
  [gen_port]="$TEST_GEN_15B_PORT"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK6=(
  [dataset]="test"
  [gen_model]="$TEST_GEN_3B"
  [gen_port]="$TEST_GEN_3B_PORT"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK7=(
  [dataset]="test"
  [gen_model]="$TEST_GEN_7B"
  [gen_port]="$TEST_GEN_7B_PORT"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK8=(
  [dataset]="test"
  [gen_model]="$TEST_GEN_15B_100STEPS"
  [gen_port]="$TEST_GEN_15B_100STEPS_PORT"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

# declare -A TASK9=(
#   [dataset]="test"
#   [gen_model]="$TEST_GEN_3B_200STEPS"
#   [gen_port]="$TEST_GEN_3B_200STEPS_PORT"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# List of tasks
TASKS=(TASK0 TASK1 TASK2 TASK3 TASK5 TASK6 TASK7 TASK8)

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

  # Target the window by index to avoid issues when the generated name
  # contains characters tmux might parse (e.g., dots).
  tmux send-keys -t "$SESSION:$idx" "
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
