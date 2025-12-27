#!/usr/bin/env bash
set -e

SESSION_BASE="evals-trained"
NAME="code-3b-140steps"

SESSION="$SESSION_BASE-$NAME"
# -------------------------
# Generator model (GLOBAL)
# -------------------------
GEN_MODEL="Harryllh/lcb_test_generator_3b_140steps"
# GEN_MODEL="my_lora"
# GEN_MODEL="Harryllh/lcb_test_generator_1.5b_400steps"
# GEN_MODEL="Harryllh/lcb_test_generator_1.5b_230steps"
# GEN_MODEL="Harryllh/lcb_test_generator_3b_400steps"
GEN_PORT=7007

# -------------------------
# Program model
# -------------------------
PROG_MODEL_7B="Qwen/Qwen2.5-Coder-7B-Instruct"
PROG_MODEL_15B="Qwen/Qwen2.5-Coder-1.5B-Instruct"
PROG_MODEL_3B="Qwen/Qwen2.5-Coder-3B-Instruct"
PROG_MODEL_3B_100STEPS="Harryllh/lcb_prog_generator_3b_100steps"
PROG_MODEL_15B_100STEPS="Harryllh/lcb_prog_generator_1.5b_100steps"

PROG_PORT_15B=7001
PROG_PORT_3B=7002
PROG_PORT_7B=7003
PROG_PORT_3B_100STEPS=7005
PROG_PORT_15B_100STEPS=7004

# -------------------------
# Paths & env
# -------------------------
BASE_PATH="/work/jiashu/assertion-data/livecodebench/evals/new_results/ranker-original/trained/$NAME"
LOG_DIR=$BASE_PATH/logs
CONDA_SH="/home/eecs/jiashu.chen/.zshrc"
ENV_NAME="assertion"
NUM_WORKERS="10"
NUM_RUN="3"
# ------------------------------------------------------------
# Eval Tasks (dictionaries)
# ------------------------------------------------------------

# declare -A TASK0=(
#   [dataset]="train"
#   [prog_model]="$PROG_MODEL_15B"
#   [prog_port]="$PROG_PORT_15B"
#   [num_workers]="1"
#   [num_runs]="1"
# )
# TASKS=(TASK0)

# declare -A TASK0=(
#   [dataset]="test"
#   [prog_model]="$PROG_MODEL_15B"
#   [prog_port]="$PROG_PORT_15B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# declare -A TASK1=(
#   [dataset]="train"
#   [prog_model]="$PROG_MODEL_15B"
#   [prog_port]="$PROG_PORT_15B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# declare -A TASK2=(
#   [dataset]="test"
#   [prog_model]="$PROG_MODEL_3B"
#   [prog_port]="$PROG_PORT_3B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# declare -A TASK3=(
#   [dataset]="train"
#   [prog_model]="$PROG_MODEL_3B"
#   [prog_port]="$PROG_PORT_3B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# declare -A TASK4=(
#   [dataset]="test"
#   [prog_model]="$PROG_MODEL_7B"
#   [prog_port]="$PROG_PORT_7B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

# declare -A TASK5=(
#   [dataset]="train"
#   [prog_model]="$PROG_MODEL_7B"
#   [prog_port]="$PROG_PORT_7B"
#   [num_workers]="$NUM_WORKERS"
#   [num_runs]="$NUM_RUN"
# )

declare -A TASK0=(
  [dataset]="train"
  [prog_model]="$PROG_MODEL_3B_100STEPS"
  [prog_port]="$PROG_PORT_3B_100STEPS"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK1=(
  [dataset]="test"
  [prog_model]="$PROG_MODEL_3B_100STEPS"
  [prog_port]="$PROG_PORT_3B_100STEPS"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK2=(
  [dataset]="train"
  [prog_model]="$PROG_MODEL_15B_100STEPS"
  [prog_port]="$PROG_PORT_15B_100STEPS"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK3=(
  [dataset]="test"
  [prog_model]="$PROG_MODEL_15B_100STEPS"
  [prog_port]="$PROG_PORT_15B_100STEPS"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK4=(
  [dataset]="train"
  [prog_model]="$PROG_MODEL_7B"
  [prog_port]="$PROG_PORT_7B"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK5=(
  [dataset]="test"
  [prog_model]="$PROG_MODEL_7B"
  [prog_port]="$PROG_PORT_7B"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK6=(
  [dataset]="train"
  [prog_model]="$PROG_MODEL_3B"
  [prog_port]="$PROG_PORT_3B"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK7=(
  [dataset]="test"
  [prog_model]="$PROG_MODEL_3B"
  [prog_port]="$PROG_PORT_3B"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK8=(
  [dataset]="train"
  [prog_model]="$PROG_MODEL_15B"
  [prog_port]="$PROG_PORT_15B"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

declare -A TASK9=(
  [dataset]="test"
  [prog_model]="$PROG_MODEL_15B"
  [prog_port]="$PROG_PORT_15B"
  [num_workers]="$NUM_WORKERS"
  [num_runs]="$NUM_RUN"
)

TASKS=(TASK0 TASK1 TASK2 TASK3 TASK4 TASK5 TASK6 TASK7 TASK8 TASK9)

# TASKS=(TASK0)
# ------------------------------------------------------------
# SAFE model tag generator (no dots, no uppercase, no special chars)
# ------------------------------------------------------------
model_tag () {
  basename=$(echo "$1" | sed 's|.*/||')  # strip path

  clean=$(echo "$basename" \
          | tr '[:upper:]' '[:lower:]' \
          | sed -E 's/[^a-z0-9_-]+/-/g')  # replace unsafe chars

  echo "$clean"
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

  PROG_TAG=$(model_tag "${T[prog_model]}")
  WINDOW="${idx}-${PROG_TAG}-${T[dataset]}"

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
      --gen_model \"$GEN_MODEL\" \
      --prog_model \"${T[prog_model]}\" \
      --gen_port $GEN_PORT \
      --prog_port ${T[prog_port]} \
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
