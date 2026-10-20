#!/usr/bin/env bash


# cd /mlp_vepfs/share/wzg/project/fwam_rollout
# bash client_rollout_p.sh \
#   --task_set pretrain300 \
#   --parallel_clients 4 \
#   --server_url http://10.2.0.6:7891 \
#   --split pretrain \
#   --num_trials 2 \
#   --seed 12345 \
#   --save_video_every 0 \
#   --replan_steps 24 \
#   --video_dir  /mlp_vepfs/share/wzg/project/fwam_rollout/0728 \
#   --rollout_root /dexmal-aa-wzg-data/fastwam3e_rollout/try  \
#   --run_name fastwam_rollout




set -euo pipefail

echo "Start Robocasa Parallel Eval Client WITH rollout"

CODE_PATH="/mlp_vepfs/share/wzg/project/fwam_rollout"
ROBOCASA_REPO="/workspace/robocasa"
ROBOCASA_WORKDIR="/workspace/robocasa/robocasa"
CLIENT_EXP="eval_robocasa_client_rollout.py"
PYTHON_BIN="/usr/local/bin/python"

TASKS=""
TASK_SET=""
TASK_GROUPS=""
PARALLEL_CLIENTS="1"
SERVER_URL=""
SPLIT="pretrain"
NUM_TRIALS="50"
SEED="42"
SAVE_VIDEO_EVERY="0"
REPLAN_STEPS=""
VIDEO_DIR=""
RUN_NAME=""
LOG_FILE=""
ROLLOUT_ROOT=""
NO_RESUME="0"
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --code_path) CODE_PATH="$2"; shift ;;
    --robocasa_repo) ROBOCASA_REPO="$2"; shift ;;
    --robocasa_workdir) ROBOCASA_WORKDIR="$2"; shift ;;
    --client_exp) CLIENT_EXP="$2"; shift ;;
    --python_bin) PYTHON_BIN="$2"; shift ;;
    --tasks) TASKS="$2"; shift ;;
    --task_set) TASK_SET="$2"; shift ;;
    --task_groups) TASK_GROUPS="$2"; shift ;;
    --parallel_clients) PARALLEL_CLIENTS="$2"; shift ;;
    --server_url) SERVER_URL="$2"; shift ;;
    --split) SPLIT="$2"; shift ;;
    --num_trials) NUM_TRIALS="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    --save_video_every) SAVE_VIDEO_EVERY="$2"; shift ;;
    --replan_steps) REPLAN_STEPS="$2"; shift ;;
    --video_dir) VIDEO_DIR="$2"; shift ;;
    --run_name) RUN_NAME="$2"; shift ;;
    --log_file) LOG_FILE="$2"; shift ;;
    --rollout_root) ROLLOUT_ROOT="$2"; shift ;;
    --no_resume) NO_RESUME="1" ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

[[ -n "$SERVER_URL" ]] || { echo "ERROR: --server_url is required"; exit 1; }
[[ -n "$ROLLOUT_ROOT" ]] || { echo "ERROR: --rollout_root is required"; exit 1; }
[[ "$PARALLEL_CLIENTS" =~ ^[0-9]+$ ]] || { echo "ERROR: --parallel_clients must be integer"; exit 1; }
[[ "$SEED" =~ ^-?[0-9]+$ ]] || { echo "ERROR: --seed must be integer"; exit 1; }
[[ "$PARALLEL_CLIENTS" -ge 1 ]] || { echo "ERROR: --parallel_clients must be >= 1"; exit 1; }

MODE_COUNT=0
[[ -n "$TASKS" ]] && MODE_COUNT=$((MODE_COUNT + 1))
[[ -n "$TASK_SET" ]] && MODE_COUNT=$((MODE_COUNT + 1))
[[ -n "$TASK_GROUPS" ]] && MODE_COUNT=$((MODE_COUNT + 1))
[[ "$MODE_COUNT" -eq 1 ]] || { echo "ERROR: exactly one of --tasks, --task_set, or --task_groups must be specified"; exit 1; }

[[ -n "$RUN_NAME" ]] || RUN_NAME="robocasa_rollout_parallel_eval_$(date +%Y%m%d_%H%M%S)"
[[ -n "$VIDEO_DIR" ]] || VIDEO_DIR="$CODE_PATH/eval_videos"
[[ -n "$LOG_FILE" ]] || LOG_FILE="$CODE_PATH/logs/eval/${RUN_NAME}.log"
CLIENT_SCRIPT="$CODE_PATH/$CLIENT_EXP"

[[ -d "$CODE_PATH" ]] || { echo "ERROR: CODE_PATH does not exist: $CODE_PATH"; exit 1; }
[[ -d "$ROBOCASA_REPO" ]] || { echo "ERROR: ROBOCASA_REPO does not exist: $ROBOCASA_REPO"; exit 1; }
[[ -d "$ROBOCASA_WORKDIR" ]] || { echo "ERROR: ROBOCASA_WORKDIR does not exist: $ROBOCASA_WORKDIR"; exit 1; }
[[ -f "$CLIENT_SCRIPT" ]] || { echo "ERROR: CLIENT_SCRIPT does not exist: $CLIENT_SCRIPT"; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN"; exit 1; }
mkdir -p "$(dirname "$LOG_FILE")" "$VIDEO_DIR" "$ROLLOUT_ROOT"

SERVER_HOST="$($PYTHON_BIN - <<PY
from urllib.parse import urlparse
u = urlparse("$SERVER_URL")
print(u.hostname or "")
PY
)"
export HTTP_PROXY=http://10.1.4.219:3143
export HTTPS_PROXY=http://10.1.4.219:3143
export NO_PROXY="127.0.0.1,localhost,${SERVER_HOST}"
export no_proxy="127.0.0.1,localhost,${SERVER_HOST}"
export PYTHONPATH="$CODE_PATH:$ROBOCASA_REPO:$ROBOCASA_WORKDIR:${PYTHONPATH:-}"

cd "$ROBOCASA_WORKDIR"
mkdir -p models/assets
ASSET_SRC_ROOT="/mlp_vepfs/share/wzg/robocasa_assets"
for name in fixtures objects textures generative_textures; do
  src="$ASSET_SRC_ROOT/$name"
  dst="models/assets/$name"
  [[ -d "$src" ]] || { echo "ERROR: missing asset dir: $src"; exit 1; }
  rm -rf "$dst"
  ln -s "$src" "$dst"
done

EVAL_GROUPS=()
if [[ -n "$TASK_SET" ]]; then
  TASK_ARRAY_STR="$($PYTHON_BIN - <<PY
import sys
sys.path.insert(0, "$CODE_PATH")
from eval_robocasa_client_rollout import TASK_SETS
print(",".join(TASK_SETS["$TASK_SET"]))
PY
)"
  IFS=',' read -r -a TASK_ARRAY <<< "$TASK_ARRAY_STR"
  if [[ "$PARALLEL_CLIENTS" -le 1 ]]; then
    EVAL_GROUPS+=("__TASK_SET__:$TASK_SET")
  else
    declare -a GROUP_TASKS
    for ((i = 0; i < PARALLEL_CLIENTS; i++)); do GROUP_TASKS[$i]=""; done
    for task_idx in "${!TASK_ARRAY[@]}"; do
      task="${TASK_ARRAY[$task_idx]}"
      group_id=$((task_idx % PARALLEL_CLIENTS))
      if [[ -z "${GROUP_TASKS[$group_id]}" ]]; then GROUP_TASKS[$group_id]="$task"; else GROUP_TASKS[$group_id]="${GROUP_TASKS[$group_id]},$task"; fi
    done
    for ((i = 0; i < PARALLEL_CLIENTS; i++)); do [[ -n "${GROUP_TASKS[$i]}" ]] && EVAL_GROUPS+=("__TASKS__:${GROUP_TASKS[$i]}"); done
  fi
elif [[ -n "$TASK_GROUPS" ]]; then
  IFS=';' read -r -a RAW_GROUPS <<< "$TASK_GROUPS"
  for task_group in "${RAW_GROUPS[@]}"; do
    [[ -n "$task_group" ]] && EVAL_GROUPS+=("__TASKS__:$task_group")
  done
else
  IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
  declare -a GROUP_TASKS
  for ((i = 0; i < PARALLEL_CLIENTS; i++)); do GROUP_TASKS[$i]=""; done
  for task_idx in "${!TASK_ARRAY[@]}"; do
    task="${TASK_ARRAY[$task_idx]}"
    group_id=$((task_idx % PARALLEL_CLIENTS))
    if [[ -z "${GROUP_TASKS[$group_id]}" ]]; then GROUP_TASKS[$group_id]="$task"; else GROUP_TASKS[$group_id]="${GROUP_TASKS[$group_id]},$task"; fi
  done
  for ((i = 0; i < PARALLEL_CLIENTS; i++)); do [[ -n "${GROUP_TASKS[$i]}" ]] && EVAL_GROUPS+=("__TASKS__:${GROUP_TASKS[$i]}"); done
fi
[[ "${#EVAL_GROUPS[@]}" -gt 0 ]] || { echo "ERROR: no eval groups generated"; exit 1; }

PIDS=()
cleanup() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Cleaning up child eval processes: ${PIDS[*]}"
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" || true
      fi
    done
  fi
}
trap cleanup INT TERM

for group_idx in "${!EVAL_GROUPS[@]}"; do
  eval_group="${EVAL_GROUPS[$group_idx]}"
  group_log="${LOG_FILE%.log}_group${group_idx}.log"
  group_run_name="${RUN_NAME}_group${group_idx}"
  if [[ "$eval_group" == __TASK_SET__:* ]]; then
    value="${eval_group#__TASK_SET__:}"
    TASK_ARGS=(--task_set "$value")
  else
    value="${eval_group#__TASKS__:}"
    TASK_ARGS=(--tasks "$value")
  fi
  (
    set -o pipefail
    (
      CMD=("$PYTHON_BIN" "$CLIENT_SCRIPT" "${TASK_ARGS[@]}" --server_url "$SERVER_URL" --split "$SPLIT" --num_trials "$NUM_TRIALS" --seed "$SEED" --save_video_every "$SAVE_VIDEO_EVERY" --video_dir "$VIDEO_DIR" --run_name "$group_run_name" --save_rollout --rollout_root "$ROLLOUT_ROOT")
      [[ -n "$REPLAN_STEPS" ]] && CMD+=(--replan_steps "$REPLAN_STEPS")
      [[ "$NO_RESUME" == "1" ]] && CMD+=(--no_resume)
      [[ "${#EXTRA_ARGS[@]}" -gt 0 ]] && CMD+=("${EXTRA_ARGS[@]}")
      printf 'CMD: %q ' "${CMD[@]}"; echo
      "${CMD[@]}"
    ) 2>&1 | sed -u "s/^/[group${group_idx}] /" | tee -a "$group_log"
  ) &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "ERROR: eval process failed, pid=$pid"
    FAILED=1
  fi
done
trap - INT TERM

if [[ "$FAILED" -ne 0 ]]; then
  echo "Some eval groups failed. Please check logs:"
  ls -lh "${LOG_FILE%.log}"_group*.log || true
  exit 1
fi

echo "All eval groups finished successfully."
ls -lh "${LOG_FILE%.log}"_group*.log || true
