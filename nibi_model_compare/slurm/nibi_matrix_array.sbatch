#!/bin/bash
# Template: one model across many videos using Slurm array.

#SBATCH --job-name=sam3_nibi_array
#SBATCH --account=rpp-kmoran
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=10:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/%u/sam3/logs/%x-%A_%a.out
#SBATCH --error=/home/%u/sam3/logs/%x-%A_%a.err

set -euo pipefail

safe_delete_dir() {
  local target="$1"
  if [[ -z "$target" || "$target" == "/" ]]; then
    echo "Refusing to delete unsafe path: '$target'"
    return 1
  fi
  if [[ "$target" != "${SLURM_TMPDIR:-/tmp/$USER}"* && "$target" != "/tmp/$USER/"* && "$target" != "/scratch/$USER/"* ]]; then
    echo "Refusing to delete outside temporary areas: $target"
    return 1
  fi
  local parent
  parent="$(dirname "$target")"
  local to_delete_dir="${parent}/toDelete"
  local staged="${to_delete_dir}/$(basename "$target")_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$to_delete_dir"
  mv "$target" "$staged"
  echo "Verifying staged delete content: $staged"
  find "$staged" -maxdepth 2 -mindepth 1 | head -n 40 || true
  rm -rf -- "$staged"
}

REPO_ROOT="${REPO_ROOT:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
DEFAULT_PROJECT_PREFIX="${DEFAULT_PROJECT_PREFIX:-/project/${SLURM_ACCOUNT:-${ACCOUNT:-$USER}}/$USER}"
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-$DEFAULT_PROJECT_PREFIX/hf-cache}"
LOG_ROOT="${LOG_ROOT:-/home/$USER/sam3/logs}"
mkdir -p "$LOG_ROOT" "$PROJECT_ROOT" "$PROJECT_CACHE_ROOT"
cd "$REPO_ROOT"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

module purge
module load StdEnv/2023
module load python/3.11 cuda cudnn
module load scipy-stack
module load opencv/4.12.0
module list

VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found: $VENV_PATH"
  exit 1
fi
source "$VENV_PATH/bin/activate"

PERSISTENT_CACHE_ROOT="${PERSISTENT_CACHE_ROOT:-$PROJECT_CACHE_ROOT}"
JOB_CACHE_ROOT="${JOB_CACHE_ROOT:-${SLURM_TMPDIR:-$PROJECT_CACHE_ROOT}/hf-cache}"
mkdir -p "$PERSISTENT_CACHE_ROOT" "$JOB_CACHE_ROOT"

export HF_HOME="${HF_HOME:-$JOB_CACHE_ROOT/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME/xdg}"
export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-$HF_HOME/tmp}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

if command -v rsync >/dev/null 2>&1; then
  mkdir -p "$PERSISTENT_CACHE_ROOT/hub"
  rsync -a --ignore-existing "$PERSISTENT_CACHE_ROOT/hub/" "$HF_HUB_CACHE/" || true
fi

VIDEOS_MANIFEST="${VIDEOS_MANIFEST:-nibi_model_compare/videos_manifest.txt}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
SERVER_PORT="${SERVER_PORT:-8001}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
PROMPT="${PROMPT:-segment all visible marine organisms}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-1024}"
SAVE_PROMPTS="${SAVE_PROMPTS:-0}"
DEBUG="${DEBUG:-0}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-}"
RUNNER_CUDA_VISIBLE_DEVICES="${RUNNER_CUDA_VISIBLE_DEVICES:-}"
RUNNER_GPU_IDS="${RUNNER_GPU_IDS:-}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/runs/array_${SLURM_ARRAY_JOB_ID}_${RUN_TS}}"
export SERVER_PORT

VIDEO_PATH="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$VIDEOS_MANIFEST")"
if [[ -z "$VIDEO_PATH" ]]; then
  echo "No video at array index $SLURM_ARRAY_TASK_ID in $VIDEOS_MANIFEST"
  exit 1
fi

TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER/slurm_${SLURM_JOB_ID}}"
MODEL_SAFE="${MODEL_ID//\//_}"
STAGE_ROOT="${TMP_BASE}/sam3_stage/task_${SLURM_ARRAY_TASK_ID}"
STAGE_OUT="${STAGE_ROOT}/${MODEL_SAFE}"
FINAL_TASK_DIR="${OUT_ROOT}/task_${SLURM_ARRAY_TASK_ID}"
FINAL_OUT="${FINAL_TASK_DIR}/${MODEL_SAFE}"
mkdir -p "$STAGE_OUT" "$FINAL_OUT" "$FINAL_TASK_DIR"

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "No GPUs visible."
  exit 1
fi
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((GPU_COUNT - 1)))}"
if [[ -n "$RUNNER_CUDA_VISIBLE_DEVICES" && -z "$RUNNER_GPU_IDS" ]]; then
  RUNNER_GPU_IDS="0"
fi
RUNNER_GPU_IDS="${RUNNER_GPU_IDS:-$GPU_IDS}"

VLLM_CMD=(
  vllm serve "$MODEL_ID"
  --tensor-parallel-size "$TP_SIZE"
  --trust-remote-code
  --allowed-local-media-path /
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --port "$SERVER_PORT"
)
if [[ -n "$MAX_MODEL_LEN" ]]; then
  VLLM_CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$MAX_NUM_SEQS" ]]; then
  VLLM_CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
  VLLM_CMD+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi
if [[ -n "$VLLM_CUDA_VISIBLE_DEVICES" ]]; then
  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" "${VLLM_CMD[@]}" &
else
  "${VLLM_CMD[@]}" &
fi
VLLM_PID=$!

cleanup() {
  kill "$VLLM_PID" || true
}
trap cleanup EXIT

python3 - <<'PY'
import os
import time
import urllib.request

port = os.environ.get("SERVER_PORT", "8001")
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 900
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print("vLLM is ready.")
                break
    except Exception:
        pass
    time.sleep(2)
else:
    raise RuntimeError(f"Timed out waiting for {url}")
PY

RUN_CMD=(
  python3 nibi_model_compare/run_video_agent_openai.py
  --video_path "$VIDEO_PATH"
  --prompt "$PROMPT"
  --server_url "http://127.0.0.1:${SERVER_PORT}/v1"
  --model "$MODEL_ID"
  --output_dir "$STAGE_OUT"
  --gpus "$RUNNER_GPU_IDS"
  --image_size "$IMAGE_SIZE"
  --max_completion_tokens "$MAX_COMPLETION_TOKENS"
)
if [[ "$SAVE_PROMPTS" == "1" ]]; then
  RUN_CMD+=(--save_prompts)
fi
if [[ "$DEBUG" == "1" ]]; then
  RUN_CMD+=(--debug)
fi
if [[ -n "$RUNNER_CUDA_VISIBLE_DEVICES" ]]; then
  CUDA_VISIBLE_DEVICES="$RUNNER_CUDA_VISIBLE_DEVICES" "${RUN_CMD[@]}"
else
  "${RUN_CMD[@]}"
fi

cp -a "$STAGE_OUT/." "$FINAL_OUT/"

cat > "$FINAL_TASK_DIR/README.md" <<EOF
# SAM3 Nibi Array Task Output

- Array job ID: ${SLURM_ARRAY_JOB_ID}
- Task ID: ${SLURM_ARRAY_TASK_ID}
- Job ID: ${SLURM_JOB_ID}
- Model: ${MODEL_ID}
- Video: ${VIDEO_PATH}
- Prompt: ${PROMPT}
- Final output folder: ${FINAL_OUT}
- Temporary stage folder: ${STAGE_OUT}

Key files:
- ${FINAL_OUT}/run_metrics.json
- ${FINAL_OUT}/manifest.json
EOF

if command -v rsync >/dev/null 2>&1; then
  mkdir -p "$PERSISTENT_CACHE_ROOT/hub"
  rsync -a --ignore-existing "$HF_HUB_CACHE/" "$PERSISTENT_CACHE_ROOT/hub/" || true
fi

if [[ -d "$STAGE_OUT" ]]; then
  safe_delete_dir "$STAGE_OUT"
fi

echo "Task done: ${SLURM_ARRAY_TASK_ID} -> ${VIDEO_PATH}"
