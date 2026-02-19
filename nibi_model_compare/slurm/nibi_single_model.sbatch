#!/bin/bash
# Template: one model, one video, one job on Nibi.

#SBATCH --job-name=sam3_nibi_single
#SBATCH --account=rpp-kmoran
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/sam3/logs/%x-%j.out
#SBATCH --error=/home/%u/sam3/logs/%x-%j.err

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
PROJECT_ROOT="${PROJECT_ROOT:-/project/rpp-kmoran/merileo/sam3}"
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-/project/rpp-kmoran/merileo/hf-cache}"
LOG_ROOT="/home/$USER/sam3/logs"
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

echo "=== Cluster sanity ==="
sinfo -o "%P %D %G %m %l" | head -n 30 || true

echo "=== Modules ==="
module purge
module load StdEnv/2023
module load python/3.11 cuda cudnn
module load scipy-stack
module load opencv/4.12.0
module list

VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found: $VENV_PATH"
  echo "Set VENV_PATH=/path/to/venv before sbatch."
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

VIDEO_PATH="${VIDEO_PATH:-/project/rpp-kmoran/merileo/data/onc/pilot.mp4}"
PROMPT="${PROMPT:-segment all visible marine organisms}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
SERVER_PORT="${SERVER_PORT:-8001}"
TP_SIZE="${TP_SIZE:-4}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/runs/single_${SLURM_JOB_ID}_${RUN_TS}}"
export SERVER_PORT

TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER/slurm_${SLURM_JOB_ID}}"
STAGE_ROOT="${TMP_BASE}/sam3_stage"
MODEL_SAFE="${MODEL_ID//\//_}"
STAGE_OUT="${STAGE_ROOT}/${MODEL_SAFE}"
FINAL_OUT="${OUT_ROOT}/${MODEL_SAFE}"

mkdir -p "$STAGE_OUT" "$FINAL_OUT" "$OUT_ROOT"

echo "=== GPU diagnostics ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "No GPUs visible."
  exit 1
fi
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((GPU_COUNT - 1)))}"
echo "Using GPU IDs: $GPU_IDS"

echo "=== Launching vLLM ==="
vllm serve "$MODEL_ID" \
  --tensor-parallel-size "$TP_SIZE" \
  --trust-remote-code \
  --allowed-local-media-path / \
  --gpu-memory-utilization 0.90 \
  --port "$SERVER_PORT" &
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

python3 nibi_model_compare/run_video_agent_openai.py \
  --video_path "$VIDEO_PATH" \
  --prompt "$PROMPT" \
  --server_url "http://127.0.0.1:${SERVER_PORT}/v1" \
  --model "$MODEL_ID" \
  --output_dir "$STAGE_OUT" \
  --gpus "$GPU_IDS"

cp -a "$STAGE_OUT/." "$FINAL_OUT/"

python3 nibi_model_compare/summarize_runs.py \
  --output_root "$OUT_ROOT"

python3 nibi_model_compare/generate_paste_report.py \
  --summary_csv "$OUT_ROOT/summary.csv" \
  --output_path "$OUT_ROOT/PASTE_TO_CODEX.md"

if command -v rsync >/dev/null 2>&1; then
  mkdir -p "$PERSISTENT_CACHE_ROOT/hub"
  rsync -a --ignore-existing "$HF_HUB_CACHE/" "$PERSISTENT_CACHE_ROOT/hub/" || true
fi

cat > "$OUT_ROOT/README.md" <<EOF
# SAM3 Nibi Single-Run Output

- Job ID: ${SLURM_JOB_ID}
- Node list: ${SLURM_JOB_NODELIST}
- Model: ${MODEL_ID}
- Video: ${VIDEO_PATH}
- Prompt: ${PROMPT}
- Output root: ${OUT_ROOT}
- Model output folder: ${FINAL_OUT}
- Temporary stage folder: ${STAGE_OUT}

Files to review first:
- ${OUT_ROOT}/summary.md
- ${OUT_ROOT}/PASTE_TO_CODEX.md
- ${FINAL_OUT}/run_metrics.json
- ${FINAL_OUT}/manifest.json
EOF

if [[ -d "$STAGE_OUT" ]]; then
  safe_delete_dir "$STAGE_OUT"
fi

echo "Done. Paste report file: $OUT_ROOT/PASTE_TO_CODEX.md"
