#!/bin/bash
# Experimental template for Qwen3.5 local serving on Nibi.
# Use only on 8xH100 nodes; model is very large and may still OOM.

#SBATCH --job-name=sam3_qwen35_tp8
#SBATCH --account=rpp-kmoran
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256000M
#SBATCH --time=10:00:00
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

module purge
module load StdEnv/2023
module load python/3.11 cuda cudnn
module load scipy-stack
module load opencv/4.12.0

VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
if [[ ! -d "$VENV_PATH" ]]; then
  echo "Virtual environment not found: $VENV_PATH"
  exit 1
fi
source "$VENV_PATH/bin/activate"

export HF_HOME="${HF_HOME:-$PROJECT_CACHE_ROOT/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT_CACHE_ROOT/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PROJECT_CACHE_ROOT/transformers}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_CACHE_ROOT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_CACHE_ROOT/xdg}"
export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-$PROJECT_CACHE_ROOT/tmp}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

VIDEO_PATH="${VIDEO_PATH:-/project/rpp-kmoran/merileo/data/onc/pilot.mp4}"
PROMPT="${PROMPT:-segment all visible marine organisms}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-397B-A17B}"
SERVER_PORT="${SERVER_PORT:-8005}"
export SERVER_PORT
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/runs/qwen35_tp8_${SLURM_JOB_ID}_${RUN_TS}}"

TMP_BASE="${SLURM_TMPDIR:-/tmp/$USER/slurm_${SLURM_JOB_ID}}"
STAGE_ROOT="${TMP_BASE}/sam3_stage"
MODEL_SAFE="${MODEL_ID//\//_}"
STAGE_OUT="${STAGE_ROOT}/${MODEL_SAFE}"
FINAL_OUT="${OUT_ROOT}/${MODEL_SAFE}"
mkdir -p "$STAGE_OUT" "$FINAL_OUT" "$OUT_ROOT"

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((GPU_COUNT - 1)))}"
echo "GPUs detected: $GPU_COUNT ; using $GPU_IDS"

vllm serve "$MODEL_ID" \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --limit-mm-per-prompt image=4,video=2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
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

port = os.environ.get("SERVER_PORT", "8005")
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

python3 nibi_model_compare/summarize_runs.py --output_root "$OUT_ROOT"
python3 nibi_model_compare/generate_paste_report.py \
  --summary_csv "$OUT_ROOT/summary.csv" \
  --output_path "$OUT_ROOT/PASTE_TO_CODEX.md"

if [[ -d "$STAGE_OUT" ]]; then
  safe_delete_dir "$STAGE_OUT"
fi

echo "Done. Report: $OUT_ROOT/PASTE_TO_CODEX.md"
