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

ensure_runner_deps() {
  local wheelhouse_args=(
    -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4
    -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
    -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
    -f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
  )
  local runner_dep_specs=(
    "timm>=1.0.17"
    "ftfy==6.1.1"
    "scikit-image"
    "scikit-learn"
    "pandas"
    "matplotlib"
  )
  export PIP_NO_USER=1

  if ! python3 - <<'PY'
import importlib.util
import sys
missing = [
    m
    for m in ("timm", "ftfy", "skimage", "sklearn", "pandas", "matplotlib")
    if importlib.util.find_spec(m) is None
]
sys.exit(0 if not missing else 1)
PY
  then
    echo "[Info] Installing missing SAM3 runner deps (timm, ftfy, scikit-image, scikit-learn, pandas, matplotlib) into active venv..."
    if ! python3 -m pip --isolated install --no-index "${wheelhouse_args[@]}" "${runner_dep_specs[@]}"; then
      PIP_CONFIG_FILE=/dev/null python3 -m pip install -i https://pypi.org/simple "${runner_dep_specs[@]}"
    fi
  fi

  if ! python3 - <<'PY'
import numpy as np
if int(np.__version__.split(".")[0]) >= 2:
    raise SystemExit(1)
try:
    import pycocotools._mask  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
  then
    echo "[Info] Repairing numpy/pycocotools ABI compatibility (numpy<2)..."
    if ! python3 -m pip --isolated install --no-index "${wheelhouse_args[@]}" --force-reinstall "numpy>=1.26,<2" "pycocotools"; then
      PIP_CONFIG_FILE=/dev/null python3 -m pip install -i https://pypi.org/simple --force-reinstall "numpy>=1.26,<2" "pycocotools"
    fi
  fi
}

REPO_ROOT="${REPO_ROOT:-$PWD}"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
ACCOUNT_NAME="${SLURM_JOB_ACCOUNT:-${SLURM_ACCOUNT:-${ACCOUNT:-}}}"
if [[ -z "${DEFAULT_PROJECT_PREFIX:-}" ]]; then
  if [[ -n "$ACCOUNT_NAME" ]]; then
    DEFAULT_PROJECT_PREFIX="/project/${ACCOUNT_NAME}/$USER"
  elif [[ -n "${VIDEO_PATH:-}" && "${VIDEO_PATH}" =~ ^/project/([^/]+)/ ]]; then
    DEFAULT_PROJECT_PREFIX="/project/${BASH_REMATCH[1]}/$USER"
  else
    DEFAULT_PROJECT_PREFIX="/project/$USER/$USER"
  fi
fi
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-$DEFAULT_PROJECT_PREFIX/hf-cache}"
LOG_ROOT="${LOG_ROOT:-/home/$USER/sam3/logs}"
mkdir -p "$LOG_ROOT" "$PROJECT_ROOT" "$PROJECT_CACHE_ROOT"
echo "Resolved account='${ACCOUNT_NAME:-<unset>}' DEFAULT_PROJECT_PREFIX='${DEFAULT_PROJECT_PREFIX}'"
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
ensure_runner_deps

PERSISTENT_CACHE_ROOT="${PERSISTENT_CACHE_ROOT:-$PROJECT_CACHE_ROOT}"
JOB_CACHE_ROOT="${JOB_CACHE_ROOT:-${SLURM_TMPDIR:-$PROJECT_CACHE_ROOT}/hf-cache}"
mkdir -p "$PERSISTENT_CACHE_ROOT" "$JOB_CACHE_ROOT"

export HF_HOME="${HF_HOME:-$JOB_CACHE_ROOT/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME/xdg}"
export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-$HF_HOME/tmp}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$JOB_CACHE_ROOT/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$JOB_CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$JOB_CACHE_ROOT/torchinductor}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$JOB_CACHE_ROOT/numba}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR"
echo "Cache dirs: HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT TMPDIR=$TMPDIR"

if command -v rsync >/dev/null 2>&1; then
  mkdir -p "$PERSISTENT_CACHE_ROOT/hub"
  rsync -a --ignore-existing "$PERSISTENT_CACHE_ROOT/hub/" "$HF_HUB_CACHE/" || true
fi

VIDEO_PATH="${VIDEO_PATH:-$REPO_ROOT/assets/videos/chinacreekclipped.mp4}"
PROMPT="${PROMPT:-segment all visible marine organisms}"
PROMPT_PROFILE="${PROMPT_PROFILE:-${SAM3_AGENT_PROMPT_PROFILE:-underwater}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-}"
if [[ -z "$MODEL_REVISION" && "$MODEL_ID" == *:* ]]; then
  MODEL_REVISION="${MODEL_ID#*:}"
  MODEL_ID="${MODEL_ID%%:*}"
  echo "Interpreting MODEL_ID suffix as revision: MODEL_ID='${MODEL_ID}' MODEL_REVISION='${MODEL_REVISION}'"
fi
if [[ "$MODEL_ID" == *:* ]]; then
  echo "Invalid MODEL_ID '${MODEL_ID}'. Use plain repo id, and pass revision as MODEL_REVISION."
  exit 1
fi
if [[ -z "${SAM3_TOOL_CALL_FALLBACK_POLICY:-}" ]]; then
  if [[ "${MODEL_ID,,}" == *"qwen3.5"* && "${PROMPT_PROFILE,,}" == "underwater" ]]; then
    export SAM3_TOOL_CALL_FALLBACK_POLICY="auto"
  fi
fi
VLLM_RUNTIME="${VLLM_RUNTIME:-auto}"
if [[ "$VLLM_RUNTIME" == "auto" ]]; then
  model_id_lc="${MODEL_ID,,}"
  if [[ "$model_id_lc" == *"qwen3.5"* ]]; then
    VLLM_RUNTIME="apptainer"
  else
    VLLM_RUNTIME="venv"
  fi
fi
if [[ "$VLLM_RUNTIME" != "venv" && "$VLLM_RUNTIME" != "apptainer" ]]; then
  echo "Invalid VLLM_RUNTIME '${VLLM_RUNTIME}'. Expected one of: auto, venv, apptainer."
  exit 1
fi
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${SCRATCH:-/scratch/$USER}/vllm-openai-nightly.sif}"
SERVER_PORT="${SERVER_PORT:-8001}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-1024}"
SAVE_PROMPTS="${SAVE_PROMPTS:-0}"
DEBUG="${DEBUG:-0}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-}"
RUNNER_CUDA_VISIBLE_DEVICES="${RUNNER_CUDA_VISIBLE_DEVICES:-}"
RUNNER_GPU_IDS="${RUNNER_GPU_IDS:-}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${SCRATCH:-/scratch/$USER}/sam3/runs/single_${SLURM_JOB_ID}_${RUN_TS}}"
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
if [[ -n "$RUNNER_CUDA_VISIBLE_DEVICES" && -z "$RUNNER_GPU_IDS" ]]; then
  RUNNER_GPU_IDS="0"
fi
RUNNER_GPU_IDS="${RUNNER_GPU_IDS:-$GPU_IDS}"
echo "Using GPU IDs: $GPU_IDS"

echo "=== Launching vLLM ==="
echo "vLLM runtime: $VLLM_RUNTIME"
if [[ "$VLLM_RUNTIME" == "apptainer" ]]; then
  module load apptainer
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH:-/scratch/$USER}/apptainer-cache}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SLURM_TMPDIR:-$JOB_CACHE_ROOT/apptainer-tmp}}"
  export APPTAINER_HOME="${APPTAINER_HOME:-$JOB_CACHE_ROOT/apptainer-home}"
  export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-$JOB_CACHE_ROOT,$APPTAINER_HOME,$APPTAINER_CACHEDIR,$APPTAINER_TMPDIR}"
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${SLURM_TMPDIR}:${SLURM_TMPDIR}"
  fi
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$APPTAINER_HOME"
  if [[ ! -f "$APPTAINER_IMAGE" ]]; then
    echo "Apptainer image missing; pulling: $APPTAINER_IMAGE"
    apptainer pull "$APPTAINER_IMAGE" docker://vllm/vllm-openai:nightly
  fi
  export APPTAINERENV_HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  export APPTAINERENV_HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
  export APPTAINERENV_HF_HOME="$HF_HOME"
  export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
  export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
  export APPTAINERENV_HF_XET_CACHE="$HF_XET_CACHE"
  export APPTAINERENV_HF_ASSETS_CACHE="$HF_ASSETS_CACHE"
  export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
  export APPTAINERENV_TMPDIR="$TMPDIR"
  export APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
  export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
  export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
  export APPTAINERENV_NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR"
  export APPTAINERENV_TRANSFORMERS_CACHE="$HF_HUB_CACHE"
  export APPTAINERENV_HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
  export APPTAINERENV_REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
  export APPTAINERENV_CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
  export APPTAINERENV_SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
  VLLM_CMD=(
    apptainer exec --cleanenv --nv --bind "$APPTAINER_BINDPATH" --home "$APPTAINER_HOME" "$APPTAINER_IMAGE"
    vllm serve "$MODEL_ID"
    --tensor-parallel-size "$TP_SIZE"
    --trust-remote-code
    --allowed-local-media-path /
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --port "$SERVER_PORT"
  )
else
  VLLM_CMD=(
    vllm serve "$MODEL_ID"
    --tensor-parallel-size "$TP_SIZE"
    --trust-remote-code
    --allowed-local-media-path /
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --port "$SERVER_PORT"
  )
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  VLLM_CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$MAX_NUM_SEQS" ]]; then
  VLLM_CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$MODEL_REVISION" ]]; then
  VLLM_CMD+=(--revision "$MODEL_REVISION")
fi
if [[ "${MODEL_ID,,}" == *"qwen3.5"* ]]; then
  # Qwen3.5 behaves better for agentic tool-calling when thinking is disabled.
  VLLM_CMD+=(--reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}')
fi
if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
  VLLM_CMD+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi
if [[ "$VLLM_RUNTIME" == "apptainer" ]]; then
  if [[ -n "$VLLM_CUDA_VISIBLE_DEVICES" ]]; then
    APPTAINERENV_CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" "${VLLM_CMD[@]}" &
  else
    "${VLLM_CMD[@]}" &
  fi
else
  if [[ -n "$VLLM_CUDA_VISIBLE_DEVICES" ]]; then
    CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" "${VLLM_CMD[@]}" &
  else
    "${VLLM_CMD[@]}" &
  fi
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

TEMPORAL_KEYFRAME_PIPELINE="${TEMPORAL_KEYFRAME_PIPELINE:-0}"
DISCOVERY_MODE="${DISCOVERY_MODE:-hybrid}"
MAX_KEYFRAMES="${MAX_KEYFRAMES:-6}"
MIN_KEYFRAME_GAP="${MIN_KEYFRAME_GAP:-24}"
DROP_INVALID_FRAMES="${DROP_INVALID_FRAMES:-0}"
INVALID_FRAME_SOURCE="${INVALID_FRAME_SOURCE:-hybrid}"
MLLM_INVALID_WINDOW_SIZE="${MLLM_INVALID_WINDOW_SIZE:-4}"
MLLM_INVALID_WINDOW_STRIDE="${MLLM_INVALID_WINDOW_STRIDE:-4}"
MLLM_INVALID_MAX_JSON_RETRIES="${MLLM_INVALID_MAX_JSON_RETRIES:-2}"
MLLM_DISCOVERY_WINDOW_SIZE="${MLLM_DISCOVERY_WINDOW_SIZE:-4}"
MLLM_DISCOVERY_WINDOW_STRIDE="${MLLM_DISCOVERY_WINDOW_STRIDE:-24}"
MLLM_DISCOVERY_MAX_JSON_RETRIES="${MLLM_DISCOVERY_MAX_JSON_RETRIES:-2}"
POSTPROP_MISSED_CREATURES="${POSTPROP_MISSED_CREATURES:-0}"
POSTPROP_MISSED_WINDOW_SIZE="${POSTPROP_MISSED_WINDOW_SIZE:-20}"
POSTPROP_MISSED_WINDOW_STRIDE="${POSTPROP_MISSED_WINDOW_STRIDE:-10}"
POSTPROP_MISSED_MAX_ISSUES_PER_WINDOW="${POSTPROP_MISSED_MAX_ISSUES_PER_WINDOW:-4}"
POSTPROP_MISSED_MAX_ROUNDS="${POSTPROP_MISSED_MAX_ROUNDS:-10}"
POSTPROP_MISSED_MAX_ATTEMPTS_PER_ISSUE="${POSTPROP_MISSED_MAX_ATTEMPTS_PER_ISSUE:-10}"
POSTPROP_MISSED_MAX_IMAGES_PER_REQUEST="${POSTPROP_MISSED_MAX_IMAGES_PER_REQUEST:-20}"
POSTPROP_MISSED_MAX_JSON_RETRIES="${POSTPROP_MISSED_MAX_JSON_RETRIES:-2}"
POSTPROP_MISSED_PROMPT_PATH="${POSTPROP_MISSED_PROMPT_PATH:-}"
POSTPROP_MISSED_VERIFY_PROMPT_PATH="${POSTPROP_MISSED_VERIFY_PROMPT_PATH:-}"
POSTPROP_QA_MLLM="${POSTPROP_QA_MLLM:-0}"
POSTPROP_QA_WINDOW_SIZE="${POSTPROP_QA_WINDOW_SIZE:-10}"
POSTPROP_QA_WINDOW_STRIDE="${POSTPROP_QA_WINDOW_STRIDE:-1}"
POSTPROP_QA_MAX_COMPLETION_TOKENS="${POSTPROP_QA_MAX_COMPLETION_TOKENS:-512}"
POSTPROP_QA_MAX_OBJECT_CROPS="${POSTPROP_QA_MAX_OBJECT_CROPS:-8}"
POSTPROP_QA_CROP_CONTEXT_RATIO="${POSTPROP_QA_CROP_CONTEXT_RATIO:-0.25}"
POSTPROP_QA_MAX_JSON_RETRIES="${POSTPROP_QA_MAX_JSON_RETRIES:-2}"
POSTPROP_QA_MERGE_BAD_INTO_INVALID="${POSTPROP_QA_MERGE_BAD_INTO_INVALID:-1}"
POSTPROP_REPAIR="${POSTPROP_REPAIR:-0}"
POSTPROP_REPAIR_WINDOW="${POSTPROP_REPAIR_WINDOW:-8}"
POSTPROP_REPAIR_MAX_ATTEMPTS="${POSTPROP_REPAIR_MAX_ATTEMPTS:-2}"
POSTPROP_REPAIR_MAX_CANDIDATES="${POSTPROP_REPAIR_MAX_CANDIDATES:-5}"
POSTPROP_REPAIR_PROMPT_PATH="${POSTPROP_REPAIR_PROMPT_PATH:-}"
POSTPROP_REPAIR_VERIFY="${POSTPROP_REPAIR_VERIFY:-1}"

RUN_CMD=(
  python3 nibi_model_compare/run_video_agent_openai.py
  --video_path "$VIDEO_PATH"
  --prompt "$PROMPT"
  --prompt-profile "$PROMPT_PROFILE"
  --server_url "http://127.0.0.1:${SERVER_PORT}/v1"
  --model "$MODEL_ID"
  --output_dir "$STAGE_OUT"
  --gpus "$RUNNER_GPU_IDS"
  --image_size "$IMAGE_SIZE"
  --max_completion_tokens "$MAX_COMPLETION_TOKENS"
)
if [[ "$TEMPORAL_KEYFRAME_PIPELINE" == "1" ]]; then
  RUN_CMD+=(--temporal_keyframe_pipeline)
  RUN_CMD+=(--discovery_mode "$DISCOVERY_MODE")
  RUN_CMD+=(--max_keyframes "$MAX_KEYFRAMES")
  RUN_CMD+=(--min_keyframe_gap "$MIN_KEYFRAME_GAP")
  RUN_CMD+=(--mllm_discovery_window_size "$MLLM_DISCOVERY_WINDOW_SIZE")
  RUN_CMD+=(--mllm_discovery_window_stride "$MLLM_DISCOVERY_WINDOW_STRIDE")
  RUN_CMD+=(--mllm_discovery_max_json_retries "$MLLM_DISCOVERY_MAX_JSON_RETRIES")
fi
if [[ "$DROP_INVALID_FRAMES" == "1" ]]; then
  RUN_CMD+=(--drop_invalid_frames)
fi
if [[ "$TEMPORAL_KEYFRAME_PIPELINE" == "1" || "$DROP_INVALID_FRAMES" == "1" ]]; then
  RUN_CMD+=(--invalid_frame_source "$INVALID_FRAME_SOURCE")
  RUN_CMD+=(--mllm_invalid_window_size "$MLLM_INVALID_WINDOW_SIZE")
  RUN_CMD+=(--mllm_invalid_window_stride "$MLLM_INVALID_WINDOW_STRIDE")
  RUN_CMD+=(--mllm_invalid_max_json_retries "$MLLM_INVALID_MAX_JSON_RETRIES")
fi
if [[ "$POSTPROP_MISSED_CREATURES" == "1" ]]; then
  RUN_CMD+=(--postprop_missed_creatures)
  RUN_CMD+=(--postprop_missed_window_size "$POSTPROP_MISSED_WINDOW_SIZE")
  RUN_CMD+=(--postprop_missed_window_stride "$POSTPROP_MISSED_WINDOW_STRIDE")
  RUN_CMD+=(--postprop_missed_max_issues_per_window "$POSTPROP_MISSED_MAX_ISSUES_PER_WINDOW")
  RUN_CMD+=(--postprop_missed_max_rounds "$POSTPROP_MISSED_MAX_ROUNDS")
  RUN_CMD+=(--postprop_missed_max_attempts_per_issue "$POSTPROP_MISSED_MAX_ATTEMPTS_PER_ISSUE")
  RUN_CMD+=(--postprop_missed_max_images_per_request "$POSTPROP_MISSED_MAX_IMAGES_PER_REQUEST")
  RUN_CMD+=(--postprop_missed_max_json_retries "$POSTPROP_MISSED_MAX_JSON_RETRIES")
  if [[ -n "$POSTPROP_MISSED_PROMPT_PATH" ]]; then
    RUN_CMD+=(--postprop_missed_prompt_path "$POSTPROP_MISSED_PROMPT_PATH")
  fi
  if [[ -n "$POSTPROP_MISSED_VERIFY_PROMPT_PATH" ]]; then
    RUN_CMD+=(--postprop_missed_verify_prompt_path "$POSTPROP_MISSED_VERIFY_PROMPT_PATH")
  fi
fi
if [[ "$POSTPROP_QA_MLLM" == "1" ]]; then
  RUN_CMD+=(--postprop_qa_mllm)
  RUN_CMD+=(--postprop_qa_window_size "$POSTPROP_QA_WINDOW_SIZE")
  RUN_CMD+=(--postprop_qa_window_stride "$POSTPROP_QA_WINDOW_STRIDE")
  RUN_CMD+=(--postprop_qa_max_completion_tokens "$POSTPROP_QA_MAX_COMPLETION_TOKENS")
  RUN_CMD+=(--postprop_qa_max_object_crops "$POSTPROP_QA_MAX_OBJECT_CROPS")
  RUN_CMD+=(--postprop_qa_crop_context_ratio "$POSTPROP_QA_CROP_CONTEXT_RATIO")
  RUN_CMD+=(--postprop_qa_max_json_retries "$POSTPROP_QA_MAX_JSON_RETRIES")
  if [[ "$POSTPROP_QA_MERGE_BAD_INTO_INVALID" != "1" ]]; then
    RUN_CMD+=(--no_postprop_qa_merge_bad_into_invalid)
  fi
fi
if [[ "$POSTPROP_REPAIR" == "1" ]]; then
  RUN_CMD+=(--postprop_repair)
  RUN_CMD+=(--postprop_repair_window "$POSTPROP_REPAIR_WINDOW")
  RUN_CMD+=(--postprop_repair_max_attempts "$POSTPROP_REPAIR_MAX_ATTEMPTS")
  RUN_CMD+=(--postprop_repair_max_candidates "$POSTPROP_REPAIR_MAX_CANDIDATES")
  if [[ -n "$POSTPROP_REPAIR_PROMPT_PATH" ]]; then
    RUN_CMD+=(--postprop_repair_prompt_path "$POSTPROP_REPAIR_PROMPT_PATH")
  fi
  if [[ "$POSTPROP_REPAIR_VERIFY" != "1" ]]; then
    RUN_CMD+=(--no_postprop_repair_verify)
  fi
fi
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

# Normalize metrics paths to persistent output location (FINAL_OUT) so reports
# do not keep references to SLURM_TMPDIR staging paths.
if [[ -f "$FINAL_OUT/run_metrics.json" ]]; then
  python3 - <<PY
import json
import os

run_metrics_path = os.path.join("$FINAL_OUT", "run_metrics.json")
stage_out = os.path.abspath("$STAGE_OUT")
final_out = os.path.abspath("$FINAL_OUT")

with open(run_metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

changed = False
for key in (
    "output_video_path",
    "output_dir",
    "frame_outputs_json_path",
    "generated_prompts_path",
    "postprop_missed_creatures_report_path",
    "postprop_qa_report_path",
    "postprop_repair_report_path",
    "postprop_repair_visual_debug_dir",
    "postprop_repair_visual_debug_manifest_path",
    "frame_keep_drop_path",
    "codex_debug_report_path",
):
    value = metrics.get(key)
    if isinstance(value, str) and value:
        abs_value = os.path.abspath(value)
        if abs_value.startswith(stage_out):
            new_value = final_out + abs_value[len(stage_out):]
            metrics[key] = new_value
            changed = True

# Also normalize per-keyframe frame paths when present.
for row_key in ("agent_runs", "failed_agent_keyframes"):
    rows = metrics.get(row_key)
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            frame_path = row.get("frame_path")
            if isinstance(frame_path, str) and frame_path:
                abs_frame = os.path.abspath(frame_path)
                if abs_frame.startswith(stage_out):
                    row["frame_path"] = final_out + abs_frame[len(stage_out):]
                    changed = True

if changed:
    with open(run_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\\n")

debug_report_path = os.path.join(final_out, "codex_debug_report.json")
if os.path.exists(debug_report_path):
    def _replace_stage_paths(value):
        if isinstance(value, str):
            abs_value = os.path.abspath(value)
            if abs_value.startswith(stage_out):
                return final_out + abs_value[len(stage_out):]
            return value
        if isinstance(value, list):
            return [_replace_stage_paths(v) for v in value]
        if isinstance(value, dict):
            return {k: _replace_stage_paths(v) for k, v in value.items()}
        return value

    with open(debug_report_path, "r", encoding="utf-8") as f:
        debug_payload = json.load(f)
    debug_payload = _replace_stage_paths(debug_payload)
    with open(debug_report_path, "w", encoding="utf-8") as f:
        json.dump(debug_payload, f, indent=2)
        f.write("\\n")
PY
fi

if [[ -f "$FINAL_OUT/output_video.mp4" ]]; then
  echo "Final overlay video (persistent): $FINAL_OUT/output_video.mp4"
else
  echo "Final overlay video not found in persistent output: $FINAL_OUT/output_video.mp4"
fi

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
